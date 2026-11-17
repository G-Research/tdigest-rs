"""Exercise a real PyPI 0.1.0 -> candidate upgrade, uninstall and reinstall on Linux."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import urllib.request

from check_upstream_tests import ORIGINAL_TEST_CASES
from test_legacy_wheel import stage_compatibility_tests, verify_upstream_snapshot
from test_python_support import COMPATIBILITY_CASES, check_results, inspect_wheel

BASELINE_FILE = "tdigest_rs-0.1.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
BASELINE_SHA256 = "62ee8d5a01226d020a2ab11a956292c38beab3f34d2d9fa7a8fca22c3e5d4bbf"

STATES = r"""
import base64, json, pathlib, pickle, struct, sys
import numpy as np
from tdigest_rs import TDigest
from importlib.metadata import version

def snapshot(d):
    return {'means_dtype': str(d.means.dtype), 'weights_dtype': str(d.weights.dtype),
            'means': base64.b64encode(d.means.tobytes()).decode(),
            'weights': base64.b64encode(d.weights.tobytes()).decode(),
            'queries': [struct.pack('<d', float(d.quantile(q))).hex() for q in (0., .01, .5, .99, 1.)]}

path = pathlib.Path(sys.argv[2])
if sys.argv[1] == 'record':
    assert version('tdigest-rs') == '0.1.0'
    rows=[]
    for dtype in (np.float32, np.float64):
        try:
            TDigest.from_array(np.array([], dtype=dtype))
        except ValueError:
            pass
        else:
            raise AssertionError('published legacy must reject empty construction')
        for n in (1, 2, 32, 1000):
            for seed in (31, 4242):
                for delta in (30., 300.):
                    values=np.random.default_rng(seed).normal(size=n).astype(dtype)
                    original=TDigest.from_array(values, delta=delta)
                    payload=pickle.dumps(original, protocol=5)
                    rows.append({'dtype': str(values.dtype), 'n': n, 'seed': seed, 'delta': delta,
                                 'pickle': base64.b64encode(payload).decode(),
                                 'saved': snapshot(original), 'restored': snapshot(pickle.loads(payload))})
    assert len(rows) == 32
    path.write_text(json.dumps(rows, indent=2)+'\n')
else:
    assert version('tdigest-rs') == sys.argv[3]
    import tdigest_rs
    assert tdigest_rs.__version__ == version('tdigest-rs')
    rows=json.loads(path.read_text())
    assert len(rows)==32
    for row in rows:
        saved = row['saved']
        means = np.frombuffer(base64.b64decode(saved['means']), dtype=saved['means_dtype'])
        weights = np.frombuffer(base64.b64decode(saved['weights']), dtype=saved['weights_dtype'])
        expected = TDigest.from_means_weights(means, weights)
        actual = pickle.loads(base64.b64decode(row['pickle']))
        assert snapshot(actual) == snapshot(expected) == row['restored'], row
        assert actual.weights.sum() == weights.sum(), row
print(json.dumps({'version': version('tdigest-rs'), 'restored_states': len(rows), 'status': 'PASS'}))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("this gate uses the published Linux x86_64 artifact")
    root = Path(__file__).resolve().parents[1]
    verify_upstream_snapshot(root)
    wheel = args.wheel.resolve(strict=True)
    candidate = inspect_wheel(wheel)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = args.baseline_wheel.resolve(strict=True) if args.baseline_wheel else output / BASELINE_FILE
    if not args.baseline_wheel:
        with urllib.request.urlopen("https://pypi.org/pypi/tdigest-rs/0.1.0/json") as response:
            release = json.load(response)
        artifact = next(item for item in release["urls"] if item["filename"] == BASELINE_FILE)
        if artifact["digests"]["sha256"] != BASELINE_SHA256:
            raise RuntimeError("published baseline hash differs from the reviewed artifact")
        with urllib.request.urlopen(artifact["url"]) as response:
            baseline.write_bytes(response.read())
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise RuntimeError("baseline wheel is not the pinned published 0.1.0 artifact")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTEST_ADDOPTS", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    journal = []

    def run(label, command):
        log = output / f"{label}.log"
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=output, env=env, stdout=stream, stderr=subprocess.STDOUT)
        journal.append({"step": label, "exit_code": completed.returncode, "command": command})
        (output / "commands.json").write_text(json.dumps(journal, indent=2) + "\n")
        completed.check_returncode()
        print(f"{label}: PASS", flush=True)

    venv = output / "venv"
    run("create-environment", ["uv", "venv", "--python", "3.12", "--seed", str(venv)])
    python = str(venv / "bin/python")
    run(
        "install-published",
        [python, "-I", "-m", "pip", "install", "numpy==1.26.4", "scipy==1.15.3", "pytest==8.4.2", str(baseline)],
    )
    tests, parity = stage_compatibility_tests(root, output)
    config = output / "pytest.ini"
    config.write_text("[pytest]\n")

    def suite(label, directories, count):
        xml = output / f"{label}.xml"
        run(
            label,
            [
                python,
                "-I",
                "-m",
                "pytest",
                "-q",
                "-c",
                str(config),
                "--rootdir",
                str(output),
                "-p",
                "no:cacheprovider",
                "-o",
                "xfail_strict=true",
                "--junitxml",
                str(xml),
                *(str(path) for path in directories),
            ],
        )
        check_results(xml, count)

    suite("published-original-suite", [tests], ORIGINAL_TEST_CASES)
    states = output / "published-pickles.json"
    run("record-published-states", [python, "-I", "-c", STATES, "record", str(states)])
    run("upgrade", [python, "-I", "-m", "pip", "install", "--upgrade", "--no-index", str(wheel)])
    run("upgraded-pip-check", [python, "-I", "-m", "pip", "check"])
    run("upgraded-states", [python, "-I", "-c", STATES, "verify", str(states), candidate["version"]])
    suite("upgraded-original-suite", [tests], ORIGINAL_TEST_CASES)
    suite("upgraded-compatibility", [parity], COMPATIBILITY_CASES - ORIGINAL_TEST_CASES)
    run("uninstall", [python, "-I", "-m", "pip", "uninstall", "-y", "tdigest-rs"])
    run(
        "uninstalled-imports",
        [
            python,
            "-I",
            "-c",
            "import importlib.util as u; "
            "assert u.find_spec('tdigest_rs') is None; assert u.find_spec('gr_tdigest') is None",
        ],
    )
    run("reinstall", [python, "-I", "-m", "pip", "install", "--no-index", str(wheel)])
    run("reinstalled-pip-check", [python, "-I", "-m", "pip", "check"])
    run("reinstalled-states", [python, "-I", "-c", STATES, "verify", str(states), candidate["version"]])
    suite("reinstalled-original-suite", [tests], ORIGINAL_TEST_CASES)
    suite("reinstalled-compatibility", [parity], COMPATIBILITY_CASES - ORIGINAL_TEST_CASES)
    report = {
        "status": "PASS",
        "baseline": {"filename": baseline.name, "sha256": BASELINE_SHA256},
        "candidate": candidate,
        "published_original_tests": ORIGINAL_TEST_CASES,
        "upgraded_original_tests": ORIGINAL_TEST_CASES,
        "reinstalled_original_tests": ORIGINAL_TEST_CASES,
        "additional_compatibility_tests_per_candidate_install": COMPATIBILITY_CASES - ORIGINAL_TEST_CASES,
        "published_pickle_states_migrated": 32,
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
