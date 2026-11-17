"""Build one sdist wheel and check CPython3.10-3.15, Polars1/2 and Polars-free baselines.

Linux gate. --wheel reuses a real release artifact without rebuilding it.
The matrix intentionally pins tested dependency combinations; the project lock
continues to govern the development environment, not these independent cells.
"""

from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from check_python_sdist import verify_sdist_contents
from check_upstream_tests import ORIGINAL_TEST_CASES
from test_legacy_wheel import (
    build_and_check_sdist,
    stage_compatibility_tests,
    verify_upstream_snapshot,
    verify_wheel_contents,
)

MATRIX = (
    ("3.10", "1.26.4", "1.15.3", "8.4.2"),
    ("3.10", "2.2.6", "1.15.3", "8.4.2"),
    ("3.11", "1.26.4", "1.17.1", "8.4.2"),
    ("3.11", "2.3.4", "1.17.1", "8.4.2"),
    ("3.12", "1.26.4", "1.17.1", "8.4.2"),
    ("3.12", "2.3.4", "1.17.1", "8.4.2"),
    ("3.12", "2.5.4", "1.18.1", "9.1.1"),
    ("3.13", "2.3.4", "1.17.1", "8.4.2"),
    ("3.14", "2.5.4", "1.18.1", "9.1.1"),
    ("3.15", "2.5.4", "1.18.1", "9.1.1"),
)
# Coverage inventory for the staged suites in this revision. Keep source-build
# totals derived from these values, and reject failures/skips independently.
COMPATIBILITY_CASES = 247
MODERN_CASES = 236
NUMPY_ONLY_CASES = 123

POLARS_VERSIONS = ("1.0.0", "1.20.0", "1.38.1", "1.44.2", "2.0.0")

MODERN_TESTS = (
    "bindings/python/tests/test_distribution_identity.py",
    "bindings/python/tests/test_api_python.py",
    "bindings/python/tests/test_consumer_contracts.py",
    "bindings/python/tests/test_api_polars.py",
    "bindings/python/tests/test_day_numpy_input.py",
    "bindings/python/tests/test_day_numpy_output.py",
    "bindings/python/tests/test_digest_regressions.py",
    "integration/api_coherence/test_cluster_mean_bounds.py",
    "integration/api_coherence/test_subnormal_query_coherence.py",
    "integration/api_coherence/test_transactional_weight_scaling.py",
    "integration/api_coherence/test_k2norm_domain.py",
)

# Run by the installed cell interpreter with -I, outside the source checkout.
INSPECT = """
import hashlib, importlib, importlib.metadata as metadata, importlib.util, json, pathlib, sys
expected = json.loads(sys.argv[1])
assert sys.flags.isolated == 1
assert '.'.join(map(str, sys.version_info[:2])) == expected['python']
assert sys.implementation.name == 'cpython' and sys.version_info.releaselevel == 'final', sys.version
prefix = pathlib.Path(sys.prefix).resolve()
names = ('tdigest_rs', 'tdigest_rs.lib', 'tdigest_rs.tdigest_rs',
         'tdigest_rs._modern', 'tdigest_rs._native')
assert importlib.util.find_spec('gr_tdigest') is None
modules = {name: importlib.import_module(name) for name in names}
paths = {name: pathlib.Path(module.__file__).resolve() for name, module in modules.items()}
assert all(path.is_relative_to(prefix) for path in paths.values()), paths
assert paths['tdigest_rs.tdigest_rs'].name == 'tdigest_rs.py', paths
assert modules['tdigest_rs'].TDigest.__module__ == 'tdigest_rs.lib'
import inspect
package = paths['tdigest_rs'].parent
assert all((package / name).is_file() for name in ('py.typed', '__init__.pyi', '_native.pyi'))
assert 'max_size' in inspect.signature(modules['tdigest_rs'].TDigest.from_array).parameters
assert 'Midpoint CDF' in modules['tdigest_rs'].TDigest.from_array([1], max_size=32).cdf.__doc__
native_hash = hashlib.sha256(paths['tdigest_rs._native'].read_bytes()).hexdigest()
assert native_hash == expected['native_sha256'], native_hash
dist = metadata.distribution(expected['distribution'])
assert dist.version == expected['version'], dist.version
assert sorted(part.strip() for part in dist.metadata['Requires-Python'].split(',')) == ['<3.16', '>=3.10'], dist.metadata['Requires-Python']
versions = {name: metadata.version(name) for name in ('numpy', 'scipy', 'pytest', 'pip')}
if expected['polars'] is None:
    assert importlib.util.find_spec('polars') is None
    assert not any(name == 'polars' or name.startswith('polars.') for name in sys.modules)
    try:
        metadata.distribution('polars')
    except metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError('base installation unexpectedly includes Polars')
else:
    versions['polars'] = metadata.version('polars')
    assert versions['polars'] == expected['polars']
for name in ('numpy', 'scipy', 'pytest'):
    assert versions[name] == expected[name], (name, versions[name])
print(json.dumps({'python': sys.version, 'executable': sys.executable,
                  'dependencies': versions, 'native_sha256': native_hash,
                  'imports': {name: str(path) for name, path in paths.items()},
                  'distribution': dist.metadata['Name'], 'version': dist.version}, indent=2))
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def inspect_wheel(wheel: Path) -> dict:
    verify_wheel_contents(wheel)
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        (metadata_name,) = (name for name in names if name.endswith(".dist-info/METADATA"))
        (wheel_name,) = (name for name in names if name.endswith(".dist-info/WHEEL"))
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        wheel_metadata = Parser().parsestr(archive.read(wheel_name).decode())
        tags = wheel_metadata.get_all("Tag", [])
        require(
            bool(tags) and all(tag.startswith("cp310-abi3-") for tag in tags),
            f"expected a genuine cp310 ABI3 wheel, got {tags}",
        )
        require(
            sorted(part.strip() for part in metadata["Requires-Python"].split(",")) == ["<3.16", ">=3.10"],
            "unexpected Python support metadata",
        )
        require("polars" in metadata.get_all("Provides-Extra", []), "wheel omits the Polars extra")
        polars_requirements = [item for item in metadata.get_all("Requires-Dist", []) if item.startswith("polars")]
        require(
            len(polars_requirements) == 1
            and polars_requirements[0].replace(" ", "").replace("'", '"') == 'polars>=1.0.0;extra=="polars"',
            f"Polars must only be the tested optional extra, got {polars_requirements}",
        )
        native_sha = hashlib.sha256(archive.read("tdigest_rs/_native.abi3.so")).hexdigest()
    return {
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "native_sha256": native_sha,
        "distribution": metadata["Name"],
        "version": metadata["Version"],
        "requires_python": metadata["Requires-Python"],
        "requires_dist": metadata.get_all("Requires-Dist"),
        "tags": tags,
    }


def check_results(path: Path, expected: int) -> int:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    require(len(cases) == expected, f"expected {expected} cases in {path}; got {len(cases)}")
    require(
        not any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")),
        f"skipped or failing cases in {path}",
    )
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="test this existing wheel; never retag or rebuild it")
    parser.add_argument("--sdist", type=Path, help="also check this prebuilt source archive with --wheel")
    parser.add_argument("--output-dir", type=Path, help="new directory for persistent logs and results")
    args = parser.parse_args()
    if args.sdist and not args.wheel:
        parser.error("--sdist requires --wheel; otherwise the gate builds both artifacts")
    root = Path(__file__).resolve().parents[1]
    verify_upstream_snapshot(root)
    output = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="tdigest-support-results-"))
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTEST_ADDOPTS", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    env.update(
        {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PIP_CONFIG_FILE": "/dev/null",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
        }
    )
    # Source archives are extracted into temporary directories. Keep dependency
    # artifacts in the same cached target as the checkout's release-wheel build;
    # Cargo still fingerprints and compiles the extracted source independently.
    target = Path(env.get("CARGO_TARGET_DIR", root / "target"))
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    journal = []

    def run(label: str, command: list[str], cwd: Path, *, build: bool = False) -> str:
        print(f"{label} (log: {output / (label + '.log')})", flush=True)
        command_env = env.copy()
        if build:
            # The project tools already exist in the development environment;
            # these are never used by the isolated installed-wheel test cells.
            command_env["UV_PROJECT_ENVIRONMENT"] = str(root / ".venv")
            command_env["CARGO_TARGET_DIR"] = str(target)
        entry = {"label": label, "command": command, "cwd": str(cwd), "exit_code": None}
        if build:
            entry["cargo_target_dir"] = str(target)
        journal.append(entry)
        (output / "commands.json").write_text(json.dumps(journal, indent=2) + "\n")
        start = time.perf_counter()
        log = output / (label + ".log")
        with log.open("w", buffering=1) as stream:
            process = subprocess.Popen(
                command, cwd=cwd, env=command_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                if build:
                    print(line, end="", flush=True)
            entry["exit_code"] = process.wait()
        entry["elapsed_seconds"] = time.perf_counter() - start
        (output / "commands.json").write_text(json.dumps(journal, indent=2) + "\n")
        print(f"{label} completed in {entry['elapsed_seconds']:.1f}s", flush=True)
        require(entry["exit_code"] == 0, f"{label} failed; see {log}")
        return log.read_text()

    # Large dependency matrices can exhaust a tmpfs-backed /tmp. Keep cells
    # beside build outputs so uv can also hardlink from its disk-backed cache.
    scratch = root / "build"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tdigest-support-cells-", dir=scratch) as temporary:
        work = Path(temporary)
        tests, parity = stage_compatibility_tests(root, work)
        modern = work / "modern"
        modern.mkdir()
        for relative in MODERN_TESTS:
            shutil.copyfile(root / relative, modern / Path(relative).name)
        run("python-runtimes", ["uv", "python", "install", *dict.fromkeys(row[0] for row in MATRIX)], work)
        cells = []
        variants = [(*row, polars) for row in MATRIX for polars in POLARS_VERSIONS]
        baselines = {}
        for minor, numpy, scipy, pytest in MATRIX:
            baselines.setdefault(minor, (minor, numpy, scipy, pytest, None))
        variants.extend(baselines.values())
        for minor, numpy, scipy, pytest, polars in variants:
            label = f"py{minor.replace('.', '')}-numpy{numpy}" + (
                "-no-polars" if polars is None else f"-polars{polars}"
            )
            venv = work / label
            run(label + "-venv", ["uv", "venv", "--python", minor, str(venv)], work)
            python = str(venv / "bin/python")
            run(label + "-pip", [python, "-I", "-m", "ensurepip", "--upgrade"], work)
            requirements = [f"numpy=={numpy}", f"scipy=={scipy}", f"pytest=={pytest}"]
            if polars:
                requirements.append(f"polars=={polars}")
            run(
                label + "-deps",
                ["uv", "pip", "install", "--only-binary", ":all:", "--python", python, *requirements],
                work,
            )
            cells.append(
                {
                    "label": label,
                    "python": minor,
                    "executable": python,
                    "numpy": numpy,
                    "scipy": scipy,
                    "polars": polars,
                    "pytest": pytest,
                }
            )
        if args.wheel:
            wheel = args.wheel.resolve(strict=True)
            sdist = args.sdist.resolve(strict=True) if args.sdist else None
            if sdist:
                verify_sdist_contents(sdist, root)
        else:
            wheels = output / "wheels"
            sdist = build_and_check_sdist(
                root, output / "sdist", lambda *command, cwd: run("source-archive", list(command), cwd, build=True)
            )
            unpack = work / "source"
            unpack.mkdir()
            with tarfile.open(sdist, "r:gz") as archive:
                # Verification rejected traversal, links and duplicate paths.
                if hasattr(tarfile, "data_filter"):
                    archive.extractall(unpack, filter="data")
                else:
                    archive.extractall(unpack)
            (source,) = unpack.iterdir()
            # Compile the archive once on the oldest supported interpreter.
            # Every runtime cell below checks this same ABI3 native binary.
            run(
                "release-wheel",
                [
                    "uv",
                    "run",
                    "--project",
                    str(root / "bindings/python"),
                    "--no-sync",
                    "maturin",
                    "build",
                    "--release",
                    "--locked",
                    "-F",
                    "python",
                    "--zig",
                    "--compatibility",
                    "manylinux_2_28",
                    "--interpreter",
                    cells[0]["executable"],
                    "-o",
                    str(wheels),
                ],
                source,
                build=True,
            )
            (wheel,) = wheels.glob("*.whl")
        artifact = inspect_wheel(wheel)
        if sdist:
            artifact.update({"sdist": str(sdist), "sdist_sha256": hashlib.sha256(sdist.read_bytes()).hexdigest()})
        if not args.wheel:
            artifact["source_build_python"] = cells[0]["python"]
        (output / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
        results = []
        for cell in cells:
            label, python = cell["label"], cell["executable"]
            result = {**cell, "status": "RUNNING"}
            results.append(result)
            try:
                # Start with the user's existing Polars. Neither an ordinary
                # install nor opting into the extra may replace that version.
                run(label + "-base-install", [python, "-I", "-m", "pip", "install", "--no-index", str(wheel)], work)
                run(label + "-base-pip-check", [python, "-I", "-m", "pip", "check"], work)
                imports = run(
                    label + "-base-imports", [python, "-I", "-c", INSPECT, json.dumps({**artifact, **cell})], work
                )
                result["base_installed"] = json.loads(imports)
                if cell["polars"]:
                    run(
                        label + "-extra-install",
                        [python, "-I", "-m", "pip", "install", "--no-index", str(wheel) + "[polars]"],
                        work,
                    )
                    run(label + "-extra-pip-check", [python, "-I", "-m", "pip", "check"], work)
                    imports = run(
                        label + "-extra-imports", [python, "-I", "-c", INSPECT, json.dumps({**artifact, **cell})], work
                    )
                result["installed"] = json.loads(imports)
                suites = [
                    ("upstream_legacy", [tests], ORIGINAL_TEST_CASES),
                    ("compatibility", [parity], COMPATIBILITY_CASES - ORIGINAL_TEST_CASES),
                ]
                if cell["polars"]:
                    suites.append(("modern", [modern], MODERN_CASES))
                else:
                    suites.append(
                        (
                            "modern_python",
                            [
                                modern / name
                                for name in (
                                    "test_distribution_identity.py",
                                    "test_api_python.py",
                                    "test_day_numpy_input.py",
                                    "test_day_numpy_output.py",
                                    "test_digest_regressions.py",
                                )
                            ],
                            NUMPY_ONLY_CASES,
                        )
                    )
                for suite, directories, expected in suites:
                    xml = output / f"{label}-{suite}.xml"
                    run(
                        label + "-" + suite,
                        [
                            python,
                            "-I",
                            "-m",
                            "pytest",
                            "-q",
                            "-c",
                            "/dev/null",
                            "-p",
                            "no:cacheprovider",
                            "-o",
                            "xfail_strict=true",
                            "--junitxml",
                            str(xml),
                            *(str(path) for path in directories),
                        ],
                        work,
                    )
                    result[suite + "_passed"] = check_results(xml, expected)
                result["status"] = "PASS"
            except Exception as error:
                result["status"] = "FAIL"
                result["error"] = str(error)
                print(f"{label}: FAIL: {error}", flush=True)
            (output / "matrix.json").write_text(json.dumps(results, indent=2) + "\n")
        require(hashlib.sha256(wheel.read_bytes()).hexdigest() == artifact["wheel_sha256"], "wheel changed during gate")
        require(
            len(results) == len(variants) and all(result["status"] == "PASS" for result in results),
            f"runtime support failed; inspect {output}",
        )
    print(
        f"{len(MATRIX) * len(POLARS_VERSIONS)} full cells passed on Polars {POLARS_VERSIONS}; "
        f"{len(baselines)} Polars-free cells passed; evidence: {output}"
    )
    if not args.wheel:
        print(f"Source archive compiled once with Python {cells[0]['python']}; wheel passed every runtime cell")


if __name__ == "__main__":
    main()
