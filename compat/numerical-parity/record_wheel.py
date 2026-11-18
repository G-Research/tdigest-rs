"""Run with the published 0.1.0 environment: python -I record_wheel.py PATH_TO_WHEEL."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import runpy
import sys
import zipfile

import tdigest_rs
import tdigest_rs.tdigest_rs as native

wheel = Path(sys.argv[1])
sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
assert sys.flags.isolated and version("tdigest-rs") == "0.1.0"
assert sha == "62ee8d5a01226d020a2ab11a956292c38beab3f34d2d9fa7a8fca22c3e5d4bbf"
with zipfile.ZipFile(wheel) as archive:
    for path in [Path(tdigest_rs.__file__).with_name("lib.py"), Path(native.__file__)]:
        assert path.read_bytes() == archive.read(f"tdigest_rs/{path.name}")
tests = Path(__file__).resolve().parents[2] / "bindings/python/tests"
sys.path.insert(0, str(tests))
suite = runpy.run_path(str(tests / "test_delta_wheel_parity.py"))
cases = {}
for name, dtype, delta in suite["CASES"]:
    states = {}
    for stage, digest in suite["stages"](name, dtype, delta):
        snapshot = suite["snapshot"](digest)
        # Equal states reference their first recording so the fixture
        # displays each set of centroid and query bits just once per case.
        states[stage] = next((key for key, value in states.items() if value == snapshot), snapshot)
    cases[f"{name}-{dtype.__name__}-{delta}"] = states
out = {"wheel": wheel.name, "sha256": sha, "numpy": suite["np"].__version__, "cases": cases}
suite["FIXTURE"].write_text(json.dumps(out, indent=2) + "\n")
