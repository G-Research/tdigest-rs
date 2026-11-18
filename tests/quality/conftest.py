"""Pytest profiles, fixtures and optional retained measurement reports."""

from functools import lru_cache
import gzip
import hashlib
import importlib.metadata
from itertools import product
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pytest

from . import measurements as quality

DIRECTORY = Path(__file__).resolve().parent
PROTOCOL_SHA256 = hashlib.sha256(json.dumps(quality.PROTOCOL, sort_keys=True).encode()).hexdigest()


def pytest_addoption(parser):
    group = parser.getgroup("quality")
    group.addoption("--quality-extended", action="store_true", help="also collect the slow extended accuracy matrix")
    group.addoption("--quality-report", type=Path, help="retain results.json per mode, including failures")
    group.addoption("--quality-baseline-output", type=Path, help="write candidate frozen baselines for explicit review")


@lru_cache(maxsize=2)
def inputs(mode):
    return {name: (kind, values) for name, kind, values in quality.datasets(DIRECTORY / "data", mode)}


def profiles(mode):
    return {
        f"{name}/{precision}/size{budget}/{order}/{path}": {
            "id": f"{name}/{precision}/size{budget}/{order}/{path}",
            "dataset": name,
            "kind": inputs(mode)[name][0],
            "precision": precision,
            "max_size": budget,
            "order": order,
            "path": path,
        }
        for name, precision, budget, order, path in product(
            inputs(mode),
            quality.PROTOCOL["precisions"],
            quality.PROTOCOL["budgets"][mode],
            quality.PROTOCOL["orders"][mode],
            quality.PROTOCOL["paths"][mode],
        )
    }


def pytest_generate_tests(metafunc):
    if "quality_profile" not in metafunc.fixturenames:
        return
    modes = ["fast", "extended"] if metafunc.config.getoption("--quality-extended") else ["fast"]
    cases = [
        pytest.param((mode, ident), id=f"{mode}/{ident}", marks=pytest.mark.slow if mode == "extended" else ())
        for mode in modes
        for ident in profiles(mode)
    ]
    metafunc.parametrize("quality_profile", cases)


def load_run(mode):
    manifest_sha256 = quality.sha(DIRECTORY / "data/manifest.json")
    frozen = quality.read_json(DIRECTORY / f"baseline-{mode}.json.gz")
    assert frozen["protocol_sha256"] == PROTOCOL_SHA256, "baseline protocol changed"
    assert frozen["data_manifest_sha256"] == manifest_sha256, "baseline data changed"
    baseline = {row["id"]: row for row in frozen["rows"]}
    cases = profiles(mode)
    assert len(baseline) == len(frozen["rows"]) and set(baseline) == set(cases), "baseline/case coverage differs"
    known_findings = {}
    if mode == "extended":
        known = quality.read_json(DIRECTORY / "known-limits.json")
        assert known["protocol_sha256"] == PROTOCOL_SHA256, "known-limit protocol changed"
        known_findings = known["findings"]
        assert set(known_findings) <= set(cases), "known-limit case absent from matrix"
    native = Path(sys.modules["tdigest_rs._native"].__file__)
    receipt = {
        "baseline_protocol": quality.PROTOCOL,
        "engine": "shared Rust core, capacity mode",
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": quality.sha(Path(quality.__file__)),
        "test_inputs_sha256": {path.name: quality.sha(path) for path in sorted(DIRECTORY.glob("*.py"))},
        "data_manifest_sha256": manifest_sha256,
        "native_sha256": quality.sha(native),
        "python": sys.version,
        "package": importlib.metadata.version("tdigest-rs"),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "source_inputs_sha256": quality.source_hashes(),
        "mode": mode,
        "rows": [],
        "status": "INCOMPLETE",
        "regression_status": "INCOMPLETE",
        "complete": False,
        "input_status": "NOT_VERIFIED",
    }
    return dict(
        baseline=baseline,
        cases=cases,
        known_findings=known_findings,
        receipt=receipt,
        native=native,
        start=time.perf_counter(),
    )


def write_report(run, directory):
    receipt = run["receipt"]
    rows = receipt["rows"]
    receipt["elapsed_seconds"] = time.perf_counter() - run["start"]
    receipt["summary"] = quality.summarize(rows)
    receipt["facets"] = {
        field: {
            str(value): quality.summarize([row for row in rows if row[field] == value])["all"]
            for value in sorted({row[field] for row in rows})
        }
        for field in ("precision", "max_size", "path", "dataset")
    }
    findings = {row["id"]: row["accuracy_limits"] for row in rows if row.get("accuracy_limits")}
    errors = any("error" in row for row in rows)
    measured = {row["id"] for row in rows}
    receipt["complete"] = measured == set(run["cases"])
    expected = {ident: limits for ident, limits in run["known_findings"].items() if ident in measured}
    receipt["accuracy_status"] = "FAIL" if findings or errors else "PASS"
    receipt["regression_status"] = (
        "FAIL" if errors or findings != expected else "INCOMPLETE" if not receipt["complete"] else "PASS"
    )
    receipt["status"] = (
        "KNOWN_LIMITS" if receipt["regression_status"] == "PASS" and findings else receipt["regression_status"]
    )
    if receipt["input_status"] != "INVALID_INPUTS_CHANGED":
        try:
            quality.verify_inputs_unchanged(receipt, run["native"])
        except (RuntimeError, OSError) as error:
            receipt["input_status"] = "INVALID_INPUTS_CHANGED"
            receipt["input_error"] = str(error)
        else:
            receipt["input_status"] = "UNCHANGED"
    if receipt["input_status"] == "INVALID_INPUTS_CHANGED":
        receipt["regression_status"] = "FAIL"
        receipt["status"] = "INVALID_INPUTS_CHANGED"
    if directory:
        target = directory / receipt["mode"]
        target.mkdir(parents=True, exist_ok=True)
        quality.atomic_json(target / "results.json", receipt)


@pytest.fixture(scope="session")
def quality_runs(request):
    runs = {}
    yield runs
    for run in runs.values():
        write_report(run, request.config.getoption("--quality-report"))
    invalid = [mode for mode, run in runs.items() if run["receipt"]["input_status"] != "UNCHANGED"]
    assert not invalid, f"Source or installed native binary changed during quality measurements: {invalid}"
    for mode, run in runs.items():
        output = request.config.getoption("--quality-baseline-output")
        if output:
            rows = run["receipt"]["rows"]
            assert run["receipt"]["complete"] and all("modern" in row for row in rows), (
                "candidate baseline requires a complete measurement matrix"
            )
            candidate = {key: run["receipt"][key] for key in ("protocol_sha256", "data_manifest_sha256")}
            candidate["source_results_sha256"] = hashlib.sha256(
                (json.dumps(run["receipt"], indent=2, allow_nan=False) + "\n").encode()
            ).hexdigest()
            candidate["rows"] = [
                {"id": row["id"], "n": row["n"], "modern": {key: row["modern"][key] for key in quality.GATED}}
                for row in rows
            ]
            output.mkdir(parents=True, exist_ok=True)
            # Candidates never replace the checked-in baseline implicitly.
            with (output / f"baseline-{mode}.json.gz").open("xb") as stream:
                stream.write(gzip.compress(json.dumps(candidate, sort_keys=True).encode(), mtime=0))


@pytest.fixture
def quality_row(quality_profile, quality_runs):
    mode, ident = quality_profile
    if mode not in quality_runs:
        quality_runs[mode] = load_run(mode)
    run = quality_runs[mode]
    row = dict(run["cases"][ident])
    run["receipt"]["rows"].append(row)
    raw = inputs(mode)[row["dataset"]][1]
    values = np.array(raw, dtype=np.float32 if row["precision"] == "f32" else np.float64)
    if row["order"] == "reverse":
        values = values[::-1].copy()
    row["n"] = len(values)
    try:
        modern = quality.build(values, row["precision"], row["max_size"], row["path"])
        modern_bytes = len(modern) * (values.dtype.itemsize + 8 + 1)
        row["modern_state"] = dict(
            centroids=len(modern),
            field_bytes=modern_bytes,
            wire_bytes=len(modern.to_bytes()),
            **quality.invariants(modern, values, row["precision"]),
        )
        exact = quality.Exact(values.tolist())
        row["modern"] = quality.score(modern, exact)
        bound = 0.1 if row["max_size"] <= 32 else 0.05
        row["accuracy_limits"] = [metric for metric in ("rank_max", "cdf_max") if row["modern"][metric] > bound]
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        raise
    return row


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item):
    outcome = yield
    report = outcome.get_result()
    runs = item.funcargs.get("quality_runs", {})
    profile = item.funcargs.get("quality_profile")
    if profile and profile[0] in runs:
        run = runs[profile[0]]
        row = next((row for row in run["receipt"]["rows"] if row["id"] == profile[1]), None)
        if row is not None and report.failed:
            row["error"] = report.longreprtext
        directory = item.config.getoption("--quality-report")
        # Retain failures immediately and checkpoint long runs in batches.
        # Final summaries are calculated once by the session fixture.
        if report.failed:
            write_report(run, directory)
        elif directory and report.when == "call" and len(run["receipt"]["rows"]) % 20 == 0:
            target = directory / profile[0]
            target.mkdir(parents=True, exist_ok=True)
            quality.atomic_json(target / "results.json", run["receipt"])
