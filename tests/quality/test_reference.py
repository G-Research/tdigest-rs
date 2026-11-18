"""Small hand-checkable controls for the independent accuracy measurements."""

from fractions import Fraction
from pathlib import Path

import pytest

from . import measurements as quality
from .measurements import Exact


def test_ties_have_a_rank_interval_instead_of_a_penalty_for_correct_values():
    exact = Exact([0, 0, 0, 10])
    assert exact.rank_interval(0) == (0.0, 0.75)
    assert exact.rank_error(0.5, 0) == 0
    assert exact.rank_error(0.9, 0) == pytest.approx(0.15)
    assert exact.cdf(0) == 0.375
    assert exact.cdf(5) == 0.75


@pytest.mark.parametrize("values", [[7] * 8, [-3, -1, 0, 0, 0, 2, 90], list(range(8))])
def test_ecdf_reference_matches_exact_rational_counting(values):
    reference = Exact(values)
    for probe in sorted(set(values + [min(values) - 1, max(values) + 1, 0.5])):
        left = Fraction(sum(v < probe for v in values), len(values))
        right = Fraction(sum(v <= probe for v in values), len(values))
        assert reference.cdf(probe) == float((left + right) / 2)
        assert reference.rank_interval(probe) == (float(left), float(right))


def test_type7_value_reference_is_hand_checkable_and_handles_endpoints():
    exact = Exact([0, 2, 10, 20])
    assert [exact.quantile(q) for q in [0, 0.25, 0.5, 0.75, 1]] == [0, 1.5, 6, 12.5, 20]
    assert exact.value_scale == 11
    assert Exact([4, 4]).value_scale == 1
    with pytest.raises(ValueError):
        exact.quantile(1.01)


@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf")]])
def test_bad_reference_data_is_rejected(bad):
    with pytest.raises(ValueError):
        Exact(bad)


class Fake:
    def inner_kind(self):
        return "f64"

    def quantile(self, q):
        return 0.0

    def cdf(self, x):
        return 0.0


def test_bad_predictor_negative_control_is_detected_in_separate_metrics():
    metrics = quality.score(Fake(), Exact([0, 0, 0, 10]))
    assert metrics["rank_max"] == 0.25
    assert metrics["value_abs_max"] == 10
    assert metrics["median_rank"] == 0
    assert metrics["cdf_max"] == 0.875


def test_accuracy_summary_counts_errors_and_limits_by_dataset():
    rows = [
        {"kind": "real", "accuracy_limits": ["rank_max"]},
        {"kind": "synthetic", "error": "invalid centroid mass"},
    ]
    summary = quality.summarize(rows)
    assert summary["real"] == {"profiles": 1, "errors": 0, "accuracy_limit_failures": 1}
    assert summary["synthetic"] == {"profiles": 1, "errors": 1, "accuracy_limit_failures": 0}
    assert summary["all"] == {"profiles": 2, "errors": 1, "accuracy_limit_failures": 1}


def test_baseline_limits_are_small_and_metric_specific():
    assert quality.regression_limit("rank_max", 0.01, 1000, "f64") == pytest.approx(0.011)
    assert quality.regression_limit("value_iqr_mean", 0.01, 1000, "f64") == pytest.approx(0.01, abs=2e-12)
    assert quality.regression_limit("cdf_max", 0.01, 1000, "f32") == pytest.approx(0.010002)


def test_real_corpus_contains_all_nine_measured_columns():
    import json

    directory = Path(__file__).resolve().parent / "data"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert [s["rows"] for s in manifest] == [17379, 19735]
    assert all(s["license"] == "CC-BY-4.0" for s in manifest)
    assert not {"rv1", "rv2"} & {column for s in manifest for column in s["columns"]}
    real = [data for data in quality.datasets(directory, "fast") if data[1] == "real"]
    assert len(real) == 9
    assert all(len(values) == 512 for _, _, values in real)


def test_corrupted_real_fixture_is_rejected_before_use(tmp_path):
    import shutil

    directory = Path(__file__).resolve().parent / "data"
    shutil.copyfile(directory / "manifest.json", tmp_path / "manifest.json")
    (tmp_path / "bike.csv.gz").write_bytes(b"changed fixture")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        list(quality.datasets(tmp_path, "fast"))


def test_quality_measurement_rejects_source_or_installed_binary_changes(monkeypatch, tmp_path):
    binary = tmp_path / "native.so"
    binary.write_bytes(b"measured binary")
    monkeypatch.setattr(quality, "source_hashes", lambda: {"source.rs": "a"})
    receipt = {"source_inputs_sha256": {"source.rs": "a"}, "native_sha256": quality.sha(binary)}
    quality.verify_inputs_unchanged(receipt, binary)
    monkeypatch.setattr(quality, "source_hashes", lambda: {"source.rs": "changed"})
    with pytest.raises(RuntimeError, match="changed"):
        quality.verify_inputs_unchanged(receipt, binary)
    monkeypatch.setattr(quality, "source_hashes", lambda: {"source.rs": "a"})
    binary.write_bytes(b"different binary")
    with pytest.raises(RuntimeError, match="changed"):
        quality.verify_inputs_unchanged(receipt, binary)


def test_changed_native_invalidates_pytest_receipt_and_blocks_candidate_baselines(tmp_path):
    import json
    from types import SimpleNamespace

    from . import conftest as fixtures

    report = tmp_path / "report"
    candidates = tmp_path / "candidates"
    options = {"--quality-report": report, "--quality-baseline-output": candidates}
    request = SimpleNamespace(config=SimpleNamespace(getoption=options.__getitem__))
    lifecycle = fixtures.quality_runs.__wrapped__(request)
    runs = next(lifecycle)
    run = fixtures.load_run("fast")
    native = tmp_path / "native.so"
    native.write_bytes(b"measured binary")
    run["native"] = native
    run["receipt"]["native_sha256"] = quality.sha(native)
    runs["fast"] = run
    native.write_bytes(b"changed during measurement")

    with pytest.raises(AssertionError, match="native binary changed"):
        next(lifecycle)

    saved = json.loads((report / "fast" / "results.json").read_text())
    assert saved["input_status"] == "INVALID_INPUTS_CHANGED"
    assert saved["regression_status"] == "FAIL"
    assert not candidates.exists()
