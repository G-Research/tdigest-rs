"""One independently rerunnable accuracy regression per input/storage/merge profile."""

import pytest

from .measurements import GATED, regression_limit


def test_accuracy_against_sorted_input(quality_profile, quality_row, quality_runs):
    mode, ident = quality_profile
    run = quality_runs[mode]
    frozen = run["baseline"][ident]
    assert quality_row["n"] == frozen["n"], "baseline observation count changed"
    for metric in GATED:
        actual = quality_row["modern"][metric]
        limit = regression_limit(metric, frozen["modern"][metric], quality_row["n"], quality_row["precision"])
        assert actual <= limit, f"{metric}: {actual} exceeds frozen limit {limit}"

    expected = run["known_findings"].get(ident, [])
    assert quality_row["accuracy_limits"] == expected, "accuracy-limit inventory changed; review the measurements"
    # Only the known accuracy envelope failure is expected. Invariant failures,
    # baseline regressions and changed findings above must still fail normally.
    if expected:
        pytest.xfail("known accuracy limits: " + ", ".join(expected))
