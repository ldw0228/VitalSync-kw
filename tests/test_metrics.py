import numpy as np
import pytest

from snn_rr.metrics import (
    clustered_bootstrap_mae,
    grouped_oof_metrics,
    identity_cluster_bootstrap_ci,
    regression_metrics,
    risk_coverage_curve,
)


def test_metrics_and_selective_curve():
    target = np.array([10.0, 20.0, 30.0, 40.0])
    prediction = np.array([10.0, 21.0, 34.0, 40.0])
    uncertainty = np.array([0.1, 0.2, 1.0, 0.1])
    metrics = regression_metrics(target, prediction)
    assert metrics["mae"] == 1.25
    curve = risk_coverage_curve(target, prediction, uncertainty, [1.0, 0.5])
    assert curve[1]["mae"] == 0.0


def test_cluster_bootstrap_is_deterministic():
    target = np.array([10.0, 10.0, 20.0, 20.0])
    prediction = np.array([11.0, 11.0, 24.0, 24.0])
    point, low, high = clustered_bootstrap_mae(
        target, prediction, ["a", "a", "b", "b"], samples=100, seed=3
    )
    assert point == 2.5
    assert low <= point <= high


def test_risk_coverage_never_undershoots_and_reports_macro_metrics():
    target = np.array([10.0, 20.0, 30.0])
    prediction = np.array([10.0, 23.0, 31.0])
    uncertainty = np.array([0.1, 0.9, 0.2])
    curve = risk_coverage_curve(
        target,
        prediction,
        uncertainty,
        [0.5],
        identities=["a", "b", "c"],
    )
    assert curve[0]["requested_coverage"] == 0.5
    assert curve[0]["coverage"] == 2 / 3
    assert curve[0]["n"] == 2.0
    assert curve[0]["mae"] == 0.5
    assert curve[0]["macro_mae"] == 0.5
    assert curve[0]["n_identities"] == 2.0


def test_general_identity_cluster_bootstrap_ci():
    target = np.array([10.0, 10.0, 20.0, 20.0])
    prediction = np.array([11.0, 11.0, 24.0, 24.0])
    first = identity_cluster_bootstrap_ci(
        target, prediction, ["a", "a", "b", "b"], samples=200, seed=7
    )
    second = identity_cluster_bootstrap_ci(
        target, prediction, ["a", "a", "b", "b"], samples=200, seed=7
    )
    assert first == second
    assert first["estimate"] == 2.5
    assert first["n_identities"] == 2.0


def test_grouped_oof_summary_and_group_integrity_audit():
    target = np.array([10.0, 11.0, 20.0, 21.0, 30.0, 31.0])
    prediction = target + np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    identities = np.array(["a", "a", "b", "b", "c", "c"])
    folds = np.array([0, 0, 1, 1, 2, 2])
    result = grouped_oof_metrics(
        target,
        prediction,
        identities,
        fold_ids=folds,
        bootstrap_samples=100,
    )
    assert result["overall"]["mae"] == 2.0
    assert result["identity_macro"]["macro_mae"] == 2.0
    assert result["identity_cluster_bootstrap_mae"]["estimate"] == 2.0
    assert result["n_identities"] == 3
    assert set(result["per_fold"]) == {"0", "1", "2"}

    bad_folds = np.array([0, 1, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="span multiple OOF folds"):
        grouped_oof_metrics(
            target,
            prediction,
            identities,
            fold_ids=bad_folds,
            bootstrap_samples=10,
        )


def test_metrics_reject_empty_nonfinite_and_invalid_uncertainty():
    with pytest.raises(ValueError, match="must not be empty"):
        regression_metrics(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="finite"):
        regression_metrics(np.array([1.0]), np.array([np.nan]))
    with pytest.raises(ValueError, match="uncertainty"):
        risk_coverage_curve(
            np.array([1.0]), np.array([1.0]), np.array([np.inf]), [1.0]
        )
