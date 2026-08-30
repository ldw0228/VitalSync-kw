from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "commercial_stack.py"
_SPEC = importlib.util.spec_from_file_location("snn_rr_commercial_stack", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_STACK = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STACK
_SPEC.loader.exec_module(_STACK)

apply_calibrator = _STACK.apply_calibrator
apply_empirical_uncertainty_scale = _STACK.apply_empirical_uncertainty_scale
causal_kalman_filter = _STACK.causal_kalman_filter
fit_calibrator = _STACK.fit_calibrator
fixed_nonoverlap_mask = _STACK.fixed_nonoverlap_mask
fit_empirical_uncertainty_scale = _STACK.fit_empirical_uncertainty_scale
identity_balanced_weights = _STACK.identity_balanced_weights
select_calibrator_leave_one_identity_out = (
    _STACK.select_calibrator_leave_one_identity_out
)
select_convex_stack = _STACK.select_convex_stack
simplex_grid = _STACK.simplex_grid
stacked_uncertainty = _STACK.stacked_uncertainty
uncertainty_gated_weights = _STACK.uncertainty_gated_weights


def test_simplex_grid_is_complete_and_valid() -> None:
    grid = simplex_grid(3, 0.25)
    assert grid.shape == (15, 3)
    np.testing.assert_allclose(grid.sum(axis=1), 1.0)
    assert np.all(grid >= 0)
    assert any(np.array_equal(row, [1.0, 0.0, 0.0]) for row in grid)


def test_convex_stack_selects_validation_macro_optimum() -> None:
    target = np.zeros(2)
    predictions = np.array([[1.0, -1.0], [-3.0, 1.0]])
    identities = np.array(["a", "b"])
    weights, report = select_convex_stack(
        target,
        predictions,
        identities,
        step=0.25,
        minimum_improvement=0.0,
    )
    np.testing.assert_allclose(weights, [0.25, 0.75])
    assert report["selected_macro_mae"] == pytest.approx(0.25)


def test_identity_weights_and_uncertainty_gate_are_normalized() -> None:
    identities = np.array(["a", "a", "a", "b"])
    weights = identity_balanced_weights(identities)
    assert weights[:3].sum() == pytest.approx(weights[3])

    dynamic = uncertainty_gated_weights(
        np.array([0.5, 0.5]),
        np.array([[1.0, 10.0], [10.0, 1.0]]),
        np.array([1.0, 1.0]),
        alpha=1.0,
        uniform_floor=0.0,
    )
    np.testing.assert_allclose(dynamic.sum(axis=1), 1.0)
    assert dynamic[0, 0] > dynamic[0, 1]
    assert dynamic[1, 1] > dynamic[1, 0]


def test_robust_affine_and_loio_guard() -> None:
    prediction = np.tile(np.array([8.0, 12.0, 16.0, 20.0]), 3)
    target = 1.2 * prediction - 1.0
    identities = np.repeat(["a", "b", "c"], 4)
    affine = fit_calibrator("robust_affine", prediction, target, identities)
    assert affine["slope"] == pytest.approx(1.2)
    assert affine["intercept"] == pytest.approx(-1.0)
    np.testing.assert_allclose(apply_calibrator(prediction, affine), target)

    spec, calibrated, report = select_calibrator_leave_one_identity_out(
        prediction,
        prediction,
        identities,
        minimum_improvement=0.01,
    )
    assert spec["kind"] == "identity"
    assert report["guard_rejected_calibration"] is True
    np.testing.assert_allclose(calibrated, prediction)


def test_causal_filter_never_uses_future_and_resets_sessions() -> None:
    prediction = np.array([10.0, 12.0, 40.0, 20.0, 21.0])
    uncertainty = np.ones(5)
    sessions = np.array(["a", "a", "a", "b", "b"])
    windows = np.array([0, 1, 2, 0, 1])
    filtered, score = causal_kalman_filter(
        prediction,
        uncertainty,
        sessions,
        windows,
        process_noise=0.5,
        measurement_scale=1.0,
        innovation_clip=4.0,
        score_median=1.0,
    )
    changed = prediction.copy()
    changed[2] = 6.0
    filtered_changed, _ = causal_kalman_filter(
        changed,
        uncertainty,
        sessions,
        windows,
        process_noise=0.5,
        measurement_scale=1.0,
        innovation_clip=4.0,
        score_median=1.0,
    )
    np.testing.assert_allclose(filtered[:2], filtered_changed[:2])
    assert filtered[3] == pytest.approx(prediction[3])
    assert np.all(score >= uncertainty)


def test_fixed_nonoverlap_and_stack_disagreement() -> None:
    mask = fixed_nonoverlap_mask(
        np.array(["a"] * 10 + ["b"] * 3),
        np.array(list(range(10)) + [5, 6, 13]),
        windows_apart=8,
    )
    assert np.flatnonzero(mask).tolist() == [0, 8, 10, 12]

    predictions = np.array([[10.0, 14.0], [20.0, 20.0]])
    uncertainty = np.ones_like(predictions)
    dynamic = np.full_like(predictions, 0.5)
    score = stacked_uncertainty(
        predictions, uncertainty, dynamic, disagreement_coefficient=1.0
    )
    assert score[0] > score[1]


def test_validation_empirical_uncertainty_scale_is_monotone_and_bounded() -> None:
    validation = np.linspace(10.0, 20.0, 101)
    rule = fit_empirical_uncertainty_scale(validation, quantile_count=11)
    transformed = apply_empirical_uncertainty_scale(
        np.array([5.0, 10.0, 15.0, 20.0, 25.0]), rule
    )
    assert np.all(np.diff(transformed) >= 0)
    assert transformed[0] == pytest.approx(0.05)
    assert transformed[-1] == pytest.approx(1.05)
