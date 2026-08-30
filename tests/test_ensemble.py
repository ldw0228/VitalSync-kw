from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_ENSEMBLE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ensemble.py"
_SPEC = importlib.util.spec_from_file_location("snn_rr_ensemble_script", _ENSEMBLE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ENSEMBLE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ENSEMBLE
_SPEC.loader.exec_module(_ENSEMBLE)

PredictionBundle = _ENSEMBLE.PredictionBundle
align_prediction_bundles = _ENSEMBLE.align_prediction_bundles
ensemble_uncertainty = _ENSEMBLE.ensemble_uncertainty
fit_identity_balanced_affine = _ENSEMBLE.fit_identity_balanced_affine
select_blend_weight = _ENSEMBLE.select_blend_weight
select_disagreement_coefficient = _ENSEMBLE.select_disagreement_coefficient


def _bundle(index: np.ndarray) -> PredictionBundle:
    count = len(index)
    return PredictionBundle(
        index=np.asarray(index),
        target=np.arange(count, dtype=np.float32),
        prediction=np.arange(count, dtype=np.float32),
        rr_std=np.ones(count, dtype=np.float32),
        uncertainty=np.ones(count, dtype=np.float32),
        quality=np.ones(count, dtype=np.float32),
        observable=np.ones(count, dtype=bool),
        reference_valid=np.ones(count, dtype=bool),
        spike_rate=np.zeros(count, dtype=np.float32),
        radar_weights=np.full((count, 3), 1 / 3, dtype=np.float32),
    )


def test_blend_weight_uses_validation_macro_mae() -> None:
    target = np.zeros(2)
    prediction_a = np.array([1.0, -3.0])
    prediction_b = np.array([-1.0, 1.0])
    identity = np.array(["one", "two"])

    selected, rows = select_blend_weight(
        target,
        prediction_a,
        prediction_b,
        identity,
        grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    )

    assert selected == pytest.approx(0.25)
    assert len(rows) == 5
    assert min(row["validation_macro_mae"] for row in rows) == pytest.approx(0.25)


def test_disagreement_coefficient_is_selected_by_validation_risk() -> None:
    target = np.zeros(4)
    prediction = np.array([0.0, 4.0, 1.0, 3.0])
    source_a = np.array([0.0, 0.0, 1.0, 0.0])
    source_b = np.array([0.0, 1.0, 1.0, 1.0])
    base_a = np.array([0.0, 0.0, 1.0, 1.0])
    base_b = base_a.copy()
    identity = np.array(["a", "b", "c", "d"])

    selected, rows = select_disagreement_coefficient(
        target,
        prediction,
        identity,
        base_a,
        base_b,
        source_a,
        source_b,
        weight_a=0.5,
        coefficient_grid=(0.0, 1.0, 2.0),
        coverages=(0.5,),
    )

    assert selected == pytest.approx(2.0)
    assert rows[-1]["validation_selective_macro_mae"] == pytest.approx(0.5)


def test_uncertainty_is_weighted_score_plus_disagreement() -> None:
    result = ensemble_uncertainty(
        np.array([2.0]),
        np.array([4.0]),
        np.array([10.0]),
        np.array([13.0]),
        weight_a=0.25,
        disagreement_coefficient=2.0,
    )
    assert result == pytest.approx(np.array([9.5]))


def test_bundle_alignment_sorts_and_rejects_different_indices() -> None:
    first = _bundle(np.array([8, 3]))
    second = _bundle(np.array([3, 8]))
    # Make row targets describe cache index rather than original row position.
    first.target = np.array([8.0, 3.0])
    second.target = np.array([3.0, 8.0])
    aligned_a, aligned_b = align_prediction_bundles(first, second)
    assert aligned_a.index.tolist() == [3, 8]
    assert aligned_b.index.tolist() == [3, 8]

    with pytest.raises(ValueError, match="indices do not match"):
        align_prediction_bundles(first, _bundle(np.array([3, 9])))


def test_identity_balanced_affine_is_clipped_and_guarded() -> None:
    calibration = fit_identity_balanced_affine(
        np.array([1.0, 16.0]),
        np.array([0.0, 10.0]),
        np.array(["a", "b"]),
        slope_bounds=(0.7, 1.2),
        intercept_bounds=(-2.0, 2.0),
    )
    assert calibration["candidate_slope"] == pytest.approx(1.2)
    assert calibration["candidate_intercept"] == pytest.approx(2.0)
    assert calibration["selected_affine"] is True

    guarded = fit_identity_balanced_affine(
        np.array([10.0, 10.0]),
        np.array([10.0, 10.0]),
        np.array(["a", "b"]),
    )
    assert guarded["selected_affine"] is False
    assert guarded["slope"] == 1.0
    assert guarded["intercept"] == 0.0
