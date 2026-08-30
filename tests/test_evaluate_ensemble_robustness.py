from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_ensemble_robustness.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "snn_rr_evaluate_ensemble_robustness", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
ROBUST = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ROBUST
_SPEC.loader.exec_module(ROBUST)


def _bundle(index, prediction, *, fold=(0, 1)):
    index = np.asarray(index, dtype=np.int64)
    return {
        "index": index,
        "target": np.asarray([12.0, 24.0], dtype=np.float32),
        "prediction": np.asarray(prediction, dtype=np.float32),
        "rr_std": np.asarray([1.0, 2.0], dtype=np.float32),
        "quality": np.asarray([0.5, 1.0], dtype=np.float32),
        "fold": np.asarray(fold, dtype=np.int16),
    }


def test_alignment_and_locked_fold_blend_are_order_invariant():
    a = _bundle([8, 3], [10.0, 20.0])
    a["target"] = a["target"][::-1].copy()
    b = _bundle([3, 8], [22.0, 14.0], fold=(1, 0))
    selection = {
        "0": {
            "selected": {
                "weight_a": 0.25,
                "weight_b": 0.75,
                "disagreement_coefficient": 2.0,
            }
        },
        "1": {
            "selected": {
                "weight_a": 0.75,
                "weight_b": 0.25,
                "disagreement_coefficient": 1.0,
            }
        },
    }
    aligned_a, aligned_b = ROBUST.align_masked_sources(a, b)
    weight, coefficient = ROBUST.locked_parameters(
        selection, aligned_a["fold"], "a", "b"
    )
    result = ROBUST.apply_locked_mask_blend(
        aligned_a,
        aligned_b,
        weight_a=weight,
        disagreement=coefficient,
    )
    assert result["index"].tolist() == [3, 8]
    assert result["prediction"].tolist() == pytest.approx([20.5, 13.0])
    assert result["disagreement_coefficient"].tolist() == [1.0, 2.0]
    assert np.isfinite(result["uncertainty"]).all()


def test_alignment_rejects_target_or_fold_drift():
    a = _bundle([3, 8], [10.0, 20.0])
    b = _bundle([3, 8], [11.0, 21.0])
    b["target"][0] += 1.0
    with pytest.raises(RuntimeError, match="targets differ"):
        ROBUST.align_masked_sources(a, b)


def test_locked_parameters_reject_weights_that_do_not_sum_to_one():
    selection = {
        "0": {
            "selected": {
                "weight_a": 0.6,
                "weight_b": 0.5,
                "disagreement_coefficient": 1.0,
            }
        }
    }
    with pytest.raises(RuntimeError, match="sum to one"):
        ROBUST.locked_parameters(selection, np.asarray([0]), "a", "b")


def test_full_mask_consistency_records_tolerance_and_rejects_drift():
    locked = {
        "index": np.asarray([3, 8]),
        "target": np.asarray([12.0, 24.0]),
        "prediction": np.asarray([11.0, 23.0]),
        "fold": np.asarray([1, 0]),
    }
    full = {
        "index": np.asarray([8, 3]),
        "target": np.asarray([24.0, 12.0]),
        "prediction": np.asarray([23.01, 11.01]),
        "fold": np.asarray([0, 1]),
    }
    audit = ROBUST.validate_full_mask_consistency(
        full, locked, prediction_atol_bpm=0.025
    )
    assert audit["passed"]
    assert audit["prediction_max_abs_difference_bpm"] == pytest.approx(0.01)
    assert audit["prediction_atol_bpm"] == 0.025

    full["prediction"][0] += 0.1
    with pytest.raises(RuntimeError, match="do not reproduce"):
        ROBUST.validate_full_mask_consistency(
            full, locked, prediction_atol_bpm=0.025
        )


def test_alignment_rejects_duplicate_indices_in_either_source():
    a = _bundle([3, 8], [10.0, 20.0])
    b = _bundle([3, 3], [11.0, 21.0])
    with pytest.raises(RuntimeError, match="duplicate"):
        ROBUST.align_masked_sources(a, b)
