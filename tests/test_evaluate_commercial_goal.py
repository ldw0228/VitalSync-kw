from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_commercial_goal.py"
_SPEC = importlib.util.spec_from_file_location("snn_rr_commercial_goal", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
GOAL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = GOAL
_SPEC.loader.exec_module(GOAL)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cache_index": [1, 2, 3, 4],
            "fold": [0, 0, 1, 1],
            "identity": ["a", "a", "b", "b"],
            "session_id": ["s1", "s1", "s2", "s2"],
            "window_number": [0, 8, 0, 8],
            "window_start_s": [0.0, 32.0, 0.0, 32.0],
            "window_end_s": [31.9, 63.9, 31.9, 63.9],
            "rr_bpm": [10.0, 20.0, 26.0, 30.0],
            "prediction": [10.5, 19.5, 27.0, 29.0],
        }
    )


def test_candidate_audit_and_goal_checks_pass_for_small_fixture():
    frame = _frame()
    audit = GOAL.validate_locked_oof(frame, "prediction")
    full = GOAL.summarize_subset(frame, "prediction")
    high = GOAL.summarize_subset(frame.loc[frame.rr_bpm >= 25], "prediction")
    checks = GOAL.evaluate_goal_checks(full, high)
    assert audit["one_test_fold_per_identity"]
    assert checks == {
        "overall_mae": True,
        "identity_macro_mae": True,
        "overall_rmse": True,
        "within_2": True,
        "over_5": True,
        "high_rr_25_35_mae": True,
    }


def test_provenance_rejects_duplicate_index_or_cross_fold_identity():
    duplicate = _frame()
    duplicate.loc[1, "cache_index"] = duplicate.loc[0, "cache_index"]
    with pytest.raises(ValueError, match="duplicate"):
        GOAL.validate_locked_oof(duplicate, "prediction")

    cross_fold = _frame()
    cross_fold.loc[2, "identity"] = "a"
    with pytest.raises(ValueError, match="more than one"):
        GOAL.validate_locked_oof(cross_fold, "prediction")


def test_exact_expectation_rejects_subset_and_fold_drift():
    expected = _frame().drop(columns="prediction")
    audit = GOAL.validate_locked_oof(
        _frame().sample(frac=1.0, random_state=7),
        "prediction",
        expected=expected,
        expected_rows=4,
        expected_folds=2,
        expected_identities=2,
    )
    assert audit["candidate_complete"]
    assert audit["exact_valid_reference_index_match"]
    assert audit["exact_locked_fold_assignment_match"]

    with pytest.raises(RuntimeError, match="not the complete"):
        GOAL.validate_locked_oof(
            _frame().iloc[:-1],
            "prediction",
            expected=expected,
            expected_rows=4,
            expected_folds=2,
            expected_identities=2,
        )

    drift = _frame()
    drift.loc[drift["identity"] == "b", "fold"] = 0
    with pytest.raises(RuntimeError, match="fold assignment"):
        GOAL.validate_locked_oof(
            drift,
            "prediction",
            expected=expected,
            expected_rows=4,
            expected_folds=2,
            expected_identities=2,
        )


def test_greedy_nonoverlap_uses_intervals_not_only_window_modulo():
    frame = _frame()
    extra = frame.iloc[[0]].copy()
    extra["cache_index"] = 9
    extra["window_number"] = 1
    extra["window_start_s"] = 4.0
    extra["window_end_s"] = 35.9
    frame = pd.concat([frame, extra], ignore_index=True)
    mask = GOAL.greedy_nonoverlap_mask(frame)
    assert mask.sum() == 4
    assert not mask[-1]


def test_required_evidence_gates_distinguish_ranking_from_calibration():
    nonoverlap = GOAL.summarize_subset(_frame(), "prediction")
    assert GOAL._nonoverlap_evidence(nonoverlap, expected_identities=2)["passed"]
    assert not GOAL._calibration_evidence(
        "ranking_only_not_interval_calibrated"
    )["passed"]
    assert GOAL._calibration_evidence(
        "validation_locked_interval_calibrated"
    )["passed"]

    components = [
        {
            "input_kind": "raw_file_window",
            "warnings": [],
            "production_feature_bit_exact": True,
            "checkpoint_config_provenance": {"verified": True},
            "devices": {
                "cpu": {
                    "raw_window_in_memory_p50_ms": 10.0,
                    "raw_window_in_memory_p95_ms": 20.0,
                }
            },
        },
        {
            "input_kind": "raw_file_window",
            "warnings": [],
            "production_feature_bit_exact": True,
            "checkpoint_config_provenance": {"verified": True},
            "devices": {
                "cpu": {
                    "raw_window_in_memory_p50_ms": 15.0,
                    "raw_window_in_memory_p95_ms": 25.0,
                }
            },
        },
    ]
    benchmark, evidence = GOAL._e2e_evidence(components, stride_budget_ms=50.0)
    assert evidence["passed"]
    assert benchmark["conservative_sequential_estimate"]["cpu"][
        "sum_of_component_e2e_p95_ms"
    ] == 45.0


def test_e2e_summary_combines_nested_warnings_and_rejects_non_bit_exact(tmp_path):
    path = tmp_path / "e2e.json"
    path.write_text(
        json.dumps(
            {
                "checkpoint": {"path": "model.pt", "trainable_parameters": 12},
                "input": {
                    "kind": "raw_file_window",
                    "warnings": [
                        "outlier repair is initialized at the selected window boundary"
                    ],
                },
                "warnings": ["top-level warning"],
                "measurement_contract": {"production_feature_bit_exact": False},
                "pipeline": {
                    "checkpoint_config_provenance": {
                        "verified": True,
                        "status": "sha256_match",
                        "training_sha256": "abc",
                        "supplied_sha256": "abc",
                    }
                },
                "devices": {
                    "cpu": {
                        "device_name": "cpu",
                        "paths": {
                            "raw_window_in_memory": {
                                "stages": {
                                    "total_ms": {"p50_ms": 10.0, "p95_ms": 20.0}
                                }
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    component = GOAL._e2e_summary(path)
    assert component["warnings"] == [
        "top-level warning",
        "outlier repair is initialized at the selected window boundary",
    ]
    assert component["checkpoint_config_provenance"]["verified"] is True
    assert component["production_feature_bit_exact"] is False

    _, evidence = GOAL._e2e_evidence([component], stride_budget_ms=50.0)
    assert not evidence["passed"]
    assert evidence["timing_complete"]
    assert evidence["checkpoint_config_provenance_verified"]
    assert not evidence["production_feature_bit_exact"]
    assert evidence["window_boundary_warning"]
    assert evidence["status"] == "timing_complete_not_feature_bit_exact"


def test_e2e_evidence_requires_verified_checkpoint_config_provenance():
    component = {
        "input_kind": "raw_file_window",
        "warnings": [],
        "production_feature_bit_exact": True,
        "checkpoint_config_provenance": {"verified": False},
        "devices": {
            "cpu": {
                "raw_window_in_memory_p50_ms": 10.0,
                "raw_window_in_memory_p95_ms": 20.0,
            }
        },
    }
    _, evidence = GOAL._e2e_evidence([component], stride_budget_ms=50.0)
    assert not evidence["passed"]
    assert evidence["status"] == "timing_complete_provenance_unverified"


def test_robustness_evidence_requires_all_conditions_and_provenance(tmp_path):
    condition = {
        "overall": {"n": 4.0},
        "n_identities": 2,
    }
    path = tmp_path / "robustness.json"
    path.write_text(
        json.dumps(
            {
                "provenance_audit": {"passed": True},
                "radar_conditions": {
                    name: condition for name in GOAL.REQUIRED_RADAR_CONDITIONS
                },
            }
        ),
        encoding="utf-8",
    )
    _, evidence = GOAL._robustness_evidence(
        path, expected_rows=4, expected_identities=2
    )
    assert evidence["passed"]

    document = json.loads(path.read_text(encoding="utf-8"))
    document["provenance_audit"]["passed"] = False
    path.write_text(json.dumps(document), encoding="utf-8")
    _, evidence = GOAL._robustness_evidence(
        path, expected_rows=4, expected_identities=2
    )
    assert not evidence["passed"]
