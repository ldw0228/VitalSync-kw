from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_harmonic_candidate_gate.py"
)
_SPEC = importlib.util.spec_from_file_location("audit_harmonic_candidate_gate", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_AUDIT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _AUDIT
_SPEC.loader.exec_module(_AUDIT)

StageAThresholds = _AUDIT.StageAThresholds
evaluate_policy = _AUDIT.evaluate_policy
evaluate_stage_gate = _AUDIT.evaluate_stage_gate
regression_metrics = _AUDIT.regression_metrics
target_dependent_candidate_oracle = _AUDIT.target_dependent_candidate_oracle
validate_identity_splits = _AUDIT.validate_identity_splits


def _thresholds() -> StageAThresholds:
    return StageAThresholds(
        action_auc_min=0.8,
        action_average_precision_min=0.45,
        factor_accuracy_gain_over_x1_prevalence_min=0.05,
        correction_precision_min=0.8,
        correction_recall_min=0.2,
        baseline_good_false_positive_fraction_max=0.01,
        estimated_macro_mae_gain_bpm_min=0.1,
    )


def test_regression_metrics_include_identity_macro_and_exact_boundaries() -> None:
    metrics = regression_metrics(
        np.asarray([10.0, 20.0, 30.0, 40.0]),
        np.asarray([11.0, 18.0, 35.0, 40.0]),
        np.asarray(["A", "A", "B", "B"]),
    )
    assert metrics["n"] == 4
    assert metrics["identity_count"] == 2
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["macro_mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(7.5))
    assert metrics["within_2"] == pytest.approx(0.75)
    # Five bpm is not a catastrophic >5 error.
    assert metrics["over_5"] == pytest.approx(0.0)


def test_policy_requires_precision_recall_safety_and_macro_gain_together() -> None:
    common = {
        "target": np.full(4, 20.0),
        "base_prediction": np.asarray([10.0, 10.0, 20.0, 20.0]),
        "correction": np.asarray([20.0, 20.0, 40.0, 20.0]),
        "identities": np.asarray(["A", "B", "C", "D"]),
        "actionable": np.asarray([True, True, False, False]),
        "base_good": np.asarray([False, False, True, True]),
        "pull": 1.0,
        "threshold": 0.5,
        "thresholds": _thresholds(),
    }
    safe = evaluate_policy(
        **common,
        selected=np.asarray([True, True, False, False]),
    )
    assert safe["precision"] == pytest.approx(1.0)
    assert safe["recall"] == pytest.approx(1.0)
    assert safe["base_good_false_positive_fraction"] == pytest.approx(0.0)
    assert safe["macro_mae_gain_bpm"] == pytest.approx(5.0)
    assert safe["passes"]
    assert all(safe["gate_checks"].values())

    unsafe = evaluate_policy(
        **common,
        selected=np.asarray([True, True, True, False]),
    )
    assert unsafe["precision"] == pytest.approx(2.0 / 3.0)
    assert unsafe["base_good_false_positive_fraction"] == pytest.approx(0.5)
    assert not unsafe["passes"]
    assert not unsafe["gate_checks"]["correction_precision"]
    assert not unsafe["gate_checks"]["baseline_good_false_positive_fraction"]


def test_stage_gate_requires_both_predeclared_discovery_partitions() -> None:
    passing = {
        "action_auc": 0.81,
        "action_average_precision": 0.46,
        "factor_accuracy_gain_over_x1_prevalence": 0.051,
        "passing_policy_count": 1,
    }
    result = evaluate_stage_gate({"3": passing, "4": dict(passing)}, _thresholds())
    assert result["passed"]
    assert result["decision"] == "advance_to_locked_stage_B_neural_validation"

    failed_partition = dict(passing)
    failed_partition["action_average_precision"] = 0.449
    failed = evaluate_stage_gate(
        {"3": passing, "4": failed_partition}, _thresholds()
    )
    assert not failed["passed"]
    assert not failed["partitions"]["4"]["checks"][
        "action_average_precision"
    ]
    assert (
        failed["decision"]
        == "kill_before_neural_training_and_preserve_exact_frozen_base"
    )


def test_identity_split_validation_fails_closed_on_group_leakage() -> None:
    identities = np.asarray(["A", "A", "B", "B"])
    clean = [
        (np.asarray([2, 3]), np.asarray([0, 1])),
        (np.asarray([0, 1]), np.asarray([2, 3])),
    ]
    audit = validate_identity_splits(identities, clean)
    assert [item["held_identities"] for item in audit] == [["A"], ["B"]]

    leaked = [(np.asarray([1, 2, 3]), np.asarray([0]))]
    with pytest.raises(RuntimeError, match="identity leakage"):
        validate_identity_splits(identities, leaked)


def test_oracle_diagnostic_rejects_set_equal_but_reordered_cache_index(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "cache_index": [11, 10],
            "fold": [1, 0],
            "session_id": ["S2", "S1"],
            "identity": ["B", "A"],
            "rr_bpm": [20.0, 10.0],
            "classical_rr_bpm": [20.0, 10.0],
            **{
                f"posterior_top{rank}_rr_bpm": [20.0, 10.0]
                for rank in range(1, 6)
            },
        }
    )
    path = tmp_path / "alias_oof.csv"
    frame.to_csv(path, index=False)
    metadata = pd.DataFrame(
        {
            "session_id": ["unused"] * 10 + ["S1", "S2"],
            "identity": ["unused"] * 10 + ["A", "B"],
            "classical_rr_bpm": [0.0] * 10 + [10.0, 20.0],
        }
    )
    base = {
        "index": np.asarray([10, 11], dtype=np.int64),
        "fold": np.asarray([0, 1], dtype=np.int16),
        "target": np.asarray([10.0, 20.0]),
        "prediction": np.asarray([10.0, 20.0]),
    }
    with pytest.raises(RuntimeError, match="cache_index is not exactly aligned"):
        target_dependent_candidate_oracle(
            alias_oof_csv=path,
            metadata=metadata,
            base=base,
        )
