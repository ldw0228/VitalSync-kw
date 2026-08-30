from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/diagnose_v2_i3_failure_for_hfr_v3.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnose_v2_i3_failure_for_hfr_v3", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def test_semantic_hash_uses_canonical_json() -> None:
    left = {"b": [2, 1], "a": {"x": False}}
    right = {"a": {"x": False}, "b": [2, 1]}
    assert diagnostic.semantic_sha256(left) == diagnostic.semantic_sha256(right)


def test_oracle_selection_is_masked_and_uses_stable_lower_tie() -> None:
    bpm = np.asarray([[9.0, 11.0, 10.0], [20.0, 21.0, 22.0]])
    mask = np.asarray([[True, True, False], [False, True, True]])
    index, prediction, error = diagnostic._oracle_selection(
        bpm, mask, np.asarray([10.0, 21.75])
    )
    assert index.tolist() == [0, 2]
    assert prediction.tolist() == [9.0, 22.0]
    assert error.tolist() == [1.0, 0.25]


def test_prediction_metrics_include_high_rr_and_identity_macro() -> None:
    target = np.asarray([10.0, 12.0, 25.0, 30.0])
    prediction = np.asarray([11.0, 14.0, 26.0, 34.0])
    identity = np.asarray(["A", "A", "B", "B"])
    result = diagnostic.prediction_metrics(prediction, target, identity)
    assert result["mae_bpm"] == 2.0
    assert result["identity_macro_mae_bpm"] == 2.0
    assert result["within_1_bpm_fraction"] == 0.5
    assert result["high_rr_25_35"]["rows"] == 2
    assert result["high_rr_25_35"]["mae_bpm"] == 2.5


def _synthetic_raw() -> dict[str, np.ndarray]:
    target = np.asarray([10.0, 20.0, 30.0, 32.0])
    fallback = np.asarray([12.0, 15.0, 20.0, 20.0])
    bpm = np.zeros((4, 12), dtype=np.float64)
    mask = np.zeros((4, 12), dtype=bool)
    source = np.zeros((4, 12, 9), dtype=bool)
    primary = np.full((4, 12), -1, dtype=np.int16)
    for row, values in enumerate(
        ([10.0, 20.0], [10.0, 20.0], [10.0, 30.0], [8.0, 32.0])
    ):
        bpm[row, :2] = values
        mask[row, :2] = True
        source[row, 0, 2] = True
        source[row, 1, row % 4 + 2] = True
        primary[row, :2] = [2, row % 4 + 2]
    return {
        "target": target,
        "identity": np.asarray(["A", "A", "B", "B"]),
        "fallback": fallback,
        "source": fallback + 1.0,
        "candidate_bpm": bpm,
        "candidate_mask": mask,
        "candidate_source_mask": source,
        "candidate_primary_source": primary,
        "classical_rr": np.asarray([10.0, 10.0, 10.0, 8.0]),
    }


def test_discovery_analysis_reports_candidate_factor_and_regret() -> None:
    result = diagnostic.analyze_discovery_arrays(_synthetic_raw())
    assert result["candidate_oracle"]["mae_bpm"] == 0.0
    assert result["candidate_oracle"]["coverage_within_1_bpm_fraction"] == 1.0
    assert result["routing_regret"]["candidate_oracle"]["mean_bpm"] > 0.0
    factor = result["classical_factor_class_confusion_and_coverage"]
    assert factor["factor_classes"] == [1, 2, 3, 4]
    assert factor["confident_rows"] == 4
    assert len(factor["target_vs_fallback_implied_confusion_counts"]) == 4


def test_adaptive_entry_requires_every_fixed_seed() -> None:
    aggregate = {}
    for seed in diagnostic.FIXED_SEEDS:
        aggregate[str(seed)] = {
            "fallback": {
                "mae_bpm": 1.5,
                "high_rr_25_35": {"rows": 10, "mae_bpm": 4.0},
            },
            "candidate_oracle": {
                "mae_bpm": 0.5,
                "coverage_within_1_bpm_fraction": 0.9,
                "high_rr_25_35": {"rows": 10, "mae_bpm": 1.0},
            },
        }
    result = diagnostic.evaluate_adaptive_routing_entry(aggregate)
    assert result["all_fixed_seeds_passed"] is True
    aggregate[str(diagnostic.FIXED_SEEDS[-1])]["candidate_oracle"][
        "coverage_within_1_bpm_fraction"
    ] = 0.79
    result = diagnostic.evaluate_adaptive_routing_entry(aggregate)
    assert result["all_fixed_seeds_passed"] is False


def test_candidate_oracle_rejects_non_discovery_outer_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="restricted"):
        diagnostic.load_discovery_unit(
            seed=20260828,
            outer_run=0,
            discovery_root=tmp_path,
            cache_root=tmp_path,
        )


def test_locked_outcome_is_copied_from_report_without_raw_target() -> None:
    per_seed = {}
    for seed in diagnostic.FIXED_SEEDS:
        per_seed[str(seed)] = {
            "candidates": {
                "locked_final": {
                    "strata": {
                        "fold": {
                            str(fold): {"rows": fold + 1}
                            for fold in diagnostic.FULL_OUTER_FOLDS
                        }
                    }
                }
            }
        }
    result = diagnostic.locked_outcome_from_primary_report({"per_seed": per_seed})
    assert result["locked_raw_target_or_joined_array_opened"] is False
    assert result["design_selection_eligible"] is False
    assert set(result["per_seed_per_outer_fold"]["20260828"]) == {
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
    }


def test_create_once_report_is_immutable_and_content_hash_verifiable(
    tmp_path: Path,
) -> None:
    payload = {"classification": "test", "value": 1}
    payload["content_sha256"] = diagnostic.semantic_sha256(payload)
    output = tmp_path / "diagnostics" / "report.json"
    diagnostic._write_create_once_json(output, payload)
    saved = json.loads(output.read_text(encoding="utf-8"))
    claimed = saved.pop("content_sha256")
    assert diagnostic.semantic_sha256(saved) == claimed
    assert output.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError):
        diagnostic._write_create_once_json(output, payload)


def test_source_never_opens_locked_joined_npz() -> None:
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")
    assert 'np.load(joined_path' not in source
    assert "locked_raw_target_or_joined_array_opened" in source
