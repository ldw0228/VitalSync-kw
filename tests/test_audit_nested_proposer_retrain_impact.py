from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_nested_proposer_retrain_impact.py"
SPEC = importlib.util.spec_from_file_location("audit_nested_proposer_retrain_impact", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _record(prediction: str, checkpoint: str) -> dict[str, object]:
    return {
        "all_window_prediction": {"sha256": prediction},
        "checkpoint": {"sha256": checkpoint},
    }


def _maps() -> tuple[dict[tuple[int, int, str], dict[str, object]], dict[tuple[int, int, str], dict[str, object]]]:
    main: dict[tuple[int, int, str], dict[str, object]] = {}
    retrain: dict[tuple[int, int, str], dict[str, object]] = {}
    for outer in sorted(AUDIT.merge.RETRAIN_FOLDS):
        for seed in AUDIT.merge.SEEDS:
            for position, name in enumerate(sorted(AUDIT.merge._manifest_name_for_outer(outer))):
                digest = f"{outer}{seed}{position}".encode().hex().ljust(64, "0")[:64]
                checkpoint = f"c{outer}{seed}{position}".encode().hex().ljust(64, "1")[:64]
                key = (outer, seed, name)
                main[key] = _record(digest, checkpoint)
                retrain[key] = _record(digest, checkpoint)
    return main, retrain


def test_no_prediction_drift_requires_no_forced_hcs_units() -> None:
    main, retrain = _maps()
    result = AUDIT.compare_record_maps(main, retrain)
    assert result["comparison_units"] == 30
    assert result["changed_prediction_units"] == 0
    assert result["force_retrain_unit_count"] == 0
    assert result["force_retrain_units_cli_value"] == ""
    assert result["force_retrain_argument"] == []


def test_one_prediction_change_forces_whole_fold_seed_pair() -> None:
    main, retrain = _maps()
    keys = sorted(key for key in retrain if key[:2] == (3, 20260828))
    retrain[keys[0]]["all_window_prediction"] = {"sha256": "f" * 64}
    retrain[keys[1]]["all_window_prediction"] = {"sha256": "e" * 64}
    result = AUDIT.compare_record_maps(main, retrain)
    assert result["changed_prediction_units"] == 2
    assert result["force_retrain_unit_count"] == 1
    assert result["force_retrain_units_cli_value"] == "3:20260828"
    assert result["force_retrain_argument"] == [
        "--force-retrain-units",
        "3:20260828",
    ]


def test_checkpoint_change_alone_does_not_force_hcs_rebuild() -> None:
    main, retrain = _maps()
    key = sorted(retrain)[0]
    retrain[key]["checkpoint"] = {"sha256": "d" * 64}
    result = AUDIT.compare_record_maps(main, retrain)
    assert result["changed_checkpoint_units"] == 1
    assert result["changed_prediction_units"] == 0
    assert result["force_retrain_unit_count"] == 0


def test_incomplete_retrain_cover_is_rejected() -> None:
    main, retrain = _maps()
    retrain.pop(next(iter(retrain)))
    with pytest.raises(RuntimeError, match="exact folds-3/4 30-unit cover"):
        AUDIT.compare_record_maps(main, retrain)


def test_invalid_prediction_hash_is_rejected() -> None:
    main, retrain = _maps()
    key = sorted(retrain)[0]
    retrain[key]["all_window_prediction"] = {"sha256": "not-a-hash"}
    with pytest.raises(RuntimeError, match="prediction SHA-256 is invalid"):
        AUDIT.compare_record_maps(main, retrain)
