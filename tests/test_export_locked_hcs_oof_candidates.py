from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORT = _load("export_locked_hcs_oof_candidates", ROOT / "scripts/export_locked_hcs_oof_candidates.py")
RUN = _load("run_locked_hcs_oof_for_candidate_export", ROOT / "scripts/run_locked_hcs_oof.py")
EVALUATE = _load("evaluate_commercial_goal_for_candidate_export", ROOT / "scripts/evaluate_commercial_goal.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }


def _write_json(path: Path, document: dict[str, Any], *, content_hash: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(document)
    if content_hash:
        value["content_sha256"] = EXPORT.canonical_json_sha256(value)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _prediction(path: Path, *, fold: int, seed: int, target: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    index = np.asarray([2 * fold, 2 * fold + 1], dtype=np.int64)
    fallback = target[index].astype(np.float32) + np.float32((seed - 100) * 0.1)
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            cache_index=index,
            outer_fold=np.asarray(fold, dtype=np.int16),
            seed=np.asarray(seed, dtype=np.int64),
            fallback_rr_bpm=fallback,
            source_rr_bpm=fallback + np.float32(0.25),
            final_rr_bpm=fallback,
            applied_pull=np.zeros(2, dtype=np.float32),
            target_joined=np.asarray(False),
        )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "locked"
    target_path = root / "canonical_locked_hcs_targets.npz"
    receipt_path = root / "canonical_locked_hcs_targets_receipt.json"
    indices = np.arange(12, dtype=np.int64)
    folds = np.repeat(np.arange(6, dtype=np.int16), 2)
    target_rr = np.asarray([25.0 + 0.25 * row for row in range(12)], dtype=np.float32)
    identity = np.asarray([f"id{fold}" for fold in range(6) for _ in range(2)])
    valid = np.asarray([True, False] * 6, dtype=bool)
    session = np.asarray([f"S{fold:02d}" for fold in range(6) for _ in range(2)])
    window = np.asarray([position for _ in range(6) for position in range(2)], dtype=np.int64)
    target_arrays = {
        "cache_index": indices,
        "outer_fold": folds,
        "target_rr_bpm": target_rr,
        "identity": identity,
        "reference_valid": valid,
        "session_id": session,
        "window_number": window,
        "protocol": np.asarray(["paced" if fold % 2 else "rest" for fold in range(6) for _ in range(2)]),
        "window_start_s": window.astype(np.float64) * 32.0,
        "window_end_s": (window.astype(np.float64) + 1.0) * 32.0,
        "cache_session_position": folds.astype(np.int32),
        "cache_session_row": window.astype(np.int32),
    }
    root.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as stream:
        np.savez_compressed(stream, **target_arrays)

    seeds = [101, 102, 103]
    seal_units = []
    for seed in seeds:
        for fold in range(6):
            unit = root / "units" / f"outer_{fold}_seed_{seed}"
            prediction = unit / "sealed_label_free_predictions.npz"
            _prediction(prediction, fold=fold, seed=seed, target=target_rr)
            derived = unit / "derived_lock.json"
            _write_json(derived, {"fold": fold, "seed": seed})
            seal_units.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "derived_lock": _binding(derived),
                    "prediction": _binding(prediction),
                }
            )
    predictions_seal = {
        "schema_version": 1,
        "classification": "locked_hcs_oof_prediction_seal",
        "unit_count": 18,
        "target_join_authorized": True,
        "units": seal_units,
    }
    _write_json(root / "predictions_seal.json", predictions_seal)

    receipt = {
        "schema_version": 1,
        "classification": "retrospective_locked_hcs_canonical_target_artifact_receipt",
        "target_artifact_created_once": True,
        "target_artifact_overwrite_allowed": False,
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
        "target_artifact": _binding(target_path),
        "target_schema": EXPORT._array_schema(target_arrays),
        "row_count": 12,
        "valid_reference_rows": 6,
        "prediction_topology": {
            "folds": list(range(6)),
            "seeds": seeds,
            "unit_count": 18,
            "same_fold_indices_across_seeds": True,
            "disjoint_fold_indices_with_exact_contiguous_union": True,
        },
    }
    _write_json(receipt_path, receipt, content_hash=True)
    RUN.join_and_evaluate(root, target_path, orchestrator_command=["synthetic-join"])
    return {
        "root": root,
        "target": target_path,
        "target_receipt": receipt_path,
        "evaluation_lock": root / "evaluation_lock.json",
        "joined": root / "locked_hcs_oof_joined.npz",
        "metrics": root / "locked_hcs_oof_metrics.json",
        "output": tmp_path / "exports",
        "export_receipt": tmp_path / "exports" / "candidate_export_receipt.json",
    }


def _export(paths: dict[str, Path]) -> dict[str, Any]:
    return EXPORT.export_candidates(
        locked_oof_root=paths["root"],
        evaluation_lock=paths["evaluation_lock"],
        target_receipt=paths["target_receipt"],
        output_dir=paths["output"],
        receipt_output=paths["export_receipt"],
        expected_rows=6,
        expected_folds=6,
        expected_identities=6,
        expected_seed_count=3,
        orchestrator_command=["synthetic-export"],
    )


def test_actual_join_exports_three_immutable_commercial_goal_compatible_csvs(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    receipt = _export(paths)

    assert receipt["commercial_claim_authorized"] is False
    assert receipt["result_selection_performed"] is False
    assert receipt["threshold_fitting_performed"] is False
    assert receipt["topology_audit"]["seed_count"] == 3
    assert receipt["topology_audit"]["fold_count"] == 6
    assert receipt["topology_audit"]["identity_count"] == 6
    assert receipt["topology_audit"]["valid_reference_rows_per_seed"] == 6
    assert receipt["content_sha256"] == EXPORT.canonical_json_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    assert len(receipt["exports"]) == 3
    assert (paths["export_receipt"].stat().st_mode & 0o777) == 0o444

    expected_indices = set(range(0, 12, 2))
    for record in receipt["exports"]:
        path = Path(record["csv"]["path"])
        assert _sha(path) == record["csv"]["sha256"]
        assert (path.stat().st_mode & 0o777) == 0o444
        frame = pd.read_csv(path)
        assert set(frame["cache_index"]) == expected_indices
        assert (frame["seed"] == record["seed"]).all()
        assert np.array_equal(
            frame["final_rr_bpm"].to_numpy(),
            frame["prediction_uncalibrated_bpm"].to_numpy(),
        )
        audit = EVALUATE.validate_locked_oof(
            frame,
            "final_rr_bpm",
            expected_rows=6,
            expected_folds=6,
            expected_identities=6,
        )
        assert audit["unique_cache_indices"] is True
        assert audit["one_test_fold_per_identity"] is True
        summary = EVALUATE.summarize_subset(frame, "prediction_uncalibrated_bpm")
        assert summary["overall"]["n"] == 6.0


def test_export_is_create_once_and_never_overwrites_seed_csvs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = _export(paths)
    bytes_before = {
        Path(record["csv"]["path"]): Path(record["csv"]["path"]).read_bytes()
        for record in first["exports"]
    }
    receipt_before = paths["export_receipt"].read_bytes()
    with pytest.raises(EXPORT.CandidateExportError, match="overwrite forbidden"):
        _export(paths)
    assert paths["export_receipt"].read_bytes() == receipt_before
    assert all(path.read_bytes() == value for path, value in bytes_before.items())


def test_target_receipt_tamper_and_joined_topology_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    receipt_tamper = _fixture(tmp_path / "receipt")
    receipt_tamper["target_receipt"].chmod(0o644)
    document = json.loads(receipt_tamper["target_receipt"].read_text(encoding="utf-8"))
    document["valid_reference_rows"] = 7
    receipt_tamper["target_receipt"].write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(EXPORT.CandidateExportError, match="content_sha256 mismatch"):
        _export(receipt_tamper)
    assert not receipt_tamper["output"].exists()

    topology = _fixture(tmp_path / "topology")
    with np.load(topology["joined"], allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["cache_index"] = arrays["cache_index"].copy()
    arrays["cache_index"][1] = arrays["cache_index"][0]
    topology["joined"].chmod(0o644)
    with topology["joined"].open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    lock = json.loads(topology["evaluation_lock"].read_text(encoding="utf-8"))
    lock["outputs"]["joined_oof"] = _binding(topology["joined"])
    topology["evaluation_lock"].chmod(0o644)
    topology["evaluation_lock"].write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(EXPORT.CandidateExportError, match="exact unique valid-reference cover"):
        _export(topology)
    assert not topology["output"].exists()


def test_missing_evaluation_lock_blocks_before_target_receipt_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    paths["evaluation_lock"].unlink()
    original = EXPORT._read_json

    def guarded(path: Path, label: str, **kwargs: Any) -> dict[str, Any]:
        if path.resolve() == paths["target_receipt"].resolve():
            pytest.fail("target receipt must not be opened before evaluation lock")
        return original(path, label, **kwargs)

    monkeypatch.setattr(EXPORT, "_read_json", guarded)
    with pytest.raises(EXPORT.CandidateExportError, match="evaluation_lock.json is absent"):
        _export(paths)
    assert not paths["output"].exists()


def test_metrics_hash_and_values_are_both_enforced(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    metrics["per_seed"]["101"]["locked_final"]["mae"] += 0.5
    paths["metrics"].chmod(0o644)
    paths["metrics"].write_text(json.dumps(metrics), encoding="utf-8")
    lock = json.loads(paths["evaluation_lock"].read_text(encoding="utf-8"))
    lock["outputs"]["metrics"] = _binding(paths["metrics"])
    paths["evaluation_lock"].chmod(0o644)
    paths["evaluation_lock"].write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(EXPORT.CandidateExportError, match="metrics value mismatch"):
        _export(paths)
    assert not paths["output"].exists()
