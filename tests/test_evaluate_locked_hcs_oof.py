from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load(
    "evaluate_locked_hcs_oof_under_test", ROOT / "scripts/evaluate_locked_hcs_oof.py"
)
RUN = _load("run_locked_hcs_oof_for_primary_test", ROOT / "scripts/run_locked_hcs_oof.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }


def _write_json(
    path: Path, document: dict[str, Any], *, content_hash: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(document)
    if content_hash:
        value["content_sha256"] = EVALUATOR.canonical_json_sha256(value)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path, *, add_uncertainty: bool = False) -> dict[str, Path]:
    root = tmp_path / "locked"
    root.mkdir(parents=True)
    seeds = [20260828, 20260829, 20260830]
    identities = [f"P{fold:02d}" for fold in range(6)]
    windows_per_identity = 8
    rows = len(identities) * windows_per_identity
    cache_index = np.arange(rows, dtype=np.int64)
    fold = np.repeat(np.arange(6, dtype=np.int16), windows_per_identity)
    identity = np.repeat(np.asarray(identities), windows_per_identity)
    window = np.tile(np.arange(windows_per_identity, dtype=np.int64), len(identities))
    session = np.repeat(
        np.asarray([f"session-{name}" for name in identities]), windows_per_identity
    )
    # Exercise every fixed RR bin and the inclusive 25--35 tail definition.
    rr_pattern = np.asarray([10.0, 14.0, 20.0, 25.0, 30.0, 35.0, 36.0, 22.0])
    target_rr = np.tile(rr_pattern, len(identities)).astype(np.float32)
    start = window.astype(np.float64) * 4.0
    target_arrays = {
        "cache_index": cache_index,
        "outer_fold": fold,
        "target_rr_bpm": target_rr,
        "identity": identity,
        "reference_valid": np.ones(rows, dtype=bool),
        "session_id": session,
        "window_number": window,
        "protocol": np.repeat(
            np.asarray(["rest", "paced", "rest", "exercise", "paced", "exercise"]),
            windows_per_identity,
        ),
        "window_start_s": start,
        "window_end_s": start + 32.0,
        "cache_session_position": fold.astype(np.int32),
        "cache_session_row": window.astype(np.int32),
        "reference_quality": np.tile(
            np.asarray([0.2, 0.5, 0.7, 0.8, 0.9, 0.4, 0.6, 1.0], dtype=np.float32),
            len(identities),
        ),
        "reference_sigma_bpm": np.tile(
            np.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 0.8, 1.8, 3.0], dtype=np.float32),
            len(identities),
        ),
        "radar_observable": np.tile(
            np.asarray([True, True, False, True, False, True, True, False]),
            len(identities),
        ),
    }
    evaluation_spec = root / "locked_primary_evaluation_spec.json"
    EVALUATOR.freeze_evaluation_spec(
        evaluation_spec,
        expected_rows=rows,
        expected_identities=len(identities),
        bootstrap_samples=128,
        bootstrap_seed=17,
        bootstrap_confidence=0.95,
    )
    target_path = root / "canonical_locked_hcs_targets.npz"
    with target_path.open("wb") as stream:
        np.savez_compressed(stream, **target_arrays)

    fold_assignments = root / "fold_assignments.json"
    _write_json(
        fold_assignments,
        {"identity_to_fold": {name: position for position, name in enumerate(identities)}},
    )
    pretest_lock = root / "pretest_lock.json"
    _write_json(
        pretest_lock,
        {
            "schema_version": 1,
            "classification": "locked_hcs_oof_pretest_lock",
            "commercial_claim_authorized": False,
        },
    )
    pretest_sha = _sha(pretest_lock)

    seal_units: list[dict[str, Any]] = []
    receipt_inventory: list[dict[str, Any]] = []
    for seed_position, seed in enumerate(seeds):
        for outer_fold in range(6):
            selected = np.flatnonzero(fold == outer_fold)
            unit_root = root / "units" / f"outer_{outer_fold}_seed_{seed}"
            unit_root.mkdir(parents=True)
            prediction_path = unit_root / "sealed_label_free_predictions.npz"
            # Seed 0 passes easily; later seeds remain separate and visibly differ.
            fallback = target_rr[selected] + np.float32(0.4 + 0.1 * seed_position)
            source = target_rr[selected] + np.float32(0.2 + 0.1 * seed_position)
            final = target_rr[selected] + np.float32(0.1 + 0.1 * seed_position)
            with prediction_path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    cache_index=cache_index[selected],
                    outer_fold=np.asarray(outer_fold, dtype=np.int16),
                    seed=np.asarray(seed, dtype=np.int64),
                    fallback_rr_bpm=fallback.astype(np.float32),
                    source_rr_bpm=source.astype(np.float32),
                    final_rr_bpm=final.astype(np.float32),
                    applied_pull=np.ones(len(selected), dtype=np.float32),
                    target_joined=np.asarray(False),
                )
            derived_path = unit_root / "derived_inference_lock.json"
            _write_json(
                derived_path,
                {
                    "schema_version": 1,
                    "classification": "locked_hcs_oof_derived_test_inference",
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "target_artifact_opened": False,
                    "pretest_lock_sha256": pretest_sha,
                    "sealed_prediction": _binding(prediction_path),
                },
            )
            record = {
                "outer_fold": outer_fold,
                "seed": seed,
                "derived_lock": _binding(derived_path),
                "prediction": _binding(prediction_path),
            }
            seal_units.append(record)
            receipt_inventory.append(dict(record))

    predictions_seal_path = root / "predictions_seal.json"
    _write_json(
        predictions_seal_path,
        {
            "schema_version": 1,
            "classification": "locked_hcs_oof_all_label_free_predictions_sealed",
            "pretest_lock_sha256": pretest_sha,
            "unit_count": 18,
            "outer_folds": list(range(6)),
            "target_artifact_opened_before_seal": False,
            "target_join_authorized": True,
            "units": seal_units,
        },
    )
    target_receipt_path = root / "canonical_locked_hcs_targets_receipt.json"
    _write_json(
        target_receipt_path,
        {
            "schema_version": 1,
            "classification": "retrospective_locked_hcs_canonical_target_artifact_receipt",
            "target_artifact_created_once": True,
            "target_artifact_overwrite_allowed": False,
            "commercial_claim_authorized": False,
            "prospective_confirmation_required": True,
            "target_artifact": _binding(target_path),
            "target_schema": EVALUATOR._array_schema(target_arrays),
            "row_count": rows,
            "valid_reference_rows": rows,
            "prediction_topology": {
                "folds": list(range(6)),
                "seeds": seeds,
                "unit_count": 18,
                "same_fold_indices_across_seeds": True,
                "disjoint_fold_indices_with_exact_contiguous_union": True,
            },
            "source_bindings": {
                "predictions_seal": _binding(predictions_seal_path),
                "pretest_lock": _binding(pretest_lock),
                "fold_assignments": _binding(fold_assignments),
            },
            "prediction_inventory": receipt_inventory,
        },
        content_hash=True,
    )

    RUN.join_and_evaluate(root, target_path, orchestrator_command=["synthetic-join"])
    joined_path = root / "locked_hcs_oof_joined.npz"
    evaluation_lock_path = root / "evaluation_lock.json"
    if add_uncertainty:
        with np.load(joined_path, allow_pickle=False) as archive:
            joined = {name: np.asarray(archive[name]) for name in archive.files}
        joined["uncertainty_uncalibrated"] = np.abs(
            np.asarray(joined["final_rr_bpm"], float)
            - np.asarray(joined["target_rr_bpm"], float)
        ).astype(np.float32)
        joined_path.chmod(0o644)
        with joined_path.open("wb") as stream:
            np.savez_compressed(stream, **joined)
        evaluation_lock_path.chmod(0o644)
        lock = json.loads(evaluation_lock_path.read_text(encoding="utf-8"))
        lock["outputs"]["joined_oof"] = _binding(joined_path)
        evaluation_lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")

    output = tmp_path / "primary"
    return {
        "root": root,
        "target": target_path,
        "target_receipt": target_receipt_path,
        "joined": joined_path,
        "evaluation_lock": evaluation_lock_path,
        "evaluation_spec": evaluation_spec,
        "output": output,
        "report": output / "report.json",
        "csv": output / "metrics.csv",
        "receipt": output / "receipt.json",
    }


def _evaluate(paths: dict[str, Path]) -> dict[str, Any]:
    return EVALUATOR.evaluate_locked_oof(
        locked_oof_root=paths["root"],
        evaluation_lock=paths["evaluation_lock"],
        target_receipt=paths["target_receipt"],
        evaluation_spec=paths["evaluation_spec"],
        output_dir=paths["output"],
        report_output=paths["report"],
        csv_output=paths["csv"],
        receipt_output=paths["receipt"],
        expected_rows=48,
        expected_identities=6,
        bootstrap_samples=128,
        bootstrap_seed=17,
        bootstrap_confidence=0.95,
        orchestrator_command=["synthetic-primary-evaluation"],
    )


def test_primary_evaluator_publishes_complete_per_seed_immutable_evidence(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    receipt = _evaluate(paths)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))

    assert report["commercial_claim_authorized"] is False
    assert report["commercial_performance_proven"] is False
    assert report["prospective_confirmation_required"] is True
    assert report["cross_seed_pooling_performed"] is False
    assert report["evaluation_specification"] == _binding(paths["evaluation_spec"])
    spec = json.loads(paths["evaluation_spec"].read_text(encoding="utf-8"))
    assert spec["must_be_frozen_before_outer_test_inference"] is True
    assert spec["population"]["fixed_seeds"] == [20260828, 20260829, 20260830]
    assert spec["nonoverlap"]["fixed_phases"]["phases"] == list(range(8))
    assert spec["bootstrap"]["unit"] == "physical_identity"
    assert spec["paired_comparisons"] == {
        name: {"challenger": pair[0], "reference": pair[1]}
        for name, pair in EVALUATOR.PAIRED_COMPARISONS.items()
    }
    assert spec["post_target_prohibitions"]["calibration_or_uncertainty_fit"] is True
    assert (paths["evaluation_spec"].stat().st_mode & 0o777) == 0o444
    assert set(report["per_seed"]) == {"20260828", "20260829", "20260830"}
    assert report["provenance_audit"]["exact_seed_fold_cache_identity_alignment"] is True
    for seed, result in report["per_seed"].items():
        assert result["seed_evaluated_independently"] is True
        assert set(result["candidates"]) == {"fallback", "source", "locked_final"}
        assert set(result["paired_candidate_deltas"]) == set(
            EVALUATOR.PAIRED_COMPARISONS
        )
        assert set(result["candidates"]["locked_final"]["eight_fixed_window_phases"]) == {
            str(value) for value in range(8)
        }
        assert result["nonoverlap_audit"]["all_eight_fixed_phases_reported"] is True
        assert result["nonoverlap_audit"]["greedy_intervals_nonoverlapping"] is True
        assert result["identity_cluster_bootstrap"]["fixed_spec"]["samples"] == 128
        assert (
            result["identity_cluster_bootstrap"]["fixed_spec"]["unit"]
            == "physical_identity"
        )
        strata = result["candidates"]["locked_final"]["strata"]
        assert {"identity", "fold", "session", "protocol", "rr_band"} <= set(strata)
        assert "qc:reference_quality" in strata
        assert "qc:reference_sigma_bpm" in strata
        assert "qc:radar_observable" in strata

    assert receipt["commercial_claim_authorized"] is False
    assert receipt["one_report_per_prespecified_seed_no_pooling"] is True
    assert receipt["outputs"]["report"] == _binding(paths["report"])
    assert receipt["outputs"]["metrics_csv"] == _binding(paths["csv"])
    assert (paths["report"].stat().st_mode & 0o777) == 0o444
    assert (paths["csv"].stat().st_mode & 0o777) == 0o444
    assert (paths["receipt"].stat().st_mode & 0o777) == 0o444
    with paths["csv"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["record_type"] for row in rows} == {"metric", "paired_delta"}
    assert {int(row["seed"]) for row in rows} == {20260828, 20260829, 20260830}


def test_evaluation_lock_is_validated_before_any_target_bearing_npz_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    paths["evaluation_lock"].chmod(0o644)
    lock = json.loads(paths["evaluation_lock"].read_text(encoding="utf-8"))
    lock["commercial_claim_authorized"] = True
    paths["evaluation_lock"].write_text(json.dumps(lock), encoding="utf-8")

    opened: list[Path] = []
    original = EVALUATOR._load_npz

    def guarded(path: Path, *, label: str) -> dict[str, np.ndarray]:
        opened.append(path)
        return original(path, label=label)

    monkeypatch.setattr(EVALUATOR, "_load_npz", guarded)
    with pytest.raises(EVALUATOR.LockedPrimaryEvaluationError, match="authorization"):
        _evaluate(paths)
    assert opened == []
    assert not paths["output"].exists()


def test_evaluation_spec_is_hash_verified_before_target_bearing_artifact_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    paths["evaluation_spec"].chmod(0o644)
    spec = json.loads(paths["evaluation_spec"].read_text(encoding="utf-8"))
    spec["point_gates"]["overall_mae"]["threshold"] = 99.0
    spec["content_sha256"] = EVALUATOR.canonical_json_sha256(
        {key: value for key, value in spec.items() if key != "content_sha256"}
    )
    paths["evaluation_spec"].write_text(json.dumps(spec), encoding="utf-8")

    opened: list[Path] = []
    original = EVALUATOR._load_npz

    def guarded(path: Path, *, label: str) -> dict[str, np.ndarray]:
        opened.append(path)
        return original(path, label=label)

    monkeypatch.setattr(EVALUATOR, "_load_npz", guarded)
    with pytest.raises(EVALUATOR.LockedPrimaryEvaluationError, match="exact frozen primary"):
        _evaluate(paths)
    assert opened == []
    assert not paths["output"].exists()


def test_freeze_spec_is_create_once_and_target_independent(tmp_path: Path) -> None:
    destination = tmp_path / "pre_inference" / "spec.json"
    binding = EVALUATOR.freeze_evaluation_spec(
        destination,
        expected_rows=48,
        expected_identities=6,
        bootstrap_samples=128,
        bootstrap_seed=17,
    )
    assert binding == _binding(destination)
    document = json.loads(destination.read_text(encoding="utf-8"))
    payload = {key: value for key, value in document.items() if key != "content_sha256"}
    assert document["content_sha256"] == EVALUATOR.canonical_json_sha256(payload)
    assert document["target_values_or_target_bearing_artifacts_used_to_build_spec"] is False
    before = destination.read_bytes()
    with pytest.raises(EVALUATOR.LockedPrimaryEvaluationError, match="already exists"):
        EVALUATOR.freeze_evaluation_spec(
            destination,
            expected_rows=48,
            expected_identities=6,
            bootstrap_samples=128,
            bootstrap_seed=17,
        )
    assert destination.read_bytes() == before


def test_target_receipt_and_joined_seed_topology_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    receipt_paths = _fixture(tmp_path / "receipt")
    receipt_paths["target_receipt"].chmod(0o644)
    document = json.loads(receipt_paths["target_receipt"].read_text(encoding="utf-8"))
    document["valid_reference_rows"] = 47
    receipt_paths["target_receipt"].write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(EVALUATOR.LockedPrimaryEvaluationError, match="content_sha256"):
        _evaluate(receipt_paths)
    assert not receipt_paths["output"].exists()

    topology_paths = _fixture(tmp_path / "topology")
    with np.load(topology_paths["joined"], allow_pickle=False) as archive:
        joined = {name: np.asarray(archive[name]) for name in archive.files}
    joined["seed"] = joined["seed"].copy()
    joined["seed"][0] = 999
    topology_paths["joined"].chmod(0o644)
    with topology_paths["joined"].open("wb") as stream:
        np.savez_compressed(stream, **joined)
    topology_paths["evaluation_lock"].chmod(0o644)
    lock = json.loads(topology_paths["evaluation_lock"].read_text(encoding="utf-8"))
    lock["outputs"]["joined_oof"] = _binding(topology_paths["joined"])
    topology_paths["evaluation_lock"].write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(EVALUATOR.LockedPrimaryEvaluationError, match="row count mismatch"):
        _evaluate(topology_paths)
    assert not topology_paths["output"].exists()


def test_uncertainty_is_diagnostic_only_and_no_calibration_is_fit(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, add_uncertainty=True)
    _evaluate(paths)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    for result in report["per_seed"].values():
        diagnostics = result["uncertainty_diagnostics"]
        assert diagnostics["available"] is True
        assert diagnostics["calibration_fit_performed"] is False
        assert diagnostics["threshold_fit_performed"] is False
        field = diagnostics["fields"]["uncertainty_uncalibrated"]
        assert field["role"] == "ranking_diagnostic_only_not_a_calibrated_interval"
        assert set(field["risk_coverage"]) == {"1.0", "0.9", "0.8", "0.5"}


def test_bootstrap_is_deterministic_per_seed_and_outputs_never_overwrite(
    tmp_path: Path,
) -> None:
    first = _fixture(tmp_path / "first")
    second = _fixture(tmp_path / "second")
    _evaluate(first)
    _evaluate(second)
    first_report = json.loads(first["report"].read_text(encoding="utf-8"))
    second_report = json.loads(second["report"].read_text(encoding="utf-8"))
    for seed in first_report["per_seed"]:
        assert (
            first_report["per_seed"][seed]["identity_cluster_bootstrap"]
            == second_report["per_seed"][seed]["identity_cluster_bootstrap"]
        )

    before = {
        path: path.read_bytes() for path in (first["report"], first["csv"], first["receipt"])
    }
    with pytest.raises(EVALUATOR.LockedPrimaryEvaluationError, match="already exists"):
        _evaluate(first)
    assert all(path.read_bytes() == payload for path, payload in before.items())
