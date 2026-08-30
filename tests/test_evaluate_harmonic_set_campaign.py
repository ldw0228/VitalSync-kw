from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_harmonic_set_campaign.py"
SPEC = importlib.util.spec_from_file_location("evaluate_harmonic_set_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


SEEDS = (101, 202)
IDENTITY_TO_FOLD = {"A": 0, "B": 1, "C": 2}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Any]:
    fold_sha = "1" * 64
    rf_sha = "2" * 64
    svd_sha = "3" * 64
    contract = tmp_path / "contract.json"
    _json(
        contract,
        {
            "campaign_id": "synthetic_hcs",
            "immutable_population": {
                "fold_count": 3,
                "valid_reference_rows": 6,
                "identity_count": 3,
                "identity_to_fold": IDENTITY_TO_FOLD,
                "fold_assignments": {"sha256": fold_sha},
                "rf_cache_manifest": {"sha256": rf_sha},
                "svd_cache_manifest": {"sha256": svd_sha},
            },
            "accuracy_targets_per_seed": {
                "overall_mae_bpm_max": 1.0,
                "identity_macro_mae_bpm_max": 1.0,
                "rmse_bpm_max": 1.8,
                "within_2_fraction_min": 0.90,
                "over_5_fraction_max": 0.03,
                "high_rr_25_35_mae_bpm_max": 2.0,
                "required_seeds": list(SEEDS),
            },
        },
    )

    cache = tmp_path / "cache"
    cache.mkdir()
    metadata_rows = []
    for fold, identity in enumerate(("A", "B", "C")):
        for local, target in enumerate((10.0 + fold, 25.0 + fold)):
            metadata_rows.append(
                {
                    "cache_index": 2 * fold + local,
                    "fold": fold,
                    "identity": identity,
                    "reference_valid": True,
                    "rr_bpm": target,
                }
            )
    metadata_path = cache / "metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    cache_manifest = {
        "format_version": 1,
        "complete": True,
        "row_count": 6,
        "row_lineage_sha256": "4" * 64,
        "settings": {"merge_radius_bpm": 0.5, "maximum_candidates": 12},
        "candidate_policy": {"maximum_candidates": 12, "merge_radius_bpm": 0.5},
        "evidence_policy": {"harmonic_ratios": [0.5, 1.0, 2.0]},
        "model_boundary": {"labels_forwarded": False},
        "inputs": {
            "fold_assignments": {"sha256": fold_sha},
            "rf_root_manifest": {"sha256": rf_sha},
            "svd_root_manifest": {"sha256": svd_sha},
        },
        "outputs": {
            "metadata": {
                "filename": "metadata.csv",
                "bytes": metadata_path.stat().st_size,
                "sha256": evaluation.sha256_file(metadata_path),
            }
        },
    }
    cache_manifest["content_sha256"] = evaluation._manifest_content_sha256(
        cache_manifest
    )
    cache_manifest_path = cache / "manifest.json"
    _json(cache_manifest_path, cache_manifest)
    cache_sha = evaluation.sha256_file(cache_manifest_path)

    fallback = tmp_path / "fallback.csv"
    fallback.write_text("cache_index,prediction_bpm,rr_std_bpm\n", encoding="utf-8")
    fallback_sha = evaluation.sha256_file(fallback)
    runs: list[Path] = []
    by_key: dict[tuple[int, int], Path] = {}
    for seed in SEEDS:
        for fold in range(3):
            run = tmp_path / "runs" / f"seed_{seed}" / f"fold_{fold}"
            run.mkdir(parents=True)
            train_fold = (fold + 2) % 3
            validation_fold = (fold + 1) % 3
            train_id = next(name for name, owner in IDENTITY_TO_FOLD.items() if owner == train_fold)
            validation_id = next(
                name for name, owner in IDENTITY_TO_FOLD.items() if owner == validation_fold
            )
            test_id = next(name for name, owner in IDENTITY_TO_FOLD.items() if owner == fold)
            manifest = {
                "schema_version": 1,
                "retrospective_only": True,
                "commercial_claim_authorized": False,
                "outer_fold": fold,
                "validation_fold": validation_fold,
                "training_folds": [train_fold],
                "training_identities": [train_id],
                "validation_identities": [validation_id],
                "test_identities_declared_but_not_iterated": [test_id],
                "model_config": {"node_features": 8, "hidden_channels": 4},
                "optimization": {
                    "seed": seed,
                    "adaptive_iteration": 1,
                    "epochs": 5,
                    "learning_rate": 3e-4,
                },
                "input_bindings": {
                    "cache_manifest_path": str(cache_manifest_path),
                    "cache_manifest_sha256": cache_sha,
                    "fallback_oof_path": str(fallback),
                    "fallback_oof_sha256": fallback_sha,
                    "trainer_sha256": "5" * 64,
                },
                "leakage_boundary": {
                    "outer_test_iterator_before_atomic_lock": False,
                    "reference_and_reference_qc_are_loss_or_evaluation_only": True,
                },
            }
            run_manifest = run / "run_manifest.json"
            _json(run_manifest, manifest)
            (run / "best_checkpoint.pt").write_bytes(b"checkpoint")
            _json(run / "scaler.json", {"center": [0], "scale": [1]})
            _json(run / "fallback_policy.json", {"policy": {"pull": 0}})
            lock = {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "outer_fold": fold,
                "validation_fold": validation_fold,
                "training_folds": [train_fold],
                "adaptive_iteration": 1,
                "outer_test_not_opened_before_this_lock": True,
                "test_access_policy": "construct iterator only after atomic lock",
                "run_manifest_sha256": evaluation.sha256_file(run_manifest),
                "checkpoint_sha256": evaluation.sha256_file(run / "best_checkpoint.pt"),
                "scaler_sha256": evaluation.sha256_file(run / "scaler.json"),
                "policy_sha256": evaluation.sha256_file(run / "fallback_policy.json"),
                "cache_manifest_sha256": cache_sha,
                "fallback_oof_sha256": fallback_sha,
            }
            lock_path = run / "selection_lock.json"
            _json(lock_path, lock)
            positions = np.asarray([2 * fold, 2 * fold + 1], dtype=np.int64)
            target = np.asarray([10.0 + fold, 25.0 + fold], dtype=np.float32)
            prediction = target + np.float32(0.5)
            prediction_path = run / "test_predictions.npz"
            np.savez_compressed(
                prediction_path,
                position=positions,
                cache_index=positions,
                target_rr_bpm=target,
                identity=np.asarray([test_id, test_id]),
                final_rr_bpm=prediction,
            )
            prediction_path.chmod(0o444)
            runs.append(run)
            by_key[(seed, fold)] = run
    return {
        "contract": contract,
        "cache_manifest": cache_manifest_path,
        "runs_root": tmp_path / "runs",
        "runs": runs,
        "by_key": by_key,
    }


def _rewrite_prediction(path: Path, transform) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays = transform(arrays)
    path.chmod(0o644)
    np.savez_compressed(path, **arrays)
    path.chmod(0o444)


def _update_lock_manifest(run: Path, mutate) -> None:
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    _json(manifest_path, manifest)
    lock_path = run / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["run_manifest_sha256"] = evaluation.sha256_file(manifest_path)
    if "adaptive_iteration" in manifest.get("optimization", {}):
        lock["adaptive_iteration"] = manifest["optimization"]["adaptive_iteration"]
    _json(lock_path, lock)
    # Test evidence must remain observably after the rewritten lock.
    test = run / "test_predictions.npz"
    timestamp = lock_path.stat().st_mtime_ns + 1_000_000
    os.utime(test, ns=(timestamp, timestamp))


def _radar_files(paths: dict[str, Any]) -> dict[str, list[Path]]:
    result = {name: [] for name in evaluation.RADAR_MASKS}
    for (seed, fold), run in paths["by_key"].items():
        primary = run / "test_predictions.npz"
        with np.load(primary, allow_pickle=False) as archive:
            base = {name: np.asarray(archive[name]).copy() for name in archive.files}
        lock_sha = evaluation.sha256_file(run / "selection_lock.json")
        lock = json.loads((run / "selection_lock.json").read_text(encoding="utf-8"))
        for condition, mask in evaluation.RADAR_MASKS.items():
            output = run / "radar_masks" / condition / "test_predictions.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                cache_index=base["cache_index"],
                target_rr_bpm=base["target_rr_bpm"],
                identity=base["identity"],
                final_rr_bpm=base["final_rr_bpm"],
                radar_mask=np.asarray(mask, dtype=bool),
                seed=np.asarray(seed),
                outer_fold=np.asarray(fold),
                adaptive_iteration=np.asarray(lock["adaptive_iteration"]),
                selection_lock_sha256=np.asarray(lock_sha),
                cache_manifest_sha256=np.asarray(lock["cache_manifest_sha256"]),
            )
            result[condition].append(output)
    return result


def test_complete_campaign_writes_metrics_bootstrap_stability_and_provenance(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    radar = _radar_files(paths)
    output = tmp_path / "evaluation"
    report = evaluation.evaluate_campaign(
        [paths["runs_root"]],
        output_dir=output,
        spec=evaluation.load_evaluation_spec(paths["contract"]),
        bootstrap_samples=80,
        bootstrap_seed=9,
        radar_mask_inputs=radar,
    )
    assert report["acceptance"]["campaign_engineering_pass"] is True
    assert set(report["radar_masks"]["conditions"]) == set(evaluation.RADAR_MASKS)
    assert report["primary"]["seed_stability"]["seed_count"] == 2
    assert (
        report["primary"]["seed_stability"]["row_prediction_standard_deviation_bpm"][
            "maximum"
        ]
        == 0.0
    )
    seed_summary = report["primary"]["by_seed"][str(SEEDS[0])]
    assert seed_summary["metrics"]["mae"] == pytest.approx(0.5)
    assert seed_summary["metrics"]["identity_macro_rmse"] == pytest.approx(0.5)
    assert seed_summary["identity_bootstrap_ci"]["mae"]["bootstrap_unit"] == "physical_identity"
    for name in (
        "evaluation.json",
        "metrics.csv",
        "per_identity.csv",
        "provenance.json",
        "provenance.csv",
        "provenance.md",
        "evaluation_manifest.json",
    ):
        assert (output / name).is_file()
    artifact_manifest = json.loads((output / "evaluation_manifest.json").read_text())
    for name, binding in artifact_manifest["artifacts"].items():
        assert evaluation.sha256_file(output / name) == binding["sha256"]
    provenance = json.loads((output / "provenance.json").read_text())
    assert len(provenance["runs"]) == len(SEEDS) * 3
    assert len(provenance["radar_mask_artifacts"]) == len(SEEDS) * 3 * 7


def test_missing_and_duplicate_outer_test_rows_fail_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    prediction = paths["by_key"][(SEEDS[0], 0)] / "test_predictions.npz"
    _rewrite_prediction(
        prediction,
        lambda arrays: {name: value[:-1] for name, value in arrays.items()},
    )
    with pytest.raises(RuntimeError, match="exactly cover"):
        evaluation.evaluate_campaign(
            [paths["runs_root"]],
            output_dir=tmp_path / "out",
            spec=evaluation.load_evaluation_spec(paths["contract"]),
            bootstrap_samples=10,
        )

    paths = _fixture(tmp_path / "duplicate")
    prediction = paths["by_key"][(SEEDS[0], 0)] / "test_predictions.npz"
    _rewrite_prediction(
        prediction,
        lambda arrays: {
            name: np.concatenate([value, value[:1]]) for name, value in arrays.items()
        },
    )
    with pytest.raises(RuntimeError, match="duplicate rows"):
        evaluation.evaluate_campaign(
            [paths["runs_root"]],
            output_dir=tmp_path / "out2",
            spec=evaluation.load_evaluation_spec(paths["contract"]),
            bootstrap_samples=10,
        )


def test_test_before_lock_evidence_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    run = paths["by_key"][(SEEDS[0], 0)]
    lock_path = run / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["outer_test_not_opened_before_this_lock"] = False
    _json(lock_path, lock)
    with pytest.raises(RuntimeError, match="test-before-lock"):
        evaluation.evaluate_campaign(
            [paths["runs_root"]],
            output_dir=tmp_path / "out",
            spec=evaluation.load_evaluation_spec(paths["contract"]),
            bootstrap_samples=10,
        )


def test_adaptive_iteration_mixing_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    run = paths["by_key"][(SEEDS[0], 0)]
    _update_lock_manifest(
        run,
        lambda manifest: manifest["optimization"].update(adaptive_iteration=2),
    )
    with pytest.raises(RuntimeError, match="adaptive result mixing"):
        evaluation.evaluate_campaign(
            [paths["runs_root"]],
            output_dir=tmp_path / "out",
            spec=evaluation.load_evaluation_spec(paths["contract"]),
            bootstrap_samples=10,
        )


def test_identity_leakage_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    run = paths["by_key"][(SEEDS[0], 0)]

    def leak(manifest: dict[str, Any]) -> None:
        manifest["training_identities"].append("A")

    _update_lock_manifest(run, leak)
    with pytest.raises(RuntimeError, match="identity leakage"):
        evaluation.evaluate_campaign(
            [paths["runs_root"]],
            output_dir=tmp_path / "out",
            spec=evaluation.load_evaluation_spec(paths["contract"]),
            bootstrap_samples=10,
        )


def test_inconsistent_cache_and_manifest_hashes_are_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    run = paths["by_key"][(SEEDS[0], 0)]
    lock_path = run / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["cache_manifest_sha256"] = "0" * 64
    _json(lock_path, lock)
    timestamp = lock_path.stat().st_mtime_ns + 1_000_000
    os.utime(run / "test_predictions.npz", ns=(timestamp, timestamp))
    with pytest.raises(RuntimeError, match="cache-manifest hashes are inconsistent"):
        evaluation.evaluate_campaign(
            [paths["runs_root"]],
            output_dir=tmp_path / "out",
            spec=evaluation.load_evaluation_spec(paths["contract"]),
            bootstrap_samples=10,
        )


def test_partial_or_misdeclared_radar_masks_fail_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    radar = _radar_files(paths)
    radar.pop("radar_3")
    with pytest.raises(RuntimeError, match="all seven"):
        evaluation.evaluate_campaign(
            [paths["runs_root"]],
            output_dir=tmp_path / "out",
            spec=evaluation.load_evaluation_spec(paths["contract"]),
            bootstrap_samples=10,
            radar_mask_inputs=radar,
        )

    paths = _fixture(tmp_path / "wrong_mask")
    radar = _radar_files(paths)
    first = radar["radar_1"][0]
    with np.load(first, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["radar_mask"] = np.asarray([False, True, False])
    np.savez_compressed(first, **arrays)
    with pytest.raises(RuntimeError, match="declared condition"):
        evaluation.evaluate_campaign(
            [paths["runs_root"]],
            output_dir=tmp_path / "out2",
            spec=evaluation.load_evaluation_spec(paths["contract"]),
            bootstrap_samples=10,
            radar_mask_inputs=radar,
        )
