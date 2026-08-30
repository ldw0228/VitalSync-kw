#!/usr/bin/env python3
"""Fail-closed evaluator for locked Harmonic Candidate-Set SNN campaigns.

The evaluator is intentionally not a model-selection utility.  It accepts only
outer-test predictions produced after an immutable selection lock, verifies the
identity-disjoint population and artifact bindings, and then computes a fixed
set of acceptance metrics.  Seed or radar-mask specific thresholds are never
searched from test labels.

Radar-mask files use the same row arrays as ``test_predictions.npz`` and must
also contain scalar ``seed``, ``outer_fold`` (or ``fold``),
``adaptive_iteration``, ``selection_lock_sha256`` and
``cache_manifest_sha256`` values plus a three-element ``radar_mask``.  This
explicit binding prevents a filename from being treated as evidence that a
particular sensor mask was actually evaluated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/ADAPTIVE_CAMPAIGN_CONTRACT.json"
)

RADAR_MASKS: dict[str, tuple[bool, bool, bool]] = {
    "radars_123": (True, True, True),
    "radars_12": (True, True, False),
    "radars_13": (True, False, True),
    "radars_23": (False, True, True),
    "radar_1": (True, False, False),
    "radar_2": (False, True, False),
    "radar_3": (False, False, True),
}

GATE_NAMES = (
    "overall_mae",
    "identity_macro_mae",
    "overall_rmse",
    "within_2_fraction",
    "over_5_fraction",
    "tail_25_35_mae",
)

METRIC_NAMES = (
    "mae",
    "rmse",
    "within_2_fraction",
    "over_5_fraction",
    "tail_25_35_mae",
    "identity_macro_mae",
    "identity_macro_rmse",
    "identity_macro_within_2_fraction",
    "identity_macro_over_5_fraction",
    "identity_macro_tail_25_35_mae",
)


@dataclass(frozen=True)
class EvaluationSpec:
    """Immutable population and targets parsed from the campaign contract."""

    campaign_id: str
    contract_path: Path
    contract_sha256: str
    fold_count: int
    valid_reference_rows: int
    identity_count: int
    required_seeds: tuple[int, ...]
    identity_to_fold: Mapping[str, int]
    fold_assignments_sha256: str
    rf_manifest_sha256: str
    svd_manifest_sha256: str
    mae_max: float
    identity_macro_mae_max: float
    rmse_max: float
    within_2_min: float
    over_5_max: float
    tail_25_35_mae_max: float


@dataclass
class RunRecord:
    run_dir: Path
    seed: int
    fold: int
    adaptive_iteration: int
    frame: pd.DataFrame
    selection_lock_path: Path
    selection_lock_sha256: str
    cache_manifest_path: Path
    cache_manifest_sha256: str
    test_predictions_path: Path
    test_predictions_sha256: str
    design_signature_sha256: str
    row_lineage_sha256: str
    provenance: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _strict_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_content_sha256(document: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in document.items() if key != "content_sha256"}
    )


def _strict_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("output contains a non-finite floating-point value")
        return result
    if value is pd.NA:
        return None
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _resolve_recorded_path(value: Any, *, relative_to: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise TypeError("artifact binding path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def load_evaluation_spec(path: Path = DEFAULT_CONTRACT) -> EvaluationSpec:
    path = path.expanduser().resolve()
    document = _load_json(path, "campaign contract")
    population = document.get("immutable_population")
    targets = document.get("accuracy_targets_per_seed")
    if not isinstance(population, Mapping) or not isinstance(targets, Mapping):
        raise RuntimeError("campaign contract lacks population or per-seed targets")
    identity_to_fold_raw = population.get("identity_to_fold")
    if not isinstance(identity_to_fold_raw, Mapping):
        raise RuntimeError("campaign contract lacks identity_to_fold")
    identity_to_fold = {
        str(identity): int(fold) for identity, fold in identity_to_fold_raw.items()
    }
    fold_count = int(population["fold_count"])
    if len(identity_to_fold) != int(population["identity_count"]):
        raise RuntimeError("contract identity mapping/count are inconsistent")
    if set(identity_to_fold.values()) != set(range(fold_count)):
        raise RuntimeError("contract identity mapping does not cover every outer fold")
    required_seeds = tuple(int(seed) for seed in targets.get("required_seeds", ()))
    if not required_seeds or len(set(required_seeds)) != len(required_seeds):
        raise RuntimeError("contract required_seeds must be non-empty and unique")
    fold_binding = population.get("fold_assignments", {})
    rf_binding = population.get("rf_cache_manifest", {})
    svd_binding = population.get("svd_cache_manifest", {})
    try:
        return EvaluationSpec(
            campaign_id=str(document["campaign_id"]),
            contract_path=path,
            contract_sha256=sha256_file(path),
            fold_count=fold_count,
            valid_reference_rows=int(population["valid_reference_rows"]),
            identity_count=int(population["identity_count"]),
            required_seeds=required_seeds,
            identity_to_fold=identity_to_fold,
            fold_assignments_sha256=str(fold_binding["sha256"]),
            rf_manifest_sha256=str(rf_binding["sha256"]),
            svd_manifest_sha256=str(svd_binding["sha256"]),
            mae_max=float(targets["overall_mae_bpm_max"]),
            identity_macro_mae_max=float(
                targets["identity_macro_mae_bpm_max"]
            ),
            rmse_max=float(targets["rmse_bpm_max"]),
            within_2_min=float(targets["within_2_fraction_min"]),
            over_5_max=float(targets["over_5_fraction_max"]),
            tail_25_35_mae_max=float(targets["high_rr_25_35_mae_bpm_max"]),
        )
    except KeyError as error:
        raise RuntimeError(f"campaign contract is missing {error.args[0]}") from error


def _as_bool(values: pd.Series, *, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    allowed = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(allowed).all():
        raise RuntimeError(f"{label} contains non-boolean values")
    return normalized.map(allowed).to_numpy(dtype=bool)


def _validate_file_binding(
    path: Path, binding: Mapping[str, Any], *, label: str
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"bound {label} is missing: {path}")
    expected_bytes = binding.get("bytes")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise RuntimeError(f"{label} byte-size differs from its manifest binding")
    actual = sha256_file(path)
    if actual != str(binding.get("sha256", "")):
        raise RuntimeError(f"{label} hash differs from its manifest binding")
    return actual


def _validate_cache_manifest(
    path: Path,
    *,
    expected_sha256: str,
    spec: EvaluationSpec,
    memo: dict[Path, tuple[dict[str, Any], pd.DataFrame]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    path = path.resolve()
    if path in memo:
        return memo[path]
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("cache manifest hash is inconsistent with the run binding")
    manifest = _load_json(path, "HCS cache manifest")
    if not bool(manifest.get("complete")):
        raise RuntimeError("HCS cache manifest is incomplete")
    if str(manifest.get("content_sha256", "")) != _manifest_content_sha256(manifest):
        raise RuntimeError("HCS cache manifest canonical content hash is inconsistent")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise RuntimeError("HCS cache manifest lacks input/output bindings")
    fold_binding = inputs.get("fold_assignments")
    rf_binding = inputs.get("rf_root_manifest")
    svd_binding = inputs.get("svd_root_manifest")
    if not all(isinstance(item, Mapping) for item in (fold_binding, rf_binding, svd_binding)):
        raise RuntimeError("HCS cache lacks canonical split/RF/SVD bindings")
    if str(fold_binding.get("sha256")) != spec.fold_assignments_sha256:
        raise RuntimeError("cache fold-assignment hash differs from the contract")
    if str(rf_binding.get("sha256")) != spec.rf_manifest_sha256:
        raise RuntimeError("cache RF manifest hash differs from the contract")
    if str(svd_binding.get("sha256")) != spec.svd_manifest_sha256:
        raise RuntimeError("cache SVD manifest hash differs from the contract")
    metadata_binding = outputs.get("metadata")
    if not isinstance(metadata_binding, Mapping):
        raise RuntimeError("HCS cache manifest has no metadata output binding")
    metadata_path = path.parent / str(metadata_binding.get("filename", ""))
    _validate_file_binding(metadata_path, metadata_binding, label="cache metadata")
    metadata = pd.read_csv(metadata_path)
    required = {
        "cache_index",
        "fold",
        "identity",
        "reference_valid",
        "rr_bpm",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise RuntimeError(f"cache metadata is missing columns: {missing}")
    if len(metadata) != int(manifest.get("row_count", -1)):
        raise RuntimeError("cache metadata row count differs from its manifest")
    index = pd.to_numeric(metadata["cache_index"], errors="raise").to_numpy(np.int64)
    if len(np.unique(index)) != len(index):
        raise RuntimeError("cache metadata contains duplicate cache indices")
    if not np.array_equal(index, np.arange(len(index), dtype=np.int64)):
        raise RuntimeError("cache metadata is not in canonical cache-index order")
    folds = pd.to_numeric(metadata["fold"], errors="raise").to_numpy(np.int64)
    identities = metadata["identity"].astype(str).to_numpy()
    for identity, fold in zip(identities, folds, strict=True):
        if spec.identity_to_fold.get(identity) != int(fold):
            raise RuntimeError("cache identity/fold ownership differs from the contract")
    valid = _as_bool(metadata["reference_valid"], label="reference_valid")
    target = pd.to_numeric(metadata["rr_bpm"], errors="coerce").to_numpy(float)
    if int(valid.sum()) != spec.valid_reference_rows or not np.isfinite(target[valid]).all():
        raise RuntimeError("cache valid-reference population differs from the contract")
    if len(set(identities[valid])) != spec.identity_count:
        raise RuntimeError("cache valid-reference identity count differs from the contract")
    memo[path] = manifest, metadata
    return manifest, metadata


def _require_hash(lock: Mapping[str, Any], key: str, path: Path, label: str) -> str:
    expected = str(lock.get(key, ""))
    actual = sha256_file(path)
    if not expected or expected != actual:
        raise RuntimeError(f"selection-lock hash mismatch for {label}")
    return actual


def _load_prediction_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"cache_index", "target_rr_bpm", "identity", "final_rr_bpm"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"prediction archive is missing arrays: {missing}")
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _scalar(array: np.ndarray, *, label: str) -> Any:
    value = np.asarray(array)
    if value.size != 1:
        raise RuntimeError(f"{label} must be a scalar")
    return value.reshape(()).item()


def _prediction_frame(
    arrays: Mapping[str, np.ndarray],
    *,
    path: Path,
    expected: pd.DataFrame,
    prediction_key: str = "final_rr_bpm",
) -> pd.DataFrame:
    required = ("cache_index", "target_rr_bpm", "identity", prediction_key)
    lengths = {name: len(np.asarray(arrays[name]).reshape(-1)) for name in required}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"prediction arrays have inconsistent lengths: {path}")
    index = np.asarray(arrays["cache_index"]).reshape(-1).astype(np.int64)
    if len(index) == 0:
        raise RuntimeError(f"prediction archive has no rows: {path}")
    if len(np.unique(index)) != len(index):
        raise RuntimeError(f"prediction archive contains duplicate rows: {path}")
    order = np.argsort(index, kind="stable")
    index = index[order]
    expected_ordered = expected.sort_values("cache_index", kind="stable")
    expected_index = expected_ordered["cache_index"].to_numpy(np.int64)
    if not np.array_equal(index, expected_index):
        missing = np.setdiff1d(expected_index, index, assume_unique=True)
        extra = np.setdiff1d(index, expected_index, assume_unique=True)
        raise RuntimeError(
            "prediction rows do not exactly cover the locked outer-test population "
            f"(missing={missing[:8].tolist()}, extra={extra[:8].tolist()})"
        )
    identity = np.asarray(arrays["identity"]).reshape(-1).astype(str)[order]
    expected_identity = expected_ordered["identity"].astype(str).to_numpy()
    if not np.array_equal(identity, expected_identity):
        raise RuntimeError("prediction identity lineage differs from cache metadata")
    target = np.asarray(arrays["target_rr_bpm"]).reshape(-1).astype(float)[order]
    expected_target = pd.to_numeric(
        expected_ordered["rr_bpm"], errors="raise"
    ).to_numpy(float)
    if not np.isfinite(target).all() or not np.allclose(
        target, expected_target, rtol=1.0e-6, atol=2.0e-5
    ):
        raise RuntimeError("prediction targets differ from canonical cache metadata")
    prediction = np.asarray(arrays[prediction_key]).reshape(-1).astype(float)[order]
    if not np.isfinite(prediction).all():
        raise RuntimeError("prediction archive contains non-finite predictions")
    return pd.DataFrame(
        {
            "cache_index": index,
            "identity": identity,
            "target_rr_bpm": target,
            "prediction_rr_bpm": prediction,
        }
    )


def _expected_identity_partition(
    spec: EvaluationSpec, fold: int
) -> tuple[set[str], set[str], set[str], set[int]]:
    validation_fold = (fold + 1) % spec.fold_count
    test = {
        identity for identity, owner in spec.identity_to_fold.items() if owner == fold
    }
    validation = {
        identity
        for identity, owner in spec.identity_to_fold.items()
        if owner == validation_fold
    }
    train = set(spec.identity_to_fold) - test - validation
    training_folds = set(range(spec.fold_count)) - {fold, validation_fold}
    return train, validation, test, training_folds


def validate_run(
    run_dir: Path,
    *,
    spec: EvaluationSpec,
    cache_memo: dict[Path, tuple[dict[str, Any], pd.DataFrame]],
) -> RunRecord:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    lock_path = run_dir / "selection_lock.json"
    prediction_path = run_dir / "test_predictions.npz"
    manifest = _load_json(manifest_path, "run manifest")
    lock = _load_json(lock_path, "selection lock")
    if not prediction_path.is_file():
        raise FileNotFoundError(f"locked run has no outer-test predictions: {run_dir}")
    if not bool(lock.get("outer_test_not_opened_before_this_lock")):
        raise RuntimeError("selection lock does not attest test-before-lock exclusion")
    if lock.get("test_access_policy") != "construct iterator only after atomic lock":
        raise RuntimeError("selection lock lacks the contracted test-access policy")
    leakage = manifest.get("leakage_boundary")
    if not isinstance(leakage, Mapping) or leakage.get(
        "outer_test_iterator_before_atomic_lock"
    ) is not False:
        raise RuntimeError("run manifest contains test-before-lock evidence")
    if lock_path.stat().st_mtime_ns > prediction_path.stat().st_mtime_ns:
        raise RuntimeError("outer-test prediction predates the atomic selection lock")
    try:
        lock_created = datetime.fromisoformat(str(lock["created_utc"])).timestamp()
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("selection lock lacks a valid creation timestamp") from error
    if prediction_path.stat().st_mtime + 1.0e-6 < lock_created:
        raise RuntimeError("outer-test prediction predates the declared lock creation time")
    if prediction_path.stat().st_mode & 0o222:
        raise RuntimeError("outer-test prediction is mutable; immutable evidence required")

    run_manifest_sha = _require_hash(
        lock, "run_manifest_sha256", manifest_path, "run manifest"
    )
    checkpoint_sha = _require_hash(
        lock, "checkpoint_sha256", run_dir / "best_checkpoint.pt", "checkpoint"
    )
    scaler_sha = _require_hash(lock, "scaler_sha256", run_dir / "scaler.json", "scaler")
    policy_sha = _require_hash(
        lock, "policy_sha256", run_dir / "fallback_policy.json", "fallback policy"
    )

    bindings = manifest.get("input_bindings")
    optimization = manifest.get("optimization")
    if not isinstance(bindings, Mapping) or not isinstance(optimization, Mapping):
        raise RuntimeError("run manifest lacks optimization/input bindings")
    cache_path = _resolve_recorded_path(
        bindings.get("cache_manifest_path"), relative_to=run_dir
    )
    cache_sha = sha256_file(cache_path)
    if cache_sha != str(bindings.get("cache_manifest_sha256", "")):
        raise RuntimeError("run-manifest cache hash is inconsistent with the file")
    if cache_sha != str(lock.get("cache_manifest_sha256", "")):
        raise RuntimeError("selection-lock/cache-manifest hashes are inconsistent")
    cache_manifest, cache_metadata = _validate_cache_manifest(
        cache_path, expected_sha256=cache_sha, spec=spec, memo=cache_memo
    )
    fallback_path = _resolve_recorded_path(
        bindings.get("fallback_oof_path"), relative_to=run_dir
    )
    fallback_sha = sha256_file(fallback_path)
    if fallback_sha != str(bindings.get("fallback_oof_sha256", "")):
        raise RuntimeError("run-manifest fallback hash is inconsistent with the file")
    if fallback_sha != str(lock.get("fallback_oof_sha256", "")):
        raise RuntimeError("selection-lock/fallback hashes are inconsistent")

    seed = int(lock.get("seed", -1))
    fold = int(lock.get("outer_fold", -1))
    adaptive_iteration = int(lock.get("adaptive_iteration", -1))
    if seed != int(optimization.get("seed", -2)):
        raise RuntimeError("selection lock and run manifest disagree on seed")
    if fold != int(manifest.get("outer_fold", -2)) or not 0 <= fold < spec.fold_count:
        raise RuntimeError("selection lock and run manifest disagree on outer fold")
    if adaptive_iteration != int(optimization.get("adaptive_iteration", -2)):
        raise RuntimeError("selection lock and run manifest disagree on adaptive iteration")
    if adaptive_iteration not in (1, 2, 3):
        raise RuntimeError("adaptive iteration is outside the contracted family")
    train_ids, validation_ids, test_ids, training_folds = _expected_identity_partition(
        spec, fold
    )
    observed_train = set(map(str, manifest.get("training_identities", ())))
    observed_validation = set(map(str, manifest.get("validation_identities", ())))
    observed_test = set(
        map(str, manifest.get("test_identities_declared_but_not_iterated", ()))
    )
    if any(
        (
            observed_train & observed_validation,
            observed_train & observed_test,
            observed_validation & observed_test,
        )
    ):
        raise RuntimeError("identity leakage across train/validation/outer-test partitions")
    if (
        observed_train != train_ids
        or observed_validation != validation_ids
        or observed_test != test_ids
    ):
        raise RuntimeError("run identity partitions differ from the campaign contract")
    if int(manifest.get("validation_fold", -1)) != (fold + 1) % spec.fold_count:
        raise RuntimeError("validation fold differs from the contracted rotation")
    if set(map(int, manifest.get("training_folds", ()))) != training_folds:
        raise RuntimeError("training folds differ from the contracted outer split")
    if set(map(int, lock.get("training_folds", ()))) != training_folds:
        raise RuntimeError("selection-lock training folds are inconsistent")

    cache_valid = _as_bool(
        cache_metadata["reference_valid"], label="reference_valid"
    )
    cache_fold = pd.to_numeric(cache_metadata["fold"], errors="raise").to_numpy(
        np.int64
    )
    expected = cache_metadata.loc[cache_valid & (cache_fold == fold)].copy()
    if set(expected["identity"].astype(str)) != test_ids:
        raise RuntimeError("cache outer-test identities differ from the contract")
    arrays = _load_prediction_npz(prediction_path)
    frame = _prediction_frame(arrays, path=prediction_path, expected=expected)
    if "position" in arrays:
        order = np.argsort(np.asarray(arrays["cache_index"]).reshape(-1), kind="stable")
        position = np.asarray(arrays["position"]).reshape(-1).astype(np.int64)[order]
        if np.any(position < 0) or np.any(position >= len(cache_metadata)):
            raise RuntimeError("prediction position is outside cache metadata")
        positional_index = cache_metadata.iloc[position]["cache_index"].to_numpy(np.int64)
        if not np.array_equal(positional_index, frame["cache_index"].to_numpy(np.int64)):
            raise RuntimeError("prediction position/cache-index lineage is inconsistent")
    if set(frame["identity"]) != test_ids:
        raise RuntimeError("outer-test predictions contain train/validation identities")
    frame.insert(0, "fold", fold)
    frame.insert(0, "seed", seed)

    design_payload = {
        "adaptive_iteration": adaptive_iteration,
        "model_config": manifest.get("model_config"),
        "optimization": {
            key: value for key, value in optimization.items() if key != "seed"
        },
        "trainer_sha256": bindings.get("trainer_sha256"),
        "cache_settings": cache_manifest.get("settings"),
        "candidate_policy": cache_manifest.get("candidate_policy"),
        "evidence_policy": cache_manifest.get("evidence_policy"),
        "model_boundary": cache_manifest.get("model_boundary"),
    }
    design_signature = _canonical_sha256(design_payload)
    row_lineage = str(cache_manifest.get("row_lineage_sha256", ""))
    if len(row_lineage) != 64:
        raise RuntimeError("cache manifest lacks a valid row-lineage hash")
    lock_sha = sha256_file(lock_path)
    prediction_sha = sha256_file(prediction_path)
    provenance = {
        "run_dir": str(run_dir),
        "seed": seed,
        "outer_fold": fold,
        "adaptive_iteration": adaptive_iteration,
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": run_manifest_sha,
        "selection_lock": str(lock_path),
        "selection_lock_sha256": lock_sha,
        "checkpoint_sha256": checkpoint_sha,
        "scaler_sha256": scaler_sha,
        "policy_sha256": policy_sha,
        "cache_manifest": str(cache_path),
        "cache_manifest_sha256": cache_sha,
        "cache_row_lineage_sha256": row_lineage,
        "cache_metadata_sha256": str(cache_manifest["outputs"]["metadata"]["sha256"]),
        "fallback_oof": str(fallback_path),
        "fallback_oof_sha256": fallback_sha,
        "test_predictions": str(prediction_path),
        "test_predictions_sha256": prediction_sha,
        "test_rows": len(frame),
        "test_identities": sorted(test_ids),
        "design_signature_sha256": design_signature,
        "test_after_lock": True,
        "identity_disjoint": True,
        "hash_bindings_verified": True,
    }
    return RunRecord(
        run_dir=run_dir,
        seed=seed,
        fold=fold,
        adaptive_iteration=adaptive_iteration,
        frame=frame,
        selection_lock_path=lock_path,
        selection_lock_sha256=lock_sha,
        cache_manifest_path=cache_path,
        cache_manifest_sha256=cache_sha,
        test_predictions_path=prediction_path,
        test_predictions_sha256=prediction_sha,
        design_signature_sha256=design_signature,
        row_lineage_sha256=row_lineage,
        provenance=provenance,
    )


def _expand_run_dirs(inputs: Sequence[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for raw in inputs:
        path = raw.expanduser().resolve()
        if (path / "selection_lock.json").is_file():
            discovered.add(path)
        elif path.is_dir():
            discovered.update(item.parent.resolve() for item in path.rglob("selection_lock.json"))
        else:
            raise FileNotFoundError(f"run input does not exist: {path}")
    if not discovered:
        raise RuntimeError("no locked run directories were discovered")
    return sorted(discovered)


def _metric_summary(target: np.ndarray, prediction: np.ndarray, identity: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    identity = np.asarray(identity).astype(str)
    if not (len(target) == len(prediction) == len(identity)) or len(target) == 0:
        raise ValueError("metric arrays must be non-empty and equal length")
    error = np.abs(prediction - target)
    squared = np.square(prediction - target)
    tail = (target >= 25.0) & (target <= 35.0)
    identity_rows: list[dict[str, Any]] = []
    for name in sorted(set(identity)):
        selected = identity == name
        local_error = error[selected]
        local_squared = squared[selected]
        local_tail = tail[selected]
        identity_rows.append(
            {
                "identity": name,
                "rows": int(selected.sum()),
                "mae": float(local_error.mean()),
                "rmse": float(np.sqrt(local_squared.mean())),
                "within_2_fraction": float(np.mean(local_error <= 2.0)),
                "over_5_fraction": float(np.mean(local_error > 5.0)),
                "tail_25_35_rows": int(local_tail.sum()),
                "tail_25_35_mae": (
                    float(local_error[local_tail].mean()) if local_tail.any() else None
                ),
            }
        )
    per_identity = pd.DataFrame(identity_rows)
    tail_identity = per_identity["tail_25_35_rows"] > 0
    metrics = {
        "rows": int(len(target)),
        "identities": int(len(per_identity)),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(squared.mean())),
        "within_2_fraction": float(np.mean(error <= 2.0)),
        "over_5_fraction": float(np.mean(error > 5.0)),
        "tail_25_35_rows": int(tail.sum()),
        "tail_25_35_mae": float(error[tail].mean()) if tail.any() else None,
        "identity_macro_mae": float(per_identity["mae"].mean()),
        "identity_macro_rmse": float(per_identity["rmse"].mean()),
        "identity_macro_within_2_fraction": float(
            per_identity["within_2_fraction"].mean()
        ),
        "identity_macro_over_5_fraction": float(per_identity["over_5_fraction"].mean()),
        "identity_macro_tail_25_35_mae": (
            float(per_identity.loc[tail_identity, "tail_25_35_mae"].mean())
            if tail_identity.any()
            else None
        ),
    }
    return metrics, per_identity


def identity_cluster_bootstrap(
    target: np.ndarray,
    prediction: np.ndarray,
    identity: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Cluster bootstrap confidence intervals with physical identity as unit."""

    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie between zero and one")
    target = np.asarray(target, float)
    prediction = np.asarray(prediction, float)
    identity = np.asarray(identity).astype(str)
    estimate, per_identity = _metric_summary(target, prediction, identity)
    names = per_identity["identity"].astype(str).tolist()
    n_identities = len(names)
    if n_identities < 2:
        raise RuntimeError("identity bootstrap requires at least two identities")
    cluster = []
    error = np.abs(prediction - target)
    squared = np.square(prediction - target)
    tail = (target >= 25.0) & (target <= 35.0)
    for name in names:
        selected = identity == name
        cluster.append(
            (
                int(selected.sum()),
                float(error[selected].sum()),
                float(squared[selected].sum()),
                int(np.sum(error[selected] <= 2.0)),
                int(np.sum(error[selected] > 5.0)),
                int(np.sum(selected & tail)),
                float(error[selected & tail].sum()),
            )
        )
    values = np.asarray(cluster, dtype=float)
    rng = np.random.default_rng(int(seed))
    weights = rng.multinomial(
        n_identities,
        np.full(n_identities, 1.0 / n_identities),
        size=int(samples),
    ).astype(float)
    row_count = weights @ values[:, 0]
    abs_sum = weights @ values[:, 1]
    squared_sum = weights @ values[:, 2]
    within_count = weights @ values[:, 3]
    over_count = weights @ values[:, 4]
    tail_count = weights @ values[:, 5]
    tail_abs_sum = weights @ values[:, 6]
    per_tail = pd.to_numeric(per_identity["tail_25_35_mae"], errors="coerce").to_numpy(float)
    tail_capable = np.isfinite(per_tail)
    sampled: dict[str, np.ndarray] = {
        "mae": abs_sum / row_count,
        "rmse": np.sqrt(squared_sum / row_count),
        "within_2_fraction": within_count / row_count,
        "over_5_fraction": over_count / row_count,
        "tail_25_35_mae": np.divide(
            tail_abs_sum,
            tail_count,
            out=np.full(samples, np.nan, dtype=float),
            where=tail_count > 0,
        ),
        "identity_macro_mae": weights @ per_identity["mae"].to_numpy(float) / n_identities,
        "identity_macro_rmse": weights @ per_identity["rmse"].to_numpy(float) / n_identities,
        "identity_macro_within_2_fraction": weights
        @ per_identity["within_2_fraction"].to_numpy(float)
        / n_identities,
        "identity_macro_over_5_fraction": weights
        @ per_identity["over_5_fraction"].to_numpy(float)
        / n_identities,
    }
    macro_tail_n = weights[:, tail_capable].sum(axis=1)
    sampled["identity_macro_tail_25_35_mae"] = np.divide(
        weights[:, tail_capable] @ per_tail[tail_capable],
        macro_tail_n,
        out=np.full(samples, np.nan, dtype=float),
        where=macro_tail_n > 0,
    )
    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, Any] = {}
    for name in METRIC_NAMES:
        finite = sampled[name][np.isfinite(sampled[name])]
        point = estimate[name]
        if point is None or len(finite) == 0:
            intervals[name] = None
            continue
        lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
        intervals[name] = {
            "estimate": float(point),
            "lower": float(lower),
            "upper": float(upper),
            "confidence": float(confidence),
            "bootstrap_unit": "physical_identity",
            "samples_requested": int(samples),
            "samples_finite": int(len(finite)),
        }
    return intervals


def gate_decision(metrics: Mapping[str, Any], spec: EvaluationSpec) -> dict[str, Any]:
    values = {
        "overall_mae": {
            "value": metrics["mae"], "operator": "<=", "target": spec.mae_max,
            "passed": metrics["mae"] <= spec.mae_max,
        },
        "identity_macro_mae": {
            "value": metrics["identity_macro_mae"], "operator": "<=",
            "target": spec.identity_macro_mae_max,
            "passed": metrics["identity_macro_mae"] <= spec.identity_macro_mae_max,
        },
        "overall_rmse": {
            "value": metrics["rmse"], "operator": "<=", "target": spec.rmse_max,
            "passed": metrics["rmse"] <= spec.rmse_max,
        },
        "within_2_fraction": {
            "value": metrics["within_2_fraction"], "operator": ">=",
            "target": spec.within_2_min,
            "passed": metrics["within_2_fraction"] >= spec.within_2_min,
        },
        "over_5_fraction": {
            "value": metrics["over_5_fraction"], "operator": "<=",
            "target": spec.over_5_max,
            "passed": metrics["over_5_fraction"] <= spec.over_5_max,
        },
        "tail_25_35_mae": {
            "value": metrics["tail_25_35_mae"], "operator": "<=",
            "target": spec.tail_25_35_mae_max,
            "passed": metrics["tail_25_35_mae"] is not None
            and metrics["tail_25_35_mae"] <= spec.tail_25_35_mae_max,
        },
    }
    return {"all_passed": bool(all(item["passed"] for item in values.values())), "gates": values}


def _bootstrap_seed(base: int, condition: str, seed: int) -> int:
    digest = hashlib.sha256(f"{base}:{condition}:{seed}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _summarize_seed_frames(
    frames: Mapping[int, pd.DataFrame],
    *,
    condition: str,
    spec: EvaluationSpec,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for seed in sorted(frames):
        frame = frames[seed].sort_values("cache_index", kind="stable").reset_index(drop=True)
        metrics, per_identity = _metric_summary(
            frame["target_rr_bpm"].to_numpy(float),
            frame["prediction_rr_bpm"].to_numpy(float),
            frame["identity"].astype(str).to_numpy(),
        )
        intervals = identity_cluster_bootstrap(
            frame["target_rr_bpm"].to_numpy(float),
            frame["prediction_rr_bpm"].to_numpy(float),
            frame["identity"].astype(str).to_numpy(),
            samples=bootstrap_samples,
            seed=_bootstrap_seed(bootstrap_seed, condition, seed),
        )
        gates = gate_decision(metrics, spec)
        by_seed[str(seed)] = {
            "metrics": metrics,
            "identity_bootstrap_ci": intervals,
            "acceptance": gates,
        }
        metric_rows.append(
            {
                "condition": condition,
                "seed": seed,
                **metrics,
                "all_gates_passed": gates["all_passed"],
            }
        )
        for row in per_identity.to_dict(orient="records"):
            identity_rows.append({"condition": condition, "seed": seed, **row})
    return by_seed, metric_rows, identity_rows


def seed_stability(frames: Mapping[int, pd.DataFrame]) -> dict[str, Any]:
    seeds = sorted(frames)
    reference = frames[seeds[0]].sort_values("cache_index", kind="stable")
    reference_index = reference["cache_index"].to_numpy(np.int64)
    reference_target = reference["target_rr_bpm"].to_numpy(float)
    reference_identity = reference["identity"].astype(str).to_numpy()
    predictions: list[np.ndarray] = []
    metric_values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for seed in seeds:
        frame = frames[seed].sort_values("cache_index", kind="stable")
        if not np.array_equal(frame["cache_index"].to_numpy(np.int64), reference_index):
            raise RuntimeError("seed outputs do not cover the same cache rows")
        if not np.array_equal(frame["identity"].astype(str).to_numpy(), reference_identity):
            raise RuntimeError("seed outputs have inconsistent identity lineage")
        if not np.allclose(
            frame["target_rr_bpm"].to_numpy(float), reference_target, rtol=0, atol=1e-7
        ):
            raise RuntimeError("seed outputs have inconsistent targets")
        prediction = frame["prediction_rr_bpm"].to_numpy(float)
        predictions.append(prediction)
        metrics, _ = _metric_summary(reference_target, prediction, reference_identity)
        for name in METRIC_NAMES:
            if metrics[name] is not None:
                metric_values[name].append(float(metrics[name]))
    stack = np.stack(predictions)
    row_std = np.std(stack, axis=0, ddof=0)
    pairwise = []
    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            pairwise.append(
                {
                    "seed_a": seeds[left],
                    "seed_b": seeds[right],
                    "prediction_mae_bpm": float(
                        np.mean(np.abs(stack[left] - stack[right]))
                    ),
                    "prediction_max_abs_delta_bpm": float(
                        np.max(np.abs(stack[left] - stack[right]))
                    ),
                }
            )
    metrics_across_seeds = {}
    for name, raw in metric_values.items():
        values = np.asarray(raw, dtype=float)
        metrics_across_seeds[name] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "range": float(values.max() - values.min()),
        }
    return {
        "seed_count": len(seeds),
        "seeds": seeds,
        "metrics_across_seeds": metrics_across_seeds,
        "row_prediction_standard_deviation_bpm": {
            "mean": float(row_std.mean()),
            "p95": float(np.quantile(row_std, 0.95)),
            "maximum": float(row_std.max()),
        },
        "pairwise_prediction_stability": pairwise,
    }


def _expand_mask_inputs(condition: str, paths: Sequence[Path]) -> list[Path]:
    result: set[Path] = set()
    for raw in paths:
        path = raw.expanduser().resolve()
        if path.is_file():
            result.add(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"radar-mask input does not exist: {path}")
        result.update(path.rglob(f"{condition}_test_predictions.npz"))
        result.update(
            item
            for item in path.rglob("test_predictions.npz")
            if condition in item.parts
        )
    if not result:
        raise RuntimeError(f"no prediction files found for radar condition {condition}")
    return sorted(item.resolve() for item in result)


def _load_mask_frame(
    path: Path,
    *,
    condition: str,
    records: Mapping[tuple[int, int], RunRecord],
) -> tuple[tuple[int, int], pd.DataFrame, dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "cache_index",
            "target_rr_bpm",
            "identity",
            "radar_mask",
            "seed",
            "adaptive_iteration",
            "selection_lock_sha256",
            "cache_manifest_sha256",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"radar-mask archive is missing provenance arrays: {missing}")
        prediction_key = (
            "final_rr_bpm" if "final_rr_bpm" in archive else "prediction"
            if "prediction" in archive
            else ""
        )
        if not prediction_key:
            raise RuntimeError("radar-mask archive lacks final_rr_bpm/prediction")
        fold_key = "outer_fold" if "outer_fold" in archive else "fold" if "fold" in archive else ""
        if not fold_key:
            raise RuntimeError("radar-mask archive lacks outer_fold/fold provenance")
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    seed = int(_scalar(arrays["seed"], label="radar seed"))
    fold = int(_scalar(arrays[fold_key], label="radar outer fold"))
    key = (seed, fold)
    if key not in records:
        raise RuntimeError(f"radar-mask archive has no matching locked run: {key}")
    record = records[key]
    if int(_scalar(arrays["adaptive_iteration"], label="adaptive iteration")) != record.adaptive_iteration:
        raise RuntimeError("radar-mask adaptive iteration differs from the locked run")
    if str(_scalar(arrays["selection_lock_sha256"], label="selection lock hash")) != record.selection_lock_sha256:
        raise RuntimeError("radar-mask selection-lock hash is inconsistent")
    if str(_scalar(arrays["cache_manifest_sha256"], label="cache manifest hash")) != record.cache_manifest_sha256:
        raise RuntimeError("radar-mask cache-manifest hash is inconsistent")
    observed_mask = np.asarray(arrays["radar_mask"], dtype=bool)
    if observed_mask.ndim == 2:
        if observed_mask.shape[1] != 3 or not np.all(observed_mask == observed_mask[0]):
            raise RuntimeError("radar-mask archive contains varying/malformed masks")
        observed_mask = observed_mask[0]
    if observed_mask.shape != (3,) or tuple(map(bool, observed_mask)) != RADAR_MASKS[condition]:
        raise RuntimeError("radar-mask array differs from the declared condition")
    if record.selection_lock_path.stat().st_mtime_ns > path.stat().st_mtime_ns:
        raise RuntimeError("radar-mask prediction predates the selection lock")
    expected = record.frame.rename(
        columns={"target_rr_bpm": "rr_bpm"}
    )[["cache_index", "identity", "rr_bpm"]]
    frame = _prediction_frame(
        arrays, path=path, expected=expected, prediction_key=prediction_key
    )
    primary = record.frame.sort_values("cache_index", kind="stable")
    aligned = frame.sort_values("cache_index", kind="stable")
    if not np.array_equal(
        primary["identity"].astype(str).to_numpy(), aligned["identity"].astype(str).to_numpy()
    ) or not np.allclose(
        primary["target_rr_bpm"].to_numpy(float),
        aligned["target_rr_bpm"].to_numpy(float),
        rtol=0,
        atol=1e-7,
    ):
        raise RuntimeError("radar-mask rows differ from the locked primary OOF")
    frame.insert(0, "fold", fold)
    frame.insert(0, "seed", seed)
    provenance = {
        "condition": condition,
        "seed": seed,
        "outer_fold": fold,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "radar_mask": list(RADAR_MASKS[condition]),
        "selection_lock_sha256": record.selection_lock_sha256,
        "cache_manifest_sha256": record.cache_manifest_sha256,
        "row_binding_verified": True,
    }
    return key, frame, provenance


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_strict_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _markdown(
    report: Mapping[str, Any],
    provenance_rows: Sequence[Mapping[str, Any]],
    mask_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"# {report['campaign_id']} locked evaluation provenance",
        "",
        f"Generated: `{report['created_utc']}`",
        "",
        "This is a retrospective, identity-disjoint internal evaluation. It does not "
        "authorize a commercial-performance claim; prospective confirmation remains required.",
        "",
        "## Acceptance",
        "",
        f"- Primary fixed-seed gates: **{'PASS' if report['acceptance']['primary_all_fixed_seeds_pass'] else 'FAIL'}**",
        f"- Radar-mask gates: **{'PASS' if report['acceptance']['radar_masks_all_pass'] else 'FAIL'}**",
        f"- Campaign engineering gates: **{'PASS' if report['acceptance']['campaign_engineering_pass'] else 'FAIL'}**",
        "",
        "## Primary metrics by seed",
        "",
        "| Seed | MAE | Identity-macro MAE | RMSE | ±2 bpm | >5 bpm | 25–35 bpm MAE | Gates |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed, summary in sorted(report["primary"]["by_seed"].items(), key=lambda item: int(item[0])):
        metric = summary["metrics"]
        tail = metric["tail_25_35_mae"]
        lines.append(
            f"| {seed} | {metric['mae']:.4f} | {metric['identity_macro_mae']:.4f} "
            f"| {metric['rmse']:.4f} | {metric['within_2_fraction']:.3%} "
            f"| {metric['over_5_fraction']:.3%} | "
            f"{'N/A' if tail is None else f'{tail:.4f}'} | "
            f"{'PASS' if summary['acceptance']['all_passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Verified run artifacts",
            "",
            "| Seed | Fold | Iteration | Rows | Selection lock SHA-256 | Test prediction SHA-256 |",
            "|---:|---:|---:|---:|:---|:---|",
        ]
    )
    for row in sorted(provenance_rows, key=lambda item: (int(item["seed"]), int(item["outer_fold"]))):
        lines.append(
            f"| {row['seed']} | {row['outer_fold']} | {row['adaptive_iteration']} "
            f"| {row['test_rows']} | `{row['selection_lock_sha256']}` "
            f"| `{row['test_predictions_sha256']}` |"
        )
    if mask_rows:
        lines.extend(
            [
                "",
                "## Radar-mask artifact coverage",
                "",
                "| Condition | Mask | Artifacts | Aggregate content SHA-256 |",
                "|:---|:---:|---:|:---|",
            ]
        )
        for condition, mask in RADAR_MASKS.items():
            selected = [row for row in mask_rows if row["condition"] == condition]
            aggregate = _canonical_sha256(
                sorted(str(row["sha256"]) for row in selected)
            )
            lines.append(
                f"| {condition} | `{''.join('1' if value else '0' for value in mask)}` "
                f"| {len(selected)} | `{aggregate}` |"
            )
    lines.extend(
        [
            "",
            "## Validation boundary",
            "",
            "Every outer-test artifact was observed after its atomic selection lock; cache, "
            "fallback, checkpoint, scaler, policy, row-lineage, and identity partitions were "
            "verified before metrics were computed. Bootstrap intervals resample physical "
            "identities, not overlapping windows.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(
    output_dir: Path,
    *,
    report: dict[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise RuntimeError("evaluation output directory is non-empty; refusing to overwrite")
        output_dir.rmdir()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building.", dir=output_dir.parent))
    try:
        _write_json(stage / "evaluation.json", report)
        pd.DataFrame(metric_rows).to_csv(stage / "metrics.csv", index=False)
        pd.DataFrame(identity_rows).to_csv(stage / "per_identity.csv", index=False)
        _write_json(stage / "provenance.json", provenance)
        provenance_csv_rows = [
            {"artifact_type": "locked_outer_test_run", **row}
            for row in provenance["runs"]
        ] + [
            {"artifact_type": "radar_mask_prediction", **row}
            for row in provenance["radar_mask_artifacts"]
        ]
        pd.DataFrame(provenance_csv_rows).to_csv(
            stage / "provenance.csv", index=False
        )
        (stage / "provenance.md").write_text(
            _markdown(
                report,
                provenance["runs"],
                provenance["radar_mask_artifacts"],
            ),
            encoding="utf-8",
        )
        names = (
            "evaluation.json",
            "metrics.csv",
            "per_identity.csv",
            "provenance.json",
            "provenance.csv",
            "provenance.md",
        )
        artifact_manifest: dict[str, Any] = {
            "schema_version": 1,
            "campaign_id": report["campaign_id"],
            "classification": "immutable_retrospective_evaluation_bundle",
            "artifacts": {
                name: {
                    "bytes": (stage / name).stat().st_size,
                    "sha256": sha256_file(stage / name),
                }
                for name in names
            },
        }
        artifact_manifest["content_sha256"] = _manifest_content_sha256(
            artifact_manifest
        )
        _write_json(stage / "evaluation_manifest.json", artifact_manifest)
        for path in stage.iterdir():
            path.chmod(0o444)
        stage.replace(output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def evaluate_campaign(
    run_inputs: Sequence[Path],
    *,
    output_dir: Path,
    spec: EvaluationSpec,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260828,
    radar_mask_inputs: Mapping[str, Sequence[Path]] | None = None,
) -> dict[str, Any]:
    run_dirs = _expand_run_dirs(run_inputs)
    cache_memo: dict[Path, tuple[dict[str, Any], pd.DataFrame]] = {}
    records: dict[tuple[int, int], RunRecord] = {}
    for run_dir in run_dirs:
        record = validate_run(run_dir, spec=spec, cache_memo=cache_memo)
        key = (record.seed, record.fold)
        if key in records:
            raise RuntimeError(f"duplicate fixed-seed outer-fold result: {key}")
        records[key] = record
    expected_keys = {
        (seed, fold) for seed in spec.required_seeds for fold in range(spec.fold_count)
    }
    observed_keys = set(records)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise RuntimeError(
            f"fixed-seed fold outputs are incomplete (missing={missing}, extra={extra})"
        )
    iterations = {record.adaptive_iteration for record in records.values()}
    designs = {record.design_signature_sha256 for record in records.values()}
    lineages = {record.row_lineage_sha256 for record in records.values()}
    if len(iterations) != 1:
        raise RuntimeError("adaptive result mixing is forbidden")
    if len(designs) != 1:
        raise RuntimeError("run design/cache manifests are inconsistent across folds or seeds")
    if len(lineages) != 1:
        raise RuntimeError("cache row-lineage hashes are inconsistent across runs")

    primary_frames: dict[int, pd.DataFrame] = {}
    for seed in spec.required_seeds:
        frame = pd.concat(
            [records[(seed, fold)].frame for fold in range(spec.fold_count)],
            ignore_index=True,
        )
        if frame["cache_index"].duplicated().any():
            raise RuntimeError(f"seed {seed} contains duplicate OOF rows")
        if len(frame) != spec.valid_reference_rows:
            raise RuntimeError(f"seed {seed} has missing OOF rows")
        if frame["identity"].nunique() != spec.identity_count:
            raise RuntimeError(f"seed {seed} has an incomplete identity population")
        primary_frames[seed] = frame
    primary_stability = seed_stability(primary_frames)
    primary_by_seed, metric_rows, identity_rows = _summarize_seed_frames(
        primary_frames,
        condition="primary_locked_oof",
        spec=spec,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    primary_pass = all(item["acceptance"]["all_passed"] for item in primary_by_seed.values())

    mask_report: dict[str, Any] = {"provided": False, "conditions": {}}
    mask_provenance: list[dict[str, Any]] = []
    mask_pass: bool | None = None
    if radar_mask_inputs:
        if set(radar_mask_inputs) != set(RADAR_MASKS):
            raise RuntimeError(
                "radar robustness is all-or-nothing: supply exactly all seven conditions"
            )
        mask_report["provided"] = True
        mask_pass = True
        for condition in RADAR_MASKS:
            condition_records: dict[tuple[int, int], pd.DataFrame] = {}
            for path in _expand_mask_inputs(condition, radar_mask_inputs[condition]):
                key, frame, provenance = _load_mask_frame(
                    path, condition=condition, records=records
                )
                if key in condition_records:
                    raise RuntimeError(
                        f"duplicate radar-mask result for {condition}, seed/fold {key}"
                    )
                condition_records[key] = frame
                mask_provenance.append(provenance)
            if set(condition_records) != expected_keys:
                raise RuntimeError(f"radar condition {condition} lacks exact seed/fold coverage")
            seed_frames = {
                seed: pd.concat(
                    [condition_records[(seed, fold)] for fold in range(spec.fold_count)],
                    ignore_index=True,
                )
                for seed in spec.required_seeds
            }
            stability = seed_stability(seed_frames)
            by_seed, local_metrics, local_identity = _summarize_seed_frames(
                seed_frames,
                condition=condition,
                spec=spec,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            condition_pass = all(
                item["acceptance"]["all_passed"] for item in by_seed.values()
            )
            mask_pass = bool(mask_pass and condition_pass)
            mask_report["conditions"][condition] = {
                "radar_mask": list(RADAR_MASKS[condition]),
                "by_seed": by_seed,
                "seed_stability": stability,
                "all_fixed_seeds_pass": condition_pass,
            }
            metric_rows.extend(local_metrics)
            identity_rows.extend(local_identity)

    created = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": created,
        "campaign_id": spec.campaign_id,
        "classification": "retrospective_identity_disjoint_locked_oof_evaluation",
        "retrospective_only": True,
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
        "adaptive_iteration": next(iter(iterations)),
        "population": {
            "folds": spec.fold_count,
            "seeds": list(spec.required_seeds),
            "valid_reference_rows_per_seed": spec.valid_reference_rows,
            "physical_identities": spec.identity_count,
        },
        "targets": {
            "overall_mae_bpm_max": spec.mae_max,
            "identity_macro_mae_bpm_max": spec.identity_macro_mae_max,
            "rmse_bpm_max": spec.rmse_max,
            "within_2_fraction_min": spec.within_2_min,
            "over_5_fraction_max": spec.over_5_max,
            "high_rr_25_35_mae_bpm_max": spec.tail_25_35_mae_max,
        },
        "primary": {
            "by_seed": primary_by_seed,
            "seed_stability": primary_stability,
        },
        "radar_masks": mask_report,
        "acceptance": {
            "primary_all_fixed_seeds_pass": primary_pass,
            "radar_masks_supplied": bool(radar_mask_inputs),
            "radar_masks_all_pass": mask_pass,
            "campaign_engineering_pass": bool(
                primary_pass and radar_mask_inputs and mask_pass
            ),
            "rule": "every fixed seed must pass every fixed target; when supplied, every radar-mask seed must also pass",
        },
        "bootstrap": {
            "unit": "physical_identity",
            "samples": int(bootstrap_samples),
            "confidence": 0.95,
            "seed": int(bootstrap_seed),
        },
    }
    provenance = {
        "schema_version": 1,
        "created_utc": created,
        "campaign_id": spec.campaign_id,
        "contract": {
            "path": str(spec.contract_path),
            "sha256": spec.contract_sha256,
        },
        "verification": {
            "fixed_seed_fold_exact_cover": True,
            "outer_test_after_atomic_lock": True,
            "identity_partitions_disjoint_and_contract_exact": True,
            "cache_and_manifest_hashes_consistent": True,
            "adaptive_iteration_uniform": True,
            "design_signature_uniform": True,
            "row_lineage_uniform": True,
        },
        "runs": [
            records[key].provenance for key in sorted(records)
        ],
        "radar_mask_artifacts": mask_provenance,
    }
    _write_outputs(
        output_dir,
        report=report,
        metric_rows=metric_rows,
        identity_rows=identity_rows,
        provenance=provenance,
    )
    return report


def _parse_mask_arguments(values: Sequence[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--radar-mask must be CONDITION=PATH")
        condition, raw_path = value.split("=", 1)
        if condition not in RADAR_MASKS:
            raise ValueError(f"unknown radar-mask condition: {condition}")
        if not raw_path:
            raise ValueError("--radar-mask path is empty")
        result.setdefault(condition, []).append(Path(raw_path))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Locked HCS run or a root recursively containing locked runs; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument(
        "--radar-mask",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="Bound radar-mask NPZ or discovery root; all seven conditions are required if used.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")
    try:
        radar = _parse_mask_arguments(args.radar_mask)
        report = evaluate_campaign(
            args.run_dir,
            output_dir=args.output_dir,
            spec=load_evaluation_spec(args.contract),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            radar_mask_inputs=radar or None,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "campaign_engineering_pass": report["acceptance"][
                    "campaign_engineering_pass"
                ],
                "commercial_claim_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
