#!/usr/bin/env python3
"""Seal all label-free uncertainty fields before locked HCS target access.

The primary OOF archive intentionally contains only point predictions.  This
utility reopens the already hash-bound raw inference artifacts, proves exact
row/point-prediction parity with all 18 sealed primary units, and publishes one
strictly target-free uncertainty archive.  The complete archive and every raw
source are sealed before canonical targets may be created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
DEFAULT_CALIBRATION = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/current_source_merged/uncertainty_calibration.json"
)
SCHEMA_VERSION = 1
EXPECTED_UNITS = 18
FOLDS = tuple(range(6))
REQUIRED_RAW = frozenset(
    {
        "cache_index",
        "fallback_rr_bpm",
        "fallback_std_bpm",
        "fallback_available",
        "source_rr_bpm",
        "source_scale_bpm",
        "source_available",
        "selected_probability",
        "margin",
        "entropy",
        "quality",
        "valid_candidate_count",
    }
)
SEALED_POINT_FIELDS = frozenset(
    {
        "cache_index",
        "outer_fold",
        "seed",
        "fallback_rr_bpm",
        "source_rr_bpm",
        "final_rr_bpm",
        "applied_pull",
        "target_joined",
    }
)
FORBIDDEN_FIELDS = frozenset(
    {
        "target",
        "targets",
        "target_rr_bpm",
        "reference_rr_bpm",
        "reference_valid",
        "ground_truth",
        "label",
        "labels",
        "rr_bpm",
    }
)


class UncertaintySealError(RuntimeError):
    """The target-free uncertainty boundary is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return canonical_json_sha256(payload)


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    candidate = Path(os.path.abspath(os.fspath(path.expanduser())))
    if candidate.is_symlink() or not candidate.is_file():
        raise UncertaintySealError(f"bound file must be regular and non-symlink: {candidate}")
    resolved = candidate.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _resolve(value: Any, *, relative_to: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    binding = bind_file(resolved)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UncertaintySealError(f"invalid {label}: {resolved} ({exc})") from exc
    if not isinstance(value, dict):
        raise UncertaintySealError(f"{label} root must be an object")
    if "content_sha256" in value and canonical_content_sha256(value) != value.get(
        "content_sha256"
    ):
        raise UncertaintySealError(f"{label} content hash mismatch")
    return value, binding


def _verify_binding(raw: Any, *, relative_to: Path, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise UncertaintySealError(f"missing binding: {label}")
    path = _resolve(raw.get("path", ""), relative_to=relative_to)
    expected = str(raw.get("sha256", ""))
    if len(expected) != 64 or not path.is_file() or path.is_symlink():
        raise UncertaintySealError(f"invalid binding: {label}")
    if sha256_file(path) != expected:
        raise UncertaintySealError(f"binding hash mismatch: {label}")
    if "bytes" in raw and path.stat().st_size != int(raw["bytes"]):
        raise UncertaintySealError(f"binding byte-size mismatch: {label}")
    return {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}


def _scalar(archive: Mapping[str, Any], name: str) -> Any:
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise UncertaintySealError(f"{name} must be scalar")
    return value.item()


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        Path(str(left.get("path", ""))).resolve()
        == Path(str(right.get("path", ""))).resolve()
        and str(left.get("sha256", "")) == str(right.get("sha256", ""))
        and int(left.get("bytes", -1)) == int(right.get("bytes", -2))
    )


def _validate_calibration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration, binding = _read_json(path, "pretest uncertainty calibration")
    if (
        calibration.get("schema_version") != 1
        or calibration.get("classification")
        != "locked_pretest_cross_fitted_proposer_uncertainty_calibration"
        or calibration.get("commercial_claim_authorized") is not False
        or calibration.get("prospective_confirmation_required") is not True
        or calibration.get("outer_test_opened") is not False
        or calibration.get("target_artifact_opened") is not False
        or calibration.get("point_prediction_modified") is not False
        or int(calibration.get("unit_count", -1)) != EXPECTED_UNITS
    ):
        raise UncertaintySealError("pretest uncertainty calibration invariants are invalid")
    units = calibration.get("units")
    if not isinstance(units, list):
        raise UncertaintySealError("pretest calibration lacks units")
    keys = {(int(unit["outer_fold"]), int(unit["seed"])) for unit in units}
    if len(keys) != EXPECTED_UNITS or {fold for fold, _ in keys} != set(FOLDS):
        raise UncertaintySealError("pretest calibration unit topology is incomplete")
    return calibration, binding


def _load_unit(
    unit: Mapping[str, Any], *, root: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fold = int(unit.get("outer_fold", -1))
    seed = int(unit.get("seed", -1))
    if fold not in FOLDS or seed < 0:
        raise UncertaintySealError("prediction seal contains an invalid fold/seed")
    derived_binding = _verify_binding(
        unit.get("derived_lock"), relative_to=root, label=f"derived lock {fold}/{seed}"
    )
    prediction_binding = _verify_binding(
        unit.get("prediction"), relative_to=root, label=f"sealed prediction {fold}/{seed}"
    )
    derived_path = Path(derived_binding["path"])
    derived, _ = _read_json(derived_path, f"derived inference lock {fold}/{seed}")
    if (
        derived.get("classification") != "locked_hcs_oof_derived_test_inference"
        or int(derived.get("outer_fold", -1)) != fold
        or int(derived.get("seed", -1)) != seed
        or derived.get("target_artifact_opened") is not False
        or derived.get("frozen_policy_status") != "fail_closed_no_action"
        or derived.get("no_action_bit_exact_float32_fallback") is not True
    ):
        raise UncertaintySealError(f"derived lock invariants are invalid: {fold}/{seed}")
    derived_prediction = _verify_binding(
        derived.get("sealed_prediction"),
        relative_to=derived_path.parent,
        label=f"derived sealed prediction {fold}/{seed}",
    )
    if not _same_binding(derived_prediction, prediction_binding):
        raise UncertaintySealError("prediction seal and derived lock bind different points")
    artifacts = derived.get("derived_artifacts")
    if not isinstance(artifacts, Mapping):
        raise UncertaintySealError("derived lock lacks artifacts")
    raw_binding = _verify_binding(
        artifacts.get("raw_hcs_prediction"),
        relative_to=derived_path.parent,
        label=f"raw inference {fold}/{seed}",
    )
    raw_path = Path(raw_binding["path"])
    try:
        with np.load(raw_path, allow_pickle=False) as archive:
            forbidden = FORBIDDEN_FIELDS & {str(name).lower() for name in archive.files}
            missing = REQUIRED_RAW - set(archive.files)
            if forbidden or missing:
                raise UncertaintySealError(
                    f"raw uncertainty schema invalid (forbidden={sorted(forbidden)}, missing={sorted(missing)})"
                )
            raw = {name: np.asarray(archive[name]).copy() for name in archive.files}
        with np.load(prediction_binding["path"], allow_pickle=False) as archive:
            if set(archive.files) != SEALED_POINT_FIELDS:
                raise UncertaintySealError("sealed point prediction schema differs")
            if int(_scalar(archive, "outer_fold")) != fold or int(
                _scalar(archive, "seed")
            ) != seed:
                raise UncertaintySealError("sealed point fold/seed differs")
            if bool(_scalar(archive, "target_joined")):
                raise UncertaintySealError("sealed point prediction already records a target join")
            point = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise UncertaintySealError(f"invalid unit NPZ {fold}/{seed}: {exc}") from exc
    raw_index = np.asarray(raw["cache_index"])
    if raw_index.ndim != 1 or not np.issubdtype(raw_index.dtype, np.integer) or not len(raw_index):
        raise UncertaintySealError("raw cache_index is invalid")
    rows = len(raw_index)
    for name in REQUIRED_RAW - {"cache_index"}:
        if np.asarray(raw[name]).shape != (rows,):
            raise UncertaintySealError(f"raw row field has wrong shape: {name}")
    order = np.argsort(raw_index.astype(np.int64), kind="stable")
    index = raw_index.astype(np.int64, copy=False)[order]
    if len(np.unique(index)) != rows or not np.array_equal(index, point["cache_index"]):
        raise UncertaintySealError("raw and sealed cache indices do not match")
    for name in ("fallback_rr_bpm", "source_rr_bpm"):
        raw_value = np.asarray(raw[name], dtype=np.float32)[order]
        point_value = np.asarray(point[name])
        if raw_value.dtype != point_value.dtype or not np.array_equal(
            raw_value.view(np.uint32), point_value.view(np.uint32)
        ):
            raise UncertaintySealError(f"raw/sealed point parity failed: {name}")
    fallback = np.asarray(point["fallback_rr_bpm"], dtype=np.float32)
    final = np.asarray(point["final_rr_bpm"], dtype=np.float32)
    applied = np.asarray(point["applied_pull"], dtype=np.float32)
    if not np.array_equal(fallback.view(np.uint32), final.view(np.uint32)) or not np.array_equal(
        applied.view(np.uint32), np.zeros(rows, dtype=np.float32).view(np.uint32)
    ):
        raise UncertaintySealError("no-action primary point prediction is not bit exact")
    candidate_count = np.asarray(raw["valid_candidate_count"], dtype=np.int16)[order]
    entropy = np.asarray(raw["entropy"], dtype=np.float32)[order]
    if "normalized_entropy" in raw:
        normalized_entropy = np.asarray(raw["normalized_entropy"], dtype=np.float32)[order]
    else:
        normalized_entropy = np.where(
            candidate_count > 1,
            entropy
            / np.maximum(np.log(np.maximum(candidate_count.astype(np.float32), 2.0)), 1e-8),
            0.0,
        ).astype(np.float32)
    arrays = {
        "cache_index": index,
        "outer_fold": np.full(rows, fold, dtype=np.int16),
        "seed": np.full(rows, seed, dtype=np.int64),
        "final_rr_bpm": final,
        "fallback_std_bpm": np.asarray(raw["fallback_std_bpm"], dtype=np.float32)[order],
        "source_scale_bpm": np.asarray(raw["source_scale_bpm"], dtype=np.float32)[order],
        "selected_probability": np.asarray(raw["selected_probability"], dtype=np.float32)[order],
        "margin": np.asarray(raw["margin"], dtype=np.float32)[order],
        "normalized_entropy": normalized_entropy,
        "quality": np.asarray(raw["quality"], dtype=np.float32)[order],
        "valid_candidate_count": candidate_count,
        "fallback_available": np.asarray(raw["fallback_available"], dtype=bool)[order],
        "source_available": np.asarray(raw["source_available"], dtype=bool)[order],
    }
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.shape != (rows,):
            raise UncertaintySealError(f"uncertainty output shape differs: {name}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise UncertaintySealError(f"uncertainty output is non-finite: {name}")
    if (arrays["fallback_std_bpm"] < 0).any() or (arrays["source_scale_bpm"] < 0).any():
        raise UncertaintySealError("uncertainty scale is negative")
    return arrays, {
        "outer_fold": fold,
        "seed": seed,
        "derived_lock": derived_binding,
        "raw_hcs_prediction": raw_binding,
        "sealed_prediction": prediction_binding,
        "rows": rows,
    }


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    candidate = Path(os.path.abspath(os.fspath(path.expanduser())))
    if candidate.is_symlink():
        raise UncertaintySealError(f"uncertainty archive path is a symlink: {candidate}")
    destination = candidate.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise UncertaintySealError(f"refusing to overwrite uncertainty archive: {destination}")
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise UncertaintySealError("uncertainty archive appeared concurrently") from exc
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    candidate = Path(os.path.abspath(os.fspath(path.expanduser())))
    if candidate.is_symlink():
        raise UncertaintySealError(f"uncertainty seal path is a symlink: {candidate}")
    destination = candidate.resolve()
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != payload:
            raise UncertaintySealError(f"immutable seal differs: {destination}")
        destination.chmod(0o444)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise UncertaintySealError("uncertainty seal appeared concurrently") from exc
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_existing_seal(path: Path) -> dict[str, Any]:
    candidate = Path(os.path.abspath(os.fspath(path.expanduser())))
    if candidate.is_symlink() or not candidate.is_file():
        raise UncertaintySealError("existing uncertainty seal must be regular/non-symlink")
    resolved = candidate.resolve()
    if resolved.stat().st_mode & 0o777 != 0o444:
        raise UncertaintySealError("existing uncertainty seal mode must be exactly 0444")
    seal, _ = _read_json(resolved, "uncertainty input seal")
    expected_keys = {
        "schema_version",
        "classification",
        "commercial_claim_authorized",
        "prospective_confirmation_required",
        "target_artifact_opened_before_seal",
        "target_fields_present",
        "point_prediction_modified",
        "no_action_primary_bit_exact_verified",
        "unit_count",
        "folds",
        "seeds",
        "row_count",
        "rows_per_seed",
        "predictions_seal",
        "pretest_calibration",
        "uncertainty_archive",
        "array_schema",
        "units",
        "effective_sources",
        "content_sha256",
    }
    if (
        set(seal) != expected_keys
        or seal.get("schema_version") != SCHEMA_VERSION
        or seal.get("classification") != "locked_hcs_all_target_free_uncertainty_inputs_sealed"
        or seal.get("commercial_claim_authorized") is not False
        or seal.get("prospective_confirmation_required") is not True
        or seal.get("target_artifact_opened_before_seal") is not False
        or seal.get("target_fields_present") is not False
        or seal.get("point_prediction_modified") is not False
        or seal.get("no_action_primary_bit_exact_verified") is not True
        or int(seal.get("unit_count", -1)) != EXPECTED_UNITS
        or seal.get("folds") != list(FOLDS)
        or canonical_content_sha256(seal) != seal.get("content_sha256")
    ):
        raise UncertaintySealError("existing uncertainty seal invariants are invalid")
    archive_binding = _verify_binding(
        seal.get("uncertainty_archive"), relative_to=resolved.parent, label="uncertainty archive"
    )
    archive_path = Path(archive_binding["path"])
    if archive_path.is_symlink() or archive_path.stat().st_mode & 0o777 != 0o444:
        raise UncertaintySealError("existing uncertainty archive mode must be exactly 0444")
    predictions_binding = _verify_binding(
        seal.get("predictions_seal"), relative_to=resolved.parent, label="primary predictions seal"
    )
    predictions, live_predictions_binding = _read_json(
        Path(predictions_binding["path"]), "primary predictions seal"
    )
    if not _same_binding(predictions_binding, live_predictions_binding):
        raise UncertaintySealError("existing uncertainty seal prediction binding differs")
    prediction_units = predictions.get("units")
    if (
        predictions.get("schema_version") != SCHEMA_VERSION
        or predictions.get("classification")
        != "locked_hcs_oof_all_label_free_predictions_sealed"
        or predictions.get("target_artifact_opened_before_seal") is not False
        or predictions.get("target_join_authorized") is not True
        or int(predictions.get("unit_count", -1)) != EXPECTED_UNITS
        or not isinstance(prediction_units, list)
        or len(prediction_units) != EXPECTED_UNITS
    ):
        raise UncertaintySealError("existing uncertainty seal binds an invalid prediction policy")
    recorded_calibration = _verify_binding(
        seal.get("pretest_calibration"),
        relative_to=resolved.parent,
        label="pretest calibration",
    )
    calibration, calibration_binding = _validate_calibration(
        Path(recorded_calibration["path"])
    )
    if not _same_binding(calibration_binding, recorded_calibration):
        raise UncertaintySealError("existing uncertainty seal calibration binding differs")
    effective_sources = seal.get("effective_sources")
    if not isinstance(effective_sources, Mapping) or set(effective_sources) != {"sealer"}:
        raise UncertaintySealError("existing uncertainty seal source topology differs")
    sealer_binding = _verify_binding(
        effective_sources["sealer"], relative_to=resolved.parent, label="uncertainty sealer"
    )
    if not _same_binding(sealer_binding, bind_file(Path(__file__))):
        raise UncertaintySealError("existing uncertainty seal was produced by another sealer")
    units = seal.get("units")
    if not isinstance(units, list) or len(units) != EXPECTED_UNITS:
        raise UncertaintySealError("existing uncertainty seal unit matrix is incomplete")
    parts: list[dict[str, np.ndarray]] = []
    records: list[dict[str, Any]] = []
    keys: set[tuple[int, int]] = set()
    for prediction_unit in prediction_units:
        if not isinstance(prediction_unit, Mapping):
            raise UncertaintySealError("existing prediction seal unit is not an object")
        arrays, record = _load_unit(prediction_unit, root=Path(predictions_binding["path"]).parent)
        key = (int(record["outer_fold"]), int(record["seed"]))
        if key in keys:
            raise UncertaintySealError("existing uncertainty seal has duplicate units")
        keys.add(key)
        parts.append(arrays)
        records.append(record)
    if len(keys) != EXPECTED_UNITS or {fold for fold, _ in keys} != set(FOLDS):
        raise UncertaintySealError("existing uncertainty seal topology is incomplete")
    calibration_keys = {
        (int(unit["outer_fold"]), int(unit["seed"])) for unit in calibration["units"]
    }
    if calibration_keys != keys:
        raise UncertaintySealError("existing uncertainty calibration topology differs")
    order = sorted(range(len(records)), key=lambda index: (records[index]["seed"], records[index]["outer_fold"]))
    expected_records = [records[position] for position in order]
    if units != expected_records:
        raise UncertaintySealError("existing uncertainty seal unit provenance differs")
    names = tuple(parts[0])
    combined = {
        name: np.concatenate([parts[position][name] for position in order]) for name in names
    }
    seeds = sorted(set(combined["seed"].astype(int).tolist()))
    if seal.get("seeds") != seeds or int(seal.get("row_count", -1)) != len(
        combined["cache_index"]
    ):
        raise UncertaintySealError("existing uncertainty seal row/seed topology differs")
    rows_per_seed = [int(np.sum(combined["seed"] == seed)) for seed in seeds]
    if len(set(rows_per_seed)) != 1 or int(seal.get("rows_per_seed", -1)) != rows_per_seed[0]:
        raise UncertaintySealError("existing uncertainty seal per-seed row count differs")
    schema = seal.get("array_schema")
    if (
        not isinstance(schema, Mapping)
        or set(schema) != set(names)
        or set(schema) & FORBIDDEN_FIELDS
    ):
        raise UncertaintySealError("existing uncertainty archive schema is invalid")
    try:
        with np.load(archive_binding["path"], allow_pickle=False) as archive:
            if set(archive.files) != set(schema):
                raise UncertaintySealError("existing uncertainty archive fields differ from seal")
            for name, raw_schema in schema.items():
                if not isinstance(raw_schema, Mapping):
                    raise UncertaintySealError(f"invalid array schema: {name}")
                value = np.asarray(archive[name])
                if (
                    value.dtype.str != str(raw_schema.get("dtype", ""))
                    or list(value.shape) != raw_schema.get("shape")
                    or array_sha256(value) != str(raw_schema.get("sha256", ""))
                    or not np.array_equal(value, combined[name])
                ):
                    raise UncertaintySealError(f"uncertainty array/schema mismatch: {name}")
    except (OSError, ValueError) as exc:
        raise UncertaintySealError(f"invalid uncertainty archive: {exc}") from exc
    return seal


def _validate_recoverable_archive(
    path: Path, arrays: Mapping[str, Any]
) -> None:
    """Accept only the exact deterministic arrays from a seal-publication crash."""

    candidate = Path(os.path.abspath(os.fspath(path.expanduser())))
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or candidate.stat().st_mode & 0o777 != 0o444
    ):
        raise UncertaintySealError(
            "unsealed uncertainty archive must be a regular mode-0444 file"
        )
    try:
        with np.load(candidate, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise UncertaintySealError(
                    "unsealed uncertainty archive field topology differs"
                )
            for name, expected in arrays.items():
                observed = np.asarray(archive[name])
                value = np.asarray(expected)
                if (
                    observed.dtype != value.dtype
                    or observed.shape != value.shape
                    or observed.tobytes(order="C") != value.tobytes(order="C")
                ):
                    raise UncertaintySealError(
                        f"unsealed uncertainty archive payload differs: {name}"
                    )
    except (OSError, ValueError) as exc:
        raise UncertaintySealError(
            f"invalid recoverable uncertainty archive: {candidate} ({exc})"
        ) from exc


def seal_uncertainty_inputs(
    *,
    root: Path,
    calibration_path: Path,
    output_path: Path,
    seal_path: Path,
) -> dict[str, Any]:
    locked_root = root.expanduser().resolve()
    output = Path(os.path.abspath(os.fspath(output_path.expanduser())))
    seal_destination = Path(os.path.abspath(os.fspath(seal_path.expanduser())))
    if output.is_symlink() or seal_destination.is_symlink():
        raise UncertaintySealError("uncertainty output/seal path must not be a symlink")
    if seal_destination.exists():
        existing = _validate_existing_seal(seal_destination)
        archive = existing.get("uncertainty_archive")
        if (
            not isinstance(archive, Mapping)
            or Path(str(archive.get("path", ""))).expanduser().resolve()
            != output.resolve()
        ):
            raise UncertaintySealError("existing seal binds another uncertainty archive")
        return existing
    recover_unsealed_archive = output.exists()
    for target_bearing in (
        locked_root / "evaluation_lock.json",
        locked_root / "locked_hcs_oof_joined.npz",
        locked_root / "canonical_locked_hcs_targets.npz",
        locked_root / "canonical_locked_hcs_targets_receipt.json",
    ):
        if target_bearing.exists():
            raise UncertaintySealError(
                f"uncertainty inputs must be sealed before target artifact access: {target_bearing}"
            )
    calibration, calibration_binding = _validate_calibration(calibration_path)
    predictions_path = locked_root / "predictions_seal.json"
    predictions, predictions_binding = _read_json(predictions_path, "primary predictions seal")
    units = predictions.get("units")
    if (
        predictions.get("classification")
        != "locked_hcs_oof_all_label_free_predictions_sealed"
        or predictions.get("target_artifact_opened_before_seal") is not False
        or predictions.get("target_join_authorized") is not True
        or int(predictions.get("unit_count", -1)) != EXPECTED_UNITS
        or not isinstance(units, list)
        or len(units) != EXPECTED_UNITS
    ):
        raise UncertaintySealError("primary predictions seal is incomplete")
    parts: list[dict[str, np.ndarray]] = []
    records: list[dict[str, Any]] = []
    keys: set[tuple[int, int]] = set()
    for unit in units:
        arrays, record = _load_unit(unit, root=locked_root)
        key = (int(record["outer_fold"]), int(record["seed"]))
        if key in keys:
            raise UncertaintySealError("primary prediction seal contains duplicate units")
        keys.add(key)
        parts.append(arrays)
        records.append(record)
    calibration_keys = {
        (int(unit["outer_fold"]), int(unit["seed"])) for unit in calibration["units"]
    }
    if keys != calibration_keys:
        raise UncertaintySealError("calibration and primary prediction unit topologies differ")
    order = sorted(range(len(records)), key=lambda index: (records[index]["seed"], records[index]["outer_fold"]))
    names = tuple(parts[0])
    combined = {
        name: np.concatenate([parts[position][name] for position in order]) for name in names
    }
    seeds = sorted(set(combined["seed"].astype(int).tolist()))
    expected_index: np.ndarray | None = None
    for seed in seeds:
        selected = combined["seed"] == seed
        values = combined["cache_index"][selected]
        if len(np.unique(values)) != len(values):
            raise UncertaintySealError(f"uncertainty rows duplicate cache indices for seed {seed}")
        current = np.sort(values)
        if expected_index is None:
            expected_index = current
        elif not np.array_equal(current, expected_index):
            raise UncertaintySealError("uncertainty cache-index coverage differs across seeds")
    if recover_unsealed_archive:
        _validate_recoverable_archive(output, combined)
    else:
        _atomic_npz(output, combined)
    archive_binding = bind_file(output)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_all_target_free_uncertainty_inputs_sealed",
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
        "target_artifact_opened_before_seal": False,
        "target_fields_present": False,
        "point_prediction_modified": False,
        "no_action_primary_bit_exact_verified": True,
        "unit_count": len(records),
        "folds": list(FOLDS),
        "seeds": seeds,
        "row_count": int(len(combined["cache_index"])),
        "rows_per_seed": int(len(expected_index if expected_index is not None else ())),
        "predictions_seal": predictions_binding,
        "pretest_calibration": calibration_binding,
        "uncertainty_archive": archive_binding,
        "array_schema": {
            name: {
                "dtype": np.asarray(value).dtype.str,
                "shape": list(np.asarray(value).shape),
                "sha256": array_sha256(value),
            }
            for name, value in combined.items()
        },
        "units": [records[position] for position in order],
        "effective_sources": {"sealer": bind_file(Path(__file__))},
    }
    document["content_sha256"] = canonical_content_sha256(document)
    _atomic_json(seal_destination, document)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seal", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    try:
        result = seal_uncertainty_inputs(
            root=root,
            calibration_path=args.calibration,
            output_path=(args.output if args.output is not None else root / "locked_hcs_uncertainty_inputs.npz"),
            seal_path=(args.seal if args.seal is not None else root / "uncertainty_inputs_seal.json"),
        )
    except (UncertaintySealError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
