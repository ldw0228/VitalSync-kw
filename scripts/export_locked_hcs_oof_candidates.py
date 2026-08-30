#!/usr/bin/env python3
"""Export immutable, per-seed commercial-audit CSVs from the locked HCS join.

This is a provenance/export boundary, not a selector.  It first verifies the
single-target evaluation lock, every file bound by that lock, and the canonical
target receipt.  Only then are target and joined arrays opened.  All three
pre-specified seeds are exported independently; no score, threshold, seed, or
prediction column is selected using the joined targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
FOLDS = tuple(range(6))
EXPECTED_SEED_COUNT = 3
EXPECTED_VALID_REFERENCE_ROWS = 2327
EXPECTED_IDENTITIES = 18
REQUIRED_JOINED_FIELDS = {
    "cache_index",
    "outer_fold",
    "seed",
    "target_rr_bpm",
    "identity",
    "fallback_rr_bpm",
    "source_rr_bpm",
    "final_rr_bpm",
}
REQUIRED_TARGET_FIELDS = {
    "cache_index",
    "outer_fold",
    "target_rr_bpm",
    "identity",
    "reference_valid",
    "session_id",
    "window_number",
    "protocol",
    "window_start_s",
    "window_end_s",
}
# These are carried through only when the immutable joined NPZ already contains
# them.  No uncertainty proxy is synthesized after targets become available.
OPTIONAL_JOINED_FIELDS = (
    "uncertainty_uncalibrated",
    "uncertainty_bpm",
    "fallback_std_bpm",
    "source_scale_bpm",
    "selected_probability",
    "margin",
    "entropy",
    "quality",
    "valid_candidate_count",
    "spike_rate",
    "spike_rate_per_sample",
    "applied_pull",
)
CSV_BASE_COLUMNS = (
    "cache_index",
    "fold",
    "identity",
    "session_id",
    "protocol",
    "window_number",
    "window_start_s",
    "window_end_s",
    "rr_bpm",
    "seed",
    "fallback_rr_bpm",
    "source_rr_bpm",
    "final_rr_bpm",
    # Compatibility with evaluate_commercial_goal.py's default prediction name.
    "prediction_uncalibrated_bpm",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCKED_ROOT = (
    PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
)
DEFAULT_EXPORT_ROOT = DEFAULT_LOCKED_ROOT / "commercial_goal_candidates"


class CandidateExportError(RuntimeError):
    """A lock, topology, alignment, metric, or publication invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
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


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _read_json(path: Path, label: str, *, verify_content_hash: bool = False) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CandidateExportError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise CandidateExportError(f"{label} contains non-finite JSON number {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateExportError(f"invalid {label}: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise CandidateExportError(f"{label} root must be an object: {path}")
    if verify_content_hash:
        payload = dict(value)
        expected = str(payload.pop("content_sha256", "")).lower()
        if not _is_sha256(expected) or canonical_json_sha256(payload) != expected:
            raise CandidateExportError(f"{label} content_sha256 mismatch")
    return value


def _resolve(value: Any, *, relative_to: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CandidateExportError(f"{label}.path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise CandidateExportError(f"expected file is absent: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _verify_binding(raw: Any, *, relative_to: Path, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CandidateExportError(f"missing file binding: {label}")
    path = _resolve(raw.get("path"), relative_to=relative_to, label=label)
    expected = str(raw.get("sha256", "")).lower()
    if not _is_sha256(expected):
        raise CandidateExportError(f"invalid SHA-256 binding: {label}")
    if not path.is_file():
        raise CandidateExportError(f"bound file is absent: {label} ({path})")
    size = path.stat().st_size
    declared = raw.get("bytes")
    if declared is not None and (
        isinstance(declared, bool) or not isinstance(declared, int) or declared != size
    ):
        raise CandidateExportError(f"file byte-size binding mismatch: {label}")
    if sha256_file(path) != expected:
        raise CandidateExportError(f"file SHA-256 binding mismatch: {label} ({path})")
    return {"path": str(path), "sha256": expected, "bytes": size}


def _strict_int_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise CandidateExportError(f"{label} must be a one-dimensional integer array")
    return array.astype(np.int64, copy=False)


def _array_schema(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": np.asarray(value).dtype.str,
            "shape": list(np.asarray(value).shape),
            "array_sha256": array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }


def _load_npz(path: Path, *, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise CandidateExportError(f"invalid {label}: {path} ({exc})") from exc


def _verify_target_schema(
    arrays: Mapping[str, np.ndarray], receipt: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    schema = receipt.get("target_schema")
    if not isinstance(schema, Mapping) or set(schema) != set(arrays):
        raise CandidateExportError("target receipt schema differs from target NPZ fields")
    observed = _array_schema(arrays)
    for name, actual in observed.items():
        declared = schema.get(name)
        if not isinstance(declared, Mapping):
            raise CandidateExportError(f"target receipt schema is invalid: {name}")
        if (
            declared.get("dtype") != actual["dtype"]
            or declared.get("shape") != actual["shape"]
            or declared.get("array_sha256") != actual["array_sha256"]
        ):
            raise CandidateExportError(f"target array hash/schema mismatch: {name}")
    return observed


def _metric_record(target: np.ndarray, prediction: np.ndarray, identity: np.ndarray) -> dict[str, Any]:
    error = np.abs(prediction.astype(np.float64) - target.astype(np.float64))
    identity_mae = {
        name: float(np.mean(error[identity == name])) for name in sorted(set(identity.tolist()))
    }
    tail = (target >= 25.0) & (target <= 35.0)
    return {
        "rows": int(len(target)),
        "mae": float(np.mean(error)),
        "identity_macro_mae": float(np.mean(list(identity_mae.values()))),
        "identity_mae": identity_mae,
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "within_2": float(np.mean(error <= 2.0)),
        "catastrophic_over_5": float(np.mean(error > 5.0)),
        "tail_25_35_mae": float(np.mean(error[tail])) if tail.any() else None,
    }


def _compare_metric(actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    scalar_fields = (
        "mae",
        "identity_macro_mae",
        "rmse",
        "within_2",
        "catastrophic_over_5",
        "tail_25_35_mae",
    )
    if actual.get("rows") != expected["rows"]:
        raise CandidateExportError(f"metrics row count mismatch: {label}")
    for field in scalar_fields:
        observed = actual.get(field)
        wanted = expected[field]
        if wanted is None:
            equal = observed is None
        else:
            try:
                equal = math.isclose(float(observed), float(wanted), rel_tol=1e-12, abs_tol=1e-12)
            except (TypeError, ValueError):
                equal = False
        if not equal:
            raise CandidateExportError(f"metrics value mismatch: {label}.{field}")
    identity_actual = actual.get("identity_mae")
    if not isinstance(identity_actual, Mapping) or set(identity_actual) != set(expected["identity_mae"]):
        raise CandidateExportError(f"metrics identity topology mismatch: {label}")
    for identity, wanted in expected["identity_mae"].items():
        if not math.isclose(
            float(identity_actual[identity]), float(wanted), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise CandidateExportError(f"metrics identity value mismatch: {label}/{identity}")


def _validate_context(
    *,
    locked_oof_root: Path,
    evaluation_lock: Path,
    target_receipt: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    root = locked_oof_root.expanduser().resolve()
    lock_path = evaluation_lock.expanduser().resolve()
    receipt_path = target_receipt.expanduser().resolve()
    if not lock_path.is_file():
        # Nothing target-bearing is resolved or opened before this check.
        raise CandidateExportError("evaluation_lock.json is absent; candidate export forbidden")

    lock = _read_json(lock_path, "evaluation lock")
    if (
        lock.get("schema_version") != SCHEMA_VERSION
        or lock.get("classification") != "locked_hcs_oof_single_target_join_seal"
        or lock.get("target_join_count") != 1
        or lock.get("commercial_claim_authorized") is not False
    ):
        raise CandidateExportError("evaluation lock classification/claim invariant is invalid")
    try:
        lock_path.relative_to(root)
    except ValueError as exc:
        raise CandidateExportError("evaluation lock must reside under --locked-oof-root") from exc
    predictions_seal = _verify_binding(
        lock.get("predictions_seal"), relative_to=lock_path.parent, label="predictions seal"
    )
    target_binding = _verify_binding(
        lock.get("target_artifact"), relative_to=lock_path.parent, label="target artifact"
    )
    outputs = lock.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"joined_oof", "metrics"}:
        raise CandidateExportError("evaluation lock output topology is invalid")
    joined_binding = _verify_binding(
        outputs["joined_oof"], relative_to=lock_path.parent, label="joined OOF"
    )
    metrics_binding = _verify_binding(
        outputs["metrics"], relative_to=lock_path.parent, label="locked metrics"
    )

    metrics = _read_json(Path(metrics_binding["path"]), "locked metrics")
    if (
        metrics.get("schema_version") != SCHEMA_VERSION
        or metrics.get("classification") != "retrospective_locked_hcs_oof_evaluation"
        or metrics.get("commercial_claim_authorized") is not False
    ):
        raise CandidateExportError("locked metrics classification/claim invariant is invalid")

    if not receipt_path.is_file():
        raise CandidateExportError("canonical target receipt is absent")
    receipt = _read_json(receipt_path, "canonical target receipt", verify_content_hash=True)
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("classification")
        != "retrospective_locked_hcs_canonical_target_artifact_receipt"
        or receipt.get("commercial_claim_authorized") is not False
        or receipt.get("target_artifact_created_once") is not True
        or receipt.get("target_artifact_overwrite_allowed") is not False
    ):
        raise CandidateExportError("canonical target receipt invariants are invalid")
    receipt_target = _verify_binding(
        receipt.get("target_artifact"), relative_to=receipt_path.parent, label="receipt target"
    )
    if receipt_target != target_binding:
        raise CandidateExportError("evaluation lock and target receipt bind different target artifacts")

    # Target/joined array values are opened only after both locks and all file
    # bindings above have been verified.
    target_arrays = _load_npz(Path(target_binding["path"]), label="canonical target NPZ")
    joined_arrays = _load_npz(Path(joined_binding["path"]), label="joined OOF NPZ")
    target_schema = _verify_target_schema(target_arrays, receipt)
    joined_schema = _array_schema(joined_arrays)
    bindings = {
        "evaluation_lock": bind_file(lock_path),
        "predictions_seal": predictions_seal,
        "target_receipt": bind_file(receipt_path),
        "target_artifact": target_binding,
        "joined_oof": joined_binding,
        "metrics": metrics_binding,
    }
    return lock, receipt, metrics, target_arrays, joined_arrays, target_schema, {
        "bindings": bindings,
        "joined_schema": joined_schema,
    }


def _validate_and_build_candidates(
    *,
    receipt: Mapping[str, Any],
    metrics: Mapping[str, Any],
    target: Mapping[str, np.ndarray],
    joined: Mapping[str, np.ndarray],
    expected_rows: int,
    expected_folds: int,
    expected_identities: int,
    expected_seed_count: int,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    if expected_folds != len(FOLDS):
        raise CandidateExportError("this locked HCS export requires exactly six folds")
    missing_target = sorted(REQUIRED_TARGET_FIELDS - set(target))
    missing_joined = sorted(REQUIRED_JOINED_FIELDS - set(joined))
    if missing_target or missing_joined:
        raise CandidateExportError(
            f"locked arrays are missing fields (target={missing_target}, joined={missing_joined})"
        )
    target_rows = len(np.asarray(target["cache_index"]))
    if any(np.asarray(value).shape != (target_rows,) for value in target.values()):
        raise CandidateExportError("canonical target arrays do not share one row shape")
    target_index = _strict_int_array(target["cache_index"], label="target cache_index")
    target_fold = _strict_int_array(target["outer_fold"], label="target outer_fold")
    if len(np.unique(target_index)) != target_rows:
        raise CandidateExportError("canonical target cache_index is not unique")
    if set(np.unique(target_fold).tolist()) != set(range(expected_folds)):
        raise CandidateExportError("canonical target folds do not cover 0..5")
    target_rr = np.asarray(target["target_rr_bpm"])
    target_identity = np.asarray(target["identity"]).astype(str)
    reference_valid_raw = np.asarray(target["reference_valid"])
    if reference_valid_raw.dtype.kind != "b":
        raise CandidateExportError("canonical reference_valid must be boolean")
    reference_valid = reference_valid_raw.astype(bool, copy=False)
    valid = reference_valid & np.isfinite(target_rr.astype(float))
    valid_index = target_index[valid]
    if len(valid_index) != expected_rows:
        raise CandidateExportError(
            f"valid-reference target rows are {len(valid_index)}, expected exactly {expected_rows}"
        )
    if receipt.get("row_count") != target_rows or receipt.get("valid_reference_rows") != expected_rows:
        raise CandidateExportError("target receipt row counts differ from canonical arrays")

    session = np.asarray(target["session_id"]).astype(str)
    protocol = np.asarray(target["protocol"]).astype(str)
    window_number = _strict_int_array(target["window_number"], label="target window_number")
    start = np.asarray(target["window_start_s"], dtype=np.float64)
    end = np.asarray(target["window_end_s"], dtype=np.float64)
    if (
        any(not value or value != value.strip() for value in session.tolist())
        or any(not value or value != value.strip() for value in protocol.tolist())
        or (window_number < 0).any()
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or (end <= start).any()
    ):
        raise CandidateExportError("canonical target session/window/protocol semantics are invalid")
    semantic = list(zip(session.tolist(), window_number.tolist(), strict=True))
    if len(set(semantic)) != target_rows:
        raise CandidateExportError("canonical target repeats a session/window semantic key")

    valid_identity = target_identity[valid]
    valid_fold = target_fold[valid]
    identity_names = set(valid_identity.tolist())
    if len(identity_names) != expected_identities:
        raise CandidateExportError(
            f"valid-reference identities are {len(identity_names)}, expected {expected_identities}"
        )
    identity_fold: dict[str, int] = {}
    for name in identity_names:
        owned = set(valid_fold[valid_identity == name].tolist())
        if len(owned) != 1:
            raise CandidateExportError("an identity crosses outer-test folds")
        identity_fold[name] = int(next(iter(owned)))

    joined_rows = len(np.asarray(joined["cache_index"]))
    if any(np.asarray(value).shape != (joined_rows,) for value in joined.values()):
        raise CandidateExportError("joined OOF arrays do not share one row shape")
    joined_index = _strict_int_array(joined["cache_index"], label="joined cache_index")
    joined_fold = _strict_int_array(joined["outer_fold"], label="joined outer_fold")
    joined_seed = _strict_int_array(joined["seed"], label="joined seed")
    if joined_rows != expected_rows * expected_seed_count:
        raise CandidateExportError("joined OOF total row count is not rows x fixed seeds")
    seeds = sorted(set(joined_seed.tolist()))
    if len(seeds) != expected_seed_count:
        raise CandidateExportError("joined OOF does not contain exactly three fixed seeds")
    topology = receipt.get("prediction_topology")
    if not isinstance(topology, Mapping):
        raise CandidateExportError("target receipt prediction topology is absent")
    declared_seeds = topology.get("seeds")
    if (
        not isinstance(declared_seeds, list)
        or sorted(int(value) for value in declared_seeds) != seeds
        or topology.get("folds") != list(FOLDS)
        or topology.get("unit_count") != expected_seed_count * expected_folds
    ):
        raise CandidateExportError("target receipt and joined seed/fold topology differ")

    lookup = {int(value): position for position, value in enumerate(target_index)}
    candidate_fields = (
        "fallback_rr_bpm",
        "source_rr_bpm",
        "final_rr_bpm",
    )
    for name in candidate_fields:
        values = np.asarray(joined[name], dtype=np.float64)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise CandidateExportError(f"joined prediction is non-finite/non-positive: {name}")
    joined_target = np.asarray(joined["target_rr_bpm"])
    joined_identity = np.asarray(joined["identity"]).astype(str)
    optional = [name for name in OPTIONAL_JOINED_FIELDS if name in joined]
    candidates: dict[int, dict[str, np.ndarray]] = {}
    metrics_per_seed = metrics.get("per_seed")
    if not isinstance(metrics_per_seed, Mapping) or set(metrics_per_seed) != {str(seed) for seed in seeds}:
        raise CandidateExportError("locked metrics seed topology differs from joined OOF")

    for seed in seeds:
        rows = np.flatnonzero(joined_seed == seed)
        if len(rows) != expected_rows:
            raise CandidateExportError(f"seed {seed} does not have exactly {expected_rows} rows")
        index = joined_index[rows]
        if len(np.unique(index)) != expected_rows or set(index.tolist()) != set(valid_index.tolist()):
            raise CandidateExportError(f"seed {seed} is not an exact unique valid-reference cover")
        order = np.argsort(index, kind="stable")
        rows = rows[order]
        index = joined_index[rows]
        positions = np.asarray([lookup[int(value)] for value in index], dtype=np.int64)
        if not valid[positions].all():
            raise CandidateExportError(f"seed {seed} joined an invalid reference row")
        if (
            not np.array_equal(joined_fold[rows], target_fold[positions])
            or not np.array_equal(joined_identity[rows], target_identity[positions])
            or not np.array_equal(joined_target[rows], target_rr[positions])
        ):
            raise CandidateExportError(f"seed {seed} target/fold/identity lineage mismatch")
        if set(joined_fold[rows].tolist()) != set(FOLDS):
            raise CandidateExportError(f"seed {seed} does not cover all six folds")
        if set(joined_identity[rows].tolist()) != identity_names:
            raise CandidateExportError(f"seed {seed} does not cover all fixed identities")
        for fold in FOLDS:
            expected_fold_index = set(target_index[valid & (target_fold == fold)].tolist())
            observed_fold_index = set(index[joined_fold[rows] == fold].tolist())
            if observed_fold_index != expected_fold_index:
                raise CandidateExportError(f"seed {seed}/fold {fold} coverage differs from target")

        candidate: dict[str, np.ndarray] = {
            "cache_index": index.astype(np.int64, copy=False),
            "fold": target_fold[positions].astype(np.int64, copy=False),
            "identity": target_identity[positions].astype(np.str_),
            "session_id": session[positions].astype(np.str_),
            "protocol": protocol[positions].astype(np.str_),
            "window_number": window_number[positions].astype(np.int64, copy=False),
            "window_start_s": start[positions],
            "window_end_s": end[positions],
            "rr_bpm": target_rr[positions],
            "seed": np.full(expected_rows, seed, dtype=np.int64),
            "fallback_rr_bpm": np.asarray(joined["fallback_rr_bpm"])[rows],
            "source_rr_bpm": np.asarray(joined["source_rr_bpm"])[rows],
            "final_rr_bpm": np.asarray(joined["final_rr_bpm"])[rows],
            "prediction_uncalibrated_bpm": np.asarray(joined["final_rr_bpm"])[rows],
        }
        for name in optional:
            value = np.asarray(joined[name])[rows]
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise CandidateExportError(f"optional joined field is non-finite: {name}")
            candidate[name] = value

        metric_group = metrics_per_seed[str(seed)]
        if not isinstance(metric_group, Mapping) or set(metric_group) != {
            "fallback", "source", "locked_final"
        }:
            raise CandidateExportError(f"locked metrics candidate topology is invalid: seed {seed}")
        for metric_name, column in (
            ("fallback", "fallback_rr_bpm"),
            ("source", "source_rr_bpm"),
            ("locked_final", "final_rr_bpm"),
        ):
            expected_metric = _metric_record(
                candidate["rr_bpm"], candidate[column], candidate["identity"]
            )
            actual_metric = metric_group[metric_name]
            if not isinstance(actual_metric, Mapping):
                raise CandidateExportError(f"locked metric record is invalid: {seed}/{metric_name}")
            _compare_metric(actual_metric, expected_metric, label=f"{seed}/{metric_name}")
        candidates[seed] = candidate

    topology_audit = {
        "seeds": seeds,
        "seed_count": len(seeds),
        "folds": list(FOLDS),
        "fold_count": expected_folds,
        "identities": sorted(identity_names),
        "identity_count": len(identity_names),
        "valid_reference_rows_per_seed": expected_rows,
        "joined_rows": joined_rows,
        "exact_unique_valid_reference_cover_per_seed": True,
        "one_outer_fold_per_identity": True,
        "target_fold_identity_lineage_cross_checked": True,
        "session_window_protocol_lineage_cross_checked": True,
        "locked_metrics_recomputed_and_cross_checked": True,
        "optional_joined_fields_exported": optional,
    }
    return candidates, topology_audit


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")


def _format_csv_value(value: Any) -> str:
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise CandidateExportError("candidate CSV contains a non-finite number")
        return np.format_float_positional(number, unique=True, trim="-")
    return str(value)


def _write_csv(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    columns = list(CSV_BASE_COLUMNS) + [
        name for name in OPTIONAL_JOINED_FIELDS if name in arrays
    ]
    if set(columns) != set(arrays):
        raise CandidateExportError("candidate array/CSV column topology is inconsistent")
    rows = len(arrays[columns[0]])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for position in range(rows):
            writer.writerow([_format_csv_value(arrays[name][position]) for name in columns])
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = json.dumps(
        document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_exclusive(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise CandidateExportError(f"immutable output already exists: {destination}") from exc


def export_candidates(
    *,
    locked_oof_root: Path,
    evaluation_lock: Path,
    target_receipt: Path,
    output_dir: Path,
    receipt_output: Path,
    expected_rows: int = EXPECTED_VALID_REFERENCE_ROWS,
    expected_folds: int = len(FOLDS),
    expected_identities: int = EXPECTED_IDENTITIES,
    expected_seed_count: int = EXPECTED_SEED_COUNT,
    orchestrator_command: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify the complete locked join and publish one immutable CSV per seed."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (expected_rows, expected_folds, expected_identities, expected_seed_count)
    ):
        raise CandidateExportError("expected topology values must be positive integers")
    output_root = output_dir.expanduser().resolve()
    receipt_path = receipt_output.expanduser().resolve()
    if receipt_path.parent != output_root:
        raise CandidateExportError("export receipt must reside directly in --output-dir")
    if receipt_path.exists():
        raise CandidateExportError("immutable export receipt already exists; overwrite forbidden")

    (
        _lock,
        target_receipt_document,
        metrics,
        target,
        joined,
        target_schema,
        context,
    ) = _validate_context(
        locked_oof_root=locked_oof_root,
        evaluation_lock=evaluation_lock,
        target_receipt=target_receipt,
    )
    candidates, topology = _validate_and_build_candidates(
        receipt=target_receipt_document,
        metrics=metrics,
        target=target,
        joined=joined,
        expected_rows=expected_rows,
        expected_folds=expected_folds,
        expected_identities=expected_identities,
        expected_seed_count=expected_seed_count,
    )

    destinations = {
        seed: output_root / f"locked_hcs_oof_seed_{seed}.csv" for seed in candidates
    }
    preexisting = [path for path in (*destinations.values(), receipt_path) if path.exists()]
    if preexisting:
        raise CandidateExportError(
            f"immutable candidate output already exists; overwrite forbidden: {preexisting[0]}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    csv_temporaries: dict[int, Path] = {}
    receipt_temporary = _temporary_path(receipt_path)
    try:
        for seed, arrays in candidates.items():
            temporary = _temporary_path(destinations[seed])
            _write_csv(temporary, arrays)
            csv_temporaries[seed] = temporary

        exports: list[dict[str, Any]] = []
        for seed in sorted(candidates):
            temporary = csv_temporaries[seed]
            destination = destinations[seed]
            exports.append(
                {
                    "seed": seed,
                    "rows": expected_rows,
                    "csv": {
                        "path": str(destination),
                        "sha256": sha256_file(temporary),
                        "bytes": temporary.stat().st_size,
                    },
                    "columns": list(CSV_BASE_COLUMNS)
                    + [name for name in OPTIONAL_JOINED_FIELDS if name in candidates[seed]],
                    "column_schema": _array_schema(candidates[seed]),
                    "prediction_column_for_commercial_goal_audit": "final_rr_bpm",
                    "default_prediction_alias": "prediction_uncalibrated_bpm",
                    "uncertainty_column": (
                        "uncertainty_uncalibrated"
                        if "uncertainty_uncalibrated" in candidates[seed]
                        else None
                    ),
                }
            )
        receipt_document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": "retrospective_locked_hcs_oof_candidate_export_receipt",
            "commercial_claim_authorized": False,
            "prospective_confirmation_required": True,
            "target_and_joined_arrays_opened_only_after_evaluation_and_target_lock_verification": True,
            "all_input_file_hashes_verified": True,
            "all_target_array_hashes_verified_against_receipt": True,
            "joined_array_hashes_recorded": True,
            "one_csv_per_prespecified_fixed_seed": True,
            "result_selection_performed": False,
            "threshold_fitting_performed": False,
            "prediction_column_search_performed": False,
            "seed_ranking_or_suppression_performed": False,
            "export_overwrite_allowed": False,
            "inputs": context["bindings"],
            "input_array_schema": {
                "canonical_target": target_schema,
                "joined_oof": context["joined_schema"],
            },
            "topology_audit": topology,
            "exports": exports,
            "orchestrator_command": list(orchestrator_command),
        }
        receipt_document["content_sha256"] = canonical_json_sha256(receipt_document)
        _write_json(receipt_temporary, receipt_document)

        # Publish only after all source checks and all temporary outputs finish.
        # Hard-link publication is exclusive: an operator-owned path always wins.
        for seed in sorted(candidates):
            _publish_exclusive(csv_temporaries[seed], destinations[seed])
            destinations[seed].chmod(0o444)
        _publish_exclusive(receipt_temporary, receipt_path)
        receipt_path.chmod(0o444)
    finally:
        for temporary in (*csv_temporaries.values(), receipt_temporary):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    published = _read_json(receipt_path, "published export receipt", verify_content_hash=True)
    for record in published.get("exports", []):
        _verify_binding(record.get("csv"), relative_to=receipt_path.parent, label="published seed CSV")
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-oof-root", type=Path, default=DEFAULT_LOCKED_ROOT)
    parser.add_argument("--evaluation-lock", type=Path)
    parser.add_argument("--target-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_VALID_REFERENCE_ROWS)
    parser.add_argument("--expected-folds", type=int, default=len(FOLDS))
    parser.add_argument("--expected-identities", type=int, default=EXPECTED_IDENTITIES)
    parser.add_argument("--expected-seed-count", type=int, default=EXPECTED_SEED_COUNT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.locked_oof_root.expanduser().resolve()
    evaluation_lock = args.evaluation_lock or root / "evaluation_lock.json"
    target_receipt = (
        args.target_receipt or root / "canonical_locked_hcs_targets_receipt.json"
    )
    output_dir = args.output_dir.expanduser().resolve()
    receipt_output = args.receipt_output or output_dir / "candidate_export_receipt.json"
    try:
        result = export_candidates(
            locked_oof_root=root,
            evaluation_lock=evaluation_lock,
            target_receipt=target_receipt,
            output_dir=output_dir,
            receipt_output=receipt_output,
            expected_rows=args.expected_rows,
            expected_folds=args.expected_folds,
            expected_identities=args.expected_identities,
            expected_seed_count=args.expected_seed_count,
            orchestrator_command=[sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        )
    except CandidateExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
