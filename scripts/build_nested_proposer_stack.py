#!/usr/bin/env python3
"""Stitch five label-free custom-split predictions into one nested stack.

The outer-test partition is deliberately represented by unavailable rows.  This
program has no option for supplying an exploratory or test prediction and fails
closed if such an artifact appears in the selected discovery records.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_rr.split_authority import (  # noqa: E402
    canonical_content_sha256,
    load_identity_split_authority,
    sha256_file,
)


FORMAT_VERSION = 1
ROW_SCALARS = (
    "prediction",
    "map_prediction",
    "rr_std",
    "uncertainty",
    "quality",
    "alias_probability",
    "posterior_entropy",
    "spike_rate",
)
ROW_VECTORS = ("topk_rr", "topk_probability", "posterior_probability", "radar_weights")
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/manifests/plan.json"
)
DEFAULT_INDEX = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/nested_proposer/discovery_index.json"
)
DEFAULT_CACHE = PROJECT_ROOT / "artifacts/cache/rf32s"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object: {path}")
    return value


def _resolve(value: Any, *, relative_to: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_signature(arrays: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    digest.update(
        json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )
    return digest.hexdigest()


def load_canonical_metadata(cache_dir: Path) -> pd.DataFrame:
    """Load only metadata, preserving the cache manifest's canonical row order."""

    root_manifest = _load_json(cache_dir / "manifest.json", "cache manifest")
    sessions = root_manifest.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("cache manifest sessions must be an array")
    frames: list[pd.DataFrame] = []
    for item in sessions:
        if not isinstance(item, Mapping) or item.get("status") != "ok":
            continue
        session_id = item.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("cache manifest has an invalid successful session")
        frame = pd.read_csv(cache_dir / session_id / "metadata.csv")
        if len(frame) != int(item.get("window_count", -1)):
            raise RuntimeError(f"cache metadata row-count mismatch for {session_id}")
        frames.append(frame)
    if not frames:
        raise RuntimeError("cache has no successful session metadata")
    result = pd.concat(frames, ignore_index=True)
    required = {
        "session_id",
        "identity",
        "protocol",
        "window_number",
        "window_start_s",
        "window_end_s",
        "reference_valid",
        "rr_bpm",
    }
    missing = sorted(required - set(result.columns))
    if missing:
        raise RuntimeError(f"cache metadata is missing columns: {missing}")
    return result


def _flat_plan_outer(
    plan: Mapping[str, Any], *, outer_fold: int
) -> Mapping[str, Any]:
    """Normalize the hash-complete campaign plan's flat seed/unit matrix.

    The original discovery manifest plan stored one seed-independent five-unit
    group below ``outer_folds[str(fold)]``.  The sealed 90-unit campaign plan
    stores the same five manifests once per fixed seed in a flat ``units``
    array and records ``outer_folds`` as the ordered fold list.  Collapse only
    byte-identical semantic manifest groups; disagreement across seeds is a
    provenance error, not something the stack builder may choose between.
    """

    folds = plan.get("outer_folds")
    if (
        not isinstance(folds, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in folds)
        or len(folds) != len(set(folds))
        or outer_fold not in folds
    ):
        raise RuntimeError("nested flat plan outer_folds are invalid")
    seeds = plan.get("seeds")
    units = plan.get("units")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
        or len(seeds) != len(set(seeds))
        or not isinstance(units, list)
    ):
        raise RuntimeError("nested flat plan seeds/units are invalid")

    semantic_fields = (
        "manifest",
        "manifest_sha256",
        "manifest_content_sha256",
        "role",
    )
    canonical_group: list[Mapping[str, Any]] | None = None
    canonical_semantics: list[tuple[Any, ...]] | None = None
    for seed in seeds:
        group = [
            unit
            for unit in units
            if isinstance(unit, Mapping)
            and unit.get("outer_fold") == outer_fold
            and unit.get("seed") == seed
        ]
        if len(group) != 5:
            raise RuntimeError(
                f"nested flat plan must prescribe five units for fold {outer_fold}, seed {seed}"
            )
        semantics = sorted(
            (tuple(unit.get(field) for field in semantic_fields) for unit in group),
            key=lambda row: tuple(str(value) for value in row),
        )
        if canonical_semantics is None:
            canonical_group = group
            canonical_semantics = semantics
        elif semantics != canonical_semantics:
            raise RuntimeError(
                f"nested flat plan manifest semantics differ across seeds for fold {outer_fold}"
            )
    assert canonical_group is not None
    return {"outer_test_fold": outer_fold, "units": canonical_group}


def _validate_plan(plan: Mapping[str, Any], *, outer_fold: int) -> Mapping[str, Any]:
    if plan.get("schema_version") != 1:
        raise RuntimeError("nested plan schema_version must equal 1")
    if canonical_content_sha256(plan) != plan.get("content_sha256"):
        raise RuntimeError("nested plan canonical content hash mismatch")
    outer_folds = plan.get("outer_folds")
    if isinstance(outer_folds, Mapping):
        outer = outer_folds.get(str(outer_fold))
    elif isinstance(outer_folds, list):
        outer = _flat_plan_outer(plan, outer_fold=outer_fold)
    else:
        outer = None
    if not isinstance(outer, Mapping):
        raise RuntimeError(f"outer fold {outer_fold} is absent from nested plan")
    if int(outer.get("outer_test_fold", -1)) != outer_fold:
        raise RuntimeError("nested plan outer-test fold mismatch")
    units = outer.get("units")
    if not isinstance(units, list):
        raise RuntimeError("nested plan units must be an array")
    allowed = [unit for unit in units if unit.get("role") in {"hcs_train_oof", "hcs_validation"}]
    if len(allowed) != 5 or sum(unit.get("role") == "hcs_train_oof" for unit in allowed) != 4:
        raise RuntimeError("nested plan must prescribe four train-OOF and one validation unit")
    return outer


def _planned_prediction_fold(unit: Mapping[str, Any]) -> int:
    raw = unit.get("prediction_fold")
    if raw is not None:
        if isinstance(raw, bool):
            raise RuntimeError("nested plan prediction fold is invalid")
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("nested plan prediction fold is invalid") from exc
    stem = Path(str(unit.get("manifest", ""))).stem
    for prefix in ("inner_pred_", "validation_pred_"):
        if stem.startswith(prefix):
            suffix = stem.removeprefix(prefix)
            if suffix.isdigit():
                return int(suffix)
    raise RuntimeError("nested plan prediction fold is absent")


def _selected_records(
    index: Mapping[str, Any], *, plan_outer: Mapping[str, Any], outer_fold: int, seed: int
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if index.get("schema_version") != 1 or index.get("outer_test_opened") is not False:
        raise RuntimeError("discovery index is not an unopened version-1 discovery index")
    records = index.get("records")
    if not isinstance(records, list):
        raise RuntimeError("discovery index records must be an array")
    matching = [
        record
        for record in records
        if isinstance(record, Mapping)
        and int(record.get("outer_fold", -1)) == outer_fold
        and int(record.get("seed", -1)) == seed
    ]
    if any(
        "test_pred_" in str(record.get("manifest", ""))
        or "test" in str(record.get("role", "")).lower()
        for record in matching
    ):
        raise RuntimeError("outer-test prediction artifact entered discovery index")

    planned = {
        Path(str(unit["manifest"])).name: unit
        for unit in plan_outer["units"]
        if unit.get("role") in {"hcs_train_oof", "hcs_validation"}
    }
    observed: dict[str, Mapping[str, Any]] = {}
    for record in matching:
        name = Path(str(record.get("manifest", ""))).name
        if name not in planned:
            raise RuntimeError(f"unexpected discovery manifest for outer fold: {name}")
        if name in observed:
            raise RuntimeError(f"duplicate discovery unit: {name}")
        observed[name] = record
    missing = sorted(set(planned) - set(observed))
    if missing:
        raise RuntimeError(f"missing discovery units: {missing}")
    return [(planned[name], observed[name]) for name in sorted(planned)]


def _validate_checkpoint(
    record: Mapping[str, Any], *, authority: Any, seed: int
) -> tuple[Path, str, Path, Mapping[str, Any]]:
    binding = record.get("checkpoint")
    if not isinstance(binding, Mapping):
        raise RuntimeError("discovery record checkpoint binding is missing")
    path = _resolve(binding.get("path"), relative_to=PROJECT_ROOT)
    expected_hash = str(binding.get("sha256", ""))
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise RuntimeError(f"checkpoint hash mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 2 or checkpoint.get("model_type") != "snn":
        raise RuntimeError("nested proposer checkpoint must be a format-2 SNN")
    if int(checkpoint.get("fold", -1)) != authority.fold_id:
        raise RuntimeError("checkpoint/custom-manifest fold mismatch")
    if checkpoint.get("split_authority_provenance") != authority.checkpoint_provenance():
        raise RuntimeError("checkpoint custom split provenance mismatch")

    run_config_path = path.parent.parent / "run_config.json"
    run_config = _load_json(run_config_path, "run config")
    arguments = run_config.get("arguments")
    if not isinstance(arguments, Mapping) or int(arguments.get("seed", -1)) != seed:
        raise RuntimeError("checkpoint run seed mismatch")
    if arguments.get("identity_split_manifest_sha256") != authority.content_sha256:
        raise RuntimeError("checkpoint run custom-manifest hash mismatch")
    if checkpoint.get("run_signature") != run_config.get("run_signature"):
        raise RuntimeError("checkpoint/run-config signature mismatch")
    return path, expected_hash, run_config_path, run_config


def _scalar_string(archive: Mapping[str, Any], name: str) -> str:
    array = np.asarray(archive[name])
    if array.ndim != 0:
        raise RuntimeError(f"prediction field {name} must be scalar")
    return str(array.item())


def _validate_prediction(
    record: Mapping[str, Any],
    *,
    authority: Any,
    checkpoint_path: Path,
    checkpoint_hash: str,
    metadata: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    binding = record.get("all_window_prediction")
    if not isinstance(binding, Mapping):
        raise RuntimeError("discovery prediction binding is missing")
    path = _resolve(binding.get("path"), relative_to=PROJECT_ROOT)
    if "test_pred_" in str(path):
        raise RuntimeError("test prediction artifact is forbidden before policy lock")
    if path.name != "snn_prediction_all_windows.npz" or path.parent != checkpoint_path.parent:
        raise RuntimeError("prediction is not the checkpoint unit's all-window artifact")
    expected_hash = str(binding.get("sha256", ""))
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise RuntimeError(f"prediction hash mismatch: {path}")

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "cache_index",
            "session_id",
            "identity",
            "protocol",
            "window_number",
            "posterior_rr_grid_bpm",
            "checkpoint_sha256",
            "split_manifest_file_sha256",
            "split_manifest_content_sha256",
            "fold_assignments_sha256",
            "cache_manifest_sha256",
            "inference_signature_sha256",
            "strict_nested_prediction_role",
            "provenance_json",
            *ROW_SCALARS,
            *ROW_VECTORS,
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"prediction is missing fields: {missing}")
        if not bool(np.asarray(archive["strict_nested_prediction_role"]).item()):
            raise RuntimeError("prediction is not marked as a nested prediction role")
        if _scalar_string(archive, "checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError("prediction/checkpoint hash mismatch")
        expected_scalars = {
            "split_manifest_file_sha256": authority.manifest_file_sha256,
            "split_manifest_content_sha256": authority.content_sha256,
            "fold_assignments_sha256": authority.fold_assignments_sha256,
            "cache_manifest_sha256": authority.cache_manifest_sha256,
        }
        for field, expected in expected_scalars.items():
            if _scalar_string(archive, field) != expected:
                raise RuntimeError(f"prediction provenance mismatch: {field}")

        provenance = json.loads(_scalar_string(archive, "provenance_json"))
        if not isinstance(provenance, dict):
            raise RuntimeError("prediction provenance must be an object")
        recorded_signature = str(provenance.pop("inference_signature_sha256", ""))
        if recorded_signature != _canonical_hash(provenance):
            raise RuntimeError("prediction inference provenance signature mismatch")
        if _scalar_string(archive, "inference_signature_sha256") != recorded_signature:
            raise RuntimeError("prediction inference signature field mismatch")
        if provenance.get("checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError("prediction provenance checkpoint mismatch")
        if provenance.get("split_manifest_content_sha256") != authority.content_sha256:
            raise RuntimeError("prediction provenance split mismatch")
        if provenance.get("strict_nested_role") != "prediction":
            raise RuntimeError("prediction provenance has the wrong semantic role")
        if provenance.get("labels_forwarded_to_model") is not False:
            raise RuntimeError("prediction provenance permits target forwarding")

        index = np.asarray(archive["cache_index"], dtype=np.int64)
        if index.ndim != 1 or len(np.unique(index)) != len(index):
            raise RuntimeError("prediction cache indices are invalid or duplicated")
        expected = np.flatnonzero(
            np.isin(
                metadata["identity"].astype(str).to_numpy(),
                authority.prediction_identities,
            )
        )
        if not np.array_equal(index, expected):
            raise RuntimeError("prediction does not exactly cover manifest-owned cache rows")
        rows = metadata.iloc[index]
        semantic = {
            "session_id": rows["session_id"].astype(str).to_numpy(),
            "identity": rows["identity"].astype(str).to_numpy(),
            "protocol": rows["protocol"].astype(str).to_numpy(),
            "window_number": rows["window_number"].to_numpy(dtype=np.int64),
        }
        for field, expected_values in semantic.items():
            actual = np.asarray(archive[field])
            if actual.shape != expected_values.shape or not np.array_equal(
                actual.astype(expected_values.dtype), expected_values
            ):
                raise RuntimeError(f"prediction row semantics differ from cache: {field}")

        arrays = {field: np.asarray(archive[field]).copy() for field in (*ROW_SCALARS, *ROW_VECTORS)}
        arrays["cache_index"] = index
        arrays["posterior_rr_grid_bpm"] = np.asarray(
            archive["posterior_rr_grid_bpm"], dtype=np.float32
        ).copy()
    for field in ROW_SCALARS:
        if arrays[field].shape != (len(index),):
            raise RuntimeError(f"prediction row field has wrong shape: {field}")
    for field in ROW_VECTORS:
        if arrays[field].ndim != 2 or arrays[field].shape[0] != len(index):
            raise RuntimeError(f"prediction row-vector field has wrong shape: {field}")
    if not all(np.isfinite(arrays[field]).all() for field in (*ROW_SCALARS, *ROW_VECTORS)):
        raise RuntimeError("available proposer outputs contain non-finite values")
    return arrays, {"path": str(path), "sha256": expected_hash, "checkpoint": str(checkpoint_path)}


def build_nested_stack(
    *,
    discovery_index_path: Path,
    plan_path: Path,
    cache_dir: Path,
    outer_fold: int,
    seed: int,
) -> dict[str, Any]:
    metadata = load_canonical_metadata(cache_dir)
    plan = _load_json(plan_path, "nested plan")
    plan_outer = _validate_plan(plan, outer_fold=outer_fold)
    index_document = _load_json(discovery_index_path, "discovery index")
    selected = _selected_records(
        index_document, plan_outer=plan_outer, outer_fold=outer_fold, seed=seed
    )

    row_count = len(metadata)
    available = np.zeros(row_count, dtype=bool)
    role = np.full(row_count, "outer_test_unavailable", dtype="<U24")
    proposer_fold_id = np.full(row_count, -1, dtype=np.int16)
    output: dict[str, np.ndarray] = {}
    source_units: list[dict[str, Any]] = []
    dimensions: dict[str, tuple[int, ...]] | None = None
    posterior_grid: np.ndarray | None = None
    prediction_identities: set[str] = set()

    for planned, record in selected:
        manifest_path = _resolve(record.get("manifest"), relative_to=PROJECT_ROOT)
        expected_manifest = _resolve(planned.get("manifest"), relative_to=plan_path.parent)
        if manifest_path != expected_manifest or "test_pred_" in manifest_path.name:
            raise RuntimeError("discovery manifest path differs from nested plan")
        if sha256_file(manifest_path) != str(record.get("manifest_sha256", "")):
            raise RuntimeError(f"custom manifest file hash mismatch: {manifest_path}")
        manifest = _load_json(manifest_path, "custom split manifest")
        if manifest.get("content_sha256") != planned.get("manifest_content_sha256"):
            raise RuntimeError("custom manifest content differs from nested plan")
        authority = load_identity_split_authority(
            manifest_path, metadata=metadata, cache_dir=cache_dir
        )
        if set(authority.prediction_identities) & prediction_identities:
            raise RuntimeError("nested units duplicate prediction identities")
        prediction_identities.update(authority.prediction_identities)
        if record.get("role") != planned.get("role"):
            raise RuntimeError("discovery unit semantic role differs from nested plan")
        planned_prediction_fold = _planned_prediction_fold(planned)
        observed_prediction_folds = {
            int(authority.identity_to_fold[identity])
            for identity in authority.prediction_identities
        }
        if observed_prediction_folds != {planned_prediction_fold}:
            raise RuntimeError("custom manifest prediction ownership differs from nested plan")
        checkpoint_path, checkpoint_hash, run_config_path, _ = _validate_checkpoint(
            record, authority=authority, seed=seed
        )
        arrays, source = _validate_prediction(
            record,
            authority=authority,
            checkpoint_path=checkpoint_path,
            checkpoint_hash=checkpoint_hash,
            metadata=metadata,
        )
        positions = arrays.pop("cache_index")
        if available[positions].any():
            raise RuntimeError("nested units duplicate cache rows")
        observed_dimensions = {
            field: tuple(arrays[field].shape[1:]) for field in (*ROW_SCALARS, *ROW_VECTORS)
        }
        if dimensions is None:
            dimensions = observed_dimensions
            for field in ROW_SCALARS:
                output[field] = np.zeros(row_count, dtype=np.float32)
            for field in ROW_VECTORS:
                output[field] = np.zeros(
                    (row_count, *observed_dimensions[field]), dtype=np.float32
                )
            posterior_grid = arrays["posterior_rr_grid_bpm"]
        elif observed_dimensions != dimensions:
            raise RuntimeError("nested proposer output topology differs across units")
        if not np.array_equal(posterior_grid, arrays["posterior_rr_grid_bpm"]):
            raise RuntimeError("nested proposer posterior grids differ across units")
        for field in (*ROW_SCALARS, *ROW_VECTORS):
            output[field][positions] = arrays[field]
        available[positions] = True
        unit_role = str(planned["role"])
        role[positions] = unit_role
        proposer_fold_id[positions] = authority.fold_id
        source.update(
            manifest=str(manifest_path),
            manifest_file_sha256=authority.manifest_file_sha256,
            manifest_content_sha256=authority.content_sha256,
            role=unit_role,
            prediction_identities=list(authority.prediction_identities),
            prediction_rows=len(positions),
            run_config=str(run_config_path),
            run_config_sha256=sha256_file(run_config_path),
        )
        source_units.append(source)

    if dimensions is None or posterior_grid is None:
        raise RuntimeError("no nested proposer units were stitched")
    identities = metadata["identity"].astype(str).to_numpy()
    fold_map = json.loads(authority.fold_assignments_path.read_text(encoding="utf-8"))
    identity_to_fold = fold_map.get("identity_to_fold", fold_map)
    test_identities = {
        str(identity) for identity, fold in identity_to_fold.items() if int(fold) == outer_fold
    }
    expected_available = ~np.isin(identities, tuple(test_identities))
    if not np.array_equal(available, expected_available):
        missing = np.flatnonzero(expected_available & ~available)
        unexpected = np.flatnonzero(~expected_available & available)
        raise RuntimeError(
            "nested stack role cover mismatch "
            f"(missing_rows={missing[:10].tolist()}, unexpected_rows={unexpected[:10].tolist()})"
        )
    if set(identities[~available].tolist()) != test_identities:
        raise RuntimeError("outer-test identities are not exactly the unavailable partition")

    reference_valid = metadata["reference_valid"].to_numpy(dtype=bool) & available
    reference_rr = metadata["rr_bpm"].to_numpy(dtype=np.float32)
    reference_rr = np.where(reference_valid, reference_rr, np.nan).astype(np.float32)
    canonical_fold = np.asarray(
        [identity_to_fold[str(identity)] for identity in identities], dtype=np.int16
    )
    arrays_out: dict[str, Any] = {
        "cache_index": np.arange(row_count, dtype=np.int64),
        "session_id": metadata["session_id"].astype(str).to_numpy(dtype=np.str_),
        "identity": identities.astype(np.str_),
        "protocol": metadata["protocol"].astype(str).to_numpy(dtype=np.str_),
        "window_number": metadata["window_number"].to_numpy(dtype=np.int32),
        "window_start_s": metadata["window_start_s"].to_numpy(dtype=np.float64),
        "window_end_s": metadata["window_end_s"].to_numpy(dtype=np.float64),
        "fold": canonical_fold,
        "reference_valid": reference_valid,
        "reference_rr_bpm": reference_rr,
        "proposal_available": available,
        "nested_role": role,
        "proposer_fold_id": proposer_fold_id,
        **output,
        "posterior_rr_grid_bpm": posterior_grid,
        "outer_fold": np.asarray(outer_fold, dtype=np.int16),
        "seed": np.asarray(seed, dtype=np.int64),
        "strict_nested": np.asarray(True),
        "outer_test_opened": np.asarray(False),
    }
    provenance: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "retrospective_strict_nested_proposer_stack",
        "strict_nested": True,
        "outer_test_opened": False,
        "commercial_performance_claim_eligible": False,
        "target_consulted_for_stitching": False,
        "outer_fold": outer_fold,
        "seed": seed,
        "row_count": row_count,
        "available_rows": int(available.sum()),
        "outer_test_rows": int((~available).sum()),
        "outer_test_identities": sorted(test_identities),
        "discovery_index": {
            "path": str(discovery_index_path.resolve()),
            "sha256": sha256_file(discovery_index_path),
        },
        "nested_plan": {
            "path": str(plan_path.resolve()),
            "sha256": sha256_file(plan_path),
            "content_sha256": plan["content_sha256"],
        },
        "cache_manifest_sha256": authority.cache_manifest_sha256,
        "fold_assignments_sha256": authority.fold_assignments_sha256,
        "source_code_sha256": {
            str(Path(__file__).resolve().relative_to(PROJECT_ROOT)): sha256_file(
                Path(__file__).resolve()
            ),
            "src/snn_rr/split_authority.py": sha256_file(
                PROJECT_ROOT / "src/snn_rr/split_authority.py"
            ),
        },
        "source_units": source_units,
    }
    signature = _array_signature(arrays_out, provenance)
    provenance["content_signature_sha256"] = signature
    arrays_out["content_signature_sha256"] = np.asarray(signature)
    arrays_out["provenance_json"] = np.asarray(
        json.dumps(provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    return arrays_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"output exists; pass --force to replace: {output_path}")
    arrays = build_nested_stack(
        discovery_index_path=args.discovery_index.expanduser().resolve(),
        plan_path=args.plan.expanduser().resolve(),
        cache_dir=args.cache_dir.expanduser().resolve(),
        outer_fold=args.outer_fold,
        seed=args.seed,
    )
    _atomic_npz(output_path, arrays)
    return {
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "content_signature_sha256": str(arrays["content_signature_sha256"]),
        "rows": len(arrays["cache_index"]),
        "available_rows": int(np.asarray(arrays["proposal_available"]).sum()),
        "outer_test_opened": False,
        "strict_nested": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
