#!/usr/bin/env python3
"""Evaluate the sealed seven-mask HCS robustness matrix without mask selection.

The 126-unit radar-mask plan, every target-free unit receipt/output, and the
complete mask seal are validated before the primary evaluation lock is allowed
to authorize target access.  The already-frozen primary evaluation spec and
canonical target receipt then govern an exact semantic join.  All seven masks
are reported for each fixed seed independently; no mask or seed is selected,
ranked, pooled, or suppressed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import secrets
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_locked_hcs_oof as PRIMARY  # noqa: E402
import run_locked_hcs_radar_mask_campaign as MASK_CAMPAIGN  # noqa: E402


SCHEMA_VERSION = 1
FOLDS = tuple(range(6))
MASKS: dict[str, tuple[bool, bool, bool]] = dict(MASK_CAMPAIGN.MASKS)
MASK_NAMES = tuple(MASKS)
BASELINE_MASK = "radars_123"
EXPECTED_UNITS = len(FOLDS) * len(PRIMARY.FIXED_SEEDS) * len(MASKS)
SEALED_FIELDS = {
    "cache_index",
    "outer_fold",
    "seed",
    "fallback_rr_bpm",
    "source_rr_bpm",
    "final_rr_bpm",
    "applied_pull",
    "target_joined",
}
DEGRADATION_SIGN = {
    "mae": 1.0,
    "identity_macro_mae": 1.0,
    "rmse": 1.0,
    "within_2_fraction": -1.0,
    "over_5_fraction": 1.0,
    "tail_25_35_mae": 1.0,
}
CSV_COLUMNS = (
    "record_type",
    "seed",
    "radar_mask",
    "baseline_mask",
    "scope",
    "stratum_type",
    "stratum",
    "rows",
    "identities",
    "tail_25_35_rows",
    *PRIMARY.METRIC_FIELDS,
    *(f"degradation_{name}" for name in PRIMARY.METRIC_FIELDS),
    "improved_fraction",
    "tied_fraction",
    "worsened_fraction",
)

PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MASK_ROOT = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_radar_masks"
)
DEFAULT_PRIMARY_ROOT = PRIMARY.DEFAULT_LOCKED_ROOT


class LockedRadarMaskEvaluationError(RuntimeError):
    """A seal, provenance, topology, semantic-join, or publication check failed."""


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    try:
        return (
            Path(str(left["path"])).resolve() == Path(str(right["path"])).resolve()
            and str(left["sha256"]) == str(right["sha256"])
            and int(left.get("bytes", Path(str(left["path"])).stat().st_size))
            == int(right.get("bytes", Path(str(right["path"])).stat().st_size))
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _read_json(
    path: Path, label: str, *, require_content_hash: bool = False
) -> dict[str, Any]:
    try:
        return PRIMARY._read_json(  # noqa: SLF001
            path, label, require_content_hash=require_content_hash
        )
    except Exception as exc:
        raise LockedRadarMaskEvaluationError(str(exc)) from exc


def _verify_binding(
    raw: Any,
    *,
    relative_to: Path,
    label: str,
    memo: dict[Path, tuple[str, int, int, int]] | None = None,
) -> dict[str, Any]:
    try:
        shaped = PRIMARY._binding_shape(  # noqa: SLF001
            raw, relative_to=relative_to, label=label
        )
    except Exception as exc:
        raise LockedRadarMaskEvaluationError(str(exc)) from exc
    path = Path(shaped["path"])
    if memo is not None and path in memo:
        actual_hash, actual_size, recorded_mtime_ns, recorded_inode = memo[path]
        observed_stat = path.stat()
        if (
            observed_stat.st_size != actual_size
            or observed_stat.st_mtime_ns != recorded_mtime_ns
            or observed_stat.st_ino != recorded_inode
        ):
            raise LockedRadarMaskEvaluationError(
                f"bound file changed during radar-mask validation: {label} ({path})"
            )
    else:
        if not path.is_file():
            raise LockedRadarMaskEvaluationError(f"bound file is absent: {label} ({path})")
        observed_stat = path.stat()
        actual_size = observed_stat.st_size
        actual_hash = PRIMARY.sha256_file(path)
        if memo is not None:
            memo[path] = (
                actual_hash,
                actual_size,
                observed_stat.st_mtime_ns,
                observed_stat.st_ino,
            )
    if "bytes" in shaped and shaped["bytes"] != actual_size:
        raise LockedRadarMaskEvaluationError(f"file byte-size binding mismatch: {label}")
    if shaped["sha256"] != actual_hash:
        raise LockedRadarMaskEvaluationError(f"file SHA-256 binding mismatch: {label} ({path})")
    return {"path": str(path), "sha256": actual_hash, "bytes": actual_size}


def _verify_binding_tree(
    value: Any,
    *,
    relative_to: Path,
    label: str,
    memo: dict[Path, tuple[str, int, int, int]] | None = None,
) -> int:
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            _verify_binding(value, relative_to=relative_to, label=label, memo=memo)
            return 1
        return sum(
            _verify_binding_tree(
                child,
                relative_to=relative_to,
                label=f"{label}.{key}",
                memo=memo,
            )
            for key, child in value.items()
        )
    if isinstance(value, list):
        return sum(
            _verify_binding_tree(
                child,
                relative_to=relative_to,
                label=f"{label}[{position}]",
                memo=memo,
            )
            for position, child in enumerate(value)
        )
    return 0


def _scalar_integer(value: Any, *, label: str) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise LockedRadarMaskEvaluationError(f"{label} must be an integer scalar")
    return int(array.item())


def _load_mask_prediction(
    path: Path, *, fold: int, seed: int, label: str
) -> dict[str, np.ndarray]:
    """Open only a receipt-sealed target-free prediction archive."""

    try:
        arrays = MASK_CAMPAIGN._read_label_free_npz(  # noqa: SLF001
            path, label=label, required=set(SEALED_FIELDS)
        )
    except Exception as exc:
        raise LockedRadarMaskEvaluationError(str(exc)) from exc
    if set(arrays) != SEALED_FIELDS:
        raise LockedRadarMaskEvaluationError(f"{label} sealed schema is not exact")
    index = np.asarray(arrays["cache_index"])
    if index.dtype != np.int64 or index.ndim != 1 or len(index) == 0:
        raise LockedRadarMaskEvaluationError(f"{label} cache_index is invalid")
    if (index < 0).any() or not np.all(index[1:] > index[:-1]):
        raise LockedRadarMaskEvaluationError(
            f"{label} cache_index must be non-negative, unique, and sorted"
        )
    if _scalar_integer(arrays["outer_fold"], label=f"{label} fold") != fold:
        raise LockedRadarMaskEvaluationError(f"{label} outer-fold identity mismatch")
    if _scalar_integer(arrays["seed"], label=f"{label} seed") != seed:
        raise LockedRadarMaskEvaluationError(f"{label} seed identity mismatch")
    joined = np.asarray(arrays["target_joined"])
    if joined.ndim != 0 or joined.dtype.kind != "b" or bool(joined.item()):
        raise LockedRadarMaskEvaluationError(f"{label} is not target-free")
    for field in ("fallback_rr_bpm", "source_rr_bpm", "final_rr_bpm", "applied_pull"):
        value = np.asarray(arrays[field])
        if value.dtype.kind not in "fc" or value.shape != index.shape:
            raise LockedRadarMaskEvaluationError(f"{label} field shape/type: {field}")
        if not np.isfinite(value).all():
            raise LockedRadarMaskEvaluationError(f"{label} field is non-finite: {field}")
    for field in ("fallback_rr_bpm", "source_rr_bpm", "final_rr_bpm"):
        if (np.asarray(arrays[field]) <= 0).any():
            raise LockedRadarMaskEvaluationError(f"{label} prediction is non-positive: {field}")
    return arrays


def _bit_equal(left: Any, right: Any) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _canonical_masks(raw: Any) -> bool:
    return raw == [
        {"name": name, "pattern": list(pattern)} for name, pattern in MASKS.items()
    ]


def _validate_mask_campaign(
    *, mask_root: Path, evaluation_spec: Mapping[str, Any]
) -> tuple[
    dict[tuple[int, int, str], dict[str, np.ndarray]],
    dict[str, Any],
]:
    """Validate the complete target-free 126-unit graph before target access."""

    root = mask_root.expanduser().resolve()
    binding_memo: dict[Path, tuple[str, int, int, int]] = {}
    seal_path = root / "complete_seal.json"
    if not seal_path.is_file():
        raise LockedRadarMaskEvaluationError(
            "complete 126-unit radar-mask seal is absent; evaluation forbidden"
        )
    seal = _read_json(seal_path, "radar-mask complete seal")
    population = evaluation_spec["population"]
    expected_seeds = list(population["fixed_seeds"])
    if (
        seal.get("schema_version") != SCHEMA_VERSION
        or seal.get("classification")
        != "locked_hcs_all_seven_radar_mask_predictions_sealed"
        or seal.get("folds") != list(FOLDS)
        or seal.get("seeds") != expected_seeds
        or not _canonical_masks(seal.get("radar_masks"))
        or seal.get("unit_count") != EXPECTED_UNITS
        or seal.get("complete_matrix") is not True
        or seal.get("target_or_label_artifact_opened_before_seal") is not False
        or seal.get("best_mask_selection_performed") is not False
        or seal.get("all_masks_retained_as_fixed_conditions") is not True
        or seal.get("evaluation_authorized") is not True
    ):
        raise LockedRadarMaskEvaluationError("radar-mask complete seal invariant is invalid")
    plan_binding = _verify_binding(
        seal.get("plan"),
        relative_to=seal_path.parent,
        label="radar-mask plan",
        memo=binding_memo,
    )
    plan_path = Path(plan_binding["path"])
    plan = _read_json(plan_path, "radar-mask plan")
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("classification") != "locked_hcs_seven_radar_mask_label_free_plan"
        or plan.get("folds") != list(FOLDS)
        or plan.get("seeds") != expected_seeds
        or not _canonical_masks(plan.get("radar_masks"))
        or plan.get("primary_unit_count") != len(FOLDS) * len(expected_seeds)
        or plan.get("unit_count") != EXPECTED_UNITS
        or plan.get("target_or_label_artifact_bound") is not False
        or plan.get("evaluation_permitted_before_complete_seal") is not False
    ):
        raise LockedRadarMaskEvaluationError("radar-mask plan invariant is invalid")
    contract = plan.get("mask_selection_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("mask_order_fixed_before_inference") != list(MASK_NAMES)
        or contract.get("best_mask_selection_allowed") is not False
        or contract.get("target_or_metric_dependent_mask_selection_allowed") is not False
        or contract.get("all_masks_are_required_conditions") is not True
        or not isinstance(contract.get("radars_123_primary_parity"), Mapping)
        or contract["radars_123_primary_parity"].get("required") is not True
    ):
        raise LockedRadarMaskEvaluationError("radar-mask no-selection/parity contract is invalid")
    if seal.get("mask_selection_contract") != contract:
        raise LockedRadarMaskEvaluationError("plan/seal mask-selection contract mismatch")
    execution = plan.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("device") != "cpu"
        or execution.get("amp") is not False
        or execution.get("shell") is not False
        or execution.get("publication") != "atomic_unit_directory_rename"
    ):
        raise LockedRadarMaskEvaluationError("radar-mask execution contract is invalid")

    preexecution_binding = _verify_binding(
        seal.get("preexecution_lock"),
        relative_to=seal_path.parent,
        label="radar-mask preexecution lock",
        memo=binding_memo,
    )
    preexecution_path = Path(preexecution_binding["path"])
    preexecution = _read_json(preexecution_path, "radar-mask preexecution lock")
    if (
        preexecution.get("schema_version") != SCHEMA_VERSION
        or preexecution.get("classification")
        != "locked_hcs_radar_mask_preexecution_input_seal"
        or preexecution.get("unit_count") != EXPECTED_UNITS
        or preexecution.get("target_or_label_artifact_opened") is not False
        or preexecution.get("evaluation_authorized") is not False
    ):
        raise LockedRadarMaskEvaluationError("radar-mask preexecution lock is invalid")
    recorded_plan = _verify_binding(
        preexecution.get("plan"),
        relative_to=preexecution_path.parent,
        label="preexecution radar-mask plan",
        memo=binding_memo,
    )
    if not _same_binding(recorded_plan, plan_binding):
        raise LockedRadarMaskEvaluationError("preexecution lock is bound to another plan")

    plan_units_raw = plan.get("units")
    seal_units_raw = seal.get("units")
    if (
        not isinstance(plan_units_raw, list)
        or len(plan_units_raw) != EXPECTED_UNITS
        or not isinstance(seal_units_raw, list)
        or len(seal_units_raw) != EXPECTED_UNITS
    ):
        raise LockedRadarMaskEvaluationError("radar-mask unit matrix is incomplete")
    expected_keys = {
        (fold, seed, mask)
        for seed in expected_seeds
        for fold in FOLDS
        for mask in MASK_NAMES
    }
    plan_units: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for raw in plan_units_raw:
        if not isinstance(raw, Mapping):
            raise LockedRadarMaskEvaluationError("radar-mask plan unit must be an object")
        key = (
            int(raw.get("outer_fold", -1)),
            int(raw.get("seed", -1)),
            str(raw.get("radar_mask", "")),
        )
        expected_id = MASK_CAMPAIGN._mask_unit_id(key[0], key[1], key[2])  # noqa: SLF001
        if (
            key not in expected_keys
            or key in plan_units
            or raw.get("unit_id") != expected_id
            or raw.get("radar_mask_pattern") != list(MASKS[key[2]])
        ):
            raise LockedRadarMaskEvaluationError(f"invalid/duplicate mask plan unit: {key}")
        commands = raw.get("commands")
        if commands:
            try:
                MASK_CAMPAIGN._validate_command_contract(raw, root)  # noqa: SLF001
            except Exception as exc:
                raise LockedRadarMaskEvaluationError(str(exc)) from exc
        plan_units[key] = raw
    if set(plan_units) != expected_keys:
        raise LockedRadarMaskEvaluationError("radar-mask plan is not the exact Cartesian matrix")

    seal_units: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for raw in seal_units_raw:
        if not isinstance(raw, Mapping):
            raise LockedRadarMaskEvaluationError("radar-mask seal unit must be an object")
        key = (
            int(raw.get("outer_fold", -1)),
            int(raw.get("seed", -1)),
            str(raw.get("radar_mask", "")),
        )
        if (
            key not in expected_keys
            or key in seal_units
            or raw.get("unit_id") != plan_units[key].get("unit_id")
            or raw.get("radar_mask_pattern") != list(MASKS[key[2]])
        ):
            raise LockedRadarMaskEvaluationError(f"invalid/duplicate mask seal unit: {key}")
        seal_units[key] = raw
    if set(seal_units) != expected_keys:
        raise LockedRadarMaskEvaluationError("radar-mask seal is not the exact Cartesian matrix")

    verified_binding_count = (
        _verify_binding_tree(
            plan,
            relative_to=plan_path.parent,
            label="radar-mask plan",
            memo=binding_memo,
        )
        + _verify_binding_tree(
            preexecution,
            relative_to=preexecution_path.parent,
            label="radar-mask preexecution lock",
            memo=binding_memo,
        )
        + _verify_binding_tree(
            seal,
            relative_to=seal_path.parent,
            label="radar-mask seal",
            memo=binding_memo,
        )
    )
    predictions: dict[tuple[int, int, str], dict[str, np.ndarray]] = {}
    primary_indices: dict[tuple[int, int], np.ndarray] = {}
    for key in sorted(expected_keys, key=lambda value: (value[1], value[0], MASK_NAMES.index(value[2]))):
        fold, seed, mask = key
        unit = plan_units[key]
        sealed_unit = seal_units[key]
        receipt_binding = _verify_binding(
            sealed_unit.get("receipt"),
            relative_to=seal_path.parent,
            label=f"mask receipt {fold}/{seed}/{mask}",
            memo=binding_memo,
        )
        receipt_path = Path(receipt_binding["path"])
        receipt = _read_json(
            receipt_path,
            f"mask receipt {fold}/{seed}/{mask}",
            require_content_hash=True,
        )
        if (
            receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("classification")
            != "locked_hcs_radar_mask_label_free_unit_receipt"
            or receipt.get("unit_id") != unit.get("unit_id")
            or receipt.get("outer_fold") != fold
            or receipt.get("seed") != seed
            or receipt.get("radar_mask") != mask
            or receipt.get("radar_mask_pattern") != list(MASKS[mask])
            or receipt.get("target_or_label_fields_read") is not False
            or receipt.get("target_or_label_fields_present") is not False
            or receipt.get("evaluation_performed") is not False
            or receipt.get("source_semantics") != "frozen_no_action_placeholder"
        ):
            raise LockedRadarMaskEvaluationError(f"mask receipt invariant: {fold}/{seed}/{mask}")
        runtime_guards = receipt.get("runtime_guards")
        if (
            not isinstance(runtime_guards, Mapping)
            or runtime_guards.get("device") != "cpu"
            or runtime_guards.get("amp") is not False
            or runtime_guards.get("shell") is not False
        ):
            raise LockedRadarMaskEvaluationError(
                f"mask receipt runtime guard invariant: {fold}/{seed}/{mask}"
            )
        receipt_plan = _verify_binding(
            receipt.get("plan"),
            relative_to=receipt_path.parent,
            label="receipt mask plan",
            memo=binding_memo,
        )
        if not _same_binding(receipt_plan, plan_binding):
            raise LockedRadarMaskEvaluationError(f"mask receipt plan mismatch: {key}")
        if (
            receipt.get("inputs") != unit.get("inputs")
            or receipt.get("primary") != unit.get("primary")
            or receipt.get("commands") != unit.get("commands")
        ):
            raise LockedRadarMaskEvaluationError(
                f"mask receipt input/primary/command provenance mismatch: {key}"
            )
        verified_binding_count += _verify_binding_tree(
            receipt,
            relative_to=receipt_path.parent,
            label=f"mask receipt {key}",
            memo=binding_memo,
        )
        outputs = receipt.get("outputs")
        if not isinstance(outputs, Mapping) or "sealed_prediction" not in outputs:
            raise LockedRadarMaskEvaluationError(f"mask receipt output is absent: {key}")
        receipt_prediction = _verify_binding(
            outputs["sealed_prediction"],
            relative_to=receipt_path.parent,
            label=f"receipt mask prediction {key}",
            memo=binding_memo,
        )
        seal_prediction = _verify_binding(
            sealed_unit.get("sealed_prediction"),
            relative_to=seal_path.parent,
            label=f"seal mask prediction {key}",
            memo=binding_memo,
        )
        if not _same_binding(receipt_prediction, seal_prediction):
            raise LockedRadarMaskEvaluationError(f"receipt/seal prediction mismatch: {key}")
        for artifact_name in ("proposer_prediction", "raw_source_prediction"):
            if artifact_name in sealed_unit or artifact_name in outputs:
                if artifact_name not in sealed_unit or artifact_name not in outputs:
                    raise LockedRadarMaskEvaluationError(
                        f"receipt/seal artifact topology mismatch: {key}/{artifact_name}"
                    )
                receipt_artifact = _verify_binding(
                    outputs[artifact_name],
                    relative_to=receipt_path.parent,
                    label=f"receipt {artifact_name} {key}",
                    memo=binding_memo,
                )
                seal_artifact = _verify_binding(
                    sealed_unit[artifact_name],
                    relative_to=seal_path.parent,
                    label=f"seal {artifact_name} {key}",
                    memo=binding_memo,
                )
                if not _same_binding(receipt_artifact, seal_artifact):
                    raise LockedRadarMaskEvaluationError(
                        f"receipt/seal artifact mismatch: {key}/{artifact_name}"
                    )
        if sealed_unit.get("receipt_content_sha256") != receipt.get("content_sha256"):
            raise LockedRadarMaskEvaluationError(f"receipt content hash mismatch in seal: {key}")
        arrays = _load_mask_prediction(
            Path(seal_prediction["path"]), fold=fold, seed=seed, label=f"mask prediction {key}"
        )

        primary_raw = unit.get("primary")
        if not isinstance(primary_raw, Mapping):
            raise LockedRadarMaskEvaluationError(f"mask plan primary provenance is absent: {key}")
        primary_binding = _verify_binding(
            primary_raw.get("sealed_prediction"),
            relative_to=plan_path.parent,
            label=f"primary sealed prediction {fold}/{seed}",
            memo=binding_memo,
        )
        primary_arrays = _load_mask_prediction(
            Path(primary_binding["path"]),
            fold=fold,
            seed=seed,
            label=f"primary prediction {fold}/{seed}",
        )
        if not _bit_equal(arrays["cache_index"], primary_arrays["cache_index"]):
            raise LockedRadarMaskEvaluationError(f"mask changed primary cache ownership: {key}")
        prior = primary_indices.get((fold, seed))
        if prior is None:
            primary_indices[(fold, seed)] = np.asarray(arrays["cache_index"]).copy()
        elif not _bit_equal(prior, arrays["cache_index"]):
            raise LockedRadarMaskEvaluationError(f"mask cache topology differs: {key}")
        if mask == BASELINE_MASK:
            for field in SEALED_FIELDS:
                if not _bit_equal(arrays[field], primary_arrays[field]):
                    raise LockedRadarMaskEvaluationError(
                        f"radars_123 primary parity failed: {fold}/{seed}/{field}"
                    )
            parity = receipt.get("radars_123_bit_exact_primary_comparison")
            if (
                not isinstance(parity, Mapping)
                or parity.get("required") is not True
                or parity.get("performed") is not True
            ):
                raise LockedRadarMaskEvaluationError(
                    f"radars_123 parity receipt is incomplete: {fold}/{seed}"
                )

        # Re-open optional proposer/source archives through the orchestrator's
        # target-field guard.  Their complete bytes were already hash-verified.
        for output_name, required in (
            ("proposer_prediction", MASK_CAMPAIGN.CORE_PROPOSER_FIELDS),
            ("raw_source_prediction", MASK_CAMPAIGN.RAW_REQUIRED),
        ):
            if output_name in outputs:
                binding = _verify_binding(
                    outputs[output_name],
                    relative_to=receipt_path.parent,
                    label=f"receipt {output_name} {key}",
                    memo=binding_memo,
                )
                try:
                    MASK_CAMPAIGN._read_label_free_npz(  # noqa: SLF001
                        Path(binding["path"]),
                        label=f"mask {output_name} {key}",
                        required=set(required),
                    )
                except Exception as exc:
                    raise LockedRadarMaskEvaluationError(str(exc)) from exc
        if {"proposer_prediction", "raw_source_prediction"} <= set(outputs):
            proposer_verified = _verify_binding(
                outputs["proposer_prediction"],
                relative_to=receipt_path.parent,
                label=f"validated proposer prediction {key}",
                memo=binding_memo,
            )
            raw_verified = _verify_binding(
                outputs["raw_source_prediction"],
                relative_to=receipt_path.parent,
                label=f"validated raw source prediction {key}",
                memo=binding_memo,
            )
            try:
                comparisons = MASK_CAMPAIGN._validate_mask_outputs(  # noqa: SLF001
                    proposer_path=Path(proposer_verified["path"]),
                    raw_path=Path(raw_verified["path"]),
                    sealed_path=Path(seal_prediction["path"]),
                    unit=unit,
                    compare_full=mask == BASELINE_MASK,
                )
            except Exception as exc:
                raise LockedRadarMaskEvaluationError(str(exc)) from exc
            if receipt.get("radars_123_bit_exact_primary_comparison") != comparisons:
                raise LockedRadarMaskEvaluationError(
                    f"receipt/orchestrator mask comparison mismatch: {key}"
                )
        predictions[key] = arrays

    # Same fold ownership must hold across all three fixed seeds.
    for fold in FOLDS:
        reference = primary_indices[(fold, expected_seeds[0])]
        for seed in expected_seeds[1:]:
            if not _bit_equal(reference, primary_indices[(fold, seed)]):
                raise LockedRadarMaskEvaluationError(
                    f"radar-mask cache ownership differs across seeds: fold {fold}"
                )
    primary = plan.get("primary")
    if not isinstance(primary, Mapping):
        raise LockedRadarMaskEvaluationError("mask plan primary binding group is absent")
    primary_seal = _verify_binding(
        primary.get("predictions_seal"),
        relative_to=plan_path.parent,
        label="mask-plan primary predictions seal",
        memo=binding_memo,
    )
    seal_primary = _verify_binding(
        seal.get("primary_predictions_seal"),
        relative_to=seal_path.parent,
        label="mask-seal primary predictions seal",
        memo=binding_memo,
    )
    if not _same_binding(primary_seal, seal_primary):
        raise LockedRadarMaskEvaluationError(
            "radar-mask plan/seal bind different primary prediction seals"
        )
    # Rehash every unique bound input once more at the authorization boundary.
    # This preserves the performance benefit of deduplicating repeated plan /
    # receipt bindings while detecting any mutation during the full audit.
    for path, (expected_hash, expected_size, expected_mtime_ns, expected_inode) in (
        binding_memo.items()
    ):
        observed = path.stat()
        if (
            observed.st_size != expected_size
            or observed.st_mtime_ns != expected_mtime_ns
            or observed.st_ino != expected_inode
            or PRIMARY.sha256_file(path) != expected_hash
        ):
            raise LockedRadarMaskEvaluationError(
                f"bound target-free artifact changed during complete audit: {path}"
            )
    context = {
        "bindings": {
            "radar_mask_complete_seal": PRIMARY.bind_file(seal_path),
            "radar_mask_plan": plan_binding,
            "radar_mask_preexecution_lock": preexecution_binding,
            "primary_predictions_seal": primary_seal,
        },
        "verified_bound_artifact_count": verified_binding_count,
        "unique_bound_artifacts_rehashed_at_authorization_boundary": len(binding_memo),
        "unit_count": len(predictions),
        "folds": list(FOLDS),
        "seeds": expected_seeds,
        "radar_masks": [
            {"name": name, "pattern": list(pattern)} for name, pattern in MASKS.items()
        ],
        "complete_exact_cartesian_matrix": True,
        "all_units_target_free": True,
        "best_mask_selection_performed": False,
        "radars_123_target_free_primary_parity_verified": True,
    }
    return predictions, context


def _load_primary_context(
    *,
    primary_root: Path,
    primary_evaluation_lock: Path,
    target_receipt: Path,
    spec: Mapping[str, Any],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    """The only boundary that may authorize/open canonical target arrays."""

    population = spec["population"]
    try:
        return PRIMARY._validate_context(  # noqa: SLF001
            locked_oof_root=primary_root,
            evaluation_lock=primary_evaluation_lock,
            target_receipt=target_receipt,
            expected_rows=int(population["valid_reference_rows_per_seed"]),
            expected_identities=int(population["physical_identity_count"]),
            expected_seeds=population["fixed_seeds"],
        )
    except Exception as exc:
        raise LockedRadarMaskEvaluationError(str(exc)) from exc


def _join_masks(
    *,
    target_frames: Mapping[int, Mapping[str, np.ndarray]],
    predictions: Mapping[tuple[int, int, str], Mapping[str, np.ndarray]],
) -> tuple[dict[int, dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    joined: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    parity: dict[str, Any] = {}
    for seed, target_frame_raw in sorted(target_frames.items()):
        target_frame = {name: np.asarray(value) for name, value in target_frame_raw.items()}
        target_index = np.asarray(target_frame["cache_index"], dtype=np.int64)
        target_fold = np.asarray(target_frame["outer_fold"], dtype=np.int64)
        if not np.all(target_index[1:] > target_index[:-1]):
            raise LockedRadarMaskEvaluationError("primary valid target frame is not cache-sorted")
        per_mask: dict[str, dict[str, np.ndarray]] = {}
        for mask in MASK_NAMES:
            indices: list[np.ndarray] = []
            fields: dict[str, list[np.ndarray]] = {
                "fallback_rr_bpm": [],
                "source_rr_bpm": [],
                "final_rr_bpm": [],
            }
            fold_values: list[np.ndarray] = []
            for fold in FOLDS:
                arrays = predictions[(fold, seed, mask)]
                index = np.asarray(arrays["cache_index"], dtype=np.int64)
                indices.append(index)
                fold_values.append(np.full(len(index), fold, dtype=np.int64))
                for field in fields:
                    fields[field].append(np.asarray(arrays[field]))
            full_index = np.concatenate(indices)
            full_fold = np.concatenate(fold_values)
            order = np.argsort(full_index, kind="stable")
            full_index = full_index[order]
            full_fold = full_fold[order]
            if len(np.unique(full_index)) != len(full_index):
                raise LockedRadarMaskEvaluationError(f"mask folds overlap: {seed}/{mask}")
            positions = np.searchsorted(full_index, target_index)
            if (
                (positions >= len(full_index)).any()
                or not np.array_equal(full_index[positions], target_index)
                or not np.array_equal(full_fold[positions], target_fold)
            ):
                raise LockedRadarMaskEvaluationError(
                    f"mask/target semantic cache-fold join mismatch: {seed}/{mask}"
                )
            frame = {name: value.copy() for name, value in target_frame.items()}
            for field, parts in fields.items():
                full_value = np.concatenate(parts)[order]
                frame[field] = full_value[positions]
            frame["radar_mask"] = np.full(len(target_index), mask, dtype=np.str_)
            per_mask[mask] = frame
        primary = target_frame
        full = per_mask[BASELINE_MASK]
        array_checks = {
            field: _bit_equal(full[field], primary[field])
            for field in ("fallback_rr_bpm", "source_rr_bpm", "final_rr_bpm")
        }
        metric_primary = PRIMARY._metric_summary(  # noqa: SLF001
            primary["target_rr_bpm"], primary["final_rr_bpm"], primary["identity"]
        )
        metric_full = PRIMARY._metric_summary(  # noqa: SLF001
            full["target_rr_bpm"], full["final_rr_bpm"], full["identity"]
        )
        metric_equal = PRIMARY.canonical_json_sha256(metric_primary) == PRIMARY.canonical_json_sha256(
            metric_full
        )
        passed = all(array_checks.values()) and metric_equal
        if not passed:
            raise LockedRadarMaskEvaluationError(
                f"radars_123 primary semantic/metric parity gate failed: seed {seed}"
            )
        parity[str(seed)] = {
            "passed": True,
            "array_bit_exact": array_checks,
            "locked_final_metrics_exact": metric_equal,
            "rows": len(target_index),
        }
        joined[seed] = per_mask
    return joined, {
        "required": True,
        "all_fixed_seeds_passed": all(item["passed"] for item in parity.values()),
        "per_seed": parity,
        "failure_policy": "fail_closed_no_robustness_report_publication",
    }


def _metric_bundle(
    frame: Mapping[str, np.ndarray]
) -> tuple[dict[str, Any], list[tuple[str, str, str, dict[str, Any]]]]:
    target = np.asarray(frame["target_rr_bpm"], float)
    prediction = np.asarray(frame["final_rr_bpm"], float)
    identity = np.asarray(frame["identity"]).astype(str)
    nonoverlap = PRIMARY._greedy_nonoverlap_mask(frame)  # noqa: SLF001
    phases = {
        str(phase): np.asarray(frame["window_number"], dtype=np.int64) % 8 == phase
        for phase in range(8)
    }
    strata = PRIMARY._strata(frame)  # noqa: SLF001
    full = PRIMARY._metric_summary(target, prediction, identity)  # noqa: SLF001
    greedy = PRIMARY._metric_summary(  # noqa: SLF001
        target[nonoverlap], prediction[nonoverlap], identity[nonoverlap]
    )
    phase_metrics = {
        phase: PRIMARY._metric_summary(  # noqa: SLF001
            target[mask], prediction[mask], identity[mask]
        )
        for phase, mask in phases.items()
    }
    stratum_metrics = {
        kind: {
            name: PRIMARY._metric_summary(  # noqa: SLF001
                target[mask], prediction[mask], identity[mask]
            )
            for name, mask in groups.items()
        }
        for kind, groups in strata.items()
    }
    records: list[tuple[str, str, str, dict[str, Any]]] = [
        ("full", "all", "all", full),
        ("greedy_nonoverlap_32s", "all", "all", greedy),
    ]
    records.extend(
        (f"fixed_window_phase_{phase}", "window_phase_mod_8", phase, metrics)
        for phase, metrics in phase_metrics.items()
    )
    records.extend(
        ("stratum", kind, name, metrics)
        for kind, groups in stratum_metrics.items()
        for name, metrics in groups.items()
    )
    return {
        "full": full,
        "fixed_point_goal_gate": PRIMARY._gate_decision(full),  # noqa: SLF001
        "greedy_nonoverlap_32s": greedy,
        "eight_fixed_window_phases": phase_metrics,
        "strata": stratum_metrics,
        "nonoverlap_audit": {
            "greedy_rows": int(nonoverlap.sum()),
            "greedy_intervals_nonoverlapping": PRIMARY._intervals_nonoverlap(  # noqa: SLF001
                frame, nonoverlap
            ),
            "all_eight_fixed_phases_reported": set(phases)
            == {str(value) for value in range(8)},
            "fixed_phase_intervals_nonoverlapping": {
                phase: PRIMARY._intervals_nonoverlap(frame, mask)  # noqa: SLF001
                for phase, mask in phases.items()
            },
        },
    }, records


def _paired_degradation_bootstrap(
    frames: Mapping[str, Mapping[str, np.ndarray]],
    *,
    seed: int,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap = spec["bootstrap"]
    samples = int(bootstrap["samples"])
    confidence = float(bootstrap["confidence"])
    base_seed = int(bootstrap["base_seed"])
    baseline = frames[BASELINE_MASK]
    target = np.asarray(baseline["target_rr_bpm"], float)
    identity = np.asarray(baseline["identity"]).astype(str)
    names = sorted(set(identity.tolist()))
    derived_seed = PRIMARY._derived_bootstrap_seed(base_seed, seed)  # noqa: SLF001
    rng = np.random.default_rng(derived_seed)
    weights = rng.multinomial(
        len(names), np.full(len(names), 1.0 / len(names)), size=samples
    ).astype(float)
    sampled: dict[str, dict[str, np.ndarray]] = {}
    point: dict[str, dict[str, Any]] = {}
    for mask, frame in frames.items():
        prediction = np.asarray(frame["final_rr_bpm"], float)
        components = PRIMARY._cluster_components(  # noqa: SLF001
            target, prediction, identity, names
        )
        sampled[mask] = PRIMARY._bootstrap_values(components, weights)  # noqa: SLF001
        point[mask] = PRIMARY._metric_summary(target, prediction, identity)  # noqa: SLF001
    baseline_error = np.abs(np.asarray(baseline["final_rr_bpm"], float) - target)
    results: dict[str, Any] = {}
    for mask in MASK_NAMES:
        if mask == BASELINE_MASK:
            continue
        mask_error = np.abs(np.asarray(frames[mask]["final_rr_bpm"], float) - target)
        error_delta = mask_error - baseline_error
        raw_delta = {
            metric: (
                None
                if point[mask][metric] is None or point[BASELINE_MASK][metric] is None
                else float(point[mask][metric] - point[BASELINE_MASK][metric])
            )
            for metric in PRIMARY.METRIC_FIELDS
        }
        direction_adjusted = {
            metric: None if value is None else float(DEGRADATION_SIGN[metric] * value)
            for metric, value in raw_delta.items()
        }
        raw_intervals: dict[str, Any] = {}
        degradation_intervals: dict[str, Any] = {}
        for metric in PRIMARY.METRIC_FIELDS:
            values = sampled[mask][metric] - sampled[BASELINE_MASK][metric]
            raw_intervals[metric] = PRIMARY._percentile_interval(  # noqa: SLF001
                values,
                estimate=raw_delta[metric],
                samples=samples,
                confidence=confidence,
            )
            degradation_intervals[metric] = PRIMARY._percentile_interval(  # noqa: SLF001
                DEGRADATION_SIGN[metric] * values,
                estimate=direction_adjusted[metric],
                samples=samples,
                confidence=confidence,
            )
        tolerance = 1.0e-12
        results[mask] = {
            "comparison": f"{mask}_minus_{BASELINE_MASK}",
            "paired_on_exact_cache_index_and_physical_identity": True,
            "raw_metric_delta_mask_minus_radars_123": raw_delta,
            "direction_adjusted_degradation_positive_is_worse": direction_adjusted,
            "raw_metric_delta_intervals": raw_intervals,
            "direction_adjusted_degradation_intervals": degradation_intervals,
            "row_absolute_error": {
                "mean_delta_bpm": float(error_delta.mean()),
                "improved_fraction": float(np.mean(error_delta < -tolerance)),
                "tied_fraction": float(np.mean(np.abs(error_delta) <= tolerance)),
                "worsened_fraction": float(np.mean(error_delta > tolerance)),
            },
        }
    return {
        "fixed_spec": {
            "unit": "physical_identity",
            "identity_count": len(names),
            "samples": samples,
            "confidence": confidence,
            "base_seed": base_seed,
            "per_seed_derived_seed": derived_seed,
            "same_cluster_resamples_shared_across_all_seven_masks": True,
            "cross_seed_pooling": False,
        },
        "baseline_mask": BASELINE_MASK,
        "degradation_vs_radars_123": results,
    }


def _empty_csv_record() -> dict[str, Any]:
    return {name: "" for name in CSV_COLUMNS}


def _metric_csv_record(
    *,
    seed: int,
    mask: str,
    scope: str,
    stratum_type: str,
    stratum: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    record = _empty_csv_record()
    record.update(
        {
            "record_type": "metric",
            "seed": seed,
            "radar_mask": mask,
            "baseline_mask": BASELINE_MASK,
            "scope": scope,
            "stratum_type": stratum_type,
            "stratum": stratum,
            "rows": metrics["rows"],
            "identities": metrics["identities"],
            "tail_25_35_rows": metrics["tail_25_35_rows"],
            **{name: metrics[name] for name in PRIMARY.METRIC_FIELDS},
        }
    )
    return record


def _degradation_csv_record(
    *, seed: int, mask: str, result: Mapping[str, Any], rows: int, identities: int
) -> dict[str, Any]:
    raw = result["raw_metric_delta_mask_minus_radars_123"]
    degradation = result["direction_adjusted_degradation_positive_is_worse"]
    error = result["row_absolute_error"]
    record = _empty_csv_record()
    record.update(
        {
            "record_type": "paired_degradation",
            "seed": seed,
            "radar_mask": mask,
            "baseline_mask": BASELINE_MASK,
            "scope": "full",
            "stratum_type": "all",
            "stratum": "all",
            "rows": rows,
            "identities": identities,
            **raw,
            **{f"degradation_{name}": value for name, value in degradation.items()},
            "improved_fraction": error["improved_fraction"],
            "tied_fraction": error["tied_fraction"],
            "worsened_fraction": error["worsened_fraction"],
        }
    )
    return record


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = json.dumps(
        document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _format_csv(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise LockedRadarMaskEvaluationError("radar-mask CSV contains non-finite value")
        return np.format_float_positional(number, unique=True, trim="-")
    return str(value)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if set(row) != set(CSV_COLUMNS):
                raise LockedRadarMaskEvaluationError("radar-mask CSV row schema mismatch")
            writer.writerow({name: _format_csv(row[name]) for name in CSV_COLUMNS})
        stream.flush()
        os.fsync(stream.fileno())


def _publish_exclusive(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise LockedRadarMaskEvaluationError(
            f"immutable radar-mask evaluation output exists: {destination}"
        ) from exc


def evaluate_radar_masks(
    *,
    radar_mask_root: Path,
    primary_root: Path,
    primary_evaluation_lock: Path,
    target_receipt: Path,
    evaluation_spec: Path,
    output_dir: Path,
    report_output: Path,
    csv_output: Path,
    receipt_output: Path,
    orchestrator_command: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate, join, evaluate every mask, and publish create-once evidence."""

    output_root = output_dir.expanduser().resolve()
    report_path = report_output.expanduser().resolve()
    csv_path = csv_output.expanduser().resolve()
    receipt_path = receipt_output.expanduser().resolve()
    if len({report_path, csv_path, receipt_path}) != 3 or any(
        path.parent != output_root for path in (report_path, csv_path, receipt_path)
    ):
        raise LockedRadarMaskEvaluationError(
            "report, CSV, and receipt must be distinct direct children of output_dir"
        )
    if any(path.exists() for path in (report_path, csv_path, receipt_path)):
        raise LockedRadarMaskEvaluationError("immutable radar-mask evaluation output already exists")

    # The frozen spec is opened first.  Mask seal/plan/receipts and all 126
    # target-free outputs are then validated before the target authorization
    # boundary is invoked.
    try:
        spec, spec_binding = PRIMARY._load_evaluation_spec(evaluation_spec)  # noqa: SLF001
    except Exception as exc:
        raise LockedRadarMaskEvaluationError(str(exc)) from exc
    predictions, mask_context = _validate_mask_campaign(
        mask_root=radar_mask_root, evaluation_spec=spec
    )
    target_frames, primary_context = _load_primary_context(
        primary_root=primary_root,
        primary_evaluation_lock=primary_evaluation_lock,
        target_receipt=target_receipt,
        spec=spec,
    )
    if not _same_binding(
        mask_context["bindings"]["primary_predictions_seal"],
        primary_context["bindings"]["predictions_seal"],
    ):
        raise LockedRadarMaskEvaluationError(
            "radar-mask plan and canonical target evaluation use different primary prediction seals"
        )
    joined, parity_gate = _join_masks(
        target_frames=target_frames, predictions=predictions
    )

    csv_rows: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    for seed in spec["population"]["fixed_seeds"]:
        mask_metrics: dict[str, Any] = {}
        for mask in MASK_NAMES:
            bundle, records = _metric_bundle(joined[seed][mask])
            mask_metrics[mask] = bundle
            csv_rows.extend(
                _metric_csv_record(
                    seed=seed,
                    mask=mask,
                    scope=scope,
                    stratum_type=stratum_type,
                    stratum=stratum,
                    metrics=metrics,
                )
                for scope, stratum_type, stratum, metrics in records
            )
        degradation = _paired_degradation_bootstrap(
            joined[seed], seed=seed, spec=spec
        )
        baseline_full = mask_metrics[BASELINE_MASK]["full"]
        for mask, result in degradation["degradation_vs_radars_123"].items():
            csv_rows.append(
                _degradation_csv_record(
                    seed=seed,
                    mask=mask,
                    result=result,
                    rows=baseline_full["rows"],
                    identities=baseline_full["identities"],
                )
            )
        per_seed[str(seed)] = {
            "seed": seed,
            "seed_evaluated_independently": True,
            "all_seven_masks_reported_without_selection": True,
            "all_seven_masks_fixed_point_gates_passed": all(
                value["fixed_point_goal_gate"]["all_point_gates_passed"]
                for value in mask_metrics.values()
            ),
            "radar_masks": mask_metrics,
            "paired_physical_identity_cluster_bootstrap": degradation,
        }

    inputs = {
        **mask_context["bindings"],
        **primary_context["bindings"],
        "evaluation_spec": spec_binding,
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "retrospective_locked_hcs_all_radar_masks_evaluation",
        "commercial_claim_authorized": False,
        "commercial_performance_proven": False,
        "prospective_confirmation_required": True,
        "independent_prospective_cohort_evaluated": False,
        "mask_selection_or_ranking_performed": False,
        "seed_pooling_ranking_or_suppression_performed": False,
        "post_target_fitting_performed": False,
        "all_seven_masks_are_required_fixed_conditions": True,
        "all_masks_all_fixed_seeds_point_gates_passed": all(
            value["all_seven_masks_fixed_point_gates_passed"]
            for value in per_seed.values()
        ),
        "evaluation_specification": spec_binding,
        "radars_123_primary_parity_gate": parity_gate,
        "provenance_audit": {
            "evaluation_spec_verified_before_mask_or_target_evaluation": True,
            "complete_126_unit_target_free_mask_graph_verified_before_target_access": True,
            "canonical_target_receipt_and_primary_evaluation_lock_verified_before_target_arrays": True,
            "radar_mask_campaign": mask_context,
            "primary_target_context": primary_context,
            "exact_cache_fold_identity_session_window_protocol_join": True,
        },
        "per_seed": per_seed,
        "orchestrator_command": list(orchestrator_command),
    }
    report["content_sha256"] = PRIMARY.canonical_json_sha256(report)

    output_root.mkdir(parents=True, exist_ok=True)
    report_tmp = _temporary_path(report_path)
    csv_tmp = _temporary_path(csv_path)
    receipt_tmp = _temporary_path(receipt_path)
    try:
        _write_json(report_tmp, report)
        _write_csv(csv_tmp, csv_rows)
        outputs = {
            "report": {
                "path": str(report_path),
                "sha256": PRIMARY.sha256_file(report_tmp),
                "bytes": report_tmp.stat().st_size,
            },
            "metrics_csv": {
                "path": str(csv_path),
                "sha256": PRIMARY.sha256_file(csv_tmp),
                "bytes": csv_tmp.stat().st_size,
            },
        }
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": "retrospective_locked_hcs_all_radar_masks_evaluation_receipt",
            "commercial_claim_authorized": False,
            "commercial_performance_proven": False,
            "prospective_confirmation_required": True,
            "independent_prospective_cohort_evaluated": False,
            "outputs_create_once": True,
            "output_overwrite_allowed": False,
            "all_seven_masks_per_seed_without_selection": True,
            "complete_126_unit_seal_validated_before_target_access": True,
            "radars_123_primary_parity_gate_passed": True,
            "inputs": inputs,
            "outputs": outputs,
            "metrics_csv_rows": len(csv_rows),
            "orchestrator_command": list(orchestrator_command),
        }
        receipt["content_sha256"] = PRIMARY.canonical_json_sha256(receipt)
        _write_json(receipt_tmp, receipt)
        _publish_exclusive(report_tmp, report_path)
        report_path.chmod(0o444)
        _publish_exclusive(csv_tmp, csv_path)
        csv_path.chmod(0o444)
        _publish_exclusive(receipt_tmp, receipt_path)
        receipt_path.chmod(0o444)
    finally:
        for temporary in (report_tmp, csv_tmp, receipt_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    published = _read_json(
        receipt_path, "published radar-mask evaluation receipt", require_content_hash=True
    )
    if PRIMARY.bind_file(report_path) != published["outputs"]["report"] or PRIMARY.bind_file(
        csv_path
    ) != published["outputs"]["metrics_csv"]:
        raise LockedRadarMaskEvaluationError("published radar-mask output differs from receipt")
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radar-mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--primary-evaluation-lock", type=Path)
    parser.add_argument("--target-receipt", type=Path)
    parser.add_argument("--evaluation-spec", type=Path, default=PRIMARY.DEFAULT_EVALUATION_SPEC)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    primary_root = args.primary_root.expanduser().resolve()
    primary_lock = args.primary_evaluation_lock or primary_root / "evaluation_lock.json"
    target_receipt = (
        args.target_receipt
        or primary_root / "canonical_locked_hcs_targets_receipt.json"
    )
    output_dir = (
        args.output_dir
        or args.radar_mask_root / "evaluation"
    ).expanduser().resolve()
    report = args.report_output or output_dir / "locked_hcs_radar_masks_evaluation.json"
    csv_output = args.csv_output or output_dir / "locked_hcs_radar_masks_metrics.csv"
    receipt = (
        args.receipt_output
        or output_dir / "locked_hcs_radar_masks_evaluation_receipt.json"
    )
    command = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        *(argv or sys.argv[1:]),
    ]
    try:
        result = evaluate_radar_masks(
            radar_mask_root=args.radar_mask_root,
            primary_root=primary_root,
            primary_evaluation_lock=primary_lock,
            target_receipt=target_receipt,
            evaluation_spec=args.evaluation_spec,
            output_dir=output_dir,
            report_output=report,
            csv_output=csv_output,
            receipt_output=receipt,
            orchestrator_command=command,
        )
    except (LockedRadarMaskEvaluationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
