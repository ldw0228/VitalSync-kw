#!/usr/bin/env python3
"""Internal one-time canonical target publisher for locked HCS OOF evaluation.

This command is deliberately a separate, fail-closed boundary.  It validates
the complete 6-fold x 3-seed label-free prediction seal, every derived lock,
stage receipt, declared output, and provenance binding *before* it opens a
feature-cache metadata CSV or reads a reference value.  It then binds the
canonical cache row order to the frozen identity folds and publishes one
immutable NPZ plus one immutable receipt without overwriting either path.

The resulting archive contains the fields required by
``run_locked_hcs_oof.py join`` as well as stable row semantics and finite QC
lineage fields needed for non-overlap, protocol, RR-bin, and quality audits.
It remains retrospective and can never authorize a commercial claim.

Direct invocation is disabled.  The public authorization boundary is
``build_locked_hcs_targets_after_release_lock.py``, which passes an in-memory
capability only after every primary, radar-mask, uncertainty, and runtime seal
has been revalidated.
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
EXPECTED_UNIT_COUNT = len(FOLDS) * EXPECTED_SEED_COUNT
PREDICTION_FIELDS = {
    "cache_index",
    "outer_fold",
    "seed",
    "fallback_rr_bpm",
    "source_rr_bpm",
    "final_rr_bpm",
    "applied_pull",
    "target_joined",
}
FAST_STAGES = (
    "test_proposer_bind",
    "test_proposer_predict",
    "no_action_fallback_adapter",
)
FULL_STAGES = (
    "test_proposer_train",
    "test_proposer_predict",
    "derived_test_cache_build",
    "hcs_label_free_infer",
)
CORE_METADATA_FIELDS = (
    "session_id",
    "identity",
    "protocol",
    "window_number",
    "window_start_s",
    "window_end_s",
    "rr_bpm",
    "reference_valid",
)
OPTIONAL_METADATA_FIELDS: dict[str, str] = {
    "session_number": "int64",
    "reference_quality": "float32",
    "reference_sigma_bpm": "float32",
    "spectral_concentration": "float32",
    "periodicity": "float32",
    "estimator_disagreement_bpm": "float32",
    "phase_residual_rad": "float32",
    "clip_fraction": "float32",
    "guard_clip_fraction": "float32",
    "plateau_fraction": "float32",
    "breath_count": "int64",
    "radar_observable": "bool",
    "classical_confidence": "float32",
    "radar_peak_spread_bpm": "float32",
}
FORBIDDEN_PREJOIN_FIELDS = {
    "label",
    "labels",
    "target",
    "targets",
    "target_rr_bpm",
    "rr_bpm",
    "reference_rr_bpm",
    "reference_valid",
    "ground_truth",
    "ground_truth_rr_bpm",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCKED_ROOT = (
    PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "artifacts/cache/rf32s"
DEFAULT_FOLD_ASSIGNMENTS = (
    PROJECT_ROOT
    / "artifacts/runs/final_alias_gate_s12_deterministic/fold_assignments.json"
)


class TargetBuildError(RuntimeError):
    """A prediction-seal, provenance, cache, or publication invariant failed."""


# The canonical target builder is intentionally not a public target-opening
# capability.  Only the release-lock wrapper imports and passes this in-memory
# token after revalidating every target-free side seal and runtime payload.
_RELEASE_AUTHORIZATION_CAPABILITY = object()


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


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TargetBuildError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise TargetBuildError(f"{label} contains non-finite JSON number {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetBuildError(f"invalid {label}: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise TargetBuildError(f"{label} root must be an object: {path}")
    _verify_content_hash(value, label)
    return value


def _verify_content_hash(document: Mapping[str, Any], label: str) -> None:
    if "content_sha256" not in document:
        return
    payload = dict(document)
    expected = str(payload.pop("content_sha256", ""))
    if not _is_sha256(expected) or canonical_json_sha256(payload) != expected:
        raise TargetBuildError(f"{label} content_sha256 mismatch")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _resolve(value: Any, *, relative_to: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TargetBuildError(f"{label}.path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise TargetBuildError(f"bound file is absent: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _verify_binding(raw: Any, *, relative_to: Path, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TargetBuildError(f"missing file binding: {label}")
    path = _resolve(raw.get("path"), relative_to=relative_to, label=label)
    expected = str(raw.get("sha256", "")).lower()
    if not _is_sha256(expected):
        raise TargetBuildError(f"invalid SHA-256 binding: {label}")
    if not path.is_file():
        raise TargetBuildError(f"bound file is absent: {label} ({path})")
    size = path.stat().st_size
    declared_size = raw.get("bytes")
    if declared_size is not None and (
        isinstance(declared_size, bool)
        or not isinstance(declared_size, int)
        or declared_size != size
    ):
        raise TargetBuildError(f"file byte-size binding mismatch: {label}")
    if sha256_file(path) != expected:
        raise TargetBuildError(f"file SHA-256 binding mismatch: {label} ({path})")
    return {"path": str(path), "sha256": expected, "bytes": size}


def _scalar(array: Any, *, label: str) -> Any:
    value = np.asarray(array)
    if value.ndim != 0:
        raise TargetBuildError(f"{label} must be a scalar")
    return value.item()


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TargetBuildError(f"{label} must be an integer")
    return int(value)


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_target_free_prediction(
    path: Path, *, expected_fold: int, expected_seed: int
) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = set(archive.files)
            if fields != PREDICTION_FIELDS:
                raise TargetBuildError(
                    "sealed prediction schema mismatch "
                    f"(missing={sorted(PREDICTION_FIELDS - fields)}, "
                    f"extra={sorted(fields - PREDICTION_FIELDS)})"
                )
            forbidden = (fields - {"target_joined"}) & FORBIDDEN_PREJOIN_FIELDS
            if forbidden:
                raise TargetBuildError(
                    f"sealed prediction exposes pre-join target fields: {sorted(forbidden)}"
                )
            index_raw = np.asarray(archive["cache_index"])
            fold = _strict_int(
                _scalar(archive["outer_fold"], label="prediction outer_fold"),
                label="prediction outer_fold",
            )
            seed = _strict_int(
                _scalar(archive["seed"], label="prediction seed"),
                label="prediction seed",
            )
            joined = _scalar(archive["target_joined"], label="prediction target_joined")
            row_arrays = {
                name: np.asarray(archive[name])
                for name in (
                    "fallback_rr_bpm",
                    "source_rr_bpm",
                    "final_rr_bpm",
                    "applied_pull",
                )
            }
    except (OSError, ValueError) as exc:
        raise TargetBuildError(f"invalid sealed prediction: {path} ({exc})") from exc
    if fold != expected_fold or seed != expected_seed:
        raise TargetBuildError("sealed prediction fold/seed identity mismatch")
    if not isinstance(joined, (bool, np.bool_)) or bool(joined):
        raise TargetBuildError("sealed prediction was already target-joined")
    if index_raw.ndim != 1 or not np.issubdtype(index_raw.dtype, np.integer):
        raise TargetBuildError("sealed prediction cache_index must be an integer vector")
    index = index_raw.astype(np.int64, copy=False)
    if len(index) == 0 or (index < 0).any() or not np.all(index[1:] > index[:-1]):
        raise TargetBuildError(
            "sealed prediction cache_index must be non-empty, unique, and sorted"
        )
    for name, value in row_arrays.items():
        if value.shape != index.shape or not np.issubdtype(value.dtype, np.floating):
            raise TargetBuildError(f"sealed prediction {name} has an invalid shape/type")
        if not np.isfinite(value).all():
            raise TargetBuildError(f"sealed prediction {name} contains non-finite values")
    if (row_arrays["fallback_rr_bpm"] <= 0).any() or (
        row_arrays["final_rr_bpm"] <= 0
    ).any():
        raise TargetBuildError("sealed prediction contains non-positive RR values")
    return index.copy()


def _validate_receipt(
    binding_raw: Any,
    *,
    derived_lock_path: Path,
    expected_stage: str,
    expected_argv: Sequence[str],
    expected_outputs: Sequence[Path],
    unit_root: Path,
    position: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _verify_binding(
        binding_raw,
        relative_to=derived_lock_path.parent,
        label=f"stage receipt {expected_stage}",
    )
    receipt_path = Path(binding["path"])
    expected_receipt_path = (
        unit_root / "receipts" / f"{position:02d}_{expected_stage}.json"
    ).resolve()
    if receipt_path != expected_receipt_path or not _under(receipt_path, unit_root):
        raise TargetBuildError(f"stage receipt escapes canonical unit root: {expected_stage}")
    receipt = _read_json(receipt_path, f"stage receipt {expected_stage}")
    expected_keys = {
        "schema_version",
        "classification",
        "stage",
        "argv",
        "outputs",
        "stdout_stderr_log",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("classification") != "locked_hcs_oof_stage_receipt"
        or receipt.get("stage") != expected_stage
    ):
        raise TargetBuildError(f"stage receipt identity mismatch: {expected_stage}")
    argv = receipt.get("argv")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(token, str) or not token for token in argv
    ) or argv != list(expected_argv):
        raise TargetBuildError(f"stage receipt argv is invalid: {expected_stage}")
    outputs_raw = receipt.get("outputs")
    expected_paths = [path.expanduser().resolve() for path in expected_outputs]
    if not isinstance(outputs_raw, list) or len(outputs_raw) != len(expected_paths):
        raise TargetBuildError(f"stage receipt output topology differs: {expected_stage}")
    outputs: list[dict[str, Any]] = []
    for output_position, (raw, expected_path) in enumerate(
        zip(outputs_raw, expected_paths, strict=True)
    ):
        if not _under(expected_path, unit_root):
            raise TargetBuildError(f"stage output escapes canonical unit root: {expected_stage}")
        output = _verify_binding(
            raw,
            relative_to=receipt_path.parent,
            label=f"stage output {expected_stage}/{output_position}",
        )
        if Path(output["path"]) != expected_path:
            raise TargetBuildError(f"stage receipt has a surplus/different output: {expected_stage}")
        outputs.append(output)
    log = _verify_binding(
        receipt.get("stdout_stderr_log"),
        relative_to=receipt_path.parent,
        label=f"stage log {expected_stage}",
    )
    if not _under(Path(log["path"]), unit_root):
        raise TargetBuildError(f"stage log escapes canonical unit root: {expected_stage}")
    return binding, {"stage": expected_stage, "outputs": outputs, "log": log, "argv": argv}


def _validate_test_manifest(
    binding_raw: Any,
    *,
    derived_lock_path: Path,
    fold: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    binding = _verify_binding(
        binding_raw,
        relative_to=derived_lock_path.parent,
        label=f"test manifest fold {fold}",
    )
    path = Path(binding["path"])
    document = _read_json(path, f"test manifest fold {fold}")
    if document.get("schema_version") != 1:
        raise TargetBuildError(f"test manifest schema mismatch: fold {fold}")
    fold_id = document.get("fold_id")
    if fold_id is not None and _strict_int(fold_id, label="test manifest fold_id") != 100 * fold + 60:
        raise TargetBuildError(f"test manifest fold identity mismatch: fold {fold}")
    fold_binding = _verify_binding(
        document.get("fold_assignments"),
        relative_to=path.parent,
        label=f"test manifest fold assignment {fold}",
    )
    cache_raw = document.get("cache")
    if not isinstance(cache_raw, Mapping):
        raise TargetBuildError(f"test manifest cache binding is absent: fold {fold}")
    cache_binding = _verify_binding(
        {
            "path": cache_raw.get("manifest_path"),
            "sha256": cache_raw.get("manifest_sha256"),
        },
        relative_to=path.parent,
        label=f"test manifest cache {fold}",
    )
    return binding, fold_binding, cache_binding


def verify_prediction_seal(locked_oof_root: Path) -> dict[str, Any]:
    """Verify the complete label-free boundary without opening target metadata."""

    root = locked_oof_root.expanduser().resolve()
    seal_path = root / "predictions_seal.json"
    if not seal_path.is_file():
        raise TargetBuildError(
            "target construction forbidden: predictions_seal.json is absent"
        )
    seal = _read_json(seal_path, "predictions seal")
    units_raw = seal.get("units")
    if (
        seal.get("schema_version") != SCHEMA_VERSION
        or seal.get("classification")
        != "locked_hcs_oof_all_label_free_predictions_sealed"
        or seal.get("target_artifact_opened_before_seal") is not False
        or seal.get("target_join_authorized") is not True
        or seal.get("outer_folds") != list(FOLDS)
        or int(seal.get("unit_count", -1)) != EXPECTED_UNIT_COUNT
        or not isinstance(units_raw, list)
        or len(units_raw) != EXPECTED_UNIT_COUNT
    ):
        raise TargetBuildError("prediction seal is incomplete or has the wrong classification")

    pretest_path = root / "pretest_lock.json"
    if not pretest_path.is_file():
        raise TargetBuildError("prediction seal has no colocated pretest lock")
    pretest_sha = sha256_file(pretest_path)
    if seal.get("pretest_lock_sha256") != pretest_sha:
        raise TargetBuildError("prediction seal/pretest lock hash mismatch")
    pretest = _read_json(pretest_path, "pretest lock")
    if (
        pretest.get("schema_version") != 1
        or pretest.get("classification") != "locked_hcs_oof_all_pretest_assets_sealed"
        or pretest.get("outer_test_opened_before_lock") is not False
        or pretest.get("target_artifact_opened") is not False
        or int(pretest.get("unit_count", -1)) != EXPECTED_UNIT_COUNT
        or pretest.get("folds") != list(FOLDS)
    ):
        raise TargetBuildError("pretest lock is incomplete or records target access")
    plan_binding = _verify_binding(
        pretest.get("plan"), relative_to=pretest_path.parent, label="inference plan"
    )
    plan_path = Path(plan_binding["path"])
    plan = _read_json(plan_path, "inference plan")
    if (
        plan.get("schema_version") != 1
        or plan.get("classification") != "locked_hcs_oof_inference_plan"
        or plan.get("folds") != list(FOLDS)
    ):
        raise TargetBuildError("inference plan identity is invalid")
    plan_seeds_raw = plan.get("seeds")
    if not isinstance(plan_seeds_raw, list):
        raise TargetBuildError("inference plan seeds are absent")
    plan_seeds = tuple(_strict_int(seed, label="inference plan seed") for seed in plan_seeds_raw)
    if len(plan_seeds) != EXPECTED_SEED_COUNT or len(set(plan_seeds)) != len(plan_seeds):
        raise TargetBuildError("inference plan must bind three distinct seeds")
    if pretest.get("seeds") != list(plan_seeds):
        raise TargetBuildError("pretest lock/inference plan seed mismatch")

    expected_keys = {(fold, seed) for seed in plan_seeds for fold in FOLDS}
    plan_units_raw = plan.get("units")
    if not isinstance(plan_units_raw, list) or len(plan_units_raw) != EXPECTED_UNIT_COUNT:
        raise TargetBuildError("inference plan does not contain exactly 18 units")
    plan_units: dict[tuple[int, int], Mapping[str, Any]] = {}
    for unit in plan_units_raw:
        if not isinstance(unit, Mapping):
            raise TargetBuildError("inference plan unit must be an object")
        key = (
            _strict_int(unit.get("outer_fold"), label="plan unit outer_fold"),
            _strict_int(unit.get("seed"), label="plan unit seed"),
        )
        if key not in expected_keys or key in plan_units:
            raise TargetBuildError(f"inference plan has an invalid/duplicate unit: {key}")
        plan_units[key] = unit
    if set(plan_units) != expected_keys:
        raise TargetBuildError("inference plan unit topology is incomplete")

    rf_cache_binding = _verify_binding(
        plan.get("rf_cache_manifest"),
        relative_to=plan_path.parent,
        label="plan RF cache manifest",
    )
    effective_sources_raw = plan.get("effective_sources")
    if not isinstance(effective_sources_raw, Mapping) or not effective_sources_raw:
        raise TargetBuildError("inference plan effective source bindings are absent")
    effective_sources = {
        str(name): _verify_binding(
            binding,
            relative_to=plan_path.parent,
            label=f"effective source {name}",
        )
        for name, binding in sorted(effective_sources_raw.items())
    }
    if isinstance(plan.get("pretest_index"), Mapping):
        pretest_index_binding = _verify_binding(
            plan["pretest_index"],
            relative_to=plan_path.parent,
            label="fixed-i3 pretest index",
        )
        _read_json(Path(pretest_index_binding["path"]), "fixed-i3 pretest index")
    else:
        pretest_index_binding = None

    observed_keys: set[tuple[int, int]] = set()
    fold_indices_by_seed: dict[int, dict[int, np.ndarray]] = {
        seed: {} for seed in plan_seeds
    }
    provenance_units: list[dict[str, Any]] = []
    fold_bindings: list[dict[str, Any]] = []
    test_cache_bindings: list[dict[str, Any]] = []
    for unit_raw in units_raw:
        if not isinstance(unit_raw, Mapping):
            raise TargetBuildError("prediction seal unit must be an object")
        fold = _strict_int(unit_raw.get("outer_fold"), label="sealed unit outer_fold")
        seed = _strict_int(unit_raw.get("seed"), label="sealed unit seed")
        key = (fold, seed)
        if key not in expected_keys or key in observed_keys:
            raise TargetBuildError(f"prediction seal has an invalid/duplicate unit: {key}")
        observed_keys.add(key)
        plan_unit = plan_units[key]
        unit_root = root / "units" / f"outer_{fold}_seed_{seed}"
        derived_binding = _verify_binding(
            unit_raw.get("derived_lock"), relative_to=root, label=f"derived lock {fold}/{seed}"
        )
        derived_path = Path(derived_binding["path"])
        expected_derived_path = unit_root / "derived_inference_lock.json"
        if derived_path != expected_derived_path.resolve() or not _under(derived_path, root):
            raise TargetBuildError(f"derived lock escapes canonical unit root: {fold}/{seed}")
        derived = _read_json(derived_path, f"derived lock {fold}/{seed}")
        if (
            derived.get("schema_version") != 1
            or derived.get("classification") != "locked_hcs_oof_derived_test_inference"
            or derived.get("target_artifact_opened") is not False
            or _strict_int(derived.get("outer_fold"), label="derived outer_fold") != fold
            or _strict_int(derived.get("seed"), label="derived seed") != seed
            or derived.get("pretest_lock_sha256") != pretest_sha
        ):
            raise TargetBuildError(f"derived lock identity/provenance mismatch: {fold}/{seed}")

        receipts_raw = derived.get("stage_receipts")
        if not isinstance(receipts_raw, list):
            raise TargetBuildError(f"derived stage receipts are absent: {fold}/{seed}")
        commands_raw = derived.get("commands")
        if not isinstance(commands_raw, list):
            raise TargetBuildError(f"derived commands are absent: {fold}/{seed}")
        stages = tuple(
            str(command.get("stage", "")) if isinstance(command, Mapping) else ""
            for command in commands_raw
        )
        if stages not in (FAST_STAGES, FULL_STAGES) or len(receipts_raw) != len(stages):
            raise TargetBuildError(f"derived stage topology is invalid: {fold}/{seed}")
        plan_stages_raw = plan_unit.get("stages")
        if not isinstance(plan_stages_raw, list) or len(plan_stages_raw) != len(stages):
            raise TargetBuildError(f"inference plan/derived stage count mismatch: {fold}/{seed}")
        plan_commands: list[dict[str, Any]] = []
        for position, plan_stage in enumerate(plan_stages_raw):
            if not isinstance(plan_stage, Mapping):
                raise TargetBuildError(f"inference plan stage is invalid: {fold}/{seed}")
            plan_commands.append(
                {"stage": plan_stage.get("name"), "argv": plan_stage.get("argv")}
            )
        if plan_commands != commands_raw:
            raise TargetBuildError(f"inference plan/derived commands mismatch: {fold}/{seed}")
        verified_receipts: list[dict[str, Any]] = []
        receipt_output_by_path: dict[str, dict[str, Any]] = {}
        for position, (receipt_raw, command, stage) in enumerate(
            zip(receipts_raw, commands_raw, stages, strict=True)
        ):
            plan_stage = plan_stages_raw[position]
            if not isinstance(plan_stage, Mapping):
                raise TargetBuildError(f"inference plan stage is invalid: {fold}/{seed}")
            planned_outputs_raw = plan_stage.get("outputs")
            if (
                not isinstance(planned_outputs_raw, list)
                or not planned_outputs_raw
                or any(not isinstance(value, str) or not value for value in planned_outputs_raw)
            ):
                raise TargetBuildError(
                    f"inference plan stage outputs are invalid: {fold}/{seed}/{position}"
                )
            planned_outputs = [
                _resolve(
                    value,
                    relative_to=plan_path.parent,
                    label=f"planned stage output {fold}/{seed}/{position}/{output_position}",
                )
                for output_position, value in enumerate(planned_outputs_raw)
            ]
            receipt_binding, verified = _validate_receipt(
                receipt_raw,
                derived_lock_path=derived_path,
                expected_stage=stage,
                expected_argv=command.get("argv", []) if isinstance(command, Mapping) else [],
                expected_outputs=planned_outputs,
                unit_root=unit_root.resolve(),
                position=position,
            )
            if not isinstance(command, Mapping) or command.get("argv") != verified["argv"]:
                raise TargetBuildError(
                    f"derived command/receipt argv mismatch: {fold}/{seed}/{position}"
                )
            verified_receipts.append({"receipt": receipt_binding, **verified})
            for output in verified["outputs"]:
                if output["path"] in receipt_output_by_path:
                    raise TargetBuildError(f"unit receipts repeat output: {fold}/{seed}")
                receipt_output_by_path[output["path"]] = output

        artifacts_raw = derived.get("derived_artifacts")
        if not isinstance(artifacts_raw, Mapping) or not artifacts_raw:
            raise TargetBuildError(f"derived artifacts are absent: {fold}/{seed}")
        artifacts = {
            str(name): _verify_binding(
                binding,
                relative_to=derived_path.parent,
                label=f"derived artifact {fold}/{seed}/{name}",
            )
            for name, binding in sorted(artifacts_raw.items())
        }
        for artifact in artifacts.values():
            receipt_output = receipt_output_by_path.get(artifact["path"])
            if receipt_output is None or receipt_output["sha256"] != artifact["sha256"]:
                raise TargetBuildError(
                    f"derived artifact is not bound by a stage receipt: {fold}/{seed}"
                )
        plan_artifacts_raw = plan_unit.get("derived_artifacts")
        if not isinstance(plan_artifacts_raw, Mapping):
            raise TargetBuildError(f"inference plan derived artifacts are absent: {fold}/{seed}")
        for name, artifact in artifacts.items():
            if name not in plan_artifacts_raw:
                raise TargetBuildError(
                    f"derived artifact is absent from inference plan: {fold}/{seed}/{name}"
                )
            planned_path = _resolve(
                plan_artifacts_raw[name],
                relative_to=plan_path.parent,
                label=f"planned derived artifact {fold}/{seed}/{name}",
            )
            if planned_path != Path(artifact["path"]):
                raise TargetBuildError(
                    f"inference plan/derived artifact path mismatch: {fold}/{seed}/{name}"
                )

        derived_prediction = _verify_binding(
            derived.get("sealed_prediction"),
            relative_to=derived_path.parent,
            label=f"derived sealed prediction {fold}/{seed}",
        )
        seal_prediction = _verify_binding(
            unit_raw.get("prediction"),
            relative_to=root,
            label=f"seal prediction {fold}/{seed}",
        )
        expected_prediction_path = unit_root / "sealed_label_free_predictions.npz"
        if (
            derived_prediction != seal_prediction
            or Path(seal_prediction["path"]) != expected_prediction_path.resolve()
        ):
            raise TargetBuildError(f"seal/derived prediction binding mismatch: {fold}/{seed}")
        index = _validate_target_free_prediction(
            Path(seal_prediction["path"]), expected_fold=fold, expected_seed=seed
        )
        fold_indices_by_seed[seed][fold] = index

        test_manifest, fold_binding, test_cache_binding = _validate_test_manifest(
            derived.get("test_manifest"), derived_lock_path=derived_path, fold=fold
        )
        planned_test_manifest = _verify_binding(
            plan_unit.get("test_manifest"),
            relative_to=plan_path.parent,
            label=f"planned test manifest {fold}/{seed}",
        )
        if planned_test_manifest != test_manifest:
            raise TargetBuildError(f"inference plan/derived test manifest mismatch: {fold}/{seed}")
        fold_bindings.append(fold_binding)
        test_cache_bindings.append(test_cache_binding)
        provenance_units.append(
            {
                "outer_fold": fold,
                "seed": seed,
                "derived_lock": derived_binding,
                "prediction": seal_prediction,
                "prediction_rows": int(len(index)),
                "prediction_index_sha256": array_sha256(index),
                "test_manifest": test_manifest,
                "stage_receipts": verified_receipts,
            }
        )
    if observed_keys != expected_keys:
        raise TargetBuildError("prediction seal does not exactly cover six folds x three seeds")

    canonical_fold_sets: dict[int, np.ndarray] = {}
    seed_unions: dict[int, np.ndarray] = {}
    for seed in plan_seeds:
        parts = []
        for fold in FOLDS:
            index = fold_indices_by_seed[seed][fold]
            if fold not in canonical_fold_sets:
                canonical_fold_sets[fold] = index
            elif not np.array_equal(canonical_fold_sets[fold], index):
                raise TargetBuildError(f"prediction fold coverage differs across seeds: fold {fold}")
            parts.append(index)
        union = np.concatenate(parts)
        if len(np.unique(union)) != len(union):
            raise TargetBuildError(f"prediction folds overlap for seed {seed}")
        seed_unions[seed] = np.sort(union)
    first_union = seed_unions[plan_seeds[0]]
    if not np.array_equal(first_union, np.arange(len(first_union), dtype=np.int64)):
        raise TargetBuildError("sealed prediction coverage is not exact contiguous cache coverage")
    for seed in plan_seeds[1:]:
        if not np.array_equal(seed_unions[seed], first_union):
            raise TargetBuildError("sealed prediction cache coverage differs across seeds")

    unique_fold_bindings = {(item["path"], item["sha256"]) for item in fold_bindings}
    unique_cache_bindings = {(item["path"], item["sha256"]) for item in test_cache_bindings}
    if len(unique_fold_bindings) != 1:
        raise TargetBuildError("test manifests do not share one fold-assignment binding")
    if len(unique_cache_bindings) != 1 or next(iter(unique_cache_bindings)) != (
        rf_cache_binding["path"],
        rf_cache_binding["sha256"],
    ):
        raise TargetBuildError("test manifests/plan do not share one RF cache binding")

    provenance_units.sort(key=lambda item: (item["seed"], item["outer_fold"]))
    return {
        "root": root,
        "predictions_seal": bind_file(seal_path),
        "pretest_lock": bind_file(pretest_path),
        "inference_plan": plan_binding,
        "fixed_i3_pretest_index": pretest_index_binding,
        "rf_cache_manifest": rf_cache_binding,
        "fold_assignments": {
            "path": next(iter(unique_fold_bindings))[0],
            "sha256": next(iter(unique_fold_bindings))[1],
            "bytes": Path(next(iter(unique_fold_bindings))[0]).stat().st_size,
        },
        "effective_sources": effective_sources,
        "seeds": list(plan_seeds),
        "fold_indices": canonical_fold_sets,
        "expected_indices": first_union,
        "units": provenance_units,
    }


def _parse_bool(value: str, *, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise TargetBuildError(f"{label} must be a strict boolean")


def _parse_int(value: str, *, label: str, nonnegative: bool = False) -> int:
    stripped = value.strip()
    try:
        parsed_float = float(stripped)
    except ValueError as exc:
        raise TargetBuildError(f"{label} must be an integer") from exc
    if not math.isfinite(parsed_float) or parsed_float != round(parsed_float):
        raise TargetBuildError(f"{label} must be a finite integer")
    parsed = int(parsed_float)
    if nonnegative and parsed < 0:
        raise TargetBuildError(f"{label} must be non-negative")
    return parsed


def _parse_float(value: str, *, label: str, positive: bool = False) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as exc:
        raise TargetBuildError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or (positive and parsed <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise TargetBuildError(f"{label} must be {qualifier}")
    return parsed


def _fold_identity_map(document: Mapping[str, Any]) -> dict[str, int]:
    raw = document.get("identity_to_fold", document)
    if not isinstance(raw, Mapping) or not raw:
        raise TargetBuildError("fold assignments contain no identity mapping")
    result: dict[str, int] = {}
    for identity, value in raw.items():
        if not isinstance(identity, str) or not identity or identity != identity.strip():
            raise TargetBuildError("fold assignment identities must be non-empty and trimmed")
        fold = _strict_int(value, label=f"fold assignment {identity}")
        if fold not in FOLDS:
            raise TargetBuildError(f"fold assignment is outside 0..5: {identity}")
        result[identity] = fold
    if set(result.values()) != set(FOLDS):
        raise TargetBuildError("fold assignments do not cover all six folds")
    return result


def _read_metadata_csv(path: Path, *, expected_session: str) -> tuple[list[dict[str, str]], set[str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise TargetBuildError(f"metadata CSV has no header: {path}")
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise TargetBuildError(f"metadata CSV repeats a header: {path}")
            fields = set(reader.fieldnames)
            missing = sorted(set(CORE_METADATA_FIELDS) - fields)
            if missing:
                raise TargetBuildError(f"metadata CSV is missing fields {missing}: {path}")
            rows = [dict(row) for row in reader]
    except (OSError, csv.Error) as exc:
        raise TargetBuildError(f"cannot read metadata CSV: {path} ({exc})") from exc
    if not rows:
        raise TargetBuildError(f"metadata CSV has no rows: {path}")
    if any(row.get("session_id") != expected_session for row in rows):
        raise TargetBuildError(f"metadata session_id disagrees with cache manifest: {path}")
    return rows, fields


def _load_cache_targets(
    *,
    cache_dir: Path,
    cache_manifest_binding: Mapping[str, Any],
    fold_assignments_path: Path,
    fold_binding: Mapping[str, Any],
    expected_indices: np.ndarray,
    fold_indices: Mapping[int, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read target-bearing metadata.  Call only after ``verify_prediction_seal``."""

    cache_root = cache_dir.expanduser().resolve()
    cache_manifest_path = cache_root / "manifest.json"
    if cache_manifest_path != Path(str(cache_manifest_binding["path"])):
        raise TargetBuildError("--cache-dir differs from the prediction plan cache")
    if sha256_file(cache_manifest_path) != cache_manifest_binding["sha256"]:
        raise TargetBuildError("RF cache manifest changed after prediction sealing")
    supplied_folds = fold_assignments_path.expanduser().resolve()
    if supplied_folds != Path(str(fold_binding["path"])):
        raise TargetBuildError("--fold-assignments differs from the sealed test manifests")
    if sha256_file(supplied_folds) != fold_binding["sha256"]:
        raise TargetBuildError("fold assignments changed after prediction sealing")

    cache_manifest = _read_json(cache_manifest_path, "RF cache manifest")
    fold_document = _read_json(supplied_folds, "fold assignments")
    identity_to_fold = _fold_identity_map(fold_document)
    sessions_raw = cache_manifest.get("sessions")
    if not isinstance(sessions_raw, list):
        raise TargetBuildError("RF cache manifest sessions are absent")
    sessions = [item for item in sessions_raw if isinstance(item, Mapping) and item.get("status") == "ok"]
    if not sessions:
        raise TargetBuildError("RF cache manifest has no usable sessions")
    session_ids = [str(item.get("session_id", "")) for item in sessions]
    if any(not value or value != value.strip() for value in session_ids):
        raise TargetBuildError("RF cache manifest has invalid session IDs")
    if len(set(session_ids)) != len(session_ids):
        raise TargetBuildError("RF cache manifest repeats a usable session")

    parsed_rows: list[dict[str, Any]] = []
    metadata_bindings: list[dict[str, Any]] = []
    optional_presence: set[str] | None = None
    semantic_keys: set[tuple[str, int]] = set()
    session_to_identity: dict[str, str] = {}
    for session_position, (session, session_id) in enumerate(zip(sessions, session_ids, strict=True)):
        metadata_path = cache_root / session_id / "metadata.csv"
        before = bind_file(metadata_path)
        raw_rows, fields = _read_metadata_csv(metadata_path, expected_session=session_id)
        after_hash = sha256_file(metadata_path)
        if before["sha256"] != after_hash or before["bytes"] != metadata_path.stat().st_size:
            raise TargetBuildError(f"metadata changed while being read: {metadata_path}")
        metadata_bindings.append(before)
        present = set(OPTIONAL_METADATA_FIELDS) & fields
        if optional_presence is None:
            optional_presence = present
        elif present != optional_presence:
            raise TargetBuildError("optional metadata schema differs across cache sessions")
        declared_rows = session.get("window_count")
        if (
            isinstance(declared_rows, bool)
            or not isinstance(declared_rows, int)
            or declared_rows != len(raw_rows)
        ):
            raise TargetBuildError(f"cache manifest row count mismatch: {session_id}")
        declared_valid = session.get("valid_reference_count")
        valid_count = 0
        identities: set[str] = set()
        for local_position, raw in enumerate(raw_rows):
            identity = str(raw["identity"])
            protocol = str(raw["protocol"])
            if (
                not identity
                or identity != identity.strip()
                or not protocol
                or protocol != protocol.strip()
            ):
                raise TargetBuildError(f"metadata identity/protocol is invalid: {session_id}")
            if identity not in identity_to_fold:
                raise TargetBuildError(f"metadata identity lacks a fold assignment: {identity}")
            window_number = _parse_int(
                raw["window_number"],
                label=f"{session_id}.window_number",
                nonnegative=True,
            )
            semantic = (session_id, window_number)
            if semantic in semantic_keys:
                raise TargetBuildError(f"duplicate session/window metadata row: {semantic}")
            semantic_keys.add(semantic)
            start = _parse_float(raw["window_start_s"], label=f"{session_id}.window_start_s")
            end = _parse_float(raw["window_end_s"], label=f"{session_id}.window_end_s")
            if end <= start:
                raise TargetBuildError(f"metadata window interval is invalid: {semantic}")
            target = _parse_float(
                raw["rr_bpm"], label=f"{session_id}.rr_bpm", positive=True
            )
            reference_valid = _parse_bool(
                raw["reference_valid"], label=f"{session_id}.reference_valid"
            )
            valid_count += int(reference_valid)
            row: dict[str, Any] = {
                "cache_index": len(parsed_rows),
                "outer_fold": identity_to_fold[identity],
                "target_rr_bpm": target,
                "identity": identity,
                "reference_valid": reference_valid,
                "session_id": session_id,
                "window_number": window_number,
                "protocol": protocol,
                "window_start_s": start,
                "window_end_s": end,
                "cache_session_position": session_position,
                "cache_session_row": local_position,
            }
            for name in sorted(present):
                dtype = OPTIONAL_METADATA_FIELDS[name]
                if dtype == "bool":
                    row[name] = _parse_bool(raw[name], label=f"{session_id}.{name}")
                elif dtype == "int64":
                    row[name] = _parse_int(
                        raw[name], label=f"{session_id}.{name}", nonnegative=True
                    )
                else:
                    row[name] = _parse_float(raw[name], label=f"{session_id}.{name}")
            parsed_rows.append(row)
            identities.add(identity)
        if len(identities) != 1:
            raise TargetBuildError(f"one cache session crosses identities: {session_id}")
        session_to_identity[session_id] = next(iter(identities))
        if (
            isinstance(declared_valid, bool)
            or not isinstance(declared_valid, int)
            or declared_valid != valid_count
        ):
            raise TargetBuildError(f"cache manifest valid-reference count mismatch: {session_id}")

    if set(session_to_identity.values()) != set(identity_to_fold):
        raise TargetBuildError("fold assignments/cache identity cover mismatch")
    rows = len(parsed_rows)
    cache_index = np.arange(rows, dtype=np.int64)
    if not np.array_equal(cache_index, expected_indices):
        raise TargetBuildError("cache metadata does not exactly match sealed prediction coverage")
    observed_fold = np.asarray([row["outer_fold"] for row in parsed_rows], dtype=np.int16)
    for fold in FOLDS:
        observed = cache_index[observed_fold == fold]
        expected = np.asarray(fold_indices[fold], dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise TargetBuildError(f"cache/fold/prediction ownership mismatch: fold {fold}")

    arrays: dict[str, np.ndarray] = {
        "cache_index": cache_index,
        "outer_fold": observed_fold,
        "target_rr_bpm": np.asarray([row["target_rr_bpm"] for row in parsed_rows], dtype=np.float32),
        "identity": np.asarray([row["identity"] for row in parsed_rows], dtype=np.str_),
        "reference_valid": np.asarray([row["reference_valid"] for row in parsed_rows], dtype=bool),
        "session_id": np.asarray([row["session_id"] for row in parsed_rows], dtype=np.str_),
        "window_number": np.asarray([row["window_number"] for row in parsed_rows], dtype=np.int64),
        "protocol": np.asarray([row["protocol"] for row in parsed_rows], dtype=np.str_),
        "window_start_s": np.asarray([row["window_start_s"] for row in parsed_rows], dtype=np.float64),
        "window_end_s": np.asarray([row["window_end_s"] for row in parsed_rows], dtype=np.float64),
        "cache_session_position": np.asarray(
            [row["cache_session_position"] for row in parsed_rows], dtype=np.int32
        ),
        "cache_session_row": np.asarray(
            [row["cache_session_row"] for row in parsed_rows], dtype=np.int32
        ),
    }
    for name in sorted(optional_presence or set()):
        arrays[name] = np.asarray(
            [row[name] for row in parsed_rows], dtype=np.dtype(OPTIONAL_METADATA_FIELDS[name])
        )
    for name, array in arrays.items():
        if array.shape != (rows,):
            raise TargetBuildError(f"target array length mismatch: {name}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise TargetBuildError(f"target array contains non-finite values: {name}")
    if not np.isfinite(arrays["target_rr_bpm"]).all() or (
        arrays["target_rr_bpm"] <= 0
    ).any():
        raise TargetBuildError("target RR array is invalid after float32 conversion")

    audit = {
        "cache_manifest": bind_file(cache_manifest_path),
        "fold_assignments": bind_file(supplied_folds),
        "metadata_files": metadata_bindings,
        "cache_rows": rows,
        "valid_reference_rows": int(arrays["reference_valid"].sum()),
        "identity_count": len(identity_to_fold),
        "session_count": len(sessions),
        "fold_row_counts": {
            str(fold): int(np.sum(arrays["outer_fold"] == fold)) for fold in FOLDS
        },
        "optional_qc_fields_included": sorted(optional_presence or set()),
    }
    return arrays, audit


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )


def _publish_exclusive(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise TargetBuildError(f"immutable output already exists: {destination}") from exc


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def build_targets(
    *,
    locked_oof_root: Path,
    cache_dir: Path,
    fold_assignments: Path,
    output: Path,
    receipt: Path,
    orchestrator_command: Sequence[str] = (),
    _release_capability: object | None = None,
) -> dict[str, Any]:
    """Verify the label-free seal, then publish one immutable target artifact."""

    if _release_capability is not _RELEASE_AUTHORIZATION_CAPABILITY:
        raise TargetBuildError(
            "direct canonical target construction is disabled; use "
            "build_locked_hcs_targets_after_release_lock.py"
        )

    output_path = output.expanduser().resolve()
    receipt_path = receipt.expanduser().resolve()
    if output_path == receipt_path:
        raise TargetBuildError("target NPZ and receipt paths must differ")
    if output_path.exists() or receipt_path.exists():
        raise TargetBuildError("immutable target output/receipt already exists; overwrite forbidden")

    # Ordering invariant: this call completes every label-free provenance and
    # artifact check before _load_cache_targets can open metadata or RR/QC.
    verified = verify_prediction_seal(locked_oof_root)
    arrays, cache_audit = _load_cache_targets(
        cache_dir=cache_dir,
        cache_manifest_binding=verified["rf_cache_manifest"],
        fold_assignments_path=fold_assignments,
        fold_binding=verified["fold_assignments"],
        expected_indices=verified["expected_indices"],
        fold_indices=verified["fold_indices"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    target_tmp = _temporary_path(output_path)
    receipt_tmp = _temporary_path(receipt_path)
    try:
        _write_npz(target_tmp, arrays)
        target_binding = {
            "path": str(output_path),
            "sha256": sha256_file(target_tmp),
            "bytes": target_tmp.stat().st_size,
        }
        array_schema = {
            name: {
                "dtype": np.asarray(value).dtype.str,
                "shape": list(np.asarray(value).shape),
                "array_sha256": array_sha256(value),
            }
            for name, value in sorted(arrays.items())
        }
        prediction_inventory = [
            {
                "outer_fold": unit["outer_fold"],
                "seed": unit["seed"],
                "derived_lock": unit["derived_lock"],
                "prediction": unit["prediction"],
                "prediction_rows": unit["prediction_rows"],
                "prediction_index_sha256": unit["prediction_index_sha256"],
                "test_manifest": unit["test_manifest"],
                "stage_receipts": [item["receipt"] for item in unit["stage_receipts"]],
            }
            for unit in verified["units"]
        ]
        receipt_document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": "retrospective_locked_hcs_canonical_target_artifact_receipt",
            "prediction_seal_verified_before_any_target_metadata_access": True,
            "target_metadata_opened_only_after_complete_prediction_seal_verification": True,
            "outer_test_labels_first_explicit_joinable_artifact_created_after_prediction_seal": True,
            "target_artifact_created_once": True,
            "target_artifact_overwrite_allowed": False,
            "pretarget_release_capability_verified": True,
            "commercial_claim_authorized": False,
            "prospective_confirmation_required": True,
            "target_artifact": target_binding,
            "target_schema": array_schema,
            "row_count": int(len(arrays["cache_index"])),
            "valid_reference_rows": int(arrays["reference_valid"].sum()),
            "exact_cache_coverage": True,
            "identity_fold_consistency_verified": True,
            "prediction_topology": {
                "folds": list(FOLDS),
                "seeds": verified["seeds"],
                "unit_count": EXPECTED_UNIT_COUNT,
                "same_fold_indices_across_seeds": True,
                "disjoint_fold_indices_with_exact_contiguous_union": True,
            },
            "source_bindings": {
                "predictions_seal": verified["predictions_seal"],
                "pretest_lock": verified["pretest_lock"],
                "inference_plan": verified["inference_plan"],
                "fixed_i3_pretest_index": verified["fixed_i3_pretest_index"],
                "effective_sources": verified["effective_sources"],
                "rf_cache_manifest": cache_audit["cache_manifest"],
                "fold_assignments": cache_audit["fold_assignments"],
                "metadata_files": cache_audit["metadata_files"],
            },
            "prediction_inventory": prediction_inventory,
            "cache_audit": {
                key: value
                for key, value in cache_audit.items()
                if key not in {"cache_manifest", "fold_assignments", "metadata_files"}
            },
            "orchestrator_command": list(orchestrator_command),
        }
        receipt_document["content_sha256"] = canonical_json_sha256(receipt_document)
        _write_json(receipt_tmp, receipt_document)
        # Publish without rename/replace: any pre-existing destination wins and
        # a crash after the NPZ link leaves an obvious fail-closed quarantine.
        _publish_exclusive(target_tmp, output_path)
        output_path.chmod(0o444)
        _publish_exclusive(receipt_tmp, receipt_path)
        receipt_path.chmod(0o444)
    finally:
        for temporary in (target_tmp, receipt_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    published = _read_json(receipt_path, "published target receipt")
    observed_target = bind_file(output_path)
    if published.get("target_artifact") != observed_target:
        raise TargetBuildError("published target artifact differs from its immutable receipt")
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-oof-root", type=Path, default=DEFAULT_LOCKED_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--fold-assignments", type=Path, default=DEFAULT_FOLD_ASSIGNMENTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.locked_oof_root.expanduser().resolve()
    output = (
        args.output
        if args.output is not None
        else root / "canonical_locked_hcs_targets.npz"
    )
    receipt = (
        args.receipt
        if args.receipt is not None
        else root / "canonical_locked_hcs_targets_receipt.json"
    )
    command = [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    try:
        result = build_targets(
            locked_oof_root=root,
            cache_dir=args.cache_dir,
            fold_assignments=args.fold_assignments,
            output=output,
            receipt=receipt,
            orchestrator_command=command,
        )
    except (TargetBuildError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
