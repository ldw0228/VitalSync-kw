#!/usr/bin/env python3
"""Leakage-safe, post-lock orchestration for the 18-unit HCS OOF campaign.

This driver intentionally separates two irreversible phases:

1. all model/capacity/policy/source and per-unit training artifacts are hashed
   into ``pretest_lock.json`` before any test-split artifact is touched;
2. all six-fold x three-seed label-free predictions are sealed before a target
   artifact may be opened by ``join_and_evaluate``.

The long-running proposer/cache/model commands are supplied as argv arrays in
an immutable plan.  They are executed without a shell and every stage gets an
immutable receipt.  The common policy is applied here, never selected here.
In particular, ``fail_closed_no_action`` produces a bit-exact float32 copy of
the fallback prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
FOLDS = tuple(range(6))
EXPECTED_UNIT_COUNT = 18
STAGES = (
    "test_proposer_train",
    "test_proposer_predict",
    "derived_test_cache_build",
    "hcs_label_free_infer",
)
FAST_NO_ACTION_STAGES = (
    "test_proposer_bind",
    "test_proposer_predict",
    "no_action_fallback_adapter",
)
ARTIFACT_FOR_STAGE = {
    "test_proposer_train": "test_proposer_checkpoint",
    "test_proposer_bind": "test_proposer_checkpoint",
    "test_proposer_predict": "test_proposer_prediction",
    "derived_test_cache_build": "derived_cache_manifest",
    "hcs_label_free_infer": "raw_hcs_prediction",
    "no_action_fallback_adapter": "raw_hcs_prediction",
}
LOCKED_ASSETS = (
    "checkpoint",
    "scaler",
    "cache_manifest",
    "fallback_oof",
    "run_manifest",
    "original_policy",
)
SOURCE_BINDING_NAMES = {
    "trainer": "train_harmonic_set_snn.py",
    "harmonic_set_model": "harmonic_set_models.py",
    "adaptive_campaign_contract": "ADAPTIVE_CAMPAIGN_CONTRACT.json",
    "campaign_config": "harmonic_set_v2.yaml",
}
FORBIDDEN_LABEL_FIELDS = {
    "label",
    "labels",
    "target",
    "target_rr_bpm",
    "reference_rr_bpm",
    "reference_valid",
    "ground_truth",
    "ground_truth_rr_bpm",
    "rr_bpm",
}
RAW_REQUIRED = {
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETEST_INDEX = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_fixed_i3_pretest/pretest_index.json"
)
DEFAULT_COMMON_ROOT = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_discovery/i3_common_lock"
)
DEFAULT_FREEZE_MANIFEST = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/source_snapshots/i3_final/MANIFEST.json"
)
DEFAULT_TEST_MANIFEST_ROOT = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/full_oof_test/manifests"
)
DEFAULT_RF_CACHE = PROJECT_ROOT / "artifacts/cache/rf32s"
DEFAULT_POSTLOCK_ROOT = (
    PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
)


class LockedOOFError(RuntimeError):
    """A fail-closed provenance, ordering, or artifact error."""


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


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LockedOOFError(f"invalid {label}: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise LockedOOFError(f"{label} root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if immutable:
            path.chmod(0o444)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, Any], *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if immutable:
            path.chmod(0o444)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _resolve(path_value: Any, *, relative_to: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _binding(
    raw: Any,
    *,
    relative_to: Path,
    label: str,
    verify: bool = True,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise LockedOOFError(f"missing file binding: {label}")
    path = _resolve(raw.get("path"), relative_to=relative_to)
    expected = str(raw.get("sha256", ""))
    if len(expected) != 64:
        raise LockedOOFError(f"invalid expected hash for {label}")
    if verify:
        if not path.is_file() or sha256_file(path) != expected:
            raise LockedOOFError(f"file hash mismatch: {label} ({path})")
    return {"path": str(path), "sha256": expected}


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise LockedOOFError(f"expected artifact file is absent: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _safe_unit_name(fold: int, seed: int) -> str:
    return f"outer_{fold}_seed_{seed}"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_plan(path: Path) -> tuple[dict[str, Any], Path]:
    resolved = path.expanduser().resolve()
    plan = _json(resolved, "locked OOF plan")
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise LockedOOFError("locked OOF plan schema_version must equal 1")
    if plan.get("classification") != "locked_hcs_oof_inference_plan":
        raise LockedOOFError("locked OOF plan has the wrong classification")
    seeds = plan.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise LockedOOFError("plan must bind exactly three distinct seeds")
    if plan.get("folds") != list(FOLDS):
        raise LockedOOFError("plan must bind outer folds [0,1,2,3,4,5]")
    units = plan.get("units")
    if not isinstance(units, list) or len(units) != EXPECTED_UNIT_COUNT:
        raise LockedOOFError("plan must contain exactly 18 outer-fold x seed units")
    expected = {(fold, int(seed)) for seed in seeds for fold in FOLDS}
    observed: set[tuple[int, int]] = set()
    for unit in units:
        if not isinstance(unit, Mapping):
            raise LockedOOFError("plan unit must be an object")
        key = (int(unit.get("outer_fold", -1)), int(unit.get("seed", -1)))
        if key in observed:
            raise LockedOOFError(f"duplicate unit in plan: {key}")
        observed.add(key)
    if observed != expected:
        raise LockedOOFError("plan units do not exactly cover six folds x three seeds")
    return plan, resolved


def _verified_existing_binding(raw: Any, *, relative_to: Path, label: str) -> dict[str, Any]:
    binding = _binding(raw, relative_to=relative_to, label=label)
    path = Path(binding["path"])
    return {**binding, "bytes": path.stat().st_size}


def _validation_proposer_from_stack(path: Path) -> tuple[Path, Path, Path]:
    """Resolve the already trained validation proposer without loading stack labels."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            provenance = json.loads(str(np.asarray(archive["provenance_json"]).item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise LockedOOFError(f"cannot read strict-stack provenance: {path} ({exc})") from exc
    source_units = provenance.get("source_units")
    if not isinstance(source_units, list):
        raise LockedOOFError("strict stack has no proposer source-unit provenance")
    selected = [unit for unit in source_units if unit.get("role") == "hcs_validation"]
    if len(selected) != 1:
        raise LockedOOFError("strict stack must bind exactly one validation proposer")
    checkpoint = Path(str(selected[0].get("checkpoint", ""))).expanduser().resolve()
    run_config = Path(str(selected[0].get("run_config", ""))).expanduser().resolve()
    original_manifest = Path(str(selected[0].get("manifest", ""))).expanduser().resolve()
    for candidate, label in (
        (checkpoint, "validation proposer checkpoint"),
        (run_config, "validation proposer run_config"),
        (original_manifest, "validation proposer split manifest"),
    ):
        if not candidate.is_file():
            raise LockedOOFError(f"strict-stack {label} is absent: {candidate}")
    if selected[0].get("run_config_sha256") != sha256_file(run_config):
        raise LockedOOFError("strict-stack validation proposer run_config hash mismatch")
    return checkpoint, run_config, original_manifest


def prepare_plan(
    *,
    pretest_index: Path,
    test_manifest_root: Path,
    output_root: Path,
    plan_output: Path,
    rf_cache: Path = DEFAULT_RF_CACHE,
    python_executable: Path = Path(sys.executable),
    proposer_trainer: Path = PROJECT_ROOT / "scripts/train.py",
    safe_helper: Path = PROJECT_ROOT / "scripts/build_locked_hcs_test_inputs.py",
    gpu_wrapper: Path = PROJECT_ROOT / "scripts/run_gpu_admitted.py",
    gpu_lock: Path = DEFAULT_POSTLOCK_ROOT / "test_proposer_gpu.lock",
    gpu_ledger: Path = DEFAULT_POSTLOCK_ROOT / "test_proposer_gpu_ledger.jsonl",
    train_device: str = "cuda",
    prediction_device: str = "cpu",
) -> dict[str, Any]:
    """Materialize the concrete 18-unit no-action plan from the pretest index."""

    index_path = pretest_index.expanduser().resolve()
    index = _json(index_path, "fixed-i3 pretest index")
    if (
        index.get("schema_version") != 1
        or index.get("classification") != "retrospective_fixed_i3_pretest_index"
        or index.get("status") != "complete"
        or int(index.get("completed_units", -1)) != EXPECTED_UNIT_COUNT
        or index.get("outer_test_opened") is not False
        or index.get("ready_for_separately_locked_label_free_outer_test_construction")
        is not True
    ):
        raise LockedOOFError("fixed-i3 pretest index is not a complete unopened 18-unit seal")
    if "content_sha256" in index:
        payload = dict(index)
        recorded = str(payload.pop("content_sha256"))
        if canonical_json_sha256(payload) != recorded:
            raise LockedOOFError("fixed-i3 pretest index content hash mismatch")
    matrix = index.get("matrix")
    if not isinstance(matrix, Mapping) or matrix.get("folds") != list(FOLDS):
        raise LockedOOFError("fixed-i3 pretest matrix does not contain folds 0..5")
    seeds = matrix.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise LockedOOFError("fixed-i3 pretest matrix does not bind three seeds")
    common_index = index.get("common")
    if not isinstance(common_index, Mapping):
        raise LockedOOFError("fixed-i3 pretest index lacks common lock bindings")
    common = {
        name: _verified_existing_binding(
            common_index.get(name), relative_to=index_path.parent, label=f"pretest common {name}"
        )
        for name in (
            "selection_lock",
            "capacity_selection",
            "policy",
            "source_freeze_manifest",
        )
    }
    units_raw = index.get("units")
    if not isinstance(units_raw, list) or len(units_raw) != EXPECTED_UNIT_COUNT:
        raise LockedOOFError("fixed-i3 pretest index unit cover is incomplete")
    unit_map: dict[tuple[int, int], Mapping[str, Any]] = {}
    for unit in units_raw:
        if not isinstance(unit, Mapping) or unit.get("status") != "complete":
            raise LockedOOFError("fixed-i3 pretest index contains an incomplete unit")
        key = (int(unit.get("outer_fold", -1)), int(unit.get("seed", -1)))
        if key in unit_map:
            raise LockedOOFError(f"fixed-i3 pretest index duplicates unit {key}")
        unit_map[key] = unit
    expected_keys = {(fold, int(seed)) for seed in seeds for fold in FOLDS}
    if set(unit_map) != expected_keys:
        raise LockedOOFError("fixed-i3 pretest index does not exactly cover 18 units")

    output = output_root.expanduser().resolve()
    manifest_root = test_manifest_root.expanduser().resolve()
    executable = python_executable.expanduser().absolute()
    sources = [proposer_trainer, safe_helper, gpu_wrapper, executable]
    for source in sources:
        if not source.is_file():
            raise LockedOOFError(f"post-lock executable source is absent: {source}")
    rf_manifest = rf_cache.expanduser().resolve() / "manifest.json"
    if not rf_manifest.is_file():
        raise LockedOOFError(f"RF cache manifest is absent: {rf_manifest}")
    units: list[dict[str, Any]] = []
    for seed in map(int, seeds):
        for fold in FOLDS:
            pretest = unit_map[(fold, seed)]
            artifacts_raw = pretest.get("artifacts")
            if not isinstance(artifacts_raw, Mapping):
                raise LockedOOFError(f"pretest unit artifacts are missing: {fold}/{seed}")
            selection = _verified_existing_binding(
                artifacts_raw.get("selection_lock"),
                relative_to=index_path.parent,
                label=f"pretest selection lock {fold}/{seed}",
            )
            locked_assets = {
                name: _verified_existing_binding(
                    artifacts_raw.get(index_name),
                    relative_to=index_path.parent,
                    label=f"pretest {name} {fold}/{seed}",
                )
                for name, index_name in {
                    "checkpoint": "checkpoint",
                    "scaler": "scaler",
                    "cache_manifest": "cache_manifest",
                    "fallback_oof": "fallback_oof",
                    "run_manifest": "run_manifest",
                    "original_policy": "original_policy",
                }.items()
            }
            if isinstance(artifacts_raw.get("history"), Mapping):
                locked_assets["history"] = _verified_existing_binding(
                    artifacts_raw.get("history"),
                    relative_to=index_path.parent,
                    label=f"pretest history {fold}/{seed}",
                )
            manifest_path = manifest_root / f"outer_{fold}" / f"test_pred_{fold}.json"
            if not manifest_path.is_file():
                raise LockedOOFError(f"full-OOF test manifest is absent: {manifest_path}")
            manifest_document = _json(manifest_path, "post-lock test proposer manifest")
            if (
                manifest_document.get("schema_version") != 1
                or int(manifest_document.get("fold_id", -1)) != 100 * fold + 60
                or not isinstance(manifest_document.get("identities"), Mapping)
            ):
                raise LockedOOFError(f"post-lock test manifest identity is invalid: {manifest_path}")
            manifest_binding = bind_file(manifest_path)
            strict_stack = _verified_existing_binding(
                artifacts_raw.get("strict_stack"),
                relative_to=index_path.parent,
                label=f"pretest strict stack {fold}/{seed}",
            )
            source_checkpoint, source_run_config, source_manifest = (
                _validation_proposer_from_stack(Path(strict_stack["path"]))
            )
            unit_root = output / "units" / _safe_unit_name(fold, seed)
            bound_root = unit_root / "work/bound_validation_proposer"
            test_checkpoint = bound_root / "fold_0/snn_best.pt"
            bound_run_config = bound_root / "run_config.json"
            safe_prediction = unit_root / "work/test_proposer_safe.npz"
            raw_prediction = unit_root / "work/no_action_raw_hcs.npz"
            bind_proposer = [
                str(executable),
                str(safe_helper.resolve()),
                "bind-proposer",
                "--source-checkpoint",
                str(source_checkpoint),
                "--source-run-config",
                str(source_run_config),
                "--source-manifest",
                str(source_manifest),
                "--test-manifest",
                str(manifest_path),
                "--output-checkpoint",
                str(test_checkpoint),
                "--output-run-config",
                str(bound_run_config),
            ]
            safe_predict = [
                str(executable),
                str(safe_helper.resolve()),
                "proposer-predict",
                "--cache-dir",
                str(rf_cache.expanduser().resolve()),
                "--checkpoint",
                str(test_checkpoint),
                "--run-config",
                str(bound_run_config),
                "--test-manifest",
                str(manifest_path),
                "--output",
                str(safe_prediction),
                "--device",
                str(prediction_device),
                "--batch-size",
                "128",
            ]
            if str(prediction_device).startswith("cuda"):
                safe_predict.append("--amp")
            adapter = [
                str(executable),
                str(safe_helper.resolve()),
                "no-action-adapter",
                "--proposer",
                str(safe_prediction),
                "--outer-fold",
                str(fold),
                "--seed",
                str(seed),
                "--output",
                str(raw_prediction),
            ]
            units.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "selection_lock": selection,
                    "locked_assets": locked_assets,
                    "test_manifest": manifest_binding,
                    "no_action_fast_path": True,
                    "bound_but_not_executed_hcs_checkpoint": locked_assets["checkpoint"],
                    "stages": [
                        {
                            "name": "test_proposer_bind",
                            "argv": bind_proposer,
                            "outputs": [str(test_checkpoint), str(bound_run_config)],
                        },
                        {
                            "name": "test_proposer_predict",
                            "argv": safe_predict,
                            "outputs": [str(safe_prediction)],
                        },
                        {
                            "name": "no_action_fallback_adapter",
                            "argv": adapter,
                            "outputs": [str(raw_prediction)],
                        },
                    ],
                    "derived_artifacts": {
                        "test_proposer_checkpoint": str(test_checkpoint),
                        "test_proposer_prediction": str(safe_prediction),
                        "raw_hcs_prediction": str(raw_prediction),
                    },
                    "source_validation_proposer": {
                        "checkpoint": bind_file(source_checkpoint),
                        "run_config": bind_file(source_run_config),
                        "split_manifest": bind_file(source_manifest),
                        "strict_stack": strict_stack,
                    },
                }
            )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_oof_inference_plan",
        "folds": list(FOLDS),
        "seeds": list(map(int, seeds)),
        "common": common,
        "pretest_index": bind_file(index_path),
        "rf_cache_manifest": bind_file(rf_manifest),
        "effective_sources": {
            "plan_builder": bind_file(Path(__file__)),
            "proposer_trainer": bind_file(proposer_trainer.resolve()),
            "safe_test_input_helper": bind_file(safe_helper.resolve()),
            "gpu_wrapper": bind_file(gpu_wrapper.resolve()),
            "python_executable": bind_file(executable),
        },
        "frozen_common_policy_fast_path": "fail_closed_no_action",
        "hcs_model_bound_but_not_executed": True,
        "units": units,
    }
    plan_path = plan_output.expanduser().resolve()
    if plan_path.exists():
        observed = _json(plan_path, "existing locked OOF plan")
        if observed != plan:
            raise LockedOOFError(f"existing locked OOF plan differs: {plan_path}")
    else:
        _atomic_json(plan_path, plan, immutable=True)
    # Exercise the same validators used by infer before declaring readiness.
    load_plan(plan_path)
    validated_common = _validate_common(plan, plan_path=plan_path)
    if validated_common["policy_selection_status"] != "fail_closed_no_action":
        raise LockedOOFError("prepared fast path requires frozen fail_closed_no_action")
    return {
        "status": "locked_oof_plan_prepared",
        "plan": bind_file(plan_path),
        "units": EXPECTED_UNIT_COUNT,
        "fast_path": "fail_closed_no_action",
        "target_artifact_opened": False,
    }


def _validate_common(plan: Mapping[str, Any], *, plan_path: Path) -> dict[str, Any]:
    common = plan.get("common")
    if not isinstance(common, Mapping):
        raise LockedOOFError("plan common bindings are missing")
    bindings = {
        name: _binding(common.get(name), relative_to=plan_path.parent, label=f"common.{name}")
        for name in (
            "selection_lock",
            "capacity_selection",
            "policy",
            "source_freeze_manifest",
        )
    }
    selection = _json(Path(bindings["selection_lock"]["path"]), "common selection lock")
    capacity = _json(Path(bindings["capacity_selection"]["path"]), "capacity selection")
    policy_document = _json(Path(bindings["policy"]["path"]), "common policy")
    freeze = _json(Path(bindings["source_freeze_manifest"]["path"]), "source freeze")
    if selection.get("schema_version") != 1 or selection.get(
        "outer_test_opened_before_lock"
    ) is not False:
        raise LockedOOFError("common lock was not sealed before outer-test access")
    if selection.get("capacity_selection_sha256") != bindings["capacity_selection"]["sha256"]:
        raise LockedOOFError("common lock/capacity hash mismatch")
    if selection.get("common_fallback_policy_sha256") != bindings["policy"]["sha256"]:
        raise LockedOOFError("common lock/policy hash mismatch")
    if capacity.get("outer_test_opened") is not False or policy_document.get(
        "outer_test_opened"
    ) is not False:
        raise LockedOOFError("common selection artifact records outer-test access")
    selected_preset = str(selection.get("selected_preset", ""))
    if selected_preset not in {"default", "large"}:
        raise LockedOOFError("common lock selected an unsupported capacity")
    if capacity.get("selected_preset") != selected_preset or policy_document.get(
        "selected_preset"
    ) != selected_preset:
        raise LockedOOFError("capacity/policy/common-lock preset mismatch")
    policy = policy_document.get("policy")
    if not isinstance(policy, Mapping):
        raise LockedOOFError("common policy payload is missing")
    if str(policy.get("selection_status", "")) != str(
        selection.get("policy_selection_status", "")
    ):
        raise LockedOOFError("common policy selection status mismatch")
    if freeze.get("outer_test_opened") is not False or freeze.get(
        "declared_before_any_i3_score"
    ) is not True:
        raise LockedOOFError("source freeze is not an unopened pre-i3 freeze")
    freeze_files = freeze.get("files")
    if not isinstance(freeze_files, Mapping) or selection.get("source_freeze") != freeze_files:
        raise LockedOOFError("common lock/source-freeze mapping mismatch")
    if capacity.get("source_freeze_manifest_sha256") != bindings[
        "source_freeze_manifest"
    ]["sha256"]:
        raise LockedOOFError("capacity selection/source-freeze hash mismatch")
    freeze_root = Path(bindings["source_freeze_manifest"]["path"]).parent
    for name, expected in freeze_files.items():
        frozen_path = freeze_root / str(name)
        if not frozen_path.is_file() or sha256_file(frozen_path) != str(expected):
            raise LockedOOFError(f"frozen source hash mismatch: {name}")
    return {
        "bindings": bindings,
        "selected_preset": selected_preset,
        "selected_parameter_count": int(selection.get("selected_parameter_count", -1)),
        "policy": dict(policy),
        "policy_selection_status": str(selection.get("policy_selection_status", "")),
        "source_freeze": dict(freeze_files),
    }


def _validate_unit_prelock(
    unit: Mapping[str, Any],
    *,
    plan_path: Path,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    fold = int(unit.get("outer_fold", -1))
    seed = int(unit.get("seed", -1))
    assets_raw = unit.get("locked_assets")
    if not isinstance(assets_raw, Mapping):
        raise LockedOOFError(f"locked assets missing for unit {fold}/{seed}")
    selection_binding = _binding(
        unit.get("selection_lock"),
        relative_to=plan_path.parent,
        label=f"unit {fold}/{seed} selection lock",
    )
    selection = _json(Path(selection_binding["path"]), "unit selection lock")
    if (
        int(selection.get("outer_fold", -1)) != fold
        or int(selection.get("seed", -1)) != seed
        or int(selection.get("adaptive_iteration", -1)) != 3
        or selection.get("outer_test_not_opened_before_this_lock") is not True
    ):
        raise LockedOOFError(f"unit selection-lock identity/leakage mismatch: {fold}/{seed}")
    assets = {
        name: _binding(
            assets_raw.get(name),
            relative_to=plan_path.parent,
            label=f"unit {fold}/{seed} {name}",
        )
        for name in LOCKED_ASSETS
    }
    if "history_sha256" in selection:
        assets["history"] = _binding(
            assets_raw.get("history"),
            relative_to=plan_path.parent,
            label=f"unit {fold}/{seed} history",
        )
    expected_hashes = {
        "checkpoint_sha256": "checkpoint",
        "scaler_sha256": "scaler",
        "cache_manifest_sha256": "cache_manifest",
        "fallback_oof_sha256": "fallback_oof",
        "run_manifest_sha256": "run_manifest",
        "policy_sha256": "original_policy",
        "history_sha256": "history",
    }
    for lock_key, asset_name in expected_hashes.items():
        if lock_key in selection and selection.get(lock_key) != assets[asset_name]["sha256"]:
            raise LockedOOFError(f"unit lock hash mismatch: {fold}/{seed}/{asset_name}")
    run_manifest = _json(Path(assets["run_manifest"]["path"]), "unit run manifest")
    if (
        int(run_manifest.get("outer_fold", -1)) != fold
        or int(run_manifest.get("optimization", {}).get("seed", -1)) != seed
    ):
        raise LockedOOFError(f"unit run-manifest identity mismatch: {fold}/{seed}")
    model = run_manifest.get("model_config")
    expected_hidden = {"default": 64, "large": 96}[str(common["selected_preset"])]
    if not isinstance(model, Mapping) or int(model.get("hidden_channels", -1)) != expected_hidden:
        raise LockedOOFError(f"unit capacity differs from frozen common capacity: {fold}/{seed}")
    effective = run_manifest.get("iteration_effective_configuration")
    if not isinstance(effective, Mapping):
        raise LockedOOFError(f"unit effective configuration is missing: {fold}/{seed}")
    capacity = effective.get("model_capacity")
    if not isinstance(capacity, Mapping) or int(capacity.get("parameter_count", -1)) != int(
        common["selected_parameter_count"]
    ):
        raise LockedOOFError(f"unit parameter count differs from common lock: {fold}/{seed}")
    source_bindings = selection.get("source_bindings")
    if not isinstance(source_bindings, Mapping):
        raise LockedOOFError(f"unit source bindings are missing: {fold}/{seed}")
    for binding_name, freeze_name in SOURCE_BINDING_NAMES.items():
        binding = source_bindings.get(binding_name)
        expected = common["source_freeze"].get(freeze_name)
        if not isinstance(binding, Mapping) or binding.get("sha256") != expected:
            raise LockedOOFError(
                f"unit source differs from frozen source: {fold}/{seed}/{binding_name}"
            )
    return {
        "outer_fold": fold,
        "seed": seed,
        "selection_lock": selection_binding,
        "locked_assets": assets,
    }


def _pretest_document(
    plan: Mapping[str, Any],
    *,
    plan_path: Path,
    common: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_oof_all_pretest_assets_sealed",
        "outer_test_opened_before_lock": False,
        "capacity_policy_checkpoint_reselection_allowed": False,
        "unit_count": EXPECTED_UNIT_COUNT,
        "folds": list(FOLDS),
        "seeds": [int(value) for value in plan["seeds"]],
        "plan": bind_file(plan_path),
        "common": common["bindings"],
        "selected_preset": common["selected_preset"],
        "selected_parameter_count": common["selected_parameter_count"],
        "policy_selection_status": common["policy_selection_status"],
        "source_freeze": common["source_freeze"],
        "units": list(units),
        "target_artifact_opened": False,
    }


def _ensure_immutable_document(path: Path, expected: Mapping[str, Any]) -> None:
    if path.exists():
        observed = _json(path, path.name)
        if observed != expected:
            raise LockedOOFError(f"immutable lock disagrees with current inputs: {path}")
        return
    _atomic_json(path, expected, immutable=True)


def _validate_stage_plan(
    unit: Mapping[str, Any],
    *,
    plan_path: Path,
    output_root: Path,
    locked_paths: set[Path],
    no_action_fast_path: bool,
) -> list[dict[str, Any]]:
    fold = int(unit["outer_fold"])
    seed = int(unit["seed"])
    expected_root = output_root / "units" / _safe_unit_name(fold, seed)
    stages = unit.get("stages")
    artifacts = unit.get("derived_artifacts")
    if not isinstance(stages, list) or not isinstance(artifacts, Mapping):
        raise LockedOOFError(f"unit stage/artifact plan missing: {fold}/{seed}")
    observed_stages = [stage.get("name") for stage in stages if isinstance(stage, Mapping)]
    allowed_orders = (FAST_NO_ACTION_STAGES, STAGES) if no_action_fast_path else (STAGES,)
    if not any(observed_stages == list(order) for order in allowed_orders):
        raise LockedOOFError(f"unit stages do not use the frozen stage order: {fold}/{seed}")
    result: list[dict[str, Any]] = []
    for stage in stages:
        name = str(stage["name"])
        argv = stage.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(token, str) or not token for token in argv)
        ):
            raise LockedOOFError(f"stage argv must be a non-empty string array: {fold}/{seed}/{name}")
        lowered = " ".join(argv).lower()
        if "train_harmonic_set_snn" in lowered or "select_i3_common" in lowered:
            raise LockedOOFError(f"stage attempts forbidden model/policy reselection: {fold}/{seed}/{name}")
        outputs = stage.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise LockedOOFError(f"stage outputs are missing: {fold}/{seed}/{name}")
        resolved_outputs: list[Path] = []
        for value in outputs:
            path = _resolve(value, relative_to=plan_path.parent)
            if not _is_relative_to(path, expected_root):
                raise LockedOOFError(f"derived output escapes its unit root: {path}")
            if path in locked_paths:
                raise LockedOOFError(f"derived output aliases a locked input: {path}")
            resolved_outputs.append(path)
        artifact_name = ARTIFACT_FOR_STAGE[name]
        artifact_path = _resolve(artifacts.get(artifact_name), relative_to=plan_path.parent)
        if artifact_path not in resolved_outputs:
            raise LockedOOFError(
                f"stage does not declare its required artifact: {fold}/{seed}/{name}"
            )
        result.append({"name": name, "argv": list(argv), "outputs": resolved_outputs})
    return result


def _stage_receipt_path(unit_root: Path, position: int, stage_name: str) -> Path:
    return unit_root / "receipts" / f"{position:02d}_{stage_name}.json"


_STAGE_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "classification",
        "stage",
        "argv",
        "outputs",
        "stdout_stderr_log",
    }
)
_FILE_BINDING_KEYS = frozenset({"path", "sha256", "bytes"})


def _validate_stage_receipt(
    receipt_path: Path,
    *,
    stage: Mapping[str, Any],
    unit_root: Path,
    position: int,
) -> dict[str, Any]:
    """Revalidate one receipt against the exact frozen stage contract."""

    expected_receipt = _stage_receipt_path(unit_root, position, str(stage["name"])).resolve()
    resolved_receipt = receipt_path.expanduser().resolve()
    if (
        resolved_receipt != expected_receipt
        or not _is_relative_to(resolved_receipt, unit_root.resolve())
        or receipt_path.is_symlink()
    ):
        raise LockedOOFError(f"stage receipt escapes its unit root: {receipt_path}")
    receipt = _json(resolved_receipt, "stage receipt")
    if set(receipt) != _STAGE_RECEIPT_KEYS:
        raise LockedOOFError(f"stage receipt key topology differs: {stage['name']}")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("classification") != "locked_hcs_oof_stage_receipt"
        or receipt.get("stage") != stage["name"]
        or receipt.get("argv") != stage["argv"]
    ):
        raise LockedOOFError(f"stage receipt identity/command differs: {stage['name']}")
    raw_outputs = receipt.get("outputs")
    expected_outputs = list(stage["outputs"])
    if not isinstance(raw_outputs, list) or len(raw_outputs) != len(expected_outputs):
        raise LockedOOFError(f"stage receipt output topology differs: {stage['name']}")
    for output_position, (raw, expected_path) in enumerate(
        zip(raw_outputs, expected_outputs, strict=True)
    ):
        expected = Path(expected_path).resolve()
        if not _is_relative_to(expected, unit_root.resolve()):
            raise LockedOOFError(f"stage output escapes its unit root: {expected}")
        if not isinstance(raw, Mapping) or set(raw) != _FILE_BINDING_KEYS:
            raise LockedOOFError(
                f"stage output binding topology differs: {stage['name']}/{output_position}"
            )
        if dict(raw) != bind_file(expected):
            raise LockedOOFError(
                f"stage receipt output differs from frozen plan: {stage['name']}/{output_position}"
            )
    expected_log = (
        unit_root / "logs" / f"{position:02d}_{stage['name']}.log"
    ).resolve()
    raw_log = receipt.get("stdout_stderr_log")
    if (
        not isinstance(raw_log, Mapping)
        or set(raw_log) != _FILE_BINDING_KEYS
        or not _is_relative_to(expected_log, unit_root.resolve())
        or dict(raw_log) != bind_file(expected_log)
    ):
        raise LockedOOFError(f"stage receipt log binding differs: {stage['name']}")
    return receipt


def _run_stage(
    stage: Mapping[str, Any],
    *,
    unit_root: Path,
    position: int,
    cwd: Path,
) -> dict[str, Any]:
    receipt_path = _stage_receipt_path(unit_root, position, str(stage["name"]))
    if receipt_path.exists():
        return _validate_stage_receipt(
            receipt_path,
            stage=stage,
            unit_root=unit_root,
            position=position,
        )
    partial = [path for path in stage["outputs"] if path.exists()]
    if partial:
        raise LockedOOFError(
            f"unreceipted partial stage artifacts require operator quarantine: {partial}"
        )
    completed = subprocess.run(
        stage["argv"],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log_path = unit_root / "logs" / f"{position:02d}_{stage['name']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        completed.stdout + ("\n[stderr]\n" if completed.stderr else "") + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise LockedOOFError(
            f"stage failed with status {completed.returncode}: {stage['name']} ({log_path})"
        )
    missing = [str(path) for path in stage["outputs"] if not path.is_file()]
    if missing:
        raise LockedOOFError(f"stage completed without declared outputs: {missing}")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_oof_stage_receipt",
        "stage": stage["name"],
        "argv": stage["argv"],
        "outputs": [bind_file(path) for path in stage["outputs"]],
        "stdout_stderr_log": bind_file(log_path),
    }
    _atomic_json(receipt_path, receipt, immutable=True)
    return _validate_stage_receipt(
        receipt_path,
        stage=stage,
        unit_root=unit_root,
        position=position,
    )


def _forbid_label_arrays(names: Sequence[str], *, label: str) -> None:
    forbidden = sorted({str(name).lower() for name in names} & FORBIDDEN_LABEL_FIELDS)
    if forbidden:
        raise LockedOOFError(f"{label} contains pre-seal label fields: {forbidden}")


def _validate_target_free_npz(path: Path, *, label: str) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            _forbid_label_arrays(archive.files, label=label)
    except (OSError, ValueError) as exc:
        raise LockedOOFError(f"invalid target-free NPZ {path}: {exc}") from exc


def _validate_derived_cache_manifest(path: Path, *, fold: int, seed: int) -> None:
    manifest = _json(path, "derived test-only cache manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "hcs_test_only_inference_cache"
        or manifest.get("row_scope") != "outer_test_only"
        or manifest.get("target_fields_present") is not False
        or int(manifest.get("outer_fold", -1)) != fold
        or int(manifest.get("seed", -1)) != seed
    ):
        raise LockedOOFError("derived cache is not an attested target-free outer-test cache")
    encoded = json.dumps(manifest, sort_keys=True).lower()
    if '"target_fields_present": true' in encoded:
        raise LockedOOFError("derived cache manifest exposes target fields")


def _raw_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            _forbid_label_arrays(archive.files, label="raw HCS inference")
            missing = sorted(RAW_REQUIRED - set(archive.files))
            if missing:
                raise LockedOOFError(f"raw HCS inference fields are missing: {missing}")
            return {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise LockedOOFError(f"invalid raw HCS inference archive: {path} ({exc})") from exc


def apply_frozen_policy(
    raw: Mapping[str, np.ndarray], policy: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a frozen policy without any target-dependent selection."""

    fallback = np.asarray(raw["fallback_rr_bpm"])
    if fallback.dtype != np.dtype(np.float32) or fallback.ndim != 1:
        raise LockedOOFError("fallback_rr_bpm must be a one-dimensional float32 array")
    rows = len(fallback)
    for name in RAW_REQUIRED - {"cache_index", "fallback_rr_bpm"}:
        if np.asarray(raw[name]).shape != (rows,):
            raise LockedOOFError(f"raw inference field has wrong shape: {name}")
    if not np.isfinite(fallback).all():
        raise LockedOOFError("fallback contains non-finite predictions")
    status = str(policy.get("selection_status", ""))
    pull = float(policy.get("correction_pull", math.nan))
    fallback_available = np.asarray(raw["fallback_available"], dtype=bool)
    if status == "fail_closed_no_action":
        if pull != 0.0 or not fallback_available.all():
            raise LockedOOFError("no-action policy requires pull=0 and a fallback on every row")
        final = fallback.copy()
        if not np.array_equal(final.view(np.uint32), fallback.view(np.uint32)):
            raise LockedOOFError("no-action output is not bit-exact float32 fallback")
        return final, np.zeros(rows, dtype=np.float32)

    source = np.asarray(raw["source_rr_bpm"], dtype=np.float32)
    source_available = np.asarray(raw["source_available"], dtype=bool)
    base_std = np.asarray(raw["fallback_std_bpm"], dtype=np.float32)
    source_scale = np.asarray(raw["source_scale_bpm"], dtype=np.float32)
    candidate_count = np.asarray(raw["valid_candidate_count"], dtype=np.int64)
    if "normalized_entropy" in raw:
        normalized_entropy = np.asarray(raw["normalized_entropy"], dtype=np.float32)
    else:
        entropy = np.asarray(raw["entropy"], dtype=np.float32)
        normalized_entropy = np.where(
            candidate_count > 1,
            entropy / np.maximum(np.log(np.maximum(candidate_count, 2)), 1.0e-8),
            0.0,
        )

    def maximum(name: str) -> float:
        value = policy.get(name)
        return math.inf if value is None else float(value)

    disagreement = np.abs(source.astype(np.float64) - fallback.astype(np.float64))
    action = (
        fallback_available
        & source_available
        & (np.asarray(raw["selected_probability"], dtype=float) >= float(policy["probability_threshold"]))
        & (np.asarray(raw["margin"], dtype=float) >= float(policy["margin_threshold"]))
        & (normalized_entropy <= float(policy["entropy_threshold"]))
        & (np.asarray(raw["quality"], dtype=float) >= float(policy["quality_threshold"]))
        & (base_std <= maximum("base_std_max"))
        & (source_scale <= maximum("source_scale_max"))
        & (disagreement >= float(policy.get("disagreement_min") or 0.0))
        & (disagreement <= maximum("disagreement_max"))
        & (candidate_count >= int(policy.get("minimum_valid_candidates", 1)))
    )
    applied = action.astype(np.float32) * np.float32(pull)
    applied = np.where(~fallback_available & source_available, 1.0, applied).astype(np.float32)
    applied = np.where(fallback_available & ~source_available, 0.0, applied).astype(np.float32)
    base_or_source = np.where(fallback_available, fallback, source)
    final = ((1.0 - applied) * base_or_source + applied * source).astype(np.float32)
    final[~fallback_available & source_available] = source[~fallback_available & source_available]
    final[fallback_available & ~source_available] = fallback[fallback_available & ~source_available]
    if not np.isfinite(final).all():
        raise LockedOOFError("frozen policy produced non-finite predictions")
    return final, applied


def _sealed_prediction_arrays(
    raw: Mapping[str, np.ndarray],
    *,
    fold: int,
    seed: int,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    index = np.asarray(raw["cache_index"], dtype=np.int64)
    if index.ndim != 1 or len(index) == 0 or len(np.unique(index)) != len(index):
        raise LockedOOFError("raw inference cache_index is empty, duplicated, or non-vector")
    final, applied = apply_frozen_policy(raw, policy)
    source = np.asarray(raw["source_rr_bpm"], dtype=np.float32)
    fallback = np.asarray(raw["fallback_rr_bpm"], dtype=np.float32)
    order = np.argsort(index, kind="stable")
    return {
        "cache_index": index[order],
        "outer_fold": np.asarray(fold, dtype=np.int16),
        "seed": np.asarray(seed, dtype=np.int64),
        "fallback_rr_bpm": fallback[order],
        "source_rr_bpm": source[order],
        "final_rr_bpm": final[order],
        "applied_pull": applied[order],
        "target_joined": np.asarray(False),
    }


def _compare_prediction(path: Path, expected: Mapping[str, Any]) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise LockedOOFError("existing sealed prediction schema mismatch")
            for name, value in expected.items():
                if not np.array_equal(np.asarray(archive[name]), np.asarray(value)):
                    raise LockedOOFError(f"existing sealed prediction differs: {name}")
    except (OSError, ValueError) as exc:
        raise LockedOOFError(f"invalid sealed prediction: {path} ({exc})") from exc


def _verify_derived_lock(
    path: Path,
    *,
    expected_pretest_sha: str,
    stages: Sequence[Mapping[str, Any]],
    unit_root: Path,
) -> dict[str, Any]:
    lock = _json(path, "derived unit lock")
    if lock.get("pretest_lock_sha256") != expected_pretest_sha:
        raise LockedOOFError("derived unit lock is bound to another pretest lock")
    raw_receipts = lock.get("stage_receipts")
    if not isinstance(raw_receipts, list) or len(raw_receipts) != len(stages):
        raise LockedOOFError("derived unit lock stage-receipt topology differs")
    for position, (raw, stage) in enumerate(zip(raw_receipts, stages, strict=True)):
        receipt_path = _stage_receipt_path(unit_root, position, str(stage["name"]))
        if not isinstance(raw, Mapping) or set(raw) != _FILE_BINDING_KEYS:
            raise LockedOOFError("derived stage-receipt binding topology differs")
        if dict(raw) != bind_file(receipt_path):
            raise LockedOOFError("derived stage-receipt binding differs")
        _validate_stage_receipt(
            receipt_path,
            stage=stage,
            unit_root=unit_root,
            position=position,
        )
    for binding in lock.get("derived_artifacts", {}).values():
        _binding(binding, relative_to=path.parent, label="derived artifact")
    _binding(lock.get("sealed_prediction"), relative_to=path.parent, label="sealed prediction")
    return lock


def _run_unit(
    unit: Mapping[str, Any],
    *,
    plan_path: Path,
    output_root: Path,
    pretest_lock: Path,
    common: Mapping[str, Any],
    orchestrator_command: Sequence[str],
) -> dict[str, Any]:
    fold = int(unit["outer_fold"])
    seed = int(unit["seed"])
    unit_root = output_root / "units" / _safe_unit_name(fold, seed)
    derived_lock_path = unit_root / "derived_inference_lock.json"
    pretest_sha = sha256_file(pretest_lock)
    locked_paths = {
        Path(binding["path"])
        for binding in unit["_validated_prelock"]["locked_assets"].values()
    }
    locked_paths.add(Path(unit["_validated_prelock"]["selection_lock"]["path"]))
    stages = _validate_stage_plan(
        unit,
        plan_path=plan_path,
        output_root=output_root,
        locked_paths=locked_paths,
        no_action_fast_path=common["policy_selection_status"] == "fail_closed_no_action",
    )
    if derived_lock_path.exists():
        return _verify_derived_lock(
            derived_lock_path,
            expected_pretest_sha=pretest_sha,
            stages=stages,
            unit_root=unit_root,
        )
    # This is the first access to the test-split manifest.  The all-unit
    # pretest lock exists and was fsync'd before this hash is computed.
    test_manifest = _binding(
        unit.get("test_manifest"),
        relative_to=plan_path.parent,
        label=f"unit {fold}/{seed} test manifest",
    )
    receipts = [
        _run_stage(stage, unit_root=unit_root, position=position, cwd=plan_path.parent)
        for position, stage in enumerate(stages)
    ]
    artifacts_raw = unit["derived_artifacts"]
    required_artifacts = {
        ARTIFACT_FOR_STAGE[str(stage["name"])] for stage in stages
    }
    artifacts = {
        name: bind_file(_resolve(artifacts_raw[name], relative_to=plan_path.parent))
        for name in required_artifacts
    }
    _validate_target_free_npz(
        Path(artifacts["test_proposer_prediction"]["path"]),
        label="test proposer prediction",
    )
    if "derived_cache_manifest" in artifacts:
        _validate_derived_cache_manifest(
            Path(artifacts["derived_cache_manifest"]["path"]), fold=fold, seed=seed
        )
    raw = _raw_arrays(Path(artifacts["raw_hcs_prediction"]["path"]))
    prediction_arrays = _sealed_prediction_arrays(
        raw, fold=fold, seed=seed, policy=common["policy"]
    )
    prediction_path = unit_root / "sealed_label_free_predictions.npz"
    if prediction_path.exists():
        _compare_prediction(prediction_path, prediction_arrays)
    else:
        _atomic_npz(prediction_path, prediction_arrays, immutable=True)
    lock = {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_oof_derived_test_inference",
        "outer_fold": fold,
        "seed": seed,
        "target_artifact_opened": False,
        "capacity_policy_checkpoint_reselection_performed": False,
        "pretest_lock_sha256": pretest_sha,
        "common_selection_lock": common["bindings"]["selection_lock"],
        "common_capacity_selection": common["bindings"]["capacity_selection"],
        "common_policy": common["bindings"]["policy"],
        "source_freeze_manifest": common["bindings"]["source_freeze_manifest"],
        "unit_selection_lock": unit["_validated_prelock"]["selection_lock"],
        "unit_locked_assets": unit["_validated_prelock"]["locked_assets"],
        "test_manifest": test_manifest,
        "stage_receipts": [bind_file(_stage_receipt_path(unit_root, i, stage["name"])) for i, stage in enumerate(stages)],
        "commands": [{"stage": stage["name"], "argv": stage["argv"]} for stage in stages],
        "orchestrator_command": list(orchestrator_command),
        "derived_artifacts": artifacts,
        "sealed_prediction": bind_file(prediction_path),
        "frozen_policy_status": common["policy_selection_status"],
        "no_action_bit_exact_float32_fallback": bool(
            common["policy_selection_status"] == "fail_closed_no_action"
        ),
    }
    _atomic_json(derived_lock_path, lock, immutable=True)
    return lock


def _progress(output_root: Path, locks: Sequence[Mapping[str, Any]], *, sealed: bool) -> None:
    records = sorted(
        (
            {
                "outer_fold": int(lock["outer_fold"]),
                "seed": int(lock["seed"]),
                "derived_lock": bind_file(
                    output_root
                    / "units"
                    / _safe_unit_name(int(lock["outer_fold"]), int(lock["seed"]))
                    / "derived_inference_lock.json"
                ),
            }
            for lock in locks
        ),
        key=lambda value: (value["seed"], value["outer_fold"]),
    )
    _atomic_json(
        output_root / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "classification": "locked_hcs_oof_label_free_progress",
            "completed_units": len(records),
            "expected_units": EXPECTED_UNIT_COUNT,
            "predictions_sealed": bool(sealed),
            "target_artifact_opened": False,
            "units": records,
        },
    )


def _seal_document(
    *, output_root: Path, pretest_lock: Path, locks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    records = []
    for lock in sorted(locks, key=lambda value: (int(value["seed"]), int(value["outer_fold"]))):
        unit_root = output_root / "units" / _safe_unit_name(
            int(lock["outer_fold"]), int(lock["seed"])
        )
        records.append(
            {
                "outer_fold": int(lock["outer_fold"]),
                "seed": int(lock["seed"]),
                "derived_lock": bind_file(unit_root / "derived_inference_lock.json"),
                "prediction": bind_file(unit_root / "sealed_label_free_predictions.npz"),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_oof_all_label_free_predictions_sealed",
        "pretest_lock_sha256": sha256_file(pretest_lock),
        "unit_count": EXPECTED_UNIT_COUNT,
        "outer_folds": list(FOLDS),
        "target_artifact_opened_before_seal": False,
        "target_join_authorized": True,
        "units": records,
    }


def run_inference(
    plan_path: Path,
    output_root: Path,
    *,
    max_units: int | None = None,
    orchestrator_command: Sequence[str] = (),
) -> dict[str, Any]:
    """Run/resume only the target-free half of the locked OOF campaign."""

    plan, resolved_plan = load_plan(plan_path)
    output = output_root.expanduser().resolve()
    common = _validate_common(plan, plan_path=resolved_plan)
    validated_units = [
        _validate_unit_prelock(unit, plan_path=resolved_plan, common=common)
        for unit in plan["units"]
    ]
    pretest = _pretest_document(
        plan,
        plan_path=resolved_plan,
        common=common,
        units=validated_units,
    )
    pretest_path = output / "pretest_lock.json"
    _ensure_immutable_document(pretest_path, pretest)
    units = []
    for raw, validated in zip(plan["units"], validated_units, strict=True):
        enriched = dict(raw)
        enriched["_validated_prelock"] = validated
        units.append(enriched)
    units.sort(key=lambda value: (int(value["seed"]), int(value["outer_fold"])))
    limit = len(units) if max_units is None else int(max_units)
    if limit < 0:
        raise ValueError("max_units cannot be negative")
    completed: list[dict[str, Any]] = []
    for unit in units[:limit]:
        completed.append(
            _run_unit(
                unit,
                plan_path=resolved_plan,
                output_root=output,
                pretest_lock=pretest_path,
                common=common,
                orchestrator_command=orchestrator_command,
            )
        )
        _progress(output, completed, sealed=False)
    if len(completed) != EXPECTED_UNIT_COUNT:
        _progress(output, completed, sealed=False)
        return {
            "status": "label_free_inference_incomplete",
            "completed_units": len(completed),
            "expected_units": EXPECTED_UNIT_COUNT,
            "pretest_lock": bind_file(pretest_path),
            "target_join_authorized": False,
        }
    seal_path = output / "predictions_seal.json"
    seal = _seal_document(output_root=output, pretest_lock=pretest_path, locks=completed)
    _ensure_immutable_document(seal_path, seal)
    _progress(output, completed, sealed=True)
    return {
        "status": "label_free_predictions_sealed",
        "completed_units": EXPECTED_UNIT_COUNT,
        "pretest_lock": bind_file(pretest_path),
        "predictions_seal": bind_file(seal_path),
        "target_join_authorized": True,
    }


def _verify_predictions_seal(output_root: Path) -> dict[str, Any]:
    seal_path = output_root / "predictions_seal.json"
    if not seal_path.is_file():
        raise LockedOOFError("target join forbidden: all 18 predictions are not sealed")
    seal = _json(seal_path, "predictions seal")
    units = seal.get("units")
    if (
        seal.get("target_join_authorized") is not True
        or int(seal.get("unit_count", -1)) != EXPECTED_UNIT_COUNT
        or not isinstance(units, list)
        or len(units) != EXPECTED_UNIT_COUNT
    ):
        raise LockedOOFError("target join forbidden: prediction seal is incomplete")
    keys: set[tuple[int, int]] = set()
    for unit in units:
        key = (int(unit.get("outer_fold", -1)), int(unit.get("seed", -1)))
        if key in keys:
            raise LockedOOFError("prediction seal contains a duplicate unit")
        keys.add(key)
        _binding(unit.get("derived_lock"), relative_to=output_root, label="sealed derived lock")
        prediction = _binding(unit.get("prediction"), relative_to=output_root, label="sealed prediction")
        _validate_target_free_npz(Path(prediction["path"]), label="sealed prediction")
    if {fold for fold, _ in keys} != set(FOLDS) or len({seed for _, seed in keys}) != 3:
        raise LockedOOFError("prediction seal topology is inconsistent")
    return seal


def _metrics(target: np.ndarray, prediction: np.ndarray, identity: np.ndarray) -> dict[str, Any]:
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


def join_and_evaluate(
    output_root: Path,
    target_npz: Path,
    *,
    orchestrator_command: Sequence[str] = (),
) -> dict[str, Any]:
    """Open targets only after the complete label-free prediction seal exists."""

    output = output_root.expanduser().resolve()
    seal = _verify_predictions_seal(output)  # Must precede target resolve/open.
    seal_path = output / "predictions_seal.json"
    target_path = target_npz.expanduser().resolve()
    if not target_path.is_file():
        raise LockedOOFError(f"target artifact is absent: {target_path}")
    target_hash = sha256_file(target_path)
    evaluation_lock_path = output / "evaluation_lock.json"
    if evaluation_lock_path.exists():
        lock = _json(evaluation_lock_path, "evaluation lock")
        if lock.get("target_artifact", {}).get("sha256") != target_hash:
            raise LockedOOFError("evaluation already sealed against another target artifact")
        for binding in lock.get("outputs", {}).values():
            _binding(binding, relative_to=output, label="evaluation output")
        return lock
    try:
        with np.load(target_path, allow_pickle=False) as archive:
            required = {"cache_index", "outer_fold", "target_rr_bpm", "identity"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise LockedOOFError(f"target artifact fields are missing: {missing}")
            target_index = np.asarray(archive["cache_index"], dtype=np.int64)
            target_fold = np.asarray(archive["outer_fold"], dtype=np.int16)
            target_rr = np.asarray(archive["target_rr_bpm"], dtype=np.float32)
            identity = np.asarray(archive["identity"]).astype(str)
            reference_valid = (
                np.asarray(archive["reference_valid"], dtype=bool)
                if "reference_valid" in archive.files
                else np.isfinite(target_rr)
            )
    except (OSError, ValueError) as exc:
        raise LockedOOFError(f"invalid target artifact: {target_path} ({exc})") from exc
    rows = len(target_index)
    if any(array.shape != (rows,) for array in (target_fold, target_rr, identity, reference_valid)):
        raise LockedOOFError("target artifact arrays have inconsistent shapes")
    if len(np.unique(target_index)) != rows or not np.isin(target_fold, FOLDS).all():
        raise LockedOOFError("target artifact indices/folds are invalid")
    target_lookup = {int(index): pos for pos, index in enumerate(target_index)}
    joined: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "cache_index", "outer_fold", "seed", "target_rr_bpm", "identity",
            "fallback_rr_bpm", "source_rr_bpm", "final_rr_bpm",
        )
    }
    per_seed_rows: dict[int, list[int]] = {}
    coverage_by_seed: dict[int, list[int]] = {}
    for unit in seal["units"]:
        fold = int(unit["outer_fold"])
        seed = int(unit["seed"])
        prediction_path = Path(unit["prediction"]["path"])
        with np.load(prediction_path, allow_pickle=False) as archive:
            index = np.asarray(archive["cache_index"], dtype=np.int64)
            fallback = np.asarray(archive["fallback_rr_bpm"], dtype=np.float32)
            source = np.asarray(archive["source_rr_bpm"], dtype=np.float32)
            final = np.asarray(archive["final_rr_bpm"], dtype=np.float32)
        positions: list[int] = []
        for value in index:
            if int(value) not in target_lookup:
                raise LockedOOFError("prediction cache_index is absent from target artifact")
            position = target_lookup[int(value)]
            if int(target_fold[position]) != fold:
                raise LockedOOFError("prediction outer fold disagrees with target ownership")
            positions.append(position)
        if len(index) != len(np.unique(index)) or any(
            array.shape != index.shape for array in (fallback, source, final)
        ):
            raise LockedOOFError("sealed prediction arrays are inconsistent")
        coverage_by_seed.setdefault(seed, []).extend(map(int, index))
        valid_position = np.asarray(positions, dtype=np.int64)
        valid = reference_valid[valid_position] & np.isfinite(target_rr[valid_position])
        if not valid.any():
            raise LockedOOFError("sealed unit has no valid reference rows")
        selected_position = valid_position[valid]
        selected_local = np.flatnonzero(valid)
        start = sum(len(part) for part in joined["cache_index"])
        count = len(selected_position)
        per_seed_rows.setdefault(seed, []).extend(range(start, start + count))
        joined["cache_index"].append(index[selected_local])
        joined["outer_fold"].append(np.full(count, fold, dtype=np.int16))
        joined["seed"].append(np.full(count, seed, dtype=np.int64))
        joined["target_rr_bpm"].append(target_rr[selected_position])
        joined["identity"].append(identity[selected_position].astype(np.str_))
        joined["fallback_rr_bpm"].append(fallback[selected_local])
        joined["source_rr_bpm"].append(source[selected_local])
        joined["final_rr_bpm"].append(final[selected_local])
    expected_indices = set(map(int, target_index))
    for seed, values in coverage_by_seed.items():
        if len(values) != len(set(values)) or set(values) != expected_indices:
            raise LockedOOFError(f"fold predictions do not exactly cover target rows for seed {seed}")
    arrays = {name: np.concatenate(parts) for name, parts in joined.items()}
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "classification": "retrospective_locked_hcs_oof_evaluation",
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
        "per_seed": {},
    }
    for seed, selected_rows in sorted(per_seed_rows.items()):
        position = np.asarray(selected_rows, dtype=np.int64)
        metrics["per_seed"][str(seed)] = {
            "fallback": _metrics(
                arrays["target_rr_bpm"][position],
                arrays["fallback_rr_bpm"][position],
                arrays["identity"][position],
            ),
            "source": _metrics(
                arrays["target_rr_bpm"][position],
                arrays["source_rr_bpm"][position],
                arrays["identity"][position],
            ),
            "locked_final": _metrics(
                arrays["target_rr_bpm"][position],
                arrays["final_rr_bpm"][position],
                arrays["identity"][position],
            ),
        }
    joined_path = output / "locked_hcs_oof_joined.npz"
    metrics_path = output / "locked_hcs_oof_metrics.json"
    if joined_path.exists() or metrics_path.exists():
        raise LockedOOFError("unsealed partial target-join outputs require quarantine")
    _atomic_npz(joined_path, arrays, immutable=True)
    _atomic_json(metrics_path, metrics, immutable=True)
    lock = {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_oof_single_target_join_seal",
        "predictions_seal": bind_file(seal_path),
        "target_artifact": bind_file(target_path),
        "target_join_count": 1,
        "orchestrator_command": list(orchestrator_command),
        "outputs": {
            "joined_oof": bind_file(joined_path),
            "metrics": bind_file(metrics_path),
        },
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
    }
    _atomic_json(evaluation_lock_path, lock, immutable=True)
    return lock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="materialize the concrete 18-unit plan from the fixed-i3 index"
    )
    prepare.add_argument("--pretest-index", type=Path, default=DEFAULT_PRETEST_INDEX)
    prepare.add_argument("--test-manifest-root", type=Path, default=DEFAULT_TEST_MANIFEST_ROOT)
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_POSTLOCK_ROOT)
    prepare.add_argument("--plan-output", type=Path)
    prepare.add_argument("--rf-cache", type=Path, default=DEFAULT_RF_CACHE)
    prepare.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    prepare.add_argument(
        "--proposer-trainer", type=Path, default=PROJECT_ROOT / "scripts/train.py"
    )
    prepare.add_argument(
        "--safe-helper",
        type=Path,
        default=PROJECT_ROOT / "scripts/build_locked_hcs_test_inputs.py",
    )
    prepare.add_argument(
        "--gpu-wrapper", type=Path, default=PROJECT_ROOT / "scripts/run_gpu_admitted.py"
    )
    prepare.add_argument("--gpu-lock", type=Path)
    prepare.add_argument("--gpu-ledger", type=Path)
    prepare.add_argument("--train-device", default="cuda")
    prepare.add_argument("--prediction-device", default="cpu")
    inference = subparsers.add_parser("infer", help="run/resume target-free inference")
    inference.add_argument("--plan", type=Path, required=True)
    inference.add_argument("--output-root", type=Path, required=True)
    inference.add_argument("--max-units", type=int)
    join = subparsers.add_parser("join", help="join targets after the 18-unit seal")
    join.add_argument("--output-root", type=Path, required=True)
    join.add_argument("--targets", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(parsed_argv)
    command = [str(Path(__file__).resolve()), *parsed_argv]
    if args.mode == "prepare":
        output_root = args.output_root.expanduser().resolve()
        result = prepare_plan(
            pretest_index=args.pretest_index,
            test_manifest_root=args.test_manifest_root,
            output_root=output_root,
            plan_output=(
                args.plan_output
                if args.plan_output is not None
                else output_root / "locked_oof_plan.json"
            ),
            rf_cache=args.rf_cache,
            python_executable=args.python_executable,
            proposer_trainer=args.proposer_trainer,
            safe_helper=args.safe_helper,
            gpu_wrapper=args.gpu_wrapper,
            gpu_lock=(
                args.gpu_lock
                if args.gpu_lock is not None
                else output_root / "test_proposer_gpu.lock"
            ),
            gpu_ledger=(
                args.gpu_ledger
                if args.gpu_ledger is not None
                else output_root / "test_proposer_gpu_ledger.jsonl"
            ),
            train_device=args.train_device,
            prediction_device=args.prediction_device,
        )
    elif args.mode == "infer":
        result = run_inference(
            args.plan,
            args.output_root,
            max_units=args.max_units,
            orchestrator_command=command,
        )
    else:
        result = join_and_evaluate(
            args.output_root,
            args.targets,
            orchestrator_command=command,
        )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
