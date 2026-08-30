#!/usr/bin/env python3
"""Run the six-fold, three-seed nested proposer campaign without outer-test access.

This driver deliberately has no outer-test input.  It constructs a separate set
of non-test manifests, reuses content-identical completed proposer units from
the discovery run root, and writes its plan/index/progress outside that reusable
run root.  A selected unit is always one of four inner-OOF predictions or the
outer-validation prediction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_nested_proposer_manifests as manifest_builder  # noqa: E402


CAMPAIGN_ROOT = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer"
    / "full_oof_non_test"
)
DEFAULT_MANIFEST_ROOT = CAMPAIGN_ROOT / "manifests"
DEFAULT_CONTROL_ROOT = CAMPAIGN_ROOT / "control"
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/nested_proposer"
)
DEFAULT_ASSIGNMENTS = (
    PROJECT_ROOT
    / "artifacts/runs/final_alias_gate_s12_deterministic/fold_assignments.json"
)
# Keep the command-line spelling used by the completed discovery units.  The
# manifest binding below always resolves and hashes this path before use.
DEFAULT_CACHE_DIR = Path("artifacts/cache/rf32s")
DEFAULT_GPU_LOCK = (
    PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/gpu_admission.lock"
)
DEFAULT_GPU_LEDGER = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/gpu_admission_ledger.jsonl"
)
ALLOWED_ROLES = frozenset({"hcs_train_oof", "hcs_validation"})
CAMPAIGN_SOURCE_PATHS = (
    Path("scripts/run_full_nested_proposer_campaign.py"),
    Path("scripts/build_nested_proposer_manifests.py"),
    Path("scripts/train.py"),
    Path("scripts/predict_custom_split_all_windows.py"),
    Path("scripts/run_gpu_admitted.py"),
)
CRITICAL_TRAIN_ARGUMENTS: Mapping[str, Any] = {
    "model": "both",
    "batch_size": 48,
    "deterministic": True,
    "causal_history": True,
    "harmonic_head": True,
    "alias_gate": True,
    "aux_fusion": "structured",
    "exact_aux_alignment": True,
    "simulation_steps": 12,
    "hidden_dim": 192,
    "radar_dropout": 0.20,
    "distill_weight": 0.35,
    "distill_temperature": 2.0,
    "alias_loss_weight": 0.05,
    "quality_loss_weight": 0.15,
    "spike_rate_weight": 0.0005,
}


CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _has_test_manifest_marker(path: Path) -> bool:
    return any(part.lower().startswith("test_pred_") for part in path.parts)


def _assert_non_test_path(path: Path, label: str) -> None:
    if _has_test_manifest_marker(path):
        raise RuntimeError(f"outer-test path is forbidden in {label}: {path}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _assert_non_test_path(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label}: {path} ({error})") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _parse_int_csv(raw: str, label: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{label} must be a comma-separated integer list") from error
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be non-empty and contain no duplicates")
    return values


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _validate_roots(manifest_root: Path, control_root: Path, run_root: Path) -> None:
    if _paths_overlap(control_root, run_root):
        raise RuntimeError("control/index root must be separate from reusable run root")
    if _paths_overlap(manifest_root, run_root):
        raise RuntimeError("non-test manifest root must be separate from reusable run root")
    if _paths_overlap(manifest_root, control_root):
        raise RuntimeError("manifest and control roots must be separate siblings")
    for root, label in (
        (manifest_root, "manifest root"),
        (control_root, "control root"),
        (run_root, "run root"),
    ):
        _assert_non_test_path(root, label)


def _assert_no_test_manifest_files(root: Path) -> None:
    if not root.exists():
        return
    # Path enumeration is intentional: forbidden files are rejected by name
    # without opening or parsing them.
    forbidden = next(root.rglob("test_pred_*.json"), None)
    if forbidden is not None:
        raise RuntimeError(
            f"outer-test manifest is forbidden in the non-test campaign root: {forbidden}"
        )


def _write_immutable_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    if path.exists():
        actual = _load_json(path, label)
        if actual != value:
            raise RuntimeError(f"existing immutable {label} differs: {path}")
        expected_hash = hashlib.sha256(_json_bytes(value)).hexdigest()
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"existing immutable {label} byte hash differs: {path}")
        return
    atomic_json(path, value)


def _role_for_manifest(relative: Path) -> str:
    if relative.name.startswith("inner_pred_"):
        return "hcs_train_oof"
    if relative.name.startswith("validation_pred_"):
        return "hcs_validation"
    raise RuntimeError(f"unexpected non-test manifest name: {relative}")


def campaign_source_bindings() -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for relative in CAMPAIGN_SOURCE_PATHS:
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_file():
            raise RuntimeError(f"campaign launch source is missing: {path}")
        bindings[str(relative)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return bindings


def verify_campaign_source_bindings(plan: Mapping[str, Any]) -> None:
    observed = plan.get("source_bindings")
    if not isinstance(observed, Mapping):
        raise RuntimeError("campaign plan has no launch-source bindings")
    expected_names = {str(path) for path in CAMPAIGN_SOURCE_PATHS}
    if set(observed) != expected_names:
        raise RuntimeError("campaign plan launch-source set differs from required sources")
    for name in sorted(expected_names):
        binding = observed.get(name)
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"campaign source binding is invalid: {name}")
        path = Path(str(binding.get("path", ""))).expanduser().resolve()
        expected_path = (PROJECT_ROOT / name).resolve()
        if path != expected_path or not path.is_file():
            raise RuntimeError(f"campaign source path binding mismatch: {name}")
        if path.stat().st_size != int(binding.get("bytes", -1)):
            raise RuntimeError(f"campaign source size changed after plan lock: {name}")
        if sha256_file(path) != binding.get("sha256"):
            raise RuntimeError(f"campaign source hash changed after plan lock: {name}")


def materialize_non_test_manifests(
    *,
    assignments_path: Path,
    cache_manifest: Path,
    manifest_root: Path,
    outer_folds: Sequence[int],
) -> tuple[list[tuple[Path, dict[str, Any]]], dict[str, Any]]:
    """Create only inner/validation manifests and verify existing copies exactly."""

    _assert_no_test_manifest_files(manifest_root)
    records, summary = manifest_builder.build_plan(
        assignments_path=assignments_path,
        cache_manifest=cache_manifest,
        outer_folds=outer_folds,
        include_outer_test=False,
    )
    expected_count = 5 * len(outer_folds)
    if len(records) != expected_count:
        raise RuntimeError(
            f"non-test topology must contain {expected_count} manifests, got {len(records)}"
        )
    observed_outer_counts = {int(outer): {"inner": 0, "validation": 0} for outer in outer_folds}
    for relative, manifest in records:
        _assert_non_test_path(relative, "generated manifest")
        if len(relative.parts) != 2 or not relative.parent.name.startswith("outer_"):
            raise RuntimeError(f"invalid non-test manifest layout: {relative}")
        outer = int(relative.parent.name.removeprefix("outer_"))
        if outer not in observed_outer_counts:
            raise RuntimeError(f"unexpected outer fold in non-test manifest: {outer}")
        role = _role_for_manifest(relative)
        observed_outer_counts[outer][
            "inner" if role == "hcs_train_oof" else "validation"
        ] += 1
        if manifest.get("content_sha256") != manifest_builder.canonical_content_sha256(
            manifest
        ):
            raise RuntimeError(f"generated manifest content hash mismatch: {relative}")
        _write_immutable_json(manifest_root / relative, manifest, "non-test manifest")
    for outer, counts in observed_outer_counts.items():
        if counts != {"inner": 4, "validation": 1}:
            raise RuntimeError(f"outer {outer} has invalid non-test topology: {counts}")
    _assert_no_test_manifest_files(manifest_root)
    return records, summary


def _unit_paths(
    *, run_root: Path, seed: int, outer: int, relative_manifest: Path, fold_id: int
) -> dict[str, Path]:
    stem = relative_manifest.stem
    _assert_non_test_path(Path(stem), "unit stem")
    output_dir = run_root / f"seed_{seed}" / f"outer_{outer}" / stem
    fold_dir = output_dir / f"fold_{fold_id}"
    return {
        "output_dir": output_dir,
        "fold_dir": fold_dir,
        "checkpoint": fold_dir / "snn_best.pt",
        "metrics": output_dir / "metrics.json",
        "run_config": output_dir / "run_config.json",
        "split_authority": output_dir / "split_authority.json",
        "prediction": fold_dir / "snn_prediction_all_windows.npz",
    }


def build_campaign_plan(
    *,
    manifest_records: Sequence[tuple[Path, Mapping[str, Any]]],
    manifest_summary: Mapping[str, Any],
    manifest_root: Path,
    run_root: Path,
    assignments_path: Path,
    cache_manifest: Path,
    outer_folds: Sequence[int],
    seeds: Sequence[int],
    epochs: int,
    patience: int,
) -> dict[str, Any]:
    units: list[dict[str, Any]] = []
    for seed in seeds:
        for relative, manifest in manifest_records:
            role = _role_for_manifest(relative)
            if role not in ALLOWED_ROLES:
                raise RuntimeError(f"forbidden campaign role: {role}")
            outer = int(relative.parent.name.removeprefix("outer_"))
            manifest_path = (manifest_root / relative).resolve()
            paths = _unit_paths(
                run_root=run_root,
                seed=seed,
                outer=outer,
                relative_manifest=relative,
                fold_id=int(manifest["fold_id"]),
            )
            unit_id = f"seed_{seed}/outer_{outer}/{relative.stem}"
            units.append(
                {
                    "unit_id": unit_id,
                    "seed": int(seed),
                    "outer_fold": outer,
                    "role": role,
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "manifest_content_sha256": manifest["content_sha256"],
                    "output_dir": str(paths["output_dir"]),
                    "checkpoint": str(paths["checkpoint"]),
                    "all_window_prediction": str(paths["prediction"]),
                }
            )
    result: dict[str, Any] = {
        "schema_version": 1,
        "classification": "retrospective_fully_nested_non_test_proposer_campaign_plan",
        "outer_test_opened": False,
        "outer_test_record_count": 0,
        "outer_folds": [int(value) for value in outer_folds],
        "seeds": [int(value) for value in seeds],
        "requested_units": len(units),
        "roles": sorted(ALLOWED_ROLES),
        "manifest_plan_content_sha256": manifest_summary["content_sha256"],
        "manifest_root": str(manifest_root),
        "fold_assignments": {
            "path": str(assignments_path),
            "sha256": sha256_file(assignments_path),
        },
        "cache_manifest": {
            "path": str(cache_manifest),
            "sha256": sha256_file(cache_manifest),
        },
        "reusable_run_root": str(run_root),
        "control_index_separate_from_run_root": True,
        "source_bindings": campaign_source_bindings(),
        "training_specification": {
            "epochs": int(epochs),
            "patience": int(patience),
            **dict(CRITICAL_TRAIN_ARGUMENTS),
        },
        "units": units,
    }
    if any(
        unit.get("role") not in ALLOWED_ROLES
        or _has_test_manifest_marker(Path(str(unit.get("manifest", ""))))
        for unit in units
    ):
        raise RuntimeError("outer-test unit entered the campaign plan")
    result["content_sha256"] = canonical_content_sha256(result)
    return result


def _expected_authority(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    identities = manifest.get("identities")
    if not isinstance(identities, Mapping):
        raise RuntimeError(f"manifest identities are invalid: {manifest_path}")
    return {
        "mode": "custom_identity_split",
        "schema_version": 1,
        "fold_id": int(manifest["fold_id"]),
        "split_manifest_content_sha256": manifest["content_sha256"],
        "split_manifest_file_sha256": sha256_file(manifest_path),
        "fold_assignments_sha256": manifest["fold_assignments"]["sha256"],
        "cache_manifest_sha256": manifest["cache"]["manifest_sha256"],
        "train_identities": list(identities["train"]),
        "validation_identities": list(identities["validation"]),
        "prediction_identities": list(identities["prediction"]),
        "excluded_identities": list(identities["excluded"]),
        "scaler_identities": list(identities["scaler"]),
    }


def _validate_manifest(
    path: Path, *, expected_content_hash: str, expected_file_hash: str
) -> dict[str, Any]:
    manifest = _load_json(path, "non-test manifest")
    if canonical_content_sha256(manifest) != manifest.get("content_sha256"):
        raise RuntimeError(f"manifest canonical content hash mismatch: {path}")
    if manifest.get("content_sha256") != expected_content_hash:
        raise RuntimeError(f"manifest differs from immutable campaign plan: {path}")
    if sha256_file(path) != expected_file_hash:
        raise RuntimeError(f"manifest byte hash differs from immutable campaign plan: {path}")
    return manifest


def _validate_run_config(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    run_config = _load_json(path, "run config")
    arguments = run_config.get("arguments")
    if not isinstance(arguments, Mapping):
        raise RuntimeError(f"run config arguments are missing: {path}")
    if int(arguments.get("seed", -1)) != seed:
        raise RuntimeError(f"run config seed mismatch: {path}")
    if arguments.get("identity_split_manifest_sha256") != manifest["content_sha256"]:
        raise RuntimeError(f"run config manifest binding mismatch: {path}")
    for key, expected in CRITICAL_TRAIN_ARGUMENTS.items():
        if arguments.get(key) != expected:
            raise RuntimeError(f"run config argument mismatch for {key}: {path}")
    if run_config.get("split_authority") != authority:
        raise RuntimeError(f"run config split authority mismatch: {path}")
    if not isinstance(run_config.get("run_signature"), str):
        raise RuntimeError(f"run config signature is missing: {path}")
    return run_config


def _validate_checkpoint(
    path: Path,
    *,
    authority: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"invalid proposer checkpoint: {path} ({error})") from error
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"proposer checkpoint must be a mapping: {path}")
    if checkpoint.get("format_version") != 2 or checkpoint.get("model_type") != "snn":
        raise RuntimeError(f"proposer checkpoint is not a format-2 SNN: {path}")
    if int(checkpoint.get("fold", -1)) != int(authority["fold_id"]):
        raise RuntimeError(f"proposer checkpoint fold mismatch: {path}")
    if checkpoint.get("split_authority_provenance") != authority:
        raise RuntimeError(f"proposer checkpoint split authority mismatch: {path}")
    if checkpoint.get("run_signature") != run_config.get("run_signature"):
        raise RuntimeError(f"proposer checkpoint run signature mismatch: {path}")
    return checkpoint, sha256_file(path)


def _npz_scalar(archive: Mapping[str, Any], name: str) -> Any:
    if name not in archive:
        raise RuntimeError(f"prediction is missing scalar field: {name}")
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise RuntimeError(f"prediction field must be scalar: {name}")
    return value.item()


def _validate_prediction(
    path: Path,
    *,
    authority: Mapping[str, Any],
    checkpoint_hash: str,
) -> str:
    _assert_non_test_path(path, "prediction")
    required_rows = ("cache_index", "identity", "prediction")
    required_scalars = (
        "fold_id",
        "checkpoint_sha256",
        "split_manifest_file_sha256",
        "split_manifest_content_sha256",
        "fold_assignments_sha256",
        "cache_manifest_sha256",
        "strict_retrospective",
        "strict_nested_prediction_role",
        "provenance_json",
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set((*required_rows, *required_scalars)) - set(archive.files))
            if missing:
                raise RuntimeError(f"prediction is missing fields: {missing}")
            if int(_npz_scalar(archive, "fold_id")) != int(authority["fold_id"]):
                raise RuntimeError("prediction fold binding mismatch")
            expected_scalars = {
                "checkpoint_sha256": checkpoint_hash,
                "split_manifest_file_sha256": authority[
                    "split_manifest_file_sha256"
                ],
                "split_manifest_content_sha256": authority[
                    "split_manifest_content_sha256"
                ],
                "fold_assignments_sha256": authority["fold_assignments_sha256"],
                "cache_manifest_sha256": authority["cache_manifest_sha256"],
            }
            for field, expected in expected_scalars.items():
                if str(_npz_scalar(archive, field)) != str(expected):
                    raise RuntimeError(f"prediction provenance mismatch: {field}")
            if bool(_npz_scalar(archive, "strict_retrospective")) is not True:
                raise RuntimeError("prediction is not marked strict retrospective")
            if bool(_npz_scalar(archive, "strict_nested_prediction_role")) is not True:
                raise RuntimeError("prediction is not marked as a nested prediction role")
            indices = np.asarray(archive["cache_index"], dtype=np.int64)
            identities = np.asarray(archive["identity"]).astype(str)
            predictions = np.asarray(archive["prediction"])
            if indices.ndim != 1 or len(indices) == 0 or len(np.unique(indices)) != len(indices):
                raise RuntimeError("prediction cache indices are empty, invalid or duplicated")
            if identities.shape != indices.shape or predictions.shape != indices.shape:
                raise RuntimeError("prediction row fields have inconsistent shapes")
            expected_identities = set(authority["prediction_identities"])
            if set(identities.tolist()) != expected_identities:
                raise RuntimeError("prediction identity ownership differs from manifest")
            if not np.isfinite(predictions).all():
                raise RuntimeError("prediction contains non-finite model outputs")
            provenance = json.loads(str(_npz_scalar(archive, "provenance_json")))
            if not isinstance(provenance, dict):
                raise RuntimeError("prediction provenance must be an object")
            provenance_expectations = {
                "checkpoint_sha256": checkpoint_hash,
                "split_manifest_file_sha256": authority[
                    "split_manifest_file_sha256"
                ],
                "split_manifest_content_sha256": authority[
                    "split_manifest_content_sha256"
                ],
                "fold_assignments_sha256": authority["fold_assignments_sha256"],
                "cache_manifest_sha256": authority["cache_manifest_sha256"],
                "strict_nested_role": "prediction",
                "labels_forwarded_to_model": False,
            }
            for field, expected in provenance_expectations.items():
                if provenance.get(field) != expected:
                    raise RuntimeError(f"prediction JSON provenance mismatch: {field}")
            if set(provenance.get("prediction_identities", [])) != expected_identities:
                raise RuntimeError("prediction JSON identity ownership mismatch")
            recorded_path = Path(str(provenance.get("split_manifest_path", "")))
            _assert_non_test_path(recorded_path, "prediction provenance manifest")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid proposer prediction: {path} ({error})") from error
    return sha256_file(path)


def _validate_metrics(path: Path, *, fold_id: int, checkpoint_path: Path) -> None:
    metrics = _load_json(path, "training metrics")
    fold = metrics.get("folds", {}).get(str(fold_id))
    if not isinstance(fold, Mapping):
        raise RuntimeError(f"training metrics omit fold {fold_id}: {path}")
    model = fold.get("models", {}).get("snn")
    if not isinstance(model, Mapping):
        raise RuntimeError(f"training metrics omit SNN result: {path}")
    recorded = Path(str(model.get("best_checkpoint", ""))).expanduser()
    if not recorded.is_absolute():
        recorded = (PROJECT_ROOT / recorded).resolve()
    if recorded != checkpoint_path:
        raise RuntimeError(f"training metrics checkpoint path mismatch: {path}")


def _complete_record(
    *,
    unit: Mapping[str, Any],
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    authority = _expected_authority(manifest, Path(str(unit["manifest"])))
    run_config = _validate_run_config(
        paths["run_config"],
        manifest=manifest,
        authority=authority,
        seed=int(unit["seed"]),
    )
    split_authority = _load_json(paths["split_authority"], "split authority")
    expected_run_authority = {
        "cache_manifest_path": str(
            Path(str(manifest["cache"]["manifest_path"])).resolve()
        ),
        "cache_manifest_sha256": authority["cache_manifest_sha256"],
        "excluded_identities": authority["excluded_identities"],
        "fold_assignments_path": str(
            Path(str(manifest["fold_assignments"]["path"])).resolve()
        ),
        "fold_assignments_sha256": authority["fold_assignments_sha256"],
        "fold_id": authority["fold_id"],
        "mode": "custom_identity_split",
        "prediction_identities": authority["prediction_identities"],
        "scaler_identities": authority["scaler_identities"],
        "schema_version": 1,
        "split_manifest_content_sha256": authority[
            "split_manifest_content_sha256"
        ],
        "split_manifest_file_sha256": authority["split_manifest_file_sha256"],
        "train_identities": authority["train_identities"],
        "validation_identities": authority["validation_identities"],
    }
    for key, expected in expected_run_authority.items():
        if split_authority.get(key) != expected:
            raise RuntimeError(
                f"split-authority run provenance mismatch for {key}: {paths['split_authority']}"
            )
    recorded_manifest_path = Path(str(split_authority.get("split_manifest_path", "")))
    _assert_non_test_path(recorded_manifest_path, "split authority manifest")
    _, checkpoint_hash = _validate_checkpoint(
        paths["checkpoint"], authority=authority, run_config=run_config
    )
    _validate_metrics(
        paths["metrics"],
        fold_id=int(manifest["fold_id"]),
        checkpoint_path=paths["checkpoint"],
    )
    prediction_hash = _validate_prediction(
        paths["prediction"],
        authority=authority,
        checkpoint_hash=checkpoint_hash,
    )
    return {
        "unit_id": unit["unit_id"],
        "seed": int(unit["seed"]),
        "outer_fold": int(unit["outer_fold"]),
        "role": unit["role"],
        "manifest": str(Path(str(unit["manifest"])).resolve()),
        "manifest_sha256": unit["manifest_sha256"],
        "manifest_content_sha256": unit["manifest_content_sha256"],
        "output_dir": str(paths["output_dir"]),
        "checkpoint": {
            "path": str(paths["checkpoint"]),
            "sha256": checkpoint_hash,
            "bytes": paths["checkpoint"].stat().st_size,
        },
        "all_window_prediction": {
            "path": str(paths["prediction"]),
            "sha256": prediction_hash,
            "bytes": paths["prediction"].stat().st_size,
        },
    }


def inspect_unit(
    unit: Mapping[str, Any], *, run_root: Path
) -> tuple[str, dict[str, Any] | None, str | None]:
    role = str(unit.get("role", ""))
    if role not in ALLOWED_ROLES:
        raise RuntimeError(f"forbidden unit role: {role}")
    manifest_path = Path(str(unit["manifest"])).resolve()
    _assert_non_test_path(manifest_path, "unit manifest")
    manifest = _validate_manifest(
        manifest_path,
        expected_content_hash=str(unit["manifest_content_sha256"]),
        expected_file_hash=str(unit["manifest_sha256"]),
    )
    relative = Path(f"outer_{int(unit['outer_fold'])}") / manifest_path.name
    paths = _unit_paths(
        run_root=run_root,
        seed=int(unit["seed"]),
        outer=int(unit["outer_fold"]),
        relative_manifest=relative,
        fold_id=int(manifest["fold_id"]),
    )
    if paths["output_dir"] != Path(str(unit["output_dir"])):
        raise RuntimeError(f"unit output path differs from campaign plan: {unit['unit_id']}")
    recognized = {
        name: paths[name].is_file()
        for name in (
            "checkpoint",
            "metrics",
            "run_config",
            "split_authority",
            "prediction",
        )
    }
    training_complete = all(
        recognized[name]
        for name in ("checkpoint", "metrics", "run_config", "split_authority")
    )
    if recognized["prediction"] and not training_complete:
        raise RuntimeError(
            f"prediction exists without a complete bound training unit: {unit['unit_id']}"
        )
    if training_complete and recognized["prediction"]:
        return "complete", _complete_record(unit=unit, manifest=manifest, paths=paths), None
    authority = _expected_authority(manifest, manifest_path)
    if recognized["run_config"]:
        _validate_run_config(
            paths["run_config"],
            manifest=manifest,
            authority=authority,
            seed=int(unit["seed"]),
        )
    if recognized["split_authority"]:
        split_authority = _load_json(paths["split_authority"], "split authority")
        if split_authority.get("split_manifest_content_sha256") != manifest[
            "content_sha256"
        ]:
            raise RuntimeError(
                f"partial unit has inconsistent split authority: {unit['unit_id']}"
            )
    if recognized["checkpoint"]:
        if not recognized["run_config"]:
            raise RuntimeError(
                f"checkpoint exists without a run config: {unit['unit_id']}"
            )
        run_config = _load_json(paths["run_config"], "run config")
        _validate_checkpoint(
            paths["checkpoint"], authority=authority, run_config=run_config
        )
    if training_complete:
        # A complete training result needs only deterministic label-free
        # prediction materialization; it is not retrained.
        _validate_metrics(
            paths["metrics"],
            fold_id=int(manifest["fold_id"]),
            checkpoint_path=paths["checkpoint"],
        )
        return "prediction_pending", None, "all-window prediction is absent"
    if any(recognized.values()) or paths["output_dir"].exists():
        return "training_pending", None, "validated partial training unit"
    return "training_pending", None, "unit has not started"


def build_train_command(
    *, args: argparse.Namespace, unit: Mapping[str, Any]
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train.py"),
        "--identity-split-manifest",
        str(unit["manifest"]),
        "--output-dir",
        str(unit["output_dir"]),
        "--cache-dir",
        str(args.cache_dir),
        "--model",
        "both",
        "--resume",
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        "48",
        "--workers",
        str(args.workers),
        "--device",
        str(args.train_device),
        "--amp",
        "--deterministic",
        "--seed",
        str(unit["seed"]),
        "--causal-history",
        "--harmonic-head",
        "--alias-gate",
        "--aux-fusion",
        "structured",
        "--exact-aux-alignment",
        "--simulation-steps",
        "12",
        "--hidden-dim",
        "192",
        "--radar-dropout",
        "0.20",
        "--distill-weight",
        "0.35",
        "--distill-temperature",
        "2.0",
        "--alias-loss-weight",
        "0.05",
        "--quality-loss-weight",
        "0.15",
        "--spike-rate-weight",
        "0.0005",
        "--bootstrap-samples",
        "500",
    ]


def build_prediction_command(
    *, args: argparse.Namespace, unit: Mapping[str, Any]
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/predict_custom_split_all_windows.py"),
        "--cache-dir",
        str(args.cache_dir),
        "--checkpoint",
        str(unit["checkpoint"]),
        "--identity-split-manifest",
        str(unit["manifest"]),
        "--output",
        str(unit["all_window_prediction"]),
        "--device",
        str(args.prediction_device),
        "--batch-size",
        "128",
        "--workers",
        "0",
    ]
    command.append(
        "--amp" if str(args.prediction_device).startswith("cuda") else "--no-amp"
    )
    return command


def gpu_admitted_command(command: Sequence[str], *, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_gpu_admitted.py"),
        "--lock-file",
        str(args.gpu_lock),
        "--ledger",
        str(args.gpu_ledger),
        "--",
        *command,
    ]


def _execute_command(
    command: Sequence[str],
    *,
    device: str,
    args: argparse.Namespace,
    command_runner: CommandRunner,
) -> None:
    final = (
        gpu_admitted_command(command, args=args)
        if str(device).startswith("cuda")
        else list(command)
    )
    completed = command_runner(final, cwd=PROJECT_ROOT, check=False)
    if int(completed.returncode) != 0:
        raise RuntimeError(
            f"campaign command failed with exit code {completed.returncode}: {final[1]}"
        )


def _safe_existing_control_document(
    path: Path, *, plan_hash: str, allow_missing: bool = True
) -> None:
    if not path.exists():
        if allow_missing:
            return
        raise RuntimeError(f"required control document is missing: {path}")
    value = _load_json(path, "campaign control document")
    if value.get("content_sha256") != canonical_content_sha256(value):
        raise RuntimeError(f"control document canonical hash mismatch: {path}")
    if value.get("outer_test_opened") is not False:
        raise RuntimeError(f"control document is not sealed from outer test: {path}")
    if int(value.get("outer_test_record_count", -1)) != 0:
        raise RuntimeError(f"control document contains outer-test records: {path}")
    binding = value.get("campaign_plan_content_sha256")
    if binding is not None and binding != plan_hash:
        raise RuntimeError(f"control document plan binding mismatch: {path}")
    plan_binding = value.get("campaign_plan")
    if plan_binding is not None:
        if not isinstance(plan_binding, Mapping):
            raise RuntimeError(f"control document plan binding is invalid: {path}")
        if plan_binding.get("content_sha256") != plan_hash:
            raise RuntimeError(f"control document nested plan binding mismatch: {path}")
    records = value.get("records", [])
    if not isinstance(records, list):
        raise RuntimeError(f"control document records must be an array: {path}")
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError(f"control document record is invalid: {path}")
        if record.get("role") not in ALLOWED_ROLES:
            raise RuntimeError(f"control document has a forbidden role: {path}")
        _assert_non_test_path(Path(str(record.get("manifest", ""))), "control record")


def _documents(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_file_sha256: str,
    states: Mapping[str, tuple[str, dict[str, Any] | None, str | None]],
    current_state: str,
    failed_unit: str | None = None,
    failure: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    completed_records = [
        record
        for unit in plan["units"]
        for state, record, _ in [states[str(unit["unit_id"])]]
        if state == "complete" and record is not None
    ]
    progress_units = [
        {
            "unit_id": unit["unit_id"],
            "seed": unit["seed"],
            "outer_fold": unit["outer_fold"],
            "role": unit["role"],
            "manifest": unit["manifest"],
            "state": states[str(unit["unit_id"])][0],
            "detail": states[str(unit["unit_id"])][2],
        }
        for unit in plan["units"]
    ]
    counts: dict[str, int] = {}
    for value, _, _ in states.values():
        counts[value] = counts.get(value, 0) + 1
    common: dict[str, Any] = {
        "schema_version": 1,
        "campaign_plan_content_sha256": plan["content_sha256"],
        "campaign_plan": {
            "path": str(plan_path),
            "sha256": plan_file_sha256,
            "content_sha256": plan["content_sha256"],
        },
        "manifest_root": plan["manifest_root"],
        "manifest_plan_content_sha256": plan["manifest_plan_content_sha256"],
        "outer_test_opened": False,
        "outer_test_record_count": 0,
        "requested_units": int(plan["requested_units"]),
        "completed_units": len(completed_records),
        "updated_utc": utc_now(),
    }
    index: dict[str, Any] = {
        **common,
        "classification": "retrospective_fully_nested_non_test_proposer_index",
        "records": completed_records,
    }
    progress: dict[str, Any] = {
        **common,
        "classification": "retrospective_fully_nested_non_test_proposer_progress",
        "state_counts": counts,
        "failed_unit": failed_unit,
        "failure": failure,
        "units": progress_units,
    }
    status: dict[str, Any] = {
        **common,
        "classification": "retrospective_fully_nested_non_test_proposer_status",
        "state": current_state,
        "pending_units": int(plan["requested_units"]) - len(completed_records),
        "failed_unit": failed_unit,
        "failure": failure,
        "records": [],
    }
    for document in (index, progress, status):
        document["content_sha256"] = canonical_content_sha256(document)
    return index, progress, status


def _write_documents(
    *,
    control_root: Path,
    plan: Mapping[str, Any],
    states: Mapping[str, tuple[str, dict[str, Any] | None, str | None]],
    current_state: str,
    failed_unit: str | None = None,
    failure: str | None = None,
) -> None:
    index, progress, status = _documents(
        plan=plan,
        plan_path=control_root / "plan.json",
        plan_file_sha256=sha256_file(control_root / "plan.json"),
        states=states,
        current_state=current_state,
        failed_unit=failed_unit,
        failure=failure,
    )
    atomic_json(control_root / "index.json", index)
    atomic_json(control_root / "progress.json", progress)
    atomic_json(control_root / "status.json", status)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--outer-folds", default="0,1,2,3,4,5")
    parser.add_argument("--seeds", default="20260828,20260829,20260830")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--prediction-device", default="cpu")
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--gpu-ledger", type=Path, default=DEFAULT_GPU_LEDGER)
    parser.add_argument(
        "--max-new-units",
        type=int,
        help="execute at most this many currently incomplete units in this invocation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="materialize and validate control state without launching training/inference",
    )
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace, *, command_runner: CommandRunner = subprocess.run
) -> dict[str, Any]:
    assignments_path = args.fold_assignments.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_manifest = cache_dir / "manifest.json"
    manifest_root = args.manifest_root.expanduser().resolve()
    control_root = args.control_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    args.gpu_lock = args.gpu_lock.expanduser().resolve()
    args.gpu_ledger = args.gpu_ledger.expanduser().resolve()
    outer_folds = _parse_int_csv(args.outer_folds, "--outer-folds")
    seeds = _parse_int_csv(args.seeds, "--seeds")
    if any(fold not in range(6) for fold in outer_folds):
        raise ValueError("--outer-folds must be drawn from 0..5")
    if args.epochs < 1 or args.patience < 1 or args.workers < 0:
        raise ValueError("epochs/patience must be positive and workers non-negative")
    if args.max_new_units is not None and args.max_new_units < 1:
        raise ValueError("--max-new-units must be positive")
    _validate_roots(manifest_root, control_root, run_root)
    if not assignments_path.is_file() or not cache_manifest.is_file():
        raise RuntimeError("fold assignments and cache manifest must exist")

    manifest_records, manifest_summary = materialize_non_test_manifests(
        assignments_path=assignments_path,
        cache_manifest=cache_manifest,
        manifest_root=manifest_root,
        outer_folds=outer_folds,
    )
    plan = build_campaign_plan(
        manifest_records=manifest_records,
        manifest_summary=manifest_summary,
        manifest_root=manifest_root,
        run_root=run_root,
        assignments_path=assignments_path,
        cache_manifest=cache_manifest,
        outer_folds=outer_folds,
        seeds=seeds,
        epochs=args.epochs,
        patience=args.patience,
    )
    plan_path = control_root / "plan.json"
    _write_immutable_json(plan_path, plan, "campaign plan")
    verify_campaign_source_bindings(plan)
    for name in ("index.json", "progress.json", "status.json"):
        _safe_existing_control_document(
            control_root / name, plan_hash=str(plan["content_sha256"])
        )

    states: dict[str, tuple[str, dict[str, Any] | None, str | None]] = {}
    try:
        for unit in plan["units"]:
            states[str(unit["unit_id"])] = inspect_unit(unit, run_root=run_root)
    except Exception as error:
        # If a scan failed before all units were reached, retain a structurally
        # complete status document without treating unscanned units as results.
        for unit in plan["units"]:
            states.setdefault(
                str(unit["unit_id"]), ("unscanned", None, "scan stopped fail-closed")
            )
        _write_documents(
            control_root=control_root,
            plan=plan,
            states=states,
            current_state="failed",
            failure=str(error),
        )
        raise

    if args.dry_run:
        _write_documents(
            control_root=control_root,
            plan=plan,
            states=states,
            current_state="complete" if all(v[0] == "complete" for v in states.values()) else "dry_run",
        )
        return _load_json(control_root / "status.json", "campaign status")

    executed = 0
    for unit in plan["units"]:
        unit_id = str(unit["unit_id"])
        state = states[unit_id][0]
        if state == "complete":
            continue
        if args.max_new_units is not None and executed >= args.max_new_units:
            break
        try:
            _write_documents(
                control_root=control_root,
                plan=plan,
                states=states,
                current_state="running",
            )
            if state == "training_pending":
                verify_campaign_source_bindings(plan)
                _execute_command(
                    build_train_command(args=args, unit=unit),
                    device=str(args.train_device),
                    args=args,
                    command_runner=command_runner,
                )
            state_after_train, _, _ = inspect_unit(unit, run_root=run_root)
            if state_after_train == "training_pending":
                raise RuntimeError(f"training command did not complete unit: {unit_id}")
            if state_after_train == "prediction_pending":
                verify_campaign_source_bindings(plan)
                _execute_command(
                    build_prediction_command(args=args, unit=unit),
                    device=str(args.prediction_device),
                    args=args,
                    command_runner=command_runner,
                )
            states[unit_id] = inspect_unit(unit, run_root=run_root)
            if states[unit_id][0] != "complete":
                raise RuntimeError(f"unit did not validate complete after execution: {unit_id}")
            executed += 1
            _write_documents(
                control_root=control_root,
                plan=plan,
                states=states,
                current_state="running",
            )
        except Exception as error:
            states[unit_id] = ("failed", None, str(error))
            _write_documents(
                control_root=control_root,
                plan=plan,
                states=states,
                current_state="failed",
                failed_unit=unit_id,
                failure=str(error),
            )
            raise

    final_state = "complete" if all(v[0] == "complete" for v in states.values()) else "paused"
    _write_documents(
        control_root=control_root,
        plan=plan,
        states=states,
        current_state=final_state,
    )
    return _load_json(control_root / "status.json", "campaign status")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        status = run(args)
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "state": status["state"],
                "completed_units": status["completed_units"],
                "requested_units": status["requested_units"],
                "outer_test_opened": False,
                "outer_test_record_count": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
