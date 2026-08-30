#!/usr/bin/env python3
"""Run the sealed, label-free seven-radar-mask HCS campaign.

This program is deliberately an inference orchestrator, not an evaluator.  It
derives 126 immutable work units from a *complete* 18-unit
``run_locked_hcs_oof`` prediction seal, reruns the safe proposer for each of
the seven predeclared non-empty radar masks on CPU, and adapts the result to
the already frozen ``fail_closed_no_action`` source ABI.  No target path is
accepted by the CLI and no target/reference array is permitted in an output.

Every input is hash-bound before execution.  A work unit is published by one
atomic directory rename, so an interrupted process cannot create a unit that
looks complete.  Existing units are fully revalidated on resume.  Only after
all 126 receipts validate is ``complete_seal.json`` created; until that point
downstream evaluation remains unauthorized.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# These imports intentionally make the two existing sealed interfaces the
# executable contract rather than inventing a parallel prediction ABI.
import build_locked_hcs_test_inputs as SAFE_INPUTS  # noqa: E402
import run_locked_hcs_oof as LOCKED_OOF  # noqa: E402


SCHEMA_VERSION = 1
FOLDS = tuple(range(6))
EXPECTED_PRIMARY_UNITS = 18
EXPECTED_MASK_UNITS = 126
LOCKED_SEEDS = (20260828, 20260829, 20260830)
MASKS: dict[str, tuple[bool, bool, bool]] = dict(SAFE_INPUTS.RADAR_MASKS)
MASK_NAMES = tuple(MASKS)
CORE_PROPOSER_FIELDS = {"cache_index", "prediction", "rr_std"}
RAW_REQUIRED = set(LOCKED_OOF.RAW_REQUIRED)
SEALED_REQUIRED = {
    "cache_index",
    "outer_fold",
    "seed",
    "fallback_rr_bpm",
    "source_rr_bpm",
    "final_rr_bpm",
    "applied_pull",
    "target_joined",
}
FORBIDDEN_LABEL_FIELDS = set(LOCKED_OOF.FORBIDDEN_LABEL_FIELDS) | set(
    SAFE_INPUTS.LABEL_FIELDS
)

PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_PRIMARY_ROOT = (
    PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_radar_masks"
)


class RadarMaskCampaignError(RuntimeError):
    """A fail-closed campaign topology, provenance, or artifact error."""


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
        raise RadarMaskCampaignError(f"invalid {label}: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise RadarMaskCampaignError(f"{label} must be a JSON object: {path}")
    return value


def _json_payload(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _atomic_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_json_payload(value))
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


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def bind_file(path: Path, *, recorded_path: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RadarMaskCampaignError(f"required file is absent: {resolved}")
    return {
        "path": str((recorded_path or resolved).expanduser().resolve()),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def bind_python_launcher(path: Path) -> dict[str, Any]:
    """Hash the interpreter binary while preserving the venv launcher path."""

    launcher = Path(os.path.abspath(path.expanduser()))
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise RadarMaskCampaignError("Python launcher must be an executable file")
    binding = bind_file(launcher)
    return {**binding, "path": str(launcher)}


def _resolve(value: Any, *, relative_to: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise RadarMaskCampaignError("artifact binding path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _verified_binding(
    raw: Any, *, relative_to: Path, label: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RadarMaskCampaignError(f"missing file binding: {label}")
    path = _resolve(raw.get("path"), relative_to=relative_to)
    expected = str(raw.get("sha256", ""))
    if len(expected) != 64 or not path.is_file() or sha256_file(path) != expected:
        raise RadarMaskCampaignError(f"file hash mismatch: {label} ({path})")
    if "bytes" in raw and int(raw["bytes"]) != path.stat().st_size:
        raise RadarMaskCampaignError(f"file size mismatch: {label} ({path})")
    return {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        Path(str(left.get("path", ""))).resolve()
        == Path(str(right.get("path", ""))).resolve()
        and str(left.get("sha256", "")) == str(right.get("sha256", ""))
    )


def _ensure_document(path: Path, expected: Mapping[str, Any]) -> None:
    if path.exists():
        observed = _json(path, path.name)
        if observed != expected:
            raise RadarMaskCampaignError(
                f"immutable control document differs from current bindings: {path}"
            )
        return
    _atomic_json(path, expected, immutable=True)


def _argv_value(argv: Sequence[str], option: str, *, label: str) -> str:
    positions = [position for position, token in enumerate(argv) if token == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RadarMaskCampaignError(f"{label} must contain exactly one {option}")
    return str(argv[positions[0] + 1])


def _unit_name(fold: int, seed: int) -> str:
    return f"outer_{fold}_seed_{seed}"


def _mask_unit_id(fold: int, seed: int, mask: str) -> str:
    return f"{_unit_name(fold, seed)}__{mask}"


def _unit_root(output_root: Path, unit: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "units"
        / _unit_name(int(unit["outer_fold"]), int(unit["seed"]))
        / str(unit["radar_mask"])
    )


_UNIT_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "classification",
        "unit_id",
        "outer_fold",
        "seed",
        "radar_mask",
        "radar_mask_pattern",
        "plan",
        "inputs",
        "primary",
        "commands",
        "runtime_guards",
        "outputs",
        "radars_123_bit_exact_primary_comparison",
        "source_semantics",
        "target_or_label_fields_read",
        "target_or_label_fields_present",
        "evaluation_performed",
        "content_sha256",
    }
)
_UNIT_OUTPUT_KEYS = frozenset(
    {
        "proposer_prediction",
        "raw_source_prediction",
        "sealed_prediction",
        "proposer_log",
        "source_log",
    }
)


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _forbid_fields(names: Sequence[str], *, label: str) -> None:
    lowered = {str(name).lower() for name in names}
    forbidden = sorted(lowered & {name.lower() for name in FORBIDDEN_LABEL_FIELDS})
    if forbidden:
        raise RadarMaskCampaignError(f"{label} contains target/label fields: {forbidden}")


def _read_label_free_npz(
    path: Path, *, label: str, required: set[str]
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            _forbid_fields(archive.files, label=label)
            missing = sorted(required - set(archive.files))
            if missing:
                raise RadarMaskCampaignError(f"{label} fields are missing: {missing}")
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except RadarMaskCampaignError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise RadarMaskCampaignError(f"invalid label-free NPZ {path}: {exc}") from exc
    if "target_fields_present" in arrays and bool(
        np.asarray(arrays["target_fields_present"]).item()
    ):
        raise RadarMaskCampaignError(f"{label} attests that target fields are present")
    return arrays


def _array_bit_equal(left: np.ndarray, right: np.ndarray) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(
        order="C"
    )


def _compare_npz_bit_exact(
    reference_path: Path,
    candidate_path: Path,
    *,
    label: str,
    allowed_candidate_only: set[str] = frozenset(),
) -> list[str]:
    reference = _read_label_free_npz(reference_path, label=f"primary {label}", required=set())
    candidate = _read_label_free_npz(candidate_path, label=f"masked {label}", required=set())
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference) - set(allowed_candidate_only))
    if missing or extra:
        raise RadarMaskCampaignError(
            f"radars_123 {label} schema differs from primary: missing={missing}, extra={extra}"
        )
    compared: list[str] = []
    for name in sorted(reference):
        if not _array_bit_equal(reference[name], candidate[name]):
            raise RadarMaskCampaignError(
                f"radars_123 {label} is not bit-exact to primary: {name}"
            )
        compared.append(name)
    return compared


def _validate_mask_outputs(
    *,
    proposer_path: Path,
    raw_path: Path,
    sealed_path: Path,
    unit: Mapping[str, Any],
    compare_full: bool,
) -> dict[str, Any]:
    mask_name = str(unit["radar_mask"])
    pattern = np.asarray(unit["radar_mask_pattern"], dtype=bool)
    proposer = _read_label_free_npz(
        proposer_path, label="masked proposer prediction", required=CORE_PROPOSER_FIELDS
    )
    if "radar_mask_name" not in proposer or str(
        np.asarray(proposer["radar_mask_name"]).item()
    ) != mask_name:
        raise RadarMaskCampaignError("proposer output radar-mask name is absent or wrong")
    if "radar_mask_pattern" not in proposer or not _array_bit_equal(
        np.asarray(proposer["radar_mask_pattern"]), pattern
    ):
        raise RadarMaskCampaignError("proposer output radar-mask pattern is absent or wrong")
    index = np.asarray(proposer["cache_index"])
    if index.dtype != np.int64 or index.ndim != 1 or len(index) == 0:
        raise RadarMaskCampaignError("masked proposer cache_index is invalid")
    if len(np.unique(index)) != len(index):
        raise RadarMaskCampaignError("masked proposer cache_index contains duplicates")

    raw = _read_label_free_npz(raw_path, label="masked raw source", required=RAW_REQUIRED)
    if not _array_bit_equal(raw["cache_index"], index):
        raise RadarMaskCampaignError("raw source/proposer cache_index differs")
    if "outer_fold" not in raw or int(np.asarray(raw["outer_fold"]).item()) != int(
        unit["outer_fold"]
    ):
        raise RadarMaskCampaignError("raw source outer-fold identity differs")
    if "seed" not in raw or int(np.asarray(raw["seed"]).item()) != int(unit["seed"]):
        raise RadarMaskCampaignError("raw source seed identity differs")
    if "source_is_no_action_placeholder" not in raw or not bool(
        np.asarray(raw["source_is_no_action_placeholder"]).item()
    ):
        raise RadarMaskCampaignError("raw source is not the frozen no-action adapter ABI")

    sealed = _read_label_free_npz(
        sealed_path, label="masked sealed prediction", required=SEALED_REQUIRED
    )
    if not _array_bit_equal(sealed["cache_index"], index):
        raise RadarMaskCampaignError("sealed/proposer cache_index differs")
    if bool(np.asarray(sealed["target_joined"]).item()):
        raise RadarMaskCampaignError("masked prediction reports a target join")
    if int(np.asarray(sealed["outer_fold"]).item()) != int(unit["outer_fold"]) or int(
        np.asarray(sealed["seed"]).item()
    ) != int(unit["seed"]):
        raise RadarMaskCampaignError("sealed prediction unit identity differs")
    final = np.asarray(sealed["final_rr_bpm"])
    fallback = np.asarray(sealed["fallback_rr_bpm"])
    if final.dtype != np.float32 or not _array_bit_equal(final, fallback):
        raise RadarMaskCampaignError("no-action final prediction is not bit-exact fallback")
    if np.any(np.asarray(sealed["applied_pull"], dtype=np.float32) != 0):
        raise RadarMaskCampaignError("no-action sealed prediction has nonzero applied pull")

    primary = unit["primary"]
    primary_proposer = Path(str(primary["proposer_prediction"]["path"]))
    primary_arrays = _read_label_free_npz(
        primary_proposer, label="primary proposer prediction", required={"cache_index"}
    )
    if not _array_bit_equal(primary_arrays["cache_index"], index):
        raise RadarMaskCampaignError("radar mask changed primary row ownership/order")

    comparisons: dict[str, Any] = {
        "required": bool(compare_full),
        "performed": False,
        "proposer_fields": [],
        "source_fields": [],
        "sealed_fields": [],
    }
    if compare_full:
        comparisons["proposer_fields"] = _compare_npz_bit_exact(
            primary_proposer,
            proposer_path,
            label="proposer prediction",
            allowed_candidate_only={"radar_mask_name", "radar_mask_pattern"},
        )
        comparisons["source_fields"] = _compare_npz_bit_exact(
            Path(str(primary["raw_source_prediction"]["path"])),
            raw_path,
            label="raw source prediction",
        )
        comparisons["sealed_fields"] = _compare_npz_bit_exact(
            Path(str(primary["sealed_prediction"]["path"])),
            sealed_path,
            label="sealed prediction",
        )
        comparisons["performed"] = True
    return comparisons


def _extract_primary_unit(
    *,
    plan_unit: Mapping[str, Any],
    seal_unit: Mapping[str, Any],
    plan_path: Path,
    primary_output_root: Path,
    pretest_sha256: str,
) -> dict[str, Any]:
    fold = int(plan_unit.get("outer_fold", -1))
    seed = int(plan_unit.get("seed", -1))
    derived_binding = _verified_binding(
        seal_unit.get("derived_lock"),
        relative_to=primary_output_root,
        label=f"primary derived lock {fold}/{seed}",
    )
    raw_stages = plan_unit.get("stages")
    if not isinstance(raw_stages, list):
        raise RadarMaskCampaignError(f"primary stage plan is missing: {fold}/{seed}")
    observed_stage_names = [
        stage.get("name") if isinstance(stage, Mapping) else None for stage in raw_stages
    ]
    if observed_stage_names not in (
        list(LOCKED_OOF.FAST_NO_ACTION_STAGES),
        list(LOCKED_OOF.STAGES),
    ):
        raise RadarMaskCampaignError(f"primary stage plan order is invalid: {fold}/{seed}")
    stage_contracts: list[dict[str, Any]] = []
    for stage in raw_stages:
        assert isinstance(stage, Mapping)
        argv = stage.get("argv")
        outputs = stage.get("outputs")
        if (
            not isinstance(argv, list)
            or any(not isinstance(token, str) or not token for token in argv)
            or not isinstance(outputs, list)
            or not outputs
        ):
            raise RadarMaskCampaignError(f"primary stage contract is invalid: {fold}/{seed}")
        stage_contracts.append(
            {
                "name": str(stage["name"]),
                "argv": list(argv),
                "outputs": [
                    _resolve(output, relative_to=plan_path.parent) for output in outputs
                ],
            }
        )
    unit_root = Path(derived_binding["path"]).parent
    try:
        derived = LOCKED_OOF._verify_derived_lock(  # noqa: SLF001
            Path(derived_binding["path"]),
            expected_pretest_sha=pretest_sha256,
            stages=stage_contracts,
            unit_root=unit_root,
        )
    except Exception as exc:
        raise RadarMaskCampaignError(
            f"invalid primary derived lock {fold}/{seed}: {exc}"
        ) from exc
    if (
        int(derived.get("outer_fold", -1)) != fold
        or int(derived.get("seed", -1)) != seed
        or derived.get("target_artifact_opened") is not False
        or derived.get("frozen_policy_status") != "fail_closed_no_action"
    ):
        raise RadarMaskCampaignError(f"primary derived-lock identity/policy mismatch: {fold}/{seed}")
    artifacts = derived.get("derived_artifacts")
    if not isinstance(artifacts, Mapping):
        raise RadarMaskCampaignError(f"primary derived artifacts are missing: {fold}/{seed}")
    proposer = _verified_binding(
        artifacts.get("test_proposer_prediction"),
        relative_to=Path(derived_binding["path"]).parent,
        label=f"primary proposer prediction {fold}/{seed}",
    )
    raw = _verified_binding(
        artifacts.get("raw_hcs_prediction"),
        relative_to=Path(derived_binding["path"]).parent,
        label=f"primary raw source prediction {fold}/{seed}",
    )
    sealed = _verified_binding(
        seal_unit.get("prediction"),
        relative_to=primary_output_root,
        label=f"primary sealed prediction {fold}/{seed}",
    )
    derived_sealed = _verified_binding(
        derived.get("sealed_prediction"),
        relative_to=Path(derived_binding["path"]).parent,
        label=f"derived sealed prediction {fold}/{seed}",
    )
    if not _same_binding(sealed, derived_sealed):
        raise RadarMaskCampaignError(f"primary seal/derived prediction mismatch: {fold}/{seed}")

    matches = [
        stage
        for stage in raw_stages
        if isinstance(stage, Mapping) and stage.get("name") == "test_proposer_predict"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("argv"), list):
        raise RadarMaskCampaignError(f"primary proposer command is missing: {fold}/{seed}")
    argv = [str(token) for token in matches[0]["argv"]]
    if len(argv) < 3 or argv[2] != "proposer-predict":
        raise RadarMaskCampaignError(
            f"primary proposer stage does not use proposer-predict: {fold}/{seed}"
        )
    primary_command_python = bind_file(_resolve(argv[0], relative_to=plan_path.parent))
    primary_command_helper = bind_file(_resolve(argv[1], relative_to=plan_path.parent))
    checkpoint_path = _resolve(
        _argv_value(argv, "--checkpoint", label="primary proposer command"),
        relative_to=plan_path.parent,
    )
    run_config_path = _resolve(
        _argv_value(argv, "--run-config", label="primary proposer command"),
        relative_to=plan_path.parent,
    )
    manifest_path = _resolve(
        _argv_value(argv, "--test-manifest", label="primary proposer command"),
        relative_to=plan_path.parent,
    )
    cache_dir = _resolve(
        _argv_value(argv, "--cache-dir", label="primary proposer command"),
        relative_to=plan_path.parent,
    )
    primary_output_path = _resolve(
        _argv_value(argv, "--output", label="primary proposer command"),
        relative_to=plan_path.parent,
    )
    primary_batch_size = int(
        _argv_value(argv, "--batch-size", label="primary proposer command")
    )
    if _argv_value(argv, "--device", label="primary proposer command") != "cpu" or "--amp" in argv:
        raise RadarMaskCampaignError(f"primary proposer was not the CPU deterministic path: {fold}/{seed}")
    if primary_batch_size <= 0:
        raise RadarMaskCampaignError(f"primary proposer batch size is invalid: {fold}/{seed}")
    if "--radar-mask" in argv and _argv_value(
        argv, "--radar-mask", label="primary proposer command"
    ) != "radars_123":
        raise RadarMaskCampaignError(f"primary proposer is not the full-mask reference: {fold}/{seed}")
    checkpoint = bind_file(checkpoint_path)
    run_config = bind_file(run_config_path)
    manifest = bind_file(manifest_path)
    if primary_output_path != Path(proposer["path"]):
        raise RadarMaskCampaignError(f"primary command/derived proposer output mismatch: {fold}/{seed}")
    recorded_manifest = _verified_binding(
        derived.get("test_manifest"),
        relative_to=Path(derived_binding["path"]).parent,
        label=f"primary test manifest {fold}/{seed}",
    )
    if not _same_binding(manifest, recorded_manifest):
        raise RadarMaskCampaignError(f"primary command/derived manifest mismatch: {fold}/{seed}")
    checkpoint_artifact = artifacts.get("test_proposer_checkpoint")
    if checkpoint_artifact is not None:
        recorded_checkpoint = _verified_binding(
            checkpoint_artifact,
            relative_to=Path(derived_binding["path"]).parent,
            label=f"primary bound checkpoint {fold}/{seed}",
        )
        if not _same_binding(checkpoint, recorded_checkpoint):
            raise RadarMaskCampaignError(
                f"primary command/derived checkpoint mismatch: {fold}/{seed}"
            )
    source_validation: dict[str, Any] = {}
    source_raw = plan_unit.get("source_validation_proposer")
    if isinstance(source_raw, Mapping):
        for name in ("checkpoint", "run_config", "split_manifest", "strict_stack"):
            if name in source_raw:
                source_validation[name] = _verified_binding(
                    source_raw[name],
                    relative_to=plan_path.parent,
                    label=f"primary source validation proposer {fold}/{seed}/{name}",
                )
    _read_label_free_npz(Path(proposer["path"]), label="primary proposer", required=CORE_PROPOSER_FIELDS)
    raw_arrays = _read_label_free_npz(
        Path(raw["path"]), label="primary raw source", required=RAW_REQUIRED
    )
    if "source_is_no_action_placeholder" not in raw_arrays or not bool(
        np.asarray(raw_arrays["source_is_no_action_placeholder"]).item()
    ):
        raise RadarMaskCampaignError(
            f"primary source is not semantically comparable no-action output: {fold}/{seed}"
        )
    _read_label_free_npz(Path(sealed["path"]), label="primary sealed", required=SEALED_REQUIRED)
    return {
        "outer_fold": fold,
        "seed": seed,
        "checkpoint": checkpoint,
        "run_config": run_config,
        "test_manifest": manifest,
        "rf_cache_dir": str(cache_dir),
        "primary_batch_size": primary_batch_size,
        "primary_command_sources": {
            "python_executable": primary_command_python,
            "safe_test_input_helper": primary_command_helper,
        },
        "primary": {
            "derived_lock": derived_binding,
            "proposer_prediction": proposer,
            "raw_source_prediction": raw,
            "sealed_prediction": sealed,
        },
        "source_validation_proposer": source_validation,
    }


def build_plan(
    *,
    primary_plan_path: Path,
    primary_output_root: Path,
    output_root: Path,
    python_executable: Path,
    safe_helper: Path,
    locked_oof_source: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Derive the exact 126-unit plan from an existing primary OOF seal."""

    if batch_size <= 0:
        raise RadarMaskCampaignError("batch_size must be positive")
    if tuple(MASKS) != (
        "radars_123",
        "radars_12",
        "radars_13",
        "radars_23",
        "radar_1",
        "radar_2",
        "radar_3",
    ) or len(set(MASKS.values())) != 7 or not all(any(mask) for mask in MASKS.values()):
        raise RadarMaskCampaignError("safe helper no longer exposes the locked seven-mask topology")

    plan_path = primary_plan_path.expanduser().resolve()
    primary_root = primary_output_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    try:
        primary_plan, loaded_plan_path = LOCKED_OOF.load_plan(plan_path)
        primary_seal = LOCKED_OOF._verify_predictions_seal(primary_root)  # noqa: SLF001
    except Exception as exc:
        raise RadarMaskCampaignError(f"primary locked OOF contract is invalid: {exc}") from exc
    if loaded_plan_path != plan_path:
        raise RadarMaskCampaignError("primary plan resolved unexpectedly")
    if primary_seal.get("target_artifact_opened_before_seal") is not False:
        raise RadarMaskCampaignError("primary predictions were not sealed before target access")

    primary_pretest_path = primary_root / "pretest_lock.json"
    primary_pretest = _json(primary_pretest_path, "primary pretest lock")
    primary_plan_binding = bind_file(plan_path)
    recorded_primary_plan = _verified_binding(
        primary_pretest.get("plan"),
        relative_to=primary_pretest_path.parent,
        label="primary pretest plan",
    )
    if not _same_binding(primary_plan_binding, recorded_primary_plan):
        raise RadarMaskCampaignError("primary pretest lock is bound to another plan")
    primary_pretest_binding = bind_file(primary_pretest_path)
    if primary_seal.get("pretest_lock_sha256") != primary_pretest_binding["sha256"]:
        raise RadarMaskCampaignError("primary prediction seal/pretest-lock hash mismatch")

    common = primary_plan.get("common")
    if not isinstance(common, Mapping):
        raise RadarMaskCampaignError("primary plan common bindings are missing")
    policy_binding = _verified_binding(
        common.get("policy"), relative_to=plan_path.parent, label="primary common policy"
    )
    policy_document = _json(Path(policy_binding["path"]), "primary common policy")
    policy = policy_document.get("policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("selection_status") != "fail_closed_no_action"
        or float(policy.get("correction_pull", -1.0)) != 0.0
    ):
        raise RadarMaskCampaignError(
            "radar-mask campaign requires the frozen fail_closed_no_action policy"
        )

    executable_binding = bind_python_launcher(python_executable)
    helper_binding = bind_file(safe_helper)
    oof_source_binding = bind_file(locked_oof_source)
    imported_helper_binding = bind_file(Path(SAFE_INPUTS.__file__))
    imported_oof_binding = bind_file(Path(LOCKED_OOF.__file__))
    if helper_binding["sha256"] != imported_helper_binding["sha256"]:
        raise RadarMaskCampaignError(
            "executed safe helper differs from the helper contract imported by the orchestrator"
        )
    if oof_source_binding["sha256"] != imported_oof_binding["sha256"]:
        raise RadarMaskCampaignError(
            "bound locked-OOF source differs from the imported sealing contract"
        )
    primary_sources = primary_plan.get("effective_sources")
    if not isinstance(primary_sources, Mapping):
        raise RadarMaskCampaignError("primary plan effective-source bindings are missing")
    verified_primary_sources = {
        str(name): _verified_binding(
            raw, relative_to=plan_path.parent, label=f"primary effective source {name}"
        )
        for name, raw in sorted(primary_sources.items())
    }
    for current, primary_name, label in (
        (helper_binding, "safe_test_input_helper", "safe helper"),
        (oof_source_binding, "plan_builder", "locked OOF source"),
        (executable_binding, "python_executable", "Python executable"),
    ):
        if primary_name not in verified_primary_sources or current["sha256"] != verified_primary_sources[
            primary_name
        ]["sha256"]:
            raise RadarMaskCampaignError(f"{label} drifted from the primary sealed runtime")

    primary_units_raw = primary_plan["units"]
    primary_unit_map = {
        (int(unit["outer_fold"]), int(unit["seed"])): unit for unit in primary_units_raw
    }
    seal_units_raw = primary_seal.get("units")
    if not isinstance(seal_units_raw, list) or len(seal_units_raw) != EXPECTED_PRIMARY_UNITS:
        raise RadarMaskCampaignError("primary seal does not contain 18 units")
    seal_unit_map = {
        (int(unit["outer_fold"]), int(unit["seed"])): unit for unit in seal_units_raw
    }
    seeds = [int(value) for value in primary_plan["seeds"]]
    if seeds != list(LOCKED_SEEDS):
        raise RadarMaskCampaignError(
            f"primary seed order differs from the locked campaign: {seeds}"
        )
    expected_keys = {(fold, seed) for seed in seeds for fold in FOLDS}
    if set(primary_unit_map) != expected_keys or set(seal_unit_map) != expected_keys:
        raise RadarMaskCampaignError("primary plan/seal unit topology differs")

    primary_contexts: dict[tuple[int, int], dict[str, Any]] = {}
    cache_manifests: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        for fold in FOLDS:
            context = _extract_primary_unit(
                plan_unit=primary_unit_map[(fold, seed)],
                seal_unit=seal_unit_map[(fold, seed)],
                plan_path=plan_path,
                primary_output_root=primary_root,
                pretest_sha256=primary_pretest_binding["sha256"],
            )
            if int(context["primary_batch_size"]) != int(batch_size):
                raise RadarMaskCampaignError(
                    f"requested batch size differs from primary full-mask inference: "
                    f"{fold}/{seed}/{context['primary_batch_size']} != {batch_size}"
                )
            if context["primary_command_sources"]["python_executable"][
                "sha256"
            ] != executable_binding["sha256"]:
                raise RadarMaskCampaignError(
                    f"primary proposer command used another Python runtime: {fold}/{seed}"
                )
            if context["primary_command_sources"]["safe_test_input_helper"][
                "sha256"
            ] != helper_binding["sha256"]:
                raise RadarMaskCampaignError(
                    f"primary proposer command used another safe helper: {fold}/{seed}"
                )
            manifest_binding = bind_file(Path(context["rf_cache_dir"]) / "manifest.json")
            cache_manifests.setdefault(manifest_binding["path"], manifest_binding)
            primary_contexts[(fold, seed)] = context
    if len(cache_manifests) != 1:
        raise RadarMaskCampaignError("primary units do not share one sealed RF cache manifest")
    recorded_rf = _verified_binding(
        primary_plan.get("rf_cache_manifest"),
        relative_to=plan_path.parent,
        label="primary RF cache manifest",
    )
    observed_rf = next(iter(cache_manifests.values()))
    if not _same_binding(recorded_rf, observed_rf):
        raise RadarMaskCampaignError("primary proposer commands/RF cache binding mismatch")

    units: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in FOLDS:
            context = primary_contexts[(fold, seed)]
            for mask_name, pattern in MASKS.items():
                unit_stub = {
                    "outer_fold": fold,
                    "seed": seed,
                    "radar_mask": mask_name,
                }
                final_root = _unit_root(output, unit_stub)
                proposer_output = final_root / "proposer_prediction.npz"
                raw_output = final_root / "raw_source_prediction.npz"
                sealed_output = final_root / "sealed_label_free_predictions.npz"
                proposer_argv = [
                    executable_binding["path"],
                    helper_binding["path"],
                    "proposer-predict",
                    "--cache-dir",
                    context["rf_cache_dir"],
                    "--checkpoint",
                    context["checkpoint"]["path"],
                    "--run-config",
                    context["run_config"]["path"],
                    "--test-manifest",
                    context["test_manifest"]["path"],
                    "--output",
                    str(proposer_output),
                    "--device",
                    "cpu",
                    "--batch-size",
                    str(batch_size),
                    "--radar-mask",
                    mask_name,
                ]
                adapter_argv = [
                    executable_binding["path"],
                    helper_binding["path"],
                    "no-action-adapter",
                    "--proposer",
                    str(proposer_output),
                    "--outer-fold",
                    str(fold),
                    "--seed",
                    str(seed),
                    "--output",
                    str(raw_output),
                ]
                units.append(
                    {
                        "unit_id": _mask_unit_id(fold, seed, mask_name),
                        "outer_fold": fold,
                        "seed": seed,
                        "radar_mask": mask_name,
                        "radar_mask_pattern": list(pattern),
                        "inputs": {
                            "checkpoint": context["checkpoint"],
                            "run_config": context["run_config"],
                            "test_manifest": context["test_manifest"],
                            "rf_cache_manifest": observed_rf,
                            "primary_command_sources": context[
                                "primary_command_sources"
                            ],
                            "primary_batch_size": int(context["primary_batch_size"]),
                            "source_validation_proposer": context[
                                "source_validation_proposer"
                            ],
                        },
                        "primary": context["primary"],
                        "commands": [
                            {"stage": "proposer_predict", "argv": proposer_argv},
                            {"stage": "no_action_source_adapter", "argv": adapter_argv},
                        ],
                        "outputs": {
                            "proposer_prediction": str(proposer_output),
                            "raw_source_prediction": str(raw_output),
                            "sealed_prediction": str(sealed_output),
                            "receipt": str(final_root / "receipt.json"),
                        },
                    }
                )
    if len(units) != EXPECTED_MASK_UNITS:
        raise RadarMaskCampaignError("internal error: campaign plan is not 18 x 7")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_seven_radar_mask_label_free_plan",
        "folds": list(FOLDS),
        "seeds": seeds,
        "radar_masks": [
            {"name": name, "pattern": list(pattern)} for name, pattern in MASKS.items()
        ],
        "primary_unit_count": EXPECTED_PRIMARY_UNITS,
        "unit_count": EXPECTED_MASK_UNITS,
        "target_or_label_artifact_bound": False,
        "evaluation_permitted_before_complete_seal": False,
        "mask_selection_contract": {
            "mask_order_fixed_before_inference": list(MASK_NAMES),
            "best_mask_selection_allowed": False,
            "target_or_metric_dependent_mask_selection_allowed": False,
            "all_masks_are_required_conditions": True,
            "radars_123_primary_parity": {
                "required": True,
                "comparison": "dtype_shape_and_array_bytes",
                "artifacts": [
                    "proposer_prediction",
                    "raw_source_prediction",
                    "sealed_prediction",
                ],
            },
        },
        "frozen_policy": policy_binding,
        "primary": {
            "plan": primary_plan_binding,
            "pretest_lock": primary_pretest_binding,
            "predictions_seal": bind_file(primary_root / "predictions_seal.json"),
        },
        "effective_sources": {
            "radar_mask_orchestrator": bind_file(Path(__file__)),
            "safe_test_input_helper": helper_binding,
            "locked_oof_contract": oof_source_binding,
            "python_executable": executable_binding,
            "primary_effective_sources": verified_primary_sources,
        },
        "rf_cache_manifest": observed_rf,
        "execution": {
            "device": "cpu",
            "amp": False,
            "batch_size": int(batch_size),
            "shell": False,
            "publication": "atomic_unit_directory_rename",
        },
        "units": units,
    }


def _validate_command_contract(unit: Mapping[str, Any], output_root: Path) -> None:
    commands = unit.get("commands")
    if not isinstance(commands, list) or [item.get("stage") for item in commands] != [
        "proposer_predict",
        "no_action_source_adapter",
    ]:
        raise RadarMaskCampaignError(f"unit command order changed: {unit.get('unit_id')}")
    proposer = commands[0].get("argv")
    adapter = commands[1].get("argv")
    if not isinstance(proposer, list) or not isinstance(adapter, list):
        raise RadarMaskCampaignError("unit commands must be argv arrays")
    if len(proposer) < 3 or len(adapter) < 3 or proposer[:2] != adapter[:2]:
        raise RadarMaskCampaignError("unit stages do not share one bound Python/helper pair")
    sources = unit["inputs"]
    expected_root = _unit_root(output_root, unit)
    expected_proposer = Path(str(unit["outputs"]["proposer_prediction"])).resolve()
    expected_raw = Path(str(unit["outputs"]["raw_source_prediction"])).resolve()
    if expected_proposer.parent != expected_root or expected_raw.parent != expected_root:
        raise RadarMaskCampaignError("unit outputs escape deterministic unit root")
    if "--amp" in proposer or _argv_value(proposer, "--device", label="proposer") != "cpu":
        raise RadarMaskCampaignError("radar-mask proposer command is not CPU-only")
    if len(proposer) < 3 or proposer[2] != "proposer-predict":
        raise RadarMaskCampaignError("radar-mask command does not use proposer-predict")
    if _argv_value(proposer, "--radar-mask", label="proposer") != unit["radar_mask"]:
        raise RadarMaskCampaignError("radar-mask command uses the wrong mask")
    expected_values = {
        "--checkpoint": sources["checkpoint"]["path"],
        "--run-config": sources["run_config"]["path"],
        "--test-manifest": sources["test_manifest"]["path"],
        "--output": str(expected_proposer),
    }
    for option, expected in expected_values.items():
        if Path(_argv_value(proposer, option, label="proposer")).resolve() != Path(
            str(expected)
        ).resolve():
            raise RadarMaskCampaignError(f"proposer command {option} binding differs")
    if len(adapter) < 3 or adapter[2] != "no-action-adapter":
        raise RadarMaskCampaignError("source command does not use no-action-adapter")
    if Path(_argv_value(adapter, "--proposer", label="adapter")).resolve() != expected_proposer:
        raise RadarMaskCampaignError("adapter does not consume its unit proposer")
    if Path(_argv_value(adapter, "--output", label="adapter")).resolve() != expected_raw:
        raise RadarMaskCampaignError("adapter output differs from unit contract")
    if int(_argv_value(adapter, "--outer-fold", label="adapter")) != int(
        unit["outer_fold"]
    ) or int(_argv_value(adapter, "--seed", label="adapter")) != int(unit["seed"]):
        raise RadarMaskCampaignError("adapter unit identity differs")


def _runtime_argv(argv: Sequence[str], *, final_root: Path, staging_root: Path) -> list[str]:
    final = str(final_root.resolve())
    staging = str(staging_root.resolve())
    result = []
    for token in argv:
        value = str(token)
        if value == final or value.startswith(final + os.sep):
            value = staging + value[len(final) :]
        result.append(value)
    return result


def _run_command(argv: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        completed.stdout + ("\n[stderr]\n" if completed.stderr else "") + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RadarMaskCampaignError(
            f"unit command failed with status {completed.returncode}: {argv[2]} ({log_path})"
        )


def _binding_from_staging(actual: Path, final: Path) -> dict[str, Any]:
    return bind_file(actual, recorded_path=final)


def _receipt_content(document: Mapping[str, Any]) -> dict[str, Any]:
    content = dict(document)
    recorded = str(content.pop("content_sha256", ""))
    if len(recorded) != 64 or canonical_json_sha256(content) != recorded:
        raise RadarMaskCampaignError("unit receipt content hash mismatch")
    return content


def _execute_unit(
    *,
    unit: Mapping[str, Any],
    output_root: Path,
    plan_binding: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    final_root = _unit_root(output_root, unit)
    if final_root.exists():
        raise RadarMaskCampaignError(f"unit root exists without a validated receipt: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{unit['radar_mask']}.staging.", dir=final_root.parent)
    ).resolve()
    proposer_path = staging_root / "proposer_prediction.npz"
    raw_path = staging_root / "raw_source_prediction.npz"
    sealed_path = staging_root / "sealed_label_free_predictions.npz"
    try:
        declared_commands = unit["commands"]
        proposer_argv = _runtime_argv(
            declared_commands[0]["argv"], final_root=final_root, staging_root=staging_root
        )
        adapter_argv = _runtime_argv(
            declared_commands[1]["argv"], final_root=final_root, staging_root=staging_root
        )
        _run_command(
            proposer_argv,
            cwd=output_root,
            log_path=staging_root / "logs/proposer_predict.log",
        )
        if not proposer_path.is_file():
            raise RadarMaskCampaignError("proposer command did not create its declared output")
        _read_label_free_npz(
            proposer_path, label="masked proposer prediction", required=CORE_PROPOSER_FIELDS
        )
        _run_command(
            adapter_argv,
            cwd=output_root,
            log_path=staging_root / "logs/no_action_source_adapter.log",
        )
        if not raw_path.is_file():
            raise RadarMaskCampaignError("adapter command did not create its declared output")
        raw_arrays = _read_label_free_npz(
            raw_path, label="masked raw source", required=RAW_REQUIRED
        )
        try:
            sealed_arrays = LOCKED_OOF._sealed_prediction_arrays(  # noqa: SLF001
                raw_arrays,
                fold=int(unit["outer_fold"]),
                seed=int(unit["seed"]),
                policy=policy,
            )
        except Exception as exc:
            raise RadarMaskCampaignError(f"locked OOF sealing contract rejected unit: {exc}") from exc
        _atomic_npz(sealed_path, sealed_arrays)
        comparisons = _validate_mask_outputs(
            proposer_path=proposer_path,
            raw_path=raw_path,
            sealed_path=sealed_path,
            unit=unit,
            compare_full=str(unit["radar_mask"]) == "radars_123",
        )
        final_paths = {
            "proposer_prediction": final_root / "proposer_prediction.npz",
            "raw_source_prediction": final_root / "raw_source_prediction.npz",
            "sealed_prediction": final_root / "sealed_label_free_predictions.npz",
        }
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "classification": "locked_hcs_radar_mask_label_free_unit_receipt",
            "unit_id": unit["unit_id"],
            "outer_fold": int(unit["outer_fold"]),
            "seed": int(unit["seed"]),
            "radar_mask": unit["radar_mask"],
            "radar_mask_pattern": list(unit["radar_mask_pattern"]),
            "plan": dict(plan_binding),
            "inputs": unit["inputs"],
            "primary": unit["primary"],
            "commands": unit["commands"],
            "runtime_guards": {
                "device": "cpu",
                "cuda_visible_devices": "",
                "amp": False,
                "shell": False,
            },
            "outputs": {
                "proposer_prediction": _binding_from_staging(
                    proposer_path, final_paths["proposer_prediction"]
                ),
                "raw_source_prediction": _binding_from_staging(
                    raw_path, final_paths["raw_source_prediction"]
                ),
                "sealed_prediction": _binding_from_staging(
                    sealed_path, final_paths["sealed_prediction"]
                ),
                "proposer_log": _binding_from_staging(
                    staging_root / "logs/proposer_predict.log",
                    final_root / "logs/proposer_predict.log",
                ),
                "source_log": _binding_from_staging(
                    staging_root / "logs/no_action_source_adapter.log",
                    final_root / "logs/no_action_source_adapter.log",
                ),
            },
            "radars_123_bit_exact_primary_comparison": comparisons,
            "source_semantics": "frozen_no_action_placeholder",
            "target_or_label_fields_read": False,
            "target_or_label_fields_present": False,
            "evaluation_performed": False,
        }
        receipt["content_sha256"] = canonical_json_sha256(receipt)
        _atomic_json(staging_root / "receipt.json", receipt, immutable=True)
        for path in staging_root.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        os.replace(staging_root, final_root)
        return _validate_receipt(unit=unit, output_root=output_root, plan_binding=plan_binding)
    except BaseException:
        # This directory was created by this invocation and has never been
        # published.  Removing it prevents a rejected target-bearing or
        # otherwise invalid archive from lingering inside the campaign root.
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def _validate_receipt(
    *, unit: Mapping[str, Any], output_root: Path, plan_binding: Mapping[str, Any]
) -> dict[str, Any]:
    root = _unit_root(output_root, unit)
    receipt_path = root / "receipt.json"
    if not receipt_path.is_file():
        if root.exists():
            raise RadarMaskCampaignError(f"unit directory has no receipt: {root}")
        raise FileNotFoundError(receipt_path)
    if receipt_path.is_symlink() or not _under(receipt_path, root):
        raise RadarMaskCampaignError(f"unit receipt escapes deterministic unit root: {root}")
    receipt = _json(receipt_path, "radar-mask unit receipt")
    _receipt_content(receipt)
    expected_identity = (
        set(receipt) == _UNIT_RECEIPT_KEYS
        and receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("classification")
        == "locked_hcs_radar_mask_label_free_unit_receipt"
        and receipt.get("unit_id") == unit["unit_id"]
        and int(receipt.get("outer_fold", -1)) == int(unit["outer_fold"])
        and int(receipt.get("seed", -1)) == int(unit["seed"])
        and receipt.get("radar_mask") == unit["radar_mask"]
        and receipt.get("radar_mask_pattern") == unit["radar_mask_pattern"]
        and receipt.get("target_or_label_fields_read") is False
        and receipt.get("target_or_label_fields_present") is False
        and receipt.get("evaluation_performed") is False
        and receipt.get("source_semantics") == "frozen_no_action_placeholder"
        and receipt.get("runtime_guards")
        == {
            "device": "cpu",
            "cuda_visible_devices": "",
            "amp": False,
            "shell": False,
        }
    )
    if not expected_identity:
        raise RadarMaskCampaignError(f"unit receipt identity/label-free attestation differs: {root}")
    recorded_plan = receipt.get("plan")
    if not isinstance(recorded_plan, Mapping) or not _same_binding(recorded_plan, plan_binding):
        raise RadarMaskCampaignError(f"unit receipt is bound to another campaign plan: {root}")
    if receipt.get("inputs") != unit["inputs"] or receipt.get("primary") != unit["primary"]:
        raise RadarMaskCampaignError(f"unit receipt input provenance differs: {root}")
    if receipt.get("commands") != unit["commands"]:
        raise RadarMaskCampaignError(f"unit receipt command provenance differs: {root}")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != _UNIT_OUTPUT_KEYS:
        raise RadarMaskCampaignError(f"unit receipt output key topology differs: {root}")
    verified = {
        name: _verified_binding(raw, relative_to=root, label=f"unit output {name}")
        for name, raw in outputs.items()
    }
    expected_paths = {
        "proposer_prediction": root / "proposer_prediction.npz",
        "raw_source_prediction": root / "raw_source_prediction.npz",
        "sealed_prediction": root / "sealed_label_free_predictions.npz",
        "proposer_log": root / "logs/proposer_predict.log",
        "source_log": root / "logs/no_action_source_adapter.log",
    }
    for name, expected in expected_paths.items():
        observed = Path(verified[name]["path"])
        if not _under(observed, root) or observed != expected.resolve():
            raise RadarMaskCampaignError(f"unit output path differs: {root}/{name}")
    comparisons = _validate_mask_outputs(
        proposer_path=expected_paths["proposer_prediction"],
        raw_path=expected_paths["raw_source_prediction"],
        sealed_path=expected_paths["sealed_prediction"],
        unit=unit,
        compare_full=str(unit["radar_mask"]) == "radars_123",
    )
    if receipt.get("radars_123_bit_exact_primary_comparison") != comparisons:
        raise RadarMaskCampaignError(f"unit bit-exact comparison receipt differs: {root}")
    return receipt


def _progress_document(
    *, plan_binding: Mapping[str, Any], completed: Sequence[Mapping[str, Any]], sealed: bool
) -> dict[str, Any]:
    records = [
        {
            "unit_id": receipt["unit_id"],
            "outer_fold": int(receipt["outer_fold"]),
            "seed": int(receipt["seed"]),
            "radar_mask": receipt["radar_mask"],
            "receipt_content_sha256": receipt["content_sha256"],
        }
        for receipt in completed
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_radar_mask_label_free_progress",
        "plan": dict(plan_binding),
        "completed_units": len(records),
        "expected_units": EXPECTED_MASK_UNITS,
        "complete_seal_present": bool(sealed),
        "evaluation_authorized": bool(sealed),
        "target_or_label_artifact_opened": False,
        "units": records,
    }


def _complete_seal(
    *,
    output_root: Path,
    plan_binding: Mapping[str, Any],
    preexecution_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(receipts) != EXPECTED_MASK_UNITS:
        raise RadarMaskCampaignError("complete seal requires exactly 126 validated receipts")
    records = []
    for unit, receipt in zip(plan["units"], receipts, strict=True):
        root = _unit_root(output_root, unit)
        records.append(
            {
                "unit_id": unit["unit_id"],
                "outer_fold": int(unit["outer_fold"]),
                "seed": int(unit["seed"]),
                "radar_mask": unit["radar_mask"],
                "radar_mask_pattern": list(unit["radar_mask_pattern"]),
                "receipt": bind_file(root / "receipt.json"),
                "proposer_prediction": bind_file(root / "proposer_prediction.npz"),
                "raw_source_prediction": bind_file(root / "raw_source_prediction.npz"),
                "sealed_prediction": bind_file(root / "sealed_label_free_predictions.npz"),
                "receipt_content_sha256": receipt["content_sha256"],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_all_seven_radar_mask_predictions_sealed",
        "plan": dict(plan_binding),
        "preexecution_lock": dict(preexecution_binding),
        "primary_predictions_seal": plan["primary"]["predictions_seal"],
        "folds": list(FOLDS),
        "seeds": list(plan["seeds"]),
        "radar_masks": list(plan["radar_masks"]),
        "unit_count": EXPECTED_MASK_UNITS,
        "complete_matrix": True,
        "target_or_label_artifact_opened_before_seal": False,
        "best_mask_selection_performed": False,
        "all_masks_retained_as_fixed_conditions": True,
        "evaluation_authorized": True,
        "effective_sources": plan["effective_sources"],
        "frozen_policy": plan["frozen_policy"],
        "rf_cache_manifest": plan["rf_cache_manifest"],
        "mask_selection_contract": plan["mask_selection_contract"],
        "units": records,
    }


def run_campaign(
    *,
    primary_plan_path: Path,
    primary_output_root: Path,
    output_root: Path,
    python_executable: Path = Path(sys.executable),
    safe_helper: Path = SCRIPT_DIR / "build_locked_hcs_test_inputs.py",
    locked_oof_source: Path = SCRIPT_DIR / "run_locked_hcs_oof.py",
    batch_size: int = 128,
    max_new_units: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create/validate the plan and run at most ``max_new_units`` pending units."""

    if max_new_units is not None and int(max_new_units) < 0:
        raise RadarMaskCampaignError("max_new_units cannot be negative")
    output = output_root.expanduser().resolve()
    control = output / "control"
    control.mkdir(parents=True, exist_ok=True)
    lock_path = control / "campaign.lock"
    with lock_path.open("a+b") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RadarMaskCampaignError("another radar-mask campaign process holds the lock") from exc
        plan = build_plan(
            primary_plan_path=primary_plan_path,
            primary_output_root=primary_output_root,
            output_root=output,
            python_executable=python_executable,
            safe_helper=safe_helper,
            locked_oof_source=locked_oof_source,
            batch_size=batch_size,
        )
        plan_path = control / "plan.json"
        _ensure_document(plan_path, plan)
        plan_binding = bind_file(plan_path)
        preexecution = {
            "schema_version": SCHEMA_VERSION,
            "classification": "locked_hcs_radar_mask_preexecution_input_seal",
            "plan": plan_binding,
            "primary": plan["primary"],
            "effective_sources": plan["effective_sources"],
            "rf_cache_manifest": plan["rf_cache_manifest"],
            "unit_count": EXPECTED_MASK_UNITS,
            "target_or_label_artifact_opened": False,
            "evaluation_authorized": False,
        }
        preexecution_path = control / "preexecution_lock.json"
        _ensure_document(preexecution_path, preexecution)
        preexecution_binding = bind_file(preexecution_path)

        # Validate the complete command/output topology before the first unit
        # is allowed to execute.  This prevents a late malformed unit from
        # being discovered only after earlier mask predictions already exist.
        for unit in plan["units"]:
            _validate_command_contract(unit, output)

        policy_document = _json(Path(plan["frozen_policy"]["path"]), "frozen policy")
        policy = policy_document.get("policy")
        if not isinstance(policy, Mapping):
            raise RadarMaskCampaignError("frozen policy payload is missing")
        limit = 0 if dry_run else (
            EXPECTED_MASK_UNITS if max_new_units is None else int(max_new_units)
        )
        # Audit every pre-existing unit before launching any new command.  A
        # corrupted late unit must therefore stop the resume before it can
        # extend the campaign elsewhere.
        existing: dict[str, dict[str, Any]] = {}
        for unit in plan["units"]:
            root = _unit_root(output, unit)
            receipt_path = root / "receipt.json"
            if receipt_path.is_file():
                existing[str(unit["unit_id"])] = _validate_receipt(
                    unit=unit, output_root=output, plan_binding=plan_binding
                )
            elif root.exists():
                raise RadarMaskCampaignError(
                    f"unit directory exists without a complete receipt: {root}"
                )

        seal_path = output / "complete_seal.json"
        if seal_path.exists() and len(existing) != EXPECTED_MASK_UNITS:
            raise RadarMaskCampaignError(
                "complete seal exists for an incomplete 126-unit matrix"
            )

        completed: list[dict[str, Any]] = []
        new_units = 0
        for unit in plan["units"]:
            unit_id = str(unit["unit_id"])
            if unit_id in existing:
                receipt = existing[unit_id]
                completed.append(receipt)
            elif new_units < limit:
                receipt = _execute_unit(
                    unit=unit,
                    output_root=output,
                    plan_binding=plan_binding,
                    policy=policy,
                )
                completed.append(receipt)
                new_units += 1
            _atomic_json(
                control / "progress.json",
                _progress_document(plan_binding=plan_binding, completed=completed, sealed=False),
            )

        if len(completed) == EXPECTED_MASK_UNITS:
            seal = _complete_seal(
                output_root=output,
                plan_binding=plan_binding,
                preexecution_binding=preexecution_binding,
                plan=plan,
                receipts=completed,
            )
            _ensure_document(seal_path, seal)
            sealed = True
            status = "label_free_radar_mask_predictions_sealed"
        else:
            sealed = False
            status = "dry_run" if dry_run else "label_free_radar_mask_incomplete"
        progress = _progress_document(
            plan_binding=plan_binding, completed=completed, sealed=sealed
        )
        _atomic_json(control / "progress.json", progress)
        result: dict[str, Any] = {
            "status": status,
            "completed_units": len(completed),
            "new_units": new_units,
            "expected_units": EXPECTED_MASK_UNITS,
            "target_or_label_artifact_opened": False,
            "evaluation_authorized": sealed,
            "plan": plan_binding,
            "preexecution_lock": preexecution_binding,
        }
        if sealed:
            result["complete_seal"] = bind_file(seal_path)
        _atomic_json(control / "status.json", result)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-plan",
        type=Path,
        default=DEFAULT_PRIMARY_ROOT / "locked_oof_plan.json",
    )
    parser.add_argument("--primary-output-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--safe-helper", type=Path, default=SCRIPT_DIR / "build_locked_hcs_test_inputs.py"
    )
    parser.add_argument(
        "--locked-oof-source", type=Path, default=SCRIPT_DIR / "run_locked_hcs_oof.py"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-units", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_campaign(
        primary_plan_path=args.primary_plan,
        primary_output_root=args.primary_output_root,
        output_root=args.output_root,
        python_executable=args.python_executable,
        safe_helper=args.safe_helper,
        locked_oof_source=args.locked_oof_source,
        batch_size=args.batch_size,
        max_new_units=args.max_new_units,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
