#!/usr/bin/env python3
"""Launch one V8R4 campaign shard in a capability-only bubblewrap runtime.

This is an *outer* runtime boundary.  The campaign, ``run_gpu_admitted.py``,
its watchdog and the admitted trainer all run below the same bubblewrap
process.  The admission pipe and the admission-lock descriptor are therefore
created and handed directly from the existing watchdog to the trainer; this
launcher never reads, duplicates, renumbers, or closes either descriptor.

The host namespace contributes only immutable program/governance inputs, one
outer-fold-specific sealed-pack shard, explicitly named writable lifecycle
directories, a minimal system/Python/CUDA runtime, and standard pseudo
filesystems.  The child starts with a cleared environment and no network.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import ctypes
from dataclasses import dataclass, field
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Callable, Final, Iterable, Mapping, Sequence
import zipfile


CAMPAIGN_ID: Final[str] = (
    "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
)
SCHEMA_VERSION: Final[int] = 1
SCIENTIFIC_CAMPAIGN_REVISION: Final[str] = "V8R4"
INFRASTRUCTURE_REVISION: Final[str] = "V8R4A"
BWRAP_BINARY: Final[Path] = Path("/usr/bin/bwrap")
BWRAP_VERSION: Final[str] = "bubblewrap 0.11.1"
RECEIPT_CLASSIFICATION: Final[str] = (
    "adaptive_v3r1_v8r4a_outer_target_sealed_runtime_capability_receipt"
)
PACK_INDEX_CLASSIFICATION: Final[str] = (
    "adaptive_v3r1_v8r4_nonouter_training_index"
)
PREDICTION_PACK_INDEX_CLASSIFICATION: Final[str] = (
    "adaptive_v3r1_v8r4a_model_bound_target_free_prediction_shard_index"
)
PREDICTION_PACK_INDEX_FILENAME: Final[str] = (
    "V8R4A_MODEL_BOUND_TARGET_FREE_PREDICTION_INDEX.json"
)
MODEL_BOUND_PREDICTION_PACK_CLASSIFICATION: Final[str] = (
    "adaptive_v3r1_v8r4a_model_bound_target_free_prediction_pack"
)
MODEL_SOURCE_CAPABILITY_CLASSIFICATION: Final[str] = (
    "adaptive_v3r1_v8r4a_promotion_model_source_capability"
)
MODEL_SOURCE_CAPABILITY_FILENAME: Final[str] = "MODEL_SOURCE_CAPABILITY.json"
MODEL_SOURCE_SHARD_SEAL_FILENAME: Final[str] = "MODEL_SOURCE_SHARD_SEAL.json"
MODEL_BOUND_PREDICTION_MANIFEST_FILENAME: Final[str] = (
    "MODEL_BOUND_OUTER_PREDICTION_PACK_MANIFEST.json"
)
INTERNAL_SPEC_PATH: Final[Path] = Path("/run/snn_rr/v8r4a_runtime_spec.json")
CAPABILITY_RECEIPT_FILENAME: Final[str] = (
    "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json"
)
COMPLETION_RECEIPT_FILENAME: Final[str] = (
    "TARGET_SEALED_COMPLETION_RECEIPT_V8R4A.json"
)
COMPLETION_RECEIPT_CLASSIFICATION: Final[str] = (
    "adaptive_v3r1_v8r4a_target_sealed_runtime_completion_receipt"
)
MIGRATION_MODULE_RELATIVE_PATH: Final[Path] = Path(
    "scripts/migrate_hfr_v3r1_gpu_state_v8r4a.py"
)
GPU_ADMISSION_WRAPPER_RELATIVE_PATH: Final[Path] = Path(
    "scripts/run_gpu_admitted.py"
)
GPU_BUDGET_MODULE_RELATIVE_PATH: Final[Path] = Path(
    "src/snn_rr/gpu_budget_ledger.py"
)
SNN_RR_PACKAGE_INIT_RELATIVE_PATH: Final[Path] = Path("src/snn_rr/__init__.py")
GPU_STATE_ROOT_RELATIVE: Final[Path] = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/gpu_state_v8r4a"
)
GPU_STATE_ROOT_AUTHORIZED_ST_DEV: Final[int] = 66306
GPU_STATE_ROOT_AUTHORIZED_ST_INO: Final[int] = 6970105
RUNS_ROOT_RELATIVE: Final[Path] = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1"
)
TARGET_LIFECYCLE_ROOT_RELATIVE: Final[Path] = (
    RUNS_ROOT_RELATIVE / "target_sealed_lifecycle_v8r4a_context1"
)
BENCHMARK_OUTPUT_RELATIVE: Final[Path] = (
    RUNS_ROOT_RELATIVE / "efficiency_benchmark_v8r4a_context1"
)
OTHER_OUTPUT_ROOT_RELATIVE: Final[Path] = (
    RUNS_ROOT_RELATIVE / "efficiency_benchmark_v8"
)
LEGACY_EXECUTION_CLOSURE_BENCHMARK_OUTPUT_RELATIVE: Final[Path] = (
    RUNS_ROOT_RELATIVE / "efficiency_benchmark_v8r4a"
)
SUPERSEDED_V8R4A_LIFECYCLE_ROOT_RELATIVE: Final[Path] = (
    RUNS_ROOT_RELATIVE / "target_sealed_lifecycle_v8r4a"
)
SUPERSEDED_V8R4A_OUTPUT_ROOT_RELATIVE: Final[Path] = (
    RUNS_ROOT_RELATIVE / "efficiency_benchmark_v8r4a"
)
SUPERSEDED_V8R4A_CONTRACT1_LIFECYCLE_ROOT_RELATIVE: Final[Path] = (
    RUNS_ROOT_RELATIVE / "target_sealed_lifecycle_v8r4a_contract1"
)
SUPERSEDED_V8R4A_CONTRACT1_OUTPUT_ROOT_RELATIVE: Final[Path] = (
    RUNS_ROOT_RELATIVE / "efficiency_benchmark_v8r4a_contract1"
)
SUPERSEDED_V8R4A_ROOTBIND1_LIFECYCLE_ROOT_RELATIVE: Final[Path] = (
    RUNS_ROOT_RELATIVE / "target_sealed_lifecycle_v8r4a_rootbind1"
)
SUPERSEDED_V8R4A_ROOTBIND1_OUTPUT_ROOT_RELATIVE: Final[Path] = (
    RUNS_ROOT_RELATIVE / "efficiency_benchmark_v8r4a_rootbind1"
)
DISCOVERY_OUTPUT_ROOT_RELATIVE: Final[Path] = RUNS_ROOT_RELATIVE / "discovery_v8r4"
FIXED_OUTPUT_ROOT_RELATIVE: Final[Path] = RUNS_ROOT_RELATIVE / "fixed_oof_v8r4"
CAMPAIGN_CONTRACT_RELATIVE_PATH: Final[Path] = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json"
)
GPU_STATE_DIRECTORY_RELATIVE_PATHS: Final[Mapping[str, Path]] = {
    role: GPU_STATE_ROOT_RELATIVE / role for role in ("admission", "execution", "usage")
}
GPU_STATE_EXACT_ENTRIES: Final[Mapping[str, frozenset[str]]] = {
    "admission": frozenset({"gpu_admission_v7.lock"}),
    "execution": frozenset(
        {"gpu_execution_ledger_v7.jsonl", "gpu_execution_ledger_v7.jsonl.lock"}
    ),
    "usage": frozenset(
        {"campaign_gpu_usage_chain_v6.jsonl", "campaign_gpu_usage_chain_v6.jsonl.lock"}
    ),
}
GPU_STATE_FILE_ROLES: Final[Mapping[str, tuple[str, str]]] = {
    "admission_lock": ("admission", "gpu_admission_v7.lock"),
    "execution_ledger": ("execution", "gpu_execution_ledger_v7.jsonl"),
    "execution_ledger_lock": ("execution", "gpu_execution_ledger_v7.jsonl.lock"),
    "usage_ledger": ("usage", "campaign_gpu_usage_chain_v6.jsonl"),
    "usage_ledger_lock": ("usage", "campaign_gpu_usage_chain_v6.jsonl.lock"),
}
ADMITTED_CHILD_FD_ENV: Final[str] = "SNN_RR_ADMITTED_CHILD_FD"

SEEDS: Final[tuple[int, ...]] = (20260828, 20260829, 20260830)
DISCOVERY_OUTER_FOLDS: Final[frozenset[int]] = frozenset({3, 4})
ALL_OUTER_FOLDS: Final[frozenset[int]] = frozenset(range(6))
PHASES: Final[frozenset[str]] = frozenset(
    {
        "efficiency_benchmark",
        "discovery",
        "promotion_training",
        "promotion_prediction",
        "discovery_aggregation",
        "promotion_aggregation",
    }
)

ENTRY_SCRIPT_BY_PHASE: Final[Mapping[str, frozenset[str]]] = {
    "efficiency_benchmark": frozenset({"benchmark_hfr_v3r1_efficiency.py"}),
    "discovery": frozenset({"run_hfr_v3r1_discovery_campaign.py"}),
    "promotion_training": frozenset({"run_fixed_hfr_v3r1_oof_campaign.py"}),
    "promotion_prediction": frozenset({"run_fixed_hfr_v3r1_oof_campaign.py"}),
    "discovery_aggregation": frozenset({"run_hfr_v3r1_discovery_campaign.py"}),
    "promotion_aggregation": frozenset({"run_fixed_hfr_v3r1_oof_campaign.py"}),
}

COMMON_GOVERNANCE_ROLES: Final[frozenset[str]] = frozenset(
    {
        "correction_authorization",
        "infrastructure_correction_authorization",
        "failure_diagnostic",
        "infrastructure_failure_diagnostic",
        "source_closure_correction_authorization",
        "source_closure_dependency_authorization",
        "source_closure_failure_diagnostic",
        "kill_safe_correction_authorization",
        "kill_safe_failure_diagnostic",
        "open_lifecycle_recovery_correction_authorization",
        "open_lifecycle_recovery_failure_diagnostic",
        "execution_closure_correction_authorization",
        "execution_closure_failure_diagnostic",
        "migration_source_succession_correction_authorization",
        "migration_source_succession_failure_diagnostic",
        "fd_closure_correction_authorization",
        "fd_closure_failure_diagnostic",
        "canary_boundary_correction_authorization",
        "canary_boundary_failure_diagnostic",
        "frozen_contract_encoding_correction_authorization",
        "frozen_contract_encoding_failure_diagnostic",
        "gpu_state_parent_bind_correction_authorization",
        "gpu_state_parent_bind_failure_diagnostic",
        "admitted_context_correction_authorization",
        "admitted_context_failure_diagnostic",
        "gpu_state_migration_receipt",
        "active_authorization",
        "source_snapshot",
        "implementation_test_receipt",
        "campaign_contract",
    }
)
GOVERNANCE_ROLES_BY_PHASE: Final[Mapping[str, frozenset[str]]] = {
    "efficiency_benchmark": COMMON_GOVERNANCE_ROLES
    | frozenset({"sealed_pack_index"}),
    "discovery": COMMON_GOVERNANCE_ROLES
    | frozenset(
        {
            "sealed_pack_index",
            "benchmark_receipt",
            "quarantine_owner_receipt",
            "quarantined_output_seal",
            *(f"quarantined_material_{number:02d}" for number in range(11)),
        }
    ),
    "promotion_training": COMMON_GOVERNANCE_ROLES
    | frozenset(
        {
            "sealed_pack_index",
            "discovery_completion_seal",
            "selection_lock",
            "promotion_authorization",
        }
    ),
    "promotion_prediction": COMMON_GOVERNANCE_ROLES
    | frozenset(
        {
            "sealed_pack_index",
            "discovery_completion_seal",
            "selection_lock",
            "promotion_authorization",
        }
    ),
    "discovery_aggregation": COMMON_GOVERNANCE_ROLES,
    "promotion_aggregation": COMMON_GOVERNANCE_ROLES
    | frozenset({"selection_lock", "promotion_authorization"}),
}

DISCOVERY_AGGREGATION_GOVERNANCE_ROLES: Final[frozenset[str]] = frozenset(
    {"discovery_shard_seal_outer3", "discovery_shard_seal_outer4"}
)
FIXED_AGGREGATION_GOVERNANCE_ROLES: Final[frozenset[str]] = frozenset(
    {
        *(f"model_source_seal_outer{outer}" for outer in range(6)),
        *(f"prediction_shard_seal_outer{outer}" for outer in range(6)),
    }
)


def _governance_roles_for(*, phase: str, entry_name: str) -> frozenset[str]:
    base = GOVERNANCE_ROLES_BY_PHASE[phase]
    if phase == "discovery_aggregation":
        if entry_name != "run_hfr_v3r1_discovery_campaign.py":
            raise TargetSealedError("discovery aggregation entry is not canonical")
        return base | DISCOVERY_AGGREGATION_GOVERNANCE_ROLES
    if phase == "promotion_aggregation":
        if entry_name != "run_fixed_hfr_v3r1_oof_campaign.py":
            raise TargetSealedError("promotion aggregation entry is not canonical")
        return base | FIXED_AGGREGATION_GOVERNANCE_ROLES
    if phase in PHASES:
        return base
    raise TargetSealedError("phase governance mapping is not canonical")


def _canonical_output_relative(
    *, phase: str, outer_fold: int | None, entry_name: str
) -> Path:
    if phase == "efficiency_benchmark":
        return BENCHMARK_OUTPUT_RELATIVE
    if phase == "discovery":
        assert outer_fold is not None
        return DISCOVERY_OUTPUT_ROOT_RELATIVE / "shards" / f"outer_{outer_fold}"
    if phase == "promotion_training":
        assert outer_fold is not None
        return (
            FIXED_OUTPUT_ROOT_RELATIVE
            / "promotion_training_shards"
            / f"outer_{outer_fold}"
        )
    if phase == "promotion_prediction":
        assert outer_fold is not None
        return FIXED_OUTPUT_ROOT_RELATIVE / "prediction_shards" / f"outer_{outer_fold}"
    if phase == "discovery_aggregation" and entry_name == "run_hfr_v3r1_discovery_campaign.py":
        return DISCOVERY_OUTPUT_ROOT_RELATIVE / "aggregation_v8r4a"
    if phase == "promotion_aggregation" and entry_name == "run_fixed_hfr_v3r1_oof_campaign.py":
        return FIXED_OUTPUT_ROOT_RELATIVE / "aggregation_v8r4a"
    raise TargetSealedError("phase/entry output topology is not canonical")


def _canonical_lifecycle_relative(
    *, phase: str, outer_fold: int | None, entry_name: str
) -> Path:
    scope = "global" if outer_fold is None else f"outer_{outer_fold}"
    return (
        TARGET_LIFECYCLE_ROOT_RELATIVE
        / phase
        / Path(entry_name).stem
        / scope
    )

WRITABLE_ROLES: Final[frozenset[str]] = frozenset(
    {
        "output",
        "lifecycle",
        "usage",
        "execution",
        "admission",
    }
)
WRITABLE_DIRECTORY_ROLES: Final[frozenset[str]] = frozenset(
    WRITABLE_ROLES
)
GPU_STATE_DIRECTORY_ROLES: Final[frozenset[str]] = frozenset(
    {"usage", "execution", "admission"}
)
MANDATORY_DENIED_CANARY_ROLES: Final[frozenset[str]] = frozenset(
    {
        "legacy_combined_cache",
        "raw_input_root",
        "target_root",
        "hai_experiment",
        "unadmitted_pack_root",
        "other_output_root",
        "superseded_v8r4a_lifecycle_root",
        "superseded_v8r4a_output_root",
        "superseded_v8r4a_contract1_lifecycle_root",
        "superseded_v8r4a_contract1_output_root",
        "superseded_v8r4a_rootbind1_lifecycle_root",
        "superseded_v8r4a_rootbind1_output_root",
    }
)

SAFE_PROPAGATED_ENV: Final[frozenset[str]] = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NVIDIA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
)
FIXED_CHILD_ENV: Final[Mapping[str, str]] = {
    "HOME": "/tmp/home",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPYCACHEPREFIX": "/tmp/pycache",
    "TZ": "UTC",
    "XDG_CACHE_HOME": "/tmp/cache",
    "TORCH_HOME": "/tmp/torch",
    "TRITON_CACHE_DIR": "/tmp/triton",
    "NUMBA_CACHE_DIR": "/tmp/numba",
}

FORBIDDEN_ENV_NAMES: Final[frozenset[str]] = frozenset(
    {
        "HAI_EXPERIMENT",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONUSERBASE",
        "LD_PRELOAD",
        "BASH_ENV",
        "ENV",
    }
)
CUDA_DEVICE_BASENAMES: Final[tuple[str, ...]] = (
    "nvidiactl",
    "nvidia-uvm",
    "nvidia-uvm-tools",
    "nvidia-modeset",
)


class TargetSealedError(RuntimeError):
    """A deterministic fail-closed V8R4 runtime error."""


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TargetSealedError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TargetSealedError(f"value is not canonical JSON: {error}") from error


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _decode_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                TargetSealedError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TargetSealedError) as error:
        raise TargetSealedError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise TargetSealedError(f"{label} must be a JSON object")
    return value


def _mode(status: os.stat_result) -> int:
    return stat.S_IMODE(status.st_mode)


def _absolute_lexical(path: Path, *, label: str) -> Path:
    value = os.fspath(path.expanduser())
    if not value or "\x00" in value:
        raise TargetSealedError(f"{label} is invalid")
    if ".." in PurePosixPath(value).parts:
        raise TargetSealedError(f"{label} contains traversal")
    absolute = Path(os.path.abspath(value))
    return absolute


def _canonical_existing(path: Path, *, label: str) -> Path:
    absolute = _absolute_lexical(path, label=label)
    try:
        lexical = os.lstat(absolute)
    except OSError as error:
        raise TargetSealedError(f"{label} is unavailable: {error}") from error
    if stat.S_ISLNK(lexical.st_mode):
        raise TargetSealedError(f"{label} final component is a symlink")
    resolved = absolute.resolve(strict=True)
    if resolved != absolute:
        raise TargetSealedError(f"{label} has a symlinked ancestor")
    return resolved


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class FileBinding:
    path: Path
    sha256: str
    bytes: int
    st_dev: int
    st_ino: int
    mode: int

    def document(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
            "st_dev": self.st_dev,
            "st_ino": self.st_ino,
            "mode": f"{self.mode:04o}",
        }


@dataclass(frozen=True, slots=True)
class DirectoryBinding:
    path: Path
    st_dev: int
    st_ino: int
    mode: int
    tree_sha256: str | None = None
    tree_files: int | None = None
    tree_bytes: int | None = None

    def document(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "path": str(self.path),
            "st_dev": self.st_dev,
            "st_ino": self.st_ino,
            "mode": f"{self.mode:04o}",
        }
        if self.tree_sha256 is not None:
            value.update(
                {
                    "tree_sha256": self.tree_sha256,
                    "tree_files": self.tree_files,
                    "tree_bytes": self.tree_bytes,
                }
            )
        return value


@dataclass(frozen=True, slots=True)
class LiveStateSnapshot:
    """Canonical, JSON-safe result of one public migration live validation."""

    receipt_binding: dict[str, Any]
    directory_bindings: dict[str, dict[str, Any]]
    current_file_bindings: dict[str, dict[str, Any]]
    usage_state: dict[str, Any]
    execution_state: dict[str, Any]

    def document(self) -> dict[str, Any]:
        return {
            "migration_receipt": dict(self.receipt_binding),
            "directories": {
                role: dict(binding)
                for role, binding in sorted(self.directory_bindings.items())
            },
            "files": {
                role: dict(binding)
                for role, binding in sorted(self.current_file_bindings.items())
            },
            "usage_state": dict(self.usage_state),
            "execution_state": dict(self.execution_state),
        }


def _json_safe_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TargetSealedError(f"{label} is not a mapping")
    try:
        normalized = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TargetSealedError) as error:
        raise TargetSealedError(f"{label} is not canonical JSON data: {error}") from error
    if not isinstance(normalized, dict):
        raise TargetSealedError(f"{label} is not a JSON object")
    return normalized


def _load_migration_validator(project_root: Path) -> Callable[..., Any]:
    module_path = _canonical_existing(
        project_root / MIGRATION_MODULE_RELATIVE_PATH,
        label="V8R4A migration validator module",
    )
    module_name = "_snn_rr_v8r4a_migration_" + hashlib.sha256(
        os.fspath(module_path).encode("utf-8")
    ).hexdigest()[:16]
    specification = importlib.util.spec_from_file_location(module_name, module_path)
    if specification is None or specification.loader is None:
        raise TargetSealedError("cannot load V8R4A migration validator module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    prior_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.dont_write_bytecode = prior_dont_write_bytecode
    validator = getattr(module, "validate_migrated_state", None)
    if not callable(validator):
        raise TargetSealedError("V8R4A migration module lacks its public validator")
    return validator


def _expected_gpu_state_paths(project_root: Path) -> dict[str, Path]:
    result = {
        f"{role}_directory": project_root / relative
        for role, relative in GPU_STATE_DIRECTORY_RELATIVE_PATHS.items()
    }
    result.update(
        {
            file_role: result[f"{directory_role}_directory"] / filename
            for file_role, (directory_role, filename) in GPU_STATE_FILE_ROLES.items()
        }
    )
    return result


def _validate_live_state_result(
    result: object,
    *,
    project_root: Path,
    receipt_binding: FileBinding,
) -> LiveStateSnapshot:
    expected_paths = _expected_gpu_state_paths(project_root)
    receipt_path = getattr(result, "receipt_path", None)
    receipt = getattr(result, "receipt", None)
    returned_receipt_binding = _json_safe_mapping(
        getattr(result, "receipt_binding", None), label="migration receipt binding"
    )
    canonical_paths = getattr(result, "canonical_paths", None)
    directories = _json_safe_mapping(
        getattr(result, "directory_bindings", None), label="migrated directory bindings"
    )
    files = _json_safe_mapping(
        getattr(result, "current_file_bindings", None), label="migrated file bindings"
    )
    usage_state = _json_safe_mapping(
        getattr(result, "usage_state", None), label="migrated usage state"
    )
    execution_state = _json_safe_mapping(
        getattr(result, "execution_state", None), label="migrated execution state"
    )
    if receipt_path != receipt_binding.path or not isinstance(receipt, Mapping):
        raise TargetSealedError("migration validator receipt path/result drifted")
    if not (
        receipt.get("schema_version") == 1
        and receipt.get("classification")
        == "adaptive_v3r1_v8r4a_gpu_state_migration_receipt"
        and receipt.get("campaign_id") == CAMPAIGN_ID
        and receipt.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and receipt.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and receipt.get("production_runtime_authorized") is False
    ):
        raise TargetSealedError("migration validator receipt identity drifted")
    if (
        returned_receipt_binding.get("path")
        != receipt_binding.path.relative_to(project_root).as_posix()
        or returned_receipt_binding.get("sha256") != receipt_binding.sha256
        or returned_receipt_binding.get("bytes") != receipt_binding.bytes
        or returned_receipt_binding.get("mode") != "0444"
        or returned_receipt_binding.get("nlink") != 1
    ):
        raise TargetSealedError("migration validator receipt binding drifted")
    if not isinstance(canonical_paths, Mapping) or set(canonical_paths) != set(
        expected_paths
    ):
        raise TargetSealedError("migration validator canonical path set drifted")
    if any(canonical_paths[role] != path for role, path in expected_paths.items()):
        raise TargetSealedError("migration validator canonical path drifted")

    root_relative = GPU_STATE_ROOT_RELATIVE.as_posix()
    if set(directories) != {"root", *GPU_STATE_DIRECTORY_ROLES}:
        raise TargetSealedError("migrated directory role set drifted")
    expected_directory_rows = {
        "root": (root_relative, frozenset(GPU_STATE_DIRECTORY_ROLES)),
        **{
            role: (
                GPU_STATE_DIRECTORY_RELATIVE_PATHS[role].as_posix(),
                GPU_STATE_EXACT_ENTRIES[role],
            )
            for role in GPU_STATE_DIRECTORY_ROLES
        },
    }
    directory_identities: set[tuple[int, int]] = set()
    for role, (relative, exact_entries) in expected_directory_rows.items():
        row = directories.get(role)
        if not isinstance(row, dict) or set(row) != {
            "exact_entries",
            "mode",
            "path",
            "st_dev",
            "st_ino",
        }:
            raise TargetSealedError(f"migrated {role} directory binding schema drifted")
        identity = (row.get("st_dev"), row.get("st_ino"))
        if not (
            row.get("path") == relative
            and row.get("mode") == "0700"
            and row.get("exact_entries") == sorted(exact_entries)
            and type(identity[0]) is int
            and type(identity[1]) is int
            and identity[0] >= 0
            and identity[1] > 0
            and identity not in directory_identities
        ):
            raise TargetSealedError(f"migrated {role} directory binding drifted")
        directory_identities.add((int(identity[0]), int(identity[1])))

    if set(files) != set(GPU_STATE_FILE_ROLES):
        raise TargetSealedError("migrated state file role set drifted")
    file_identities: set[tuple[int, int]] = set()
    for file_role, (directory_role, filename) in GPU_STATE_FILE_ROLES.items():
        row = files.get(file_role)
        expected_relative = (
            GPU_STATE_DIRECTORY_RELATIVE_PATHS[directory_role] / filename
        ).as_posix()
        if not isinstance(row, dict) or set(row) != {
            "bytes",
            "mode",
            "nlink",
            "path",
            "sha256",
            "st_dev",
            "st_ino",
        }:
            raise TargetSealedError(f"migrated {file_role} binding schema drifted")
        identity = (row.get("st_dev"), row.get("st_ino"))
        if not (
            row.get("path") == expected_relative
            and row.get("mode") == "0644"
            and row.get("nlink") == 1
            and type(row.get("bytes")) is int
            and row["bytes"] >= 0
            and _is_sha256(row.get("sha256"))
            and type(identity[0]) is int
            and type(identity[1]) is int
            and identity[0] >= 0
            and identity[1] > 0
            and identity not in file_identities
        ):
            raise TargetSealedError(f"migrated {file_role} binding drifted")
        file_identities.add((int(identity[0]), int(identity[1])))
    if usage_state.get("open_reservation_count") != 0:
        raise TargetSealedError("migration validator returned open usage state")
    if execution_state.get("open_start_count") != 0:
        raise TargetSealedError("migration validator returned open execution state")
    return LiveStateSnapshot(
        receipt_binding=returned_receipt_binding,
        directory_bindings={role: dict(row) for role, row in directories.items()},
        current_file_bindings={role: dict(row) for role, row in files.items()},
        usage_state=usage_state,
        execution_state=execution_state,
    )


def _validate_migrated_state_live(
    *, project_root: Path, receipt_binding: FileBinding
) -> LiveStateSnapshot:
    validator = _load_migration_validator(project_root)
    try:
        result = validator(project_root, receipt_binding.path, require_closed=True)
    except BaseException as error:
        raise TargetSealedError(f"V8R4A migrated-state live validation failed: {error}") from error
    return _validate_live_state_result(
        result, project_root=project_root, receipt_binding=receipt_binding
    )


def _migrated_state_open_counts(
    *, project_root: Path, receipt_binding: FileBinding
) -> tuple[int, int]:
    """Inspect only whether recovery is needed; strict replay still follows."""

    validator = _load_migration_validator(project_root)
    try:
        result = validator(project_root, receipt_binding.path, require_closed=False)
        if getattr(result, "receipt_path", None) != receipt_binding.path:
            raise TargetSealedError("migration recovery probe receipt path drifted")
        returned_binding = _json_safe_mapping(
            getattr(result, "receipt_binding", None),
            label="migration recovery-probe receipt binding",
        )
        if not (
            returned_binding.get("path")
            == receipt_binding.path.relative_to(project_root).as_posix()
            and returned_binding.get("sha256") == receipt_binding.sha256
            and returned_binding.get("bytes") == receipt_binding.bytes
            and returned_binding.get("mode") == "0444"
            and returned_binding.get("nlink") == 1
            and returned_binding.get("st_dev") == receipt_binding.st_dev
            and returned_binding.get("st_ino") == receipt_binding.st_ino
        ):
            raise TargetSealedError("migration recovery-probe receipt binding drifted")
        usage = _json_safe_mapping(
            getattr(result, "usage_state", None),
            label="migration recovery-probe usage state",
        )
        execution = _json_safe_mapping(
            getattr(result, "execution_state", None),
            label="migration recovery-probe execution state",
        )
        usage_count = usage.get("open_reservation_count")
        execution_count = execution.get("open_start_count")
        if not (
            type(usage_count) is int
            and usage_count >= 0
            and type(execution_count) is int
            and execution_count >= 0
        ):
            raise TargetSealedError("migration recovery-probe counts are invalid")
        return usage_count, execution_count
    except TargetSealedError:
        raise
    except BaseException as error:
        raise TargetSealedError(
            f"V8R4A migrated-state recovery probe failed: {error}"
        ) from error


def _load_exact_gpu_recovery_modules(
    project_root: Path,
    *,
    source_snapshot_bindings: Mapping[Path, FileBinding],
) -> tuple[ModuleType, ModuleType]:
    """Load the frozen wrapper/accounting pair without borrowing host modules.

    ``run_gpu_admitted.py`` intentionally imports ``snn_rr.gpu_budget_ledger``
    by its normal package name.  The launcher can already have an unrelated
    ``snn_rr`` generation in ``sys.modules`` (especially under pytest), so load
    the exact snapshot-bound package generation in a short isolated module
    namespace and restore the caller's namespace before returning.  The
    wrapper retains a direct reference to the exact budget module object.
    """

    paths = {
        "package": _canonical_existing(
            project_root / SNN_RR_PACKAGE_INIT_RELATIVE_PATH,
            label="snapshot-bound snn_rr package initializer",
        ),
        "budget": _canonical_existing(
            project_root / GPU_BUDGET_MODULE_RELATIVE_PATH,
            label="snapshot-bound GPU budget module",
        ),
        "wrapper": _canonical_existing(
            project_root / GPU_ADMISSION_WRAPPER_RELATIVE_PATH,
            label="snapshot-bound GPU admission wrapper",
        ),
    }
    for label, path in paths.items():
        binding = source_snapshot_bindings.get(path)
        if binding is None:
            raise TargetSealedError(
                f"source snapshot omits the GPU recovery {label}"
            )
        refreshed, _ = _read_file_binding(
            path,
            label=f"GPU recovery {label}",
            require_immutable=True,
        )
        if refreshed != binding:
            raise TargetSealedError(
                f"GPU recovery {label} differs from the pinned source snapshot"
            )

    wrapper_name = "_snn_rr_v8r4a_gpu_recovery_" + hashlib.sha256(
        canonical_json_bytes(
            {
                label: source_snapshot_bindings[path].sha256
                for label, path in sorted(paths.items())
            }
        )
    ).hexdigest()[:16]
    module_names = ("snn_rr", "snn_rr.gpu_budget_ledger", wrapper_name)
    absent = object()
    saved: dict[str, object] = {
        name: sys.modules.get(name, absent) for name in module_names
    }
    saved_sys_path = list(sys.path)
    prior_dont_write_bytecode = sys.dont_write_bytecode
    try:
        package_spec = importlib.util.spec_from_file_location(
            "snn_rr",
            paths["package"],
            submodule_search_locations=[os.fspath(paths["package"].parent)],
        )
        if package_spec is None or package_spec.loader is None:
            raise TargetSealedError("cannot load snapshot-bound snn_rr package")
        package = importlib.util.module_from_spec(package_spec)
        sys.modules["snn_rr"] = package
        sys.dont_write_bytecode = True
        package_spec.loader.exec_module(package)

        budget_spec = importlib.util.spec_from_file_location(
            "snn_rr.gpu_budget_ledger", paths["budget"]
        )
        if budget_spec is None or budget_spec.loader is None:
            raise TargetSealedError("cannot load snapshot-bound GPU budget module")
        budget_module = importlib.util.module_from_spec(budget_spec)
        sys.modules["snn_rr.gpu_budget_ledger"] = budget_module
        setattr(package, "gpu_budget_ledger", budget_module)
        budget_spec.loader.exec_module(budget_module)

        wrapper_spec = importlib.util.spec_from_file_location(
            wrapper_name, paths["wrapper"]
        )
        if wrapper_spec is None or wrapper_spec.loader is None:
            raise TargetSealedError("cannot load snapshot-bound GPU admission wrapper")
        wrapper_module = importlib.util.module_from_spec(wrapper_spec)
        sys.modules[wrapper_name] = wrapper_module
        wrapper_spec.loader.exec_module(wrapper_module)
        if (
            getattr(wrapper_module, "budget", None) is not budget_module
            or getattr(wrapper_module, "PROJECT_ROOT", None) != project_root
            or Path(str(getattr(wrapper_module, "__file__", ""))).resolve()
            != paths["wrapper"]
            or Path(str(getattr(budget_module, "__file__", ""))).resolve()
            != paths["budget"]
            or Path(str(getattr(package, "__file__", ""))).resolve()
            != paths["package"]
        ):
            raise TargetSealedError("GPU recovery module source identity drifted")
        required_wrapper = (
            "cleanup_execution_ledger_replace_residue",
            "_pinned_protected_tree",
            "_exclusive_gpu_lock",
            "_legacy_expectation",
            "_read_execution_locked",
            "_open_execution_starts",
            "_recover_closed_usage_execution_starts",
            "_validate_bound_reservation_authorization",
            "_validate_execution_recovery_records",
            "execution_ledger_lock_path",
        )
        required_budget = (
            "cleanup_usage_ledger_replace_residue",
            "ledger_lock_path",
            "verify_ledger",
            "reconcile_open_reservations",
            "_reconciliation_record",
            "boot_id",
            "process_start_ticks",
        )
        if any(not callable(getattr(wrapper_module, name, None)) for name in required_wrapper):
            raise TargetSealedError("GPU admission wrapper recovery ABI drifted")
        if any(not callable(getattr(budget_module, name, None)) for name in required_budget):
            raise TargetSealedError("GPU budget recovery ABI drifted")
        return wrapper_module, budget_module
    except TargetSealedError:
        raise
    except BaseException as error:
        raise TargetSealedError(
            f"cannot load snapshot-bound GPU recovery modules: {error}"
        ) from error
    finally:
        sys.dont_write_bytecode = prior_dont_write_bytecode
        sys.path[:] = saved_sys_path
        for name in reversed(module_names):
            previous = saved[name]
            if previous is absent:
                sys.modules.pop(name, None)
            else:
                assert isinstance(previous, ModuleType)
                sys.modules[name] = previous


def _require_demonstrably_dead_open_reservation(
    reservation: Mapping[str, Any], *, budget_module: ModuleType
) -> None:
    """Refuse recovery unless boot/PID identity proves the wrapper cannot live."""

    try:
        current_boot = budget_module.boot_id()
        reservation_boot = reservation["boot_id"]
        wrapper_pid = reservation["wrapper_pid"]
        wrapper_ticks = reservation["wrapper_start_ticks"]
        if not (
            isinstance(reservation_boot, str)
            and reservation_boot
            and type(wrapper_pid) is int
            and wrapper_pid > 0
            and type(wrapper_ticks) is int
            and wrapper_ticks > 0
        ):
            raise TargetSealedError("open GPU reservation process identity is invalid")
        if reservation_boot != current_boot:
            # A Linux boot-id change is itself conclusive process-death proof.
            return
        observed_ticks = budget_module.process_start_ticks(wrapper_pid)
    except TargetSealedError:
        raise
    except BaseException as error:
        raise TargetSealedError(
            f"cannot prove open GPU reservation owner death: {error}"
        ) from error
    if observed_ticks is not None and observed_ticks == wrapper_ticks:
        raise TargetSealedError(
            "open GPU reservation still belongs to its matching live wrapper"
        )


def _recover_dead_gpu_lifecycle_before_closed_validation(
    *,
    project_root: Path,
    receipt_binding: FileBinding,
    source_snapshot_bindings: Mapping[Path, FileBinding],
) -> int:
    """Conservatively close one dead wrapper lifecycle, then return.

    This is the host-side counterpart of the unchanged wrapper's normal
    admission preamble.  It acquires the stable admission lock first, checks
    the exact usage/execution generations from pinned directory capabilities,
    proves that the sole open reservation's wrapper is dead, lets the existing
    budget code charge a reconciled terminal, and lets the existing wrapper
    code append the matching execution end.  Callers must still run the strict
    ``require_closed=True`` migrated-state validator immediately afterwards.
    """

    paths = _expected_gpu_state_paths(project_root)
    admission_lock = paths["admission_lock"]
    usage_ledger = paths["usage_ledger"]
    execution_ledger = paths["execution_ledger"]
    usage_lock = paths["usage_ledger_lock"]
    execution_lock = paths["execution_ledger_lock"]
    residue_paths = (
        usage_ledger.with_name(f".{usage_ledger.name}.v8r4a-replace.tmp"),
        execution_ledger.with_name(f".{execution_ledger.name}.v8r4a-replace.tmp"),
    )
    # The strict migration probe is the cheapest complete proof when no exact
    # deterministic replace residue is even named.  If either residue exists,
    # the strict-inventory validator is expected to fail, so proceed directly
    # to the admission-guarded narrow cleanup instead of trusting an unlocked
    # inspection of its type or bytes.
    if not any(os.path.lexists(path) for path in residue_paths) and (
        _migrated_state_open_counts(
            project_root=project_root, receipt_binding=receipt_binding
        )
        == (0, 0)
    ):
        return 0

    wrapper, budget_module = _load_exact_gpu_recovery_modules(
        project_root, source_snapshot_bindings=source_snapshot_bindings
    )
    if wrapper.execution_ledger_lock_path(execution_ledger) != execution_lock:
        raise TargetSealedError("GPU execution recovery lock path drifted")
    if budget_module.ledger_lock_path(usage_ledger) != usage_lock:
        raise TargetSealedError("GPU usage recovery lock path drifted")

    protected_paths = (
        admission_lock,
        usage_ledger,
        usage_lock,
        execution_ledger,
        execution_lock,
    )
    stable_paths = (admission_lock, usage_lock, execution_lock)
    try:
        with wrapper._pinned_protected_tree(
            protected_paths, stable_paths=stable_paths
        ) as pins, wrapper._exclusive_gpu_lock(
            admission_lock, pinned_directory_fd=pins.lease_for(admission_lock)
        ):
            pins.revalidate()
            usage_fd = pins.fd_for(usage_ledger)
            execution_fd = pins.fd_for(execution_ledger)
            expected_genesis = wrapper._legacy_expectation(
                usage_ledger, pinned_directory_fd=usage_fd
            )
            cleaned_usage = budget_module.cleanup_usage_ledger_replace_residue(
                usage_ledger,
                expected_legacy_genesis_sha256=expected_genesis,
                pinned_directory_fd=pins.lease_for(usage_ledger),
                admission_revalidate=pins.revalidate,
            )
            cleaned_execution = wrapper.cleanup_execution_ledger_replace_residue(
                execution_ledger,
                pinned_directory_fd=pins.lease_for(execution_ledger),
                admission_revalidate=pins.revalidate,
            )
            pins.revalidate()
            if _migrated_state_open_counts(
                project_root=project_root, receipt_binding=receipt_binding
            ) == (0, 0):
                return int(cleaned_usage) + int(cleaned_execution)
            state = budget_module.verify_ledger(
                usage_ledger,
                expected_legacy_genesis_sha256=expected_genesis,
                pinned_directory_fd=usage_fd,
            )
            open_reservations = list(state.open_reservations.values())
            if len(open_reservations) > 1:
                raise TargetSealedError(
                    "GPU preflight refuses multiple open reservations"
                )

            _raw, execution_rows = wrapper._read_execution_locked(
                execution_ledger, execution_fd
            )
            open_starts = wrapper._open_execution_starts(execution_rows)
            if open_reservations:
                item = open_reservations[0]
                reservation = item.reservation
                lifecycle_id = reservation.get("lifecycle_id")
                start = open_starts.get(lifecycle_id)
                if len(open_starts) > 1 or (
                    start is not None
                    and start.get("reservation_record_sha256")
                    != reservation.get("record_sha256")
                ) or (open_starts and start is None):
                    raise TargetSealedError(
                        "open GPU usage reservation conflicts with execution state"
                    )
                _require_demonstrably_dead_open_reservation(
                    reservation, budget_module=budget_module
                )
                wrapper._validate_bound_reservation_authorization(reservation)
                if start is not None:
                    command = start.get("command")
                    if not (
                        isinstance(command, list)
                        and command
                        and all(isinstance(part, str) and part for part in command)
                    ):
                        raise TargetSealedError(
                            "open GPU execution command is invalid"
                        )
                    tentative_terminal = budget_module._reconciliation_record(
                        item,
                        current_boot_id=budget_module.boot_id(),
                        realtime_ns=time.time_ns(),
                        monotonic_ns=time.monotonic_ns(),
                    )
                    wrapper._validate_execution_recovery_records(
                        start=start,
                        end=None,
                        terminal=tentative_terminal,
                        reservation=reservation,
                        expected_command=command,
                        expected_lock_file=admission_lock,
                        gpu_ledger=execution_ledger,
                        usage_ledger=usage_ledger,
                    )
            elif len(open_starts) > 1:
                raise TargetSealedError(
                    "GPU preflight refuses multiple stale execution starts"
                )

            pins.revalidate()
            reconciled, state = budget_module.reconcile_open_reservations(
                usage_ledger,
                realtime_ns=time.time_ns(),
                monotonic_ns=time.monotonic_ns(),
                current_boot_id=budget_module.boot_id(),
                expected_legacy_genesis_sha256=expected_genesis,
                pinned_directory_fd=pins.lease_for(usage_ledger),
            )
            pins.revalidate()
            recovered_execution = wrapper._recover_closed_usage_execution_starts(
                execution_ledger,
                state,
                expected_lock_file=admission_lock,
                usage_ledger=usage_ledger,
                pinned_directory_fd=pins.lease_for(execution_ledger),
            )
            pins.revalidate()
            if len(reconciled) != (1 if open_reservations else 0):
                raise TargetSealedError("GPU usage recovery count drifted")
            expected_execution_recovery = len(open_starts)
            if recovered_execution != expected_execution_recovery:
                raise TargetSealedError("GPU execution recovery count drifted")
            return (
                int(cleaned_usage)
                + int(cleaned_execution)
                + len(reconciled)
                + recovered_execution
            )
    except TargetSealedError:
        raise
    except BaseException as error:
        raise TargetSealedError(
            f"V8R4A open GPU lifecycle kill recovery failed: {error}"
        ) from error


def _read_file_binding(path: Path, *, label: str, require_immutable: bool) -> tuple[FileBinding, bytes]:
    resolved = _canonical_existing(path, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TargetSealedError(f"{label} is not a single-link regular file")
        if require_immutable and _mode(before) != 0o444:
            raise TargetSealedError(f"{label} mode must be 0444")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(resolved, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise TargetSealedError(f"{label} changed while hashing")
        if (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino):
            raise TargetSealedError(f"{label} inode changed while hashing")
        if after.st_nlink != 1 or len(raw) != after.st_size:
            raise TargetSealedError(f"{label} link/size changed while hashing")
        return (
            FileBinding(
                path=resolved,
                sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),
                st_dev=after.st_dev,
                st_ino=after.st_ino,
                mode=_mode(after),
            ),
            raw,
        )
    finally:
        os.close(descriptor)


def _open_pinned_directory(path: Path, *, label: str) -> tuple[int, DirectoryBinding]:
    resolved = _canonical_existing(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    status = os.fstat(descriptor)
    named = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISDIR(status.st_mode):
        os.close(descriptor)
        raise TargetSealedError(f"{label} is not a directory")
    if (status.st_dev, status.st_ino) != (named.st_dev, named.st_ino):
        os.close(descriptor)
        raise TargetSealedError(f"{label} inode changed while pinning")
    return descriptor, DirectoryBinding(
        path=resolved,
        st_dev=status.st_dev,
        st_ino=status.st_ino,
        mode=_mode(status),
    )


def _tree_binding(path: Path, *, label: str, frozen: bool) -> DirectoryBinding:
    root = _canonical_existing(path, label=label)
    root_status = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(root_status.st_mode):
        raise TargetSealedError(f"{label} is not a directory")
    if frozen and _mode(root_status) != 0o555:
        raise TargetSealedError(f"{label} root mode must be 0555")
    rows: list[dict[str, Any]] = []
    seen_inodes: set[tuple[int, int]] = set()
    total_bytes = 0
    file_count = 0
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        current_status = os.stat(current_path, follow_symlinks=False)
        if not stat.S_ISDIR(current_status.st_mode):
            raise TargetSealedError(f"{label} tree contains a non-directory ancestor")
        if current_path != root and frozen and _mode(current_status) != 0o555:
            raise TargetSealedError(
                f"{label} directory mode must be 0555: {current_path}"
            )
        relative_dir = current_path.relative_to(root).as_posix()
        rows.append(
            {
                "kind": "directory",
                "path": relative_dir,
                "mode": f"{_mode(current_status):04o}",
            }
        )
        for name in list(directories):
            entry = current_path / name
            status = os.lstat(entry)
            if stat.S_ISLNK(status.st_mode):
                raise TargetSealedError(f"{label} contains a symlink: {entry}")
            if not stat.S_ISDIR(status.st_mode):
                raise TargetSealedError(f"{label} contains an invalid directory entry")
        for name in files:
            entry = current_path / name
            status = os.lstat(entry)
            if stat.S_ISLNK(status.st_mode):
                raise TargetSealedError(f"{label} contains a symlink: {entry}")
            if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                raise TargetSealedError(f"{label} contains an aliased/special file: {entry}")
            inode = (status.st_dev, status.st_ino)
            if inode in seen_inodes:
                raise TargetSealedError(f"{label} contains duplicate file inode: {entry}")
            seen_inodes.add(inode)
            if frozen and _mode(status) != 0o444:
                raise TargetSealedError(f"{label} file mode must be 0444: {entry}")
            binding, _ = _read_file_binding(
                entry, label=f"{label} member", require_immutable=frozen
            )
            file_count += 1
            total_bytes += binding.bytes
            rows.append(
                {
                    "kind": "file",
                    "path": entry.relative_to(root).as_posix(),
                    "mode": f"{binding.mode:04o}",
                    "bytes": binding.bytes,
                    "sha256": binding.sha256,
                    "st_dev": binding.st_dev,
                    "st_ino": binding.st_ino,
                }
            )
    if file_count == 0:
        raise TargetSealedError(f"{label} is empty")
    final_status = os.stat(root, follow_symlinks=False)
    if (root_status.st_dev, root_status.st_ino) != (
        final_status.st_dev,
        final_status.st_ino,
    ):
        raise TargetSealedError(f"{label} root changed while sealing")
    return DirectoryBinding(
        path=root,
        st_dev=root_status.st_dev,
        st_ino=root_status.st_ino,
        mode=_mode(root_status),
        tree_sha256=semantic_sha256(rows),
        tree_files=file_count,
        tree_bytes=total_bytes,
    )


def _open_pinned_file(
    path: Path, *, label: str, require_immutable: bool
) -> tuple[int, FileBinding, bytes]:
    resolved = _canonical_existing(path, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        status = os.fstat(descriptor)
        named = os.stat(resolved, follow_symlinks=False)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or (status.st_dev, status.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise TargetSealedError(f"{label} is not a stable single-link file")
        if require_immutable and _mode(status) != 0o444:
            raise TargetSealedError(f"{label} mode must be 0444")
        raw = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            raw.extend(block)
        after = os.fstat(descriptor)
        if (
            (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or after.st_nlink != 1
            or len(raw) != after.st_size
        ):
            raise TargetSealedError(f"{label} changed while pinning")
        return (
            descriptor,
            FileBinding(
                path=resolved,
                sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),
                st_dev=after.st_dev,
                st_ino=after.st_ino,
                mode=_mode(after),
            ),
            bytes(raw),
        )
    except BaseException:
        os.close(descriptor)
        raise


def _validate_self_hash(document: Mapping[str, Any], *, label: str) -> None:
    value = dict(document)
    observed = value.pop("content_sha256", None)
    if not _is_sha256(observed) or observed != semantic_sha256(value):
        raise TargetSealedError(f"{label} content_sha256 drifted")


DISCOVERY_PACK_INDEX_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "outer_fold",
        "seeds",
        "unit_count",
        "completed_units",
        "status",
        "units",
        "outer_test_opened",
        "combined_target_bearing_cache_consumer_access_authorized",
        "physical_nonouter_training_packs",
        "outer_prediction_packs_absent",
        "cross_outer_shard_mounted",
        "content_sha256",
    }
)
PROMOTION_TRAINING_PACK_INDEX_KEYS: Final[frozenset[str]] = (
    DISCOVERY_PACK_INDEX_KEYS | frozenset({"promotion_scope", "promotion_authorization"})
)
PREDICTION_PACK_INDEX_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "outer_fold",
        "seeds",
        "unit_count",
        "completed_units",
        "status",
        "selected_variant",
        "units",
        "outer_test_opened",
        "combined_target_bearing_cache_consumer_access_authorized",
        "physical_target_free_input_and_model_packs",
        "source_paths_or_peer_outputs_authorized_in_child",
        "cross_outer_shard_mounted",
        "promotion_authorization",
        "model_source_shard_seal",
        "content_sha256",
    }
)
PACK_INDEX_UNIT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "outer_fold",
        "seed",
        "relative_path",
        "artifacts",
    }
)
PACK_INDEX_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {"cache_manifest", "proposer_stack", "partition_manifest"}
)
PREDICTION_PACK_INDEX_UNIT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "outer_fold",
        "seed",
        "relative_path",
        "scientific_signature_sha256",
        "row_count",
        "global_cache_index_sha256",
        "source_kind",
        "artifacts",
    }
)
PREDICTION_PACK_INDEX_ARTIFACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prediction_pack_manifest",
        "model_bound_prediction_pack_manifest",
        "outer_predict_input",
        "model_checkpoint",
        "model_scaler",
        "model_source_capability",
    }
)
MODEL_BOUND_UNIT_FILES: Final[frozenset[str]] = frozenset(
    {
        "OUTER_PREDICTION_PACK_MANIFEST.json",
        MODEL_BOUND_PREDICTION_MANIFEST_FILENAME,
        MODEL_SOURCE_CAPABILITY_FILENAME,
        "outer_predict_input.npz",
        "model_checkpoint.pt",
        "model_scaler.json",
    }
)
OUTER_PREDICT_FIELDS: Final[tuple[str, ...]] = (
    "cache_index",
    "node_features",
    "candidate_rr_bpm",
    "candidate_mask",
    "joint_radar_mask",
    "proposer_anchor_bpm",
    "proposer_anchor_std_bpm",
    "proposer_anchor_available",
    "classical_rr_bpm",
    "session_reset",
)
PREDICTION_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "outer_fold",
        "seed",
        "row_count",
        "fields",
        "exact_allowlist",
        "forbidden_fields_emitted",
        "reference_identity_protocol_quality_decoded",
        "legacy_index",
        "legacy_cache_manifest",
        "legacy_proposer_stack",
        "promotion_authorization",
        "output",
        "global_cache_index_sha256",
        "object_arrays",
        "pickle",
        "commercial_or_confirmatory_claim_allowed",
        "content_sha256",
    }
)
MODEL_BOUND_PREDICTION_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "outer_fold",
        "seed",
        "selected_variant",
        "row_count",
        "global_cache_index_sha256",
        "fields",
        "exact_target_free_allowlist",
        "selection_lock",
        "promotion_authorization",
        "base_target_free_manifest",
        "artifacts",
        "exact_unit_file_inventory",
        "prediction_child_reads_model_only_from_this_pack",
        "source_paths_or_peer_outputs_authorized_in_child",
        "target_reference_quality_identity_protocol_present",
        "model_bytes_changed",
        "commercial_or_confirmatory_claim_allowed",
        "content_sha256",
    }
)
MODEL_SOURCE_CAPABILITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "outer_fold",
        "seed",
        "selected_variant",
        "source_kind",
        "scientific_signature_sha256",
        "source_receipt",
        "source_checkpoint",
        "source_scaler",
        "packed_checkpoint",
        "packed_scaler",
        "selection_lock",
        "promotion_authorization",
        "source_deep_validated_before_copy",
        "source_paths_or_peer_outputs_authorized_in_child",
        "target_reference_quality_identity_protocol_present",
        "model_bytes_changed",
        "commercial_or_confirmatory_claim_allowed",
        "content_sha256",
    }
)
PROMOTION_CACHE_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "format_version",
        "complete",
        "outer_fold",
        "partition",
        "source_combined_cache_open_authorized_by_consumer",
        "outer_test_rows_physically_present",
        "outer_prediction_pack_absent",
        "inputs",
        "outputs",
        "promotion_scope",
        "promotion_authorization",
        "content_sha256",
    }
)
PROMOTION_PARTITION_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "campaign_id",
        "campaign_revision",
        "classification",
        "outer_fold",
        "seed",
        "legacy_row_count",
        "partition",
        "legacy_inputs",
        "outputs",
        "integration_interface",
        "protected_outer_access",
        "preselection_prediction_boundary",
        "serialization",
        "claim_boundary",
        "promotion_scope",
        "promotion_authorization",
        "content_sha256",
    }
)
PACK_INDEX_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {"path", "sha256", "bytes"}
)


def _validate_prediction_unit_payload(
    *,
    pack_root: DirectoryBinding,
    relative: str,
    outer_fold: int,
    seed: int,
    selected_variant: str,
    selection_lock: Mapping[str, Any],
    authorization: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    index_row: Mapping[str, Any],
) -> None:
    unit_root = pack_root.path / Path(relative)
    manifest_path = unit_root / "OUTER_PREDICTION_PACK_MANIFEST.json"
    _binding, raw = _read_file_binding(
        manifest_path, label="prediction-pack manifest", require_immutable=True
    )
    manifest = _decode_json_bytes(raw, label="prediction-pack manifest")
    if set(manifest) != PREDICTION_MANIFEST_KEYS:
        raise TargetSealedError("prediction-pack manifest schema drifted")
    _validate_self_hash(manifest, label="prediction-pack manifest")
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("classification")
        == "adaptive_v3r1_v8r4_authorized_outer_prediction_pack"
        and manifest.get("campaign_id") == CAMPAIGN_ID
        and manifest.get("campaign_revision") == SCIENTIFIC_CAMPAIGN_REVISION
        and manifest.get("outer_fold") == outer_fold
        and manifest.get("seed") == seed
        and type(manifest.get("row_count")) is int
        and manifest["row_count"] > 0
        and manifest.get("fields") == list(OUTER_PREDICT_FIELDS)
        and manifest.get("exact_allowlist") is True
        and manifest.get("forbidden_fields_emitted") is False
        and manifest.get("reference_identity_protocol_quality_decoded") is False
        and manifest.get("promotion_authorization") == dict(authorization)
        and manifest.get("object_arrays") is False
        and manifest.get("pickle") is False
        and manifest.get("commercial_or_confirmatory_claim_allowed") is False
        and _is_sha256(manifest.get("global_cache_index_sha256"))
    ):
        raise TargetSealedError("prediction-pack manifest capability drifted")
    for role in ("legacy_index", "legacy_cache_manifest", "legacy_proposer_stack"):
        row = manifest.get(role)
        if not (
            isinstance(row, dict)
            and set(row) == PACK_INDEX_BINDING_KEYS
            and isinstance(row.get("path"), str)
            and row["path"]
            and _is_sha256(row.get("sha256"))
            and type(row.get("bytes")) is int
            and row["bytes"] > 0
        ):
            raise TargetSealedError(f"prediction-pack {role} binding drifted")
    output = manifest.get("output")
    indexed_output = artifacts.get("outer_predict_input")
    if not (
        isinstance(output, dict)
        and set(output) == PACK_INDEX_BINDING_KEYS
        and isinstance(indexed_output, Mapping)
        and output.get("path") == "outer_predict_input.npz"
        and output.get("sha256") == indexed_output.get("sha256")
        and output.get("bytes") == indexed_output.get("bytes")
    ):
        raise TargetSealedError("prediction-pack NPZ manifest binding drifted")

    successor_path = unit_root / MODEL_BOUND_PREDICTION_MANIFEST_FILENAME
    _successor_binding, successor_raw = _read_file_binding(
        successor_path,
        label="model-bound prediction-pack manifest",
        require_immutable=True,
    )
    successor = _decode_json_bytes(
        successor_raw, label="model-bound prediction-pack manifest"
    )
    successor_artifacts = successor.get("artifacts")
    if set(successor) != MODEL_BOUND_PREDICTION_MANIFEST_KEYS:
        raise TargetSealedError("model-bound prediction-pack manifest schema drifted")
    _validate_self_hash(successor, label="model-bound prediction-pack manifest")
    if not (
        successor.get("schema_version") == 1
        and successor.get("classification")
        == MODEL_BOUND_PREDICTION_PACK_CLASSIFICATION
        and successor.get("campaign_id") == CAMPAIGN_ID
        and successor.get("campaign_revision") == SCIENTIFIC_CAMPAIGN_REVISION
        and successor.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and successor.get("outer_fold") == outer_fold
        and successor.get("seed") == seed
        and successor.get("selected_variant") == selected_variant
        and successor.get("row_count") == manifest.get("row_count")
        and successor.get("row_count") == index_row.get("row_count")
        and successor.get("global_cache_index_sha256")
        == manifest.get("global_cache_index_sha256")
        == index_row.get("global_cache_index_sha256")
        and successor.get("fields") == list(OUTER_PREDICT_FIELDS)
        and successor.get("exact_target_free_allowlist") is True
        and successor.get("selection_lock") == dict(selection_lock)
        and successor.get("promotion_authorization") == dict(authorization)
        and successor.get("exact_unit_file_inventory")
        == sorted(MODEL_BOUND_UNIT_FILES)
        and successor.get("prediction_child_reads_model_only_from_this_pack")
        is True
        and successor.get("source_paths_or_peer_outputs_authorized_in_child")
        is False
        and successor.get("target_reference_quality_identity_protocol_present")
        is False
        and successor.get("model_bytes_changed") is False
        and successor.get("commercial_or_confirmatory_claim_allowed") is False
        and isinstance(successor_artifacts, Mapping)
        and set(successor_artifacts)
        == {
            "outer_predict_input",
            "model_checkpoint",
            "model_scaler",
            "model_source_capability",
        }
    ):
        raise TargetSealedError("model-bound prediction-pack capability drifted")
    if manifest.get("promotion_authorization") != successor.get(
        "promotion_authorization"
    ):
        raise TargetSealedError("base/model-bound prediction authority drifted")
    _validate_relative_pack_member_binding(
        successor.get("base_target_free_manifest"),
        unit_root=unit_root,
        relative_path="OUTER_PREDICTION_PACK_MANIFEST.json",
        label="model-bound base target-free manifest",
    )
    for role, filename in {
        "outer_predict_input": "outer_predict_input.npz",
        "model_checkpoint": "model_checkpoint.pt",
        "model_scaler": "model_scaler.json",
        "model_source_capability": MODEL_SOURCE_CAPABILITY_FILENAME,
    }.items():
        _validate_relative_pack_member_binding(
            successor_artifacts.get(role),
            unit_root=unit_root,
            relative_path=filename,
            label=f"model-bound local artifact:{role}",
        )
    if not (
        output.get("sha256") == successor_artifacts["outer_predict_input"].get("sha256")
        and output.get("bytes") == successor_artifacts["outer_predict_input"].get("bytes")
    ):
        raise TargetSealedError("base/model-bound prediction input bytes drifted")

    capability_path = unit_root / MODEL_SOURCE_CAPABILITY_FILENAME
    _capability_binding, capability_raw = _read_file_binding(
        capability_path,
        label="model-source capability",
        require_immutable=True,
    )
    capability = _decode_json_bytes(capability_raw, label="model-source capability")
    if set(capability) != MODEL_SOURCE_CAPABILITY_KEYS:
        raise TargetSealedError("model-source capability schema drifted")
    _validate_self_hash(capability, label="model-source capability")
    opaque_fields = (
        "source_receipt",
        "source_checkpoint",
        "source_scaler",
        "packed_checkpoint",
        "packed_scaler",
        "selection_lock",
        "promotion_authorization",
    )
    if not (
        capability.get("schema_version") == 1
        and capability.get("classification")
        == MODEL_SOURCE_CAPABILITY_CLASSIFICATION
        and capability.get("campaign_id") == CAMPAIGN_ID
        and capability.get("campaign_revision") == SCIENTIFIC_CAMPAIGN_REVISION
        and capability.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and capability.get("outer_fold") == outer_fold
        and capability.get("seed") == seed
        and capability.get("selected_variant") == selected_variant
        and capability.get("source_kind") == index_row.get("source_kind")
        and capability.get("scientific_signature_sha256")
        == index_row.get("scientific_signature_sha256")
        and all(
            isinstance(capability.get(field), Mapping)
            and set(capability[field]) == PACK_INDEX_BINDING_KEYS
            and isinstance(capability[field].get("path"), str)
            and bool(capability[field]["path"])
            and "\x00" not in capability[field]["path"]
            and _is_sha256(capability[field].get("sha256"))
            and type(capability[field].get("bytes")) is int
            and capability[field]["bytes"] >= 0
            for field in opaque_fields
        )
        and capability.get("selection_lock") == dict(selection_lock)
        and capability.get("promotion_authorization") == dict(authorization)
        and capability.get("source_deep_validated_before_copy") is True
        and capability.get("source_paths_or_peer_outputs_authorized_in_child")
        is False
        and capability.get("target_reference_quality_identity_protocol_present")
        is False
        and capability.get("model_bytes_changed") is False
        and capability.get("commercial_or_confirmatory_claim_allowed") is False
    ):
        raise TargetSealedError("model-source capability drifted")
    for field, filename in (
        ("packed_checkpoint", "model_checkpoint.pt"),
        ("packed_scaler", "model_scaler.json"),
    ):
        _validate_relative_pack_member_binding(
            capability[field],
            unit_root=unit_root,
            relative_path=filename,
            label=f"model-source {field}",
        )
    if not (
        capability["source_checkpoint"]["sha256"]
        == successor_artifacts["model_checkpoint"]["sha256"]
        and capability["source_checkpoint"]["bytes"]
        == successor_artifacts["model_checkpoint"]["bytes"]
        and capability["source_scaler"]["sha256"]
        == successor_artifacts["model_scaler"]["sha256"]
        and capability["source_scaler"]["bytes"]
        == successor_artifacts["model_scaler"]["bytes"]
    ):
        raise TargetSealedError("packed model differs from opaque source evidence")

    npz_path = unit_root / "outer_predict_input.npz"
    try:
        with zipfile.ZipFile(npz_path, mode="r") as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise TargetSealedError(f"prediction-pack NPZ is invalid: {error}") from error
    expected_members = {f"{field}.npy" for field in OUTER_PREDICT_FIELDS}
    if not (
        len(members) == len(expected_members)
        and {member.filename for member in members} == expected_members
        and all(
            not member.is_dir()
            and not (member.flag_bits & 0x1)
            and member.file_size > 0
            and PurePosixPath(member.filename).name == member.filename
            for member in members
        )
    ):
        raise TargetSealedError("prediction-pack NPZ ten-field allowlist drifted")


def _validate_relative_pack_member_binding(
    row: object,
    *,
    unit_root: Path,
    relative_path: str,
    label: str,
) -> None:
    if not (
        isinstance(row, Mapping)
        and set(row) == PACK_INDEX_BINDING_KEYS
        and row.get("path") == relative_path
        and _is_sha256(row.get("sha256"))
        and type(row.get("bytes")) is int
        and row["bytes"] > 0
    ):
        raise TargetSealedError(f"{label} binding drifted")
    binding, _ = _read_file_binding(
        unit_root / relative_path,
        label=label,
        require_immutable=True,
    )
    if binding.sha256 != row["sha256"] or binding.bytes != row["bytes"]:
        raise TargetSealedError(f"{label} bytes drifted")


def _validate_promotion_training_unit_payload(
    *,
    pack_root: DirectoryBinding,
    relative: str,
    outer_fold: int,
    seed: int,
    authorization: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> None:
    unit_root = pack_root.path / relative
    _cache_binding, cache_raw = _read_file_binding(
        unit_root / "discovery_cache/manifest.json",
        label="promotion-training cache manifest",
        require_immutable=True,
    )
    cache = _decode_json_bytes(cache_raw, label="promotion-training cache manifest")
    if set(cache) != PROMOTION_CACHE_MANIFEST_KEYS:
        raise TargetSealedError("promotion-training cache manifest schema drifted")
    _validate_self_hash(cache, label="promotion-training cache manifest")
    if not (
        cache.get("schema_version") == 1
        and cache.get("classification")
        == "adaptive_v3r1_v8r4_nonouter_training_validation_pack"
        and cache.get("campaign_id") == CAMPAIGN_ID
        and cache.get("campaign_revision") == SCIENTIFIC_CAMPAIGN_REVISION
        and cache.get("format_version") == 1
        and cache.get("complete") is True
        and cache.get("outer_fold") == outer_fold
        and cache.get("partition") == "outer_excluded_training_validation"
        and cache.get("source_combined_cache_open_authorized_by_consumer") is False
        and cache.get("outer_test_rows_physically_present") is False
        and cache.get("outer_prediction_pack_absent") is True
        and cache.get("promotion_scope") == "promotion_training_pack"
        and cache.get("promotion_authorization") == dict(authorization)
    ):
        raise TargetSealedError(
            "promotion-training cache manifest capability drifted"
        )
    cache_outputs = cache.get("outputs")
    expected_cache_outputs = {
        "feature_names": "feature_names.json",
        "metadata": "metadata.csv",
        "node_features": "node_features.npy",
        "candidate_bpm": "candidate_bpm.npy",
        "candidate_mask": "candidate_mask.npy",
        "joint_radar_mask": "joint_radar_mask.npy",
        "local_to_global_cache_index": "local_to_global_cache_index.npy",
    }
    if not isinstance(cache_outputs, Mapping) or set(cache_outputs) != set(
        expected_cache_outputs
    ):
        raise TargetSealedError("promotion-training cache output schema drifted")
    for role, filename in expected_cache_outputs.items():
        row = cache_outputs.get(role)
        if not (
            isinstance(row, Mapping)
            and set(row) == {"filename", "sha256", "bytes"}
            and row.get("filename") == filename
        ):
            raise TargetSealedError(
                f"promotion-training cache output binding drifted: {role}"
            )
        _validate_relative_pack_member_binding(
            {
                "path": f"discovery_cache/{filename}",
                "sha256": row.get("sha256"),
                "bytes": row.get("bytes"),
            },
            unit_root=unit_root,
            relative_path=f"discovery_cache/{filename}",
            label=f"promotion-training cache output:{role}",
        )

    _partition_binding, partition_raw = _read_file_binding(
        unit_root / "PARTITION_MANIFEST.json",
        label="promotion-training partition manifest",
        require_immutable=True,
    )
    partition = _decode_json_bytes(
        partition_raw, label="promotion-training partition manifest"
    )
    if set(partition) != PROMOTION_PARTITION_MANIFEST_KEYS:
        raise TargetSealedError("promotion-training partition manifest schema drifted")
    _validate_self_hash(partition, label="promotion-training partition manifest")
    protected = partition.get("protected_outer_access")
    prediction = partition.get("preselection_prediction_boundary")
    serialization = partition.get("serialization")
    claim = partition.get("claim_boundary")
    if not (
        partition.get("schema_version") == 1
        and partition.get("classification")
        == "adaptive_v3r1_v8r4_sealed_nonouter_partition"
        and partition.get("campaign_id") == CAMPAIGN_ID
        and partition.get("campaign_revision") == SCIENTIFIC_CAMPAIGN_REVISION
        and partition.get("outer_fold") == outer_fold
        and partition.get("seed") == seed
        and type(partition.get("legacy_row_count")) is int
        and partition["legacy_row_count"] > 0
        and partition.get("promotion_scope") == "promotion_training_pack"
        and partition.get("promotion_authorization") == dict(authorization)
        and isinstance(protected, Mapping)
        and protected.get("forbidden_fields_emitted") is False
        and protected.get("outer_reference_decoded") is False
        and protected.get("outer_reference_validity_decoded") is False
        and protected.get("outer_identity_decoded") is False
        and protected.get("outer_protocol_decoded") is False
        and protected.get("outer_quality_decoded") is False
        and isinstance(prediction, Mapping)
        and prediction.get("outer_prediction_pack_absent") is True
        and prediction.get("outer_prediction_path_bound") is False
        and prediction.get("outer_prediction_values_materialized") is False
        and isinstance(serialization, Mapping)
        and serialization.get("object_arrays") is False
        and serialization.get("pickle") is False
        and serialization.get("outputs_mode") == "0444"
        and isinstance(claim, Mapping)
        and claim.get("outer_targets_opened") is False
    ):
        raise TargetSealedError(
            "promotion-training partition manifest capability drifted"
        )
    outputs = partition.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "discovery_cache_manifest",
        "discovery_proposer_stack",
        "discovery_local_to_global_map",
    }:
        raise TargetSealedError("promotion-training partition output schema drifted")
    for role, expected_path in {
        "discovery_cache_manifest": "discovery_cache/manifest.json",
        "discovery_proposer_stack": "discovery_proposer_stack.npz",
        "discovery_local_to_global_map": (
            "discovery_cache/local_to_global_cache_index.npy"
        ),
    }.items():
        _validate_relative_pack_member_binding(
            outputs.get(role),
            unit_root=unit_root,
            relative_path=expected_path,
            label=f"promotion-training partition output:{role}",
        )
    if not (
        outputs["discovery_cache_manifest"].get("sha256")
        == artifacts["cache_manifest"].get("sha256")
        and outputs["discovery_cache_manifest"].get("bytes")
        == artifacts["cache_manifest"].get("bytes")
        and outputs["discovery_proposer_stack"].get("sha256")
        == artifacts["proposer_stack"].get("sha256")
        and outputs["discovery_proposer_stack"].get("bytes")
        == artifacts["proposer_stack"].get("bytes")
    ):
        raise TargetSealedError("promotion-training nested artifact binding drifted")


def _validate_authorized_pack_inventory(
    *,
    pack_root: DirectoryBinding,
    index_binding: FileBinding,
    phase: str,
    unit_relatives: Sequence[str],
) -> None:
    expected_index = {
        "promotion_training": "V8R4_NONOUTER_TRAINING_INDEX.json",
        "promotion_prediction": PREDICTION_PACK_INDEX_FILENAME,
    }[phase]
    if index_binding.path.name != expected_index:
        raise TargetSealedError("authorized pack index filename drifted")
    expected_directories = {"units"}
    expected_files = {expected_index}
    if phase == "promotion_prediction":
        expected_files.add(MODEL_SOURCE_SHARD_SEAL_FILENAME)
    for relative in unit_relatives:
        unit = PurePosixPath(relative)
        expected_directories.update(
            str(PurePosixPath(*unit.parts[:length]))
            for length in range(1, len(unit.parts) + 1)
        )
        if phase == "promotion_prediction":
            expected_files.update(
                {
                    *(str(unit / name) for name in MODEL_BOUND_UNIT_FILES),
                }
            )
        else:
            expected_directories.add(str(unit / "discovery_cache"))
            expected_files.update(
                {
                    str(unit / "PARTITION_MANIFEST.json"),
                    str(unit / "discovery_proposer_stack.npz"),
                    *(str(unit / "discovery_cache" / name) for name in (
                        "manifest.json",
                        "feature_names.json",
                        "metadata.csv",
                        "node_features.npy",
                        "candidate_bpm.npy",
                        "candidate_mask.npy",
                        "joint_radar_mask.npy",
                        "local_to_global_cache_index.npy",
                    )),
                }
            )
    observed_directories: set[str] = set()
    observed_files: set[str] = set()
    for path in pack_root.path.rglob("*"):
        relative = path.relative_to(pack_root.path).as_posix()
        if path.is_dir():
            observed_directories.add(relative)
        else:
            observed_files.add(relative)
    if (
        observed_directories != expected_directories
        or observed_files != expected_files
    ):
        raise TargetSealedError("authorized pack exact inventory drifted")


def _validate_prediction_model_source_shard_seal(
    *,
    pack_root: DirectoryBinding,
    index_document: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> None:
    _validate_relative_pack_member_binding(
        index_document.get("model_source_shard_seal"),
        unit_root=pack_root.path,
        relative_path=MODEL_SOURCE_SHARD_SEAL_FILENAME,
        label="prediction model-source shard seal",
    )
    _seal_binding, raw = _read_file_binding(
        pack_root.path / MODEL_SOURCE_SHARD_SEAL_FILENAME,
        label="prediction model-source shard seal",
        require_immutable=True,
    )
    seal = _decode_json_bytes(raw, label="prediction model-source shard seal")
    if set(seal) != MODEL_SOURCE_SHARD_SEAL_KEYS:
        raise TargetSealedError("prediction model-source shard seal schema drifted")
    _validate_self_hash(seal, label="prediction model-source shard seal")
    units = seal.get("units")
    index_units = index_document.get("units")
    if not (
        seal.get("schema_version") == 1
        and seal.get("classification")
        == "adaptive_v3r1_v8r4a_model_source_shard_seal"
        and seal.get("campaign_id") == CAMPAIGN_ID
        and seal.get("campaign_revision") == SCIENTIFIC_CAMPAIGN_REVISION
        and seal.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and seal.get("outer_fold") == index_document.get("outer_fold")
        and seal.get("seeds") == list(SEEDS)
        and seal.get("selected_variant") == index_document.get("selected_variant")
        and seal.get("unit_count") == 3
        and seal.get("exact_three_seed_cover") is True
        and seal.get("selection_lock") == dict(selection_lock)
        and seal.get("promotion_authorization") == dict(authorization)
        and seal.get("target_or_prediction_values_present") is False
        and seal.get("source_paths_or_peer_outputs_authorized_in_child") is False
        and seal.get("commercial_or_confirmatory_claim_allowed") is False
        and isinstance(units, list)
        and len(units) == 3
        and isinstance(index_units, list)
        and len(index_units) == 3
    ):
        raise TargetSealedError("prediction model-source shard seal drifted")
    index_by_seed = {
        row.get("seed"): row for row in index_units if isinstance(row, Mapping)
    }
    observed: set[int] = set()
    for row in units:
        if not isinstance(row, Mapping) or set(row) != {
            "outer_fold",
            "seed",
            "source_kind",
            "scientific_signature_sha256",
            "row_count",
            "global_cache_index_sha256",
            "model_bound_prediction_pack_manifest",
            "model_checkpoint",
            "model_scaler",
            "model_source_capability",
        }:
            raise TargetSealedError("prediction model-source seal unit schema drifted")
        seed = row.get("seed")
        indexed = index_by_seed.get(seed)
        indexed_artifacts = (
            indexed.get("artifacts") if isinstance(indexed, Mapping) else None
        )
        if not (
            type(seed) is int
            and seed in SEEDS
            and seed not in observed
            and isinstance(indexed, Mapping)
            and isinstance(indexed_artifacts, Mapping)
            and row.get("outer_fold") == index_document.get("outer_fold")
            and row.get("source_kind") == indexed.get("source_kind")
            and row.get("scientific_signature_sha256")
            == indexed.get("scientific_signature_sha256")
            and row.get("row_count") == indexed.get("row_count")
            and row.get("global_cache_index_sha256")
            == indexed.get("global_cache_index_sha256")
            and row.get("model_bound_prediction_pack_manifest")
            == indexed_artifacts.get("model_bound_prediction_pack_manifest")
            and row.get("model_checkpoint")
            == indexed_artifacts.get("model_checkpoint")
            and row.get("model_scaler") == indexed_artifacts.get("model_scaler")
            and row.get("model_source_capability")
            == indexed_artifacts.get("model_source_capability")
        ):
            raise TargetSealedError("prediction model-source seal unit drifted")
        observed.add(seed)
    if observed != set(SEEDS):
        raise TargetSealedError("prediction model-source shard exact cover drifted")


def _validate_pack_index(
    *,
    raw: bytes,
    binding: FileBinding,
    pack_root: DirectoryBinding,
    phase: str,
    outer_fold: int,
    promotion_authorization_binding: FileBinding | None = None,
    selection_lock_binding: FileBinding | None = None,
) -> dict[str, Any]:
    document = _decode_json_bytes(raw, label="sealed-pack shard index")
    expected_keys = {
        "promotion_training": PROMOTION_TRAINING_PACK_INDEX_KEYS,
        "promotion_prediction": PREDICTION_PACK_INDEX_KEYS,
    }.get(phase, DISCOVERY_PACK_INDEX_KEYS)
    if set(document) != expected_keys:
        raise TargetSealedError("sealed-pack shard index schema drifted")
    _validate_self_hash(document, label="sealed-pack shard index")
    expected_classification = (
        PREDICTION_PACK_INDEX_CLASSIFICATION
        if phase == "promotion_prediction"
        else PACK_INDEX_CLASSIFICATION
    )
    is_prediction = phase == "promotion_prediction"
    if not (
        type(document.get("schema_version")) is int
        and document["schema_version"] == 1
        and document.get("classification") == expected_classification
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == "V8R4"
        and type(document.get("outer_fold")) is int
        and document["outer_fold"] == outer_fold
        and document.get("seeds") == list(SEEDS)
        and type(document.get("unit_count")) is int
        and document["unit_count"] == 3
        and type(document.get("completed_units")) is int
        and document["completed_units"] == 3
        and document.get("status") == "complete"
        and document.get("outer_test_opened") is False
        and document.get(
            "combined_target_bearing_cache_consumer_access_authorized"
        )
        is False
        and (
            document.get("physical_target_free_input_and_model_packs") is True
            if is_prediction
            else document.get("physical_nonouter_training_packs") is True
        )
        and (
            document.get("source_paths_or_peer_outputs_authorized_in_child")
            is False
            and document.get("infrastructure_revision")
            == INFRASTRUCTURE_REVISION
            and isinstance(document.get("selected_variant"), str)
            and bool(document["selected_variant"])
            if is_prediction
            else document.get("outer_prediction_packs_absent") is True
        )
        and document.get("cross_outer_shard_mounted") is False
    ):
        raise TargetSealedError("sealed-pack shard index identity/capability drifted")
    if phase == "promotion_training" and document.get("promotion_scope") != (
        "promotion_training_pack"
    ):
        raise TargetSealedError("promotion-training pack scope drifted")
    if phase in {"promotion_training", "promotion_prediction"}:
        if promotion_authorization_binding is None:
            raise TargetSealedError("promotion pack lacks its authorization capability")
        expected_authorization = {
            "path": str(promotion_authorization_binding.path),
            "sha256": promotion_authorization_binding.sha256,
            "bytes": promotion_authorization_binding.bytes,
        }
        if document.get("promotion_authorization") != expected_authorization:
            raise TargetSealedError("promotion pack authorization binding drifted")
        if is_prediction:
            if selection_lock_binding is None:
                raise TargetSealedError(
                    "model-bound prediction pack lacks its selection capability"
                )
            expected_selection = {
                "path": str(selection_lock_binding.path),
                "sha256": selection_lock_binding.sha256,
                "bytes": selection_lock_binding.bytes,
            }
    elif "promotion_authorization" in document:
        raise TargetSealedError("discovery pack unexpectedly carries promotion authority")
    units = document.get("units")
    if not isinstance(units, list) or len(units) != 3:
        raise TargetSealedError("sealed-pack shard index is not an exact three-seed cover")
    seen: set[tuple[int, int]] = set()
    unit_relatives: list[str] = []
    for row in units:
        expected_unit_keys = (
            PREDICTION_PACK_INDEX_UNIT_KEYS
            if is_prediction
            else PACK_INDEX_UNIT_KEYS
        )
        if not isinstance(row, dict) or set(row) != expected_unit_keys:
            raise TargetSealedError("sealed-pack unit schema drifted")
        key = (row.get("outer_fold"), row.get("seed"))
        if (
            type(key[0]) is not int
            or type(key[1]) is not int
            or key[0] != outer_fold
            or key[1] not in SEEDS
            or key in seen
        ):
            raise TargetSealedError("sealed-pack unit identity/cover drifted")
        seen.add((int(key[0]), int(key[1])))
        relative = row.get("relative_path")
        artifacts = row.get("artifacts")
        expected_unit_relative = f"units/outer_{outer_fold}_seed_{int(key[1])}"
        if (
            not isinstance(relative, str)
            or not relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or relative != expected_unit_relative
            or not isinstance(artifacts, dict)
            or set(artifacts)
            != (
                PREDICTION_PACK_INDEX_ARTIFACT_KEYS
                if is_prediction
                else PACK_INDEX_ARTIFACT_KEYS
            )
        ):
            raise TargetSealedError("sealed-pack unit path/artifact map drifted")
        if is_prediction and not (
            _is_sha256(row.get("scientific_signature_sha256"))
            and type(row.get("row_count")) is int
            and row["row_count"] > 0
            and _is_sha256(row.get("global_cache_index_sha256"))
            and row.get("source_kind")
            in {"local_training", "discovery", "discovery_pointer"}
        ):
            raise TargetSealedError("model-bound prediction unit evidence drifted")
        unit_relatives.append(relative)
        expected_artifacts = (
            {
                "prediction_pack_manifest": PurePosixPath(relative)
                / "OUTER_PREDICTION_PACK_MANIFEST.json",
                "model_bound_prediction_pack_manifest": PurePosixPath(relative)
                / MODEL_BOUND_PREDICTION_MANIFEST_FILENAME,
                "outer_predict_input": PurePosixPath(relative)
                / "outer_predict_input.npz",
                "model_checkpoint": PurePosixPath(relative)
                / "model_checkpoint.pt",
                "model_scaler": PurePosixPath(relative) / "model_scaler.json",
                "model_source_capability": PurePosixPath(relative)
                / MODEL_SOURCE_CAPABILITY_FILENAME,
            }
            if is_prediction
            else {
                "cache_manifest": PurePosixPath(relative)
                / "discovery_cache/manifest.json",
                "proposer_stack": PurePosixPath(relative)
                / "discovery_proposer_stack.npz",
                "partition_manifest": PurePosixPath(relative)
                / "PARTITION_MANIFEST.json",
            }
        )
        for artifact_role, expected_relative in expected_artifacts.items():
            artifact = artifacts.get(artifact_role)
            if (
                not isinstance(artifact, dict)
                or set(artifact) != PACK_INDEX_BINDING_KEYS
                or artifact.get("path") != str(expected_relative)
                or not _is_sha256(artifact.get("sha256"))
                or type(artifact.get("bytes")) is not int
                or artifact["bytes"] <= 0
            ):
                raise TargetSealedError(
                    f"sealed-pack {artifact_role} binding drifted"
                )
            member = pack_root.path / Path(expected_relative.as_posix())
            member_binding, _ = _read_file_binding(
                member,
                label=f"sealed-pack {artifact_role}",
                require_immutable=True,
            )
            if (
                member_binding.sha256 != artifact["sha256"]
                or member_binding.bytes != artifact["bytes"]
            ):
                raise TargetSealedError(
                    f"sealed-pack {artifact_role} bytes drifted"
                )
        if is_prediction:
            _validate_prediction_unit_payload(
                pack_root=pack_root,
                relative=relative,
                outer_fold=outer_fold,
                seed=int(key[1]),
                selected_variant=str(document["selected_variant"]),
                selection_lock=expected_selection,
                authorization=document["promotion_authorization"],
                artifacts=artifacts,
                index_row=row,
            )
        elif phase == "promotion_training":
            _validate_promotion_training_unit_payload(
                pack_root=pack_root,
                relative=relative,
                outer_fold=outer_fold,
                seed=int(key[1]),
                authorization=document["promotion_authorization"],
                artifacts=artifacts,
            )
    if seen != {(outer_fold, seed) for seed in SEEDS}:
        raise TargetSealedError("sealed-pack shard index exact cover drifted")
    if binding.path.parent != pack_root.path:
        raise TargetSealedError("sealed-pack index must be a direct shard-root member")
    if is_prediction:
        _validate_prediction_model_source_shard_seal(
            pack_root=pack_root,
            index_document=document,
            selection_lock=expected_selection,
            authorization=document["promotion_authorization"],
        )
    if phase in {"promotion_training", "promotion_prediction"}:
        _validate_authorized_pack_inventory(
            pack_root=pack_root,
            index_binding=binding,
            phase=phase,
            unit_relatives=unit_relatives,
        )
    return document


def _parse_role_paths(
    values: Sequence[str], *, label: str
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        role, separator, path = raw.partition("=")
        if (
            separator != "="
            or not role
            or not role.replace("_", "").isalnum()
            or role in result
            or not path
        ):
            raise TargetSealedError(f"invalid or duplicate {label}: {raw}")
        result[role] = Path(path)
    return result


@dataclass(frozen=True, slots=True)
class RuntimeRequest:
    project_root: Path
    phase: str
    outer_fold: int | None
    pack_root: Path | None
    pack_index: Path | None
    governance_files: Mapping[str, Path]
    writable_roots: Mapping[str, Path]
    denied_canaries: Mapping[str, Path]
    capability_receipt: Path
    interpreter: Path
    venv_root: Path
    python_runtime_root: Path
    command: tuple[str, ...]
    bwrap_binary: Path = BWRAP_BINARY
    cuda_runtime_roots: tuple[Path, ...] = ()
    cuda_devices: tuple[Path, ...] = ()
    propagated_environment: Mapping[str, str] = field(default_factory=dict)
    production: bool = True


@dataclass(slots=True)
class PreparedRuntime(AbstractContextManager["PreparedRuntime"]):
    request: RuntimeRequest
    descriptors: list[int]
    mount_entries: list[dict[str, Any]]
    directory_bindings: dict[str, DirectoryBinding]
    governance_bindings: dict[str, FileBinding]
    source_snapshot_bindings: dict[Path, FileBinding]
    writable_bindings: dict[str, DirectoryBinding]
    gpu_state_root_binding: DirectoryBinding
    gpu_state_root_descriptor: int
    gpu_state_child_descriptors: dict[str, int]
    pack_binding: DirectoryBinding | None
    pack_index_binding: FileBinding | None
    migration_receipt_binding: FileBinding
    prelaunch_state: LiveStateSnapshot
    prelaunch_output_inventory: list[dict[str, Any]]
    child_environment: dict[str, str]
    mount_spec_sha256: str
    command_sha256: str
    receipt: dict[str, Any]
    bwrap_command: list[str]
    spec_fd: int

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.descriptors.clear()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _validate_request_shape(request: RuntimeRequest) -> tuple[Path, Path, Path]:
    if request.phase not in PHASES:
        raise TargetSealedError("invalid target-sealed phase")
    project_root = _canonical_existing(request.project_root, label="project root")
    if not project_root.is_dir():
        raise TargetSealedError("project root is not a directory")
    if set(request.writable_roots) != WRITABLE_ROLES:
        raise TargetSealedError("writable role set is not exact")
    for role, relative in GPU_STATE_DIRECTORY_RELATIVE_PATHS.items():
        expected = project_root / relative
        observed = _absolute_lexical(
            request.writable_roots[role], label=f"writable:{role}"
        )
        if observed != expected:
            raise TargetSealedError(
                f"writable:{role} is not its canonical V8R4A state directory"
            )
    canary_roles = set(request.denied_canaries)
    if not MANDATORY_DENIED_CANARY_ROLES <= canary_roles or any(
        role not in MANDATORY_DENIED_CANARY_ROLES
        and not role.startswith(("other_pack_", "other_output_"))
        for role in canary_roles
    ):
        raise TargetSealedError("denied-canary role set is incomplete or invalid")
    expected_superseded_canaries = {
        "other_output_root": project_root / OTHER_OUTPUT_ROOT_RELATIVE,
        "superseded_v8r4a_lifecycle_root": (
            project_root / SUPERSEDED_V8R4A_LIFECYCLE_ROOT_RELATIVE
        ),
        "superseded_v8r4a_output_root": (
            project_root / SUPERSEDED_V8R4A_OUTPUT_ROOT_RELATIVE
        ),
        "superseded_v8r4a_contract1_lifecycle_root": (
            project_root / SUPERSEDED_V8R4A_CONTRACT1_LIFECYCLE_ROOT_RELATIVE
        ),
        "superseded_v8r4a_contract1_output_root": (
            project_root / SUPERSEDED_V8R4A_CONTRACT1_OUTPUT_ROOT_RELATIVE
        ),
        "superseded_v8r4a_rootbind1_lifecycle_root": (
            project_root / SUPERSEDED_V8R4A_ROOTBIND1_LIFECYCLE_ROOT_RELATIVE
        ),
        "superseded_v8r4a_rootbind1_output_root": (
            project_root / SUPERSEDED_V8R4A_ROOTBIND1_OUTPUT_ROOT_RELATIVE
        ),
    }
    for role, expected in expected_superseded_canaries.items():
        observed = _absolute_lexical(
            request.denied_canaries[role], label=f"{role} denied canary"
        )
        if observed != expected:
            raise TargetSealedError(f"{role} is not its exact historical output root")
    if request.phase in {"discovery_aggregation", "promotion_aggregation"}:
        if request.outer_fold is not None or request.pack_root is not None or request.pack_index is not None:
            raise TargetSealedError("aggregation must be pack-free and outer-fold-free")
    else:
        allowed_outer = {
            "efficiency_benchmark": frozenset({3}),
            "discovery": DISCOVERY_OUTER_FOLDS,
            "promotion_training": frozenset({0, 1, 2, 5}),
            "promotion_prediction": ALL_OUTER_FOLDS,
        }[request.phase]
        if request.outer_fold not in allowed_outer:
            raise TargetSealedError("phase outer fold is outside its authorized set")
        if request.pack_root is None or request.pack_index is None:
            raise TargetSealedError("phase requires one outer-fold shard root/index")
    if not request.command or any(
        not isinstance(part, str) or not part or "\x00" in part
        for part in request.command
    ):
        raise TargetSealedError("child command is invalid")
    interpreter_lexical = _absolute_lexical(request.interpreter, label="interpreter")
    try:
        interpreter_real = interpreter_lexical.resolve(strict=True)
    except OSError as error:
        raise TargetSealedError(f"interpreter is unavailable: {error}") from error
    interpreter_binding, _ = _read_file_binding(
        interpreter_real, label="Python interpreter", require_immutable=False
    )
    if not os.access(interpreter_real, os.X_OK) or interpreter_binding.mode & 0o111 == 0:
        raise TargetSealedError("Python interpreter is not executable")
    if Path(request.command[0]) != interpreter_lexical:
        raise TargetSealedError("child command must use the exact mounted interpreter")
    if len(request.command) < 2:
        raise TargetSealedError("child command lacks its canonical entry script")
    entry = _canonical_existing(Path(request.command[1]), label="campaign entry script")
    scripts_root = project_root / "scripts"
    if (
        entry.parent != scripts_root
        or entry.name not in ENTRY_SCRIPT_BY_PHASE[request.phase]
    ):
        raise TargetSealedError("child entry script is not canonical for phase")
    expected_governance = _governance_roles_for(
        phase=request.phase, entry_name=entry.name
    )
    if set(request.governance_files) != expected_governance:
        raise TargetSealedError("governance role set is not exact for phase/entry")
    expected_output = project_root / _canonical_output_relative(
        phase=request.phase,
        outer_fold=request.outer_fold,
        entry_name=entry.name,
    )
    observed_output = _absolute_lexical(
        request.writable_roots["output"], label="writable:output"
    )
    if observed_output != expected_output:
        raise TargetSealedError(
            "writable:output is not canonical for phase/outer/entry"
        )
    expected_lifecycle = project_root / _canonical_lifecycle_relative(
        phase=request.phase,
        outer_fold=request.outer_fold,
        entry_name=entry.name,
    )
    observed_lifecycle = _absolute_lexical(
        request.writable_roots["lifecycle"], label="writable:lifecycle"
    )
    if observed_lifecycle != expected_lifecycle:
        raise TargetSealedError(
            "writable:lifecycle is not canonical for phase/outer/entry"
        )
    if ADMITTED_CHILD_FD_ENV in request.propagated_environment:
        raise TargetSealedError("outer runtime must not synthesize an admitted-child fd")
    if any(name in FORBIDDEN_ENV_NAMES for name in request.propagated_environment):
        raise TargetSealedError("forbidden environment requested")
    if not set(request.propagated_environment) <= SAFE_PROPAGATED_ENV:
        raise TargetSealedError("environment allowlist drifted")
    for name, value in request.propagated_environment.items():
        if not isinstance(value, str) or "\x00" in value or "\n" in value:
            raise TargetSealedError(f"invalid environment value: {name}")
    return project_root, interpreter_lexical, interpreter_real


def _open_special_binding(path: Path, *, label: str) -> tuple[int, dict[str, Any]]:
    resolved = _canonical_existing(path, label=label)
    flags = getattr(os, "O_PATH", os.O_RDONLY)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    status = os.fstat(descriptor)
    named = os.stat(resolved, follow_symlinks=False)
    if (
        not stat.S_ISCHR(status.st_mode)
        or (status.st_dev, status.st_ino, status.st_rdev)
        != (named.st_dev, named.st_ino, named.st_rdev)
    ):
        os.close(descriptor)
        raise TargetSealedError(f"{label} is not a stable character device")
    return descriptor, {
        "path": str(resolved),
        "st_dev": status.st_dev,
        "st_ino": status.st_ino,
        "st_rdev": status.st_rdev,
        "mode": f"{_mode(status):04o}",
    }


def _validate_bwrap_binary(
    path: Path,
    *,
    version_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> FileBinding:
    binding, _ = _read_file_binding(
        path, label="bubblewrap binary", require_immutable=False
    )
    if binding.path != BWRAP_BINARY:
        raise TargetSealedError("bubblewrap path must be /usr/bin/bwrap")
    try:
        result = version_runner(
            [str(binding.path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TargetSealedError(f"cannot execute bubblewrap version probe: {error}") from error
    if result.returncode != 0 or result.stdout.strip() != BWRAP_VERSION or result.stderr:
        raise TargetSealedError("bubblewrap must be exact version 0.11.1")
    refreshed, _ = _read_file_binding(
        binding.path, label="bubblewrap binary", require_immutable=False
    )
    if refreshed != binding:
        raise TargetSealedError("bubblewrap binary changed across version probe")
    return binding


GOVERNANCE_TOP_LEVEL_KEYS: Final[Mapping[str, frozenset[str]]] = {
    "implementation_test_receipt": frozenset(
        {
            "all_tests_passed",
            "authorization_generation",
            "campaign_id",
            "classification",
            "commercial_claim_authorized",
            "content_sha256",
            "correction_authorization",
            "created_utc",
            "gpu_accessed",
            "gpu_state_migration_receipt",
            "implementation_files",
            "infrastructure_correction_authorization",
            "infrastructure_revision",
            "kill_safe_correction_authorization",
            "kill_safe_failure_diagnostic",
            "open_lifecycle_recovery_correction_authorization",
            "open_lifecycle_recovery_failure_diagnostic",
            "execution_closure_correction_authorization",
            "execution_closure_failure_diagnostic",
            "migration_source_succession_correction_authorization",
            "migration_source_succession_failure_diagnostic",
            "fd_closure_correction_authorization",
            "fd_closure_failure_diagnostic",
            "canary_boundary_correction_authorization",
            "canary_boundary_failure_diagnostic",
            "frozen_contract_encoding_correction_authorization",
            "frozen_contract_encoding_failure_diagnostic",
            "gpu_state_parent_bind_correction_authorization",
            "gpu_state_parent_bind_failure_diagnostic",
            "admitted_context_correction_authorization",
            "admitted_context_failure_diagnostic",
            "return_code",
            "runtime_state_after",
            "runtime_state_before",
            "schema_version",
            "scientific_campaign_revision",
            "source_closure_correction_authorization",
            "source_closure_dependency_authorization",
            "source_closure_failure_diagnostic",
            "stdout_bytes",
            "stdout_is_complete",
            "stdout_sha256",
            "stdout_tail",
            "target_or_outer_reference_accessed",
            "test_paths",
            "command",
        }
    ),
    "source_snapshot": frozenset(
        {
            "adaptive_retrospective_only",
            "authorization_generation",
            "campaign_id",
            "classification",
            "commercial_claim_authorized",
            "content_sha256",
            "contract_file_sha256",
            "correction_authorization",
            "created_utc",
            "entry_evidence",
            "environment",
            "gpu_state_migration_receipt",
            "historical_v8r3_parent",
            "implementation_files",
            "implementation_test_receipt",
            "infrastructure_correction_authorization",
            "infrastructure_revision",
            "kill_safe_correction_authorization",
            "kill_safe_failure_diagnostic",
            "open_lifecycle_recovery_correction_authorization",
            "open_lifecycle_recovery_failure_diagnostic",
            "execution_closure_correction_authorization",
            "execution_closure_failure_diagnostic",
            "migration_source_succession_correction_authorization",
            "migration_source_succession_failure_diagnostic",
            "fd_closure_correction_authorization",
            "fd_closure_failure_diagnostic",
            "canary_boundary_correction_authorization",
            "canary_boundary_failure_diagnostic",
            "frozen_contract_encoding_correction_authorization",
            "frozen_contract_encoding_failure_diagnostic",
            "gpu_state_parent_bind_correction_authorization",
            "gpu_state_parent_bind_failure_diagnostic",
            "admitted_context_correction_authorization",
            "admitted_context_failure_diagnostic",
            "read_only_ancestry",
            "runtime_state_at_snapshot",
            "schema_version",
            "scientific_campaign_revision",
            "sealed_discovery_pack_indexes",
            "source_closure_correction_authorization",
            "source_closure_dependency_authorization",
            "source_closure_failure_diagnostic",
            "training_authorized_by_snapshot_alone",
        }
    ),
    "active_authorization": frozenset(
        {
            "adaptive_retrospective_only",
            "admitted_child_scope",
            "authorization_generation",
            "campaign_id",
            "canonical_gpu_state_paths",
            "classification",
            "commercial_claim_authorized",
            "content_sha256",
            "correction_authorization",
            "created_utc",
            "discovery_requires_passing_efficiency_benchmark",
            "discovery_scope",
            "efficiency_benchmark_authorized",
            "efficiency_benchmark_scope",
            "gpu_budget_protocol",
            "gpu_hours_hard",
            "gpu_state_migration_receipt",
            "implementation_test_receipt",
            "infrastructure_correction_authorization",
            "infrastructure_revision",
            "kill_safe_correction_authorization",
            "kill_safe_failure_diagnostic",
            "open_lifecycle_recovery_correction_authorization",
            "open_lifecycle_recovery_failure_diagnostic",
            "execution_closure_correction_authorization",
            "execution_closure_failure_diagnostic",
            "migration_source_succession_correction_authorization",
            "migration_source_succession_failure_diagnostic",
            "fd_closure_correction_authorization",
            "fd_closure_failure_diagnostic",
            "canary_boundary_correction_authorization",
            "canary_boundary_failure_diagnostic",
            "frozen_contract_encoding_correction_authorization",
            "frozen_contract_encoding_failure_diagnostic",
            "gpu_state_parent_bind_correction_authorization",
            "gpu_state_parent_bind_failure_diagnostic",
            "admitted_context_correction_authorization",
            "admitted_context_failure_diagnostic",
            "maximum_parallel_gpu_training_jobs",
            "outer_fold_numeric_reference_authorized",
            "production_target_sealed_runtime_authorized",
            "promotion_authorized",
            "promotion_reuse_scope",
            "runtime_ledger_prefixes",
            "schema_version",
            "scientific_campaign_revision",
            "snapshot_content_sha256",
            "source_closure_correction_authorization",
            "source_closure_dependency_authorization",
            "source_closure_failure_diagnostic",
            "source_snapshot",
            "status",
            "training_authorized",
        }
    ),
    "correction_authorization": frozenset(
        {
            "authorized_modifications",
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "created_utc",
            "diagnostic",
            "forbidden_changes",
            "mandatory_invariants",
            "pack_conversion_boundary",
            "parent_v8r3_chain",
            "quarantine_owner_receipt",
            "required_reauthorization",
            "schema_version",
        }
    ),
    "infrastructure_correction_authorization": frozenset(
        {
            "authority_basis",
            "authorized_modifications",
            "campaign_id",
            "canonical_gpu_state_layout",
            "claim_boundary",
            "classification",
            "content_sha256",
            "created_utc",
            "diagnostic",
            "forbidden_changes",
            "frozen_implementation_bindings",
            "historical_evidence_acceptance",
            "infrastructure_revision",
            "mandatory_invariants",
            "migration_protocol",
            "required_reauthorization",
            "schema_version",
            "scientific_campaign_revision",
        }
    ),
    "failure_diagnostic": frozenset(
        {
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "created_utc",
            "decision",
            "failed_postcondition",
            "first_discovery_unit",
            "immutable_parent_v8r3_chain",
            "outer_capability_breach",
            "quarantine_owner_receipt",
            "required_v8r4_boundary",
            "runtime_ledgers_after_failure",
            "schema_version",
        }
    ),
    "infrastructure_failure_diagnostic": frozenset(
        {
            "actual_bubblewrap_reproduction",
            "campaign_id",
            "classification",
            "content_sha256",
            "created_utc",
            "decision",
            "failure_boundary",
            "infrastructure_revision",
            "original_gpu_state",
            "required_correction",
            "schema_version",
            "scientific_campaign_revision",
            "source_bindings",
            "test_evidence",
        }
    ),
    "source_closure_correction_authorization": frozenset(
        {
            "authority_basis",
            "authorized_modifications",
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "created_utc",
            "forbidden_changes",
            "infrastructure_revision",
            "mandatory_invariants",
            "newly_governed_unchanged_dependencies",
            "required_reauthorization",
            "schema_version",
            "scientific_campaign_revision",
        }
    ),
    "source_closure_dependency_authorization": frozenset(
        {
            "authority_basis",
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "created_utc",
            "dependency_derivation",
            "forbidden_changes",
            "infrastructure_revision",
            "newly_governed_unchanged_dependencies",
            "required_reauthorization",
            "schema_version",
            "scientific_campaign_revision",
        }
    ),
    "source_closure_failure_diagnostic": frozenset(
        {
            "affected_boundary",
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "deterministic_cpu_regression_observation",
            "evidence",
            "failure_modes",
            "infrastructure_revision",
            "observed_utc",
            "required_correction",
            "schema_version",
            "scientific_campaign_revision",
            "status",
        }
    ),
    "kill_safe_correction_authorization": frozenset(
        {
            "authority_basis",
            "authorized_modifications",
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "created_utc",
            "forbidden_changes",
            "infrastructure_revision",
            "mandatory_invariants",
            "required_reauthorization",
            "schema_version",
            "scientific_campaign_revision",
        }
    ),
    "kill_safe_failure_diagnostic": frozenset(
        {
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "evidence",
            "failure_modes",
            "infrastructure_revision",
            "observed_utc",
            "required_correction",
            "schema_version",
            "scientific_campaign_revision",
            "status",
        }
    ),
    "open_lifecycle_recovery_correction_authorization": frozenset(
        {
            "authority_basis",
            "authorized_modifications",
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "created_utc",
            "forbidden_changes",
            "infrastructure_revision",
            "mandatory_invariants",
            "required_reauthorization",
            "schema_version",
            "scientific_campaign_revision",
        }
    ),
    "open_lifecycle_recovery_failure_diagnostic": frozenset(
        {
            "affected_boundary",
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "evidence",
            "failure_modes",
            "infrastructure_revision",
            "observed_utc",
            "required_correction",
            "schema_version",
            "scientific_campaign_revision",
            "status",
        }
    ),
    "execution_closure_correction_authorization": frozenset(
        {
            "authority_basis",
            "authorized_modifications",
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "created_utc",
            "forbidden_changes",
            "infrastructure_revision",
            "mandatory_invariants",
            "required_reauthorization",
            "schema_version",
            "scientific_campaign_revision",
        }
    ),
    "execution_closure_failure_diagnostic": frozenset(
        {
            "affected_boundary",
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "evidence",
            "failure_modes",
            "infrastructure_revision",
            "observed_utc",
            "required_correction",
            "schema_version",
            "scientific_campaign_revision",
            "status",
        }
    ),
    "migration_source_succession_correction_authorization": frozenset(
        {
            "authority_basis",
            "authorized_modifications",
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "created_utc",
            "forbidden_changes",
            "infrastructure_revision",
            "mandatory_invariants",
            "required_reauthorization",
            "schema_version",
            "scientific_campaign_revision",
        }
    ),
    "migration_source_succession_failure_diagnostic": frozenset(
        {
            "campaign_id",
            "claim_boundary",
            "classification",
            "content_sha256",
            "created_utc",
            "demonstrated_failure",
            "immutable_facts",
            "required_correction",
            "root_cause",
            "schema_version",
            "scientific_campaign_revision",
            "status",
        }
    ),
    "fd_closure_correction_authorization": frozenset(
        {
            "schema_version",
            "classification",
            "campaign_id",
            "scientific_campaign_revision",
            "infrastructure_revision",
            "created_utc",
            "authority_basis",
            "authorized_modifications",
            "mandatory_invariants",
            "forbidden_changes",
            "required_reauthorization",
            "claim_boundary",
            "content_sha256",
        }
    ),
    "fd_closure_failure_diagnostic": frozenset(
        {
            "schema_version",
            "classification",
            "campaign_id",
            "scientific_campaign_revision",
            "infrastructure_revision",
            "observed_utc",
            "status",
            "failed_attempt",
            "ledger_evidence",
            "root_cause",
            "reproduction",
            "superseded_pretrain_chain",
            "required_correction",
            "claim_boundary",
            "content_sha256",
        }
    ),
    "canary_boundary_correction_authorization": frozenset(
        {
            "schema_version",
            "classification",
            "campaign_id",
            "scientific_campaign_revision",
            "infrastructure_revision",
            "created_utc",
            "authority_basis",
            "authorized_modifications",
            "mandatory_invariants",
            "forbidden_changes",
            "required_reauthorization",
            "claim_boundary",
            "content_sha256",
        }
    ),
    "canary_boundary_failure_diagnostic": frozenset(
        {
            "schema_version",
            "classification",
            "campaign_id",
            "scientific_campaign_revision",
            "infrastructure_revision",
            "observed_utc",
            "status",
            "failed_attempt",
            "ledger_evidence",
            "root_cause",
            "reproduction",
            "superseded_pretrain_chain",
            "required_correction",
            "claim_boundary",
            "content_sha256",
        }
    ),
    "frozen_contract_encoding_correction_authorization": frozenset(
        {
            "schema_version",
            "classification",
            "campaign_id",
            "scientific_campaign_revision",
            "infrastructure_revision",
            "created_utc",
            "authority_basis",
            "authorized_modifications",
            "mandatory_invariants",
            "forbidden_changes",
            "required_reauthorization",
            "claim_boundary",
            "content_sha256",
        }
    ),
    "frozen_contract_encoding_failure_diagnostic": frozenset(
        {
            "schema_version",
            "classification",
            "campaign_id",
            "scientific_campaign_revision",
            "infrastructure_revision",
            "observed_utc",
            "status",
            "failed_attempt",
            "frozen_contract_evidence",
            "root_cause",
            "immutable_failure_receipts",
            "failed_namespace_inventory",
            "ledger_evidence",
            "required_correction",
            "claim_boundary",
            "content_sha256",
        }
    ),
    "gpu_state_parent_bind_correction_authorization": frozenset(
        {
            "schema_version",
            "classification",
            "campaign_id",
            "scientific_campaign_revision",
            "infrastructure_revision",
            "created_utc",
            "authority_basis",
            "authorized_modifications",
            "mandatory_invariants",
            "forbidden_changes",
            "required_reauthorization",
            "claim_boundary",
            "content_sha256",
        }
    ),
    "gpu_state_parent_bind_failure_diagnostic": frozenset(
        {
            "schema_version",
            "classification",
            "campaign_id",
            "scientific_campaign_revision",
            "infrastructure_revision",
            "observed_utc",
            "status",
            "failed_attempt",
            "trusted_host_gpu_state_root",
            "failed_mount_topology",
            "independent_bubblewrap_reproduction",
            "root_cause",
            "immutable_failure_receipts",
            "failed_namespace_inventory",
            "ledger_evidence",
            "required_correction",
            "claim_boundary",
            "content_sha256",
        }
    ),
    "admitted_context_correction_authorization": frozenset(
        {
            "schema_version",
            "classification",
            "campaign_id",
            "scientific_campaign_revision",
            "infrastructure_revision",
            "created_utc",
            "authority_basis",
            "authorized_modifications",
            "mandatory_invariants",
            "forbidden_changes",
            "required_reauthorization",
            "claim_boundary",
            "content_sha256",
        }
    ),
    "admitted_context_failure_diagnostic": frozenset(
        {
            "schema_version",
            "classification",
            "campaign_id",
            "scientific_campaign_revision",
            "infrastructure_revision",
            "status",
            "observed_utc",
            "failed_attempt",
            "immutable_failure_receipts",
            "ledger_evidence",
            "root_cause",
            "required_correction",
            "claim_boundary",
            "content_sha256",
        }
    ),
    "gpu_state_migration_receipt": frozenset(
        {
            "authority",
            "campaign_id",
            "classification",
            "content_sha256",
            "created_utc",
            "directory_inventory",
            "historical_evidence",
            "infrastructure_revision",
            "lifecycle_state",
            "migrated_state",
            "original_state",
            "path_inode_lineage",
            "prefix_replay",
            "production_runtime_authorized",
            "schema_version",
            "scientific_campaign_revision",
        }
    ),
    "campaign_contract": frozenset(
        {
            "architecture",
            "campaign_id",
            "classification",
            "content_sha256",
            "created_date",
            "discovery",
            "evaluation",
            "failure_and_iteration_policy",
            "feature_contract",
            "governance",
            "immutable_inputs",
            "implementation_authorization",
            "objective",
            "population_and_splits",
            "promotion",
            "resource_budget",
            "schema_version",
            "status",
            "training",
        }
    ),
}


def _validate_governance_json(raw: bytes, *, role: str) -> dict[str, Any]:
    document = _decode_json_bytes(raw, label=f"{role} governance file")
    expected_keys = GOVERNANCE_TOP_LEVEL_KEYS.get(role)
    if expected_keys is not None and set(document) != expected_keys:
        raise TargetSealedError(f"{role} governance exact schema drifted")
    if "content_sha256" not in document:
        raise TargetSealedError(f"{role} governance self-hash is absent")
    _validate_self_hash(document, label=f"{role} governance file")
    if document.get("campaign_id") != CAMPAIGN_ID:
        raise TargetSealedError(f"{role} governance campaign drifted")
    return document


def _require_exact_governance_binding(
    row: object,
    *,
    binding: FileBinding,
    project_root: Path,
    label: str,
) -> None:
    if not isinstance(row, Mapping):
        raise TargetSealedError(f"{label} is not a file binding")
    try:
        relative = binding.path.relative_to(project_root).as_posix()
    except ValueError as error:
        raise TargetSealedError(f"{label} binding escapes project root") from error
    if set(row) != {"path", "sha256", "bytes", "mode", "nlink", "st_dev", "st_ino"}:
        raise TargetSealedError(f"{label} exact binding schema drifted")
    if not (
        row.get("path") == relative
        and row.get("sha256") == binding.sha256
        and row.get("bytes") == binding.bytes
        and row.get("mode") == "0444"
        and row.get("nlink") == 1
        and row.get("st_dev") == binding.st_dev
        and row.get("st_ino") == binding.st_ino
    ):
        raise TargetSealedError(f"{label} exact binding drifted")


def _require_snapshot_file_binding(
    row: object,
    *,
    binding: FileBinding,
    project_root: Path,
    label: str,
) -> None:
    if not isinstance(row, Mapping) or set(row) != {
        "path",
        "file_sha256",
        "size_bytes",
        "mode",
    }:
        raise TargetSealedError(f"{label} snapshot binding schema drifted")
    if not (
        row.get("path") == binding.path.relative_to(project_root).as_posix()
        and row.get("file_sha256") == binding.sha256
        and row.get("size_bytes") == binding.bytes
        and row.get("mode") == 0o444
    ):
        raise TargetSealedError(f"{label} snapshot binding drifted")


def _require_authority_legacy_binding(
    row: object,
    *,
    binding: FileBinding,
    document: Mapping[str, Any],
    project_root: Path,
    label: str,
) -> None:
    if not isinstance(row, Mapping) or set(row) != {
        "path",
        "file_sha256",
        "bytes",
        "content_sha256",
    }:
        raise TargetSealedError(f"{label} binding schema drifted")
    if not (
        row.get("path") == binding.path.relative_to(project_root).as_posix()
        and row.get("file_sha256") == binding.sha256
        and row.get("bytes") == binding.bytes
        and row.get("content_sha256") == document.get("content_sha256")
    ):
        raise TargetSealedError(f"{label} exact binding drifted")


def _require_authority_sha_binding(
    row: object,
    *,
    binding: FileBinding,
    document: Mapping[str, Any],
    project_root: Path,
    label: str,
    content_hash_required: bool = True,
) -> None:
    """Match the newer sha256/mode authority evidence to one mounted file."""

    expected_keys = {"path", "sha256", "bytes", "mode"}
    if content_hash_required:
        expected_keys.add("content_sha256")
    if not isinstance(row, Mapping) or set(row) != expected_keys:
        raise TargetSealedError(f"{label} binding schema drifted")
    if not (
        row.get("path") == binding.path.relative_to(project_root).as_posix()
        and row.get("sha256") == binding.sha256
        and row.get("bytes") == binding.bytes
        and row.get("mode") == "0444"
        and binding.mode == 0o444
        and (
            not content_hash_required
            or row.get("content_sha256") == document.get("content_sha256")
        )
    ):
        raise TargetSealedError(f"{label} exact binding drifted")


def _canonical_gpu_state_path_document() -> dict[str, str]:
    root = GPU_STATE_ROOT_RELATIVE
    return {
        "root": root.as_posix(),
        "usage_directory": (root / "usage").as_posix(),
        "usage_ledger": (
            root / "usage/campaign_gpu_usage_chain_v6.jsonl"
        ).as_posix(),
        "execution_directory": (root / "execution").as_posix(),
        "execution_ledger": (
            root / "execution/gpu_execution_ledger_v7.jsonl"
        ).as_posix(),
        "admission_directory": (root / "admission").as_posix(),
        "admission_lock": (root / "admission/gpu_admission_v7.lock").as_posix(),
    }


EXECUTION_CLOSURE_HISTORICAL_ROWS: Final[tuple[tuple[str, str, int, str, str], ...]] = (
    ("artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8/BENCHMARK_INVOCATION.json", "202958d02a8280c89bb1561e3f003c7cbf8cf05539820f11f94cf864e0ba63a3", 5344, "0444", "v8_unit_invocation"),
    ("artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8/BENCHMARK_INVOCATION_V8R2.json", "f5451c96f5985769af56adf63107d157717ed97275cc9f93a158e951a33ac1cc", 5346, "0444", "v8r2_unit_invocation"),
    ("artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8/BENCHMARK_INVOCATION_V8R3.json", "b0c5e985441c3cd8a4a335cf91a9a840207673a6ce66a99603e813e383cabb6b", 5349, "0444", "v8r3_unit_invocation"),
    ("artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8/attempts/attempt_000/invocation.json", "cd9727c64de434a0946adb99e6a3dff3005282ee811350eb5102b91b5a5bac7b", 2557, "0444", "v8_execution_invocation"),
    ("artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8/attempts/attempt_001/invocation.json", "fdca8a8488f0e302aaa65054870a82f0557029bcd576130439596a3c09761cb1", 2564, "0444", "v8r2_execution_invocation"),
    ("artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8/attempts/attempt_002/invocation.json", "2c9fc326d2304c4fb690316f5bcdfc4503505f4026c8150cd7f71003c086cd08", 2564, "0444", "v8r3_execution_invocation"),
    ("artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8/attempts/attempt_000/GPU_TERMINAL_RESULT.json", "7aea7714e9f5f248254a0052442e632d946930ff8b45ee1f28a39d653fdf41be", 1699, "0444", "v8_terminal_result"),
    ("artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8/attempts/attempt_001/GPU_TERMINAL_RESULT.json", "fccf0bdea0ab17f8f6da33acc7ab4ebf9e0f28301a4a4ea6b4b9d8fa245e70c2", 1703, "0444", "v8r2_terminal_result"),
    ("artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8/attempts/attempt_002/GPU_TERMINAL_RESULT.json", "0b12a5420441faa69555f50890d4618a16e4de24fca76bd0d1a5797cc0656338", 1700, "0644", "v8r3_quarantined_terminal_result"),
    ("artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8/attempts/attempt_002/QUARANTINED_TIMING_TELEMETRY.json", "b1e36d03d98c5516512dcc349812707a851a719b3b9a3b6d17d45ccacce8660c", 4763, "0444", "v8r3_quarantined_telemetry"),
    ("artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/PRETRAIN_AUTHORIZATION_V8.json", "91e949cade57da4555f522de378cabcb5f99b1515bf1f87d6e9d26606acedb2b", 4115, "0444", "v8_pretrain_authorization"),
    ("artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/PRETRAIN_AUTHORIZATION_V8R2.json", "4918d6b4396bca694db43deb01d26cdfca43286465e811fd8695c258ab917ded", 4122, "0444", "v8r2_pretrain_authorization"),
    ("artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/PRETRAIN_AUTHORIZATION_V8R3.json", "26ef02cf9f5abb8ec44ed4f82c0f3e738a46f4f5a5e1719ef94a087aee2bd10f", 4124, "0444", "v8r3_pretrain_authorization"),
)


def _validate_execution_closure_historical_projection(
    authority: Mapping[str, Any],
) -> None:
    basis = authority.get("authority_basis")
    if not isinstance(basis, Mapping) or set(basis) != {
        "authorization_limited_to_terminal_execution_closure",
        "diagnostic",
        "historical_benchmark_prefix",
        "parent_kill_safe_addendum",
        "parent_open_lifecycle_recovery_addendum",
        "parent_source_closure_addendum",
        "user_goal_scope",
    }:
        raise TargetSealedError("V8R4A execution-closure authority basis drifted")
    history = basis.get("historical_benchmark_prefix")
    if not isinstance(history, Mapping) or set(history) != {
        "active_output_root",
        "entries",
        "historical_root_mounted_or_mutated",
        "known_v8r3_mode_0644_is_read_only_quarantined_evidence",
    }:
        raise TargetSealedError("V8R4A historical benchmark projection schema drifted")
    expected_entries = [
        {"path": path, "file_sha256": digest, "bytes": size, "mode": mode, "role": role}
        for path, digest, size, mode, role in EXECUTION_CLOSURE_HISTORICAL_ROWS
    ]
    if not (
        basis.get("authorization_limited_to_terminal_execution_closure") is True
        and history.get("active_output_root")
        == LEGACY_EXECUTION_CLOSURE_BENCHMARK_OUTPUT_RELATIVE.as_posix()
        and history.get("historical_root_mounted_or_mutated") is False
        and history.get("known_v8r3_mode_0644_is_read_only_quarantined_evidence") is True
        and history.get("entries") == expected_entries
    ):
        raise TargetSealedError("V8R4A historical benchmark projection drifted")


FD_CLOSURE_DIAGNOSTIC_LEGACY_BINDING: Final[Mapping[str, Any]] = {
    "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/v3r1_v8r4a_outer_guard_urandom_descriptor_failure.json",
    "file_sha256": "75766bbbcc2e1bdc6cdcc61ddee559a2ffa647586f7cbbb87fea0696034d8fbd",
    "bytes": 5207,
    "content_sha256": "31834a6b67074314b5e6440d035a040cd19c81d8408e81cf681c4e54e1350f1a",
}
FD_CLOSURE_AUTHORITY_FILE_SHA256: Final[str] = (
    "1ad3bdaa0b78937c5b6ce98bc2e4e02d31e41951baf57dde6d68aa8029b25110"
)
FD_CLOSURE_AUTHORITY_BYTES: Final[int] = 8447
FD_CLOSURE_DIAGNOSTIC_FILE_SHA256: Final[str] = (
    "75766bbbcc2e1bdc6cdcc61ddee559a2ffa647586f7cbbb87fea0696034d8fbd"
)
FD_CLOSURE_DIAGNOSTIC_BYTES: Final[int] = 5207
FD_CLOSURE_PARENT_BINDINGS: Final[Mapping[str, Mapping[str, Any]]] = {
    "parent_implementation_test_receipt": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/IMPLEMENTATION_TEST_RECEIPT_V8R4A.json",
        "file_sha256": "cb7bb51558d89ea79728063a33fe719edc9f813416d356bf78665b303b56a5b4",
        "bytes": 28291,
        "content_sha256": "fd650ea690889236e5216b5c7110d0bcfe43ee374d6d5d7a712915bdff4cdd42",
    },
    "parent_source_snapshot": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/V3R1_SOURCE_SNAPSHOT_V8R4A.json",
        "file_sha256": "4a0278d146255caa2a50669d1ad05750b672988142598db8b8160cec9b50ccf1",
        "bytes": 20862,
        "content_sha256": "89c4e85f6a638d660a89dff4d12e46cb430efa76f8264d80b80364caf30606ba",
    },
    "parent_pretrain_authorization": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/PRETRAIN_AUTHORIZATION_V8R4A.json",
        "file_sha256": "f42cc8e1a144eacb0440fb6c7287c3a3fac8b38c47258b65000e2708b002b6e3",
        "bytes": 10385,
        "content_sha256": "b75f18580158feaee363297731b69fd6e144c6c2219d55a9623d438cffa7f41c",
    },
    "parent_execution_closure_authority": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_EXECUTION_CLOSURE.json",
        "file_sha256": "11e5e7d00d3d837e0eb542b3482499cd21bf90b38883d0f4a0f86b68ba16d754",
        "bytes": 21621,
        "content_sha256": "92d96a4f513a7d7f93bbd4baf227b626106dab54e000f3a01c97b25504c58c1c",
    },
}
FD_CLOSURE_AUTHORIZED_BEFORE: Final[Mapping[str, str]] = {
    "scripts/run_hfr_v3r1_target_sealed.py": "dd0f295f7789924e8f59ee23c569138e8ebf6725b50b4349eb469da15ad56e49",
    "tests/test_run_hfr_v3r1_target_sealed.py": "d616e41b5183501f5cacf577f9ae7547534324e17d3f2beb407915148b1c12ee",
    "scripts/validate_hfr_v3r1_authorization.py": "80e31dec623d658c43bc7f57b3d738d0d8a7f7cb5772c508a5c9104e3e53150e",
    "tests/test_validate_hfr_v3r1_authorization_v8r4a.py": "844716e5a4fca4b405a427869a3c7914c870fc0b7356a81e97be80d2edd254fe",
    "scripts/run_hfr_v3r1_v8r4a_campaign.py": "aff4f1e3d774ca62e210c9ccb32ec2e310824a90e299f177cab0e18cfceeb9dc",
    "tests/test_run_hfr_v3r1_v8r4a_campaign.py": "898117c19c4be0232a6680feb5e8c6181b66d2551243a2b5f2ee4e023a234291",
    "scripts/benchmark_hfr_v3r1_efficiency.py": "e07bc950e43f20d1a1c2a531ed62d0a568381f5a249b14c17d951d76d8834723",
    "scripts/train_harmonic_factor_router_snn_v3r1.py": "91a927d43aa893577ba921e01f6f17c1f9e9ee5d2c5e3afbff7b2451d92bebea",
    "scripts/select_hfr_v3r1_common_variant.py": "964d15cd6507ab948b96bb2f2f72c7232fa45e65711a0a8a9e2d0b65fda57edc",
    "tests/test_benchmark_hfr_v3r1_efficiency.py": "14249bc244a08f4eefc86a43947c453d400dc9a72748f459b32cad377744c461",
    "tests/test_train_harmonic_factor_router_snn_v3r1.py": "46d9810a4bd20b68cc10313b95896f880898b0ad6bc37e3444526180bb023793",
    "tests/test_run_hfr_v3r1_campaign.py": "438624c7f6f8d33c32ee05c796c1d9b6a04d8935f7f8e46fe20ceab5d741f8ed",
}


def _validate_fd_closure_projection(
    authority: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    *,
    enforce_production_literal_bindings: bool = True,
) -> None:
    basis = authority.get("authority_basis")
    failed = diagnostic.get("failed_attempt")
    reproduction = diagnostic.get("reproduction")
    required = authority.get("required_reauthorization")
    claim = authority.get("claim_boundary")
    rows = authority.get("authorized_modifications")
    if not (
        authority.get("schema_version") == 1
        and diagnostic.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_outer_guard_urandom_descriptor_closure_correction_addendum"
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_outer_guard_urandom_descriptor_failure_diagnostic"
        and authority.get("campaign_id") == CAMPAIGN_ID
        and diagnostic.get("campaign_id") == CAMPAIGN_ID
        and authority.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and diagnostic.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and authority.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and diagnostic.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and isinstance(basis, Mapping)
        and set(basis)
        == {"diagnostic", *FD_CLOSURE_PARENT_BINDINGS, "user_goal_scope"}
        and (
            not enforce_production_literal_bindings
            or canonical_json_bytes(basis.get("diagnostic"))
            == canonical_json_bytes(FD_CLOSURE_DIAGNOSTIC_LEGACY_BINDING)
        )
        and all(
            canonical_json_bytes(basis.get(field)) == canonical_json_bytes(binding)
            for field, binding in FD_CLOSURE_PARENT_BINDINGS.items()
            if (
                enforce_production_literal_bindings
                or field != "parent_execution_closure_authority"
            )
        )
        and isinstance(basis.get("user_goal_scope"), str)
        and bool(basis["user_goal_scope"])
        and isinstance(failed, Mapping)
        and failed.get("first_phase") == "efficiency_benchmark"
        and failed.get("outer_fold") == 3
        and failed.get("target_runtime_return_code") == 73
        and failed.get("coordinator_return_code") == 79
        and failed.get("gpu_child_launched") is False
        and isinstance(reproduction, Mapping)
        and reproduction.get("descriptor_3_target") == "/dev/urandom"
        and reproduction.get("descriptor_3_fd_cloexec") is True
        and reproduction.get("arbitrary_unexpected_descriptor_rejection_must_remain")
        is True
        and isinstance(required, Mapping)
        and required.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8R4A_FD1.json"
        and required.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8R4A_FD1.json"
        and required.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8R4A_FD1.json"
        and required.get("all_fixed_tests_pass") is True
        and required.get("fresh_interpreter_subprocess_regression_passes") is True
        and required.get("gpu_retry_only_after_successor_pretrain_validation")
        is True
        and isinstance(claim, Mapping)
        and claim.get("correction_is_infrastructure_only") is True
        and claim.get("gpu_execution_authorized_by_this_document") is False
        and claim.get("successor_pretrain_authorization_required") is True
        and claim.get("commercial_claim_authorized") is False
        and isinstance(rows, list)
    ):
        raise TargetSealedError("V8R4A FD-closure projection drifted")
    observed: dict[str, str] = {}
    for row in rows:
        if not (
            isinstance(row, Mapping)
            and set(row) == {"path", "before_sha256", "allowed_change"}
            and isinstance(row.get("path"), str)
            and isinstance(row.get("before_sha256"), str)
            and isinstance(row.get("allowed_change"), str)
            and bool(row["allowed_change"])
        ):
            raise TargetSealedError("V8R4A FD-closure modification row drifted")
        path = str(row["path"])
        if path in observed:
            raise TargetSealedError("V8R4A FD-closure modification is duplicated")
        observed[path] = str(row["before_sha256"])
    if observed != FD_CLOSURE_AUTHORIZED_BEFORE:
        raise TargetSealedError("V8R4A FD-closure modification cover drifted")
    superseded = diagnostic.get("superseded_pretrain_chain")
    expected_superseded = {
        "implementation_test_receipt": FD_CLOSURE_PARENT_BINDINGS[
            "parent_implementation_test_receipt"
        ],
        "source_snapshot": FD_CLOSURE_PARENT_BINDINGS[
            "parent_source_snapshot"
        ],
        "pretrain_authorization": FD_CLOSURE_PARENT_BINDINGS[
            "parent_pretrain_authorization"
        ],
        "preserved_as_immutable_audit_evidence": True,
        "may_authorize_retry_without_successor_chain": False,
    }
    if canonical_json_bytes(superseded) != canonical_json_bytes(expected_superseded):
        raise TargetSealedError("V8R4A FD-closure superseded chain drifted")


CANARY_BOUNDARY_DIAGNOSTIC_LEGACY_BINDING: Final[Mapping[str, Any]] = {
    "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/v3r1_v8r4a_denied_canary_prefix_collision_failure.json",
    "file_sha256": "7ea5aea6f661dc0e3157a6ee71e20b2e5e04da5a4dc81035e652e9902ecb6294",
    "bytes": 5551,
    "content_sha256": "00b87df937342a6d3a6f1cd13d1bf7bdc51d33df8b09f276a8fc1017ba39d63e",
}
CANARY_BOUNDARY_AUTHORITY_FILE_SHA256: Final[str] = (
    "8187bd7c306419114c25020cf2a89c5a740d766b12df5ddd2e6d29ca498a84c3"
)
CANARY_BOUNDARY_AUTHORITY_BYTES: Final[int] = 8659
CANARY_BOUNDARY_DIAGNOSTIC_FILE_SHA256: Final[str] = (
    "7ea5aea6f661dc0e3157a6ee71e20b2e5e04da5a4dc81035e652e9902ecb6294"
)
CANARY_BOUNDARY_DIAGNOSTIC_BYTES: Final[int] = 5551
CANARY_BOUNDARY_PARENT_BINDINGS: Final[Mapping[str, Mapping[str, Any]]] = {
    "parent_fd_closure_authority": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_FD_CLOSURE.json",
        "file_sha256": "1ad3bdaa0b78937c5b6ce98bc2e4e02d31e41951baf57dde6d68aa8029b25110",
        "bytes": 8447,
        "content_sha256": "c199171692d8c13be883ada42ad8d7b25cd44f19544f38f2ad1685fab6e498a7",
    },
    "parent_implementation_test_receipt": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/IMPLEMENTATION_TEST_RECEIPT_V8R4A_FD1.json",
        "file_sha256": "0c8414408439af38b7c8a0ac5b5a81967185131ba9ec81b87849b793b553616c",
        "bytes": 29127,
        "content_sha256": "c13cff4ce5dad8d3079a125d9218403773d2bcc464bf2038731fc97801c7d67b",
    },
    "parent_source_snapshot": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/V3R1_SOURCE_SNAPSHOT_V8R4A_FD1.json",
        "file_sha256": "debf86e706856c2b55ba590cf33fa7c9b3fe53b8b29372190c87d18d5bb3c783",
        "bytes": 21597,
        "content_sha256": "85c4ed4c9df1cf67f08cfad7b240ae13ebf9ce9331d9fe2d303adfc1ba0cbdea",
    },
    "parent_pretrain_authorization": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/PRETRAIN_AUTHORIZATION_V8R4A_FD1.json",
        "file_sha256": "f9b18fb1186123b4fee77265601b1165fb7b47e5d321b8277905434e3337b79d",
        "bytes": 11124,
        "content_sha256": "281168deeee7f6e503b4386ef5742a36f6074479e30263a902ebffad8d37448a",
    },
}
CANARY_BOUNDARY_AUTHORIZED_BEFORE: Final[Mapping[str, str]] = {
    "scripts/run_hfr_v3r1_target_sealed.py": "07bc135f246023f59097800e049318464ee82bb664c8b4e56dcb36aa0d7d5b55",
    "tests/test_run_hfr_v3r1_target_sealed.py": "dcbe90d9c21748bde39381682be51e95f1dc56de92cc6cb7e091f5a11dbdfc2e",
    "scripts/validate_hfr_v3r1_authorization.py": "fd5efced383b8d77cc990bc6380b293d17b63d175b12268a1ff2e3dd4eeb29d9",
    "tests/test_validate_hfr_v3r1_authorization_v8r4a.py": "80dbcebabb08aac06ea40408806ba0f3a6581a13e007a0397fbbbd14f5d658af",
    "scripts/run_hfr_v3r1_v8r4a_campaign.py": "8983a2aaf95174f86256420ff857a769caaadb781e2be18b09ff6be7fadfa2b6",
    "tests/test_run_hfr_v3r1_v8r4a_campaign.py": "606b4f52798437fb289648dc53b2eaaaddc9c06bcb3a1381a5f8a86e685046c0",
    "scripts/benchmark_hfr_v3r1_efficiency.py": "5f2a929ac74304d2fa08b6f626e71c45cd16eecd25a3682268858003d737afc5",
    "tests/test_benchmark_hfr_v3r1_efficiency.py": "ae6701b229489e8236ad5a6e1a17ad2f274f2b6cfe5a0a73d76c1e9ad7f3ffec",
    "scripts/train_harmonic_factor_router_snn_v3r1.py": "f374b13326bdb1eb5d54a7dc3b164b756e7480592b72364dcfd34e830ad94625",
    "scripts/select_hfr_v3r1_common_variant.py": "243dc69d6c68a2e7191e76d2c7e2f52a32716074c605075c8484f5bf83b7ec2f",
}
CANARY_BOUNDARY_ACTIVE_FILENAMES: Final[Mapping[str, str]] = {
    "implementation_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CANARY1.json",
    "source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_CANARY1.json",
    "active_authorization": "PRETRAIN_AUTHORIZATION_V8R4A_CANARY1.json",
}
CANARY_BOUNDARY_REQUIRED_REAUTHORIZATION: Final[Mapping[str, Any]] = {
    "new_test_receipt": CANARY_BOUNDARY_ACTIVE_FILENAMES[
        "implementation_test_receipt"
    ],
    "new_source_snapshot": CANARY_BOUNDARY_ACTIVE_FILENAMES["source_snapshot"],
    "new_pretrain_authorization": CANARY_BOUNDARY_ACTIVE_FILENAMES[
        "active_authorization"
    ],
    "all_fixed_tests_pass": True,
    "component_boundary_regressions_pass": True,
    "active_usage_and_execution_ledgers_closed_and_unchanged_during_cpu_tests": True,
    "diagnostic_and_authority_bound_in_every_target_capability": True,
    "gpu_retry_only_after_successor_pretrain_validation": True,
}


def _validate_canary_boundary_projection(
    authority: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    *,
    enforce_production_literal_bindings: bool = True,
) -> None:
    """Validate CANARY1 without opening any superseded FD1 issuance file."""

    basis = authority.get("authority_basis")
    rows = authority.get("authorized_modifications")
    required = authority.get("required_reauthorization")
    mandatory = authority.get("mandatory_invariants")
    forbidden = authority.get("forbidden_changes")
    claim = authority.get("claim_boundary")
    failed = diagnostic.get("failed_attempt")
    root_cause = diagnostic.get("root_cause")
    reproduction = diagnostic.get("reproduction")
    diagnostic_required = diagnostic.get("required_correction")
    diagnostic_claim = diagnostic.get("claim_boundary")
    if not (
        authority.get("schema_version") == 1
        and diagnostic.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_denied_canary_component_boundary_correction_addendum"
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_denied_canary_prefix_collision_failure_diagnostic"
        and authority.get("campaign_id") == CAMPAIGN_ID
        and diagnostic.get("campaign_id") == CAMPAIGN_ID
        and authority.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and diagnostic.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and authority.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and diagnostic.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and isinstance(basis, Mapping)
        and set(basis)
        == {"diagnostic", *CANARY_BOUNDARY_PARENT_BINDINGS, "user_goal_scope"}
        and (
            not enforce_production_literal_bindings
            or canonical_json_bytes(basis.get("diagnostic"))
            == canonical_json_bytes(CANARY_BOUNDARY_DIAGNOSTIC_LEGACY_BINDING)
        )
        and all(
            canonical_json_bytes(basis.get(field)) == canonical_json_bytes(binding)
            for field, binding in CANARY_BOUNDARY_PARENT_BINDINGS.items()
            if (
                enforce_production_literal_bindings
                or field != "parent_fd_closure_authority"
            )
        )
        and isinstance(basis.get("user_goal_scope"), str)
        and bool(basis["user_goal_scope"])
        and isinstance(rows, list)
        and canonical_json_bytes(required)
        == canonical_json_bytes(CANARY_BOUNDARY_REQUIRED_REAUTHORIZATION)
        and isinstance(mandatory, Mapping)
        and set(mandatory)
        == {
            "scientific_campaign_revision_unchanged",
            "infrastructure_revision_unchanged",
            "variants_unchanged",
            "seeds_unchanged",
            "discovery_outer_folds_unchanged",
            "target_and_outer_reference_sealing_unchanged",
            "gpu_budget_and_append_only_ledgers_unchanged",
            "single_gpu_owner_and_kill_safe_lifecycle_unchanged",
            "denied_canary_paths_unchanged",
            "exact_denied_paths_and_descendants_fail_closed",
            "path_distinct_sibling_prefixes_are_not_capabilities",
            "embedded_absolute_option_paths_fail_closed",
            "path_traversal_fails_before_normalization",
            "component_aware_mount_validation_unchanged",
            "sandbox_denied_canary_probe_unchanged",
            "parent_fd1_chain_preserved_immutable",
            "empty_failed_attempt_directories_may_be_reused_only_under_existing_exact_resume_rules",
        }
        and mandatory.get("scientific_campaign_revision_unchanged") == "V8R4"
        and mandatory.get("infrastructure_revision_unchanged") == "V8R4A"
        and mandatory.get("variants_unchanged")
        == ["H0_no_factor", "H1_factor", "H2_full"]
        and mandatory.get("seeds_unchanged") == list(SEEDS)
        and mandatory.get("discovery_outer_folds_unchanged") == [3, 4]
        and all(
            mandatory.get(field) is True
            for field in (
                "target_and_outer_reference_sealing_unchanged",
                "gpu_budget_and_append_only_ledgers_unchanged",
                "single_gpu_owner_and_kill_safe_lifecycle_unchanged",
                "denied_canary_paths_unchanged",
                "exact_denied_paths_and_descendants_fail_closed",
                "path_distinct_sibling_prefixes_are_not_capabilities",
                "embedded_absolute_option_paths_fail_closed",
                "path_traversal_fails_before_normalization",
                "component_aware_mount_validation_unchanged",
                "sandbox_denied_canary_probe_unchanged",
                "parent_fd1_chain_preserved_immutable",
                "empty_failed_attempt_directories_may_be_reused_only_under_existing_exact_resume_rules",
            )
        )
        and isinstance(forbidden, Mapping)
        and set(forbidden)
        == {
            "model_architecture_or_loss",
            "hyperparameters_or_epoch_counts",
            "data_rows_splits_or_pack_bytes",
            "seed_variant_or_fold_matrix",
            "metric_or_selection_thresholds",
            "outer_target_or_reference_access",
            "renaming_or_weakening_denied_canaries_to_avoid_the_collision",
            "removing_component_aware_mount_validation",
            "removing_sandbox_denied_canary_probes",
            "allowing_exact_or_descendant_denied_paths",
            "filesystem_resolution_or_existence_dependency_for_absent_canaries",
            "ledger_reset_truncation_or_rewrite",
            "mutation_or_replacement_of_parent_evidence",
            "commercial_or_confirmatory_claim",
        }
        and all(value is True for value in forbidden.values())
        and isinstance(claim, Mapping)
        and set(claim)
        == {
            "adaptive_retrospective_only",
            "correction_is_infrastructure_only",
            "outer_test_features_or_targets_opened",
            "accuracy_metric_used",
            "gpu_execution_authorized_by_this_document",
            "successor_pretrain_authorization_required",
            "commercial_claim_authorized",
        }
        and claim.get("adaptive_retrospective_only") is True
        and claim.get("correction_is_infrastructure_only") is True
        and claim.get("outer_test_features_or_targets_opened") is False
        and claim.get("accuracy_metric_used") is False
        and claim.get("gpu_execution_authorized_by_this_document") is False
        and claim.get("successor_pretrain_authorization_required") is True
        and claim.get("commercial_claim_authorized") is False
        and isinstance(failed, Mapping)
        and failed.get("first_phase") == "efficiency_benchmark"
        and failed.get("outer_fold") == 3
        and failed.get("target_runtime_return_code") == 73
        and failed.get("coordinator_return_code") == 79
        and failed.get("stderr") == "child command names a denied capability"
        and failed.get("capability_receipt_created") is False
        and failed.get("completion_receipt_created") is False
        and failed.get("gpu_child_launched") is False
        and failed.get("gpu_usage_ledger_mutated") is False
        and failed.get("gpu_execution_ledger_mutated") is False
        and isinstance(root_cause, Mapping)
        and root_cause.get("raw_substring_relation") is True
        and root_cause.get("path_component_ancestor_relation") is False
        and isinstance(reproduction, Mapping)
        and canonical_json_bytes(reproduction)
        == canonical_json_bytes(
            {
                "python_substring_result": True,
                "lexical_relative_to_denied_result": False,
                "component_aware_mount_boundary_validation_passed": True,
                "exact_denied_path_must_fail": True,
                "denied_descendant_must_fail": True,
                "path_distinct_prefix_siblings_must_pass": True,
                "embedded_absolute_option_path_must_not_bypass_validation": True,
                "path_traversal_must_fail_before_normalization": True,
            }
        )
        and canonical_json_bytes(diagnostic_required)
        == canonical_json_bytes(
            {
                "replace_raw_substring_with_lexical_component_boundary_check": True,
                "normalize_only_absolute_path_tokens_without_filesystem_resolution": True,
                "reject_exact_denied_paths_and_descendants": True,
                "allow_path_distinct_sibling_prefixes": True,
                "reject_traversal_and_embedded_absolute_option_paths": True,
                "retain_component_aware_mount_validation_and_denied_canary_probe": True,
                "bind_diagnostic_and_correction_in_runtime_governance": True,
                "new_test_receipt": CANARY_BOUNDARY_ACTIVE_FILENAMES[
                    "implementation_test_receipt"
                ],
                "new_source_snapshot": CANARY_BOUNDARY_ACTIVE_FILENAMES[
                    "source_snapshot"
                ],
                "new_pretrain_authorization": CANARY_BOUNDARY_ACTIVE_FILENAMES[
                    "active_authorization"
                ],
                "full_reauthorization_before_gpu_retry": True,
            }
        )
        and isinstance(diagnostic_claim, Mapping)
        and diagnostic_claim.get("adaptive_retrospective_only") is True
        and diagnostic_claim.get("outer_test_features_or_targets_opened") is False
        and diagnostic_claim.get("accuracy_metric_computed") is False
        and diagnostic_claim.get("gpu_accessed") is False
        and diagnostic_claim.get("scientific_configuration_change_authorized")
        is False
        and diagnostic_claim.get("commercial_claim_authorized") is False
    ):
        raise TargetSealedError("V8R4A canary-boundary projection drifted")

    observed: dict[str, str] = {}
    for row in rows:
        if not (
            isinstance(row, Mapping)
            and set(row) == {"path", "before_sha256", "allowed_change"}
            and isinstance(row.get("path"), str)
            and isinstance(row.get("before_sha256"), str)
            and isinstance(row.get("allowed_change"), str)
            and bool(row["allowed_change"])
        ):
            raise TargetSealedError("V8R4A canary-boundary modification row drifted")
        path = str(row["path"])
        if path in observed:
            raise TargetSealedError(
                "V8R4A canary-boundary modification is duplicated"
            )
        observed[path] = str(row["before_sha256"])
    if observed != CANARY_BOUNDARY_AUTHORIZED_BEFORE:
        raise TargetSealedError("V8R4A canary-boundary modification cover drifted")

    expected_superseded = {
        "implementation_test_receipt": CANARY_BOUNDARY_PARENT_BINDINGS[
            "parent_implementation_test_receipt"
        ],
        "source_snapshot": CANARY_BOUNDARY_PARENT_BINDINGS[
            "parent_source_snapshot"
        ],
        "pretrain_authorization": CANARY_BOUNDARY_PARENT_BINDINGS[
            "parent_pretrain_authorization"
        ],
        "preserved_as_immutable_audit_evidence": True,
        "may_authorize_retry_without_successor_chain": False,
    }
    if canonical_json_bytes(
        diagnostic.get("superseded_pretrain_chain")
    ) != canonical_json_bytes(expected_superseded):
        raise TargetSealedError("V8R4A canary-boundary superseded chain drifted")


FROZEN_CONTRACT_DIAGNOSTIC_LEGACY_BINDING: Final[Mapping[str, Any]] = {
    "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/v3r1_v8r4a_frozen_contract_encoding_false_rejection_failure.json",
    "file_sha256": "1d358bdecb1a5605b138b1916451f8c306f957a27257d328aa71de8d83cbf8ee",
    "bytes": 8653,
    "content_sha256": "bec8b74c21c0b0882f9ab147f68c1aa947e259ec39bf36557f3fdcdeb86abcc7",
}
FROZEN_CONTRACT_AUTHORITY_FILE_SHA256: Final[str] = (
    "2a59c7de8b8aefcbb3d21691db5a8da1e4b8f1a5461024f0338f389ff8ddcda1"
)
FROZEN_CONTRACT_AUTHORITY_BYTES: Final[int] = 13460
FROZEN_CONTRACT_DIAGNOSTIC_FILE_SHA256: Final[str] = (
    "1d358bdecb1a5605b138b1916451f8c306f957a27257d328aa71de8d83cbf8ee"
)
FROZEN_CONTRACT_DIAGNOSTIC_BYTES: Final[int] = 8653
FROZEN_CAMPAIGN_CONTRACT_EVIDENCE_BINDING: Final[Mapping[str, Any]] = {
    "path": CAMPAIGN_CONTRACT_RELATIVE_PATH.as_posix(),
    "file_sha256": "532d150f0241d9675873368107d09adec7aeaee5e018e09537e8a340eb6fa2bd",
    "bytes": 16179,
    "content_sha256": "6912e9760d1ab937604ba7868fe4742554804bd7179b5be2d6c8c5b34115aa2d",
}
FROZEN_CONTRACT_PARENT_BINDINGS: Final[Mapping[str, Mapping[str, Any]]] = {
    "parent_canary_boundary_authority": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_CANARY_BOUNDARY.json",
        "file_sha256": "8187bd7c306419114c25020cf2a89c5a740d766b12df5ddd2e6d29ca498a84c3",
        "bytes": 8659,
        "content_sha256": "318e0979a4732ff8f3b2e39f3f57ec069e352259864045f43fd7bdef54243aa4",
    },
    "parent_canary_boundary_diagnostic": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/v3r1_v8r4a_denied_canary_prefix_collision_failure.json",
        "file_sha256": "7ea5aea6f661dc0e3157a6ee71e20b2e5e04da5a4dc81035e652e9902ecb6294",
        "bytes": 5551,
        "content_sha256": "00b87df937342a6d3a6f1cd13d1bf7bdc51d33df8b09f276a8fc1017ba39d63e",
    },
    "parent_implementation_test_receipt": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/IMPLEMENTATION_TEST_RECEIPT_V8R4A_CANARY1.json",
        "file_sha256": "646ac34da7ed1032b21cdffc3a65885b9ac7a13210bbf704d7995b58970f27fe",
        "bytes": 29873,
        "content_sha256": "d47dd9e0f7e0642db7515c71158ec8cdfea6b6ffb04c760f8ba7624d9490b449",
    },
    "parent_source_snapshot": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/V3R1_SOURCE_SNAPSHOT_V8R4A_CANARY1.json",
        "file_sha256": "fe313e568b2e0dc9a19ef6d3d4be397d04244f18249fc6ba017d129d1491f53c",
        "bytes": 22347,
        "content_sha256": "545069175693b981510b2219d6bc72369f7d6c4fd265bda279e79960f0bd3093",
    },
    "parent_pretrain_authorization": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/PRETRAIN_AUTHORIZATION_V8R4A_CANARY1.json",
        "file_sha256": "fb051a410499599b3cabd5418fd338c3a39248e05f9db21de91047aea1672d07",
        "bytes": 11878,
        "content_sha256": "378f4233095002f240cdaef4a52a43b41583f5f512a8f8e3040bb5f1d9585110",
    },
    "frozen_campaign_contract": FROZEN_CAMPAIGN_CONTRACT_EVIDENCE_BINDING,
    "failed_capability_receipt": {
        "path": "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/target_sealed_lifecycle_v8r4a/efficiency_benchmark/benchmark_hfr_v3r1_efficiency/outer_3/TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json",
        "file_sha256": "5bf98a9be31ef92aea43a1c777fd2ab5725317e1efe7b003d8283c725f16206d",
        "bytes": 50077,
        "content_sha256": "8e49dbfeaca212673662e0ca7379796c98b26cf106069c6bbf721b7e0d56faa0",
    },
    "failed_completion_receipt": {
        "path": "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/target_sealed_lifecycle_v8r4a/efficiency_benchmark/benchmark_hfr_v3r1_efficiency/outer_3/TARGET_SEALED_COMPLETION_RECEIPT_V8R4A.json",
        "file_sha256": "d16ca7006448386f569fd50c9ae7be61066219dd5dd93d417bb7784d3f2226d2",
        "bytes": 7680,
        "content_sha256": "d6f79fb027609d03420eca3ef2f8327e8664617e64edfabed99a7aa93c0cac54",
    },
}
FROZEN_CONTRACT_AUTHORIZED_BEFORE: Final[Mapping[str, str]] = {
    "scripts/run_hfr_v3r1_target_sealed.py": "add657735c578b54629616953b5664668080bb2b905cee7b7bf06d4a4a29add5",
    "tests/test_run_hfr_v3r1_target_sealed.py": "62cc31e84518e39dca8f4301768665add6885a52f3a71317492915b8b8d3aea2",
    "scripts/validate_hfr_v3r1_authorization.py": "910092df5fe99e4325fa203f981072e6396ff3b331d762d302bb0173673bb8b3",
    "tests/test_validate_hfr_v3r1_authorization_v8r4a.py": "ddcdb282f5444defdb700ef9e1a65032f1bfa4fc27a97efb21faf9b5746ba352",
    "scripts/run_hfr_v3r1_v8r4a_campaign.py": "a10e5638faef69bdff2af8e4c27d73860d83933dee01ff5bcd8097de776f8156",
    "tests/test_run_hfr_v3r1_v8r4a_campaign.py": "0a01a0c5946235ab04a910f7ba2d0c59689cb1778da16df633e08fe3d3db8a9a",
    "scripts/benchmark_hfr_v3r1_efficiency.py": "3ac6375188103fad2be7dd926e30db6afa75407f40cdbe00e8d71d242e17a1ad",
    "tests/test_benchmark_hfr_v3r1_efficiency.py": "919ca330dacd2eaf402bb021050627b8ea61181837ad8b1db5a77cc36dffaac3",
    "scripts/train_harmonic_factor_router_snn_v3r1.py": "aee78c22c52184507e5c9312105e55066a1f9d619d3c54124b7137def7a67176",
    "scripts/select_hfr_v3r1_common_variant.py": "927b257b93373904939f8a563cbcd4e5ce5c7a935ac77dc7014028a20446ed8f",
    "scripts/run_hfr_v3r1_discovery_campaign.py": "811cb108a497ac768fa62e402fff4c33eb24213668fc67f9d52f9cacec8dc10d",
    "tests/test_run_hfr_v3r1_campaign.py": "438624c7f6f8d33c32ee05c796c1d9b6a04d8935f7f8e46fe20ceab5d741f8ed",
}
FROZEN_CONTRACT_ACTIVE_FILENAMES: Final[Mapping[str, str]] = {
    "implementation_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTRACT1.json",
    "source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTRACT1.json",
    "active_authorization": "PRETRAIN_AUTHORIZATION_V8R4A_CONTRACT1.json",
}
FROZEN_CONTRACT_REQUIRED_REAUTHORIZATION: Final[Mapping[str, Any]] = {
    "new_test_receipt": FROZEN_CONTRACT_ACTIVE_FILENAMES[
        "implementation_test_receipt"
    ],
    "new_source_snapshot": FROZEN_CONTRACT_ACTIVE_FILENAMES["source_snapshot"],
    "new_pretrain_authorization": FROZEN_CONTRACT_ACTIVE_FILENAMES[
        "active_authorization"
    ],
    "new_governance_roles": [
        "frozen_contract_encoding_correction_authorization",
        "frozen_contract_encoding_failure_diagnostic",
    ],
    "new_denied_canary_roles": [
        "superseded_v8r4a_lifecycle_root",
        "superseded_v8r4a_output_root",
    ],
    "all_fixed_tests_pass": True,
    "four_exact_contract_encoding_regressions_pass": True,
    "dual_root_succession_regressions_pass": True,
    "active_usage_and_execution_ledgers_closed_and_unchanged_during_cpu_tests": True,
    "diagnostic_and_authority_bound_in_every_target_capability": True,
    "superseded_roots_bound_as_denied_canaries_in_every_target_capability": True,
    "gpu_retry_only_after_successor_pretrain_validation": True,
}


def _validate_frozen_contract_encoding_projection(
    authority: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    *,
    enforce_production_literal_bindings: bool = True,
) -> None:
    """Validate CONTRACT1 without opening failed roots or CANARY1 issuance."""

    basis = authority.get("authority_basis")
    rows = authority.get("authorized_modifications")
    mandatory = authority.get("mandatory_invariants")
    forbidden = authority.get("forbidden_changes")
    required = authority.get("required_reauthorization")
    claim = authority.get("claim_boundary")
    failed = diagnostic.get("failed_attempt")
    contract = diagnostic.get("frozen_contract_evidence")
    receipts = diagnostic.get("immutable_failure_receipts")
    namespace = diagnostic.get("failed_namespace_inventory")
    correction = diagnostic.get("required_correction")
    diagnostic_claim = diagnostic.get("claim_boundary")
    dynamic_basis_fields = {
        "parent_canary_boundary_authority",
        "parent_canary_boundary_diagnostic",
        "frozen_campaign_contract",
    }
    if not (
        authority.get("schema_version") == 1
        and diagnostic.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_frozen_contract_exact_byte_encoding_correction_addendum"
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_frozen_contract_encoding_false_rejection_failure_diagnostic"
        and authority.get("campaign_id") == CAMPAIGN_ID
        and diagnostic.get("campaign_id") == CAMPAIGN_ID
        and authority.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and diagnostic.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and authority.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and diagnostic.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and isinstance(basis, Mapping)
        and set(basis)
        == {"diagnostic", *FROZEN_CONTRACT_PARENT_BINDINGS, "user_goal_scope"}
        and (
            not enforce_production_literal_bindings
            or canonical_json_bytes(basis.get("diagnostic"))
            == canonical_json_bytes(FROZEN_CONTRACT_DIAGNOSTIC_LEGACY_BINDING)
        )
        and all(
            canonical_json_bytes(basis.get(field)) == canonical_json_bytes(binding)
            for field, binding in FROZEN_CONTRACT_PARENT_BINDINGS.items()
            if enforce_production_literal_bindings or field not in dynamic_basis_fields
        )
        and isinstance(basis.get("user_goal_scope"), str)
        and bool(basis["user_goal_scope"])
        and isinstance(rows, list)
        and isinstance(mandatory, Mapping)
        and set(mandatory)
        == {
            "scientific_campaign_revision_unchanged",
            "infrastructure_revision_unchanged",
            "variants_unchanged",
            "seeds_unchanged",
            "discovery_outer_folds_unchanged",
            "target_and_outer_reference_sealing_unchanged",
            "gpu_budget_and_append_only_ledgers_unchanged",
            "single_gpu_owner_and_kill_safe_lifecycle_unchanged",
            "exact_frozen_contract_path_sha_bytes_and_content_unchanged",
            "frozen_contract_mode_nlink_and_identity_validation_unchanged",
            "global_noncanonical_governance_rejection_unchanged_except_exact_contract",
            "superseded_canary1_lifecycle_root_preserved_immutable",
            "superseded_canary1_output_root_preserved_immutable",
            "successor_contract1_lifecycle_root",
            "successor_contract1_output_root",
            "both_superseded_roots_denied_unmounted_and_command_inaccessible",
            "failed_capability_and_completion_receipts_preserved_immutable",
            "failed_completion_closed_replay_remains_valid",
            "historical_execution_closure_authority_literal_unchanged",
            "all_preexisting_denied_canaries_mount_checks_command_checks_descriptor_checks_and_sandbox_probes_retained",
            "parent_canary1_chain_preserved_immutable",
        }
        and mandatory.get("scientific_campaign_revision_unchanged") == "V8R4"
        and mandatory.get("infrastructure_revision_unchanged") == "V8R4A"
        and mandatory.get("variants_unchanged")
        == ["H0_no_factor", "H1_factor", "H2_full"]
        and mandatory.get("seeds_unchanged") == list(SEEDS)
        and mandatory.get("discovery_outer_folds_unchanged") == [3, 4]
        and mandatory.get("superseded_canary1_lifecycle_root_preserved_immutable")
        == SUPERSEDED_V8R4A_LIFECYCLE_ROOT_RELATIVE.as_posix()
        and mandatory.get("superseded_canary1_output_root_preserved_immutable")
        == SUPERSEDED_V8R4A_OUTPUT_ROOT_RELATIVE.as_posix()
        and mandatory.get("successor_contract1_lifecycle_root")
        == SUPERSEDED_V8R4A_CONTRACT1_LIFECYCLE_ROOT_RELATIVE.as_posix()
        and mandatory.get("successor_contract1_output_root")
        == SUPERSEDED_V8R4A_CONTRACT1_OUTPUT_ROOT_RELATIVE.as_posix()
        and mandatory.get("historical_execution_closure_authority_literal_unchanged")
        == LEGACY_EXECUTION_CLOSURE_BENCHMARK_OUTPUT_RELATIVE.as_posix()
        and all(
            mandatory.get(field) is True
            for field in (
                "target_and_outer_reference_sealing_unchanged",
                "gpu_budget_and_append_only_ledgers_unchanged",
                "single_gpu_owner_and_kill_safe_lifecycle_unchanged",
                "exact_frozen_contract_path_sha_bytes_and_content_unchanged",
                "frozen_contract_mode_nlink_and_identity_validation_unchanged",
                "global_noncanonical_governance_rejection_unchanged_except_exact_contract",
                "both_superseded_roots_denied_unmounted_and_command_inaccessible",
                "failed_capability_and_completion_receipts_preserved_immutable",
                "failed_completion_closed_replay_remains_valid",
                "all_preexisting_denied_canaries_mount_checks_command_checks_descriptor_checks_and_sandbox_probes_retained",
                "parent_canary1_chain_preserved_immutable",
            )
        )
        and isinstance(forbidden, Mapping)
        and set(forbidden)
        == {
            "model_architecture_or_loss",
            "hyperparameters_or_epoch_counts",
            "data_rows_splits_or_pack_bytes",
            "seed_variant_or_fold_matrix",
            "metric_or_selection_thresholds",
            "outer_target_or_reference_access",
            "global_relaxation_of_json_encoding_validation",
            "contract_reformat_rewrite_copy_or_replacement",
            "acceptance_by_semantic_hash_without_exact_file_sha_and_bytes",
            "reuse_of_superseded_lifecycle_root",
            "reuse_or_mutation_of_superseded_output_root",
            "deletion_or_replacement_of_failed_receipts",
            "reinterpretation_of_frozen_execution_closure_authority",
            "removing_or_weakening_any_denied_canary",
            "removing_mount_command_descriptor_or_sandbox_checks",
            "ledger_reset_truncation_or_rewrite",
            "mutation_or_replacement_of_parent_evidence",
            "commercial_or_confirmatory_claim",
        }
        and all(value is True for value in forbidden.values())
        and canonical_json_bytes(required)
        == canonical_json_bytes(FROZEN_CONTRACT_REQUIRED_REAUTHORIZATION)
        and isinstance(claim, Mapping)
        and claim.get("adaptive_retrospective_only") is True
        and claim.get("correction_is_infrastructure_only") is True
        and claim.get("outer_test_features_or_targets_opened") is False
        and claim.get("accuracy_metric_used") is False
        and claim.get("gpu_execution_authorized_by_this_document") is False
        and claim.get("successor_pretrain_authorization_required") is True
        and claim.get("commercial_claim_authorized") is False
        and isinstance(failed, Mapping)
        and failed.get("first_phase") == "efficiency_benchmark"
        and failed.get("outer_fold") == 3
        and failed.get("target_runtime_child_return_code") == 1
        and failed.get("coordinator_return_code") == 79
        and failed.get("target_sandbox_child_launched") is True
        and failed.get("gpu_wrapper_reached") is False
        and failed.get("gpu_admission_reached") is False
        and failed.get("training_reached") is False
        and failed.get("gpu_usage_ledger_mutated") is False
        and failed.get("gpu_execution_ledger_mutated") is False
        and failed.get("benchmark_output_files_created") is False
        and isinstance(contract, Mapping)
        and contract.get("path") == CAMPAIGN_CONTRACT_RELATIVE_PATH.as_posix()
        and contract.get("file_sha256")
        == FROZEN_CAMPAIGN_CONTRACT_EVIDENCE_BINDING[
            "file_sha256"
        ]
        and contract.get("bytes")
        == FROZEN_CAMPAIGN_CONTRACT_EVIDENCE_BINDING["bytes"]
        and contract.get("content_sha256")
        == FROZEN_CAMPAIGN_CONTRACT_EVIDENCE_BINDING[
            "content_sha256"
        ]
        and contract.get("mode") == "0444"
        and contract.get("valid_unique_key_finite_json") is True
        and contract.get("semantic_content_hash_valid") is True
        and contract.get("exact_file_binding_valid") is True
        and contract.get("roundtrip_differs_from_frozen_bytes") is True
        and contract.get("may_be_rewritten_or_reformatted") is False
        and isinstance(receipts, Mapping)
        and isinstance(namespace, Mapping)
        and namespace.get("lifecycle_root")
        == SUPERSEDED_V8R4A_LIFECYCLE_ROOT_RELATIVE.as_posix()
        and namespace.get("benchmark_output_root")
        == SUPERSEDED_V8R4A_OUTPUT_ROOT_RELATIVE.as_posix()
        and namespace.get("completion_receipt_binds_live_output_inventory") is True
        and namespace.get("adding_success_artifacts_under_old_output_root_would_invalidate_failure_evidence")
        is True
        and namespace.get("old_lifecycle_and_output_roots_must_be_preserved_and_denied_to_successor")
        is True
        and isinstance(correction, Mapping)
        and correction.get("successor_lifecycle_root")
        == SUPERSEDED_V8R4A_CONTRACT1_LIFECYCLE_ROOT_RELATIVE.as_posix()
        and correction.get("successor_benchmark_output_root")
        == SUPERSEDED_V8R4A_CONTRACT1_OUTPUT_ROOT_RELATIVE.as_posix()
        and correction.get("preserve_frozen_execution_closure_authority_historical_output_literal")
        == LEGACY_EXECUTION_CLOSURE_BENCHMARK_OUTPUT_RELATIVE.as_posix()
        and correction.get("new_test_receipt")
        == FROZEN_CONTRACT_ACTIVE_FILENAMES["implementation_test_receipt"]
        and correction.get("new_source_snapshot")
        == FROZEN_CONTRACT_ACTIVE_FILENAMES["source_snapshot"]
        and correction.get("new_pretrain_authorization")
        == FROZEN_CONTRACT_ACTIVE_FILENAMES["active_authorization"]
        and correction.get("deny_and_unmount_superseded_lifecycle_root") is True
        and correction.get("deny_and_unmount_superseded_output_root") is True
        and correction.get("full_reauthorization_before_gpu_retry") is True
        and isinstance(diagnostic_claim, Mapping)
        and diagnostic_claim.get("adaptive_retrospective_only") is True
        and diagnostic_claim.get("outer_test_features_or_targets_opened") is False
        and diagnostic_claim.get("accuracy_metric_computed") is False
        and diagnostic_claim.get("gpu_accessed") is False
        and diagnostic_claim.get("scientific_configuration_change_authorized")
        is False
        and diagnostic_claim.get("commercial_claim_authorized") is False
    ):
        raise TargetSealedError("V8R4A frozen-contract projection drifted")

    observed: dict[str, str] = {}
    for row in rows:
        if not (
            isinstance(row, Mapping)
            and set(row) == {"path", "before_sha256", "allowed_change"}
            and isinstance(row.get("path"), str)
            and isinstance(row.get("before_sha256"), str)
            and isinstance(row.get("allowed_change"), str)
            and bool(row["allowed_change"])
        ):
            raise TargetSealedError("V8R4A frozen-contract modification row drifted")
        path = str(row["path"])
        if path in observed:
            raise TargetSealedError("V8R4A frozen-contract modification is duplicated")
        observed[path] = str(row["before_sha256"])
    if observed != FROZEN_CONTRACT_AUTHORIZED_BEFORE:
        raise TargetSealedError("V8R4A frozen-contract modification cover drifted")

    for receipt_role, diagnostic_role in (
        ("failed_capability_receipt", "capability_receipt"),
        ("failed_completion_receipt", "completion_receipt"),
    ):
        row = receipts.get(diagnostic_role)
        expected = FROZEN_CONTRACT_PARENT_BINDINGS[receipt_role]
        if not (
            isinstance(row, Mapping)
            and row.get("path") == expected["path"]
            and row.get("file_sha256") == expected["file_sha256"]
            and row.get("bytes") == expected["bytes"]
            and row.get("content_sha256") == expected["content_sha256"]
            and row.get("mode") == "0444"
        ):
            raise TargetSealedError(
                f"V8R4A frozen-contract {diagnostic_role} drifted"
            )
    completion = receipts["completion_receipt"]
    if not (
        completion.get("return_code") == 1
        and completion.get("closed_replay_validated") is True
        and receipts.get("same_exact_lifecycle_replays_recorded_return_code") is True
        and receipts.get("mutation_replacement_or_deletion_allowed") is False
    ):
        raise TargetSealedError("V8R4A frozen-contract failed replay drifted")


GPU_STATE_PARENT_BIND_AUTHORITY_FILE_SHA256: Final[str] = (
    "b73d68199acad6fff780c76f05bd3daadc62b03c160af6efc407792efa87a4cd"
)
GPU_STATE_PARENT_BIND_AUTHORITY_BYTES: Final[int] = 14858
GPU_STATE_PARENT_BIND_DIAGNOSTIC_FILE_SHA256: Final[str] = (
    "fd2aa011020289d94ab0548c679888ecc924d3ec179ca5908560a6e82074d628"
)
GPU_STATE_PARENT_BIND_DIAGNOSTIC_BYTES: Final[int] = 10709
GPU_STATE_PARENT_BIND_DIAGNOSTIC_LEGACY_BINDING: Final[Mapping[str, Any]] = {
    "path": (
        "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "diagnostics/v3r1_v8r4a_gpu_state_parent_mount_identity_failure.json"
    ),
    "file_sha256": GPU_STATE_PARENT_BIND_DIAGNOSTIC_FILE_SHA256,
    "bytes": GPU_STATE_PARENT_BIND_DIAGNOSTIC_BYTES,
    "content_sha256": (
        "e4a315bc83e333d31920baef4c3db0f8cb2adc3b5f7e59b73a39795986073b67"
    ),
}
GPU_STATE_PARENT_BIND_PARENT_BINDINGS: Final[Mapping[str, Mapping[str, Any]]] = {
    "parent_frozen_contract_authority": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_FROZEN_CONTRACT_ENCODING.json",
        "file_sha256": FROZEN_CONTRACT_AUTHORITY_FILE_SHA256,
        "bytes": FROZEN_CONTRACT_AUTHORITY_BYTES,
        "content_sha256": "b0df4b5d34bb5f55c6254d83459f81ee297177909658d101866d9e32c6c48c6f",
    },
    "parent_frozen_contract_diagnostic": FROZEN_CONTRACT_DIAGNOSTIC_LEGACY_BINDING,
    "parent_implementation_test_receipt": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTRACT1.json",
        "file_sha256": "ccf4f35c817b7540fb13c760712bc61c471fbcbbea31994ab50e74cb1863c23a",
        "bytes": 30657,
        "content_sha256": "195e1df10a94705df25aaf39adc5c001c2992c4fd013958520d355065d18c0ba",
    },
    "parent_source_snapshot": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/V3R1_SOURCE_SNAPSHOT_V8R4A_CONTRACT1.json",
        "file_sha256": "6841386e58b0390689f60cfddd58f54fb688132f2119afcf8a592fee2d68c1d2",
        "bytes": 23133,
        "content_sha256": "897e1c9e91086820c1eef6529739571acc9bc1802867537b0c9471a6694f4cee",
    },
    "parent_pretrain_authorization": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/PRETRAIN_AUTHORIZATION_V8R4A_CONTRACT1.json",
        "file_sha256": "6c8f072481cbcf5b5ac7971a2c3eeb8c4b7d2cf8ae84a946583294d1ec68584d",
        "bytes": 12666,
        "content_sha256": "ffa7e8e7abaac670961363748896ba0f01da7b01ef662785258e0c603bf6199d",
    },
    "frozen_campaign_contract": FROZEN_CAMPAIGN_CONTRACT_EVIDENCE_BINDING,
    "gpu_state_migration_receipt": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/GPU_STATE_MIGRATION_RECEIPT_V8R4A.json",
        "file_sha256": "d70c921eba40907c76122a8492841d1f490abae4cb4c20058dc340f829582f31",
        "bytes": 14926,
        "content_sha256": "e73b38b390d00533243b23670e3ddbe3ec41461d8dab5336c4ff36b10431328b",
    },
    "failed_contract1_capability_receipt": {
        "path": "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/target_sealed_lifecycle_v8r4a_contract1/efficiency_benchmark/benchmark_hfr_v3r1_efficiency/outer_3/TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json",
        "file_sha256": "13a1fe9e922a4c945e32ac756cc29fe054b36f958c7163fc460932408b086c0f",
        "bytes": 52334,
        "content_sha256": "513c5ecd976254be018105b209bfb707fd7163937dcef174ea13500467846fb3",
    },
    "failed_contract1_completion_receipt": {
        "path": "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/target_sealed_lifecycle_v8r4a_contract1/efficiency_benchmark/benchmark_hfr_v3r1_efficiency/outer_3/TARGET_SEALED_COMPLETION_RECEIPT_V8R4A.json",
        "file_sha256": "6c54435f50ab6f1157894b22a5a94c0b76d33e75a66d762e83a29c29a8bb6f91",
        "bytes": 7690,
        "content_sha256": "58f190e5a86aa2fc7f6a821eb7a6b37dd4f5904e701fd4bce33f1e34756e2270",
    },
}
GPU_STATE_PARENT_BIND_AUTHORIZED_BEFORE: Final[Mapping[str, str]] = {
    "scripts/run_hfr_v3r1_target_sealed.py": "d84ad39e62c2e1087d00ccb818d3bc6b456ef7514c74da7161718ec18629da56",
    "tests/test_run_hfr_v3r1_target_sealed.py": "bbc9cf7e48afccff9ff3e51ac7da3db7f36f336c94b65717f058530c01e51678",
    "scripts/validate_hfr_v3r1_authorization.py": "e0d3e6b3ed38183ecf42b78de911aa9601d10a0764cd729fa16574a2cf441900",
    "tests/test_validate_hfr_v3r1_authorization_v8r4a.py": "289f76fde31cd0fc8b93a57ec6a2771f41b922e03390cf6e496810238727a560",
    "scripts/run_hfr_v3r1_v8r4a_campaign.py": "c6a053b2fd0d6e91209f8ed8dde2f7f48f2032451fd0a1c5ceda453b21da9100",
    "tests/test_run_hfr_v3r1_v8r4a_campaign.py": "9f117aaff1054974f3e6202f45412f5e7c05b0f463123de0253b6d08dca9c70b",
    "scripts/benchmark_hfr_v3r1_efficiency.py": "e340387a125b3898d3c626ea2c7f9f8dad12fff633528ab9e1c4df93bcd2417c",
    "tests/test_benchmark_hfr_v3r1_efficiency.py": "ce5c633d7d8f852c6caf142eb4e70c58c89d4b368be029f4a2588d0c30590c60",
    "scripts/train_harmonic_factor_router_snn_v3r1.py": "8a4ebc77eddd6c8440c6d146e9e66ad6f8bdb988dc1e8150f5fda0a44d6cb30a",
    "scripts/select_hfr_v3r1_common_variant.py": "f53a6f77497b1eedaa1ab227ff29f7a78598655526d0064aea5bcaf3ea8dec02",
    "scripts/run_hfr_v3r1_discovery_campaign.py": "4a418d3888e49e58ed433df1db1b91c72bd6f6ebfa81b2fdc515140b43a1b7f8",
    "tests/test_run_hfr_v3r1_campaign.py": "752e049a9781aa55b9682b01af6fca86d9d73b7dcecdc93dc7ed67d7436dd255",
}
GPU_STATE_PARENT_BIND_ACTIVE_FILENAMES: Final[Mapping[str, str]] = {
    "implementation_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_ROOTBIND1.json",
    "source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_ROOTBIND1.json",
    "active_authorization": "PRETRAIN_AUTHORIZATION_V8R4A_ROOTBIND1.json",
}
GPU_STATE_PARENT_BIND_REQUIRED_REAUTHORIZATION: Final[Mapping[str, Any]] = {
    "new_test_receipt": GPU_STATE_PARENT_BIND_ACTIVE_FILENAMES[
        "implementation_test_receipt"
    ],
    "new_source_snapshot": GPU_STATE_PARENT_BIND_ACTIVE_FILENAMES[
        "source_snapshot"
    ],
    "new_pretrain_authorization": GPU_STATE_PARENT_BIND_ACTIVE_FILENAMES[
        "active_authorization"
    ],
    "new_governance_roles": [
        "gpu_state_parent_bind_correction_authorization",
        "gpu_state_parent_bind_failure_diagnostic",
    ],
    "new_denied_canary_roles": [
        "superseded_v8r4a_contract1_lifecycle_root",
        "superseded_v8r4a_contract1_output_root",
    ],
    "required_true_security_boundary": "gpu_state_parent_identity_readonly_bind",
    "all_fixed_tests_pass": True,
    "real_bubblewrap_parent_readonly_child_readwrite_topology_passes": True,
    "negative_mount_abi_and_unrelated_overlap_regressions_pass": True,
    "descriptor_closure_and_mountinfo_regressions_pass": True,
    "dual_generation_root_succession_regressions_pass": True,
    "active_usage_and_execution_ledgers_closed_and_unchanged_during_cpu_tests": True,
    "diagnostic_and_authority_bound_in_every_target_capability": True,
    "all_four_superseded_roots_bound_as_denied_canaries_in_every_target_capability": True,
    "gpu_retry_only_after_successor_pretrain_validation": True,
}


def _validate_gpu_state_parent_bind_projection(
    authority: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    *,
    enforce_production_literal_bindings: bool = True,
) -> None:
    """Validate ROOTBIND1 without opening CONTRACT1 issuance or failed roots."""

    basis = authority.get("authority_basis")
    rows = authority.get("authorized_modifications")
    mandatory = authority.get("mandatory_invariants")
    forbidden = authority.get("forbidden_changes")
    required = authority.get("required_reauthorization")
    claim = authority.get("claim_boundary")
    trusted = diagnostic.get("trusted_host_gpu_state_root")
    failed = diagnostic.get("failed_attempt")
    topology = diagnostic.get("failed_mount_topology")
    correction = diagnostic.get("required_correction")
    diagnostic_claim = diagnostic.get("claim_boundary")
    dynamic_basis = {
        "parent_frozen_contract_authority",
        "parent_frozen_contract_diagnostic",
        "frozen_campaign_contract",
        "gpu_state_migration_receipt",
    }
    if not (
        authority.get("schema_version") == diagnostic.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_gpu_state_parent_readonly_bind_correction_addendum"
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_gpu_state_parent_mount_identity_failure_diagnostic"
        and authority.get("campaign_id") == diagnostic.get("campaign_id") == CAMPAIGN_ID
        and authority.get("scientific_campaign_revision")
        == diagnostic.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and authority.get("infrastructure_revision")
        == diagnostic.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and isinstance(basis, Mapping)
        and set(basis)
        == {"diagnostic", *GPU_STATE_PARENT_BIND_PARENT_BINDINGS, "user_goal_scope"}
        and (
            not enforce_production_literal_bindings
            or canonical_json_bytes(basis.get("diagnostic"))
            == canonical_json_bytes(GPU_STATE_PARENT_BIND_DIAGNOSTIC_LEGACY_BINDING)
        )
        and all(
            canonical_json_bytes(basis.get(field)) == canonical_json_bytes(binding)
            for field, binding in GPU_STATE_PARENT_BIND_PARENT_BINDINGS.items()
            if enforce_production_literal_bindings or field not in dynamic_basis
        )
        and isinstance(basis.get("user_goal_scope"), str)
        and bool(basis["user_goal_scope"])
        and isinstance(rows, list)
    ):
        raise TargetSealedError("V8R4A GPU-state parent-bind projection drifted")

    observed: dict[str, str] = {}
    for row in rows:
        if not (
            isinstance(row, Mapping)
            and set(row) == {"path", "before_sha256", "allowed_change"}
            and isinstance(row.get("path"), str)
            and isinstance(row.get("before_sha256"), str)
            and isinstance(row.get("allowed_change"), str)
            and bool(row["allowed_change"])
            and row["path"] not in observed
        ):
            raise TargetSealedError("V8R4A parent-bind modification cover drifted")
        observed[str(row["path"])] = str(row["before_sha256"])
    if observed != GPU_STATE_PARENT_BIND_AUTHORIZED_BEFORE:
        raise TargetSealedError("V8R4A parent-bind modification cover drifted")

    mandatory_keys = {
        "scientific_campaign_revision_unchanged",
        "infrastructure_revision_unchanged",
        "variants_seeds_folds_hyperparameters_and_metrics_unchanged",
        "target_and_outer_reference_sealing_unchanged",
        "gpu_budget_and_append_only_ledgers_unchanged",
        "single_gpu_owner_and_kill_safe_lifecycle_unchanged",
        "migration_validator_and_migration_receipt_unchanged",
        "gpu_state_root_path",
        "gpu_state_root_exact_mode",
        "gpu_state_root_exact_st_dev",
        "gpu_state_root_exact_st_ino",
        "gpu_state_root_mount_kind",
        "gpu_state_root_mount_precedes_children",
        "gpu_state_root_direct_mutation_denied",
        "gpu_state_mutable_direct_children",
        "exactly_three_mutable_state_directory_mounts",
        "child_atomic_replace_fsync_and_existing_recovery_unchanged",
        "all_other_readonly_writable_overlap_rejected",
        "internal_guard_child_descriptor_set_exactly_zero_one_two",
        "superseded_v8r4a_lifecycle_root_preserved_immutable",
        "superseded_v8r4a_output_root_preserved_immutable",
        "superseded_v8r4a_contract1_lifecycle_root_preserved_immutable",
        "superseded_v8r4a_contract1_output_root_preserved_immutable",
        "successor_rootbind1_lifecycle_root",
        "successor_rootbind1_output_root",
        "all_four_superseded_roots_denied_unmounted_and_command_inaccessible",
        "both_failed_capability_and_completion_generations_preserved_immutable",
        "both_failed_completions_closed_replay_remain_valid",
        "historical_execution_closure_authority_literal_unchanged",
        "all_preexisting_mount_command_descriptor_sandbox_and_denied_canary_checks_retained",
        "parent_contract1_chain_preserved_immutable",
    }
    true_mandatory = mandatory_keys - {
        "scientific_campaign_revision_unchanged",
        "infrastructure_revision_unchanged",
        "gpu_state_root_path",
        "gpu_state_root_exact_mode",
        "gpu_state_root_exact_st_dev",
        "gpu_state_root_exact_st_ino",
        "gpu_state_root_mount_kind",
        "gpu_state_mutable_direct_children",
        "superseded_v8r4a_lifecycle_root_preserved_immutable",
        "superseded_v8r4a_output_root_preserved_immutable",
        "superseded_v8r4a_contract1_lifecycle_root_preserved_immutable",
        "superseded_v8r4a_contract1_output_root_preserved_immutable",
        "successor_rootbind1_lifecycle_root",
        "successor_rootbind1_output_root",
        "historical_execution_closure_authority_literal_unchanged",
    }
    if not (
        isinstance(mandatory, Mapping)
        and set(mandatory) == mandatory_keys
        and mandatory.get("scientific_campaign_revision_unchanged") == "V8R4"
        and mandatory.get("infrastructure_revision_unchanged") == "V8R4A"
        and mandatory.get("gpu_state_root_path") == GPU_STATE_ROOT_RELATIVE.as_posix()
        and mandatory.get("gpu_state_root_exact_mode") == "0700"
        and mandatory.get("gpu_state_root_mount_kind") == "ro_bind_fd"
        and mandatory.get("gpu_state_mutable_direct_children")
        == ["admission", "execution", "usage"]
        and mandatory.get("superseded_v8r4a_lifecycle_root_preserved_immutable")
        == SUPERSEDED_V8R4A_LIFECYCLE_ROOT_RELATIVE.as_posix()
        and mandatory.get("superseded_v8r4a_output_root_preserved_immutable")
        == SUPERSEDED_V8R4A_OUTPUT_ROOT_RELATIVE.as_posix()
        and mandatory.get("superseded_v8r4a_contract1_lifecycle_root_preserved_immutable")
        == SUPERSEDED_V8R4A_CONTRACT1_LIFECYCLE_ROOT_RELATIVE.as_posix()
        and mandatory.get("superseded_v8r4a_contract1_output_root_preserved_immutable")
        == SUPERSEDED_V8R4A_CONTRACT1_OUTPUT_ROOT_RELATIVE.as_posix()
        and mandatory.get("successor_rootbind1_lifecycle_root")
        == SUPERSEDED_V8R4A_ROOTBIND1_LIFECYCLE_ROOT_RELATIVE.as_posix()
        and mandatory.get("successor_rootbind1_output_root")
        == SUPERSEDED_V8R4A_ROOTBIND1_OUTPUT_ROOT_RELATIVE.as_posix()
        and mandatory.get("historical_execution_closure_authority_literal_unchanged")
        == LEGACY_EXECUTION_CLOSURE_BENCHMARK_OUTPUT_RELATIVE.as_posix()
        and all(mandatory.get(field) is True for field in true_mandatory)
        and type(mandatory.get("gpu_state_root_exact_st_dev")) is int
        and type(mandatory.get("gpu_state_root_exact_st_ino")) is int
    ):
        raise TargetSealedError("V8R4A parent-bind mandatory invariants drifted")
    if enforce_production_literal_bindings and not (
        mandatory["gpu_state_root_exact_st_dev"] == GPU_STATE_ROOT_AUTHORIZED_ST_DEV
        and mandatory["gpu_state_root_exact_st_ino"] == GPU_STATE_ROOT_AUTHORIZED_ST_INO
    ):
        raise TargetSealedError("V8R4A parent-bind root identity literal drifted")

    forbidden_keys = {
        "model_architecture_or_loss",
        "hyperparameters_or_epoch_counts",
        "data_rows_splits_or_pack_bytes",
        "seed_variant_or_fold_matrix",
        "metric_or_selection_thresholds",
        "outer_target_or_reference_access",
        "migration_validator_relaxation",
        "gpu_state_root_writable_bind",
        "synthetic_parent_chmod_as_identity_substitute",
        "adding_gpu_state_root_to_writable_roots",
        "allowing_more_or_fewer_than_three_mutable_state_children",
        "generic_readonly_writable_overlap_relaxation",
        "broad_runs_or_campaign_root_mount",
        "reuse_or_mutation_of_any_superseded_lifecycle_or_output_root",
        "deletion_or_replacement_of_any_failed_receipt",
        "reinterpretation_of_frozen_execution_closure_authority",
        "removing_or_weakening_any_denied_canary",
        "ledger_reset_truncation_or_rewrite",
        "mutation_or_replacement_of_parent_evidence",
        "commercial_or_confirmatory_claim",
    }
    if not (
        isinstance(forbidden, Mapping)
        and set(forbidden) == forbidden_keys
        and all(value is True for value in forbidden.values())
        and canonical_json_bytes(required)
        == canonical_json_bytes(GPU_STATE_PARENT_BIND_REQUIRED_REAUTHORIZATION)
        and isinstance(claim, Mapping)
        and claim
        == {
            "adaptive_retrospective_only": True,
            "correction_is_infrastructure_only": True,
            "outer_test_features_or_targets_opened": False,
            "accuracy_metric_used": False,
            "gpu_execution_authorized_by_this_document": False,
            "successor_pretrain_authorization_required": True,
            "commercial_claim_authorized": False,
        }
    ):
        raise TargetSealedError("V8R4A parent-bind authorization boundary drifted")

    if not (
        isinstance(failed, Mapping)
        and failed.get("first_phase") == "efficiency_benchmark"
        and failed.get("outer_fold") == 3
        and failed.get("target_runtime_child_return_code") == 1
        and failed.get("coordinator_return_code") == 79
        and failed.get("target_sandbox_child_launched") is True
        and failed.get("target_scoped_pretrain_validation_reached") is True
        and all(
            failed.get(field) is False
            for field in (
                "gpu_wrapper_reached",
                "gpu_admission_reached",
                "training_reached",
                "accuracy_metric_computed",
                "gpu_usage_ledger_mutated",
                "gpu_execution_ledger_mutated",
                "benchmark_output_files_created",
            )
        )
        and isinstance(trusted, Mapping)
        and trusted.get("path") == GPU_STATE_ROOT_RELATIVE.as_posix()
        and trusted.get("mode") == "0700"
        and trusted.get("exact_entries") == ["admission", "execution", "usage"]
        and trusted.get("st_dev") == mandatory.get("gpu_state_root_exact_st_dev")
        and trusted.get("st_ino") == mandatory.get("gpu_state_root_exact_st_ino")
        and isinstance(topology, Mapping)
        and topology.get("parent_identity_mount_absent") is True
        and topology.get("exactly_three_mutable_child_mounts_present") is True
        and isinstance(correction, Mapping)
        and correction.get("migration_validator_relaxation_allowed") is False
        and correction.get("gpu_state_root_writable_mount_allowed") is False
        and correction.get("synthetic_parent_chmod_substitution_allowed") is False
        and correction.get("exact_parent_readonly_fd_bind_required")
        == {
            "path": GPU_STATE_ROOT_RELATIVE.as_posix(),
            "mode": "0700",
            "st_dev": mandatory.get("gpu_state_root_exact_st_dev"),
            "st_ino": mandatory.get("gpu_state_root_exact_st_ino"),
            "kind": "ro_bind_fd",
        }
        and correction.get("exactly_three_mutable_direct_child_overlays_required")
        == ["admission", "execution", "usage"]
        and correction.get("parent_mount_must_precede_child_overlays") is True
        and correction.get("successor_lifecycle_root")
        == SUPERSEDED_V8R4A_ROOTBIND1_LIFECYCLE_ROOT_RELATIVE.as_posix()
        and correction.get("successor_benchmark_output_root")
        == SUPERSEDED_V8R4A_ROOTBIND1_OUTPUT_ROOT_RELATIVE.as_posix()
        and correction.get("new_test_receipt")
        == GPU_STATE_PARENT_BIND_ACTIVE_FILENAMES["implementation_test_receipt"]
        and correction.get("new_source_snapshot")
        == GPU_STATE_PARENT_BIND_ACTIVE_FILENAMES["source_snapshot"]
        and correction.get("new_pretrain_authorization")
        == GPU_STATE_PARENT_BIND_ACTIVE_FILENAMES["active_authorization"]
        and correction.get("full_reauthorization_before_gpu_retry") is True
        and diagnostic_claim
        == {
            "adaptive_retrospective_only": True,
            "outer_test_features_or_targets_opened": False,
            "accuracy_metric_computed": False,
            "gpu_accessed": False,
            "scientific_configuration_change_authorized": False,
            "commercial_claim_authorized": False,
        }
    ):
        raise TargetSealedError("V8R4A parent-bind failure projection drifted")


BENCHMARK_ADMITTED_CONTEXT_AUTHORITY_FILE_SHA256: Final[str] = (
    "c0646a3fb0e5b673850e570f7d0a1e91676e5116890d1a8e758e6603bbfa31e2"
)
BENCHMARK_ADMITTED_CONTEXT_AUTHORITY_BYTES: Final[int] = 16_684
BENCHMARK_ADMITTED_CONTEXT_AUTHORITY_CONTENT_SHA256: Final[str] = (
    "d48ff6cb78fcf94e6d994cca96b144daca9da19f873bc8f5ef7e15246e6a1f5c"
)
BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_FILE_SHA256: Final[str] = (
    "b7a360902a68c4a7cb72d320c2042bccaf965a6ea9df64b0d203a40dc64dd088"
)
BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_BYTES: Final[int] = 9_019
BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_CONTENT_SHA256: Final[str] = (
    "51ffa6135eec896c385878b42ecd3d6bb440fad5965532d04341cec4cb4eb83e"
)
BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES: Final[Mapping[str, str]] = {
    "implementation_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTEXT1.json",
    "source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTEXT1.json",
    "active_authorization": "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json",
}
BENCHMARK_ADMITTED_CONTEXT_EXPECTED: Final[Mapping[str, Any]] = {
    "campaign_revision": "V8R4",
    "infrastructure_revision": "V8R4A",
    "authorization_generation": "CONTEXT1",
    "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
    "outer_fold": 3,
    "seed": 20260828,
    "variant": "H0_no_factor",
}
BENCHMARK_ADMITTED_CONTEXT_EFFICIENCY_SCOPE: Final[Mapping[str, Any]] = {
    "phase": "efficiency_benchmark",
    **BENCHMARK_ADMITTED_CONTEXT_EXPECTED,
    "epochs": 2,
    "epoch_2_train_plus_target_free_validation_seconds_max": 23.0,
    "accuracy_metrics_authorized": False,
    "required_before_discovery": True,
}
BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES: Final[
    Mapping[str, Mapping[str, Any]]
] = {
    "usage_ledger": {
        "path": (
            GPU_STATE_DIRECTORY_RELATIVE_PATHS["usage"]
            / "campaign_gpu_usage_chain_v6.jsonl"
        ).as_posix(),
        "bytes": 113_257,
        "sha256": (
            "c59c9234c25c09a4cb7e7cc753942e0517acccdced5ba527336295db32dac029"
        ),
        "record_count": 77,
        "tail_record_sha256": (
            "9da404e120c87940617fed8ad8438d9470b785b304fe48c4ec8727d50fcafafd"
        ),
        "settled_usage_ns": 1_411_550_918_574,
        "open_reservation_count": 0,
    },
    "execution_ledger": {
        "path": (
            GPU_STATE_DIRECTORY_RELATIVE_PATHS["execution"]
            / "gpu_execution_ledger_v7.jsonl"
        ).as_posix(),
        "bytes": 29_961,
        "sha256": (
            "8acffd84488a1f9b4f9a0a95804108127135023d0e58c46afe7ffa7b553609b5"
        ),
        "record_count": 10,
        "last_line_sha256": (
            "a2aac7f38810230c332bc6d389d7baf8d83d29bd4cf1d0e940d259bfff3f1272"
        ),
        "open_start_count": 0,
    },
}
BENCHMARK_ADMITTED_CONTEXT_REQUIRED_REAUTHORIZATION: Final[Mapping[str, Any]] = {
    "new_test_receipt": BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES[
        "implementation_test_receipt"
    ],
    "new_source_snapshot": BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES[
        "source_snapshot"
    ],
    "new_pretrain_authorization": BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES[
        "active_authorization"
    ],
    "new_governance_roles": [
        "admitted_context_correction_authorization",
        "admitted_context_failure_diagnostic",
    ],
    "new_denied_canary_roles": [
        "superseded_v8r4a_rootbind1_lifecycle_root",
        "superseded_v8r4a_rootbind1_output_root",
    ],
    "required_true_security_boundary": (
        "benchmark_admitted_context_generation_isolated"
    ),
    "all_fixed_tests_pass": True,
    "programmatic_benchmark_bridge_exact_argument_and_order_tests_pass": True,
    "missing_extra_projected_or_binding_substituted_context_tests_fail_closed": True,
    "rootbind1_failure_prefix_hash_context_path_return_code_and_order_tests_pass": True,
    "global_and_pack_free_ledger_reconciliation_accounts_failed_charge": True,
    "rootbind_real_bubblewrap_topology_and_live_mountinfo_tests_pass": True,
    "all_six_superseded_roots_bound_as_denied_canaries_in_every_target_capability": True,
    "diagnostic_and_authority_bound_in_every_target_capability": True,
    "active_usage_and_execution_ledgers_closed_and_unchanged_during_cpu_tests": True,
    "fresh_context1_roots_absent_before_first_launch": True,
    "successor_pretrain_validation_passes_before_gpu_retry": True,
    "gpu_retry_only_after_full_reauthorization": True,
}
BENCHMARK_ADMITTED_CONTEXT_AUTHORIZED_BEFORE: Final[Mapping[str, str]] = {
    "scripts/run_hfr_v3r1_target_sealed.py": "b9bc11b40cb4f2960decca156afd6ffc827332ad83ece5877ce323c146342f44",
    "tests/test_run_hfr_v3r1_target_sealed.py": "a68a5bdc4b93c4a0b8ab5d591822b6af3739b4ce58b795edbd7db30ba65563d5",
    "scripts/validate_hfr_v3r1_authorization.py": "ff05a132bd74d10ea1a283a78ed1b56ad03b3a250d3638bbe2aaa03aa3f70a60",
    "tests/test_validate_hfr_v3r1_authorization_v8r4a.py": "5ae569ef190cedacdcb434bf6afee42f34fd4502e996a97cee51cf4e4201b781",
    "scripts/run_hfr_v3r1_v8r4a_campaign.py": "3c895c8583eadd573a8e3aca3b252ddea55a2b2358fb1119bb45b996d2c8c886",
    "tests/test_run_hfr_v3r1_v8r4a_campaign.py": "ca635362f4a6bc3683c00f97d8937483b9df4ca1f6ef0f88b82ca333e918beae",
    "scripts/benchmark_hfr_v3r1_efficiency.py": "47de1f7598ae3248b6fba8cbc40abf9e3e748824378d4a7d7ef19d9699f8ceb7",
    "tests/test_benchmark_hfr_v3r1_efficiency.py": "507e12f385f52d639166c030c8c7014d1647a8ee787cf8ace6096e61debdde61",
    "scripts/train_harmonic_factor_router_snn_v3r1.py": "6b0841314f2d41b35f2f23a68d993611f78aa86a0775cdd099e5440567a37a0c",
    "tests/test_train_harmonic_factor_router_snn_v3r1.py": "46d9810a4bd20b68cc10313b95896f880898b0ad6bc37e3444526180bb023793",
    "scripts/select_hfr_v3r1_common_variant.py": "9d86057fcec49372b38a48788b79b0287d1285359b87a1939ad92aebfd01dcdc",
    "scripts/run_hfr_v3r1_discovery_campaign.py": "7722670259ce753683c3370acde5e6da20b40723f7d43ad25d998f4f82df7fb1",
    "tests/test_run_hfr_v3r1_campaign.py": "46415f1dd61b8abbf599c8b60274b7b6e3f473995cc46012c72606346336de0a",
}
BENCHMARK_ADMITTED_CONTEXT_PINNED_UNMOUNTED_BASIS: Final[
    Mapping[str, Mapping[str, Any]]
] = {
    "parent_implementation_test_receipt": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/IMPLEMENTATION_TEST_RECEIPT_V8R4A_ROOTBIND1.json",
        "sha256": "5654eb89eab4ccb97f20633dbe1832e8600694312d3b52162c0d8f1711f57ec5",
        "bytes": 31_505,
        "mode": "0444",
        "content_sha256": "623d0c7c86274c04f3d0f38b8032485ae2e403461f1de44d25beff6f9368c726",
    },
    "parent_source_snapshot": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/V3R1_SOURCE_SNAPSHOT_V8R4A_ROOTBIND1.json",
        "sha256": "8ea5873fd2ebf43d975db123f4551e7d3aa849ff4aa404dfb5c862c23b735cae",
        "bytes": 23_900,
        "mode": "0444",
        "content_sha256": "c6d3819b3b9b52a6ec2ed6d2139eb0bb2b2b1768b97fc167b8f67cb62d4691a4",
    },
    "parent_pretrain_authorization": {
        "path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/PRETRAIN_AUTHORIZATION_V8R4A_ROOTBIND1.json",
        "sha256": "49ba2637b9c957d382f83c8847198f129eda2f08c099c184f717003a1129fba6",
        "bytes": 13_433,
        "mode": "0444",
        "content_sha256": "94dd087b03af3fb0d9e8e3726b615f003e6c9540d1445f15c315de91c9de873d",
    },
    "failed_rootbind1_target_capability_receipt": {
        "path": "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/target_sealed_lifecycle_v8r4a_rootbind1/efficiency_benchmark/benchmark_hfr_v3r1_efficiency/outer_3/TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json",
        "sha256": "b6ceccf7b4d3f0738de1cbead9038fe937a209295294a4af772a902dccbd20d8",
        "bytes": 54_959,
        "mode": "0444",
        "content_sha256": "0db3ec473188efda5b60f8b635d07e790d632ddca48a91547f61ac6906c41218",
    },
    "failed_rootbind1_target_completion_receipt": {
        "path": "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/target_sealed_lifecycle_v8r4a_rootbind1/efficiency_benchmark/benchmark_hfr_v3r1_efficiency/outer_3/TARGET_SEALED_COMPLETION_RECEIPT_V8R4A.json",
        "sha256": "e08456bcdecddb10e38e7378837f785314dd8494125d2363b1063df7b4723747",
        "bytes": 8_676,
        "mode": "0444",
        "content_sha256": "c4d73ab37fe61b049f59ea00fa81e6dbdcd3585b565319e9326b299959dd12fe",
    },
    "failed_rootbind1_benchmark_invocation": {
        "path": "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8r4a_rootbind1/BENCHMARK_INVOCATION_V8R4.json",
        "sha256": "a52232d9dc4428550039149589d8f0e718b39d146f7d271c195710a26ef8a3f9",
        "bytes": 6_789,
        "mode": "0444",
        "content_sha256": "58ed6edf641aa5d4a559802af966727913a9a7bca5f2cdf3c4f7377efc25f670",
    },
    "failed_rootbind1_gpu_invocation": {
        "path": "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8r4a_rootbind1/attempts/attempt_000/invocation.json",
        "sha256": "e06bbc723706fd6756f3224dc806ac54cfab6fe8f7852da1f7c372f740730961",
        "bytes": 3_189,
        "mode": "0444",
        "content_sha256": "b86e18808c7e3440200dee669fc356aa51876733bdf9d5ee808cc505086cdb9f",
    },
    "failed_rootbind1_gpu_terminal_result": {
        "path": "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/efficiency_benchmark_v8r4a_rootbind1/attempts/attempt_000/GPU_TERMINAL_RESULT.json",
        "sha256": "b575caa298db286cad2a3ad3231aa84dcdb2af76ab04d15ea38eec7b1a50fbda",
        "bytes": 1_832,
        "mode": "0444",
        "content_sha256": "e3ddad8b65692a05cdabc1f74bdd29fc4fdbe86991e0d956ee2ad1af000599e5",
    },
}


def _validate_active_pretrain_live_ledger_prefixes(
    *,
    project_root: Path,
    runtime_ledger_prefixes: object,
    live_state: LiveStateSnapshot | None,
    require_closed: bool,
) -> dict[str, dict[str, Any]]:
    """Prove that the live ledgers retain CONTEXT1's charged failure prefix.

    The active authorization freezes the exact bytes at the ROOTBIND1 failure
    boundary.  Later target-owned recovery and admitted work may only append to
    those bytes.  A host coordinator may call this with ``live_state=None`` to
    enforce the immutable lower bound while leaving a dead open suffix for the
    target runtime to recover.  Every target prelaunch supplies a strict,
    closed migration replay and therefore also binds current file identities,
    counts, and settled usage before GPU admission.
    """

    if not isinstance(runtime_ledger_prefixes, Mapping) or canonical_json_bytes(
        runtime_ledger_prefixes
    ) != canonical_json_bytes(BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES):
        raise TargetSealedError(
            "V8R4A CONTEXT1 active pretrain ledger prefix projection drifted"
        )

    expected_paths = _expected_gpu_state_paths(project_root)
    observed: dict[str, dict[str, Any]] = {}
    terminal_context = dict(BENCHMARK_ADMITTED_CONTEXT_EXPECTED)
    terminal_context.pop("authorization_generation")
    for role in ("usage_ledger", "execution_ledger"):
        expected = BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES[role]
        path = _canonical_existing(
            project_root / str(expected["path"]),
            label=f"CONTEXT1 {role}",
        )
        if path != expected_paths[role]:
            raise TargetSealedError(
                f"V8R4A CONTEXT1 {role} canonical path drifted"
            )
        binding, raw = _read_file_binding(
            path, label=f"CONTEXT1 live {role}", require_immutable=False
        )
        size = int(expected["bytes"])
        if not (
            binding.mode == 0o644
            and binding.bytes >= size
            and hashlib.sha256(raw[:size]).hexdigest() == expected["sha256"]
        ):
            raise TargetSealedError(
                f"V8R4A CONTEXT1 {role} frozen postfailure prefix was truncated or rewritten"
            )
        prefix = raw[:size]
        if not prefix.endswith(b"\n"):
            raise TargetSealedError(
                f"V8R4A CONTEXT1 {role} frozen postfailure prefix boundary drifted"
            )
        lines = prefix.splitlines()
        if len(lines) != expected["record_count"]:
            raise TargetSealedError(
                f"V8R4A CONTEXT1 {role} frozen postfailure record count drifted"
            )
        rows = [
            _decode_json_bytes(
                line,
                label=f"CONTEXT1 {role} frozen prefix record {number}",
            )
            for number, line in enumerate(lines)
        ]
        terminal = rows[-1]
        if not (
            terminal.get("event")
            == ("terminal" if role == "usage_ledger" else "end")
            and terminal.get("phase") == "efficiency_benchmark"
            and terminal.get("context") == terminal_context
        ):
            raise TargetSealedError(
                f"V8R4A CONTEXT1 {role} frozen postfailure tail identity drifted"
            )
        if role == "usage_ledger":
            if terminal.get("record_sha256") != expected["tail_record_sha256"]:
                raise TargetSealedError(
                    "V8R4A CONTEXT1 usage frozen postfailure tail hash drifted"
                )
        elif not (
            hashlib.sha256(lines[-1]).hexdigest() == expected["last_line_sha256"]
            and terminal.get("terminal_record_sha256")
            == BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES["usage_ledger"][
                "tail_record_sha256"
            ]
        ):
            raise TargetSealedError(
                "V8R4A CONTEXT1 execution frozen postfailure tail hash drifted"
            )

        observed_row = {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": binding.sha256,
            "bytes": binding.bytes,
            "mode": f"{binding.mode:04o}",
            "nlink": 1,
            "st_dev": binding.st_dev,
            "st_ino": binding.st_ino,
        }
        observed[role] = observed_row
        if live_state is not None and (
            canonical_json_bytes(live_state.current_file_bindings.get(role))
            != canonical_json_bytes(observed_row)
        ):
            raise TargetSealedError(
                f"V8R4A CONTEXT1 {role} changed after strict live replay"
            )

    if live_state is not None:
        usage = live_state.usage_state
        execution = live_state.execution_state
        if not (
            type(usage.get("record_count")) is int
            and usage["record_count"]
            >= BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES["usage_ledger"][
                "record_count"
            ]
            and type(usage.get("settled_usage_ns")) is int
            and usage["settled_usage_ns"]
            >= BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES["usage_ledger"][
                "settled_usage_ns"
            ]
            and type(execution.get("record_count")) is int
            and execution["record_count"]
            >= BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES["execution_ledger"][
                "record_count"
            ]
        ):
            raise TargetSealedError(
                "V8R4A CONTEXT1 live ledger lower-bound state drifted"
            )
        if require_closed and not (
            usage.get("open_reservation_count") == 0
            and execution.get("open_start_count") == 0
        ):
            raise TargetSealedError(
                "V8R4A CONTEXT1 live ledgers are not closed before GPU admission"
            )
    elif require_closed:
        raise TargetSealedError(
            "V8R4A CONTEXT1 closed-prefix validation lacks strict live replay"
        )
    return observed


def _validate_benchmark_admitted_context_projection(
    authority: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    *,
    enforce_production_literal_bindings: bool = True,
) -> None:
    """Validate CONTEXT1 entirely from mounted documents, never old root paths."""

    basis = authority.get("authority_basis")
    modifications = authority.get("authorized_modifications")
    mandatory = authority.get("mandatory_invariants")
    forbidden = authority.get("forbidden_changes")
    required = authority.get("required_reauthorization")
    failed = diagnostic.get("failed_attempt")
    receipts = diagnostic.get("immutable_failure_receipts")
    ledger = diagnostic.get("ledger_evidence")
    root_cause = diagnostic.get("root_cause")
    correction = diagnostic.get("required_correction")
    terminal_receipt = (
        receipts.get("gpu_terminal_result")
        if isinstance(receipts, Mapping)
        else None
    )
    terminal_receipt = (
        terminal_receipt if isinstance(terminal_receipt, Mapping) else {}
    )
    usage_postlaunch = (
        ledger.get("usage_postlaunch") if isinstance(ledger, Mapping) else None
    )
    usage_postlaunch = (
        usage_postlaunch if isinstance(usage_postlaunch, Mapping) else {}
    )
    execution_postlaunch = (
        ledger.get("execution_postlaunch")
        if isinstance(ledger, Mapping)
        else None
    )
    execution_postlaunch = (
        execution_postlaunch if isinstance(execution_postlaunch, Mapping) else {}
    )
    expected_superseded = dict(BENCHMARK_ADMITTED_CONTEXT_EXPECTED)
    expected_superseded.pop("authorization_generation")
    if not (
        authority.get("schema_version") == diagnostic.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_correction_addendum"
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_failure_diagnostic"
        and authority.get("campaign_id") == diagnostic.get("campaign_id") == CAMPAIGN_ID
        and authority.get("scientific_campaign_revision")
        == diagnostic.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and authority.get("infrastructure_revision")
        == diagnostic.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and isinstance(basis, Mapping)
        and set(basis)
        == {
            "diagnostic",
            "parent_gpu_state_parent_bind_authority",
            "parent_gpu_state_parent_bind_diagnostic",
            "parent_implementation_test_receipt",
            "parent_source_snapshot",
            "parent_pretrain_authorization",
            "frozen_campaign_contract",
            "gpu_state_migration_receipt",
            *BENCHMARK_ADMITTED_CONTEXT_PINNED_UNMOUNTED_BASIS,
            "user_goal_scope",
        }
        and isinstance(basis.get("user_goal_scope"), str)
        and bool(basis["user_goal_scope"])
        and all(
            canonical_json_bytes(basis.get(role)) == canonical_json_bytes(row)
            for role, row in BENCHMARK_ADMITTED_CONTEXT_PINNED_UNMOUNTED_BASIS.items()
        )
        and isinstance(modifications, list)
    ):
        raise TargetSealedError("V8R4A admitted-context projection drifted")
    observed: dict[str, str] = {}
    for row in modifications:
        if not (
            isinstance(row, Mapping)
            and set(row) == {"path", "before_sha256", "allowed_change"}
            and isinstance(row.get("path"), str)
            and _is_sha256(row.get("before_sha256"))
            and isinstance(row.get("allowed_change"), str)
            and bool(row["allowed_change"])
            and row["path"] not in observed
        ):
            raise TargetSealedError("V8R4A admitted-context modification cover drifted")
        observed[str(row["path"])] = str(row["before_sha256"])
    if observed != BENCHMARK_ADMITTED_CONTEXT_AUTHORIZED_BEFORE:
        raise TargetSealedError("V8R4A admitted-context modification cover drifted")
    if not (
        isinstance(mandatory, Mapping)
        and mandatory.get("active_benchmark_context")
        == BENCHMARK_ADMITTED_CONTEXT_EXPECTED
        and mandatory.get("superseded_rootbind1_context") == expected_superseded
        and mandatory.get("superseded_rootbind1_terminal_record_sha256")
        == "9da404e120c87940617fed8ad8438d9470b785b304fe48c4ec8727d50fcafafd"
        and mandatory.get("usage_postfailure_sha256")
        == "c59c9234c25c09a4cb7e7cc753942e0517acccdced5ba527336295db32dac029"
        and mandatory.get("usage_postfailure_bytes") == 113_257
        and mandatory.get("usage_postfailure_record_count") == 77
        and mandatory.get("usage_postfailure_open_reservation_count") == 0
        and mandatory.get("execution_postfailure_sha256")
        == "8acffd84488a1f9b4f9a0a95804108127135023d0e58c46afe7ffa7b553609b5"
        and mandatory.get("execution_postfailure_bytes") == 29_961
        and mandatory.get("execution_postfailure_record_count") == 10
        and mandatory.get("execution_postfailure_open_start_count") == 0
        and mandatory.get("successor_context1_lifecycle_root")
        == TARGET_LIFECYCLE_ROOT_RELATIVE.as_posix()
        and mandatory.get("successor_context1_output_root")
        == BENCHMARK_OUTPUT_RELATIVE.as_posix()
        and mandatory.get("all_six_superseded_roots_denied_unmounted_and_command_inaccessible")
        is True
        and mandatory.get("rootbind_parent_readonly_and_three_child_readwrite_topology_unchanged")
        is True
        and mandatory.get("trainer_fail_closed_context_free_admitted_validation_retained")
        is True
        and mandatory.get("benchmark_worker_validates_target_scoped_admitted_pretrain_before_primitive")
        is True
        and mandatory.get("new_environment_or_descriptor_context_channel_absent")
        is True
        and canonical_json_bytes(required)
        == canonical_json_bytes(BENCHMARK_ADMITTED_CONTEXT_REQUIRED_REAUTHORIZATION)
        and isinstance(forbidden, Mapping)
        and forbidden
        and all(value is True for value in forbidden.values())
        and authority.get("claim_boundary")
        == {
            "adaptive_retrospective_only": True,
            "correction_is_infrastructure_only": True,
            "outer_test_features_or_targets_opened": False,
            "accuracy_metric_used": False,
            "prior_gpu_admission_and_cuda_availability_probe_recorded": True,
            "prior_model_or_training_kernel_executed": False,
            "gpu_execution_authorized_by_this_document": False,
            "successor_pretrain_authorization_required": True,
            "commercial_claim_authorized": False,
        }
    ):
        raise TargetSealedError("V8R4A admitted-context authorization boundary drifted")
    if not (
        isinstance(failed, Mapping)
        and failed.get("first_phase") == "efficiency_benchmark"
        and failed.get("outer_fold") == 3
        and failed.get("target_runtime_return_code") == 87
        and failed.get("gpu_wrapper_return_code") == 1
        and failed.get("gpu_admission_reached") is True
        and failed.get("admitted_child_binding_consumed_once") is True
        and failed.get("cuda_availability_probe_occurred") is True
        and failed.get("model_constructed") is False
        and failed.get("accuracy_metric_computed") is False
        and isinstance(receipts, Mapping)
        and terminal_receipt.get("terminal_record_sha256")
        == "9da404e120c87940617fed8ad8438d9470b785b304fe48c4ec8727d50fcafafd"
        and isinstance(ledger, Mapping)
        and usage_postlaunch.get("sha256")
        == "c59c9234c25c09a4cb7e7cc753942e0517acccdced5ba527336295db32dac029"
        and usage_postlaunch.get("record_count") == 77
        and usage_postlaunch.get("open_reservation_count") == 0
        and execution_postlaunch.get("sha256")
        == "8acffd84488a1f9b4f9a0a95804108127135023d0e58c46afe7ffa7b553609b5"
        and execution_postlaunch.get("record_count") == 10
        and execution_postlaunch.get("open_start_count") == 0
        and ledger.get("append_only_prefix_preserved") is True
        and ledger.get("both_ledgers_closed_after_failure") is True
        and isinstance(root_cause, Mapping)
        and root_cause.get("benchmark_internal_worker_called_trainer_primitive_without_prevalidated_pretrain")
        is True
        and root_cause.get("trainer_fail_closed_default_correctly_rejected_context_free_admitted_validation")
        is True
        and isinstance(correction, Mapping)
        and correction.get("independent_expected_context")
        == BENCHMARK_ADMITTED_CONTEXT_EXPECTED
        and correction.get("independent_expected_phase") == "efficiency_benchmark"
        and correction.get("independent_expected_outer_fold") == 3
        and correction.get("validated_pretrain_passed_to_primitive") is True
        and correction.get("successor_lifecycle_root")
        == TARGET_LIFECYCLE_ROOT_RELATIVE.as_posix()
        and correction.get("successor_benchmark_output_root")
        == BENCHMARK_OUTPUT_RELATIVE.as_posix()
        and correction.get("deny_and_unmount_all_six_superseded_roots") is True
        and correction.get("new_test_receipt")
        == BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES["implementation_test_receipt"]
        and correction.get("new_source_snapshot")
        == BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES["source_snapshot"]
        and correction.get("new_pretrain_authorization")
        == BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES["active_authorization"]
        and diagnostic.get("claim_boundary")
        == {
            "adaptive_retrospective_only": True,
            "outer_test_features_or_targets_opened": False,
            "accuracy_metric_computed": False,
            "gpu_admission_reached": True,
            "cuda_availability_probe_occurred": True,
            "model_or_training_kernel_executed": False,
            "scientific_configuration_change_authorized": False,
            "commercial_claim_authorized": False,
        }
    ):
        raise TargetSealedError("V8R4A admitted-context failure projection drifted")
    if enforce_production_literal_bindings and not (
        authority.get("content_sha256")
        == BENCHMARK_ADMITTED_CONTEXT_AUTHORITY_CONTENT_SHA256
        and diagnostic.get("content_sha256")
        == BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_CONTENT_SHA256
    ):
        raise TargetSealedError("V8R4A admitted-context immutable content drifted")


def _validate_v8r4a_governance_chain(
    *,
    project_root: Path,
    documents: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, FileBinding],
    production: bool,
) -> None:
    test_receipt = documents.get("implementation_test_receipt")
    snapshot = documents.get("source_snapshot")
    pretrain = documents.get("active_authorization")
    parent_authority = documents.get("correction_authorization")
    infrastructure_authority = documents.get("infrastructure_correction_authorization")
    source_closure_authority = documents.get(
        "source_closure_correction_authorization"
    )
    source_closure_dependency_authority = documents.get(
        "source_closure_dependency_authorization"
    )
    kill_safe_authority = documents.get("kill_safe_correction_authorization")
    open_lifecycle_authority = documents.get(
        "open_lifecycle_recovery_correction_authorization"
    )
    execution_closure_authority = documents.get(
        "execution_closure_correction_authorization"
    )
    migration_source_succession_authority = documents.get(
        "migration_source_succession_correction_authorization"
    )
    fd_closure_authority = documents.get("fd_closure_correction_authorization")
    canary_boundary_authority = documents.get(
        "canary_boundary_correction_authorization"
    )
    frozen_contract_authority = documents.get(
        "frozen_contract_encoding_correction_authorization"
    )
    gpu_state_parent_bind_authority = documents.get(
        "gpu_state_parent_bind_correction_authorization"
    )
    admitted_context_authority = documents.get(
        "admitted_context_correction_authorization"
    )
    migration_receipt = documents.get("gpu_state_migration_receipt")
    historical_diagnostic = documents.get("failure_diagnostic")
    diagnostic = documents.get("infrastructure_failure_diagnostic")
    source_closure_diagnostic = documents.get("source_closure_failure_diagnostic")
    kill_safe_diagnostic = documents.get("kill_safe_failure_diagnostic")
    open_lifecycle_diagnostic = documents.get(
        "open_lifecycle_recovery_failure_diagnostic"
    )
    execution_closure_diagnostic = documents.get(
        "execution_closure_failure_diagnostic"
    )
    migration_source_succession_diagnostic = documents.get(
        "migration_source_succession_failure_diagnostic"
    )
    fd_closure_diagnostic = documents.get("fd_closure_failure_diagnostic")
    canary_boundary_diagnostic = documents.get(
        "canary_boundary_failure_diagnostic"
    )
    frozen_contract_diagnostic = documents.get(
        "frozen_contract_encoding_failure_diagnostic"
    )
    gpu_state_parent_bind_diagnostic = documents.get(
        "gpu_state_parent_bind_failure_diagnostic"
    )
    admitted_context_diagnostic = documents.get(
        "admitted_context_failure_diagnostic"
    )
    if any(
        not isinstance(value, Mapping)
        for value in (
            test_receipt,
            snapshot,
            pretrain,
            parent_authority,
            infrastructure_authority,
            source_closure_authority,
            source_closure_dependency_authority,
            kill_safe_authority,
            open_lifecycle_authority,
            execution_closure_authority,
            migration_source_succession_authority,
            fd_closure_authority,
            canary_boundary_authority,
            frozen_contract_authority,
            gpu_state_parent_bind_authority,
            admitted_context_authority,
            migration_receipt,
            historical_diagnostic,
            diagnostic,
            source_closure_diagnostic,
            kill_safe_diagnostic,
            open_lifecycle_diagnostic,
            execution_closure_diagnostic,
            migration_source_succession_diagnostic,
            fd_closure_diagnostic,
            canary_boundary_diagnostic,
            frozen_contract_diagnostic,
            gpu_state_parent_bind_diagnostic,
            admitted_context_diagnostic,
        )
    ):
        raise TargetSealedError("V8R4A governance chain is incomplete")
    assert isinstance(test_receipt, Mapping)
    assert isinstance(snapshot, Mapping)
    assert isinstance(pretrain, Mapping)
    assert isinstance(parent_authority, Mapping)
    assert isinstance(infrastructure_authority, Mapping)
    assert isinstance(source_closure_authority, Mapping)
    assert isinstance(source_closure_dependency_authority, Mapping)
    assert isinstance(kill_safe_authority, Mapping)
    assert isinstance(open_lifecycle_authority, Mapping)
    assert isinstance(execution_closure_authority, Mapping)
    assert isinstance(migration_source_succession_authority, Mapping)
    assert isinstance(fd_closure_authority, Mapping)
    assert isinstance(canary_boundary_authority, Mapping)
    assert isinstance(frozen_contract_authority, Mapping)
    assert isinstance(gpu_state_parent_bind_authority, Mapping)
    assert isinstance(admitted_context_authority, Mapping)
    assert isinstance(migration_receipt, Mapping)
    assert isinstance(historical_diagnostic, Mapping)
    assert isinstance(diagnostic, Mapping)
    assert isinstance(source_closure_diagnostic, Mapping)
    assert isinstance(kill_safe_diagnostic, Mapping)
    assert isinstance(open_lifecycle_diagnostic, Mapping)
    assert isinstance(execution_closure_diagnostic, Mapping)
    assert isinstance(migration_source_succession_diagnostic, Mapping)
    assert isinstance(fd_closure_diagnostic, Mapping)
    assert isinstance(canary_boundary_diagnostic, Mapping)
    assert isinstance(frozen_contract_diagnostic, Mapping)
    assert isinstance(gpu_state_parent_bind_diagnostic, Mapping)
    assert isinstance(admitted_context_diagnostic, Mapping)
    for label, document, classification in (
        (
            "implementation test receipt",
            test_receipt,
            "adaptive_v3r1_v8r4a_implementation_test_receipt",
        ),
        ("source snapshot", snapshot, "adaptive_v3r1_v8r4a_source_snapshot"),
        (
            "active pretrain authorization",
            pretrain,
            "pretrain_adaptive_v3r1_v8r4a_authorization",
        ),
        (
            "GPU-state migration receipt",
            migration_receipt,
            "adaptive_v3r1_v8r4a_gpu_state_migration_receipt",
        ),
    ):
        if not (
            document.get("schema_version") == 1
            and document.get("classification") == classification
            and document.get("campaign_id") == CAMPAIGN_ID
            and document.get("scientific_campaign_revision")
            == SCIENTIFIC_CAMPAIGN_REVISION
            and document.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        ):
            raise TargetSealedError(f"{label} revision/identity drifted")
    for label, document in (
        ("implementation test receipt", test_receipt),
        ("source snapshot", snapshot),
        ("active pretrain authorization", pretrain),
    ):
        generation = document.get("authorization_generation")
        if type(generation) is not str or generation != "CONTEXT1":
            raise TargetSealedError(
                f"{label} authorization generation is not exact CONTEXT1"
            )
    if not (
        parent_authority.get("classification")
        == "posttrain_preselection_adaptive_v3r1_v8r4_physical_target_capability_and_pickle_free_output_correction_authorization"
        and parent_authority.get("campaign_id") == CAMPAIGN_ID
        and historical_diagnostic.get("classification")
        == "posttrain_preselection_v8r4_outer_capability_and_pickle_free_npz_failure_diagnostic"
        and historical_diagnostic.get("campaign_id") == CAMPAIGN_ID
    ):
        raise TargetSealedError("V8R4 correction authority/diagnostic drifted")
    _require_authority_legacy_binding(
        parent_authority.get("diagnostic"),
        binding=bindings["failure_diagnostic"],
        document=historical_diagnostic,
        project_root=project_root,
        label="V8R4 authority diagnostic",
    )
    if not (
        infrastructure_authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_dedicated_gpu_state_directory_migration_correction_authorization"
        and infrastructure_authority.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and infrastructure_authority.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
        and diagnostic.get("classification")
        == "pretrain_v8r4a_dedicated_gpu_state_directory_migration_diagnostic"
        and diagnostic.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and diagnostic.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
    ):
        raise TargetSealedError("V8R4A correction authority/diagnostic drifted")
    if not (
        source_closure_authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_validator_and_executable_source_closure_correction_addendum"
        and source_closure_authority.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and source_closure_authority.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
        and source_closure_diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_validator_deadlock_and_executable_source_closure_failure_diagnostic"
        and source_closure_diagnostic.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and source_closure_diagnostic.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
    ):
        raise TargetSealedError("V8R4A source-closure authority/diagnostic drifted")
    if not (
        source_closure_dependency_authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_executable_source_transitive_dependency_closure_addendum"
        and source_closure_dependency_authority.get(
            "scientific_campaign_revision"
        )
        == SCIENTIFIC_CAMPAIGN_REVISION
        and source_closure_dependency_authority.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
    ):
        raise TargetSealedError("V8R4A source dependency authority drifted")
    if not (
        kill_safe_authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_kill_safe_atomic_output_and_append_only_completion_correction_authorization"
        and kill_safe_authority.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and kill_safe_authority.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
        and kill_safe_diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_kill_safe_output_and_shared_ledger_resume_failure_diagnostic"
        and kill_safe_diagnostic.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and kill_safe_diagnostic.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
    ):
        raise TargetSealedError("V8R4A kill-safe authority/diagnostic drifted")
    if not (
        open_lifecycle_authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_open_gpu_lifecycle_kill_recovery_correction_addendum"
        and open_lifecycle_authority.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and open_lifecycle_authority.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
        and open_lifecycle_diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_open_gpu_lifecycle_kill_recovery_failure_diagnostic"
        and open_lifecycle_diagnostic.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and open_lifecycle_diagnostic.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
    ):
        raise TargetSealedError(
            "V8R4A open-lifecycle recovery authority/diagnostic drifted"
        )
    if not (
        execution_closure_authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_kill_safe_capability_and_promotion_execution_closure_correction_addendum"
        and execution_closure_authority.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and execution_closure_authority.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
        and execution_closure_diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_terminal_execution_closure_failure_diagnostic"
        and execution_closure_diagnostic.get("scientific_campaign_revision")
        == SCIENTIFIC_CAMPAIGN_REVISION
        and execution_closure_diagnostic.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
    ):
        raise TargetSealedError(
            "V8R4A execution-closure authority/diagnostic drifted"
        )
    _validate_execution_closure_historical_projection(execution_closure_authority)
    if not (
        migration_source_succession_authority.get("schema_version") == 1
        and migration_source_succession_authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_correction_addendum"
        and migration_source_succession_authority.get("campaign_id") == CAMPAIGN_ID
        and migration_source_succession_authority.get(
            "scientific_campaign_revision"
        ) == SCIENTIFIC_CAMPAIGN_REVISION
        and migration_source_succession_authority.get("infrastructure_revision")
        == INFRASTRUCTURE_REVISION
        and migration_source_succession_diagnostic.get("schema_version") == 1
        and migration_source_succession_diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_failure_diagnostic"
        and migration_source_succession_diagnostic.get("campaign_id") == CAMPAIGN_ID
        and migration_source_succession_diagnostic.get(
            "scientific_campaign_revision"
        ) == SCIENTIFIC_CAMPAIGN_REVISION
        and migration_source_succession_diagnostic.get("status")
        == "diagnosed_not_authorized_by_diagnostic"
    ):
        raise TargetSealedError(
            "V8R4A migrated-state source-succession authority/diagnostic drifted"
        )
    if production and not (
        bindings["fd_closure_correction_authorization"].sha256
        == FD_CLOSURE_AUTHORITY_FILE_SHA256
        and bindings["fd_closure_correction_authorization"].bytes
        == FD_CLOSURE_AUTHORITY_BYTES
        and bindings["fd_closure_failure_diagnostic"].sha256
        == FD_CLOSURE_DIAGNOSTIC_FILE_SHA256
        and bindings["fd_closure_failure_diagnostic"].bytes
        == FD_CLOSURE_DIAGNOSTIC_BYTES
    ):
        raise TargetSealedError("V8R4A FD-closure immutable binding drifted")
    _validate_fd_closure_projection(
        fd_closure_authority,
        fd_closure_diagnostic,
        enforce_production_literal_bindings=production,
    )
    fd_closure_basis = fd_closure_authority.get("authority_basis")
    assert isinstance(fd_closure_basis, Mapping)
    _require_authority_legacy_binding(
        fd_closure_basis.get("diagnostic"),
        binding=bindings["fd_closure_failure_diagnostic"],
        document=fd_closure_diagnostic,
        project_root=project_root,
        label="V8R4A FD-closure diagnostic",
    )
    _require_authority_legacy_binding(
        fd_closure_basis.get("parent_execution_closure_authority"),
        binding=bindings["execution_closure_correction_authorization"],
        document=execution_closure_authority,
        project_root=project_root,
        label="V8R4A FD-closure execution-closure parent",
    )
    if any(
        bindings[role].path.name != filename
        or bindings[role].path.parent
        != bindings["gpu_state_parent_bind_correction_authorization"].path.parent
        for role, filename in BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES.items()
    ):
        raise TargetSealedError("V8R4A CONTEXT1 active issuance path drifted")
    if production and not (
        bindings["canary_boundary_correction_authorization"].sha256
        == CANARY_BOUNDARY_AUTHORITY_FILE_SHA256
        and bindings["canary_boundary_correction_authorization"].bytes
        == CANARY_BOUNDARY_AUTHORITY_BYTES
        and bindings["canary_boundary_failure_diagnostic"].sha256
        == CANARY_BOUNDARY_DIAGNOSTIC_FILE_SHA256
        and bindings["canary_boundary_failure_diagnostic"].bytes
        == CANARY_BOUNDARY_DIAGNOSTIC_BYTES
    ):
        raise TargetSealedError("V8R4A canary-boundary immutable binding drifted")
    _validate_canary_boundary_projection(
        canary_boundary_authority,
        canary_boundary_diagnostic,
        enforce_production_literal_bindings=production,
    )
    canary_boundary_basis = canary_boundary_authority.get("authority_basis")
    assert isinstance(canary_boundary_basis, Mapping)
    _require_authority_legacy_binding(
        canary_boundary_basis.get("diagnostic"),
        binding=bindings["canary_boundary_failure_diagnostic"],
        document=canary_boundary_diagnostic,
        project_root=project_root,
        label="V8R4A canary-boundary diagnostic",
    )
    _require_authority_legacy_binding(
        canary_boundary_basis.get("parent_fd_closure_authority"),
        binding=bindings["fd_closure_correction_authorization"],
        document=fd_closure_authority,
        project_root=project_root,
        label="V8R4A canary-boundary FD-closure parent",
    )
    if production and not (
        bindings["frozen_contract_encoding_correction_authorization"].sha256
        == FROZEN_CONTRACT_AUTHORITY_FILE_SHA256
        and bindings["frozen_contract_encoding_correction_authorization"].bytes
        == FROZEN_CONTRACT_AUTHORITY_BYTES
        and bindings["frozen_contract_encoding_failure_diagnostic"].sha256
        == FROZEN_CONTRACT_DIAGNOSTIC_FILE_SHA256
        and bindings["frozen_contract_encoding_failure_diagnostic"].bytes
        == FROZEN_CONTRACT_DIAGNOSTIC_BYTES
    ):
        raise TargetSealedError("V8R4A frozen-contract immutable binding drifted")
    _validate_frozen_contract_encoding_projection(
        frozen_contract_authority,
        frozen_contract_diagnostic,
        enforce_production_literal_bindings=production,
    )
    frozen_contract_basis = frozen_contract_authority.get("authority_basis")
    assert isinstance(frozen_contract_basis, Mapping)
    for field, target_role, target_document, label in (
        (
            "diagnostic",
            "frozen_contract_encoding_failure_diagnostic",
            frozen_contract_diagnostic,
            "V8R4A frozen-contract diagnostic",
        ),
        (
            "parent_canary_boundary_authority",
            "canary_boundary_correction_authorization",
            canary_boundary_authority,
            "V8R4A frozen-contract canary-boundary parent authority",
        ),
        (
            "parent_canary_boundary_diagnostic",
            "canary_boundary_failure_diagnostic",
            canary_boundary_diagnostic,
            "V8R4A frozen-contract canary-boundary parent diagnostic",
        ),
        (
            "frozen_campaign_contract",
            "campaign_contract",
            documents["campaign_contract"],
            "V8R4A frozen-contract campaign contract",
        ),
    ):
        _require_authority_legacy_binding(
            frozen_contract_basis.get(field),
            binding=bindings[target_role],
            document=target_document,
            project_root=project_root,
            label=label,
        )
    if production and not (
        bindings["gpu_state_parent_bind_correction_authorization"].sha256
        == GPU_STATE_PARENT_BIND_AUTHORITY_FILE_SHA256
        and bindings["gpu_state_parent_bind_correction_authorization"].bytes
        == GPU_STATE_PARENT_BIND_AUTHORITY_BYTES
        and bindings["gpu_state_parent_bind_failure_diagnostic"].sha256
        == GPU_STATE_PARENT_BIND_DIAGNOSTIC_FILE_SHA256
        and bindings["gpu_state_parent_bind_failure_diagnostic"].bytes
        == GPU_STATE_PARENT_BIND_DIAGNOSTIC_BYTES
    ):
        raise TargetSealedError("V8R4A GPU-state parent-bind immutable binding drifted")
    _validate_gpu_state_parent_bind_projection(
        gpu_state_parent_bind_authority,
        gpu_state_parent_bind_diagnostic,
        enforce_production_literal_bindings=production,
    )
    gpu_state_parent_basis = gpu_state_parent_bind_authority.get("authority_basis")
    assert isinstance(gpu_state_parent_basis, Mapping)
    for field, target_role, target_document, label in (
        (
            "diagnostic",
            "gpu_state_parent_bind_failure_diagnostic",
            gpu_state_parent_bind_diagnostic,
            "V8R4A GPU-state parent-bind diagnostic",
        ),
        (
            "parent_frozen_contract_authority",
            "frozen_contract_encoding_correction_authorization",
            frozen_contract_authority,
            "V8R4A GPU-state parent-bind frozen-contract authority",
        ),
        (
            "parent_frozen_contract_diagnostic",
            "frozen_contract_encoding_failure_diagnostic",
            frozen_contract_diagnostic,
            "V8R4A GPU-state parent-bind frozen-contract diagnostic",
        ),
        (
            "frozen_campaign_contract",
            "campaign_contract",
            documents["campaign_contract"],
            "V8R4A GPU-state parent-bind campaign contract",
        ),
        (
            "gpu_state_migration_receipt",
            "gpu_state_migration_receipt",
            migration_receipt,
            "V8R4A GPU-state parent-bind migration receipt",
        ),
    ):
        _require_authority_legacy_binding(
            gpu_state_parent_basis.get(field),
            binding=bindings[target_role],
            document=target_document,
            project_root=project_root,
            label=label,
        )
    if production and not (
        bindings["admitted_context_correction_authorization"].sha256
        == BENCHMARK_ADMITTED_CONTEXT_AUTHORITY_FILE_SHA256
        and bindings["admitted_context_correction_authorization"].bytes
        == BENCHMARK_ADMITTED_CONTEXT_AUTHORITY_BYTES
        and bindings["admitted_context_failure_diagnostic"].sha256
        == BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_FILE_SHA256
        and bindings["admitted_context_failure_diagnostic"].bytes
        == BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_BYTES
    ):
        raise TargetSealedError(
            "V8R4A benchmark admitted-context immutable binding drifted"
        )
    _validate_benchmark_admitted_context_projection(
        admitted_context_authority,
        admitted_context_diagnostic,
        enforce_production_literal_bindings=production,
    )
    admitted_context_basis = admitted_context_authority.get("authority_basis")
    assert isinstance(admitted_context_basis, Mapping)
    _require_exact_governance_binding(
        admitted_context_basis.get("diagnostic"),
        binding=bindings["admitted_context_failure_diagnostic"],
        project_root=project_root,
        label="V8R4A benchmark admitted-context diagnostic",
    )
    for field, target_role, target_document, label in (
        (
            "parent_gpu_state_parent_bind_authority",
            "gpu_state_parent_bind_correction_authorization",
            gpu_state_parent_bind_authority,
            "V8R4A admitted-context parent-bind authority",
        ),
        (
            "parent_gpu_state_parent_bind_diagnostic",
            "gpu_state_parent_bind_failure_diagnostic",
            gpu_state_parent_bind_diagnostic,
            "V8R4A admitted-context parent-bind diagnostic",
        ),
        (
            "frozen_campaign_contract",
            "campaign_contract",
            documents["campaign_contract"],
            "V8R4A admitted-context campaign contract",
        ),
        (
            "gpu_state_migration_receipt",
            "gpu_state_migration_receipt",
            migration_receipt,
            "V8R4A admitted-context migration receipt",
        ),
    ):
        _require_authority_sha_binding(
            admitted_context_basis.get(field),
            binding=bindings[target_role],
            document=target_document,
            project_root=project_root,
            label=label,
            content_hash_required=field != "frozen_campaign_contract",
        )
    _require_authority_legacy_binding(
        infrastructure_authority.get("diagnostic"),
        binding=bindings["infrastructure_failure_diagnostic"],
        document=diagnostic,
        project_root=project_root,
        label="V8R4A authority diagnostic",
    )
    source_closure_basis = source_closure_authority.get("authority_basis")
    if not isinstance(source_closure_basis, Mapping):
        raise TargetSealedError("V8R4A source-closure authority basis drifted")
    _require_authority_legacy_binding(
        source_closure_basis.get("diagnostic"),
        binding=bindings["source_closure_failure_diagnostic"],
        document=source_closure_diagnostic,
        project_root=project_root,
        label="V8R4A source-closure diagnostic",
    )
    _require_authority_legacy_binding(
        source_closure_basis.get("parent_correction_authorization"),
        binding=bindings["infrastructure_correction_authorization"],
        document=infrastructure_authority,
        project_root=project_root,
        label="V8R4A source-closure parent authority",
    )
    source_dependency_basis = source_closure_dependency_authority.get(
        "authority_basis"
    )
    if not isinstance(source_dependency_basis, Mapping):
        raise TargetSealedError("V8R4A source dependency authority basis drifted")
    _require_authority_legacy_binding(
        source_dependency_basis.get("diagnostic"),
        binding=bindings["source_closure_failure_diagnostic"],
        document=source_closure_diagnostic,
        project_root=project_root,
        label="V8R4A source dependency diagnostic",
    )
    _require_authority_legacy_binding(
        source_dependency_basis.get("parent_source_closure_addendum"),
        binding=bindings["source_closure_correction_authorization"],
        document=source_closure_authority,
        project_root=project_root,
        label="V8R4A source dependency parent authority",
    )
    kill_safe_basis = kill_safe_authority.get("authority_basis")
    if not isinstance(kill_safe_basis, Mapping):
        raise TargetSealedError("V8R4A kill-safe authority basis drifted")
    for field, target_role, target_document, label in (
        (
            "diagnostic",
            "kill_safe_failure_diagnostic",
            kill_safe_diagnostic,
            "V8R4A kill-safe diagnostic",
        ),
        (
            "parent_source_closure_addendum",
            "source_closure_correction_authorization",
            source_closure_authority,
            "V8R4A kill-safe source-closure authority",
        ),
        (
            "transitive_dependency_addendum",
            "source_closure_dependency_authorization",
            source_closure_dependency_authority,
            "V8R4A kill-safe dependency authority",
        ),
    ):
        _require_authority_legacy_binding(
            kill_safe_basis.get(field),
            binding=bindings[target_role],
            document=target_document,
            project_root=project_root,
            label=label,
        )
    succession_basis = migration_source_succession_authority.get(
        "authority_basis"
    )
    if not isinstance(succession_basis, Mapping) or set(succession_basis) != {
        "diagnostic",
        "execution_closure_authority",
        "immutable_migration_receipt",
        "original_migration_authority",
        "user_goal_scope",
    } or not isinstance(succession_basis.get("user_goal_scope"), str) or not (
        succession_basis["user_goal_scope"]
    ):
        raise TargetSealedError(
            "V8R4A migrated-state source-succession authority basis drifted"
        )
    for field, target_role, target_document, label in (
        (
            "diagnostic",
            "migration_source_succession_failure_diagnostic",
            migration_source_succession_diagnostic,
            "V8R4A migrated-state source-succession diagnostic",
        ),
        (
            "execution_closure_authority",
            "execution_closure_correction_authorization",
            execution_closure_authority,
            "V8R4A migrated-state source-succession execution authority",
        ),
        (
            "immutable_migration_receipt",
            "gpu_state_migration_receipt",
            migration_receipt,
            "V8R4A migrated-state source-succession migration receipt",
        ),
        (
            "original_migration_authority",
            "infrastructure_correction_authorization",
            infrastructure_authority,
            "V8R4A migrated-state source-succession original authority",
        ),
    ):
        _require_authority_legacy_binding(
            succession_basis.get(field),
            binding=bindings[target_role],
            document=target_document,
            project_root=project_root,
            label=label,
        )
    open_lifecycle_basis = open_lifecycle_authority.get("authority_basis")
    if not isinstance(open_lifecycle_basis, Mapping):
        raise TargetSealedError(
            "V8R4A open-lifecycle recovery authority basis drifted"
        )
    for field, target_role, target_document, label in (
        (
            "diagnostic",
            "open_lifecycle_recovery_failure_diagnostic",
            open_lifecycle_diagnostic,
            "V8R4A open-lifecycle recovery diagnostic",
        ),
        (
            "parent_kill_safe_addendum",
            "kill_safe_correction_authorization",
            kill_safe_authority,
            "V8R4A open-lifecycle parent kill-safe authority",
        ),
        (
            "parent_source_closure_addendum",
            "source_closure_correction_authorization",
            source_closure_authority,
            "V8R4A open-lifecycle parent source-closure authority",
        ),
    ):
        _require_authority_legacy_binding(
            open_lifecycle_basis.get(field),
            binding=bindings[target_role],
            document=target_document,
            project_root=project_root,
            label=label,
        )
    execution_closure_basis = execution_closure_authority.get("authority_basis")
    assert isinstance(execution_closure_basis, Mapping)
    for field, target_role, target_document, label in (
        (
            "diagnostic",
            "execution_closure_failure_diagnostic",
            execution_closure_diagnostic,
            "V8R4A execution-closure diagnostic",
        ),
        (
            "parent_kill_safe_addendum",
            "kill_safe_correction_authorization",
            kill_safe_authority,
            "V8R4A execution-closure parent kill-safe authority",
        ),
        (
            "parent_open_lifecycle_recovery_addendum",
            "open_lifecycle_recovery_correction_authorization",
            open_lifecycle_authority,
            "V8R4A execution-closure parent lifecycle authority",
        ),
        (
            "parent_source_closure_addendum",
            "source_closure_correction_authorization",
            source_closure_authority,
            "V8R4A execution-closure parent source authority",
        ),
    ):
        _require_authority_legacy_binding(
            execution_closure_basis.get(field),
            binding=bindings[target_role],
            document=target_document,
            project_root=project_root,
            label=label,
        )
    if not (
        test_receipt.get("all_tests_passed") is True
        and test_receipt.get("gpu_accessed") is False
        and test_receipt.get("target_or_outer_reference_accessed") is False
    ):
        raise TargetSealedError("V8R4A implementation test boundary drifted")
    if not (
        pretrain.get("status") == "authorized"
        and pretrain.get("adaptive_retrospective_only") is True
        and pretrain.get("training_authorized") is True
        and pretrain.get("production_target_sealed_runtime_authorized") is True
        and pretrain.get("promotion_authorized") is False
        and pretrain.get("commercial_claim_authorized") is False
        and pretrain.get("efficiency_benchmark_scope")
        == BENCHMARK_ADMITTED_CONTEXT_EFFICIENCY_SCOPE
        and pretrain.get("runtime_ledger_prefixes")
        == BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES
        and pretrain.get("canonical_gpu_state_paths")
        == _canonical_gpu_state_path_document()
    ):
        raise TargetSealedError("V8R4A pretrain production boundary drifted")

    cross_bindings: tuple[
        tuple[str, Mapping[str, Any], str, str], ...
    ] = (
        ("test parent V8R4 authority", test_receipt, "correction_authorization", "correction_authorization"),
        (
            "test V8R4A authority",
            test_receipt,
            "infrastructure_correction_authorization",
            "infrastructure_correction_authorization",
        ),
        (
            "test source-closure authority",
            test_receipt,
            "source_closure_correction_authorization",
            "source_closure_correction_authorization",
        ),
        (
            "test source dependency authority",
            test_receipt,
            "source_closure_dependency_authorization",
            "source_closure_dependency_authorization",
        ),
        (
            "test kill-safe authority",
            test_receipt,
            "kill_safe_correction_authorization",
            "kill_safe_correction_authorization",
        ),
        (
            "test open-lifecycle recovery authority",
            test_receipt,
            "open_lifecycle_recovery_correction_authorization",
            "open_lifecycle_recovery_correction_authorization",
        ),
        (
            "test execution-closure authority",
            test_receipt,
            "execution_closure_correction_authorization",
            "execution_closure_correction_authorization",
        ),
        (
            "test source-closure diagnostic",
            test_receipt,
            "source_closure_failure_diagnostic",
            "source_closure_failure_diagnostic",
        ),
        (
            "test kill-safe diagnostic",
            test_receipt,
            "kill_safe_failure_diagnostic",
            "kill_safe_failure_diagnostic",
        ),
        (
            "test open-lifecycle recovery diagnostic",
            test_receipt,
            "open_lifecycle_recovery_failure_diagnostic",
            "open_lifecycle_recovery_failure_diagnostic",
        ),
        (
            "test execution-closure diagnostic",
            test_receipt,
            "execution_closure_failure_diagnostic",
            "execution_closure_failure_diagnostic",
        ),
        (
            "test migration receipt",
            test_receipt,
            "gpu_state_migration_receipt",
            "gpu_state_migration_receipt",
        ),
        ("snapshot parent V8R4 authority", snapshot, "correction_authorization", "correction_authorization"),
        (
            "snapshot V8R4A authority",
            snapshot,
            "infrastructure_correction_authorization",
            "infrastructure_correction_authorization",
        ),
        (
            "snapshot source-closure authority",
            snapshot,
            "source_closure_correction_authorization",
            "source_closure_correction_authorization",
        ),
        (
            "snapshot source dependency authority",
            snapshot,
            "source_closure_dependency_authorization",
            "source_closure_dependency_authorization",
        ),
        (
            "snapshot kill-safe authority",
            snapshot,
            "kill_safe_correction_authorization",
            "kill_safe_correction_authorization",
        ),
        (
            "snapshot open-lifecycle recovery authority",
            snapshot,
            "open_lifecycle_recovery_correction_authorization",
            "open_lifecycle_recovery_correction_authorization",
        ),
        (
            "snapshot execution-closure authority",
            snapshot,
            "execution_closure_correction_authorization",
            "execution_closure_correction_authorization",
        ),
        (
            "snapshot source-closure diagnostic",
            snapshot,
            "source_closure_failure_diagnostic",
            "source_closure_failure_diagnostic",
        ),
        (
            "snapshot kill-safe diagnostic",
            snapshot,
            "kill_safe_failure_diagnostic",
            "kill_safe_failure_diagnostic",
        ),
        (
            "snapshot open-lifecycle recovery diagnostic",
            snapshot,
            "open_lifecycle_recovery_failure_diagnostic",
            "open_lifecycle_recovery_failure_diagnostic",
        ),
        (
            "snapshot execution-closure diagnostic",
            snapshot,
            "execution_closure_failure_diagnostic",
            "execution_closure_failure_diagnostic",
        ),
        (
            "snapshot migration receipt",
            snapshot,
            "gpu_state_migration_receipt",
            "gpu_state_migration_receipt",
        ),
        ("pretrain source snapshot", pretrain, "source_snapshot", "source_snapshot"),
        (
            "pretrain test receipt",
            pretrain,
            "implementation_test_receipt",
            "implementation_test_receipt",
        ),
        ("pretrain parent V8R4 authority", pretrain, "correction_authorization", "correction_authorization"),
        (
            "pretrain V8R4A authority",
            pretrain,
            "infrastructure_correction_authorization",
            "infrastructure_correction_authorization",
        ),
        (
            "pretrain source-closure authority",
            pretrain,
            "source_closure_correction_authorization",
            "source_closure_correction_authorization",
        ),
        (
            "pretrain source dependency authority",
            pretrain,
            "source_closure_dependency_authorization",
            "source_closure_dependency_authorization",
        ),
        (
            "pretrain kill-safe authority",
            pretrain,
            "kill_safe_correction_authorization",
            "kill_safe_correction_authorization",
        ),
        (
            "pretrain open-lifecycle recovery authority",
            pretrain,
            "open_lifecycle_recovery_correction_authorization",
            "open_lifecycle_recovery_correction_authorization",
        ),
        (
            "pretrain execution-closure authority",
            pretrain,
            "execution_closure_correction_authorization",
            "execution_closure_correction_authorization",
        ),
        (
            "pretrain source-closure diagnostic",
            pretrain,
            "source_closure_failure_diagnostic",
            "source_closure_failure_diagnostic",
        ),
        (
            "pretrain kill-safe diagnostic",
            pretrain,
            "kill_safe_failure_diagnostic",
            "kill_safe_failure_diagnostic",
        ),
        (
            "pretrain open-lifecycle recovery diagnostic",
            pretrain,
            "open_lifecycle_recovery_failure_diagnostic",
            "open_lifecycle_recovery_failure_diagnostic",
        ),
        (
            "pretrain execution-closure diagnostic",
            pretrain,
            "execution_closure_failure_diagnostic",
            "execution_closure_failure_diagnostic",
        ),
        (
            "pretrain migration receipt",
            pretrain,
            "gpu_state_migration_receipt",
            "gpu_state_migration_receipt",
        ),
    )
    _require_snapshot_file_binding(
        snapshot.get("implementation_test_receipt"),
        binding=bindings["implementation_test_receipt"],
        project_root=project_root,
        label="snapshot test receipt",
    )
    contract_binding = bindings["campaign_contract"]
    contract_relative = CAMPAIGN_CONTRACT_RELATIVE_PATH.as_posix()
    implementation_rows = snapshot.get("implementation_files")
    if not isinstance(implementation_rows, list):
        raise TargetSealedError("snapshot implementation file cover drifted")
    contract_rows = [
        row
        for row in implementation_rows
        if isinstance(row, Mapping) and row.get("path") == contract_relative
    ]
    if len(contract_rows) != 1 or contract_binding.path != (
        project_root / CAMPAIGN_CONTRACT_RELATIVE_PATH
    ):
        raise TargetSealedError("snapshot canonical campaign contract cover drifted")
    _require_snapshot_file_binding(
        contract_rows[0],
        binding=contract_binding,
        project_root=project_root,
        label="snapshot campaign contract",
    )
    for label, owner, field, target_role in cross_bindings:
        target = bindings.get(target_role)
        if target is None:
            raise TargetSealedError(f"{label} target is unavailable")
        _require_exact_governance_binding(
            owner.get(field),
            binding=target,
            project_root=project_root,
            label=label,
        )
    for owner_label, owner in (
        ("test", test_receipt),
        ("snapshot", snapshot),
        ("pretrain", pretrain),
    ):
        for role, role_label in (
            (
                "migration_source_succession_correction_authorization",
                "source-succession authority",
            ),
            (
                "migration_source_succession_failure_diagnostic",
                "source-succession diagnostic",
            ),
            (
                "fd_closure_correction_authorization",
                "FD-closure authority",
            ),
            (
                "fd_closure_failure_diagnostic",
                "FD-closure diagnostic",
            ),
            (
                "canary_boundary_correction_authorization",
                "canary-boundary authority",
            ),
            (
                "canary_boundary_failure_diagnostic",
                "canary-boundary diagnostic",
            ),
            (
                "frozen_contract_encoding_correction_authorization",
                "frozen-contract authority",
            ),
            (
                "frozen_contract_encoding_failure_diagnostic",
                "frozen-contract diagnostic",
            ),
            (
                "gpu_state_parent_bind_correction_authorization",
                "GPU-state parent-bind authority",
            ),
            (
                "gpu_state_parent_bind_failure_diagnostic",
                "GPU-state parent-bind diagnostic",
            ),
            (
                "admitted_context_correction_authorization",
                "benchmark admitted-context authority",
            ),
            (
                "admitted_context_failure_diagnostic",
                "benchmark admitted-context diagnostic",
            ),
        ):
            _require_exact_governance_binding(
                owner.get(role),
                binding=bindings[role],
                project_root=project_root,
                label=f"{owner_label} migrated-state {role_label}",
            )


def _snapshot_file_rows(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Extract the exact frozen file open-set attested by a source snapshot."""

    rows: list[Mapping[str, Any]] = []
    for field in ("entry_evidence", "implementation_files", "read_only_ancestry"):
        value = document.get(field)
        if not isinstance(value, list):
            raise TargetSealedError(f"source snapshot {field} is not a list")
        for row in value:
            if not isinstance(row, dict):
                raise TargetSealedError(f"source snapshot {field} row is invalid")
            rows.append(row)
    receipt = document.get("implementation_test_receipt")
    if not isinstance(receipt, dict):
        raise TargetSealedError("source snapshot implementation receipt is invalid")
    rows.append(receipt)
    environment = document.get("environment")
    if not isinstance(environment, dict) or not isinstance(
        environment.get("pyproject"), dict
    ):
        raise TargetSealedError("source snapshot environment binding is invalid")
    rows.append(environment["pyproject"])
    return rows


def _validate_snapshot_row(
    row: Mapping[str, Any], *, project_root: Path
) -> tuple[Path, str, int, int]:
    allowed_keys = {"path", "file_sha256", "size_bytes", "mode"}
    if set(row) != allowed_keys:
        raise TargetSealedError("source snapshot file-binding schema drifted")
    relative = row.get("path")
    digest = row.get("file_sha256")
    size = row.get("size_bytes")
    mode = row.get("mode")
    if (
        not isinstance(relative, str)
        or not relative
        or PurePosixPath(relative).is_absolute()
        or ".." in PurePosixPath(relative).parts
        or not _is_sha256(digest)
        or type(size) is not int
        or size < 0
        or type(mode) is not int
        or mode != 0o444
    ):
        raise TargetSealedError("source snapshot file binding is not frozen V8R4")
    path = _canonical_existing(
        project_root / Path(relative), label="source snapshot frozen material"
    )
    if not _path_within(path, project_root):
        raise TargetSealedError("source snapshot material escapes project root")
    return path, str(digest), int(size), int(mode)


def _validate_quarantined_material_cover(
    *,
    project_root: Path,
    governance_documents: Mapping[str, Mapping[str, Any]],
    governance_bindings: Mapping[str, FileBinding],
) -> None:
    seal = governance_documents.get("quarantined_output_seal")
    if not isinstance(seal, Mapping):
        raise TargetSealedError("quarantined output seal is absent")
    rows = seal.get("files")
    if not isinstance(rows, list) or len(rows) != 11:
        raise TargetSealedError("quarantined output seal does not bind 11 materials")
    for number, row in enumerate(rows):
        role = f"quarantined_material_{number:02d}"
        binding = governance_bindings.get(role)
        if not isinstance(row, dict) or binding is None:
            raise TargetSealedError("quarantined material cover is incomplete")
        if set(row) != {"path", "sha256", "bytes", "mode"}:
            raise TargetSealedError("quarantined material binding schema drifted")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or binding.path != project_root / Path(relative)
            or row.get("sha256") != binding.sha256
            or row.get("bytes") != binding.bytes
            or row.get("mode") != 0o444
            or binding.mode != 0o444
        ):
            raise TargetSealedError("quarantined material binding drifted")
    owner = seal.get("quarantine_owner_receipt")
    diagnostic = seal.get("diagnostic")
    for role, expected in (
        ("quarantine_owner_receipt", owner),
        ("failure_diagnostic", diagnostic),
    ):
        binding = governance_bindings.get(role)
        if (
            not isinstance(expected, dict)
            or binding is None
            or expected.get("path")
            != str(binding.path.relative_to(project_root))
            or expected.get("sha256") != binding.sha256
            or expected.get("bytes") != binding.bytes
        ):
            raise TargetSealedError(f"quarantined seal {role} binding drifted")


DISCOVERY_SHARD_SEAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version", "classification", "campaign_id", "campaign_revision",
        "infrastructure_revision", "outer_fold_shard", "contract",
        "pretrain_authorization", "training_index", "completed_units",
        "peer_outer_shard_pack_mounted_or_opened",
        "combined_target_bearing_cache_opened", "outer_prediction_pack_absent",
        "physical_boundary", "gpu_usage_ledger_prefix",
        "pre_discovery_efficiency_benchmark", "v8r3_quarantine_owner", "units",
        "cross_outer_validation_reuse_present", "fully_nested_confirmatory_oof",
        "prospective_confirmation_required",
        "ready_for_pack_free_shard_aggregation", "commercial_claim_authorized",
        "content_sha256",
    }
)
DISCOVERY_PHYSICAL_BOUNDARY: Final[Mapping[str, Any]] = {
    "campaign_revision": "V8R4",
    "physical_input_partition": "outer_excluded_training_validation_pack",
    "combined_target_bearing_cache_opened": False,
    "outer_test_rows_physically_present_in_training_pack": False,
    "outer_test_identity_or_classical_context_materialized": False,
    "outer_test_feature_values_materialized_or_forwarded": False,
    "outer_test_reference_fields_opened": False,
    "outer_test_model_or_evaluation_iterator_constructed": False,
    "numpy_row_access_audit_enforced": True,
    "outer_prediction_pack_absent_during_discovery": True,
    "commercial_claim_allowed": False,
}
PORTABLE_ARTIFACT_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {"path", "sha256", "bytes"}
)


def _is_portable_artifact_binding(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != PORTABLE_ARTIFACT_BINDING_KEYS:
        return False
    path = value.get("path")
    return (
        isinstance(path, str)
        and bool(path)
        and "\x00" not in path
        and _is_sha256(value.get("sha256"))
        and type(value.get("bytes")) is int
        and value["bytes"] >= 0
    )


def _portable_binding_matches(
    value: object,
    binding: FileBinding | None,
    *,
    project_root: Path,
) -> bool:
    """Match either producer-supported path spelling to one pinned inode."""

    if not _is_portable_artifact_binding(value) or binding is None:
        return False
    assert isinstance(value, Mapping)
    raw_path = Path(str(value["path"]))
    projected_path = (
        raw_path if raw_path.is_absolute() else project_root / raw_path
    )
    return (
        Path(os.path.abspath(projected_path)) == binding.path
        and value.get("sha256") == binding.sha256
        and value.get("bytes") == binding.bytes
    )


def _validate_discovery_aggregation_shard_seals(
    governance_documents: Mapping[str, Mapping[str, Any]],
    governance_bindings: Mapping[str, FileBinding],
    *,
    project_root: Path,
) -> None:
    contract_binding = governance_bindings.get("campaign_contract")
    authorization_binding = governance_bindings.get("active_authorization")
    for outer_fold in (3, 4):
        role = f"discovery_shard_seal_outer{outer_fold}"
        document = governance_documents.get(role)
        units = document.get("units") if isinstance(document, Mapping) else None
        if not (
            isinstance(document, Mapping)
            and set(document) == DISCOVERY_SHARD_SEAL_KEYS
            and document.get("schema_version") == 1
            and document.get("classification")
            == "adaptive_v3r1_v8r4_discovery_capability_shard_seal"
            and document.get("campaign_id") == CAMPAIGN_ID
            and document.get("campaign_revision") == "V8R4"
            and document.get("infrastructure_revision") == "V8R4A"
            and document.get("outer_fold_shard") == outer_fold
            and document.get("completed_units") == 9
            and document.get("peer_outer_shard_pack_mounted_or_opened") is False
            and document.get("combined_target_bearing_cache_opened") is False
            and document.get("outer_prediction_pack_absent") is True
            and document.get("physical_boundary") == DISCOVERY_PHYSICAL_BOUNDARY
            and document.get("cross_outer_validation_reuse_present") is True
            and document.get("fully_nested_confirmatory_oof") is False
            and document.get("prospective_confirmation_required") is True
            and document.get("ready_for_pack_free_shard_aggregation") is True
            and document.get("commercial_claim_authorized") is False
            and _portable_binding_matches(
                document.get("contract"),
                contract_binding,
                project_root=project_root,
            )
            and _portable_binding_matches(
                document.get("pretrain_authorization"),
                authorization_binding,
                project_root=project_root,
            )
            and all(
                _is_portable_artifact_binding(document.get(field))
                for field in (
                    "training_index", "pre_discovery_efficiency_benchmark",
                    "v8r3_quarantine_owner",
                )
            )
            and isinstance(units, list)
            and len(units) == 9
        ):
            raise TargetSealedError(f"aggregation {role} identity drifted")
        observed: set[tuple[int, int, str]] = set()
        for unit in units:
            if not isinstance(unit, Mapping) or set(unit) != {
                "outer_fold", "seed", "variant", "receipt"
            }:
                raise TargetSealedError(f"aggregation {role} unit schema drifted")
            unit_outer = unit.get("outer_fold")
            unit_seed = unit.get("seed")
            unit_variant = unit.get("variant")
            if not (
                type(unit_outer) is int
                and unit_outer == outer_fold
                and type(unit_seed) is int
                and unit_seed in SEEDS
                and isinstance(unit_variant, str)
                and unit_variant in {"H0_no_factor", "H1_factor", "H2_full"}
                and _is_portable_artifact_binding(unit.get("receipt"))
            ):
                raise TargetSealedError(f"aggregation {role} unit cover drifted")
            identity = (unit_outer, unit_seed, unit_variant)
            if identity in observed:
                raise TargetSealedError(f"aggregation {role} unit cover drifted")
            observed.add(identity)
        if observed != {
            (outer_fold, seed, variant)
            for seed in SEEDS
            for variant in ("H0_no_factor", "H1_factor", "H2_full")
        }:
            raise TargetSealedError(f"aggregation {role} exact cover drifted")
        usage = document.get("gpu_usage_ledger_prefix")
        if not isinstance(usage, Mapping) or set(usage) != {
            "path", "sha256", "bytes", "records", "terminal_record_sha256",
            "settled_usage_ns", "elapsed_seconds", "open_reservations",
        } or not (
            _is_sha256(usage.get("sha256"))
            and _is_sha256(usage.get("terminal_record_sha256"))
            and usage.get("open_reservations") == 0
            and type(usage.get("records")) is int
            and usage["records"] >= 0
            and type(usage.get("settled_usage_ns")) is int
            and usage["settled_usage_ns"] >= 0
        ):
            raise TargetSealedError(f"aggregation {role} ledger projection drifted")


MODEL_SOURCE_SHARD_SEAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version", "classification", "campaign_id", "campaign_revision",
        "infrastructure_revision", "outer_fold", "seeds", "selected_variant",
        "unit_count", "exact_three_seed_cover", "selection_lock",
        "promotion_authorization", "units", "target_or_prediction_values_present",
        "source_paths_or_peer_outputs_authorized_in_child",
        "commercial_or_confirmatory_claim_allowed", "content_sha256",
    }
)
PREDICTION_SHARD_SEAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version", "classification", "campaign_id", "campaign_revision",
        "infrastructure_revision", "outer_fold", "seeds", "selected_variant",
        "selected_release_mode", "unit_count", "exact_three_seed_cover",
        "row_count_per_seed", "cache_index_sha256", "prediction_pack_index",
        "model_source_shard_seal", "selection_lock", "promotion_authorization",
        "gpu_usage_ledger_prefix", "units", "target_fields_accessed_or_emitted",
        "ready_for_pack_free_promotion_aggregation",
        "commercial_claim_authorized", "content_sha256",
    }
)


def _validate_fixed_aggregation_shard_seals(
    governance_documents: Mapping[str, Mapping[str, Any]],
    governance_bindings: Mapping[str, FileBinding],
    *,
    project_root: Path,
) -> None:
    selection_binding = governance_bindings.get("selection_lock")
    authorization_binding = governance_bindings.get("promotion_authorization")
    selected_variant: str | None = None
    selected_release_mode: str | None = None
    for outer_fold in range(6):
        model_role = f"model_source_seal_outer{outer_fold}"
        prediction_role = f"prediction_shard_seal_outer{outer_fold}"
        model_document = governance_documents.get(model_role)
        prediction_document = governance_documents.get(prediction_role)
        model_units = (
            model_document.get("units")
            if isinstance(model_document, Mapping)
            else None
        )
        if not (
            isinstance(model_document, Mapping)
            and set(model_document) == MODEL_SOURCE_SHARD_SEAL_KEYS
            and model_document.get("schema_version") == 1
            and model_document.get("classification")
            == "adaptive_v3r1_v8r4a_model_source_shard_seal"
            and model_document.get("campaign_id") == CAMPAIGN_ID
            and model_document.get("campaign_revision") == "V8R4"
            and model_document.get("infrastructure_revision") == "V8R4A"
            and model_document.get("outer_fold") == outer_fold
            and model_document.get("seeds") == list(SEEDS)
            and model_document.get("unit_count") == 3
            and model_document.get("exact_three_seed_cover") is True
            and isinstance(model_document.get("selected_variant"), str)
            and bool(model_document["selected_variant"])
            and _portable_binding_matches(
                model_document.get("selection_lock"),
                selection_binding,
                project_root=project_root,
            )
            and _portable_binding_matches(
                model_document.get("promotion_authorization"),
                authorization_binding,
                project_root=project_root,
            )
            and model_document.get("target_or_prediction_values_present") is False
            and model_document.get(
                "source_paths_or_peer_outputs_authorized_in_child"
            ) is False
            and model_document.get(
                "commercial_or_confirmatory_claim_allowed"
            ) is False
            and isinstance(model_units, list)
            and len(model_units) == 3
        ):
            raise TargetSealedError(f"aggregation {model_role} identity drifted")
        if selected_variant is None:
            selected_variant = str(model_document["selected_variant"])
        elif model_document.get("selected_variant") != selected_variant:
            raise TargetSealedError("fixed aggregation variant binding drifted")
        observed_model_seeds: set[int] = set()
        model_cover_proofs: set[tuple[int, str]] = set()
        for unit in model_units:
            if not isinstance(unit, Mapping) or set(unit) != {
                "outer_fold", "seed", "source_kind", "scientific_signature_sha256",
                "row_count", "global_cache_index_sha256",
                "model_bound_prediction_pack_manifest", "model_checkpoint",
                "model_scaler", "model_source_capability",
            }:
                raise TargetSealedError(f"aggregation {model_role} unit schema drifted")
            seed = unit.get("seed")
            if not (
                unit.get("outer_fold") == outer_fold
                and type(seed) is int
                and seed in SEEDS
                and unit.get("source_kind")
                in {"local_training", "discovery", "discovery_pointer"}
                and _is_sha256(unit.get("scientific_signature_sha256"))
                and type(unit.get("row_count")) is int
                and unit["row_count"] > 0
                and _is_sha256(unit.get("global_cache_index_sha256"))
                and all(
                    _is_portable_artifact_binding(unit.get(field))
                    for field in (
                        "model_bound_prediction_pack_manifest", "model_checkpoint",
                        "model_scaler", "model_source_capability",
                    )
                )
                and seed not in observed_model_seeds
            ):
                raise TargetSealedError(f"aggregation {model_role} unit cover drifted")
            observed_model_seeds.add(int(seed))
            model_cover_proofs.add(
                (int(unit["row_count"]), str(unit["global_cache_index_sha256"]))
            )
        if observed_model_seeds != set(SEEDS) or len(model_cover_proofs) != 1:
            raise TargetSealedError(f"aggregation {model_role} exact cover drifted")

        prediction_units = (
            prediction_document.get("units")
            if isinstance(prediction_document, Mapping)
            else None
        )
        expected_model_binding = governance_bindings.get(model_role)
        portable_model_binding = (
            {
                "path": str(expected_model_binding.path),
                "sha256": expected_model_binding.sha256,
                "bytes": expected_model_binding.bytes,
            }
            if expected_model_binding is not None
            else None
        )
        model_rows, model_cache_sha256 = next(iter(model_cover_proofs))
        if not (
            isinstance(prediction_document, Mapping)
            and set(prediction_document) == PREDICTION_SHARD_SEAL_KEYS
            and prediction_document.get("schema_version") == 1
            and prediction_document.get("classification")
            == "adaptive_v3r1_v8r4a_prediction_shard_completion_seal"
            and prediction_document.get("campaign_id") == CAMPAIGN_ID
            and prediction_document.get("campaign_revision") == "V8R4"
            and prediction_document.get("infrastructure_revision") == "V8R4A"
            and prediction_document.get("outer_fold") == outer_fold
            and prediction_document.get("seeds") == list(SEEDS)
            and prediction_document.get("selected_variant") == selected_variant
            and isinstance(prediction_document.get("selected_release_mode"), str)
            and bool(prediction_document["selected_release_mode"])
            and prediction_document.get("unit_count") == 3
            and prediction_document.get("exact_three_seed_cover") is True
            and prediction_document.get("row_count_per_seed") == model_rows
            and prediction_document.get("cache_index_sha256")
            == model_cache_sha256
            and _is_portable_artifact_binding(
                prediction_document.get("prediction_pack_index")
            )
            and prediction_document.get("model_source_shard_seal")
            == portable_model_binding
            and _portable_binding_matches(
                prediction_document.get("selection_lock"),
                selection_binding,
                project_root=project_root,
            )
            and _portable_binding_matches(
                prediction_document.get("promotion_authorization"),
                authorization_binding,
                project_root=project_root,
            )
            and prediction_document.get("target_fields_accessed_or_emitted") is False
            and prediction_document.get(
                "ready_for_pack_free_promotion_aggregation"
            ) is True
            and prediction_document.get("commercial_claim_authorized") is False
            and isinstance(prediction_units, list)
            and len(prediction_units) == 3
        ):
            raise TargetSealedError(
                f"aggregation {prediction_role} identity drifted"
            )
        if selected_release_mode is None:
            selected_release_mode = str(
                prediction_document["selected_release_mode"]
            )
        elif prediction_document.get("selected_release_mode") != selected_release_mode:
            raise TargetSealedError("fixed aggregation release-mode binding drifted")
        observed_prediction_seeds: set[int] = set()
        for unit in prediction_units:
            if not isinstance(unit, Mapping) or set(unit) != {
                "outer_fold", "seed", "completion_receipt",
                "promotion_model_source", "rows", "cache_index_sha256",
                "prediction", "prediction_manifest",
            }:
                raise TargetSealedError(
                    f"aggregation {prediction_role} unit schema drifted"
                )
            seed = unit.get("seed")
            if not (
                unit.get("outer_fold") == outer_fold
                and type(seed) is int
                and seed in SEEDS
                and type(unit.get("rows")) is int
                and unit["rows"] == model_rows
                and unit.get("cache_index_sha256") == model_cache_sha256
                and all(
                    _is_portable_artifact_binding(unit.get(field))
                    for field in (
                        "completion_receipt", "promotion_model_source",
                        "prediction", "prediction_manifest",
                    )
                )
                and seed not in observed_prediction_seeds
            ):
                raise TargetSealedError(
                    f"aggregation {prediction_role} unit cover drifted"
                )
            observed_prediction_seeds.add(int(seed))
        if observed_prediction_seeds != set(SEEDS):
            raise TargetSealedError(
                f"aggregation {prediction_role} exact cover drifted"
            )
        usage = prediction_document.get("gpu_usage_ledger_prefix")
        if not isinstance(usage, Mapping) or set(usage) != {
            "path", "sha256", "bytes", "records", "terminal_record_sha256",
            "settled_usage_ns", "elapsed_seconds", "open_reservations",
        } or not (
            _is_sha256(usage.get("sha256"))
            and _is_sha256(usage.get("terminal_record_sha256"))
            and usage.get("open_reservations") == 0
            and type(usage.get("records")) is int
            and usage["records"] >= 0
            and type(usage.get("settled_usage_ns")) is int
            and usage["settled_usage_ns"] >= 0
            and type(usage.get("elapsed_seconds")) in {int, float}
            and usage["elapsed_seconds"] >= 0
        ):
            raise TargetSealedError(
                f"aggregation {prediction_role} ledger projection drifted"
            )


def _validate_aggregation_shard_seals(
    governance_documents: Mapping[str, Mapping[str, Any]],
    governance_bindings: Mapping[str, FileBinding],
    *,
    entry_name: str,
    project_root: Path,
) -> None:
    if entry_name == "run_hfr_v3r1_discovery_campaign.py":
        _validate_discovery_aggregation_shard_seals(
            governance_documents,
            governance_bindings,
            project_root=project_root,
        )
        return
    if entry_name == "run_fixed_hfr_v3r1_oof_campaign.py":
        _validate_fixed_aggregation_shard_seals(
            governance_documents,
            governance_bindings,
            project_root=project_root,
        )
        return
    raise TargetSealedError("aggregation seal validator entry drifted")


def _directory_revalidate(binding: DirectoryBinding, descriptor: int, *, label: str) -> None:
    status = os.fstat(descriptor)
    named = os.stat(binding.path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(status.st_mode)
        or (status.st_dev, status.st_ino) != (binding.st_dev, binding.st_ino)
        or (named.st_dev, named.st_ino) != (binding.st_dev, binding.st_ino)
    ):
        raise TargetSealedError(f"{label} directory capability drifted")


def _state_directory_mount_source(
    binding: DirectoryBinding, state_row: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Bind a directory identity to the migration-validated direct inventory."""

    expected_entries = state_row.get("exact_entries")
    if not (
        isinstance(expected_entries, list)
        and all(isinstance(value, str) and value for value in expected_entries)
        and expected_entries == sorted(set(expected_entries))
        and state_row.get("mode") == f"{binding.mode:04o}"
        and state_row.get("st_dev") == binding.st_dev
        and state_row.get("st_ino") == binding.st_ino
    ):
        raise TargetSealedError(f"{label} migration identity drifted before mount")
    return {**binding.document(), "exact_entries": list(expected_entries)}


def _revalidate_state_directory_descriptor(
    binding: DirectoryBinding,
    descriptor: int,
    state_row: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Revalidate pathname, fd identity, exact mode, and direct inventory."""

    source = _state_directory_mount_source(binding, state_row, label=label)
    before = os.fstat(descriptor)
    try:
        entries = sorted(os.listdir(descriptor))
    except OSError as error:
        raise TargetSealedError(f"{label} inventory cannot be read") from error
    after = os.fstat(descriptor)
    named = os.stat(binding.path, follow_symlinks=False)
    if not (
        stat.S_ISDIR(before.st_mode)
        and stat.S_ISDIR(after.st_mode)
        and stat.S_ISDIR(named.st_mode)
        and (before.st_dev, before.st_ino)
        == (after.st_dev, after.st_ino)
        == (named.st_dev, named.st_ino)
        == (binding.st_dev, binding.st_ino)
        and _mode(before) == _mode(after) == _mode(named) == binding.mode == 0o700
        and entries == source["exact_entries"]
    ):
        raise TargetSealedError(f"{label} descriptor identity/inventory drifted")


def _file_revalidate(binding: FileBinding, descriptor: int, *, label: str) -> None:
    status = os.fstat(descriptor)
    named = os.stat(binding.path, follow_symlinks=False)
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or (status.st_dev, status.st_ino, status.st_size)
        != (binding.st_dev, binding.st_ino, binding.bytes)
        or (named.st_dev, named.st_ino) != (binding.st_dev, binding.st_ino)
        or _mode(status) != binding.mode
    ):
        raise TargetSealedError(f"{label} file capability drifted")


def _create_memfd(payload: bytes) -> int:
    if hasattr(os, "memfd_create"):
        descriptor = os.memfd_create(
            "snn-v8r4a-runtime-spec", flags=getattr(os, "MFD_CLOEXEC", 0)
        )
    else:
        descriptor = -1
        if hasattr(os, "O_TMPFILE"):
            try:
                descriptor = os.open(
                    "/tmp",
                    os.O_RDWR | os.O_TMPFILE | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                )
            except OSError:
                descriptor = -1
        if descriptor < 0:
            descriptor, raw_path = tempfile.mkstemp(prefix=".snn-v8r4a-spec-", dir="/tmp")
            os.unlink(raw_path)
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count <= 0:
            os.close(descriptor)
            raise TargetSealedError("short write to sealed runtime spec")
        written += count
    os.lseek(descriptor, 0, os.SEEK_SET)
    # Prevent later modification through this descriptor where Linux supports it.
    try:
        seals = (
            getattr(fcntl, "F_SEAL_SEAL", 0)
            | getattr(fcntl, "F_SEAL_SHRINK", 0)
            | getattr(fcntl, "F_SEAL_GROW", 0)
            | getattr(fcntl, "F_SEAL_WRITE", 0)
        )
        if seals:
            fcntl.fcntl(descriptor, getattr(fcntl, "F_ADD_SEALS"), seals)
    except (OSError, AttributeError):
        # MFD_ALLOW_SEALING is not universally exposed/enabled.  The fd is never
        # shared with caller code and its exact bytes are hashed by the guard.
        pass
    return descriptor


def _linkat_empty_path(
    source_fd: int, destination_directory_fd: int, destination_name: str
) -> None:
    """Link one complete O_TMPFILE inode by its descriptor.

    Python does not expose Linux ``linkat(AT_EMPTY_PATH)`` directly.  Keeping
    this one syscall behind a narrow function also gives the CPU fault tests a
    deterministic pre/post-link kill boundary without ever creating a named
    temporary.
    """

    if not destination_name or "/" in destination_name or "\x00" in destination_name:
        raise TargetSealedError("anonymous publication destination is invalid")
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = libc.linkat
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    if linkat(
        source_fd,
        ctypes.c_char_p(b""),
        destination_directory_fd,
        ctypes.c_char_p(os.fsencode(destination_name)),
        0x1000,  # AT_EMPTY_PATH
    ) != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination_name)
        raise OSError(error_number, os.strerror(error_number), destination_name)


def _create_once_immutable_json(
    path: Path, document: Mapping[str, Any], *, label: str = "capability receipt"
) -> FileBinding:
    requested = _absolute_lexical(path, label=label)
    parent = _canonical_existing(requested.parent, label=f"{label} parent")
    if requested.parent != parent:
        raise TargetSealedError(f"{label} parent is not canonical")
    payload = canonical_json_bytes(dict(document)) + b"\n"
    if not hasattr(os, "O_TMPFILE"):
        raise TargetSealedError(f"{label} requires Linux O_TMPFILE")
    directory_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                ".",
                os.O_RDWR
                | os.O_TMPFILE
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
        except OSError as error:
            raise TargetSealedError(
                f"{label} anonymous inode creation failed: {error}"
            ) from error
        try:
            offset = 0
            while offset < len(payload):
                count = os.write(descriptor, payload[offset:])
                if count <= 0:
                    raise TargetSealedError(f"short {label} write")
                offset += count
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_nlink != 0
                or _mode(status) != 0o444
                or status.st_size != len(payload)
            ):
                raise TargetSealedError(f"{label} anonymous inode drifted")
            try:
                _linkat_empty_path(descriptor, directory_fd, requested.name)
            except FileExistsError:
                binding, existing = _read_file_binding(
                    requested, label=label, require_immutable=True
                )
                if existing != payload:
                    raise TargetSealedError(f"{label} resume bytes differ")
                return binding
            os.fsync(directory_fd)
            status = os.fstat(descriptor)
            named = os.stat(
                requested.name, dir_fd=directory_fd, follow_symlinks=False
            )
            if not (
                status.st_nlink == 1
                and named.st_nlink == 1
                and _mode(status) == _mode(named) == 0o444
                and (status.st_dev, status.st_ino, status.st_size)
                == (named.st_dev, named.st_ino, named.st_size)
            ):
                raise TargetSealedError(f"{label} publication is aliased")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    finally:
        os.close(directory_fd)
    binding, published = _read_file_binding(
        requested, label=label, require_immutable=True
    )
    if published != payload:
        raise TargetSealedError(f"{label} publication drifted")
    return binding


def _canonical_child_environment(
    *, project_root: Path, propagated: Mapping[str, str]
) -> dict[str, str]:
    environment = dict(FIXED_CHILD_ENV)
    environment.update(propagated)
    environment["PATH"] = "/usr/bin:/usr/sbin"
    environment["PWD"] = str(project_root)
    environment["PYTHONPATH"] = str(project_root / "src")
    environment["SNN_RR_TARGET_SEALED_RUNTIME"] = "V8R4A"
    # PWD is retained/set by bubblewrap itself and is checked separately.
    if set(environment) & FORBIDDEN_ENV_NAMES or ADMITTED_CHILD_FD_ENV in environment:
        raise TargetSealedError("outer environment contains a forbidden capability")
    return dict(sorted(environment.items()))


def _validate_mount_boundaries(
    *,
    ro_roots: Sequence[Path],
    rw_roots: Sequence[Path],
    governance_files: Sequence[Path],
    denied_canaries: Mapping[str, Path],
    gpu_state_readonly_parent: Path | None = None,
    gpu_state_mutable_children: Sequence[Path] = (),
) -> dict[str, str]:
    unique_ro = tuple(dict.fromkeys(ro_roots))
    unique_rw = tuple(dict.fromkeys(rw_roots))
    state_parent = gpu_state_readonly_parent
    state_children = tuple(dict.fromkeys(gpu_state_mutable_children))
    if state_parent is None:
        if state_children:
            raise TargetSealedError("GPU-state children require one read-only parent")
    elif not (
        state_parent in unique_ro
        and len(state_children) == 3
        and set(state_children) <= set(unique_rw)
        and all(child.parent == state_parent for child in state_children)
        and {child.name for child in state_children} == GPU_STATE_DIRECTORY_ROLES
    ):
        raise TargetSealedError("GPU-state parent/child overlap exception is not exact")

    def allowed_state_overlap(writable: Path, readonly: Path) -> bool:
        return (
            state_parent is not None
            and readonly == state_parent
            and writable in state_children
            and writable.parent == state_parent
        )

    for index, left in enumerate(unique_rw):
        for right in unique_rw[index + 1 :]:
            if left != right and (_path_within(left, right) or _path_within(right, left)):
                raise TargetSealedError("nested writable roots broaden capability")
        for readonly in unique_ro:
            if (
                _path_within(left, readonly) or _path_within(readonly, left)
            ) and not allowed_state_overlap(left, readonly):
                raise TargetSealedError("read-only and writable capabilities overlap")
        for governance in governance_files:
            if _path_within(governance, left):
                raise TargetSealedError("governance file is writable inside sandbox")
    result: dict[str, str] = {}
    mounted = (*unique_ro, *unique_rw)
    for role, raw in sorted(denied_canaries.items()):
        canary = _absolute_lexical(raw, label=f"{role} denied canary")
        if any(_path_within(canary, root) or canary == root for root in mounted):
            raise TargetSealedError(f"denied canary would be reachable: {role}")
        result[role] = str(canary)
    if len(set(result.values())) != len(result):
        raise TargetSealedError("denied canaries must be path-distinct")
    return result


def _validate_command_paths(
    command: Sequence[str], *, project_root: Path, mounted_roots: Sequence[Path]
) -> None:
    allowed = tuple(dict.fromkeys(mounted_roots))
    for argument in command:
        if ".." in PurePosixPath(argument).parts:
            raise TargetSealedError("child command contains path traversal")
        if argument.startswith("-") and "=/" in argument:
            raise TargetSealedError(
                "child command embeds an absolute option path; use a separate argument"
            )
        if not argument.startswith("/"):
            continue
        path = _absolute_lexical(Path(argument), label="child command path")
        # The project root itself is an intentionally empty directory
        # skeleton in the sandbox and is passed as the canonical
        # ``--project-root`` value.  Its arbitrary descendants are not
        # capabilities and remain forbidden unless individually mounted.
        if path == project_root:
            continue
        if not any(path == root or _path_within(path, root) for root in allowed):
            raise TargetSealedError(f"child command path is not mounted: {path}")


def _validate_command_denied_canaries(
    command: Sequence[str], *, denied_canaries: Mapping[str, str]
) -> None:
    """Reject only absolute command capabilities at/below denied path boundaries."""

    denied = {
        role: _absolute_lexical(Path(path), label=f"{role} denied command canary")
        for role, path in denied_canaries.items()
    }
    for argument in command:
        # _validate_command_paths rejects traversal and embedded option paths
        # first.  Only standalone lexical absolute tokens denote capabilities.
        if not argument.startswith("/"):
            continue
        path = _absolute_lexical(Path(argument), label="child command capability")
        for role, canary in denied.items():
            if path == canary or _path_within(path, canary):
                raise TargetSealedError(
                    f"child command names a denied capability: {role}"
                )


def _mount_entry(
    *, kind: str, destination: Path | str, source: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {"kind": kind, "destination": str(destination)}
    if source is not None:
        value["source"] = dict(source)
    return value


def audit_process_fds(*, allowed: Iterable[int]) -> tuple[int, ...]:
    """Return the exact live descriptor set and reject an inherited surprise."""

    expected = set(int(value) for value in allowed)
    if any(value < 0 for value in expected):
        raise TargetSealedError("negative fd in audit allowlist")
    try:
        names = os.listdir("/proc/self/fd")
    except OSError as error:
        raise TargetSealedError(f"cannot audit /proc/self/fd: {error}") from error
    observed: set[int] = set()
    for name in names:
        if not name.isascii() or not name.isdigit():
            raise TargetSealedError("non-numeric /proc/self/fd entry")
        descriptor = int(name)
        try:
            os.fstat(descriptor)
        except OSError:
            # The directory descriptor used by os.listdir may appear in its own
            # snapshot and is already closed by the time we inspect it.
            continue
        observed.add(descriptor)
    if observed != expected:
        targets: dict[int, str] = {}
        for descriptor in sorted(observed - expected):
            try:
                targets[descriptor] = os.readlink(f"/proc/self/fd/{descriptor}")
            except OSError:
                targets[descriptor] = "<unreadable>"
        raise TargetSealedError(
            f"unexpected inherited descriptor set: observed={sorted(observed)} "
            f"expected={sorted(expected)} unexpected_targets={targets}"
        )
    return tuple(sorted(observed))


def _close_guard_runtime_noise_fds(
    *, allowed: Iterable[int] = (0, 1, 2)
) -> None:
    """Close only Python's single, exact CLOEXEC urandom cache descriptor."""

    expected = {int(value) for value in allowed}
    if any(value < 0 for value in expected):
        raise TargetSealedError("negative fd in runtime normalization allowlist")

    def require_single_task() -> None:
        try:
            tasks = {
                int(name)
                for name in os.listdir("/proc/self/task")
                if name.isascii() and name.isdigit()
            }
        except OSError as error:
            raise TargetSealedError(
                f"cannot audit runtime threads before fd normalization: {error}"
            ) from error
        if tasks != {os.getpid()}:
            raise TargetSealedError(
                "runtime fd normalization requires the fresh single-thread process"
            )

    def identity(status: os.stat_result) -> tuple[int, int, int, int]:
        return (
            int(status.st_dev),
            int(status.st_ino),
            stat.S_IFMT(status.st_mode),
            int(status.st_rdev),
        )

    blockable = set(signal.valid_signals()) - {signal.SIGKILL, signal.SIGSTOP}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blockable)
    try:
        require_single_task()
        try:
            urandom_status = os.stat("/dev/urandom", follow_symlinks=True)
            names = os.listdir("/proc/self/fd")
        except OSError as error:
            raise TargetSealedError(
                f"cannot inspect runtime descriptors before normalization: {error}"
            ) from error
        urandom_identity = identity(urandom_status)
        if not stat.S_ISCHR(urandom_status.st_mode):
            raise TargetSealedError("/dev/urandom is not a character device")

        candidates: list[tuple[int, tuple[int, int, int, int]]] = []
        unexpected: dict[int, str] = {}
        for name in names:
            if not name.isascii() or not name.isdigit():
                raise TargetSealedError("non-numeric /proc/self/fd entry")
            descriptor = int(name)
            if descriptor in expected:
                continue
            try:
                before = os.fstat(descriptor)
                target = os.readlink(f"/proc/self/fd/{descriptor}")
                flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
                after = os.fstat(descriptor)
            except OSError as error:
                if error.errno in {errno.EBADF, errno.ENOENT}:
                    # The descriptor used to enumerate /proc/self/fd can occur
                    # in its own already-closed snapshot.
                    continue
                raise TargetSealedError(
                    f"cannot inspect runtime descriptor {descriptor}: {error}"
                ) from error
            before_identity = identity(before)
            if before_identity != identity(after):
                raise TargetSealedError(
                    f"runtime descriptor {descriptor} changed during normalization"
                )
            if (
                target == "/dev/urandom"
                and flags == fcntl.FD_CLOEXEC
                and stat.S_ISCHR(after.st_mode)
                and before_identity == urandom_identity
            ):
                candidates.append((descriptor, before_identity))
            else:
                unexpected[descriptor] = target

        # Validate the complete set before closing anything.  This prevents a
        # malformed or caller-owned descriptor from causing partial mutation.
        if unexpected:
            raise TargetSealedError(
                "unexpected live descriptor before runtime normalization: "
                f"{dict(sorted(unexpected.items()))}"
            )
        if len(candidates) > 1:
            raise TargetSealedError(
                "multiple CLOEXEC /dev/urandom descriptors are not runtime noise"
            )
        if not candidates:
            return

        descriptor, expected_identity = candidates[0]
        require_single_task()
        try:
            current_urandom = os.stat("/dev/urandom", follow_symlinks=True)
            current = os.fstat(descriptor)
            current_target = os.readlink(f"/proc/self/fd/{descriptor}")
            current_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
        except OSError as error:
            raise TargetSealedError(
                "runtime urandom descriptor changed before close"
            ) from error
        if not (
            identity(current_urandom) == urandom_identity
            and identity(current) == expected_identity
            and current_target == "/dev/urandom"
            and current_flags == fcntl.FD_CLOEXEC
        ):
            raise TargetSealedError(
                "runtime urandom descriptor identity drifted before close"
            )
        os.close(descriptor)
        try:
            os.fstat(descriptor)
        except OSError as error:
            if error.errno != errno.EBADF:
                raise TargetSealedError(
                    "runtime urandom descriptor close could not be verified"
                ) from error
        else:
            raise TargetSealedError(
                "runtime urandom descriptor remained live after close"
            )
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _runtime_receipt_document(
    *,
    request: RuntimeRequest,
    bwrap_binding: FileBinding,
    launcher_binding: FileBinding,
    interpreter_binding: FileBinding,
    pack_binding: DirectoryBinding | None,
    pack_index_binding: FileBinding | None,
    pack_index_document: Mapping[str, Any] | None,
    governance_bindings: Mapping[str, FileBinding],
    writable_bindings: Mapping[str, DirectoryBinding],
    prelaunch_state: LiveStateSnapshot,
    denied_canaries: Mapping[str, str],
    mount_entries: Sequence[Mapping[str, Any]],
    child_environment: Mapping[str, str],
) -> dict[str, Any]:
    mount_spec = [dict(entry) for entry in mount_entries]
    command = list(request.command)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": RECEIPT_CLASSIFICATION,
        "campaign_id": CAMPAIGN_ID,
        "campaign_revision": SCIENTIFIC_CAMPAIGN_REVISION,
        "infrastructure_revision": INFRASTRUCTURE_REVISION,
        "phase": request.phase,
        "outer_fold": request.outer_fold,
        "bubblewrap": {
            **bwrap_binding.document(),
            "version": BWRAP_VERSION,
        },
        "launcher": launcher_binding.document(),
        "interpreter": interpreter_binding.document(),
        "sealed_pack_root": (
            pack_binding.document() if pack_binding is not None else None
        ),
        "sealed_pack_index": (
            {
                **pack_index_binding.document(),
                "document_content_sha256": pack_index_document.get(
                    "content_sha256"
                ),
                "unit_count": pack_index_document.get("unit_count"),
            }
            if pack_index_binding is not None and pack_index_document is not None
            else None
        ),
        "governance_files": {
            role: binding.document()
            for role, binding in sorted(governance_bindings.items())
        },
        "writable_roots": {
            role: binding.document()
            for role, binding in sorted(writable_bindings.items())
        },
        "prelaunch_gpu_state": prelaunch_state.document(),
        "denied_canaries": dict(sorted(denied_canaries.items())),
        "mount_specification": mount_spec,
        "mount_specification_sha256": semantic_sha256(mount_spec),
        "environment": dict(child_environment),
        "environment_sha256": semantic_sha256(dict(child_environment)),
        "command": command,
        "command_sha256": semantic_sha256(command),
        "security_boundary": {
            "outer_campaign_runtime": True,
            "network_namespace_unshared": True,
            "ipc_namespace_unshared": True,
            "uts_namespace_unshared": True,
            "pid_namespace_unshared": False,
            "new_session_created": False,
            "tmp_is_private_tmpfs": True,
            "die_with_parent": True,
            "capabilities_dropped": True,
            "environment_cleared_before_allowlist": True,
            "hai_experiment_propagated": False,
            "legacy_combined_cache_mounted": False,
            "raw_or_target_root_mounted": False,
            "cross_outer_shard_mounted": False,
            "other_pack_or_output_mounted": False,
            "admitted_child_fd_created_or_consumed_by_outer_launcher": False,
            "admission_lock_fd_created_or_consumed_by_outer_launcher": False,
            "admitted_fd_direct_watchdog_to_trainer_contract_preserved": True,
            "child_fd_audit_required": True,
            "denied_canary_probe_required": True,
            "target_reference_or_selection_evidence_accessed": False,
            "commercial_claim_authorized": False,
            "production_execution_authorized": request.production,
            "atomic_replace_compatible": True,
            "synthetic_validation_only": not request.production,
            "v8r4a_ledger_migration_required": False,
            "v8r4a_migration_live_replay_validated": True,
            "dedicated_gpu_state_directory_capabilities": True,
            "gpu_state_parent_identity_readonly_bind": True,
            "exactly_three_mutable_state_directory_mounts": True,
            "benchmark_admitted_context_generation_isolated": True,
            "active_pretrain_postfailure_ledger_prefix_enforced": True,
            "usage_and_execution_closed_prelaunch": True,
            "lifecycle_mounted_read_only": True,
            "complete_project_source_or_config_trees_mounted": False,
            "source_snapshot_exact_file_mounts": True,
        },
    }
    document["content_sha256"] = semantic_sha256(document)
    return document


def _internal_spec_document(
    *,
    request: RuntimeRequest,
    receipt_binding: FileBinding,
    mount_spec_sha256: str,
    child_environment: Mapping[str, str],
    denied_canaries: Mapping[str, str],
    available_paths: Sequence[str],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": 1,
        "classification": "adaptive_v3r1_v8r4a_target_sealed_child_guard_spec",
        "campaign_id": CAMPAIGN_ID,
        "campaign_revision": SCIENTIFIC_CAMPAIGN_REVISION,
        "infrastructure_revision": INFRASTRUCTURE_REVISION,
        "phase": request.phase,
        "outer_fold": request.outer_fold,
        "capability_receipt": receipt_binding.document(),
        "mount_specification_sha256": mount_spec_sha256,
        "environment": dict(child_environment),
        "environment_sha256": semantic_sha256(dict(child_environment)),
        "denied_canaries": dict(sorted(denied_canaries.items())),
        "available_paths": list(available_paths),
        "command": list(request.command),
        "command_sha256": semantic_sha256(list(request.command)),
        "required_open_fds": [0, 1, 2],
        "forbidden_environment": sorted(FORBIDDEN_ENV_NAMES | {ADMITTED_CHILD_FD_ENV}),
    }
    document["content_sha256"] = semantic_sha256(document)
    return document


def prepare_runtime(
    request: RuntimeRequest,
    *,
    launcher_path: Path | None = None,
    version_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> PreparedRuntime:
    """Pin every host capability and construct the deterministic bwrap argv."""

    if request.production:
        _close_guard_runtime_noise_fds()
        audit_process_fds(allowed=(0, 1, 2))
    project_root, interpreter_lexical, interpreter_real = _validate_request_shape(
        request
    )
    bwrap_binding = _validate_bwrap_binary(
        request.bwrap_binary, version_runner=version_runner
    )
    launcher = _canonical_existing(
        launcher_path or Path(__file__), label="target-sealed launcher"
    )
    launcher_binding, _ = _read_file_binding(
        launcher, label="target-sealed launcher", require_immutable=request.production
    )
    expected_launcher = project_root / "scripts" / Path(__file__).name
    if request.production and launcher != expected_launcher:
        raise TargetSealedError("production launcher path is not canonical")
    interpreter_binding, _ = _read_file_binding(
        interpreter_real, label="Python interpreter", require_immutable=False
    )

    descriptors: list[int] = []
    operations: list[tuple[str, int | None, str, str | None]] = []
    mount_entries: list[dict[str, Any]] = []
    directory_bindings: dict[str, DirectoryBinding] = {}
    governance_bindings: dict[str, FileBinding] = {}
    descriptor_bindings: list[tuple[str, int, DirectoryBinding | FileBinding]] = []
    mounted_destinations: set[str] = set()
    pinned_file_destinations: dict[str, tuple[FileBinding, bytes, int]] = {}

    def add_directory(
        role: str,
        path: Path,
        *,
        readonly: bool,
        destination: Path | None = None,
        tree_binding: DirectoryBinding | None = None,
        state_row: Mapping[str, Any] | None = None,
    ) -> DirectoryBinding:
        descriptor, binding = _open_pinned_directory(path, label=role)
        destination_path = destination or binding.path
        destination_string = str(destination_path)
        if destination_string in mounted_destinations:
            os.close(descriptor)
            existing = next(
                value
                for name, value in directory_bindings.items()
                if str(value.path) == destination_string
            )
            directory_bindings[role] = existing
            return existing
        descriptors.append(descriptor)
        mounted_destinations.add(destination_string)
        if tree_binding is not None:
            if (
                binding.path != tree_binding.path
                or binding.st_dev != tree_binding.st_dev
                or binding.st_ino != tree_binding.st_ino
            ):
                raise TargetSealedError(f"{role} tree changed before pin")
            binding = tree_binding
        directory_bindings[role] = binding
        descriptor_bindings.append((role, descriptor, binding))
        kind = "ro_bind_fd" if readonly else "rw_bind_fd"
        operations.append((kind, descriptor, destination_string, None))
        source_document = (
            _state_directory_mount_source(binding, state_row, label=role)
            if state_row is not None
            else binding.document()
        )
        mount_entries.append(
            _mount_entry(
                kind=kind,
                destination=destination_string,
                source=source_document,
            )
        )
        return binding

    def add_file(role: str, path: Path) -> tuple[FileBinding, bytes, int]:
        resolved = _canonical_existing(path, label=role)
        prior = pinned_file_destinations.get(str(resolved))
        if prior is not None:
            return prior
        descriptor, binding, raw = _open_pinned_file(
            resolved, label=role, require_immutable=True
        )
        descriptors.append(descriptor)
        descriptor_bindings.append((role, descriptor, binding))
        operations.append(("ro_bind_fd", descriptor, str(binding.path), None))
        mount_entries.append(
            _mount_entry(
                kind="ro_bind_fd",
                destination=binding.path,
                source=binding.document(),
            )
        )
        pinned_file_destinations[str(binding.path)] = (binding, raw, descriptor)
        return binding, raw, descriptor

    try:
        scripts_root = _canonical_existing(project_root / "scripts", label="project scripts")
        src_root = _canonical_existing(project_root / "src", label="project source")
        tests_root = _canonical_existing(project_root / "tests", label="project tests")
        configs_root = _canonical_existing(project_root / "configs", label="project configs")
        venv_root = _canonical_existing(request.venv_root, label="Python virtualenv")
        python_runtime_root = _canonical_existing(
            request.python_runtime_root, label="uv Python runtime"
        )
        if not _path_within(interpreter_lexical, venv_root) and not _path_within(
            interpreter_real, python_runtime_root
        ):
            raise TargetSealedError("interpreter is outside the admitted venv/runtime")

        add_directory("system_usr", Path("/usr"), readonly=True)
        add_directory("system_sys", Path("/sys"), readonly=True)
        add_directory("python_runtime", python_runtime_root, readonly=True)
        add_directory("python_venv", venv_root, readonly=True)
        # A uv virtualenv commonly points at a version alias rather than the
        # resolved interpreter root.  Bind the pinned runtime at that exact alias
        # destination without ever following the alias inside the sandbox.
        if interpreter_lexical.is_symlink():
            raw_target = os.readlink(interpreter_lexical)
            alias_binary = Path(raw_target)
            if not alias_binary.is_absolute():
                alias_binary = interpreter_lexical.parent / alias_binary
            alias_root = _absolute_lexical(alias_binary, label="uv Python alias").parent.parent
            if alias_root != python_runtime_root:
                descriptor, binding = _open_pinned_directory(
                    python_runtime_root, label="uv Python runtime alias source"
                )
                descriptors.append(descriptor)
                descriptor_bindings.append(("python_runtime_alias", descriptor, binding))
                operations.append(("ro_bind_fd", descriptor, str(alias_root), None))
                mount_entries.append(
                    _mount_entry(
                        kind="ro_bind_fd",
                        destination=alias_root,
                        source={**binding.document(), "alias_destination": str(alias_root)},
                    )
                )

        pack_binding: DirectoryBinding | None = None
        pack_index_binding: FileBinding | None = None
        pack_index_document: dict[str, Any] | None = None
        if request.pack_root is not None:
            sealed_tree = _tree_binding(
                request.pack_root, label="sealed-pack shard", frozen=True
            )
            pack_binding = add_directory(
                "sealed_pack_shard",
                sealed_tree.path,
                readonly=True,
                tree_binding=sealed_tree,
            )

        governance_documents: dict[str, dict[str, Any]] = {}
        governance_fds: dict[str, int] = {}
        governance_raw: dict[str, bytes] = {}
        for role, path in sorted(request.governance_files.items()):
            binding, raw, descriptor = add_file(f"governance:{role}", path)
            governance_bindings[role] = binding
            governance_raw[role] = raw
            governance_fds[role] = descriptor
            if not role.startswith("quarantined_material_"):
                governance_documents[role] = _validate_governance_json(raw, role=role)
        snapshot = governance_documents["source_snapshot"]
        snapshot_seen: dict[Path, tuple[str, int, int]] = {}
        source_snapshot_bindings: dict[Path, FileBinding] = {}
        for number, row in enumerate(_snapshot_file_rows(snapshot)):
            path, expected_sha256, expected_bytes, expected_mode = _validate_snapshot_row(
                row, project_root=project_root
            )
            expected_identity = (expected_sha256, expected_bytes, expected_mode)
            if path in snapshot_seen:
                if snapshot_seen[path] != expected_identity:
                    raise TargetSealedError(
                        "source snapshot repeats one path with conflicting bytes"
                    )
                continue
            snapshot_seen[path] = expected_identity
            binding, _raw, _descriptor = add_file(
                f"source_snapshot_material:{number:03d}", path
            )
            if (
                binding.sha256 != expected_sha256
                or binding.bytes != expected_bytes
                or binding.mode != expected_mode
            ):
                raise TargetSealedError("source snapshot frozen material drifted")
            source_snapshot_bindings[path] = binding
        _validate_v8r4a_governance_chain(
            project_root=project_root,
            documents=governance_documents,
            bindings=governance_bindings,
            production=request.production,
        )
        required_snapshot_paths = {
            launcher,
            _canonical_existing(Path(request.command[1]), label="campaign entry script"),
            _canonical_existing(
                project_root / MIGRATION_MODULE_RELATIVE_PATH,
                label="V8R4A migration validator module",
            ),
            _canonical_existing(
                project_root / GPU_ADMISSION_WRAPPER_RELATIVE_PATH,
                label="GPU admission wrapper",
            ),
            _canonical_existing(
                project_root / GPU_BUDGET_MODULE_RELATIVE_PATH,
                label="GPU budget module",
            ),
            _canonical_existing(
                project_root / SNN_RR_PACKAGE_INIT_RELATIVE_PATH,
                label="snn_rr package initializer",
            ),
        }
        if not required_snapshot_paths <= set(source_snapshot_bindings):
            raise TargetSealedError(
                "source snapshot omits a runtime-executable project file"
            )
        migration_receipt_binding = governance_bindings[
            "gpu_state_migration_receipt"
        ]
        # Prove the immutable historical floor before recovery is allowed to
        # append anything.  Closure is intentionally deferred until the one
        # target-owned dead-lifecycle recovery opportunity has completed.
        _validate_active_pretrain_live_ledger_prefixes(
            project_root=project_root,
            runtime_ledger_prefixes=governance_documents[
                "active_authorization"
            ].get("runtime_ledger_prefixes"),
            live_state=None,
            require_closed=False,
        )
        _recover_dead_gpu_lifecycle_before_closed_validation(
            project_root=project_root,
            receipt_binding=migration_receipt_binding,
            source_snapshot_bindings=source_snapshot_bindings,
        )
        prelaunch_state = _validate_migrated_state_live(
            project_root=project_root,
            receipt_binding=migration_receipt_binding,
        )
        _validate_active_pretrain_live_ledger_prefixes(
            project_root=project_root,
            runtime_ledger_prefixes=governance_documents[
                "active_authorization"
            ].get("runtime_ledger_prefixes"),
            live_state=prelaunch_state,
            require_closed=True,
        )
        gpu_state_root_path = project_root / GPU_STATE_ROOT_RELATIVE
        gpu_state_root_binding = add_directory(
            "gpu_state_parent",
            gpu_state_root_path,
            readonly=True,
            state_row=prelaunch_state.directory_bindings["root"],
        )
        gpu_state_root_descriptor = next(
            descriptor
            for label, descriptor, binding in reversed(descriptor_bindings)
            if label == "gpu_state_parent" and binding == gpu_state_root_binding
        )
        _revalidate_state_directory_descriptor(
            gpu_state_root_binding,
            gpu_state_root_descriptor,
            prelaunch_state.directory_bindings["root"],
            label="gpu_state_parent",
        )
        if request.phase == "discovery":
            _validate_quarantined_material_cover(
                project_root=project_root,
                governance_documents=governance_documents,
                governance_bindings=governance_bindings,
            )
        if request.phase in {"discovery_aggregation", "promotion_aggregation"}:
            _validate_aggregation_shard_seals(
                governance_documents,
                governance_bindings,
                entry_name=Path(request.command[1]).name,
                project_root=project_root,
            )
        if request.pack_index is not None:
            pack_index_binding = governance_bindings["sealed_pack_index"]
            if pack_index_binding.path != _canonical_existing(
                request.pack_index, label="sealed-pack index"
            ):
                raise TargetSealedError("sealed-pack index CLI/governance paths differ")
            assert pack_binding is not None and request.outer_fold is not None
            pack_index_document = _validate_pack_index(
                raw=governance_raw["sealed_pack_index"],
                binding=pack_index_binding,
                pack_root=pack_binding,
                phase=request.phase,
                outer_fold=request.outer_fold,
                promotion_authorization_binding=governance_bindings.get(
                    "promotion_authorization"
                ),
                selection_lock_binding=governance_bindings.get("selection_lock"),
            )

        writable_bindings: dict[str, DirectoryBinding] = {}
        gpu_state_child_descriptors: dict[str, int] = {}
        for role in sorted(WRITABLE_DIRECTORY_ROLES):
            binding = add_directory(
                f"writable:{role}",
                request.writable_roots[role],
                readonly=role == "lifecycle",
                state_row=(
                    prelaunch_state.directory_bindings[role]
                    if role in GPU_STATE_DIRECTORY_ROLES
                    else None
                ),
            )
            writable_bindings[role] = binding
        if writable_bindings["output"].path == writable_bindings["lifecycle"].path:
            raise TargetSealedError(
                "output and lifecycle must be distinct dedicated directories"
            )
        writable_identities = {
            role: (binding.st_dev, binding.st_ino)
            for role, binding in writable_bindings.items()
        }
        if len({writable_identities[role] for role in GPU_STATE_DIRECTORY_ROLES}) != 3:
            raise TargetSealedError("V8R4A state directory capabilities alias")
        if (gpu_state_root_binding.st_dev, gpu_state_root_binding.st_ino) in {
            writable_identities[role] for role in GPU_STATE_DIRECTORY_ROLES
        }:
            raise TargetSealedError("V8R4A state parent aliases a mutable child")
        if len(set(writable_identities.values())) != len(writable_identities):
            raise TargetSealedError("output, lifecycle, or state directories alias")
        for role in GPU_STATE_DIRECTORY_ROLES:
            binding = writable_bindings[role]
            migrated = prelaunch_state.directory_bindings[role]
            if not (
                binding.mode == 0o700
                and binding.st_dev == migrated["st_dev"]
                and binding.st_ino == migrated["st_ino"]
                and binding.path
                == project_root / GPU_STATE_DIRECTORY_RELATIVE_PATHS[role]
            ):
                raise TargetSealedError(
                    f"writable:{role} does not match validated migrated state"
                )
            child_descriptor = next(
                descriptor
                for label, descriptor, candidate in descriptor_bindings
                if label == f"writable:{role}" and candidate == binding
            )
            gpu_state_child_descriptors[role] = child_descriptor
            _revalidate_state_directory_descriptor(
                binding,
                child_descriptor,
                migrated,
                label=f"writable:{role}",
            )
        receipt_path = _absolute_lexical(
            request.capability_receipt, label="capability receipt"
        )
        lifecycle_root = writable_bindings["lifecycle"].path
        if receipt_path.parent != lifecycle_root:
            raise TargetSealedError("capability receipt must be a direct lifecycle-root file")
        if receipt_path.name != CAPABILITY_RECEIPT_FILENAME:
            raise TargetSealedError("capability receipt filename is not canonical V8R4A")
        lifecycle_entries = set(os.listdir(lifecycle_root))
        allowed_initial_lifecycle = (
            {receipt_path.name} if os.path.lexists(receipt_path) else set()
        )
        if lifecycle_entries != allowed_initial_lifecycle:
            raise TargetSealedError(
                "dedicated lifecycle attempt root is not empty or capability-receipt-only"
            )
        output_root = writable_bindings["output"].path
        prelaunch_output_inventory = _immutable_output_inventory(output_root)

        cuda_runtime_bindings: list[DirectoryBinding] = []
        for number, path in enumerate(request.cuda_runtime_roots):
            binding = add_directory(
                f"cuda_runtime:{number}", path, readonly=True
            )
            cuda_runtime_bindings.append(binding)

        cuda_device_rows: list[dict[str, Any]] = []
        cuda_device_fds: list[int] = []
        for path in request.cuda_devices:
            descriptor, binding = _open_special_binding(path, label="CUDA device")
            descriptors.append(descriptor)
            cuda_device_fds.append(descriptor)
            cuda_device_rows.append(binding)
            operations.append(("dev_bind_fd", descriptor, binding["path"], None))
            mount_entries.append(
                _mount_entry(
                    kind="dev_bind_fd", destination=binding["path"], source=binding
                )
            )
        if request.production:
            basenames = {Path(row["path"]).name for row in cuda_device_rows}
            if not {"nvidiactl", "nvidia-uvm"} <= basenames or not any(
                name.startswith("nvidia") and name[6:].isdigit() for name in basenames
            ):
                raise TargetSealedError("production CUDA device capability is incomplete")

        # Expose only an empty project directory skeleton plus exact frozen
        # files named by the active source snapshot/governance chain.  Binding
        # the host scripts/src/tests/configs trees would leak executable or
        # configuration bytes outside the active V8R4A source closure.
        skeleton_directories: set[Path] = {
            project_root,
            scripts_root,
            src_root,
            tests_root,
            configs_root,
        }
        project_destinations = {
            *(binding.path for binding in governance_bindings.values()),
            *source_snapshot_bindings,
            *((pack_binding.path,) if pack_binding is not None else ()),
            *(
                binding.path
                for role, binding in writable_bindings.items()
                if _path_within(binding.path, project_root)
            ),
        }
        for destination in project_destinations:
            # Every destination itself is supplied by an exact file/directory
            # bind; only its otherwise-empty ancestors belong in the skeleton.
            cursor = destination.parent
            while cursor != project_root and _path_within(cursor, project_root):
                skeleton_directories.add(cursor)
                cursor = cursor.parent
            skeleton_directories.add(project_root)
        # The exact migrated GPU-state parent is a descriptor-backed mount,
        # never a synthetic bubblewrap ``--dir``.  Its ordinary ancestors
        # remain part of the empty project skeleton.
        skeleton_directories.discard(gpu_state_root_binding.path)
        for directory in sorted(
            skeleton_directories, key=lambda value: (len(value.parts), str(value))
        ):
            operations.append(("directory", None, str(directory), None))
            mount_entries.append(
                _mount_entry(kind="directory", destination=directory)
            )

        # Canonical root and pseudo-filesystem operations have no host data
        # capability beyond their documented sources.
        for link_path in (Path("/bin"), Path("/sbin"), Path("/lib"), Path("/lib64")):
            if not link_path.is_symlink():
                raise TargetSealedError(f"system compatibility link drifted: {link_path}")
            target = os.readlink(link_path)
            if Path(target).is_absolute() or ".." in PurePosixPath(target).parts:
                raise TargetSealedError(f"unsafe system compatibility link: {link_path}")
            operations.append(("symlink", None, str(link_path), target))
            mount_entries.append(
                _mount_entry(kind="symlink", destination=link_path, source={"target": target})
            )
        for path in (Path("/proc"), Path("/dev"), Path("/dev/shm"), Path("/tmp")):
            kind = {
                "/proc": "proc",
                "/dev": "dev",
                "/dev/shm": "tmpfs",
                "/tmp": "tmpfs",
            }[str(path)]
            operations.append((kind, None, str(path), None))
            mount_entries.append(_mount_entry(kind=kind, destination=path))

        ro_roots = [
            venv_root,
            python_runtime_root,
            gpu_state_root_binding.path,
            *source_snapshot_bindings,
            *(binding.path for binding in cuda_runtime_bindings),
        ]
        if pack_binding is not None:
            ro_roots.append(pack_binding.path)
        denied_canaries = _validate_mount_boundaries(
            ro_roots=ro_roots,
            rw_roots=[binding.path for binding in writable_bindings.values()],
            governance_files=[binding.path for binding in governance_bindings.values()],
            denied_canaries=request.denied_canaries,
            gpu_state_readonly_parent=gpu_state_root_binding.path,
            gpu_state_mutable_children=[
                writable_bindings[role].path
                for role in sorted(GPU_STATE_DIRECTORY_ROLES)
            ],
        )
        _validate_command_paths(
            request.command,
            project_root=project_root,
            mounted_roots=[
                Path("/usr"),
                venv_root,
                python_runtime_root,
                *source_snapshot_bindings,
                *(binding.path for binding in writable_bindings.values()),
                *(binding.path for binding in cuda_runtime_bindings),
                *((pack_binding.path,) if pack_binding is not None else ()),
                *(binding.path for binding in governance_bindings.values()),
            ],
        )
        _validate_command_denied_canaries(
            request.command, denied_canaries=denied_canaries
        )

        child_environment = _canonical_child_environment(
            project_root=project_root,
            propagated=request.propagated_environment,
        )
        # The private /tmp and fresh /dev must be created before capabilities
        # whose destinations live below them; CUDA device fd mounts must be last.
        def operation_priority(row: tuple[str, int | None, str, str | None]) -> int:
            kind, _descriptor, destination, _auxiliary = row
            if kind == "ro_bind_fd" and destination == "/usr":
                return 0
            if kind == "symlink":
                return 1
            if kind in {"proc", "dev"}:
                return 2
            if kind == "tmpfs":
                return 3
            if kind == "directory":
                return 4
            if destination == str(gpu_state_root_binding.path):
                return 5
            if destination in {
                str(writable_bindings[role].path)
                for role in GPU_STATE_DIRECTORY_ROLES
            }:
                return 6
            if kind == "dev_bind_fd":
                return 7
            return 5

        def entry_priority(row: Mapping[str, Any]) -> int:
            kind = row.get("kind")
            destination = row.get("destination")
            if kind == "ro_bind_fd" and destination == "/usr":
                return 0
            if kind == "symlink":
                return 1
            if kind in {"proc", "dev"}:
                return 2
            if kind == "tmpfs":
                return 3
            if kind == "directory":
                return 4
            if destination == str(gpu_state_root_binding.path):
                return 5
            if destination in {
                str(writable_bindings[role].path)
                for role in GPU_STATE_DIRECTORY_ROLES
            }:
                return 6
            if kind == "dev_bind_fd":
                return 7
            return 5

        operations = sorted(operations, key=operation_priority)
        mount_entries = sorted(mount_entries, key=entry_priority)
        expected_state_destinations = {
            role: str(project_root / GPU_STATE_DIRECTORY_RELATIVE_PATHS[role])
            for role in sorted(GPU_STATE_DIRECTORY_ROLES)
        }
        expected_state_root = str(gpu_state_root_binding.path)
        state_mounts = [
            row
            for row in mount_entries
            if row.get("destination")
            in {expected_state_root, *expected_state_destinations.values()}
        ]
        if not (
            len(state_mounts) == 4
            and [row.get("destination") for row in state_mounts]
            == [
                expected_state_root,
                *(expected_state_destinations[role] for role in sorted(GPU_STATE_DIRECTORY_ROLES)),
            ]
            and state_mounts[0].get("kind") == "ro_bind_fd"
            and all(row.get("kind") == "rw_bind_fd" for row in state_mounts[1:])
            and all(row.get("kind") != "rw_bind_file_fd" for row in mount_entries)
        ):
            raise TargetSealedError(
                "V8R4A requires one parent RO mount before exactly three child RW mounts"
            )
        for runtime_dir in (
            "/run",
            "/run/snn_rr",
            "/tmp/home",
            "/tmp/cache",
            "/tmp/torch",
            "/tmp/triton",
            "/tmp/numba",
            "/tmp/pycache",
        ):
            mount_entries.append(_mount_entry(kind="directory", destination=runtime_dir))
        mount_entries.append(
            _mount_entry(
                kind="ro_bind_data",
                destination=INTERNAL_SPEC_PATH,
                source={"classification": "v8r4a_internal_guard_spec_memfd"},
            )
        )
        mount_spec_sha256 = semantic_sha256(mount_entries)
        proposed_receipt = _runtime_receipt_document(
            request=request,
            bwrap_binding=bwrap_binding,
            launcher_binding=launcher_binding,
            interpreter_binding=interpreter_binding,
            pack_binding=pack_binding,
            pack_index_binding=pack_index_binding,
            pack_index_document=pack_index_document,
            governance_bindings=governance_bindings,
            writable_bindings=writable_bindings,
            prelaunch_state=prelaunch_state,
            denied_canaries=denied_canaries,
            mount_entries=mount_entries,
            child_environment=child_environment,
        )
        receipt = proposed_receipt
        try:
            receipt_binding = _create_once_immutable_json(
                receipt_path, proposed_receipt
            )
        except TargetSealedError as publication_error:
            # A killed child reuses the same immutable lifecycle capability.
            # Only the closed append-only ledger prefix may have advanced; all
            # other capability bytes must still match the newly pinned launch.
            if not os.path.lexists(receipt_path):
                raise
            try:
                prior = validate_capability_receipt(
                    receipt_path,
                    expected_phase=request.phase,
                    expected_outer_fold=request.outer_fold,
                    expected_mount_spec_sha256=mount_spec_sha256,
                    expected_command_sha256=semantic_sha256(list(request.command)),
                )
            except TargetSealedError as validation_error:
                raise publication_error from validation_error
            prior_document = prior["document"]
            static_keys = RECEIPT_KEYS - {
                "prelaunch_gpu_state",
                "content_sha256",
            }
            if any(
                prior_document.get(key) != proposed_receipt.get(key)
                for key in static_keys
            ):
                raise TargetSealedError(
                    "existing lifecycle capability differs outside append-only state"
                )
            if prior_document["security_boundary"].get(
                "production_execution_authorized"
            ) is not request.production:
                raise TargetSealedError(
                    "existing lifecycle capability production boundary drifted"
                )
            recorded_prelaunch = _state_snapshot_from_document(
                prior_document.get("prelaunch_gpu_state"),
                label="existing capability prelaunch GPU state",
            )
            _require_closed_append_only_completion_state(
                project_root=project_root,
                recorded=recorded_prelaunch,
                current=prelaunch_state,
            )
            receipt_binding, prior_raw = _read_file_binding(
                receipt_path,
                label="existing target-sealed capability receipt",
                require_immutable=True,
            )
            if (
                receipt_binding.document() != prior["binding"]
                or prior_raw != canonical_json_bytes(prior_document) + b"\n"
            ):
                raise TargetSealedError(
                    "existing lifecycle capability changed during retry admission"
                )
            receipt = prior_document
        available_paths = sorted(
            {
                str(project_root),
                str(scripts_root),
                str(src_root),
                str(tests_root),
                str(configs_root),
                str(venv_root),
                str(python_runtime_root),
                str(gpu_state_root_binding.path),
                str(receipt_binding.path),
                *(str(path) for path in source_snapshot_bindings),
                *(str(binding.path) for binding in governance_bindings.values()),
                *(str(binding.path) for binding in writable_bindings.values()),
                *((str(pack_binding.path),) if pack_binding is not None else ()),
            }
        )
        internal_spec = _internal_spec_document(
            request=request,
            receipt_binding=receipt_binding,
            mount_spec_sha256=mount_spec_sha256,
            child_environment=child_environment,
            denied_canaries=denied_canaries,
            available_paths=available_paths,
        )
        spec_fd = _create_memfd(canonical_json_bytes(internal_spec) + b"\n")
        descriptors.append(spec_fd)

        bwrap_command: list[str] = [
            str(bwrap_binding.path),
            "--die-with-parent",
            "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--cap-drop",
            "ALL",
            "--clearenv",
        ]
        for kind, descriptor, destination, auxiliary in operations:
            if kind == "ro_bind_fd":
                assert descriptor is not None
                bwrap_command.extend(["--ro-bind-fd", str(descriptor), destination])
            elif kind == "rw_bind_fd":
                assert descriptor is not None
                bwrap_command.extend(["--bind-fd", str(descriptor), destination])
            elif kind == "dev_bind_fd":
                assert descriptor is not None
                bwrap_command.extend(["--bind-fd", str(descriptor), destination])
            elif kind == "symlink":
                assert auxiliary is not None
                bwrap_command.extend(["--symlink", auxiliary, destination])
            elif kind == "proc":
                bwrap_command.extend(["--proc", destination])
            elif kind == "dev":
                bwrap_command.extend(["--dev", destination])
            elif kind == "tmpfs":
                bwrap_command.extend(["--tmpfs", destination])
            elif kind == "directory":
                bwrap_command.extend(["--dir", destination])
            else:
                raise AssertionError(f"unhandled mount operation: {kind}")
        bwrap_command.extend(
            [
                "--dir",
                "/run",
                "--dir",
                "/run/snn_rr",
                "--perms",
                "0400",
                "--ro-bind-data",
                str(spec_fd),
                str(INTERNAL_SPEC_PATH),
                "--dir",
                "/tmp/home",
                "--dir",
                "/tmp/cache",
                "--dir",
                "/tmp/torch",
                "--dir",
                "/tmp/triton",
                "--dir",
                "/tmp/numba",
                "--dir",
                "/tmp/pycache",
            ]
        )
        for name, value in child_environment.items():
            bwrap_command.extend(["--setenv", name, value])
        bwrap_command.extend(
            [
                "--chdir",
                str(project_root),
                "--",
                str(interpreter_lexical),
                str(launcher),
                "--internal-guard",
                str(INTERNAL_SPEC_PATH),
            ]
        )

        for label, descriptor, binding in descriptor_bindings:
            if isinstance(binding, DirectoryBinding):
                _directory_revalidate(binding, descriptor, label=label)
            else:
                _file_revalidate(binding, descriptor, label=label)
        if pack_binding is not None:
            refreshed_tree = _tree_binding(
                pack_binding.path, label="sealed-pack shard", frozen=True
            )
            if refreshed_tree != pack_binding:
                raise TargetSealedError("sealed-pack shard changed before launch")
        refreshed_prelaunch = _validate_migrated_state_live(
            project_root=project_root,
            receipt_binding=migration_receipt_binding,
        )
        _validate_active_pretrain_live_ledger_prefixes(
            project_root=project_root,
            runtime_ledger_prefixes=governance_documents[
                "active_authorization"
            ].get("runtime_ledger_prefixes"),
            live_state=refreshed_prelaunch,
            require_closed=True,
        )
        if refreshed_prelaunch != prelaunch_state:
            raise TargetSealedError("migrated GPU state changed while preparing launch")
        _revalidate_state_directory_descriptor(
            gpu_state_root_binding,
            gpu_state_root_descriptor,
            prelaunch_state.directory_bindings["root"],
            label="gpu_state_parent",
        )
        if _immutable_output_inventory(output_root) != prelaunch_output_inventory:
            raise TargetSealedError("shard output changed while preparing launch")
        if request.production:
            _close_guard_runtime_noise_fds(
                allowed={0, 1, 2, *descriptors}
            )
            audit_process_fds(allowed={0, 1, 2, *descriptors})
        return PreparedRuntime(
            request=request,
            descriptors=descriptors,
            mount_entries=mount_entries,
            directory_bindings=directory_bindings,
            governance_bindings=governance_bindings,
            source_snapshot_bindings=source_snapshot_bindings,
            writable_bindings=writable_bindings,
            gpu_state_root_binding=gpu_state_root_binding,
            gpu_state_root_descriptor=gpu_state_root_descriptor,
            gpu_state_child_descriptors=gpu_state_child_descriptors,
            pack_binding=pack_binding,
            pack_index_binding=pack_index_binding,
            migration_receipt_binding=migration_receipt_binding,
            prelaunch_state=prelaunch_state,
            prelaunch_output_inventory=prelaunch_output_inventory,
            child_environment=child_environment,
            mount_spec_sha256=mount_spec_sha256,
            command_sha256=semantic_sha256(list(request.command)),
            receipt=receipt,
            bwrap_command=bwrap_command,
            spec_fd=spec_fd,
        )
    except BaseException:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "phase",
        "outer_fold",
        "bubblewrap",
        "launcher",
        "interpreter",
        "sealed_pack_root",
        "sealed_pack_index",
        "governance_files",
        "writable_roots",
        "prelaunch_gpu_state",
        "denied_canaries",
        "mount_specification",
        "mount_specification_sha256",
        "environment",
        "environment_sha256",
        "command",
        "command_sha256",
        "security_boundary",
        "content_sha256",
    }
)
INTERNAL_SPEC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "phase",
        "outer_fold",
        "capability_receipt",
        "mount_specification_sha256",
        "environment",
        "environment_sha256",
        "denied_canaries",
        "available_paths",
        "command",
        "command_sha256",
        "required_open_fds",
        "forbidden_environment",
        "content_sha256",
    }
)
SECURITY_BOUNDARY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "outer_campaign_runtime",
        "network_namespace_unshared",
        "ipc_namespace_unshared",
        "uts_namespace_unshared",
        "pid_namespace_unshared",
        "new_session_created",
        "tmp_is_private_tmpfs",
        "die_with_parent",
        "capabilities_dropped",
        "environment_cleared_before_allowlist",
        "hai_experiment_propagated",
        "legacy_combined_cache_mounted",
        "raw_or_target_root_mounted",
        "cross_outer_shard_mounted",
        "other_pack_or_output_mounted",
        "admitted_child_fd_created_or_consumed_by_outer_launcher",
        "admission_lock_fd_created_or_consumed_by_outer_launcher",
        "admitted_fd_direct_watchdog_to_trainer_contract_preserved",
        "child_fd_audit_required",
        "denied_canary_probe_required",
        "target_reference_or_selection_evidence_accessed",
        "commercial_claim_authorized",
        "production_execution_authorized",
        "atomic_replace_compatible",
        "synthetic_validation_only",
        "v8r4a_ledger_migration_required",
        "v8r4a_migration_live_replay_validated",
        "dedicated_gpu_state_directory_capabilities",
        "gpu_state_parent_identity_readonly_bind",
        "exactly_three_mutable_state_directory_mounts",
        "benchmark_admitted_context_generation_isolated",
        "active_pretrain_postfailure_ledger_prefix_enforced",
        "usage_and_execution_closed_prelaunch",
        "lifecycle_mounted_read_only",
        "complete_project_source_or_config_trees_mounted",
        "source_snapshot_exact_file_mounts",
    }
)

STATE_SNAPSHOT_KEYS: Final[frozenset[str]] = frozenset(
    {"migration_receipt", "directories", "files", "usage_state", "execution_state"}
)


def _state_snapshot_from_document(value: object, *, label: str) -> LiveStateSnapshot:
    if not isinstance(value, dict) or set(value) != STATE_SNAPSHOT_KEYS:
        raise TargetSealedError(f"{label} schema drifted")
    receipt = value.get("migration_receipt")
    directories = value.get("directories")
    files = value.get("files")
    usage = value.get("usage_state")
    execution = value.get("execution_state")
    if not (
        isinstance(receipt, dict)
        and set(receipt)
        == {
            "bytes",
            "content_sha256",
            "mode",
            "nlink",
            "path",
            "sha256",
            "st_dev",
            "st_ino",
        }
        and receipt.get("mode") == "0444"
        and receipt.get("nlink") == 1
        and _is_sha256(receipt.get("sha256"))
        and _is_sha256(receipt.get("content_sha256"))
        and isinstance(directories, dict)
        and set(directories) == {"root", *GPU_STATE_DIRECTORY_ROLES}
        and isinstance(files, dict)
        and set(files) == set(GPU_STATE_FILE_ROLES)
        and isinstance(usage, dict)
        and usage.get("open_reservation_count") == 0
        and isinstance(execution, dict)
        and execution.get("open_start_count") == 0
    ):
        raise TargetSealedError(f"{label} semantic binding drifted")
    for role, row in directories.items():
        if not (
            isinstance(row, dict)
            and set(row)
            == {"exact_entries", "mode", "path", "st_dev", "st_ino"}
            and row.get("mode") == "0700"
        ):
            raise TargetSealedError(f"{label} {role} directory binding drifted")
    for role, row in files.items():
        if not (
            isinstance(row, dict)
            and set(row)
            == {"bytes", "mode", "nlink", "path", "sha256", "st_dev", "st_ino"}
            and row.get("mode") == "0644"
            and row.get("nlink") == 1
            and _is_sha256(row.get("sha256"))
        ):
            raise TargetSealedError(f"{label} {role} file binding drifted")
    return LiveStateSnapshot(
        receipt_binding=dict(receipt),
        directory_bindings={role: dict(row) for role, row in directories.items()},
        current_file_bindings={role: dict(row) for role, row in files.items()},
        usage_state=dict(usage),
        execution_state=dict(execution),
    )


def validate_capability_receipt(
    path: Path,
    *,
    expected_phase: str | None = None,
    expected_outer_fold: int | None | object = ...,
    expected_mount_spec_sha256: str | None = None,
    expected_command_sha256: str | None = None,
) -> dict[str, Any]:
    binding, raw = _read_file_binding(
        path, label="target-sealed capability receipt", require_immutable=True
    )
    document = _decode_json_bytes(raw, label="target-sealed capability receipt")
    if set(document) != RECEIPT_KEYS:
        raise TargetSealedError("target-sealed receipt schema drifted")
    if raw != canonical_json_bytes(document) + b"\n":
        raise TargetSealedError("target-sealed receipt encoding is non-canonical")
    _validate_self_hash(document, label="target-sealed capability receipt")
    mount_spec = document.get("mount_specification")
    environment = document.get("environment")
    command = document.get("command")
    boundary = document.get("security_boundary")
    if not (
        document.get("schema_version") == 1
        and document.get("classification") == RECEIPT_CLASSIFICATION
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == SCIENTIFIC_CAMPAIGN_REVISION
        and document.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and document.get("phase") in PHASES
        and isinstance(mount_spec, list)
        and document.get("mount_specification_sha256")
        == semantic_sha256(mount_spec)
        and isinstance(environment, dict)
        and document.get("environment_sha256") == semantic_sha256(environment)
        and isinstance(command, list)
        and command
        and all(isinstance(part, str) and part for part in command)
        and document.get("command_sha256") == semantic_sha256(command)
        and isinstance(boundary, dict)
        and set(boundary) == SECURITY_BOUNDARY_KEYS
    ):
        raise TargetSealedError("target-sealed receipt semantic binding drifted")
    prelaunch_state = _state_snapshot_from_document(
        document.get("prelaunch_gpu_state"), label="capability prelaunch GPU state"
    )
    required_true = {
        "outer_campaign_runtime",
        "network_namespace_unshared",
        "ipc_namespace_unshared",
        "uts_namespace_unshared",
        "tmp_is_private_tmpfs",
        "die_with_parent",
        "capabilities_dropped",
        "environment_cleared_before_allowlist",
        "admitted_fd_direct_watchdog_to_trainer_contract_preserved",
        "child_fd_audit_required",
        "denied_canary_probe_required",
        "atomic_replace_compatible",
        "v8r4a_migration_live_replay_validated",
        "dedicated_gpu_state_directory_capabilities",
        "gpu_state_parent_identity_readonly_bind",
        "exactly_three_mutable_state_directory_mounts",
        "benchmark_admitted_context_generation_isolated",
        "active_pretrain_postfailure_ledger_prefix_enforced",
        "usage_and_execution_closed_prelaunch",
        "lifecycle_mounted_read_only",
        "source_snapshot_exact_file_mounts",
    }
    if any(boundary.get(key) is not True for key in required_true):
        raise TargetSealedError("target-sealed receipt positive boundary drifted")
    dynamic = {"production_execution_authorized", "synthetic_validation_only"}
    if any(
        boundary.get(key) is not False
        for key in SECURITY_BOUNDARY_KEYS - required_true - dynamic
    ):
        raise TargetSealedError("target-sealed receipt negative boundary drifted")
    if not (
        type(boundary.get("production_execution_authorized")) is bool
        and boundary.get("synthetic_validation_only")
        is (not boundary["production_execution_authorized"])
    ):
        raise TargetSealedError("target-sealed receipt production boundary drifted")
    writable = document.get("writable_roots")
    if not isinstance(writable, dict) or set(writable) != WRITABLE_ROLES:
        raise TargetSealedError("target-sealed writable capability schema drifted")
    state_destinations = {
        role: str(writable[role].get("path"))
        for role in sorted(GPU_STATE_DIRECTORY_ROLES)
        if isinstance(writable.get(role), dict)
    }
    if len(state_destinations) != 3:
        raise TargetSealedError("target-sealed state mount ABI drifted")
    state_root_path = str(Path(state_destinations["admission"]).parent)
    expected_order = [
        state_root_path,
        *(state_destinations[role] for role in sorted(GPU_STATE_DIRECTORY_ROLES)),
    ]
    state_mounts = [
        row
        for row in mount_spec
        if isinstance(row, dict) and row.get("destination") in set(expected_order)
    ]
    expected_rows = {
        state_root_path: prelaunch_state.directory_bindings["root"],
        **{
            state_destinations[role]: prelaunch_state.directory_bindings[role]
            for role in sorted(GPU_STATE_DIRECTORY_ROLES)
        },
    }

    def state_source_matches(row: Mapping[str, Any]) -> bool:
        destination = str(row.get("destination"))
        source = row.get("source")
        expected = expected_rows.get(destination)
        return bool(
            isinstance(source, Mapping)
            and isinstance(expected, Mapping)
            and set(source)
            == {"exact_entries", "mode", "path", "st_dev", "st_ino"}
            and source.get("path") == destination
            and source.get("mode") == expected.get("mode") == "0700"
            and source.get("st_dev") == expected.get("st_dev")
            and source.get("st_ino") == expected.get("st_ino")
            and source.get("exact_entries") == expected.get("exact_entries")
        )

    if not (
        len(state_mounts) == 4
        and [row.get("destination") for row in state_mounts] == expected_order
        and state_mounts[0].get("kind") == "ro_bind_fd"
        and all(row.get("kind") == "rw_bind_fd" for row in state_mounts[1:])
        and all(state_source_matches(row) for row in state_mounts)
        and not any(
            isinstance(row, dict) and row.get("kind") == "rw_bind_file_fd"
            for row in mount_spec
        )
    ):
        raise TargetSealedError("target-sealed state mount ABI drifted")
    lifecycle_row = writable.get("lifecycle")
    output_row = writable.get("output")
    if not isinstance(lifecycle_row, Mapping) or not isinstance(output_row, Mapping):
        raise TargetSealedError("target-sealed output/lifecycle binding drifted")
    lifecycle_path = lifecycle_row.get("path")
    output_path = output_row.get("path")
    lifecycle_mounts = [
        row
        for row in mount_spec
        if isinstance(row, dict) and row.get("destination") == lifecycle_path
    ]
    output_mounts = [
        row
        for row in mount_spec
        if isinstance(row, dict) and row.get("destination") == output_path
    ]
    if not (
        lifecycle_path != output_path
        and len(lifecycle_mounts) == 1
        and lifecycle_mounts[0].get("kind") == "ro_bind_fd"
        and len(output_mounts) == 1
        and output_mounts[0].get("kind") == "rw_bind_fd"
    ):
        raise TargetSealedError("target-sealed output/lifecycle mount ABI drifted")
    phase = str(document["phase"])
    outer_fold = document.get("outer_fold")
    if phase in {"discovery_aggregation", "promotion_aggregation"}:
        if outer_fold is not None or document.get("sealed_pack_root") is not None or document.get("sealed_pack_index") is not None:
            raise TargetSealedError("target-sealed aggregation scope drifted")
    else:
        allowed_outer = {
            "efficiency_benchmark": frozenset({3}),
            "discovery": DISCOVERY_OUTER_FOLDS,
            "promotion_training": frozenset({0, 1, 2, 5}),
            "promotion_prediction": ALL_OUTER_FOLDS,
        }[phase]
        if outer_fold not in allowed_outer:
            raise TargetSealedError("target-sealed receipt outer scope drifted")
    if boundary["production_execution_authorized"] is True:
        entry_path = Path(str(command[1])) if len(command) >= 2 else Path()
        if not (
            entry_path.is_absolute()
            and entry_path.parent.name == "scripts"
            and entry_path.name in ENTRY_SCRIPT_BY_PHASE[phase]
        ):
            raise TargetSealedError("target-sealed receipt entry path drifted")
        project_root = entry_path.parent.parent
        if Path(str(output_path)) != project_root / _canonical_output_relative(
            phase=phase, outer_fold=outer_fold, entry_name=entry_path.name
        ):
            raise TargetSealedError("target-sealed canonical topology drifted")
        if Path(str(lifecycle_path)) != project_root / _canonical_lifecycle_relative(
            phase=phase, outer_fold=outer_fold, entry_name=entry_path.name
        ):
            raise TargetSealedError("target-sealed canonical topology drifted")
        governance = document.get("governance_files")
        if not isinstance(governance, Mapping) or set(governance) != _governance_roles_for(
            phase=phase, entry_name=entry_path.name
        ):
            raise TargetSealedError("target-sealed governance phase/entry cover drifted")
    if expected_phase is not None and document.get("phase") != expected_phase:
        raise TargetSealedError("target-sealed receipt phase drifted")
    if expected_outer_fold is not ... and document.get("outer_fold") != expected_outer_fold:
        raise TargetSealedError("target-sealed receipt outer fold drifted")
    if (
        expected_mount_spec_sha256 is not None
        and document.get("mount_specification_sha256")
        != expected_mount_spec_sha256
    ):
        raise TargetSealedError("target-sealed receipt mount spec drifted")
    if (
        expected_command_sha256 is not None
        and document.get("command_sha256") != expected_command_sha256
    ):
        raise TargetSealedError("target-sealed receipt command drifted")
    return {"document": document, "binding": binding.document()}


def _guard_read(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        status_before = os.fstat(descriptor)
        if not stat.S_ISREG(status_before.st_mode):
            raise TargetSealedError(f"{label} is not a regular file")
        raw = bytearray()
        while True:
            block = os.read(descriptor, 65536)
            if not block:
                break
            raw.extend(block)
        status_after = os.fstat(descriptor)
        if (
            (status_before.st_dev, status_before.st_ino, status_before.st_size)
            != (status_after.st_dev, status_after.st_ino, status_after.st_size)
            or (status_after.st_size > 0 and len(raw) != status_after.st_size)
        ):
            raise TargetSealedError(f"{label} changed while reading")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _unescape_mountinfo(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _mount_table() -> dict[str, list[frozenset[str]]]:
    raw = _guard_read(Path("/proc/self/mountinfo"), label="mountinfo")
    result: dict[str, list[frozenset[str]]] = {}
    for line in raw.decode("utf-8", errors="strict").splitlines():
        fields = line.split(" ")
        if len(fields) < 10 or "-" not in fields:
            raise TargetSealedError("mountinfo row is malformed")
        separator = fields.index("-")
        if separator < 6 or len(fields) <= separator + 3:
            raise TargetSealedError("mountinfo row schema is malformed")
        mount_point = _unescape_mountinfo(fields[4])
        options = frozenset(fields[5].split(","))
        if not options or not ({"ro", "rw"} & options):
            raise TargetSealedError("mountinfo row lacks access mode")
        result.setdefault(mount_point, []).append(options)
    return result


def _mount_points() -> set[str]:
    return set(_mount_table())


def _validate_live_gpu_state_mounts(
    mount_table: Mapping[str, Sequence[frozenset[str]]],
    receipt: Mapping[str, Any],
) -> None:
    """Verify live access modes and the exact GPU-state directory identities."""

    writable = receipt.get("writable_roots")
    state = receipt.get("prelaunch_gpu_state")
    mounts = receipt.get("mount_specification")
    if not (
        isinstance(writable, Mapping)
        and isinstance(state, Mapping)
        and isinstance(state.get("directories"), Mapping)
        and isinstance(mounts, list)
    ):
        raise TargetSealedError("live GPU-state mount evidence is incomplete")
    child_paths = {
        role: str(writable[role]["path"])
        for role in sorted(GPU_STATE_DIRECTORY_ROLES)
    }
    root_path = str(Path(child_paths["admission"]).parent)
    expected_paths = [
        root_path,
        *(child_paths[role] for role in sorted(GPU_STATE_DIRECTORY_ROLES)),
    ]
    state_mounts = [
        row
        for row in mounts
        if isinstance(row, Mapping) and row.get("destination") in set(expected_paths)
    ]
    if not (
        len(state_mounts) == 4
        and [row.get("destination") for row in state_mounts] == expected_paths
        and state_mounts[0].get("kind") == "ro_bind_fd"
        and all(row.get("kind") == "rw_bind_fd" for row in state_mounts[1:])
    ):
        raise TargetSealedError("live GPU-state mount order drifted")
    expected_access = {root_path: "ro"}
    expected_access.update({path: "rw" for path in child_paths.values()})
    directory_rows = state["directories"]
    expected_rows: dict[str, Mapping[str, Any]] = {
        root_path: directory_rows["root"],
        **{child_paths[role]: directory_rows[role] for role in child_paths},
    }

    def require_access(path: str, access: str, *, label: str) -> None:
        rows = mount_table.get(path)
        opposite = "rw" if access == "ro" else "ro"
        if not (
            isinstance(rows, Sequence)
            and len(rows) == 1
            and access in rows[0]
            and opposite not in rows[0]
        ):
            raise TargetSealedError(f"live {label} mount access mode drifted: {path}")

    for path in expected_paths:
        access = expected_access[path]
        require_access(path, access, label="GPU-state")
        expected = expected_rows[path]
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            entries = sorted(os.listdir(descriptor))
            after = os.fstat(descriptor)
            named = os.stat(path, follow_symlinks=False)
        finally:
            os.close(descriptor)
        if not (
            stat.S_ISDIR(before.st_mode)
            and (before.st_dev, before.st_ino)
            == (after.st_dev, after.st_ino)
            == (named.st_dev, named.st_ino)
            == (expected.get("st_dev"), expected.get("st_ino"))
            and _mode(before) == _mode(after) == _mode(named) == 0o700
            and expected.get("mode") == "0700"
            and entries == expected.get("exact_entries")
        ):
            raise TargetSealedError(
                f"live GPU-state mount identity/inventory drifted: {path}"
            )

    # These access modes are part of the same production boundary: the child
    # may append results only below its dedicated output root, while its
    # immutable capability receipt remains protected by a read-only lifecycle
    # directory.  Check the live namespace rather than trusting argv/receipt
    # projection alone.
    for role, access in (("lifecycle", "ro"), ("output", "rw")):
        row = writable.get(role)
        if not (
            isinstance(row, Mapping)
            and isinstance(row.get("path"), str)
            and type(row.get("st_dev")) is int
            and type(row.get("st_ino")) is int
            and isinstance(row.get("mode"), str)
            and re.fullmatch(r"0[0-7]{3}", str(row["mode"])) is not None
        ):
            raise TargetSealedError(f"live {role} mount evidence is incomplete")
        path = str(row["path"])
        expected_mode = int(str(row["mode"]), 8)
        require_access(path, access, label=role)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            after = os.fstat(descriptor)
            named = os.stat(path, follow_symlinks=False)
        finally:
            os.close(descriptor)
        if not (
            stat.S_ISDIR(before.st_mode)
            and (before.st_dev, before.st_ino)
            == (after.st_dev, after.st_ino)
            == (named.st_dev, named.st_ino)
            == (row["st_dev"], row["st_ino"])
            and _mode(before) == _mode(after) == _mode(named) == expected_mode
        ):
            raise TargetSealedError(f"live {role} mount identity drifted: {path}")


def _probe_denied_canaries(canaries: Mapping[str, str]) -> None:
    for role, value in sorted(canaries.items()):
        path = Path(value)
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            if error.errno not in {
                errno.ENOENT,
                errno.ENOTDIR,
                errno.EACCES,
                errno.EPERM,
                errno.ELOOP,
            }:
                raise TargetSealedError(
                    f"denied canary probe failed unexpectedly: {role}: {error}"
                ) from error
            continue
        else:
            os.close(descriptor)
            raise TargetSealedError(f"denied canary unexpectedly opened: {role}")


def internal_guard(spec_path: Path) -> int:
    """Validate the live namespace, then replace this process with the campaign."""

    _close_guard_runtime_noise_fds()
    audit_process_fds(allowed=(0, 1, 2))
    raw = _guard_read(spec_path, label="target-sealed internal guard spec")
    document = _decode_json_bytes(raw, label="target-sealed internal guard spec")
    if set(document) != INTERNAL_SPEC_KEYS:
        raise TargetSealedError("internal guard spec schema drifted")
    if raw != canonical_json_bytes(document) + b"\n":
        raise TargetSealedError("internal guard spec encoding drifted")
    _validate_self_hash(document, label="target-sealed internal guard spec")
    environment = document.get("environment")
    command = document.get("command")
    canaries = document.get("denied_canaries")
    available = document.get("available_paths")
    if not (
        document.get("schema_version") == 1
        and document.get("classification")
        == "adaptive_v3r1_v8r4a_target_sealed_child_guard_spec"
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == SCIENTIFIC_CAMPAIGN_REVISION
        and document.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and document.get("phase") in PHASES
        and isinstance(environment, dict)
        and document.get("environment_sha256") == semantic_sha256(environment)
        and isinstance(command, list)
        and command
        and document.get("command_sha256") == semantic_sha256(command)
        and isinstance(canaries, dict)
        and set(canaries) >= MANDATORY_DENIED_CANARY_ROLES
        and isinstance(available, list)
        and all(isinstance(path, str) and path.startswith("/") for path in available)
        and document.get("required_open_fds") == [0, 1, 2]
        and document.get("forbidden_environment")
        == sorted(FORBIDDEN_ENV_NAMES | {ADMITTED_CHILD_FD_ENV})
    ):
        raise TargetSealedError("internal guard identity/schema drifted")
    if dict(os.environ) != environment:
        raise TargetSealedError("live child environment is not the exact allowlist")
    if any(name in os.environ for name in FORBIDDEN_ENV_NAMES | {ADMITTED_CHILD_FD_ENV}):
        raise TargetSealedError("forbidden outer environment survived clearenv")

    receipt = document.get("capability_receipt")
    if not isinstance(receipt, dict) or set(receipt) != {
        "path",
        "sha256",
        "bytes",
        "st_dev",
        "st_ino",
        "mode",
    }:
        raise TargetSealedError("internal receipt binding schema drifted")
    validated = validate_capability_receipt(
        Path(str(receipt["path"])),
        expected_phase=str(document["phase"]),
        expected_outer_fold=document.get("outer_fold"),
        expected_mount_spec_sha256=str(document["mount_specification_sha256"]),
        expected_command_sha256=str(document["command_sha256"]),
    )
    observed_binding = validated["binding"]
    if (
        observed_binding.get("sha256") != receipt.get("sha256")
        or observed_binding.get("bytes") != receipt.get("bytes")
        or observed_binding.get("mode") != receipt.get("mode")
    ):
        raise TargetSealedError("live capability receipt bytes drifted")

    for value in available:
        flags = getattr(os, "O_PATH", os.O_RDONLY)
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(value, flags)
        os.close(descriptor)
    _probe_denied_canaries(canaries)

    mount_table = _mount_table()
    mounted = set(mount_table)
    mount_spec = validated["document"]["mount_specification"]
    expected_mounts = {
        row["destination"]
        for row in mount_spec
        if isinstance(row, dict)
        and row.get("kind")
        in {
            "ro_bind_fd",
            "rw_bind_fd",
            "dev_bind_fd",
            "proc",
            "dev",
            "tmpfs",
            "ro_bind_data",
        }
    }
    if not expected_mounts <= mounted:
        raise TargetSealedError(
            f"live namespace lacks mounts: {sorted(expected_mounts - mounted)}"
        )
    if any(path in mounted for path in canaries.values()):
        raise TargetSealedError("a denied canary is a live mount point")
    if validated["document"]["security_boundary"][
        "production_execution_authorized"
    ] is True:
        _validate_live_gpu_state_mounts(mount_table, validated["document"])
    governance = validated["document"].get("governance_files")
    required_ledger_roles = {"active_authorization", "gpu_state_migration_receipt"}
    present_ledger_roles = (
        required_ledger_roles & set(governance)
        if isinstance(governance, Mapping)
        else set()
    )
    if (
        validated["document"]["security_boundary"][
            "production_execution_authorized"
        ]
        is True
        or present_ledger_roles
    ):
        if not (
            isinstance(governance, Mapping)
            and required_ledger_roles <= set(governance)
        ):
            raise TargetSealedError(
                "internal guard lacks CONTEXT1 live-ledger governance"
            )
        project_root = _canonical_existing(Path.cwd(), label="internal project root")
        guard_bindings: dict[str, FileBinding] = {}
        guard_documents: dict[str, dict[str, Any]] = {}
        for role in sorted(required_ledger_roles):
            row = governance.get(role)
            if not isinstance(row, Mapping) or set(row) != {
                "path",
                "sha256",
                "bytes",
                "st_dev",
                "st_ino",
                "mode",
            }:
                raise TargetSealedError(
                    f"internal guard {role} binding schema drifted"
                )
            path = _canonical_existing(
                Path(str(row["path"])), label=f"internal guard {role}"
            )
            if not _path_within(path, project_root):
                raise TargetSealedError(
                    f"internal guard {role} escapes the project root"
                )
            binding, governance_raw = _read_file_binding(
                path, label=f"internal guard {role}", require_immutable=True
            )
            if binding.document() != dict(row):
                raise TargetSealedError(
                    f"internal guard {role} live binding drifted"
                )
            guard_bindings[role] = binding
            guard_documents[role] = _validate_governance_json(
                governance_raw, role=role
            )
        guard_state = _validate_migrated_state_live(
            project_root=project_root,
            receipt_binding=guard_bindings["gpu_state_migration_receipt"],
        )
        _validate_active_pretrain_live_ledger_prefixes(
            project_root=project_root,
            runtime_ledger_prefixes=guard_documents[
                "active_authorization"
            ].get("runtime_ledger_prefixes"),
            live_state=guard_state,
            require_closed=True,
        )
    _close_guard_runtime_noise_fds()
    audit_process_fds(allowed=(0, 1, 2))
    os.execve(command[0], command, environment)
    raise AssertionError("os.execve returned")


def _revalidate_immutable_capabilities(prepared: PreparedRuntime) -> None:
    for path, binding in prepared.source_snapshot_bindings.items():
        refreshed, _ = _read_file_binding(
            path, label="source snapshot frozen material", require_immutable=True
        )
        if refreshed != binding:
            raise TargetSealedError(
                f"source snapshot capability changed during run: {path}"
            )
    for role, binding in prepared.governance_bindings.items():
        refreshed, _ = _read_file_binding(
            binding.path, label=f"governance:{role}", require_immutable=True
        )
        if refreshed != binding:
            raise TargetSealedError(f"governance capability changed during run: {role}")
    if prepared.pack_binding is not None:
        refreshed_tree = _tree_binding(
            prepared.pack_binding.path, label="sealed-pack shard", frozen=True
        )
        if refreshed_tree != prepared.pack_binding:
            raise TargetSealedError("sealed-pack shard changed during run")
    validated = validate_capability_receipt(
        prepared.request.capability_receipt,
        expected_phase=prepared.request.phase,
        expected_outer_fold=prepared.request.outer_fold,
        expected_mount_spec_sha256=prepared.mount_spec_sha256,
        expected_command_sha256=prepared.command_sha256,
    )
    if canonical_json_bytes(validated["document"]) != canonical_json_bytes(
        prepared.receipt
    ):
        raise TargetSealedError("capability receipt changed during run")


def _pre_popen_revalidate(prepared: PreparedRuntime) -> None:
    _revalidate_immutable_capabilities(prepared)
    state = _validate_migrated_state_live(
        project_root=prepared.request.project_root,
        receipt_binding=prepared.migration_receipt_binding,
    )
    _validate_active_pretrain_live_ledger_prefixes(
        project_root=prepared.request.project_root,
        runtime_ledger_prefixes=BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES,
        live_state=state,
        require_closed=True,
    )
    if state != prepared.prelaunch_state:
        raise TargetSealedError("migrated GPU state changed before Popen")
    if _immutable_output_inventory(
        prepared.writable_bindings["output"].path
    ) != prepared.prelaunch_output_inventory:
        raise TargetSealedError("shard output changed before Popen")
    _revalidate_state_directory_descriptor(
        prepared.gpu_state_root_binding,
        prepared.gpu_state_root_descriptor,
        state.directory_bindings["root"],
        label="gpu_state_parent",
    )
    for role in GPU_STATE_DIRECTORY_ROLES:
        binding = prepared.writable_bindings[role]
        named = os.stat(binding.path, follow_symlinks=False)
        if (
            not stat.S_ISDIR(named.st_mode)
            or _mode(named) != 0o700
            or (named.st_dev, named.st_ino) != (binding.st_dev, binding.st_ino)
        ):
            raise TargetSealedError(f"writable:{role} drifted before Popen")
        _revalidate_state_directory_descriptor(
            binding,
            prepared.gpu_state_child_descriptors[role],
            state.directory_bindings[role],
            label=f"writable:{role}",
        )


def _post_run_revalidate(prepared: PreparedRuntime) -> LiveStateSnapshot:
    _revalidate_immutable_capabilities(prepared)
    state = _validate_migrated_state_live(
        project_root=prepared.request.project_root,
        receipt_binding=prepared.migration_receipt_binding,
    )
    _validate_active_pretrain_live_ledger_prefixes(
        project_root=prepared.request.project_root,
        runtime_ledger_prefixes=BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES,
        live_state=state,
        require_closed=True,
    )
    if state.directory_bindings != prepared.prelaunch_state.directory_bindings:
        raise TargetSealedError("migrated GPU-state directory inode changed during run")
    _revalidate_state_directory_descriptor(
        prepared.gpu_state_root_binding,
        prepared.gpu_state_root_descriptor,
        state.directory_bindings["root"],
        label="gpu_state_parent",
    )
    for role in GPU_STATE_DIRECTORY_ROLES:
        _revalidate_state_directory_descriptor(
            prepared.writable_bindings[role],
            prepared.gpu_state_child_descriptors[role],
            state.directory_bindings[role],
            label=f"writable:{role}",
        )
    for role in ("admission_lock", "execution_ledger_lock", "usage_ledger_lock"):
        if (
            state.current_file_bindings[role]
            != prepared.prelaunch_state.current_file_bindings[role]
        ):
            raise TargetSealedError(f"migrated {role} changed during run")
    return state


def validate_atomic_replace_compatibility(request: RuntimeRequest) -> None:
    """Reject the retired exact-file ABI; V8R4A uses three role directories."""

    if set(request.writable_roots) != WRITABLE_ROLES or any(
        role in request.writable_roots
        for role in ("usage_ledger", "execution_ledger", "admission_lock")
    ):
        raise TargetSealedError(
            "retired exact-file mutable capability ABI is not V8R4A compatible"
        )


COMPLETION_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "phase",
        "outer_fold",
        "capability_receipt",
        "mount_specification_sha256",
        "command_sha256",
        "return_code",
        "closed_replay_validated",
        "output_inventory",
        "output_inventory_sha256",
        "prelaunch_gpu_state",
        "postlaunch_gpu_state",
        "state_transition_sha256",
        "content_sha256",
    }
)


def _completion_receipt_path(request: RuntimeRequest) -> Path:
    capability = _absolute_lexical(
        request.capability_receipt, label="capability receipt"
    )
    return capability.with_name(COMPLETION_RECEIPT_FILENAME)


def _immutable_output_inventory(root: Path) -> list[dict[str, Any]]:
    output = _canonical_existing(root, label="shard output root")
    root_status = os.stat(output, follow_symlinks=False)
    rows: list[dict[str, Any]] = [
        {
            "kind": "directory",
            "path": ".",
            "mode": f"{_mode(root_status):04o}",
            "st_dev": root_status.st_dev,
            "st_ino": root_status.st_ino,
        }
    ]
    seen_files: set[tuple[int, int]] = set()
    seen_directories: set[tuple[int, int]] = {(root_status.st_dev, root_status.st_ino)}
    directory_generations: list[tuple[Path, tuple[int, int, int, int, int]]] = [
        (
            output,
            (
                root_status.st_dev,
                root_status.st_ino,
                root_status.st_mtime_ns,
                root_status.st_ctime_ns,
                root_status.st_nlink,
            ),
        )
    ]
    for current, directories, files in os.walk(output, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            status = os.lstat(path)
            identity = (status.st_dev, status.st_ino)
            if (
                stat.S_ISLNK(status.st_mode)
                or not stat.S_ISDIR(status.st_mode)
                or identity in seen_directories
            ):
                raise TargetSealedError("shard output contains an aliased directory")
            seen_directories.add(identity)
            directory_generations.append(
                (
                    path,
                    (
                        status.st_dev,
                        status.st_ino,
                        status.st_mtime_ns,
                        status.st_ctime_ns,
                        status.st_nlink,
                    ),
                )
            )
            rows.append(
                {
                    "kind": "directory",
                    "path": path.relative_to(output).as_posix(),
                    "mode": f"{_mode(status):04o}",
                    "st_dev": status.st_dev,
                    "st_ino": status.st_ino,
                }
            )
        for name in files:
            path = current_path / name
            lexical = os.lstat(path)
            if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
                raise TargetSealedError(
                    "shard output contains a symlink or non-regular entry"
                )
            binding, _ = _read_file_binding(
                path, label="shard output", require_immutable=True
            )
            identity = (binding.st_dev, binding.st_ino)
            if identity in seen_files:
                raise TargetSealedError("shard output contains an aliased file")
            seen_files.add(identity)
            rows.append(
                {
                    "kind": "file",
                    "path": path.relative_to(output).as_posix(),
                    "mode": "0444",
                    "bytes": binding.bytes,
                    "sha256": binding.sha256,
                    "st_dev": binding.st_dev,
                    "st_ino": binding.st_ino,
                }
            )
    for path, expected in directory_generations:
        refreshed = os.stat(path, follow_symlinks=False)
        if (
            refreshed.st_dev,
            refreshed.st_ino,
            refreshed.st_mtime_ns,
            refreshed.st_ctime_ns,
            refreshed.st_nlink,
        ) != expected:
            raise TargetSealedError("shard output directory changed while snapshotting")
    return sorted(rows, key=lambda row: (str(row["path"]), str(row["kind"])))


def _completion_receipt_document(
    prepared: PreparedRuntime,
    *,
    postlaunch_state: LiveStateSnapshot,
    return_code: int,
) -> dict[str, Any]:
    capability_binding, _ = _read_file_binding(
        prepared.request.capability_receipt,
        label="target-sealed capability receipt",
        require_immutable=True,
    )
    prelaunch = prepared.prelaunch_state.document()
    postlaunch = postlaunch_state.document()
    output_inventory = _immutable_output_inventory(
        prepared.writable_bindings["output"].path
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "classification": COMPLETION_RECEIPT_CLASSIFICATION,
        "campaign_id": CAMPAIGN_ID,
        "campaign_revision": SCIENTIFIC_CAMPAIGN_REVISION,
        "infrastructure_revision": INFRASTRUCTURE_REVISION,
        "phase": prepared.request.phase,
        "outer_fold": prepared.request.outer_fold,
        "capability_receipt": capability_binding.document(),
        "mount_specification_sha256": prepared.mount_spec_sha256,
        "command_sha256": prepared.command_sha256,
        "return_code": return_code,
        "closed_replay_validated": True,
        "output_inventory": output_inventory,
        "output_inventory_sha256": semantic_sha256(output_inventory),
        "prelaunch_gpu_state": prelaunch,
        "postlaunch_gpu_state": postlaunch,
        "state_transition_sha256": semantic_sha256(
            {"prelaunch": prelaunch, "postlaunch": postlaunch}
        ),
    }
    document["content_sha256"] = semantic_sha256(document)
    return document


def validate_completion_receipt(
    path: Path,
    *,
    expected_phase: str | None = None,
    expected_outer_fold: int | None | object = ...,
    expected_command_sha256: str | None = None,
) -> dict[str, Any]:
    binding, raw = _read_file_binding(
        path, label="target-sealed completion receipt", require_immutable=True
    )
    document = _decode_json_bytes(raw, label="target-sealed completion receipt")
    if set(document) != COMPLETION_RECEIPT_KEYS:
        raise TargetSealedError("target-sealed completion receipt schema drifted")
    if raw != canonical_json_bytes(document) + b"\n":
        raise TargetSealedError("target-sealed completion receipt encoding drifted")
    _validate_self_hash(document, label="target-sealed completion receipt")
    prelaunch = _state_snapshot_from_document(
        document.get("prelaunch_gpu_state"), label="completion prelaunch GPU state"
    )
    postlaunch = _state_snapshot_from_document(
        document.get("postlaunch_gpu_state"), label="completion postlaunch GPU state"
    )
    if not (
        document.get("schema_version") == 1
        and document.get("classification") == COMPLETION_RECEIPT_CLASSIFICATION
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == SCIENTIFIC_CAMPAIGN_REVISION
        and document.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and document.get("phase") in PHASES
        and type(document.get("return_code")) is int
        and document.get("closed_replay_validated") is True
        and isinstance(document.get("output_inventory"), list)
        and document.get("output_inventory_sha256")
        == semantic_sha256(document["output_inventory"])
        and _is_sha256(document.get("mount_specification_sha256"))
        and _is_sha256(document.get("command_sha256"))
        and document.get("state_transition_sha256")
        == semantic_sha256(
            {
                "prelaunch": prelaunch.document(),
                "postlaunch": postlaunch.document(),
            }
        )
        and prelaunch.receipt_binding == postlaunch.receipt_binding
        and prelaunch.directory_bindings == postlaunch.directory_bindings
    ):
        raise TargetSealedError("target-sealed completion receipt binding drifted")
    for role in ("admission_lock", "execution_ledger_lock", "usage_ledger_lock"):
        if (
            prelaunch.current_file_bindings[role]
            != postlaunch.current_file_bindings[role]
        ):
            raise TargetSealedError(f"completion {role} binding drifted")
    if expected_phase is not None and document.get("phase") != expected_phase:
        raise TargetSealedError("target-sealed completion phase drifted")
    if (
        expected_outer_fold is not ...
        and document.get("outer_fold") != expected_outer_fold
    ):
        raise TargetSealedError("target-sealed completion outer fold drifted")
    if (
        expected_command_sha256 is not None
        and document.get("command_sha256") != expected_command_sha256
    ):
        raise TargetSealedError("target-sealed completion command drifted")
    return {
        "document": document,
        "binding": binding.document(),
        "prelaunch_state": prelaunch,
        "postlaunch_state": postlaunch,
    }


def _require_closed_append_only_completion_state(
    *,
    project_root: Path,
    recorded: LiveStateSnapshot,
    current: LiveStateSnapshot,
) -> None:
    if (
        current.receipt_binding != recorded.receipt_binding
        or current.directory_bindings != recorded.directory_bindings
    ):
        raise TargetSealedError(
            "completed run migrated directory/receipt lineage drifted"
        )
    for role in ("admission_lock", "execution_ledger_lock", "usage_ledger_lock"):
        if current.current_file_bindings[role] != recorded.current_file_bindings[role]:
            raise TargetSealedError(f"completed run {role} lineage drifted")

    for role in ("usage_ledger", "execution_ledger"):
        prior = recorded.current_file_bindings[role]
        live = current.current_file_bindings[role]
        if prior.get("path") != live.get("path"):
            raise TargetSealedError(f"completed run {role} path drifted")
        path = project_root / str(live["path"])
        binding, raw = _read_file_binding(
            path, label=f"completed run current {role}", require_immutable=False
        )
        if not (
            binding.path == path
            and binding.sha256 == live.get("sha256")
            and binding.bytes == live.get("bytes")
            and binding.mode == 0o644
            and binding.st_dev == live.get("st_dev")
            and binding.st_ino == live.get("st_ino")
            and live.get("nlink") == 1
            and type(prior.get("bytes")) is int
            and prior["bytes"] <= len(raw)
            and hashlib.sha256(raw[: prior["bytes"]]).hexdigest()
            == prior.get("sha256")
        ):
            raise TargetSealedError(
                f"completed run {role} is not an exact live ledger prefix"
            )

    recorded_usage_count = recorded.usage_state.get("record_count")
    current_usage_count = current.usage_state.get("record_count")
    recorded_execution_count = recorded.execution_state.get("record_count")
    current_execution_count = current.execution_state.get("record_count")
    if not (
        type(recorded_usage_count) is int
        and type(current_usage_count) is int
        and current_usage_count >= recorded_usage_count
        and type(recorded_execution_count) is int
        and type(current_execution_count) is int
        and current_execution_count >= recorded_execution_count
        and current.usage_state.get("open_reservation_count") == 0
        and current.execution_state.get("open_start_count") == 0
    ):
        raise TargetSealedError("completed run ledger lifecycle regressed or is open")
    for field, direction in (
        ("settled_usage_ns", "nondecreasing"),
        ("remaining_ns", "nonincreasing"),
    ):
        before = recorded.usage_state.get(field)
        after = current.usage_state.get(field)
        if before is None and after is None:
            continue
        if not (
            type(before) is int
            and type(after) is int
            and (
                after >= before
                if direction == "nondecreasing"
                else after <= before
            )
        ):
            raise TargetSealedError(f"completed run usage {field} regressed")
    if (
        "budget_ns" in recorded.usage_state
        or "budget_ns" in current.usage_state
    ) and recorded.usage_state.get("budget_ns") != current.usage_state.get(
        "budget_ns"
    ):
        raise TargetSealedError("completed run usage budget drifted")


def _try_resume_completed(request: RuntimeRequest) -> int | None:
    completion_path = _completion_receipt_path(request)
    if not os.path.lexists(completion_path):
        return None
    project_root, _interpreter_lexical, _interpreter_real = _validate_request_shape(
        request
    )
    command_sha256 = semantic_sha256(list(request.command))
    completion = validate_completion_receipt(
        completion_path,
        expected_phase=request.phase,
        expected_outer_fold=request.outer_fold,
        expected_command_sha256=command_sha256,
    )
    capability = validate_capability_receipt(
        request.capability_receipt,
        expected_phase=request.phase,
        expected_outer_fold=request.outer_fold,
        expected_mount_spec_sha256=completion["document"][
            "mount_specification_sha256"
        ],
        expected_command_sha256=command_sha256,
    )
    if (
        completion["document"].get("capability_receipt")
        != capability["binding"]
        or capability["document"]["security_boundary"].get(
            "production_execution_authorized"
        )
        is not request.production
    ):
        raise TargetSealedError("completed run capability binding drifted")

    output = _canonical_existing(request.writable_roots["output"], label="output")
    lifecycle = _canonical_existing(
        request.writable_roots["lifecycle"], label="lifecycle"
    )
    if output == lifecycle or completion_path.parent != lifecycle:
        raise TargetSealedError("completed run lifecycle root drifted")
    if set(os.listdir(lifecycle)) != {
        CAPABILITY_RECEIPT_FILENAME,
        COMPLETION_RECEIPT_FILENAME,
    }:
        raise TargetSealedError("completed shard output receipts are absent")
    if _immutable_output_inventory(output) != completion["document"].get(
        "output_inventory"
    ):
        raise TargetSealedError("completed shard output inventory drifted")

    governance_bindings: dict[str, FileBinding] = {}
    governance_documents: dict[str, dict[str, Any]] = {}
    governance_raw: dict[str, bytes] = {}
    recorded_governance = capability["document"].get("governance_files")
    if not isinstance(recorded_governance, dict) or set(recorded_governance) != set(
        request.governance_files
    ):
        raise TargetSealedError("completed run governance role set drifted")
    for role, path in sorted(request.governance_files.items()):
        binding, raw = _read_file_binding(
            path, label=f"governance:{role}", require_immutable=True
        )
        if binding.document() != recorded_governance.get(role):
            raise TargetSealedError(f"completed run governance binding drifted: {role}")
        governance_bindings[role] = binding
        governance_raw[role] = raw
        if not role.startswith("quarantined_material_"):
            governance_documents[role] = _validate_governance_json(raw, role=role)
    _validate_v8r4a_governance_chain(
        project_root=project_root,
        documents=governance_documents,
        bindings=governance_bindings,
        production=request.production,
    )
    snapshot = governance_documents["source_snapshot"]
    source_snapshot_bindings: dict[Path, FileBinding] = {}
    for row in _snapshot_file_rows(snapshot):
        path, expected_sha256, expected_bytes, expected_mode = _validate_snapshot_row(
            row, project_root=project_root
        )
        binding, _ = _read_file_binding(
            path, label="completed source snapshot material", require_immutable=True
        )
        if (
            binding.sha256,
            binding.bytes,
            binding.mode,
        ) != (expected_sha256, expected_bytes, expected_mode):
            raise TargetSealedError("completed source snapshot material drifted")
        prior = source_snapshot_bindings.get(path)
        if prior is not None and prior != binding:
            raise TargetSealedError(
                "completed source snapshot repeats conflicting material"
            )
        source_snapshot_bindings[path] = binding
    if request.phase == "discovery":
        _validate_quarantined_material_cover(
            project_root=project_root,
            governance_documents=governance_documents,
            governance_bindings=governance_bindings,
        )
    if request.phase in {"discovery_aggregation", "promotion_aggregation"}:
        _validate_aggregation_shard_seals(
            governance_documents,
            governance_bindings,
            entry_name=Path(request.command[1]).name,
            project_root=project_root,
        )

    recorded_pack = capability["document"].get("sealed_pack_root")
    if request.pack_root is None:
        if recorded_pack is not None:
            raise TargetSealedError("completed pack-free run gained a pack")
    else:
        pack = _tree_binding(request.pack_root, label="completed sealed pack", frozen=True)
        if pack.document() != recorded_pack:
            raise TargetSealedError("completed sealed pack binding drifted")
        index_binding = governance_bindings["sealed_pack_index"]
        if request.pack_index is None or index_binding.path != _canonical_existing(
            request.pack_index, label="completed sealed-pack index"
        ):
            raise TargetSealedError("completed sealed-pack index path drifted")
        assert request.outer_fold is not None
        _validate_pack_index(
            raw=governance_raw["sealed_pack_index"],
            binding=index_binding,
            pack_root=pack,
            phase=request.phase,
            outer_fold=request.outer_fold,
            promotion_authorization_binding=governance_bindings.get(
                "promotion_authorization"
            ),
            selection_lock_binding=governance_bindings.get("selection_lock"),
        )

    migration_binding = governance_bindings["gpu_state_migration_receipt"]
    _recover_dead_gpu_lifecycle_before_closed_validation(
        project_root=project_root,
        receipt_binding=migration_binding,
        source_snapshot_bindings=source_snapshot_bindings,
    )
    current_state = _validate_migrated_state_live(
        project_root=project_root, receipt_binding=migration_binding
    )
    _require_closed_append_only_completion_state(
        project_root=project_root,
        recorded=completion["postlaunch_state"],
        current=current_state,
    )
    if request.production:
        _close_guard_runtime_noise_fds()
        audit_process_fds(allowed=(0, 1, 2))
    return int(completion["document"]["return_code"])


def run_target_sealed(
    request: RuntimeRequest,
    *,
    launcher_path: Path | None = None,
    version_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> int:
    """Execute bwrap without changing the campaign/watchdog process group."""

    validate_atomic_replace_compatibility(request)
    if request.production:
        _close_guard_runtime_noise_fds()
    resumed = _try_resume_completed(request)
    if resumed is not None:
        return resumed
    with prepare_runtime(
        request, launcher_path=launcher_path, version_runner=version_runner
    ) as prepared:
        child: subprocess.Popen[Any] | None = None
        received_signal: int | None = None
        previous_handlers: dict[int, Any] = {}

        def forward(signum: int, _frame: Any) -> None:
            nonlocal received_signal
            if received_signal is None:
                received_signal = int(signum)
            if child is not None and child.poll() is None:
                try:
                    os.kill(child.pid, int(signum))
                except ProcessLookupError:
                    pass

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[int(signum)] = signal.getsignal(signum)
            signal.signal(signum, forward)
        try:
            _pre_popen_revalidate(prepared)
            # No preexec_fn, setsid, start_new_session, process-group mutation, or
            # launcher timeout is allowed here.  The existing run_gpu_admitted
            # watchdog remains the sole owner of GPU session containment/deadline.
            child = popen_factory(
                prepared.bwrap_command,
                pass_fds=tuple(sorted(prepared.descriptors)),
                close_fds=True,
                env={"LC_ALL": "C", "PATH": "/usr/bin:/usr/sbin"},
                start_new_session=False,
            )
            return_code = int(child.wait())
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        if received_signal is not None and return_code == 0:
            raise TargetSealedError("sandbox returned success after a forwarded signal")
        postlaunch_state = _post_run_revalidate(prepared)
        completion_document = _completion_receipt_document(
            prepared,
            postlaunch_state=postlaunch_state,
            return_code=return_code,
        )
        _create_once_immutable_json(
            _completion_receipt_path(request),
            completion_document,
            label="completion receipt",
        )
        return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-guard", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--phase", choices=sorted(PHASES))
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--pack-root", type=Path)
    parser.add_argument("--pack-index", type=Path)
    parser.add_argument("--governance", action="append", default=[])
    parser.add_argument("--writable-root", action="append", default=[])
    parser.add_argument("--deny-canary", action="append", default=[])
    parser.add_argument("--capability-receipt", type=Path)
    parser.add_argument("--interpreter", type=Path)
    parser.add_argument("--venv-root", type=Path)
    parser.add_argument("--python-runtime-root", type=Path)
    parser.add_argument("--cuda-runtime-root", action="append", type=Path, default=[])
    parser.add_argument("--cuda-device", action="append", type=Path, default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _runtime_request_from_args(args: argparse.Namespace) -> RuntimeRequest:
    required = (
        args.project_root,
        args.phase,
        args.capability_receipt,
        args.interpreter,
        args.venv_root,
        args.python_runtime_root,
    )
    if any(value is None for value in required):
        raise TargetSealedError("outer runtime arguments are incomplete")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    # Path parsing is intentionally not used for environment values: reconstruct
    # the exact original substring after '=' instead of Path normalization.
    environment: dict[str, str] = {}
    for raw in args.env:
        name, separator, value = raw.partition("=")
        if separator != "=" or not name or name in environment:
            raise TargetSealedError(f"invalid or duplicate environment: {raw}")
        environment[name] = value
    return RuntimeRequest(
        project_root=Path(args.project_root),
        phase=str(args.phase),
        outer_fold=args.outer_fold,
        pack_root=args.pack_root,
        pack_index=args.pack_index,
        governance_files=_parse_role_paths(args.governance, label="governance role"),
        writable_roots=_parse_role_paths(
            args.writable_root, label="writable role"
        ),
        denied_canaries=_parse_role_paths(
            args.deny_canary, label="denied-canary role"
        ),
        capability_receipt=Path(args.capability_receipt),
        interpreter=Path(args.interpreter),
        venv_root=Path(args.venv_root),
        python_runtime_root=Path(args.python_runtime_root),
        command=tuple(command),
        cuda_runtime_roots=tuple(args.cuda_runtime_root),
        cuda_devices=tuple(args.cuda_device),
        propagated_environment=environment,
        production=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.internal_guard is not None:
            forbidden_outer = any(
                value
                for value in (
                    args.project_root,
                    args.phase,
                    args.outer_fold,
                    args.pack_root,
                    args.pack_index,
                    args.governance,
                    args.writable_root,
                    args.deny_canary,
                    args.capability_receipt,
                    args.interpreter,
                    args.venv_root,
                    args.python_runtime_root,
                    args.cuda_runtime_root,
                    args.cuda_device,
                    args.env,
                    args.command,
                )
            )
            if forbidden_outer:
                raise TargetSealedError("internal guard cannot accept outer CLI options")
            return internal_guard(args.internal_guard)
        return run_target_sealed(_runtime_request_from_args(args))
    except (TargetSealedError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 73


if __name__ == "__main__":
    raise SystemExit(main())
