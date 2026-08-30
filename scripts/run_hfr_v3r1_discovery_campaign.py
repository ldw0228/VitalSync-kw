#!/usr/bin/env python3
"""Run the fixed 18-unit, target-sealed DHFER-v3r1 discovery matrix.

The orchestrator is intentionally boring.  It has no option for selecting a
fold, seed, variant, release threshold, or an outer-test input.  It validates
the additive v3r1 pretrain authorization, consumes the exact non-test proposer
stacks that were sealed before the v2 target release, and runs the Cartesian
product declared by the v3r1 contract.  Completed units are reused only after
rehashing every required output; incomplete units resume in place with the
same immutable invocation.  All trainer processes pass through the repository
GPU admission wrapper and a hash-chained elapsed-time ledger enforces the
ten-GPU-hour campaign ceiling.

Only outer-validation predictions may contain references.  The validator
checks their cache-index set against ``validation_fold=(outer+1)%6`` and
rejects any outer-test artifact before publishing the create-once completion
seal.  This module is also the small shared provenance library used by the
selection and fixed-promotion drivers.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_rr import gpu_budget_ledger  # noqa: E402


CAMPAIGN_ID = "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
CAMPAIGN_REVISION = "V8R4"
INFRASTRUCTURE_REVISION = "V8R4A"
NONOUTER_PACK_CLASSIFICATION = (
    "adaptive_v3r1_v8r4_nonouter_training_validation_pack"
)
NONOUTER_STACK_CLASSIFICATION = (
    "adaptive_v3r1_v8r4_nonouter_causal_proposer_stack"
)
DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION = {
    "campaign_revision": CAMPAIGN_REVISION,
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
CAMPAIGN_RELATIVE = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1"
)
CONTRACT_RELATIVE = CAMPAIGN_RELATIVE / "ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json"
CONTRACT_FILE_SHA256 = (
    "532d150f0241d9675873368107d09adec7aeaee5e018e09537e8a340eb6fa2bd"
)
SHARD_TRAINING_INDEX = {
    outer_fold: Path(
        "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        f"v8r4_split_inputs/discovery_shard_outer_{outer_fold}/"
        "V8R4_NONOUTER_TRAINING_INDEX.json"
    )
    for outer_fold in (3, 4)
}
SHARD_TRAINING_INDEX_SHA256 = {
    3: "db204cdba72e9f3023c58ef37c1761cfd4ec2f4310449f8eaeeef7003afadb9b",
    4: "1ce1b1a0154c609b1fa1693ff5702b3e81be178f8ddbdd7ec25f559c39029f0a",
}
SHARD_TRAINING_INDEX_BYTES = {3: 3172, 4: 3172}
DEFAULT_RUN_ROOT = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/discovery_v8r4"
)
AGGREGATION_OUTPUT_RELATIVE = DEFAULT_RUN_ROOT / "aggregation_v8r4a"
DISCOVERY_AGGREGATION_PHASE = "discovery_aggregation"
TARGET_SEALED_LIFECYCLE_ROOT_RELATIVE = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "target_sealed_lifecycle_v8r4a_context1"
)
V8R3_QUARANTINE_RELATIVE = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "DISCOVERY_V8R3_ATTEMPT_000_QUARANTINE_OWNER_RECEIPT_V8R4.json"
)
V8R3_QUARANTINE_FILE_SHA256 = (
    "b53d65f7d107d3f82f033a9c8ed4cb835884cead13b3685f284b058b440660b5"
)
V8R3_QUARANTINE_CONTENT_SHA256 = (
    "afc8dd96446888fbc538a0b86623aa49453ffbc9c9605fd6996668becd0ed1d7"
)
V8R3_QUARANTINE_BYTES = 3159
V8R3_QUARANTINED_OUTPUT_SEAL_RELATIVE = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "DISCOVERY_V8R3_ATTEMPT_000_QUARANTINED_OUTPUT_SEAL_V8R4.json"
)
V8R3_QUARANTINED_OUTPUT_SEAL_FILE_SHA256 = (
    "12261aec9e199311dc89a07733fb00c4fa5753ac045298ba49526e84859a7ef3"
)
V8R3_QUARANTINED_OUTPUT_SEAL_CONTENT_SHA256 = (
    "d2fa4e4c22cdcacd0bb84048d03a470d55a3aa40d5b35ad01ab2f7462e2f0817"
)
V8R3_QUARANTINED_OUTPUT_SEAL_BYTES = 5517
DEFAULT_GPU_LOCK = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "gpu_state_v8r4a/admission/gpu_admission_v7.lock"
)
DEFAULT_GPU_LEDGER = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "gpu_state_v8r4a/execution/gpu_execution_ledger_v7.jsonl"
)
DEFAULT_USAGE_LEDGER = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "gpu_state_v8r4a/usage/campaign_gpu_usage_chain_v6.jsonl"
)
GPU_STATE_MIGRATOR_RELATIVE = Path("scripts/migrate_hfr_v3r1_gpu_state_v8r4a.py")
GPU_STATE_MIGRATION_RECEIPT_RELATIVE = CAMPAIGN_RELATIVE / (
    "GPU_STATE_MIGRATION_RECEIPT_V8R4A.json"
)
TARGET_SEALED_RUNTIME_RELATIVE = Path("scripts/run_hfr_v3r1_target_sealed.py")
TARGET_SEALED_CAPABILITY_NAME = "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json"
OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_RELATIVE = CAMPAIGN_RELATIVE / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_OPEN_LIFECYCLE_RECOVERY.json"
)
OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_FILE_SHA256 = (
    "92b7e3e4b911dbf7450e3447b84f5a5762aee1212348c151cd57f20f10f5e1f6"
)
OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_CONTENT_SHA256 = (
    "b415258de20e65cea95f8b303500fd16e8834eff8eb2848a8792fd04c2c90381"
)
OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_BYTES = 8_481
OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC_RELATIVE = CAMPAIGN_RELATIVE / (
    "diagnostics/v3r1_v8r4a_open_gpu_lifecycle_kill_recovery_failure.json"
)
OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC_FILE_SHA256 = (
    "a9cd41355ff98502153d61ad83d7f01da0d3c52462acdd24ade7bef73cf80b5e"
)
OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC_CONTENT_SHA256 = (
    "24d8ed14fc68766a71d09a6fc263cc88aaa7038bb71052f7183328c9a6003016"
)
OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC_BYTES = 4_067
EXECUTION_CLOSURE_AUTHORIZATION_RELATIVE = CAMPAIGN_RELATIVE / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_EXECUTION_CLOSURE.json"
)
EXECUTION_CLOSURE_AUTHORIZATION_FILE_SHA256 = (
    "11e5e7d00d3d837e0eb542b3482499cd21bf90b38883d0f4a0f86b68ba16d754"
)
EXECUTION_CLOSURE_AUTHORIZATION_CONTENT_SHA256 = (
    "92d96a4f513a7d7f93bbd4baf227b626106dab54e000f3a01c97b25504c58c1c"
)
EXECUTION_CLOSURE_AUTHORIZATION_BYTES = 21_621
EXECUTION_CLOSURE_DIAGNOSTIC_RELATIVE = CAMPAIGN_RELATIVE / (
    "diagnostics/v3r1_v8r4a_terminal_execution_closure_failure.json"
)
EXECUTION_CLOSURE_DIAGNOSTIC_FILE_SHA256 = (
    "0ca492c98f94e73e21873c41287c24d3d135466c4ca4d085388f9c39e5d9560e"
)
EXECUTION_CLOSURE_DIAGNOSTIC_CONTENT_SHA256 = (
    "c5dbe569ebcf3b720b06089a6b3c5d8eed4d00a542815edfe25ecb8a6751b774"
)
EXECUTION_CLOSURE_DIAGNOSTIC_BYTES = 8_498
TRAINER_RELATIVE = Path("scripts/train_harmonic_factor_router_snn_v3r1.py")
GPU_WRAPPER_RELATIVE = Path("scripts/run_gpu_admitted.py")
EFFICIENCY_BENCHMARK_RELATIVE = Path("scripts/benchmark_hfr_v3r1_efficiency.py")
EFFICIENCY_BENCHMARK_RECEIPT_RELATIVE = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "efficiency_benchmark_v8r4a_context1/BENCHMARK_COMPLETION_RECEIPT_V8R4.json"
)
HISTORICAL_BENCHMARK_TERMINAL_SHA256S = (
    "d7d3e26a5ba19d648343d19ed4593e503c1017b02f70e8c27517b1d1af0d4fbf",
    "ca89b52ef526fb68854879207c0e56b8dc79a1ee0e31f2b0f143a7a2f43741e2",
    "aaacf4a93f96f5308eb4d31ceceac4de27b39ddd95f9957ee37ab1b42793289b",
)
HISTORICAL_BENCHMARK_USAGE_IDENTITY = {
    "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
    "outer_fold": 3,
    "seed": 20260828,
    "variant": "H0_no_factor",
}
ROOTBIND1_BENCHMARK_USAGE_IDENTITY = {
    "campaign_revision": CAMPAIGN_REVISION,
    "infrastructure_revision": INFRASTRUCTURE_REVISION,
    "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
    "outer_fold": 3,
    "seed": 20260828,
    "variant": "H0_no_factor",
}
ROOTBIND1_BENCHMARK_FAILURE = {
    "terminal_record_sha256": (
        "9da404e120c87940617fed8ad8438d9470b785b304fe48c4ec8727d50fcafafd"
    ),
    "invocation_sha256": (
        "e06bbc723706fd6756f3224dc806ac54cfab6fe8f7852da1f7c372f740730961"
    ),
    "command_sha256": (
        "64ca91360996f9df0df22d7a3b4e063b2d6b362e601689c7a3341ab4dad8cce6"
    ),
    "reservation_record_sha256": (
        "fc872d59d7fcea7fe7cfd4d09e1f5e1931ab93cb4a1ba01e4b2a72007acb7ca9"
    ),
    "return_code": 1,
    "wrapper_exit_code": 1,
    "charged_usage_ns": 2_847_219_074,
    "result_relative_path": (
        "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "efficiency_benchmark_v8r4a_rootbind1/attempts/attempt_000/"
        "GPU_TERMINAL_RESULT.json"
    ),
}
V8R3_QUARANTINE_TERMINAL_SHA256 = (
    "8dbc0493125f22c130444e1344533d1f3d9c4ac445df6adfe8b597972e9691c5"
)

DISCOVERY_SHARD_SEAL_NAME = "DISCOVERY_SHARD_COMPLETION_SEAL.json"
DISCOVERY_SHARD_SEAL_CLASSIFICATION = (
    "adaptive_v3r1_v8r4_discovery_capability_shard_seal"
)
DISCOVERY_SHARD_SEAL_KEYS = frozenset(
    {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "outer_fold_shard",
        "contract",
        "pretrain_authorization",
        "training_index",
        "completed_units",
        "peer_outer_shard_pack_mounted_or_opened",
        "combined_target_bearing_cache_opened",
        "outer_prediction_pack_absent",
        "physical_boundary",
        "gpu_usage_ledger_prefix",
        "pre_discovery_efficiency_benchmark",
        "v8r3_quarantine_owner",
        "units",
        "cross_outer_validation_reuse_present",
        "fully_nested_confirmatory_oof",
        "prospective_confirmation_required",
        "ready_for_pack_free_shard_aggregation",
        "commercial_claim_authorized",
        "content_sha256",
    }
)
DISCOVERY_SHARD_UNIT_KEYS = frozenset(
    {"outer_fold", "seed", "variant", "receipt"}
)

OUTER_RUNS = (3, 4)
SEEDS = (20260828, 20260829, 20260830)
VARIANTS = ("H0_no_factor", "H1_factor", "H2_full")
RELEASE_MODES = ("raw_anchor", "hard_source_argmax", "fixed_confidence_switch")
EXPECTED_DISCOVERY_UNITS = tuple(
    (fold, seed, variant)
    for fold in OUTER_RUNS
    for seed in SEEDS
    for variant in VARIANTS
)
GPU_HOURS_HARD = 10.0
GPU_BUDGET_SECONDS = gpu_budget_ledger.GPU_BUDGET_NS // 1_000_000_000

REQUIRED_TRAIN_OUTPUTS = (
    "run_manifest.json",
    "scaler.json",
    "history.json",
    "last.pt",
    "best.pt",
    "checkpoint_selection_lock.json",
    "validation_predictions.npz",
    "validation_metrics.json",
)
FORBIDDEN_DISCOVERY_OUTPUT_TOKENS = (
    "test_prediction",
    "test_metric",
    "outer_test_prediction",
    "target_access",
)


class CampaignError(RuntimeError):
    """The campaign cannot proceed without violating a locked invariant."""


class BudgetExhausted(CampaignError):
    """No additional GPU work may be opened under the fixed hard budget."""


@dataclass(frozen=True)
class TrainingInput:
    outer_fold: int
    seed: int
    cache_dir: Path
    cache_manifest_sha256: str
    proposer_stack: Path
    proposer_stack_sha256: str
    cache_input_binding: Mapping[str, Any] | None = None
    proposer_stack_binding: Mapping[str, Any] | None = None
    partition_manifest_binding: Mapping[str, Any] | None = None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    document = dict(value)
    document.pop("content_sha256", None)
    return semantic_sha256(document)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CampaignError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def bind_file(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise CampaignError(f"bound regular file is missing or symlinked: {resolved}")
    rendered = (
        resolved.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None and resolved.is_relative_to(relative_to.resolve())
        else str(resolved)
    )
    return {
        "path": rendered,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def projected_file_binding_identity(
    binding: Any, *, project_root: Path
) -> tuple[Path, str, int] | None:
    """Normalize a portable projection without opening its named artifact."""

    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256", "bytes"}:
        return None
    raw_path = Path(str(binding.get("path", "")))
    path = raw_path if raw_path.is_absolute() else project_root / raw_path
    digest = binding.get("sha256")
    size = binding.get("bytes")
    if not (_is_sha256(digest) and type(size) is int and size >= 0):
        return None
    return Path(os.path.abspath(path)), str(digest), int(size)


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CampaignError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CampaignError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"invalid {label}: {path} ({error})") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be a JSON object: {path}")
    return value


def _content_document(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    document = dict(value)
    document.pop("content_sha256", None)
    document["content_sha256"] = semantic_sha256(document)
    raw = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return document, raw


_AT_EMPTY_PATH = 0x1000
_RENAME_NOREPLACE = 1
_PUBLICATION_FAULT_HOOK: Callable[[str, Path], None] | None = None


def _publication_fault(stage: str, path: Path) -> None:
    hook = _PUBLICATION_FAULT_HOOK
    if hook is not None:
        hook(stage, path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_staged_directory_once(staging: Path, final: Path) -> None:
    """Atomically publish one staged directory without replacing a final."""

    if staging.parent != final.parent or staging.name == final.name:
        raise CampaignError("GPU staged-directory rename scope drifted")
    parent_fd = os.open(
        staging.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as error:
            raise CampaignError("renameat2 is unavailable for indexed publication") from error
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(
            parent_fd,
            os.fsencode(staging.name),
            parent_fd,
            os.fsencode(final.name),
            _RENAME_NOREPLACE,
        ) != 0:
            error_number = ctypes.get_errno()
            raise CampaignError(
                f"cannot publish GPU indexed directory without replacement: "
                f"{final} ({os.strerror(error_number)})"
            )
    finally:
        os.close(parent_fd)


def _read_exact_immutable(path: Path, *, expected: bytes, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CampaignError(f"cannot open {label}: {path} ({error})") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if not (
        stat.S_ISREG(after.st_mode)
        and after.st_nlink == 1
        and stat.S_IMODE(after.st_mode) == 0o444
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        and (after.st_dev, after.st_ino) == (named.st_dev, named.st_ino)
        and raw == expected
    ):
        raise CampaignError(f"{label} is not exact immutable bytes: {path}")
    return raw


def _anonymous_create_once(path: Path, raw: bytes) -> None:
    """Link one fully durable anonymous 0444 inode at ``path`` exactly once."""

    path = Path(os.path.abspath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    anonymous_fd = -1
    try:
        try:
            anonymous_fd = os.open(
                path.parent,
                os.O_RDWR
                | getattr(os, "O_TMPFILE", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
        except OSError as error:
            raise CampaignError(
                f"campaign filesystem cannot create an anonymous artifact: {path} ({error})"
            ) from error
        _publication_fault("anonymous_opened", path)
        view = memoryview(raw)
        while view:
            written = os.write(anonymous_fd, view)
            if written <= 0:
                raise CampaignError(f"short anonymous artifact write: {path}")
            view = view[written:]
        os.fchmod(anonymous_fd, 0o444)
        os.fsync(anonymous_fd)
        status = os.fstat(anonymous_fd)
        if not (
            stat.S_ISREG(status.st_mode)
            and status.st_nlink == 0
            and stat.S_IMODE(status.st_mode) == 0o444
            and status.st_size == len(raw)
        ):
            raise CampaignError(f"anonymous artifact pre-link state drifted: {path}")
        _publication_fault("anonymous_fsynced", path)
        libc = ctypes.CDLL(None, use_errno=True)
        linkat = libc.linkat
        linkat.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        linkat.restype = ctypes.c_int
        result = linkat(
            anonymous_fd,
            b"",
            directory_fd,
            os.fsencode(path.name),
            _AT_EMPTY_PATH,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == 17:
                try:
                    _read_exact_immutable(path, expected=raw, label="immutable artifact")
                except CampaignError as error:
                    raise CampaignError(
                        f"immutable artifact collision: {path}"
                    ) from error
                os.fsync(directory_fd)
                return
            raise CampaignError(
                f"cannot link anonymous immutable artifact: {path} "
                f"({os.strerror(error_number)})"
            )
        _publication_fault("linked", path)
        linked = os.fstat(anonymous_fd)
        if linked.st_nlink != 1 or stat.S_IMODE(linked.st_mode) != 0o444:
            raise CampaignError(f"linked artifact inode state drifted: {path}")
        os.fsync(directory_fd)
        _publication_fault("directory_fsynced", path)
    finally:
        if anonymous_fd >= 0:
            os.close(anonymous_fd)
        os.close(directory_fd)
    _read_exact_immutable(path, expected=raw, label="immutable artifact")


def create_once_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Publish complete exact JSON without ever exposing a named partial."""

    document, raw = _content_document(value)
    _anonymous_create_once(path, raw)
    existing = load_json(path, "immutable artifact")
    if existing != document or canonical_content_sha256(existing) != existing.get(
        "content_sha256"
    ):
        raise CampaignError(f"immutable artifact content hash drifted: {path}")
    return document


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    document, raw = _content_document(value)
    del document
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def resolve_binding_path(raw: Any, *, project_root: Path, owner: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = ((project_root / path).resolve(), (owner.parent / path).resolve())
    existing = [candidate for candidate in candidates if candidate.exists()]
    unique = list(dict.fromkeys(existing))
    if len(unique) != 1:
        raise CampaignError(f"bound path does not resolve uniquely: {raw}")
    return unique[0]


def verify_binding(
    binding: Mapping[str, Any], *, project_root: Path, owner: Path, label: str
) -> Path:
    if not isinstance(binding, Mapping) or "path" not in binding:
        raise CampaignError(f"{label} binding is missing")
    path = resolve_binding_path(binding["path"], project_root=project_root, owner=owner)
    expected = binding.get("sha256", binding.get("file_sha256"))
    if not isinstance(expected, str) or len(expected) != 64:
        raise CampaignError(f"{label} binding has no SHA-256")
    if sha256_file(path) != expected:
        raise CampaignError(f"{label} hash drifted: {path}")
    expected_bytes = binding.get("bytes", binding.get("size_bytes"))
    if expected_bytes is not None and int(expected_bytes) != path.stat().st_size:
        raise CampaignError(f"{label} byte size drifted: {path}")
    return path


def validate_contract(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = (project_root / CONTRACT_RELATIVE).resolve()
    if sha256_file(path) != CONTRACT_FILE_SHA256:
        raise CampaignError("v3r1 contract byte hash drifted")
    contract = load_json(path, "v3r1 contract")
    if canonical_content_sha256(contract) != contract.get("content_sha256"):
        raise CampaignError("v3r1 contract canonical content hash drifted")
    if contract.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignError("v3r1 campaign id drifted")
    discovery = contract.get("discovery")
    if not isinstance(discovery, Mapping) or not (
        discovery.get("outer_runs") == list(OUTER_RUNS)
        and discovery.get("seeds") == list(SEEDS)
        and discovery.get("variants") == list(VARIANTS)
        and int(discovery.get("training_job_count", -1)) == 18
        and discovery.get("outer_test_features_or_targets_allowed") is False
    ):
        raise CampaignError("v3r1 discovery matrix or leakage boundary drifted")
    resource = contract.get("resource_budget")
    if not isinstance(resource, Mapping) or not (
        float(resource.get("gpu_hours_hard", -1)) == GPU_HOURS_HARD
        and int(resource.get("maximum_parallel_gpu_training_jobs", -1)) == 1
    ):
        raise CampaignError("v3r1 resource budget drifted")
    return contract, bind_file(path, relative_to=project_root)


def _load_registered_module(name: str, path: Path, *, label: str) -> Any:
    """Load one snapshot-mounted module with ordinary import semantics."""

    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise CampaignError(f"cannot import {label}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException as error:
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
        raise CampaignError(f"cannot load {label}: {error}") from error
    return module


def validate_target_sealed_capability(
    project_root: Path,
    capability_receipt: Path,
    *,
    expected_phase: str,
    expected_outer_fold: int | None,
    runtime_module: Any | None = None,
) -> dict[str, Any]:
    """Replay the explicit one-shot capability before any governance read.

    The receipt is the root of authority inside the sparse bubblewrap child.
    In particular, callers must not discover an authorization, migration
    receipt, peer shard, or pack by walking the project tree.
    """

    root = project_root.expanduser().resolve()
    path = capability_receipt.expanduser()
    if not path.is_absolute():
        path = root / path
    path = Path(os.path.abspath(path))
    if path.name != TARGET_SEALED_CAPABILITY_NAME:
        raise CampaignError("target-sealed capability filename is non-canonical")
    runtime = runtime_module or _load_registered_module(
        "hfr_v8r4a_target_sealed_for_discovery",
        root / TARGET_SEALED_RUNTIME_RELATIVE,
        label="target-sealed runtime capability validator",
    )
    validator = getattr(runtime, "validate_capability_receipt", None)
    if not callable(validator):
        raise CampaignError("target-sealed runtime capability validator is unavailable")
    try:
        result = validator(
            path,
            expected_phase=expected_phase,
            expected_outer_fold=expected_outer_fold,
        )
    except BaseException as error:
        raise CampaignError(f"target-sealed capability rejected: {error}") from error
    if not isinstance(result, Mapping) or set(result) != {"document", "binding"}:
        raise CampaignError("target-sealed validator result schema drifted")
    document = result.get("document")
    binding = result.get("binding")
    boundary = document.get("security_boundary") if isinstance(document, Mapping) else None
    writable = document.get("writable_roots") if isinstance(document, Mapping) else None
    governance = document.get("governance_files") if isinstance(document, Mapping) else None
    lifecycle = writable.get("lifecycle") if isinstance(writable, Mapping) else None
    output = writable.get("output") if isinstance(writable, Mapping) else None
    required_governance = {
        "campaign_contract",
        "active_authorization",
        "source_snapshot",
        "implementation_test_receipt",
        "gpu_state_migration_receipt",
        "open_lifecycle_recovery_correction_authorization",
        "open_lifecycle_recovery_failure_diagnostic",
        "execution_closure_correction_authorization",
        "execution_closure_failure_diagnostic",
    }
    expected_output: Path | None = None
    expected_lifecycle: Path | None = None
    if expected_phase == "discovery" and expected_outer_fold in OUTER_RUNS:
        expected_output = (
            root
            / DEFAULT_RUN_ROOT
            / "shards"
            / f"outer_{int(expected_outer_fold)}"
        ).resolve()
        expected_lifecycle = (
            root
            / TARGET_SEALED_LIFECYCLE_ROOT_RELATIVE
            / "discovery/run_hfr_v3r1_discovery_campaign"
            / f"outer_{int(expected_outer_fold)}"
        ).resolve()
    elif (
        expected_phase == DISCOVERY_AGGREGATION_PHASE
        and expected_outer_fold is None
    ):
        expected_output = (root / AGGREGATION_OUTPUT_RELATIVE).resolve()
        expected_lifecycle = (
            root
            / TARGET_SEALED_LIFECYCLE_ROOT_RELATIVE
            / "discovery_aggregation/run_hfr_v3r1_discovery_campaign/global"
        ).resolve()
    if not (
        isinstance(document, Mapping)
        and isinstance(binding, Mapping)
        and set(binding) == _CAPABILITY_FILE_BINDING_KEYS
        and document.get("classification")
        == "adaptive_v3r1_v8r4a_outer_target_sealed_runtime_capability_receipt"
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == CAMPAIGN_REVISION
        and document.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and document.get("phase") == expected_phase
        and document.get("outer_fold") == expected_outer_fold
        and isinstance(boundary, Mapping)
        and isinstance(governance, Mapping)
        and required_governance <= set(governance)
        and all(isinstance(governance.get(role), Mapping) for role in governance)
        and isinstance(lifecycle, Mapping)
        and isinstance(output, Mapping)
        and Path(str(lifecycle.get("path", ""))).resolve() == path.parent.resolve()
        and (expected_lifecycle is None or path.parent.resolve() == expected_lifecycle)
        and (
            expected_output is None
            or Path(str(output.get("path", ""))).resolve() == expected_output
        )
        and Path(str(output.get("path", ""))).resolve()
        != Path(str(lifecycle.get("path", ""))).resolve()
        and boundary.get("target_reference_or_selection_evidence_accessed") is False
        and boundary.get("legacy_combined_cache_mounted") is False
        and boundary.get("raw_or_target_root_mounted") is False
        and boundary.get("cross_outer_shard_mounted") is False
        and boundary.get("other_pack_or_output_mounted") is False
        and boundary.get("atomic_replace_compatible") is True
        and boundary.get("v8r4a_ledger_migration_required") is False
        and boundary.get("v8r4a_migration_live_replay_validated") is True
        and boundary.get("dedicated_gpu_state_directory_capabilities") is True
        and boundary.get("exactly_three_mutable_state_directory_mounts") is True
        and boundary.get("usage_and_execution_closed_prelaunch") is True
        and boundary.get("lifecycle_mounted_read_only") is True
        and boundary.get("source_snapshot_exact_file_mounts") is True
        and boundary.get("complete_project_source_or_config_trees_mounted") is False
        and type(boundary.get("production_execution_authorized")) is bool
        and type(boundary.get("synthetic_validation_only")) is bool
        and boundary.get("synthetic_validation_only")
        is (not boundary.get("production_execution_authorized"))
    ):
        raise CampaignError("target-sealed capability boundary drifted")
    capability = {"document": dict(document), "binding": dict(binding)}
    live_path, _portable, _normalized, _raw = _live_capability_file_binding(
        root, capability["binding"], label="capability receipt"
    )
    if live_path != path:
        raise CampaignError("target-sealed capability validator rebound its receipt")
    _validate_recovery_governance(root, capability)
    return capability


_CAPABILITY_FILE_BINDING_KEYS = frozenset(
    {"path", "sha256", "bytes", "st_dev", "st_ino", "mode"}
)
_VALIDATOR_FILE_BINDING_KEYS = _CAPABILITY_FILE_BINDING_KEYS | frozenset({"nlink"})
_RECOVERY_GOVERNANCE = {
    "open_lifecycle_recovery_correction_authorization": {
        "path": OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_RELATIVE,
        "sha256": OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_FILE_SHA256,
        "content_sha256": OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_CONTENT_SHA256,
        "bytes": OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_BYTES,
    },
    "open_lifecycle_recovery_failure_diagnostic": {
        "path": OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC_RELATIVE,
        "sha256": OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC_FILE_SHA256,
        "content_sha256": OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC_CONTENT_SHA256,
        "bytes": OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC_BYTES,
    },
    "execution_closure_correction_authorization": {
        "path": EXECUTION_CLOSURE_AUTHORIZATION_RELATIVE,
        "sha256": EXECUTION_CLOSURE_AUTHORIZATION_FILE_SHA256,
        "content_sha256": EXECUTION_CLOSURE_AUTHORIZATION_CONTENT_SHA256,
        "bytes": EXECUTION_CLOSURE_AUTHORIZATION_BYTES,
    },
    "execution_closure_failure_diagnostic": {
        "path": EXECUTION_CLOSURE_DIAGNOSTIC_RELATIVE,
        "sha256": EXECUTION_CLOSURE_DIAGNOSTIC_FILE_SHA256,
        "content_sha256": EXECUTION_CLOSURE_DIAGNOSTIC_CONTENT_SHA256,
        "bytes": EXECUTION_CLOSURE_DIAGNOSTIC_BYTES,
    },
}


def _capability_governance_row(
    capability: Mapping[str, Any], role: str
) -> dict[str, Any]:
    document = capability.get("document")
    governance = document.get("governance_files") if isinstance(document, Mapping) else None
    binding = governance.get(role) if isinstance(governance, Mapping) else None
    if not isinstance(binding, Mapping) or set(binding) != _CAPABILITY_FILE_BINDING_KEYS:
        raise CampaignError(f"target-sealed capability lacks exact {role} binding")
    if not (
        isinstance(binding.get("path"), str)
        and Path(str(binding["path"])).is_absolute()
        and _is_sha256(binding.get("sha256"))
        and type(binding.get("bytes")) is int
        and binding["bytes"] >= 0
        and type(binding.get("st_dev")) is int
        and binding["st_dev"] >= 0
        and type(binding.get("st_ino")) is int
        and binding["st_ino"] > 0
        and binding.get("mode") == "0444"
    ):
        raise CampaignError(f"target-sealed capability {role} binding is malformed")
    return dict(binding)


def _live_capability_file_binding(
    project_root: Path,
    row: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, dict[str, Any], dict[str, Any], bytes]:
    """Revalidate one inode-sealed runtime row and normalize its two ABIs."""

    root = project_root.resolve()
    path = Path(str(row.get("path", "")))
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise CampaignError(f"target-sealed {label} path escapes the project root") from error
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CampaignError(f"target-sealed {label} is unavailable: {error}") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    live = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "st_dev": after.st_dev,
        "st_ino": after.st_ino,
        "mode": f"{stat.S_IMODE(after.st_mode):04o}",
    }
    if not (
        stat.S_ISREG(after.st_mode)
        and after.st_nlink == 1
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        and (after.st_dev, after.st_ino, after.st_mode)
        == (named.st_dev, named.st_ino, named.st_mode)
        and len(raw) == after.st_size
        and live == dict(row)
    ):
        raise CampaignError(f"target-sealed {label} binding changed after issuance")
    portable = {
        "path": relative.as_posix(),
        "sha256": live["sha256"],
        "bytes": live["bytes"],
    }
    validator = {
        **portable,
        "mode": live["mode"],
        "nlink": after.st_nlink,
        "st_dev": live["st_dev"],
        "st_ino": live["st_ino"],
    }
    return path, portable, validator, raw


def _capability_governance_binding(
    project_root: Path,
    capability: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    row = _capability_governance_row(capability, role)
    _path, portable, _validator, _raw = _live_capability_file_binding(
        project_root, row, label=role
    )
    return portable


def _capability_bound_path(
    project_root: Path,
    capability: Mapping[str, Any],
    role: str,
) -> Path:
    row = _capability_governance_row(capability, role)
    path, _portable, _validator, _raw = _live_capability_file_binding(
        project_root, row, label=role
    )
    return path


def _validate_recovery_governance(
    project_root: Path, capability: Mapping[str, Any]
) -> None:
    for role, expected in _RECOVERY_GOVERNANCE.items():
        row = _capability_governance_row(capability, role)
        _path, portable, _validator, raw = _live_capability_file_binding(
            project_root, row, label=role
        )
        try:
            document = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    CampaignError(f"non-finite JSON constant in {role}: {token}")
                ),
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise CampaignError(f"target-sealed {role} JSON is invalid: {error}") from error
        if not (
            portable["path"] == Path(expected["path"]).as_posix()
            and portable["sha256"] == expected["sha256"]
            and portable["bytes"] == expected["bytes"]
            and isinstance(document, Mapping)
            and document.get("content_sha256") == expected["content_sha256"]
        ):
            raise CampaignError(f"target-sealed {role} exact governance drifted")


def validate_pretrain_authorization(
    project_root: Path,
    admitted_binding: Mapping[str, Any] | None = None,
    *,
    capability_receipt: Path | None = None,
    expected_phase: str | None = None,
    expected_outer_fold: int | None = None,
    expected_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke the registered validator rather than duplicating its authority.

    Campaign parents enter through the globally-closed-ledger validator.  A
    wrapper-admitted trainer already owns the sole live reservation, so it must
    enter through the validator's equivalent admitted-child path instead of
    weakening (or accidentally bypassing) the same source-snapshot checks.
    """

    root = project_root.expanduser().resolve()
    capability: dict[str, Any] | None = None
    if capability_receipt is not None:
        if expected_phase is None:
            raise CampaignError("target-scoped pretrain requires an expected phase")
        capability = validate_target_sealed_capability(
            root,
            capability_receipt,
            expected_phase=expected_phase,
            expected_outer_fold=expected_outer_fold,
        )
    elif admitted_binding is None and (
        expected_phase is not None or expected_context is not None
    ):
        raise CampaignError("phase/context scope requires an explicit runtime capability")
    if admitted_binding is not None and (
        expected_phase is None or expected_context is None
    ):
        raise CampaignError(
            "admitted validation requires caller-owned expected phase and context"
        )
    validator_path = root / "scripts/validate_hfr_v3r1_authorization.py"
    module = _load_registered_module(
        "hfr_v3r1_authorization_for_discovery",
        validator_path,
        label="registered v3r1 authorization validator",
    )
    try:
        if capability_receipt is None:
            result = (
                module.validate_pretrain(root)
                if admitted_binding is None
                else module.validate_pretrain_admitted_child(
                    root,
                    admitted_binding,
                    expected_phase=str(expected_phase),
                    expected_context=dict(expected_context),
                )
            )
        elif admitted_binding is None:
            result = module.validate_pretrain_target_scoped(
                root,
                Path(capability_receipt),
                expected_phase=str(expected_phase),
                expected_outer_fold=expected_outer_fold,
            )
        else:
            if expected_context is None:
                raise CampaignError("admitted target scope requires exact caller context")
            result = module.validate_pretrain_target_scoped_admitted_child(
                root,
                Path(capability_receipt),
                admitted_binding,
                expected_phase=str(expected_phase),
                expected_context=dict(expected_context),
            )
    except Exception as error:
        raise CampaignError(f"v3r1 pretrain authorization failed: {error}") from error
    reported_revision = (
        result.get("scientific_campaign_revision")
        if isinstance(result, Mapping) and capability is not None
        else result.get("campaign_revision") if isinstance(result, Mapping) else None
    )
    if not isinstance(result, Mapping) or not (
        result.get("valid") is True
        and result.get("training_authorized") is True
        and result.get("promotion_authorized") is False
        and result.get("commercial_claim_authorized") is False
        and reported_revision == CAMPAIGN_REVISION
        and result.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and result.get("contract_file_sha256") == CONTRACT_FILE_SHA256
    ):
        raise CampaignError("v3r1 pretrain validator returned an unsafe result")
    if capability is not None:
        authorization_binding = _capability_governance_binding(
            root, capability, "active_authorization"
        )
        contract_binding = _capability_governance_binding(
            root, capability, "campaign_contract"
        )
        normalized_governance: dict[str, dict[str, Any]] = {}
        governance = capability["document"].get("governance_files")
        if not isinstance(governance, Mapping):
            raise CampaignError("target capability governance bindings are absent")
        for role in governance:
            row = _capability_governance_row(capability, str(role))
            _path, _portable, normalized, _raw = _live_capability_file_binding(
                root, row, label=str(role)
            )
            normalized_governance[str(role)] = normalized
        _capability_path, _capability_portable, normalized_capability, _raw = (
            _live_capability_file_binding(
                root, capability["binding"], label="capability receipt"
            )
        )
        required_result_bindings = {
            "authorization_binding": normalized_governance["active_authorization"],
            "contract_binding": normalized_governance["campaign_contract"],
            "source_snapshot_binding": normalized_governance["source_snapshot"],
            "implementation_test_receipt_binding": normalized_governance[
                "implementation_test_receipt"
            ],
        }
        if not (
            all(
                isinstance(result.get(name), Mapping)
                and canonical_json_bytes(result[name]) == canonical_json_bytes(expected)
                for name, expected in required_result_bindings.items()
            )
            and isinstance(result.get("capability_binding"), Mapping)
            and canonical_json_bytes(result["capability_binding"])
            == canonical_json_bytes(normalized_capability)
            and isinstance(result.get("capability_document"), Mapping)
            and canonical_json_bytes(result["capability_document"])
            == canonical_json_bytes(capability["document"])
            and isinstance(result.get("governance_bindings"), Mapping)
            and canonical_json_bytes(result["governance_bindings"])
            == canonical_json_bytes(normalized_governance)
        ):
            raise CampaignError(
                "target-scoped pretrain validator capability binding drifted"
            )
    else:
        active_relative = getattr(module, "PRETRAIN_AUTHORIZATION", None)
        if not isinstance(active_relative, Path):
            raise CampaignError(
                "registered validator does not export its active pretrain authorization path"
            )
        authorization_path = (
            active_relative.resolve()
            if active_relative.is_absolute()
            else (root / active_relative).resolve()
        )
        authorization_binding = bind_file(authorization_path, relative_to=root)
        contract_binding = bind_file(root / CONTRACT_RELATIVE, relative_to=root)
    if authorization_binding["sha256"] != result.get(
        "pretrain_authorization_file_sha256"
    ):
        raise CampaignError(
            "registered active pretrain authorization path/hash disagree"
        )
    reported_path = result.get("pretrain_authorization_path")
    if reported_path is not None and reported_path != authorization_binding["path"]:
        raise CampaignError(
            "registered active pretrain authorization result/path disagree"
        )
    return {
        **dict(result),
        "authorization_binding": authorization_binding,
        "contract_binding": contract_binding,
        "target_sealed_capability": capability,
    }


def validate_v8r4a_gpu_state(
    project_root: Path,
    *,
    migration_receipt: Path | None = None,
    migration_module: Any | None = None,
) -> dict[str, Any]:
    """Live-replay the authorized V8R4A state tree and exact path capabilities."""

    root = project_root.expanduser().resolve()
    receipt_path = (
        (root / GPU_STATE_MIGRATION_RECEIPT_RELATIVE).resolve()
        if migration_receipt is None
        else migration_receipt.expanduser().resolve()
    )
    canonical_receipt = (root / GPU_STATE_MIGRATION_RECEIPT_RELATIVE).resolve()
    if receipt_path != canonical_receipt:
        raise CampaignError("V8R4A migration receipt path is non-canonical")
    module = migration_module
    if module is None:
        module_path = (root / GPU_STATE_MIGRATOR_RELATIVE).resolve()
        specification = importlib.util.spec_from_file_location(
            "hfr_v3r1_v8r4a_gpu_state_for_discovery", module_path
        )
        if specification is None or specification.loader is None:
            raise CampaignError("cannot import the V8R4A GPU-state validator")
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        try:
            specification.loader.exec_module(module)
        except BaseException as error:
            raise CampaignError(
                f"cannot load the V8R4A GPU-state validator: {error}"
            ) from error
        finally:
            sys.modules.pop(specification.name, None)
    validator = getattr(module, "validate_migrated_state", None)
    if not callable(validator):
        raise CampaignError("V8R4A GPU-state validator entry point is missing")
    try:
        result = validator(root, receipt_path, require_closed=True)
    except BaseException as error:
        raise CampaignError(f"V8R4A GPU-state migration replay failed: {error}") from error
    canonical_paths = getattr(result, "canonical_paths", None)
    receipt = getattr(result, "receipt", None)
    receipt_binding = getattr(result, "receipt_binding", None)
    expected_paths = {
        "admission_lock": (root / DEFAULT_GPU_LOCK).resolve(),
        "execution_ledger": (root / DEFAULT_GPU_LEDGER).resolve(),
        "usage_ledger": (root / DEFAULT_USAGE_LEDGER).resolve(),
    }
    if not (
        isinstance(canonical_paths, Mapping)
        and all(canonical_paths.get(name) == path for name, path in expected_paths.items())
        and isinstance(receipt, Mapping)
        and receipt.get("classification")
        == "adaptive_v3r1_v8r4a_gpu_state_migration_receipt"
        and receipt.get("campaign_id") == CAMPAIGN_ID
        and receipt.get("scientific_campaign_revision") == CAMPAIGN_REVISION
        and receipt.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and isinstance(receipt_binding, Mapping)
    ):
        raise CampaignError("V8R4A GPU-state validator returned unsafe capabilities")
    return {
        "migration_receipt": dict(receipt_binding),
        "canonical_paths": expected_paths,
    }


def _call_trainer_input_verifier(
    project_root: Path, entry_name: str, *args: Any, **kwargs: Any
) -> Any:
    trainer_path = (project_root / TRAINER_RELATIVE).resolve()
    specification = importlib.util.spec_from_file_location(
        "hfr_v3r1_trainer_cache_verifier", trainer_path
    )
    if specification is None or specification.loader is None:
        raise CampaignError("cannot import the registered trainer cache verifier")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    try:
        specification.loader.exec_module(module)
        verifier = getattr(module, entry_name, None)
        if not callable(verifier):
            raise CampaignError(f"registered trainer lacks {entry_name}")
        return verifier(*args, **kwargs)
    except CampaignError:
        raise
    except BaseException as error:
        raise CampaignError(f"training input verification failed: {error}") from error
    finally:
        sys.modules.pop(specification.name, None)


def verify_training_cache_inputs(
    project_root: Path, cache_dir: Path, *, outer_fold: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the trainer's single canonical cache-inode verifier.

    Keeping the verifier in the trainer makes campaign, benchmark, signature,
    and the actual mmap/reference consumer agree on one exact schema and one
    no-follow/single-link/same-inode implementation.
    """

    result = _call_trainer_input_verifier(
        project_root,
        "verify_cache_manifest_outputs",
        cache_dir,
        outer_fold=int(outer_fold),
    )
    try:
        manifest, binding = result
    except (TypeError, ValueError) as error:
        raise CampaignError("trainer cache verifier returned an invalid result") from error
    if not isinstance(manifest, dict) or not isinstance(binding, dict):
        raise CampaignError("trainer cache verifier returned an invalid result")
    return manifest, binding


def verify_training_bound_file(
    project_root: Path,
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    result = _call_trainer_input_verifier(
        project_root,
        "verify_bound_regular_file",
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    if not isinstance(result, dict):
        raise CampaignError("trainer bound-file verifier returned an invalid result")
    return result


def load_efficiency_benchmark_module(project_root: Path) -> Any:
    """Load the frozen V8 benchmark without creating a circular module import."""

    path = (project_root / EFFICIENCY_BENCHMARK_RELATIVE).resolve()
    specification = importlib.util.spec_from_file_location(
        "hfr_v3r1_efficiency_benchmark_for_discovery", path
    )
    if specification is None or specification.loader is None:
        raise CampaignError("cannot import the registered V8 efficiency benchmark")
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise CampaignError(f"cannot load the V8 efficiency benchmark: {error}") from error
    return module


def completion_receipt_command_resolver(
    receipt: Mapping[str, Any],
    *,
    expected_phase: str,
    expected_identity: Mapping[str, Any],
) -> Callable[[Mapping[str, Any]], str]:
    """Reconstruct exact workload hashes from a receipt's bound invocations."""

    bindings = receipt.get("lifecycle_invocations")
    if not isinstance(bindings, list) or not bindings:
        raise CampaignError("completion receipt lacks lifecycle invocation bindings")
    commands: dict[str, str] = {}
    for binding in bindings:
        invocation_path = _bound_absolute_file(binding, "lifecycle invocation")
        invocation = load_json(invocation_path, "GPU lifecycle invocation")
        workload = invocation.get("workload_command")
        context = invocation.get("context")
        if not (
            canonical_content_sha256(invocation) == invocation.get("content_sha256")
            and invocation.get("campaign_id") == CAMPAIGN_ID
            and invocation.get("phase") == expected_phase
            and isinstance(context, Mapping)
            and all(context.get(key) == value for key, value in expected_identity.items())
            and isinstance(workload, list)
            and all(isinstance(part, str) for part in workload)
        ):
            raise CampaignError("completion lifecycle invocation scope drifted")
        command_sha256 = semantic_sha256(workload)
        if invocation.get("workload_command_sha256") != command_sha256:
            raise CampaignError("completion lifecycle invocation command drifted")
        invocation_sha256 = sha256_file(invocation_path)
        if invocation_sha256 in commands:
            raise CampaignError("completion lifecycle invocation is duplicated")
        commands[invocation_sha256] = command_sha256

    def resolve(record: Mapping[str, Any]) -> str:
        invocation_sha256 = str(record.get("invocation_sha256", ""))
        try:
            return commands[invocation_sha256]
        except KeyError as error:
            raise CampaignError(
                "GPU ledger names an invocation outside its completion receipt"
            ) from error

    return resolve


def validate_pre_discovery_efficiency_benchmark(
    *,
    project_root: Path,
    receipt_path: Path,
    usage_ledger: Path,
    gpu_ledger: Path,
    gpu_lock: Path,
    authorization: Mapping[str, Any],
    usage_state: Any | None = None,
    require_no_discovery_terminal: bool = True,
) -> dict[str, Any]:
    """Prove the mandatory quarantined throughput gate on the active ledgers."""

    if usage_state is None:
        try:
            with gpu_budget_ledger.locked_closed_snapshot(
                usage_ledger,
                budget_ns=gpu_budget_ledger.GPU_BUDGET_NS,
                expected_legacy_genesis_sha256=_expected_legacy_genesis(
                    usage_ledger
                ),
            ) as locked_state:
                return validate_pre_discovery_efficiency_benchmark(
                    project_root=project_root,
                    receipt_path=receipt_path,
                    usage_ledger=usage_ledger,
                    gpu_ledger=gpu_ledger,
                    gpu_lock=gpu_lock,
                    authorization=authorization,
                    usage_state=locked_state,
                    require_no_discovery_terminal=require_no_discovery_terminal,
                )
        except (OSError, ValueError, RuntimeError) as error:
            if isinstance(error, CampaignError):
                raise
            raise CampaignError(
                f"cannot validate the benchmark on a stable closed GPU snapshot: {error}"
            ) from error
    if not (
        authorization.get("efficiency_benchmark_authorized") is True
        and authorization.get("discovery_requires_passing_efficiency_benchmark") is True
    ):
        raise CampaignError("active authorization does not require the V8 benchmark gate")
    receipt_path = receipt_path.expanduser().resolve()
    benchmark = load_efficiency_benchmark_module(project_root)
    try:
        validated = benchmark.validate_benchmark_receipt_pack_free(
            project_root=project_root,
            receipt_path=receipt_path,
            expected_pretrain_authorization=authorization.get(
                "authorization_binding"
            ),
        )
    except Exception as error:
        if isinstance(error, CampaignError):
            raise
        raise CampaignError(f"V8 efficiency benchmark gate failed: {error}") from error
    positions = {
        str(record["record_sha256"]): index
        for index, record in enumerate(usage_state.records)
    }
    historical_order = [
        str(item["terminal_result"]["terminal_record_sha256"])
        for item in benchmark.HISTORICAL_BENCHMARK_ATTEMPTS
    ]
    active_hashes = [
        str(value) for value in validated.get("usage_record_sha256s", [])
    ]
    if len(active_hashes) != 1 or active_hashes[0] not in positions:
        raise CampaignError("active benchmark terminal ownership is not exact")
    quarantine_hash = "8dbc0493125f22c130444e1344533d1f3d9c4ac445df6adfe8b597972e9691c5"
    rootbind_hash = str(ROOTBIND1_BENCHMARK_FAILURE["terminal_record_sha256"])
    if any(
        value not in positions
        for value in [*historical_order, quarantine_hash, rootbind_hash]
    ):
        raise CampaignError("historical benchmark/quarantine ledger prefix is incomplete")
    rootbind_record = usage_state.records[positions[rootbind_hash]]
    validate_rootbind1_failed_benchmark_terminal(
        rootbind_record, project_root=project_root
    )
    active_record = usage_state.records[positions[active_hashes[0]]]
    if not (
        active_record.get("phase") == benchmark.BENCHMARK_PHASE
        and _execution_record_context(active_record)
        == dict(benchmark.BENCHMARK_USAGE_IDENTITY)
        and _usage_record_succeeded(active_record)
        and active_hashes[0] == validated.get("usage_record_sha256")
        and validated.get("gpu_execution_ledger_path") == str(gpu_ledger.resolve())
        and validated.get("gpu_admission_lock_path") == str(gpu_lock.resolve())
    ):
        raise CampaignError("active benchmark ledger terminal drifted")
    historical_positions = [positions[value] for value in historical_order]
    quarantine_position = positions[quarantine_hash]
    rootbind_position = positions[rootbind_hash]
    active_position = positions[active_hashes[0]]
    if not (
        historical_positions == sorted(historical_positions)
        and max(historical_positions)
        < quarantine_position
        < rootbind_position
        < active_position
    ):
        raise CampaignError("benchmark/quarantine ledger ordering drifted")
    observed_historical: list[str] = []
    observed_rootbind: list[str] = []
    observed_active: list[str] = []
    for record in usage_state.records:
        if not _is_execution_terminal(record) or record.get("phase") != benchmark.BENCHMARK_PHASE:
            continue
        record_hash = str(record.get("record_sha256", ""))
        context = _execution_record_context(record)
        if record_hash in historical_order:
            if context != dict(benchmark.LEGACY_BENCHMARK_USAGE_IDENTITY):
                raise CampaignError("historical benchmark terminal identity drifted")
            observed_historical.append(record_hash)
        elif record_hash == rootbind_hash:
            validate_rootbind1_failed_benchmark_terminal(
                record, project_root=project_root
            )
            observed_rootbind.append(record_hash)
        elif record_hash == active_hashes[0]:
            if context != dict(benchmark.BENCHMARK_USAGE_IDENTITY):
                raise CampaignError("active benchmark terminal identity drifted")
            observed_active.append(record_hash)
        else:
            raise CampaignError("unowned efficiency-benchmark terminal in ledger")
    if (
        observed_historical != historical_order
        or observed_rootbind != [rootbind_hash]
        or observed_active != active_hashes
    ):
        raise CampaignError("benchmark terminal whitelist order or cover drifted")
    active_discovery_positions: list[int] = []
    for number, record in enumerate(usage_state.records):
        if not _is_execution_terminal(record) or record.get("phase") != "discovery":
            continue
        record_hash = str(record.get("record_sha256", ""))
        if record_hash == quarantine_hash and _record_matches_exact_v8r3_quarantine(
            record
        ):
            continue
        context = _execution_record_context(record)
        if not (
            context.get("campaign_revision") == CAMPAIGN_REVISION
            and context.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        ):
            raise CampaignError("unowned discovery terminal precedes the active campaign")
        active_discovery_positions.append(number)
    if require_no_discovery_terminal and active_discovery_positions:
        raise CampaignError("active discovery terminal exists before benchmark admission")
    if active_discovery_positions and active_position >= min(active_discovery_positions):
        raise CampaignError("active benchmark did not precede V8R4 discovery work")
    return validated


def _binding_from_unit(unit: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    artifacts = unit.get("artifacts")
    if isinstance(artifacts, Mapping):
        aliases = {
            "cache_manifest": ("cache_manifest",),
            "proposer_stack": ("strict_stack", "proposer_stack"),
        }
        for alias in aliases[name]:
            value = artifacts.get(alias)
            if isinstance(value, Mapping):
                return value
    value = unit.get(name)
    if isinstance(value, Mapping):
        return value
    if name == "cache_manifest":
        cache = unit.get("cache_root", unit.get("cache"))
        if isinstance(cache, Mapping):
            raw_path = cache.get("path")
            expected = cache.get("manifest_sha256", cache.get("sha256"))
            if raw_path is not None and expected is not None:
                return {
                    "path": str(Path(str(raw_path)) / "manifest.json"),
                    "sha256": expected,
                }
    raise CampaignError(f"training unit lacks {name} binding")


def _validate_stack_scope(path: Path, outer_fold: int, seed: int) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "classification",
                "campaign_revision",
                "partition",
                "cache_index",
                "fold",
                "prediction",
                "rr_std",
                "proposal_available",
                "nested_role",
                "outer_fold",
                "seed",
                "outer_test_opened",
                "outer_rows_present",
            }
            if set(archive.files) != required:
                raise CampaignError("V8R4 nonouter proposer stack schema drifted")
            if str(np.asarray(archive["classification"]).item()) != NONOUTER_STACK_CLASSIFICATION:
                raise CampaignError("V8R4 proposer stack classification drifted")
            if str(np.asarray(archive["campaign_revision"]).item()) != CAMPAIGN_REVISION:
                raise CampaignError("V8R4 proposer stack revision drifted")
            if str(np.asarray(archive["partition"]).item()) != "outer_excluded_training_validation":
                raise CampaignError("V8R4 proposer stack partition drifted")
            if bool(np.asarray(archive["outer_test_opened"]).item()):
                raise CampaignError("training proposer stack has opened outer test")
            if bool(np.asarray(archive["outer_rows_present"]).item()):
                raise CampaignError("training proposer stack physically contains outer rows")
            if int(np.asarray(archive["outer_fold"]).item()) != outer_fold:
                raise CampaignError("training proposer stack outer-fold mismatch")
            if int(np.asarray(archive["seed"]).item()) != seed:
                raise CampaignError("training proposer stack seed mismatch")
            index = np.asarray(archive["cache_index"], dtype=np.int64)
            fold = np.asarray(archive["fold"], dtype=np.int16)
            available = np.asarray(archive["proposal_available"], dtype=bool)
            raw_role = np.asarray(archive["nested_role"])
            prediction = np.asarray(archive["prediction"], dtype=np.float32)
            rr_std = np.asarray(archive["rr_std"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as error:
        raise CampaignError(f"invalid non-test proposer stack: {path} ({error})") from error
    if (
        index.ndim != 1
        or len(index) == 0
        or len(np.unique(index)) != len(index)
        or fold.shape != index.shape
        or available.shape != index.shape
        or raw_role.shape != index.shape
        or raw_role.dtype.kind not in "US"
        or prediction.shape != index.shape
        or rr_std.shape != index.shape
        or np.any(np.diff(index) <= 0)
    ):
        raise CampaignError("training proposer stack row topology is invalid")
    role = raw_role.astype(str)
    outer = fold == outer_fold
    validation = fold == ((outer_fold + 1) % 6)
    train = ~(outer | validation)
    if outer.any() or not validation.any() or not train.any():
        raise CampaignError("training proposer stack is not a physical nonouter cover")
    if set(map(int, fold)) != (set(range(6)) - {int(outer_fold)}):
        raise CampaignError("training proposer stack omits a required nonouter fold")
    if not available.all():
        raise CampaignError("train/validation proposer exact cover is incomplete")
    if any(
        "test" in item.lower() or "outer" in item.lower()
        for item in set(role)
    ):
        raise CampaignError("outer-test role leaked into train/validation proposer rows")
    valid = (
        np.isfinite(prediction)
        & np.isfinite(rr_std)
        & (prediction >= 6.0)
        & (prediction <= 45.0)
        & (rr_std > 0.0)
    )
    if np.any(~valid):
        raise CampaignError("V8R4 nonouter proposer values are invalid")
    return index


def _validate_v8r4_partition_manifest(
    path: Path,
    *,
    project_root: Path,
    outer_fold: int,
    seed: int,
    cache_manifest: Path,
    proposer_stack: Path,
) -> dict[str, Any]:
    document = load_json(path, "V8R4 sealed nonouter partition")
    expected_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
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
        "content_sha256",
    }
    if not (
        set(document) == expected_keys
        and canonical_content_sha256(document) == document.get("content_sha256")
        and document.get("schema_version") == 1
        and document.get("classification")
        == "adaptive_v3r1_v8r4_sealed_nonouter_partition"
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == CAMPAIGN_REVISION
        and document.get("outer_fold") == outer_fold
        and document.get("seed") == seed
        and type(document.get("legacy_row_count")) is int
        and int(document["legacy_row_count"]) > 0
    ):
        raise CampaignError("V8R4 sealed partition identity/schema drifted")
    partition = document.get("partition")
    partition_keys = {
        "discovery_rows",
        "discovery_outer_rows",
        "legacy_outer_complement_rows",
        "outer_prediction_pack_rows",
        "intersection_rows",
        "union_rows",
        "exact_disjoint_complement",
        "global_index_sha256",
        "discovery_global_index_sha256",
        "outer_global_index_sha256",
        "fold_topology_sha256",
    }
    if not (
        isinstance(partition, Mapping)
        and set(partition) == partition_keys
        and type(partition.get("discovery_rows")) is int
        and int(partition["discovery_rows"]) > 0
        and partition.get("discovery_outer_rows") == 0
        and type(partition.get("legacy_outer_complement_rows")) is int
        and int(partition["legacy_outer_complement_rows"]) > 0
        and partition.get("outer_prediction_pack_rows") == 0
        and partition.get("intersection_rows") == 0
        and partition.get("union_rows") == document["legacy_row_count"]
        and int(partition["discovery_rows"])
        + int(partition["legacy_outer_complement_rows"])
        == int(document["legacy_row_count"])
        and partition.get("exact_disjoint_complement") is True
        and all(
            _is_sha256(partition.get(name))
            for name in (
                "global_index_sha256",
                "discovery_global_index_sha256",
                "outer_global_index_sha256",
                "fold_topology_sha256",
            )
        )
    ):
        raise CampaignError("V8R4 sealed partition exact-cover proof drifted")
    if document.get("protected_outer_access") != {
        "topology_columns_decoded_first": ["cache_index", "fold"],
        "target_free_columns_decoded_after_partition": [],
        "emitted_fields": [],
        "exact_allowlist": True,
        "forbidden_fields_emitted": False,
        "outer_reference_decoded": False,
        "outer_reference_validity_decoded": False,
        "outer_identity_decoded": False,
        "outer_protocol_decoded": False,
        "outer_quality_decoded": False,
        "whole_legacy_metadata_hashed_as_opaque_binding": True,
    }:
        raise CampaignError("V8R4 protected-outer access proof drifted")
    if document.get("preselection_prediction_boundary") != {
        "outer_prediction_pack_absent": True,
        "outer_prediction_path_bound": False,
        "outer_prediction_values_materialized": False,
        "promotion_authorization_required_before_prediction_pack": True,
    }:
        raise CampaignError("V8R4 preselection prediction boundary drifted")
    if document.get("claim_boundary") != {
        "adaptive_retrospective_only": True,
        "commercial_or_confirmatory_claim_allowed": False,
        "outer_targets_opened": False,
    }:
        raise CampaignError("V8R4 partition claim boundary drifted")
    interface = document.get("integration_interface")
    if not (
        isinstance(interface, Mapping)
        and interface.get("training_cache") == "discovery_cache"
        and interface.get("training_proposer_stack")
        == "discovery_proposer_stack.npz"
        and interface.get("trainer_outer_fold") == outer_fold
        and interface.get("trainer_seed") == seed
        and interface.get("cache_index_translation")
        == "discovery_cache/local_to_global_cache_index.npy"
        and interface.get("science_row_order_preserved") is True
        and interface.get("model_feature_values_preserved") is True
        and interface.get("discovery_reference_and_context_values_preserved") is True
    ):
        raise CampaignError("V8R4 partition integration interface drifted")
    serialization = document.get("serialization")
    if not (
        isinstance(serialization, Mapping)
        and serialization.get("object_arrays") is False
        and serialization.get("pickle") is False
        and serialization.get("outputs_mode") == "0444"
        and serialization.get("create_once_resume_requires_byte_equality") is True
    ):
        raise CampaignError("V8R4 partition serialization proof drifted")
    outputs = document.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "discovery_cache_manifest",
        "discovery_proposer_stack",
        "discovery_local_to_global_map",
    }:
        raise CampaignError("V8R4 partition output schema drifted")
    cache_path = verify_binding(
        outputs["discovery_cache_manifest"],
        project_root=project_root,
        owner=path,
        label="partition-bound discovery cache manifest",
    )
    stack_path = verify_binding(
        outputs["discovery_proposer_stack"],
        project_root=project_root,
        owner=path,
        label="partition-bound discovery proposer stack",
    )
    map_path = verify_binding(
        outputs["discovery_local_to_global_map"],
        project_root=project_root,
        owner=path,
        label="partition-bound local-to-global map",
    )
    if not (
        cache_path == cache_manifest
        and stack_path == proposer_stack
        and map_path == cache_manifest.parent / "local_to_global_cache_index.npy"
    ):
        raise CampaignError("V8R4 partition outputs differ from shard index")
    # Legacy target-bearing bindings remain opaque provenance only.  The
    # consumer must never resolve or open them.
    legacy = document.get("legacy_inputs")
    if not isinstance(legacy, Mapping) or set(legacy) != {
        "training_index",
        "cache_manifest",
        "proposer_stack",
        "cache_outputs",
    }:
        raise CampaignError("V8R4 opaque legacy binding schema drifted")
    return document


def load_training_index(
    project_root: Path, index_path: Path, *, outer_fold_shard: int
) -> tuple[dict[tuple[int, int], TrainingInput], dict[str, Any]]:
    if outer_fold_shard not in OUTER_RUNS:
        raise CampaignError("V8R4 discovery requires one outer-fold capability shard")
    path = index_path.expanduser().resolve()
    canonical_path = (project_root / SHARD_TRAINING_INDEX[outer_fold_shard]).resolve()
    if path != canonical_path:
        raise CampaignError("training index must be the canonical immutable trust anchor")
    expected_index_sha256 = SHARD_TRAINING_INDEX_SHA256[outer_fold_shard]
    expected_index_bytes = SHARD_TRAINING_INDEX_BYTES[outer_fold_shard]
    if not _is_sha256(expected_index_sha256) or expected_index_bytes <= 0:
        raise CampaignError(
            "V8R4 training-index constants have not been integrated from the sealed split campaign"
        )
    try:
        index_stat = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise CampaignError("canonical training index is unavailable") from error
    if not (
        not path.is_symlink()
        and index_stat.st_nlink == 1
        and index_stat.st_mode & 0o777 == 0o444
    ):
        raise CampaignError("canonical training index mode/link invariant drifted")
    verified_index = verify_training_bound_file(
        project_root,
        path,
        expected_sha256=expected_index_sha256,
        expected_bytes=expected_index_bytes,
    )
    if Path(str(verified_index.get("path", ""))) != path:
        raise CampaignError("canonical training index path/inode binding drifted")
    document = load_json(path, "fixed non-test training index")
    if document.get("content_sha256") is not None and (
        canonical_content_sha256(document) != document.get("content_sha256")
    ):
        raise CampaignError("fixed non-test training index content hash drifted")
    expected_index_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "outer_fold",
        "seeds",
        "unit_count",
        "completed_units",
        "status",
        "outer_test_opened",
        "combined_target_bearing_cache_consumer_access_authorized",
        "physical_nonouter_training_packs",
        "outer_prediction_packs_absent",
        "cross_outer_shard_mounted",
        "units",
        "content_sha256",
    }
    if not (
        set(document) == expected_index_keys
        and document.get("schema_version") == 1
        and document.get("classification")
        == "adaptive_v3r1_v8r4_nonouter_training_index"
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == CAMPAIGN_REVISION
        and document.get("outer_fold") == outer_fold_shard
        and document.get("seeds") == list(SEEDS)
        and document.get("unit_count") == 3
        and document.get("outer_test_opened") is False
        and document.get("combined_target_bearing_cache_consumer_access_authorized") is False
        and document.get("physical_nonouter_training_packs") is True
        and document.get("outer_prediction_packs_absent") is True
        and document.get("cross_outer_shard_mounted") is False
        and int(document.get("completed_units", -1)) == 3
        and document.get("status") == "complete"
    ):
        raise CampaignError("fixed V8R4 split training index is incomplete or unsafe")
    units = document.get("units")
    if not isinstance(units, list) or len(units) != 3:
        raise CampaignError("fixed discovery shard index must contain three seed packs")
    expected = {(outer_fold_shard, seed) for seed in SEEDS}
    result: dict[tuple[int, int], TrainingInput] = {}
    for unit in units:
        if not isinstance(unit, Mapping) or set(unit) != {
            "outer_fold",
            "seed",
            "relative_path",
            "artifacts",
        }:
            raise CampaignError("fixed non-test training unit is not an object")
        if not isinstance(unit.get("relative_path"), str):
            raise CampaignError("training unit relative path is invalid")
        artifacts = unit.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "cache_manifest",
            "proposer_stack",
            "partition_manifest",
        }:
            raise CampaignError("training unit artifact exact schema drifted")
        key = (int(unit.get("outer_fold", -1)), int(unit.get("seed", -1)))
        if key not in expected or unit.get("relative_path") != (
            f"units/outer_{key[0]}_seed_{key[1]}"
        ):
            raise CampaignError("training unit escaped its active capability shard")
        if key in result:
            raise CampaignError(f"duplicate fixed non-test training unit: {key}")
        cache_binding = _binding_from_unit(unit, "cache_manifest")
        stack_binding = _binding_from_unit(unit, "proposer_stack")
        partition_binding = artifacts["partition_manifest"]
        cache_manifest = verify_binding(
            cache_binding,
            project_root=project_root,
            owner=path,
            label=f"cache manifest {key}",
        )
        stack = verify_binding(
            stack_binding,
            project_root=project_root,
            owner=path,
            label=f"proposer stack {key}",
        )
        partition_manifest = verify_binding(
            partition_binding,
            project_root=project_root,
            owner=path,
            label=f"partition manifest {key}",
        )
        _validate_v8r4_partition_manifest(
            partition_manifest,
            project_root=project_root,
            outer_fold=key[0],
            seed=key[1],
            cache_manifest=cache_manifest,
            proposer_stack=stack,
        )
        stack_sha256 = stack_binding.get(
            "sha256", stack_binding.get("file_sha256")
        )
        stack_bytes = stack_binding.get("bytes", stack_binding.get("size_bytes"))
        if not isinstance(stack_sha256, str) or type(stack_bytes) is not int:
            raise CampaignError(f"proposer stack exact binding is incomplete: {key}")
        proposer_stack_binding = verify_training_bound_file(
            project_root,
            stack,
            expected_sha256=stack_sha256,
            expected_bytes=stack_bytes,
        )
        if Path(str(proposer_stack_binding.get("path", ""))) != stack:
            raise CampaignError(f"proposer stack canonical path drifted: {key}")
        cache_dir = cache_manifest.parent
        _, cache_input_binding = verify_training_cache_inputs(
            project_root, cache_dir, outer_fold=key[0]
        )
        verified_manifest = cache_input_binding.get("manifest")
        if not isinstance(verified_manifest, Mapping) or not (
            Path(str(verified_manifest.get("path", ""))) == cache_manifest
            and verified_manifest.get("sha256")
            == cache_binding.get("sha256", cache_binding.get("file_sha256"))
            and verified_manifest.get("bytes") == cache_manifest.stat().st_size
        ):
            raise CampaignError(
                f"training cache manifest/index canonical binding failed: {key}"
            )
        required_cache = (
            "feature_names.json",
            "metadata.csv",
            "node_features.npy",
            "candidate_bpm.npy",
            "candidate_mask.npy",
            "joint_radar_mask.npy",
            "local_to_global_cache_index.npy",
        )
        missing = [name for name in required_cache if not (cache_dir / name).is_file()]
        if missing:
            raise CampaignError(f"training cache files missing for {key}: {missing}")
        stack_index = _validate_stack_scope(stack, *key)
        try:
            metadata_index = np.genfromtxt(
                cache_dir / "metadata.csv",
                delimiter=",",
                names=True,
                usecols=(0,),
                dtype=np.int64,
                encoding="utf-8",
            )
            if metadata_index.dtype.names:
                metadata_index = np.asarray(metadata_index[metadata_index.dtype.names[0]])
            metadata_index = np.atleast_1d(metadata_index).astype(np.int64)
        except (OSError, ValueError) as error:
            raise CampaignError(f"cannot read training cache index for {key}: {error}") from error
        if not np.array_equal(metadata_index, stack_index):
            raise CampaignError(f"training cache/proposer exact index binding failed: {key}")
        result[key] = TrainingInput(
            outer_fold=key[0],
            seed=key[1],
            cache_dir=cache_dir,
            cache_manifest_sha256=sha256_file(cache_manifest),
            proposer_stack=stack,
            proposer_stack_sha256=sha256_file(stack),
            cache_input_binding=cache_input_binding,
            proposer_stack_binding=proposer_stack_binding,
            partition_manifest_binding=bind_file(partition_manifest),
        )
    if set(result) != expected:
        raise CampaignError("V8R4 shard index is not an exact one-outer x three-seed cover")
    return result, bind_file(path, relative_to=project_root)


def _read_cache_index_and_fold(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read only the first two non-target bookkeeping columns."""

    try:
        values = np.genfromtxt(
            cache_dir / "metadata.csv",
            delimiter=",",
            names=True,
            usecols=(0, 1),
            dtype=None,
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        raise CampaignError(f"cannot read cache index/fold: {cache_dir} ({error})") from error
    names = values.dtype.names or ()
    if names != ("cache_index", "fold"):
        raise CampaignError("cache metadata first columns must be cache_index,fold")
    return (
        np.atleast_1d(values["cache_index"]).astype(np.int64),
        np.atleast_1d(values["fold"]).astype(np.int16),
    )


def _artifact_hash_from_lock(
    lock: Mapping[str, Any], filename: str, aliases: Sequence[str]
) -> str | None:
    artifacts = lock.get("artifacts")
    if isinstance(artifacts, Mapping):
        value = artifacts.get(filename)
        if isinstance(value, Mapping):
            candidate = value.get("sha256", value.get("file_sha256"))
            if isinstance(candidate, str):
                return candidate
        if isinstance(value, str):
            return value
    for alias in aliases:
        candidate = lock.get(alias)
        if isinstance(candidate, str):
            return candidate
    return None


def validation_metrics_by_release_mode(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return the three fixed release metrics with a narrow ABI tolerance."""

    for container_name in ("release_modes", "metrics_by_release_mode", "validation"):
        container = document.get(container_name)
        if isinstance(container, Mapping) and all(
            isinstance(container.get(mode), Mapping) for mode in RELEASE_MODES
        ):
            result: dict[str, Mapping[str, Any]] = {}
            for mode in RELEASE_MODES:
                value = container[mode]  # type: ignore[index]
                nested = value.get("metrics") if isinstance(value, Mapping) else None
                result[mode] = nested if isinstance(nested, Mapping) else value
            return result
    if all(isinstance(document.get(mode), Mapping) for mode in RELEASE_MODES):
        result = {}
        for mode in RELEASE_MODES:
            value = document[mode]  # type: ignore[index]
            nested = value.get("metrics") if isinstance(value, Mapping) else None
            result[mode] = nested if isinstance(nested, Mapping) else value
        return result
    raise CampaignError("validation metrics lack the three fixed release modes")


def normalize_accuracy_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    aliases = {
        "overall_mae_bpm": ("overall_mae_bpm", "mae"),
        "identity_macro_mae_bpm": ("identity_macro_mae_bpm", "identity_macro_mae"),
        "rmse_bpm": ("rmse_bpm", "rmse"),
        "within_2_fraction": ("within_2_fraction", "within_2"),
        "over_5_fraction": ("over_5_fraction", "catastrophic_over_5"),
        "high_rr_25_35_mae_bpm": ("high_rr_25_35_mae_bpm", "tail_25_35_mae"),
    }
    result: dict[str, float] = {}
    for canonical, names in aliases.items():
        raw = next((metrics[name] for name in names if name in metrics), None)
        if raw is None:
            raise CampaignError(f"validation metric is missing: {canonical}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise CampaignError(f"validation metric is not numeric: {canonical}") from error
        if not math.isfinite(value):
            raise CampaignError(f"validation metric is not finite: {canonical}")
        result[canonical] = value
    return result


def validate_training_output(
    output_dir: Path,
    *,
    outer_fold: int,
    seed: int,
    variant: str,
    cache_dir: Path,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_TRAIN_OUTPUTS if not (output_dir / name).is_file()]
    if missing:
        raise CampaignError(f"training outputs are incomplete: {output_dir} ({missing})")
    forbidden = [
        path.name
        for path in output_dir.rglob("*")
        if path.is_file()
        and any(token in path.name.lower() for token in FORBIDDEN_DISCOVERY_OUTPUT_TOKENS)
    ]
    if forbidden:
        raise CampaignError(f"outer-test/target artifact appeared in discovery: {forbidden}")
    manifest = load_json(output_dir / "run_manifest.json", "v3r1 run manifest")
    validation_fold = (outer_fold + 1) % 6
    identity_checks = {
        "campaign_revision": CAMPAIGN_REVISION,
        "outer_fold": outer_fold,
        "validation_fold": validation_fold,
        "seed": seed,
        "variant": variant,
    }
    effective = manifest.get("effective_configuration")
    identity_source = effective if isinstance(effective, Mapping) else manifest
    for key, expected in identity_checks.items():
        if identity_source.get(key) != expected:
            raise CampaignError(f"training run manifest identity drifted: {key}")
    leakage = manifest.get("leakage_boundary")
    if leakage != DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION:
        raise CampaignError(
            "training run manifest lacks the exact V8R4 physical boundary declaration"
        )
    parameter_count = manifest.get("parameter_count")
    if parameter_count is None and isinstance(manifest.get("model"), Mapping):
        parameter_count = manifest["model"].get("parameter_count")
    if parameter_count is None and isinstance(manifest.get("model_config"), Mapping):
        parameter_count = manifest["model_config"].get("parameter_count")
    if parameter_count is None:
        try:
            checkpoint = torch.load(
                output_dir / "best.pt", map_location="cpu", weights_only=False
            )
            state = checkpoint.get("model_state") if isinstance(checkpoint, Mapping) else None
            if isinstance(state, Mapping):
                parameter_count = sum(
                    int(value.numel()) for value in state.values() if hasattr(value, "numel")
                )
        except (OSError, RuntimeError, ValueError):
            parameter_count = None
    if parameter_count is None or not (0 < int(parameter_count) <= 400_000):
        raise CampaignError("training run manifest has invalid parameter count")
    lock = load_json(
        output_dir / "checkpoint_selection_lock.json", "checkpoint selection lock"
    )
    for key, expected in identity_checks.items():
        observed = lock.get(key, manifest.get(key))
        if observed != expected:
            raise CampaignError(f"checkpoint selection lock identity drifted: {key}")
    if lock.get("leakage_boundary") != DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION:
        raise CampaignError("checkpoint lock lacks the V8R4 boundary declaration")
    access_audit = lock.get("row_access_audit")
    expected_audit_keys = {
        "campaign_revision",
        "outer_fold",
        "physical_pack_rows",
        "outer_rows_in_physical_pack",
        "outer_row_access_attempts",
        "implicit_whole_array_conversions",
        "accesses_by_array",
        "selected_rows_by_array",
        "unique_accessed_cache_indexes",
        "accessed_cache_indexes_sha256",
    }
    if not (
        isinstance(access_audit, Mapping)
        and set(access_audit) == expected_audit_keys
        and access_audit.get("campaign_revision") == CAMPAIGN_REVISION
        and access_audit.get("outer_fold") == outer_fold
        and type(access_audit.get("physical_pack_rows")) is int
        and int(access_audit["physical_pack_rows"]) > 0
        and access_audit.get("outer_rows_in_physical_pack") == 0
        and access_audit.get("outer_row_access_attempts") == 0
        and access_audit.get("implicit_whole_array_conversions") == 0
        and isinstance(access_audit.get("accesses_by_array"), Mapping)
        and isinstance(access_audit.get("selected_rows_by_array"), Mapping)
        and _is_sha256(access_audit.get("accessed_cache_indexes_sha256"))
    ):
        raise CampaignError("checkpoint lock row-access audit is unsafe or malformed")
    aliases = {
        "best.pt": ("best_sha256", "best_checkpoint_sha256", "checkpoint_sha256"),
        "scaler.json": ("scaler_sha256",),
        "history.json": ("history_sha256",),
        "run_manifest.json": ("run_manifest_sha256", "manifest_sha256"),
    }
    for filename, names in aliases.items():
        expected = _artifact_hash_from_lock(lock, filename, names)
        if expected is None or expected != sha256_file(output_dir / filename):
            raise CampaignError(f"checkpoint lock does not bind {filename}")
    scientific_signature = manifest.get("scientific_signature")
    scientific_signature_sha = manifest.get("scientific_signature_sha256")
    forbidden_signature_fields = {
        "output_directory",
        "campaign_phase_label",
        "promotion_authorization_path",
        "release_mode",
        "resume_flag",
    }
    if not isinstance(scientific_signature, Mapping):
        raise CampaignError("training run manifest lacks a scientific signature")
    if forbidden_signature_fields.intersection(map(str, scientific_signature)):
        raise CampaignError("scientific signature retained an orchestration field")
    if not (
        isinstance(scientific_signature_sha, str)
        and len(scientific_signature_sha) == 64
        and semantic_sha256(dict(scientific_signature)) == scientific_signature_sha
        and lock.get("scientific_signature_sha256") == scientific_signature_sha
    ):
        raise CampaignError("training scientific signature binding drifted")
    metrics_document = load_json(
        output_dir / "validation_metrics.json", "validation metrics"
    )
    if not (
        metrics_document.get("campaign_revision") == CAMPAIGN_REVISION
        and metrics_document.get("classification")
        == "adaptive_v3r1_v8r4_discovery_validation_only"
        and metrics_document.get("outer_test_rows_present") is False
    ):
        raise CampaignError("validation metrics predate the V8R4 boundary")
    metrics = validation_metrics_by_release_mode(metrics_document)
    normalized = {
        mode: normalize_accuracy_metrics(metrics[mode]) for mode in RELEASE_MODES
    }
    expected_index, fold = _read_cache_index_and_fold(cache_dir)
    expected_validation_index = expected_index[fold == validation_fold]
    if np.any(fold == outer_fold):
        raise CampaignError("training cache physically contains its outer fold")
    try:
        with np.load(output_dir / "validation_predictions.npz", allow_pickle=False) as archive:
            required = {
            "cache_index",
            "reference_rr_bpm",
            "reference_valid",
            "identity",
            "raw_anchor_bpm",
            "raw_anchor_available",
            "hard_source_bpm",
            "hard_source_available",
            "fixed_confidence_switch_bpm",
            "fixed_confidence_switch_available",
            "selected_source_probability",
            "selected_source_code",
            "source_scale_bpm",
            "quality",
            "factor_probabilities",
            "spike_rate",
            }
            missing_predictions = sorted(required - set(archive.files))
            if missing_predictions:
                raise CampaignError(
                    f"validation prediction fields missing: {missing_predictions}"
                )
            observed_index = np.asarray(archive["cache_index"], dtype=np.int64)
            reference = np.asarray(archive["reference_rr_bpm"])
            valid = np.asarray(archive["reference_valid"], dtype=bool)
            identity = np.asarray(archive["identity"])
            factor = np.asarray(archive["factor_probabilities"])
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, CampaignError):
            raise
        raise CampaignError(
            "validation prediction NPZ is not pickle-free replayable"
        ) from error
    if not np.array_equal(observed_index, expected_validation_index):
        raise CampaignError("validation predictions are not the exact validation-fold cover")
    if reference.shape != observed_index.shape or valid.shape != observed_index.shape:
        raise CampaignError("validation reference topology is invalid")
    if (
        identity.shape != observed_index.shape
        or identity.dtype.kind not in "US"
        or factor.shape != (len(observed_index), 4)
    ):
        raise CampaignError("validation identity/factor topology is invalid")
    artifacts = {
        name: bind_file(output_dir / name) for name in REQUIRED_TRAIN_OUTPUTS
    }
    return {
        "campaign_revision": CAMPAIGN_REVISION,
        "outer_fold": outer_fold,
        "validation_fold": validation_fold,
        "seed": seed,
        "variant": variant,
        "parameter_count": int(parameter_count),
        "validation_rows": int(len(observed_index)),
        "valid_reference_rows": int(valid.sum()),
        "release_metrics": normalized,
        "scientific_signature_sha256": scientific_signature_sha,
        "physical_boundary": dict(DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
        "row_access_audit": dict(access_audit),
        "artifacts": artifacts,
    }


def _expected_legacy_genesis(path: Path) -> str | None:
    """Pin the inherited V6 ledger while permitting isolated test ledgers."""

    if path.expanduser().resolve().name == DEFAULT_USAGE_LEDGER.name:
        return gpu_budget_ledger.LEGACY_V1_GENESIS_RECORD_SHA256
    return None


def _verify_usage_state(path: Path) -> Any:
    try:
        return gpu_budget_ledger.verify_ledger(
            path,
            budget_ns=gpu_budget_ledger.GPU_BUDGET_NS,
            expected_legacy_genesis_sha256=_expected_legacy_genesis(path),
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise CampaignError(f"invalid GPU usage ledger: {error}") from error


def _require_closed_usage_state(path: Path) -> Any:
    try:
        return gpu_budget_ledger.require_closed_ledger(
            path,
            budget_ns=gpu_budget_ledger.GPU_BUDGET_NS,
            expected_legacy_genesis_sha256=_expected_legacy_genesis(path),
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise CampaignError(f"GPU usage ledger is not completely settled: {error}") from error


def _verify_usage_chain(path: Path) -> tuple[list[dict[str, Any]], float]:
    """Compatibility view over the authoritative mixed V1/V2 reducer."""

    state = _verify_usage_state(path)
    return list(state.records), float(state.settled_usage_ns) / 1_000_000_000.0


def usage_snapshot_binding(path: Path, state: Any) -> dict[str, Any]:
    """Bind the exact immutable ledger prefix represented by a locked state."""

    raw = bytes(state.raw_bytes)
    return {
        "path": str(path.expanduser().resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "records": len(state.records),
        "terminal_record_sha256": state.tail_sha256,
        "settled_usage_ns": int(state.settled_usage_ns),
        "elapsed_seconds": float(state.settled_usage_ns) / 1_000_000_000.0,
        "open_reservations": 0,
    }


def append_usage_record(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Append a legacy V1 fixture record; production execution never calls this.

    V7 GPU lifecycle records are written solely by ``run_gpu_admitted.py`` under
    the supervisor's ledger lock.  This narrowly retained helper keeps the
    historical V1-prefix and mixed-chain tests readable.
    """

    records, _ = _verify_usage_chain(path)
    previous = records[-1]["record_sha256"] if records else None
    document = {**dict(value), "previous_record_sha256": previous}
    document["record_sha256"] = semantic_sha256(document)
    raw = canonical_json_bytes(document) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return document


EXECUTION_USAGE_PHASES = frozenset(
    {
        "efficiency_benchmark",
        "discovery",
        "promotion_training",
        "promotion_prediction",
    }
)


def _usage_ledger_path(path: Path) -> str:
    """Return the canonical ledger identity embedded in every completion receipt."""

    return str(path.expanduser().resolve())


def bind_run_usage_ledger(
    run_root: Path, usage_ledger: Path, *, execution_scope: str
) -> dict[str, Any]:
    """Create the run-level ledger identity before any output can be reused."""

    if execution_scope not in {"discovery", "promotion"}:
        raise CampaignError("invalid GPU usage ledger execution scope")
    return create_once_json(
        run_root / "GPU_USAGE_LEDGER_IDENTITY.json",
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_gpu_usage_ledger_identity",
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "execution_scope": execution_scope,
            "usage_ledger_path": _usage_ledger_path(usage_ledger),
            "path_change_or_budget_reset_allowed": False,
        },
    )


def _usage_record_succeeded(record: Mapping[str, Any]) -> bool:
    if record.get("schema_version") == 2:
        return record.get("event") == "terminal" and record.get(
            "reuse_eligible"
        ) is True
    return (
        type(record.get("return_code")) is int
        and int(record["return_code"]) == 0
        and record.get("hard_timeout_reached") is False
    )


def _execution_record_context(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if record.get("schema_version") == 2:
        context = record.get("context")
        if not isinstance(context, Mapping):
            raise CampaignError("V2 GPU terminal record lacks canonical context")
        return context
    return record


def _is_execution_terminal(record: Mapping[str, Any]) -> bool:
    if record.get("schema_version") == 2:
        return record.get("event") in {"terminal", "reconciled_terminal"}
    return record.get("phase") in EXECUTION_USAGE_PHASES


def _record_matches_execution(
    record: Mapping[str, Any],
    *,
    expected_phase: str,
    expected_identity: Mapping[str, Any],
) -> bool:
    if record.get("phase") != expected_phase or not _is_execution_terminal(record):
        return False
    if (
        expected_phase == "discovery"
        and expected_identity.get("campaign_revision") != CAMPAIGN_REVISION
    ):
        return False
    context = _execution_record_context(record)
    observed_identity = {
        str(name): value
        for name, value in context.items()
        if str(name) not in {"execution_number", "attempt_number", "resume"}
    }
    return observed_identity == dict(expected_identity)


def _record_matches_exact_v8r3_quarantine(record: Mapping[str, Any]) -> bool:
    """Match only the one pre-correction context shape, with no null wildcard."""

    if record.get("phase") != "discovery" or not _is_execution_terminal(record):
        return False
    context = _execution_record_context(record)
    return dict(context) == {
        "execution_number": 0,
        "outer_fold": 3,
        "resume": False,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }


def validate_rootbind1_failed_benchmark_terminal(
    record: Mapping[str, Any], *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Own the one charged ROOTBIND1 infrastructure failure, exactly."""

    expected = ROOTBIND1_BENCHMARK_FAILURE
    expected_result = str(
        (project_root.expanduser().resolve() / expected["result_relative_path"]).resolve()
    )
    if not (
        record.get("schema_version") == 2
        and record.get("event") == "terminal"
        and record.get("campaign_id") == CAMPAIGN_ID
        and record.get("phase") == "efficiency_benchmark"
        and dict(_execution_record_context(record))
        == ROOTBIND1_BENCHMARK_USAGE_IDENTITY
        and record.get("record_sha256") == expected["terminal_record_sha256"]
        and record.get("invocation_sha256") == expected["invocation_sha256"]
        and record.get("command_sha256") == expected["command_sha256"]
        and record.get("reservation_record_sha256")
        == expected["reservation_record_sha256"]
        and record.get("return_code") == expected["return_code"]
        and record.get("wrapper_exit_code") == expected["wrapper_exit_code"]
        and record.get("charged_usage_ns") == expected["charged_usage_ns"]
        and record.get("elapsed_ns") == expected["charged_usage_ns"]
        and record.get("result_path") == expected_result
        and record.get("reuse_eligible") is False
        and record.get("containment_anomaly") is False
        and record.get("hard_timeout_reached") is False
        and record.get("reservation_deadline_breached") is False
        and record.get("termination_escalated") is False
    ):
        raise CampaignError("ROOTBIND1 benchmark failure terminal drifted")
    return dict(record)


def _validate_execution_usage_record(
    record: Mapping[str, Any],
    *,
    expected_phase: str,
    expected_identity: Mapping[str, Any],
) -> None:
    if record.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignError("GPU usage record campaign identity mismatched")
    if record.get("phase") != expected_phase:
        raise CampaignError("GPU usage record phase mismatched")
    context = _execution_record_context(record)
    for name, expected in expected_identity.items():
        if context.get(name) != expected:
            raise CampaignError(f"GPU usage record unit identity mismatched: {name}")
    if record.get("schema_version") == 2:
        if record.get("event") not in {"terminal", "reconciled_terminal"}:
            raise CampaignError("GPU usage lifecycle is not terminal")
        charged = record.get("charged_usage_ns")
        if type(charged) is not int or int(charged) < 0:
            raise CampaignError("GPU terminal record charged usage is invalid")
        invocation_sha256 = record.get("invocation_sha256")
        if not _is_sha256(invocation_sha256):
            raise CampaignError("GPU terminal record invocation hash is invalid")
    else:
        elapsed = record.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
            raise CampaignError("GPU usage record elapsed time is invalid")
        if not math.isfinite(float(elapsed)) or float(elapsed) < 0.0:
            raise CampaignError("GPU usage record elapsed time is invalid")
    if record.get("schema_version") == 2 and record.get("event") == "reconciled_terminal":
        if record.get("return_code") is not None:
            raise CampaignError("reconciled GPU terminal cannot claim a return code")
    elif type(record.get("return_code")) is not int:
        raise CampaignError("GPU execution usage record lacks an observed return code")
    if record.get("schema_version") == 2 and record.get("event") == "reconciled_terminal":
        if record.get("reuse_eligible") is not False:
            raise CampaignError("reconciled GPU terminal cannot be reusable")
    elif type(record.get("hard_timeout_reached")) is not bool:
        raise CampaignError("GPU execution usage record has invalid timeout status")
    command_sha256 = record.get("command_sha256")
    if not _is_sha256(command_sha256):
        raise CampaignError("GPU execution usage record has invalid command hash")


def _validate_non_execution_usage_record(record: Mapping[str, Any]) -> None:
    """Allow only the explicitly governed forced-termination carry-forward event."""

    if not (
        record.get("campaign_id") == CAMPAIGN_ID
        and record.get("phase") == "quarantine_carry_forward"
        and record.get("event") == "forced_termination_usage_carry_forward"
        and record.get("quarantined") is True
        and record.get("training_result_eligible_for_reuse") is False
        and record.get("return_code") is None
        and record.get("return_code_observed") is False
        and record.get("termination_signal") == "SIGTERM"
        and record.get("hard_timeout_reached") is False
    ):
        raise CampaignError("unrecognized non-execution GPU usage record")
    elapsed = record.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
        raise CampaignError("carry-forward GPU elapsed time is invalid")
    if not math.isfinite(float(elapsed)) or float(elapsed) <= 0.0:
        raise CampaignError("carry-forward GPU elapsed time is invalid")


def _bound_absolute_file(binding: Mapping[str, Any], label: str) -> Path:
    if not isinstance(binding, Mapping):
        raise CampaignError(f"{label} binding is missing")
    path = Path(str(binding.get("path", ""))).expanduser()
    if not path.is_absolute():
        raise CampaignError(f"{label} binding must use its canonical absolute path")
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise CampaignError(f"{label} bound file is missing or symlinked")
    if binding.get("sha256") != sha256_file(path):
        raise CampaignError(f"{label} hash drifted")
    if binding.get("bytes") != path.stat().st_size:
        raise CampaignError(f"{label} byte size drifted")
    return path


def _validate_terminal_result_bindings(
    usage_ledger: Path,
    receipt: Mapping[str, Any],
    matching: Sequence[Mapping[str, Any]],
    *,
    usage_state: Any,
) -> None:
    snapshot_hashes = {str(record["record_sha256"]) for record in usage_state.records}
    if any(str(record.get("record_sha256")) not in snapshot_hashes for record in matching):
        raise CampaignError("completion terminal record is outside locked ledger state")
    v2_terminals = {
        str(record["record_sha256"]): record
        for record in matching
        if record.get("schema_version") == 2 and record.get("event") == "terminal"
    }
    declared = receipt.get("terminal_results")
    if not v2_terminals:
        if declared not in (None, []):
            raise CampaignError("legacy completion receipt unexpectedly binds terminal results")
        return
    if not isinstance(declared, list) or len(declared) != len(v2_terminals):
        raise CampaignError("completion receipt does not exactly cover V2 terminal results")
    seen: set[str] = set()
    for item in declared:
        if not isinstance(item, Mapping):
            raise CampaignError("completion terminal-result entry is invalid")
        terminal_sha256 = item.get("terminal_record_sha256")
        if terminal_sha256 not in v2_terminals or terminal_sha256 in seen:
            raise CampaignError("completion terminal-result cover is duplicated or foreign")
        record = v2_terminals[str(terminal_sha256)]
        invocation_path = _bound_absolute_file(
            item.get("execution_invocation", {}), "execution invocation"
        )
        result_path = _bound_absolute_file(item.get("result", {}), "terminal result")
        invocation = load_json(invocation_path, "GPU execution invocation")
        if canonical_content_sha256(invocation) != invocation.get("content_sha256"):
            raise CampaignError("GPU execution invocation content drifted")
        context = invocation.get("context")
        workload_command = invocation.get("workload_command")
        if not isinstance(context, Mapping) or not isinstance(workload_command, list):
            raise CampaignError("GPU execution invocation topology is invalid")
        if not all(isinstance(part, str) for part in workload_command):
            raise CampaignError("GPU execution invocation command is invalid")
        invocation_sha256 = sha256_file(invocation_path)
        command_sha256 = semantic_sha256(workload_command)
        if (
            record.get("context") != context
            or record.get("invocation_sha256") != invocation_sha256
            or record.get("command_sha256") != command_sha256
        ):
            raise CampaignError("GPU terminal record differs from execution invocation")
        result = load_json(result_path, "GPU terminal result")
        content = dict(result)
        recorded_content_sha256 = content.pop("content_sha256", None)
        if recorded_content_sha256 != semantic_sha256(content):
            raise CampaignError("GPU terminal result content hash drifted")
        expected_result = gpu_budget_ledger.result_from_terminal(
            record,
            usage_ledger=usage_ledger,
            gpu_execution_ledger=Path(
                str(result.get("gpu_execution_ledger_path", ""))
            ),
        )
        expected_result["content_sha256"] = semantic_sha256(expected_result)
        if result != expected_result:
            raise CampaignError("GPU terminal result differs from locked ledger state")
        if result.get("terminal_record_sha256") != terminal_sha256:
            raise CampaignError("terminal result names a different ledger record")
        if receipt.get("gpu_execution_ledger_path") != result.get(
            "gpu_execution_ledger_path"
        ):
            raise CampaignError(
                "terminal result belongs to a different GPU execution ledger"
            )
        seen.add(str(terminal_sha256))
    if seen != set(v2_terminals):
        raise CampaignError("completion receipt omits a V2 terminal result")

    declared_invocations = receipt.get("lifecycle_invocations")
    if not isinstance(declared_invocations, list) or not declared_invocations:
        raise CampaignError("completion receipt lacks its V2 lifecycle invocations")
    invocation_documents: dict[str, dict[str, Any]] = {}
    for binding in declared_invocations:
        path = _bound_absolute_file(binding, "lifecycle invocation")
        invocation = load_json(path, "GPU lifecycle invocation")
        if canonical_content_sha256(invocation) != invocation.get("content_sha256"):
            raise CampaignError("GPU lifecycle invocation content drifted")
        invocation_hash = sha256_file(path)
        if invocation_hash in invocation_documents:
            raise CampaignError("completion lifecycle invocation is duplicated")
        invocation_documents[invocation_hash] = invocation
    expected_invocations = {
        str(record.get("invocation_sha256"))
        for record in matching
        if record.get("schema_version") == 2
    }
    if set(invocation_documents) != expected_invocations:
        raise CampaignError(
            "completion receipt does not exactly cover lifecycle invocations"
        )
    for record in matching:
        if record.get("schema_version") != 2:
            continue
        invocation = invocation_documents[str(record["invocation_sha256"])]
        if not (
            invocation.get("campaign_id") == CAMPAIGN_ID
            and invocation.get("phase") == record.get("phase")
            and invocation.get("context") == record.get("context")
            and invocation.get("workload_command_sha256")
            == record.get("command_sha256")
        ):
            raise CampaignError(
                "GPU lifecycle invocation differs from its terminal record"
            )


def validate_completion_receipt_usage(
    usage_ledger: Path,
    receipt: Mapping[str, Any],
    *,
    expected_phase: str,
    expected_identity: Mapping[str, Any],
    expected_command_sha256: Callable[[Mapping[str, Any]], str] | None = None,
    expected_gpu_ledger: Path | None = None,
    expected_gpu_lock: Path | None = None,
    usage_state: Any | None = None,
) -> list[dict[str, Any]]:
    """Bind one completion receipt to every attempt for its unit in this ledger.

    A receipt names the entire ordered attempt history, not only its successful
    tail.  Consequently copying a completed output to a fresh ledger, switching
    ledger paths, dropping a failed/resumed attempt, or appending another attempt
    for the same unit all fail closed.
    """

    if expected_phase not in EXECUTION_USAGE_PHASES:
        raise CampaignError(f"invalid completion usage phase: {expected_phase}")
    for name, expected in expected_identity.items():
        if receipt.get(name) != expected:
            raise CampaignError(f"completion receipt unit identity mismatched: {name}")
    canonical_path = _usage_ledger_path(usage_ledger)
    if receipt.get("usage_ledger_path") != canonical_path:
        raise CampaignError("completion receipt belongs to a different GPU usage ledger")
    if expected_gpu_ledger is not None and receipt.get(
        "gpu_execution_ledger_path"
    ) != str(expected_gpu_ledger.expanduser().resolve()):
        raise CampaignError("completion receipt belongs to a different GPU execution ledger")
    if expected_gpu_lock is not None and receipt.get(
        "gpu_admission_lock_path"
    ) != str(expected_gpu_lock.expanduser().resolve()):
        raise CampaignError("completion receipt belongs to a different GPU admission lock")
    if usage_state is None:
        try:
            with gpu_budget_ledger.locked_closed_snapshot(
                usage_ledger,
                budget_ns=gpu_budget_ledger.GPU_BUDGET_NS,
                expected_legacy_genesis_sha256=_expected_legacy_genesis(
                    usage_ledger
                ),
            ) as locked_state:
                return validate_completion_receipt_usage(
                    usage_ledger,
                    receipt,
                    expected_phase=expected_phase,
                    expected_identity=expected_identity,
                    expected_command_sha256=expected_command_sha256,
                    expected_gpu_ledger=expected_gpu_ledger,
                    expected_gpu_lock=expected_gpu_lock,
                    usage_state=locked_state,
                )
        except (OSError, ValueError, RuntimeError) as error:
            if isinstance(error, CampaignError):
                raise
            raise CampaignError(
                f"GPU usage ledger is not a stable closed snapshot: {error}"
            ) from error
    records = list(usage_state.records)
    matching: list[dict[str, Any]] = []
    for record in records:
        if _record_matches_execution(
            record,
            expected_phase=expected_phase,
            expected_identity=expected_identity,
        ):
            matching.append(record)
    declared_hashes = receipt.get("usage_record_sha256s")
    if not (
        isinstance(declared_hashes, list)
        and declared_hashes
        and all(isinstance(value, str) for value in declared_hashes)
        and len(declared_hashes) == len(set(declared_hashes))
    ):
        raise CampaignError("completion receipt has invalid GPU usage record list")
    observed_hashes = [str(record["record_sha256"]) for record in matching]
    if declared_hashes != observed_hashes:
        raise CampaignError("completion receipt GPU usage history differs from active ledger")
    if receipt.get("usage_record_sha256") != observed_hashes[-1]:
        raise CampaignError("completion receipt does not name its successful GPU usage tail")
    for record in matching:
        _validate_execution_usage_record(
            record,
            expected_phase=expected_phase,
            expected_identity=expected_identity,
        )
        if expected_command_sha256 is not None:
            expected = expected_command_sha256(record)
            if record.get("command_sha256") != expected:
                raise CampaignError("GPU usage record command differs from the unit invocation")
    if any(_usage_record_succeeded(record) for record in matching[:-1]):
        raise CampaignError("completion receipt contains an unexplained earlier successful attempt")
    if not _usage_record_succeeded(matching[-1]):
        raise CampaignError("completion receipt GPU usage tail is not successful")
    _validate_terminal_result_bindings(
        usage_ledger, receipt, matching, usage_state=usage_state
    )
    return matching


def completion_usage_fields(
    usage_ledger: Path,
    *,
    final_record_sha256: str,
    expected_phase: str,
    expected_identity: Mapping[str, Any],
    expected_command_sha256: Callable[[Mapping[str, Any]], str],
    terminal_results: Sequence[Mapping[str, Any]] | None = None,
    lifecycle_invocations: Sequence[Mapping[str, Any]] | None = None,
    gpu_ledger: Path | None = None,
    gpu_lock: Path | None = None,
    usage_state: Any | None = None,
) -> dict[str, Any]:
    """Build and validate the immutable usage fields before publishing a receipt."""

    if usage_state is None:
        try:
            with gpu_budget_ledger.locked_closed_snapshot(
                usage_ledger,
                budget_ns=gpu_budget_ledger.GPU_BUDGET_NS,
                expected_legacy_genesis_sha256=_expected_legacy_genesis(
                    usage_ledger
                ),
            ) as locked_state:
                return completion_usage_fields(
                    usage_ledger,
                    final_record_sha256=final_record_sha256,
                    expected_phase=expected_phase,
                    expected_identity=expected_identity,
                    expected_command_sha256=expected_command_sha256,
                    terminal_results=terminal_results,
                    lifecycle_invocations=lifecycle_invocations,
                    gpu_ledger=gpu_ledger,
                    gpu_lock=gpu_lock,
                    usage_state=locked_state,
                )
        except (OSError, ValueError, RuntimeError) as error:
            if isinstance(error, CampaignError):
                raise
            raise CampaignError(
                f"GPU usage ledger is not a stable closed snapshot: {error}"
            ) from error
    records = list(usage_state.records)
    hashes = [
        str(record["record_sha256"])
        for record in records
        if _record_matches_execution(
            record,
            expected_phase=expected_phase,
            expected_identity=expected_identity,
        )
    ]
    fields = {
        **dict(expected_identity),
        "usage_ledger_path": _usage_ledger_path(usage_ledger),
        "usage_record_sha256": final_record_sha256,
        "usage_record_sha256s": hashes,
        "terminal_results": list(terminal_results or []),
        "lifecycle_invocations": list(lifecycle_invocations or []),
    }
    if gpu_ledger is not None:
        fields["gpu_execution_ledger_path"] = str(gpu_ledger.expanduser().resolve())
    if gpu_lock is not None:
        fields["gpu_admission_lock_path"] = str(gpu_lock.expanduser().resolve())
    validate_completion_receipt_usage(
        usage_ledger,
        fields,
        expected_phase=expected_phase,
        expected_identity=expected_identity,
        expected_command_sha256=expected_command_sha256,
        expected_gpu_ledger=gpu_ledger,
        expected_gpu_lock=gpu_lock,
        usage_state=usage_state,
    )
    return fields


def validate_v8r3_quarantine_receipt_usage(
    usage_ledger: Path,
    receipt: Mapping[str, Any],
    *,
    usage_state: Any,
    expected_gpu_ledger: Path,
    expected_gpu_lock: Path,
) -> list[dict[str, Any]]:
    """Validate accounting ownership of the exact legacy V8R3 terminal."""

    matching = [
        record
        for record in usage_state.records
        if _record_matches_exact_v8r3_quarantine(record)
    ]
    hashes = [str(record["record_sha256"]) for record in matching]
    if not (
        len(matching) == 1
        and _usage_record_succeeded(matching[0])
        and receipt.get("classification")
        == "adaptive_v3r1_v8r3_discovery_unit_quarantine_owner_receipt"
        and receipt.get("campaign_revision") is None
        and receipt.get("execution_number") == 0
        and receipt.get("outer_fold") == 3
        and receipt.get("resume") is False
        and receipt.get("seed") == 20260828
        and receipt.get("variant") == "H0_no_factor"
        and receipt.get("usage_ledger_path") == _usage_ledger_path(usage_ledger)
        and receipt.get("usage_record_sha256s") == hashes
        and receipt.get("usage_record_sha256") == hashes[0]
        and receipt.get("gpu_execution_ledger_path")
        == str(expected_gpu_ledger.expanduser().resolve())
        and receipt.get("gpu_admission_lock_path")
        == str(expected_gpu_lock.expanduser().resolve())
        and receipt.get("reusable_success_overridden_by_quarantine") is True
        and receipt.get("excluded_from_discovery_selection") is True
    ):
        raise CampaignError("V8R3 quarantine receipt usage ownership drifted")
    _validate_execution_usage_record(
        matching[0],
        expected_phase="discovery",
        expected_identity={
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
        },
    )
    _validate_terminal_result_bindings(
        usage_ledger, receipt, matching, usage_state=usage_state
    )
    return matching


def reconcile_usage_ledger(
    usage_ledger: Path,
    receipt_specs: Sequence[
        tuple[Mapping[str, Any], str, Mapping[str, Any]]
    ],
    *,
    usage_state: Any | None = None,
    allow_exact_historical_benchmark_prefix: bool = False,
) -> tuple[list[dict[str, Any]], float]:
    """Prove that every executable ledger record is owned by exactly one receipt."""

    if usage_state is None:
        try:
            with gpu_budget_ledger.locked_closed_snapshot(
                usage_ledger,
                budget_ns=gpu_budget_ledger.GPU_BUDGET_NS,
                expected_legacy_genesis_sha256=_expected_legacy_genesis(
                    usage_ledger
                ),
            ) as locked_state:
                return reconcile_usage_ledger(
                    usage_ledger,
                    receipt_specs,
                    usage_state=locked_state,
                    allow_exact_historical_benchmark_prefix=(
                        allow_exact_historical_benchmark_prefix
                    ),
                )
        except (OSError, ValueError, RuntimeError) as error:
            if isinstance(error, CampaignError):
                raise
            raise CampaignError(
                f"GPU usage ledger is not a stable closed snapshot: {error}"
            ) from error
    state = usage_state
    if type(allow_exact_historical_benchmark_prefix) is not bool:
        raise CampaignError("historical benchmark whitelist flag must be boolean")
    records = list(state.records)
    elapsed = float(state.settled_usage_ns) / 1_000_000_000.0
    claimed: set[str] = set()
    for receipt, phase, identity in receipt_specs:
        if phase == "discovery_v8r3_quarantine":
            gpu_ledger_path = Path(str(receipt.get("gpu_execution_ledger_path", "")))
            gpu_lock_path = Path(str(receipt.get("gpu_admission_lock_path", "")))
            matched = validate_v8r3_quarantine_receipt_usage(
                usage_ledger,
                receipt,
                usage_state=state,
                expected_gpu_ledger=gpu_ledger_path,
                expected_gpu_lock=gpu_lock_path,
            )
        else:
            matched = validate_completion_receipt_usage(
                usage_ledger,
                receipt,
                expected_phase=phase,
                expected_identity=identity,
                usage_state=state,
            )
        for record in matched:
            record_hash = str(record["record_sha256"])
            if record_hash in claimed:
                raise CampaignError("GPU usage record is claimed by multiple completion receipts")
            claimed.add(record_hash)
    historical_seen: list[str] = []
    rootbind_seen: list[str] = []
    for record in records:
        record_hash = str(record["record_sha256"])
        if _is_execution_terminal(record):
            if record_hash in claimed:
                continue
            if (
                allow_exact_historical_benchmark_prefix
                and record_hash in HISTORICAL_BENCHMARK_TERMINAL_SHA256S
                and record.get("phase") == "efficiency_benchmark"
                and _execution_record_context(record)
                == HISTORICAL_BENCHMARK_USAGE_IDENTITY
            ):
                historical_seen.append(record_hash)
                continue
            if (
                allow_exact_historical_benchmark_prefix
                and record_hash
                == ROOTBIND1_BENCHMARK_FAILURE["terminal_record_sha256"]
            ):
                validate_rootbind1_failed_benchmark_terminal(record)
                rootbind_seen.append(record_hash)
                continue
            raise CampaignError("unexplained training or prediction GPU usage record")
        elif record.get("schema_version") == 2:
            # Reservations and heartbeats are reduced and linked to their exact
            # terminal by the shared supervisor.  Closed-ledger validation above
            # proves no lifecycle remains outstanding.
            if record.get("event") not in {"reservation", "heartbeat"}:
                raise CampaignError("unrecognized V2 GPU lifecycle record")
        else:
            _validate_non_execution_usage_record(record)
    if allow_exact_historical_benchmark_prefix:
        if historical_seen != list(HISTORICAL_BENCHMARK_TERMINAL_SHA256S):
            raise CampaignError("historical benchmark whitelist order or cover drifted")
        if rootbind_seen != [ROOTBIND1_BENCHMARK_FAILURE["terminal_record_sha256"]]:
            raise CampaignError("ROOTBIND1 benchmark failure ownership is not exact")
        positions = {
            str(record["record_sha256"]): number
            for number, record in enumerate(records)
        }
        quarantine_hash = V8R3_QUARANTINE_TERMINAL_SHA256
        rootbind_hash = ROOTBIND1_BENCHMARK_FAILURE["terminal_record_sha256"]
        rootbind_context_hashes = [
            str(record["record_sha256"])
            for record in records
            if _is_execution_terminal(record)
            and record.get("phase") == "efficiency_benchmark"
            and dict(_execution_record_context(record))
            == ROOTBIND1_BENCHMARK_USAGE_IDENTITY
        ]
        active_benchmark_hashes = [
            str(record["record_sha256"])
            for record in records
            if _is_execution_terminal(record)
            and record.get("phase") == "efficiency_benchmark"
            and _execution_record_context(record).get("authorization_generation")
            == "CONTEXT1"
        ]
        discovery_hashes = [
            str(record["record_sha256"])
            for record in records
            if _is_execution_terminal(record)
            and record.get("phase") == "discovery"
            and str(record.get("record_sha256")) != quarantine_hash
            and _execution_record_context(record).get("campaign_revision")
            == CAMPAIGN_REVISION
            and _execution_record_context(record).get("infrastructure_revision")
            == INFRASTRUCTURE_REVISION
        ]
        historical_positions = [
            positions[value] for value in HISTORICAL_BENCHMARK_TERMINAL_SHA256S
        ]
        if not (
            quarantine_hash in positions
            and rootbind_context_hashes == [rootbind_hash]
            and len(active_benchmark_hashes) == 1
            and discovery_hashes
            and historical_positions == sorted(historical_positions)
            and max(historical_positions)
            < positions[quarantine_hash]
            < positions[rootbind_hash]
            < positions[active_benchmark_hashes[0]]
            < min(positions[value] for value in discovery_hashes)
        ):
            raise CampaignError(
                "historical/quarantine/ROOTBIND1/CONTEXT1/discovery order drifted"
            )
    return records, elapsed


def validate_v8r3_quarantine_owner_receipt(
    *,
    project_root: Path,
    usage_ledger: Path,
    gpu_ledger: Path,
    gpu_lock: Path,
    usage_state: Any | None = None,
) -> dict[str, Any]:
    """Strictly replay the pre-issued immutable V8R3 quarantine owner.

    This function intentionally never creates, repairs, or supersedes the
    receipt.  The existing V8R4 correction artifact is the sole accounting
    owner of the successful-but-unsafe V8R3 terminal.
    """

    if usage_state is None:
        try:
            with gpu_budget_ledger.locked_closed_snapshot(
                usage_ledger,
                budget_ns=gpu_budget_ledger.GPU_BUDGET_NS,
                expected_legacy_genesis_sha256=_expected_legacy_genesis(usage_ledger),
            ) as locked_state:
                return validate_v8r3_quarantine_owner_receipt(
                    project_root=project_root,
                    usage_ledger=usage_ledger,
                    gpu_ledger=gpu_ledger,
                    gpu_lock=gpu_lock,
                    usage_state=locked_state,
                )
        except (OSError, ValueError, RuntimeError) as error:
            if isinstance(error, CampaignError):
                raise
            raise CampaignError("cannot lock ledger for V8R3 quarantine") from error

    receipt_path = (project_root / V8R3_QUARANTINE_RELATIVE).resolve()
    try:
        file_stat = os.stat(receipt_path, follow_symlinks=False)
    except OSError as error:
        raise CampaignError("immutable V8R3 quarantine owner receipt is absent") from error
    if not (
        receipt_path.is_file()
        and not receipt_path.is_symlink()
        and file_stat.st_nlink == 1
        and file_stat.st_mode & 0o777 == 0o444
        and file_stat.st_size == V8R3_QUARANTINE_BYTES
        and sha256_file(receipt_path) == V8R3_QUARANTINE_FILE_SHA256
    ):
        raise CampaignError("immutable V8R3 quarantine owner file binding drifted")
    receipt = load_json(receipt_path, "V8R3 quarantine owner receipt")
    expected_keys = {
        "campaign_id",
        "campaign_revision",
        "classification",
        "commercial_claim_authorized",
        "content_sha256",
        "discovery_completion_receipt_created",
        "excluded_from_discovery_selection",
        "execution_number",
        "gpu_admission_lock_path",
        "gpu_execution_ledger_path",
        "lifecycle_id",
        "lifecycle_invocations",
        "outer_fold",
        "parent_campaign_postvalidation_passed",
        "quarantine_reasons",
        "reusable_success_overridden_by_quarantine",
        "resume",
        "schema_version",
        "seed",
        "terminal_results",
        "trainer_return_code",
        "usage_ledger_path",
        "usage_record_sha256",
        "usage_record_sha256s",
        "variant",
    }
    if not (
        set(receipt) == expected_keys
        and receipt.get("content_sha256") == V8R3_QUARANTINE_CONTENT_SHA256
        and canonical_content_sha256(receipt) == V8R3_QUARANTINE_CONTENT_SHA256
        and receipt.get("classification")
        == "adaptive_v3r1_v8r3_discovery_unit_quarantine_owner_receipt"
        and receipt.get("campaign_id") == CAMPAIGN_ID
        and receipt.get("campaign_revision") is None
        and receipt.get("execution_number") == 0
        and receipt.get("outer_fold") == 3
        and receipt.get("resume") is False
        and receipt.get("seed") == 20260828
        and receipt.get("variant") == "H0_no_factor"
        and receipt.get("trainer_return_code") == 0
        and receipt.get("reusable_success_overridden_by_quarantine") is True
        and receipt.get("excluded_from_discovery_selection") is True
        and receipt.get("parent_campaign_postvalidation_passed") is False
        and receipt.get("discovery_completion_receipt_created") is False
        and receipt.get("commercial_claim_authorized") is False
        and receipt.get("usage_ledger_path") == _usage_ledger_path(usage_ledger)
        and receipt.get("gpu_execution_ledger_path")
        == str(gpu_ledger.expanduser().resolve())
        and receipt.get("gpu_admission_lock_path")
        == str(gpu_lock.expanduser().resolve())
        and isinstance(receipt.get("quarantine_reasons"), list)
        and len(receipt["quarantine_reasons"]) == 3
    ):
        raise CampaignError("immutable V8R3 quarantine owner schema/invariants drifted")
    validate_v8r3_quarantine_receipt_usage(
        usage_ledger,
        receipt,
        usage_state=usage_state,
        expected_gpu_ledger=gpu_ledger,
        expected_gpu_lock=gpu_lock,
    )
    validate_v8r3_quarantined_output_seal(
        project_root=project_root,
        usage_ledger=usage_ledger,
        gpu_ledger=gpu_ledger,
        owner_receipt=receipt,
        usage_state=usage_state,
    )
    return receipt


def verify_usage_ledger_prefix_binding(
    usage_ledger: Path,
    binding: Mapping[str, Any],
    *,
    project_root: Path,
    owner: Path,
    terminal_record_sha256: str,
    usage_state: Any | None = None,
) -> None:
    """Verify that a historical ledger binding is an exact active-ledger prefix."""

    if not isinstance(binding, Mapping) or "path" not in binding:
        raise CampaignError("historical GPU usage ledger binding is missing")
    bound_path = resolve_binding_path(
        binding["path"], project_root=project_root, owner=owner
    )
    if bound_path != usage_ledger.expanduser().resolve():
        raise CampaignError("historical seal belongs to a different GPU usage ledger")
    if usage_state is None:
        try:
            with gpu_budget_ledger.locked_closed_snapshot(
                usage_ledger,
                budget_ns=gpu_budget_ledger.GPU_BUDGET_NS,
                expected_legacy_genesis_sha256=_expected_legacy_genesis(
                    usage_ledger
                ),
            ) as locked_state:
                return verify_usage_ledger_prefix_binding(
                    usage_ledger,
                    binding,
                    project_root=project_root,
                    owner=owner,
                    terminal_record_sha256=terminal_record_sha256,
                    usage_state=locked_state,
                )
        except (OSError, ValueError, RuntimeError) as error:
            if isinstance(error, CampaignError):
                raise
            raise CampaignError(
                f"GPU usage ledger is not a stable closed snapshot: {error}"
            ) from error
    records = list(usage_state.records)
    positions = [
        index
        for index, record in enumerate(records)
        if record.get("record_sha256") == terminal_record_sha256
    ]
    if len(positions) != 1:
        raise CampaignError("historical GPU usage ledger terminal record is absent")
    try:
        raw_lines = bytes(usage_state.raw_bytes).splitlines(keepends=True)
    except (TypeError, ValueError) as error:
        raise CampaignError(f"cannot read historical GPU usage ledger prefix: {error}") from error
    if len(raw_lines) != len(records) or any(not line.endswith(b"\n") for line in raw_lines):
        raise CampaignError("GPU usage ledger is not a newline-terminated record stream")
    prefix = b"".join(raw_lines[: positions[0] + 1])
    if binding.get("bytes") != len(prefix):
        raise CampaignError("historical GPU usage ledger prefix length drifted")
    expected = binding.get("sha256", binding.get("file_sha256"))
    if not isinstance(expected, str) or hashlib.sha256(prefix).hexdigest() != expected:
        raise CampaignError("historical GPU usage ledger prefix hash drifted")
    try:
        prefix_state = gpu_budget_ledger.verify_ledger_bytes(
            prefix,
            budget_ns=gpu_budget_ledger.GPU_BUDGET_NS,
            expected_legacy_genesis_sha256=_expected_legacy_genesis(usage_ledger),
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise CampaignError(f"historical GPU usage ledger prefix is invalid: {error}") from error
    if prefix_state.open_reservations:
        raise CampaignError("historical GPU usage ledger prefix is not closed")
    declared_elapsed = binding.get("elapsed_seconds")
    if declared_elapsed is not None and float(declared_elapsed) != (
        float(prefix_state.settled_usage_ns) / 1_000_000_000.0
    ):
        raise CampaignError("historical GPU usage ledger elapsed time drifted")
    if binding.get("settled_usage_ns", prefix_state.settled_usage_ns) != int(
        prefix_state.settled_usage_ns
    ):
        raise CampaignError("historical GPU usage ledger settled usage drifted")
    if binding.get("records", len(prefix_state.records)) != len(prefix_state.records):
        raise CampaignError("historical GPU usage ledger record count drifted")
    if binding.get(
        "terminal_record_sha256", prefix_state.tail_sha256
    ) != prefix_state.tail_sha256:
        raise CampaignError("historical GPU usage ledger tail drifted")


def _verify_exact_file_prefix(
    binding: Mapping[str, Any], *, project_root: Path, owner: Path, label: str
) -> None:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "sha256",
        "bytes",
        "records",
    }:
        raise CampaignError(f"{label} prefix binding schema drifted")
    path = resolve_binding_path(
        binding["path"], project_root=project_root, owner=owner
    )
    length = binding.get("bytes")
    if type(length) is not int or int(length) <= 0:
        raise CampaignError(f"{label} prefix length is invalid")
    try:
        with path.open("rb") as stream:
            prefix = stream.read(int(length))
    except OSError as error:
        raise CampaignError(f"cannot read {label} prefix") from error
    if len(prefix) != int(length) or hashlib.sha256(prefix).hexdigest() != binding.get(
        "sha256"
    ):
        raise CampaignError(f"{label} prefix byte binding drifted")
    if prefix.count(b"\n") != binding.get("records") or not prefix.endswith(b"\n"):
        raise CampaignError(f"{label} prefix record topology drifted")


def validate_v8r3_quarantined_output_seal(
    *,
    project_root: Path,
    usage_ledger: Path,
    gpu_ledger: Path,
    owner_receipt: Mapping[str, Any],
    usage_state: Any,
) -> dict[str, Any]:
    """Strict replay of the immutable 11-file V8R3 output quarantine."""

    path = (project_root / V8R3_QUARANTINED_OUTPUT_SEAL_RELATIVE).resolve()
    try:
        file_stat = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise CampaignError("V8R3 quarantined-output seal is absent") from error
    if not (
        path.is_file()
        and not path.is_symlink()
        and file_stat.st_nlink == 1
        and file_stat.st_mode & 0o777 == 0o444
        and file_stat.st_size == V8R3_QUARANTINED_OUTPUT_SEAL_BYTES
        and sha256_file(path) == V8R3_QUARANTINED_OUTPUT_SEAL_FILE_SHA256
    ):
        raise CampaignError("V8R3 quarantined-output seal file binding drifted")
    seal = load_json(path, "V8R3 quarantined-output seal")
    if not (
        canonical_content_sha256(seal)
        == V8R3_QUARANTINED_OUTPUT_SEAL_CONTENT_SHA256
        == seal.get("content_sha256")
        and seal.get("classification")
        == "adaptive_v3r1_v8r3_discovery_postcondition_failure_quarantined_output_seal"
        and seal.get("campaign_id") == CAMPAIGN_ID
        and seal.get("files_are_single_link_exact_0444_regular") is True
        and seal.get("npz_identity_dtype") == "object"
        and seal.get("npz_pickle_free_replayable") is False
        and seal.get("output_repair_resume_reuse_selection_or_promotion_allowed")
        is False
        and seal.get("unit_completion_receipt_created") is False
        and seal.get("discovery_completion_seal_created") is False
        and seal.get("commercial_claim_authorized") is False
    ):
        raise CampaignError("V8R3 quarantined-output seal invariant drifted")
    files = seal.get("files")
    if not isinstance(files, list) or len(files) != 11:
        raise CampaignError("V8R3 quarantined-output material exact cover drifted")
    seen: set[Path] = set()
    for binding in files:
        if not isinstance(binding, Mapping) or set(binding) != {
            "path",
            "sha256",
            "bytes",
            "mode",
        }:
            raise CampaignError("V8R3 quarantined-output file schema drifted")
        file_path = resolve_binding_path(
            binding["path"], project_root=project_root, owner=path
        )
        if file_path in seen:
            raise CampaignError("V8R3 quarantined-output file is duplicated")
        seen.add(file_path)
        try:
            material_stat = os.stat(file_path, follow_symlinks=False)
        except OSError as error:
            raise CampaignError("V8R3 quarantined output file is absent") from error
        if not (
            file_path.is_file()
            and not file_path.is_symlink()
            and material_stat.st_nlink == 1
            and material_stat.st_mode & 0o777 == 0o444
            and binding.get("mode") == 0o444
            and binding.get("bytes") == material_stat.st_size
            and binding.get("sha256") == sha256_file(file_path)
        ):
            raise CampaignError("V8R3 quarantined output file binding drifted")
    owner_binding = seal.get("quarantine_owner_receipt")
    if not (
        isinstance(owner_binding, Mapping)
        and owner_binding.get("sha256") == V8R3_QUARANTINE_FILE_SHA256
        and owner_binding.get("bytes") == V8R3_QUARANTINE_BYTES
        and owner_binding.get("content_sha256") == V8R3_QUARANTINE_CONTENT_SHA256
        and owner_receipt.get("content_sha256") == V8R3_QUARANTINE_CONTENT_SHA256
    ):
        raise CampaignError("V8R3 quarantine owner/output seal binding drifted")
    diagnostic = seal.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise CampaignError("V8R3 quarantine diagnostic binding is absent")
    diagnostic_path = resolve_binding_path(
        diagnostic.get("path"), project_root=project_root, owner=path
    )
    diagnostic_doc = load_json(diagnostic_path, "V8R3 boundary diagnostic")
    if not (
        diagnostic.get("sha256") == sha256_file(diagnostic_path)
        and diagnostic.get("bytes") == diagnostic_path.stat().st_size
        and diagnostic.get("content_sha256")
        == canonical_content_sha256(diagnostic_doc)
        == diagnostic_doc.get("content_sha256")
    ):
        raise CampaignError("V8R3 quarantine diagnostic binding drifted")
    verify_usage_ledger_prefix_binding(
        usage_ledger,
        seal.get("usage_ledger_prefix", {}),
        project_root=project_root,
        owner=path,
        terminal_record_sha256=str(owner_receipt["usage_record_sha256"]),
        usage_state=usage_state,
    )
    execution_prefix = seal.get("execution_ledger_prefix")
    _verify_exact_file_prefix(
        execution_prefix,
        project_root=project_root,
        owner=path,
        label="V8R3 GPU execution ledger",
    )
    bound_execution_path = resolve_binding_path(
        execution_prefix["path"], project_root=project_root, owner=path
    )
    if bound_execution_path != gpu_ledger.expanduser().resolve():
        raise CampaignError("V8R3 quarantine belongs to another execution ledger")
    return seal


def _run_with_hard_timeout(command: Sequence[str], timeout_seconds: float) -> tuple[int, float, bool]:
    """Run the V7 supervisor; the parent never times or accounts the GPU child.

    ``timeout_seconds`` remains in the callable protocol solely for explicit
    injected test executors.  The wrapper owns the durable reservation, hard
    timeout, signal propagation, reap, and terminal accounting transaction.
    """

    del timeout_seconds
    started = time.monotonic()
    process = subprocess.run(list(command), check=False)
    return_code = int(process.returncode)
    elapsed = time.monotonic() - started
    return return_code, elapsed, False


def _trainer_command(
    *,
    python: Path,
    trainer: Path,
    training_input: TrainingInput,
    output_dir: Path,
    target_sealed_capability_receipt: Path,
    expected_admitted_context: Mapping[str, Any],
    variant: str,
    device: str,
    amp: bool,
    smoke_test: bool,
    resume: bool,
    campaign_phase: str = "discovery",
    promotion_authorization: Path | None = None,
    release_mode: str | None = None,
) -> list[str]:
    command = [
        str(python),
        str(trainer),
        "--mode",
        "train",
        "--campaign-phase",
        campaign_phase,
        "--cache",
        str(training_input.cache_dir),
        "--proposer-stack",
        str(training_input.proposer_stack),
        "--output-dir",
        str(output_dir),
        "--target-sealed-capability-receipt",
        str(target_sealed_capability_receipt),
        "--expected-admitted-context-json",
        json.dumps(
            dict(expected_admitted_context),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--outer-fold",
        str(training_input.outer_fold),
        "--seed",
        str(training_input.seed),
        "--variant",
        variant,
        "--device",
        device,
        "--deterministic",
        "--epochs",
        "120",
        "--minimum-epochs",
        "20",
        "--patience",
        "18",
        "--learning-rate",
        "0.0003",
        "--weight-decay",
        "0.0001",
        "--chunk-windows",
        "32",
        "--warmup-windows",
        "2",
        "--gradient-accumulation-sessions",
        "4",
    ]
    command.append("--amp" if amp else "--no-amp")
    if smoke_test:
        command.append("--smoke-test")
    if resume:
        command.append("--resume")
    if campaign_phase == "promotion":
        if promotion_authorization is None:
            raise CampaignError("promotion trainer command lacks promotion authorization")
        if release_mode not in RELEASE_MODES:
            raise CampaignError("promotion trainer command lacks locked release mode")
        command.extend(["--promotion-authorization", str(promotion_authorization)])
        command.extend(["--release-mode", str(release_mode)])
    elif promotion_authorization is not None:
        raise CampaignError("discovery trainer command cannot accept promotion authorization")
    elif release_mode is not None:
        raise CampaignError("discovery trainer command cannot lock a promotion release mode")
    return command


def _admitted_command(
    *,
    python: Path,
    wrapper: Path,
    gpu_lock: Path,
    gpu_ledger: Path,
    usage_ledger: Path,
    result_file: Path,
    phase: str,
    context: Mapping[str, Any],
    invocation_sha256: str,
    authorization_path: Path,
    authorization_sha256: str,
    trainer_command: Sequence[str],
) -> list[str]:
    if phase not in EXECUTION_USAGE_PHASES:
        raise CampaignError("invalid GPU lifecycle phase")
    if not _is_sha256(invocation_sha256):
        raise CampaignError("GPU execution invocation SHA-256 is invalid")
    authorization_path = authorization_path.expanduser().resolve()
    if (
        not authorization_path.is_file()
        or authorization_path.is_symlink()
        or authorization_path.stat().st_nlink != 1
        or authorization_path.stat().st_mode & 0o777 != 0o444
        or not _is_sha256(authorization_sha256)
        or sha256_file(authorization_path) != authorization_sha256
    ):
        raise CampaignError("V8 admitted-child authorization binding is invalid")
    return [
        str(python),
        str(wrapper),
        "--lock-file",
        str(gpu_lock),
        "--ledger",
        str(gpu_ledger),
        "--usage-ledger",
        str(usage_ledger),
        "--result-file",
        str(result_file),
        "--campaign-id",
        CAMPAIGN_ID,
        "--phase",
        phase,
        "--context-json",
        canonical_json_bytes(dict(context)).decode("utf-8"),
        "--invocation-sha256",
        invocation_sha256,
        "--authorization-path",
        str(authorization_path),
        "--authorization-sha256",
        authorization_sha256,
        "--budget-seconds",
        str(GPU_BUDGET_SECONDS),
        "--",
        *trainer_command,
    ]


def _execution_context(
    identity: Mapping[str, Any], *, execution_number: int, resume: bool
) -> dict[str, Any]:
    if identity.get("campaign_revision") != CAMPAIGN_REVISION:
        raise CampaignError("GPU execution identity is not owned by V8R4")
    if identity.get("infrastructure_revision") != INFRASTRUCTURE_REVISION:
        raise CampaignError("GPU execution identity is not owned by V8R4A")
    return {
        **dict(identity),
        "execution_number": int(execution_number),
        "resume": bool(resume),
    }


def _create_execution_invocation(
    path: Path,
    *,
    phase: str,
    context: Mapping[str, Any],
    unit_invocation_path: Path,
    workload_command: Sequence[str],
) -> dict[str, Any]:
    return create_once_json(
        path,
        {
            "schema_version": 2,
            "classification": "adaptive_v3r1_v8r4_gpu_execution_invocation",
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "phase": phase,
            "context": dict(context),
            "unit_invocation": bind_file(unit_invocation_path),
            "workload_command": list(workload_command),
            "workload_command_sha256": semantic_sha256(list(workload_command)),
            "parent_side_elapsed_accounting": False,
        },
    )


def _validate_execution_invocation(
    path: Path,
    *,
    phase: str,
    context: Mapping[str, Any],
    unit_invocation_path: Path,
    workload_command: Sequence[str],
) -> dict[str, Any]:
    invocation = load_json(path, "GPU execution invocation")
    if canonical_content_sha256(invocation) != invocation.get("content_sha256"):
        raise CampaignError("GPU execution invocation content drifted")
    unit_binding = invocation.get("unit_invocation")
    expected_unit = bind_file(unit_invocation_path)
    expected_command = list(workload_command)
    if not (
        invocation.get("schema_version") == 2
        and invocation.get("classification")
        == "adaptive_v3r1_v8r4_gpu_execution_invocation"
        and invocation.get("campaign_id") == CAMPAIGN_ID
        and invocation.get("campaign_revision") == CAMPAIGN_REVISION
        and invocation.get("phase") == phase
        and invocation.get("context") == dict(context)
        and unit_binding == expected_unit
        and invocation.get("workload_command") == expected_command
        and invocation.get("workload_command_sha256")
        == semantic_sha256(expected_command)
        and invocation.get("parent_side_elapsed_accounting") is False
    ):
        raise CampaignError("GPU execution invocation differs from its locked context")
    return invocation


def _load_execution_terminal_result(
    *,
    invocation_path: Path,
    result_path: Path,
    usage_ledger: Path,
    phase: str,
    context: Mapping[str, Any],
    unit_invocation_path: Path,
    workload_command: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_execution_invocation(
        invocation_path,
        phase=phase,
        context=context,
        unit_invocation_path=unit_invocation_path,
        workload_command=workload_command,
    )
    invocation_sha256 = sha256_file(invocation_path)
    command_sha256 = semantic_sha256(list(workload_command))
    try:
        result = gpu_budget_ledger.load_validate_terminal_result(
            result_path,
            usage_ledger=usage_ledger,
            expected_campaign_id=CAMPAIGN_ID,
            expected_phase=phase,
            expected_context=dict(context),
            expected_command_sha256=command_sha256,
            expected_invocation_sha256=invocation_sha256,
            expected_legacy_genesis_sha256=_expected_legacy_genesis(usage_ledger),
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise CampaignError(f"invalid GPU terminal result: {error}") from error
    binding = {
        "terminal_record_sha256": result["terminal_record_sha256"],
        "execution_invocation": bind_file(invocation_path),
        "result": bind_file(result_path),
    }
    return result, binding


def _execution_directories(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise CampaignError("GPU executions root is not a canonical directory")
    entries = sorted(root.iterdir(), key=lambda item: item.name)
    directories = [path for path in entries if path.name.startswith("execution_")]
    expected = [f"execution_{number:03d}" for number in range(len(directories))]
    if (
        [path.name for path in directories] != expected
        or any(path.is_symlink() or not path.is_dir() for path in directories)
    ):
        raise CampaignError("GPU execution directories are not a contiguous history")
    staging_name = f".execution_{len(directories):03d}.staging"
    foreign = [path for path in entries if path not in directories]
    if len(foreign) > 1 or (foreign and foreign[0].name != staging_name):
        raise CampaignError("GPU execution history has a foreign or non-tail entry")
    if foreign:
        staging = foreign[0]
        if staging.is_symlink() or not staging.is_dir():
            raise CampaignError("GPU execution staging tail is aliased")
        staged_entries = sorted(staging.iterdir(), key=lambda item: item.name)
        if [item.name for item in staged_entries] not in ([], ["invocation.json"]):
            raise CampaignError("GPU execution staging tail has unknown content")
        if staged_entries:
            status = os.stat(staged_entries[0], follow_symlinks=False)
            if not (
                stat.S_ISREG(status.st_mode)
                and status.st_nlink == 1
                and stat.S_IMODE(status.st_mode) == 0o444
            ):
                raise CampaignError("GPU execution staged invocation is not immutable")
    for directory in directories:
        invocation = directory / "invocation.json"
        status = os.stat(invocation, follow_symlinks=False)
        if not (
            stat.S_ISREG(status.st_mode)
            and status.st_nlink == 1
            and stat.S_IMODE(status.st_mode) == 0o444
        ):
            raise CampaignError("GPU execution directory lacks an immutable invocation")
    # Complete the durability half of a rename observed after a killed parent.
    # No new index is created and the immutable invocation is revalidated first.
    _fsync_directory(root)
    return directories


def _publish_execution_directory(
    root: Path,
    *,
    execution_number: int,
    create_invocation: Callable[[Path], Mapping[str, Any]],
    validate_invocation: Callable[[Path], Mapping[str, Any]],
) -> Path:
    """Publish one indexed directory only after its invocation is durable."""

    existing = _execution_directories(root)
    if execution_number != len(existing):
        raise CampaignError("GPU execution staging index is not the exact tail")
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".execution_{execution_number:03d}.staging"
    final = root / f"execution_{execution_number:03d}"
    if final.exists():
        raise CampaignError("GPU execution final tail appeared unexpectedly")
    if not staging.exists():
        staging.mkdir(mode=0o700)
        _fsync_directory(root)
        _publication_fault("indexed_directory_staged", final)
    elif staging.is_symlink() or not staging.is_dir():
        raise CampaignError("GPU execution staging tail is not a directory")
    invocation = staging / "invocation.json"
    entries = sorted(staging.iterdir(), key=lambda item: item.name)
    if not entries:
        create_invocation(invocation)
    elif [item.name for item in entries] == ["invocation.json"]:
        validate_invocation(invocation)
    else:
        raise CampaignError("GPU execution staging tail has unknown content")
    validate_invocation(invocation)
    _fsync_directory(staging)
    _publication_fault("indexed_invocation_durable", final)
    _rename_staged_directory_once(staging, final)
    _publication_fault("indexed_directory_linked", final)
    _fsync_directory(root)
    _publication_fault("indexed_directory_published", final)
    return final


def run_training_unit(
    *,
    project_root: Path,
    run_root: Path,
    training_input: TrainingInput,
    variant: str,
    authorization: Mapping[str, Any],
    contract_binding: Mapping[str, Any],
    target_sealed_capability_receipt: Path,
    python: Path,
    trainer: Path,
    wrapper: Path,
    gpu_lock: Path,
    gpu_ledger: Path,
    usage_ledger: Path,
    device: str,
    amp: bool,
    smoke_test: bool,
    command_runner: Callable[[Sequence[str], float], tuple[int, float, bool]] = _run_with_hard_timeout,
) -> dict[str, Any]:
    require_canonical_gpu_lock(project_root, gpu_lock)
    if training_input.partition_manifest_binding is not None:
        partition_path = Path(
            str(training_input.partition_manifest_binding.get("path", ""))
        )
        if bind_file(partition_path) != dict(training_input.partition_manifest_binding):
            raise CampaignError("V8R4 partition manifest binding drifted before launch")
    elif not smoke_test:
        raise CampaignError("non-smoke training lacks its V8R4 partition manifest")
    if training_input.cache_input_binding is not None:
        expected_manifest = training_input.cache_input_binding.get("manifest")
        if not isinstance(expected_manifest, Mapping) or expected_manifest.get(
            "sha256"
        ) != training_input.cache_manifest_sha256:
            raise CampaignError("training cache stored manifest SHA-256 drifted")
        _, live_cache_binding = verify_training_cache_inputs(
            project_root,
            training_input.cache_dir,
            outer_fold=training_input.outer_fold,
        )
        if canonical_json_bytes(live_cache_binding) != canonical_json_bytes(
            dict(training_input.cache_input_binding)
        ):
            raise CampaignError("training cache input binding drifted before launch")
    elif not smoke_test:
        raise CampaignError("non-smoke training lacks its canonical cache input binding")
    if training_input.proposer_stack_binding is not None:
        expected_stack = training_input.proposer_stack_binding
        if not (
            set(expected_stack) == {"path", "sha256", "bytes"}
            and Path(str(expected_stack.get("path", "")))
            == training_input.proposer_stack
            and expected_stack.get("sha256")
            == training_input.proposer_stack_sha256
            and type(expected_stack.get("bytes")) is int
        ):
            raise CampaignError("training proposer stored binding is malformed")
        live_stack_binding = verify_training_bound_file(
            project_root,
            training_input.proposer_stack,
            expected_sha256=training_input.proposer_stack_sha256,
            expected_bytes=int(expected_stack["bytes"]),
        )
        if canonical_json_bytes(live_stack_binding) != canonical_json_bytes(
            dict(expected_stack)
        ):
            raise CampaignError("training proposer input binding drifted before launch")
    elif not smoke_test:
        raise CampaignError("non-smoke training lacks its canonical proposer binding")
    bind_run_usage_ledger(run_root, usage_ledger, execution_scope="discovery")
    usage_ledger_identity_path = run_root / "GPU_USAGE_LEDGER_IDENTITY.json"
    unit_root = (
        run_root
        / "units"
        / f"outer_{training_input.outer_fold}_seed_{training_input.seed}_{variant}"
    )
    output_dir = unit_root / "attempt_000" / "output"
    invocation_path = unit_root / "attempt_000" / "invocation.json"
    executions_root = unit_root / "attempt_000" / "executions"
    completion_path = unit_root / "completion_receipt.json"
    usage_identity = {
        "campaign_revision": CAMPAIGN_REVISION,
        "infrastructure_revision": INFRASTRUCTURE_REVISION,
        "outer_fold": training_input.outer_fold,
        "seed": training_input.seed,
        "variant": variant,
    }
    authorization_binding = authorization.get("authorization_binding")
    if not isinstance(authorization_binding, Mapping):
        raise CampaignError("discovery lacks its V8 pretrain authorization binding")
    authorization_path_value = authorization_binding.get("path")
    authorization_sha256 = authorization_binding.get("sha256")
    if not isinstance(authorization_path_value, str) or not isinstance(
        authorization_sha256, str
    ):
        raise CampaignError("discovery V8 pretrain authorization binding is malformed")
    authorization_path = Path(authorization_path_value)
    if not authorization_path.is_absolute():
        authorization_path = project_root / authorization_path

    def expected_usage_command_sha256(record: Mapping[str, Any]) -> str:
        if record.get("schema_version") != 2:
            raise CampaignError("V7 discovery units cannot reuse a legacy V1 execution")
        context = _execution_record_context(record)
        resume_value = context.get("resume")
        if type(resume_value) is not bool:
            raise CampaignError("discovery GPU usage record has invalid resume state")
        execution_number = context.get("execution_number")
        if type(execution_number) is not int or int(execution_number) < 0:
            raise CampaignError("discovery GPU usage record has invalid execution number")
        expected_command = _trainer_command(
            python=python,
            trainer=trainer,
            training_input=training_input,
            output_dir=output_dir,
            target_sealed_capability_receipt=target_sealed_capability_receipt,
            expected_admitted_context=dict(context),
            variant=variant,
            device=device,
            amp=amp,
            smoke_test=smoke_test,
            resume=resume_value,
        )
        return semantic_sha256(expected_command)

    if completion_path.exists():
        receipt = load_json(completion_path, "discovery completion receipt")
        if canonical_content_sha256(receipt) != receipt.get("content_sha256"):
            raise CampaignError(f"discovery receipt content hash drifted: {completion_path}")
        validate_completion_receipt_usage(
            usage_ledger,
            receipt,
            expected_phase="discovery",
            expected_identity=usage_identity,
            expected_command_sha256=expected_usage_command_sha256,
            expected_gpu_ledger=gpu_ledger,
            expected_gpu_lock=gpu_lock,
        )
        validation = validate_training_output(
            output_dir,
            outer_fold=training_input.outer_fold,
            seed=training_input.seed,
            variant=variant,
            cache_dir=training_input.cache_dir,
        )
        if receipt.get("validated_output") != validation:
            raise CampaignError(f"completed discovery output drifted: {unit_root}")
        return receipt
    base_command = _trainer_command(
        python=python,
        trainer=trainer,
        training_input=training_input,
        output_dir=output_dir,
        target_sealed_capability_receipt=target_sealed_capability_receipt,
        expected_admitted_context=_execution_context(
            usage_identity, execution_number=0, resume=False
        ),
        variant=variant,
        device=device,
        amp=amp,
        smoke_test=smoke_test,
        resume=False,
    )
    invocation = {
        "schema_version": 1,
        "classification": "adaptive_v3r1_v8r4_discovery_training_invocation",
        "campaign_id": CAMPAIGN_ID,
        "campaign_revision": CAMPAIGN_REVISION,
        "infrastructure_revision": INFRASTRUCTURE_REVISION,
        "contract": dict(contract_binding),
        "pretrain_authorization": dict(authorization["authorization_binding"]),
        "outer_fold": training_input.outer_fold,
        "validation_fold": (training_input.outer_fold + 1) % 6,
        "seed": training_input.seed,
        "variant": variant,
        "outer_test_opened": False,
        "outer_test_features_or_targets_authorized": False,
        "cache_manifest": bind_file(training_input.cache_dir / "manifest.json"),
        "cache_inputs": (
            dict(training_input.cache_input_binding)
            if training_input.cache_input_binding is not None
            else None
        ),
        "proposer_stack": (
            dict(training_input.proposer_stack_binding)
            if training_input.proposer_stack_binding is not None
            else bind_file(training_input.proposer_stack)
        ),
        "partition_manifest": (
            dict(training_input.partition_manifest_binding)
            if training_input.partition_manifest_binding is not None
            else None
        ),
        "trainer": bind_file(trainer),
        "gpu_wrapper": bind_file(wrapper),
        "usage_ledger_identity": bind_file(usage_ledger_identity_path),
        "base_trainer_command": base_command,
        "gpu_hours_hard": GPU_HOURS_HARD,
    }
    create_once_json(invocation_path, invocation)
    def workload(execution_number: int, resume_value: bool) -> list[str]:
        expected_context = _execution_context(
            usage_identity,
            execution_number=execution_number,
            resume=resume_value,
        )
        return _trainer_command(
            python=python,
            trainer=trainer,
            training_input=training_input,
            output_dir=output_dir,
            target_sealed_capability_receipt=target_sealed_capability_receipt,
            expected_admitted_context=expected_context,
            variant=variant,
            device=device,
            amp=amp,
            smoke_test=smoke_test,
            resume=resume_value,
        )

    terminal_results: list[dict[str, Any]] = []
    lifecycle_invocations: list[dict[str, Any]] = []
    successful_result: dict[str, Any] | None = None
    for number, execution_root in enumerate(_execution_directories(executions_root)):
        execution_invocation_path = execution_root / "invocation.json"
        if not execution_invocation_path.is_file():
            raise CampaignError("discovery execution directory lacks its invocation")
        execution_invocation = load_json(
            execution_invocation_path, "discovery GPU execution invocation"
        )
        context = execution_invocation.get("context")
        if not isinstance(context, Mapping):
            raise CampaignError("discovery execution invocation lacks context")
        resume_value = context.get("resume")
        expected_context = _execution_context(
            usage_identity,
            execution_number=number,
            resume=bool(resume_value) if type(resume_value) is bool else False,
        )
        if type(resume_value) is not bool or context != expected_context:
            raise CampaignError("discovery execution history context drifted")
        result_path = execution_root / "terminal_result.json"
        if not result_path.exists():
            _validate_execution_invocation(
                execution_invocation_path,
                phase="discovery",
                context=expected_context,
                unit_invocation_path=invocation_path,
                workload_command=workload(number, resume_value),
            )
            lifecycle_invocations.append(bind_file(execution_invocation_path))
            recovery_command = _admitted_command(
                python=python,
                wrapper=wrapper,
                gpu_lock=gpu_lock,
                gpu_ledger=gpu_ledger,
                usage_ledger=usage_ledger,
                result_file=result_path,
                phase="discovery",
                context=expected_context,
                invocation_sha256=sha256_file(execution_invocation_path),
                authorization_path=authorization_path,
                authorization_sha256=authorization_sha256,
                trainer_command=workload(number, resume_value),
            )
            command_runner(recovery_command, float(GPU_BUDGET_SECONDS))
            if not result_path.is_file():
                state = _verify_usage_state(usage_ledger)
                if int(state.remaining_ns) <= 0:
                    raise BudgetExhausted("ten-GPU-hour budget is exhausted")
                raise CampaignError(
                    "GPU supervisor could not recover the existing execution result"
                )
        result, binding = _load_execution_terminal_result(
            invocation_path=execution_invocation_path,
            result_path=result_path,
            usage_ledger=usage_ledger,
            phase="discovery",
            context=expected_context,
            unit_invocation_path=invocation_path,
            workload_command=workload(number, resume_value),
        )
        if not any(
            item.get("sha256") == sha256_file(execution_invocation_path)
            for item in lifecycle_invocations
        ):
            lifecycle_invocations.append(bind_file(execution_invocation_path))
        terminal_results.append(binding)
        if result.get("reusable_success") is True:
            if successful_result is not None:
                raise CampaignError("discovery unit has multiple successful GPU executions")
            successful_result = result

    if successful_result is None:
        execution_number = len(_execution_directories(executions_root))
        resume = output_dir.exists() and any(output_dir.iterdir())
        context = _execution_context(
            usage_identity, execution_number=execution_number, resume=resume
        )
        trainer_command = workload(execution_number, resume)
        execution_root = _publish_execution_directory(
            executions_root,
            execution_number=execution_number,
            create_invocation=lambda staged: _create_execution_invocation(
                staged,
                phase="discovery",
                context=context,
                unit_invocation_path=invocation_path,
                workload_command=trainer_command,
            ),
            validate_invocation=lambda staged: _validate_execution_invocation(
                staged,
                phase="discovery",
                context=context,
                unit_invocation_path=invocation_path,
                workload_command=trainer_command,
            ),
        )
        execution_invocation_path = execution_root / "invocation.json"
        result_path = execution_root / "terminal_result.json"
        lifecycle_invocations.append(bind_file(execution_invocation_path))
        command = _admitted_command(
            python=python,
            wrapper=wrapper,
            gpu_lock=gpu_lock,
            gpu_ledger=gpu_ledger,
            usage_ledger=usage_ledger,
            result_file=result_path,
            phase="discovery",
            context=context,
            invocation_sha256=sha256_file(execution_invocation_path),
            authorization_path=authorization_path,
            authorization_sha256=authorization_sha256,
            trainer_command=trainer_command,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        command_runner(command, float(GPU_BUDGET_SECONDS))
        if not result_path.is_file():
            state = _verify_usage_state(usage_ledger)
            if int(state.remaining_ns) <= 0:
                raise BudgetExhausted("ten-GPU-hour budget is exhausted")
            raise CampaignError("GPU supervisor exited without an atomic terminal result")
        successful_result, binding = _load_execution_terminal_result(
            invocation_path=execution_invocation_path,
            result_path=result_path,
            usage_ledger=usage_ledger,
            phase="discovery",
            context=context,
            unit_invocation_path=invocation_path,
            workload_command=trainer_command,
        )
        terminal_results.append(binding)
        if successful_result.get("reusable_success") is not True:
            if successful_result.get("hard_timeout_reached") is True:
                raise BudgetExhausted(
                    "trainer was stopped at the ten-GPU-hour hard ceiling"
                )
            raise CampaignError(
                f"v3r1 discovery trainer failed for "
                f"{training_input.outer_fold}/{training_input.seed}/{variant}: "
                f"{successful_result.get('return_code')}"
            )
    validation = validate_training_output(
        output_dir,
        outer_fold=training_input.outer_fold,
        seed=training_input.seed,
        variant=variant,
        cache_dir=training_input.cache_dir,
    )
    usage_fields = completion_usage_fields(
        usage_ledger,
        final_record_sha256=str(successful_result["terminal_record_sha256"]),
        expected_phase="discovery",
        expected_identity=usage_identity,
        expected_command_sha256=expected_usage_command_sha256,
        terminal_results=terminal_results,
        lifecycle_invocations=lifecycle_invocations,
        gpu_ledger=gpu_ledger,
        gpu_lock=gpu_lock,
    )
    return create_once_json(
        completion_path,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_v8r4_discovery_unit_completion",
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "outer_test_opened": False,
            "outer_fold": training_input.outer_fold,
            "validation_fold": (training_input.outer_fold + 1) % 6,
            "seed": training_input.seed,
            "variant": variant,
            "invocation": bind_file(invocation_path),
            **usage_fields,
            "validated_output": validation,
            "commercial_claim_authorized": False,
        },
    )


def build_discovery_shard_completion(
    *,
    project_root: Path,
    shard_root: Path,
    outer_fold_shard: int,
    receipts: Sequence[Mapping[str, Any]],
    contract_binding: Mapping[str, Any],
    authorization: Mapping[str, Any],
    training_index_binding: Mapping[str, Any],
    usage_ledger: Path,
    gpu_ledger: Path,
    gpu_lock: Path,
    benchmark_receipt_path: Path,
) -> dict[str, Any]:
    expected = {
        (outer_fold_shard, seed, variant)
        for seed in SEEDS
        for variant in VARIANTS
    }
    keys = {
        (int(item["outer_fold"]), int(item["seed"]), str(item["variant"]))
        for item in receipts
    }
    if outer_fold_shard not in OUTER_RUNS or len(receipts) != 9 or keys != expected:
        raise CampaignError("V8R4 discovery shard is not an exact 3x3 unit cover")
    units: list[dict[str, Any]] = []
    with gpu_budget_ledger.locked_closed_snapshot(
        usage_ledger,
        budget_ns=gpu_budget_ledger.GPU_BUDGET_NS,
        expected_legacy_genesis_sha256=_expected_legacy_genesis(usage_ledger),
    ) as usage_state:
        quarantine = validate_v8r3_quarantine_owner_receipt(
            project_root=project_root,
            usage_ledger=usage_ledger,
            gpu_ledger=gpu_ledger,
            gpu_lock=gpu_lock,
            usage_state=usage_state,
        )
        benchmark = load_efficiency_benchmark_module(project_root)
        benchmark_receipt = validate_pre_discovery_efficiency_benchmark(
            project_root=project_root,
            receipt_path=benchmark_receipt_path,
            usage_ledger=usage_ledger,
            gpu_ledger=gpu_ledger,
            gpu_lock=gpu_lock,
            authorization=authorization,
            usage_state=usage_state,
            require_no_discovery_terminal=False,
        )
        for receipt in receipts:
            key = (
                int(receipt["outer_fold"]),
                int(receipt["seed"]),
                str(receipt["variant"]),
            )
            path = (
                shard_root
                / "units"
                / f"outer_{key[0]}_seed_{key[1]}_{key[2]}"
                / "completion_receipt.json"
            )
            live = load_json(path, "V8R4 shard unit completion receipt")
            if not (
                live == receipt
                and canonical_content_sha256(live) == live.get("content_sha256")
                and live.get("campaign_revision") == CAMPAIGN_REVISION
                and live.get("classification")
                == "adaptive_v3r1_v8r4_discovery_unit_completion"
            ):
                raise CampaignError("V8R4 shard unit receipt drifted")
            validate_completion_receipt_usage(
                usage_ledger,
                live,
                expected_phase="discovery",
                expected_identity={
                    "campaign_revision": CAMPAIGN_REVISION,
                    "infrastructure_revision": INFRASTRUCTURE_REVISION,
                    "outer_fold": key[0],
                    "seed": key[1],
                    "variant": key[2],
                },
                expected_gpu_ledger=gpu_ledger,
                expected_gpu_lock=gpu_lock,
                usage_state=usage_state,
            )
            units.append(
                {
                    "outer_fold": key[0],
                    "seed": key[1],
                    "variant": key[2],
                    "receipt": bind_file(path),
                }
            )
        ledger_binding = usage_snapshot_binding(usage_ledger, usage_state)
        return create_once_json(
            shard_root / DISCOVERY_SHARD_SEAL_NAME,
            {
                "schema_version": 1,
                "classification": DISCOVERY_SHARD_SEAL_CLASSIFICATION,
                "campaign_id": CAMPAIGN_ID,
                "campaign_revision": CAMPAIGN_REVISION,
                "infrastructure_revision": INFRASTRUCTURE_REVISION,
                "outer_fold_shard": outer_fold_shard,
                "contract": dict(contract_binding),
                "pretrain_authorization": dict(authorization["authorization_binding"]),
                "training_index": dict(training_index_binding),
                "completed_units": 9,
                "peer_outer_shard_pack_mounted_or_opened": False,
                "combined_target_bearing_cache_opened": False,
                "outer_prediction_pack_absent": True,
                "physical_boundary": dict(DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
                "gpu_usage_ledger_prefix": ledger_binding,
                "pre_discovery_efficiency_benchmark": bind_file(
                    benchmark_receipt_path, relative_to=project_root
                ),
                "v8r3_quarantine_owner": bind_file(
                    (project_root / V8R3_QUARANTINE_RELATIVE).resolve(),
                    relative_to=project_root,
                ),
                "units": sorted(
                    units, key=lambda item: (item["seed"], item["variant"])
                ),
                "cross_outer_validation_reuse_present": True,
                "fully_nested_confirmatory_oof": False,
                "prospective_confirmation_required": True,
                "ready_for_pack_free_shard_aggregation": True,
                "commercial_claim_authorized": False,
            },
        )


def _pack_free_discovery_usage_cover(
    *,
    project_root: Path,
    usage_ledger: Path,
    shard_seals: Sequence[tuple[Path, Mapping[str, Any]]],
    usage_state: Any,
) -> tuple[list[dict[str, Any]], float]:
    """Prove the ledger cover using seal identities, never unit artifacts."""

    benchmark = load_efficiency_benchmark_module(project_root)
    historical_hash_order = [
        str(item["terminal_result"]["terminal_record_sha256"])
        for item in benchmark.HISTORICAL_BENCHMARK_ATTEMPTS
    ]
    historical_hashes = set(historical_hash_order)
    expected_units = {
        (
            int(unit.get("outer_fold", -1)),
            int(unit.get("seed", -1)),
            str(unit.get("variant", "")),
        )
        for _path, seal in shard_seals
        for unit in seal.get("units", [])
        if isinstance(unit, Mapping)
    }
    if expected_units != set(EXPECTED_DISCOVERY_UNITS):
        raise CampaignError("pack-free shard identities do not cover 18 discovery units")
    active_benchmark: list[dict[str, Any]] = []
    rootbind_failure: list[dict[str, Any]] = []
    historical_seen: set[str] = set()
    quarantine: list[dict[str, Any]] = []
    by_unit: dict[tuple[int, int, str], list[dict[str, Any]]] = {
        key: [] for key in expected_units
    }
    records = list(usage_state.records)
    positions = {
        str(record["record_sha256"]): number
        for number, record in enumerate(records)
    }
    for record in records:
        if not _is_execution_terminal(record):
            if record.get("schema_version") == 2:
                if record.get("event") not in {"reservation", "heartbeat"}:
                    raise CampaignError("pack-free ledger has an unknown lifecycle event")
            else:
                _validate_non_execution_usage_record(record)
            continue
        record_hash = str(record.get("record_sha256", ""))
        phase = record.get("phase")
        context = dict(_execution_record_context(record))
        if phase == benchmark.BENCHMARK_PHASE:
            if record_hash in historical_hashes:
                historical_seen.add(record_hash)
                _validate_execution_usage_record(
                    record,
                    expected_phase=benchmark.BENCHMARK_PHASE,
                    expected_identity={
                        "benchmark_id": benchmark.BENCHMARK_ID,
                        "outer_fold": 3,
                        "seed": 20260828,
                        "variant": "H0_no_factor",
                    },
                )
            elif record_hash == ROOTBIND1_BENCHMARK_FAILURE["terminal_record_sha256"]:
                validate_rootbind1_failed_benchmark_terminal(
                    record, project_root=project_root
                )
                rootbind_failure.append(dict(record))
            elif context == dict(benchmark.BENCHMARK_USAGE_IDENTITY):
                _validate_execution_usage_record(
                    record,
                    expected_phase=benchmark.BENCHMARK_PHASE,
                    expected_identity=benchmark.BENCHMARK_USAGE_IDENTITY,
                )
                active_benchmark.append(dict(record))
            else:
                raise CampaignError("unowned efficiency-benchmark terminal in ledger")
            continue
        if _record_matches_exact_v8r3_quarantine(record):
            if record_hash != "8dbc0493125f22c130444e1344533d1f3d9c4ac445df6adfe8b597972e9691c5":
                raise CampaignError("legacy quarantine terminal hash drifted")
            _validate_execution_usage_record(
                record,
                expected_phase="discovery",
                expected_identity={
                    "outer_fold": 3,
                    "seed": 20260828,
                    "variant": "H0_no_factor",
                },
            )
            quarantine.append(dict(record))
            continue
        identity = {
            name: context.get(name)
            for name in (
                "campaign_revision",
                "infrastructure_revision",
                "outer_fold",
                "seed",
                "variant",
            )
        }
        key = (
            int(identity.get("outer_fold", -1)),
            int(identity.get("seed", -1)),
            str(identity.get("variant", "")),
        )
        if (
            phase != "discovery"
            or identity.get("campaign_revision") != CAMPAIGN_REVISION
            or identity.get("infrastructure_revision") != INFRASTRUCTURE_REVISION
            or key not in by_unit
            or set(context)
            != {
                "campaign_revision",
                "infrastructure_revision",
                "outer_fold",
                "seed",
                "variant",
                "execution_number",
                "resume",
            }
        ):
            raise CampaignError("unowned executable terminal in pack-free ledger")
        _validate_execution_usage_record(
            record,
            expected_phase="discovery",
            expected_identity=identity,
        )
        by_unit[key].append(dict(record))
    if historical_seen != historical_hashes:
        raise CampaignError("historical benchmark terminal whitelist is incomplete")
    if len(rootbind_failure) != 1:
        raise CampaignError("ROOTBIND1 benchmark failure ownership is not exact")
    if len(active_benchmark) != 1 or not _usage_record_succeeded(
        active_benchmark[0]
    ):
        raise CampaignError("active V8R4 benchmark is not exactly one success")
    if len(quarantine) != 1 or not _usage_record_succeeded(quarantine[0]):
        raise CampaignError("legacy V8R3 quarantine ownership is not exact")
    historical_positions = [positions[value] for value in historical_hash_order]
    quarantine_position = positions[str(quarantine[0]["record_sha256"])]
    rootbind_position = positions[str(rootbind_failure[0]["record_sha256"])]
    active_benchmark_tail = positions[str(active_benchmark[0]["record_sha256"])]
    if not (
        historical_positions == sorted(historical_positions)
        and max(historical_positions)
        < quarantine_position
        < rootbind_position
        < active_benchmark_tail
    ):
        raise CampaignError(
            "historical benchmark, quarantine, ROOTBIND1 failure, and CONTEXT1 benchmark ordering drifted"
        )
    discovery_positions: list[int] = []
    for key, matched in by_unit.items():
        if not matched or not _usage_record_succeeded(matched[-1]):
            raise CampaignError(f"discovery unit does not end in success: {key}")
        if any(_usage_record_succeeded(record) for record in matched[:-1]):
            raise CampaignError(f"discovery unit has an earlier reusable success: {key}")
        execution_numbers = [
            int(_execution_record_context(record).get("execution_number", -1))
            for record in matched
        ]
        if execution_numbers != list(range(len(matched))):
            raise CampaignError(f"discovery execution sequence is not contiguous: {key}")
        discovery_positions.extend(
            positions[str(record["record_sha256"])] for record in matched
        )
    if not discovery_positions or active_benchmark_tail >= min(discovery_positions):
        raise CampaignError("active benchmark did not precede every V8R4 discovery unit")
    return records, float(usage_state.settled_usage_ns) / 1_000_000_000.0


def build_discovery_completion(
    *,
    project_root: Path,
    run_root: Path,
    shard_seals: Sequence[tuple[Path, Mapping[str, Any]]],
    contract_binding: Mapping[str, Any],
    authorization: Mapping[str, Any],
    usage_ledger: Path,
    gpu_ledger: Path,
    gpu_lock: Path,
) -> dict[str, Any]:
    """Aggregate two immutable shard seals without opening either shard tree."""

    del gpu_ledger, gpu_lock
    if len(shard_seals) != 2:
        raise CampaignError("pack-free aggregation requires exactly two shard seals")
    observed: set[int] = set()
    normalized_shards: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    benchmark_binding: dict[str, Any] | None = None
    quarantine_binding: dict[str, Any] | None = None
    for seal_path, seal in shard_seals:
        outer = int(seal.get("outer_fold_shard", -1))
        seal_units = seal.get("units")
        canonical_seal_path = Path(
            os.path.abspath(
                project_root
                / DEFAULT_RUN_ROOT
                / "shards"
                / f"outer_{outer}"
                / DISCOVERY_SHARD_SEAL_NAME
            )
        )
        expected_index_identity = (
            Path(
                os.path.abspath(
                    project_root
                    / SHARD_TRAINING_INDEX.get(outer, Path("__invalid__"))
                )
            ),
            SHARD_TRAINING_INDEX_SHA256.get(outer),
            SHARD_TRAINING_INDEX_BYTES.get(outer),
        )
        projected_index_identity = projected_file_binding_identity(
            seal.get("training_index"), project_root=project_root
        )
        projected_benchmark_identity = projected_file_binding_identity(
            seal.get("pre_discovery_efficiency_benchmark"),
            project_root=project_root,
        )
        projected_quarantine_identity = projected_file_binding_identity(
            seal.get("v8r3_quarantine_owner"), project_root=project_root
        )
        if not (
            set(seal) == DISCOVERY_SHARD_SEAL_KEYS
            and seal.get("schema_version") == 1
            and outer in OUTER_RUNS
            and outer not in observed
            and seal_path.resolve() == canonical_seal_path
            and canonical_content_sha256(seal) == seal.get("content_sha256")
            and seal.get("classification") == DISCOVERY_SHARD_SEAL_CLASSIFICATION
            and seal.get("campaign_id") == CAMPAIGN_ID
            and seal.get("campaign_revision") == CAMPAIGN_REVISION
            and seal.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
            and seal.get("contract") == dict(contract_binding)
            and seal.get("pretrain_authorization")
            == dict(authorization["authorization_binding"])
            and seal.get("completed_units") == 9
            and seal.get("peer_outer_shard_pack_mounted_or_opened") is False
            and seal.get("combined_target_bearing_cache_opened") is False
            and seal.get("outer_prediction_pack_absent") is True
            and seal.get("physical_boundary") == DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION
            and projected_index_identity == expected_index_identity
            and projected_benchmark_identity is not None
            and projected_benchmark_identity[0]
            == Path(
                os.path.abspath(project_root / EFFICIENCY_BENCHMARK_RECEIPT_RELATIVE)
            )
            and projected_quarantine_identity
            == (
                Path(os.path.abspath(project_root / V8R3_QUARANTINE_RELATIVE)),
                V8R3_QUARANTINE_FILE_SHA256,
                V8R3_QUARANTINE_BYTES,
            )
            and isinstance(seal.get("gpu_usage_ledger_prefix"), Mapping)
            and seal.get("cross_outer_validation_reuse_present") is True
            and seal.get("fully_nested_confirmatory_oof") is False
            and seal.get("prospective_confirmation_required") is True
            and seal.get("ready_for_pack_free_shard_aggregation") is True
            and seal.get("commercial_claim_authorized") is False
            and isinstance(seal_units, list)
            and len(seal_units) == 9
        ):
            raise CampaignError("V8R4 discovery shard seal is unsafe or malformed")
        observed.add(outer)
        local_keys: set[tuple[int, int, str]] = set()
        for item in seal_units:
            if not isinstance(item, Mapping) or set(item) != DISCOVERY_SHARD_UNIT_KEYS:
                raise CampaignError("V8R4 shard unit binding is malformed")
            key = (int(item["outer_fold"]), int(item["seed"]), str(item["variant"]))
            if key[0] != outer or key in local_keys:
                raise CampaignError("V8R4 shard unit identity is duplicated or foreign")
            receipt = item.get("receipt")
            if not isinstance(receipt, Mapping) or set(receipt) != {"path", "sha256", "bytes"}:
                raise CampaignError("V8R4 shard unit receipt binding is malformed")
            local_keys.add(key)
            units.append(dict(item))
        expected_local = {
            (outer, seed, variant) for seed in SEEDS for variant in VARIANTS
        }
        if local_keys != expected_local:
            raise CampaignError("V8R4 shard unit identity cover drifted")
        current_benchmark = seal.get("pre_discovery_efficiency_benchmark")
        current_quarantine = seal.get("v8r3_quarantine_owner")
        if not isinstance(current_benchmark, Mapping) or not isinstance(
            current_quarantine, Mapping
        ):
            raise CampaignError("V8R4 shard owner bindings are absent")
        if benchmark_binding is None:
            benchmark_binding = dict(current_benchmark)
            quarantine_binding = dict(current_quarantine)
        elif benchmark_binding != dict(current_benchmark) or quarantine_binding != dict(
            current_quarantine
        ):
            raise CampaignError("the two shards bind different benchmark/quarantine owners")
        normalized_shards.append(
            {
                "outer_fold": outer,
                "seal": bind_file(seal_path),
                "training_index": dict(seal["training_index"]),
            }
        )
    if observed != set(OUTER_RUNS) or {
        (item["outer_fold"], item["seed"], item["variant"]) for item in units
    } != set(EXPECTED_DISCOVERY_UNITS):
        raise CampaignError("two shard seals do not exactly cover discovery")
    assert benchmark_binding is not None and quarantine_binding is not None
    try:
        with gpu_budget_ledger.locked_closed_snapshot(
            usage_ledger,
            budget_ns=gpu_budget_ledger.GPU_BUDGET_NS,
            expected_legacy_genesis_sha256=_expected_legacy_genesis(usage_ledger),
        ) as usage_state:
            _records, elapsed = _pack_free_discovery_usage_cover(
                project_root=project_root,
                usage_ledger=usage_ledger,
                shard_seals=shard_seals,
                usage_state=usage_state,
            )
            for seal_path, seal in shard_seals:
                prefix = seal.get("gpu_usage_ledger_prefix")
                if not isinstance(prefix, Mapping):
                    raise CampaignError("V8R4 shard seal lacks its GPU ledger prefix")
                verify_usage_ledger_prefix_binding(
                    usage_ledger,
                    prefix,
                    project_root=project_root,
                    owner=seal_path,
                    terminal_record_sha256=str(prefix.get("terminal_record_sha256", "")),
                    usage_state=usage_state,
                )
            ledger_binding = usage_snapshot_binding(usage_ledger, usage_state)
            return create_once_json(
                run_root / "DISCOVERY_COMPLETION_SEAL.json",
                {
                    "schema_version": 1,
                    "classification": "adaptive_v3r1_v8r4_target_sealed_discovery_completion",
                    "campaign_id": CAMPAIGN_ID,
                    "campaign_revision": CAMPAIGN_REVISION,
                    "infrastructure_revision": INFRASTRUCTURE_REVISION,
                    "contract": dict(contract_binding),
                    "pretrain_authorization": dict(authorization["authorization_binding"]),
                    "training_shards": sorted(normalized_shards, key=lambda item: item["outer_fold"]),
                    "outer_runs": list(OUTER_RUNS),
                    "seeds": list(SEEDS),
                    "variants": list(VARIANTS),
                    "completed_units": 18,
                    "physical_boundary": dict(DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
                    "validation_targets_only": True,
                    "gpu_elapsed_seconds": elapsed,
                    "gpu_hours_hard": GPU_HOURS_HARD,
                    "gpu_usage_ledger": ledger_binding,
                    "gpu_usage_ledger_path": _usage_ledger_path(usage_ledger),
                    "pre_discovery_efficiency_benchmark": {
                        "receipt": benchmark_binding,
                        "included_in_gpu_exact_cover": True,
                        "excluded_from_selection": True,
                        "artifacts_quarantined": True,
                    },
                    "v8r3_successful_terminal_quarantine": quarantine_binding,
                    "units": sorted(units, key=lambda item: (item["outer_fold"], item["seed"], item["variant"])),
                    "cross_outer_validation_reuse_present": True,
                    "fully_nested_confirmatory_oof": False,
                    "prospective_confirmation_required": True,
                    "ready_for_global_discovery_selection": True,
                    "commercial_claim_authorized": False,
                },
            )
    except (OSError, ValueError, RuntimeError) as error:
        if isinstance(error, CampaignError):
            raise
        raise CampaignError(
            f"cannot seal a stable closed GPU usage snapshot: {error}"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--outer-fold-shard", type=int, choices=OUTER_RUNS)
    mode.add_argument("--aggregate-shards", action="store_true")
    parser.add_argument("--training-index", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--target-sealed-capability-receipt", type=Path, required=True
    )
    parser.add_argument("--python", type=Path)
    parser.add_argument("--trainer", type=Path, default=TRAINER_RELATIVE)
    parser.add_argument("--gpu-wrapper", type=Path, default=GPU_WRAPPER_RELATIVE)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--gpu-ledger", type=Path, default=DEFAULT_GPU_LEDGER)
    parser.add_argument("--usage-ledger", type=Path, default=DEFAULT_USAGE_LEDGER)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def _under(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def require_canonical_gpu_lock(project_root: Path, gpu_lock: Path) -> Path:
    canonical = (project_root.expanduser().resolve() / DEFAULT_GPU_LOCK).resolve()
    actual = gpu_lock.expanduser().resolve()
    if actual != canonical:
        raise CampaignError(
            "GPU admission lock must be the fixed V8 campaign lock path under V8R4A"
        )
    return actual


def require_canonical_gpu_state_paths(
    project_root: Path,
    *,
    gpu_lock: Path,
    gpu_ledger: Path,
    usage_ledger: Path,
) -> tuple[Path, Path, Path]:
    expected = (
        (project_root / DEFAULT_GPU_LOCK).resolve(),
        (project_root / DEFAULT_GPU_LEDGER).resolve(),
        (project_root / DEFAULT_USAGE_LEDGER).resolve(),
    )
    observed = tuple(
        value.expanduser().resolve()
        for value in (gpu_lock, gpu_ledger, usage_ledger)
    )
    if observed != expected:
        raise CampaignError("GPU state must use the exact V8R4A canonical paths")
    return observed


def executable_path_without_symlink_dereference(root: Path, path: Path) -> Path:
    """Normalize an executable path while preserving a venv's final symlink.

    CPython discovers ``pyvenv.cfg`` from the invoked executable path.  Calling
    ``Path.resolve`` here would replace ``.venv/bin/python`` with the base uv
    interpreter and silently disable the project environment.
    """

    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return Path(os.path.abspath(candidate))


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # Canonical capability overrides fail before argparse's required shard
    # choice and therefore before any governance or pack artifact can open.
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    early.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    early_args, _ = early.parse_known_args(raw_argv)
    early_root = early_args.project_root.expanduser().resolve()
    try:
        require_canonical_gpu_lock(
            early_root, _under(early_root, early_args.gpu_lock)
        )
    except CampaignError as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}, sort_keys=True))
        return 2
    args = build_parser().parse_args(raw_argv)
    project_root = args.project_root.expanduser().resolve()
    try:
        # Resolve the sole parent/child admission-lock identity before any
        # invocation or output artifact can be created.
        gpu_lock = require_canonical_gpu_lock(
            project_root, _under(project_root, args.gpu_lock)
        )
        run_root = _under(project_root, args.run_root)
        expected_phase = (
            DISCOVERY_AGGREGATION_PHASE if args.aggregate_shards else "discovery"
        )
        expected_outer_fold = (
            None if args.aggregate_shards else int(args.outer_fold_shard)
        )
        authorization = validate_pretrain_authorization(
            project_root,
            capability_receipt=_under(
                project_root, args.target_sealed_capability_receipt
            ),
            expected_phase=expected_phase,
            expected_outer_fold=expected_outer_fold,
        )
        capability = authorization.get("target_sealed_capability")
        contract_binding = authorization.get("contract_binding")
        document = capability.get("document") if isinstance(capability, Mapping) else None
        writable = document.get("writable_roots") if isinstance(document, Mapping) else None
        output_binding = writable.get("output") if isinstance(writable, Mapping) else None
        canonical_output = (
            (project_root / AGGREGATION_OUTPUT_RELATIVE).resolve()
            if args.aggregate_shards
            else (
                project_root
                / DEFAULT_RUN_ROOT
                / "shards"
                / f"outer_{int(args.outer_fold_shard)}"
            ).resolve()
        )
        if not (
            isinstance(contract_binding, Mapping)
            and isinstance(output_binding, Mapping)
            and Path(str(output_binding.get("path", ""))).resolve()
            == run_root.resolve()
            == canonical_output
        ):
            raise CampaignError(
                "--run-root must be the canonical dedicated output directory in the runtime capability"
            )
        python = executable_path_without_symlink_dereference(
            project_root,
            args.python if args.python is not None else Path(".venv/bin/python"),
        )
        trainer = _under(project_root, args.trainer)
        wrapper = _under(project_root, args.gpu_wrapper)
        gpu_ledger = _under(project_root, args.gpu_ledger)
        usage_ledger = _under(project_root, args.usage_ledger)
        require_canonical_gpu_state_paths(
            project_root,
            gpu_lock=gpu_lock,
            gpu_ledger=gpu_ledger,
            usage_ledger=usage_ledger,
        )
        if not args.aggregate_shards:
            for required in (python, trainer, wrapper):
                if not required.is_file():
                    raise CampaignError(f"required executable/source is missing: {required}")
        if args.aggregate_shards:
            if args.training_index is not None:
                raise CampaignError("pack-free aggregation refuses a training-index path")
            shard_seals: list[tuple[Path, Mapping[str, Any]]] = []
            for outer_fold in OUTER_RUNS:
                role = f"discovery_shard_seal_outer{outer_fold}"
                path = _capability_bound_path(
                    project_root, capability, role
                )
                shard_seals.append(
                    (path, load_json(path, f"outer-{outer_fold} discovery shard seal"))
                )
            seal = build_discovery_completion(
                project_root=project_root,
                run_root=run_root,
                shard_seals=shard_seals,
                contract_binding=contract_binding,
                authorization=authorization,
                usage_ledger=usage_ledger,
                gpu_ledger=gpu_ledger,
                gpu_lock=gpu_lock,
            )
        else:
            outer_fold_shard = int(args.outer_fold_shard)
            benchmark_receipt_path = _capability_bound_path(
                project_root, capability, "benchmark_receipt"
            )
            validate_pre_discovery_efficiency_benchmark(
                project_root=project_root,
                receipt_path=benchmark_receipt_path,
                usage_ledger=usage_ledger,
                gpu_ledger=gpu_ledger,
                gpu_lock=gpu_lock,
                authorization=authorization,
                require_no_discovery_terminal=False,
            )
            # The exact quarantine owner is mounted as governance for this
            # shard.  It is never discovered by walking a legacy run tree.
            quarantine_path = _capability_bound_path(
                project_root, capability, "quarantine_owner_receipt"
            )
            if quarantine_path != (project_root / V8R3_QUARANTINE_RELATIVE).resolve():
                raise CampaignError("runtime mounted a non-canonical quarantine owner")
            validate_v8r3_quarantine_owner_receipt(
                project_root=project_root,
                usage_ledger=usage_ledger,
                gpu_ledger=gpu_ledger,
                gpu_lock=gpu_lock,
            )
            index_path = (
                _under(project_root, args.training_index)
                if args.training_index is not None
                else Path(
                    str(
                        document.get("sealed_pack_index", {}).get("path", "")
                    )
                ).resolve()
            )
            sealed_index = document.get("sealed_pack_index")
            if not isinstance(sealed_index, Mapping) or bind_file(index_path) != {
                key: sealed_index[key] for key in ("path", "sha256", "bytes")
            }:
                raise CampaignError("training index differs from the sealed runtime pack")
            training, training_index_binding = load_training_index(
                project_root,
                index_path,
                outer_fold_shard=outer_fold_shard,
            )
            shard_root = run_root
            receipts = []
            for seed in SEEDS:
                for variant in VARIANTS:
                    receipts.append(
                        run_training_unit(
                            project_root=project_root,
                            run_root=shard_root,
                            training_input=training[(outer_fold_shard, seed)],
                            variant=variant,
                            authorization=authorization,
                            contract_binding=contract_binding,
                            target_sealed_capability_receipt=_under(
                                project_root,
                                args.target_sealed_capability_receipt,
                            ),
                            python=python,
                            trainer=trainer,
                            wrapper=wrapper,
                            gpu_lock=gpu_lock,
                            gpu_ledger=gpu_ledger,
                            usage_ledger=usage_ledger,
                            device=args.device,
                            amp=not args.no_amp,
                            smoke_test=args.smoke_test,
                        )
                    )
            seal = build_discovery_shard_completion(
                project_root=project_root,
                shard_root=shard_root,
                outer_fold_shard=outer_fold_shard,
                receipts=receipts,
                contract_binding=contract_binding,
                authorization=authorization,
                training_index_binding=training_index_binding,
                usage_ledger=usage_ledger,
                gpu_ledger=gpu_ledger,
                gpu_lock=gpu_lock,
                benchmark_receipt_path=benchmark_receipt_path,
            )
    except BudgetExhausted as error:
        print(json.dumps({"status": "budget_exhausted_resumable", "error": str(error)}, sort_keys=True))
        return 75
    except CampaignError as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(seal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
