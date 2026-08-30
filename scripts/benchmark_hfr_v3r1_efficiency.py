#!/usr/bin/env python3
"""Run the single V8 two-epoch, no-accuracy efficiency benchmark.

The outer process owns V2 lifecycle reconciliation and the immutable completion
receipt.  The admitted worker consumes a one-shot wrapper capability before it
imports the trainer or opens any cache input, then persists timing telemetry
only.  No model, scaler, prediction, score, or checkpoint is reusable.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
CAMPAIGN_REVISION = "V8R4"
INFRASTRUCTURE_REVISION = "V8R4A"
BENCHMARK_ID = "v8_hfr_2epoch_no_accuracy_metric_efficiency"
BENCHMARK_UNIT = "outer_3_seed_20260828_H0_no_factor"
BENCHMARK_PHASE = "efficiency_benchmark"
AUTHORIZATION_GENERATION = "CONTEXT1"
BENCHMARK_USAGE_IDENTITY = {
    "campaign_revision": CAMPAIGN_REVISION,
    "infrastructure_revision": INFRASTRUCTURE_REVISION,
    "authorization_generation": AUTHORIZATION_GENERATION,
    "benchmark_id": BENCHMARK_ID,
    "outer_fold": 3,
    "seed": 20260828,
    "variant": "H0_no_factor",
}
HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "efficiency_benchmark_v8"
)
FROZEN_EXECUTION_CLOSURE_ACTIVE_OUTPUT_ROOT_RELATIVE = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "efficiency_benchmark_v8r4a"
)
BENCHMARK_RUN_ROOT_RELATIVE = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "efficiency_benchmark_v8r4a_context1"
)
BENCHMARK_RECEIPT_NAME = "BENCHMARK_COMPLETION_RECEIPT_V8R4.json"
BENCHMARK_RECEIPT_RELATIVE = (
    BENCHMARK_RUN_ROOT_RELATIVE / BENCHMARK_RECEIPT_NAME
)
BENCHMARK_LIFECYCLE_RELATIVE = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "target_sealed_lifecycle_v8r4a_context1/efficiency_benchmark/"
    "benchmark_hfr_v3r1_efficiency/outer_3"
)
CURRENT_UNIT_INVOCATION_NAME = "BENCHMARK_INVOCATION_V8R4.json"
LEGACY_UNIT_INVOCATION_NAME = "BENCHMARK_INVOCATION.json"
V8R2_UNIT_INVOCATION_NAME = "BENCHMARK_INVOCATION_V8R2.json"
DISCOVERY_SCRIPT_RELATIVE = Path("scripts/run_hfr_v3r1_discovery_campaign.py")
TRAINER_RELATIVE = Path("scripts/train_harmonic_factor_router_snn_v3r1.py")
WRAPPER_RELATIVE = Path("scripts/run_gpu_admitted.py")
DEFAULT_TRAINING_INDEX = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "v8r4_split_inputs/discovery_shard_outer_3/"
    "V8R4_NONOUTER_TRAINING_INDEX.json"
)
DEFAULT_TRAINING_INDEX_SHA256 = (
    "db204cdba72e9f3023c58ef37c1761cfd4ec2f4310449f8eaeeef7003afadb9b"
)
DEFAULT_TRAINING_INDEX_BYTES = 3_172
CONTRACT_RELATIVE = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json"
)
CONTRACT_FILE_SHA256 = (
    "532d150f0241d9675873368107d09adec7aeaee5e018e09537e8a340eb6fa2bd"
)
PRETRAIN_AUTHORIZATION_RELATIVE = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"
)
GPU_STATE_MIGRATION_RECEIPT_RELATIVE = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "GPU_STATE_MIGRATION_RECEIPT_V8R4A.json"
)
TARGET_SEALED_RUNTIME_RELATIVE = Path("scripts/run_hfr_v3r1_target_sealed.py")
TARGET_SEALED_CAPABILITY_NAME = "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json"
OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_RELATIVE = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_OPEN_LIFECYCLE_RECOVERY.json"
)
OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC_RELATIVE = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v8r4a_open_gpu_lifecycle_kill_recovery_failure.json"
)
EXECUTION_CLOSURE_AUTHORIZATION_RELATIVE = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_EXECUTION_CLOSURE.json"
)
EXECUTION_CLOSURE_DIAGNOSTIC_RELATIVE = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v8r4a_terminal_execution_closure_failure.json"
)
_RECOVERY_GOVERNANCE = {
    "open_lifecycle_recovery_correction_authorization": {
        "path": OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_RELATIVE,
        "sha256": "92b7e3e4b911dbf7450e3447b84f5a5762aee1212348c151cd57f20f10f5e1f6",
        "content_sha256": "b415258de20e65cea95f8b303500fd16e8834eff8eb2848a8792fd04c2c90381",
        "bytes": 8_481,
    },
    "open_lifecycle_recovery_failure_diagnostic": {
        "path": OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC_RELATIVE,
        "sha256": "a9cd41355ff98502153d61ad83d7f01da0d3c52462acdd24ade7bef73cf80b5e",
        "content_sha256": "24d8ed14fc68766a71d09a6fc263cc88aaa7038bb71052f7183328c9a6003016",
        "bytes": 4_067,
    },
    "execution_closure_correction_authorization": {
        "path": EXECUTION_CLOSURE_AUTHORIZATION_RELATIVE,
        "sha256": "11e5e7d00d3d837e0eb542b3482499cd21bf90b38883d0f4a0f86b68ba16d754",
        "content_sha256": "92d96a4f513a7d7f93bbd4baf227b626106dab54e000f3a01c97b25504c58c1c",
        "bytes": 21_621,
    },
    "execution_closure_failure_diagnostic": {
        "path": EXECUTION_CLOSURE_DIAGNOSTIC_RELATIVE,
        "sha256": "0ca492c98f94e73e21873c41287c24d3d135466c4ca4d085388f9c39e5d9560e",
        "content_sha256": "c5dbe569ebcf3b720b06089a6b3c5d8eed4d00a542815edfe25ecb8a6751b774",
        "bytes": 8_498,
    },
}
LEGACY_BENCHMARK_RECEIPT_RELATIVE = (
    HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE / "BENCHMARK_COMPLETION_RECEIPT.json"
)
DEFAULT_GPU_LOCK = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "gpu_state_v8r4a/admission/gpu_admission_v7.lock"
)
DEFAULT_EXECUTION_LEDGER = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "gpu_state_v8r4a/execution/gpu_execution_ledger_v7.jsonl"
)
DEFAULT_USAGE_LEDGER = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "gpu_state_v8r4a/usage/campaign_gpu_usage_chain_v6.jsonl"
)
EPOCHS = 2
OPTIMIZER_STEPS_PER_EPOCH = 5
# Exact processed row/window covers for the immutable outer-3 benchmark
# split.  The old 216/54 values were pre-batching temporal forward-chunk
# counts, not row covers; changing these expectations does not change work.
TRAINING_WINDOWS_PER_EPOCH = 6_583
VALIDATION_WINDOWS_PER_EPOCH = 1_658
STEADY_GATE_NS = 23_000_000_000
GATE_FAILURE_EXIT = 86

# This is the first pre-V8R3 benchmark lifecycle admitted transitively by the
# immutable V8R3 correction.  It is intentionally a closed exact whitelist,
# not a pattern.
LEGACY_V8_ATTEMPT = {
    "authority": "V8",
    "attempt_index": 0,
    "unit_invocation": {
        "relative_path": (
            HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE / LEGACY_UNIT_INVOCATION_NAME
        ).as_posix(),
        "sha256": "202958d02a8280c89bb1561e3f003c7cbf8cf05539820f11f94cf864e0ba63a3",
        "bytes": 5_344,
        "content_sha256": "52f3c1cd0a7004f7537e65c102a3b7649abc33d1101cb02a26681966b1a24152",
    },
    "execution_invocation": {
        "relative_path": (
            HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE
            / "attempts/attempt_000/invocation.json"
        ).as_posix(),
        "sha256": "cd9727c64de434a0946adb99e6a3dff3005282ee811350eb5102b91b5a5bac7b",
        "bytes": 2_557,
        "content_sha256": "a9e5f7969778853e1064c7394844edbe9fd1dbf848ef58517c4254f17c9c0049",
    },
    "terminal_result": {
        "relative_path": (
            HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE
            / "attempts/attempt_000/GPU_TERMINAL_RESULT.json"
        ).as_posix(),
        "sha256": "7aea7714e9f5f248254a0052442e632d946930ff8b45ee1f28a39d653fdf41be",
        "bytes": 1_699,
        "content_sha256": "b4a5809cbfbbd0f153867034f4ea36b4cb110b3110d6722c08177cbb36acb6b4",
        "terminal_record_sha256": "d7d3e26a5ba19d648343d19ed4593e503c1017b02f70e8c27517b1d1af0d4fbf",
        "return_code": 1,
        "charged_usage_ns": 1_023_036_848,
        "reusable_success": False,
    },
    "pretrain_authorization": {
        "relative_path": (
            "artifacts/campaigns/"
            "directed_harmonic_factor_expert_snn_v3r1/"
            "PRETRAIN_AUTHORIZATION_V8.json"
        ),
        "sha256": "91e949cade57da4555f522de378cabcb5f99b1515bf1f87d6e9d26606acedb2b",
        "bytes": 4_115,
    },
}

FAILED_V8R2_ATTEMPT = {
    "authority": "V8R2",
    "attempt_index": 1,
    "unit_invocation": {
        "relative_path": (
            HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE / V8R2_UNIT_INVOCATION_NAME
        ).as_posix(),
        "sha256": "f5451c96f5985769af56adf63107d157717ed97275cc9f93a158e951a33ac1cc",
        "bytes": 5_346,
        "content_sha256": "bcd1831b2978e47b18f311041100ae7686cf64addea2b0161dd6a3d678e894a1",
    },
    "execution_invocation": {
        "relative_path": (
            HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE
            / "attempts/attempt_001/invocation.json"
        ).as_posix(),
        "sha256": "fdca8a8488f0e302aaa65054870a82f0557029bcd576130439596a3c09761cb1",
        "bytes": 2_564,
        "content_sha256": "2a4171ef1fb4b7bac52f1bf1f750fc410069af42b6175ee57ae80c24e4aa0fb2",
    },
    "terminal_result": {
        "relative_path": (
            HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE
            / "attempts/attempt_001/GPU_TERMINAL_RESULT.json"
        ).as_posix(),
        "sha256": "fccf0bdea0ab17f8f6da33acc7ab4ebf9e0f28301a4a4ea6b4b9d8fa245e70c2",
        "bytes": 1_703,
        "content_sha256": "abfa054fa41a1502addcac41cb76b49281d83b89b97f3af6ce138fa74fc756f8",
        "terminal_record_sha256": "ca89b52ef526fb68854879207c0e56b8dc79a1ee0e31f2b0f143a7a2f43741e2",
        "return_code": 87,
        "charged_usage_ns": 42_457_966_370,
        "reusable_success": False,
    },
    "pretrain_authorization": {
        "relative_path": (
            "artifacts/campaigns/"
            "directed_harmonic_factor_expert_snn_v3r1/"
            "PRETRAIN_AUTHORIZATION_V8R2.json"
        ),
        "sha256": "4918d6b4396bca694db43deb01d26cdfca43286465e811fd8695c258ab917ded",
        "bytes": 4_122,
    },
}

# The successful V8R3 terminal is immutable historical evidence, but it was
# produced from the forbidden combined cache and therefore is never reusable
# by V8R4.  Its unusual 0644 terminal mode is pinned as an observed historical
# defect; V8R4A projects it read-only from frozen governance and runs its sole
# target-sealed child in a separate clean output root.
SUCCESSFUL_V8R3_ATTEMPT = {
    "authority": "V8R3",
    "attempt_index": 2,
    "unit_invocation": {
        "relative_path": (
            HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE / "BENCHMARK_INVOCATION_V8R3.json"
        ).as_posix(),
        "sha256": "b0c5e985441c3cd8a4a335cf91a9a840207673a6ce66a99603e813e383cabb6b",
        "bytes": 5_349,
        "content_sha256": "0ca82efba537d7ed0ae1347d3b8a18b3c1694ea46dc27bcedd990d7d9bec1e2b",
        "mode": 0o444,
    },
    "execution_invocation": {
        "relative_path": (
            HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE / "attempts/attempt_002/invocation.json"
        ).as_posix(),
        "sha256": "2c9fc326d2304c4fb690316f5bcdfc4503505f4026c8150cd7f71003c086cd08",
        "bytes": 2_564,
        "content_sha256": "cbbdcb371e228023007c68acf35d18af06e3facb739b5cb64aee840825a4f707",
        "mode": 0o444,
    },
    "terminal_result": {
        "relative_path": (
            HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE
            / "attempts/attempt_002/GPU_TERMINAL_RESULT.json"
        ).as_posix(),
        "sha256": "0b12a5420441faa69555f50890d4618a16e4de24fca76bd0d1a5797cc0656338",
        "bytes": 1_700,
        "content_sha256": "294f81db056d12ed202ef489cc72f57c426d7565ab1344738c176c130ec1cdf6",
        "terminal_record_sha256": "aaacf4a93f96f5308eb4d31ceceac4de27b39ddd95f9957ee37ab1b42793289b",
        "return_code": 0,
        "charged_usage_ns": 44_131_225_055,
        "reusable_success": True,
        "mode": 0o644,
    },
    "telemetry": {
        "relative_path": (
            HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE
            / "attempts/attempt_002/QUARANTINED_TIMING_TELEMETRY.json"
        ).as_posix(),
        "sha256": "b1e36d03d98c5516512dcc349812707a851a719b3b9a3b6d17d45ccacce8660c",
        "bytes": 4_763,
        "content_sha256": "4c6361088816bb1614aa249617bee81b4708a4cc55bafbfb88eb10cdcaa3cebb",
        "mode": 0o444,
    },
    "pretrain_authorization": {
        "relative_path": (
            "artifacts/campaigns/"
            "directed_harmonic_factor_expert_snn_v3r1/"
            "PRETRAIN_AUTHORIZATION_V8R3.json"
        ),
        "sha256": "26ef02cf9f5abb8ec44ed4f82c0f3e738a46f4f5a5e1719ef94a087aee2bd10f",
        "bytes": 4_124,
    },
}

HISTORICAL_BENCHMARK_ATTEMPTS = (
    LEGACY_V8_ATTEMPT,
    FAILED_V8R2_ATTEMPT,
    SUCCESSFUL_V8R3_ATTEMPT,
)

LEGACY_BENCHMARK_USAGE_IDENTITY = {
    "benchmark_id": BENCHMARK_ID,
    "outer_fold": 3,
    "seed": 20260828,
    "variant": "H0_no_factor",
}

# The ROOTBIND1 child was admitted with the pre-generation six-field context
# and failed after the CUDA probe, before model construction.  It remains an
# immutable, charged infrastructure prefix.  CONTEXT1 never treats it as an
# active completion and never retries that context.
ROOTBIND1_BENCHMARK_USAGE_IDENTITY = {
    "campaign_revision": CAMPAIGN_REVISION,
    "infrastructure_revision": INFRASTRUCTURE_REVISION,
    "benchmark_id": BENCHMARK_ID,
    "outer_fold": 3,
    "seed": 20260828,
    "variant": "H0_no_factor",
}
ROOTBIND1_FAILED_BENCHMARK_TERMINAL = {
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

BENCHMARK_PROFILE = {
    "schema_version": 1,
    "benchmark_id": BENCHMARK_ID,
    "unit": BENCHMARK_UNIT,
    "outer_fold": 3,
    "seed": 20260828,
    "variant": "H0_no_factor",
    "epochs": EPOCHS,
    "epoch_1_is_warmup": True,
    "epoch_2_train_plus_target_free_validation_ns_max": STEADY_GATE_NS,
    "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
    "training_windows_per_epoch": TRAINING_WINDOWS_PER_EPOCH,
    "target_free_validation_windows_per_epoch": VALIDATION_WINDOWS_PER_EPOCH,
    "learning_rate": "0.0003",
    "weight_decay": "0.0001",
    "chunk_windows": 32,
    "warmup_windows": 2,
    "gradient_accumulation_sessions": 4,
    "gradient_clip": "2.0",
    "deterministic": True,
    "amp": True,
    "device": "cuda",
    "accuracy_metrics_allowed": False,
    "checkpoint_selection_allowed": False,
    "reusable_output_allowed": False,
}


class BenchmarkError(RuntimeError):
    """The V8 benchmark lifecycle or telemetry failed closed."""


class BenchmarkGateFailed(BenchmarkError):
    """The authorized 23-second steady epoch gate was not met."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BenchmarkError(f"non-canonical benchmark value: {error}") from error


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


BENCHMARK_PROFILE_SHA256 = semantic_sha256(BENCHMARK_PROFILE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise BenchmarkError(f"cannot hash benchmark file {path}: {error}") from error
    return digest.hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BenchmarkError(f"duplicate benchmark JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                BenchmarkError(f"non-finite {label} value: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"invalid {label}: {path} ({error})") from error
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label} must be a JSON object")
    return value


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    document = dict(value)
    document.pop("content_sha256", None)
    return semantic_sha256(document)


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
        raise BenchmarkError("benchmark staged-directory rename scope drifted")
    parent_fd = os.open(
        staging.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as error:
            raise BenchmarkError("renameat2 is unavailable for indexed publication") from error
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
            raise BenchmarkError(
                f"cannot publish benchmark indexed directory without replacement: "
                f"{final} ({os.strerror(error_number)})"
            )
    finally:
        os.close(parent_fd)


def _read_exact_immutable(path: Path, *, expected: bytes, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BenchmarkError(f"cannot open {label}: {path} ({error})") from error
    try:
        before = os.fstat(descriptor)
        pieces: list[bytes] = []
        while True:
            piece = os.read(descriptor, 1 << 20)
            if not piece:
                break
            pieces.append(piece)
        raw = b"".join(pieces)
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
        raise BenchmarkError(f"{label} is not exact immutable bytes: {path}")
    return raw


def _anonymous_create_once(path: Path, raw: bytes) -> None:
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
            raise BenchmarkError(
                f"benchmark filesystem cannot create an anonymous artifact: {path} ({error})"
            ) from error
        _publication_fault("anonymous_opened", path)
        view = memoryview(raw)
        while view:
            written = os.write(anonymous_fd, view)
            if written <= 0:
                raise BenchmarkError(f"short anonymous benchmark write: {path}")
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
            raise BenchmarkError(f"anonymous benchmark pre-link state drifted: {path}")
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
            anonymous_fd, b"", directory_fd, os.fsencode(path.name), _AT_EMPTY_PATH
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == 17:
                try:
                    _read_exact_immutable(path, expected=raw, label="benchmark artifact")
                except BenchmarkError as error:
                    raise BenchmarkError(
                        f"immutable benchmark artifact collision: {path}"
                    ) from error
                os.fsync(directory_fd)
                return
            raise BenchmarkError(
                f"cannot link anonymous benchmark artifact: {path} "
                f"({os.strerror(error_number)})"
            )
        _publication_fault("linked", path)
        status = os.fstat(anonymous_fd)
        if status.st_nlink != 1 or stat.S_IMODE(status.st_mode) != 0o444:
            raise BenchmarkError(f"linked benchmark inode state drifted: {path}")
        os.fsync(directory_fd)
        _publication_fault("directory_fsynced", path)
    finally:
        if anonymous_fd >= 0:
            os.close(anonymous_fd)
        os.close(directory_fd)
    _read_exact_immutable(path, expected=raw, label="benchmark artifact")


def create_once_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Publish one complete exact JSON document from an anonymous inode."""

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
    _anonymous_create_once(path, raw)
    existing = load_json(path, "benchmark artifact")
    if existing != document or canonical_content_sha256(existing) != existing.get(
        "content_sha256"
    ):
        raise BenchmarkError(f"benchmark artifact content hash drifted: {path}")
    return document


def _load_script(name: str, path: Path) -> Any:
    if name in sys.modules:
        raise BenchmarkError(f"benchmark module name is already registered: {name}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise BenchmarkError(f"cannot import required benchmark module: {path}")
    module = importlib.util.module_from_spec(specification)
    # Python 3.12 dataclass processing resolves forward annotations through
    # sys.modules[cls.__module__].  Register before exec_module exactly as the
    # ordinary import machinery does; otherwise every dataclass-bearing frozen
    # campaign/trainer module fails before the benchmark primitive can start.
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
        raise
    if sys.modules.get(name) is not module:
        raise BenchmarkError("benchmark module registration changed during import")
    return module


_CAPABILITY_FILE_BINDING_KEYS = frozenset(
    {"path", "sha256", "bytes", "st_dev", "st_ino", "mode"}
)


def _revalidate_capability_file(
    *, project_root: Path, row: Mapping[str, Any], label: str
) -> tuple[Path, bytes]:
    if not (
        set(row) == _CAPABILITY_FILE_BINDING_KEYS
        and isinstance(row.get("path"), str)
        and Path(str(row["path"])).is_absolute()
        and isinstance(row.get("sha256"), str)
        and len(str(row["sha256"])) == 64
        and type(row.get("bytes")) is int
        and row["bytes"] >= 0
        and type(row.get("st_dev")) is int
        and row["st_dev"] >= 0
        and type(row.get("st_ino")) is int
        and row["st_ino"] > 0
        and row.get("mode") == "0444"
    ):
        raise BenchmarkError(f"target-sealed {label} binding schema drifted")
    path = Path(str(row["path"]))
    try:
        path.relative_to(project_root.resolve())
    except ValueError as error:
        raise BenchmarkError(f"target-sealed {label} escapes the project root") from error
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BenchmarkError(f"target-sealed {label} is unavailable: {error}") from error
    try:
        before = os.fstat(descriptor)
        parts: list[bytes] = []
        while True:
            part = os.read(descriptor, 1024 * 1024)
            if not part:
                break
            parts.append(part)
        raw = b"".join(parts)
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
        raise BenchmarkError(f"target-sealed {label} changed after issuance")
    return path, raw


def validate_target_sealed_capability(
    *,
    project_root: Path,
    capability_receipt: Path,
    runtime_module: Any | None = None,
) -> dict[str, Any]:
    """Live-replay the V8R4 benchmark capability before any pack is opened."""

    root = project_root.expanduser().resolve()
    path = capability_receipt.expanduser()
    if not path.is_absolute():
        path = root / path
    path = Path(os.path.abspath(path))
    canonical_output = (root / BENCHMARK_RUN_ROOT_RELATIVE).resolve()
    canonical_lifecycle = (root / BENCHMARK_LIFECYCLE_RELATIVE).resolve()
    if path.name != TARGET_SEALED_CAPABILITY_NAME:
        raise BenchmarkError("benchmark target-sealed capability path is non-canonical")
    runtime = runtime_module or _load_script(
        "hfr_v8r4_target_sealed_for_benchmark",
        root / TARGET_SEALED_RUNTIME_RELATIVE,
    )
    validator = getattr(runtime, "validate_capability_receipt", None)
    if not callable(validator):
        raise BenchmarkError("target-sealed runtime capability validator is unavailable")
    try:
        validated = validator(
            path,
            expected_phase=BENCHMARK_PHASE,
            expected_outer_fold=3,
        )
    except Exception as error:
        raise BenchmarkError(f"target-sealed benchmark capability rejected: {error}") from error
    if not isinstance(validated, Mapping) or set(validated) != {"document", "binding"}:
        raise BenchmarkError("target-sealed validator result schema drifted")
    document = validated.get("document")
    binding = validated.get("binding")
    if not isinstance(document, Mapping) or not isinstance(binding, Mapping):
        raise BenchmarkError("target-sealed validator omitted its exact document/binding")
    boundary = document.get("security_boundary")
    writable = document.get("writable_roots")
    governance = document.get("governance_files")
    lifecycle = writable.get("lifecycle") if isinstance(writable, Mapping) else None
    output = writable.get("output") if isinstance(writable, Mapping) else None
    common_safe = (
        document.get("classification")
        == "adaptive_v3r1_v8r4a_outer_target_sealed_runtime_capability_receipt"
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == CAMPAIGN_REVISION
        and document.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and document.get("phase") == BENCHMARK_PHASE
        and document.get("outer_fold") == 3
        and isinstance(boundary, Mapping)
        and isinstance(lifecycle, Mapping)
        and isinstance(output, Mapping)
        and Path(str(lifecycle.get("path", ""))).resolve() == path.parent.resolve()
        and path.parent.resolve() == canonical_lifecycle
        and Path(str(output.get("path", ""))).resolve() == canonical_output
        and Path(str(lifecycle.get("path", ""))).resolve() != canonical_output
        and isinstance(governance, Mapping)
        and {
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
        <= set(governance)
        and all(isinstance(row, Mapping) for row in governance.values())
        and boundary.get("target_reference_or_selection_evidence_accessed") is False
        and boundary.get("legacy_combined_cache_mounted") is False
        and boundary.get("raw_or_target_root_mounted") is False
        and boundary.get("cross_outer_shard_mounted") is False
        and boundary.get("other_pack_or_output_mounted") is False
    )
    v8r4a_runtime = (
        type(boundary.get("synthetic_validation_only")) is bool
        and type(boundary.get("production_execution_authorized")) is bool
        and boundary.get("synthetic_validation_only")
        is (not boundary.get("production_execution_authorized"))
        and boundary.get("atomic_replace_compatible") is True
        and boundary.get("v8r4a_ledger_migration_required") is False
        and boundary.get("v8r4a_migration_live_replay_validated") is True
        and boundary.get("dedicated_gpu_state_directory_capabilities") is True
        and boundary.get("exactly_three_mutable_state_directory_mounts") is True
        and boundary.get("usage_and_execution_closed_prelaunch") is True
        and boundary.get("lifecycle_mounted_read_only") is True
        and boundary.get("source_snapshot_exact_file_mounts") is True
        and boundary.get("complete_project_source_or_config_trees_mounted") is False
    )
    if not (common_safe and v8r4a_runtime):
        raise BenchmarkError("target-sealed benchmark security boundary drifted")
    if not isinstance(binding, Mapping):
        raise BenchmarkError("target-sealed benchmark receipt binding is absent")
    rebound, _raw = _revalidate_capability_file(
        project_root=root, row=binding, label="benchmark capability receipt"
    )
    if rebound != path:
        raise BenchmarkError("target-sealed benchmark capability path rebound")
    for role, expected in _RECOVERY_GOVERNANCE.items():
        row = governance.get(role)
        if not isinstance(row, Mapping):
            raise BenchmarkError(f"target-sealed benchmark lacks {role}")
        material_path, raw = _revalidate_capability_file(
            project_root=root, row=row, label=role
        )
        try:
            relative = material_path.relative_to(root).as_posix()
            material = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError, json.JSONDecodeError) as error:
            raise BenchmarkError(f"target-sealed benchmark {role} is invalid: {error}") from error
        if not (
            relative == Path(expected["path"]).as_posix()
            and row.get("sha256") == expected["sha256"]
            and row.get("bytes") == expected["bytes"]
            and isinstance(material, Mapping)
            and material.get("content_sha256") == expected["content_sha256"]
        ):
            raise BenchmarkError(f"target-sealed benchmark {role} exact binding drifted")
    return {"document": dict(document), "binding": dict(binding)}


def require_v8r4a_benchmark_runtime(
    *, project_root: Path, run_root: Path, discovery_module: Any | None = None
) -> dict[str, Any] | None:
    """Replay the migration capability for the canonical production lineage."""

    canonical = (project_root / BENCHMARK_RUN_ROOT_RELATIVE).resolve()
    if run_root.resolve() != canonical:
        return None
    discovery = discovery_module or _load_script(
        "hfr_v8r4a_discovery_for_benchmark_migration",
        project_root / DISCOVERY_SCRIPT_RELATIVE,
    )
    validator = getattr(discovery, "validate_v8r4a_gpu_state", None)
    if not callable(validator):
        raise BenchmarkError("V8R4A migration validator is unavailable")
    try:
        validated = validator(project_root)
    except Exception as error:
        raise BenchmarkError(f"V8R4A benchmark migration rejected: {error}") from error
    if not isinstance(validated, Mapping) or set(validated) != {
        "migration_receipt", "canonical_paths"
    }:
        raise BenchmarkError("V8R4A benchmark migration result drifted")
    paths = validated.get("canonical_paths")
    if not isinstance(paths, Mapping) or not (
        paths.get("admission_lock") == (project_root / DEFAULT_GPU_LOCK).resolve()
        and paths.get("execution_ledger")
        == (project_root / DEFAULT_EXECUTION_LEDGER).resolve()
        and paths.get("usage_ledger") == (project_root / DEFAULT_USAGE_LEDGER).resolve()
    ):
        raise BenchmarkError("V8R4A benchmark state paths drifted")
    return dict(validated)


def _require_exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise BenchmarkError(f"{label} must be an integer >= {minimum}")
    return int(value)


_TRAINER_TELEMETRY_KEYS = frozenset(
    {
        "invocation_sha256",
        "epochs_completed",
        "epochs",
        "optimizer_steps",
        "training_windows",
        "validation_windows",
        "peak_cuda_memory_bytes",
    }
)
_EPOCH_TELEMETRY_KEYS = frozenset(
    {
        "epoch",
        "train_ns",
        "validation_ns",
        "total_ns",
        "optimizer_steps",
        "training_windows",
        "validation_windows",
        "warmup",
    }
)


def validate_trainer_telemetry(
    value: Mapping[str, Any], *, invocation_sha256: str
) -> dict[str, Any]:
    """Accept timing/count telemetry only; scores and reusable artifacts fail."""

    telemetry = dict(value)
    if set(telemetry) != _TRAINER_TELEMETRY_KEYS:
        raise BenchmarkError("trainer benchmark telemetry key set drifted")
    if telemetry.get("invocation_sha256") != invocation_sha256:
        raise BenchmarkError("trainer benchmark invocation binding drifted")
    if _require_exact_int(
        telemetry.get("epochs_completed"), "completed benchmark epochs", minimum=1
    ) != EPOCHS:
        raise BenchmarkError("benchmark did not complete exactly two epochs")
    epochs = telemetry.get("epochs")
    if not isinstance(epochs, list) or len(epochs) != EPOCHS:
        raise BenchmarkError("benchmark telemetry must contain exactly two epochs")
    normalized_epochs: list[dict[str, int]] = []
    for expected_epoch, raw_epoch in enumerate(epochs, 1):
        if not isinstance(raw_epoch, Mapping) or set(raw_epoch) != _EPOCH_TELEMETRY_KEYS:
            raise BenchmarkError("benchmark epoch timing schema drifted")
        epoch = {
            key: _require_exact_int(
                raw_epoch.get(key), f"benchmark epoch {expected_epoch} {key}", minimum=1
            )
            for key in _EPOCH_TELEMETRY_KEYS
            if key != "warmup"
        }
        if type(raw_epoch.get("warmup")) is not bool or raw_epoch.get(
            "warmup"
        ) is not (expected_epoch == 1):
            raise BenchmarkError("benchmark warmup/steady epoch designation drifted")
        epoch["warmup"] = bool(raw_epoch["warmup"])
        if epoch["epoch"] != expected_epoch:
            raise BenchmarkError("benchmark epoch ordering drifted")
        if epoch["total_ns"] != epoch["train_ns"] + epoch["validation_ns"]:
            raise BenchmarkError("benchmark epoch timing does not add exactly")
        if epoch["optimizer_steps"] != OPTIMIZER_STEPS_PER_EPOCH:
            raise BenchmarkError("benchmark optimizer-step schedule drifted")
        if epoch["training_windows"] != TRAINING_WINDOWS_PER_EPOCH:
            raise BenchmarkError("benchmark training window cover drifted")
        if epoch["validation_windows"] != VALIDATION_WINDOWS_PER_EPOCH:
            raise BenchmarkError("benchmark target-free validation cover drifted")
        normalized_epochs.append(epoch)
    totals = {
        "optimizer_steps": OPTIMIZER_STEPS_PER_EPOCH * EPOCHS,
        "training_windows": TRAINING_WINDOWS_PER_EPOCH * EPOCHS,
        "validation_windows": VALIDATION_WINDOWS_PER_EPOCH * EPOCHS,
    }
    for field, expected in totals.items():
        if _require_exact_int(telemetry.get(field), field, minimum=1) != expected:
            raise BenchmarkError(f"benchmark aggregate {field} drifted")
    _require_exact_int(
        telemetry.get("peak_cuda_memory_bytes"),
        "benchmark peak CUDA memory",
        minimum=1,
    )
    for key in telemetry:
        lowered = key.lower()
        if any(token in lowered for token in ("accuracy", "checkpoint", "selection", "score")):
            raise BenchmarkError("benchmark trainer emitted forbidden telemetry")
    telemetry["epochs"] = normalized_epochs
    canonical_json_bytes(telemetry)
    return telemetry


def _worker_trainer_arguments(
    trainer: Any,
    *,
    cache: Path,
    proposer_stack: Path,
    forbidden_output_dir: Path,
    target_sealed_capability_receipt: Path,
) -> argparse.Namespace:
    return trainer.parse_args(
        [
            "--mode",
            "efficiency_benchmark",
            "--campaign-phase",
            "discovery",
            "--cache",
            str(cache),
            "--proposer-stack",
            str(proposer_stack),
            "--output-dir",
            str(forbidden_output_dir),
            "--target-sealed-capability-receipt",
            str(target_sealed_capability_receipt),
            "--expected-admitted-context-json",
            json.dumps(
                BENCHMARK_USAGE_IDENTITY,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "--outer-fold",
            "3",
            "--seed",
            "20260828",
            "--variant",
            "H0_no_factor",
            "--device",
            "cuda",
            "--amp",
            "--deterministic",
            "--epochs",
            "2",
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
            "--gradient-clip",
            "2.0",
        ]
    )


def run_internal_worker(
    args: argparse.Namespace,
    *,
    admitted_module: Any | None = None,
    trainer_module: Any | None = None,
    runtime_module: Any | None = None,
) -> int:
    """Consume admission first, run two epochs, and publish timing only."""

    required = (
        args.project_root,
        args.trainer,
        args.cache,
        args.proposer_stack,
        args.telemetry_output,
        args.trainer_output_dir,
        args.usage_ledger,
        args.execution_ledger,
        args.authorization_path,
        args.authorization_sha256,
        args.target_sealed_capability_receipt,
    )
    if any(value is None for value in required):
        raise BenchmarkError("internal benchmark worker arguments are incomplete")
    project_root = args.project_root.expanduser().resolve()
    usage_ledger = _under(project_root, args.usage_ledger)
    execution_ledger = _under(project_root, args.execution_ledger)
    authorization_path = _under(project_root, args.authorization_path)

    # This live replay is deliberately before trainer import and before either
    # the cache manifest or proposer archive can be opened.
    target_capability_path = _under(
        project_root, args.target_sealed_capability_receipt
    )
    validate_target_sealed_capability(
        project_root=project_root,
        capability_receipt=target_capability_path,
        runtime_module=runtime_module,
    )

    admitted = admitted_module or _load_script(
        "hfr_v8_gpu_admitted_child", project_root / WRAPPER_RELATIVE
    )
    binding = admitted.consume_admitted_child_binding(
        BENCHMARK_PHASE,
        usage_ledger,
        execution_ledger,
        authorization_path,
        str(args.authorization_sha256),
        expected_campaign_id=CAMPAIGN_ID,
        expected_gpu_lock_file=_under(project_root, DEFAULT_GPU_LOCK),
    )
    if not (
        isinstance(binding, Mapping)
        and binding.get("valid") is True
        and binding.get("classification")
        == "verified_v8_gpu_admitted_child_lifecycle"
        and binding.get("phase") == BENCHMARK_PHASE
        and binding.get("context") == BENCHMARK_USAGE_IDENTITY
    ):
        raise BenchmarkError("benchmark wrapper binding scope drifted")
    trainer_path = _under(project_root, args.trainer)
    cache = _under(project_root, args.cache)
    proposer_stack = _under(project_root, args.proposer_stack)
    telemetry_output = _under(project_root, args.telemetry_output)
    forbidden_output = _under(project_root, args.trainer_output_dir)
    if os.path.lexists(forbidden_output):
        raise BenchmarkError("benchmark reusable trainer output path already exists")
    trainer = trainer_module or _load_script("hfr_v8_benchmark_trainer", trainer_path)
    trainer_args = _worker_trainer_arguments(
        trainer,
        cache=cache,
        proposer_stack=proposer_stack,
        forbidden_output_dir=forbidden_output,
        target_sealed_capability_receipt=_under(
            project_root, args.target_sealed_capability_receipt
        ),
    )
    # The benchmark worker, rather than the primitive, owns the independent
    # phase/context constants.  Validate the already-consumed binding against
    # the exact target capability before any scientific primitive can run.
    pretrain = trainer.validate_pretrain_authorization(
        dict(binding),
        target_sealed_capability_receipt=target_capability_path,
        expected_phase=BENCHMARK_PHASE,
        expected_context=dict(BENCHMARK_USAGE_IDENTITY),
        expected_outer_fold=3,
    )
    telemetry = validate_trainer_telemetry(
        trainer.run_efficiency_benchmark(
            trainer_args,
            admitted_binding=dict(binding),
            pretrain=pretrain,
        ),
        invocation_sha256=str(binding["invocation_sha256"]),
    )
    if os.path.lexists(forbidden_output):
        raise BenchmarkError("benchmark trainer emitted a reusable output tree")
    steady = telemetry["epochs"][1]
    steady_ns = int(steady["total_ns"])
    gate_passed = steady_ns <= STEADY_GATE_NS
    create_once_json(
        telemetry_output,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_v8r4_quarantined_efficiency_telemetry",
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "phase": BENCHMARK_PHASE,
            "benchmark_id": BENCHMARK_ID,
            "unit": BENCHMARK_UNIT,
            "usage_identity": dict(BENCHMARK_USAGE_IDENTITY),
            "profile_sha256": BENCHMARK_PROFILE_SHA256,
            "epochs": EPOCHS,
            "epoch_1_is_warmup": True,
            "epoch_2_train_ns": int(steady["train_ns"]),
            "epoch_2_target_free_validation_ns": int(steady["validation_ns"]),
            "epoch_2_train_plus_target_free_validation_ns": steady_ns,
            "epoch_2_gate_ns_max": STEADY_GATE_NS,
            "gate_passed": gate_passed,
            "trainer_telemetry": telemetry,
            "admitted_child_binding": dict(binding),
            "outer_test_opened": False,
            "accuracy_metrics_emitted_or_used": False,
            "checkpoint_selection_performed": False,
            "training_result_reusable": False,
            "selection_or_promotion_input": False,
            "artifacts_quarantined": True,
            "commercial_claim_authorized": False,
        },
    )
    return 0 if gate_passed else GATE_FAILURE_EXIT


def validate_worker_telemetry(
    path: Path, *, invocation_sha256: str
) -> dict[str, Any]:
    document = load_json(path, "quarantined benchmark telemetry")
    expected_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "phase",
        "benchmark_id",
        "unit",
        "usage_identity",
        "profile_sha256",
        "epochs",
        "epoch_1_is_warmup",
        "epoch_2_train_ns",
        "epoch_2_target_free_validation_ns",
        "epoch_2_train_plus_target_free_validation_ns",
        "epoch_2_gate_ns_max",
        "gate_passed",
        "trainer_telemetry",
        "admitted_child_binding",
        "outer_test_opened",
        "accuracy_metrics_emitted_or_used",
        "checkpoint_selection_performed",
        "training_result_reusable",
        "selection_or_promotion_input",
        "artifacts_quarantined",
        "commercial_claim_authorized",
        "content_sha256",
    }
    if set(document) != expected_keys:
        raise BenchmarkError("quarantined benchmark telemetry schema drifted")
    if canonical_content_sha256(document) != document.get("content_sha256"):
        raise BenchmarkError("quarantined benchmark telemetry content drifted")
    trainer = document.get("trainer_telemetry")
    if not isinstance(trainer, Mapping):
        raise BenchmarkError("quarantined trainer telemetry is missing")
    validated = validate_trainer_telemetry(
        trainer, invocation_sha256=invocation_sha256
    )
    binding = document.get("admitted_child_binding")
    steady = validated["epochs"][1]
    steady_ns = int(steady["total_ns"])
    gate_passed = steady_ns <= STEADY_GATE_NS
    if not (
        type(document.get("schema_version")) is int
        and document.get("schema_version") == 1
        and document.get("classification")
        == "adaptive_v3r1_v8r4_quarantined_efficiency_telemetry"
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == CAMPAIGN_REVISION
        and document.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and document.get("phase") == BENCHMARK_PHASE
        and document.get("benchmark_id") == BENCHMARK_ID
        and document.get("unit") == BENCHMARK_UNIT
        and document.get("usage_identity") == BENCHMARK_USAGE_IDENTITY
        and document.get("profile_sha256") == BENCHMARK_PROFILE_SHA256
        and type(document.get("epochs")) is int
        and document.get("epochs") == EPOCHS
        and document.get("epoch_1_is_warmup") is True
        and type(document.get("epoch_2_train_ns")) is int
        and document.get("epoch_2_train_ns") == steady["train_ns"]
        and type(document.get("epoch_2_target_free_validation_ns")) is int
        and document.get("epoch_2_target_free_validation_ns")
        == steady["validation_ns"]
        and type(
            document.get("epoch_2_train_plus_target_free_validation_ns")
        )
        is int
        and document.get("epoch_2_train_plus_target_free_validation_ns")
        == steady_ns
        and type(document.get("epoch_2_gate_ns_max")) is int
        and document.get("epoch_2_gate_ns_max") == STEADY_GATE_NS
        and document.get("gate_passed") is gate_passed
        and isinstance(binding, Mapping)
        and binding.get("valid") is True
        and binding.get("classification")
        == "verified_v8_gpu_admitted_child_lifecycle"
        and binding.get("phase") == BENCHMARK_PHASE
        and binding.get("invocation_sha256") == invocation_sha256
        and binding.get("context") == BENCHMARK_USAGE_IDENTITY
        and document.get("outer_test_opened") is False
        and document.get("accuracy_metrics_emitted_or_used") is False
        and document.get("checkpoint_selection_performed") is False
        and document.get("training_result_reusable") is False
        and document.get("selection_or_promotion_input") is False
        and document.get("artifacts_quarantined") is True
        and document.get("commercial_claim_authorized") is False
    ):
        raise BenchmarkError("quarantined benchmark telemetry invariants drifted")
    status = path.stat()
    if path.is_symlink() or status.st_nlink != 1 or stat.S_IMODE(status.st_mode) != 0o444:
        raise BenchmarkError("quarantined benchmark telemetry is not exact 0444")
    return document


def _under(root: Path, path: Path) -> Path:
    candidate = path.expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _python_path(root: Path, path: Path | None) -> Path:
    candidate = path.expanduser() if path is not None else Path(".venv/bin/python")
    if not candidate.is_absolute():
        candidate = root / candidate
    return Path(os.path.abspath(candidate))


def _canonical_parent_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    """Reject lifecycle/source overrides before any artifact is opened."""

    project_root = args.project_root.expanduser().resolve()
    python = _python_path(project_root, args.python)
    actual = (
        _under(project_root, args.trainer),
        _under(project_root, args.gpu_wrapper),
        _under(project_root, args.gpu_lock),
        _under(project_root, args.execution_ledger),
        _under(project_root, args.usage_ledger),
    )
    expected = (
        _under(project_root, TRAINER_RELATIVE),
        _under(project_root, WRAPPER_RELATIVE),
        _under(project_root, DEFAULT_GPU_LOCK),
        _under(project_root, DEFAULT_EXECUTION_LEDGER),
        _under(project_root, DEFAULT_USAGE_LEDGER),
    )
    if actual != expected or python != Path(os.path.abspath(sys.executable)):
        raise BenchmarkError(
            "benchmark Python/trainer/wrapper/lock/ledger paths must be canonical defaults"
        )
    return (project_root, python, *actual)


def benchmark_worker_command(
    *,
    python: Path,
    project_root: Path,
    trainer: Path,
    cache: Path,
    proposer_stack: Path,
    telemetry_output: Path,
    forbidden_output_dir: Path,
    usage_ledger: Path,
    execution_ledger: Path,
    authorization_path: Path,
    authorization_sha256: str,
    target_sealed_capability_receipt: Path | None = None,
) -> list[str]:
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "--internal-worker",
        "--project-root",
        str(project_root),
        "--trainer",
        str(trainer),
        "--cache",
        str(cache),
        "--proposer-stack",
        str(proposer_stack),
        "--telemetry-output",
        str(telemetry_output),
        "--trainer-output-dir",
        str(forbidden_output_dir),
        "--usage-ledger",
        str(usage_ledger),
        "--execution-ledger",
        str(execution_ledger),
        "--authorization-path",
        str(authorization_path),
        "--authorization-sha256",
        authorization_sha256,
    ]
    if target_sealed_capability_receipt is not None:
        command.extend(
            ["--target-sealed-capability-receipt", str(target_sealed_capability_receipt)]
        )
    return command


def admitted_wrapper_command(
    *,
    python: Path,
    wrapper: Path,
    gpu_lock: Path,
    execution_ledger: Path,
    usage_ledger: Path,
    result_file: Path,
    invocation_sha256: str,
    authorization_path: Path,
    authorization_sha256: str,
    worker_command: Sequence[str],
) -> list[str]:
    return [
        str(python),
        str(wrapper),
        "--lock-file",
        str(gpu_lock),
        "--ledger",
        str(execution_ledger),
        "--usage-ledger",
        str(usage_ledger),
        "--result-file",
        str(result_file),
        "--campaign-id",
        CAMPAIGN_ID,
        "--phase",
        BENCHMARK_PHASE,
        "--context-json",
        canonical_json_bytes(BENCHMARK_USAGE_IDENTITY).decode("utf-8"),
        "--invocation-sha256",
        invocation_sha256,
        "--authorization-path",
        str(authorization_path),
        "--authorization-sha256",
        authorization_sha256,
        "--budget-seconds",
        "36000",
        "--",
        *worker_command,
    ]


def _attempt_directories(run_root: Path) -> list[Path]:
    attempts_root = run_root / "attempts"
    if not attempts_root.exists():
        return []
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise BenchmarkError("benchmark attempts root is not a directory")
    all_entries = sorted(attempts_root.iterdir(), key=lambda item: item.name)
    entries = [item for item in all_entries if item.name.startswith("attempt_")]
    expected_names = [f"attempt_{index:03d}" for index in range(len(entries))]
    if (
        [entry.name for entry in entries] != expected_names
        or any(entry.is_symlink() or not entry.is_dir() for entry in entries)
        or len(entries) > 1
    ):
        raise BenchmarkError("benchmark attempt sequence is non-canonical")
    staging_name = f".attempt_{len(entries):03d}.staging"
    foreign = [item for item in all_entries if item not in entries]
    if len(foreign) > 1 or (foreign and foreign[0].name != staging_name):
        raise BenchmarkError("benchmark attempts contain a foreign or non-tail entry")
    if foreign:
        staging = foreign[0]
        if staging.is_symlink() or not staging.is_dir():
            raise BenchmarkError("benchmark staged attempt is aliased")
        names = [item.name for item in sorted(staging.iterdir(), key=lambda item: item.name)]
        if names not in ([], ["invocation.json"]):
            raise BenchmarkError("benchmark staged attempt has unknown content")
        if names:
            status = os.stat(staging / "invocation.json", follow_symlinks=False)
            if not (
                stat.S_ISREG(status.st_mode)
                and status.st_nlink == 1
                and stat.S_IMODE(status.st_mode) == 0o444
            ):
                raise BenchmarkError("benchmark staged invocation is not immutable")
    for entry in entries:
        invocation = entry / "invocation.json"
        status = os.stat(invocation, follow_symlinks=False)
        if not (
            stat.S_ISREG(status.st_mode)
            and status.st_nlink == 1
            and stat.S_IMODE(status.st_mode) == 0o444
        ):
            raise BenchmarkError("benchmark final attempt lacks an immutable invocation")
    # If a parent died after rename but before its directory fsync, observing
    # and validating the final entry is enough to finish that same publication.
    _fsync_directory(attempts_root)
    return entries


def _publish_benchmark_attempt(
    run_root: Path,
    *,
    create_invocation: Callable[[Path], Mapping[str, Any]],
    validate_invocation: Callable[[Path], Mapping[str, Any]],
) -> Path:
    """Publish the sole scientific attempt after its invocation is durable."""

    attempts = _attempt_directories(run_root)
    if attempts:
        return attempts[0]
    attempts_root = run_root / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    staging = attempts_root / ".attempt_000.staging"
    final = attempts_root / "attempt_000"
    if not staging.exists():
        staging.mkdir(mode=0o700)
        _fsync_directory(attempts_root)
        _publication_fault("indexed_directory_staged", final)
    elif staging.is_symlink() or not staging.is_dir():
        raise BenchmarkError("benchmark staged attempt is not a directory")
    invocation = staging / "invocation.json"
    entries = sorted(staging.iterdir(), key=lambda item: item.name)
    if not entries:
        create_invocation(invocation)
    elif [item.name for item in entries] == ["invocation.json"]:
        validate_invocation(invocation)
    else:
        raise BenchmarkError("benchmark staged attempt has unknown content")
    validate_invocation(invocation)
    _fsync_directory(staging)
    _publication_fault("indexed_invocation_durable", final)
    _rename_staged_directory_once(staging, final)
    _publication_fault("indexed_directory_linked", final)
    _fsync_directory(attempts_root)
    _publication_fault("indexed_directory_published", final)
    return final


def _no_discovery_terminal(state: Any) -> None:
    """Allow only the one exact frozen V8R3 quarantine discovery terminal."""

    quarantine_seen = 0
    for record in state.records:
        if record.get("phase") != "discovery":
            continue
        if record.get("schema_version") == 2:
            terminal = record.get("event") in {"terminal", "reconciled_terminal"}
        else:
            terminal = True
        if not terminal:
            continue
        context = record.get("context")
        if (
            record.get("record_sha256")
            == "8dbc0493125f22c130444e1344533d1f3d9c4ac445df6adfe8b597972e9691c5"
            and isinstance(context, Mapping)
            and context
            == {
                "execution_number": 0,
                "outer_fold": 3,
                "resume": False,
                "seed": 20260828,
                "variant": "H0_no_factor",
            }
        ):
            quarantine_seen += 1
            continue
        raise BenchmarkError(
            "efficiency benchmark must precede every non-quarantined discovery terminal"
        )
    if quarantine_seen > 1:
        raise BenchmarkError("V8R3 discovery quarantine terminal is duplicated")


def _historical_projection_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefixes = ("v8", "v8r2", "v8r3")
    for prefix, attempt in zip(
        prefixes, HISTORICAL_BENCHMARK_ATTEMPTS, strict=True
    ):
        for name, role in (
            ("unit_invocation", f"{prefix}_unit_invocation"),
            ("execution_invocation", f"{prefix}_execution_invocation"),
            (
                "terminal_result",
                (
                    "v8r3_quarantined_terminal_result"
                    if prefix == "v8r3"
                    else f"{prefix}_terminal_result"
                ),
            ),
        ):
            record = attempt[name]
            rows.append(
                {
                    "bytes": record["bytes"],
                    "file_sha256": record["sha256"],
                    "mode": f"{int(record.get('mode', 0o444)):04o}",
                    "path": record["relative_path"],
                    "role": role,
                }
            )
        telemetry = attempt.get("telemetry")
        if isinstance(telemetry, Mapping):
            rows.append(
                {
                    "bytes": telemetry["bytes"],
                    "file_sha256": telemetry["sha256"],
                    "mode": f"{int(telemetry.get('mode', 0o444)):04o}",
                    "path": telemetry["relative_path"],
                    "role": f"{prefix}_quarantined_telemetry",
                }
            )
        authorization = attempt["pretrain_authorization"]
        rows.append(
            {
                "bytes": authorization["bytes"],
                "file_sha256": authorization["sha256"],
                "mode": "0444",
                "path": authorization["relative_path"],
                "role": f"{prefix}_pretrain_authorization",
            }
        )
    return rows


def _validate_historical_projection_authority(project_root: Path) -> dict[str, Any]:
    """Validate the frozen history projection without opening the legacy root."""

    path = (project_root / EXECUTION_CLOSURE_AUTHORIZATION_RELATIVE).resolve()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BenchmarkError("execution-closure authority is unavailable") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        status = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    expected = _RECOVERY_GOVERNANCE[
        "execution_closure_correction_authorization"
    ]
    if not (
        stat.S_ISREG(status.st_mode)
        and status.st_nlink == 1
        and stat.S_IMODE(status.st_mode) == 0o444
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
        and (status.st_dev, status.st_ino, status.st_mode)
        == (named.st_dev, named.st_ino, named.st_mode)
        and len(raw) == expected["bytes"]
        and hashlib.sha256(raw).hexdigest() == expected["sha256"]
    ):
        raise BenchmarkError("execution-closure authority file binding drifted")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"execution-closure authority JSON is invalid: {error}") from error
    if not isinstance(document, Mapping):
        raise BenchmarkError("execution-closure authority is not an object")
    history = document.get("authority_basis", {}).get("historical_benchmark_prefix")
    if not (
        canonical_content_sha256(document) == document.get("content_sha256")
        and document.get("content_sha256") == expected["content_sha256"]
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and document.get("mandatory_invariants", {}).get(
            "benchmark_active_scientific_attempts"
        )
        == 1
        and isinstance(history, Mapping)
        and history.get("active_output_root")
        == FROZEN_EXECUTION_CLOSURE_ACTIVE_OUTPUT_ROOT_RELATIVE.as_posix()
        and history.get("historical_root_mounted_or_mutated") is False
        and history.get("known_v8r3_mode_0644_is_read_only_quarantined_evidence")
        is True
        and isinstance(history.get("entries"), list)
        and sorted(history["entries"], key=lambda row: str(row.get("role", "")))
        == sorted(_historical_projection_rows(), key=lambda row: str(row["role"]))
    ):
        raise BenchmarkError("execution-closure historical projection drifted")
    return {
        "path": EXECUTION_CLOSURE_AUTHORIZATION_RELATIVE.as_posix(),
        "sha256": expected["sha256"],
        "bytes": expected["bytes"],
    }


def validate_rootbind1_failed_benchmark_terminal(
    record: Mapping[str, Any], *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Validate the sole charged pre-CONTEXT1 infrastructure terminal."""

    expected = ROOTBIND1_FAILED_BENCHMARK_TERMINAL
    expected_result = str(
        (project_root.expanduser().resolve() / expected["result_relative_path"]).resolve()
    )
    if not (
        record.get("schema_version") == 2
        and record.get("event") == "terminal"
        and record.get("campaign_id") == CAMPAIGN_ID
        and record.get("phase") == BENCHMARK_PHASE
        and record.get("context") == ROOTBIND1_BENCHMARK_USAGE_IDENTITY
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
        raise BenchmarkError("ROOTBIND1 benchmark failure terminal drifted")
    return dict(record)


def _validate_benchmark_entry_prefix(
    state: Any, *, project_root: Path = PROJECT_ROOT
) -> None:
    """Require exact historical -> quarantine ordering before active launch."""

    _no_discovery_terminal(state)
    historical = [
        str(item["terminal_result"]["terminal_record_sha256"])
        for item in HISTORICAL_BENCHMARK_ATTEMPTS
    ]
    quarantine = "8dbc0493125f22c130444e1344533d1f3d9c4ac445df6adfe8b597972e9691c5"
    positions = {
        str(record.get("record_sha256", "")): number
        for number, record in enumerate(state.records)
    }
    rootbind_failure = str(
        ROOTBIND1_FAILED_BENCHMARK_TERMINAL["terminal_record_sha256"]
    )
    if any(
        value not in positions
        for value in [*historical, quarantine, rootbind_failure]
    ):
        raise BenchmarkError("benchmark historical/quarantine ledger prefix is incomplete")
    observed_historical: list[str] = []
    observed_rootbind: list[str] = []
    active: list[dict[str, Any]] = []
    for record in state.records:
        if record.get("phase") != BENCHMARK_PHASE:
            continue
        terminal = (
            record.get("schema_version") != 2
            or record.get("event") in {"terminal", "reconciled_terminal"}
        )
        if not terminal:
            continue
        record_hash = str(record.get("record_sha256", ""))
        context = record.get("context")
        if record_hash in historical:
            if context != LEGACY_BENCHMARK_USAGE_IDENTITY:
                raise BenchmarkError("historical benchmark terminal identity drifted")
            observed_historical.append(record_hash)
        elif record_hash == rootbind_failure:
            validate_rootbind1_failed_benchmark_terminal(
                record, project_root=project_root
            )
            observed_rootbind.append(record_hash)
        elif record.get("schema_version") == 2 and context == BENCHMARK_USAGE_IDENTITY:
            active.append(dict(record))
        else:
            raise BenchmarkError("unowned efficiency-benchmark terminal in ledger")
    if observed_historical != historical or observed_rootbind != [rootbind_failure]:
        raise BenchmarkError("historical benchmark terminal order or cover drifted")
    historical_positions = [positions[value] for value in historical]
    if not (
        historical_positions == sorted(historical_positions)
        and max(historical_positions)
        < positions[quarantine]
        < positions[rootbind_failure]
    ):
        raise BenchmarkError("benchmark historical/quarantine ledger ordering drifted")
    if len(active) > 1:
        raise BenchmarkError("more than one active scientific benchmark terminal exists")
    if active and positions[str(active[0]["record_sha256"])] <= positions[rootbind_failure]:
        raise BenchmarkError("active benchmark terminal precedes its quarantine prefix")


def _run_command(command: Sequence[str]) -> int:
    return int(subprocess.run(list(command), check=False).returncode)


def _active_authorization_path(
    project_root: Path, authorization: Mapping[str, Any]
) -> tuple[Path, str]:
    binding = authorization.get("authorization_binding")
    if not isinstance(binding, Mapping):
        raise BenchmarkError("active pretrain authorization binding is missing")
    path = Path(str(binding.get("path", "")))
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    digest = binding.get("sha256")
    if not isinstance(digest, str) or sha256_file(path) != digest:
        raise BenchmarkError("active pretrain authorization binding drifted")
    return path, digest


def _binding_path_without_live_read(
    binding: Any, *, project_root: Path, label: str
) -> Path:
    """Resolve one already-authorized file binding without blessing live drift.

    Current authority is validated against live bytes by
    ``_validate_benchmark_invocation_binding``.  Historical V8 and V8R2 units
    are instead authorized by their exact immutable whole-file hashes, because
    source files named inside those documents have legitimately advanced.  This
    helper only extracts paths after strict binding-shape checks; it never
    substitutes for either authority validation.
    """

    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise BenchmarkError(f"{label} binding schema drifted")
    raw_path = binding.get("path")
    digest = binding.get("sha256")
    size = binding.get("bytes")
    if not (
        isinstance(raw_path, str)
        and raw_path
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        and type(size) is int
        and size >= 0
    ):
        raise BenchmarkError(f"{label} binding identity drifted")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    lexical = Path(os.path.abspath(candidate))
    if lexical != candidate.resolve():
        raise BenchmarkError(f"{label} binding path is aliased")
    return lexical


def _worker_command_from_unit_authority(
    *,
    python: Path,
    project_root: Path,
    unit_invocation: Mapping[str, Any],
    attempt_root: Path,
    usage_ledger: Path,
    execution_ledger: Path,
) -> list[str]:
    """Reconstruct an attempt from the authority named by that attempt."""

    trainer = _binding_path_without_live_read(
        unit_invocation.get("trainer"),
        project_root=project_root,
        label="benchmark authority trainer",
    )
    proposer = _binding_path_without_live_read(
        unit_invocation.get("proposer_stack"),
        project_root=project_root,
        label="benchmark authority proposer",
    )
    authorization = _binding_path_without_live_read(
        unit_invocation.get("pretrain_authorization"),
        project_root=project_root,
        label="benchmark authority pretrain authorization",
    )
    manifest = _binding_path_without_live_read(
        unit_invocation.get("cache_manifest"),
        project_root=project_root,
        label="benchmark authority cache manifest",
    )
    authorization_binding = unit_invocation.get("pretrain_authorization")
    authorization_sha = (
        authorization_binding.get("sha256")
        if isinstance(authorization_binding, Mapping)
        else None
    )
    if not isinstance(authorization_sha, str):
        raise BenchmarkError("benchmark authority authorization SHA-256 is missing")
    if not (
        unit_invocation.get("usage_ledger_path") == str(usage_ledger.resolve())
        and unit_invocation.get("gpu_execution_ledger_path")
        == str(execution_ledger.resolve())
    ):
        raise BenchmarkError("benchmark authority lifecycle ledger paths drifted")
    return benchmark_worker_command(
        python=python,
        project_root=project_root,
        trainer=trainer,
        cache=manifest.parent,
        proposer_stack=proposer,
        telemetry_output=attempt_root / "QUARANTINED_TIMING_TELEMETRY.json",
        forbidden_output_dir=attempt_root / "FORBIDDEN_REUSABLE_TRAINER_OUTPUT",
        usage_ledger=usage_ledger,
        execution_ledger=execution_ledger,
        authorization_path=authorization,
        authorization_sha256=authorization_sha,
    )


def _read_exact_legacy_file(
    *, project_root: Path, record: Mapping[str, Any], label: str
) -> tuple[Path, dict[str, Any]]:
    relative_path = record.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise BenchmarkError(f"{label} whitelist path is malformed")
    path = (project_root / relative_path).resolve()
    binding = {
        "path": str(path),
        "sha256": record.get("sha256"),
        "bytes": record.get("bytes"),
    }
    verified_path, raw = _read_verified_binding(
        binding, project_root=project_root, label=label
    )
    status = verified_path.stat()
    expected_mode = record.get("mode", 0o444)
    if type(expected_mode) is not int or stat.S_IMODE(status.st_mode) != expected_mode:
        raise BenchmarkError(f"{label} historical mode drifted")
    document = _load_json_bytes(raw, label)
    if not (
        canonical_content_sha256(document) == document.get("content_sha256")
        and document.get("content_sha256") == record.get("content_sha256")
    ):
        raise BenchmarkError(f"{label} canonical content drifted")
    return verified_path, document


def _validate_exact_historical_attempt(
    discovery: Any,
    *,
    historical_attempt: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    usage_ledger: Path,
    execution_ledger: Path,
    python: Path,
) -> tuple[Path, Path, dict[str, Any], list[str]]:
    """Validate and reconstruct one exact immutable failed lifecycle."""

    canonical_run_root = (
        project_root / HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE
    ).resolve()
    if run_root.resolve() != canonical_run_root:
        raise BenchmarkError("historical benchmark attempt is outside its exact run root")
    authority = historical_attempt.get("authority")
    attempt_index = historical_attempt.get("attempt_index")
    if not (
        isinstance(authority, str)
        and authority
        and type(attempt_index) is int
        and attempt_index >= 0
    ):
        raise BenchmarkError("historical benchmark whitelist identity is malformed")
    unit_record = historical_attempt["unit_invocation"]
    execution_record = historical_attempt["execution_invocation"]
    terminal_record = historical_attempt["terminal_result"]
    unit_path, unit = _read_exact_legacy_file(
        project_root=project_root,
        record=unit_record,
        label=f"authorized historical {authority} benchmark unit invocation",
    )
    invocation_path, invocation = _read_exact_legacy_file(
        project_root=project_root,
        record=execution_record,
        label=f"authorized historical {authority} execution invocation",
    )
    result_path, result = _read_exact_legacy_file(
        project_root=project_root,
        record=terminal_record,
        label=f"authorized historical {authority} terminal result",
    )
    expected_attempt_root = run_root / "attempts" / f"attempt_{attempt_index:03d}"
    expected_authorization = historical_attempt["pretrain_authorization"]
    authorization_binding = unit.get("pretrain_authorization")
    expected_authorization_path = (
        project_root / str(expected_authorization["relative_path"])
    ).resolve()
    if not (
        unit_path == (project_root / str(unit_record["relative_path"])).resolve()
        and unit_path.parent == run_root
        and invocation_path == expected_attempt_root / "invocation.json"
        and result_path == expected_attempt_root / "GPU_TERMINAL_RESULT.json"
        and isinstance(authorization_binding, Mapping)
        and authorization_binding.get("path") == str(expected_authorization_path)
        and authorization_binding.get("sha256")
        == expected_authorization.get("sha256")
        and authorization_binding.get("bytes") == expected_authorization.get("bytes")
        and sha256_file(expected_authorization_path)
        == expected_authorization.get("sha256")
        and invocation.get("unit_invocation") == discovery.bind_file(unit_path)
        and result.get("invocation_sha256") == execution_record.get("sha256")
        and result.get("terminal_record_sha256")
        == terminal_record.get("terminal_record_sha256")
        and type(result.get("return_code")) is int
        and result.get("return_code") == terminal_record.get("return_code")
        and type(result.get("charged_usage_ns")) is int
        and result.get("charged_usage_ns") == terminal_record.get("charged_usage_ns")
        and result.get("reusable_success")
        is terminal_record.get("reusable_success")
        and (
            isinstance(historical_attempt.get("telemetry"), Mapping)
            or not (expected_attempt_root / "QUARANTINED_TIMING_TELEMETRY.json").exists()
        )
        and not (expected_attempt_root / "FORBIDDEN_REUSABLE_TRAINER_OUTPUT").exists()
    ):
        raise BenchmarkError(
            f"authorized historical {authority} benchmark lineage drifted"
        )
    telemetry_record = historical_attempt.get("telemetry")
    if telemetry_record is not None:
        if not isinstance(telemetry_record, Mapping):
            raise BenchmarkError(f"authorized historical {authority} telemetry malformed")
        telemetry_path, telemetry = _read_exact_legacy_file(
            project_root=project_root,
            record=telemetry_record,
            label=f"authorized historical {authority} benchmark telemetry",
        )
        if telemetry_path != expected_attempt_root / "QUARANTINED_TIMING_TELEMETRY.json":
            raise BenchmarkError(f"authorized historical {authority} telemetry path drifted")
    legacy_usage = Path(str(unit.get("usage_ledger_path", "")))
    legacy_execution = Path(str(unit.get("gpu_execution_ledger_path", "")))
    isolated_exact_paths = (
        legacy_usage.resolve() == usage_ledger.resolve()
        and legacy_execution.resolve() == execution_ledger.resolve()
    )
    migrated_production_paths = (
        legacy_usage.is_absolute()
        and legacy_execution.is_absolute()
        and legacy_usage.name == DEFAULT_USAGE_LEDGER.name
        and legacy_execution.name == DEFAULT_EXECUTION_LEDGER.name
        and usage_ledger.resolve() == (project_root / DEFAULT_USAGE_LEDGER).resolve()
        and execution_ledger.resolve()
        == (project_root / DEFAULT_EXECUTION_LEDGER).resolve()
    )
    if not (isolated_exact_paths or migrated_production_paths):
        raise BenchmarkError(
            f"authorized historical {authority} old-path lineage is not scoped "
            "to the migrated prefix"
        )
    worker_command = _worker_command_from_unit_authority(
        python=python,
        project_root=project_root,
        unit_invocation=unit,
        attempt_root=expected_attempt_root,
        usage_ledger=legacy_usage,
        execution_ledger=legacy_execution,
    )
    if not (
        invocation.get("workload_command") == worker_command
        and invocation.get("workload_command_sha256")
        == semantic_sha256(worker_command)
        and result.get("command_sha256") == semantic_sha256(worker_command)
    ):
        raise BenchmarkError(
            f"authorized historical {authority} command reconstruction drifted"
        )
    return invocation_path, result_path, result, worker_command


def _validate_exact_legacy_v8_attempt(
    discovery: Any,
    *,
    project_root: Path,
    run_root: Path,
    usage_ledger: Path,
    execution_ledger: Path,
    python: Path,
) -> tuple[Path, Path, dict[str, Any], list[str]]:
    """Compatibility entry point for the exact transitive V8 attempt."""

    return _validate_exact_historical_attempt(
        discovery,
        historical_attempt=LEGACY_V8_ATTEMPT,
        project_root=project_root,
        run_root=run_root,
        usage_ledger=usage_ledger,
        execution_ledger=execution_ledger,
        python=python,
    )


def _create_unit_invocation(
    discovery: Any,
    *,
    path: Path,
    contract_binding: Mapping[str, Any],
    training_index_binding: Mapping[str, Any],
    cache_input_binding: Mapping[str, Any],
    proposer_stack_binding: Mapping[str, Any],
    trainer: Path,
    wrapper: Path,
    authorization_path: Path,
    usage_ledger: Path,
    execution_ledger: Path,
    gpu_lock: Path,
    target_sealed_capability: Mapping[str, Any],
    gpu_state_migration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "classification": "adaptive_v3r1_v8r4_target_sealed_efficiency_benchmark_invocation",
        "campaign_id": CAMPAIGN_ID,
        "campaign_revision": CAMPAIGN_REVISION,
        "infrastructure_revision": INFRASTRUCTURE_REVISION,
        "phase": BENCHMARK_PHASE,
        "benchmark_id": BENCHMARK_ID,
        "unit": BENCHMARK_UNIT,
        "usage_identity": dict(BENCHMARK_USAGE_IDENTITY),
        "profile": dict(BENCHMARK_PROFILE),
        "profile_sha256": BENCHMARK_PROFILE_SHA256,
        "contract": dict(contract_binding),
        "training_index": dict(training_index_binding),
        "cache_manifest": dict(cache_input_binding["manifest"]),
        "cache_inputs": dict(cache_input_binding),
        "proposer_stack": dict(proposer_stack_binding),
        "trainer": discovery.bind_file(trainer),
        "gpu_wrapper": discovery.bind_file(wrapper),
        "pretrain_authorization": discovery.bind_file(authorization_path),
        "target_sealed_capability": dict(target_sealed_capability),
        "gpu_state_migration": (
            None if gpu_state_migration is None else dict(gpu_state_migration)
        ),
        "usage_ledger_path": str(usage_ledger),
        "gpu_execution_ledger_path": str(execution_ledger),
        "gpu_admission_lock_path": str(gpu_lock),
        "outer_test_opened": False,
        "accuracy_metrics_authorized": False,
        "checkpoint_selection_authorized": False,
        "training_result_reusable": False,
        "selection_or_promotion_input": False,
        "legacy_combined_cache_used": False,
        "physical_nonouter_pack_only": True,
    }
    if not path.exists():
        return discovery.create_once_json(path, payload)
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise BenchmarkError("current benchmark unit invocation is aliased")
    document = discovery.load_json(path, "current V8R4 benchmark unit invocation")
    expected = dict(payload)
    expected["content_sha256"] = semantic_sha256(expected)
    if not (
        document == expected
        and canonical_content_sha256(document) == document.get("content_sha256")
        and stat.S_IMODE(path.stat().st_mode) == 0o444
    ):
        raise BenchmarkError(
            "current V8R4 benchmark unit invocation resume binding drifted"
        )
    return document


def _create_execution_invocation(
    discovery: Any,
    *,
    path: Path,
    unit_invocation: Path,
    worker_command: Sequence[str],
    usage_identity: Mapping[str, Any] = BENCHMARK_USAGE_IDENTITY,
) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "classification": "adaptive_v3r1_gpu_execution_invocation",
        "campaign_id": CAMPAIGN_ID,
        "phase": BENCHMARK_PHASE,
        "context": dict(usage_identity),
        "unit_invocation": discovery.bind_file(unit_invocation),
        "workload_command": list(worker_command),
        "workload_command_sha256": semantic_sha256(list(worker_command)),
        "parent_side_elapsed_accounting": False,
    }
    # The two immutable pre-migration attempts retain their byte-exact V8/V8R2
    # schema.  Only the live child is issued under the V8R4/V8R4A authority.
    if dict(usage_identity) != LEGACY_BENCHMARK_USAGE_IDENTITY:
        payload["campaign_revision"] = CAMPAIGN_REVISION
        payload["infrastructure_revision"] = INFRASTRUCTURE_REVISION
    return discovery.create_once_json(
        path,
        payload,
    )


def _validate_execution_invocation(
    discovery: Any,
    *,
    path: Path,
    unit_invocation: Path,
    worker_command: Sequence[str],
) -> dict[str, Any]:
    document = discovery.load_json(path, "efficiency GPU execution invocation")
    if not (
        discovery.canonical_content_sha256(document) == document.get("content_sha256")
        and document.get("schema_version") == 2
        and document.get("classification")
        == "adaptive_v3r1_gpu_execution_invocation"
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == CAMPAIGN_REVISION
        and document.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and document.get("phase") == BENCHMARK_PHASE
        and document.get("context") == BENCHMARK_USAGE_IDENTITY
        and document.get("unit_invocation") == discovery.bind_file(unit_invocation)
        and document.get("workload_command") == list(worker_command)
        and document.get("workload_command_sha256")
        == semantic_sha256(list(worker_command))
        and document.get("parent_side_elapsed_accounting") is False
    ):
        raise BenchmarkError("efficiency GPU execution invocation drifted")
    return document


_REQUIRED_CACHE_OUTPUTS = {
    "feature_names": "feature_names.json",
    "metadata": "metadata.csv",
    "node_features": "node_features.npy",
    "candidate_bpm": "candidate_bpm.npy",
    "candidate_mask": "candidate_mask.npy",
    "joint_radar_mask": "joint_radar_mask.npy",
    "local_to_global_cache_index": "local_to_global_cache_index.npy",
}


def _load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                BenchmarkError(f"non-finite {label} value: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label} must be a JSON object")
    return value


def _read_verified_binding(
    binding: Any, *, project_root: Path, label: str
) -> tuple[Path, bytes]:
    """Read one exact path/hash/size binding through a stable no-follow FD."""

    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise BenchmarkError(f"{label} binding schema drifted")
    raw_path = binding.get("path")
    expected_sha256 = binding.get("sha256")
    expected_bytes = binding.get("bytes")
    if not (
        isinstance(raw_path, str)
        and raw_path
        and isinstance(expected_sha256, str)
        and len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256)
        and type(expected_bytes) is int
        and expected_bytes >= 0
    ):
        raise BenchmarkError(f"{label} binding identity drifted")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    lexical = Path(os.path.abspath(candidate))
    if lexical != candidate.resolve():
        raise BenchmarkError(f"{label} binding path is aliased")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        before_path = os.stat(lexical, follow_symlinks=False)
        descriptor = os.open(lexical, flags)
    except OSError as error:
        raise BenchmarkError(f"cannot open {label}: {error}") from error
    try:
        before_fd = os.fstat(descriptor)
        if not (
            stat.S_ISREG(before_fd.st_mode)
            and before_fd.st_nlink == 1
            and not stat.S_ISLNK(before_path.st_mode)
            and before_path.st_nlink == 1
            and (before_path.st_dev, before_path.st_ino)
            == (before_fd.st_dev, before_fd.st_ino)
        ):
            raise BenchmarkError(f"{label} is not a single-link regular inode")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after_fd = os.fstat(descriptor)
        after_path = os.stat(lexical, follow_symlinks=False)
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before_fd, name) != getattr(after_fd, name) for name in stable
        ) or any(
            getattr(after_fd, name) != getattr(after_path, name) for name in stable
        ):
            raise BenchmarkError(f"{label} changed while it was verified")
        raw = b"".join(chunks)
        if (
            len(raw) != expected_bytes
            or digest.hexdigest() != expected_sha256
        ):
            raise BenchmarkError(f"{label} binding bytes drifted")
        return lexical, raw
    finally:
        os.close(descriptor)


def _validate_cache_input_binding(
    binding: Any, *, project_root: Path
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(binding, Mapping) or set(binding) != {"manifest", "outputs"}:
        raise BenchmarkError("benchmark cache-input binding schema drifted")
    manifest_path, manifest_raw = _read_verified_binding(
        binding.get("manifest"),
        project_root=project_root,
        label="benchmark cache manifest",
    )
    manifest = _load_json_bytes(manifest_raw, "benchmark cache manifest")
    if not (
        manifest.get("complete") is True
        and type(manifest.get("format_version")) is int
        and manifest.get("format_version") == 1
        and manifest.get("classification")
        == "adaptive_v3r1_v8r4_nonouter_training_validation_pack"
        and manifest.get("campaign_id") == CAMPAIGN_ID
        and manifest.get("campaign_revision") == CAMPAIGN_REVISION
        and manifest.get("outer_fold") == 3
        and manifest.get("partition") == "outer_excluded_training_validation"
        and manifest.get("outer_prediction_pack_absent") is True
        and manifest.get("outer_test_rows_physically_present") is False
        and manifest.get("source_combined_cache_open_authorized_by_consumer") is False
        and canonical_content_sha256(manifest) == manifest.get("content_sha256")
    ):
        raise BenchmarkError("benchmark cache manifest invariants drifted")
    outputs = binding.get("outputs")
    manifest_outputs = manifest.get("outputs")
    if not (
        isinstance(outputs, Mapping)
        and set(outputs) == set(_REQUIRED_CACHE_OUTPUTS)
        and isinstance(manifest_outputs, Mapping)
    ):
        raise BenchmarkError("benchmark cache output cover drifted")
    normalized: dict[str, Any] = {}
    for logical_name, filename in _REQUIRED_CACHE_OUTPUTS.items():
        record = outputs.get(logical_name)
        if not isinstance(record, Mapping) or set(record) != {
            "filename",
            "sha256",
            "bytes",
        } or record.get("filename") != filename:
            raise BenchmarkError(f"benchmark cache output binding drifted: {logical_name}")
        if manifest_outputs.get(logical_name) != record:
            raise BenchmarkError(
                f"benchmark cache output differs from manifest: {logical_name}"
            )
        _read_verified_binding(
            {
                "path": str(manifest_path.parent / filename),
                "sha256": record.get("sha256"),
                "bytes": record.get("bytes"),
            },
            project_root=project_root,
            label=f"benchmark cache output {logical_name}",
        )
        normalized[logical_name] = dict(record)
    return manifest_path, {
        "manifest": dict(binding["manifest"]),
        "outputs": normalized,
    }


def _validate_benchmark_invocation_binding(
    *,
    project_root: Path,
    binding: Any,
    receipt: Mapping[str, Any],
    usage_ledger: Path,
    execution_ledger: Path,
    gpu_lock: Path,
) -> tuple[Path, dict[str, Any]]:
    invocation_path, raw = _read_verified_binding(
        binding, project_root=project_root, label="benchmark invocation"
    )
    invocation = _load_json_bytes(raw, "benchmark invocation")
    expected_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "phase",
        "benchmark_id",
        "unit",
        "usage_identity",
        "profile",
        "profile_sha256",
        "contract",
        "training_index",
        "cache_manifest",
        "cache_inputs",
        "proposer_stack",
        "trainer",
        "gpu_wrapper",
        "pretrain_authorization",
        "target_sealed_capability",
        "gpu_state_migration",
        "usage_ledger_path",
        "gpu_execution_ledger_path",
        "gpu_admission_lock_path",
        "outer_test_opened",
        "accuracy_metrics_authorized",
        "checkpoint_selection_authorized",
        "training_result_reusable",
        "selection_or_promotion_input",
        "legacy_combined_cache_used",
        "physical_nonouter_pack_only",
        "content_sha256",
    }
    if not (
        set(invocation) == expected_keys
        and canonical_content_sha256(invocation) == invocation.get("content_sha256")
        and type(invocation.get("schema_version")) is int
        and invocation.get("schema_version") == 1
        and invocation.get("classification")
        == "adaptive_v3r1_v8r4_target_sealed_efficiency_benchmark_invocation"
        and invocation.get("campaign_id") == CAMPAIGN_ID
        and invocation.get("campaign_revision") == CAMPAIGN_REVISION
        and invocation.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and invocation.get("phase") == BENCHMARK_PHASE
        and invocation.get("benchmark_id") == BENCHMARK_ID
        and invocation.get("unit") == BENCHMARK_UNIT
        and invocation.get("usage_identity") == BENCHMARK_USAGE_IDENTITY
        and invocation.get("profile") == BENCHMARK_PROFILE
        and invocation.get("profile_sha256") == BENCHMARK_PROFILE_SHA256
        and invocation.get("usage_ledger_path") == str(usage_ledger.resolve())
        and invocation.get("gpu_execution_ledger_path")
        == str(execution_ledger.resolve())
        and invocation.get("gpu_admission_lock_path") == str(gpu_lock.resolve())
        and invocation.get("outer_test_opened") is False
        and invocation.get("accuracy_metrics_authorized") is False
        and invocation.get("checkpoint_selection_authorized") is False
        and invocation.get("training_result_reusable") is False
        and invocation.get("selection_or_promotion_input") is False
        and invocation.get("legacy_combined_cache_used") is False
        and invocation.get("physical_nonouter_pack_only") is True
    ):
        raise BenchmarkError("benchmark invocation invariants drifted")
    capability_binding = invocation.get("target_sealed_capability")
    if not isinstance(capability_binding, Mapping):
        raise BenchmarkError("benchmark invocation capability binding is missing")
    capability_path = Path(str(capability_binding.get("path", "")))
    if not capability_path.is_absolute():
        capability_path = project_root / capability_path
    capability = validate_target_sealed_capability(
        project_root=project_root,
        capability_receipt=capability_path,
    )
    migration_binding = invocation.get("gpu_state_migration")
    if migration_binding is not None and not isinstance(migration_binding, Mapping):
        raise BenchmarkError("benchmark invocation migration binding is malformed")
    if migration_binding is not None:
        boundary = capability["document"].get("security_boundary", {})
        if not (
            isinstance(boundary, Mapping)
            and boundary.get("production_execution_authorized") is True
            and boundary.get("atomic_replace_compatible") is True
            and boundary.get("synthetic_validation_only") is False
            and boundary.get("v8r4a_ledger_migration_required") is False
        ):
            raise BenchmarkError(
                "V8R4A migration is valid but the target-sealed benchmark "
                "capability remains non-production"
            )
    if capability["binding"] != dict(capability_binding):
        raise BenchmarkError("benchmark invocation capability live replay drifted")
    contract_path, _ = _read_verified_binding(
        invocation.get("contract"), project_root=project_root, label="benchmark contract"
    )
    index_path, index_raw = _read_verified_binding(
        invocation.get("training_index"),
        project_root=project_root,
        label="benchmark training index",
    )
    trainer_path, _ = _read_verified_binding(
        invocation.get("trainer"), project_root=project_root, label="benchmark trainer"
    )
    wrapper_path, _ = _read_verified_binding(
        invocation.get("gpu_wrapper"),
        project_root=project_root,
        label="benchmark GPU wrapper",
    )
    authorization_path, _ = _read_verified_binding(
        invocation.get("pretrain_authorization"),
        project_root=project_root,
        label="benchmark pretrain authorization",
    )
    proposer_path, _ = _read_verified_binding(
        invocation.get("proposer_stack"),
        project_root=project_root,
        label="benchmark proposer stack",
    )
    manifest_path, normalized_cache = _validate_cache_input_binding(
        invocation.get("cache_inputs"), project_root=project_root
    )
    if not (
        contract_path == (project_root / CONTRACT_RELATIVE).resolve()
        and invocation["contract"].get("sha256") == CONTRACT_FILE_SHA256
        and index_path == (project_root / DEFAULT_TRAINING_INDEX).resolve()
        and invocation["training_index"].get("sha256")
        == DEFAULT_TRAINING_INDEX_SHA256
        and invocation["training_index"].get("bytes")
        == DEFAULT_TRAINING_INDEX_BYTES
        and trainer_path == (project_root / TRAINER_RELATIVE).resolve()
        and wrapper_path == (project_root / WRAPPER_RELATIVE).resolve()
        and authorization_path
        == (project_root / PRETRAIN_AUTHORIZATION_RELATIVE).resolve()
        and invocation.get("cache_manifest") == normalized_cache["manifest"]
        and invocation.get("cache_inputs") == normalized_cache
        and receipt.get("contract") == invocation.get("contract")
        and receipt.get("training_index") == invocation.get("training_index")
        and receipt.get("trainer") == invocation.get("trainer")
        and receipt.get("gpu_wrapper") == invocation.get("gpu_wrapper")
        and receipt.get("pretrain_authorization")
        == invocation.get("pretrain_authorization")
        and receipt.get("target_sealed_capability")
        == invocation.get("target_sealed_capability")
        and receipt.get("gpu_state_migration")
        == invocation.get("gpu_state_migration")
    ):
        raise BenchmarkError("benchmark invocation provenance drifted")
    index = _load_json_bytes(index_raw, "benchmark canonical training index")
    if index.get("content_sha256") is not None and canonical_content_sha256(
        index
    ) != index.get("content_sha256"):
        raise BenchmarkError("benchmark training-index content drifted")
    units = index.get("units")
    selected = [
        unit
        for unit in units or []
        if isinstance(unit, Mapping)
        and unit.get("outer_fold") == 3
        and unit.get("seed") == 20260828
    ]
    if len(selected) != 1 or not isinstance(selected[0].get("artifacts"), Mapping):
        raise BenchmarkError("benchmark training-index unit ownership drifted")
    artifacts = selected[0]["artifacts"]
    cache_record = artifacts.get("cache_manifest")
    proposer_record = artifacts.get("proposer_stack")
    if not isinstance(cache_record, Mapping) or not isinstance(proposer_record, Mapping):
        raise BenchmarkError("benchmark training-index artifacts are malformed")
    index_cache_path = (index_path.parent / str(cache_record.get("path", ""))).resolve()
    index_proposer_path = (
        index_path.parent / str(proposer_record.get("path", ""))
    ).resolve()
    if not (
        cache_record.get("sha256") == invocation["cache_manifest"].get("sha256")
        and cache_record.get("bytes") == invocation["cache_manifest"].get("bytes")
        and proposer_record.get("sha256") == invocation["proposer_stack"].get("sha256")
        and proposer_record.get("bytes") == invocation["proposer_stack"].get("bytes")
        and index_cache_path == manifest_path
        and index_proposer_path == proposer_path
    ):
        raise BenchmarkError("benchmark scientific inputs differ from canonical index")
    return invocation_path, invocation


def _validate_receipt_lifecycle_invocations(
    discovery: Any,
    *,
    project_root: Path,
    receipt: Mapping[str, Any],
    unit_invocation_path: Path,
    unit_invocation: Mapping[str, Any],
    usage_ledger: Path,
    execution_ledger: Path,
) -> tuple[dict[str, str], Path]:
    """Rebuild every attempt command from that attempt's own unit authority."""

    lifecycle = receipt.get("lifecycle_invocations")
    terminals = receipt.get("terminal_results")
    if not (
        isinstance(lifecycle, list)
        and isinstance(terminals, list)
        and len(lifecycle) == len(terminals)
        and len(lifecycle) >= 1
    ):
        raise BenchmarkError("benchmark lifecycle ownership is missing")
    run_root = unit_invocation_path.parent
    if unit_invocation_path != run_root / CURRENT_UNIT_INVOCATION_NAME:
        raise BenchmarkError("benchmark current unit invocation path is non-canonical")
    python = Path(os.path.abspath(sys.executable))
    command_by_invocation: dict[str, str] = {}
    successful_telemetry = Path()
    historical_required = (
        run_root
        == (project_root / HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE).resolve()
    )
    historical_by_index = (
        {
            int(record["attempt_index"]): record
            for record in HISTORICAL_BENCHMARK_ATTEMPTS
        }
        if historical_required
        else {}
    )
    if historical_required and set(historical_by_index) != set(
        range(len(HISTORICAL_BENCHMARK_ATTEMPTS))
    ):
        raise BenchmarkError("historical benchmark whitelist indices drifted")
    historical_seen: set[int] = set()
    final_uses_current_authority = False
    legacy_execution_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "phase",
        "context",
        "unit_invocation",
        "workload_command",
        "workload_command_sha256",
        "parent_side_elapsed_accounting",
        "content_sha256",
    }
    active_execution_keys = legacy_execution_keys | {
        "campaign_revision", "infrastructure_revision"
    }
    for number, (binding, terminal) in enumerate(
        zip(lifecycle, terminals, strict=True)
    ):
        if not isinstance(binding, Mapping) or not isinstance(terminal, Mapping):
            raise BenchmarkError("benchmark attempt binding is malformed")
        invocation_path, raw = _read_verified_binding(
            binding,
            project_root=project_root,
            label=f"benchmark execution invocation {number}",
        )
        expected_path = (
            run_root / "attempts" / f"attempt_{number:03d}" / "invocation.json"
        )
        if invocation_path != expected_path:
            raise BenchmarkError("benchmark execution invocation path is non-canonical")
        if terminal.get("execution_invocation") != binding:
            raise BenchmarkError(
                "benchmark terminal/lifecycle invocation bindings disagree"
            )
        document = _load_json_bytes(raw, f"benchmark execution invocation {number}")
        telemetry_path = invocation_path.parent / "QUARANTINED_TIMING_TELEMETRY.json"
        claimed_unit = document.get("unit_invocation")
        historical_attempt = historical_by_index.get(number)
        if historical_attempt is not None:
            unit_path = (
                project_root
                / str(historical_attempt["unit_invocation"]["relative_path"])
            ).resolve()
            if claimed_unit != discovery.bind_file(unit_path):
                raise BenchmarkError(
                    "benchmark execution invocation command/unit binding drifted: "
                    "historical authority mismatch"
                )
            (
                historical_invocation_path,
                historical_result_path,
                historical_result,
                worker_command,
            ) = _validate_exact_historical_attempt(
                discovery,
                historical_attempt=historical_attempt,
                project_root=project_root,
                run_root=run_root,
                usage_ledger=usage_ledger,
                execution_ledger=execution_ledger,
                python=python,
            )
            if not (
                invocation_path == historical_invocation_path
                and terminal.get("result")
                == discovery.bind_file(historical_result_path)
                and terminal.get("terminal_record_sha256")
                == historical_result.get("terminal_record_sha256")
            ):
                raise BenchmarkError("historical benchmark terminal ownership drifted")
            historical_seen.add(number)
            uses_current_authority = False
            expected_context = LEGACY_BENCHMARK_USAGE_IDENTITY
        else:
            if claimed_unit != receipt.get("benchmark_invocation"):
                raise BenchmarkError(
                    "benchmark execution invocation command/unit binding drifted: "
                    "unauthorized unit"
                )
            worker_command = _worker_command_from_unit_authority(
                python=python,
                project_root=project_root,
                unit_invocation=unit_invocation,
                attempt_root=invocation_path.parent,
                usage_ledger=usage_ledger,
                execution_ledger=execution_ledger,
            )
            uses_current_authority = True
            expected_context = BENCHMARK_USAGE_IDENTITY
        if not (
            set(document)
            == (legacy_execution_keys if historical_attempt is not None else active_execution_keys)
            and canonical_content_sha256(document) == document.get("content_sha256")
            and type(document.get("schema_version")) is int
            and document.get("schema_version") == 2
            and document.get("classification")
            == "adaptive_v3r1_gpu_execution_invocation"
            and document.get("campaign_id") == CAMPAIGN_ID
            and (
                historical_attempt is not None
                or (
                    document.get("campaign_revision") == CAMPAIGN_REVISION
                    and document.get("infrastructure_revision")
                    == INFRASTRUCTURE_REVISION
                )
            )
            and document.get("phase") == BENCHMARK_PHASE
            and document.get("context") == expected_context
            and document.get("workload_command") == worker_command
            and document.get("workload_command_sha256")
            == semantic_sha256(worker_command)
            and document.get("parent_side_elapsed_accounting") is False
        ):
            raise BenchmarkError(
                "benchmark execution invocation command/unit binding drifted"
            )
        invocation_sha = str(binding.get("sha256", ""))
        if invocation_sha in command_by_invocation:
            raise BenchmarkError("benchmark execution invocation is duplicated")
        command_by_invocation[invocation_sha] = semantic_sha256(worker_command)
        successful_telemetry = telemetry_path
        final_uses_current_authority = uses_current_authority
    if historical_seen != set(historical_by_index):
        raise BenchmarkError("benchmark historical attempt exact cover is incomplete")
    if not final_uses_current_authority:
        raise BenchmarkError(
            "benchmark successful tail is not owned by the current V8R4 authority"
        )
    return command_by_invocation, successful_telemetry


def _receipt_expected_keys() -> set[str]:
    return {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "phase",
        "benchmark_id",
        "unit",
        "usage_identity",
        "profile_sha256",
        "epochs",
        "epoch_1_is_warmup",
        "epoch_2_train_ns",
        "epoch_2_target_free_validation_ns",
        "epoch_2_train_plus_target_free_validation_ns",
        "epoch_2_gate_ns_max",
        "gate_passed",
        "outer_test_opened",
        "accuracy_metrics_emitted_or_used",
        "checkpoint_selection_performed",
        "training_result_reusable",
        "selection_or_promotion_input",
        "artifacts_quarantined",
        "artifact_disposition",
        "same_usage_and_execution_ledgers",
        "benchmark_before_first_active_discovery_terminal",
        "all_failed_reconciled_and_successful_attempts_owned_by_one_receipt",
        "historical_benchmark_attempts",
        "historical_projection_authority",
        "active_scientific_attempt_count",
        "killed_lifecycle_replay_only",
        "attempt_count",
        "telemetry",
        "benchmark_invocation",
        "pretrain_authorization",
        "target_sealed_capability",
        "gpu_state_migration",
        "contract",
        "training_index",
        "trainer",
        "gpu_wrapper",
        "commercial_claim_authorized",
        "legacy_v8r3_success_quarantined",
        "legacy_combined_cache_used_by_active_attempt",
        "physical_nonouter_pack_only",
        "production_execution_authorized",
        "v8r4a_ledger_migration_required",
        "outer_fold",
        "seed",
        "variant",
        "usage_ledger_path",
        "usage_record_sha256",
        "usage_record_sha256s",
        "terminal_results",
        "lifecycle_invocations",
        "gpu_execution_ledger_path",
        "gpu_admission_lock_path",
        "content_sha256",
    }


def validate_benchmark_receipt(
    discovery: Any,
    *,
    project_root: Path,
    receipt_path: Path,
    usage_ledger: Path,
    execution_ledger: Path,
    gpu_lock: Path,
    expected_command_sha256: Callable[[Mapping[str, Any]], str],
    usage_state: Any | None = None,
    require_no_discovery_terminal: bool = True,
) -> dict[str, Any]:
    if usage_state is None:
        try:
            with discovery.gpu_budget_ledger.locked_closed_snapshot(
                usage_ledger,
                budget_ns=discovery.gpu_budget_ledger.GPU_BUDGET_NS,
                expected_legacy_genesis_sha256=discovery._expected_legacy_genesis(
                    usage_ledger
                ),
            ) as locked_state:
                return validate_benchmark_receipt(
                    discovery,
                    project_root=project_root,
                    receipt_path=receipt_path,
                    usage_ledger=usage_ledger,
                    execution_ledger=execution_ledger,
                    gpu_lock=gpu_lock,
                    expected_command_sha256=expected_command_sha256,
                    usage_state=locked_state,
                    require_no_discovery_terminal=require_no_discovery_terminal,
                )
        except (OSError, ValueError, RuntimeError) as error:
            if isinstance(error, BenchmarkError):
                raise
            raise BenchmarkError(
                f"cannot validate benchmark against a stable closed GPU snapshot: {error}"
            ) from error
    receipt = discovery.load_json(receipt_path, "V8 efficiency benchmark receipt")
    if set(receipt) != _receipt_expected_keys():
        raise BenchmarkError("V8 benchmark receipt schema drifted")
    if discovery.canonical_content_sha256(receipt) != receipt.get("content_sha256"):
        raise BenchmarkError("V8 benchmark receipt content drifted")
    unit_invocation_path, unit_invocation = _validate_benchmark_invocation_binding(
        project_root=project_root.expanduser().resolve(),
        binding=receipt.get("benchmark_invocation"),
        receipt=receipt,
        usage_ledger=usage_ledger,
        execution_ledger=execution_ledger,
        gpu_lock=gpu_lock,
    )
    telemetry_binding = receipt.get("telemetry")
    if not isinstance(telemetry_binding, Mapping):
        raise BenchmarkError("V8 benchmark receipt telemetry binding is missing")
    telemetry_path = Path(str(telemetry_binding.get("path", ""))).resolve()
    if not (
        telemetry_path.is_file()
        and not telemetry_path.is_symlink()
        and telemetry_binding.get("sha256") == sha256_file(telemetry_path)
        and telemetry_binding.get("bytes") == telemetry_path.stat().st_size
    ):
        raise BenchmarkError("V8 benchmark receipt telemetry binding drifted")
    terminal_results = receipt.get("terminal_results")
    lifecycle_invocations = receipt.get("lifecycle_invocations")
    if not (
        type(receipt.get("attempt_count")) is int
        and isinstance(terminal_results, list)
        and isinstance(lifecycle_invocations, list)
        and receipt["attempt_count"] == len(terminal_results)
        and receipt["attempt_count"] == len(lifecycle_invocations)
        and receipt["attempt_count"] == 1
    ):
        raise BenchmarkError("V8 benchmark receipt attempt ownership drifted")
    canonical_commands, canonical_telemetry_path = (
        _validate_receipt_lifecycle_invocations(
            discovery,
            project_root=project_root.expanduser().resolve(),
            receipt=receipt,
            unit_invocation_path=unit_invocation_path,
            unit_invocation=unit_invocation,
            usage_ledger=usage_ledger,
            execution_ledger=execution_ledger,
        )
    )
    if telemetry_path != canonical_telemetry_path:
        raise BenchmarkError("V8 benchmark telemetry is not owned by the final attempt")
    for invocation_sha, command_sha in canonical_commands.items():
        if expected_command_sha256(
            {"invocation_sha256": invocation_sha}
        ) != command_sha:
            raise BenchmarkError("external benchmark command validator drifted")

    def canonical_expected_command(record: Mapping[str, Any]) -> str:
        invocation_sha = str(record.get("invocation_sha256", ""))
        try:
            return canonical_commands[invocation_sha]
        except KeyError as error:
            raise BenchmarkError(
                "benchmark ledger names a non-canonical execution invocation"
            ) from error

    final_invocation = terminal_results[-1].get("execution_invocation", {})
    invocation_hash = (
        final_invocation.get("sha256")
        if isinstance(final_invocation, Mapping)
        else None
    )
    if not isinstance(invocation_hash, str):
        raise BenchmarkError("V8 benchmark successful invocation binding is missing")
    telemetry = validate_worker_telemetry(
        telemetry_path, invocation_sha256=invocation_hash
    )
    steady = telemetry["trainer_telemetry"]["epochs"][1]
    if not (
        type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 1
        and receipt.get("classification")
        == "adaptive_v3r1_v8r4_target_sealed_efficiency_benchmark_completion"
        and receipt.get("campaign_id") == CAMPAIGN_ID
        and receipt.get("campaign_revision") == CAMPAIGN_REVISION
        and receipt.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and receipt.get("phase") == BENCHMARK_PHASE
        and receipt.get("benchmark_id") == BENCHMARK_ID
        and receipt.get("unit") == BENCHMARK_UNIT
        and receipt.get("usage_identity") == BENCHMARK_USAGE_IDENTITY
        and receipt.get("profile_sha256") == BENCHMARK_PROFILE_SHA256
        and type(receipt.get("epochs")) is int
        and receipt.get("epochs") == EPOCHS
        and receipt.get("epoch_1_is_warmup") is True
        and type(receipt.get("epoch_2_train_ns")) is int
        and receipt.get("epoch_2_train_ns") == steady["train_ns"]
        and type(receipt.get("epoch_2_target_free_validation_ns")) is int
        and receipt.get("epoch_2_target_free_validation_ns")
        == steady["validation_ns"]
        and type(
            receipt.get("epoch_2_train_plus_target_free_validation_ns")
        )
        is int
        and receipt.get("epoch_2_train_plus_target_free_validation_ns")
        == steady["total_ns"]
        and type(receipt.get("epoch_2_gate_ns_max")) is int
        and receipt.get("epoch_2_gate_ns_max") == STEADY_GATE_NS
        and receipt.get("gate_passed") is True
        and receipt.get("outer_test_opened") is False
        and receipt.get("accuracy_metrics_emitted_or_used") is False
        and receipt.get("checkpoint_selection_performed") is False
        and receipt.get("training_result_reusable") is False
        and receipt.get("selection_or_promotion_input") is False
        and receipt.get("artifacts_quarantined") is True
        and receipt.get("artifact_disposition")
        == "quarantined_timing_telemetry_only"
        and receipt.get("same_usage_and_execution_ledgers") is True
        and receipt.get("benchmark_before_first_active_discovery_terminal") is True
        and receipt.get(
            "all_failed_reconciled_and_successful_attempts_owned_by_one_receipt"
        )
        is True
        and receipt.get("historical_benchmark_attempts")
        == [dict(item) for item in HISTORICAL_BENCHMARK_ATTEMPTS]
        and receipt.get("historical_projection_authority")
        == _validate_historical_projection_authority(project_root)
        and receipt.get("active_scientific_attempt_count") == 1
        and receipt.get("killed_lifecycle_replay_only") is True
        and receipt.get("attempt_count") == 1
        and receipt.get("commercial_claim_authorized") is False
        and receipt.get("legacy_v8r3_success_quarantined") is True
        and receipt.get("legacy_combined_cache_used_by_active_attempt") is False
        and receipt.get("physical_nonouter_pack_only") is True
        and receipt.get("production_execution_authorized") is True
        and receipt.get("v8r4a_ledger_migration_required") is False
    ):
        raise BenchmarkError("V8 benchmark receipt invariants drifted")
    state = usage_state
    if state is not None and require_no_discovery_terminal:
        _no_discovery_terminal(state)
    discovery.validate_completion_receipt_usage(
        usage_ledger,
        receipt,
        expected_phase=BENCHMARK_PHASE,
        expected_identity=BENCHMARK_USAGE_IDENTITY,
        expected_command_sha256=canonical_expected_command,
        expected_gpu_ledger=execution_ledger,
        expected_gpu_lock=gpu_lock,
        usage_state=state,
    )
    status = receipt_path.stat()
    if (
        receipt_path.is_symlink()
        or status.st_nlink != 1
        or stat.S_IMODE(status.st_mode) != 0o444
    ):
        raise BenchmarkError("V8 benchmark receipt is not exact 0444")
    return receipt


def _projected_file_binding_identity(
    binding: Any, *, project_root: Path, require_portable_schema: bool = True
) -> tuple[Path, str, int] | None:
    """Normalize an already-sealed file projection without opening the file."""

    required = {"path", "sha256", "bytes"}
    if not isinstance(binding, Mapping) or (
        set(binding) != required
        if require_portable_schema
        else not required <= set(binding)
    ):
        return None
    raw_path = Path(str(binding.get("path", "")))
    projected_path = raw_path if raw_path.is_absolute() else project_root / raw_path
    digest = binding.get("sha256")
    size = binding.get("bytes")
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        and type(size) is int
        and size >= 0
    ):
        return None
    return Path(os.path.abspath(projected_path)), digest, size


def validate_benchmark_receipt_pack_free(
    *,
    project_root: Path,
    receipt_path: Path,
    expected_pretrain_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the V8R4 benchmark owner without opening its sealed pack.

    This entry point is for selection/aggregation processes, which are not
    granted a training-pack capability.  Scientific pack replay remains the
    responsibility of ``validate_benchmark_receipt`` in the benchmark shard.
    """

    root = project_root.expanduser().resolve()
    canonical = (root / BENCHMARK_RECEIPT_RELATIVE).resolve()
    path = receipt_path.expanduser().resolve()
    if path != canonical or path == (root / LEGACY_BENCHMARK_RECEIPT_RELATIVE).resolve():
        raise BenchmarkError("pack-free validator requires the canonical V8R4 receipt")
    status = os.stat(path, follow_symlinks=False)
    if not (
        not path.is_symlink()
        and stat.S_ISREG(status.st_mode)
        and status.st_nlink == 1
        and stat.S_IMODE(status.st_mode) == 0o444
    ):
        raise BenchmarkError("V8R4 benchmark receipt is not immutable")
    receipt = load_json(path, "V8R4 efficiency benchmark receipt")
    train_ns = receipt.get("epoch_2_train_ns")
    validation_ns = receipt.get("epoch_2_target_free_validation_ns")
    total_ns = receipt.get("epoch_2_train_plus_target_free_validation_ns")
    if not (
        set(receipt) == _receipt_expected_keys()
        and canonical_content_sha256(receipt) == receipt.get("content_sha256")
        and type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 1
        and receipt.get("classification")
        == "adaptive_v3r1_v8r4_target_sealed_efficiency_benchmark_completion"
        and receipt.get("campaign_id") == CAMPAIGN_ID
        and receipt.get("campaign_revision") == CAMPAIGN_REVISION
        and receipt.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and receipt.get("phase") == BENCHMARK_PHASE
        and receipt.get("benchmark_id") == BENCHMARK_ID
        and receipt.get("unit") == BENCHMARK_UNIT
        and receipt.get("usage_identity") == BENCHMARK_USAGE_IDENTITY
        and receipt.get("profile_sha256") == BENCHMARK_PROFILE_SHA256
        and type(receipt.get("epochs")) is int
        and receipt.get("epochs") == EPOCHS
        and receipt.get("epoch_1_is_warmup") is True
        and type(train_ns) is int
        and train_ns >= 0
        and type(validation_ns) is int
        and validation_ns >= 0
        and type(total_ns) is int
        and total_ns == train_ns + validation_ns
        and type(receipt.get("epoch_2_gate_ns_max")) is int
        and receipt.get("epoch_2_gate_ns_max") == STEADY_GATE_NS
        and total_ns <= STEADY_GATE_NS
        and receipt.get("gate_passed") is True
        and receipt.get("outer_test_opened") is False
        and receipt.get("accuracy_metrics_emitted_or_used") is False
        and receipt.get("checkpoint_selection_performed") is False
        and receipt.get("training_result_reusable") is False
        and receipt.get("selection_or_promotion_input") is False
        and receipt.get("artifacts_quarantined") is True
        and receipt.get("artifact_disposition")
        == "quarantined_timing_telemetry_only"
        and receipt.get("same_usage_and_execution_ledgers") is True
        and receipt.get("benchmark_before_first_active_discovery_terminal") is True
        and receipt.get(
            "all_failed_reconciled_and_successful_attempts_owned_by_one_receipt"
        )
        is True
        and receipt.get("legacy_v8r3_success_quarantined") is True
        and receipt.get("legacy_combined_cache_used_by_active_attempt") is False
        and receipt.get("physical_nonouter_pack_only") is True
        and receipt.get("production_execution_authorized") is True
        and receipt.get("v8r4a_ledger_migration_required") is False
        and receipt.get("commercial_claim_authorized") is False
        and receipt.get("outer_fold") == 3
        and receipt.get("seed") == 20260828
        and receipt.get("variant") == "H0_no_factor"
        and receipt.get("usage_ledger_path")
        == str(Path(os.path.abspath(root / DEFAULT_USAGE_LEDGER)))
        and receipt.get("gpu_execution_ledger_path")
        == str(Path(os.path.abspath(root / DEFAULT_EXECUTION_LEDGER)))
        and receipt.get("gpu_admission_lock_path")
        == str(Path(os.path.abspath(root / DEFAULT_GPU_LOCK)))
    ):
        raise BenchmarkError("pack-free V8R4 benchmark receipt invariants drifted")
    index_identity = _projected_file_binding_identity(
        receipt.get("training_index"), project_root=root
    )
    if index_identity != (
        Path(os.path.abspath(root / DEFAULT_TRAINING_INDEX)),
        DEFAULT_TRAINING_INDEX_SHA256,
        DEFAULT_TRAINING_INDEX_BYTES,
    ):
        raise BenchmarkError("pack-free benchmark index authority drifted")
    invocation_identity = _projected_file_binding_identity(
        receipt.get("benchmark_invocation"), project_root=root
    )
    if invocation_identity is None or invocation_identity[0] != Path(
        os.path.abspath(root / BENCHMARK_RUN_ROOT_RELATIVE / CURRENT_UNIT_INVOCATION_NAME)
    ):
        raise BenchmarkError("pack-free benchmark invocation authority drifted")
    authorization = receipt.get("pretrain_authorization")
    authorization_identity = _projected_file_binding_identity(
        authorization, project_root=root
    )
    if authorization_identity is None or authorization_identity[0] != Path(
        os.path.abspath(root / PRETRAIN_AUTHORIZATION_RELATIVE)
    ):
        raise BenchmarkError("pack-free benchmark V8R4 authorization drifted")
    if expected_pretrain_authorization is not None:
        expected_authorization_identity = _projected_file_binding_identity(
            expected_pretrain_authorization,
            project_root=root,
            require_portable_schema=False,
        )
        if authorization_identity != expected_authorization_identity:
            raise BenchmarkError("pack-free benchmark pretrain authority differs")
    projected_governance = {
        "gpu_state_migration": GPU_STATE_MIGRATION_RECEIPT_RELATIVE,
        "contract": CONTRACT_RELATIVE,
        "trainer": TRAINER_RELATIVE,
        "gpu_wrapper": WRAPPER_RELATIVE,
    }
    for field, relative in projected_governance.items():
        identity = _projected_file_binding_identity(
            receipt.get(field), project_root=root
        )
        if identity is None or identity[0] != Path(os.path.abspath(root / relative)):
            raise BenchmarkError(f"pack-free benchmark {field} projection drifted")
        if field == "contract" and identity[1] != CONTRACT_FILE_SHA256:
            raise BenchmarkError("pack-free benchmark contract authority drifted")
    capability = receipt.get("target_sealed_capability")
    capability_path = Path(str(capability.get("path", ""))) if isinstance(
        capability, Mapping
    ) else Path("")
    if not (
        isinstance(capability, Mapping)
        and set(capability) == _CAPABILITY_FILE_BINDING_KEYS
        and capability_path.is_absolute()
        and Path(os.path.abspath(capability_path))
        == Path(
            os.path.abspath(
                root / BENCHMARK_LIFECYCLE_RELATIVE / TARGET_SEALED_CAPABILITY_NAME
            )
        )
        and isinstance(capability.get("sha256"), str)
        and len(str(capability["sha256"])) == 64
        and all(
            character in "0123456789abcdef"
            for character in str(capability["sha256"])
        )
        and type(capability.get("bytes")) is int
        and int(capability["bytes"]) > 0
        and type(capability.get("st_dev")) is int
        and int(capability["st_dev"]) >= 0
        and type(capability.get("st_ino")) is int
        and int(capability["st_ino"]) > 0
        and capability.get("mode") == "0444"
    ):
        raise BenchmarkError("pack-free benchmark capability binding is absent")
    lifecycle = receipt.get("lifecycle_invocations")
    terminals = receipt.get("terminal_results")
    historical = receipt.get("historical_benchmark_attempts")
    if not (
        historical == [dict(item) for item in HISTORICAL_BENCHMARK_ATTEMPTS]
        and receipt.get("historical_projection_authority")
        == _validate_historical_projection_authority(root)
        and receipt.get("active_scientific_attempt_count") == 1
        and receipt.get("killed_lifecycle_replay_only") is True
        and type(receipt.get("attempt_count")) is int
        and int(receipt["attempt_count"]) == 1
        and isinstance(lifecycle, list)
        and isinstance(terminals, list)
        and len(lifecycle) == len(terminals) == receipt["attempt_count"]
    ):
        raise BenchmarkError("V8R4A active benchmark lifecycle cover drifted")
    invocation_binding = lifecycle[0]
    terminal = terminals[0]
    telemetry = receipt.get("telemetry")
    projected_invocation = _projected_file_binding_identity(
        invocation_binding, project_root=root
    )
    projected_result = _projected_file_binding_identity(
        terminal.get("result") if isinstance(terminal, Mapping) else None,
        project_root=root,
    )
    projected_telemetry = _projected_file_binding_identity(
        telemetry, project_root=root
    )
    if not (
        projected_invocation is not None
        and isinstance(terminal, Mapping)
        and set(terminal)
        == {"terminal_record_sha256", "execution_invocation", "result"}
        and terminal.get("execution_invocation") == invocation_binding
        and isinstance(terminal.get("terminal_record_sha256"), str)
        and len(str(terminal["terminal_record_sha256"])) == 64
        and all(
            character in "0123456789abcdef"
            for character in str(terminal["terminal_record_sha256"])
        )
        and projected_result is not None
        and projected_telemetry is not None
    ):
        raise BenchmarkError("pack-free benchmark projected lifecycle schema drifted")
    active_attempt = Path(
        os.path.abspath(root / BENCHMARK_RUN_ROOT_RELATIVE / "attempts/attempt_000")
    )
    projected_paths = {
        "invocation": projected_invocation[0],
        "result": projected_result[0],
        "telemetry": projected_telemetry[0],
    }
    if projected_paths != {
        "invocation": active_attempt / "invocation.json",
        "result": active_attempt / "GPU_TERMINAL_RESULT.json",
        "telemetry": active_attempt / "QUARANTINED_TIMING_TELEMETRY.json",
    }:
        raise BenchmarkError("pack-free benchmark projected paths are non-canonical")
    usage_hashes = receipt.get("usage_record_sha256s")
    if not (
        isinstance(usage_hashes, list)
        and usage_hashes == [terminal.get("terminal_record_sha256")]
        and receipt.get("usage_record_sha256") == terminal.get("terminal_record_sha256")
    ):
        raise BenchmarkError("pack-free benchmark terminal hash cover drifted")
    return receipt


def run_benchmark(
    args: argparse.Namespace,
    *,
    command_runner: Callable[[Sequence[str]], int] = _run_command,
    discovery_module: Any | None = None,
) -> dict[str, Any]:
    """Reconcile attempts and publish the sole quarantined benchmark receipt."""

    (
        project_root,
        python,
        trainer,
        wrapper,
        gpu_lock,
        execution_ledger,
        usage_ledger,
    ) = _canonical_parent_paths(args)
    discovery = discovery_module or _load_script(
        "hfr_v8_benchmark_discovery", project_root / DISCOVERY_SCRIPT_RELATIVE
    )
    capability_path = _under(project_root, args.target_sealed_capability_receipt)
    # The explicit runtime receipt is the first governance artifact opened.
    capability = validate_target_sealed_capability(
        project_root=project_root,
        capability_receipt=capability_path,
    )
    authorization = discovery.validate_pretrain_authorization(
        project_root,
        capability_receipt=capability_path,
        expected_phase=BENCHMARK_PHASE,
        expected_outer_fold=3,
    )
    scoped_capability = authorization.get("target_sealed_capability")
    contract_binding = authorization.get("contract_binding")
    authorization_binding = authorization.get("authorization_binding")
    if not (
        authorization.get("efficiency_benchmark_authorized") is True
        and authorization.get("scientific_campaign_revision") == CAMPAIGN_REVISION
        and authorization.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and isinstance(authorization_binding, Mapping)
        and isinstance(contract_binding, Mapping)
        and isinstance(scoped_capability, Mapping)
        and scoped_capability.get("binding") == capability["binding"]
        and Path(str(authorization_binding.get("path", ""))).name
        == PRETRAIN_AUTHORIZATION_RELATIVE.name
    ):
        raise BenchmarkError("active pretrain authorization is not exact V8R4 authority")
    run_root = _under(project_root, args.run_root)
    boundary = capability["document"].get("security_boundary", {})
    output = capability["document"].get("writable_roots", {}).get("output", {})
    migration_capability_row = capability["document"].get("governance_files", {}).get(
        "gpu_state_migration_receipt"
    )
    migration_binding = discovery._capability_governance_binding(
        project_root, capability, "gpu_state_migration_receipt"
    )
    if not (
        run_root == (project_root / BENCHMARK_RUN_ROOT_RELATIVE).resolve()
        and isinstance(output, Mapping)
        and Path(str(output.get("path", ""))).resolve() == run_root
        and isinstance(migration_capability_row, Mapping)
        and isinstance(migration_binding, Mapping)
        and isinstance(boundary, Mapping)
        and boundary.get("production_execution_authorized") is True
        and boundary.get("atomic_replace_compatible") is True
        and boundary.get("synthetic_validation_only") is False
        and boundary.get("v8r4a_ledger_migration_required") is False
    ):
        raise BenchmarkError("benchmark V8R4A production capability is not exact")
    index_path = _under(project_root, args.training_index)
    sealed_index = capability["document"].get("sealed_pack_index")
    if not (
        isinstance(sealed_index, Mapping)
        and discovery.bind_file(index_path)
        == {name: sealed_index.get(name) for name in ("path", "sha256", "bytes")}
    ):
        raise BenchmarkError("benchmark index differs from the one mounted pack")
    training, training_index_binding = discovery.load_training_index(
        project_root,
        index_path,
        outer_fold_shard=3,
    )
    training_input = training.get((3, 20260828))
    if training_input is None:
        raise BenchmarkError("fixed benchmark training input is missing")
    cache_input_binding = getattr(training_input, "cache_input_binding", None)
    proposer_stack_binding = getattr(training_input, "proposer_stack_binding", None)
    if not isinstance(cache_input_binding, Mapping) or not isinstance(
        proposer_stack_binding, Mapping
    ):
        raise BenchmarkError("benchmark input lacks canonical cache/proposer bindings")
    try:
        _, live_cache_binding = discovery.verify_training_cache_inputs(
            project_root, training_input.cache_dir, outer_fold=3
        )
        live_proposer_binding = discovery.verify_training_bound_file(
            project_root,
            training_input.proposer_stack,
            expected_sha256=str(proposer_stack_binding.get("sha256", "")),
            expected_bytes=proposer_stack_binding.get("bytes"),
        )
    except Exception as error:
        raise BenchmarkError(f"benchmark scientific input verification failed: {error}") from error
    if canonical_json_bytes(live_cache_binding) != canonical_json_bytes(
        dict(cache_input_binding)
    ) or canonical_json_bytes(live_proposer_binding) != canonical_json_bytes(
        dict(proposer_stack_binding)
    ):
        raise BenchmarkError("benchmark scientific input binding drifted")
    authorization_path, authorization_sha256 = _active_authorization_path(
        project_root, authorization
    )
    expected_usage = _under(
        project_root, Path(str(authorization.get("gpu_usage_ledger_path", "")))
    )
    expected_execution = _under(
        project_root,
        Path(str(authorization.get("gpu_execution_ledger_path", ""))),
    )
    if usage_ledger != expected_usage or execution_ledger != expected_execution:
        raise BenchmarkError("benchmark must use the active V2 lifecycle ledgers")
    for required in (python, trainer, wrapper, authorization_path):
        if not required.is_file():
            raise BenchmarkError(f"required benchmark executable/source is missing: {required}")

    historical_projection = _validate_historical_projection_authority(project_root)
    budget = discovery.gpu_budget_ledger
    expected_genesis = discovery._expected_legacy_genesis(usage_ledger)
    with budget.locked_closed_snapshot(
        usage_ledger,
        budget_ns=budget.GPU_BUDGET_NS,
        expected_legacy_genesis_sha256=expected_genesis,
    ) as entry_state:
        _validate_benchmark_entry_prefix(entry_state, project_root=project_root)

    unit_invocation_path = run_root / CURRENT_UNIT_INVOCATION_NAME
    _create_unit_invocation(
        discovery,
        path=unit_invocation_path,
        contract_binding=contract_binding,
        training_index_binding=training_index_binding,
        cache_input_binding=cache_input_binding,
        proposer_stack_binding=proposer_stack_binding,
        trainer=trainer,
        wrapper=wrapper,
        authorization_path=authorization_path,
        usage_ledger=usage_ledger,
        execution_ledger=execution_ledger,
        gpu_lock=gpu_lock,
        target_sealed_capability=capability["binding"],
        gpu_state_migration=dict(migration_binding),
    )

    attempts = _attempt_directories(run_root)
    command_by_invocation: dict[str, str] = {}
    attempt_bindings: list[dict[str, Any]] = []
    successful: tuple[Path, Path, dict[str, Any], dict[str, Any]] | None = None
    receipt_path = run_root / BENCHMARK_RECEIPT_NAME

    def expected_command(record: Mapping[str, Any]) -> str:
        invocation = str(record.get("invocation_sha256", ""))
        try:
            return command_by_invocation[invocation]
        except KeyError as error:
            raise BenchmarkError(
                "benchmark ledger names an unowned execution invocation"
            ) from error

    def active_worker_command(attempt_root: Path) -> list[str]:
        return benchmark_worker_command(
            python=python,
            project_root=project_root,
            trainer=trainer,
            cache=training_input.cache_dir,
            proposer_stack=training_input.proposer_stack,
            telemetry_output=attempt_root / "QUARANTINED_TIMING_TELEMETRY.json",
            forbidden_output_dir=attempt_root / "FORBIDDEN_REUSABLE_TRAINER_OUTPUT",
            usage_ledger=usage_ledger,
            execution_ledger=execution_ledger,
            authorization_path=authorization_path,
            authorization_sha256=authorization_sha256,
            target_sealed_capability_receipt=capability_path,
        )

    def process_attempt(
        attempt_root: Path, *, attempt_index: int, launch_if_missing: bool
    ) -> bool:
        nonlocal successful
        invocation_path = attempt_root / "invocation.json"
        result_path = attempt_root / "GPU_TERMINAL_RESULT.json"
        telemetry_path = attempt_root / "QUARANTINED_TIMING_TELEMETRY.json"
        forbidden_output = attempt_root / "FORBIDDEN_REUSABLE_TRAINER_OUTPUT"
        worker_command = active_worker_command(attempt_root)
        if not invocation_path.exists():
            raise BenchmarkError("published benchmark attempt lacks its invocation")
        _validate_execution_invocation(
            discovery,
            path=invocation_path,
            unit_invocation=unit_invocation_path,
            worker_command=worker_command,
        )
        invocation_sha256 = discovery.sha256_file(invocation_path)
        command_by_invocation[invocation_sha256] = semantic_sha256(worker_command)
        wrapper_command = admitted_wrapper_command(
            python=python,
            wrapper=wrapper,
            gpu_lock=gpu_lock,
            execution_ledger=execution_ledger,
            usage_ledger=usage_ledger,
            result_file=result_path,
            invocation_sha256=invocation_sha256,
            authorization_path=authorization_path,
            authorization_sha256=authorization_sha256,
            worker_command=worker_command,
        )
        launched = False
        wrapper_return: int | None = None
        if not result_path.exists() and launch_if_missing:
            launched = True
            wrapper_return = int(command_runner(wrapper_command))
        if not result_path.exists():
            if launched:
                raise BenchmarkError(
                    "benchmark attempt ended without a durable terminal result; rerun to reconcile"
                )
            return False
        result = budget.load_validate_terminal_result(
            result_path,
            usage_ledger=usage_ledger,
            expected_campaign_id=CAMPAIGN_ID,
            expected_phase=BENCHMARK_PHASE,
            expected_context=BENCHMARK_USAGE_IDENTITY,
            expected_command_sha256=semantic_sha256(worker_command),
            expected_invocation_sha256=invocation_sha256,
            budget_ns=budget.GPU_BUDGET_NS,
            expected_legacy_genesis_sha256=expected_genesis,
        )
        if wrapper_return is not None and int(result["wrapper_exit_code"]) != wrapper_return:
            raise BenchmarkError("benchmark wrapper/result exit code drifted")
        attempt_bindings.append(
            {
                "terminal_record_sha256": result["terminal_record_sha256"],
                "execution_invocation": discovery.bind_file(invocation_path),
                "result": discovery.bind_file(result_path),
            }
        )
        telemetry: dict[str, Any] | None = None
        if telemetry_path.exists():
            telemetry = validate_worker_telemetry(
                telemetry_path, invocation_sha256=invocation_sha256
            )
        if result.get("reusable_success") is True:
            if telemetry is None or telemetry.get("gate_passed") is not True:
                raise BenchmarkError("successful benchmark lifecycle lacks passing telemetry")
            if successful is not None:
                raise BenchmarkError("multiple successful efficiency benchmarks exist")
            successful = (invocation_path, result_path, result, telemetry)
        elif telemetry is not None and telemetry.get("gate_passed") is False:
            if launched:
                raise BenchmarkGateFailed(
                    "epoch 2 train plus target-free validation exceeded 23 seconds; "
                    "discovery remains prohibited"
                )
            # A prior completed gate failure is durable and terminal.  The
            # caller below refuses a second scientific attempt.
        elif launched:
            raise BenchmarkError(
                "benchmark attempt failed and is durably terminal; a second attempt is forbidden"
            )
        return True

    if not attempts:
        final_attempt = run_root / "attempts" / "attempt_000"
        final_command = active_worker_command(final_attempt)
        _publish_benchmark_attempt(
            run_root,
            create_invocation=lambda staged: _create_execution_invocation(
                discovery,
                path=staged,
                unit_invocation=unit_invocation_path,
                worker_command=final_command,
            ),
            validate_invocation=lambda staged: _validate_execution_invocation(
                discovery,
                path=staged,
                unit_invocation=unit_invocation_path,
                worker_command=final_command,
            ),
        )
        attempts = _attempt_directories(run_root)
    if len(attempts) != 1:
        raise BenchmarkError("exactly one active scientific benchmark attempt is allowed")
    process_attempt(attempts[0], attempt_index=0, launch_if_missing=True)
    if successful is None:
        raise BenchmarkError(
            "the sole benchmark attempt is terminal without a passing gate; "
            "a second scientific attempt is forbidden"
        )

    invocation_path, _result_path, successful_result, telemetry = successful
    lifecycle_invocations = [
        item["execution_invocation"] for item in attempt_bindings
    ]
    with budget.locked_closed_snapshot(
        usage_ledger,
        budget_ns=budget.GPU_BUDGET_NS,
        expected_legacy_genesis_sha256=expected_genesis,
    ) as usage_state:
        _no_discovery_terminal(usage_state)
        usage_fields = discovery.completion_usage_fields(
            usage_ledger,
            final_record_sha256=str(successful_result["terminal_record_sha256"]),
            expected_phase=BENCHMARK_PHASE,
            expected_identity=BENCHMARK_USAGE_IDENTITY,
            expected_command_sha256=expected_command,
            terminal_results=attempt_bindings,
            lifecycle_invocations=lifecycle_invocations,
            gpu_ledger=execution_ledger,
            gpu_lock=gpu_lock,
            usage_state=usage_state,
        )
        steady = telemetry["trainer_telemetry"]["epochs"][1]
        receipt = discovery.create_once_json(
            receipt_path,
            {
                "schema_version": 1,
                "classification": "adaptive_v3r1_v8r4_target_sealed_efficiency_benchmark_completion",
                "campaign_id": CAMPAIGN_ID,
                "campaign_revision": CAMPAIGN_REVISION,
                "infrastructure_revision": INFRASTRUCTURE_REVISION,
                "phase": BENCHMARK_PHASE,
                "benchmark_id": BENCHMARK_ID,
                "unit": BENCHMARK_UNIT,
                "usage_identity": dict(BENCHMARK_USAGE_IDENTITY),
                "profile_sha256": BENCHMARK_PROFILE_SHA256,
                "epochs": EPOCHS,
                "epoch_1_is_warmup": True,
                "epoch_2_train_ns": int(steady["train_ns"]),
                "epoch_2_target_free_validation_ns": int(steady["validation_ns"]),
                "epoch_2_train_plus_target_free_validation_ns": int(
                    steady["total_ns"]
                ),
                "epoch_2_gate_ns_max": STEADY_GATE_NS,
                "gate_passed": True,
                "outer_test_opened": False,
                "accuracy_metrics_emitted_or_used": False,
                "checkpoint_selection_performed": False,
                "training_result_reusable": False,
                "selection_or_promotion_input": False,
                "artifacts_quarantined": True,
                "artifact_disposition": "quarantined_timing_telemetry_only",
                "same_usage_and_execution_ledgers": True,
                "benchmark_before_first_active_discovery_terminal": True,
                "all_failed_reconciled_and_successful_attempts_owned_by_one_receipt": True,
                "historical_benchmark_attempts": [
                    dict(item) for item in HISTORICAL_BENCHMARK_ATTEMPTS
                ],
                "historical_projection_authority": dict(historical_projection),
                "active_scientific_attempt_count": 1,
                "killed_lifecycle_replay_only": True,
                "attempt_count": len(attempt_bindings),
                "telemetry": discovery.bind_file(
                    invocation_path.parent / "QUARANTINED_TIMING_TELEMETRY.json"
                ),
                "benchmark_invocation": discovery.bind_file(unit_invocation_path),
                "pretrain_authorization": discovery.bind_file(authorization_path),
                "target_sealed_capability": dict(capability["binding"]),
                "gpu_state_migration": dict(migration_binding),
                "contract": dict(contract_binding),
                "training_index": dict(training_index_binding),
                "trainer": discovery.bind_file(trainer),
                "gpu_wrapper": discovery.bind_file(wrapper),
                "commercial_claim_authorized": False,
                "legacy_v8r3_success_quarantined": True,
                "legacy_combined_cache_used_by_active_attempt": False,
                "physical_nonouter_pack_only": True,
                "production_execution_authorized": True,
                "v8r4a_ledger_migration_required": False,
                **usage_fields,
            },
        )
        validate_benchmark_receipt(
            discovery,
            project_root=project_root,
            receipt_path=receipt_path,
            usage_ledger=usage_ledger,
            execution_ledger=execution_ledger,
            gpu_lock=gpu_lock,
            expected_command_sha256=expected_command,
            usage_state=usage_state,
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--training-index", type=Path, default=DEFAULT_TRAINING_INDEX)
    parser.add_argument("--run-root", type=Path, default=BENCHMARK_RUN_ROOT_RELATIVE)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--trainer", type=Path, default=TRAINER_RELATIVE)
    parser.add_argument("--gpu-wrapper", type=Path, default=WRAPPER_RELATIVE)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--execution-ledger", type=Path, default=DEFAULT_EXECUTION_LEDGER)
    parser.add_argument("--usage-ledger", type=Path, default=DEFAULT_USAGE_LEDGER)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--proposer-stack", type=Path)
    parser.add_argument("--telemetry-output", type=Path)
    parser.add_argument("--trainer-output-dir", type=Path)
    parser.add_argument("--authorization-path", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument(
        "--target-sealed-capability-receipt",
        type=Path,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _canonical_parent_paths(args)
        if args.internal_worker:
            return run_internal_worker(args)
        receipt = run_benchmark(args)
    except BenchmarkGateFailed as error:
        print(json.dumps({"status": "gate_failed", "error": str(error)}, sort_keys=True))
        return GATE_FAILURE_EXIT
    except (BenchmarkError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}, sort_keys=True))
        return 87
    print(
        json.dumps(
            {
                "status": "efficiency_benchmark_complete",
                "receipt": str(
                    (_under(args.project_root.resolve(), args.run_root) / BENCHMARK_RECEIPT_NAME)
                ),
                "content_sha256": receipt["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
