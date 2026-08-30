#!/usr/bin/env python3
"""Fail-closed governance validator for adaptive-retrospective DHFER v3r1.

The superseded v3 declaration is deliberately *not* accepted as authority.
This validator binds the post-outcome v3r1 contract, its explicit governance
corrections, the v2 failure entry evidence, the discovery-only diagnostics, the
exact versioned implementation surface, tests, and a create-once source
snapshot.  Training is valid only after a separate create-once pretrain
authorization has been issued from those exact bytes.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1"
)
CONTRACT = CAMPAIGN_DIR / "ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json"
CONTRACT_FILE_SHA256 = (
    "532d150f0241d9675873368107d09adec7aeaee5e018e09537e8a340eb6fa2bd"
)
CONTRACT_FILE_BYTES = 16179
CONTRACT_CONTENT_SHA256 = (
    "6912e9760d1ab937604ba7868fe4742554804bd7179b5be2d6c8c5b34115aa2d"
)
TEST_RECEIPT = CAMPAIGN_DIR / "IMPLEMENTATION_TEST_RECEIPT_V8R3.json"
SOURCE_SNAPSHOT = CAMPAIGN_DIR / "V3R1_SOURCE_SNAPSHOT_V8R3.json"
PRETRAIN_AUTHORIZATION = CAMPAIGN_DIR / "PRETRAIN_AUTHORIZATION_V8R3.json"
V6_USAGE_LEDGER = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "campaign_gpu_usage_chain_v6.jsonl"
)
V7_GPU_EXECUTION_LEDGER = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "gpu_execution_ledger_v7.jsonl"
)
V6_USAGE_GENESIS_RECORD_SHA256 = (
    "c7b463e4db2e8d475f428dc61dfc8fa0d27910f62fb2d811e2845fcf4932e035"
)
GPU_BUDGET_NS = 36_000_000_000_000
HEARTBEAT_INTERVAL_NS = 15_000_000_000
TERMINATION_GRACE_NS = 10_000_000_000
ACCOUNTING_MARGIN_NS = 5_000_000_000
GPU_LIFECYCLE_SCHEMA_VERSION = 2
RECOVERY_MARGIN_NS = 30_000_000_000

EXPECTED_IMPLEMENTATION_PATHS = (
    "src/snn_rr/harmonic_feature_layout_v3r1.py",
    "src/snn_rr/harmonic_factor_router_models_v3r1.py",
    "scripts/train_harmonic_factor_router_snn_v3r1.py",
    "scripts/run_hfr_v3r1_discovery_campaign.py",
    "scripts/select_hfr_v3r1_common_variant.py",
    "scripts/run_fixed_hfr_v3r1_oof_campaign.py",
    "scripts/build_locked_hfr_v3r1_test_inputs.py",
    "scripts/validate_hfr_v3r1_authorization.py",
    "tests/test_harmonic_feature_layout_v3r1.py",
    "tests/test_harmonic_factor_router_models_v3r1.py",
    "tests/test_train_harmonic_factor_router_snn_v3r1.py",
    "tests/test_run_hfr_v3r1_campaign.py",
    "tests/test_locked_hfr_v3r1_oof.py",
)
ADDITIVE_RESOURCE_SAFETY_PATHS = (
    "src/snn_rr/gpu_budget_ledger.py",
    "scripts/run_gpu_admitted.py",
    "tests/test_run_gpu_admitted.py",
)
V8_EFFICIENCY_PATHS = (
    "scripts/benchmark_hfr_v3r1_efficiency.py",
    "tests/test_benchmark_hfr_v3r1_efficiency.py",
)
ALL_IMPLEMENTATION_PATHS = (
    *EXPECTED_IMPLEMENTATION_PATHS,
    *ADDITIVE_RESOURCE_SAFETY_PATHS,
    *V8_EFFICIENCY_PATHS,
)
READ_ONLY_ANCESTRY = {
    "src/snn_rr/harmonic_factor_router_v3.py": (
        "1669399c3bb3925370e8b94a1c8bc79cc489b5cd55c6e21e7ec5ed191297edbc"
    ),
    "configs/harmonic_factor_router_v3.yaml": (
        "357804255e538eda6520938ab36fb9af0efeeb928fe65ac1d1384f34b5da1669"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/"
    "RETROSPECTIVE_CAMPAIGN_CONTRACT.json": (
        "fbad12762e535e34c9e15496db983e97eb1ad26ebd7833ca6b923e7f054e9538"
    ),
    "scripts/validate_hfr_v3_contract.py": (
        "85437b4a900f8f98f1cd1977b83a8d5955022c44bc3df140c51d8e8533b79e91"
    ),
}
ENTRY_BINDINGS = {
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V2.json": (
        "6b58637398f8d9d102893b2ec0532b40d869b304387277503484d2f671335334"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V3.json": (
        "dbaa7a1d67bd0dbd8da8f78ccbbef48c38065f875533c74d77410db8d9676837"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_nonfinite_gradient_root_cause_pre_discovery.json": (
        "60071ab185534f08b48d983bf53d389379476a63b94083520f605f5d46c34221"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V4.json": (
        "87abbe30916a65f939683267c824017e0929c8921fb2985159afea74438949ca"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_amp_gradient_scale_root_cause_pre_discovery.json": (
        "04a4add8ba42a1228a91f7f794317a9864b36d4a94f24bffd5488823ba433a59"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "DISCOVERY_ATTEMPT_000_AUTHORITY_MISMATCH_QUARANTINE_V5.json": (
        "9153f3255ff0fc1fdeb9b57bdddb182995b549a035789bc7142c039c9ea19822"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V5.json": (
        "b963ab5b8cfa0fafdbae1d3ba6854e3f85e3cf9b2452d9cb789218408b9f67be"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "DISCOVERY_ATTEMPT_000_AUTHORITY_MISMATCH_FINAL_QUARANTINE_V6.json": (
        "bd7f9369b21a5187df478b08564467fa26d52851a2f9530dddd2f26d8bd4e5cd"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V6.json": (
        "15aaae6dcbbbc39c6b56d1d43ff6c4009c75f4d8ff6517e48144e7fb705b627d"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V6A.json": (
        "fd40e123c9f056ca30b84971f808724f432650bb86e1f636ecdc8df743760bd2"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_gpu_budget_crash_safety_root_cause_pre_discovery_v7.json": (
        "81a04af68e817d3e3a40c01b24f4da37c1057999540675d51b86244b4012c340"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V7.json": (
        "23797a31140b717a211ecddc9280c7a53701018f68155dcd4f8fd047163d0ccc"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v7_adversarial_crash_safety_review.json": (
        "023be635870246b00eb33678a2887b90621d2231f93afd40b5d7417247f4b4ad"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V7R1.json": (
        "f1d24a427d9722ada338faeb918c9d632e785cb55ab0ef93aa983472958fc270"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v7r1_lifecycle_reliability_review_pretrain_v7r2.json": (
        "30764103166f067611d87551fd17ef4c56110e2f90ff9fd01c4de1b5c58ce87b"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V7R2.json": (
        "1e4fc9edb2d5c5026e3d78386fb3bec76be6beeb6eb83fddd91c8aca02cda01d"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_TEST_RECEIPT_V7R2.json": (
        "b11cca58c2de7cedef0b0ea9a07af29d7fdc1d785a02edb5471a33022f4a184c"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "V3R1_SOURCE_SNAPSHOT_V7R2.json": (
        "272a59b6b4acbb7f2da1bbf50c9e4be07120fd32416a84602c2d4fe89c16fd25"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "PRETRAIN_AUTHORIZATION_V7R2.json": (
        "78d13d8a562ab9f96d9f52ab4c8fe9d7da273b670e59ed6fea4860fa54d15d22"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v8_batched_execution_reuse_and_benchmark_design.json": (
        "0fdb84cfb4468d3b6be6c2462ca86f8320fea4497880cebbe8f3f1955e1b3433"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8.json": (
        "2de81ba231a696f56c5d37256c7d157fd4e2d98d8b766f4c593cc7b048c13fe9"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_TEST_RECEIPT_V8.json": (
        "97f6a25600ee55354c9798da5ea09080d63d7f591e1ea34f7835dbd72b063951"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "V3R1_SOURCE_SNAPSHOT_V8.json": (
        "8d6745367de7154e6b547846194c3b50cfba32fe1fd3a9ecb5f46cfac144c01c"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "PRETRAIN_AUTHORIZATION_V8.json": (
        "91e949cade57da4555f522de378cabcb5f99b1515bf1f87d6e9d26606acedb2b"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v8_benchmark_import_loader_failure_pre_discovery_v8r1.json": (
        "dce110f94d27a664cfcbb866837d7c72848ac4e6422fb449ff6528161d5b6417"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R1.json": (
        "aa2380f237cc4454ea911e1e72683e47add1ffae3e5fc0c6949a28938891b726"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R1A.json": (
        "ae1f4039aff62eb9a22f2821854433e70e4574f540f97d2a05ef00c0644275d5"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R1B.json": (
        "7b98b2abfe7449baa572786e4b1b498ea5651ead36632191b668c08a973d3d1a"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_TEST_RECEIPT_V8R1.json": (
        "63ec8d67b7b7acfeaa980ed7decfcdafc234b406c27effbbc5440aa7bd9f8315"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "V3R1_SOURCE_SNAPSHOT_V8R1.json": (
        "a549862e2840f47d8b66e3ab8c443e03d242f0041a59c57872190ab690b1c422"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "PRETRAIN_AUTHORIZATION_V8R1.json": (
        "75fa878f4b63a929c5a1bf4b5fb8d5df685cef9e0e7f4f934e114ca22ecb44d4"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v8r1_immutable_invocation_resume_failure_pre_discovery_v8r2.json": (
        "78f061338c2871d690ae62f4b38ea3409bab72d372fcd789d20e6650e0548807"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v8r1_historical_validator_live_child_deadlock_pre_discovery_v8r2a.json": (
        "f9c6a3f003ff46499ec830f0c7324b1b6240a402ee770a0c743d7a2895b25a76"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R2.json": (
        "57736f56cf1aa05df88ae409fc3453a9447ed2e26048bb25b6280c4635123382"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_TEST_RECEIPT_V8R2.json": (
        "0a6bb5eb3aaed78a4dd74ebb5045d112b4ec8c94d32360c4dad2a6492cd1a90d"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "V3R1_SOURCE_SNAPSHOT_V8R2.json": (
        "5fa049bc9fb4899c7b5d3a54d0cdf3b25bbd5c8ef2924b11edfc8e38056eb0f6"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "PRETRAIN_AUTHORIZATION_V8R2.json": (
        "4918d6b4396bca694db43deb01d26cdfca43286465e811fd8695c258ab917ded"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v8r2_benchmark_cover_unit_mismatch_pre_discovery_v8r3.json": (
        "f1016757c50b24df825c05d97dff65603960001d9956ecd4263d5bc15ca0927e"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R3.json": (
        "a06cb4c89d44763fe8beb84c6248cfb7a0c0a2c752ead4185bcefab960f0421f"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/"
    "V3_PREDECLARATION_QUARANTINE_AND_V3R1_SUPERSESSION.json": (
        "ed3ecfe1690fe6a4bbf2393e070107fe689ad67cfd58438fd6d420b4056bff76"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/"
    "V2_I3_FAILURE_ENTRY_LOCK.json": (
        "899178ea0e7bce2a80e467ecd93c683ec4fe8c35c05ead87c1af8c2cdc8a3c86"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/diagnostics/"
    "v2_i3_failure_decomposition_for_adaptive_v3r1.json": (
        "2999cd866ab59fa6061f72c91b4fec677ae712be68eb13c27442224001a9b3c4"
    ),
}
FIXED_TEST_PATHS = (
    "tests/test_validate_hfr_v3_contract.py",
    "tests/test_diagnose_v2_i3_failure_for_hfr_v3.py",
    "tests/test_harmonic_feature_layout_v3r1.py",
    "tests/test_harmonic_factor_router_models_v3r1.py",
    "tests/test_train_harmonic_factor_router_snn_v3r1.py",
    "tests/test_run_hfr_v3r1_campaign.py",
    "tests/test_locked_hfr_v3r1_oof.py",
    "tests/test_run_gpu_admitted.py",
    "tests/test_benchmark_hfr_v3r1_efficiency.py",
)
FIXED_TEST_COUNT = 248


class AuthorizationError(RuntimeError):
    """A contract, source, test, or create-once authorization violation."""


def _fail(message: str) -> None:
    raise AuthorizationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        _fail(f"cannot hash {path}: {error}")
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail(f"non-canonical JSON value: {error}")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def exact_json_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""

    return canonical_bytes(observed) == canonical_bytes(expected)


def _has_exact_keys(value: Any, expected: Sequence[str]) -> bool:
    """Require an object with exactly the declared JSON keys."""

    return isinstance(value, Mapping) and set(value) == set(expected)


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: _fail(f"non-finite JSON constant {token}"),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"invalid JSON {path}: {error}")
    if not isinstance(value, dict):
        _fail(f"JSON root is not an object: {path}")
    return value


def validate_v6_usage_genesis(root: Path) -> dict[str, Any]:
    """Verify the immutable first record while allowing later ledger appends."""

    path = root / V6_USAGE_LEDGER
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        _fail(f"cannot read V6 GPU usage ledger: {error}")
    if not lines:
        _fail("V6 GPU usage ledger lacks the mandatory carry-forward genesis")
    try:
        genesis = json.loads(
            lines[0],
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: _fail(
                f"non-finite V6 GPU usage genesis constant {token}"
            ),
        )
    except json.JSONDecodeError as error:
        _fail(f"invalid V6 GPU usage genesis: {error}")
    if not isinstance(genesis, dict):
        _fail("V6 GPU usage genesis is not an object")
    recorded = genesis.get("record_sha256")
    payload = {key: value for key, value in genesis.items() if key != "record_sha256"}
    actual = semantic_sha256(payload)
    if recorded != V6_USAGE_GENESIS_RECORD_SHA256 or actual != recorded:
        _fail("V6 GPU usage genesis hash drifted")
    expected = {
        "campaign_id": "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective",
        "phase": "quarantine_carry_forward",
        "event": "forced_termination_usage_carry_forward",
        "previous_record_sha256": None,
        "elapsed_seconds": 377.0,
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
        "quarantined": True,
        "training_result_eligible_for_reuse": False,
        "completion_receipt_created": False,
        "return_code_observed": False,
        "termination_signal": "SIGTERM",
        "source_gpu_execution_ledger_sha256": (
            "dc5837af0c1f8119a70c1e673b99b9daf40578ba3944aac2be1b7fd4f9a4c362"
        ),
        "source_invocation_sha256": (
            "a6dfd8f44b1ceb186ef389fb22aef8c3a172397ebfb223944986a8660a249ad3"
        ),
    }
    for key, value in expected.items():
        if genesis.get(key) != value:
            _fail(f"V6 GPU usage genesis field drifted: {key}")
    return genesis


def validate_active_runtime_ledgers(
    root: Path, *, frozen_prefix_sizes: tuple[int, int] | None = None
) -> dict[str, Any]:
    """Validate the mutable runtime streams with the exact implementation bytes.

    These files may grow after authorization, so snapshots bind an immutable
    byte prefix rather than incorrectly freezing the whole mutable stream.
    """

    interpreter = root / ".venv/bin/python"
    program = r'''import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "src"))
from snn_rr import gpu_budget_ledger as budget

usage_path = root / sys.argv[2]
execution_path = root / sys.argv[3]
usage_state = budget.require_closed_ledger(
    usage_path,
    budget_ns=budget.GPU_BUDGET_NS,
    expected_legacy_genesis_sha256=budget.LEGACY_V1_GENESIS_RECORD_SHA256,
)
usage_raw = usage_state.raw_bytes

def load_execution_validator():
    specification = importlib.util.spec_from_file_location(
        "_v7_runtime_wrapper_validator", root / "scripts/run_gpu_admitted.py"
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load GPU execution ledger validator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module

execution_raw = b""
execution_rows = []
execution_validator = None
execution_exists = os.path.lexists(execution_path)
if execution_exists:
    resolved = budget._canonical_no_final_symlink(
        execution_path, "active GPU execution ledger"
    )
    status = os.lstat(resolved)
    if (
        stat.S_ISLNK(status.st_mode)
        or not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
    ):
        raise RuntimeError("active GPU execution ledger is aliased or non-regular")
    execution_raw = resolved.read_bytes()
    execution_validator = load_execution_validator()
    execution_rows = execution_validator._decode_execution_ledger(execution_raw)

starts = {
    row.get("lifecycle_id", row.get("job_id"))
    for row in execution_rows
    if row.get("event") == "start"
}
terminals = {
    row.get("lifecycle_id", row.get("job_id"))
    for row in execution_rows
    if row.get("event") in {"end", "wrapper_exception"}
}
result = {
    "usage_ledger": {
        "path": sys.argv[2],
        "exists": True,
        "size_bytes": len(usage_raw),
        "file_sha256": hashlib.sha256(usage_raw).hexdigest(),
        "record_count": len(usage_state.records),
        "tail_record_sha256": usage_state.tail_sha256,
        "settled_usage_ns": usage_state.settled_usage_ns,
        "open_reservations": len(usage_state.open_reservations),
    },
    "execution_ledger": {
        "path": sys.argv[3],
        # Canonicalize an absent path and an empty regular stream identically;
        # byte-prefix history cannot distinguish those two representations.
        "exists": bool(execution_raw),
        "size_bytes": len(execution_raw),
        "file_sha256": hashlib.sha256(execution_raw).hexdigest(),
        "record_count": len(execution_rows),
        "open_lifecycle_count": len(starts - terminals),
    },
}

if len(sys.argv) == 6:
    usage_prefix_size = int(sys.argv[4])
    execution_prefix_size = int(sys.argv[5])
    if not 0 <= usage_prefix_size <= len(usage_raw):
        raise RuntimeError("usage prefix size is outside the active stream")
    if not 0 <= execution_prefix_size <= len(execution_raw):
        raise RuntimeError("execution prefix size is outside the active stream")
    usage_prefix_raw = usage_raw[:usage_prefix_size]
    usage_prefix_state = budget.verify_ledger_bytes(
        usage_prefix_raw,
        budget_ns=budget.GPU_BUDGET_NS,
        expected_legacy_genesis_sha256=budget.LEGACY_V1_GENESIS_RECORD_SHA256,
    )
    if usage_prefix_state.open_reservations:
        raise RuntimeError("authorized usage prefix contains open reservations")
    execution_prefix_raw = execution_raw[:execution_prefix_size]
    if execution_validator is None:
        execution_validator = load_execution_validator()
    execution_prefix_rows = execution_validator._decode_execution_ledger(
        execution_prefix_raw
    )
    prefix_starts = {
        row.get("lifecycle_id", row.get("job_id"))
        for row in execution_prefix_rows
        if row.get("event") == "start"
    }
    prefix_terminals = {
        row.get("lifecycle_id", row.get("job_id"))
        for row in execution_prefix_rows
        if row.get("event") in {"end", "wrapper_exception"}
    }
    result["frozen_prefix"] = {
        "usage_ledger": {
            "path": sys.argv[2],
            "exists": True,
            "size_bytes": len(usage_prefix_raw),
            "file_sha256": hashlib.sha256(usage_prefix_raw).hexdigest(),
            "record_count": len(usage_prefix_state.records),
            "tail_record_sha256": usage_prefix_state.tail_sha256,
            "settled_usage_ns": usage_prefix_state.settled_usage_ns,
            "open_reservations": len(usage_prefix_state.open_reservations),
        },
        "execution_ledger": {
            "path": sys.argv[3],
            "exists": bool(execution_prefix_raw),
            "size_bytes": len(execution_prefix_raw),
            "file_sha256": hashlib.sha256(execution_prefix_raw).hexdigest(),
            "record_count": len(execution_prefix_rows),
            "open_lifecycle_count": len(prefix_starts - prefix_terminals),
        },
    }

print(json.dumps(result, sort_keys=True))
'''
    command = [
        str(interpreter),
        "-c",
        program,
        str(root),
        V6_USAGE_LEDGER.as_posix(),
        V7_GPU_EXECUTION_LEDGER.as_posix(),
    ]
    if frozen_prefix_sizes is not None:
        usage_prefix_size, execution_prefix_size = frozen_prefix_sizes
        command.extend((str(usage_prefix_size), str(execution_prefix_size)))
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        _fail("active runtime ledger validation failed:\n" + completed.stdout[-12000:])
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        _fail(f"active runtime ledger validator returned invalid JSON: {error}")
    if not isinstance(result, dict):
        _fail("active runtime ledger validator result is not an object")
    usage = result.get("usage_ledger", {})
    execution = result.get("execution_ledger", {})
    if not (
        usage.get("path") == V6_USAGE_LEDGER.as_posix()
        and usage.get("exists") is True
        and type(usage.get("record_count")) is int
        and usage.get("record_count", 0) >= 1
        and usage.get("tail_record_sha256")
        and type(usage.get("settled_usage_ns")) is int
        and usage.get("settled_usage_ns", -1) >= 377_000_000_000
        and type(usage.get("open_reservations")) is int
        and usage.get("open_reservations") == 0
        and execution.get("path") == V7_GPU_EXECUTION_LEDGER.as_posix()
        and type(execution.get("exists")) is bool
        and type(execution.get("record_count")) is int
        and execution.get("record_count", -1) >= 0
        and type(execution.get("open_lifecycle_count")) is int
        and execution.get("open_lifecycle_count") == 0
    ):
        _fail("active runtime ledger validation summary drifted")
    return result


def verify_runtime_ledger_prefixes(
    root: Path, frozen: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the active canonical streams to retain both frozen prefixes."""

    if set(frozen) != {"usage_ledger", "execution_ledger"}:
        _fail("runtime prefix document keys drifted")
    expected_usage = frozen.get("usage_ledger")
    expected_execution = frozen.get("execution_ledger")
    if not isinstance(expected_usage, Mapping) or not isinstance(
        expected_execution, Mapping
    ):
        _fail("runtime prefix summaries must be objects")
    usage_prefix_size = expected_usage.get("size_bytes")
    execution_prefix_size = expected_execution.get("size_bytes")
    if (
        type(usage_prefix_size) is not int
        or usage_prefix_size <= 0
        or type(execution_prefix_size) is not int
        or execution_prefix_size < 0
    ):
        _fail("runtime prefix sizes are invalid")
    result = validate_active_runtime_ledgers(
        root,
        frozen_prefix_sizes=(usage_prefix_size, execution_prefix_size),
    )
    derived_prefix = result.get("frozen_prefix")
    if not exact_json_equal(frozen, derived_prefix):
        _fail("runtime prefix derived summary drifted")
    current = {
        "usage_ledger": result["usage_ledger"],
        "execution_ledger": result["execution_ledger"],
    }
    for key in ("usage_ledger", "execution_ledger"):
        expected = frozen.get(key, {})
        observed = current.get(key, {})
        if not isinstance(expected, Mapping) or expected.get("path") != observed.get(
            "path"
        ):
            _fail(f"runtime {key} prefix path drifted")
        prefix_size = expected.get("size_bytes")
        prefix_sha = expected.get("file_sha256")
        if type(prefix_size) is not int or prefix_size < 0 or not isinstance(
            prefix_sha, str
        ):
            _fail(f"runtime {key} prefix binding is invalid")
        path = root / str(expected["path"])
        if not os.path.lexists(path):
            raw = b""
        else:
            if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
                _fail(f"runtime {key} prefix path is aliased or non-regular")
            raw = path.read_bytes()
        if len(raw) < prefix_size:
            _fail(f"runtime {key} was truncated before its authorized prefix")
        if hashlib.sha256(raw[:prefix_size]).hexdigest() != prefix_sha:
            _fail(f"runtime {key} authorized byte prefix drifted")
    return current


def verify_admitted_runtime_prefix_bytes(
    root: Path,
    frozen: Mapping[str, Any],
    admitted_binding: Mapping[str, Any],
) -> None:
    """Verify V8 snapshot prefixes inside one wrapper-authenticated live child.

    The wrapper consumer has already canonically reduced the current ledgers and
    proved that exactly this child owns their sole live lifecycle.  Requiring a
    globally closed reducer here would reject that authorized reservation, so
    this layer only rechecks the immutable historical byte prefixes and their
    exact ledger identities.
    """

    if admitted_binding.get("classification") != (
        "verified_v8_gpu_admitted_child_lifecycle"
    ):
        _fail("admitted-child lifecycle binding is not verified")
    expected_paths = {
        "usage_ledger": V6_USAGE_LEDGER,
        "execution_ledger": V7_GPU_EXECUTION_LEDGER,
    }
    if set(frozen) != set(expected_paths):
        _fail("admitted runtime prefix document keys drifted")
    for key, relative in expected_paths.items():
        expected = frozen.get(key)
        if not isinstance(expected, Mapping):
            _fail(f"admitted runtime {key} prefix is not an object")
        size = expected.get("size_bytes")
        digest = expected.get("file_sha256")
        if (
            expected.get("path") != relative.as_posix()
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            _fail(f"admitted runtime {key} prefix binding is invalid")
        path = root / relative
        binding_path_key = (
            "usage_ledger_path" if key == "usage_ledger" else "execution_ledger_path"
        )
        if admitted_binding.get(binding_path_key) != str(path.resolve()):
            _fail(f"admitted runtime {key} path differs from the wrapper binding")
        try:
            raw = b"" if not os.path.lexists(path) else path.read_bytes()
        except OSError as error:
            _fail(f"cannot read admitted runtime {key}: {error}")
        if len(raw) < size or hashlib.sha256(raw[:size]).hexdigest() != digest:
            _fail(f"admitted runtime {key} historical byte prefix drifted")
        live_prefix_size = admitted_binding.get(
            "usage_ledger_prefix_bytes"
            if key == "usage_ledger"
            else "execution_ledger_prefix_bytes"
        )
        live_prefix_sha = admitted_binding.get(
            "usage_ledger_prefix_sha256"
            if key == "usage_ledger"
            else "execution_ledger_prefix_sha256"
        )
        if (
            type(live_prefix_size) is not int
            or live_prefix_size < size
            or live_prefix_size > len(raw)
            or not isinstance(live_prefix_sha, str)
            or hashlib.sha256(raw[:live_prefix_size]).hexdigest() != live_prefix_sha
        ):
            _fail(f"admitted runtime {key} live prefix drifted")


def _load_gpu_admitted_validator(root: Path) -> Any:
    """Load the snapshotted wrapper implementation used to mint capabilities."""

    path = root / "scripts/run_gpu_admitted.py"
    specification = importlib.util.spec_from_file_location(
        "hfr_v3r1_v8_admitted_binding_validator", path
    )
    if specification is None or specification.loader is None:
        _fail("cannot load the V8 GPU admitted-child validator")
    module = importlib.util.module_from_spec(specification)
    prior_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    except BaseException as error:
        _fail(f"cannot load the V8 GPU admitted-child validator: {error}")
    finally:
        sys.dont_write_bytecode = prior_dont_write_bytecode
    return module


def revalidate_admitted_child_binding(
    root: Path, admitted_binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-prove a consumed wrapper capability before trusting any of its fields."""

    if not isinstance(admitted_binding, Mapping):
        _fail("admitted-child binding must be an object")
    phase = admitted_binding.get("phase")
    if phase not in {
        "efficiency_benchmark",
        "discovery",
        "promotion_training",
        "promotion_prediction",
    }:
        _fail("admitted-child phase is outside the V8 campaign")
    authorization_path = root / PRETRAIN_AUTHORIZATION
    require_frozen_regular_file(authorization_path, "pretrain authorization")
    module = _load_gpu_admitted_validator(root)
    entry = getattr(module, "revalidate_consumed_admitted_child_binding", None)
    if not callable(entry):
        _fail("V8 GPU admitted-child revalidation entry point is missing")
    try:
        verified = entry(
            admitted_binding,
            expected_campaign_id=(
                "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
            ),
            expected_phase=str(phase),
            expected_gpu_lock_file=(
                root
                / "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
                "gpu_admission_v7.lock"
            ),
            expected_usage_ledger=root / V6_USAGE_LEDGER,
            expected_execution_ledger=root / V7_GPU_EXECUTION_LEDGER,
            expected_authorization_path=authorization_path,
            expected_authorization_sha256=sha256_file(authorization_path),
        )
    except BaseException as error:
        _fail(f"GPU admitted-child capability revalidation failed: {error}")
    if not isinstance(verified, Mapping) or not exact_json_equal(
        verified, admitted_binding
    ):
        _fail("GPU admitted-child revalidator returned different binding bytes")
    return dict(verified)


def verify_content_hash(document: Mapping[str, Any], *, path: Path) -> str:
    recorded = document.get("content_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        _fail(f"missing canonical content_sha256: {path}")
    payload = {key: value for key, value in document.items() if key != "content_sha256"}
    actual = semantic_sha256(payload)
    if actual != recorded:
        _fail(f"canonical content hash mismatch: {path}")
    return actual


def bind_file(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        _fail(f"required regular file is missing or symlinked: {relative}")
    info = path.stat()
    return {
        "path": relative,
        "file_sha256": sha256_file(path),
        "size_bytes": info.st_size,
        "mode": stat.S_IMODE(info.st_mode),
    }


def require_frozen_regular_file(path: Path, label: str) -> None:
    try:
        status = os.lstat(path)
    except OSError as error:
        _fail(f"{label} is unavailable: {error}")
    if (
        stat.S_ISLNK(status.st_mode)
        or not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or stat.S_IMODE(status.st_mode) != 0o444
    ):
        _fail(f"{label} must be one single-link regular 0444 file: {path}")


def _verify_bound_path(root: Path, relative: str, expected: str) -> None:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        _fail(f"bound file missing or symlinked: {relative}")
    actual = sha256_file(path)
    if actual != expected:
        _fail(f"bound file drifted: {relative}")


def validate_contract(root: Path) -> dict[str, Any]:
    path = root / CONTRACT
    if sha256_file(path) != CONTRACT_FILE_SHA256:
        _fail("v3r1 contract byte hash drifted")
    contract = load_json(path)
    if verify_content_hash(contract, path=path) != CONTRACT_CONTENT_SHA256:
        _fail("v3r1 contract semantic hash drifted")
    expected = list(EXPECTED_IMPLEMENTATION_PATHS)
    actual = contract.get("implementation_authorization", {}).get(
        "allowed_new_or_modified_paths"
    )
    if actual != expected:
        _fail("contract implementation path allowlist drifted")
    required = {
        "campaign_id": "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective",
        "status": "implementation_authorized_training_not_yet_authorized",
        "classification": "adaptive_retrospective_historical_cohort_engineering_not_confirmatory",
    }
    for key, expected_value in required.items():
        if contract.get(key) != expected_value:
            _fail(f"contract field drifted: {key}")
    governance = contract.get("governance", {})
    if not (
        governance.get("old_v3_predeclaration_is_quarantined") is True
        and governance.get("old_v3_contract_may_authorize_this_campaign") is False
        and governance.get("this_contract_was_created_after_v2_locked_outcomes_were_observed") is True
        and governance.get("commercial_or_production_claim_allowed") is False
    ):
        _fail("v3r1 governance/claim boundary drifted")
    return contract


def _fixed_gpu_protocol() -> dict[str, Any]:
    return {
        "schema_version": GPU_LIFECYCLE_SCHEMA_VERSION,
        "usage_ledger_path": V6_USAGE_LEDGER.as_posix(),
        "gpu_budget_ns": GPU_BUDGET_NS,
        "heartbeat_interval_ns": HEARTBEAT_INTERVAL_NS,
        "termination_grace_ns": TERMINATION_GRACE_NS,
        "accounting_margin_ns": ACCOUNTING_MARGIN_NS,
        "recovery_margin_ns": RECOVERY_MARGIN_NS,
        "legacy_v1_genesis_record_sha256": V6_USAGE_GENESIS_RECORD_SHA256,
        "uncertain_crash_charge": "full_reservation",
        "open_reservations_allowed_at_unit_or_campaign_completion": False,
    }


def _validate_v7_resource_safety_authorization(root: Path) -> None:
    diagnostic_path = (
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
        "v3r1_gpu_budget_crash_safety_root_cause_pre_discovery_v7.json"
    )
    correction_path = (
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V7.json"
    )
    diagnostic = load_json(diagnostic_path)
    correction = load_json(correction_path)

    lifecycle = diagnostic.get("required_lifecycle_protocol", {})
    containment = diagnostic.get("containment", {})
    durability = diagnostic.get("ledger_durability", {})
    reconciliation = diagnostic.get("crash_reconciliation", {})
    decision = diagnostic.get("decision", {})
    claim = diagnostic.get("claim_boundary", {})
    diagnostic_bindings = diagnostic.get("evidence_bindings", {})
    failures = diagnostic.get("confirmed_failures", [])
    failure_names = [
        item.get("failure") for item in failures if isinstance(item, Mapping)
    ]
    expected_failures = [
        "usage_is_appended_only_after_child_return",
        "wrapper_and_trainer_can_outlive_campaign_parent",
        "remaining_budget_is_given_entirely_to_workload_before_termination_grace",
        "budget_read_launch_and_terminal_append_are_not_one_admission_transaction",
        "hash_tail_read_and_jsonl_append_are_unlocked_and_non_atomic",
        "successful_terminal_before_completion_receipt_is_not_recoverable",
    ]
    if not (
        diagnostic.get("classification")
        == "pre_discovery_gpu_budget_crash_safety_root_cause_diagnostic"
        and diagnostic_bindings.get("v6_pretrain_authorization", {}).get(
            "file_sha256"
        )
        == sha256_file(root / CAMPAIGN_DIR / "PRETRAIN_AUTHORIZATION_V6.json")
        and diagnostic_bindings.get("v6_source_snapshot", {}).get("file_sha256")
        == sha256_file(root / CAMPAIGN_DIR / "V3R1_SOURCE_SNAPSHOT_V6.json")
        and diagnostic_bindings.get("final_incident_quarantine", {}).get(
            "file_sha256"
        )
        == sha256_file(
            root
            / CAMPAIGN_DIR
            / "DISCOVERY_ATTEMPT_000_AUTHORITY_MISMATCH_FINAL_QUARANTINE_V6.json"
        )
        and diagnostic_bindings.get("v6_usage_genesis", {}).get(
            "first_record_sha256"
        )
        == V6_USAGE_GENESIS_RECORD_SHA256
        and diagnostic_bindings.get("v6_usage_genesis", {}).get(
            "settled_elapsed_seconds"
        )
        == 377.0
        and failure_names == expected_failures
        and lifecycle.get("schema_version") == GPU_LIFECYCLE_SCHEMA_VERSION
        and lifecycle.get("event_order")
        == [
            "gpu_lock_acquired",
            "open_reservations_reconciled",
            "durable_reservation_created",
            "child_process_group_spawned",
            "periodic_heartbeat",
            "child_group_reaped",
            "durable_terminal_created",
            "result_receipt_created",
            "gpu_lock_released",
        ]
        and lifecycle.get("budget_nanoseconds") == GPU_BUDGET_NS
        and lifecycle.get("heartbeat_seconds")
        == HEARTBEAT_INTERVAL_NS // 1_000_000_000
        and lifecycle.get("termination_grace_seconds")
        == TERMINATION_GRACE_NS // 1_000_000_000
        and lifecycle.get("accounting_margin_seconds")
        == ACCOUNTING_MARGIN_NS // 1_000_000_000
        and lifecycle.get("authoritative_clock") == "monotonic_ns"
        and lifecycle.get("maximum_workload_timeout")
        == "remaining_budget_minus_termination_grace_and_accounting_margin"
        and lifecycle.get("core_invariant")
        == "settled_usage_ns plus open_reservation_ns is never greater than budget_ns"
        and lifecycle.get("child_may_spawn_before_reservation_fsync") is False
        and lifecycle.get("gpu_lock_may_release_before_terminal_fsync") is False
        and containment.get("wrapper_parent_death_signal_required") is True
        and containment.get("trainer_parent_death_signal_required") is True
        and containment.get("trainer_separate_process_group_required") is True
        and containment.get("signal_forwarding_and_reap_required") is True
        and containment.get("term_ignoring_child_escalated_to_sigkill") is True
        and containment.get("pid_identity_must_include_start_ticks") is True
        and durability.get("dedicated_stable_lock_inode_required") is True
        and durability.get("lock_order")
        == ["gpu_admission_lock", "usage_ledger_lock"]
        and durability.get("mixed_v1_v2_required") is True
        and durability.get("v1_quarantine_genesis_must_not_change") is True
        and durability.get("reservation_heartbeat_terminal_hash_binding_required")
        is True
        and durability.get("open_reservation_allowed_at_campaign_seal") is False
        and durability.get("append_implementation")
        == [
            "verify_entire_existing_chain_under_lock",
            "write_old_bytes_plus_new_record_to_same_directory_temp",
            "fsync_temp_file",
            "atomic_replace",
            "fsync_parent_directory",
        ]
        and reconciliation.get("same_boot_proven_dead_process")
        == "charge_last_heartbeat_conservative_ceiling_plus_recovery_margin"
        and reconciliation.get("different_boot_or_unprovable_process_identity")
        == "charge_full_reservation"
        and reconciliation.get("reconciled_terminal_reuse_eligible") is False
        and reconciliation.get("successful_terminal_without_campaign_receipt")
        == "recover_output_and_terminal_without_gpu_rerun_after_full_validation"
        and decision.get("gpu_discovery_relaunch_safe_under_v6") is False
        and decision.get("goal_should_be_terminated") is False
        and decision.get("crash_safety_correction_required_before_any_new_gpu_training")
        is True
        and decision.get("outer_test_must_remain_sealed") is True
        and decision.get("commercial_claim_authorized") is False
        and claim.get("accuracy_or_commercial_target_result_used") is False
        and claim.get("outer_test_features_or_targets_opened") is False
        and claim.get("adaptive_retrospective_only") is True
        and claim.get("commercial_claim_authorized") is False
    ):
        _fail("V7 GPU-budget crash-safety diagnostic drifted")

    diagnostic_evidence = correction.get("diagnostic_evidence", {})
    extension = correction.get("additive_resource_safety_surface_extension", {})
    fixed = correction.get("fixed_protocol", {})
    recovery = correction.get("required_recovery_behavior", {})
    forbidden = correction.get("forbidden_changes", {})
    reauthorization = correction.get("required_reauthorization", {})
    correction_claim = correction.get("claim_boundary", {})
    superseded = correction.get("superseded_active_pretrain_chain", {})
    authorized = correction.get("authorized_modifications", [])
    authorized_paths = [
        item.get("path") for item in authorized if isinstance(item, Mapping)
    ]
    expected_authorized_paths = [
        "src/snn_rr/gpu_budget_ledger.py",
        "scripts/run_gpu_admitted.py",
        "scripts/run_hfr_v3r1_discovery_campaign.py",
        "scripts/run_fixed_hfr_v3r1_oof_campaign.py",
        "tests/test_run_gpu_admitted.py",
        "tests/test_run_hfr_v3r1_campaign.py",
        "tests/test_locked_hfr_v3r1_oof.py",
        "scripts/validate_hfr_v3r1_authorization.py",
    ]
    authorized_by_path = {
        item.get("path"): item for item in authorized if isinstance(item, Mapping)
    }
    if not (
        correction.get("classification")
        == "pre_discovery_adaptive_v3r1_crash_safe_gpu_budget_lifecycle_correction_authorization"
        and diagnostic_evidence.get("file_sha256") == sha256_file(diagnostic_path)
        and diagnostic_evidence.get("content_sha256")
        == diagnostic.get("content_sha256")
        and superseded.get("implementation_test_receipt", {}).get("file_sha256")
        == sha256_file(root / CAMPAIGN_DIR / "IMPLEMENTATION_TEST_RECEIPT_V6.json")
        and superseded.get("source_snapshot", {}).get("file_sha256")
        == sha256_file(root / CAMPAIGN_DIR / "V3R1_SOURCE_SNAPSHOT_V6.json")
        and superseded.get("pretrain_authorization", {}).get("file_sha256")
        == sha256_file(root / CAMPAIGN_DIR / "PRETRAIN_AUTHORIZATION_V6.json")
        and superseded.get("may_authorize_new_gpu_execution") is False
        and superseded.get("preserve_as_immutable_audit_evidence") is True
        and extension.get("base_contract_implementation_allowlist_unchanged") is True
        and extension.get("new_paths") == ["src/snn_rr/gpu_budget_ledger.py"]
        and extension.get("existing_paths_newly_governed")
        == ["scripts/run_gpu_admitted.py", "tests/test_run_gpu_admitted.py"]
        and extension.get("scientific_surface_expanded") is False
        and extension.get("resource_safety_surface_expanded") is True
        and authorized_paths == expected_authorized_paths
        and authorized_by_path.get("src/snn_rr/gpu_budget_ledger.py", {}).get(
            "before_state"
        )
        == "absent"
        and authorized_by_path.get("scripts/run_gpu_admitted.py", {}).get(
            "before_sha256"
        )
        == "3cb7c5ef83bcc5a9a0a275907023af2cc9b01e6a8ebc7fd3c35f5d2389e6ef8a"
        and authorized_by_path.get(
            "scripts/run_hfr_v3r1_discovery_campaign.py", {}
        ).get("before_sha256")
        == "c42b845f1d46ef855a9b0d22c4c17d632863af1a2e8bac28c37325bff02adef8"
        and authorized_by_path.get(
            "scripts/run_fixed_hfr_v3r1_oof_campaign.py", {}
        ).get("before_sha256")
        == "7afbaf66bf70e15e552605e579dc32c58d316f0f867ff7037fb1f2c9497e49c3"
        and authorized_by_path.get("tests/test_run_gpu_admitted.py", {}).get(
            "before_sha256"
        )
        == "8e9f421b2fcb9dd64f9b3f713aaf1d271475e1b44122c9b202f44b7eb31e6e20"
        and authorized_by_path.get(
            "scripts/validate_hfr_v3r1_authorization.py", {}
        ).get("before_sha256")
        == "0f5e3e8f5bc2bf42f159b0a91ed7f2eb14eaab55b4acba8a714c795f03eaa771"
        and fixed.get("gpu_budget_ns") == GPU_BUDGET_NS
        and fixed.get("heartbeat_interval_ns") == HEARTBEAT_INTERVAL_NS
        and fixed.get("termination_grace_ns") == TERMINATION_GRACE_NS
        and fixed.get("accounting_margin_ns") == ACCOUNTING_MARGIN_NS
        and fixed.get("legacy_v1_genesis_record_sha256")
        == V6_USAGE_GENESIS_RECORD_SHA256
        and fixed.get("legacy_v1_genesis_must_remain_first") is True
        and fixed.get("uncertain_crash_charge") == "full_reservation"
        and fixed.get("same_boot_proven_dead_recovery_ceiling")
        == "last_heartbeat_conservative_ceiling_plus_30_seconds_bounded_by_reservation"
        and fixed.get("lock_order")
        == ["gpu_admission_lock", "stable_usage_ledger_lock"]
        and fixed.get("open_reservations_allowed_at_unit_or_campaign_completion")
        is False
        and recovery.get("reservation_must_be_durable_before_child_spawn") is True
        and recovery.get("wrapper_and_child_parent_death_containment") is True
        and recovery.get("signal_forward_child_group_and_reap") is True
        and recovery.get("terminal_must_be_durable_before_gpu_lock_release") is True
        and recovery.get("terminal_before_campaign_receipt_recovered_without_gpu_rerun")
        is True
        and recovery.get("unprovable_process_identity_charged_conservatively")
        is True
        and recovery.get("torn_or_forked_chain_fails_closed") is True
        and all(value is True for value in forbidden.values())
        and set(forbidden)
        == {
            "model_architecture_forward_loss_schedule_or_optimizer_change",
            "data_cache_split_window_target_or_outer_test_change",
            "training_matrix_minimum_epochs_patience_or_gpu_cap_change",
            "selection_release_decoder_metric_or_threshold_change",
            "quarantined_attempt_reuse",
            "scientific_result_driven_change",
            "commercial_or_confirmatory_claim",
        }
        and reauthorization.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V7.json"
        and reauthorization.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V7.json"
        and reauthorization.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V7.json"
        and reauthorization.get("all_base_and_additive_implementation_files_frozen_0444")
        is True
        and reauthorization.get("all_fixed_and_resource_safety_tests_pass") is True
        and reauthorization.get("no_gpu_training_before_v7_pretrain_validation")
        is True
        and correction_claim.get("adaptive_retrospective_only") is True
        and correction_claim.get("accuracy_or_commercial_target_result_used")
        is False
        and correction_claim.get("outer_test_opened") is False
        and correction_claim.get("commercial_claim_authorized") is False
    ):
        _fail("V7 crash-safe resource authorization drifted")

    for path in (diagnostic_path, correction_path):
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            _fail(f"V7 crash-safety governance evidence became writable: {path}")


def _validate_v7r1_adversarial_authorization(root: Path) -> None:
    review_path = (
        root
        / CAMPAIGN_DIR
        / "diagnostics/v3r1_v7_adversarial_crash_safety_review.json"
    )
    correction_path = (
        root / CAMPAIGN_DIR / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V7R1.json"
    )
    parent_path = root / CAMPAIGN_DIR / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V7.json"
    review = load_json(review_path)
    correction = load_json(correction_path)
    parent = load_json(parent_path)

    reviewed = review.get("reviewed_implementation", {})
    expected_reviewed = {
        "gpu_budget_ledger": (
            "src/snn_rr/gpu_budget_ledger.py",
            "448547725503c93db2cc455435a925e9ebd9aa8097ae615839aaebf39db6f043",
        ),
        "gpu_wrapper": (
            "scripts/run_gpu_admitted.py",
            "cda8d305c422d85ea1ae6ba9cad92db38ab57b5632d4bb8015ab55902fe245ea",
        ),
        "discovery_runner": (
            "scripts/run_hfr_v3r1_discovery_campaign.py",
            "a76ade74db88660544b2d13918b4f2cfd7da0022bee5f83d02ac758cd09791e8",
        ),
        "fixed_promotion_runner": (
            "scripts/run_fixed_hfr_v3r1_oof_campaign.py",
            "53dbc530fe51a99d879077b2357bddb26f4c66b89ffe34a00466f89bf260e2ce",
        ),
        "selector": (
            "scripts/select_hfr_v3r1_common_variant.py",
            "b235445d3f79abc3a17bce92b41da5180591eb506715acb3fef9e73bc05b4865",
        ),
        "validator": (
            "scripts/validate_hfr_v3r1_authorization.py",
            "e480aa2106181a7bd733fcbfab12ba82c18e4cf9d7b9aae778f2d58d5b83d643",
        ),
    }
    reviewed_matches = (
        set(reviewed) == set(expected_reviewed)
        and all(
            reviewed.get(name, {}).get("path") == expected[0]
            and reviewed.get(name, {}).get("file_sha256") == expected[1]
            for name, expected in expected_reviewed.items()
        )
    )
    findings = review.get("findings", [])
    finding_severities = {
        item.get("id"): item.get("severity")
        for item in findings
        if isinstance(item, Mapping)
    }
    expected_regressions = [
        "wrapper SIGSTOP while child runs is bounded by an independent watchdog",
        "same-boot dead reconciliation accounts current monotonic elapsed",
        "same-boot matching live process fails busy without changing ledger bytes",
        "raw noncanonical whitespace, CRLF, and alternate escapes are rejected",
        "every lifecycle and result binds one absolute GPU execution ledger path",
        "torn execution end recovers only from the exact durable usage terminal",
        "ancestor symlink, directory replacement, and protected hard-link aliases fail closed",
        "discovery seal rejects unexplained appended terminal",
        "concurrent append cannot enter between exact-cover reconciliation and seal binding",
        "selector revalidates the historical prefix and all 18 terminal receipts before ranking",
    ]
    review_decision = review.get("decision", {})
    review_claim = review.get("claim_boundary", {})
    review_method = review.get("review_method", {})
    if not (
        review.get("classification")
        == "pre_discovery_v7_adversarial_resource_safety_review"
        and reviewed_matches
        and finding_severities
        == {
            "V7-R1": "high",
            "V7-R2": "high",
            "V7-R3": "high",
            "V7-R4": "high",
            "V7-R5": "medium",
            "V7-R6": "medium",
            "V7-R7": "medium",
            "V7-R8": "medium",
        }
        and review.get("mandatory_regressions") == expected_regressions
        and review_method.get("outer_test_opened") is False
        and review_method.get("accuracy_values_used") is False
        and review_decision.get(
            "v7_pretrain_authorization_may_be_issued_from_reviewed_bytes"
        )
        is False
        and review_decision.get("gpu_training_may_resume") is False
        and review_decision.get("goal_should_be_terminated") is False
        and review_decision.get("corrected_v7_chain_required") is True
        and review_decision.get("commercial_claim_authorized") is False
        and review_claim.get("adaptive_retrospective_only") is True
        and review_claim.get("outer_test_features_or_targets_opened") is False
        and review_claim.get("accuracy_or_commercial_target_result_used") is False
        and review_claim.get("confirmatory") is False
        and review_claim.get("commercial_claim_authorized") is False
    ):
        _fail("V7R1 adversarial review drifted")

    parent_binding = correction.get("parent_correction", {})
    review_binding = correction.get("adversarial_review", {})
    authorized = correction.get("authorized_modifications", [])
    authorized_paths = [
        item.get("path") for item in authorized if isinstance(item, Mapping)
    ]
    authorized_by_path = {
        item.get("path"): item for item in authorized if isinstance(item, Mapping)
    }
    expected_before_hashes = {
        "src/snn_rr/gpu_budget_ledger.py": (
            "448547725503c93db2cc455435a925e9ebd9aa8097ae615839aaebf39db6f043"
        ),
        "scripts/run_gpu_admitted.py": (
            "cda8d305c422d85ea1ae6ba9cad92db38ab57b5632d4bb8015ab55902fe245ea"
        ),
        "scripts/run_hfr_v3r1_discovery_campaign.py": (
            "a76ade74db88660544b2d13918b4f2cfd7da0022bee5f83d02ac758cd09791e8"
        ),
        "scripts/run_fixed_hfr_v3r1_oof_campaign.py": (
            "53dbc530fe51a99d879077b2357bddb26f4c66b89ffe34a00466f89bf260e2ce"
        ),
        "scripts/select_hfr_v3r1_common_variant.py": (
            "b235445d3f79abc3a17bce92b41da5180591eb506715acb3fef9e73bc05b4865"
        ),
        "tests/test_run_gpu_admitted.py": (
            "5347c97aefaf7e963e0843dbe7f84e70a7931f262eea1fa6a32be19ef1b18300"
        ),
        "tests/test_run_hfr_v3r1_campaign.py": (
            "6b712bc8646396664c270b8fa4a2d60208cae3e4b1c784c8fb6b1b9f49a374ce"
        ),
        "tests/test_locked_hfr_v3r1_oof.py": (
            "e81dc56ff208127af02e4c2141cfd7ce0d0979592ef1efa06d6d21b8ed783171"
        ),
        "scripts/validate_hfr_v3r1_authorization.py": (
            "e480aa2106181a7bd733fcbfab12ba82c18e4cf9d7b9aae778f2d58d5b83d643"
        ),
    }
    corrected = correction.get("mandatory_corrected_invariants", {})
    expected_corrected_keys = {
        "independent_deadline_watchdog",
        "actual_elapsed_beyond_reservation_is_never_silently_clamped",
        "same_boot_dead_reconciliation_includes_current_monotonic_elapsed",
        "matching_live_process_is_never_reconciled",
        "one_canonical_jsonl_byte_representation",
        "gpu_execution_ledger_path_is_mandatory_lifecycle_identity",
        "authoritative_terminal_can_recover_only_an_exact_torn_execution_end",
        "protected_paths_use_trusted_no_follow_inode_checked_operations",
        "discovery_and_promotion_seals_are_published_under_locked_closed_snapshots",
        "selector_revalidates_exact_ledger_prefix_and_receipt_cover",
    }
    forbidden = correction.get("forbidden_changes", {})
    expected_forbidden_keys = {
        "gpu_budget_heartbeat_grace_margin_or_matrix_change",
        "model_loss_optimizer_schedule_data_split_target_selection_key_or_decoder_change",
        "outer_test_opening",
        "quarantined_attempt_reuse",
        "review_finding_suppression_without_regression",
        "commercial_or_confirmatory_claim",
    }
    reauthorization = correction.get("required_reauthorization", {})
    correction_claim = correction.get("claim_boundary", {})
    if not (
        correction.get("classification")
        == "pre_discovery_adaptive_v3r1_adversarial_gpu_lifecycle_and_selection_snapshot_correction_authorization"
        and parent_binding.get("file_sha256") == sha256_file(parent_path)
        and parent_binding.get("content_sha256") == parent.get("content_sha256")
        and review_binding.get("file_sha256") == sha256_file(review_path)
        and review_binding.get("content_sha256") == review.get("content_sha256")
        and review_binding.get("high_findings") == 4
        and review_binding.get("medium_findings") == 4
        and review_binding.get("reviewed_bytes_may_authorize_gpu_training") is False
        and authorized_paths == list(expected_before_hashes)
        and all(
            authorized_by_path.get(path, {}).get("before_sha256") == expected
            for path, expected in expected_before_hashes.items()
        )
        and set(corrected) == expected_corrected_keys
        and all(value is True for value in corrected.values())
        and set(forbidden) == expected_forbidden_keys
        and all(value is True for value in forbidden.values())
        and reauthorization.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V7R1.json"
        and reauthorization.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V7R1.json"
        and reauthorization.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V7R1.json"
        and reauthorization.get("all_base_and_additive_files_frozen_0444") is True
        and reauthorization.get("all_adversarial_regressions_pass") is True
        and reauthorization.get("no_gpu_training_before_v7r1_pretrain_validation")
        is True
        and correction_claim.get("adaptive_retrospective_only") is True
        and correction_claim.get("outer_test_opened") is False
        and correction_claim.get("accuracy_or_commercial_target_result_used") is False
        and correction_claim.get("confirmatory") is False
        and correction_claim.get("commercial_claim_authorized") is False
    ):
        _fail("V7R1 adversarial correction authorization drifted")

    for path in (review_path, correction_path):
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            _fail(f"V7R1 adversarial governance evidence became writable: {path}")


def _validate_v7r2_reliability_authorization(root: Path) -> None:
    review_path = (
        root
        / CAMPAIGN_DIR
        / "diagnostics/v3r1_v7r1_lifecycle_reliability_review_pretrain_v7r2.json"
    )
    correction_path = (
        root / CAMPAIGN_DIR / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V7R2.json"
    )
    parent_path = (
        root / CAMPAIGN_DIR / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V7R1.json"
    )
    review = load_json(review_path)
    correction = load_json(correction_path)
    parent = load_json(parent_path)

    expected_reviewed = {
        "gpu_budget_ledger": (
            "src/snn_rr/gpu_budget_ledger.py",
            "c93357d233339f99b0af787498953cf2f85d033231ff4de9eaf5419cf6624e1a",
        ),
        "gpu_wrapper": (
            "scripts/run_gpu_admitted.py",
            "2add808c54407177e89653e03f97f8f0f262f2034d3931123b40237305e66bb5",
        ),
        "resource_safety_tests": (
            "tests/test_run_gpu_admitted.py",
            "94859f60a0c73773e7899a15b9eaea753e9021f2c99804ef33edd8c447592ac4",
        ),
        "validator": (
            "scripts/validate_hfr_v3r1_authorization.py",
            "38f3790e33aea75d844b29ba4198a08878515e498979cb0ff2ce3e878a483756",
        ),
    }
    reviewed = review.get("reviewed_implementation", {})
    reviewed_matches = (
        set(reviewed) == set(expected_reviewed)
        and all(
            reviewed.get(name, {}).get("path") == expected[0]
            and reviewed.get(name, {}).get("file_sha256") == expected[1]
            for name, expected in expected_reviewed.items()
        )
    )
    findings = review.get("findings", [])
    finding_severities = {
        item.get("id"): item.get("severity")
        for item in findings
        if isinstance(item, Mapping)
    }
    expected_regressions = [
        "dead watchdog plus TERM-ignoring same-session descendant is fully reaped before publication",
        "setpgid descendant inside the workload session cannot survive the deadline",
        "protected root rename and recreate cannot reset an exhausted budget",
        "usage admission and execution lock replacement cannot admit a lost successful append",
        "active execution ledger bytes are canonical and atomically written",
        "execution end terminal-derived drift blocks result recovery",
        "concurrent different result publications cannot both succeed",
        "termination escalation is never reusable success",
        "hardlinked result receipt is rejected at read time",
    ]
    boundary = review.get("review_boundary", {})
    decision = review.get("decision", {})
    claim = review.get("claim_boundary", {})
    if not (
        review.get("classification")
        == "pretrain_v7r1_adversarial_lifecycle_reliability_review"
        and reviewed_matches
        and finding_severities
        == {
            "V7R1-R1": "high",
            "V7R1-R2": "high",
            "V7R1-R3": "high",
            "V7R1-R4": "high",
            "V7R1-R5": "medium",
            "V7R1-R6": "medium",
            "V7R1-R7": "medium",
            "V7R1-R8": "medium",
            "V7R1-R9": "medium",
        }
        and review.get("mandatory_regressions") == expected_regressions
        and boundary.get("outer_test_features_or_targets_opened") is False
        and boundary.get("accuracy_or_commercial_target_result_used") is False
        and boundary.get("new_gpu_training_started") is False
        and boundary.get("usage_ledger_mutated") is False
        and boundary.get("targeted_tests_passed_before_fault_injection") == 34
        and decision.get(
            "v7r1_pretrain_authorization_may_be_issued_from_reviewed_bytes"
        )
        is False
        and decision.get("gpu_training_may_resume") is False
        and decision.get("goal_should_be_terminated") is False
        and decision.get("v7r2_correction_required") is True
        and decision.get("commercial_claim_authorized") is False
        and claim.get("adaptive_retrospective_only") is True
        and claim.get("outer_test_features_or_targets_opened") is False
        and claim.get("accuracy_or_commercial_target_result_used") is False
        and claim.get("confirmatory") is False
        and claim.get("commercial_claim_authorized") is False
    ):
        _fail("V7R2 lifecycle reliability review drifted")

    parent_binding = correction.get("parent_correction", {})
    review_binding = correction.get("reliability_review", {})
    authorized = correction.get("authorized_modifications", [])
    authorized_paths = [
        item.get("path") for item in authorized if isinstance(item, Mapping)
    ]
    expected_before = {
        "src/snn_rr/gpu_budget_ledger.py": (
            "c93357d233339f99b0af787498953cf2f85d033231ff4de9eaf5419cf6624e1a"
        ),
        "scripts/run_gpu_admitted.py": (
            "2add808c54407177e89653e03f97f8f0f262f2034d3931123b40237305e66bb5"
        ),
        "tests/test_run_gpu_admitted.py": (
            "94859f60a0c73773e7899a15b9eaea753e9021f2c99804ef33edd8c447592ac4"
        ),
        "scripts/validate_hfr_v3r1_authorization.py": (
            "38f3790e33aea75d844b29ba4198a08878515e498979cb0ff2ce3e878a483756"
        ),
    }
    authorized_by_path = {
        item.get("path"): item for item in authorized if isinstance(item, Mapping)
    }
    corrected = correction.get("mandatory_corrected_invariants", {})
    expected_corrected = {
        "watchdog_failure_cannot_publish_before_full_job_cleanup",
        "complete_trusted_workload_session_is_deadline_bounded",
        "protected_directory_identity_is_pinned_for_the_transaction",
        "replacement_lock_generation_cannot_admit_a_lost_successful_append",
        "active_execution_stream_is_atomic_and_canonical",
        "execution_recovery_requires_exact_terminal_derived_fields",
        "result_receipt_is_create_once_under_concurrent_publishers",
        "termination_escalation_is_nonzero_and_never_reusable",
        "result_read_requires_pinned_no_follow_single_link_inode",
    }
    explicit = correction.get("explicit_boundary", {})
    expected_explicit = {
        "authorized_workload_is_the_frozen_trusted_trainer_command": True,
        "authorized_trainer_intentionally_daemonizes_or_creates_a_new_session": False,
        "arbitrary_noncooperative_same_uid_filesystem_mutation_is_a_supported_workload": False,
        "directory_or_lock_generation_drift_detected_at_any_critical_boundary_fails_closed": True,
        "arbitrary_pre_atomic_torn_execution_bytes_are_repaired": False,
        "arbitrary_torn_execution_bytes_fail_closed": True,
    }
    forbidden = correction.get("forbidden_changes", {})
    expected_forbidden = {
        "gpu_budget_heartbeat_grace_margin_or_campaign_matrix_change",
        "model_loss_optimizer_schedule_data_split_target_selection_key_or_decoder_change",
        "outer_test_opening",
        "quarantined_attempt_reuse",
        "reliability_assertion_relaxation",
        "commercial_or_confirmatory_claim",
    }
    reauthorization = correction.get("required_reauthorization", {})
    correction_claim = correction.get("claim_boundary", {})
    if not (
        correction.get("classification")
        == "pretrain_adaptive_v3r1_v7r2_lifecycle_reliability_correction_authorization"
        and parent_binding.get("file_sha256") == sha256_file(parent_path)
        and parent_binding.get("content_sha256") == parent.get("content_sha256")
        and review_binding.get("file_sha256") == sha256_file(review_path)
        and review_binding.get("content_sha256") == review.get("content_sha256")
        and review_binding.get("high_findings") == 4
        and review_binding.get("medium_findings") == 5
        and review_binding.get("reviewed_bytes_may_authorize_gpu_training") is False
        and authorized_paths == list(expected_before)
        and all(
            authorized_by_path.get(path, {}).get("before_sha256") == expected
            for path, expected in expected_before.items()
        )
        and set(corrected) == expected_corrected
        and all(value is True for value in corrected.values())
        and explicit == expected_explicit
        and set(forbidden) == expected_forbidden
        and all(value is True for value in forbidden.values())
        and reauthorization.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V7R2.json"
        and reauthorization.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V7R2.json"
        and reauthorization.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V7R2.json"
        and reauthorization.get("all_base_and_additive_files_frozen_0444") is True
        and reauthorization.get("all_fixed_and_v7r2_regressions_pass") is True
        and reauthorization.get("active_usage_and_execution_ledgers_validate_before_authorization")
        is True
        and reauthorization.get("no_gpu_training_before_v7r2_pretrain_validation")
        is True
        and correction_claim.get("adaptive_retrospective_only") is True
        and correction_claim.get("outer_test_opened") is False
        and correction_claim.get("accuracy_or_commercial_target_result_used") is False
        and correction_claim.get("confirmatory") is False
        and correction_claim.get("commercial_claim_authorized") is False
    ):
        _fail("V7R2 lifecycle reliability correction authorization drifted")

    for path in (review_path, correction_path):
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            _fail(f"V7R2 reliability governance evidence became writable: {path}")


def _validate_v8_efficiency_authorization(root: Path) -> None:
    diagnostic_path = (
        root
        / CAMPAIGN_DIR
        / "diagnostics/v3r1_v8_batched_execution_reuse_and_benchmark_design.json"
    )
    correction_path = root / CAMPAIGN_DIR / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8.json"
    parent_path = root / CAMPAIGN_DIR / "PRETRAIN_AUTHORIZATION_V7R2.json"
    for path, label in (
        (diagnostic_path, "V8 efficiency design"),
        (correction_path, "V8 correction authorization"),
        (parent_path, "V7R2 parent pretrain authorization"),
    ):
        require_frozen_regular_file(path, label)
    diagnostic = load_json(diagnostic_path)
    correction = load_json(correction_path)
    parent = load_json(parent_path)
    verify_content_hash(diagnostic, path=diagnostic_path)
    verify_content_hash(correction, path=correction_path)
    verify_content_hash(parent, path=parent_path)

    findings = diagnostic.get("findings")
    finding_profile = {
        item.get("id"): item.get("severity")
        for item in findings
        if isinstance(item, Mapping)
    } if isinstance(findings, list) else {}
    benchmark = diagnostic.get("efficiency_benchmark_design", {})
    ownership = diagnostic.get("final_gpu_ownership_target", {})
    decision = diagnostic.get("decision", {})
    claim = diagnostic.get("claim_boundary", {})
    if not (
        diagnostic.get("classification")
        == "pretrain_v8_efficiency_reuse_and_launch_integrity_design"
        and finding_profile
        == {
            "V8-E1": "blocking",
            "V8-E2": "high",
            "V8-E3": "high",
            "V8-E4": "high",
            "V8-E5": "blocking",
        }
        and benchmark.get("phase") == "efficiency_benchmark"
        and benchmark.get("epochs") == 2
        and benchmark.get(
            "epoch_2_steady_train_plus_target_free_validation_seconds_max"
        )
        == 23.0
        and benchmark.get("outer_test_opened") is False
        and benchmark.get("accuracy_metrics_emitted_or_used") is False
        and benchmark.get("checkpoint_selection_performed") is False
        and benchmark.get("training_result_reusable") is False
        and benchmark.get("selection_or_promotion_input") is False
        and ownership.get("benchmark_execution_receipts") == 1
        and ownership.get("discovery_training_receipts") == 18
        and ownership.get("new_promotion_training_receipts") == 12
        and ownership.get("promotion_pointer_receipts_without_gpu_records") == 6
        and ownership.get("prediction_execution_receipts") == 18
        and ownership.get("total_gpu_execution_owners") == 49
        and decision.get("v7r2_bytes_may_be_used_for_gpu_discovery_without_v8")
        is False
        and decision.get("v8_correction_required_before_any_new_gpu_work") is True
        and decision.get("goal_should_be_terminated") is False
        and decision.get("outer_test_must_remain_sealed") is True
        and decision.get("commercial_claim_authorized") is False
        and claim.get("adaptive_retrospective_only") is True
        and claim.get("outer_test_features_or_targets_opened") is False
        and claim.get("accuracy_or_commercial_target_result_used") is False
        and claim.get("confirmatory") is False
        and claim.get("commercial_claim_authorized") is False
    ):
        _fail("V8 efficiency/reuse/launch design drifted")

    expected_before = {
        "scripts/train_harmonic_factor_router_snn_v3r1.py": "2ae208e7dab8e7e4776387442479f1277f34d4d9b90d70d9d8cc8a9c58b945a5",
        "scripts/run_hfr_v3r1_discovery_campaign.py": "803094d3c66b08790ee8d477bcc48298d1da8230981faa437d963e214ac2d535",
        "scripts/select_hfr_v3r1_common_variant.py": "bad0b0eeac339a931f3a83a30f36cbf9d7edf4150c22f46c85b320096ae3e882",
        "scripts/run_fixed_hfr_v3r1_oof_campaign.py": "7c4bf8e2fbf76486a8bdc1c44dd5a5d8acdb90cf71b4242f30a801af825193b6",
        "scripts/build_locked_hfr_v3r1_test_inputs.py": "f0aee3edeb0fa109c70ec561451829342c52840f3e788079ff476eedaad73a39",
        "scripts/run_gpu_admitted.py": "c5f69643b1f98544e19964a1aaae288fdfb3ae0c200572b4e7bf0a08fecff666",
        "scripts/validate_hfr_v3r1_authorization.py": "1ba21c36d9442401cf433ec43d0fbd1084a4cc89beb47d16222e020c03491f3f",
        "tests/test_train_harmonic_factor_router_snn_v3r1.py": "9e2774c2cb0ba97d306bbf6ca0f2ba546e64a7c7f7c1a57c9d5aee775bff2b4f",
        "tests/test_run_hfr_v3r1_campaign.py": "54bb942d49ea6bd32a18b072f7b72b05e457432f657ccffe590f604d55db1957",
        "tests/test_locked_hfr_v3r1_oof.py": "179d76ca74c0477ad84e02bca4b345a634c053a188a850891832be9a4d4830c7",
        "tests/test_run_gpu_admitted.py": "8b71344a2243d42d0878f1eb451d865c4cfe128c5688c6e3c66bad710a96ee0d",
        "scripts/benchmark_hfr_v3r1_efficiency.py": None,
        "tests/test_benchmark_hfr_v3r1_efficiency.py": None,
    }
    authorized = correction.get("authorized_modifications")
    authorized_by_path = {
        item.get("path"): item
        for item in authorized
        if isinstance(item, Mapping)
    } if isinstance(authorized, list) else {}
    parent_binding = correction.get("parent_pretrain_authorization", {})
    design_binding = correction.get("design_diagnostic", {})
    mandatory = correction.get("mandatory_invariants", {})
    reauthorization = correction.get("required_reauthorization", {})
    correction_claim = correction.get("claim_boundary", {})
    unchanged = correction.get("unchanged_frozen_scientific_core", {})
    if not (
        correction.get("classification")
        == "pretrain_adaptive_v3r1_v8_efficiency_reuse_and_launch_integrity_correction_authorization"
        and parent_binding.get("file_sha256") == sha256_file(parent_path)
        and parent_binding.get("content_sha256") == parent.get("content_sha256")
        and design_binding.get("file_sha256") == sha256_file(diagnostic_path)
        and design_binding.get("content_sha256") == diagnostic.get("content_sha256")
        and list(authorized_by_path) == list(expected_before)
        and all(
            authorized_by_path.get(path, {}).get("before_sha256") == expected
            for path, expected in expected_before.items()
        )
        and exact_json_equal(
            unchanged,
            {
                "src/snn_rr/harmonic_feature_layout_v3r1.py": "f779b46a5f3818dc051e3f0758388ec0baccda9a3e05d8d14da9f8553456bf07",
                "src/snn_rr/harmonic_factor_router_models_v3r1.py": "097024fb50046fd18a1fa48fb440f495807c49b9c04ac3f31d8b4c5acc9e6bc5",
                "src/snn_rr/gpu_budget_ledger.py": "b71e792b860764ef78a3d9f77806dc2af77cc8f421856a0666d75650634f4c41",
            },
        )
        and mandatory.get("outer_test_features_or_targets_remain_sealed_before_selection")
        is True
        and mandatory.get("model_candidate_loss_optimizer_and_selection_rule_unchanged")
        is True
        and mandatory.get("benchmark_epoch2_train_plus_validation_seconds_max")
        == 23.0
        and mandatory.get("benchmark_accuracy_metrics_forbidden") is True
        and mandatory.get("six_reuse_pointers_own_no_gpu_records") is True
        and mandatory.get("twelve_new_promotion_training_units") is True
        and mandatory.get("final_gpu_execution_owner_count") == 49
        and mandatory.get("maximum_parallel_gpu_jobs") == 1
        and mandatory.get("gpu_hours_hard") == 10.0
        and mandatory.get("unrelated_open_lifecycle_rejected") is True
        and mandatory.get("direct_child_binding_forgery_rejected") is True
        and reauthorization.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8.json"
        and reauthorization.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8.json"
        and reauthorization.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8.json"
        and reauthorization.get("all_authorized_and_unchanged_core_files_frozen_0444")
        is True
        and reauthorization.get("all_fixed_v7r2_and_v8_regressions_pass") is True
        and reauthorization.get("active_usage_and_execution_ledgers_validate_before_authorization")
        is True
        and reauthorization.get("no_gpu_work_before_v8_pretrain_validation") is True
        and correction_claim.get("adaptive_retrospective_only") is True
        and correction_claim.get("outer_test_opened") is False
        and correction_claim.get("accuracy_or_commercial_target_result_used") is False
        and correction_claim.get("confirmatory") is False
        and correction_claim.get("commercial_claim_authorized") is False
    ):
        _fail("V8 efficiency/reuse correction authorization drifted")


def _read_exact_historical_runtime_prefix(
    path: Path, *, size: int, digest: str, label: str
) -> bytes:
    """Read an immutable ledger prefix without requiring its live suffix closed."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        _fail(f"cannot open {label}: {error}")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < size
        ):
            _fail(f"{label} is aliased, non-regular, or truncated")
        chunks: list[bytes] = []
        offset = 0
        while offset < size:
            block = os.pread(descriptor, size - offset, offset)
            if not block:
                _fail(f"{label} was truncated during prefix read")
            chunks.append(block)
            offset += len(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino)
        if (
            (after.st_dev, after.st_ino) != identity
            or (named.st_dev, named.st_ino) != identity
            or after.st_nlink != 1
            or named.st_nlink != 1
            or after.st_size < size
            or not stat.S_ISREG(named.st_mode)
        ):
            _fail(f"{label} identity changed during prefix read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        _fail(f"{label} exact historical prefix drifted")
    return raw


def _decode_exact_jsonl_prefix(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n"):
        _fail(f"{label} is not a complete newline-terminated stream")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(keepends=True)):
        try:
            row = json.loads(
                line[:-1].decode("utf-8", errors="strict"),
                object_pairs_hook=_unique_object,
                parse_constant=lambda token: _fail(
                    f"non-finite {label} value at row {number}: {token}"
                ),
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            _fail(f"invalid {label} row {number}: {error}")
        if not isinstance(row, dict) or canonical_bytes(row) + b"\n" != line:
            _fail(f"non-canonical {label} row {number}")
        rows.append(row)
    return rows


def _validate_v8r1_failed_runtime_prefix(
    root: Path, *, runtime: Mapping[str, Any], incident: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate only the closed V8R1 failure prefix, allowing a live child tail."""

    usage_path = root / V6_USAGE_LEDGER
    execution_path = root / V7_GPU_EXECUTION_LEDGER
    if not (
        runtime.get("usage_ledger_path") == V6_USAGE_LEDGER.as_posix()
        and runtime.get("usage_ledger_file_sha256")
        == "a2209e527d5dab7b09d4006a097ea67da09166db7a696b68c6e7fd83fd861793"
        and runtime.get("execution_ledger_path")
        == V7_GPU_EXECUTION_LEDGER.as_posix()
        and runtime.get("execution_ledger_file_sha256")
        == "681d1275996c0f07d3bf397359cc839dfc2746280ab4c4bbf4e4e4dbc9f585a0"
    ):
        _fail("V8R1 diagnostic runtime-prefix binding drifted")
    usage_raw = _read_exact_historical_runtime_prefix(
        usage_path,
        size=4835,
        digest=str(runtime["usage_ledger_file_sha256"]),
        label="V8R1 usage ledger",
    )
    execution_raw = _read_exact_historical_runtime_prefix(
        execution_path,
        size=5977,
        digest=str(runtime["execution_ledger_file_sha256"]),
        label="V8R1 execution ledger",
    )
    usage_rows = _decode_exact_jsonl_prefix(usage_raw, label="V8R1 usage ledger")
    execution_rows = _decode_exact_jsonl_prefix(
        execution_raw, label="V8R1 execution ledger"
    )
    for number, row in enumerate(usage_rows):
        recorded = row.get("record_sha256")
        payload = {key: value for key, value in row.items() if key != "record_sha256"}
        if not isinstance(recorded, str) or semantic_sha256(payload) != recorded:
            _fail(f"V8R1 usage ledger record hash drifted at row {number}")
        predecessor = None if number == 0 else usage_rows[number - 1]["record_sha256"]
        if row.get("previous_record_sha256") != predecessor:
            _fail(f"V8R1 usage ledger chain drifted at row {number}")
    terminal_hash = str(incident.get("failed_lifecycle_terminal_record_sha256", ""))
    if not (
        len(usage_rows) == 3
        and len(execution_rows) == 2
        and usage_rows[0].get("record_sha256") == V6_USAGE_GENESIS_RECORD_SHA256
        and usage_rows[1].get("event") == "reservation"
        and usage_rows[2].get("event") == "terminal"
        and usage_rows[2].get("record_sha256") == terminal_hash
        and usage_rows[2].get("reservation_record_sha256")
        == usage_rows[1].get("record_sha256")
        and usage_rows[2].get("lifecycle_id") == usage_rows[1].get("lifecycle_id")
        and usage_rows[2].get("return_code") == 1
        and usage_rows[2].get("charged_usage_ns") == 1_023_036_848
        and usage_rows[2].get("reuse_eligible") is False
        and execution_rows[0].get("event") == "start"
        and execution_rows[1].get("event") == "end"
        and execution_rows[0].get("lifecycle_id")
        == execution_rows[1].get("lifecycle_id")
        == usage_rows[2].get("lifecycle_id")
        and execution_rows[1].get("terminal_record_sha256") == terminal_hash
        and execution_rows[1].get("exit_code") == 1
        and execution_rows[0].get("invocation_sha256")
        == execution_rows[1].get("invocation_sha256")
        == usage_rows[2].get("invocation_sha256")
        and execution_rows[0].get("command_sha256")
        == execution_rows[1].get("command_sha256")
        == usage_rows[2].get("command_sha256")
    ):
        _fail("V8R1 failed benchmark closed-prefix lifecycle drifted")
    return usage_rows, execution_rows


def _validate_v8r1_benchmark_loader_authorization(root: Path) -> None:
    """Validate the pre-retry loader repair and its already charged failure."""

    diagnostic_path = (
        root
        / CAMPAIGN_DIR
        / "diagnostics/v3r1_v8_benchmark_import_loader_failure_pre_discovery_v8r1.json"
    )
    correction_path = (
        root / CAMPAIGN_DIR / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R1.json"
    )
    propagation_path = (
        root / CAMPAIGN_DIR / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R1A.json"
    )
    test_addendum_path = (
        root / CAMPAIGN_DIR / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R1B.json"
    )
    parent_receipt_path = root / CAMPAIGN_DIR / "IMPLEMENTATION_TEST_RECEIPT_V8.json"
    parent_snapshot_path = root / CAMPAIGN_DIR / "V3R1_SOURCE_SNAPSHOT_V8.json"
    parent_authorization_path = root / CAMPAIGN_DIR / "PRETRAIN_AUTHORIZATION_V8.json"
    for path, label in (
        (diagnostic_path, "V8R1 loader failure diagnostic"),
        (correction_path, "V8R1 loader correction authorization"),
        (propagation_path, "V8R1 active-authority propagation authorization"),
        (test_addendum_path, "V8R1 test-expectation consistency addendum"),
        (parent_receipt_path, "V8 parent test receipt"),
        (parent_snapshot_path, "V8 parent source snapshot"),
        (parent_authorization_path, "V8 parent pretrain authorization"),
    ):
        require_frozen_regular_file(path, label)
    diagnostic = load_json(diagnostic_path)
    correction = load_json(correction_path)
    propagation = load_json(propagation_path)
    test_addendum = load_json(test_addendum_path)
    parent_receipt = load_json(parent_receipt_path)
    parent_snapshot = load_json(parent_snapshot_path)
    parent_authorization = load_json(parent_authorization_path)
    for path, document in (
        (diagnostic_path, diagnostic),
        (correction_path, correction),
        (propagation_path, propagation),
        (test_addendum_path, test_addendum),
        (parent_receipt_path, parent_receipt),
        (parent_snapshot_path, parent_snapshot),
        (parent_authorization_path, parent_authorization),
    ):
        verify_content_hash(document, path=path)

    incident = diagnostic.get("incident", {})
    required = diagnostic.get("required_correction", {})
    decision = diagnostic.get("decision", {})
    runtime = diagnostic.get("runtime_ledger_after_failure", {})
    parent = diagnostic.get("parent_v8_chain", {})
    if not (
        diagnostic.get("classification")
        == "pretrain_v8r1_benchmark_import_loader_failure_diagnostic"
        and parent.get("implementation_test_receipt", {}).get("file_sha256")
        == sha256_file(parent_receipt_path)
        and parent.get("source_snapshot", {}).get("file_sha256")
        == sha256_file(parent_snapshot_path)
        and parent.get("pretrain_authorization", {}).get("file_sha256")
        == sha256_file(parent_authorization_path)
        and incident.get("stage") == "pre_discovery_efficiency_benchmark"
        and incident.get("outer_test_features_or_targets_opened") is False
        and incident.get("accuracy_metrics_emitted_or_used") is False
        and incident.get("discovery_training_started") is False
        and incident.get("first_cli_failure_before_gpu_reservation") is True
        and incident.get("second_public_api_attempt_reached_wrapper") is True
        and incident.get("failed_lifecycle_terminal_record_sha256")
        == "d7d3e26a5ba19d648343d19ed4593e503c1017b02f70e8c27517b1d1af0d4fbf"
        and incident.get("failed_lifecycle_return_code") == 1
        and incident.get("failed_lifecycle_charged_usage_ns") == 1_023_036_848
        and incident.get("failed_lifecycle_reuse_eligible") is False
        and incident.get("failed_attempt_must_remain_owned_by_eventual_benchmark_receipt")
        is True
        and runtime.get("usage_ledger_file_sha256")
        == "a2209e527d5dab7b09d4006a097ea67da09166db7a696b68c6e7fd83fd861793"
        and runtime.get("execution_ledger_file_sha256")
        == "681d1275996c0f07d3bf397359cc839dfc2746280ab4c4bbf4e4e4dbc9f585a0"
        and required.get("scientific_model_or_training_change_allowed") is False
        and required.get("benchmark_gate_or_profile_change_allowed") is False
        and required.get("ledger_reset_or_failed_attempt_deletion_allowed") is False
        and decision.get("goal_should_be_terminated") is False
        and decision.get("v8_bytes_may_continue_without_v8r1") is False
        and decision.get("v8r1_loader_correction_required_before_benchmark_retry")
        is True
        and decision.get("outer_test_must_remain_sealed") is True
        and decision.get("commercial_claim_authorized") is False
    ):
        _fail("V8R1 benchmark-loader diagnostic drifted")

    authorized = correction.get("authorized_modifications")
    authorized_by_path = {
        item.get("path"): item for item in authorized if isinstance(item, Mapping)
    } if isinstance(authorized, list) else {}
    expected_before = {
        "scripts/benchmark_hfr_v3r1_efficiency.py": "86a86f13398157633a29570a8d27c34f3b0ecf935006e310413980365cee0474",
        "tests/test_benchmark_hfr_v3r1_efficiency.py": "d845aa9c7b68f88075e02f84676e235fe5c793070e65c6477a5c1c0614581f4a",
        "scripts/validate_hfr_v3r1_authorization.py": "3c891911ac3f670ccad4fd17b814c0f36270d724841aa6f10e241ccb309c1106",
    }
    correction_parent = correction.get("parent_pretrain_authorization", {})
    correction_diagnostic = correction.get("failure_diagnostic", {})
    correction_mandatory = correction.get("mandatory_invariants", {})
    correction_required = correction.get("required_reauthorization", {})
    correction_claim = correction.get("claim_boundary", {})
    if not (
        correction.get("classification")
        == "pretrain_adaptive_v3r1_v8r1_benchmark_loader_correction_authorization"
        and correction_parent.get("file_sha256")
        == sha256_file(parent_authorization_path)
        and correction_parent.get("content_sha256")
        == parent_authorization.get("content_sha256")
        and correction_diagnostic.get("file_sha256") == sha256_file(diagnostic_path)
        and correction_diagnostic.get("content_sha256")
        == diagnostic.get("content_sha256")
        and list(authorized_by_path) == list(expected_before)
        and all(
            authorized_by_path.get(path, {}).get("before_sha256") == expected
            for path, expected in expected_before.items()
        )
        and correction_mandatory.get("outer_test_features_or_targets_remain_sealed")
        is True
        and correction_mandatory.get("accuracy_metrics_remain_forbidden_in_benchmark")
        is True
        and correction_mandatory.get("benchmark_epoch2_train_plus_validation_seconds_max")
        == 23.0
        and correction_mandatory.get("failed_attempt_terminal_record_sha256")
        == incident.get("failed_lifecycle_terminal_record_sha256")
        and correction_mandatory.get("failed_attempt_charged_usage_ns")
        == incident.get("failed_lifecycle_charged_usage_ns")
        and correction_mandatory.get("failed_attempt_owned_by_eventual_receipt")
        is True
        and correction_mandatory.get("failed_attempt_reuse_eligible") is False
        and correction_mandatory.get("ledger_reset_allowed") is False
        and correction_required.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8R1.json"
        and correction_required.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8R1.json"
        and correction_required.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8R1.json"
        and correction_required.get("all_fixed_v8_and_v8r1_regressions_pass") is True
        and correction_required.get("active_usage_and_execution_ledgers_closed_and_bound")
        is True
        and correction_required.get("benchmark_retry_only_after_v8r1_validation")
        is True
        and correction_claim.get("adaptive_retrospective_only") is True
        and correction_claim.get("outer_test_opened") is False
        and correction_claim.get("accuracy_or_commercial_target_result_used") is False
        and correction_claim.get("confirmatory") is False
        and correction_claim.get("commercial_claim_authorized") is False
    ):
        _fail("V8R1 benchmark-loader correction authorization drifted")

    propagation_parent = propagation.get("parent_correction_authorization", {})
    propagation_diagnostic = propagation.get("failure_diagnostic", {})
    propagation_authorized = propagation.get("authorized_modifications")
    propagation_by_path = {
        item.get("path"): item
        for item in propagation_authorized
        if isinstance(item, Mapping)
    } if isinstance(propagation_authorized, list) else {}
    propagation_before = {
        "scripts/benchmark_hfr_v3r1_efficiency.py": "6e41101e437020fcc21ced2ad4a2cd1adfc1ce2bd502df14fd97917873a8d24c",
        "scripts/train_harmonic_factor_router_snn_v3r1.py": "7f2a29e1a8320ddcedf6ae778171517e36370c2ea5077b7972bd3a24ee09892d",
        "scripts/select_hfr_v3r1_common_variant.py": "892467d210e07e19f89daf23ea43326cecbd619d24ea387b47406aedefca2129",
        "scripts/validate_hfr_v3r1_authorization.py": "3c891911ac3f670ccad4fd17b814c0f36270d724841aa6f10e241ccb309c1106",
    }
    propagation_mandatory = propagation.get("mandatory_invariants", {})
    propagation_required = propagation.get("required_reauthorization", {})
    if not (
        propagation.get("classification")
        == "pretrain_adaptive_v3r1_v8r1_active_authority_path_propagation_authorization"
        and propagation_parent.get("file_sha256") == sha256_file(correction_path)
        and propagation_parent.get("content_sha256") == correction.get("content_sha256")
        and propagation_diagnostic.get("file_sha256") == sha256_file(diagnostic_path)
        and propagation_diagnostic.get("content_sha256")
        == diagnostic.get("content_sha256")
        and list(propagation_by_path) == list(propagation_before)
        and all(
            propagation_by_path.get(path, {}).get("before_sha256") == expected
            for path, expected in propagation_before.items()
        )
        and propagation_mandatory.get("one_canonical_active_pretrain_authorization_path")
        == (
            CAMPAIGN_DIR / "PRETRAIN_AUTHORIZATION_V8R1.json"
        ).as_posix()
        and propagation_mandatory.get("parent_v8_authorization_remains_immutable_evidence")
        is True
        and propagation_mandatory.get("benchmark_failed_attempt_remains_charged_and_nonreusable")
        is True
        and propagation_mandatory.get("outer_test_features_or_targets_remain_sealed")
        is True
        and propagation_mandatory.get("scientific_model_training_inference_and_selection_logic_unchanged")
        is True
        and propagation_mandatory.get("ledger_reset_allowed") is False
        and propagation_mandatory.get("commercial_claim_authorized") is False
        and propagation_required.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8R1.json"
        and propagation_required.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8R1.json"
        and propagation_required.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8R1.json"
        and propagation_required.get("all_files_frozen_0444") is True
        and propagation_required.get("benchmark_retry_only_after_validation") is True
    ):
        _fail("V8R1 active-authority propagation authorization drifted")

    addendum_parent = test_addendum.get("parent_correction_authorization", {})
    addendum_issue = test_addendum.get("issue", {})
    addendum_authorized = test_addendum.get("authorized_modifications")
    addendum_by_path = {
        item.get("path"): item
        for item in addendum_authorized
        if isinstance(item, Mapping)
    } if isinstance(addendum_authorized, list) else {}
    addendum_before = {
        "tests/test_train_harmonic_factor_router_snn_v3r1.py": (
            "1e6c05a21f503040d91ed46d5269eaa14b9e3a96feffa918e505e4bb2b728c5d"
        ),
        "scripts/validate_hfr_v3r1_authorization.py": (
            "45070386c315b60033f85867a0a8655a6b8794ae6d010034363fd8780b2a1668"
        ),
    }
    addendum_mandatory = test_addendum.get("mandatory_invariants", {})
    addendum_required = test_addendum.get("required_reauthorization", {})
    addendum_claim = test_addendum.get("claim_boundary", {})
    if not (
        test_addendum.get("classification")
        == "pretrain_adaptive_v3r1_v8r1_test_expectation_consistency_addendum"
        and addendum_parent.get("file_sha256") == sha256_file(propagation_path)
        and addendum_parent.get("content_sha256")
        == propagation.get("content_sha256")
        and list(addendum_by_path) == list(addendum_before)
        and all(
            addendum_by_path.get(path, {}).get("before_sha256") == expected
            for path, expected in addendum_before.items()
        )
        and addendum_issue.get("stage") == "pre_reauthorization_fixed_test"
        and addendum_issue.get("production_behavior_is_incorrect") is False
        and addendum_issue.get("test_expectation_is_stale") is True
        and addendum_issue.get("outer_test_features_or_targets_opened") is False
        and addendum_issue.get("accuracy_or_commercial_target_result_used") is False
        and addendum_issue.get("gpu_work_started_after_failure") is False
        and addendum_mandatory.get("trainer_active_authorization_remains_v8r1")
        is True
        and addendum_mandatory.get("missing_authorization_still_raises_runtime_error")
        is True
        and addendum_mandatory.get("exception_assertion_removal_or_weakening_allowed")
        is False
        and addendum_mandatory.get(
            "scientific_model_training_inference_and_selection_logic_unchanged"
        )
        is True
        and addendum_mandatory.get("benchmark_gate_or_budget_change_allowed")
        is False
        and addendum_mandatory.get("ledger_reset_or_failed_attempt_deletion_allowed")
        is False
        and addendum_mandatory.get("outer_test_features_or_targets_remain_sealed")
        is True
        and addendum_mandatory.get("commercial_claim_authorized") is False
        and addendum_required.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8R1.json"
        and addendum_required.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8R1.json"
        and addendum_required.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8R1.json"
        and addendum_required.get("all_files_frozen_0444") is True
        and addendum_required.get("all_fixed_tests_pass") is True
        and addendum_required.get("benchmark_retry_only_after_validation") is True
        and addendum_claim.get("adaptive_retrospective_only") is True
        and addendum_claim.get("outer_test_opened") is False
        and addendum_claim.get("accuracy_or_commercial_target_result_used") is False
        and addendum_claim.get("confirmatory") is False
        and addendum_claim.get("commercial_claim_authorized") is False
    ):
        _fail("V8R1 test-expectation consistency addendum drifted")

    # This closed prefix is immutable historical evidence.  The active suffix
    # may contain exactly one wrapper-owned lifecycle while an admitted child
    # validates its authority, so historical validation must not demand that
    # mutable suffix be globally closed.
    usage_rows, execution_rows = _validate_v8r1_failed_runtime_prefix(
        root, runtime=runtime, incident=incident
    )
    terminal_hash = str(incident["failed_lifecycle_terminal_record_sha256"])
    terminals = [
        row for row in usage_rows if row.get("record_sha256") == terminal_hash
    ]
    if not (
        len(terminals) == 1
        and terminals[0].get("event") == "terminal"
        and terminals[0].get("phase") == "efficiency_benchmark"
        and terminals[0].get("return_code") == 1
        and terminals[0].get("charged_usage_ns") == 1_023_036_848
        and terminals[0].get("reuse_eligible") is False
        and terminals[0].get("context")
        == {
            "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
        }
        and len(
            [
                row
                for row in execution_rows
                if row.get("event") == "end"
                and row.get("terminal_record_sha256") == terminal_hash
                and row.get("exit_code") == 1
            ]
        )
        == 1
    ):
        _fail("V8R1 failed benchmark lifecycle evidence drifted")


def _validate_v8r2_benchmark_resume_authorization(root: Path) -> None:
    resume_diagnostic_path = (
        root
        / CAMPAIGN_DIR
        / "diagnostics/v3r1_v8r1_immutable_invocation_resume_failure_pre_discovery_v8r2.json"
    )
    live_child_diagnostic_path = (
        root
        / CAMPAIGN_DIR
        / "diagnostics/v3r1_v8r1_historical_validator_live_child_deadlock_pre_discovery_v8r2a.json"
    )
    correction_path = root / CAMPAIGN_DIR / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R2.json"
    parent_receipt_path = root / CAMPAIGN_DIR / "IMPLEMENTATION_TEST_RECEIPT_V8R1.json"
    parent_snapshot_path = root / CAMPAIGN_DIR / "V3R1_SOURCE_SNAPSHOT_V8R1.json"
    parent_authorization_path = root / CAMPAIGN_DIR / "PRETRAIN_AUTHORIZATION_V8R1.json"
    legacy_unit_path = (
        root
        / "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "efficiency_benchmark_v8/BENCHMARK_INVOCATION.json"
    )
    legacy_execution_path = (
        root
        / "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "efficiency_benchmark_v8/attempts/attempt_000/invocation.json"
    )
    legacy_result_path = legacy_execution_path.parent / "GPU_TERMINAL_RESULT.json"
    legacy_authorization_path = root / CAMPAIGN_DIR / "PRETRAIN_AUTHORIZATION_V8.json"
    for path, label in (
        (resume_diagnostic_path, "V8R2 immutable-invocation diagnostic"),
        (live_child_diagnostic_path, "V8R2 live-child validator diagnostic"),
        (correction_path, "V8R2 benchmark resume correction"),
        (parent_receipt_path, "V8R1 parent test receipt"),
        (parent_snapshot_path, "V8R1 parent source snapshot"),
        (parent_authorization_path, "V8R1 parent pretrain authorization"),
        (legacy_unit_path, "V8 legacy benchmark invocation"),
        (legacy_execution_path, "V8 legacy execution invocation"),
        (legacy_result_path, "V8 legacy terminal result"),
        (legacy_authorization_path, "V8 legacy pretrain authorization"),
    ):
        require_frozen_regular_file(path, label)
    documents = {
        "resume": load_json(resume_diagnostic_path),
        "live_child": load_json(live_child_diagnostic_path),
        "correction": load_json(correction_path),
        "parent_receipt": load_json(parent_receipt_path),
        "parent_snapshot": load_json(parent_snapshot_path),
        "parent_authorization": load_json(parent_authorization_path),
        "legacy_unit": load_json(legacy_unit_path),
        "legacy_execution": load_json(legacy_execution_path),
        "legacy_result": load_json(legacy_result_path),
        "legacy_authorization": load_json(legacy_authorization_path),
    }
    for path, document in (
        (resume_diagnostic_path, documents["resume"]),
        (live_child_diagnostic_path, documents["live_child"]),
        (correction_path, documents["correction"]),
        (parent_receipt_path, documents["parent_receipt"]),
        (parent_snapshot_path, documents["parent_snapshot"]),
        (parent_authorization_path, documents["parent_authorization"]),
        (legacy_unit_path, documents["legacy_unit"]),
        (legacy_execution_path, documents["legacy_execution"]),
        (legacy_result_path, documents["legacy_result"]),
        (legacy_authorization_path, documents["legacy_authorization"]),
    ):
        verify_content_hash(document, path=path)

    resume = documents["resume"]
    resume_parent = resume.get("parent_v8r1_chain", {})
    resume_incident = resume.get("incident", {})
    resume_lineage = resume.get("immutable_prior_attempt_lineage", {})
    resume_runtime = resume.get("runtime_ledger_after_collision", {})
    resume_root_cause = resume.get("root_cause", {})
    resume_required = resume.get("required_correction", {})
    resume_decision = resume.get("decision", {})
    if not (
        resume.get("classification")
        == "pretrain_v8r2_immutable_benchmark_invocation_resume_failure_diagnostic"
        and resume_parent.get("implementation_test_receipt", {}).get("file_sha256")
        == sha256_file(parent_receipt_path)
        and resume_parent.get("source_snapshot", {}).get("file_sha256")
        == sha256_file(parent_snapshot_path)
        and resume_parent.get("pretrain_authorization", {}).get("file_sha256")
        == sha256_file(parent_authorization_path)
        and resume_incident.get("stage")
        == "pre_discovery_efficiency_benchmark_parent_reconciliation"
        and resume_incident.get("return_code") == 1
        and resume_incident.get("error_type") == "immutable_artifact_collision"
        and resume_incident.get("wrapper_or_worker_launched") is False
        and resume_incident.get("new_gpu_lifecycle_created") is False
        and resume_incident.get("usage_or_execution_ledger_appended") is False
        and resume_incident.get("outer_test_features_or_targets_opened") is False
        and resume_incident.get("accuracy_metrics_emitted_or_used") is False
        and resume_incident.get("discovery_training_started") is False
        and resume_incident.get("goal_should_be_terminated") is False
        and resume_lineage.get("must_remain_owned_by_eventual_single_benchmark_receipt")
        is True
        and resume_lineage.get("deletion_rewrite_or_ledger_reset_allowed") is False
        and resume_root_cause.get("unconditional_create_once_before_attempt_reconciliation")
        is True
        and resume_root_cause.get("skipping_only_the_create_call_would_be_safe")
        is False
        and resume_root_cause.get("single_current_unit_assumption_would_reject_the_historical_attempt_command")
        is True
        and resume_root_cause.get("scientific_model_or_training_algorithm_defect")
        is False
        and resume_required.get("per_authorization_or_per_attempt_immutable_unit_invocations")
        is True
        and resume_required.get("historical_attempt_command_must_be_reconstructed_from_its_own_unit_invocation")
        is True
        and resume_required.get("historical_source_hashes_must_be_verified_through_the_bound_immutable_authorization_snapshot")
        is True
        and resume_required.get("eventual_receipt_must_exactly_cover_all_failed_and_successful_lifecycles_in_order")
        is True
        and resume_required.get("tampered_unknown_or_cross_lineage_unit_invocation_must_fail_closed")
        is True
        and resume_required.get("benchmark_profile_gate_budget_and_scientific_inputs_may_change")
        is False
        and resume_required.get("failed_attempt_or_ledger_history_may_be_deleted_or_reset")
        is False
        and resume_decision.get("v8r1_bytes_may_retry_benchmark_without_v8r2")
        is False
        and resume_decision.get("v8r2_resume_correction_and_full_reauthorization_required")
        is True
        and resume_decision.get("outer_test_must_remain_sealed") is True
        and resume_decision.get("commercial_claim_authorized") is False
    ):
        _fail("V8R2 immutable-invocation resume diagnostic drifted")

    live_child = documents["live_child"]
    finding = live_child.get("finding", {})
    historical_prefix = live_child.get("historical_closed_prefix", {})
    live_required = live_child.get("required_correction", {})
    live_decision = live_child.get("decision", {})
    if not (
        live_child.get("classification")
        == "pretrain_v8r2_admitted_child_historical_validator_static_diagnostic"
        and live_child.get("parent_v8r1_pretrain_authorization", {}).get("file_sha256")
        == sha256_file(parent_authorization_path)
        and finding.get("stage") == "pre_retry_static_admitted_path_audit"
        and finding.get("validator_function")
        == "_validate_v8r1_benchmark_loader_authorization"
        and finding.get("active_ledger_validator_requires_global_closure") is True
        and finding.get("wrapper_admitted_child_necessarily_has_exactly_one_live_owned_lifecycle")
        is True
        and finding.get("historical_v8r1_failure_proof_requires_only_its_exact_closed_byte_prefix")
        is True
        and finding.get("would_reject_a_valid_admitted_child_before_training") is True
        and finding.get("observed_as_a_new_gpu_lifecycle_failure") is False
        and finding.get("outer_test_features_or_targets_opened") is False
        and finding.get("accuracy_metrics_emitted_or_used") is False
        and exact_json_equal(
            historical_prefix,
            {
                "usage_ledger": {
                    "path": V6_USAGE_LEDGER.as_posix(),
                    "size_bytes": 4835,
                    "file_sha256": "a2209e527d5dab7b09d4006a097ea67da09166db7a696b68c6e7fd83fd861793",
                    "record_count": 3,
                    "tail_record_sha256": "d7d3e26a5ba19d648343d19ed4593e503c1017b02f70e8c27517b1d1af0d4fbf",
                    "settled_usage_ns": 378023036848,
                    "open_reservations": 0,
                },
                "execution_ledger": {
                    "path": V7_GPU_EXECUTION_LEDGER.as_posix(),
                    "size_bytes": 5977,
                    "file_sha256": "681d1275996c0f07d3bf397359cc839dfc2746280ab4c4bbf4e4e4dbc9f585a0",
                    "record_count": 2,
                    "open_lifecycle_count": 0,
                },
            },
        )
        and live_required.get("historical_validator_must_parse_and_verify_the_exact_closed_prefix_independently")
        is True
        and live_required.get("historical_validator_must_not_require_the_mutable_active_suffix_to_be_closed")
        is True
        and live_required.get("parent_test_snapshot_and_authorization_issuance_must_still_require_full_global_closure")
        is True
        and live_required.get("admitted_runtime_path_must_still_revalidate_the_wrapper_capability")
        is True
        and live_required.get("admitted_runtime_path_must_allow_exactly_one_owned_live_lifecycle_only")
        is True
        and live_required.get("unrelated_open_lifecycle_allowed") is False
        and live_decision.get("v8r1_bytes_may_retry_benchmark_without_v8r2") is False
        and live_decision.get("v8r2_prefix_only_historical_validation_required")
        is True
        and live_decision.get("goal_should_be_terminated") is False
        and live_decision.get("commercial_claim_authorized") is False
    ):
        _fail("V8R2 live-child historical-validator diagnostic drifted")

    correction = documents["correction"]
    correction_parent = correction.get("parent_v8r1_chain", {})
    correction_diagnostics = correction.get("diagnostics")
    correction_legacy = correction.get("authorized_legacy_benchmark_lineage", {})
    metadata = correction.get("authorized_metadata_hardening", {})
    authorized = correction.get("authorized_modifications")
    authorized_by_path = {
        item.get("path"): item for item in authorized if isinstance(item, Mapping)
    } if isinstance(authorized, list) else {}
    expected_before = {
        "scripts/benchmark_hfr_v3r1_efficiency.py": "9cf123388c273f44672f757adb240fa0f604e98407402791826e9ee3ad53299e",
        "tests/test_benchmark_hfr_v3r1_efficiency.py": "f83e59e0a4870d799d3263efdd2d10f919a6c95e67da581d9f557819a663bb5e",
        "scripts/train_harmonic_factor_router_snn_v3r1.py": "70018557ce5c8c0c2ffcda2a904c5c4ee7a37dba725d68c9e089e98c2286f5f1",
        "scripts/select_hfr_v3r1_common_variant.py": "6fff56071ccabc29737b4d5512b90ff738ec78fc7dfb888f1c00f324b3085362",
        "tests/test_train_harmonic_factor_router_snn_v3r1.py": "814cf83e6567f6704f3f6e50933ce761749645e5f43669cc73b075ec1b2d28b2",
        "scripts/validate_hfr_v3r1_authorization.py": "091786e7e77dd52679e77a99d5c9268ba23cf1eba01112f3c8a6bd629ff3e0a9",
    }
    mandatory = correction.get("mandatory_invariants", {})
    required = correction.get("required_reauthorization", {})
    claim = correction.get("claim_boundary", {})
    expected_diagnostics = [
        {
            "path": resume_diagnostic_path.relative_to(root).as_posix(),
            "file_sha256": sha256_file(resume_diagnostic_path),
            "content_sha256": resume.get("content_sha256"),
        },
        {
            "path": live_child_diagnostic_path.relative_to(root).as_posix(),
            "file_sha256": sha256_file(live_child_diagnostic_path),
            "content_sha256": live_child.get("content_sha256"),
        },
    ]
    if not (
        correction.get("classification")
        == "pretrain_adaptive_v3r1_v8r2_mixed_authority_benchmark_resume_correction_authorization"
        and correction_parent.get("implementation_test_receipt", {}).get("file_sha256")
        == sha256_file(parent_receipt_path)
        and correction_parent.get("source_snapshot", {}).get("file_sha256")
        == sha256_file(parent_snapshot_path)
        and correction_parent.get("pretrain_authorization", {}).get("file_sha256")
        == sha256_file(parent_authorization_path)
        and exact_json_equal(correction_diagnostics, expected_diagnostics)
        and list(authorized_by_path) == list(expected_before)
        and all(
            authorized_by_path.get(path, {}).get("before_sha256") == expected
            for path, expected in expected_before.items()
        )
        and metadata.get("file_sha256_before_and_after")
        == sha256_file(legacy_result_path)
        and metadata.get("mode_before") == 420
        and metadata.get("mode_after") == 292
        and metadata.get("content_change_allowed") is False
        and mandatory.get("benchmark_id_unchanged")
        == "v8_hfr_2epoch_no_accuracy_metric_efficiency"
        and mandatory.get("benchmark_profile_sha256_unchanged")
        == "70cecae47dc978638cfa79a13c5b2d1cd6c15854b4bad70f405d642510e4c521"
        and mandatory.get("epoch_2_train_plus_target_free_validation_ns_max")
        == 23_000_000_000
        and mandatory.get("accuracy_metrics_authorized") is False
        and mandatory.get("checkpoint_selection_authorized") is False
        and mandatory.get("training_result_reusable") is False
        and mandatory.get("outer_test_features_or_targets_authorized") is False
        and mandatory.get("new_active_unit_invocation_name")
        == "BENCHMARK_INVOCATION_V8R2.json"
        and mandatory.get("legacy_attempt_exact_whitelist_only") is True
        and mandatory.get("legacy_attempt_relaunch_or_reexecution_allowed") is False
        and mandatory.get("legacy_terminal_result_frozen_0444_before_retry") is True
        and mandatory.get("all_failed_and_successful_terminals_owned_exactly_once")
        is True
        and mandatory.get("final_receipt_binds_current_v8r2_authority") is True
        and mandatory.get("final_gpu_execution_owner_count") == 49
        and mandatory.get("gpu_hours_hard") == 10.0
        and mandatory.get("maximum_parallel_gpu_jobs") == 1
        and mandatory.get("historical_v8r1_prefix_verified_without_requiring_mutable_suffix_closure")
        is True
        and mandatory.get("parent_reauthorization_issuance_requires_full_closed_ledgers")
        is True
        and mandatory.get("admitted_child_requires_exactly_one_wrapper_owned_live_lifecycle")
        is True
        and mandatory.get("unrelated_open_lifecycle_allowed") is False
        and mandatory.get("ledger_reset_truncation_or_failed_attempt_deletion_allowed")
        is False
        and mandatory.get("scientific_model_loss_optimizer_and_selection_rule_unchanged")
        is True
        and mandatory.get("commercial_claim_authorized") is False
        and required.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8R2.json"
        and required.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8R2.json"
        and required.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8R2.json"
        and required.get("all_files_frozen_0444") is True
        and required.get("all_fixed_and_v8r2_regressions_pass") is True
        and required.get("active_usage_and_execution_ledgers_closed_and_bound_for_issuance")
        is True
        and required.get("benchmark_retry_only_after_v8r2_validation") is True
        and claim.get("adaptive_retrospective_only") is True
        and claim.get("outer_test_opened") is False
        and claim.get("accuracy_or_commercial_target_result_used") is False
        and claim.get("confirmatory") is False
        and claim.get("commercial_claim_authorized") is False
    ):
        _fail("V8R2 benchmark resume correction authorization drifted")

    expected_legacy = {
        "benchmark_invocation": (legacy_unit_path, "202958d02a8280c89bb1561e3f003c7cbf8cf05539820f11f94cf864e0ba63a3", "52f3c1cd0a7004f7537e65c102a3b7649abc33d1101cb02a26681966b1a24152", 5344),
        "execution_invocation": (legacy_execution_path, "cd9727c64de434a0946adb99e6a3dff3005282ee811350eb5102b91b5a5bac7b", "a9e5f7969778853e1064c7394844edbe9fd1dbf848ef58517c4254f17c9c0049", 2557),
        "terminal_result": (legacy_result_path, "7aea7714e9f5f248254a0052442e632d946930ff8b45ee1f28a39d653fdf41be", "b4a5809cbfbbd0f153867034f4ea36b4cb110b3110d6722c08177cbb36acb6b4", 1699),
    }
    for key, (path, file_hash, content_hash, size) in expected_legacy.items():
        binding = correction_legacy.get(key, {})
        if not (
            binding.get("path") == path.relative_to(root).as_posix()
            and binding.get("file_sha256") == file_hash == sha256_file(path)
            and binding.get("content_sha256") == content_hash
            and binding.get("bytes") == size == path.stat().st_size
        ):
            _fail(f"V8R2 authorized legacy {key} binding drifted")
    if not (
        correction_legacy.get("benchmark_invocation", {}).get("mode") == 292
        and correction_legacy.get("execution_invocation", {}).get("mode") == 292
        and correction_legacy.get("only_authorized_legacy_attempt_index") == 0
        and correction_legacy.get("legacy_child_relaunch_allowed") is False
        and correction_legacy.get("must_be_owned_by_eventual_single_completion_receipt")
        is True
        and correction_legacy.get("pretrain_authorization", {}).get("file_sha256")
        == sha256_file(legacy_authorization_path)
        and documents["legacy_execution"].get("unit_invocation", {}).get("sha256")
        == sha256_file(legacy_unit_path)
        and documents["legacy_result"].get("invocation_sha256")
        == sha256_file(legacy_execution_path)
        and documents["legacy_result"].get("terminal_record_sha256")
        == "d7d3e26a5ba19d648343d19ed4593e503c1017b02f70e8c27517b1d1af0d4fbf"
        and documents["legacy_result"].get("return_code") == 1
        and documents["legacy_result"].get("charged_usage_ns") == 1_023_036_848
        and documents["legacy_result"].get("reusable_success") is False
    ):
        _fail("V8R2 exact legacy benchmark lineage drifted")
    _validate_v8r1_failed_runtime_prefix(
        root, runtime=resume_runtime, incident={
            "failed_lifecycle_terminal_record_sha256": (
                "d7d3e26a5ba19d648343d19ed4593e503c1017b02f70e8c27517b1d1af0d4fbf"
            )
        }
    )
    # V8R2 is historical after V8R3.  Its exact receipt/snapshot/authorization
    # bytes are bound in ENTRY_BINDINGS; active source literals are validated
    # only by the V8R3 authorization below.


def _validate_v8r3_failed_runtime_prefix(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the exact closed through-attempt_001 prefix, allowing a live tail."""

    usage_raw = _read_exact_historical_runtime_prefix(
        root / V6_USAGE_LEDGER,
        size=11_594,
        digest="2dee641cb19ef651abd73e905a1643b10772854c922dd7e45f373f9a2d68dd67",
        label="V8R3 pre-correction usage ledger",
    )
    execution_raw = _read_exact_historical_runtime_prefix(
        root / V7_GPU_EXECUTION_LEDGER,
        size=11_960,
        digest="0634c317480c23aca983744982f809b172525e360ae8262715bff2faf9a24200",
        label="V8R3 pre-correction execution ledger",
    )
    usage_rows = _decode_exact_jsonl_prefix(
        usage_raw, label="V8R3 pre-correction usage ledger"
    )
    execution_rows = _decode_exact_jsonl_prefix(
        execution_raw, label="V8R3 pre-correction execution ledger"
    )
    for number, row in enumerate(usage_rows):
        recorded = row.get("record_sha256")
        payload = {key: value for key, value in row.items() if key != "record_sha256"}
        predecessor = None if number == 0 else usage_rows[number - 1]["record_sha256"]
        if not (
            isinstance(recorded, str)
            and semantic_sha256(payload) == recorded
            and row.get("previous_record_sha256") == predecessor
        ):
            _fail(f"V8R3 usage ledger chain drifted at row {number}")
    context = {
        "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    second_reservation = usage_rows[3] if len(usage_rows) > 3 else {}
    second_terminal = usage_rows[6] if len(usage_rows) > 6 else {}
    if not (
        len(usage_rows) == 7
        and len(execution_rows) == 4
        and usage_rows[0].get("record_sha256") == V6_USAGE_GENESIS_RECORD_SHA256
        and usage_rows[1].get("event") == "reservation"
        and usage_rows[2].get("event") == "terminal"
        and usage_rows[2].get("record_sha256")
        == "d7d3e26a5ba19d648343d19ed4593e503c1017b02f70e8c27517b1d1af0d4fbf"
        and second_reservation.get("event") == "reservation"
        and second_reservation.get("record_sha256")
        == "6ac9c82dcf2eee87ad493846a7ff49221d47e690d76694afaaebc585fd7c24b6"
        and usage_rows[4].get("event") == "heartbeat"
        and usage_rows[5].get("event") == "heartbeat"
        and second_terminal.get("event") == "terminal"
        and second_terminal.get("record_sha256")
        == "ca89b52ef526fb68854879207c0e56b8dc79a1ee0e31f2b0f143a7a2f43741e2"
        and second_terminal.get("reservation_record_sha256")
        == second_reservation.get("record_sha256")
        and second_terminal.get("lifecycle_id")
        == second_reservation.get("lifecycle_id")
        and second_terminal.get("invocation_sha256")
        == "fdca8a8488f0e302aaa65054870a82f0557029bcd576130439596a3c09761cb1"
        and second_terminal.get("command_sha256")
        == "283d5fc26b270aa375ee4c693ddd542381ecaadf260c05988dc151709e151237"
        and second_terminal.get("context") == context
        and second_terminal.get("return_code") == 87
        and second_terminal.get("charged_usage_ns") == 42_457_966_370
        and second_terminal.get("reuse_eligible") is False
        and sum(
            int(row.get("charged_usage_ns", 0))
            for row in usage_rows
            if row.get("event") in {"terminal", "reconciled_terminal"}
        )
        + int(usage_rows[0].get("elapsed_seconds", 0) * 1_000_000_000)
        == 420_481_003_218
        and execution_rows[2].get("event") == "start"
        and execution_rows[3].get("event") == "end"
        and execution_rows[2].get("lifecycle_id")
        == execution_rows[3].get("lifecycle_id")
        == second_terminal.get("lifecycle_id")
        and execution_rows[2].get("invocation_sha256")
        == execution_rows[3].get("invocation_sha256")
        == second_terminal.get("invocation_sha256")
        and execution_rows[2].get("command_sha256")
        == execution_rows[3].get("command_sha256")
        == second_terminal.get("command_sha256")
        and execution_rows[3].get("terminal_record_sha256")
        == second_terminal.get("record_sha256")
        and execution_rows[3].get("exit_code") == 87
    ):
        _fail("V8R3 failed benchmark closed-prefix lifecycle drifted")
    return usage_rows, execution_rows


def _validate_v8r3_benchmark_cover_authorization(root: Path) -> None:
    """Validate the processed-window correction and exact V8R2 failed lineage."""

    diagnostic_path = (
        root
        / CAMPAIGN_DIR
        / "diagnostics/v3r1_v8r2_benchmark_cover_unit_mismatch_pre_discovery_v8r3.json"
    )
    correction_path = (
        root / CAMPAIGN_DIR / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R3.json"
    )
    parent_paths = {
        "implementation_test_receipt": root
        / CAMPAIGN_DIR
        / "IMPLEMENTATION_TEST_RECEIPT_V8R2.json",
        "source_snapshot": root / CAMPAIGN_DIR / "V3R1_SOURCE_SNAPSHOT_V8R2.json",
        "pretrain_authorization": root
        / CAMPAIGN_DIR
        / "PRETRAIN_AUTHORIZATION_V8R2.json",
    }
    unit_path = (
        root
        / "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "efficiency_benchmark_v8/BENCHMARK_INVOCATION_V8R2.json"
    )
    invocation_path = (
        root
        / "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "efficiency_benchmark_v8/attempts/attempt_001/invocation.json"
    )
    result_path = invocation_path.parent / "GPU_TERMINAL_RESULT.json"
    metadata_path = (
        root
        / "artifacts/cache/harmonic_set_v2_fixed_i3_pretest_v2/"
        "outer_3_seed_20260828/metadata.csv"
    )
    for path, label in (
        (diagnostic_path, "V8R3 processed-cover diagnostic"),
        (correction_path, "V8R3 processed-cover correction"),
        *((path, f"V8R2 parent {name}") for name, path in parent_paths.items()),
        (unit_path, "V8R2 benchmark unit invocation"),
        (invocation_path, "V8R2 benchmark execution invocation"),
        (result_path, "V8R2 benchmark terminal result"),
    ):
        require_frozen_regular_file(path, label)
    documents = {
        "diagnostic": load_json(diagnostic_path),
        "correction": load_json(correction_path),
        "unit": load_json(unit_path),
        "invocation": load_json(invocation_path),
        "result": load_json(result_path),
        **{name: load_json(path) for name, path in parent_paths.items()},
    }
    for path, document in (
        (diagnostic_path, documents["diagnostic"]),
        (correction_path, documents["correction"]),
        (unit_path, documents["unit"]),
        (invocation_path, documents["invocation"]),
        (result_path, documents["result"]),
        *((path, documents[name]) for name, path in parent_paths.items()),
    ):
        verify_content_hash(document, path=path)

    diagnostic = documents["diagnostic"]
    parent = diagnostic.get("immutable_parent_v8r2_chain", {})
    failure = diagnostic.get("failed_attempt", {})
    failure_authorization = failure.get("pretrain_authorization", {})
    failure_terminal = failure.get("terminal_result", {})
    root_cause = diagnostic.get("root_cause", {})
    derivation = diagnostic.get("structural_cover_derivation", {})
    runtime = diagnostic.get("runtime_ledgers_after_failed_attempt", {})
    decision = diagnostic.get("decision", {})
    claim = diagnostic.get("claim_boundary", {})
    if not (
        _has_exact_keys(
            diagnostic,
            (
                "schema_version",
                "classification",
                "campaign_id",
                "created_utc",
                "immutable_parent_v8r2_chain",
                "failed_attempt",
                "root_cause",
                "structural_cover_derivation",
                "runtime_ledgers_after_failed_attempt",
                "decision",
                "claim_boundary",
                "content_sha256",
            ),
        )
        and diagnostic.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        and diagnostic.get("created_utc") == "2026-08-29T05:21:22Z"
        and _has_exact_keys(
            parent,
            (
                "implementation_test_receipt",
                "source_snapshot",
                "pretrain_authorization",
            ),
        )
        and all(
            _has_exact_keys(binding, ("path", "file_sha256", "content_sha256"))
            for binding in parent.values()
        )
        and _has_exact_keys(
            failure,
            (
                "attempt_index",
                "benchmark_invocation",
                "execution_invocation",
                "observed_failure",
                "pretrain_authorization",
                "terminal_result",
            ),
        )
        and _has_exact_keys(
            failure.get("benchmark_invocation"),
            ("path", "file_sha256", "content_sha256", "bytes", "mode"),
        )
        and _has_exact_keys(
            failure.get("execution_invocation"),
            ("path", "file_sha256", "content_sha256", "bytes", "mode"),
        )
        and _has_exact_keys(
            failure_authorization, ("path", "file_sha256", "bytes")
        )
        and _has_exact_keys(
            failure_terminal,
            (
                "path",
                "file_sha256",
                "content_sha256",
                "bytes",
                "mode_at_diagnosis",
                "terminal_record_sha256",
                "return_code",
                "charged_usage_ns",
                "reuse_eligible",
            ),
        )
        and _has_exact_keys(
            failure.get("observed_failure"),
            (
                "error",
                "failure_occurs_before_quarantined_telemetry_publication",
                "quarantined_timing_telemetry_exists",
                "reusable_trainer_output_exists",
            ),
        )
        and _has_exact_keys(
            root_cause,
            (
                "benchmark_before_sha256",
                "trainer_before_sha256",
                "failure_mechanism",
                "incorrect_interpretation",
                "correct_interpretation",
                "scientific_model_loss_optimizer_or_sampler_defect",
            ),
        )
        and exact_json_equal(
            root_cause,
            {
                "benchmark_before_sha256": "ce928f029ad4c566b73a7cd972e662d9ee754a993af27665659d8f097df37306",
                "trainer_before_sha256": "5333d93eba650b9dc3584c13c67ad7980e0c4ea36b2600283d638c56de4ced62",
                "failure_mechanism": "The benchmark validator compared trainer telemetry containing processed metadata rows against constants copied from pre-batching logical temporal-chunk counts.",
                "incorrect_interpretation": {
                    "training_windows_per_epoch": 216,
                    "target_free_validation_windows_per_epoch": 54,
                },
                "correct_interpretation": {
                    "training_processed_windows_per_epoch": 6583,
                    "target_free_validation_processed_windows_per_epoch": 1658,
                    "two_epoch_training_processed_windows": 13166,
                    "two_epoch_target_free_validation_processed_windows": 3316,
                },
                "scientific_model_loss_optimizer_or_sampler_defect": False,
            },
        )
        and _has_exact_keys(
            derivation,
            (
                "allowed_metadata_columns_read",
                "forbidden_metadata_columns_read",
                "metadata",
                "fold_row_counts",
                "outer_fold",
                "validation_fold",
                "outer_test_rows_excluded",
                "chunk_windows",
                "logical_chunk_formula",
                "optimizer_steps_per_epoch",
                "training",
                "target_free_validation",
            ),
        )
        and _has_exact_keys(derivation.get("metadata"), ("path", "sha256", "bytes"))
        and _has_exact_keys(
            derivation.get("training"),
            (
                "folds",
                "identities",
                "physical_sessions",
                "logical_temporal_chunks",
                "processed_windows",
            ),
        )
        and _has_exact_keys(
            derivation.get("target_free_validation"),
            ("fold", "physical_sessions", "logical_temporal_chunks", "processed_windows"),
        )
        and _has_exact_keys(runtime, ("usage", "execution", "ledgers_closed"))
        and _has_exact_keys(
            runtime.get("usage"),
            ("path", "sha256", "bytes", "records", "settled_usage_ns"),
        )
        and _has_exact_keys(
            runtime.get("execution"), ("path", "sha256", "bytes", "records")
        )
        and exact_json_equal(
            derivation,
            {
                "allowed_metadata_columns_read": [
                    "fold",
                    "identity",
                    "session_id",
                    "window_number",
                ],
                "forbidden_metadata_columns_read": ["rr_bpm", "reference_valid"],
                "metadata": {
                    "path": "artifacts/cache/harmonic_set_v2_fixed_i3_pretest_v2/outer_3_seed_20260828/metadata.csv",
                    "sha256": "bec6668447b9f15e3a274ab32f8ea375cb1294e3fa02f061783afc875a0dbbdb",
                    "bytes": 3962582,
                },
                "fold_row_counts": {
                    "0": 2043,
                    "1": 1584,
                    "2": 1616,
                    "3": 1335,
                    "4": 1658,
                    "5": 1340,
                },
                "outer_fold": 3,
                "validation_fold": 4,
                "outer_test_rows_excluded": 1335,
                "chunk_windows": 32,
                "logical_chunk_formula": "sum(ceil(physical_session_rows / 32))",
                "optimizer_steps_per_epoch": 5,
                "training": {
                    "folds": [0, 1, 2, 5],
                    "identities": 12,
                    "physical_sessions": 20,
                    "logical_temporal_chunks": 216,
                    "processed_windows": 6583,
                },
                "target_free_validation": {
                    "fold": 4,
                    "physical_sessions": 5,
                    "logical_temporal_chunks": 54,
                    "processed_windows": 1658,
                },
            },
        )
        and exact_json_equal(
            runtime,
            {
                "usage": {
                    "path": V6_USAGE_LEDGER.as_posix(),
                    "sha256": "2dee641cb19ef651abd73e905a1643b10772854c922dd7e45f373f9a2d68dd67",
                    "bytes": 11594,
                    "records": 7,
                    "settled_usage_ns": 420481003218,
                },
                "execution": {
                    "path": V7_GPU_EXECUTION_LEDGER.as_posix(),
                    "sha256": "0634c317480c23aca983744982f809b172525e360ae8262715bff2faf9a24200",
                    "bytes": 11960,
                    "records": 4,
                },
                "ledgers_closed": True,
            },
        )
        and exact_json_equal(
            decision,
            {
                "benchmark_gate_result_available": False,
                "benchmark_retry_without_correction_allowed": False,
                "discovery_may_start": False,
                "goal_should_be_terminated": False,
                "v8r3_correction_and_full_reauthorization_required": True,
            },
        )
        and exact_json_equal(
            claim,
            {
                "accuracy_metrics_emitted_or_used": False,
                "adaptive_retrospective_only": True,
                "commercial_claim_authorized": False,
                "confirmatory": False,
                "discovery_training_started": False,
                "outer_test_features_or_targets_opened": False,
                "selection_or_promotion_input_created": False,
            },
        )
    ):
        _fail("V8R3 diagnostic schema drifted")
    expected_parent_content = {
        "implementation_test_receipt": "f31b8ac49065811a7a7f94ae5365e74f9f8d1d4625048e88d88c80ee37ef27d3",
        "source_snapshot": "4e15a0e235c0d512235e5ef701c502642e974502068512d27ce1b5a2296810f1",
        "pretrain_authorization": "e6d7913c2e24c1b57285e747cbb068e0151da30b8b9c116153f5c3a17969e432",
    }
    for name, path in parent_paths.items():
        binding = parent.get(name, {})
        if not (
            binding.get("path") == path.relative_to(root).as_posix()
            and binding.get("file_sha256") == sha256_file(path)
            and binding.get("content_sha256") == expected_parent_content[name]
            == documents[name].get("content_sha256")
        ):
            _fail(f"V8R3 parent {name} binding drifted")
    failure_bindings = {
        "benchmark_invocation": (
            unit_path,
            "f5451c96f5985769af56adf63107d157717ed97275cc9f93a158e951a33ac1cc",
            "bcd1831b2978e47b18f311041100ae7686cf64addea2b0161dd6a3d678e894a1",
            5346,
        ),
        "execution_invocation": (
            invocation_path,
            "fdca8a8488f0e302aaa65054870a82f0557029bcd576130439596a3c09761cb1",
            "2a4171ef1fb4b7bac52f1bf1f750fc410069af42b6175ee57ae80c24e4aa0fb2",
            2564,
        ),
        "terminal_result": (
            result_path,
            "fccf0bdea0ab17f8f6da33acc7ab4ebf9e0f28301a4a4ea6b4b9d8fa245e70c2",
            "abfa054fa41a1502addcac41cb76b49281d83b89b97f3af6ce138fa74fc756f8",
            1703,
        ),
    }
    failure_document_keys = {
        "benchmark_invocation": "unit",
        "execution_invocation": "invocation",
        "terminal_result": "result",
    }
    for name, (path, file_hash, content_hash, size) in failure_bindings.items():
        binding = failure.get(name, {})
        if not (
            binding.get("path") == path.relative_to(root).as_posix()
            and binding.get("file_sha256") == file_hash == sha256_file(path)
            and binding.get("content_sha256") == content_hash
            == documents[failure_document_keys[name]].get("content_sha256")
            and binding.get("bytes") == size == path.stat().st_size
        ):
            _fail(f"V8R3 diagnostic failed {name} binding drifted")
    observation = failure.get("observed_failure", {})
    if not (
        type(diagnostic.get("schema_version")) is int
        and diagnostic.get("schema_version") == 1
        and diagnostic.get("classification")
        == "pretrain_v8r3_benchmark_processed_window_cover_unit_mismatch_diagnostic"
        and type(failure.get("attempt_index")) is int
        and failure.get("attempt_index") == 1
        and failure.get("benchmark_invocation", {}).get("mode") == 292
        and failure.get("execution_invocation", {}).get("mode") == 292
        and failure_authorization.get("path")
        == parent_paths["pretrain_authorization"].relative_to(root).as_posix()
        and failure_authorization.get("file_sha256")
        == sha256_file(parent_paths["pretrain_authorization"])
        and failure_authorization.get("bytes")
        == parent_paths["pretrain_authorization"].stat().st_size
        == 4122
        and failure_terminal.get("mode_at_diagnosis") == 420
        and failure_terminal.get("terminal_record_sha256")
        == "ca89b52ef526fb68854879207c0e56b8dc79a1ee0e31f2b0f143a7a2f43741e2"
        and failure_terminal.get("return_code") == 87
        and failure_terminal.get("charged_usage_ns") == 42_457_966_370
        and failure_terminal.get("reuse_eligible") is False
        and observation.get("error") == "benchmark training window cover drifted"
        and observation.get("failure_occurs_before_quarantined_telemetry_publication")
        is True
        and observation.get("quarantined_timing_telemetry_exists") is False
        and observation.get("reusable_trainer_output_exists") is False
        and not (invocation_path.parent / "QUARANTINED_TIMING_TELEMETRY.json").exists()
        and not (invocation_path.parent / "FORBIDDEN_REUSABLE_TRAINER_OUTPUT").exists()
        and root_cause.get("failure_mechanism")
        == "The benchmark validator compared trainer telemetry containing processed metadata rows against constants copied from pre-batching logical temporal-chunk counts."
        and exact_json_equal(
            root_cause.get("incorrect_interpretation"),
            {
                "training_windows_per_epoch": 216,
                "target_free_validation_windows_per_epoch": 54,
            },
        )
        and exact_json_equal(
            root_cause.get("correct_interpretation"),
            {
                "training_processed_windows_per_epoch": 6583,
                "target_free_validation_processed_windows_per_epoch": 1658,
                "two_epoch_training_processed_windows": 13166,
                "two_epoch_target_free_validation_processed_windows": 3316,
            },
        )
        and root_cause.get("scientific_model_loss_optimizer_or_sampler_defect")
        is False
        and derivation.get("allowed_metadata_columns_read")
        == ["fold", "identity", "session_id", "window_number"]
        and derivation.get("forbidden_metadata_columns_read")
        == ["rr_bpm", "reference_valid"]
        and derivation.get("fold_row_counts")
        == {"0": 2043, "1": 1584, "2": 1616, "3": 1335, "4": 1658, "5": 1340}
        and derivation.get("outer_fold") == 3
        and derivation.get("validation_fold") == 4
        and derivation.get("outer_test_rows_excluded") == 1335
        and derivation.get("optimizer_steps_per_epoch") == 5
        and derivation.get("training", {}).get("folds") == [0, 1, 2, 5]
        and derivation.get("training", {}).get("processed_windows") == 6583
        and derivation.get("training", {}).get("logical_temporal_chunks") == 216
        and derivation.get("training", {}).get("physical_sessions") == 20
        and derivation.get("target_free_validation", {}).get("fold") == 4
        and derivation.get("target_free_validation", {}).get("processed_windows")
        == 1658
        and derivation.get("target_free_validation", {}).get(
            "logical_temporal_chunks"
        )
        == 54
        and derivation.get("target_free_validation", {}).get("physical_sessions")
        == 5
        and derivation.get("metadata", {}).get("path")
        == metadata_path.relative_to(root).as_posix()
        and derivation.get("metadata", {}).get("sha256")
        == sha256_file(metadata_path)
        == "bec6668447b9f15e3a274ab32f8ea375cb1294e3fa02f061783afc875a0dbbdb"
        and derivation.get("metadata", {}).get("bytes")
        == metadata_path.stat().st_size
        == 3_962_582
        and runtime.get("ledgers_closed") is True
        and runtime.get("usage", {}).get("bytes") == 11_594
        and runtime.get("usage", {}).get("path") == V6_USAGE_LEDGER.as_posix()
        and runtime.get("usage", {}).get("records") == 7
        and runtime.get("usage", {}).get("sha256")
        == "2dee641cb19ef651abd73e905a1643b10772854c922dd7e45f373f9a2d68dd67"
        and runtime.get("usage", {}).get("settled_usage_ns") == 420_481_003_218
        and runtime.get("execution", {}).get("bytes") == 11_960
        and runtime.get("execution", {}).get("path")
        == V7_GPU_EXECUTION_LEDGER.as_posix()
        and runtime.get("execution", {}).get("records") == 4
        and runtime.get("execution", {}).get("sha256")
        == "0634c317480c23aca983744982f809b172525e360ae8262715bff2faf9a24200"
        and decision.get("benchmark_gate_result_available") is False
        and decision.get("benchmark_retry_without_correction_allowed") is False
        and decision.get("discovery_may_start") is False
        and decision.get("goal_should_be_terminated") is False
        and decision.get("v8r3_correction_and_full_reauthorization_required")
        is True
        and claim.get("adaptive_retrospective_only") is True
        and claim.get("outer_test_features_or_targets_opened") is False
        and claim.get("accuracy_metrics_emitted_or_used") is False
        and claim.get("commercial_claim_authorized") is False
    ):
        _fail("V8R3 processed-window diagnostic drifted")

    correction = documents["correction"]
    correction_parent = correction.get("parent_v8r2_chain", {})
    correction_diagnostic = correction.get("diagnostic", {})
    lineage = correction.get("authorized_failed_v8r2_benchmark_lineage", {})
    hardening = correction.get("authorized_metadata_hardening", {})
    profile = correction.get("profile_correction", {})
    mandatory = correction.get("mandatory_invariants", {})
    required = correction.get("required_reauthorization", {})
    correction_claim = correction.get("claim_boundary", {})
    forbidden = correction.get("forbidden_changes", {})
    transitive = correction.get("transitive_legacy_v8_lineage", {})
    authorized = correction.get("authorized_modifications")
    authorized_by_path = {
        item.get("path"): item for item in authorized if isinstance(item, Mapping)
    } if isinstance(authorized, list) else {}
    expected_before = {
        "scripts/benchmark_hfr_v3r1_efficiency.py": "ce928f029ad4c566b73a7cd972e662d9ee754a993af27665659d8f097df37306",
        "tests/test_benchmark_hfr_v3r1_efficiency.py": "fe652ec4938bacfec31ae443401e7d682197a423abe8929bc9c17e38591cb542",
        "scripts/train_harmonic_factor_router_snn_v3r1.py": "5333d93eba650b9dc3584c13c67ad7980e0c4ea36b2600283d638c56de4ced62",
        "scripts/select_hfr_v3r1_common_variant.py": "d85607e572f60a7b997203f102e2908dd277c088fa8f4ab2d1fd255591856615",
        "tests/test_train_harmonic_factor_router_snn_v3r1.py": "3ffdbfc74dc52e2775ef057822d950b1579eb5a346bb111aeb6c35f7a7b72542",
        "scripts/validate_hfr_v3r1_authorization.py": "7869d29d4e14e6b20ebd37c159f638c3d2c8e5fc06db9327becc5295170a2b9c",
    }
    expected_authorized_paths = list(expected_before)
    if not (
        _has_exact_keys(
            correction,
            (
                "schema_version",
                "classification",
                "campaign_id",
                "created_utc",
                "parent_v8r2_chain",
                "diagnostic",
                "authorized_failed_v8r2_benchmark_lineage",
                "authorized_metadata_hardening",
                "profile_correction",
                "authorized_modifications",
                "mandatory_invariants",
                "required_reauthorization",
                "forbidden_changes",
                "transitive_legacy_v8_lineage",
                "claim_boundary",
                "content_sha256",
            ),
        )
        and correction.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        and correction.get("created_utc") == "2026-08-29T05:21:22Z"
        and _has_exact_keys(
            correction_parent,
            (
                "implementation_test_receipt",
                "source_snapshot",
                "pretrain_authorization",
            ),
        )
        and all(
            _has_exact_keys(binding, ("path", "file_sha256", "content_sha256"))
            for binding in correction_parent.values()
        )
        and _has_exact_keys(
            correction_diagnostic, ("path", "file_sha256", "content_sha256")
        )
        and _has_exact_keys(
            lineage,
            (
                "attempt_index",
                "benchmark_invocation",
                "execution_invocation",
                "historical_child_relaunch_allowed",
                "must_be_owned_by_eventual_single_completion_receipt",
                "pretrain_authorization",
                "terminal_result",
            ),
        )
        and _has_exact_keys(
            lineage.get("benchmark_invocation"),
            ("path", "file_sha256", "content_sha256", "bytes", "mode"),
        )
        and _has_exact_keys(
            lineage.get("execution_invocation"),
            ("path", "file_sha256", "content_sha256", "bytes", "mode"),
        )
        and _has_exact_keys(
            lineage.get("pretrain_authorization"), ("path", "file_sha256", "bytes")
        )
        and _has_exact_keys(
            lineage.get("terminal_result"),
            (
                "path",
                "file_sha256",
                "content_sha256",
                "bytes",
                "terminal_record_sha256",
                "return_code",
                "charged_usage_ns",
                "reuse_eligible",
            ),
        )
        and exact_json_equal(
            hardening,
            {
                "path": result_path.relative_to(root).as_posix(),
                "file_sha256_before_and_after": "fccf0bdea0ab17f8f6da33acc7ab4ebf9e0f28301a4a4ea6b4b9d8fa245e70c2",
                "mode_before": 420,
                "mode_after": 292,
                "content_change_allowed": False,
                "reason": "Freeze the complete ledger-bound failed V8R2 terminal before admitting a corrected authority epoch.",
            },
        )
        and exact_json_equal(
            profile,
            {
                "changed_fields_only": {
                    "training_windows_per_epoch": {"before": 216, "after": 6583},
                    "target_free_validation_windows_per_epoch": {
                        "before": 54,
                        "after": 1658,
                    },
                },
                "old_profile_sha256": "70cecae47dc978638cfa79a13c5b2d1cd6c15854b4bad70f405d642510e4c521",
                "new_profile_sha256": "8d031ce6808e622361944828cb9338afbe28e2f63fc7f4016d47bde2c5e6b9d0",
                "workload_changed": False,
            },
        )
        and exact_json_equal(
            mandatory,
            {
                "accuracy_metrics_authorized": False,
                "admitted_child_requires_exactly_one_wrapper_owned_live_lifecycle": True,
                "all_failed_and_successful_benchmark_terminals_owned_exactly_once": True,
                "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
                "checkpoint_selection_authorized": False,
                "commercial_claim_authorized": False,
                "epoch_2_train_plus_target_free_validation_ns_max": 23000000000,
                "final_gpu_execution_owner_count": 49,
                "five_outer3_optimizer_steps_per_epoch": True,
                "gpu_hours_hard": 10.0,
                "historical_v8_and_v8r2_attempt_relaunch_allowed": False,
                "maximum_parallel_gpu_jobs": 1,
                "new_active_unit_invocation_name": "BENCHMARK_INVOCATION_V8R3.json",
                "outer_test_features_or_targets_authorized": False,
                "parent_reauthorization_issuance_requires_full_closed_ledgers": True,
                "target_free_validation_processed_windows_per_epoch": 1658,
                "training_processed_windows_per_epoch": 6583,
                "training_result_reusable": False,
                "unrelated_open_lifecycle_allowed": False,
            },
        )
        and exact_json_equal(
            required,
            {
                "active_usage_and_execution_ledgers_closed_and_bound_for_issuance": True,
                "all_files_frozen_0444": True,
                "all_fixed_and_v8r3_regressions_pass": True,
                "benchmark_retry_only_after_v8r3_validation": True,
                "new_pretrain_authorization": "PRETRAIN_AUTHORIZATION_V8R3.json",
                "new_source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R3.json",
                "new_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R3.json",
            },
        )
        and exact_json_equal(
            correction_claim,
            {
                "accuracy_or_commercial_target_result_used": False,
                "adaptive_retrospective_only": True,
                "commercial_claim_authorized": False,
                "confirmatory": False,
                "outer_test_features_or_targets_opened": False,
            },
        )
        and exact_json_equal(
            forbidden,
            {
                "accuracy_based_efficiency_tuning": True,
                "benchmark_identity_or_23_second_gate_change": True,
                "benchmark_or_failed_attempt_artifact_reuse_as_trained_model": True,
                "commercial_or_confirmatory_claim": True,
                "discovery_matrix_seed_fold_or_variant_change": True,
                "gpu_budget_reset_truncation_or_new_unaccounted_ledger": True,
                "historical_attempt_deletion_rewrite_or_relaunch": True,
                "outer_test_opening_or_use_in_benchmark_discovery_or_selection": True,
                "processed_cover_check_removal_or_weakening": True,
                "scientific_model_loss_optimizer_sampler_or_selection_rule_change": True,
                "subset_to_216_training_or_54_validation_rows": True,
            },
        )
        and exact_json_equal(
            transitive,
            {
                "attempt_index": 0,
                "correction_authorization_file_sha256": "57736f56cf1aa05df88ae409fc3453a9447ed2e26048bb25b6280c4635123382",
                "correction_authorization_path": "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R2.json",
                "must_remain_exact_whitelisted_and_owned": True,
                "relaunch_allowed": False,
            },
        )
        and isinstance(authorized, list)
        and len(authorized) == len(expected_authorized_paths)
        and all(isinstance(item, Mapping) for item in authorized)
        and [item.get("path") for item in authorized] == expected_authorized_paths
        and len(set(expected_authorized_paths)) == len(expected_authorized_paths)
        and all(
            set(item)
            == {
                "path",
                "before_sha256",
                (
                    "allowed_change"
                    if item.get("path")
                    in {
                        "scripts/train_harmonic_factor_router_snn_v3r1.py",
                        "scripts/select_hfr_v3r1_common_variant.py",
                        "tests/test_train_harmonic_factor_router_snn_v3r1.py",
                    }
                    else "allowed_changes"
                ),
            }
            for item in authorized
        )
    ):
        _fail("V8R3 correction schema drifted")
    for name, path in parent_paths.items():
        binding = correction_parent.get(name, {})
        if not (
            binding.get("path") == path.relative_to(root).as_posix()
            and binding.get("file_sha256") == sha256_file(path)
            and binding.get("content_sha256") == documents[name].get("content_sha256")
        ):
            _fail(f"V8R3 correction parent {name} binding drifted")
    expected_lineage = {
        "benchmark_invocation": failure_bindings["benchmark_invocation"],
        "execution_invocation": failure_bindings["execution_invocation"],
        "terminal_result": failure_bindings["terminal_result"],
    }
    for name, (path, file_hash, content_hash, size) in expected_lineage.items():
        binding = lineage.get(name, {})
        if not (
            binding.get("path") == path.relative_to(root).as_posix()
            and binding.get("file_sha256") == file_hash == sha256_file(path)
            and binding.get("content_sha256") == content_hash
            == documents[failure_document_keys[name]].get("content_sha256")
            and binding.get("bytes") == size
        ):
            _fail(f"V8R3 correction lineage {name} binding drifted")
    if not (
        type(correction.get("schema_version")) is int
        and correction.get("schema_version") == 1
        and correction.get("classification")
        == "pretrain_adaptive_v3r1_v8r3_benchmark_processed_window_cover_correction_authorization"
        and correction_diagnostic.get("path")
        == diagnostic_path.relative_to(root).as_posix()
        and correction_diagnostic.get("file_sha256") == sha256_file(diagnostic_path)
        and correction_diagnostic.get("content_sha256")
        == diagnostic.get("content_sha256")
        and list(authorized_by_path) == list(expected_before)
        and all(
            authorized_by_path.get(path, {}).get("before_sha256") == digest
            for path, digest in expected_before.items()
        )
        and type(lineage.get("attempt_index")) is int
        and lineage.get("attempt_index") == 1
        and lineage.get("historical_child_relaunch_allowed") is False
        and lineage.get("must_be_owned_by_eventual_single_completion_receipt")
        is True
        and lineage.get("pretrain_authorization", {}).get("file_sha256")
        == sha256_file(parent_paths["pretrain_authorization"])
        and lineage.get("terminal_result", {}).get("return_code") == 87
        and lineage.get("terminal_result", {}).get("charged_usage_ns")
        == 42_457_966_370
        and lineage.get("terminal_result", {}).get("reuse_eligible") is False
        and lineage.get("terminal_result", {}).get("terminal_record_sha256")
        == "ca89b52ef526fb68854879207c0e56b8dc79a1ee0e31f2b0f143a7a2f43741e2"
        and hardening.get("path") == result_path.relative_to(root).as_posix()
        and hardening.get("file_sha256_before_and_after") == sha256_file(result_path)
        and hardening.get("mode_before") == 420
        and hardening.get("mode_after") == 292
        and hardening.get("content_change_allowed") is False
        and stat.S_IMODE(result_path.stat().st_mode) == 0o444
        and exact_json_equal(
            profile.get("changed_fields_only"),
            {
                "training_windows_per_epoch": {"before": 216, "after": 6583},
                "target_free_validation_windows_per_epoch": {
                    "before": 54,
                    "after": 1658,
                },
            },
        )
        and profile.get("old_profile_sha256")
        == "70cecae47dc978638cfa79a13c5b2d1cd6c15854b4bad70f405d642510e4c521"
        and profile.get("new_profile_sha256")
        == "8d031ce6808e622361944828cb9338afbe28e2f63fc7f4016d47bde2c5e6b9d0"
        and profile.get("workload_changed") is False
        and mandatory.get("benchmark_id")
        == "v8_hfr_2epoch_no_accuracy_metric_efficiency"
        and mandatory.get("training_processed_windows_per_epoch") == 6583
        and mandatory.get("target_free_validation_processed_windows_per_epoch")
        == 1658
        and mandatory.get("five_outer3_optimizer_steps_per_epoch") is True
        and mandatory.get("epoch_2_train_plus_target_free_validation_ns_max")
        == 23_000_000_000
        and mandatory.get("new_active_unit_invocation_name")
        == "BENCHMARK_INVOCATION_V8R3.json"
        and mandatory.get("historical_v8_and_v8r2_attempt_relaunch_allowed")
        is False
        and mandatory.get("all_failed_and_successful_benchmark_terminals_owned_exactly_once")
        is True
        and mandatory.get("parent_reauthorization_issuance_requires_full_closed_ledgers")
        is True
        and mandatory.get("admitted_child_requires_exactly_one_wrapper_owned_live_lifecycle")
        is True
        and mandatory.get("unrelated_open_lifecycle_allowed") is False
        and mandatory.get("final_gpu_execution_owner_count") == 49
        and mandatory.get("gpu_hours_hard") == 10.0
        and mandatory.get("maximum_parallel_gpu_jobs") == 1
        and mandatory.get("accuracy_metrics_authorized") is False
        and mandatory.get("checkpoint_selection_authorized") is False
        and mandatory.get("training_result_reusable") is False
        and mandatory.get("outer_test_features_or_targets_authorized") is False
        and mandatory.get("commercial_claim_authorized") is False
        and required.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8R3.json"
        and required.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8R3.json"
        and required.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8R3.json"
        and required.get("all_files_frozen_0444") is True
        and required.get("all_fixed_and_v8r3_regressions_pass") is True
        and required.get("active_usage_and_execution_ledgers_closed_and_bound_for_issuance")
        is True
        and required.get("benchmark_retry_only_after_v8r3_validation") is True
        and correction_claim.get("adaptive_retrospective_only") is True
        and correction_claim.get("outer_test_features_or_targets_opened") is False
        and correction_claim.get("accuracy_or_commercial_target_result_used") is False
        and correction_claim.get("confirmatory") is False
        and correction_claim.get("commercial_claim_authorized") is False
        and all(
            forbidden.get(key) is True
            for key in (
                "accuracy_based_efficiency_tuning",
                "benchmark_identity_or_23_second_gate_change",
                "benchmark_or_failed_attempt_artifact_reuse_as_trained_model",
                "commercial_or_confirmatory_claim",
                "discovery_matrix_seed_fold_or_variant_change",
                "gpu_budget_reset_truncation_or_new_unaccounted_ledger",
                "historical_attempt_deletion_rewrite_or_relaunch",
                "outer_test_opening_or_use_in_benchmark_discovery_or_selection",
                "processed_cover_check_removal_or_weakening",
                "scientific_model_loss_optimizer_sampler_or_selection_rule_change",
                "subset_to_216_training_or_54_validation_rows",
            )
        )
        and correction.get("transitive_legacy_v8_lineage", {}).get(
            "correction_authorization_file_sha256"
        )
        == "57736f56cf1aa05df88ae409fc3453a9447ed2e26048bb25b6280c4635123382"
        and correction.get("transitive_legacy_v8_lineage", {}).get(
            "must_remain_exact_whitelisted_and_owned"
        )
        is True
        and correction.get("transitive_legacy_v8_lineage", {}).get(
            "relaunch_allowed"
        )
        is False
    ):
        _fail("V8R3 processed-window correction authorization drifted")
    if not (
        documents["invocation"].get("unit_invocation", {}).get("sha256")
        == sha256_file(unit_path)
        and documents["result"].get("invocation_sha256")
        == sha256_file(invocation_path)
        and documents["result"].get("terminal_record_sha256")
        == "ca89b52ef526fb68854879207c0e56b8dc79a1ee0e31f2b0f143a7a2f43741e2"
        and documents["result"].get("return_code") == 87
        and documents["result"].get("charged_usage_ns") == 42_457_966_370
        and documents["result"].get("reusable_success") is False
    ):
        _fail("V8R3 exact V8R2 failed benchmark lineage drifted")
    _validate_v8r3_failed_runtime_prefix(root)

    def parse_authority_source(relative: str) -> ast.Module:
        path = root / relative
        try:
            return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as error:
            _fail(f"cannot parse V8R3 authority source {relative}: {error}")

    def one_assignment(tree: ast.Module, name: str, *, label: str) -> ast.expr:
        values = [
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]
        if len(values) != 1:
            _fail(f"{label} must have exactly one top-level assignment")
        return values[0]

    benchmark_tree = parse_authority_source(
        "scripts/benchmark_hfr_v3r1_efficiency.py"
    )
    current_unit_node = one_assignment(
        benchmark_tree,
        "CURRENT_UNIT_INVOCATION_NAME",
        label="V8R3 current benchmark invocation",
    )
    historical_unit_node = one_assignment(
        benchmark_tree,
        "V8R2_UNIT_INVOCATION_NAME",
        label="V8R2 historical benchmark invocation",
    )
    benchmark_authorization_node = one_assignment(
        benchmark_tree,
        "PRETRAIN_AUTHORIZATION_RELATIVE",
        label="V8R3 benchmark pretrain authorization",
    )
    training_cover_node = one_assignment(
        benchmark_tree,
        "TRAINING_WINDOWS_PER_EPOCH",
        label="V8R3 benchmark training cover",
    )
    validation_cover_node = one_assignment(
        benchmark_tree,
        "VALIDATION_WINDOWS_PER_EPOCH",
        label="V8R3 benchmark validation cover",
    )
    benchmark_authorization_ok = (
        isinstance(benchmark_authorization_node, ast.Call)
        and isinstance(benchmark_authorization_node.func, ast.Name)
        and benchmark_authorization_node.func.id == "Path"
        and len(benchmark_authorization_node.args) == 1
        and not benchmark_authorization_node.keywords
        and isinstance(benchmark_authorization_node.args[0], ast.Constant)
        and benchmark_authorization_node.args[0].value
        == "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/PRETRAIN_AUTHORIZATION_V8R3.json"
    )

    trainer_tree = parse_authority_source(
        "scripts/train_harmonic_factor_router_snn_v3r1.py"
    )
    trainer_functions = [
        node
        for node in trainer_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_admitted_child_binding_for_cli"
    ]
    if len(trainer_functions) != 1:
        _fail("V8R3 admitted-child loader definition drifted")
    trainer_function = trainer_functions[0]
    authorization_assignments = [
        node.value
        for node in trainer_function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "authorization_path"
    ]
    trainer_returns = [
        node.value for node in trainer_function.body if isinstance(node, ast.Return)
    ]
    trainer_authorization_ok = False
    if len(authorization_assignments) == 1 and len(trainer_returns) == 1:
        assigned = authorization_assignments[0]
        returned = trainer_returns[0]
        trainer_authorization_ok = (
            isinstance(assigned, ast.BinOp)
            and isinstance(assigned.op, ast.Div)
            and isinstance(assigned.left, ast.Name)
            and assigned.left.id == "PROJECT_ROOT"
            and isinstance(assigned.right, ast.Constant)
            and assigned.right.value
            == "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/PRETRAIN_AUTHORIZATION_V8R3.json"
            and isinstance(returned, ast.Call)
            and isinstance(returned.func, ast.Attribute)
            and returned.func.attr == "consume_admitted_child_binding"
            and len(returned.args) >= 5
            and isinstance(returned.args[3], ast.Name)
            and returned.args[3].id == "authorization_path"
            and isinstance(returned.args[4], ast.Call)
            and isinstance(returned.args[4].func, ast.Name)
            and returned.args[4].func.id == "sha256_file"
            and len(returned.args[4].args) == 1
            and isinstance(returned.args[4].args[0], ast.Name)
            and returned.args[4].args[0].id == "authorization_path"
            and not any(
                isinstance(node, ast.Constant)
                and node.value == "PRETRAIN_AUTHORIZATION_V8R2.json"
                for node in ast.walk(trainer_function)
            )
        )

    selector_tree = parse_authority_source(
        "scripts/select_hfr_v3r1_common_variant.py"
    )
    selector_authorization_node = one_assignment(
        selector_tree,
        "PRETRAIN_AUTHORIZATION_RELATIVE",
        label="V8R3 selector pretrain authorization",
    )
    selector_authorization_ok = (
        isinstance(selector_authorization_node, ast.BinOp)
        and isinstance(selector_authorization_node.op, ast.Div)
        and isinstance(selector_authorization_node.left, ast.Attribute)
        and isinstance(selector_authorization_node.left.value, ast.Name)
        and selector_authorization_node.left.value.id == "discovery"
        and selector_authorization_node.left.attr == "CAMPAIGN_RELATIVE"
        and isinstance(selector_authorization_node.right, ast.Constant)
        and selector_authorization_node.right.value
        == "PRETRAIN_AUTHORIZATION_V8R3.json"
    )
    if not (
        isinstance(current_unit_node, ast.Constant)
        and current_unit_node.value == "BENCHMARK_INVOCATION_V8R3.json"
        and isinstance(historical_unit_node, ast.Constant)
        and historical_unit_node.value == "BENCHMARK_INVOCATION_V8R2.json"
        and benchmark_authorization_ok
        and isinstance(training_cover_node, ast.Constant)
        and type(training_cover_node.value) is int
        and training_cover_node.value == 6583
        and isinstance(validation_cover_node, ast.Constant)
        and type(validation_cover_node.value) is int
        and validation_cover_node.value == 1658
        and trainer_authorization_ok
        and selector_authorization_ok
    ):
        _fail("V8R3 executable authority propagation or processed cover drifted")


def validate_immutable_evidence(root: Path) -> None:
    for relative, expected in {**READ_ONLY_ANCESTRY, **ENTRY_BINDINGS}.items():
        _verify_bound_path(root, relative, expected)
    for relative in READ_ONLY_ANCESTRY:
        mode = stat.S_IMODE((root / relative).stat().st_mode)
        if mode & 0o222:
            _fail(f"read-only ancestry is writable: {relative}")
    for relative in ENTRY_BINDINGS:
        document = load_json(root / relative)
        verify_content_hash(document, path=root / relative)
    entry = load_json(
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/"
        "V2_I3_FAILURE_ENTRY_LOCK.json"
    )
    if entry.get("v2_entry_condition", {}).get("satisfied") is not True:
        _fail("v2 locked failure entry condition is not satisfied")
    diagnostic = load_json(
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/diagnostics/"
        "v2_i3_failure_decomposition_for_adaptive_v3r1.json"
    )
    decision = diagnostic.get("adaptive_routing_entry_decision", {})
    if not (
        decision.get("adaptive_v3r1_routing_focused_design_entry_supported") is True
        and decision.get("causal_claim") is False
        and decision.get("confirmatory_claim") is False
        and decision.get("locked_full_oof_outcomes_used_in_criterion") is False
    ):
        _fail("adaptive diagnostic claim boundary or decision drifted")
    numeric_diagnostic_path = (
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
        "v3r1_nonfinite_gradient_root_cause_pre_discovery.json"
    )
    numeric_diagnostic = load_json(numeric_diagnostic_path)
    if not (
        numeric_diagnostic.get("classification")
        == "pre_discovery_implementation_smoke_failure_root_cause_diagnostic"
        and numeric_diagnostic.get("claim_boundary", {}).get(
            "accuracy_or_commercial_target_result_used"
        )
        is False
        and numeric_diagnostic.get("claim_boundary", {}).get(
            "outer_test_features_or_targets_opened"
        )
        is False
        and numeric_diagnostic.get("read_only_reproduction", {}).get(
            "only_nonfinite_component"
        )
        == "factor_candidate_js"
        and numeric_diagnostic.get("failed_smoke_evidence", {}).get(
            "optimizer_steps_completed"
        )
        == 0
        and numeric_diagnostic.get("minimal_correction", {}).get(
            "model_forward_change_required"
        )
        is False
    ):
        _fail("numeric-gradient diagnostic boundary or root cause drifted")
    correction = load_json(
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V3.json"
    )
    correction_evidence = correction.get("diagnostic_evidence", {})
    if not (
        correction.get("classification")
        == "pre_discovery_adaptive_v3r1_numeric_gradient_correction_authorization"
        and correction.get("issue", {}).get(
            "accuracy_or_commercial_target_result_observed"
        )
        is False
        and correction.get("issue", {}).get("optimizer_steps_completed") == 0
        and correction_evidence.get("file_sha256")
        == sha256_file(numeric_diagnostic_path)
        and correction_evidence.get("content_sha256")
        == numeric_diagnostic.get("content_sha256")
        and correction.get("claim_boundary", {}).get("commercial_claim_authorized")
        is False
    ):
        _fail("numeric-gradient correction authorization drifted")
    amp_diagnostic_path = (
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
        "v3r1_amp_gradient_scale_root_cause_pre_discovery.json"
    )
    amp_diagnostic = load_json(amp_diagnostic_path)
    amp_smokes = amp_diagnostic.get("controlled_smokes", {})
    amp_correction_plan = amp_diagnostic.get("minimal_correction", {})
    if not (
        amp_diagnostic.get("classification")
        == "pre_discovery_amp_gradient_scale_smoke_diagnostic"
        and amp_diagnostic.get("claim_boundary", {}).get(
            "accuracy_or_commercial_target_result_used"
        )
        is False
        and amp_diagnostic.get("claim_boundary", {}).get(
            "outer_test_features_or_targets_opened"
        )
        is False
        and amp_smokes.get("default_amp_scale_65536", {}).get("exit_code") == 1
        and amp_smokes.get("no_amp", {}).get("exit_code") == 0
        and amp_smokes.get("amp_initial_scale_8192", {}).get("exit_code") == 0
        and amp_correction_plan.get("fixed_initial_amp_gradient_scale") == 8192.0
        and amp_correction_plan.get("deterministic_same-group_replay_on_amp_overflow")
        is True
        and amp_correction_plan.get("no_amp_nonfinite_remains_fatal") is True
    ):
        _fail("AMP gradient-scale diagnostic boundary or evidence drifted")
    amp_correction = load_json(
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V4.json"
    )
    amp_correction_evidence = amp_correction.get("diagnostic_evidence", {})
    if not (
        amp_correction.get("classification")
        == "pre_discovery_adaptive_v3r1_amp_gradient_scale_correction_authorization"
        and amp_correction.get("issue", {}).get(
            "accuracy_or_commercial_target_result_observed"
        )
        is False
        and amp_correction_evidence.get("file_sha256")
        == sha256_file(amp_diagnostic_path)
        and amp_correction_evidence.get("content_sha256")
        == amp_diagnostic.get("content_sha256")
        and amp_correction.get("forbidden_changes", {}).get("failed_amp_group_skip")
        is True
        and amp_correction.get("claim_boundary", {}).get(
            "commercial_claim_authorized"
        )
        is False
    ):
        _fail("AMP gradient-scale correction authorization drifted")

    preliminary_quarantine_path = (
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "DISCOVERY_ATTEMPT_000_AUTHORITY_MISMATCH_QUARANTINE_V5.json"
    )
    preliminary_correction_path = (
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V5.json"
    )
    final_quarantine_path = (
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "DISCOVERY_ATTEMPT_000_AUTHORITY_MISMATCH_FINAL_QUARANTINE_V6.json"
    )
    final_correction_path = (
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V6.json"
    )
    final_quarantine = load_json(final_quarantine_path)
    incident = final_quarantine.get("incident", {})
    stable = final_quarantine.get("final_stable_quarantined_tree", {})
    process = final_quarantine.get("host_process_and_execution_ledger", {})
    genesis_binding = final_quarantine.get("active_v6_usage_chain_genesis", {})
    if not (
        final_quarantine.get("classification")
        == "pre_discovery_authority_mismatch_final_quarantine_and_budget_carry_forward"
        and incident.get("outer_test_features_or_targets_opened") is False
        and incident.get("quarantined_validation_values_used_for_variant_selection_or_promotion")
        is False
        and stable.get("epochs_durably_checkpointed") == 5
        and stable.get("optimizer_steps_durably_checkpointed") == 25
        and stable.get("unit_completion_receipt_created") is False
        and stable.get("discovery_completion_seal_created") is False
        and stable.get("partial_output", {}).get("history.json", {}).get("file_sha256")
        == "5c7d92d635d8a1991424f4f1c62607492d7818d447a6aed0fa42aa2864c576e6"
        and stable.get("partial_output", {}).get("best.pt", {}).get("file_sha256")
        == "7dabe60981ee972d971d88b46ff8cb5cb40945e14c79960c4dbbf98d37b3f9dd"
        and stable.get("partial_output", {}).get("last.pt", {}).get("file_sha256")
        == "332b9f7efadcf773e2bd9f421ce4b6a3eeb51dec7b0f003b49610b83208d1bcc"
        and process.get("final_termination", {}).get(
            "host_process_scan_after_termination_found_matching_processes"
        )
        is False
        and process.get("final_termination", {}).get(
            "conservative_budget_charge_seconds"
        )
        == 377.0
        and genesis_binding.get("record_sha256")
        == V6_USAGE_GENESIS_RECORD_SHA256
        and final_quarantine.get("claim_boundary", {}).get(
            "commercial_claim_authorized"
        )
        is False
    ):
        _fail("final V6 authority-mismatch quarantine evidence drifted")
    for path in (preliminary_quarantine_path, preliminary_correction_path):
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            _fail(f"superseded preliminary evidence became writable: {path}")
    for path in (final_quarantine_path, final_correction_path):
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            _fail(f"final V6 governance evidence became writable: {path}")

    final_correction = load_json(final_correction_path)
    quarantine_evidence = final_correction.get("final_quarantine_evidence", {})
    supersession = final_correction.get("supersession", {})
    invariants = final_correction.get("required_invariants", {})
    if not (
        final_correction.get("classification")
        == "pre_discovery_adaptive_v3r1_authority_and_gpu_usage_integrity_correction_authorization"
        and quarantine_evidence.get("file_sha256")
        == sha256_file(final_quarantine_path)
        and quarantine_evidence.get("content_sha256")
        == final_quarantine.get("content_sha256")
        and supersession.get("preliminary_v5_quarantine", {}).get("authoritative")
        is False
        and supersession.get("preliminary_v5_correction", {}).get("authoritative")
        is False
        and invariants.get("active_authorization_path_owned_by_validator") is True
        and invariants.get("completed_receipt_requires_current_ledger_membership")
        is True
        and invariants.get("changing_usage_ledger_path_cannot_reuse_completed_units_or_reset_budget")
        is True
        and final_correction.get("forbidden_changes", {}).get(
            "trainer_model_loss_schedule_or_optimization_change"
        )
        is True
        and final_correction.get("claim_boundary", {}).get(
            "commercial_claim_authorized"
        )
        is False
    ):
        _fail("V6 authority/usage-integrity correction authorization drifted")

    fixture_addendum_path = (
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V6A.json"
    )
    fixture_addendum = load_json(fixture_addendum_path)
    authorized = fixture_addendum.get("authorized_modifications", [])
    fixture_paths = [item.get("path") for item in authorized if isinstance(item, Mapping)]
    if not (
        fixture_addendum.get("classification")
        == "pre_discovery_adaptive_v3r1_usage_integrity_test_fixture_additive_authorization"
        and fixture_addendum.get("parent_correction", {}).get("file_sha256")
        == sha256_file(final_correction_path)
        and fixture_addendum.get("issue", {}).get("production_behavior_is_incorrect")
        is False
        and fixture_addendum.get("issue", {}).get("test_fixture_is_incomplete")
        is True
        and fixture_paths
        == [
            "tests/test_locked_hfr_v3r1_oof.py",
            "scripts/validate_hfr_v3r1_authorization.py",
        ]
        and fixture_addendum.get("forbidden_changes", {}).get(
            "production_usage_validation_weakening"
        )
        is True
        and fixture_addendum.get("forbidden_changes", {}).get(
            "test_assertion_removal_or_relaxation"
        )
        is True
        and fixture_addendum.get("claim_boundary", {}).get(
            "commercial_claim_authorized"
        )
        is False
        and not (stat.S_IMODE(fixture_addendum_path.stat().st_mode) & 0o222)
    ):
        _fail("V6A usage-integrity fixture authorization drifted")
    _validate_v7_resource_safety_authorization(root)
    _validate_v7r1_adversarial_authorization(root)
    _validate_v7r2_reliability_authorization(root)
    _validate_v8_efficiency_authorization(root)
    _validate_v8r1_benchmark_loader_authorization(root)
    _validate_v8r2_benchmark_resume_authorization(root)
    _validate_v8r3_benchmark_cover_authorization(root)
    validate_v6_usage_genesis(root)


def _scan_registered_surface(root: Path) -> None:
    allowed = set(ALL_IMPLEMENTATION_PATHS)
    tokens = (
        "hfr_v3r1",
        "harmonic_feature_layout_v3r1",
        "harmonic_factor_router_models_v3r1",
        "harmonic_factor_router_snn_v3r1",
        "locked_hfr_v3r1",
        "gpu_budget_ledger",
        "run_gpu_admitted",
    )
    discovered: set[str] = set()
    for prefix in ("src", "scripts", "tests"):
        for path in (root / prefix).rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            if any(token in path.name for token in tokens):
                discovered.add(relative)
    unknown = sorted(discovered - allowed)
    if unknown:
        _fail("unregistered v3r1 implementation paths: " + ", ".join(unknown))


def _validate_gpu_budget_module_constants(path: Path) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        _fail(f"cannot parse GPU budget module constants: {error}")
    values: dict[str, Any] = {}
    for node in tree.body:
        name: str | None = None
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
                value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value_node = node.value
        if name is None or value_node is None:
            continue
        try:
            values[name] = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            continue
    expected = {
        "SCHEMA_VERSION": GPU_LIFECYCLE_SCHEMA_VERSION,
        "GPU_BUDGET_NS": GPU_BUDGET_NS,
        "HEARTBEAT_INTERVAL_NS": HEARTBEAT_INTERVAL_NS,
        "TERMINATION_GRACE_NS": TERMINATION_GRACE_NS,
        "ACCOUNTING_MARGIN_NS": ACCOUNTING_MARGIN_NS,
        "RECOVERY_MARGIN_NS": RECOVERY_MARGIN_NS,
        "LEGACY_V1_GENESIS_RECORD_SHA256": V6_USAGE_GENESIS_RECORD_SHA256,
    }
    drifted = [name for name, value in expected.items() if values.get(name) != value]
    if drifted:
        _fail("GPU budget lifecycle constants drifted: " + ", ".join(drifted))


def validate_implementation(root: Path, *, require_complete: bool) -> list[dict[str, Any]]:
    _scan_registered_surface(root)
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative in ALL_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if relative == "src/snn_rr/gpu_budget_ledger.py":
            _validate_gpu_budget_module_constants(path)
        files.append(bind_file(root, relative))
    if require_complete and missing:
        _fail("authorized implementation is incomplete: " + ", ".join(missing))
    return files


def _atomic_create_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        _fail(f"create-once artifact already exists: {path}")
    payload = dict(document)
    payload["content_sha256"] = semantic_sha256(payload)
    raw = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    os.chmod(path, 0o444)


def create_test_receipt(root: Path) -> dict[str, Any]:
    validate_contract(root)
    validate_immutable_evidence(root)
    implementation = validate_implementation(root, require_complete=True)
    unfrozen = [item["path"] for item in implementation if item["mode"] != 0o444]
    if unfrozen:
        _fail(
            "implementation must be exactly 0444 before fixed tests: "
            + ", ".join(unfrozen)
        )
    runtime_before = validate_active_runtime_ledgers(root)
    command = [
        str(root / ".venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        *FIXED_TEST_PATHS,
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1800,
    )
    if completed.returncode != 0:
        _fail("fixed implementation tests failed:\n" + completed.stdout[-12000:])
    stdout_bytes = completed.stdout.encode("utf-8")
    if len(stdout_bytes) > 4000:
        _fail("fixed-test stdout exceeds the complete receipt evidence limit")
    runtime_after = validate_active_runtime_ledgers(root)
    if runtime_after != runtime_before:
        _fail("fixed tests mutated the active GPU runtime ledgers")
    document = {
        "schema_version": 1,
        "classification": "adaptive_v3r1_fixed_implementation_test_receipt",
        "campaign_id": "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective",
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "test_paths": list(FIXED_TEST_PATHS),
        "command": command,
        "return_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stdout_tail": completed.stdout,
        "stdout_is_complete": True,
        "implementation_files": implementation,
        "base_contract_implementation_paths": list(EXPECTED_IMPLEMENTATION_PATHS),
        "additive_resource_safety_paths": list(ADDITIVE_RESOURCE_SAFETY_PATHS),
        "v8_efficiency_paths": list(V8_EFFICIENCY_PATHS),
        "gpu_budget_protocol": _fixed_gpu_protocol(),
        "runtime_ledger_prefixes": runtime_after,
        "all_tests_passed": True,
        "commercial_claim_authorized": False,
    }
    _atomic_create_json(root / TEST_RECEIPT, document)
    return load_json(root / TEST_RECEIPT)


def validate_test_receipt_document(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    implementation: Sequence[Mapping[str, Any]],
    runtime_prefixes: Mapping[str, Any],
) -> None:
    expected_command = [
        str(root / ".venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        *FIXED_TEST_PATHS,
    ]
    expected_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "contract_file_sha256",
        "test_paths",
        "command",
        "return_code",
        "stdout_sha256",
        "stdout_tail",
        "stdout_is_complete",
        "implementation_files",
        "base_contract_implementation_paths",
        "additive_resource_safety_paths",
        "v8_efficiency_paths",
        "gpu_budget_protocol",
        "runtime_ledger_prefixes",
        "all_tests_passed",
        "commercial_claim_authorized",
        "content_sha256",
    }
    stdout_sha = receipt.get("stdout_sha256")
    stdout = receipt.get("stdout_tail")
    terminal_summary = ""
    if isinstance(stdout, str) and stdout.rstrip("\n"):
        terminal_summary = stdout.rstrip("\n").splitlines()[-1]
    summary_matches = re.fullmatch(
        rf"{FIXED_TEST_COUNT} passed(?:, [1-9][0-9]* warnings?)? "
        r"in [0-9]+(?:\.[0-9]+)?s",
        terminal_summary,
    )
    forbidden_outcome = re.search(
        r"(?i)(?:\bfailed\b|\berrors?\b|no tests ran|interrupted)",
        stdout if isinstance(stdout, str) else "",
    )
    if not (
        set(receipt) == expected_keys
        and type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 1
        and receipt.get("classification")
        == "adaptive_v3r1_fixed_implementation_test_receipt"
        and receipt.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        and receipt.get("contract_file_sha256") == CONTRACT_FILE_SHA256
        and exact_json_equal(receipt.get("test_paths"), list(FIXED_TEST_PATHS))
        and exact_json_equal(receipt.get("command"), expected_command)
        and type(receipt.get("return_code")) is int
        and receipt.get("return_code") == 0
        and isinstance(stdout_sha, str)
        and len(stdout_sha) == 64
        and all(character in "0123456789abcdef" for character in stdout_sha)
        and isinstance(stdout, str)
        and receipt.get("stdout_is_complete") is True
        and len(stdout.encode("utf-8")) <= 4000
        and hashlib.sha256(stdout.encode("utf-8")).hexdigest() == stdout_sha
        and "[100%]" in stdout
        and summary_matches is not None
        and forbidden_outcome is None
        and exact_json_equal(receipt.get("implementation_files"), list(implementation))
        and exact_json_equal(
            receipt.get("base_contract_implementation_paths"),
            list(EXPECTED_IMPLEMENTATION_PATHS),
        )
        and exact_json_equal(
            receipt.get("additive_resource_safety_paths"),
            list(ADDITIVE_RESOURCE_SAFETY_PATHS),
        )
        and exact_json_equal(
            receipt.get("v8_efficiency_paths"), list(V8_EFFICIENCY_PATHS)
        )
        and exact_json_equal(receipt.get("gpu_budget_protocol"), _fixed_gpu_protocol())
        and exact_json_equal(receipt.get("runtime_ledger_prefixes"), runtime_prefixes)
        and receipt.get("all_tests_passed") is True
        and receipt.get("commercial_claim_authorized") is False
    ):
        _fail("V8 implementation test receipt fields drifted")


def create_source_snapshot(root: Path) -> dict[str, Any]:
    validate_contract(root)
    validate_immutable_evidence(root)
    implementation = validate_implementation(root, require_complete=True)
    unfrozen = [item["path"] for item in implementation if item["mode"] != 0o444]
    if unfrozen:
        _fail("implementation must be exactly 0444 before snapshot: " + ", ".join(unfrozen))
    test_receipt_path = root / TEST_RECEIPT
    require_frozen_regular_file(test_receipt_path, "implementation test receipt")
    receipt = load_json(test_receipt_path)
    verify_content_hash(receipt, path=test_receipt_path)
    runtime_prefixes = validate_active_runtime_ledgers(root)
    validate_test_receipt_document(
        root,
        receipt,
        implementation=implementation,
        runtime_prefixes=runtime_prefixes,
    )
    interpreter = (root / ".venv/bin/python").resolve()
    document = {
        "schema_version": 1,
        "classification": "adaptive_v3r1_pretrain_source_snapshot",
        "campaign_id": "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective",
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "contract_content_sha256": CONTRACT_CONTENT_SHA256,
        "implementation_test_receipt": bind_file(root, TEST_RECEIPT.as_posix()),
        "implementation_files": implementation,
        "base_contract_implementation_paths": list(EXPECTED_IMPLEMENTATION_PATHS),
        "additive_resource_safety_paths": list(ADDITIVE_RESOURCE_SAFETY_PATHS),
        "v8_efficiency_paths": list(V8_EFFICIENCY_PATHS),
        "gpu_budget_protocol": _fixed_gpu_protocol(),
        "runtime_ledger_prefixes": runtime_prefixes,
        "read_only_ancestry": [bind_file(root, path) for path in READ_ONLY_ANCESTRY],
        "entry_evidence": [bind_file(root, path) for path in ENTRY_BINDINGS],
        "environment": {
            "python_executable_resolved": str(interpreter),
            "python_executable_sha256": sha256_file(interpreter),
            "pyproject": bind_file(root, "pyproject.toml"),
        },
        "training_authorized_by_snapshot_alone": False,
        "commercial_claim_authorized": False,
    }
    _atomic_create_json(root / SOURCE_SNAPSHOT, document)
    return load_json(root / SOURCE_SNAPSHOT)


def _verify_snapshot(
    root: Path,
    *,
    require_runtime_exact: bool = False,
    admitted_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = root / SOURCE_SNAPSHOT
    require_frozen_regular_file(path, "source snapshot")
    snapshot = load_json(path)
    verify_content_hash(snapshot, path=path)
    current = validate_implementation(root, require_complete=True)
    runtime_prefixes = snapshot.get("runtime_ledger_prefixes")
    if not isinstance(runtime_prefixes, Mapping):
        _fail("V7R2 source snapshot lacks runtime ledger prefixes")
    interpreter = (root / ".venv/bin/python").resolve()
    expected_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "contract_file_sha256",
        "contract_content_sha256",
        "implementation_test_receipt",
        "implementation_files",
        "base_contract_implementation_paths",
        "additive_resource_safety_paths",
        "v8_efficiency_paths",
        "gpu_budget_protocol",
        "runtime_ledger_prefixes",
        "read_only_ancestry",
        "entry_evidence",
        "environment",
        "training_authorized_by_snapshot_alone",
        "commercial_claim_authorized",
        "content_sha256",
    }
    if not (
        set(snapshot) == expected_keys
        and type(snapshot.get("schema_version")) is int
        and snapshot.get("schema_version") == 1
        and snapshot.get("classification") == "adaptive_v3r1_pretrain_source_snapshot"
        and snapshot.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        and snapshot.get("contract_file_sha256") == CONTRACT_FILE_SHA256
        and snapshot.get("contract_content_sha256") == CONTRACT_CONTENT_SHA256
        and exact_json_equal(
            snapshot.get("implementation_test_receipt"),
            bind_file(root, TEST_RECEIPT.as_posix()),
        )
        and exact_json_equal(snapshot.get("implementation_files"), current)
        and exact_json_equal(
            snapshot.get("base_contract_implementation_paths"),
            list(EXPECTED_IMPLEMENTATION_PATHS),
        )
        and exact_json_equal(
            snapshot.get("additive_resource_safety_paths"),
            list(ADDITIVE_RESOURCE_SAFETY_PATHS),
        )
        and exact_json_equal(
            snapshot.get("v8_efficiency_paths"), list(V8_EFFICIENCY_PATHS)
        )
        and exact_json_equal(snapshot.get("gpu_budget_protocol"), _fixed_gpu_protocol())
        and exact_json_equal(
            snapshot.get("read_only_ancestry"),
            [bind_file(root, item) for item in READ_ONLY_ANCESTRY],
        )
        and exact_json_equal(
            snapshot.get("entry_evidence"),
            [bind_file(root, item) for item in ENTRY_BINDINGS],
        )
        and exact_json_equal(
            snapshot.get("environment"),
            {
                "python_executable_resolved": str(interpreter),
                "python_executable_sha256": sha256_file(interpreter),
                "pyproject": bind_file(root, "pyproject.toml"),
            },
        )
        and snapshot.get("training_authorized_by_snapshot_alone") is False
        and snapshot.get("commercial_claim_authorized") is False
    ):
        _fail("V8 source snapshot fields drifted")
    if admitted_binding is None:
        current_runtime = verify_runtime_ledger_prefixes(root, runtime_prefixes)
        if require_runtime_exact and not exact_json_equal(
            current_runtime, runtime_prefixes
        ):
            _fail("active runtime ledgers changed before pretrain authorization")
    else:
        if require_runtime_exact:
            _fail("admitted-child validation cannot issue pretrain authorization")
        verify_admitted_runtime_prefix_bytes(root, runtime_prefixes, admitted_binding)
    for item in current:
        if item["mode"] != 0o444:
            _fail(f"snapshotted implementation is not exactly 0444: {item['path']}")
    test_path = root / TEST_RECEIPT
    require_frozen_regular_file(test_path, "implementation test receipt")
    test = load_json(test_path)
    verify_content_hash(test, path=test_path)
    validate_test_receipt_document(
        root,
        test,
        implementation=current,
        runtime_prefixes=runtime_prefixes,
    )
    return snapshot


def create_pretrain_authorization(root: Path) -> dict[str, Any]:
    validate_contract(root)
    validate_immutable_evidence(root)
    snapshot = _verify_snapshot(root, require_runtime_exact=True)
    document = {
        "schema_version": 1,
        "classification": "adaptive_v3r1_pretrain_authorization",
        "campaign_id": "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective",
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "source_snapshot": bind_file(root, SOURCE_SNAPSHOT.as_posix()),
        "implementation_test_receipt": bind_file(root, TEST_RECEIPT.as_posix()),
        "all_implementation_hashes_verified": True,
        "old_v3_contract_used_as_authority": False,
        "adaptive_retrospective_only": True,
        "discovery_scope": {
            "outer_runs": [3, 4],
            "seeds": [20260828, 20260829, 20260830],
            "variants": ["H0_no_factor", "H1_factor", "H2_full"],
            "training_jobs_max": 18,
            "outer_test_features_or_targets_authorized": False,
        },
        "efficiency_benchmark_scope": {
            "phase": "efficiency_benchmark",
            "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
            "epochs": 2,
            "epoch_2_train_plus_target_free_validation_seconds_max": 23.0,
            "accuracy_metrics_authorized": False,
            "outer_test_features_or_targets_authorized": False,
            "checkpoint_selection_authorized": False,
            "training_result_reusable": False,
            "required_before_discovery": True,
        },
        "promotion_reuse_scope": {
            "selected_discovery_pointer_units": 6,
            "new_promotion_training_units": 12,
            "prediction_execution_units": 18,
            "final_gpu_execution_owners": 49,
            "nonaccounting_pointer_receipts": 6,
            "phase_independent_scientific_signature_required": True,
        },
        "admitted_child_scope": {
            "globally_closed_ledger_required_for_issuance": True,
            "exact_single_live_child_lifecycle_allowed_at_runtime": True,
            "unrelated_open_lifecycle_allowed": False,
            "direct_launch_without_inherited_wrapper_binding_allowed": False,
        },
        "gpu_hours_hard": 10.0,
        "gpu_budget_protocol": _fixed_gpu_protocol(),
        "runtime_ledger_prefixes": snapshot["runtime_ledger_prefixes"],
        "maximum_parallel_gpu_training_jobs": 1,
        "training_authorized": True,
        "promotion_authorized": False,
        "commercial_claim_authorized": False,
        "snapshot_content_sha256": snapshot["content_sha256"],
    }
    _atomic_create_json(root / PRETRAIN_AUTHORIZATION, document)
    return load_json(root / PRETRAIN_AUTHORIZATION)


def _validate_pretrain_common(
    root: Path, *, admitted_binding: Mapping[str, Any] | None
) -> dict[str, Any]:
    validate_contract(root)
    validate_immutable_evidence(root)
    if admitted_binding is not None:
        admitted_binding = revalidate_admitted_child_binding(root, admitted_binding)
    snapshot = _verify_snapshot(root, admitted_binding=admitted_binding)
    path = root / PRETRAIN_AUTHORIZATION
    require_frozen_regular_file(path, "pretrain authorization")
    authorization = load_json(path)
    verify_content_hash(authorization, path=path)
    if admitted_binding is not None:
        expected_authorization = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "mode": "0444",
        }
        if not (
            admitted_binding.get("classification")
            == "verified_v8_gpu_admitted_child_lifecycle"
            and admitted_binding.get("phase")
            in {
                "efficiency_benchmark",
                "discovery",
                "promotion_training",
                "promotion_prediction",
            }
            and exact_json_equal(
                admitted_binding.get("authorization"), expected_authorization
            )
        ):
            _fail("admitted-child pretrain authorization binding drifted")
    expected_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "contract_file_sha256",
        "source_snapshot",
        "implementation_test_receipt",
        "all_implementation_hashes_verified",
        "old_v3_contract_used_as_authority",
        "adaptive_retrospective_only",
        "discovery_scope",
        "efficiency_benchmark_scope",
        "promotion_reuse_scope",
        "admitted_child_scope",
        "gpu_hours_hard",
        "gpu_budget_protocol",
        "runtime_ledger_prefixes",
        "maximum_parallel_gpu_training_jobs",
        "training_authorized",
        "promotion_authorized",
        "commercial_claim_authorized",
        "snapshot_content_sha256",
        "content_sha256",
    }
    if not (
        set(authorization) == expected_keys
        and type(authorization.get("schema_version")) is int
        and authorization.get("schema_version") == 1
        and authorization.get("classification")
        == "adaptive_v3r1_pretrain_authorization"
        and authorization.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        and authorization.get("contract_file_sha256") == CONTRACT_FILE_SHA256
        and exact_json_equal(
            authorization.get("source_snapshot"),
            bind_file(root, SOURCE_SNAPSHOT.as_posix()),
        )
        and exact_json_equal(
            authorization.get("implementation_test_receipt"),
            bind_file(root, TEST_RECEIPT.as_posix()),
        )
        and authorization.get("all_implementation_hashes_verified") is True
        and authorization.get("old_v3_contract_used_as_authority") is False
        and authorization.get("adaptive_retrospective_only") is True
        and exact_json_equal(
            authorization.get("discovery_scope"),
            {
                "outer_runs": [3, 4],
                "seeds": [20260828, 20260829, 20260830],
                "variants": ["H0_no_factor", "H1_factor", "H2_full"],
                "training_jobs_max": 18,
                "outer_test_features_or_targets_authorized": False,
            },
        )
        and exact_json_equal(
            authorization.get("efficiency_benchmark_scope"),
            {
                "phase": "efficiency_benchmark",
                "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
                "outer_fold": 3,
                "seed": 20260828,
                "variant": "H0_no_factor",
                "epochs": 2,
                "epoch_2_train_plus_target_free_validation_seconds_max": 23.0,
                "accuracy_metrics_authorized": False,
                "outer_test_features_or_targets_authorized": False,
                "checkpoint_selection_authorized": False,
                "training_result_reusable": False,
                "required_before_discovery": True,
            },
        )
        and exact_json_equal(
            authorization.get("promotion_reuse_scope"),
            {
                "selected_discovery_pointer_units": 6,
                "new_promotion_training_units": 12,
                "prediction_execution_units": 18,
                "final_gpu_execution_owners": 49,
                "nonaccounting_pointer_receipts": 6,
                "phase_independent_scientific_signature_required": True,
            },
        )
        and exact_json_equal(
            authorization.get("admitted_child_scope"),
            {
                "globally_closed_ledger_required_for_issuance": True,
                "exact_single_live_child_lifecycle_allowed_at_runtime": True,
                "unrelated_open_lifecycle_allowed": False,
                "direct_launch_without_inherited_wrapper_binding_allowed": False,
            },
        )
        and type(authorization.get("gpu_hours_hard")) is float
        and authorization.get("gpu_hours_hard")
        == GPU_BUDGET_NS / 3_600_000_000_000
        and exact_json_equal(
            authorization.get("gpu_budget_protocol"), _fixed_gpu_protocol()
        )
        and exact_json_equal(
            authorization.get("runtime_ledger_prefixes"),
            snapshot.get("runtime_ledger_prefixes"),
        )
        and type(authorization.get("maximum_parallel_gpu_training_jobs")) is int
        and authorization.get("maximum_parallel_gpu_training_jobs") == 1
        and authorization.get("training_authorized") is True
        and authorization.get("promotion_authorized") is False
        and authorization.get("commercial_claim_authorized") is False
        and authorization.get("snapshot_content_sha256") == snapshot["content_sha256"]
    ):
        _fail("V8 pretrain authorization fields drifted")
    return {
        "valid": True,
        "phase": "pretrain",
        "campaign_id": "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective",
        "adaptive_retrospective_only": True,
        "training_authorized": True,
        "efficiency_benchmark_authorized": True,
        "discovery_requires_passing_efficiency_benchmark": True,
        "promotion_reuse_scope": authorization["promotion_reuse_scope"],
        "admitted_child_scope": authorization["admitted_child_scope"],
        "promotion_authorized": False,
        "commercial_claim_authorized": False,
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "source_snapshot_file_sha256": sha256_file(root / SOURCE_SNAPSHOT),
        "pretrain_authorization_path": PRETRAIN_AUTHORIZATION.as_posix(),
        "pretrain_authorization_file_sha256": sha256_file(path),
        "gpu_budget_protocol": _fixed_gpu_protocol(),
        "gpu_lifecycle_schema_version": GPU_LIFECYCLE_SCHEMA_VERSION,
        "gpu_budget_ns": GPU_BUDGET_NS,
        "heartbeat_interval_ns": HEARTBEAT_INTERVAL_NS,
        "termination_grace_ns": TERMINATION_GRACE_NS,
        "accounting_margin_ns": ACCOUNTING_MARGIN_NS,
        "gpu_usage_ledger_path": V6_USAGE_LEDGER.as_posix(),
        "gpu_usage_genesis_record_sha256": V6_USAGE_GENESIS_RECORD_SHA256,
        "gpu_execution_ledger_path": V7_GPU_EXECUTION_LEDGER.as_posix(),
        "runtime_ledger_prefixes": snapshot["runtime_ledger_prefixes"],
    }


def validate_pretrain(root: Path) -> dict[str, Any]:
    """Validate V8 issuance/campaign entry while the runtime ledgers are closed."""

    return _validate_pretrain_common(root, admitted_binding=None)


def validate_pretrain_admitted_child(
    root: Path, admitted_binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the same stable pretrain summary inside one admitted workload."""

    if not isinstance(admitted_binding, Mapping):
        _fail("admitted-child binding must be an object")
    return _validate_pretrain_common(root, admitted_binding=admitted_binding)


def validate_phase(root: Path, phase: str) -> dict[str, Any]:
    validate_contract(root)
    validate_immutable_evidence(root)
    files = validate_implementation(root, require_complete=phase != "implementation")
    if phase == "implementation":
        return {
            "valid": True,
            "phase": phase,
            "campaign_id": "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective",
            "implementation_authorized": True,
            "training_authorized": False,
            "present_authorized_files": len(files),
            "expected_authorized_files": len(ALL_IMPLEMENTATION_PATHS),
            "base_contract_authorized_files": len(EXPECTED_IMPLEMENTATION_PATHS),
            "additive_resource_safety_files": len(ADDITIVE_RESOURCE_SAFETY_PATHS),
            "v8_efficiency_files": len(V8_EFFICIENCY_PATHS),
            "active_pretrain_authorization_path": PRETRAIN_AUTHORIZATION.as_posix(),
            "gpu_budget_protocol": _fixed_gpu_protocol(),
            "commercial_claim_authorized": False,
        }
    return validate_pretrain(root)


# ---------------------------------------------------------------------------
# Active V8R4 / V8R4A issuance layer
# ---------------------------------------------------------------------------
#
# The functions above remain the byte-for-byte historical V8R3 replay
# implementation.  V8R4 corrected the physical data capability and V8R4A
# migrated mutable ledgers into three dedicated directories.  Keeping the new
# issuance layer separate prevents historical documents from being silently
# reinterpreted while making the public validator names point only at the
# current, fail-closed chain.

ACTIVE_TEST_RECEIPT = CAMPAIGN_DIR / (
    "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTEXT1.json"
)
ACTIVE_SOURCE_SNAPSHOT = CAMPAIGN_DIR / "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTEXT1.json"
ACTIVE_PRETRAIN_AUTHORIZATION = CAMPAIGN_DIR / (
    "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"
)
ACTIVE_V8R4_CORRECTION = CAMPAIGN_DIR / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4.json"
ACTIVE_V8R4A_CORRECTION = CAMPAIGN_DIR / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A.json"
ACTIVE_V8R4_FAILURE_DIAGNOSTIC = CAMPAIGN_DIR / (
    "diagnostics/v3r1_v8r3_outer_capability_and_identity_dtype_failure_v8r4.json"
)
ACTIVE_V8R4A_FAILURE_DIAGNOSTIC = CAMPAIGN_DIR / (
    "diagnostics/v3r1_v8r4_exact_file_atomic_replace_failure_v8r4a.json"
)
ACTIVE_MIGRATION_RECEIPT = CAMPAIGN_DIR / "GPU_STATE_MIGRATION_RECEIPT_V8R4A.json"
ACTIVE_SOURCE_CLOSURE_CORRECTION = CAMPAIGN_DIR / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_SOURCE_CLOSURE.json"
)
ACTIVE_SOURCE_CLOSURE_DEPENDENCIES = CAMPAIGN_DIR / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_SOURCE_CLOSURE_DEPENDENCIES.json"
)
ACTIVE_KILL_SAFE_CORRECTION = CAMPAIGN_DIR / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_KILL_SAFE_RESUME.json"
)
ACTIVE_SOURCE_CLOSURE_DIAGNOSTIC = CAMPAIGN_DIR / (
    "diagnostics/v3r1_v8r4a_pretrain_validator_and_source_closure_failure.json"
)
ACTIVE_KILL_SAFE_DIAGNOSTIC = CAMPAIGN_DIR / (
    "diagnostics/v3r1_v8r4a_kill_safe_output_and_shared_ledger_resume_failure.json"
)
ACTIVE_OPEN_LIFECYCLE_RECOVERY_CORRECTION = CAMPAIGN_DIR / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_OPEN_LIFECYCLE_RECOVERY.json"
)
ACTIVE_OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC = CAMPAIGN_DIR / (
    "diagnostics/v3r1_v8r4a_open_gpu_lifecycle_kill_recovery_failure.json"
)
ACTIVE_EXECUTION_CLOSURE_CORRECTION = CAMPAIGN_DIR / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_EXECUTION_CLOSURE.json"
)
ACTIVE_EXECUTION_CLOSURE_DIAGNOSTIC = CAMPAIGN_DIR / (
    "diagnostics/v3r1_v8r4a_terminal_execution_closure_failure.json"
)
ACTIVE_MIGRATION_SOURCE_SUCCESSION_CORRECTION = CAMPAIGN_DIR / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_MIGRATION_SOURCE_SUCCESSION.json"
)
ACTIVE_MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC = CAMPAIGN_DIR / (
    "diagnostics/v3r1_v8r4a_migrated_state_source_succession_failure.json"
)
ACTIVE_FD_CLOSURE_CORRECTION = CAMPAIGN_DIR / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_FD_CLOSURE.json"
)
ACTIVE_FD_CLOSURE_DIAGNOSTIC = CAMPAIGN_DIR / (
    "diagnostics/v3r1_v8r4a_outer_guard_urandom_descriptor_failure.json"
)
ACTIVE_CANARY_BOUNDARY_CORRECTION = CAMPAIGN_DIR / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_CANARY_BOUNDARY.json"
)
ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC = CAMPAIGN_DIR / (
    "diagnostics/v3r1_v8r4a_denied_canary_prefix_collision_failure.json"
)
ACTIVE_FROZEN_CONTRACT_ENCODING_CORRECTION = CAMPAIGN_DIR / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_FROZEN_CONTRACT_ENCODING.json"
)
ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC = CAMPAIGN_DIR / (
    "diagnostics/v3r1_v8r4a_frozen_contract_encoding_false_rejection_failure.json"
)
ACTIVE_GPU_STATE_PARENT_BIND_CORRECTION = CAMPAIGN_DIR / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_GPU_STATE_PARENT_BIND.json"
)
ACTIVE_GPU_STATE_PARENT_BIND_DIAGNOSTIC = CAMPAIGN_DIR / (
    "diagnostics/v3r1_v8r4a_gpu_state_parent_mount_identity_failure.json"
)
ACTIVE_ADMITTED_CONTEXT_CORRECTION = CAMPAIGN_DIR / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_BENCHMARK_ADMITTED_CONTEXT.json"
)
ACTIVE_ADMITTED_CONTEXT_DIAGNOSTIC = CAMPAIGN_DIR / (
    "diagnostics/"
    "v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_failure.json"
)
ACTIVE_SCIENTIFIC_REVISION = "V8R4"
ACTIVE_INFRASTRUCTURE_REVISION = "V8R4A"
ACTIVE_AUTHORIZATION_GENERATION = "CONTEXT1"
ACTIVE_TEST_CLASSIFICATION = "adaptive_v3r1_v8r4a_implementation_test_receipt"
ACTIVE_SNAPSHOT_CLASSIFICATION = "adaptive_v3r1_v8r4a_source_snapshot"
ACTIVE_PRETRAIN_CLASSIFICATION = "pretrain_adaptive_v3r1_v8r4a_authorization"

ACTIVE_USAGE_LEDGER = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "gpu_state_v8r4a/usage/campaign_gpu_usage_chain_v6.jsonl"
)
ACTIVE_EXECUTION_LEDGER = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "gpu_state_v8r4a/execution/gpu_execution_ledger_v7.jsonl"
)
ACTIVE_ADMISSION_LOCK = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "gpu_state_v8r4a/admission/gpu_admission_v7.lock"
)
ACTIVE_STATE_ROOT = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/gpu_state_v8r4a"
)

ACTIVE_IMPLEMENTATION_PATHS = (
    "src/snn_rr/__init__.py",
    "src/snn_rr/harmonic_factor_router_v3.py",
    "src/snn_rr/svd_episode_models.py",
    "src/snn_rr/models.py",
    "src/snn_rr/harmonic_feature_layout_v3r1.py",
    "src/snn_rr/harmonic_factor_router_models_v3r1.py",
    "configs/harmonic_factor_router_v3.yaml",
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json",
    "pyproject.toml",
    "scripts/train_harmonic_factor_router_snn_v3r1.py",
    "scripts/run_hfr_v3r1_discovery_campaign.py",
    "scripts/select_hfr_v3r1_common_variant.py",
    "scripts/run_fixed_hfr_v3r1_oof_campaign.py",
    "scripts/build_locked_hfr_v3r1_test_inputs.py",
    "scripts/validate_hfr_v3r1_authorization.py",
    "tests/test_harmonic_feature_layout_v3r1.py",
    "tests/test_harmonic_factor_router_models_v3r1.py",
    "tests/test_train_harmonic_factor_router_snn_v3r1.py",
    "tests/test_run_hfr_v3r1_campaign.py",
    "tests/test_locked_hfr_v3r1_oof.py",
    "src/snn_rr/gpu_budget_ledger.py",
    "scripts/run_gpu_admitted.py",
    "tests/test_run_gpu_admitted.py",
    "scripts/benchmark_hfr_v3r1_efficiency.py",
    "tests/test_benchmark_hfr_v3r1_efficiency.py",
    "scripts/build_hfr_v3r1_sealed_input_pack_v8r4.py",
    "scripts/run_hfr_v3r1_target_sealed.py",
    "tests/test_build_hfr_v3r1_sealed_input_pack_v8r4.py",
    "tests/test_run_hfr_v3r1_target_sealed.py",
    "scripts/migrate_hfr_v3r1_gpu_state_v8r4a.py",
    "tests/test_migrate_hfr_v3r1_gpu_state_v8r4a.py",
    "tests/test_validate_hfr_v3r1_authorization_v8r4a.py",
    "scripts/run_hfr_v3r1_v8r4a_campaign.py",
    "tests/test_hfr_v3r1_execution_closure_sigkill.py",
    "tests/test_run_hfr_v3r1_v8r4a_campaign.py",
)
ACTIVE_FIXED_TEST_PATHS = (
    "tests/test_harmonic_feature_layout_v3r1.py",
    "tests/test_harmonic_factor_router_models_v3r1.py",
    "tests/test_train_harmonic_factor_router_snn_v3r1.py",
    "tests/test_run_hfr_v3r1_campaign.py",
    "tests/test_locked_hfr_v3r1_oof.py",
    "tests/test_run_gpu_admitted.py",
    "tests/test_benchmark_hfr_v3r1_efficiency.py",
    "tests/test_build_hfr_v3r1_sealed_input_pack_v8r4.py",
    "tests/test_run_hfr_v3r1_target_sealed.py",
    "tests/test_migrate_hfr_v3r1_gpu_state_v8r4a.py",
    "tests/test_validate_hfr_v3r1_authorization_v8r4a.py",
    "tests/test_hfr_v3r1_execution_closure_sigkill.py",
    "tests/test_run_hfr_v3r1_v8r4a_campaign.py",
)

ACTIVE_HISTORICAL_FILES = {
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_TEST_RECEIPT_V8R3.json": (
        "0da3aa9450e316554d40056aeb1cabac1945faebc55320ce50b3524a18d81479"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "V3R1_SOURCE_SNAPSHOT_V8R3.json": (
        "067fbd38a72fe2d3ec00a6645a8cb4a928f22175986d2558ca0c2c646cd97629"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "PRETRAIN_AUTHORIZATION_V8R3.json": (
        "26ef02cf9f5abb8ec44ed4f82c0f3e738a46f4f5a5e1719ef94a087aee2bd10f"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "DISCOVERY_V8R3_ATTEMPT_000_QUARANTINE_OWNER_RECEIPT_V8R4.json": (
        "b53d65f7d107d3f82f033a9c8ed4cb835884cead13b3685f284b058b440660b5"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "DISCOVERY_V8R3_ATTEMPT_000_QUARANTINED_OUTPUT_SEAL_V8R4.json": (
        "12261aec9e199311dc89a07733fb00c4fa5753ac045298ba49526e84859a7ef3"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v8r3_outer_capability_and_identity_dtype_failure_v8r4.json": (
        "547bb5640f3a29d125e36d7b00b63247e0ef424925ddec6e0619441d403be149"
    ),
    ACTIVE_V8R4_CORRECTION.as_posix(): (
        "d5d0d79f449240b37f0b83473d09268bee81a70bf4423b152b4eccd320ca9170"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v8r4_exact_file_atomic_replace_failure_v8r4a.json": (
        "5b7606a3ba0c86ce46e24d53049dc92c32553f8d9b12774944a2b2d2f4b78ad5"
    ),
    ACTIVE_V8R4A_CORRECTION.as_posix(): (
        "5427853d4c530d5996a4bc63f12c7cd5fef5ca505a6d754c392546a17d2b6b30"
    ),
    ACTIVE_SOURCE_CLOSURE_DIAGNOSTIC.as_posix(): (
        "6c8bd045bfcccf9cb25a311666ab7ca9350e4de5479110b6650ded6a06f81f9d"
    ),
    ACTIVE_SOURCE_CLOSURE_CORRECTION.as_posix(): (
        "fa7bcf26acbd5ca85f1037247038ba6a144bec7177bc1bb4ecc3fd63723c435a"
    ),
    ACTIVE_SOURCE_CLOSURE_DEPENDENCIES.as_posix(): (
        "aa099de0148e8ed3233c2768b4940d9aa4b7b1fd55ddbb66608c5bf9505d9d02"
    ),
    ACTIVE_KILL_SAFE_DIAGNOSTIC.as_posix(): (
        "77dfab728e9e9438a79efbadf528ff33ce1bca1edf532c33aa7e7469ac070d54"
    ),
    ACTIVE_KILL_SAFE_CORRECTION.as_posix(): (
        "57753ce48d18ce354bc523141238a53b9c296f4da232348b4902f6c482b04d0c"
    ),
    ACTIVE_OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC.as_posix(): (
        "a9cd41355ff98502153d61ad83d7f01da0d3c52462acdd24ade7bef73cf80b5e"
    ),
    ACTIVE_OPEN_LIFECYCLE_RECOVERY_CORRECTION.as_posix(): (
        "92b7e3e4b911dbf7450e3447b84f5a5762aee1212348c151cd57f20f10f5e1f6"
    ),
    ACTIVE_EXECUTION_CLOSURE_DIAGNOSTIC.as_posix(): (
        "0ca492c98f94e73e21873c41287c24d3d135466c4ca4d085388f9c39e5d9560e"
    ),
    ACTIVE_EXECUTION_CLOSURE_CORRECTION.as_posix(): (
        "11e5e7d00d3d837e0eb542b3482499cd21bf90b38883d0f4a0f86b68ba16d754"
    ),
    ACTIVE_MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC.as_posix(): (
        "265eb0cb62f6412d26bc7491ad959c8b3d6e49ffc47241573ed0fecf5111ac1e"
    ),
    ACTIVE_MIGRATION_SOURCE_SUCCESSION_CORRECTION.as_posix(): (
        "4a3673a406f49287b5abe16cc9ddde5d90d55f3a18d82a346ed390b55ccd91d9"
    ),
    ACTIVE_FD_CLOSURE_DIAGNOSTIC.as_posix(): (
        "75766bbbcc2e1bdc6cdcc61ddee559a2ffa647586f7cbbb87fea0696034d8fbd"
    ),
    ACTIVE_FD_CLOSURE_CORRECTION.as_posix(): (
        "1ad3bdaa0b78937c5b6ce98bc2e4e02d31e41951baf57dde6d68aa8029b25110"
    ),
    ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC.as_posix(): (
        "7ea5aea6f661dc0e3157a6ee71e20b2e5e04da5a4dc81035e652e9902ecb6294"
    ),
    ACTIVE_CANARY_BOUNDARY_CORRECTION.as_posix(): (
        "8187bd7c306419114c25020cf2a89c5a740d766b12df5ddd2e6d29ca498a84c3"
    ),
    ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC.as_posix(): (
        "1d358bdecb1a5605b138b1916451f8c306f957a27257d328aa71de8d83cbf8ee"
    ),
    ACTIVE_FROZEN_CONTRACT_ENCODING_CORRECTION.as_posix(): (
        "2a59c7de8b8aefcbb3d21691db5a8da1e4b8f1a5461024f0338f389ff8ddcda1"
    ),
    ACTIVE_GPU_STATE_PARENT_BIND_DIAGNOSTIC.as_posix(): (
        "fd2aa011020289d94ab0548c679888ecc924d3ec179ca5908560a6e82074d628"
    ),
    ACTIVE_GPU_STATE_PARENT_BIND_CORRECTION.as_posix(): (
        "b73d68199acad6fff780c76f05bd3daadc62b03c160af6efc407792efa87a4cd"
    ),
    ACTIVE_ADMITTED_CONTEXT_DIAGNOSTIC.as_posix(): (
        "b7a360902a68c4a7cb72d320c2042bccaf965a6ea9df64b0d203a40dc64dd088"
    ),
    ACTIVE_ADMITTED_CONTEXT_CORRECTION.as_posix(): (
        "c0646a3fb0e5b673850e570f7d0a1e91676e5116890d1a8e758e6603bbfa31e2"
    ),
    ACTIVE_MIGRATION_RECEIPT.as_posix(): (
        "d70c921eba40907c76122a8492841d1f490abae4cb4c20058dc340f829582f31"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_TEST_RECEIPT_V8R4A_ROOTBIND1.json": (
        "5654eb89eab4ccb97f20633dbe1832e8600694312d3b52162c0d8f1711f57ec5"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "V3R1_SOURCE_SNAPSHOT_V8R4A_ROOTBIND1.json": (
        "8ea5873fd2ebf43d975db123f4551e7d3aa849ff4aa404dfb5c862c23b735cae"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "PRETRAIN_AUTHORIZATION_V8R4A_ROOTBIND1.json": (
        "49ba2637b9c957d382f83c8847198f129eda2f08c099c184f717003a1129fba6"
    ),
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "target_sealed_lifecycle_v8r4a_rootbind1/efficiency_benchmark/"
    "benchmark_hfr_v3r1_efficiency/outer_3/"
    "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json": (
        "b6ceccf7b4d3f0738de1cbead9038fe937a209295294a4af772a902dccbd20d8"
    ),
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "target_sealed_lifecycle_v8r4a_rootbind1/efficiency_benchmark/"
    "benchmark_hfr_v3r1_efficiency/outer_3/"
    "TARGET_SEALED_COMPLETION_RECEIPT_V8R4A.json": (
        "e08456bcdecddb10e38e7378837f785314dd8494125d2363b1063df7b4723747"
    ),
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "efficiency_benchmark_v8r4a_rootbind1/BENCHMARK_INVOCATION_V8R4.json": (
        "a52232d9dc4428550039149589d8f0e718b39d146f7d271c195710a26ef8a3f9"
    ),
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "efficiency_benchmark_v8r4a_rootbind1/attempts/attempt_000/invocation.json": (
        "e06bbc723706fd6756f3224dc806ac54cfab6fe8f7852da1f7c372f740730961"
    ),
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "efficiency_benchmark_v8r4a_rootbind1/attempts/attempt_000/"
    "GPU_TERMINAL_RESULT.json": (
        "b575caa298db286cad2a3ad3231aa84dcdb2af76ab04d15ea38eec7b1a50fbda"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTRACT1.json": (
        "ccf4f35c817b7540fb13c760712bc61c471fbcbbea31994ab50e74cb1863c23a"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTRACT1.json": (
        "6841386e58b0390689f60cfddd58f54fb688132f2119afcf8a592fee2d68c1d2"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "PRETRAIN_AUTHORIZATION_V8R4A_CONTRACT1.json": (
        "6c8f072481cbcf5b5ac7971a2c3eeb8c4b7d2cf8ae84a946583294d1ec68584d"
    ),
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "target_sealed_lifecycle_v8r4a_contract1/efficiency_benchmark/"
    "benchmark_hfr_v3r1_efficiency/outer_3/"
    "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json": (
        "13a1fe9e922a4c945e32ac756cc29fe054b36f958c7163fc460932408b086c0f"
    ),
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "target_sealed_lifecycle_v8r4a_contract1/efficiency_benchmark/"
    "benchmark_hfr_v3r1_efficiency/outer_3/"
    "TARGET_SEALED_COMPLETION_RECEIPT_V8R4A.json": (
        "6c54435f50ab6f1157894b22a5a94c0b76d33e75a66d762e83a29c29a8bb6f91"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CANARY1.json": (
        "646ac34da7ed1032b21cdffc3a65885b9ac7a13210bbf704d7995b58970f27fe"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "V3R1_SOURCE_SNAPSHOT_V8R4A_CANARY1.json": (
        "fe313e568b2e0dc9a19ef6d3d4be397d04244f18249fc6ba017d129d1491f53c"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "PRETRAIN_AUTHORIZATION_V8R4A_CANARY1.json": (
        "fb051a410499599b3cabd5418fd338c3a39248e05f9db21de91047aea1672d07"
    ),
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "target_sealed_lifecycle_v8r4a/efficiency_benchmark/"
    "benchmark_hfr_v3r1_efficiency/outer_3/"
    "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json": (
        "5bf98a9be31ef92aea43a1c777fd2ab5725317e1efe7b003d8283c725f16206d"
    ),
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "target_sealed_lifecycle_v8r4a/efficiency_benchmark/"
    "benchmark_hfr_v3r1_efficiency/outer_3/"
    "TARGET_SEALED_COMPLETION_RECEIPT_V8R4A.json": (
        "d16ca7006448386f569fd50c9ae7be61066219dd5dd93d417bb7784d3f2226d2"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_TEST_RECEIPT_V8R4A_FD1.json": (
        "0c8414408439af38b7c8a0ac5b5a81967185131ba9ec81b87849b793b553616c"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "V3R1_SOURCE_SNAPSHOT_V8R4A_FD1.json": (
        "debf86e706856c2b55ba590cf33fa7c9b3fe53b8b29372190c87d18d5bb3c783"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "PRETRAIN_AUTHORIZATION_V8R4A_FD1.json": (
        "f9b18fb1186123b4fee77265601b1165fb7b47e5d321b8277905434e3337b79d"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_TEST_RECEIPT_V8R4A.json": (
        "cb7bb51558d89ea79728063a33fe719edc9f813416d356bf78665b303b56a5b4"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "V3R1_SOURCE_SNAPSHOT_V8R4A.json": (
        "4a0278d146255caa2a50669d1ad05750b672988142598db8b8160cec9b50ccf1"
    ),
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "PRETRAIN_AUTHORIZATION_V8R4A.json": (
        "f42cc8e1a144eacb0440fb6c7287c3a3fac8b38c47258b65000e2708b002b6e3"
    ),
}
ACTIVE_PACK_INDEX_FILES = {
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/v8r4_split_inputs/"
    "V8R4_DISCOVERY_SHARD_AGGREGATOR.json": (
        "0f397c0c8d25c6e1db912aabc812ccff60ea593389977dc12cda4667d340c553"
    ),
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/v8r4_split_inputs/"
    "discovery_shard_outer_3/V8R4_NONOUTER_TRAINING_INDEX.json": (
        "db204cdba72e9f3023c58ef37c1761cfd4ec2f4310449f8eaeeef7003afadb9b"
    ),
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/v8r4_split_inputs/"
    "discovery_shard_outer_4/V8R4_NONOUTER_TRAINING_INDEX.json": (
        "1ce1b1a0154c609b1fa1693ff5702b3e81be178f8ddbdd7ec25f559c39029f0a"
    ),
}


def _active_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _active_read_binding(
    root: Path,
    relative: str | Path,
    *,
    require_frozen: bool = True,
) -> tuple[dict[str, Any], bytes]:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        _fail(f"active binding path is not project-relative: {relative_path}")
    path = root / relative_path
    try:
        lexical = os.lstat(path)
    except OSError as error:
        _fail(f"active binding is unavailable: {relative_path}: {error}")
    if (
        stat.S_ISLNK(lexical.st_mode)
        or not stat.S_ISREG(lexical.st_mode)
        or lexical.st_nlink != 1
    ):
        _fail(f"active binding is aliased or non-regular: {relative_path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        _fail(f"active binding cannot be resolved: {relative_path}: {error}")
    if (
        resolved != path.absolute()
    ):
        _fail(f"active binding is aliased or non-regular: {relative_path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
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
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or before.st_mode != after.st_mode
        or (after.st_dev, after.st_ino, after.st_mode)
        != (named.st_dev, named.st_ino, named.st_mode)
        or after.st_nlink != 1
        or len(raw) != after.st_size
    ):
        _fail(f"active binding changed while hashing: {relative_path}")
    mode = stat.S_IMODE(after.st_mode)
    if require_frozen and mode != 0o444:
        _fail(f"active binding must be 0444: {relative_path}")
    return {
        "path": relative_path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "mode": f"{mode:04o}",
        "nlink": after.st_nlink,
        "st_dev": after.st_dev,
        "st_ino": after.st_ino,
    }, raw


def _active_binding(
    root: Path,
    relative: str | Path,
    *,
    require_frozen: bool = True,
) -> dict[str, Any]:
    binding, _raw = _active_read_binding(
        root, relative, require_frozen=require_frozen
    )
    return binding


def _active_snapshot_binding(
    root: Path,
    relative: str | Path,
    *,
    require_frozen: bool = True,
) -> dict[str, Any]:
    binding = _active_binding(
        root, relative, require_frozen=require_frozen
    )
    return {
        "path": binding["path"],
        "file_sha256": binding["sha256"],
        "size_bytes": binding["bytes"],
        "mode": int(str(binding["mode"]), 8),
    }


def _active_load_document(
    root: Path,
    relative: str | Path,
    *,
    classification: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding, raw = _active_read_binding(root, relative)
    path = root / binding["path"]
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: _fail(
                f"non-finite JSON constant {token}"
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(f"invalid JSON {path}: {error}")
    if not isinstance(document, dict):
        _fail(f"JSON root is not an object: {path}")
    canonical = (
        json.dumps(
            document,
            indent=2,
            # Preserve the frozen document's insertion order.  Several
            # correction addenda were create-once encoded in schema order;
            # their exact file SHA-256 and semantic content hash, not a later
            # lexical reordering, are the authority.
            sort_keys=False,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    relative_path = Path(binding["path"])
    exact_frozen_contract_encoding = (
        relative_path == CONTRACT
        and binding["sha256"] == CONTRACT_FILE_SHA256
        and binding["bytes"] == CONTRACT_FILE_BYTES
        and document.get("schema_version") == 1
        and document.get("classification")
        == "adaptive_retrospective_historical_cohort_engineering_not_confirmatory"
        and document.get("content_sha256") == CONTRACT_CONTENT_SHA256
    )
    if raw != canonical and not exact_frozen_contract_encoding:
        _fail(f"active governance JSON encoding is noncanonical: {relative}")
    verify_content_hash(document, path=path)
    if classification is not None and document.get("classification") != classification:
        _fail(f"active governance classification drifted: {relative}")
    if document.get("campaign_id") != (
        "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
    ):
        _fail(f"active governance campaign id drifted: {relative}")
    return document, binding


_ACTIVE_UNSET = object()
_TARGET_CAPABILITY_CLASSIFICATION = (
    "adaptive_v3r1_v8r4a_outer_target_sealed_runtime_capability_receipt"
)
_TARGET_CAPABILITY_KEYS = {
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
_TARGET_CAPABILITY_FILE_KEYS = {
    "path",
    "sha256",
    "bytes",
    "st_dev",
    "st_ino",
    "mode",
}
_TARGET_REQUIRED_TRUE_BOUNDARIES = {
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
    "benchmark_admitted_context_generation_isolated",
    "exactly_three_mutable_state_directory_mounts",
    "usage_and_execution_closed_prelaunch",
    "lifecycle_mounted_read_only",
    "source_snapshot_exact_file_mounts",
}
_TARGET_SECURITY_BOUNDARY_KEYS = _TARGET_REQUIRED_TRUE_BOUNDARIES | {
    "pid_namespace_unshared",
    "new_session_created",
    "hai_experiment_propagated",
    "legacy_combined_cache_mounted",
    "raw_or_target_root_mounted",
    "cross_outer_shard_mounted",
    "other_pack_or_output_mounted",
    "admitted_child_fd_created_or_consumed_by_outer_launcher",
    "admission_lock_fd_created_or_consumed_by_outer_launcher",
    "target_reference_or_selection_evidence_accessed",
    "commercial_claim_authorized",
    "production_execution_authorized",
    "synthetic_validation_only",
    "v8r4a_ledger_migration_required",
    "complete_project_source_or_config_trees_mounted",
}
_TARGET_DYNAMIC_BOUNDARIES = {
    "production_execution_authorized",
    "synthetic_validation_only",
}
_TARGET_WRITABLE_ROLES = {"output", "lifecycle", "usage", "execution", "admission"}
_TARGET_GPU_STATE_DIRECTORY_ROLES = {"usage", "execution", "admission"}
_TARGET_GOVERNANCE_ROLE_PATHS = {
    "active_authorization": ACTIVE_PRETRAIN_AUTHORIZATION,
    "source_snapshot": ACTIVE_SOURCE_SNAPSHOT,
    "implementation_test_receipt": ACTIVE_TEST_RECEIPT,
    "campaign_contract": CONTRACT,
    "gpu_state_migration_receipt": ACTIVE_MIGRATION_RECEIPT,
    "correction_authorization": ACTIVE_V8R4_CORRECTION,
    "infrastructure_correction_authorization": ACTIVE_V8R4A_CORRECTION,
    "failure_diagnostic": ACTIVE_V8R4_FAILURE_DIAGNOSTIC,
    "infrastructure_failure_diagnostic": ACTIVE_V8R4A_FAILURE_DIAGNOSTIC,
    "source_closure_correction_authorization": ACTIVE_SOURCE_CLOSURE_CORRECTION,
    "source_closure_dependency_authorization": ACTIVE_SOURCE_CLOSURE_DEPENDENCIES,
    "source_closure_failure_diagnostic": ACTIVE_SOURCE_CLOSURE_DIAGNOSTIC,
    "kill_safe_correction_authorization": ACTIVE_KILL_SAFE_CORRECTION,
    "kill_safe_failure_diagnostic": ACTIVE_KILL_SAFE_DIAGNOSTIC,
    "open_lifecycle_recovery_correction_authorization": (
        ACTIVE_OPEN_LIFECYCLE_RECOVERY_CORRECTION
    ),
    "open_lifecycle_recovery_failure_diagnostic": (
        ACTIVE_OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC
    ),
    "execution_closure_correction_authorization": (
        ACTIVE_EXECUTION_CLOSURE_CORRECTION
    ),
    "execution_closure_failure_diagnostic": (
        ACTIVE_EXECUTION_CLOSURE_DIAGNOSTIC
    ),
    "migration_source_succession_correction_authorization": (
        ACTIVE_MIGRATION_SOURCE_SUCCESSION_CORRECTION
    ),
    "migration_source_succession_failure_diagnostic": (
        ACTIVE_MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC
    ),
    "fd_closure_correction_authorization": ACTIVE_FD_CLOSURE_CORRECTION,
    "fd_closure_failure_diagnostic": ACTIVE_FD_CLOSURE_DIAGNOSTIC,
    "canary_boundary_correction_authorization": (
        ACTIVE_CANARY_BOUNDARY_CORRECTION
    ),
    "canary_boundary_failure_diagnostic": ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC,
    "frozen_contract_encoding_correction_authorization": (
        ACTIVE_FROZEN_CONTRACT_ENCODING_CORRECTION
    ),
    "frozen_contract_encoding_failure_diagnostic": (
        ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC
    ),
    "gpu_state_parent_bind_correction_authorization": (
        ACTIVE_GPU_STATE_PARENT_BIND_CORRECTION
    ),
    "gpu_state_parent_bind_failure_diagnostic": (
        ACTIVE_GPU_STATE_PARENT_BIND_DIAGNOSTIC
    ),
    "admitted_context_correction_authorization": (
        ACTIVE_ADMITTED_CONTEXT_CORRECTION
    ),
    "admitted_context_failure_diagnostic": ACTIVE_ADMITTED_CONTEXT_DIAGNOSTIC,
}
_TARGET_COMMON_GOVERNANCE_ROLES = set(_TARGET_GOVERNANCE_ROLE_PATHS)
_TARGET_GOVERNANCE_ROLES_BY_PHASE = {
    "efficiency_benchmark": _TARGET_COMMON_GOVERNANCE_ROLES
    | {"sealed_pack_index"},
    "discovery": _TARGET_COMMON_GOVERNANCE_ROLES
    | {
        "sealed_pack_index",
        "benchmark_receipt",
        "quarantine_owner_receipt",
        "quarantined_output_seal",
        *(f"quarantined_material_{number:02d}" for number in range(11)),
    },
    "promotion_training": _TARGET_COMMON_GOVERNANCE_ROLES
    | {
        "sealed_pack_index",
        "discovery_completion_seal",
        "selection_lock",
        "promotion_authorization",
    },
    "promotion_prediction": _TARGET_COMMON_GOVERNANCE_ROLES
    | {
        "sealed_pack_index",
        "discovery_completion_seal",
        "selection_lock",
        "promotion_authorization",
    },
    "discovery_aggregation": _TARGET_COMMON_GOVERNANCE_ROLES
    | {"discovery_shard_seal_outer3", "discovery_shard_seal_outer4"},
    "promotion_aggregation": _TARGET_COMMON_GOVERNANCE_ROLES
    | {
        "selection_lock",
        "promotion_authorization",
        *(f"model_source_seal_outer{outer}" for outer in range(6)),
        *(f"prediction_shard_seal_outer{outer}" for outer in range(6)),
    },
}

_TARGET_RUNS_ROOT = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1"
)
_ROOTBIND1_LIFECYCLE_ROOT = (
    _TARGET_RUNS_ROOT / "target_sealed_lifecycle_v8r4a_rootbind1"
)
_ROOTBIND1_BENCHMARK_OUTPUT_ROOT = (
    _TARGET_RUNS_ROOT / "efficiency_benchmark_v8r4a_rootbind1"
)
_TARGET_LIFECYCLE_ROOT = (
    _TARGET_RUNS_ROOT / "target_sealed_lifecycle_v8r4a_context1"
)
_TARGET_SUPERSEDED_LIFECYCLE_ROOT = (
    _TARGET_RUNS_ROOT / "target_sealed_lifecycle_v8r4a"
)
_TARGET_SUPERSEDED_BENCHMARK_OUTPUT_ROOT = (
    _TARGET_RUNS_ROOT / "efficiency_benchmark_v8r4a"
)
_TARGET_ACTIVE_BENCHMARK_OUTPUT_ROOT = (
    _TARGET_RUNS_ROOT / "efficiency_benchmark_v8r4a_context1"
)
_TARGET_SUPERSEDED_CONTRACT1_LIFECYCLE_ROOT = (
    _TARGET_RUNS_ROOT / "target_sealed_lifecycle_v8r4a_contract1"
)
_TARGET_SUPERSEDED_CONTRACT1_BENCHMARK_OUTPUT_ROOT = (
    _TARGET_RUNS_ROOT / "efficiency_benchmark_v8r4a_contract1"
)
_TARGET_REQUIRED_SUPERSEDED_CANARIES = {
    "superseded_v8r4a_lifecycle_root": _TARGET_SUPERSEDED_LIFECYCLE_ROOT,
    "superseded_v8r4a_output_root": _TARGET_SUPERSEDED_BENCHMARK_OUTPUT_ROOT,
    "superseded_v8r4a_contract1_lifecycle_root": (
        _TARGET_SUPERSEDED_CONTRACT1_LIFECYCLE_ROOT
    ),
    "superseded_v8r4a_contract1_output_root": (
        _TARGET_SUPERSEDED_CONTRACT1_BENCHMARK_OUTPUT_ROOT
    ),
    "superseded_v8r4a_rootbind1_lifecycle_root": _ROOTBIND1_LIFECYCLE_ROOT,
    "superseded_v8r4a_rootbind1_output_root": _ROOTBIND1_BENCHMARK_OUTPUT_ROOT,
}
_TARGET_ENTRY_BY_PHASE = {
    "efficiency_benchmark": "benchmark_hfr_v3r1_efficiency.py",
    "discovery": "run_hfr_v3r1_discovery_campaign.py",
    "promotion_training": "run_fixed_hfr_v3r1_oof_campaign.py",
    "promotion_prediction": "run_fixed_hfr_v3r1_oof_campaign.py",
    "discovery_aggregation": "run_hfr_v3r1_discovery_campaign.py",
    "promotion_aggregation": "run_fixed_hfr_v3r1_oof_campaign.py",
}


def _target_expected_roots(
    root: Path, *, phase: str, outer_fold: int | None, entry_name: str
) -> tuple[Path, Path]:
    if entry_name != _TARGET_ENTRY_BY_PHASE[phase]:
        _fail("target capability entry/phase drifted")
    if phase == "efficiency_benchmark":
        output = _TARGET_ACTIVE_BENCHMARK_OUTPUT_ROOT
    elif phase == "discovery":
        output = _TARGET_RUNS_ROOT / "discovery_v8r4/shards" / f"outer_{outer_fold}"
    elif phase == "promotion_training":
        output = _TARGET_RUNS_ROOT / "fixed_oof_v8r4/promotion_training_shards" / f"outer_{outer_fold}"
    elif phase == "promotion_prediction":
        output = _TARGET_RUNS_ROOT / "fixed_oof_v8r4/prediction_shards" / f"outer_{outer_fold}"
    elif phase == "discovery_aggregation":
        output = _TARGET_RUNS_ROOT / "discovery_v8r4/aggregation_v8r4a"
    elif phase == "promotion_aggregation":
        output = _TARGET_RUNS_ROOT / "fixed_oof_v8r4/aggregation_v8r4a"
    else:
        _fail("target capability phase topology drifted")
    scope = "global" if outer_fold is None else f"outer_{outer_fold}"
    lifecycle = _TARGET_LIFECYCLE_ROOT / phase / Path(entry_name).stem / scope
    return root / output, root / lifecycle


def _active_decode_document_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: _fail(
                f"non-finite JSON constant {token} in {label}"
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(f"invalid {label} JSON: {error}")
    if not isinstance(value, dict):
        _fail(f"{label} JSON root is not an object")
    return value


def _active_capability_file_binding(
    root: Path, row: Any, *, role: str, expected: Path | None = None
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(row, Mapping) or set(row) != _TARGET_CAPABILITY_FILE_KEYS:
        _fail(f"target capability governance binding schema drifted: {role}")
    raw_path = row.get("path")
    if not isinstance(raw_path, str):
        _fail(f"target capability governance path drifted: {role}")
    path = Path(raw_path)
    if not path.is_absolute():
        _fail(f"target capability governance path is not absolute: {role}")
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail(f"target capability governance path escapes project root: {role}")
    if expected is not None and relative.as_posix() != expected.as_posix():
        _fail(f"target capability governance role path drifted: {role}")
    binding, raw = _active_read_binding(root, relative)
    if not (
        row.get("sha256") == binding["sha256"]
        and row.get("bytes") == binding["bytes"]
        and row.get("st_dev") == binding["st_dev"]
        and row.get("st_ino") == binding["st_ino"]
        and row.get("mode") == binding["mode"]
    ):
        _fail(f"target capability governance binding drifted: {role}")
    return binding, raw


def _active_denied_capability_roots(root: Path) -> tuple[Path, ...]:
    return tuple(
        root / relative
        for _role, relative in sorted(_TARGET_REQUIRED_SUPERSEDED_CANARIES.items())
    )


def _active_path_intersects_denied_root(
    path: Path, denied_roots: Sequence[Path], *, reject_ancestor: bool
) -> bool:
    return any(
        path == denied
        or denied in path.parents
        or (reject_ancestor and path in denied.parents)
        for denied in denied_roots
    )


def _active_preflight_capability_path(
    raw_path: Any,
    *,
    root: Path,
    denied_roots: Sequence[Path],
    label: str,
    reject_ancestor: bool,
    project_relative: bool = False,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        _fail(f"target capability {label} path is invalid")
    supplied = Path(raw_path)
    if any(part in {".", ".."} for part in raw_path.split("/")):
        _fail(f"target capability {label} path contains traversal")
    if project_relative:
        if supplied.is_absolute():
            _fail(f"target capability {label} path must be project-relative")
        lexical = root / supplied
    else:
        if not supplied.is_absolute():
            _fail(f"target capability {label} path must be absolute")
        lexical = supplied
    lexical = Path(os.path.abspath(lexical))
    if _active_path_intersects_denied_root(
        lexical, denied_roots, reject_ancestor=reject_ancestor
    ):
        _fail(f"target capability {label} intersects a denied historical root")
    try:
        resolved = lexical.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        _fail(f"target capability {label} path cannot be resolved safely: {error}")
    if _active_path_intersects_denied_root(
        resolved, denied_roots, reject_ancestor=reject_ancestor
    ):
        _fail(
            f"target capability {label} resolves into a denied historical root"
        )
    return lexical


def _active_preflight_capability_paths(
    root: Path,
    document: Mapping[str, Any],
    *,
    denied_roots: Sequence[Path],
) -> None:
    """Reject cross-generation paths before opening or statting their sources."""

    governance = document.get("governance_files")
    if not isinstance(governance, Mapping):
        _fail("target capability governance paths are absent")
    for role, row in governance.items():
        if not isinstance(row, Mapping):
            _fail(f"target capability governance path row drifted: {role}")
        path = _active_preflight_capability_path(
            row.get("path"),
            root=root,
            denied_roots=denied_roots,
            label=f"governance:{role}",
            reject_ancestor=True,
        )
        expected = _TARGET_GOVERNANCE_ROLE_PATHS.get(str(role))
        if expected is not None and path != root / expected:
            _fail(f"target capability governance role path drifted: {role}")
        if role in _TARGET_COMMON_GOVERNANCE_ROLES and expected is None:
            _fail(f"target common governance role lacks a canonical path: {role}")

    for role in ("bubblewrap", "launcher", "interpreter", "sealed_pack_root", "sealed_pack_index"):
        row = document.get(role)
        if row is None:
            continue
        if not isinstance(row, Mapping):
            _fail(f"target capability {role} binding drifted")
        raw_path = row.get("path")
        if raw_path is not None:
            _active_preflight_capability_path(
                raw_path,
                root=root,
                denied_roots=denied_roots,
                label=role,
                reject_ancestor=True,
            )

    writable = document.get("writable_roots")
    if not isinstance(writable, Mapping):
        _fail("target capability writable paths are absent")
    for role, row in writable.items():
        if not isinstance(row, Mapping):
            _fail(f"target capability writable path row drifted: {role}")
        _active_preflight_capability_path(
            row.get("path"),
            root=root,
            denied_roots=denied_roots,
            label=f"writable:{role}",
            reject_ancestor=True,
        )

    prelaunch = document.get("prelaunch_gpu_state")
    if isinstance(prelaunch, Mapping):
        migration = prelaunch.get("migration_receipt")
        if isinstance(migration, Mapping) and migration.get("path") is not None:
            _active_preflight_capability_path(
                migration.get("path"),
                root=root,
                denied_roots=denied_roots,
                label="prelaunch migration receipt",
                reject_ancestor=True,
                project_relative=not Path(str(migration.get("path"))).is_absolute(),
            )
        for group in ("directories", "files"):
            rows = prelaunch.get(group)
            if not isinstance(rows, Mapping):
                continue
            for role, row in rows.items():
                if not isinstance(row, Mapping) or row.get("path") is None:
                    continue
                raw_path = str(row["path"])
                _active_preflight_capability_path(
                    raw_path,
                    root=root,
                    denied_roots=denied_roots,
                    label=f"prelaunch {group}:{role}",
                    reject_ancestor=True,
                    project_relative=not Path(raw_path).is_absolute(),
                )

    mount_spec = document.get("mount_specification")
    if not isinstance(mount_spec, list):
        _fail("target capability mount paths are absent")
    host_backed = {"ro_bind_fd", "rw_bind_fd", "rw_bind_file_fd", "dev_bind_fd"}
    for number, row in enumerate(mount_spec):
        if not isinstance(row, Mapping):
            _fail(f"target capability mount row {number} is not an object")
        destination = row.get("destination")
        if destination is not None:
            _active_preflight_capability_path(
                destination,
                root=root,
                denied_roots=denied_roots,
                label=f"mount destination {number}",
                reject_ancestor=row.get("kind") in host_backed,
            )
        source = row.get("source")
        if isinstance(source, Mapping) and source.get("path") is not None:
            _active_preflight_capability_path(
                source.get("path"),
                root=root,
                denied_roots=denied_roots,
                label=f"mount source {number}",
                reject_ancestor=True,
            )

    command = document.get("command")
    if isinstance(command, list):
        for number, argument in enumerate(command):
            if not isinstance(argument, str):
                continue
            if ".." in argument.split("/"):
                _fail(f"target capability command argument {number} contains traversal")
            if argument.startswith("-") and "=/" in argument:
                _fail(
                    f"target capability command argument {number} embeds an absolute path"
                )
            if isinstance(argument, str) and argument.startswith("/"):
                _active_preflight_capability_path(
                    argument,
                    root=root,
                    denied_roots=denied_roots,
                    label=f"command argument {number}",
                    reject_ancestor=False,
                )
    environment = document.get("environment")
    if isinstance(environment, Mapping):
        for name, value in environment.items():
            if not isinstance(name, str) or not isinstance(value, str):
                continue
            for number, component in enumerate(value.split(":")):
                if component.startswith("/"):
                    _active_preflight_capability_path(
                        component,
                        root=root,
                        denied_roots=denied_roots,
                        label=f"environment {name}:{number}",
                        reject_ancestor=False,
                    )


def _active_directory_mount_source(path: Path, *, label: str) -> dict[str, Any]:
    try:
        status = os.stat(path, follow_symlinks=True)
    except OSError as error:
        _fail(f"target capability {label} directory is unavailable: {error}")
    if not stat.S_ISDIR(status.st_mode):
        _fail(f"target capability {label} source is not a directory")
    return {
        "path": str(path),
        "st_dev": status.st_dev,
        "st_ino": status.st_ino,
        "mode": f"{stat.S_IMODE(status.st_mode):04o}",
    }


def _active_file_mount_source(root: Path, relative: str | Path) -> dict[str, Any]:
    binding = _active_binding(root, relative)
    return {
        "path": str(root / Path(relative)),
        "sha256": binding["sha256"],
        "bytes": binding["bytes"],
        "st_dev": binding["st_dev"],
        "st_ino": binding["st_ino"],
        "mode": binding["mode"],
    }


def _active_validate_capability_bind_mount_cover(
    root: Path,
    document: Mapping[str, Any],
    mount_spec: Sequence[Any],
    *,
    expected_state_mounts: Sequence[Mapping[str, Any]],
    writable: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Require every host-backed RO/RW bind to have one canonical owner."""

    expected_by_destination: dict[str, dict[str, Any]] = {}

    def add(row: Mapping[str, Any], *, label: str) -> None:
        destination = row.get("destination")
        if not isinstance(destination, str) or not destination:
            _fail(f"target capability {label} mount destination drifted")
        value = dict(row)
        prior = expected_by_destination.get(destination)
        if prior is not None and not exact_json_equal(prior, value):
            _fail(f"target capability bind owners conflict at {destination}")
        expected_by_destination[destination] = value

    for system_path in (Path("/usr"), Path("/sys")):
        add(
            {
                "destination": str(system_path),
                "kind": "ro_bind_fd",
                "source": _active_directory_mount_source(
                    system_path, label=str(system_path)
                ),
            },
            label="system",
        )

    command = document.get("command")
    interpreter = document.get("interpreter")
    if isinstance(command, list) and command and Path(str(command[0])).is_absolute():
        venv_root = Path(str(command[0])).parent.parent
    else:
        venv_root = None
    if isinstance(interpreter, Mapping) and isinstance(interpreter.get("path"), str):
        interpreter_path = Path(str(interpreter["path"]))
        if not interpreter_path.is_absolute():
            _fail("target capability interpreter path is not absolute")
        runtime_root = interpreter_path.parent.parent
        runtime_source = _active_directory_mount_source(
            runtime_root, label="Python runtime"
        )
        add(
            {
                "destination": str(runtime_root),
                "kind": "ro_bind_fd",
                "source": runtime_source,
            },
            label="Python runtime",
        )
        if venv_root is not None:
            add(
                {
                    "destination": str(venv_root),
                    "kind": "ro_bind_fd",
                    "source": _active_directory_mount_source(
                        venv_root, label="Python virtualenv"
                    ),
                },
                label="Python virtualenv",
            )
            lexical_interpreter = Path(str(command[0]))
            if lexical_interpreter.is_symlink():
                raw_target = Path(os.readlink(lexical_interpreter))
                alias_binary = (
                    raw_target
                    if raw_target.is_absolute()
                    else lexical_interpreter.parent / raw_target
                )
                alias_root = Path(os.path.abspath(alias_binary)).parent.parent
                if alias_root != runtime_root:
                    add(
                        {
                            "destination": str(alias_root),
                            "kind": "ro_bind_fd",
                            "source": {
                                **runtime_source,
                                "alias_destination": str(alias_root),
                            },
                        },
                        label="Python runtime alias",
                    )

    sealed_pack = document.get("sealed_pack_root")
    if isinstance(sealed_pack, Mapping) and isinstance(sealed_pack.get("path"), str):
        add(
            {
                "destination": str(sealed_pack["path"]),
                "kind": "ro_bind_fd",
                "source": dict(sealed_pack),
            },
            label="sealed pack",
        )

    governance = document.get("governance_files")
    if not isinstance(governance, Mapping):
        _fail("target capability governance mount cover is absent")
    for role, row in governance.items():
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            _fail(f"target capability governance mount drifted: {role}")
        add(
            {
                "destination": str(row["path"]),
                "kind": "ro_bind_fd",
                "source": dict(row),
            },
            label=f"governance:{role}",
        )

    snapshot_row = governance.get("source_snapshot")
    if snapshot_row is not None:
        _snapshot_binding, snapshot_raw = _active_capability_file_binding(
            root,
            snapshot_row,
            role="source_snapshot",
            expected=ACTIVE_SOURCE_SNAPSHOT,
        )
        snapshot = _active_decode_document_bytes(
            snapshot_raw, label="target source snapshot mount cover"
        )
        implementation = snapshot.get("implementation_files")
        if not (
            isinstance(implementation, list)
            and [row.get("path") for row in implementation if isinstance(row, Mapping)]
            == list(ACTIVE_IMPLEMENTATION_PATHS)
        ):
            _fail("target source snapshot mount path cover drifted")
        for relative in ACTIVE_IMPLEMENTATION_PATHS:
            source = _active_file_mount_source(root, relative)
            add(
                {
                    "destination": source["path"],
                    "kind": "ro_bind_fd",
                    "source": source,
                },
                label=f"source snapshot:{relative}",
            )

    for row in expected_state_mounts:
        add(row, label="GPU state")
    for role, kind in (("lifecycle", "ro_bind_fd"), ("output", "rw_bind_fd")):
        binding = writable[role]
        add(
            {
                "destination": str(binding["path"]),
                "kind": kind,
                "source": dict(binding),
            },
            label=role,
        )

    observed = [
        row
        for row in mount_spec
        if isinstance(row, Mapping)
        and row.get("kind") in {"ro_bind_fd", "rw_bind_fd", "rw_bind_file_fd"}
    ]
    observed_destinations = [row.get("destination") for row in observed]
    expected_rows = list(expected_by_destination.values())
    if not (
        len(observed_destinations) == len(set(observed_destinations))
        and len(observed) == len(expected_rows)
        and sorted(canonical_bytes(row) for row in observed)
        == sorted(canonical_bytes(row) for row in expected_rows)
    ):
        _fail("target capability host-backed bind mount cover drifted")
    return expected_by_destination


_ACTIVE_FIXED_RUNTIME_DIRECTORIES = (
    "/run",
    "/run/snn_rr",
    "/tmp/home",
    "/tmp/cache",
    "/tmp/torch",
    "/tmp/triton",
    "/tmp/numba",
    "/tmp/pycache",
)
_ACTIVE_CUDA_DEVICE_BASENAMES = {
    "nvidiactl",
    "nvidia-uvm",
    "nvidia-uvm-tools",
    "nvidia-modeset",
}


def _active_canonical_device_mount(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != {
        "kind",
        "destination",
        "source",
    }:
        _fail("target capability CUDA device mount schema drifted")
    destination = row.get("destination")
    source = row.get("source")
    if not (
        row.get("kind") == "dev_bind_fd"
        and isinstance(destination, str)
        and isinstance(source, Mapping)
        and set(source) == {"mode", "path", "st_dev", "st_ino", "st_rdev"}
        and source.get("path") == destination
    ):
        _fail("target capability CUDA device mount binding drifted")
    path = Path(destination)
    basename = path.name
    if not (
        path.parent == Path("/dev")
        and (
            basename in _ACTIVE_CUDA_DEVICE_BASENAMES
            or re.fullmatch(r"nvidia[0-9]+", basename) is not None
        )
    ):
        _fail("target capability CUDA device path is outside the canonical policy")
    try:
        status = os.stat(path, follow_symlinks=True)
    except OSError as error:
        _fail(f"target capability CUDA device is unavailable: {error}")
    expected_source = {
        "path": destination,
        "st_dev": status.st_dev,
        "st_ino": status.st_ino,
        "st_rdev": status.st_rdev,
        "mode": f"{stat.S_IMODE(status.st_mode):04o}",
    }
    if not (
        stat.S_ISCHR(status.st_mode)
        and exact_json_equal(source, expected_source)
    ):
        _fail("target capability CUDA device identity drifted")
    return dict(row)


def _active_canonical_capability_mount_specification(
    root: Path,
    document: Mapping[str, Any],
    bind_mounts: Mapping[str, Mapping[str, Any]],
    mount_spec: Sequence[Any],
) -> list[dict[str, Any]]:
    """Reconstruct the sole canonical full bwrap mount specification."""

    symlinks: list[dict[str, Any]] = []
    for destination in ("/bin", "/sbin", "/lib", "/lib64"):
        path = Path(destination)
        if not path.is_symlink():
            _fail(f"target capability system compatibility link drifted: {path}")
        target = os.readlink(path)
        if Path(target).is_absolute() or ".." in Path(target).parts:
            _fail(f"target capability system compatibility link is unsafe: {path}")
        symlinks.append(
            {
                "kind": "symlink",
                "destination": destination,
                "source": {"target": target},
            }
        )

    project_destinations = {
        Path(destination)
        for destination in bind_mounts
        if Path(destination) == root or root in Path(destination).parents
    }
    skeleton = {
        root,
        root / "scripts",
        root / "src",
        root / "tests",
        root / "configs",
    }
    for destination in project_destinations:
        cursor = destination.parent
        while cursor != root and root in cursor.parents:
            skeleton.add(cursor)
            cursor = cursor.parent
        skeleton.add(root)
    skeleton.discard(root / ACTIVE_STATE_ROOT)
    skeleton_rows = [
        {"kind": "directory", "destination": str(path)}
        for path in sorted(skeleton, key=lambda value: (len(value.parts), str(value)))
    ]

    device_rows = [
        _active_canonical_device_mount(row)
        for row in mount_spec
        if isinstance(row, Mapping) and row.get("kind") == "dev_bind_fd"
    ]
    if [row["destination"] for row in device_rows] != sorted(
        row["destination"] for row in device_rows
    ):
        _fail("target capability CUDA device mounts are not canonically ordered")
    if document.get("security_boundary", {}).get(
        "production_execution_authorized"
    ) is True:
        basenames = {Path(row["destination"]).name for row in device_rows}
        if not (
            {"nvidiactl", "nvidia-uvm"} <= basenames
            and any(re.fullmatch(r"nvidia[0-9]+", name) for name in basenames)
        ):
            _fail("target capability production CUDA device cover is incomplete")

    bind_rows = list(bind_mounts.values())
    state_children = {
        str(root / ACTIVE_STATE_ROOT / role)
        for role in ("admission", "execution", "usage")
    }
    early_usr = [row for row in bind_rows if row["destination"] == "/usr"]
    ordinary_binds = [
        row
        for row in bind_rows
        if row["destination"] != "/usr"
        and row["destination"] not in state_children
    ]
    child_destinations = (
        str(root / ACTIVE_STATE_ROOT / "admission"),
        str(root / ACTIVE_STATE_ROOT / "execution"),
        str(root / ACTIVE_STATE_ROOT / "usage"),
    )
    if any(destination not in bind_mounts for destination in child_destinations):
        _fail("target capability canonical GPU-state child mount is absent")
    child_binds = [bind_mounts[destination] for destination in child_destinations]
    return [
        *early_usr,
        *symlinks,
        {"kind": "proc", "destination": "/proc"},
        {"kind": "dev", "destination": "/dev"},
        {"kind": "tmpfs", "destination": "/dev/shm"},
        {"kind": "tmpfs", "destination": "/tmp"},
        *skeleton_rows,
        *ordinary_binds,
        *child_binds,
        *device_rows,
        *(
            {"kind": "directory", "destination": destination}
            for destination in _ACTIVE_FIXED_RUNTIME_DIRECTORIES
        ),
        {
            "kind": "ro_bind_data",
            "destination": "/run/snn_rr/v8r4a_runtime_spec.json",
            "source": {"classification": "v8r4a_internal_guard_spec_memfd"},
        },
    ]


def _active_validate_capability_full_mount_cover(
    root: Path,
    document: Mapping[str, Any],
    mount_spec: Sequence[Any],
    bind_mounts: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = _active_canonical_capability_mount_specification(
        root, document, bind_mounts, mount_spec
    )
    destinations = [
        row.get("destination") for row in mount_spec if isinstance(row, Mapping)
    ]
    if not (
        len(mount_spec) == len(destinations)
        and len(destinations) == len(set(destinations))
        and exact_json_equal(list(mount_spec), expected)
    ):
        _fail("target capability full mount specification drifted")


def _active_validate_target_capability(
    root: Path,
    capability_receipt_path: Path,
    *,
    expected_phase: str,
    expected_outer_fold: int | None | object,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the outer runtime capability before reading authorization."""

    if expected_phase not in {
        "efficiency_benchmark",
        "discovery",
        "promotion_training",
        "promotion_prediction",
        "discovery_aggregation",
        "promotion_aggregation",
    }:
        _fail("independent target capability phase is invalid")
    path = (
        capability_receipt_path
        if capability_receipt_path.is_absolute()
        else root / capability_receipt_path
    ).absolute()
    denied_roots = _active_denied_capability_roots(root)
    _active_preflight_capability_path(
        str(path),
        root=root,
        denied_roots=denied_roots,
        label="capability receipt",
        reject_ancestor=False,
    )
    if path.name != "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json":
        _fail("target capability receipt filename is noncanonical")
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail("target capability receipt escapes project root")
    binding, raw = _active_read_binding(root, relative)
    document = _active_decode_document_bytes(raw, label="target capability receipt")
    if set(document) != _TARGET_CAPABILITY_KEYS:
        _fail("target capability receipt schema drifted")
    if raw != canonical_bytes(document) + b"\n":
        _fail("target capability receipt encoding is noncanonical")
    verify_content_hash(document, path=path)
    mount_spec = document.get("mount_specification")
    environment = document.get("environment")
    command = document.get("command")
    boundary = document.get("security_boundary")
    governance = document.get("governance_files")
    prelaunch = document.get("prelaunch_gpu_state")
    writable = document.get("writable_roots")
    denied_canaries = document.get("denied_canaries")
    phase_roles = _TARGET_GOVERNANCE_ROLES_BY_PHASE[expected_phase]
    outer_fold = document.get("outer_fold")
    phase_outer_valid = (
        outer_fold is None
        if expected_phase in {"discovery_aggregation", "promotion_aggregation"}
        else type(outer_fold) is int
        and outer_fold
        in (
            {3}
            if expected_phase == "efficiency_benchmark"
            else {3, 4}
            if expected_phase == "discovery"
            else {0, 1, 2, 5}
            if expected_phase == "promotion_training"
            else set(range(6))
        )
    )
    pack_scope_valid = (
        document.get("sealed_pack_root") is None
        and document.get("sealed_pack_index") is None
        if expected_phase in {"discovery_aggregation", "promotion_aggregation"}
        else isinstance(document.get("sealed_pack_root"), Mapping)
        and isinstance(document.get("sealed_pack_index"), Mapping)
    )
    if not (
        document.get("schema_version") == 1
        and document.get("classification") == _TARGET_CAPABILITY_CLASSIFICATION
        and document.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        and document.get("campaign_revision") == ACTIVE_SCIENTIFIC_REVISION
        and document.get("infrastructure_revision") == ACTIVE_INFRASTRUCTURE_REVISION
        and document.get("phase") == expected_phase
        and phase_outer_valid
        and pack_scope_valid
        and (expected_outer_fold is _ACTIVE_UNSET or document.get("outer_fold") == expected_outer_fold)
        and isinstance(mount_spec, list)
        and document.get("mount_specification_sha256") == semantic_sha256(mount_spec)
        and isinstance(environment, dict)
        and document.get("environment_sha256") == semantic_sha256(environment)
        and isinstance(command, list)
        and command
        and all(isinstance(part, str) and part and "\x00" not in part for part in command)
        and document.get("command_sha256") == semantic_sha256(command)
        and isinstance(boundary, dict)
        and set(boundary) == _TARGET_SECURITY_BOUNDARY_KEYS
        and all(boundary.get(key) is True for key in _TARGET_REQUIRED_TRUE_BOUNDARIES)
        and all(
            boundary.get(key) is False
            for key in _TARGET_SECURITY_BOUNDARY_KEYS
            - _TARGET_REQUIRED_TRUE_BOUNDARIES
            - _TARGET_DYNAMIC_BOUNDARIES
        )
        and type(boundary.get("production_execution_authorized")) is bool
        and boundary.get("synthetic_validation_only")
        is (not boundary["production_execution_authorized"])
        and isinstance(governance, dict)
        and set(governance) == phase_roles
        and isinstance(writable, dict)
        and set(writable) == _TARGET_WRITABLE_ROLES
        and all(
            isinstance(row, dict)
            and set(row) == {"path", "st_dev", "st_ino", "mode"}
            and isinstance(row.get("path"), str)
            and Path(row["path"]).is_absolute()
            and type(row.get("st_dev")) is int
            and row["st_dev"] >= 0
            and type(row.get("st_ino")) is int
            and row["st_ino"] > 0
            and row.get("mode") == "0700"
            for row in writable.values()
        )
        and isinstance(prelaunch, dict)
        and set(prelaunch)
        == {"migration_receipt", "directories", "files", "usage_state", "execution_state"}
        and isinstance(prelaunch.get("directories"), Mapping)
        and set(prelaunch["directories"])
        == {"root", *_TARGET_GPU_STATE_DIRECTORY_ROLES}
        and isinstance(denied_canaries, dict)
        and all(
            isinstance(role, str)
            and isinstance(canary, str)
            and Path(canary).is_absolute()
            for role, canary in denied_canaries.items()
        )
    ):
        _fail("target capability receipt semantic binding drifted")
    expected_canaries = {
        role: str(root / relative)
        for role, relative in _TARGET_REQUIRED_SUPERSEDED_CANARIES.items()
    }
    if not (
        set(denied_canaries) == set(expected_canaries)
        and all(
            denied_canaries.get(role) == path
            for role, path in expected_canaries.items()
        )
    ):
        _fail("target capability superseded-root denied canaries drifted")
    _active_preflight_capability_paths(
        root, document, denied_roots=denied_roots
    )
    state_root_path = str(root / ACTIVE_STATE_ROOT)
    child_roles = ("admission", "execution", "usage")
    state_destinations = {
        state_root_path,
        *(str(writable[role]["path"]) for role in child_roles),
    }
    indexed_state_mounts = [
        (index, row)
        for index, row in enumerate(mount_spec)
        if isinstance(row, dict) and row.get("destination") in state_destinations
    ]
    state_mounts = [row for _index, row in indexed_state_mounts]
    prelaunch_directories = prelaunch["directories"]
    exact_state_entries = {
        "root": ["admission", "execution", "usage"],
        "admission": ["gpu_admission_v7.lock"],
        "execution": [
            "gpu_execution_ledger_v7.jsonl",
            "gpu_execution_ledger_v7.jsonl.lock",
        ],
        "usage": [
            "campaign_gpu_usage_chain_v6.jsonl",
            "campaign_gpu_usage_chain_v6.jsonl.lock",
        ],
    }

    def expected_state_source(role: str, absolute_path: str) -> dict[str, Any]:
        row = prelaunch_directories.get(role)
        if not isinstance(row, Mapping) or set(row) != {
            "exact_entries",
            "mode",
            "path",
            "st_dev",
            "st_ino",
        }:
            _fail(f"target capability prelaunch state directory drifted: {role}")
        expected_relative = (
            ACTIVE_STATE_ROOT
            if role == "root"
            else ACTIVE_STATE_ROOT / role
        ).as_posix()
        if not (
            row.get("path") == expected_relative
            and row.get("exact_entries") == exact_state_entries[role]
            and row.get("mode") == "0700"
            and type(row.get("st_dev")) is int
            and row["st_dev"] >= 0
            and type(row.get("st_ino")) is int
            and row["st_ino"] > 0
        ):
            _fail(f"target capability prelaunch state directory drifted: {role}")
        return {**dict(row), "path": absolute_path}

    expected_state_mounts = [
        {
            "destination": state_root_path,
            "kind": "ro_bind_fd",
            "source": expected_state_source("root", state_root_path),
        },
        *(
            {
                "destination": str(writable[role]["path"]),
                "kind": "rw_bind_fd",
                "source": expected_state_source(
                    role, str(writable[role]["path"])
                ),
            }
            for role in child_roles
        ),
    ]
    child_sources_match_writable = all(
        all(
            expected_state_mounts[index]["source"].get(key)
            == writable[role].get(key)
            for key in ("path", "mode", "st_dev", "st_ino")
        )
        for index, role in enumerate(child_roles, start=1)
    )
    lifecycle_path = str(writable["lifecycle"]["path"])
    output_path = str(writable["output"]["path"])
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
    bind_mounts = _active_validate_capability_bind_mount_cover(
        root,
        document,
        mount_spec,
        expected_state_mounts=expected_state_mounts,
        writable=writable,
    )
    if not (
        len(state_destinations) == 4
        and len(state_mounts) == 4
        and state_mounts == expected_state_mounts
        and child_sources_match_writable
        and not any(
            isinstance(row, dict) and row.get("kind") == "rw_bind_file_fd"
            for row in mount_spec
        )
        and lifecycle_path != output_path
        and len(lifecycle_mounts) == 1
        and lifecycle_mounts[0].get("kind") == "ro_bind_fd"
        and len(output_mounts) == 1
        and output_mounts[0].get("kind") == "rw_bind_fd"
    ):
        _fail("target capability writable mount ABI drifted")
    _active_validate_capability_full_mount_cover(
        root, document, mount_spec, bind_mounts
    )
    if len(command) < 2:
        _fail("target capability command lacks its entry")
    entry_path = Path(str(command[1]))
    if not entry_path.is_absolute() or entry_path.parent != root / "scripts":
        _fail("target capability entry path is noncanonical")
    expected_output, expected_lifecycle = _target_expected_roots(
        root,
        phase=expected_phase,
        outer_fold=(int(outer_fold) if type(outer_fold) is int else None),
        entry_name=entry_path.name,
    )
    expected_state_paths = {
        role: root / ACTIVE_STATE_ROOT / role
        for role in _TARGET_GPU_STATE_DIRECTORY_ROLES
    }
    if (
        Path(output_path) != expected_output
        or Path(lifecycle_path) != expected_lifecycle
        or path.parent != expected_lifecycle
        or any(
            Path(str(writable[role]["path"])) != expected
            for role, expected in expected_state_paths.items()
        )
    ):
        _fail("target capability phase/outer/entry topology drifted")
    normalized: dict[str, dict[str, Any]] = {}
    for role, row in governance.items():
        expected = _TARGET_GOVERNANCE_ROLE_PATHS.get(str(role))
        if role in _TARGET_COMMON_GOVERNANCE_ROLES and expected is None:
            _fail(f"target common governance role lacks a canonical path: {role}")
        active, _material = _active_capability_file_binding(
            root, row, role=str(role), expected=expected
        )
        normalized[str(role)] = active
    migration = normalized["gpu_state_migration_receipt"]
    receipt = prelaunch.get("migration_receipt")
    if not isinstance(receipt, Mapping) or not (
        receipt.get("path") == ACTIVE_MIGRATION_RECEIPT.as_posix()
        and receipt.get("sha256") == migration["sha256"]
        and receipt.get("bytes") == migration["bytes"]
        and receipt.get("st_dev") == migration["st_dev"]
        and receipt.get("st_ino") == migration["st_ino"]
        and receipt.get("mode") == "0444"
        and receipt.get("nlink") == 1
    ):
        _fail("target capability migration receipt binding drifted")
    return document, binding, normalized


def _active_validate_exact_files(root: Path) -> None:
    for relative, expected in {
        **ACTIVE_HISTORICAL_FILES,
        **ACTIVE_PACK_INDEX_FILES,
    }.items():
        binding = _active_binding(root, relative)
        if binding["sha256"] != expected:
            _fail(f"immutable V8R4/V8R4A evidence drifted: {relative}")
    for relative in ACTIVE_PACK_INDEX_FILES:
        document, _ = _active_load_document(root, relative)
        if document.get("campaign_revision") != ACTIVE_SCIENTIFIC_REVISION:
            _fail(f"sealed-pack index scientific revision drifted: {relative}")
    aggregator, _ = _active_load_document(
        root,
        next(
            path
            for path in ACTIVE_PACK_INDEX_FILES
            if path.endswith("V8R4_DISCOVERY_SHARD_AGGREGATOR.json")
        ),
    )
    if not (
        aggregator.get("classification")
        == "adaptive_v3r1_v8r4_discovery_shard_aggregator"
        and aggregator.get("status") == "complete"
        and aggregator.get("shard_count") == 2
        and aggregator.get("runtime_mount_of_aggregator_authorized") is False
        and aggregator.get("target_bearing_pack_directories_bound_by_aggregator")
        is False
    ):
        _fail("V8R4 sealed-pack aggregator boundary drifted")


_EXECUTION_CLOSURE_AUTHORITY_KEYS = {
    "authority_basis", "authorized_modifications", "campaign_id", "claim_boundary",
    "classification", "content_sha256", "created_utc", "forbidden_changes",
    "infrastructure_revision", "mandatory_invariants", "required_reauthorization",
    "schema_version", "scientific_campaign_revision",
}
_EXECUTION_CLOSURE_DIAGNOSTIC_KEYS = {
    "affected_boundary", "campaign_id", "claim_boundary", "classification",
    "content_sha256", "evidence", "failure_modes", "infrastructure_revision",
    "observed_utc", "required_correction", "schema_version",
    "scientific_campaign_revision", "status",
}
_MIGRATION_SOURCE_SUCCESSION_AUTHORITY_KEYS = {
    "authority_basis", "authorized_modifications", "campaign_id", "claim_boundary",
    "classification", "content_sha256", "created_utc", "forbidden_changes",
    "infrastructure_revision", "mandatory_invariants", "required_reauthorization",
    "schema_version", "scientific_campaign_revision",
}
_MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC_KEYS = {
    "campaign_id", "claim_boundary", "classification", "content_sha256",
    "created_utc", "demonstrated_failure", "immutable_facts",
    "required_correction", "root_cause", "schema_version",
    "scientific_campaign_revision", "status",
}
_FD_CLOSURE_AUTHORITY_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision", "created_utc",
    "authority_basis", "authorized_modifications", "mandatory_invariants",
    "forbidden_changes", "required_reauthorization", "claim_boundary",
    "content_sha256",
}
_FD_CLOSURE_DIAGNOSTIC_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision", "observed_utc",
    "status", "failed_attempt", "ledger_evidence", "root_cause",
    "reproduction", "superseded_pretrain_chain", "required_correction",
    "claim_boundary", "content_sha256",
}
_FD_CLOSURE_PARENT_PATHS = {
    "parent_implementation_test_receipt": CAMPAIGN_DIR
    / "IMPLEMENTATION_TEST_RECEIPT_V8R4A.json",
    "parent_source_snapshot": CAMPAIGN_DIR
    / "V3R1_SOURCE_SNAPSHOT_V8R4A.json",
    "parent_pretrain_authorization": CAMPAIGN_DIR
    / "PRETRAIN_AUTHORIZATION_V8R4A.json",
    "parent_execution_closure_authority": ACTIVE_EXECUTION_CLOSURE_CORRECTION,
}
_FD_CLOSURE_DIAGNOSTIC_LEGACY_BINDING = {
    "path": ACTIVE_FD_CLOSURE_DIAGNOSTIC.as_posix(),
    "file_sha256": "75766bbbcc2e1bdc6cdcc61ddee559a2ffa647586f7cbbb87fea0696034d8fbd",
    "bytes": 5207,
    "content_sha256": "31834a6b67074314b5e6440d035a040cd19c81d8408e81cf681c4e54e1350f1a",
}
_FD_CLOSURE_PARENT_BINDINGS = {
    "parent_implementation_test_receipt": {
        "path": _FD_CLOSURE_PARENT_PATHS[
            "parent_implementation_test_receipt"
        ].as_posix(),
        "file_sha256": "cb7bb51558d89ea79728063a33fe719edc9f813416d356bf78665b303b56a5b4",
        "bytes": 28291,
        "content_sha256": "fd650ea690889236e5216b5c7110d0bcfe43ee374d6d5d7a712915bdff4cdd42",
    },
    "parent_source_snapshot": {
        "path": _FD_CLOSURE_PARENT_PATHS["parent_source_snapshot"].as_posix(),
        "file_sha256": "4a0278d146255caa2a50669d1ad05750b672988142598db8b8160cec9b50ccf1",
        "bytes": 20862,
        "content_sha256": "89c4e85f6a638d660a89dff4d12e46cb430efa76f8264d80b80364caf30606ba",
    },
    "parent_pretrain_authorization": {
        "path": _FD_CLOSURE_PARENT_PATHS[
            "parent_pretrain_authorization"
        ].as_posix(),
        "file_sha256": "f42cc8e1a144eacb0440fb6c7287c3a3fac8b38c47258b65000e2708b002b6e3",
        "bytes": 10385,
        "content_sha256": "b75f18580158feaee363297731b69fd6e144c6c2219d55a9623d438cffa7f41c",
    },
    "parent_execution_closure_authority": {
        "path": ACTIVE_EXECUTION_CLOSURE_CORRECTION.as_posix(),
        "file_sha256": "11e5e7d00d3d837e0eb542b3482499cd21bf90b38883d0f4a0f86b68ba16d754",
        "bytes": 21621,
        "content_sha256": "92d96a4f513a7d7f93bbd4baf227b626106dab54e000f3a01c97b25504c58c1c",
    },
}
_FD_CLOSURE_AUTHORIZED_BEFORE = {
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
_FD_CLOSURE_SUCCESSOR_CHAIN_NAMES = {
    "new_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_FD1.json",
    "new_source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_FD1.json",
    "new_pretrain_authorization": "PRETRAIN_AUTHORIZATION_V8R4A_FD1.json",
}
_CANARY_BOUNDARY_AUTHORITY_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision", "created_utc",
    "authority_basis", "authorized_modifications", "mandatory_invariants",
    "forbidden_changes", "required_reauthorization", "claim_boundary",
    "content_sha256",
}
_CANARY_BOUNDARY_DIAGNOSTIC_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision", "observed_utc",
    "status", "failed_attempt", "ledger_evidence", "root_cause",
    "reproduction", "superseded_pretrain_chain", "required_correction",
    "claim_boundary", "content_sha256",
}
_CANARY_BOUNDARY_PARENT_PATHS = {
    "parent_fd_closure_authority": ACTIVE_FD_CLOSURE_CORRECTION,
    "parent_implementation_test_receipt": CAMPAIGN_DIR
    / "IMPLEMENTATION_TEST_RECEIPT_V8R4A_FD1.json",
    "parent_source_snapshot": CAMPAIGN_DIR
    / "V3R1_SOURCE_SNAPSHOT_V8R4A_FD1.json",
    "parent_pretrain_authorization": CAMPAIGN_DIR
    / "PRETRAIN_AUTHORIZATION_V8R4A_FD1.json",
}
_CANARY_BOUNDARY_DIAGNOSTIC_BINDING = {
    "path": ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC.as_posix(),
    "file_sha256": "7ea5aea6f661dc0e3157a6ee71e20b2e5e04da5a4dc81035e652e9902ecb6294",
    "bytes": 5551,
    "content_sha256": "00b87df937342a6d3a6f1cd13d1bf7bdc51d33df8b09f276a8fc1017ba39d63e",
}
_CANARY_BOUNDARY_PARENT_BINDINGS = {
    "parent_fd_closure_authority": {
        "path": ACTIVE_FD_CLOSURE_CORRECTION.as_posix(),
        "file_sha256": "1ad3bdaa0b78937c5b6ce98bc2e4e02d31e41951baf57dde6d68aa8029b25110",
        "bytes": 8447,
        "content_sha256": "c199171692d8c13be883ada42ad8d7b25cd44f19544f38f2ad1685fab6e498a7",
    },
    "parent_implementation_test_receipt": {
        "path": _CANARY_BOUNDARY_PARENT_PATHS[
            "parent_implementation_test_receipt"
        ].as_posix(),
        "file_sha256": "0c8414408439af38b7c8a0ac5b5a81967185131ba9ec81b87849b793b553616c",
        "bytes": 29127,
        "content_sha256": "c13cff4ce5dad8d3079a125d9218403773d2bcc464bf2038731fc97801c7d67b",
    },
    "parent_source_snapshot": {
        "path": _CANARY_BOUNDARY_PARENT_PATHS[
            "parent_source_snapshot"
        ].as_posix(),
        "file_sha256": "debf86e706856c2b55ba590cf33fa7c9b3fe53b8b29372190c87d18d5bb3c783",
        "bytes": 21597,
        "content_sha256": "85c4ed4c9df1cf67f08cfad7b240ae13ebf9ce9331d9fe2d303adfc1ba0cbdea",
    },
    "parent_pretrain_authorization": {
        "path": _CANARY_BOUNDARY_PARENT_PATHS[
            "parent_pretrain_authorization"
        ].as_posix(),
        "file_sha256": "f9b18fb1186123b4fee77265601b1165fb7b47e5d321b8277905434e3337b79d",
        "bytes": 11124,
        "content_sha256": "281168deeee7f6e503b4386ef5742a36f6074479e30263a902ebffad8d37448a",
    },
}
_CANARY_BOUNDARY_AUTHORIZED_BEFORE = {
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
_CANARY_BOUNDARY_AUTHORITY_FILE_SHA256 = (
    "8187bd7c306419114c25020cf2a89c5a740d766b12df5ddd2e6d29ca498a84c3"
)
_CANARY_BOUNDARY_AUTHORITY_CONTENT_SHA256 = (
    "318e0979a4732ff8f3b2e39f3f57ec069e352259864045f43fd7bdef54243aa4"
)
_CANARY_BOUNDARY_DIAGNOSTIC_FILE_SHA256 = (
    "7ea5aea6f661dc0e3157a6ee71e20b2e5e04da5a4dc81035e652e9902ecb6294"
)
_CANARY_BOUNDARY_DIAGNOSTIC_CONTENT_SHA256 = (
    "00b87df937342a6d3a6f1cd13d1bf7bdc51d33df8b09f276a8fc1017ba39d63e"
)
_CANARY_BOUNDARY_SUCCESSOR_CHAIN_NAMES = {
    "new_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CANARY1.json",
    "new_source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_CANARY1.json",
    "new_pretrain_authorization": "PRETRAIN_AUTHORIZATION_V8R4A_CANARY1.json",
}
_FROZEN_CONTRACT_ENCODING_AUTHORITY_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision", "created_utc",
    "authority_basis", "authorized_modifications", "mandatory_invariants",
    "forbidden_changes", "required_reauthorization", "claim_boundary",
    "content_sha256",
}
_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision", "observed_utc",
    "status", "failed_attempt", "frozen_contract_evidence", "root_cause",
    "immutable_failure_receipts", "failed_namespace_inventory",
    "ledger_evidence", "required_correction", "claim_boundary", "content_sha256",
}
_FROZEN_CONTRACT_ENCODING_PARENT_PATHS = {
    "parent_canary_boundary_authority": ACTIVE_CANARY_BOUNDARY_CORRECTION,
    "parent_canary_boundary_diagnostic": ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC,
    "parent_implementation_test_receipt": CAMPAIGN_DIR
    / "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CANARY1.json",
    "parent_source_snapshot": CAMPAIGN_DIR
    / "V3R1_SOURCE_SNAPSHOT_V8R4A_CANARY1.json",
    "parent_pretrain_authorization": CAMPAIGN_DIR
    / "PRETRAIN_AUTHORIZATION_V8R4A_CANARY1.json",
    "frozen_campaign_contract": CONTRACT,
    "failed_capability_receipt": _TARGET_SUPERSEDED_LIFECYCLE_ROOT
    / "efficiency_benchmark/benchmark_hfr_v3r1_efficiency/outer_3/"
    "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json",
    "failed_completion_receipt": _TARGET_SUPERSEDED_LIFECYCLE_ROOT
    / "efficiency_benchmark/benchmark_hfr_v3r1_efficiency/outer_3/"
    "TARGET_SEALED_COMPLETION_RECEIPT_V8R4A.json",
}
_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC_BINDING = {
    "path": ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC.as_posix(),
    "file_sha256": "1d358bdecb1a5605b138b1916451f8c306f957a27257d328aa71de8d83cbf8ee",
    "bytes": 8653,
    "content_sha256": "bec8b74c21c0b0882f9ab147f68c1aa947e259ec39bf36557f3fdcdeb86abcc7",
}
_FROZEN_CONTRACT_ENCODING_PARENT_BINDINGS = {
    "parent_canary_boundary_authority": {
        "path": ACTIVE_CANARY_BOUNDARY_CORRECTION.as_posix(),
        "file_sha256": "8187bd7c306419114c25020cf2a89c5a740d766b12df5ddd2e6d29ca498a84c3",
        "bytes": 8659,
        "content_sha256": "318e0979a4732ff8f3b2e39f3f57ec069e352259864045f43fd7bdef54243aa4",
    },
    "parent_canary_boundary_diagnostic": {
        "path": ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC.as_posix(),
        "file_sha256": "7ea5aea6f661dc0e3157a6ee71e20b2e5e04da5a4dc81035e652e9902ecb6294",
        "bytes": 5551,
        "content_sha256": "00b87df937342a6d3a6f1cd13d1bf7bdc51d33df8b09f276a8fc1017ba39d63e",
    },
    "parent_implementation_test_receipt": {
        "path": _FROZEN_CONTRACT_ENCODING_PARENT_PATHS[
            "parent_implementation_test_receipt"
        ].as_posix(),
        "file_sha256": "646ac34da7ed1032b21cdffc3a65885b9ac7a13210bbf704d7995b58970f27fe",
        "bytes": 29873,
        "content_sha256": "d47dd9e0f7e0642db7515c71158ec8cdfea6b6ffb04c760f8ba7624d9490b449",
    },
    "parent_source_snapshot": {
        "path": _FROZEN_CONTRACT_ENCODING_PARENT_PATHS[
            "parent_source_snapshot"
        ].as_posix(),
        "file_sha256": "fe313e568b2e0dc9a19ef6d3d4be397d04244f18249fc6ba017d129d1491f53c",
        "bytes": 22347,
        "content_sha256": "545069175693b981510b2219d6bc72369f7d6c4fd265bda279e79960f0bd3093",
    },
    "parent_pretrain_authorization": {
        "path": _FROZEN_CONTRACT_ENCODING_PARENT_PATHS[
            "parent_pretrain_authorization"
        ].as_posix(),
        "file_sha256": "fb051a410499599b3cabd5418fd338c3a39248e05f9db21de91047aea1672d07",
        "bytes": 11878,
        "content_sha256": "378f4233095002f240cdaef4a52a43b41583f5f512a8f8e3040bb5f1d9585110",
    },
    "frozen_campaign_contract": {
        "path": CONTRACT.as_posix(),
        "file_sha256": CONTRACT_FILE_SHA256,
        "bytes": CONTRACT_FILE_BYTES,
        "content_sha256": CONTRACT_CONTENT_SHA256,
    },
    "failed_capability_receipt": {
        "path": _FROZEN_CONTRACT_ENCODING_PARENT_PATHS[
            "failed_capability_receipt"
        ].as_posix(),
        "file_sha256": "5bf98a9be31ef92aea43a1c777fd2ab5725317e1efe7b003d8283c725f16206d",
        "bytes": 50077,
        "content_sha256": "8e49dbfeaca212673662e0ca7379796c98b26cf106069c6bbf721b7e0d56faa0",
    },
    "failed_completion_receipt": {
        "path": _FROZEN_CONTRACT_ENCODING_PARENT_PATHS[
            "failed_completion_receipt"
        ].as_posix(),
        "file_sha256": "d16ca7006448386f569fd50c9ae7be61066219dd5dd93d417bb7784d3f2226d2",
        "bytes": 7680,
        "content_sha256": "d6f79fb027609d03420eca3ef2f8327e8664617e64edfabed99a7aa93c0cac54",
    },
}
_FROZEN_CONTRACT_ENCODING_AUTHORIZED_BEFORE = {
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
_FROZEN_CONTRACT_ENCODING_AUTHORITY_FILE_SHA256 = (
    "2a59c7de8b8aefcbb3d21691db5a8da1e4b8f1a5461024f0338f389ff8ddcda1"
)
_FROZEN_CONTRACT_ENCODING_AUTHORITY_CONTENT_SHA256 = (
    "b0df4b5d34bb5f55c6254d83459f81ee297177909658d101866d9e32c6c48c6f"
)
_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC_FILE_SHA256 = (
    "1d358bdecb1a5605b138b1916451f8c306f957a27257d328aa71de8d83cbf8ee"
)
_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC_CONTENT_SHA256 = (
    "bec8b74c21c0b0882f9ab147f68c1aa947e259ec39bf36557f3fdcdeb86abcc7"
)
_FROZEN_CONTRACT_ENCODING_SUCCESSOR_CHAIN_NAMES = {
    "new_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTRACT1.json",
    "new_source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTRACT1.json",
    "new_pretrain_authorization": "PRETRAIN_AUTHORIZATION_V8R4A_CONTRACT1.json",
}
_GPU_STATE_PARENT_BIND_AUTHORITY_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision", "created_utc",
    "authority_basis", "authorized_modifications", "mandatory_invariants",
    "forbidden_changes", "required_reauthorization", "claim_boundary",
    "content_sha256",
}
_GPU_STATE_PARENT_BIND_DIAGNOSTIC_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision", "observed_utc",
    "status", "failed_attempt", "trusted_host_gpu_state_root",
    "failed_mount_topology", "independent_bubblewrap_reproduction", "root_cause",
    "immutable_failure_receipts", "failed_namespace_inventory",
    "ledger_evidence", "required_correction", "claim_boundary", "content_sha256",
}
_GPU_STATE_PARENT_BIND_PARENT_PATHS = {
    "parent_frozen_contract_authority": ACTIVE_FROZEN_CONTRACT_ENCODING_CORRECTION,
    "parent_frozen_contract_diagnostic": ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC,
    "parent_implementation_test_receipt": CAMPAIGN_DIR
    / "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTRACT1.json",
    "parent_source_snapshot": CAMPAIGN_DIR
    / "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTRACT1.json",
    "parent_pretrain_authorization": CAMPAIGN_DIR
    / "PRETRAIN_AUTHORIZATION_V8R4A_CONTRACT1.json",
    "frozen_campaign_contract": CONTRACT,
    "gpu_state_migration_receipt": ACTIVE_MIGRATION_RECEIPT,
    "failed_contract1_capability_receipt": (
        _TARGET_SUPERSEDED_CONTRACT1_LIFECYCLE_ROOT
        / "efficiency_benchmark/benchmark_hfr_v3r1_efficiency/outer_3/"
        "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json"
    ),
    "failed_contract1_completion_receipt": (
        _TARGET_SUPERSEDED_CONTRACT1_LIFECYCLE_ROOT
        / "efficiency_benchmark/benchmark_hfr_v3r1_efficiency/outer_3/"
        "TARGET_SEALED_COMPLETION_RECEIPT_V8R4A.json"
    ),
}
_GPU_STATE_PARENT_BIND_DIAGNOSTIC_BINDING = {
    "path": ACTIVE_GPU_STATE_PARENT_BIND_DIAGNOSTIC.as_posix(),
    "file_sha256": "fd2aa011020289d94ab0548c679888ecc924d3ec179ca5908560a6e82074d628",
    "bytes": 10709,
    "content_sha256": "e4a315bc83e333d31920baef4c3db0f8cb2adc3b5f7e59b73a39795986073b67",
}
_GPU_STATE_PARENT_BIND_PARENT_BINDINGS = {
    "parent_frozen_contract_authority": {
        "path": ACTIVE_FROZEN_CONTRACT_ENCODING_CORRECTION.as_posix(),
        "file_sha256": "2a59c7de8b8aefcbb3d21691db5a8da1e4b8f1a5461024f0338f389ff8ddcda1",
        "bytes": 13460,
        "content_sha256": "b0df4b5d34bb5f55c6254d83459f81ee297177909658d101866d9e32c6c48c6f",
    },
    "parent_frozen_contract_diagnostic": {
        "path": ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC.as_posix(),
        "file_sha256": "1d358bdecb1a5605b138b1916451f8c306f957a27257d328aa71de8d83cbf8ee",
        "bytes": 8653,
        "content_sha256": "bec8b74c21c0b0882f9ab147f68c1aa947e259ec39bf36557f3fdcdeb86abcc7",
    },
    "parent_implementation_test_receipt": {
        "path": _GPU_STATE_PARENT_BIND_PARENT_PATHS[
            "parent_implementation_test_receipt"
        ].as_posix(),
        "file_sha256": "ccf4f35c817b7540fb13c760712bc61c471fbcbbea31994ab50e74cb1863c23a",
        "bytes": 30657,
        "content_sha256": "195e1df10a94705df25aaf39adc5c001c2992c4fd013958520d355065d18c0ba",
    },
    "parent_source_snapshot": {
        "path": _GPU_STATE_PARENT_BIND_PARENT_PATHS[
            "parent_source_snapshot"
        ].as_posix(),
        "file_sha256": "6841386e58b0390689f60cfddd58f54fb688132f2119afcf8a592fee2d68c1d2",
        "bytes": 23133,
        "content_sha256": "897e1c9e91086820c1eef6529739571acc9bc1802867537b0c9471a6694f4cee",
    },
    "parent_pretrain_authorization": {
        "path": _GPU_STATE_PARENT_BIND_PARENT_PATHS[
            "parent_pretrain_authorization"
        ].as_posix(),
        "file_sha256": "6c8f072481cbcf5b5ac7971a2c3eeb8c4b7d2cf8ae84a946583294d1ec68584d",
        "bytes": 12666,
        "content_sha256": "ffa7e8e7abaac670961363748896ba0f01da7b01ef662785258e0c603bf6199d",
    },
    "frozen_campaign_contract": {
        "path": CONTRACT.as_posix(),
        "file_sha256": CONTRACT_FILE_SHA256,
        "bytes": CONTRACT_FILE_BYTES,
        "content_sha256": CONTRACT_CONTENT_SHA256,
    },
    "gpu_state_migration_receipt": {
        "path": ACTIVE_MIGRATION_RECEIPT.as_posix(),
        "file_sha256": "d70c921eba40907c76122a8492841d1f490abae4cb4c20058dc340f829582f31",
        "bytes": 14926,
        "content_sha256": "e73b38b390d00533243b23670e3ddbe3ec41461d8dab5336c4ff36b10431328b",
    },
    "failed_contract1_capability_receipt": {
        "path": _GPU_STATE_PARENT_BIND_PARENT_PATHS[
            "failed_contract1_capability_receipt"
        ].as_posix(),
        "file_sha256": "13a1fe9e922a4c945e32ac756cc29fe054b36f958c7163fc460932408b086c0f",
        "bytes": 52334,
        "content_sha256": "513c5ecd976254be018105b209bfb707fd7163937dcef174ea13500467846fb3",
    },
    "failed_contract1_completion_receipt": {
        "path": _GPU_STATE_PARENT_BIND_PARENT_PATHS[
            "failed_contract1_completion_receipt"
        ].as_posix(),
        "file_sha256": "6c54435f50ab6f1157894b22a5a94c0b76d33e75a66d762e83a29c29a8bb6f91",
        "bytes": 7690,
        "content_sha256": "58f190e5a86aa2fc7f6a821eb7a6b37dd4f5904e701fd4bce33f1e34756e2270",
    },
}
_GPU_STATE_PARENT_BIND_AUTHORIZED_BEFORE = {
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
_GPU_STATE_PARENT_BIND_AUTHORITY_FILE_SHA256 = (
    "b73d68199acad6fff780c76f05bd3daadc62b03c160af6efc407792efa87a4cd"
)
_GPU_STATE_PARENT_BIND_AUTHORITY_CONTENT_SHA256 = (
    "7917a0b003241181b6dd6fca6c127301538717050cff0bd63dd087fbcdaa07bf"
)
_GPU_STATE_PARENT_BIND_DIAGNOSTIC_FILE_SHA256 = (
    "fd2aa011020289d94ab0548c679888ecc924d3ec179ca5908560a6e82074d628"
)
_GPU_STATE_PARENT_BIND_DIAGNOSTIC_CONTENT_SHA256 = (
    "e4a315bc83e333d31920baef4c3db0f8cb2adc3b5f7e59b73a39795986073b67"
)
_GPU_STATE_PARENT_BIND_SUCCESSOR_CHAIN_NAMES = {
    "new_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_ROOTBIND1.json",
    "new_source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_ROOTBIND1.json",
    "new_pretrain_authorization": "PRETRAIN_AUTHORIZATION_V8R4A_ROOTBIND1.json",
}
_ADMITTED_CONTEXT_AUTHORITY_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision", "created_utc",
    "authority_basis", "authorized_modifications", "mandatory_invariants",
    "forbidden_changes", "required_reauthorization", "claim_boundary",
    "content_sha256",
}
_ADMITTED_CONTEXT_DIAGNOSTIC_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision", "status",
    "observed_utc", "failed_attempt", "immutable_failure_receipts",
    "ledger_evidence", "root_cause", "required_correction", "claim_boundary",
    "content_sha256",
}
_ADMITTED_CONTEXT_AUTHORITY_FILE_SHA256 = (
    "c0646a3fb0e5b673850e570f7d0a1e91676e5116890d1a8e758e6603bbfa31e2"
)
_ADMITTED_CONTEXT_AUTHORITY_CONTENT_SHA256 = (
    "d48ff6cb78fcf94e6d994cca96b144daca9da19f873bc8f5ef7e15246e6a1f5c"
)
_ADMITTED_CONTEXT_DIAGNOSTIC_FILE_SHA256 = (
    "b7a360902a68c4a7cb72d320c2042bccaf965a6ea9df64b0d203a40dc64dd088"
)
_ADMITTED_CONTEXT_DIAGNOSTIC_CONTENT_SHA256 = (
    "51ffa6135eec896c385878b42ecd3d6bb440fad5965532d04341cec4cb4eb83e"
)
_ADMITTED_CONTEXT_SUCCESSOR_CHAIN_NAMES = {
    "new_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTEXT1.json",
    "new_source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTEXT1.json",
    "new_pretrain_authorization": "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json",
}
_CONTEXT1_FULL_BENCHMARK_CONTEXT = {
    "campaign_revision": ACTIVE_SCIENTIFIC_REVISION,
    "infrastructure_revision": ACTIVE_INFRASTRUCTURE_REVISION,
    "authorization_generation": ACTIVE_AUTHORIZATION_GENERATION,
    "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
    "outer_fold": 3,
    "seed": 20260828,
    "variant": "H0_no_factor",
}
_ROOTBIND1_BENCHMARK_CONTEXT = {
    key: value
    for key, value in _CONTEXT1_FULL_BENCHMARK_CONTEXT.items()
    if key != "authorization_generation"
}
_CONTEXT1_EFFICIENCY_BENCHMARK_SCOPE = {
    "phase": "efficiency_benchmark",
    **_CONTEXT1_FULL_BENCHMARK_CONTEXT,
    "epochs": 2,
    "epoch_2_train_plus_target_free_validation_seconds_max": 23.0,
    "accuracy_metrics_authorized": False,
    "required_before_discovery": True,
}


def _active_expected_pretrain_scopes() -> dict[str, dict[str, Any]]:
    """Return fresh, canonical CONTEXT1 scope objects for issue and replay."""

    return {
        "discovery_scope": {
            "outer_runs": [3, 4],
            "seeds": [20260828, 20260829, 20260830],
            "variants": ["H0_no_factor", "H1_factor", "H2_full"],
            "training_jobs_max": 18,
            "outer_test_features_or_targets_authorized": False,
        },
        "efficiency_benchmark_scope": dict(
            _CONTEXT1_EFFICIENCY_BENCHMARK_SCOPE
        ),
        "promotion_reuse_scope": {
            "selected_discovery_pointer_units": 6,
            "new_promotion_training_units": 12,
            "prediction_execution_units": 18,
            "final_gpu_execution_owners": 49,
            "nonaccounting_pointer_receipts": 6,
        },
        "admitted_child_scope": {
            "globally_closed_ledger_required_for_issuance": True,
            "exact_single_live_child_lifecycle_allowed_at_runtime": True,
            "unrelated_open_lifecycle_allowed": False,
            "direct_launch_without_inherited_wrapper_binding_allowed": False,
        },
    }


def _active_validate_pretrain_scope_fields(
    authorization: Mapping[str, Any], *, label: str
) -> None:
    """Fail closed on any missing, extra, mistyped, or changed scope field."""

    expected = _active_expected_pretrain_scopes()
    if not all(
        exact_json_equal(authorization.get(role), scope)
        for role, scope in expected.items()
    ):
        _fail(f"{label} CONTEXT1 pretrain scope drifted")
_CONTEXT1_POSTFAILURE_PREFIXES = {
    "usage_ledger": {
        "path": ACTIVE_USAGE_LEDGER.as_posix(),
        "bytes": 113257,
        "sha256": "c59c9234c25c09a4cb7e7cc753942e0517acccdced5ba527336295db32dac029",
        "record_count": 77,
        "tail_record_sha256": "9da404e120c87940617fed8ad8438d9470b785b304fe48c4ec8727d50fcafafd",
        "settled_usage_ns": 1411550918574,
        "open_reservation_count": 0,
    },
    "execution_ledger": {
        "path": ACTIVE_EXECUTION_LEDGER.as_posix(),
        "bytes": 29961,
        "sha256": "8acffd84488a1f9b4f9a0a95804108127135023d0e58c46afe7ffa7b553609b5",
        "record_count": 10,
        "last_line_sha256": "a2aac7f38810230c332bc6d389d7baf8d83d29bd4cf1d0e940d259bfff3f1272",
        "open_start_count": 0,
    },
}
_ROOTBIND1_PARENT_PATHS = {
    "parent_implementation_test_receipt": CAMPAIGN_DIR
    / "IMPLEMENTATION_TEST_RECEIPT_V8R4A_ROOTBIND1.json",
    "parent_source_snapshot": CAMPAIGN_DIR
    / "V3R1_SOURCE_SNAPSHOT_V8R4A_ROOTBIND1.json",
    "parent_pretrain_authorization": CAMPAIGN_DIR
    / "PRETRAIN_AUTHORIZATION_V8R4A_ROOTBIND1.json",
    "parent_gpu_state_parent_bind_authority": ACTIVE_GPU_STATE_PARENT_BIND_CORRECTION,
    "parent_gpu_state_parent_bind_diagnostic": ACTIVE_GPU_STATE_PARENT_BIND_DIAGNOSTIC,
    "failed_rootbind1_target_capability_receipt": (
        _ROOTBIND1_LIFECYCLE_ROOT
        / "efficiency_benchmark/benchmark_hfr_v3r1_efficiency/outer_3/"
        "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json"
    ),
    "failed_rootbind1_target_completion_receipt": (
        _ROOTBIND1_LIFECYCLE_ROOT
        / "efficiency_benchmark/benchmark_hfr_v3r1_efficiency/outer_3/"
        "TARGET_SEALED_COMPLETION_RECEIPT_V8R4A.json"
    ),
    "failed_rootbind1_benchmark_invocation": (
        _ROOTBIND1_BENCHMARK_OUTPUT_ROOT / "BENCHMARK_INVOCATION_V8R4.json"
    ),
    "failed_rootbind1_gpu_invocation": (
        _ROOTBIND1_BENCHMARK_OUTPUT_ROOT
        / "attempts/attempt_000/invocation.json"
    ),
    "failed_rootbind1_gpu_terminal_result": (
        _ROOTBIND1_BENCHMARK_OUTPUT_ROOT
        / "attempts/attempt_000/GPU_TERMINAL_RESULT.json"
    ),
}
_ROOTBIND1_PARENT_BINDINGS = {
    "parent_implementation_test_receipt": {
        "path": _ROOTBIND1_PARENT_PATHS[
            "parent_implementation_test_receipt"
        ].as_posix(),
        "sha256": "5654eb89eab4ccb97f20633dbe1832e8600694312d3b52162c0d8f1711f57ec5",
        "bytes": 31505,
        "mode": "0444",
        "content_sha256": "623d0c7c86274c04f3d0f38b8032485ae2e403461f1de44d25beff6f9368c726",
    },
    "parent_source_snapshot": {
        "path": _ROOTBIND1_PARENT_PATHS["parent_source_snapshot"].as_posix(),
        "sha256": "8ea5873fd2ebf43d975db123f4551e7d3aa849ff4aa404dfb5c862c23b735cae",
        "bytes": 23900,
        "mode": "0444",
        "content_sha256": "c6d3819b3b9b52a6ec2ed6d2139eb0bb2b2b1768b97fc167b8f67cb62d4691a4",
    },
    "parent_pretrain_authorization": {
        "path": _ROOTBIND1_PARENT_PATHS[
            "parent_pretrain_authorization"
        ].as_posix(),
        "sha256": "49ba2637b9c957d382f83c8847198f129eda2f08c099c184f717003a1129fba6",
        "bytes": 13433,
        "mode": "0444",
        "content_sha256": "94dd087b03af3fb0d9e8e3726b615f003e6c9540d1445f15c315de91c9de873d",
    },
    "parent_gpu_state_parent_bind_authority": {
        "path": ACTIVE_GPU_STATE_PARENT_BIND_CORRECTION.as_posix(),
        "sha256": _GPU_STATE_PARENT_BIND_AUTHORITY_FILE_SHA256,
        "bytes": 14858,
        "mode": "0444",
        "content_sha256": _GPU_STATE_PARENT_BIND_AUTHORITY_CONTENT_SHA256,
    },
    "parent_gpu_state_parent_bind_diagnostic": {
        "path": ACTIVE_GPU_STATE_PARENT_BIND_DIAGNOSTIC.as_posix(),
        "sha256": _GPU_STATE_PARENT_BIND_DIAGNOSTIC_FILE_SHA256,
        "bytes": 10709,
        "mode": "0444",
        "content_sha256": _GPU_STATE_PARENT_BIND_DIAGNOSTIC_CONTENT_SHA256,
    },
    "failed_rootbind1_target_capability_receipt": {
        "path": _ROOTBIND1_PARENT_PATHS[
            "failed_rootbind1_target_capability_receipt"
        ].as_posix(),
        "sha256": "b6ceccf7b4d3f0738de1cbead9038fe937a209295294a4af772a902dccbd20d8",
        "bytes": 54959,
        "mode": "0444",
        "content_sha256": "0db3ec473188efda5b60f8b635d07e790d632ddca48a91547f61ac6906c41218",
    },
    "failed_rootbind1_target_completion_receipt": {
        "path": _ROOTBIND1_PARENT_PATHS[
            "failed_rootbind1_target_completion_receipt"
        ].as_posix(),
        "sha256": "e08456bcdecddb10e38e7378837f785314dd8494125d2363b1063df7b4723747",
        "bytes": 8676,
        "mode": "0444",
        "content_sha256": "c4d73ab37fe61b049f59ea00fa81e6dbdcd3585b565319e9326b299959dd12fe",
    },
    "failed_rootbind1_benchmark_invocation": {
        "path": _ROOTBIND1_PARENT_PATHS[
            "failed_rootbind1_benchmark_invocation"
        ].as_posix(),
        "sha256": "a52232d9dc4428550039149589d8f0e718b39d146f7d271c195710a26ef8a3f9",
        "bytes": 6789,
        "mode": "0444",
        "content_sha256": "58ed6edf641aa5d4a559802af966727913a9a7bca5f2cdf3c4f7377efc25f670",
    },
    "failed_rootbind1_gpu_invocation": {
        "path": _ROOTBIND1_PARENT_PATHS[
            "failed_rootbind1_gpu_invocation"
        ].as_posix(),
        "sha256": "e06bbc723706fd6756f3224dc806ac54cfab6fe8f7852da1f7c372f740730961",
        "bytes": 3189,
        "mode": "0444",
        "content_sha256": "b86e18808c7e3440200dee669fc356aa51876733bdf9d5ee808cc505086cdb9f",
    },
    "failed_rootbind1_gpu_terminal_result": {
        "path": _ROOTBIND1_PARENT_PATHS[
            "failed_rootbind1_gpu_terminal_result"
        ].as_posix(),
        "sha256": "b575caa298db286cad2a3ad3231aa84dcdb2af76ab04d15ea38eec7b1a50fbda",
        "bytes": 1832,
        "mode": "0444",
        "content_sha256": "e3ddad8b65692a05cdabc1f74bdd29fc4fdbe86991e0d956ee2ad1af000599e5",
    },
}
_ADMITTED_CONTEXT_AUTHORIZED_BEFORE = {
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
_EXECUTION_CLOSURE_LEGACY_ACTIVE_OUTPUT_ROOT = (
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
    "efficiency_benchmark_v8r4a"
)
_EXECUTION_CLOSURE_HISTORY = (
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


def _active_validate_execution_closure_projection(
    authority: Mapping[str, Any]
) -> None:
    basis = authority.get("authority_basis")
    expected_basis_keys = {
        "authorization_limited_to_terminal_execution_closure", "diagnostic",
        "historical_benchmark_prefix", "parent_kill_safe_addendum",
        "parent_open_lifecycle_recovery_addendum", "parent_source_closure_addendum",
        "user_goal_scope",
    }
    if not isinstance(basis, Mapping) or set(basis) != expected_basis_keys:
        _fail("V8R4A execution-closure authority basis schema drifted")
    history = basis.get("historical_benchmark_prefix")
    expected_entries = [
        {"path": path, "file_sha256": digest, "bytes": size, "mode": mode, "role": role}
        for path, digest, size, mode, role in _EXECUTION_CLOSURE_HISTORY
    ]
    if not (
        basis.get("authorization_limited_to_terminal_execution_closure") is True
        and isinstance(history, Mapping)
        and set(history) == {
            "active_output_root", "entries", "historical_root_mounted_or_mutated",
            "known_v8r3_mode_0644_is_read_only_quarantined_evidence",
        }
        and history.get("active_output_root")
        == _EXECUTION_CLOSURE_LEGACY_ACTIVE_OUTPUT_ROOT
        and history.get("entries") == expected_entries
        and history.get("historical_root_mounted_or_mutated") is False
        and history.get("known_v8r3_mode_0644_is_read_only_quarantined_evidence") is True
    ):
        _fail("V8R4A execution-closure historical projection drifted")


def _active_validate_execution_closure_history(
    root: Path, authority: Mapping[str, Any]
) -> None:
    _active_validate_execution_closure_projection(authority)
    for path, digest, size, mode, _role in _EXECUTION_CLOSURE_HISTORY:
        binding, _raw = _active_read_binding(
            root, path, require_frozen=(mode == "0444")
        )
        if not (
            binding["sha256"] == digest
            and binding["bytes"] == size
            and binding["mode"] == mode
            and binding["nlink"] == 1
        ):
            _fail(f"historical benchmark projection drifted: {path}")


def _active_validate_migration_source_succession_projection(
    root: Path,
    authority: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> None:
    basis = authority.get("authority_basis")
    if not (
        set(authority) == _MIGRATION_SOURCE_SUCCESSION_AUTHORITY_KEYS
        and set(diagnostic) == _MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC_KEYS
        and authority.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_correction_addendum"
        and authority.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        and authority.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and authority.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("schema_version") == 1
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_failure_diagnostic"
        and diagnostic.get("campaign_id") == authority.get("campaign_id")
        and diagnostic.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and isinstance(basis, Mapping)
        and set(basis)
        == {
            "diagnostic",
            "execution_closure_authority",
            "immutable_migration_receipt",
            "original_migration_authority",
            "user_goal_scope",
        }
        and isinstance(basis.get("user_goal_scope"), str)
        and bool(basis["user_goal_scope"])
    ):
        _fail("V8R4A migration source-succession projection drifted")

    def expected_binding(relative: Path) -> dict[str, Any]:
        document, binding = _active_load_document(root, relative)
        return {
            "path": relative.as_posix(),
            "file_sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "content_sha256": document["content_sha256"],
        }

    for field, relative in (
        ("diagnostic", ACTIVE_MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC),
        ("execution_closure_authority", ACTIVE_EXECUTION_CLOSURE_CORRECTION),
        ("immutable_migration_receipt", ACTIVE_MIGRATION_RECEIPT),
        ("original_migration_authority", ACTIVE_V8R4A_CORRECTION),
    ):
        if not exact_json_equal(basis.get(field), expected_binding(relative)):
            _fail(f"V8R4A migration source-succession binding drifted: {field}")


def _active_validate_fd_closure_projection(
    authority: Mapping[str, Any], diagnostic: Mapping[str, Any]
) -> None:
    """Validate FD1 semantics without opening historical parent capabilities."""

    basis = authority.get("authority_basis")
    failed = diagnostic.get("failed_attempt")
    reproduction = diagnostic.get("reproduction")
    required = authority.get("required_reauthorization")
    claim = authority.get("claim_boundary")
    rows = authority.get("authorized_modifications")
    if not (
        set(authority) == _FD_CLOSURE_AUTHORITY_KEYS
        and set(diagnostic) == _FD_CLOSURE_DIAGNOSTIC_KEYS
        and authority.get("schema_version") == 1
        and diagnostic.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_outer_guard_urandom_descriptor_closure_correction_addendum"
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_outer_guard_urandom_descriptor_failure_diagnostic"
        and authority.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        and diagnostic.get("campaign_id") == authority.get("campaign_id")
        and authority.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and diagnostic.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and authority.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and isinstance(basis, Mapping)
        and set(basis)
        == {"diagnostic", *_FD_CLOSURE_PARENT_BINDINGS, "user_goal_scope"}
        and exact_json_equal(
            basis.get("diagnostic"), _FD_CLOSURE_DIAGNOSTIC_LEGACY_BINDING
        )
        and all(
            exact_json_equal(basis.get(field), binding)
            for field, binding in _FD_CLOSURE_PARENT_BINDINGS.items()
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
        == _FD_CLOSURE_SUCCESSOR_CHAIN_NAMES["new_test_receipt"]
        and required.get("new_source_snapshot")
        == _FD_CLOSURE_SUCCESSOR_CHAIN_NAMES["new_source_snapshot"]
        and required.get("new_pretrain_authorization")
        == _FD_CLOSURE_SUCCESSOR_CHAIN_NAMES["new_pretrain_authorization"]
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
        _fail("V8R4A FD-closure target projection drifted")
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
            _fail("V8R4A FD-closure target modification row drifted")
        path = str(row["path"])
        if path in observed:
            _fail("V8R4A FD-closure target modification path is duplicated")
        observed[path] = str(row["before_sha256"])
    if observed != _FD_CLOSURE_AUTHORIZED_BEFORE:
        _fail("V8R4A FD-closure target modification cover drifted")
    expected_superseded = {
        "implementation_test_receipt": _FD_CLOSURE_PARENT_BINDINGS[
            "parent_implementation_test_receipt"
        ],
        "source_snapshot": _FD_CLOSURE_PARENT_BINDINGS[
            "parent_source_snapshot"
        ],
        "pretrain_authorization": _FD_CLOSURE_PARENT_BINDINGS[
            "parent_pretrain_authorization"
        ],
        "preserved_as_immutable_audit_evidence": True,
        "may_authorize_retry_without_successor_chain": False,
    }
    if not exact_json_equal(
        diagnostic.get("superseded_pretrain_chain"), expected_superseded
    ):
        _fail("V8R4A FD-closure target superseded chain drifted")


def _active_validate_fd_closure_correction(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the additive FD1 launcher correction and its exact parent chain."""

    authority, _authority_binding = _active_load_document(
        root, ACTIVE_FD_CLOSURE_CORRECTION
    )
    diagnostic, _diagnostic_binding = _active_load_document(
        root, ACTIVE_FD_CLOSURE_DIAGNOSTIC
    )
    _active_validate_fd_closure_projection(authority, diagnostic)
    basis = authority.get("authority_basis")
    failed = diagnostic.get("failed_attempt")
    reproduction = diagnostic.get("reproduction")
    diagnostic_required = diagnostic.get("required_correction")
    diagnostic_claim = diagnostic.get("claim_boundary")
    mandatory = authority.get("mandatory_invariants")
    forbidden = authority.get("forbidden_changes")
    required = authority.get("required_reauthorization")
    claim = authority.get("claim_boundary")
    rows = authority.get("authorized_modifications")
    if not (
        set(authority) == _FD_CLOSURE_AUTHORITY_KEYS
        and set(diagnostic) == _FD_CLOSURE_DIAGNOSTIC_KEYS
        and authority.get("schema_version") == 1
        and diagnostic.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_outer_guard_urandom_descriptor_closure_correction_addendum"
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_outer_guard_urandom_descriptor_failure_diagnostic"
        and authority.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and diagnostic.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and authority.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and isinstance(basis, Mapping)
        and set(basis)
        == {"diagnostic", *_FD_CLOSURE_PARENT_PATHS, "user_goal_scope"}
        and isinstance(basis.get("user_goal_scope"), str)
        and bool(basis["user_goal_scope"])
        and isinstance(failed, Mapping)
        and failed.get("first_phase") == "efficiency_benchmark"
        and failed.get("outer_fold") == 3
        and failed.get("target_runtime_return_code") == 73
        and failed.get("coordinator_return_code") == 79
        and failed.get("capability_receipt_created") is False
        and failed.get("completion_receipt_created") is False
        and failed.get("gpu_child_launched") is False
        and isinstance(reproduction, Mapping)
        and reproduction.get("descriptor_3_target") == "/dev/urandom"
        and reproduction.get("descriptor_3_fd_cloexec") is True
        and reproduction.get("existing_cleanup_then_exact_audit_passes") is True
        and reproduction.get("arbitrary_unexpected_descriptor_rejection_must_remain")
        is True
        and isinstance(diagnostic_required, Mapping)
        and diagnostic_required.get(
            "close_only_verified_cloexec_dev_urandom_runtime_noise_before_outer_production_audit"
        )
        is True
        and diagnostic_required.get("full_reauthorization_before_gpu_retry") is True
        and isinstance(diagnostic_claim, Mapping)
        and diagnostic_claim.get("outer_test_features_or_targets_opened") is False
        and diagnostic_claim.get("accuracy_metric_computed") is False
        and diagnostic_claim.get("gpu_accessed") is False
        and diagnostic_claim.get("commercial_claim_authorized") is False
        and isinstance(mandatory, Mapping)
        and mandatory.get("only_cloexec_dev_urandom_may_be_normalized") is True
        and mandatory.get("all_other_unexpected_descriptors_fail_closed") is True
        and mandatory.get("target_and_outer_reference_sealing_unchanged") is True
        and mandatory.get("gpu_budget_and_append_only_ledgers_unchanged") is True
        and isinstance(forbidden, Mapping)
        and forbidden.get("model_architecture_or_loss") is True
        and forbidden.get("data_rows_splits_or_pack_bytes") is True
        and forbidden.get("metric_or_selection_thresholds") is True
        and forbidden.get("outer_target_or_reference_access") is True
        and forbidden.get("ledger_reset_truncation_or_rewrite") is True
        and isinstance(required, Mapping)
        and required.get("new_test_receipt")
        == _FD_CLOSURE_SUCCESSOR_CHAIN_NAMES["new_test_receipt"]
        and required.get("new_source_snapshot")
        == _FD_CLOSURE_SUCCESSOR_CHAIN_NAMES["new_source_snapshot"]
        and required.get("new_pretrain_authorization")
        == _FD_CLOSURE_SUCCESSOR_CHAIN_NAMES["new_pretrain_authorization"]
        and required.get("all_fixed_tests_pass") is True
        and required.get("fresh_interpreter_subprocess_regression_passes") is True
        and required.get("gpu_retry_only_after_successor_pretrain_validation")
        is True
        and isinstance(claim, Mapping)
        and claim.get("adaptive_retrospective_only") is True
        and claim.get("correction_is_infrastructure_only") is True
        and claim.get("outer_test_features_or_targets_opened") is False
        and claim.get("accuracy_metric_used") is False
        and claim.get("gpu_execution_authorized_by_this_document") is False
        and claim.get("successor_pretrain_authorization_required") is True
        and claim.get("commercial_claim_authorized") is False
        and isinstance(rows, list)
    ):
        _fail("V8R4A FD-closure correction identity or boundary drifted")

    def legacy_binding(relative: Path) -> dict[str, Any]:
        document, binding = _active_load_document(root, relative)
        return {
            "path": relative.as_posix(),
            "file_sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "content_sha256": document["content_sha256"],
        }

    if not exact_json_equal(
        basis.get("diagnostic"), legacy_binding(ACTIVE_FD_CLOSURE_DIAGNOSTIC)
    ):
        _fail("V8R4A FD-closure diagnostic binding drifted")
    for field, relative in _FD_CLOSURE_PARENT_PATHS.items():
        if not exact_json_equal(basis.get(field), legacy_binding(relative)):
            _fail(f"V8R4A FD-closure parent binding drifted: {field}")

    observed_rows: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not (
            isinstance(row, Mapping)
            and set(row) == {"path", "before_sha256", "allowed_change"}
            and isinstance(row.get("path"), str)
            and isinstance(row.get("allowed_change"), str)
            and bool(row["allowed_change"])
        ):
            _fail("V8R4A FD-closure authorized modification row drifted")
        path = str(row["path"])
        if path in observed_rows:
            _fail("V8R4A FD-closure authorized path is duplicated")
        observed_rows[path] = row
    if set(observed_rows) != set(_FD_CLOSURE_AUTHORIZED_BEFORE):
        _fail("V8R4A FD-closure authorized path cover drifted")
    for path, digest in _FD_CLOSURE_AUTHORIZED_BEFORE.items():
        if observed_rows[path].get("before_sha256") != digest:
            _fail(f"V8R4A FD-closure before hash drifted: {path}")

    superseded = diagnostic.get("superseded_pretrain_chain")
    expected_superseded = {
        "implementation_test_receipt": legacy_binding(
            _FD_CLOSURE_PARENT_PATHS["parent_implementation_test_receipt"]
        ),
        "source_snapshot": legacy_binding(
            _FD_CLOSURE_PARENT_PATHS["parent_source_snapshot"]
        ),
        "pretrain_authorization": legacy_binding(
            _FD_CLOSURE_PARENT_PATHS["parent_pretrain_authorization"]
        ),
        "preserved_as_immutable_audit_evidence": True,
        "may_authorize_retry_without_successor_chain": False,
    }
    if not exact_json_equal(superseded, expected_superseded):
        _fail("V8R4A FD-closure superseded parent chain drifted")
    return authority, diagnostic


def _active_validate_canary_boundary_projection(
    authority: Mapping[str, Any], diagnostic: Mapping[str, Any]
) -> None:
    """Validate CANARY1 semantics without opening its historical FD1 parents."""

    basis = authority.get("authority_basis")
    rows = authority.get("authorized_modifications")
    mandatory = authority.get("mandatory_invariants")
    forbidden = authority.get("forbidden_changes")
    required = authority.get("required_reauthorization")
    claim = authority.get("claim_boundary")
    failed = diagnostic.get("failed_attempt")
    cause = diagnostic.get("root_cause")
    reproduction = diagnostic.get("reproduction")
    diagnostic_required = diagnostic.get("required_correction")
    diagnostic_claim = diagnostic.get("claim_boundary")
    authority_payload = {
        key: value for key, value in authority.items() if key != "content_sha256"
    }
    diagnostic_payload = {
        key: value for key, value in diagnostic.items() if key != "content_sha256"
    }
    if not (
        set(authority) == _CANARY_BOUNDARY_AUTHORITY_KEYS
        and set(diagnostic) == _CANARY_BOUNDARY_DIAGNOSTIC_KEYS
        and authority.get("schema_version") == 1
        and diagnostic.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_denied_canary_component_boundary_correction_addendum"
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_denied_canary_prefix_collision_failure_diagnostic"
        and authority.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        and diagnostic.get("campaign_id") == authority.get("campaign_id")
        and authority.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and diagnostic.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and authority.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and authority.get("content_sha256")
        == _CANARY_BOUNDARY_AUTHORITY_CONTENT_SHA256
        and semantic_sha256(authority_payload)
        == _CANARY_BOUNDARY_AUTHORITY_CONTENT_SHA256
        and diagnostic.get("content_sha256")
        == _CANARY_BOUNDARY_DIAGNOSTIC_CONTENT_SHA256
        and semantic_sha256(diagnostic_payload)
        == _CANARY_BOUNDARY_DIAGNOSTIC_CONTENT_SHA256
        and isinstance(basis, Mapping)
        and set(basis)
        == {"diagnostic", *_CANARY_BOUNDARY_PARENT_BINDINGS, "user_goal_scope"}
        and exact_json_equal(
            basis.get("diagnostic"), _CANARY_BOUNDARY_DIAGNOSTIC_BINDING
        )
        and all(
            exact_json_equal(basis.get(field), binding)
            for field, binding in _CANARY_BOUNDARY_PARENT_BINDINGS.items()
        )
        and isinstance(basis.get("user_goal_scope"), str)
        and bool(basis["user_goal_scope"])
        and isinstance(failed, Mapping)
        and failed.get("first_phase") == "efficiency_benchmark"
        and failed.get("outer_fold") == 3
        and failed.get("target_runtime_return_code") == 73
        and failed.get("coordinator_return_code") == 79
        and failed.get("capability_receipt_created") is False
        and failed.get("completion_receipt_created") is False
        and failed.get("gpu_child_launched") is False
        and failed.get("gpu_usage_ledger_mutated") is False
        and failed.get("gpu_execution_ledger_mutated") is False
        and isinstance(cause, Mapping)
        and cause.get("raw_substring_relation") is True
        and cause.get("path_component_ancestor_relation") is False
        and isinstance(reproduction, Mapping)
        and reproduction.get("python_substring_result") is True
        and reproduction.get("lexical_relative_to_denied_result") is False
        and reproduction.get("component_aware_mount_boundary_validation_passed")
        is True
        and reproduction.get("exact_denied_path_must_fail") is True
        and reproduction.get("denied_descendant_must_fail") is True
        and reproduction.get("path_distinct_prefix_siblings_must_pass") is True
        and reproduction.get("embedded_absolute_option_path_must_not_bypass_validation")
        is True
        and reproduction.get("path_traversal_must_fail_before_normalization")
        is True
        and isinstance(mandatory, Mapping)
        and mandatory.get("target_and_outer_reference_sealing_unchanged") is True
        and mandatory.get("gpu_budget_and_append_only_ledgers_unchanged") is True
        and mandatory.get("denied_canary_paths_unchanged") is True
        and mandatory.get("exact_denied_paths_and_descendants_fail_closed") is True
        and mandatory.get("path_distinct_sibling_prefixes_are_not_capabilities")
        is True
        and mandatory.get("embedded_absolute_option_paths_fail_closed") is True
        and mandatory.get("path_traversal_fails_before_normalization") is True
        and mandatory.get("component_aware_mount_validation_unchanged") is True
        and mandatory.get("sandbox_denied_canary_probe_unchanged") is True
        and mandatory.get("parent_fd1_chain_preserved_immutable") is True
        and isinstance(forbidden, Mapping)
        and forbidden.get("model_architecture_or_loss") is True
        and forbidden.get("hyperparameters_or_epoch_counts") is True
        and forbidden.get("data_rows_splits_or_pack_bytes") is True
        and forbidden.get("seed_variant_or_fold_matrix") is True
        and forbidden.get("outer_target_or_reference_access") is True
        and forbidden.get("renaming_or_weakening_denied_canaries_to_avoid_the_collision")
        is True
        and forbidden.get("allowing_exact_or_descendant_denied_paths") is True
        and forbidden.get("ledger_reset_truncation_or_rewrite") is True
        and forbidden.get("mutation_or_replacement_of_parent_evidence") is True
        and isinstance(required, Mapping)
        and required.get("new_test_receipt")
        == _CANARY_BOUNDARY_SUCCESSOR_CHAIN_NAMES["new_test_receipt"]
        and required.get("new_source_snapshot")
        == _CANARY_BOUNDARY_SUCCESSOR_CHAIN_NAMES["new_source_snapshot"]
        and required.get("new_pretrain_authorization")
        == _CANARY_BOUNDARY_SUCCESSOR_CHAIN_NAMES["new_pretrain_authorization"]
        and required.get("all_fixed_tests_pass") is True
        and required.get("component_boundary_regressions_pass") is True
        and required.get("diagnostic_and_authority_bound_in_every_target_capability")
        is True
        and required.get("gpu_retry_only_after_successor_pretrain_validation")
        is True
        and isinstance(claim, Mapping)
        and claim.get("correction_is_infrastructure_only") is True
        and claim.get("outer_test_features_or_targets_opened") is False
        and claim.get("accuracy_metric_used") is False
        and claim.get("gpu_execution_authorized_by_this_document") is False
        and claim.get("successor_pretrain_authorization_required") is True
        and claim.get("commercial_claim_authorized") is False
        and isinstance(diagnostic_required, Mapping)
        and diagnostic_required.get(
            "replace_raw_substring_with_lexical_component_boundary_check"
        ) is True
        and diagnostic_required.get("reject_exact_denied_paths_and_descendants")
        is True
        and diagnostic_required.get("allow_path_distinct_sibling_prefixes") is True
        and diagnostic_required.get("reject_traversal_and_embedded_absolute_option_paths")
        is True
        and diagnostic_required.get("bind_diagnostic_and_correction_in_runtime_governance")
        is True
        and diagnostic_required.get("new_test_receipt")
        == _CANARY_BOUNDARY_SUCCESSOR_CHAIN_NAMES["new_test_receipt"]
        and diagnostic_required.get("new_source_snapshot")
        == _CANARY_BOUNDARY_SUCCESSOR_CHAIN_NAMES["new_source_snapshot"]
        and diagnostic_required.get("new_pretrain_authorization")
        == _CANARY_BOUNDARY_SUCCESSOR_CHAIN_NAMES["new_pretrain_authorization"]
        and diagnostic_required.get("full_reauthorization_before_gpu_retry") is True
        and isinstance(diagnostic_claim, Mapping)
        and diagnostic_claim.get("outer_test_features_or_targets_opened") is False
        and diagnostic_claim.get("accuracy_metric_computed") is False
        and diagnostic_claim.get("gpu_accessed") is False
        and diagnostic_claim.get("scientific_configuration_change_authorized")
        is False
        and diagnostic_claim.get("commercial_claim_authorized") is False
        and isinstance(rows, list)
    ):
        _fail("V8R4A CANARY-boundary target projection drifted")

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
            _fail("V8R4A CANARY-boundary modification row drifted")
        path = str(row["path"])
        if path in observed:
            _fail("V8R4A CANARY-boundary modification path is duplicated")
        observed[path] = str(row["before_sha256"])
    if observed != _CANARY_BOUNDARY_AUTHORIZED_BEFORE:
        _fail("V8R4A CANARY-boundary modification cover drifted")

    expected_superseded = {
        "implementation_test_receipt": _CANARY_BOUNDARY_PARENT_BINDINGS[
            "parent_implementation_test_receipt"
        ],
        "source_snapshot": _CANARY_BOUNDARY_PARENT_BINDINGS[
            "parent_source_snapshot"
        ],
        "pretrain_authorization": _CANARY_BOUNDARY_PARENT_BINDINGS[
            "parent_pretrain_authorization"
        ],
        "preserved_as_immutable_audit_evidence": True,
        "may_authorize_retry_without_successor_chain": False,
    }
    if not exact_json_equal(
        diagnostic.get("superseded_pretrain_chain"), expected_superseded
    ):
        _fail("V8R4A CANARY-boundary superseded FD1 chain drifted")


def _active_validate_canary_boundary_correction(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the additive denied-canary component-boundary correction."""

    authority, authority_binding = _active_load_document(
        root, ACTIVE_CANARY_BOUNDARY_CORRECTION
    )
    diagnostic, diagnostic_binding = _active_load_document(
        root, ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC
    )
    _active_validate_canary_boundary_projection(authority, diagnostic)
    if not (
        authority_binding["sha256"] == _CANARY_BOUNDARY_AUTHORITY_FILE_SHA256
        and authority_binding["bytes"] == 8659
        and diagnostic_binding["sha256"]
        == _CANARY_BOUNDARY_DIAGNOSTIC_FILE_SHA256
        and diagnostic_binding["bytes"] == 5551
    ):
        _fail("V8R4A CANARY-boundary immutable file binding drifted")

    def legacy_binding(relative: Path) -> dict[str, Any]:
        document, binding = _active_load_document(root, relative)
        return {
            "path": relative.as_posix(),
            "file_sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "content_sha256": document["content_sha256"],
        }

    basis = authority.get("authority_basis")
    if not isinstance(basis, Mapping):
        _fail("V8R4A CANARY-boundary authority basis drifted")
    if not exact_json_equal(
        basis.get("diagnostic"), legacy_binding(ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC)
    ):
        _fail("V8R4A CANARY-boundary diagnostic binding drifted")
    for field, relative in _CANARY_BOUNDARY_PARENT_PATHS.items():
        if not exact_json_equal(basis.get(field), legacy_binding(relative)):
            _fail(f"V8R4A CANARY-boundary parent binding drifted: {field}")
    return authority, diagnostic


def _active_validate_frozen_contract_encoding_projection(
    authority: Mapping[str, Any], diagnostic: Mapping[str, Any]
) -> None:
    """Validate CONTRACT1 literals without opening superseded capabilities."""

    basis = authority.get("authority_basis")
    rows = authority.get("authorized_modifications")
    mandatory = authority.get("mandatory_invariants")
    forbidden = authority.get("forbidden_changes")
    required = authority.get("required_reauthorization")
    claim = authority.get("claim_boundary")
    failed = diagnostic.get("failed_attempt")
    contract_evidence = diagnostic.get("frozen_contract_evidence")
    receipts = diagnostic.get("immutable_failure_receipts")
    namespace = diagnostic.get("failed_namespace_inventory")
    diagnostic_required = diagnostic.get("required_correction")
    diagnostic_claim = diagnostic.get("claim_boundary")
    authority_payload = {
        key: value for key, value in authority.items() if key != "content_sha256"
    }
    diagnostic_payload = {
        key: value for key, value in diagnostic.items() if key != "content_sha256"
    }
    if not (
        set(authority) == _FROZEN_CONTRACT_ENCODING_AUTHORITY_KEYS
        and set(diagnostic) == _FROZEN_CONTRACT_ENCODING_DIAGNOSTIC_KEYS
        and authority.get("schema_version") == 1
        and diagnostic.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_frozen_contract_exact_byte_encoding_correction_addendum"
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_frozen_contract_encoding_false_rejection_failure_diagnostic"
        and authority.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        and diagnostic.get("campaign_id") == authority.get("campaign_id")
        and authority.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and diagnostic.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and authority.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and authority.get("content_sha256")
        == _FROZEN_CONTRACT_ENCODING_AUTHORITY_CONTENT_SHA256
        and semantic_sha256(authority_payload)
        == _FROZEN_CONTRACT_ENCODING_AUTHORITY_CONTENT_SHA256
        and diagnostic.get("content_sha256")
        == _FROZEN_CONTRACT_ENCODING_DIAGNOSTIC_CONTENT_SHA256
        and semantic_sha256(diagnostic_payload)
        == _FROZEN_CONTRACT_ENCODING_DIAGNOSTIC_CONTENT_SHA256
        and isinstance(basis, Mapping)
        and set(basis)
        == {
            "diagnostic",
            *_FROZEN_CONTRACT_ENCODING_PARENT_BINDINGS,
            "user_goal_scope",
        }
        and exact_json_equal(
            basis.get("diagnostic"),
            _FROZEN_CONTRACT_ENCODING_DIAGNOSTIC_BINDING,
        )
        and all(
            exact_json_equal(basis.get(field), binding)
            for field, binding in _FROZEN_CONTRACT_ENCODING_PARENT_BINDINGS.items()
        )
        and isinstance(basis.get("user_goal_scope"), str)
        and bool(basis["user_goal_scope"])
        and isinstance(failed, Mapping)
        and failed.get("first_phase") == "efficiency_benchmark"
        and failed.get("outer_fold") == 3
        and failed.get("target_runtime_child_return_code") == 1
        and failed.get("target_sandbox_child_launched") is True
        and failed.get("gpu_wrapper_reached") is False
        and failed.get("gpu_admission_reached") is False
        and failed.get("training_reached") is False
        and failed.get("accuracy_metric_computed") is False
        and failed.get("gpu_usage_ledger_mutated") is False
        and failed.get("gpu_execution_ledger_mutated") is False
        and isinstance(contract_evidence, Mapping)
        and contract_evidence.get("path") == CONTRACT.as_posix()
        and contract_evidence.get("file_sha256") == CONTRACT_FILE_SHA256
        and contract_evidence.get("bytes") == CONTRACT_FILE_BYTES
        and contract_evidence.get("content_sha256") == CONTRACT_CONTENT_SHA256
        and contract_evidence.get("mode") == "0444"
        and contract_evidence.get("valid_unique_key_finite_json") is True
        and contract_evidence.get("semantic_content_hash_valid") is True
        and contract_evidence.get("exact_file_binding_valid") is True
        and contract_evidence.get("roundtrip_differs_from_frozen_bytes") is True
        and contract_evidence.get("may_be_rewritten_or_reformatted") is False
        and isinstance(receipts, Mapping)
        and exact_json_equal(
            receipts.get("capability_receipt"),
            {
                **_FROZEN_CONTRACT_ENCODING_PARENT_BINDINGS[
                    "failed_capability_receipt"
                ],
                "mode": "0444",
            },
        )
        and isinstance(receipts.get("completion_receipt"), Mapping)
        and all(
            receipts["completion_receipt"].get(key) == value
            for key, value in {
                **_FROZEN_CONTRACT_ENCODING_PARENT_BINDINGS[
                    "failed_completion_receipt"
                ],
                "mode": "0444",
                "return_code": 1,
                "closed_replay_validated": True,
            }.items()
        )
        and receipts.get("same_exact_lifecycle_replays_recorded_return_code")
        is True
        and receipts.get("mutation_replacement_or_deletion_allowed") is False
        and isinstance(namespace, Mapping)
        and namespace.get("lifecycle_root")
        == _TARGET_SUPERSEDED_LIFECYCLE_ROOT.as_posix()
        and namespace.get("benchmark_output_root")
        == _TARGET_SUPERSEDED_BENCHMARK_OUTPUT_ROOT.as_posix()
        and namespace.get("completion_receipt_binds_live_output_inventory") is True
        and namespace.get("old_lifecycle_and_output_roots_must_be_preserved_and_denied_to_successor")
        is True
        and isinstance(mandatory, Mapping)
        and mandatory.get("exact_frozen_contract_path_sha_bytes_and_content_unchanged")
        is True
        and mandatory.get("frozen_contract_mode_nlink_and_identity_validation_unchanged")
        is True
        and mandatory.get("global_noncanonical_governance_rejection_unchanged_except_exact_contract")
        is True
        and mandatory.get("superseded_canary1_lifecycle_root_preserved_immutable")
        == _TARGET_SUPERSEDED_LIFECYCLE_ROOT.as_posix()
        and mandatory.get("superseded_canary1_output_root_preserved_immutable")
        == _TARGET_SUPERSEDED_BENCHMARK_OUTPUT_ROOT.as_posix()
        and mandatory.get("successor_contract1_lifecycle_root")
        == _TARGET_SUPERSEDED_CONTRACT1_LIFECYCLE_ROOT.as_posix()
        and mandatory.get("successor_contract1_output_root")
        == _TARGET_SUPERSEDED_CONTRACT1_BENCHMARK_OUTPUT_ROOT.as_posix()
        and mandatory.get("both_superseded_roots_denied_unmounted_and_command_inaccessible")
        is True
        and mandatory.get("failed_capability_and_completion_receipts_preserved_immutable")
        is True
        and mandatory.get("historical_execution_closure_authority_literal_unchanged")
        == _EXECUTION_CLOSURE_LEGACY_ACTIVE_OUTPUT_ROOT
        and mandatory.get("parent_canary1_chain_preserved_immutable") is True
        and isinstance(forbidden, Mapping)
        and forbidden.get("global_relaxation_of_json_encoding_validation") is True
        and forbidden.get("contract_reformat_rewrite_copy_or_replacement") is True
        and forbidden.get("acceptance_by_semantic_hash_without_exact_file_sha_and_bytes")
        is True
        and forbidden.get("reuse_of_superseded_lifecycle_root") is True
        and forbidden.get("reuse_or_mutation_of_superseded_output_root") is True
        and forbidden.get("deletion_or_replacement_of_failed_receipts") is True
        and forbidden.get("reinterpretation_of_frozen_execution_closure_authority")
        is True
        and forbidden.get("removing_or_weakening_any_denied_canary") is True
        and forbidden.get("outer_target_or_reference_access") is True
        and forbidden.get("ledger_reset_truncation_or_rewrite") is True
        and isinstance(required, Mapping)
        and required.get("new_test_receipt")
        == _FROZEN_CONTRACT_ENCODING_SUCCESSOR_CHAIN_NAMES["new_test_receipt"]
        and required.get("new_source_snapshot")
        == _FROZEN_CONTRACT_ENCODING_SUCCESSOR_CHAIN_NAMES["new_source_snapshot"]
        and required.get("new_pretrain_authorization")
        == _FROZEN_CONTRACT_ENCODING_SUCCESSOR_CHAIN_NAMES[
            "new_pretrain_authorization"
        ]
        and required.get("new_governance_roles")
        == [
            "frozen_contract_encoding_correction_authorization",
            "frozen_contract_encoding_failure_diagnostic",
        ]
        and required.get("new_denied_canary_roles")
        == [
            "superseded_v8r4a_lifecycle_root",
            "superseded_v8r4a_output_root",
        ]
        and required.get("all_fixed_tests_pass") is True
        and required.get("four_exact_contract_encoding_regressions_pass") is True
        and required.get("dual_root_succession_regressions_pass") is True
        and required.get("gpu_retry_only_after_successor_pretrain_validation")
        is True
        and isinstance(claim, Mapping)
        and claim.get("correction_is_infrastructure_only") is True
        and claim.get("outer_test_features_or_targets_opened") is False
        and claim.get("accuracy_metric_used") is False
        and claim.get("gpu_execution_authorized_by_this_document") is False
        and claim.get("successor_pretrain_authorization_required") is True
        and claim.get("commercial_claim_authorized") is False
        and isinstance(diagnostic_required, Mapping)
        and diagnostic_required.get("retain_global_pretty_roundtrip_validation_for_other_documents")
        is True
        and exact_json_equal(
            diagnostic_required.get("allow_only_exact_frozen_contract_exception"),
            {
                "relative_path": CONTRACT.as_posix(),
                "file_sha256": CONTRACT_FILE_SHA256,
                "bytes": CONTRACT_FILE_BYTES,
            },
        )
        and diagnostic_required.get("reject_same_bytes_at_any_other_relative_path")
        is True
        and diagnostic_required.get("reject_semantically_equal_whitespace_or_escape_mutations")
        is True
        and diagnostic_required.get("successor_lifecycle_root")
        == _TARGET_SUPERSEDED_CONTRACT1_LIFECYCLE_ROOT.as_posix()
        and diagnostic_required.get("successor_benchmark_output_root")
        == _TARGET_SUPERSEDED_CONTRACT1_BENCHMARK_OUTPUT_ROOT.as_posix()
        and diagnostic_required.get("preserve_frozen_execution_closure_authority_historical_output_literal")
        == _EXECUTION_CLOSURE_LEGACY_ACTIVE_OUTPUT_ROOT
        and diagnostic_required.get("new_test_receipt")
        == _FROZEN_CONTRACT_ENCODING_SUCCESSOR_CHAIN_NAMES["new_test_receipt"]
        and diagnostic_required.get("new_source_snapshot")
        == _FROZEN_CONTRACT_ENCODING_SUCCESSOR_CHAIN_NAMES["new_source_snapshot"]
        and diagnostic_required.get("new_pretrain_authorization")
        == _FROZEN_CONTRACT_ENCODING_SUCCESSOR_CHAIN_NAMES[
            "new_pretrain_authorization"
        ]
        and diagnostic_required.get("full_reauthorization_before_gpu_retry") is True
        and isinstance(diagnostic_claim, Mapping)
        and diagnostic_claim.get("outer_test_features_or_targets_opened") is False
        and diagnostic_claim.get("accuracy_metric_computed") is False
        and diagnostic_claim.get("gpu_accessed") is False
        and diagnostic_claim.get("scientific_configuration_change_authorized")
        is False
        and diagnostic_claim.get("commercial_claim_authorized") is False
        and isinstance(rows, list)
    ):
        _fail("V8R4A frozen-contract target projection drifted")

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
            _fail("V8R4A frozen-contract modification row drifted")
        path = str(row["path"])
        if path in observed:
            _fail("V8R4A frozen-contract modification path is duplicated")
        observed[path] = str(row["before_sha256"])
    if observed != _FROZEN_CONTRACT_ENCODING_AUTHORIZED_BEFORE:
        _fail("V8R4A frozen-contract modification cover drifted")


def _active_validate_frozen_contract_encoding_correction(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate CONTRACT1 authority and every exact host-side parent binding."""

    authority, authority_binding = _active_load_document(
        root, ACTIVE_FROZEN_CONTRACT_ENCODING_CORRECTION
    )
    diagnostic, diagnostic_binding = _active_load_document(
        root, ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC
    )
    _active_validate_frozen_contract_encoding_projection(authority, diagnostic)
    if not (
        authority_binding["sha256"]
        == _FROZEN_CONTRACT_ENCODING_AUTHORITY_FILE_SHA256
        and authority_binding["bytes"] == 13460
        and diagnostic_binding["sha256"]
        == _FROZEN_CONTRACT_ENCODING_DIAGNOSTIC_FILE_SHA256
        and diagnostic_binding["bytes"] == 8653
    ):
        _fail("V8R4A frozen-contract immutable file binding drifted")

    def exact_parent_binding(
        field: str, relative: Path
    ) -> dict[str, Any]:
        expected = _FROZEN_CONTRACT_ENCODING_PARENT_BINDINGS[field]
        if field in {"failed_capability_receipt", "failed_completion_receipt"}:
            binding, raw = _active_read_binding(root, relative)
            document = _active_decode_document_bytes(
                raw, label=f"frozen-contract {field}"
            )
            if raw != canonical_bytes(document) + b"\n":
                _fail(f"V8R4A frozen-contract receipt encoding drifted: {field}")
            verify_content_hash(document, path=root / relative)
        else:
            document, binding = _active_load_document(root, relative)
        observed = {
            "path": relative.as_posix(),
            "file_sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "content_sha256": document["content_sha256"],
        }
        if not exact_json_equal(observed, expected):
            _fail(f"V8R4A frozen-contract parent binding drifted: {field}")
        return observed

    basis = authority.get("authority_basis")
    if not isinstance(basis, Mapping):
        _fail("V8R4A frozen-contract authority basis drifted")
    diagnostic_parent, _ = _active_load_document(
        root, ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC
    )
    if diagnostic_parent.get("content_sha256") != (
        _FROZEN_CONTRACT_ENCODING_DIAGNOSTIC_CONTENT_SHA256
    ):
        _fail("V8R4A frozen-contract diagnostic content drifted")
    if not exact_json_equal(
        basis.get("diagnostic"), _FROZEN_CONTRACT_ENCODING_DIAGNOSTIC_BINDING
    ):
        _fail("V8R4A frozen-contract diagnostic binding drifted")
    for field, relative in _FROZEN_CONTRACT_ENCODING_PARENT_PATHS.items():
        if not exact_json_equal(
            basis.get(field), exact_parent_binding(field, relative)
        ):
            _fail(f"V8R4A frozen-contract authority parent drifted: {field}")
    return authority, diagnostic


def _active_validate_gpu_state_parent_bind_projection(
    authority: Mapping[str, Any], diagnostic: Mapping[str, Any]
) -> None:
    """Validate ROOTBIND1 literals without opening superseded capabilities."""

    basis = authority.get("authority_basis")
    rows = authority.get("authorized_modifications")
    mandatory = authority.get("mandatory_invariants")
    forbidden = authority.get("forbidden_changes")
    required = authority.get("required_reauthorization")
    claim = authority.get("claim_boundary")
    failed = diagnostic.get("failed_attempt")
    trusted_root = diagnostic.get("trusted_host_gpu_state_root")
    failed_mount = diagnostic.get("failed_mount_topology")
    receipts = diagnostic.get("immutable_failure_receipts")
    namespace = diagnostic.get("failed_namespace_inventory")
    diagnostic_required = diagnostic.get("required_correction")
    diagnostic_claim = diagnostic.get("claim_boundary")
    authority_payload = {
        key: value for key, value in authority.items() if key != "content_sha256"
    }
    diagnostic_payload = {
        key: value for key, value in diagnostic.items() if key != "content_sha256"
    }
    root_relative = ACTIVE_STATE_ROOT.as_posix()
    if not (
        set(authority) == _GPU_STATE_PARENT_BIND_AUTHORITY_KEYS
        and set(diagnostic) == _GPU_STATE_PARENT_BIND_DIAGNOSTIC_KEYS
        and authority.get("schema_version") == 1
        and diagnostic.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_gpu_state_parent_readonly_bind_correction_addendum"
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_gpu_state_parent_mount_identity_failure_diagnostic"
        and authority.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        and diagnostic.get("campaign_id") == authority.get("campaign_id")
        and authority.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and diagnostic.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and authority.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and authority.get("content_sha256")
        == _GPU_STATE_PARENT_BIND_AUTHORITY_CONTENT_SHA256
        and semantic_sha256(authority_payload)
        == _GPU_STATE_PARENT_BIND_AUTHORITY_CONTENT_SHA256
        and diagnostic.get("content_sha256")
        == _GPU_STATE_PARENT_BIND_DIAGNOSTIC_CONTENT_SHA256
        and semantic_sha256(diagnostic_payload)
        == _GPU_STATE_PARENT_BIND_DIAGNOSTIC_CONTENT_SHA256
        and isinstance(basis, Mapping)
        and set(basis)
        == {
            "diagnostic",
            *_GPU_STATE_PARENT_BIND_PARENT_BINDINGS,
            "user_goal_scope",
        }
        and exact_json_equal(
            basis.get("diagnostic"), _GPU_STATE_PARENT_BIND_DIAGNOSTIC_BINDING
        )
        and all(
            exact_json_equal(basis.get(field), binding)
            for field, binding in _GPU_STATE_PARENT_BIND_PARENT_BINDINGS.items()
        )
        and isinstance(basis.get("user_goal_scope"), str)
        and bool(basis["user_goal_scope"])
        and isinstance(failed, Mapping)
        and failed.get("first_phase") == "efficiency_benchmark"
        and failed.get("outer_fold") == 3
        and failed.get("target_runtime_child_return_code") == 1
        and failed.get("target_sandbox_child_launched") is True
        and failed.get("target_scoped_pretrain_validation_reached") is True
        and failed.get("gpu_wrapper_reached") is False
        and failed.get("gpu_admission_reached") is False
        and failed.get("training_reached") is False
        and failed.get("accuracy_metric_computed") is False
        and failed.get("gpu_usage_ledger_mutated") is False
        and failed.get("gpu_execution_ledger_mutated") is False
        and isinstance(trusted_root, Mapping)
        and trusted_root.get("path") == root_relative
        and trusted_root.get("mode") == "0700"
        and trusted_root.get("st_dev") == 66306
        and trusted_root.get("st_ino") == 6970105
        and trusted_root.get("exact_entries")
        == ["admission", "execution", "usage"]
        and trusted_root.get("migration_receipt_path")
        == ACTIVE_MIGRATION_RECEIPT.as_posix()
        and trusted_root.get("migration_receipt_file_sha256")
        == _GPU_STATE_PARENT_BIND_PARENT_BINDINGS[
            "gpu_state_migration_receipt"
        ]["file_sha256"]
        and trusted_root.get("migration_receipt_bytes") == 14926
        and trusted_root.get("migration_receipt_content_sha256")
        == _GPU_STATE_PARENT_BIND_PARENT_BINDINGS[
            "gpu_state_migration_receipt"
        ]["content_sha256"]
        and isinstance(failed_mount, Mapping)
        and failed_mount.get("exactly_three_mutable_child_mounts_present") is True
        and failed_mount.get("parent_identity_mount_absent") is True
        and isinstance(failed_mount.get("gpu_state_root_operation"), Mapping)
        and failed_mount["gpu_state_root_operation"].get("kind") == "directory"
        and failed_mount["gpu_state_root_operation"].get(
            "host_directory_fd_bound"
        ) is False
        and isinstance(receipts, Mapping)
        and isinstance(receipts.get("capability_receipt"), Mapping)
        and all(
            receipts["capability_receipt"].get(key) == value
            for key, value in {
                **_GPU_STATE_PARENT_BIND_PARENT_BINDINGS[
                    "failed_contract1_capability_receipt"
                ],
                "mode": "0444",
            }.items()
        )
        and isinstance(receipts.get("completion_receipt"), Mapping)
        and all(
            receipts["completion_receipt"].get(key) == value
            for key, value in {
                **_GPU_STATE_PARENT_BIND_PARENT_BINDINGS[
                    "failed_contract1_completion_receipt"
                ],
                "mode": "0444",
                "return_code": 1,
                "closed_replay_validated": True,
            }.items()
        )
        and receipts.get("same_exact_lifecycle_replays_recorded_return_code")
        is True
        and receipts.get("mutation_replacement_or_deletion_allowed") is False
        and isinstance(namespace, Mapping)
        and namespace.get("lifecycle_root")
        == _TARGET_SUPERSEDED_CONTRACT1_LIFECYCLE_ROOT.as_posix()
        and namespace.get("benchmark_output_root")
        == _TARGET_SUPERSEDED_CONTRACT1_BENCHMARK_OUTPUT_ROOT.as_posix()
        and namespace.get("completion_receipt_binds_live_output_inventory") is True
        and namespace.get(
            "old_lifecycle_and_output_roots_must_be_preserved_and_denied_to_successor"
        ) is True
        and isinstance(mandatory, Mapping)
        and mandatory.get("scientific_campaign_revision_unchanged")
        == ACTIVE_SCIENTIFIC_REVISION
        and mandatory.get("infrastructure_revision_unchanged")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and mandatory.get("migration_validator_and_migration_receipt_unchanged")
        is True
        and mandatory.get("gpu_state_root_path") == root_relative
        and mandatory.get("gpu_state_root_exact_mode") == "0700"
        and mandatory.get("gpu_state_root_exact_st_dev") == 66306
        and mandatory.get("gpu_state_root_exact_st_ino") == 6970105
        and mandatory.get("gpu_state_root_mount_kind") == "ro_bind_fd"
        and mandatory.get("gpu_state_root_mount_precedes_children") is True
        and mandatory.get("gpu_state_root_direct_mutation_denied") is True
        and mandatory.get("gpu_state_mutable_direct_children")
        == ["admission", "execution", "usage"]
        and mandatory.get("exactly_three_mutable_state_directory_mounts") is True
        and mandatory.get("all_other_readonly_writable_overlap_rejected") is True
        and mandatory.get("superseded_v8r4a_lifecycle_root_preserved_immutable")
        == _TARGET_SUPERSEDED_LIFECYCLE_ROOT.as_posix()
        and mandatory.get("superseded_v8r4a_output_root_preserved_immutable")
        == _TARGET_SUPERSEDED_BENCHMARK_OUTPUT_ROOT.as_posix()
        and mandatory.get(
            "superseded_v8r4a_contract1_lifecycle_root_preserved_immutable"
        ) == _TARGET_SUPERSEDED_CONTRACT1_LIFECYCLE_ROOT.as_posix()
        and mandatory.get(
            "superseded_v8r4a_contract1_output_root_preserved_immutable"
        ) == _TARGET_SUPERSEDED_CONTRACT1_BENCHMARK_OUTPUT_ROOT.as_posix()
        and mandatory.get("successor_rootbind1_lifecycle_root")
        == _ROOTBIND1_LIFECYCLE_ROOT.as_posix()
        and mandatory.get("successor_rootbind1_output_root")
        == _ROOTBIND1_BENCHMARK_OUTPUT_ROOT.as_posix()
        and mandatory.get(
            "all_four_superseded_roots_denied_unmounted_and_command_inaccessible"
        ) is True
        and mandatory.get(
            "historical_execution_closure_authority_literal_unchanged"
        ) == _EXECUTION_CLOSURE_LEGACY_ACTIVE_OUTPUT_ROOT
        and mandatory.get("parent_contract1_chain_preserved_immutable") is True
        and isinstance(forbidden, Mapping)
        and all(value is True for value in forbidden.values())
        and forbidden.get("migration_validator_relaxation") is True
        and forbidden.get("gpu_state_root_writable_bind") is True
        and forbidden.get("synthetic_parent_chmod_as_identity_substitute") is True
        and forbidden.get("adding_gpu_state_root_to_writable_roots") is True
        and forbidden.get("generic_readonly_writable_overlap_relaxation") is True
        and forbidden.get(
            "reuse_or_mutation_of_any_superseded_lifecycle_or_output_root"
        ) is True
        and forbidden.get("reinterpretation_of_frozen_execution_closure_authority")
        is True
        and isinstance(required, Mapping)
        and required.get("new_test_receipt")
        == _GPU_STATE_PARENT_BIND_SUCCESSOR_CHAIN_NAMES["new_test_receipt"]
        and required.get("new_source_snapshot")
        == _GPU_STATE_PARENT_BIND_SUCCESSOR_CHAIN_NAMES["new_source_snapshot"]
        and required.get("new_pretrain_authorization")
        == _GPU_STATE_PARENT_BIND_SUCCESSOR_CHAIN_NAMES[
            "new_pretrain_authorization"
        ]
        and required.get("new_governance_roles")
        == [
            "gpu_state_parent_bind_correction_authorization",
            "gpu_state_parent_bind_failure_diagnostic",
        ]
        and required.get("new_denied_canary_roles")
        == [
            "superseded_v8r4a_contract1_lifecycle_root",
            "superseded_v8r4a_contract1_output_root",
        ]
        and required.get("required_true_security_boundary")
        == "gpu_state_parent_identity_readonly_bind"
        and required.get("all_fixed_tests_pass") is True
        and required.get(
            "all_four_superseded_roots_bound_as_denied_canaries_in_every_target_capability"
        ) is True
        and required.get("gpu_retry_only_after_successor_pretrain_validation")
        is True
        and isinstance(claim, Mapping)
        and claim.get("adaptive_retrospective_only") is True
        and claim.get("correction_is_infrastructure_only") is True
        and claim.get("outer_test_features_or_targets_opened") is False
        and claim.get("accuracy_metric_used") is False
        and claim.get("gpu_execution_authorized_by_this_document") is False
        and claim.get("successor_pretrain_authorization_required") is True
        and claim.get("commercial_claim_authorized") is False
        and isinstance(diagnostic_required, Mapping)
        and diagnostic_required.get("migration_validator_relaxation_allowed") is False
        and diagnostic_required.get("gpu_state_root_writable_mount_allowed") is False
        and diagnostic_required.get("synthetic_parent_chmod_substitution_allowed")
        is False
        and exact_json_equal(
            diagnostic_required.get("exact_parent_readonly_fd_bind_required"),
            {
                "path": root_relative,
                "mode": "0700",
                "st_dev": 66306,
                "st_ino": 6970105,
                "kind": "ro_bind_fd",
            },
        )
        and diagnostic_required.get(
            "exactly_three_mutable_direct_child_overlays_required"
        ) == ["admission", "execution", "usage"]
        and diagnostic_required.get("parent_mount_must_precede_child_overlays")
        is True
        and diagnostic_required.get("deny_and_unmount_all_four_superseded_roots")
        is True
        and diagnostic_required.get("successor_lifecycle_root")
        == _ROOTBIND1_LIFECYCLE_ROOT.as_posix()
        and diagnostic_required.get("successor_benchmark_output_root")
        == _ROOTBIND1_BENCHMARK_OUTPUT_ROOT.as_posix()
        and diagnostic_required.get("new_test_receipt")
        == _GPU_STATE_PARENT_BIND_SUCCESSOR_CHAIN_NAMES["new_test_receipt"]
        and diagnostic_required.get("new_source_snapshot")
        == _GPU_STATE_PARENT_BIND_SUCCESSOR_CHAIN_NAMES["new_source_snapshot"]
        and diagnostic_required.get("new_pretrain_authorization")
        == _GPU_STATE_PARENT_BIND_SUCCESSOR_CHAIN_NAMES[
            "new_pretrain_authorization"
        ]
        and diagnostic_required.get("full_reauthorization_before_gpu_retry") is True
        and isinstance(diagnostic_claim, Mapping)
        and diagnostic_claim.get("adaptive_retrospective_only") is True
        and diagnostic_claim.get("outer_test_features_or_targets_opened") is False
        and diagnostic_claim.get("accuracy_metric_computed") is False
        and diagnostic_claim.get("gpu_accessed") is False
        and diagnostic_claim.get("scientific_configuration_change_authorized")
        is False
        and diagnostic_claim.get("commercial_claim_authorized") is False
        and isinstance(rows, list)
    ):
        _fail("V8R4A GPU-state parent-bind target projection drifted")

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
            _fail("V8R4A GPU-state parent-bind modification row drifted")
        relative = str(row["path"])
        if relative in observed:
            _fail("V8R4A GPU-state parent-bind modification path duplicated")
        observed[relative] = str(row["before_sha256"])
    if observed != _GPU_STATE_PARENT_BIND_AUTHORIZED_BEFORE:
        _fail("V8R4A GPU-state parent-bind modification cover drifted")


def _active_validate_gpu_state_parent_bind_correction(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate ROOTBIND1 authority and each exact host-side parent binding."""

    authority, authority_binding = _active_load_document(
        root, ACTIVE_GPU_STATE_PARENT_BIND_CORRECTION
    )
    diagnostic, diagnostic_binding = _active_load_document(
        root, ACTIVE_GPU_STATE_PARENT_BIND_DIAGNOSTIC
    )
    _active_validate_gpu_state_parent_bind_projection(authority, diagnostic)
    if not (
        authority_binding["sha256"]
        == _GPU_STATE_PARENT_BIND_AUTHORITY_FILE_SHA256
        and authority_binding["bytes"] == 14858
        and diagnostic_binding["sha256"]
        == _GPU_STATE_PARENT_BIND_DIAGNOSTIC_FILE_SHA256
        and diagnostic_binding["bytes"] == 10709
    ):
        _fail("V8R4A GPU-state parent-bind immutable file binding drifted")

    def exact_parent_binding(field: str, relative: Path) -> dict[str, Any]:
        if field in {
            "failed_contract1_capability_receipt",
            "failed_contract1_completion_receipt",
        }:
            binding, raw = _active_read_binding(root, relative)
            document = _active_decode_document_bytes(
                raw, label=f"GPU-state parent-bind {field}"
            )
            if raw != canonical_bytes(document) + b"\n":
                _fail(f"V8R4A GPU-state parent-bind receipt encoding drifted: {field}")
            verify_content_hash(document, path=root / relative)
        else:
            document, binding = _active_load_document(root, relative)
        observed = {
            "path": relative.as_posix(),
            "file_sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "content_sha256": document["content_sha256"],
        }
        expected = _GPU_STATE_PARENT_BIND_PARENT_BINDINGS[field]
        if not exact_json_equal(observed, expected):
            _fail(f"V8R4A GPU-state parent-bind parent drifted: {field}")
        return observed

    basis = authority.get("authority_basis")
    if not isinstance(basis, Mapping):
        _fail("V8R4A GPU-state parent-bind authority basis drifted")
    if not exact_json_equal(
        basis.get("diagnostic"), _GPU_STATE_PARENT_BIND_DIAGNOSTIC_BINDING
    ):
        _fail("V8R4A GPU-state parent-bind diagnostic binding drifted")
    for field, relative in _GPU_STATE_PARENT_BIND_PARENT_PATHS.items():
        if not exact_json_equal(
            basis.get(field), exact_parent_binding(field, relative)
        ):
            _fail(f"V8R4A GPU-state parent-bind authority parent drifted: {field}")
    return authority, diagnostic


def _active_validate_admitted_context_projection(
    authority: Mapping[str, Any], diagnostic: Mapping[str, Any]
) -> None:
    """Validate CONTEXT1 semantics without opening superseded capabilities."""

    basis = authority.get("authority_basis")
    mandatory = authority.get("mandatory_invariants")
    forbidden = authority.get("forbidden_changes")
    required = authority.get("required_reauthorization")
    claim = authority.get("claim_boundary")
    rows = authority.get("authorized_modifications")
    failed = diagnostic.get("failed_attempt")
    receipts = diagnostic.get("immutable_failure_receipts")
    ledgers = diagnostic.get("ledger_evidence")
    cause = diagnostic.get("root_cause")
    diagnostic_required = diagnostic.get("required_correction")
    diagnostic_claim = diagnostic.get("claim_boundary")
    authority_payload = {
        key: value for key, value in authority.items() if key != "content_sha256"
    }
    diagnostic_payload = {
        key: value for key, value in diagnostic.items() if key != "content_sha256"
    }
    expected_basis_fields = {
        "diagnostic",
        *_ROOTBIND1_PARENT_BINDINGS,
        "frozen_campaign_contract",
        "gpu_state_migration_receipt",
        "user_goal_scope",
    }
    if not (
        set(authority) == _ADMITTED_CONTEXT_AUTHORITY_KEYS
        and set(diagnostic) == _ADMITTED_CONTEXT_DIAGNOSTIC_KEYS
        and authority.get("schema_version") == 1
        and diagnostic.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_correction_addendum"
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_failure_diagnostic"
        and authority.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        and diagnostic.get("campaign_id") == authority.get("campaign_id")
        and authority.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and diagnostic.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and authority.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and authority.get("content_sha256")
        == _ADMITTED_CONTEXT_AUTHORITY_CONTENT_SHA256
        and semantic_sha256(authority_payload)
        == _ADMITTED_CONTEXT_AUTHORITY_CONTENT_SHA256
        and diagnostic.get("content_sha256")
        == _ADMITTED_CONTEXT_DIAGNOSTIC_CONTENT_SHA256
        and semantic_sha256(diagnostic_payload)
        == _ADMITTED_CONTEXT_DIAGNOSTIC_CONTENT_SHA256
        and isinstance(basis, Mapping)
        and set(basis) == expected_basis_fields
        and isinstance(basis.get("user_goal_scope"), str)
        and bool(basis["user_goal_scope"])
    ):
        _fail("V8R4A admitted-context target projection drifted")

    diagnostic_basis = basis.get("diagnostic")
    if not (
        isinstance(diagnostic_basis, Mapping)
        and set(diagnostic_basis)
        == {
            "path", "sha256", "bytes", "mode", "nlink", "st_dev", "st_ino",
            "content_sha256",
        }
        and diagnostic_basis.get("path")
        == ACTIVE_ADMITTED_CONTEXT_DIAGNOSTIC.as_posix()
        and diagnostic_basis.get("sha256")
        == _ADMITTED_CONTEXT_DIAGNOSTIC_FILE_SHA256
        and diagnostic_basis.get("bytes") == 9019
        and diagnostic_basis.get("mode") == "0444"
        and diagnostic_basis.get("nlink") == 1
        and diagnostic_basis.get("content_sha256")
        == _ADMITTED_CONTEXT_DIAGNOSTIC_CONTENT_SHA256
    ):
        _fail("V8R4A admitted-context diagnostic binding drifted")
    for field, expected in _ROOTBIND1_PARENT_BINDINGS.items():
        if not exact_json_equal(basis.get(field), expected):
            _fail(f"V8R4A admitted-context parent binding drifted: {field}")
    if not exact_json_equal(
        basis.get("frozen_campaign_contract"),
        {
            "path": CONTRACT.as_posix(),
            "sha256": CONTRACT_FILE_SHA256,
            "bytes": CONTRACT_FILE_BYTES,
            "mode": "0444",
        },
    ):
        _fail("V8R4A admitted-context contract binding drifted")
    migration_basis = basis.get("gpu_state_migration_receipt")
    if not (
        isinstance(migration_basis, Mapping)
        and migration_basis.get("path") == ACTIVE_MIGRATION_RECEIPT.as_posix()
        and migration_basis.get("sha256")
        == ACTIVE_HISTORICAL_FILES[ACTIVE_MIGRATION_RECEIPT.as_posix()]
        and migration_basis.get("bytes") == 14926
        and migration_basis.get("mode") == "0444"
        and migration_basis.get("content_sha256")
        == "e73b38b390d00533243b23670e3ddbe3ec41461d8dab5336c4ff36b10431328b"
    ):
        _fail("V8R4A admitted-context migration binding drifted")

    observed_rows: dict[str, str] = {}
    if not isinstance(rows, list):
        _fail("V8R4A admitted-context modification cover drifted")
    for row in rows:
        if not (
            isinstance(row, Mapping)
            and set(row) == {"path", "before_sha256", "allowed_change"}
            and isinstance(row.get("path"), str)
            and isinstance(row.get("before_sha256"), str)
            and isinstance(row.get("allowed_change"), str)
            and bool(row["allowed_change"])
        ):
            _fail("V8R4A admitted-context modification row drifted")
        path = str(row["path"])
        if path in observed_rows:
            _fail("V8R4A admitted-context modification path duplicated")
        observed_rows[path] = str(row["before_sha256"])
    if observed_rows != _ADMITTED_CONTEXT_AUTHORIZED_BEFORE:
        _fail("V8R4A admitted-context modification cover drifted")

    if not (
        isinstance(mandatory, Mapping)
        and mandatory.get("scientific_campaign_revision_unchanged")
        == ACTIVE_SCIENTIFIC_REVISION
        and mandatory.get("infrastructure_revision_unchanged")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and mandatory.get("variants_seeds_folds_hyperparameters_and_metrics_unchanged")
        is True
        and mandatory.get("model_architecture_loss_data_and_pack_bytes_unchanged")
        is True
        and mandatory.get("benchmark_profile_gate_epochs_and_unit_unchanged")
        is True
        and mandatory.get("target_and_outer_reference_sealing_unchanged") is True
        and mandatory.get("gpu_budget_append_only_charge_and_single_owner_unchanged")
        is True
        and mandatory.get("rootbind_parent_readonly_and_three_child_readwrite_topology_unchanged")
        is True
        and mandatory.get("trainer_fail_closed_context_free_admitted_validation_retained")
        is True
        and mandatory.get("benchmark_worker_validates_target_scoped_admitted_pretrain_before_primitive")
        is True
        and mandatory.get("expected_phase_context_and_outer_fold_are_independent_fixed_constants")
        is True
        and mandatory.get("validated_pretrain_object_is_passed_to_trainer_primitive")
        is True
        and mandatory.get("admitted_binding_consumed_exactly_once") is True
        and mandatory.get("new_environment_or_descriptor_context_channel_absent")
        is True
        and exact_json_equal(
            mandatory.get("active_benchmark_context"),
            _CONTEXT1_FULL_BENCHMARK_CONTEXT,
        )
        and exact_json_equal(
            mandatory.get("superseded_rootbind1_context"),
            _ROOTBIND1_BENCHMARK_CONTEXT,
        )
        and mandatory.get("superseded_rootbind1_terminal_record_sha256")
        == _CONTEXT1_POSTFAILURE_PREFIXES["usage_ledger"][
            "tail_record_sha256"
        ]
        and mandatory.get("superseded_terminal_exactly_one_charged_failed_infrastructure_prefix")
        is True
        and mandatory.get("superseded_terminal_not_counted_as_active_benchmark_completion")
        is True
        and mandatory.get("same_context_retry_forbidden") is True
        and mandatory.get("historical_then_quarantine_then_rootbind1_failure_then_context1_order_required")
        is True
        and mandatory.get("usage_postfailure_sha256")
        == _CONTEXT1_POSTFAILURE_PREFIXES["usage_ledger"]["sha256"]
        and mandatory.get("usage_postfailure_bytes") == 113257
        and mandatory.get("usage_postfailure_record_count") == 77
        and mandatory.get("usage_postfailure_open_reservation_count") == 0
        and mandatory.get("usage_postfailure_settled_ns") == 1411550918574
        and mandatory.get("execution_postfailure_sha256")
        == _CONTEXT1_POSTFAILURE_PREFIXES["execution_ledger"]["sha256"]
        and mandatory.get("execution_postfailure_bytes") == 29961
        and mandatory.get("execution_postfailure_record_count") == 10
        and mandatory.get("execution_postfailure_open_start_count") == 0
        and mandatory.get("failed_rootbind1_lifecycle_and_output_preserved_immutable")
        is True
        and mandatory.get("all_six_superseded_roots_denied_unmounted_and_command_inaccessible")
        is True
        and mandatory.get("successor_context1_lifecycle_root")
        == _TARGET_LIFECYCLE_ROOT.as_posix()
        and mandatory.get("successor_context1_output_root")
        == _TARGET_ACTIVE_BENCHMARK_OUTPUT_ROOT.as_posix()
        and mandatory.get("frozen_execution_closure_authority_and_historical_literal_unchanged")
        is True
        and mandatory.get("parent_rootbind1_chain_and_all_failure_receipts_preserved_immutable")
        is True
        and mandatory.get("outer_test_features_targets_and_accuracy_metrics_not_opened_by_correction")
        is True
        and isinstance(forbidden, Mapping)
        and forbidden
        and all(value is True for value in forbidden.values())
        and isinstance(required, Mapping)
        and required.get("new_test_receipt")
        == _ADMITTED_CONTEXT_SUCCESSOR_CHAIN_NAMES["new_test_receipt"]
        and required.get("new_source_snapshot")
        == _ADMITTED_CONTEXT_SUCCESSOR_CHAIN_NAMES["new_source_snapshot"]
        and required.get("new_pretrain_authorization")
        == _ADMITTED_CONTEXT_SUCCESSOR_CHAIN_NAMES[
            "new_pretrain_authorization"
        ]
        and required.get("new_governance_roles")
        == [
            "admitted_context_correction_authorization",
            "admitted_context_failure_diagnostic",
        ]
        and required.get("new_denied_canary_roles")
        == [
            "superseded_v8r4a_rootbind1_lifecycle_root",
            "superseded_v8r4a_rootbind1_output_root",
        ]
        and required.get("required_true_security_boundary")
        == "benchmark_admitted_context_generation_isolated"
        and required.get("all_fixed_tests_pass") is True
        and required.get("programmatic_benchmark_bridge_exact_argument_and_order_tests_pass")
        is True
        and required.get("missing_extra_projected_or_binding_substituted_context_tests_fail_closed")
        is True
        and required.get("rootbind1_failure_prefix_hash_context_path_return_code_and_order_tests_pass")
        is True
        and required.get("global_and_pack_free_ledger_reconciliation_accounts_failed_charge")
        is True
        and required.get("all_six_superseded_roots_bound_as_denied_canaries_in_every_target_capability")
        is True
        and required.get("diagnostic_and_authority_bound_in_every_target_capability")
        is True
        and required.get("active_usage_and_execution_ledgers_closed_and_unchanged_during_cpu_tests")
        is True
        and required.get("fresh_context1_roots_absent_before_first_launch") is True
        and required.get("successor_pretrain_validation_passes_before_gpu_retry")
        is True
        and required.get("gpu_retry_only_after_full_reauthorization") is True
        and isinstance(claim, Mapping)
        and claim.get("adaptive_retrospective_only") is True
        and claim.get("correction_is_infrastructure_only") is True
        and claim.get("outer_test_features_or_targets_opened") is False
        and claim.get("accuracy_metric_used") is False
        and claim.get("prior_gpu_admission_and_cuda_availability_probe_recorded")
        is True
        and claim.get("prior_model_or_training_kernel_executed") is False
        and claim.get("gpu_execution_authorized_by_this_document") is False
        and claim.get("successor_pretrain_authorization_required") is True
        and claim.get("commercial_claim_authorized") is False
    ):
        _fail("V8R4A admitted-context authority boundary drifted")

    if not (
        isinstance(failed, Mapping)
        and failed.get("coordinator_return_code") == 79
        and failed.get("first_phase") == "efficiency_benchmark"
        and failed.get("outer_fold") == 3
        and failed.get("target_runtime_return_code") == 87
        and failed.get("gpu_wrapper_return_code") == 1
        and failed.get("target_sandbox_child_launched") is True
        and failed.get("target_scoped_capability_validation_passed") is True
        and failed.get("gpu_state_parent_readonly_bind_passed") is True
        and failed.get("gpu_state_three_mutable_child_overlays_passed") is True
        and failed.get("gpu_admission_reached") is True
        and failed.get("admitted_child_binding_consumed_once") is True
        and failed.get("admitted_phase_and_context_verified") is True
        and failed.get("cuda_availability_probe_occurred") is True
        and failed.get("cache_or_proposer_opened") is False
        and failed.get("model_constructed") is False
        and failed.get("cuda_model_allocation_or_training_kernel_reached")
        is False
        and failed.get("epoch_reached") is False
        and failed.get("accuracy_metric_computed") is False
        and failed.get("benchmark_timing_telemetry_created") is False
        and failed.get("reusable_training_output_created") is False
        and isinstance(receipts, Mapping)
        and set(receipts)
        == {
            "target_capability_receipt", "target_completion_receipt",
            "benchmark_invocation", "gpu_invocation", "gpu_terminal_result",
        }
        and isinstance(ledgers, Mapping)
        and ledgers.get("append_only_prefix_preserved") is True
        and ledgers.get("both_ledgers_closed_after_failure") is True
        and ledgers.get("ledger_reset_or_rewrite_required") is False
        and exact_json_equal(
            ledgers.get("usage_postlaunch"),
            {
                "sha256": _CONTEXT1_POSTFAILURE_PREFIXES["usage_ledger"][
                    "sha256"
                ],
                "bytes": 113257,
                "record_count": 77,
                "open_reservation_count": 0,
                "settled_usage_ns": 1411550918574,
                "tail_record_sha256": _CONTEXT1_POSTFAILURE_PREFIXES[
                    "usage_ledger"
                ]["tail_record_sha256"],
            },
        )
        and exact_json_equal(
            ledgers.get("execution_postlaunch"),
            {
                "sha256": _CONTEXT1_POSTFAILURE_PREFIXES["execution_ledger"][
                    "sha256"
                ],
                "bytes": 29961,
                "record_count": 10,
                "open_start_count": 0,
                "last_terminal_record_sha256": _CONTEXT1_POSTFAILURE_PREFIXES[
                    "usage_ledger"
                ]["tail_record_sha256"],
            },
        )
        and isinstance(cause, Mapping)
        and cause.get("benchmark_internal_worker_verified_fixed_binding") is True
        and cause.get("benchmark_internal_worker_parsed_target_receipt_and_expected_context")
        is True
        and cause.get("benchmark_internal_worker_called_trainer_primitive_without_prevalidated_pretrain")
        is True
        and cause.get("trainer_fail_closed_default_correctly_rejected_context_free_admitted_validation")
        is True
        and isinstance(diagnostic_required, Mapping)
        and diagnostic_required.get("target_scoped_admitted_validation_before_trainer_primitive")
        is True
        and diagnostic_required.get("independent_expected_phase")
        == "efficiency_benchmark"
        and exact_json_equal(
            diagnostic_required.get("independent_expected_context"),
            _CONTEXT1_FULL_BENCHMARK_CONTEXT,
        )
        and diagnostic_required.get("independent_expected_outer_fold") == 3
        and diagnostic_required.get("validated_pretrain_passed_to_primitive")
        is True
        and diagnostic_required.get("trainer_fail_closed_default_unchanged")
        is True
        and diagnostic_required.get("new_environment_or_descriptor_channel_allowed")
        is False
        and exact_json_equal(
            diagnostic_required.get("superseded_rootbind1_context"),
            _ROOTBIND1_BENCHMARK_CONTEXT,
        )
        and diagnostic_required.get("superseded_rootbind1_terminal_record_sha256")
        == _CONTEXT1_POSTFAILURE_PREFIXES["usage_ledger"][
            "tail_record_sha256"
        ]
        and diagnostic_required.get("same_context_retry_allowed") is False
        and diagnostic_required.get("successor_lifecycle_root")
        == _TARGET_LIFECYCLE_ROOT.as_posix()
        and diagnostic_required.get("successor_benchmark_output_root")
        == _TARGET_ACTIVE_BENCHMARK_OUTPUT_ROOT.as_posix()
        and diagnostic_required.get("deny_and_unmount_all_six_superseded_roots")
        is True
        and diagnostic_required.get("new_test_receipt")
        == _ADMITTED_CONTEXT_SUCCESSOR_CHAIN_NAMES["new_test_receipt"]
        and diagnostic_required.get("new_source_snapshot")
        == _ADMITTED_CONTEXT_SUCCESSOR_CHAIN_NAMES["new_source_snapshot"]
        and diagnostic_required.get("new_pretrain_authorization")
        == _ADMITTED_CONTEXT_SUCCESSOR_CHAIN_NAMES[
            "new_pretrain_authorization"
        ]
        and diagnostic_required.get("full_reauthorization_before_gpu_retry")
        is True
        and isinstance(diagnostic_claim, Mapping)
        and diagnostic_claim.get("adaptive_retrospective_only") is True
        and diagnostic_claim.get("outer_test_features_or_targets_opened") is False
        and diagnostic_claim.get("accuracy_metric_computed") is False
        and diagnostic_claim.get("gpu_admission_reached") is True
        and diagnostic_claim.get("cuda_availability_probe_occurred") is True
        and diagnostic_claim.get("model_or_training_kernel_executed") is False
        and diagnostic_claim.get("scientific_configuration_change_authorized")
        is False
        and diagnostic_claim.get("commercial_claim_authorized") is False
    ):
        _fail("V8R4A admitted-context failure diagnostic drifted")

    receipt_roles = {
        "target_capability_receipt": "failed_rootbind1_target_capability_receipt",
        "target_completion_receipt": "failed_rootbind1_target_completion_receipt",
        "benchmark_invocation": "failed_rootbind1_benchmark_invocation",
        "gpu_invocation": "failed_rootbind1_gpu_invocation",
        "gpu_terminal_result": "failed_rootbind1_gpu_terminal_result",
    }
    for diagnostic_role, authority_role in receipt_roles.items():
        row = receipts.get(diagnostic_role)
        expected = _ROOTBIND1_PARENT_BINDINGS[authority_role]
        if not (
            isinstance(row, Mapping)
            and all(row.get(key) == value for key, value in expected.items())
        ):
            _fail(
                f"V8R4A admitted-context failed receipt drifted: {diagnostic_role}"
            )
    terminal = receipts["gpu_terminal_result"]
    completion = receipts["target_completion_receipt"]
    if not (
        terminal.get("return_code") == 1
        and terminal.get("charged_usage_ns") == 2847219074
        and terminal.get("reusable_success") is False
        and terminal.get("terminal_record_sha256")
        == _CONTEXT1_POSTFAILURE_PREFIXES["usage_ledger"][
            "tail_record_sha256"
        ]
        and completion.get("return_code") == 87
        and completion.get("closed_replay_validated") is True
        and completion.get("state_transition_sha256")
        == "e881a3c1092b854294c6402a45b1becaa65c73827c016cb7af9c0d21690aca0a"
        and completion.get("output_inventory_sha256")
        == "3c180f141dc34924889ab54af363c1d5fc1f25eef50c1ea7970b97defdc99cd5"
    ):
        _fail("V8R4A admitted-context failure terminal semantics drifted")


def _active_validate_postfailure_ledger_prefixes(
    root: Path, *, require_exact: bool
) -> None:
    """Bind the charged ROOTBIND1 failure as CONTEXT1's immutable prefix."""

    for role, relative in (
        ("usage_ledger", ACTIVE_USAGE_LEDGER),
        ("execution_ledger", ACTIVE_EXECUTION_LEDGER),
    ):
        expected = _CONTEXT1_POSTFAILURE_PREFIXES[role]
        binding, raw = _active_read_binding(root, relative, require_frozen=False)
        size = int(expected["bytes"])
        if not (
            binding["mode"] == "0644"
            and len(raw) >= size
            and hashlib.sha256(raw[:size]).hexdigest() == expected["sha256"]
            and (not require_exact or len(raw) == size)
        ):
            _fail(f"V8R4A CONTEXT1 {role} postfailure prefix drifted")
        prefix_lines = raw[:size].splitlines()
        if len(prefix_lines) != expected["record_count"]:
            _fail(f"V8R4A CONTEXT1 {role} postfailure record count drifted")
        try:
            terminal = json.loads(prefix_lines[-1])
        except (IndexError, UnicodeError, json.JSONDecodeError) as error:
            _fail(f"V8R4A CONTEXT1 {role} terminal prefix is invalid: {error}")
        if not (
            isinstance(terminal, Mapping)
            and terminal.get("event")
            == ("terminal" if role == "usage_ledger" else "end")
            and terminal.get("phase") == "efficiency_benchmark"
            and exact_json_equal(terminal.get("context"), _ROOTBIND1_BENCHMARK_CONTEXT)
        ):
            _fail(f"V8R4A CONTEXT1 {role} terminal identity drifted")
        terminal_identity = (
            terminal.get("record_sha256")
            if role == "usage_ledger"
            else terminal.get("terminal_record_sha256")
        )
        line_hash = hashlib.sha256(prefix_lines[-1]).hexdigest()
        if not (
            terminal_identity
            == _CONTEXT1_POSTFAILURE_PREFIXES["usage_ledger"][
                "tail_record_sha256"
            ]
            and (
                role != "execution_ledger"
                or line_hash
                == _CONTEXT1_POSTFAILURE_PREFIXES["execution_ledger"][
                    "last_line_sha256"
                ]
            )
        ):
            _fail(f"V8R4A CONTEXT1 {role} terminal hash drifted")


def _active_validate_rootbind1_parent_chain(
    root: Path, documents: Mapping[str, Mapping[str, Any]]
) -> None:
    test = documents["parent_implementation_test_receipt"]
    snapshot = documents["parent_source_snapshot"]
    pretrain = documents["parent_pretrain_authorization"]
    old_roles = _ACTIVE_ADDENDUM_ROLES - {
        "admitted_context_correction_authorization",
        "admitted_context_failure_diagnostic",
    }
    if not (
        set(test) == (_ACTIVE_TEST_RECEIPT_KEYS - {
            "authorization_generation",
            "admitted_context_correction_authorization",
            "admitted_context_failure_diagnostic",
        })
        and set(snapshot) == (_ACTIVE_SNAPSHOT_KEYS - {
            "authorization_generation",
            "admitted_context_correction_authorization",
            "admitted_context_failure_diagnostic",
        })
        and set(pretrain) == (_ACTIVE_PRETRAIN_KEYS - {
            "authorization_generation",
            "admitted_context_correction_authorization",
            "admitted_context_failure_diagnostic",
        })
        and all(
            document.get("scientific_campaign_revision")
            == ACTIVE_SCIENTIFIC_REVISION
            and document.get("infrastructure_revision")
            == ACTIVE_INFRASTRUCTURE_REVISION
            for document in (test, snapshot, pretrain)
        )
        and all(role in test and role in snapshot and role in pretrain for role in old_roles)
        and test.get("all_tests_passed") is True
        and test.get("return_code") == 0
        and test.get("gpu_accessed") is False
        and test.get("target_or_outer_reference_accessed") is False
        and test.get("commercial_claim_authorized") is False
        and isinstance(test.get("implementation_files"), list)
        and len(test["implementation_files"]) == len(ACTIVE_IMPLEMENTATION_PATHS)
        and {row.get("path") for row in test["implementation_files"]}
        == set(ACTIVE_IMPLEMENTATION_PATHS)
        and exact_json_equal(test.get("test_paths"), list(ACTIVE_FIXED_TEST_PATHS))
        and exact_json_equal(
            snapshot.get("implementation_files"), test["implementation_files"]
        )
        and snapshot.get("training_authorized_by_snapshot_alone") is False
        and snapshot.get("adaptive_retrospective_only") is True
        and snapshot.get("commercial_claim_authorized") is False
        and pretrain.get("status") == "authorized"
        and pretrain.get("training_authorized") is True
        and pretrain.get("efficiency_benchmark_authorized") is True
        and pretrain.get("production_target_sealed_runtime_authorized") is True
        and pretrain.get("promotion_authorized") is False
        and pretrain.get("commercial_claim_authorized") is False
        and pretrain.get("outer_fold_numeric_reference_authorized") is False
        and exact_json_equal(
            pretrain.get("runtime_ledger_prefixes"),
            {
                "usage_ledger": {
                    "path": ACTIVE_USAGE_LEDGER.as_posix(),
                    "bytes": 109121,
                    "sha256": "9ce990030f51b40c5ccffc5146d20a0c754bf763e37e6cc6f76b8854edfaacba",
                    "record_count": 75,
                    "tail_record_sha256": "8dbc0493125f22c130444e1344533d1f3d9c4ac445df6adfe8b597972e9691c5",
                    "settled_usage_ns": 1408703699500,
                    "open_reservation_count": 0,
                },
                "execution_ledger": {
                    "path": ACTIVE_EXECUTION_LEDGER.as_posix(),
                    "bytes": 22822,
                    "sha256": "079dcca8066a976e6a8746ac33479360b7fd39bab82efa7eb7991a3c95514cf4",
                    "record_count": 8,
                    "last_line_sha256": "9ef15bd977a4accb2c013f6553d4d218fe942f38c1af598f777b7d6ae347f04f",
                    "open_start_count": 0,
                },
            },
        )
        and exact_json_equal(
            pretrain.get("efficiency_benchmark_scope"),
            {
                "phase": "efficiency_benchmark",
                "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
                "outer_fold": 3,
                "seed": 20260828,
                "variant": "H0_no_factor",
                "epochs": 2,
                "epoch_2_train_plus_target_free_validation_seconds_max": 23.0,
                "accuracy_metrics_authorized": False,
                "required_before_discovery": True,
            },
        )
    ):
        _fail("V8R4A CONTEXT1 parent ROOTBIND1 chain drifted")
    parent_test_binding = _active_binding(
        root, _ROOTBIND1_PARENT_PATHS["parent_implementation_test_receipt"]
    )
    parent_snapshot_binding = _active_binding(
        root, _ROOTBIND1_PARENT_PATHS["parent_source_snapshot"]
    )
    if not (
        snapshot.get("implementation_test_receipt")
        == {
            "path": parent_test_binding["path"],
            "file_sha256": parent_test_binding["sha256"],
            "size_bytes": parent_test_binding["bytes"],
            "mode": 0o444,
        }
        and exact_json_equal(
            pretrain.get("source_snapshot"), parent_snapshot_binding
        )
        and exact_json_equal(
            pretrain.get("implementation_test_receipt"), parent_test_binding
        )
        and pretrain.get("snapshot_content_sha256")
        == snapshot.get("content_sha256")
    ):
        _fail("V8R4A CONTEXT1 parent ROOTBIND1 issuance order drifted")


def _active_validate_rootbind1_failure_documents(
    root: Path, documents: Mapping[str, Mapping[str, Any]]
) -> None:
    capability = documents["failed_rootbind1_target_capability_receipt"]
    completion = documents["failed_rootbind1_target_completion_receipt"]
    benchmark = documents["failed_rootbind1_benchmark_invocation"]
    invocation = documents["failed_rootbind1_gpu_invocation"]
    terminal = documents["failed_rootbind1_gpu_terminal_result"]
    capability_path = root / _ROOTBIND1_PARENT_PATHS[
        "failed_rootbind1_target_capability_receipt"
    ]
    if not (
        capability.get("classification") == _TARGET_CAPABILITY_CLASSIFICATION
        and capability.get("campaign_revision") == ACTIVE_SCIENTIFIC_REVISION
        and capability.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and capability.get("phase") == "efficiency_benchmark"
        and capability.get("outer_fold") == 3
        and capability.get("security_boundary", {}).get(
            "gpu_state_parent_identity_readonly_bind"
        ) is True
        and capability.get("security_boundary", {}).get(
            "exactly_three_mutable_state_directory_mounts"
        ) is True
        and capability.get("security_boundary", {}).get(
            "production_execution_authorized"
        ) is True
        and capability.get("security_boundary", {}).get(
            "target_reference_or_selection_evidence_accessed"
        ) is False
        and capability.get("security_boundary", {}).get(
            "commercial_claim_authorized"
        ) is False
        and str(capability_path)
        == completion.get("capability_receipt", {}).get("path")
        and completion.get("capability_receipt", {}).get("sha256")
        == _ROOTBIND1_PARENT_BINDINGS[
            "failed_rootbind1_target_capability_receipt"
        ]["sha256"]
        and completion.get("classification")
        == "adaptive_v3r1_v8r4a_target_sealed_runtime_completion_receipt"
        and completion.get("phase") == "efficiency_benchmark"
        and completion.get("outer_fold") == 3
        and completion.get("return_code") == 87
        and completion.get("closed_replay_validated") is True
        and completion.get("state_transition_sha256")
        == "e881a3c1092b854294c6402a45b1becaa65c73827c016cb7af9c0d21690aca0a"
        and completion.get("output_inventory_sha256")
        == "3c180f141dc34924889ab54af363c1d5fc1f25eef50c1ea7970b97defdc99cd5"
        and benchmark.get("classification")
        == "adaptive_v3r1_v8r4_target_sealed_efficiency_benchmark_invocation"
        and benchmark.get("phase") == "efficiency_benchmark"
        and benchmark.get("benchmark_id")
        == _ROOTBIND1_BENCHMARK_CONTEXT["benchmark_id"]
        and benchmark.get("unit") == "outer_3_seed_20260828_H0_no_factor"
        and exact_json_equal(
            benchmark.get("usage_identity"), _ROOTBIND1_BENCHMARK_CONTEXT
        )
        and benchmark.get("accuracy_metrics_authorized") is False
        and benchmark.get("checkpoint_selection_authorized") is False
        and benchmark.get("outer_test_opened") is False
        and benchmark.get("training_result_reusable") is False
        and invocation.get("classification")
        == "adaptive_v3r1_gpu_execution_invocation"
        and invocation.get("phase") == "efficiency_benchmark"
        and exact_json_equal(
            invocation.get("context"), _ROOTBIND1_BENCHMARK_CONTEXT
        )
        and terminal.get("classification") == "gpu_budget_terminal_result"
        and terminal.get("phase") == "efficiency_benchmark"
        and exact_json_equal(terminal.get("context"), _ROOTBIND1_BENCHMARK_CONTEXT)
        and terminal.get("invocation_sha256")
        == _ROOTBIND1_PARENT_BINDINGS["failed_rootbind1_gpu_invocation"][
            "sha256"
        ]
        and terminal.get("return_code") == 1
        and terminal.get("wrapper_exit_code") == 1
        and terminal.get("charged_usage_ns") == 2847219074
        and terminal.get("elapsed_ns") == 2847219074
        and terminal.get("reusable_success") is False
        and terminal.get("containment_anomaly") is False
        and terminal.get("terminal_record_sha256")
        == _CONTEXT1_POSTFAILURE_PREFIXES["usage_ledger"][
            "tail_record_sha256"
        ]
    ):
        _fail("V8R4A CONTEXT1 failed ROOTBIND1 evidence drifted")


def _active_validate_admitted_context_correction(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Host-validate CONTEXT1 and every immutable ROOTBIND1 predecessor."""

    authority, authority_binding = _active_load_document(
        root, ACTIVE_ADMITTED_CONTEXT_CORRECTION
    )
    diagnostic, diagnostic_binding = _active_load_document(
        root, ACTIVE_ADMITTED_CONTEXT_DIAGNOSTIC
    )
    _active_validate_admitted_context_projection(authority, diagnostic)
    if not (
        authority_binding["sha256"] == _ADMITTED_CONTEXT_AUTHORITY_FILE_SHA256
        and authority_binding["bytes"] == 16684
        and diagnostic_binding["sha256"]
        == _ADMITTED_CONTEXT_DIAGNOSTIC_FILE_SHA256
        and diagnostic_binding["bytes"] == 9019
    ):
        _fail("V8R4A admitted-context immutable file binding drifted")
    _active_validate_gpu_state_parent_bind_correction(root)

    documents: dict[str, dict[str, Any]] = {}
    for field, relative in _ROOTBIND1_PARENT_PATHS.items():
        if field.startswith("failed_rootbind1_"):
            binding, raw = _active_read_binding(root, relative)
            document = _active_decode_document_bytes(
                raw, label=f"CONTEXT1 {field}"
            )
            pretty = (
                json.dumps(
                    document,
                    indent=2,
                    sort_keys=False,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            if raw not in {canonical_bytes(document) + b"\n", pretty}:
                _fail(f"V8R4A admitted-context parent encoding drifted: {field}")
            verify_content_hash(document, path=root / relative)
        else:
            document, binding = _active_load_document(root, relative)
        expected = _ROOTBIND1_PARENT_BINDINGS[field]
        observed = {
            "path": binding["path"],
            "sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "mode": binding["mode"],
            "content_sha256": document.get("content_sha256"),
        }
        if not exact_json_equal(observed, expected):
            _fail(f"V8R4A admitted-context immutable parent drifted: {field}")
        if not exact_json_equal(authority["authority_basis"].get(field), expected):
            _fail(f"V8R4A admitted-context authority parent drifted: {field}")
        documents[field] = document
    _active_validate_rootbind1_parent_chain(root, documents)
    _active_validate_rootbind1_failure_documents(root, documents)
    _active_validate_postfailure_ledger_prefixes(root, require_exact=False)
    state, _state_document = _active_state(root, require_closed=True)
    usage_state = state.usage_state
    execution_state = state.execution_state
    if not (
        usage_state.get("record_count") >= 77
        and usage_state.get("open_reservation_count") == 0
        and usage_state.get("settled_usage_ns") >= 1411550918574
        and execution_state.get("record_count") >= 10
        and execution_state.get("open_start_count") == 0
    ):
        _fail("V8R4A admitted-context live ledger succession drifted")
    return authority, diagnostic


def _active_authorized_modifications(
    root: Path,
) -> tuple[tuple[dict[str, Any], str], ...]:
    v8r4, _ = _active_load_document(root, ACTIVE_V8R4_CORRECTION)
    v8r4a, _ = _active_load_document(root, ACTIVE_V8R4A_CORRECTION)
    source_closure, _ = _active_load_document(
        root, ACTIVE_SOURCE_CLOSURE_CORRECTION
    )
    source_dependencies, _ = _active_load_document(
        root, ACTIVE_SOURCE_CLOSURE_DEPENDENCIES
    )
    kill_safe, _ = _active_load_document(root, ACTIVE_KILL_SAFE_CORRECTION)
    recovery, _ = _active_load_document(
        root, ACTIVE_OPEN_LIFECYCLE_RECOVERY_CORRECTION
    )
    execution_closure, _ = _active_load_document(
        root, ACTIVE_EXECUTION_CLOSURE_CORRECTION
    )
    migration_source_succession, _ = _active_load_document(
        root, ACTIVE_MIGRATION_SOURCE_SUCCESSION_CORRECTION
    )
    fd_closure, _fd_closure_diagnostic = _active_validate_fd_closure_correction(
        root
    )
    canary_boundary, _canary_boundary_diagnostic = (
        _active_validate_canary_boundary_correction(root)
    )
    frozen_contract_encoding, _frozen_contract_encoding_diagnostic = (
        _active_validate_frozen_contract_encoding_correction(root)
    )
    gpu_state_parent_bind, _gpu_state_parent_bind_diagnostic = (
        _active_validate_gpu_state_parent_bind_correction(root)
    )
    admitted_context, _admitted_context_diagnostic = (
        _active_validate_admitted_context_correction(root)
    )
    source_diagnostic, _ = _active_load_document(
        root, ACTIVE_SOURCE_CLOSURE_DIAGNOSTIC
    )
    kill_diagnostic, _ = _active_load_document(root, ACTIVE_KILL_SAFE_DIAGNOSTIC)
    recovery_diagnostic, _ = _active_load_document(
        root, ACTIVE_OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC
    )
    execution_closure_diagnostic, _ = _active_load_document(
        root, ACTIVE_EXECUTION_CLOSURE_DIAGNOSTIC
    )
    migration_source_succession_diagnostic, _ = _active_load_document(
        root, ACTIVE_MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC
    )
    recovery_basis = recovery.get("authority_basis")
    execution_closure_basis = execution_closure.get("authority_basis")
    migration_source_succession_basis = migration_source_succession.get(
        "authority_basis"
    )
    _active_validate_migration_source_succession_projection(
        root,
        migration_source_succession,
        migration_source_succession_diagnostic,
    )
    def authority_basis_binding(relative: Path) -> dict[str, Any]:
        document, binding = _active_load_document(root, relative)
        return {
            "path": relative.as_posix(),
            "file_sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "content_sha256": document["content_sha256"],
        }
    if not (
        v8r4.get("classification")
        == "posttrain_preselection_adaptive_v3r1_v8r4_physical_target_capability_and_pickle_free_output_correction_authorization"
        and v8r4a.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_dedicated_gpu_state_directory_migration_correction_authorization"
        and v8r4a.get("scientific_campaign_revision") == ACTIVE_SCIENTIFIC_REVISION
        and v8r4a.get("infrastructure_revision") == ACTIVE_INFRASTRUCTURE_REVISION
        and source_closure.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_validator_and_executable_source_closure_correction_addendum"
        and source_dependencies.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_executable_source_transitive_dependency_closure_addendum"
        and kill_safe.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_kill_safe_atomic_output_and_append_only_completion_correction_authorization"
        and recovery.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_open_gpu_lifecycle_kill_recovery_correction_addendum"
        and source_diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_validator_deadlock_and_executable_source_closure_failure_diagnostic"
        and kill_diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_kill_safe_output_and_shared_ledger_resume_failure_diagnostic"
        and recovery_diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_open_gpu_lifecycle_kill_recovery_failure_diagnostic"
        and execution_closure.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_kill_safe_capability_and_promotion_execution_closure_correction_addendum"
        and execution_closure_diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_terminal_execution_closure_failure_diagnostic"
        and migration_source_succession.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_correction_addendum"
        and migration_source_succession_diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_failure_diagnostic"
        and set(recovery)
        == {
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
        and set(recovery_diagnostic)
        == {
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
        and set(execution_closure) == _EXECUTION_CLOSURE_AUTHORITY_KEYS
        and set(execution_closure_diagnostic) == _EXECUTION_CLOSURE_DIAGNOSTIC_KEYS
        and set(migration_source_succession)
        == _MIGRATION_SOURCE_SUCCESSION_AUTHORITY_KEYS
        and set(migration_source_succession_diagnostic)
        == _MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC_KEYS
        and isinstance(recovery_basis, Mapping)
        and set(recovery_basis)
        == {
            "authorization_limited_to_open_lifecycle_kill_recovery",
            "diagnostic",
            "parent_kill_safe_addendum",
            "parent_source_closure_addendum",
            "user_goal_scope",
        }
        and recovery_basis.get("authorization_limited_to_open_lifecycle_kill_recovery")
        is True
        and exact_json_equal(
            recovery_basis.get("diagnostic"),
            authority_basis_binding(ACTIVE_OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC),
        )
        and exact_json_equal(
            recovery_basis.get("parent_kill_safe_addendum"),
            authority_basis_binding(ACTIVE_KILL_SAFE_CORRECTION),
        )
        and exact_json_equal(
            recovery_basis.get("parent_source_closure_addendum"),
            authority_basis_binding(ACTIVE_SOURCE_CLOSURE_CORRECTION),
        )
        and isinstance(execution_closure_basis, Mapping)
        and exact_json_equal(
            execution_closure_basis.get("diagnostic"),
            authority_basis_binding(ACTIVE_EXECUTION_CLOSURE_DIAGNOSTIC),
        )
        and exact_json_equal(
            execution_closure_basis.get("parent_kill_safe_addendum"),
            authority_basis_binding(ACTIVE_KILL_SAFE_CORRECTION),
        )
        and exact_json_equal(
            execution_closure_basis.get("parent_open_lifecycle_recovery_addendum"),
            authority_basis_binding(ACTIVE_OPEN_LIFECYCLE_RECOVERY_CORRECTION),
        )
        and exact_json_equal(
            execution_closure_basis.get("parent_source_closure_addendum"),
            authority_basis_binding(ACTIVE_SOURCE_CLOSURE_CORRECTION),
        )
        and isinstance(migration_source_succession_basis, Mapping)
        and set(migration_source_succession_basis)
        == {
            "diagnostic",
            "execution_closure_authority",
            "immutable_migration_receipt",
            "original_migration_authority",
            "user_goal_scope",
        }
        and exact_json_equal(
            migration_source_succession_basis.get("diagnostic"),
            authority_basis_binding(ACTIVE_MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC),
        )
        and exact_json_equal(
            migration_source_succession_basis.get("execution_closure_authority"),
            authority_basis_binding(ACTIVE_EXECUTION_CLOSURE_CORRECTION),
        )
        and exact_json_equal(
            migration_source_succession_basis.get("immutable_migration_receipt"),
            authority_basis_binding(ACTIVE_MIGRATION_RECEIPT),
        )
        and exact_json_equal(
            migration_source_succession_basis.get("original_migration_authority"),
            authority_basis_binding(ACTIVE_V8R4A_CORRECTION),
        )
        and isinstance(
            migration_source_succession_basis.get("user_goal_scope"), str
        )
        and bool(migration_source_succession_basis["user_goal_scope"])
        and migration_source_succession_diagnostic.get("status")
        == "diagnosed_not_authorized_by_diagnostic"
        and migration_source_succession_diagnostic.get(
            "scientific_campaign_revision"
        ) == ACTIVE_SCIENTIFIC_REVISION
        and all(
            document.get("scientific_campaign_revision")
            == ACTIVE_SCIENTIFIC_REVISION
            and document.get("infrastructure_revision")
            == ACTIVE_INFRASTRUCTURE_REVISION
            for document in (
                source_closure,
                source_dependencies,
                kill_safe,
                recovery,
                source_diagnostic,
                kill_diagnostic,
                recovery_diagnostic,
                execution_closure,
                execution_closure_diagnostic,
                migration_source_succession,
            )
        )
    ):
        _fail("V8R4/V8R4A correction authority identity drifted")
    _active_validate_execution_closure_history(root, execution_closure)
    return (
        (v8r4, "V8R4"),
        (v8r4a, "V8R4A"),
        (source_closure, "V8R4A source closure"),
        (source_dependencies, "V8R4A source dependency closure"),
        (kill_safe, "V8R4A kill-safe"),
        (recovery, "V8R4A open-lifecycle recovery"),
        (execution_closure, "V8R4A execution closure"),
        (migration_source_succession, "V8R4A migration source succession"),
        (fd_closure, "V8R4A outer guard FD closure"),
        (canary_boundary, "V8R4A denied-canary component boundary"),
        (frozen_contract_encoding, "V8R4A frozen-contract encoding"),
        (gpu_state_parent_bind, "V8R4A GPU-state parent bind"),
        (admitted_context, "V8R4A admitted benchmark context"),
    )


def _active_addendum_bindings(root: Path) -> dict[str, dict[str, Any]]:
    return {
        "source_closure_correction_authorization": _active_binding(
            root, ACTIVE_SOURCE_CLOSURE_CORRECTION
        ),
        "source_closure_dependency_authorization": _active_binding(
            root, ACTIVE_SOURCE_CLOSURE_DEPENDENCIES
        ),
        "source_closure_failure_diagnostic": _active_binding(
            root, ACTIVE_SOURCE_CLOSURE_DIAGNOSTIC
        ),
        "kill_safe_correction_authorization": _active_binding(
            root, ACTIVE_KILL_SAFE_CORRECTION
        ),
        "kill_safe_failure_diagnostic": _active_binding(
            root, ACTIVE_KILL_SAFE_DIAGNOSTIC
        ),
        "open_lifecycle_recovery_correction_authorization": _active_binding(
            root, ACTIVE_OPEN_LIFECYCLE_RECOVERY_CORRECTION
        ),
        "open_lifecycle_recovery_failure_diagnostic": _active_binding(
            root, ACTIVE_OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC
        ),
        "execution_closure_correction_authorization": _active_binding(
            root, ACTIVE_EXECUTION_CLOSURE_CORRECTION
        ),
        "execution_closure_failure_diagnostic": _active_binding(
            root, ACTIVE_EXECUTION_CLOSURE_DIAGNOSTIC
        ),
        "migration_source_succession_correction_authorization": _active_binding(
            root, ACTIVE_MIGRATION_SOURCE_SUCCESSION_CORRECTION
        ),
        "migration_source_succession_failure_diagnostic": _active_binding(
            root, ACTIVE_MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC
        ),
        "fd_closure_correction_authorization": _active_binding(
            root, ACTIVE_FD_CLOSURE_CORRECTION
        ),
        "fd_closure_failure_diagnostic": _active_binding(
            root, ACTIVE_FD_CLOSURE_DIAGNOSTIC
        ),
        "canary_boundary_correction_authorization": _active_binding(
            root, ACTIVE_CANARY_BOUNDARY_CORRECTION
        ),
        "canary_boundary_failure_diagnostic": _active_binding(
            root, ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC
        ),
        "frozen_contract_encoding_correction_authorization": _active_binding(
            root, ACTIVE_FROZEN_CONTRACT_ENCODING_CORRECTION
        ),
        "frozen_contract_encoding_failure_diagnostic": _active_binding(
            root, ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC
        ),
        "gpu_state_parent_bind_correction_authorization": _active_binding(
            root, ACTIVE_GPU_STATE_PARENT_BIND_CORRECTION
        ),
        "gpu_state_parent_bind_failure_diagnostic": _active_binding(
            root, ACTIVE_GPU_STATE_PARENT_BIND_DIAGNOSTIC
        ),
        "admitted_context_correction_authorization": _active_binding(
            root, ACTIVE_ADMITTED_CONTEXT_CORRECTION
        ),
        "admitted_context_failure_diagnostic": _active_binding(
            root, ACTIVE_ADMITTED_CONTEXT_DIAGNOSTIC
        ),
    }


def _active_validate_execution_closure_target_chain(root: Path) -> None:
    context_authority, context_authority_binding = _active_load_document(
        root, ACTIVE_ADMITTED_CONTEXT_CORRECTION
    )
    context_diagnostic, context_diagnostic_binding = _active_load_document(
        root, ACTIVE_ADMITTED_CONTEXT_DIAGNOSTIC
    )
    _active_validate_admitted_context_projection(
        context_authority, context_diagnostic
    )
    if not (
        context_authority_binding["sha256"]
        == _ADMITTED_CONTEXT_AUTHORITY_FILE_SHA256
        and context_authority_binding["bytes"] == 16684
        and context_diagnostic_binding["sha256"]
        == _ADMITTED_CONTEXT_DIAGNOSTIC_FILE_SHA256
        and context_diagnostic_binding["bytes"] == 9019
    ):
        _fail("target admitted-context immutable file binding drifted")
    parent_bind_authority, parent_bind_authority_binding = _active_load_document(
        root, ACTIVE_GPU_STATE_PARENT_BIND_CORRECTION
    )
    parent_bind_diagnostic, parent_bind_diagnostic_binding = _active_load_document(
        root, ACTIVE_GPU_STATE_PARENT_BIND_DIAGNOSTIC
    )
    _active_validate_gpu_state_parent_bind_projection(
        parent_bind_authority, parent_bind_diagnostic
    )
    if not (
        parent_bind_authority_binding["sha256"]
        == _GPU_STATE_PARENT_BIND_AUTHORITY_FILE_SHA256
        and parent_bind_authority_binding["bytes"] == 14858
        and parent_bind_diagnostic_binding["sha256"]
        == _GPU_STATE_PARENT_BIND_DIAGNOSTIC_FILE_SHA256
        and parent_bind_diagnostic_binding["bytes"] == 10709
    ):
        _fail("target GPU-state parent-bind immutable file binding drifted")
    contract_authority, contract_authority_binding = _active_load_document(
        root, ACTIVE_FROZEN_CONTRACT_ENCODING_CORRECTION
    )
    contract_diagnostic, contract_diagnostic_binding = _active_load_document(
        root, ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC
    )
    _active_validate_frozen_contract_encoding_projection(
        contract_authority, contract_diagnostic
    )
    if not (
        contract_authority_binding["sha256"]
        == _FROZEN_CONTRACT_ENCODING_AUTHORITY_FILE_SHA256
        and contract_authority_binding["bytes"] == 13460
        and contract_diagnostic_binding["sha256"]
        == _FROZEN_CONTRACT_ENCODING_DIAGNOSTIC_FILE_SHA256
        and contract_diagnostic_binding["bytes"] == 8653
    ):
        _fail("target frozen-contract immutable file binding drifted")
    canary_authority, canary_authority_binding = _active_load_document(
        root, ACTIVE_CANARY_BOUNDARY_CORRECTION
    )
    canary_diagnostic, canary_diagnostic_binding = _active_load_document(
        root, ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC
    )
    _active_validate_canary_boundary_projection(
        canary_authority, canary_diagnostic
    )
    if not (
        canary_authority_binding["sha256"]
        == _CANARY_BOUNDARY_AUTHORITY_FILE_SHA256
        and canary_authority_binding["bytes"] == 8659
        and canary_diagnostic_binding["sha256"]
        == _CANARY_BOUNDARY_DIAGNOSTIC_FILE_SHA256
        and canary_diagnostic_binding["bytes"] == 5551
    ):
        _fail("target CANARY-boundary immutable file binding drifted")
    fd_authority, _fd_authority_binding = _active_load_document(
        root, ACTIVE_FD_CLOSURE_CORRECTION
    )
    fd_diagnostic, _fd_diagnostic_binding = _active_load_document(
        root, ACTIVE_FD_CLOSURE_DIAGNOSTIC
    )
    _active_validate_fd_closure_projection(fd_authority, fd_diagnostic)
    if not (
        _fd_authority_binding["sha256"]
        == ACTIVE_HISTORICAL_FILES[ACTIVE_FD_CLOSURE_CORRECTION.as_posix()]
        and _fd_authority_binding["bytes"] == 8447
        and _fd_diagnostic_binding["sha256"]
        == ACTIVE_HISTORICAL_FILES[ACTIVE_FD_CLOSURE_DIAGNOSTIC.as_posix()]
        and _fd_diagnostic_binding["bytes"]
        == _FD_CLOSURE_DIAGNOSTIC_LEGACY_BINDING["bytes"]
    ):
        _fail("target FD-closure immutable file binding drifted")
    authority, _authority_binding = _active_load_document(
        root, ACTIVE_EXECUTION_CLOSURE_CORRECTION
    )
    diagnostic, _diagnostic_binding = _active_load_document(
        root, ACTIVE_EXECUTION_CLOSURE_DIAGNOSTIC
    )
    if not (
        set(authority) == _EXECUTION_CLOSURE_AUTHORITY_KEYS
        and set(diagnostic) == _EXECUTION_CLOSURE_DIAGNOSTIC_KEYS
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_kill_safe_capability_and_promotion_execution_closure_correction_addendum"
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_terminal_execution_closure_failure_diagnostic"
        and authority.get("scientific_campaign_revision") == ACTIVE_SCIENTIFIC_REVISION
        and authority.get("infrastructure_revision") == ACTIVE_INFRASTRUCTURE_REVISION
        and diagnostic.get("scientific_campaign_revision") == ACTIVE_SCIENTIFIC_REVISION
        and diagnostic.get("infrastructure_revision") == ACTIVE_INFRASTRUCTURE_REVISION
    ):
        _fail("target execution-closure governance identity drifted")
    _active_validate_execution_closure_projection(authority)

    def legacy_binding(relative: Path) -> dict[str, Any]:
        document, binding = _active_load_document(root, relative)
        return {
            "path": relative.as_posix(),
            "file_sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "content_sha256": document["content_sha256"],
        }

    basis = authority["authority_basis"]
    for field, relative in (
        ("diagnostic", ACTIVE_EXECUTION_CLOSURE_DIAGNOSTIC),
        ("parent_kill_safe_addendum", ACTIVE_KILL_SAFE_CORRECTION),
        (
            "parent_open_lifecycle_recovery_addendum",
            ACTIVE_OPEN_LIFECYCLE_RECOVERY_CORRECTION,
        ),
        ("parent_source_closure_addendum", ACTIVE_SOURCE_CLOSURE_CORRECTION),
    ):
        if not exact_json_equal(basis.get(field), legacy_binding(relative)):
            _fail(f"target execution-closure parent binding drifted: {field}")
    fd_basis = fd_authority.get("authority_basis")
    if not isinstance(fd_basis, Mapping):
        _fail("target FD-closure authority basis drifted")
    for field, relative in (
        ("diagnostic", ACTIVE_FD_CLOSURE_DIAGNOSTIC),
        ("parent_execution_closure_authority", ACTIVE_EXECUTION_CLOSURE_CORRECTION),
    ):
        if not exact_json_equal(fd_basis.get(field), legacy_binding(relative)):
            _fail(f"target FD-closure mounted parent binding drifted: {field}")
    canary_basis = canary_authority.get("authority_basis")
    if not isinstance(canary_basis, Mapping):
        _fail("target CANARY-boundary authority basis drifted")
    for field, relative in (
        ("diagnostic", ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC),
        ("parent_fd_closure_authority", ACTIVE_FD_CLOSURE_CORRECTION),
    ):
        if not exact_json_equal(
            canary_basis.get(field), legacy_binding(relative)
        ):
            _fail(
                f"target CANARY-boundary mounted parent binding drifted: {field}"
            )
    contract_basis = contract_authority.get("authority_basis")
    if not isinstance(contract_basis, Mapping):
        _fail("target frozen-contract authority basis drifted")
    for field, relative in (
        ("diagnostic", ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC),
        ("parent_canary_boundary_authority", ACTIVE_CANARY_BOUNDARY_CORRECTION),
        ("parent_canary_boundary_diagnostic", ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC),
        ("frozen_campaign_contract", CONTRACT),
    ):
        if not exact_json_equal(
            contract_basis.get(field), legacy_binding(relative)
        ):
            _fail(
                f"target frozen-contract mounted parent binding drifted: {field}"
            )
    parent_bind_basis = parent_bind_authority.get("authority_basis")
    if not isinstance(parent_bind_basis, Mapping):
        _fail("target GPU-state parent-bind authority basis drifted")
    for field, relative in (
        ("diagnostic", ACTIVE_GPU_STATE_PARENT_BIND_DIAGNOSTIC),
        (
            "parent_frozen_contract_authority",
            ACTIVE_FROZEN_CONTRACT_ENCODING_CORRECTION,
        ),
        (
            "parent_frozen_contract_diagnostic",
            ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC,
        ),
        ("frozen_campaign_contract", CONTRACT),
        ("gpu_state_migration_receipt", ACTIVE_MIGRATION_RECEIPT),
    ):
        if not exact_json_equal(
            parent_bind_basis.get(field), legacy_binding(relative)
        ):
            _fail(
                f"target GPU-state parent-bind mounted parent drifted: {field}"
            )

    succession, _succession_binding = _active_load_document(
        root, ACTIVE_MIGRATION_SOURCE_SUCCESSION_CORRECTION
    )
    succession_diagnostic, _succession_diagnostic_binding = _active_load_document(
        root, ACTIVE_MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC
    )
    _active_validate_migration_source_succession_projection(
        root, succession, succession_diagnostic
    )
    succession_basis = succession.get("authority_basis")
    if not (
        set(succession) == _MIGRATION_SOURCE_SUCCESSION_AUTHORITY_KEYS
        and set(succession_diagnostic)
        == _MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC_KEYS
        and succession.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_correction_addendum"
        and succession_diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_failure_diagnostic"
        and succession.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and succession.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and succession_diagnostic.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and succession_diagnostic.get("status")
        == "diagnosed_not_authorized_by_diagnostic"
        and isinstance(succession_basis, Mapping)
        and set(succession_basis)
        == {
            "diagnostic",
            "execution_closure_authority",
            "immutable_migration_receipt",
            "original_migration_authority",
            "user_goal_scope",
        }
        and isinstance(succession_basis.get("user_goal_scope"), str)
        and bool(succession_basis["user_goal_scope"])
    ):
        _fail("target migrated-state source-succession governance drifted")
    for field, relative in (
        ("diagnostic", ACTIVE_MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC),
        ("execution_closure_authority", ACTIVE_EXECUTION_CLOSURE_CORRECTION),
        ("immutable_migration_receipt", ACTIVE_MIGRATION_RECEIPT),
        ("original_migration_authority", ACTIVE_V8R4A_CORRECTION),
    ):
        if not exact_json_equal(
            succession_basis.get(field), legacy_binding(relative)
        ):
            _fail(
                f"target migrated-state source-succession parent binding drifted: {field}"
            )


def _active_document_addendums_match(
    root: Path, document: Mapping[str, Any]
) -> bool:
    return all(
        exact_json_equal(document.get(role), binding)
        for role, binding in _active_addendum_bindings(root).items()
    )


_ACTIVE_ADDENDUM_ROLES = {
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
}
_ACTIVE_TEST_RECEIPT_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision",
    "authorization_generation", "created_utc",
    "correction_authorization", "infrastructure_correction_authorization",
    "gpu_state_migration_receipt", *_ACTIVE_ADDENDUM_ROLES,
    "implementation_files", "test_paths", "command", "return_code",
    "stdout_sha256", "stdout_tail", "stdout_is_complete", "runtime_state_before",
    "stdout_bytes",
    "runtime_state_after", "all_tests_passed", "gpu_accessed",
    "target_or_outer_reference_accessed", "commercial_claim_authorized",
    "content_sha256",
}
_ACTIVE_SNAPSHOT_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision",
    "authorization_generation", "created_utc",
    "contract_file_sha256", "implementation_files", "implementation_test_receipt",
    "entry_evidence", "read_only_ancestry", "correction_authorization",
    "infrastructure_correction_authorization", "gpu_state_migration_receipt",
    *_ACTIVE_ADDENDUM_ROLES, "historical_v8r3_parent",
    "sealed_discovery_pack_indexes", "runtime_state_at_snapshot", "environment",
    "training_authorized_by_snapshot_alone", "adaptive_retrospective_only",
    "commercial_claim_authorized", "content_sha256",
}
_ACTIVE_PRETRAIN_KEYS = {
    "schema_version", "classification", "campaign_id",
    "scientific_campaign_revision", "infrastructure_revision",
    "authorization_generation", "created_utc",
    "status", "source_snapshot", "implementation_test_receipt",
    "correction_authorization", "infrastructure_correction_authorization",
    "gpu_state_migration_receipt", *_ACTIVE_ADDENDUM_ROLES,
    "canonical_gpu_state_paths", "runtime_ledger_prefixes", "snapshot_content_sha256",
    "gpu_hours_hard", "maximum_parallel_gpu_training_jobs", "gpu_budget_protocol",
    "discovery_scope", "efficiency_benchmark_scope", "promotion_reuse_scope",
    "admitted_child_scope", "adaptive_retrospective_only", "training_authorized",
    "efficiency_benchmark_authorized", "discovery_requires_passing_efficiency_benchmark",
    "production_target_sealed_runtime_authorized", "promotion_authorized",
    "commercial_claim_authorized", "outer_fold_numeric_reference_authorized",
    "content_sha256",
}


def _active_validate_surface(
    root: Path, *, require_frozen: bool
) -> list[dict[str, Any]]:
    authorities = _active_authorized_modifications(root)
    v8r3_snapshot_path = (
        root
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "V3R1_SOURCE_SNAPSHOT_V8R3.json"
    )
    v8r3 = load_json(v8r3_snapshot_path)
    prior_rows = v8r3.get("implementation_files")
    if not isinstance(prior_rows, list) or len(prior_rows) != 18:
        _fail("V8R3 source snapshot implementation cover drifted")
    prior = {
        str(row.get("path")): str(row.get("file_sha256"))
        for row in prior_rows
        if isinstance(row, Mapping)
    }
    modifications: dict[str, Mapping[str, Any]] = {}
    governed: dict[str, Mapping[str, Any]] = {}
    for authority, label in authorities:
        rows = authority.get("authorized_modifications")
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            _fail(f"{label} authorized modification list is invalid")
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                _fail(f"{label} authorized modification row is invalid")
            path = str(row["path"])
            before = row.get("before_sha256")
            if before is not None and (
                not isinstance(before, str)
                or len(before) != 64
                or any(character not in "0123456789abcdef" for character in before)
            ):
                _fail(f"{label} before hash is invalid: {path}")
            modifications[path] = row
        dependency_rows = authority.get("newly_governed_unchanged_dependencies", [])
        if not isinstance(dependency_rows, list):
            _fail(f"{label} governed dependency list is invalid")
        for row in dependency_rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                _fail(f"{label} governed dependency row is invalid")
            path = str(row["path"])
            if path in governed and not exact_json_equal(governed[path], row):
                _fail(f"governed dependency authority conflicts: {path}")
            governed[path] = row
    active_set = set(ACTIVE_IMPLEMENTATION_PATHS)
    if len(active_set) != len(ACTIVE_IMPLEMENTATION_PATHS):
        _fail("active implementation path list contains duplicates")
    if set(prior) - active_set:
        _fail("V8R3 implementation path was removed from the active cover")
    files: list[dict[str, Any]] = []
    for relative in ACTIVE_IMPLEMENTATION_PATHS:
        binding = _active_binding(
            root, relative, require_frozen=require_frozen
        )
        governed_row = governed.get(relative)
        if governed_row is not None and not (
            governed_row.get("file_sha256") == binding["sha256"]
            and governed_row.get("bytes") == binding["bytes"]
            and governed_row.get("mode_required") == "0444"
            and (not require_frozen or binding["mode"] == "0444")
        ):
            _fail(f"governed unchanged dependency drifted: {relative}")
        if (
            binding["sha256"] != prior.get(relative)
            and relative not in modifications
            and relative not in governed
        ):
            _fail(f"active implementation drift lacks correction authority: {relative}")
        if relative not in prior and relative not in modifications and relative not in governed:
            _fail(f"new active implementation lacks correction authority: {relative}")
        files.append(
            {
                "path": binding["path"],
                "file_sha256": binding["sha256"],
                "size_bytes": binding["bytes"],
                "mode": int(str(binding["mode"]), 8),
            }
        )
    token_paths: set[str] = set()
    for prefix in ("src", "scripts", "tests"):
        for path in (root / prefix).rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            _scan_binding, raw = _active_read_binding(
                root, relative, require_frozen=False
            )
            if (
                "v8r4" in path.name.lower()
                or b"V8R4" in raw
                or b"v8r4" in raw
            ):
                token_paths.add(relative)
    unknown = sorted(token_paths - active_set)
    if unknown:
        _fail("unregistered V8R4/V8R4A source paths: " + ", ".join(unknown))
    _validate_gpu_budget_module_constants(root / "src/snn_rr/gpu_budget_ledger.py")
    return files


def _active_load_migration_module(root: Path) -> Any:
    path = root / "scripts/migrate_hfr_v3r1_gpu_state_v8r4a.py"
    specification = importlib.util.spec_from_file_location(
        "_hfr_v3r1_v8r4a_migration_validator", path
    )
    if specification is None or specification.loader is None:
        _fail("cannot load V8R4A migration validator")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    prior_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    except BaseException as error:
        _fail(f"cannot load V8R4A migration validator: {error}")
    finally:
        sys.dont_write_bytecode = prior_dont_write_bytecode
        sys.modules.pop(specification.name, None)
    return module


def _active_state(
    root: Path,
    *,
    require_closed: bool,
    trusted_prelaunch_state: Mapping[str, Any] | None = None,
    lock_free: bool = False,
) -> tuple[Any, dict[str, Any]]:
    module = _active_load_migration_module(root)
    try:
        if trusted_prelaunch_state is None:
            validated = module.validate_migrated_state(
                root,
                root / ACTIVE_MIGRATION_RECEIPT,
                require_closed=require_closed,
            )
        else:
            validator = (
                module.validate_migrated_state_lock_free
                if lock_free
                else module.validate_migrated_state_target_scoped
            )
            validated = validator(
                root,
                root / ACTIVE_MIGRATION_RECEIPT,
                trusted_prelaunch_state=trusted_prelaunch_state,
                require_closed=require_closed,
            )
    except BaseException as error:
        _fail(f"V8R4A migrated GPU state validation failed: {error}")
    canonical = {
        key: str(path)
        for key, path in sorted(validated.canonical_paths.items())
    }
    summary = {
        "migration_receipt": dict(validated.receipt_binding),
        "canonical_paths": canonical,
        "directories": dict(validated.directory_bindings),
        "files": dict(validated.current_file_bindings),
        "usage_state": dict(validated.usage_state),
        "execution_state": dict(validated.execution_state),
    }
    return validated, summary


def _active_canonical_gpu_state_paths() -> dict[str, str]:
    return {
        "root": ACTIVE_STATE_ROOT.as_posix(),
        "usage_directory": ACTIVE_USAGE_LEDGER.parent.as_posix(),
        "usage_ledger": ACTIVE_USAGE_LEDGER.as_posix(),
        "execution_directory": ACTIVE_EXECUTION_LEDGER.parent.as_posix(),
        "execution_ledger": ACTIVE_EXECUTION_LEDGER.as_posix(),
        "admission_directory": ACTIVE_ADMISSION_LOCK.parent.as_posix(),
        "admission_lock": ACTIVE_ADMISSION_LOCK.as_posix(),
    }


def _active_runtime_prefixes(root: Path, state: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, relative, state_key in (
        ("usage_ledger", ACTIVE_USAGE_LEDGER, "usage_state"),
        ("execution_ledger", ACTIVE_EXECUTION_LEDGER, "execution_state"),
    ):
        binding, raw = _active_read_binding(
            root, relative, require_frozen=False
        )
        if binding["mode"] != "0644":
            _fail(f"V8R4A {role} mode drifted")
        reduced = getattr(state, state_key)
        row: dict[str, Any] = {
            "path": relative.as_posix(),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if role == "usage_ledger":
            row.update(
                {
                    "record_count": reduced["record_count"],
                    "tail_record_sha256": reduced["tail_record_sha256"],
                    "settled_usage_ns": reduced["settled_usage_ns"],
                    "open_reservation_count": reduced["open_reservation_count"],
                }
            )
        else:
            row.update(
                {
                    "record_count": reduced["record_count"],
                    "last_line_sha256": reduced["last_line_sha256"],
                    "open_start_count": reduced["open_start_count"],
                }
            )
        result[role] = row
    return result


def _active_verify_prefixes(
    root: Path, prefixes: Mapping[str, Any], *, admitted_binding: Mapping[str, Any] | None
) -> None:
    expected = {
        "usage_ledger": ACTIVE_USAGE_LEDGER,
        "execution_ledger": ACTIVE_EXECUTION_LEDGER,
    }
    if set(prefixes) != set(expected):
        _fail("V8R4A pretrain runtime prefix role set drifted")
    if not exact_json_equal(prefixes, _CONTEXT1_POSTFAILURE_PREFIXES):
        _fail("V8R4A CONTEXT1 authorization ledger prefix drifted")
    for role, relative in expected.items():
        row = prefixes.get(role)
        if not isinstance(row, Mapping) or row.get("path") != relative.as_posix():
            _fail(f"V8R4A {role} prefix path drifted")
        size = row.get("bytes")
        digest = row.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(digest, str):
            _fail(f"V8R4A {role} prefix binding is invalid")
        live_binding, raw = _active_read_binding(
            root, relative, require_frozen=False
        )
        if live_binding["mode"] != "0644":
            _fail(f"V8R4A {role} mode drifted")
        if len(raw) < size or hashlib.sha256(raw[:size]).hexdigest() != digest:
            _fail(f"V8R4A {role} was truncated or rewritten")
        if admitted_binding is not None:
            path_key = (
                "usage_ledger_path"
                if role == "usage_ledger"
                else "execution_ledger_path"
            )
            live_size_key = (
                "usage_ledger_prefix_bytes"
                if role == "usage_ledger"
                else "execution_ledger_prefix_bytes"
            )
            live_sha_key = (
                "usage_ledger_prefix_sha256"
                if role == "usage_ledger"
                else "execution_ledger_prefix_sha256"
            )
            live_size = admitted_binding.get(live_size_key)
            if not (
                admitted_binding.get(path_key) == str((root / relative).resolve())
                and type(live_size) is int
                and live_size >= size
                and live_size <= len(raw)
                and hashlib.sha256(raw[:live_size]).hexdigest()
                == admitted_binding.get(live_sha_key)
            ):
                _fail(f"V8R4A admitted {role} live prefix drifted")


def _active_trusted_state_from_authorization(
    root: Path, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the immutable lower bound used for global admitted replay."""

    receipt, receipt_binding = _active_load_document(root, ACTIVE_MIGRATION_RECEIPT)
    inventory = receipt.get("directory_inventory")
    migrated = receipt.get("migrated_state")
    replay = receipt.get("prefix_replay")
    prefixes = authorization.get("runtime_ledger_prefixes")
    if not (
        isinstance(inventory, Mapping)
        and isinstance(inventory.get("root"), Mapping)
        and isinstance(inventory.get("roles"), Mapping)
        and isinstance(migrated, Mapping)
        and isinstance(migrated.get("files"), Mapping)
        and isinstance(replay, Mapping)
        and isinstance(prefixes, Mapping)
    ):
        _fail("V8R4A admitted replay lower bound is incomplete")
    files = {
        role: dict(row)
        for role, row in migrated["files"].items()
        if isinstance(row, Mapping)
    }
    if set(files) != {
        "admission_lock",
        "execution_ledger",
        "execution_ledger_lock",
        "usage_ledger",
        "usage_ledger_lock",
    }:
        _fail("V8R4A admitted replay file role set drifted")
    for role in ("usage_ledger", "execution_ledger"):
        row = prefixes.get(role)
        if not isinstance(row, Mapping):
            _fail(f"V8R4A admitted replay {role} prefix is absent")
        files[role]["bytes"] = row.get("bytes")
        files[role]["sha256"] = row.get("sha256")
    usage_prefix = prefixes["usage_ledger"]
    execution_prefix = prefixes["execution_ledger"]
    return {
        "migration_receipt": {
            **receipt_binding,
            "content_sha256": receipt["content_sha256"],
        },
        "directories": {
            "root": dict(inventory["root"]),
            **{
                role: dict(row)
                for role, row in inventory["roles"].items()
            },
        },
        "files": files,
        "usage_state": {
            "record_count": usage_prefix.get("record_count"),
            "open_reservation_count": usage_prefix.get("open_reservation_count"),
            "settled_usage_ns": usage_prefix.get("settled_usage_ns"),
            "budget_ns": GPU_BUDGET_NS,
        },
        "execution_state": {
            "record_count": execution_prefix.get("record_count"),
            "open_start_count": execution_prefix.get("open_start_count"),
        },
    }


def _active_static_validation(
    root: Path, *, require_frozen: bool, validate_live_state: bool = True
) -> list[dict[str, Any]]:
    validate_contract(root)
    _active_validate_exact_files(root)
    files = _active_validate_surface(root, require_frozen=require_frozen)
    if validate_live_state:
        _active_state(root, require_closed=True)
    return files


def _active_venv_interpreter(root: Path) -> Path:
    """Keep the lexical venv entry so Python applies its pyvenv.cfg."""

    interpreter = Path(os.path.abspath(root / ".venv/bin/python"))
    if not interpreter.is_file():
        _fail("active virtualenv interpreter is unavailable")
    return interpreter


_ACTIVE_TEST_STDOUT_LIMIT_BYTES = 16_000
_ACTIVE_RUNTIME_STATE_KEYS = {
    "migration_receipt",
    "canonical_paths",
    "directories",
    "files",
    "usage_state",
    "execution_state",
}
_ACTIVE_TEST_SUCCESS_SUMMARY = re.compile(
    r"[1-9][0-9]* passed"
    r"(?:, [1-9][0-9]* skipped)?"
    r"(?:, [1-9][0-9]* warnings?)?"
    r" in [0-9]+(?:\.[0-9]+)?s"
)
_ACTIVE_TEST_FORBIDDEN_OUTCOME = re.compile(
    r"(?i)(?:\bfailed\b|\berrors?\b|no tests ran|interrupted)"
)


def _active_fixed_test_command(root: Path) -> list[str]:
    """Return the sole command whose evidence can authorize the active chain."""

    return [
        str(Path(os.path.abspath(root / ".venv/bin/python"))),
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        *ACTIVE_FIXED_TEST_PATHS,
    ]


_ACTIVE_CREATE_STAGE_PREDECESSORS = {
    "test_receipt": frozenset(),
    "source_snapshot": frozenset({"test_receipt"}),
    "pretrain_authorization": frozenset({"test_receipt", "source_snapshot"}),
}


def _active_validate_create_stage(root: Path, stage: str) -> None:
    """Reject stale successors, aliases, and any pre-existing fresh runtime root."""

    expected = _ACTIVE_CREATE_STAGE_PREDECESSORS.get(stage)
    if expected is None:
        _fail(f"unknown CONTEXT1 create stage: {stage}")
    trio = {
        "test_receipt": ACTIVE_TEST_RECEIPT,
        "source_snapshot": ACTIVE_SOURCE_SNAPSHOT,
        "pretrain_authorization": ACTIVE_PRETRAIN_AUTHORIZATION,
    }
    for role, relative in trio.items():
        path = root / relative
        present = os.path.lexists(path)
        if role not in expected:
            if present:
                _fail(
                    f"CONTEXT1 {stage} requires absent create-once successor: "
                    f"{relative}"
                )
            continue
        if not present:
            _fail(
                f"CONTEXT1 {stage} predecessor is absent: {relative}"
            )
        try:
            status = os.lstat(path)
            resolved = path.resolve(strict=True)
        except OSError as error:
            _fail(f"CONTEXT1 {stage} predecessor is unavailable: {relative}: {error}")
        if not (
            stat.S_ISREG(status.st_mode)
            and not stat.S_ISLNK(status.st_mode)
            and status.st_nlink == 1
            and stat.S_IMODE(status.st_mode) == 0o444
            and resolved == path.absolute()
        ):
            _fail(
                f"CONTEXT1 {stage} predecessor is not one frozen regular file: "
                f"{relative}"
            )
    for relative in (_TARGET_LIFECYCLE_ROOT, _TARGET_ACTIVE_BENCHMARK_OUTPUT_ROOT):
        if os.path.lexists(root / relative):
            _fail(
                "CONTEXT1 issuance requires a truly absent fresh runtime root: "
                f"{relative}"
            )
    _active_validate_postfailure_ledger_prefixes(root, require_exact=True)


def _active_validate_test_receipt_evidence(
    root: Path,
    document: Mapping[str, Any],
    *,
    implementation: Sequence[Mapping[str, Any]],
) -> None:
    """Purely validate the exact fixed-test evidence carried by one receipt."""

    stdout_tail = document.get("stdout_tail")
    stdout_sha256 = document.get("stdout_sha256")
    stdout_bytes = document.get("stdout_bytes")
    runtime_before = document.get("runtime_state_before")
    runtime_after = document.get("runtime_state_after")
    encoded_stdout = (
        stdout_tail.encode("utf-8") if isinstance(stdout_tail, str) else b""
    )
    terminal_summary = ""
    if isinstance(stdout_tail, str) and stdout_tail.rstrip("\n"):
        terminal_summary = stdout_tail.rstrip("\n").splitlines()[-1]
    if not (
        exact_json_equal(document.get("test_paths"), list(ACTIVE_FIXED_TEST_PATHS))
        and exact_json_equal(document.get("command"), _active_fixed_test_command(root))
        and type(document.get("return_code")) is int
        and document.get("return_code") == 0
        and isinstance(stdout_tail, str)
        and document.get("stdout_is_complete") is True
        and type(stdout_bytes) is int
        and stdout_bytes == len(encoded_stdout)
        and stdout_bytes <= _ACTIVE_TEST_STDOUT_LIMIT_BYTES
        and isinstance(stdout_sha256, str)
        and len(stdout_sha256) == 64
        and all(character in "0123456789abcdef" for character in stdout_sha256)
        and hashlib.sha256(encoded_stdout).hexdigest() == stdout_sha256
        and "[100%]" in stdout_tail
        and _ACTIVE_TEST_SUCCESS_SUMMARY.fullmatch(terminal_summary) is not None
        and _ACTIVE_TEST_FORBIDDEN_OUTCOME.search(stdout_tail) is None
        and exact_json_equal(
            document.get("implementation_files"), list(implementation)
        )
        and isinstance(runtime_before, Mapping)
        and isinstance(runtime_after, Mapping)
        and set(runtime_before) == _ACTIVE_RUNTIME_STATE_KEYS
        and set(runtime_after) == _ACTIVE_RUNTIME_STATE_KEYS
        and exact_json_equal(runtime_before, runtime_after)
    ):
        _fail("V8R4A implementation test receipt evidence drifted")


def create_test_receipt(root: Path) -> dict[str, Any]:
    _active_validate_create_stage(root, "test_receipt")
    implementation = _active_static_validation(root, require_frozen=True)
    state_before, state_before_document = _active_state(root, require_closed=True)
    _active_venv_interpreter(root)
    command = _active_fixed_test_command(root)
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=3600,
    )
    if completed.returncode != 0:
        _fail("V8R4A fixed implementation tests failed:\n" + completed.stdout[-20000:])
    _state_after, state_after_document = _active_state(root, require_closed=True)
    if not exact_json_equal(state_before_document, state_after_document):
        _fail("V8R4A fixed tests mutated migrated GPU state")
    stdout = completed.stdout.encode("utf-8")
    if len(stdout) > _ACTIVE_TEST_STDOUT_LIMIT_BYTES:
        _fail("V8R4A fixed-test stdout exceeds the complete evidence limit")
    correction = _active_binding(root, ACTIVE_V8R4_CORRECTION)
    infrastructure = _active_binding(root, ACTIVE_V8R4A_CORRECTION)
    migration = _active_binding(root, ACTIVE_MIGRATION_RECEIPT)
    document = {
        "schema_version": 1,
        "classification": ACTIVE_TEST_CLASSIFICATION,
        "campaign_id": (
            "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        ),
        "scientific_campaign_revision": ACTIVE_SCIENTIFIC_REVISION,
        "infrastructure_revision": ACTIVE_INFRASTRUCTURE_REVISION,
        "authorization_generation": ACTIVE_AUTHORIZATION_GENERATION,
        "created_utc": _active_utc_now(),
        "correction_authorization": correction,
        "infrastructure_correction_authorization": infrastructure,
        "gpu_state_migration_receipt": migration,
        **_active_addendum_bindings(root),
        "implementation_files": implementation,
        "test_paths": list(ACTIVE_FIXED_TEST_PATHS),
        "command": command,
        "return_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_bytes": len(stdout),
        "stdout_tail": completed.stdout,
        "stdout_is_complete": True,
        "runtime_state_before": state_before_document,
        "runtime_state_after": state_after_document,
        "all_tests_passed": True,
        "gpu_accessed": False,
        "target_or_outer_reference_accessed": False,
        "commercial_claim_authorized": False,
    }
    _atomic_create_json(root / ACTIVE_TEST_RECEIPT, document)
    return load_json(root / ACTIVE_TEST_RECEIPT)


def _active_validate_test_receipt(
    root: Path, implementation: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, binding = _active_load_document(
        root, ACTIVE_TEST_RECEIPT, classification=ACTIVE_TEST_CLASSIFICATION
    )
    _active_validate_trio_identity(document, label="host test receipt")
    _active_validate_test_receipt_evidence(
        root, document, implementation=implementation
    )
    if not (
        set(document) == _ACTIVE_TEST_RECEIPT_KEYS
        and type(document.get("schema_version")) is int
        and document.get("schema_version") == 1
        and document.get("scientific_campaign_revision") == ACTIVE_SCIENTIFIC_REVISION
        and document.get("infrastructure_revision") == ACTIVE_INFRASTRUCTURE_REVISION
        and document.get("authorization_generation")
        == ACTIVE_AUTHORIZATION_GENERATION
        and document.get("all_tests_passed") is True
        and document.get("return_code") == 0
        and document.get("gpu_accessed") is False
        and document.get("target_or_outer_reference_accessed") is False
        and document.get("commercial_claim_authorized") is False
        and exact_json_equal(
            document.get("correction_authorization"),
            _active_binding(root, ACTIVE_V8R4_CORRECTION),
        )
        and exact_json_equal(
            document.get("infrastructure_correction_authorization"),
            _active_binding(root, ACTIVE_V8R4A_CORRECTION),
        )
        and exact_json_equal(
            document.get("gpu_state_migration_receipt"),
            _active_binding(root, ACTIVE_MIGRATION_RECEIPT),
        )
        and _active_document_addendums_match(root, document)
    ):
        _fail("V8R4A implementation test receipt drifted")
    return document, binding


_ACTIVE_HISTORICAL_V8R3_SNAPSHOT = CAMPAIGN_DIR / "V3R1_SOURCE_SNAPSHOT_V8R3.json"
_CONTEXT1_HISTORICAL_V8R3_PARENT_BINDING = {
    "path": _ACTIVE_HISTORICAL_V8R3_SNAPSHOT.as_posix(),
    "sha256": "067fbd38a72fe2d3ec00a6645a8cb4a928f22175986d2558ca0c2c646cd97629",
    "bytes": 19888,
    "mode": "0444",
    "nlink": 1,
    "st_dev": 66306,
    "st_ino": 6969987,
}


def _active_validate_trio_identity(document: Mapping[str, Any], *, label: str) -> None:
    if not (
        type(document.get("schema_version")) is int
        and document.get("schema_version") == 1
        and document.get("authorization_generation")
        == ACTIVE_AUTHORIZATION_GENERATION
    ):
        _fail(f"{label} CONTEXT1 schema/generation identity drifted")


def _active_expected_snapshot_metadata(
    root: Path,
    test_receipt: Mapping[str, Any],
    *,
    target_safe: bool = False,
) -> dict[str, Any]:
    runtime_state = test_receipt.get("runtime_state_after")
    if not isinstance(runtime_state, Mapping):
        _fail("CONTEXT1 test receipt lacks canonical post-test runtime state")
    interpreter = (root / ".venv/bin/python").resolve(strict=True)
    # Decode canonical bytes to return a fresh, JSON-only value rather than an
    # alias into the predecessor receipt supplied by a caller.
    runtime_copy = json.loads(canonical_bytes(runtime_state).decode("utf-8"))
    historical = dict(_CONTEXT1_HISTORICAL_V8R3_PARENT_BINDING)
    if not target_safe and not exact_json_equal(
        _active_binding(root, _ACTIVE_HISTORICAL_V8R3_SNAPSHOT), historical
    ):
        _fail("CONTEXT1 historical V8R3 parent binding drifted")
    return {
        "historical_v8r3_parent": historical,
        "runtime_state_at_snapshot": runtime_copy,
        "environment": {
            "python_executable_resolved": str(interpreter),
            "python_executable_sha256": sha256_file(interpreter),
            "pyproject": _active_snapshot_binding(root, "pyproject.toml"),
        },
    }


def _active_validate_snapshot_metadata(
    root: Path,
    snapshot: Mapping[str, Any],
    test_receipt: Mapping[str, Any],
    *,
    label: str,
    target_safe: bool = False,
) -> None:
    expected = _active_expected_snapshot_metadata(
        root, test_receipt, target_safe=target_safe
    )
    if not all(
        exact_json_equal(snapshot.get(role), value)
        for role, value in expected.items()
    ):
        _fail(f"{label} CONTEXT1 source snapshot metadata drifted")


def create_source_snapshot(root: Path) -> dict[str, Any]:
    _active_validate_create_stage(root, "source_snapshot")
    implementation = _active_static_validation(root, require_frozen=True)
    test, _test_binding = _active_validate_test_receipt(root, implementation)
    _state, state_document = _active_state(root, require_closed=True)
    expected_metadata = _active_expected_snapshot_metadata(root, test)
    if not exact_json_equal(
        state_document, expected_metadata["runtime_state_at_snapshot"]
    ):
        _fail("CONTEXT1 runtime state changed between test receipt and snapshot")
    pack_indexes = [
        _active_binding(root, relative)
        for relative in sorted(ACTIVE_PACK_INDEX_FILES)
    ]
    document = {
        "schema_version": 1,
        "classification": ACTIVE_SNAPSHOT_CLASSIFICATION,
        "campaign_id": (
            "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        ),
        "scientific_campaign_revision": ACTIVE_SCIENTIFIC_REVISION,
        "infrastructure_revision": ACTIVE_INFRASTRUCTURE_REVISION,
        "authorization_generation": ACTIVE_AUTHORIZATION_GENERATION,
        "created_utc": _active_utc_now(),
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "implementation_files": implementation,
        "implementation_test_receipt": _active_snapshot_binding(
            root, ACTIVE_TEST_RECEIPT
        ),
        "entry_evidence": [],
        "read_only_ancestry": [],
        "correction_authorization": _active_binding(root, ACTIVE_V8R4_CORRECTION),
        "infrastructure_correction_authorization": _active_binding(
            root, ACTIVE_V8R4A_CORRECTION
        ),
        "gpu_state_migration_receipt": _active_binding(
            root, ACTIVE_MIGRATION_RECEIPT
        ),
        **_active_addendum_bindings(root),
        "historical_v8r3_parent": expected_metadata["historical_v8r3_parent"],
        "sealed_discovery_pack_indexes": pack_indexes,
        "runtime_state_at_snapshot": expected_metadata["runtime_state_at_snapshot"],
        "environment": expected_metadata["environment"],
        "training_authorized_by_snapshot_alone": False,
        "adaptive_retrospective_only": True,
        "commercial_claim_authorized": False,
    }
    _atomic_create_json(root / ACTIVE_SOURCE_SNAPSHOT, document)
    return load_json(root / ACTIVE_SOURCE_SNAPSHOT)


def _active_validate_snapshot(
    root: Path, *, validate_live_state: bool = True
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    implementation = _active_static_validation(
        root,
        require_frozen=True,
        validate_live_state=validate_live_state,
    )
    test, _test_binding = _active_validate_test_receipt(root, implementation)
    document, binding = _active_load_document(
        root, ACTIVE_SOURCE_SNAPSHOT, classification=ACTIVE_SNAPSHOT_CLASSIFICATION
    )
    _active_validate_trio_identity(document, label="host source snapshot")
    _active_validate_snapshot_metadata(
        root, document, test, label="host V8R4A"
    )
    if not (
        set(document) == _ACTIVE_SNAPSHOT_KEYS
        and type(document.get("schema_version")) is int
        and document.get("schema_version") == 1
        and document.get("scientific_campaign_revision") == ACTIVE_SCIENTIFIC_REVISION
        and document.get("infrastructure_revision") == ACTIVE_INFRASTRUCTURE_REVISION
        and document.get("authorization_generation")
        == ACTIVE_AUTHORIZATION_GENERATION
        and document.get("contract_file_sha256") == CONTRACT_FILE_SHA256
        and exact_json_equal(document.get("implementation_files"), implementation)
        and exact_json_equal(
            document.get("implementation_test_receipt"),
            _active_snapshot_binding(root, ACTIVE_TEST_RECEIPT),
        )
        and document.get("entry_evidence") == []
        and document.get("read_only_ancestry") == []
        and exact_json_equal(
            document.get("correction_authorization"),
            _active_binding(root, ACTIVE_V8R4_CORRECTION),
        )
        and exact_json_equal(
            document.get("infrastructure_correction_authorization"),
            _active_binding(root, ACTIVE_V8R4A_CORRECTION),
        )
        and exact_json_equal(
            document.get("gpu_state_migration_receipt"),
            _active_binding(root, ACTIVE_MIGRATION_RECEIPT),
        )
        and _active_document_addendums_match(root, document)
        and exact_json_equal(
            document.get("sealed_discovery_pack_indexes"),
            [
                _active_binding(root, relative)
                for relative in sorted(ACTIVE_PACK_INDEX_FILES)
            ],
        )
        and document.get("training_authorized_by_snapshot_alone") is False
        and document.get("adaptive_retrospective_only") is True
        and document.get("commercial_claim_authorized") is False
    ):
        _fail("V8R4A source snapshot drifted")
    return document, binding, implementation


def create_pretrain_authorization(root: Path) -> dict[str, Any]:
    _active_validate_create_stage(root, "pretrain_authorization")
    snapshot, snapshot_binding, implementation = _active_validate_snapshot(root)
    _test, test_binding = _active_validate_test_receipt(root, implementation)
    state, _state_document = _active_state(root, require_closed=True)
    prefixes = _active_runtime_prefixes(root, state)
    _active_validate_postfailure_ledger_prefixes(root, require_exact=True)
    if not exact_json_equal(prefixes, _CONTEXT1_POSTFAILURE_PREFIXES):
        _fail("V8R4A CONTEXT1 issuance requires the exact postfailure prefix")
    expected_scopes = _active_expected_pretrain_scopes()
    document = {
        "schema_version": 1,
        "classification": ACTIVE_PRETRAIN_CLASSIFICATION,
        "campaign_id": (
            "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        ),
        "scientific_campaign_revision": ACTIVE_SCIENTIFIC_REVISION,
        "infrastructure_revision": ACTIVE_INFRASTRUCTURE_REVISION,
        "authorization_generation": ACTIVE_AUTHORIZATION_GENERATION,
        "created_utc": _active_utc_now(),
        "status": "authorized",
        "source_snapshot": snapshot_binding,
        "implementation_test_receipt": test_binding,
        "correction_authorization": _active_binding(root, ACTIVE_V8R4_CORRECTION),
        "infrastructure_correction_authorization": _active_binding(
            root, ACTIVE_V8R4A_CORRECTION
        ),
        "gpu_state_migration_receipt": _active_binding(
            root, ACTIVE_MIGRATION_RECEIPT
        ),
        **_active_addendum_bindings(root),
        "canonical_gpu_state_paths": _active_canonical_gpu_state_paths(),
        "runtime_ledger_prefixes": prefixes,
        "snapshot_content_sha256": snapshot["content_sha256"],
        "gpu_hours_hard": 10.0,
        "maximum_parallel_gpu_training_jobs": 1,
        "gpu_budget_protocol": _fixed_gpu_protocol(),
        **expected_scopes,
        "adaptive_retrospective_only": True,
        "training_authorized": True,
        "efficiency_benchmark_authorized": True,
        "discovery_requires_passing_efficiency_benchmark": True,
        "production_target_sealed_runtime_authorized": True,
        "promotion_authorized": False,
        "commercial_claim_authorized": False,
        "outer_fold_numeric_reference_authorized": False,
    }
    _atomic_create_json(root / ACTIVE_PRETRAIN_AUTHORIZATION, document)
    return load_json(root / ACTIVE_PRETRAIN_AUTHORIZATION)


def _active_revalidate_admitted(
    root: Path,
    admitted_binding: Mapping[str, Any],
    authorization_path: Path,
    *,
    expected_phase: str,
    expected_context: Mapping[str, Any],
) -> dict[str, Any]:
    if expected_phase not in {
        "efficiency_benchmark",
        "discovery",
        "promotion_training",
        "promotion_prediction",
    }:
        _fail("independent V8R4A admitted-child phase is outside the campaign")
    if not isinstance(expected_context, Mapping):
        _fail("independent V8R4A admitted-child context is not an object")
    if expected_phase == "efficiency_benchmark" and not exact_json_equal(
        expected_context, _CONTEXT1_FULL_BENCHMARK_CONTEXT
    ):
        _fail("V8R4A CONTEXT1 benchmark context differs from fixed identity")
    if not (
        admitted_binding.get("phase") == expected_phase
        and exact_json_equal(admitted_binding.get("context"), dict(expected_context))
    ):
        _fail("V8R4A admitted-child phase/context differs from caller expectation")
    module = _load_gpu_admitted_validator(root)
    relative_authorization = authorization_path.relative_to(root)
    authorization_binding, _authorization_raw = _active_read_binding(
        root, relative_authorization
    )
    try:
        verified = module.revalidate_consumed_admitted_child_binding(
            admitted_binding,
            expected_campaign_id=(
                "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
            ),
            expected_phase=expected_phase,
            expected_gpu_lock_file=root / ACTIVE_ADMISSION_LOCK,
            expected_usage_ledger=root / ACTIVE_USAGE_LEDGER,
            expected_execution_ledger=root / ACTIVE_EXECUTION_LEDGER,
            expected_authorization_path=authorization_path,
            expected_authorization_sha256=authorization_binding["sha256"],
        )
    except BaseException as error:
        _fail(f"V8R4A admitted-child capability revalidation failed: {error}")
    if not (
        isinstance(verified, Mapping)
        and exact_json_equal(verified, admitted_binding)
        and verified.get("phase") == expected_phase
        and exact_json_equal(verified.get("context"), dict(expected_context))
    ):
        _fail("V8R4A admitted-child revalidator returned different bytes")
    return dict(verified)


def _active_validate_pretrain_common(
    root: Path,
    admitted_binding: Mapping[str, Any] | None,
    *,
    expected_phase: str | None = None,
    expected_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if admitted_binding is None:
        if expected_phase is not None or expected_context is not None:
            _fail("admitted phase/context supplied without an admitted binding")
    elif expected_phase is None or expected_context is None:
        _fail("admitted validation requires independent phase and context")
    snapshot, snapshot_binding, implementation = _active_validate_snapshot(
        root, validate_live_state=admitted_binding is None
    )
    _test, test_binding = _active_validate_test_receipt(root, implementation)
    path = root / ACTIVE_PRETRAIN_AUTHORIZATION
    authorization, authorization_binding = _active_load_document(
        root,
        ACTIVE_PRETRAIN_AUTHORIZATION,
        classification=ACTIVE_PRETRAIN_CLASSIFICATION,
    )
    _active_validate_trio_identity(
        authorization, label="host pretrain authorization"
    )
    _active_validate_pretrain_scope_fields(
        authorization, label="host V8R4A"
    )
    required_true = {
        "adaptive_retrospective_only",
        "training_authorized",
        "efficiency_benchmark_authorized",
        "discovery_requires_passing_efficiency_benchmark",
        "production_target_sealed_runtime_authorized",
    }
    required_false = {
        "promotion_authorized",
        "commercial_claim_authorized",
        "outer_fold_numeric_reference_authorized",
    }
    if not (
        set(authorization) == _ACTIVE_PRETRAIN_KEYS
        and type(authorization.get("schema_version")) is int
        and authorization.get("schema_version") == 1
        and authorization.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and authorization.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and authorization.get("authorization_generation")
        == ACTIVE_AUTHORIZATION_GENERATION
        and authorization.get("status") == "authorized"
        and all(authorization.get(key) is True for key in required_true)
        and all(authorization.get(key) is False for key in required_false)
        and exact_json_equal(authorization.get("source_snapshot"), snapshot_binding)
        and exact_json_equal(
            authorization.get("implementation_test_receipt"), test_binding
        )
        and exact_json_equal(
            authorization.get("correction_authorization"),
            _active_binding(root, ACTIVE_V8R4_CORRECTION),
        )
        and exact_json_equal(
            authorization.get("infrastructure_correction_authorization"),
            _active_binding(root, ACTIVE_V8R4A_CORRECTION),
        )
        and exact_json_equal(
            authorization.get("gpu_state_migration_receipt"),
            _active_binding(root, ACTIVE_MIGRATION_RECEIPT),
        )
        and _active_document_addendums_match(root, authorization)
        and exact_json_equal(
            authorization.get("canonical_gpu_state_paths"),
            _active_canonical_gpu_state_paths(),
        )
        and authorization.get("snapshot_content_sha256")
        == snapshot.get("content_sha256")
        and authorization.get("gpu_hours_hard") == 10.0
        and authorization.get("maximum_parallel_gpu_training_jobs") == 1
        and exact_json_equal(
            authorization.get("efficiency_benchmark_scope"),
            _CONTEXT1_EFFICIENCY_BENCHMARK_SCOPE,
        )
        and exact_json_equal(
            authorization.get("gpu_budget_protocol"), _fixed_gpu_protocol()
        )
    ):
        _fail("V8R4A pretrain authorization drifted")
    verified_admitted: dict[str, Any] | None = None
    if admitted_binding is not None:
        verified_admitted = _active_revalidate_admitted(
            root,
            admitted_binding,
            path,
            expected_phase=str(expected_phase),
            expected_context=expected_context,
        )
        trusted = _active_trusted_state_from_authorization(root, authorization)
        _active_state(
            root,
            require_closed=False,
            trusted_prelaunch_state=trusted,
            lock_free=True,
        )
        revalidated = _active_revalidate_admitted(
            root,
            admitted_binding,
            path,
            expected_phase=str(expected_phase),
            expected_context=expected_context,
        )
        if not exact_json_equal(revalidated, verified_admitted):
            _fail("V8R4A admitted capability changed across lock-free replay")
    else:
        _active_state(root, require_closed=True)
    prefixes = authorization.get("runtime_ledger_prefixes")
    if not isinstance(prefixes, Mapping):
        _fail("V8R4A pretrain ledger prefixes are absent")
    _active_verify_prefixes(root, prefixes, admitted_binding=verified_admitted)
    return {
        "valid": True,
        "phase": "pretrain",
        "campaign_id": (
            "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        ),
        "scientific_campaign_revision": ACTIVE_SCIENTIFIC_REVISION,
        "infrastructure_revision": ACTIVE_INFRASTRUCTURE_REVISION,
        "authorization_generation": ACTIVE_AUTHORIZATION_GENERATION,
        "adaptive_retrospective_only": True,
        "training_authorized": True,
        "efficiency_benchmark_authorized": True,
        "discovery_requires_passing_efficiency_benchmark": True,
        "production_target_sealed_runtime_authorized": True,
        "promotion_reuse_scope": authorization["promotion_reuse_scope"],
        "admitted_child_scope": authorization["admitted_child_scope"],
        "promotion_authorized": False,
        "commercial_claim_authorized": False,
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "source_snapshot_file_sha256": snapshot_binding["sha256"],
        "pretrain_authorization_path": ACTIVE_PRETRAIN_AUTHORIZATION.as_posix(),
        "pretrain_authorization_file_sha256": authorization_binding["sha256"],
        "authorization_binding": dict(authorization_binding),
        "contract_binding": _active_binding(root, CONTRACT),
        "gpu_budget_protocol": authorization["gpu_budget_protocol"],
        "gpu_lifecycle_schema_version": GPU_LIFECYCLE_SCHEMA_VERSION,
        "gpu_budget_ns": GPU_BUDGET_NS,
        "heartbeat_interval_ns": HEARTBEAT_INTERVAL_NS,
        "termination_grace_ns": TERMINATION_GRACE_NS,
        "accounting_margin_ns": ACCOUNTING_MARGIN_NS,
        "gpu_usage_ledger_path": ACTIVE_USAGE_LEDGER.as_posix(),
        "gpu_usage_genesis_record_sha256": V6_USAGE_GENESIS_RECORD_SHA256,
        "gpu_execution_ledger_path": ACTIVE_EXECUTION_LEDGER.as_posix(),
        "gpu_admission_lock_path": ACTIVE_ADMISSION_LOCK.as_posix(),
        "runtime_ledger_prefixes": dict(prefixes),
        "canonical_gpu_state_paths": authorization[
            "canonical_gpu_state_paths"
        ],
    }


def validate_pretrain(root: Path) -> dict[str, Any]:
    return _active_validate_pretrain_common(root.resolve(), None)


def validate_pretrain_admitted_child(
    root: Path,
    admitted_binding: Mapping[str, Any],
    *,
    expected_phase: str,
    expected_context: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(admitted_binding, Mapping):
        _fail("V8R4A admitted-child binding must be an object")
    return _active_validate_pretrain_common(
        root.resolve(),
        admitted_binding,
        expected_phase=expected_phase,
        expected_context=expected_context,
    )


def _active_target_documents(
    root: Path,
    capability_document: Mapping[str, Any],
    governance_bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Validate the create-once chain using only capability-mounted files."""

    _active_validate_execution_closure_target_chain(root)

    authorization, authorization_binding = _active_load_document(
        root,
        ACTIVE_PRETRAIN_AUTHORIZATION,
        classification=ACTIVE_PRETRAIN_CLASSIFICATION,
    )
    _active_validate_pretrain_scope_fields(
        authorization, label="target V8R4A"
    )
    snapshot, snapshot_binding = _active_load_document(
        root,
        ACTIVE_SOURCE_SNAPSHOT,
        classification=ACTIVE_SNAPSHOT_CLASSIFICATION,
    )
    test, test_binding = _active_load_document(
        root,
        ACTIVE_TEST_RECEIPT,
        classification=ACTIVE_TEST_CLASSIFICATION,
    )
    contract, contract_binding = _active_load_document(root, CONTRACT)
    _active_validate_trio_identity(
        test, label="target implementation test receipt"
    )
    _active_validate_trio_identity(snapshot, label="target source snapshot")
    _active_validate_trio_identity(
        authorization, label="target pretrain authorization"
    )
    expected_roles = {
        "active_authorization": authorization_binding,
        "source_snapshot": snapshot_binding,
        "implementation_test_receipt": test_binding,
        "campaign_contract": contract_binding,
        "gpu_state_migration_receipt": _active_binding(
            root, ACTIVE_MIGRATION_RECEIPT
        ),
        "correction_authorization": _active_binding(root, ACTIVE_V8R4_CORRECTION),
        "infrastructure_correction_authorization": _active_binding(
            root, ACTIVE_V8R4A_CORRECTION
        ),
        **_active_addendum_bindings(root),
    }
    for role, expected in expected_roles.items():
        if not exact_json_equal(governance_bindings.get(role), expected):
            _fail(f"target capability governance material changed: {role}")

    implementation = snapshot.get("implementation_files")
    if not isinstance(implementation, list) or not implementation:
        _fail("target source snapshot implementation cover is absent")
    observed_files: list[dict[str, Any]] = []
    for row in implementation:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "file_sha256",
            "size_bytes",
            "mode",
        }:
            _fail("target source snapshot implementation row schema drifted")
        relative = row.get("path")
        if not isinstance(relative, str):
            _fail("target source snapshot implementation path drifted")
        binding = _active_snapshot_binding(root, relative)
        if not exact_json_equal(binding, row):
            _fail(f"target source snapshot material drifted: {relative}")
        observed_files.append(binding)
    _active_validate_test_receipt_evidence(
        root, test, implementation=observed_files
    )
    _active_validate_snapshot_metadata(
        root, snapshot, test, label="target V8R4A", target_safe=True
    )
    if not (
        set(snapshot) == _ACTIVE_SNAPSHOT_KEYS
        and set(test) == _ACTIVE_TEST_RECEIPT_KEYS
        and type(snapshot.get("schema_version")) is int
        and snapshot.get("schema_version") == 1
        and type(test.get("schema_version")) is int
        and test.get("schema_version") == 1
        and snapshot.get("scientific_campaign_revision") == ACTIVE_SCIENTIFIC_REVISION
        and snapshot.get("infrastructure_revision") == ACTIVE_INFRASTRUCTURE_REVISION
        and snapshot.get("authorization_generation")
        == ACTIVE_AUTHORIZATION_GENERATION
        and test.get("authorization_generation")
        == ACTIVE_AUTHORIZATION_GENERATION
        and snapshot.get("contract_file_sha256") == CONTRACT_FILE_SHA256
        and snapshot.get("training_authorized_by_snapshot_alone") is False
        and snapshot.get("adaptive_retrospective_only") is True
        and snapshot.get("commercial_claim_authorized") is False
        and exact_json_equal(
            snapshot.get("correction_authorization"),
            expected_roles["correction_authorization"],
        )
        and exact_json_equal(
            snapshot.get("infrastructure_correction_authorization"),
            expected_roles["infrastructure_correction_authorization"],
        )
        and exact_json_equal(
            snapshot.get("gpu_state_migration_receipt"),
            expected_roles["gpu_state_migration_receipt"],
        )
        and exact_json_equal(
            snapshot.get("implementation_test_receipt"),
            _active_snapshot_binding(root, ACTIVE_TEST_RECEIPT),
        )
        and _active_document_addendums_match(root, snapshot)
        and test.get("all_tests_passed") is True
        and test.get("gpu_accessed") is False
        and test.get("target_or_outer_reference_accessed") is False
        and test.get("commercial_claim_authorized") is False
        and exact_json_equal(
            test.get("correction_authorization"),
            expected_roles["correction_authorization"],
        )
        and exact_json_equal(
            test.get("infrastructure_correction_authorization"),
            expected_roles["infrastructure_correction_authorization"],
        )
        and exact_json_equal(
            test.get("gpu_state_migration_receipt"),
            expected_roles["gpu_state_migration_receipt"],
        )
        and _active_document_addendums_match(root, test)
        and contract_binding["sha256"] == CONTRACT_FILE_SHA256
        and contract.get("campaign_id")
        == "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
    ):
        _fail("target create-once source/test/contract chain drifted")

    required_true = {
        "adaptive_retrospective_only",
        "training_authorized",
        "efficiency_benchmark_authorized",
        "discovery_requires_passing_efficiency_benchmark",
        "production_target_sealed_runtime_authorized",
    }
    required_false = {
        "promotion_authorized",
        "commercial_claim_authorized",
        "outer_fold_numeric_reference_authorized",
    }
    if not (
        set(authorization) == _ACTIVE_PRETRAIN_KEYS
        and type(authorization.get("schema_version")) is int
        and authorization.get("schema_version") == 1
        and authorization.get("scientific_campaign_revision")
        == ACTIVE_SCIENTIFIC_REVISION
        and authorization.get("infrastructure_revision")
        == ACTIVE_INFRASTRUCTURE_REVISION
        and authorization.get("authorization_generation")
        == ACTIVE_AUTHORIZATION_GENERATION
        and authorization.get("status") == "authorized"
        and all(authorization.get(key) is True for key in required_true)
        and all(authorization.get(key) is False for key in required_false)
        and exact_json_equal(authorization.get("source_snapshot"), snapshot_binding)
        and exact_json_equal(
            authorization.get("implementation_test_receipt"), test_binding
        )
        and exact_json_equal(
            authorization.get("correction_authorization"),
            expected_roles["correction_authorization"],
        )
        and exact_json_equal(
            authorization.get("infrastructure_correction_authorization"),
            expected_roles["infrastructure_correction_authorization"],
        )
        and exact_json_equal(
            authorization.get("gpu_state_migration_receipt"),
            expected_roles["gpu_state_migration_receipt"],
        )
        and _active_document_addendums_match(root, authorization)
        and exact_json_equal(
            authorization.get("canonical_gpu_state_paths"),
            _active_canonical_gpu_state_paths(),
        )
        and authorization.get("snapshot_content_sha256")
        == snapshot.get("content_sha256")
        and authorization.get("gpu_hours_hard") == 10.0
        and authorization.get("maximum_parallel_gpu_training_jobs") == 1
        and exact_json_equal(
            authorization.get("efficiency_benchmark_scope"),
            _CONTEXT1_EFFICIENCY_BENCHMARK_SCOPE,
        )
        and exact_json_equal(
            authorization.get("gpu_budget_protocol"), _fixed_gpu_protocol()
        )
    ):
        _fail("target V8R4A pretrain authorization drifted")
    if capability_document.get("governance_files", {}).get(
        "active_authorization"
    ) is None:
        _fail("target capability lacks the active authorization role")
    return (
        authorization,
        authorization_binding,
        snapshot,
        snapshot_binding,
        test_binding,
        contract_binding,
    )


def _active_target_result(
    *,
    authorization: Mapping[str, Any],
    authorization_binding: Mapping[str, Any],
    snapshot_binding: Mapping[str, Any],
    test_binding: Mapping[str, Any],
    contract_binding: Mapping[str, Any],
    capability_document: Mapping[str, Any],
    capability_binding: Mapping[str, Any],
    governance_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "valid": True,
        "phase": "pretrain",
        "campaign_id": (
            "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        ),
        "scientific_campaign_revision": ACTIVE_SCIENTIFIC_REVISION,
        "infrastructure_revision": ACTIVE_INFRASTRUCTURE_REVISION,
        "authorization_generation": ACTIVE_AUTHORIZATION_GENERATION,
        "target_scoped": True,
        "target_runtime_phase": capability_document["phase"],
        "target_runtime_outer_fold": capability_document["outer_fold"],
        "adaptive_retrospective_only": True,
        "training_authorized": True,
        "efficiency_benchmark_authorized": True,
        "discovery_requires_passing_efficiency_benchmark": True,
        "production_target_sealed_runtime_authorized": True,
        "promotion_reuse_scope": authorization["promotion_reuse_scope"],
        "admitted_child_scope": authorization["admitted_child_scope"],
        "promotion_authorized": False,
        "commercial_claim_authorized": False,
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "source_snapshot_file_sha256": snapshot_binding["sha256"],
        "pretrain_authorization_path": ACTIVE_PRETRAIN_AUTHORIZATION.as_posix(),
        "pretrain_authorization_file_sha256": authorization_binding["sha256"],
        "authorization_binding": dict(authorization_binding),
        "source_snapshot_binding": dict(snapshot_binding),
        "implementation_test_receipt_binding": dict(test_binding),
        "contract_binding": dict(contract_binding),
        "capability_binding": dict(capability_binding),
        "capability_document": dict(capability_document),
        "governance_bindings": {
            role: dict(row) for role, row in governance_bindings.items()
        },
        "gpu_budget_protocol": authorization["gpu_budget_protocol"],
        "gpu_lifecycle_schema_version": GPU_LIFECYCLE_SCHEMA_VERSION,
        "gpu_budget_ns": GPU_BUDGET_NS,
        "heartbeat_interval_ns": HEARTBEAT_INTERVAL_NS,
        "termination_grace_ns": TERMINATION_GRACE_NS,
        "accounting_margin_ns": ACCOUNTING_MARGIN_NS,
        "gpu_usage_ledger_path": ACTIVE_USAGE_LEDGER.as_posix(),
        "gpu_usage_genesis_record_sha256": V6_USAGE_GENESIS_RECORD_SHA256,
        "gpu_execution_ledger_path": ACTIVE_EXECUTION_LEDGER.as_posix(),
        "gpu_admission_lock_path": ACTIVE_ADMISSION_LOCK.as_posix(),
        "runtime_ledger_prefixes": dict(authorization["runtime_ledger_prefixes"]),
        "canonical_gpu_state_paths": authorization["canonical_gpu_state_paths"],
    }


def _active_validate_target_pretrain_common(
    root: Path,
    capability_receipt_path: Path,
    admitted_binding: Mapping[str, Any] | None,
    *,
    expected_phase: str,
    expected_outer_fold: int | None | object,
    expected_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    capability, capability_binding, governance = (
        _active_validate_target_capability(
            root,
            capability_receipt_path,
            expected_phase=expected_phase,
            expected_outer_fold=expected_outer_fold,
        )
    )
    (
        authorization,
        authorization_binding,
        _snapshot,
        snapshot_binding,
        test_binding,
        contract_binding,
    ) = _active_target_documents(root, capability, governance)
    prefixes = authorization.get("runtime_ledger_prefixes")
    if not isinstance(prefixes, Mapping):
        _fail("target pretrain ledger prefixes are absent")
    path = root / ACTIVE_PRETRAIN_AUTHORIZATION
    verified_admitted: dict[str, Any] | None = None
    if admitted_binding is None:
        if expected_context is not None:
            _fail("target context supplied without an admitted binding")
        _active_state(
            root,
            require_closed=True,
            trusted_prelaunch_state=capability["prelaunch_gpu_state"],
        )
    else:
        if not isinstance(expected_context, Mapping):
            _fail("target admitted validation requires independent context")
        verified_admitted = _active_revalidate_admitted(
            root,
            admitted_binding,
            path,
            expected_phase=expected_phase,
            expected_context=expected_context,
        )
        _active_state(
            root,
            require_closed=False,
            trusted_prelaunch_state=capability["prelaunch_gpu_state"],
            lock_free=True,
        )
        second = _active_revalidate_admitted(
            root,
            admitted_binding,
            path,
            expected_phase=expected_phase,
            expected_context=expected_context,
        )
        if not exact_json_equal(second, verified_admitted):
            _fail("target admitted capability changed across lock-free replay")
    _active_verify_prefixes(root, prefixes, admitted_binding=verified_admitted)
    capability_after, binding_after, governance_after = (
        _active_validate_target_capability(
            root,
            capability_receipt_path,
            expected_phase=expected_phase,
            expected_outer_fold=expected_outer_fold,
        )
    )
    if not (
        exact_json_equal(capability_after, capability)
        and exact_json_equal(binding_after, capability_binding)
        and exact_json_equal(governance_after, governance)
    ):
        _fail("target capability changed during pretrain validation")
    return _active_target_result(
        authorization=authorization,
        authorization_binding=authorization_binding,
        snapshot_binding=snapshot_binding,
        test_binding=test_binding,
        contract_binding=contract_binding,
        capability_document=capability,
        capability_binding=capability_binding,
        governance_bindings=governance,
    )


def validate_pretrain_target_scoped(
    root: Path,
    capability_receipt_path: Path,
    *,
    expected_phase: str,
    expected_outer_fold: int | None | object = _ACTIVE_UNSET,
) -> dict[str, Any]:
    return _active_validate_target_pretrain_common(
        root.resolve(),
        capability_receipt_path,
        None,
        expected_phase=expected_phase,
        expected_outer_fold=expected_outer_fold,
        expected_context=None,
    )


def validate_pretrain_target_scoped_admitted_child(
    root: Path,
    capability_receipt_path: Path,
    admitted_binding: Mapping[str, Any],
    *,
    expected_phase: str,
    expected_context: Mapping[str, Any],
    expected_outer_fold: int | None | object = _ACTIVE_UNSET,
) -> dict[str, Any]:
    if not isinstance(admitted_binding, Mapping):
        _fail("target V8R4A admitted-child binding must be an object")
    return _active_validate_target_pretrain_common(
        root.resolve(),
        capability_receipt_path,
        admitted_binding,
        expected_phase=expected_phase,
        expected_outer_fold=expected_outer_fold,
        expected_context=expected_context,
    )


def validate_phase(root: Path, phase: str) -> dict[str, Any]:
    root = root.resolve()
    if phase == "implementation":
        files = _active_static_validation(root, require_frozen=False)
        return {
            "valid": True,
            "phase": "implementation",
            "campaign_id": (
                "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
            ),
            "scientific_campaign_revision": ACTIVE_SCIENTIFIC_REVISION,
            "infrastructure_revision": ACTIVE_INFRASTRUCTURE_REVISION,
            "authorization_generation": ACTIVE_AUTHORIZATION_GENERATION,
            "implementation_authorized": True,
            "training_authorized": False,
            "present_authorized_files": len(files),
            "expected_authorized_files": len(ACTIVE_IMPLEMENTATION_PATHS),
            "active_pretrain_authorization_path": (
                ACTIVE_PRETRAIN_AUTHORIZATION.as_posix()
            ),
            "commercial_claim_authorized": False,
        }
    return validate_pretrain(root)


# Consumers intentionally import this legacy public name.  Repoint it only
# after all historical functions have been defined so V8R3 replay constants
# above remain untouched.
PRETRAIN_AUTHORIZATION = ACTIVE_PRETRAIN_AUTHORIZATION
TEST_RECEIPT = ACTIVE_TEST_RECEIPT
SOURCE_SNAPSHOT = ACTIVE_SOURCE_SNAPSHOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--action",
        choices=(
            "validate-implementation",
            "create-test-receipt",
            "create-source-snapshot",
            "create-pretrain-authorization",
            "validate-pretrain",
        ),
        default="validate-implementation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()
    try:
        if args.action == "validate-implementation":
            result = validate_phase(root, "implementation")
        elif args.action == "create-test-receipt":
            result = create_test_receipt(root)
        elif args.action == "create-source-snapshot":
            result = create_source_snapshot(root)
        elif args.action == "create-pretrain-authorization":
            result = create_pretrain_authorization(root)
        else:
            result = validate_pretrain(root)
    except (AuthorizationError, subprocess.SubprocessError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
