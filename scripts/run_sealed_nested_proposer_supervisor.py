#!/usr/bin/env python3
"""Resume a non-test proposer campaign one sealed unit at a time.

The underlying runner and its immutable plan remain unchanged.  This narrow
supervisor rehashes a prelaunch source/cache payload seal immediately before
and after every one-unit invocation, preventing a long campaign from silently
continuing after runtime-input drift.  It accepts no test manifest or target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import seal_runtime_inputs as runtime_seal  # noqa: E402


CAMPAIGN_ROOT = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer"
)
DEFAULT_MANIFEST_ROOT = CAMPAIGN_ROOT / "full_oof_non_test/manifests"
DEFAULT_CONTROL_ROOT = CAMPAIGN_ROOT / "current_source_retrain_f34/control"
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/nested_proposer_current_source_retrain_f34"
)
DEFAULT_RUNTIME_SEAL = (
    CAMPAIGN_ROOT
    / "current_source_retrain_f34/execution_prelaunch_runtime_input_seal.json"
)
DEFAULT_ATTESTATION = (
    CAMPAIGN_ROOT / "current_source_retrain_f34/execution_attestation.json"
)
DEFAULT_GPU_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2"

CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(
    path: Path,
    label: str,
    *,
    content_hash: Callable[[Mapping[str, Any]], str] = canonical_content_sha256,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    if "content_sha256" in value and content_hash(value) != value.get("content_sha256"):
        raise RuntimeError(f"{label} content hash mismatch")
    return value


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"missing bound file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _assert_non_test_text(values: Sequence[str]) -> None:
    for value in values:
        lowered = value.lower()
        if "test_pred_" in lowered or "--test" in lowered or "target" in lowered:
            raise RuntimeError(f"test/target input is forbidden in sealed supervisor: {value}")


def build_unit_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python_executable),
        str(args.runner),
        "--manifest-root",
        str(args.manifest_root),
        "--control-root",
        str(args.control_root),
        "--run-root",
        str(args.run_root),
        "--outer-folds",
        args.outer_folds,
        "--gpu-lock",
        str(args.gpu_lock),
        "--gpu-ledger",
        str(args.gpu_ledger),
        "--max-new-units",
        "1",
    ]
    _assert_non_test_text(command)
    return command


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or target.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite supervisor attestation: {target}")
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(
    args: argparse.Namespace,
    *,
    command_runner: CommandRunner = subprocess.run,
    seal_verifier: Callable[[Path], Mapping[str, Any]] = runtime_seal.verify,
) -> dict[str, Any]:
    _assert_non_test_text(
        [
            str(args.manifest_root),
            str(args.control_root),
            str(args.run_root),
            str(args.runtime_seal),
        ]
    )
    # Runtime seals deliberately exclude ``created_utc`` from their semantic
    # content hash, whereas campaign status/index documents do not.  Validate
    # each document with the hash ABI of its producer.
    seal_document = _read_json(
        args.runtime_seal,
        "execution runtime seal",
        content_hash=runtime_seal.canonical_sha256,
    )
    if (
        seal_document.get("post_launch_attestation") is not False
        or seal_document.get("attestation_phase") != "prelaunch"
    ):
        raise RuntimeError("execution runtime seal must be a prelaunch attestation")
    expected_seal_hash = str(seal_document.get("content_sha256", ""))
    command = build_unit_command(args)
    completed_before: int | None = None
    invocations = 0
    while True:
        before = seal_verifier(args.runtime_seal)
        if before.get("content_sha256") != expected_seal_hash:
            raise RuntimeError("runtime seal identity changed before unit invocation")
        status_path = args.control_root / "status.json"
        if status_path.is_file():
            status = _read_json(status_path, "campaign status")
            completed = int(status.get("completed_units", -1))
            requested = int(status.get("requested_units", -1))
            if requested != args.expected_units or not 0 <= completed <= requested:
                raise RuntimeError("campaign status population differs from supervisor")
            if status.get("state") == "complete":
                if completed != requested:
                    raise RuntimeError("campaign reports complete without full cover")
                break
            if status.get("state") == "failed":
                raise RuntimeError(f"underlying campaign failed: {status.get('failure')}")
            if completed_before is not None and completed < completed_before:
                raise RuntimeError("campaign completed-unit count moved backwards")
            completed_before = completed

        result = command_runner(command, cwd=PROJECT_ROOT, check=False)
        invocations += 1
        if int(result.returncode) != 0:
            raise RuntimeError(
                "sealed one-unit campaign invocation failed with status "
                f"{result.returncode}: {shlex.join(command)}"
            )
        after = seal_verifier(args.runtime_seal)
        if after.get("content_sha256") != expected_seal_hash:
            raise RuntimeError("runtime inputs changed during unit invocation")
        status = _read_json(status_path, "campaign status")
        now = int(status.get("completed_units", -1))
        if completed_before is not None and now <= completed_before:
            raise RuntimeError("successful one-unit invocation made no forward progress")
        completed_before = now
        if invocations > args.expected_units + 1:
            raise RuntimeError("supervisor exceeded the bounded unit invocation count")

    final_index_path = args.control_root / "index.json"
    final_index = _read_json(final_index_path, "completed campaign index")
    if (
        int(final_index.get("requested_units", -1)) != args.expected_units
        or int(final_index.get("completed_units", -1)) != args.expected_units
        or final_index.get("outer_test_opened") is not False
        or int(final_index.get("outer_test_record_count", -1)) != 0
        or len(final_index.get("records", ())) != args.expected_units
    ):
        raise RuntimeError("completed campaign index is incomplete or not test-sealed")
    final_verify = seal_verifier(args.runtime_seal)
    if final_verify.get("content_sha256") != expected_seal_hash:
        raise RuntimeError("runtime inputs changed before final attestation")
    attestation: dict[str, Any] = {
        "schema_version": 1,
        "classification": "sealed_non_test_proposer_execution_attestation",
        "outer_test_opened": False,
        "outer_test_record_count": 0,
        "commercial_claim_authorized": False,
        "expected_units": args.expected_units,
        "completed_units": args.expected_units,
        "one_new_unit_per_invocation": True,
        "runtime_seal_verified_before_and_after_every_invocation": True,
        "invocations_this_resume": invocations,
        "runtime_input_seal": {
            **_binding(args.runtime_seal),
            "content_sha256": expected_seal_hash,
            "verified_files": int(final_verify["verified_files"]),
        },
        "campaign_index": {
            **_binding(final_index_path),
            "content_sha256": final_index.get("content_sha256"),
        },
        "unit_command": command,
        "supervisor": _binding(Path(__file__)),
    }
    attestation["content_sha256"] = canonical_content_sha256(attestation)
    _write_immutable(args.attestation, attestation)
    return attestation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner",
        type=Path,
        default=PROJECT_ROOT / "scripts/run_full_nested_proposer_campaign.py",
    )
    parser.add_argument("--runtime-seal", type=Path, default=DEFAULT_RUNTIME_SEAL)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--outer-folds", default="3,4")
    parser.add_argument("--expected-units", type=int, default=30)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_ROOT / "gpu_admission.lock")
    parser.add_argument(
        "--gpu-ledger", type=Path, default=DEFAULT_GPU_ROOT / "gpu_admission_ledger.jsonl"
    )
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--attestation", type=Path, default=DEFAULT_ATTESTATION)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    for name in (
        "runner",
        "runtime_seal",
        "manifest_root",
        "control_root",
        "run_root",
        "gpu_lock",
        "gpu_ledger",
        "attestation",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    # Keep the virtual-environment launcher path itself.  Resolving its symlink
    # selects the base interpreter and drops the venv's site-packages.
    args.python_executable = Path(
        os.path.abspath(args.python_executable.expanduser())
    )
    if not args.python_executable.is_file() or not os.access(
        args.python_executable, os.X_OK
    ):
        raise ValueError("--python-executable must be an executable file")
    if args.expected_units < 1:
        raise ValueError("--expected-units must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "status": "complete",
                "completed_units": result["completed_units"],
                "content_sha256": result["content_sha256"],
                "outer_test_opened": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
