from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_sealed_nested_proposer_supervisor.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_sealed_nested_proposer_supervisor", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
SUPERVISOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPERVISOR)


def _sealed(value: dict) -> dict:
    result = dict(value)
    result["content_sha256"] = SUPERVISOR.canonical_content_sha256(result)
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _args(tmp_path: Path, *, expected: int = 3) -> argparse.Namespace:
    runner = tmp_path / "runner.py"
    runner.write_text("# runner\n", encoding="utf-8")
    seal = tmp_path / "prelaunch_seal.json"
    seal_value = {
        "schema_version": 1,
        "classification": "supplemental_runtime_input_byte_inventory",
        "attestation_phase": "prelaunch",
        "post_launch_attestation": False,
        "created_utc": "2026-08-28T00:00:00+00:00",
    }
    seal_value["content_sha256"] = SUPERVISOR.runtime_seal.canonical_sha256(
        seal_value
    )
    _write_json(seal, seal_value)
    control = tmp_path / "control"
    _write_json(
        control / "status.json",
        _sealed(
            {
                "state": "dry_run",
                "requested_units": expected,
                "completed_units": 0,
            }
        ),
    )
    return argparse.Namespace(
        runner=runner.resolve(),
        runtime_seal=seal.resolve(),
        manifest_root=(tmp_path / "non_test_manifests").resolve(),
        control_root=control.resolve(),
        run_root=(tmp_path / "non_test_runs").resolve(),
        outer_folds="3,4",
        expected_units=expected,
        gpu_lock=(tmp_path / "gpu.lock").resolve(),
        gpu_ledger=(tmp_path / "gpu.jsonl").resolve(),
        python_executable=Path("/bin/true"),
        attestation=(tmp_path / "execution_attestation.json").resolve(),
    )


def test_supervisor_reverifies_and_advances_exactly_one_unit_per_call(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    seal_document = json.loads(args.runtime_seal.read_text(encoding="utf-8"))
    verifier_calls = 0

    def verify(_: Path) -> dict:
        nonlocal verifier_calls
        verifier_calls += 1
        return {
            "content_sha256": seal_document["content_sha256"],
            "verified_files": 9,
        }

    calls: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess:
        calls.append(command)
        status_path = args.control_root / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        completed = int(status["completed_units"]) + 1
        state = "complete" if completed == args.expected_units else "stopped_after_max_new_units"
        _write_json(
            status_path,
            _sealed(
                {
                    "state": state,
                    "requested_units": args.expected_units,
                    "completed_units": completed,
                }
            ),
        )
        records = [{"unit": index} for index in range(completed)]
        _write_json(
            args.control_root / "index.json",
            _sealed(
                {
                    "requested_units": args.expected_units,
                    "completed_units": completed,
                    "outer_test_opened": False,
                    "outer_test_record_count": 0,
                    "records": records,
                }
            ),
        )
        return subprocess.CompletedProcess(command, 0)

    result = SUPERVISOR.run(
        args, command_runner=runner, seal_verifier=verify
    )
    assert len(calls) == args.expected_units
    assert all(command[-2:] == ["--max-new-units", "1"] for command in calls)
    assert verifier_calls == 2 * args.expected_units + 2
    assert result["runtime_seal_verified_before_and_after_every_invocation"] is True
    assert result["completed_units"] == args.expected_units
    assert result["commercial_claim_authorized"] is False
    assert args.attestation.is_file()


def test_success_without_progress_fails_closed(tmp_path: Path) -> None:
    args = _args(tmp_path, expected=1)
    seal = json.loads(args.runtime_seal.read_text(encoding="utf-8"))
    verify = lambda _: {
        "content_sha256": seal["content_sha256"],
        "verified_files": 1,
    }
    with pytest.raises(RuntimeError, match="no forward progress"):
        SUPERVISOR.run(
            args,
            command_runner=lambda command, **_: subprocess.CompletedProcess(command, 0),
            seal_verifier=verify,
        )


def test_runtime_drift_after_unit_fails_before_next_invocation(tmp_path: Path) -> None:
    args = _args(tmp_path, expected=1)
    seal = json.loads(args.runtime_seal.read_text(encoding="utf-8"))
    count = 0

    def verify(_: Path) -> dict:
        nonlocal count
        count += 1
        return {
            "content_sha256": (
                seal["content_sha256"] if count == 1 else "f" * 64
            ),
            "verified_files": 1,
        }

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess:
        _write_json(
            args.control_root / "status.json",
            _sealed(
                {
                    "state": "complete",
                    "requested_units": 1,
                    "completed_units": 1,
                }
            ),
        )
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(RuntimeError, match="changed during"):
        SUPERVISOR.run(args, command_runner=runner, seal_verifier=verify)


def test_test_or_target_paths_are_forbidden_without_opening_them(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.manifest_root = (tmp_path / "test_pred_0" / "not-json").resolve()
    with pytest.raises(RuntimeError, match="forbidden"):
        SUPERVISOR.run(args)
    assert not args.manifest_root.exists()


def test_parse_args_preserves_virtualenv_launcher_symlink(tmp_path: Path) -> None:
    launcher = tmp_path / "venv-python"
    launcher.symlink_to(Path(sys.executable))
    args = SUPERVISOR.parse_args(["--python-executable", str(launcher)])
    assert args.python_executable == launcher.absolute()
    assert args.python_executable != launcher.resolve()
