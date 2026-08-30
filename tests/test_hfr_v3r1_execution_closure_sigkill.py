from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import stat
import sys
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = ROOT / "scripts/run_gpu_admitted.py"
TRAINER_PATH = ROOT / "scripts/train_harmonic_factor_router_snn_v3r1.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


admitted = _load("sigkill_run_gpu_admitted", WRAPPER_PATH)
budget = admitted.budget
trainer = _load("sigkill_train_hfr_v3r1", TRAINER_PATH)


def _fork_and_require_sigkill(action: Callable[[], None]) -> None:
    pid = os.fork()
    if pid == 0:
        try:
            action()
        except BaseException:
            os._exit(91)
        os._exit(92)
    waited, status = os.waitpid(pid, 0)
    assert waited == pid
    assert os.WIFSIGNALED(status), status
    assert os.WTERMSIG(status) == signal.SIGKILL


def _kill_current_process(_point: str) -> None:
    os.kill(os.getpid(), signal.SIGKILL)


def _assert_immutable_single_link(path: Path, *, size: int | None = None) -> None:
    status = path.stat()
    assert stat.S_ISREG(status.st_mode)
    assert stat.S_IMODE(status.st_mode) == 0o444
    assert status.st_nlink == 1
    if size is not None:
        assert status.st_size == size


def test_sigkill_usage_state_replace_leaves_one_valid_recoverable_residue(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "usage.jsonl"

    def killed_append() -> None:
        budget._FAULT_INJECTION_HOOK = (
            lambda point: _kill_current_process(point)
            if point == "state_replace_after_payload_fsync"
            else None
        )
        budget.append_record(
            ledger,
            {"schema_version": 1, "event": "killed", "elapsed_seconds": 0},
        )

    _fork_and_require_sigkill(killed_append)
    residue = ledger.parent / budget.atomic_replace_residue_name(ledger)
    residue_status = residue.stat()
    assert stat.S_IMODE(residue_status.st_mode) == 0o644
    assert residue_status.st_nlink == 1
    assert not ledger.exists()
    callbacks: list[int] = []
    assert budget.cleanup_usage_ledger_replace_residue(
        ledger,
        admission_revalidate=lambda: callbacks.append(1),
    ) is True
    assert len(callbacks) >= 3
    assert not residue.exists()
    assert budget.verify_ledger(ledger).records == ()
    budget.append_record(
        ledger,
        {"schema_version": 1, "event": "retry", "elapsed_seconds": 0},
    )
    assert [row["event"] for row in budget.verify_ledger(ledger).records] == [
        "retry"
    ]


def test_sigkill_execution_state_replace_leaves_one_valid_recoverable_residue(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "execution.jsonl"

    def killed_append() -> None:
        budget._FAULT_INJECTION_HOOK = (
            lambda point: _kill_current_process(point)
            if point == "state_replace_after_payload_fsync"
            else None
        )
        admitted.append_ledger(
            ledger,
            {
                "schema_version": 1,
                "job_id": "killed",
                "event": "start",
                "utc": "2026-08-29T00:00:00+00:00",
            },
        )

    _fork_and_require_sigkill(killed_append)
    residue = ledger.parent / budget.atomic_replace_residue_name(ledger)
    assert residue.exists() and not ledger.exists()
    callbacks: list[int] = []
    assert admitted.cleanup_execution_ledger_replace_residue(
        ledger,
        admission_revalidate=lambda: callbacks.append(1),
    ) is True
    assert len(callbacks) >= 3
    assert not residue.exists()
    directory_fd = budget.open_pinned_directory(tmp_path)
    try:
        assert admitted._read_execution_locked(ledger, directory_fd)[0] == b""
    finally:
        os.close(directory_fd)


def test_state_residue_cleanup_refuses_non_successor_without_mutation(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "usage.jsonl"
    budget.append_record(
        ledger,
        {"schema_version": 1, "event": "current", "elapsed_seconds": 0},
    )
    before = ledger.read_bytes()
    residue = ledger.parent / budget.atomic_replace_residue_name(ledger)
    residue.write_bytes(b"not-a-successor\n")
    residue.chmod(0o644)
    with pytest.raises(budget.LedgerError, match="strict current-ledger successor"):
        budget.cleanup_usage_ledger_replace_residue(
            ledger, admission_revalidate=lambda: None
        )
    assert ledger.read_bytes() == before
    assert residue.read_bytes() == b"not-a-successor\n"


def _valid_v1_successor_bytes() -> bytes:
    record = budget._with_hash(
        {"schema_version": 1, "event": "successor", "elapsed_seconds": 0},
        None,
    )
    payload = budget.canonical_json_bytes(record) + b"\n"
    budget.verify_ledger_bytes(payload)
    return payload


@pytest.mark.parametrize("tamper", ["mode", "hardlink", "symlink"])
def test_state_residue_cleanup_refuses_metadata_tamper(
    tmp_path: Path, tamper: str
) -> None:
    ledger = tmp_path / "usage.jsonl"
    residue = ledger.parent / budget.atomic_replace_residue_name(ledger)
    payload = _valid_v1_successor_bytes()
    if tamper == "symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(payload)
        outside.chmod(0o644)
        residue.symlink_to(outside.name)
    else:
        residue.write_bytes(payload)
        residue.chmod(0o444 if tamper == "mode" else 0o644)
        if tamper == "hardlink":
            os.link(residue, tmp_path / "alias")
    with pytest.raises((budget.LedgerError, OSError)):
        budget.cleanup_usage_ledger_replace_residue(
            ledger, admission_revalidate=lambda: None
        )
    assert not ledger.exists()
    assert residue.exists() or residue.is_symlink()


def test_state_residue_cleanup_live_owner_callback_refuses_without_mutation(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "usage.jsonl"
    residue = ledger.parent / budget.atomic_replace_residue_name(ledger)
    payload = _valid_v1_successor_bytes()
    residue.write_bytes(payload)
    residue.chmod(0o644)

    def live_owner() -> None:
        raise RuntimeError("matching live owner")

    with pytest.raises(RuntimeError, match="live owner"):
        budget.cleanup_usage_ledger_replace_residue(
            ledger, admission_revalidate=live_owner
        )
    assert not ledger.exists()
    assert residue.read_bytes() == payload


@pytest.mark.parametrize(
    ("point", "result_exists_after_kill"),
    [
        ("anonymous_create_before_link", False),
        ("anonymous_create_after_link_before_directory_fsync", True),
    ],
)
def test_sigkill_anonymous_terminal_publication_is_idempotent(
    tmp_path: Path, point: str, result_exists_after_kill: bool
) -> None:
    result = tmp_path / "terminal.json"
    lock = budget.result_receipt_lock_path(result)
    budget.atomic_create_immutable_bytes(lock, b"")
    value = {"classification": "sigkill-test", "value": 3}

    def killed_publish() -> None:
        budget._FAULT_INJECTION_HOOK = (
            lambda observed: _kill_current_process(observed)
            if observed == point
            else None
        )
        budget.atomic_result_receipt(result, value)

    _fork_and_require_sigkill(killed_publish)
    assert result.exists() is result_exists_after_kill
    recovered = budget.atomic_result_receipt(result, value)
    assert recovered["value"] == 3
    _assert_immutable_single_link(result)
    _assert_immutable_single_link(lock, size=0)
    assert set(item.name for item in tmp_path.iterdir()) == {
        result.name,
        lock.name,
    }


@pytest.mark.parametrize(
    "point",
    ["wrapper_after_terminal_commit", "wrapper_after_result_publication"],
)
def test_sigkill_wrapper_terminal_boundaries_replay_without_new_science(
    tmp_path: Path, point: str
) -> None:
    arguments: dict[str, object] = {
        "lock_file": tmp_path / "gpu.lock",
        "ledger": tmp_path / "execution.jsonl",
        "command": [sys.executable, "-c", "raise SystemExit(0)"],
        "usage_ledger": tmp_path / "usage.jsonl",
        "result_file": tmp_path / "terminal.json",
        "campaign_id": "test-campaign",
        "phase": "discovery",
        "context": {"unit": "sigkill-wrapper"},
        "invocation_sha256": "a" * 64,
    }

    def killed_wrapper() -> None:
        admitted._FAULT_INJECTION_HOOK = (
            lambda observed: _kill_current_process(observed)
            if observed == point
            else None
        )
        admitted.run(**arguments)

    _fork_and_require_sigkill(killed_wrapper)
    assert admitted.run(**arguments) == 0
    result = Path(arguments["result_file"])
    lock = budget.result_receipt_lock_path(result)
    _assert_immutable_single_link(result)
    _assert_immutable_single_link(lock, size=0)
    usage = budget.require_closed_ledger(Path(arguments["usage_ledger"]))
    reservations = [row for row in usage.records if row.get("event") == "reservation"]
    terminals = [row for row in usage.records if row.get("event") == "terminal"]
    assert len(reservations) == len(terminals) == 1
    execution_rows = [
        admitted.json.loads(line)
        for line in Path(arguments["ledger"]).read_text().splitlines()
    ]
    assert [row["event"] for row in execution_rows] == ["start", "end"]


@pytest.mark.parametrize(
    ("point", "final_exists"),
    [
        ("trainer_atomic_after_payload_fsync", False),
        ("trainer_atomic_after_replace_before_directory_fsync", True),
    ],
)
def test_sigkill_trainer_named_temp_retry_is_exact(
    tmp_path: Path, point: str, final_exists: bool
) -> None:
    output = tmp_path / "last.pt"
    payload = b"complete-checkpoint-bytes"

    def killed_publish() -> None:
        trainer._FAULT_INJECTION_HOOK = (
            lambda observed: _kill_current_process(observed)
            if observed == point
            else None
        )
        trainer._atomic_publish_immutable(
            output,
            lambda descriptor: trainer._write_all(descriptor, payload),
        )

    _fork_and_require_sigkill(killed_publish)
    assert output.exists() is final_exists
    if not final_exists:
        temporaries = [
            item
            for item in tmp_path.iterdir()
            if trainer._KILL_SAFE_TEMP_PATTERN.fullmatch(item.name)
        ]
        assert len(temporaries) == 1
        _assert_immutable_single_link(temporaries[0])
        assert trainer.cleanup_stale_atomic_temporaries(tmp_path) == (
            temporaries[0].name,
        )
    trainer._atomic_publish_immutable(
        output,
        lambda descriptor: trainer._write_all(descriptor, payload),
    )
    _assert_immutable_single_link(output, size=len(payload))
    assert output.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [output]
