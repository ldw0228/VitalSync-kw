from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gpu_admitted.py"
SPEC = importlib.util.spec_from_file_location("run_gpu_admitted", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
admitted = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(admitted)
budget = admitted.budget


def _immutable_authorization(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "PRETRAIN_AUTHORIZATION_V8.json"
    path.write_bytes(b'{"classification":"test-v8-pretrain-authorization"}\n')
    path.chmod(0o444)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _binding_consumer_command(
    *,
    lock_file: Path,
    usage_ledger: Path,
    execution_ledger: Path,
    authorization: Path,
    authorization_sha256: str,
    output: Path,
    consume_twice: bool = False,
) -> list[str]:
    lines = [
        "import importlib.util,json,pathlib,sys",
        "sys.dont_write_bytecode=True",
        f"spec=importlib.util.spec_from_file_location('admitted_child',{str(SCRIPT)!r})",
        "module=importlib.util.module_from_spec(spec)",
        "spec.loader.exec_module(module)",
        (
            "binding=module.consume_admitted_child_binding("
            f"'efficiency_benchmark',pathlib.Path({str(usage_ledger)!r}),"
            f"pathlib.Path({str(execution_ledger)!r}),"
            f"pathlib.Path({str(authorization)!r}),{authorization_sha256!r},"
            "expected_campaign_id='test-campaign',"
            f"expected_gpu_lock_file=pathlib.Path({str(lock_file)!r}))"
        ),
        f"pathlib.Path({str(output)!r}).write_text(json.dumps(binding,sort_keys=True),encoding='utf-8')",
    ]
    if consume_twice:
        lines.extend(
            [
                "try:",
                (
                    " module.consume_admitted_child_binding("
                    f"'efficiency_benchmark',pathlib.Path({str(usage_ledger)!r}),"
                    f"pathlib.Path({str(execution_ledger)!r}),"
                    f"pathlib.Path({str(authorization)!r}),{authorization_sha256!r},"
                    "expected_campaign_id='test-campaign',"
                    f"expected_gpu_lock_file=pathlib.Path({str(lock_file)!r}))"
                ),
                "except RuntimeError as error:",
                " assert 'descriptor is absent' in str(error)",
                "else:",
                " raise AssertionError('admitted-child capability was reusable')",
            ]
        )
    return [sys.executable, "-c", "\n".join(lines)]


def _budgeted_cli_command(
    *,
    lock_file: Path,
    execution_ledger: Path,
    usage_ledger: Path,
    result_file: Path,
    authorization: Path,
    authorization_sha256: str,
    command: list[str],
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--lock-file",
        str(lock_file),
        "--ledger",
        str(execution_ledger),
        "--usage-ledger",
        str(usage_ledger),
        "--result-file",
        str(result_file),
        "--campaign-id",
        "test-campaign",
        "--phase",
        "efficiency_benchmark",
        "--context-json",
        '{"unit":"v8-recovery"}',
        "--invocation-sha256",
        "a" * 64,
        "--authorization-path",
        str(authorization),
        "--authorization-sha256",
        authorization_sha256,
        "--budget-seconds",
        "36000",
        "--",
        *command,
    ]


def _v7_arguments(tmp_path: Path, *, name: str = "job") -> dict[str, object]:
    return {
        "lock_file": tmp_path / "gpu.lock",
        "ledger": tmp_path / "gpu.jsonl",
        "command": [sys.executable, "-c", "raise SystemExit(0)"],
        "usage_ledger": tmp_path / "usage.jsonl",
        "result_file": tmp_path / f"{name}.result.json",
        "campaign_id": "test-campaign",
        "phase": "discovery",
        "context": {"unit": name},
        "invocation_sha256": "a" * 64,
    }


def _short_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(budget, "HEARTBEAT_INTERVAL_NS", 20_000_000)
    monkeypatch.setattr(budget, "TERMINATION_GRACE_NS", 100_000_000)
    monkeypatch.setattr(budget, "ACCOUNTING_MARGIN_NS", 50_000_000)
    monkeypatch.setattr(budget, "RECOVERY_MARGIN_NS", 30_000_000)


def _watchdog_test_pin(path: Path) -> tuple[int, object, list[dict[str, object]]]:
    descriptor = budget.open_pinned_directory(path)
    lease = budget.acquire_directory_generation_fence(descriptor)
    status = os.fstat(descriptor)
    return descriptor, lease, [
        {
            "path": str(path.resolve()),
            "fd": descriptor,
            "st_dev": status.st_dev,
            "st_ino": status.st_ino,
        }
    ]


def test_run_records_start_and_end(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    ledger = tmp_path / "ledger.jsonl"
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    assert admitted.run(lock, ledger, command) == 7
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["start", "end"]
    assert rows[0]["job_id"] == rows[1]["job_id"]
    assert rows[1]["exit_code"] == 7
    assert rows[0]["command"] == command
    assert len(rows[0]["command_sha256"]) == 64


def test_held_lock_fails_closed_without_ledger_entry(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    ledger = tmp_path / "ledger.jsonl"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already held"):
            admitted.run(lock, ledger, [sys.executable, "-c", "pass"])
    assert not ledger.exists()


def test_symlinked_lifecycle_paths_fail_before_reservation_or_spawn(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("")
    usage_link = tmp_path / "usage.jsonl"
    usage_link.symlink_to(target)
    arguments = _v7_arguments(tmp_path)
    arguments["usage_ledger"] = usage_link
    with pytest.raises(RuntimeError, match="symlinked"):
        admitted.run(**arguments)
    assert target.read_bytes() == b""
    assert not arguments["ledger"].exists()


def test_budgeted_run_reserves_before_spawn_and_publishes_terminal_result(
    tmp_path: Path,
) -> None:
    arguments = _v7_arguments(tmp_path)
    assert admitted.run(**arguments) == 0
    state = budget.require_closed_ledger(arguments["usage_ledger"])
    assert [record["event"] for record in state.records] == ["reservation", "terminal"]
    reservation, terminal = state.records
    assert reservation["record_sha256"] == terminal["reservation_record_sha256"]
    assert state.settled_usage_ns == terminal["charged_usage_ns"]
    result = budget.load_validate_terminal_result(
        arguments["result_file"],
        usage_ledger=arguments["usage_ledger"],
        expected_campaign_id="test-campaign",
        expected_phase="discovery",
        expected_context={"unit": "job"},
        expected_command_sha256=budget.command_sha256(arguments["command"]),
        expected_invocation_sha256="a" * 64,
    )
    assert result["reusable_success"] is True
    gpu_events = [
        json.loads(line) for line in arguments["ledger"].read_text().splitlines()
    ]
    assert [record["event"] for record in gpu_events] == ["start", "end"]


def test_reservation_durability_failure_prevents_child_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned = False

    def forbidden_popen(*_args: object, **_kwargs: object) -> object:
        nonlocal spawned
        spawned = True
        raise AssertionError("Popen must not run before reservation durability")

    def fail_replace(_path: Path, _payload: bytes, **_kwargs: object) -> None:
        raise OSError("injected fsync/replace failure")

    monkeypatch.setattr(admitted.subprocess, "Popen", forbidden_popen)
    monkeypatch.setattr(budget, "_atomic_replace_bytes", fail_replace)
    with pytest.raises(OSError, match="injected"):
        admitted.run(**_v7_arguments(tmp_path))
    assert spawned is False
    assert not (tmp_path / "gpu.jsonl").exists()
    assert not (tmp_path / "usage.jsonl").exists()


def test_spawn_failure_closes_durable_reservation_and_publishes_failure_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_spawn(*_args: object, **_kwargs: object) -> object:
        raise OSError("injected Popen failure")

    arguments = _v7_arguments(tmp_path, name="spawn-failure")
    monkeypatch.setattr(admitted.subprocess, "Popen", fail_spawn)
    with pytest.raises(OSError, match="injected Popen"):
        admitted.run(**arguments)
    state = budget.require_closed_ledger(arguments["usage_ledger"])
    assert [record["event"] for record in state.records] == ["reservation", "terminal"]
    terminal = state.records[-1]
    assert terminal["return_code"] == 127
    assert terminal["reuse_eligible"] is False
    result = budget.load_validate_terminal_result(
        arguments["result_file"], usage_ledger=arguments["usage_ledger"]
    )
    assert result["reusable_success"] is False


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signal_is_forwarded_child_is_reaped_and_terminal_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signum: signal.Signals,
) -> None:
    _short_lifecycle(monkeypatch)
    arguments = _v7_arguments(tmp_path, name=signum.name)
    arguments["command"] = [sys.executable, "-c", "import time; time.sleep(5)"]
    arguments["budget_ns"] = 2_000_000_000
    sender = threading.Timer(0.15, os.kill, args=(os.getpid(), int(signum)))
    sender.start()
    try:
        assert admitted.run(**arguments) == 128 + int(signum)
    finally:
        sender.join(timeout=2)
    state = budget.require_closed_ledger(
        arguments["usage_ledger"], budget_ns=2_000_000_000
    )
    terminal = state.records[-1]
    assert terminal["event"] == "terminal"
    assert terminal["received_signal"] == int(signum)
    assert terminal["return_code"] == -int(signum)


def test_term_ignoring_child_is_sigkilled_inside_reserved_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    arguments = _v7_arguments(tmp_path, name="ignore-term")
    arguments["command"] = [
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(5)",
    ]
    arguments["budget_ns"] = 2_000_000_000
    sender = threading.Timer(0.2, os.kill, args=(os.getpid(), signal.SIGTERM))
    started = time.monotonic()
    sender.start()
    try:
        assert admitted.run(**arguments) == 128 + signal.SIGTERM
    finally:
        sender.join(timeout=2)
    assert time.monotonic() - started < 2.0
    state = budget.require_closed_ledger(
        arguments["usage_ledger"], budget_ns=2_000_000_000
    )
    terminal = state.records[-1]
    assert terminal["termination_escalated"] is True
    assert terminal["return_code"] == -signal.SIGKILL


def test_hard_timeout_keeps_term_grace_and_margin_inside_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    arguments = _v7_arguments(tmp_path, name="hard-timeout")
    arguments["command"] = [
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(5)",
    ]
    arguments["budget_ns"] = 350_000_000
    started = time.monotonic()
    assert admitted.run(**arguments) == 124
    assert time.monotonic() - started < 1.0
    state = budget.require_closed_ledger(
        arguments["usage_ledger"], budget_ns=350_000_000
    )
    reservation, *_, terminal = state.records
    assert reservation["workload_timeout_ns"] == 200_000_000
    assert terminal["hard_timeout_reached"] is True
    assert terminal["termination_escalated"] is True
    assert terminal["charged_usage_ns"] <= reservation["reservation_ns"]


def test_periodic_heartbeat_is_bound_to_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    arguments = _v7_arguments(tmp_path, name="heartbeat")
    arguments["command"] = [sys.executable, "-c", "import time; time.sleep(.12)"]
    arguments["budget_ns"] = 2_000_000_000
    assert admitted.run(**arguments) == 0
    state = budget.require_closed_ledger(
        arguments["usage_ledger"], budget_ns=2_000_000_000
    )
    heartbeats = [record for record in state.records if record["event"] == "heartbeat"]
    assert heartbeats
    reservation = state.records[0]
    assert all(
        record["reservation_record_sha256"] == reservation["record_sha256"]
        for record in heartbeats
    )
    assert state.records[-1]["last_heartbeat_record_sha256"] == heartbeats[-1][
        "record_sha256"
    ]


def test_stable_lock_serializes_concurrent_atomic_appends(tmp_path: Path) -> None:
    ledger = tmp_path / "usage.jsonl"

    def append(index: int) -> str:
        record = budget.append_record(
            ledger,
            {
                "schema_version": 1,
                "event": "concurrency-test",
                "index": index,
                "elapsed_seconds": 0,
            },
        )
        return record["record_sha256"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(append, range(32)))
    state = budget.verify_ledger(ledger)
    assert len(state.records) == 32
    assert len(set(hashes)) == 32
    assert {record["index"] for record in state.records} == set(range(32))
    assert budget.ledger_lock_path(ledger).exists()


def test_atomic_replace_failure_preserves_exact_old_valid_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "usage.jsonl"
    budget.append_record(
        ledger, {"schema_version": 1, "event": "old", "elapsed_seconds": 1}
    )
    old = ledger.read_bytes()

    def fail_replace(_source: object, _target: object, **_kwargs: object) -> None:
        raise OSError("injected replace crash")

    monkeypatch.setattr(budget.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        budget.append_record(
            ledger, {"schema_version": 1, "event": "new", "elapsed_seconds": 1}
        )
    assert ledger.read_bytes() == old
    state = budget.verify_ledger(ledger)
    assert len(state.records) == 1


def test_mixed_frozen_v1_genesis_and_v2_lifecycle_validates(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "campaign_gpu_usage_chain_v6.jsonl"
    )
    ledger = tmp_path / "usage.jsonl"
    ledger.write_bytes(source.read_bytes())
    arguments = _v7_arguments(tmp_path, name="mixed")
    arguments["usage_ledger"] = ledger
    assert admitted.run(**arguments) == 0
    state = budget.require_closed_ledger(
        ledger,
        expected_legacy_genesis_sha256=budget.LEGACY_V1_GENESIS_RECORD_SHA256,
    )
    assert state.records[0]["schema_version"] == 1
    assert state.records[1]["event"] == "reservation"
    assert state.records[-1]["event"] == "terminal"
    assert state.settled_usage_ns >= 377_000_000_000


def test_schema_omitted_unfrozen_v1_genesis_is_rejected_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _v7_arguments(tmp_path, name="schema-omitted-v1")
    usage = arguments["usage_ledger"]
    assert isinstance(usage, Path)
    genesis = budget._with_hash(  # noqa: SLF001
        {"event": "legacy", "elapsed_seconds": 0}, None
    )
    usage.write_bytes(budget.canonical_json_bytes(genesis) + b"\n")
    spawned = False

    def forbidden_popen(*_args: object, **_kwargs: object) -> object:
        nonlocal spawned
        spawned = True
        raise AssertionError("unfrozen V1 genesis must fail before child spawn")

    monkeypatch.setattr(admitted.subprocess, "Popen", forbidden_popen)
    with pytest.raises(budget.LedgerError, match="frozen V6 record"):
        admitted.run(**arguments)
    assert spawned is False


def test_same_boot_dead_reconciliation_uses_heartbeat_ceiling_plus_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    ledger = tmp_path / "usage.jsonl"
    template = {
        "lifecycle_id": "abandoned",
        "campaign_id": "test-campaign",
        "phase": "discovery",
        "context": {"unit": "dead"},
        "invocation_sha256": "b" * 64,
        "command_sha256": "c" * 64,
        "result_path": str((tmp_path / "dead.result.json").resolve()),
        "gpu_execution_ledger_path": str((tmp_path / "gpu.jsonl").resolve()),
        "boot_id": budget.boot_id(),
        "wrapper_pid": 999_999_999,
        "wrapper_start_ticks": 1,
        "realtime_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
    }
    reservation, _, _ = budget.reconcile_and_reserve(
        ledger, template, budget_ns=2_000_000_000
    )
    heartbeat = budget.append_heartbeat(
        ledger,
        reservation,
        sequence=1,
        elapsed_ceiling_ns=40_000_000,
        realtime_ns=time.time_ns(),
        monotonic_ns=time.monotonic_ns(),
        child_pid=999_999_998,
        child_start_ticks=1,
        budget_ns=2_000_000_000,
    )
    assert heartbeat["event"] == "heartbeat"
    reconciled, state = budget.reconcile_open_reservations(
        ledger,
        realtime_ns=time.time_ns(),
        monotonic_ns=time.monotonic_ns(),
        budget_ns=2_000_000_000,
    )
    assert reconciled[0]["reconciliation_mode"] == "same_boot_proven_dead_ceiling"
    assert reconciled[0]["charged_usage_ns"] == 70_000_000
    assert not state.open_reservations


def test_same_boot_reconciliation_charges_current_monotonic_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    ledger = tmp_path / "usage.jsonl"
    started = time.monotonic_ns()
    template = {
        "lifecycle_id": "stopped-then-dead",
        "campaign_id": "test-campaign",
        "phase": "discovery",
        "context": {"unit": "stopped"},
        "invocation_sha256": "1" * 64,
        "command_sha256": "2" * 64,
        "result_path": str((tmp_path / "stopped.result.json").resolve()),
        "gpu_execution_ledger_path": str((tmp_path / "gpu.jsonl").resolve()),
        "boot_id": budget.boot_id(),
        "wrapper_pid": 999_999_999,
        "wrapper_start_ticks": 1,
        "realtime_ns": time.time_ns(),
        "monotonic_ns": started,
    }
    reservation, _, _ = budget.reconcile_and_reserve(
        ledger, template, budget_ns=2_000_000_000
    )
    budget.append_heartbeat(
        ledger,
        reservation,
        sequence=1,
        elapsed_ceiling_ns=40_000_000,
        realtime_ns=time.time_ns(),
        monotonic_ns=started + 40_000_000,
        child_pid=999_999_998,
        child_start_ticks=1,
        budget_ns=2_000_000_000,
    )
    reconciled, _ = budget.reconcile_open_reservations(
        ledger,
        realtime_ns=time.time_ns(),
        monotonic_ns=started + 400_000_000,
        budget_ns=2_000_000_000,
    )
    assert reconciled[0]["charged_usage_ns"] == 400_000_000


def test_live_matching_reservation_cannot_be_reconciled_or_mutated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    ledger = tmp_path / "usage.jsonl"
    pid = os.getpid()
    ticks = budget.process_start_ticks(pid)
    assert ticks is not None
    template = {
        "lifecycle_id": "live",
        "campaign_id": "test-campaign",
        "phase": "discovery",
        "context": {"unit": "live"},
        "invocation_sha256": "3" * 64,
        "command_sha256": "4" * 64,
        "result_path": str((tmp_path / "live.result.json").resolve()),
        "gpu_execution_ledger_path": str((tmp_path / "gpu.jsonl").resolve()),
        "boot_id": budget.boot_id(),
        "wrapper_pid": pid,
        "wrapper_start_ticks": ticks,
        "realtime_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
    }
    budget.reconcile_and_reserve(ledger, template, budget_ns=2_000_000_000)
    before = ledger.read_bytes()
    with pytest.raises(budget.LedgerBusy, match="matching live wrapper"):
        budget.reconcile_open_reservations(
            ledger,
            realtime_ns=time.time_ns(),
            monotonic_ns=time.monotonic_ns(),
            budget_ns=2_000_000_000,
        )
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("variant", ["whitespace", "crlf", "unicode_escape"])
def test_usage_jsonl_rejects_noncanonical_equivalent_raw_line(
    tmp_path: Path, variant: str
) -> None:
    ledger = tmp_path / "usage.jsonl"
    record = budget.append_record(
        ledger, {"schema_version": 1, "event": "café", "elapsed_seconds": 1}
    )
    if variant == "whitespace":
        raw = json.dumps(
            record,
            sort_keys=True,
            separators=(", ", ": "),
            ensure_ascii=False,
        ).encode("utf-8") + b"\n"
    elif variant == "crlf":
        raw = budget.canonical_json_bytes(record) + b"\r\n"
    else:
        raw = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8") + b"\n"
    ledger.write_bytes(raw)
    with pytest.raises(budget.LedgerError, match="canonical"):
        budget.verify_ledger(ledger)


def test_v2_reservation_requires_absolute_execution_ledger_identity(
    tmp_path: Path,
) -> None:
    template = {
        "lifecycle_id": "missing-execution-path",
        "campaign_id": "test-campaign",
        "phase": "discovery",
        "context": {"unit": "missing"},
        "invocation_sha256": "5" * 64,
        "command_sha256": "6" * 64,
        "result_path": str((tmp_path / "missing.result.json").resolve()),
        "boot_id": budget.boot_id(),
        "wrapper_pid": 999_999_999,
        "wrapper_start_ticks": 1,
        "realtime_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
    }
    ledger = tmp_path / "usage.jsonl"
    with pytest.raises(budget.LedgerError, match="gpu_execution_ledger_path"):
        budget.reconcile_and_reserve(ledger, template)
    assert not ledger.exists()


def test_terminal_over_reservation_is_full_charged_invariant_breach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    ledger = tmp_path / "usage.jsonl"
    started = time.monotonic_ns()
    template = {
        "lifecycle_id": "deadline-breach",
        "campaign_id": "test-campaign",
        "phase": "discovery",
        "context": {"unit": "breach"},
        "invocation_sha256": "7" * 64,
        "command_sha256": "8" * 64,
        "result_path": str((tmp_path / "breach.result.json").resolve()),
        "gpu_execution_ledger_path": str((tmp_path / "gpu.jsonl").resolve()),
        "boot_id": budget.boot_id(),
        "wrapper_pid": 999_999_999,
        "wrapper_start_ticks": 1,
        "realtime_ns": time.time_ns(),
        "monotonic_ns": started,
    }
    reservation, _, _ = budget.reconcile_and_reserve(
        ledger, template, budget_ns=300_000_000
    )
    elapsed = int(reservation["reservation_ns"]) + 1
    terminal = budget.append_terminal(
        ledger,
        reservation,
        last_heartbeat=None,
        elapsed_ns=elapsed,
        charged_usage_ns=int(reservation["reservation_ns"]),
        realtime_ns=time.time_ns(),
        monotonic_ns=started + elapsed,
        child_pid=999_999_998,
        child_start_ticks=1,
        return_code=0,
        wrapper_exit_code=125,
        hard_timeout_reached=False,
        received_signal=None,
        termination_escalated=False,
        budget_ns=300_000_000,
    )
    assert terminal["reservation_deadline_breached"] is True
    assert terminal["charged_usage_ns"] == reservation["reservation_ns"]
    assert terminal["reuse_eligible"] is False


def test_execution_ledger_atomic_failure_keeps_exact_valid_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "gpu.jsonl"
    common = {"schema_version": 1, "job_id": "atomic-execution"}
    admitted.append_ledger(ledger, {**common, "event": "start"})
    old = ledger.read_bytes()

    def fail_replace(
        _path: Path, _payload: bytes, **_kwargs: object
    ) -> None:
        raise OSError("injected execution replace failure")

    monkeypatch.setattr(budget, "_atomic_replace_bytes", fail_replace)
    with pytest.raises(OSError, match="execution replace"):
        admitted.append_ledger(ledger, {**common, "event": "end"})
    assert ledger.read_bytes() == old
    assert [row["event"] for row in admitted._decode_execution_ledger(old)] == [
        "start"
    ]


def test_ancestor_symlink_and_protected_hardlink_aliases_fail_closed(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real, target_is_directory=True)
    arguments = _v7_arguments(tmp_path, name="ancestor-link")
    arguments["usage_ledger"] = linked_parent / "usage.jsonl"
    with pytest.raises(RuntimeError, match="symlinked component"):
        admitted.run(**arguments)

    arguments = _v7_arguments(tmp_path, name="hardlink")
    execution = arguments["ledger"]
    usage = arguments["usage_ledger"]
    assert isinstance(execution, Path) and isinstance(usage, Path)
    execution.write_bytes(b"")
    os.link(execution, usage)
    with pytest.raises(RuntimeError, match="aliased|inode aliases"):
        admitted.run(**arguments)

    external_case = tmp_path / "external-case"
    external_case.mkdir()
    arguments = _v7_arguments(external_case, name="external-hardlink")
    usage = arguments["usage_ledger"]
    assert isinstance(usage, Path)
    usage.write_bytes(b"")
    os.link(usage, tmp_path / "unprotected-alias")
    with pytest.raises(RuntimeError, match="aliased"):
        admitted.run(**arguments)


def test_locked_ledger_detects_parent_directory_replacement(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    ledger = protected / "usage.jsonl"
    budget.append_record(
        ledger, {"schema_version": 1, "event": "initial", "elapsed_seconds": 0}
    )
    moved = tmp_path / "moved"
    with budget._exclusive_ledger_lock(ledger) as locked:
        protected.rename(moved)
        protected.mkdir()
        try:
            with pytest.raises(budget.LedgerError, match="directory identity changed"):
                locked.revalidate()
        finally:
            protected.rmdir()
            moved.rename(protected)


def test_stable_usage_lock_hardlink_alias_fails_without_mutation(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "usage.jsonl"
    budget.append_record(
        ledger, {"schema_version": 1, "event": "initial", "elapsed_seconds": 0}
    )
    before = ledger.read_bytes()
    os.link(budget.ledger_lock_path(ledger), tmp_path / "external-lock-alias")
    with pytest.raises(budget.LedgerError, match="lock"):
        budget.append_record(
            ledger, {"schema_version": 1, "event": "later", "elapsed_seconds": 0}
        )
    assert ledger.read_bytes() == before


def test_locked_closed_snapshot_blocks_writer_until_context_exit(tmp_path: Path) -> None:
    ledger = tmp_path / "usage.jsonl"
    budget.append_record(
        ledger, {"schema_version": 1, "event": "initial", "elapsed_seconds": 0}
    )
    completed = threading.Event()

    def writer() -> None:
        budget.append_record(
            ledger, {"schema_version": 1, "event": "later", "elapsed_seconds": 0}
        )
        completed.set()

    with budget.locked_closed_snapshot(ledger) as snapshot:
        assert len(snapshot.records) == 1
        thread = threading.Thread(target=writer)
        thread.start()
        time.sleep(0.05)
        assert not completed.is_set()
    thread.join(timeout=2)
    assert completed.is_set()


def test_uncertain_boot_change_charges_full_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    ledger = tmp_path / "usage.jsonl"
    template = {
        "lifecycle_id": "different-boot",
        "campaign_id": "test-campaign",
        "phase": "discovery",
        "context": {"unit": "uncertain"},
        "invocation_sha256": "d" * 64,
        "command_sha256": "e" * 64,
        "result_path": str((tmp_path / "uncertain.result.json").resolve()),
        "gpu_execution_ledger_path": str((tmp_path / "gpu.jsonl").resolve()),
        "boot_id": "prior-boot",
        "wrapper_pid": 999_999_999,
        "wrapper_start_ticks": 1,
        "realtime_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
    }
    reservation, _, _ = budget.reconcile_and_reserve(
        ledger, template, budget_ns=2_000_000_000
    )
    reconciled, state = budget.reconcile_open_reservations(
        ledger,
        realtime_ns=time.time_ns(),
        monotonic_ns=time.monotonic_ns(),
        budget_ns=2_000_000_000,
    )
    assert reconciled[0]["reconciliation_mode"] == "full_reservation"
    assert reconciled[0]["charged_usage_ns"] == reservation["reservation_ns"]
    assert state.remaining_ns == 0


def test_terminal_before_result_gap_recovers_without_rerunning_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _v7_arguments(tmp_path, name="recover")
    assert admitted.run(**arguments) == 0
    result = arguments["result_file"]
    result.unlink()

    def forbidden_popen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a durable terminal must recover without GPU rerun")

    monkeypatch.setattr(admitted.subprocess, "Popen", forbidden_popen)
    assert admitted.run(**arguments) == 0
    recovered = budget.load_validate_terminal_result(
        result,
        usage_ledger=arguments["usage_ledger"],
        expected_context={"unit": "recover"},
    )
    assert recovered["reusable_success"] is True
    state = budget.require_closed_ledger(arguments["usage_ledger"])
    assert [record["event"] for record in state.records] == ["reservation", "terminal"]


def test_durable_terminal_recovers_missing_execution_end_before_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _v7_arguments(tmp_path, name="recover-end")
    original_append = admitted.append_ledger

    def fail_end(
        path: Path, record: dict[str, object], **kwargs: object
    ) -> None:
        if record.get("event") == "end":
            raise OSError("injected execution end publication crash")
        original_append(path, record, **kwargs)

    monkeypatch.setattr(admitted, "append_ledger", fail_end)
    with pytest.raises(OSError, match="end publication"):
        admitted.run(**arguments)
    state = budget.require_closed_ledger(arguments["usage_ledger"])
    assert state.records[-1]["event"] == "terminal"
    assert not arguments["result_file"].exists()
    gpu_rows = arguments["ledger"].read_text().splitlines()
    assert [json.loads(line)["event"] for line in gpu_rows] == ["start"]
    monkeypatch.setattr(admitted, "append_ledger", original_append)

    def forbidden_popen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("durable terminal recovery must not respawn the child")

    monkeypatch.setattr(admitted.subprocess, "Popen", forbidden_popen)
    assert admitted.run(**arguments) == 0
    recovered_rows = [
        json.loads(line) for line in arguments["ledger"].read_text().splitlines()
    ]
    assert [row["event"] for row in recovered_rows] == ["start", "end"]
    assert recovered_rows[-1]["recovered_from_durable_usage_terminal"] is True


def test_campaign_parent_sigkill_contains_wrapper_and_child_group(tmp_path: Path) -> None:
    gpu_lock = tmp_path / "gpu.lock"
    gpu_ledger = tmp_path / "gpu.jsonl"
    usage_ledger = tmp_path / "usage.jsonl"
    result_file = tmp_path / "result.json"
    child_ready = tmp_path / "child.ready"
    escaped = tmp_path / "child.escaped"
    wrapper_pid_file = tmp_path / "wrapper.pid"
    child_command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys,time; "
            "Path(sys.argv[1]).write_text('ready'); time.sleep(3); "
            "Path(sys.argv[2]).write_text('escaped')"
        ),
        str(child_ready),
        str(escaped),
    ]
    wrapper_command = [
        sys.executable,
        str(SCRIPT),
        "--lock-file",
        str(gpu_lock),
        "--ledger",
        str(gpu_ledger),
        "--usage-ledger",
        str(usage_ledger),
        "--result-file",
        str(result_file),
        "--campaign-id",
        "test-campaign",
        "--phase",
        "discovery",
        "--context-json",
        '{"unit":"parent-death"}',
        "--invocation-sha256",
        "f" * 64,
        "--budget-seconds",
        "36000",
        "--",
        *child_command,
    ]
    launcher_code = (
        "from pathlib import Path; import subprocess,sys; "
        "p=subprocess.Popen(sys.argv[2:]); Path(sys.argv[1]).write_text(str(p.pid)); p.wait()"
    )
    launcher = subprocess.Popen(
        [sys.executable, "-c", launcher_code, str(wrapper_pid_file), *wrapper_command]
    )

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not child_ready.exists():
        time.sleep(0.02)
    assert child_ready.exists(), "contained child did not start"
    assert wrapper_pid_file.exists()
    os.kill(launcher.pid, signal.SIGKILL)
    launcher.wait(timeout=2)
    assert launcher.returncode == -signal.SIGKILL

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not result_file.exists():
        time.sleep(0.02)
    assert result_file.exists(), "wrapper did not publish a terminal after parent death"
    time.sleep(0.2)
    assert not escaped.exists()
    state = budget.require_closed_ledger(usage_ledger)
    terminal = state.records[-1]
    assert terminal["event"] == "terminal"
    assert terminal["received_signal"] == signal.SIGTERM
    assert terminal["return_code"] == -signal.SIGTERM


def test_external_watchdog_enforces_deadline_while_wrapper_is_sigstopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    ready = tmp_path / "ready"
    escaped = tmp_path / "escaped"
    lock = tmp_path / "gpu.lock"
    execution = tmp_path / "gpu.jsonl"
    usage = tmp_path / "usage.jsonl"
    result = tmp_path / "result.json"
    harness = (
        "import importlib.util,sys; from pathlib import Path; "
        f"s=importlib.util.spec_from_file_location('w', {str(SCRIPT)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "m.budget.HEARTBEAT_INTERVAL_NS=20000000; "
        "m.budget.TERMINATION_GRACE_NS=100000000; "
        "m.budget.ACCOUNTING_MARGIN_NS=50000000; "
        "m.budget.RECOVERY_MARGIN_NS=30000000; "
        "cmd=[sys.executable,'-c',"
        "'import signal,time,sys; from pathlib import Path; '"
        "+'signal.signal(signal.SIGTERM, signal.SIG_IGN); '"
        "+'Path(sys.argv[1]).write_text(\\\"ready\\\"); time.sleep(2); '"
        "+'Path(sys.argv[2]).write_text(\\\"escaped\\\")',sys.argv[7],sys.argv[8]]; "
        "raise SystemExit(m.run(Path(sys.argv[1]),Path(sys.argv[2]),cmd,"
        "usage_ledger=Path(sys.argv[3]),result_file=Path(sys.argv[4]),"
        "campaign_id='test-campaign',phase='discovery',"
        "context={'unit':'sigstop'},invocation_sha256='9'*64,budget_ns=350000000))"
    )
    wrapper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            harness,
            str(lock),
            str(execution),
            str(usage),
            str(result),
            "unused5",
            "unused6",
            str(ready),
            str(escaped),
        ]
    )
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.01)
    assert ready.exists()
    os.kill(wrapper.pid, signal.SIGSTOP)
    time.sleep(0.7)
    assert not escaped.exists()
    os.kill(wrapper.pid, signal.SIGCONT)
    assert wrapper.wait(timeout=3) == 124
    terminal_result = budget.load_validate_terminal_result(
        result, usage_ledger=usage, budget_ns=350_000_000
    )
    assert terminal_result["hard_timeout_reached"] is True
    assert terminal_result["reservation_deadline_breached"] is False
    budget.require_closed_ledger(usage, budget_ns=350_000_000)


def test_watchdog_never_spawns_workload_after_absolute_deadline(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "spawned"
    ready_read, ready_write = os.pipe()
    final_read, final_write = os.pipe()
    pin_fd, pin_lease, pin_spec = _watchdog_test_pin(tmp_path)
    spec = {
        "wrapper_pid": os.getpid(),
        "ready_fd": ready_write,
        "final_fd": final_write,
        "command": [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('bad')",
            str(marker),
        ],
        "workload_deadline_ns": time.monotonic_ns() - 1,
        "termination_grace_ns": 20_000_000,
        "accounting_margin_ns": 20_000_000,
        "protected_directory_pins": pin_spec,
    }
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--internal-watchdog-json",
            budget.canonical_json_bytes(spec).decode("utf-8"),
        ],
        pass_fds=(ready_write, final_write, pin_fd),
        close_fds=True,
    )
    os.close(ready_write)
    os.close(final_write)
    try:
        assert process.wait(timeout=2) == 71
        ready = admitted._read_pipe_json(ready_read, timeout_seconds=1)
        final = admitted._read_pipe_json(final_read, timeout_seconds=1)
    finally:
        os.close(ready_read)
        os.close(final_read)
        budget.release_directory_generation_fence(pin_lease)
        os.close(pin_fd)
    assert not marker.exists()
    assert ready["child_pid"] is None
    assert ready["exception"]["type"] == "ReservationDeadlineExpired"
    assert final["hard_timeout_reached"] is True
    assert final["child_pid"] is None


def test_watchdog_spawn_exception_writes_ready_and_final_once(tmp_path: Path) -> None:
    ready_read, ready_write = os.pipe()
    final_read, final_write = os.pipe()
    pin_fd, pin_lease, pin_spec = _watchdog_test_pin(tmp_path)
    spec = {
        "wrapper_pid": os.getpid(),
        "ready_fd": ready_write,
        "final_fd": final_write,
        "command": [str(tmp_path / "definitely-missing-command")],
        "workload_deadline_ns": time.monotonic_ns() + 1_000_000_000,
        "termination_grace_ns": 20_000_000,
        "accounting_margin_ns": 20_000_000,
        "protected_directory_pins": pin_spec,
    }
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--internal-watchdog-json",
            budget.canonical_json_bytes(spec).decode("utf-8"),
        ],
        pass_fds=(ready_write, final_write, pin_fd),
        close_fds=True,
    )
    os.close(ready_write)
    os.close(final_write)
    try:
        assert process.wait(timeout=2) == 71
        ready = admitted._read_pipe_json(ready_read, timeout_seconds=1)
        final = admitted._read_pipe_json(final_read, timeout_seconds=1)
    finally:
        os.close(ready_read)
        os.close(final_read)
        budget.release_directory_generation_fence(pin_lease)
        os.close(pin_fd)
    assert ready["exception"]["type"] == "FileNotFoundError"
    assert final["exception"]["type"] == "FileNotFoundError"
    assert final["return_code"] == 127


def test_watchdog_revalidates_inherited_directory_pins_before_workload_spawn(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "protected"
    moved = tmp_path / "moved-protected"
    marker = tmp_path / "must-not-spawn"
    protected.mkdir()
    ready_read, ready_write = os.pipe()
    final_read, final_write = os.pipe()
    pin_fd, pin_lease, pin_spec = _watchdog_test_pin(protected)
    protected.rename(moved)
    protected.mkdir()
    spec = {
        "wrapper_pid": os.getpid(),
        "ready_fd": ready_write,
        "final_fd": final_write,
        "command": [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('bad')",
            str(marker),
        ],
        "workload_deadline_ns": time.monotonic_ns() + 1_000_000_000,
        "termination_grace_ns": 20_000_000,
        "accounting_margin_ns": 20_000_000,
        "protected_directory_pins": pin_spec,
    }
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--internal-watchdog-json",
            budget.canonical_json_bytes(spec).decode("utf-8"),
        ],
        pass_fds=(ready_write, final_write, pin_fd),
        close_fds=True,
    )
    os.close(ready_write)
    os.close(final_write)
    try:
        assert process.wait(timeout=2) == 71
        ready = admitted._read_pipe_json(ready_read, timeout_seconds=1)
        final = admitted._read_pipe_json(final_read, timeout_seconds=1)
        assert ready["child_pid"] is None
        assert ready["exception"]["type"] in {"LedgerError", "RuntimeError"}
        assert final["containment_empty"] is True
        assert not marker.exists()
    finally:
        os.close(ready_read)
        os.close(final_read)
        budget.release_directory_generation_fence(pin_lease)
        os.close(pin_fd)
        protected.rmdir()
        moved.rename(protected)


def test_dead_watchdog_cleans_term_ignoring_session_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    monkeypatch.setattr(budget, "ACCOUNTING_MARGIN_NS", 200_000_000)
    ready = tmp_path / "session.ready"
    grandchild_ready = tmp_path / "descendant.ready"
    escaped = tmp_path / "descendant.escaped"
    lock = tmp_path / "gpu.lock"
    execution = tmp_path / "gpu.jsonl"
    usage = tmp_path / "usage.jsonl"
    result = tmp_path / "result.json"
    child_code = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        "grand='import signal,sys,time; from pathlib import Path; '"
        "+'signal.signal(signal.SIGTERM,signal.SIG_IGN); '"
        "+'Path(sys.argv[2]).write_text(\"ready\"); time.sleep(1); '"
        "+'Path(sys.argv[1]).write_text(\"escaped\")'; "
        "subprocess.Popen([sys.executable,'-c',grand,sys.argv[2],sys.argv[3]]); "
        "deadline=time.monotonic()+1; "
        "\nwhile not Path(sys.argv[3]).exists() and time.monotonic()<deadline: time.sleep(.005)\n"
        "Path(sys.argv[1]).write_text(str(os.getsid(0))); time.sleep(2)"
    )
    harness = (
        "import importlib.util,sys; from pathlib import Path; "
        f"s=importlib.util.spec_from_file_location('w', {str(SCRIPT)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "m.budget.HEARTBEAT_INTERVAL_NS=20000000; "
        "m.budget.TERMINATION_GRACE_NS=100000000; "
        "m.budget.ACCOUNTING_MARGIN_NS=200000000; "
        "m.budget.RECOVERY_MARGIN_NS=30000000; "
        "cmd=[sys.executable,'-c',sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10]]; "
        "\ntry: m.run(Path(sys.argv[1]),Path(sys.argv[2]),cmd,"
        "usage_ledger=Path(sys.argv[3]),result_file=Path(sys.argv[4]),"
        "campaign_id='test-campaign',phase='discovery',"
        "context={'unit':'dead-watchdog'},invocation_sha256='b'*64,"
        "budget_ns=800000000)\nexcept BaseException: raise SystemExit(73)"
    )
    wrapper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            harness,
            str(lock),
            str(execution),
            str(usage),
            str(result),
            "unused5",
            "unused6",
            child_code,
            str(ready),
            str(escaped),
            str(grandchild_ready),
        ]
    )
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.01)
    assert ready.exists()
    session_id = int(ready.read_text())
    os.kill(session_id, signal.SIGKILL)
    assert wrapper.wait(timeout=4) == 73
    receipt = budget.load_validate_terminal_result(
        result, usage_ledger=usage, budget_ns=800_000_000
    )
    assert receipt["wrapper_exit_code"] == 125
    assert receipt["termination_escalated"] is True
    assert receipt["containment_anomaly"] is True
    assert receipt["reusable_success"] is False
    assert admitted._session_members(session_id) == ()
    time.sleep(1.05)
    assert not escaped.exists()


def test_setpgid_descendant_cannot_escape_workload_session_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    ready = tmp_path / "setpgid.ready"
    escaped = tmp_path / "setpgid.escaped"
    grandchild = (
        "import signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(1); "
        "Path(sys.argv[1]).write_text('escaped')"
    )
    child = (
        "import os,signal,subprocess,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "subprocess.Popen([sys.executable,'-c',sys.argv[3],sys.argv[2]],"
        "preexec_fn=lambda: os.setpgid(0,0)); "
        "Path(sys.argv[1]).write_text('ready'); time.sleep(2)"
    )
    arguments = _v7_arguments(tmp_path, name="setpgid")
    arguments["command"] = [
        sys.executable,
        "-c",
        child,
        str(ready),
        str(escaped),
        grandchild,
    ]
    arguments["budget_ns"] = 350_000_000
    assert admitted.run(**arguments) == 124
    assert ready.exists()
    time.sleep(1.05)
    assert not escaped.exists()
    state = budget.require_closed_ledger(
        arguments["usage_ledger"], budget_ns=350_000_000
    )
    terminal = state.records[-1]
    assert terminal["hard_timeout_reached"] is True
    assert terminal["termination_escalated"] is True
    assert terminal["reuse_eligible"] is False


def test_protected_directory_rename_recreate_fails_before_spawn_or_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    arguments = _v7_arguments(protected, name="directory-reset")
    usage = arguments["usage_ledger"]
    assert isinstance(usage, Path)
    template = {
        "lifecycle_id": "exhausted-before-reset",
        "campaign_id": "test-campaign",
        "phase": "discovery",
        "context": {"unit": "exhausted"},
        "invocation_sha256": "c" * 64,
        "command_sha256": "d" * 64,
        "result_path": str((protected / "old.result.json").resolve()),
        "gpu_execution_ledger_path": str((protected / "old.gpu.jsonl").resolve()),
        "boot_id": "prior-boot",
        "wrapper_pid": 999_999_999,
        "wrapper_start_ticks": 1,
        "realtime_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
    }
    budget.reconcile_and_reserve(usage, template)
    budget.reconcile_open_reservations(
        usage,
        realtime_ns=time.time_ns(),
        monotonic_ns=time.monotonic_ns(),
        current_boot_id=budget.boot_id(),
    )
    assert budget.require_closed_ledger(usage).remaining_ns == 0
    moved = tmp_path / "moved-protected"
    original = admitted._legacy_expectation
    renamed = False

    def rename_after_read(path: Path, **kwargs: object) -> str | None:
        nonlocal renamed
        value = original(path, **kwargs)
        protected.rename(moved)
        protected.mkdir()
        renamed = True
        return value

    spawned = False

    def forbidden_popen(*_args: object, **_kwargs: object) -> object:
        nonlocal spawned
        spawned = True
        raise AssertionError("directory drift must fail before child spawn")

    monkeypatch.setattr(admitted, "_legacy_expectation", rename_after_read)
    monkeypatch.setattr(admitted.subprocess, "Popen", forbidden_popen)
    try:
        with pytest.raises((RuntimeError, budget.LedgerError), match="identity"):
            admitted.run(**arguments)
        assert renamed is True
        assert spawned is False
        assert not (protected / "usage.jsonl").exists()
        assert not (protected / "directory-reset.result.json").exists()
    finally:
        protected.rmdir()
        moved.rename(protected)


def test_trusted_root_replacement_during_initial_pin_fails_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protected = tmp_path / "protected"
    moved = tmp_path / "moved-protected"
    protected.mkdir()
    sentinel = protected / "prior-budget-bytes"
    sentinel.write_text("preserve")
    arguments = _v7_arguments(protected, name="root-pin-race")
    original_open = budget.open_pinned_directory
    replaced = False

    def replace_before_open(path: Path, **kwargs: object) -> int:
        nonlocal replaced
        if Path(path) == protected and not replaced:
            protected.rename(moved)
            protected.mkdir()
            replaced = True
        return original_open(path, **kwargs)

    def forbidden_popen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("trusted-root drift must fail before child spawn")

    monkeypatch.setattr(budget, "open_pinned_directory", replace_before_open)
    monkeypatch.setattr(admitted.subprocess, "Popen", forbidden_popen)
    try:
        with pytest.raises(RuntimeError, match="trusted root changed"):
            admitted.run(**arguments)
        assert replaced is True
        assert (moved / sentinel.name).read_text() == "preserve"
        assert not (protected / "usage.jsonl").exists()
    finally:
        protected.rmdir()
        moved.rename(protected)


def test_existing_protected_component_is_stat_open_identity_sandwiched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    child = root / "child"
    moved = root / "moved-child"
    child.mkdir(parents=True)
    (child / "sentinel").write_text("old")
    root_fd = budget.open_pinned_directory(root)
    lease = budget.acquire_directory_generation_fence(root_fd)
    original_open = admitted.os.open
    replaced = False

    def replace_between_stat_and_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if path == "child" and not replaced:
            child.rename(moved)
            child.mkdir()
            replaced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(admitted.os, "open", replace_between_stat_and_open)
    try:
        with pytest.raises(RuntimeError, match="changed while being pinned"):
            admitted._open_or_create_pinned_directory(root, root_fd, child)
        assert replaced is True
        assert (moved / "sentinel").read_text() == "old"
    finally:
        budget.release_directory_generation_fence(lease)
        os.close(root_fd)
        if replaced:
            child.rmdir()
            moved.rename(child)


def test_stable_admission_generation_drift_blocks_result_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _v7_arguments(tmp_path, name="admission-drift")
    lock = arguments["lock_file"]
    result = arguments["result_file"]
    assert isinstance(lock, Path) and isinstance(result, Path)
    original_publish = admitted._publish_result
    replaced = False

    def replace_before_publish(**kwargs: object) -> dict[str, object]:
        nonlocal replaced
        if not replaced:
            lock.unlink()
            lock.touch()
            replaced = True
        return original_publish(**kwargs)

    monkeypatch.setattr(admitted, "_publish_result", replace_before_publish)
    with pytest.raises((RuntimeError, budget.LedgerError), match="generation changed"):
        admitted.run(**arguments)
    assert replaced is True
    assert not result.exists()


def test_usage_lock_generation_replacement_cannot_lose_successful_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "usage.jsonl"
    budget.append_record(
        ledger, {"schema_version": 1, "event": "initial", "elapsed_seconds": 0}
    )
    original_replace = budget._atomic_replace_bytes
    entered = threading.Event()
    resume = threading.Event()
    second_done = threading.Event()
    outcomes: dict[str, object] = {}

    def paused_replace(path: Path, payload: bytes, **kwargs: object) -> None:
        if threading.current_thread().name == "usage-w1":
            entered.set()
            assert resume.wait(timeout=2)
        original_replace(path, payload, **kwargs)

    monkeypatch.setattr(budget, "_atomic_replace_bytes", paused_replace)

    def first() -> None:
        try:
            budget.append_record(
                ledger, {"schema_version": 1, "event": "w1", "elapsed_seconds": 0}
            )
        except BaseException as error:
            outcomes["w1"] = error

    def second() -> None:
        outcomes["w2"] = budget.append_record(
            ledger, {"schema_version": 1, "event": "w2", "elapsed_seconds": 0}
        )
        second_done.set()

    w1 = threading.Thread(target=first, name="usage-w1")
    w1.start()
    assert entered.wait(timeout=2)
    lock = budget.ledger_lock_path(ledger)
    lock.unlink()
    lock.touch()
    w2 = threading.Thread(target=second, name="usage-w2")
    w2.start()
    time.sleep(0.05)
    assert not second_done.is_set()
    resume.set()
    w1.join(timeout=2)
    w2.join(timeout=2)
    assert second_done.is_set()
    assert isinstance(outcomes.get("w1"), budget.LedgerError)
    events = [row["event"] for row in budget.verify_ledger(ledger).records]
    assert events == ["initial", "w1", "w2"]


def test_raw_pinned_fd_cannot_borrow_another_transactions_directory_fence(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "usage.jsonl"
    budget.append_record(
        ledger, {"schema_version": 1, "event": "initial", "elapsed_seconds": 0}
    )
    holder_fd = budget.open_pinned_directory(tmp_path)
    raw_fd = budget.open_pinned_directory(tmp_path)
    lease = budget.acquire_directory_generation_fence(holder_fd)
    try:
        with pytest.raises(budget.LedgerError, match="active generation lease"):
            budget.append_record(
                ledger,
                {"schema_version": 1, "event": "lost", "elapsed_seconds": 0},
                pinned_directory_fd=raw_fd,
            )
    finally:
        budget.release_directory_generation_fence(lease)
        os.close(raw_fd)
        os.close(holder_fd)
    assert [row["event"] for row in budget.verify_ledger(ledger).records] == [
        "initial"
    ]


def test_same_active_lease_serializes_full_mutation_transactions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "same-lease.jsonl"
    budget.append_record(
        ledger, {"schema_version": 1, "event": "initial", "elapsed_seconds": 0}
    )
    directory_fd = budget.open_pinned_directory(tmp_path)
    lease = budget.acquire_directory_generation_fence(directory_fd)
    original_replace = budget._atomic_replace_bytes
    entered = threading.Event()
    resume = threading.Event()
    second_done = threading.Event()
    outcomes: dict[str, object] = {}

    def paused_replace(path: Path, payload: bytes, **kwargs: object) -> None:
        if threading.current_thread().name == "same-lease-w1":
            entered.set()
            assert resume.wait(timeout=2)
        original_replace(path, payload, **kwargs)

    monkeypatch.setattr(budget, "_atomic_replace_bytes", paused_replace)

    def append(name: str) -> None:
        try:
            outcomes[name] = budget.append_record(
                ledger,
                {"schema_version": 1, "event": name, "elapsed_seconds": 0},
                pinned_directory_fd=lease,
            )
        except BaseException as error:
            outcomes[name] = error
        if name == "w2":
            second_done.set()

    first = threading.Thread(target=append, args=("w1",), name="same-lease-w1")
    first.start()
    assert entered.wait(timeout=2)
    lock = budget.ledger_lock_path(ledger)
    lock.unlink()
    lock.touch()
    second = threading.Thread(target=append, args=("w2",), name="same-lease-w2")
    second.start()
    time.sleep(0.05)
    assert not second_done.is_set()
    resume.set()
    first.join(timeout=2)
    second.join(timeout=2)
    try:
        assert isinstance(outcomes.get("w1"), budget.LedgerError)
        assert isinstance(outcomes.get("w2"), dict)
        assert [row["event"] for row in budget.verify_ledger(ledger).records] == [
            "initial",
            "w1",
            "w2",
        ]
    finally:
        budget.release_directory_generation_fence(lease)
        os.close(directory_fd)


def test_directory_generation_lease_fork_copy_cannot_mutate(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "fork-lease.jsonl"
    budget.append_record(
        ledger, {"schema_version": 1, "event": "initial", "elapsed_seconds": 0}
    )
    directory_fd = budget.open_pinned_directory(tmp_path)
    lease = budget.acquire_directory_generation_fence(directory_fd)
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        try:
            budget.append_record(
                ledger,
                {"schema_version": 1, "event": "forked", "elapsed_seconds": 0},
                pinned_directory_fd=lease,
            )
        except BaseException as error:
            os.write(write_fd, str(error).encode("utf-8"))
            os._exit(0)
        os._exit(2)
    os.close(write_fd)
    try:
        _, status = os.waitpid(child_pid, 0)
        message = os.read(read_fd, 4096).decode("utf-8")
        assert os.waitstatus_to_exitcode(status) == 0
        assert "fork boundary" in message
    finally:
        os.close(read_fd)
        budget.release_directory_generation_fence(lease)
        os.close(directory_fd)
    assert [row["event"] for row in budget.verify_ledger(ledger).records] == [
        "initial"
    ]


def test_execution_and_admission_lock_generations_are_fenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = tmp_path / "gpu.jsonl"
    admitted.append_ledger(
        execution, {"schema_version": 1, "job_id": "one", "event": "start"}
    )
    original_replace = budget._atomic_replace_bytes
    entered = threading.Event()
    resume = threading.Event()
    second_done = threading.Event()
    outcomes: dict[str, object] = {}

    def paused_replace(path: Path, payload: bytes, **kwargs: object) -> None:
        if threading.current_thread().name == "execution-w1":
            entered.set()
            assert resume.wait(timeout=2)
        original_replace(path, payload, **kwargs)

    monkeypatch.setattr(budget, "_atomic_replace_bytes", paused_replace)

    def first_execution() -> None:
        try:
            admitted.append_ledger(
                execution,
                {"schema_version": 1, "job_id": "one", "event": "end"},
            )
        except BaseException as error:
            outcomes["execution_w1"] = error

    def second_execution() -> None:
        admitted.append_ledger(
            execution,
            {"schema_version": 1, "job_id": "two", "event": "start"},
        )
        second_done.set()

    w1 = threading.Thread(target=first_execution, name="execution-w1")
    w1.start()
    assert entered.wait(timeout=2)
    execution_lock = admitted.execution_ledger_lock_path(execution)
    execution_lock.unlink()
    execution_lock.touch()
    w2 = threading.Thread(target=second_execution, name="execution-w2")
    w2.start()
    time.sleep(0.05)
    assert not second_done.is_set()
    resume.set()
    w1.join(timeout=2)
    w2.join(timeout=2)
    assert second_done.is_set()
    assert isinstance(outcomes.get("execution_w1"), budget.LedgerError)
    assert [
        row["event"] for row in admitted._decode_execution_ledger(execution.read_bytes())
    ] == ["start", "end", "start"]

    admission = tmp_path / "admission.lock"
    admitted_entered = threading.Event()
    admitted_resume = threading.Event()
    admitted_second = threading.Event()
    admission_errors: list[BaseException] = []

    def first_admission() -> None:
        try:
            with admitted._exclusive_gpu_lock(admission):
                admitted_entered.set()
                assert admitted_resume.wait(timeout=2)
        except BaseException as error:
            admission_errors.append(error)

    def second_admission() -> None:
        with admitted._exclusive_gpu_lock(admission):
            admitted_second.set()

    a1 = threading.Thread(target=first_admission)
    a1.start()
    assert admitted_entered.wait(timeout=2)
    admission.unlink()
    admission.touch()
    a2 = threading.Thread(target=second_admission)
    a2.start()
    time.sleep(0.05)
    assert not admitted_second.is_set()
    admitted_resume.set()
    a1.join(timeout=2)
    a2.join(timeout=2)
    assert admitted_second.is_set()
    assert admission_errors and "inode changed" in str(admission_errors[0])


def test_arbitrary_torn_execution_tail_remains_explicitly_fail_closed(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "gpu.jsonl"
    ledger.write_bytes(b'{"event":"end"')
    before = ledger.read_bytes()
    with pytest.raises(budget.LedgerError, match="torn tail"):
        admitted.append_ledger(
            ledger, {"schema_version": 1, "job_id": "x", "event": "start"}
        )
    assert ledger.read_bytes() == before


def test_execution_end_terminal_field_drift_blocks_result_recovery(
    tmp_path: Path,
) -> None:
    arguments = _v7_arguments(tmp_path, name="end-drift")
    assert admitted.run(**arguments) == 0
    arguments["result_file"].unlink()
    rows = [json.loads(line) for line in arguments["ledger"].read_text().splitlines()]
    rows[-1]["hard_timeout_reached"] = True
    arguments["ledger"].write_bytes(
        b"".join(budget.canonical_json_bytes(row) + b"\n" for row in rows)
    )
    with pytest.raises(budget.LedgerError, match="terminal-derived field drifted"):
        admitted.run(**arguments)
    assert not arguments["result_file"].exists()


def test_concurrent_different_result_receipts_cannot_both_succeed(
    tmp_path: Path,
) -> None:
    result = tmp_path / "receipt.json"
    barrier = threading.Barrier(2)

    def publish(value: int) -> tuple[str, object]:
        barrier.wait()
        try:
            return "ok", budget.atomic_result_receipt(result, {"value": value})
        except BaseException as error:
            return "error", error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (1, 2)))
    assert [status for status, _ in outcomes].count("ok") == 1
    assert [status for status, _ in outcomes].count("error") == 1
    error = next(value for status, value in outcomes if status == "error")
    assert isinstance(error, budget.LedgerError)
    published = json.loads(result.read_text())
    successful = next(value for status, value in outcomes if status == "ok")
    assert published == successful


def test_v8r4a_terminal_result_and_stable_lock_are_exact_immutable_files(
    tmp_path: Path,
) -> None:
    result = tmp_path / "receipt.json"
    budget.atomic_result_receipt(result, {"classification": "test", "value": 1})
    lock = budget.result_receipt_lock_path(result)
    for path in (result, lock):
        status = path.stat()
        assert stat.S_ISREG(status.st_mode)
        assert stat.S_IMODE(status.st_mode) == 0o444
        assert status.st_nlink == 1
    assert lock.stat().st_size == 0
    assert not any("create.tmp" in item.name for item in tmp_path.iterdir())


def test_v8r4a_active_mutable_result_lock_is_refused_but_explicit_legacy_is_read_only(
    tmp_path: Path,
) -> None:
    result = tmp_path / "legacy.json"
    value = {"classification": "legacy-test", "value": 2}
    budget.atomic_result_receipt(result, value)
    lock = budget.result_receipt_lock_path(result)
    before = lock.read_bytes()
    lock.chmod(0o644)
    with pytest.raises(budget.LedgerError, match="lock is aliased or non-regular"):
        budget.atomic_result_receipt(result, value)
    with budget._exclusive_result_lock(
        result, allow_legacy_mutable_evidence=True
    ) as locked:
        assert locked.read() is not None
    assert lock.read_bytes() == before == b""
    assert stat.S_IMODE(lock.stat().st_mode) == 0o644


def test_v8r4a_legacy_read_does_not_create_a_missing_lock(tmp_path: Path) -> None:
    result = tmp_path / "legacy-without-lock.json"
    result.write_bytes(b"{}\n")
    result.chmod(0o444)
    lock = budget.result_receipt_lock_path(result)
    with pytest.raises(FileNotFoundError):
        with budget._exclusive_result_lock(
            result, allow_legacy_mutable_evidence=True
        ):
            raise AssertionError("missing legacy lock must not be synthesized")
    assert not lock.exists()


def test_v8r4a_anonymous_create_failure_leaves_no_named_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "anonymous.json"

    def fail(point: str) -> None:
        if point == "anonymous_create_before_link":
            raise RuntimeError("injected anonymous prelink failure")

    monkeypatch.setattr(budget, "_FAULT_INJECTION_HOOK", fail)
    with pytest.raises(RuntimeError, match="prelink"):
        budget.atomic_create_immutable_bytes(target, b"complete\n")
    assert list(tmp_path.iterdir()) == []


def test_forced_descendant_cleanup_is_nonzero_and_never_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    escaped = tmp_path / "background.escaped"
    grandchild = (
        "import signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(1); "
        "Path(sys.argv[1]).write_text('escaped')"
    )
    child = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[1]])"
    )
    arguments = _v7_arguments(tmp_path, name="forced-cleanup")
    arguments["command"] = [
        sys.executable,
        "-c",
        child,
        str(escaped),
        grandchild,
    ]
    arguments["budget_ns"] = 1_000_000_000
    assert admitted.run(**arguments) == 125
    result = budget.load_validate_terminal_result(
        arguments["result_file"],
        usage_ledger=arguments["usage_ledger"],
        budget_ns=1_000_000_000,
    )
    assert result["termination_escalated"] is True
    assert result["containment_anomaly"] is True
    assert result["wrapper_exit_code"] == 125
    assert result["reusable_success"] is False
    time.sleep(1.05)
    assert not escaped.exists()


def test_hardlinked_result_is_rejected_at_consumption_time(tmp_path: Path) -> None:
    arguments = _v7_arguments(tmp_path, name="result-hardlink")
    assert admitted.run(**arguments) == 0
    os.link(arguments["result_file"], tmp_path / "outside-result-alias")
    with pytest.raises(budget.LedgerError, match="aliased"):
        budget.load_validate_terminal_result(
            arguments["result_file"], usage_ledger=arguments["usage_ledger"]
        )


def test_result_hardlink_inserted_during_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "result.json"
    alias = tmp_path / "mid-read-alias"
    result.write_bytes(b"{}\n")
    directory_fd = budget.open_pinned_directory(tmp_path)
    original_read = budget.os.read
    injected = False

    def link_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal injected
        payload = original_read(descriptor, size)
        if not injected:
            os.link(result, alias)
            injected = True
        return payload

    monkeypatch.setattr(budget.os, "read", link_after_first_read)
    try:
        with pytest.raises(budget.LedgerError, match="identity or link count"):
            budget._read_regular_bytes_at(
                directory_fd, result.name, label="GPU terminal result"
            )
    finally:
        os.close(directory_fd)
    assert injected is True
    assert result.stat().st_nlink == 2


def test_terminal_result_receipt_cannot_be_replayed_at_another_path(
    tmp_path: Path,
) -> None:
    arguments = _v7_arguments(tmp_path, name="receipt-a")
    assert admitted.run(**arguments) == 0
    result_a = arguments["result_file"]
    usage = arguments["usage_ledger"]
    assert isinstance(result_a, Path) and isinstance(usage, Path)
    result_b = tmp_path / "receipt-b.result.json"
    result_b.write_bytes(result_a.read_bytes())
    result_b.chmod(0o444)
    with pytest.raises(budget.LedgerError, match="result path binding"):
        budget.load_validate_terminal_result(result_b, usage_ledger=usage)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("hostname", "forged-host"),
        ("cwd", "/forged/cwd"),
        ("lock_file", "/forged/gpu.lock"),
        ("cuda_visible_devices", "99"),
        ("command", [sys.executable, "-c", "raise SystemExit(9)"]),
        ("context", {"unit": "forged-context"}),
    ],
)
def test_existing_result_fast_path_validates_authoritative_execution_identity(
    tmp_path: Path, field: str, replacement: object
) -> None:
    arguments = _v7_arguments(tmp_path, name=f"existing-{field}")
    assert admitted.run(**arguments) == 0
    execution = arguments["ledger"]
    assert isinstance(execution, Path)
    rows = [json.loads(line) for line in execution.read_text().splitlines()]
    rows[0][field] = replacement
    rows[1][field] = replacement
    execution.write_bytes(
        b"".join(budget.canonical_json_bytes(row) + b"\n" for row in rows)
    )
    with pytest.raises(budget.LedgerError, match="authoritative reservation"):
        admitted.run(**arguments)


def test_result_retry_after_post_rename_directory_fsync_failure_is_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "durable-retry.json"
    value = {"classification": "test-receipt", "value": 7}
    budget.atomic_create_immutable_bytes(
        budget.result_receipt_lock_path(result), b""
    )
    original_fsync = budget.os.fsync
    failed_directory_fsync = False
    successful_directory_fsyncs = 0

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal failed_directory_fsync, successful_directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            if not failed_directory_fsync:
                failed_directory_fsync = True
                raise OSError("injected post-rename directory fsync failure")
            successful_directory_fsyncs += 1
        original_fsync(descriptor)

    monkeypatch.setattr(budget.os, "fsync", fail_first_directory_fsync)
    with pytest.raises(OSError, match="post-rename"):
        budget.atomic_result_receipt(result, value)
    assert result.exists()
    recovered = budget.atomic_result_receipt(result, value)
    assert recovered["value"] == 7
    assert successful_directory_fsyncs >= 1


def test_result_inode_replacement_after_read_is_rejected_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _v7_arguments(tmp_path, name="result-generation")
    assert admitted.run(**arguments) == 0
    result = arguments["result_file"]
    usage = arguments["usage_ledger"]
    assert isinstance(result, Path) and isinstance(usage, Path)
    replacement = tmp_path / "replacement-result"
    replacement.write_bytes(result.read_bytes())
    replacement.chmod(0o444)
    original_read = budget._read_regular_bytes_identity_at
    swapped = False

    def replace_after_read(*args: object, **kwargs: object) -> object:
        nonlocal swapped
        value = original_read(*args, **kwargs)
        if not swapped and value is not None:
            os.replace(replacement, result)
            swapped = True
        return value

    monkeypatch.setattr(
        budget, "_read_regular_bytes_identity_at", replace_after_read
    )
    with pytest.raises(budget.LedgerError, match="generation changed"):
        budget.load_validate_terminal_result(result, usage_ledger=usage)
    assert swapped is True


def test_directory_fence_acquire_failure_leaks_no_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "acquire-failure.jsonl"
    before = len(os.listdir("/proc/self/fd"))
    original_flock = budget.fcntl.flock
    injected = False

    def fail_directory_acquire(descriptor: int, operation: int) -> None:
        nonlocal injected
        if (
            not injected
            and operation == fcntl.LOCK_EX
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
        ):
            injected = True
            raise OSError("injected directory fence acquire failure")
        original_flock(descriptor, operation)

    monkeypatch.setattr(budget.fcntl, "flock", fail_directory_acquire)
    with pytest.raises(budget.LedgerError, match="cannot acquire.*fence"):
        budget.append_record(
            ledger,
            {"schema_version": 1, "event": "never", "elapsed_seconds": 0},
        )
    assert injected is True
    assert len(os.listdir("/proc/self/fd")) == before
    assert not ledger.exists()


def test_directory_fence_unlock_failure_still_releases_lock_and_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "unlock-failure.jsonl"
    before = len(os.listdir("/proc/self/fd"))
    original_flock = budget.fcntl.flock
    injected = False

    def fail_directory_unlock(descriptor: int, operation: int) -> None:
        nonlocal injected
        if (
            not injected
            and operation == fcntl.LOCK_UN
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
        ):
            injected = True
            raise OSError("injected directory fence unlock failure")
        original_flock(descriptor, operation)

    monkeypatch.setattr(budget.fcntl, "flock", fail_directory_unlock)
    with pytest.raises(budget.LedgerError, match="fence cleanup failed"):
        budget.append_record(
            ledger,
            {"schema_version": 1, "event": "committed", "elapsed_seconds": 0},
        )
    assert injected is True
    assert len(os.listdir("/proc/self/fd")) == before
    state = budget.verify_ledger(ledger)
    assert [record["event"] for record in state.records] == ["committed"]


def test_efficiency_phase_issues_one_shot_child_binding_and_preserves_prefixes(
    tmp_path: Path,
) -> None:
    authorization, authorization_sha256 = _immutable_authorization(tmp_path)
    output = tmp_path / "verified-binding.json"
    output.write_text("", encoding="utf-8")
    arguments = _v7_arguments(tmp_path, name="efficiency")
    arguments.update(
        {
            "phase": "efficiency_benchmark",
            "context": {
                "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
                "outer_fold": 3,
                "seed": 20260828,
                "variant": "H0_no_factor",
            },
            "authorization_path": authorization,
            "authorization_sha256": authorization_sha256,
        }
    )
    arguments["command"] = _binding_consumer_command(
        lock_file=arguments["lock_file"],
        usage_ledger=arguments["usage_ledger"],
        execution_ledger=arguments["ledger"],
        authorization=authorization,
        authorization_sha256=authorization_sha256,
        output=output,
        consume_twice=True,
    )

    assert admitted.run(**arguments) == 0
    binding = json.loads(output.read_text(encoding="utf-8"))
    assert binding["classification"] == "verified_v8_gpu_admitted_child_lifecycle"
    assert set(binding) == admitted._VERIFIED_ADMITTED_CHILD_BINDING_KEYS
    assert len(binding["nonce"]) == 64
    assert binding["binding_sha256"] == admitted._verified_admitted_child_digest(
        binding
    )
    assert binding["phase"] == "efficiency_benchmark"
    assert binding["invocation_sha256"] == "a" * 64
    assert binding["authorization"] == {
        "path": str(authorization.resolve()),
        "sha256": authorization_sha256,
        "bytes": authorization.stat().st_size,
        "mode": "0444",
    }
    usage_raw = arguments["usage_ledger"].read_bytes()
    usage_prefix = usage_raw[: binding["usage_ledger_prefix_bytes"]]
    assert hashlib.sha256(usage_prefix).hexdigest() == binding[
        "usage_ledger_prefix_sha256"
    ]
    assert len(usage_raw) > len(usage_prefix)
    execution_raw = arguments["ledger"].read_bytes()
    execution_prefix = execution_raw[: binding["execution_ledger_prefix_bytes"]]
    assert hashlib.sha256(execution_prefix).hexdigest() == binding[
        "execution_ledger_prefix_sha256"
    ]
    assert len(execution_raw) > len(execution_prefix)
    state = budget.require_closed_ledger(arguments["usage_ledger"])
    reservation = state.records[0]
    assert reservation["admitted_child_authorization"] == binding["authorization"]


def test_efficiency_phase_fails_before_reservation_without_immutable_authorization(
    tmp_path: Path,
) -> None:
    arguments = _v7_arguments(tmp_path, name="missing-authorization")
    arguments["phase"] = "efficiency_benchmark"
    with pytest.raises(ValueError, match="requires an immutable"):
        admitted.run(**arguments)
    assert not arguments["usage_ledger"].exists()
    assert not arguments["ledger"].exists()


def test_efficiency_phase_rejects_mutable_or_hash_drifted_authorization(
    tmp_path: Path,
) -> None:
    authorization, authorization_sha256 = _immutable_authorization(tmp_path)
    authorization.chmod(0o644)
    arguments = _v7_arguments(tmp_path, name="mutable-authorization")
    arguments.update(
        {
            "phase": "efficiency_benchmark",
            "authorization_path": authorization,
            "authorization_sha256": authorization_sha256,
        }
    )
    with pytest.raises(RuntimeError, match="exact mode 0444"):
        admitted.run(**arguments)
    authorization.chmod(0o444)
    arguments["authorization_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="SHA-256 drifted"):
        admitted.run(**arguments)
    assert not arguments["usage_ledger"].exists()


def test_direct_child_launch_has_no_forgeable_command_line_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization, authorization_sha256 = _immutable_authorization(tmp_path)
    monkeypatch.delenv(admitted._ADMITTED_CHILD_FD_ENV, raising=False)
    with pytest.raises(RuntimeError, match="descriptor is absent"):
        admitted.consume_admitted_child_binding(
            "efficiency_benchmark",
            tmp_path / "usage.jsonl",
            tmp_path / "execution.jsonl",
            authorization,
            authorization_sha256,
            expected_campaign_id="test-campaign",
            expected_gpu_lock_file=tmp_path / "gpu.lock",
        )


def test_wrong_first_consume_burns_pipe_and_lock_capability(tmp_path: Path) -> None:
    authorization, authorization_sha256 = _immutable_authorization(tmp_path)
    arguments = _v7_arguments(tmp_path, name="wrong-first-consume")
    arguments.update(
        {
            "phase": "efficiency_benchmark",
            "context": {"unit": "wrong-first-consume"},
            "authorization_path": authorization,
            "authorization_sha256": authorization_sha256,
        }
    )
    code = "\n".join(
        [
            "import importlib.util,pathlib,sys",
            "sys.dont_write_bytecode=True",
            f"spec=importlib.util.spec_from_file_location('admitted_child',{str(SCRIPT)!r})",
            "module=importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "try:",
            (
                " module.consume_admitted_child_binding("
                f"'discovery',pathlib.Path({str(arguments['usage_ledger'])!r}),"
                f"pathlib.Path({str(arguments['ledger'])!r}),"
                f"pathlib.Path({str(authorization)!r}),{authorization_sha256!r},"
                "expected_campaign_id='test-campaign',"
                f"expected_gpu_lock_file=pathlib.Path({str(arguments['lock_file'])!r}))"
            ),
            "except RuntimeError as error:",
            " assert 'identity is invalid' in str(error)",
            "else:",
            " raise AssertionError('wrong first phase was accepted')",
            "try:",
            (
                " module.consume_admitted_child_binding("
                f"'efficiency_benchmark',pathlib.Path({str(arguments['usage_ledger'])!r}),"
                f"pathlib.Path({str(arguments['ledger'])!r}),"
                f"pathlib.Path({str(authorization)!r}),{authorization_sha256!r},"
                "expected_campaign_id='test-campaign',"
                f"expected_gpu_lock_file=pathlib.Path({str(arguments['lock_file'])!r}))"
            ),
            "except RuntimeError as error:",
            " assert 'descriptor is absent' in str(error)",
            "else:",
            " raise AssertionError('burned capability was reused')",
        ]
    )
    arguments["command"] = [sys.executable, "-c", code]
    assert admitted.run(**arguments) == 0


@pytest.mark.parametrize("foreign_campaign", (True, False))
def test_consumer_rejects_foreign_campaign_or_alternate_lock(
    tmp_path: Path, foreign_campaign: bool
) -> None:
    authorization, authorization_sha256 = _immutable_authorization(tmp_path)
    arguments = _v7_arguments(tmp_path, name="foreign-scope")
    arguments.update(
        {
            "phase": "efficiency_benchmark",
            "context": {"unit": "foreign-scope"},
            "authorization_path": authorization,
            "authorization_sha256": authorization_sha256,
        }
    )
    alternate_lock = tmp_path / "alternate-gpu.lock"
    alternate_lock.touch()
    expected_campaign = "foreign-campaign" if foreign_campaign else "test-campaign"
    expected_lock = arguments["lock_file"] if foreign_campaign else alternate_lock
    code = "\n".join(
        [
            "import importlib.util,pathlib,sys",
            "sys.dont_write_bytecode=True",
            f"spec=importlib.util.spec_from_file_location('admitted_child',{str(SCRIPT)!r})",
            "module=importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "try:",
            (
                " module.consume_admitted_child_binding("
                f"'efficiency_benchmark',pathlib.Path({str(arguments['usage_ledger'])!r}),"
                f"pathlib.Path({str(arguments['ledger'])!r}),"
                f"pathlib.Path({str(authorization)!r}),{authorization_sha256!r},"
                f"expected_campaign_id={expected_campaign!r},"
                f"expected_gpu_lock_file=pathlib.Path({str(expected_lock)!r}))"
            ),
            "except RuntimeError as error:",
            " assert 'identity is invalid' in str(error)",
            "else:",
            " raise AssertionError('foreign binding scope was accepted')",
        ]
    )
    arguments["command"] = [sys.executable, "-c", code]
    assert admitted.run(**arguments) == 0


def test_revalidator_rejects_fabricated_live_mapping(tmp_path: Path) -> None:
    authorization, authorization_sha256 = _immutable_authorization(tmp_path)
    arguments = _v7_arguments(tmp_path, name="fabricated-live")
    arguments.update(
        {
            "phase": "efficiency_benchmark",
            "context": {"unit": "fabricated-live"},
            "authorization_path": authorization,
            "authorization_sha256": authorization_sha256,
        }
    )
    code = "\n".join(
        [
            "import importlib.util,pathlib,sys",
            "sys.dont_write_bytecode=True",
            f"spec=importlib.util.spec_from_file_location('admitted_child',{str(SCRIPT)!r})",
            "module=importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            (
                "binding=module.consume_admitted_child_binding("
                f"'efficiency_benchmark',pathlib.Path({str(arguments['usage_ledger'])!r}),"
                f"pathlib.Path({str(arguments['ledger'])!r}),"
                f"pathlib.Path({str(authorization)!r}),{authorization_sha256!r},"
                "expected_campaign_id='test-campaign',"
                f"expected_gpu_lock_file=pathlib.Path({str(arguments['lock_file'])!r}))"
            ),
            "forged=dict(binding)",
            "forged['lifecycle_id']='fabricated-live-lifecycle'",
            "forged['binding_sha256']=module._verified_admitted_child_digest(forged)",
            "try:",
            (
                " module.revalidate_consumed_admitted_child_binding(forged,"
                "expected_campaign_id='test-campaign',"
                "expected_phase='efficiency_benchmark',"
                f"expected_gpu_lock_file=pathlib.Path({str(arguments['lock_file'])!r}),"
                f"expected_usage_ledger=pathlib.Path({str(arguments['usage_ledger'])!r}),"
                f"expected_execution_ledger=pathlib.Path({str(arguments['ledger'])!r}),"
                f"expected_authorization_path=pathlib.Path({str(authorization)!r}),"
                f"expected_authorization_sha256={authorization_sha256!r})"
            ),
            "except RuntimeError as error:",
            " assert 'unrelated open lifecycle' in str(error)",
            "else:",
            " raise AssertionError('fabricated live binding was accepted')",
            (
                "module.revalidate_consumed_admitted_child_binding(binding,"
                "expected_campaign_id='test-campaign',"
                "expected_phase='efficiency_benchmark',"
                f"expected_gpu_lock_file=pathlib.Path({str(arguments['lock_file'])!r}),"
                f"expected_usage_ledger=pathlib.Path({str(arguments['usage_ledger'])!r}),"
                f"expected_execution_ledger=pathlib.Path({str(arguments['ledger'])!r}),"
                f"expected_authorization_path=pathlib.Path({str(authorization)!r}),"
                f"expected_authorization_sha256={authorization_sha256!r})"
            ),
        ]
    )
    arguments["command"] = [sys.executable, "-c", code]
    assert admitted.run(**arguments) == 0


def test_revalidator_rejects_binding_after_ledger_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization, authorization_sha256 = _immutable_authorization(tmp_path)
    output = tmp_path / "closed-binding.json"
    arguments = _v7_arguments(tmp_path, name="closed-binding")
    arguments.update(
        {
            "phase": "efficiency_benchmark",
            "context": {"unit": "closed-binding"},
            "authorization_path": authorization,
            "authorization_sha256": authorization_sha256,
        }
    )
    arguments["command"] = _binding_consumer_command(
        lock_file=arguments["lock_file"],
        usage_ledger=arguments["usage_ledger"],
        execution_ledger=arguments["ledger"],
        authorization=authorization,
        authorization_sha256=authorization_sha256,
        output=output,
    )
    assert admitted.run(**arguments) == 0
    binding = json.loads(output.read_text(encoding="utf-8"))
    assert budget.require_closed_ledger(arguments["usage_ledger"]).open_reservations == {}
    child_pid = os.getpid()
    child_ticks = budget.process_start_ticks(child_pid)
    assert child_ticks is not None
    watchdog_pid = 320_001
    wrapper_pid = 320_002
    watchdog_ticks = 420_001
    wrapper_ticks = 420_002
    binding.update(
        {
            "child_pid": child_pid,
            "child_start_ticks": child_ticks,
            "watchdog_pid": watchdog_pid,
            "watchdog_start_ticks": watchdog_ticks,
            "wrapper_pid": wrapper_pid,
            "wrapper_start_ticks": wrapper_ticks,
            "gpu_lock_fd": 999_998,
        }
    )
    binding["binding_sha256"] = admitted._verified_admitted_child_digest(binding)
    monkeypatch.setattr(admitted.os, "getppid", lambda: watchdog_pid)
    monkeypatch.setattr(
        admitted,
        "_proc_parent_pid",
        lambda pid: wrapper_pid if pid == watchdog_pid else None,
    )
    original_ticks = budget.process_start_ticks

    def process_ticks(pid: int) -> int | None:
        synthetic = {
            child_pid: child_ticks,
            watchdog_pid: watchdog_ticks,
            wrapper_pid: wrapper_ticks,
        }
        return synthetic.get(pid, original_ticks(pid))

    monkeypatch.setattr(budget, "process_start_ticks", process_ticks)
    monkeypatch.setattr(
        admitted,
        "_verify_wrapper_admission_lock",
        lambda _reservation, *, capability_fd: None,
    )
    with pytest.raises(RuntimeError, match="unrelated open lifecycle"):
        admitted.revalidate_consumed_admitted_child_binding(
            binding,
            expected_campaign_id="test-campaign",
            expected_phase="efficiency_benchmark",
            expected_gpu_lock_file=arguments["lock_file"],
            expected_usage_ledger=arguments["usage_ledger"],
            expected_execution_ledger=arguments["ledger"],
            expected_authorization_path=authorization,
            expected_authorization_sha256=authorization_sha256,
        )


def test_wrapper_lock_proof_rejects_sibling_holder(tmp_path: Path) -> None:
    lock_file = (tmp_path / "gpu.lock").resolve()
    lock_file.touch()
    capability_fd = os.open(lock_file, os.O_RDWR)
    marker = tmp_path / "sibling-locked"
    holder_code = "\n".join(
        [
            "import fcntl,os,pathlib,sys,time",
            "descriptor=os.open(sys.argv[1],os.O_RDWR)",
            "fcntl.flock(descriptor,fcntl.LOCK_EX)",
            "pathlib.Path(sys.argv[2]).touch()",
            "time.sleep(30)",
        ]
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(lock_file), str(marker)]
    )
    try:
        deadline = time.monotonic() + 3.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        status = os.fstat(capability_fd)
        wrapper_pid = os.getpid()
        wrapper_ticks = budget.process_start_ticks(wrapper_pid)
        assert wrapper_ticks is not None
        with pytest.raises(
            budget.LedgerError, match="not owned by the exact wrapper"
        ):
            admitted._verify_wrapper_admission_lock(
                {
                    "wrapper_pid": wrapper_pid,
                    "wrapper_start_ticks": wrapper_ticks,
                    "gpu_lock_file": str(lock_file),
                    "gpu_lock_st_dev": status.st_dev,
                    "gpu_lock_st_ino": status.st_ino,
                },
                capability_fd=capability_fd,
            )
    finally:
        holder.terminate()
        holder.wait(timeout=3)
        os.close(capability_fd)


def test_reconciled_wrapper_crash_closes_stale_execution_before_retry(
    tmp_path: Path,
) -> None:
    authorization, authorization_sha256 = _immutable_authorization(tmp_path)
    lock_file = tmp_path / "gpu.lock"
    execution_ledger = tmp_path / "execution.jsonl"
    usage_ledger = tmp_path / "usage.jsonl"
    result_file = tmp_path / "result.json"
    started = tmp_path / "child-started"
    proceed = tmp_path / "proceed"
    worker_code = "\n".join(
        [
            "import importlib.util,os,pathlib,sys,time",
            "sys.dont_write_bytecode=True",
            f"spec=importlib.util.spec_from_file_location('admitted_child',{str(SCRIPT)!r})",
            "module=importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            (
                "binding=module.consume_admitted_child_binding("
                f"'efficiency_benchmark',pathlib.Path({str(usage_ledger)!r}),"
                f"pathlib.Path({str(execution_ledger)!r}),"
                f"pathlib.Path({str(authorization)!r}),{authorization_sha256!r},"
                "expected_campaign_id='test-campaign',"
                f"expected_gpu_lock_file=pathlib.Path({str(lock_file)!r}))"
            ),
            (
                "module.revalidate_consumed_admitted_child_binding(binding,"
                "expected_campaign_id='test-campaign',"
                "expected_phase='efficiency_benchmark',"
                f"expected_gpu_lock_file=pathlib.Path({str(lock_file)!r}),"
                f"expected_usage_ledger=pathlib.Path({str(usage_ledger)!r}),"
                f"expected_execution_ledger=pathlib.Path({str(execution_ledger)!r}),"
                f"expected_authorization_path=pathlib.Path({str(authorization)!r}),"
                f"expected_authorization_sha256={authorization_sha256!r})"
            ),
            f"pathlib.Path({str(started)!r}).write_text(str(os.getpid()))",
            f"time.sleep(30) if not pathlib.Path({str(proceed)!r}).exists() else None",
        ]
    )
    command = _budgeted_cli_command(
        lock_file=lock_file,
        execution_ledger=execution_ledger,
        usage_ledger=usage_ledger,
        result_file=result_file,
        authorization=authorization,
        authorization_sha256=authorization_sha256,
        command=[sys.executable, "-c", worker_code],
    )
    first = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 8.0
    while not started.exists() and first.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert started.exists(), first.stderr.read().decode("utf-8", errors="replace")
    child_pid = int(started.read_text(encoding="utf-8"))
    os.kill(first.pid, signal.SIGKILL)
    assert first.wait(timeout=3) == -signal.SIGKILL
    deadline = time.monotonic() + 5.0
    while budget.process_start_ticks(child_pid) is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert budget.process_start_ticks(child_pid) is None
    proceed.touch()

    second = subprocess.run(command, capture_output=True, check=False, timeout=15)
    assert second.returncode == 0, second.stderr.decode("utf-8", errors="replace")
    usage_rows = [
        json.loads(line) for line in usage_ledger.read_text(encoding="utf-8").splitlines()
    ]
    execution_rows = [
        json.loads(line)
        for line in execution_ledger.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in usage_rows] == [
        "reservation",
        "reconciled_terminal",
        "reservation",
        "terminal",
    ]
    assert [row["event"] for row in execution_rows] == [
        "start",
        "end",
        "start",
        "end",
    ]
    recovered_end = execution_rows[1]
    assert recovered_end["recovered_from_durable_usage_terminal"] is True
    assert recovered_end["exit_code"] is None
    assert recovered_end["terminal_record_sha256"] == usage_rows[1]["record_sha256"]
    assert budget.require_closed_ledger(usage_ledger).open_reservations == {}
    assert admitted._open_execution_starts(execution_rows) == {}

    usage_before = usage_ledger.read_bytes()
    execution_before = execution_ledger.read_bytes()
    third = subprocess.run(command, capture_output=True, check=False, timeout=10)
    assert third.returncode == 0, third.stderr.decode("utf-8", errors="replace")
    assert usage_ledger.read_bytes() == usage_before
    assert execution_ledger.read_bytes() == execution_before


def test_stale_execution_recovery_append_failure_is_atomic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lifecycle(monkeypatch)
    usage_ledger = (tmp_path / "usage.jsonl").resolve()
    execution_ledger = (tmp_path / "execution.jsonl").resolve()
    lock_file = (tmp_path / "gpu.lock").resolve()
    lock_file.touch()
    command = [sys.executable, "-c", "raise SystemExit(0)"]
    template = {
        "lifecycle_id": "stale-execution-recovery",
        "campaign_id": "test-campaign",
        "phase": "discovery",
        "context": {"unit": "stale-recovery"},
        "invocation_sha256": "d" * 64,
        "command_sha256": budget.command_sha256(command),
        "result_path": str((tmp_path / "result.json").resolve()),
        "gpu_execution_ledger_path": str(execution_ledger),
        "boot_id": budget.boot_id(),
        "wrapper_pid": 999_999_999,
        "wrapper_start_ticks": 1,
        "wrapper_parent_pid": 1,
        "hostname": "test-host",
        "cwd": str(tmp_path.resolve()),
        "gpu_lock_file": str(lock_file),
        "cuda_visible_devices": None,
        "realtime_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
    }
    reservation, _, _ = budget.reconcile_and_reserve(
        usage_ledger, template, budget_ns=2_000_000_000
    )
    admitted.append_ledger(
        execution_ledger,
        {
            "schema_version": 1,
            "job_id": reservation["lifecycle_id"],
            "lifecycle_id": reservation["lifecycle_id"],
            "reservation_record_sha256": reservation["record_sha256"],
            "wrapper_pid": reservation["wrapper_pid"],
            "hostname": reservation["hostname"],
            "cwd": reservation["cwd"],
            "lock_file": reservation["gpu_lock_file"],
            "usage_ledger": str(usage_ledger),
            "result_file": reservation["result_path"],
            "campaign_id": reservation["campaign_id"],
            "phase": reservation["phase"],
            "context": reservation["context"],
            "invocation_sha256": reservation["invocation_sha256"],
            "command": command,
            "command_sha256": reservation["command_sha256"],
            "cuda_visible_devices": reservation["cuda_visible_devices"],
            "event": "start",
            "utc": admitted.utc_now(),
        },
    )
    _, state = budget.reconcile_open_reservations(
        usage_ledger,
        realtime_ns=time.time_ns(),
        monotonic_ns=time.monotonic_ns(),
        budget_ns=2_000_000_000,
    )
    before = execution_ledger.read_bytes()
    original_append = admitted._append_execution_locked

    def fail_recovery_append(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected stale execution recovery append failure")

    monkeypatch.setattr(admitted, "_append_execution_locked", fail_recovery_append)
    with pytest.raises(OSError, match="injected stale"):
        admitted._recover_closed_usage_execution_starts(
            execution_ledger,
            state,
            expected_lock_file=lock_file,
            usage_ledger=usage_ledger,
        )
    assert execution_ledger.read_bytes() == before
    monkeypatch.setattr(admitted, "_append_execution_locked", original_append)
    assert (
        admitted._recover_closed_usage_execution_starts(
            execution_ledger,
            state,
            expected_lock_file=lock_file,
            usage_ledger=usage_ledger,
        )
        == 1
    )
    recovered = execution_ledger.read_bytes()
    assert (
        admitted._recover_closed_usage_execution_starts(
            execution_ledger,
            state,
            expected_lock_file=lock_file,
            usage_ledger=usage_ledger,
        )
        == 0
    )
    assert execution_ledger.read_bytes() == recovered
    rows = admitted._decode_execution_ledger(recovered)
    assert [row["event"] for row in rows] == ["start", "end"]
    assert rows[-1]["usage_terminal_event"] == "reconciled_terminal"
    assert rows[-1]["recovered_from_durable_usage_terminal"] is True
    assert rows[-1]["wrapper_exit_code"] == 125
    assert rows[-1]["containment_anomaly"] is True


def test_admitted_child_rejects_unrelated_open_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization_path, authorization_sha256 = _immutable_authorization(tmp_path)
    authorization = admitted._immutable_authorization_binding(
        authorization_path, authorization_sha256
    )
    usage_ledger = (tmp_path / "usage.jsonl").resolve()
    execution_ledger = (tmp_path / "execution.jsonl").resolve()
    execution_ledger.write_bytes(b"{}\n")
    lock_file = (tmp_path / "gpu.lock").resolve()
    lock_file.touch()
    lock_status = lock_file.stat()
    wrapper_source = admitted._wrapper_source_binding()
    child_pid = os.getpid()
    watchdog_pid = 220_001
    wrapper_pid = 220_002
    child_ticks = budget.process_start_ticks(child_pid)
    watchdog_ticks = 330_001
    wrapper_ticks = 330_002
    assert child_ticks
    reservation_ns = budget.GPU_BUDGET_NS // 3
    command_sha256 = budget.command_sha256(["benchmark-worker"])

    def reservation(lifecycle_id: str, previous: str | None) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": 2,
            "event": "reservation",
            "lifecycle_id": lifecycle_id,
            "campaign_id": "test-campaign",
            "phase": "efficiency_benchmark",
            "context": {
                "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
                "outer_fold": 3,
                "seed": 20260828,
                "variant": "H0_no_factor",
            },
            "invocation_sha256": "a" * 64,
            "command_sha256": command_sha256,
            "result_path": str((tmp_path / f"{lifecycle_id}.json").resolve()),
            "gpu_execution_ledger_path": str(execution_ledger),
            "boot_id": budget.boot_id(),
            "wrapper_pid": wrapper_pid,
            "wrapper_start_ticks": wrapper_ticks,
            "wrapper_parent_pid": 1,
            "hostname": "test-host",
            "cwd": str(tmp_path.resolve()),
            "gpu_lock_file": str(lock_file),
            "gpu_lock_st_dev": lock_status.st_dev,
            "gpu_lock_st_ino": lock_status.st_ino,
            "wrapper_source": wrapper_source,
            "cuda_visible_devices": "0",
            "realtime_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "budget_ns": budget.GPU_BUDGET_NS,
            "reservation_ns": reservation_ns,
            "workload_timeout_ns": reservation_ns
            - budget.TERMINATION_GRACE_NS
            - budget.ACCOUNTING_MARGIN_NS,
            "heartbeat_interval_ns": budget.HEARTBEAT_INTERVAL_NS,
            "termination_grace_ns": budget.TERMINATION_GRACE_NS,
            "accounting_margin_ns": budget.ACCOUNTING_MARGIN_NS,
            "recovery_margin_ns": budget.RECOVERY_MARGIN_NS,
            "admitted_child_authorization": authorization,
            "previous_record_sha256": previous,
        }
        record["record_sha256"] = budget.semantic_sha256(record)
        return record

    first = reservation("benchmark-live", None)
    second = reservation("foreign-live", str(first["record_sha256"]))
    first_line = budget.canonical_json_bytes(first) + b"\n"
    usage_ledger.write_bytes(
        first_line + budget.canonical_json_bytes(second) + b"\n"
    )
    payload = {
        "schema_version": 1,
        "classification": admitted._ADMITTED_CHILD_CLASSIFICATION,
        "nonce": "c" * 64,
        "boot_id": budget.boot_id(),
        "lifecycle_id": "benchmark-live",
        "reservation_record_sha256": first["record_sha256"],
        "campaign_id": "test-campaign",
        "phase": "efficiency_benchmark",
        "context": first["context"],
        "invocation_sha256": "a" * 64,
        "command_sha256": command_sha256,
        "wrapper_pid": wrapper_pid,
        "wrapper_start_ticks": wrapper_ticks,
        "watchdog_pid": watchdog_pid,
        "watchdog_start_ticks": watchdog_ticks,
        "child_pid": child_pid,
        "child_start_ticks": child_ticks,
        "gpu_lock_file": str(lock_file),
        "gpu_lock_fd": 999_999,
        "gpu_lock_st_dev": lock_status.st_dev,
        "gpu_lock_st_ino": lock_status.st_ino,
        "wrapper_source": wrapper_source,
        "authorization": authorization,
        "usage_ledger_path": str(usage_ledger),
        "usage_ledger_prefix_bytes": len(first_line),
        "usage_ledger_prefix_sha256": hashlib.sha256(first_line).hexdigest(),
        "execution_ledger_path": str(execution_ledger),
        "execution_ledger_prefix_bytes": 3,
        "execution_ledger_prefix_sha256": hashlib.sha256(b"{}\n").hexdigest(),
        "execution_start_sha256": "d" * 64,
    }
    monkeypatch.setattr(admitted, "_read_admitted_child_pipe", lambda: payload)
    monkeypatch.setattr(admitted.os, "getppid", lambda: watchdog_pid)
    monkeypatch.setattr(
        admitted,
        "_proc_parent_pid",
        lambda pid: wrapper_pid if pid == watchdog_pid else None,
    )
    original_process_start_ticks = budget.process_start_ticks

    def process_ticks(pid: int) -> int | None:
        synthetic = {
            child_pid: child_ticks,
            watchdog_pid: watchdog_ticks,
            wrapper_pid: wrapper_ticks,
        }
        return synthetic[pid] if pid in synthetic else original_process_start_ticks(pid)

    monkeypatch.setattr(
        budget,
        "process_start_ticks",
        process_ticks,
    )
    monkeypatch.setattr(
        admitted,
        "_verify_wrapper_admission_lock",
        lambda _reservation, *, capability_fd: None,
    )
    with pytest.raises(RuntimeError, match="unrelated open lifecycle"):
        admitted.consume_admitted_child_binding(
            "efficiency_benchmark",
            usage_ledger,
            execution_ledger,
            authorization_path,
            authorization_sha256,
            expected_campaign_id="test-campaign",
            expected_gpu_lock_file=lock_file,
        )
