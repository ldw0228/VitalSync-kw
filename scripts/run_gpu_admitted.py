#!/usr/bin/env python3
"""Run one GPU job under exclusive admission and crash-safe budget accounting."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import ctypes
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_rr import gpu_budget_ledger as budget  # noqa: E402


_PR_SET_PDEATHSIG = 1
_PR_SET_CHILD_SUBREAPER = 36
_FORWARDED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_EXECUTION_PHASES = frozenset(
    {
        "efficiency_benchmark",
        "discovery",
        "promotion_training",
        "promotion_prediction",
    }
)
_ADMITTED_CHILD_FD_ENV = "SNN_RR_ADMITTED_CHILD_FD"
_ADMITTED_CHILD_CLASSIFICATION = "v8_gpu_admitted_child_lifecycle_binding"
_ADMITTED_CHILD_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "classification",
        "nonce",
        "boot_id",
        "lifecycle_id",
        "reservation_record_sha256",
        "campaign_id",
        "phase",
        "context",
        "invocation_sha256",
        "command_sha256",
        "wrapper_pid",
        "wrapper_start_ticks",
        "watchdog_pid",
        "watchdog_start_ticks",
        "child_pid",
        "child_start_ticks",
        "gpu_lock_file",
        "gpu_lock_fd",
        "gpu_lock_st_dev",
        "gpu_lock_st_ino",
        "wrapper_source",
        "authorization",
        "usage_ledger_path",
        "usage_ledger_prefix_bytes",
        "usage_ledger_prefix_sha256",
        "execution_ledger_path",
        "execution_ledger_prefix_bytes",
        "execution_ledger_prefix_sha256",
        "execution_start_sha256",
    }
)
_VERIFIED_ADMITTED_CHILD_BINDING_KEYS = frozenset(
    _ADMITTED_CHILD_BINDING_KEYS | {"valid", "binding_sha256"}
)
_WRAPPER_SOURCE_BINDING_KEYS = frozenset(
    {"path", "sha256", "bytes", "st_dev", "st_ino", "mode"}
)
_FAULT_INJECTION_HOOK: Callable[[str], None] | None = None


def _fault_inject(point: str) -> None:
    """Invoke the process-local deterministic crash-test hook, when installed."""

    hook = _FAULT_INJECTION_HOOK
    if hook is not None:
        hook(point)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execution_ledger_lock_path(path: Path) -> Path:
    resolved = _canonical_no_final_symlink(path, "GPU execution ledger")
    return resolved.with_name(resolved.name + ".lock")


def _unique_execution_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise budget.LedgerError(f"duplicate GPU execution ledger key: {key}")
        value[key] = item
    return value


def _decode_execution_ledger(raw: bytes) -> list[dict[str, Any]]:
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise budget.LedgerError("GPU execution ledger has a torn tail")
    records: list[dict[str, Any]] = []
    starts: set[str] = set()
    terminals: set[str] = set()
    for number, raw_line in enumerate(raw.splitlines(keepends=True), 1):
        if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
            raise budget.LedgerError(
                f"GPU execution ledger line {number} has a noncanonical newline"
            )
        encoded = raw_line[:-1]
        try:
            value = json.loads(
                encoded,
                object_pairs_hook=_unique_execution_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    budget.LedgerError(f"non-finite JSON constant: {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, budget.LedgerError) as error:
            raise budget.LedgerError(
                f"GPU execution ledger line {number} is invalid canonical JSON: {error}"
            ) from error
        if not isinstance(value, dict):
            raise budget.LedgerError(
                f"GPU execution ledger line {number} is not an object"
            )
        if raw_line != budget.canonical_json_bytes(value) + b"\n":
            raise budget.LedgerError(
                f"GPU execution ledger line {number} is not canonical JSON"
            )
        event = value.get("event")
        if event not in {"start", "end", "wrapper_exception"}:
            raise budget.LedgerError(
                f"GPU execution ledger line {number} has an invalid event"
            )
        identity_value = value.get("lifecycle_id", value.get("job_id"))
        if not isinstance(identity_value, str) or not identity_value:
            raise budget.LedgerError(
                f"GPU execution ledger line {number} lacks a lifecycle identity"
            )
        if event == "start":
            if identity_value in starts:
                raise budget.LedgerError("duplicate GPU execution start")
            starts.add(identity_value)
        else:
            if identity_value not in starts or identity_value in terminals:
                raise budget.LedgerError("orphan or duplicate GPU execution terminal")
            terminals.add(identity_value)
        records.append(value)
    return records


@contextmanager
def _exclusive_execution_ledger_lock(
    path: Path, *, pinned_directory_fd: object | None = None
) -> Any:
    resolved = _canonical_no_final_symlink(path, "GPU execution ledger")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved = _canonical_no_final_symlink(resolved, "GPU execution ledger")
    with ExitStack() as cleanup:
        if pinned_directory_fd is None:
            directory_fd = budget.open_pinned_directory(
                resolved.parent, label="GPU execution ledger parent"
            )
            directory_status = os.fstat(directory_fd)
            directory_identity = (
                directory_status.st_dev,
                directory_status.st_ino,
            )
            cleanup.callback(
                budget._close_fd_fail_safe,  # noqa: SLF001
                directory_fd,
                expected_identity=directory_identity,
                label="GPU execution ledger directory",
            )
            budget.validate_pinned_directory(
                resolved.parent,
                directory_fd,
                label="GPU execution ledger parent",
            )
            directory_lease = budget.acquire_directory_generation_fence(
                directory_fd
            )
            cleanup.callback(
                budget.release_directory_generation_fence, directory_lease
            )
        else:
            directory_fd = cleanup.enter_context(
                budget.borrow_directory_generation_lease(pinned_directory_fd)
            )
        lock_path = execution_ledger_lock_path(resolved)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path.name, flags, 0o644, dir_fd=directory_fd)
        try:
            descriptor_status = os.fstat(descriptor)
        except BaseException:
            budget._close_fd_fail_safe(  # noqa: SLF001
                descriptor, label="GPU execution ledger lock"
            )
            raise
        lock_identity = (descriptor_status.st_dev, descriptor_status.st_ino)
        cleanup.callback(
            budget._unlock_and_close_flocked_fd,  # noqa: SLF001
            descriptor,
            expected_identity=lock_identity,
            label="GPU execution ledger lock",
        )
        descriptor_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or descriptor_status.st_nlink != 1
        ):
            raise budget.LedgerError("GPU execution ledger lock is not regular")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        budget._path_names_open_directory(  # noqa: SLF001
            resolved.parent, directory_fd, "GPU execution ledger parent"
        )
        named_lock = os.stat(
            lock_path.name, dir_fd=directory_fd, follow_symlinks=False
        )
        current_status = os.fstat(descriptor)
        if (
            named_lock.st_dev,
            named_lock.st_ino,
        ) != (
            current_status.st_dev,
            current_status.st_ino,
        ) or current_status.st_nlink != 1:
            raise budget.LedgerError("GPU execution ledger lock inode changed")
        yield resolved, directory_fd
        budget.validate_pinned_directory(
            resolved.parent,
            directory_fd,
            label="GPU execution ledger parent",
        )
        named_lock = os.stat(
            lock_path.name, dir_fd=directory_fd, follow_symlinks=False
        )
        current_status = os.fstat(descriptor)
        if (
            not budget._same_inode(named_lock, current_status)  # noqa: SLF001
            or current_status.st_nlink != 1
            or named_lock.st_nlink != 1
        ):
            raise budget.LedgerError("GPU execution ledger lock inode changed")


def _read_execution_locked(path: Path, directory_fd: int) -> tuple[bytes, list[dict[str, Any]]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path.name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        raw = b""
    else:
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                raise budget.LedgerError(
                    "GPU execution ledger is aliased or not regular"
                )
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                raw = handle.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    return raw, _decode_execution_ledger(raw)


def _append_execution_locked(
    path: Path,
    directory_fd: int,
    raw: bytes,
    records: Sequence[Mapping[str, Any]],
    record: Mapping[str, object],
) -> None:
    document = dict(record)
    payload = raw + budget.canonical_json_bytes(document) + b"\n"
    _decode_execution_ledger(payload)
    budget._path_names_open_directory(  # noqa: SLF001
        path.parent, directory_fd, "GPU execution ledger parent"
    )
    current_raw, _ = _read_execution_locked(path, directory_fd)
    if current_raw != raw:
        raise budget.LedgerError(
            "GPU execution ledger changed during locked transaction"
        )
    budget._atomic_replace_bytes(  # noqa: SLF001
        path, payload, directory_fd=directory_fd
    )
    published, _ = _read_execution_locked(path, directory_fd)
    if published != payload:
        raise budget.LedgerError(
            "GPU execution ledger commit generation verification failed"
        )


def cleanup_execution_ledger_replace_residue(
    path: Path,
    *,
    pinned_directory_fd: object | None = None,
    admission_revalidate: Callable[[], None],
) -> bool:
    """Clean one killed execution-ledger commit under admission then execution lock."""

    with _exclusive_execution_ledger_lock(
        path, pinned_directory_fd=pinned_directory_fd
    ) as (resolved, directory_fd):
        raw, _records = _read_execution_locked(resolved, directory_fd)

        def validate_candidate(payload: bytes) -> list[dict[str, Any]]:
            return _decode_execution_ledger(payload)

        removed = budget._cleanup_atomic_replace_residue_locked(  # noqa: SLF001
            path=resolved,
            directory_fd=directory_fd,
            current_payload=raw,
            validate_candidate=validate_candidate,
            admission_revalidate=admission_revalidate,
        )
        current_raw, _current_records = _read_execution_locked(
            resolved, directory_fd
        )
        if current_raw != raw:
            raise budget.LedgerError(
                "GPU execution ledger changed during residue cleanup"
            )
        return removed


def append_ledger(
    path: Path,
    record: Mapping[str, object],
    *,
    pinned_directory_fd: object | None = None,
) -> None:
    """Durably preserve the pre-V7 GPU execution start/end event stream."""

    with _exclusive_execution_ledger_lock(
        path, pinned_directory_fd=pinned_directory_fd
    ) as (resolved, directory_fd):
        raw, records = _read_execution_locked(resolved, directory_fd)
        _append_execution_locked(
            resolved, directory_fd, raw, records, record
        )


def command_sha256(command: Sequence[str]) -> str:
    """Compatibility alias for the canonical child-command digest."""

    encoded = json.dumps(list(command), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _set_parent_death_signal(signum: int) -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("GPU parent-death containment requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, int(signum), 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def _set_child_subreaper() -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("GPU descendant containment requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def enable_wrapper_parent_death_containment() -> None:
    """Arm the wrapper before it can reserve or launch GPU work."""

    parent = os.getppid()
    _set_parent_death_signal(signal.SIGTERM)
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGTERM)


def _watchdog_pin_fds(pin_spec: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    descriptors: list[int] = []
    for item in pin_spec:
        descriptor = item.get("fd")
        if type(descriptor) is not int or descriptor <= 2 or descriptor in descriptors:
            raise RuntimeError("watchdog protected-directory descriptor is invalid")
        descriptors.append(descriptor)
    if not descriptors:
        raise RuntimeError("watchdog has no pinned protected directories")
    return tuple(descriptors)


def _validate_watchdog_pins(pin_spec: Sequence[Mapping[str, Any]]) -> None:
    """Bind inherited dirfds to exact path/inode identities and live fences."""

    _watchdog_pin_fds(pin_spec)
    for item in pin_spec:
        if set(item) != {"path", "fd", "st_dev", "st_ino"}:
            raise RuntimeError("watchdog protected-directory pin schema drifted")
        path_value = item.get("path")
        descriptor = item.get("fd")
        expected_device = item.get("st_dev")
        expected_inode = item.get("st_ino")
        if (
            not isinstance(path_value, str)
            or not path_value
            or type(descriptor) is not int
            or type(expected_device) is not int
            or type(expected_inode) is not int
        ):
            raise RuntimeError("watchdog protected-directory pin is invalid")
        status = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(status.st_mode)
            or (status.st_dev, status.st_ino) != (expected_device, expected_inode)
        ):
            raise RuntimeError("watchdog protected-directory fd identity drifted")
        budget.validate_pinned_directory(
            Path(path_value),
            descriptor,
            label="watchdog protected directory",
        )
        budget.require_directory_generation_fence(descriptor)


def _child_process_setup(
    expected_parent_pid: int,
    deadline_ns: int,
    pin_spec: Sequence[Mapping[str, Any]],
) -> None:
    """Gate workload exec on parent, deadline, and protected-tree identity."""

    _set_parent_death_signal(signal.SIGKILL)
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
    _validate_watchdog_pins(pin_spec)
    for descriptor in _watchdog_pin_fds(pin_spec):
        os.close(descriptor)
    if time.monotonic_ns() >= deadline_ns:
        os._exit(124)


def _normalize_exit_code(return_code: int, *, received_signal: int | None, timed_out: bool) -> int:
    if received_signal is not None:
        return 128 + int(received_signal)
    if timed_out:
        return 124
    if return_code < 0:
        return 128 + abs(return_code)
    return return_code


def _proc_session_id(pid: int) -> int | None:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as error:
        if not (Path("/proc") / str(pid)).exists():
            return None
        raise RuntimeError(f"cannot verify process session for PID {pid}: {error}") from error
    closing = raw.rfind(")")
    fields = raw[closing + 2 :].split() if closing >= 0 else []
    if len(fields) <= 3:
        raise RuntimeError(f"malformed /proc session identity for PID {pid}")
    try:
        return int(fields[3])
    except ValueError as error:
        raise RuntimeError(f"invalid /proc session identity for PID {pid}") from error


def _proc_parent_pid(pid: int) -> int | None:
    """Return the live Linux parent PID for one exact process identity."""

    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as error:
        if not (Path("/proc") / str(pid)).exists():
            return None
        raise RuntimeError(
            f"cannot verify process parent for PID {pid}: {error}"
        ) from error
    closing = raw.rfind(")")
    fields = raw[closing + 2 :].split() if closing >= 0 else []
    if len(fields) <= 1:
        raise RuntimeError(f"malformed /proc parent identity for PID {pid}")
    try:
        return int(fields[1])
    except ValueError as error:
        raise RuntimeError(f"invalid /proc parent identity for PID {pid}") from error


def _session_members(session_id: int, *, exclude: Sequence[int] = ()) -> tuple[int, ...]:
    if type(session_id) is not int or session_id <= 0:
        raise RuntimeError("workload session identity is invalid")
    excluded = set(exclude)
    members: list[int] = []
    try:
        entries = tuple(os.scandir("/proc"))
    except OSError as error:
        raise RuntimeError(f"cannot enumerate workload session: {error}") from error
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        observed = _proc_session_id(pid)
        if observed == session_id:
            members.append(pid)
    return tuple(sorted(members))


def _signal_session(
    session_id: int, signum: int, *, exclude: Sequence[int] = ()
) -> tuple[int, ...]:
    members = _session_members(session_id, exclude=exclude)
    for pid in members:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass
    return members


def _reap_session_children(
    session_id: int, *, exclude: Sequence[int] = ()
) -> None:
    for pid in _session_members(session_id, exclude=exclude):
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, ProcessLookupError):
            pass


def _terminate_session_until_empty(
    session_id: int,
    *,
    grace_ns: int,
    margin_ns: int,
    exclude: Sequence[int] = (),
    reap: Any | None = None,
    reap_exclude: Sequence[int] = (),
    initial_signal: int = signal.SIGTERM,
) -> bool:
    """TERM, grace, KILL, and prove a complete workload session is empty."""

    started = time.monotonic_ns()
    grace_deadline = started + grace_ns
    final_deadline = grace_deadline + margin_ns
    _signal_session(session_id, initial_signal, exclude=exclude)
    escalated = False
    while True:
        if reap is not None:
            reap()
        _reap_session_children(
            session_id, exclude=tuple(exclude) + tuple(reap_exclude)
        )
        members = _session_members(session_id, exclude=exclude)
        if not members:
            return escalated
        now = time.monotonic_ns()
        if now >= grace_deadline:
            escalated = True
            _signal_session(session_id, signal.SIGKILL, exclude=exclude)
        if now >= final_deadline:
            if reap is not None:
                reap()
            _reap_session_children(
                session_id, exclude=tuple(exclude) + tuple(reap_exclude)
            )
            remaining = _session_members(session_id, exclude=exclude)
            if remaining:
                raise RuntimeError(
                    f"workload session is not empty after forced cleanup: {remaining}"
                )
            return escalated
        time.sleep(0.01)


def _write_pipe_json(descriptor: int, value: Mapping[str, Any]) -> None:
    payload = budget.canonical_json_bytes(dict(value)) + b"\n"
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write to watchdog pipe")
        view = view[written:]


def _read_pipe_json(descriptor: int, *, timeout_seconds: float) -> dict[str, Any]:
    ready, _, _ = select.select([descriptor], [], [], timeout_seconds)
    if not ready:
        raise RuntimeError("GPU watchdog status pipe timed out")
    payload = bytearray()
    while True:
        block = os.read(descriptor, 4096)
        if not block:
            break
        payload.extend(block)
        if b"\n" in block:
            break
    if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        raise RuntimeError("GPU watchdog returned a torn status message")
    try:
        value = json.loads(payload[:-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"GPU watchdog returned invalid JSON: {error}") from error
    if not isinstance(value, dict) or payload != budget.canonical_json_bytes(value) + b"\n":
        raise RuntimeError("GPU watchdog returned non-canonical status")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_single_link_regular(path: Path, label: str) -> bytes:
    """Read one pathname without accepting a symlink or hard-link alias."""

    resolved = _canonical_no_final_symlink(path, label)
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
            or named.st_nlink != 1
            or not budget._same_inode(named, status)  # noqa: SLF001
        ):
            raise RuntimeError(f"{label} is aliased or not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _immutable_authorization_binding(
    path: Path, expected_sha256: str
) -> dict[str, Any]:
    """Bind an exact single-link, read-only authorization document."""

    if not _is_sha256(expected_sha256):
        raise ValueError(
            "authorization SHA-256 must be 64 lowercase hexadecimal characters"
        )
    resolved = _canonical_no_final_symlink(path, "GPU child authorization")
    raw = _read_single_link_regular(resolved, "GPU child authorization")
    status = os.stat(resolved, follow_symlinks=False)
    if stat.S_IMODE(status.st_mode) != 0o444:
        raise RuntimeError("GPU child authorization must have exact mode 0444")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise RuntimeError("GPU child authorization SHA-256 drifted")
    return {
        "path": str(resolved),
        "sha256": observed,
        "bytes": len(raw),
        "mode": "0444",
    }


def _wrapper_source_binding() -> dict[str, Any]:
    """Bind the exact wrapper module bytes used by this process."""

    source = _canonical_no_final_symlink(
        Path(__file__).resolve(), "GPU admission wrapper source"
    )
    raw = _read_single_link_regular(source, "GPU admission wrapper source")
    status = os.stat(source, follow_symlinks=False)
    return {
        "path": str(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "st_dev": status.st_dev,
        "st_ino": status.st_ino,
        "mode": f"{stat.S_IMODE(status.st_mode):04o}",
    }


def _validate_wrapper_source_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _WRAPPER_SOURCE_BINDING_KEYS:
        raise budget.LedgerError("GPU admission wrapper source binding is invalid")
    if not (
        isinstance(value.get("path"), str)
        and _is_sha256(value.get("sha256"))
        and type(value.get("bytes")) is int
        and value["bytes"] > 0
        and type(value.get("st_dev")) is int
        and value["st_dev"] >= 0
        and type(value.get("st_ino")) is int
        and value["st_ino"] > 0
        and isinstance(value.get("mode"), str)
        and len(value["mode"]) == 4
        and all(character in "01234567" for character in value["mode"])
    ):
        raise budget.LedgerError("GPU admission wrapper source identity is invalid")
    refreshed = _wrapper_source_binding()
    if budget.canonical_json_bytes(dict(value)) != budget.canonical_json_bytes(
        refreshed
    ):
        raise budget.LedgerError("GPU admission wrapper source drifted")
    return refreshed


def _flock_records_for_inode(status: os.stat_result) -> list[tuple[str, ...]]:
    """Return canonical Linux flock records for one exact inode."""

    try:
        lines = Path("/proc/locks").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise budget.LedgerError("cannot inspect Linux GPU admission locks") from error
    expected = (os.major(status.st_dev), os.minor(status.st_dev), status.st_ino)
    matching: list[tuple[str, ...]] = []
    for line in lines:
        fields = tuple(line.split())
        if len(fields) != 8 or fields[1:4] != ("FLOCK", "ADVISORY", "WRITE"):
            continue
        identity = fields[5].split(":")
        if len(identity) != 3:
            continue
        try:
            observed = (int(identity[0], 16), int(identity[1], 16), int(identity[2]))
        except ValueError:
            continue
        if observed == expected:
            matching.append(fields)
    return matching


def _validate_reservation_authorization(
    reservation: Mapping[str, Any],
    authorization: Mapping[str, Any] | None,
) -> None:
    observed = reservation.get("admitted_child_authorization")
    if authorization is None:
        if observed is not None:
            raise budget.LedgerError(
                "GPU lifecycle recovery omitted its admitted-child authorization"
            )
        return
    if budget.canonical_json_bytes(observed) != budget.canonical_json_bytes(
        dict(authorization)
    ):
        raise budget.LedgerError("GPU reservation authorization binding drifted")


def _verify_wrapper_admission_lock(
    reservation: Mapping[str, Any], *, capability_fd: int
) -> None:
    """Prove an inherited flock OFD is owned by the exact live wrapper.

    Merely observing that the wrapper has the inode open while *some* process
    blocks a probe lock is insufficient: a sibling can be that lock holder.
    Linux exposes the PID that acquired an active ``flock`` in ``/proc/locks``;
    the inherited descriptor, its exact wrapper descriptor number, the named
    inode, and that owner PID must all agree before the child is admitted.
    """

    wrapper_pid = reservation.get("wrapper_pid")
    wrapper_ticks = reservation.get("wrapper_start_ticks")
    lock_value = reservation.get("gpu_lock_file")
    expected_device = reservation.get("gpu_lock_st_dev")
    expected_inode = reservation.get("gpu_lock_st_ino")
    if not (
        type(wrapper_pid) is int
        and wrapper_pid > 0
        and type(wrapper_ticks) is int
        and wrapper_ticks > 0
        and budget.process_start_ticks(wrapper_pid) == wrapper_ticks
        and isinstance(lock_value, str)
        and type(expected_device) is int
        and expected_device >= 0
        and type(expected_inode) is int
        and expected_inode > 0
        and type(capability_fd) is int
        and capability_fd > 2
    ):
        raise budget.LedgerError("GPU wrapper admission-lock identity is invalid")
    lock_path = _canonical_no_final_symlink(
        Path(lock_value), "GPU admission lock"
    )
    try:
        capability_status = os.fstat(capability_fd)
        capability_access = fcntl.fcntl(capability_fd, fcntl.F_GETFL) & os.O_ACCMODE
        wrapper_status = os.stat(
            Path("/proc") / str(wrapper_pid) / "fd" / str(capability_fd)
        )
        named_status = os.stat(lock_path, follow_symlinks=False)
    except (FileNotFoundError, PermissionError, OSError) as error:
        raise budget.LedgerError(
            "cannot inspect exact wrapper admission-lock capability"
        ) from error
    if not (
        stat.S_ISREG(capability_status.st_mode)
        and capability_status.st_nlink == 1
        and capability_access == os.O_RDWR
        and (capability_status.st_dev, capability_status.st_ino)
        == (expected_device, expected_inode)
        and budget._same_inode(capability_status, wrapper_status)  # noqa: SLF001
        and budget._same_inode(capability_status, named_status)  # noqa: SLF001
        and named_status.st_nlink == 1
    ):
        raise budget.LedgerError("GPU wrapper admission-lock capability drifted")
    lock_records = _flock_records_for_inode(capability_status)
    if not (
        len(lock_records) == 1
        and lock_records[0][4].isdigit()
        and int(lock_records[0][4]) == wrapper_pid
        and lock_records[0][6:] == ("0", "EOF")
    ):
        raise budget.LedgerError(
            "GPU admission flock is not owned by the exact wrapper process"
        )

    flags = os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags)
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or not budget._same_inode(status, capability_status)  # noqa: SLF001
        ):
            raise budget.LedgerError("GPU admission lock is aliased or not regular")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            raise budget.LedgerError("GPU admission lock is not held by the wrapper")
    finally:
        os.close(descriptor)


def _open_execution_starts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    starts: dict[str, Mapping[str, Any]] = {}
    closed: set[str] = set()
    for record in records:
        lifecycle = str(record.get("lifecycle_id", record.get("job_id", "")))
        if record.get("event") == "start":
            starts[lifecycle] = record
        else:
            closed.add(lifecycle)
    return {
        lifecycle: record
        for lifecycle, record in starts.items()
        if lifecycle not in closed
    }


def _validate_live_execution_start(
    start: Mapping[str, Any],
    reservation: Mapping[str, Any],
    *,
    usage_ledger: Path,
    execution_ledger: Path,
) -> None:
    if set(start) != _EXECUTION_START_FIELDS or start.get("event") != "start":
        raise budget.LedgerError("GPU execution live start schema drifted")
    command = start.get("command")
    if not (
        isinstance(command, list)
        and command
        and all(isinstance(part, str) for part in command)
        and budget.command_sha256(command) == reservation.get("command_sha256")
    ):
        raise budget.LedgerError("GPU execution live command binding drifted")
    expected = {
        "schema_version": 1,
        "job_id": reservation.get("lifecycle_id"),
        "lifecycle_id": reservation.get("lifecycle_id"),
        "reservation_record_sha256": reservation.get("record_sha256"),
        "wrapper_pid": reservation.get("wrapper_pid"),
        "hostname": reservation.get("hostname"),
        "cwd": reservation.get("cwd"),
        "lock_file": reservation.get("gpu_lock_file"),
        "usage_ledger": str(usage_ledger),
        "result_file": reservation.get("result_path"),
        "campaign_id": reservation.get("campaign_id"),
        "phase": reservation.get("phase"),
        "context": reservation.get("context"),
        "invocation_sha256": reservation.get("invocation_sha256"),
        "command": command,
        "command_sha256": reservation.get("command_sha256"),
        "cuda_visible_devices": reservation.get("cuda_visible_devices"),
        "event": "start",
    }
    for field, expected_value in expected.items():
        observed = start.get(field)
        if field == "context":
            if budget.canonical_json_bytes(observed) != budget.canonical_json_bytes(
                expected_value
            ):
                raise budget.LedgerError("GPU execution live context drifted")
        elif observed != expected_value:
            raise budget.LedgerError(
                f"GPU execution live start field drifted: {field}"
            )
    if not isinstance(start.get("utc"), str) or not start.get("utc"):
        raise budget.LedgerError("GPU execution live start UTC is invalid")
    if reservation.get("gpu_execution_ledger_path") != str(execution_ledger):
        raise budget.LedgerError("GPU reservation execution-ledger path drifted")


def _build_admitted_child_binding(
    *,
    reservation: Mapping[str, Any],
    usage_ledger: Path,
    execution_ledger: Path,
    authorization: Mapping[str, Any],
    watchdog_pid: int,
    watchdog_start_ticks: int,
    child_pid: int,
    child_start_ticks: int,
    admission_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the one-shot capability only after reservation and start are durable."""

    refreshed_authorization = _immutable_authorization_binding(
        Path(str(authorization["path"])), str(authorization["sha256"])
    )
    if budget.canonical_json_bytes(refreshed_authorization) != budget.canonical_json_bytes(
        dict(authorization)
    ):
        raise RuntimeError("GPU child authorization identity drifted before launch")
    _validate_reservation_authorization(reservation, authorization)
    _validate_wrapper_source_binding(reservation.get("wrapper_source"))
    lock_fd = admission_lock.get("fd")
    if not (
        set(admission_lock) == {"path", "fd", "st_dev", "st_ino"}
        and admission_lock.get("path") == reservation.get("gpu_lock_file")
        and admission_lock.get("st_dev") == reservation.get("gpu_lock_st_dev")
        and admission_lock.get("st_ino") == reservation.get("gpu_lock_st_ino")
        and type(lock_fd) is int
    ):
        raise budget.LedgerError("GPU admitted-child lock capability drifted")
    _verify_wrapper_admission_lock(reservation, capability_fd=lock_fd)
    usage_state = budget.verify_ledger(
        usage_ledger,
        expected_legacy_genesis_sha256=_legacy_expectation(usage_ledger),
    )
    lifecycle_id = str(reservation["lifecycle_id"])
    if (
        set(usage_state.open_reservations) != {lifecycle_id}
        or budget.canonical_json_bytes(
            usage_state.open_reservations[lifecycle_id].reservation
        )
        != budget.canonical_json_bytes(dict(reservation))
        or usage_state.tail_sha256 != reservation.get("record_sha256")
    ):
        raise budget.LedgerError(
            "GPU admitted child does not own the exact sole live reservation"
        )
    execution_raw = _read_single_link_regular(
        execution_ledger, "GPU execution ledger"
    )
    execution_records = _decode_execution_ledger(execution_raw)
    open_starts = _open_execution_starts(execution_records)
    if set(open_starts) != {lifecycle_id}:
        raise budget.LedgerError(
            "GPU admitted child does not own the exact sole live execution start"
        )
    start = open_starts[lifecycle_id]
    _validate_live_execution_start(
        start,
        reservation,
        usage_ledger=usage_ledger,
        execution_ledger=execution_ledger,
    )
    wrapper_pid = int(reservation["wrapper_pid"])
    wrapper_ticks = int(reservation["wrapper_start_ticks"])
    if (
        budget.process_start_ticks(wrapper_pid) != wrapper_ticks
        or budget.process_start_ticks(watchdog_pid) != watchdog_start_ticks
        or _proc_parent_pid(watchdog_pid) != wrapper_pid
        or budget.process_start_ticks(child_pid) != child_start_ticks
        or budget.boot_id() != reservation.get("boot_id")
    ):
        raise budget.LedgerError("GPU admitted child process ancestry drifted")
    return {
        "schema_version": 1,
        "classification": _ADMITTED_CHILD_CLASSIFICATION,
        "nonce": os.urandom(32).hex(),
        "boot_id": reservation["boot_id"],
        "lifecycle_id": lifecycle_id,
        "reservation_record_sha256": reservation["record_sha256"],
        "campaign_id": reservation["campaign_id"],
        "phase": reservation["phase"],
        "context": reservation["context"],
        "invocation_sha256": reservation["invocation_sha256"],
        "command_sha256": reservation["command_sha256"],
        "wrapper_pid": wrapper_pid,
        "wrapper_start_ticks": wrapper_ticks,
        "watchdog_pid": watchdog_pid,
        "watchdog_start_ticks": watchdog_start_ticks,
        "child_pid": child_pid,
        "child_start_ticks": child_start_ticks,
        "gpu_lock_file": reservation["gpu_lock_file"],
        "gpu_lock_fd": lock_fd,
        "gpu_lock_st_dev": reservation["gpu_lock_st_dev"],
        "gpu_lock_st_ino": reservation["gpu_lock_st_ino"],
        "wrapper_source": dict(reservation["wrapper_source"]),
        "authorization": dict(authorization),
        "usage_ledger_path": str(usage_ledger),
        "usage_ledger_prefix_bytes": len(usage_state.raw_bytes),
        "usage_ledger_prefix_sha256": hashlib.sha256(
            usage_state.raw_bytes
        ).hexdigest(),
        "execution_ledger_path": str(execution_ledger),
        "execution_ledger_prefix_bytes": len(execution_raw),
        "execution_ledger_prefix_sha256": hashlib.sha256(execution_raw).hexdigest(),
        "execution_start_sha256": hashlib.sha256(
            budget.canonical_json_bytes(dict(start))
        ).hexdigest(),
    }


def _read_admitted_child_pipe(*, timeout_seconds: float = 30.0) -> dict[str, Any]:
    raw_descriptor = os.environ.pop(_ADMITTED_CHILD_FD_ENV, None)
    if raw_descriptor is None or not raw_descriptor.isascii() or not raw_descriptor.isdigit():
        raise RuntimeError("GPU admitted-child inherited descriptor is absent")
    descriptor = int(raw_descriptor)
    if descriptor <= 2:
        raise RuntimeError("GPU admitted-child inherited descriptor is invalid")
    try:
        status = os.fstat(descriptor)
        access_mode = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        if not stat.S_ISFIFO(status.st_mode) or access_mode != os.O_RDONLY:
            raise RuntimeError(
                "GPU admitted-child capability is not an inherited read-only pipe"
            )
        deadline = time.monotonic() + timeout_seconds
        payload = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("GPU admitted-child capability pipe timed out")
            ready, _, _ = select.select([descriptor], [], [], remaining)
            if not ready:
                raise RuntimeError("GPU admitted-child capability pipe timed out")
            block = os.read(descriptor, 4096)
            if not block:
                break
            payload.extend(block)
            if len(payload) > 65_536:
                raise RuntimeError("GPU admitted-child capability is oversized")
    finally:
        os.close(descriptor)
    if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        raise RuntimeError("GPU admitted-child capability is torn")
    try:
        value = json.loads(
            payload[:-1].decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_execution_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RuntimeError(f"non-finite admitted-child value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, budget.LedgerError) as error:
        raise RuntimeError(f"GPU admitted-child capability is invalid: {error}") from error
    if (
        not isinstance(value, dict)
        or set(value) != _ADMITTED_CHILD_BINDING_KEYS
        or payload != budget.canonical_json_bytes(value) + b"\n"
    ):
        raise RuntimeError("GPU admitted-child capability schema is non-canonical")
    return value


def _verified_admitted_child_digest(value: Mapping[str, Any]) -> str:
    document = dict(value)
    document.pop("binding_sha256", None)
    return hashlib.sha256(budget.canonical_json_bytes(document)).hexdigest()


def revalidate_consumed_admitted_child_binding(
    binding: Mapping[str, Any],
    *,
    expected_campaign_id: str,
    expected_phase: str,
    expected_gpu_lock_file: Path,
    expected_usage_ledger: Path,
    expected_execution_ledger: Path,
    expected_authorization_path: Path,
    expected_authorization_sha256: str,
) -> dict[str, Any]:
    """Fail closed if a previously consumed binding is no longer exactly live."""

    if expected_phase not in _EXECUTION_PHASES:
        raise ValueError("invalid expected GPU admitted-child phase")
    if not isinstance(expected_campaign_id, str) or not expected_campaign_id:
        raise ValueError("expected GPU admitted-child campaign must be non-empty")
    gpu_lock_file = _canonical_no_final_symlink(
        expected_gpu_lock_file, "GPU admission lock"
    )
    usage_ledger = _canonical_no_final_symlink(
        expected_usage_ledger, "GPU usage ledger"
    )
    execution_ledger = _canonical_no_final_symlink(
        expected_execution_ledger, "GPU execution ledger"
    )
    authorization = _immutable_authorization_binding(
        expected_authorization_path, expected_authorization_sha256
    )
    if not isinstance(binding, Mapping):
        raise RuntimeError("GPU admitted-child verified binding is not an object")
    payload = dict(binding)
    if set(payload) != _VERIFIED_ADMITTED_CHILD_BINDING_KEYS:
        raise RuntimeError("GPU admitted-child verified binding schema drifted")
    integer_fields = (
        "wrapper_pid",
        "wrapper_start_ticks",
        "watchdog_pid",
        "watchdog_start_ticks",
        "child_pid",
        "child_start_ticks",
        "usage_ledger_prefix_bytes",
        "execution_ledger_prefix_bytes",
    )
    if not (
        type(payload.get("schema_version")) is int
        and payload.get("schema_version") == 1
        and payload.get("classification")
        == "verified_v8_gpu_admitted_child_lifecycle"
        and payload.get("valid") is True
        and _is_sha256(payload.get("binding_sha256"))
        and payload["binding_sha256"] == _verified_admitted_child_digest(payload)
        and isinstance(payload.get("nonce"), str)
        and len(payload["nonce"]) == 64
        and all(character in "0123456789abcdef" for character in payload["nonce"])
        and payload.get("phase") == expected_phase
        and isinstance(payload.get("context"), dict)
        and all(type(payload.get(field)) is int and payload[field] > 0 for field in integer_fields)
        and all(
            _is_sha256(payload.get(field))
            for field in (
                "reservation_record_sha256",
                "invocation_sha256",
                "command_sha256",
                "usage_ledger_prefix_sha256",
                "execution_ledger_prefix_sha256",
                "execution_start_sha256",
            )
        )
        and isinstance(payload.get("lifecycle_id"), str)
        and payload["lifecycle_id"]
        and payload.get("campaign_id") == expected_campaign_id
        and isinstance(payload.get("boot_id"), str)
        and payload["boot_id"]
        and isinstance(payload.get("gpu_lock_file"), str)
        and payload["gpu_lock_file"] == str(gpu_lock_file)
        and type(payload.get("gpu_lock_fd")) is int
        and payload["gpu_lock_fd"] > 2
        and type(payload.get("gpu_lock_st_dev")) is int
        and payload["gpu_lock_st_dev"] >= 0
        and type(payload.get("gpu_lock_st_ino")) is int
        and payload["gpu_lock_st_ino"] > 0
        and payload.get("usage_ledger_path") == str(usage_ledger)
        and payload.get("execution_ledger_path") == str(execution_ledger)
        and budget.canonical_json_bytes(payload.get("authorization"))
        == budget.canonical_json_bytes(authorization)
    ):
        raise RuntimeError("GPU admitted-child capability identity is invalid")
    _validate_wrapper_source_binding(payload.get("wrapper_source"))

    child_pid = int(payload["child_pid"])
    watchdog_pid = int(payload["watchdog_pid"])
    wrapper_pid = int(payload["wrapper_pid"])
    if (
        os.getpid() != child_pid
        or budget.process_start_ticks(child_pid) != payload["child_start_ticks"]
        or os.getppid() != watchdog_pid
        or budget.process_start_ticks(watchdog_pid)
        != payload["watchdog_start_ticks"]
        or _proc_parent_pid(watchdog_pid) != wrapper_pid
        or budget.process_start_ticks(wrapper_pid) != payload["wrapper_start_ticks"]
        or budget.boot_id() != payload["boot_id"]
    ):
        raise RuntimeError("GPU admitted-child process ancestry is not authoritative")
    _verify_wrapper_admission_lock(
        {
            "wrapper_pid": payload["wrapper_pid"],
            "wrapper_start_ticks": payload["wrapper_start_ticks"],
            "gpu_lock_file": payload["gpu_lock_file"],
            "gpu_lock_st_dev": payload["gpu_lock_st_dev"],
            "gpu_lock_st_ino": payload["gpu_lock_st_ino"],
        },
        capability_fd=int(payload["gpu_lock_fd"]),
    )

    expected_genesis = _legacy_expectation(usage_ledger)
    usage_state = budget.verify_ledger(
        usage_ledger,
        expected_legacy_genesis_sha256=expected_genesis,
    )
    usage_prefix_bytes = int(payload["usage_ledger_prefix_bytes"])
    if (
        usage_prefix_bytes > len(usage_state.raw_bytes)
        or hashlib.sha256(usage_state.raw_bytes[:usage_prefix_bytes]).hexdigest()
        != payload["usage_ledger_prefix_sha256"]
    ):
        raise RuntimeError("GPU admitted-child usage-ledger prefix drifted")
    prefix_state = budget.verify_ledger_bytes(
        usage_state.raw_bytes[:usage_prefix_bytes],
        expected_legacy_genesis_sha256=expected_genesis,
    )
    lifecycle_id = str(payload["lifecycle_id"])
    if (
        set(prefix_state.open_reservations) != {lifecycle_id}
        or set(usage_state.open_reservations) != {lifecycle_id}
        or prefix_state.tail_sha256 != payload["reservation_record_sha256"]
    ):
        raise RuntimeError(
            "GPU admitted-child usage ledger has an unrelated open lifecycle"
        )
    reservation = usage_state.open_reservations[lifecycle_id].reservation
    prefix_reservation = prefix_state.open_reservations[lifecycle_id].reservation
    if budget.canonical_json_bytes(reservation) != budget.canonical_json_bytes(
        prefix_reservation
    ):
        raise RuntimeError("GPU admitted-child reservation changed after issuance")
    expected_reservation = {
        "record_sha256": payload["reservation_record_sha256"],
        "campaign_id": payload["campaign_id"],
        "phase": payload["phase"],
        "context": payload["context"],
        "invocation_sha256": payload["invocation_sha256"],
        "command_sha256": payload["command_sha256"],
        "wrapper_pid": payload["wrapper_pid"],
        "wrapper_start_ticks": payload["wrapper_start_ticks"],
        "boot_id": payload["boot_id"],
        "gpu_lock_file": payload["gpu_lock_file"],
        "gpu_lock_st_dev": payload["gpu_lock_st_dev"],
        "gpu_lock_st_ino": payload["gpu_lock_st_ino"],
        "wrapper_source": payload["wrapper_source"],
        "gpu_execution_ledger_path": str(execution_ledger),
        "admitted_child_authorization": authorization,
    }
    for field, expected in expected_reservation.items():
        observed = reservation.get(field)
        if field in {"context", "wrapper_source", "admitted_child_authorization"}:
            if budget.canonical_json_bytes(observed) != budget.canonical_json_bytes(
                expected
            ):
                raise RuntimeError(
                    f"GPU admitted-child reservation field drifted: {field}"
                )
        elif observed != expected:
            raise RuntimeError(
                f"GPU admitted-child reservation field drifted: {field}"
            )
    _verify_wrapper_admission_lock(
        reservation, capability_fd=int(payload["gpu_lock_fd"])
    )

    execution_raw = _read_single_link_regular(
        execution_ledger, "GPU execution ledger"
    )
    execution_prefix_bytes = int(payload["execution_ledger_prefix_bytes"])
    if (
        execution_prefix_bytes > len(execution_raw)
        or hashlib.sha256(execution_raw[:execution_prefix_bytes]).hexdigest()
        != payload["execution_ledger_prefix_sha256"]
    ):
        raise RuntimeError("GPU admitted-child execution-ledger prefix drifted")
    prefix_execution = _decode_execution_ledger(
        execution_raw[:execution_prefix_bytes]
    )
    current_execution = _decode_execution_ledger(execution_raw)
    prefix_open = _open_execution_starts(prefix_execution)
    current_open = _open_execution_starts(current_execution)
    if set(prefix_open) != {lifecycle_id} or set(current_open) != {lifecycle_id}:
        raise RuntimeError(
            "GPU admitted-child execution ledger has an unrelated open lifecycle"
        )
    start = prefix_open[lifecycle_id]
    if (
        hashlib.sha256(budget.canonical_json_bytes(dict(start))).hexdigest()
        != payload["execution_start_sha256"]
        or budget.canonical_json_bytes(start)
        != budget.canonical_json_bytes(current_open[lifecycle_id])
    ):
        raise RuntimeError("GPU admitted-child execution start drifted")
    _validate_live_execution_start(
        start,
        reservation,
        usage_ledger=usage_ledger,
        execution_ledger=execution_ledger,
    )
    return payload


def consume_admitted_child_binding(
    expected_phase: str,
    expected_usage_ledger: Path,
    expected_execution_ledger: Path,
    expected_authorization_path: Path,
    expected_authorization_sha256: str,
    *,
    expected_campaign_id: str,
    expected_gpu_lock_file: Path,
) -> dict[str, Any]:
    """Consume and verify the wrapper-only V8 live-lifecycle capability once.

    The pipe is consumed before *any* caller-supplied expectation is evaluated,
    so a wrong first phase/path/hash burns the one-shot capability.  The
    invocation digest deliberately comes from the inherited capability: adding
    it to the workload command would make the invocation self-referential.
    """

    payload = _read_admitted_child_pipe()
    verified = dict(payload)
    verified["classification"] = "verified_v8_gpu_admitted_child_lifecycle"
    verified["valid"] = True
    verified["binding_sha256"] = _verified_admitted_child_digest(verified)
    try:
        return revalidate_consumed_admitted_child_binding(
            verified,
            expected_campaign_id=expected_campaign_id,
            expected_phase=expected_phase,
            expected_gpu_lock_file=expected_gpu_lock_file,
            expected_usage_ledger=expected_usage_ledger,
            expected_execution_ledger=expected_execution_ledger,
            expected_authorization_path=expected_authorization_path,
            expected_authorization_sha256=expected_authorization_sha256,
        )
    except BaseException:
        lock_fd = verified.get("gpu_lock_fd")
        if type(lock_fd) is int and lock_fd > 2:
            try:
                os.close(lock_fd)
            except OSError:
                pass
        raise


def _watchdog_main(raw_spec: str) -> int:
    """Spawn/reap one trusted workload session under an absolute deadline."""

    try:
        spec = json.loads(raw_spec)
    except json.JSONDecodeError:
        return 70
    if not isinstance(spec, dict):
        return 70
    expected_parent = spec.get("wrapper_pid")
    ready_fd = spec.get("ready_fd")
    final_fd = spec.get("final_fd")
    command = spec.get("command")
    deadline_ns = spec.get("workload_deadline_ns")
    grace_ns = spec.get("termination_grace_ns")
    margin_ns = spec.get("accounting_margin_ns")
    pin_spec = spec.get("protected_directory_pins")
    admitted_child_fd = spec.get("admitted_child_fd")
    admission_lock = spec.get("admission_lock_capability")
    wrapper_start_ticks = spec.get("wrapper_start_ticks")
    wrapper_source = spec.get("wrapper_source")
    if not (
        type(expected_parent) is int
        and type(ready_fd) is int
        and type(final_fd) is int
        and isinstance(command, list)
        and command
        and all(isinstance(part, str) for part in command)
        and type(deadline_ns) is int
        and type(grace_ns) is int
        and grace_ns > 0
        and type(margin_ns) is int
        and margin_ns > 0
        and isinstance(pin_spec, list)
        and pin_spec
        and all(isinstance(item, dict) for item in pin_spec)
        and (
            admitted_child_fd is None
            or (type(admitted_child_fd) is int and admitted_child_fd > 2)
        )
        and (
            admitted_child_fd is None
            or (
                isinstance(admission_lock, dict)
                and set(admission_lock) == {"path", "fd", "st_dev", "st_ino"}
                and type(admission_lock.get("fd")) is int
                and admission_lock["fd"] > 2
                and admission_lock["fd"] != admitted_child_fd
                and type(wrapper_start_ticks) is int
                and wrapper_start_ticks > 0
                and isinstance(wrapper_source, dict)
            )
        )
    ):
        return 70
    try:
        _set_parent_death_signal(signal.SIGTERM)
        if os.getppid() != expected_parent:
            os.kill(os.getpid(), signal.SIGTERM)
        termination_requested = False
        termination_signal = int(signal.SIGTERM)

        def request_termination(signum: int, _frame: Any) -> None:
            nonlocal termination_requested, termination_signal
            termination_requested = True
            if signum != signal.SIGUSR1:
                termination_signal = int(signum)

        for forwarded in (signal.SIGUSR1, *_FORWARDED_SIGNALS):
            signal.signal(forwarded, request_termination)
        os.setsid()
        _set_child_subreaper()
        session_id = os.getpid()
        child: subprocess.Popen[Any] | None = None
        child_ticks: int | None = None
        hard_timeout = False
        escalated = False
        containment_anomaly = False
        error_value: dict[str, str] | None = None
        return_code = 127
        try:
            if termination_requested:
                error_value = {
                    "type": "SupervisorTerminationBeforeSpawn",
                    "message": "watchdog termination arrived before workload spawn",
                }
            elif time.monotonic_ns() >= deadline_ns:
                hard_timeout = True
                error_value = {
                    "type": "ReservationDeadlineExpired",
                    "message": "workload deadline elapsed before watchdog spawn",
                }
            if error_value is not None:
                _write_pipe_json(
                    ready_fd,
                    {
                        "child_pid": None,
                        "child_start_ticks": None,
                        "watchdog_pid": os.getpid(),
                        "workload_session_id": session_id,
                        "workload_deadline_ns": deadline_ns,
                        "exception": error_value,
                    },
                )
                os.close(ready_fd)
                ready_fd = -1
            else:
                watchdog_pid = os.getpid()
                _validate_watchdog_pins(pin_spec)
                inherited_pin_fds = _watchdog_pin_fds(pin_spec)
                child_environment = dict(os.environ)
                child_environment.pop(_ADMITTED_CHILD_FD_ENV, None)
                child_pass_fds = set(inherited_pin_fds)
                admission_lock_fd: int | None = None
                if admitted_child_fd is not None:
                    assert isinstance(admission_lock, dict)
                    admission_lock_fd = int(admission_lock["fd"])
                    binding_status = os.fstat(admitted_child_fd)
                    binding_access = (
                        fcntl.fcntl(admitted_child_fd, fcntl.F_GETFL)
                        & os.O_ACCMODE
                    )
                    if (
                        not stat.S_ISFIFO(binding_status.st_mode)
                        or binding_access != os.O_RDONLY
                        or admitted_child_fd in inherited_pin_fds
                        or admission_lock_fd in inherited_pin_fds
                    ):
                        raise RuntimeError(
                            "watchdog admitted-child descriptor is not a read-only pipe"
                        )
                    child_environment[_ADMITTED_CHILD_FD_ENV] = str(
                        admitted_child_fd
                    )
                    _validate_wrapper_source_binding(wrapper_source)
                    _verify_wrapper_admission_lock(
                        {
                            "wrapper_pid": expected_parent,
                            "wrapper_start_ticks": wrapper_start_ticks,
                            "gpu_lock_file": admission_lock["path"],
                            "gpu_lock_st_dev": admission_lock["st_dev"],
                            "gpu_lock_st_ino": admission_lock["st_ino"],
                        },
                        capability_fd=admission_lock_fd,
                    )
                    child_pass_fds.add(admitted_child_fd)
                    child_pass_fds.add(admission_lock_fd)
                try:
                    child = subprocess.Popen(
                        list(command),
                        preexec_fn=lambda: _child_process_setup(
                            watchdog_pid, deadline_ns, pin_spec
                        ),
                        pass_fds=tuple(sorted(child_pass_fds)),
                        close_fds=True,
                        env=child_environment,
                    )
                finally:
                    if admitted_child_fd is not None:
                        os.close(admitted_child_fd)
                        admitted_child_fd = None
                    if admission_lock_fd is not None:
                        os.close(admission_lock_fd)
                        admission_lock_fd = None
                _validate_watchdog_pins(pin_spec)
                child_ticks = budget.process_start_ticks(child.pid)
                if child_ticks is None:
                    raise RuntimeError("watchdog cannot establish workload identity")
                if _proc_session_id(child.pid) != session_id:
                    raise RuntimeError("workload escaped the trusted watchdog session")
                _write_pipe_json(
                    ready_fd,
                    {
                        "child_pid": child.pid,
                        "child_start_ticks": child_ticks,
                        "watchdog_pid": os.getpid(),
                        "workload_session_id": session_id,
                        "workload_deadline_ns": deadline_ns,
                    },
                )
                os.close(ready_fd)
                ready_fd = -1
            while child is not None:
                now = time.monotonic_ns()
                if now >= deadline_ns:
                    hard_timeout = True
                    escalated = _terminate_session_until_empty(
                        session_id,
                        grace_ns=grace_ns,
                        margin_ns=margin_ns,
                        exclude=(os.getpid(),),
                        reap=child.poll,
                        reap_exclude=(child.pid,),
                    ) or escalated
                    break
                child_status = child.poll()
                members = _session_members(session_id, exclude=(os.getpid(),))
                if termination_requested:
                    escalated = _terminate_session_until_empty(
                        session_id,
                        grace_ns=grace_ns,
                        margin_ns=margin_ns,
                        exclude=(os.getpid(),),
                        reap=child.poll,
                        reap_exclude=(child.pid,),
                        initial_signal=termination_signal,
                    ) or escalated
                    break
                if child_status is not None:
                    if members:
                        containment_anomaly = True
                        escalated = _terminate_session_until_empty(
                            session_id,
                            grace_ns=grace_ns,
                            margin_ns=margin_ns,
                            exclude=(os.getpid(),),
                            reap=child.poll,
                            reap_exclude=(child.pid,),
                        ) or escalated
                    break
                time.sleep(
                    min(0.05, max(0.001, (deadline_ns - now) / 1_000_000_000))
                )
            if child is not None:
                return_code = int(child.wait())
        except BaseException as error:
            error_value = {"type": type(error).__name__, "message": str(error)}
            containment_anomaly = child is not None
            escalated = _terminate_session_until_empty(
                session_id,
                grace_ns=grace_ns,
                margin_ns=margin_ns,
                exclude=(os.getpid(),),
                reap=(child.poll if child is not None else None),
                reap_exclude=((child.pid,) if child is not None else ()),
            ) or escalated
            if child is not None:
                return_code = int(child.wait())
            if ready_fd >= 0:
                _write_pipe_json(
                    ready_fd,
                    {
                        "child_pid": (child.pid if child is not None else None),
                        "child_start_ticks": child_ticks,
                        "watchdog_pid": os.getpid(),
                        "workload_session_id": session_id,
                        "workload_deadline_ns": deadline_ns,
                        "exception": error_value,
                    },
                )
                os.close(ready_fd)
                ready_fd = -1
        _reap_session_children(session_id, exclude=(os.getpid(),))
        remaining = _session_members(session_id, exclude=(os.getpid(),))
        if remaining:
            raise RuntimeError(f"watchdog cannot prove empty workload session: {remaining}")
        _write_pipe_json(
            final_fd,
            {
                "child_pid": (child.pid if child is not None else None),
                "child_start_ticks": child_ticks,
                "workload_session_id": session_id,
                "return_code": return_code,
                "hard_timeout_reached": hard_timeout,
                "termination_escalated": escalated,
                "containment_anomaly": containment_anomaly,
                "containment_empty": True,
                "ended_monotonic_ns": time.monotonic_ns(),
                "exception": error_value,
            },
        )
        os.close(final_fd)
        return 0 if error_value is None else 71
    except BaseException:
        return 72


def _independent_cleanup_after_watchdog_failure(
    session_id: int,
    watchdog: subprocess.Popen[Any],
    *,
    grace_ns: int,
    margin_ns: int,
) -> bool:
    """Clean the entire job even when watchdog status is absent or corrupt."""

    try:
        os.kill(watchdog.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    _signal_session(session_id, signal.SIGTERM)
    started = time.monotonic_ns()
    grace_deadline = started + grace_ns
    final_deadline = grace_deadline + margin_ns
    escalated = False
    while True:
        watchdog.poll()
        _reap_session_children(session_id, exclude=(watchdog.pid,))
        members = _session_members(session_id)
        watchdog_alive = watchdog.poll() is None
        if not members and not watchdog_alive:
            watchdog.wait()
            return escalated
        now = time.monotonic_ns()
        if now >= grace_deadline:
            escalated = True
            _signal_session(session_id, signal.SIGKILL)
            if watchdog_alive:
                try:
                    os.kill(watchdog.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if now >= final_deadline:
            watchdog.poll()
            _reap_session_children(session_id, exclude=(watchdog.pid,))
            remaining = _session_members(session_id)
            if remaining or watchdog.poll() is None:
                raise RuntimeError(
                    "cannot prove empty workload containment after watchdog failure"
                )
            watchdog.wait()
            return escalated
        time.sleep(0.01)


def _canonical_no_final_symlink(path: Path, label: str) -> Path:
    try:
        return budget._canonical_no_final_symlink(path, label)  # noqa: SLF001
    except budget.LedgerError as error:
        raise RuntimeError(str(error)) from error


def _validate_protected_paths(paths: Sequence[Path]) -> Path:
    """Require one non-root trusted tree and reject pathname/inode aliases."""

    canonical = [
        _canonical_no_final_symlink(path, "protected GPU lifecycle path")
        for path in paths
    ]
    if len(set(canonical)) != len(canonical):
        raise RuntimeError("GPU lifecycle protected paths must be distinct")
    common = Path(os.path.commonpath([os.fspath(path.parent) for path in canonical]))
    if common == Path(common.anchor):
        raise RuntimeError("GPU lifecycle protected paths lack a bounded trusted root")
    _canonical_no_final_symlink(common, "GPU lifecycle trusted root")
    inode_owners: dict[tuple[int, int], Path] = {}
    for path in canonical:
        try:
            status = os.lstat(path)
        except FileNotFoundError:
            continue
        if (
            stat.S_ISLNK(status.st_mode)
            or not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
        ):
            raise RuntimeError(
                f"protected GPU lifecycle path is aliased or not regular: {path}"
            )
        identity = (status.st_dev, status.st_ino)
        owner = inode_owners.get(identity)
        if owner is not None:
            raise RuntimeError(
                f"protected GPU lifecycle paths are inode aliases: {owner} and {path}"
            )
        inode_owners[identity] = path
    return common


def _existing_trusted_root(common: Path) -> tuple[Path, tuple[int, int]]:
    """Choose a bounded existing anchor before creating any protected parent."""

    candidate = common
    filesystem_root = Path(common.anchor)
    while candidate != filesystem_root:
        try:
            status = os.lstat(candidate)
        except FileNotFoundError:
            candidate = candidate.parent
            continue
        except OSError as error:
            raise RuntimeError(
                f"cannot inspect GPU lifecycle trusted root {candidate}: {error}"
            ) from error
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise RuntimeError(
                f"GPU lifecycle trusted root is symlinked or non-directory: {candidate}"
            )
        resolved = _canonical_no_final_symlink(candidate, "GPU lifecycle trusted root")
        return resolved, (status.st_dev, status.st_ino)
    raise RuntimeError("GPU lifecycle protected paths lack an existing bounded trusted root")


def _open_or_create_pinned_directory(
    trusted_root: Path,
    trusted_root_fd: int,
    target: Path,
) -> int:
    """Create ``target`` only through the already pinned trusted root."""

    try:
        relative = target.relative_to(trusted_root)
    except ValueError as error:
        raise RuntimeError(
            f"protected parent escapes the pinned trusted root: {target}"
        ) from error
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.dup(trusted_root_fd)
    current_path = trusted_root
    try:
        for component in relative.parts:
            budget.validate_pinned_directory(
                current_path,
                descriptor,
                label="GPU lifecycle protected ancestor",
            )
            try:
                expected_status = os.stat(
                    component, dir_fd=descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o755, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
                expected_status = os.stat(
                    component, dir_fd=descriptor, follow_symlinks=False
                )
            if not stat.S_ISDIR(expected_status.st_mode):
                raise RuntimeError(
                    f"protected GPU lifecycle ancestor is not a directory: "
                    f"{current_path / component}"
                )
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            next_status = os.fstat(next_descriptor)
            if (
                not stat.S_ISDIR(next_status.st_mode)
                or not budget._same_inode(expected_status, next_status)  # noqa: SLF001
            ):
                os.close(next_descriptor)
                raise RuntimeError(
                    f"protected GPU lifecycle ancestor changed while being pinned: "
                    f"{current_path / component}"
                )
            os.close(descriptor)
            descriptor = next_descriptor
            current_path = current_path / component
        budget.validate_pinned_directory(
            target,
            descriptor,
            label="GPU lifecycle protected parent",
        )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class _PinnedProtectedTree:
    def __init__(
        self,
        *,
        trusted_root: Path,
        trusted_root_fd: int,
        directory_fds: dict[Path, int],
        directory_leases: dict[Path, object],
        protected_paths: tuple[Path, ...],
        stable_paths: tuple[Path, ...],
    ) -> None:
        self.trusted_root = trusted_root
        self.trusted_root_fd = trusted_root_fd
        self.directory_fds = directory_fds
        self.directory_leases = directory_leases
        self.protected_paths = protected_paths
        self.stable_paths = stable_paths
        self.stable_identities: dict[Path, tuple[int, int]] = {}

    def fd_for(self, path: Path) -> int:
        resolved = _canonical_no_final_symlink(path, "protected GPU lifecycle path")
        try:
            return self.directory_fds[resolved.parent]
        except KeyError as error:
            raise RuntimeError(f"un-pinned protected parent: {resolved.parent}") from error

    def lease_for(self, path: Path) -> object:
        resolved = _canonical_no_final_symlink(path, "protected GPU lifecycle path")
        try:
            lease = self.directory_leases[resolved.parent]
        except KeyError as error:
            raise RuntimeError(f"unleased protected parent: {resolved.parent}") from error
        budget.directory_generation_lease_fd(lease)
        return lease

    def watchdog_pin_spec(self) -> list[dict[str, Any]]:
        entries = [(self.trusted_root, self.trusted_root_fd)] + list(
            self.directory_fds.items()
        )
        result: list[dict[str, Any]] = []
        seen_fds: set[int] = set()
        for path, descriptor in entries:
            if descriptor in seen_fds:
                continue
            seen_fds.add(descriptor)
            status = os.fstat(descriptor)
            result.append(
                {
                    "path": str(path),
                    "fd": descriptor,
                    "st_dev": status.st_dev,
                    "st_ino": status.st_ino,
                }
            )
        return result

    def watchdog_pass_fds(self) -> tuple[int, ...]:
        return tuple(item["fd"] for item in self.watchdog_pin_spec())

    def bind_stable_path(self, path: Path, *, missing_ok: bool = False) -> None:
        resolved = _canonical_no_final_symlink(path, "stable GPU lifecycle path")
        if resolved not in self.stable_paths:
            raise RuntimeError(f"undeclared stable GPU lifecycle path: {resolved}")
        directory_fd = self.fd_for(resolved)
        try:
            status = os.stat(
                resolved.name, dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            if missing_ok and resolved not in self.stable_identities:
                return
            raise RuntimeError(f"stable GPU lifecycle path is missing: {resolved}")
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise RuntimeError(
                f"stable GPU lifecycle path is aliased or non-regular: {resolved}"
            )
        identity = (status.st_dev, status.st_ino)
        prior = self.stable_identities.get(resolved)
        if prior is not None and prior != identity:
            raise RuntimeError(
                f"stable GPU lifecycle path generation changed: {resolved}"
            )
        self.stable_identities[resolved] = identity

    def revalidate(self) -> None:
        budget.validate_pinned_directory(
            self.trusted_root,
            self.trusted_root_fd,
            label="GPU lifecycle trusted root",
        )
        for path, descriptor in self.directory_fds.items():
            budget.validate_pinned_directory(
                path, descriptor, label="GPU lifecycle protected parent"
            )
        _validate_protected_paths(self.protected_paths)
        for path in self.stable_paths:
            self.bind_stable_path(path, missing_ok=True)


@contextmanager
def _pinned_protected_tree(
    paths: Sequence[Path], *, stable_paths: Sequence[Path] = ()
) -> Any:
    canonical = tuple(
        _canonical_no_final_symlink(path, "protected GPU lifecycle path")
        for path in paths
    )
    canonical_stable = tuple(
        _canonical_no_final_symlink(path, "stable GPU lifecycle path")
        for path in stable_paths
    )
    if not set(canonical_stable).issubset(canonical):
        raise RuntimeError("stable GPU lifecycle paths must be protected")
    if len(set(canonical)) != len(canonical):
        raise RuntimeError("GPU lifecycle protected paths must be distinct")
    common = Path(os.path.commonpath([os.fspath(path.parent) for path in canonical]))
    if common == Path(common.anchor):
        raise RuntimeError("GPU lifecycle protected paths lack a bounded trusted root")
    trusted_root, expected_root_identity = _existing_trusted_root(common)
    trusted_root_fd = budget.open_pinned_directory(
        trusted_root, label="GPU lifecycle trusted root"
    )
    opened_root_status = os.fstat(trusted_root_fd)
    if (opened_root_status.st_dev, opened_root_status.st_ino) != expected_root_identity:
        os.close(trusted_root_fd)
        raise RuntimeError("GPU lifecycle trusted root changed while being pinned")
    directory_fds: dict[Path, int] = {}
    directory_leases: dict[Path, object] = {}
    acquired: list[object] = []
    leases_by_identity: dict[tuple[int, int], object] = {}
    try:
        root_lease = budget.acquire_directory_generation_fence(trusted_root_fd)
        acquired.append(root_lease)
        root_status = os.fstat(trusted_root_fd)
        leases_by_identity[(root_status.st_dev, root_status.st_ino)] = root_lease
        for parent in sorted({path.parent for path in canonical}, key=os.fspath):
            descriptor = _open_or_create_pinned_directory(
                trusted_root, trusted_root_fd, parent
            )
            directory_fds[parent] = descriptor
            descriptor_status = os.fstat(descriptor)
            identity = (descriptor_status.st_dev, descriptor_status.st_ino)
            lease = leases_by_identity.get(identity)
            if lease is None:
                lease = budget.acquire_directory_generation_fence(descriptor)
                acquired.append(lease)
                leases_by_identity[identity] = lease
            directory_leases[parent] = lease
        _validate_protected_paths(canonical)
        pins = _PinnedProtectedTree(
            trusted_root=trusted_root,
            trusted_root_fd=trusted_root_fd,
            directory_fds=directory_fds,
            directory_leases=directory_leases,
            protected_paths=canonical,
            stable_paths=canonical_stable,
        )
        pins.revalidate()
        yield pins
        pins.revalidate()
    finally:
        cleanup_errors: list[BaseException] = []
        for lease in reversed(acquired):
            try:
                budget.release_directory_generation_fence(lease)
            except BaseException as error:
                cleanup_errors.append(error)
        for descriptor in directory_fds.values():
            try:
                status = os.fstat(descriptor)
                budget._close_fd_fail_safe(  # noqa: SLF001
                    descriptor,
                    expected_identity=(status.st_dev, status.st_ino),
                    label="GPU lifecycle protected directory",
                )
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            budget._close_fd_fail_safe(  # noqa: SLF001
                trusted_root_fd,
                expected_identity=expected_root_identity,
                label="GPU lifecycle trusted root",
            )
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise RuntimeError(
                "GPU lifecycle protected-tree cleanup failed: "
                + "; ".join(str(error) for error in cleanup_errors)
            ) from cleanup_errors[0]


@contextmanager
def _exclusive_gpu_lock(
    path: Path, *, pinned_directory_fd: object | None = None
) -> Any:
    """Acquire the stable GPU admission inode without following a symlink."""

    resolved = _canonical_no_final_symlink(path, "GPU admission lock")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved = _canonical_no_final_symlink(resolved, "GPU admission lock")
    with ExitStack() as cleanup:
        if pinned_directory_fd is None:
            directory_fd = budget.open_pinned_directory(
                resolved.parent, label="GPU admission lock parent"
            )
            directory_status = os.fstat(directory_fd)
            directory_identity = (
                directory_status.st_dev,
                directory_status.st_ino,
            )
            cleanup.callback(
                budget._close_fd_fail_safe,  # noqa: SLF001
                directory_fd,
                expected_identity=directory_identity,
                label="GPU admission lock directory",
            )
            budget.validate_pinned_directory(
                resolved.parent,
                directory_fd,
                label="GPU admission lock parent",
            )
            directory_lease = budget.acquire_directory_generation_fence(
                directory_fd
            )
            cleanup.callback(
                budget.release_directory_generation_fence, directory_lease
            )
        else:
            directory_fd = cleanup.enter_context(
                budget.borrow_directory_generation_lease(pinned_directory_fd)
            )
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(resolved.name, flags, 0o644, dir_fd=directory_fd)
        try:
            descriptor_status = os.fstat(descriptor)
        except BaseException:
            budget._close_fd_fail_safe(  # noqa: SLF001
                descriptor, label="GPU admission lock"
            )
            raise
        lock_identity = (descriptor_status.st_dev, descriptor_status.st_ino)
        cleanup.callback(
            budget._unlock_and_close_flocked_fd,  # noqa: SLF001
            descriptor,
            expected_identity=lock_identity,
            label="GPU admission lock",
        )
        descriptor_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or descriptor_status.st_nlink != 1
        ):
            raise RuntimeError(f"GPU admission lock is not a regular file: {resolved}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"GPU admission lock is already held: {resolved}") from error
        budget._path_names_open_directory(  # noqa: SLF001
            resolved.parent, directory_fd, "GPU admission lock parent"
        )
        named_lock = os.stat(
            resolved.name, dir_fd=directory_fd, follow_symlinks=False
        )
        descriptor_status = os.fstat(descriptor)
        if (named_lock.st_dev, named_lock.st_ino) != (
            descriptor_status.st_dev,
            descriptor_status.st_ino,
        ) or descriptor_status.st_nlink != 1:
            raise RuntimeError("GPU admission lock inode changed while held")
        yield {
            "path": str(resolved),
            "fd": descriptor,
            "st_dev": descriptor_status.st_dev,
            "st_ino": descriptor_status.st_ino,
        }
        budget.validate_pinned_directory(
            resolved.parent, directory_fd, label="GPU admission lock parent"
        )
        named_lock = os.stat(
            resolved.name, dir_fd=directory_fd, follow_symlinks=False
        )
        descriptor_status = os.fstat(descriptor)
        if (
            not budget._same_inode(named_lock, descriptor_status)  # noqa: SLF001
            or descriptor_status.st_nlink != 1
            or named_lock.st_nlink != 1
        ):
            raise RuntimeError("GPU admission lock inode changed while held")


def _revalidate_admission_lock_capability(
    capability: Mapping[str, Any],
) -> None:
    """Revalidate the exact stable admission inode held by this wrapper."""

    if set(capability) != {"path", "fd", "st_dev", "st_ino"}:
        raise budget.LedgerError("GPU admission lock capability schema drifted")
    path_value = capability.get("path")
    descriptor = capability.get("fd")
    expected_device = capability.get("st_dev")
    expected_inode = capability.get("st_ino")
    if not (
        isinstance(path_value, str)
        and path_value
        and type(descriptor) is int
        and descriptor > 2
        and type(expected_device) is int
        and expected_device >= 0
        and type(expected_inode) is int
        and expected_inode > 0
    ):
        raise budget.LedgerError("GPU admission lock capability is invalid")
    resolved = _canonical_no_final_symlink(
        Path(path_value), "GPU admission lock"
    )
    descriptor_status = os.fstat(descriptor)
    named_status = os.stat(resolved, follow_symlinks=False)
    if (
        not stat.S_ISREG(descriptor_status.st_mode)
        or descriptor_status.st_nlink != 1
        or named_status.st_nlink != 1
        or (descriptor_status.st_dev, descriptor_status.st_ino)
        != (expected_device, expected_inode)
        or not budget._same_inode(descriptor_status, named_status)  # noqa: SLF001
    ):
        raise budget.LedgerError("GPU admission lock capability generation drifted")


def _legacy_expectation(
    path: Path, *, pinned_directory_fd: int | None = None
) -> str | None:
    """Enforce the frozen V1 genesis whenever a legacy prefix is present."""

    state = budget.verify_ledger(
        path, pinned_directory_fd=pinned_directory_fd
    )
    if state.records and state.records[0].get("schema_version", 1) == 1:
        if (
            state.records[0].get("record_sha256")
            != budget.LEGACY_V1_GENESIS_RECORD_SHA256
        ):
            raise budget.LedgerError("legacy schema-v1 genesis is not the frozen V6 record")
        return budget.LEGACY_V1_GENESIS_RECORD_SHA256
    return None


def _terminal_matches(
    record: Mapping[str, Any],
    *,
    campaign_id: str,
    phase: str,
    context: Mapping[str, Any],
    invocation_sha256: str,
    child_command_sha256: str,
    result_file: Path,
) -> bool:
    return (
        record.get("event") == "terminal"
        and record.get("campaign_id") == campaign_id
        and record.get("phase") == phase
        and budget.canonical_json_bytes(record.get("context"))
        == budget.canonical_json_bytes(dict(context))
        and record.get("invocation_sha256") == invocation_sha256
        and record.get("command_sha256") == child_command_sha256
        and record.get("result_path") == str(result_file.resolve())
    )


def _publish_result(
    *,
    result_file: Path,
    terminal: Mapping[str, Any],
    usage_ledger: Path,
    gpu_ledger: Path,
    pinned_result_directory_fd: object | None = None,
    revalidate: Any | None = None,
) -> dict[str, Any]:
    if revalidate is not None:
        revalidate()
    return budget.atomic_result_receipt(
        result_file,
        budget.result_from_terminal(
            terminal,
            usage_ledger=usage_ledger,
            gpu_execution_ledger=gpu_ledger,
        ),
        pinned_directory_fd=pinned_result_directory_fd,
    )


_EXECUTION_START_FIELDS = frozenset(
    {
        "schema_version",
        "job_id",
        "lifecycle_id",
        "reservation_record_sha256",
        "wrapper_pid",
        "hostname",
        "cwd",
        "lock_file",
        "usage_ledger",
        "result_file",
        "campaign_id",
        "phase",
        "context",
        "invocation_sha256",
        "command",
        "command_sha256",
        "cuda_visible_devices",
        "event",
        "utc",
    }
)


def _execution_terminal_projection(
    terminal: Mapping[str, Any], reservation: Mapping[str, Any]
) -> dict[str, Any]:
    """Project normal and reconciled usage terminals into one end schema."""

    if terminal.get("event") == "terminal":
        fields = {
            "exit_code": terminal.get("return_code"),
            "wrapper_exit_code": terminal.get("wrapper_exit_code"),
            "hard_timeout_reached": terminal.get("hard_timeout_reached"),
            "termination_escalated": terminal.get("termination_escalated"),
            "containment_anomaly": terminal.get("containment_anomaly"),
            "reservation_deadline_breached": terminal.get(
                "reservation_deadline_breached"
            ),
            "terminal_record_sha256": terminal.get("record_sha256"),
        }
        if not (
            type(fields["exit_code"]) is int
            and type(fields["wrapper_exit_code"]) is int
            and all(
                type(fields[name]) is bool
                for name in (
                    "hard_timeout_reached",
                    "termination_escalated",
                    "containment_anomaly",
                    "reservation_deadline_breached",
                )
            )
            and _is_sha256(fields["terminal_record_sha256"])
        ):
            raise budget.LedgerError("GPU usage terminal projection is invalid")
        return fields
    if terminal.get("event") != "reconciled_terminal":
        raise budget.LedgerError("GPU execution recovery terminal event is invalid")
    charged = terminal.get("charged_usage_ns")
    reservation_ns = reservation.get("reservation_ns")
    if not (
        terminal.get("return_code") is None
        and terminal.get("reuse_eligible") is False
        and type(charged) is int
        and charged >= 0
        and type(reservation_ns) is int
        and reservation_ns > 0
        and charged <= reservation_ns
        and _is_sha256(terminal.get("record_sha256"))
    ):
        raise budget.LedgerError("reconciled GPU usage terminal is invalid")
    return {
        "exit_code": None,
        "wrapper_exit_code": 125,
        "hard_timeout_reached": False,
        "termination_escalated": False,
        # A dead wrapper cannot authoritatively prove normal containment.
        "containment_anomaly": True,
        "reservation_deadline_breached": charged == reservation_ns,
        "terminal_record_sha256": terminal["record_sha256"],
    }


def _validate_execution_recovery_records(
    *,
    start: Mapping[str, Any],
    end: Mapping[str, Any] | None,
    terminal: Mapping[str, Any],
    reservation: Mapping[str, Any],
    expected_command: Sequence[str],
    expected_lock_file: Path,
    gpu_ledger: Path,
    usage_ledger: Path,
) -> None:
    if set(start) != _EXECUTION_START_FIELDS or start.get("event") != "start":
        raise budget.LedgerError("GPU execution start schema or identity drifted")
    if (
        reservation.get("event") != "reservation"
        or reservation.get("record_sha256")
        != terminal.get("reservation_record_sha256")
        or reservation.get("lifecycle_id") != terminal.get("lifecycle_id")
    ):
        raise budget.LedgerError("GPU execution authoritative reservation drifted")
    resolved_lock = str(
        _canonical_no_final_symlink(expected_lock_file, "GPU admission lock")
    )
    if reservation.get("gpu_lock_file") != resolved_lock:
        raise budget.LedgerError("GPU reservation admission lock binding drifted")
    expected_command_list = list(expected_command)
    if budget.command_sha256(expected_command_list) != reservation.get(
        "command_sha256"
    ):
        raise budget.LedgerError("GPU reservation command binding drifted")
    expected_start = {
        "schema_version": 1,
        "job_id": reservation.get("lifecycle_id"),
        "lifecycle_id": reservation.get("lifecycle_id"),
        "reservation_record_sha256": reservation.get("record_sha256"),
        "wrapper_pid": reservation.get("wrapper_pid"),
        "hostname": reservation.get("hostname"),
        "cwd": reservation.get("cwd"),
        "lock_file": resolved_lock,
        "usage_ledger": str(
            _canonical_no_final_symlink(usage_ledger, "GPU usage ledger")
        ),
        "result_file": reservation.get("result_path"),
        "campaign_id": reservation.get("campaign_id"),
        "phase": reservation.get("phase"),
        "context": reservation.get("context"),
        "invocation_sha256": reservation.get("invocation_sha256"),
        "command": expected_command_list,
        "command_sha256": reservation.get("command_sha256"),
        "cuda_visible_devices": reservation.get("cuda_visible_devices"),
        "event": "start",
    }
    for field, expected in expected_start.items():
        observed = start.get(field)
        if field == "context":
            if budget.canonical_json_bytes(observed) != budget.canonical_json_bytes(
                expected
            ):
                raise budget.LedgerError(
                    "GPU execution start context differs from authoritative reservation"
                )
        elif observed != expected:
            raise budget.LedgerError(
                f"GPU execution start {field} differs from authoritative reservation"
            )
    if not isinstance(start.get("utc"), str) or not start.get("utc"):
        raise budget.LedgerError("GPU execution start utc is invalid")
    if terminal.get("gpu_execution_ledger_path") != str(
        _canonical_no_final_symlink(gpu_ledger, "GPU execution ledger")
    ):
        raise budget.LedgerError("usage terminal binds a different execution ledger")
    if end is None:
        return
    identity_fields = set(start) - {"event", "utc"}
    for field in identity_fields:
        if end.get(field) != start.get(field):
            raise budget.LedgerError(
                f"GPU execution end identity field drifted: {field}"
            )
    terminal_fields = _execution_terminal_projection(terminal, reservation)
    for field, expected in terminal_fields.items():
        if end.get(field) != expected:
            raise budget.LedgerError(
                f"GPU execution end terminal-derived field drifted: {field}"
            )
    required = identity_fields | {"event", "utc"} | set(terminal_fields)
    allowed = required | {
        "recovered_from_durable_usage_terminal",
        "usage_terminal_event",
    }
    if (
        not required.issubset(end)
        or set(end) - allowed
        or end.get("event") != "end"
        or not isinstance(end.get("utc"), str)
        or not end.get("utc")
    ):
        raise budget.LedgerError("GPU execution end schema drifted")
    recovered = end.get("recovered_from_durable_usage_terminal")
    if recovered is not None and recovered is not True:
        raise budget.LedgerError("GPU execution recovery marker drifted")
    usage_event = end.get("usage_terminal_event")
    if usage_event is not None and usage_event != terminal.get("event"):
        raise budget.LedgerError("GPU execution recovery usage-event marker drifted")


def _recover_gpu_execution_end(
    gpu_ledger: Path,
    terminal: Mapping[str, Any],
    *,
    reservation: Mapping[str, Any],
    expected_command: Sequence[str],
    expected_lock_file: Path,
    usage_ledger: Path,
    pinned_directory_fd: object | None = None,
    require_existing_end: bool = False,
) -> None:
    """Close the execution stream after a terminal-before-end wrapper crash."""

    with _exclusive_execution_ledger_lock(
        gpu_ledger, pinned_directory_fd=pinned_directory_fd
    ) as (
        resolved,
        directory_fd,
    ):
        raw, rows = _read_execution_locked(resolved, directory_fd)
        lifecycle_id = terminal.get("lifecycle_id")
        starts = [
            row
            for row in rows
            if row.get("lifecycle_id") == lifecycle_id
            and row.get("event") == "start"
        ]
        ends = [
            row
            for row in rows
            if row.get("lifecycle_id") == lifecycle_id
            and row.get("event") == "end"
        ]
        if len(starts) != 1 or len(ends) > 1:
            raise budget.LedgerError("GPU execution lifecycle is missing or duplicated")
        _validate_execution_recovery_records(
            start=starts[0],
            end=(ends[0] if ends else None),
            terminal=terminal,
            reservation=reservation,
            expected_command=expected_command,
            expected_lock_file=expected_lock_file,
            gpu_ledger=resolved,
            usage_ledger=usage_ledger,
        )
        if ends:
            return
        if require_existing_end:
            raise budget.LedgerError(
                "GPU execution end is missing behind an existing result"
            )
        start = dict(starts[0])
        start.pop("event", None)
        start.pop("utc", None)
        terminal_projection = _execution_terminal_projection(terminal, reservation)
        _append_execution_locked(
            resolved,
            directory_fd,
            raw,
            rows,
            {
                **start,
                "event": "end",
                "utc": utc_now(),
                **terminal_projection,
                "recovered_from_durable_usage_terminal": True,
                "usage_terminal_event": terminal["event"],
            },
        )


def _validate_bound_reservation_authorization(
    reservation: Mapping[str, Any],
) -> None:
    observed = reservation.get("admitted_child_authorization")
    if observed is None:
        _validate_reservation_authorization(reservation, None)
        return
    if not isinstance(observed, Mapping):
        raise budget.LedgerError("GPU reservation authorization binding is invalid")
    path_value = observed.get("path")
    sha_value = observed.get("sha256")
    if not isinstance(path_value, str) or not isinstance(sha_value, str):
        raise budget.LedgerError("GPU reservation authorization identity is invalid")
    refreshed = _immutable_authorization_binding(Path(path_value), sha_value)
    _validate_reservation_authorization(reservation, refreshed)


def _recover_closed_usage_execution_starts(
    gpu_ledger: Path,
    usage_state: budget.LedgerState,
    *,
    expected_lock_file: Path,
    usage_ledger: Path,
    pinned_directory_fd: object | None = None,
) -> int:
    """Close every stale execution start backed by one durable usage terminal.

    Usage reconciliation is authoritative for charging and non-reusability.  A
    wrapper crash can nevertheless leave the compatibility execution stream at
    ``start``.  Before another reservation is admitted, append an idempotent
    recovery ``end`` for each such lifecycle only after its reservation,
    command/invocation, authorization, ledger path, and terminal are exact.
    """

    if usage_state.open_reservations:
        raise budget.LedgerError(
            "GPU execution recovery requires a closed authoritative usage state"
        )

    with _exclusive_execution_ledger_lock(
        gpu_ledger, pinned_directory_fd=pinned_directory_fd
    ) as (resolved, directory_fd):
        raw, rows = _read_execution_locked(resolved, directory_fd)
        open_starts = _open_execution_starts(rows)
        recovered = 0
        for lifecycle_id, start_value in open_starts.items():
            start = dict(start_value)
            terminals = [
                dict(record)
                for record in usage_state.records
                if record.get("schema_version") == 2
                and record.get("event") in {"terminal", "reconciled_terminal"}
                and record.get("lifecycle_id") == lifecycle_id
                and record.get("reservation_record_sha256")
                == start.get("reservation_record_sha256")
            ]
            if len(terminals) != 1:
                raise budget.LedgerError(
                    "open GPU execution start lacks one authoritative usage terminal"
                )
            terminal = terminals[0]
            reservation = _reservation_for_terminal(usage_state, terminal)
            _validate_bound_reservation_authorization(reservation)
            command = start.get("command")
            if not (
                isinstance(command, list)
                and command
                and all(isinstance(part, str) for part in command)
            ):
                raise budget.LedgerError(
                    "open GPU execution start command is invalid"
                )
            _validate_execution_recovery_records(
                start=start,
                end=None,
                terminal=terminal,
                reservation=reservation,
                expected_command=command,
                expected_lock_file=expected_lock_file,
                gpu_ledger=resolved,
                usage_ledger=usage_ledger,
            )
            identity = dict(start)
            identity.pop("event", None)
            identity.pop("utc", None)
            terminal_projection = _execution_terminal_projection(
                terminal, reservation
            )
            end = {
                **identity,
                "event": "end",
                "utc": utc_now(),
                **terminal_projection,
                "recovered_from_durable_usage_terminal": True,
                "usage_terminal_event": terminal["event"],
            }
            _append_execution_locked(
                resolved, directory_fd, raw, rows, end
            )
            raw += budget.canonical_json_bytes(end) + b"\n"
            rows.append(end)
            recovered += 1
        if _open_execution_starts(rows):
            raise budget.LedgerError(
                "GPU execution ledger remains open after authoritative recovery"
            )
        return recovered


def _reservation_for_terminal(
    state: budget.LedgerState, terminal: Mapping[str, Any]
) -> dict[str, Any]:
    matches = [
        dict(record)
        for record in state.records
        if record.get("event") == "reservation"
        and record.get("record_sha256")
        == terminal.get("reservation_record_sha256")
        and record.get("lifecycle_id") == terminal.get("lifecycle_id")
    ]
    if len(matches) != 1:
        raise budget.LedgerError(
            "usage terminal does not bind one authoritative reservation"
        )
    return matches[0]


def _run_legacy(lock_file: Path, ledger: Path, command: Sequence[str]) -> int:
    """Preserve the original CLI/API behavior for non-campaign callers."""

    lock_file = _canonical_no_final_symlink(lock_file, "GPU admission lock")
    ledger = _canonical_no_final_symlink(ledger, "GPU execution ledger")
    execution_lock = execution_ledger_lock_path(ledger)
    protected = (lock_file, ledger, execution_lock)
    with _pinned_protected_tree(
        protected, stable_paths=(lock_file, execution_lock)
    ) as pins, _exclusive_gpu_lock(
        lock_file, pinned_directory_fd=pins.lease_for(lock_file)
    ):
        pins.revalidate()
        job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{os.getpid()}"
        common = {
            "schema_version": 1,
            "job_id": job_id,
            "wrapper_pid": os.getpid(),
            "hostname": socket.gethostname(),
            "cwd": str(Path.cwd().resolve()),
            "lock_file": str(lock_file),
            "command": list(command),
            "command_sha256": command_sha256(command),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        append_ledger(
            ledger,
            {**common, "event": "start", "utc": utc_now()},
            pinned_directory_fd=pins.lease_for(ledger),
        )
        try:
            process = subprocess.run(list(command), check=False)
            exit_code = int(process.returncode)
        except BaseException as error:
            append_ledger(
                ledger,
                {
                    **common,
                    "event": "wrapper_exception",
                    "utc": utc_now(),
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                },
                pinned_directory_fd=pins.lease_for(ledger),
            )
            raise
        append_ledger(
            ledger,
            {**common, "event": "end", "utc": utc_now(), "exit_code": exit_code},
            pinned_directory_fd=pins.lease_for(ledger),
        )
        return exit_code


def _validate_new_arguments(
    *,
    usage_ledger: Path | None,
    result_file: Path | None,
    campaign_id: str | None,
    phase: str | None,
    context: Mapping[str, Any] | None,
    invocation_sha256: str | None,
    authorization_path: Path | None,
    authorization_sha256: str | None,
) -> tuple[Path, Path, str, str, dict[str, Any], str, Path | None, str | None]:
    values = (usage_ledger, result_file, campaign_id, phase, context, invocation_sha256)
    if all(value is None for value in values):
        raise ValueError("internal legacy-mode sentinel")
    if any(value is None for value in values):
        raise ValueError(
            "V7 accounting requires --usage-ledger, --result-file, --campaign-id, "
            "--phase, --context-json, and --invocation-sha256 together"
        )
    assert usage_ledger is not None and result_file is not None
    assert campaign_id is not None and phase is not None
    assert context is not None and invocation_sha256 is not None
    if not campaign_id or not phase:
        raise ValueError("campaign and phase must be non-empty")
    if phase not in _EXECUTION_PHASES:
        raise ValueError(f"unsupported V7 GPU execution phase: {phase}")
    if not isinstance(context, Mapping):
        raise ValueError("context must be a JSON object")
    context_value = dict(context)
    budget.canonical_json_bytes(context_value)
    if not (
        len(invocation_sha256) == 64
        and all(character in "0123456789abcdef" for character in invocation_sha256)
    ):
        raise ValueError("invocation SHA-256 must be 64 lowercase hexadecimal characters")
    if (authorization_path is None) != (authorization_sha256 is None):
        raise ValueError(
            "--authorization-path and --authorization-sha256 are required together"
        )
    if phase == "efficiency_benchmark" and authorization_path is None:
        raise ValueError(
            "efficiency_benchmark requires an immutable admitted-child authorization"
        )
    resolved_authorization: Path | None = None
    if authorization_path is not None and authorization_sha256 is not None:
        resolved_authorization = _canonical_no_final_symlink(
            authorization_path, "GPU child authorization"
        )
        _immutable_authorization_binding(
            resolved_authorization, authorization_sha256
        )
    return (
        _canonical_no_final_symlink(usage_ledger, "GPU usage ledger"),
        _canonical_no_final_symlink(result_file, "GPU terminal result"),
        campaign_id,
        phase,
        context_value,
        invocation_sha256,
        resolved_authorization,
        authorization_sha256,
    )


def _run_budgeted(
    lock_file: Path,
    gpu_ledger: Path,
    command: Sequence[str],
    *,
    usage_ledger: Path,
    result_file: Path,
    campaign_id: str,
    phase: str,
    context: Mapping[str, Any],
    invocation_sha256: str,
    authorization_path: Path | None,
    authorization_sha256: str | None,
    budget_ns: int,
) -> int:
    lock_file = _canonical_no_final_symlink(lock_file, "GPU admission lock")
    gpu_ledger = _canonical_no_final_symlink(gpu_ledger, "GPU execution ledger")
    usage_ledger = _canonical_no_final_symlink(usage_ledger, "GPU usage ledger")
    result_file = _canonical_no_final_symlink(result_file, "GPU terminal result")
    authorization = (
        _immutable_authorization_binding(
            authorization_path, authorization_sha256
        )
        if authorization_path is not None and authorization_sha256 is not None
        else None
    )
    _set_child_subreaper()
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("budgeted GPU supervision must run in the main thread")
    protected_paths = (
        lock_file,
        gpu_ledger,
        execution_ledger_lock_path(gpu_ledger),
        usage_ledger,
        budget.ledger_lock_path(usage_ledger),
        result_file,
        budget.result_receipt_lock_path(result_file),
    )
    stable_paths = (
        lock_file,
        execution_ledger_lock_path(gpu_ledger),
        budget.ledger_lock_path(usage_ledger),
        budget.result_receipt_lock_path(result_file),
    )
    with _pinned_protected_tree(
        protected_paths, stable_paths=stable_paths
    ) as pins, _exclusive_gpu_lock(
        lock_file, pinned_directory_fd=pins.lease_for(lock_file)
    ) as admission_lock:
        pins.revalidate()
        usage_directory_fd = pins.fd_for(usage_ledger)
        execution_directory_fd = pins.fd_for(gpu_ledger)
        result_directory_fd = pins.fd_for(result_file)
        usage_directory_lease = pins.lease_for(usage_ledger)
        execution_directory_lease = pins.lease_for(gpu_ledger)
        result_directory_lease = pins.lease_for(result_file)

        def revalidate_admission() -> None:
            pins.revalidate()
            _revalidate_admission_lock_capability(admission_lock)

        expected_genesis = _legacy_expectation(
            usage_ledger, pinned_directory_fd=usage_directory_fd
        )
        budget.cleanup_usage_ledger_replace_residue(
            usage_ledger,
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_genesis,
            pinned_directory_fd=usage_directory_lease,
            admission_revalidate=revalidate_admission,
        )
        cleanup_execution_ledger_replace_residue(
            gpu_ledger,
            pinned_directory_fd=execution_directory_lease,
            admission_revalidate=revalidate_admission,
        )
        revalidate_admission()
        now_real = time.time_ns()
        now_mono = time.monotonic_ns()
        _, state = budget.reconcile_open_reservations(
            usage_ledger,
            realtime_ns=now_real,
            monotonic_ns=now_mono,
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_genesis,
            pinned_directory_fd=usage_directory_lease,
        )
        pins.revalidate()
        _recover_closed_usage_execution_starts(
            gpu_ledger,
            state,
            expected_lock_file=lock_file,
            usage_ledger=usage_ledger,
            pinned_directory_fd=execution_directory_lease,
        )
        pins.revalidate()
        child_digest = budget.command_sha256(command)

        existing_result = budget._read_regular_bytes_at(  # noqa: SLF001
            result_directory_fd,
            result_file.name,
            label="GPU terminal result",
            missing_ok=True,
        )
        if existing_result is not None:
            pins.revalidate()
            recovered = budget.load_validate_terminal_result(
                result_file,
                usage_ledger=usage_ledger,
                expected_campaign_id=campaign_id,
                expected_phase=phase,
                expected_context=context,
                expected_command_sha256=child_digest,
                expected_invocation_sha256=invocation_sha256,
                budget_ns=budget_ns,
                expected_legacy_genesis_sha256=expected_genesis,
                pinned_result_directory_fd=result_directory_lease,
                pinned_usage_directory_fd=usage_directory_fd,
            )
            receipt_terminals = [
                record
                for record in state.records
                if record.get("event") == "terminal"
                and record.get("record_sha256")
                == recovered.get("terminal_record_sha256")
            ]
            if len(receipt_terminals) != 1:
                raise budget.LedgerError(
                    "existing result does not bind one execution terminal"
                )
            receipt_terminal = receipt_terminals[0]
            receipt_reservation = _reservation_for_terminal(
                state, receipt_terminal
            )
            _validate_reservation_authorization(
                receipt_reservation, authorization
            )
            _recover_gpu_execution_end(
                gpu_ledger,
                receipt_terminal,
                reservation=receipt_reservation,
                expected_command=command,
                expected_lock_file=lock_file,
                usage_ledger=usage_ledger,
                pinned_directory_fd=execution_directory_lease,
                require_existing_end=True,
            )
            pins.revalidate()
            return int(recovered["wrapper_exit_code"])

        observed_terminals = [
            record
            for record in state.records
            if _terminal_matches(
                record,
                campaign_id=campaign_id,
                phase=phase,
                context=context,
                invocation_sha256=invocation_sha256,
                child_command_sha256=child_digest,
                result_file=result_file,
            )
        ]
        if len(observed_terminals) > 1:
            raise budget.LedgerError("multiple observed terminals claim one result path")
        if observed_terminals:
            pins.revalidate()
            observed_reservation = _reservation_for_terminal(
                state, observed_terminals[0]
            )
            _validate_reservation_authorization(
                observed_reservation, authorization
            )
            _recover_gpu_execution_end(
                gpu_ledger,
                observed_terminals[0],
                reservation=observed_reservation,
                expected_command=command,
                expected_lock_file=lock_file,
                usage_ledger=usage_ledger,
                pinned_directory_fd=execution_directory_lease,
            )
            pins.revalidate()
            recovered = _publish_result(
                result_file=result_file,
                terminal=observed_terminals[0],
                usage_ledger=usage_ledger,
                gpu_ledger=gpu_ledger,
                pinned_result_directory_fd=result_directory_lease,
                revalidate=pins.revalidate,
            )
            pins.revalidate()
            return int(recovered["wrapper_exit_code"])

        wrapper_pid = os.getpid()
        wrapper_ticks = budget.process_start_ticks(wrapper_pid)
        if wrapper_ticks is None:
            raise budget.LedgerError("cannot establish wrapper process identity")
        lifecycle_id = f"{time.time_ns()}-{wrapper_pid}-{uuid.uuid4().hex}"
        reservation_template = {
            "lifecycle_id": lifecycle_id,
            "campaign_id": campaign_id,
            "phase": phase,
            "context": dict(context),
            "invocation_sha256": invocation_sha256,
            "command_sha256": child_digest,
            "result_path": str(result_file),
            "gpu_execution_ledger_path": str(gpu_ledger),
            "boot_id": budget.boot_id(),
            "wrapper_pid": wrapper_pid,
            "wrapper_start_ticks": wrapper_ticks,
            "wrapper_parent_pid": os.getppid(),
            "hostname": socket.gethostname(),
            "cwd": str(Path.cwd().resolve()),
            "gpu_lock_file": str(lock_file),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "realtime_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
        }
        if authorization is not None:
            wrapper_source = _wrapper_source_binding()
            reservation_template.update(
                {
                    "gpu_lock_st_dev": admission_lock["st_dev"],
                    "gpu_lock_st_ino": admission_lock["st_ino"],
                    "wrapper_source": wrapper_source,
                }
            )
            reservation_template["admitted_child_authorization"] = dict(
                authorization
            )
        reservation, _, _ = budget.reconcile_and_reserve(
            usage_ledger,
            reservation_template,
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_genesis,
            pinned_directory_fd=usage_directory_lease,
        )
        _fault_inject("wrapper_after_reservation_commit")
        pins.revalidate()

        job_id = str(reservation["lifecycle_id"])
        execution_common = {
            "schema_version": 1,
            "job_id": job_id,
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
            "command": list(command),
            "command_sha256": reservation["command_sha256"],
            "cuda_visible_devices": reservation["cuda_visible_devices"],
        }
        append_ledger(
            gpu_ledger,
            {**execution_common, "event": "start", "utc": utc_now()},
            pinned_directory_fd=execution_directory_lease,
        )
        _fault_inject("wrapper_after_execution_start_commit")
        pins.revalidate()

        watchdog_process: subprocess.Popen[Any] | None = None
        child_pid: int | None = None
        child_ticks: int | None = None
        heartbeat: dict[str, Any] | None = None
        heartbeat_sequence = 0
        received_signal: int | None = None
        hard_timeout = False
        termination_escalated = False
        containment_anomaly = False
        workload_ended_monotonic_ns: int | None = None
        workload_session_id: int | None = None
        pending_error: BaseException | None = None
        cleanup_failure: BaseException | None = None
        prior_handlers: dict[int, Any] = {}
        ready_read_fd = -1
        ready_write_fd = -1
        final_read_fd = -1
        final_write_fd = -1
        binding_read_fd = -1
        binding_write_fd = -1

        def forward_signal(signum: int, _frame: Any) -> None:
            nonlocal received_signal
            if received_signal is None:
                received_signal = int(signum)
            if workload_session_id is not None:
                try:
                    _signal_session(
                        workload_session_id,
                        int(signum),
                        exclude=(
                            watchdog_process.pid
                            if watchdog_process is not None
                            else -1,
                        ),
                    )
                except RuntimeError:
                    pass
            if watchdog_process is not None and watchdog_process.poll() is None:
                try:
                    os.kill(watchdog_process.pid, int(signum))
                except ProcessLookupError:
                    pass

        for forwarded in _FORWARDED_SIGNALS:
            prior_handlers[int(forwarded)] = signal.getsignal(forwarded)
            signal.signal(forwarded, forward_signal)

        child_return_code = 127
        try:
            try:
                ready_read_fd, ready_write_fd = os.pipe()
                final_read_fd, final_write_fd = os.pipe()
                if authorization is not None:
                    binding_read_fd, binding_write_fd = os.pipe()
                reservation_started = int(reservation["monotonic_ns"])
                workload_deadline = reservation_started + int(
                    reservation["workload_timeout_ns"]
                )
                watchdog_spec = {
                    "wrapper_pid": wrapper_pid,
                    "ready_fd": ready_write_fd,
                    "final_fd": final_write_fd,
                    "command": list(command),
                    "workload_deadline_ns": workload_deadline,
                    "termination_grace_ns": budget.TERMINATION_GRACE_NS,
                    "accounting_margin_ns": budget.ACCOUNTING_MARGIN_NS,
                    "protected_directory_pins": pins.watchdog_pin_spec(),
                    "admitted_child_fd": (
                        binding_read_fd if authorization is not None else None
                    ),
                    "admission_lock_capability": (
                        dict(admission_lock) if authorization is not None else None
                    ),
                    "wrapper_start_ticks": (
                        wrapper_ticks if authorization is not None else None
                    ),
                    "wrapper_source": (
                        dict(reservation["wrapper_source"])
                        if authorization is not None
                        else None
                    ),
                }
                pins.revalidate()
                watchdog_pass_fds = tuple(
                    sorted(
                        {
                            ready_write_fd,
                            final_write_fd,
                            *(
                                (binding_read_fd, int(admission_lock["fd"]))
                                if authorization is not None
                                else ()
                            ),
                            *pins.watchdog_pass_fds(),
                        }
                    )
                )
                watchdog_process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--internal-watchdog-json",
                        budget.canonical_json_bytes(watchdog_spec).decode("utf-8"),
                    ],
                    pass_fds=watchdog_pass_fds,
                    close_fds=True,
                )
                workload_session_id = watchdog_process.pid
                if binding_read_fd >= 0:
                    os.close(binding_read_fd)
                    binding_read_fd = -1
                os.close(ready_write_fd)
                ready_write_fd = -1
                os.close(final_write_fd)
                final_write_fd = -1
                ready_status = _read_pipe_json(
                    ready_read_fd,
                    timeout_seconds=max(
                        1.0,
                        min(
                            30.0,
                            int(reservation["workload_timeout_ns"])
                            / 1_000_000_000,
                        ),
                    ),
                )
                os.close(ready_read_fd)
                ready_read_fd = -1
                child_pid_value = ready_status.get("child_pid")
                child_ticks_value = ready_status.get("child_start_ticks")
                if type(child_pid_value) is int and child_pid_value > 0:
                    child_pid = child_pid_value
                if type(child_ticks_value) is int and child_ticks_value > 0:
                    child_ticks = child_ticks_value
                if ready_status.get("exception") is not None:
                    raise RuntimeError(
                        f"GPU watchdog workload spawn failed: {ready_status['exception']}"
                    )
                if child_pid is None or child_ticks is None:
                    raise budget.LedgerError("watchdog returned invalid child identity")
                if ready_status.get("workload_session_id") != workload_session_id:
                    raise budget.LedgerError("watchdog session binding drifted")
                if ready_status.get("workload_deadline_ns") != workload_deadline:
                    raise budget.LedgerError("watchdog deadline binding drifted")
                if authorization is not None:
                    watchdog_ticks = budget.process_start_ticks(
                        watchdog_process.pid
                    )
                    if watchdog_ticks is None:
                        raise budget.LedgerError(
                            "cannot establish admitted-child watchdog identity"
                        )
                    binding = _build_admitted_child_binding(
                        reservation=reservation,
                        usage_ledger=usage_ledger,
                        execution_ledger=gpu_ledger,
                        authorization=authorization,
                        watchdog_pid=watchdog_process.pid,
                        watchdog_start_ticks=watchdog_ticks,
                        child_pid=child_pid,
                        child_start_ticks=child_ticks,
                        admission_lock=admission_lock,
                    )
                    _write_pipe_json(binding_write_fd, binding)
                    os.close(binding_write_fd)
                    binding_write_fd = -1
                if received_signal is not None:
                    _signal_session(
                        workload_session_id,
                        received_signal,
                        exclude=(watchdog_process.pid,),
                    )
                    os.kill(watchdog_process.pid, received_signal)
                next_heartbeat = reservation_started + budget.HEARTBEAT_INTERVAL_NS

                while watchdog_process.poll() is None:
                    pins.revalidate()
                    now = time.monotonic_ns()
                    if now >= next_heartbeat:
                        heartbeat_sequence += 1
                        ceiling = min(
                            int(reservation["reservation_ns"]),
                            max(0, now - reservation_started),
                        )
                        heartbeat = budget.append_heartbeat(
                            usage_ledger,
                            reservation,
                            sequence=heartbeat_sequence,
                            elapsed_ceiling_ns=ceiling,
                            realtime_ns=time.time_ns(),
                            monotonic_ns=now,
                            child_pid=child_pid,
                            child_start_ticks=child_ticks,
                            budget_ns=budget_ns,
                            expected_legacy_genesis_sha256=expected_genesis,
                            pinned_directory_fd=usage_directory_lease,
                        )
                        pins.revalidate()
                        next_heartbeat = now + budget.HEARTBEAT_INTERVAL_NS
                    wait_ns = 200_000_000
                    for deadline in (next_heartbeat, workload_deadline):
                        if deadline is not None:
                            wait_ns = min(wait_ns, max(1_000_000, deadline - now))
                    try:
                        watchdog_process.wait(timeout=wait_ns / 1_000_000_000)
                    except subprocess.TimeoutExpired:
                        pass
                watchdog_exit = int(watchdog_process.wait())
                final_status = _read_pipe_json(final_read_fd, timeout_seconds=2.0)
                os.close(final_read_fd)
                final_read_fd = -1
                if (
                    final_status.get("child_pid") != child_pid
                    or final_status.get("child_start_ticks") != child_ticks
                    or final_status.get("workload_session_id")
                    != workload_session_id
                    or type(final_status.get("return_code")) is not int
                    or type(final_status.get("hard_timeout_reached")) is not bool
                    or type(final_status.get("termination_escalated")) is not bool
                    or type(final_status.get("containment_anomaly")) is not bool
                    or final_status.get("containment_empty") is not True
                    or type(final_status.get("ended_monotonic_ns")) is not int
                ):
                    raise budget.LedgerError("GPU watchdog final identity/status drifted")
                child_return_code = int(final_status["return_code"])
                hard_timeout = bool(final_status["hard_timeout_reached"])
                termination_escalated = bool(final_status["termination_escalated"])
                containment_anomaly = bool(final_status["containment_anomaly"])
                workload_ended_monotonic_ns = int(
                    final_status["ended_monotonic_ns"]
                )
                if watchdog_exit != 0 or final_status.get("exception") is not None:
                    raise RuntimeError(
                        f"GPU watchdog failed closed: {final_status.get('exception')}"
                    )
                if _session_members(workload_session_id):
                    raise RuntimeError("workload session is nonempty after watchdog exit")
                pins.revalidate()
            except BaseException as error:
                pending_error = error
                containment_anomaly = watchdog_process is not None
                if watchdog_process is not None and workload_session_id is not None:
                    try:
                        termination_escalated = (
                            _independent_cleanup_after_watchdog_failure(
                                workload_session_id,
                                watchdog_process,
                                grace_ns=budget.TERMINATION_GRACE_NS,
                                margin_ns=budget.ACCOUNTING_MARGIN_NS,
                            )
                            or termination_escalated
                        )
                    except BaseException as cleanup_error:
                        cleanup_failure = cleanup_error
                if final_read_fd >= 0:
                    try:
                        final_status = _read_pipe_json(
                            final_read_fd, timeout_seconds=0.2
                        )
                    except (OSError, RuntimeError):
                        pass
                    else:
                        if type(final_status.get("return_code")) is int:
                            child_return_code = int(final_status["return_code"])
                        if type(final_status.get("hard_timeout_reached")) is bool:
                            hard_timeout = bool(
                                final_status["hard_timeout_reached"]
                            )
                        if type(final_status.get("termination_escalated")) is bool:
                            termination_escalated = (
                                bool(final_status["termination_escalated"])
                                or termination_escalated
                            )
                        if type(final_status.get("containment_anomaly")) is bool:
                            containment_anomaly = (
                                bool(final_status["containment_anomaly"])
                                or containment_anomaly
                            )
                        if type(final_status.get("ended_monotonic_ns")) is int:
                            workload_ended_monotonic_ns = int(
                                final_status["ended_monotonic_ns"]
                            )
                workload_ended_monotonic_ns = max(
                    workload_ended_monotonic_ns or 0, time.monotonic_ns()
                )
                if workload_session_id is not None:
                    try:
                        remaining = _session_members(workload_session_id)
                    except BaseException as containment_error:
                        cleanup_failure = cleanup_failure or containment_error
                    else:
                        if remaining:
                            cleanup_failure = cleanup_failure or RuntimeError(
                                f"workload session remains live: {remaining}"
                            )

            if cleanup_failure is not None:
                raise cleanup_failure
            pins.revalidate()

            ended_mono = (
                workload_ended_monotonic_ns
                if workload_ended_monotonic_ns is not None
                else time.monotonic_ns()
            )
            elapsed_ns = max(0, ended_mono - int(reservation["monotonic_ns"]))
            charged_ns = min(int(reservation["reservation_ns"]), elapsed_ns)
            deadline_breached = elapsed_ns > int(reservation["reservation_ns"])
            if deadline_breached:
                wrapper_exit = 125
            elif received_signal is not None or hard_timeout:
                wrapper_exit = _normalize_exit_code(
                    child_return_code,
                    received_signal=received_signal,
                    timed_out=hard_timeout,
                )
            elif (
                pending_error is not None
                or termination_escalated
                or containment_anomaly
            ):
                wrapper_exit = 125
            else:
                wrapper_exit = _normalize_exit_code(
                    child_return_code,
                    received_signal=None,
                    timed_out=False,
                )
            exception_value = (
                {"type": type(pending_error).__name__, "message": str(pending_error)}
                if pending_error is not None
                else None
            )
            terminal = budget.append_terminal(
                usage_ledger,
                reservation,
                last_heartbeat=heartbeat,
                elapsed_ns=elapsed_ns,
                charged_usage_ns=charged_ns,
                realtime_ns=time.time_ns(),
                monotonic_ns=ended_mono,
                child_pid=child_pid,
                child_start_ticks=child_ticks,
                return_code=child_return_code,
                wrapper_exit_code=wrapper_exit,
                hard_timeout_reached=hard_timeout,
                received_signal=received_signal,
                termination_escalated=termination_escalated,
                containment_anomaly=containment_anomaly,
                exception=exception_value,
                budget_ns=budget_ns,
                expected_legacy_genesis_sha256=expected_genesis,
                pinned_directory_fd=usage_directory_lease,
            )
            _fault_inject("wrapper_after_terminal_commit")
            pins.revalidate()
            append_ledger(
                gpu_ledger,
                {
                    **execution_common,
                    "event": "end",
                    "utc": utc_now(),
                    "exit_code": child_return_code,
                    "wrapper_exit_code": wrapper_exit,
                    "hard_timeout_reached": hard_timeout,
                    "termination_escalated": termination_escalated,
                    "containment_anomaly": containment_anomaly,
                    "reservation_deadline_breached": terminal[
                        "reservation_deadline_breached"
                    ],
                    "terminal_record_sha256": terminal["record_sha256"],
                },
                pinned_directory_fd=execution_directory_lease,
            )
            _fault_inject("wrapper_after_execution_end_commit")
            pins.revalidate()
            _publish_result(
                result_file=result_file,
                terminal=terminal,
                usage_ledger=usage_ledger,
                gpu_ledger=gpu_ledger,
                pinned_result_directory_fd=result_directory_lease,
                revalidate=pins.revalidate,
            )
            _fault_inject("wrapper_after_result_publication")
            pins.revalidate()
            if pending_error is not None:
                raise pending_error
            return wrapper_exit
        finally:
            for descriptor in (
                ready_read_fd,
                ready_write_fd,
                final_read_fd,
                final_write_fd,
                binding_read_fd,
                binding_write_fd,
            ):
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            for forwarded, previous in prior_handlers.items():
                signal.signal(forwarded, previous)


def run(
    lock_file: Path,
    ledger: Path,
    command: Sequence[str],
    *,
    usage_ledger: Path | None = None,
    result_file: Path | None = None,
    campaign_id: str | None = None,
    phase: str | None = None,
    context: Mapping[str, Any] | None = None,
    invocation_sha256: str | None = None,
    authorization_path: Path | None = None,
    authorization_sha256: str | None = None,
    budget_ns: int = budget.GPU_BUDGET_NS,
) -> int:
    if not command:
        raise ValueError("a command is required after --")
    if all(
        value is None
        for value in (
            usage_ledger,
            result_file,
            campaign_id,
            phase,
            context,
            invocation_sha256,
            authorization_path,
            authorization_sha256,
        )
    ):
        return _run_legacy(lock_file, ledger, command)
    validated = _validate_new_arguments(
        usage_ledger=usage_ledger,
        result_file=result_file,
        campaign_id=campaign_id,
        phase=phase,
        context=context,
        invocation_sha256=invocation_sha256,
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha256,
    )
    return _run_budgeted(
        lock_file,
        ledger,
        command,
        usage_ledger=validated[0],
        result_file=validated[1],
        campaign_id=validated[2],
        phase=validated[3],
        context=validated[4],
        invocation_sha256=validated[5],
        authorization_path=validated[6],
        authorization_sha256=validated[7],
        budget_ns=budget_ns,
    )


def _unique_cli_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key {key}")
        value[key] = item
    return value


def _parse_context(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_cli_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite context value: {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"invalid --context-json: {error}") from error
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("--context-json must encode an object")
    budget.canonical_json_bytes(value)
    return value


def _parse_budget_seconds(value: str) -> int:
    try:
        nanoseconds = Decimal(value) * Decimal(1_000_000_000)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("invalid --budget-seconds") from error
    if nanoseconds != Decimal(budget.GPU_BUDGET_NS):
        raise argparse.ArgumentTypeError("--budget-seconds is fixed at 36000")
    return int(nanoseconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--usage-ledger", type=Path)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--phase")
    parser.add_argument("--context-json")
    parser.add_argument("--invocation-sha256")
    parser.add_argument("--authorization-path", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument(
        "--budget-seconds",
        type=_parse_budget_seconds,
        default=budget.GPU_BUDGET_NS,
        help="fixed campaign budget; only 36000 is accepted",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    try:
        context = _parse_context(args.context_json)
        return run(
            args.lock_file,
            args.ledger,
            command,
            usage_ledger=args.usage_ledger,
            result_file=args.result_file,
            campaign_id=args.campaign_id,
            phase=args.phase,
            context=context,
            invocation_sha256=args.invocation_sha256,
            authorization_path=args.authorization_path,
            authorization_sha256=args.authorization_sha256,
            budget_ns=args.budget_seconds,
        )
    except budget.BudgetExhausted as error:
        print(str(error), file=sys.stderr)
        return 75
    except (
        budget.LedgerError,
        RuntimeError,
        ValueError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 73


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--internal-watchdog-json":
        raise SystemExit(_watchdog_main(sys.argv[2]))
    enable_wrapper_parent_death_containment()
    raise SystemExit(main())
