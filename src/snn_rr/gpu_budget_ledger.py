"""Crash-safe, hash-chained GPU budget lifecycle accounting.

The usage ledger is an immutable byte prefix.  Every mutation is serialized by
the stable sibling ``.lock`` inode, validates the complete old chain, writes
that exact prefix plus new canonical records to a same-directory temporary
file, fsyncs it, atomically replaces the ledger, and fsyncs the directory.

Schema-v1 records are retained only as legacy settled-usage records.  Schema-v2
records implement reservation, heartbeat, terminal, and reconciled-terminal
lifecycles.  All schema-v2 accounting is integer nanoseconds.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import ctypes
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 2
GPU_BUDGET_NS = 36_000_000_000_000
HEARTBEAT_INTERVAL_NS = 15_000_000_000
TERMINATION_GRACE_NS = 10_000_000_000
ACCOUNTING_MARGIN_NS = 5_000_000_000
RECOVERY_MARGIN_NS = 30_000_000_000
LEGACY_V1_GENESIS_RECORD_SHA256 = "c7b463e4db2e8d475f428dc61dfc8fa0d27910f62fb2d811e2845fcf4932e035"

_AT_EMPTY_PATH = 0x1000
_STATE_REPLACE_TEMP_SUFFIX = ".v8r4a-replace.tmp"
_FAULT_INJECTION_HOOK: Callable[[str], None] | None = None

_HEX = frozenset("0123456789abcdef")
_V2_EVENTS = frozenset(
    {"reservation", "heartbeat", "terminal", "reconciled_terminal"}
)


class LedgerError(RuntimeError):
    """The GPU budget ledger is missing, corrupt, forked, or inconsistent."""


class BudgetExhausted(LedgerError):
    """No safe workload reservation remains under the fixed GPU budget."""


class LedgerBusy(LedgerError):
    """An open lifecycle still belongs to the same live wrapper process."""


def _fault_inject(point: str) -> None:
    """Invoke the process-local deterministic crash-test hook, when installed."""

    hook = _FAULT_INJECTION_HOOK
    if hook is not None:
        hook(point)


@dataclass(frozen=True)
class OpenReservation:
    """Reduced state for one reservation that has no terminal record yet."""

    reservation: dict[str, Any]
    last_heartbeat: dict[str, Any] | None

    @property
    def reservation_ns(self) -> int:
        return int(self.reservation["reservation_ns"])


@dataclass(frozen=True)
class LedgerState:
    """Fully verified reduction of the usage ledger."""

    records: tuple[dict[str, Any], ...]
    settled_usage_ns: int
    open_reservations: Mapping[str, OpenReservation]
    budget_ns: int
    tail_sha256: str | None
    raw_bytes: bytes

    @property
    def open_reservation_ns(self) -> int:
        return sum(item.reservation_ns for item in self.open_reservations.values())

    @property
    def remaining_ns(self) -> int:
        return self.budget_ns - self.settled_usage_ns - self.open_reservation_ns


@dataclass(frozen=True)
class _LockedLedger:
    path: Path
    lock_path: Path
    directory_fd: int
    lock_fd: int
    directory_identity: tuple[int, int]
    lock_identity: tuple[int, int]
    owns_directory_fd: bool
    owns_directory_fence: bool

    def revalidate(self) -> None:
        _canonical_no_final_symlink(self.path, "GPU usage ledger")
        _path_names_open_directory(
            self.path.parent, self.directory_fd, "GPU usage ledger parent"
        )
        directory_status = os.fstat(self.directory_fd)
        if (directory_status.st_dev, directory_status.st_ino) != self.directory_identity:
            raise LedgerError("GPU usage ledger directory descriptor identity drifted")
        try:
            lock_status = os.stat(
                self.lock_path.name,
                dir_fd=self.directory_fd,
                follow_symlinks=False,
            )
        except OSError as error:
            raise LedgerError(f"GPU usage ledger lock identity is unavailable: {error}") from error
        descriptor_status = os.fstat(self.lock_fd)
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or descriptor_status.st_nlink != 1
            or (descriptor_status.st_dev, descriptor_status.st_ino)
            != self.lock_identity
            or not _same_inode(lock_status, descriptor_status)
        ):
            raise LedgerError("GPU usage ledger lock inode changed while held")


@dataclass(eq=False)
class _DirectoryGenerationLease:
    """Opaque, creator-bound proof of one live directory generation fence."""

    directory_fd: int
    fence_fd: int
    identity: tuple[int, int]
    creator_pid: int
    nonce: object
    mutex: threading.RLock = field(default_factory=threading.RLock, repr=False)
    state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    borrow_count: int = 0
    released: bool = False


_ACTIVE_DIRECTORY_LEASES: set[_DirectoryGenerationLease] = set()
_DIRECTORY_LEASE_REGISTRY_LOCK = threading.Lock()


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one accepted JSON representation for hashes and JSONL."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LedgerError(f"value is not canonical finite JSON: {error}") from error


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def command_sha256(command: Sequence[str]) -> str:
    if not command or any(not isinstance(item, str) for item in command):
        raise LedgerError("GPU workload command must be a non-empty string sequence")
    return semantic_sha256(list(command))


def _canonical_no_final_symlink(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, component in enumerate(parts):
        current = current / component
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as error:
            raise LedgerError(f"cannot inspect {label} component {current}: {error}") from error
        if stat.S_ISLNK(status.st_mode):
            raise LedgerError(f"{label} has a symlinked component: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(status.st_mode):
            raise LedgerError(f"{label} ancestor is not a directory: {current}")
    return absolute


def _open_directory_nofollow(path: Path, label: str) -> int:
    """Open an absolute directory component-by-component without symlinks."""

    absolute = _canonical_no_final_symlink(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _path_names_open_directory(path: Path, descriptor: int, label: str) -> None:
    check = _open_directory_nofollow(path, label)
    try:
        if not _same_inode(os.fstat(check), os.fstat(descriptor)):
            raise LedgerError(f"{label} directory identity changed: {path}")
    finally:
        os.close(check)


def open_pinned_directory(path: Path, *, label: str = "protected directory") -> int:
    """Return a no-follow directory descriptor whose inode callers can pin."""

    return _open_directory_nofollow(path, label)


def validate_pinned_directory(
    path: Path, descriptor: int, *, label: str = "protected directory"
) -> None:
    """Fail if ``path`` no longer names the directory held by ``descriptor``."""

    _path_names_open_directory(path, descriptor, label)


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _close_fd_fail_safe(
    descriptor: int,
    *,
    expected_identity: tuple[int, int] | None = None,
    label: str = "descriptor",
) -> None:
    """Close once normally, then use libc only when the same fd stayed open."""

    try:
        os.close(descriptor)
        return
    except BaseException as first_error:
        try:
            status = os.fstat(descriptor)
        except OSError as probe_error:
            if probe_error.errno == errno.EBADF:
                raise LedgerError(f"{label} close reported failure after closing") from first_error
            raise LedgerError(f"cannot verify {label} after close failure") from first_error
        if expected_identity is not None and (
            status.st_dev,
            status.st_ino,
        ) != expected_identity:
            raise LedgerError(f"{label} fd was reused after close failure") from first_error
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.close(int(descriptor)) != 0:
            error_number = ctypes.get_errno()
            raise LedgerError(
                f"cannot close {label}: {os.strerror(error_number)}"
            ) from first_error
        raise LedgerError(f"{label} close required fail-safe fallback") from first_error


def _unlock_and_close_flocked_fd(
    descriptor: int,
    *,
    expected_identity: tuple[int, int],
    label: str,
) -> None:
    """Attempt unlock and close independently; close still releases the flock."""

    errors: list[BaseException] = []
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except BaseException as error:
        errors.append(error)
    try:
        _close_fd_fail_safe(
            descriptor,
            expected_identity=expected_identity,
            label=label,
        )
    except BaseException as error:
        errors.append(error)
    if errors:
        raise LedgerError(
            f"{label} cleanup failed: " + "; ".join(str(error) for error in errors)
        ) from errors[0]


def acquire_directory_generation_fence(descriptor: int) -> _DirectoryGenerationLease:
    """Acquire a fence and return its creator-bound mutation lease."""

    status = os.fstat(descriptor)
    identity = (status.st_dev, status.st_ino)
    if not stat.S_ISDIR(status.st_mode):
        raise LedgerError("directory generation fence target is not a directory")
    with _DIRECTORY_LEASE_REGISTRY_LOCK:
        if any(
            item.creator_pid == os.getpid()
            and item.directory_fd == descriptor
            and not item.released
            for item in _ACTIVE_DIRECTORY_LEASES
        ):
            raise LedgerError("directory generation fence descriptor is already leased")
    fence_fd = -1
    try:
        fence_fd = os.open(".", _directory_open_flags(), dir_fd=descriptor)
        fence_status = os.fstat(fence_fd)
        if not stat.S_ISDIR(fence_status.st_mode) or (
            fence_status.st_dev,
            fence_status.st_ino,
        ) != identity:
            raise LedgerError("directory generation fence identity drifted while opening")
        fcntl.flock(fence_fd, fcntl.LOCK_EX)
    except BaseException as error:
        cleanup_error: BaseException | None = None
        if fence_fd >= 0:
            try:
                _close_fd_fail_safe(
                    fence_fd,
                    expected_identity=identity,
                    label="directory generation fence",
                )
            except BaseException as close_error:
                cleanup_error = close_error
        if cleanup_error is not None:
            raise LedgerError(
                f"cannot acquire directory generation fence; cleanup failed: {cleanup_error}"
            ) from error
        if isinstance(error, LedgerError):
            raise
        raise LedgerError(f"cannot acquire directory generation fence: {error}") from error
    lease = _DirectoryGenerationLease(
        directory_fd=descriptor,
        fence_fd=fence_fd,
        identity=identity,
        creator_pid=os.getpid(),
        nonce=object(),
    )
    try:
        with _DIRECTORY_LEASE_REGISTRY_LOCK:
            _ACTIVE_DIRECTORY_LEASES.add(lease)
    except BaseException as error:
        try:
            _unlock_and_close_flocked_fd(
                fence_fd,
                expected_identity=identity,
                label="unregistered directory generation fence",
            )
        except BaseException as cleanup_error:
            raise LedgerError(
                "cannot register directory generation fence; cleanup failed: "
                f"{cleanup_error}"
            ) from error
        raise
    return lease


def _validate_directory_generation_lease(lease: object) -> _DirectoryGenerationLease:
    if not isinstance(lease, _DirectoryGenerationLease):
        raise LedgerError("pinned directory mutation requires an active generation lease")
    if lease.creator_pid != os.getpid():
        raise LedgerError("directory generation lease cannot cross a fork boundary")
    with _DIRECTORY_LEASE_REGISTRY_LOCK:
        active = lease in _ACTIVE_DIRECTORY_LEASES
    with lease.state_lock:
        released = lease.released
    if not active or released:
        raise LedgerError("pinned directory mutation requires an active generation lease")
    status = os.fstat(lease.directory_fd)
    fence_status = os.fstat(lease.fence_fd)
    if (
        not stat.S_ISDIR(status.st_mode)
        or not stat.S_ISDIR(fence_status.st_mode)
        or (status.st_dev, status.st_ino) != lease.identity
        or (fence_status.st_dev, fence_status.st_ino) != lease.identity
    ):
        raise LedgerError("directory generation lease identity drifted")
    require_directory_generation_fence(lease.directory_fd)
    return lease


def directory_generation_lease_fd(lease: object) -> int:
    """Return the fd bound to one live lease; reject raw/fork impersonation."""

    return _validate_directory_generation_lease(lease).directory_fd


@contextmanager
def borrow_directory_generation_lease(lease: object) -> Iterator[int]:
    """Serialize a complete mutation transaction using one active lease."""

    validated = _validate_directory_generation_lease(lease)
    validated.mutex.acquire()
    borrowed = False
    try:
        _validate_directory_generation_lease(validated)
        with validated.state_lock:
            if validated.released:
                raise LedgerError("directory generation lease was released")
            validated.borrow_count += 1
            borrowed = True
        yield validated.directory_fd
    finally:
        if borrowed:
            with validated.state_lock:
                validated.borrow_count -= 1
        validated.mutex.release()


def release_directory_generation_fence(lease: object) -> None:
    # A release must not allocate a probe descriptor or depend on pathname
    # validation: either operation can fail, but must never strand the flock
    # already represented by an active creator-bound capability.
    if not isinstance(lease, _DirectoryGenerationLease):
        raise LedgerError("directory generation fence release requires an active lease")
    validated = lease
    if validated.creator_pid != os.getpid():
        raise LedgerError("directory generation lease cannot cross a fork boundary")
    with _DIRECTORY_LEASE_REGISTRY_LOCK:
        active = validated in _ACTIVE_DIRECTORY_LEASES
    with validated.state_lock:
        released = validated.released
    if not active or released:
        raise LedgerError("directory generation fence release requires an active lease")
    if not validated.mutex.acquire(blocking=False):
        raise LedgerError("directory generation lease is borrowed by another thread")
    errors: list[BaseException] = []
    closed = False
    try:
        with validated.state_lock:
            if validated.borrow_count:
                raise LedgerError("directory generation lease is still borrowed")
        try:
            fcntl.flock(validated.fence_fd, fcntl.LOCK_UN)
        except BaseException as error:
            errors.append(error)
        try:
            _close_fd_fail_safe(
                validated.fence_fd,
                expected_identity=validated.identity,
                label="directory generation fence",
            )
            closed = True
        except BaseException as error:
            errors.append(error)
            try:
                os.fstat(validated.fence_fd)
            except OSError as probe_error:
                if probe_error.errno == errno.EBADF:
                    closed = True
        if closed:
            with validated.state_lock:
                validated.released = True
            with _DIRECTORY_LEASE_REGISTRY_LOCK:
                _ACTIVE_DIRECTORY_LEASES.discard(validated)
    finally:
        validated.mutex.release()
    if errors:
        raise LedgerError(
            "directory generation fence cleanup failed: "
            + "; ".join(str(error) for error in errors)
        ) from errors[0]


def require_directory_generation_fence(descriptor: int) -> None:
    """Fail unless another open description already fences this directory.

    Pinned-directory mutation APIs deliberately do not acquire or release the
    caller's transaction-wide fence.  Probe the inode through a distinct open
    description so an unfenced raw descriptor cannot silently bypass the
    generation protocol.
    """

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        probe = os.open(".", flags, dir_fd=descriptor)
    except OSError as error:
        raise LedgerError(f"cannot probe directory generation fence: {error}") from error
    probe_status = os.fstat(probe)
    probe_identity = (probe_status.st_dev, probe_status.st_ino)
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        except OSError as error:
            raise LedgerError(
                f"cannot verify directory generation fence: {error}"
            ) from error
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise LedgerError("pinned directory is not generation fenced")
    finally:
        _close_fd_fail_safe(
            probe,
            expected_identity=probe_identity,
            label="directory generation fence probe",
        )


def ledger_lock_path(path: Path) -> Path:
    """Return the fixed stable lock inode path for a replace-on-append ledger."""

    resolved = _canonical_no_final_symlink(path, "GPU usage ledger")
    return resolved.with_name(resolved.name + ".lock")


def result_receipt_lock_path(path: Path) -> Path:
    """Return the stable generation lock used for create-once receipts."""

    resolved = _canonical_no_final_symlink(path, "GPU terminal result")
    return resolved.with_name(resolved.name + ".lock")


def _is_hex_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise LedgerError(f"{label} must be an integer >= {minimum}")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LedgerError(f"{label} must be a non-empty string")
    return value


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LedgerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _legacy_elapsed_ns(record: Mapping[str, Any], line_number: int) -> int:
    value = record.get("elapsed_seconds", 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(f"legacy ledger line {line_number} has invalid elapsed_seconds")
    if isinstance(value, float) and not math.isfinite(value):
        raise LedgerError(f"legacy ledger line {line_number} has non-finite elapsed_seconds")
    try:
        nanoseconds = Decimal(str(value)) * Decimal(1_000_000_000)
    except InvalidOperation as error:
        raise LedgerError(
            f"legacy ledger line {line_number} has invalid elapsed_seconds"
        ) from error
    if nanoseconds < 0 or nanoseconds != nanoseconds.to_integral_value():
        raise LedgerError(
            f"legacy ledger line {line_number} is not exact integer nanoseconds"
        )
    return int(nanoseconds)


def _identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("campaign_id"),
        record.get("phase"),
        record.get("context"),
        record.get("invocation_sha256"),
        record.get("command_sha256"),
        record.get("result_path"),
        record.get("gpu_execution_ledger_path"),
        record.get("boot_id"),
        record.get("wrapper_pid"),
        record.get("wrapper_start_ticks"),
    )


def _validate_v2_common(record: Mapping[str, Any], line_number: int) -> None:
    event = record.get("event")
    if event not in _V2_EVENTS:
        raise LedgerError(f"ledger line {line_number} has unknown schema-v2 event")
    _require_text(record.get("lifecycle_id"), f"ledger line {line_number} lifecycle_id")
    _require_text(record.get("campaign_id"), f"ledger line {line_number} campaign_id")
    _require_text(record.get("phase"), f"ledger line {line_number} phase")
    if not isinstance(record.get("context"), dict):
        raise LedgerError(f"ledger line {line_number} context must be an object")
    canonical_json_bytes(record["context"])
    for field in ("invocation_sha256", "command_sha256"):
        if not _is_hex_sha256(record.get(field)):
            raise LedgerError(f"ledger line {line_number} has invalid {field}")
    _require_text(record.get("result_path"), f"ledger line {line_number} result_path")
    if not Path(str(record["result_path"])).is_absolute():
        raise LedgerError(f"ledger line {line_number} result_path is not absolute")
    _require_text(
        record.get("gpu_execution_ledger_path"),
        f"ledger line {line_number} gpu_execution_ledger_path",
    )
    if not Path(str(record["gpu_execution_ledger_path"])).is_absolute():
        raise LedgerError(
            f"ledger line {line_number} gpu_execution_ledger_path is not absolute"
        )
    _require_text(record.get("boot_id"), f"ledger line {line_number} boot_id")
    _require_int(record.get("wrapper_pid"), f"ledger line {line_number} wrapper_pid", minimum=1)
    _require_int(
        record.get("wrapper_start_ticks"),
        f"ledger line {line_number} wrapper_start_ticks",
        minimum=1,
    )
    _require_int(record.get("realtime_ns"), f"ledger line {line_number} realtime_ns")
    _require_int(record.get("monotonic_ns"), f"ledger line {line_number} monotonic_ns")


def _same_lifecycle_identity(
    record: Mapping[str, Any], reservation: Mapping[str, Any], line_number: int
) -> None:
    if _identity(record) != _identity(reservation):
        raise LedgerError(f"ledger line {line_number} lifecycle identity drifted")
    if record.get("reservation_record_sha256") != reservation.get("record_sha256"):
        raise LedgerError(f"ledger line {line_number} reservation binding drifted")


def _decode_lines(raw: bytes) -> list[dict[str, Any]]:
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise LedgerError("GPU usage ledger has a torn non-newline-terminated tail")
    records: list[dict[str, Any]] = []
    for number, raw_line in enumerate(raw.splitlines(keepends=True), 1):
        if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
            raise LedgerError(
                f"GPU usage ledger line {number} is not canonical newline JSON"
            )
        encoded = raw_line[:-1]
        if not encoded:
            raise LedgerError(f"GPU usage ledger line {number} is empty")
        try:
            line = encoded.decode("utf-8", errors="strict")
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    LedgerError(f"non-finite JSON constant: {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, LedgerError) as error:
            raise LedgerError(
                f"GPU usage ledger line {number} is invalid canonical JSON: {error}"
            ) from error
        if not isinstance(value, dict):
            raise LedgerError(f"GPU usage ledger line {number} is not an object")
        if raw_line != canonical_json_bytes(value) + b"\n":
            raise LedgerError(f"GPU usage ledger line {number} is not canonical JSON")
        records.append(value)
    return records


def verify_ledger_bytes(
    raw: bytes,
    *,
    budget_ns: int = GPU_BUDGET_NS,
    expected_legacy_genesis_sha256: str | None = None,
) -> LedgerState:
    """Verify and reduce an exact ledger byte stream."""

    _require_int(budget_ns, "budget_ns", minimum=1)
    if expected_legacy_genesis_sha256 is not None and not _is_hex_sha256(
        expected_legacy_genesis_sha256
    ):
        raise LedgerError("expected legacy genesis hash is invalid")
    decoded = _decode_lines(raw)
    previous: str | None = None
    settled_ns = 0
    open_items: dict[str, OpenReservation] = {}
    seen_lifecycles: set[str] = set()
    saw_v2 = False
    records: list[dict[str, Any]] = []

    for number, original in enumerate(decoded, 1):
        record = dict(original)
        recorded_hash = record.pop("record_sha256", None)
        if not _is_hex_sha256(recorded_hash):
            raise LedgerError(f"GPU usage ledger line {number} lacks a valid record hash")
        if record.get("previous_record_sha256") != previous:
            raise LedgerError(f"GPU usage ledger chain forked at line {number}")
        if semantic_sha256(record) != recorded_hash:
            raise LedgerError(f"GPU usage ledger hash drifted at line {number}")
        record["record_sha256"] = recorded_hash
        # The pre-V7 helper emitted some test-only V1 records before the
        # schema_version field became mandatory.  They retain the identical
        # legacy elapsed-time semantics and are accepted only in the V1 prefix.
        schema = record.get("schema_version", 1)
        if type(schema) is not int:
            raise LedgerError(f"GPU usage ledger line {number} has invalid schema version")

        if schema == 1:
            if saw_v2:
                raise LedgerError("legacy schema-v1 records cannot follow schema-v2 records")
            settled_ns += _legacy_elapsed_ns(record, number)
        elif schema == SCHEMA_VERSION:
            saw_v2 = True
            _validate_v2_common(record, number)
            event = str(record["event"])
            lifecycle_id = str(record["lifecycle_id"])
            if event == "reservation":
                if lifecycle_id in seen_lifecycles or lifecycle_id in open_items:
                    raise LedgerError(f"duplicate lifecycle reservation at line {number}")
                reservation_ns = _require_int(
                    record.get("reservation_ns"),
                    f"ledger line {number} reservation_ns",
                    minimum=1,
                )
                workload_ns = _require_int(
                    record.get("workload_timeout_ns"),
                    f"ledger line {number} workload_timeout_ns",
                    minimum=1,
                )
                for field, expected in (
                    ("budget_ns", budget_ns),
                    ("heartbeat_interval_ns", HEARTBEAT_INTERVAL_NS),
                    ("termination_grace_ns", TERMINATION_GRACE_NS),
                    ("accounting_margin_ns", ACCOUNTING_MARGIN_NS),
                    ("recovery_margin_ns", RECOVERY_MARGIN_NS),
                ):
                    if record.get(field) != expected:
                        raise LedgerError(f"ledger line {number} fixed {field} drifted")
                if workload_ns + TERMINATION_GRACE_NS + ACCOUNTING_MARGIN_NS != reservation_ns:
                    raise LedgerError(f"ledger line {number} reservation decomposition drifted")
                open_items[lifecycle_id] = OpenReservation(record, None)
                seen_lifecycles.add(lifecycle_id)
            elif event == "heartbeat":
                item = open_items.get(lifecycle_id)
                if item is None:
                    raise LedgerError(f"orphan or post-terminal heartbeat at line {number}")
                _same_lifecycle_identity(record, item.reservation, number)
                sequence = _require_int(
                    record.get("sequence"), f"ledger line {number} heartbeat sequence", minimum=1
                )
                ceiling = _require_int(
                    record.get("elapsed_ceiling_ns"),
                    f"ledger line {number} heartbeat ceiling",
                )
                if ceiling > item.reservation_ns:
                    raise LedgerError(f"ledger line {number} heartbeat exceeds reservation")
                reservation_clock = int(item.reservation["monotonic_ns"])
                heartbeat_clock = int(record["monotonic_ns"])
                if heartbeat_clock < reservation_clock:
                    raise LedgerError(f"ledger line {number} heartbeat predates reservation")
                observed_elapsed = min(
                    item.reservation_ns, heartbeat_clock - reservation_clock
                )
                if ceiling < observed_elapsed:
                    raise LedgerError(
                        f"ledger line {number} heartbeat is not a conservative ceiling"
                    )
                child_pid = _require_int(
                    record.get("child_pid"),
                    f"ledger line {number} heartbeat child_pid",
                    minimum=1,
                )
                child_ticks = _require_int(
                    record.get("child_start_ticks"),
                    f"ledger line {number} heartbeat child_start_ticks",
                    minimum=1,
                )
                prior = item.last_heartbeat
                if prior is None:
                    if sequence != 1:
                        raise LedgerError(f"ledger line {number} heartbeat sequence forked")
                else:
                    if sequence != int(prior["sequence"]) + 1:
                        raise LedgerError(f"ledger line {number} heartbeat sequence forked")
                    if ceiling < int(prior["elapsed_ceiling_ns"]):
                        raise LedgerError(f"ledger line {number} heartbeat ceiling regressed")
                    if int(record["monotonic_ns"]) < int(prior["monotonic_ns"]):
                        raise LedgerError(f"ledger line {number} heartbeat clock regressed")
                    if (
                        child_pid != int(prior["child_pid"])
                        or child_ticks != int(prior["child_start_ticks"])
                    ):
                        raise LedgerError(f"ledger line {number} heartbeat child drifted")
                open_items[lifecycle_id] = OpenReservation(item.reservation, record)
            else:
                item = open_items.get(lifecycle_id)
                if item is None:
                    raise LedgerError(f"duplicate or orphan terminal at line {number}")
                _same_lifecycle_identity(record, item.reservation, number)
                charge = _require_int(
                    record.get("charged_usage_ns"),
                    f"ledger line {number} charged_usage_ns",
                )
                if charge > item.reservation_ns:
                    raise LedgerError(f"ledger line {number} charge exceeds reservation")
                last_hash = (
                    item.last_heartbeat.get("record_sha256")
                    if item.last_heartbeat is not None
                    else None
                )
                if record.get("last_heartbeat_record_sha256") != last_hash:
                    raise LedgerError(f"ledger line {number} heartbeat binding drifted")
                if event == "terminal":
                    elapsed_ns = _require_int(
                        record.get("elapsed_ns"), f"ledger line {number} elapsed_ns"
                    )
                    reservation_clock = int(item.reservation["monotonic_ns"])
                    terminal_clock = int(record["monotonic_ns"])
                    if terminal_clock < reservation_clock:
                        raise LedgerError(f"ledger line {number} terminal predates reservation")
                    if elapsed_ns < terminal_clock - reservation_clock:
                        raise LedgerError(
                            f"ledger line {number} terminal elapsed time undercounts its clock"
                        )
                    if charge != min(item.reservation_ns, elapsed_ns):
                        raise LedgerError(
                            f"ledger line {number} terminal charge undercounts elapsed time"
                        )
                    deadline_breached = record.get(
                        "reservation_deadline_breached"
                    )
                    if type(deadline_breached) is not bool or deadline_breached is not (
                        elapsed_ns > item.reservation_ns
                    ):
                        raise LedgerError(
                            f"ledger line {number} reservation breach flag drifted"
                        )
                    if type(record.get("return_code")) is not int:
                        raise LedgerError(f"ledger line {number} terminal return_code is invalid")
                    if type(record.get("wrapper_exit_code")) is not int:
                        raise LedgerError(
                            f"ledger line {number} terminal wrapper_exit_code is invalid"
                        )
                    if type(record.get("hard_timeout_reached")) is not bool:
                        raise LedgerError(f"ledger line {number} timeout flag is invalid")
                    if type(record.get("termination_escalated")) is not bool:
                        raise LedgerError(
                            f"ledger line {number} escalation flag is invalid"
                        )
                    if type(record.get("containment_anomaly")) is not bool:
                        raise LedgerError(
                            f"ledger line {number} containment flag is invalid"
                        )
                    received_signal = record.get("received_signal")
                    if received_signal is not None and (
                        type(received_signal) is not int
                        or received_signal <= 0
                        or received_signal >= signal.NSIG
                    ):
                        raise LedgerError(
                            f"ledger line {number} received signal is invalid"
                        )
                    exception = record.get("exception")
                    if exception is not None and not isinstance(exception, dict):
                        raise LedgerError(f"ledger line {number} exception is invalid")
                    child_pid = record.get("child_pid")
                    child_ticks = record.get("child_start_ticks")
                    if (child_pid is None) != (child_ticks is None):
                        if exception is None:
                            raise LedgerError(
                                f"ledger line {number} child identity is incomplete"
                            )
                    elif child_pid is None:
                        if exception is None:
                            raise LedgerError(
                                f"ledger line {number} missing child lacks an exception"
                            )
                    else:
                        _require_int(
                            child_pid,
                            f"ledger line {number} terminal child_pid",
                            minimum=1,
                        )
                        _require_int(
                            child_ticks,
                            f"ledger line {number} terminal child_start_ticks",
                            minimum=1,
                        )
                    if item.last_heartbeat is not None and (
                        child_pid != item.last_heartbeat.get("child_pid")
                        or child_ticks != item.last_heartbeat.get("child_start_ticks")
                    ):
                        raise LedgerError(
                            f"ledger line {number} terminal heartbeat child drifted"
                        )
                    if deadline_breached:
                        expected_wrapper_exit = 125
                    elif received_signal is not None:
                        expected_wrapper_exit = 128 + int(received_signal)
                    elif record["hard_timeout_reached"] is True:
                        expected_wrapper_exit = 124
                    elif (
                        record["termination_escalated"] is True
                        or record["containment_anomaly"] is True
                        or exception is not None
                    ):
                        expected_wrapper_exit = 125
                    elif int(record["return_code"]) < 0:
                        expected_wrapper_exit = 128 + abs(int(record["return_code"]))
                    else:
                        expected_wrapper_exit = int(record["return_code"])
                    if int(record["wrapper_exit_code"]) != expected_wrapper_exit:
                        raise LedgerError(
                            f"ledger line {number} wrapper exit code drifted"
                        )
                    expected_reuse = (
                        int(record["return_code"]) == 0
                        and int(record["wrapper_exit_code"]) == 0
                        and record["hard_timeout_reached"] is False
                        and received_signal is None
                        and exception is None
                        and not deadline_breached
                        and record["termination_escalated"] is False
                        and record["containment_anomaly"] is False
                    )
                    if record.get("reuse_eligible") is not expected_reuse:
                        raise LedgerError(f"ledger line {number} reuse eligibility drifted")
                else:
                    if record.get("return_code") is not None:
                        raise LedgerError(
                            f"ledger line {number} reconciled return_code must be null"
                        )
                    if record.get("reuse_eligible") is not False:
                        raise LedgerError(
                            f"ledger line {number} reconciled terminal cannot be reusable"
                        )
                    mode = record.get("reconciliation_mode")
                    if mode not in {"same_boot_proven_dead_ceiling", "full_reservation"}:
                        raise LedgerError(
                            f"ledger line {number} reconciliation mode is invalid"
                        )
                    ceiling = (
                        int(item.last_heartbeat["elapsed_ceiling_ns"])
                        if item.last_heartbeat is not None
                        else 0
                    )
                    trusted = record.get("reconciliation_monotonic_trusted")
                    observed_elapsed = record.get(
                        "reconciliation_observed_elapsed_ns"
                    )
                    if type(trusted) is not bool:
                        raise LedgerError(
                            f"ledger line {number} reconciliation clock trust is invalid"
                        )
                    if mode == "same_boot_proven_dead_ceiling":
                        if trusted is not True or type(observed_elapsed) is not int:
                            raise LedgerError(
                                f"ledger line {number} reconciliation clock is untrusted"
                            )
                        if observed_elapsed < 0 or observed_elapsed != (
                            int(record["monotonic_ns"])
                            - int(item.reservation["monotonic_ns"])
                        ):
                            raise LedgerError(
                                f"ledger line {number} reconciliation elapsed drifted"
                            )
                        if record.get("reconciled_by_boot_id") != item.reservation.get(
                            "boot_id"
                        ):
                            raise LedgerError(
                                f"ledger line {number} reconciliation boot identity drifted"
                            )
                        expected_charge = min(
                            item.reservation_ns,
                            max(
                                ceiling + RECOVERY_MARGIN_NS,
                                int(observed_elapsed),
                            ),
                        )
                    else:
                        if observed_elapsed is not None and type(observed_elapsed) is not int:
                            raise LedgerError(
                                f"ledger line {number} reconciliation elapsed is invalid"
                            )
                        expected_charge = item.reservation_ns
                    if charge != expected_charge:
                        raise LedgerError(
                            f"ledger line {number} reconciled charge is not conservative"
                        )
                settled_ns += charge
                del open_items[lifecycle_id]
        else:
            raise LedgerError(f"GPU usage ledger line {number} has unknown schema version")

        open_ns = sum(item.reservation_ns for item in open_items.values())
        if settled_ns < 0 or settled_ns + open_ns > budget_ns:
            raise LedgerError(f"GPU usage budget invariant failed at line {number}")
        records.append(record)
        previous = str(recorded_hash)

    if expected_legacy_genesis_sha256 is not None:
        if not records:
            raise LedgerError("required legacy genesis record is missing")
        first = records[0]
        if (
            first.get("schema_version") != 1
            or first.get("record_sha256") != expected_legacy_genesis_sha256
        ):
            raise LedgerError("legacy genesis record identity drifted")

    return LedgerState(
        records=tuple(records),
        settled_usage_ns=settled_ns,
        open_reservations=dict(open_items),
        budget_ns=budget_ns,
        tail_sha256=previous,
        raw_bytes=raw,
    )


def verify_ledger(
    path: Path,
    *,
    budget_ns: int = GPU_BUDGET_NS,
    expected_legacy_genesis_sha256: str | None = None,
    pinned_directory_fd: int | None = None,
) -> LedgerState:
    resolved = _canonical_no_final_symlink(path, "GPU usage ledger")
    if pinned_directory_fd is None and not resolved.parent.exists():
        return verify_ledger_bytes(
            b"",
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        )
    owns_directory = pinned_directory_fd is None
    directory_fd = (
        _open_directory_nofollow(resolved.parent, "GPU usage ledger parent")
        if pinned_directory_fd is None
        else pinned_directory_fd
    )
    _path_names_open_directory(
        resolved.parent, directory_fd, "GPU usage ledger parent"
    )
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved.name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        raw = b""
    except OSError as error:
        raise LedgerError(f"cannot read GPU usage ledger: {error}") from error
    else:
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                raise LedgerError(
                    f"GPU usage ledger is aliased or not a regular file: {resolved}"
                )
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                raw = handle.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    finally:
        if owns_directory:
            os.close(directory_fd)
    return verify_ledger_bytes(
        raw,
        budget_ns=budget_ns,
        expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
    )


def require_closed_ledger(
    path: Path,
    *,
    budget_ns: int = GPU_BUDGET_NS,
    expected_legacy_genesis_sha256: str | None = None,
    pinned_directory_fd: int | None = None,
) -> LedgerState:
    state = verify_ledger(
        path,
        budget_ns=budget_ns,
        expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        pinned_directory_fd=pinned_directory_fd,
    )
    if state.open_reservations:
        identifiers = ", ".join(sorted(state.open_reservations))
        raise LedgerError(f"GPU usage ledger has open reservations: {identifiers}")
    return state


@contextmanager
def locked_closed_snapshot(
    path: Path,
    *,
    budget_ns: int = GPU_BUDGET_NS,
    expected_legacy_genesis_sha256: str | None = None,
) -> Iterator[LedgerState]:
    """Yield one closed verified state while holding the stable ledger lock."""

    with _exclusive_ledger_lock(path) as locked:
        state = _read_locked(
            locked,
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        )
        if state.open_reservations:
            identifiers = ", ".join(sorted(state.open_reservations))
            raise LedgerError(f"GPU usage ledger has open reservations: {identifiers}")
        yield state


@contextmanager
def _exclusive_ledger_lock(
    path: Path, *, pinned_directory_fd: object | None = None
) -> Iterator[_LockedLedger]:
    resolved = _canonical_no_final_symlink(path, "GPU usage ledger")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved = _canonical_no_final_symlink(resolved, "GPU usage ledger")
    lock = ledger_lock_path(resolved)
    owns_directory = pinned_directory_fd is None
    with ExitStack() as cleanup:
        if pinned_directory_fd is None:
            directory_fd = _open_directory_nofollow(
                resolved.parent, "GPU usage ledger parent"
            )
            directory_status = os.fstat(directory_fd)
            directory_identity = (
                directory_status.st_dev,
                directory_status.st_ino,
            )
            cleanup.callback(
                _close_fd_fail_safe,
                directory_fd,
                expected_identity=directory_identity,
                label="GPU usage ledger directory",
            )
            _path_names_open_directory(
                resolved.parent, directory_fd, "GPU usage ledger parent"
            )
            directory_lease = acquire_directory_generation_fence(directory_fd)
            cleanup.callback(release_directory_generation_fence, directory_lease)
        else:
            directory_fd = cleanup.enter_context(
                borrow_directory_generation_lease(pinned_directory_fd)
            )
            directory_status = os.fstat(directory_fd)
            directory_identity = (
                directory_status.st_dev,
                directory_status.st_ino,
            )
            _path_names_open_directory(
                resolved.parent, directory_fd, "GPU usage ledger parent"
            )
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock.name, flags, 0o644, dir_fd=directory_fd)
        except OSError as error:
            raise LedgerError(f"cannot open GPU usage ledger lock: {error}") from error
        try:
            descriptor_status = os.fstat(descriptor)
        except BaseException:
            _close_fd_fail_safe(descriptor, label="GPU usage ledger lock")
            raise
        lock_identity = (descriptor_status.st_dev, descriptor_status.st_ino)
        cleanup.callback(
            _unlock_and_close_flocked_fd,
            descriptor,
            expected_identity=lock_identity,
            label="GPU usage ledger lock",
        )
        descriptor_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or descriptor_status.st_nlink != 1
        ):
            raise LedgerError(f"GPU usage ledger lock is not a regular file: {lock}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = _LockedLedger(
            path=resolved,
            lock_path=lock,
            directory_fd=directory_fd,
            lock_fd=descriptor,
            directory_identity=directory_identity,
            lock_identity=lock_identity,
            owns_directory_fd=owns_directory,
            owns_directory_fence=owns_directory,
        )
        locked.revalidate()
        yield locked
        locked.revalidate()


def _atomic_replace_bytes(
    path: Path, payload: bytes, *, directory_fd: int | None = None
) -> None:
    """Durably replace one file through its one registered residue name.

    The stable sibling ledger lock serializes writers, so a random temporary
    name adds no safety and makes a killed writer impossible to recover without
    scanning a directory.  The exact deterministic residue is instead left for
    the admission-guarded cleanup API when this process is killed.
    """

    resolved = _canonical_no_final_symlink(path, "atomic replacement target")
    owned_directory = directory_fd is None
    parent_fd = (
        _open_directory_nofollow(resolved.parent, "atomic replacement parent")
        if directory_fd is None
        else directory_fd
    )
    assert parent_fd is not None
    parent_status = os.fstat(parent_fd)
    if not stat.S_ISDIR(parent_status.st_mode):
        if owned_directory:
            os.close(parent_fd)
        raise LedgerError("atomic replacement parent is not a directory")
    _path_names_open_directory(
        resolved.parent, parent_fd, "atomic replacement parent"
    )
    temporary_name = atomic_replace_residue_name(resolved)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    created_temporary = False
    try:
        try:
            target_status = os.stat(
                resolved.name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            pass
        else:
            if (
                not stat.S_ISREG(target_status.st_mode)
                or target_status.st_nlink != 1
            ):
                raise LedgerError(
                    f"atomic replacement target is aliased or non-regular: {resolved}"
                )
        try:
            descriptor = os.open(temporary_name, flags, 0o644, dir_fd=parent_fd)
        except FileExistsError as error:
            raise LedgerError(
                f"registered state replacement residue requires recovery: "
                f"{resolved.parent / temporary_name}"
            ) from error
        created_temporary = True
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_status = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(temporary_status.st_mode)
                or stat.S_IMODE(temporary_status.st_mode) != 0o644
                or temporary_status.st_nlink != 1
            ):
                raise LedgerError("state replacement residue metadata drifted")
        _fault_inject("state_replace_after_payload_fsync")
        _path_names_open_directory(
            resolved.parent, parent_fd, "atomic replacement parent"
        )
        os.replace(
            temporary_name,
            resolved.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        created_temporary = False
        _fault_inject("state_replace_after_replace_before_directory_fsync")
        os.fsync(parent_fd)
        _path_names_open_directory(
            resolved.parent, parent_fd, "atomic replacement parent"
        )
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if created_temporary:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if owned_directory:
            os.close(parent_fd)


def atomic_replace_residue_name(path: Path) -> str:
    """Return the sole registered replacement-residue basename for ``path``."""

    name = Path(path).name
    if not name or name in {".", ".."} or "/" in name:
        raise LedgerError("state replacement target basename is invalid")
    return f".{name}{_STATE_REPLACE_TEMP_SUFFIX}"


def _cleanup_atomic_replace_residue_locked(
    *,
    path: Path,
    directory_fd: int,
    current_payload: bytes,
    validate_candidate: Callable[[bytes], object],
    admission_revalidate: Callable[[], None],
) -> bool:
    """Remove only the exact, valid successor residue from one locked ledger."""

    if not callable(admission_revalidate):
        raise LedgerError("admission revalidation callback is required")
    admission_revalidate()
    residue_name = atomic_replace_residue_name(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(residue_name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        admission_revalidate()
        return False
    except OSError as error:
        raise LedgerError(
            f"cannot open registered state replacement residue: {error}"
        ) from error
    identity: tuple[int, int] | None = None
    try:
        before = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o644
            or before.st_nlink != 1
        ):
            raise LedgerError(
                "registered state replacement residue is mutable-mode, aliased, "
                "or non-regular"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        candidate = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(residue_name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not _same_inode(before, after)
            or not _same_inode(after, named)
            or after.st_nlink != 1
            or named.st_nlink != 1
            or stat.S_IMODE(after.st_mode) != 0o644
            or stat.S_IMODE(named.st_mode) != 0o644
        ):
            raise LedgerError(
                "registered state replacement residue generation changed"
            )
        if len(candidate) <= len(current_payload) or not candidate.startswith(
            current_payload
        ):
            raise LedgerError(
                "registered state replacement residue is not a strict current-ledger successor"
            )
        validate_candidate(candidate)
        admission_revalidate()
        final_descriptor = os.fstat(descriptor)
        final_named = os.stat(
            residue_name, dir_fd=directory_fd, follow_symlinks=False
        )
        if (
            not _same_inode(final_descriptor, final_named)
            or (final_descriptor.st_dev, final_descriptor.st_ino) != identity
            or final_descriptor.st_nlink != 1
            or final_named.st_nlink != 1
            or stat.S_IMODE(final_descriptor.st_mode) != 0o644
            or stat.S_IMODE(final_named.st_mode) != 0o644
        ):
            raise LedgerError(
                "registered state replacement residue changed before cleanup"
            )
        os.unlink(residue_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        try:
            os.stat(residue_name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise LedgerError("registered state replacement residue survived cleanup")
        admission_revalidate()
        return True
    finally:
        _close_fd_fail_safe(
            descriptor,
            expected_identity=identity,
            label="registered state replacement residue",
        )


def cleanup_usage_ledger_replace_residue(
    path: Path,
    *,
    budget_ns: int = GPU_BUDGET_NS,
    expected_legacy_genesis_sha256: str | None = None,
    pinned_directory_fd: object | None = None,
    admission_revalidate: Callable[[], None],
) -> bool:
    """Clean one killed usage-ledger commit under admission then usage lock."""

    with _exclusive_ledger_lock(
        path, pinned_directory_fd=pinned_directory_fd
    ) as locked:
        state = _read_locked(
            locked,
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        )

        def validate_candidate(payload: bytes) -> LedgerState:
            return verify_ledger_bytes(
                payload,
                budget_ns=budget_ns,
                expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
            )

        removed = _cleanup_atomic_replace_residue_locked(
            path=locked.path,
            directory_fd=locked.directory_fd,
            current_payload=state.raw_bytes,
            validate_candidate=validate_candidate,
            admission_revalidate=admission_revalidate,
        )
        locked.revalidate()
        current = _read_locked(
            locked,
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        )
        if current.raw_bytes != state.raw_bytes:
            raise LedgerError("GPU usage ledger changed during residue cleanup")
        return removed


def _read_regular_bytes_identity_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    missing_ok: bool = False,
    durable: bool = False,
) -> tuple[bytes, tuple[int, int]] | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise LedgerError(f"{label} is missing")
    except OSError as error:
        raise LedgerError(f"cannot open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise LedgerError(f"{label} is aliased or not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        if durable:
            os.fsync(descriptor)
        after = os.fstat(descriptor)
        try:
            named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise LedgerError(f"{label} identity changed during read: {error}") from error
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or named.st_nlink != 1
            or not _same_inode(before, after)
            or not _same_inode(after, named)
        ):
            raise LedgerError(f"{label} identity or link count changed during read")
        payload = b"".join(chunks)
        if durable:
            os.fsync(directory_fd)
            final_descriptor = os.fstat(descriptor)
            final_named = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False
            )
            if (
                final_descriptor.st_nlink != 1
                or final_named.st_nlink != 1
                or (final_descriptor.st_dev, final_descriptor.st_ino) != identity
                or not _same_inode(final_descriptor, final_named)
            ):
                raise LedgerError(
                    f"{label} generation changed during durability recovery"
                )
        return payload, identity
    finally:
        if descriptor >= 0:
            _close_fd_fail_safe(
                descriptor,
                expected_identity=(before.st_dev, before.st_ino)
                if "before" in locals()
                else None,
                label=label,
            )


def _read_regular_bytes_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    missing_ok: bool = False,
) -> bytes | None:
    value = _read_regular_bytes_identity_at(
        directory_fd,
        name,
        label=label,
        missing_ok=missing_ok,
    )
    return None if value is None else value[0]


def _atomic_create_bytes(path: Path, payload: bytes, *, directory_fd: int) -> None:
    """Durably link a complete immutable anonymous inode exactly once."""

    resolved = _canonical_no_final_symlink(path, "atomic create target")
    _path_names_open_directory(
        resolved.parent, directory_fd, "atomic create parent"
    )
    anonymous_flag = getattr(os, "O_TMPFILE", 0)
    if not anonymous_flag:
        raise LedgerError("immutable create-once publication requires Linux O_TMPFILE")
    flags = os.O_RDWR | anonymous_flag
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    linked = False
    try:
        descriptor = os.open(".", flags, 0o600, dir_fd=directory_fd)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise LedgerError("short write to anonymous immutable publication inode")
            offset += written
        os.fsync(descriptor)
        _fault_inject("anonymous_create_after_payload_fsync")
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        anonymous_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(anonymous_status.st_mode)
            or stat.S_IMODE(anonymous_status.st_mode) != 0o444
            or anonymous_status.st_nlink != 0
            or anonymous_status.st_size != len(payload)
        ):
            raise LedgerError("anonymous immutable publication inode metadata drifted")
        _fault_inject("anonymous_create_before_link")
        try:
            linkat = ctypes.CDLL(None, use_errno=True).linkat
        except AttributeError as error:
            raise LedgerError(
                "immutable create-once publication requires Linux linkat"
            ) from error
        linkat.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        )
        linkat.restype = ctypes.c_int
        result = linkat(
            descriptor,
            b"",
            directory_fd,
            os.fsencode(resolved.name),
            _AT_EMPTY_PATH,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(error_number, os.strerror(error_number), resolved)
            raise OSError(error_number, os.strerror(error_number), resolved)
        linked = True
        named = os.stat(resolved.name, dir_fd=directory_fd, follow_symlinks=False)
        linked_status = os.fstat(descriptor)
        if (
            not _same_inode(named, linked_status)
            or not stat.S_ISREG(named.st_mode)
            or stat.S_IMODE(named.st_mode) != 0o444
            or stat.S_IMODE(linked_status.st_mode) != 0o444
            or named.st_nlink != 1
            or linked_status.st_nlink != 1
            or named.st_size != len(payload)
        ):
            raise LedgerError("linked immutable publication metadata drifted")
        _fault_inject("anonymous_create_after_link_before_directory_fsync")
        os.fsync(directory_fd)
        _path_names_open_directory(
            resolved.parent, directory_fd, "atomic create parent"
        )
        final_named = os.stat(
            resolved.name, dir_fd=directory_fd, follow_symlinks=False
        )
        final_descriptor = os.fstat(descriptor)
        if (
            not _same_inode(final_named, final_descriptor)
            or stat.S_IMODE(final_named.st_mode) != 0o444
            or stat.S_IMODE(final_descriptor.st_mode) != 0o444
            or final_named.st_nlink != 1
            or final_descriptor.st_nlink != 1
            or final_named.st_size != len(payload)
        ):
            raise LedgerError("immutable publication postcondition failed")
    except BaseException:
        # Once linked, the inode is already complete, fsynced, immutable, and
        # create-once.  Preserve it for byte-identical retry/replay.
        raise
    finally:
        if descriptor >= 0:
            _close_fd_fail_safe(
                descriptor,
                expected_identity=(anonymous_status.st_dev, anonymous_status.st_ino)
                if "anonymous_status" in locals()
                else None,
                label=(
                    "linked immutable publication inode"
                    if linked
                    else "anonymous immutable publication inode"
                ),
            )


def atomic_create_immutable_bytes(
    path: Path,
    payload: bytes,
    *,
    pinned_directory_fd: object | None = None,
) -> None:
    """Create one 0444/nlink1 file through O_TMPFILE plus AT_EMPTY_PATH."""

    if not isinstance(payload, bytes):
        raise LedgerError("immutable publication payload must be bytes")
    resolved = _canonical_no_final_symlink(path, "immutable publication target")
    with ExitStack() as cleanup:
        if pinned_directory_fd is None:
            directory_fd = _open_directory_nofollow(
                resolved.parent, "immutable publication parent"
            )
            directory_status = os.fstat(directory_fd)
            cleanup.callback(
                _close_fd_fail_safe,
                directory_fd,
                expected_identity=(directory_status.st_dev, directory_status.st_ino),
                label="immutable publication directory",
            )
            lease = acquire_directory_generation_fence(directory_fd)
            cleanup.callback(release_directory_generation_fence, lease)
        else:
            directory_fd = cleanup.enter_context(
                borrow_directory_generation_lease(pinned_directory_fd)
            )
        _path_names_open_directory(
            resolved.parent, directory_fd, "immutable publication parent"
        )
        _atomic_create_bytes(resolved, payload, directory_fd=directory_fd)


@dataclass
class _LockedResult:
    path: Path
    directory_fd: int
    allowed_result_modes: frozenset[int] = frozenset({0o444})
    identity: tuple[int, int] | None = None
    payload: bytes | None = None

    def read(self, *, missing_ok: bool = False, durable: bool = False) -> bytes | None:
        value = _read_regular_bytes_identity_at(
            self.directory_fd,
            self.path.name,
            label="GPU terminal result",
            missing_ok=missing_ok,
            durable=durable,
        )
        if value is None:
            if self.identity is not None:
                raise LedgerError("GPU terminal result disappeared while locked")
            return None
        payload, identity = value
        named_status = os.stat(
            self.path.name, dir_fd=self.directory_fd, follow_symlinks=False
        )
        if stat.S_IMODE(named_status.st_mode) not in self.allowed_result_modes:
            raise LedgerError("GPU terminal result mode is not authorized")
        if self.identity is not None and (
            identity != self.identity or payload != self.payload
        ):
            raise LedgerError("GPU terminal result generation changed while locked")
        self.identity = identity
        self.payload = payload
        return payload

    def revalidate(self) -> None:
        if self.identity is None:
            raise LedgerError("GPU terminal result was not bound while locked")
        self.read()


@contextmanager
def _exclusive_result_lock(
    path: Path,
    *,
    pinned_directory_fd: object | None = None,
    allow_legacy_mutable_evidence: bool = False,
) -> Iterator[_LockedResult]:
    resolved = _canonical_no_final_symlink(path, "GPU terminal result")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved = _canonical_no_final_symlink(resolved, "GPU terminal result")
    with ExitStack() as cleanup:
        if pinned_directory_fd is None:
            directory_fd = _open_directory_nofollow(
                resolved.parent, "GPU terminal result parent"
            )
            directory_status = os.fstat(directory_fd)
            directory_identity = (
                directory_status.st_dev,
                directory_status.st_ino,
            )
            cleanup.callback(
                _close_fd_fail_safe,
                directory_fd,
                expected_identity=directory_identity,
                label="GPU terminal result directory",
            )
            _path_names_open_directory(
                resolved.parent, directory_fd, "GPU terminal result parent"
            )
            directory_lease = acquire_directory_generation_fence(directory_fd)
            cleanup.callback(release_directory_generation_fence, directory_lease)
        else:
            directory_fd = cleanup.enter_context(
                borrow_directory_generation_lease(pinned_directory_fd)
            )
            _path_names_open_directory(
                resolved.parent, directory_fd, "GPU terminal result parent"
            )
        lock_path = result_receipt_lock_path(resolved)
        if not allow_legacy_mutable_evidence:
            try:
                _atomic_create_bytes(lock_path, b"", directory_fd=directory_fd)
            except FileExistsError:
                pass
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path.name, flags, dir_fd=directory_fd)
        try:
            status = os.fstat(descriptor)
        except BaseException:
            _close_fd_fail_safe(descriptor, label="GPU terminal result lock")
            raise
        lock_identity = (status.st_dev, status.st_ino)
        cleanup.callback(
            _unlock_and_close_flocked_fd,
            descriptor,
            expected_identity=lock_identity,
            label="GPU terminal result lock",
        )
        allowed_modes = (
            frozenset({0o444, 0o644})
            if allow_legacy_mutable_evidence
            else frozenset({0o444})
        )
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or status.st_size != 0
            or stat.S_IMODE(status.st_mode) not in allowed_modes
        ):
            raise LedgerError("GPU terminal result lock is aliased or non-regular")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        named = os.stat(
            lock_path.name, dir_fd=directory_fd, follow_symlinks=False
        )
        current = os.fstat(descriptor)
        if not _same_inode(named, current) or current.st_nlink != 1:
            raise LedgerError("GPU terminal result lock generation changed")
        _path_names_open_directory(
            resolved.parent, directory_fd, "GPU terminal result parent"
        )
        locked_result = _LockedResult(
            resolved,
            directory_fd,
            allowed_result_modes=allowed_modes,
        )
        yield locked_result
        named = os.stat(
            lock_path.name, dir_fd=directory_fd, follow_symlinks=False
        )
        current = os.fstat(descriptor)
        if (
            not _same_inode(named, current)
            or current.st_nlink != 1
            or named.st_nlink != 1
            or current.st_size != 0
            or named.st_size != 0
            or stat.S_IMODE(current.st_mode) not in allowed_modes
            or stat.S_IMODE(named.st_mode) not in allowed_modes
        ):
            raise LedgerError("GPU terminal result lock generation changed")
        locked_result.revalidate()


def _read_locked(
    locked: _LockedLedger,
    *,
    budget_ns: int,
    expected_legacy_genesis_sha256: str | None,
) -> LedgerState:
    locked.revalidate()
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            locked.path.name, flags, dir_fd=locked.directory_fd
        )
    except FileNotFoundError:
        raw = b""
    except OSError as error:
        raise LedgerError(f"cannot open GPU usage ledger: {error}") from error
    else:
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                raise LedgerError(
                    f"GPU usage ledger is aliased or not a regular file: {locked.path}"
                )
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                raw = handle.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    return verify_ledger_bytes(
        raw,
        budget_ns=budget_ns,
        expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
    )


def _with_hash(record: Mapping[str, Any], previous: str | None) -> dict[str, Any]:
    if "record_sha256" in record or "previous_record_sha256" in record:
        raise LedgerError("caller cannot supply ledger linkage fields")
    document = dict(record)
    document["previous_record_sha256"] = previous
    document["record_sha256"] = semantic_sha256(document)
    return document


def _commit_records_locked(
    locked: _LockedLedger,
    state: LedgerState,
    values: Sequence[Mapping[str, Any]],
    *,
    budget_ns: int,
    expected_legacy_genesis_sha256: str | None,
) -> tuple[list[dict[str, Any]], LedgerState]:
    locked.revalidate()
    current = _read_locked(
        locked,
        budget_ns=budget_ns,
        expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
    )
    if current.raw_bytes != state.raw_bytes or current.tail_sha256 != state.tail_sha256:
        raise LedgerError("GPU usage ledger changed during locked transaction")
    documents: list[dict[str, Any]] = []
    previous = state.tail_sha256
    payload = bytearray(state.raw_bytes)
    for value in values:
        document = _with_hash(value, previous)
        payload.extend(canonical_json_bytes(document))
        payload.extend(b"\n")
        documents.append(document)
        previous = str(document["record_sha256"])
    candidate = verify_ledger_bytes(
        bytes(payload),
        budget_ns=budget_ns,
        expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
    )
    locked.revalidate()
    _atomic_replace_bytes(
        locked.path, bytes(payload), directory_fd=locked.directory_fd
    )
    locked.revalidate()
    published = _read_locked(
        locked,
        budget_ns=budget_ns,
        expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
    )
    if (
        published.raw_bytes != bytes(payload)
        or published.tail_sha256 != candidate.tail_sha256
    ):
        raise LedgerError("GPU usage ledger commit generation verification failed")
    return documents, candidate


def append_record(
    path: Path,
    value: Mapping[str, Any],
    *,
    budget_ns: int = GPU_BUDGET_NS,
    expected_legacy_genesis_sha256: str | None = None,
    pinned_directory_fd: object | None = None,
) -> dict[str, Any]:
    """Atomically append one fully validated record under the stable lock."""

    with _exclusive_ledger_lock(
        path, pinned_directory_fd=pinned_directory_fd
    ) as locked:
        state = _read_locked(
            locked,
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        )
        documents, _ = _commit_records_locked(
            locked,
            state,
            [value],
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        )
        return documents[0]


def boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as error:
        raise LedgerError(f"cannot establish Linux boot identity: {error}") from error
    return _require_text(value, "Linux boot identity")


def process_start_ticks(pid: int) -> int | None:
    """Read Linux /proc start ticks, returning None only when the PID is absent."""

    _require_int(pid, "pid", minimum=1)
    proc = Path("/proc") / str(pid)
    try:
        raw = (proc / "stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as error:
        if not proc.exists():
            return None
        raise LedgerError(f"cannot prove process identity for PID {pid}: {error}") from error
    closing = raw.rfind(")")
    if closing < 0:
        raise LedgerError(f"malformed /proc identity for PID {pid}")
    fields = raw[closing + 2 :].split()
    if len(fields) <= 19:
        raise LedgerError(f"truncated /proc identity for PID {pid}")
    try:
        ticks = int(fields[19])
    except ValueError as error:
        raise LedgerError(f"invalid /proc identity for PID {pid}") from error
    if ticks <= 0:
        raise LedgerError(f"invalid /proc start ticks for PID {pid}")
    return ticks


def _base_from_reservation(
    reservation: Mapping[str, Any], *, realtime_ns: int, monotonic_ns: int
) -> dict[str, Any]:
    result = {
        "schema_version": SCHEMA_VERSION,
        "lifecycle_id": reservation["lifecycle_id"],
        "campaign_id": reservation["campaign_id"],
        "phase": reservation["phase"],
        "context": reservation["context"],
        "invocation_sha256": reservation["invocation_sha256"],
        "command_sha256": reservation["command_sha256"],
        "result_path": reservation["result_path"],
        "boot_id": reservation["boot_id"],
        "wrapper_pid": reservation["wrapper_pid"],
        "wrapper_start_ticks": reservation["wrapper_start_ticks"],
        "reservation_record_sha256": reservation["record_sha256"],
        "realtime_ns": realtime_ns,
        "monotonic_ns": monotonic_ns,
    }
    result["gpu_execution_ledger_path"] = reservation["gpu_execution_ledger_path"]
    return result


def _reconciliation_record(
    item: OpenReservation,
    *,
    current_boot_id: str,
    realtime_ns: int,
    monotonic_ns: int,
) -> dict[str, Any]:
    reservation = item.reservation
    same_boot = reservation["boot_id"] == current_boot_id
    identity_status = "different_boot"
    proven_dead = False
    if same_boot:
        try:
            observed_ticks = process_start_ticks(int(reservation["wrapper_pid"]))
        except LedgerError:
            observed_ticks = -1
            identity_status = "unprovable"
        else:
            if observed_ticks is None:
                identity_status = "pid_absent"
                proven_dead = True
            elif observed_ticks != int(reservation["wrapper_start_ticks"]):
                identity_status = "pid_reused"
                proven_dead = True
            else:
                identity_status = "matching_process_still_present"
                raise LedgerBusy(
                    "open GPU reservation still belongs to the matching live wrapper"
                )
    ceiling = (
        int(item.last_heartbeat["elapsed_ceiling_ns"])
        if item.last_heartbeat is not None
        else 0
    )
    reservation_clock = int(reservation["monotonic_ns"])
    monotonic_trusted = same_boot and monotonic_ns >= reservation_clock
    observed_elapsed_ns = monotonic_ns - reservation_clock if monotonic_trusted else None
    if same_boot and proven_dead and monotonic_trusted:
        mode = "same_boot_proven_dead_ceiling"
        charged_ns = min(
            item.reservation_ns,
            max(ceiling + RECOVERY_MARGIN_NS, int(observed_elapsed_ns)),
        )
    else:
        mode = "full_reservation"
        charged_ns = item.reservation_ns
    return {
        **_base_from_reservation(
            reservation, realtime_ns=realtime_ns, monotonic_ns=monotonic_ns
        ),
        "event": "reconciled_terminal",
        "last_heartbeat_record_sha256": (
            item.last_heartbeat["record_sha256"]
            if item.last_heartbeat is not None
            else None
        ),
        "charged_usage_ns": charged_ns,
        "return_code": None,
        "reuse_eligible": False,
        "reconciliation_mode": mode,
        "process_identity_status": identity_status,
        "process_identity_proven_dead": proven_dead,
        "reconciled_by_boot_id": current_boot_id,
        "reconciliation_monotonic_trusted": monotonic_trusted,
        "reconciliation_observed_elapsed_ns": observed_elapsed_ns,
    }


def reconcile_open_reservations(
    path: Path,
    *,
    realtime_ns: int,
    monotonic_ns: int,
    current_boot_id: str | None = None,
    budget_ns: int = GPU_BUDGET_NS,
    expected_legacy_genesis_sha256: str | None = None,
    pinned_directory_fd: object | None = None,
) -> tuple[list[dict[str, Any]], LedgerState]:
    """Close every abandoned reservation conservatively in one transaction."""

    current_boot = current_boot_id or boot_id()
    with _exclusive_ledger_lock(
        path, pinned_directory_fd=pinned_directory_fd
    ) as locked:
        state = _read_locked(
            locked,
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        )
        values = [
            _reconciliation_record(
                state.open_reservations[key],
                current_boot_id=current_boot,
                realtime_ns=realtime_ns,
                monotonic_ns=monotonic_ns,
            )
            for key in sorted(state.open_reservations)
        ]
        if not values:
            return [], state
        return _commit_records_locked(
            locked,
            state,
            values,
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        )


def reconcile_and_reserve(
    path: Path,
    reservation: Mapping[str, Any],
    *,
    budget_ns: int = GPU_BUDGET_NS,
    expected_legacy_genesis_sha256: str | None = None,
    pinned_directory_fd: object | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], LedgerState]:
    """Reconcile abandoned work and durably reserve all available budget.

    The supplied reservation is an unhashed identity/timestamp template.  This
    function fills ``reservation_ns`` and ``workload_timeout_ns`` while holding
    the usage-ledger lock, so budget read and reservation publication are one
    transaction.
    """

    current_boot = _require_text(reservation.get("boot_id"), "reservation boot_id")
    realtime_ns = _require_int(reservation.get("realtime_ns"), "reservation realtime_ns")
    monotonic_ns = _require_int(
        reservation.get("monotonic_ns"), "reservation monotonic_ns"
    )
    with _exclusive_ledger_lock(
        path, pinned_directory_fd=pinned_directory_fd
    ) as locked:
        state = _read_locked(
            locked,
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        )
        reconciliation_values = [
            _reconciliation_record(
                state.open_reservations[key],
                current_boot_id=current_boot,
                realtime_ns=realtime_ns,
                monotonic_ns=monotonic_ns,
            )
            for key in sorted(state.open_reservations)
        ]
        if reconciliation_values:
            reconciliation_documents, state = _commit_records_locked(
                locked,
                state,
                reconciliation_values,
                budget_ns=budget_ns,
                expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
            )
        else:
            reconciliation_documents = []
        available = state.remaining_ns
        fixed_overhead = TERMINATION_GRACE_NS + ACCOUNTING_MARGIN_NS
        if available <= fixed_overhead:
            raise BudgetExhausted(
                "GPU budget has no reservation that preserves termination grace and margin"
            )
        value = dict(reservation)
        value.update(
            {
                "schema_version": SCHEMA_VERSION,
                "event": "reservation",
                "budget_ns": budget_ns,
                "reservation_ns": available,
                "workload_timeout_ns": available - fixed_overhead,
                "heartbeat_interval_ns": HEARTBEAT_INTERVAL_NS,
                "termination_grace_ns": TERMINATION_GRACE_NS,
                "accounting_margin_ns": ACCOUNTING_MARGIN_NS,
                "recovery_margin_ns": RECOVERY_MARGIN_NS,
            }
        )
        documents, final_state = _commit_records_locked(
            locked,
            state,
            [value],
            budget_ns=budget_ns,
            expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        )
        return documents[0], tuple(reconciliation_documents), final_state


def append_heartbeat(
    path: Path,
    reservation: Mapping[str, Any],
    *,
    sequence: int,
    elapsed_ceiling_ns: int,
    realtime_ns: int,
    monotonic_ns: int,
    child_pid: int,
    child_start_ticks: int,
    budget_ns: int = GPU_BUDGET_NS,
    expected_legacy_genesis_sha256: str | None = None,
    pinned_directory_fd: object | None = None,
) -> dict[str, Any]:
    return append_record(
        path,
        {
            **_base_from_reservation(
                reservation, realtime_ns=realtime_ns, monotonic_ns=monotonic_ns
            ),
            "event": "heartbeat",
            "sequence": sequence,
            "elapsed_ceiling_ns": elapsed_ceiling_ns,
            "child_pid": child_pid,
            "child_start_ticks": child_start_ticks,
        },
        budget_ns=budget_ns,
        expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        pinned_directory_fd=pinned_directory_fd,
    )


def append_terminal(
    path: Path,
    reservation: Mapping[str, Any],
    *,
    last_heartbeat: Mapping[str, Any] | None,
    elapsed_ns: int,
    charged_usage_ns: int,
    realtime_ns: int,
    monotonic_ns: int,
    child_pid: int | None,
    child_start_ticks: int | None,
    return_code: int,
    wrapper_exit_code: int,
    hard_timeout_reached: bool,
    received_signal: int | None,
    termination_escalated: bool,
    containment_anomaly: bool = False,
    exception: Mapping[str, Any] | None = None,
    budget_ns: int = GPU_BUDGET_NS,
    expected_legacy_genesis_sha256: str | None = None,
    pinned_directory_fd: object | None = None,
) -> dict[str, Any]:
    deadline_breached = elapsed_ns > int(reservation["reservation_ns"])
    value: dict[str, Any] = {
        **_base_from_reservation(
            reservation, realtime_ns=realtime_ns, monotonic_ns=monotonic_ns
        ),
        "event": "terminal",
        "last_heartbeat_record_sha256": (
            last_heartbeat.get("record_sha256") if last_heartbeat is not None else None
        ),
        "elapsed_ns": elapsed_ns,
        "charged_usage_ns": charged_usage_ns,
        "child_pid": child_pid,
        "child_start_ticks": child_start_ticks,
        "return_code": return_code,
        "wrapper_exit_code": wrapper_exit_code,
        "hard_timeout_reached": hard_timeout_reached,
        "received_signal": received_signal,
        "termination_escalated": termination_escalated,
        "containment_anomaly": containment_anomaly,
        "reservation_deadline_breached": deadline_breached,
        "reuse_eligible": (
            return_code == 0
            and wrapper_exit_code == 0
            and not hard_timeout_reached
            and received_signal is None
            and exception is None
            and not deadline_breached
            and not termination_escalated
            and not containment_anomaly
        ),
    }
    if exception is not None:
        value["exception"] = dict(exception)
    return append_record(
        path,
        value,
        budget_ns=budget_ns,
        expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        pinned_directory_fd=pinned_directory_fd,
    )


def terminal_records_for_context(
    state: LedgerState,
    *,
    campaign_id: str,
    phase: str,
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    canonical_context = canonical_json_bytes(dict(context))
    return [
        dict(record)
        for record in state.records
        if record.get("schema_version") == SCHEMA_VERSION
        and record.get("event") in {"terminal", "reconciled_terminal"}
        and record.get("campaign_id") == campaign_id
        and record.get("phase") == phase
        and canonical_json_bytes(record.get("context")) == canonical_context
    ]


def _result_content_document(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document.pop("content_sha256", None)
    document["content_sha256"] = semantic_sha256(document)
    return document


def atomic_result_receipt(
    path: Path,
    value: Mapping[str, Any],
    *,
    pinned_directory_fd: object | None = None,
) -> dict[str, Any]:
    """Create exactly once, or recover one byte-identical terminal receipt."""

    resolved = _canonical_no_final_symlink(path, "GPU terminal result")
    document = _result_content_document(value)
    payload = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with _exclusive_result_lock(
        resolved, pinned_directory_fd=pinned_directory_fd
    ) as locked_result:
        existing = locked_result.read(missing_ok=True, durable=True)
        if existing is not None:
            if existing != payload:
                raise LedgerError(
                    f"GPU terminal result collision: {locked_result.path}"
                )
            return document
        try:
            _atomic_create_bytes(
                locked_result.path,
                payload,
                directory_fd=locked_result.directory_fd,
            )
        except FileExistsError as error:
            raise LedgerError(
                "GPU terminal result appeared outside its generation lock: "
                f"{locked_result.path}"
            ) from error
        published = locked_result.read()
        if published != payload:
            raise LedgerError("GPU terminal result publication verification failed")
        return document


def result_from_terminal(
    terminal: Mapping[str, Any], *, usage_ledger: Path, gpu_execution_ledger: Path
) -> dict[str, Any]:
    if terminal.get("event") != "terminal":
        raise LedgerError("only an observed terminal can produce a reusable result receipt")
    resolved_execution_ledger = str(
        _canonical_no_final_symlink(gpu_execution_ledger, "GPU execution ledger")
    )
    resolved_usage_ledger = str(
        _canonical_no_final_symlink(usage_ledger, "GPU usage ledger")
    )
    resolved_result_path = str(
        _canonical_no_final_symlink(
            Path(str(terminal.get("result_path"))), "GPU terminal result"
        )
    )
    if terminal.get("result_path") != resolved_result_path:
        raise LedgerError("terminal result path is not canonical")
    if terminal.get("gpu_execution_ledger_path") != resolved_execution_ledger:
        raise LedgerError("terminal belongs to a different GPU execution ledger")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "gpu_budget_terminal_result",
        "campaign_id": terminal["campaign_id"],
        "phase": terminal["phase"],
        "context": terminal["context"],
        "invocation_sha256": terminal["invocation_sha256"],
        "usage_ledger_path": resolved_usage_ledger,
        "gpu_execution_ledger_path": resolved_execution_ledger,
        "result_path": resolved_result_path,
        "lifecycle_id": terminal["lifecycle_id"],
        "reservation_record_sha256": terminal["reservation_record_sha256"],
        "terminal_record_sha256": terminal["record_sha256"],
        "terminal_event": terminal["event"],
        "command_sha256": terminal["command_sha256"],
        "return_code": terminal["return_code"],
        "wrapper_exit_code": terminal["wrapper_exit_code"],
        "hard_timeout_reached": terminal["hard_timeout_reached"],
        "termination_escalated": terminal["termination_escalated"],
        "containment_anomaly": terminal["containment_anomaly"],
        "reservation_deadline_breached": terminal[
            "reservation_deadline_breached"
        ],
        "charged_usage_ns": terminal["charged_usage_ns"],
        "elapsed_ns": terminal["elapsed_ns"],
        "reusable_success": terminal["reuse_eligible"],
    }


def load_validate_terminal_result(
    path: Path,
    *,
    usage_ledger: Path,
    expected_campaign_id: str | None = None,
    expected_phase: str | None = None,
    expected_context: Mapping[str, Any] | None = None,
    expected_command_sha256: str | None = None,
    expected_invocation_sha256: str | None = None,
    budget_ns: int = GPU_BUDGET_NS,
    expected_legacy_genesis_sha256: str | None = None,
    pinned_result_directory_fd: object | None = None,
    pinned_usage_directory_fd: int | None = None,
    allow_legacy_mutable_evidence: bool = False,
) -> dict[str, Any]:
    resolved = _canonical_no_final_symlink(path, "GPU terminal result")
    try:
        with _exclusive_result_lock(
            resolved,
            pinned_directory_fd=pinned_result_directory_fd,
            allow_legacy_mutable_evidence=allow_legacy_mutable_evidence,
        ) as locked_result:
            raw_bytes = locked_result.read()
        assert raw_bytes is not None
        raw = raw_bytes.decode("utf-8", errors="strict")
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                LedgerError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, LedgerError) as error:
        raise LedgerError(f"invalid GPU terminal result {resolved}: {error}") from error
    if not isinstance(value, dict):
        raise LedgerError("GPU terminal result must be an object")
    exact_payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if raw_bytes != exact_payload:
        raise LedgerError("GPU terminal result bytes are not canonical")
    recorded_content = value.get("content_sha256")
    if not _is_hex_sha256(recorded_content):
        raise LedgerError("GPU terminal result has invalid content hash")
    document = dict(value)
    document.pop("content_sha256", None)
    if semantic_sha256(document) != recorded_content:
        raise LedgerError("GPU terminal result content hash drifted")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise LedgerError("GPU terminal result schema drifted")
    if value.get("classification") != "gpu_budget_terminal_result":
        raise LedgerError("GPU terminal result classification drifted")
    state = verify_ledger(
        usage_ledger,
        budget_ns=budget_ns,
        expected_legacy_genesis_sha256=expected_legacy_genesis_sha256,
        pinned_directory_fd=pinned_usage_directory_fd,
    )
    matches = [
        record
        for record in state.records
        if record.get("record_sha256") == value.get("terminal_record_sha256")
    ]
    if len(matches) != 1 or matches[0].get("event") != "terminal":
        raise LedgerError("GPU terminal result does not bind one observed terminal")
    terminal = matches[0]
    resolved_result_path = str(resolved)
    if (
        value.get("result_path") != resolved_result_path
        or terminal.get("result_path") != resolved_result_path
    ):
        raise LedgerError("GPU terminal result path binding mismatched")
    expected = _result_content_document(
        result_from_terminal(
            terminal,
            usage_ledger=usage_ledger,
            gpu_execution_ledger=Path(str(value.get("gpu_execution_ledger_path"))),
        )
    )
    if value != expected:
        raise LedgerError("GPU terminal result differs from its ledger terminal")
    checks = (
        ("campaign_id", expected_campaign_id),
        ("phase", expected_phase),
        ("command_sha256", expected_command_sha256),
        ("invocation_sha256", expected_invocation_sha256),
    )
    for field, expected_value in checks:
        if expected_value is not None and value.get(field) != expected_value:
            raise LedgerError(f"GPU terminal result {field} mismatched")
    if expected_context is not None and canonical_json_bytes(value.get("context")) != canonical_json_bytes(
        dict(expected_context)
    ):
        raise LedgerError("GPU terminal result context mismatched")
    if value.get("usage_ledger_path") != str(
        _canonical_no_final_symlink(usage_ledger, "GPU usage ledger")
    ):
        raise LedgerError("GPU terminal result belongs to a different usage ledger")
    return value
