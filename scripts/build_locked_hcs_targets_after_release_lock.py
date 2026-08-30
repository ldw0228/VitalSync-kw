#!/usr/bin/env python3
"""Build canonical HCS targets only through the immutable pretarget release lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_locked_hcs_targets as TARGETS  # noqa: E402
import create_locked_hcs_pretarget_release_lock as RELEASE  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = TARGETS.DEFAULT_CACHE_DIR
DEFAULT_FOLD_ASSIGNMENTS = TARGETS.DEFAULT_FOLD_ASSIGNMENTS
DEFAULT_RELEASE_LOCK = RELEASE.DEFAULT_RELEASE_LOCK


class ReleasedTargetBuildError(RuntimeError):
    """The release lock or create-once target publication is inconsistent."""


def _atomic_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ReleasedTargetBuildError(
                f"release receipt appeared concurrently: {destination}"
            ) from exc
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _published_target_receipt(
    *, target: Path, receipt: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target_raw = target.expanduser()
    receipt_raw = receipt.expanduser()
    if target_raw.is_symlink():
        raise ReleasedTargetBuildError("canonical target must not be a symlink")
    if receipt_raw.is_symlink():
        raise ReleasedTargetBuildError("canonical target receipt must not be a symlink")
    target_path = target_raw.resolve()
    receipt_path = receipt_raw.resolve()
    if not target_path.is_file():
        raise ReleasedTargetBuildError("canonical target is absent or not a regular file")
    if not receipt_path.is_file():
        raise ReleasedTargetBuildError("canonical target receipt is absent or not a regular file")
    if stat.S_IMODE(target_path.stat().st_mode) != 0o444:
        raise ReleasedTargetBuildError("canonical target mode must be exactly 0444")
    if stat.S_IMODE(receipt_path.stat().st_mode) != 0o444:
        raise ReleasedTargetBuildError("canonical target receipt mode must be exactly 0444")
    try:
        document = TARGETS._read_json(receipt_path, "canonical target receipt")  # noqa: SLF001
        target_binding = TARGETS.bind_file(target_path)
        receipt_binding = TARGETS.bind_file(receipt_path)
    except Exception as exc:
        raise ReleasedTargetBuildError(f"canonical target publication is invalid: {exc}") from exc
    if (
        document.get("classification")
        != "retrospective_locked_hcs_canonical_target_artifact_receipt"
        or document.get("target_artifact_created_once") is not True
        or document.get("target_artifact_overwrite_allowed") is not False
        or document.get("commercial_claim_authorized") is not False
        or document.get("target_artifact") != target_binding
    ):
        raise ReleasedTargetBuildError("canonical target receipt invariants are invalid")
    command = document.get("orchestrator_command")
    if (
        not isinstance(command, list)
        or len(command) < 2
        or Path(str(command[1])).expanduser().resolve() != Path(__file__).resolve()
    ):
        raise ReleasedTargetBuildError(
            "canonical target receipt was not executed through the release-lock wrapper"
        )
    return document, target_binding, receipt_binding


def _release_receipt_document(
    *,
    release_lock_path: Path,
    release_lock: Mapping[str, Any],
    target_document: Mapping[str, Any],
    target_binding: Mapping[str, Any],
    target_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_canonical_targets_built_after_pretarget_release",
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
        "release_lock_revalidated_before_target_builder_call": True,
        "all_release_bound_artifacts_rehashed_before_target_builder_call": True,
        "target_metadata_access_before_release_validation": False,
        "canonical_target_builder_called_only_after_release_authorization": True,
        "pretarget_release_lock": RELEASE.bind_file(release_lock_path),
        "pretarget_release_content_sha256": release_lock["content_sha256"],
        "canonical_target": dict(target_binding),
        "canonical_target_receipt": dict(target_receipt_binding),
        "canonical_target_receipt_content_sha256": target_document["content_sha256"],
        "target_builder_orchestrator_command": list(
            target_document.get("orchestrator_command", [])
        ),
        "effective_sources": {
            "released_target_builder": RELEASE.bind_file(Path(__file__)),
            "release_lock_creator": RELEASE.bind_file(Path(RELEASE.__file__)),
            "canonical_target_builder": RELEASE.bind_file(Path(TARGETS.__file__)),
            "python_executable": RELEASE.bind_file(Path(sys.executable)),
        },
    }
    document["content_sha256"] = RELEASE.canonical_json_sha256(document)
    return document


def _validate_release_receipt(
    *,
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ReleasedTargetBuildError(f"release receipt is absent: {resolved}")
    if stat.S_IMODE(resolved.stat().st_mode) != 0o444:
        raise ReleasedTargetBuildError("release receipt mode must be exactly 0444")
    try:
        observed = TARGETS._read_json(resolved, "target release receipt")  # noqa: SLF001
    except Exception as exc:
        raise ReleasedTargetBuildError(f"invalid target release receipt: {exc}") from exc
    if observed != expected:
        raise ReleasedTargetBuildError("target release receipt differs from live bindings")
    return observed


def build_targets_after_release_lock(
    *,
    release_lock_path: Path = DEFAULT_RELEASE_LOCK,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    fold_assignments: Path = DEFAULT_FOLD_ASSIGNMENTS,
    output: Path | None = None,
    receipt: Path | None = None,
    release_receipt: Path | None = None,
    orchestrator_command: Sequence[str] = (),
) -> dict[str, Any]:
    """Revalidate the release graph, build once, and publish a wrapper receipt."""

    lock_path = release_lock_path.expanduser().resolve()
    # Reading the non-target release lock is the first operation.  If either
    # canonical output already exists we take only the idempotent/resume path;
    # otherwise the validator requires every target/evaluation path absent.
    try:
        initial_lock = RELEASE.validate_release_lock(lock_path, require_target_absence=False)
    except Exception as exc:
        raise ReleasedTargetBuildError(f"pretarget release validation failed: {exc}") from exc
    locations = initial_lock["locations"]
    target_raw = Path(output or locations["canonical_target"]).expanduser()
    target_receipt_raw = Path(
        receipt or locations["canonical_target_receipt"]
    ).expanduser()
    wrapper_receipt_raw = Path(
        release_receipt or locations["release_receipt"]
    ).expanduser()
    for raw, label in (
        (target_raw, "canonical target"),
        (target_receipt_raw, "canonical target receipt"),
        (wrapper_receipt_raw, "target release receipt"),
    ):
        # This must precede resolve(): resolving first erases the evidence that
        # a locked publication path itself was replaced by a symlink.
        if raw.is_symlink():
            raise ReleasedTargetBuildError(f"{label} must not be a symlink")
    target_path = target_raw.resolve()
    target_receipt_path = target_receipt_raw.resolve()
    wrapper_receipt_path = wrapper_receipt_raw.resolve()
    if target_path != Path(locations["canonical_target"]).resolve():
        raise ReleasedTargetBuildError("target output differs from the release-locked path")
    if target_receipt_path != Path(locations["canonical_target_receipt"]).resolve():
        raise ReleasedTargetBuildError("target receipt differs from the release-locked path")
    if wrapper_receipt_path != Path(locations["release_receipt"]).resolve():
        raise ReleasedTargetBuildError("release receipt differs from the release-locked path")

    target_exists = target_path.exists()
    receipt_exists = target_receipt_path.exists()
    wrapper_exists = wrapper_receipt_path.exists()
    if target_exists != receipt_exists:
        raise ReleasedTargetBuildError(
            "partial canonical target publication is quarantined; automatic overwrite is forbidden"
        )
    if wrapper_exists and not (target_exists and receipt_exists):
        raise ReleasedTargetBuildError("release receipt exists without canonical target outputs")

    if not target_exists:
        # This is the decisive gate: it re-runs every primary/mask/uncertainty
        # validator and spec hash check before TARGETS.build_targets can be
        # reached.  Recheck create-once paths immediately afterward.
        try:
            release_lock = RELEASE.validate_release_lock(
                lock_path, require_target_absence=True
            )
        except Exception as exc:
            raise ReleasedTargetBuildError(f"pretarget release validation failed: {exc}") from exc
        if any(path.exists() for path in (target_path, target_receipt_path, wrapper_receipt_path)):
            raise ReleasedTargetBuildError("target output appeared during release validation")
        # Deliberately last: RF/SVD tree membership and bytes plus all three
        # completion/runtime guards are rehashed immediately before the only
        # call capable of opening canonical target metadata.
        try:
            RELEASE.reverify_runtime_inputs_from_lock(release_lock)
        except Exception as exc:
            raise ReleasedTargetBuildError(
                f"runtime payload closure revalidation failed: {exc}"
            ) from exc
        if any(path.exists() for path in (target_path, target_receipt_path, wrapper_receipt_path)):
            raise ReleasedTargetBuildError(
                "target output appeared during runtime closure revalidation"
            )
        try:
            builder_command = list(orchestrator_command) or [
                str(Path(sys.executable).resolve()),
                str(Path(__file__).resolve()),
            ]
            TARGETS.build_targets(
                locked_oof_root=Path(locations["primary_root"]),
                cache_dir=cache_dir,
                fold_assignments=fold_assignments,
                output=target_path,
                receipt=target_receipt_path,
                orchestrator_command=builder_command,
                _release_capability=TARGETS._RELEASE_AUTHORIZATION_CAPABILITY,  # noqa: SLF001
            )
        except Exception as exc:
            raise ReleasedTargetBuildError(f"canonical target builder failed: {exc}") from exc
    else:
        release_lock = initial_lock

    target_document, target_binding, target_receipt_binding = _published_target_receipt(
        target=target_raw, receipt=target_receipt_raw
    )
    expected = _release_receipt_document(
        release_lock_path=lock_path,
        release_lock=release_lock,
        target_document=target_document,
        target_binding=target_binding,
        target_receipt_binding=target_receipt_binding,
    )
    if wrapper_receipt_path.exists():
        return _validate_release_receipt(path=wrapper_receipt_path, expected=expected)
    _atomic_immutable_json(wrapper_receipt_path, expected)
    return _validate_release_receipt(path=wrapper_receipt_path, expected=expected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-lock", type=Path, default=DEFAULT_RELEASE_LOCK)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--fold-assignments", type=Path, default=DEFAULT_FOLD_ASSIGNMENTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--release-receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        *(argv or sys.argv[1:]),
    ]
    try:
        result = build_targets_after_release_lock(
            release_lock_path=args.release_lock,
            cache_dir=args.cache_dir,
            fold_assignments=args.fold_assignments,
            output=args.output,
            receipt=args.receipt,
            release_receipt=args.release_receipt,
            orchestrator_command=command,
        )
    except (ReleasedTargetBuildError, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
