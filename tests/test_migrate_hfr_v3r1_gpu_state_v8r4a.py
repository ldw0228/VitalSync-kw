from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import pytest


REAL_PROJECT = Path(__file__).resolve().parents[3] / "home/hwiseong/Documents/SnnProject"
if not REAL_PROJECT.exists():
    REAL_PROJECT = Path("/home/hwiseong/Documents/SnnProject")
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/migrate_hfr_v3r1_gpu_state_v8r4a.py"
)
SPEC = importlib.util.spec_from_file_location("v8r4a_migration_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
migration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = migration
SPEC.loader.exec_module(migration)

FIXED_UTC = "2026-08-29T07:00:00Z"
TOKEN = "cpuonlyfixture"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

GOVERNANCE_FILES = (
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_MIGRATION_SOURCE_SUCCESSION.json",
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A.json",
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4.json",
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "DISCOVERY_V8R3_ATTEMPT_000_QUARANTINE_OWNER_RECEIPT_V8R4.json",
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "DISCOVERY_V8R3_ATTEMPT_000_QUARANTINED_OUTPUT_SEAL_V8R4.json",
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v8r4_exact_file_atomic_replace_failure_v8r4a.json",
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v8r3_outer_capability_and_identity_dtype_failure_v8r4.json",
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/"
    "v3r1_v8r4a_migrated_state_source_succession_failure.json",
)
SOURCE_FILES = (
    "scripts/run_gpu_admitted.py",
    "src/snn_rr/gpu_budget_ledger.py",
)


def _copy_file(project: Path, relative: str, *, mode: int) -> Path:
    source = REAL_PROJECT / relative
    destination = project / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(mode)
    return destination


def _make_project(tmp_path: Path) -> tuple[Path, dict[str, tuple[int, int]]]:
    project = tmp_path / "project"
    project.mkdir()
    for relative in GOVERNANCE_FILES:
        _copy_file(project, relative, mode=0o444)
    for relative in SOURCE_FILES:
        _copy_file(project, relative, mode=0o444)
    for role, relative in migration.LEGACY_RELATIVE.items():
        source = REAL_PROJECT.joinpath(*relative.parts)
        destination = project.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o644)
    identities = {
        role: (path.stat().st_dev, path.stat().st_ino)
        for role, relative in migration.LEGACY_RELATIVE.items()
        if (path := project.joinpath(*relative.parts))
    }
    return project, identities


def _run(
    project: Path, identities: Mapping[str, tuple[int, int]]
) -> Any:
    return migration._migrate_gpu_state(
        project,
        created_utc=FIXED_UTC,
        temporary_token=TOKEN,
        expected_original_identities=identities,
    )


def _validate(
    project: Path,
    identities: Mapping[str, tuple[int, int]],
    *,
    require_closed: bool = True,
) -> Any:
    root = migration._canonical_project_root(project)
    authority = migration._validate_authority(root)
    return migration._live_validation(
        root,
        authority,
        receipt_path=project.joinpath(*migration.RECEIPT_RELATIVE.parts),
        expected_identities=identities,
        require_closed=require_closed,
    )


def _receipt_path(project: Path) -> Path:
    return project.joinpath(*migration.RECEIPT_RELATIVE.parts)


def _target_path(project: Path, role: str) -> Path:
    return project.joinpath(*migration.TARGET_RELATIVE[role].parts)


def _legacy_path(project: Path, role: str) -> Path:
    return project.joinpath(*migration.LEGACY_RELATIVE[role].parts)


def _trusted_prelaunch(validated: Any) -> dict[str, Any]:
    return {
        "migration_receipt": dict(validated.receipt_binding),
        "directories": dict(validated.directory_bindings),
        "files": dict(validated.current_file_bindings),
        "usage_state": dict(validated.usage_state),
        "execution_state": dict(validated.execution_state),
    }


def _rewrite_json(path: Path, document: Mapping[str, Any], *, mode: int = 0o444) -> None:
    path.chmod(0o644)
    path.write_bytes(migration.pretty_json_bytes(document))
    path.chmod(mode)


def test_migration_publishes_exact_tree_receipt_and_retires_legacy(
    tmp_path: Path,
) -> None:
    project, identities = _make_project(tmp_path)
    original_bytes = {
        role: _legacy_path(project, role).read_bytes()
        for role in migration.FILE_ROLES
    }
    original_inodes = {
        role: _legacy_path(project, role).stat().st_ino
        for role in migration.FILE_ROLES
    }

    result = _run(project, identities)

    assert result.resumed is False
    assert result.receipt_path == _receipt_path(project)
    assert result.receipt_path.stat().st_mode & 0o777 == 0o444
    assert result.receipt_path.stat().st_nlink == 1
    assert set(result.receipt) == migration.RECEIPT_KEYS
    assert result.receipt["production_runtime_authorized"] is False
    assert result.receipt["content_sha256"] == migration.semantic_sha256(
        result.receipt
    )
    assert result.receipt_path.read_bytes() == migration.pretty_json_bytes(
        result.receipt
    )
    assert result.receipt["lifecycle_state"] == {
        "all_lifecycles_closed": True,
        "execution_open_start_count": 0,
        "usage_open_reservation_count": 0,
    }
    assert result.receipt["prefix_replay"]["usage"][
        "tail_record_sha256"
    ] == "8dbc0493125f22c130444e1344533d1f3d9c4ac445df6adfe8b597972e9691c5"
    root = project.joinpath(*migration.TARGET_ROOT_RELATIVE.parts)
    assert sorted(path.name for path in root.iterdir()) == [
        "admission",
        "execution",
        "usage",
    ]
    for role, entries in migration.ROLE_ENTRIES.items():
        directory = project.joinpath(*migration.ROLE_DIRECTORIES[role].parts)
        assert directory.stat().st_mode & 0o777 == 0o700
        assert sorted(path.name for path in directory.iterdir()) == sorted(entries)
    for role in migration.FILE_ROLES:
        legacy = _legacy_path(project, role)
        migrated = _target_path(project, role)
        assert legacy.read_bytes() == original_bytes[role]
        assert legacy.stat().st_ino == original_inodes[role]
        assert legacy.stat().st_mode & 0o777 == 0o444
        assert legacy.stat().st_nlink == 1
        assert migrated.stat().st_mode & 0o777 == 0o644
        assert migrated.stat().st_nlink == 1
        assert migrated.stat().st_ino != legacy.stat().st_ino
        expected = original_bytes[role] if role in migration.LEDGER_ROLES else b""
        assert migrated.read_bytes() == expected


def test_resume_is_create_once_byte_identical_and_read_only(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    first = _run(project, identities)
    receipt_before = first.receipt_path.read_bytes()
    file_state_before = {
        role: (
            _target_path(project, role).stat().st_dev,
            _target_path(project, role).stat().st_ino,
            _target_path(project, role).read_bytes(),
        )
        for role in migration.FILE_ROLES
    }

    second = _run(project, identities)

    assert second.resumed is True
    assert second.receipt_path.read_bytes() == receipt_before
    assert {
        role: (
            _target_path(project, role).stat().st_dev,
            _target_path(project, role).stat().st_ino,
            _target_path(project, role).read_bytes(),
        )
        for role in migration.FILE_ROLES
    } == file_state_before


def test_live_validator_returns_exact_capability_contract(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)

    validated = _validate(project, identities)

    assert set(validated.canonical_paths) == {
        *migration.FILE_ROLES,
        "admission_directory",
        "execution_directory",
        "usage_directory",
    }
    assert all(path.is_absolute() for path in validated.canonical_paths.values())
    assert set(validated.directory_bindings) == {
        "root",
        "admission",
        "execution",
        "usage",
    }
    assert set(validated.current_file_bindings) == set(migration.FILE_ROLES)
    assert validated.usage_state["open_reservation_count"] == 0
    assert validated.usage_state["record_count"] == 75
    assert validated.execution_state["open_start_count"] == 0
    assert validated.execution_state["record_count"] == 8
    assert validated.current_file_bindings["usage_ledger"]["path"] == (
        migration.TARGET_RELATIVE["usage_ledger"].as_posix()
    )


def test_lock_free_validator_accepts_atomic_ledger_inode_replacement(
    tmp_path: Path,
) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    validated = _validate(project, identities)
    trusted = _trusted_prelaunch(validated)
    before = {
        role: _target_path(project, role).stat().st_ino
        for role in migration.LEDGER_ROLES
    }
    for role in migration.LEDGER_ROLES:
        path = _target_path(project, role)
        temporary = path.with_name(f".{path.name}.atomic-fixture")
        temporary.write_bytes(path.read_bytes())
        temporary.chmod(0o644)
        os.replace(temporary, path)
        assert path.stat().st_ino != before[role]

    replayed = migration.validate_migrated_state_lock_free(
        project,
        _receipt_path(project),
        trusted_prelaunch_state=trusted,
        require_closed=True,
    )

    assert replayed.usage_state == validated.usage_state
    assert replayed.execution_state == validated.execution_state
    assert (
        replayed.current_file_bindings["usage_ledger"]["st_ino"]
        != trusted["files"]["usage_ledger"]["st_ino"]
    )


def test_target_scoped_validator_takes_locks_around_replay(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    trusted = _trusted_prelaunch(_validate(project, identities))
    admission = _target_path(project, "admission_lock")
    descriptor = os.open(admission, os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(migration.MigrationError, match="already locked"):
            migration.validate_migrated_state_target_scoped(
                project,
                _receipt_path(project),
                trusted_prelaunch_state=trusted,
                require_closed=True,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_lock_free_validator_requires_stable_lock_inode(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    validated = _validate(project, identities)
    trusted = _trusted_prelaunch(validated)
    path = _target_path(project, "usage_ledger_lock")
    replacement = path.with_name(f".{path.name}.replacement")
    replacement.write_bytes(b"")
    replacement.chmod(0o644)
    os.replace(replacement, path)

    with pytest.raises(migration.MigrationError, match="stable inode"):
        migration.validate_migrated_state_lock_free(
            project,
            _receipt_path(project),
            trusted_prelaunch_state=trusted,
            require_closed=True,
        )


def test_lock_free_validator_rejects_capability_prefix_rewrite(
    tmp_path: Path,
) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    trusted = _trusted_prelaunch(_validate(project, identities))
    path = _target_path(project, "execution_ledger")
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    replacement = path.with_name(f".{path.name}.replacement")
    replacement.write_bytes(raw)
    replacement.chmod(0o644)
    os.replace(replacement, path)

    with pytest.raises(migration.MigrationError, match="trusted prefix"):
        migration.validate_migrated_state_lock_free(
            project,
            _receipt_path(project),
            trusted_prelaunch_state=trusted,
            require_closed=False,
        )


def test_public_validator_requires_diagnostic_bound_production_inodes(
    tmp_path: Path,
) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    with pytest.raises(migration.MigrationError, match="identity or mode"):
        migration.validate_migrated_state(
            project, _receipt_path(project), require_closed=True
        )


def test_live_validator_refuses_noncanonical_receipt_argument(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    root = migration._canonical_project_root(project)
    authority = migration._validate_authority(root)
    with pytest.raises(migration.MigrationError, match="receipt path"):
        migration._live_validation(
            root,
            authority,
            receipt_path=project / "wrong.json",
            expected_identities=identities,
            require_closed=True,
        )


def test_legacy_admission_lock_contention_refuses_before_publication(
    tmp_path: Path,
) -> None:
    project, identities = _make_project(tmp_path)
    descriptor = os.open(_legacy_path(project, "admission_lock"), os.O_RDONLY)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(migration.MigrationError, match="already locked"):
            _run(project, identities)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert not project.joinpath(*migration.TARGET_ROOT_RELATIVE.parts).exists()
    assert not _receipt_path(project).exists()


@pytest.mark.parametrize("role", ["usage_ledger_lock", "execution_ledger_lock"])
def test_legacy_ledger_lock_contention_refuses(
    tmp_path: Path, role: str
) -> None:
    project, identities = _make_project(tmp_path)
    descriptor = os.open(_legacy_path(project, role), os.O_RDONLY)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(migration.MigrationError, match="already locked"):
            _run(project, identities)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@pytest.mark.parametrize("role", ["usage_ledger", "execution_ledger"])
def test_legacy_ledger_tamper_refuses_exact_prefix(
    tmp_path: Path, role: str
) -> None:
    project, identities = _make_project(tmp_path)
    path = _legacy_path(project, role)
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(raw)
    with pytest.raises(migration.MigrationError, match="diagnostic state"):
        _run(project, identities)
    assert not _receipt_path(project).exists()


def test_frozen_gpu_budget_source_tamper_refuses(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    source = project / "src/snn_rr/gpu_budget_ledger.py"
    source.chmod(0o644)
    source.write_bytes(source.read_bytes() + b"\n")
    source.chmod(0o444)
    with pytest.raises(migration.MigrationError, match="source gpu_budget_ledger drifted"):
        _run(project, identities)


def test_successor_wrapper_source_tamper_refuses(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    source = project / "scripts/run_gpu_admitted.py"
    source.chmod(0o644)
    source.write_bytes(source.read_bytes() + b"\n")
    source.chmod(0o444)
    with pytest.raises(migration.MigrationError, match="source run_gpu_admitted drifted"):
        _run(project, identities)


@pytest.mark.parametrize(
    "relative",
    [
        migration.SOURCE_SUCCESSION_AUTHORIZATION_RELATIVE,
        migration.SOURCE_SUCCESSION_DIAGNOSTIC_RELATIVE,
    ],
)
def test_source_succession_governance_tamper_refuses(
    tmp_path: Path, relative: Any
) -> None:
    project, identities = _make_project(tmp_path)
    path = project.joinpath(*relative.parts)
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    path.chmod(0o444)
    with pytest.raises(migration.MigrationError, match="file sha256"):
        _run(project, identities)


def test_historical_quarantine_evidence_tamper_refuses(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    path = project / (
        "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "DISCOVERY_V8R3_ATTEMPT_000_QUARANTINE_OWNER_RECEIPT_V8R4.json"
    )
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    path.chmod(0o444)
    with pytest.raises(migration.MigrationError, match="file sha256"):
        _run(project, identities)


def test_authorization_tamper_refuses_even_with_valid_json(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    path = project.joinpath(*migration.AUTHORIZATION_RELATIVE.parts)
    document = json.loads(path.read_text())
    document["created_utc"] = "2026-08-29T00:00:00Z"
    document["content_sha256"] = migration.semantic_sha256(document)
    _rewrite_json(path, document)
    with pytest.raises(migration.MigrationError, match="file sha256"):
        _run(project, identities)


def test_authorization_duplicate_key_refuses(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    path = project.joinpath(*migration.AUTHORIZATION_RELATIVE.parts)
    path.chmod(0o644)
    path.write_text('{"schema_version":1,"schema_version":1}\n')
    path.chmod(0o444)
    with pytest.raises(migration.MigrationError, match="file sha256"):
        _run(project, identities)


def test_legacy_symlink_refuses_nofollow(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    path = _legacy_path(project, "usage_ledger")
    backup = path.with_name("outside-ledger")
    path.rename(backup)
    path.symlink_to(backup.name)
    with pytest.raises(migration.MigrationError, match="legacy usage_ledger"):
        _run(project, identities)


def test_legacy_hardlink_refuses_single_link(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    path = _legacy_path(project, "execution_ledger")
    os.link(path, path.with_name("execution-alias"))
    with pytest.raises(migration.MigrationError, match="aliased"):
        _run(project, identities)


def test_receiptless_partial_tree_refuses_without_repair(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    root = project.joinpath(*migration.TARGET_ROOT_RELATIVE.parts)
    root.mkdir(parents=True)
    with pytest.raises(migration.MigrationError, match="partial migration state"):
        _run(project, identities)
    assert root.exists()
    assert not _receipt_path(project).exists()


def test_receipt_without_tree_refuses_without_overwrite(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    path = _receipt_path(project)
    path.write_text("{}\n")
    path.chmod(0o444)
    before = path.read_bytes()
    with pytest.raises(migration.MigrationError, match="partial migration state"):
        _run(project, identities)
    assert path.read_bytes() == before


def test_existing_empty_target_root_is_not_reused(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    root = project.joinpath(*migration.TARGET_ROOT_RELATIVE.parts)
    root.mkdir(parents=True)
    receipt = _receipt_path(project)
    receipt.write_text("{}\n")
    receipt.chmod(0o444)
    with pytest.raises(migration.MigrationError):
        _run(project, identities)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    "relative",
    [
        migration.TARGET_ROOT_RELATIVE / "unexpected",
        migration.ROLE_DIRECTORIES["usage"] / "unexpected",
    ],
)
def test_live_validator_refuses_extra_tree_entry(
    tmp_path: Path, relative: Any
) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    project.joinpath(*relative.parts).write_text("poison")
    with pytest.raises(migration.MigrationError, match="inventory drifted"):
        _validate(project, identities)


def test_live_validator_refuses_migrated_hardlink(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    path = _target_path(project, "execution_ledger_lock")
    os.link(path, path.with_name("lock-alias"))
    with pytest.raises(migration.MigrationError):
        _validate(project, identities)


def test_live_validator_refuses_migrated_symlink(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    path = _target_path(project, "admission_lock")
    backup = path.with_name("lock-backup")
    path.rename(backup)
    path.symlink_to(backup.name)
    with pytest.raises(migration.MigrationError):
        _validate(project, identities)


def test_live_validator_refuses_legacy_same_bytes_new_inode(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    path = _legacy_path(project, "usage_ledger")
    raw = path.read_bytes()
    detached = project / "detached-legacy-usage"
    path.rename(detached)
    path.write_bytes(raw)
    path.chmod(0o444)
    with pytest.raises(migration.MigrationError, match="identity or mode"):
        _validate(project, identities)


def test_live_validator_refuses_replaced_role_directory_inode(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    directory = project.joinpath(*migration.ROLE_DIRECTORIES["usage"].parts)
    detached = project / "detached-usage-directory"
    directory.rename(detached)
    directory.mkdir(mode=0o700)
    for filename in migration.ROLE_ENTRIES["usage"]:
        shutil.copyfile(detached / filename, directory / filename)
        (directory / filename).chmod(0o644)
    with pytest.raises(migration.MigrationError, match="directory inode inventory"):
        _validate(project, identities)


def test_live_validator_refuses_role_directory_mode_drift(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    directory = project.joinpath(*migration.ROLE_DIRECTORIES["execution"].parts)
    directory.chmod(0o755)
    with pytest.raises(migration.MigrationError, match="mode drifted"):
        _validate(project, identities)


def test_receipt_content_tamper_refuses(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    path = _receipt_path(project)
    document = json.loads(path.read_text())
    document["created_utc"] = "2026-08-29T07:00:01Z"
    _rewrite_json(path, document)
    with pytest.raises(migration.MigrationError, match="content_sha256"):
        _validate(project, identities)


def test_receipt_duplicate_key_refuses(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    path = _receipt_path(project)
    path.chmod(0o644)
    path.write_text('{"schema_version":1,"schema_version":1}\n')
    path.chmod(0o444)
    with pytest.raises(migration.MigrationError, match="duplicate JSON key"):
        _validate(project, identities)


@pytest.mark.parametrize("role", ["usage_ledger", "execution_ledger"])
def test_live_validator_refuses_migrated_prefix_tamper(
    tmp_path: Path, role: str
) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    path = _target_path(project, role)
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(raw)
    with pytest.raises(migration.MigrationError):
        _validate(project, identities)


def test_live_validator_refuses_broken_usage_hash_chain_suffix(
    tmp_path: Path,
) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    path = _target_path(project, "usage_ledger")
    path.write_bytes(
        path.read_bytes()
        + migration.canonical_json_bytes(
            {
                "event": "reservation",
                "previous_record_sha256": "0" * 64,
                "record_sha256": "1" * 64,
                "schema_version": 2,
            }
        )
        + b"\n"
    )
    with pytest.raises(migration.MigrationError, match="usage ledger replay failed"):
        _validate(project, identities)


def test_live_validator_refuses_orphan_execution_suffix(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    path = _target_path(project, "execution_ledger")
    path.write_bytes(
        path.read_bytes()
        + migration.canonical_json_bytes(
            {"event": "end", "lifecycle_id": "orphan-v8r4a"}
        )
        + b"\n"
    )
    with pytest.raises(migration.MigrationError, match="orphan"):
        _validate(project, identities)


def test_valid_future_usage_prefix_with_open_reservation_obeys_closed_gate(
    tmp_path: Path,
) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    root = migration._canonical_project_root(project)
    authority = migration._validate_authority(root)
    budget = migration._load_budget_module(root, authority)
    usage = _target_path(project, "usage_ledger")
    template = {
        "boot_id": "cpu-only-fixture-boot",
        "campaign_id": migration.CAMPAIGN_ID,
        "command_sha256": "b" * 64,
        "context": {"fixture": True},
        "gpu_execution_ledger_path": str(
            _target_path(project, "execution_ledger")
        ),
        "invocation_sha256": "a" * 64,
        "lifecycle_id": "cpu-only-open-reservation",
        "monotonic_ns": 2_000_000_000,
        "phase": "discovery",
        "realtime_ns": 2_000_000_000,
        "result_path": str(project / "synthetic-result.json"),
        "wrapper_pid": os.getpid(),
        "wrapper_start_ticks": 1,
    }
    budget.reconcile_and_reserve(
        usage,
        template,
        expected_legacy_genesis_sha256=migration.LEGACY_GENESIS_SHA256,
    )
    with pytest.raises(migration.MigrationError, match="open reservations"):
        _validate(project, identities, require_closed=True)
    validated = _validate(project, identities, require_closed=False)
    assert validated.usage_state["open_reservation_count"] == 1
    assert _target_path(project, "usage_ledger").read_bytes().startswith(
        _legacy_path(project, "usage_ledger").read_bytes()
    )


def test_valid_future_execution_prefix_with_open_start_obeys_closed_gate(
    tmp_path: Path,
) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    path = _target_path(project, "execution_ledger")
    path.write_bytes(
        path.read_bytes()
        + migration.canonical_json_bytes(
            {"event": "start", "lifecycle_id": "cpu-only-open-start"}
        )
        + b"\n"
    )
    with pytest.raises(migration.MigrationError, match="open starts"):
        _validate(project, identities, require_closed=True)
    validated = _validate(project, identities, require_closed=False)
    assert validated.execution_state["open_start_count"] == 1


def test_historical_state_path_is_forbidden_outside_exact_prefix(
    tmp_path: Path,
) -> None:
    project, identities = _make_project(tmp_path)
    _run(project, identities)
    path = _target_path(project, "execution_ledger")
    path.write_bytes(
        path.read_bytes()
        + migration.canonical_json_bytes(
            {
                "event": "start",
                "legacy_lock_file": str(_legacy_path(project, "admission_lock")),
                "lifecycle_id": "old-path-outside-prefix",
            }
        )
        + b"\n"
    )
    with pytest.raises(migration.MigrationError, match="legacy state path"):
        _validate(project, identities, require_closed=False)


@pytest.mark.parametrize(
    "role", ["admission_lock", "usage_ledger_lock", "execution_ledger_lock"]
)
def test_nonempty_legacy_lock_refuses(tmp_path: Path, role: str) -> None:
    project, identities = _make_project(tmp_path)
    _legacy_path(project, role).write_text("not-a-lock\n")
    with pytest.raises(migration.MigrationError, match="is not empty"):
        _run(project, identities)


def test_original_mode_drift_refuses_before_copy(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    _legacy_path(project, "usage_ledger").chmod(0o444)
    with pytest.raises(migration.MigrationError, match="diagnostic state"):
        _run(project, identities)


def test_identity_override_must_cover_exact_five_roles(tmp_path: Path) -> None:
    project, identities = _make_project(tmp_path)
    identities.pop("admission_lock")
    with pytest.raises(migration.MigrationError, match="role set"):
        _run(project, identities)


def test_execution_decoder_rejects_duplicate_keys_and_orphan_terminal() -> None:
    with pytest.raises(migration.MigrationError, match="duplicate JSON key"):
        migration._decode_execution_ledger(
            b'{"event":"start","event":"start","lifecycle_id":"x"}\n'
        )
    with pytest.raises(migration.MigrationError, match="orphan"):
        migration._decode_execution_ledger(
            migration.canonical_json_bytes(
                {"event": "wrapper_exception", "lifecycle_id": "x"}
            )
            + b"\n"
        )


def test_no_gpu_or_target_paths_are_part_of_migration_contract() -> None:
    source = MODULE_PATH.read_text()
    assert "subprocess" not in source
    assert "torch" not in source
    assert "outer_fold" not in source
    assert "target_rr" not in source
    assert "CUDA" not in source
