from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
REAL_PROJECT = Path("/home/hwiseong/Documents/SnnProject")
SOURCE_ROOT = ROOT if (ROOT / "artifacts").is_dir() else REAL_PROJECT
MODULE_PATH = ROOT / "scripts/validate_hfr_v3r1_authorization.py"
SPEC = importlib.util.spec_from_file_location("v8r4a_authorization_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
authorization = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = authorization
SPEC.loader.exec_module(authorization)

RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "v8r4a_target_runtime_for_governance_abi_test",
    ROOT / "scripts/run_hfr_v3r1_target_sealed.py",
)
assert RUNTIME_SPEC is not None and RUNTIME_SPEC.loader is not None
target_runtime = importlib.util.module_from_spec(RUNTIME_SPEC)
sys.modules[RUNTIME_SPEC.name] = target_runtime
RUNTIME_SPEC.loader.exec_module(target_runtime)


def test_active_test_runner_uses_lexical_virtualenv_entry() -> None:
    interpreter = authorization._active_venv_interpreter(REAL_PROJECT)
    assert interpreter == REAL_PROJECT / ".venv/bin/python"
    assert interpreter != interpreter.resolve()


_ACTIVE_COLLECTION_CACHE: tuple[str, ...] | None = None


def _exact_active_test_collection() -> tuple[str, ...]:
    """Collect the covered suite exactly once for receipt-inventory fixtures."""

    global _ACTIVE_COLLECTION_CACHE
    if _ACTIVE_COLLECTION_CACHE is None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-o",
                "addopts=",
                *authorization.ACTIVE_FIXED_TEST_PATHS,
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout
        nodeids = sorted(
            line
            for line in completed.stdout.splitlines()
            if any(
                line == path or line.startswith(path + "::")
                for path in authorization.ACTIVE_FIXED_TEST_PATHS
            )
        )
        assert len(nodeids) == authorization._ACTIVE_FIXED_TEST_COLLECTION_COUNT
        assert (
            authorization._active_test_collection_sha256(nodeids)
            == authorization._ACTIVE_FIXED_TEST_COLLECTION_SHA256
        )
        _ACTIVE_COLLECTION_CACHE = tuple(nodeids)
    return _ACTIVE_COLLECTION_CACHE


def _valid_active_test_evidence(
    root: Path, implementation: list[Mapping[str, Any]]
) -> dict[str, Any]:
    collection = list(_exact_active_test_collection())
    outcomes = [{"nodeid": nodeid, "outcome": "passed"} for nodeid in collection]
    counts = {
        "collected": len(collection),
        "passed": len(collection),
        "skipped": 0,
        "failed": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "deselected": 0,
    }
    stdout = (
        "................................................................ [100%]\n"
        f"{len(collection)} passed in 5.86s\n"
    )
    encoded = stdout.encode("utf-8")
    runtime_state = {
        key: {} for key in authorization._ACTIVE_RUNTIME_STATE_KEYS
    }
    return {
        "test_paths": list(authorization.ACTIVE_FIXED_TEST_PATHS),
        "command": authorization._active_fixed_test_command(root),
        "return_code": 0,
        "stdout_sha256": hashlib.sha256(encoded).hexdigest(),
        "stdout_bytes": len(encoded),
        "stdout_tail": stdout,
        "stdout_is_complete": True,
        "collection_inventory": collection,
        "outcome_inventory": outcomes,
        "outcome_counts": counts,
        "inventory_sha256": authorization._active_test_inventory_sha256(
            collection, outcomes, counts
        ),
        "sandbox_enforcement_receipt": {
            "path": authorization.ACTIVE_TEST_ENFORCEMENT_RECEIPT.as_posix(),
            "sha256": "a" * 64,
            "bytes": 8192,
            "mode": "0444",
            "nlink": 1,
            "st_dev": 11,
            "st_ino": 22,
        },
        "implementation_files": list(implementation),
        "runtime_state_before": runtime_state,
        "runtime_state_after": json.loads(json.dumps(runtime_state)),
    }


def _replace_test_stdout(document: dict[str, Any], stdout: str) -> None:
    encoded = stdout.encode("utf-8")
    document["stdout_tail"] = stdout
    document["stdout_bytes"] = len(encoded)
    document["stdout_sha256"] = hashlib.sha256(encoded).hexdigest()


def _rehash_test_inventory(document: dict[str, Any]) -> None:
    document["inventory_sha256"] = authorization._active_test_inventory_sha256(
        document["collection_inventory"],
        document["outcome_inventory"],
        document["outcome_counts"],
    )


def test_active_test_result_payload_accepts_only_the_exact_pinned_collection() -> None:
    root = Path("/project")
    implementation = [{"path": "scripts/example.py"}]
    document = _valid_active_test_evidence(root, implementation)

    authorization._active_validate_test_result_payload(document)


def test_active_test_receipt_shape_cannot_bypass_missing_external_authority() -> None:
    root = Path("/project")
    implementation = [{"path": "scripts/example.py"}]
    document = _valid_active_test_evidence(root, implementation)

    with pytest.raises(
        authorization.AuthorizationError, match="no governed independently"
    ):
        authorization._active_validate_test_receipt_evidence(
            root, document, implementation=implementation
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "test_paths",
        "command",
        "boolean_return_code",
        "nonzero_return_code",
        "incomplete_stdout",
        "stdout_bytes",
        "stdout_sha256",
        "missing_progress_completion",
        "failure_summary",
        "skipped_test",
        "missing_bwrap_test",
        "inventory_hash",
        "enforcement_binding",
        "implementation_files",
        "runtime_state",
    ),
)
def test_active_test_receipt_exact_evidence_rejects_mutation(
    mutation: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Structural-only exercise: production calls never replace the authority
    # check, and separate tests above prove the real entry point is blocked.
    monkeypatch.setattr(
        authorization, "_active_require_test_enforcement_authority", lambda: None
    )
    root = Path("/project")
    implementation = [{"path": "scripts/example.py"}]
    document = _valid_active_test_evidence(root, implementation)
    if mutation == "test_paths":
        document["test_paths"] = ["tests/not_the_fixed_suite.py"]
    elif mutation == "command":
        document["command"] = ["/bin/true"]
    elif mutation == "boolean_return_code":
        document["return_code"] = False
    elif mutation == "nonzero_return_code":
        document["return_code"] = 1
    elif mutation == "incomplete_stdout":
        document["stdout_is_complete"] = False
    elif mutation == "stdout_bytes":
        document["stdout_bytes"] += 1
    elif mutation == "stdout_sha256":
        document["stdout_sha256"] = "0" * 64
    elif mutation == "missing_progress_completion":
        _replace_test_stdout(document, "17 passed in 5.86s\n")
    elif mutation == "failure_summary":
        _replace_test_stdout(
            document,
            "........................................................ [100%]\n"
            "1 failed, 16 passed in 5.86s\n",
        )
    elif mutation == "skipped_test":
        document["outcome_inventory"][0]["outcome"] = "skipped"
        document["outcome_counts"]["passed"] -= 1
        document["outcome_counts"]["skipped"] = 1
        _rehash_test_inventory(document)
    elif mutation == "missing_bwrap_test":
        missing = next(iter(authorization._ACTIVE_CRITICAL_BWRAP_TEST_NODEIDS))
        index = document["collection_inventory"].index(missing)
        document["collection_inventory"].pop(index)
        document["outcome_inventory"].pop(index)
        document["outcome_counts"]["collected"] -= 1
        document["outcome_counts"]["passed"] -= 1
        _rehash_test_inventory(document)
    elif mutation == "inventory_hash":
        document["inventory_sha256"] = "0" * 64
    elif mutation == "enforcement_binding":
        document["sandbox_enforcement_receipt"]["mode"] = "0644"
    elif mutation == "implementation_files":
        document["implementation_files"] = []
    else:
        document["runtime_state_after"]["usage_state"] = {"mutated": True}

    with pytest.raises(authorization.AuthorizationError, match="V8R4A"):
        authorization._active_validate_test_receipt_evidence(
            root, document, implementation=implementation
        )


def _valid_test_enforcement_document(
    root: Path, implementation: list[Mapping[str, Any]]
) -> dict[str, Any]:
    evidence = _valid_active_test_evidence(root, implementation)
    evidence.pop("sandbox_enforcement_receipt")
    document: dict[str, Any] = {
        "schema_version": 1,
        "classification": authorization.ACTIVE_TEST_ENFORCEMENT_CLASSIFICATION,
        "campaign_id": (
            "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        ),
        "scientific_campaign_revision": authorization.ACTIVE_SCIENTIFIC_REVISION,
        "infrastructure_revision": authorization.ACTIVE_INFRASTRUCTURE_REVISION,
        "authorization_generation": authorization.ACTIVE_AUTHORIZATION_GENERATION,
        "created_utc": "2026-08-31T00:00:00+00:00",
        **evidence,
        "sandbox_capability": {
            "enforcement_kind": "external_full_capability_sandbox_v1",
            "separate_mount_namespace": True,
            "separate_network_namespace": True,
            "minimal_device_namespace": True,
            "filesystem_allowlist_complete": True,
            "gpu_device_nodes_available": False,
            "raw_dataset_roots_available": False,
            "outer_reference_roots_available": False,
            "denied_canary_probes_passed": True,
        },
        "sandbox_completion": {
            "sandbox_process_started": True,
            "sandbox_process_exited": True,
            "return_code": 0,
            "stdout_sha256": evidence["stdout_sha256"],
            "inventory_sha256": evidence["inventory_sha256"],
            "gpu_accessed": False,
            "target_or_outer_reference_accessed": False,
        },
    }
    document["content_sha256"] = authorization.semantic_sha256(document)
    return document


def test_handmade_self_hashed_enforcement_receipt_is_non_authoritative(
    tmp_path: Path,
) -> None:
    implementation = [{"path": "scripts/example.py"}]
    document = _valid_test_enforcement_document(tmp_path, implementation)
    receipt = _write_frozen_json(
        tmp_path, authorization.ACTIVE_TEST_ENFORCEMENT_RECEIPT, document
    )

    with pytest.raises(
        authorization.AuthorizationError, match="handmade.*self-hashed"
    ):
        authorization._active_validate_test_enforcement_receipt(
            tmp_path, implementation
        )

    assert receipt.stat().st_mode & 0o777 == 0o444


def test_self_signed_enforcement_receipt_is_non_authoritative(
    tmp_path: Path,
) -> None:
    implementation = [{"path": "scripts/example.py"}]
    document = _valid_test_enforcement_document(tmp_path, implementation)
    unsigned = authorization.semantic_sha256(document)
    document["authority"] = {
        "trust_root_id": "repository-local-key",
        "issuer_id": "same-repository-writer",
        "runner_id": "same-repository-writer",
        "signature_scheme": "sha256-self-signature",
        "signed_payload_sha256": unsigned,
        "signature": unsigned,
    }
    document.pop("content_sha256")
    document["content_sha256"] = authorization.semantic_sha256(document)
    _write_frozen_json(
        tmp_path, authorization.ACTIVE_TEST_ENFORCEMENT_RECEIPT, document
    )

    with pytest.raises(
        authorization.AuthorizationError, match="self-signed.*non-authoritative"
    ):
        authorization._active_validate_test_enforcement_receipt(
            tmp_path, implementation
        )


def test_partial_rehashed_collection_cannot_satisfy_exact_inventory() -> None:
    document = _valid_active_test_evidence(Path("/project"), [])
    document["collection_inventory"].pop()
    document["outcome_inventory"].pop()
    document["outcome_counts"]["collected"] -= 1
    document["outcome_counts"]["passed"] -= 1
    _rehash_test_inventory(document)

    with pytest.raises(authorization.AuthorizationError, match="exact pinned"):
        authorization._active_validate_test_inventory(document)


def test_invented_one_node_per_file_cannot_satisfy_exact_inventory() -> None:
    document = _valid_active_test_evidence(Path("/project"), [])
    collection = sorted(
        {
            *(
                f"{path}::test_receipt_inventory_fixture"
                for path in authorization.ACTIVE_FIXED_TEST_PATHS
            ),
            *authorization._ACTIVE_CRITICAL_BWRAP_TEST_NODEIDS,
        }
    )
    document["collection_inventory"] = collection
    document["outcome_inventory"] = [
        {"nodeid": nodeid, "outcome": "passed"} for nodeid in collection
    ]
    document["outcome_counts"]["collected"] = len(collection)
    document["outcome_counts"]["passed"] = len(collection)
    _rehash_test_inventory(document)

    with pytest.raises(authorization.AuthorizationError, match="exact pinned"):
        authorization._active_validate_test_inventory(document)


def test_context1_authority_blocker_requires_independent_signed_bindings() -> None:
    assert authorization._ACTIVE_CONTEXT1_TEST_AUTHORITY_STATUS == (
        "blocked_no_independent_trust_root"
    )
    assert authorization._ACTIVE_CONTEXT1_TEST_TRUST_ROOT is None
    assert not authorization._ACTIVE_CONTEXT1_TRUSTED_ISSUER_IDS
    assert not authorization._ACTIVE_CONTEXT1_TRUSTED_RUNNER_IDS
    assert not authorization._ACTIVE_CONTEXT1_ACCEPTED_SIGNATURE_SCHEMES
    assert {
        "issuer_identity",
        "runner_identity",
        "exact_pytest_collection_inventory",
        "exact_terminal_outcome_inventory",
        "runner_environment_manifest",
        "sandbox_policy_manifest",
        "sandbox_observation_manifest",
    } <= authorization._ACTIVE_CONTEXT1_REQUIRED_SIGNED_TEST_BINDINGS

    with pytest.raises(
        authorization.AuthorizationError, match="no governed independently"
    ):
        authorization._active_require_test_enforcement_authority()


def test_repository_constants_cannot_be_promoted_into_a_pseudo_trust_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        authorization, "_ACTIVE_CONTEXT1_TEST_AUTHORITY_STATUS", "trusted"
    )
    monkeypatch.setattr(
        authorization, "_ACTIVE_CONTEXT1_TEST_TRUST_ROOT", {"key": "local"}
    )
    monkeypatch.setattr(
        authorization,
        "_ACTIVE_CONTEXT1_TRUSTED_ISSUER_IDS",
        frozenset({"repository-writer"}),
    )
    monkeypatch.setattr(
        authorization,
        "_ACTIVE_CONTEXT1_TRUSTED_RUNNER_IDS",
        frozenset({"repository-writer"}),
    )
    monkeypatch.setattr(
        authorization,
        "_ACTIVE_CONTEXT1_ACCEPTED_SIGNATURE_SCHEMES",
        frozenset({"self-signature"}),
    )

    with pytest.raises(
        authorization.AuthorizationError, match="verifier is not implemented"
    ):
        authorization._active_require_test_enforcement_authority()


@pytest.mark.parametrize(
    "mutation",
    ("replace_with_fake", "add_fake", "duplicate", "reorder"),
)
def test_rehashed_nonexact_nodeid_inventory_is_rejected(mutation: str) -> None:
    document = _valid_active_test_evidence(Path("/project"), [])
    collection = document["collection_inventory"]
    outcomes = document["outcome_inventory"]
    if mutation == "replace_with_fake":
        real = collection[0]
        path = next(
            path
            for path in authorization.ACTIVE_FIXED_TEST_PATHS
            if real.startswith(path + "::")
        )
        collection[0] = f"{path}::test_repository_writer_invented_node"
        outcomes[0]["nodeid"] = collection[0]
        paired = sorted(zip(collection, outcomes, strict=True), key=lambda row: row[0])
        document["collection_inventory"] = [row[0] for row in paired]
        document["outcome_inventory"] = [row[1] for row in paired]
    elif mutation == "add_fake":
        path = authorization.ACTIVE_FIXED_TEST_PATHS[0]
        nodeid = f"{path}::test_repository_writer_invented_node"
        collection.append(nodeid)
        outcomes.append({"nodeid": nodeid, "outcome": "passed"})
        document["collection_inventory"].sort()
        document["outcome_inventory"].sort(key=lambda row: row["nodeid"])
        document["outcome_counts"]["collected"] += 1
        document["outcome_counts"]["passed"] += 1
    elif mutation == "duplicate":
        collection.append(collection[-1])
        outcomes.append(dict(outcomes[-1]))
        document["outcome_counts"]["collected"] += 1
        document["outcome_counts"]["passed"] += 1
    else:
        collection[0], collection[1] = collection[1], collection[0]
        outcomes[0], outcomes[1] = outcomes[1], outcomes[0]
    _rehash_test_inventory(document)

    with pytest.raises(authorization.AuthorizationError):
        authorization._active_validate_test_inventory(document)


@pytest.mark.parametrize(
    "mutation",
    (
        "skip",
        "missing_bwrap",
        "gpu_available",
        "raw_available",
        "outer_available",
        "completion_gpu_access",
        "completion_target_access",
        "inventory_hash",
    ),
)
def test_external_test_enforcement_receipt_rejects_unsafe_or_inexact_evidence(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Structural-only exercise behind the terminal production authority gate.
    monkeypatch.setattr(
        authorization, "_active_require_test_enforcement_authority", lambda: None
    )
    implementation = [{"path": "scripts/example.py"}]
    document = _valid_test_enforcement_document(tmp_path, implementation)
    if mutation == "skip":
        document["outcome_inventory"][0]["outcome"] = "skipped"
        document["outcome_counts"]["passed"] -= 1
        document["outcome_counts"]["skipped"] = 1
        _rehash_test_inventory(document)
        document["sandbox_completion"]["inventory_sha256"] = document[
            "inventory_sha256"
        ]
    elif mutation == "missing_bwrap":
        missing = next(iter(authorization._ACTIVE_CRITICAL_BWRAP_TEST_NODEIDS))
        index = document["collection_inventory"].index(missing)
        document["collection_inventory"].pop(index)
        document["outcome_inventory"].pop(index)
        document["outcome_counts"]["collected"] -= 1
        document["outcome_counts"]["passed"] -= 1
        _rehash_test_inventory(document)
        document["sandbox_completion"]["inventory_sha256"] = document[
            "inventory_sha256"
        ]
    elif mutation == "gpu_available":
        document["sandbox_capability"]["gpu_device_nodes_available"] = True
    elif mutation == "raw_available":
        document["sandbox_capability"]["raw_dataset_roots_available"] = True
    elif mutation == "outer_available":
        document["sandbox_capability"]["outer_reference_roots_available"] = True
    elif mutation == "completion_gpu_access":
        document["sandbox_completion"]["gpu_accessed"] = True
    elif mutation == "completion_target_access":
        document["sandbox_completion"][
            "target_or_outer_reference_accessed"
        ] = True
    else:
        document["inventory_sha256"] = "0" * 64
    document.pop("content_sha256")
    document["content_sha256"] = authorization.semantic_sha256(document)
    _write_frozen_json(
        tmp_path, authorization.ACTIVE_TEST_ENFORCEMENT_RECEIPT, document
    )

    with pytest.raises(authorization.AuthorizationError):
        authorization._active_validate_test_enforcement_receipt(
            tmp_path, implementation
        )


def test_create_test_receipt_fails_before_any_issuance_work_without_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        authorization,
        "_active_validate_create_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("issuance work must never start without authority")
        ),
    )

    with pytest.raises(
        authorization.AuthorizationError, match="no governed independently"
    ):
        authorization.create_test_receipt(tmp_path)
    assert not os.path.lexists(tmp_path / authorization.ACTIVE_TEST_RECEIPT)


@pytest.mark.parametrize(
    ("entrypoint", "relative"),
    (
        (authorization.create_source_snapshot, authorization.ACTIVE_SOURCE_SNAPSHOT),
        (
            authorization.create_pretrain_authorization,
            authorization.ACTIVE_PRETRAIN_AUTHORIZATION,
        ),
    ),
)
def test_downstream_create_once_stages_cannot_start_without_test_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: Any,
    relative: Path,
) -> None:
    monkeypatch.setattr(
        authorization,
        "_active_validate_create_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("downstream issuance work must not start")
        ),
    )

    with pytest.raises(
        authorization.AuthorizationError, match="no governed independently"
    ):
        entrypoint(tmp_path)
    assert not os.path.lexists(tmp_path / relative)


def test_target_document_chain_blocks_before_opening_repository_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        authorization,
        "_active_validate_execution_closure_target_chain",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("target receipt loading must not start")
        ),
    )

    with pytest.raises(
        authorization.AuthorizationError, match="no governed independently"
    ):
        authorization._active_target_documents(tmp_path, {}, {})


def _content_document(**extra: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "classification": "cpu_fixture",
        "campaign_id": (
            "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        ),
        **extra,
    }
    value["content_sha256"] = authorization.semantic_sha256(value)
    return value


def _write_frozen_json(root: Path, relative: Path, value: Mapping[str, Any]) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    return path


def test_active_document_decode_uses_the_single_pinned_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = Path("governance.json")
    document = _content_document()
    _write_frozen_json(tmp_path, relative, document)

    def forbidden_read_bytes(_self: Path) -> bytes:
        raise AssertionError("path-level reopen is forbidden")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    observed, binding = authorization._active_load_document(tmp_path, relative)

    assert observed == document
    assert binding["sha256"] == hashlib.sha256(
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    assert binding["mode"] == "0444"


def test_exact_frozen_contract_nonpretty_encoding_is_accepted() -> None:
    raw = (SOURCE_ROOT / authorization.CONTRACT).read_bytes()
    document = json.loads(raw)
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
    assert raw != pretty

    observed, binding = authorization._active_load_document(
        SOURCE_ROOT, authorization.CONTRACT
    )

    assert observed["schema_version"] == 1
    assert observed["classification"] == (
        "adaptive_retrospective_historical_cohort_engineering_not_confirmatory"
    )
    assert observed["content_sha256"] == authorization.CONTRACT_CONTENT_SHA256
    assert binding["sha256"] == authorization.CONTRACT_FILE_SHA256
    assert binding["bytes"] == authorization.CONTRACT_FILE_BYTES == 16179
    assert binding["mode"] == "0444"
    assert binding["nlink"] == 1


def test_frozen_contract_bytes_at_different_relative_path_are_rejected(
    tmp_path: Path,
) -> None:
    relative = Path("copied_contract.json")
    path = tmp_path / relative
    path.write_bytes((SOURCE_ROOT / authorization.CONTRACT).read_bytes())
    path.chmod(0o444)

    with pytest.raises(
        authorization.AuthorizationError, match="encoding is noncanonical"
    ):
        authorization._active_load_document(tmp_path, relative)


def test_semantically_equal_frozen_contract_byte_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / authorization.CONTRACT
    path.parent.mkdir(parents=True)
    raw = (SOURCE_ROOT / authorization.CONTRACT).read_bytes()
    mutated = raw.replace(b"{\n", b"{ \n", 1)
    assert json.loads(mutated) == json.loads(raw)
    assert len(mutated) == len(raw) + 1
    path.write_bytes(mutated)
    path.chmod(0o444)

    with pytest.raises(
        authorization.AuthorizationError, match="encoding is noncanonical"
    ):
        authorization._active_load_document(tmp_path, authorization.CONTRACT)


def test_unrelated_nonpretty_governance_encoding_remains_rejected(
    tmp_path: Path,
) -> None:
    relative = Path("unrelated.json")
    document = _content_document(values=[1, 2, 3])
    path = tmp_path / relative
    path.write_bytes(authorization.canonical_bytes(document) + b"\n")
    path.chmod(0o444)

    with pytest.raises(
        authorization.AuthorizationError, match="encoding is noncanonical"
    ):
        authorization._active_load_document(tmp_path, relative)


def test_admitted_revalidator_rejects_binding_owned_phase_and_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden_loader(_root: Path) -> Any:
        nonlocal called
        called = True
        raise AssertionError("untrusted binding reached wrapper validator")

    monkeypatch.setattr(authorization, "_load_gpu_admitted_validator", forbidden_loader)
    with pytest.raises(
        authorization.AuthorizationError, match="caller expectation"
    ):
        authorization._active_revalidate_admitted(
            tmp_path,
            {"phase": "discovery", "context": {"outer_fold": 4}},
            tmp_path / authorization.ACTIVE_PRETRAIN_AUTHORIZATION,
            expected_phase="efficiency_benchmark",
            expected_context=authorization._CONTEXT1_FULL_BENCHMARK_CONTEXT,
        )
    assert called is False


def test_admitted_revalidator_passes_independent_phase_and_exact_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_path = _write_frozen_json(
        tmp_path, authorization.ACTIVE_PRETRAIN_AUTHORIZATION, _content_document()
    )
    admitted = {"phase": "discovery", "context": {"outer_fold": 3, "seed": 7}}
    observed: dict[str, Any] = {}

    class FakeWrapper:
        @staticmethod
        def revalidate_consumed_admitted_child_binding(
            binding: Mapping[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            observed.update(kwargs)
            return dict(binding)

    monkeypatch.setattr(
        authorization, "_load_gpu_admitted_validator", lambda _root: FakeWrapper
    )
    result = authorization._active_revalidate_admitted(
        tmp_path,
        admitted,
        auth_path,
        expected_phase="discovery",
        expected_context={"outer_fold": 3, "seed": 7},
    )

    assert result == admitted
    assert observed["expected_phase"] == "discovery"
    assert observed["expected_authorization_sha256"] == hashlib.sha256(
        auth_path.read_bytes()
    ).hexdigest()


def _target_capability_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any]]:
    migration_relative = Path("governance/migration.json")
    migration_path = tmp_path / migration_relative
    migration_path.parent.mkdir(parents=True)
    migration_path.write_bytes(b"{}\n")
    migration_path.chmod(0o444)
    migration = authorization._active_binding(tmp_path, migration_relative)
    monkeypatch.setattr(
        authorization,
        "_TARGET_GOVERNANCE_ROLE_PATHS",
        {"gpu_state_migration_receipt": migration_relative},
    )
    monkeypatch.setattr(
        authorization, "ACTIVE_MIGRATION_RECEIPT", migration_relative
    )
    monkeypatch.setattr(
        authorization,
        "_TARGET_GOVERNANCE_ROLES_BY_PHASE",
        {**authorization._TARGET_GOVERNANCE_ROLES_BY_PHASE, "discovery": {
            "gpu_state_migration_receipt"
        }},
    )
    entry = tmp_path / "scripts/run_hfr_v3r1_discovery_campaign.py"
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"# synthetic exact entry\n")
    entry.chmod(0o444)
    canonical_paths = {
        "output": (
            tmp_path
            / "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
            "discovery_v8r4/shards/outer_3"
        ),
        "lifecycle": (
            tmp_path
            / authorization._TARGET_LIFECYCLE_ROOT
            / "discovery/"
            "run_hfr_v3r1_discovery_campaign/outer_3"
        ),
        "usage": tmp_path / authorization.ACTIVE_STATE_ROOT / "usage",
        "execution": tmp_path / authorization.ACTIVE_STATE_ROOT / "execution",
        "admission": tmp_path / authorization.ACTIVE_STATE_ROOT / "admission",
    }
    writable = {
        role: {
            "path": str(canonical_paths[role]),
            "st_dev": 1,
            "st_ino": number + 1,
            "mode": "0700",
        }
        for number, role in enumerate(
            ("output", "lifecycle", "usage", "execution", "admission")
        )
    }
    exact_entries = {
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
    prelaunch_directories = {
        "root": {
            "exact_entries": exact_entries["root"],
            "mode": "0700",
            "path": authorization.ACTIVE_STATE_ROOT.as_posix(),
            "st_dev": 1,
            "st_ino": 100,
        },
        **{
            role: {
                "exact_entries": exact_entries[role],
                "mode": writable[role]["mode"],
                "path": (authorization.ACTIVE_STATE_ROOT / role).as_posix(),
                "st_dev": writable[role]["st_dev"],
                "st_ino": writable[role]["st_ino"],
            }
            for role in ("admission", "execution", "usage")
        },
    }
    state_root = str(tmp_path / authorization.ACTIVE_STATE_ROOT)
    governance_binding = {
        key: (str(migration_path) if key == "path" else migration[key])
        for key in authorization._TARGET_CAPABILITY_FILE_KEYS
    }
    mount_specification = [
        {
            "destination": state_root,
            "kind": "ro_bind_fd",
            "source": {**prelaunch_directories["root"], "path": state_root},
        },
        *(
            {
                "destination": writable[role]["path"],
                "kind": "rw_bind_fd",
                "source": {
                    **prelaunch_directories[role],
                    "path": writable[role]["path"],
                },
            }
            for role in ("admission", "execution", "usage")
        ),
    ] + [
        {
            "kind": "ro_bind_fd",
            "destination": writable["lifecycle"]["path"],
            "source": dict(writable["lifecycle"]),
        },
        {
            "kind": "rw_bind_fd",
            "destination": writable["output"]["path"],
            "source": dict(writable["output"]),
        },
        *(
            {
                "kind": "ro_bind_fd",
                "destination": str(path),
                "source": {
                    "path": str(path),
                    "st_dev": path.stat().st_dev,
                    "st_ino": path.stat().st_ino,
                    "mode": f"{path.stat().st_mode & 0o7777:04o}",
                },
            }
            for path in (Path("/usr"), Path("/sys"))
        ),
        {
            "kind": "ro_bind_fd",
            "destination": str(migration_path),
            "source": dict(governance_binding),
        },
    ]
    boundary = {
        key: key in authorization._TARGET_REQUIRED_TRUE_BOUNDARIES
        for key in authorization._TARGET_SECURITY_BOUNDARY_KEYS
    }
    boundary["production_execution_authorized"] = False
    boundary["synthetic_validation_only"] = True
    document: dict[str, Any] = {
        "schema_version": 1,
        "classification": authorization._TARGET_CAPABILITY_CLASSIFICATION,
        "campaign_id": (
            "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
        ),
        "campaign_revision": authorization.ACTIVE_SCIENTIFIC_REVISION,
        "infrastructure_revision": authorization.ACTIVE_INFRASTRUCTURE_REVISION,
        "phase": "discovery",
        "outer_fold": 3,
        "bubblewrap": {},
        "launcher": {},
        "interpreter": {},
        "sealed_pack_root": {},
        "sealed_pack_index": {},
        "governance_files": {
            "gpu_state_migration_receipt": governance_binding
        },
        "writable_roots": writable,
        "prelaunch_gpu_state": {
            "migration_receipt": {
                **migration,
                "content_sha256": "0" * 64,
            },
            "directories": prelaunch_directories,
            "files": {},
            "usage_state": {},
            "execution_state": {},
        },
        "denied_canaries": {
            role: str(tmp_path / relative)
            for role, relative in (
                authorization._TARGET_REQUIRED_SUPERSEDED_CANARIES.items()
            )
        },
        "mount_specification": mount_specification,
        "mount_specification_sha256": authorization.semantic_sha256(
            mount_specification
        ),
        "environment": {},
        "environment_sha256": authorization.semantic_sha256({}),
        "command": ["python", str(entry)],
        "command_sha256": authorization.semantic_sha256(
            ["python", str(entry)]
        ),
        "security_boundary": boundary,
    }
    expected_state_mounts = mount_specification[:4]
    bind_mounts = authorization._active_validate_capability_bind_mount_cover(
        tmp_path,
        document,
        mount_specification,
        expected_state_mounts=expected_state_mounts,
        writable=writable,
    )
    mount_specification = (
        authorization._active_canonical_capability_mount_specification(
            tmp_path, document, bind_mounts, mount_specification
        )
    )
    document["mount_specification"] = mount_specification
    document["mount_specification_sha256"] = authorization.semantic_sha256(
        mount_specification
    )
    receipt = (
        canonical_paths["lifecycle"]
        / "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json"
    )

    def publish() -> None:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        document.pop("content_sha256", None)
        document["content_sha256"] = authorization.semantic_sha256(document)
        receipt.write_bytes(authorization.canonical_bytes(document) + b"\n")
        receipt.chmod(0o444)

    publish()
    document["_publish"] = publish
    return receipt, document


def test_target_capability_exact_boundary_and_mount_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, fixture = _target_capability_fixture(tmp_path, monkeypatch)
    publish = fixture.pop("_publish")
    capability, binding, governance = authorization._active_validate_target_capability(
        tmp_path,
        receipt,
        expected_phase="discovery",
        expected_outer_fold=3,
    )
    assert capability["security_boundary"]["production_execution_authorized"] is False
    assert binding["mode"] == "0444"
    assert set(governance) == {"gpu_state_migration_receipt"}
    assert callable(publish)


@pytest.mark.parametrize(
    "mutation",
    (
        "boundary_addition",
        "negative_boundary_true",
        "governance_addition",
        "invalid_outer_fold",
        "missing_writable_role",
    ),
)
def test_target_capability_schema_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    receipt, fixture = _target_capability_fixture(tmp_path, monkeypatch)
    publish = fixture.pop("_publish")
    receipt.chmod(0o644)
    if mutation == "boundary_addition":
        fixture["security_boundary"]["unexpected"] = False
    elif mutation == "negative_boundary_true":
        fixture["security_boundary"]["commercial_claim_authorized"] = True
    elif mutation == "governance_addition":
        fixture["governance_files"]["unexpected"] = dict(
            fixture["governance_files"]["gpu_state_migration_receipt"]
        )
    elif mutation == "invalid_outer_fold":
        fixture["outer_fold"] = 2
    else:
        del fixture["writable_roots"]["usage"]
    publish()
    with pytest.raises(authorization.AuthorizationError):
        authorization._active_validate_target_capability(
            tmp_path,
            receipt,
            expected_phase="discovery",
            expected_outer_fold=(2 if mutation == "invalid_outer_fold" else 3),
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_parent",
        "duplicate_parent",
        "parent_kind",
        "parent_source_path",
        "parent_source_identity",
        "parent_after_child",
        "child_order",
        "child_kind",
        "child_source_path",
        "child_source_identity",
        "boundary_false",
        "missing_contract1_canary_role",
        "wrong_contract1_canary_path",
        "extra_denied_canary_role",
        "extra_rw_bind",
        "extra_ro_bind",
        "state_descendant_destination",
        "state_descendant_source",
        "state_sibling_destination",
        "state_sibling_source",
        "duplicate_bind_destination",
    ),
)
def test_target_capability_rootbind_mount_abi_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    receipt, fixture = _target_capability_fixture(tmp_path, monkeypatch)
    publish = fixture.pop("_publish")
    receipt.chmod(0o644)
    mounts = fixture["mount_specification"]
    state_root = str(tmp_path / authorization.ACTIVE_STATE_ROOT)
    state_paths = [
        state_root,
        *(str(tmp_path / authorization.ACTIVE_STATE_ROOT / role) for role in (
            "admission",
            "execution",
            "usage",
        )),
    ]
    state_positions = [
        next(
            index
            for index, row in enumerate(mounts)
            if row.get("destination") == destination
        )
        for destination in state_paths
    ]
    parent_position, admission_position, execution_position, _usage_position = (
        state_positions
    )
    parent_mount = mounts[parent_position]
    admission_mount = mounts[admission_position]
    if mutation == "missing_parent":
        mounts.pop(parent_position)
    elif mutation == "duplicate_parent":
        mounts.insert(parent_position + 1, json.loads(json.dumps(parent_mount)))
    elif mutation == "parent_kind":
        parent_mount["kind"] = "rw_bind_fd"
    elif mutation == "parent_source_path":
        parent_mount["source"]["path"] += "/alias"
    elif mutation == "parent_source_identity":
        parent_mount["source"]["st_ino"] += 1
    elif mutation == "parent_after_child":
        mounts[parent_position], mounts[admission_position] = (
            mounts[admission_position],
            mounts[parent_position],
        )
    elif mutation == "child_order":
        mounts[admission_position], mounts[execution_position] = (
            mounts[execution_position],
            mounts[admission_position],
        )
    elif mutation == "child_kind":
        admission_mount["kind"] = "ro_bind_fd"
    elif mutation == "child_source_path":
        admission_mount["source"]["path"] += "/alias"
    elif mutation == "child_source_identity":
        admission_mount["source"]["st_dev"] += 1
    elif mutation == "boundary_false":
        fixture["security_boundary"][
            "gpu_state_parent_identity_readonly_bind"
        ] = False
    elif mutation == "missing_contract1_canary_role":
        del fixture["denied_canaries"][
            "superseded_v8r4a_contract1_lifecycle_root"
        ]
    elif mutation == "wrong_contract1_canary_path":
        fixture["denied_canaries"][
            "superseded_v8r4a_contract1_output_root"
        ] += "/alias"
    elif mutation == "extra_denied_canary_role":
        fixture["denied_canaries"]["unexpected"] = str(tmp_path / "unexpected")
    elif mutation == "extra_rw_bind":
        mounts.append(
            {
                "kind": "rw_bind_fd",
                "destination": str(tmp_path / "arbitrary-fourth-mutable"),
                "source": dict(fixture["writable_roots"]["admission"]),
            }
        )
    elif mutation == "extra_ro_bind":
        mounts.append(
            json.loads(
                json.dumps(next(row for row in mounts if row.get("destination") == "/usr"))
            )
        )
        mounts[-1]["destination"] = str(tmp_path / "arbitrary-readonly")
    elif mutation == "state_descendant_destination":
        mounts.append(json.loads(json.dumps(admission_mount)))
        mounts[-1]["destination"] = parent_mount["destination"] + "/shadow"
    elif mutation == "state_descendant_source":
        mounts.append(json.loads(json.dumps(admission_mount)))
        mounts[-1]["destination"] = str(tmp_path / "shadow")
        mounts[-1]["source"]["path"] = parent_mount["destination"] + "/shadow"
    elif mutation == "state_sibling_destination":
        mounts.append(json.loads(json.dumps(admission_mount)))
        mounts[-1]["destination"] = parent_mount["destination"] + "_sibling"
    elif mutation == "state_sibling_source":
        mounts.append(json.loads(json.dumps(admission_mount)))
        mounts[-1]["destination"] = str(tmp_path / "other-shadow")
        mounts[-1]["source"]["path"] = parent_mount["destination"] + "_sibling"
    else:
        mounts.append(json.loads(json.dumps(admission_mount)))
    fixture["mount_specification_sha256"] = authorization.semantic_sha256(mounts)
    publish()

    with pytest.raises(authorization.AuthorizationError):
        authorization._active_validate_target_capability(
            tmp_path,
            receipt,
            expected_phase="discovery",
            expected_outer_fold=3,
        )


@pytest.mark.parametrize("projection", ("host", "target"))
def test_arbitrary_fourth_mutable_mount_is_rejected_by_host_and_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    projection: str,
) -> None:
    receipt, fixture = _target_capability_fixture(tmp_path, monkeypatch)
    publish = fixture.pop("_publish")
    receipt.chmod(0o644)
    fixture["mount_specification"].append(
        {
            "kind": "rw_bind_fd",
            "destination": str(tmp_path / "arbitrary-fourth-mutable"),
            "source": dict(fixture["writable_roots"]["admission"]),
        }
    )
    fixture["mount_specification_sha256"] = authorization.semantic_sha256(
        fixture["mount_specification"]
    )
    publish()
    authorization.verify_content_hash(
        json.loads(receipt.read_text(encoding="utf-8")), path=receipt
    )

    with pytest.raises(authorization.AuthorizationError, match="bind mount cover"):
        if projection == "host":
            authorization._active_validate_target_capability(
                tmp_path,
                receipt,
                expected_phase="discovery",
                expected_outer_fold=3,
            )
        else:
            authorization.validate_pretrain_target_scoped(
                tmp_path,
                receipt,
                expected_phase="discovery",
                expected_outer_fold=3,
            )


@pytest.mark.parametrize("mutation", ("arbitrary", "superseded_root", "symlink"))
def test_common_failure_diagnostic_has_one_canonical_path_without_opening_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    receipt, fixture = _target_capability_fixture(tmp_path, monkeypatch)
    publish = fixture.pop("_publish")
    canonical = Path("governance/canonical_failure.json")
    denied_relative = (
        authorization._ROOTBIND1_LIFECYCLE_ROOT / "untrusted-evidence.json"
    )
    denied = tmp_path / denied_relative
    denied.parent.mkdir(parents=True, exist_ok=True)
    denied.write_bytes(b"{}\n")
    denied.chmod(0o444)
    arbitrary = tmp_path / "governance/arbitrary.json"
    arbitrary.parent.mkdir(parents=True, exist_ok=True)
    arbitrary.write_bytes(b"{}\n")
    arbitrary.chmod(0o444)
    if mutation == "arbitrary":
        selected = arbitrary
    elif mutation == "superseded_root":
        selected = denied
    else:
        selected = tmp_path / canonical
        selected.symlink_to(denied)
    source_path = denied if mutation == "symlink" else selected
    source_binding = authorization._active_binding(
        tmp_path, source_path.relative_to(tmp_path)
    )
    capability_binding = {
        key: (
            str(selected)
            if key == "path"
            else source_binding[key]
        )
        for key in authorization._TARGET_CAPABILITY_FILE_KEYS
    }
    fixture["governance_files"]["failure_diagnostic"] = capability_binding
    fixture["mount_specification"].append(
        {
            "kind": "ro_bind_fd",
            "destination": str(selected),
            "source": dict(capability_binding),
        }
    )
    fixture["mount_specification_sha256"] = authorization.semantic_sha256(
        fixture["mount_specification"]
    )
    monkeypatch.setattr(
        authorization,
        "_TARGET_GOVERNANCE_ROLE_PATHS",
        {
            **authorization._TARGET_GOVERNANCE_ROLE_PATHS,
            "failure_diagnostic": canonical,
        },
    )
    monkeypatch.setattr(
        authorization,
        "_TARGET_GOVERNANCE_ROLES_BY_PHASE",
        {
            **authorization._TARGET_GOVERNANCE_ROLES_BY_PHASE,
            "discovery": {
                "gpu_state_migration_receipt",
                "failure_diagnostic",
            },
        },
    )
    receipt.chmod(0o644)
    publish()
    opened: list[Path] = []
    real_open = authorization.os.open

    def open_spy(path: Any, *args: Any, **kwargs: Any) -> int:
        opened.append(Path(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(authorization.os, "open", open_spy)
    with pytest.raises(authorization.AuthorizationError):
        authorization._active_validate_target_capability(
            tmp_path,
            receipt,
            expected_phase="discovery",
            expected_outer_fold=3,
        )
    assert selected not in opened
    assert denied not in opened


def _patch_target_chain(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    revalidation_results: list[Mapping[str, Any]] | None = None,
) -> None:
    capability = {
        "phase": "discovery",
        "outer_fold": 3,
        "prelaunch_gpu_state": {"trusted": True},
    }
    calls = 0

    def validate_capability(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        events.append(f"capability{calls}")
        return capability, {"sha256": "c" * 64}, {"active_authorization": {}}

    def target_documents(*_args: Any, **_kwargs: Any) -> Any:
        events.append("documents")
        return (
            {
                "runtime_ledger_prefixes": {},
                "promotion_reuse_scope": {},
                "admitted_child_scope": {},
                "gpu_budget_protocol": {},
                "canonical_gpu_state_paths": {},
            },
            {"sha256": "a" * 64},
            {},
            {"sha256": "s" * 64},
            {"sha256": "t" * 64},
            {"sha256": "d" * 64},
        )

    def replay(*_args: Any, **kwargs: Any) -> Any:
        events.append("lock_free_replay")
        assert kwargs["trusted_prelaunch_state"] == {"trusted": True}
        assert kwargs.get("lock_free", False) is (
            not kwargs["require_closed"]
        )
        return object(), {}

    def verify_prefixes(*_args: Any, **_kwargs: Any) -> None:
        events.append("prefixes")

    results = list(revalidation_results or [])

    def revalidate(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("revalidate")
        assert kwargs["expected_phase"] == "discovery"
        assert kwargs["expected_context"] == {"outer_fold": 3, "seed": 7}
        return dict(results.pop(0) if results else _args[1])

    monkeypatch.setattr(
        authorization, "_active_validate_target_capability", validate_capability
    )
    monkeypatch.setattr(authorization, "_active_target_documents", target_documents)
    monkeypatch.setattr(authorization, "_active_state", replay)
    monkeypatch.setattr(authorization, "_active_verify_prefixes", verify_prefixes)
    monkeypatch.setattr(authorization, "_active_revalidate_admitted", revalidate)
    monkeypatch.setattr(
        authorization,
        "_active_target_result",
        lambda **_kwargs: {"valid": True, "target_scoped": True},
    )


def test_target_scoped_validation_is_capability_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_target_chain(monkeypatch, events)

    result = authorization.validate_pretrain_target_scoped(
        tmp_path,
        tmp_path / "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json",
        expected_phase="discovery",
        expected_outer_fold=3,
    )

    assert result == {"valid": True, "target_scoped": True}
    assert events == [
        "capability1",
        "documents",
        "lock_free_replay",
        "prefixes",
        "capability2",
    ]


def test_target_admitted_order_is_revalidate_replay_revalidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    admitted = {"phase": "discovery", "context": {"outer_fold": 3, "seed": 7}}
    _patch_target_chain(monkeypatch, events)

    result = authorization.validate_pretrain_target_scoped_admitted_child(
        tmp_path,
        tmp_path / "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json",
        admitted,
        expected_phase="discovery",
        expected_context={"outer_fold": 3, "seed": 7},
        expected_outer_fold=3,
    )

    assert result["valid"] is True
    assert events == [
        "capability1",
        "documents",
        "revalidate",
        "lock_free_replay",
        "revalidate",
        "prefixes",
        "capability2",
    ]


def test_target_admitted_refuses_capability_change_across_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    admitted = {"phase": "discovery", "context": {"outer_fold": 3, "seed": 7}}
    _patch_target_chain(
        monkeypatch,
        events,
        revalidation_results=[admitted, {**admitted, "nonce": "changed"}],
    )

    with pytest.raises(authorization.AuthorizationError, match="changed across"):
        authorization.validate_pretrain_target_scoped_admitted_child(
            tmp_path,
            tmp_path / "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json",
            admitted,
            expected_phase="discovery",
            expected_context={"outer_fold": 3, "seed": 7},
            expected_outer_fold=3,
        )


def test_active_surface_registers_complete_v8r4a_dependency_closure() -> None:
    expected = {
        "src/snn_rr/__init__.py",
        "src/snn_rr/harmonic_factor_router_v3.py",
        "src/snn_rr/svd_episode_models.py",
        "src/snn_rr/models.py",
        "configs/harmonic_factor_router_v3.yaml",
        "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json",
        "pyproject.toml",
        "tests/test_validate_hfr_v3r1_authorization_v8r4a.py",
    }
    assert expected <= set(authorization.ACTIVE_IMPLEMENTATION_PATHS)
    assert (
        "tests/test_validate_hfr_v3r1_authorization_v8r4a.py"
        in authorization.ACTIVE_FIXED_TEST_PATHS
    )


def test_v8r4_scanner_excludes_only_the_validated_dependency_bound_v8r5_proposal() -> None:
    relative = "src/snn_rr/axis_risk_router_snn_v8r5.py"
    raw = (SOURCE_ROOT / relative).read_bytes()

    authorization._active_validate_independent_successor_proposal(relative, raw)

    assert relative in authorization._ACTIVE_INDEPENDENT_SUCCESSOR_PROPOSAL_PATHS
    assert relative not in authorization.ACTIVE_IMPLEMENTATION_PATHS


@pytest.mark.parametrize(
    "mutation",
    (
        "claim_true",
        "protected_import",
        "extra_internal_dependency",
        "missing_dependency_hash",
        "missing_boundary",
        "wrong_path",
    ),
)
def test_dependency_bound_v8r5_scanner_exclusion_fails_closed(
    mutation: str,
) -> None:
    relative = "src/snn_rr/axis_risk_router_snn_v8r5.py"
    raw = (SOURCE_ROOT / relative).read_bytes()
    if mutation == "claim_true":
        raw = raw.replace(
            b'"training_authorized": False',
            b'"training_authorized": True',
        )
    elif mutation == "protected_import":
        raw += b"\nfrom .harmonic_factor_router_v3 import HarmonicFactorSNN\n"
    elif mutation == "extra_internal_dependency":
        raw += b"\nfrom .models import StructuredTriRadarRRSNN\n"
    elif mutation == "missing_dependency_hash":
        raw = raw.replace(
            b"_FEATURE_LAYOUT_SOURCE_SHA256",
            b"_FEATURE_LAYOUT_SOURCE_DIGEST_MISSING",
        )
    elif mutation == "missing_boundary":
        raw = raw.replace(
            b"does not authorize protected training",
            b"may authorize protected training",
        )
    else:
        relative = "src/snn_rr/unregistered_v8r5.py"

    with pytest.raises(authorization.AuthorizationError, match="successor"):
        authorization._active_validate_independent_successor_proposal(relative, raw)


def test_open_lifecycle_recovery_authority_has_exact_cross_bindings() -> None:
    authorities = authorization._active_authorized_modifications(SOURCE_ROOT)
    assert "V8R4A open-lifecycle recovery" in {label for _document, label in authorities}
    addendums = authorization._active_addendum_bindings(SOURCE_ROOT)
    assert addendums["open_lifecycle_recovery_correction_authorization"][
        "sha256"
    ] == "92b7e3e4b911dbf7450e3447b84f5a5762aee1212348c151cd57f20f10f5e1f6"
    assert addendums["open_lifecycle_recovery_failure_diagnostic"][
        "sha256"
    ] == "a9cd41355ff98502153d61ad83d7f01da0d3c52462acdd24ade7bef73cf80b5e"
    assert (
        authorization._TARGET_GOVERNANCE_ROLE_PATHS[
            "open_lifecycle_recovery_correction_authorization"
        ]
        == authorization.ACTIVE_OPEN_LIFECYCLE_RECOVERY_CORRECTION
    )


def test_execution_closure_and_migration_succession_are_exact_registered() -> None:
    authorities = authorization._active_authorized_modifications(SOURCE_ROOT)
    labels = {label for _document, label in authorities}
    assert "V8R4A execution closure" in labels
    assert "V8R4A migration source succession" in labels
    addendums = authorization._active_addendum_bindings(SOURCE_ROOT)
    assert addendums["execution_closure_correction_authorization"]["sha256"] == (
        "11e5e7d00d3d837e0eb542b3482499cd21bf90b38883d0f4a0f86b68ba16d754"
    )
    assert addendums["execution_closure_failure_diagnostic"]["sha256"] == (
        "0ca492c98f94e73e21873c41287c24d3d135466c4ca4d085388f9c39e5d9560e"
    )
    assert addendums[
        "migration_source_succession_correction_authorization"
    ]["sha256"] == (
        "4a3673a406f49287b5abe16cc9ddde5d90d55f3a18d82a346ed390b55ccd91d9"
    )
    assert addendums["migration_source_succession_failure_diagnostic"][
        "sha256"
    ] == "265eb0cb62f6412d26bc7491ad959c8b3d6e49ffc47241573ed0fecf5111ac1e"
    assert {
        "execution_closure_correction_authorization",
        "execution_closure_failure_diagnostic",
        "migration_source_succession_correction_authorization",
        "migration_source_succession_failure_diagnostic",
    } <= authorization._ACTIVE_ADDENDUM_ROLES


def test_fd_closure_successor_chain_is_exact_registered() -> None:
    authority_document, diagnostic_document = (
        authorization._active_validate_fd_closure_correction(SOURCE_ROOT)
    )
    assert authority_document["claim_boundary"][
        "gpu_execution_authorized_by_this_document"
    ] is False
    assert diagnostic_document["failed_attempt"]["gpu_child_launched"] is False
    authorities = authorization._active_authorized_modifications(SOURCE_ROOT)
    assert "V8R4A outer guard FD closure" in {
        label for _document, label in authorities
    }
    addendums = authorization._active_addendum_bindings(SOURCE_ROOT)
    assert addendums["fd_closure_correction_authorization"]["sha256"] == (
        "1ad3bdaa0b78937c5b6ce98bc2e4e02d31e41951baf57dde6d68aa8029b25110"
    )
    assert addendums["fd_closure_failure_diagnostic"]["sha256"] == (
        "75766bbbcc2e1bdc6cdcc61ddee559a2ffa647586f7cbbb87fea0696034d8fbd"
    )
    assert authorization._FD_CLOSURE_SUCCESSOR_CHAIN_NAMES == {
        "new_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_FD1.json",
        "new_source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_FD1.json",
        "new_pretrain_authorization": "PRETRAIN_AUTHORIZATION_V8R4A_FD1.json",
    }
    assert {
        "fd_closure_correction_authorization",
        "fd_closure_failure_diagnostic",
    } <= authorization._ACTIVE_ADDENDUM_ROLES


def test_canary_boundary_successor_chain_and_fd1_history_are_exact() -> None:
    authority_document, diagnostic_document = (
        authorization._active_validate_canary_boundary_correction(SOURCE_ROOT)
    )
    assert authority_document["claim_boundary"][
        "gpu_execution_authorized_by_this_document"
    ] is False
    assert diagnostic_document["failed_attempt"]["gpu_child_launched"] is False
    authorities = authorization._active_authorized_modifications(SOURCE_ROOT)
    labels = [label for _document, label in authorities]
    assert labels[-4:] == [
        "V8R4A denied-canary component boundary",
        "V8R4A frozen-contract encoding",
        "V8R4A GPU-state parent bind",
        "V8R4A admitted benchmark context",
    ]
    addendums = authorization._active_addendum_bindings(SOURCE_ROOT)
    assert addendums["canary_boundary_correction_authorization"]["sha256"] == (
        "8187bd7c306419114c25020cf2a89c5a740d766b12df5ddd2e6d29ca498a84c3"
    )
    assert addendums["canary_boundary_failure_diagnostic"]["sha256"] == (
        "7ea5aea6f661dc0e3157a6ee71e20b2e5e04da5a4dc81035e652e9902ecb6294"
    )
    assert authorization._CANARY_BOUNDARY_SUCCESSOR_CHAIN_NAMES == {
        "new_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CANARY1.json",
        "new_source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_CANARY1.json",
        "new_pretrain_authorization": (
            "PRETRAIN_AUTHORIZATION_V8R4A_CANARY1.json"
        ),
    }
    assert {
        "canary_boundary_correction_authorization",
        "canary_boundary_failure_diagnostic",
    } <= authorization._ACTIVE_ADDENDUM_ROLES
    assert authorization._TARGET_GOVERNANCE_ROLE_PATHS[
        "canary_boundary_correction_authorization"
    ] == authorization.ACTIVE_CANARY_BOUNDARY_CORRECTION
    assert authorization._TARGET_GOVERNANCE_ROLE_PATHS[
        "canary_boundary_failure_diagnostic"
    ] == authorization.ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC
    for field, binding in authorization._CANARY_BOUNDARY_PARENT_BINDINGS.items():
        if field == "parent_fd_closure_authority":
            continue
        assert authorization.ACTIVE_HISTORICAL_FILES[binding["path"]] == (
            binding["file_sha256"]
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "historical_parent",
        "modification_cover",
        "superseded_chain",
        "active_filename",
        "component_boundary",
        "content_hash",
    ),
)
def test_canary_boundary_target_projection_fails_closed(mutation: str) -> None:
    authority_document, _ = authorization._active_load_document(
        SOURCE_ROOT, authorization.ACTIVE_CANARY_BOUNDARY_CORRECTION
    )
    diagnostic_document, _ = authorization._active_load_document(
        SOURCE_ROOT, authorization.ACTIVE_CANARY_BOUNDARY_DIAGNOSTIC
    )
    changed_authority = json.loads(json.dumps(authority_document))
    changed_diagnostic = json.loads(json.dumps(diagnostic_document))
    if mutation == "historical_parent":
        changed_authority["authority_basis"]["parent_source_snapshot"][
            "file_sha256"
        ] = "0" * 64
    elif mutation == "modification_cover":
        changed_authority["authorized_modifications"].pop()
    elif mutation == "superseded_chain":
        changed_diagnostic["superseded_pretrain_chain"]["source_snapshot"][
            "file_sha256"
        ] = "0" * 64
    elif mutation == "active_filename":
        changed_authority["required_reauthorization"]["new_test_receipt"] = (
            "IMPLEMENTATION_TEST_RECEIPT_V8R4A_FD1.json"
        )
    elif mutation == "component_boundary":
        changed_authority["mandatory_invariants"][
            "path_distinct_sibling_prefixes_are_not_capabilities"
        ] = False
    else:
        changed_authority["content_sha256"] = "0" * 64
    with pytest.raises(
        authorization.AuthorizationError, match="CANARY-boundary"
    ):
        authorization._active_validate_canary_boundary_projection(
            changed_authority, changed_diagnostic
        )


def test_canary_target_projection_never_opens_historical_fd1_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = authorization._active_load_document
    forbidden = set(authorization._CANARY_BOUNDARY_PARENT_PATHS.values()) - {
        authorization.ACTIVE_FD_CLOSURE_CORRECTION
    }

    def guarded_loader(
        root: Path, relative: str | Path, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if Path(relative) in forbidden:
            raise AssertionError(f"historical FD1 capability opened: {relative}")
        return original(root, relative, **kwargs)

    monkeypatch.setattr(authorization, "_active_load_document", guarded_loader)
    authorization._active_validate_execution_closure_target_chain(SOURCE_ROOT)


def test_frozen_contract_successor_chain_roots_and_roles_are_exact() -> None:
    authority_document, diagnostic_document = (
        authorization._active_validate_frozen_contract_encoding_correction(
            SOURCE_ROOT
        )
    )
    assert authority_document["claim_boundary"][
        "gpu_execution_authorized_by_this_document"
    ] is False
    assert diagnostic_document["failed_attempt"][
        "target_runtime_child_return_code"
    ] == 1
    authorities = authorization._active_authorized_modifications(SOURCE_ROOT)
    assert [label for _document, label in authorities][-4:] == [
        "V8R4A denied-canary component boundary",
        "V8R4A frozen-contract encoding",
        "V8R4A GPU-state parent bind",
        "V8R4A admitted benchmark context",
    ]
    addendums = authorization._active_addendum_bindings(SOURCE_ROOT)
    assert addendums[
        "frozen_contract_encoding_correction_authorization"
    ]["sha256"] == (
        "2a59c7de8b8aefcbb3d21691db5a8da1e4b8f1a5461024f0338f389ff8ddcda1"
    )
    assert addendums["frozen_contract_encoding_failure_diagnostic"][
        "sha256"
    ] == "1d358bdecb1a5605b138b1916451f8c306f957a27257d328aa71de8d83cbf8ee"
    assert authorization.ACTIVE_TEST_RECEIPT.name.endswith("_V8R4A_CONTEXT1.json")
    assert authorization.ACTIVE_SOURCE_SNAPSHOT.name.endswith("_V8R4A_CONTEXT1.json")
    assert authorization.ACTIVE_PRETRAIN_AUTHORIZATION.name.endswith(
        "_V8R4A_CONTEXT1.json"
    )
    assert {
        "frozen_contract_encoding_correction_authorization",
        "frozen_contract_encoding_failure_diagnostic",
    } <= authorization._ACTIVE_ADDENDUM_ROLES
    assert authorization._TARGET_GOVERNANCE_ROLE_PATHS[
        "frozen_contract_encoding_correction_authorization"
    ] == authorization.ACTIVE_FROZEN_CONTRACT_ENCODING_CORRECTION
    assert authorization._TARGET_GOVERNANCE_ROLE_PATHS[
        "frozen_contract_encoding_failure_diagnostic"
    ] == authorization.ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC

    assert authorization._TARGET_SUPERSEDED_CONTRACT1_LIFECYCLE_ROOT.as_posix().endswith(
        "target_sealed_lifecycle_v8r4a_contract1"
    )
    assert authorization._TARGET_SUPERSEDED_LIFECYCLE_ROOT.as_posix().endswith(
        "target_sealed_lifecycle_v8r4a"
    )
    assert authorization._TARGET_SUPERSEDED_CONTRACT1_BENCHMARK_OUTPUT_ROOT.as_posix().endswith(
        "efficiency_benchmark_v8r4a_contract1"
    )
    assert authorization._TARGET_SUPERSEDED_BENCHMARK_OUTPUT_ROOT.as_posix().endswith(
        "efficiency_benchmark_v8r4a"
    )
    assert authorization._EXECUTION_CLOSURE_LEGACY_ACTIVE_OUTPUT_ROOT.endswith(
        "efficiency_benchmark_v8r4a"
    )
    assert authorization._EXECUTION_CLOSURE_LEGACY_ACTIVE_OUTPUT_ROOT != (
        authorization._TARGET_ACTIVE_BENCHMARK_OUTPUT_ROOT.as_posix()
    )

    output, lifecycle = authorization._target_expected_roots(
        Path("/project"),
        phase="efficiency_benchmark",
        outer_fold=3,
        entry_name="benchmark_hfr_v3r1_efficiency.py",
    )
    assert output.as_posix().endswith("efficiency_benchmark_v8r4a_context1")
    assert lifecycle.as_posix().endswith(
        "target_sealed_lifecycle_v8r4a_context1/efficiency_benchmark/"
        "benchmark_hfr_v3r1_efficiency/outer_3"
    )

    for field in (
        "parent_implementation_test_receipt",
        "parent_source_snapshot",
        "parent_pretrain_authorization",
        "failed_capability_receipt",
        "failed_completion_receipt",
    ):
        binding = authorization._FROZEN_CONTRACT_ENCODING_PARENT_BINDINGS[field]
        assert authorization.ACTIVE_HISTORICAL_FILES[binding["path"]] == (
            binding["file_sha256"]
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "historical_parent",
        "modification_cover",
        "active_filename",
        "successor_root",
        "legacy_root",
        "failed_receipt",
        "content_hash",
    ),
)
def test_frozen_contract_target_projection_fails_closed(mutation: str) -> None:
    authority_document, _ = authorization._active_load_document(
        SOURCE_ROOT,
        authorization.ACTIVE_FROZEN_CONTRACT_ENCODING_CORRECTION,
    )
    diagnostic_document, _ = authorization._active_load_document(
        SOURCE_ROOT,
        authorization.ACTIVE_FROZEN_CONTRACT_ENCODING_DIAGNOSTIC,
    )
    changed_authority = json.loads(json.dumps(authority_document))
    changed_diagnostic = json.loads(json.dumps(diagnostic_document))
    if mutation == "historical_parent":
        changed_authority["authority_basis"]["parent_source_snapshot"][
            "file_sha256"
        ] = "0" * 64
    elif mutation == "modification_cover":
        changed_authority["authorized_modifications"].pop()
    elif mutation == "active_filename":
        changed_authority["required_reauthorization"]["new_test_receipt"] = (
            "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CANARY1.json"
        )
    elif mutation == "successor_root":
        changed_authority["mandatory_invariants"][
            "successor_contract1_lifecycle_root"
        ] = changed_authority["mandatory_invariants"][
            "superseded_canary1_lifecycle_root_preserved_immutable"
        ]
    elif mutation == "legacy_root":
        changed_authority["mandatory_invariants"][
            "historical_execution_closure_authority_literal_unchanged"
        ] = changed_authority["mandatory_invariants"][
            "successor_contract1_output_root"
        ]
    elif mutation == "failed_receipt":
        changed_diagnostic["immutable_failure_receipts"]["completion_receipt"][
            "file_sha256"
        ] = "0" * 64
    else:
        changed_authority["content_sha256"] = "0" * 64
    with pytest.raises(
        authorization.AuthorizationError, match="frozen-contract"
    ):
        authorization._active_validate_frozen_contract_encoding_projection(
            changed_authority, changed_diagnostic
        )


def test_contract1_target_projection_never_opens_superseded_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_loader = authorization._active_load_document
    original_reader = authorization._active_read_binding
    forbidden = {
        authorization._FROZEN_CONTRACT_ENCODING_PARENT_PATHS[field]
        for field in (
            "parent_implementation_test_receipt",
            "parent_source_snapshot",
            "parent_pretrain_authorization",
            "failed_capability_receipt",
            "failed_completion_receipt",
        )
    }

    def guarded_reader(
        root: Path, relative: str | Path, **kwargs: Any
    ) -> tuple[dict[str, Any], bytes]:
        if Path(relative) in forbidden:
            raise AssertionError(f"superseded capability opened: {relative}")
        return original_reader(root, relative, **kwargs)

    def guarded_loader(
        root: Path, relative: str | Path, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if Path(relative) in forbidden:
            raise AssertionError(f"superseded capability loaded: {relative}")
        return original_loader(root, relative, **kwargs)

    monkeypatch.setattr(authorization, "_active_read_binding", guarded_reader)
    monkeypatch.setattr(authorization, "_active_load_document", guarded_loader)
    authorization._active_validate_execution_closure_target_chain(SOURCE_ROOT)


def test_rootbind1_successor_chain_roots_roles_and_parents_are_exact() -> None:
    authority_document, diagnostic_document = (
        authorization._active_validate_gpu_state_parent_bind_correction(
            SOURCE_ROOT
        )
    )
    assert authority_document["claim_boundary"][
        "gpu_execution_authorized_by_this_document"
    ] is False
    assert diagnostic_document["failed_attempt"][
        "target_scoped_pretrain_validation_reached"
    ] is True
    assert authorization._GPU_STATE_PARENT_BIND_SUCCESSOR_CHAIN_NAMES == {
        "new_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_ROOTBIND1.json",
        "new_source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_ROOTBIND1.json",
        "new_pretrain_authorization": (
            "PRETRAIN_AUTHORIZATION_V8R4A_ROOTBIND1.json"
        ),
    }
    assert authorization._ROOTBIND1_PARENT_PATHS[
        "parent_implementation_test_receipt"
    ].name == (
        authorization._GPU_STATE_PARENT_BIND_SUCCESSOR_CHAIN_NAMES[
            "new_test_receipt"
        ]
    )
    assert authorization._ROOTBIND1_PARENT_PATHS["parent_source_snapshot"].name == (
        authorization._GPU_STATE_PARENT_BIND_SUCCESSOR_CHAIN_NAMES[
            "new_source_snapshot"
        ]
    )
    assert authorization._ROOTBIND1_PARENT_PATHS[
        "parent_pretrain_authorization"
    ].name == (
        authorization._GPU_STATE_PARENT_BIND_SUCCESSOR_CHAIN_NAMES[
            "new_pretrain_authorization"
        ]
    )
    assert authorization._ROOTBIND1_LIFECYCLE_ROOT.as_posix().endswith(
        "target_sealed_lifecycle_v8r4a_rootbind1"
    )
    assert authorization._ROOTBIND1_BENCHMARK_OUTPUT_ROOT.as_posix().endswith(
        "efficiency_benchmark_v8r4a_rootbind1"
    )
    assert set(authorization._TARGET_REQUIRED_SUPERSEDED_CANARIES) == {
        "superseded_v8r4a_lifecycle_root",
        "superseded_v8r4a_output_root",
        "superseded_v8r4a_contract1_lifecycle_root",
        "superseded_v8r4a_contract1_output_root",
        "superseded_v8r4a_rootbind1_lifecycle_root",
        "superseded_v8r4a_rootbind1_output_root",
    }
    assert {
        "gpu_state_parent_bind_correction_authorization",
        "gpu_state_parent_bind_failure_diagnostic",
    } <= authorization._ACTIVE_ADDENDUM_ROLES
    assert authorization._TARGET_GOVERNANCE_ROLE_PATHS[
        "gpu_state_parent_bind_correction_authorization"
    ] == authorization.ACTIVE_GPU_STATE_PARENT_BIND_CORRECTION
    assert authorization._TARGET_GOVERNANCE_ROLE_PATHS[
        "gpu_state_parent_bind_failure_diagnostic"
    ] == authorization.ACTIVE_GPU_STATE_PARENT_BIND_DIAGNOSTIC
    addendums = authorization._active_addendum_bindings(SOURCE_ROOT)
    assert addendums["gpu_state_parent_bind_correction_authorization"][
        "sha256"
    ] == "b73d68199acad6fff780c76f05bd3daadc62b03c160af6efc407792efa87a4cd"
    assert addendums["gpu_state_parent_bind_failure_diagnostic"][
        "sha256"
    ] == "fd2aa011020289d94ab0548c679888ecc924d3ec179ca5908560a6e82074d628"
    assert [
        label
        for _document, label in authorization._active_authorized_modifications(
            SOURCE_ROOT
        )
    ][-2:] == [
        "V8R4A GPU-state parent bind",
        "V8R4A admitted benchmark context",
    ]
    for field in (
        "parent_implementation_test_receipt",
        "parent_source_snapshot",
        "parent_pretrain_authorization",
        "gpu_state_migration_receipt",
        "failed_contract1_capability_receipt",
        "failed_contract1_completion_receipt",
    ):
        binding = authorization._GPU_STATE_PARENT_BIND_PARENT_BINDINGS[field]
        assert authorization.ACTIVE_HISTORICAL_FILES[binding["path"]] == (
            binding["file_sha256"]
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "historical_parent",
        "migration_receipt",
        "failed_receipt",
        "modification_cover",
        "issuance",
        "governance_role",
        "canary_role",
        "parent_kind",
        "successor_root",
        "legacy_root",
        "security_boundary",
        "content_hash",
    ),
)
def test_rootbind1_target_projection_fails_closed(mutation: str) -> None:
    authority_document, _ = authorization._active_load_document(
        SOURCE_ROOT, authorization.ACTIVE_GPU_STATE_PARENT_BIND_CORRECTION
    )
    diagnostic_document, _ = authorization._active_load_document(
        SOURCE_ROOT, authorization.ACTIVE_GPU_STATE_PARENT_BIND_DIAGNOSTIC
    )
    changed_authority = json.loads(json.dumps(authority_document))
    changed_diagnostic = json.loads(json.dumps(diagnostic_document))
    if mutation == "historical_parent":
        changed_authority["authority_basis"]["parent_source_snapshot"][
            "file_sha256"
        ] = "0" * 64
    elif mutation == "migration_receipt":
        changed_authority["authority_basis"]["gpu_state_migration_receipt"][
            "file_sha256"
        ] = "0" * 64
    elif mutation == "failed_receipt":
        changed_diagnostic["immutable_failure_receipts"]["completion_receipt"][
            "file_sha256"
        ] = "0" * 64
    elif mutation == "modification_cover":
        changed_authority["authorized_modifications"].pop()
    elif mutation == "issuance":
        changed_authority["required_reauthorization"]["new_test_receipt"] = (
            "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTRACT1.json"
        )
    elif mutation == "governance_role":
        changed_authority["required_reauthorization"]["new_governance_roles"][
            0
        ] = "frozen_contract_encoding_correction_authorization"
    elif mutation == "canary_role":
        changed_authority["required_reauthorization"]["new_denied_canary_roles"][
            0
        ] = "superseded_v8r4a_lifecycle_root"
    elif mutation == "parent_kind":
        changed_authority["mandatory_invariants"][
            "gpu_state_root_mount_kind"
        ] = "rw_bind_fd"
    elif mutation == "successor_root":
        changed_authority["mandatory_invariants"][
            "successor_rootbind1_lifecycle_root"
        ] = authorization._TARGET_SUPERSEDED_CONTRACT1_LIFECYCLE_ROOT.as_posix()
    elif mutation == "legacy_root":
        changed_authority["mandatory_invariants"][
            "historical_execution_closure_authority_literal_unchanged"
        ] = authorization._TARGET_ACTIVE_BENCHMARK_OUTPUT_ROOT.as_posix()
    elif mutation == "security_boundary":
        changed_authority["required_reauthorization"][
            "required_true_security_boundary"
        ] = "exactly_three_mutable_state_directory_mounts"
    else:
        changed_authority["content_sha256"] = "0" * 64
    with pytest.raises(
        authorization.AuthorizationError, match="parent-bind target projection"
    ):
        authorization._active_validate_gpu_state_parent_bind_projection(
            changed_authority, changed_diagnostic
        )


def test_rootbind1_target_projection_never_opens_contract1_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_loader = authorization._active_load_document
    original_reader = authorization._active_read_binding
    forbidden = {
        authorization._GPU_STATE_PARENT_BIND_PARENT_PATHS[field]
        for field in (
            "parent_implementation_test_receipt",
            "parent_source_snapshot",
            "parent_pretrain_authorization",
            "failed_contract1_capability_receipt",
            "failed_contract1_completion_receipt",
        )
    }

    def guarded_reader(
        root: Path, relative: str | Path, **kwargs: Any
    ) -> tuple[dict[str, Any], bytes]:
        if Path(relative) in forbidden:
            raise AssertionError(f"historical ROOTBIND1 parent opened: {relative}")
        return original_reader(root, relative, **kwargs)

    def guarded_loader(
        root: Path, relative: str | Path, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if Path(relative) in forbidden:
            raise AssertionError(f"historical ROOTBIND1 parent loaded: {relative}")
        return original_loader(root, relative, **kwargs)

    monkeypatch.setattr(authorization, "_active_read_binding", guarded_reader)
    monkeypatch.setattr(authorization, "_active_load_document", guarded_loader)
    authorization._active_validate_execution_closure_target_chain(SOURCE_ROOT)


def test_context1_host_chain_roles_prefixes_and_roots_are_exact() -> None:
    authority_document, diagnostic_document = (
        authorization._active_validate_admitted_context_correction(SOURCE_ROOT)
    )
    assert authority_document["content_sha256"] == (
        authorization._ADMITTED_CONTEXT_AUTHORITY_CONTENT_SHA256
    )
    assert diagnostic_document["content_sha256"] == (
        authorization._ADMITTED_CONTEXT_DIAGNOSTIC_CONTENT_SHA256
    )
    assert authorization._ADMITTED_CONTEXT_SUCCESSOR_CHAIN_NAMES == {
        "new_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTEXT1.json",
        "new_source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTEXT1.json",
        "new_pretrain_authorization": "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json",
    }
    assert authorization.ACTIVE_TEST_RECEIPT.name == (
        authorization._ADMITTED_CONTEXT_SUCCESSOR_CHAIN_NAMES["new_test_receipt"]
    )
    assert authorization.ACTIVE_SOURCE_SNAPSHOT.name == (
        authorization._ADMITTED_CONTEXT_SUCCESSOR_CHAIN_NAMES[
            "new_source_snapshot"
        ]
    )
    assert authorization.ACTIVE_PRETRAIN_AUTHORIZATION.name == (
        authorization._ADMITTED_CONTEXT_SUCCESSOR_CHAIN_NAMES[
            "new_pretrain_authorization"
        ]
    )
    assert authorization.ACTIVE_AUTHORIZATION_GENERATION == "CONTEXT1"
    assert authorization._CONTEXT1_FULL_BENCHMARK_CONTEXT == {
        "campaign_revision": "V8R4",
        "infrastructure_revision": "V8R4A",
        "authorization_generation": "CONTEXT1",
        "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    assert authorization._TARGET_LIFECYCLE_ROOT.as_posix().endswith(
        "target_sealed_lifecycle_v8r4a_context1"
    )
    assert authorization._TARGET_ACTIVE_BENCHMARK_OUTPUT_ROOT.as_posix().endswith(
        "efficiency_benchmark_v8r4a_context1"
    )
    assert {
        "admitted_context_correction_authorization",
        "admitted_context_failure_diagnostic",
    } <= authorization._ACTIVE_ADDENDUM_ROLES
    assert (
        "benchmark_admitted_context_generation_isolated"
        in authorization._TARGET_REQUIRED_TRUE_BOUNDARIES
    )
    assert len(authorization.ACTIVE_IMPLEMENTATION_PATHS) == 35
    assert len(authorization.ACTIVE_FIXED_TEST_PATHS) == 13
    authorization._active_validate_postfailure_ledger_prefixes(
        SOURCE_ROOT, require_exact=True
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "parent",
        "modification_cover",
        "context_generation",
        "terminal_prefix",
        "issuance",
        "canary",
        "security_boundary",
        "successor_root",
        "failed_receipt",
        "content_hash",
    ),
)
def test_context1_target_projection_fails_closed(mutation: str) -> None:
    authority_document, _ = authorization._active_load_document(
        SOURCE_ROOT, authorization.ACTIVE_ADMITTED_CONTEXT_CORRECTION
    )
    diagnostic_document, _ = authorization._active_load_document(
        SOURCE_ROOT, authorization.ACTIVE_ADMITTED_CONTEXT_DIAGNOSTIC
    )
    changed_authority = json.loads(json.dumps(authority_document))
    changed_diagnostic = json.loads(json.dumps(diagnostic_document))
    if mutation == "parent":
        changed_authority["authority_basis"]["parent_source_snapshot"][
            "sha256"
        ] = "0" * 64
    elif mutation == "modification_cover":
        changed_authority["authorized_modifications"].pop()
    elif mutation == "context_generation":
        changed_authority["mandatory_invariants"]["active_benchmark_context"][
            "authorization_generation"
        ] = "ROOTBIND1"
    elif mutation == "terminal_prefix":
        changed_diagnostic["ledger_evidence"]["usage_postlaunch"][
            "tail_record_sha256"
        ] = "0" * 64
    elif mutation == "issuance":
        changed_authority["required_reauthorization"]["new_test_receipt"] = (
            "IMPLEMENTATION_TEST_RECEIPT_V8R4A_ROOTBIND1.json"
        )
    elif mutation == "canary":
        changed_authority["required_reauthorization"]["new_denied_canary_roles"] = []
    elif mutation == "security_boundary":
        changed_authority["required_reauthorization"][
            "required_true_security_boundary"
        ] = "gpu_state_parent_identity_readonly_bind"
    elif mutation == "successor_root":
        changed_authority["mandatory_invariants"][
            "successor_context1_output_root"
        ] = authorization._ROOTBIND1_BENCHMARK_OUTPUT_ROOT.as_posix()
    elif mutation == "failed_receipt":
        changed_diagnostic["immutable_failure_receipts"][
            "gpu_terminal_result"
        ]["sha256"] = "0" * 64
    else:
        changed_authority["content_sha256"] = "0" * 64
    with pytest.raises(authorization.AuthorizationError, match="admitted-context"):
        authorization._active_validate_admitted_context_projection(
            changed_authority, changed_diagnostic
        )


def test_context1_target_projection_never_opens_rootbind1_denied_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_loader = authorization._active_load_document
    original_reader = authorization._active_read_binding
    forbidden = {
        authorization._ROOTBIND1_PARENT_PATHS[field]
        for field in (
            "parent_implementation_test_receipt",
            "parent_source_snapshot",
            "parent_pretrain_authorization",
            "failed_rootbind1_target_capability_receipt",
            "failed_rootbind1_target_completion_receipt",
            "failed_rootbind1_benchmark_invocation",
            "failed_rootbind1_gpu_invocation",
            "failed_rootbind1_gpu_terminal_result",
        )
    }

    def guarded_reader(
        root: Path, relative: str | Path, **kwargs: Any
    ) -> tuple[dict[str, Any], bytes]:
        if Path(relative) in forbidden:
            raise AssertionError(f"superseded ROOTBIND1 path opened: {relative}")
        return original_reader(root, relative, **kwargs)

    def guarded_loader(
        root: Path, relative: str | Path, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if Path(relative) in forbidden:
            raise AssertionError(f"superseded ROOTBIND1 document loaded: {relative}")
        return original_loader(root, relative, **kwargs)

    monkeypatch.setattr(authorization, "_active_read_binding", guarded_reader)
    monkeypatch.setattr(authorization, "_active_load_document", guarded_loader)
    authorization._active_validate_execution_closure_target_chain(SOURCE_ROOT)


def test_context1_postfailure_prefix_accepts_append_but_issuance_requires_exact(
    tmp_path: Path,
) -> None:
    for relative in (
        authorization.ACTIVE_USAGE_LEDGER,
        authorization.ACTIVE_EXECUTION_LEDGER,
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((SOURCE_ROOT / relative).read_bytes())
        destination.chmod(0o644)
    authorization._active_validate_postfailure_ledger_prefixes(
        tmp_path, require_exact=True
    )
    execution = tmp_path / authorization.ACTIVE_EXECUTION_LEDGER
    execution.write_bytes(execution.read_bytes() + b"{}\n")
    authorization._active_validate_postfailure_ledger_prefixes(
        tmp_path, require_exact=False
    )
    with pytest.raises(authorization.AuthorizationError, match="postfailure prefix"):
        authorization._active_validate_postfailure_ledger_prefixes(
            tmp_path, require_exact=True
        )


@pytest.mark.parametrize("mutation", ("missing_generation", "extra_key", "old_context"))
def test_context1_efficiency_admitted_scope_is_exact_before_wrapper_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    called = False

    def forbidden_loader(_root: Path) -> Any:
        nonlocal called
        called = True
        raise AssertionError("invalid context reached wrapper loader")

    expected = dict(authorization._CONTEXT1_FULL_BENCHMARK_CONTEXT)
    if mutation == "missing_generation":
        expected.pop("authorization_generation")
    elif mutation == "extra_key":
        expected["unexpected"] = False
    else:
        expected = dict(authorization._ROOTBIND1_BENCHMARK_CONTEXT)
    monkeypatch.setattr(authorization, "_load_gpu_admitted_validator", forbidden_loader)
    with pytest.raises(authorization.AuthorizationError, match="fixed identity"):
        authorization._active_revalidate_admitted(
            tmp_path,
            {"phase": "efficiency_benchmark", "context": expected},
            tmp_path / authorization.ACTIVE_PRETRAIN_AUTHORIZATION,
            expected_phase="efficiency_benchmark",
            expected_context=expected,
        )
    assert called is False


def test_context1_expected_pretrain_scope_helper_is_fresh_and_exact() -> None:
    first = authorization._active_expected_pretrain_scopes()
    second = authorization._active_expected_pretrain_scopes()
    assert first == second
    assert first is not second
    assert first["discovery_scope"] is not second["discovery_scope"]
    assert first["efficiency_benchmark_scope"] == (
        authorization._CONTEXT1_EFFICIENCY_BENCHMARK_SCOPE
    )
    first["discovery_scope"]["training_jobs_max"] = 17
    assert second["discovery_scope"]["training_jobs_max"] == 18


@pytest.mark.parametrize(
    "role",
    (
        "discovery_scope",
        "efficiency_benchmark_scope",
        "promotion_reuse_scope",
        "admitted_child_scope",
    ),
)
@pytest.mark.parametrize("mutation", ("missing", "extra", "tamper"))
def test_context1_pretrain_scope_missing_extra_and_tamper_fail_closed(
    role: str, mutation: str
) -> None:
    document: dict[str, Any] = authorization._active_expected_pretrain_scopes()
    scope = document[role]
    if mutation == "missing":
        scope.pop(next(iter(scope)))
    elif mutation == "extra":
        scope["unexpected"] = False
    elif role == "discovery_scope":
        scope["training_jobs_max"] = True
    elif role == "efficiency_benchmark_scope":
        scope["authorization_generation"] = "ROOTBIND1"
    elif role == "promotion_reuse_scope":
        scope["final_gpu_execution_owners"] = 48
    else:
        scope["unrelated_open_lifecycle_allowed"] = True
    with pytest.raises(authorization.AuthorizationError, match="pretrain scope"):
        authorization._active_validate_pretrain_scope_fields(
            document, label="fixture"
        )


def test_context1_recomputed_self_hash_cannot_authorize_scope_tamper() -> None:
    document = _content_document(
        authorization_generation=authorization.ACTIVE_AUTHORIZATION_GENERATION,
        **authorization._active_expected_pretrain_scopes()
    )
    document["promotion_reuse_scope"]["final_gpu_execution_owners"] = 50
    document.pop("content_sha256")
    document["content_sha256"] = authorization.semantic_sha256(document)
    authorization.verify_content_hash(document, path=Path("synthetic.json"))
    with pytest.raises(authorization.AuthorizationError, match="pretrain scope"):
        authorization._active_validate_pretrain_scope_fields(
            document, label="self-hashed fixture"
        )


@pytest.mark.parametrize("projection", ("host", "target"))
def test_context1_host_and_target_reject_same_self_hashed_scope_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    projection: str,
) -> None:
    # Exercise the schema validators behind the independently tested terminal
    # authority gate; this fixture is intentionally non-authoritative.
    monkeypatch.setattr(
        authorization, "_active_require_test_enforcement_authority", lambda: None
    )
    document = _content_document(
        authorization_generation=authorization.ACTIVE_AUTHORIZATION_GENERATION,
        **authorization._active_expected_pretrain_scopes()
    )
    document["discovery_scope"]["outer_test_features_or_targets_authorized"] = (
        True
    )
    document.pop("content_sha256")
    document["content_sha256"] = authorization.semantic_sha256(document)
    authorization.verify_content_hash(document, path=Path("synthetic.json"))
    binding = {
        "path": authorization.ACTIVE_PRETRAIN_AUTHORIZATION.as_posix(),
        "sha256": "0" * 64,
        "bytes": 1,
        "mode": "0444",
        "nlink": 1,
        "st_dev": 1,
        "st_ino": 1,
    }
    monkeypatch.setattr(
        authorization,
        "_active_load_document",
        lambda *_args, **_kwargs: (document, binding),
    )
    with pytest.raises(authorization.AuthorizationError, match="pretrain scope"):
        if projection == "host":
            monkeypatch.setattr(
                authorization,
                "_active_validate_snapshot",
                lambda *_args, **_kwargs: ({}, {}, []),
            )
            monkeypatch.setattr(
                authorization,
                "_active_validate_test_receipt",
                lambda *_args, **_kwargs: ({}, {}),
            )
            authorization._active_validate_pretrain_common(tmp_path, None)
        else:
            monkeypatch.setattr(
                authorization,
                "_active_validate_execution_closure_target_chain",
                lambda *_args, **_kwargs: None,
            )
            authorization._active_target_documents(tmp_path, {}, {})


@pytest.mark.parametrize(
    "mutation", ("historical_parent", "modification_cover", "superseded_chain")
)
def test_fd_closure_target_projection_fails_closed(mutation: str) -> None:
    authority_document, _ = authorization._active_load_document(
        SOURCE_ROOT, authorization.ACTIVE_FD_CLOSURE_CORRECTION
    )
    diagnostic_document, _ = authorization._active_load_document(
        SOURCE_ROOT, authorization.ACTIVE_FD_CLOSURE_DIAGNOSTIC
    )
    changed_authority = json.loads(json.dumps(authority_document))
    changed_diagnostic = json.loads(json.dumps(diagnostic_document))
    if mutation == "historical_parent":
        changed_authority["authority_basis"]["parent_source_snapshot"][
            "file_sha256"
        ] = "0" * 64
    elif mutation == "modification_cover":
        changed_authority["authorized_modifications"].pop()
    else:
        changed_diagnostic["superseded_pretrain_chain"]["source_snapshot"][
            "file_sha256"
        ] = "0" * 64
    with pytest.raises(authorization.AuthorizationError, match="FD-closure target"):
        authorization._active_validate_fd_closure_projection(
            changed_authority, changed_diagnostic
        )


@pytest.mark.parametrize("mutation", ("addition", "deletion", "tamper"))
def test_migration_source_succession_projection_fails_closed(
    mutation: str,
) -> None:
    authority, _ = authorization._active_load_document(
        SOURCE_ROOT,
        authorization.ACTIVE_MIGRATION_SOURCE_SUCCESSION_CORRECTION,
    )
    diagnostic, _ = authorization._active_load_document(
        SOURCE_ROOT,
        authorization.ACTIVE_MIGRATION_SOURCE_SUCCESSION_DIAGNOSTIC,
    )
    changed = json.loads(json.dumps(authority))
    if mutation == "addition":
        changed["unexpected"] = False
    elif mutation == "deletion":
        del changed["authority_basis"]["diagnostic"]
    else:
        changed["authority_basis"]["execution_closure_authority"][
            "file_sha256"
        ] = "0" * 64
    with pytest.raises(authorization.AuthorizationError, match="source-succession"):
        authorization._active_validate_migration_source_succession_projection(
            SOURCE_ROOT, changed, diagnostic
        )


@pytest.mark.parametrize("mutation", ("addition", "deletion", "tamper"))
def test_execution_closure_historical_projection_fails_closed(
    mutation: str,
) -> None:
    authority, _ = authorization._active_load_document(
        SOURCE_ROOT, authorization.ACTIVE_EXECUTION_CLOSURE_CORRECTION
    )
    changed = json.loads(json.dumps(authority))
    history = changed["authority_basis"]["historical_benchmark_prefix"]
    if mutation == "addition":
        history["unexpected"] = False
    elif mutation == "deletion":
        history["entries"].pop()
    else:
        history["entries"][0]["file_sha256"] = "0" * 64
    with pytest.raises(authorization.AuthorizationError, match="historical projection"):
        authorization._active_validate_execution_closure_projection(changed)


def test_target_topology_and_split_aggregation_roles_are_exact() -> None:
    root = Path("/project")
    cases = (
        (
            "efficiency_benchmark",
            3,
            "benchmark_hfr_v3r1_efficiency.py",
            "efficiency_benchmark_v8r4a_context1",
        ),
        (
            "discovery",
            4,
            "run_hfr_v3r1_discovery_campaign.py",
            "discovery_v8r4/shards/outer_4",
        ),
        (
            "promotion_training",
            5,
            "run_fixed_hfr_v3r1_oof_campaign.py",
            "fixed_oof_v8r4/promotion_training_shards/outer_5",
        ),
        (
            "promotion_prediction",
            2,
            "run_fixed_hfr_v3r1_oof_campaign.py",
            "fixed_oof_v8r4/prediction_shards/outer_2",
        ),
        (
            "discovery_aggregation",
            None,
            "run_hfr_v3r1_discovery_campaign.py",
            "discovery_v8r4/aggregation_v8r4a",
        ),
        (
            "promotion_aggregation",
            None,
            "run_fixed_hfr_v3r1_oof_campaign.py",
            "fixed_oof_v8r4/aggregation_v8r4a",
        ),
    )
    for phase, outer, entry, output_suffix in cases:
        output, lifecycle = authorization._target_expected_roots(
            root, phase=phase, outer_fold=outer, entry_name=entry
        )
        assert output.as_posix().endswith(output_suffix)
        assert lifecycle == (
            root
            / authorization._TARGET_LIFECYCLE_ROOT
            / phase
            / Path(entry).stem
            / ("global" if outer is None else f"outer_{outer}")
        )
    discovery_roles = authorization._TARGET_GOVERNANCE_ROLES_BY_PHASE[
        "discovery_aggregation"
    ]
    promotion_roles = authorization._TARGET_GOVERNANCE_ROLES_BY_PHASE[
        "promotion_aggregation"
    ]
    assert {"discovery_shard_seal_outer3", "discovery_shard_seal_outer4"} <= (
        discovery_roles
    )
    assert not any(role.startswith("model_source_seal_") for role in discovery_roles)
    assert {
        *(f"model_source_seal_outer{outer}" for outer in range(6)),
        *(f"prediction_shard_seal_outer{outer}" for outer in range(6)),
    } <= promotion_roles


def test_validator_and_target_runtime_governance_role_abi_are_identical() -> None:
    for phase, entry in authorization._TARGET_ENTRY_BY_PHASE.items():
        runtime_roles = target_runtime._governance_roles_for(
            phase=phase, entry_name=entry
        )
        assert set(runtime_roles) == authorization._TARGET_GOVERNANCE_ROLES_BY_PHASE[
            phase
        ]


def _publish_stage_predecessor(root: Path, relative: Path) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"{}\n")
    path.chmod(0o444)
    return path


@pytest.mark.parametrize(
    ("stage", "predecessors"),
    (
        ("test_receipt", ()),
        ("source_snapshot", (authorization.ACTIVE_TEST_RECEIPT,)),
        (
            "pretrain_authorization",
            (
                authorization.ACTIVE_TEST_RECEIPT,
                authorization.ACTIVE_SOURCE_SNAPSHOT,
            ),
        ),
    ),
)
def test_context1_create_stage_accepts_only_exact_predecessor_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    predecessors: tuple[Path, ...],
) -> None:
    exact_calls: list[bool] = []
    monkeypatch.setattr(
        authorization,
        "_active_validate_postfailure_ledger_prefixes",
        lambda *_args, **kwargs: exact_calls.append(kwargs["require_exact"]),
    )
    for relative in predecessors:
        _publish_stage_predecessor(tmp_path, relative)
    authorization._active_validate_create_stage(tmp_path, stage)
    assert exact_calls == [True]


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_predecessor",
        "successor_present",
        "predecessor_directory",
        "predecessor_symlink",
        "predecessor_dangling_symlink",
        "fresh_directory",
        "fresh_file",
        "fresh_symlink",
        "fresh_dangling_symlink",
    ),
)
def test_context1_create_stage_rejects_wedging_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    monkeypatch.setattr(
        authorization,
        "_active_validate_postfailure_ledger_prefixes",
        lambda *_args, **_kwargs: None,
    )
    stage = "source_snapshot"
    predecessor = tmp_path / authorization.ACTIVE_TEST_RECEIPT
    if mutation != "missing_predecessor":
        predecessor.parent.mkdir(parents=True, exist_ok=True)
        if mutation == "predecessor_directory":
            predecessor.mkdir()
        elif mutation in {"predecessor_symlink", "predecessor_dangling_symlink"}:
            target = tmp_path / (
                "backing.json"
                if mutation == "predecessor_symlink"
                else "missing-backing.json"
            )
            if mutation == "predecessor_symlink":
                target.write_bytes(b"{}\n")
                target.chmod(0o444)
            predecessor.symlink_to(target)
        else:
            predecessor.write_bytes(b"{}\n")
            predecessor.chmod(0o444)
    if mutation == "successor_present":
        _publish_stage_predecessor(
            tmp_path, authorization.ACTIVE_SOURCE_SNAPSHOT
        )
    elif mutation.startswith("fresh_"):
        fresh = tmp_path / authorization._TARGET_ACTIVE_BENCHMARK_OUTPUT_ROOT
        fresh.parent.mkdir(parents=True, exist_ok=True)
        if mutation == "fresh_directory":
            fresh.mkdir()
        elif mutation == "fresh_file":
            fresh.write_bytes(b"occupied\n")
        else:
            target = tmp_path / (
                "fresh-backing"
                if mutation == "fresh_symlink"
                else "missing-fresh-backing"
            )
            if mutation == "fresh_symlink":
                target.mkdir()
            fresh.symlink_to(target, target_is_directory=True)
    with pytest.raises(authorization.AuthorizationError, match="CONTEXT1"):
        authorization._active_validate_create_stage(tmp_path, stage)


def test_target_snapshot_metadata_projection_never_opens_historical_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"synthetic-python\n")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_bytes(b"[project]\nname='fixture'\n")
    pyproject.chmod(0o444)
    runtime = {key: {} for key in authorization._ACTIVE_RUNTIME_STATE_KEYS}
    test_receipt = {"runtime_state_after": runtime}
    real_binding = authorization._active_binding
    opened: list[Path] = []

    def binding_spy(root: Path, relative: Any, **kwargs: Any) -> dict[str, Any]:
        relative_path = Path(relative)
        opened.append(relative_path)
        if relative_path == authorization._ACTIVE_HISTORICAL_V8R3_SNAPSHOT:
            raise AssertionError("target projection opened historical V8R3 bytes")
        return real_binding(root, relative_path, **kwargs)

    monkeypatch.setattr(authorization, "_active_binding", binding_spy)
    expected = authorization._active_expected_snapshot_metadata(
        tmp_path, test_receipt, target_safe=True
    )
    authorization._active_validate_snapshot_metadata(
        tmp_path,
        expected,
        test_receipt,
        label="target fixture",
        target_safe=True,
    )
    assert authorization._ACTIVE_HISTORICAL_V8R3_SNAPSHOT not in opened
    assert expected["historical_v8r3_parent"] == (
        authorization._CONTEXT1_HISTORICAL_V8R3_PARENT_BINDING
    )


@pytest.mark.parametrize(
    ("role", "mutation"),
    (
        ("historical_v8r3_parent", "missing"),
        ("historical_v8r3_parent", "extra"),
        ("historical_v8r3_parent", "tamper"),
        ("runtime_state_at_snapshot", "missing"),
        ("runtime_state_at_snapshot", "extra"),
        ("runtime_state_at_snapshot", "tamper"),
        ("environment", "missing"),
        ("environment", "extra"),
        ("environment", "tamper"),
    ),
)
def test_self_hashed_snapshot_metadata_mutation_fails_host_and_target_projection(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    mutation: str,
) -> None:
    expected = {
        "historical_v8r3_parent": {"path": "parent", "sha256": "a" * 64},
        "runtime_state_at_snapshot": {"usage_state": {"record_count": 77}},
        "environment": {"python_executable_sha256": "b" * 64},
    }
    monkeypatch.setattr(
        authorization,
        "_active_expected_snapshot_metadata",
        lambda *_args, **_kwargs: json.loads(json.dumps(expected)),
    )
    document = _content_document(
        authorization_generation=authorization.ACTIVE_AUTHORIZATION_GENERATION,
        **json.loads(json.dumps(expected)),
    )
    if mutation == "missing":
        document.pop(role)
    elif mutation == "extra":
        document[role]["unexpected"] = False
    elif role == "historical_v8r3_parent":
        document[role]["sha256"] = "0" * 64
    elif role == "runtime_state_at_snapshot":
        document[role]["usage_state"]["record_count"] = 76
    else:
        document[role]["python_executable_sha256"] = "0" * 64
    document.pop("content_sha256")
    document["content_sha256"] = authorization.semantic_sha256(document)
    authorization.verify_content_hash(document, path=Path("synthetic.json"))
    for label, target_safe in (("host", False), ("target", True)):
        with pytest.raises(authorization.AuthorizationError, match="metadata"):
            authorization._active_validate_snapshot_metadata(
                Path("/unused"),
                document,
                {"runtime_state_after": {}},
                label=label,
                target_safe=target_safe,
            )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_schema",
        "boolean_schema",
        "missing_generation",
        "rootbind1_generation",
        "other_generation",
    ),
)
def test_self_hashed_trio_identity_mutation_fails_closed(mutation: str) -> None:
    document = _content_document(
        authorization_generation=authorization.ACTIVE_AUTHORIZATION_GENERATION
    )
    if mutation == "missing_schema":
        del document["schema_version"]
    elif mutation == "boolean_schema":
        document["schema_version"] = True
    elif mutation == "missing_generation":
        del document["authorization_generation"]
    elif mutation == "rootbind1_generation":
        document["authorization_generation"] = "ROOTBIND1"
    else:
        document["authorization_generation"] = "CONTEXT2"
    document.pop("content_sha256")
    document["content_sha256"] = authorization.semantic_sha256(document)
    authorization.verify_content_hash(document, path=Path("synthetic.json"))
    with pytest.raises(authorization.AuthorizationError, match="schema/generation"):
        authorization._active_validate_trio_identity(document, label="fixture")


def test_context1_trio_generation_schema_and_rootbind1_absence_are_explicit() -> None:
    assert all(
        "authorization_generation" in keys
        for keys in (
            authorization._ACTIVE_TEST_RECEIPT_KEYS,
            authorization._ACTIVE_SNAPSHOT_KEYS,
            authorization._ACTIVE_PRETRAIN_KEYS,
        )
    )
    for relative in authorization._ROOTBIND1_PARENT_PATHS.values():
        if relative.name not in {
            "IMPLEMENTATION_TEST_RECEIPT_V8R4A_ROOTBIND1.json",
            "V3R1_SOURCE_SNAPSHOT_V8R4A_ROOTBIND1.json",
            "PRETRAIN_AUTHORIZATION_V8R4A_ROOTBIND1.json",
        }:
            continue
        document = json.loads((SOURCE_ROOT / relative).read_text(encoding="utf-8"))
        assert "authorization_generation" not in document
