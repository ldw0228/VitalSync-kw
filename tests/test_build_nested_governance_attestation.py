from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_nested_governance_attestation.py"
SPEC = importlib.util.spec_from_file_location(
    "build_nested_governance_attestation", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

SEAL_SCRIPT = ROOT / "scripts/seal_runtime_inputs.py"
SEAL_SPEC = importlib.util.spec_from_file_location(
    "seal_runtime_inputs_for_governance_test", SEAL_SCRIPT
)
assert SEAL_SPEC is not None and SEAL_SPEC.loader is not None
SEAL = importlib.util.module_from_spec(SEAL_SPEC)
SEAL_SPEC.loader.exec_module(SEAL)


def _seal(value: dict) -> dict:
    value = dict(value)
    value["content_sha256"] = AUDIT.canonical_content_sha256(value)
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _file_binding(path: Path, payload: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": AUDIT.sha256_file(path),
        "bytes": len(payload),
    }


def _execution_fixture(tmp_path: Path) -> tuple[dict, Path, dict, str]:
    retrain_index = _seal({"kind": "retrain-index", "completed_units": 30})
    retrain_path = tmp_path / "retrain/index.json"
    _write_json(retrain_path, retrain_index)
    retrain_binding = {
        "path": str(retrain_path.resolve()),
        "sha256": AUDIT.sha256_file(retrain_path),
        "bytes": retrain_path.stat().st_size,
        "content_sha256": retrain_index["content_sha256"],
    }
    source = tmp_path / "sealed_source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    seal = SEAL.inventory(
        sources=[source], trees=[], bindings=[], post_launch_attestation=False
    )
    seal_path = tmp_path / "execution_runtime_seal.json"
    SEAL.atomic_json(seal_path, seal)
    seal_binding = {
        "path": str(seal_path.resolve()),
        "sha256": AUDIT.sha256_file(seal_path),
        "bytes": seal_path.stat().st_size,
        "content_sha256": seal["content_sha256"],
        "verified_files": 1,
    }
    supervisor = AUDIT.CANONICAL_EXECUTION_SUPERVISOR
    supervisor_binding = {
        "path": str(supervisor.resolve()),
        "sha256": AUDIT.sha256_file(supervisor),
        "bytes": supervisor.stat().st_size,
    }
    receipt = _seal(
        {
            "schema_version": 1,
            "classification": "sealed_non_test_proposer_execution_attestation",
            "outer_test_opened": False,
            "outer_test_record_count": 0,
            "commercial_claim_authorized": False,
            "expected_units": 30,
            "completed_units": 30,
            "one_new_unit_per_invocation": True,
            "runtime_seal_verified_before_and_after_every_invocation": True,
            "invocations_this_resume": 30,
            "runtime_input_seal": seal_binding,
            "campaign_index": retrain_binding,
            "unit_command": ["python", "runner.py"],
            "supervisor": supervisor_binding,
        }
    )
    receipt_path = tmp_path / "execution_attestation.json"
    _write_json(receipt_path, receipt)
    receipt_binding = {
        "path": str(receipt_path.resolve()),
        "sha256": AUDIT.sha256_file(receipt_path),
        "bytes": receipt_path.stat().st_size,
        "content_sha256": receipt["content_sha256"],
        "classification": receipt["classification"],
    }
    provenance = {
        "retrain_execution_attestation_required": True,
        "retrain_execution_attestation": receipt_binding,
        "retrain_execution": {
            "required": True,
            "execution_attestation": receipt_binding,
            "authoritative_runtime_input_seal": {
                **seal_binding,
                "attestation_phase": "prelaunch",
                "payloads_rehashed_during_merge": True,
            },
            "supervisor": supervisor_binding,
            "completion_evidence": {
                "expected_units": 30,
                "completed_units": 30,
                "campaign_index": retrain_binding,
                "one_new_unit_per_invocation": True,
                "runtime_seal_verified_before_and_after_every_invocation": True,
                "invocations_this_resume": 30,
                "single_supervisor_execution_covered_all_units": True,
            },
            "supersession_note": None,
            "attestation_content_verified": True,
            "campaign_index_30_of_30_verified": True,
            "runtime_seal_live_rehashed": True,
            "supervisor_live_rehashed": True,
            "canonical_supervisor_and_unit_command_verified": True,
        },
    }
    return provenance, retrain_path, retrain_binding, retrain_index["content_sha256"]


def test_index_validator_rehashes_every_record_artifact(tmp_path: Path) -> None:
    manifest = tmp_path / "outer_0/inner_pred_2.json"
    _write_json(manifest, {"split": 1})
    record = {
        "outer_fold": 0,
        "seed": 20260828,
        "role": "hcs_train_oof",
        "manifest": str(manifest),
        "manifest_sha256": AUDIT.sha256_file(manifest),
        "checkpoint": _file_binding(tmp_path / "checkpoint.pt", b"checkpoint"),
        "all_window_prediction": _file_binding(
            tmp_path / "prediction.npz", b"prediction"
        ),
    }
    index = _seal(
        {
            "schema_version": 1,
            "classification": AUDIT.INDEX_CLASSIFICATION,
            "outer_test_opened": False,
            "outer_test_record_count": 0,
            "requested_units": 1,
            "completed_units": 1,
            "records": [record],
        }
    )
    path = tmp_path / "index.json"
    _write_json(path, index)
    _, _, records = AUDIT._validate_index(
        path, expected_records=1, label="fixture index"
    )
    assert len(records) == 1

    Path(record["checkpoint"]["path"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        AUDIT._validate_index(path, expected_records=1, label="fixture index")


def test_runtime_seal_phase_and_payload_are_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    tree = tmp_path / "cache"
    tree.mkdir()
    (tree / "payload.bin").write_bytes(b"payload")
    document = SEAL.inventory(
        sources=[source],
        trees=[tree],
        bindings=[],
        post_launch_attestation=False,
    )
    seal_path = tmp_path / "seal.json"
    SEAL.atomic_json(seal_path, document)
    result = AUDIT._verify_runtime_seal(seal_path, expected_phase="prelaunch")
    assert result["payloads_rehashed_during_this_audit"] is True
    assert result["verified_files"] == 2
    with pytest.raises(RuntimeError, match="expected post_launch"):
        AUDIT._verify_runtime_seal(seal_path, expected_phase="post_launch")


def test_attestation_records_exact_60_plus_30_and_honest_label_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds = (20260828, 20260829, 20260830)

    def records(folds: tuple[int, ...], source: str) -> dict:
        result = {}
        for seed in seeds:
            for fold in folds:
                names = [
                    f"inner_pred_{i}.json" for i in range(4)
                ] + ["validation_pred_5.json"]
                for position, name in enumerate(names):
                    key = (fold, seed, f"{position}_{name}")
                    result[key] = {
                        "outer_fold": fold,
                        "seed": seed,
                        "manifest": f"/{source}/{position}_{name}",
                        "source": source,
                    }
        return result

    main_records = records((0, 1, 2, 3, 4, 5), "main")
    retrain_records = records((3, 4), "retrain")
    merged_records = {
        key: (retrain_records[key] if key[0] in (3, 4) else value)
        for key, value in main_records.items()
    }
    main_path = tmp_path / "main.json"
    retrain_path = tmp_path / "retrain.json"
    merged_path = tmp_path / "merged.json"
    full_plan_path = tmp_path / "full_plan.json"
    retrain_plan_path = tmp_path / "retrain_plan.json"
    full_plan = _seal({"kind": "full-plan"})
    retrain_plan = _seal({"kind": "retrain-plan"})
    _write_json(full_plan_path, full_plan)
    _write_json(retrain_plan_path, retrain_plan)
    full_plan_binding = {
        "path": str(full_plan_path.resolve()),
        "sha256": AUDIT.sha256_file(full_plan_path),
        "bytes": full_plan_path.stat().st_size,
        "content_sha256": full_plan["content_sha256"],
    }
    retrain_plan_binding = {
        "path": str(retrain_plan_path.resolve()),
        "sha256": AUDIT.sha256_file(retrain_plan_path),
        "bytes": retrain_plan_path.stat().st_size,
        "content_sha256": retrain_plan["content_sha256"],
    }
    runtime_path = tmp_path / "execution_runtime_seal.json"
    runtime_path.write_text("runtime seal fixture", encoding="utf-8")
    supervisor_path = AUDIT.CANONICAL_EXECUTION_SUPERVISOR
    supervisor_binding = {
        "path": str(supervisor_path.resolve()),
        "sha256": AUDIT.sha256_file(supervisor_path),
        "bytes": supervisor_path.stat().st_size,
    }
    runtime_binding = {
        "path": str(runtime_path.resolve()),
        "sha256": "4" * 64,
        "bytes": runtime_path.stat().st_size,
        "content_sha256": "5" * 64,
        "verified_files": 1,
    }
    retrain_campaign_binding = {
        "path": str(retrain_path.resolve()),
        "sha256": "2" * 64,
        "bytes": 1,
        "content_sha256": "b" * 64,
    }
    receipt = _seal(
        {
            "schema_version": 1,
            "classification": "sealed_non_test_proposer_execution_attestation",
            "outer_test_opened": False,
            "outer_test_record_count": 0,
            "commercial_claim_authorized": False,
            "expected_units": 30,
            "completed_units": 30,
            "one_new_unit_per_invocation": True,
            "runtime_seal_verified_before_and_after_every_invocation": True,
            "invocations_this_resume": 30,
            "runtime_input_seal": runtime_binding,
            "campaign_index": retrain_campaign_binding,
            "unit_command": ["python", "runner.py"],
            "supervisor": supervisor_binding,
        }
    )
    receipt_path = tmp_path / "execution_attestation.json"
    _write_json(receipt_path, receipt)
    receipt_binding = {
        "path": str(receipt_path.resolve()),
        "sha256": AUDIT.sha256_file(receipt_path),
        "bytes": receipt_path.stat().st_size,
        "content_sha256": receipt["content_sha256"],
        "classification": receipt["classification"],
    }
    execution = {
        "required": True,
        "execution_attestation": receipt_binding,
        "authoritative_runtime_input_seal": {
            **runtime_binding,
            "attestation_phase": "prelaunch",
            "payloads_rehashed_during_merge": True,
        },
        "supervisor": supervisor_binding,
        "completion_evidence": {
            "expected_units": 30,
            "completed_units": 30,
            "campaign_index": retrain_campaign_binding,
            "one_new_unit_per_invocation": True,
            "runtime_seal_verified_before_and_after_every_invocation": True,
            "invocations_this_resume": 30,
            "single_supervisor_execution_covered_all_units": True,
        },
        "supersession_note": None,
        "attestation_content_verified": True,
        "campaign_index_30_of_30_verified": True,
        "runtime_seal_live_rehashed": True,
        "supervisor_live_rehashed": True,
        "canonical_supervisor_and_unit_command_verified": True,
    }
    paths = {
        main_path.resolve(): (
            {
                "content_sha256": "a" * 64,
                "campaign_plan": full_plan_binding,
            },
            {"path": str(main_path.resolve()), "sha256": "1" * 64, "bytes": 1},
            main_records,
        ),
        retrain_path.resolve(): (
            {
                "content_sha256": "b" * 64,
                "campaign_plan": retrain_plan_binding,
            },
            {"path": str(retrain_path.resolve()), "sha256": "2" * 64, "bytes": 1},
            retrain_records,
        ),
        merged_path.resolve(): (
            {
                "content_sha256": "c" * 64,
                "merge_classification": (
                    "retrospective_current_source_uniform_90_unit_proposer_index"
                ),
                "merge_provenance": {
                    "full_split_authority_plan": full_plan_binding,
                    "retrain_plan": retrain_plan_binding,
                    "source_indexes": {
                        "main": {
                            "path": str(main_path.resolve()),
                            "sha256": "1" * 64,
                            "bytes": 1,
                            "content_sha256": "a" * 64,
                            "selected_outer_folds": [0, 1, 2, 5],
                            "selected_units": 60,
                        },
                        "current_source_retrain_f34": {
                            "path": str(retrain_path.resolve()),
                            "sha256": "2" * 64,
                            "bytes": 1,
                            "content_sha256": "b" * 64,
                            "selected_outer_folds": [3, 4],
                            "selected_units": 30,
                        },
                    },
                    "retrain_execution_attestation_required": True,
                    "retrain_execution_attestation": receipt_binding,
                    "retrain_execution": execution,
                },
            },
            {"path": str(merged_path.resolve()), "sha256": "3" * 64, "bytes": 1},
            merged_records,
        ),
    }

    monkeypatch.setattr(
        AUDIT,
        "_validate_index",
        lambda path, **_: paths[path.resolve()],
    )
    def fake_verify_runtime(path: Path, expected_phase: str) -> dict:
        if expected_phase == "prelaunch":
            return {
                **runtime_binding,
                "attestation_phase": "prelaunch",
                "payloads_rehashed_during_this_audit": True,
            }
        return {
            "path": str(path),
            "sha256": "6" * 64,
            "bytes": 1,
            "content_sha256": "7" * 64,
            "attestation_phase": expected_phase,
            "verified_files": 1,
            "payloads_rehashed_during_this_audit": True,
        }

    monkeypatch.setattr(AUDIT, "_verify_runtime_seal", fake_verify_runtime)
    result = AUDIT.build_attestation(
        main_index_path=main_path,
        retrain_index_path=retrain_path,
        merged_index_path=merged_path,
        main_runtime_seal=tmp_path / "main_seal.json",
    )
    assert result["primary_plan_provenance_gap"]["selected_current_source_cover"] == {
        "main": 60,
        "current_source_retrain_f34": 30,
    }
    boundary = result["independent_code_path_audit"]
    assert boundary["literal_outer_test_label_file_unopened_claim_valid"] is False
    assert boundary["excluded_outer_metadata_target_columns_materialized_host_side"] is True
    assert boundary["outer_test_targets_consulted_for_model_fit_or_selection"] is False
    assert result["commercial_claim_authorized"] is False
    assert result["execution_provenance_complete"] is True
    assert result["execution_provenance"]["retrain_index_30_of_30_verified"] is True
    assert result["runtime_input_attestations"][
        "f3_f4_retrain_authoritative_prelaunch"
    ]["path"] == str(runtime_path.resolve())
    assert result["content_sha256"] == AUDIT.canonical_content_sha256(result)


def test_attestation_output_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "attestation.json"
    first = _seal({"classification": "audit", "commercial_claim_authorized": False})
    AUDIT.write_immutable(output, first)
    AUDIT.write_immutable(output, first)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        AUDIT.write_immutable(output, _seal({"classification": "other"}))


def test_execution_provenance_requires_receipt_and_required_true(tmp_path: Path) -> None:
    provenance, retrain_path, retrain_binding, content_hash = _execution_fixture(
        tmp_path
    )
    verified = AUDIT._verify_execution_provenance(
        provenance,
        retrain_index_path=retrain_path,
        retrain_index_binding=retrain_binding,
        retrain_index_content_sha256=content_hash,
    )
    assert verified["retrain_index_30_of_30_verified"] is True

    provenance["retrain_execution"]["required"] = False
    with pytest.raises(RuntimeError, match="require complete retrain execution evidence"):
        AUDIT._verify_execution_provenance(
            provenance,
            retrain_index_path=retrain_path,
            retrain_index_binding=retrain_binding,
            retrain_index_content_sha256=content_hash,
        )


def test_execution_provenance_rejects_partial_supervisor_resume(tmp_path: Path) -> None:
    provenance, retrain_path, retrain_binding, content_hash = _execution_fixture(
        tmp_path
    )
    provenance["retrain_execution"]["completion_evidence"][
        "invocations_this_resume"
    ] = 0
    provenance["retrain_execution"]["completion_evidence"][
        "single_supervisor_execution_covered_all_units"
    ] = False
    with pytest.raises(RuntimeError, match="completion evidence is not 30/30"):
        AUDIT._verify_execution_provenance(
            provenance,
            retrain_index_path=retrain_path,
            retrain_index_binding=retrain_binding,
            retrain_index_content_sha256=content_hash,
        )


def test_execution_provenance_rejects_missing_receipt_and_nested_content_drift(
    tmp_path: Path,
) -> None:
    provenance, retrain_path, retrain_binding, content_hash = _execution_fixture(
        tmp_path
    )
    receipt_path = Path(
        provenance["retrain_execution"]["execution_attestation"]["path"]
    )
    receipt_path.unlink()
    with pytest.raises(RuntimeError, match="file hash mismatch"):
        AUDIT._verify_execution_provenance(
            provenance,
            retrain_index_path=retrain_path,
            retrain_index_binding=retrain_binding,
            retrain_index_content_sha256=content_hash,
        )

    provenance, retrain_path, retrain_binding, content_hash = _execution_fixture(
        tmp_path / "second"
    )
    provenance["retrain_execution"]["execution_attestation"][
        "content_sha256"
    ] = "f" * 64
    provenance["retrain_execution_attestation"]["content_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="execution attestation live binding mismatch"):
        AUDIT._verify_execution_provenance(
            provenance,
            retrain_index_path=retrain_path,
            retrain_index_binding=retrain_binding,
            retrain_index_content_sha256=content_hash,
        )


def test_execution_provenance_live_rehashes_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = tmp_path / "canonical_supervisor.py"
    supervisor.write_text("# supervisor\n", encoding="utf-8")
    monkeypatch.setattr(AUDIT, "CANONICAL_EXECUTION_SUPERVISOR", supervisor)
    provenance, retrain_path, retrain_binding, content_hash = _execution_fixture(
        tmp_path
    )
    supervisor.write_text("# tampered supervisor\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="execution supervisor file hash mismatch"):
        AUDIT._verify_execution_provenance(
            provenance,
            retrain_index_path=retrain_path,
            retrain_index_binding=retrain_binding,
            retrain_index_content_sha256=content_hash,
        )
