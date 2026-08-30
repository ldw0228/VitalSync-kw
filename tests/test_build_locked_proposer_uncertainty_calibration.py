from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_locked_proposer_uncertainty_calibration.py"
SPEC = importlib.util.spec_from_file_location(
    "build_locked_proposer_uncertainty_calibration", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
CAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAL)


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document["content_sha256"] = CAL.canonical_content_sha256(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return document


def _unit(tmp_path: Path, position: int) -> dict[str, Any]:
    identities = [f"I{position}{sub}" for sub in range(3)]
    manifest_path = tmp_path / "manifests" / f"inner_pred_{position}.json"
    manifest = _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "identities": {
                "prediction": identities,
                "train": [f"T{position}"],
                "validation": [f"V{position}"],
                "excluded": ["OUT"],
            },
        },
    )
    checkpoint = tmp_path / "checkpoints" / f"unit_{position}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint-{position}".encode())
    checkpoint_binding = CAL.bind_file(checkpoint)
    manifest_binding = CAL.bind_file(manifest_path)

    rows = 240
    index = np.arange(position * 1000, position * 1000 + rows, dtype=np.int64)
    target = np.linspace(10.0, 30.0, rows, dtype=np.float32)
    error = np.where(index % 2 == 0, 0.5, -0.75).astype(np.float32)
    prediction_path = tmp_path / "predictions" / f"unit_{position}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        cache_index=index,
        session_id=np.asarray([f"S{position}"] * rows),
        identity=np.asarray([identities[row % 3] for row in range(rows)]),
        window_number=np.arange(rows, dtype=np.int32),
        reference_valid=np.ones(rows, dtype=bool),
        reference_rr_bpm=target,
        prediction=target + error,
        rr_std=np.full(rows, 0.5 + position * 0.1, dtype=np.float32),
        fold_id=np.asarray(position, dtype=np.int16),
        checkpoint_sha256=np.asarray(checkpoint_binding["sha256"]),
        split_manifest_file_sha256=np.asarray(manifest_binding["sha256"]),
        split_manifest_content_sha256=np.asarray(manifest["content_sha256"]),
        strict_retrospective=np.asarray(True),
        strict_nested_prediction_role=np.asarray(True),
    )
    return {
        "name": manifest_path.name,
        "role": "hcs_validation" if position == 4 else "hcs_train_oof",
        "manifest": manifest_binding,
        "checkpoint": checkpoint_binding,
        "all_window_prediction": CAL.bind_file(prediction_path),
    }


def _retrain_source_bindings(tmp_path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for position, name in enumerate(
        ("full_plan", "main_index", "retrain_plan", "retrain_index")
    ):
        path = tmp_path / "audit_sources" / f"{name}.json"
        document = _write_json(path, {"schema_version": 1, "kind": name})
        result[name] = {
            **CAL.bind_file(path),
            "content_sha256": document["content_sha256"],
        }
    return result


def _write_valid_audit(
    path: Path, sources: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    force = (
        "3:20260828,3:20260829,3:20260830,"
        "4:20260828,4:20260829,4:20260830"
    )
    return _write_json(
        path,
        {
            "schema_version": 1,
            "classification": "retrospective_nested_proposer_retrain_impact_audit",
            "commercial_claim_authorized": False,
            "outer_test_opened": False,
            "outer_test_record_count": 0,
            "target_or_reference_accessed": False,
            "source_campaigns_hash_complete": True,
            "source_plans_compatible": True,
            "inputs": sources,
            "plan_content_sha256": {
                "full": sources["full_plan"]["content_sha256"],
                "retrain": sources["retrain_plan"]["content_sha256"],
            },
            "comparison": {
                "comparison_units": 30,
                "changed_prediction_units": 30,
                "changed_checkpoint_units": 30,
                "force_retrain_unit_count": 6,
                "force_retrain_units_cli_value": force,
                "force_retrain_argument": ["--force-retrain-units", force],
                "checkpoint_change_alone_forces_hcs_retrain": False,
                "prediction_paths_ignored_after_bound_file_validation": True,
                "units": [{"position": position} for position in range(30)],
            },
        },
    )


def _governance_fixture(
    tmp_path: Path, index_binding: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    retrain_sources = _retrain_source_bindings(tmp_path)
    retrain_index = tmp_path / "retrain_index.json"
    retrain_index.write_text("retrain-index", encoding="utf-8")
    retrain_binding = {
        **CAL.bind_file(retrain_index),
        "content_sha256": "c" * 64,
    }
    runtime_seal = tmp_path / "execution_runtime_seal.json"
    runtime_seal.write_text("runtime-seal", encoding="utf-8")
    runtime_binding = {
        **CAL.bind_file(runtime_seal),
        "content_sha256": "d" * 64,
        "verified_files": 0,
    }
    supervisor = tmp_path / "supervisor.py"
    supervisor.write_text("# supervisor\n", encoding="utf-8")
    supervisor_binding = CAL.bind_file(supervisor)
    receipt_path = tmp_path / "execution_attestation.json"
    receipt = _write_json(
        receipt_path,
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
            "campaign_index": retrain_binding,
            "unit_command": ["python", "runner.py"],
            "supervisor": supervisor_binding,
        },
    )
    receipt_binding = {
        **CAL.bind_file(receipt_path),
        "content_sha256": receipt["content_sha256"],
        "classification": receipt["classification"],
    }
    selected_runtime = {
        **runtime_binding,
        "attestation_phase": "prelaunch",
        "payloads_rehashed_during_this_audit": True,
    }
    completion = {
        "expected_units": 30,
        "completed_units": 30,
        "campaign_index": retrain_binding,
        "one_new_unit_per_invocation": True,
        "runtime_seal_verified_before_and_after_every_invocation": True,
        "invocations_this_resume": 30,
        "single_supervisor_execution_covered_all_units": True,
    }
    governance_path = tmp_path / "governance.json"
    governance = _write_json(
        governance_path,
        {
            "classification": "retrospective_nested_proposer_governance_attestation",
            "commercial_claim_authorized": False,
            "prospective_confirmation_required": True,
            "outer_test_artifact_evaluated_during_non_test_campaigns": False,
            "target_metric_used_for_proposer_fit_or_selection": False,
            "verified_documents": {
                "merged_index": dict(index_binding),
                "retrain_execution_attestation": receipt_binding,
                "retrain_execution_supervisor": supervisor_binding,
            },
            "verified_retrain_impact_sources": retrain_sources,
            "runtime_input_attestations": {
                "f3_f4_retrain_authoritative_prelaunch": selected_runtime,
            },
            "execution_provenance_complete": True,
            "execution_provenance": {
                "required": True,
                "execution_attestation": receipt_binding,
                "authoritative_runtime_input_seal": selected_runtime,
                "supervisor": supervisor_binding,
                "completion_evidence": completion,
                "supersession_note": None,
                "merged_nested_binding_verified": True,
                "execution_attestation_live_rehashed": True,
                "authoritative_runtime_seal_live_rehashed": True,
                "supervisor_live_rehashed": True,
                "retrain_index_30_of_30_verified": True,
                "canonical_supervisor_and_unit_command_verified": True,
            },
        },
    )
    return governance_path, {
        "governance": governance,
        "runtime_binding": runtime_binding,
        "retrain_sources": retrain_sources,
    }


def test_finite_sample_higher_quantile_uses_ceil_n_plus_one_rank() -> None:
    values = np.arange(1.0, 11.0)
    assert CAL._higher_quantile(values, 0.50) == 6.0
    assert CAL._higher_quantile(values, 0.90) == 10.0
    assert CAL._higher_quantile(values, 0.95) == 10.0


def test_group_calibration_is_five_way_identity_disjoint_and_phase_fixed(
    tmp_path: Path,
) -> None:
    units = [_unit(tmp_path, position) for position in range(5)]
    result = CAL._calibrate_group(units, outer_fold=2, seed=CAL.SEEDS[0])
    assert result["source_unit_count"] == 5
    assert result["source_identity_count"] == 15
    assert result["source_rows_all"] == 1200
    assert result["source_rows_valid_phase_0"] == 150
    assert set(result["interval_calibration"]) == {"0.50", "0.80", "0.90", "0.95"}
    assert result["interval_calibration"]["0.90"]["calibration_empirical_coverage"] >= 0.9


def test_group_rejects_prediction_identity_overlap_with_training(tmp_path: Path) -> None:
    units = [_unit(tmp_path, position) for position in range(5)]
    path = Path(units[0]["manifest"]["path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("content_sha256")
    manifest["identities"]["train"] = [manifest["identities"]["prediction"][0]]
    changed = _write_json(path, manifest)
    units[0]["manifest"] = CAL.bind_file(path)
    with np.load(units[0]["all_window_prediction"]["path"], allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["split_manifest_file_sha256"] = np.asarray(units[0]["manifest"]["sha256"])
    arrays["split_manifest_content_sha256"] = np.asarray(changed["content_sha256"])
    np.savez_compressed(units[0]["all_window_prediction"]["path"], **arrays)
    units[0]["all_window_prediction"] = CAL.bind_file(
        Path(units[0]["all_window_prediction"]["path"])
    )
    with pytest.raises(CAL.CalibrationBuildError, match="ownership"):
        CAL._calibrate_group(units, outer_fold=0, seed=CAL.SEEDS[0])


def test_full_build_freezes_all_18_rules_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    units = [_unit(tmp_path, position) for position in range(5)]
    plan_path = tmp_path / "plan.json"
    index_path = tmp_path / "index.json"
    _write_json(plan_path, {"kind": "plan"})
    _write_json(index_path, {"kind": "index"})
    source_bindings = {
        "plan": {**CAL.bind_file(plan_path), "content_sha256": "a" * 64},
        "index": {**CAL.bind_file(index_path), "content_sha256": "b" * 64},
    }
    groups = {
        (fold, seed): {"status": "ready", "units": units}
        for fold in CAL.FOLDS
        for seed in CAL.SEEDS
    }
    fake_fixed = SimpleNamespace(
        validate_non_test_plan_index=lambda _plan, _index: (groups, source_bindings)
    )
    monkeypatch.setattr(CAL, "_load_fixed_runner_module", lambda: fake_fixed)
    monkeypatch.setattr(CAL, "PROJECT_ROOT", tmp_path)

    governance_path, governance_fixture = _governance_fixture(
        tmp_path, source_bindings["index"]
    )
    audit_path = tmp_path / "audit.json"
    _write_valid_audit(audit_path, governance_fixture["retrain_sources"])
    monkeypatch.setattr(
        CAL,
        "_load_runtime_seal_module",
        lambda: SimpleNamespace(
            verify=lambda _path: {
                "content_sha256": governance_fixture["runtime_binding"][
                    "content_sha256"
                ],
                "verified_files": 0,
            }
        ),
    )
    output = tmp_path / "calibration.json"
    first = CAL.build_calibration(
        plan_path=plan_path,
        index_path=index_path,
        retrain_audit_path=audit_path,
        governance_path=governance_path,
        output_path=output,
    )
    second = CAL.build_calibration(
        plan_path=plan_path,
        index_path=index_path,
        retrain_audit_path=audit_path,
        governance_path=governance_path,
        output_path=output,
    )
    assert first == second
    assert first["unit_count"] == 18
    assert first["point_prediction_modified"] is False
    assert first["outer_test_opened"] is False
    assert first["content_sha256"] == CAL.canonical_content_sha256(first)
    assert output.stat().st_mode & 0o777 == 0o444


def test_full_build_refuses_after_target_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CAL, "PROJECT_ROOT", tmp_path)
    lock = (
        tmp_path
        / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof/evaluation_lock.json"
    )
    lock.parent.mkdir(parents=True)
    lock.write_text("{}", encoding="utf-8")
    with pytest.raises(CAL.CalibrationBuildError, match="before outer-test target join"):
        CAL.build_calibration(
            plan_path=tmp_path / "missing-plan",
            index_path=tmp_path / "missing-index",
            retrain_audit_path=tmp_path / "missing-audit",
            governance_path=tmp_path / "missing-governance",
            output_path=tmp_path / "output",
        )


def test_attestation_validation_rejects_governance_with_digest_string_only(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text("index", encoding="utf-8")
    index_binding = {
        **CAL.bind_file(index_path),
        "content_sha256": "b" * 64,
    }
    audit_path = tmp_path / "audit.json"
    _write_valid_audit(audit_path, _retrain_source_bindings(tmp_path))
    governance_path = tmp_path / "governance.json"
    _write_json(
        governance_path,
        {
            "classification": "retrospective_nested_proposer_governance_attestation",
            "commercial_claim_authorized": False,
            "prospective_confirmation_required": True,
            "outer_test_artifact_evaluated_during_non_test_campaigns": False,
            "target_metric_used_for_proposer_fit_or_selection": False,
            "unstructured_digest_string": index_binding["sha256"],
        },
    )
    with pytest.raises(CAL.CalibrationBuildError, match="execution provenance is incomplete"):
        CAL._validate_attestations(audit_path, governance_path, index_binding)


def test_attestation_validation_rejects_missing_execution_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text("index", encoding="utf-8")
    index_binding = {
        **CAL.bind_file(index_path),
        "content_sha256": "b" * 64,
    }
    governance_path, fixture = _governance_fixture(tmp_path, index_binding)
    audit_path = tmp_path / "audit.json"
    _write_valid_audit(audit_path, fixture["retrain_sources"])
    monkeypatch.setattr(
        CAL,
        "_load_runtime_seal_module",
        lambda: SimpleNamespace(
            verify=lambda _path: {
                "content_sha256": fixture["runtime_binding"]["content_sha256"],
                "verified_files": 0,
            }
        ),
    )
    receipt = Path(
        fixture["governance"]["execution_provenance"]["execution_attestation"][
            "path"
        ]
    )
    receipt.unlink()
    with pytest.raises(CAL.CalibrationBuildError, match="regular non-symlink file"):
        CAL._validate_attestations(audit_path, governance_path, index_binding)


def test_attestation_validation_rejects_wrong_structured_merged_binding(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text("index", encoding="utf-8")
    index_binding = {
        **CAL.bind_file(index_path),
        "content_sha256": "b" * 64,
    }
    governance_path, fixture = _governance_fixture(tmp_path, index_binding)
    audit_path = tmp_path / "audit.json"
    _write_valid_audit(audit_path, fixture["retrain_sources"])
    governance = fixture["governance"]
    governance["verified_documents"]["merged_index"]["sha256"] = "e" * 64
    _write_json(governance_path, governance)
    with pytest.raises(CAL.CalibrationBuildError, match="merged index binding mismatch"):
        CAL._validate_attestations(audit_path, governance_path, index_binding)


def test_attestation_validation_rejects_minimal_self_declared_audit(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text("index", encoding="utf-8")
    index_binding = {
        **CAL.bind_file(index_path),
        "content_sha256": "b" * 64,
    }
    governance_path, _ = _governance_fixture(tmp_path, index_binding)
    audit_path = tmp_path / "minimal_audit.json"
    _write_json(
        audit_path,
        {
            "classification": "retrospective_nested_proposer_retrain_impact_audit",
            "outer_test_opened": False,
            "target_or_reference_accessed": False,
            "source_campaigns_hash_complete": True,
        },
    )
    with pytest.raises(CAL.CalibrationBuildError, match="not a sealed non-test audit"):
        CAL._validate_attestations(audit_path, governance_path, index_binding)


def test_attestation_validation_rejects_identical_bytes_from_another_source_path(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text("index", encoding="utf-8")
    index_binding = {
        **CAL.bind_file(index_path),
        "content_sha256": "b" * 64,
    }
    governance_path, fixture = _governance_fixture(tmp_path, index_binding)
    audit_path = tmp_path / "audit.json"
    audit = _write_valid_audit(audit_path, fixture["retrain_sources"])
    original = Path(audit["inputs"]["main_index"]["path"])
    copied = tmp_path / "copied_main_index.json"
    copied.write_bytes(original.read_bytes())
    audit.pop("content_sha256")
    audit["inputs"]["main_index"] = {
        **CAL.bind_file(copied),
        "content_sha256": audit["inputs"]["main_index"]["content_sha256"],
    }
    _write_json(audit_path, audit)
    with pytest.raises(CAL.CalibrationBuildError, match="source main_index binding mismatch"):
        CAL._validate_attestations(audit_path, governance_path, index_binding)


def test_attestation_validation_live_rehashes_all_audit_sources(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text("index", encoding="utf-8")
    index_binding = {
        **CAL.bind_file(index_path),
        "content_sha256": "b" * 64,
    }
    governance_path, fixture = _governance_fixture(tmp_path, index_binding)
    audit_path = tmp_path / "audit.json"
    _write_valid_audit(audit_path, fixture["retrain_sources"])
    full_plan = Path(fixture["retrain_sources"]["full_plan"]["path"])
    full_plan.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(
        CAL.CalibrationBuildError,
        match="live retrain impact source full_plan binding mismatch",
    ):
        CAL._validate_attestations(audit_path, governance_path, index_binding)
