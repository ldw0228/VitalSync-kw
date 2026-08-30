from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/create_locked_hcs_pretarget_release_lock.py"
)
SPEC = importlib.util.spec_from_file_location(
    "create_locked_hcs_pretarget_release_lock", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
LOCK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOCK)


def _binding(name: str) -> dict[str, Any]:
    return {"path": f"/{name}", "sha256": name[0] * 64, "bytes": len(name)}


def _write_attestation(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document["content_sha256"] = LOCK.canonical_json_sha256(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return document


def _patch_target_free_graph(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    primary = {
        "classification": "primary_18_unit_predictions_revalidated",
        "unit_count": 18,
        "predictions_seal": _binding("primary"),
    }

    def primary_summary(path: Path) -> dict[str, Any]:
        events.append("primary")
        return primary

    def mask_summary(path: Path, observed: dict[str, Any]) -> dict[str, Any]:
        events.append("mask")
        assert observed is primary
        return {
            "classification": "all_126_radar_mask_units_revalidated",
            "unit_count": 126,
            "full_mask_parity_unit_count": 18,
        }

    def uncertainty_summary(path: Path, observed: dict[str, Any]) -> dict[str, Any]:
        events.append("uncertainty")
        assert observed is primary
        return {
            "classification": "uncertainty_18_unit_inputs_and_pretest_calibration_revalidated",
            "unit_count": 18,
            "pretest_calibration": _binding("calibration"),
            "pretest_calibration_content_sha256": "f" * 64,
        }

    def specs(
        eval_path: Path,
        uncertainty_eval_path: Path,
        deployment_path: Path,
        readiness_path: Path,
        calibration_binding: dict[str, Any],
        calibration_content_sha256: str,
    ) -> dict[str, Any]:
        events.append("specs")
        return {
            "primary_evaluation": {"binding": _binding("evaluation")},
            "secondary_uncertainty_evaluation": {
                "binding": _binding("uncertainty_evaluation"),
                "calibration": calibration_binding,
                "primary_diagnostic_only_contract_preserved": True,
            },
            "deployment_benchmark": {"binding": _binding("deployment")},
            "release_readiness": {
                "binding": _binding("readiness"),
                "content_sha256": "9" * 64,
                "target_or_target_bearing_artifact_opened": False,
                "commercial_release_ready_must_equal": False,
                "prospective_confirmation_required": True,
            },
        }

    monkeypatch.setattr(LOCK, "_primary_summary", primary_summary)
    monkeypatch.setattr(LOCK, "_mask_summary", mask_summary)
    monkeypatch.setattr(LOCK, "_uncertainty_summary", uncertainty_summary)
    monkeypatch.setattr(
        LOCK,
        "_runtime_guard_summary",
        lambda runtime, completion, postlock, mask_guard, observed_primary, observed_masks, **kwargs: (
            events.append("runtime")
            or {
                "classification": "fixed_i3_and_postlock_runtime_payload_closure_revalidated",
                "completed_primary_units": 18,
                "completed_radar_mask_units": 126,
            }
        ),
    )
    monkeypatch.setattr(LOCK, "_spec_summary", specs)
    monkeypatch.setattr(
        LOCK,
        "_source_bindings",
        lambda: {"creator": _binding("creator"), "wrapper": _binding("wrapper")},
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "primary_root": tmp_path / "primary",
        "mask_root": tmp_path / "masks",
        "uncertainty_seal": tmp_path / "uncertainty.json",
        "evaluation_spec": tmp_path / "evaluation.json",
        "uncertainty_evaluation_spec": tmp_path / "uncertainty_evaluation.json",
        "deployment_spec": tmp_path / "deployment.json",
        "release_readiness_spec": tmp_path / "readiness.json",
        "output": tmp_path / "pretarget_release_lock.json",
        "target_output": tmp_path / "canonical_targets.npz",
        "target_receipt": tmp_path / "canonical_targets_receipt.json",
        "evaluation_lock": tmp_path / "evaluation_lock.json",
        "joined_output": tmp_path / "joined.npz",
        "release_receipt": tmp_path / "release_receipt.json",
        "fixed_i3_runtime_seal": tmp_path / "fixed_runtime_seal.json",
        "fixed_runtime_completion": tmp_path / "fixed_completion.json",
        "postlock_runtime_guard": tmp_path / "postlock_guard.json",
        "radar_mask_runtime_guard": tmp_path / "mask_guard.json",
    }


def test_create_lock_revalidates_three_boundaries_and_specs_before_0444_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_target_free_graph(monkeypatch, events)
    paths = _paths(tmp_path)
    result = LOCK.create_release_lock(**paths)
    assert events == [
        "primary",
        "mask",
        "uncertainty",
        "runtime",
        "specs",
        "primary",
        "mask",
        "uncertainty",
        "runtime",
        "specs",
    ]
    assert result["classification"] == "locked_hcs_pretarget_release_lock"
    assert result["canonical_target_build_authorized"] is True
    assert result["target_or_label_artifact_opened"] is False
    assert result["content_sha256"] == LOCK.canonical_json_sha256(
        {key: value for key, value in result.items() if key != "content_sha256"}
    )
    assert paths["output"].stat().st_mode & 0o777 == 0o444


def test_create_lock_is_idempotent_only_while_target_boundary_remains_unopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_target_free_graph(monkeypatch, events)
    paths = _paths(tmp_path)
    first = LOCK.create_release_lock(**paths)
    second = LOCK.create_release_lock(**paths)
    assert first == second
    paths["target_output"].write_bytes(b"opened")
    with pytest.raises(LOCK.PretargetReleaseLockError, match="after a target"):
        LOCK.create_release_lock(**paths)


def test_existing_target_is_rejected_before_any_validator_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paths["target_receipt"].write_text("target-bearing", encoding="utf-8")
    called = False

    def forbidden(**kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("deep validators must not run")

    monkeypatch.setattr(LOCK, "_release_document", forbidden)
    with pytest.raises(LOCK.PretargetReleaseLockError, match="after a target"):
        LOCK.create_release_lock(**paths)
    assert called is False
    assert not paths["output"].exists()


def test_target_appearing_after_validation_prevents_lock_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)

    def racing_document(**kwargs: Any) -> dict[str, Any]:
        paths["target_output"].write_bytes(b"raced")
        return {
            "schema_version": 1,
            "classification": "locked_hcs_pretarget_release_lock",
            "content_sha256": "0" * 64,
        }

    monkeypatch.setattr(LOCK, "_release_document", racing_document)
    with pytest.raises(LOCK.PretargetReleaseLockError, match="after a target"):
        LOCK.create_release_lock(**paths)
    assert not paths["output"].exists()


def test_validate_rejects_content_tamper_even_if_mode_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_target_free_graph(monkeypatch, events)
    paths = _paths(tmp_path)
    LOCK.create_release_lock(**paths)
    paths["output"].chmod(0o644)
    document = json.loads(paths["output"].read_text(encoding="utf-8"))
    document["status"] = "tampered"
    paths["output"].write_text(json.dumps(document), encoding="utf-8")
    paths["output"].chmod(0o444)
    with pytest.raises(LOCK.PretargetReleaseLockError, match="content hash"):
        LOCK.validate_release_lock(paths["output"])


def _patch_runtime_producer_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    paths = {
        name: tmp_path / name
        for name in (
            "runtime.json",
            "completion.json",
            "pretest_index.json",
            "postlock.json",
            "oof_plan.json",
            "pretest_lock.json",
            "predictions_seal.json",
            "mask_guard.json",
            "mask_plan.json",
            "complete_seal.json",
        )
    }
    runtime_document = {
        "schema_version": 1,
        "classification": "supplemental_runtime_input_byte_inventory",
        "attestation_phase": "prelaunch",
        "post_launch_attestation": False,
        "content_sha256": "a" * 64,
        "input_trees": [{"root": "/rf"}, {"root": "/svd"}],
        "fixed_i3_context": {
            "classification": "retrospective_fixed_i3_pretest_runtime_input_context",
            "outer_test_opened": False,
            "target_or_evaluation_artifact_accessed": False,
            "proposer_matrix_groups": 18,
            "proposer_matrix_units": 90,
        },
    }
    for name, path in paths.items():
        path.write_text(
            json.dumps(runtime_document if name == "runtime.json" else {"name": name}),
            encoding="utf-8",
        )
    binding = {name: LOCK.bind_file(path) for name, path in paths.items()}
    oof_order = [(fold, seed) for seed in (101, 102, 103) for fold in range(6)]
    oof_receipts = [
        {
            "path": str((tmp_path / f"oof_receipt_{position:03d}.json").resolve()),
            "sha256": f"{position:064x}",
            "bytes": position,
        }
        for position in range(1, 19)
    ]
    mask_units = [{"unit_id": f"mask-unit-{position}"} for position in range(1, 127)]
    mask_receipts = [
        {
            "path": str((tmp_path / f"mask_receipt_{position:03d}.json").resolve()),
            "sha256": f"{position + 1000:064x}",
            "bytes": position + 1000,
        }
        for position in range(1, 127)
    ]
    completion_document = {
        "content_sha256": "b" * 64,
        "commercial_claim_authorized": False,
        "pretest_index": binding["pretest_index.json"],
        "unit_payloads": [{"position": position} for position in range(1, 19)],
        "fixed_campaign_execution_evidence": {
            "completed_prefixes_revalidated": list(range(1, 19)),
            "status_snapshot_count": 18,
            "status_snapshot_graph_sha256": "c" * 64,
        },
    }
    closure = {
        "runtime_input_seal": binding["runtime.json"],
        "runtime_content_sha256": "a" * 64,
        "completion_attestation": binding["completion.json"],
        "completion_content_sha256": "b" * 64,
        "pretest_index": binding["pretest_index.json"],
    }
    guard_document = {
        "content_sha256": "d" * 64,
        "commercial_claim_authorized": False,
        "locked_oof_plan": binding["oof_plan.json"],
        "pretest_lock": binding["pretest_lock.json"],
        "predictions_seal": binding["predictions_seal.json"],
        "unit_runtime_guard_receipts": oof_receipts,
    }
    mask_document = {
        "schema_version": 1,
        "classification": "locked_hcs_radar_mask_runtime_guard_attestation",
        "content_sha256": "e" * 64,
        "commercial_claim_authorized": False,
        "radar_mask_plan": binding["mask_plan.json"],
        "complete_seal": binding["complete_seal.json"],
        "unit_runtime_guard_receipts": mask_receipts,
    }
    state: dict[str, Any] = {
        "calls": [],
        "missing_receipt": None,
        "tampered_oof_receipt": None,
        "drifted_mask_output": None,
    }

    monkeypatch.setattr(
        LOCK.RUNTIME_SEAL,
        "verify",
        lambda path: {"content_sha256": "a" * 64, "verified_files": 321},
    )

    def fixed_verify(path: Path, **kwargs: Any) -> dict[str, Any]:
        state["calls"].append(("fixed", kwargs))
        if state.get("fixed_failure"):
            raise RuntimeError("status prefix chain missing")
        return {"document": completion_document, "binding": binding["completion.json"]}

    def verify_binding(raw: Any, **kwargs: Any) -> dict[str, Any]:
        if raw == state["missing_receipt"]:
            raise RuntimeError("receipt binding missing")
        return dict(raw)

    monkeypatch.setattr(
        LOCK.FIXED_COMPLETION, "verify_completion_attestation", fixed_verify
    )
    monkeypatch.setattr(LOCK.FIXED_COMPLETION, "verify_binding", verify_binding)
    monkeypatch.setattr(LOCK.OOF_RUNTIME_GUARD, "verify_closure", lambda **kwargs: closure)

    def verify_oof(path: Path, **kwargs: Any) -> dict[str, Any]:
        state["calls"].append(("oof", kwargs))
        return {"document": guard_document, "binding": binding["postlock.json"]}

    monkeypatch.setattr(LOCK.OOF_RUNTIME_GUARD, "verify_guard_attestation", verify_oof)
    monkeypatch.setattr(
        LOCK.OOF_RUNTIME_GUARD,
        "_validated_plan",
        lambda path, root: {"units": [{"outer_fold": f, "seed": s} for f, s in oof_order]},
    )
    monkeypatch.setattr(
        LOCK.OOF_RUNTIME_GUARD, "_ordered_units", lambda plan: list(oof_order)
    )

    def validate_oof_receipt(path: Path, **kwargs: Any) -> dict[str, Any]:
        position = int(kwargs["position"])
        if state["tampered_oof_receipt"] == position:
            raise RuntimeError("receipt content tampered")
        return {
            "content_sha256": f"{position + 2000:064x}",
            "derived_lock": {"path": f"/derived/{position}", "sha256": "1" * 64, "bytes": 1},
            "prediction": {"path": f"/prediction/{position}", "sha256": "2" * 64, "bytes": 2},
        }

    monkeypatch.setattr(
        LOCK.OOF_RUNTIME_GUARD, "_validate_unit_receipt", validate_oof_receipt
    )

    def verify_radar(path: Path, **kwargs: Any) -> dict[str, Any]:
        state["calls"].append(("radar", kwargs))
        return {"document": mask_document, "binding": binding["mask_guard.json"]}

    monkeypatch.setattr(
        LOCK.RADAR_RUNTIME_GUARD, "verify_radar_guard_attestation", verify_radar
    )
    monkeypatch.setattr(
        LOCK.RADAR_RUNTIME_GUARD,
        "_validated_plan",
        lambda path, root: {"units": mask_units},
    )

    def validate_mask_receipt(path: Path, **kwargs: Any) -> dict[str, Any]:
        position = int(kwargs["position"])
        if state["drifted_mask_output"] == position:
            raise RuntimeError("live sealed output binding drift")
        return {
            "content_sha256": f"{position + 3000:064x}",
            "outputs": {
                "sealed_prediction": {
                    "path": f"/mask/{position}",
                    "sha256": "3" * 64,
                    "bytes": 3,
                }
            },
        }

    monkeypatch.setattr(
        LOCK.RADAR_RUNTIME_GUARD, "_validate_receipt", validate_mask_receipt
    )
    primary = {
        "inference_plan": binding["oof_plan.json"],
        "pretest_lock": binding["pretest_lock.json"],
        "predictions_seal": binding["predictions_seal.json"],
    }
    masks = {"complete_seal": binding["complete_seal.json"]}
    arguments = (
        paths["runtime.json"],
        paths["completion.json"],
        paths["postlock.json"],
        paths["mask_guard.json"],
        primary,
        masks,
    )
    return state, arguments


def test_runtime_payload_closure_replays_all_three_producer_verifiers_and_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, arguments = _patch_runtime_producer_graph(tmp_path, monkeypatch)
    result = LOCK._runtime_guard_summary(*arguments)
    assert [name for name, _ in state["calls"]] == ["fixed", "oof", "radar"]
    assert state["calls"][0][1]["reverify_payload"] is True
    assert state["calls"][1][1]["reverify_closure"] is True
    assert result["fixed_completed_prefixes_revalidated"] == list(range(1, 19))
    assert result["postlock_live_receipt_count"] == 18
    assert result["radar_mask_live_receipt_count"] == 126


def test_fixed_status_prefix_chain_failure_is_pretarget_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, arguments = _patch_runtime_producer_graph(tmp_path, monkeypatch)
    state["fixed_failure"] = True
    target = tmp_path / "canonical_targets.npz"
    with pytest.raises(LOCK.PretargetReleaseLockError, match="producer runtime verifier"):
        LOCK._runtime_guard_summary(*arguments)
    assert not target.exists()


@pytest.mark.parametrize("failure", ["missing_receipt", "tampered_oof_receipt"])
def test_oof_receipt_missing_or_content_tamper_is_pretarget_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    state, arguments = _patch_runtime_producer_graph(tmp_path, monkeypatch)
    if failure == "missing_receipt":
        state[failure] = {
            "path": str((tmp_path / "oof_receipt_007.json").resolve()),
            "sha256": f"{7:064x}",
            "bytes": 7,
        }
    else:
        state[failure] = 7
    target = tmp_path / "canonical_targets.npz"
    with pytest.raises(LOCK.PretargetReleaseLockError, match="18-unit live receipt"):
        LOCK._runtime_guard_summary(*arguments)
    assert not target.exists()


def test_radar_live_output_binding_drift_is_pretarget_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, arguments = _patch_runtime_producer_graph(tmp_path, monkeypatch)
    state["drifted_mask_output"] = 91
    target = tmp_path / "canonical_targets.npz"
    with pytest.raises(LOCK.PretargetReleaseLockError, match="126-unit live receipt"):
        LOCK._runtime_guard_summary(*arguments)
    assert not target.exists()


def test_release_lock_binds_all_three_producer_verifier_sources() -> None:
    sources = LOCK._source_bindings()
    assert {
        "fixed_completion_producer_verifier",
        "locked_oof_runtime_guard_producer_verifier",
        "radar_mask_runtime_guard_producer_verifier",
    } <= set(sources)


def test_dedicated_uncertainty_spec_is_required_separate_and_calibration_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary_path = tmp_path / "primary.json"
    secondary_path = tmp_path / "secondary.json"
    deployment_path = tmp_path / "deployment.json"
    readiness_path = tmp_path / "readiness.json"
    calibration_path = tmp_path / "calibration.json"
    primary_binding = {"path": str(primary_path.resolve()), "sha256": "a" * 64, "bytes": 1}
    secondary_binding = {"path": str(secondary_path.resolve()), "sha256": "b" * 64, "bytes": 2}
    calibration_binding = {
        "path": str(calibration_path.resolve()),
        "sha256": "c" * 64,
        "bytes": 3,
    }
    primary = {
        "content_sha256": "d" * 64,
        "uncertainty": {
            "role": "diagnostic_ranking_only_not_calibrated_interval",
            "calibration_fit_allowed": False,
            "threshold_fit_allowed": False,
            "model_or_candidate_selection_allowed": False,
        },
        "post_target_prohibitions": {"calibration_or_uncertainty_fit": True},
    }
    relationship = {
        "role": "separate_secondary_retrospective_engineering_protocol",
        "primary_uncertainty_contract_overridden": False,
        "primary_point_evaluation_or_gates_modified": False,
        "primary_diagnostic_only_uncertainty_claim_preserved": True,
        "secondary_interval_results_are_part_of_primary_evaluation": False,
    }
    secondary = {
        "content_sha256": "e" * 64,
        "calibration_content_sha256": "f" * 64,
        "protocol_relationship": relationship,
        "bound_inputs": {
            "unchanged_primary_evaluation_spec": primary_binding,
            "completed_pretest_uncertainty_calibration": calibration_binding,
        },
    }
    readiness = {
        "classification": "locked_hcs_release_readiness_aggregate_specification",
        "content_sha256": "9" * 64,
        "target_or_target_bearing_artifact_opened": False,
        "prospective_release_policy": {
            "independent_prospective_cohort_present": False,
            "commercial_release_ready_must_equal": False,
        },
    }
    readiness_binding = {
        "path": str(readiness_path.resolve()),
        "sha256": "8" * 64,
        "bytes": 5,
    }
    calls: list[tuple[Path, Path, Path]] = []
    monkeypatch.setattr(
        LOCK.PRIMARY_EVALUATION,
        "_load_evaluation_spec",
        lambda path: (primary, primary_binding),
    )

    def load_secondary(
        path: Path, *, expected_primary_spec_path: Path, expected_calibration_path: Path
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append((path, expected_primary_spec_path, expected_calibration_path))
        return secondary, secondary_binding

    monkeypatch.setattr(
        LOCK.UNCERTAINTY_SPEC, "load_uncertainty_evaluation_spec", load_secondary
    )
    monkeypatch.setattr(
        LOCK.DEPLOYMENT,
        "load_freeze_spec",
        lambda path: (
            {"content_sha256": "1" * 64},
            {"path": str(path.resolve()), "sha256": "2" * 64, "bytes": 4},
        ),
    )
    monkeypatch.setattr(
        LOCK.RELEASE_READINESS,
        "load_release_readiness_spec",
        lambda path: (readiness, readiness_binding),
    )
    result = LOCK._spec_summary(
        primary_path,
        secondary_path,
        deployment_path,
        readiness_path,
        calibration_binding,
        "f" * 64,
    )
    assert calls == [(secondary_path, primary_path, calibration_path.resolve())]
    assert result["secondary_uncertainty_evaluation"][
        "primary_diagnostic_only_contract_preserved"
    ] is True
    assert result["secondary_uncertainty_evaluation"]["calibration"] == calibration_binding
    assert result["release_readiness"] == {
        "binding": readiness_binding,
        "content_sha256": "9" * 64,
        "target_or_target_bearing_artifact_opened": False,
        "commercial_release_ready_must_equal": False,
        "prospective_confirmation_required": True,
    }

    secondary["protocol_relationship"] = {
        **relationship,
        "primary_uncertainty_contract_overridden": True,
    }
    with pytest.raises(LOCK.PretargetReleaseLockError, match="diagnostic-only"):
        LOCK._spec_summary(
            primary_path,
            secondary_path,
            deployment_path,
            readiness_path,
            calibration_binding,
            "f" * 64,
        )

    secondary["protocol_relationship"] = relationship
    readiness["prospective_release_policy"]["commercial_release_ready_must_equal"] = True
    with pytest.raises(LOCK.PretargetReleaseLockError, match="prospective fail-closed"):
        LOCK._spec_summary(
            primary_path,
            secondary_path,
            deployment_path,
            readiness_path,
            calibration_binding,
            "f" * 64,
        )
