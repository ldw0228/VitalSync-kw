from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import sys
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_locked_hcs_release_readiness.py"
SPEC = importlib.util.spec_from_file_location("evaluate_locked_hcs_release_readiness", SCRIPT)
assert SPEC and SPEC.loader
RR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RR
SPEC.loader.exec_module(RR)
RR.SOURCE_PATHS["release_evaluation_orchestrator"] = (
    SCRIPT.parent / "run_release_locked_hcs_evaluation.py"
)


def _doc(path: Path, value: dict[str, Any], *, content: bool = True) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = RR._content_document(value) if content else dict(value)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    return document


def test_runtime_inventory_content_hash_excludes_only_created_utc() -> None:
    runtime = {
        "classification": "supplemental_runtime_input_byte_inventory",
        "created_utc": "2026-08-28T00:00:00+00:00",
        "sources": [],
    }
    runtime["content_sha256"] = RR.canonical_sha256(
        {"classification": runtime["classification"], "sources": []}
    )
    RR._validate_content(runtime, "runtime inventory")

    tampered = dict(runtime)
    tampered["sources"] = [{"path": "forged"}]
    with pytest.raises(RR.ReleaseReadinessError, match="content hash mismatch"):
        RR._validate_content(tampered, "runtime inventory")

    ordinary = {
        "classification": "ordinary_document",
        "created_utc": runtime["created_utc"],
        "sources": [],
        "content_sha256": runtime["content_sha256"],
    }
    with pytest.raises(RR.ReleaseReadinessError, match="content hash mismatch"):
        RR._validate_content(ordinary, "ordinary document")


def test_runtime_supersession_closes_selected_but_not_failed_payload_trees(
    tmp_path: Path,
) -> None:
    failed_source = tmp_path / "failed_source.py"
    failed_source.write_text("old failed source\n", encoding="utf-8")
    failed_seal_path = tmp_path / "failed_seal.json"
    failed_seal = _doc(
        failed_seal_path,
        {"classification": "supplemental_runtime_input_byte_inventory",
         "source": RR.bind_file(failed_source)},
    )
    del failed_seal

    selected_source = tmp_path / "selected_source.py"
    selected_source.write_text("selected source\n", encoding="utf-8")
    selected_seal_path = tmp_path / "selected_seal.json"
    _doc(
        selected_seal_path,
        {"classification": "selected_runtime_seal",
         "source": RR.bind_file(selected_source)},
    )
    supersession = RR._content_document(
        {
            "classification": "non_test_proposer_execution_runtime_seal_supersession",
            "selected_runtime_seal": RR.bind_file(selected_seal_path),
            "superseded_runtime_seals": [RR.bind_file(failed_seal_path)],
        }
    )

    failed_source.write_text("later corrected source\n", encoding="utf-8")
    RR._verify_closure(supersession, label="supersession")

    selected_source.write_text("tampered selected source\n", encoding="utf-8")
    with pytest.raises(RR.ReleaseReadinessError, match="binding changed"):
        RR._verify_closure(supersession, label="supersession")


def test_only_canonical_venv_python_launcher_symlink_is_bindable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / "python-real"
    interpreter.write_bytes(b"interpreter bytes")
    launcher = tmp_path / ".venv/bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(interpreter)
    monkeypatch.setattr(RR, "PROJECT_ROOT", tmp_path)
    binding = {
        "path": str(launcher),
        "sha256": RR.sha256_file(interpreter),
        "bytes": interpreter.stat().st_size,
    }
    assert RR._verify_binding(binding, label="python executable") == binding

    arbitrary = tmp_path / "other-python"
    arbitrary.symlink_to(interpreter)
    with pytest.raises(RR.ReleaseReadinessError, match="is a symlink"):
        RR._verify_binding(
            {**binding, "path": str(arbitrary)}, label="arbitrary executable"
        )

    interpreter.write_bytes(b"changed interpreter bytes")
    with pytest.raises(RR.ReleaseReadinessError, match="binding changed"):
        RR._verify_binding(binding, label="python executable")


def test_json_array_artifact_is_byte_bound_and_nested_bindings_are_closed(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"sealed payload")
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps([{"epoch": 1}, {"artifact": RR.bind_file(payload)}]),
        encoding="utf-8",
    )
    root = RR._content_document(
        {"classification": "root_document", "history": RR.bind_file(history)}
    )
    RR._verify_closure(root, label="root")

    payload.write_bytes(b"tampered payload")
    with pytest.raises(RR.ReleaseReadinessError, match="binding changed"):
        RR._verify_closure(root, label="root")


def _accuracy(passed: bool = True) -> dict[str, Any]:
    values = {
        "overall_mae": 0.8 if passed else 1.2,
        "identity_macro_mae": 0.8,
        "overall_rmse": 1.4,
        "within_2_fraction": 0.94,
        "over_5_fraction": 0.01,
        "tail_25_35_mae": 1.5,
    }
    checks = {}
    for name, (operator, threshold) in RR.ACCURACY_CHECKS.items():
        decision = RR._gate_pass(values[name], operator, threshold)
        checks[name] = {
            "value": values[name], "operator": operator,
            "threshold": threshold, "passed": decision,
        }
    return {
        "all_point_gates_passed": all(row["passed"] for row in checks.values()),
        "checks": checks,
    }


def _uncertainty_decision(passed: bool = True) -> dict[str, Any]:
    observed = {
        "conformal_max_absolute_calibration_error_all_levels": 0.04 if passed else 0.08,
        "conformal_90_marginal_coverage_min": 0.90,
        "conformal_90_identity_macro_coverage_min": 0.88,
        "conformal_90_fixed_phase_0_coverage_min": 0.88,
        "conformal_90_mean_full_width_bpm_max": 4.0,
        "conformal_90_p95_full_width_bpm_max": 8.0,
        "selective_80_mae_bpm_max": 0.8,
        "selective_80_catastrophic_over_5_max": 0.01,
    }
    gates = {}
    for name, value in observed.items():
        operator = ">=" if name.endswith("_min") else "<="
        threshold = float(RR.UNCERTAINTY_GATES[name])
        gates[name] = {
            "observed": value, "operator": operator, "threshold": threshold,
            "passed": RR._gate_pass(value, operator, threshold),
        }
    return {"all_gates_passed": all(row["passed"] for row in gates.values()), "gates": gates}


def _streaming_engineering(spike_rate: float | None = 0.10) -> dict[str, Any]:
    return {
        "latency": {
            "cpu": {
                "applicable": True, "value_ms": 10.0,
                "maximum_ms": RR.ENGINEERING_GATES["cpu_warm_p99_ms_max"],
                "pass": True,
                "stride_fraction": 10.0 / RR.ENGINEERING_GATES["stride_budget_ms"],
                "stride_fraction_maximum": RR.ENGINEERING_GATES["p99_stride_budget_fraction_max"],
                "stride_pass": True,
            },
            "cuda": {
                "applicable": False, "value_ms": None,
                "maximum_ms": RR.ENGINEERING_GATES["cuda_warm_p99_ms_max"],
                "pass": True, "stride_fraction": None,
                "stride_fraction_maximum": RR.ENGINEERING_GATES["p99_stride_budget_fraction_max"],
                "stride_pass": True,
            },
        },
        "checkpoint_bytes": {
            "value": 1000, "maximum": RR.ENGINEERING_GATES["checkpoint_bytes_max"], "pass": True,
        },
        "parameter_count": {
            "value": 195603, "maximum": RR.ENGINEERING_GATES["parameter_count_max"], "pass": True,
        },
        "cpu_process_peak_rss_bytes": {
            "applicable": True, "value": 1000,
            "maximum": RR.ENGINEERING_GATES["cpu_process_peak_rss_bytes_max"], "pass": True,
        },
        "cuda_peak_reserved_bytes": {
            "applicable": False, "value": None,
            "maximum": RR.ENGINEERING_GATES["cuda_peak_reserved_bytes_max"], "pass": True,
        },
        "spike_rate_diagnostic": {
            "applicable": spike_rate is not None, "value": spike_rate,
            "minimum": RR.ENGINEERING_GATES["spike_rate_diagnostic_min"],
            "maximum": RR.ENGINEERING_GATES["spike_rate_diagnostic_max"],
            "pass": True,
            "unavailable_policy": RR.ENGINEERING_GATES["spike_rate_unavailable_policy"],
        },
    }


def _deployment_gates(spike_rate: float | None = 0.10) -> dict[str, Any]:
    return {
        "all_applicable_pass": True,
        "results": {
            "warm_p99_device_limit": {
                "value_ms": 10.0, "maximum_ms": RR.ENGINEERING_GATES["cpu_warm_p99_ms_max"],
                "pass": True,
            },
            "warm_p99_stride_fraction": {
                "value": 0.0025,
                "maximum": RR.ENGINEERING_GATES["p99_stride_budget_fraction_max"], "pass": True,
            },
            "checkpoint_bytes": {
                "value": 1000, "maximum": RR.ENGINEERING_GATES["checkpoint_bytes_max"], "pass": True,
            },
            "parameter_count": {
                "value": 195603, "maximum": RR.ENGINEERING_GATES["parameter_count_max"], "pass": True,
            },
            "cpu_process_peak_rss": {
                "value_bytes": 1000,
                "maximum_bytes": RR.ENGINEERING_GATES["cpu_process_peak_rss_bytes_max"], "pass": True,
            },
            "spike_rate_diagnostic": {
                "available": spike_rate is not None, "value": spike_rate,
                "minimum": RR.ENGINEERING_GATES["spike_rate_diagnostic_min"],
                "maximum": RR.ENGINEERING_GATES["spike_rate_diagnostic_max"],
                "pass": True,
                "unavailable_policy": RR.ENGINEERING_GATES["spike_rate_unavailable_policy"],
            },
        },
    }


def _roles(root: Path) -> dict[str, Path | None]:
    return {
        "target_release_receipt": root / "locked/canonical_targets_release_receipt.json",
        "pretarget_release_lock": root / "locked/pretarget_release_lock.json",
        "primary_evaluation_lock": root / "locked/evaluation_lock.json",
        "canonical_target": root / "locked/targets.npz",
        "canonical_target_receipt": root / "locked/targets_receipt.json",
        "joined_output": root / "locked/joined.npz",
        "predictions_seal": root / "locked/predictions_seal.json",
        "release_evaluation_execution_attestation": root / "locked/release_evaluation_execution_attestation.json",
        "primary_report": root / "primary/report.json",
        "primary_receipt": root / "primary/receipt.json",
        "radar_report": root / "radar/report.json",
        "radar_receipt": root / "radar/receipt.json",
        "uncertainty_spec": root / "campaign/uncertainty_spec.json",
        "primary_evaluation_spec": root / "campaign/primary_spec.json",
        "uncertainty_report": root / "uncertainty/report.json",
        "uncertainty_receipt": root / "uncertainty/receipt.json",
        "streaming_complete_seal": root / "streaming/complete_seal.json",
        "radar_mask_complete_seal": root / "radar/complete_seal.json",
        "uncertainty_inputs_seal": root / "locked/uncertainty_inputs_seal.json",
        "proposer_cpu_complete_seal": root / "cpu/complete_seal.json",
        "proposer_cuda_complete_seal": root / "cuda/complete_seal.json",
        "fixed_i3_runtime_seal": root / "guards/runtime_seal.json",
        "fixed_runtime_completion": root / "guards/fixed_completion.json",
        "postlock_runtime_guard": root / "guards/postlock.json",
        "radar_mask_runtime_guard": root / "guards/radar.json",
        "commercial_execution_plan": root / "campaign/COMMERCIAL_PLAN_V4.md",
    }


def _target_boundaries(root: Path, roles: dict[str, Path | None]) -> dict[str, str]:
    return {
        "canonical_target": str((root / "locked/targets.npz").resolve()),
        "canonical_target_receipt": str((root / "locked/targets_receipt.json").resolve()),
        "evaluation_lock": str(Path(roles["primary_evaluation_lock"]).resolve()),
        "joined_output": str((root / "locked/joined.npz").resolve()),
        "release_receipt": str(Path(roles["target_release_receipt"]).resolve()),
    }


def _pretarget_graph(
    root: Path,
    roles: dict[str, Path | None],
    *,
    cuda_available: bool = False,
    spike_rate: float | None = 0.10,
    one_unit_spike_missing: bool = False,
) -> None:
    for name in (
        "fixed_i3_runtime_seal", "fixed_runtime_completion", "postlock_runtime_guard",
        "radar_mask_runtime_guard",
    ):
        _doc(Path(roles[name]), {"schema_version": 1, "classification": f"synthetic_{name}"})
    calibration_path = root / "campaign/uncertainty_calibration.json"
    _doc(calibration_path, {
        "schema_version": 1,
        "classification": "locked_pretest_cross_fitted_proposer_uncertainty_calibration",
    })
    _doc(
        Path(roles["uncertainty_spec"]),
        {
            "schema_version": 1,
            "classification": "locked_hcs_secondary_uncertainty_evaluation_specification",
            "fixed_evaluation_gates": RR.UNCERTAINTY_GATES,
            "bound_inputs": {
                "completed_pretest_uncertainty_calibration": RR.bind_file(calibration_path),
            },
        },
    )
    _doc(Path(roles["primary_evaluation_spec"]), {
        "classification": "locked_hcs_oof_primary_evaluation_specification",
    })
    _doc(Path(roles["predictions_seal"]), {
        "classification": "locked_hcs_oof_all_label_free_predictions_sealed",
    })
    _doc(Path(roles["radar_mask_complete_seal"]), {
        "classification": "locked_hcs_all_seven_radar_mask_predictions_sealed",
    })
    _doc(Path(roles["uncertainty_inputs_seal"]), {
        "classification": "locked_hcs_all_target_free_uncertainty_inputs_sealed",
    })
    commercial_plan = Path(roles["commercial_execution_plan"])
    commercial_plan.parent.mkdir(parents=True, exist_ok=True)
    commercial_plan.write_text(
        "# Synthetic immutable commercial SNN continuous execution plan V4\n",
        encoding="utf-8",
    )
    commercial_plan.chmod(0o444)

    streaming_freeze = root / "streaming/freeze.json"
    _doc(streaming_freeze, {
        "schema_version": 1,
        "classification": "locked_hcs_streaming_deployment_freeze_spec",
        "runtime_identity": {"cuda_available_at_freeze": cuda_available},
        "mandatory_gates": list(RR.MANDATORY_STREAMING_GATES),
        "engineering_gates": {
            "cpu_stateful_one_window_warm_p99_ms_max": RR.ENGINEERING_GATES["cpu_warm_p99_ms_max"],
            "cuda_stateful_one_window_warm_p99_ms_max": RR.ENGINEERING_GATES["cuda_warm_p99_ms_max"],
            "stride_budget_ms": RR.ENGINEERING_GATES["stride_budget_ms"],
            "p99_stride_budget_fraction_max": RR.ENGINEERING_GATES["p99_stride_budget_fraction_max"],
            "checkpoint_bytes_max": RR.ENGINEERING_GATES["checkpoint_bytes_max"],
            "parameter_count_max": RR.ENGINEERING_GATES["parameter_count_max"],
            "cpu_process_peak_rss_bytes_max": RR.ENGINEERING_GATES["cpu_process_peak_rss_bytes_max"],
            "cuda_peak_reserved_bytes_max": RR.ENGINEERING_GATES["cuda_peak_reserved_bytes_max"],
            "spike_rate_diagnostic_min": RR.ENGINEERING_GATES["spike_rate_diagnostic_min"],
            "spike_rate_diagnostic_max": RR.ENGINEERING_GATES["spike_rate_diagnostic_max"],
            "spike_rate_unavailable_policy": RR.ENGINEERING_GATES["spike_rate_unavailable_policy"],
        },
    })
    streaming_plan = root / "streaming/plan.json"
    _doc(streaming_plan, {"freeze_spec": RR.bind_file(streaming_freeze)})
    stream_units = []
    for fold in RR.FOLDS:
        for seed in RR.SEEDS:
            unit_id = f"f{fold}_s{seed}"
            receipt_path = root / f"streaming/{unit_id}/receipt.json"
            unit_spike = None if one_unit_spike_missing and fold == 0 and seed == RR.SEEDS[0] else spike_rate
            _doc(receipt_path, {
                "classification": "locked_hcs_streaming_deployment_unit_receipt",
                "gates": {
                    "mandatory": {name: True for name in RR.MANDATORY_STREAMING_GATES},
                    "all_mandatory_pass": True,
                    "all_applicable_engineering_pass": True,
                    "engineering": _streaming_engineering(unit_spike),
                    "unit_integrated_pass": True,
                },
            })
            stream_units.append({
                "unit_id": unit_id, "outer_fold": fold, "seed": seed,
                "receipt": RR.bind_file(receipt_path), "unit_integrated_pass": True,
            })
    _doc(Path(roles["streaming_complete_seal"]), {
        "classification": "locked_hcs_streaming_deployment_all_18_complete_seal",
        "plan": RR.bind_file(streaming_plan),
        "freeze_spec": RR.bind_file(streaming_freeze),
        "unit_count": 18,
        "all_18_receipts_validated": True,
        "all_mandatory_gates_pass": True,
        "all_applicable_engineering_gates_pass": True,
        "integrated_pass": True,
        "unit_selection_or_ranking_performed": False,
        "commercial_performance_claim_authorized": False,
        "prospective_cohort_required_for_commercial_claim": True,
        "units": stream_units,
    })

    deployment_freeze = root / "cpu/freeze.json"
    frozen_gates = {
        "cpu_raw_resident_warm_p99_ms_max": RR.ENGINEERING_GATES["cpu_warm_p99_ms_max"],
        "cuda_raw_resident_warm_p99_ms_max": RR.ENGINEERING_GATES["cuda_warm_p99_ms_max"],
        "p99_stride_budget_fraction_max": RR.ENGINEERING_GATES["p99_stride_budget_fraction_max"],
        "checkpoint_bytes_max": RR.ENGINEERING_GATES["checkpoint_bytes_max"],
        "parameter_count_max": RR.ENGINEERING_GATES["parameter_count_max"],
        "cpu_process_peak_rss_bytes_max": RR.ENGINEERING_GATES["cpu_process_peak_rss_bytes_max"],
        "cuda_peak_reserved_bytes_max": RR.ENGINEERING_GATES["cuda_peak_reserved_bytes_max"],
        "spike_rate_diagnostic_min": RR.ENGINEERING_GATES["spike_rate_diagnostic_min"],
        "spike_rate_diagnostic_max": RR.ENGINEERING_GATES["spike_rate_diagnostic_max"],
        "spike_rate_unavailable_policy": RR.ENGINEERING_GATES["spike_rate_unavailable_policy"],
    }
    _doc(deployment_freeze, {
        "classification": "locked_proposer_deployment_benchmark_freeze_spec",
        "engineering_gates": frozen_gates,
        "measurement": {"stride_budget_ms": RR.ENGINEERING_GATES["stride_budget_ms"]},
    })
    deployment_plan = root / "cpu/plan.json"
    _doc(deployment_plan, {"runtime": {"device_type": "cpu"}, "freeze_spec": RR.bind_file(deployment_freeze)})
    deployment_units = []
    for fold in RR.FOLDS:
        for seed in RR.SEEDS:
            unit_id = f"f{fold}_s{seed}"
            receipt_path = root / f"cpu/{unit_id}/receipt.json"
            unit_spike = None if one_unit_spike_missing and fold == 0 and seed == RR.SEEDS[0] else spike_rate
            _doc(receipt_path, {
                "classification": "locked_proposer_deployment_benchmark_receipt",
                "runtime": {"device_type": "cpu"},
                "engineering_gates": _deployment_gates(unit_spike),
            })
            deployment_units.append({
                "unit_id": unit_id, "outer_fold": fold, "seed": seed,
                "receipt": RR.bind_file(receipt_path), "engineering_gates_pass": True,
            })
    _doc(Path(roles["proposer_cpu_complete_seal"]), {
        "classification": "locked_proposer_deployment_all_18_complete_seal",
        "plan": RR.bind_file(deployment_plan),
        "unit_count": 18,
        "all_18_reported": True,
        "all_applicable_engineering_gates_pass": True,
        "best_unit_selection_performed": False,
        "unit_ranking_performed": False,
        "commercial_performance_claim_authorized": False,
        "units": deployment_units,
    })

def _release_lock(root: Path, roles: dict[str, Path | None], spec: Path) -> None:
    radar_mask_plan = root / "radar/mask_plan.json"
    radar_mask_preexecution_lock = root / "radar/preexecution_lock.json"
    uncertainty_archive = root / "locked/uncertainty_inputs.npz"
    _doc(radar_mask_plan, {"classification": "synthetic_radar_mask_plan"})
    _doc(
        radar_mask_preexecution_lock,
        {"classification": "synthetic_radar_mask_preexecution_lock"},
    )
    uncertainty_archive.parent.mkdir(parents=True, exist_ok=True)
    uncertainty_archive.write_bytes(b"opaque target-free uncertainty arrays")
    uncertainty_archive.chmod(0o444)
    locations = {
        **_target_boundaries(root, roles),
        "fixed_i3_runtime_seal": str(Path(roles["fixed_i3_runtime_seal"]).resolve()),
        "fixed_runtime_completion": str(Path(roles["fixed_runtime_completion"]).resolve()),
        "postlock_runtime_guard": str(Path(roles["postlock_runtime_guard"]).resolve()),
        "radar_mask_runtime_guard": str(Path(roles["radar_mask_runtime_guard"]).resolve()),
        "uncertainty_evaluation_spec": str(Path(roles["uncertainty_spec"]).resolve()),
        "release_readiness_spec": str(spec.resolve()),
    }
    _doc(Path(roles["pretarget_release_lock"]), {
        "classification": "locked_hcs_pretarget_release_lock",
        "locations": locations,
        "frozen_specs": {
            "release_readiness": {
                "binding": RR.bind_file(spec),
                "content_sha256": json.loads(spec.read_text())["content_sha256"],
                "target_or_target_bearing_artifact_opened": False,
                "commercial_release_ready_must_equal": False,
                "prospective_confirmation_required": True,
            }
        },
        "boundaries": {
            "radar_masks": {
                "plan": RR.bind_file(radar_mask_plan),
                "preexecution_lock": RR.bind_file(radar_mask_preexecution_lock),
            },
            "uncertainty": {
                "uncertainty_archive": RR.bind_file(uncertainty_archive),
            },
            "runtime_payload_closure": {
                "runtime_input_seal": RR.bind_file(Path(roles["fixed_i3_runtime_seal"])),
                "fixed_pretest_completion_attestation": RR.bind_file(Path(roles["fixed_runtime_completion"])),
                "postlock_runtime_guard_attestation": RR.bind_file(Path(roles["postlock_runtime_guard"])),
                "radar_mask_runtime_guard_attestation": RR.bind_file(Path(roles["radar_mask_runtime_guard"])),
            }
        },
    })


def _posttarget_evidence(root: Path, roles: dict[str, Path | None], *, pass_primary: bool = True) -> None:
    target = Path(roles["canonical_target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"opaque canonical target bytes; aggregate never opens arrays")
    target.chmod(0o444)
    target_binding = RR.bind_file(target)
    _doc(Path(roles["canonical_target_receipt"]), {
        "classification": "retrospective_locked_hcs_canonical_target_artifact_receipt",
        "target_artifact": target_binding,
    })
    joined = Path(roles["joined_output"])
    joined.write_bytes(b"opaque joined evidence bytes; aggregate never opens arrays")
    joined.chmod(0o444)
    joined_metrics = root / "locked/joined_metrics.json"
    _doc(joined_metrics, {
        "classification": "retrospective_locked_hcs_oof_evaluation",
    })
    _doc(Path(roles["target_release_receipt"]), {
        "classification": "locked_hcs_canonical_targets_built_after_pretarget_release",
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
        "release_lock_revalidated_before_target_builder_call": True,
        "pretarget_release_lock": RR.bind_file(Path(roles["pretarget_release_lock"])),
        "canonical_target": target_binding,
        "canonical_target_receipt": RR.bind_file(Path(roles["canonical_target_receipt"])),
    })
    _doc(Path(roles["primary_evaluation_lock"]), {
        "classification": "locked_hcs_oof_single_target_join_seal",
        "target_artifact": target_binding,
        "target_join_count": 1,
        "commercial_claim_authorized": False,
        "outputs": {
            "joined_oof": RR.bind_file(joined),
            "metrics": RR.bind_file(joined_metrics),
        },
    })

    per_seed = {}
    statuses = {}
    for seed in RR.SEEDS:
        decision = _accuracy(pass_primary)
        statuses[str(seed)] = decision["all_point_gates_passed"]
        per_seed[str(seed)] = {
            "seed": seed, "seed_evaluated_independently": True,
            "cross_seed_pooling_performed": False, "locked_final_goal": decision,
        }
    primary = _doc(Path(roles["primary_report"]), {
        "classification": "retrospective_locked_hcs_oof_primary_evaluation",
        "evaluation_specification": RR.bind_file(Path(roles["primary_evaluation_spec"])),
        "commercial_claim_authorized": False,
        "commercial_performance_proven": False,
        "prospective_confirmation_required": True,
        "independent_prospective_cohort_evaluated": False,
        "goal_targets": RR.ACCURACY_GATES,
        "selection_or_retraining_performed": False,
        "seed_ranking_or_suppression_performed": False,
        "cross_seed_pooling_performed": False,
        "fixed_seed_gate_status": statuses,
        "all_fixed_seeds_point_gates_passed": all(statuses.values()),
        "per_seed": per_seed,
    })
    primary_csv = root / "primary/metrics.csv"
    primary_csv.write_text("metric,value\nsynthetic,1\n", encoding="utf-8")
    primary_csv.chmod(0o444)
    common_inputs = {
        "evaluation_lock": RR.bind_file(Path(roles["primary_evaluation_lock"])),
        "predictions_seal": RR.bind_file(Path(roles["predictions_seal"])),
        "target_receipt": RR.bind_file(Path(roles["canonical_target_receipt"])),
        "target_artifact": target_binding,
        "joined_oof": RR.bind_file(joined),
        "locked_metrics": RR.bind_file(joined_metrics),
    }
    _doc(Path(roles["primary_receipt"]), {
        "classification": "retrospective_locked_hcs_oof_primary_evaluation_receipt",
        "commercial_claim_authorized": False, "commercial_performance_proven": False,
        "prospective_confirmation_required": True,
        "independent_prospective_cohort_evaluated": False,
        "outputs_create_once": True, "output_overwrite_allowed": False,
        "seeds": list(RR.SEEDS),
        "inputs": {
            **common_inputs,
            "evaluation_spec": RR.bind_file(Path(roles["primary_evaluation_spec"])),
        },
        "outputs": {
            "report": RR.bind_file(Path(roles["primary_report"])),
            "metrics_csv": RR.bind_file(primary_csv),
        },
    })

    parity = {}
    radar_seeds = {}
    for seed in RR.SEEDS:
        parity[str(seed)] = {
            "passed": True,
            "locked_final_metrics_exact": True,
            "array_bit_exact": {
                "fallback_rr_bpm": True, "source_rr_bpm": True, "final_rr_bpm": True,
            },
        }
        radar_seeds[str(seed)] = {
            "seed": seed, "seed_evaluated_independently": True,
            "all_seven_masks_reported_without_selection": True,
            "all_seven_masks_fixed_point_gates_passed": True,
            "radar_masks": {
                mask: {"fixed_point_goal_gate": _accuracy(True)} for mask in RR.MASKS
            },
        }
    _doc(Path(roles["radar_report"]), {
        "classification": "retrospective_locked_hcs_all_radar_masks_evaluation",
        "evaluation_specification": RR.bind_file(Path(roles["primary_evaluation_spec"])),
        "commercial_claim_authorized": False,
        "commercial_performance_proven": False,
        "prospective_confirmation_required": True,
        "independent_prospective_cohort_evaluated": False,
        "mask_selection_or_ranking_performed": False,
        "seed_pooling_ranking_or_suppression_performed": False,
        "all_seven_masks_are_required_fixed_conditions": True,
        "all_masks_all_fixed_seeds_point_gates_passed": True,
        "radars_123_primary_parity_gate": {
            "required": True, "all_fixed_seeds_passed": True, "per_seed": parity,
        },
        "per_seed": radar_seeds,
    })
    radar_csv = root / "radar/metrics.csv"
    radar_csv.write_text("metric,value\nsynthetic,1\n", encoding="utf-8")
    radar_csv.chmod(0o444)
    radar_mask_plan = root / "radar/mask_plan.json"
    radar_mask_preexecution_lock = root / "radar/preexecution_lock.json"
    _doc(Path(roles["radar_receipt"]), {
        "classification": "retrospective_locked_hcs_all_radar_masks_evaluation_receipt",
        "commercial_claim_authorized": False, "commercial_performance_proven": False,
        "prospective_confirmation_required": True,
        "independent_prospective_cohort_evaluated": False,
        "outputs_create_once": True, "output_overwrite_allowed": False,
        "inputs": {
            **common_inputs,
            "evaluation_spec": RR.bind_file(Path(roles["primary_evaluation_spec"])),
            "primary_predictions_seal": RR.bind_file(Path(roles["predictions_seal"])),
            "radar_mask_complete_seal": RR.bind_file(
                Path(roles["radar_mask_complete_seal"])
            ),
            "radar_mask_plan": RR.bind_file(radar_mask_plan),
            "radar_mask_preexecution_lock": RR.bind_file(
                radar_mask_preexecution_lock
            ),
        },
        "outputs": {
            "report": RR.bind_file(Path(roles["radar_report"])),
            "metrics_csv": RR.bind_file(radar_csv),
        },
    })

    uncertainty_status = {str(seed): True for seed in RR.SEEDS}
    calibration_path = root / "campaign/uncertainty_calibration.json"
    uncertainty_archive = root / "locked/uncertainty_inputs.npz"
    uncertainty_audit = {
        "secondary_uncertainty_evaluation_spec": RR.bind_file(
            Path(roles["uncertainty_spec"])
        ),
        "evaluation_spec": RR.bind_file(Path(roles["primary_evaluation_spec"])),
        "calibration": RR.bind_file(calibration_path),
        "predictions_seal": RR.bind_file(Path(roles["predictions_seal"])),
        "uncertainty_inputs_seal": RR.bind_file(
            Path(roles["uncertainty_inputs_seal"])
        ),
        "uncertainty_archive": RR.bind_file(uncertainty_archive),
        "calibration_declared_bindings_rehashed": 1,
        "prediction_declared_bindings_rehashed": 1,
        "uncertainty_declared_bindings_rehashed": 1,
        "secondary_protocol_role": "separate_secondary_retrospective_engineering_protocol",
        "primary_uncertainty_contract_overridden": False,
        "dedicated_secondary_spec_verified_before_prediction_uncertainty_or_target_access": True,
        "all_target_free_inputs_verified_before_evaluation_lock_access": True,
        "all_uncertainty_array_schema_and_hashes_verified": True,
    }
    uncertainty = _doc(Path(roles["uncertainty_report"]), {
        "classification": "retrospective_locked_hcs_uncertainty_evaluation",
        "commercial_claim_authorized": False,
        "commercial_performance_proven": False,
        "prospective_confirmation_required": True,
        "independent_prospective_cohort_evaluated": False,
        "uncertainty_evaluation_specification": RR.bind_file(Path(roles["uncertainty_spec"])),
        "frozen_evaluation_gates": RR.UNCERTAINTY_GATES,
        "selection_retraining_or_test_time_fitting_performed": False,
        "interval_scale_or_threshold_refit_performed": False,
        "seed_pooling_ranking_or_suppression_performed": False,
        "point_prediction_modified": False,
        "fixed_seed_gate_status": uncertainty_status,
        "all_fixed_seed_uncertainty_gates_passed": True,
        "pretarget_provenance_audit": uncertainty_audit,
        "per_seed": {
            str(seed): {"fixed_gate_decision": _uncertainty_decision(True)}
            for seed in RR.SEEDS
        },
    })
    uncertainty_csv = root / "uncertainty/metrics.csv"
    uncertainty_csv.write_text("metric,value\nsynthetic,1\n", encoding="utf-8")
    uncertainty_csv.chmod(0o444)
    _doc(Path(roles["uncertainty_receipt"]), {
        "classification": "retrospective_locked_hcs_uncertainty_evaluation_receipt",
        "commercial_claim_authorized": False, "commercial_performance_proven": False,
        "prospective_confirmation_required": True,
        "independent_prospective_cohort_evaluated": False,
        "outputs_create_once": True, "output_overwrite_allowed": False,
        "seeds": list(RR.SEEDS),
        "inputs": {
            **common_inputs,
            **uncertainty_audit,
        },
        "outputs": {
            "report": RR.bind_file(Path(roles["uncertainty_report"])),
            "metrics_csv": RR.bind_file(uncertainty_csv),
        },
    })
    readiness_binding = json.loads(
        Path(roles["pretarget_release_lock"]).read_text()
    )["frozen_specs"]["release_readiness"]["binding"]
    runner = RR._load_release_runner_module()
    commands = runner.build_frozen_argv_from_evidence(
        {
            "raw_join_source": RR.PROJECT_ROOT / "scripts/run_locked_hcs_oof.py",
            "primary_source": RR.SOURCE_PATHS["primary_evaluator"],
            "radar_source": RR.SOURCE_PATHS["radar_mask_evaluator"],
            "uncertainty_source": RR.SOURCE_PATHS["uncertainty_evaluator"],
            "primary_root": Path(roles["joined_output"]).parent,
            "mask_root": Path(roles["radar_mask_complete_seal"]).parent,
            "canonical_target": Path(roles["canonical_target"]),
            "evaluation_lock": Path(roles["primary_evaluation_lock"]),
            "canonical_target_receipt": Path(roles["canonical_target_receipt"]),
            "primary_evaluation_spec": Path(roles["primary_evaluation_spec"]),
            "uncertainty_evaluation_spec": Path(roles["uncertainty_spec"]),
            "uncertainty_calibration": calibration_path,
            "predictions_seal": Path(roles["predictions_seal"]),
            "uncertainty_inputs_seal": Path(roles["uncertainty_inputs_seal"]),
            "primary_output_dir": Path(roles["primary_report"]).parent,
            "primary_report": Path(roles["primary_report"]),
            "primary_csv": primary_csv,
            "primary_receipt": Path(roles["primary_receipt"]),
            "radar_output_dir": Path(roles["radar_report"]).parent,
            "radar_report": Path(roles["radar_report"]),
            "radar_csv": radar_csv,
            "radar_receipt": Path(roles["radar_receipt"]),
            "uncertainty_output_dir": Path(roles["uncertainty_report"]).parent,
            "uncertainty_report": Path(roles["uncertainty_report"]),
            "uncertainty_csv": uncertainty_csv,
            "uncertainty_receipt": Path(roles["uncertainty_receipt"]),
        },
        python_executable=sys.executable,
    )
    _doc(Path(roles["release_evaluation_execution_attestation"]), {
        "classification": "locked_hcs_release_evaluation_execution_attestation",
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
        "target_re_evaluation_performed": False,
        "all_steps_executed_once": True,
        "all_inputs_and_outputs_live_rehashed": True,
        "execution_source": RR.bind_file(RR.SOURCE_PATHS["release_evaluation_orchestrator"]),
        "release_readiness_spec": readiness_binding,
        "pretarget_release_lock": RR.bind_file(Path(roles["pretarget_release_lock"])),
        "target_release_receipt": RR.bind_file(Path(roles["target_release_receipt"])),
        "canonical_target": RR.bind_file(Path(roles["canonical_target"])),
        "canonical_target_receipt": RR.bind_file(Path(roles["canonical_target_receipt"])),
        "predictions_seal": RR.bind_file(Path(roles["predictions_seal"])),
        "evaluation_lock": RR.bind_file(Path(roles["primary_evaluation_lock"])),
        "primary_evaluation_spec": RR.bind_file(Path(roles["primary_evaluation_spec"])),
        "uncertainty_evaluation_spec": RR.bind_file(Path(roles["uncertainty_spec"])),
        "radar_mask_complete_seal": RR.bind_file(Path(roles["radar_mask_complete_seal"])),
        "uncertainty_inputs_seal": RR.bind_file(Path(roles["uncertainty_inputs_seal"])),
        "evaluations": {
            "primary": {
                "report": RR.bind_file(Path(roles["primary_report"])),
                "receipt": RR.bind_file(Path(roles["primary_receipt"])),
            },
            "radar_masks": {
                "report": RR.bind_file(Path(roles["radar_report"])),
                "receipt": RR.bind_file(Path(roles["radar_receipt"])),
            },
            "uncertainty": {
                "report": RR.bind_file(Path(roles["uncertainty_report"])),
                "receipt": RR.bind_file(Path(roles["uncertainty_receipt"])),
            },
        },
        "frozen_argv": commands,
        "executed_commands": commands,
    })


def _frozen_fixture(
    tmp_path: Path,
    *,
    pass_primary: bool = True,
    spike_rate: float | None = 0.10,
    one_unit_spike_missing: bool = False,
) -> tuple[dict[str, Path | None], Path]:
    roles = _roles(tmp_path)
    _pretarget_graph(
        tmp_path, roles, spike_rate=spike_rate,
        one_unit_spike_missing=one_unit_spike_missing,
    )
    spec = tmp_path / "campaign/release_readiness_spec.json"
    RR.freeze_spec(output=spec, roles=roles)
    _release_lock(tmp_path, roles, spec)
    _posttarget_evidence(tmp_path, roles, pass_primary=pass_primary)
    return roles, spec


def _evaluate(tmp_path: Path, roles: dict[str, Path | None], spec: Path) -> dict[str, Any]:
    output = tmp_path / "ready"
    return RR.evaluate(
        spec_path=spec, roles=roles, output_dir=output,
        report_output=output / "report.json", csv_output=output / "gates.csv",
        receipt_output=output / "receipt.json", orchestrator_command=["synthetic"],
    )


def test_freeze_spec_is_independent_pre_target_create_once_and_0444(tmp_path: Path) -> None:
    roles = _roles(tmp_path)
    _pretarget_graph(tmp_path, roles)
    spec = tmp_path / "release_spec.json"
    result = RR.freeze_spec(output=spec, roles=roles)
    assert result["target_or_target_bearing_artifact_opened"] is False
    assert not Path(roles["pretarget_release_lock"]).exists()
    assert not (tmp_path / "locked/targets.npz").exists()
    assert stat.S_IMODE(spec.stat().st_mode) == 0o444
    assert result["prospective_release_policy"]["commercial_release_ready_must_equal"] is False
    with pytest.raises(RR.ReleaseReadinessError, match="already exists"):
        RR.freeze_spec(output=spec, roles=roles)


def test_synthetic_all_pass_is_internal_candidate_but_never_commercial(tmp_path: Path) -> None:
    roles, spec = _frozen_fixture(tmp_path)
    receipt = _evaluate(tmp_path, roles, spec)
    report = json.loads((tmp_path / "ready/report.json").read_text())
    assert receipt["internal_retrospective_engineering_candidate_ready"] is True
    assert report["internal_retrospective_engineering_candidate_ready"] is True
    assert report["commercial_release_ready"] is False
    assert report["commercial_claim_authorized"] is False
    assert report["independent_prospective_cohort_evaluated"] is False
    assert len(report["commercial_release_blocked_reasons"]) == 3
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in (tmp_path / "ready").iterdir())


def test_failed_fixed_seed_is_reported_without_seed_suppression(tmp_path: Path) -> None:
    roles, spec = _frozen_fixture(tmp_path, pass_primary=False)
    receipt = _evaluate(tmp_path, roles, spec)
    report = json.loads((tmp_path / "ready/report.json").read_text())
    assert receipt["internal_retrospective_engineering_candidate_ready"] is False
    assert report["category_gate_status"]["primary_accuracy"] is False
    assert report["commercial_release_ready"] is False
    csv_text = (tmp_path / "ready/gates.csv").read_text()
    assert all(str(seed) in csv_text for seed in RR.SEEDS)


def test_tampered_bound_report_fails_closed_without_outputs(tmp_path: Path) -> None:
    roles, spec = _frozen_fixture(tmp_path)
    primary = Path(roles["primary_report"])
    primary.chmod(0o644)
    primary.write_text(primary.read_text() + " ", encoding="utf-8")
    primary.chmod(0o444)
    with pytest.raises(RR.ReleaseReadinessError, match="binding changed|does not bind"):
        _evaluate(tmp_path, roles, spec)
    assert not (tmp_path / "ready/report.json").exists()


def test_attestation_alternate_argv_is_rejected_even_when_executed_matches(
    tmp_path: Path,
) -> None:
    roles, spec = _frozen_fixture(tmp_path)
    attestation_path = Path(roles["release_evaluation_execution_attestation"])
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation.pop("content_sha256")
    commands = [list(command) for command in attestation["frozen_argv"]]
    commands[2] = [*commands[2], "--unlocked-alternate-output"]
    attestation["frozen_argv"] = commands
    attestation["executed_commands"] = [list(command) for command in commands]
    attestation_path.chmod(0o644)
    _doc(attestation_path, attestation)
    with pytest.raises(RR.ReleaseReadinessError, match="exact fixed argv differs"):
        _evaluate(tmp_path, roles, spec)
    assert not (tmp_path / "ready/report.json").exists()


def test_readiness_rejects_uncertainty_receipt_scalar_even_with_rebound_attestation(
    tmp_path: Path,
) -> None:
    roles, spec = _frozen_fixture(tmp_path)
    receipt_path = Path(roles["uncertainty_receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("content_sha256")
    receipt["inputs"]["secondary_protocol_role"] = "forged_protocol"
    receipt_path.chmod(0o644)
    _doc(receipt_path, receipt)

    attestation_path = Path(roles["release_evaluation_execution_attestation"])
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation.pop("content_sha256")
    attestation["evaluations"]["uncertainty"]["receipt"] = RR.bind_file(receipt_path)
    attestation_path.chmod(0o644)
    _doc(attestation_path, attestation)

    with pytest.raises(RR.ReleaseReadinessError, match="secondary_protocol_role"):
        _evaluate(tmp_path, roles, spec)
    assert not (tmp_path / "ready/report.json").exists()


def test_evaluate_rechecks_mode_for_every_frozen_role(tmp_path: Path) -> None:
    roles, spec = _frozen_fixture(tmp_path)
    # This non-JSON plan is a pretarget role that was previously omitted from
    # evaluate-time mode enforcement; its bytes and frozen hash stay unchanged.
    plan = Path(roles["commercial_execution_plan"])
    plan.chmod(0o644)
    with pytest.raises(
        RR.ReleaseReadinessError,
        match="commercial_execution_plan must be a regular mode-0444 file",
    ):
        _evaluate(tmp_path, roles, spec)
    assert not (tmp_path / "ready/report.json").exists()


def test_spec_drift_is_rejected_before_missing_target_receipt(tmp_path: Path) -> None:
    roles = _roles(tmp_path)
    _pretarget_graph(tmp_path, roles)
    spec = tmp_path / "release_spec.json"
    RR.freeze_spec(output=spec, roles=roles)
    spec.chmod(0o644)
    document = json.loads(spec.read_text())
    document["exact_gates"]["accuracy"]["overall_mae_max_bpm"] = 9.0
    spec.write_text(json.dumps(document), encoding="utf-8")
    spec.chmod(0o444)
    output = tmp_path / "ready"
    with pytest.raises(RR.ReleaseReadinessError, match="content hash mismatch"):
        RR.evaluate(
            spec_path=spec, roles=roles, output_dir=output,
            report_output=output / "report.json", csv_output=output / "gates.csv",
            receipt_output=output / "receipt.json",
        )
    assert not Path(roles["target_release_receipt"]).exists()


def test_evaluation_outputs_are_create_once(tmp_path: Path) -> None:
    roles, spec = _frozen_fixture(tmp_path)
    _evaluate(tmp_path, roles, spec)
    with pytest.raises(RR.ReleaseReadinessError, match="already exists"):
        _evaluate(tmp_path, roles, spec)


def test_freeze_fails_if_canonical_target_already_exists(tmp_path: Path) -> None:
    roles = _roles(tmp_path)
    _pretarget_graph(tmp_path, roles)
    target = tmp_path / "locked/targets.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"target")
    with pytest.raises(RR.ReleaseReadinessError, match="must precede target access"):
        RR.freeze_spec(output=tmp_path / "release_spec.json", roles=roles)


def test_cuda_available_at_streaming_freeze_requires_separate_cuda_seal(tmp_path: Path) -> None:
    roles = _roles(tmp_path)
    _pretarget_graph(tmp_path, roles, cuda_available=True)
    assert not Path(roles["proposer_cuda_complete_seal"]).exists()
    with pytest.raises(RR.ReleaseReadinessError, match="CUDA proposer seal is required"):
        RR.freeze_spec(output=tmp_path / "release_spec.json", roles=roles)


def test_unavailable_none_spike_telemetry_fails_internal_candidate(tmp_path: Path) -> None:
    roles, spec = _frozen_fixture(tmp_path, spike_rate=None)
    _evaluate(tmp_path, roles, spec)
    report = json.loads((tmp_path / "ready/report.json").read_text())
    assert report["category_gate_status"]["streaming"] is False
    assert report["category_gate_status"]["proposer_cpu"] is False
    assert report["internal_retrospective_engineering_candidate_ready"] is False
    assert report["commercial_release_ready"] is False


@pytest.mark.parametrize("spike_rate", [0.01, 0.20])
def test_spike_operating_band_inclusive_boundaries_pass(
    tmp_path: Path, spike_rate: float
) -> None:
    roles, spec = _frozen_fixture(tmp_path, spike_rate=spike_rate)
    _evaluate(tmp_path, roles, spec)
    report = json.loads((tmp_path / "ready/report.json").read_text())
    assert report["category_gate_status"]["streaming"] is True
    assert report["category_gate_status"]["proposer_cpu"] is True
    assert report["internal_retrospective_engineering_candidate_ready"] is True


def test_one_of_18_units_missing_spike_telemetry_fails_without_suppression(
    tmp_path: Path,
) -> None:
    roles, spec = _frozen_fixture(tmp_path, one_unit_spike_missing=True)
    _evaluate(tmp_path, roles, spec)
    report = json.loads((tmp_path / "ready/report.json").read_text())
    assert report["category_gate_status"]["streaming"] is False
    assert report["category_gate_status"]["proposer_cpu"] is False
    csv_text = (tmp_path / "ready/gates.csv").read_text()
    assert "streaming,f0_s20260828,false" in csv_text
    assert "proposer_cpu,f0_s20260828,false" in csv_text


@pytest.mark.parametrize("mode", ["freeze-spec", "evaluate"])
def test_production_cli_default_seal_paths_match_producer_abi(mode: str) -> None:
    args = RR.build_parser().parse_args([mode])
    benchmark = RR.PROJECT_ROOT / "artifacts/benchmarks/locked_proposer_deployment"
    streaming = (
        RR.PROJECT_ROOT
        / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_streaming_deployment"
    )
    masks = (
        RR.PROJECT_ROOT
        / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_radar_masks"
    )
    assert args.proposer_cpu_complete_seal == benchmark / "cpu/complete_seal.json"
    assert args.proposer_cuda_complete_seal == benchmark / "cuda/complete_seal.json"
    assert args.streaming_complete_seal == streaming / "complete_seal.json"
    assert args.radar_mask_complete_seal == masks / "complete_seal.json"
    assert "locked_proposer_deployment_cuda" not in str(args.proposer_cuda_complete_seal)
