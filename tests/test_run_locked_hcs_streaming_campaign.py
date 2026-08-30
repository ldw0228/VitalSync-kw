from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_locked_hcs_streaming_campaign.py"
)
SPEC = importlib.util.spec_from_file_location("run_locked_hcs_streaming_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUN
SPEC.loader.exec_module(RUN)


def test_python_launcher_binding_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    launcher = tmp_path / "venv-python"
    launcher.symlink_to(Path(sys.executable))
    binding = RUN.bind_python_launcher(launcher)
    assert binding["path"] == str(launcher.absolute())
    assert Path(binding["path"]) != launcher.resolve()
    assert binding["sha256"] == RUN.sha256_file(launcher.resolve())


def _synthetic_report(*, cpu_p99: float = 15.0, cuda: bool = False) -> dict[str, Any]:
    masks = [
        {
            "mask": [int(bool(bits & (1 << index))) for index in range(3)],
            "finite": True,
            "source_available": True,
            "missing_feature_corruption_invariant": True,
        }
        for bits in range(1, 8)
    ]
    benchmarks = []
    devices = ["cpu"] + (["cuda"] if cuda else [])
    for device in devices:
        for operation, repeats in (
            ("checkpoint_load_model_init_plus_first_window_cold", 3),
            ("stateful_one_window_warm", 50),
            ("stateless_chunk_warm", 50),
        ):
            p99 = cpu_p99 if device == "cpu" else 10.0
            benchmarks.append(
                {
                    "device": device,
                    "operation": operation,
                    "repeats": repeats,
                    "p99_ms": p99,
                    "peak_memory_bytes": 128 * 1024**2,
                    **(
                        {"cuda_peak_reserved_bytes": 256 * 1024**2}
                        if device == "cuda"
                        else {}
                    ),
                }
            )
    return {
        "bindings": {
            "locked_files": {"checkpoint": {"bytes": 4 * 1024**2}},
        },
        "parity": {
            "passed": True,
            "cases": [
                {"schedule": [7, 3]},
                {"schedule": [13, 1]},
                {"schedule": [2, 4]},
                {"schedule": [3, 5]},
                {"schedule": [1], "mode": "one_window_streaming"},
            ],
        },
        "session_reset": {"passed": True, "session_count": 2},
        "robustness": {
            "passed": True,
            "zero_features_finite": True,
            "no_candidate_structural_fallback_route": True,
            "corrupt_nan_inf_inputs_finite_and_unavailable": True,
            "all_radars_missing_finite_and_unavailable": True,
            "wrong_feature_shape_rejected": True,
            "seven_nonempty_radar_masks": masks,
        },
        "benchmarks": benchmarks,
    }


def _spec() -> dict[str, Any]:
    return {
        "verification": {"minimum_sessions_required": 2},
        "engineering_gates": {
            "cpu_stateful_one_window_warm_p99_ms_max": 250.0,
            "cuda_stateful_one_window_warm_p99_ms_max": 50.0,
            "stride_budget_ms": 4000.0,
            "p99_stride_budget_fraction_max": 0.10,
            "checkpoint_bytes_max": 50 * 1024**2,
            "parameter_count_max": 5_000_000,
            "cpu_process_peak_rss_bytes_max": 2 * 1024**3,
            "cuda_peak_reserved_bytes_max": 1024**3,
            "spike_rate_diagnostic_min": 0.01,
            "spike_rate_diagnostic_max": 0.20,
            "spike_rate_unavailable_policy": "reported_not_applicable_without_failure",
        },
    }


def _write(path: Path, data: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return RUN.bind_file(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return RUN.bind_file(path)


def _synthetic_pretest_index(tmp_path: Path) -> Path:
    freeze_root = tmp_path / "freeze"
    frozen_names = (
        "train_harmonic_set_snn.py",
        "harmonic_set_models.py",
        "harmonic_set_v2.yaml",
        "ADAPTIVE_CAMPAIGN_CONTRACT.json",
    )
    frozen_hashes = {}
    for name in frozen_names:
        path = freeze_root / name
        _write(path, name.encode())
        frozen_hashes[name] = RUN.sha256_file(path)
    freeze = _write_json(
        freeze_root / "MANIFEST.json",
        {
            "declared_before_any_i3_score": True,
            "outer_test_opened": False,
            "files": frozen_hashes,
        },
    )
    common_root = tmp_path / "common"
    capacity = _write_json(
        common_root / "capacity.json",
        {
            "outer_test_opened": False,
            "selected_preset": "default",
            "selected_parameter_count": 195_603,
            "source_freeze_manifest_sha256": freeze["sha256"],
        },
    )
    policy = _write_json(
        common_root / "policy.json",
        {
            "outer_test_opened": False,
            "selected_preset": "default",
            "policy": {"selection_status": "locked"},
        },
    )
    selection = _write_json(
        common_root / "selection.json",
        {
            "schema_version": 1,
            "classification": "retrospective_i3_common_discovery_lock",
            "outer_test_opened_before_lock": False,
            "selected_preset": "default",
            "selected_parameter_count": 195_603,
            "capacity_selection_sha256": capacity["sha256"],
            "common_fallback_policy_sha256": policy["sha256"],
            "source_freeze": frozen_hashes,
            "policy_selection_status": "locked",
        },
    )
    units = []
    for fold in RUN.FOLDS:
        for seed in RUN.SEEDS:
            unit_root = tmp_path / "units" / f"outer_{fold}_seed_{seed}"
            checkpoint = _write(unit_root / "best_checkpoint.pt", b"sealed checkpoint")
            scaler = _write_json(unit_root / "scaler.json", {"center": [0], "scale": [1]})
            run_manifest = _write_json(
                unit_root / "run_manifest.json",
                {
                    "outer_fold": fold,
                    "validation_fold": (fold + 1) % 6,
                    "optimization": {"seed": seed},
                    "retrospective_only": True,
                    "commercial_claim_authorized": False,
                },
            )
            original_policy = _write_json(
                unit_root / "fallback_policy.json", {"diagnostic": True}
            )
            history = _write_json(unit_root / "history.json", {"complete": True})
            validation_metrics = _write(
                unit_root / "validation_metrics.json", b'{"reference_metric":999}'
            )
            # Deliberately invalid NPZ bytes: the campaign must only hash this
            # target-bearing retrospective output, never deserialize it.
            validation_predictions = _write(
                unit_root / "validation_predictions.npz", b"not-an-npz-reference-array"
            )
            cache_root = tmp_path / "cache" / f"outer_{fold}_seed_{seed}"
            cache_manifest = _write_json(cache_root / "manifest.json", {"complete": True})
            fallback = _write(
                tmp_path / "fallback" / f"outer_{fold}_seed_{seed}.csv",
                b"cache_index,reference_rr_bpm\n0,99\n",
            )
            fallback_provenance = _write_json(
                Path(fallback["path"] + ".provenance.json"), {"outer_test_opened": False}
            )
            strict_stack = _write(
                tmp_path / "stacks" / f"outer_{fold}_seed_{seed}.npz",
                b"not-an-npz-strict-stack-with-reference-arrays",
            )
            source_bindings = {
                "trainer": {"sha256": frozen_hashes["train_harmonic_set_snn.py"]},
                "harmonic_set_model": {"sha256": frozen_hashes["harmonic_set_models.py"]},
                "campaign_config": {"sha256": frozen_hashes["harmonic_set_v2.yaml"]},
                "adaptive_campaign_contract": {
                    "sha256": frozen_hashes["ADAPTIVE_CAMPAIGN_CONTRACT.json"]
                },
            }
            lock = _write_json(
                unit_root / "selection_lock.json",
                {
                    "schema_version": 1,
                    "outer_fold": fold,
                    "seed": seed,
                    "adaptive_iteration": 3,
                    "outer_test_not_opened_before_this_lock": True,
                    "checkpoint_sha256": checkpoint["sha256"],
                    "scaler_sha256": scaler["sha256"],
                    "cache_manifest_sha256": cache_manifest["sha256"],
                    "fallback_oof_sha256": fallback["sha256"],
                    "run_manifest_sha256": run_manifest["sha256"],
                    "policy_sha256": original_policy["sha256"],
                    "history_sha256": history["sha256"],
                    "source_bindings": source_bindings,
                },
            )
            artifacts = {
                "selection_lock": lock,
                "checkpoint": checkpoint,
                "scaler": scaler,
                "cache_manifest": cache_manifest,
                "fallback_oof": fallback,
                "fallback_provenance": fallback_provenance,
                "run_manifest": run_manifest,
                "original_policy": original_policy,
                "history": history,
                "validation_metrics": validation_metrics,
                "validation_predictions": validation_predictions,
                "strict_stack": strict_stack,
            }
            units.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "status": "complete",
                    "output_root": str(unit_root.resolve()),
                    "output_tree": RUN._tree_document(unit_root),
                    "cache_root": {
                        "path": str(cache_root.resolve()),
                        "manifest_sha256": cache_manifest["sha256"],
                    },
                    "artifacts": artifacts,
                }
            )
    index = RUN._content_document(
        {
            "schema_version": 1,
            "classification": "retrospective_fixed_i3_pretest_index",
            "status": "complete",
            "completed_units": 18,
            "matrix": {"folds": list(RUN.FOLDS), "seeds": list(RUN.SEEDS), "unit_count": 18},
            "selected_preset": "default",
            "selected_parameter_count": 195_603,
            "capacity_reselected": False,
            "common_policy_reselected": False,
            "validation_scores_control_execution": False,
            "common": {
                "selection_lock": selection,
                "capacity_selection": capacity,
                "policy": policy,
                "source_freeze_manifest": freeze,
            },
            "units": units,
            "outer_test_opened": False,
            "outer_test_artifact_count": 0,
            "commercial_claim_authorized": False,
        }
    )
    path = tmp_path / "pretest_index.json"
    path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    return path


def test_freeze_spec_is_immutable_target_free_and_source_bound(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    first = RUN.freeze_spec(path)
    second = RUN.freeze_spec(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert first == second
    assert document["content_sha256"] == RUN.canonical_sha256(
        {key: value for key, value in document.items() if key != "content_sha256"}
    )
    assert document["matrix"]["unit_count"] == 18
    assert document["matrix"]["integrated_pass_only_after_all_18"] is True
    assert document["target_or_reference_value_artifact_consulted"] is False
    assert document["target_or_reference_arrays_deserialized"] is False
    assert document["test_prediction_artifact_opened"] is False
    assert document["commercial_performance_claim_authorized"] is False
    assert document["prospective_cohort_required_for_commercial_claim"] is True
    assert set(document["runtime_sources"]) >= {
        "orchestrator",
        "unit_verifier",
        "python_executable",
    }


def test_pretest_validator_hashes_reference_outputs_without_deserializing_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = _synthetic_pretest_index(tmp_path)

    def forbidden_np_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("reference/test NPZ payload must not be deserialized")

    monkeypatch.setattr(RUN.np, "load", forbidden_np_load)
    index, binding, records = RUN.validate_pretest_matrix(index_path)
    assert index["outer_test_opened"] is False
    assert binding["sha256"] == RUN.sha256_file(index_path)
    assert len(records) == 18
    assert [(row["outer_fold"], row["seed"]) for row in records] == [
        (fold, seed) for fold in RUN.FOLDS for seed in RUN.SEEDS
    ]
    assert all(row["target_or_reference_arrays_deserialized"] is False for row in records)


def test_pretest_validator_fails_on_output_tree_tamper(tmp_path: Path) -> None:
    index_path = _synthetic_pretest_index(tmp_path)
    document = json.loads(index_path.read_text(encoding="utf-8"))
    unit_root = Path(document["units"][0]["output_root"])
    (unit_root / "validation_predictions.npz").write_bytes(b"tampered")
    with pytest.raises(RUN.StreamingCampaignError, match="file hash mismatch"):
        RUN.validate_pretest_matrix(index_path)


def test_gate_evaluation_covers_mandatory_latency_memory_parameters_and_spikes() -> None:
    result = RUN.evaluate_gates(
        _synthetic_report(cuda=True),
        {"parameter_count": 195_603, "spike_rate": 0.08},
        _spec(),
    )
    assert result["all_mandatory_pass"] is True
    assert result["all_applicable_engineering_pass"] is True
    assert result["unit_integrated_pass"] is True
    assert result["engineering"]["latency"]["cpu"]["pass"] is True
    assert result["engineering"]["latency"]["cuda"]["pass"] is True
    assert result["engineering"]["parameter_count"]["value"] == 195_603
    assert result["engineering"]["spike_rate_diagnostic"]["applicable"] is True


def test_hardware_gate_failure_is_reported_without_hiding_mandatory_pass() -> None:
    result = RUN.evaluate_gates(
        _synthetic_report(cpu_p99=401.0),
        {"parameter_count": 195_603, "spike_rate": None},
        _spec(),
    )
    assert result["all_mandatory_pass"] is True
    assert result["engineering"]["latency"]["cpu"]["pass"] is False
    assert result["engineering"]["latency"]["cpu"]["stride_pass"] is False
    assert result["engineering"]["spike_rate_diagnostic"]["pass"] is True
    assert result["unit_integrated_pass"] is False


def test_missing_one_window_or_radar_mask_fails_mandatory_gate() -> None:
    report = _synthetic_report()
    report["parity"]["cases"][-1].pop("mode")
    report["robustness"]["seven_nonempty_radar_masks"].pop()
    result = RUN.evaluate_gates(
        report,
        {"parameter_count": 195_603, "spike_rate": 0.08},
        _spec(),
    )
    assert result["mandatory"]["whole_chunk_one_window_parity"] is False
    assert result["mandatory"]["seven_nonempty_structural_radar_masks"] is False
    assert result["unit_integrated_pass"] is False


def test_runtime_argv_retargets_only_immutable_output_and_has_no_label_argument(
    tmp_path: Path,
) -> None:
    final = tmp_path / "final"
    stage = tmp_path / "stage"
    argv = [
        sys.executable,
        str(RUN.VERIFIER.__file__),
        "--run-dir",
        "/sealed/run",
        "--cache",
        "/sealed/cache",
        "--output-dir",
        str(final / "verification"),
        "--devices",
        "auto",
    ]
    runtime = RUN._runtime_argv(argv, final_root=final, stage_root=stage)
    assert runtime[runtime.index("--output-dir") + 1] == str(stage / "verification")
    options = [token.lower() for token in runtime if token.startswith("--")]
    assert not any("target" in token or "reference" in token for token in options)


def test_complete_seal_requires_all_18_and_never_authorizes_commercial_claim(
    tmp_path: Path,
) -> None:
    units = []
    receipts = []
    for fold in RUN.FOLDS:
        for seed in RUN.SEEDS:
            unit_id = f"outer_{fold}_seed_{seed}"
            units.append({"unit_id": unit_id, "outer_fold": fold, "seed": seed})
            root = tmp_path / "units" / unit_id
            (root / "verification").mkdir(parents=True)
            (root / "receipt.json").write_text("{}", encoding="utf-8")
            (root / "verification/deployment_verification.json").write_text(
                "{}", encoding="utf-8"
            )
            (root / "verification/deployment_verification.csv").write_text(
                "category,status\n", encoding="utf-8"
            )
            (root / "verification/deployment_verification.md").write_text(
                "# synthetic\n", encoding="utf-8"
            )
            receipts.append(
                {
                    "unit_id": unit_id,
                    "gates": {
                        "all_mandatory_pass": True,
                        "all_applicable_engineering_pass": True,
                        "unit_integrated_pass": True,
                    },
                }
            )
    plan = {
        "units": units,
        "freeze_spec": {"path": "freeze", "sha256": "a" * 64, "bytes": 1},
        "freeze_spec_content_sha256": "b" * 64,
        "pretest_index": {"path": "index", "sha256": "c" * 64, "bytes": 1},
    }
    with pytest.raises(RUN.StreamingCampaignError, match="18 receipts"):
        RUN._complete_seal(
            plan=plan,
            plan_binding={"path": "plan", "sha256": "d" * 64, "bytes": 1},
            output=tmp_path,
            receipts=receipts[:-1],
        )
    seal = RUN._complete_seal(
        plan=plan,
        plan_binding={"path": "plan", "sha256": "d" * 64, "bytes": 1},
        output=tmp_path,
        receipts=receipts,
    )
    assert seal["all_18_receipts_validated"] is True
    assert seal["integrated_pass"] is True
    assert seal["commercial_performance_claim_authorized"] is False
    assert seal["prospective_cohort_required_for_commercial_claim"] is True


def test_progress_never_reports_integrated_pass_before_complete_seal() -> None:
    receipts = [
        {
            "unit_id": f"u{index}",
            "content_sha256": "a" * 64,
            "gates": {"unit_integrated_pass": True},
        }
        for index in range(18)
    ]
    binding = {"path": "plan", "sha256": "b" * 64, "bytes": 1}
    assert RUN._progress(binding, receipts, sealed=False)["integrated_pass_reported"] is False
    assert RUN._progress(binding, receipts, sealed=True)["integrated_pass_reported"] is True


def test_cli_exposes_resume_limit_alias_and_dry_run() -> None:
    parser = RUN.build_parser()
    args = parser.parse_args(["run", "--max-units", "3", "--dry-run"])
    assert args.max_new_units == 3
    assert args.dry_run is True
    alias = parser.parse_args(["run", "--max-new-units", "4"])
    assert alias.max_new_units == 4
