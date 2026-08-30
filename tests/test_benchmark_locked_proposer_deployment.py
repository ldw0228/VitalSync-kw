from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_locked_proposer_deployment.py"
SPEC = importlib.util.spec_from_file_location("benchmark_locked_proposer_deployment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_python_launcher_binding_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    launcher = tmp_path / "venv-python"
    launcher.symlink_to(Path(sys.executable))
    binding = RUN.bind_python_launcher(launcher)
    assert binding["path"] == str(launcher.absolute())
    assert Path(binding["path"]) != launcher.resolve()
    assert binding["sha256"] == RUN.sha256_file(launcher.resolve())


def _write(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return RUN.bind_file(path)


def _json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return RUN.bind_file(path)


def _npz(path: Path, **arrays: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return RUN.bind_file(path)


def _fixture(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    freeze_root = tmp_path / "source_freeze"
    frozen_names = (
        "train_harmonic_set_snn.py",
        "harmonic_set_models.py",
        "harmonic_set_v2.yaml",
        "ADAPTIVE_CAMPAIGN_CONTRACT.json",
    )
    frozen_hashes = {}
    for name in frozen_names:
        path = freeze_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        frozen_hashes[name] = RUN.sha256_file(path)
    freeze = _json(
        freeze_root / "MANIFEST.json",
        {
            "schema_version": 1,
            "declared_before_any_i3_score": True,
            "outer_test_opened": False,
            "files": frozen_hashes,
        },
    )
    common_root = tmp_path / "common"
    capacity = _json(
        common_root / "capacity.json",
        {
            "outer_test_opened": False,
            "source_freeze_manifest_sha256": freeze["sha256"],
        },
    )
    policy = _json(common_root / "policy.json", {"outer_test_opened": False})
    selection = _json(
        common_root / "selection.json",
        {
            "outer_test_opened_before_lock": False,
            "capacity_selection_sha256": capacity["sha256"],
            "common_fallback_policy_sha256": policy["sha256"],
            "source_freeze": frozen_hashes,
        },
    )
    common = {
        "selection_lock": selection,
        "capacity_selection": capacity,
        "policy": policy,
        "source_freeze_manifest": freeze,
    }
    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text(
        "data:\n"
        "  radar_hz: 40.0\n"
        "  model_hz: 10.0\n"
        "  window_seconds: 4.0\n"
        "  respiration_band_hz: [0.08, 0.8]\n"
        "  rr_range_bpm: [6.0, 45.0]\n"
        "  range_pool: 2\n"
        "  fft_size: 256\n",
        encoding="utf-8",
    )
    proposer_source = tmp_path / "sealed_proposer_source.py"
    proposer_source.write_text("# sealed proposer model source\n", encoding="utf-8")
    stack_source = tmp_path / "sealed_stack_source.py"
    stack_source.write_text("# sealed stack provenance source\n", encoding="utf-8")

    units = []
    serving_checkpoints: dict[tuple[int, int], Path] = {}
    for fold in RUN.FOLDS:
        for seed in RUN.SEEDS:
            proposer_root = tmp_path / "proposers" / f"outer_{fold}_seed_{seed}"
            manifest = _json(
                proposer_root / "validation_manifest.json",
                {"fold": fold, "seed": seed, "prediction_partition_only": True},
            )
            run_config = _json(
                proposer_root / "run_config.json",
                {
                    "run_signature": f"signature-{fold}-{seed}",
                    "arguments": {
                        "config": str(pipeline_config.resolve()),
                        "identity_split_manifest": manifest["path"],
                    },
                    "split_authority": {
                        "split_manifest_file_sha256": manifest["sha256"],
                    },
                },
            )
            checkpoint = _write(
                proposer_root / "fold_0/snn_best.pt",
                f"serving proposer checkpoint {fold}/{seed}".encode(),
            )
            serving_checkpoints[(fold, seed)] = Path(checkpoint["path"])
            prediction_provenance = {
                "checkpoint_sha256": checkpoint["sha256"],
                "run_config_sha256": run_config["sha256"],
                "labels_forwarded_to_model": False,
                "source_hashes": {
                    str(proposer_source.resolve()): RUN.sha256_file(proposer_source)
                },
            }
            prediction = _npz(
                proposer_root / "snn_prediction_all_windows.npz",
                cache_index=np.asarray([fold], dtype=np.int64),
                checkpoint_sha256=np.asarray(checkpoint["sha256"]),
                run_config_sha256=np.asarray(run_config["sha256"]),
                provenance_json=np.asarray(json.dumps(prediction_provenance, sort_keys=True)),
                prediction=np.asarray([20.0 + fold], dtype=np.float32),
                reference_rr_bpm=np.asarray([99.0], dtype=np.float32),
            )
            source = {
                "role": "hcs_validation",
                "path": prediction["path"],
                "sha256": prediction["sha256"],
                "checkpoint": checkpoint["path"],
                "run_config": run_config["path"],
                "run_config_sha256": run_config["sha256"],
                "manifest": manifest["path"],
                "manifest_file_sha256": manifest["sha256"],
            }
            stack = _npz(
                tmp_path / "stacks" / f"outer_{fold}_seed_{seed}.npz",
                # These label arrays prove the orchestrator can use only the
                # provenance scalar without deserializing reference payloads.
                reference_rr_bpm=np.asarray([99.0], dtype=np.float32),
                provenance_json=np.asarray(
                    json.dumps(
                        {
                            "strict_nested": True,
                            "outer_test_opened": False,
                            "outer_fold": fold,
                            "seed": seed,
                            "source_code_sha256": {
                                str(stack_source.resolve()): RUN.sha256_file(stack_source)
                            },
                            "source_units": [source],
                        },
                        sort_keys=True,
                    )
                ),
            )
            hcs_root = tmp_path / "hcs" / f"outer_{fold}_seed_{seed}"
            hcs_checkpoint = _write(hcs_root / "best_checkpoint.pt", b"dormant hcs")
            scaler = _json(hcs_root / "scaler.json", {"center": [0], "scale": [1]})
            cache = _json(hcs_root / "cache_manifest.json", {"complete": True})
            fallback = _write(hcs_root / "fallback.csv", b"cache_index,prediction_bpm\n")
            run_manifest = _json(
                hcs_root / "run_manifest.json", {"outer_fold": fold, "seed": seed}
            )
            policy = _json(hcs_root / "fallback_policy.json", {"diagnostic": True})
            history = _json(hcs_root / "history.json", {"complete": True})
            source_bindings = {
                "trainer": {"sha256": frozen_hashes["train_harmonic_set_snn.py"]},
                "harmonic_set_model": {"sha256": frozen_hashes["harmonic_set_models.py"]},
                "campaign_config": {"sha256": frozen_hashes["harmonic_set_v2.yaml"]},
                "adaptive_campaign_contract": {
                    "sha256": frozen_hashes["ADAPTIVE_CAMPAIGN_CONTRACT.json"]
                },
            }
            lock = _json(
                hcs_root / "selection_lock.json",
                {
                    "schema_version": 1,
                    "outer_fold": fold,
                    "seed": seed,
                    "adaptive_iteration": 3,
                    "outer_test_not_opened_before_this_lock": True,
                    "checkpoint_sha256": hcs_checkpoint["sha256"],
                    "scaler_sha256": scaler["sha256"],
                    "cache_manifest_sha256": cache["sha256"],
                    "fallback_oof_sha256": fallback["sha256"],
                    "run_manifest_sha256": run_manifest["sha256"],
                    "policy_sha256": policy["sha256"],
                    "history_sha256": history["sha256"],
                    "source_bindings": source_bindings,
                },
            )
            units.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "status": "complete",
                    "artifacts": {
                        "selection_lock": lock,
                        "checkpoint": hcs_checkpoint,
                        "scaler": scaler,
                        "cache_manifest": cache,
                        "fallback_oof": fallback,
                        "run_manifest": run_manifest,
                        "original_policy": policy,
                        "history": history,
                        "strict_stack": stack,
                    },
                }
            )
    index = RUN._content_document(
        {
            "schema_version": 1,
            "classification": "retrospective_fixed_i3_pretest_index",
            "status": "complete",
            "matrix": {
                "folds": list(RUN.FOLDS),
                "seeds": list(RUN.SEEDS),
                "unit_count": 18,
            },
            "completed_units": 18,
            "common": common,
            "units": units,
            "outer_test_opened": False,
            "commercial_claim_authorized": False,
        }
    )
    index_path = tmp_path / "pretest_index.json"
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    spec_path = tmp_path / "freeze_spec.json"
    RUN.freeze_spec(spec_path)
    return {
        "index": index_path,
        "spec": spec_path,
        "output": tmp_path / "benchmark_output",
        "serving_checkpoints": serving_checkpoints,
    }


def _fake_worker(calls: list[list[str]], *, slow: bool = False) -> Callable[..., None]:
    def execute(argv: list[str], *, cwd: Path, log_path: Path) -> None:
        del cwd
        calls.append(list(argv))
        contract_path = Path(argv[argv.index("--contract") + 1])
        output_path = Path(argv[argv.index("--output") + 1])
        contract = json.loads(contract_path.read_text())
        spec = json.loads(Path(contract["freeze_spec"]["path"]).read_text())
        p99 = 300.0 if slow else 12.0
        checkpoint = contract["serving_proposer"]["checkpoint"]
        report: dict[str, Any] = {
            "schema_version": 1,
            "classification": "locked_proposer_deployment_unit_benchmark",
            "unit_id": contract["unit_id"],
            "outer_fold": contract["outer_fold"],
            "seed": contract["seed"],
            "freeze_spec": contract["freeze_spec"],
            "freeze_spec_content_sha256": contract["freeze_spec_content_sha256"],
            "runtime": contract["runtime"],
            "measurement_scope": "current_host_not_target_device",
            "commercial_performance_claim_authorized": False,
            "target_or_label_artifact_read": False,
            "target_or_label_arrays_deserialized": False,
            "target_or_label_values_used_for_measurement_or_selection": False,
            "serving_role": "hcs_validation",
            "model": {
                "checkpoint": checkpoint,
                "run_config": contract["serving_proposer"]["run_config"],
                "pipeline_config": contract["serving_proposer"]["pipeline_config"],
                "checkpoint_bytes": checkpoint["bytes"],
                "total_parameters": 1_000_000,
                "trainable_parameters": 999_000,
            },
            "cold": {
                "model_load_plus_first_resident_raw_window_inference_ms": 45.0,
            },
            "warm": {
                "total_latency": {"p50_ms": 10.0, "p95_ms": 11.0, "p99_ms": p99},
                "p50_ms": 10.0,
                "p95_ms": 11.0,
                "p99_ms": p99,
                "throughput_windows_per_second": 100.0,
            },
            "memory": {
                "cpu_process_peak_rss_bytes": 400_000_000,
                "cpu_peak_rss_caveat": spec["memory_caveats"]["cpu_peak_rss"],
                "cuda_peak_reserved_bytes": None,
            },
            "spike_activity": {
                "available": True,
                "target_or_label_used": False,
                "overall_rate": 0.08,
                "fields": {},
            },
        }
        report["engineering_gates"] = RUN._evaluate_unit_gates(
            report, spec["engineering_gates"]
        )
        output_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("synthetic isolated CPU worker\n", encoding="utf-8")

    return execute


def _run(paths: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return RUN.run_campaign(
        freeze_spec_path=paths["spec"],
        pretest_index=paths["index"],
        output_root=paths["output"],
        device_name="cpu",
        **kwargs,
    )


def test_freeze_spec_predeclares_all_engineering_gates_without_outer_inputs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spec.json"
    first = RUN.freeze_spec(path)
    spec = json.loads(path.read_text())
    assert first["target_or_label_artifact_consulted"] is False
    assert spec["matrix"]["unit_count"] == 18
    assert spec["matrix"]["best_unit_selection_allowed"] is False
    assert spec["engineering_gates"] == {
        "checkpoint_bytes_max": 50 * 1024 * 1024,
        "cpu_process_peak_rss_bytes_max": 2 * 1024**3,
        "cpu_raw_resident_warm_p99_ms_max": 250.0,
        "cuda_peak_reserved_bytes_max": 1024**3,
        "cuda_raw_resident_warm_p99_ms_max": 50.0,
        "p99_stride_budget_fraction_max": 0.10,
        "parameter_count_max": 5_000_000,
        "spike_rate_diagnostic_max": 0.20,
        "spike_rate_diagnostic_min": 0.01,
        "spike_rate_unavailable_policy": "reported_not_applicable_without_failure",
    }
    first_bytes = path.read_bytes()
    RUN.freeze_spec(path)
    assert path.read_bytes() == first_bytes


def test_dry_run_fixes_all_18_serving_proposers_and_never_selects_dormant_hcs(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    result = _run(paths, dry_run=True)
    assert result["status"] == "dry_run"
    assert result["completed_units"] == 0
    plan = json.loads((paths["output"] / "control/plan.json").read_text())
    assert plan["unit_count"] == 18
    assert plan["execution_order"] == [
        f"outer_{fold}_seed_{seed}" for fold in RUN.FOLDS for seed in RUN.SEEDS
    ]
    assert plan["best_unit_selection_allowed"] is False
    assert plan["measurement_scope"] == "current_host_not_target_device"
    for unit in plan["units"]:
        assert unit["serving_proposer"]["role"] == "hcs_validation"
        key = (unit["outer_fold"], unit["seed"])
        assert Path(unit["serving_proposer"]["checkpoint"]["path"]) == paths[
            "serving_checkpoints"
        ][key]
        assert unit["selection_basis"].endswith("no_metric_or_best_unit_selection")
        assert unit["serving_proposer"]["stack_reference_or_target_arrays_read"] is False
    assert not (paths["output"] / "complete_seal.json").exists()


def test_resume_max_new_units_and_complete_seal_require_all_18(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(RUN, "_run_worker_command", _fake_worker(calls))
    first = _run(paths, max_new_units=1)
    assert first["completed_units"] == first["new_units"] == 1
    assert len(calls) == 1
    resumed = _run(paths, max_new_units=0)
    assert resumed["completed_units"] == 1
    assert len(calls) == 1
    partial = _run(paths, max_new_units=16)
    assert partial["completed_units"] == 17
    assert not (paths["output"] / "complete_seal.json").exists()
    complete = _run(paths, max_new_units=1)
    assert complete["completed_units"] == 18
    seal_path = paths["output"] / "complete_seal.json"
    seal = json.loads(seal_path.read_text())
    assert seal["all_18_reported"] is True
    assert seal["best_unit_selection_performed"] is False
    assert seal["unit_ranking_performed"] is False
    assert seal["commercial_performance_claim_authorized"] is False
    assert len(seal["units"]) == 18
    sealed_bytes = seal_path.read_bytes()
    prior_calls = len(calls)
    again = _run(paths)
    assert again["new_units"] == 0
    assert len(calls) == prior_calls
    assert seal_path.read_bytes() == sealed_bytes


def test_gate_failure_is_reported_for_all_units_without_best_unit_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(RUN, "_run_worker_command", _fake_worker(calls, slow=True))
    result = _run(paths)
    assert result["completed_units"] == 18
    seal = json.loads((paths["output"] / "complete_seal.json").read_text())
    assert seal["all_applicable_engineering_gates_pass"] is False
    assert seal["best_unit_selection_performed"] is False
    assert all(unit["engineering_gates_pass"] is False for unit in seal["units"])


def test_checkpoint_or_receipt_drift_fails_before_any_new_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    first_checkpoint = paths["serving_checkpoints"][(0, RUN.SEEDS[0])]
    first_checkpoint.write_bytes(b"tampered serving checkpoint")
    calls: list[list[str]] = []
    monkeypatch.setattr(RUN, "_run_worker_command", _fake_worker(calls))
    with pytest.raises(RUN.LockedBenchmarkError, match="checkpoint hash mismatch"):
        _run(paths, max_new_units=1)
    assert calls == []

    fresh = _fixture(tmp_path / "receipt")
    fresh_calls: list[list[str]] = []
    monkeypatch.setattr(RUN, "_run_worker_command", _fake_worker(fresh_calls))
    _run(fresh, max_new_units=1)
    benchmark = (
        fresh["output"] / "units" / f"outer_0_seed_{RUN.SEEDS[0]}" / "benchmark.json"
    )
    benchmark.chmod(0o644)
    benchmark.write_text("{}", encoding="utf-8")
    with pytest.raises(RUN.LockedBenchmarkError, match="hash mismatch"):
        _run(fresh, max_new_units=1)
    assert len(fresh_calls) == 1


def test_cuda_dry_run_never_executes_gpu_on_cpu_test_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(RUN, "_run_worker_command", _fake_worker(calls))
    result = RUN.run_campaign(
        freeze_spec_path=paths["spec"],
        pretest_index=paths["index"],
        output_root=tmp_path / "cuda_plan",
        device_name="cuda:0",
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert calls == []
