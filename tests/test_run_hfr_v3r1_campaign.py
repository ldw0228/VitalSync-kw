from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGING_ONLY = (ROOT / "artifacts").is_symlink() or not (ROOT / "artifacts").is_dir()
EVIDENCE_ROOT = Path.cwd().resolve() if STAGING_ONLY else ROOT
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) in sys.path:
    sys.path.remove(str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS))
for _module_name in (
    "run_hfr_v3r1_discovery_campaign",
    "run_fixed_hfr_v3r1_oof_campaign",
    "select_hfr_v3r1_common_variant",
    "build_locked_hfr_v3r1_test_inputs",
):
    sys.modules.pop(_module_name, None)

import run_hfr_v3r1_discovery_campaign as campaign  # noqa: E402
import select_hfr_v3r1_common_variant as selector  # noqa: E402
import run_fixed_hfr_v3r1_oof_campaign as fixed_campaign  # noqa: E402
import build_locked_hfr_v3r1_test_inputs as sanitizer  # noqa: E402


def test_context1_active_lifecycle_benchmark_and_pretrain_paths_are_exact() -> None:
    assert campaign.TARGET_SEALED_LIFECYCLE_ROOT_RELATIVE.as_posix() == (
        "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "target_sealed_lifecycle_v8r4a_context1"
    )
    assert campaign.EFFICIENCY_BENCHMARK_RECEIPT_RELATIVE.as_posix() == (
        "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "efficiency_benchmark_v8r4a_context1/"
        "BENCHMARK_COMPLETION_RECEIPT_V8R4.json"
    )
    assert selector.PRETRAIN_AUTHORIZATION_RELATIVE.name == (
        "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"
    )
    trainer_source = (
        SCRIPTS / "train_harmonic_factor_router_snn_v3r1.py"
    ).read_text(encoding="utf-8")
    assert '"PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"' in trainer_source


def _context1_terminal(
    record_hash: str,
    *,
    phase: str,
    context: Mapping[str, Any],
    reusable: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "campaign_id": campaign.CAMPAIGN_ID,
        "event": "terminal",
        "phase": phase,
        "context": dict(context),
        "record_sha256": record_hash,
        "command_sha256": "a" * 64,
        "invocation_sha256": "b" * 64,
        "charged_usage_ns": 1,
        "elapsed_ns": 1,
        "return_code": 0 if reusable else 1,
        "wrapper_exit_code": 0 if reusable else 1,
        "hard_timeout_reached": False,
        "reuse_eligible": reusable,
    }


def _rootbind1_failure_terminal(project_root: Path) -> dict[str, Any]:
    expected = campaign.ROOTBIND1_BENCHMARK_FAILURE
    return {
        "schema_version": 2,
        "campaign_id": campaign.CAMPAIGN_ID,
        "event": "terminal",
        "phase": "efficiency_benchmark",
        "context": dict(campaign.ROOTBIND1_BENCHMARK_USAGE_IDENTITY),
        "record_sha256": expected["terminal_record_sha256"],
        "invocation_sha256": expected["invocation_sha256"],
        "command_sha256": expected["command_sha256"],
        "reservation_record_sha256": expected["reservation_record_sha256"],
        "charged_usage_ns": expected["charged_usage_ns"],
        "elapsed_ns": expected["charged_usage_ns"],
        "return_code": expected["return_code"],
        "wrapper_exit_code": expected["wrapper_exit_code"],
        "result_path": str(
            (project_root / str(expected["result_relative_path"])).resolve()
        ),
        "reuse_eligible": False,
        "containment_anomaly": False,
        "hard_timeout_reached": False,
        "reservation_deadline_breached": False,
        "termination_escalated": False,
    }


def _publish_test_terminal(
    admitted_command: Sequence[str],
    *,
    return_code: int,
    elapsed: float,
    timed_out: bool,
    publish_result: bool = True,
) -> dict[str, Any]:
    """Test-only adapter that emulates the V7 wrapper's durable transaction."""

    command = list(admitted_command)
    delimiter = command.index("--")
    workload = command[delimiter + 1 :]
    value = lambda name: command[command.index(name) + 1]
    usage_ledger = Path(value("--usage-ledger"))
    result_path = Path(value("--result-file"))
    gpu_ledger = Path(value("--ledger"))
    context = json.loads(value("--context-json"))
    now_realtime = time.time_ns()
    now_monotonic = time.monotonic_ns()
    pid = os.getpid()
    start_ticks = campaign.gpu_budget_ledger.process_start_ticks(pid)
    assert start_ticks is not None
    state = campaign.gpu_budget_ledger.verify_ledger(usage_ledger)
    reservation, _, _ = campaign.gpu_budget_ledger.reconcile_and_reserve(
        usage_ledger,
        {
            "lifecycle_id": f"test-lifecycle-{len(state.records):06d}",
            "campaign_id": value("--campaign-id"),
            "phase": value("--phase"),
            "context": context,
            "invocation_sha256": value("--invocation-sha256"),
            "command_sha256": campaign.semantic_sha256(workload),
            "result_path": str(result_path.resolve()),
            "gpu_execution_ledger_path": str(gpu_ledger.resolve()),
            "boot_id": campaign.gpu_budget_ledger.boot_id(),
            "wrapper_pid": pid,
            "wrapper_start_ticks": start_ticks,
            "realtime_ns": now_realtime,
            "monotonic_ns": now_monotonic,
        },
    )
    charged = int(round(elapsed * 1_000_000_000))
    wrapper_exit = 124 if timed_out else int(return_code)
    terminal = campaign.gpu_budget_ledger.append_terminal(
        usage_ledger,
        reservation,
        last_heartbeat=None,
        elapsed_ns=charged,
        charged_usage_ns=charged,
        realtime_ns=now_realtime + charged,
        monotonic_ns=now_monotonic + charged,
        child_pid=pid,
        child_start_ticks=start_ticks,
        return_code=int(return_code),
        wrapper_exit_code=wrapper_exit,
        hard_timeout_reached=bool(timed_out),
        received_signal=None,
        termination_escalated=False,
    )
    execution_common = {
        "schema_version": 1,
        "job_id": reservation["lifecycle_id"],
        "lifecycle_id": reservation["lifecycle_id"],
        "reservation_record_sha256": reservation["record_sha256"],
        "wrapper_pid": pid,
        "lock_file": value("--lock-file"),
        "usage_ledger": str(usage_ledger.resolve()),
        "result_file": str(result_path.resolve()),
        "campaign_id": value("--campaign-id"),
        "phase": value("--phase"),
        "context": context,
        "invocation_sha256": value("--invocation-sha256"),
        "command": workload,
        "command_sha256": campaign.semantic_sha256(workload),
    }
    gpu_ledger.parent.mkdir(parents=True, exist_ok=True)
    with gpu_ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {**execution_common, "event": "start", "utc": "test"},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        handle.write(
            json.dumps(
                {
                    **execution_common,
                    "event": "end",
                    "utc": "test",
                    "exit_code": int(return_code),
                    "wrapper_exit_code": wrapper_exit,
                    "hard_timeout_reached": bool(timed_out),
                    "terminal_record_sha256": terminal["record_sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    if publish_result:
        campaign.gpu_budget_ledger.atomic_result_receipt(
            result_path,
            campaign.gpu_budget_ledger.result_from_terminal(
                terminal,
                usage_ledger=usage_ledger,
                gpu_execution_ledger=gpu_ledger,
            ),
        )
    return terminal


def _supervised_test_runner(
    callback: Callable[[list[str], float], tuple[int, float, bool]],
) -> Callable[[Sequence[str], float], tuple[int, float, bool]]:
    def run(command: Sequence[str], timeout: float) -> tuple[int, float, bool]:
        outcome = callback(list(command), timeout)
        _publish_test_terminal(
            command,
            return_code=outcome[0],
            elapsed=outcome[1],
            timed_out=outcome[2],
        )
        return outcome

    return run


def _leave_abandoned_test_reservation(admitted_command: Sequence[str]) -> None:
    command = list(admitted_command)
    delimiter = command.index("--")
    workload = command[delimiter + 1 :]
    value = lambda name: command[command.index(name) + 1]
    usage_ledger = Path(value("--usage-ledger"))
    result_path = Path(value("--result-file"))
    gpu_ledger = Path(value("--ledger"))
    state = campaign.gpu_budget_ledger.verify_ledger(usage_ledger)
    campaign.gpu_budget_ledger.reconcile_and_reserve(
        usage_ledger,
        {
            "lifecycle_id": f"abandoned-test-{len(state.records):06d}",
            "campaign_id": value("--campaign-id"),
            "phase": value("--phase"),
            "context": json.loads(value("--context-json")),
            "invocation_sha256": value("--invocation-sha256"),
            "command_sha256": campaign.semantic_sha256(workload),
            "result_path": str(result_path.resolve()),
            "gpu_execution_ledger_path": str(gpu_ledger.resolve()),
            "boot_id": campaign.gpu_budget_ledger.boot_id(),
            "wrapper_pid": 999_999_999,
            "wrapper_start_ticks": 1,
            "realtime_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
        },
    )


def test_virtualenv_executable_path_preserves_final_symlink(tmp_path: Path) -> None:
    base = tmp_path / "base-python"
    base.write_bytes(b"python")
    venv = tmp_path / ".venv" / "bin"
    venv.mkdir(parents=True)
    link = venv / "python"
    link.symlink_to(base)
    observed = campaign.executable_path_without_symlink_dereference(
        tmp_path, Path(".venv/bin/python")
    )
    assert observed == link
    assert observed.is_symlink()
    assert observed.resolve() == base


def test_discovery_anonymous_publication_never_exposes_a_partial_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "seal.json"

    def fail_before_link(stage: str, _path: Path) -> None:
        if stage == "anonymous_fsynced":
            raise RuntimeError("stop-before-link")

    monkeypatch.setattr(campaign, "_PUBLICATION_FAULT_HOOK", fail_before_link)
    with pytest.raises(RuntimeError, match="stop-before-link"):
        campaign.create_once_json(path, {"classification": "test"})
    assert not path.exists()
    monkeypatch.setattr(campaign, "_PUBLICATION_FAULT_HOOK", None)
    campaign.create_once_json(path, {"classification": "test"})
    assert path.stat().st_nlink == 1
    assert path.stat().st_mode & 0o777 == 0o444


def test_discovery_staged_execution_recovers_exact_empty_tail(tmp_path: Path) -> None:
    root = tmp_path / "executions"
    (root / ".execution_000.staging").mkdir(parents=True)

    def create(path: Path) -> Mapping[str, Any]:
        return campaign.create_once_json(path, {"classification": "test"})

    def validate(path: Path) -> Mapping[str, Any]:
        value = campaign.load_json(path, "test staged invocation")
        assert value["classification"] == "test"
        return value

    final = campaign._publish_execution_directory(
        root,
        execution_number=0,
        create_invocation=create,
        validate_invocation=validate,
    )
    assert final == root / "execution_000"
    assert campaign._execution_directories(root) == [final]


def test_discovery_staged_execution_recovers_rename_before_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "executions"

    def create(path: Path) -> Mapping[str, Any]:
        return campaign.create_once_json(path, {"classification": "test"})

    def validate(path: Path) -> Mapping[str, Any]:
        return campaign.load_json(path, "test staged invocation")

    def killed_after_rename(stage: str, _path: Path) -> None:
        if stage == "indexed_directory_linked":
            raise RuntimeError("killed-after-directory-rename")

    monkeypatch.setattr(campaign, "_PUBLICATION_FAULT_HOOK", killed_after_rename)
    with pytest.raises(RuntimeError, match="killed-after-directory-rename"):
        campaign._publish_execution_directory(
            root,
            execution_number=0,
            create_invocation=create,
            validate_invocation=validate,
        )
    final = root / "execution_000"
    assert final.is_dir()
    monkeypatch.setattr(campaign, "_PUBLICATION_FAULT_HOOK", None)
    assert campaign._execution_directories(root) == [final]


def test_selector_uses_only_the_dedicated_discovery_aggregation_root() -> None:
    assert selector.DEFAULT_DISCOVERY_ROOT == campaign.AGGREGATION_OUTPUT_RELATIVE
    assert selector.DEFAULT_DISCOVERY_ROOT != campaign.DEFAULT_RUN_ROOT
    assert campaign.DISCOVERY_AGGREGATION_PHASE == "discovery_aggregation"
    source = inspect.getsource(selector._validate_discovery_seal)
    assert "allow_exact_historical_benchmark_prefix=True" in source


def test_discovery_aggregation_is_two_seal_and_unit_receipt_free() -> None:
    source = inspect.getsource(campaign.build_discovery_completion)
    assert "len(shard_seals) != 2" in source
    assert "locked_closed_snapshot(" in source
    assert "load_json(" not in source
    assert "verify_binding(" not in source
    assert "set(seal) == DISCOVERY_SHARD_SEAL_KEYS" in source
    assert "DISCOVERY_SHARD_SEAL_NAME" in source
    assert "DISCOVERY_AGGREGATION_PHASE" in inspect.getsource(campaign.main)


def test_discovery_aggregation_main_requests_only_canonical_capability_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, Any] = {}

    def validate(
        project_root: Path,
        *,
        capability_receipt: Path,
        expected_phase: str,
        expected_outer_fold: int | None,
    ) -> Mapping[str, Any]:
        observed.update(
            {
                "project_root": project_root,
                "capability_receipt": capability_receipt,
                "expected_phase": expected_phase,
                "expected_outer_fold": expected_outer_fold,
            }
        )
        raise campaign.CampaignError("stop after canonical capability scope")

    monkeypatch.setattr(campaign, "validate_pretrain_authorization", validate)
    capability = tmp_path / "lifecycle/TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json"
    status = campaign.main(
        [
            "--project-root",
            str(tmp_path),
            "--aggregate-shards",
            "--run-root",
            str(campaign.AGGREGATION_OUTPUT_RELATIVE),
            "--target-sealed-capability-receipt",
            str(capability),
        ]
    )
    assert status == 2
    assert observed == {
        "project_root": tmp_path.resolve(),
        "capability_receipt": capability.resolve(),
        "expected_phase": campaign.DISCOVERY_AGGREGATION_PHASE,
        "expected_outer_fold": None,
    }
    assert "stop after canonical capability scope" in capsys.readouterr().out


def test_discovery_main_rejects_alternate_gpu_lock_before_any_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    alternate = tmp_path / "alternate.lock"
    status = campaign.main(
        [
            "--project-root",
            str(tmp_path),
            "--gpu-lock",
            str(alternate),
            "--run-root",
            str(tmp_path / "must_not_exist"),
        ]
    )
    assert status == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed_closed"
    assert "fixed V8 campaign lock path" in result["error"]
    assert not (tmp_path / "must_not_exist").exists()


def test_training_index_override_cannot_replace_the_immutable_trust_anchor(
    tmp_path: Path,
) -> None:
    forged = tmp_path / "self_consistent_pretest_index.json"
    forged.write_bytes((ROOT / campaign.SHARD_TRAINING_INDEX[3]).read_bytes())
    forged.chmod(0o444)
    with pytest.raises(campaign.CampaignError, match="canonical immutable trust anchor"):
        campaign.load_training_index(ROOT, forged, outer_fold_shard=3)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _cache(tmp_path: Path) -> Path:
    root = tmp_path / "cache"
    root.mkdir(parents=True)
    (root / "metadata.csv").write_text(
        "cache_index,fold\n10,0\n11,1\n12,4\n13,4\n",
        encoding="utf-8",
    )
    _write_json(root / "manifest.json", {"complete": True})
    return root


def _stack(tmp_path: Path) -> Path:
    path = tmp_path / "stack.npz"
    np.savez_compressed(
        path,
        classification=np.asarray(campaign.NONOUTER_STACK_CLASSIFICATION),
        campaign_revision=np.asarray(campaign.CAMPAIGN_REVISION),
        partition=np.asarray("outer_excluded_training_validation"),
        cache_index=np.asarray([10, 11, 12, 13], np.int64),
        fold=np.asarray([0, 1, 4, 4], np.int16),
        prediction=np.asarray([20.0, 20.0, 20.0, 21.0], np.float32),
        rr_std=np.ones(4, np.float32),
        proposal_available=np.asarray([True, True, True, True]),
        nested_role=np.asarray(["training", "training", "validation", "validation"]),
        outer_fold=np.asarray(3, np.int16),
        seed=np.asarray(20260828, np.int64),
        outer_test_opened=np.asarray(False),
        outer_rows_present=np.asarray(False),
    )
    return path


def _metrics() -> dict[str, float]:
    return {
        "overall_mae_bpm": 0.9,
        "identity_macro_mae_bpm": 0.95,
        "rmse_bpm": 1.4,
        "within_2_fraction": 0.93,
        "over_5_fraction": 0.02,
        "high_rr_25_35_mae_bpm": 1.7,
    }


def _stub_training_outputs(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    scientific_signature = {
        "schema_version": 8,
        "campaign_id": campaign.CAMPAIGN_ID,
        "campaign_revision": campaign.CAMPAIGN_REVISION,
        "outer_fold": 3,
        "validation_fold": 4,
        "seed": 20260828,
        "variant": "H0_no_factor",
        "model": {"fixture": True},
        "optimization": {"fixture": True},
        "source_bindings": {"fixture": True},
        "input_bindings": {"fixture": True},
        "population": {"fixture": True},
        "batching_execution": {"padding_inert": True},
    }
    scientific_signature_sha256 = campaign.semantic_sha256(scientific_signature)
    manifest = {
        "campaign_revision": campaign.CAMPAIGN_REVISION,
        "outer_fold": 3,
        "validation_fold": 4,
        "seed": 20260828,
        "variant": "H0_no_factor",
        "outer_test_opened": False,
        "outer_test_predictions_constructed": False,
        "leakage_boundary": dict(campaign.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
        "parameter_count": 1234,
        "scientific_signature": scientific_signature,
        "scientific_signature_sha256": scientific_signature_sha256,
    }
    _write_json(output / "run_manifest.json", manifest)
    _write_json(output / "scaler.json", {"center": [0.0], "scale": [1.0]})
    _write_json(output / "history.json", {"epochs": [1]})
    (output / "last.pt").write_bytes(b"last")
    (output / "best.pt").write_bytes(b"best")
    metrics = {
        "classification": "adaptive_v3r1_v8r4_discovery_validation_only",
        "campaign_revision": campaign.CAMPAIGN_REVISION,
        "outer_test_rows_present": False,
        "release_modes": {
            mode: {"metrics": _metrics()} for mode in campaign.RELEASE_MODES
        },
    }
    _write_json(output / "validation_metrics.json", metrics)
    np.savez_compressed(
        output / "validation_predictions.npz",
        cache_index=np.asarray([12, 13], np.int64),
        reference_rr_bpm=np.asarray([20.0, 21.0], np.float32),
        reference_valid=np.asarray([True, True]),
        identity=np.asarray(["A", "B"]),
        raw_anchor_bpm=np.asarray([20.0, 21.0], np.float32),
        raw_anchor_available=np.asarray([True, True]),
        hard_source_bpm=np.asarray([20.1, 20.9], np.float32),
        hard_source_available=np.asarray([True, True]),
        fixed_confidence_switch_bpm=np.asarray([20.0, 20.9], np.float32),
        fixed_confidence_switch_available=np.asarray([True, True]),
        selected_source_probability=np.asarray([0.7, 0.9], np.float32),
        selected_source_code=np.asarray([0, 1], np.int16),
        source_scale_bpm=np.asarray([1.0, 1.0], np.float32),
        quality=np.asarray([0.9, 0.9], np.float32),
        factor_probabilities=np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], np.float32),
        spike_rate=np.asarray([0.1, 0.1], np.float32),
    )
    lock = {
        "campaign_revision": campaign.CAMPAIGN_REVISION,
        "outer_fold": 3,
        "validation_fold": 4,
        "seed": 20260828,
        "variant": "H0_no_factor",
        "leakage_boundary": dict(campaign.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
        "row_access_audit": {
            "campaign_revision": campaign.CAMPAIGN_REVISION,
            "outer_fold": 3,
            "physical_pack_rows": 4,
            "outer_rows_in_physical_pack": 0,
            "outer_row_access_attempts": 0,
            "implicit_whole_array_conversions": 0,
            "accesses_by_array": {"node_features": 1},
            "selected_rows_by_array": {"node_features": 2},
            "unique_accessed_cache_indexes": 2,
            "accessed_cache_indexes_sha256": "3" * 64,
        },
        "best_checkpoint_sha256": campaign.sha256_file(output / "best.pt"),
        "scaler_sha256": campaign.sha256_file(output / "scaler.json"),
        "history_sha256": campaign.sha256_file(output / "history.json"),
        "run_manifest_sha256": campaign.sha256_file(output / "run_manifest.json"),
        "scientific_signature_sha256": scientific_signature_sha256,
    }
    _write_json(output / "checkpoint_selection_lock.json", lock)


def _stub_prediction_outputs(
    output: Path, *, predict_input: Path, checkpoint: Path, scaler: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with np.load(predict_input, allow_pickle=False) as archive:
        index = np.asarray(archive["cache_index"], dtype=np.int64)
    rows = len(index)
    raw = np.full(rows, 20.0, np.float32)
    hard = np.full(rows, 21.0, np.float32)
    available = np.ones(rows, bool)
    np.savez_compressed(
        output / "predictions.npz",
        cache_index=index,
        prediction_bpm=raw,
        prediction_available=available,
        raw_anchor_bpm=raw,
        raw_anchor_available=available,
        hard_source_bpm=hard,
        hard_source_available=available,
        selected_source_probability=np.full(rows, 0.9, np.float32),
        selected_source_code=np.zeros(rows, np.int16),
        source_scale_bpm=np.ones(rows, np.float32),
        quality=np.ones(rows, np.float32),
        factor_probabilities=np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32), (rows, 1)
        ),
        spike_rate=np.full(rows, 0.1, np.float32),
    )
    _write_json(
        output / "prediction_manifest.json",
        {
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
            "release_mode": "raw_anchor",
            "target_fields_emitted": False,
            "commercial_claim_authorized": False,
            "predict_input_sha256": campaign.sha256_file(predict_input),
            "checkpoint_sha256": campaign.sha256_file(checkpoint),
            "scaler_sha256": campaign.sha256_file(scaler),
        },
    )


def _unit_arguments(tmp_path: Path) -> dict[str, Any]:
    cache = _cache(tmp_path)
    stack = _stack(tmp_path)
    trainer = tmp_path / "trainer.py"
    wrapper = tmp_path / "wrapper.py"
    python = tmp_path / "python"
    trainer.write_text("# trainer\n", encoding="utf-8")
    wrapper.write_text("# wrapper\n", encoding="utf-8")
    python.write_bytes(b"python")
    authorization_path = tmp_path / "auth.json"
    _write_json(authorization_path, {"training_authorized": True, "version": 8})
    authorization_path.chmod(0o444)
    capability_path = (
        tmp_path / "lifecycle" / campaign.TARGET_SEALED_CAPABILITY_NAME
    )
    _write_json(capability_path, {"test_only": True})
    capability_path.chmod(0o444)
    return {
        "project_root": tmp_path,
        "run_root": tmp_path / "run",
        "training_input": campaign.TrainingInput(
            outer_fold=3,
            seed=20260828,
            cache_dir=cache,
            cache_manifest_sha256=campaign.sha256_file(cache / "manifest.json"),
            proposer_stack=stack,
            proposer_stack_sha256=campaign.sha256_file(stack),
        ),
        "variant": "H0_no_factor",
        "authorization": {
            "authorization_binding": campaign.bind_file(authorization_path)
        },
        "contract_binding": {"path": "contract.json", "sha256": "b" * 64, "bytes": 1},
        "target_sealed_capability_receipt": capability_path,
        "python": python,
        "trainer": trainer,
        "wrapper": wrapper,
        "gpu_lock": tmp_path / campaign.DEFAULT_GPU_LOCK,
        "gpu_ledger": tmp_path / "gpu.jsonl",
        "usage_ledger": tmp_path / "usage.jsonl",
        "device": "cpu",
        "amp": False,
        "smoke_test": True,
    }


def test_discovery_revalidates_index_bound_proposer_before_first_write(
    tmp_path: Path,
) -> None:
    arguments = _unit_arguments(tmp_path)
    item = arguments["training_input"]
    assert isinstance(item, campaign.TrainingInput)
    stack_binding = campaign.verify_training_bound_file(
        ROOT,
        item.proposer_stack,
        expected_sha256=item.proposer_stack_sha256,
        expected_bytes=item.proposer_stack.stat().st_size,
    )
    arguments["training_input"] = campaign.TrainingInput(
        outer_fold=item.outer_fold,
        seed=item.seed,
        cache_dir=item.cache_dir,
        cache_manifest_sha256=item.cache_manifest_sha256,
        proposer_stack=item.proposer_stack,
        proposer_stack_sha256=item.proposer_stack_sha256,
        proposer_stack_binding=stack_binding,
    )
    arguments["project_root"] = ROOT
    arguments["gpu_lock"] = ROOT / campaign.DEFAULT_GPU_LOCK
    with item.proposer_stack.open("ab") as stream:
        stream.write(b"post-index mutation")
    launched = False

    def forbidden_runner(*_args: object, **_kwargs: object) -> tuple[int, float, bool]:
        nonlocal launched
        launched = True
        raise AssertionError("mutated proposer reached the workload launcher")

    with pytest.raises(campaign.CampaignError, match="verification failed"):
        campaign.run_training_unit(**arguments, command_runner=forbidden_runner)
    assert launched is False
    assert not arguments["run_root"].exists()


def test_discovery_unit_is_resumable_and_rehashes_completed_output(tmp_path: Path) -> None:
    arguments = _unit_arguments(tmp_path)
    calls: list[list[str]] = []

    def runner(command: list[str], timeout: float) -> tuple[int, float, bool]:
        calls.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        _stub_training_outputs(output)
        return 0, 2.5, False

    first = campaign.run_training_unit(
        **arguments, command_runner=_supervised_test_runner(runner)
    )
    assert first["validated_output"]["validation_rows"] == 2
    assert len(calls) == 1
    admitted = calls[0]
    trainer_command = admitted[admitted.index("--") + 1 :]
    assert Path(
        trainer_command[
            trainer_command.index("--target-sealed-capability-receipt") + 1
        ]
    ) == arguments["target_sealed_capability_receipt"]
    assert json.loads(
        trainer_command[
            trainer_command.index("--expected-admitted-context-json") + 1
        ]
    ) == json.loads(admitted[admitted.index("--context-json") + 1])

    def must_not_run(command: list[str], timeout: float) -> tuple[int, float, bool]:
        raise AssertionError("completed hash-valid unit was retrained")

    second = campaign.run_training_unit(**arguments, command_runner=must_not_run)
    assert second == first
    assert len(calls) == 1

    best = arguments["run_root"] / (
        "units/outer_3_seed_20260828_H0_no_factor/attempt_000/output/best.pt"
    )
    best.chmod(0o644)
    best.write_bytes(b"drift")
    with pytest.raises(campaign.CampaignError, match="bind best.pt|output drifted"):
        campaign.run_training_unit(**arguments, command_runner=must_not_run)


def test_discovery_recovers_terminal_before_receipt_gap_without_gpu_rerun(
    tmp_path: Path,
) -> None:
    arguments = _unit_arguments(tmp_path)
    launches = 0

    def terminal_then_parent_crash(
        command: Sequence[str], timeout: float
    ) -> tuple[int, float, bool]:
        nonlocal launches
        del timeout
        launches += 1
        command_list = list(command)
        output = Path(command_list[command_list.index("--output-dir") + 1])
        _stub_training_outputs(output)
        _publish_test_terminal(
            command_list, return_code=0, elapsed=2.5, timed_out=False
        )
        raise RuntimeError("simulated campaign crash after wrapper terminal")

    with pytest.raises(RuntimeError, match="simulated campaign crash"):
        campaign.run_training_unit(
            **arguments, command_runner=terminal_then_parent_crash
        )
    completion = (
        arguments["run_root"]
        / "units/outer_3_seed_20260828_H0_no_factor/completion_receipt.json"
    )
    assert not completion.exists()

    def must_not_run(command: Sequence[str], timeout: float) -> tuple[int, float, bool]:
        raise AssertionError("durable successful terminal was rerun")

    receipt = campaign.run_training_unit(
        **arguments, command_runner=must_not_run
    )
    assert launches == 1
    assert receipt["terminal_results"][0]["terminal_record_sha256"] == receipt[
        "usage_record_sha256"
    ]


def test_discovery_recovers_terminal_before_atomic_result_on_same_execution(
    tmp_path: Path,
) -> None:
    arguments = _unit_arguments(tmp_path)

    def terminal_without_result(
        command: Sequence[str], timeout: float
    ) -> tuple[int, float, bool]:
        del timeout
        command_list = list(command)
        trainer_command = command_list[command_list.index("--") + 1 :]
        assert Path(
            trainer_command[
                trainer_command.index("--target-sealed-capability-receipt") + 1
            ]
        ) == arguments["target_sealed_capability_receipt"]
        assert json.loads(
            trainer_command[
                trainer_command.index("--expected-admitted-context-json") + 1
            ]
        ) == json.loads(
            command_list[command_list.index("--context-json") + 1]
        )
        output = Path(command_list[command_list.index("--output-dir") + 1])
        _stub_training_outputs(output)
        _publish_test_terminal(
            command_list,
            return_code=0,
            elapsed=2.0,
            timed_out=False,
            publish_result=False,
        )
        raise RuntimeError("simulated crash before atomic result")

    with pytest.raises(RuntimeError, match="before atomic result"):
        campaign.run_training_unit(
            **arguments, command_runner=terminal_without_result
        )

    recovery_calls = 0

    def wrapper_recovery_only(
        command: Sequence[str], timeout: float
    ) -> tuple[int, float, bool]:
        nonlocal recovery_calls
        del timeout
        recovery_calls += 1
        command_list = list(command)
        value = lambda name: command_list[command_list.index(name) + 1]
        usage_ledger = Path(value("--usage-ledger"))
        result_path = Path(value("--result-file"))
        gpu_ledger = Path(value("--ledger"))
        state = campaign.gpu_budget_ledger.verify_ledger(usage_ledger)
        matches = [
            record
            for record in state.records
            if record.get("schema_version") == 2
            and record.get("event") == "terminal"
            and record.get("result_path") == str(result_path.resolve())
        ]
        assert len(matches) == 1
        campaign.gpu_budget_ledger.atomic_result_receipt(
            result_path,
            campaign.gpu_budget_ledger.result_from_terminal(
                matches[0],
                usage_ledger=usage_ledger,
                gpu_execution_ledger=gpu_ledger,
            ),
        )
        return 0, 0.0, False

    receipt = campaign.run_training_unit(
        **arguments, command_runner=wrapper_recovery_only
    )
    executions = (
        arguments["run_root"]
        / "units/outer_3_seed_20260828_H0_no_factor/attempt_000/executions"
    )
    assert recovery_calls == 1
    assert [path.name for path in executions.iterdir()] == ["execution_000"]
    assert receipt["usage_record_sha256s"] == [receipt["usage_record_sha256"]]


def test_reconciled_crash_lifecycle_and_success_share_exact_invocation_cover(
    tmp_path: Path,
) -> None:
    arguments = _unit_arguments(tmp_path)

    def abandon_wrapper(
        command: Sequence[str], timeout: float
    ) -> tuple[int, float, bool]:
        del timeout
        _leave_abandoned_test_reservation(command)
        raise RuntimeError("simulated dead wrapper")

    with pytest.raises(RuntimeError, match="dead wrapper"):
        campaign.run_training_unit(**arguments, command_runner=abandon_wrapper)

    def recovered_execution(
        command: list[str], timeout: float
    ) -> tuple[int, float, bool]:
        del timeout
        output = Path(command[command.index("--output-dir") + 1])
        _stub_training_outputs(output)
        return 0, 1.0, False

    receipt = campaign.run_training_unit(
        **arguments,
        command_runner=_supervised_test_runner(recovered_execution),
    )
    state = campaign._require_closed_usage_state(arguments["usage_ledger"])
    events = [
        record["event"]
        for record in state.records
        if record.get("schema_version") == 2
    ]
    assert events == ["reservation", "reconciled_terminal", "reservation", "terminal"]
    assert len(receipt["usage_record_sha256s"]) == 2
    assert len(receipt["terminal_results"]) == 1
    assert len(receipt["lifecycle_invocations"]) == 1


def test_completed_reuse_rejects_any_open_budget_reservation(tmp_path: Path) -> None:
    arguments = _unit_arguments(tmp_path)

    def runner(command: list[str], timeout: float) -> tuple[int, float, bool]:
        trainer_command = command[command.index("--") + 1 :]
        assert Path(
            trainer_command[
                trainer_command.index("--target-sealed-capability-receipt") + 1
            ]
        ) == arguments["target_sealed_capability_receipt"]
        assert json.loads(
            trainer_command[
                trainer_command.index("--expected-admitted-context-json") + 1
            ]
        ) == json.loads(command[command.index("--context-json") + 1])
        output = Path(command[command.index("--output-dir") + 1])
        _stub_training_outputs(output)
        return 0, 1.0, False

    campaign.run_training_unit(
        **arguments, command_runner=_supervised_test_runner(runner)
    )
    pid = os.getpid()
    start_ticks = campaign.gpu_budget_ledger.process_start_ticks(pid)
    assert start_ticks is not None
    campaign.gpu_budget_ledger.reconcile_and_reserve(
        arguments["usage_ledger"],
        {
            "lifecycle_id": "test-open-reservation",
            "campaign_id": campaign.CAMPAIGN_ID,
            "phase": "promotion_prediction",
            "context": {"outer_fold": 0, "seed": 20260828, "test": True},
            "invocation_sha256": "a" * 64,
            "command_sha256": "b" * 64,
            "result_path": str((tmp_path / "never.json").resolve()),
            "gpu_execution_ledger_path": str(arguments["gpu_ledger"].resolve()),
            "boot_id": campaign.gpu_budget_ledger.boot_id(),
            "wrapper_pid": pid,
            "wrapper_start_ticks": start_ticks,
            "realtime_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
        },
    )
    with pytest.raises(campaign.CampaignError, match="open reservations|not completely settled"):
        campaign.run_training_unit(**arguments, command_runner=runner)


def test_completed_discovery_reuse_requires_same_ledger_and_command(tmp_path: Path) -> None:
    arguments = _unit_arguments(tmp_path)

    def runner(command: list[str], timeout: float) -> tuple[int, float, bool]:
        output = Path(command[command.index("--output-dir") + 1])
        _stub_training_outputs(output)
        return 0, 2.5, False

    receipt = campaign.run_training_unit(
        **arguments, command_runner=_supervised_test_runner(runner)
    )
    assert receipt["usage_ledger_path"] == str(arguments["usage_ledger"].resolve())
    assert receipt["usage_record_sha256s"] == [receipt["usage_record_sha256"]]

    switched = {**arguments, "usage_ledger": tmp_path / "fresh-usage.jsonl"}
    with pytest.raises(
        campaign.CampaignError,
        match="immutable artifact collision|different GPU usage ledger",
    ):
        campaign.run_training_unit(**switched, command_runner=runner)

    changed_command = {**arguments, "gpu_ledger": tmp_path / "other-gpu.jsonl"}
    with pytest.raises(
        campaign.CampaignError, match="different GPU execution ledger|command differs"
    ):
        campaign.run_training_unit(**changed_command, command_runner=runner)

    unit = arguments["run_root"] / "units/outer_3_seed_20260828_H0_no_factor"
    output = unit / "attempt_000/output"
    unit_invocation = unit / "attempt_000/invocation.json"
    execution_invocation = unit / "attempt_000/executions/execution_001/invocation.json"
    terminal_result = execution_invocation.parent / "terminal_result.json"
    context = campaign._execution_context(
        {
            "campaign_revision": campaign.CAMPAIGN_REVISION,
            "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
        },
        execution_number=1,
        resume=True,
    )
    trainer_command = campaign._trainer_command(
        python=arguments["python"],
        trainer=arguments["trainer"],
        training_input=arguments["training_input"],
        output_dir=output,
        target_sealed_capability_receipt=arguments[
            "target_sealed_capability_receipt"
        ],
        expected_admitted_context=context,
        variant="H0_no_factor",
        device="cpu",
        amp=False,
        smoke_test=True,
        resume=True,
    )
    campaign._create_execution_invocation(
        execution_invocation,
        phase="discovery",
        context=context,
        unit_invocation_path=unit_invocation,
        workload_command=trainer_command,
    )
    command = campaign._admitted_command(
        python=arguments["python"],
        wrapper=arguments["wrapper"],
        gpu_lock=arguments["gpu_lock"],
        gpu_ledger=arguments["gpu_ledger"],
        usage_ledger=arguments["usage_ledger"],
        result_file=terminal_result,
        phase="discovery",
        context=context,
        invocation_sha256=campaign.sha256_file(execution_invocation),
        authorization_path=Path(
            arguments["authorization"]["authorization_binding"]["path"]
        ),
        authorization_sha256=arguments["authorization"][
            "authorization_binding"
        ]["sha256"],
        trainer_command=trainer_command,
    )
    _publish_test_terminal(command, return_code=9, elapsed=1.0, timed_out=False)
    with pytest.raises(campaign.CampaignError, match="history differs"):
        campaign.run_training_unit(**arguments, command_runner=runner)


def test_interrupted_discovery_unit_resumes_same_immutable_attempt(tmp_path: Path) -> None:
    arguments = _unit_arguments(tmp_path)
    calls: list[list[str]] = []

    def fail_once(command: list[str], timeout: float) -> tuple[int, float, bool]:
        calls.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "last.pt").write_bytes(b"partial")
        return 9, 1.0, False

    with pytest.raises(campaign.CampaignError, match="trainer failed"):
        campaign.run_training_unit(
            **arguments, command_runner=_supervised_test_runner(fail_once)
        )

    switched = {**arguments, "usage_ledger": tmp_path / "reset-usage.jsonl"}
    with pytest.raises(campaign.CampaignError, match="immutable artifact collision"):
        campaign.run_training_unit(**switched, command_runner=fail_once)

    def resume(command: list[str], timeout: float) -> tuple[int, float, bool]:
        calls.append(command)
        assert "--resume" in command
        output = Path(command[command.index("--output-dir") + 1])
        _stub_training_outputs(output)
        return 0, 1.0, False

    receipt = campaign.run_training_unit(
        **arguments, command_runner=_supervised_test_runner(resume)
    )
    assert receipt["validated_output"]["validation_fold"] == 4
    assert len(receipt["usage_record_sha256s"]) == 2
    assert len(calls) == 2
    unit = arguments["run_root"] / "units/outer_3_seed_20260828_H0_no_factor"
    assert (unit / "attempt_000/invocation.json").is_file()
    assert len(list((unit / "attempt_000/executions").glob("*/invocation.json"))) == 2


def test_hash_chained_gpu_ledger_detects_drift_and_hard_cap(tmp_path: Path) -> None:
    ledger = tmp_path / "usage.jsonl"
    campaign.append_usage_record(
        ledger, {"schema_version": 1, "elapsed_seconds": 10.0, "event": "one"}
    )
    campaign.append_usage_record(
        ledger, {"schema_version": 1, "elapsed_seconds": 20.0, "event": "two"}
    )
    _, elapsed = campaign._verify_usage_chain(ledger)
    assert elapsed == 30.0
    lines = ledger.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["elapsed_seconds"] = 11.0
    lines[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    ledger.chmod(0o644)
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="hash drifted"):
        campaign._verify_usage_chain(ledger)


def test_historical_usage_binding_must_be_exact_active_prefix(tmp_path: Path) -> None:
    ledger = tmp_path / "usage.jsonl"
    terminal = campaign.append_usage_record(
        ledger, {"schema_version": 1, "elapsed_seconds": 10.0, "event": "historical"}
    )
    binding = campaign.bind_file(ledger)
    campaign.append_usage_record(
        ledger, {"schema_version": 1, "elapsed_seconds": 20.0, "event": "later"}
    )
    campaign.verify_usage_ledger_prefix_binding(
        ledger,
        binding,
        project_root=tmp_path,
        owner=tmp_path / "seal.json",
        terminal_record_sha256=terminal["record_sha256"],
    )
    drifted = {**binding, "sha256": "0" * 64}
    with pytest.raises(campaign.CampaignError, match="prefix hash drifted"):
        campaign.verify_usage_ledger_prefix_binding(
            ledger,
            drifted,
            project_root=tmp_path,
            owner=tmp_path / "seal.json",
            terminal_record_sha256=terminal["record_sha256"],
        )


def test_usage_reconciliation_allows_governed_carry_forward_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "usage.jsonl"
    campaign.append_usage_record(
        ledger,
        {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "phase": "quarantine_carry_forward",
            "event": "forced_termination_usage_carry_forward",
            "elapsed_seconds": 377.0,
            "return_code": None,
            "return_code_observed": False,
            "termination_signal": "SIGTERM",
            "hard_timeout_reached": False,
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
            "quarantined": True,
            "training_result_eligible_for_reuse": False,
        },
    )
    successful = campaign.append_usage_record(
        ledger,
        {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "phase": "discovery",
            "campaign_revision": campaign.CAMPAIGN_REVISION,
            "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
            "resume": False,
            "command_sha256": "a" * 64,
            "elapsed_seconds": 2.0,
            "return_code": 0,
            "hard_timeout_reached": False,
        },
    )
    identity = {
        "campaign_revision": campaign.CAMPAIGN_REVISION,
        "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    ledger_state = campaign.gpu_budget_ledger.verify_ledger(ledger)
    terminal_record = {
        "schema_version": 2,
        "campaign_id": campaign.CAMPAIGN_ID,
        "event": "terminal",
        "phase": "discovery",
        "context": dict(identity),
        "command_sha256": "a" * 64,
        "invocation_sha256": "c" * 64,
        "charged_usage_ns": 2_000_000_000,
        "return_code": 0,
        "hard_timeout_reached": False,
        "reuse_eligible": True,
        "record_sha256": successful["record_sha256"],
    }
    usage_state = SimpleNamespace(
        records=(ledger_state.records[0], terminal_record),
        raw_bytes=ledger.read_bytes(),
        open_reservations={},
        settled_usage_ns=379_000_000_000,
    )
    monkeypatch.setattr(
        campaign, "_validate_terminal_result_bindings", lambda *args, **kwargs: None
    )
    receipt = campaign.completion_usage_fields(
        ledger,
        final_record_sha256=successful["record_sha256"],
        expected_phase="discovery",
        expected_identity=identity,
        expected_command_sha256=lambda record: "a" * 64,
        usage_state=usage_state,
    )
    _, elapsed = campaign.reconcile_usage_ledger(
        ledger, [(receipt, "discovery", identity)], usage_state=usage_state
    )
    assert elapsed == 379.0

    campaign.append_usage_record(
        ledger,
        {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "phase": "promotion_training",
            "outer_fold": 0,
            "seed": 20260828,
            "variant": "H0_no_factor",
            "resume": False,
            "command_sha256": "b" * 64,
            "elapsed_seconds": 1.0,
            "return_code": 0,
            "hard_timeout_reached": False,
        },
    )
    foreign = {
        "schema_version": 2,
        "campaign_id": campaign.CAMPAIGN_ID,
        "event": "terminal",
        "phase": "promotion_training",
        "context": {
            "campaign_revision": campaign.CAMPAIGN_REVISION,
            "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
            "outer_fold": 0,
            "seed": 20260828,
            "variant": "H0_no_factor",
        },
        "command_sha256": "b" * 64,
        "invocation_sha256": "d" * 64,
        "charged_usage_ns": 1_000_000_000,
        "return_code": 0,
        "hard_timeout_reached": False,
        "record_sha256": "f" * 64,
    }
    drifted_state = SimpleNamespace(
        records=usage_state.records + (foreign,),
        raw_bytes=usage_state.raw_bytes,
        open_reservations={},
        settled_usage_ns=380_000_000_000,
    )
    with pytest.raises(campaign.CampaignError, match="unexplained training or prediction"):
        campaign.reconcile_usage_ledger(
            ledger,
            [(receipt, "discovery", identity)],
            usage_state=drifted_state,
        )


def test_global_reconciliation_owns_one_rootbind1_failure_between_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    historical = [
        _context1_terminal(
            digest,
            phase="efficiency_benchmark",
            context=campaign.HISTORICAL_BENCHMARK_USAGE_IDENTITY,
            reusable=False,
        )
        for digest in campaign.HISTORICAL_BENCHMARK_TERMINAL_SHA256S
    ]
    quarantine = _context1_terminal(
        campaign.V8R3_QUARANTINE_TERMINAL_SHA256,
        phase="discovery",
        context={
            "execution_number": 0,
            "outer_fold": 3,
            "resume": False,
            "seed": 20260828,
            "variant": "H0_no_factor",
        },
        reusable=True,
    )
    rootbind = _rootbind1_failure_terminal(campaign.PROJECT_ROOT)
    active_context = {
        **campaign.ROOTBIND1_BENCHMARK_USAGE_IDENTITY,
        "authorization_generation": "CONTEXT1",
    }
    active = _context1_terminal(
        "c" * 64,
        phase="efficiency_benchmark",
        context=active_context,
        reusable=True,
    )
    discovery_context = {
        "campaign_revision": campaign.CAMPAIGN_REVISION,
        "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
        "execution_number": 0,
        "resume": False,
    }
    discovered = _context1_terminal(
        "d" * 64,
        phase="discovery",
        context=discovery_context,
        reusable=True,
    )
    records = (*historical, quarantine, rootbind, active, discovered)
    state = SimpleNamespace(records=records, settled_usage_ns=123)
    monkeypatch.setattr(
        campaign,
        "validate_completion_receipt_usage",
        lambda _ledger, receipt, **_kwargs: list(receipt["owned"]),
    )
    specs = [
        ({"owned": [quarantine]}, "quarantine", {}),
        ({"owned": [active]}, "efficiency_benchmark", active_context),
        ({"owned": [discovered]}, "discovery", discovery_context),
    ]
    campaign.reconcile_usage_ledger(
        tmp_path / "usage.jsonl",
        specs,
        usage_state=state,
        allow_exact_historical_benchmark_prefix=True,
    )

    repeated = SimpleNamespace(
        records=(*historical, quarantine, rootbind, dict(rootbind), active, discovered),
        settled_usage_ns=123,
    )
    with pytest.raises(campaign.CampaignError, match="multiple completion|ownership|unexplained"):
        campaign.reconcile_usage_ledger(
            tmp_path / "usage.jsonl",
            specs,
            usage_state=repeated,
            allow_exact_historical_benchmark_prefix=True,
        )


def test_pack_free_reconciliation_orders_rootbind1_before_context1_and_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    historical_attempts = [
        {"terminal_result": {"terminal_record_sha256": digest}}
        for digest in campaign.HISTORICAL_BENCHMARK_TERMINAL_SHA256S
    ]
    active_context = {
        **campaign.ROOTBIND1_BENCHMARK_USAGE_IDENTITY,
        "authorization_generation": "CONTEXT1",
    }
    fake_benchmark = SimpleNamespace(
        HISTORICAL_BENCHMARK_ATTEMPTS=historical_attempts,
        BENCHMARK_PHASE="efficiency_benchmark",
        BENCHMARK_ID=campaign.ROOTBIND1_BENCHMARK_USAGE_IDENTITY["benchmark_id"],
        BENCHMARK_USAGE_IDENTITY=active_context,
    )
    monkeypatch.setattr(
        campaign, "load_efficiency_benchmark_module", lambda _root: fake_benchmark
    )
    historical = [
        _context1_terminal(
            digest,
            phase="efficiency_benchmark",
            context=campaign.HISTORICAL_BENCHMARK_USAGE_IDENTITY,
            reusable=False,
        )
        for digest in campaign.HISTORICAL_BENCHMARK_TERMINAL_SHA256S
    ]
    quarantine = _context1_terminal(
        campaign.V8R3_QUARANTINE_TERMINAL_SHA256,
        phase="discovery",
        context={
            "execution_number": 0,
            "outer_fold": 3,
            "resume": False,
            "seed": 20260828,
            "variant": "H0_no_factor",
        },
        reusable=True,
    )
    rootbind = _rootbind1_failure_terminal(tmp_path)
    active = _context1_terminal(
        "c" * 64,
        phase="efficiency_benchmark",
        context=active_context,
        reusable=True,
    )
    discoveries: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for number, (outer, seed, variant) in enumerate(campaign.EXPECTED_DISCOVERY_UNITS):
        context = {
            "campaign_revision": campaign.CAMPAIGN_REVISION,
            "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
            "outer_fold": outer,
            "seed": seed,
            "variant": variant,
            "execution_number": 0,
            "resume": False,
        }
        discoveries.append(
            _context1_terminal(
                f"{number + 1:064x}",
                phase="discovery",
                context=context,
                reusable=True,
            )
        )
        units.append({"outer_fold": outer, "seed": seed, "variant": variant})
    records = [*historical, quarantine, rootbind, active, *discoveries]
    state = SimpleNamespace(records=tuple(records), settled_usage_ns=456)
    shard_seals = [
        (tmp_path / "outer3.json", {"units": units[:9]}),
        (tmp_path / "outer4.json", {"units": units[9:]}),
    ]
    campaign._pack_free_discovery_usage_cover(
        project_root=tmp_path,
        usage_ledger=tmp_path / "usage.jsonl",
        shard_seals=shard_seals,
        usage_state=state,
    )

    reordered = SimpleNamespace(
        records=tuple([*historical, quarantine, active, rootbind, *discoveries]),
        settled_usage_ns=456,
    )
    with pytest.raises(campaign.CampaignError, match="ordering drifted"):
        campaign._pack_free_discovery_usage_cover(
            project_root=tmp_path,
            usage_ledger=tmp_path / "usage.jsonl",
            shard_seals=shard_seals,
            usage_state=reordered,
        )


def test_discovery_seal_reconciles_all_receipts_to_usage_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # V8R4 deliberately removed the monolithic 18-receipt aggregation API.
    # The aggregation process is pack-free and owns exactly two independently
    # issued capability-shard seals.
    with pytest.raises(campaign.CampaignError, match="exactly two shard seals"):
        campaign.build_discovery_completion(
            project_root=tmp_path,
            run_root=tmp_path / "discovery",
            shard_seals=[],
            contract_binding={"sha256": "d" * 64},
            authorization={"authorization_binding": {"sha256": "e" * 64}},
            usage_ledger=tmp_path / "usage.jsonl",
            gpu_ledger=tmp_path / "gpu.jsonl",
            gpu_lock=tmp_path / "gpu.lock",
        )
    return

    ledger = tmp_path / "usage.jsonl"
    campaign.append_usage_record(
        ledger,
        {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "phase": "quarantine_carry_forward",
            "event": "forced_termination_usage_carry_forward",
            "elapsed_seconds": 377.0,
            "return_code": None,
            "return_code_observed": False,
            "termination_signal": "SIGTERM",
            "hard_timeout_reached": False,
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
            "quarantined": True,
            "training_result_eligible_for_reuse": False,
        },
    )
    benchmark_identity = {
        "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    benchmark_usage = campaign.append_usage_record(
        ledger,
        {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "phase": "efficiency_benchmark",
            **benchmark_identity,
            "command_sha256": "b" * 64,
            "elapsed_seconds": 1.0,
            "return_code": 0,
            "hard_timeout_reached": False,
        },
    )
    benchmark_fields = campaign.completion_usage_fields(
        ledger,
        final_record_sha256=benchmark_usage["record_sha256"],
        expected_phase="efficiency_benchmark",
        expected_identity=benchmark_identity,
        expected_command_sha256=lambda record: "b" * 64,
    )
    benchmark_receipt_path = tmp_path / "BENCHMARK_COMPLETION_RECEIPT.json"
    benchmark_receipt = campaign.create_once_json(
        benchmark_receipt_path,
        {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            **benchmark_fields,
        },
    )
    benchmark_module = type(
        "BenchmarkFixture",
        (),
        {
            "BENCHMARK_PHASE": "efficiency_benchmark",
            "BENCHMARK_USAGE_IDENTITY": benchmark_identity,
        },
    )
    monkeypatch.setattr(
        campaign, "load_efficiency_benchmark_module", lambda project_root: benchmark_module
    )
    monkeypatch.setattr(
        campaign,
        "validate_pre_discovery_efficiency_benchmark",
        lambda **kwargs: benchmark_receipt,
    )
    run_root = tmp_path / "discovery"
    campaign.bind_run_usage_ledger(run_root, ledger, execution_scope="discovery")
    receipts = []
    for number, key in enumerate(campaign.EXPECTED_DISCOVERY_UNITS):
        command_hash = f"{number + 1:064x}"
        usage = campaign.append_usage_record(
            ledger,
            {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "phase": "discovery",
                "outer_fold": key[0],
                "seed": key[1],
                "variant": key[2],
                "resume": False,
                "command_sha256": command_hash,
                "elapsed_seconds": 1.0,
                "return_code": 0,
                "hard_timeout_reached": False,
            },
        )
        identity = {"outer_fold": key[0], "seed": key[1], "variant": key[2]}
        usage_fields = campaign.completion_usage_fields(
            ledger,
            final_record_sha256=usage["record_sha256"],
            expected_phase="discovery",
            expected_identity=identity,
            expected_command_sha256=lambda record, value=command_hash: value,
        )
        path = (
            run_root
            / "units"
            / f"outer_{key[0]}_seed_{key[1]}_{key[2]}"
            / "completion_receipt.json"
        )
        receipts.append(
            campaign.create_once_json(
                path,
                {
                    "schema_version": 1,
                    "campaign_id": campaign.CAMPAIGN_ID,
                    "outer_fold": key[0],
                    "seed": key[1],
                    "variant": key[2],
                    **usage_fields,
                },
            )
        )
    original_create_once = campaign.create_once_json
    append_started = threading.Event()
    append_finished = threading.Event()
    writer: threading.Thread | None = None
    intercepted = False

    def append_unexplained_terminal() -> None:
        append_started.set()
        campaign.gpu_budget_ledger.append_record(
            ledger,
            {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "phase": "promotion_prediction",
                "outer_fold": 0,
                "seed": 20260828,
                "variant": "H0_no_factor",
                "release_mode": "raw_anchor",
                "command_sha256": "9" * 64,
                "elapsed_seconds": 1.0,
                "return_code": 0,
                "hard_timeout_reached": False,
            },
        )
        append_finished.set()

    def create_with_barrier(path: Path, value: Any) -> dict[str, Any]:
        nonlocal writer, intercepted
        if path.name == "DISCOVERY_COMPLETION_SEAL.json" and not intercepted:
            intercepted = True
            writer = threading.Thread(target=append_unexplained_terminal)
            writer.start()
            assert append_started.wait(timeout=1.0)
            time.sleep(0.05)
            assert not append_finished.is_set()
        return original_create_once(path, value)

    monkeypatch.setattr(campaign, "create_once_json", create_with_barrier)
    seal = campaign.build_discovery_completion(
        project_root=tmp_path,
        run_root=run_root,
        receipts=receipts,
        contract_binding={"sha256": "d" * 64},
        authorization={"authorization_binding": {"sha256": "e" * 64}},
        training_index_binding={"sha256": "f" * 64},
        usage_ledger=ledger,
        gpu_ledger=tmp_path / "gpu.jsonl",
        gpu_lock=tmp_path / "gpu.lock",
        benchmark_receipt_path=benchmark_receipt_path,
    )
    assert seal["completed_units"] == 18
    assert seal["gpu_elapsed_seconds"] == 396.0
    assert seal["pre_discovery_efficiency_benchmark"]["excluded_from_selection"] is True
    assert writer is not None
    writer.join(timeout=1.0)
    assert append_finished.is_set()
    assert seal["gpu_usage_ledger"]["bytes"] < ledger.stat().st_size
    with pytest.raises(campaign.CampaignError, match="unexplained"):
        campaign.build_discovery_completion(
            project_root=tmp_path,
            run_root=run_root,
            receipts=receipts,
            contract_binding={"sha256": "d" * 64},
            authorization={"authorization_binding": {"sha256": "e" * 64}},
            training_index_binding={"sha256": "f" * 64},
            usage_ledger=ledger,
            gpu_ledger=tmp_path / "gpu.jsonl",
            gpu_lock=tmp_path / "gpu.lock",
            benchmark_receipt_path=benchmark_receipt_path,
        )

    _leave_abandoned_test_reservation(
        [
            "python",
            "wrapper",
            "--ledger",
            str(tmp_path / "gpu.jsonl"),
            "--usage-ledger",
            str(ledger),
            "--result-file",
            str(tmp_path / "unpublished-result.json"),
            "--campaign-id",
            campaign.CAMPAIGN_ID,
            "--phase",
            "promotion_prediction",
            "--context-json",
            "{}",
            "--invocation-sha256",
            "a" * 64,
            "--",
            "workload",
        ]
    )
    with pytest.raises(
        campaign.CampaignError, match="open reservations|not completely settled"
    ):
        campaign.build_discovery_completion(
            project_root=tmp_path,
            run_root=run_root,
            receipts=receipts,
            contract_binding={"sha256": "d" * 64},
            authorization={"authorization_binding": {"sha256": "e" * 64}},
            training_index_binding={"sha256": "f" * 64},
            usage_ledger=ledger,
            gpu_ledger=tmp_path / "gpu.jsonl",
            gpu_lock=tmp_path / "gpu.lock",
            benchmark_receipt_path=benchmark_receipt_path,
        )


def test_completed_promotion_training_reuse_is_ledger_bound(tmp_path: Path) -> None:
    arguments = _unit_arguments(tmp_path)
    promotion_authorization = tmp_path / "promotion_authorization.json"
    _write_json(promotion_authorization, {"training_authorized": True})
    selection = {
        "selected_variant": "H0_no_factor",
        "selected_release_mode": "raw_anchor",
        "selected_parameter_count": 1234,
    }
    governance = {
        "selection_lock": {"sha256": "c" * 64},
        "pretrain_authorization": arguments["authorization"][
            "authorization_binding"
        ],
    }

    keyword = {
        "run_root": tmp_path / "promotion",
        "item": arguments["training_input"],
        "variant": "H0_no_factor",
        "selection": selection,
        "governance": governance,
        "target_sealed_capability_receipt": arguments[
            "target_sealed_capability_receipt"
        ],
        "python": arguments["python"],
        "trainer": arguments["trainer"],
        "wrapper": arguments["wrapper"],
        "promotion_authorization": promotion_authorization,
        "gpu_lock": arguments["gpu_lock"],
        "gpu_ledger": arguments["gpu_ledger"],
        "usage_ledger": arguments["usage_ledger"],
        "device": "cpu",
        "amp": False,
        "smoke_test": True,
    }

    def terminal_then_parent_crash(
        command: Sequence[str], timeout: float
    ) -> tuple[int, float, bool]:
        del timeout
        command_list = list(command)
        trainer_command = command_list[command_list.index("--") + 1 :]
        assert Path(
            trainer_command[
                trainer_command.index("--target-sealed-capability-receipt") + 1
            ]
        ) == arguments["target_sealed_capability_receipt"]
        assert json.loads(
            trainer_command[
                trainer_command.index("--expected-admitted-context-json") + 1
            ]
        ) == json.loads(
            command_list[command_list.index("--context-json") + 1]
        )
        output = Path(command_list[command_list.index("--output-dir") + 1])
        _stub_training_outputs(output)
        _publish_test_terminal(
            command_list, return_code=0, elapsed=3.0, timed_out=False
        )
        raise RuntimeError("simulated promotion campaign crash")

    with pytest.raises(RuntimeError, match="simulated promotion campaign crash"):
        fixed_campaign._run_promotion_training(
            **keyword, command_runner=terminal_then_parent_crash
        )

    def must_not_run(command: Sequence[str], timeout: float) -> tuple[int, float, bool]:
        raise AssertionError("promotion terminal-result recovery reran GPU work")

    receipt = fixed_campaign._run_promotion_training(
        **keyword, command_runner=must_not_run
    )
    assert receipt["usage_record_sha256s"] == [receipt["usage_record_sha256"]]

    switched = {**keyword, "usage_ledger": tmp_path / "fresh-promotion-usage.jsonl"}
    with pytest.raises(
        campaign.CampaignError,
        match="immutable artifact collision|different GPU usage ledger",
    ):
        fixed_campaign._run_promotion_training(
            **switched, command_runner=must_not_run
        )


def test_completed_promotion_prediction_reuse_is_ledger_bound(tmp_path: Path) -> None:
    arguments = _unit_arguments(tmp_path)
    run_root = tmp_path / "promotion-prediction"
    training_output = (
        run_root / "training/outer_3_seed_20260828/attempt_000/output"
    )
    _stub_training_outputs(training_output)
    predict_input = tmp_path / "sanitized.npz"
    np.savez_compressed(predict_input, cache_index=np.asarray([100, 101], np.int64))
    input_receipt = tmp_path / "sanitized.receipt.json"
    _write_json(input_receipt, {"safe": True})
    promotion_authorization = tmp_path / "promotion_authorization.json"
    _write_json(promotion_authorization, {"promotion_authorized": True})
    selection = {
        "selected_variant": "H0_no_factor",
        "selected_release_mode": "raw_anchor",
    }

    def runner(command: list[str], timeout: float) -> tuple[int, float, bool]:
        trainer_command = command[command.index("--") + 1 :]
        assert Path(
            trainer_command[
                trainer_command.index("--target-sealed-capability-receipt") + 1
            ]
        ) == arguments["target_sealed_capability_receipt"]
        assert json.loads(
            trainer_command[
                trainer_command.index("--expected-admitted-context-json") + 1
            ]
        ) == json.loads(command[command.index("--context-json") + 1])
        output = Path(command[command.index("--output-dir") + 1])
        _stub_prediction_outputs(
            output,
            predict_input=predict_input,
            checkpoint=training_output / "best.pt",
            scaler=training_output / "scaler.json",
        )
        return 0, 1.5, False

    keyword = {
        "run_root": run_root,
        "item": arguments["training_input"],
        "training_receipt": {"content_sha256": "a" * 64},
        "predict_input": predict_input,
        "input_receipt": input_receipt,
        "selection": selection,
        "governance": {
            "selection_lock": {"sha256": "b" * 64},
            "pretrain_authorization": arguments["authorization"][
                "authorization_binding"
            ],
        },
        "target_sealed_capability_receipt": arguments[
            "target_sealed_capability_receipt"
        ],
        "python": arguments["python"],
        "trainer": arguments["trainer"],
        "wrapper": arguments["wrapper"],
        "promotion_authorization": promotion_authorization,
        "gpu_lock": arguments["gpu_lock"],
        "gpu_ledger": arguments["gpu_ledger"],
        "usage_ledger": arguments["usage_ledger"],
        "device": "cpu",
        "amp": False,
    }
    receipt = fixed_campaign._run_prediction(
        **keyword, command_runner=_supervised_test_runner(runner)
    )
    assert receipt["usage_record_sha256s"] == [receipt["usage_record_sha256"]]

    def must_not_run(command: list[str], timeout: float) -> tuple[int, float, bool]:
        raise AssertionError("completed prediction was rerun")

    assert (
        fixed_campaign._run_prediction(**keyword, command_runner=must_not_run)
        == receipt
    )
    switched = {**keyword, "usage_ledger": tmp_path / "fresh-prediction-usage.jsonl"}
    with pytest.raises(
        campaign.CampaignError,
        match="immutable artifact collision|different GPU usage ledger",
    ):
        fixed_campaign._run_prediction(**switched, command_runner=must_not_run)


def test_selection_key_and_stable_tie_rules() -> None:
    records = [
        {"outer_fold": fold, "seed": seed, "metrics": _metrics()}
        for fold in campaign.OUTER_RUNS
        for seed in campaign.SEEDS
    ]
    key = selector.global_selection_key(records, parameter_count=1234)
    assert key == (0, 0.0, 0.0, 0.95, 0.95, 0.9, 1234)
    candidates = [
        {
            "variant": variant,
            "release_mode": mode,
            "selection_key": list(key),
        }
        for variant in reversed(campaign.VARIANTS)
        for mode in reversed(campaign.RELEASE_MODES)
    ]
    ranked = selector.rank_candidates(candidates)
    assert (ranked[0]["variant"], ranked[0]["release_mode"]) == (
        "H0_no_factor",
        "raw_anchor",
    )
    with pytest.raises(campaign.CampaignError, match="duplicate"):
        selector.rank_candidates([candidates[0], candidates[0]])


def test_selector_fails_ledger_validation_before_any_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "usage.jsonl"
    campaign.append_usage_record(
        ledger, {"schema_version": 1, "event": "fixture", "elapsed_seconds": 1.0}
    )
    discovery_root = tmp_path / selector.DEFAULT_DISCOVERY_ROOT
    campaign.create_once_json(
        discovery_root / "DISCOVERY_COMPLETION_SEAL.json",
        {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "gpu_usage_ledger": campaign.bind_file(ledger),
        },
    )
    monkeypatch.setattr(
        selector.discovery,
        "validate_contract",
        lambda root: ({"discovery": {}}, {"sha256": "a" * 64}),
    )
    monkeypatch.setattr(
        selector.discovery,
        "validate_pretrain_authorization",
        lambda root: {"authorization_binding": {"sha256": "b" * 64}},
    )
    monkeypatch.setattr(
        selector,
        "_validate_discovery_seal",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            campaign.CampaignError("unexplained lifecycle")
        ),
    )
    ranked = False

    def must_not_rank(values: Any) -> list[dict[str, Any]]:
        nonlocal ranked
        ranked = True
        raise AssertionError("ranking ran before ledger validation")

    monkeypatch.setattr(selector, "rank_candidates", must_not_rank)
    with pytest.raises(campaign.CampaignError, match="unexplained lifecycle"):
        selector.select_common_variant(
            project_root=tmp_path,
            discovery_root=discovery_root,
            selection_lock_path=tmp_path / "selection.json",
            promotion_authorization_path=tmp_path / "promotion.json",
        )
    assert ranked is False
    assert not (tmp_path / "selection.json").exists()


def test_selector_creation_rejects_legacy_discovery_parent_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_parent = tmp_path / campaign.DEFAULT_RUN_ROOT
    canonical = tmp_path / campaign.AGGREGATION_OUTPUT_RELATIVE
    assert selector.canonical_discovery_aggregation_root(tmp_path) == canonical
    opened = False

    def must_not_open(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        nonlocal opened
        opened = True
        raise AssertionError("legacy parent was opened before canonical rejection")

    monkeypatch.setattr(selector.discovery, "load_json", must_not_open)
    with pytest.raises(
        campaign.CampaignError, match="canonical dedicated discovery aggregation root"
    ):
        selector.select_common_variant(
            project_root=tmp_path,
            discovery_root=legacy_parent,
            selection_lock_path=tmp_path / selector.DEFAULT_SELECTION_LOCK,
            promotion_authorization_path=(
                tmp_path / selector.DEFAULT_PROMOTION_AUTHORIZATION
            ),
        )
    assert opened is False


def test_selector_canonical_root_helper_rejects_every_nonaggregation_owner(
    tmp_path: Path,
) -> None:
    expected = (tmp_path / campaign.AGGREGATION_OUTPUT_RELATIVE).resolve()
    assert selector.canonical_discovery_aggregation_root(tmp_path) == expected
    assert (
        selector.canonical_discovery_aggregation_root(tmp_path, expected)
        == expected
    )
    with pytest.raises(
        campaign.CampaignError, match="canonical dedicated discovery aggregation root"
    ):
        selector.canonical_discovery_aggregation_root(
            tmp_path, tmp_path / campaign.DEFAULT_RUN_ROOT
        )


def test_validation_prediction_rejects_missing_duplicate_or_outer_index(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _stub_training_outputs(output)
    campaign.validate_training_output(
        output,
        outer_fold=3,
        seed=20260828,
        variant="H0_no_factor",
        cache_dir=_cache(tmp_path / "source"),
    )
    with np.load(output / "validation_predictions.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["cache_index"] = np.asarray([12, 12], np.int64)
    np.savez_compressed(output / "validation_predictions.npz", **arrays)
    with pytest.raises(campaign.CampaignError, match="exact validation-fold cover"):
        campaign.validate_training_output(
            output,
            outer_fold=3,
            seed=20260828,
            variant="H0_no_factor",
            cache_dir=tmp_path / "source/cache",
        )


def _validator_module():
    path = EVIDENCE_ROOT / "scripts/validate_hfr_v3r1_authorization.py"
    spec = importlib.util.spec_from_file_location("v3r1_authorization_campaign_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authorization_binding_is_owned_by_validator_and_defaults_are_clean_v7(
    tmp_path: Path,
) -> None:
    authorization_relative = Path(
        "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "PRETRAIN_AUTHORIZATION_V8R4A.json"
    )
    authorization = tmp_path / authorization_relative
    _write_json(authorization, {"training_authorized": True, "version": "V8R4A"})
    contract = tmp_path / campaign.CONTRACT_RELATIVE
    _write_json(contract, {"test_contract": True})
    validator = tmp_path / "scripts/validate_hfr_v3r1_authorization.py"
    validator.parent.mkdir(parents=True)
    validator.write_text(
        "from pathlib import Path\n"
        "import hashlib\n"
        f"PRETRAIN_AUTHORIZATION = Path({authorization_relative.as_posix()!r})\n"
        "def validate_pretrain(root):\n"
        "    path = root / PRETRAIN_AUTHORIZATION\n"
        "    return {\n"
        "        'valid': True,\n"
        "        'training_authorized': True,\n"
        "        'promotion_authorized': False,\n"
        "        'commercial_claim_authorized': False,\n"
        f"        'campaign_revision': {campaign.CAMPAIGN_REVISION!r},\n"
        f"        'infrastructure_revision': {campaign.INFRASTRUCTURE_REVISION!r},\n"
        f"        'contract_file_sha256': {campaign.CONTRACT_FILE_SHA256!r},\n"
        "        'pretrain_authorization_path': PRETRAIN_AUTHORIZATION.as_posix(),\n"
        "        'pretrain_authorization_file_sha256': "
        "hashlib.sha256(path.read_bytes()).hexdigest(),\n"
        "    }\n",
        encoding="utf-8",
    )
    result = campaign.validate_pretrain_authorization(tmp_path)
    assert result["authorization_binding"]["path"] == authorization_relative.as_posix()
    assert result["authorization_binding"]["sha256"] == campaign.sha256_file(
        authorization
    )

    assert campaign.DEFAULT_RUN_ROOT.name == "discovery_v8r4"
    assert campaign.DEFAULT_GPU_LOCK.name == "gpu_admission_v7.lock"
    assert campaign.DEFAULT_GPU_LEDGER.name == "gpu_execution_ledger_v7.jsonl"
    assert campaign.DEFAULT_USAGE_LEDGER.name == "campaign_gpu_usage_chain_v6.jsonl"
    assert "gpu_state_v8r4a/admission" in campaign.DEFAULT_GPU_LOCK.as_posix()
    assert "gpu_state_v8r4a/execution" in campaign.DEFAULT_GPU_LEDGER.as_posix()
    assert "gpu_state_v8r4a/usage" in campaign.DEFAULT_USAGE_LEDGER.as_posix()
    forbidden = (
        "discovery/",
        "gpu_admission.lock",
        "gpu_execution_ledger.jsonl",
        "campaign_gpu_usage_chain.jsonl",
    )
    rendered = "\n".join(
        str(path)
        for path in (
            campaign.DEFAULT_RUN_ROOT,
            campaign.DEFAULT_GPU_LOCK,
            campaign.DEFAULT_GPU_LEDGER,
            campaign.DEFAULT_USAGE_LEDGER,
        )
    )
    assert not any(token in rendered for token in forbidden)


def test_authorization_validator_fails_closed_before_pretrain_and_on_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if STAGING_ONLY:
        pytest.skip("standalone caller staging does not carry the immutable evidence tree")
    validator = _validator_module()
    implementation = validator.validate_phase(EVIDENCE_ROOT, "implementation")
    assert implementation["implementation_authorized"] is True
    monkeypatch.setattr(
        validator,
        "SOURCE_SNAPSHOT",
        validator.CAMPAIGN_DIR / "ABSENT_TEST_SOURCE_SNAPSHOT.json",
    )
    with pytest.raises(validator.AuthorizationError):
        validator.validate_pretrain(EVIDENCE_ROOT)

    validator = _validator_module()
    monkeypatch.setattr(validator, "CONTRACT_FILE_SHA256", "0" * 64)
    with pytest.raises(validator.AuthorizationError, match="contract byte hash drifted"):
        validator.validate_contract(EVIDENCE_ROOT)

    validator = _validator_module()
    ancestry = dict(validator.READ_ONLY_ANCESTRY)
    first = next(iter(ancestry))
    ancestry[first] = "0" * 64
    monkeypatch.setattr(validator, "READ_ONLY_ANCESTRY", ancestry)
    with pytest.raises(validator.AuthorizationError, match="bound file drifted"):
        validator.validate_immutable_evidence(EVIDENCE_ROOT)


def test_v7r2_authorization_uses_type_exact_nested_json_comparisons() -> None:
    validator = _validator_module()
    assert validator.exact_json_equal(False, 0) is False
    assert validator.exact_json_equal(True, 1) is False
    assert validator.exact_json_equal(10.0, 10) is False
    expected = validator._fixed_gpu_protocol()
    drifted = json.loads(json.dumps(expected))
    drifted["open_reservations_allowed_at_unit_or_campaign_completion"] = 0
    assert validator.exact_json_equal(drifted, expected) is False


def test_v7r2_runtime_prefix_summary_is_rederived_from_bound_bytes() -> None:
    validator = _validator_module()
    runtime = validator.validate_active_runtime_ledgers(EVIDENCE_ROOT)
    assert validator.verify_runtime_ledger_prefixes(EVIDENCE_ROOT, runtime) == runtime

    drifted = json.loads(json.dumps(runtime))
    drifted["usage_ledger"]["record_count"] += 1
    with pytest.raises(
        validator.AuthorizationError, match="runtime prefix derived summary drifted"
    ):
        validator.verify_runtime_ledger_prefixes(EVIDENCE_ROOT, drifted)

    drifted = json.loads(json.dumps(runtime))
    drifted["execution_ledger"]["exists"] = 0
    with pytest.raises(
        validator.AuthorizationError, match="runtime prefix derived summary drifted"
    ):
        validator.verify_runtime_ledger_prefixes(EVIDENCE_ROOT, drifted)


def test_v8_authorization_rejects_an_open_execution_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _validator_module()
    runtime = validator.validate_active_runtime_ledgers(EVIDENCE_ROOT)
    runtime["execution_ledger"]["open_lifecycle_count"] = 1
    monkeypatch.setattr(
        validator.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(runtime, sort_keys=True)
        ),
    )
    with pytest.raises(
        validator.AuthorizationError,
        match="active runtime ledger validation summary drifted",
    ):
        validator.validate_active_runtime_ledgers(EVIDENCE_ROOT)


def test_v8_validator_rejects_a_fabricated_admitted_child_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _validator_module()
    wrapper_path = EVIDENCE_ROOT / "scripts/run_gpu_admitted.py"
    wrapper_spec = importlib.util.spec_from_file_location(
        "v8_gpu_admitted_campaign_test", wrapper_path
    )
    assert wrapper_spec is not None and wrapper_spec.loader is not None
    wrapper = importlib.util.module_from_spec(wrapper_spec)
    sys.modules[wrapper_spec.name] = wrapper
    wrapper_spec.loader.exec_module(wrapper)
    monkeypatch.setattr(
        validator,
        "_load_gpu_admitted_validator",
        lambda _root: wrapper,
    )
    authorization = tmp_path / validator.PRETRAIN_AUTHORIZATION
    authorization.parent.mkdir(parents=True)
    authorization.write_text("{}\n", encoding="utf-8")
    authorization.chmod(0o444)
    with pytest.raises(
        validator.AuthorizationError,
        match="capability revalidation failed",
    ):
        validator.revalidate_admitted_child_binding(
            tmp_path,
            {
                "classification": "verified_v8_gpu_admitted_child_lifecycle",
                "phase": "discovery",
                "valid": True,
            },
        )

def _v8_selector_unit_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    for name in campaign.REQUIRED_TRAIN_OUTPUTS:
        path = root / name
        if name == "validation_metrics.json":
            _write_json(
                path,
                {"release_modes": {mode: _metrics() for mode in campaign.RELEASE_MODES}},
            )
        elif name.endswith(".json"):
            _write_json(path, {"fixture": name})
        else:
            path.write_bytes(f"fixture:{name}".encode("utf-8"))
    return {
        name: campaign.bind_file(root / name)
        for name in campaign.REQUIRED_TRAIN_OUTPUTS
    }


def _v8_selector_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    owner_present: bool = True,
    excluded: bool = True,
    drift_binding: bool = False,
    benchmark_after_discovery: bool = False,
    authorization_mismatch: bool = False,
) -> dict[str, Any]:
    return _v8r4_selector_fixture_impl(
        tmp_path,
        monkeypatch,
        owner_present=owner_present,
        excluded=excluded,
        drift_binding=drift_binding,
        benchmark_after_discovery=benchmark_after_discovery,
        authorization_mismatch=authorization_mismatch,
    )

    discovery_root = tmp_path / "discovery"
    usage_ledger = tmp_path / "usage.jsonl"
    execution_ledger = tmp_path / "execution.jsonl"
    gpu_lock = tmp_path / "gpu.lock"
    usage_raw = b"stable-v8-selector-ledger\n"
    usage_ledger.write_bytes(usage_raw)
    execution_ledger.write_bytes(b"")
    gpu_lock.write_bytes(b"")
    authorization_path = tmp_path / selector.PRETRAIN_AUTHORIZATION_RELATIVE
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    authorization_path.write_text('{"fixture":"v8-pretrain"}\n', encoding="utf-8")
    authorization_path.chmod(0o444)
    authorization_binding = campaign.bind_file(authorization_path)
    benchmark_authorization_binding = authorization_binding
    if authorization_mismatch:
        different_authorization = tmp_path / "DIFFERENT_PRETRAIN_AUTHORIZATION.json"
        different_authorization.write_text(
            '{"fixture":"different-pretrain"}\n', encoding="utf-8"
        )
        different_authorization.chmod(0o444)
        benchmark_authorization_binding = campaign.bind_file(
            different_authorization
        )

    invocation_path = tmp_path / "benchmark/execution_000/invocation.json"
    campaign.create_once_json(
        invocation_path,
        {
            "schema_version": 2,
            "campaign_id": campaign.CAMPAIGN_ID,
            "phase": "efficiency_benchmark",
            "context": {
                "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
                "outer_fold": 3,
                "seed": 20260828,
                "variant": "H0_no_factor",
            },
            "workload_command": ["python", "benchmark-worker"],
            "workload_command_sha256": campaign.semantic_sha256(
                ["python", "benchmark-worker"]
            ),
        },
    )
    benchmark_hash = "1" * 64
    benchmark_identity = {
        "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    benchmark_receipt_path = tmp_path / "benchmark/BENCHMARK_COMPLETION_RECEIPT.json"
    benchmark_receipt = campaign.create_once_json(
        benchmark_receipt_path,
        {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "phase": "efficiency_benchmark",
            "usage_identity": benchmark_identity,
            "pretrain_authorization": benchmark_authorization_binding,
            **benchmark_identity,
            "usage_ledger_path": str(usage_ledger.resolve()),
            "usage_record_sha256": benchmark_hash,
            "usage_record_sha256s": [benchmark_hash],
            "lifecycle_invocations": [campaign.bind_file(invocation_path)],
        },
    )

    discovery_records: list[dict[str, Any]] = []
    units = []
    receipts: dict[tuple[int, int, str], dict[str, Any]] = {}
    training_inputs: dict[tuple[int, int], Any] = {}
    validated_by_output: dict[Path, dict[str, Any]] = {}
    training_index_path = tmp_path / "canonical-training-index.json"
    _write_json(training_index_path, {"fixture": "canonical-training-index"})
    training_index_binding = campaign.bind_file(training_index_path)
    for number, key in enumerate(campaign.EXPECTED_DISCOVERY_UNITS):
        record_hash = f"{number + 2:064x}"
        discovery_records.append(
            {
                "record_sha256": record_hash,
                "phase": "discovery",
                "outer_fold": key[0],
                "seed": key[1],
                "variant": key[2],
            }
        )
        receipt_path = (
            discovery_root
            / "units"
            / f"outer_{key[0]}_seed_{key[1]}_{key[2]}"
            / "completion_receipt.json"
        )
        artifacts = _v8_selector_unit_artifacts(receipt_path.parent / "output")
        validated_output = {
            "parameter_count": 100 + campaign.VARIANTS.index(key[2]),
            "release_metrics": {
                mode: _metrics() for mode in campaign.RELEASE_MODES
            },
            "artifacts": artifacts,
        }
        validated_by_output[(receipt_path.parent / "output").resolve()] = validated_output
        training_inputs[(key[0], key[1])] = SimpleNamespace(
            cache_dir=tmp_path / f"cache-{key[0]}-{key[1]}"
        )
        receipt = campaign.create_once_json(
            receipt_path,
            {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "outer_test_opened": False,
                "outer_fold": key[0],
                "seed": key[1],
                "variant": key[2],
                "gpu_execution_ledger_path": str(execution_ledger.resolve()),
                "gpu_admission_lock_path": str(gpu_lock.resolve()),
                "usage_ledger_path": str(usage_ledger.resolve()),
                "usage_record_sha256": record_hash,
                "usage_record_sha256s": [record_hash],
                "validated_output": validated_output,
            },
        )
        receipts[key] = receipt
        units.append(
            {
                "outer_fold": key[0],
                "seed": key[1],
                "variant": key[2],
                "receipt": campaign.bind_file(receipt_path),
            }
        )

    benchmark_record = {
        "record_sha256": benchmark_hash,
        "phase": "efficiency_benchmark",
        "invocation_sha256": campaign.sha256_file(invocation_path),
        **benchmark_identity,
    }
    records = (
        [discovery_records[0], benchmark_record, *discovery_records[1:]]
        if benchmark_after_discovery
        else [benchmark_record, *discovery_records]
    )
    state = SimpleNamespace(records=tuple(records), raw_bytes=usage_raw)

    benchmark_binding = campaign.bind_file(benchmark_receipt_path)
    if drift_binding:
        benchmark_binding = {**benchmark_binding, "sha256": "0" * 64}
    seal: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": campaign.CAMPAIGN_ID,
        "contract": {"sha256": campaign.CONTRACT_FILE_SHA256},
        "completed_units": 18,
        "outer_test_opened": False,
        "outer_test_features_constructed": False,
        "outer_test_targets_accessed": False,
        "validation_targets_only": True,
        "ready_for_global_discovery_selection": True,
        "pretrain_authorization": authorization_binding,
        "training_index": training_index_binding,
        "gpu_elapsed_seconds": 19.0,
        "gpu_usage_ledger": {
            "path": str(usage_ledger.resolve()),
            "sha256": campaign.sha256_file(usage_ledger),
            "bytes": len(usage_raw),
        },
        "units": units,
    }
    if owner_present:
        seal["pre_discovery_efficiency_benchmark"] = {
            "receipt": benchmark_binding,
            "included_in_gpu_exact_cover": True,
            "excluded_from_selection": excluded,
            "artifacts_quarantined": True,
        }
    campaign.create_once_json(discovery_root / "DISCOVERY_COMPLETION_SEAL.json", seal)

    monkeypatch.setattr(
        campaign,
        "load_training_index",
        lambda _root, _path: (training_inputs, training_index_binding),
    )
    monkeypatch.setattr(
        campaign,
        "validate_training_output",
        lambda output, **_scope: validated_by_output[output.resolve()],
    )

    captured: dict[str, Any] = {}

    def validate_benchmark(
        module: Any,
        *,
        project_root: Path,
        receipt_path: Path,
        usage_ledger: Path,
        execution_ledger: Path,
        gpu_lock: Path,
        expected_command_sha256: Any,
        usage_state: Any,
    ) -> dict[str, Any]:
        assert module is campaign
        assert project_root == tmp_path
        captured["benchmark_call"] = {
            "receipt_path": receipt_path,
            "usage_ledger": usage_ledger,
            "execution_ledger": execution_ledger,
            "gpu_lock": gpu_lock,
            "prefix_records": list(usage_state.records),
            "expected_command": expected_command_sha256(benchmark_record),
        }
        return benchmark_receipt

    monkeypatch.setattr(
        selector,
        "_load_benchmark_module",
        lambda: SimpleNamespace(validate_benchmark_receipt=validate_benchmark),
    )

    def reconcile(
        path: Path,
        specs: Any,
        *,
        usage_state: Any,
        allow_exact_historical_benchmark_prefix: bool = False,
    ) -> tuple[list[Any], float]:
        assert path == usage_ledger
        assert allow_exact_historical_benchmark_prefix is True
        captured["receipt_specs"] = list(specs)
        return list(usage_state.records), 19.0

    monkeypatch.setattr(campaign, "reconcile_usage_ledger", reconcile)
    monkeypatch.setattr(
        campaign, "verify_usage_ledger_prefix_binding", lambda *args, **kwargs: None
    )
    return {
        "root": discovery_root,
        "usage_ledger": usage_ledger,
        "execution_ledger": execution_ledger,
        "gpu_lock": gpu_lock,
        "state": state,
        "receipts": receipts,
        "benchmark_receipt": benchmark_receipt,
        "authorization_binding": authorization_binding,
        "captured": captured,
    }


def _v8r4_selector_fixture_impl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    owner_present: bool,
    excluded: bool,
    drift_binding: bool,
    benchmark_after_discovery: bool,
    authorization_mismatch: bool,
) -> dict[str, Any]:
    """Build the exact pack-free two-shard selector authority graph."""

    discovery_root = tmp_path / "discovery"
    usage_ledger = tmp_path / "usage.jsonl"
    execution_ledger = tmp_path / "execution.jsonl"
    gpu_lock = tmp_path / "gpu.lock"
    raw = b"closed-v8r4a-selector-prefix\n"
    usage_ledger.write_bytes(raw)
    execution_ledger.write_bytes(b"")
    gpu_lock.write_bytes(b"")

    authorization_path = tmp_path / selector.PRETRAIN_AUTHORIZATION_RELATIVE
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    authorization_path.write_text('{"authority":"V8R4A"}\n', encoding="utf-8")
    authorization_path.chmod(0o444)
    authorization_binding = campaign.bind_file(
        authorization_path, relative_to=tmp_path
    )
    benchmark_authorization = authorization_binding
    if authorization_mismatch:
        alternate = tmp_path / "DIFFERENT_PRETRAIN_AUTHORIZATION_V8R4A.json"
        alternate.write_text('{"authority":"foreign"}\n', encoding="utf-8")
        alternate.chmod(0o444)
        benchmark_authorization = campaign.bind_file(alternate, relative_to=tmp_path)

    benchmark_hash = "1" * 64
    quarantine_hash = "2" * 64
    benchmark_identity = {
        "campaign_revision": campaign.CAMPAIGN_REVISION,
        "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
        "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    benchmark_path = tmp_path / "benchmark/BENCHMARK_COMPLETION_RECEIPT_V8R4.json"
    benchmark_receipt = campaign.create_once_json(
        benchmark_path,
        {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "campaign_revision": campaign.CAMPAIGN_REVISION,
            "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
            "phase": "efficiency_benchmark",
            "usage_identity": benchmark_identity,
            "pretrain_authorization": benchmark_authorization,
            "usage_record_sha256": benchmark_hash,
            "usage_record_sha256s": [benchmark_hash],
        },
    )
    benchmark_binding = campaign.bind_file(benchmark_path, relative_to=tmp_path)
    if drift_binding:
        benchmark_binding = {**benchmark_binding, "sha256": "0" * 64}

    quarantine_path = tmp_path / campaign.V8R3_QUARANTINE_RELATIVE
    quarantine_receipt = campaign.create_once_json(
        quarantine_path,
        {
            "schema_version": 1,
            "classification": "synthetic_v8r3_quarantine_owner",
            "campaign_id": campaign.CAMPAIGN_ID,
            "usage_record_sha256": quarantine_hash,
            "usage_record_sha256s": [quarantine_hash],
        },
    )
    quarantine_binding = campaign.bind_file(quarantine_path, relative_to=tmp_path)

    contract_binding = {
        "path": "contract.json",
        "sha256": campaign.CONTRACT_FILE_SHA256,
        "bytes": 1,
    }
    records: list[dict[str, Any]] = [
        {"record_sha256": benchmark_hash, "phase": "efficiency_benchmark", **benchmark_identity},
        {"record_sha256": quarantine_hash, "phase": "discovery_v8r3_quarantine"},
    ]
    receipts: dict[tuple[int, int, str], dict[str, Any]] = {}
    receipt_paths: dict[tuple[int, int, str], Path] = {}
    metrics_paths: dict[tuple[int, int, str], Path] = {}
    final_units: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    terminal_hash = ""
    for outer in campaign.OUTER_RUNS:
        shard_units: list[dict[str, Any]] = []
        for seed in campaign.SEEDS:
            for variant in campaign.VARIANTS:
                key = (outer, seed, variant)
                terminal_hash = f"{len(records) + 1:064x}"
                record = {
                    "record_sha256": terminal_hash,
                    "phase": "discovery",
                    "campaign_revision": campaign.CAMPAIGN_REVISION,
                    "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
                    "outer_fold": outer,
                    "seed": seed,
                    "variant": variant,
                }
                records.append(record)
                receipt_path = (
                    discovery_root
                    / f"outer_{outer}/units/outer_{outer}_seed_{seed}_{variant}"
                    / "completion_receipt.json"
                )
                output = receipt_path.parent / "output"
                metrics_path = output / "validation_metrics.json"
                metrics_document = campaign.create_once_json(
                    metrics_path,
                    {
                        "classification": "adaptive_v3r1_v8r4_discovery_validation_only",
                        "campaign_revision": campaign.CAMPAIGN_REVISION,
                        "outer_test_rows_present": False,
                        "release_modes": {
                            mode: {"metrics": _metrics()}
                            for mode in campaign.RELEASE_MODES
                        },
                    },
                )
                del metrics_document
                artifacts = {
                    name: (
                        campaign.bind_file(metrics_path, relative_to=tmp_path)
                        if name == "validation_metrics.json"
                        else {
                            "path": f"opaque/{outer}_{seed}_{variant}_{name}",
                            "sha256": "5" * 64,
                            "bytes": 1,
                        }
                    )
                    for name in campaign.REQUIRED_TRAIN_OUTPUTS
                }
                validated_output = {
                    "campaign_revision": campaign.CAMPAIGN_REVISION,
                    "outer_fold": outer,
                    "validation_fold": (outer + 1) % 6,
                    "seed": seed,
                    "variant": variant,
                    "parameter_count": 100 + campaign.VARIANTS.index(variant),
                    "validation_rows": 10,
                    "valid_reference_rows": 10,
                    "release_metrics": {
                        mode: _metrics() for mode in campaign.RELEASE_MODES
                    },
                    "scientific_signature_sha256": "8" * 64,
                    "physical_boundary": dict(
                        campaign.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION
                    ),
                    "row_access_audit": {
                        "campaign_revision": campaign.CAMPAIGN_REVISION,
                        "outer_fold": outer,
                        "physical_pack_rows": 10,
                        "outer_rows_in_physical_pack": 0,
                        "outer_row_access_attempts": 0,
                        "implicit_whole_array_conversions": 0,
                        "accesses_by_array": {"node_features": 10},
                        "selected_rows_by_array": {"node_features": 10},
                        "unique_accessed_cache_indexes": 10,
                        "accessed_cache_indexes_sha256": "9" * 64,
                    },
                    "artifacts": artifacts,
                }
                receipt = campaign.create_once_json(
                    receipt_path,
                    {
                        "schema_version": 1,
                        "classification": "adaptive_v3r1_v8r4_discovery_unit_completion",
                        "campaign_id": campaign.CAMPAIGN_ID,
                        "campaign_revision": campaign.CAMPAIGN_REVISION,
                        "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
                        "outer_test_opened": False,
                        "outer_fold": outer,
                        "validation_fold": (outer + 1) % 6,
                        "seed": seed,
                        "variant": variant,
                        "invocation": {"path": "opaque", "sha256": "6" * 64, "bytes": 1},
                        "usage_ledger_path": str(usage_ledger.resolve()),
                        "usage_record_sha256": terminal_hash,
                        "usage_record_sha256s": [terminal_hash],
                        "terminal_results": [{}],
                        "lifecycle_invocations": [{}],
                        "gpu_execution_ledger_path": str(execution_ledger.resolve()),
                        "gpu_admission_lock_path": str(gpu_lock.resolve()),
                        "validated_output": validated_output,
                        "commercial_claim_authorized": False,
                    },
                )
                unit = {
                    "outer_fold": outer,
                    "seed": seed,
                    "variant": variant,
                    "receipt": campaign.bind_file(receipt_path, relative_to=tmp_path),
                }
                receipts[key] = receipt
                receipt_paths[key] = receipt_path
                metrics_paths[key] = metrics_path
                shard_units.append(unit)
                final_units.append(unit)

        index_binding = dict(selector.SHARD_INDEX_BINDINGS[outer])
        shard_path = (
            discovery_root / f"outer_{outer}/DISCOVERY_SHARD_COMPLETION_SEAL.json"
        )
        shard = campaign.create_once_json(
            shard_path,
            {
                "schema_version": 1,
                "classification": "adaptive_v3r1_v8r4_discovery_capability_shard_seal",
                "campaign_id": campaign.CAMPAIGN_ID,
                "campaign_revision": campaign.CAMPAIGN_REVISION,
                "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
                "outer_fold_shard": outer,
                "contract": contract_binding,
                "pretrain_authorization": authorization_binding,
                "training_index": index_binding,
                "completed_units": 9,
                "peer_outer_shard_pack_mounted_or_opened": False,
                "combined_target_bearing_cache_opened": False,
                "outer_prediction_pack_absent": True,
                "physical_boundary": dict(campaign.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
                "gpu_usage_ledger_prefix": {
                    "path": str(usage_ledger.resolve()),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "terminal_record_sha256": terminal_hash,
                },
                "pre_discovery_efficiency_benchmark": benchmark_binding,
                "v8r3_quarantine_owner": quarantine_binding,
                "units": shard_units,
                "cross_outer_validation_reuse_present": True,
                "fully_nested_confirmatory_oof": False,
                "prospective_confirmation_required": True,
                "ready_for_pack_free_shard_aggregation": True,
                "commercial_claim_authorized": False,
            },
        )
        del shard
        shard_records.append(
            {
                "outer_fold": outer,
                "seal": campaign.bind_file(shard_path, relative_to=tmp_path),
                "training_index": index_binding,
            }
        )

    if benchmark_after_discovery:
        records[0], records[2] = records[2], records[0]
    state = SimpleNamespace(records=tuple(records), raw_bytes=raw, open_reservations={})
    benchmark_owner: Any = {
        "receipt": benchmark_binding,
        "included_in_gpu_exact_cover": True,
        "excluded_from_selection": excluded,
        "artifacts_quarantined": True,
    }
    if not owner_present:
        benchmark_owner = None
    seal_path = discovery_root / "DISCOVERY_COMPLETION_SEAL.json"
    seal = campaign.create_once_json(
        seal_path,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_v8r4_target_sealed_discovery_completion",
            "campaign_id": campaign.CAMPAIGN_ID,
            "campaign_revision": campaign.CAMPAIGN_REVISION,
            "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
            "contract": contract_binding,
            "pretrain_authorization": authorization_binding,
            "training_shards": shard_records,
            "outer_runs": list(campaign.OUTER_RUNS),
            "seeds": list(campaign.SEEDS),
            "variants": list(campaign.VARIANTS),
            "completed_units": 18,
            "physical_boundary": dict(campaign.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
            "validation_targets_only": True,
            "gpu_elapsed_seconds": 19.0,
            "gpu_hours_hard": campaign.GPU_HOURS_HARD,
            "gpu_usage_ledger": {
                "path": str(usage_ledger.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "terminal_record_sha256": terminal_hash,
                "open_reservations": 0,
            },
            "gpu_usage_ledger_path": str(usage_ledger.resolve()),
            "pre_discovery_efficiency_benchmark": benchmark_owner,
            "v8r3_successful_terminal_quarantine": quarantine_binding,
            "units": final_units,
            "cross_outer_validation_reuse_present": True,
            "fully_nested_confirmatory_oof": False,
            "prospective_confirmation_required": True,
            "ready_for_global_discovery_selection": True,
            "commercial_claim_authorized": False,
        },
    )
    del seal

    captured: dict[str, Any] = {}

    def validate_benchmark_pack_free(**kwargs: Any) -> dict[str, Any]:
        captured["benchmark_call"] = dict(kwargs)
        return benchmark_receipt

    fake_benchmark = SimpleNamespace(
        BENCHMARK_PHASE="efficiency_benchmark",
        BENCHMARK_USAGE_IDENTITY=benchmark_identity,
        validate_benchmark_receipt_pack_free=validate_benchmark_pack_free,
    )
    monkeypatch.setattr(selector, "_load_benchmark_module", lambda: fake_benchmark)
    monkeypatch.setattr(
        campaign,
        "validate_v8r3_quarantine_owner_receipt",
        lambda **_kwargs: quarantine_receipt,
    )

    def reconcile(
        path: Path,
        specs: Any,
        *,
        usage_state: Any,
        allow_exact_historical_benchmark_prefix: bool = False,
    ) -> tuple[list[Any], float]:
        assert path == usage_ledger.resolve()
        assert allow_exact_historical_benchmark_prefix is True
        captured["receipt_specs"] = list(specs)
        return list(usage_state.records), 19.0

    monkeypatch.setattr(campaign, "reconcile_usage_ledger", reconcile)
    monkeypatch.setattr(
        campaign,
        "verify_usage_ledger_prefix_binding",
        lambda *args, **kwargs: captured.setdefault("prefix_calls", []).append(
            (args, kwargs)
        ),
    )
    return {
        "root": discovery_root,
        "usage_ledger": usage_ledger,
        "execution_ledger": execution_ledger,
        "gpu_lock": gpu_lock,
        "state": state,
        "receipts": receipts,
        "receipt_paths": receipt_paths,
        "metrics_paths": metrics_paths,
        "benchmark_receipt": benchmark_receipt,
        "authorization_binding": authorization_binding,
        "captured": captured,
    }


def _locked_v8_selector_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Publish the one byte-exact lock/auth pair over a closed seal prefix."""

    fixture = _v8_selector_fixture(tmp_path, monkeypatch)
    selection_path = tmp_path / "DISCOVERY_SELECTION_LOCK.json"
    authorization_path = tmp_path / "PROMOTION_AUTHORIZATION.json"
    contract_path = tmp_path / "ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json"
    _write_json(contract_path, {"fixture": "selector-contract"})
    contract_binding = campaign.bind_file(contract_path)
    contract = {
        "discovery": {
            "v2_baseline_key": [999.0] * 7,
            "selection_key": [f"selection_key_{number}" for number in range(7)],
        }
    }
    pretrain = {"authorization_binding": fixture["authorization_binding"]}
    monkeypatch.setattr(selector, "DEFAULT_DISCOVERY_ROOT", fixture["root"])
    monkeypatch.setattr(selector, "DEFAULT_SELECTION_LOCK", selection_path)
    monkeypatch.setattr(
        selector, "DEFAULT_PROMOTION_AUTHORIZATION", authorization_path
    )
    monkeypatch.setattr(campaign, "DEFAULT_USAGE_LEDGER", fixture["usage_ledger"])
    monkeypatch.setattr(
        campaign, "validate_contract", lambda _root: (contract, contract_binding)
    )
    monkeypatch.setattr(
        campaign,
        "validate_pretrain_authorization",
        lambda _root, admitted_binding=None: pretrain,
    )
    receipts, seal_binding = selector._validate_discovery_seal(
        fixture["root"],
        project_root=tmp_path,
        usage_ledger=fixture["usage_ledger"],
        usage_state=fixture["state"],
    )
    selection, authorization = selector._derived_selection_documents(
        project_root=tmp_path,
        contract=contract,
        contract_binding=contract_binding,
        pretrain=pretrain,
        receipts=receipts,
        seal_binding=seal_binding,
        selection_lock_path=selection_path,
    )
    assert authorization is not None
    campaign.create_once_json(selection_path, selection)
    campaign.create_once_json(authorization_path, authorization)

    sealed_raw = fixture["usage_ledger"].read_bytes()
    current_raw = sealed_raw + b"promotion-terminal-appended-after-seal\n"
    fixture["usage_ledger"].write_bytes(current_raw)
    prefix_state = SimpleNamespace(
        records=fixture["state"].records,
        raw_bytes=sealed_raw,
        open_reservations={},
    )
    current_state = SimpleNamespace(
        records=fixture["state"].records
        + ({"record_sha256": "f" * 64, "phase": "promotion"},),
        raw_bytes=current_raw,
        open_reservations={},
    )

    def replay(raw: bytes, **_kwargs: Any) -> Any:
        if raw == sealed_raw:
            return prefix_state
        if raw == current_raw:
            return current_state
        raise ValueError("unexpected selector fixture ledger bytes")

    class LockedSnapshot:
        def __enter__(self) -> Any:
            return current_state

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(campaign.gpu_budget_ledger, "verify_ledger_bytes", replay)
    monkeypatch.setattr(
        campaign.gpu_budget_ledger,
        "locked_closed_snapshot",
        lambda *_args, **_kwargs: LockedSnapshot(),
    )
    return {
        **fixture,
        "selection_path": selection_path,
        "promotion_path": authorization_path,
        "selection": selection,
        "promotion": authorization,
        "contract_binding": contract_binding,
        "pretrain": pretrain,
        "sealed_raw": sealed_raw,
        "current_raw": current_raw,
        "prefix_state": prefix_state,
        "current_state": current_state,
    }


def test_public_locked_selector_replays_sealed_prefix_after_promotion_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _locked_v8_selector_fixture(tmp_path, monkeypatch)
    selection, authorization, governance = (
        selector.validate_locked_selection_authorization(
            tmp_path,
            selection_lock_path=fixture["selection_path"],
            promotion_authorization_path=fixture["promotion_path"],
        )
    )
    assert selection == fixture["selection"]
    assert authorization == fixture["promotion"]
    assert governance == {
        "contract": fixture["contract_binding"],
        "pretrain_authorization": fixture["authorization_binding"],
        "selection_lock": campaign.bind_file(fixture["selection_path"], relative_to=tmp_path),
        "promotion_authorization": campaign.bind_file(
            fixture["promotion_path"], relative_to=tmp_path
        ),
    }


@pytest.mark.parametrize("mutation", ("schema", "mode", "seal_mode"))
def test_public_locked_selector_rejects_mutation_or_unfrozen_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    fixture = _locked_v8_selector_fixture(tmp_path, monkeypatch)
    expected = "single-link exact-0444" if mutation != "schema" else "pure nine-candidate"
    if mutation == "schema":
        fixture["selection_path"].chmod(0o644)
        selection = campaign.load_json(fixture["selection_path"], "selection")
        selection["unrecognized_schema_field"] = True
        selection["content_sha256"] = campaign.canonical_content_sha256(selection)
        fixture["selection_path"].write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        fixture["selection_path"].chmod(0o444)
    elif mutation == "mode":
        fixture["selection_path"].chmod(0o644)
    else:
        (fixture["root"] / "DISCOVERY_COMPLETION_SEAL.json").chmod(0o644)
    with pytest.raises(campaign.CampaignError, match=expected):
        selector.validate_locked_selection_authorization(
            tmp_path,
            selection_lock_path=fixture["selection_path"],
            promotion_authorization_path=fixture["promotion_path"],
        )


def test_public_locked_selector_requires_canonical_active_pretrain_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _locked_v8_selector_fixture(tmp_path, monkeypatch)
    alternate = tmp_path / "alternate/PRETRAIN_AUTHORIZATION_V8R4A.json"
    alternate.parent.mkdir(parents=True)
    alternate.write_bytes(
        (tmp_path / selector.PRETRAIN_AUTHORIZATION_RELATIVE).read_bytes()
    )
    alternate.chmod(0o444)
    monkeypatch.setattr(
        campaign,
        "validate_pretrain_authorization",
        lambda _root, admitted_binding=None: {
            "authorization_binding": campaign.bind_file(alternate)
        },
    )
    with pytest.raises(campaign.CampaignError, match="different pretrain authorization"):
        selector.validate_locked_selection_authorization(
            tmp_path,
            selection_lock_path=fixture["selection_path"],
            promotion_authorization_path=fixture["promotion_path"],
        )


def test_public_locked_selector_accepts_only_the_admitted_sole_open_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _locked_v8_selector_fixture(tmp_path, monkeypatch)
    lifecycle_id = "admitted-promotion-lifecycle"
    live_raw = fixture["current_raw"] + b"admitted-live-reservation\n"
    fixture["usage_ledger"].write_bytes(live_raw)
    live_state = SimpleNamespace(
        records=fixture["current_state"].records,
        raw_bytes=live_raw,
        open_reservations={lifecycle_id: {"phase": "promotion"}},
    )

    def replay(raw: bytes, **_kwargs: Any) -> Any:
        if raw == fixture["sealed_raw"]:
            return fixture["prefix_state"]
        if raw == live_raw:
            return live_state
        raise ValueError("unexpected admitted selector ledger bytes")

    monkeypatch.setattr(campaign.gpu_budget_ledger, "verify_ledger_bytes", replay)
    admitted = {
        "lifecycle_id": lifecycle_id,
        "usage_ledger_prefix_bytes": len(live_raw),
        "usage_ledger_prefix_sha256": hashlib.sha256(live_raw).hexdigest(),
    }
    selection, authorization, _governance = (
        selector.validate_locked_selection_authorization(
            tmp_path,
            selection_lock_path=fixture["selection_path"],
            promotion_authorization_path=fixture["promotion_path"],
            admitted_binding=admitted,
        )
    )
    assert selection == fixture["selection"]
    assert authorization == fixture["promotion"]

    live_state.open_reservations["foreign-lifecycle"] = {"phase": "foreign"}
    with pytest.raises(campaign.CampaignError, match="sole live reservation"):
        selector.validate_locked_selection_authorization(
            tmp_path,
            selection_lock_path=fixture["selection_path"],
            promotion_authorization_path=fixture["promotion_path"],
            admitted_binding=admitted,
        )


def test_selector_validates_benchmark_as_first_owner_and_excludes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _v8_selector_fixture(tmp_path, monkeypatch)
    receipts, _ = selector._validate_discovery_seal(
        fixture["root"],
        project_root=tmp_path,
        usage_ledger=fixture["usage_ledger"],
        usage_state=fixture["state"],
    )
    assert set(receipts) == set(campaign.EXPECTED_DISCOVERY_UNITS)
    assert fixture["benchmark_receipt"] not in receipts.values()
    specs = fixture["captured"]["receipt_specs"]
    assert len(specs) == 20
    assert specs[0][0] == fixture["benchmark_receipt"]
    assert specs[0][1] == "efficiency_benchmark"
    assert specs[1][1] == "discovery_v8r3_quarantine"
    assert all(spec[1] == "discovery" for spec in specs[2:])
    call = fixture["captured"]["benchmark_call"]
    assert call["project_root"] == tmp_path
    assert call["receipt_path"].name == "BENCHMARK_COMPLETION_RECEIPT_V8R4.json"


@pytest.mark.parametrize(
    "mutation",
    ("parameter_count", "scientific_signature_sha256", "schema"),
)
def test_selector_reexecutes_and_exactly_matches_each_validated_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _v8_selector_fixture(tmp_path, monkeypatch)
    first_key = campaign.EXPECTED_DISCOVERY_UNITS[0]

    receipt_path = fixture["receipt_paths"][first_key]
    receipt_path.chmod(0o644)
    value = campaign.load_json(receipt_path, "fixture receipt")
    if mutation == "parameter_count":
        value["validated_output"]["parameter_count"] += 1
    elif mutation == "scientific_signature_sha256":
        value["validated_output"]["scientific_signature_sha256"] = "f" * 64
    else:
        value["validated_output"]["unrecognized_schema_field"] = True
    value["content_sha256"] = campaign.canonical_content_sha256(value)
    receipt_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_path.chmod(0o444)
    with pytest.raises(campaign.CampaignError, match="hash drifted"):
        selector._validate_discovery_seal(
            fixture["root"],
            project_root=tmp_path,
            usage_ledger=fixture["usage_ledger"],
            usage_state=fixture["state"],
        )


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"owner_present": False}, "benchmark owner"),
        ({"excluded": False}, "benchmark owner"),
        ({"drift_binding": True}, "hash drifted"),
        ({"benchmark_after_discovery": True}, "precede discovery"),
        ({"authorization_mismatch": True}, "different pretrain authorizations"),
    ),
)
def test_selector_fails_closed_for_missing_drifted_or_late_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, bool],
    message: str,
) -> None:
    fixture = _v8_selector_fixture(tmp_path, monkeypatch, **options)
    with pytest.raises(campaign.CampaignError, match=message):
        selector._validate_discovery_seal(
            fixture["root"],
            project_root=tmp_path,
            usage_ledger=fixture["usage_ledger"],
            usage_state=fixture["state"],
        )


def test_benchmark_receipt_never_enters_ranking_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _v8_selector_fixture(tmp_path, monkeypatch)
    receipts, seal_binding = selector._validate_discovery_seal(
        fixture["root"],
        project_root=tmp_path,
        usage_ledger=fixture["usage_ledger"],
        usage_state=fixture["state"],
    )
    monkeypatch.setattr(
        selector,
        "_validate_discovery_seal",
        lambda *args, **kwargs: (receipts, seal_binding),
    )
    monkeypatch.setattr(
        campaign,
        "validate_contract",
        lambda root: (
            {
                "discovery": {
                    "v2_baseline_key": [0, 0, 0, 0, 0, 0, 0],
                    "selection_key": [f"key_{index}" for index in range(7)],
                }
            },
            {"sha256": "a" * 64},
        ),
    )
    monkeypatch.setattr(
        campaign,
        "validate_pretrain_authorization",
        lambda root: {
            "authorization_binding": fixture["authorization_binding"]
        },
    )
    monkeypatch.setattr(
        campaign,
        "create_once_json",
        lambda path, value: {
            **dict(value),
            "content_sha256": campaign.semantic_sha256(dict(value)),
        },
    )
    observed: list[dict[str, Any]] = []
    original_rank = selector.rank_candidates

    def inspect_rank(values: Any) -> list[dict[str, Any]]:
        observed.extend(dict(value) for value in values)
        return original_rank(values)

    monkeypatch.setattr(selector, "rank_candidates", inspect_rank)
    selector._select_common_variant_locked(
        project_root=tmp_path,
        discovery_root=fixture["root"],
        selection_lock_path=tmp_path / "selection.json",
        promotion_authorization_path=tmp_path / "promotion.json",
        usage_ledger=fixture["usage_ledger"],
        usage_state=fixture["state"],
    )
    assert len(observed) == 9
    rendered = json.dumps(observed, sort_keys=True)
    assert "efficiency_benchmark" not in rendered
    assert "v8_hfr_2epoch_no_accuracy_metric_efficiency" not in rendered
    assert all(len(item["units"]) == 6 for item in observed)


# V8R4 authorization additions.  These belong in this pre-authorized campaign
# regression file; the standalone staging boundary suite is only a review aid.
def test_v8r4_legacy_quarantine_context_is_exact_and_disjoint() -> None:
    legacy = {
        "schema_version": 2,
        "event": "terminal",
        "phase": "discovery",
        "context": {
            "execution_number": 0,
            "outer_fold": 3,
            "resume": False,
            "seed": 20260828,
            "variant": "H0_no_factor",
        },
    }
    identity = {
        "campaign_revision": "V8R4",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    assert campaign._record_matches_exact_v8r3_quarantine(legacy)
    assert not campaign._record_matches_execution(
        legacy, expected_phase="discovery", expected_identity=identity
    )
    corrected = {**legacy, "context": {**legacy["context"], "campaign_revision": "V8R4"}}
    assert not campaign._record_matches_exact_v8r3_quarantine(corrected)
    assert campaign._record_matches_execution(
        corrected, expected_phase="discovery", expected_identity=identity
    )


def test_v8r4_parser_requires_one_capability_mode() -> None:
    parser = campaign.build_parser()
    capability_args = [
        "--run-root",
        "output",
        "--target-sealed-capability-receipt",
        campaign.TARGET_SEALED_CAPABILITY_NAME,
    ]
    with pytest.raises(SystemExit):
        parser.parse_args([])
    shard = parser.parse_args(["--outer-fold-shard", "3", *capability_args])
    assert shard.outer_fold_shard == 3
    assert shard.aggregate_shards is False
    aggregate = parser.parse_args(["--aggregate-shards", *capability_args])
    assert aggregate.aggregate_shards is True
    assert aggregate.training_index is None
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--outer-fold-shard", "3", "--aggregate-shards", *capability_args]
        )


def test_v8r4a_capability_requires_rich_exact_recovery_governance(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()

    def install(relative: Path) -> Path:
        source = EVIDENCE_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        destination.chmod(0o444)
        return destination

    recovery_authorization = install(
        campaign.OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_RELATIVE
    )
    recovery_diagnostic = install(campaign.OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC_RELATIVE)
    execution_authorization = install(
        campaign.EXECUTION_CLOSURE_AUTHORIZATION_RELATIVE
    )
    execution_diagnostic = install(campaign.EXECUTION_CLOSURE_DIAGNOSTIC_RELATIVE)

    def rich(path: Path) -> dict[str, Any]:
        raw = path.read_bytes()
        status = path.stat()
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "st_dev": status.st_dev,
            "st_ino": status.st_ino,
            "mode": f"{status.st_mode & 0o777:04o}",
        }

    capability_path = (
        root
        / campaign.TARGET_SEALED_LIFECYCLE_ROOT_RELATIVE
        / "discovery/run_hfr_v3r1_discovery_campaign/outer_3"
        / campaign.TARGET_SEALED_CAPABILITY_NAME
    )
    _write_json(capability_path, {"test_only": True})
    capability_path.chmod(0o444)
    placeholder = rich(recovery_authorization)
    governance = {
        "campaign_contract": placeholder,
        "active_authorization": placeholder,
        "source_snapshot": placeholder,
        "implementation_test_receipt": placeholder,
        "gpu_state_migration_receipt": placeholder,
        "open_lifecycle_recovery_correction_authorization": rich(
            recovery_authorization
        ),
        "open_lifecycle_recovery_failure_diagnostic": rich(recovery_diagnostic),
        "execution_closure_correction_authorization": rich(
            execution_authorization
        ),
        "execution_closure_failure_diagnostic": rich(execution_diagnostic),
    }
    boundary = {
        "target_reference_or_selection_evidence_accessed": False,
        "legacy_combined_cache_mounted": False,
        "raw_or_target_root_mounted": False,
        "cross_outer_shard_mounted": False,
        "other_pack_or_output_mounted": False,
        "atomic_replace_compatible": True,
        "v8r4a_ledger_migration_required": False,
        "v8r4a_migration_live_replay_validated": True,
        "dedicated_gpu_state_directory_capabilities": True,
        "exactly_three_mutable_state_directory_mounts": True,
        "usage_and_execution_closed_prelaunch": True,
        "lifecycle_mounted_read_only": True,
        "source_snapshot_exact_file_mounts": True,
        "complete_project_source_or_config_trees_mounted": False,
        "production_execution_authorized": False,
        "synthetic_validation_only": True,
    }
    document = {
        "classification": (
            "adaptive_v3r1_v8r4a_outer_target_sealed_runtime_capability_receipt"
        ),
        "campaign_id": campaign.CAMPAIGN_ID,
        "campaign_revision": campaign.CAMPAIGN_REVISION,
        "infrastructure_revision": campaign.INFRASTRUCTURE_REVISION,
        "phase": "discovery",
        "outer_fold": 3,
        "security_boundary": boundary,
        "governance_files": governance,
        "writable_roots": {
            "lifecycle": {"path": str(capability_path.parent)},
                "output": {
                    "path": str(
                        root / campaign.DEFAULT_RUN_ROOT / "shards/outer_3"
                    )
                },
        },
    }

    def runtime(value: Mapping[str, Any]) -> Any:
        return SimpleNamespace(
            validate_capability_receipt=lambda path, **scope: {
                "document": dict(value),
                "binding": rich(path),
            }
        )

    accepted = campaign.validate_target_sealed_capability(
        root,
        capability_path,
        expected_phase="discovery",
        expected_outer_fold=3,
        runtime_module=runtime(document),
    )
    assert set(accepted["document"]["governance_files"]) >= set(governance)

    missing = json.loads(json.dumps(document))
    del missing["governance_files"][
        "open_lifecycle_recovery_failure_diagnostic"
    ]
    with pytest.raises(campaign.CampaignError, match="boundary drifted"):
        campaign.validate_target_sealed_capability(
            root,
            capability_path,
            expected_phase="discovery",
            expected_outer_fold=3,
            runtime_module=runtime(missing),
        )

    swapped = json.loads(json.dumps(document))
    swapped["governance_files"][
        "open_lifecycle_recovery_correction_authorization"
    ] = rich(recovery_diagnostic)
    with pytest.raises(campaign.CampaignError, match="exact governance drifted"):
        campaign.validate_target_sealed_capability(
            root,
            capability_path,
            expected_phase="discovery",
            expected_outer_fold=3,
            runtime_module=runtime(swapped),
        )


def test_v8r4_real_shard_index_bindings_are_exact_and_disjoint() -> None:
    assert campaign.SHARD_TRAINING_INDEX[3] != campaign.SHARD_TRAINING_INDEX[4]
    assert campaign.SHARD_TRAINING_INDEX_SHA256 == {
        3: "db204cdba72e9f3023c58ef37c1761cfd4ec2f4310449f8eaeeef7003afadb9b",
        4: "1ce1b1a0154c609b1fa1693ff5702b3e81be178f8ddbdd7ec25f559c39029f0a",
    }
    assert campaign.SHARD_TRAINING_INDEX_BYTES == {3: 3172, 4: 3172}
    assert all(campaign._is_sha256(value) for value in campaign.SHARD_TRAINING_INDEX_SHA256.values())


def test_v8r4_training_input_binds_partition_manifest() -> None:
    binding = {"path": "/tmp/partition.json", "sha256": "0" * 64, "bytes": 1}
    value = campaign.TrainingInput(
        outer_fold=3,
        seed=20260828,
        cache_dir=Path("/tmp/cache"),
        cache_manifest_sha256="1" * 64,
        proposer_stack=Path("/tmp/stack.npz"),
        proposer_stack_sha256="2" * 64,
        partition_manifest_binding=binding,
    )
    assert value.partition_manifest_binding == binding


def test_v8r4_discovery_claim_boundary_is_adaptive_not_confirmatory() -> None:
    boundary = campaign.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION
    assert boundary["campaign_revision"] == "V8R4"
    assert boundary["outer_prediction_pack_absent_during_discovery"] is True
    assert boundary["combined_target_bearing_cache_opened"] is False
    source = Path(campaign.__file__).read_text(encoding="utf-8")
    assert '"cross_outer_validation_reuse_present": True' in source
    assert '"fully_nested_confirmatory_oof": False' in source
    assert '"prospective_confirmation_required": True' in source


def test_v8r4_campaign_rejects_object_identity_pickle_free_replay(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path)
    output = tmp_path / "output"
    _stub_training_outputs(output)
    prediction_path = output / "validation_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["identity"] = np.asarray(["A", "B"], dtype=object)
    np.savez_compressed(prediction_path, **arrays)
    with pytest.raises(campaign.CampaignError, match="pickle-free"):
        campaign.validate_training_output(
            output,
            outer_fold=3,
            seed=20260828,
            variant="H0_no_factor",
            cache_dir=cache,
        )


def test_v8r4_campaign_rejects_v8r3_before_prediction_npz_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _cache(tmp_path)
    output = tmp_path / "output"
    _stub_training_outputs(output)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["campaign_revision"] = "V8R3"
    manifest["effective_configuration"] = {
        "campaign_revision": "V8R3",
        "outer_fold": 3,
        "validation_fold": 4,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    _write_json(manifest_path, manifest)
    calls = 0
    original = campaign.np.load

    def recording_load(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(campaign.np, "load", recording_load)
    with pytest.raises(campaign.CampaignError, match="campaign_revision"):
        campaign.validate_training_output(
            output,
            outer_fold=3,
            seed=20260828,
            variant="H0_no_factor",
            cache_dir=cache,
        )
    assert calls == 0


def test_v8r4_selector_derives_exact_pack_free_authority() -> None:
    metrics = _metrics()
    receipts: dict[tuple[int, int, str], dict[str, Any]] = {}
    for fold in campaign.OUTER_RUNS:
        for seed in campaign.SEEDS:
            for variant in campaign.VARIANTS:
                receipts[(fold, seed, variant)] = {
                    "content_sha256": "a" * 64,
                    "validated_output": {
                        "parameter_count": 1234,
                        "release_metrics": {
                            mode: dict(metrics) for mode in campaign.RELEASE_MODES
                        },
                        "artifacts": {
                            "validation_metrics.json": {
                                "path": f"metrics/{fold}/{seed}/{variant}.json",
                                "sha256": "b" * 64,
                                "bytes": 1,
                            }
                        },
                    },
                }
    contract = {
        "discovery": {
            "v2_baseline_key": [9, 9.0, 9.0, 9.0, 9.0, 9.0, 999999],
            "selection_key": [
                "violation_count", "max_violation", "sum_violation",
                "worst_identity_macro_mae", "mean_identity_macro_mae",
                "mean_overall_mae", "parameter_count",
            ],
        }
    }
    lock, authorization = selector._derived_selection_documents(
        project_root=ROOT,
        contract=contract,
        contract_binding={"path": "contract", "sha256": "c" * 64, "bytes": 1},
        pretrain={
            "authorization_binding": {
                "path": "PRETRAIN_AUTHORIZATION_V8R4.json",
                "sha256": "d" * 64,
                "bytes": 1,
            }
        },
        receipts=receipts,
        seal_binding={"path": "seal", "sha256": "e" * 64, "bytes": 1},
        selection_lock_path=ROOT / selector.DEFAULT_SELECTION_LOCK,
    )
    assert lock["classification"] == (
        "adaptive_v3r1_v8r4_global_discovery_selection_lock"
    )
    assert lock["campaign_revision"] == "V8R4"
    assert lock["selection_process_pack_free"] is True
    assert lock["cross_outer_validation_reuse_present"] is True
    assert lock["fully_nested_confirmatory_oof"] is False
    assert authorization is not None
    assert authorization["classification"] == (
        "adaptive_v3r1_v8r4_promotion_authorization"
    )
    assert authorization["authorized_scopes"] == [
        "promotion_training_pack", "outer_prediction_pack"
    ]
    assert authorization["authorized_now"] is True


def test_v8r4_selector_final_replay_has_no_training_pack_opener() -> None:
    source = inspect.getsource(selector._validate_discovery_seal)
    assert "load_training_index" not in source
    assert "validate_training_output" not in source
    assert "validation_metrics.json" in source
    assert "training_shards" in source
    assert "discovery_capability_shard_seal" in source


def test_v8r4_selector_opaque_indices_are_exact_without_materialization(
    tmp_path: Path,
) -> None:
    for outer in campaign.OUTER_RUNS:
        binding = dict(selector.SHARD_INDEX_BINDINGS[outer])
        assert selector._opaque_index_binding_matches(
            binding, outer_fold=outer, project_root=ROOT
        )
        tampered = dict(binding)
        tampered["sha256"] = "0" * 64
        assert not selector._opaque_index_binding_matches(
            tampered, outer_fold=outer, project_root=ROOT
        )
    alternate = dict(selector.SHARD_INDEX_BINDINGS[3])
    alternate["path"] = str(tmp_path / Path(alternate["path"]).name)
    assert not selector._opaque_index_binding_matches(
        alternate, outer_fold=3, project_root=ROOT
    )


def test_v8r4_selector_replays_two_shards_without_pack_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _v8_selector_fixture(tmp_path, monkeypatch)
    receipts, binding = selector._validate_discovery_seal(
        fixture["root"],
        project_root=tmp_path,
        usage_ledger=fixture["usage_ledger"],
        usage_state=fixture["state"],
    )
    assert set(receipts) == set(campaign.EXPECTED_DISCOVERY_UNITS)
    assert binding == campaign.bind_file(
        fixture["root"] / "DISCOVERY_COMPLETION_SEAL.json",
        relative_to=tmp_path,
    )
    return

    root = tmp_path / "discovery"
    ledger = tmp_path / "usage.jsonl"
    gpu_ledger = tmp_path / "execution.jsonl"
    gpu_lock = tmp_path / "gpu.lock"
    raw_ledger = b"closed-v8r4-ledger-prefix\n"
    benchmark_path = tmp_path / "BENCHMARK_COMPLETION_RECEIPT_V8R4.json"
    benchmark_receipt = {
        "usage_record_sha256s": ["1" * 64],
        "usage_identity": {
            "campaign_revision": "V8R4",
            "benchmark_id": "fixture",
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
        },
    }
    campaign.create_once_json(benchmark_path, benchmark_receipt)
    benchmark_binding = campaign.bind_file(benchmark_path, relative_to=tmp_path)
    quarantine_path = tmp_path / campaign.V8R3_QUARANTINE_RELATIVE
    quarantine = {
        "content_sha256": "2" * 64,
        "usage_record_sha256s": ["3" * 64],
    }
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_text("quarantine\n", encoding="utf-8")
    quarantine_path.chmod(0o444)
    quarantine_binding = campaign.bind_file(quarantine_path, relative_to=tmp_path)
    contract_binding = {
        "path": "contract.json",
        "sha256": campaign.CONTRACT_FILE_SHA256,
        "bytes": 1,
    }
    pretrain_binding = {
        "path": "PRETRAIN_AUTHORIZATION_V8R4.json",
        "sha256": "4" * 64,
        "bytes": 1,
    }
    all_units: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    for outer in campaign.OUTER_RUNS:
        shard_units: list[dict[str, Any]] = []
        for seed in campaign.SEEDS:
            for variant in campaign.VARIANTS:
                key = (outer, seed, variant)
                metrics_path = tmp_path / "metrics" / f"{outer}_{seed}_{variant}.json"
                _write_json(
                    metrics_path,
                    {
                        "classification": "adaptive_v3r1_v8r4_discovery_validation_only",
                        "campaign_revision": "V8R4",
                        "outer_test_rows_present": False,
                        "release_modes": {
                            mode: {"metrics": _metrics()}
                            for mode in campaign.RELEASE_MODES
                        },
                    },
                )
                artifact_bindings = {
                    name: (
                        campaign.bind_file(metrics_path)
                        if name == "validation_metrics.json"
                        else {
                            "path": str(tmp_path / "opaque" / f"{outer}_{seed}_{variant}_{name}"),
                            "sha256": "5" * 64,
                            "bytes": 1,
                        }
                    )
                    for name in campaign.REQUIRED_TRAIN_OUTPUTS
                }
                receipt_path = (
                    root / f"outer_{outer}" / "units"
                    / f"outer_{outer}_seed_{seed}_{variant}"
                    / "completion_receipt.json"
                )
                receipt = campaign.create_once_json(
                    receipt_path,
                    {
                        "schema_version": 1,
                        "classification": "adaptive_v3r1_v8r4_discovery_unit_completion",
                        "campaign_id": campaign.CAMPAIGN_ID,
                        "campaign_revision": "V8R4",
                        "outer_test_opened": False,
                        "outer_fold": outer,
                        "validation_fold": (outer + 1) % 6,
                        "seed": seed,
                        "variant": variant,
                        "invocation": {"path": "opaque", "sha256": "6" * 64, "bytes": 1},
                        "usage_ledger_path": str(ledger),
                        "usage_record_sha256": "7" * 64,
                        "usage_record_sha256s": ["7" * 64],
                        "terminal_results": [{}],
                        "lifecycle_invocations": [{}],
                        "gpu_execution_ledger_path": str(gpu_ledger),
                        "gpu_admission_lock_path": str(gpu_lock),
                        "validated_output": {
                            "campaign_revision": "V8R4",
                            "outer_fold": outer,
                            "validation_fold": (outer + 1) % 6,
                            "seed": seed,
                            "variant": variant,
                            "parameter_count": 1234,
                            "validation_rows": 10,
                            "valid_reference_rows": 10,
                            "release_metrics": {
                                mode: _metrics() for mode in campaign.RELEASE_MODES
                            },
                            "scientific_signature_sha256": "8" * 64,
                            "physical_boundary": dict(
                                campaign.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION
                            ),
                            "row_access_audit": {"outer_row_access_attempts": 0},
                            "artifacts": artifact_bindings,
                        },
                        "commercial_claim_authorized": False,
                    },
                )
                del receipt
                unit = {
                    "outer_fold": outer,
                    "seed": seed,
                    "variant": variant,
                    "receipt": campaign.bind_file(receipt_path),
                }
                shard_units.append(unit)
                all_units.append(unit)
        index_binding = dict(selector.SHARD_INDEX_BINDINGS[outer])
        shard_path = root / f"outer_{outer}" / "DISCOVERY_SHARD_COMPLETION_SEAL.json"
        campaign.create_once_json(
            shard_path,
            {
                "schema_version": 1,
                "classification": "adaptive_v3r1_v8r4_discovery_capability_shard_seal",
                "campaign_id": campaign.CAMPAIGN_ID,
                "campaign_revision": "V8R4",
                "outer_fold_shard": outer,
                "contract": contract_binding,
                "pretrain_authorization": pretrain_binding,
                "training_index": index_binding,
                "completed_units": 9,
                "peer_outer_shard_pack_mounted_or_opened": False,
                "combined_target_bearing_cache_opened": False,
                "outer_prediction_pack_absent": True,
                "physical_boundary": dict(campaign.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
                "gpu_usage_ledger_prefix": {
                    "path": str(ledger), "sha256": hashlib.sha256(raw_ledger).hexdigest(),
                    "bytes": len(raw_ledger), "terminal_record_sha256": "7" * 64,
                },
                "pre_discovery_efficiency_benchmark": benchmark_binding,
                "v8r3_quarantine_owner": quarantine_binding,
                "units": shard_units,
                "cross_outer_validation_reuse_present": True,
                "fully_nested_confirmatory_oof": False,
                "prospective_confirmation_required": True,
                "ready_for_pack_free_shard_aggregation": True,
                "commercial_claim_authorized": False,
            },
        )
        shard_records.append(
            {
                "outer_fold": outer,
                "seal": campaign.bind_file(shard_path),
                "training_index": index_binding,
            }
        )
    seal_path = root / "DISCOVERY_COMPLETION_SEAL.json"
    campaign.create_once_json(
        seal_path,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_v8r4_target_sealed_discovery_completion",
            "campaign_id": campaign.CAMPAIGN_ID,
            "campaign_revision": "V8R4",
            "contract": contract_binding,
            "pretrain_authorization": pretrain_binding,
            "training_shards": shard_records,
            "outer_runs": list(campaign.OUTER_RUNS),
            "seeds": list(campaign.SEEDS),
            "variants": list(campaign.VARIANTS),
            "completed_units": 18,
            "physical_boundary": dict(campaign.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
            "validation_targets_only": True,
            "gpu_elapsed_seconds": 12.5,
            "gpu_hours_hard": campaign.GPU_HOURS_HARD,
            "gpu_usage_ledger": {
                "path": str(ledger), "sha256": hashlib.sha256(raw_ledger).hexdigest(),
                "bytes": len(raw_ledger), "terminal_record_sha256": "7" * 64,
            },
            "gpu_usage_ledger_path": str(ledger),
            "pre_discovery_efficiency_benchmark": {
                "receipt": benchmark_binding,
                "included_in_gpu_exact_cover": True,
                "excluded_from_selection": True,
                "artifacts_quarantined": True,
            },
            "v8r3_successful_terminal_quarantine": quarantine_binding,
            "units": all_units,
            "cross_outer_validation_reuse_present": True,
            "fully_nested_confirmatory_oof": False,
            "prospective_confirmation_required": True,
            "ready_for_global_discovery_selection": True,
            "commercial_claim_authorized": False,
        },
    )
    fake_benchmark = SimpleNamespace(
        BENCHMARK_PHASE="efficiency_benchmark",
        BENCHMARK_USAGE_IDENTITY=benchmark_receipt["usage_identity"],
        validate_benchmark_receipt_pack_free=lambda **kwargs: benchmark_receipt,
    )
    monkeypatch.setattr(selector, "_load_benchmark_module", lambda: fake_benchmark)
    monkeypatch.setattr(
        campaign, "validate_v8r3_quarantine_owner_receipt", lambda **kwargs: quarantine
    )
    monkeypatch.setattr(
        campaign, "reconcile_usage_ledger", lambda *args, **kwargs: ([], 12.5)
    )
    state = SimpleNamespace(raw_bytes=raw_ledger, records=())
    receipts, binding = selector._validate_discovery_seal(
        root, project_root=tmp_path, usage_ledger=ledger, usage_state=state
    )
    assert set(receipts) == set(campaign.EXPECTED_DISCOVERY_UNITS)
    assert binding == campaign.bind_file(seal_path, relative_to=tmp_path)
    assert not any((tmp_path / record["path"]).exists() for record in selector.SHARD_INDEX_BINDINGS.values())


def test_v8r4_campaign_rejects_physical_outer_row(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path)
    output = tmp_path / "output"
    _stub_training_outputs(output)
    with (cache / "metadata.csv").open("a", encoding="utf-8") as stream:
        stream.write("14,3\n")
    with pytest.raises(campaign.CampaignError, match="physically contains"):
        campaign.validate_training_output(
            output,
            outer_fold=3,
            seed=20260828,
            variant="H0_no_factor",
            cache_dir=cache,
        )


def _v8r4_authorized_outer_pack(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "authorized_outer"
    root.mkdir()
    cache_index = np.asarray([7, 9], np.int64)
    pack = root / "outer_predict_input.npz"
    np.savez(
        pack,
        cache_index=cache_index,
        node_features=np.ones((2, 2, 571), np.float32),
        candidate_rr_bpm=np.ones((2, 2), np.float32) * 12,
        candidate_mask=np.ones((2, 2), bool),
        joint_radar_mask=np.ones((2, 3), bool),
        proposer_anchor_bpm=np.ones(2, np.float32) * 12,
        proposer_anchor_std_bpm=np.ones(2, np.float32),
        proposer_anchor_available=np.ones(2, bool),
        classical_rr_bpm=np.ones(2, np.float32) * 12,
        session_reset=np.asarray([True, False]),
    )
    opaque = {"path": "opaque", "sha256": "1" * 64, "bytes": 1}
    manifest = {
        "schema_version": 1,
        "classification": sanitizer.OUTER_PACK_CLASSIFICATION,
        "campaign_id": campaign.CAMPAIGN_ID,
        "campaign_revision": sanitizer.CAMPAIGN_REVISION,
        "outer_fold": 3,
        "seed": 20260828,
        "row_count": 2,
        "fields": list(sanitizer.SAFE_OUTPUT_FIELDS),
        "exact_allowlist": True,
        "forbidden_fields_emitted": False,
        "reference_identity_protocol_quality_decoded": False,
        "legacy_index": dict(opaque),
        "legacy_cache_manifest": dict(opaque),
        "legacy_proposer_stack": dict(opaque),
        "promotion_authorization": dict(opaque),
        "output": {
            "path": "outer_predict_input.npz",
            "sha256": campaign.sha256_file(pack),
            "bytes": pack.stat().st_size,
        },
        "global_cache_index_sha256": hashlib.sha256(
            np.ascontiguousarray(cache_index).view(np.uint8)
        ).hexdigest(),
        "object_arrays": False,
        "pickle": False,
        "commercial_or_confirmatory_claim_allowed": False,
    }
    manifest["content_sha256"] = campaign.canonical_content_sha256(manifest)
    _write_json(root / "OUTER_PREDICTION_PACK_MANIFEST.json", manifest)
    return root, pack


def test_v8r4_sanitizer_consumes_only_authorized_outer_pack(
    tmp_path: Path,
) -> None:
    root, pack = _v8r4_authorized_outer_pack(tmp_path)
    document, binding = sanitizer._validate_outer_pack_manifest(
        root, outer_fold=3, seed=20260828
    )
    assert document["fields"] == list(sanitizer.SAFE_OUTPUT_FIELDS)
    assert Path(binding["output"]["path"]) == pack
    assert not (root / "metadata.csv").exists()


def test_v8r4_sanitizer_manifest_tamper_fails_before_npz_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _v8r4_authorized_outer_pack(tmp_path)
    manifest_path = root / "OUTER_PREDICTION_PACK_MANIFEST.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["forged_target_path"] = "target.npz"
    document["content_sha256"] = campaign.canonical_content_sha256(document)
    _write_json(manifest_path, document)
    calls = 0

    def forbidden_np_load(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("NPZ must not open before manifest validation")

    monkeypatch.setattr(sanitizer.np, "load", forbidden_np_load)
    with pytest.raises(campaign.CampaignError, match="schema drifted"):
        sanitizer._validate_outer_pack_manifest(root, outer_fold=3, seed=20260828)
    assert calls == 0
