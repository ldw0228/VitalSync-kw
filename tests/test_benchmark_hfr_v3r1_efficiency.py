from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_hfr_v3r1_efficiency.py"
SPEC = importlib.util.spec_from_file_location("benchmark_hfr_v3r1_efficiency", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)

DISCOVERY_SCRIPT = PROJECT_ROOT / "scripts" / "run_hfr_v3r1_discovery_campaign.py"
DISCOVERY_SPEC = importlib.util.spec_from_file_location(
    "benchmark_test_discovery", DISCOVERY_SCRIPT
)
assert DISCOVERY_SPEC is not None and DISCOVERY_SPEC.loader is not None
discovery = importlib.util.module_from_spec(DISCOVERY_SPEC)
sys.modules[DISCOVERY_SPEC.name] = discovery
DISCOVERY_SPEC.loader.exec_module(discovery)


def test_load_script_registers_dataclass_module_and_cleans_failed_import(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "valid_dataclass_module.py"
    valid_path.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Item:\n"
        "    value: int\n",
        encoding="utf-8",
    )
    valid_name = "hfr_v8r1_valid_dataclass_fixture"
    module = benchmark._load_script(valid_name, valid_path)
    try:
        assert sys.modules[valid_name] is module
        assert module.Item(3).value == 3
    finally:
        sys.modules.pop(valid_name, None)

    failed_path = tmp_path / "failed_dataclass_module.py"
    failed_path.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Item:\n"
        "    value: int\n"
        "raise RuntimeError('fixture failure')\n",
        encoding="utf-8",
    )
    failed_name = "hfr_v8r1_failed_dataclass_fixture"
    with pytest.raises(RuntimeError, match="fixture failure"):
        benchmark._load_script(failed_name, failed_path)
    assert failed_name not in sys.modules


@pytest.mark.parametrize(
    ("name", "path"),
    (
        (
            "hfr_v8r1_real_discovery_loader_fixture",
            PROJECT_ROOT / "scripts/run_hfr_v3r1_discovery_campaign.py",
        ),
        (
            "hfr_v8r1_real_trainer_loader_fixture",
            PROJECT_ROOT / "scripts/train_harmonic_factor_router_snn_v3r1.py",
        ),
    ),
)
def test_load_script_imports_real_frozen_dataclass_modules(
    name: str, path: Path
) -> None:
    module = benchmark._load_script(name, path)
    try:
        assert sys.modules[name] is module
    finally:
        sys.modules.pop(name, None)


def _trainer_telemetry(
    invocation_sha256: str = "a" * 64,
    *,
    epoch_2_total_ns: int = 10_000_000_000,
) -> dict[str, Any]:
    epoch_2_train = epoch_2_total_ns * 3 // 4
    epoch_2_validation = epoch_2_total_ns - epoch_2_train
    epochs = [
        {
            "epoch": 1,
            "train_ns": 8_000_000_000,
            "validation_ns": 2_000_000_000,
            "total_ns": 10_000_000_000,
            "optimizer_steps": benchmark.OPTIMIZER_STEPS_PER_EPOCH,
            "training_windows": benchmark.TRAINING_WINDOWS_PER_EPOCH,
            "validation_windows": benchmark.VALIDATION_WINDOWS_PER_EPOCH,
            "warmup": True,
        },
        {
            "epoch": 2,
            "train_ns": epoch_2_train,
            "validation_ns": epoch_2_validation,
            "total_ns": epoch_2_total_ns,
            "optimizer_steps": benchmark.OPTIMIZER_STEPS_PER_EPOCH,
            "training_windows": benchmark.TRAINING_WINDOWS_PER_EPOCH,
            "validation_windows": benchmark.VALIDATION_WINDOWS_PER_EPOCH,
            "warmup": False,
        },
    ]
    return {
        "invocation_sha256": invocation_sha256,
        "epochs_completed": 2,
        "epochs": epochs,
        "optimizer_steps": 10,
        "training_windows": benchmark.TRAINING_WINDOWS_PER_EPOCH * 2,
        "validation_windows": benchmark.VALIDATION_WINDOWS_PER_EPOCH * 2,
        "peak_cuda_memory_bytes": 123_456,
    }


def _verified_binding(invocation_sha256: str = "a" * 64) -> dict[str, Any]:
    return {
        "valid": True,
        "classification": "verified_v8_gpu_admitted_child_lifecycle",
        "phase": benchmark.BENCHMARK_PHASE,
        "context": dict(benchmark.BENCHMARK_USAGE_IDENTITY),
        "invocation_sha256": invocation_sha256,
    }


def _worker_args(tmp_path: Path) -> argparse.Namespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    authorization = tmp_path / "PRETRAIN_AUTHORIZATION_V8.json"
    authorization.write_text("{}\n", encoding="utf-8")
    benchmark_output = tmp_path / benchmark.BENCHMARK_RUN_ROOT_RELATIVE
    capability = (
        tmp_path
        / benchmark.BENCHMARK_LIFECYCLE_RELATIVE
        / benchmark.TARGET_SEALED_CAPABILITY_NAME
    )
    capability.parent.mkdir(parents=True, exist_ok=True)
    capability.write_text("{}\n", encoding="utf-8")
    capability.chmod(0o444)
    return argparse.Namespace(
        project_root=tmp_path,
        trainer=tmp_path / "trainer.py",
        cache=tmp_path / "cache",
        proposer_stack=tmp_path / "proposer.npz",
        telemetry_output=tmp_path / "QUARANTINED_TIMING_TELEMETRY.json",
        trainer_output_dir=tmp_path / "FORBIDDEN_REUSABLE_TRAINER_OUTPUT",
        usage_ledger=tmp_path / "usage.jsonl",
        execution_ledger=tmp_path / "execution.jsonl",
        authorization_path=authorization,
        authorization_sha256=hashlib.sha256(authorization.read_bytes()).hexdigest(),
        target_sealed_capability_receipt=capability,
    )


class _FakeRuntime:
    @staticmethod
    def validate_capability_receipt(
        path: Path, *, expected_phase: str, expected_outer_fold: int
    ) -> dict[str, Any]:
        raw = path.read_bytes()
        project_root = path.parent
        for _part in benchmark.BENCHMARK_LIFECYCLE_RELATIVE.parts:
            project_root = project_root.parent
        output = project_root / benchmark.BENCHMARK_RUN_ROOT_RELATIVE
        evidence_root = (
            PROJECT_ROOT
            if (PROJECT_ROOT / benchmark.OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_RELATIVE).is_file()
            else Path.cwd()
        )

        def install(relative: Path) -> Path:
            source = evidence_root / relative
            destination = project_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            destination.chmod(0o444)
            return destination

        recovery_authorization = install(
            benchmark.OPEN_LIFECYCLE_RECOVERY_AUTHORIZATION_RELATIVE
        )
        recovery_diagnostic = install(
            benchmark.OPEN_LIFECYCLE_RECOVERY_DIAGNOSTIC_RELATIVE
        )
        execution_authorization = install(
            benchmark.EXECUTION_CLOSURE_AUTHORIZATION_RELATIVE
        )
        execution_diagnostic = install(
            benchmark.EXECUTION_CLOSURE_DIAGNOSTIC_RELATIVE
        )

        def rich_binding(material: Path) -> dict[str, Any]:
            status = material.stat()
            material_raw = material.read_bytes()
            return {
                "path": str(material.resolve()),
                "sha256": hashlib.sha256(material_raw).hexdigest(),
                "bytes": len(material_raw),
                "st_dev": status.st_dev,
                "st_ino": status.st_ino,
                "mode": f"{status.st_mode & 0o777:04o}",
            }

        placeholder = rich_binding(recovery_authorization)
        boundary = {
            "synthetic_validation_only": True,
            "production_execution_authorized": False,
            "atomic_replace_compatible": True,
            "v8r4a_ledger_migration_required": False,
            "v8r4a_migration_live_replay_validated": True,
            "dedicated_gpu_state_directory_capabilities": True,
            "exactly_three_mutable_state_directory_mounts": True,
            "usage_and_execution_closed_prelaunch": True,
            "lifecycle_mounted_read_only": True,
            "source_snapshot_exact_file_mounts": True,
            "complete_project_source_or_config_trees_mounted": False,
            "target_reference_or_selection_evidence_accessed": False,
            "legacy_combined_cache_mounted": False,
            "raw_or_target_root_mounted": False,
            "cross_outer_shard_mounted": False,
            "other_pack_or_output_mounted": False,
        }
        return {
            "document": {
                "classification": (
                    "adaptive_v3r1_v8r4a_outer_target_sealed_runtime_capability_receipt"
                ),
                "campaign_id": benchmark.CAMPAIGN_ID,
                "campaign_revision": benchmark.CAMPAIGN_REVISION,
                "infrastructure_revision": benchmark.INFRASTRUCTURE_REVISION,
                "phase": expected_phase,
                "outer_fold": expected_outer_fold,
                "writable_roots": {
                    "output": {"path": str(output.resolve())},
                    "lifecycle": {"path": str(path.parent.resolve())},
                },
                "governance_files": {
                    "campaign_contract": placeholder,
                    "active_authorization": placeholder,
                    "source_snapshot": placeholder,
                    "implementation_test_receipt": placeholder,
                    "gpu_state_migration_receipt": placeholder,
                    "open_lifecycle_recovery_correction_authorization": rich_binding(
                        recovery_authorization
                    ),
                    "open_lifecycle_recovery_failure_diagnostic": rich_binding(
                        recovery_diagnostic
                    ),
                    "execution_closure_correction_authorization": rich_binding(
                        execution_authorization
                    ),
                    "execution_closure_failure_diagnostic": rich_binding(
                        execution_diagnostic
                    ),
                },
                "security_boundary": boundary,
            },
            "binding": rich_binding(path),
        }


class _FakeAdmitted:
    def __init__(self, binding: Mapping[str, Any] | None = None) -> None:
        self.binding = dict(binding or _verified_binding())
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def consume_admitted_child_binding(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, Any]:
        self.calls.append((args, kwargs))
        return dict(self.binding)


class _FakeTrainer:
    def __init__(self, telemetry: Mapping[str, Any]) -> None:
        self.telemetry = dict(telemetry)
        self.calls: list[tuple[argparse.Namespace, Mapping[str, Any]]] = []
        self.validation_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.pretrains: list[Mapping[str, Any]] = []
        self.events: list[str] = []

    def validate_pretrain_authorization(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, Any]:
        self.events.append("validate_pretrain")
        self.validation_calls.append((args, kwargs))
        return {
            "valid": True,
            "training_authorized": True,
            "commercial_claim_authorized": False,
        }

    @staticmethod
    def parse_args(argv: Sequence[str]) -> argparse.Namespace:
        values = list(argv)
        return argparse.Namespace(
            mode=values[values.index("--mode") + 1],
            outer_fold=int(values[values.index("--outer-fold") + 1]),
            seed=int(values[values.index("--seed") + 1]),
            variant=values[values.index("--variant") + 1],
            epochs=int(values[values.index("--epochs") + 1]),
            device=values[values.index("--device") + 1],
            target_sealed_capability_receipt=Path(
                values[
                    values.index("--target-sealed-capability-receipt") + 1
                ]
            ),
            expected_admitted_context=json.loads(
                values[values.index("--expected-admitted-context-json") + 1]
            ),
        )

    def run_efficiency_benchmark(
        self,
        args: argparse.Namespace,
        *,
        admitted_binding: Mapping[str, Any],
        pretrain: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.events.append("primitive")
        self.calls.append((args, admitted_binding))
        self.pretrains.append(dict(pretrain))
        return dict(self.telemetry)


def test_canonical_benchmark_identity_path_and_profile_are_fixed() -> None:
    assert benchmark.BENCHMARK_USAGE_IDENTITY == {
        "campaign_revision": "V8R4",
        "infrastructure_revision": "V8R4A",
        "authorization_generation": "CONTEXT1",
        "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    assert benchmark.BENCHMARK_RECEIPT_RELATIVE.as_posix() == (
        "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "efficiency_benchmark_v8r4a_context1/"
        "BENCHMARK_COMPLETION_RECEIPT_V8R4.json"
    )
    assert benchmark.HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE.name == (
        "efficiency_benchmark_v8"
    )
    assert benchmark.BENCHMARK_PROFILE_SHA256 == benchmark.semantic_sha256(
        benchmark.BENCHMARK_PROFILE
    )
    assert benchmark.BENCHMARK_PROFILE_SHA256 == (
        "8d031ce6808e622361944828cb9338afbe28e2f63fc7f4016d47bde2c5e6b9d0"
    )
    assert benchmark.CURRENT_UNIT_INVOCATION_NAME == "BENCHMARK_INVOCATION_V8R4.json"
    assert benchmark.PRETRAIN_AUTHORIZATION_RELATIVE.name == (
        "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"
    )
    assert benchmark.BENCHMARK_LIFECYCLE_RELATIVE.as_posix() == (
        "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "target_sealed_lifecycle_v8r4a_context1/efficiency_benchmark/"
        "benchmark_hfr_v3r1_efficiency/outer_3"
    )
    assert benchmark.FROZEN_EXECUTION_CLOSURE_ACTIVE_OUTPUT_ROOT_RELATIVE.name == (
        "efficiency_benchmark_v8r4a"
    )
    assert (
        benchmark.FROZEN_EXECUTION_CLOSURE_ACTIVE_OUTPUT_ROOT_RELATIVE
        != benchmark.BENCHMARK_RUN_ROOT_RELATIVE
    )
    assert benchmark.BENCHMARK_PROFILE["epochs"] == 2
    assert benchmark.BENCHMARK_PROFILE["training_windows_per_epoch"] == 6_583
    assert benchmark.BENCHMARK_PROFILE[
        "target_free_validation_windows_per_epoch"
    ] == 1_658
    assert benchmark.BENCHMARK_PROFILE[
        "epoch_2_train_plus_target_free_validation_ns_max"
    ] == 23_000_000_000
    assert benchmark.BENCHMARK_PROFILE["accuracy_metrics_allowed"] is False


def test_frozen_execution_closure_root_is_independent_of_rootbind1_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = benchmark._validate_historical_projection_authority(PROJECT_ROOT)
    assert expected["sha256"] == benchmark._RECOVERY_GOVERNANCE[
        "execution_closure_correction_authorization"
    ]["sha256"]
    monkeypatch.setattr(
        benchmark,
        "BENCHMARK_RUN_ROOT_RELATIVE",
        Path(
            "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
            "unrelated_successor_output"
        ),
    )
    assert benchmark._validate_historical_projection_authority(PROJECT_ROOT) == expected
    monkeypatch.setattr(
        benchmark,
        "FROZEN_EXECUTION_CLOSURE_ACTIVE_OUTPUT_ROOT_RELATIVE",
        benchmark.BENCHMARK_RUN_ROOT_RELATIVE,
    )
    with pytest.raises(benchmark.BenchmarkError, match="historical projection"):
        benchmark._validate_historical_projection_authority(PROJECT_ROOT)


def test_v8r4_nonouter_pack_proves_full_processed_covers() -> None:
    metadata_path = PROJECT_ROOT / (
        "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "v8r4_split_inputs/discovery_shard_outer_3/units/"
        "outer_3_seed_20260828/discovery_cache/metadata.csv"
    )
    structural_columns = (
        "fold",
        "identity",
        "session_id",
        "window_number",
    )
    metadata = pd.read_csv(metadata_path, usecols=list(structural_columns))
    assert set(metadata.columns) == set(structural_columns)
    assert {"rr_bpm", "reference_valid"}.isdisjoint(metadata.columns)
    fold_rows = metadata.groupby("fold", sort=True).size().to_dict()
    assert fold_rows == {0: 2043, 1: 1584, 2: 1616, 4: 1658, 5: 1340}
    training = metadata[metadata["fold"] != 4]
    validation = metadata[metadata["fold"] == 4]
    assert len(training) == benchmark.TRAINING_WINDOWS_PER_EPOCH == 6_583
    assert len(validation) == benchmark.VALIDATION_WINDOWS_PER_EPOCH == 1_658
    assert metadata[metadata["fold"] == 3].shape[0] == 0
    assert training[["identity", "session_id"]].drop_duplicates().shape[0] == 20
    assert validation[["identity", "session_id"]].drop_duplicates().shape[0] == 5

    def unbatched_forward_chunks(frame: pd.DataFrame) -> int:
        lengths = frame.groupby(["identity", "session_id"], sort=True).size()
        return int(((lengths + 31) // 32).sum())

    # These are the stale values that were previously mislabeled as windows.
    assert unbatched_forward_chunks(training) == 216
    assert unbatched_forward_chunks(validation) == 54


def test_trainer_telemetry_is_exact_timing_only_and_two_epoch() -> None:
    telemetry = _trainer_telemetry()
    assert benchmark.validate_trainer_telemetry(
        telemetry, invocation_sha256="a" * 64
    ) == telemetry
    forbidden = ("accuracy", "metric", "checkpoint", "selection", "score")
    serialized_keys = " ".join(telemetry).lower()
    assert not any(token in serialized_keys for token in forbidden)
    assert telemetry["epochs"][0]["warmup"] is True
    assert telemetry["epochs"][1]["warmup"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"accuracy": 0.9}), "key set"),
        (lambda value: value.update({"epochs_completed": True}), "integer"),
        (lambda value: value["epochs"][1].update({"warmup": True}), "warmup"),
        (
            lambda value: value["epochs"][1].update({"optimizer_steps": 4}),
            "optimizer-step",
        ),
        (
            lambda value: value["epochs"][1].update(
                {"training_windows": benchmark.TRAINING_WINDOWS_PER_EPOCH - 1}
            ),
            "training window",
        ),
        (
            lambda value: value["epochs"][1].update(
                {"validation_windows": benchmark.VALIDATION_WINDOWS_PER_EPOCH - 1}
            ),
            "validation cover",
        ),
        (
            lambda value: value["epochs"][1].update({"total_ns": 9_999}),
            "does not add",
        ),
    ],
)
def test_trainer_telemetry_mutations_fail_closed(
    mutation: Any, message: str
) -> None:
    telemetry = _trainer_telemetry()
    mutation(telemetry)
    with pytest.raises(benchmark.BenchmarkError, match=message):
        benchmark.validate_trainer_telemetry(
            telemetry, invocation_sha256="a" * 64
        )


def test_worker_consumes_binding_before_trainer_and_quarantines_only_telemetry(
    tmp_path: Path,
) -> None:
    args = _worker_args(tmp_path)
    admitted = _FakeAdmitted()
    trainer = _FakeTrainer(_trainer_telemetry())
    assert benchmark.run_internal_worker(
        args, admitted_module=admitted, trainer_module=trainer,
        runtime_module=_FakeRuntime(),
    ) == 0
    assert len(admitted.calls) == 1
    assert admitted.calls[0][0][0] == benchmark.BENCHMARK_PHASE
    assert admitted.calls[0][1] == {
        "expected_campaign_id": benchmark.CAMPAIGN_ID,
        "expected_gpu_lock_file": (tmp_path / benchmark.DEFAULT_GPU_LOCK).resolve(),
    }
    assert len(trainer.calls) == 1
    assert trainer.events == ["validate_pretrain", "primitive"]
    assert len(trainer.validation_calls) == 1
    validation_args, validation_kwargs = trainer.validation_calls[0]
    assert validation_args == (_verified_binding(),)
    assert validation_kwargs == {
        "target_sealed_capability_receipt": (
            args.target_sealed_capability_receipt.resolve()
        ),
        "expected_phase": benchmark.BENCHMARK_PHASE,
        "expected_context": dict(benchmark.BENCHMARK_USAGE_IDENTITY),
        "expected_outer_fold": 3,
    }
    assert trainer.pretrains == [
        {
            "valid": True,
            "training_authorized": True,
            "commercial_claim_authorized": False,
        }
    ]
    assert trainer.calls[0][1] == _verified_binding()
    assert trainer.calls[0][0].target_sealed_capability_receipt == (
        args.target_sealed_capability_receipt
    )
    assert trainer.calls[0][0].expected_admitted_context == (
        benchmark.BENCHMARK_USAGE_IDENTITY
    )
    document = benchmark.validate_worker_telemetry(
        args.telemetry_output, invocation_sha256="a" * 64
    )
    assert document["gate_passed"] is True
    assert document["training_result_reusable"] is False
    assert document["selection_or_promotion_input"] is False
    assert document["accuracy_metrics_emitted_or_used"] is False
    assert document["checkpoint_selection_performed"] is False
    assert document["artifacts_quarantined"] is True
    assert not args.trainer_output_dir.exists()
    assert args.telemetry_output.stat().st_mode & 0o777 == 0o444


def test_worker_pretrain_bridge_failure_prevents_trainer_primitive(
    tmp_path: Path,
) -> None:
    args = _worker_args(tmp_path)
    trainer = _FakeTrainer(_trainer_telemetry())

    def refuse(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        trainer.events.append("validate_pretrain")
        raise RuntimeError("independent target scope rejected")

    trainer.validate_pretrain_authorization = refuse  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="independent target scope rejected"):
        benchmark.run_internal_worker(
            args,
            admitted_module=_FakeAdmitted(),
            trainer_module=trainer,
            runtime_module=_FakeRuntime(),
        )
    assert trainer.events == ["validate_pretrain"]
    assert trainer.calls == []
    assert not args.telemetry_output.exists()


def test_worker_gate_failure_is_durable_but_not_successful(
    tmp_path: Path,
) -> None:
    args = _worker_args(tmp_path)
    trainer = _FakeTrainer(
        _trainer_telemetry(epoch_2_total_ns=benchmark.STEADY_GATE_NS + 1)
    )
    assert benchmark.run_internal_worker(
        args,
        admitted_module=_FakeAdmitted(),
        trainer_module=trainer,
        runtime_module=_FakeRuntime(),
    ) == benchmark.GATE_FAILURE_EXIT
    document = benchmark.validate_worker_telemetry(
        args.telemetry_output, invocation_sha256="a" * 64
    )
    assert document["gate_passed"] is False
    assert document["training_result_reusable"] is False
    assert not args.trainer_output_dir.exists()


def test_worker_telemetry_rejects_json_number_type_substitution(
    tmp_path: Path,
) -> None:
    args = _worker_args(tmp_path)
    assert benchmark.run_internal_worker(
        args,
        admitted_module=_FakeAdmitted(),
        trainer_module=_FakeTrainer(_trainer_telemetry()),
        runtime_module=_FakeRuntime(),
    ) == 0
    document = benchmark.load_json(args.telemetry_output, "worker telemetry")
    document["epoch_2_train_ns"] = float(document["epoch_2_train_ns"])
    mutated = tmp_path / "MUTATED_TELEMETRY.json"
    benchmark.create_once_json(mutated, document)
    with pytest.raises(benchmark.BenchmarkError, match="invariants drifted"):
        benchmark.validate_worker_telemetry(
            mutated, invocation_sha256="a" * 64
        )


def test_direct_worker_launch_fails_before_trainer_or_cache_use(tmp_path: Path) -> None:
    args = _worker_args(tmp_path)

    class RefusingAdmitted:
        @staticmethod
        def consume_admitted_child_binding(
            *_args: Any, **_kwargs: Any
        ) -> Mapping[str, Any]:
            raise RuntimeError("wrapper-issued inherited descriptor is absent")

    class ForbiddenTrainer:
        @staticmethod
        def parse_args(_argv: Sequence[str]) -> argparse.Namespace:
            raise AssertionError("trainer must not load before admission")

    with pytest.raises(RuntimeError, match="descriptor is absent"):
        benchmark.run_internal_worker(
            args,
            admitted_module=RefusingAdmitted(),
            trainer_module=ForbiddenTrainer(),
            runtime_module=_FakeRuntime(),
        )
    assert not args.telemetry_output.exists()
    assert not args.trainer_output_dir.exists()


def test_worker_rejects_accuracy_or_reusable_trainer_output(tmp_path: Path) -> None:
    args = _worker_args(tmp_path)
    telemetry = _trainer_telemetry()
    telemetry["checkpoint"] = "forbidden.pt"
    with pytest.raises(benchmark.BenchmarkError, match="key set"):
        benchmark.run_internal_worker(
            args,
            admitted_module=_FakeAdmitted(),
            trainer_module=_FakeTrainer(telemetry),
            runtime_module=_FakeRuntime(),
        )
    assert not args.telemetry_output.exists()

    args = _worker_args(tmp_path / "second")

    class OutputTrainer(_FakeTrainer):
        def run_efficiency_benchmark(
            self,
            parsed: argparse.Namespace,
            *,
            admitted_binding: Mapping[str, Any],
            pretrain: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            del parsed, admitted_binding, pretrain
            args.trainer_output_dir.mkdir(parents=True)
            return _trainer_telemetry()

    with pytest.raises(benchmark.BenchmarkError, match="reusable output"):
        benchmark.run_internal_worker(
            args,
            admitted_module=_FakeAdmitted(),
            trainer_module=OutputTrainer(_trainer_telemetry()),
            runtime_module=_FakeRuntime(),
        )
    assert not args.telemetry_output.exists()


def test_admitted_command_binds_authorization_without_self_referential_invocation(
    tmp_path: Path,
) -> None:
    worker = ["python", "worker.py"]
    command = benchmark.admitted_wrapper_command(
        python=tmp_path / "python",
        wrapper=tmp_path / "wrapper.py",
        gpu_lock=tmp_path / "gpu.lock",
        execution_ledger=tmp_path / "execution.jsonl",
        usage_ledger=tmp_path / "usage.jsonl",
        result_file=tmp_path / "result.json",
        invocation_sha256="d" * 64,
        authorization_path=tmp_path / "PRETRAIN_AUTHORIZATION_V8.json",
        authorization_sha256="e" * 64,
        worker_command=worker,
    )
    assert command[command.index("--phase") + 1] == "efficiency_benchmark"
    assert command[command.index("--authorization-sha256") + 1] == "e" * 64
    assert command[command.index("--invocation-sha256") + 1] == "d" * 64
    assert "d" * 64 not in worker
    assert command[-2:] == worker


@pytest.mark.parametrize(
    "field",
    ("trainer", "gpu_wrapper", "gpu_lock", "execution_ledger", "usage_ledger"),
)
def test_parent_override_fails_before_discovery_or_artifact_read(
    tmp_path: Path, field: str
) -> None:
    args = argparse.Namespace(
        project_root=tmp_path,
        training_index=tmp_path / benchmark.DEFAULT_TRAINING_INDEX,
        run_root=tmp_path / "must-not-exist",
        python=Path(sys.executable),
        trainer=tmp_path / benchmark.TRAINER_RELATIVE,
        gpu_wrapper=tmp_path / benchmark.WRAPPER_RELATIVE,
        gpu_lock=tmp_path / benchmark.DEFAULT_GPU_LOCK,
        execution_ledger=tmp_path / benchmark.DEFAULT_EXECUTION_LEDGER,
        usage_ledger=tmp_path / benchmark.DEFAULT_USAGE_LEDGER,
    )
    setattr(args, field, tmp_path / f"alternate-{field}")

    class ForbiddenDiscovery:
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError(f"discovery was read before override rejection: {name}")

    with pytest.raises(benchmark.BenchmarkError, match="canonical defaults"):
        benchmark.run_benchmark(args, discovery_module=ForbiddenDiscovery())
    assert not args.run_root.exists()


def _synthetic_worker_command(
    *,
    marker: Path,
    telemetry_output: Path,
    usage_ledger: Path,
    execution_ledger: Path,
    gpu_lock: Path,
    authorization_path: Path,
    authorization_sha256: str,
) -> list[str]:
    code = f"""
import importlib.util,json,pathlib,sys
sys.dont_write_bytecode=True
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec)
 spec.loader.exec_module(module)
 return module
admitted=load('synthetic_admitted',{str(PROJECT_ROOT / 'scripts/run_gpu_admitted.py')!r})
bench=load('synthetic_benchmark',{str(SCRIPT)!r})
binding=admitted.consume_admitted_child_binding(
 'efficiency_benchmark',pathlib.Path({str(usage_ledger)!r}),
 pathlib.Path({str(execution_ledger)!r}),pathlib.Path({str(authorization_path)!r}),
 {authorization_sha256!r},expected_campaign_id=bench.CAMPAIGN_ID,
 expected_gpu_lock_file=pathlib.Path({str(gpu_lock)!r}))
control=pathlib.Path({str(marker)!r}).read_text(encoding='utf-8').strip()
if control == 'fail':
 raise SystemExit(9)
gate_failed=control == 'gate'
steady_total=bench.STEADY_GATE_NS + 1 if gate_failed else 9000000000
steady_train=steady_total - 2000000000
trainer={{
 'invocation_sha256':binding['invocation_sha256'],'epochs_completed':2,
 'epochs':[
  {{'epoch':1,'train_ns':8000000000,'validation_ns':2000000000,'total_ns':10000000000,'optimizer_steps':5,'training_windows':bench.TRAINING_WINDOWS_PER_EPOCH,'validation_windows':bench.VALIDATION_WINDOWS_PER_EPOCH,'warmup':True}},
  {{'epoch':2,'train_ns':steady_train,'validation_ns':2000000000,'total_ns':steady_total,'optimizer_steps':5,'training_windows':bench.TRAINING_WINDOWS_PER_EPOCH,'validation_windows':bench.VALIDATION_WINDOWS_PER_EPOCH,'warmup':False}}
 ],
 'optimizer_steps':10,'training_windows':bench.TRAINING_WINDOWS_PER_EPOCH*2,
 'validation_windows':bench.VALIDATION_WINDOWS_PER_EPOCH*2,
 'peak_cuda_memory_bytes':123456
}}
bench.create_once_json(pathlib.Path({str(telemetry_output)!r}),{{
 'schema_version':1,
 'classification':'adaptive_v3r1_v8r4_quarantined_efficiency_telemetry',
 'campaign_id':bench.CAMPAIGN_ID,'campaign_revision':bench.CAMPAIGN_REVISION,
 'infrastructure_revision':bench.INFRASTRUCTURE_REVISION,
 'phase':bench.BENCHMARK_PHASE,
 'benchmark_id':bench.BENCHMARK_ID,'unit':bench.BENCHMARK_UNIT,
 'usage_identity':dict(bench.BENCHMARK_USAGE_IDENTITY),
 'profile_sha256':bench.BENCHMARK_PROFILE_SHA256,'epochs':2,
 'epoch_1_is_warmup':True,'epoch_2_train_ns':steady_train,
 'epoch_2_target_free_validation_ns':2000000000,
 'epoch_2_train_plus_target_free_validation_ns':steady_total,
 'epoch_2_gate_ns_max':bench.STEADY_GATE_NS,'gate_passed':not gate_failed,
 'trainer_telemetry':trainer,'admitted_child_binding':dict(binding),
 'outer_test_opened':False,'accuracy_metrics_emitted_or_used':False,
 'checkpoint_selection_performed':False,'training_result_reusable':False,
 'selection_or_promotion_input':False,'artifacts_quarantined':True,
 'commercial_claim_authorized':False
}})
raise SystemExit(bench.GATE_FAILURE_EXIT if gate_failed else 0)
"""
    return [str(PROJECT_ROOT / ".venv/bin/python"), "-c", code]


def test_full_cpu_only_lifecycle_never_retries_completed_gate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization_path = tmp_path / "PRETRAIN_AUTHORIZATION_V8R4A.json"
    authorization_path.write_text('{"classification":"synthetic-v8"}\n', encoding="utf-8")
    authorization_path.chmod(0o444)
    authorization_sha256 = hashlib.sha256(authorization_path.read_bytes()).hexdigest()
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_outputs: dict[str, dict[str, Any]] = {}
    for logical_name, filename in benchmark._REQUIRED_CACHE_OUTPUTS.items():
        output = cache_dir / filename
        output.write_bytes(f"synthetic:{logical_name}".encode("utf-8"))
        cache_outputs[logical_name] = {
            "filename": filename,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "bytes": output.stat().st_size,
        }
    manifest_document: dict[str, Any] = {
        "complete": True,
        "format_version": 1,
        "classification": "adaptive_v3r1_v8r4_nonouter_training_validation_pack",
        "campaign_id": benchmark.CAMPAIGN_ID,
        "campaign_revision": benchmark.CAMPAIGN_REVISION,
        "outer_fold": 3,
        "partition": "outer_excluded_training_validation",
        "outer_prediction_pack_absent": True,
        "outer_test_rows_physically_present": False,
        "source_combined_cache_open_authorized_by_consumer": False,
        "outputs": cache_outputs,
    }
    manifest_document["content_sha256"] = benchmark.semantic_sha256(
        manifest_document
    )
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_document, sort_keys=True) + "\n", encoding="utf-8"
    )
    cache_input_binding = {
        "manifest": discovery.bind_file(manifest_path),
        "outputs": cache_outputs,
    }
    proposer_stack = tmp_path / "proposer.npz"
    proposer_stack.write_bytes(b"synthetic")
    proposer_stack_binding = discovery.bind_file(proposer_stack)
    training_index_path = tmp_path / "training-index.json"
    training_index_document: dict[str, Any] = {
        "units": [
            {
                "outer_fold": 3,
                "seed": 20260828,
                "artifacts": {
                    "cache_manifest": cache_input_binding["manifest"],
                    "proposer_stack": proposer_stack_binding,
                },
            }
        ]
    }
    training_index_document["content_sha256"] = benchmark.semantic_sha256(
        training_index_document
    )
    training_index_path.write_text(
        json.dumps(training_index_document, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker = tmp_path / "attempt-control.txt"
    marker.write_text("gate", encoding="utf-8")
    usage_ledger = tmp_path / "usage.jsonl"
    execution_ledger = tmp_path / "execution.jsonl"
    gpu_lock = tmp_path / benchmark.DEFAULT_GPU_LOCK
    run_root = tmp_path / "benchmark-run"

    monkeypatch.setattr(
        discovery,
        "validate_contract",
        lambda _root: ({}, discovery.bind_file(contract_path)),
    )
    monkeypatch.setattr(
        discovery,
        "validate_pretrain_authorization",
        lambda _root, **_scope: {
            "efficiency_benchmark_authorized": True,
            "scientific_campaign_revision": benchmark.CAMPAIGN_REVISION,
            "infrastructure_revision": benchmark.INFRASTRUCTURE_REVISION,
            "authorization_binding": discovery.bind_file(authorization_path),
            "contract_binding": discovery.bind_file(contract_path),
            "target_sealed_capability": capability_value,
            "gpu_usage_ledger_path": str(usage_ledger),
            "gpu_execution_ledger_path": str(execution_ledger),
        },
    )
    monkeypatch.setattr(
        discovery,
        "load_training_index",
        lambda _root, _path, *, outer_fold_shard: (
            {
                (3, 20260828): SimpleNamespace(
                    cache_dir=cache_dir,
                    proposer_stack=proposer_stack,
                    cache_input_binding=cache_input_binding,
                    proposer_stack_binding=proposer_stack_binding,
                )
            },
            discovery.bind_file(training_index_path),
        ),
    )
    monkeypatch.setattr(
        discovery,
        "verify_training_cache_inputs",
        lambda _root, _cache, *, outer_fold: (manifest_document, cache_input_binding),
    )
    monkeypatch.setattr(
        discovery,
        "verify_training_bound_file",
        lambda _root, _path, **_expected: proposer_stack_binding,
    )
    monkeypatch.setattr(benchmark, "CONTRACT_RELATIVE", Path("contract.json"))
    monkeypatch.setattr(
        benchmark,
        "CONTRACT_FILE_SHA256",
        hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        benchmark, "DEFAULT_TRAINING_INDEX", Path("training-index.json")
    )
    monkeypatch.setattr(
        benchmark,
        "DEFAULT_TRAINING_INDEX_SHA256",
        hashlib.sha256(training_index_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        benchmark, "DEFAULT_TRAINING_INDEX_BYTES", training_index_path.stat().st_size
    )
    monkeypatch.setattr(
        benchmark, "PRETRAIN_AUTHORIZATION_RELATIVE", Path(authorization_path.name)
    )
    monkeypatch.setattr(
        benchmark,
        "TRAINER_RELATIVE",
        PROJECT_ROOT / "scripts/train_harmonic_factor_router_snn_v3r1.py",
    )
    monkeypatch.setattr(
        benchmark, "WRAPPER_RELATIVE", PROJECT_ROOT / "scripts/run_gpu_admitted.py"
    )
    monkeypatch.setattr(benchmark, "DEFAULT_EXECUTION_LEDGER", Path("execution.jsonl"))
    monkeypatch.setattr(benchmark, "DEFAULT_USAGE_LEDGER", Path("usage.jsonl"))
    monkeypatch.setattr(benchmark, "BENCHMARK_RUN_ROOT_RELATIVE", Path("benchmark-run"))
    capability_path = run_root / benchmark.TARGET_SEALED_CAPABILITY_NAME
    capability_path.parent.mkdir(parents=True, exist_ok=True)
    capability_path.write_text("{}\n", encoding="utf-8")
    capability_binding = discovery.bind_file(capability_path)
    migration_binding = {
        "path": "migration.json",
        "sha256": "f" * 64,
        "bytes": 1,
    }
    capability_value = {
        "document": {
            "security_boundary": {
                "production_execution_authorized": True,
                "synthetic_validation_only": False,
                "atomic_replace_compatible": True,
                "v8r4a_ledger_migration_required": False,
            },
            "writable_roots": {"output": {"path": str(run_root.resolve())}},
            "governance_files": {"gpu_state_migration_receipt": migration_binding},
            "sealed_pack_index": discovery.bind_file(training_index_path),
        },
        "binding": capability_binding,
    }
    monkeypatch.setattr(
        benchmark,
        "validate_target_sealed_capability",
        lambda **kwargs: capability_value,
    )
    monkeypatch.setattr(
        discovery,
        "_capability_governance_binding",
        lambda *_args, **_kwargs: migration_binding,
    )
    monkeypatch.setattr(
        benchmark,
        "_validate_historical_projection_authority",
        lambda _root: {"path": "execution-closure.json", "sha256": "e" * 64, "bytes": 1},
    )
    monkeypatch.setattr(
        benchmark, "_validate_benchmark_entry_prefix", lambda _state, **_kwargs: None
    )
    monkeypatch.setattr(benchmark, "_no_discovery_terminal", lambda _state: None)

    def synthetic_command(**kwargs: Any) -> list[str]:
        return _synthetic_worker_command(
            marker=marker,
            telemetry_output=kwargs["telemetry_output"],
            usage_ledger=kwargs["usage_ledger"],
            execution_ledger=kwargs["execution_ledger"],
            gpu_lock=gpu_lock,
            authorization_path=kwargs["authorization_path"],
            authorization_sha256=kwargs["authorization_sha256"],
        )

    monkeypatch.setattr(benchmark, "benchmark_worker_command", synthetic_command)
    args = argparse.Namespace(
        project_root=tmp_path,
        training_index=training_index_path,
        run_root=run_root,
        python=Path(sys.executable),
        trainer=PROJECT_ROOT / "scripts/train_harmonic_factor_router_snn_v3r1.py",
        gpu_wrapper=PROJECT_ROOT / "scripts/run_gpu_admitted.py",
        gpu_lock=gpu_lock,
        execution_ledger=execution_ledger,
        usage_ledger=usage_ledger,
        target_sealed_capability_receipt=capability_path,
    )

    with pytest.raises(benchmark.BenchmarkGateFailed, match="exceeded 23 seconds"):
        benchmark.run_benchmark(args, discovery_module=discovery)
    first_state = discovery.gpu_budget_ledger.require_closed_ledger(usage_ledger)
    assert [record["event"] for record in first_state.records] == [
        "reservation",
        "terminal",
    ]
    assert first_state.records[-1]["return_code"] == benchmark.GATE_FAILURE_EXIT
    assert not (run_root / benchmark.BENCHMARK_RECEIPT_NAME).exists()
    current_unit_path = run_root / benchmark.CURRENT_UNIT_INVOCATION_NAME
    current_unit_after_failure = current_unit_path.read_bytes()

    marker.write_text("pass", encoding="utf-8")
    with pytest.raises(
        benchmark.BenchmarkError,
        match="sole benchmark attempt is terminal|second scientific attempt is forbidden",
    ):
        benchmark.run_benchmark(args, discovery_module=discovery)
    assert current_unit_path.read_bytes() == current_unit_after_failure
    assert len(benchmark._attempt_directories(run_root)) == 1
    assert tuple(discovery.gpu_budget_ledger.require_closed_ledger(usage_ledger).records) == tuple(
        first_state.records
    )
    return

    receipt = benchmark.run_benchmark(args, discovery_module=discovery)
    assert current_unit_path.read_bytes() == current_unit_after_failure
    assert receipt["attempt_count"] == 2
    assert receipt["usage_identity"] == benchmark.BENCHMARK_USAGE_IDENTITY
    assert receipt["training_result_reusable"] is False
    assert receipt["selection_or_promotion_input"] is False
    assert receipt["artifacts_quarantined"] is True
    assert len(receipt["terminal_results"]) == 2
    assert len(receipt["lifecycle_invocations"]) == 2
    final_state = discovery.gpu_budget_ledger.require_closed_ledger(usage_ledger)
    assert [record["event"] for record in final_state.records] == [
        "reservation",
        "terminal",
        "reservation",
        "terminal",
    ]
    assert final_state.records[-1]["return_code"] == 0
    receipt_path = run_root / benchmark.BENCHMARK_RECEIPT_NAME
    assert receipt_path.stat().st_mode & 0o777 == 0o444
    assert current_unit_path.is_file()
    assert not (run_root / benchmark.LEGACY_UNIT_INVOCATION_NAME).exists()
    receipt_before_rerun = receipt_path.read_bytes()
    state_before_rerun = tuple(final_state.records)

    def forbidden_completed_relaunch(_command: Sequence[str]) -> int:
        raise AssertionError("completed V8R3 benchmark attempted a GPU relaunch")

    assert benchmark.run_benchmark(
        args,
        command_runner=forbidden_completed_relaunch,
        discovery_module=discovery,
    ) == receipt
    assert receipt_path.read_bytes() == receipt_before_rerun
    assert tuple(
        discovery.gpu_budget_ledger.require_closed_ledger(usage_ledger).records
    ) == state_before_rerun
    invocation_commands = [
        discovery.load_json(
            run_root / "attempts" / f"attempt_{index:03d}" / "invocation.json",
            "attempt invocation",
        )["workload_command"]
        for index in range(2)
    ]
    assert invocation_commands[0] != invocation_commands[1]
    assert all("--device" not in command for command in invocation_commands)
    command_hashes = {
        discovery.sha256_file(
            run_root / "attempts" / f"attempt_{index:03d}" / "invocation.json"
        ): benchmark.semantic_sha256(command)
        for index, command in enumerate(invocation_commands)
    }
    final_invocation_path = run_root / "attempts/attempt_001/invocation.json"
    original_invocation_raw = final_invocation_path.read_bytes()
    original_invocation = discovery.load_json(
        final_invocation_path, "original final benchmark invocation"
    )
    for mutation, message in (
        ("command", "command/unit binding drifted"),
        ("unit", "command/unit binding drifted"),
    ):
        forged_invocation = json.loads(json.dumps(original_invocation))
        if mutation == "command":
            forged_invocation["workload_command"] = ["/bin/sh", "-c", "arbitrary"]
            forged_invocation["workload_command_sha256"] = benchmark.semantic_sha256(
                forged_invocation["workload_command"]
            )
        else:
            forged_invocation["unit_invocation"] = {
                "path": str(tmp_path / "foreign-unit.json"),
                "sha256": "f" * 64,
                "bytes": 1,
            }
        forged_invocation.pop("content_sha256", None)
        forged_invocation["content_sha256"] = benchmark.semantic_sha256(
            forged_invocation
        )
        final_invocation_path.chmod(0o644)
        final_invocation_path.write_text(
            json.dumps(forged_invocation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        forged_binding = discovery.bind_file(final_invocation_path)
        forged_receipt = json.loads(json.dumps(receipt))
        forged_receipt["lifecycle_invocations"][-1] = forged_binding
        forged_receipt["terminal_results"][-1][
            "execution_invocation"
        ] = forged_binding
        forged_receipt_path = run_root / f"FORGED_{mutation.upper()}_RECEIPT.json"
        benchmark.create_once_json(forged_receipt_path, forged_receipt)
        with pytest.raises(benchmark.BenchmarkError, match=message):
            benchmark.validate_benchmark_receipt(
                discovery,
                project_root=tmp_path,
                receipt_path=forged_receipt_path,
                usage_ledger=usage_ledger,
                execution_ledger=execution_ledger,
                gpu_lock=gpu_lock,
                expected_command_sha256=lambda record: command_hashes[
                    str(record["invocation_sha256"])
                ],
                usage_state=final_state,
            )
        final_invocation_path.write_bytes(original_invocation_raw)
        final_invocation_path.chmod(0o444)
    mutated_receipt = dict(receipt)
    mutated_receipt["epoch_2_train_ns"] = float(
        mutated_receipt["epoch_2_train_ns"]
    )
    mutated_path = run_root / "MUTATED_BENCHMARK_COMPLETION_RECEIPT.json"
    benchmark.create_once_json(mutated_path, mutated_receipt)
    with pytest.raises(benchmark.BenchmarkError, match="invariants drifted"):
        benchmark.validate_benchmark_receipt(
            discovery,
            project_root=tmp_path,
            receipt_path=mutated_path,
            usage_ledger=usage_ledger,
            execution_ledger=execution_ledger,
            gpu_lock=gpu_lock,
            expected_command_sha256=lambda record: command_hashes[
                str(record["invocation_sha256"])
            ],
            usage_state=final_state,
        )
    drifted_cache_output = cache_dir / benchmark._REQUIRED_CACHE_OUTPUTS["feature_names"]
    drifted_cache_output.write_bytes(b"post-receipt scientific-input drift")
    with pytest.raises(benchmark.BenchmarkError, match="binding bytes drifted"):
        benchmark.validate_benchmark_receipt(
            discovery,
            project_root=tmp_path,
            receipt_path=receipt_path,
            usage_ledger=usage_ledger,
            execution_ledger=execution_ledger,
            gpu_lock=gpu_lock,
            expected_command_sha256=lambda record: command_hashes[
                str(record["invocation_sha256"])
            ],
            usage_state=final_state,
        )


def test_discovery_terminal_before_benchmark_is_rejected() -> None:
    state = SimpleNamespace(
        records=(
            {
                "schema_version": 2,
                "phase": "discovery",
                "event": "terminal",
            },
        )
    )
    with pytest.raises(benchmark.BenchmarkError, match="non-quarantined discovery"):
        benchmark._no_discovery_terminal(state)


def test_attempt_directory_sequence_rejects_gaps_or_foreign_entries(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    (attempts / "attempt_001").mkdir()
    with pytest.raises(benchmark.BenchmarkError, match="non-canonical"):
        benchmark._attempt_directories(tmp_path)
    (attempts / "attempt_001").rmdir()
    (attempts / "foreign.txt").write_text("x", encoding="utf-8")
    with pytest.raises(benchmark.BenchmarkError, match="foreign or non-tail"):
        benchmark._attempt_directories(tmp_path)


def test_anonymous_create_once_has_no_named_partial_and_replays_linked_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "receipt.json"

    def before_link(stage: str, _path: Path) -> None:
        if stage == "anonymous_fsynced":
            raise RuntimeError("kill-before-link")

    monkeypatch.setattr(benchmark, "_PUBLICATION_FAULT_HOOK", before_link)
    with pytest.raises(RuntimeError, match="kill-before-link"):
        benchmark.create_once_json(path, {"value": 1})
    assert not path.exists()

    def after_link(stage: str, _path: Path) -> None:
        if stage == "linked":
            raise RuntimeError("kill-after-link")

    monkeypatch.setattr(benchmark, "_PUBLICATION_FAULT_HOOK", after_link)
    with pytest.raises(RuntimeError, match="kill-after-link"):
        benchmark.create_once_json(path, {"value": 1})
    assert path.is_file()
    assert path.stat().st_nlink == 1
    assert path.stat().st_mode & 0o777 == 0o444
    monkeypatch.setattr(benchmark, "_PUBLICATION_FAULT_HOOK", None)
    assert benchmark.create_once_json(path, {"value": 1})["value"] == 1


def test_staged_benchmark_attempt_recovers_only_exact_empty_tail(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    staging = attempts / ".attempt_000.staging"
    staging.mkdir(parents=True)

    def create(path: Path) -> Mapping[str, Any]:
        return benchmark.create_once_json(path, {"classification": "test"})

    def validate(path: Path) -> Mapping[str, Any]:
        value = benchmark.load_json(path, "staged test invocation")
        assert value["classification"] == "test"
        return value

    final = benchmark._publish_benchmark_attempt(
        tmp_path, create_invocation=create, validate_invocation=validate
    )
    assert final == attempts / "attempt_000"
    assert not staging.exists()
    assert benchmark._attempt_directories(tmp_path) == [final]


def test_staged_benchmark_attempt_recovers_rename_before_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def create(path: Path) -> Mapping[str, Any]:
        return benchmark.create_once_json(path, {"classification": "test"})

    def validate(path: Path) -> Mapping[str, Any]:
        return benchmark.load_json(path, "staged test invocation")

    def killed_after_rename(stage: str, _path: Path) -> None:
        if stage == "indexed_directory_linked":
            raise RuntimeError("killed-after-directory-rename")

    monkeypatch.setattr(benchmark, "_PUBLICATION_FAULT_HOOK", killed_after_rename)
    with pytest.raises(RuntimeError, match="killed-after-directory-rename"):
        benchmark._publish_benchmark_attempt(
            tmp_path, create_invocation=create, validate_invocation=validate
        )
    final = tmp_path / "attempts/attempt_000"
    assert final.is_dir()
    monkeypatch.setattr(benchmark, "_PUBLICATION_FAULT_HOOK", None)
    assert benchmark._publish_benchmark_attempt(
        tmp_path, create_invocation=create, validate_invocation=validate
    ) == final


def test_staged_benchmark_attempt_never_replaces_a_racing_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "attempts/attempt_000"

    def create(path: Path) -> Mapping[str, Any]:
        return benchmark.create_once_json(path, {"classification": "test"})

    def validate(path: Path) -> Mapping[str, Any]:
        return benchmark.load_json(path, "staged test invocation")

    def create_racing_final(stage: str, _path: Path) -> None:
        if stage == "indexed_invocation_durable":
            final.mkdir()
            (final / "sentinel").write_text("owned", encoding="utf-8")

    monkeypatch.setattr(benchmark, "_PUBLICATION_FAULT_HOOK", create_racing_final)
    with pytest.raises(benchmark.BenchmarkError, match="without replacement"):
        benchmark._publish_benchmark_attempt(
            tmp_path, create_invocation=create, validate_invocation=validate
        )
    assert (final / "sentinel").read_text(encoding="utf-8") == "owned"


def _synthetic_mixed_authority_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    project_root = tmp_path.resolve()
    run_root = project_root / "benchmark-run"
    attempts_root = run_root / "attempts"
    old_attempt = attempts_root / "attempt_000"
    v8r2_attempt = attempts_root / "attempt_001"
    current_attempt = attempts_root / "attempt_002"
    for attempt in (old_attempt, v8r2_attempt, current_attempt):
        attempt.mkdir(parents=True)
    usage_ledger = project_root / "usage.jsonl"
    execution_ledger = project_root / "execution.jsonl"

    def immutable_file(name: str, raw: bytes) -> Path:
        path = project_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o444)
        return path

    trainer = immutable_file("trainer.py", b"# trainer\n")
    proposer = immutable_file("proposer.npz", b"proposer")
    manifest = immutable_file("cache/manifest.json", b"{}\n")
    old_authorization = immutable_file(
        "PRETRAIN_AUTHORIZATION_V8.json", b'{"authority":"V8"}\n'
    )
    v8r2_authorization = immutable_file(
        "PRETRAIN_AUTHORIZATION_V8R2.json", b'{"authority":"V8R2"}\n'
    )
    current_authorization = immutable_file(
        "PRETRAIN_AUTHORIZATION_V8R4A.json", b'{"authority":"V8R4A"}\n'
    )

    def unit_document(authorization: Path) -> dict[str, Any]:
        return {
            "trainer": discovery.bind_file(trainer),
            "proposer_stack": discovery.bind_file(proposer),
            "cache_manifest": discovery.bind_file(manifest),
            "pretrain_authorization": discovery.bind_file(authorization),
            "usage_ledger_path": str(usage_ledger),
            "gpu_execution_ledger_path": str(execution_ledger),
        }

    old_unit_path = run_root / benchmark.LEGACY_UNIT_INVOCATION_NAME
    v8r2_unit_path = run_root / benchmark.V8R2_UNIT_INVOCATION_NAME
    current_unit_path = run_root / benchmark.CURRENT_UNIT_INVOCATION_NAME
    old_unit = benchmark.create_once_json(
        old_unit_path, unit_document(old_authorization)
    )
    v8r2_unit = benchmark.create_once_json(
        v8r2_unit_path, unit_document(v8r2_authorization)
    )
    current_unit = benchmark.create_once_json(
        current_unit_path, unit_document(current_authorization)
    )
    python = Path(os.path.abspath(sys.executable))
    old_command = benchmark._worker_command_from_unit_authority(
        python=python,
        project_root=project_root,
        unit_invocation=old_unit,
        attempt_root=old_attempt,
        usage_ledger=usage_ledger,
        execution_ledger=execution_ledger,
    )
    v8r2_command = benchmark._worker_command_from_unit_authority(
        python=python,
        project_root=project_root,
        unit_invocation=v8r2_unit,
        attempt_root=v8r2_attempt,
        usage_ledger=usage_ledger,
        execution_ledger=execution_ledger,
    )
    current_command = benchmark._worker_command_from_unit_authority(
        python=python,
        project_root=project_root,
        unit_invocation=current_unit,
        attempt_root=current_attempt,
        usage_ledger=usage_ledger,
        execution_ledger=execution_ledger,
    )
    old_invocation_path = old_attempt / "invocation.json"
    v8r2_invocation_path = v8r2_attempt / "invocation.json"
    current_invocation_path = current_attempt / "invocation.json"
    benchmark._create_execution_invocation(
        discovery,
        path=old_invocation_path,
        unit_invocation=old_unit_path,
        worker_command=old_command,
        usage_identity=benchmark.LEGACY_BENCHMARK_USAGE_IDENTITY,
    )
    benchmark._create_execution_invocation(
        discovery,
        path=v8r2_invocation_path,
        unit_invocation=v8r2_unit_path,
        worker_command=v8r2_command,
        usage_identity=benchmark.LEGACY_BENCHMARK_USAGE_IDENTITY,
    )
    benchmark._create_execution_invocation(
        discovery,
        path=current_invocation_path,
        unit_invocation=current_unit_path,
        worker_command=current_command,
    )
    old_terminal_path = old_attempt / "GPU_TERMINAL_RESULT.json"
    old_terminal = benchmark.create_once_json(
        old_terminal_path,
        {
            "invocation_sha256": discovery.sha256_file(old_invocation_path),
            "terminal_record_sha256": "1" * 64,
            "return_code": 1,
            "charged_usage_ns": 101,
            "reusable_success": False,
            "command_sha256": benchmark.semantic_sha256(old_command),
        },
    )
    v8r2_terminal_path = v8r2_attempt / "GPU_TERMINAL_RESULT.json"
    v8r2_terminal = benchmark.create_once_json(
        v8r2_terminal_path,
        {
            "invocation_sha256": discovery.sha256_file(v8r2_invocation_path),
            "terminal_record_sha256": "2" * 64,
            "return_code": 87,
            "charged_usage_ns": 202,
            "reusable_success": False,
            "command_sha256": benchmark.semantic_sha256(v8r2_command),
        },
    )
    current_terminal_path = current_attempt / "GPU_TERMINAL_RESULT.json"
    benchmark.create_once_json(
        current_terminal_path,
        {
            "invocation_sha256": discovery.sha256_file(current_invocation_path),
            "terminal_record_sha256": "3" * 64,
            "return_code": 0,
            "charged_usage_ns": 303,
            "reusable_success": True,
            "command_sha256": benchmark.semantic_sha256(current_command),
        },
    )
    legacy_whitelist = {
        "authority": "V8",
        "attempt_index": 0,
        "unit_invocation": {
            "relative_path": old_unit_path.relative_to(project_root).as_posix(),
            "sha256": discovery.sha256_file(old_unit_path),
            "bytes": old_unit_path.stat().st_size,
            "content_sha256": old_unit["content_sha256"],
        },
        "execution_invocation": {
            "relative_path": old_invocation_path.relative_to(project_root).as_posix(),
            "sha256": discovery.sha256_file(old_invocation_path),
            "bytes": old_invocation_path.stat().st_size,
            "content_sha256": discovery.load_json(
                old_invocation_path, "synthetic legacy invocation"
            )["content_sha256"],
        },
        "terminal_result": {
            "relative_path": old_terminal_path.relative_to(project_root).as_posix(),
            "sha256": discovery.sha256_file(old_terminal_path),
            "bytes": old_terminal_path.stat().st_size,
            "content_sha256": old_terminal["content_sha256"],
            "terminal_record_sha256": "1" * 64,
            "return_code": 1,
            "charged_usage_ns": 101,
            "reusable_success": False,
        },
        "pretrain_authorization": {
            "relative_path": old_authorization.relative_to(project_root).as_posix(),
            "sha256": discovery.sha256_file(old_authorization),
            "bytes": old_authorization.stat().st_size,
        },
    }
    v8r2_whitelist = {
        "authority": "V8R2",
        "attempt_index": 1,
        "unit_invocation": {
            "relative_path": v8r2_unit_path.relative_to(project_root).as_posix(),
            "sha256": discovery.sha256_file(v8r2_unit_path),
            "bytes": v8r2_unit_path.stat().st_size,
            "content_sha256": v8r2_unit["content_sha256"],
        },
        "execution_invocation": {
            "relative_path": v8r2_invocation_path.relative_to(
                project_root
            ).as_posix(),
            "sha256": discovery.sha256_file(v8r2_invocation_path),
            "bytes": v8r2_invocation_path.stat().st_size,
            "content_sha256": discovery.load_json(
                v8r2_invocation_path, "synthetic V8R2 invocation"
            )["content_sha256"],
        },
        "terminal_result": {
            "relative_path": v8r2_terminal_path.relative_to(project_root).as_posix(),
            "sha256": discovery.sha256_file(v8r2_terminal_path),
            "bytes": v8r2_terminal_path.stat().st_size,
            "content_sha256": v8r2_terminal["content_sha256"],
            "terminal_record_sha256": "2" * 64,
            "return_code": 87,
            "charged_usage_ns": 202,
            "reusable_success": False,
        },
        "pretrain_authorization": {
            "relative_path": v8r2_authorization.relative_to(
                project_root
            ).as_posix(),
            "sha256": discovery.sha256_file(v8r2_authorization),
            "bytes": v8r2_authorization.stat().st_size,
        },
    }
    monkeypatch.setattr(
        benchmark, "BENCHMARK_RUN_ROOT_RELATIVE", Path("benchmark-run")
    )
    monkeypatch.setattr(
        benchmark, "HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE", Path("benchmark-run")
    )
    monkeypatch.setattr(benchmark, "LEGACY_V8_ATTEMPT", legacy_whitelist)
    monkeypatch.setattr(benchmark, "FAILED_V8R2_ATTEMPT", v8r2_whitelist)
    monkeypatch.setattr(
        benchmark,
        "HISTORICAL_BENCHMARK_ATTEMPTS",
        (legacy_whitelist, v8r2_whitelist),
    )
    receipt = {
        "benchmark_invocation": discovery.bind_file(current_unit_path),
        "lifecycle_invocations": [
            discovery.bind_file(old_invocation_path),
            discovery.bind_file(v8r2_invocation_path),
            discovery.bind_file(current_invocation_path),
        ],
        "terminal_results": [
            {
                "terminal_record_sha256": "1" * 64,
                "execution_invocation": discovery.bind_file(old_invocation_path),
                "result": discovery.bind_file(old_terminal_path),
            },
            {
                "terminal_record_sha256": "2" * 64,
                "execution_invocation": discovery.bind_file(v8r2_invocation_path),
                "result": discovery.bind_file(v8r2_terminal_path),
            },
            {
                "terminal_record_sha256": "3" * 64,
                "execution_invocation": discovery.bind_file(current_invocation_path),
                "result": discovery.bind_file(current_terminal_path),
            },
        ],
    }
    return SimpleNamespace(
        project_root=project_root,
        run_root=run_root,
        usage_ledger=usage_ledger,
        execution_ledger=execution_ledger,
        python=python,
        old_unit_path=old_unit_path,
        v8r2_unit_path=v8r2_unit_path,
        current_unit_path=current_unit_path,
        old_unit=old_unit,
        v8r2_unit=v8r2_unit,
        current_unit=current_unit,
        old_invocation_path=old_invocation_path,
        v8r2_invocation_path=v8r2_invocation_path,
        current_invocation_path=current_invocation_path,
        old_terminal_path=old_terminal_path,
        v8r2_terminal_path=v8r2_terminal_path,
        old_command=old_command,
        v8r2_command=v8r2_command,
        current_command=current_command,
        whitelist=legacy_whitelist,
        whitelists=(legacy_whitelist, v8r2_whitelist),
        receipt=receipt,
    )


def test_v8r3_real_historical_attempts_reconstruct_their_own_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Staging overlays may bind the immutable artifact tree by symlink.  The
    # historical documents intentionally contain canonical workspace paths.
    artifact_root = (PROJECT_ROOT / "artifacts").resolve()
    canonical_root = artifact_root.parent
    monkeypatch.setattr(
        benchmark,
        "__file__",
        str(canonical_root / "scripts/benchmark_hfr_v3r1_efficiency.py"),
    )
    observed: list[tuple[str, int, str]] = []
    for historical in benchmark.HISTORICAL_BENCHMARK_ATTEMPTS:
        invocation_path, result_path, result, command = (
            benchmark._validate_exact_historical_attempt(
                discovery,
                historical_attempt=historical,
                project_root=canonical_root,
                    run_root=(
                        canonical_root
                        / benchmark.HISTORICAL_BENCHMARK_RUN_ROOT_RELATIVE
                    ),
                usage_ledger=canonical_root / benchmark.DEFAULT_USAGE_LEDGER,
                execution_ledger=canonical_root / benchmark.DEFAULT_EXECUTION_LEDGER,
                    python=canonical_root / ".venv/bin/python",
            )
        )
        invocation = discovery.load_json(
            invocation_path, "authorized historical invocation"
        )
        assert result_path.stat().st_mode & 0o777 == historical[
            "terminal_result"
        ].get("mode", 0o444)
        assert result["reusable_success"] is historical["terminal_result"][
            "reusable_success"
        ]
        assert invocation["workload_command"] == command
        observed.append(
            (
                str(historical["authority"]),
                int(result["return_code"]),
                Path(command[command.index("--authorization-path") + 1]).name,
            )
        )
    assert observed == [
        ("V8", 1, "PRETRAIN_AUTHORIZATION_V8.json"),
        ("V8R2", 87, "PRETRAIN_AUTHORIZATION_V8R2.json"),
        ("V8R3", 0, "PRETRAIN_AUTHORIZATION_V8R3.json"),
    ]
    assert benchmark.PRETRAIN_AUTHORIZATION_RELATIVE.name == (
        "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"
    )


def test_v8r3_mixed_authority_exact_cover_rebuilds_each_unit_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _synthetic_mixed_authority_lineage(tmp_path, monkeypatch)
    commands, telemetry = benchmark._validate_receipt_lifecycle_invocations(
        discovery,
        project_root=context.project_root,
        receipt=context.receipt,
        unit_invocation_path=context.current_unit_path,
        unit_invocation=context.current_unit,
        usage_ledger=context.usage_ledger,
        execution_ledger=context.execution_ledger,
    )
    assert commands == {
        discovery.sha256_file(context.old_invocation_path): benchmark.semantic_sha256(
            context.old_command
        ),
        discovery.sha256_file(
            context.v8r2_invocation_path
        ): benchmark.semantic_sha256(context.v8r2_command),
        discovery.sha256_file(
            context.current_invocation_path
        ): benchmark.semantic_sha256(context.current_command),
    }
    assert telemetry == (
        context.current_invocation_path.parent
        / "QUARANTINED_TIMING_TELEMETRY.json"
    )
    assert len({tuple(command) for command in (
        context.old_command,
        context.v8r2_command,
        context.current_command,
    )}) == 3


def test_v8r3_historical_tamper_and_reorder_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _synthetic_mixed_authority_lineage(tmp_path, monkeypatch)
    forged_whitelist = json.loads(json.dumps(context.whitelists[1]))
    forged_whitelist["terminal_result"]["sha256"] = "f" * 64
    with pytest.raises(benchmark.BenchmarkError, match="binding bytes drifted"):
        benchmark._validate_exact_historical_attempt(
            discovery,
            historical_attempt=forged_whitelist,
            project_root=context.project_root,
            run_root=context.run_root,
            usage_ledger=context.usage_ledger,
            execution_ledger=context.execution_ledger,
            python=context.python,
        )
    reordered = json.loads(json.dumps(context.receipt))
    reordered["lifecycle_invocations"].reverse()
    reordered["terminal_results"].reverse()
    with pytest.raises(benchmark.BenchmarkError, match="path is non-canonical"):
        benchmark._validate_receipt_lifecycle_invocations(
            discovery,
            project_root=context.project_root,
            receipt=reordered,
            unit_invocation_path=context.current_unit_path,
            unit_invocation=context.current_unit,
            usage_ledger=context.usage_ledger,
            execution_ledger=context.execution_ledger,
        )


def test_v8r3_cross_authority_command_and_historical_success_tail_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _synthetic_mixed_authority_lineage(tmp_path, monkeypatch)
    current = discovery.load_json(
        context.current_invocation_path, "synthetic current invocation"
    )
    current["workload_command"] = list(context.old_command)
    current["workload_command_sha256"] = benchmark.semantic_sha256(
        context.old_command
    )
    current.pop("content_sha256")
    current["content_sha256"] = benchmark.semantic_sha256(current)
    context.current_invocation_path.chmod(0o644)
    context.current_invocation_path.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    context.current_invocation_path.chmod(0o444)
    crossed = json.loads(json.dumps(context.receipt))
    crossed_binding = discovery.bind_file(context.current_invocation_path)
    crossed["lifecycle_invocations"][-1] = crossed_binding
    crossed["terminal_results"][-1]["execution_invocation"] = crossed_binding
    with pytest.raises(benchmark.BenchmarkError, match="command/unit binding drifted"):
        benchmark._validate_receipt_lifecycle_invocations(
            discovery,
            project_root=context.project_root,
            receipt=crossed,
            unit_invocation_path=context.current_unit_path,
            unit_invocation=context.current_unit,
            usage_ledger=context.usage_ledger,
            execution_ledger=context.execution_ledger,
        )
    legacy_tail = {
        "benchmark_invocation": discovery.bind_file(context.current_unit_path),
        "lifecycle_invocations": context.receipt["lifecycle_invocations"][:2],
        "terminal_results": context.receipt["terminal_results"][:2],
    }
    with pytest.raises(benchmark.BenchmarkError, match="successful tail"):
        benchmark._validate_receipt_lifecycle_invocations(
            discovery,
            project_root=context.project_root,
            receipt=legacy_tail,
            unit_invocation_path=context.current_unit_path,
            unit_invocation=context.current_unit,
            usage_ledger=context.usage_ledger,
            execution_ledger=context.execution_ledger,
        )


def test_v8r3_historical_reconciliation_is_read_only_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _synthetic_mixed_authority_lineage(tmp_path, monkeypatch)
    protected = (
        context.old_unit_path,
        context.old_invocation_path,
        context.old_terminal_path,
        context.v8r2_unit_path,
        context.v8r2_invocation_path,
        context.v8r2_terminal_path,
    )
    before = {
        path: (path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns)
        for path in protected
    }
    def validate_all() -> list[tuple[Path, Path, dict[str, Any], list[str]]]:
        return [
            benchmark._validate_exact_historical_attempt(
                discovery,
                historical_attempt=historical,
                project_root=context.project_root,
                run_root=context.run_root,
                usage_ledger=context.usage_ledger,
                execution_ledger=context.execution_ledger,
                python=context.python,
            )
            for historical in context.whitelists
        ]

    first = validate_all()
    second = validate_all()
    assert first == second
    assert {
        path: (path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns)
        for path in protected
    } == before
    assert all((path.stat().st_mode & 0o777) == 0o444 for path in protected)


def test_v8r4_authority_pins_real_nonouter_index_and_historical_prefix() -> None:
    assert benchmark.DEFAULT_TRAINING_INDEX_SHA256 == (
        "db204cdba72e9f3023c58ef37c1761cfd4ec2f4310449f8eaeeef7003afadb9b"
    )
    assert benchmark.DEFAULT_TRAINING_INDEX_BYTES == 3172
    assert len(benchmark.HISTORICAL_BENCHMARK_ATTEMPTS) == 3
    v8r3 = benchmark.HISTORICAL_BENCHMARK_ATTEMPTS[-1]
    assert v8r3["authority"] == "V8R3"
    assert v8r3["attempt_index"] == 2
    assert v8r3["terminal_result"]["reusable_success"] is True
    assert v8r3["terminal_result"]["mode"] == 0o644
    assert v8r3["telemetry"]["sha256"] == (
        "b1e36d03d98c5516512dcc349812707a851a719b3b9a3b6d17d45ccacce8660c"
    )
    assert benchmark.LEGACY_BENCHMARK_USAGE_IDENTITY == {
        "benchmark_id": benchmark.BENCHMARK_ID,
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    assert benchmark.BENCHMARK_USAGE_IDENTITY["campaign_revision"] == "V8R4"
    assert benchmark.BENCHMARK_USAGE_IDENTITY["authorization_generation"] == "CONTEXT1"
    assert "authorization_generation" not in (
        benchmark.ROOTBIND1_BENCHMARK_USAGE_IDENTITY
    )


def test_context1_entry_owns_exact_rootbind1_failure_and_rejects_tamper() -> None:
    usage_ledger = PROJECT_ROOT / benchmark.DEFAULT_USAGE_LEDGER
    state = discovery.gpu_budget_ledger.verify_ledger(
        usage_ledger,
        budget_ns=discovery.gpu_budget_ledger.GPU_BUDGET_NS,
        expected_legacy_genesis_sha256=discovery._expected_legacy_genesis(
            usage_ledger
        ),
    )
    benchmark._validate_benchmark_entry_prefix(
        state, project_root=PROJECT_ROOT
    )
    rootbind_hash = benchmark.ROOTBIND1_FAILED_BENCHMARK_TERMINAL[
        "terminal_record_sha256"
    ]
    records = [dict(record) for record in state.records]
    rootbind_index = next(
        number
        for number, record in enumerate(records)
        if record.get("record_sha256") == rootbind_hash
    )
    records[rootbind_index]["charged_usage_ns"] += 1
    with pytest.raises(benchmark.BenchmarkError, match="ROOTBIND1.*drifted"):
        benchmark._validate_benchmark_entry_prefix(
            SimpleNamespace(records=tuple(records)), project_root=PROJECT_ROOT
        )

    records = [dict(record) for record in state.records]
    quarantine_index = next(
        number
        for number, record in enumerate(records)
        if record.get("record_sha256")
        == "8dbc0493125f22c130444e1344533d1f3d9c4ac445df6adfe8b597972e9691c5"
    )
    records[quarantine_index], records[rootbind_index] = (
        records[rootbind_index],
        records[quarantine_index],
    )
    with pytest.raises(benchmark.BenchmarkError, match="ordering drifted"):
        benchmark._validate_benchmark_entry_prefix(
            SimpleNamespace(records=tuple(records)), project_root=PROJECT_ROOT
        )


def test_v8r4_capability_replays_before_any_pack_opener(tmp_path: Path) -> None:
    args = _worker_args(tmp_path)
    events: list[str] = []

    class Runtime(_FakeRuntime):
        @staticmethod
        def validate_capability_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
            events.append("capability")
            return _FakeRuntime.validate_capability_receipt(*args, **kwargs)

    class RefusingAdmitted:
        @staticmethod
        def consume_admitted_child_binding(*_args: Any, **_kwargs: Any) -> Any:
            events.append("admission")
            raise RuntimeError("stop before cache")

    with pytest.raises(RuntimeError, match="stop before cache"):
        benchmark.run_internal_worker(
            args,
            admitted_module=RefusingAdmitted(),
            trainer_module=object(),
            runtime_module=Runtime(),
        )
    assert events == ["capability", "admission"]


def test_v8r4_real_pack_cache_binding_includes_immutable_global_map() -> None:
    training, index_binding = discovery.load_training_index(
        PROJECT_ROOT,
        PROJECT_ROOT / benchmark.DEFAULT_TRAINING_INDEX,
        outer_fold_shard=3,
    )
    assert index_binding["sha256"] == benchmark.DEFAULT_TRAINING_INDEX_SHA256
    item = training[(3, 20260828)]
    _, binding = benchmark._validate_cache_input_binding(
        item.cache_input_binding, project_root=PROJECT_ROOT
    )
    assert set(binding["outputs"]) == set(benchmark._REQUIRED_CACHE_OUTPUTS)
    assert binding["outputs"]["local_to_global_cache_index"]["filename"] == (
        "local_to_global_cache_index.npy"
    )


def test_v8r4_pack_free_validator_rejects_legacy_receipt_path() -> None:
    legacy = PROJECT_ROOT / benchmark.LEGACY_BENCHMARK_RECEIPT_RELATIVE
    with pytest.raises(benchmark.BenchmarkError, match="canonical V8R4"):
        benchmark.validate_benchmark_receipt_pack_free(
            project_root=PROJECT_ROOT, receipt_path=legacy
        )


def test_v8r4_pack_free_validator_never_reopens_peer_capabilities() -> None:
    source = inspect.getsource(benchmark.validate_benchmark_receipt_pack_free)
    assert "validate_target_sealed_capability(" not in source
    assert "_validate_receipt_lifecycle_invocations(" not in source
    assert "validate_worker_telemetry(" not in source
    assert "load_validate_terminal_result(" not in source


def test_v8r4_pack_free_validator_accepts_rich_capability_projection(
    tmp_path: Path,
) -> None:
    authority_source = PROJECT_ROOT / benchmark.EXECUTION_CLOSURE_AUTHORIZATION_RELATIVE
    authority = tmp_path / benchmark.EXECUTION_CLOSURE_AUTHORIZATION_RELATIVE
    authority.parent.mkdir(parents=True, exist_ok=True)
    authority.write_bytes(authority_source.read_bytes())
    authority.chmod(0o444)
    attempt = tmp_path / benchmark.BENCHMARK_RUN_ROOT_RELATIVE / "attempts/attempt_000"
    invocation = attempt / "invocation.json"
    result = attempt / "GPU_TERMINAL_RESULT.json"
    telemetry = attempt / "QUARANTINED_TIMING_TELEMETRY.json"
    terminal_sha = "a" * 64
    binding = lambda path, digest="b" * 64, size=1: {
        "path": str(path.resolve()), "sha256": digest, "bytes": size
    }
    payload = {
        key: None for key in benchmark._receipt_expected_keys() if key != "content_sha256"
    }
    payload.update(
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_v8r4_target_sealed_efficiency_benchmark_completion",
            "campaign_id": benchmark.CAMPAIGN_ID,
            "campaign_revision": benchmark.CAMPAIGN_REVISION,
            "infrastructure_revision": benchmark.INFRASTRUCTURE_REVISION,
            "phase": benchmark.BENCHMARK_PHASE,
            "benchmark_id": benchmark.BENCHMARK_ID,
            "unit": benchmark.BENCHMARK_UNIT,
            "usage_identity": dict(benchmark.BENCHMARK_USAGE_IDENTITY),
            "profile_sha256": benchmark.BENCHMARK_PROFILE_SHA256,
            "epochs": 2,
            "epoch_1_is_warmup": True,
            "epoch_2_train_ns": 10,
            "epoch_2_target_free_validation_ns": 11,
            "epoch_2_train_plus_target_free_validation_ns": 21,
            "epoch_2_gate_ns_max": benchmark.STEADY_GATE_NS,
            "gate_passed": True,
            "outer_test_opened": False,
            "accuracy_metrics_emitted_or_used": False,
            "checkpoint_selection_performed": False,
            "training_result_reusable": False,
            "selection_or_promotion_input": False,
            "artifacts_quarantined": True,
            "artifact_disposition": "quarantined_timing_telemetry_only",
            "same_usage_and_execution_ledgers": True,
            "benchmark_before_first_active_discovery_terminal": True,
            "all_failed_reconciled_and_successful_attempts_owned_by_one_receipt": True,
            "legacy_v8r3_success_quarantined": True,
            "legacy_combined_cache_used_by_active_attempt": False,
            "physical_nonouter_pack_only": True,
            "production_execution_authorized": True,
            "v8r4a_ledger_migration_required": False,
            "commercial_claim_authorized": False,
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
            "usage_ledger_path": str(
                (tmp_path / benchmark.DEFAULT_USAGE_LEDGER).resolve()
            ),
            "gpu_execution_ledger_path": str(
                (tmp_path / benchmark.DEFAULT_EXECUTION_LEDGER).resolve()
            ),
            "gpu_admission_lock_path": str(
                (tmp_path / benchmark.DEFAULT_GPU_LOCK).resolve()
            ),
            "training_index": {
                "path": str(tmp_path / benchmark.DEFAULT_TRAINING_INDEX),
                "sha256": benchmark.DEFAULT_TRAINING_INDEX_SHA256,
                "bytes": benchmark.DEFAULT_TRAINING_INDEX_BYTES,
            },
            "benchmark_invocation": binding(
                tmp_path / benchmark.BENCHMARK_RUN_ROOT_RELATIVE
                / benchmark.CURRENT_UNIT_INVOCATION_NAME
            ),
            "pretrain_authorization": binding(
                tmp_path / benchmark.PRETRAIN_AUTHORIZATION_RELATIVE
            ),
            "gpu_state_migration": binding(
                tmp_path / benchmark.GPU_STATE_MIGRATION_RECEIPT_RELATIVE
            ),
            "contract": binding(
                tmp_path / benchmark.CONTRACT_RELATIVE,
                digest=benchmark.CONTRACT_FILE_SHA256,
            ),
            "trainer": binding(tmp_path / benchmark.TRAINER_RELATIVE),
            "gpu_wrapper": binding(tmp_path / benchmark.WRAPPER_RELATIVE),
            "target_sealed_capability": {
                **binding(
                    tmp_path / benchmark.BENCHMARK_LIFECYCLE_RELATIVE
                    / benchmark.TARGET_SEALED_CAPABILITY_NAME
                ),
                "st_dev": 1,
                "st_ino": 2,
                "mode": "0444",
            },
            "historical_benchmark_attempts": [
                dict(item) for item in benchmark.HISTORICAL_BENCHMARK_ATTEMPTS
            ],
            "historical_projection_authority": benchmark._validate_historical_projection_authority(
                tmp_path
            ),
            "active_scientific_attempt_count": 1,
            "killed_lifecycle_replay_only": True,
            "attempt_count": 1,
            "lifecycle_invocations": [binding(invocation)],
            "terminal_results": [
                {
                    "terminal_record_sha256": terminal_sha,
                    "execution_invocation": binding(invocation),
                    "result": binding(result),
                }
            ],
            "telemetry": binding(telemetry),
            "usage_record_sha256": terminal_sha,
            "usage_record_sha256s": [terminal_sha],
        }
    )
    receipt_path = tmp_path / benchmark.BENCHMARK_RECEIPT_RELATIVE
    benchmark.create_once_json(receipt_path, payload)
    validated = benchmark.validate_benchmark_receipt_pack_free(
        project_root=tmp_path,
        receipt_path=receipt_path,
        expected_pretrain_authorization=payload["pretrain_authorization"],
    )
    assert validated["attempt_count"] == 1


def test_v8r4_benchmark_replays_v8r4a_migrated_state_capability(
    tmp_path: Path,
) -> None:
    expected_paths = {
        "admission_lock": (PROJECT_ROOT / benchmark.DEFAULT_GPU_LOCK).resolve(),
        "execution_ledger": (
            PROJECT_ROOT / benchmark.DEFAULT_EXECUTION_LEDGER
        ).resolve(),
        "usage_ledger": (PROJECT_ROOT / benchmark.DEFAULT_USAGE_LEDGER).resolve(),
    }
    migration = benchmark.require_v8r4a_benchmark_runtime(
        project_root=PROJECT_ROOT,
        run_root=PROJECT_ROOT / benchmark.BENCHMARK_RUN_ROOT_RELATIVE,
        discovery_module=SimpleNamespace(
            validate_v8r4a_gpu_state=lambda _root: {
                "migration_receipt": {
                    "path": "receipt",
                    "sha256": "a" * 64,
                    "bytes": 1,
                },
                "canonical_paths": expected_paths,
            }
        ),
    )
    assert migration is not None
    assert migration["canonical_paths"] == expected_paths
    benchmark.require_v8r4a_benchmark_runtime(
        project_root=tmp_path, run_root=tmp_path / "synthetic-isolated-run"
    )
