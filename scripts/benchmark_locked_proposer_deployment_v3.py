#!/usr/bin/env python3
"""Benchmark all 18 locked proposer-serving checkpoints without labels.

The campaign is intentionally separate from HCS evaluation.  It validates the
complete fixed-i3 pre-test index and its 18 unit locks, then follows only the
``hcs_validation`` provenance edge in each strict proposer stack.  Those are
the serving proposer checkpoints; the dormant HCS checkpoints are verified as
locked inputs but are never benchmarked or ranked.

Use ``freeze-spec`` before any outer-test inference, then ``run`` on the host
whose CPU or CUDA device is to be measured.  Results describe that current
host only, are never a target-device or commercial-performance claim, and are
sealed only after all 18 fixed units have receipts.

Version 3 preserves the v2 protocol and adds an explicit CUDA runtime/device
initialization before resetting allocator peak statistics.  PyTorch 2.13 can
otherwise report ``Invalid device argument`` when peak statistics are reset in
a fresh worker whose availability check used the NVML path and has not created
a CUDA context yet.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_e2e as E2E  # noqa: E402

torch = E2E.torch

SCHEMA_VERSION = 1
FOLDS = tuple(range(6))
SEEDS = (20260828, 20260829, 20260830)
EXPECTED_UNITS = 18

DEFAULT_PRETEST_INDEX = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_fixed_i3_pretest/pretest_index.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/benchmarks/locked_proposer_deployment"
DEFAULT_FREEZE_SPEC = DEFAULT_OUTPUT_ROOT / "freeze_spec.json"


class LockedBenchmarkError(RuntimeError):
    """A fail-closed specification, provenance, or receipt error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LockedBenchmarkError(f"invalid {label}: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise LockedBenchmarkError(f"{label} must be a JSON object: {path}")
    return value


def _payload(value: Any) -> str:
    return json.dumps(
        _strict_json_value(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _atomic_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_payload(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if immutable:
            path.chmod(0o444)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def bind_file(path: Path, *, recorded_path: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise LockedBenchmarkError(f"required file is absent: {resolved}")
    return {
        "path": str((recorded_path or resolved).expanduser().resolve()),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def bind_python_launcher(path: Path) -> dict[str, Any]:
    """Hash the interpreter binary without erasing its venv launcher path."""

    launcher = Path(os.path.abspath(path.expanduser()))
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise LockedBenchmarkError("Python launcher must be an executable file")
    binding = bind_file(launcher)
    return {**binding, "path": str(launcher)}


def _resolve(value: Any, *, relative_to: Path) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise LockedBenchmarkError("artifact path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _binding(raw: Any, *, relative_to: Path, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise LockedBenchmarkError(f"missing binding: {label}")
    path = _resolve(raw.get("path"), relative_to=relative_to)
    expected = str(raw.get("sha256", ""))
    if len(expected) != 64 or not path.is_file() or sha256_file(path) != expected:
        raise LockedBenchmarkError(f"file hash mismatch: {label} ({path})")
    if "bytes" in raw and int(raw["bytes"]) != path.stat().st_size:
        raise LockedBenchmarkError(f"file size mismatch: {label} ({path})")
    return {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        Path(str(left.get("path", ""))).resolve()
        == Path(str(right.get("path", ""))).resolve()
        and str(left.get("sha256", "")) == str(right.get("sha256", ""))
        and int(left.get("bytes", -1)) == int(right.get("bytes", -2))
    )


def _ensure_document(path: Path, expected: Mapping[str, Any]) -> None:
    if path.exists():
        if _json(path, path.name) != expected:
            raise LockedBenchmarkError(f"immutable document differs: {path}")
        return
    _atomic_json(path, expected, immutable=True)


def _content_document(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document.pop("content_sha256", None)
    document["content_sha256"] = canonical_sha256(document)
    return document


def _validate_content_hash(value: Mapping[str, Any], *, label: str) -> None:
    content = dict(value)
    recorded = str(content.pop("content_sha256", ""))
    if len(recorded) != 64 or canonical_sha256(content) != recorded:
        raise LockedBenchmarkError(f"{label} content hash mismatch")


def _measurement_source_bindings() -> dict[str, Any]:
    paths = {
        "orchestrator": Path(__file__),
        "benchmark_e2e": Path(E2E.__file__),
        "proposer_model_builder": PROJECT_ROOT / "scripts/train.py",
        "raw_feature_builder": PROJECT_ROOT / "scripts/build_features.py",
        "cache_contract": PROJECT_ROOT / "src/snn_rr/cache.py",
        "data_contract": PROJECT_ROOT / "src/snn_rr/data.py",
        "preprocess_contract": PROJECT_ROOT / "src/snn_rr/preprocess.py",
        "model_contract": PROJECT_ROOT / "src/snn_rr/models.py",
        "python_executable": Path(sys.executable),
    }
    return {
        name: (bind_python_launcher(path) if name == "python_executable" else bind_file(path))
        for name, path in paths.items()
    }


def default_freeze_spec() -> dict[str, Any]:
    """Return the deterministic, target-independent engineering benchmark spec."""

    value = {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_proposer_deployment_benchmark_freeze_spec",
        "target_or_label_artifact_consulted": False,
        "sealed_training_artifacts_byte_hashed_for_provenance": True,
        "target_or_label_arrays_deserialized": False,
        "target_or_label_values_used_for_measurement_or_selection": False,
        "outer_test_inference_required_to_create_spec": False,
        "commercial_performance_claim_authorized": False,
        "measurement_scope": "current_host_not_target_device",
        "matrix": {
            "folds": list(FOLDS),
            "seeds": list(SEEDS),
            "unit_count": EXPECTED_UNITS,
            "serving_role": "hcs_validation",
            "all_units_required": True,
            "best_unit_selection_allowed": False,
            "aggregate_only_after_all_18": True,
        },
        "measurement": {
            "input": "deterministic_synthetic_raw_3_radar_window_resident_in_host_memory",
            "input_seed": 20260828,
            "batch_size": 1,
            "warmup_repeats": 10,
            "measured_repeats": 50,
            "cpu_threads": 1,
            "history_state": "zero_initialized_label_free_cold_session_state",
            "stride_budget_ms": 4000.0,
            "cold_contract": (
                "fresh worker process model/checkpoint load plus first resident raw-window "
                "preprocess, transfer, and synchronized inference; OS page cache is uncontrolled"
            ),
            "warm_contract": "same loaded model, serialized batch-one resident raw-window trials",
            "profiles": {
                "cpu": {"device_type": "cpu", "amp": False},
                "cuda": {"device_type": "cuda", "amp": True},
            },
        },
        "engineering_gates": {
            "cpu_raw_resident_warm_p99_ms_max": 250.0,
            "cuda_raw_resident_warm_p99_ms_max": 50.0,
            "p99_stride_budget_fraction_max": 0.10,
            "checkpoint_bytes_max": 50 * 1024 * 1024,
            "parameter_count_max": 5_000_000,
            "cpu_process_peak_rss_bytes_max": 2 * 1024**3,
            "cuda_peak_reserved_bytes_max": 1024**3,
            "spike_rate_diagnostic_min": 0.01,
            "spike_rate_diagnostic_max": 0.20,
            "spike_rate_unavailable_policy": "reported_not_applicable_without_failure",
        },
        "memory_caveats": {
            "cpu_peak_rss": (
                "process high-water RSS from an isolated one-unit worker; it includes Python, "
                "PyTorch, preprocessing, allocator retention, and shared-page accounting, is "
                "not isolated to model memory, is not an incremental model-only measurement, "
                "and is not a target-device proxy"
            ),
            "cuda": "PyTorch peak allocated/reserved for this worker CUDA context",
        },
        "measurement_sources": _measurement_source_bindings(),
    }
    return _content_document(value)


def freeze_spec(path: Path) -> dict[str, Any]:
    expected = default_freeze_spec()
    resolved = path.expanduser().resolve()
    _ensure_document(resolved, expected)
    return {
        "status": "deployment_benchmark_spec_frozen",
        "freeze_spec": bind_file(resolved),
        "content_sha256": expected["content_sha256"],
        "target_or_label_artifact_consulted": False,
        "commercial_performance_claim_authorized": False,
    }


def load_freeze_spec(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    document = _json(resolved, "deployment benchmark freeze spec")
    _validate_content_hash(document, label="freeze spec")
    expected = default_freeze_spec()
    if document != expected:
        raise LockedBenchmarkError("freeze spec differs from the current locked default spec")
    if (
        document.get("target_or_label_artifact_consulted") is not False
        or document.get("commercial_performance_claim_authorized") is not False
        or document.get("matrix", {}).get("best_unit_selection_allowed") is not False
    ):
        raise LockedBenchmarkError("freeze spec permits labels, commercial claims, or selection")
    return document, bind_file(resolved)


def _validate_freeze(index: Mapping[str, Any], *, index_path: Path) -> dict[str, Any]:
    common = index.get("common")
    if not isinstance(common, Mapping):
        raise LockedBenchmarkError("pretest index common bindings are missing")
    bindings = {
        name: _binding(common.get(name), relative_to=index_path.parent, label=f"common {name}")
        for name in ("selection_lock", "capacity_selection", "policy", "source_freeze_manifest")
    }
    freeze = _json(Path(bindings["source_freeze_manifest"]["path"]), "source freeze")
    if freeze.get("outer_test_opened") is not False or freeze.get(
        "declared_before_any_i3_score"
    ) is not True:
        raise LockedBenchmarkError("source freeze is not the pre-i3 unopened freeze")
    files = freeze.get("files")
    if not isinstance(files, Mapping):
        raise LockedBenchmarkError("source freeze file mapping is missing")
    root = Path(bindings["source_freeze_manifest"]["path"]).parent
    for name, expected in files.items():
        path = root / str(name)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise LockedBenchmarkError(f"frozen source hash mismatch: {name}")
    selection = _json(Path(bindings["selection_lock"]["path"]), "common selection lock")
    capacity = _json(Path(bindings["capacity_selection"]["path"]), "capacity selection")
    policy = _json(Path(bindings["policy"]["path"]), "common policy")
    if (
        selection.get("outer_test_opened_before_lock") is not False
        or capacity.get("outer_test_opened") is not False
        or policy.get("outer_test_opened") is not False
        or selection.get("capacity_selection_sha256")
        != bindings["capacity_selection"]["sha256"]
        or selection.get("common_fallback_policy_sha256") != bindings["policy"]["sha256"]
        or selection.get("source_freeze") != files
        or capacity.get("source_freeze_manifest_sha256")
        != bindings["source_freeze_manifest"]["sha256"]
    ):
        raise LockedBenchmarkError("common capacity/policy/source freeze bindings differ")
    return {"bindings": bindings, "files": dict(files)}


def _scalar_string(archive: Mapping[str, Any], name: str) -> str:
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise LockedBenchmarkError(f"proposer prediction scalar is invalid: {name}")
    return str(value.item())


def _source_hash_bindings(raw: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not raw:
        raise LockedBenchmarkError(f"{label} source-hash mapping is missing")
    result: dict[str, Any] = {}
    for name, expected in sorted(raw.items()):
        path = _resolve(str(name), relative_to=PROJECT_ROOT)
        binding = bind_file(path)
        if binding["sha256"] != str(expected):
            raise LockedBenchmarkError(f"{label} source hash mismatch: {name}")
        result[str(name)] = binding
    return result


def _serving_proposer_from_stack(
    stack_binding: Mapping[str, Any], *, fold: int, seed: int
) -> dict[str, Any]:
    stack_path = Path(str(stack_binding["path"]))
    try:
        # Deliberately deserialize only the provenance scalar.  The training
        # stack also contains non-test reference arrays; they are never read by
        # this benchmark.
        with np.load(stack_path, allow_pickle=False) as archive:
            provenance = json.loads(str(np.asarray(archive["provenance_json"]).item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise LockedBenchmarkError(f"invalid strict-stack provenance: {stack_path} ({exc})") from exc
    if (
        not isinstance(provenance, dict)
        or provenance.get("strict_nested") is not True
        or provenance.get("outer_test_opened") is not False
        or int(provenance.get("outer_fold", -1)) != fold
        or int(provenance.get("seed", -1)) != seed
    ):
        raise LockedBenchmarkError(f"strict-stack identity/leakage mismatch: {fold}/{seed}")
    stack_sources = _source_hash_bindings(
        provenance.get("source_code_sha256"), label=f"strict stack {fold}/{seed}"
    )
    source_units = provenance.get("source_units")
    if not isinstance(source_units, list):
        raise LockedBenchmarkError("strict stack has no source-unit provenance")
    selected = [unit for unit in source_units if unit.get("role") == "hcs_validation"]
    if len(selected) != 1:
        raise LockedBenchmarkError(f"unit must have exactly one serving proposer: {fold}/{seed}")
    source = selected[0]
    prediction = _binding(
        {"path": source.get("path"), "sha256": source.get("sha256")},
        relative_to=stack_path.parent,
        label=f"validation proposer prediction {fold}/{seed}",
    )
    checkpoint_path = _resolve(source.get("checkpoint"), relative_to=stack_path.parent)
    checkpoint = bind_file(checkpoint_path)
    run_config = _binding(
        {"path": source.get("run_config"), "sha256": source.get("run_config_sha256")},
        relative_to=stack_path.parent,
        label=f"validation proposer run config {fold}/{seed}",
    )
    manifest = _binding(
        {"path": source.get("manifest"), "sha256": source.get("manifest_file_sha256")},
        relative_to=stack_path.parent,
        label=f"validation proposer split manifest {fold}/{seed}",
    )
    try:
        with np.load(Path(prediction["path"]), allow_pickle=False) as archive:
            # Like the strict stack, this retrospective training artifact can
            # contain reference arrays.  Only three scalar provenance fields
            # are deserialized; no row prediction/reference array is read.
            recorded_checkpoint = _scalar_string(archive, "checkpoint_sha256")
            recorded_run_config = _scalar_string(archive, "run_config_sha256")
            prediction_provenance = json.loads(_scalar_string(archive, "provenance_json"))
    except LockedBenchmarkError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise LockedBenchmarkError(f"invalid serving prediction provenance: {exc}") from exc
    if recorded_checkpoint != checkpoint["sha256"] or prediction_provenance.get(
        "checkpoint_sha256"
    ) != checkpoint["sha256"]:
        raise LockedBenchmarkError("serving prediction/checkpoint hash mismatch")
    if recorded_run_config != run_config["sha256"] or prediction_provenance.get(
        "run_config_sha256"
    ) != run_config["sha256"]:
        raise LockedBenchmarkError("serving prediction/run-config hash mismatch")
    if prediction_provenance.get("labels_forwarded_to_model") is not False:
        raise LockedBenchmarkError("serving proposer prediction permits labels in the model")
    checkpoint_sources = _source_hash_bindings(
        prediction_provenance.get("source_hashes"),
        label=f"serving proposer {fold}/{seed}",
    )

    run_document = _json(Path(run_config["path"]), "serving proposer run config")
    arguments = run_document.get("arguments")
    if not isinstance(arguments, Mapping) or not run_document.get("run_signature"):
        raise LockedBenchmarkError("serving proposer run config is incomplete")
    config_path = _resolve(arguments.get("config"), relative_to=PROJECT_ROOT)
    pipeline_config = bind_file(config_path)
    if run_document.get("config_sha256") is not None and str(
        run_document["config_sha256"]
    ) != pipeline_config["sha256"]:
        raise LockedBenchmarkError("serving run config/pipeline config hash mismatch")
    authority = run_document.get("split_authority")
    if not isinstance(authority, Mapping) or authority.get(
        "split_manifest_file_sha256"
    ) != manifest["sha256"]:
        raise LockedBenchmarkError("serving run config/split manifest hash mismatch")
    declared_manifest = arguments.get("identity_split_manifest")
    if declared_manifest and _resolve(declared_manifest, relative_to=PROJECT_ROOT) != Path(
        manifest["path"]
    ):
        raise LockedBenchmarkError("serving run config names another split manifest")
    return {
        "role": "hcs_validation",
        "strict_stack": dict(stack_binding),
        "strict_stack_source_bindings": stack_sources,
        "stack_provenance_fields_read": ["provenance_json"],
        "stack_reference_or_target_arrays_read": False,
        "checkpoint": checkpoint,
        "run_config": run_config,
        "pipeline_config": pipeline_config,
        "split_manifest": manifest,
        "source_prediction": prediction,
        "checkpoint_source_bindings": checkpoint_sources,
        "source_prediction_provenance_fields_read": [
            "checkpoint_sha256",
            "run_config_sha256",
            "provenance_json",
        ],
        "source_prediction_reference_or_target_arrays_read": False,
    }


def validate_pretest_index(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    index_path = path.expanduser().resolve()
    index = _json(index_path, "fixed-i3 pretest index")
    _validate_content_hash(index, label="fixed-i3 pretest index")
    matrix = index.get("matrix")
    if (
        index.get("schema_version") != 1
        or index.get("classification") != "retrospective_fixed_i3_pretest_index"
        or index.get("status") != "complete"
        or int(index.get("completed_units", -1)) != EXPECTED_UNITS
        or index.get("outer_test_opened") is not False
        or index.get("commercial_claim_authorized") is not False
        or not isinstance(matrix, Mapping)
        or matrix.get("folds") != list(FOLDS)
        or matrix.get("seeds") != list(SEEDS)
        or int(matrix.get("unit_count", -1)) != EXPECTED_UNITS
    ):
        raise LockedBenchmarkError("pretest index is not the final unopened fixed 18-unit seal")
    freeze = _validate_freeze(index, index_path=index_path)
    units_raw = index.get("units")
    if not isinstance(units_raw, list) or len(units_raw) != EXPECTED_UNITS:
        raise LockedBenchmarkError("pretest index does not contain 18 units")
    unit_map: dict[tuple[int, int], Mapping[str, Any]] = {}
    for unit in units_raw:
        if not isinstance(unit, Mapping) or unit.get("status") != "complete":
            raise LockedBenchmarkError("pretest index contains an incomplete unit")
        key = (int(unit.get("outer_fold", -1)), int(unit.get("seed", -1)))
        if key in unit_map:
            raise LockedBenchmarkError(f"duplicate pretest unit: {key}")
        unit_map[key] = unit
    expected = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    if set(unit_map) != expected:
        raise LockedBenchmarkError("pretest index unit topology is not fixed 6 x 3")

    records: list[dict[str, Any]] = []
    lock_hash_fields = {
        "checkpoint": "checkpoint_sha256",
        "scaler": "scaler_sha256",
        "cache_manifest": "cache_manifest_sha256",
        "fallback_oof": "fallback_oof_sha256",
        "run_manifest": "run_manifest_sha256",
        "original_policy": "policy_sha256",
        "history": "history_sha256",
    }
    source_names = {
        "trainer": "train_harmonic_set_snn.py",
        "harmonic_set_model": "harmonic_set_models.py",
        "campaign_config": "harmonic_set_v2.yaml",
        "adaptive_campaign_contract": "ADAPTIVE_CAMPAIGN_CONTRACT.json",
    }
    for fold in FOLDS:
        for seed in SEEDS:
            raw = unit_map[(fold, seed)]
            artifacts_raw = raw.get("artifacts")
            if not isinstance(artifacts_raw, Mapping):
                raise LockedBenchmarkError(f"pretest unit artifacts are missing: {fold}/{seed}")
            required_names = (*lock_hash_fields, "selection_lock", "strict_stack")
            artifacts = {
                name: _binding(
                    artifacts_raw.get(name),
                    relative_to=index_path.parent,
                    label=f"pretest unit {fold}/{seed}/{name}",
                )
                for name in required_names
            }
            lock = _json(Path(artifacts["selection_lock"]["path"]), "i3 unit selection lock")
            if (
                int(lock.get("outer_fold", -1)) != fold
                or int(lock.get("seed", -1)) != seed
                or int(lock.get("adaptive_iteration", -1)) != 3
                or lock.get("outer_test_not_opened_before_this_lock") is not True
            ):
                raise LockedBenchmarkError(f"i3 unit lock identity/leakage mismatch: {fold}/{seed}")
            for name, field in lock_hash_fields.items():
                if lock.get(field) != artifacts[name]["sha256"]:
                    raise LockedBenchmarkError(f"i3 unit lock hash mismatch: {fold}/{seed}/{name}")
            source_bindings = lock.get("source_bindings")
            if not isinstance(source_bindings, Mapping):
                raise LockedBenchmarkError(f"i3 unit source bindings are missing: {fold}/{seed}")
            for source_name, freeze_name in source_names.items():
                binding = source_bindings.get(source_name)
                if not isinstance(binding, Mapping) or binding.get("sha256") != freeze["files"].get(
                    freeze_name
                ):
                    raise LockedBenchmarkError(
                        f"i3 unit frozen source mismatch: {fold}/{seed}/{source_name}"
                    )
            proposer = _serving_proposer_from_stack(artifacts["strict_stack"], fold=fold, seed=seed)
            records.append(
                {
                    "unit_id": f"outer_{fold}_seed_{seed}",
                    "outer_fold": fold,
                    "seed": seed,
                    "pretest_unit_lock": artifacts["selection_lock"],
                    "verified_hcs_artifacts_not_benchmarked": {
                        name: artifacts[name] for name in lock_hash_fields
                    },
                    "serving_proposer": proposer,
                    "selection_basis": "fixed_hcs_validation_role_only_no_metric_or_best_unit_selection",
                }
            )
    return index, bind_file(index_path), records


def _normalize_device(value: str) -> tuple[str, str]:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise LockedBenchmarkError(f"invalid benchmark device: {value}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise LockedBenchmarkError("benchmark device must be CPU or CUDA")
    if device.type == "cpu" and str(device) != "cpu":
        raise LockedBenchmarkError("CPU benchmark device must be exactly 'cpu'")
    return str(device), device.type


def build_plan(
    *,
    freeze_spec_path: Path,
    pretest_index: Path,
    output_root: Path,
    device_name: str,
) -> dict[str, Any]:
    spec, spec_binding = load_freeze_spec(freeze_spec_path)
    _, index_binding, records = validate_pretest_index(pretest_index)
    device, device_type = _normalize_device(device_name)
    measurement = spec["measurement"]
    profile = measurement["profiles"][device_type]
    runtime = {
        "requested_device": device,
        "device_type": device_type,
        "amp": bool(profile["amp"]),
        "input": measurement["input"],
        "input_seed": int(measurement["input_seed"]),
        "batch_size": int(measurement["batch_size"]),
        "warmup_repeats": int(measurement["warmup_repeats"]),
        "measured_repeats": int(measurement["measured_repeats"]),
        "cpu_threads": int(measurement["cpu_threads"]),
        "history_state": measurement["history_state"],
        "stride_budget_ms": float(measurement["stride_budget_ms"]),
    }
    output = output_root.expanduser().resolve()
    executable = bind_python_launcher(Path(sys.executable))
    orchestrator = bind_file(Path(__file__))
    if executable["sha256"] != spec["measurement_sources"]["python_executable"]["sha256"]:
        raise LockedBenchmarkError("runtime Python differs from the frozen spec")
    if orchestrator["sha256"] != spec["measurement_sources"]["orchestrator"]["sha256"]:
        raise LockedBenchmarkError("benchmark orchestrator differs from the frozen spec")
    for name, frozen in spec["measurement_sources"].items():
        _binding(frozen, relative_to=Path(spec_binding["path"]).parent, label=f"spec source {name}")

    units = []
    for record in records:
        final_root = output / "units" / record["unit_id"]
        contract_path = final_root / "worker_contract.json"
        benchmark_path = final_root / "benchmark.json"
        units.append(
            {
                **record,
                "freeze_spec": spec_binding,
                "freeze_spec_content_sha256": spec["content_sha256"],
                "runtime": runtime,
                "worker_contract_path": str(contract_path),
                "benchmark_output_path": str(benchmark_path),
                "worker_argv": [
                    executable["path"],
                    orchestrator["path"],
                    "worker",
                    "--contract",
                    str(contract_path),
                    "--output",
                    str(benchmark_path),
                ],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_proposer_deployment_benchmark_plan",
        "freeze_spec": spec_binding,
        "freeze_spec_content_sha256": spec["content_sha256"],
        "pretest_index": index_binding,
        "matrix": spec["matrix"],
        "runtime": runtime,
        "engineering_gates": spec["engineering_gates"],
        "memory_caveats": spec["memory_caveats"],
        "measurement_sources": spec["measurement_sources"],
        "unit_count": EXPECTED_UNITS,
        "execution_order": [record["unit_id"] for record in records],
        "serial_execution": True,
        "best_unit_selection_allowed": False,
        "target_or_label_artifact_read": False,
        "sealed_pretest_artifacts_byte_hashed_for_provenance": True,
        "target_or_label_arrays_deserialized": False,
        "target_or_label_values_used_for_measurement_or_selection": False,
        "measurement_scope": "current_host_not_target_device",
        "commercial_performance_claim_authorized": False,
        "units": units,
    }


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _spike_activity(output: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in ("spike_rate", "spike_rate_per_sample", "layer_spike_rates"):
        value = output.get(name)
        if isinstance(value, torch.Tensor):
            array = value.detach().float().cpu().numpy()
            if array.size and np.isfinite(array).all():
                fields[name] = {
                    "mean": float(array.mean()),
                    "min": float(array.min()),
                    "max": float(array.max()),
                    "shape": list(array.shape),
                    "values": array.reshape(-1).tolist(),
                }
    return {
        "available": bool(fields),
        "target_or_label_used": False,
        "fields": fields,
        "overall_rate": (
            fields.get("spike_rate", fields.get("spike_rate_per_sample", {})).get("mean")
            if fields
            else None
        ),
    }


def _evaluate_unit_gates(report: Mapping[str, Any], gates: Mapping[str, Any]) -> dict[str, Any]:
    device_type = str(report["runtime"]["device_type"])
    p99 = float(report["warm"]["total_latency"]["p99_ms"])
    latency_limit = float(gates[f"{device_type}_raw_resident_warm_p99_ms_max"])
    stride_fraction = p99 / float(report["runtime"]["stride_budget_ms"])
    spike = report["spike_activity"]
    spike_rate = spike.get("overall_rate")
    spike_result = {
        "available": spike_rate is not None,
        "value": spike_rate,
        "minimum": float(gates["spike_rate_diagnostic_min"]),
        "maximum": float(gates["spike_rate_diagnostic_max"]),
        "pass": (
            True
            if spike_rate is None
            else float(gates["spike_rate_diagnostic_min"])
            <= float(spike_rate)
            <= float(gates["spike_rate_diagnostic_max"])
        ),
        "unavailable_policy": gates["spike_rate_unavailable_policy"],
    }
    results: dict[str, Any] = {
        "warm_p99_device_limit": {
            "value_ms": p99,
            "maximum_ms": latency_limit,
            "pass": p99 <= latency_limit,
        },
        "warm_p99_stride_fraction": {
            "value": stride_fraction,
            "maximum": float(gates["p99_stride_budget_fraction_max"]),
            "pass": stride_fraction <= float(gates["p99_stride_budget_fraction_max"]),
        },
        "checkpoint_bytes": {
            "value": int(report["model"]["checkpoint_bytes"]),
            "maximum": int(gates["checkpoint_bytes_max"]),
            "pass": int(report["model"]["checkpoint_bytes"])
            <= int(gates["checkpoint_bytes_max"]),
        },
        "parameter_count": {
            "value": int(report["model"]["total_parameters"]),
            "maximum": int(gates["parameter_count_max"]),
            "pass": int(report["model"]["total_parameters"])
            <= int(gates["parameter_count_max"]),
        },
        "cpu_process_peak_rss": {
            "value_bytes": int(report["memory"]["cpu_process_peak_rss_bytes"]),
            "maximum_bytes": int(gates["cpu_process_peak_rss_bytes_max"]),
            "pass": int(report["memory"]["cpu_process_peak_rss_bytes"])
            <= int(gates["cpu_process_peak_rss_bytes_max"]),
            "caveat": report["memory"]["cpu_peak_rss_caveat"],
        },
        "spike_rate_diagnostic": spike_result,
    }
    if device_type == "cuda":
        results["cuda_peak_reserved"] = {
            "value_bytes": int(report["memory"]["cuda_peak_reserved_bytes"]),
            "maximum_bytes": int(gates["cuda_peak_reserved_bytes_max"]),
            "pass": int(report["memory"]["cuda_peak_reserved_bytes"])
            <= int(gates["cuda_peak_reserved_bytes_max"]),
        }
    required = [value["pass"] for value in results.values()]
    return {"all_applicable_pass": bool(all(required)), "results": results}


def _initialize_cuda_measurement(device: torch.device) -> None:
    """Initialize the selected CUDA context before allocator peak accounting."""

    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def run_worker(contract_path: Path, output_path: Path) -> dict[str, Any]:
    """Run one isolated serving-checkpoint benchmark from a sealed contract."""

    contract = _json(contract_path.expanduser().resolve(), "benchmark worker contract")
    _validate_content_hash(contract, label="worker contract")
    _binding(contract.get("plan"), relative_to=contract_path.parent, label="campaign plan")
    spec, spec_binding = load_freeze_spec(Path(contract["freeze_spec"]["path"]))
    if not _same_binding(spec_binding, contract["freeze_spec"]):
        raise LockedBenchmarkError("worker contract freeze-spec binding differs")
    runtime = contract.get("runtime")
    if not isinstance(runtime, Mapping):
        raise LockedBenchmarkError("worker runtime contract is missing")
    device_name, device_type = _normalize_device(str(runtime.get("requested_device")))
    profile = spec["measurement"]["profiles"][device_type]
    expected_runtime = {
        "requested_device": device_name,
        "device_type": device_type,
        "amp": bool(profile["amp"]),
        "input": spec["measurement"]["input"],
        "input_seed": int(spec["measurement"]["input_seed"]),
        "batch_size": int(spec["measurement"]["batch_size"]),
        "warmup_repeats": int(spec["measurement"]["warmup_repeats"]),
        "measured_repeats": int(spec["measurement"]["measured_repeats"]),
        "cpu_threads": int(spec["measurement"]["cpu_threads"]),
        "history_state": spec["measurement"]["history_state"],
        "stride_budget_ms": float(spec["measurement"]["stride_budget_ms"]),
    }
    if dict(runtime) != expected_runtime:
        raise LockedBenchmarkError("worker runtime differs from the frozen spec")
    if device_type == "cuda" and not torch.cuda.is_available():
        raise LockedBenchmarkError("CUDA requested but unavailable on the current host")
    proposer = contract.get("serving_proposer")
    if not isinstance(proposer, Mapping):
        raise LockedBenchmarkError("worker serving proposer binding is missing")
    checkpoint = _binding(proposer.get("checkpoint"), relative_to=contract_path.parent, label="checkpoint")
    run_config = _binding(proposer.get("run_config"), relative_to=contract_path.parent, label="run config")
    pipeline_config = _binding(
        proposer.get("pipeline_config"), relative_to=contract_path.parent, label="pipeline config"
    )
    for name in ("strict_stack", "split_manifest", "source_prediction"):
        _binding(
            proposer.get(name),
            relative_to=contract_path.parent,
            label=f"serving proposer {name}",
        )
    for group_name in ("strict_stack_source_bindings", "checkpoint_source_bindings"):
        group = proposer.get(group_name)
        if not isinstance(group, Mapping) or not group:
            raise LockedBenchmarkError(f"serving proposer {group_name} is missing")
        for name, raw in group.items():
            _binding(
                raw,
                relative_to=contract_path.parent,
                label=f"serving proposer source {group_name}/{name}",
            )

    random.seed(int(runtime["input_seed"]))
    np.random.seed(int(runtime["input_seed"]))
    torch.manual_seed(int(runtime["input_seed"]))
    torch.set_num_threads(int(runtime["cpu_threads"]))
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(device_name)
    config = E2E.load_yaml_config(Path(pipeline_config["path"]))
    data_config = config["data"]
    frames = int(round(float(data_config["window_seconds"]) * float(data_config["radar_hz"])))
    raw = E2E.synthetic_raw_window(frames=frames, seed=int(runtime["input_seed"]))
    rss_before = _rss_bytes()
    if device_type == "cuda":
        _initialize_cuda_measurement(device)
    gc.collect()
    started = time.perf_counter_ns()
    prepared = E2E.prepare_checkpoint(Path(checkpoint["path"]))
    if prepared.run_config != _json(Path(run_config["path"]), "bound serving run config"):
        raise LockedBenchmarkError("benchmark_e2e discovered another run config")
    if prepared.model_type != "snn":
        raise LockedBenchmarkError("serving checkpoint is not an SNN proposer")
    model = E2E.build_model(prepared, device)
    feature = E2E.preprocess_raw_window(raw, data_config)
    if prepared.expected_aux_dim:
        base_dim = int(prepared.model_kwargs.get("aux_base_dim") or len(feature.base_aux))
        tail_dim = prepared.expected_aux_dim - base_dim
        if tail_dim < 0:
            raise LockedBenchmarkError("checkpoint auxiliary topology is invalid")
    else:
        tail_dim = 0
    history_tail = np.zeros(tail_dim, dtype=np.float32)
    radar_map, aux, radar_mask = E2E.construct_numpy_model_inputs(
        feature, prepared, history_tail
    )
    map_tensor, aux_tensor, mask_tensor = E2E.tensors_to_device(
        radar_map, aux, radar_mask, device
    )
    E2E._synchronize(device)  # noqa: SLF001
    with torch.inference_mode(), torch.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=bool(runtime["amp"] and device_type == "cuda"),
    ):
        cold_output = model(map_tensor, radar_mask=mask_tensor, aux=aux_tensor)
    E2E._synchronize(device)  # noqa: SLF001
    cold_ms = (time.perf_counter_ns() - started) / 1.0e6

    model_parameters = list(model.parameters())
    total_parameters = sum(parameter.numel() for parameter in model_parameters)
    trainable_parameters = sum(
        parameter.numel() for parameter in model_parameters if parameter.requires_grad
    )
    spike = _spike_activity(cold_output)
    expected_rr = float(cold_output["expected_rr"].detach().float().cpu().reshape(-1)[0])
    config_provenance = E2E.validate_pipeline_config_provenance(
        prepared, Path(pipeline_config["path"])
    )
    warm = E2E.run_latency_trials(
        model=model,
        prepared=prepared,
        data_config=data_config,
        history_tail=history_tail,
        device=device,
        raw_supplier=lambda: raw,
        include_raw_load=False,
        repeats=int(runtime["measured_repeats"]),
        warmup=int(runtime["warmup_repeats"]),
        amp=bool(runtime["amp"]),
    )
    total_latency = warm["stages"]["total_ms"]
    total_latency["p99_ms"] = float(np.quantile(total_latency["samples_ms"], 0.99))
    rss_after = _rss_bytes()
    memory = {
        "cpu_process_peak_rss_before_model_bytes": rss_before,
        "cpu_process_peak_rss_bytes": rss_after,
        "cpu_process_peak_rss_growth_bytes": max(0, rss_after - rss_before),
        "cpu_peak_rss_isolated_to_model_memory": False,
        "cpu_peak_rss_caveat": spec["memory_caveats"]["cpu_peak_rss"],
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device_type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device_type == "cuda" else None
        ),
        "cuda_memory_caveat": spec["memory_caveats"]["cuda"],
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_proposer_deployment_unit_benchmark",
        "unit_id": contract["unit_id"],
        "outer_fold": int(contract["outer_fold"]),
        "seed": int(contract["seed"]),
        "freeze_spec": spec_binding,
        "freeze_spec_content_sha256": spec["content_sha256"],
        "runtime": dict(runtime),
        "measurement_scope": "current_host_not_target_device",
        "commercial_performance_claim_authorized": False,
        "target_or_label_artifact_read": False,
        "target_or_label_arrays_deserialized": False,
        "target_or_label_values_used_for_measurement_or_selection": False,
        "serving_role": "hcs_validation",
        "model": {
            "checkpoint": checkpoint,
            "run_config": run_config,
            "pipeline_config": pipeline_config,
            "checkpoint_bytes": checkpoint["bytes"],
            "total_parameters": int(total_parameters),
            "trainable_parameters": int(trainable_parameters),
            "model_type": prepared.model_type,
            "model_kwargs": dict(prepared.model_kwargs),
            "pipeline_config_provenance": config_provenance,
        },
        "input": {
            "kind": "deterministic_synthetic_raw_resident",
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
            "history_state": "zero_initialized_label_free_cold_session_state",
        },
        "cold": {
            "model_load_plus_first_resident_raw_window_inference_ms": cold_ms,
            "logical_cold_fresh_process_and_model": True,
            "cold_disk_or_page_cache_claim": False,
            "first_expected_rr_bpm": expected_rr,
        },
        "warm": {
            "total_latency": total_latency,
            "p50_ms": float(total_latency["p50_ms"]),
            "p95_ms": float(total_latency["p95_ms"]),
            "p99_ms": float(total_latency["p99_ms"]),
            "throughput_windows_per_second": float(
                warm["throughput_from_p50_windows_per_second"]
            ),
            "stage_latency": warm["stages"],
        },
        "memory": memory,
        "spike_activity": spike,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cpu": E2E._cpu_model_name(),  # noqa: SLF001
            "cpu_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(device) if device_type == "cuda" else E2E._cpu_model_name()
            ),
        },
    }
    report["engineering_gates"] = _evaluate_unit_gates(report, spec["engineering_gates"])
    _atomic_json(output_path.expanduser().resolve(), report)
    return report


def _unit_root(output: Path, unit: Mapping[str, Any]) -> Path:
    return output / "units" / str(unit["unit_id"])


def _runtime_argv(argv: Sequence[str], *, final_root: Path, stage_root: Path) -> list[str]:
    final = str(final_root.resolve())
    stage = str(stage_root.resolve())
    return [
        stage + str(token)[len(final) :]
        if str(token) == final or str(token).startswith(final + os.sep)
        else str(token)
        for token in argv
    ]


def _worker_contract(unit: Mapping[str, Any], plan_binding: Mapping[str, Any]) -> dict[str, Any]:
    return _content_document(
        {
            "schema_version": SCHEMA_VERSION,
            "classification": "locked_proposer_deployment_worker_contract",
            "unit_id": unit["unit_id"],
            "outer_fold": int(unit["outer_fold"]),
            "seed": int(unit["seed"]),
            "plan": dict(plan_binding),
            "freeze_spec": unit["freeze_spec"],
            "freeze_spec_content_sha256": unit["freeze_spec_content_sha256"],
            "runtime": unit["runtime"],
            "serving_proposer": unit["serving_proposer"],
            "selection_basis": unit["selection_basis"],
            "target_or_label_artifact_read": False,
            "target_or_label_arrays_deserialized": False,
            "target_or_label_values_used_for_measurement_or_selection": False,
            "best_unit_selection_allowed": False,
            "commercial_performance_claim_authorized": False,
        }
    )


def _run_worker_command(argv: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        completed.stdout + ("\n[stderr]\n" if completed.stderr else "") + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise LockedBenchmarkError(
            f"benchmark worker failed with status {completed.returncode}: {log_path}"
        )


def _validate_benchmark_report(
    report: Mapping[str, Any], *, unit: Mapping[str, Any]
) -> None:
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("classification") != "locked_proposer_deployment_unit_benchmark"
        or report.get("unit_id") != unit["unit_id"]
        or int(report.get("outer_fold", -1)) != int(unit["outer_fold"])
        or int(report.get("seed", -1)) != int(unit["seed"])
        or report.get("runtime") != unit["runtime"]
        or report.get("freeze_spec_content_sha256") != unit["freeze_spec_content_sha256"]
        or report.get("measurement_scope") != "current_host_not_target_device"
        or report.get("commercial_performance_claim_authorized") is not False
        or report.get("target_or_label_artifact_read") is not False
        or report.get("target_or_label_arrays_deserialized") is not False
        or report.get("target_or_label_values_used_for_measurement_or_selection") is not False
        or report.get("serving_role") != "hcs_validation"
    ):
        raise LockedBenchmarkError(f"benchmark report contract mismatch: {unit['unit_id']}")
    model = report.get("model")
    warm = report.get("warm")
    memory = report.get("memory")
    gates = report.get("engineering_gates")
    if not all(isinstance(value, Mapping) for value in (model, warm, memory, gates)):
        raise LockedBenchmarkError("benchmark report metrics are incomplete")
    if report.get("freeze_spec") != unit["freeze_spec"]:
        raise LockedBenchmarkError("benchmark report freeze-spec binding differs")
    for name in ("p50_ms", "p95_ms", "p99_ms", "throughput_windows_per_second"):
        value = float(warm.get(name, math.nan))
        if not math.isfinite(value) or value <= 0:
            raise LockedBenchmarkError(f"benchmark warm metric is invalid: {name}")
    if not (
        float(warm["p50_ms"]) <= float(warm["p95_ms"]) <= float(warm["p99_ms"])
    ):
        raise LockedBenchmarkError("benchmark latency quantiles are not ordered")
    expected_throughput = 1000.0 / float(warm["p50_ms"])
    if not math.isclose(
        float(warm["throughput_windows_per_second"]),
        expected_throughput,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise LockedBenchmarkError("benchmark throughput differs from warm p50")
    cold = report.get("cold")
    if not isinstance(cold, Mapping) or float(
        cold.get("model_load_plus_first_resident_raw_window_inference_ms", math.nan)
    ) <= 0:
        raise LockedBenchmarkError("benchmark cold model-load/inference metric is invalid")
    for name in ("checkpoint", "run_config", "pipeline_config"):
        if model.get(name) != unit["serving_proposer"][name]:
            raise LockedBenchmarkError(f"benchmark model {name} binding differs")
    if int(model.get("checkpoint_bytes", -1)) != int(unit["serving_proposer"]["checkpoint"]["bytes"]):
        raise LockedBenchmarkError("benchmark checkpoint byte count differs from binding")
    total = int(model.get("total_parameters", -1))
    trainable = int(model.get("trainable_parameters", -1))
    if total < 1 or trainable < 0 or trainable > total:
        raise LockedBenchmarkError("benchmark parameter counts are invalid")
    if int(memory.get("cpu_process_peak_rss_bytes", -1)) <= 0 or not memory.get(
        "cpu_peak_rss_caveat"
    ):
        raise LockedBenchmarkError("benchmark CPU peak RSS/caveat is invalid")
    spike = report.get("spike_activity")
    if not isinstance(spike, Mapping) or spike.get("target_or_label_used") is not False:
        raise LockedBenchmarkError("benchmark spike activity is not label-free")
    if not isinstance(gates.get("all_applicable_pass"), bool):
        raise LockedBenchmarkError("benchmark engineering-gate result is missing")
    spec, _ = load_freeze_spec(Path(unit["freeze_spec"]["path"]))
    expected_gates = _evaluate_unit_gates(report, spec["engineering_gates"])
    if dict(gates) != expected_gates:
        raise LockedBenchmarkError("benchmark engineering gates were not computed from frozen spec")


def _validate_receipt(
    unit: Mapping[str, Any], *, output: Path, plan_binding: Mapping[str, Any]
) -> dict[str, Any]:
    root = _unit_root(output, unit)
    receipt = _json(root / "receipt.json", "deployment benchmark receipt")
    _validate_content_hash(receipt, label="deployment benchmark receipt")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("classification")
        != "locked_proposer_deployment_benchmark_receipt"
        or receipt.get("unit_id") != unit["unit_id"]
        or receipt.get("plan") != plan_binding
        or receipt.get("freeze_spec") != unit["freeze_spec"]
        or receipt.get("runtime") != unit["runtime"]
        or receipt.get("target_or_label_artifact_read") is not False
        or receipt.get("target_or_label_arrays_deserialized") is not False
        or receipt.get("target_or_label_values_used_for_measurement_or_selection") is not False
        or receipt.get("best_unit_selection_performed") is not False
        or receipt.get("commercial_performance_claim_authorized") is not False
    ):
        raise LockedBenchmarkError(f"receipt provenance differs: {unit['unit_id']}")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise LockedBenchmarkError("receipt outputs are missing")
    verified = {
        name: _binding(raw, relative_to=root, label=f"receipt output {name}")
        for name, raw in outputs.items()
    }
    expected = {
        "worker_contract": root / "worker_contract.json",
        "benchmark": root / "benchmark.json",
        "worker_log": root / "worker.log",
    }
    if any(Path(verified[name]["path"]) != path.resolve() for name, path in expected.items()):
        raise LockedBenchmarkError("receipt output paths differ")
    contract = _json(expected["worker_contract"], "worker contract")
    if contract != _worker_contract(unit, plan_binding):
        raise LockedBenchmarkError("worker contract differs from the campaign plan")
    report = _json(expected["benchmark"], "unit benchmark")
    _validate_benchmark_report(report, unit=unit)
    if receipt.get("engineering_gates") != report["engineering_gates"]:
        raise LockedBenchmarkError("receipt engineering gates differ from report")
    return receipt


def _execute_unit(
    unit: Mapping[str, Any], *, output: Path, plan_binding: Mapping[str, Any]
) -> dict[str, Any]:
    final_root = _unit_root(output, unit)
    if final_root.exists():
        raise LockedBenchmarkError(f"unit root exists without validated receipt: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{unit['unit_id']}.staging.", dir=final_root.parent))
    try:
        contract = _worker_contract(unit, plan_binding)
        _atomic_json(stage / "worker_contract.json", contract, immutable=True)
        argv = _runtime_argv(unit["worker_argv"], final_root=final_root, stage_root=stage)
        _run_worker_command(argv, cwd=output, log_path=stage / "worker.log")
        benchmark_path = stage / "benchmark.json"
        if not benchmark_path.is_file():
            raise LockedBenchmarkError("worker did not publish benchmark.json")
        report = _json(benchmark_path, "unit benchmark")
        _validate_benchmark_report(report, unit=unit)
        receipt = _content_document(
            {
                "schema_version": SCHEMA_VERSION,
                "classification": "locked_proposer_deployment_benchmark_receipt",
                "unit_id": unit["unit_id"],
                "outer_fold": int(unit["outer_fold"]),
                "seed": int(unit["seed"]),
                "plan": dict(plan_binding),
                "freeze_spec": unit["freeze_spec"],
                "freeze_spec_content_sha256": unit["freeze_spec_content_sha256"],
                "runtime": unit["runtime"],
                "serving_proposer": unit["serving_proposer"],
                "worker_argv": unit["worker_argv"],
                "outputs": {
                    "worker_contract": bind_file(
                        stage / "worker_contract.json",
                        recorded_path=final_root / "worker_contract.json",
                    ),
                    "benchmark": bind_file(
                        benchmark_path, recorded_path=final_root / "benchmark.json"
                    ),
                    "worker_log": bind_file(
                        stage / "worker.log", recorded_path=final_root / "worker.log"
                    ),
                },
                "engineering_gates": report["engineering_gates"],
                "measurement_scope": "current_host_not_target_device",
                "target_or_label_artifact_read": False,
                "target_or_label_arrays_deserialized": False,
                "target_or_label_values_used_for_measurement_or_selection": False,
                "best_unit_selection_performed": False,
                "commercial_performance_claim_authorized": False,
            }
        )
        _atomic_json(stage / "receipt.json", receipt, immutable=True)
        for path in stage.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        os.replace(stage, final_root)
        return _validate_receipt(unit, output=output, plan_binding=plan_binding)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _progress(
    plan_binding: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]], *, sealed: bool
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_proposer_deployment_benchmark_progress",
        "plan": dict(plan_binding),
        "completed_units": len(receipts),
        "expected_units": EXPECTED_UNITS,
        "complete_seal_present": sealed,
        "all_18_reported": len(receipts) == EXPECTED_UNITS,
        "best_unit_selection_performed": False,
        "target_or_label_artifact_read": False,
        "target_or_label_arrays_deserialized": False,
        "target_or_label_values_used_for_measurement_or_selection": False,
        "commercial_performance_claim_authorized": False,
        "units": [
            {
                "unit_id": receipt["unit_id"],
                "receipt_content_sha256": receipt["content_sha256"],
                "engineering_gates_pass": receipt["engineering_gates"]["all_applicable_pass"],
            }
            for receipt in receipts
        ],
    }


def _complete_seal(
    *, plan: Mapping[str, Any], plan_binding: Mapping[str, Any], output: Path, receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(receipts) != EXPECTED_UNITS:
        raise LockedBenchmarkError("complete benchmark seal requires 18 receipts")
    units = []
    for unit, receipt in zip(plan["units"], receipts, strict=True):
        root = _unit_root(output, unit)
        units.append(
            {
                "unit_id": unit["unit_id"],
                "outer_fold": int(unit["outer_fold"]),
                "seed": int(unit["seed"]),
                "receipt": bind_file(root / "receipt.json"),
                "benchmark": bind_file(root / "benchmark.json"),
                "engineering_gates_pass": receipt["engineering_gates"]["all_applicable_pass"],
            }
        )
    return _content_document(
        {
            "schema_version": SCHEMA_VERSION,
            "classification": "locked_proposer_deployment_all_18_complete_seal",
            "plan": dict(plan_binding),
            "freeze_spec": plan["freeze_spec"],
            "freeze_spec_content_sha256": plan["freeze_spec_content_sha256"],
            "pretest_index": plan["pretest_index"],
            "runtime": plan["runtime"],
            "measurement_sources": plan["measurement_sources"],
            "unit_count": EXPECTED_UNITS,
            "all_18_reported": True,
            "all_applicable_engineering_gates_pass": all(
                bool(receipt["engineering_gates"]["all_applicable_pass"])
                for receipt in receipts
            ),
            "best_unit_selection_performed": False,
            "unit_ranking_performed": False,
            "measurement_scope": "current_host_not_target_device",
            "target_or_label_artifact_read": False,
            "target_or_label_arrays_deserialized": False,
            "target_or_label_values_used_for_measurement_or_selection": False,
            "commercial_performance_claim_authorized": False,
            "units": units,
        }
    )


def run_campaign(
    *,
    freeze_spec_path: Path,
    pretest_index: Path,
    output_root: Path,
    device_name: str,
    max_new_units: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if max_new_units is not None and int(max_new_units) < 0:
        raise LockedBenchmarkError("max_new_units cannot be negative")
    output = output_root.expanduser().resolve()
    control = output / "control"
    control.mkdir(parents=True, exist_ok=True)
    with (control / "campaign.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockedBenchmarkError("another deployment benchmark holds the lock") from exc
        plan = build_plan(
            freeze_spec_path=freeze_spec_path,
            pretest_index=pretest_index,
            output_root=output,
            device_name=device_name,
        )
        plan_path = control / "plan.json"
        _ensure_document(plan_path, plan)
        plan_binding = bind_file(plan_path)
        preexecution = _content_document(
            {
                "schema_version": SCHEMA_VERSION,
                "classification": "locked_proposer_deployment_benchmark_preexecution_seal",
                "plan": plan_binding,
                "freeze_spec": plan["freeze_spec"],
                "pretest_index": plan["pretest_index"],
                "measurement_sources": plan["measurement_sources"],
                "runtime": plan["runtime"],
                "target_or_label_artifact_read": False,
                "target_or_label_arrays_deserialized": False,
                "target_or_label_values_used_for_measurement_or_selection": False,
                "best_unit_selection_allowed": False,
                "commercial_performance_claim_authorized": False,
            }
        )
        _ensure_document(control / "preexecution_seal.json", preexecution)
        if not dry_run and plan["runtime"]["device_type"] == "cuda" and not torch.cuda.is_available():
            raise LockedBenchmarkError("CUDA requested but unavailable on the current host")
        for unit in plan["units"]:
            argv = unit["worker_argv"]
            if (
                not isinstance(argv, list)
                or len(argv) != 7
                or argv[2] != "worker"
                or argv[3] != "--contract"
                or argv[5] != "--output"
            ):
                raise LockedBenchmarkError(f"worker command contract differs: {unit['unit_id']}")
        existing: dict[str, dict[str, Any]] = {}
        for unit in plan["units"]:
            root = _unit_root(output, unit)
            if (root / "receipt.json").is_file():
                existing[unit["unit_id"]] = _validate_receipt(
                    unit, output=output, plan_binding=plan_binding
                )
            elif root.exists():
                raise LockedBenchmarkError(f"unit root exists without receipt: {root}")
        seal_path = output / "complete_seal.json"
        if seal_path.exists() and len(existing) != EXPECTED_UNITS:
            raise LockedBenchmarkError("complete seal exists for an incomplete benchmark")
        limit = 0 if dry_run else (
            EXPECTED_UNITS if max_new_units is None else int(max_new_units)
        )
        receipts: list[dict[str, Any]] = []
        new_units = 0
        for unit in plan["units"]:
            if unit["unit_id"] in existing:
                receipts.append(existing[unit["unit_id"]])
            elif new_units < limit:
                receipts.append(_execute_unit(unit, output=output, plan_binding=plan_binding))
                new_units += 1
            _atomic_json(control / "progress.json", _progress(plan_binding, receipts, sealed=False))
        if len(receipts) == EXPECTED_UNITS:
            seal = _complete_seal(
                plan=plan, plan_binding=plan_binding, output=output, receipts=receipts
            )
            _ensure_document(seal_path, seal)
            sealed = True
            status = "all_18_proposer_deployment_benchmarks_sealed"
        else:
            sealed = False
            status = "dry_run" if dry_run else "deployment_benchmark_incomplete"
        _atomic_json(control / "progress.json", _progress(plan_binding, receipts, sealed=sealed))
        result: dict[str, Any] = {
            "status": status,
            "completed_units": len(receipts),
            "new_units": new_units,
            "expected_units": EXPECTED_UNITS,
            "all_18_reported": sealed,
            "best_unit_selection_performed": False,
            "measurement_scope": "current_host_not_target_device",
            "target_or_label_artifact_read": False,
            "target_or_label_arrays_deserialized": False,
            "target_or_label_values_used_for_measurement_or_selection": False,
            "commercial_performance_claim_authorized": False,
            "plan": plan_binding,
        }
        if sealed:
            result["complete_seal"] = bind_file(seal_path)
        _atomic_json(control / "status.json", result)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    freeze = subparsers.add_parser("freeze-spec")
    freeze.add_argument("--output", type=Path, default=DEFAULT_FREEZE_SPEC)
    run = subparsers.add_parser("run")
    run.add_argument("--freeze-spec", type=Path, default=DEFAULT_FREEZE_SPEC)
    run.add_argument("--pretest-index", type=Path, default=DEFAULT_PRETEST_INDEX)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "cpu")
    run.add_argument("--device", default="cpu")
    run.add_argument("--max-new-units", type=int)
    run.add_argument("--dry-run", action="store_true")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--contract", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "freeze-spec":
        result = freeze_spec(args.output)
    elif args.mode == "worker":
        result = run_worker(args.contract, args.output)
    else:
        result = run_campaign(
            freeze_spec_path=args.freeze_spec,
            pretest_index=args.pretest_index,
            output_root=args.output_root,
            device_name=args.device,
            max_new_units=args.max_new_units,
            dry_run=args.dry_run,
        )
    print(json.dumps(_strict_json_value(result), sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
