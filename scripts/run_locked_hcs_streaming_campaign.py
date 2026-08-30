#!/usr/bin/env python3
"""Run all 18 fixed-i3 HCS deployment verifications without labels.

The campaign reuses :mod:`verify_harmonic_set_deployment` as the executable
unit verifier.  Before execution it freezes the verifier, Python runtime,
fixed parity/robustness/latency protocol, and the complete unopened 6 x 3
pre-test index.  Units execute serially and publish by atomic directory rename.

No target, reference-value, or test-prediction path is accepted.  Some sealed
training/cache artifacts can contain retrospective reference bytes; those
files are byte-hashed only for provenance and their arrays are never
deserialized here.  A campaign pass can be reported only after all 18 receipts
validate.  It remains retrospective engineering evidence, never a commercial
performance claim; an independent prospective cohort is still required.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import verify_harmonic_set_deployment as VERIFIER  # noqa: E402


torch = VERIFIER.torch
SCHEMA_VERSION = 1
FOLDS = tuple(range(6))
SEEDS = (20260828, 20260829, 20260830)
EXPECTED_UNITS = 18

DEFAULT_PRETEST_INDEX = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_fixed_i3_pretest/pretest_index.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_streaming_deployment"
)
DEFAULT_FREEZE_SPEC = DEFAULT_OUTPUT_ROOT / "freeze_spec.json"

FORBIDDEN_OUTPUT_NAMES = frozenset(
    {
        "test_predictions.npz",
        "test_metrics.json",
        "test_evaluation_manifest.json",
        "locked_targets.npz",
        "targets.npz",
    }
)
REQUIRED_INDEX_ARTIFACTS = frozenset(
    {
        "selection_lock",
        "checkpoint",
        "scaler",
        "cache_manifest",
        "fallback_oof",
        "fallback_provenance",
        "run_manifest",
        "original_policy",
        "history",
        "validation_metrics",
        "validation_predictions",
        "strict_stack",
    }
)


class StreamingCampaignError(RuntimeError):
    """Fail-closed campaign topology, provenance, or receipt error."""


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


def _strict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _payload(value: Any) -> str:
    return json.dumps(
        _strict(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StreamingCampaignError(f"invalid {label}: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise StreamingCampaignError(f"{label} must be a JSON object: {path}")
    return value


def _content_document(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = canonical_sha256(result)
    return result


def _validate_content(value: Mapping[str, Any], *, label: str) -> None:
    content = dict(value)
    recorded = str(content.pop("content_sha256", ""))
    if len(recorded) != 64 or canonical_sha256(content) != recorded:
        raise StreamingCampaignError(f"{label} content hash mismatch")


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
        raise StreamingCampaignError(f"required file is absent: {resolved}")
    return {
        "path": str((recorded_path or resolved).expanduser().resolve()),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def bind_python_launcher(path: Path) -> dict[str, Any]:
    """Hash the interpreter binary while retaining the venv launcher path."""

    launcher = Path(os.path.abspath(path.expanduser()))
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise StreamingCampaignError("Python launcher must be an executable file")
    binding = bind_file(launcher)
    return {**binding, "path": str(launcher)}


def _resolve(raw: Any, *, relative_to: Path) -> Path:
    if not isinstance(raw, (str, os.PathLike)) or not str(raw):
        raise StreamingCampaignError("artifact path is missing")
    path = Path(raw).expanduser()
    return (relative_to / path).resolve() if not path.is_absolute() else path.resolve()


def _binding(raw: Any, *, relative_to: Path, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise StreamingCampaignError(f"missing binding: {label}")
    path = _resolve(raw.get("path"), relative_to=relative_to)
    expected = str(raw.get("sha256", ""))
    if len(expected) != 64 or not path.is_file() or sha256_file(path) != expected:
        raise StreamingCampaignError(f"file hash mismatch: {label} ({path})")
    if "bytes" in raw and int(raw["bytes"]) != path.stat().st_size:
        raise StreamingCampaignError(f"file size mismatch: {label} ({path})")
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
            raise StreamingCampaignError(f"immutable control document differs: {path}")
        return
    _atomic_json(path, expected, immutable=True)


def _runtime_sources() -> dict[str, Any]:
    return {
        "orchestrator": bind_file(Path(__file__)),
        "unit_verifier": bind_file(Path(VERIFIER.__file__)),
        "pretest_index_producer": bind_file(
            PROJECT_ROOT / "scripts/run_fixed_i3_pretest_campaign.py"
        ),
        "harmonic_set_model": bind_file(
            PROJECT_ROOT / "src/snn_rr/harmonic_set_models.py"
        ),
        "python_executable": bind_python_launcher(Path(sys.executable)),
    }


def _runtime_identity() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "pandas_version": VERIFIER.pd.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available_at_freeze": bool(torch.cuda.is_available()),
    }


def default_freeze_spec() -> dict[str, Any]:
    """Build the deterministic target-independent verification protocol."""

    return _content_document(
        {
            "schema_version": SCHEMA_VERSION,
            "classification": "locked_hcs_streaming_deployment_freeze_spec",
            "matrix": {
                "folds": list(FOLDS),
                "seeds": list(SEEDS),
                "unit_count": EXPECTED_UNITS,
                "all_units_required": True,
                "unit_selection_or_ranking_allowed": False,
                "integrated_pass_only_after_all_18": True,
            },
            "verification": {
                "maximum_sessions": 2,
                "minimum_sessions_required": 2,
                "maximum_windows_per_session": 0,
                "parity_device": "cpu",
                "chunk_schedules": "7,3,11,2;13,1,5,2,8",
                "random_schedules": 2,
                "random_chunk_max": 23,
                "schedule_seed": 20260828,
                "atol": 2.0e-6,
                "rtol": 2.0e-6,
                "robustness_windows": 8,
                "devices": "auto",
                "warmup_repeats": 5,
                "benchmark_repeats": 50,
                "cold_repeats": 3,
                "benchmark_chunk_windows": 32,
            },
            "engineering_gates": {
                "cpu_stateful_one_window_warm_p99_ms_max": 250.0,
                "cuda_stateful_one_window_warm_p99_ms_max": 50.0,
                "stride_budget_ms": 4000.0,
                "p99_stride_budget_fraction_max": 0.10,
                "checkpoint_bytes_max": 50 * 1024 * 1024,
                "parameter_count_max": 5_000_000,
                "cpu_process_peak_rss_bytes_max": 2 * 1024**3,
                "cuda_peak_reserved_bytes_max": 1024**3,
                "spike_rate_diagnostic_min": 0.01,
                "spike_rate_diagnostic_max": 0.20,
                "spike_rate_unavailable_policy": "reported_not_applicable_without_failure",
            },
            "mandatory_gates": [
                "locked_artifact_hashes",
                "whole_chunk_one_window_parity",
                "explicit_session_reset",
                "finite_outputs",
                "seven_nonempty_structural_radar_masks",
                "no_candidate_structural_fallback",
                "corrupt_input_fail_closed",
            ],
            "runtime_identity": _runtime_identity(),
            "runtime_sources": _runtime_sources(),
            "hardware_caveats": {
                "latency": (
                    "Measurements describe the current host and resident cache only; "
                    "they are not target-device guarantees."
                ),
                "cpu_memory": "Process-wide non-isolated ru_maxrss high-water mark.",
                "cuda_memory": "PyTorch allocator peak reserved bytes for this process.",
                "cold_latency": (
                    "Includes checkpoint load/model construction/first inference but "
                    "does not flush the operating-system page cache."
                ),
            },
            "target_or_reference_value_artifact_consulted": False,
            "sealed_artifacts_byte_hashed_for_provenance": True,
            "target_or_reference_arrays_deserialized": False,
            "test_prediction_artifact_opened": False,
            "commercial_performance_claim_authorized": False,
            "prospective_cohort_required_for_commercial_claim": True,
        }
    )


def freeze_spec(path: Path) -> dict[str, Any]:
    expected = default_freeze_spec()
    resolved = path.expanduser().resolve()
    _ensure_document(resolved, expected)
    return {
        "status": "streaming_deployment_spec_frozen",
        "freeze_spec": bind_file(resolved),
        "content_sha256": expected["content_sha256"],
        "target_or_reference_arrays_deserialized": False,
        "commercial_performance_claim_authorized": False,
    }


def load_freeze_spec(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    document = _json(resolved, "streaming deployment freeze spec")
    _validate_content(document, label="streaming deployment freeze spec")
    if document != default_freeze_spec():
        raise StreamingCampaignError("freeze spec differs from the current locked protocol")
    for name, raw in document["runtime_sources"].items():
        _binding(raw, relative_to=resolved.parent, label=f"frozen runtime source {name}")
    if document["runtime_identity"] != _runtime_identity():
        raise StreamingCampaignError("current Python/runtime differs from the frozen spec")
    if (
        document.get("target_or_reference_value_artifact_consulted") is not False
        or document.get("target_or_reference_arrays_deserialized") is not False
        or document.get("test_prediction_artifact_opened") is not False
        or document.get("commercial_performance_claim_authorized") is not False
    ):
        raise StreamingCampaignError("freeze spec permits forbidden label/commercial behavior")
    return document, bind_file(resolved)


def _tree_document(root: Path) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    rows = [
        [str(path.relative_to(resolved)), sha256_file(path), path.stat().st_size]
        for path in files
    ]
    return {
        "path": str(resolved),
        "files": {
            name: {"sha256": digest, "bytes": size} for name, digest, size in rows
        },
        "tree_sha256": canonical_sha256(rows),
    }


def _validate_common_pretest_lock(
    index: Mapping[str, Any], *, index_path: Path
) -> dict[str, str]:
    common = index.get("common")
    if not isinstance(common, Mapping):
        raise StreamingCampaignError("pretest common lock bindings are missing")
    bindings = {
        name: _binding(
            common.get(name),
            relative_to=index_path.parent,
            label=f"pretest common {name}",
        )
        for name in (
            "selection_lock",
            "capacity_selection",
            "policy",
            "source_freeze_manifest",
        )
    }
    freeze = _json(
        Path(bindings["source_freeze_manifest"]["path"]), "pre-i3 source freeze"
    )
    if (
        freeze.get("outer_test_opened") is not False
        or freeze.get("declared_before_any_i3_score") is not True
    ):
        raise StreamingCampaignError("source freeze is not pre-i3 unopened evidence")
    files = freeze.get("files")
    if not isinstance(files, Mapping) or not files:
        raise StreamingCampaignError("source freeze file hashes are missing")
    freeze_root = Path(bindings["source_freeze_manifest"]["path"]).parent
    frozen_hashes: dict[str, str] = {}
    for name, expected_raw in files.items():
        expected = str(expected_raw)
        path = freeze_root / str(name)
        if len(expected) != 64 or not path.is_file() or sha256_file(path) != expected:
            raise StreamingCampaignError(f"pre-i3 frozen source hash mismatch: {name}")
        frozen_hashes[str(name)] = expected
    selection = _json(Path(bindings["selection_lock"]["path"]), "common selection lock")
    capacity = _json(Path(bindings["capacity_selection"]["path"]), "capacity selection")
    policy = _json(Path(bindings["policy"]["path"]), "common fallback policy")
    if (
        selection.get("schema_version") != 1
        or selection.get("classification") != "retrospective_i3_common_discovery_lock"
        or selection.get("outer_test_opened_before_lock") is not False
        or selection.get("selected_preset") != "default"
        or int(selection.get("selected_parameter_count", -1)) != 195_603
        or selection.get("capacity_selection_sha256")
        != bindings["capacity_selection"]["sha256"]
        or selection.get("common_fallback_policy_sha256") != bindings["policy"]["sha256"]
        or selection.get("source_freeze") != frozen_hashes
    ):
        raise StreamingCampaignError("common i3 selection lock differs from frozen default")
    if (
        capacity.get("outer_test_opened") is not False
        or capacity.get("selected_preset") != "default"
        or int(capacity.get("selected_parameter_count", -1)) != 195_603
        or capacity.get("source_freeze_manifest_sha256")
        != bindings["source_freeze_manifest"]["sha256"]
    ):
        raise StreamingCampaignError("capacity selection differs from frozen default")
    if (
        policy.get("outer_test_opened") is not False
        or policy.get("selected_preset") != "default"
        or not isinstance(policy.get("policy"), Mapping)
        or policy.get("policy", {}).get("selection_status")
        != selection.get("policy_selection_status")
    ):
        raise StreamingCampaignError("common fallback policy differs from common lock")
    return frozen_hashes


def validate_pretest_matrix(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate the complete index and every HCS unit/output byte binding.

    Only JSON control documents are parsed.  Checkpoints, NPZ predictions,
    strict stacks, fallback CSVs, and metric outputs are byte-hashed without
    deserializing their payloads.
    """

    resolved = path.expanduser().resolve()
    index = _json(resolved, "fixed-i3 pretest index")
    _validate_content(index, label="fixed-i3 pretest index")
    matrix = index.get("matrix")
    if (
        index.get("schema_version") != 1
        or index.get("classification") != "retrospective_fixed_i3_pretest_index"
        or index.get("status") != "complete"
        or int(index.get("completed_units", -1)) != EXPECTED_UNITS
        or index.get("selected_preset") != "default"
        or int(index.get("selected_parameter_count", -1)) != 195_603
        or index.get("capacity_reselected") is not False
        or index.get("common_policy_reselected") is not False
        or index.get("validation_scores_control_execution") is not False
        or index.get("outer_test_opened") is not False
        or int(index.get("outer_test_artifact_count", -1)) != 0
        or index.get("commercial_claim_authorized") is not False
        or not isinstance(matrix, Mapping)
        or matrix.get("folds") != list(FOLDS)
        or matrix.get("seeds") != list(SEEDS)
        or int(matrix.get("unit_count", -1)) != EXPECTED_UNITS
    ):
        raise StreamingCampaignError("pretest index is not the final unopened fixed matrix")
    frozen_hashes = _validate_common_pretest_lock(index, index_path=resolved)
    index_binding = bind_file(resolved)
    raw_units = index.get("units")
    if not isinstance(raw_units, list) or len(raw_units) != EXPECTED_UNITS:
        raise StreamingCampaignError("pretest index must contain exactly 18 units")
    raw_map: dict[tuple[int, int], Mapping[str, Any]] = {}
    for unit in raw_units:
        if not isinstance(unit, Mapping) or unit.get("status") != "complete":
            raise StreamingCampaignError("pretest index contains an incomplete unit")
        key = (int(unit.get("outer_fold", -1)), int(unit.get("seed", -1)))
        if key in raw_map:
            raise StreamingCampaignError(f"duplicate pretest unit: {key}")
        raw_map[key] = unit
    expected_topology = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    if set(raw_map) != expected_topology:
        raise StreamingCampaignError("pretest unit topology is not fixed 6 x 3")
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
            raw = raw_map[(fold, seed)]
            artifacts_raw = raw.get("artifacts")
            if not isinstance(artifacts_raw, Mapping) or not REQUIRED_INDEX_ARTIFACTS <= set(
                map(str, artifacts_raw)
            ):
                raise StreamingCampaignError(
                    f"indexed HCS artifacts are incomplete: {fold}/{seed}"
                )
            artifacts = {
                str(name): _binding(
                    binding,
                    relative_to=resolved.parent,
                    label=f"fixed-i3 {fold}/{seed}/{name}",
                )
                for name, binding in artifacts_raw.items()
            }
            unit_root = _resolve(raw.get("output_root"), relative_to=resolved.parent)
            if any(
                path.name.lower() in FORBIDDEN_OUTPUT_NAMES
                or path.name.lower().startswith("test_pred_")
                or "test_prediction" in path.name.lower()
                for path in unit_root.rglob("*")
            ):
                raise StreamingCampaignError(
                    f"outer-test artifact entered unit tree: {fold}/{seed}"
                )
            observed_tree = _tree_document(unit_root)
            if raw.get("output_tree") != observed_tree:
                raise StreamingCampaignError(
                    f"fixed-i3 output-tree hash mismatch: {fold}/{seed}"
                )
            for name in ("selection_lock", "checkpoint", "scaler", "run_manifest"):
                if Path(artifacts[name]["path"]).parent != unit_root:
                    raise StreamingCampaignError(
                        f"unit artifact escaped output root: {fold}/{seed}/{name}"
                    )
            cache_raw = raw.get("cache_root")
            if not isinstance(cache_raw, Mapping):
                raise StreamingCampaignError(
                    f"cache root binding is missing: {fold}/{seed}"
                )
            cache_root = _resolve(cache_raw.get("path"), relative_to=resolved.parent)
            if Path(artifacts["cache_manifest"]["path"]) != cache_root / "manifest.json":
                raise StreamingCampaignError(
                    f"cache manifest/root mismatch: {fold}/{seed}"
                )
            if str(cache_raw.get("manifest_sha256", "")) != artifacts[
                "cache_manifest"
            ]["sha256"]:
                raise StreamingCampaignError(f"cache root hash mismatch: {fold}/{seed}")
            lock = _json(
                Path(artifacts["selection_lock"]["path"]), "i3 unit selection lock"
            )
            if (
                lock.get("schema_version") != 1
                or int(lock.get("outer_fold", -1)) != fold
                or int(lock.get("seed", -1)) != seed
                or int(lock.get("adaptive_iteration", -1)) != 3
                or lock.get("outer_test_not_opened_before_this_lock") is not True
            ):
                raise StreamingCampaignError(
                    f"i3 unit lock identity/leakage mismatch: {fold}/{seed}"
                )
            for name, field in lock_hash_fields.items():
                if str(lock.get(field, "")) != artifacts[name]["sha256"]:
                    raise StreamingCampaignError(
                        f"i3 unit lock hash mismatch: {fold}/{seed}/{name}"
                    )
            source_bindings = lock.get("source_bindings")
            if not isinstance(source_bindings, Mapping):
                raise StreamingCampaignError(
                    f"i3 unit source bindings are missing: {fold}/{seed}"
                )
            for source_name, freeze_name in source_names.items():
                binding = source_bindings.get(source_name)
                if not isinstance(binding, Mapping) or str(
                    binding.get("sha256", "")
                ) != str(frozen_hashes.get(freeze_name, "")):
                    raise StreamingCampaignError(
                        f"i3 frozen source mismatch: {fold}/{seed}/{source_name}"
                    )
            run_manifest = _json(
                Path(artifacts["run_manifest"]["path"]), "i3 run manifest"
            )
            optimization = run_manifest.get("optimization")
            if (
                int(run_manifest.get("outer_fold", -1)) != fold
                or int(run_manifest.get("validation_fold", -1))
                != (fold + 1) % len(FOLDS)
                or not isinstance(optimization, Mapping)
                or int(optimization.get("seed", -1)) != seed
                or run_manifest.get("retrospective_only") is not True
                or run_manifest.get("commercial_claim_authorized") is not False
            ):
                raise StreamingCampaignError(
                    f"i3 run-manifest identity differs: {fold}/{seed}"
                )
            records.append(
                {
                    "unit_id": f"outer_{fold}_seed_{seed}",
                    "outer_fold": fold,
                    "seed": seed,
                    "run_dir": str(unit_root),
                    "cache_root": str(cache_root),
                    "checkpoint": artifacts["checkpoint"],
                    "selection_lock": artifacts["selection_lock"],
                    "artifacts": artifacts,
                    "output_tree": observed_tree,
                    "sealed_artifacts_byte_hashed_for_provenance": True,
                    "target_or_reference_arrays_deserialized": False,
                    "test_prediction_artifact_opened": False,
                }
            )
    expected_order = [(fold, seed) for fold in FOLDS for seed in SEEDS]
    if [(record["outer_fold"], record["seed"]) for record in records] != expected_order:
        raise StreamingCampaignError("pretest records are not in fixed fold/seed order")
    return index, index_binding, records


def _unit_root(output: Path, unit: Mapping[str, Any]) -> Path:
    return output / "units" / str(unit["unit_id"])


def build_plan(
    *, freeze_spec_path: Path, pretest_index: Path, output_root: Path
) -> dict[str, Any]:
    # The spec/runtime is verified before the sealed training index is touched.
    spec, spec_binding = load_freeze_spec(freeze_spec_path)
    _, index_binding, records = validate_pretest_matrix(pretest_index)
    executable = bind_python_launcher(Path(sys.executable))
    verifier_source = bind_file(Path(VERIFIER.__file__))
    if not _same_binding(executable, spec["runtime_sources"]["python_executable"]):
        raise StreamingCampaignError("execution Python differs from freeze spec")
    if not _same_binding(verifier_source, spec["runtime_sources"]["unit_verifier"]):
        raise StreamingCampaignError("unit verifier differs from freeze spec")
    verification = spec["verification"]
    output = output_root.expanduser().resolve()
    units: list[dict[str, Any]] = []
    for record in records:
        final = _unit_root(output, record)
        argv = [
            executable["path"],
            verifier_source["path"],
            "--run-dir",
            record["run_dir"],
            "--cache",
            record["cache_root"],
            "--checkpoint",
            record["checkpoint"]["path"],
            "--output-dir",
            str(final / "verification"),
            "--maximum-sessions",
            str(verification["maximum_sessions"]),
            "--maximum-windows-per-session",
            str(verification["maximum_windows_per_session"]),
            "--parity-device",
            str(verification["parity_device"]),
            "--chunk-schedules",
            str(verification["chunk_schedules"]),
            "--random-schedules",
            str(verification["random_schedules"]),
            "--random-chunk-max",
            str(verification["random_chunk_max"]),
            "--schedule-seed",
            str(verification["schedule_seed"]),
            "--atol",
            str(verification["atol"]),
            "--rtol",
            str(verification["rtol"]),
            "--robustness-windows",
            str(verification["robustness_windows"]),
            "--devices",
            str(verification["devices"]),
            "--warmup-repeats",
            str(verification["warmup_repeats"]),
            "--benchmark-repeats",
            str(verification["benchmark_repeats"]),
            "--cold-repeats",
            str(verification["cold_repeats"]),
            "--benchmark-chunk-windows",
            str(verification["benchmark_chunk_windows"]),
        ]
        units.append(
            {
                **record,
                "freeze_spec": spec_binding,
                "freeze_spec_content_sha256": spec["content_sha256"],
                "verifier_argv": argv,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_streaming_deployment_plan",
        "freeze_spec": spec_binding,
        "freeze_spec_content_sha256": spec["content_sha256"],
        "pretest_index": index_binding,
        "matrix": spec["matrix"],
        "verification": verification,
        "engineering_gates": spec["engineering_gates"],
        "runtime_identity": spec["runtime_identity"],
        "runtime_sources": spec["runtime_sources"],
        "execution_order": [unit["unit_id"] for unit in units],
        "unit_count": EXPECTED_UNITS,
        "serial_execution": True,
        "resume_requires_full_receipt_revalidation": True,
        "integrated_pass_only_after_all_18": True,
        "unit_selection_or_ranking_allowed": False,
        "target_or_reference_value_artifact_consulted": False,
        "sealed_artifacts_byte_hashed_for_provenance": True,
        "target_or_reference_arrays_deserialized": False,
        "test_prediction_artifact_opened": False,
        "commercial_performance_claim_authorized": False,
        "prospective_cohort_required_for_commercial_claim": True,
        "units": units,
    }


def _run_verifier(argv: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    completed = subprocess.run(
        list(map(str, argv)),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log_path.write_text(
        completed.stdout + ("\n[stderr]\n" if completed.stderr else "") + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise StreamingCampaignError(
            f"deployment verifier failed with status {completed.returncode}: {log_path}"
        )


def _runtime_argv(
    argv: Sequence[str], *, final_root: Path, stage_root: Path
) -> list[str]:
    result = list(map(str, argv))
    final = str(final_root / "verification")
    positions = [index for index, value in enumerate(result) if value == final]
    if len(positions) != 1:
        raise StreamingCampaignError("verifier output path is not uniquely bound")
    result[positions[0]] = str(stage_root / "verification")
    return result


def _model_diagnostics(
    unit: Mapping[str, Any], report: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute target-free parameter and spike diagnostics from locked inputs."""

    checkpoint = torch.load(
        Path(unit["checkpoint"]["path"]), map_location="cpu", weights_only=False
    )
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise StreamingCampaignError("checkpoint model_state is missing")
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping):
        raise StreamingCampaignError("verifier report bindings are missing")
    anchor = bindings.get("anchor_input")
    stream = VERIFIER.load_deployment_stream(
        Path(unit["cache_root"]),
        Path(unit["run_dir"]) / "scaler.json",
        maximum_sessions=2,
        maximum_windows_per_session=0,
        anchor_input_path=(
            Path(str(anchor["path"])) if isinstance(anchor, Mapping) else None
        ),
        anchor_forward_enabled=bool(bindings.get("anchor_enabled")),
    )
    model = VERIFIER.load_model(
        Path(unit["checkpoint"]["path"]), bindings["model_config"], torch.device("cpu")
    )
    parameter_count = int(sum(value.numel() for value in model.parameters()))
    diagnostic_windows = min(
        stream.windows, int(spec["verification"]["robustness_windows"])
    )
    diagnostic_batch = stream.forward_batch().time_slice(0, diagnostic_windows)
    output = VERIFIER.run_chunk_schedule(model, diagnostic_batch, (1,))
    spike = output.get("spike_sequence")
    spike_rate: float | None = None
    if isinstance(spike, torch.Tensor) and spike.numel() and bool(torch.isfinite(spike).all()):
        spike_rate = float(spike.float().mean().item())
    del model
    return {
        "parameter_count": parameter_count,
        "spike_rate": spike_rate,
        "spike_rate_source": "label_free_spike_sequence_mean" if spike_rate is not None else None,
        "spike_diagnostic_windows": diagnostic_windows,
        "target_or_reference_value_used": False,
    }


def evaluate_gates(
    report: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate fixed mandatory and current-host engineering gates."""

    robustness = report.get("robustness", {})
    parity = report.get("parity", {})
    reset = report.get("session_reset", {})
    masks = robustness.get("seven_nonempty_radar_masks", [])
    mandatory = {
        "locked_artifact_hashes": bool(report.get("bindings")),
        "whole_chunk_one_window_parity": bool(parity.get("passed"))
        and any(case.get("mode") == "one_window_streaming" for case in parity.get("cases", [])),
        "explicit_session_reset": bool(reset.get("passed"))
        and int(reset.get("session_count", 0)) >= int(spec["verification"]["minimum_sessions_required"]),
        "finite_outputs": bool(robustness.get("zero_features_finite"))
        and bool(robustness.get("corrupt_nan_inf_inputs_finite_and_unavailable"))
        and bool(robustness.get("all_radars_missing_finite_and_unavailable")),
        "seven_nonempty_structural_radar_masks": bool(robustness.get("passed"))
        and len(masks) == 7
        and all(
            bool(row.get("finite"))
            and bool(row.get("source_available"))
            and bool(row.get("missing_feature_corruption_invariant"))
            for row in masks
        ),
        "no_candidate_structural_fallback": bool(
            robustness.get("no_candidate_structural_fallback_route")
        ),
        "corrupt_input_fail_closed": bool(
            robustness.get("corrupt_nan_inf_inputs_finite_and_unavailable")
        )
        and bool(robustness.get("wrong_feature_shape_rejected")),
    }
    gates = spec["engineering_gates"]
    benchmarks = report.get("benchmarks", [])
    by_device = {
        str(row.get("device")).split(":")[0]: row
        for row in benchmarks
        if row.get("operation") == "stateful_one_window_warm"
    }
    latency: dict[str, Any] = {}
    for device in ("cpu", "cuda"):
        row = by_device.get(device)
        available = row is not None
        limit = float(gates[f"{device}_stateful_one_window_warm_p99_ms_max"])
        p99 = None if row is None else float(row["p99_ms"])
        latency[device] = {
            "applicable": available,
            "value_ms": p99,
            "maximum_ms": limit,
            "pass": True if p99 is None else math.isfinite(p99) and p99 <= limit,
            "stride_fraction": None if p99 is None else p99 / float(gates["stride_budget_ms"]),
            "stride_fraction_maximum": float(gates["p99_stride_budget_fraction_max"]),
            "stride_pass": (
                True
                if p99 is None
                else p99 / float(gates["stride_budget_ms"])
                <= float(gates["p99_stride_budget_fraction_max"])
            ),
            "current_host_only": True,
        }
    checkpoint_bytes = int(report["bindings"]["locked_files"]["checkpoint"]["bytes"])
    parameter_count = int(diagnostics["parameter_count"])
    cpu_rows = [row for row in benchmarks if str(row.get("device")) == "cpu"]
    cuda_rows = [row for row in benchmarks if str(row.get("device")).startswith("cuda")]
    cpu_peak = max((int(row["peak_memory_bytes"]) for row in cpu_rows), default=0)
    cuda_peak = max((int(row.get("cuda_peak_reserved_bytes", 0)) for row in cuda_rows), default=0)
    spike_value = diagnostics.get("spike_rate")
    spike_pass = (
        True
        if spike_value is None
        else float(gates["spike_rate_diagnostic_min"])
        <= float(spike_value)
        <= float(gates["spike_rate_diagnostic_max"])
    )
    engineering = {
        "latency": latency,
        "checkpoint_bytes": {
            "value": checkpoint_bytes,
            "maximum": int(gates["checkpoint_bytes_max"]),
            "pass": checkpoint_bytes <= int(gates["checkpoint_bytes_max"]),
        },
        "parameter_count": {
            "value": parameter_count,
            "maximum": int(gates["parameter_count_max"]),
            "pass": parameter_count <= int(gates["parameter_count_max"]),
        },
        "cpu_process_peak_rss_bytes": {
            "applicable": bool(cpu_rows),
            "value": cpu_peak if cpu_rows else None,
            "maximum": int(gates["cpu_process_peak_rss_bytes_max"]),
            "pass": (not cpu_rows) or cpu_peak <= int(gates["cpu_process_peak_rss_bytes_max"]),
            "nonisolated_process_high_water_mark": True,
        },
        "cuda_peak_reserved_bytes": {
            "applicable": bool(cuda_rows),
            "value": cuda_peak if cuda_rows else None,
            "maximum": int(gates["cuda_peak_reserved_bytes_max"]),
            "pass": (not cuda_rows) or cuda_peak <= int(gates["cuda_peak_reserved_bytes_max"]),
        },
        "spike_rate_diagnostic": {
            "applicable": spike_value is not None,
            "value": spike_value,
            "minimum": float(gates["spike_rate_diagnostic_min"]),
            "maximum": float(gates["spike_rate_diagnostic_max"]),
            "pass": spike_pass,
            "unavailable_policy": gates["spike_rate_unavailable_policy"],
        },
    }
    applicable_passes = [
        latency[device]["pass"] and latency[device]["stride_pass"]
        for device in ("cpu", "cuda")
        if latency[device]["applicable"]
    ] + [
        engineering["checkpoint_bytes"]["pass"],
        engineering["parameter_count"]["pass"],
        engineering["cpu_process_peak_rss_bytes"]["pass"],
        engineering["cuda_peak_reserved_bytes"]["pass"],
        engineering["spike_rate_diagnostic"]["pass"],
    ]
    return {
        "mandatory": mandatory,
        "all_mandatory_pass": all(mandatory.values()),
        "engineering": engineering,
        "all_applicable_engineering_pass": all(applicable_passes),
        "unit_integrated_pass": all(mandatory.values()) and all(applicable_passes),
        "latency_and_memory_are_current_host_only": True,
    }


def _validate_verifier_report(
    report: Mapping[str, Any], *, unit: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    if (
        report.get("schema_version") != VERIFIER.SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("retrospective_only") is not True
        or report.get("commercial_claim_authorized") is not False
        or report.get("prospective_cohort_required_for_commercial_claim") is not True
        or report.get("parity_device") != spec["verification"]["parity_device"]
    ):
        raise StreamingCampaignError(f"verifier report contract mismatch: {unit['unit_id']}")
    access = report.get("label_access")
    if not isinstance(access, Mapping) or (
        access.get("target_or_test_labels_accessed") is not False
        or access.get("test_prediction_artifacts_opened") is not False
        or access.get("forward_allowlist_only") is not True
    ):
        raise StreamingCampaignError("verifier report permits target/test access")
    runtime = report.get("runtime")
    frozen_runtime = spec["runtime_identity"]
    if not isinstance(runtime, Mapping) or (
        runtime.get("python") != frozen_runtime["python_version"]
        or runtime.get("numpy") != frozen_runtime["numpy_version"]
        or runtime.get("pandas") != frozen_runtime["pandas_version"]
        or runtime.get("torch") != frozen_runtime["torch_version"]
        or runtime.get("cuda_runtime") != frozen_runtime["cuda_runtime"]
        or bool(runtime.get("cuda_available"))
        != bool(frozen_runtime["cuda_available_at_freeze"])
    ):
        raise StreamingCampaignError("verifier runtime differs from frozen runtime")
    scope = report.get("input_scope")
    if not isinstance(scope, Mapping) or (
        bool(scope.get("truncated_by_user_limit"))
        or len(scope.get("session_ids", []))
        != int(spec["verification"]["maximum_sessions"])
    ):
        raise StreamingCampaignError("verifier did not cover the fixed complete-session scope")
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping):
        raise StreamingCampaignError("verifier report bindings are missing")
    expected_paths = {
        "run_dir": Path(unit["run_dir"]),
        "cache_root": Path(unit["cache_root"]),
        "checkpoint_path": Path(unit["checkpoint"]["path"]),
    }
    for name, expected in expected_paths.items():
        if Path(str(bindings.get(name, ""))).resolve() != expected.resolve():
            raise StreamingCampaignError(f"verifier binding differs: {name}")
    locked = bindings.get("locked_files")
    if not isinstance(locked, Mapping):
        raise StreamingCampaignError("verifier locked-file bindings are missing")
    expected_artifact_names = {
        "checkpoint": "checkpoint",
        "scaler": "scaler",
        "run_manifest": "run_manifest",
        "fallback_policy": "original_policy",
    }
    for report_name, index_name in expected_artifact_names.items():
        if not _same_binding(locked.get(report_name, {}), unit["artifacts"][index_name]):
            raise StreamingCampaignError(f"verifier/index hash differs: {report_name}")
    if bindings.get("selection_lock_sha256") != unit["selection_lock"]["sha256"]:
        raise StreamingCampaignError("verifier selection-lock hash differs")
    cache = bindings.get("cache_manifest")
    if not isinstance(cache, Mapping) or str(cache.get("sha256")) != unit["artifacts"][
        "cache_manifest"
    ]["sha256"]:
        raise StreamingCampaignError("verifier cache-manifest hash differs")
    devices = {str(row.get("device")).split(":")[0] for row in report.get("benchmarks", [])}
    expected_devices = {"cpu"} | ({"cuda"} if torch.cuda.is_available() else set())
    if devices != expected_devices:
        raise StreamingCampaignError(
            f"verifier benchmark device cover differs: {devices} != {expected_devices}"
        )
    verification = spec["verification"]
    expected_repeats = {
        "checkpoint_load_model_init_plus_first_window_cold": int(
            verification["cold_repeats"]
        ),
        "stateful_one_window_warm": int(verification["benchmark_repeats"]),
        "stateless_chunk_warm": int(verification["benchmark_repeats"]),
    }
    observed_keys = set()
    for row in report.get("benchmarks", []):
        operation = str(row.get("operation"))
        if operation not in expected_repeats or int(row.get("repeats", -1)) != expected_repeats[
            operation
        ]:
            raise StreamingCampaignError("verifier benchmark repeat contract differs")
        observed_keys.add((str(row.get("device")).split(":")[0], operation))
    if observed_keys != {
        (device, operation) for device in expected_devices for operation in expected_repeats
    }:
        raise StreamingCampaignError("verifier benchmark operation cover differs")
    parity_cases = report.get("parity", {}).get("cases", [])
    if len(parity_cases) != 5 or not all(
        float(case.get("atol", -1)) == float(verification["atol"])
        and float(case.get("rtol", -1)) == float(verification["rtol"])
        for case in parity_cases
    ):
        raise StreamingCampaignError("verifier parity schedule/tolerance contract differs")


def _verification_outputs(root: Path) -> dict[str, Any]:
    expected = (
        "deployment_verification.json",
        "deployment_verification.csv",
        "deployment_verification.md",
        "artifact_hashes.json",
    )
    outputs = {name: bind_file(root / name) for name in expected}
    hashes = _json(root / "artifact_hashes.json", "verifier artifact hashes")
    artifacts = hashes.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise StreamingCampaignError("verifier artifact hash mapping is absent")
    for name in expected[:3]:
        raw = artifacts.get(name)
        if (
            not isinstance(raw, Mapping)
            or str(raw.get("sha256", "")) != outputs[name]["sha256"]
            or int(raw.get("bytes", -1)) != outputs[name]["bytes"]
        ):
            raise StreamingCampaignError(f"verifier output hash receipt differs: {name}")
    return outputs


def _validate_receipt(
    unit: Mapping[str, Any], *, output: Path, plan_binding: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    root = _unit_root(output, unit)
    receipt = _json(root / "receipt.json", "streaming deployment receipt")
    _validate_content(receipt, label="streaming deployment receipt")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("classification")
        != "locked_hcs_streaming_deployment_unit_receipt"
        or receipt.get("unit_id") != unit["unit_id"]
        or int(receipt.get("outer_fold", -1)) != int(unit["outer_fold"])
        or int(receipt.get("seed", -1)) != int(unit["seed"])
        or receipt.get("plan") != plan_binding
        or receipt.get("freeze_spec") != unit["freeze_spec"]
        or receipt.get("freeze_spec_content_sha256")
        != unit["freeze_spec_content_sha256"]
        or receipt.get("pretest_artifacts") != unit["artifacts"]
        or receipt.get("verifier_argv") != unit["verifier_argv"]
        or receipt.get("target_or_reference_value_artifact_consulted") is not False
        or receipt.get("target_or_reference_arrays_deserialized") is not False
        or receipt.get("test_prediction_artifact_opened") is not False
        or receipt.get("unit_selection_or_ranking_performed") is not False
        or receipt.get("commercial_performance_claim_authorized") is not False
    ):
        raise StreamingCampaignError(f"receipt provenance differs: {unit['unit_id']}")
    outputs = _verification_outputs(root / "verification")
    if receipt.get("outputs") != outputs:
        raise StreamingCampaignError("receipt verifier output bindings differ")
    report = _json(root / "verification/deployment_verification.json", "verifier report")
    _validate_verifier_report(report, unit=unit, spec=spec)
    diagnostics = receipt.get("model_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise StreamingCampaignError("receipt model diagnostics are missing")
    if dict(diagnostics) != _model_diagnostics(unit, report, spec):
        raise StreamingCampaignError("receipt model diagnostics differ from locked execution")
    log_binding = receipt.get("verifier_log")
    if not isinstance(log_binding, Mapping) or not _same_binding(
        _binding(log_binding, relative_to=root, label="verifier log"),
        bind_file(root / "verifier.log"),
    ):
        raise StreamingCampaignError("receipt verifier-log binding differs")
    recomputed = evaluate_gates(report, diagnostics, spec)
    if receipt.get("gates") != recomputed:
        raise StreamingCampaignError("receipt gates differ from frozen calculation")
    return receipt


def _execute_unit(
    unit: Mapping[str, Any], *, output: Path, plan_binding: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    final = _unit_root(output, unit)
    if final.exists():
        raise StreamingCampaignError(f"unit root exists without valid receipt: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{unit['unit_id']}.staging.", dir=final.parent))
    try:
        argv = _runtime_argv(unit["verifier_argv"], final_root=final, stage_root=stage)
        _run_verifier(argv, cwd=PROJECT_ROOT, log_path=stage / "verifier.log")
        verification = stage / "verification"
        outputs = _verification_outputs(verification)
        report = _json(verification / "deployment_verification.json", "verifier report")
        _validate_verifier_report(report, unit=unit, spec=spec)
        diagnostics = _model_diagnostics(unit, report, spec)
        gates = evaluate_gates(report, diagnostics, spec)
        receipt = _content_document(
            {
                "schema_version": SCHEMA_VERSION,
                "classification": "locked_hcs_streaming_deployment_unit_receipt",
                "unit_id": unit["unit_id"],
                "outer_fold": int(unit["outer_fold"]),
                "seed": int(unit["seed"]),
                "plan": dict(plan_binding),
                "freeze_spec": unit["freeze_spec"],
                "freeze_spec_content_sha256": unit["freeze_spec_content_sha256"],
                "pretest_artifacts": unit["artifacts"],
                "verifier_argv": unit["verifier_argv"],
                "outputs": {
                    name: {**binding, "path": str(final / "verification" / name)}
                    for name, binding in outputs.items()
                },
                "verifier_log": bind_file(
                    stage / "verifier.log", recorded_path=final / "verifier.log"
                ),
                "model_diagnostics": diagnostics,
                "gates": gates,
                "hardware_measurement_scope": "current_host_not_target_device",
                "target_or_reference_value_artifact_consulted": False,
                "sealed_artifacts_byte_hashed_for_provenance": True,
                "target_or_reference_arrays_deserialized": False,
                "test_prediction_artifact_opened": False,
                "unit_selection_or_ranking_performed": False,
                "commercial_performance_claim_authorized": False,
                "prospective_cohort_required_for_commercial_claim": True,
            }
        )
        _atomic_json(stage / "receipt.json", receipt, immutable=True)
        for path in stage.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        os.replace(stage, final)
        return _validate_receipt(
            unit, output=output, plan_binding=plan_binding, spec=spec
        )
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _progress(
    plan_binding: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]], *, sealed: bool
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_streaming_deployment_progress",
        "plan": dict(plan_binding),
        "completed_units": len(receipts),
        "expected_units": EXPECTED_UNITS,
        "complete_seal_present": sealed,
        "integrated_pass_reported": bool(
            sealed and all(receipt["gates"]["unit_integrated_pass"] for receipt in receipts)
        ),
        "target_or_reference_arrays_deserialized": False,
        "test_prediction_artifact_opened": False,
        "commercial_performance_claim_authorized": False,
        "units": [
            {
                "unit_id": receipt["unit_id"],
                "receipt_content_sha256": receipt["content_sha256"],
                "unit_integrated_pass": receipt["gates"]["unit_integrated_pass"],
            }
            for receipt in receipts
        ],
    }


def _complete_seal(
    *, plan: Mapping[str, Any], plan_binding: Mapping[str, Any], output: Path,
    receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(receipts) != EXPECTED_UNITS:
        raise StreamingCampaignError("complete streaming seal requires all 18 receipts")
    units = []
    for unit, receipt in zip(plan["units"], receipts, strict=True):
        root = _unit_root(output, unit)
        units.append(
            {
                "unit_id": unit["unit_id"],
                "outer_fold": int(unit["outer_fold"]),
                "seed": int(unit["seed"]),
                "receipt": bind_file(root / "receipt.json"),
                "json": bind_file(root / "verification/deployment_verification.json"),
                "csv": bind_file(root / "verification/deployment_verification.csv"),
                "markdown": bind_file(root / "verification/deployment_verification.md"),
                "unit_integrated_pass": receipt["gates"]["unit_integrated_pass"],
            }
        )
    integrated = all(unit["unit_integrated_pass"] for unit in units)
    return _content_document(
        {
            "schema_version": SCHEMA_VERSION,
            "classification": "locked_hcs_streaming_deployment_all_18_complete_seal",
            "plan": dict(plan_binding),
            "freeze_spec": plan["freeze_spec"],
            "freeze_spec_content_sha256": plan["freeze_spec_content_sha256"],
            "pretest_index": plan["pretest_index"],
            "unit_count": EXPECTED_UNITS,
            "all_18_receipts_validated": True,
            "all_mandatory_gates_pass": all(
                receipt["gates"]["all_mandatory_pass"] for receipt in receipts
            ),
            "all_applicable_engineering_gates_pass": all(
                receipt["gates"]["all_applicable_engineering_pass"] for receipt in receipts
            ),
            "integrated_pass": integrated,
            "integrated_pass_reported_only_after_all_18": True,
            "unit_selection_or_ranking_performed": False,
            "hardware_measurement_scope": "current_host_not_target_device",
            "target_or_reference_value_artifact_consulted": False,
            "sealed_artifacts_byte_hashed_for_provenance": True,
            "target_or_reference_arrays_deserialized": False,
            "test_prediction_artifact_opened": False,
            "commercial_performance_claim_authorized": False,
            "prospective_cohort_required_for_commercial_claim": True,
            "units": units,
        }
    )


def run_campaign(
    *, freeze_spec_path: Path, pretest_index: Path, output_root: Path,
    max_new_units: int | None = None, dry_run: bool = False
) -> dict[str, Any]:
    if max_new_units is not None and int(max_new_units) < 0:
        raise StreamingCampaignError("max units cannot be negative")
    output = output_root.expanduser().resolve()
    control = output / "control"
    control.mkdir(parents=True, exist_ok=True)
    with (control / "campaign.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StreamingCampaignError("another streaming campaign holds the lock") from exc
        spec, _ = load_freeze_spec(freeze_spec_path)
        plan = build_plan(
            freeze_spec_path=freeze_spec_path,
            pretest_index=pretest_index,
            output_root=output,
        )
        plan_path = control / "plan.json"
        _ensure_document(plan_path, plan)
        plan_binding = bind_file(plan_path)
        _ensure_document(
            control / "preexecution_seal.json",
            _content_document(
                {
                    "schema_version": SCHEMA_VERSION,
                    "classification": "locked_hcs_streaming_deployment_preexecution_seal",
                    "plan": plan_binding,
                    "freeze_spec": plan["freeze_spec"],
                    "pretest_index": plan["pretest_index"],
                    "runtime_sources": plan["runtime_sources"],
                    "runtime_identity": plan["runtime_identity"],
                    "target_or_reference_value_artifact_consulted": False,
                    "sealed_artifacts_byte_hashed_for_provenance": True,
                    "target_or_reference_arrays_deserialized": False,
                    "test_prediction_artifact_opened": False,
                    "integrated_pass_before_all_18_allowed": False,
                    "commercial_performance_claim_authorized": False,
                }
            ),
        )
        existing: dict[str, dict[str, Any]] = {}
        for unit in plan["units"]:
            root = _unit_root(output, unit)
            if (root / "receipt.json").is_file():
                existing[unit["unit_id"]] = _validate_receipt(
                    unit, output=output, plan_binding=plan_binding, spec=spec
                )
            elif root.exists():
                raise StreamingCampaignError(f"unit root exists without receipt: {root}")
        seal_path = output / "complete_seal.json"
        if seal_path.exists() and len(existing) != EXPECTED_UNITS:
            raise StreamingCampaignError("complete seal exists for an incomplete campaign")
        limit = 0 if dry_run else (
            EXPECTED_UNITS if max_new_units is None else int(max_new_units)
        )
        receipts: list[dict[str, Any]] = []
        new_units = 0
        for unit in plan["units"]:
            if unit["unit_id"] in existing:
                receipts.append(existing[unit["unit_id"]])
            elif new_units < limit:
                receipts.append(
                    _execute_unit(
                        unit, output=output, plan_binding=plan_binding, spec=spec
                    )
                )
                new_units += 1
            _atomic_json(control / "progress.json", _progress(plan_binding, receipts, sealed=False))
        if len(receipts) == EXPECTED_UNITS:
            seal = _complete_seal(
                plan=plan,
                plan_binding=plan_binding,
                output=output,
                receipts=receipts,
            )
            _ensure_document(seal_path, seal)
            sealed = True
            status = (
                "all_18_streaming_units_integrated_pass"
                if seal["integrated_pass"]
                else "all_18_streaming_units_sealed_with_engineering_gate_failures"
            )
        else:
            sealed = False
            status = "dry_run" if dry_run else "streaming_campaign_incomplete"
        progress = _progress(plan_binding, receipts, sealed=sealed)
        _atomic_json(control / "progress.json", progress)
        result: dict[str, Any] = {
            "status": status,
            "completed_units": len(receipts),
            "new_units": new_units,
            "expected_units": EXPECTED_UNITS,
            "complete_seal_present": sealed,
            "integrated_pass_reported": progress["integrated_pass_reported"],
            "target_or_reference_value_artifact_consulted": False,
            "sealed_artifacts_byte_hashed_for_provenance": True,
            "target_or_reference_arrays_deserialized": False,
            "test_prediction_artifact_opened": False,
            "unit_selection_or_ranking_performed": False,
            "commercial_performance_claim_authorized": False,
            "prospective_cohort_required_for_commercial_claim": True,
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
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--max-units", "--max-new-units", dest="max_new_units", type=int)
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "freeze-spec":
        result = freeze_spec(args.output)
    else:
        result = run_campaign(
            freeze_spec_path=args.freeze_spec,
            pretest_index=args.pretest_index,
            output_root=args.output_root,
            max_new_units=args.max_new_units,
            dry_run=args.dry_run,
        )
    print(json.dumps(_strict(result), sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
