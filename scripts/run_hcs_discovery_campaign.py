#!/usr/bin/env python3
"""Run the strict, resumable HCS discovery DAG without opening outer test data.

The driver deliberately treats the nested proposer index as an append-only
readiness feed.  An outer-fold/seed pair is not touched until its four inner
OOF records and one outer-validation record form an exact, hash-verified
cover.  Downstream artifacts are content/version addressed and are validated
before reuse; an interrupted trainer is preserved and resumed in a fresh
attempt directory rather than overwritten.

Discovery screens are routing information, not kill switches.  A failed i1
screen admits the predeclared i2r evidence refinement and a failed i2r screen
admits i3 tail-risk training.  All trainer calls use ``--discovery-only``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = PROJECT_ROOT / "artifacts/campaigns/harmonic_candidate_set_snn_v2"
DEFAULT_DISCOVERY_INDEX = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/nested_proposer/discovery_index.json"
)
DEFAULT_PLAN = CAMPAIGN_ROOT / "nested_proposer/manifests/plan.json"
DEFAULT_CONTRACT = CAMPAIGN_ROOT / "ADAPTIVE_CAMPAIGN_CONTRACT.json"
DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_discovery"
)
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "artifacts/cache/harmonic_set_v2_nested_discovery"
DEFAULT_RF_CACHE = PROJECT_ROOT / "artifacts/cache/rf32s"
DEFAULT_SVD_CACHE = PROJECT_ROOT / "artifacts/cache/svd_components_all_v1"
DEFAULT_FOLDS = (
    PROJECT_ROOT
    / "artifacts/runs/final_alias_gate_s12_deterministic/fold_assignments.json"
)

SCHEMA_VERSION = 1
DISCOVERY_ROLES = frozenset(("hcs_train_oof", "hcs_validation"))
ITERATION_ORDER = ("i1", "i2r", "i3")
TARGET_KEYS = {
    "overall_mae_bpm_max": ("mae", "max"),
    "identity_macro_mae_bpm_max": ("identity_macro_mae", "max"),
    "rmse_bpm_max": ("rmse", "max"),
    "within_2_fraction_min": ("within_2", "min"),
    "over_5_fraction_max": ("catastrophic_over_5", "max"),
    "high_rr_25_35_mae_bpm_max": ("tail_25_35_mae", "max"),
}


class BoundJSON:
    __slots__ = ("path", "value", "sha256", "size", "raw")

    def __init__(
        self, path: Path, value: dict[str, Any], sha256: str, size: int, raw: bytes
    ) -> None:
        self.path = path
        self.value = value
        self.sha256 = sha256
        self.size = size
        self.raw = raw

    def binding(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "bytes": self.size}


class IterationSpec:
    __slots__ = ("tag", "trainer_iteration", "evidence_flavor")

    def __init__(self, tag: str, trainer_iteration: int, evidence_flavor: str) -> None:
        self.tag = tag
        self.trainer_iteration = trainer_iteration
        self.evidence_flavor = evidence_flavor


ITERATIONS = {
    "i1": IterationSpec("i1", 1, "i1_topk_merge050"),
    "i2r": IterationSpec("i2r", 2, "i2r_posterior_nms125_svd12_merge050"),
    "i3": IterationSpec("i3", 3, "i2r_posterior_nms125_svd12_merge050"),
}


class CommandError(RuntimeError):
    """A recorded child process or its output validation failed."""


class IncompleteTrainingOutput(RuntimeError):
    """A valid sealed attempt was interrupted before validation publication."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def load_bound_json(path: Path, label: str) -> BoundJSON:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {resolved} ({exc})") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be a JSON object: {resolved}")
    return BoundJSON(resolved, value, sha256_bytes(raw), len(raw), raw)


def _require_sha(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError(f"{label} must be a lowercase hexadecimal SHA-256")
    return digest


def _resolve(value: Any, *, relative_to: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable provenance collision: {path}")


def content_address_document(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    document = dict(value)
    document.pop("content_sha256", None)
    document["content_sha256"] = semantic_sha256(document)
    payload = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return document, payload


def publish_campaign_index(root: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    document, payload = content_address_document(value)
    digest = str(document["content_sha256"])
    snapshot = root / "campaign_indexes" / f"{digest}.json"
    _exclusive_write(snapshot, payload)
    latest = root / "campaign_index.json"
    if latest.exists():
        previous = load_bound_json(latest, "current campaign index")
        previous_content = previous.value.get("content_sha256")
        if (
            not isinstance(previous_content, str)
            or canonical_content_sha256(previous.value) != previous_content
        ):
            raise RuntimeError("current campaign index was tampered; refusing to replace it")
        previous_snapshot = root / "campaign_indexes" / f"{previous_content}.json"
        if not previous_snapshot.is_file() or previous_snapshot.read_bytes() != previous.raw:
            raise RuntimeError("current campaign index has no matching immutable snapshot")
    _atomic_write(latest, payload)
    return document


def _snapshot_input(root: Path, bound: BoundJSON, label: str) -> Path:
    destination = root / "input_snapshots" / f"{label}_{bound.sha256}.json"
    _exclusive_write(destination, bound.raw)
    return destination


def _parse_csv_ints(raw: str, label: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{label} must be a comma-separated integer list") from exc
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be non-empty and unique")
    return values


def _parse_iterations(raw: str) -> list[str]:
    aliases = {"1": "i1", "2": "i2r", "i2": "i2r", "3": "i3"}
    values = [aliases.get(part.strip(), part.strip()) for part in raw.split(",") if part.strip()]
    if not values or len(values) != len(set(values)) or any(value not in ITERATIONS for value in values):
        raise ValueError("iterations must be a unique subset of i1,i2r,i3")
    if values != sorted(values, key=ITERATION_ORDER.index):
        raise ValueError("iterations must follow i1,i2r,i3 order")
    return values


def _parse_devices(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("devices must contain at least one device")
    return values


def validate_contract(contract: BoundJSON) -> dict[str, float]:
    value = contract.value
    if value.get("schema_version") != 1:
        raise RuntimeError("adaptive campaign contract schema_version must equal 1")
    raw = value.get("accuracy_targets_per_seed")
    if not isinstance(raw, Mapping):
        raise RuntimeError("adaptive campaign contract has no accuracy targets")
    targets: dict[str, float] = {}
    for key in TARGET_KEYS:
        number = raw.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise RuntimeError(f"adaptive campaign target is missing or invalid: {key}")
        targets[key] = float(number)
    return targets


def validate_plan(plan: BoundJSON, outer_folds: Sequence[int]) -> dict[int, dict[str, Any]]:
    value = plan.value
    if value.get("schema_version") != 1:
        raise RuntimeError("nested plan schema_version must equal 1")
    if canonical_content_sha256(value) != value.get("content_sha256"):
        raise RuntimeError("nested plan canonical content hash mismatch")
    raw_outer = value.get("outer_folds")
    if not isinstance(raw_outer, Mapping):
        raise RuntimeError("nested plan outer_folds must be an object")
    result: dict[int, dict[str, Any]] = {}
    for outer in outer_folds:
        entry = raw_outer.get(str(outer))
        if not isinstance(entry, Mapping) or int(entry.get("outer_test_fold", -1)) != outer:
            raise RuntimeError(f"nested plan has no valid outer fold {outer}")
        units = entry.get("units")
        if not isinstance(units, list):
            raise RuntimeError(f"nested plan outer {outer} units must be an array")
        selected = [unit for unit in units if isinstance(unit, Mapping) and unit.get("role") in DISCOVERY_ROLES]
        roles = [unit.get("role") for unit in selected]
        if len(selected) != 5 or roles.count("hcs_train_oof") != 4 or roles.count("hcs_validation") != 1:
            raise RuntimeError(f"outer {outer} must declare exactly four train-OOF and one validation unit")
        planned: dict[str, dict[str, Any]] = {}
        for unit in selected:
            relative = Path(str(unit.get("manifest", "")))
            if relative.name.startswith("test_pred_"):
                raise RuntimeError("outer-test manifest entered a discovery role in nested plan")
            manifest_path = _resolve(relative, relative_to=plan.path.parent)
            name = manifest_path.name
            if not name or name in planned:
                raise RuntimeError(f"outer {outer} discovery manifest names are duplicated")
            manifest = load_bound_json(manifest_path, "custom split manifest")
            if canonical_content_sha256(manifest.value) != manifest.value.get("content_sha256"):
                raise RuntimeError(f"custom split manifest canonical hash mismatch: {manifest_path}")
            expected_content = _require_sha(unit.get("manifest_content_sha256"), "planned manifest content hash")
            if manifest.value.get("content_sha256") != expected_content:
                raise RuntimeError(f"custom split manifest differs from nested plan: {manifest_path}")
            planned[name] = {
                "role": str(unit["role"]),
                "path": manifest_path,
                "file_sha256": manifest.sha256,
                "content_sha256": expected_content,
            }
        result[outer] = {
            "outer_validation_fold": int(entry.get("outer_validation_fold", -1)),
            "planned": planned,
        }
    return result


def _looks_like_test_record(record: Mapping[str, Any]) -> bool:
    role = str(record.get("role", "")).lower()
    manifest = Path(str(record.get("manifest", ""))).name.lower()
    if "test" in role or manifest.startswith("test_pred_"):
        return True
    for key in ("checkpoint", "all_window_prediction"):
        binding = record.get(key)
        if isinstance(binding, Mapping):
            if any(part.lower().startswith("test_pred_") for part in Path(str(binding.get("path", ""))).parts):
                return True
    return False


def validate_discovery_index(index: BoundJSON) -> list[Mapping[str, Any]]:
    value = index.value
    if value.get("schema_version") != 1 or value.get("outer_test_opened") is not False:
        raise RuntimeError("discovery index must be unopened schema version 1")
    if "content_sha256" in value and canonical_content_sha256(value) != value.get("content_sha256"):
        raise RuntimeError("discovery index canonical content hash mismatch")
    records = value.get("records")
    if not isinstance(records, list):
        raise RuntimeError("discovery index records must be an array")
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("discovery index contains a non-object record")
        if "outer_test_opened" in record and record.get("outer_test_opened") is not False:
            raise RuntimeError("a discovery record claims that outer test was opened")
        if _looks_like_test_record(record):
            raise RuntimeError("outer-test manifest/record entered discovery index")
    return records


def _validate_record_binding(binding: Any, label: str) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise RuntimeError(f"discovery record lacks {label} binding")
    path = _resolve(binding.get("path"), relative_to=PROJECT_ROOT)
    expected = _require_sha(binding.get("sha256"), f"{label} SHA-256")
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"{label} hash mismatch: {path}")
    size = path.stat().st_size
    if "bytes" in binding and int(binding["bytes"]) != size:
        raise RuntimeError(f"{label} size mismatch: {path}")
    return {"path": str(path), "sha256": expected, "bytes": size}


def ready_group(
    records: Sequence[Mapping[str, Any]],
    planned_outer: Mapping[str, Any],
    *,
    outer_fold: int,
    seed: int,
) -> dict[str, Any]:
    matching = [
        record
        for record in records
        if int(record.get("outer_fold", -1)) == outer_fold and int(record.get("seed", -1)) == seed
    ]
    planned: Mapping[str, Mapping[str, Any]] = planned_outer["planned"]
    observed: dict[str, Mapping[str, Any]] = {}
    validated: list[dict[str, Any]] = []
    for record in matching:
        name = Path(str(record.get("manifest", ""))).name
        if name not in planned:
            raise RuntimeError(f"unexpected discovery unit for outer={outer_fold}, seed={seed}: {name}")
        if name in observed:
            raise RuntimeError(f"duplicate discovery unit for outer={outer_fold}, seed={seed}: {name}")
        observed[name] = record
    missing = sorted(set(planned) - set(observed))
    for name in sorted(observed):
        record = observed[name]
        expected = planned[name]
        manifest_path = _resolve(record.get("manifest"), relative_to=PROJECT_ROOT)
        if manifest_path != expected["path"]:
            raise RuntimeError(f"discovery manifest path differs from plan: {manifest_path}")
        manifest_sha = _require_sha(record.get("manifest_sha256"), "record manifest SHA-256")
        if manifest_sha != expected["file_sha256"] or sha256_file(manifest_path) != manifest_sha:
            raise RuntimeError(f"discovery manifest file hash mismatch: {manifest_path}")
        if record.get("role") != expected["role"]:
            raise RuntimeError(f"discovery record role differs from plan: {manifest_path}")
        validated.append(
            {
                "name": name,
                "role": str(record["role"]),
                "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
                "checkpoint": _validate_record_binding(record.get("checkpoint"), "checkpoint"),
                "all_window_prediction": _validate_record_binding(
                    record.get("all_window_prediction"), "all-window prediction"
                ),
            }
        )
    return {
        "status": "ready" if not missing else "waiting_for_five_units",
        "expected_units": sorted(planned),
        "observed_units": sorted(observed),
        "missing_units": missing,
        "units": validated,
        "unit_cover_sha256": semantic_sha256(validated) if not missing else None,
    }


def _stack_array_signature(arrays: Mapping[str, np.ndarray], provenance: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    digest.update(canonical_json_bytes(provenance))
    return digest.hexdigest()


def validate_stack(path: Path, *, outer_fold: int, seed: int, group: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"strict nested stack is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "cache_index", "proposal_available", "outer_fold", "seed", "strict_nested",
            "outer_test_opened", "content_signature_sha256", "provenance_json",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"strict nested stack is missing fields: {missing}")
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    def scalar(name: str) -> Any:
        value = np.asarray(arrays[name])
        if value.ndim != 0:
            raise RuntimeError(f"strict nested stack field must be scalar: {name}")
        return value.item()

    if int(scalar("outer_fold")) != outer_fold or int(scalar("seed")) != seed:
        raise RuntimeError("strict nested stack outer-fold/seed mismatch")
    if (
        np.asarray(arrays["strict_nested"]).dtype != np.bool_
        or np.asarray(arrays["outer_test_opened"]).dtype != np.bool_
        or scalar("strict_nested") is not True
        or scalar("outer_test_opened") is not False
    ):
        raise RuntimeError("strict nested stack is not sealed discovery data")
    try:
        provenance = json.loads(str(scalar("provenance_json")))
    except json.JSONDecodeError as exc:
        raise RuntimeError("strict nested stack provenance is invalid") from exc
    if not isinstance(provenance, dict) or provenance.get("outer_test_opened") is not False:
        raise RuntimeError("strict nested stack provenance opens outer test")
    recorded = str(scalar("content_signature_sha256"))
    if provenance.get("content_signature_sha256") != recorded:
        raise RuntimeError("strict nested stack content signatures disagree")
    unsigned_provenance = dict(provenance)
    unsigned_provenance.pop("content_signature_sha256", None)
    unsigned_arrays = {
        name: value
        for name, value in arrays.items()
        if name not in {"content_signature_sha256", "provenance_json"}
    }
    if _stack_array_signature(unsigned_arrays, unsigned_provenance) != recorded:
        raise RuntimeError("strict nested stack content signature mismatch")
    sources = provenance.get("source_units")
    if not isinstance(sources, list) or len(sources) != 5:
        raise RuntimeError("strict nested stack provenance does not contain exactly five units")
    source_roles = [str(source.get("role", "")) for source in sources if isinstance(source, Mapping)]
    if source_roles.count("hcs_train_oof") != 4 or source_roles.count("hcs_validation") != 1:
        raise RuntimeError("strict nested stack source roles do not form the required cover")
    expected_predictions = {
        unit["all_window_prediction"]["sha256"] for unit in group["units"]
    }
    observed_predictions = {
        str(source.get("sha256", "")) for source in sources if isinstance(source, Mapping)
    }
    if observed_predictions != expected_predictions:
        raise RuntimeError("strict nested stack source predictions differ from discovery index")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "content_signature_sha256": recorded,
        "outer_test_opened": False,
    }


def validate_fallback(path: Path, *, stack: Mapping[str, Any], outer_fold: int, seed: int) -> dict[str, Any]:
    sidecar = path.with_name(f"{path.name}.provenance.json")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError("strict nested fallback CSV/sidecar pair is incomplete")
    provenance = load_bound_json(sidecar, "fallback provenance")
    value = provenance.value
    if value.get("schema_version") != 1 or value.get("strict_nested") is not True or value.get("outer_test_opened") is not False:
        raise RuntimeError("fallback provenance is not sealed strict nested discovery")
    if canonical_content_sha256(value) != value.get("content_sha256"):
        raise RuntimeError("fallback provenance content hash mismatch")
    if int(value.get("outer_fold", -1)) != outer_fold or int(value.get("seed", -1)) != seed:
        raise RuntimeError("fallback outer-fold/seed mismatch")
    source = value.get("source_stack")
    output = value.get("output_csv")
    if not isinstance(source, Mapping) or source.get("sha256") != stack["sha256"]:
        raise RuntimeError("fallback source stack binding mismatch")
    csv_sha = sha256_file(path)
    if not isinstance(output, Mapping) or output.get("sha256") != csv_sha:
        raise RuntimeError("fallback CSV hash mismatch")
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip().lower().split(",")
    if any(name in {"rr_bpm", "target", "label", "reference_rr_bpm", "reference_valid"} for name in header):
        raise RuntimeError("fallback CSV exports a target/reference field")
    return {
        "path": str(path.resolve()),
        "sha256": csv_sha,
        "bytes": path.stat().st_size,
        "provenance": provenance.binding(),
        "outer_test_opened": False,
    }


def _validate_output_bindings(root: Path, outputs: Any) -> None:
    if not isinstance(outputs, Mapping) or not outputs:
        raise RuntimeError("cache manifest output bindings are missing")
    for name, binding in outputs.items():
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"cache output binding is invalid: {name}")
        path = root / str(binding.get("filename", ""))
        if not path.is_file() or path.stat().st_size != int(binding.get("bytes", -1)):
            raise RuntimeError(f"cache output is missing or has wrong size: {name}")
        if sha256_file(path) != binding.get("sha256"):
            raise RuntimeError(f"cache output hash mismatch: {name}")


def validate_cache(path: Path, *, stack: Mapping[str, Any], flavor: str) -> dict[str, Any]:
    manifest = load_bound_json(path / "manifest.json", "harmonic-set cache manifest")
    value = manifest.value
    if value.get("format_version") != 1 or value.get("complete") is not True:
        raise RuntimeError("harmonic-set cache is incomplete")
    if canonical_content_sha256(value) != value.get("content_sha256"):
        raise RuntimeError("harmonic-set cache content hash mismatch")
    proposer = value.get("inputs", {}).get("proposer") if isinstance(value.get("inputs"), Mapping) else None
    if not isinstance(proposer, Mapping) or proposer.get("sha256") != stack["sha256"]:
        raise RuntimeError("harmonic-set cache proposer binding mismatch")
    policy = value.get("candidate_policy")
    evidence = value.get("evidence_policy")
    if not isinstance(policy, Mapping) or not isinstance(evidence, Mapping):
        raise RuntimeError("harmonic-set cache policy metadata is missing")
    if float(policy.get("merge_radius_bpm", -1.0)) != 0.5:
        raise RuntimeError("harmonic-set cache merge radius differs from discovery contract")
    if flavor.startswith("i1_"):
        if policy.get("proposal_selection") != "topk" or int(evidence.get("svd_components", -1)) != 6:
            raise RuntimeError("i1 cache settings mismatch")
    else:
        if (
            policy.get("proposal_selection") != "posterior-nms"
            or float(policy.get("posterior_nms_suppression_bpm", -1.0)) != 1.25
            or policy.get("base_source_policy") != "explicit_expected_then_map_before_direct_modes"
            or int(evidence.get("svd_components", -1)) != 12
            or evidence.get("proposer_posterior_feature_policy")
            != "full_posterior_candidate_local_summaries_plus_exact_row_diagnostics"
        ):
            raise RuntimeError("i2r/i3 cache settings mismatch")
    _validate_output_bindings(path, value.get("outputs"))
    return {
        "path": str(path.resolve()),
        "manifest": manifest.binding(),
        "build_signature_sha256": str(value.get("build_signature_sha256", "")),
        "row_count": int(value.get("row_count", 0)),
        "flavor": flavor,
    }


def screen_metrics(metrics: Mapping[str, Any], targets: Mapping[str, float]) -> dict[str, Any]:
    locked = metrics.get("locked_final")
    if not isinstance(locked, Mapping):
        raise RuntimeError("validation metrics has no locked_final object")
    gates: dict[str, Any] = {}
    passed = True
    for target_key, (metric_key, direction) in TARGET_KEYS.items():
        observed = locked.get(metric_key)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not np.isfinite(observed):
            raise RuntimeError(f"validation screen metric is missing/nonfinite: {metric_key}")
        limit = float(targets[target_key])
        gate_passed = float(observed) <= limit if direction == "max" else float(observed) >= limit
        gates[target_key] = {
            "metric": metric_key,
            "observed": float(observed),
            "threshold": limit,
            "direction": direction,
            "passed": bool(gate_passed),
        }
        passed = passed and gate_passed
    return {
        "status": "passed" if passed else "failed",
        "passed": bool(passed),
        "all_six_accuracy_gates_required": True,
        "gates": gates,
    }


def validate_training(
    path: Path,
    *,
    cache: Mapping[str, Any],
    fallback: Mapping[str, Any],
    outer_fold: int,
    seed: int,
    spec: IterationSpec,
    targets: Mapping[str, float],
) -> dict[str, Any]:
    lock_path = path / "selection_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("trainer attempt has no completed selection lock")
    forbidden = [path / "test_predictions.npz", path / "test_metrics.json"]
    if any(candidate.exists() for candidate in forbidden):
        raise RuntimeError("discovery trainer output contains an outer-test artifact")
    lock = load_bound_json(lock_path, "selection lock")
    value = lock.value
    if (
        value.get("schema_version") != 1
        or value.get("outer_test_not_opened_before_this_lock") is not True
        or int(value.get("outer_fold", -1)) != outer_fold
        or int(value.get("seed", -1)) != seed
        or int(value.get("adaptive_iteration", -1)) != spec.trainer_iteration
    ):
        raise RuntimeError("selection lock does not match strict discovery request")
    if value.get("cache_manifest_sha256") != cache["manifest"]["sha256"]:
        raise RuntimeError("selection lock cache binding mismatch")
    if value.get("fallback_oof_sha256") != fallback["sha256"]:
        raise RuntimeError("selection lock fallback binding mismatch")
    for key, filename in (
        ("checkpoint_sha256", "best_checkpoint.pt"),
        ("scaler_sha256", "scaler.json"),
        ("policy_sha256", "fallback_policy.json"),
        ("run_manifest_sha256", "run_manifest.json"),
    ):
        bound = path / filename
        if not bound.is_file() or sha256_file(bound) != value.get(key):
            raise RuntimeError(f"selection lock artifact binding mismatch: {filename}")
    metrics_path = path / "validation_metrics.json"
    predictions_path = path / "validation_predictions.npz"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise IncompleteTrainingOutput(
            "trainer lock is valid but immutable validation artifacts are incomplete"
        )
    metrics = load_bound_json(metrics_path, "validation metrics")
    screen = screen_metrics(metrics.value, targets)
    return {
        "path": str(path.resolve()),
        "selection_lock": lock.binding(),
        "validation_metrics": metrics.binding(),
        "validation_predictions": bind_file(predictions_path),
        "screen": screen,
        "outer_test_opened": False,
    }


def _no_test_command(command: Sequence[str], *, trainer: bool = False) -> None:
    for argument in command:
        if argument.startswith("--test") or Path(argument).name.lower().startswith("test_pred_"):
            raise RuntimeError("refusing to execute a command containing an outer-test argument")
    if trainer and "--discovery-only" not in command:
        raise RuntimeError("HCS trainer command must include --discovery-only")


class CommandRecorder:
    def __init__(self, root: Path, *, cwd: Path, dry_run: bool) -> None:
        self.root = root
        self.cwd = cwd
        self.dry_run = dry_run
        self.planned: list[dict[str, Any]] = []
        self._file_bindings: dict[str, dict[str, Any]] = {}

    def _bind_command_file(self, raw: str) -> dict[str, Any]:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.cwd / path
        key = str(path.absolute())
        if key not in self._file_bindings:
            self._file_bindings[key] = bind_file(path)
        return self._file_bindings[key]

    def execute(
        self,
        *,
        stage: str,
        semantic: Mapping[str, Any],
        command: Sequence[str],
        validator: Callable[[], dict[str, Any]],
        trainer: bool = False,
    ) -> dict[str, Any]:
        argv = [str(item) for item in command]
        _no_test_command(argv, trainer=trainer)
        command_key = semantic_sha256({"stage": stage, "semantic": semantic})
        plan = {
            "stage": stage,
            "command_key_sha256": command_key,
            "argv": argv,
            "argv_shell_rendering": shlex.join(argv),
            "semantic": dict(semantic),
            "command_source_bindings": [
                self._bind_command_file(argv[0]),
                self._bind_command_file(argv[1]),
            ],
        }
        self.planned.append(plan)
        if self.dry_run:
            return {"status": "dry_run", "command": plan}

        command_root = self.root / "command_provenance" / command_key
        attempt = 0
        while (command_root / f"attempt_{attempt:03d}" / "started.json").exists():
            attempt += 1
        attempt_root = command_root / f"attempt_{attempt:03d}"
        stdout_path = attempt_root / "stdout.log"
        stderr_path = attempt_root / "stderr.log"
        started, started_payload = content_address_document(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "hcs_discovery_command_started",
                "stage": stage,
                "attempt": attempt,
                "command_key_sha256": command_key,
                "cwd": str(self.cwd),
                "argv": argv,
                "command_source_bindings": plan["command_source_bindings"],
                "semantic": dict(semantic),
                "outer_test_opened": False,
            }
        )
        _exclusive_write(attempt_root / "started.json", started_payload)
        return_code = -1
        validation: dict[str, Any] | None = None
        error: str | None = None
        try:
            attempt_root.mkdir(parents=True, exist_ok=True)
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                completed = subprocess.run(
                    argv,
                    cwd=self.cwd,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
                return_code = int(completed.returncode)
            stdout_path.chmod(0o444)
            stderr_path.chmod(0o444)
            if return_code != 0:
                error = f"command exited with status {return_code}"
            else:
                try:
                    validation = validator()
                except Exception as exc:  # validation failures are provenance failures
                    error = f"output validation failed: {type(exc).__name__}: {exc}"
        except BaseException as exc:
            error = f"command interrupted: {type(exc).__name__}: {exc}"
            raise
        finally:
            for log_path in (stdout_path, stderr_path):
                if log_path.is_file():
                    try:
                        log_path.chmod(0o444)
                    except OSError:
                        pass
            completion_value: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "hcs_discovery_command_completed",
                "stage": stage,
                "attempt": attempt,
                "command_key_sha256": command_key,
                "started_content_sha256": started["content_sha256"],
                "return_code": return_code,
                "status": "succeeded" if error is None else "failed",
                "validation": validation,
                "error": error,
                "outer_test_opened": False,
            }
            if stdout_path.is_file():
                completion_value["stdout"] = bind_file(stdout_path)
            if stderr_path.is_file():
                completion_value["stderr"] = bind_file(stderr_path)
            _, completed_payload = content_address_document(completion_value)
            _exclusive_write(attempt_root / "completed.json", completed_payload)
        if error is not None:
            raise CommandError(f"{stage}: {error}; see {attempt_root}")
        assert validation is not None
        return validation


def _cache_flags(flavor: str) -> list[str]:
    common = ["--merge-radius-bpm", "0.5"]
    if flavor.startswith("i1_"):
        return [
            *common,
            "--proposal-selection", "topk",
            "--base-proposals", "none",
            "--svd-components", "6",
        ]
    return [
        *common,
        "--proposal-selection", "posterior-nms",
        "--posterior-nms-suppression-bpm", "1.25",
        "--base-proposals", "expected-map",
        "--svd-components", "12",
        "--proposer-features",
    ]


def _trainer_flags(args: argparse.Namespace, spec: IterationSpec, device: str) -> list[str]:
    flags = [
        "--device", device,
        "--deterministic",
        "--preset", args.preset,
        "--epochs", str(args.epochs),
        "--minimum-epochs", str(args.minimum_epochs),
        "--patience", str(args.patience),
        "--learning-rate", str(args.learning_rate),
        "--adaptive-iteration", str(spec.trainer_iteration),
        "--maximum-coverage", str(args.maximum_coverage),
        "--maximum-fpr", str(args.maximum_fpr),
        "--minimum-precision", str(args.minimum_precision),
        "--discovery-only",
    ]
    if device.startswith("cuda"):
        flags.append("--amp")
    if spec.tag == "i3":
        flags.extend(
            (
                "--tail-weight", str(args.i3_tail_weight),
                "--cvar-weight", str(args.i3_cvar_weight),
                "--warmup-windows", str(args.i3_warmup_windows),
                "--gradient-accumulation-sessions", str(args.i3_gradient_accumulation_sessions),
            )
        )
        flags.extend(args.i3_extra_arg)
    return flags


def _fallback_candidate(base: Path) -> tuple[Path, bool]:
    attempt = 0
    while True:
        candidate = base if attempt == 0 else base.with_name(f"{base.stem}.attempt_{attempt:03d}{base.suffix}")
        sidecar = candidate.with_name(f"{candidate.name}.provenance.json")
        if not candidate.exists() and not sidecar.exists():
            return candidate, False
        if candidate.is_file() and sidecar.is_file():
            return candidate, True
        attempt += 1


def _training_candidate(base: Path, validator: Callable[[Path], dict[str, Any]]) -> tuple[Path, dict[str, Any] | None]:
    base.mkdir(parents=True, exist_ok=True)
    attempts = sorted(path for path in base.glob("attempt_*") if path.is_dir())
    for path in attempts:
        if (path / "selection_lock.json").is_file():
            try:
                return path, validator(path)
            except IncompleteTrainingOutput:
                # Preserve the one-way lock and every historical byte.  A new
                # versioned attempt is the only safe generic recovery when
                # publication was interrupted after lock creation.
                continue
    used = {path.name for path in attempts}
    number = 0
    while f"attempt_{number:03d}" in used:
        number += 1
    return base / f"attempt_{number:03d}", None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-index", type=Path, default=DEFAULT_DISCOVERY_INDEX)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--rf-cache", type=Path, default=DEFAULT_RF_CACHE)
    parser.add_argument("--svd-cache", type=Path, default=DEFAULT_SVD_CACHE)
    parser.add_argument("--fold-assignments", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--stack-builder", type=Path, default=PROJECT_ROOT / "scripts/build_nested_proposer_stack.py")
    parser.add_argument("--fallback-builder", type=Path, default=PROJECT_ROOT / "scripts/build_nested_fallback_oof.py")
    parser.add_argument("--cache-builder", type=Path, default=PROJECT_ROOT / "scripts/build_harmonic_set_cache.py")
    parser.add_argument("--trainer", type=Path, default=PROJECT_ROOT / "scripts/train_harmonic_set_snn.py")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--outer-folds", default="3,4")
    parser.add_argument("--seeds", default="20260828,20260829,20260830")
    parser.add_argument("--iterations", default="i1,i2r,i3")
    parser.add_argument("--devices", default="cuda")
    parser.add_argument("--max-jobs", type=int, help="maximum newly launched HCS trainer jobs this invocation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--preset", default="default")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--maximum-coverage", type=float, default=0.20)
    parser.add_argument("--maximum-fpr", type=float, default=0.01)
    parser.add_argument("--minimum-precision", type=float, default=0.80)
    parser.add_argument("--i3-tail-weight", type=float, default=2.0)
    parser.add_argument("--i3-cvar-weight", type=float, default=0.15)
    parser.add_argument("--i3-warmup-windows", type=int, default=2)
    parser.add_argument("--i3-gradient-accumulation-sessions", type=int, default=4)
    parser.add_argument(
        "--i3-extra-arg",
        action="append",
        default=[],
        help="opaque additional i3 trainer argv; repeat and use --i3-extra-arg=--flag for leading dashes",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.outer_fold_values = _parse_csv_ints(args.outer_folds, "outer folds")
    if any(fold < 0 or fold > 5 for fold in args.outer_fold_values):
        raise SystemExit("outer folds must lie in [0,5]")
    args.seed_values = _parse_csv_ints(args.seeds, "seeds")
    args.iteration_values = _parse_iterations(args.iterations)
    args.device_values = _parse_devices(args.devices)
    if args.max_jobs is not None and args.max_jobs < 1:
        raise SystemExit("--max-jobs must be positive")
    if args.batch_size < 1 or args.minimum_epochs < 1 or args.epochs < args.minimum_epochs or args.patience < 1:
        raise SystemExit("batch/epoch/patience settings are inconsistent")
    if args.learning_rate <= 0 or args.i3_tail_weight < 0 or args.i3_cvar_weight < 0:
        raise SystemExit("learning/tail/CVaR weights are invalid")
    if args.i3_warmup_windows < 0 or args.i3_gradient_accumulation_sessions < 1:
        raise SystemExit("i3 warmup/gradient accumulation settings are invalid")
    for value, label in (
        (args.maximum_coverage, "maximum coverage"),
        (args.maximum_fpr, "maximum FPR"),
        (args.minimum_precision, "minimum precision"),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{label} must lie in [0,1]")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    rf_cache = args.rf_cache.expanduser().resolve()
    svd_cache = args.svd_cache.expanduser().resolve()
    folds = args.fold_assignments.expanduser().resolve()
    discovery = load_bound_json(args.discovery_index, "discovery index")
    plan = load_bound_json(args.plan, "nested plan")
    contract = load_bound_json(args.contract, "adaptive campaign contract")
    records = validate_discovery_index(discovery)
    planned = validate_plan(plan, args.outer_fold_values)
    targets = validate_contract(contract)
    source_paths = {
        "stack_builder": args.stack_builder.expanduser().resolve(),
        "fallback_builder": args.fallback_builder.expanduser().resolve(),
        "cache_builder": args.cache_builder.expanduser().resolve(),
        "trainer": args.trainer.expanduser().resolve(),
        # Do not resolve a virtual-environment interpreter symlink: invoking
        # its base target can silently lose the venv's site-packages.
        "python_executable": args.python_executable.expanduser().absolute(),
    }
    source_bindings = {name: bind_file(path) for name, path in source_paths.items()}
    rf_manifest = bind_file(rf_cache / "manifest.json")
    svd_manifest = bind_file(svd_cache / "manifest.json")
    folds_binding = bind_file(folds)
    if rf_manifest["sha256"] != plan.value.get("cache_manifest_sha256"):
        raise RuntimeError("RF cache manifest differs from nested plan")
    if folds_binding["sha256"] != plan.value.get("fold_assignments_sha256"):
        raise RuntimeError("fold assignments differ from nested plan")

    groups: list[dict[str, Any]] = []
    for outer in args.outer_fold_values:
        for seed in args.seed_values:
            readiness = ready_group(records, planned[outer], outer_fold=outer, seed=seed)
            groups.append(
                {
                    "outer_fold": outer,
                    "seed": seed,
                    "discovery": readiness,
                    "artifacts": {},
                    "iterations": {
                        tag: {"status": "waiting_for_five_units" if readiness["status"] != "ready" else "pending"}
                        for tag in args.iteration_values
                    },
                }
            )

    index: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "retrospective_strict_hcs_discovery_campaign",
        "strict_nested": True,
        "outer_test_opened": False,
        "commercial_performance_claim_eligible": False,
        "failed_screen_is_not_a_kill_switch": True,
        "command_provenance_root": str(artifact_root / "command_provenance"),
        "requested": {
            "outer_folds": list(args.outer_fold_values),
            "seeds": list(args.seed_values),
            "iterations": list(args.iteration_values),
            "devices": list(args.device_values),
            "max_new_training_jobs": args.max_jobs,
        },
        "configuration": {
            "i1_cache": {
                "proposal_selection": "topk",
                "merge_radius_bpm": 0.5,
                "base_proposals": "none",
                "svd_components": 6,
                "proposer_features": False,
            },
            "i2r_and_i3_cache": {
                "proposal_selection": "posterior-nms",
                "posterior_nms_suppression_bpm": 1.25,
                "merge_radius_bpm": 0.5,
                "base_proposals": "expected-map",
                "svd_components": 12,
                "proposer_features": True,
            },
            "trainer": {
                "preset": args.preset,
                "epochs": args.epochs,
                "minimum_epochs": args.minimum_epochs,
                "patience": args.patience,
                "learning_rate": args.learning_rate,
                "maximum_coverage": args.maximum_coverage,
                "maximum_fpr": args.maximum_fpr,
                "minimum_precision": args.minimum_precision,
                "deterministic": True,
                "discovery_only": True,
            },
            "i3": {
                "tail_weight": args.i3_tail_weight,
                "cvar_weight": args.i3_cvar_weight,
                "warmup_windows": args.i3_warmup_windows,
                "gradient_accumulation_sessions": args.i3_gradient_accumulation_sessions,
                "extra_argv": list(args.i3_extra_arg),
            },
        },
        "inputs": {
            "discovery_index": discovery.binding(),
            "nested_plan": {**plan.binding(), "content_sha256": plan.value["content_sha256"]},
            "adaptive_contract": contract.binding(),
            "rf_cache_manifest": rf_manifest,
            "svd_cache_manifest": svd_manifest,
            "fold_assignments": folds_binding,
        },
        "source_bindings": source_bindings,
        "accuracy_targets": targets,
        "groups": groups,
    }
    if not args.dry_run:
        artifact_root.mkdir(parents=True, exist_ok=True)
        discovery_snapshot = _snapshot_input(artifact_root, discovery, "discovery_index")
        index["inputs"]["discovery_index_snapshot"] = bind_file(discovery_snapshot)
        publish_campaign_index(artifact_root, index)
    else:
        discovery_snapshot = discovery.path

    recorder = CommandRecorder(artifact_root, cwd=PROJECT_ROOT, dry_run=bool(args.dry_run))
    launched_jobs = 0

    def save() -> None:
        if not args.dry_run:
            publish_campaign_index(artifact_root, index)

    for group_ordinal, group_record in enumerate(groups):
        outer = int(group_record["outer_fold"])
        seed = int(group_record["seed"])
        readiness = group_record["discovery"]
        if readiness["status"] != "ready":
            continue
        unit_root = artifact_root / f"outer_{outer}" / f"seed_{seed}"
        common_semantic = {
            "schema_version": SCHEMA_VERSION,
            "outer_fold": outer,
            "seed": seed,
            "plan_content_sha256": plan.value["content_sha256"],
            "unit_cover_sha256": readiness["unit_cover_sha256"],
        }
        stack_signature = semantic_sha256(
            {**common_semantic, "stage": "strict_stack", "builder": source_bindings["stack_builder"]}
        )
        stack_path = unit_root / "stack" / f"strict_stack_v1_{stack_signature}.npz"
        stack_validator = lambda: validate_stack(
            stack_path, outer_fold=outer, seed=seed, group=readiness
        )
        if stack_path.exists():
            stack = stack_validator()
            stack_status = "complete"
        else:
            command = [
                str(source_paths["python_executable"]), str(source_paths["stack_builder"]),
                "--discovery-index", str(discovery_snapshot),
                "--plan", str(plan.path),
                "--cache-dir", str(rf_cache),
                "--outer-fold", str(outer), "--seed", str(seed),
                "--output", str(stack_path),
            ]
            stack = recorder.execute(
                stage="build_strict_stack",
                semantic={**common_semantic, "signature": stack_signature, "output": str(stack_path)},
                command=command,
                validator=stack_validator,
            )
            stack_status = "dry_run" if args.dry_run else "complete"
        group_record["artifacts"]["stack"] = {"status": stack_status, **stack}
        if args.dry_run:
            # Semantic signatures, not unknown future content hashes, determine
            # all downstream immutable paths during a dry run.
            stack = {"sha256": f"semantic:{stack_signature}", "path": str(stack_path)}
        save()

        fallback_signature = semantic_sha256(
            {**common_semantic, "stage": "strict_fallback", "stack_signature": stack_signature,
             "builder": source_bindings["fallback_builder"]}
        )
        fallback_base = unit_root / "fallback" / f"strict_fallback_v1_{fallback_signature}.csv"
        fallback_path, fallback_pair_exists = _fallback_candidate(fallback_base)
        fallback_validator = lambda: validate_fallback(
            fallback_path, stack=stack, outer_fold=outer, seed=seed
        )
        if fallback_pair_exists:
            fallback = fallback_validator()
            fallback_status = "complete"
        else:
            command = [
                str(source_paths["python_executable"]), str(source_paths["fallback_builder"]),
                "--stack", str(stack_path), "--output", str(fallback_path),
            ]
            fallback = recorder.execute(
                stage="build_strict_fallback",
                semantic={**common_semantic, "signature": fallback_signature,
                          "stack_sha256": stack["sha256"], "output": str(fallback_path)},
                command=command,
                validator=fallback_validator,
            )
            fallback_status = "dry_run" if args.dry_run else "complete"
        group_record["artifacts"]["fallback"] = {"status": fallback_status, **fallback}
        if args.dry_run:
            fallback = {"sha256": f"semantic:{fallback_signature}", "path": str(fallback_path)}
        save()

        caches: dict[str, dict[str, Any]] = {}
        previous_screen: dict[str, Any] | None = None
        for iteration_ordinal, tag in enumerate(args.iteration_values):
            iteration_record = group_record["iterations"][tag]
            spec = ITERATIONS[tag]
            if previous_screen is not None and previous_screen.get("passed") is True:
                iteration_record.update(
                    status="not_required_previous_screen_passed",
                    predecessor_screen=previous_screen,
                )
                save()
                continue
            flavor = spec.evidence_flavor
            if flavor not in caches:
                cache_signature = semantic_sha256(
                    {
                        **common_semantic,
                        "stage": "harmonic_cache",
                        "stack_signature": stack_signature,
                        "flavor": flavor,
                        "builder": source_bindings["cache_builder"],
                        "rf_manifest": rf_manifest["sha256"],
                        "svd_manifest": svd_manifest["sha256"],
                        "fold_assignments": folds_binding["sha256"],
                    }
                )
                cache_path = cache_root / f"outer_{outer}" / f"seed_{seed}" / f"{flavor}_v1_{cache_signature}"
                cache_validator = lambda p=cache_path, f=flavor: validate_cache(p, stack=stack, flavor=f)
                if cache_path.exists():
                    cache = cache_validator()
                    cache_status = "complete"
                else:
                    command = [
                        str(source_paths["python_executable"]), str(source_paths["cache_builder"]),
                        "--rf-cache", str(rf_cache), "--svd-cache", str(svd_cache),
                        "--proposer", str(stack_path), "--fold-assignments", str(folds),
                        "--output-dir", str(cache_path), "--batch-size", str(args.batch_size),
                        *_cache_flags(flavor),
                    ]
                    cache = recorder.execute(
                        stage=f"build_cache_{flavor}",
                        semantic={**common_semantic, "signature": cache_signature,
                                  "stack_sha256": stack["sha256"], "flavor": flavor,
                                  "output": str(cache_path)},
                        command=command,
                        validator=cache_validator,
                    )
                    cache_status = "dry_run" if args.dry_run else "complete"
                caches[flavor] = cache
                group_record["artifacts"].setdefault("caches", {})[flavor] = {
                    "status": cache_status,
                    **cache,
                }
                if args.dry_run:
                    caches[flavor] = {
                        "path": str(cache_path),
                        "manifest": {"sha256": f"semantic:{cache_signature}"},
                    }
                save()
            cache = caches[flavor]

            if args.max_jobs is not None and launched_jobs >= args.max_jobs:
                iteration_record["status"] = "deferred_max_jobs"
                save()
                previous_screen = None
                continue
            # Device ownership is a pure function of the requested DAG, not
            # of how many jobs happened to be reused in this invocation.
            # This keeps a max-jobs resume on the same immutable run path.
            job_ordinal = group_ordinal * len(args.iteration_values) + iteration_ordinal
            device = args.device_values[job_ordinal % len(args.device_values)]
            trainer_flags = _trainer_flags(args, spec, device)
            training_signature = semantic_sha256(
                {
                    **common_semantic,
                    "stage": "hcs_training",
                    "iteration": tag,
                    "cache_flavor": flavor,
                    "cache_signature": cache.get("build_signature_sha256", cache["manifest"]["sha256"]),
                    "fallback_signature": fallback_signature,
                    "trainer": source_bindings["trainer"],
                    "trainer_flags": trainer_flags,
                }
            )
            training_base = unit_root / "training" / f"{tag}_v1_{training_signature}"
            validate_attempt = lambda path: validate_training(
                path,
                cache=cache,
                fallback=fallback,
                outer_fold=outer,
                seed=seed,
                spec=spec,
                targets=targets,
            )
            if args.dry_run:
                training_path, existing_training = training_base / "attempt_000", None
            else:
                training_path, existing_training = _training_candidate(training_base, validate_attempt)
            if existing_training is not None:
                training = existing_training
                training_status = "complete"
            else:
                command = [
                    str(source_paths["python_executable"]), str(source_paths["trainer"]),
                    "--cache", str(cache["path"]), "--fallback-oof", str(fallback["path"]),
                    "--output-dir", str(training_path), "--fold", str(outer), "--seed", str(seed),
                    *trainer_flags,
                ]
                training = recorder.execute(
                    stage=f"train_hcs_{tag}",
                    semantic={**common_semantic, "signature": training_signature,
                              "iteration": tag, "device": device, "output": str(training_path)},
                    command=command,
                    validator=lambda: validate_attempt(training_path),
                    trainer=True,
                )
                if not args.dry_run:
                    launched_jobs += 1
                training_status = "dry_run" if args.dry_run else "complete"
            iteration_record.update(
                status=training_status,
                output=training,
                device=device,
                cache_flavor=flavor,
            )
            if args.dry_run:
                previous_screen = {"passed": False, "status": "unknown_until_execution"}
            else:
                previous_screen = training["screen"]
                iteration_record["screen"] = previous_screen
            save()

    index["summary"] = {
        "ready_groups": sum(group["discovery"]["status"] == "ready" for group in groups),
        "waiting_groups": sum(group["discovery"]["status"] != "ready" for group in groups),
        "completed_training_jobs": sum(
            iteration.get("status") == "complete"
            for group in groups
            for iteration in group["iterations"].values()
        ),
        "planned_commands": len(recorder.planned) if args.dry_run else 0,
        "outer_test_opened": False,
    }
    index["planned_commands"] = recorder.planned if args.dry_run else []
    if not args.dry_run:
        index = publish_campaign_index(artifact_root, index)
    else:
        index, _ = content_address_document(index)
    return index


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "content_sha256": result["content_sha256"],
                "summary": result["summary"],
                "outer_test_opened": False,
                "dry_run": bool(args.dry_run),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
