#!/usr/bin/env python3
"""Complete the frozen i3 6-fold x 3-seed pre-test training matrix.

This driver is intentionally *not* an outer-test runner.  It consumes only the
separate, fully nested, non-test proposer plan/index, seals the already chosen
``default`` i3 capacity and common fallback policy, reuses the six discovery
units for folds 3/4, and builds/trains the twelve missing units.  Validation
scores are recorded but never branch the DAG.

The only path out of this program is a hash-complete pre-test index.  A later
post-lock runner may construct label-free outer-test inputs from that index;
this program has no test-manifest argument and rejects test artifacts at every
boundary.
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
from typing import Any, Mapping, Sequence
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

# Reuse the byte-level validators used by the already completed strict
# discovery campaign.  This file is itself hash-bound as an effective source.
import run_hcs_discovery_campaign as discovery  # noqa: E402
import seal_runtime_inputs as runtime_seal  # noqa: E402


CAMPAIGN_ROOT = PROJECT_ROOT / "artifacts/campaigns/harmonic_candidate_set_snn_v2"
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2"
CONTROL_ROOT = CAMPAIGN_ROOT / "nested_proposer/full_oof_non_test/control"
DEFAULT_PLAN = CONTROL_ROOT / "plan.json"
DEFAULT_INDEX = CONTROL_ROOT / "index.json"
DEFAULT_COMMON_ROOT = RUN_ROOT / "hcs_discovery/i3_common_lock"
DEFAULT_FREEZE_ROOT = CAMPAIGN_ROOT / "source_snapshots/i3_final"
DEFAULT_ARTIFACT_ROOT = RUN_ROOT / "hcs_fixed_i3_pretest"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "artifacts/cache/harmonic_set_v2_fixed_i3_pretest"
DEFAULT_REUSE_ROOT = RUN_ROOT / "hcs_discovery/i3_default"
DEFAULT_RF_CACHE = PROJECT_ROOT / "artifacts/cache/rf32s"
DEFAULT_SVD_CACHE = PROJECT_ROOT / "artifacts/cache/svd_components_all_v1"
DEFAULT_FOLDS = RUN_ROOT.parent / "final_alias_gate_s12_deterministic/fold_assignments.json"
DEFAULT_RUNTIME_SEAL = (
    CAMPAIGN_ROOT
    / "nested_proposer/current_source_merged/fixed_i3_pretest_runtime_seal.json"
)
DEFAULT_RETRAIN_IMPACT_AUDIT = (
    CAMPAIGN_ROOT
    / "nested_proposer/current_source_merged/retrain_impact_audit.json"
)

FOLDS = tuple(range(6))
SEEDS = (20260828, 20260829, 20260830)
REUSED_FOLDS = frozenset((3, 4))
MISSING_FOLDS = frozenset((0, 1, 2, 5))
ROLES = frozenset(("hcs_train_oof", "hcs_validation"))
SCHEMA_VERSION = 1
PARAMETER_COUNT = 195_603
MAXIMUM_PARAMETERS = 750_000

REQUIRED_TRAINING_FILES = (
    "best_checkpoint.pt",
    "history.json",
    "run_manifest.json",
    "scaler.json",
    "fallback_policy.json",
    "selection_lock.json",
    "validation_metrics.json",
    "validation_predictions.npz",
)
FORBIDDEN_OUTPUT_NAMES = frozenset(
    ("test_predictions.npz", "test_metrics.json", "test_evaluation_manifest.json")
)


def parse_force_retrain_units(raw: str) -> frozenset[tuple[int, int]]:
    """Parse explicit common-unit provenance replacements.

    Folds 3/4 are normally reused because they supplied the frozen capacity and
    common-policy evidence.  If a later proposer provenance audit replaces any
    of their five-unit source stacks, the affected HCS unit must be trained
    again from the replacement stack instead of silently reusing an artifact
    bound to the old predictions.
    """

    text = raw.strip()
    if not text:
        return frozenset()
    result: set[tuple[int, int]] = set()
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError(
                "--force-retrain-units must be comma-separated OUTER_FOLD:SEED pairs"
            )
        try:
            key = (int(parts[0]), int(parts[1]))
        except ValueError as exc:
            raise ValueError(
                "--force-retrain-units contains a non-integer fold or seed"
            ) from exc
        if key in result:
            raise ValueError("--force-retrain-units contains a duplicate pair")
        if key[0] not in REUSED_FOLDS or key[1] not in SEEDS:
            raise ValueError(
                "--force-retrain-units may name only the six common folds 3/4 units"
            )
        result.add(key)
    return frozenset(result)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    document = dict(value)
    document.pop("content_sha256", None)
    return semantic_sha256(document)


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


def load_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {resolved} ({exc})") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object: {resolved}")
    return value, {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def content_document(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    document = dict(value)
    document.pop("content_sha256", None)
    document["content_sha256"] = semantic_sha256(document)
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    return document, payload


def exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable provenance collision: {path}")
    path.chmod(0o444)


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_current_document(path: Path, snapshot_root: Path, label: str) -> None:
    if not path.exists():
        return
    value, _ = load_json(path, label)
    content_hash = value.get("content_sha256")
    if not isinstance(content_hash, str) or canonical_content_sha256(value) != content_hash:
        raise RuntimeError(f"tampered {label}: {path}")
    snapshot = snapshot_root / f"{content_hash}.json"
    if not snapshot.is_file() or snapshot.read_bytes() != path.read_bytes():
        raise RuntimeError(f"{label} lacks its immutable snapshot: {path}")


def publish_current(root: Path, filename: str, value: Mapping[str, Any]) -> dict[str, Any]:
    snapshots = root / f"{Path(filename).stem}_snapshots"
    current = root / filename
    _validate_current_document(current, snapshots, Path(filename).stem)
    document, payload = content_document(value)
    exclusive_write(snapshots / f"{document['content_sha256']}.json", payload)
    atomic_write(current, payload)
    return document


def _require_hash(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError(f"{label} must be a lowercase SHA-256")
    return digest


def _resolve(raw: Any, base: Path) -> Path:
    path = Path(str(raw)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _manifest_path(raw: Any, plan_path: Path) -> Path:
    text = str(raw)
    lowered = Path(text).name.lower()
    if "test" in lowered or lowered.startswith("test_pred_"):
        raise RuntimeError("outer-test manifest entered the non-test plan")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [
        (plan_path.parent / path).resolve(),
        (plan_path.parent.parent / "manifests" / path).resolve(),
    ]
    if path.parts and path.parts[0] == "manifests":
        candidates.append((plan_path.parent.parent / path).resolve())
    existing = {candidate for candidate in candidates if candidate.is_file()}
    if len(existing) != 1:
        raise RuntimeError(
            f"non-test manifest must resolve uniquely relative to the plan: {text}"
        )
    return existing.pop()


def _assert_non_test_record(record: Mapping[str, Any], label: str) -> None:
    role = str(record.get("role", "")).lower()
    manifest = Path(str(record.get("manifest", ""))).name.lower()
    if role not in ROLES or "test" in role or "test" in manifest:
        raise RuntimeError(f"outer-test or invalid role entered {label}")
    for key in ("checkpoint", "all_window_prediction"):
        binding = record.get(key)
        if isinstance(binding, Mapping):
            parts = Path(str(binding.get("path", ""))).parts
            if any(part.lower().startswith("test_pred_") for part in parts):
                raise RuntimeError(f"outer-test artifact entered {label}")


def validate_non_test_plan_index(
    plan_path: Path, index_path: Path
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    """Validate and hash the exact 90-unit non-test proposer cover."""

    plan, plan_binding = load_json(plan_path, "six-fold non-test proposer plan")
    index, index_binding = load_json(index_path, "six-fold non-test proposer index")
    if plan.get("schema_version") != 1:
        raise RuntimeError("non-test proposer plan schema_version must equal 1")
    if canonical_content_sha256(plan) != plan.get("content_sha256"):
        raise RuntimeError("non-test proposer plan content hash mismatch")
    if plan.get("outer_test_opened") is not False:
        raise RuntimeError("non-test proposer plan is not test-sealed")
    if int(plan.get("outer_test_record_count", 0)) != 0:
        raise RuntimeError("non-test proposer plan declares outer-test records")
    if (
        plan.get("classification")
        != "retrospective_fully_nested_non_test_proposer_campaign_plan"
        or int(plan.get("requested_units", -1)) != 90
        or plan.get("outer_folds") != list(FOLDS)
        or plan.get("seeds") != list(SEEDS)
        or plan.get("roles") != ["hcs_train_oof", "hcs_validation"]
    ):
        raise RuntimeError("non-test proposer plan does not declare the exact 90-unit matrix")
    units = plan.get("units")
    if not isinstance(units, list) or len(units) != 90:
        raise RuntimeError("non-test proposer plan must contain exactly 90 units")

    planned: dict[tuple[int, int, str], dict[str, Any]] = {}
    group_roles: dict[tuple[int, int], list[str]] = {
        (fold, seed): [] for fold in FOLDS for seed in SEEDS
    }
    for unit in units:
        if not isinstance(unit, Mapping):
            raise RuntimeError("non-test plan contains a non-object unit")
        _assert_non_test_record(unit, "non-test proposer plan")
        outer = int(unit.get("outer_fold", -1))
        seed = int(unit.get("seed", -1))
        if outer not in FOLDS or seed not in SEEDS:
            raise RuntimeError("non-test plan unit lies outside the fixed matrix")
        manifest_path = _manifest_path(unit.get("manifest"), plan_path.resolve())
        name = manifest_path.name
        key = (outer, seed, name)
        if key in planned:
            raise RuntimeError(f"duplicate planned manifest: {key}")
        validation_fold = (outer + 1) % len(FOLDS)
        training_folds = sorted(set(FOLDS) - {outer, validation_fold})
        expected_names = {f"inner_pred_{fold}.json" for fold in training_folds}
        expected_names.add(f"validation_pred_{validation_fold}.json")
        if name not in expected_names:
            raise RuntimeError(f"non-test plan unit violates the fixed split rotation: {key}")
        expected_role = (
            "hcs_validation" if name.startswith("validation_pred_") else "hcs_train_oof"
        )
        if unit.get("role") != expected_role:
            raise RuntimeError(f"non-test plan unit role/name mismatch: {key}")
        manifest, manifest_binding = load_json(manifest_path, "non-test split manifest")
        if canonical_content_sha256(manifest) != manifest.get("content_sha256"):
            raise RuntimeError(f"non-test manifest content hash mismatch: {manifest_path}")
        expected_content = _require_hash(
            unit.get("manifest_content_sha256"), "planned manifest content hash"
        )
        if manifest.get("content_sha256") != expected_content:
            raise RuntimeError(f"non-test manifest differs from plan: {manifest_path}")
        if "manifest_sha256" in unit and unit.get("manifest_sha256") != manifest_binding["sha256"]:
            raise RuntimeError(f"planned manifest file hash mismatch: {manifest_path}")
        planned[key] = {
            "role": str(unit["role"]),
            "path": manifest_path,
            "file_sha256": manifest_binding["sha256"],
            "content_sha256": expected_content,
        }
        group_roles[(outer, seed)].append(str(unit["role"]))
    for key, roles in group_roles.items():
        if len(roles) != 5 or roles.count("hcs_train_oof") != 4 or roles.count("hcs_validation") != 1:
            raise RuntimeError(f"non-test plan group does not form 4+1 cover: {key}")

    if (
        index.get("schema_version") != 1
        or index.get("classification")
        != "retrospective_fully_nested_non_test_proposer_index"
        or index.get("outer_test_opened") is not False
        or int(index.get("outer_test_record_count", -1)) != 0
    ):
        raise RuntimeError("six-fold proposer index is not the sealed non-test index")
    if "content_sha256" in index and canonical_content_sha256(index) != index.get("content_sha256"):
        raise RuntimeError("six-fold proposer index content hash mismatch")
    campaign_plan = index.get("campaign_plan")
    if (
        index.get("campaign_plan_content_sha256") != plan["content_sha256"]
        or not isinstance(campaign_plan, Mapping)
        or campaign_plan.get("content_sha256") != plan["content_sha256"]
        or campaign_plan.get("sha256") != plan_binding["sha256"]
        or _resolve(campaign_plan.get("path"), PROJECT_ROOT) != plan_path.resolve()
    ):
        raise RuntimeError("six-fold proposer index is bound to another plan")
    if int(index.get("requested_units", -1)) != 90 or int(index.get("completed_units", -1)) != 90:
        raise RuntimeError("six-fold non-test proposer readiness is incomplete (90/90 required)")
    records = index.get("records")
    if not isinstance(records, list) or len(records) != 90:
        raise RuntimeError("six-fold non-test proposer index must contain exactly 90 records")

    observed: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("six-fold proposer index contains a non-object record")
        _assert_non_test_record(record, "six-fold proposer index")
        outer = int(record.get("outer_fold", -1))
        seed = int(record.get("seed", -1))
        if outer not in FOLDS or seed not in SEEDS:
            raise RuntimeError("six-fold proposer record lies outside the fixed matrix")
        name = Path(str(record.get("manifest", ""))).name
        key = (outer, seed, name)
        if key in observed:
            raise RuntimeError(f"duplicate six-fold proposer record: {key}")
        expected = planned.get((outer, seed, name))
        if expected is None or record.get("role") != expected["role"]:
            raise RuntimeError(f"six-fold proposer record differs from plan: {key}")
        manifest_path = _resolve(record.get("manifest"), PROJECT_ROOT)
        if manifest_path != expected["path"]:
            raise RuntimeError(f"six-fold proposer manifest path differs from plan: {manifest_path}")
        if (
            _require_hash(record.get("manifest_sha256"), "record manifest hash")
            != expected["file_sha256"]
            or sha256_file(manifest_path) != expected["file_sha256"]
        ):
            raise RuntimeError(f"six-fold proposer manifest hash mismatch: {manifest_path}")
        if "manifest_content_sha256" in record and (
            record.get("manifest_content_sha256") != expected["content_sha256"]
        ):
            raise RuntimeError(f"record manifest content hash mismatch: {manifest_path}")
        for binding_name in ("checkpoint", "all_window_prediction"):
            binding = record.get(binding_name)
            if not isinstance(binding, Mapping):
                raise RuntimeError(f"proposer record lacks {binding_name} binding")
            artifact = _resolve(binding.get("path"), PROJECT_ROOT)
            expected_hash = _require_hash(binding.get("sha256"), f"{binding_name} hash")
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise RuntimeError(f"proposer {binding_name} hash mismatch: {artifact}")
            if "bytes" in binding and artifact.stat().st_size != int(binding["bytes"]):
                raise RuntimeError(f"proposer {binding_name} byte-size mismatch: {artifact}")
        observed[key] = record

    groups: dict[tuple[int, int], dict[str, Any]] = {}
    for outer in FOLDS:
        for seed in SEEDS:
            expected_names = sorted(
                name
                for planned_outer, planned_seed, name in planned
                if planned_outer == outer and planned_seed == seed
            )
            group_records = [observed[(outer, seed, name)] for name in expected_names]
            units = []
            for record in group_records:
                units.append(
                    {
                        "name": Path(str(record["manifest"])).name,
                        "role": str(record["role"]),
                        "manifest": {
                            "path": str(_resolve(record["manifest"], PROJECT_ROOT)),
                            "sha256": str(record["manifest_sha256"]),
                        },
                        "checkpoint": dict(record["checkpoint"]),
                        "all_window_prediction": dict(record["all_window_prediction"]),
                    }
                )
            groups[(outer, seed)] = {
                "status": "ready",
                "units": units,
                "unit_cover_sha256": semantic_sha256(units),
            }
    if len(groups) != 18 or len(observed) != 90:
        raise RuntimeError("non-test proposer cover is not exactly 18 groups / 90 units")
    return groups, {
        "plan": {**plan_binding, "content_sha256": plan["content_sha256"]},
        "index": {
            **index_binding,
            "content_sha256": index.get("content_sha256"),
            "completed_units": 90,
        },
    }


def validate_retrain_impact_audit(
    audit_path: Path,
    merged_index_path: Path,
    requested_force_units: frozenset[tuple[int, int]] | None,
) -> tuple[frozenset[tuple[int, int]], dict[str, Any]]:
    """Bind the byte-level proposer drift audit to the merged 60+30 index.

    An omitted CLI override means "use the audited set".  An explicit set is
    accepted only when it exactly matches the audit; operators cannot silently
    reuse a stale common HCS stack or force an unrelated replacement.
    """

    audit, audit_binding = load_json(
        audit_path, "nested proposer retrain impact audit"
    )
    if (
        audit.get("schema_version") != 1
        or audit.get("classification")
        != "retrospective_nested_proposer_retrain_impact_audit"
        or audit.get("commercial_claim_authorized") is not False
        or audit.get("outer_test_opened") is not False
        or int(audit.get("outer_test_record_count", -1)) != 0
        or audit.get("target_or_reference_accessed") is not False
        or audit.get("source_campaigns_hash_complete") is not True
        or audit.get("source_plans_compatible") is not True
        or canonical_content_sha256(audit) != audit.get("content_sha256")
    ):
        raise RuntimeError("proposer retrain impact audit invariants are invalid")
    comparison = audit.get("comparison")
    if not isinstance(comparison, Mapping):
        raise RuntimeError("proposer retrain impact audit lacks comparison")
    try:
        audited = parse_force_retrain_units(
            str(comparison.get("force_retrain_units_cli_value", ""))
        )
    except ValueError as exc:
        raise RuntimeError("proposer retrain impact audit has invalid forced units") from exc
    declared = comparison.get("affected_hcs_units")
    if not isinstance(declared, list):
        raise RuntimeError("proposer retrain impact audit lacks affected HCS units")
    try:
        declared_pairs = frozenset(
            (int(item["outer_fold"]), int(item["seed"]))
            for item in declared
            if isinstance(item, Mapping)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("proposer retrain impact audit affected units are invalid") from exc
    if (
        len(declared_pairs) != len(declared)
        or declared_pairs != audited
        or int(comparison.get("force_retrain_unit_count", -1)) != len(audited)
        or comparison.get("checkpoint_change_alone_forces_hcs_retrain") is not False
        or comparison.get("prediction_paths_ignored_after_bound_file_validation") is not True
    ):
        raise RuntimeError("proposer retrain impact decision topology is inconsistent")
    if requested_force_units is not None and requested_force_units != audited:
        raise RuntimeError(
            "--force-retrain-units differs from the sealed proposer retrain impact audit"
        )

    merged, _ = load_json(merged_index_path, "merged 60+30 proposer index")
    if (
        merged.get("merge_classification")
        != "retrospective_current_source_uniform_90_unit_proposer_index"
        or canonical_content_sha256(merged) != merged.get("content_sha256")
    ):
        raise RuntimeError("fixed-i3 index is not the sealed current-source 60+30 merge")
    provenance = merged.get("merge_provenance")
    inputs = audit.get("inputs")
    if not isinstance(provenance, Mapping) or not isinstance(inputs, Mapping):
        raise RuntimeError("merge/audit provenance is missing")

    def same_binding(left: Any, right: Any, label: str) -> None:
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise RuntimeError(f"missing merge/audit binding: {label}")
        left_path = _resolve(left.get("path"), PROJECT_ROOT)
        right_path = _resolve(right.get("path"), PROJECT_ROOT)
        if (
            left_path != right_path
            or left.get("sha256") != right.get("sha256")
            or left.get("content_sha256") != right.get("content_sha256")
        ):
            raise RuntimeError(f"merge and retrain impact audit differ: {label}")

    source_indexes = provenance.get("source_indexes")
    if not isinstance(source_indexes, Mapping):
        raise RuntimeError("merged index lacks source indexes")
    same_binding(inputs.get("main_index"), source_indexes.get("main"), "main index")
    same_binding(
        inputs.get("retrain_index"),
        source_indexes.get("current_source_retrain_f34"),
        "folds-3/4 retrain index",
    )
    same_binding(
        inputs.get("full_plan"),
        provenance.get("full_split_authority_plan"),
        "full split-authority plan",
    )
    same_binding(
        inputs.get("retrain_plan"), provenance.get("retrain_plan"), "retrain plan"
    )
    return audited, {
        **audit_binding,
        "content_sha256": audit["content_sha256"],
        "audited_force_retrain_units": [
            {"outer_fold": fold, "seed": seed} for fold, seed in sorted(audited)
        ],
    }


def validate_freeze(freeze_root: Path) -> dict[str, Any]:
    root = freeze_root.expanduser().resolve()
    manifest, binding = load_json(root / "MANIFEST.json", "frozen i3 source manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("declared_before_any_i3_score") is not True
        or manifest.get("outer_test_opened") is not False
    ):
        raise RuntimeError("i3 source freeze is not valid prelaunch evidence")
    expected = manifest.get("files")
    if not isinstance(expected, Mapping):
        raise RuntimeError("i3 source freeze has no file hashes")
    current = {
        "train_harmonic_set_snn.py": PROJECT_ROOT / "scripts/train_harmonic_set_snn.py",
        "harmonic_set_models.py": PROJECT_ROOT / "src/snn_rr/harmonic_set_models.py",
        "ADAPTIVE_CAMPAIGN_CONTRACT.json": CAMPAIGN_ROOT / "ADAPTIVE_CAMPAIGN_CONTRACT.json",
        "harmonic_set_v2.yaml": PROJECT_ROOT / "configs/harmonic_set_v2.yaml",
        "run_gpu_admitted.py": PROJECT_ROOT / "scripts/run_gpu_admitted.py",
    }
    current_bindings: dict[str, Any] = {}
    snapshot_bindings: dict[str, Any] = {}
    for name, path in current.items():
        digest = _require_hash(expected.get(name), f"frozen hash for {name}")
        current_binding = bind_file(path)
        snapshot_binding = bind_file(root / name)
        if current_binding["sha256"] != digest or snapshot_binding["sha256"] != digest:
            raise RuntimeError(f"i3 frozen source/config changed: {name}")
        current_bindings[name] = current_binding
        snapshot_bindings[name] = snapshot_binding
    limits = manifest.get("parameter_limits", {})
    if (
        int(limits.get("maximum_parameters", -1)) != MAXIMUM_PARAMETERS
        or int(limits.get("default_at_571_features", -1)) != PARAMETER_COUNT
        or limits.get("hard_fail_before_training") is not True
    ):
        raise RuntimeError("i3 source freeze parameter limits changed")
    return {
        "manifest": binding,
        "frozen_hashes": dict(expected),
        "current": current_bindings,
        "snapshots": snapshot_bindings,
    }


def validate_common_lock(common_root: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    root = common_root.expanduser().resolve()
    lock, lock_binding = load_json(root / "selection_lock.json", "common i3 selection lock")
    capacity, capacity_binding = load_json(
        root / "capacity_selection.json", "common i3 capacity selection"
    )
    policy, policy_binding = load_json(
        root / "common_fallback_policy.json", "common i3 fallback policy"
    )
    if (
        lock.get("schema_version") != 1
        or lock.get("classification") != "retrospective_i3_common_discovery_lock"
        or lock.get("outer_test_opened_before_lock") is not False
        or lock.get("selected_preset") != "default"
        or int(lock.get("selected_parameter_count", -1)) != PARAMETER_COUNT
    ):
        raise RuntimeError("common i3 lock does not freeze the default capacity")
    if (
        capacity_binding["sha256"] != lock.get("capacity_selection_sha256")
        or policy_binding["sha256"] != lock.get("common_fallback_policy_sha256")
    ):
        raise RuntimeError("common i3 lock artifact hash mismatch")
    if (
        capacity.get("outer_test_opened") is not False
        or capacity.get("selected_preset") != "default"
        or int(capacity.get("selected_parameter_count", -1)) != PARAMETER_COUNT
        or capacity.get("source_freeze_manifest_sha256") != freeze["manifest"]["sha256"]
    ):
        raise RuntimeError("common capacity selection differs from the frozen default")
    if (
        policy.get("outer_test_opened") is not False
        or policy.get("selected_preset") != "default"
        or not isinstance(policy.get("policy"), Mapping)
        or policy.get("policy", {}).get("selection_status")
        != lock.get("policy_selection_status")
    ):
        raise RuntimeError("common fallback policy differs from the common lock")
    if lock.get("source_freeze") != freeze["frozen_hashes"]:
        raise RuntimeError("common i3 lock source freeze differs from final freeze")
    unit_locks = lock.get("unit_locks")
    expected_pairs = {(fold, seed) for fold in REUSED_FOLDS for seed in SEEDS}
    if not isinstance(unit_locks, list) or {
        (int(item.get("outer_fold", -1)), int(item.get("seed", -1)))
        for item in unit_locks
        if isinstance(item, Mapping)
    } != expected_pairs:
        raise RuntimeError("common i3 lock must bind exactly the six fold-3/4 discovery units")
    return {
        "selection_lock": lock_binding,
        "capacity_selection": capacity_binding,
        "policy": policy_binding,
        "selected_preset": "default",
        "selected_parameter_count": PARAMETER_COUNT,
        "policy_selection_status": str(lock["policy_selection_status"]),
        "unit_locks": {
            (int(item["outer_fold"]), int(item["seed"])): dict(item)
            for item in unit_locks
        },
    }


def fixed_trainer_flags() -> tuple[str, ...]:
    """The immutable i3-default command, excluding split/input/output paths."""

    return (
        "--device", "cuda",
        "--amp",
        "--deterministic",
        "--preset", "default",
        "--maximum-parameters", "750000",
        "--epochs", "120",
        "--minimum-epochs", "20",
        "--patience", "18",
        "--learning-rate", "0.0003",
        "--adaptive-iteration", "3",
        "--anchor-residual-mode", "causal_posterior",
        "--anchor-max-residual-bpm", "12",
        "--anchor-minimum-scale-bpm", "0.25",
        "--anchor-maximum-scale-bpm", "12",
        "--anchor-initial-scale-bpm", "1.5",
        "--anchor-distance-weight", "1.0",
        "--anchor-source-mode", "learned_blend",
        "--anchor-residual-weight", "0.75",
        "--anchor-nll-weight", "0.20",
        "--anchor-gate-weight", "0.08",
        "--tail-weight", "2.0",
        "--cvar-weight", "0.15",
        "--warmup-windows", "2",
        "--gradient-accumulation-sessions", "4",
        "--chunk-windows", "32",
        "--maximum-coverage", "0.20",
        "--maximum-fpr", "0.01",
        "--minimum-precision", "0.80",
        "--minimum-correction-recall", "0.20",
        "--discovery-only",
    )


def fixed_cache_flags() -> tuple[str, ...]:
    return (
        "--merge-radius-bpm", "0.5",
        "--proposal-selection", "posterior-nms",
        "--posterior-nms-suppression-bpm", "1.25",
        "--base-proposals", "expected-map",
        "--svd-components", "12",
        "--proposer-features",
    )


def _assert_no_test_command(argv: Sequence[str]) -> None:
    for argument in argv:
        lowered = str(argument).lower()
        name = Path(lowered).name
        if lowered.startswith("--test") or name.startswith("test_pred_"):
            raise RuntimeError("refusing a command containing an outer-test argument")
    if "--discovery-only" in argv:
        flags = tuple(argv[-len(fixed_trainer_flags()):])
        recovery_flags = tuple(argv[-len(fixed_trainer_flags()) - 1:-1])
        exact_training = flags == fixed_trainer_flags()
        exact_recovery = recovery_flags == fixed_trainer_flags() and argv[-1] == "--recover-prelock"
        if not exact_training and not exact_recovery:
            raise RuntimeError("trainer command is not frozen i3-default training/recovery")


def trainer_command(
    python_executable: Path,
    trainer: Path,
    *,
    cache: Path,
    fallback: Path,
    output: Path,
    fold: int,
    seed: int,
    recover_prelock: bool = False,
) -> list[str]:
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("new training is restricted to the fixed 6-fold/3-seed matrix")
    command = [
        str(python_executable),
        str(trainer),
        "--cache", str(cache),
        "--fallback-oof", str(fallback),
        "--output-dir", str(output),
        "--fold", str(fold),
        "--seed", str(seed),
        *fixed_trainer_flags(),
    ]
    if recover_prelock:
        command.append("--recover-prelock")
    _assert_no_test_command(command)
    return command


def admitted_command(
    python_executable: Path,
    wrapper: Path,
    *,
    lock_file: Path,
    ledger: Path,
    trainer_argv: Sequence[str],
) -> list[str]:
    command = [
        str(python_executable), str(wrapper),
        "--lock-file", str(lock_file),
        "--ledger", str(ledger),
        "--", *map(str, trainer_argv),
    ]
    _assert_no_test_command(command)
    return command


def _directory_files(path: Path) -> dict[str, Any]:
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    return {
        "path": str(path.resolve()),
        "files": {
            str(candidate.relative_to(path)): {
                "sha256": sha256_file(candidate),
                "bytes": candidate.stat().st_size,
            }
            for candidate in files
        },
        "tree_sha256": semantic_sha256(
            [
                [str(candidate.relative_to(path)), sha256_file(candidate), candidate.stat().st_size]
                for candidate in files
            ]
        ),
    }


def _assert_no_test_artifacts(path: Path) -> None:
    if not path.exists():
        return
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        lowered = candidate.name.lower()
        if lowered in FORBIDDEN_OUTPUT_NAMES or lowered.startswith("test_pred_"):
            raise RuntimeError(f"outer-test artifact present in pre-test scope: {candidate}")


def _validate_effective_i3_configuration(manifest: Mapping[str, Any]) -> None:
    effective = manifest.get("iteration_effective_configuration")
    if not isinstance(effective, Mapping):
        raise RuntimeError("i3 run manifest lacks effective configuration")
    if semantic_sha256(effective) != manifest.get("iteration_effective_configuration_sha256"):
        raise RuntimeError("i3 run effective configuration hash mismatch")
    model = effective.get("model", {})
    capacity = effective.get("model_capacity", {})
    optimization = effective.get("optimization", {})
    objective = effective.get("iteration_objective", {})
    required_model = {
        "hidden_channels": 64,
        "graph_blocks": 2,
        "attention_heads": 4,
        "anchor_enabled": True,
        "anchor_max_residual_bpm": 12.0,
        "anchor_minimum_scale_bpm": 0.25,
        "anchor_maximum_scale_bpm": 12.0,
        "anchor_initial_scale_bpm": 1.5,
        "anchor_distance_weight": 1.0,
        "anchor_source_mode": "learned_blend",
    }
    if any(model.get(key) != value for key, value in required_model.items()):
        raise RuntimeError("i3 run is not the frozen default posterior-anchor architecture")
    if (
        int(capacity.get("maximum_parameters", -1)) != MAXIMUM_PARAMETERS
        or int(capacity.get("parameter_count", -1)) != PARAMETER_COUNT
        or capacity.get("hard_limit_enforced_before_training") is not True
    ):
        raise RuntimeError("i3 run capacity differs from the frozen default")
    expected_optimization = {
        "amp": True,
        "chunk_windows": 32,
        "epochs": 120,
        "learning_rate": 0.0003,
        "minimum_epochs": 20,
        "patience": 18,
    }
    if any(optimization.get(key) != value for key, value in expected_optimization.items()):
        raise RuntimeError("i3 run optimization differs from the frozen command")
    expected_objective = {
        "iteration": 3,
        "campaign_id": "v2_i3_causal_posterior_anchor",
        "anchor_residual_weight": 0.75,
        "anchor_nll_weight": 0.2,
        "anchor_gate_weight": 0.08,
        "tail_weight": 2.0,
        "cvar_weight": 0.15,
        "warmup_windows": 2,
        "gradient_accumulation_sessions": 4,
    }
    if any(objective.get(key) != value for key, value in expected_objective.items()):
        raise RuntimeError("i3 run objective differs from the frozen command")


def validate_training_output(
    output: Path,
    *,
    fold: int,
    seed: int,
    freeze: Mapping[str, Any],
    common: Mapping[str, Any],
    group: Mapping[str, Any],
    expected_cache: Path | None = None,
    expected_fallback: Path | None = None,
    reused: bool,
) -> dict[str, Any]:
    root = output.expanduser().resolve()
    _assert_no_test_artifacts(root)
    missing = [name for name in REQUIRED_TRAINING_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete fixed i3 output {root}: {missing}")
    lock, lock_binding = load_json(root / "selection_lock.json", "i3 unit selection lock")
    manifest, manifest_binding = load_json(root / "run_manifest.json", "i3 unit run manifest")
    if (
        lock.get("schema_version") != 1
        or lock.get("outer_test_not_opened_before_this_lock") is not True
        or int(lock.get("outer_fold", -1)) != fold
        or int(lock.get("seed", -1)) != seed
        or int(lock.get("adaptive_iteration", -1)) != 3
    ):
        raise RuntimeError("i3 unit lock identity/test-seal mismatch")
    for key, filename in (
        ("checkpoint_sha256", "best_checkpoint.pt"),
        ("history_sha256", "history.json"),
        ("run_manifest_sha256", "run_manifest.json"),
        ("scaler_sha256", "scaler.json"),
        ("policy_sha256", "fallback_policy.json"),
    ):
        if lock.get(key) != sha256_file(root / filename):
            raise RuntimeError(f"i3 unit lock artifact hash mismatch: {root / filename}")
    sources = lock.get("source_bindings")
    source_names = {
        "trainer": "train_harmonic_set_snn.py",
        "harmonic_set_model": "harmonic_set_models.py",
        "campaign_config": "harmonic_set_v2.yaml",
        "adaptive_campaign_contract": "ADAPTIVE_CAMPAIGN_CONTRACT.json",
    }
    if not isinstance(sources, Mapping):
        raise RuntimeError("i3 unit lock lacks source bindings")
    for source_key, freeze_name in source_names.items():
        binding = sources.get(source_key)
        if not isinstance(binding, Mapping) or binding.get("sha256") != freeze["frozen_hashes"][freeze_name]:
            raise RuntimeError(f"i3 unit source differs from freeze: {source_key}")
    _validate_effective_i3_configuration(manifest)
    if (
        int(manifest.get("outer_fold", -1)) != fold
        or int(manifest.get("validation_fold", -1)) != (fold + 1) % 6
        or int(manifest.get("optimization", {}).get("seed", -1)) != seed
    ):
        raise RuntimeError("i3 run manifest split/seed mismatch")
    inputs = manifest.get("input_bindings")
    if not isinstance(inputs, Mapping):
        raise RuntimeError("i3 run manifest lacks input bindings")
    cache_manifest_path = _resolve(inputs.get("cache_manifest_path"), PROJECT_ROOT)
    fallback_path = _resolve(inputs.get("fallback_oof_path"), PROJECT_ROOT)
    if expected_cache is not None and cache_manifest_path != expected_cache.resolve() / "manifest.json":
        raise RuntimeError("i3 run used another harmonic cache")
    if expected_fallback is not None and fallback_path != expected_fallback.resolve():
        raise RuntimeError("i3 run used another strict fallback")
    if (
        not cache_manifest_path.is_file()
        or sha256_file(cache_manifest_path) != lock.get("cache_manifest_sha256")
        or not fallback_path.is_file()
        or sha256_file(fallback_path) != lock.get("fallback_oof_sha256")
    ):
        raise RuntimeError("i3 run input hash mismatch")
    cache_root = cache_manifest_path.parent
    cache_manifest, _ = load_json(cache_manifest_path, "i3 harmonic cache manifest")
    proposer = cache_manifest.get("inputs", {}).get("proposer")
    if not isinstance(proposer, Mapping):
        raise RuntimeError("i3 cache lacks strict proposer binding")
    stack_path = _resolve(proposer.get("path"), PROJECT_ROOT)
    stack = discovery.validate_stack(stack_path, outer_fold=fold, seed=seed, group=group)
    cache = discovery.validate_cache(
        cache_root,
        stack=stack,
        flavor="i2r_posterior_nms125_svd12_merge050",
    )
    fallback = discovery.validate_fallback(
        fallback_path,
        stack=stack,
        outer_fold=fold,
        seed=seed,
    )
    if cache["manifest"]["sha256"] != lock.get("cache_manifest_sha256"):
        raise RuntimeError("validated cache binding differs from i3 lock")
    if fallback["sha256"] != lock.get("fallback_oof_sha256"):
        raise RuntimeError("validated fallback binding differs from i3 lock")
    if reused:
        expected_lock = common["unit_locks"].get((fold, seed))
        if expected_lock is None or expected_lock.get("selection_lock_sha256") != lock_binding["sha256"]:
            raise RuntimeError("reused discovery unit differs from common selection lock")
        if expected_lock.get("validation_predictions_sha256") != sha256_file(
            root / "validation_predictions.npz"
        ):
            raise RuntimeError("reused validation predictions differ from common lock")

    artifacts = {
        "selection_lock": lock_binding,
        "checkpoint": bind_file(root / "best_checkpoint.pt"),
        "scaler": bind_file(root / "scaler.json"),
        "cache_manifest": bind_file(cache_manifest_path),
        "fallback_oof": bind_file(fallback_path),
        "fallback_provenance": bind_file(
            fallback_path.with_name(f"{fallback_path.name}.provenance.json")
        ),
        "run_manifest": manifest_binding,
        "original_policy": bind_file(root / "fallback_policy.json"),
        "history": bind_file(root / "history.json"),
        "validation_metrics": bind_file(root / "validation_metrics.json"),
        "validation_predictions": bind_file(root / "validation_predictions.npz"),
        "strict_stack": bind_file(stack_path),
    }
    return {
        "outer_fold": fold,
        "seed": seed,
        "status": "complete",
        "reused_discovery_unit": bool(reused),
        "output_root": str(root),
        "artifacts": artifacts,
        "cache_root": {
            "path": str(cache_root),
            "manifest_sha256": artifacts["cache_manifest"]["sha256"],
            "build_signature_sha256": cache["build_signature_sha256"],
        },
        "original_policy_role": "trainer_intrinsic_validation_diagnostic_ignored_post_lock",
        "external_common_policy": common["policy"],
        "outer_test_opened": False,
        "output_tree": _directory_files(root),
    }


def _stage_paths(args: argparse.Namespace, fold: int, seed: int) -> dict[str, Path]:
    preprocessing = args.artifact_root / "preprocessing" / f"outer_{fold}_seed_{seed}"
    return {
        "stack": preprocessing / "strict_stack.npz",
        "fallback": preprocessing / "strict_fallback.csv",
        "cache": args.cache_root / f"outer_{fold}_seed_{seed}",
        "training": args.artifact_root / "units" / f"outer_{fold}_seed_{seed}",
    }


def _commands_for_unit(
    args: argparse.Namespace,
    fold: int,
    seed: int,
    *,
    attempt: int = 0,
    recover_prelock: bool = False,
) -> dict[str, list[str]]:
    paths = _stage_paths(args, fold, seed)
    training_attempt = paths["training"] / f"attempt_{attempt:03d}"
    stack = [
        str(args.python_executable), str(args.stack_builder),
        "--discovery-index", str(args.index),
        "--plan", str(args.plan),
        "--cache-dir", str(args.rf_cache),
        "--outer-fold", str(fold),
        "--seed", str(seed),
        "--output", str(paths["stack"]),
    ]
    fallback = [
        str(args.python_executable), str(args.fallback_builder),
        "--stack", str(paths["stack"]),
        "--output", str(paths["fallback"]),
    ]
    cache = [
        str(args.python_executable), str(args.cache_builder),
        "--rf-cache", str(args.rf_cache),
        "--svd-cache", str(args.svd_cache),
        "--proposer", str(paths["stack"]),
        "--fold-assignments", str(args.fold_assignments),
        "--output-dir", str(paths["cache"]),
        "--batch-size", "8",
        *fixed_cache_flags(),
    ]
    trainer = trainer_command(
        args.python_executable,
        args.trainer,
        cache=paths["cache"],
        fallback=paths["fallback"],
        output=training_attempt,
        fold=fold,
        seed=seed,
        recover_prelock=recover_prelock,
    )
    admitted = admitted_command(
        args.python_executable,
        args.gpu_wrapper,
        lock_file=args.gpu_lock,
        ledger=args.gpu_ledger,
        trainer_argv=trainer,
    )
    commands = {"stack": stack, "fallback": fallback, "cache": cache, "trainer": trainer, "admitted": admitted}
    for command in commands.values():
        _assert_no_test_command(command)
    return commands


def choose_training_attempt(training_base: Path) -> dict[str, Any]:
    """Choose a non-destructive completion action for one logical unit.

    Locked attempts are never replaced.  A completed/prelock attempt is always
    preferred to starting another training job.  Genuinely interrupted
    training is retained byte-for-byte and advances to the next version.
    """

    base = training_base.expanduser().resolve()
    attempts: list[Path] = []
    if base.is_dir():
        for path in base.iterdir():
            if not path.name.startswith("attempt_"):
                raise RuntimeError(f"unexpected entry in fixed-i3 training root: {path}")
            suffix = path.name.removeprefix("attempt_")
            if not path.is_dir() or len(suffix) < 3 or not suffix.isdigit():
                raise RuntimeError(f"invalid fixed-i3 attempt entry: {path}")
            attempts.append(path)
        attempts.sort()
    parsed: list[tuple[int, Path]] = []
    for path in attempts:
        try:
            number = int(path.name.removeprefix("attempt_"))
        except ValueError as exc:
            raise RuntimeError(f"invalid fixed-i3 attempt directory: {path}") from exc
        _assert_no_test_artifacts(path)
        parsed.append((number, path))

    # A lock is one-way.  Complete locked attempts win; incomplete locked
    # publication is repaired in place by the trainer's validation-only path.
    for number, path in parsed:
        if (path / "selection_lock.json").is_file() and all(
            (path / name).is_file() for name in REQUIRED_TRAINING_FILES
        ):
            return {
                "action": "validate_complete",
                "attempt": number,
                "path": path,
                "locked": True,
                "missing": [],
                "preserved_attempts": [candidate for _, candidate in parsed],
            }
    for number, path in parsed:
        if (path / "selection_lock.json").is_file():
            missing = [name for name in REQUIRED_TRAINING_FILES if not (path / name).is_file()]
            return {
                "action": "recover_prelock",
                "attempt": number,
                "path": path,
                "locked": True,
                "missing": missing,
                "preserved_attempts": [candidate for _, candidate in parsed],
            }

    prelock_required = (
        "best_checkpoint.pt", "history.json", "run_manifest.json", "scaler.json"
    )
    for number, path in parsed:
        if all((path / name).is_file() for name in prelock_required):
            structurally_valid = True
            try:
                history = json.loads((path / "history.json").read_text(encoding="utf-8"))
                manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
                scaler = json.loads((path / "scaler.json").read_text(encoding="utf-8"))
                structurally_valid = (
                    isinstance(history, list)
                    and bool(history)
                    and history_indicates_training_complete(history)
                    and isinstance(manifest, dict)
                    and isinstance(scaler, dict)
                    and zipfile.is_zipfile(path / "best_checkpoint.pt")
                )
            except (OSError, json.JSONDecodeError):
                structurally_valid = False
            if not structurally_valid:
                # A kill can leave a filename before its final atomic payload
                # is visible.  It is not a recoverable completed-training
                # publication and must remain preserved as a partial attempt.
                continue
            ambiguous = [
                name
                for name in (
                    "fallback_policy.json", "validation_predictions.npz",
                    "validation_metrics.json", "test_predictions.npz", "test_metrics.json",
                )
                if (path / name).exists()
            ]
            if ambiguous:
                raise RuntimeError(
                    f"prelock attempt has ambiguous publication artifacts: {path} {ambiguous}"
                )
            return {
                "action": "recover_prelock",
                "attempt": number,
                "path": path,
                "locked": False,
                "missing": [],
                "preserved_attempts": [candidate for _, candidate in parsed],
            }

    next_number = (max((number for number, _ in parsed), default=-1) + 1)
    return {
        "action": "train_fresh",
        "attempt": next_number,
        "path": base / f"attempt_{next_number:03d}",
        "locked": False,
        "missing": [],
        "preserved_attempts": [candidate for _, candidate in parsed],
    }


def history_indicates_training_complete(history: Sequence[Any]) -> bool:
    """Reproduce the frozen trainer's 120/20/18 termination condition."""

    if not history:
        return False
    best: tuple[Any, ...] | None = None
    stale = 0
    for expected_epoch, item in enumerate(history, start=1):
        if not isinstance(item, Mapping) or int(item.get("epoch", -1)) != expected_epoch:
            return False
        raw_key = item.get("retrospective_selection_key")
        if not isinstance(raw_key, (list, tuple)) or not raw_key:
            return False
        try:
            key = tuple(float(value) for value in raw_key)
        except (TypeError, ValueError):
            return False
        if not all(value == value and abs(value) != float("inf") for value in key):
            return False
        if best is None or key < best:
            best = key
            stale = 0
        else:
            stale += 1
    epochs = len(history)
    return epochs >= 120 or (epochs >= 20 and stale >= 18)


def attempt_history_bindings(
    paths: Sequence[Path], *, completed_path: Path | None = None
) -> list[dict[str, Any]]:
    completed = completed_path.resolve() if completed_path is not None else None
    result: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        if completed is not None and resolved == completed:
            continue
        result.append(
            {
                "attempt": int(resolved.name.removeprefix("attempt_")),
                "status": "preserved_incomplete",
                "output_tree": _directory_files(resolved),
            }
        )
    return result


def _command_record(stage: str, argv: Sequence[str]) -> dict[str, Any]:
    normalized = list(map(str, argv))
    return {
        "stage": stage,
        "argv": normalized,
        "shell_rendering": shlex.join(normalized),
        "command_identity_sha256": semantic_sha256({"stage": stage, "argv": normalized}),
    }


def _execute(argv: Sequence[str]) -> None:
    _assert_no_test_command(argv)
    completed = subprocess.run(list(map(str, argv)), cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"fixed pre-test command failed with status {completed.returncode}: {shlex.join(argv)}"
        )


def _build_static_plan(
    args: argparse.Namespace,
    *,
    inputs: Mapping[str, Any],
    sources: Mapping[str, Any],
    common: Mapping[str, Any],
) -> dict[str, Any]:
    units: list[dict[str, Any]] = []
    forced = args.force_retrain_units
    for fold in FOLDS:
        for seed in SEEDS:
            if fold in REUSED_FOLDS and (fold, seed) not in forced:
                output = args.reuse_root / f"outer_{fold}_seed_{seed}"
                units.append(
                    {
                        "outer_fold": fold,
                        "seed": seed,
                        "mode": "reuse_common_locked_discovery_unit",
                        "output_root": str(output),
                    }
                )
            else:
                paths = _stage_paths(args, fold, seed)
                commands = _commands_for_unit(args, fold, seed)
                units.append(
                    {
                        "outer_fold": fold,
                        "seed": seed,
                        "mode": "fixed_i3_default_training",
                        "paths": {name: str(path) for name, path in paths.items()},
                        "attempt_policy": {
                            "directory_pattern": "attempt_NNN",
                            "initial_attempt": 0,
                            "partial_attempts_preserved": True,
                            "prelock_publication_recovery": "--recover-prelock",
                            "locked_attempt_overwrite_permitted": False,
                            "next_attempt_rule": "one_plus_max_existing_attempt",
                        },
                        "initial_attempt_commands": {
                            name: _command_record(name, command)
                            for name, command in commands.items()
                        },
                        "recovery_command_derivation": (
                            "same frozen i3 flags and same attempt output; append only --recover-prelock"
                        ),
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "retrospective_fixed_i3_pretest_completion_plan",
        "matrix": {"folds": list(FOLDS), "seeds": list(SEEDS), "unit_count": 18},
        "reuse": {
            "folds": sorted(REUSED_FOLDS),
            "unit_count": 6 - len(forced),
            "excluded_provenance_replacement_units": [
                {"outer_fold": fold, "seed": seed}
                for fold, seed in sorted(forced)
            ],
        },
        "new_training": {
            "folds": sorted(MISSING_FOLDS),
            "unit_count": 12 + len(forced),
            "forced_common_provenance_replacements": [
                {"outer_fold": fold, "seed": seed}
                for fold, seed in sorted(forced)
            ],
        },
        "selected_preset": "default",
        "selected_parameter_count": PARAMETER_COUNT,
        "trainer_flags": list(fixed_trainer_flags()),
        "trainer_flag_identity_sha256": semantic_sha256(fixed_trainer_flags()),
        "validation_scores_control_execution": False,
        "capacity_reselection_permitted": False,
        "common_policy_reselection_permitted": False,
        "per_unit_original_policy_role": "diagnostic_ignored_post_lock",
        "outer_test_manifest_argument_supported": False,
        "outer_test_opened": False,
        "inputs": dict(inputs),
        "sources": dict(sources),
        "common_lock": {
            key: common[key]
            for key in ("selection_lock", "capacity_selection", "policy")
        },
        "units": units,
    }


def _bind_effective_sources(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "orchestrator": bind_file(Path(__file__)),
        "runtime_seal_verifier": bind_file(
            PROJECT_ROOT / "scripts/seal_runtime_inputs.py"
        ),
        "discovery_validation_helper": bind_file(PROJECT_ROOT / "scripts/run_hcs_discovery_campaign.py"),
        "stack_builder": bind_file(args.stack_builder),
        "fallback_builder": bind_file(args.fallback_builder),
        "cache_builder": bind_file(args.cache_builder),
        "trainer": bind_file(args.trainer),
        "gpu_wrapper": bind_file(args.gpu_wrapper),
        "python_executable": bind_file(args.python_executable),
    }


def _validate_static_inputs(args: argparse.Namespace) -> dict[str, Any]:
    rf_manifest = bind_file(args.rf_cache / "manifest.json")
    svd_manifest = bind_file(args.svd_cache / "manifest.json")
    folds = bind_file(args.fold_assignments)
    plan, _ = load_json(args.plan, "six-fold non-test proposer plan")
    cache_binding = plan.get("cache_manifest")
    fold_binding = plan.get("fold_assignments")
    if not isinstance(cache_binding, Mapping) or rf_manifest["sha256"] != cache_binding.get("sha256"):
        raise RuntimeError("RF cache manifest differs from the six-fold non-test plan")
    if not isinstance(fold_binding, Mapping) or folds["sha256"] != fold_binding.get("sha256"):
        raise RuntimeError("fold assignments differ from the six-fold non-test plan")
    seal_result = runtime_seal.verify(args.runtime_seal)
    seal_document, seal_binding = load_json(
        args.runtime_seal, "fixed-i3 prelaunch runtime seal"
    )
    if (
        seal_document.get("post_launch_attestation") is not False
        or seal_document.get("attestation_phase") != "prelaunch"
    ):
        raise RuntimeError("fixed-i3 runtime seal must be a prelaunch attestation")
    return {
        "rf_cache_manifest": rf_manifest,
        "svd_cache_manifest": svd_manifest,
        "fold_assignments": folds,
        "runtime_input_seal": {
            **seal_binding,
            "content_sha256": seal_document.get("content_sha256"),
            "verified_files": int(seal_result["verified_files"]),
            "attestation_phase": "prelaunch",
        },
    }


def _assert_previous_plan(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    planned, payload = content_document(document)
    path = root / "campaign_lock.json"
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError("fixed pre-test campaign lock/input/command identity changed")
    exclusive_write(path, payload)
    return planned


def _prior_completed_units(root: Path) -> dict[tuple[int, int], Mapping[str, Any]]:
    path = root / "pretest_status.json"
    if not path.exists():
        return {}
    _validate_current_document(path, root / "pretest_status_snapshots", "pretest_status")
    value, _ = load_json(path, "pretest status")
    units = value.get("units", [])
    if not isinstance(units, list):
        raise RuntimeError("pretest status units are invalid")
    return {
        (int(unit["outer_fold"]), int(unit["seed"])): unit
        for unit in units
        if isinstance(unit, Mapping) and unit.get("status") == "complete"
    }


def _compare_prior(current: Mapping[str, Any], prior: Mapping[str, Any] | None) -> None:
    if prior is None:
        return
    if prior.get("output_tree") != current.get("output_tree") or prior.get("artifacts") != current.get("artifacts"):
        raise RuntimeError(
            f"completed pre-test unit was tampered: outer={current['outer_fold']} seed={current['seed']}"
        )


def _publish_status(
    args: argparse.Namespace,
    *,
    campaign_lock: Mapping[str, Any],
    common: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    completed = sum(unit.get("status") == "complete" for unit in units)
    value = {
        "schema_version": SCHEMA_VERSION,
        "classification": "retrospective_fixed_i3_pretest_status",
        "campaign_lock_sha256": campaign_lock["content_sha256"],
        "common_selection_lock": common["selection_lock"],
        "matrix_unit_count": 18,
        "completed_units": completed,
        "status": "complete" if completed == 18 else "in_progress",
        "outer_test_opened": False,
        "units": list(units),
    }
    return publish_current(args.artifact_root, "pretest_status.json", value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--common-root", type=Path, default=DEFAULT_COMMON_ROOT)
    parser.add_argument("--freeze-root", type=Path, default=DEFAULT_FREEZE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--reuse-root", type=Path, default=DEFAULT_REUSE_ROOT)
    parser.add_argument("--rf-cache", type=Path, default=DEFAULT_RF_CACHE)
    parser.add_argument("--svd-cache", type=Path, default=DEFAULT_SVD_CACHE)
    parser.add_argument("--fold-assignments", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--runtime-seal",
        type=Path,
        default=DEFAULT_RUNTIME_SEAL,
        help="prelaunch source/cache payload seal reverified at every unit boundary",
    )
    parser.add_argument(
        "--retrain-impact-audit",
        type=Path,
        default=DEFAULT_RETRAIN_IMPACT_AUDIT,
        help="sealed byte-drift audit that determines common-unit rebuilds",
    )
    parser.add_argument("--stack-builder", type=Path, default=PROJECT_ROOT / "scripts/build_nested_proposer_stack.py")
    parser.add_argument("--fallback-builder", type=Path, default=PROJECT_ROOT / "scripts/build_nested_fallback_oof.py")
    parser.add_argument("--cache-builder", type=Path, default=PROJECT_ROOT / "scripts/build_harmonic_set_cache.py")
    parser.add_argument("--trainer", type=Path, default=PROJECT_ROOT / "scripts/train_harmonic_set_snn.py")
    parser.add_argument("--gpu-wrapper", type=Path, default=PROJECT_ROOT / "scripts/run_gpu_admitted.py")
    parser.add_argument("--gpu-lock", type=Path, default=RUN_ROOT / "gpu_admission.lock")
    parser.add_argument("--gpu-ledger", type=Path, default=RUN_ROOT / "gpu_execution_ledger.jsonl")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--force-retrain-units",
        default="auto",
        help=(
            "'auto' (default) or comma-separated OUTER_FOLD:SEED pairs; any "
            "explicit value must equal the sealed retrain impact audit"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    path_names = (
        "plan", "index", "common_root", "freeze_root", "artifact_root", "cache_root",
        "reuse_root", "rf_cache", "svd_cache", "fold_assignments", "stack_builder",
        "fallback_builder", "cache_builder", "trainer", "gpu_wrapper", "gpu_lock",
        "gpu_ledger",
        "runtime_seal",
        "retrain_impact_audit",
    )
    for name in path_names:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    # Preserve venv invocation semantics while still using a stable absolute path.
    args.python_executable = args.python_executable.expanduser().absolute()
    raw_force = str(args.force_retrain_units).strip()
    args.force_retrain_units = (
        None
        if raw_force.lower() == "auto"
        else parse_force_retrain_units(raw_force)
    )
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    groups, proposer_inputs = validate_non_test_plan_index(args.plan, args.index)
    audited_force_units, retrain_impact_binding = validate_retrain_impact_audit(
        args.retrain_impact_audit,
        args.index,
        args.force_retrain_units,
    )
    args.force_retrain_units = audited_force_units
    freeze = validate_freeze(args.freeze_root)
    common = validate_common_lock(args.common_root, freeze)
    sources = _bind_effective_sources(args)
    if sources["trainer"]["sha256"] != freeze["frozen_hashes"]["train_harmonic_set_snn.py"]:
        raise RuntimeError("effective trainer differs from frozen i3 trainer")
    if sources["gpu_wrapper"]["sha256"] != freeze["frozen_hashes"]["run_gpu_admitted.py"]:
        raise RuntimeError("effective GPU wrapper differs from frozen i3 wrapper")
    static_inputs = _validate_static_inputs(args)
    inputs = {
        **proposer_inputs,
        **static_inputs,
        "freeze": freeze["manifest"],
        "retrain_impact_audit": retrain_impact_binding,
    }
    static_plan = _build_static_plan(args, inputs=inputs, sources=sources, common=common)
    expected_runtime_seal = static_inputs["runtime_input_seal"]["content_sha256"]

    def reverify_runtime_inputs() -> None:
        observed = runtime_seal.verify(args.runtime_seal)
        if observed.get("content_sha256") != expected_runtime_seal:
            raise RuntimeError("fixed-i3 runtime seal identity changed during campaign")

    # Dry-run is deliberately read-only, but it still validates every ready
    # proposer record and every reused unit before returning the exact commands.
    reused_units: dict[tuple[int, int], dict[str, Any]] = {}
    for fold in sorted(REUSED_FOLDS):
        for seed in SEEDS:
            if (fold, seed) in args.force_retrain_units:
                continue
            output = args.reuse_root / f"outer_{fold}_seed_{seed}"
            reused_units[(fold, seed)] = validate_training_output(
                output,
                fold=fold,
                seed=seed,
                freeze=freeze,
                common=common,
                group=groups[(fold, seed)],
                reused=True,
            )
    if args.dry_run:
        document, _ = content_document(
            {
                **static_plan,
                "dry_run": True,
                "validated_reused_units": [
                    {"outer_fold": fold, "seed": seed, "selection_lock": unit["artifacts"]["selection_lock"]}
                    for (fold, seed), unit in sorted(reused_units.items())
                ],
            }
        )
        return document

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    campaign_lock = _assert_previous_plan(args.artifact_root, static_plan)
    prior = _prior_completed_units(args.artifact_root)
    completed: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            reverify_runtime_inputs()
            if fold in REUSED_FOLDS and (
                fold, seed
            ) not in args.force_retrain_units:
                unit = reused_units[(fold, seed)]
            else:
                paths = _stage_paths(args, fold, seed)
                preprocessing_commands = _commands_for_unit(args, fold, seed)
                stack_path = paths["stack"]
                if stack_path.exists():
                    stack = discovery.validate_stack(
                        stack_path, outer_fold=fold, seed=seed, group=groups[(fold, seed)]
                    )
                else:
                    _execute(preprocessing_commands["stack"])
                    stack = discovery.validate_stack(
                        stack_path, outer_fold=fold, seed=seed, group=groups[(fold, seed)]
                    )
                fallback_path = paths["fallback"]
                fallback_sidecar = fallback_path.with_name(f"{fallback_path.name}.provenance.json")
                if fallback_path.exists() or fallback_sidecar.exists():
                    fallback = discovery.validate_fallback(
                        fallback_path, stack=stack, outer_fold=fold, seed=seed
                    )
                else:
                    _execute(preprocessing_commands["fallback"])
                    fallback = discovery.validate_fallback(
                        fallback_path, stack=stack, outer_fold=fold, seed=seed
                    )
                cache_path = paths["cache"]
                if cache_path.exists():
                    cache = discovery.validate_cache(
                        cache_path,
                        stack=stack,
                        flavor="i2r_posterior_nms125_svd12_merge050",
                    )
                else:
                    _execute(preprocessing_commands["cache"])
                    cache = discovery.validate_cache(
                        cache_path,
                        stack=stack,
                        flavor="i2r_posterior_nms125_svd12_merge050",
                    )
                if cache["manifest"]["sha256"] == "" or fallback["sha256"] == "":
                    raise RuntimeError("pre-test preprocessing produced an empty binding")
                choice = choose_training_attempt(paths["training"])
                training_path = Path(choice["path"])
                recovery = choice["action"] == "recover_prelock"
                commands = _commands_for_unit(
                    args,
                    fold,
                    seed,
                    attempt=int(choice["attempt"]),
                    recover_prelock=recovery,
                )
                if choice["action"] != "validate_complete":
                    pending_unit = {
                        "outer_fold": fold,
                        "seed": seed,
                        "status": "recovering_prelock" if recovery else "training",
                        "attempt": int(choice["attempt"]),
                        "attempt_path": str(training_path),
                        "locked_attempt": bool(choice["locked"]),
                        "action": str(choice["action"]),
                        "preserved_attempts": attempt_history_bindings(
                            choice["preserved_attempts"]
                        ),
                        "command": _command_record(
                            "recover_prelock" if recovery else "train_fresh",
                            commands["admitted"],
                        ),
                        "outer_test_opened": False,
                    }
                    _publish_status(
                        args,
                        campaign_lock=campaign_lock,
                        common=common,
                        units=[*completed, pending_unit],
                    )
                    _execute(commands["admitted"])
                unit = validate_training_output(
                    training_path,
                    fold=fold,
                    seed=seed,
                    freeze=freeze,
                    common=common,
                    group=groups[(fold, seed)],
                    expected_cache=cache_path,
                    expected_fallback=fallback_path,
                    reused=False,
                )
                unit["training_attempt"] = int(choice["attempt"])
                unit["training_attempt_path"] = str(training_path)
                unit["completion_action"] = str(choice["action"])
                unit["preserved_incomplete_attempts"] = attempt_history_bindings(
                    choice["preserved_attempts"], completed_path=training_path
                )
                if recovery:
                    unit["publication_recovery_command"] = _command_record(
                        "recover_prelock", commands["trainer"]
                    )
                    unit["gpu_admitted_publication_recovery_command"] = _command_record(
                        "admitted_recover_prelock", commands["admitted"]
                    )
                else:
                    unit["trainer_command"] = _command_record("trainer", commands["trainer"])
                    unit["gpu_admitted_command"] = _command_record("admitted", commands["admitted"])
            reverify_runtime_inputs()
            _compare_prior(unit, prior.get((fold, seed)))
            completed.append(unit)
            _publish_status(
                args,
                campaign_lock=campaign_lock,
                common=common,
                units=completed,
            )

    if len(completed) != 18:
        raise RuntimeError("fixed pre-test matrix did not complete all 18 units")
    reverify_runtime_inputs()
    _assert_no_test_artifacts(args.artifact_root)
    final_value = {
        "schema_version": SCHEMA_VERSION,
        "classification": "retrospective_fixed_i3_pretest_index",
        "status": "complete",
        "campaign_lock_sha256": campaign_lock["content_sha256"],
        "matrix": {"folds": list(FOLDS), "seeds": list(SEEDS), "unit_count": 18},
        "completed_units": 18,
        "reused_units": 6 - len(args.force_retrain_units),
        "newly_trained_units": 12 + len(args.force_retrain_units),
        "forced_common_provenance_replacements": [
            {"outer_fold": fold, "seed": seed}
            for fold, seed in sorted(args.force_retrain_units)
        ],
        "selected_preset": "default",
        "selected_parameter_count": PARAMETER_COUNT,
        "capacity_reselected": False,
        "common_policy_reselected": False,
        "validation_scores_control_execution": False,
        "common": {
            "selection_lock": common["selection_lock"],
            "capacity_selection": common["capacity_selection"],
            "policy": common["policy"],
            "source_freeze_manifest": freeze["manifest"],
        },
        "units": completed,
        "outer_test_opened": False,
        "outer_test_artifact_count": 0,
        "ready_for_separately_locked_label_free_outer_test_construction": True,
        "commercial_claim_authorized": False,
    }
    final_document, payload = content_document(final_value)
    exclusive_write(
        args.artifact_root / "pretest_index_snapshots" / f"{final_document['content_sha256']}.json",
        payload,
    )
    exclusive_write(args.artifact_root / "pretest_index.json", payload)
    if args.gpu_ledger.is_file():
        # Publish an additive final binding without mutating the immutable index.
        ledger_document, ledger_payload = content_document(
            {
                "schema_version": 1,
                "classification": "fixed_i3_pretest_gpu_ledger_binding",
                "pretest_index_sha256": final_document["content_sha256"],
                "gpu_ledger": bind_file(args.gpu_ledger),
                "outer_test_opened": False,
            }
        )
        exclusive_write(args.artifact_root / "gpu_ledger_binding.json", ledger_payload)
    # The current status is mutable while the campaign advances.  Once the
    # exact 18-unit index is published it becomes final immutable evidence.
    (args.artifact_root / "pretest_status.json").chmod(0o444)
    return final_document


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "status": result.get("status", "dry_run"),
                "content_sha256": result["content_sha256"],
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
