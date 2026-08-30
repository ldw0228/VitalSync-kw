#!/usr/bin/env python3
"""Seal one uniform 90-unit proposer index from two completed campaigns.

The full campaign remains the split authority.  Folds 0, 1, 2 and 5 are read
from its completed index, while folds 3 and 4 are replaced by the separately
retrained current-source campaign.  This program has no outer-test or target
argument and rejects test-manifest records before opening any referenced file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_full_nested_proposer_campaign as campaign  # noqa: E402
import seal_runtime_inputs as runtime_seal  # noqa: E402


CAMPAIGN_ROOT = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer"
)
CANONICAL_SUPERVISOR = PROJECT_ROOT / "scripts/run_sealed_nested_proposer_supervisor.py"
CANONICAL_RUNNER = PROJECT_ROOT / "scripts/run_full_nested_proposer_campaign.py"
CANONICAL_PYTHON_LAUNCHER = PROJECT_ROOT / ".venv/bin/python"
CANONICAL_GPU_ROOT = (
    PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2"
)
DEFAULT_FULL_PLAN = CAMPAIGN_ROOT / "full_oof_non_test/control/plan.json"
DEFAULT_MAIN_INDEX = CAMPAIGN_ROOT / "full_oof_non_test/control/index.json"
DEFAULT_RETRAIN_PLAN = (
    CAMPAIGN_ROOT / "current_source_retrain_f34/control/plan.json"
)
DEFAULT_RETRAIN_INDEX = (
    CAMPAIGN_ROOT / "current_source_retrain_f34/control/index.json"
)
DEFAULT_OUTPUT = CAMPAIGN_ROOT / "current_source_merged/index.json"

FOLDS = tuple(range(6))
SEEDS = (20260828, 20260829, 20260830)
MAIN_FOLDS = frozenset((0, 1, 2, 5))
RETRAIN_FOLDS = frozenset((3, 4))
ROLES = frozenset(("hcs_train_oof", "hcs_validation"))
PLAN_CLASSIFICATION = (
    "retrospective_fully_nested_non_test_proposer_campaign_plan"
)
INDEX_CLASSIFICATION = "retrospective_fully_nested_non_test_proposer_index"

InspectUnit = Callable[
    [Mapping[str, Any]], tuple[str, dict[str, Any] | None, str | None]
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_hash(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise RuntimeError(f"{label} must be a lowercase SHA-256")
    return digest


def _assert_no_test_marker(path: Path, label: str) -> None:
    if any(part.lower().startswith("test_pred_") for part in path.parts):
        raise RuntimeError(f"outer-test path is forbidden in {label}: {path}")


def _resolve(raw: Any, base: Path = PROJECT_ROOT) -> Path:
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    unresolved = path.expanduser()
    _assert_no_test_marker(unresolved, label)
    if unresolved.is_symlink():
        raise RuntimeError(f"symlink is forbidden for immutable {label}: {unresolved}")
    resolved = unresolved.resolve()
    _assert_no_test_marker(resolved, label)
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


def _manifest_name_for_outer(outer: int) -> set[str]:
    validation = (outer + 1) % len(FOLDS)
    training = sorted(set(FOLDS) - {outer, validation})
    return {
        *(f"inner_pred_{fold}.json" for fold in training),
        f"validation_pred_{validation}.json",
    }


def _record_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    try:
        outer = int(record.get("outer_fold", -1))
        seed = int(record.get("seed", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("proposer unit has a non-integer fold or seed") from exc
    manifest = Path(str(record.get("manifest", "")))
    _assert_no_test_marker(manifest, "proposer unit manifest")
    return outer, seed, manifest.name


def _bind_regular_file(path: Path, label: str) -> dict[str, Any]:
    unresolved = path.expanduser()
    _assert_no_test_marker(unresolved, label)
    if unresolved.is_symlink():
        raise RuntimeError(f"symlink is forbidden for {label}: {unresolved}")
    resolved = unresolved.resolve()
    _assert_no_test_marker(resolved, label)
    if not resolved.is_file():
        raise RuntimeError(f"missing {label}: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _validate_plan(
    path: Path,
    *,
    expected_folds: Sequence[int],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[int, int, str], dict[str, Any]]]:
    plan, binding = _load_json(path, label)
    folds = [int(value) for value in expected_folds]
    expected_units = len(folds) * len(SEEDS) * 5
    if (
        plan.get("schema_version") != 1
        or plan.get("classification") != PLAN_CLASSIFICATION
        or plan.get("outer_test_opened") is not False
        or int(plan.get("outer_test_record_count", -1)) != 0
        or plan.get("outer_folds") != folds
        or plan.get("seeds") != list(SEEDS)
        or plan.get("roles") != sorted(ROLES)
        or int(plan.get("requested_units", -1)) != expected_units
    ):
        raise RuntimeError(f"{label} does not declare the required non-test matrix")
    if canonical_content_sha256(plan) != plan.get("content_sha256"):
        raise RuntimeError(f"{label} content hash mismatch")
    campaign.verify_campaign_source_bindings(plan)

    manifest_root = _resolve(plan.get("manifest_root"))
    _assert_no_test_marker(manifest_root, f"{label} manifest root")
    campaign._assert_no_test_manifest_files(manifest_root)
    for binding_name in ("fold_assignments", "cache_manifest"):
        item = plan.get(binding_name)
        if not isinstance(item, Mapping):
            raise RuntimeError(f"{label} lacks {binding_name} binding")
        bound_path = _resolve(item.get("path"))
        _assert_no_test_marker(bound_path, f"{label} {binding_name}")
        expected_hash = _require_hash(item.get("sha256"), f"{label} {binding_name} hash")
        if not bound_path.is_file() or sha256_file(bound_path) != expected_hash:
            raise RuntimeError(f"{label} {binding_name} hash mismatch: {bound_path}")

    units = plan.get("units")
    if not isinstance(units, list) or len(units) != expected_units:
        raise RuntimeError(f"{label} has an incomplete unit list")
    observed: dict[tuple[int, int, str], dict[str, Any]] = {}
    unit_ids: set[str] = set()
    for raw_unit in units:
        if not isinstance(raw_unit, dict):
            raise RuntimeError(f"{label} contains a non-object unit")
        unit = raw_unit
        campaign._assert_non_test_path(
            Path(str(unit.get("manifest", ""))), f"{label} unit"
        )
        key = _record_key(unit)
        outer, seed, name = key
        if outer not in folds or seed not in SEEDS or name not in _manifest_name_for_outer(outer):
            raise RuntimeError(f"{label} unit violates the fixed split rotation: {key}")
        expected_role = (
            "hcs_validation" if name.startswith("validation_pred_") else "hcs_train_oof"
        )
        if unit.get("role") != expected_role:
            raise RuntimeError(f"{label} unit role/name mismatch: {key}")
        unit_id = str(unit.get("unit_id", ""))
        expected_unit_id = f"seed_{seed}/outer_{outer}/{Path(name).stem}"
        if unit_id != expected_unit_id or unit_id in unit_ids:
            raise RuntimeError(f"{label} has duplicate or invalid unit_id: {unit_id}")
        if key in observed:
            raise RuntimeError(f"{label} has duplicate unit: {key}")

        unresolved_manifest = Path(str(unit.get("manifest", ""))).expanduser()
        if unresolved_manifest.is_symlink():
            raise RuntimeError(
                f"symlink is forbidden for non-test manifest: {unresolved_manifest}"
            )
        manifest_path = _resolve(unresolved_manifest)
        _assert_no_test_marker(manifest_path, f"{label} manifest")
        expected_file_hash = _require_hash(
            unit.get("manifest_sha256"), f"{label} manifest file hash"
        )
        expected_content_hash = _require_hash(
            unit.get("manifest_content_sha256"), f"{label} manifest content hash"
        )
        campaign._validate_manifest(
            manifest_path,
            expected_content_hash=expected_content_hash,
            expected_file_hash=expected_file_hash,
        )
        if not manifest_path.is_relative_to(manifest_root):
            raise RuntimeError(f"{label} manifest lies outside its sealed root: {manifest_path}")
        observed[key] = unit
        unit_ids.add(unit_id)

    expected_keys = {
        (outer, seed, name)
        for outer in folds
        for seed in SEEDS
        for name in _manifest_name_for_outer(outer)
    }
    if set(observed) != expected_keys:
        raise RuntimeError(f"{label} does not form the exact expected unit cover")
    return plan, binding, observed


def _validate_index(
    path: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    planned: Mapping[tuple[int, int, str], Mapping[str, Any]],
    label: str,
    inspect_unit: InspectUnit,
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[int, int, str], dict[str, Any]]]:
    campaign._safe_existing_control_document(
        path, plan_hash=str(plan["content_sha256"]), allow_missing=False
    )
    index, binding = _load_json(path, label)
    expected_units = len(planned)
    nested_plan = index.get("campaign_plan")
    if (
        index.get("schema_version") != 1
        or index.get("classification") != INDEX_CLASSIFICATION
        or index.get("outer_test_opened") is not False
        or int(index.get("outer_test_record_count", -1)) != 0
        or int(index.get("requested_units", -1)) != expected_units
        or int(index.get("completed_units", -1)) != expected_units
        or index.get("campaign_plan_content_sha256") != plan.get("content_sha256")
        or not isinstance(nested_plan, Mapping)
        or nested_plan.get("content_sha256") != plan.get("content_sha256")
        or nested_plan.get("sha256") != plan_binding.get("sha256")
        or _resolve(nested_plan.get("path")) != plan_path.expanduser().resolve()
        or index.get("manifest_root") != plan.get("manifest_root")
        or index.get("manifest_plan_content_sha256")
        != plan.get("manifest_plan_content_sha256")
    ):
        raise RuntimeError(f"{label} is incomplete or bound to another campaign")
    if canonical_content_sha256(index) != index.get("content_sha256"):
        raise RuntimeError(f"{label} content hash mismatch")
    records = index.get("records")
    if not isinstance(records, list) or len(records) != expected_units:
        raise RuntimeError(f"{label} must contain exactly {expected_units} records")

    observed: dict[tuple[int, int, str], dict[str, Any]] = {}
    expected_order = list(planned)
    for position, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise RuntimeError(f"{label} contains a non-object record")
        record = raw_record
        key = _record_key(record)
        if key in observed:
            raise RuntimeError(f"{label} contains duplicate record: {key}")
        if position >= len(expected_order) or key != expected_order[position]:
            raise RuntimeError(f"{label} record order/cover differs from its plan: {key}")
        unit = planned.get(key)
        if unit is None:
            raise RuntimeError(f"{label} record is absent from its plan: {key}")
        if record.get("unit_id") != unit.get("unit_id") or record.get("role") != unit.get("role"):
            raise RuntimeError(f"{label} record identity differs from its plan: {key}")
        for item in (record, unit):
            for binding_name in ("checkpoint", "all_window_prediction"):
                raw_binding = item.get(binding_name)
                raw_path = (
                    raw_binding.get("path")
                    if isinstance(raw_binding, Mapping)
                    else raw_binding
                )
                unresolved_artifact = Path(str(raw_path or "")).expanduser()
                _assert_no_test_marker(
                    unresolved_artifact, f"{label} {binding_name}"
                )
                if unresolved_artifact.is_symlink():
                    raise RuntimeError(
                        f"symlink is forbidden for {label} {binding_name}: "
                        f"{unresolved_artifact}"
                    )

        state, validated_record, detail = inspect_unit(unit)
        if state != "complete" or validated_record is None:
            raise RuntimeError(
                f"{label} unit is not hash-complete: {unit['unit_id']} ({detail})"
            )
        if record != validated_record:
            raise RuntimeError(f"{label} record differs from validated artifacts: {key}")
        observed[key] = record
    if set(observed) != set(planned):
        raise RuntimeError(f"{label} record cover differs from its plan")
    return index, binding, observed


def _assert_compatible_plans(
    full: Mapping[str, Any],
    retrain: Mapping[str, Any],
    full_units: Mapping[tuple[int, int, str], Mapping[str, Any]],
    retrain_units: Mapping[tuple[int, int, str], Mapping[str, Any]],
) -> None:
    for field in (
        "source_bindings",
        "training_specification",
        "fold_assignments",
        "cache_manifest",
        "manifest_root",
        "seeds",
        "roles",
    ):
        if full.get(field) != retrain.get(field):
            raise RuntimeError(f"retrain plan differs from full split authority: {field}")
    for key, retrain_unit in retrain_units.items():
        full_unit = full_units.get(key)
        if full_unit is None:
            raise RuntimeError(f"retrain unit is absent from the full split authority: {key}")
        for field in (
            "unit_id",
            "seed",
            "outer_fold",
            "role",
            "manifest",
            "manifest_sha256",
            "manifest_content_sha256",
        ):
            if full_unit.get(field) != retrain_unit.get(field):
                raise RuntimeError(
                    f"retrain unit differs from full split authority: {key} ({field})"
                )


def _runtime_seal_content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    payload.pop("created_utc", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _bind_runtime_seal(path: Path) -> dict[str, Any]:
    seal, binding = _load_json(path, "runtime input seal")
    if (
        seal.get("schema_version") != 1
        or seal.get("classification")
        != "supplemental_runtime_input_byte_inventory"
        or _runtime_seal_content_sha256(seal) != seal.get("content_sha256")
    ):
        raise RuntimeError(f"runtime input seal content hash mismatch: {path}")
    # Only the already-created seal is read here.  Its payload files are not
    # reopened, which keeps this merge stage target/outer-test-label blind.
    serialized = canonical_json_bytes(seal).decode("utf-8").lower()
    if "test_pred_" in serialized:
        raise RuntimeError(f"runtime seal mentions an outer-test manifest: {path}")
    return {**binding, "content_sha256": str(seal["content_sha256"])}


def _collect_runtime_seals(
    *,
    full_plan_path: Path,
    retrain_plan_path: Path,
    explicit: Sequence[Path],
) -> list[dict[str, Any]]:
    full_root = full_plan_path.expanduser().resolve().parent.parent
    retrain_root = retrain_plan_path.expanduser().resolve().parent.parent
    automatic = (
        full_root / "runtime_input_seal.json",
        full_root / "prelaunch_runtime_input_seal.json",
        retrain_root / "runtime_input_seal.json",
        retrain_root / "prelaunch_runtime_input_seal.json",
    )
    candidates: list[Path] = []
    seen: set[Path] = set()
    for path in automatic:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists():
            continue
        if not resolved.is_file():
            raise RuntimeError(f"automatic runtime input seal is not a file: {resolved}")
        candidates.append(resolved)
    for path in explicit:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise RuntimeError(f"explicit runtime input seal is missing: {resolved}")
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)
    return [_bind_runtime_seal(path) for path in candidates]


def _same_complete_binding(
    left: Mapping[str, Any], right: Mapping[str, Any], *, label: str
) -> None:
    try:
        same = (
            _resolve(left.get("path")) == _resolve(right.get("path"))
            and _require_hash(left.get("sha256"), f"{label} left hash")
            == _require_hash(right.get("sha256"), f"{label} right hash")
            and int(left.get("bytes", -1)) == int(right.get("bytes", -2))
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} binding is malformed") from exc
    if not same:
        raise RuntimeError(f"{label} binding mismatch")


def _same_identity_binding(
    left: Mapping[str, Any], right: Mapping[str, Any], *, label: str
) -> None:
    """Compare path/file/content identity when an advisory ABI omits bytes."""

    if (
        _resolve(left.get("path")) != _resolve(right.get("path"))
        or _require_hash(left.get("sha256"), f"{label} left hash")
        != _require_hash(right.get("sha256"), f"{label} right hash")
        or _require_hash(left.get("content_sha256"), f"{label} left content hash")
        != _require_hash(right.get("content_sha256"), f"{label} right content hash")
    ):
        raise RuntimeError(f"{label} identity mismatch")


def _live_verify_execution_runtime_seal(
    receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehash the exact runtime seal selected by the execution receipt.

    The receipt, rather than a filename convention or the existence of an
    older seal, is the authority for which byte inventory governed training.
    """

    if not isinstance(receipt_binding, Mapping):
        raise RuntimeError("execution attestation lacks its runtime seal binding")
    seal_path = _resolve(receipt_binding.get("path"))
    bound = _bind_runtime_seal(seal_path)
    _same_complete_binding(bound, receipt_binding, label="authoritative runtime seal")
    if bound.get("content_sha256") != receipt_binding.get("content_sha256"):
        raise RuntimeError("authoritative runtime seal content binding mismatch")
    seal_document, _ = _load_json(seal_path, "authoritative execution runtime seal")
    if (
        seal_document.get("attestation_phase") != "prelaunch"
        or seal_document.get("post_launch_attestation") is not False
    ):
        raise RuntimeError("authoritative execution runtime seal is not prelaunch")
    verified = runtime_seal.verify(seal_path)
    if verified.get("content_sha256") != bound["content_sha256"]:
        raise RuntimeError("authoritative runtime seal identity changed during live verify")
    try:
        receipt_verified_files = int(receipt_binding.get("verified_files", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("execution receipt verified_files is malformed") from exc
    if receipt_verified_files != int(verified.get("verified_files", -2)):
        raise RuntimeError("execution receipt/runtime seal verified-file count mismatch")
    sources = seal_document.get("sources")
    if not isinstance(sources, list):
        raise RuntimeError("authoritative runtime seal lacks its source inventory")
    sealed_by_path: dict[Path, Mapping[str, Any]] = {}
    for raw in sources:
        if not isinstance(raw, Mapping):
            raise RuntimeError("authoritative runtime seal source entry is malformed")
        source_path = _resolve(raw.get("path"))
        if source_path in sealed_by_path:
            raise RuntimeError("authoritative runtime seal duplicates a source path")
        sealed_by_path[source_path] = raw
    canonical_sources: dict[str, dict[str, Any]] = {}
    for name, path in (
        ("supervisor", CANONICAL_SUPERVISOR),
        ("runner", CANONICAL_RUNNER),
    ):
        live = _bind_regular_file(path, f"canonical execution {name}")
        sealed = sealed_by_path.get(path.resolve())
        if not isinstance(sealed, Mapping):
            raise RuntimeError(f"authoritative runtime seal omits canonical {name}")
        _same_complete_binding(sealed, live, label=f"sealed canonical {name}")
        canonical_sources[name] = live
    return {
        **bound,
        "attestation_phase": "prelaunch",
        "verified_files": int(verified["verified_files"]),
        "payloads_rehashed_during_merge": True,
        "canonical_execution_sources": canonical_sources,
    }


def _bind_execution_attestation(
    path: Path,
    *,
    retrain_plan: Mapping[str, Any],
    retrain_index_path: Path,
    retrain_index_binding: Mapping[str, Any],
    retrain_index_content_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the supervisor's complete execution receipt.

    The execution prelaunch seal proves the byte inventory, while this receipt
    proves that the seal was rechecked around every bounded one-unit invocation.
    Keeping both in merge provenance closes the otherwise orphaned supervisor
    evidence edge.
    """

    value, binding = _load_json(path, "sealed retrain execution attestation")
    try:
        invocations_this_resume = int(value.get("invocations_this_resume"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "sealed retrain execution attestation lacks invocation evidence"
        ) from exc
    unit_command = value.get("unit_command")
    expected_unit_command = [
        str(CANONICAL_PYTHON_LAUNCHER.absolute()),
        str(CANONICAL_RUNNER.resolve()),
        "--manifest-root",
        str(_resolve(retrain_plan.get("manifest_root"))),
        "--control-root",
        str(retrain_index_path.resolve().parent),
        "--run-root",
        str(_resolve(retrain_plan.get("reusable_run_root"))),
        "--outer-folds",
        ",".join(str(value) for value in retrain_plan.get("outer_folds", ())),
        "--gpu-lock",
        str((CANONICAL_GPU_ROOT / "gpu_admission.lock").resolve()),
        "--gpu-ledger",
        str((CANONICAL_GPU_ROOT / "gpu_admission_ledger.jsonl").resolve()),
        "--max-new-units",
        "1",
    ]
    if (
        value.get("schema_version") != 1
        or value.get("classification")
        != "sealed_non_test_proposer_execution_attestation"
        or value.get("outer_test_opened") is not False
        or int(value.get("outer_test_record_count", -1)) != 0
        or value.get("commercial_claim_authorized") is not False
        or int(value.get("expected_units", -1)) != 30
        or int(value.get("completed_units", -1)) != 30
        or value.get("one_new_unit_per_invocation") is not True
        or value.get("runtime_seal_verified_before_and_after_every_invocation")
        is not True
        # This campaign has no durable per-invocation receipt ledger from an
        # earlier supervisor process.  Consequently a complete attestation is
        # valid only when this *single sealed supervisor execution* performed
        # the whole 30-unit cover.  Accepting zero (or a partial suffix) would
        # allow a supervisor started after an already-complete campaign to
        # manufacture execution provenance retroactively.
        or invocations_this_resume != 30
        or not isinstance(unit_command, list)
        or unit_command != expected_unit_command
        or canonical_content_sha256(value) != value.get("content_sha256")
    ):
        raise RuntimeError("sealed retrain execution attestation invariants are invalid")
    serialized_command = canonical_json_bytes(unit_command).decode("utf-8").lower()
    if "test_pred_" in serialized_command or "target" in serialized_command:
        raise RuntimeError("sealed retrain execution command contains forbidden input")
    campaign_index = value.get("campaign_index")
    if (
        not isinstance(campaign_index, Mapping)
        or _resolve(campaign_index.get("path")) != retrain_index_path.resolve()
        or campaign_index.get("sha256") != retrain_index_binding.get("sha256")
        or int(campaign_index.get("bytes", -1)) != int(
            retrain_index_binding.get("bytes", -2)
        )
        or campaign_index.get("content_sha256") != retrain_index_content_sha256
    ):
        raise RuntimeError("execution attestation is bound to another retrain index")
    runtime = value.get("runtime_input_seal")
    authoritative_runtime_seal = _live_verify_execution_runtime_seal(runtime)
    supervisor = value.get("supervisor")
    if not isinstance(supervisor, Mapping):
        raise RuntimeError("execution attestation lacks its supervisor binding")
    live_supervisor = _bind_regular_file(
        _resolve(supervisor.get("path")), "execution supervisor"
    )
    _same_complete_binding(
        live_supervisor, supervisor, label="execution supervisor"
    )
    canonical_sources = authoritative_runtime_seal.get("canonical_execution_sources")
    if not isinstance(canonical_sources, Mapping):
        raise RuntimeError("authoritative runtime seal lacks canonical execution sources")
    _same_complete_binding(
        live_supervisor,
        canonical_sources.get("supervisor", {}),
        label="receipt/canonical execution supervisor",
    )
    _same_complete_binding(
        _bind_regular_file(CANONICAL_RUNNER, "canonical campaign runner"),
        canonical_sources.get("runner", {}),
        label="command/canonical campaign runner",
    )
    attestation_binding = {
        **binding,
        "content_sha256": str(value["content_sha256"]),
        "classification": str(value["classification"]),
    }
    return attestation_binding, {
        "required": True,
        "execution_attestation": attestation_binding,
        "authoritative_runtime_input_seal": authoritative_runtime_seal,
        "supervisor": live_supervisor,
        "completion_evidence": {
            "expected_units": 30,
            "completed_units": 30,
            "campaign_index": dict(campaign_index),
            "one_new_unit_per_invocation": True,
            "runtime_seal_verified_before_and_after_every_invocation": True,
            "invocations_this_resume": invocations_this_resume,
            "single_supervisor_execution_covered_all_units": True,
        },
        "attestation_content_verified": True,
        "canonical_supervisor_and_unit_command_verified": True,
        "campaign_index_30_of_30_verified": True,
        "runtime_seal_live_rehashed": True,
        "supervisor_live_rehashed": True,
    }


def _bind_execution_supersession(
    path: Path,
    *,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind and validate an optional note documenting abandoned seal ABIs."""

    value, binding = _load_json(path, "execution runtime seal supersession note")
    if (
        value.get("schema_version") != 1
        or value.get("classification")
        != "non_test_proposer_execution_runtime_seal_supersession"
        or value.get("outer_test_opened") is not False
        or value.get("target_or_reference_accessed") is not False
        or value.get("commercial_claim_authorized") is not False
    ):
        raise RuntimeError("execution runtime seal supersession note invariants are invalid")
    selected = value.get("selected_runtime_seal")
    authoritative = execution.get("authoritative_runtime_input_seal")
    supervisor = execution.get("supervisor")
    if not isinstance(selected, Mapping) or not isinstance(authoritative, Mapping):
        raise RuntimeError("execution supersession note lacks selected runtime seal")
    _same_identity_binding(selected, authoritative, label="supersession selected seal")
    if (
        selected.get("content_sha256") != authoritative.get("content_sha256")
        or int(selected.get("verified_files", -1))
        != int(authoritative.get("verified_files", -2))
        or not isinstance(supervisor, Mapping)
        or selected.get("supervisor_sha256") != supervisor.get("sha256")
    ):
        raise RuntimeError("execution supersession note selected seal metadata mismatch")
    superseded = value.get("superseded_runtime_seals")
    if not isinstance(superseded, list) or not superseded:
        raise RuntimeError("execution supersession note lacks superseded seals")
    seen: set[Path] = set()
    validated_superseded: list[dict[str, Any]] = []
    authoritative_path = _resolve(authoritative.get("path"))
    for position, raw in enumerate(superseded):
        if not isinstance(raw, Mapping) or not str(raw.get("reason", "")).strip():
            raise RuntimeError("execution supersession note has malformed entry")
        old_path = _resolve(raw.get("path"))
        if old_path == authoritative_path or old_path in seen:
            raise RuntimeError("execution supersession note selects/duplicates an old seal")
        seen.add(old_path)
        old_bound = _bind_runtime_seal(old_path)
        _same_identity_binding(old_bound, raw, label=f"superseded seal {position}")
        validated_superseded.append(
            {
                **old_bound,
                "reason": str(raw["reason"]),
            }
        )
    command = value.get("selected_execution_command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise RuntimeError("execution supersession note has invalid selected command")
    if str(authoritative_path) not in command:
        raise RuntimeError("execution supersession command does not select authoritative seal")
    return {
        **binding,
        "classification": str(value["classification"]),
        "selected_runtime_seal": dict(selected),
        "superseded_runtime_seals": validated_superseded,
        "superseded_runtime_seal_count": len(validated_superseded),
    }


def _document_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    _assert_no_test_marker(target, "merged index output")
    payload = _document_bytes(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or target.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite immutable merged index: {target}")
        target.chmod(0o444)
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        target.chmod(0o444)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def merge_indexes(
    *,
    full_plan_path: Path,
    main_index_path: Path,
    retrain_plan_path: Path,
    retrain_index_path: Path,
    output_path: Path,
    runtime_seals: Sequence[Path] = (),
    inspect_unit: InspectUnit | None = None,
) -> dict[str, Any]:
    full_plan_path = full_plan_path.expanduser().resolve()
    main_index_path = main_index_path.expanduser().resolve()
    retrain_plan_path = retrain_plan_path.expanduser().resolve()
    retrain_index_path = retrain_index_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    for path, label in (
        (full_plan_path, "full plan"),
        (main_index_path, "main index"),
        (retrain_plan_path, "retrain plan"),
        (retrain_index_path, "retrain index"),
        (output_path, "merged index output"),
    ):
        _assert_no_test_marker(path, label)
    if output_path in {
        full_plan_path,
        main_index_path,
        retrain_plan_path,
        retrain_index_path,
    }:
        raise RuntimeError("merged index output must not overwrite a source document")

    full, full_binding, full_units = _validate_plan(
        full_plan_path, expected_folds=FOLDS, label="full six-fold proposer plan"
    )
    retrain, retrain_binding, retrain_units = _validate_plan(
        retrain_plan_path,
        expected_folds=sorted(RETRAIN_FOLDS),
        label="current-source folds-3/4 retrain plan",
    )
    _assert_compatible_plans(full, retrain, full_units, retrain_units)

    # Index-specific closures remove ambiguity when both plans share the same
    # unit ids but point their outputs at different immutable run roots.
    def inspect_main(unit: Mapping[str, Any]):
        if inspect_unit is not None:
            return inspect_unit(unit)
        return campaign.inspect_unit(unit, run_root=_resolve(full["reusable_run_root"]))

    def inspect_retrain(unit: Mapping[str, Any]):
        if inspect_unit is not None:
            return inspect_unit(unit)
        return campaign.inspect_unit(unit, run_root=_resolve(retrain["reusable_run_root"]))

    main_index, main_binding, main_records = _validate_index(
        main_index_path,
        plan_path=full_plan_path,
        plan=full,
        plan_binding=full_binding,
        planned=full_units,
        label="completed main proposer index",
        inspect_unit=inspect_main,
    )
    retrain_index, retrain_index_binding, retrain_records = _validate_index(
        retrain_index_path,
        plan_path=retrain_plan_path,
        plan=retrain,
        plan_binding=retrain_binding,
        planned=retrain_units,
        label="completed current-source folds-3/4 proposer index",
        inspect_unit=inspect_retrain,
    )

    selected: list[dict[str, Any]] = []
    selected_counts = {"main": 0, "current_source_retrain_f34": 0}
    for key in full_units:
        outer = key[0]
        if outer in RETRAIN_FOLDS:
            record = retrain_records.get(key)
            source = "current_source_retrain_f34"
        elif outer in MAIN_FOLDS:
            record = main_records.get(key)
            source = "main"
        else:  # pragma: no cover - fixed fold partition is asserted below
            raise RuntimeError(f"outer fold has no declared merge source: {outer}")
        if record is None:
            raise RuntimeError(f"selected source lacks required proposer record: {key}")
        selected.append(record)
        selected_counts[source] += 1
    if (
        MAIN_FOLDS | RETRAIN_FOLDS != set(FOLDS)
        or MAIN_FOLDS & RETRAIN_FOLDS
        or len(selected) != 90
        or selected_counts != {"main": 60, "current_source_retrain_f34": 30}
        or len({_record_key(record) for record in selected}) != 90
    ):
        raise RuntimeError("merged proposer selection is not the exact 60+30 unit cover")

    seals = _collect_runtime_seals(
        full_plan_path=full_plan_path,
        retrain_plan_path=retrain_plan_path,
        explicit=runtime_seals,
    )
    retrain_campaign_root = retrain_plan_path.parent.parent
    execution_attestation_path = retrain_campaign_root / "execution_attestation.json"
    if not execution_attestation_path.is_file():
        raise RuntimeError(
            "completed retrain merge requires the supervisor's 30/30 execution attestation"
        )
    execution_attestation, retrain_execution = _bind_execution_attestation(
        execution_attestation_path,
        retrain_plan=retrain,
        retrain_index_path=retrain_index_path,
        retrain_index_binding=retrain_index_binding,
        retrain_index_content_sha256=str(retrain_index["content_sha256"]),
    )
    authoritative_seal = retrain_execution["authoritative_runtime_input_seal"]
    if not any(
        _resolve(item.get("path")) == _resolve(authoritative_seal.get("path"))
        for item in seals
    ):
        seals.append(dict(authoritative_seal))
    supersession_path = (
        retrain_campaign_root / "execution_runtime_seal_supersession.json"
    )
    retrain_execution["supersession_note"] = (
        _bind_execution_supersession(
            supersession_path,
            execution=retrain_execution,
        )
        if supersession_path.exists()
        else None
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        # Keep the exact classification consumed by fixed-i3; the explicit
        # merge classification lives in immutable provenance below.
        "classification": INDEX_CLASSIFICATION,
        "merge_classification": "retrospective_current_source_uniform_90_unit_proposer_index",
        "campaign_plan_content_sha256": full["content_sha256"],
        "campaign_plan": {
            **full_binding,
            "content_sha256": full["content_sha256"],
        },
        "manifest_root": full["manifest_root"],
        "manifest_plan_content_sha256": full["manifest_plan_content_sha256"],
        "outer_test_opened": False,
        "outer_test_record_count": 0,
        "requested_units": 90,
        "completed_units": 90,
        "source_uniformity_scope": "all_selected_units_validated_under_bound_current_sources",
        "merge_provenance": {
            "full_split_authority_plan": {
                **full_binding,
                "content_sha256": full["content_sha256"],
            },
            "retrain_plan": {
                **retrain_binding,
                "content_sha256": retrain["content_sha256"],
            },
            "source_indexes": {
                "main": {
                    **main_binding,
                    "content_sha256": main_index["content_sha256"],
                    "selected_outer_folds": sorted(MAIN_FOLDS),
                    "selected_units": 60,
                },
                "current_source_retrain_f34": {
                    **retrain_index_binding,
                    "content_sha256": retrain_index["content_sha256"],
                    "selected_outer_folds": sorted(RETRAIN_FOLDS),
                    "selected_units": 30,
                },
            },
            "excluded_main_outer_folds": sorted(RETRAIN_FOLDS),
            "runtime_seals": seals,
            "runtime_seal_count": len(seals),
            "retrain_execution": retrain_execution,
            "retrain_execution_attestation": execution_attestation,
            "retrain_execution_attestation_required": True,
            "merge_tool": _bind_regular_file(Path(__file__), "merge tool"),
            "target_or_outer_test_input_accepted": False,
        },
        "records": selected,
    }
    result["content_sha256"] = canonical_content_sha256(result)
    _write_immutable(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-plan", type=Path, default=DEFAULT_FULL_PLAN)
    parser.add_argument("--main-index", type=Path, default=DEFAULT_MAIN_INDEX)
    parser.add_argument("--retrain-plan", type=Path, default=DEFAULT_RETRAIN_PLAN)
    parser.add_argument("--retrain-index", type=Path, default=DEFAULT_RETRAIN_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--runtime-seal",
        action="append",
        type=Path,
        default=[],
        help="additional pre-existing runtime seal to bind (payloads are not reopened)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = merge_indexes(
        full_plan_path=args.full_plan,
        main_index_path=args.main_index,
        retrain_plan_path=args.retrain_plan,
        retrain_index_path=args.retrain_index,
        output_path=args.output,
        runtime_seals=args.runtime_seal,
    )
    print(
        json.dumps(
            {
                "status": "sealed",
                "output": str(args.output.expanduser().resolve()),
                "content_sha256": document["content_sha256"],
                "completed_units": document["completed_units"],
                "runtime_seal_count": document["merge_provenance"]["runtime_seal_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
