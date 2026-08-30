#!/usr/bin/env python3
"""Seal the completed fixed-i3 pre-test payload against its runtime inventory.

This is the byte-closure between the pre-test training campaign and any
label-free outer-test work.  It re-verifies the immutable runtime inventory,
the merged 90-unit proposer index bound by that inventory, all 18 fixed-i3
unit artifact bindings, and every recorded output tree.  It never accepts an
outer-test, target, or evaluation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import seal_runtime_inputs as runtime_seal  # noqa: E402


CAMPAIGN_ROOT = PROJECT_ROOT / "artifacts/campaigns/harmonic_candidate_set_snn_v2"
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2"
DEFAULT_MERGED_INDEX = (
    CAMPAIGN_ROOT / "nested_proposer/current_source_merged/index.json"
)
DEFAULT_RUNTIME_SEAL = (
    CAMPAIGN_ROOT
    / "nested_proposer/current_source_merged/fixed_i3_pretest_runtime_seal.json"
)
DEFAULT_PRETEST_ROOT = RUN_ROOT / "hcs_fixed_i3_pretest"
DEFAULT_PRETEST_INDEX = DEFAULT_PRETEST_ROOT / "pretest_index.json"
DEFAULT_OUTPUT = DEFAULT_PRETEST_ROOT / "fixed_runtime_completion_attestation.json"
FOLDS = tuple(range(6))
SEEDS = (20260828, 20260829, 20260830)
FORBIDDEN_NAMES = frozenset(
    {
        "evaluation_lock.json",
        "locked_hcs_oof_joined.npz",
        "canonical_locked_hcs_targets.npz",
        "canonical_locked_hcs_targets_receipt.json",
        "canonical_locked_hcs_targets_release_receipt.json",
    }
)


class FixedCompletionError(RuntimeError):
    """A pre-test completion or provenance invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    document = dict(value)
    document.pop("content_sha256", None)
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FixedCompletionError(f"required file is absent: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _require_mode_0444(path: Path, label: str) -> Path:
    raw = path.expanduser()
    if raw.is_symlink():
        raise FixedCompletionError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file() or resolved.stat().st_mode & 0o777 != 0o444:
        raise FixedCompletionError(f"{label} mode must be exactly 0444: {resolved}")
    return resolved


def _json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixedCompletionError(f"invalid {label}: {resolved} ({exc})") from exc
    if not isinstance(value, dict):
        raise FixedCompletionError(f"{label} must be a JSON object: {resolved}")
    return value


def _resolve(raw: Any, *, relative_to: Path) -> Path:
    if not isinstance(raw, (str, os.PathLike)) or not str(raw):
        raise FixedCompletionError("artifact path is absent")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def verify_binding(
    raw: Any, *, relative_to: Path, label: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise FixedCompletionError(f"missing file binding: {label}")
    path = _resolve(raw.get("path"), relative_to=relative_to)
    expected = str(raw.get("sha256", ""))
    if len(expected) != 64 or not path.is_file() or sha256_file(path) != expected:
        raise FixedCompletionError(f"file hash mismatch: {label} ({path})")
    size = path.stat().st_size
    if "bytes" in raw and int(raw["bytes"]) != size:
        raise FixedCompletionError(f"file byte-size mismatch: {label} ({path})")
    return {"path": str(path), "sha256": expected, "bytes": size}


def _tree_sha(records: list[list[Any]]) -> str:
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_tree(raw: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("files"), Mapping):
        raise FixedCompletionError(f"missing recorded output tree: {label}")
    root = _resolve(raw.get("path"), relative_to=PROJECT_ROOT)
    if not root.is_dir() or root.is_symlink():
        raise FixedCompletionError(f"recorded output tree root is invalid: {root}")
    expected_files = raw["files"]
    members = list(root.rglob("*"))
    symlinks = [path for path in members if path.is_symlink()]
    if symlinks:
        raise FixedCompletionError(f"output tree contains a symlink: {symlinks[0]}")
    observed_paths = sorted(path for path in members if path.is_file())
    observed_names = [str(path.relative_to(root)) for path in observed_paths]
    if set(observed_names) != set(map(str, expected_files)):
        raise FixedCompletionError(f"output tree membership mismatch: {label} ({root})")
    records: list[list[Any]] = []
    for name, path in zip(observed_names, observed_paths, strict=True):
        binding = expected_files.get(name)
        if not isinstance(binding, Mapping):
            raise FixedCompletionError(f"invalid output-tree binding: {label}/{name}")
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != binding.get("sha256") or size != int(binding.get("bytes", -1)):
            raise FixedCompletionError(f"output tree byte mismatch: {label}/{name}")
        records.append([name, digest, size])
    tree_digest = _tree_sha(records)
    if tree_digest != raw.get("tree_sha256"):
        raise FixedCompletionError(f"output tree digest mismatch: {label}")
    return {
        "path": str(root),
        "file_count": len(records),
        "tree_sha256": tree_digest,
    }


def _assert_no_forbidden_files(root: Path) -> None:
    resolved = root.expanduser().resolve()
    if not resolved.exists():
        return
    found = sorted(
        str(path)
        for path in resolved.rglob("*")
        if path.is_file() and path.name.lower() in FORBIDDEN_NAMES
    )
    if found:
        raise FixedCompletionError(
            f"target/evaluation artifact exists in pre-test scope: {found[0]}"
        )


def _runtime_bound_merged_index(
    seal: Mapping[str, Any], *, seal_path: Path, merged_index: Path
) -> dict[str, Any]:
    context = seal.get("fixed_i3_context")
    if not isinstance(context, Mapping):
        raise FixedCompletionError("runtime seal lacks fixed_i3_context")
    if (
        context.get("classification")
        != "retrospective_fixed_i3_pretest_runtime_input_context"
        or context.get("outer_test_opened") is not False
        or int(context.get("outer_test_record_count", -1)) != 0
        or context.get("target_or_evaluation_artifact_accessed") is not False
        or int(context.get("proposer_matrix_groups", -1)) != 18
        or int(context.get("proposer_matrix_units", -1)) != 90
    ):
        raise FixedCompletionError("runtime seal fixed-i3 context is not pre-test safe")
    recorded = verify_binding(
        context.get("validated_index"),
        relative_to=seal_path.parent,
        label="runtime-sealed merged proposer index",
    )
    observed = bind_file(merged_index)
    if recorded != observed:
        raise FixedCompletionError("runtime seal is bound to another merged proposer index")
    return observed


def validate_pretest_index(
    pretest_index: Path, *, expected_seeds: Sequence[int] = SEEDS
) -> dict[str, Any]:
    path = _require_mode_0444(pretest_index, "fixed-i3 pretest index")
    document = _json(path, "fixed-i3 pretest index")
    if (
        document.get("schema_version") != 1
        or document.get("classification") != "retrospective_fixed_i3_pretest_index"
        or document.get("status") != "complete"
        or document.get("outer_test_opened") is not False
        or int(document.get("outer_test_artifact_count", 0)) != 0
        or document.get(
            "ready_for_separately_locked_label_free_outer_test_construction"
        )
        is not True
        or int(document.get("completed_units", -1)) != 18
    ):
        raise FixedCompletionError("fixed-i3 pretest index is not a complete unopened seal")
    if canonical_sha256(document) != document.get("content_sha256"):
        raise FixedCompletionError("fixed-i3 pretest index content hash mismatch")
    snapshot = path.parent / "pretest_index_snapshots" / f"{document['content_sha256']}.json"
    _require_mode_0444(snapshot, "fixed-i3 pretest index snapshot")
    if snapshot.read_bytes() != path.read_bytes():
        raise FixedCompletionError("fixed-i3 pretest index snapshot differs from final index")
    matrix = document.get("matrix")
    seeds = tuple(map(int, expected_seeds))
    if (
        not isinstance(matrix, Mapping)
        or matrix.get("folds") != list(FOLDS)
        or tuple(map(int, matrix.get("seeds", []))) != seeds
        or int(matrix.get("unit_count", -1)) != 18
    ):
        raise FixedCompletionError("fixed-i3 pretest matrix is not the locked 6x3 topology")
    common = document.get("common")
    if not isinstance(common, Mapping):
        raise FixedCompletionError("fixed-i3 pretest index lacks common bindings")
    common_bindings = {
        name: verify_binding(
            common.get(name), relative_to=path.parent, label=f"pretest common {name}"
        )
        for name in (
            "selection_lock",
            "capacity_selection",
            "policy",
            "source_freeze_manifest",
        )
    }
    units = document.get("units")
    if not isinstance(units, list) or len(units) != 18:
        raise FixedCompletionError("fixed-i3 pretest index does not contain 18 units")
    expected = {(fold, seed) for seed in seeds for fold in FOLDS}
    observed: set[tuple[int, int]] = set()
    unit_records: list[dict[str, Any]] = []
    for position, unit in enumerate(units):
        if not isinstance(unit, Mapping) or unit.get("status") != "complete":
            raise FixedCompletionError(f"fixed-i3 unit is incomplete: position {position}")
        key = (int(unit.get("outer_fold", -1)), int(unit.get("seed", -1)))
        if key in observed or key not in expected:
            raise FixedCompletionError(f"fixed-i3 unit topology mismatch: {key}")
        if unit.get("outer_test_opened") is not False:
            raise FixedCompletionError(f"fixed-i3 unit is not test-sealed: {key}")
        artifacts = unit.get("artifacts")
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise FixedCompletionError(f"fixed-i3 unit lacks artifact bindings: {key}")
        verified_artifacts = {
            str(name): verify_binding(
                raw, relative_to=path.parent, label=f"unit {key} artifact {name}"
            )
            for name, raw in sorted(artifacts.items())
        }
        required = {
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
        if not required.issubset(verified_artifacts):
            raise FixedCompletionError(f"fixed-i3 unit artifact set is incomplete: {key}")
        tree = verify_tree(unit.get("output_tree"), label=f"unit {key} output_tree")
        unit_records.append(
            {
                "outer_fold": key[0],
                "seed": key[1],
                "artifact_count": len(verified_artifacts),
                "artifacts_sha256": _tree_sha(
                    [
                        [name, item["sha256"], item["bytes"]]
                        for name, item in sorted(verified_artifacts.items())
                    ]
                ),
                "output_tree": tree,
            }
        )
        observed.add(key)
    if observed != expected:
        raise FixedCompletionError("fixed-i3 pretest units do not exactly cover 6x3")
    return {
        "document": document,
        "binding": bind_file(path),
        "common_bindings": common_bindings,
        "units": sorted(unit_records, key=lambda item: (item["seed"], item["outer_fold"])),
    }


def validate_campaign_execution_evidence(
    *,
    pretest_index: Path,
    pretest_document: Mapping[str, Any],
    runtime_input_seal: Path,
    runtime_document: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the campaign crossed every unit boundary through sealed code.

    The fixed campaign did not publish a separate runtime receipt per unit.
    Its immutable evidence is instead the conjunction of (a) the campaign
    lock binding the exact runtime seal and orchestrator source, (b) the
    runtime inventory binding those source bytes, and (c) the content-addressed
    status-snapshot chain containing every completed prefix 1..18.  The bound
    orchestrator's control flow calls ``runtime_seal.verify`` immediately
    before and after each unit and before final publication; the completion
    attestation records that conclusion only after all three evidence classes
    are revalidated here.
    """

    index_path = pretest_index.expanduser().resolve()
    root = index_path.parent
    campaign_path = root / "campaign_lock.json"
    _require_mode_0444(campaign_path, "fixed-i3 campaign lock")
    campaign = _json(campaign_path, "fixed-i3 campaign lock")
    if (
        campaign.get("schema_version") != 1
        or campaign.get("classification")
        != "retrospective_fixed_i3_pretest_completion_plan"
        or campaign.get("matrix")
        != {"folds": list(FOLDS), "seeds": list(SEEDS), "unit_count": 18}
        or campaign.get("outer_test_opened") is not False
        or canonical_sha256(campaign) != campaign.get("content_sha256")
        or pretest_document.get("campaign_lock_sha256")
        != campaign.get("content_sha256")
    ):
        raise FixedCompletionError("fixed-i3 campaign lock execution evidence is invalid")
    inputs = campaign.get("inputs")
    sources = campaign.get("sources")
    if not isinstance(inputs, Mapping) or not isinstance(sources, Mapping):
        raise FixedCompletionError("fixed-i3 campaign lock lacks runtime/source evidence")
    recorded_runtime = verify_binding(
        inputs.get("runtime_input_seal"),
        relative_to=campaign_path.parent,
        label="campaign-lock runtime seal",
    )
    live_runtime = bind_file(runtime_input_seal)
    runtime_input = inputs.get("runtime_input_seal")
    if (
        recorded_runtime != live_runtime
        or not isinstance(runtime_input, Mapping)
        or runtime_input.get("content_sha256") != runtime_document.get("content_sha256")
        or runtime_input.get("attestation_phase") != "prelaunch"
    ):
        raise FixedCompletionError("campaign lock is bound to another runtime seal")
    sealed_sources = {
        str(item.get("path")): item
        for item in runtime_document.get("sources", [])
        if isinstance(item, Mapping)
    }
    source_evidence: dict[str, dict[str, Any]] = {}
    for name in ("orchestrator", "runtime_seal_verifier"):
        observed = verify_binding(
            sources.get(name), relative_to=campaign_path.parent, label=f"campaign source {name}"
        )
        sealed = sealed_sources.get(observed["path"])
        if not isinstance(sealed, Mapping) or (
            sealed.get("sha256") != observed["sha256"]
            or int(sealed.get("bytes", -1)) != observed["bytes"]
        ):
            raise FixedCompletionError(
                f"campaign execution source is absent from runtime inventory: {name}"
            )
        source_evidence[name] = observed
    if Path(source_evidence["orchestrator"]["path"]).name != "run_fixed_i3_pretest_campaign.py":
        raise FixedCompletionError("campaign orchestrator evidence names another program")
    if Path(source_evidence["runtime_seal_verifier"]["path"]).name != "seal_runtime_inputs.py":
        raise FixedCompletionError("campaign runtime verifier evidence names another program")

    final_status_path = root / "pretest_status.json"
    _require_mode_0444(final_status_path, "fixed-i3 final pretest status")
    final_status = _json(final_status_path, "fixed-i3 final pretest status")
    if (
        final_status.get("classification") != "retrospective_fixed_i3_pretest_status"
        or final_status.get("campaign_lock_sha256") != campaign["content_sha256"]
        or final_status.get("status") != "complete"
        or int(final_status.get("completed_units", -1)) != 18
        or final_status.get("outer_test_opened") is not False
        or canonical_sha256(final_status) != final_status.get("content_sha256")
        or final_status.get("units") != pretest_document.get("units")
    ):
        raise FixedCompletionError("fixed-i3 final status execution evidence is invalid")
    snapshots_root = root / "pretest_status_snapshots"
    if not snapshots_root.is_dir():
        raise FixedCompletionError("fixed-i3 status snapshot evidence is absent")
    snapshot_bindings: list[dict[str, Any]] = []
    prefix_coverage: set[int] = set()
    final_units = pretest_document["units"]
    for snapshot_path in sorted(snapshots_root.glob("*.json"), key=lambda item: item.name):
        _require_mode_0444(snapshot_path, "fixed-i3 status snapshot")
        snapshot = _json(snapshot_path, "fixed-i3 status snapshot")
        digest = snapshot.get("content_sha256")
        if (
            canonical_sha256(snapshot) != digest
            or snapshot_path.stem != digest
            or snapshot.get("classification") != "retrospective_fixed_i3_pretest_status"
            or snapshot.get("campaign_lock_sha256") != campaign["content_sha256"]
            or snapshot.get("outer_test_opened") is not False
        ):
            raise FixedCompletionError(f"invalid fixed-i3 status snapshot: {snapshot_path}")
        completed = int(snapshot.get("completed_units", -1))
        units = snapshot.get("units")
        if completed < 0 or completed > 18 or not isinstance(units, list):
            raise FixedCompletionError(f"invalid fixed-i3 status prefix: {snapshot_path}")
        completed_units = [unit for unit in units if unit.get("status") == "complete"]
        if len(completed_units) != completed:
            raise FixedCompletionError(f"status snapshot completed count differs: {snapshot_path}")
        if completed > 0 and completed_units == final_units[:completed]:
            prefix_coverage.add(completed)
        snapshot_bindings.append(bind_file(snapshot_path))
    if prefix_coverage != set(range(1, 19)):
        missing = sorted(set(range(1, 19)) - prefix_coverage)
        raise FixedCompletionError(
            f"fixed-i3 status snapshot chain lacks completed prefixes: {missing}"
        )
    final_binding = bind_file(final_status_path)
    matching_final = [item for item in snapshot_bindings if item["sha256"] == final_binding["sha256"]]
    if len(matching_final) != 1:
        raise FixedCompletionError("final fixed-i3 status lacks one immutable snapshot")
    return {
        "campaign_lock": bind_file(campaign_path),
        "campaign_lock_content_sha256": campaign["content_sha256"],
        "final_pretest_status": final_binding,
        "final_pretest_status_content_sha256": final_status["content_sha256"],
        "status_snapshot_count": len(snapshot_bindings),
        "completed_prefixes_revalidated": list(range(1, 19)),
        "status_snapshot_graph_sha256": _tree_sha(
            [
                [Path(item["path"]).name, item["sha256"], item["bytes"]]
                for item in snapshot_bindings
            ]
        ),
        "orchestrator_source": source_evidence["orchestrator"],
        "runtime_seal_verifier_source": source_evidence["runtime_seal_verifier"],
        "evidence_rule": (
            "runtime-sealed orchestrator control flow plus immutable completed-prefix "
            "status chain; runtime verify occurs immediately before and after each unit"
        ),
    }


def _payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    payload = _payload(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != payload:
            raise FixedCompletionError(f"immutable attestation collision: {target}")
        target.chmod(0o444)
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify_completion_attestation(
    attestation_path: Path,
    *,
    expected_runtime_seal: Path | None = None,
    expected_pretest_index: Path | None = None,
    reverify_payload: bool = True,
) -> dict[str, Any]:
    path = _require_mode_0444(
        attestation_path, "fixed-i3 runtime completion attestation"
    )
    document = _json(path, "fixed-i3 runtime completion attestation")
    if (
        document.get("schema_version") != 1
        or document.get("classification")
        != "fixed_i3_pretest_runtime_completion_attestation"
        or int(document.get("completed_units", -1)) != 18
        or document.get("runtime_seal_verified_before_completion") is not True
        or document.get("runtime_seal_verified_before_and_after_every_unit") is not True
        or document.get("all_artifact_and_output_tree_hashes_verified") is not True
        or document.get("outer_test_opened") is not False
        or document.get("target_artifact_opened") is not False
        or canonical_sha256(document) != document.get("content_sha256")
    ):
        raise FixedCompletionError("fixed-i3 completion attestation is invalid")
    runtime_binding = verify_binding(
        document.get("runtime_input_seal"), relative_to=path.parent, label="runtime input seal"
    )
    pretest_binding = verify_binding(
        document.get("pretest_index"), relative_to=path.parent, label="pretest index"
    )
    merged_binding = verify_binding(
        document.get("merged_proposer_index"),
        relative_to=path.parent,
        label="merged proposer index",
    )
    if expected_runtime_seal is not None and Path(runtime_binding["path"]) != expected_runtime_seal.resolve():
        raise FixedCompletionError("completion attestation binds another runtime seal")
    if expected_pretest_index is not None and Path(pretest_binding["path"]) != expected_pretest_index.resolve():
        raise FixedCompletionError("completion attestation binds another pretest index")
    verify_binding(document.get("sealer_source"), relative_to=path.parent, label="completion sealer")
    verify_binding(document.get("python_executable"), relative_to=path.parent, label="Python executable")
    evidence = document.get("fixed_campaign_execution_evidence")
    if not isinstance(evidence, Mapping):
        raise FixedCompletionError("completion attestation lacks fixed-campaign execution evidence")
    for name in (
        "campaign_lock",
        "final_pretest_status",
        "orchestrator_source",
        "runtime_seal_verifier_source",
    ):
        verify_binding(
            evidence.get(name), relative_to=path.parent, label=f"completion evidence {name}"
        )
    if evidence.get("completed_prefixes_revalidated") != list(range(1, 19)):
        raise FixedCompletionError("completion evidence lacks all 18 completed prefixes")
    if reverify_payload:
        verified_runtime = runtime_seal.verify(Path(runtime_binding["path"]))
        if verified_runtime.get("content_sha256") != document.get("runtime_content_sha256"):
            raise FixedCompletionError("runtime seal identity differs from completion attestation")
        validated = validate_pretest_index(Path(pretest_binding["path"]))
        if validated["binding"] != pretest_binding:
            raise FixedCompletionError("pretest index differs from completion attestation")
        live_runtime_document = _json(
            Path(runtime_binding["path"]), "fixed-i3 runtime input seal"
        )
        runtime_merged = _runtime_bound_merged_index(
            live_runtime_document,
            seal_path=Path(runtime_binding["path"]),
            merged_index=Path(merged_binding["path"]),
        )
        if runtime_merged != merged_binding:
            raise FixedCompletionError("merged proposer binding changed")
        live_evidence = validate_campaign_execution_evidence(
            pretest_index=Path(pretest_binding["path"]),
            pretest_document=validated["document"],
            runtime_input_seal=Path(runtime_binding["path"]),
            runtime_document=live_runtime_document,
        )
        if live_evidence != evidence:
            raise FixedCompletionError("fixed-campaign execution evidence changed")
    return {"document": document, "binding": bind_file(path)}


def seal_completion(
    *,
    merged_index: Path,
    runtime_input_seal: Path,
    pretest_index: Path,
    output: Path,
    python_executable: Path = Path(sys.executable),
) -> dict[str, Any]:
    runtime_path = runtime_input_seal.expanduser().resolve()
    try:
        runtime_result = runtime_seal.verify(runtime_path)
    except RuntimeError as exc:
        raise FixedCompletionError(str(exc)) from exc
    runtime_document = _json(runtime_path, "fixed-i3 runtime input seal")
    merged_binding = _runtime_bound_merged_index(
        runtime_document,
        seal_path=runtime_path,
        merged_index=merged_index.expanduser().resolve(),
    )
    pretest_path = pretest_index.expanduser().resolve()
    _assert_no_forbidden_files(pretest_path.parent)
    validated = validate_pretest_index(pretest_path)
    execution_evidence = validate_campaign_execution_evidence(
        pretest_index=pretest_path,
        pretest_document=validated["document"],
        runtime_input_seal=runtime_path,
        runtime_document=runtime_document,
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "classification": "fixed_i3_pretest_runtime_completion_attestation",
        "runtime_input_seal": bind_file(runtime_path),
        "runtime_content_sha256": runtime_result["content_sha256"],
        "merged_proposer_index": merged_binding,
        "pretest_index": validated["binding"],
        "pretest_index_content_sha256": validated["document"]["content_sha256"],
        "common_bindings": validated["common_bindings"],
        "completed_units": 18,
        "unit_payloads": validated["units"],
        "fixed_campaign_execution_evidence": execution_evidence,
        "runtime_seal_verified_before_completion": True,
        "runtime_seal_verified_before_and_after_every_unit": True,
        "all_artifact_and_output_tree_hashes_verified": True,
        "fixed_i3_runtime_payload_closed": True,
        "outer_test_opened": False,
        "target_artifact_opened": False,
        "commercial_claim_authorized": False,
        "sealer_source": bind_file(Path(__file__)),
        "python_executable": bind_file(python_executable),
    }
    document["content_sha256"] = canonical_sha256(document)
    immutable_json(output, document)
    verified = verify_completion_attestation(
        output,
        expected_runtime_seal=runtime_path,
        expected_pretest_index=pretest_path,
    )
    return verified["document"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-index", type=Path, default=DEFAULT_MERGED_INDEX)
    parser.add_argument("--runtime-seal", type=Path, default=DEFAULT_RUNTIME_SEAL)
    parser.add_argument("--pretest-index", type=Path, default=DEFAULT_PRETEST_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = seal_completion(
        merged_index=args.merged_index,
        runtime_input_seal=args.runtime_seal,
        pretest_index=args.pretest_index,
        output=args.output,
        python_executable=args.python_executable,
    )
    print(
        json.dumps(
            {
                "status": "fixed_i3_runtime_completion_sealed",
                "output": str(args.output.expanduser().resolve()),
                "content_sha256": document["content_sha256"],
                "completed_units": 18,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
