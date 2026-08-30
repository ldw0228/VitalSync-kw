#!/usr/bin/env python3
"""Publish an honest governance attestation for the merged proposer cover.

This utility is intentionally an audit, not an evaluator.  It never accepts a
test manifest, target artifact, or metric.  It verifies the two completed
source indexes, their exact 60+30 merge, and both runtime/cache byte seals.  It
also records the independently audited legacy-loader boundary without claiming
that excluded target columns were literally unopened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer"
)
DEFAULT_MAIN = CAMPAIGN_ROOT / "full_oof_non_test/control/index.json"
DEFAULT_RETRAIN = CAMPAIGN_ROOT / "current_source_retrain_f34/control/index.json"
DEFAULT_MERGED = CAMPAIGN_ROOT / "current_source_merged/index.json"
DEFAULT_MAIN_SEAL = CAMPAIGN_ROOT / "full_oof_non_test/runtime_input_seal.json"
DEFAULT_OUTPUT = CAMPAIGN_ROOT / "current_source_merged/governance_attestation.json"
CANONICAL_EXECUTION_SUPERVISOR = (
    PROJECT_ROOT / "scripts/run_sealed_nested_proposer_supervisor.py"
)

INDEX_CLASSIFICATION = "retrospective_fully_nested_non_test_proposer_index"
MAIN_FOLDS = frozenset((0, 1, 2, 5))
RETRAIN_FOLDS = frozenset((3, 4))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise RuntimeError(f"symlink is forbidden for {label}: {unresolved}")
    resolved = unresolved.resolve()
    if any(part.lower().startswith("test_pred_") for part in resolved.parts):
        raise RuntimeError(f"outer-test path is forbidden in governance audit: {resolved}")
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {resolved} ({exc})") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value, {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _record_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    manifest = Path(str(record.get("manifest", "")))
    if any(part.lower().startswith("test_pred_") for part in manifest.parts):
        raise RuntimeError("outer-test record entered governance audit")
    return int(record.get("outer_fold", -1)), int(record.get("seed", -1)), manifest.name


def _verify_file_binding(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"missing {label} binding")
    path = Path(str(raw.get("path", ""))).expanduser()
    if path.is_symlink():
        raise RuntimeError(f"symlink is forbidden for {label}")
    path = path.resolve()
    if any(part.lower().startswith("test_pred_") for part in path.parts):
        raise RuntimeError(f"outer-test path entered {label}")
    expected = str(raw.get("sha256", ""))
    if not path.is_file() or len(expected) != 64 or sha256_file(path) != expected:
        raise RuntimeError(f"{label} file hash mismatch: {path}")
    if "bytes" in raw and path.stat().st_size != int(raw["bytes"]):
        raise RuntimeError(f"{label} byte-size mismatch: {path}")
    return {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}


def _validate_index(
    path: Path, *, expected_records: int, label: str
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[int, int, str], dict[str, Any]]]:
    value, binding = _load(path, label)
    if (
        value.get("schema_version") != 1
        or value.get("classification") != INDEX_CLASSIFICATION
        or value.get("outer_test_opened") is not False
        or int(value.get("outer_test_record_count", -1)) != 0
        or int(value.get("completed_units", -1)) != expected_records
        or int(value.get("requested_units", -1)) != expected_records
        or canonical_content_sha256(value) != value.get("content_sha256")
    ):
        raise RuntimeError(f"{label} is incomplete, tampered, or not test-sealed")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != expected_records:
        raise RuntimeError(f"{label} has the wrong record count")
    observed: dict[tuple[int, int, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError(f"{label} contains a non-object record")
        key = _record_key(record)
        if key in observed:
            raise RuntimeError(f"{label} contains duplicate record {key}")
        if record.get("role") not in {"hcs_train_oof", "hcs_validation"}:
            raise RuntimeError(f"{label} contains an invalid role")
        manifest = Path(str(record.get("manifest", ""))).expanduser().resolve()
        if not manifest.is_file() or sha256_file(manifest) != str(
            record.get("manifest_sha256", "")
        ):
            raise RuntimeError(f"{label} manifest hash mismatch: {manifest}")
        _verify_file_binding(record.get("checkpoint"), f"{label} checkpoint")
        _verify_file_binding(
            record.get("all_window_prediction"), f"{label} prediction"
        )
        observed[key] = record
    return value, binding, observed


def _load_seal_verifier():
    path = PROJECT_ROOT / "scripts/seal_runtime_inputs.py"
    specification = importlib.util.spec_from_file_location(
        "seal_runtime_inputs_for_governance", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import runtime seal verifier: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _verify_runtime_seal(path: Path, *, expected_phase: str) -> dict[str, Any]:
    verifier = _load_seal_verifier()
    result = verifier.verify(path)
    value, binding = _load(path, "runtime input seal")
    phase = str(value.get("attestation_phase", ""))
    if not phase:
        phase = "post_launch" if value.get("post_launch_attestation") is True else "prelaunch"
    if phase != expected_phase:
        raise RuntimeError(
            f"runtime seal phase is {phase}, expected {expected_phase}: {path}"
        )
    return {
        **binding,
        "content_sha256": value["content_sha256"],
        "attestation_phase": phase,
        "verified_files": int(result["verified_files"]),
        "payloads_rehashed_during_this_audit": True,
    }


def _same_binding(
    left: Any,
    right: Any,
    *,
    label: str,
    require_content: bool = False,
) -> None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise RuntimeError(f"missing {label} binding")
    try:
        same = (
            Path(str(left.get("path", ""))).expanduser().resolve()
            == Path(str(right.get("path", ""))).expanduser().resolve()
            and str(left.get("sha256", "")) == str(right.get("sha256", ""))
            and len(str(left.get("sha256", ""))) == 64
            and int(left.get("bytes", -1)) == int(right.get("bytes", -2))
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"malformed {label} binding") from exc
    if require_content:
        same = (
            same
            and len(str(left.get("content_sha256", ""))) == 64
            and left.get("content_sha256") == right.get("content_sha256")
        )
    if not same:
        raise RuntimeError(f"{label} binding mismatch")


def _same_identity(left: Any, right: Any, *, label: str) -> None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise RuntimeError(f"missing {label} identity")
    if (
        Path(str(left.get("path", ""))).expanduser().resolve()
        != Path(str(right.get("path", ""))).expanduser().resolve()
        or left.get("sha256") != right.get("sha256")
        or left.get("content_sha256") != right.get("content_sha256")
        or len(str(left.get("sha256", ""))) != 64
        or len(str(left.get("content_sha256", ""))) != 64
    ):
        raise RuntimeError(f"{label} identity mismatch")


def _verify_execution_provenance(
    provenance: Mapping[str, Any],
    *,
    retrain_index_path: Path,
    retrain_index_binding: Mapping[str, Any],
    retrain_index_content_sha256: str,
) -> dict[str, Any]:
    """Validate the merge's nested 30/30 supervisor evidence end to end."""

    execution = provenance.get("retrain_execution")
    if (
        provenance.get("retrain_execution_attestation_required") is not True
        or not isinstance(execution, Mapping)
        or execution.get("required") is not True
        or execution.get("attestation_content_verified") is not True
        or execution.get("campaign_index_30_of_30_verified") is not True
        or execution.get("runtime_seal_live_rehashed") is not True
        or execution.get("supervisor_live_rehashed") is not True
    ):
        raise RuntimeError("merged index does not require complete retrain execution evidence")

    nested_attestation = execution.get("execution_attestation")
    legacy_attestation = provenance.get("retrain_execution_attestation")
    _same_binding(
        nested_attestation,
        legacy_attestation,
        label="nested/legacy execution attestation",
        require_content=True,
    )
    if not isinstance(nested_attestation, Mapping):  # narrowed by _same_binding
        raise RuntimeError("missing nested execution attestation")
    attestation_path = Path(str(nested_attestation.get("path", ""))).expanduser().resolve()
    live_attestation_binding = _verify_file_binding(
        nested_attestation, "retrain execution attestation"
    )
    attestation, loaded_binding = _load(
        attestation_path, "retrain execution attestation"
    )
    _same_binding(
        nested_attestation,
        {**loaded_binding, "content_sha256": attestation.get("content_sha256")},
        label="execution attestation live",
        require_content=True,
    )
    if (
        attestation.get("schema_version") != 1
        or attestation.get("classification")
        != "sealed_non_test_proposer_execution_attestation"
        or attestation.get("outer_test_opened") is not False
        or int(attestation.get("outer_test_record_count", -1)) != 0
        or attestation.get("commercial_claim_authorized") is not False
        or int(attestation.get("expected_units", -1)) != 30
        or int(attestation.get("completed_units", -1)) != 30
        or int(attestation.get("invocations_this_resume", -1)) != 30
        or attestation.get("one_new_unit_per_invocation") is not True
        or attestation.get("runtime_seal_verified_before_and_after_every_invocation")
        is not True
        or canonical_content_sha256(attestation)
        != attestation.get("content_sha256")
    ):
        raise RuntimeError("retrain execution attestation is incomplete or tampered")

    campaign_index = attestation.get("campaign_index")
    expected_retrain_index = {
        **retrain_index_binding,
        "content_sha256": retrain_index_content_sha256,
    }
    _same_binding(
        campaign_index,
        expected_retrain_index,
        label="execution receipt retrain index",
        require_content=True,
    )
    if not isinstance(campaign_index, Mapping):
        raise RuntimeError("execution receipt lacks retrain index")
    if Path(str(campaign_index.get("path", ""))).expanduser().resolve() != (
        retrain_index_path.expanduser().resolve()
    ):
        raise RuntimeError("execution receipt points at another retrain index")

    completion = execution.get("completion_evidence")
    receipt_invocations = int(attestation.get("invocations_this_resume", -1))
    if (
        not isinstance(completion, Mapping)
        or int(completion.get("expected_units", -1)) != 30
        or int(completion.get("completed_units", -1)) != 30
        or int(completion.get("invocations_this_resume", -1))
        != receipt_invocations
        or completion.get("single_supervisor_execution_covered_all_units") is not True
        or completion.get("one_new_unit_per_invocation") is not True
        or completion.get("runtime_seal_verified_before_and_after_every_invocation")
        is not True
    ):
        raise RuntimeError("merged execution completion evidence is not 30/30")
    _same_binding(
        completion.get("campaign_index"),
        campaign_index,
        label="merged completion/retrain index",
        require_content=True,
    )

    receipt_supervisor = attestation.get("supervisor")
    nested_supervisor = execution.get("supervisor")
    _same_binding(
        receipt_supervisor,
        nested_supervisor,
        label="execution supervisor",
    )
    live_supervisor = _verify_file_binding(
        nested_supervisor, "execution supervisor"
    )
    canonical_supervisor = _verify_file_binding(
        {
            "path": str(CANONICAL_EXECUTION_SUPERVISOR.resolve()),
            "sha256": sha256_file(CANONICAL_EXECUTION_SUPERVISOR),
            "bytes": CANONICAL_EXECUTION_SUPERVISOR.stat().st_size,
        },
        "canonical execution supervisor",
    )
    _same_binding(
        nested_supervisor,
        canonical_supervisor,
        label="canonical execution supervisor",
    )
    if execution.get("canonical_supervisor_and_unit_command_verified") is not True:
        raise RuntimeError("merged execution did not verify canonical command provenance")

    receipt_seal = attestation.get("runtime_input_seal")
    nested_seal = execution.get("authoritative_runtime_input_seal")
    _same_binding(
        receipt_seal,
        nested_seal,
        label="authoritative execution runtime seal",
        require_content=True,
    )
    if not isinstance(receipt_seal, Mapping) or not isinstance(nested_seal, Mapping):
        raise RuntimeError("missing authoritative execution runtime seal")
    # The supervisor receipt is authoritative.  The merged nested copy must
    # match it, but never gets to choose a different filename by convention.
    live_seal = _verify_runtime_seal(
        Path(str(receipt_seal.get("path", ""))), expected_phase="prelaunch"
    )
    _same_binding(
        nested_seal,
        live_seal,
        label="authoritative runtime seal live",
        require_content=True,
    )
    try:
        expected_verified_files = int(receipt_seal.get("verified_files", -1))
        nested_verified_files = int(nested_seal.get("verified_files", -2))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("runtime seal verified-file evidence is malformed") from exc
    if (
        expected_verified_files != int(live_seal["verified_files"])
        or nested_verified_files != int(live_seal["verified_files"])
        or nested_seal.get("payloads_rehashed_during_merge") is not True
    ):
        raise RuntimeError("runtime seal live verification evidence mismatch")

    supersession = execution.get("supersession_note")
    verified_supersession: dict[str, Any] | None = None
    if supersession is not None:
        if not isinstance(supersession, Mapping):
            raise RuntimeError("execution supersession note binding is malformed")
        verified_supersession = _verify_file_binding(
            supersession, "execution supersession note"
        )
        if (
            supersession.get("classification")
            != "non_test_proposer_execution_runtime_seal_supersession"
            or int(supersession.get("superseded_runtime_seal_count", 0)) < 1
        ):
            raise RuntimeError("execution supersession note summary is invalid")
        _same_identity(
            supersession.get("selected_runtime_seal"),
            nested_seal,
            label="supersession authoritative seal",
        )

    return {
        "required": True,
        "execution_attestation": {
            **live_attestation_binding,
            "content_sha256": str(attestation["content_sha256"]),
            "classification": str(attestation["classification"]),
        },
        "authoritative_runtime_input_seal": live_seal,
        "supervisor": live_supervisor,
        "completion_evidence": dict(completion),
        "supersession_note": (
            {**verified_supersession, "classification": supersession["classification"]}
            if verified_supersession is not None and isinstance(supersession, Mapping)
            else None
        ),
        "merged_nested_binding_verified": True,
        "execution_attestation_live_rehashed": True,
        "authoritative_runtime_seal_live_rehashed": True,
        "supervisor_live_rehashed": True,
        "retrain_index_30_of_30_verified": True,
        "canonical_supervisor_and_unit_command_verified": True,
    }


def build_attestation(
    *,
    main_index_path: Path,
    retrain_index_path: Path,
    merged_index_path: Path,
    main_runtime_seal: Path,
) -> dict[str, Any]:
    main, main_binding, main_records = _validate_index(
        main_index_path, expected_records=90, label="main proposer index"
    )
    retrain, retrain_binding, retrain_records = _validate_index(
        retrain_index_path, expected_records=30, label="f3/f4 retrain proposer index"
    )
    merged, merged_binding, merged_records = _validate_index(
        merged_index_path, expected_records=90, label="merged proposer index"
    )
    if merged.get("merge_classification") != (
        "retrospective_current_source_uniform_90_unit_proposer_index"
    ):
        raise RuntimeError("merged index lacks current-source merge classification")
    provenance = merged.get("merge_provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("merged index lacks merge provenance")
    source_indexes = provenance.get("source_indexes")
    if not isinstance(source_indexes, Mapping):
        raise RuntimeError("merged index lacks bound source indexes")
    plan_sources: dict[str, dict[str, Any]] = {}
    for provenance_name, output_name, source_index in (
        ("full_split_authority_plan", "full_plan", main),
        ("retrain_plan", "retrain_plan", retrain),
    ):
        raw_plan = provenance.get(provenance_name)
        if not isinstance(raw_plan, Mapping):
            raise RuntimeError(f"merged index lacks plan binding: {provenance_name}")
        _same_identity(
            raw_plan,
            source_index.get("campaign_plan"),
            label=f"merged/source-index plan {provenance_name}",
        )
        live_plan = _verify_file_binding(raw_plan, f"{provenance_name} file")
        plan_document, loaded_plan_binding = _load(
            Path(str(raw_plan.get("path", ""))), provenance_name
        )
        if (
            canonical_content_sha256(plan_document)
            != plan_document.get("content_sha256")
        ):
            raise RuntimeError(f"{provenance_name} content hash mismatch")
        _same_binding(
            raw_plan,
            {
                **loaded_plan_binding,
                "content_sha256": plan_document["content_sha256"],
            },
            label=f"{provenance_name} live binding",
            require_content=True,
        )
        plan_sources[output_name] = {
            **live_plan,
            "content_sha256": str(plan_document["content_sha256"]),
        }
    for name, expected_binding, source in (
        ("main", main_binding, main),
        ("current_source_retrain_f34", retrain_binding, retrain),
    ):
        bound = source_indexes.get(name)
        _same_binding(
            bound,
            {**expected_binding, "content_sha256": source["content_sha256"]},
            label=f"merged index source binding: {name}",
            require_content=True,
        )
        expected_folds = sorted(MAIN_FOLDS if name == "main" else RETRAIN_FOLDS)
        expected_units = 60 if name == "main" else 30
        if (
            not isinstance(bound, Mapping)
            or bound.get("selected_outer_folds") != expected_folds
            or int(bound.get("selected_units", -1)) != expected_units
        ):
            raise RuntimeError(f"merged index source selection mismatch: {name}")
    selected_counts = {"main": 0, "current_source_retrain_f34": 0}
    for key, record in merged_records.items():
        if key[0] in MAIN_FOLDS:
            expected = main_records.get(key)
            source = "main"
        elif key[0] in RETRAIN_FOLDS:
            expected = retrain_records.get(key)
            source = "current_source_retrain_f34"
        else:
            raise RuntimeError(f"merged index contains invalid outer fold: {key[0]}")
        if expected != record:
            raise RuntimeError(f"merged record differs from declared source: {key}")
        selected_counts[source] += 1
    if selected_counts != {"main": 60, "current_source_retrain_f34": 30}:
        raise RuntimeError("merged index is not the exact audited 60+30 cover")

    main_seal = _verify_runtime_seal(main_runtime_seal, expected_phase="post_launch")
    execution = _verify_execution_provenance(
        provenance,
        retrain_index_path=retrain_index_path,
        retrain_index_binding=retrain_binding,
        retrain_index_content_sha256=str(retrain["content_sha256"]),
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "classification": "retrospective_nested_proposer_governance_attestation",
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
        "outer_test_artifact_evaluated_during_non_test_campaigns": False,
        "target_metric_used_for_proposer_fit_or_selection": False,
        "verified_documents": {
            "main_index": {**main_binding, "content_sha256": main["content_sha256"]},
            "f3_f4_retrain_index": {
                **retrain_binding,
                "content_sha256": retrain["content_sha256"],
            },
            "merged_index": {
                **merged_binding,
                "content_sha256": merged["content_sha256"],
            },
            "retrain_execution_attestation": dict(
                execution["execution_attestation"]
            ),
            "retrain_execution_supervisor": dict(execution["supervisor"]),
        },
        "verified_retrain_impact_sources": {
            **plan_sources,
            "main_index": {
                **main_binding,
                "content_sha256": main["content_sha256"],
            },
            "retrain_index": {
                **retrain_binding,
                "content_sha256": retrain["content_sha256"],
            },
        },
        "runtime_input_attestations": {
            "main_post_launch": main_seal,
            "f3_f4_retrain_authoritative_prelaunch": dict(
                execution["authoritative_runtime_input_seal"]
            ),
        },
        "execution_provenance": execution,
        "execution_provenance_complete": True,
        "independent_code_path_audit": {
            "literal_outer_test_label_file_unopened_claim_valid": False,
            "excluded_outer_metadata_target_columns_materialized_host_side": True,
            "excluded_outer_rows_consumed_by_scaler_fit": False,
            "excluded_outer_rows_consumed_by_training_loss": False,
            "excluded_outer_rows_consumed_by_checkpoint_selection": False,
            "outer_test_targets_consulted_for_model_fit_or_selection": False,
            "accurate_boundary": (
                "excluded outer target/QC columns were resident after the legacy full-cache "
                "metadata load, but excluded rows were not consumed by scaler fitting, loss, "
                "validation checkpoint selection, or prediction metrics"
            ),
        },
        "primary_plan_provenance_gap": {
            "full_runtime_import_and_cache_payload_closure_in_primary_plan": False,
            "supplemental_runtime_and_cache_payload_seal_present": True,
            "main_seal_is_post_launch_attestation": True,
            "f3_f4_retrain_seal_is_authoritative_prelaunch_attestation": True,
            "legacy_unit_confirmed_before_current_train_source": (
                "seed_20260828/outer_3/inner_pred_0"
            ),
            "mitigation": (
                "all 30 folds-3/4 proposer units were retrained and selected from the "
                "separately sealed current-source campaign"
            ),
            "selected_current_source_cover": selected_counts,
        },
        "resume_and_gpu_admission_audit": {
            "legacy_resume_transaction_is_fully_atomic": False,
            "legacy_gpu_lock_is_globally_unified_across_all_prior_scripts": False,
            "observed_gpu_overlap_during_audited_campaign": False,
            "interpretation": "engineering deviations are disclosed, not converted into a commercial claim",
        },
        "audit_scope": (
            "non-test proposer provenance and leakage boundary only; no outer-test target, "
            "accuracy metric, robustness metric, or prospective evidence is assessed here"
        ),
    }
    result["content_sha256"] = canonical_content_sha256(result)
    return result


def write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or target.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite governance attestation: {target}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-index", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--retrain-index", type=Path, default=DEFAULT_RETRAIN)
    parser.add_argument("--merged-index", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--main-runtime-seal", type=Path, default=DEFAULT_MAIN_SEAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_attestation(
        main_index_path=args.main_index,
        retrain_index_path=args.retrain_index,
        merged_index_path=args.merged_index,
        main_runtime_seal=args.main_runtime_seal,
    )
    write_immutable(args.output, result)
    print(
        json.dumps(
            {
                "status": "attested",
                "output": str(args.output.expanduser().resolve()),
                "content_sha256": result["content_sha256"],
                "commercial_claim_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
