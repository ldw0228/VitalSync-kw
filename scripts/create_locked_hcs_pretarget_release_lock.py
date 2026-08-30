#!/usr/bin/env python3
"""Create the final target-free release lock for locked HCS evaluation.

The lock is a fail-closed authorization boundary, not an evaluation result.  It
is published only after the primary 18-unit prediction graph, all 126 fixed
radar-mask conditions (including full-mask byte parity), and the 18-unit
uncertainty/calibration graph have been revalidated without opening a target.
The already-frozen primary evaluation and deployment benchmark specifications
are also byte-bound.  Canonical targets and the evaluation lock must not exist
while this document is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_locked_proposer_deployment_v3 as DEPLOYMENT  # noqa: E402
import build_locked_hcs_targets as TARGETS  # noqa: E402
import evaluate_locked_hcs_oof as PRIMARY_EVALUATION  # noqa: E402
import evaluate_locked_hcs_release_readiness as RELEASE_READINESS  # noqa: E402
import freeze_locked_hcs_uncertainty_evaluation_spec as UNCERTAINTY_SPEC  # noqa: E402
import run_locked_hcs_radar_mask_campaign as RADAR_MASKS  # noqa: E402
import run_runtime_sealed_hcs_radar_masks as RADAR_RUNTIME_GUARD  # noqa: E402
import run_runtime_sealed_locked_hcs_oof as OOF_RUNTIME_GUARD  # noqa: E402
import seal_fixed_i3_pretest_completion as FIXED_COMPLETION  # noqa: E402
import seal_locked_hcs_uncertainty_inputs as UNCERTAINTY  # noqa: E402
import seal_runtime_inputs as RUNTIME_SEAL  # noqa: E402


SCHEMA_VERSION = 1
EXPECTED_PRIMARY_UNITS = 18
EXPECTED_MASK_UNITS = 126
EXPECTED_FULL_MASK_UNITS = 18

DEFAULT_PRIMARY_ROOT = (
    PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
)
DEFAULT_MASK_ROOT = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_radar_masks"
)
DEFAULT_UNCERTAINTY_SEAL = DEFAULT_PRIMARY_ROOT / "uncertainty_inputs_seal.json"
DEFAULT_EVALUATION_SPEC = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/locked_primary_evaluation_spec.json"
)
DEFAULT_UNCERTAINTY_EVALUATION_SPEC = UNCERTAINTY_SPEC.DEFAULT_OUTPUT
DEFAULT_DEPLOYMENT_SPEC = (
    PROJECT_ROOT / "artifacts/benchmarks/locked_proposer_deployment/freeze_spec_v3.json"
)
DEFAULT_RELEASE_READINESS_SPEC = RELEASE_READINESS.DEFAULT_SPEC
DEFAULT_RELEASE_LOCK = DEFAULT_PRIMARY_ROOT / "pretarget_release_lock.json"
DEFAULT_TARGET = DEFAULT_PRIMARY_ROOT / "canonical_locked_hcs_targets.npz"
DEFAULT_TARGET_RECEIPT = (
    DEFAULT_PRIMARY_ROOT / "canonical_locked_hcs_targets_receipt.json"
)
DEFAULT_EVALUATION_LOCK = DEFAULT_PRIMARY_ROOT / "evaluation_lock.json"
DEFAULT_JOINED = DEFAULT_PRIMARY_ROOT / "locked_hcs_oof_joined.npz"
DEFAULT_RELEASE_RECEIPT = (
    DEFAULT_PRIMARY_ROOT / "canonical_locked_hcs_targets_release_receipt.json"
)
DEFAULT_FIXED_I3_RUNTIME_SEAL = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/current_source_merged/fixed_i3_pretest_runtime_seal.json"
)
DEFAULT_FIXED_RUNTIME_COMPLETION = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_fixed_i3_pretest/fixed_runtime_completion_attestation.json"
)
DEFAULT_POSTLOCK_RUNTIME_GUARD = (
    DEFAULT_PRIMARY_ROOT / "postlock_runtime_guard_attestation.json"
)
DEFAULT_RADAR_MASK_RUNTIME_GUARD = (
    DEFAULT_MASK_ROOT / "radar_mask_runtime_guard_attestation.json"
)


class PretargetReleaseLockError(RuntimeError):
    """A target-free boundary is incomplete, inconsistent, or already crossed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PretargetReleaseLockError(
            f"bound input must be a regular non-symlink file: {resolved}"
        )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        Path(str(left.get("path", ""))).resolve()
        == Path(str(right.get("path", ""))).resolve()
        and str(left.get("sha256", "")) == str(right.get("sha256", ""))
        and int(left.get("bytes", -1)) == int(right.get("bytes", -2))
    )


def _verify_binding(
    raw: Any, *, relative_to: Path, label: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise PretargetReleaseLockError(f"missing binding: {label}")
    path = Path(str(raw.get("path", ""))).expanduser()
    path = path.resolve() if path.is_absolute() else (relative_to / path).resolve()
    expected = str(raw.get("sha256", ""))
    if len(expected) != 64:
        raise PretargetReleaseLockError(f"invalid binding hash: {label}")
    observed = bind_file(path)
    if observed["sha256"] != expected or (
        "bytes" in raw and observed["bytes"] != int(raw["bytes"])
    ):
        raise PretargetReleaseLockError(f"binding changed: {label}")
    return observed


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PretargetReleaseLockError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise PretargetReleaseLockError(f"{label} contains non-finite number {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PretargetReleaseLockError(f"invalid {label}: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise PretargetReleaseLockError(f"{label} root must be an object: {path}")
    return value


def _atomic_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise PretargetReleaseLockError(
                f"release lock appeared concurrently: {destination}"
            ) from exc
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_absent(paths: Sequence[Path]) -> None:
    present = [str(path.expanduser().resolve()) for path in paths if path.expanduser().resolve().exists()]
    if present:
        raise PretargetReleaseLockError(
            "pretarget release is forbidden after a target/evaluation artifact exists: "
            + ", ".join(present)
        )


def _primary_summary(primary_root: Path) -> dict[str, Any]:
    try:
        verified = TARGETS.verify_prediction_seal(primary_root)
    except Exception as exc:
        raise PretargetReleaseLockError(f"primary prediction boundary is invalid: {exc}") from exc
    units = verified.get("units")
    seeds = verified.get("seeds")
    if not isinstance(units, list) or len(units) != EXPECTED_PRIMARY_UNITS:
        raise PretargetReleaseLockError("primary validator did not return exactly 18 units")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise PretargetReleaseLockError("primary validator did not return three fixed seeds")
    expected_indices = np.asarray(verified["expected_indices"])
    fold_indices = verified.get("fold_indices")
    if not isinstance(fold_indices, Mapping) or set(map(int, fold_indices)) != set(range(6)):
        raise PretargetReleaseLockError("primary validator returned incomplete fold coverage")
    return {
        "classification": "primary_18_unit_predictions_revalidated",
        "unit_count": len(units),
        "folds": list(range(6)),
        "seeds": [int(seed) for seed in seeds],
        "rows_per_seed": int(len(expected_indices)),
        "expected_index_sha256": array_sha256(expected_indices),
        "predictions_seal": verified["predictions_seal"],
        "pretest_lock": verified["pretest_lock"],
        "inference_plan": verified["inference_plan"],
        "fixed_i3_pretest_index": verified["fixed_i3_pretest_index"],
        "rf_cache_manifest": verified["rf_cache_manifest"],
        "fold_assignments": verified["fold_assignments"],
        "effective_sources": verified["effective_sources"],
        "validated_unit_graph_sha256": canonical_json_sha256(units),
        "target_metadata_opened": False,
    }


def _mask_summary(mask_root: Path, primary: Mapping[str, Any]) -> dict[str, Any]:
    root = mask_root.expanduser().resolve()
    plan_path = root / "control/plan.json"
    preexecution_path = root / "control/preexecution_lock.json"
    complete_path = root / "complete_seal.json"
    try:
        plan = RADAR_MASKS._json(plan_path, "radar-mask plan")  # noqa: SLF001
        plan_binding = RADAR_MASKS.bind_file(plan_path)
        preexecution = RADAR_MASKS._json(  # noqa: SLF001
            preexecution_path, "radar-mask preexecution lock"
        )
        preexecution_binding = RADAR_MASKS.bind_file(preexecution_path)
    except Exception as exc:
        raise PretargetReleaseLockError(f"radar-mask controls are invalid: {exc}") from exc
    units = plan.get("units")
    expected_masks = [
        {"name": name, "pattern": list(pattern)}
        for name, pattern in RADAR_MASKS.MASKS.items()
    ]
    if (
        plan.get("schema_version") != 1
        or plan.get("classification") != "locked_hcs_seven_radar_mask_label_free_plan"
        or plan.get("folds") != list(range(6))
        or plan.get("radar_masks") != expected_masks
        or int(plan.get("primary_unit_count", -1)) != EXPECTED_PRIMARY_UNITS
        or int(plan.get("unit_count", -1)) != EXPECTED_MASK_UNITS
        or plan.get("target_or_label_artifact_bound") is not False
        or plan.get("evaluation_permitted_before_complete_seal") is not False
        or not isinstance(units, list)
        or len(units) != EXPECTED_MASK_UNITS
    ):
        raise PretargetReleaseLockError("radar-mask plan invariants are invalid")
    if not _same_binding(plan.get("primary", {}).get("predictions_seal", {}), primary["predictions_seal"]):
        raise PretargetReleaseLockError("radar-mask plan is bound to another primary seal")
    expected_preexecution = {
        "schema_version": 1,
        "classification": "locked_hcs_radar_mask_preexecution_input_seal",
        "plan": plan_binding,
        "primary": plan["primary"],
        "effective_sources": plan["effective_sources"],
        "rf_cache_manifest": plan["rf_cache_manifest"],
        "unit_count": EXPECTED_MASK_UNITS,
        "target_or_label_artifact_opened": False,
        "evaluation_authorized": False,
    }
    if preexecution != expected_preexecution:
        raise PretargetReleaseLockError("radar-mask preexecution lock differs from its plan")
    receipts: list[dict[str, Any]] = []
    try:
        for unit in units:
            RADAR_MASKS._validate_command_contract(unit, root)  # noqa: SLF001
            receipts.append(
                RADAR_MASKS._validate_receipt(  # noqa: SLF001
                    unit=unit, output_root=root, plan_binding=plan_binding
                )
            )
        expected_complete = RADAR_MASKS._complete_seal(  # noqa: SLF001
            output_root=root,
            plan_binding=plan_binding,
            preexecution_binding=preexecution_binding,
            plan=plan,
            receipts=receipts,
        )
        observed_complete = RADAR_MASKS._json(complete_path, "radar-mask complete seal")  # noqa: SLF001
    except Exception as exc:
        raise PretargetReleaseLockError(f"radar-mask 126-unit boundary is invalid: {exc}") from exc
    if observed_complete != expected_complete:
        raise PretargetReleaseLockError("radar-mask complete seal differs from 126 live receipts")
    full_receipts = [receipt for receipt in receipts if receipt.get("radar_mask") == "radars_123"]
    if len(full_receipts) != EXPECTED_FULL_MASK_UNITS:
        raise PretargetReleaseLockError("full-mask parity does not cover all 18 primary units")
    for receipt in full_receipts:
        comparison = receipt.get("radars_123_bit_exact_primary_comparison")
        if (
            not isinstance(comparison, Mapping)
            or comparison.get("required") is not True
            or comparison.get("performed") is not True
            or not isinstance(comparison.get("proposer_fields"), list)
            or not comparison.get("proposer_fields")
            or not isinstance(comparison.get("source_fields"), list)
            or not comparison.get("source_fields")
            or not isinstance(comparison.get("sealed_fields"), list)
            or not comparison.get("sealed_fields")
        ):
            raise PretargetReleaseLockError("a full-mask receipt lacks three-artifact parity")
    return {
        "classification": "all_126_radar_mask_units_revalidated",
        "unit_count": len(receipts),
        "full_mask_parity_unit_count": len(full_receipts),
        "all_seven_fixed_conditions_retained": True,
        "best_mask_selection_performed": False,
        "plan": plan_binding,
        "preexecution_lock": preexecution_binding,
        "complete_seal": bind_file(complete_path),
        "primary_predictions_seal": observed_complete["primary_predictions_seal"],
        "validated_receipt_content_sha256": canonical_json_sha256(
            [receipt["content_sha256"] for receipt in receipts]
        ),
        "target_or_label_artifact_opened": False,
    }


def _uncertainty_summary(
    uncertainty_seal_path: Path, primary: Mapping[str, Any]
) -> dict[str, Any]:
    seal_path = uncertainty_seal_path.expanduser().resolve()
    try:
        seal = UNCERTAINTY._validate_existing_seal(seal_path)  # noqa: SLF001
        calibration_path = Path(str(seal["pretest_calibration"]["path"])).resolve()
        calibration, calibration_binding = UNCERTAINTY._validate_calibration(  # noqa: SLF001
            calibration_path
        )
    except Exception as exc:
        raise PretargetReleaseLockError(f"uncertainty boundary is invalid: {exc}") from exc
    if not _same_binding(calibration_binding, seal["pretest_calibration"]):
        raise PretargetReleaseLockError("uncertainty seal/calibration binding mismatch")
    if not _same_binding(seal.get("predictions_seal", {}), primary["predictions_seal"]):
        raise PretargetReleaseLockError("uncertainty seal is bound to another primary seal")

    # Re-run the existing unit loader so raw/sealed point parity is proven now,
    # rather than trusting only the historical creation attestation.
    predictions = TARGETS._read_json(  # noqa: SLF001
        Path(primary["predictions_seal"]["path"]), "primary predictions seal"
    )
    source_units = predictions.get("units")
    if not isinstance(source_units, list) or len(source_units) != EXPECTED_PRIMARY_UNITS:
        raise PretargetReleaseLockError("uncertainty source topology is not 18 units")
    loaded: list[tuple[dict[str, np.ndarray], dict[str, Any]]] = []
    try:
        for unit in source_units:
            loaded.append(
                UNCERTAINTY._load_unit(  # noqa: SLF001
                    unit, root=Path(primary["predictions_seal"]["path"]).parent
                )
            )
    except Exception as exc:
        raise PretargetReleaseLockError(f"uncertainty raw/point parity failed: {exc}") from exc
    loaded.sort(key=lambda item: (int(item[1]["seed"]), int(item[1]["outer_fold"])))
    names = tuple(loaded[0][0])
    combined = {
        name: np.concatenate([arrays[name] for arrays, _ in loaded]) for name in names
    }
    archive_binding = seal["uncertainty_archive"]
    try:
        with np.load(archive_binding["path"], allow_pickle=False) as archive:
            if set(archive.files) != set(combined):
                raise PretargetReleaseLockError("uncertainty archive/live fields differ")
            for name, value in combined.items():
                observed = np.asarray(archive[name])
                if (
                    observed.dtype != value.dtype
                    or observed.shape != value.shape
                    or observed.tobytes(order="C") != value.tobytes(order="C")
                ):
                    raise PretargetReleaseLockError(
                        f"uncertainty archive/live parity differs: {name}"
                    )
    except (OSError, ValueError) as exc:
        raise PretargetReleaseLockError(f"cannot verify uncertainty archive: {exc}") from exc
    return {
        "classification": "uncertainty_18_unit_inputs_and_pretest_calibration_revalidated",
        "unit_count": len(loaded),
        "uncertainty_inputs_seal": bind_file(seal_path),
        "uncertainty_archive": archive_binding,
        "pretest_calibration": calibration_binding,
        "pretest_calibration_content_sha256": str(calibration.get("content_sha256", "")),
        "primary_predictions_seal": seal["predictions_seal"],
        "array_schema": seal["array_schema"],
        "live_array_graph_sha256": canonical_json_sha256(
            {name: array_sha256(value) for name, value in sorted(combined.items())}
        ),
        "target_or_label_artifact_opened": False,
    }


def _spec_summary(
    evaluation_spec: Path,
    uncertainty_evaluation_spec: Path,
    deployment_spec: Path,
    release_readiness_spec: Path,
    calibration_binding: Mapping[str, Any],
    calibration_content_sha256: str,
) -> dict[str, Any]:
    try:
        evaluation, evaluation_binding = PRIMARY_EVALUATION._load_evaluation_spec(  # noqa: SLF001
            evaluation_spec
        )
        uncertainty, uncertainty_binding = (
            UNCERTAINTY_SPEC.load_uncertainty_evaluation_spec(
                uncertainty_evaluation_spec,
                expected_primary_spec_path=evaluation_spec,
                expected_calibration_path=Path(str(calibration_binding["path"])),
            )
        )
        deployment, deployment_binding = DEPLOYMENT.load_freeze_spec(deployment_spec)
        readiness, readiness_binding = RELEASE_READINESS.load_release_readiness_spec(
            release_readiness_spec
        )
    except Exception as exc:
        raise PretargetReleaseLockError(f"frozen specification is invalid: {exc}") from exc
    primary_uncertainty = evaluation.get("uncertainty")
    relationship = uncertainty.get("protocol_relationship")
    bound = uncertainty.get("bound_inputs")
    if (
        not isinstance(primary_uncertainty, Mapping)
        or primary_uncertainty.get("role")
        != "diagnostic_ranking_only_not_calibrated_interval"
        or primary_uncertainty.get("calibration_fit_allowed") is not False
        or primary_uncertainty.get("threshold_fit_allowed") is not False
        or primary_uncertainty.get("model_or_candidate_selection_allowed") is not False
        or not isinstance(relationship, Mapping)
        or relationship.get("role")
        != "separate_secondary_retrospective_engineering_protocol"
        or relationship.get("primary_uncertainty_contract_overridden") is not False
        or relationship.get("primary_point_evaluation_or_gates_modified") is not False
        or relationship.get("primary_diagnostic_only_uncertainty_claim_preserved") is not True
        or relationship.get("secondary_interval_results_are_part_of_primary_evaluation")
        is not False
        or not isinstance(bound, Mapping)
        or not _same_binding(
            bound.get("unchanged_primary_evaluation_spec", {}), evaluation_binding
        )
        or not _same_binding(
            bound.get("completed_pretest_uncertainty_calibration", {}),
            calibration_binding,
        )
        or len(calibration_content_sha256) != 64
        or uncertainty.get("calibration_content_sha256")
        != calibration_content_sha256
    ):
        raise PretargetReleaseLockError(
            "secondary uncertainty specification does not preserve the separate diagnostic-only primary contract"
        )
    prospective = readiness.get("prospective_release_policy")
    if (
        readiness.get("classification")
        != "locked_hcs_release_readiness_aggregate_specification"
        or readiness.get("target_or_target_bearing_artifact_opened") is not False
        or not isinstance(prospective, Mapping)
        or prospective.get("independent_prospective_cohort_present") is not False
        or prospective.get("commercial_release_ready_must_equal") is not False
    ):
        raise PretargetReleaseLockError(
            "release-readiness specification does not preserve the prospective fail-closed policy"
        )
    return {
        "primary_evaluation": {
            "binding": evaluation_binding,
            "content_sha256": evaluation["content_sha256"],
            "post_target_prohibitions": evaluation["post_target_prohibitions"],
        },
        "secondary_uncertainty_evaluation": {
            "binding": uncertainty_binding,
            "content_sha256": uncertainty["content_sha256"],
            "calibration": dict(calibration_binding),
            "calibration_content_sha256": uncertainty["calibration_content_sha256"],
            "relationship": dict(relationship),
            "primary_diagnostic_only_contract_preserved": True,
            "primary_protocol_overridden": False,
        },
        "deployment_benchmark": {
            "binding": deployment_binding,
            "content_sha256": deployment["content_sha256"],
            "target_or_label_artifact_consulted": False,
        },
        "release_readiness": {
            "binding": readiness_binding,
            "content_sha256": readiness["content_sha256"],
            "target_or_target_bearing_artifact_opened": False,
            "commercial_release_ready_must_equal": False,
            "prospective_confirmation_required": True,
        },
    }


def _attestation_content(document: Mapping[str, Any], *, label: str) -> str:
    expected = str(document.get("content_sha256", ""))
    payload = dict(document)
    payload.pop("content_sha256", None)
    if len(expected) != 64 or canonical_json_sha256(payload) != expected:
        raise PretargetReleaseLockError(f"{label} content hash mismatch")
    return expected


def _runtime_guard_summary(
    runtime_seal_path: Path,
    completion_path: Path,
    postlock_guard_path: Path,
    radar_mask_guard_path: Path,
    primary: Mapping[str, Any],
    masks: Mapping[str, Any],
    *,
    enforce_target_free: bool = True,
) -> dict[str, Any]:
    runtime_path = runtime_seal_path.expanduser().resolve()
    completion_resolved = completion_path.expanduser().resolve()
    guard_resolved = postlock_guard_path.expanduser().resolve()
    mask_guard_resolved = radar_mask_guard_path.expanduser().resolve()
    try:
        verification = RUNTIME_SEAL.verify(runtime_path)
        runtime_document = _read_json(runtime_path, "fixed-i3 runtime input seal")
    except Exception as exc:
        raise PretargetReleaseLockError(f"fixed-i3 runtime input seal is invalid: {exc}") from exc
    context = runtime_document.get("fixed_i3_context")
    if (
        runtime_document.get("schema_version") != 1
        or runtime_document.get("classification")
        != "supplemental_runtime_input_byte_inventory"
        or runtime_document.get("attestation_phase") != "prelaunch"
        or runtime_document.get("post_launch_attestation") is not False
        or not isinstance(context, Mapping)
        or context.get("classification")
        != "retrospective_fixed_i3_pretest_runtime_input_context"
        or context.get("outer_test_opened") is not False
        or context.get("target_or_evaluation_artifact_accessed") is not False
        or int(context.get("proposer_matrix_groups", -1)) != EXPECTED_PRIMARY_UNITS
        or int(context.get("proposer_matrix_units", -1)) != 90
        or verification.get("content_sha256") != runtime_document.get("content_sha256")
    ):
        raise PretargetReleaseLockError("fixed-i3 runtime input seal invariants are invalid")
    trees = runtime_document.get("input_trees")
    if not isinstance(trees, list) or len(trees) < 2:
        raise PretargetReleaseLockError(
            "fixed-i3 runtime seal does not close both RF/SVD payload trees"
        )
    runtime_binding = bind_file(runtime_path)
    try:
        # The producer verifier replays the exact immutable status-prefix chain
        # 1..18, validates every fixed unit output tree/artifact, and rehashes
        # the RF/SVD payload closure.  A top-level completion boolean is never
        # accepted as evidence by itself.
        completed = FIXED_COMPLETION.verify_completion_attestation(
            completion_resolved,
            expected_runtime_seal=runtime_path,
            reverify_payload=True,
        )
        completion = completed["document"]
        completion_binding = completed["binding"]
        pretest_binding = FIXED_COMPLETION.verify_binding(
            completion.get("pretest_index"),
            relative_to=completion_resolved.parent,
            label="fixed completion pretest index",
        )
        closure = OOF_RUNTIME_GUARD.verify_closure(
            runtime_input_seal=runtime_path,
            completion_attestation=completion_resolved,
            pretest_index=Path(pretest_binding["path"]),
        )
        guarded = OOF_RUNTIME_GUARD.verify_guard_attestation(
            guard_resolved,
            expected_runtime_seal=runtime_path,
            expected_completion=completion_resolved,
            reverify_closure=enforce_target_free,
        )
    except Exception as exc:
        raise PretargetReleaseLockError(
            f"fixed/locked-OOF producer runtime verifier failed: {exc}"
        ) from exc
    evidence = completion.get("fixed_campaign_execution_evidence")
    if (
        completion.get("commercial_claim_authorized") is not False
        or not isinstance(evidence, Mapping)
        or evidence.get("completed_prefixes_revalidated") != list(range(1, 19))
        or int(evidence.get("status_snapshot_count", -1)) < EXPECTED_PRIMARY_UNITS
        or not _same_binding(closure["runtime_input_seal"], runtime_binding)
        or not _same_binding(closure["completion_attestation"], completion_binding)
        or not _same_binding(closure["pretest_index"], pretest_binding)
    ):
        raise PretargetReleaseLockError(
            "fixed-i3 status-prefix/runtime closure evidence is incomplete"
        )

    guard = guarded["document"]
    guard_binding = guarded["binding"]
    guard_plan = FIXED_COMPLETION.verify_binding(
        guard.get("locked_oof_plan"),
        relative_to=guard_resolved.parent,
        label="postlock locked OOF plan",
    )
    guard_pretest = FIXED_COMPLETION.verify_binding(
        guard.get("pretest_lock"),
        relative_to=guard_resolved.parent,
        label="postlock pretest lock",
    )
    guard_predictions = FIXED_COMPLETION.verify_binding(
        guard.get("predictions_seal"),
        relative_to=guard_resolved.parent,
        label="postlock predictions seal",
    )
    if (
        guard.get("commercial_claim_authorized") is not False
        or not _same_binding(guard_plan, primary["inference_plan"])
        or not _same_binding(guard_pretest, primary["pretest_lock"])
        or not _same_binding(guard_predictions, primary["predictions_seal"])
    ):
        raise PretargetReleaseLockError(
            "postlock producer guard is bound to another primary prediction graph"
        )

    # The producer's public guard verifier validates receipt hashes.  Traverse
    # the same 18 receipts through its stronger live-unit verifier as well, so
    # each receipt's derived lock and prediction binding is compared with the
    # bytes currently resident at the canonical unit path.
    try:
        oof_plan = OOF_RUNTIME_GUARD._validated_plan(  # noqa: SLF001
            Path(guard_plan["path"]), guard_resolved.parent
        )
        oof_order = OOF_RUNTIME_GUARD._ordered_units(oof_plan)  # noqa: SLF001
        oof_receipts_raw = guard.get("unit_runtime_guard_receipts")
        if not isinstance(oof_receipts_raw, list) or len(oof_receipts_raw) != EXPECTED_PRIMARY_UNITS:
            raise PretargetReleaseLockError("postlock guard lacks 18 unit receipts")
        oof_receipt_records: list[dict[str, Any]] = []
        for position, (raw, key) in enumerate(
            zip(oof_receipts_raw, oof_order, strict=True), start=1
        ):
            binding = FIXED_COMPLETION.verify_binding(
                raw,
                relative_to=guard_resolved.parent,
                label=f"postlock runtime receipt {position}",
            )
            receipt = OOF_RUNTIME_GUARD._validate_unit_receipt(  # noqa: SLF001
                Path(binding["path"]),
                position=position,
                key=key,
                output_root=guard_resolved.parent,
                closure=closure,
            )
            oof_receipt_records.append(
                {
                    "position": position,
                    "outer_fold": key[0],
                    "seed": key[1],
                    "receipt": binding,
                    "receipt_content_sha256": receipt["content_sha256"],
                    "derived_lock": receipt["derived_lock"],
                    "prediction": receipt["prediction"],
                }
            )
    except Exception as exc:
        raise PretargetReleaseLockError(
            f"locked-OOF 18-unit live receipt traversal failed: {exc}"
        ) from exc

    # Before target access use the radar producer's public verifier, which
    # itself traverses all 126 receipts and compares every campaign receipt,
    # proposer/raw/sealed output binding with the live canonical files.
    if enforce_target_free:
        try:
            radar_verified = RADAR_RUNTIME_GUARD.verify_radar_guard_attestation(
                mask_guard_resolved,
                primary_output_root=guard_resolved.parent,
                runtime_input_seal=runtime_path,
                completion_attestation=completion_resolved,
                pretest_index=Path(pretest_binding["path"]),
                postlock_guard=guard_resolved,
            )
            mask_guard = radar_verified["document"]
            mask_guard_binding = radar_verified["binding"]
        except Exception as exc:
            raise PretargetReleaseLockError(
                f"radar-mask producer runtime verifier failed: {exc}"
            ) from exc
    else:
        # Idempotent post-target validation cannot call a producer verifier that
        # deliberately requires targets absent.  It still traverses the exact
        # same 126 receipt/output validator below; only that absence assertion
        # is skipped after the already-authorized create-once publication.
        mask_guard = _read_json(
            mask_guard_resolved, "radar-mask runtime guard attestation"
        )
        mask_guard_binding = bind_file(mask_guard_resolved)
        if (
            mask_guard.get("schema_version") != 1
            or mask_guard.get("classification")
            != "locked_hcs_radar_mask_runtime_guard_attestation"
            or int(mask_guard.get("completed_units", -1)) != EXPECTED_MASK_UNITS
            or mask_guard.get("runtime_seal_verified_before_and_after_every_unit") is not True
            or mask_guard.get("postlock_guard_verified_before_and_after_every_unit") is not True
            or mask_guard.get("target_artifact_opened") is not False
            or mask_guard.get("commercial_claim_authorized") is not False
            or RADAR_RUNTIME_GUARD.primary_guard.canonical_sha256(mask_guard)
            != mask_guard.get("content_sha256")
        ):
            raise PretargetReleaseLockError("radar-mask runtime guard is invalid")

    try:
        mask_plan_binding = FIXED_COMPLETION.verify_binding(
            mask_guard.get("radar_mask_plan"),
            relative_to=mask_guard_resolved.parent,
            label="radar-mask runtime plan",
        )
        mask_complete_binding = FIXED_COMPLETION.verify_binding(
            mask_guard.get("complete_seal"),
            relative_to=mask_guard_resolved.parent,
            label="radar-mask complete seal",
        )
        if not _same_binding(mask_complete_binding, masks["complete_seal"]):
            raise PretargetReleaseLockError(
                "radar-mask runtime guard binds another complete seal"
            )
        mask_plan = RADAR_RUNTIME_GUARD._validated_plan(  # noqa: SLF001
            Path(mask_plan_binding["path"]), mask_guard_resolved.parent
        )
        radar_receipts_raw = mask_guard.get("unit_runtime_guard_receipts")
        if not isinstance(radar_receipts_raw, list) or len(radar_receipts_raw) != EXPECTED_MASK_UNITS:
            raise PretargetReleaseLockError("radar-mask guard lacks 126 unit receipts")
        guards = {**closure, "postlock_guard": guard_binding}
        radar_receipt_records: list[dict[str, Any]] = []
        for position, (raw, unit) in enumerate(
            zip(radar_receipts_raw, mask_plan["units"], strict=True), start=1
        ):
            binding = FIXED_COMPLETION.verify_binding(
                raw,
                relative_to=mask_guard_resolved.parent,
                label=f"radar-mask runtime receipt {position}",
            )
            receipt = RADAR_RUNTIME_GUARD._validate_receipt(  # noqa: SLF001
                Path(binding["path"]),
                position=position,
                unit=unit,
                output_root=mask_guard_resolved.parent,
                guards=guards,
            )
            radar_receipt_records.append(
                {
                    "position": position,
                    "unit_id": unit["unit_id"],
                    "receipt": binding,
                    "receipt_content_sha256": receipt["content_sha256"],
                    "outputs": receipt["outputs"],
                }
            )
    except Exception as exc:
        raise PretargetReleaseLockError(
            f"radar-mask 126-unit live receipt traversal failed: {exc}"
        ) from exc

    return {
        "classification": "fixed_i3_and_postlock_runtime_payload_closure_revalidated",
        "runtime_input_seal": runtime_binding,
        "runtime_input_seal_content_sha256": runtime_document["content_sha256"],
        "runtime_verified_files": int(verification["verified_files"]),
        "rf_svd_payload_tree_count": len(trees),
        "fixed_pretest_completion_attestation": completion_binding,
        "fixed_pretest_completion_content_sha256": completion["content_sha256"],
        "fixed_status_snapshot_count": int(evidence["status_snapshot_count"]),
        "fixed_completed_prefixes_revalidated": list(range(1, 19)),
        "fixed_status_snapshot_graph_sha256": evidence[
            "status_snapshot_graph_sha256"
        ],
        "fixed_unit_payload_count": len(completion["unit_payloads"]),
        "postlock_runtime_guard_attestation": guard_binding,
        "postlock_runtime_guard_content_sha256": guard["content_sha256"],
        "postlock_live_receipt_count": len(oof_receipt_records),
        "postlock_live_receipt_graph_sha256": canonical_json_sha256(
            oof_receipt_records
        ),
        "radar_mask_runtime_guard_attestation": mask_guard_binding,
        "radar_mask_runtime_guard_content_sha256": mask_guard["content_sha256"],
        "radar_mask_live_receipt_count": len(radar_receipt_records),
        "radar_mask_live_receipt_graph_sha256": canonical_json_sha256(
            radar_receipt_records
        ),
        "producer_verifiers": {
            "fixed_completion": "verify_completion_attestation",
            "locked_oof": "verify_guard_attestation_plus_live_unit_receipts",
            "radar_masks": "verify_radar_guard_attestation_plus_live_unit_receipts",
        },
        "completed_primary_units": EXPECTED_PRIMARY_UNITS,
        "completed_radar_mask_units": EXPECTED_MASK_UNITS,
        "target_artifact_opened": False,
    }


def _source_bindings() -> dict[str, Any]:
    sources = {
        "release_lock_creator": Path(__file__),
        "released_target_builder": SCRIPT_DIR / "build_locked_hcs_targets_after_release_lock.py",
        "canonical_target_builder": Path(TARGETS.__file__),
        "primary_evaluation_contract": Path(PRIMARY_EVALUATION.__file__),
        "radar_mask_contract": Path(RADAR_MASKS.__file__),
        "uncertainty_contract": Path(UNCERTAINTY.__file__),
        "deployment_benchmark_contract": Path(DEPLOYMENT.__file__),
        "uncertainty_evaluation_spec_contract": Path(UNCERTAINTY_SPEC.__file__),
        "release_readiness_spec_contract": Path(RELEASE_READINESS.__file__),
        "runtime_seal_contract": Path(RUNTIME_SEAL.__file__),
        "fixed_completion_producer_verifier": Path(FIXED_COMPLETION.__file__),
        "locked_oof_runtime_guard_producer_verifier": Path(OOF_RUNTIME_GUARD.__file__),
        "radar_mask_runtime_guard_producer_verifier": Path(
            RADAR_RUNTIME_GUARD.__file__
        ),
        "python_executable": Path(sys.executable),
    }
    return {name: bind_file(path) for name, path in sources.items()}


def _release_document(
    *,
    primary_root: Path,
    mask_root: Path,
    uncertainty_seal: Path,
    evaluation_spec: Path,
    uncertainty_evaluation_spec: Path,
    deployment_spec: Path,
    release_readiness_spec: Path,
    target_output: Path,
    target_receipt: Path,
    evaluation_lock: Path,
    joined_output: Path,
    release_receipt: Path,
    fixed_i3_runtime_seal: Path,
    fixed_runtime_completion: Path,
    postlock_runtime_guard: Path,
    radar_mask_runtime_guard: Path,
    enforce_target_free: bool = True,
) -> dict[str, Any]:
    primary = _primary_summary(primary_root)
    masks = _mask_summary(mask_root, primary)
    uncertainty = _uncertainty_summary(uncertainty_seal, primary)
    runtime = _runtime_guard_summary(
        fixed_i3_runtime_seal,
        fixed_runtime_completion,
        postlock_runtime_guard,
        radar_mask_runtime_guard,
        primary,
        masks,
        enforce_target_free=enforce_target_free,
    )
    specs = _spec_summary(
        evaluation_spec,
        uncertainty_evaluation_spec,
        deployment_spec,
        release_readiness_spec,
        uncertainty["pretest_calibration"],
        uncertainty["pretest_calibration_content_sha256"],
    )
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "locked_hcs_pretarget_release_lock",
        "status": "all_target_free_boundaries_complete",
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
        "canonical_target_build_authorized": True,
        "evaluation_authorized_before_canonical_target_join": False,
        "target_or_label_artifact_opened": False,
        "boundaries": {
            "primary_predictions": primary,
            "radar_masks": masks,
            "uncertainty": uncertainty,
            "runtime_payload_closure": runtime,
        },
        "frozen_specs": specs,
        "locations": {
            "primary_root": str(primary_root.expanduser().resolve()),
            "mask_root": str(mask_root.expanduser().resolve()),
            "uncertainty_seal": str(uncertainty_seal.expanduser().resolve()),
            "evaluation_spec": str(evaluation_spec.expanduser().resolve()),
            "uncertainty_evaluation_spec": str(
                uncertainty_evaluation_spec.expanduser().resolve()
            ),
            "deployment_spec": str(deployment_spec.expanduser().resolve()),
            "release_readiness_spec": str(
                release_readiness_spec.expanduser().resolve()
            ),
            "canonical_target": str(target_output.expanduser().resolve()),
            "canonical_target_receipt": str(target_receipt.expanduser().resolve()),
            "evaluation_lock": str(evaluation_lock.expanduser().resolve()),
            "joined_output": str(joined_output.expanduser().resolve()),
            "release_receipt": str(release_receipt.expanduser().resolve()),
            "fixed_i3_runtime_seal": str(fixed_i3_runtime_seal.expanduser().resolve()),
            "fixed_runtime_completion": str(fixed_runtime_completion.expanduser().resolve()),
            "postlock_runtime_guard": str(postlock_runtime_guard.expanduser().resolve()),
            "radar_mask_runtime_guard": str(radar_mask_runtime_guard.expanduser().resolve()),
        },
        "effective_sources": _source_bindings(),
    }
    document["content_sha256"] = canonical_json_sha256(document)
    return document


def _target_boundary_paths(document: Mapping[str, Any]) -> list[Path]:
    locations = document.get("locations")
    if not isinstance(locations, Mapping):
        raise PretargetReleaseLockError("release lock locations are absent")
    return [
        Path(str(locations[name])).expanduser().resolve()
        for name in (
            "canonical_target",
            "canonical_target_receipt",
            "evaluation_lock",
            "joined_output",
            "release_receipt",
        )
    ]


def reverify_runtime_inputs_from_lock(document: Mapping[str, Any]) -> dict[str, Any]:
    """Rehash the RF/SVD runtime closure and both completion guards on demand."""

    locations = document.get("locations")
    boundaries = document.get("boundaries")
    if not isinstance(locations, Mapping) or not isinstance(boundaries, Mapping):
        raise PretargetReleaseLockError("release lock runtime locations/boundaries are absent")
    primary = boundaries.get("primary_predictions")
    expected = boundaries.get("runtime_payload_closure")
    if not isinstance(primary, Mapping) or not isinstance(expected, Mapping):
        raise PretargetReleaseLockError("release lock runtime boundary is absent")
    observed = _runtime_guard_summary(
        Path(str(locations["fixed_i3_runtime_seal"])),
        Path(str(locations["fixed_runtime_completion"])),
        Path(str(locations["postlock_runtime_guard"])),
        Path(str(locations["radar_mask_runtime_guard"])),
        primary,
        boundaries["radar_masks"],
    )
    if observed != expected:
        raise PretargetReleaseLockError(
            "runtime payload closure differs from the pretarget release lock"
        )
    return observed


def validate_release_lock(
    path: Path, *, require_target_absence: bool = True
) -> dict[str, Any]:
    """Rehash every bound graph and return the exact immutable lock."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PretargetReleaseLockError(f"pretarget release lock is absent: {resolved}")
    if stat.S_IMODE(resolved.stat().st_mode) != 0o444:
        raise PretargetReleaseLockError("pretarget release lock mode must be exactly 0444")
    observed = _read_json(resolved, "pretarget release lock")
    content_hash = str(observed.get("content_sha256", ""))
    payload = dict(observed)
    payload.pop("content_sha256", None)
    if len(content_hash) != 64 or canonical_json_sha256(payload) != content_hash:
        raise PretargetReleaseLockError("pretarget release lock content hash mismatch")
    if (
        observed.get("schema_version") != SCHEMA_VERSION
        or observed.get("classification") != "locked_hcs_pretarget_release_lock"
        or observed.get("status") != "all_target_free_boundaries_complete"
        or observed.get("canonical_target_build_authorized") is not True
        or observed.get("target_or_label_artifact_opened") is not False
        or observed.get("commercial_claim_authorized") is not False
    ):
        raise PretargetReleaseLockError("pretarget release lock invariants are invalid")
    if require_target_absence:
        _assert_absent(_target_boundary_paths(observed))
    locations = observed["locations"]
    expected = _release_document(
        primary_root=Path(locations["primary_root"]),
        mask_root=Path(locations["mask_root"]),
        uncertainty_seal=Path(locations["uncertainty_seal"]),
        evaluation_spec=Path(locations["evaluation_spec"]),
        uncertainty_evaluation_spec=Path(locations["uncertainty_evaluation_spec"]),
        deployment_spec=Path(locations["deployment_spec"]),
        release_readiness_spec=Path(locations["release_readiness_spec"]),
        target_output=Path(locations["canonical_target"]),
        target_receipt=Path(locations["canonical_target_receipt"]),
        evaluation_lock=Path(locations["evaluation_lock"]),
        joined_output=Path(locations["joined_output"]),
        release_receipt=Path(locations["release_receipt"]),
        fixed_i3_runtime_seal=Path(locations["fixed_i3_runtime_seal"]),
        fixed_runtime_completion=Path(locations["fixed_runtime_completion"]),
        postlock_runtime_guard=Path(locations["postlock_runtime_guard"]),
        radar_mask_runtime_guard=Path(locations["radar_mask_runtime_guard"]),
        enforce_target_free=require_target_absence,
    )
    if observed != expected:
        raise PretargetReleaseLockError(
            "pretarget release lock differs from the live revalidated target-free graph"
        )
    return observed


def create_release_lock(
    *,
    primary_root: Path = DEFAULT_PRIMARY_ROOT,
    mask_root: Path = DEFAULT_MASK_ROOT,
    uncertainty_seal: Path = DEFAULT_UNCERTAINTY_SEAL,
    evaluation_spec: Path = DEFAULT_EVALUATION_SPEC,
    uncertainty_evaluation_spec: Path = DEFAULT_UNCERTAINTY_EVALUATION_SPEC,
    deployment_spec: Path = DEFAULT_DEPLOYMENT_SPEC,
    release_readiness_spec: Path = DEFAULT_RELEASE_READINESS_SPEC,
    output: Path = DEFAULT_RELEASE_LOCK,
    target_output: Path = DEFAULT_TARGET,
    target_receipt: Path = DEFAULT_TARGET_RECEIPT,
    evaluation_lock: Path = DEFAULT_EVALUATION_LOCK,
    joined_output: Path = DEFAULT_JOINED,
    release_receipt: Path = DEFAULT_RELEASE_RECEIPT,
    fixed_i3_runtime_seal: Path = DEFAULT_FIXED_I3_RUNTIME_SEAL,
    fixed_runtime_completion: Path = DEFAULT_FIXED_RUNTIME_COMPLETION,
    postlock_runtime_guard: Path = DEFAULT_POSTLOCK_RUNTIME_GUARD,
    radar_mask_runtime_guard: Path = DEFAULT_RADAR_MASK_RUNTIME_GUARD,
) -> dict[str, Any]:
    """Create or safely revalidate the one immutable pretarget release lock."""

    destination = output.expanduser().resolve()
    if destination.exists():
        return validate_release_lock(destination, require_target_absence=True)
    forbidden = [
        target_output,
        target_receipt,
        evaluation_lock,
        joined_output,
        release_receipt,
    ]
    # This check intentionally precedes all deep validators.  No target-bearing
    # file is opened by this creator under any circumstance.
    _assert_absent(forbidden)
    document = _release_document(
        primary_root=primary_root,
        mask_root=mask_root,
        uncertainty_seal=uncertainty_seal,
        evaluation_spec=evaluation_spec,
        uncertainty_evaluation_spec=uncertainty_evaluation_spec,
        deployment_spec=deployment_spec,
        release_readiness_spec=release_readiness_spec,
        target_output=target_output,
        target_receipt=target_receipt,
        evaluation_lock=evaluation_lock,
        joined_output=joined_output,
        release_receipt=release_receipt,
        fixed_i3_runtime_seal=fixed_i3_runtime_seal,
        fixed_runtime_completion=fixed_runtime_completion,
        postlock_runtime_guard=postlock_runtime_guard,
        radar_mask_runtime_guard=radar_mask_runtime_guard,
    )
    # Close the validation/publication race as far as a filesystem boundary can:
    # target publication by another process makes this operation fail closed.
    _assert_absent(forbidden)
    _atomic_immutable_json(destination, document)
    published = validate_release_lock(destination, require_target_absence=True)
    if published != document:
        raise PretargetReleaseLockError("published pretarget release lock changed")
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--uncertainty-seal", type=Path, default=DEFAULT_UNCERTAINTY_SEAL)
    parser.add_argument("--evaluation-spec", type=Path, default=DEFAULT_EVALUATION_SPEC)
    parser.add_argument(
        "--uncertainty-evaluation-spec",
        type=Path,
        default=DEFAULT_UNCERTAINTY_EVALUATION_SPEC,
    )
    parser.add_argument("--deployment-spec", type=Path, default=DEFAULT_DEPLOYMENT_SPEC)
    parser.add_argument(
        "--release-readiness-spec",
        type=Path,
        default=DEFAULT_RELEASE_READINESS_SPEC,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RELEASE_LOCK)
    parser.add_argument("--target-output", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--target-receipt", type=Path, default=DEFAULT_TARGET_RECEIPT)
    parser.add_argument("--evaluation-lock", type=Path, default=DEFAULT_EVALUATION_LOCK)
    parser.add_argument("--joined-output", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--release-receipt", type=Path, default=DEFAULT_RELEASE_RECEIPT)
    parser.add_argument(
        "--fixed-i3-runtime-seal", type=Path, default=DEFAULT_FIXED_I3_RUNTIME_SEAL
    )
    parser.add_argument(
        "--fixed-runtime-completion", type=Path, default=DEFAULT_FIXED_RUNTIME_COMPLETION
    )
    parser.add_argument(
        "--postlock-runtime-guard", type=Path, default=DEFAULT_POSTLOCK_RUNTIME_GUARD
    )
    parser.add_argument(
        "--radar-mask-runtime-guard",
        type=Path,
        default=DEFAULT_RADAR_MASK_RUNTIME_GUARD,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = create_release_lock(
            primary_root=args.primary_root,
            mask_root=args.mask_root,
            uncertainty_seal=args.uncertainty_seal,
            evaluation_spec=args.evaluation_spec,
            uncertainty_evaluation_spec=args.uncertainty_evaluation_spec,
            deployment_spec=args.deployment_spec,
            release_readiness_spec=args.release_readiness_spec,
            output=args.output,
            target_output=args.target_output,
            target_receipt=args.target_receipt,
            evaluation_lock=args.evaluation_lock,
            joined_output=args.joined_output,
            release_receipt=args.release_receipt,
            fixed_i3_runtime_seal=args.fixed_i3_runtime_seal,
            fixed_runtime_completion=args.fixed_runtime_completion,
            postlock_runtime_guard=args.postlock_runtime_guard,
            radar_mask_runtime_guard=args.radar_mask_runtime_guard,
        )
    except (PretargetReleaseLockError, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
