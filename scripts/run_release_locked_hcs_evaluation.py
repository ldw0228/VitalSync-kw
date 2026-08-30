#!/usr/bin/env python3
"""Execute the single release-authorized HCS target evaluation DAG.

This supervisor deliberately has no production path or evaluator-argument
switches.  Every target-bearing path and every child argv is fixed by the
pretarget release lock and the release-readiness specification.  It first
revalidates those target-free authorities and the target-builder wrapper
receipt, and only then live-hashes the canonical target chain.

The four create-once producers are also their own crash journal.  A resume may
skip a step only when all of that step's immutable outputs validate against the
exact frozen command.  Partial, out-of-order, or differently-provenanced output
is quarantined rather than executed over.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).absolute().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_locked_hcs_targets as TARGETS  # noqa: E402
import build_locked_hcs_targets_after_release_lock as RELEASED_TARGETS  # noqa: E402
import create_locked_hcs_pretarget_release_lock as RELEASE  # noqa: E402
import evaluate_locked_hcs_oof as PRIMARY  # noqa: E402
import evaluate_locked_hcs_release_readiness as READINESS  # noqa: E402
import evaluate_locked_hcs_uncertainty as UNCERTAINTY  # noqa: E402


SCHEMA_VERSION = 1
CLASSIFICATION = "locked_hcs_release_evaluation_execution_attestation"
PROJECT_ROOT = SCRIPT_DIR.parent


class ReleaseEvaluationError(RuntimeError):
    """The release authority, fixed execution DAG, or immutable output drifted."""


def _absolute(path: os.PathLike[str] | str) -> Path:
    """Return an absolute path without dereferencing a venv/symlink launcher."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


@dataclass(frozen=True)
class ReleasePaths:
    project_root: Path
    primary_root: Path
    mask_root: Path
    campaign_root: Path
    release_readiness_spec: Path
    pretarget_release_lock: Path
    target_release_receipt: Path
    canonical_target: Path
    canonical_target_receipt: Path
    predictions_seal: Path
    evaluation_lock: Path
    joined_output: Path
    joined_metrics: Path
    primary_evaluation_spec: Path
    uncertainty_evaluation_spec: Path
    radar_mask_complete_seal: Path
    uncertainty_inputs_seal: Path
    uncertainty_calibration: Path
    primary_output_dir: Path
    primary_report: Path
    primary_csv: Path
    primary_receipt: Path
    radar_output_dir: Path
    radar_report: Path
    radar_csv: Path
    radar_receipt: Path
    uncertainty_output_dir: Path
    uncertainty_report: Path
    uncertainty_csv: Path
    uncertainty_receipt: Path
    attestation: Path
    raw_join_source: Path
    primary_source: Path
    radar_source: Path
    uncertainty_source: Path
    execution_source: Path


def release_paths(project_root: Path = PROJECT_ROOT) -> ReleasePaths:
    root = _absolute(project_root)
    primary = root / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_oof"
    masks = root / "artifacts/runs/harmonic_candidate_set_snn_v2/hcs_locked_radar_masks"
    campaign = root / "artifacts/campaigns/harmonic_candidate_set_snn_v2"
    primary_output = primary / "primary_evaluation"
    radar_output = masks / "evaluation"
    uncertainty_output = primary / "uncertainty_evaluation"
    return ReleasePaths(
        project_root=root,
        primary_root=primary,
        mask_root=masks,
        campaign_root=campaign,
        release_readiness_spec=campaign / "locked_hcs_release_readiness_spec.json",
        pretarget_release_lock=primary / "pretarget_release_lock.json",
        target_release_receipt=primary / "canonical_locked_hcs_targets_release_receipt.json",
        canonical_target=primary / "canonical_locked_hcs_targets.npz",
        canonical_target_receipt=primary / "canonical_locked_hcs_targets_receipt.json",
        predictions_seal=primary / "predictions_seal.json",
        evaluation_lock=primary / "evaluation_lock.json",
        joined_output=primary / "locked_hcs_oof_joined.npz",
        joined_metrics=primary / "locked_hcs_oof_metrics.json",
        primary_evaluation_spec=campaign / "locked_primary_evaluation_spec.json",
        uncertainty_evaluation_spec=campaign / "locked_uncertainty_evaluation_spec.json",
        radar_mask_complete_seal=masks / "complete_seal.json",
        uncertainty_inputs_seal=primary / "uncertainty_inputs_seal.json",
        uncertainty_calibration=(
            campaign
            / "nested_proposer/current_source_merged/uncertainty_calibration.json"
        ),
        primary_output_dir=primary_output,
        primary_report=primary_output / "locked_hcs_primary_evaluation.json",
        primary_csv=primary_output / "locked_hcs_primary_metrics.csv",
        primary_receipt=primary_output / "locked_hcs_primary_evaluation_receipt.json",
        radar_output_dir=radar_output,
        radar_report=radar_output / "locked_hcs_radar_masks_evaluation.json",
        radar_csv=radar_output / "locked_hcs_radar_masks_metrics.csv",
        radar_receipt=radar_output / "locked_hcs_radar_masks_evaluation_receipt.json",
        uncertainty_output_dir=uncertainty_output,
        uncertainty_report=uncertainty_output / "uncertainty_report.json",
        uncertainty_csv=uncertainty_output / "uncertainty_metrics.csv",
        uncertainty_receipt=uncertainty_output / "uncertainty_receipt.json",
        attestation=primary / "release_evaluation_execution_attestation.json",
        raw_join_source=root / "scripts/run_locked_hcs_oof.py",
        primary_source=root / "scripts/evaluate_locked_hcs_oof.py",
        radar_source=root / "scripts/evaluate_locked_hcs_radar_masks.py",
        uncertainty_source=root / "scripts/evaluate_locked_hcs_uncertainty.py",
        execution_source=root / "scripts/run_release_locked_hcs_evaluation.py",
    )


DEFAULT_ATTESTATION = release_paths().attestation


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = _absolute(path)
    if not resolved.is_file() or resolved.is_symlink():
        raise ReleaseEvaluationError(f"bound artifact is absent/not regular: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseEvaluationError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    resolved = _absolute(path)
    if not resolved.is_file() or resolved.is_symlink():
        raise ReleaseEvaluationError(f"{label} is absent/not regular: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_pairs_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseEvaluationError(f"invalid {label}: {resolved} ({exc})") from exc
    if not isinstance(value, dict):
        raise ReleaseEvaluationError(f"{label} must be a JSON object")
    return value


def _require_0444(path: Path, label: str) -> None:
    resolved = _absolute(path)
    if not resolved.is_file() or resolved.is_symlink():
        raise ReleaseEvaluationError(f"{label} is absent/not regular: {resolved}")
    if stat.S_IMODE(resolved.stat().st_mode) != 0o444:
        raise ReleaseEvaluationError(f"{label} mode must be exactly 0444")


def _verify_content(document: Mapping[str, Any], label: str) -> str:
    expected = str(document.get("content_sha256", ""))
    payload = dict(document)
    payload.pop("content_sha256", None)
    if len(expected) != 64 or canonical_sha256(payload) != expected:
        raise ReleaseEvaluationError(f"{label} canonical content hash mismatch")
    return expected


def _binding_shape(
    raw: Any, *, relative_to: Path, label: str
) -> dict[str, Any]:
    """Validate binding syntax only; intentionally do not stat/open its path."""

    if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "bytes"}:
        raise ReleaseEvaluationError(f"{label} binding shape is invalid")
    path = Path(str(raw["path"])).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = _absolute(path)
    digest = str(raw["sha256"]).lower()
    size = raw["bytes"]
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise ReleaseEvaluationError(f"{label} binding fields are invalid")
    return {"path": str(path), "sha256": digest, "bytes": size}


def _same_binding(left: Any, right: Any) -> bool:
    try:
        return _binding_shape(
            left, relative_to=PROJECT_ROOT, label="left"
        ) == _binding_shape(right, relative_to=PROJECT_ROOT, label="right")
    except ReleaseEvaluationError:
        return False


def _assert_binding(left: Any, right: Any, label: str) -> None:
    if not _same_binding(left, right):
        raise ReleaseEvaluationError(f"binding differs: {label}")


def _expected_lock_locations(paths: ReleasePaths) -> dict[str, Path]:
    return {
        "primary_root": paths.primary_root,
        "mask_root": paths.mask_root,
        "uncertainty_seal": paths.uncertainty_inputs_seal,
        "evaluation_spec": paths.primary_evaluation_spec,
        "uncertainty_evaluation_spec": paths.uncertainty_evaluation_spec,
        "release_readiness_spec": paths.release_readiness_spec,
        "canonical_target": paths.canonical_target,
        "canonical_target_receipt": paths.canonical_target_receipt,
        "evaluation_lock": paths.evaluation_lock,
        "joined_output": paths.joined_output,
        "release_receipt": paths.target_release_receipt,
    }


def _verify_readiness_roles(spec: Mapping[str, Any], paths: ReleasePaths) -> None:
    roles = spec.get("input_roles")
    if not isinstance(roles, Mapping):
        raise ReleaseEvaluationError("release-readiness input roles are absent")
    expected = {
        "target_release_receipt": paths.target_release_receipt,
        "pretarget_release_lock": paths.pretarget_release_lock,
        "primary_evaluation_lock": paths.evaluation_lock,
        "canonical_target": paths.canonical_target,
        "canonical_target_receipt": paths.canonical_target_receipt,
        "joined_output": paths.joined_output,
        "predictions_seal": paths.predictions_seal,
        "release_evaluation_execution_attestation": paths.attestation,
        "primary_report": paths.primary_report,
        "primary_receipt": paths.primary_receipt,
        "radar_report": paths.radar_report,
        "radar_receipt": paths.radar_receipt,
        "uncertainty_spec": paths.uncertainty_evaluation_spec,
        "primary_evaluation_spec": paths.primary_evaluation_spec,
        "uncertainty_report": paths.uncertainty_report,
        "uncertainty_receipt": paths.uncertainty_receipt,
        "radar_mask_complete_seal": paths.radar_mask_complete_seal,
        "uncertainty_inputs_seal": paths.uncertainty_inputs_seal,
    }
    for role, expected_path in expected.items():
        value = roles.get(role)
        if value is None or _absolute(str(value)) != expected_path:
            raise ReleaseEvaluationError(f"release-readiness role differs: {role}")


def verify_release_authorization(paths: ReleasePaths | None = None) -> dict[str, Any]:
    """Revalidate target-free authority first, then live-open the target chain.

    The ordering in this function is an ABI.  Do not resolve/stat/hash the
    canonical target, target receipt, or prediction-derived evaluation files
    above the explicit ``target access is now authorized`` boundary.
    """

    frozen = paths or release_paths()

    # 1. Public target-free specification and release-lock verifiers.
    try:
        readiness_spec, readiness_binding = READINESS.load_release_readiness_spec(
            frozen.release_readiness_spec
        )
        release_lock = RELEASE.validate_release_lock(
            frozen.pretarget_release_lock, require_target_absence=False
        )
    except Exception as exc:
        raise ReleaseEvaluationError(f"target-free release authority is invalid: {exc}") from exc
    _verify_readiness_roles(readiness_spec, frozen)
    locations = release_lock.get("locations")
    if not isinstance(locations, Mapping):
        raise ReleaseEvaluationError("pretarget release locations are absent")
    for name, expected in _expected_lock_locations(frozen).items():
        if name not in locations or _absolute(str(locations[name])) != expected:
            raise ReleaseEvaluationError(f"pretarget release location differs: {name}")
    specs = release_lock.get("frozen_specs")
    if not isinstance(specs, Mapping):
        raise ReleaseEvaluationError("pretarget frozen specifications are absent")
    readiness_summary = specs.get("release_readiness")
    if (
        not isinstance(readiness_summary, Mapping)
        or readiness_summary.get("target_or_target_bearing_artifact_opened") is not False
        or readiness_summary.get("commercial_release_ready_must_equal") is not False
        or readiness_summary.get("prospective_confirmation_required") is not True
        or readiness_summary.get("content_sha256") != readiness_spec.get("content_sha256")
    ):
        raise ReleaseEvaluationError("pretarget readiness policy binding differs")
    _assert_binding(
        readiness_summary.get("binding"), readiness_binding, "release-readiness specification"
    )

    # 2. Read only the wrapper receipt and validate binding *shapes*.  Its
    # target paths are not followed until its lock authorization is proven.
    _require_0444(frozen.pretarget_release_lock, "pretarget release lock")
    _require_0444(frozen.target_release_receipt, "target release receipt")
    release_receipt = _read_json(frozen.target_release_receipt, "target release receipt")
    _verify_content(release_receipt, "target release receipt")
    if (
        release_receipt.get("schema_version") != SCHEMA_VERSION
        or release_receipt.get("classification")
        != "locked_hcs_canonical_targets_built_after_pretarget_release"
        or release_receipt.get("commercial_claim_authorized") is not False
        or release_receipt.get("prospective_confirmation_required") is not True
        or release_receipt.get("release_lock_revalidated_before_target_builder_call") is not True
        or release_receipt.get("all_release_bound_artifacts_rehashed_before_target_builder_call")
        is not True
        or release_receipt.get("target_metadata_access_before_release_validation") is not False
        or release_receipt.get("canonical_target_builder_called_only_after_release_authorization")
        is not True
    ):
        raise ReleaseEvaluationError("target release receipt invariants are invalid")
    release_lock_binding = bind_file(frozen.pretarget_release_lock)
    _assert_binding(
        release_receipt.get("pretarget_release_lock"),
        release_lock_binding,
        "release receipt -> pretarget release lock",
    )
    if release_receipt.get("pretarget_release_content_sha256") != release_lock.get(
        "content_sha256"
    ):
        raise ReleaseEvaluationError("release receipt/pretarget content hash differs")
    target_shape = _binding_shape(
        release_receipt.get("canonical_target"),
        relative_to=frozen.target_release_receipt.parent,
        label="release receipt canonical target",
    )
    target_receipt_shape = _binding_shape(
        release_receipt.get("canonical_target_receipt"),
        relative_to=frozen.target_release_receipt.parent,
        label="release receipt canonical target receipt",
    )
    if _absolute(target_shape["path"]) != frozen.canonical_target:
        raise ReleaseEvaluationError("release receipt authorizes another canonical target")
    if _absolute(target_receipt_shape["path"]) != frozen.canonical_target_receipt:
        raise ReleaseEvaluationError("release receipt authorizes another target receipt")

    # ---- Target access is now authorized.  Live-open/hash the exact chain. ----
    _require_0444(frozen.canonical_target, "canonical target")
    _require_0444(frozen.canonical_target_receipt, "canonical target receipt")
    target_binding = bind_file(frozen.canonical_target)
    target_receipt_binding = bind_file(frozen.canonical_target_receipt)
    _assert_binding(target_shape, target_binding, "release receipt -> canonical target")
    _assert_binding(
        target_receipt_shape,
        target_receipt_binding,
        "release receipt -> canonical target receipt",
    )
    target_receipt = _read_json(frozen.canonical_target_receipt, "canonical target receipt")
    target_receipt_content = _verify_content(target_receipt, "canonical target receipt")
    if (
        target_receipt.get("schema_version") != SCHEMA_VERSION
        or target_receipt.get("classification")
        != "retrospective_locked_hcs_canonical_target_artifact_receipt"
        or target_receipt.get("target_artifact_created_once") is not True
        or target_receipt.get("target_artifact_overwrite_allowed") is not False
        or target_receipt.get("pretarget_release_capability_verified") is not True
        or target_receipt.get("commercial_claim_authorized") is not False
        or target_receipt.get("prospective_confirmation_required") is not True
    ):
        raise ReleaseEvaluationError("canonical target receipt invariants are invalid")
    target_builder_command = target_receipt.get("orchestrator_command")
    expected_target_wrapper = (
        frozen.project_root / "scripts/build_locked_hcs_targets_after_release_lock.py"
    )
    if (
        not isinstance(target_builder_command, list)
        or len(target_builder_command) < 2
        or _absolute(str(target_builder_command[1])) != expected_target_wrapper
    ):
        raise ReleaseEvaluationError(
            "canonical target receipt was not produced through the release-lock wrapper"
        )
    _assert_binding(
        target_receipt.get("target_artifact"), target_binding, "target receipt -> target"
    )
    if release_receipt.get("canonical_target_receipt_content_sha256") != target_receipt_content:
        raise ReleaseEvaluationError("wrapper/canonical receipt content hash differs")

    # Recompute the wrapper document now that target access is authorized.  It
    # additionally catches drift in all three target-builder implementation
    # sources without trusting nested booleans.
    try:
        expected_release_receipt = RELEASED_TARGETS._release_receipt_document(  # noqa: SLF001
            release_lock_path=frozen.pretarget_release_lock,
            release_lock=release_lock,
            target_document=target_receipt,
            target_binding=target_binding,
            target_receipt_binding=target_receipt_binding,
        )
    except Exception as exc:
        raise ReleaseEvaluationError(f"cannot reconstruct target release receipt: {exc}") from exc
    if release_receipt != expected_release_receipt:
        raise ReleaseEvaluationError("target release receipt differs from live source/binding graph")

    # Public producer verifier: traverse every one of the 18 sealed prediction
    # units and return the canonical seal binding.  This is deliberately after
    # wrapper authorization even though the prediction graph is target-free.
    try:
        verified_predictions = TARGETS.verify_prediction_seal(frozen.primary_root)
    except Exception as exc:
        raise ReleaseEvaluationError(f"prediction seal revalidation failed: {exc}") from exc
    predictions_binding = bind_file(frozen.predictions_seal)
    source_bindings = target_receipt.get("source_bindings")
    if not isinstance(source_bindings, Mapping):
        raise ReleaseEvaluationError("canonical target source bindings are absent")
    _assert_binding(
        source_bindings.get("predictions_seal"),
        predictions_binding,
        "target receipt -> predictions seal",
    )
    if not isinstance(verified_predictions, Mapping):
        raise ReleaseEvaluationError("prediction verifier returned an invalid result")
    _assert_binding(
        verified_predictions.get("predictions_seal"),
        predictions_binding,
        "prediction verifier -> predictions seal",
    )
    primary_boundary = release_lock.get("boundaries", {}).get("primary_predictions", {})
    _assert_binding(
        primary_boundary.get("predictions_seal"),
        predictions_binding,
        "pretarget release lock -> predictions seal",
    )

    primary_spec_binding = bind_file(frozen.primary_evaluation_spec)
    uncertainty_spec_binding = bind_file(frozen.uncertainty_evaluation_spec)
    radar_seal_binding = bind_file(frozen.radar_mask_complete_seal)
    uncertainty_seal_binding = bind_file(frozen.uncertainty_inputs_seal)
    uncertainty_calibration_binding = bind_file(frozen.uncertainty_calibration)
    _assert_binding(
        specs.get("primary_evaluation", {}).get("binding"),
        primary_spec_binding,
        "pretarget lock -> primary evaluation spec",
    )
    _assert_binding(
        specs.get("secondary_uncertainty_evaluation", {}).get("binding"),
        uncertainty_spec_binding,
        "pretarget lock -> uncertainty evaluation spec",
    )
    _assert_binding(
        specs.get("secondary_uncertainty_evaluation", {}).get("calibration"),
        uncertainty_calibration_binding,
        "pretarget lock -> uncertainty calibration",
    )
    _assert_binding(
        release_lock.get("boundaries", {}).get("radar_masks", {}).get("complete_seal"),
        radar_seal_binding,
        "pretarget lock -> radar-mask complete seal",
    )
    _assert_binding(
        release_lock.get("boundaries", {}).get("uncertainty", {}).get(
            "uncertainty_inputs_seal"
        ),
        uncertainty_seal_binding,
        "pretarget lock -> uncertainty inputs seal",
    )
    _assert_binding(
        release_lock.get("boundaries", {}).get("uncertainty", {}).get(
            "pretest_calibration"
        ),
        uncertainty_calibration_binding,
        "uncertainty boundary -> pretest calibration",
    )

    radar_boundary = release_lock.get("boundaries", {}).get("radar_masks", {})
    uncertainty_boundary = release_lock.get("boundaries", {}).get("uncertainty", {})
    if not isinstance(radar_boundary, Mapping) or not isinstance(
        uncertainty_boundary, Mapping
    ):
        raise ReleaseEvaluationError("pretarget radar/uncertainty boundary is absent")
    radar_plan_binding = _binding_shape(
        radar_boundary.get("plan"),
        relative_to=frozen.pretarget_release_lock.parent,
        label="pretarget radar-mask plan",
    )
    radar_preexecution_binding = _binding_shape(
        radar_boundary.get("preexecution_lock"),
        relative_to=frozen.pretarget_release_lock.parent,
        label="pretarget radar-mask preexecution lock",
    )
    uncertainty_archive_binding = _binding_shape(
        uncertainty_boundary.get("uncertainty_archive"),
        relative_to=frozen.pretarget_release_lock.parent,
        label="pretarget uncertainty archive",
    )
    for label, binding in (
        ("radar-mask plan", radar_plan_binding),
        ("radar-mask preexecution lock", radar_preexecution_binding),
        ("uncertainty archive", uncertainty_archive_binding),
    ):
        _assert_binding(binding, bind_file(Path(binding["path"])), label)

    # Recompute the producer's complete target-free uncertainty audit instead
    # of trusting the receipt's dynamic rehash counts or scalar assertions.
    try:
        *_, uncertainty_pretarget_audit = UNCERTAINTY._validate_pre_target_inputs(  # noqa: SLF001
            uncertainty_evaluation_spec=frozen.uncertainty_evaluation_spec,
            evaluation_spec=frozen.primary_evaluation_spec,
            calibration_path=frozen.uncertainty_calibration,
            predictions_seal_path=frozen.predictions_seal,
            uncertainty_seal_path=frozen.uncertainty_inputs_seal,
            locked_oof_root=frozen.primary_root,
        )
    except Exception as exc:
        raise ReleaseEvaluationError(
            f"uncertainty target-free input revalidation failed: {exc}"
        ) from exc
    if not isinstance(uncertainty_pretarget_audit, Mapping):
        raise ReleaseEvaluationError("uncertainty target-free audit is invalid")
    for role, expected in (
        ("secondary_uncertainty_evaluation_spec", uncertainty_spec_binding),
        ("evaluation_spec", primary_spec_binding),
        ("calibration", uncertainty_calibration_binding),
        ("predictions_seal", predictions_binding),
        ("uncertainty_inputs_seal", uncertainty_seal_binding),
        ("uncertainty_archive", uncertainty_archive_binding),
    ):
        _assert_binding(
            uncertainty_pretarget_audit.get(role), expected,
            f"uncertainty target-free audit {role}",
        )

    return {
        "readiness_spec": readiness_spec,
        "release_lock": release_lock,
        "release_receipt": release_receipt,
        "target_receipt": target_receipt,
        "release_readiness_spec": readiness_binding,
        "pretarget_release_lock": release_lock_binding,
        "target_release_receipt": bind_file(frozen.target_release_receipt),
        "canonical_target": target_binding,
        "canonical_target_receipt": target_receipt_binding,
        "predictions_seal": predictions_binding,
        "primary_evaluation_spec": primary_spec_binding,
        "uncertainty_evaluation_spec": uncertainty_spec_binding,
        "radar_mask_complete_seal": radar_seal_binding,
        "radar_mask_plan": radar_plan_binding,
        "radar_mask_preexecution_lock": radar_preexecution_binding,
        "uncertainty_inputs_seal": uncertainty_seal_binding,
        "uncertainty_archive": uncertainty_archive_binding,
        "uncertainty_calibration": uncertainty_calibration_binding,
        "uncertainty_pretarget_audit": dict(uncertainty_pretarget_audit),
    }


def build_frozen_argv_from_evidence(
    evidence: Mapping[str, os.PathLike[str] | str],
    *,
    python_executable: os.PathLike[str] | str,
) -> list[list[str]]:
    """Build the fixed four-command DAG from exact, authorized evidence paths."""

    required = {
        "raw_join_source", "primary_source", "radar_source", "uncertainty_source",
        "primary_root", "mask_root", "canonical_target", "evaluation_lock",
        "canonical_target_receipt", "primary_evaluation_spec",
        "uncertainty_evaluation_spec", "uncertainty_calibration", "predictions_seal",
        "uncertainty_inputs_seal", "primary_output_dir", "primary_report",
        "primary_csv", "primary_receipt", "radar_output_dir", "radar_report",
        "radar_csv", "radar_receipt", "uncertainty_output_dir",
        "uncertainty_report", "uncertainty_csv", "uncertainty_receipt",
    }
    if set(evidence) != required:
        missing = sorted(required - set(evidence))
        extra = sorted(set(evidence) - required)
        raise ReleaseEvaluationError(
            f"fixed evaluation evidence topology differs (missing={missing}, extra={extra})"
        )
    values = {name: _absolute(value) for name, value in evidence.items()}
    # Do not use Path.resolve() here.  A uv/venv launcher is commonly a symlink
    # whose dereferenced base interpreter does not have the project packages.
    interpreter = os.path.abspath(os.fspath(python_executable))
    return [
        [
            interpreter,
            str(values["raw_join_source"]),
            "join",
            "--output-root",
            str(values["primary_root"]),
            "--targets",
            str(values["canonical_target"]),
        ],
        [
            interpreter,
            str(values["primary_source"]),
            "evaluate",
            "--locked-oof-root",
            str(values["primary_root"]),
            "--evaluation-lock",
            str(values["evaluation_lock"]),
            "--target-receipt",
            str(values["canonical_target_receipt"]),
            "--evaluation-spec",
            str(values["primary_evaluation_spec"]),
            "--output-dir",
            str(values["primary_output_dir"]),
            "--report-output",
            str(values["primary_report"]),
            "--csv-output",
            str(values["primary_csv"]),
            "--receipt-output",
            str(values["primary_receipt"]),
            "--expected-rows",
            str(PRIMARY.EXPECTED_VALID_REFERENCE_ROWS),
            "--expected-identities",
            str(PRIMARY.EXPECTED_IDENTITIES),
            "--bootstrap-samples",
            str(PRIMARY.BOOTSTRAP_SAMPLES),
            "--bootstrap-seed",
            str(PRIMARY.BOOTSTRAP_SEED),
            "--bootstrap-confidence",
            str(PRIMARY.BOOTSTRAP_CONFIDENCE),
        ],
        [
            interpreter,
            str(values["radar_source"]),
            "--radar-mask-root",
            str(values["mask_root"]),
            "--primary-root",
            str(values["primary_root"]),
            "--primary-evaluation-lock",
            str(values["evaluation_lock"]),
            "--target-receipt",
            str(values["canonical_target_receipt"]),
            "--evaluation-spec",
            str(values["primary_evaluation_spec"]),
            "--output-dir",
            str(values["radar_output_dir"]),
            "--report-output",
            str(values["radar_report"]),
            "--csv-output",
            str(values["radar_csv"]),
            "--receipt-output",
            str(values["radar_receipt"]),
        ],
        [
            interpreter,
            str(values["uncertainty_source"]),
            "--locked-oof-root",
            str(values["primary_root"]),
            "--uncertainty-evaluation-spec",
            str(values["uncertainty_evaluation_spec"]),
            "--evaluation-spec",
            str(values["primary_evaluation_spec"]),
            "--calibration",
            str(values["uncertainty_calibration"]),
            "--predictions-seal",
            str(values["predictions_seal"]),
            "--uncertainty-seal",
            str(values["uncertainty_inputs_seal"]),
            "--evaluation-lock",
            str(values["evaluation_lock"]),
            "--target-receipt",
            str(values["canonical_target_receipt"]),
            "--output-dir",
            str(values["uncertainty_output_dir"]),
            "--report-output",
            str(values["uncertainty_report"]),
            "--csv-output",
            str(values["uncertainty_csv"]),
            "--receipt-output",
            str(values["uncertainty_receipt"]),
        ],
    ]


def validate_frozen_argv_from_evidence(
    raw: Any,
    *,
    evidence: Mapping[str, os.PathLike[str] | str],
) -> list[list[str]]:
    """Require the attested argv to equal the shared fixed-DAG construction."""

    if (
        not isinstance(raw, list)
        or len(raw) != 4
        or not all(
            isinstance(command, list)
            and len(command) >= 2
            and all(isinstance(token, str) and token for token in command)
            for command in raw
        )
    ):
        raise ReleaseEvaluationError("execution attestation argv topology differs")
    interpreter = raw[0][0]
    if any(command[0] != interpreter for command in raw):
        raise ReleaseEvaluationError("execution attestation uses multiple Python launchers")
    try:
        same_runtime = os.path.samefile(interpreter, sys.executable)
    except OSError:
        same_runtime = os.path.abspath(interpreter) == os.path.abspath(sys.executable)
    if not same_runtime:
        raise ReleaseEvaluationError("execution attestation Python launcher identity differs")
    expected = build_frozen_argv_from_evidence(
        evidence,
        python_executable=interpreter,
    )
    if raw != expected:
        raise ReleaseEvaluationError("execution attestation argv differs from the fixed DAG")
    return expected


def _frozen_argv_evidence(paths: ReleasePaths) -> dict[str, Path]:
    return {
        name: getattr(paths, name)
        for name in (
            "raw_join_source", "primary_source", "radar_source", "uncertainty_source",
            "primary_root", "mask_root", "canonical_target", "evaluation_lock",
            "canonical_target_receipt", "primary_evaluation_spec",
            "uncertainty_evaluation_spec", "uncertainty_calibration", "predictions_seal",
            "uncertainty_inputs_seal", "primary_output_dir", "primary_report",
            "primary_csv", "primary_receipt", "radar_output_dir", "radar_report",
            "radar_csv", "radar_receipt", "uncertainty_output_dir",
            "uncertainty_report", "uncertainty_csv", "uncertainty_receipt",
        )
    }


def build_frozen_argv(
    paths: ReleasePaths | None = None,
    *,
    python_executable: os.PathLike[str] | str | None = None,
) -> list[list[str]]:
    """Return the only authorized subprocess argv, preserving a venv symlink."""

    frozen = paths or release_paths()
    return build_frozen_argv_from_evidence(
        _frozen_argv_evidence(frozen),
        python_executable=sys.executable if python_executable is None else python_executable,
    )


def _producer_command_matches(
    observed: Any, frozen: Sequence[str], *, style: str
) -> bool:
    if not isinstance(observed, list) or not all(isinstance(item, str) for item in observed):
        return False
    expected = list(frozen)
    if style in {"join", "uncertainty"}:
        return observed == expected[1:]
    if len(observed) != len(expected) or observed[1:] != expected[1:]:
        return False
    # Primary/radar currently canonicalize sys.executable in their own receipt.
    # Confirm executable identity while keeping the actual/frozen child argv
    # symlink-preserving.
    try:
        return os.path.samefile(observed[0], expected[0])
    except OSError:
        return os.path.abspath(observed[0]) == os.path.abspath(expected[0])


def _group_state(files: Sequence[Path], label: str) -> str:
    exists = [path.exists() for path in files]
    if all(exists):
        return "complete"
    if any(exists):
        raise ReleaseEvaluationError(f"partial immutable {label} publication is quarantined")
    return "absent"


def _join_outputs(paths: ReleasePaths) -> tuple[Path, ...]:
    return paths.joined_output, paths.joined_metrics, paths.evaluation_lock


def _evaluation_outputs(paths: ReleasePaths) -> tuple[tuple[Path, ...], ...]:
    return (
        (paths.primary_report, paths.primary_csv, paths.primary_receipt),
        (paths.radar_report, paths.radar_csv, paths.radar_receipt),
        (paths.uncertainty_report, paths.uncertainty_csv, paths.uncertainty_receipt),
    )


def _validate_join(
    paths: ReleasePaths, authorization: Mapping[str, Any], frozen_command: Sequence[str]
) -> dict[str, Any]:
    if _group_state(_join_outputs(paths), "target join") != "complete":
        raise ReleaseEvaluationError("target join is absent")
    for path, label in zip(
        _join_outputs(paths), ("joined OOF", "joined metrics", "evaluation lock"), strict=True
    ):
        _require_0444(path, label)
    lock = _read_json(paths.evaluation_lock, "evaluation lock")
    if (
        lock.get("schema_version") != SCHEMA_VERSION
        or lock.get("classification") != "locked_hcs_oof_single_target_join_seal"
        or lock.get("target_join_count") != 1
        or lock.get("commercial_claim_authorized") is not False
        or lock.get("prospective_confirmation_required") is not True
        or not _producer_command_matches(
            lock.get("orchestrator_command"), frozen_command, style="join"
        )
    ):
        raise ReleaseEvaluationError("evaluation lock invariants/frozen argv differ")
    _assert_binding(
        lock.get("predictions_seal"), authorization["predictions_seal"], "evaluation lock predictions"
    )
    _assert_binding(
        lock.get("target_artifact"), authorization["canonical_target"], "evaluation lock target"
    )
    outputs = lock.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"joined_oof", "metrics"}:
        raise ReleaseEvaluationError("evaluation lock output topology differs")
    _assert_binding(outputs["joined_oof"], bind_file(paths.joined_output), "joined OOF")
    _assert_binding(outputs["metrics"], bind_file(paths.joined_metrics), "joined metrics")
    return lock


_EVALUATION_CLASSES = {
    "primary": (
        "retrospective_locked_hcs_oof_primary_evaluation",
        "retrospective_locked_hcs_oof_primary_evaluation_receipt",
    ),
    "radar_masks": (
        "retrospective_locked_hcs_all_radar_masks_evaluation",
        "retrospective_locked_hcs_all_radar_masks_evaluation_receipt",
    ),
    "uncertainty": (
        "retrospective_locked_hcs_uncertainty_evaluation",
        "retrospective_locked_hcs_uncertainty_evaluation_receipt",
    ),
}

_EVALUATOR_INPUT_KEYS = {
    "primary": frozenset(
        {
            "evaluation_lock",
            "predictions_seal",
            "target_receipt",
            "target_artifact",
            "joined_oof",
            "locked_metrics",
            "evaluation_spec",
        }
    ),
    "radar_masks": frozenset(
        {
            "radar_mask_complete_seal",
            "radar_mask_plan",
            "radar_mask_preexecution_lock",
            "primary_predictions_seal",
            "evaluation_lock",
            "predictions_seal",
            "target_receipt",
            "target_artifact",
            "joined_oof",
            "locked_metrics",
            "evaluation_spec",
        }
    ),
    "uncertainty": frozenset(
        {
            "secondary_uncertainty_evaluation_spec",
            "evaluation_spec",
            "calibration",
            "predictions_seal",
            "uncertainty_inputs_seal",
            "uncertainty_archive",
            "calibration_declared_bindings_rehashed",
            "prediction_declared_bindings_rehashed",
            "uncertainty_declared_bindings_rehashed",
            "secondary_protocol_role",
            "primary_uncertainty_contract_overridden",
            "dedicated_secondary_spec_verified_before_prediction_uncertainty_or_target_access",
            "all_target_free_inputs_verified_before_evaluation_lock_access",
            "all_uncertainty_array_schema_and_hashes_verified",
            "evaluation_lock",
            "target_receipt",
            "target_artifact",
            "joined_oof",
            "locked_metrics",
        }
    ),
}

_UNCERTAINTY_PRETARGET_AUDIT_KEYS = frozenset(
    {
        "secondary_uncertainty_evaluation_spec",
        "evaluation_spec",
        "calibration",
        "predictions_seal",
        "uncertainty_inputs_seal",
        "uncertainty_archive",
        "calibration_declared_bindings_rehashed",
        "prediction_declared_bindings_rehashed",
        "uncertainty_declared_bindings_rehashed",
        "secondary_protocol_role",
        "primary_uncertainty_contract_overridden",
        "dedicated_secondary_spec_verified_before_prediction_uncertainty_or_target_access",
        "all_target_free_inputs_verified_before_evaluation_lock_access",
        "all_uncertainty_array_schema_and_hashes_verified",
    }
)


def _validate_evaluation(
    *,
    name: str,
    files: Sequence[Path],
    authorization: Mapping[str, Any],
    evaluation_lock_binding: Mapping[str, Any],
    frozen_command: Sequence[str],
) -> dict[str, Any]:
    if _group_state(files, f"{name} evaluation") != "complete":
        raise ReleaseEvaluationError(f"{name} evaluation is absent")
    report_path, csv_path, receipt_path = files
    for path, suffix in zip(files, ("report", "CSV", "receipt"), strict=True):
        _require_0444(path, f"{name} {suffix}")
    report = _read_json(report_path, f"{name} report")
    receipt = _read_json(receipt_path, f"{name} receipt")
    _verify_content(report, f"{name} report")
    _verify_content(receipt, f"{name} receipt")
    report_class, receipt_class = _EVALUATION_CLASSES[name]
    style = "uncertainty" if name == "uncertainty" else name
    if (
        report.get("classification") != report_class
        or receipt.get("classification") != receipt_class
        or report.get("commercial_claim_authorized") is not False
        or report.get("prospective_confirmation_required") is not True
        or receipt.get("commercial_claim_authorized") is not False
        or receipt.get("prospective_confirmation_required") is not True
        or receipt.get("outputs_create_once") is not True
        or receipt.get("output_overwrite_allowed") is not False
        or not _producer_command_matches(
            report.get("orchestrator_command"), frozen_command, style=style
        )
        or not _producer_command_matches(
            receipt.get("orchestrator_command"), frozen_command, style=style
        )
    ):
        raise ReleaseEvaluationError(f"{name} report/receipt invariants or argv differ")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"report", "metrics_csv"}:
        raise ReleaseEvaluationError(f"{name} receipt output topology differs")
    report_binding = bind_file(report_path)
    csv_binding = bind_file(csv_path)
    receipt_binding = bind_file(receipt_path)
    _assert_binding(outputs["report"], report_binding, f"{name} report")
    _assert_binding(outputs["metrics_csv"], csv_binding, f"{name} CSV")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ReleaseEvaluationError(f"{name} receipt inputs are absent")
    validate_evaluator_receipt_edges(
        name=name,
        inputs=inputs,
        authorization=authorization,
        evaluation_lock_binding=evaluation_lock_binding,
    )
    if name in {"primary", "radar_masks"}:
        _assert_binding(
            report.get("evaluation_specification"),
            authorization["primary_evaluation_spec"],
            f"{name} primary evaluation specification",
        )
    else:
        _assert_binding(
            report.get("uncertainty_evaluation_specification"),
            authorization["uncertainty_evaluation_spec"],
            "uncertainty evaluation specification",
        )
    return {
        "report": report_binding,
        "receipt": receipt_binding,
    }


def validate_evaluator_receipt_edges(
    *,
    name: str,
    inputs: Mapping[str, Any],
    authorization: Mapping[str, Any],
    evaluation_lock_binding: Mapping[str, Any],
) -> None:
    """Validate every canonical input edge material to one evaluator receipt."""

    expected_keys = _EVALUATOR_INPUT_KEYS.get(name)
    if expected_keys is None:
        raise ReleaseEvaluationError(f"unknown evaluator receipt role: {name}")
    if set(inputs) != expected_keys:
        missing = sorted(expected_keys - set(inputs))
        extra = sorted(set(inputs) - expected_keys)
        raise ReleaseEvaluationError(
            f"{name} receipt input topology differs (missing={missing}, extra={extra})"
        )

    expected: dict[str, Mapping[str, Any]] = {
        "evaluation_lock": evaluation_lock_binding,
        "predictions_seal": authorization["predictions_seal"],
        "target_receipt": authorization["canonical_target_receipt"],
        "target_artifact": authorization["canonical_target"],
        "joined_oof": authorization["joined_oof"],
        "locked_metrics": authorization["locked_metrics"],
    }
    if name == "primary":
        expected["evaluation_spec"] = authorization["primary_evaluation_spec"]
    elif name == "radar_masks":
        expected.update(
            {
                "evaluation_spec": authorization["primary_evaluation_spec"],
                "primary_predictions_seal": authorization["predictions_seal"],
                "radar_mask_complete_seal": authorization["radar_mask_complete_seal"],
                "radar_mask_plan": authorization["radar_mask_plan"],
                "radar_mask_preexecution_lock": authorization[
                    "radar_mask_preexecution_lock"
                ],
            }
        )
    elif name == "uncertainty":
        audit = authorization.get("uncertainty_pretarget_audit")
        if not isinstance(audit, Mapping) or set(audit) != _UNCERTAINTY_PRETARGET_AUDIT_KEYS:
            raise ReleaseEvaluationError(
                "uncertainty target-free authorization audit topology differs"
            )
        expected.update(
            {
                "secondary_uncertainty_evaluation_spec": authorization[
                    "uncertainty_evaluation_spec"
                ],
                "evaluation_spec": authorization["primary_evaluation_spec"],
                "calibration": authorization["uncertainty_calibration"],
                "uncertainty_inputs_seal": authorization["uncertainty_inputs_seal"],
                "uncertainty_archive": authorization["uncertainty_archive"],
            }
        )
    for role, binding in expected.items():
        _assert_binding(inputs.get(role), binding, f"{name} receipt {role}")
    if name == "uncertainty":
        audit = authorization["uncertainty_pretarget_audit"]
        for role in (
            "calibration_declared_bindings_rehashed",
            "prediction_declared_bindings_rehashed",
            "uncertainty_declared_bindings_rehashed",
        ):
            expected_count = audit.get(role)
            observed_count = inputs.get(role)
            if (
                type(expected_count) is not int
                or expected_count <= 0
                or type(observed_count) is not int
                or observed_count != expected_count
            ):
                raise ReleaseEvaluationError(
                    f"uncertainty receipt {role} differs"
                )
        if (
            audit.get("secondary_protocol_role")
            != "separate_secondary_retrospective_engineering_protocol"
            or inputs.get("secondary_protocol_role")
            != "separate_secondary_retrospective_engineering_protocol"
        ):
            raise ReleaseEvaluationError(
                "uncertainty receipt secondary_protocol_role differs"
            )
        expected_flags = {
            "primary_uncertainty_contract_overridden": False,
            "dedicated_secondary_spec_verified_before_prediction_uncertainty_or_target_access": True,
            "all_target_free_inputs_verified_before_evaluation_lock_access": True,
            "all_uncertainty_array_schema_and_hashes_verified": True,
        }
        for role, expected_value in expected_flags.items():
            if audit.get(role) is not expected_value or inputs.get(role) is not expected_value:
                raise ReleaseEvaluationError(f"uncertainty receipt {role} differs")


def _evaluation_authorization(
    paths: ReleasePaths, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    """Add the two target-join outputs required by every evaluator receipt."""

    return {
        **authorization,
        "joined_oof": bind_file(paths.joined_output),
        "locked_metrics": bind_file(paths.joined_metrics),
    }


def _expected_attestation(
    *,
    paths: ReleasePaths,
    authorization: Mapping[str, Any],
    frozen_argv: Sequence[Sequence[str]],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "release_readiness_spec": dict(authorization["release_readiness_spec"]),
        "pretarget_release_lock": dict(authorization["pretarget_release_lock"]),
        "target_release_receipt": dict(authorization["target_release_receipt"]),
        "canonical_target": dict(authorization["canonical_target"]),
        "canonical_target_receipt": dict(authorization["canonical_target_receipt"]),
        "predictions_seal": dict(authorization["predictions_seal"]),
        "evaluation_lock": bind_file(paths.evaluation_lock),
        "primary_evaluation_spec": dict(authorization["primary_evaluation_spec"]),
        "uncertainty_evaluation_spec": dict(
            authorization["uncertainty_evaluation_spec"]
        ),
        "radar_mask_complete_seal": dict(authorization["radar_mask_complete_seal"]),
        "uncertainty_inputs_seal": dict(authorization["uncertainty_inputs_seal"]),
        "execution_source": bind_file(paths.execution_source),
        "frozen_argv": [list(command) for command in frozen_argv],
        "executed_commands": [list(command) for command in frozen_argv],
        "evaluations": {
            name: {
                "report": dict(evaluations[name]["report"]),
                "receipt": dict(evaluations[name]["receipt"]),
            }
            for name in ("primary", "radar_masks", "uncertainty")
        },
        "all_inputs_and_outputs_live_rehashed": True,
        "target_re_evaluation_performed": False,
        "all_steps_executed_once": True,
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
    }
    document["content_sha256"] = canonical_sha256(document)
    return document


def _write_immutable_json(path: Path, document: Mapping[str, Any]) -> None:
    destination = _absolute(path)
    payload = (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
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
        # The destination is immutable at the instant it becomes visible; a
        # crash cannot strand an otherwise-valid attestation in mode 0600.
        temporary.chmod(0o444)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ReleaseEvaluationError(
                f"execution attestation appeared concurrently: {destination}"
            ) from exc
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def validate_execution_attestation(
    path: Path = DEFAULT_ATTESTATION,
    *,
    paths: ReleasePaths | None = None,
    authorization: Mapping[str, Any] | None = None,
    frozen_argv: Sequence[Sequence[str]] | None = None,
) -> dict[str, Any]:
    frozen_paths = paths or release_paths()
    if _absolute(path) != frozen_paths.attestation:
        raise ReleaseEvaluationError("execution attestation path is not release-frozen")
    auth = dict(authorization or verify_release_authorization(frozen_paths))
    commands = [list(item) for item in (frozen_argv or build_frozen_argv(frozen_paths))]
    _validate_join(frozen_paths, auth, commands[0])
    auth = _evaluation_authorization(frozen_paths, auth)
    evaluation_lock_binding = bind_file(frozen_paths.evaluation_lock)
    evaluations = {
        name: _validate_evaluation(
            name=name,
            files=files,
            authorization=auth,
            evaluation_lock_binding=evaluation_lock_binding,
            frozen_command=command,
        )
        for name, files, command in zip(
            ("primary", "radar_masks", "uncertainty"),
            _evaluation_outputs(frozen_paths),
            commands[1:],
            strict=True,
        )
    }
    expected = _expected_attestation(
        paths=frozen_paths,
        authorization=auth,
        frozen_argv=commands,
        evaluations=evaluations,
    )
    _require_0444(frozen_paths.attestation, "release evaluation execution attestation")
    observed = _read_json(
        frozen_paths.attestation, "release evaluation execution attestation"
    )
    _verify_content(observed, "release evaluation execution attestation")
    if observed != expected:
        raise ReleaseEvaluationError(
            "execution attestation differs from the live release/evaluation graph"
        )
    return observed


CommandRunner = Callable[[Sequence[str], Path], Any]


def _execute_command(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - argv is a fixed internal allow-list
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _invoke(command: Sequence[str], *, cwd: Path, runner: CommandRunner | None) -> None:
    execute = runner or _execute_command
    result = execute(list(command), cwd)
    returncode = result if isinstance(result, int) else getattr(result, "returncode", None)
    if returncode != 0:
        stderr = "" if isinstance(result, int) else str(getattr(result, "stderr", ""))
        tail = stderr[-2000:].strip()
        raise ReleaseEvaluationError(
            f"fixed evaluation command failed with exit {returncode}"
            + (f": {tail}" if tail else "")
        )


def _verify_ordered_states(paths: ReleasePaths) -> list[str]:
    groups = (_join_outputs(paths), *_evaluation_outputs(paths))
    labels = ("target join", "primary evaluation", "radar-mask evaluation", "uncertainty evaluation")
    states = [_group_state(group, label) for group, label in zip(groups, labels, strict=True)]
    absent_seen = False
    for state in states:
        if state == "absent":
            absent_seen = True
        elif absent_seen:
            raise ReleaseEvaluationError("evaluation outputs exist out of frozen DAG order")
    return states


def run_release_locked_evaluation(
    project_root: Path = PROJECT_ROOT,
    *,
    canonical_target: Path | None = None,
    requested_argv: Sequence[Sequence[str]] | None = None,
    command_runner: CommandRunner | None = None,
    python_executable: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Run/resume the one fixed release evaluation without ever re-evaluating."""

    paths = release_paths(project_root)
    if canonical_target is not None and _absolute(canonical_target) != paths.canonical_target:
        raise ReleaseEvaluationError("an alternate canonical target is forbidden")
    frozen = build_frozen_argv(paths, python_executable=python_executable)
    if requested_argv is not None and [list(row) for row in requested_argv] != frozen:
        raise ReleaseEvaluationError("requested evaluator argv differs from the frozen argv")

    # Always establish release authorization before even examining a
    # target-derived execution attestation or output state.
    authorization = verify_release_authorization(paths)
    if paths.attestation.exists():
        return validate_execution_attestation(
            paths.attestation,
            paths=paths,
            authorization=authorization,
            frozen_argv=frozen,
        )

    _verify_ordered_states(paths)
    groups = (_join_outputs(paths), *_evaluation_outputs(paths))
    names = ("join", "primary", "radar_masks", "uncertainty")
    evaluation_results: dict[str, dict[str, Any]] = {}
    for position, (name, files, command) in enumerate(
        zip(names, groups, frozen, strict=True)
    ):
        # Immediate pre-subprocess drift gate.  It reruns public release and
        # prediction verifiers; no cached authorization is trusted.
        authorization = verify_release_authorization(paths)
        current_states = _verify_ordered_states(paths)
        if current_states[position] == "absent":
            _invoke(command, cwd=paths.project_root, runner=command_runner)
            # A producer success is not trusted: live-rehash release authority
            # and every output before the next command becomes reachable.
            authorization = verify_release_authorization(paths)
        if name == "join":
            _validate_join(paths, authorization, command)
        else:
            evaluation_results[name] = _validate_evaluation(
                name=name,
                files=files,
                authorization=_evaluation_authorization(paths, authorization),
                evaluation_lock_binding=bind_file(paths.evaluation_lock),
                frozen_command=command,
            )

    # One final full live pass closes TOCTOU between the last child and the
    # immutable canonical attestation.
    authorization = verify_release_authorization(paths)
    _validate_join(paths, authorization, frozen[0])
    authorization = _evaluation_authorization(paths, authorization)
    evaluation_results = {
        name: _validate_evaluation(
            name=name,
            files=files,
            authorization=authorization,
            evaluation_lock_binding=bind_file(paths.evaluation_lock),
            frozen_command=command,
        )
        for name, files, command in zip(
            names[1:], _evaluation_outputs(paths), frozen[1:], strict=True
        )
    }
    document = _expected_attestation(
        paths=paths,
        authorization=authorization,
        frozen_argv=frozen,
        evaluations=evaluation_results,
    )
    _write_immutable_json(paths.attestation, document)
    return validate_execution_attestation(
        paths.attestation,
        paths=paths,
        authorization=authorization,
        frozen_argv=frozen,
    )


# Concise public alias for callers/tests while retaining the descriptive ABI.
run_release_evaluation = run_release_locked_evaluation


def build_parser() -> argparse.ArgumentParser:
    # Intentionally no target/path/passthrough arguments.  argparse therefore
    # fails closed for every production attempt to alter the fixed DAG.
    return argparse.ArgumentParser(description=__doc__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    try:
        result = run_release_locked_evaluation()
    except (ReleaseEvaluationError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
