#!/usr/bin/env python3
"""Train and predict the globally locked 6-fold x 3-seed v3r1 matrix.

Promotion is a fixed execution, not a second search.  This driver requires the
create-once discovery selection and additive promotion authorization, trains
the selected variant for every fold/seed without branching on validation
scores, creates exact-cover target-free inference inputs, invokes the trainer's
strict prediction mode, and finally seals all 18 predictions.  No code path in
this module reads, accepts, joins, or scores an outer-test target.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import build_locked_hfr_v3r1_test_inputs as locked_inputs  # noqa: E402
import run_hfr_v3r1_discovery_campaign as discovery  # noqa: E402


DEFAULT_RUN_ROOT = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/fixed_oof_v8r4"
)
FIXED_AGGREGATION_OUTPUT_RELATIVE = DEFAULT_RUN_ROOT / "aggregation_v8r4a"
CAMPAIGN_REVISION = "V8R4"
INFRASTRUCTURE_REVISION = discovery.INFRASTRUCTURE_REVISION
TARGET_SEALED_RUNTIME_RELATIVE = Path("scripts/run_hfr_v3r1_target_sealed.py")
CAPABILITY_RECEIPT_NAME = "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json"
PROMOTION_TRAINING_INDEX_CLASSIFICATION = (
    "adaptive_v3r1_v8r4_nonouter_training_index"
)
PREDICTION_INDEX_CLASSIFICATION = (
    "adaptive_v3r1_v8r4a_model_bound_target_free_prediction_shard_index"
)
PROMOTION_MODEL_SHARD_SEAL_CLASSIFICATION = (
    "adaptive_v3r1_v8r4a_promotion_model_shard_completion_seal"
)
PREDICTION_SHARD_SEAL_CLASSIFICATION = (
    "adaptive_v3r1_v8r4a_prediction_shard_completion_seal"
)
PROMOTION_EXACT_COVER_CLASSIFICATION = (
    "adaptive_v3r1_v8r4a_pack_free_promotion_prediction_exact_cover"
)
PREDICTION_FIELDS = (
    "cache_index",
    "prediction_bpm",
    "prediction_available",
    "raw_anchor_bpm",
    "raw_anchor_available",
    "hard_source_bpm",
    "hard_source_available",
    "selected_source_probability",
    "selected_source_code",
    "source_scale_bpm",
    "quality",
    "factor_probabilities",
    "spike_rate",
)
FORBIDDEN_PREDICTION_TOKENS = (
    "target",
    "reference",
    "identity",
    "protocol",
    "ground_truth",
    "label",
)

EXPECTED_REUSE_UNITS = frozenset(
    (fold, seed)
    for fold in locked_inputs.REUSED_DISCOVERY_FOLDS
    for seed in discovery.SEEDS
)
EXPECTED_NEW_PROMOTION_TRAINING_UNITS = frozenset(
    (fold, seed)
    for fold in locked_inputs.NEW_PROMOTION_TRAINING_FOLDS
    for seed in discovery.SEEDS
)
EXPECTED_ALL_PREDICTION_UNITS = frozenset(
    (fold, seed) for fold in range(6) for seed in discovery.SEEDS
)
FIXED_ACTIVE_GPU_OWNER_COUNT = 49


def v8r4_fixed_execution_plan() -> dict[str, Any]:
    """Return the immutable 6-reuse/12-train/18-predict owner topology."""

    reuse = {(fold, seed) for fold in (3, 4) for seed in discovery.SEEDS}
    train = {(fold, seed) for fold in (0, 1, 2, 5) for seed in discovery.SEEDS}
    predict = {(fold, seed) for fold in range(6) for seed in discovery.SEEDS}
    if not (
        reuse == set(EXPECTED_REUSE_UNITS)
        and train == set(EXPECTED_NEW_PROMOTION_TRAINING_UNITS)
        and predict == set(EXPECTED_ALL_PREDICTION_UNITS)
        and reuse.isdisjoint(train)
        and reuse | train == predict
        and 1 + 18 + len(train) + len(predict) == FIXED_ACTIVE_GPU_OWNER_COUNT
    ):
        raise discovery.CampaignError("V8R4 fixed owner topology drifted")
    return {
        "campaign_revision": CAMPAIGN_REVISION,
        "clean_discovery_reuse_units": sorted(reuse),
        "new_promotion_training_units": sorted(train),
        "target_free_prediction_units": sorted(predict),
        "nonaccounting_reuse_pointer_count": 6,
        "active_gpu_owner_count": FIXED_ACTIVE_GPU_OWNER_COUNT,
        "active_gpu_owner_breakdown": {
            "benchmark": 1,
            "clean_v8r4_discovery": 18,
            "new_promotion_training": 12,
            "target_free_prediction": 18,
        },
        "v8r3_quarantine_in_active_owner_count": False,
        "cross_outer_validation_reuse_present": True,
        "fully_nested_confirmatory_oof": False,
        "prospective_confirmation_required": True,
    }


def _admitted_authorization_from_governance(
    governance: Mapping[str, Any],
) -> tuple[Path, str]:
    binding = governance.get("pretrain_authorization")
    if not isinstance(binding, Mapping):
        raise discovery.CampaignError("promotion lacks its V8 pretrain binding")
    raw_path = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(digest, str):
        raise discovery.CampaignError("promotion V8 pretrain binding is malformed")
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(), digest


def _atomic_create_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise discovery.CampaignError(f"create-once NPZ already exists: {path}")
    if not hasattr(os, "O_TMPFILE"):
        raise discovery.CampaignError("O_TMPFILE is required for kill-safe NPZ publication")
    descriptor = os.open(path.parent, os.O_RDWR | os.O_TMPFILE, 0o600)
    try:
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        anonymous = os.fstat(descriptor)
        if not (
            stat.S_ISREG(anonymous.st_mode)
            and stat.S_IMODE(anonymous.st_mode) == 0o444
            and anonymous.st_nlink == 0
        ):
            raise discovery.CampaignError("anonymous NPZ inode is not 0444/nlink0")
        parent_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            parent_flags |= os.O_NOFOLLOW
        parent_fd = os.open(path.parent, parent_flags)
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            linkat = libc.linkat
            linkat.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
            )
            linkat.restype = ctypes.c_int
            if linkat(descriptor, b"", parent_fd, os.fsencode(path.name), 0x1000) != 0:
                number = ctypes.get_errno()
                if number == errno.EEXIST:
                    raise discovery.CampaignError(f"create-once NPZ collision: {path}")
                raise discovery.CampaignError(
                    f"anonymous NPZ publication failed: {os.strerror(number)}"
                )
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        opened = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            published = os.fstat(opened)
            if not (
                stat.S_ISREG(published.st_mode)
                and stat.S_IMODE(published.st_mode) == 0o444
                and published.st_nlink == 1
                and (published.st_dev, published.st_ino)
                == (anonymous.st_dev, anonymous.st_ino)
            ):
                raise discovery.CampaignError(
                    "published NPZ is not the exact 0444/nlink1 inode"
                )
        finally:
            os.close(opened)
    finally:
        os.close(descriptor)


def _read_verified_single_link_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
    expected_mode: int,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    """Capture and authenticate one stable inode without following aliases."""

    lexical = Path(os.path.abspath(path.expanduser()))
    if not (
        isinstance(expected_sha256, str)
        and len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256)
    ):
        raise discovery.CampaignError(f"{label} SHA-256 is malformed")
    if expected_bytes is not None and (
        type(expected_bytes) is not int or expected_bytes < 0
    ):
        raise discovery.CampaignError(f"{label} byte count is malformed")
    try:
        before_path = os.stat(lexical, follow_symlinks=False)
        descriptor = os.open(
            lexical,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise discovery.CampaignError(f"cannot open {label}: {lexical}") from error
    try:
        before_fd = os.fstat(descriptor)
        if not stat.S_ISREG(before_fd.st_mode) or before_fd.st_nlink != 1:
            raise discovery.CampaignError(
                f"{label} must be a single-link regular file"
            )
        if (
            stat.S_ISLNK(before_path.st_mode)
            or before_path.st_nlink != 1
            or (before_path.st_dev, before_path.st_ino)
            != (before_fd.st_dev, before_fd.st_ino)
        ):
            raise discovery.CampaignError(f"{label} path/inode binding is unsafe")
        if stat.S_IMODE(before_fd.st_mode) != expected_mode:
            raise discovery.CampaignError(
                f"{label} mode must be exactly {expected_mode:04o}"
            )
        digest = hashlib.sha256()
        captured = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            captured.extend(block)
        after_fd = os.fstat(descriptor)
        after_path = os.stat(lexical, follow_symlinks=False)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before_fd, name) != getattr(after_fd, name)
            for name in stable_fields
        ) or any(
            getattr(after_fd, name) != getattr(after_path, name)
            for name in stable_fields
        ):
            raise discovery.CampaignError(f"{label} changed while it was verified")
        raw = bytes(captured)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise discovery.CampaignError(f"{label} SHA-256 drifted")
        if expected_bytes is not None and len(raw) != expected_bytes:
            raise discovery.CampaignError(f"{label} byte count drifted")
        return (
            {
                "path": str(lexical),
                "sha256": actual_sha256,
                "bytes": len(raw),
            },
            raw,
        )
    finally:
        os.close(descriptor)


def _load_json_bytes(raw: bytes, *, path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=discovery._unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                discovery.CampaignError(
                    f"non-finite JSON constant in {label}: {token}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise discovery.CampaignError(f"invalid {label}: {path} ({error})") from error
    if not isinstance(value, dict):
        raise discovery.CampaignError(f"{label} must be a JSON object: {path}")
    return value


def _verify_exact_v2_binding(
    binding: Any,
    *,
    expected_path: Path,
    project_root: Path,
    owner: Path,
    label: str,
) -> Path:
    if not isinstance(binding, Mapping):
        raise discovery.CampaignError(f"{label} binding is missing")
    path = discovery.verify_binding(
        binding,
        project_root=project_root,
        owner=owner,
        label=label,
    )
    if path != expected_path.resolve() or path.is_symlink() or not path.is_file():
        raise discovery.CampaignError(f"{label} is not the canonical V2 artifact")
    return path


def _validate_exact_v2_binding_reference(
    binding: Any,
    *,
    expected_path: Path,
    project_root: Path,
    owner: Path,
    label: str,
) -> dict[str, Any]:
    """Validate a signed leaf reference without opening its path."""

    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise discovery.CampaignError(f"{label} binding is non-canonical")
    path = discovery.resolve_binding_path(
        binding["path"], project_root=project_root, owner=owner
    )
    digest = binding["sha256"]
    byte_count = binding["bytes"]
    if path != expected_path.resolve():
        raise discovery.CampaignError(f"{label} is not the canonical V2 artifact")
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    ):
        raise discovery.CampaignError(f"{label} SHA-256 is malformed")
    if type(byte_count) is not int or byte_count < 0:
        raise discovery.CampaignError(f"{label} byte count is malformed")
    return {
        "path": str(expected_path.resolve()),
        "sha256": digest,
        "bytes": byte_count,
    }


def _validate_canonical_v2_anchor_source(
    *,
    source: Path,
    outer_fold: int,
    seed: int,
    project_root: Path,
    canonical_v2_root: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Authenticate one V2 no-action anchor through its pretarget owner graph."""

    registered_root = (project_root / DEFAULT_V2_LOCKED_ROOT).resolve()
    requested_root = (
        registered_root
        if canonical_v2_root is None
        else canonical_v2_root.expanduser().resolve()
    )
    if requested_root != registered_root:
        raise discovery.CampaignError(
            "V2 anchor root override does not equal the canonical locked root"
        )
    unit_name = f"outer_{outer_fold}_seed_{seed}"
    expected_source = requested_root / unit_name / "work/no_action_raw_hcs.npz"
    lexical_source = Path(os.path.abspath(source.expanduser()))
    if lexical_source != expected_source:
        raise discovery.CampaignError(
            "V2 proposer anchor source is outside the canonical locked unit"
        )

    campaign_root = requested_root.parent
    release_path = campaign_root / "pretarget_release_lock.json"
    predictions_seal_path = campaign_root / "predictions_seal.json"
    pretest_lock_path = campaign_root / "pretest_lock.json"
    derived_lock_path = requested_root / unit_name / "derived_inference_lock.json"
    stage_receipt_path = (
        requested_root
        / unit_name
        / "receipts/02_no_action_fallback_adapter.json"
    )
    sealed_prediction_path = requested_root / unit_name / "sealed_label_free_predictions.npz"
    release_binding, release_raw = _read_verified_single_link_file(
        release_path,
        expected_sha256=V2_PRETARGET_RELEASE_LOCK_FILE_SHA256,
        expected_bytes=None,
        expected_mode=0o444,
        label="V2 pretarget release lock",
    )
    for path, label in (
        (predictions_seal_path, "V2 predictions seal"),
        (pretest_lock_path, "V2 pretest lock"),
        (derived_lock_path, "V2 derived inference lock"),
        (stage_receipt_path, "V2 no-action stage receipt"),
    ):
        if not path.is_file() or path.is_symlink():
            raise discovery.CampaignError(f"{label} is missing or symlinked")

    release = _load_json_bytes(
        release_raw,
        path=release_path,
        label="canonical V2 pretarget release lock",
    )
    primary = release.get("boundaries", {}).get("primary_predictions", {})
    if not (
        discovery.canonical_content_sha256(release)
        == release.get("content_sha256")
        and release.get("classification") == "locked_hcs_pretarget_release_lock"
        and release.get("status") == "all_target_free_boundaries_complete"
        and release.get("target_or_label_artifact_opened") is False
        and release.get("commercial_claim_authorized") is False
        and isinstance(primary, Mapping)
        and primary.get("classification")
        == "primary_18_unit_predictions_revalidated"
        and primary.get("target_metadata_opened") is False
        and primary.get("unit_count") == 18
    ):
        raise discovery.CampaignError("canonical V2 pretarget release invariants drifted")
    predictions_seal_binding = primary.get("predictions_seal")
    pretest_lock_binding = primary.get("pretest_lock")
    _verify_exact_v2_binding(
        predictions_seal_binding,
        expected_path=predictions_seal_path,
        project_root=project_root,
        owner=release_path,
        label="V2 release-bound predictions seal",
    )
    _verify_exact_v2_binding(
        pretest_lock_binding,
        expected_path=pretest_lock_path,
        project_root=project_root,
        owner=release_path,
        label="V2 release-bound pretest lock",
    )

    predictions_seal = discovery.load_json(
        predictions_seal_path, "canonical V2 predictions seal"
    )
    units = predictions_seal.get("units")
    if not (
        predictions_seal.get("classification")
        == "locked_hcs_oof_all_label_free_predictions_sealed"
        and predictions_seal.get("target_artifact_opened_before_seal") is False
        and predictions_seal.get("unit_count") == 18
        and predictions_seal.get("outer_folds") == list(range(6))
        and isinstance(units, list)
        and len(units) == 18
        and predictions_seal.get("pretest_lock_sha256")
        == discovery.sha256_file(pretest_lock_path)
    ):
        raise discovery.CampaignError("canonical V2 predictions seal drifted")
    indexed: dict[tuple[int, int], Mapping[str, Any]] = {}
    for unit in units:
        if not isinstance(unit, Mapping):
            raise discovery.CampaignError("canonical V2 predictions unit is invalid")
        key = (int(unit.get("outer_fold", -1)), int(unit.get("seed", -1)))
        if key in indexed:
            raise discovery.CampaignError("canonical V2 predictions unit is duplicated")
        indexed[key] = unit
    expected_units = {
        (fold, expected_seed)
        for fold in range(6)
        for expected_seed in discovery.SEEDS
    }
    if set(indexed) != expected_units:
        raise discovery.CampaignError("canonical V2 predictions matrix drifted")
    selected_unit = indexed[(outer_fold, seed)]
    derived_lock_binding = selected_unit.get("derived_lock")
    _verify_exact_v2_binding(
        derived_lock_binding,
        expected_path=derived_lock_path,
        project_root=project_root,
        owner=predictions_seal_path,
        label="V2 predictions-seal derived lock",
    )
    _verify_exact_v2_binding(
        selected_unit.get("prediction"),
        expected_path=sealed_prediction_path,
        project_root=project_root,
        owner=predictions_seal_path,
        label="V2 predictions-seal label-free prediction",
    )

    derived = discovery.load_json(derived_lock_path, "canonical V2 derived inference lock")
    if not (
        derived.get("classification") == "locked_hcs_oof_derived_test_inference"
        and derived.get("outer_fold") == outer_fold
        and derived.get("seed") == seed
        and derived.get("target_artifact_opened") is False
        and derived.get("frozen_policy_status") == "fail_closed_no_action"
        and derived.get("no_action_bit_exact_float32_fallback") is True
        and derived.get("capacity_policy_checkpoint_reselection_performed") is False
        and derived.get("pretest_lock_sha256")
        == discovery.sha256_file(pretest_lock_path)
        and derived.get("sealed_prediction") == selected_unit.get("prediction")
    ):
        raise discovery.CampaignError("canonical V2 derived inference lock drifted")
    raw_binding = _validate_exact_v2_binding_reference(
        derived.get("derived_artifacts", {}).get("raw_hcs_prediction"),
        expected_path=expected_source,
        project_root=project_root,
        owner=derived_lock_path,
        label="V2 derived raw HCS anchor",
    )
    stage_bindings = derived.get("stage_receipts")
    if not isinstance(stage_bindings, list) or len(stage_bindings) != 3:
        raise discovery.CampaignError("canonical V2 stage-receipt graph drifted")
    selected_stage: Mapping[str, Any] | None = None
    for number, binding in enumerate(stage_bindings):
        if not isinstance(binding, Mapping):
            raise discovery.CampaignError("canonical V2 stage binding is invalid")
        path = discovery.verify_binding(
            binding,
            project_root=project_root,
            owner=derived_lock_path,
            label=f"V2 stage receipt {number}",
        )
        if path == stage_receipt_path:
            selected_stage = binding
    if selected_stage is None:
        raise discovery.CampaignError("canonical V2 no-action stage receipt is unowned")
    stage = discovery.load_json(stage_receipt_path, "canonical V2 no-action stage receipt")
    outputs = stage.get("outputs")
    if not (
        stage.get("classification") == "locked_hcs_oof_stage_receipt"
        and stage.get("stage") == "no_action_fallback_adapter"
        and isinstance(outputs, list)
        and len(outputs) == 1
    ):
        raise discovery.CampaignError("canonical V2 no-action stage receipt drifted")
    stage_raw_binding = _validate_exact_v2_binding_reference(
        outputs[0],
        expected_path=expected_source,
        project_root=project_root,
        owner=stage_receipt_path,
        label="V2 no-action stage output",
    )
    if stage_raw_binding != raw_binding:
        raise discovery.CampaignError(
            "canonical V2 raw anchor owners bind different bytes"
        )
    assert isinstance(predictions_seal_binding, Mapping)
    assert isinstance(pretest_lock_binding, Mapping)
    assert isinstance(derived_lock_binding, Mapping)
    assert isinstance(selected_stage, Mapping)
    return (
        {
            "pretarget_release_lock": release_binding,
            "pretest_lock": dict(pretest_lock_binding),
            "predictions_seal": dict(predictions_seal_binding),
            "derived_inference_lock": dict(derived_lock_binding),
            "no_action_stage_receipt": dict(selected_stage),
        },
        raw_binding,
    )


def _safe_anchor_from_locked_v2(
    *,
    source: Path,
    output: Path,
    receipt_path: Path,
    outer_fold: int,
    seed: int,
    project_root: Path = PROJECT_ROOT,
    canonical_v2_root: Path | None = None,
) -> dict[str, Any]:
    """Narrow an already target-free v2 inference archive to four safe fields."""

    owner_chain, expected_source_binding = _validate_canonical_v2_anchor_source(
        source=source,
        outer_fold=outer_fold,
        seed=seed,
        project_root=project_root,
        canonical_v2_root=canonical_v2_root,
    )
    source_binding, source_raw = _read_verified_single_link_file(
        source,
        expected_sha256=str(expected_source_binding["sha256"]),
        expected_bytes=int(expected_source_binding["bytes"]),
        expected_mode=0o600,
        label="canonical V2 raw proposer anchor",
    )
    if source_binding != expected_source_binding:
        raise discovery.CampaignError(
            "canonical V2 raw proposer anchor binding drifted"
        )

    def receipt_value(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
        index = np.asarray(arrays["cache_index"], dtype=np.int64)
        return {
            "schema_version": 1,
            "classification": "adaptive_v3r1_narrowed_target_free_proposer_anchor",
            "campaign_id": discovery.CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "outer_fold": outer_fold,
            "seed": seed,
            "source": source_binding,
            "source_owner_chain": owner_chain,
            "source_fields_read": sorted(required),
            "output": discovery.bind_file(output),
            "output_fields": list(arrays),
            "rows": int(len(index)),
            "target_reference_identity_protocol_qc_fields_emitted": False,
            "commercial_claim_authorized": False,
        }

    try:
        with np.load(io.BytesIO(source_raw), allow_pickle=False) as archive:
            required = {
                "cache_index",
                "fallback_rr_bpm",
                "fallback_std_bpm",
                "fallback_available",
                "outer_fold",
                "seed",
                "target_fields_present",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise discovery.CampaignError(
                    f"locked v2 target-free anchor fields missing: {missing}"
                )
            if bool(np.asarray(archive["target_fields_present"]).item()):
                raise discovery.CampaignError("locked v2 anchor declares target fields")
            if int(np.asarray(archive["outer_fold"]).item()) != outer_fold:
                raise discovery.CampaignError("locked v2 anchor outer-fold mismatch")
            if int(np.asarray(archive["seed"]).item()) != seed:
                raise discovery.CampaignError("locked v2 anchor seed mismatch")
            arrays = {
                "cache_index": np.asarray(archive["cache_index"], dtype=np.int64),
                "proposer_anchor_bpm": np.asarray(
                    archive["fallback_rr_bpm"], dtype=np.float32
                ),
                "proposer_anchor_std_bpm": np.asarray(
                    archive["fallback_std_bpm"], dtype=np.float32
                ),
                "proposer_anchor_available": np.asarray(
                    archive["fallback_available"], dtype=bool
                ),
                "outer_fold": np.asarray(outer_fold, dtype=np.int16),
                "seed": np.asarray(seed, dtype=np.int64),
            }
    except (OSError, ValueError, KeyError) as error:
        raise discovery.CampaignError(f"invalid locked v2 anchor {source}: {error}") from error
    index = arrays["cache_index"]
    if (
        index.ndim != 1
        or len(index) == 0
        or len(np.unique(index)) != len(index)
        or any(np.asarray(arrays[name]).shape != index.shape for name in (
            "proposer_anchor_bpm",
            "proposer_anchor_std_bpm",
            "proposer_anchor_available",
        ))
    ):
        raise discovery.CampaignError("locked v2 anchor topology is invalid")
    available = arrays["proposer_anchor_available"]
    if np.any(available & ~np.isfinite(arrays["proposer_anchor_bpm"])) or np.any(
        available
        & (
            ~np.isfinite(arrays["proposer_anchor_std_bpm"])
            | (arrays["proposer_anchor_std_bpm"] <= 0)
        )
    ):
        raise discovery.CampaignError("locked v2 anchor values are invalid")
    if output.exists() or receipt_path.exists():
        if not (output.is_file() and receipt_path.is_file()):
            raise discovery.CampaignError("partial safe-anchor publication")
        if output.is_symlink() or receipt_path.is_symlink():
            raise discovery.CampaignError("safe-anchor publication is symlinked")
        receipt = discovery.load_json(receipt_path, "safe proposer anchor receipt")
        if discovery.canonical_content_sha256(receipt) != receipt.get("content_sha256"):
            raise discovery.CampaignError("safe proposer anchor receipt drifted")
        try:
            with np.load(output, allow_pickle=False) as archive:
                if set(archive.files) != set(arrays) or len(archive.files) != len(arrays):
                    raise discovery.CampaignError(
                        "resumed safe-anchor field set drifted"
                    )
                for name, expected in arrays.items():
                    observed = np.asarray(archive[name])
                    value = np.asarray(expected)
                    if (
                        observed.dtype != value.dtype
                        or observed.shape != value.shape
                        or not np.array_equal(observed, value, equal_nan=True)
                    ):
                        raise discovery.CampaignError(
                            f"resumed safe-anchor derivation drifted: {name}"
                        )
        except (OSError, ValueError, KeyError) as error:
            if isinstance(error, discovery.CampaignError):
                raise
            raise discovery.CampaignError(
                f"invalid resumed safe-anchor output: {error}"
            ) from error
        expected_receipt = receipt_value(arrays)
        expected_receipt["content_sha256"] = discovery.semantic_sha256(
            expected_receipt
        )
        if receipt != expected_receipt:
            raise discovery.CampaignError(
                "safe proposer anchor receipt differs from canonical V2 provenance"
            )
        return receipt
    _atomic_create_npz(output, arrays)
    return discovery.create_once_json(
        receipt_path,
        receipt_value(arrays),
    )


def _promotion_train_command(
    *,
    python: Path,
    trainer: Path,
    item: discovery.TrainingInput,
    output_dir: Path,
    target_sealed_capability_receipt: Path,
    expected_admitted_context: Mapping[str, Any],
    variant: str,
    promotion_authorization: Path,
    release_mode: str,
    device: str,
    amp: bool,
    smoke_test: bool,
    resume: bool,
) -> list[str]:
    return discovery._trainer_command(
        python=python,
        trainer=trainer,
        training_input=item,
        output_dir=output_dir,
        target_sealed_capability_receipt=target_sealed_capability_receipt,
        expected_admitted_context=expected_admitted_context,
        variant=variant,
        device=device,
        amp=amp,
        smoke_test=smoke_test,
        resume=resume,
        campaign_phase="promotion",
        promotion_authorization=promotion_authorization,
        release_mode=release_mode,
    )


def _create_discovery_reuse_pointer(
    *,
    project_root: Path,
    run_root: Path,
    item: discovery.TrainingInput,
    selection: Mapping[str, Any],
    governance: Mapping[str, Any],
) -> locked_inputs.PromotionModelSource:
    """Create one immutable, non-accounting pointer to selected discovery science."""

    key = (item.outer_fold, item.seed)
    if key not in EXPECTED_REUSE_UNITS:
        raise discovery.CampaignError("discovery pointer is outside the exact six reuse units")
    variant = str(selection.get("selected_variant"))
    seal_binding = selection.get("discovery_completion_seal")
    if not isinstance(seal_binding, Mapping):
        raise discovery.CampaignError("selection lacks the discovery completion seal")
    source = locked_inputs.validate_selected_discovery_source(
        project_root=project_root,
        discovery_completion_seal=seal_binding,
        cache_dir=item.cache_dir,
        outer_fold=item.outer_fold,
        seed=item.seed,
        variant=variant,
    )
    selected_parameter_count = selection.get("selected_parameter_count")
    if type(selected_parameter_count) is not int or int(
        source.receipt.get("validated_output", {}).get("parameter_count", -1)
    ) != int(selected_parameter_count):
        raise discovery.CampaignError(
            "selected discovery reuse parameter count differs from selection lock"
        )
    unit_root = run_root / "training" / f"outer_{item.outer_fold}_seed_{item.seed}"
    local_receipt = unit_root / "completion_receipt.json"
    if os.path.lexists(local_receipt):
        raise discovery.CampaignError(
            "reused promotion unit already has an ambiguous local training receipt"
        )
    pointer_path = unit_root / locked_inputs.REUSE_POINTER_FILENAME
    selection_binding = governance.get("selection_lock")
    promotion_binding = governance.get("promotion_authorization")
    if not isinstance(selection_binding, Mapping) or not isinstance(
        promotion_binding, Mapping
    ):
        raise discovery.CampaignError("reuse pointer governance bindings are incomplete")
    selection_path = discovery.verify_binding(
        selection_binding,
        project_root=project_root,
        owner=project_root / "selection_lock.json",
        label="reuse pointer selection lock",
    )
    promotion_path = discovery.verify_binding(
        promotion_binding,
        project_root=project_root,
        owner=project_root / "promotion_authorization.json",
        label="reuse pointer promotion authorization",
    )
    seal_path = discovery.verify_binding(
        seal_binding,
        project_root=project_root,
        owner=selection_path,
        label="reuse pointer discovery completion seal",
    )
    discovery.create_once_json(
        pointer_path,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_phase_independent_discovery_reuse_pointer",
            "campaign_id": discovery.CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "outer_fold": item.outer_fold,
            "seed": item.seed,
            "variant": variant,
            "source_phase": "discovery",
            "destination_phase": "promotion",
            "scientific_signature_sha256": source.scientific_signature_sha256,
            "discovery_completion_seal": discovery.bind_file(seal_path),
            "source_training_receipt": discovery.bind_file(source.receipt_path),
            "artifacts": dict(source.artifacts),
            "selection_lock": discovery.bind_file(selection_path),
            "promotion_authorization": discovery.bind_file(promotion_path),
            "owns_new_gpu_usage": False,
            "usage_record_sha256s": [],
            "outer_test_opened": False,
            "adaptive_retrospective_only": True,
            "commercial_claim_authorized": False,
        },
    )
    return locked_inputs.resolve_promotion_model_source(
        project_root=project_root,
        run_root=run_root,
        cache_dir=item.cache_dir,
        outer_fold=item.outer_fold,
        seed=item.seed,
        variant=variant,
    )


def _governance_sha256(
    governance: Mapping[str, Any],
    name: str,
    *,
    fallback_path: Path | None = None,
) -> str:
    binding = governance.get(name)
    if isinstance(binding, Mapping) and discovery._is_sha256(binding.get("sha256")):
        return str(binding["sha256"])
    if fallback_path is not None:
        return discovery.sha256_file(fallback_path)
    raise discovery.CampaignError(f"current governance lacks {name} SHA-256")


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        raise discovery.CampaignError(f"{label} schema drifted")


def _run_promotion_training(
    *,
    run_root: Path,
    item: discovery.TrainingInput,
    variant: str,
    selection: Mapping[str, Any],
    governance: Mapping[str, Any],
    target_sealed_capability_receipt: Path,
    python: Path,
    trainer: Path,
    wrapper: Path,
    promotion_authorization: Path,
    gpu_lock: Path,
    gpu_ledger: Path,
    usage_ledger: Path,
    device: str,
    amp: bool,
    smoke_test: bool,
    command_runner: Callable[[Sequence[str], float], tuple[int, float, bool]],
) -> dict[str, Any]:
    if (
        (item.outer_fold, item.seed) not in EXPECTED_NEW_PROMOTION_TRAINING_UNITS
        and not (smoke_test and device == "cpu")
    ):
        raise discovery.CampaignError(
            "promotion training is authorized for exactly twelve nonreused units"
        )
    discovery.bind_run_usage_ledger(
        run_root, usage_ledger, execution_scope="promotion"
    )
    usage_ledger_identity_path = run_root / "GPU_USAGE_LEDGER_IDENTITY.json"
    unit_root = run_root / "training" / f"outer_{item.outer_fold}_seed_{item.seed}"
    output_dir = unit_root / "attempt_000" / "output"
    invocation_path = unit_root / "attempt_000" / "invocation.json"
    executions_root = unit_root / "attempt_000" / "executions"
    completion_path = unit_root / "completion_receipt.json"
    usage_identity = {
        "campaign_revision": CAMPAIGN_REVISION,
        "infrastructure_revision": INFRASTRUCTURE_REVISION,
        "outer_fold": item.outer_fold,
        "seed": item.seed,
        "variant": variant,
    }
    admitted_authorization_path, admitted_authorization_sha256 = (
        _admitted_authorization_from_governance(governance)
    )
    selection_lock_sha256 = _governance_sha256(governance, "selection_lock")
    promotion_authorization_sha256 = _governance_sha256(
        governance,
        "promotion_authorization",
        fallback_path=(
            promotion_authorization if smoke_test and device == "cpu" else None
        ),
    )
    if promotion_authorization_sha256 != discovery.sha256_file(
        promotion_authorization
    ):
        raise discovery.CampaignError(
            "current promotion authorization differs from governance"
        )

    def expected_usage_command_sha256(record: Mapping[str, Any]) -> str:
        if record.get("schema_version") != 2:
            raise discovery.CampaignError(
                "V7 promotion training cannot reuse a legacy V1 execution"
            )
        context = discovery._execution_record_context(record)
        resume_value = context.get("resume")
        if type(resume_value) is not bool:
            raise discovery.CampaignError(
                "promotion training GPU usage record has invalid resume state"
            )
        execution_number = context.get("execution_number")
        if type(execution_number) is not int or int(execution_number) < 0:
            raise discovery.CampaignError(
                "promotion training GPU usage record has invalid execution number"
            )
        expected_command = _promotion_train_command(
            python=python,
            trainer=trainer,
            item=item,
            output_dir=output_dir,
            target_sealed_capability_receipt=target_sealed_capability_receipt,
            expected_admitted_context=dict(context),
            variant=variant,
            promotion_authorization=promotion_authorization,
            release_mode=str(selection["selected_release_mode"]),
            device=device,
            amp=amp,
            smoke_test=smoke_test,
            resume=resume_value,
        )
        return discovery.semantic_sha256(expected_command)

    if completion_path.exists():
        receipt = discovery.load_json(completion_path, "promotion training receipt")
        if discovery.canonical_content_sha256(receipt) != receipt.get("content_sha256"):
            raise discovery.CampaignError("promotion training receipt content drifted")
        _require_exact_keys(
            receipt,
            {
                "schema_version",
                "classification",
                "campaign_id",
                "campaign_revision",
                "infrastructure_revision",
                "outer_fold",
                "seed",
                "variant",
                "outer_test_opened",
                "selection_lock_sha256",
                "promotion_authorization_sha256",
                "invocation",
                "usage_ledger_path",
                "usage_record_sha256",
                "usage_record_sha256s",
                "terminal_results",
                "lifecycle_invocations",
                "gpu_execution_ledger_path",
                "gpu_admission_lock_path",
                "validated_output",
                "validation_scores_changed_execution",
                "commercial_claim_authorized",
                "content_sha256",
            },
            label="promotion training receipt",
        )
        if not (
            receipt.get("schema_version") == 1
            and receipt.get("classification")
            == "adaptive_v3r1_fixed_promotion_training_completion"
            and receipt.get("campaign_id") == discovery.CAMPAIGN_ID
            and receipt.get("campaign_revision") == CAMPAIGN_REVISION
            and receipt.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
            and receipt.get("outer_fold") == item.outer_fold
            and receipt.get("seed") == item.seed
            and receipt.get("variant") == variant
            and receipt.get("outer_test_opened") is False
            and receipt.get("selection_lock_sha256")
            == selection_lock_sha256
            and receipt.get("promotion_authorization_sha256")
            == promotion_authorization_sha256
            and receipt.get("validation_scores_changed_execution") is False
            and receipt.get("commercial_claim_authorized") is False
        ):
            raise discovery.CampaignError(
                "promotion training receipt identity/governance drifted"
            )
        if receipt.get("invocation") != discovery.bind_file(invocation_path):
            raise discovery.CampaignError(
                "promotion training receipt invocation binding drifted"
            )
        invocation = discovery.load_json(
            invocation_path, "promotion training invocation"
        )
        if discovery.canonical_content_sha256(invocation) != invocation.get(
            "content_sha256"
        ):
            raise discovery.CampaignError(
                "promotion training invocation content drifted"
            )
        _require_exact_keys(
            invocation,
            {
                "schema_version",
                "classification",
                "campaign_id",
                "campaign_revision",
                "infrastructure_revision",
                "outer_fold",
                "validation_fold",
                "seed",
                "variant",
                "release_mode",
                "governance",
                "cache_manifest",
                "proposer_stack",
                "trainer",
                "gpu_wrapper",
                "usage_ledger_identity",
                "base_trainer_command",
                "validation_metrics_may_change_execution",
                "outer_test_opened",
                "content_sha256",
            },
            label="promotion training invocation",
        )
        expected_invocation = {
            "schema_version": 1,
            "classification": "adaptive_v3r1_fixed_promotion_training_invocation",
            "campaign_id": discovery.CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "outer_fold": item.outer_fold,
            "validation_fold": (item.outer_fold + 1) % 6,
            "seed": item.seed,
            "variant": variant,
            "release_mode": selection["selected_release_mode"],
            "governance": dict(governance),
            "cache_manifest": discovery.bind_file(item.cache_dir / "manifest.json"),
            "proposer_stack": discovery.bind_file(item.proposer_stack),
            "trainer": discovery.bind_file(trainer),
            "gpu_wrapper": discovery.bind_file(wrapper),
            "usage_ledger_identity": discovery.bind_file(
                usage_ledger_identity_path
            ),
            "base_trainer_command": _promotion_train_command(
                python=python,
                trainer=trainer,
                item=item,
                output_dir=output_dir,
                target_sealed_capability_receipt=target_sealed_capability_receipt,
                expected_admitted_context=discovery._execution_context(
                    usage_identity, execution_number=0, resume=False
                ),
                variant=variant,
                promotion_authorization=promotion_authorization,
                release_mode=str(selection["selected_release_mode"]),
                device=device,
                amp=amp,
                smoke_test=smoke_test,
                resume=False,
            ),
            "validation_metrics_may_change_execution": False,
            "outer_test_opened": False,
        }
        expected_invocation["content_sha256"] = discovery.semantic_sha256(
            expected_invocation
        )
        if invocation != expected_invocation:
            raise discovery.CampaignError(
                "promotion training invocation differs from current governance/input ABI"
            )
        discovery.validate_completion_receipt_usage(
            usage_ledger,
            receipt,
            expected_phase="promotion_training",
            expected_identity=usage_identity,
            expected_command_sha256=expected_usage_command_sha256,
            expected_gpu_ledger=gpu_ledger,
            expected_gpu_lock=gpu_lock,
        )
        validated = discovery.validate_training_output(
            output_dir,
            outer_fold=item.outer_fold,
            seed=item.seed,
            variant=variant,
            cache_dir=item.cache_dir,
        )
        if receipt.get("validated_output") != validated:
            raise discovery.CampaignError("completed promotion training output drifted")
        if int(validated["parameter_count"]) != int(
            selection["selected_parameter_count"]
        ):
            raise discovery.CampaignError(
                "completed promotion parameter count differs from selection lock"
            )
        return receipt
    base_command = _promotion_train_command(
        python=python,
        trainer=trainer,
        item=item,
        output_dir=output_dir,
        target_sealed_capability_receipt=target_sealed_capability_receipt,
        expected_admitted_context=discovery._execution_context(
            usage_identity, execution_number=0, resume=False
        ),
        variant=variant,
        promotion_authorization=promotion_authorization,
        release_mode=str(selection["selected_release_mode"]),
        device=device,
        amp=amp,
        smoke_test=smoke_test,
        resume=False,
    )
    discovery.create_once_json(
        invocation_path,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_fixed_promotion_training_invocation",
            "campaign_id": discovery.CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "outer_fold": item.outer_fold,
            "validation_fold": (item.outer_fold + 1) % 6,
            "seed": item.seed,
            "variant": variant,
            "release_mode": selection["selected_release_mode"],
            "governance": dict(governance),
            "cache_manifest": discovery.bind_file(item.cache_dir / "manifest.json"),
            "proposer_stack": discovery.bind_file(item.proposer_stack),
            "trainer": discovery.bind_file(trainer),
            "gpu_wrapper": discovery.bind_file(wrapper),
            "usage_ledger_identity": discovery.bind_file(usage_ledger_identity_path),
            "base_trainer_command": base_command,
            "validation_metrics_may_change_execution": False,
            "outer_test_opened": False,
        },
    )
    def workload(execution_number: int, resume_value: bool) -> list[str]:
        expected_context = discovery._execution_context(
            usage_identity,
            execution_number=execution_number,
            resume=resume_value,
        )
        return _promotion_train_command(
            python=python,
            trainer=trainer,
            item=item,
            output_dir=output_dir,
            target_sealed_capability_receipt=target_sealed_capability_receipt,
            expected_admitted_context=expected_context,
            variant=variant,
            promotion_authorization=promotion_authorization,
            release_mode=str(selection["selected_release_mode"]),
            device=device,
            amp=amp,
            smoke_test=smoke_test,
            resume=resume_value,
        )

    terminal_results: list[dict[str, Any]] = []
    lifecycle_invocations: list[dict[str, Any]] = []
    successful_result: dict[str, Any] | None = None
    for number, execution_root in enumerate(
        discovery._execution_directories(executions_root)
    ):
        execution_invocation_path = execution_root / "invocation.json"
        if not execution_invocation_path.is_file():
            raise discovery.CampaignError(
                "promotion training execution lacks its invocation"
            )
        execution_invocation = discovery.load_json(
            execution_invocation_path, "promotion training GPU execution invocation"
        )
        context = execution_invocation.get("context")
        if not isinstance(context, Mapping):
            raise discovery.CampaignError(
                "promotion training execution invocation lacks context"
            )
        resume_value = context.get("resume")
        expected_context = discovery._execution_context(
            usage_identity,
            execution_number=number,
            resume=bool(resume_value) if type(resume_value) is bool else False,
        )
        if type(resume_value) is not bool or context != expected_context:
            raise discovery.CampaignError(
                "promotion training execution history context drifted"
            )
        result_path = execution_root / "terminal_result.json"
        if not result_path.exists():
            discovery._validate_execution_invocation(
                execution_invocation_path,
                phase="promotion_training",
                context=expected_context,
                unit_invocation_path=invocation_path,
                workload_command=workload(number, resume_value),
            )
            lifecycle_invocations.append(
                discovery.bind_file(execution_invocation_path)
            )
            recovery_command = discovery._admitted_command(
                python=python,
                wrapper=wrapper,
                gpu_lock=gpu_lock,
                gpu_ledger=gpu_ledger,
                usage_ledger=usage_ledger,
                result_file=result_path,
                phase="promotion_training",
                context=expected_context,
                invocation_sha256=discovery.sha256_file(
                    execution_invocation_path
                ),
                authorization_path=admitted_authorization_path,
                authorization_sha256=admitted_authorization_sha256,
                trainer_command=workload(number, resume_value),
            )
            command_runner(
                recovery_command, float(discovery.GPU_BUDGET_SECONDS)
            )
            if not result_path.is_file():
                state = discovery._verify_usage_state(usage_ledger)
                if int(state.remaining_ns) <= 0:
                    raise discovery.BudgetExhausted(
                        "ten-GPU-hour budget is exhausted"
                    )
                raise discovery.CampaignError(
                    "GPU supervisor could not recover the existing promotion-training result"
                )
        result, binding = discovery._load_execution_terminal_result(
            invocation_path=execution_invocation_path,
            result_path=result_path,
            usage_ledger=usage_ledger,
            phase="promotion_training",
            context=expected_context,
            unit_invocation_path=invocation_path,
            workload_command=workload(number, resume_value),
        )
        if not any(
            entry.get("sha256") == discovery.sha256_file(execution_invocation_path)
            for entry in lifecycle_invocations
        ):
            lifecycle_invocations.append(
                discovery.bind_file(execution_invocation_path)
            )
        terminal_results.append(binding)
        if result.get("reusable_success") is True:
            if successful_result is not None:
                raise discovery.CampaignError(
                    "promotion training has multiple successful executions"
                )
            successful_result = result

    if successful_result is None:
        execution_number = len(discovery._execution_directories(executions_root))
        resume = output_dir.exists() and any(output_dir.iterdir())
        context = discovery._execution_context(
            usage_identity, execution_number=execution_number, resume=resume
        )
        trainer_command = workload(execution_number, resume)
        execution_root = discovery._publish_execution_directory(
            executions_root,
            execution_number=execution_number,
            create_invocation=lambda staged: discovery._create_execution_invocation(
                staged,
                phase="promotion_training",
                context=context,
                unit_invocation_path=invocation_path,
                workload_command=trainer_command,
            ),
            validate_invocation=lambda staged: discovery._validate_execution_invocation(
                staged,
                phase="promotion_training",
                context=context,
                unit_invocation_path=invocation_path,
                workload_command=trainer_command,
            ),
        )
        execution_invocation_path = execution_root / "invocation.json"
        result_path = execution_root / "terminal_result.json"
        lifecycle_invocations.append(
            discovery.bind_file(execution_invocation_path)
        )
        command = discovery._admitted_command(
            python=python,
            wrapper=wrapper,
            gpu_lock=gpu_lock,
            gpu_ledger=gpu_ledger,
            usage_ledger=usage_ledger,
            result_file=result_path,
            phase="promotion_training",
            context=context,
            invocation_sha256=discovery.sha256_file(execution_invocation_path),
            authorization_path=admitted_authorization_path,
            authorization_sha256=admitted_authorization_sha256,
            trainer_command=trainer_command,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        command_runner(command, float(discovery.GPU_BUDGET_SECONDS))
        if not result_path.is_file():
            state = discovery._verify_usage_state(usage_ledger)
            if int(state.remaining_ns) <= 0:
                raise discovery.BudgetExhausted("ten-GPU-hour budget is exhausted")
            raise discovery.CampaignError(
                "GPU supervisor exited without a promotion-training terminal result"
            )
        successful_result, binding = discovery._load_execution_terminal_result(
            invocation_path=execution_invocation_path,
            result_path=result_path,
            usage_ledger=usage_ledger,
            phase="promotion_training",
            context=context,
            unit_invocation_path=invocation_path,
            workload_command=trainer_command,
        )
        terminal_results.append(binding)
        if successful_result.get("reusable_success") is not True:
            if successful_result.get("hard_timeout_reached") is True:
                raise discovery.BudgetExhausted(
                    "promotion trainer reached the hard GPU-hour ceiling"
                )
            raise discovery.CampaignError(
                f"promotion trainer failed for {item.outer_fold}/{item.seed}: "
                f"{successful_result.get('return_code')}"
            )
    validated = discovery.validate_training_output(
        output_dir,
        outer_fold=item.outer_fold,
        seed=item.seed,
        variant=variant,
        cache_dir=item.cache_dir,
    )
    if int(validated["parameter_count"]) != int(selection["selected_parameter_count"]):
        raise discovery.CampaignError("promotion parameter count differs from selection lock")
    usage_fields = discovery.completion_usage_fields(
        usage_ledger,
        final_record_sha256=str(successful_result["terminal_record_sha256"]),
        expected_phase="promotion_training",
        expected_identity=usage_identity,
        expected_command_sha256=expected_usage_command_sha256,
        terminal_results=terminal_results,
        lifecycle_invocations=lifecycle_invocations,
        gpu_ledger=gpu_ledger,
        gpu_lock=gpu_lock,
    )
    return discovery.create_once_json(
        completion_path,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_fixed_promotion_training_completion",
            "campaign_id": discovery.CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "outer_fold": item.outer_fold,
            "seed": item.seed,
            "variant": variant,
            "outer_test_opened": False,
            "selection_lock_sha256": selection_lock_sha256,
            "promotion_authorization_sha256": promotion_authorization_sha256,
            "invocation": discovery.bind_file(invocation_path),
            **usage_fields,
            "validated_output": validated,
            "validation_scores_changed_execution": False,
            "commercial_claim_authorized": False,
        },
    )


def _prediction_command(
    *,
    python: Path,
    trainer: Path,
    predict_input: Path,
    checkpoint: Path,
    scaler: Path,
    output_dir: Path,
    target_sealed_capability_receipt: Path,
    expected_admitted_context: Mapping[str, Any],
    outer_fold: int,
    seed: int,
    variant: str,
    release_mode: str,
    promotion_authorization: Path,
    device: str,
    amp: bool,
) -> list[str]:
    command = [
        str(python),
        str(trainer),
        "--mode",
        "predict",
        "--campaign-phase",
        "promotion",
        "--promotion-authorization",
        str(promotion_authorization),
        "--predict-input",
        str(predict_input),
        "--checkpoint",
        str(checkpoint),
        "--scaler",
        str(scaler),
        "--output-dir",
        str(output_dir),
        "--target-sealed-capability-receipt",
        str(target_sealed_capability_receipt),
        "--expected-admitted-context-json",
        json.dumps(
            dict(expected_admitted_context),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--outer-fold",
        str(outer_fold),
        "--seed",
        str(seed),
        "--variant",
        variant,
        "--release-mode",
        release_mode,
        "--device",
        device,
    ]
    command.append("--amp" if amp else "--no-amp")
    return command


def validate_target_free_prediction(
    output_dir: Path,
    *,
    expected_index: np.ndarray,
    outer_fold: int,
    seed: int,
    variant: str,
    release_mode: str,
    predict_input: Path,
    checkpoint: Path,
    scaler: Path,
) -> dict[str, Any]:
    prediction_path = output_dir / "predictions.npz"
    manifest_path = output_dir / "prediction_manifest.json"
    if not prediction_path.is_file() or not manifest_path.is_file():
        raise discovery.CampaignError("target-free prediction output is incomplete")
    try:
        with np.load(prediction_path, allow_pickle=False) as archive:
            fields = tuple(archive.files)
            if set(fields) != set(PREDICTION_FIELDS) or len(fields) != len(PREDICTION_FIELDS):
                raise discovery.CampaignError(
                    f"prediction field allow-list mismatch: {sorted(fields)}"
                )
            if any(
                token in name.lower()
                for name in fields
                for token in FORBIDDEN_PREDICTION_TOKENS
            ):
                raise discovery.CampaignError("prediction contains a forbidden target/context field")
            arrays = {name: np.asarray(archive[name]) for name in PREDICTION_FIELDS}
    except (OSError, ValueError, KeyError) as error:
        raise discovery.CampaignError(f"invalid target-free prediction: {error}") from error
    index = arrays["cache_index"].astype(np.int64)
    if not np.array_equal(index, expected_index):
        raise discovery.CampaignError("prediction is not the exact sanitized input cover")
    rows = len(index)
    for name in PREDICTION_FIELDS:
        expected = (rows, 4) if name == "factor_probabilities" else (rows,)
        if arrays[name].shape != expected:
            raise discovery.CampaignError(f"prediction topology is invalid: {name}")
    availability = arrays["prediction_available"].astype(bool)
    prediction = arrays["prediction_bpm"].astype(float)
    if np.any(availability & ~np.isfinite(prediction)):
        raise discovery.CampaignError("available prediction is non-finite")
    factor = arrays["factor_probabilities"].astype(float)
    if not np.isfinite(factor).all() or np.any(factor < 0) or not np.allclose(
        factor.sum(axis=1), 1.0, rtol=0, atol=2e-5
    ):
        raise discovery.CampaignError("prediction factor probabilities are invalid")
    raw = arrays["raw_anchor_bpm"].astype(float)
    raw_available = arrays["raw_anchor_available"].astype(bool)
    hard = arrays["hard_source_bpm"].astype(float)
    hard_available = arrays["hard_source_available"].astype(bool)
    probability = arrays["selected_source_probability"].astype(float)
    if release_mode == "raw_anchor":
        expected = np.where(raw_available, raw, hard)
        expected_available = raw_available | hard_available
    elif release_mode == "hard_source_argmax":
        expected = hard
        expected_available = hard_available
    elif release_mode == "fixed_confidence_switch":
        use_hard = hard_available & ((probability >= 0.8) | ~raw_available)
        expected = np.where(use_hard, hard, raw)
        expected_available = hard_available | raw_available
    else:
        raise discovery.CampaignError("unregistered release mode in prediction")
    if not np.array_equal(availability, expected_available) or not np.allclose(
        prediction[availability], expected[availability], rtol=0, atol=1e-5
    ):
        raise discovery.CampaignError("prediction violates the locked hard release decoder")
    manifest = discovery.load_json(manifest_path, "target-free prediction manifest")
    if manifest.get("content_sha256") is not None and discovery.canonical_content_sha256(
        manifest
    ) != manifest.get("content_sha256"):
        raise discovery.CampaignError("prediction manifest content hash drifted")
    if not (
        int(manifest.get("outer_fold", -1)) == outer_fold
        and int(manifest.get("seed", -1)) == seed
        and manifest.get("variant") == variant
        and manifest.get("release_mode") == release_mode
        and manifest.get("target_fields_emitted") is False
        and manifest.get("commercial_claim_authorized") in (False, None)
    ):
        raise discovery.CampaignError("prediction manifest identity/leakage boundary drifted")
    expected_bindings = {
        "predict_input": (predict_input, ("predict_input", "input")),
        "checkpoint": (checkpoint, ("checkpoint", "best_checkpoint")),
        "scaler": (scaler, ("scaler",)),
    }
    for label, (path, aliases) in expected_bindings.items():
        observed: str | None = None
        for alias in aliases:
            value = manifest.get(alias)
            if isinstance(value, Mapping):
                observed = value.get("sha256", value.get("file_sha256"))  # type: ignore[assignment]
            elif isinstance(value, str) and alias.endswith("sha256"):
                observed = value
            if observed is not None:
                break
            direct = manifest.get(f"{alias}_sha256")
            if isinstance(direct, str):
                observed = direct
                break
        if observed != discovery.sha256_file(path):
            raise discovery.CampaignError(f"prediction manifest does not bind {label}")
    return {
        "outer_fold": outer_fold,
        "seed": seed,
        "variant": variant,
        "release_mode": release_mode,
        "rows": rows,
        "cache_index_sha256": discovery.semantic_sha256(index.tolist()),
        "prediction": discovery.bind_file(prediction_path),
        "manifest": discovery.bind_file(manifest_path),
        "target_fields_present": False,
        "identity_fields_present": False,
        "protocol_fields_present": False,
    }


def _validate_prediction_input_receipt(
    *,
    project_root: Path,
    input_receipt: Path,
    predict_input: Path,
    expected_index: np.ndarray,
    outer_fold: int,
    seed: int,
    selection: Mapping[str, Any],
    governance: Mapping[str, Any],
    model_source_binding: Mapping[str, Any],
    require_v8_schema: bool,
) -> dict[str, Any]:
    binding = discovery.bind_file(input_receipt)
    receipt = discovery.load_json(input_receipt, "sanitized prediction input receipt")
    if not require_v8_schema:
        if receipt.get("content_sha256") is not None and (
            discovery.canonical_content_sha256(receipt)
            != receipt.get("content_sha256")
        ):
            raise discovery.CampaignError(
                "legacy sanitized prediction input receipt content drifted"
            )
        return binding
    if discovery.canonical_content_sha256(receipt) != receipt.get("content_sha256"):
        raise discovery.CampaignError(
            "sanitized prediction input receipt content drifted"
        )
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "classification",
            "campaign_id",
            "outer_fold",
            "seed",
            "selected_variant",
            "selected_release_mode",
            "governance",
            "source_bindings",
            "proposer_source_field_names",
            "metadata_read_usecols",
            "output",
            "validation",
            "exact_cache_index_cover",
            "target_reference_qc_identity_protocol_columns_read",
            "target_fields_present",
            "future_context_present",
            "promotion_authorized",
            "promotion_model_source",
            "commercial_claim_authorized",
            "content_sha256",
        },
        label="sanitized prediction input receipt",
    )
    validation = locked_inputs.validate_sanitized_input(
        predict_input, expected_index=expected_index
    )
    if not (
        receipt.get("schema_version") == 1
        and receipt.get("classification")
        == "adaptive_v3r1_sanitized_target_free_promotion_input"
        and receipt.get("campaign_id") == discovery.CAMPAIGN_ID
        and receipt.get("outer_fold") == outer_fold
        and receipt.get("seed") == seed
        and receipt.get("selected_variant") == selection["selected_variant"]
        and receipt.get("selected_release_mode")
        == selection["selected_release_mode"]
        and receipt.get("governance") == dict(governance)
        and receipt.get("output") == discovery.bind_file(predict_input)
        and receipt.get("validation") == validation
        and receipt.get("exact_cache_index_cover") is True
        and receipt.get("target_reference_qc_identity_protocol_columns_read")
        is False
        and receipt.get("target_fields_present") is False
        and receipt.get("future_context_present") is False
        and receipt.get("promotion_authorized") is True
        and receipt.get("promotion_model_source") == model_source_binding
        and receipt.get("commercial_claim_authorized") is False
    ):
        raise discovery.CampaignError(
            "sanitized prediction input receipt identity/provenance drifted"
        )
    source_bindings = receipt.get("source_bindings")
    if not isinstance(source_bindings, Mapping) or not source_bindings:
        raise discovery.CampaignError(
            "sanitized prediction input source bindings are incomplete"
        )
    for name, source_binding in source_bindings.items():
        discovery.verify_binding(
            source_binding,
            project_root=project_root,
            owner=input_receipt,
            label=f"sanitized prediction input source {name}",
        )
    return binding


_PREDICTION_ATTEMPT_INVOCATION_NAMES = (
    "execution_invocation.json",
    "invocation.json",
)


def _load_exact_immutable_json(path: Path, *, label: str) -> dict[str, Any]:
    """Read one create-once JSON artifact and require its exact immutable inode."""

    value = discovery.load_json(path, label)
    document, raw = discovery._content_document(value)
    if value != document:
        raise discovery.CampaignError(f"{label} content hash drifted: {path}")
    discovery._read_exact_immutable(path, expected=raw, label=label)
    return value


def _prediction_attempt_directories(root: Path) -> list[Path]:
    """Return the committed attempt prefix and admit only its exact staged tail."""

    if root.is_symlink():
        raise discovery.CampaignError(
            "promotion prediction attempts root is not a canonical directory"
        )
    if not root.exists():
        return []
    if not root.is_dir():
        raise discovery.CampaignError(
            "promotion prediction attempts root is not a canonical directory"
        )
    entries = sorted(root.iterdir(), key=lambda item: item.name)
    attempts = [path for path in entries if path.name.startswith("attempt_")]
    expected = [f"attempt_{number:03d}" for number in range(len(attempts))]
    if (
        [path.name for path in attempts] != expected
        or any(path.is_symlink() or not path.is_dir() for path in attempts)
    ):
        raise discovery.CampaignError(
            "promotion prediction attempts are not a contiguous committed history"
        )
    staging_name = f".attempt_{len(attempts):03d}.staging"
    foreign = [path for path in entries if path not in attempts]
    if len(foreign) > 1 or (foreign and foreign[0].name != staging_name):
        raise discovery.CampaignError(
            "promotion prediction attempts have a foreign or non-tail entry"
        )
    if foreign:
        staging = foreign[0]
        if staging.is_symlink() or not staging.is_dir():
            raise discovery.CampaignError(
                "promotion prediction attempt staging tail is aliased"
            )
        staged_names = tuple(
            path.name for path in sorted(staging.iterdir(), key=lambda item: item.name)
        )
        if staged_names not in (
            (),
            ("invocation.json",),
            _PREDICTION_ATTEMPT_INVOCATION_NAMES,
        ):
            raise discovery.CampaignError(
                "promotion prediction attempt staging tail is not an exact prefix"
            )
        for name in staged_names:
            _load_exact_immutable_json(
                staging / name,
                label="staged promotion prediction invocation",
            )
    for attempt in attempts:
        for name in _PREDICTION_ATTEMPT_INVOCATION_NAMES:
            _load_exact_immutable_json(
                attempt / name,
                label="committed promotion prediction invocation",
            )
    # Finish the durability half of a rename observed after a killed parent.
    discovery._fsync_directory(root)
    return attempts


def _staged_prediction_execution_invocation(
    *,
    staged_unit_invocation_path: Path,
    published_unit_invocation_path: Path,
    context: Mapping[str, Any],
    workload_command: Sequence[str],
) -> dict[str, Any]:
    """Bind staged bytes to their deterministic post-rename pathname."""

    unit_binding = discovery.bind_file(staged_unit_invocation_path)
    unit_binding["path"] = str(published_unit_invocation_path.expanduser().resolve())
    command = list(workload_command)
    return {
        "schema_version": 2,
        "classification": "adaptive_v3r1_v8r4_gpu_execution_invocation",
        "campaign_id": discovery.CAMPAIGN_ID,
        "campaign_revision": CAMPAIGN_REVISION,
        "infrastructure_revision": INFRASTRUCTURE_REVISION,
        "phase": "promotion_prediction",
        "context": dict(context),
        "unit_invocation": unit_binding,
        "workload_command": command,
        "workload_command_sha256": discovery.semantic_sha256(command),
        "parent_side_elapsed_accounting": False,
    }


def _publish_prediction_attempt_directory(
    root: Path,
    *,
    attempt_number: int,
    attempt_invocation: Mapping[str, Any],
    context: Mapping[str, Any],
    workload_command: Sequence[str],
) -> Path:
    """Publish both immutable attempt invocations with one NOREPLACE rename."""

    existing = _prediction_attempt_directories(root)
    if attempt_number != len(existing):
        raise discovery.CampaignError(
            "promotion prediction staging index is not the exact tail"
        )
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".attempt_{attempt_number:03d}.staging"
    final = root / f"attempt_{attempt_number:03d}"
    if final.exists() or final.is_symlink():
        raise discovery.CampaignError(
            "promotion prediction final tail appeared unexpectedly"
        )
    if not staging.exists():
        staging.mkdir(mode=0o700)
        discovery._fsync_directory(root)
        discovery._publication_fault("prediction_attempt_staged", final)
    elif staging.is_symlink() or not staging.is_dir():
        raise discovery.CampaignError(
            "promotion prediction staging tail is not a directory"
        )

    invocation_path = staging / "invocation.json"
    staged_names = tuple(
        path.name for path in sorted(staging.iterdir(), key=lambda item: item.name)
    )
    if staged_names == ():
        discovery.create_once_json(invocation_path, attempt_invocation)
    elif staged_names not in (
        ("invocation.json",),
        _PREDICTION_ATTEMPT_INVOCATION_NAMES,
    ):
        raise discovery.CampaignError(
            "promotion prediction staging tail is not an exact invocation prefix"
        )
    discovery.create_once_json(invocation_path, attempt_invocation)
    discovery._fsync_directory(staging)
    discovery._publication_fault("prediction_attempt_invocation_durable", final)

    execution_invocation_path = staging / "execution_invocation.json"
    published_invocation_path = final / "invocation.json"
    execution_invocation = _staged_prediction_execution_invocation(
        staged_unit_invocation_path=invocation_path,
        published_unit_invocation_path=published_invocation_path,
        context=context,
        workload_command=workload_command,
    )
    discovery.create_once_json(execution_invocation_path, execution_invocation)
    discovery._fsync_directory(staging)
    discovery._publication_fault("prediction_attempt_invocations_durable", final)
    discovery._rename_staged_directory_once(staging, final)
    discovery._publication_fault("prediction_attempt_linked", final)
    discovery._fsync_directory(root)
    discovery._publication_fault("prediction_attempt_published", final)
    discovery._validate_execution_invocation(
        final / "execution_invocation.json",
        phase="promotion_prediction",
        context=context,
        unit_invocation_path=published_invocation_path,
        workload_command=workload_command,
    )
    return final


def _run_prediction(
    *,
    run_root: Path,
    item: discovery.TrainingInput,
    model_source: locked_inputs.PromotionModelSource | None = None,
    training_receipt: Mapping[str, Any] | None = None,
    predict_input: Path,
    input_receipt: Path,
    selection: Mapping[str, Any],
    governance: Mapping[str, Any],
    target_sealed_capability_receipt: Path,
    python: Path,
    trainer: Path,
    wrapper: Path,
    promotion_authorization: Path,
    gpu_lock: Path,
    gpu_ledger: Path,
    usage_ledger: Path,
    device: str,
    amp: bool,
    command_runner: Callable[[Sequence[str], float], tuple[int, float, bool]],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    discovery.bind_run_usage_ledger(
        run_root, usage_ledger, execution_scope="promotion"
    )
    usage_ledger_identity_path = run_root / "GPU_USAGE_LEDGER_IDENTITY.json"
    unit_root = run_root / "predictions" / f"outer_{item.outer_fold}_seed_{item.seed}"
    completion_path = unit_root / "completion_receipt.json"
    if model_source is not None:
        if model_source.kind == "mounted_successor_pack":
            mounted = locked_inputs.validate_model_bound_prediction_pack(
                model_source.output_dir,
                outer_fold=item.outer_fold,
                seed=item.seed,
                selected_variant=str(selection["selected_variant"]),
                expected_promotion_authorization=governance.get(
                    "promotion_authorization"
                ),
                expected_selection_lock=governance.get("selection_lock"),
            )
            current_model_source = mounted.model_source
            if mounted.input_path != predict_input.resolve():
                raise discovery.CampaignError(
                    "promotion prediction input is outside its successor pack"
                )
        else:
            current_model_source = locked_inputs.resolve_promotion_model_source(
                project_root=project_root,
                run_root=run_root,
                cache_dir=item.cache_dir,
                outer_fold=item.outer_fold,
                seed=item.seed,
                variant=str(selection["selected_variant"]),
            )
        if current_model_source != model_source:
            raise discovery.CampaignError(
                "promotion prediction model source differs from live resolution"
            )
        model_source = current_model_source
        receipt_variant = model_source.receipt.get(
            "selected_variant",
            model_source.receipt.get("variant"),
        )
        if not (
            model_source.receipt.get("outer_fold") == item.outer_fold
            and model_source.receipt.get("seed") == item.seed
            and receipt_variant == selection["selected_variant"]
        ):
            raise discovery.CampaignError("promotion prediction model source identity drifted")
        checkpoint = model_source.checkpoint
        scaler = model_source.scaler
        model_source_kind = model_source.kind
        model_source_receipt_binding: Mapping[str, Any] = discovery.bind_file(
            model_source.receipt_path
        )
        scientific_signature_sha256 = model_source.scientific_signature_sha256
    else:
        # Compatibility for the pre-V8 lifecycle recovery regression only.  The
        # V8 campaign main path always supplies an independently resolved source.
        if (
            device != "cpu"
            or not isinstance(training_receipt, Mapping)
            or not discovery._is_sha256(training_receipt.get("content_sha256"))
        ):
            raise discovery.CampaignError("promotion prediction lacks a V8 model source")
        training_output = (
            run_root
            / "training"
            / f"outer_{item.outer_fold}_seed_{item.seed}"
            / "attempt_000/output"
        )
        checkpoint = training_output / "best.pt"
        scaler = training_output / "scaler.json"
        model_source_kind = "legacy_local_training_lifecycle_test"
        model_source_receipt_binding = {
            "content_sha256": str(training_receipt["content_sha256"])
        }
        scientific_signature_sha256 = str(training_receipt["content_sha256"])
    model_source_binding = {
        "kind": model_source_kind,
        "receipt": dict(model_source_receipt_binding),
        "checkpoint": discovery.bind_file(checkpoint),
        "scaler": discovery.bind_file(scaler),
        "scientific_signature_sha256": scientific_signature_sha256,
    }
    usage_identity = {
        "campaign_revision": CAMPAIGN_REVISION,
        "infrastructure_revision": INFRASTRUCTURE_REVISION,
        "outer_fold": item.outer_fold,
        "seed": item.seed,
        "variant": selection["selected_variant"],
        "release_mode": selection["selected_release_mode"],
    }
    admitted_authorization_path, admitted_authorization_sha256 = (
        _admitted_authorization_from_governance(governance)
    )
    selection_lock_sha256 = _governance_sha256(governance, "selection_lock")
    promotion_authorization_sha256 = _governance_sha256(
        governance,
        "promotion_authorization",
        fallback_path=(
            promotion_authorization
            if model_source is None and device == "cpu"
            else None
        ),
    )
    if promotion_authorization_sha256 != discovery.sha256_file(
        promotion_authorization
    ):
        raise discovery.CampaignError(
            "current promotion authorization differs from governance"
        )

    def expected_usage_command_sha256(record: Mapping[str, Any]) -> str:
        if record.get("schema_version") != 2:
            raise discovery.CampaignError(
                "V7 promotion prediction cannot reuse a legacy V1 execution"
            )
        context = discovery._execution_record_context(record)
        attempt_number = context.get("attempt_number")
        if type(attempt_number) is not int or int(attempt_number) < 0:
            raise discovery.CampaignError(
                "promotion prediction GPU usage record has invalid attempt number"
            )
        invocation_path = (
            unit_root / "attempts" / f"attempt_{int(attempt_number):03d}" / "invocation.json"
        )
        if not invocation_path.is_file():
            raise discovery.CampaignError(
                "promotion prediction GPU usage record lacks its invocation"
            )
        invocation = discovery.load_json(
            invocation_path, "promotion prediction attempt invocation"
        )
        if discovery.canonical_content_sha256(invocation) != invocation.get("content_sha256"):
            raise discovery.CampaignError("promotion prediction invocation content drifted")
        trainer_command = invocation.get("trainer_command")
        if not isinstance(trainer_command, list) or not all(
            isinstance(value, str) for value in trainer_command
        ):
            raise discovery.CampaignError("promotion prediction invocation command is invalid")
        if any(invocation.get(name) != value for name, value in usage_identity.items()):
            raise discovery.CampaignError("promotion prediction invocation identity mismatched")
        return discovery.semantic_sha256(trainer_command)

    with np.load(predict_input, allow_pickle=False) as archive:
        expected_index = np.asarray(archive["cache_index"], dtype=np.int64)
    if model_source is not None and model_source.kind == "mounted_successor_pack":
        if input_receipt.resolve() != model_source.receipt_path.resolve():
            raise discovery.CampaignError(
                "successor prediction input receipt is not its model capability"
            )
        input_receipt_binding = discovery.bind_file(input_receipt)
        if input_receipt_binding != model_source_receipt_binding:
            raise discovery.CampaignError(
                "successor prediction capability binding drifted"
            )
        locked_inputs.validate_sanitized_input(
            predict_input,
            expected_index=expected_index,
        )
    else:
        input_receipt_binding = _validate_prediction_input_receipt(
            project_root=project_root,
            input_receipt=input_receipt,
            predict_input=predict_input,
            expected_index=expected_index,
            outer_fold=item.outer_fold,
            seed=item.seed,
            selection=selection,
            governance=governance,
            model_source_binding=model_source_binding,
            require_v8_schema=model_source is not None,
        )
    if completion_path.exists():
        receipt = discovery.load_json(completion_path, "promotion prediction receipt")
        if discovery.canonical_content_sha256(receipt) != receipt.get("content_sha256"):
            raise discovery.CampaignError("promotion prediction receipt content drifted")
        _require_exact_keys(
            receipt,
            {
                "schema_version",
                "classification",
                "campaign_id",
                "campaign_revision",
                "infrastructure_revision",
                "outer_fold",
                "seed",
                "variant",
                "release_mode",
                "output_dir",
                "attempt_invocation",
                "promotion_model_source",
                "governance",
                "selection_lock_sha256",
                "promotion_authorization_sha256",
                "input_receipt",
                "usage_ledger_path",
                "usage_record_sha256",
                "usage_record_sha256s",
                "terminal_results",
                "lifecycle_invocations",
                "gpu_execution_ledger_path",
                "gpu_admission_lock_path",
                "validated_output",
                "target_fields_accessed_or_emitted",
                "commercial_claim_authorized",
                "content_sha256",
            },
            label="promotion prediction receipt",
        )
        if not (
            receipt.get("schema_version") == 1
            and receipt.get("classification")
            == "adaptive_v3r1_target_free_promotion_prediction_completion"
            and receipt.get("campaign_id") == discovery.CAMPAIGN_ID
            and receipt.get("campaign_revision") == CAMPAIGN_REVISION
            and receipt.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
            and receipt.get("outer_fold") == item.outer_fold
            and receipt.get("seed") == item.seed
            and receipt.get("variant") == selection["selected_variant"]
            and receipt.get("release_mode") == selection["selected_release_mode"]
            and receipt.get("promotion_model_source") == model_source_binding
            and receipt.get("governance") == dict(governance)
            and receipt.get("selection_lock_sha256") == selection_lock_sha256
            and receipt.get("promotion_authorization_sha256")
            == promotion_authorization_sha256
            and receipt.get("input_receipt") == input_receipt_binding
            and receipt.get("target_fields_accessed_or_emitted") is False
            and receipt.get("commercial_claim_authorized") is False
        ):
            raise discovery.CampaignError(
                "promotion prediction receipt identity/governance/input drifted"
            )
        discovery.validate_completion_receipt_usage(
            usage_ledger,
            receipt,
            expected_phase="promotion_prediction",
            expected_identity=usage_identity,
            expected_command_sha256=expected_usage_command_sha256,
            expected_gpu_ledger=gpu_ledger,
            expected_gpu_lock=gpu_lock,
        )
        output_dir = Path(str(receipt["output_dir"]))
        if not output_dir.is_absolute():
            output_dir = (unit_root / output_dir).resolve()
        output_dir = output_dir.resolve()
        attempt_invocation_path = output_dir.parent / "invocation.json"
        if (
            output_dir.parent.parent != unit_root / "attempts"
            or output_dir.name != "output"
            or receipt.get("attempt_invocation")
            != discovery.bind_file(attempt_invocation_path)
        ):
            raise discovery.CampaignError(
                "promotion prediction receipt attempt binding drifted"
            )
        attempt_invocation = discovery.load_json(
            attempt_invocation_path, "promotion prediction attempt invocation"
        )
        if discovery.canonical_content_sha256(
            attempt_invocation
        ) != attempt_invocation.get("content_sha256"):
            raise discovery.CampaignError(
                "promotion prediction attempt invocation content drifted"
            )
        _require_exact_keys(
            attempt_invocation,
            {
                "schema_version",
                "classification",
                "campaign_id",
                "campaign_revision",
                "infrastructure_revision",
                "outer_fold",
                "seed",
                "variant",
                "release_mode",
                "attempt_number",
                "governance",
                "model_source_kind",
                "model_source_receipt",
                "scientific_signature_sha256",
                "predict_input",
                "input_receipt",
                "checkpoint",
                "scaler",
                "trainer_command",
                "usage_ledger_identity",
                "target_access_authorized",
                "content_sha256",
            },
            label="promotion prediction attempt invocation",
        )
        attempt_number = attempt_invocation.get("attempt_number")
        if type(attempt_number) is not int or attempt_number < 0 or (
            output_dir.parent.name != f"attempt_{attempt_number:03d}"
        ):
            raise discovery.CampaignError(
                "promotion prediction attempt topology drifted"
            )
        expected_command = _prediction_command(
            python=python,
            trainer=trainer,
            predict_input=predict_input,
            checkpoint=checkpoint,
            scaler=scaler,
            output_dir=output_dir,
            target_sealed_capability_receipt=target_sealed_capability_receipt,
            expected_admitted_context={
                **usage_identity, "attempt_number": int(attempt_number)
            },
            outer_fold=item.outer_fold,
            seed=item.seed,
            variant=str(selection["selected_variant"]),
            release_mode=str(selection["selected_release_mode"]),
            promotion_authorization=promotion_authorization,
            device=device,
            amp=amp,
        )
        expected_attempt_invocation = {
            "schema_version": 1,
            "classification": "adaptive_v3r1_target_free_promotion_prediction_invocation",
            "campaign_id": discovery.CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "outer_fold": item.outer_fold,
            "seed": item.seed,
            "variant": selection["selected_variant"],
            "release_mode": selection["selected_release_mode"],
            "attempt_number": attempt_number,
            "governance": dict(governance),
            "model_source_kind": model_source_kind,
            "model_source_receipt": dict(model_source_receipt_binding),
            "scientific_signature_sha256": scientific_signature_sha256,
            "predict_input": discovery.bind_file(predict_input),
            "input_receipt": input_receipt_binding,
            "checkpoint": discovery.bind_file(checkpoint),
            "scaler": discovery.bind_file(scaler),
            "trainer_command": expected_command,
            "usage_ledger_identity": discovery.bind_file(
                usage_ledger_identity_path
            ),
            "target_access_authorized": False,
        }
        expected_attempt_invocation["content_sha256"] = discovery.semantic_sha256(
            expected_attempt_invocation
        )
        if attempt_invocation != expected_attempt_invocation:
            raise discovery.CampaignError(
                "promotion prediction invocation differs from current input/model/governance ABI"
            )
        validated = validate_target_free_prediction(
            output_dir,
            expected_index=expected_index,
            outer_fold=item.outer_fold,
            seed=item.seed,
            variant=str(selection["selected_variant"]),
            release_mode=str(selection["selected_release_mode"]),
            predict_input=predict_input,
            checkpoint=checkpoint,
            scaler=scaler,
        )
        if receipt.get("validated_output") != validated:
            raise discovery.CampaignError("completed target-free prediction drifted")
        return receipt
    attempts = unit_root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt_roots = _prediction_attempt_directories(attempts)

    terminal_results: list[dict[str, Any]] = []
    lifecycle_invocations: list[dict[str, Any]] = []
    successful_result: dict[str, Any] | None = None
    successful_attempt_root: Path | None = None
    for attempt_number, attempt_root in enumerate(attempt_roots):
        output_dir = attempt_root / "output"
        command = _prediction_command(
            python=python,
            trainer=trainer,
            predict_input=predict_input,
            checkpoint=checkpoint,
            scaler=scaler,
            output_dir=output_dir,
            target_sealed_capability_receipt=target_sealed_capability_receipt,
            expected_admitted_context={
                **usage_identity, "attempt_number": attempt_number
            },
            outer_fold=item.outer_fold,
            seed=item.seed,
            variant=str(selection["selected_variant"]),
            release_mode=str(selection["selected_release_mode"]),
            promotion_authorization=promotion_authorization,
            device=device,
            amp=amp,
        )
        attempt_invocation_path = attempt_root / "invocation.json"
        attempt_invocation = discovery.load_json(
            attempt_invocation_path, "promotion prediction attempt invocation"
        )
        if (
            discovery.canonical_content_sha256(attempt_invocation)
            != attempt_invocation.get("content_sha256")
            or attempt_invocation.get("attempt_number") != attempt_number
            or attempt_invocation.get("trainer_command") != command
            or any(
                attempt_invocation.get(name) != value
                for name, value in usage_identity.items()
            )
        ):
            raise discovery.CampaignError("promotion prediction attempt invocation drifted")
        context = {**usage_identity, "attempt_number": attempt_number}
        execution_invocation_path = attempt_root / "execution_invocation.json"
        result_path = attempt_root / "terminal_result.json"
        if not execution_invocation_path.is_file():
            raise discovery.CampaignError(
                "promotion prediction attempt lacks its V7 execution invocation"
            )
        if not result_path.exists():
            discovery._validate_execution_invocation(
                execution_invocation_path,
                phase="promotion_prediction",
                context=context,
                unit_invocation_path=attempt_invocation_path,
                workload_command=command,
            )
            recovery_command = discovery._admitted_command(
                python=python,
                wrapper=wrapper,
                gpu_lock=gpu_lock,
                gpu_ledger=gpu_ledger,
                usage_ledger=usage_ledger,
                result_file=result_path,
                phase="promotion_prediction",
                context=context,
                invocation_sha256=discovery.sha256_file(
                    execution_invocation_path
                ),
                authorization_path=admitted_authorization_path,
                authorization_sha256=admitted_authorization_sha256,
                trainer_command=command,
            )
            command_runner(
                recovery_command, float(discovery.GPU_BUDGET_SECONDS)
            )
            if not result_path.is_file():
                state = discovery._verify_usage_state(usage_ledger)
                if int(state.remaining_ns) <= 0:
                    raise discovery.BudgetExhausted(
                        "ten-GPU-hour budget is exhausted"
                    )
                raise discovery.CampaignError(
                    "GPU supervisor could not recover the existing promotion-prediction result"
                )
        result, binding = discovery._load_execution_terminal_result(
            invocation_path=execution_invocation_path,
            result_path=result_path,
            usage_ledger=usage_ledger,
            phase="promotion_prediction",
            context=context,
            unit_invocation_path=attempt_invocation_path,
            workload_command=command,
        )
        lifecycle_invocations.append(
            discovery.bind_file(execution_invocation_path)
        )
        terminal_results.append(binding)
        if result.get("reusable_success") is True:
            if successful_result is not None:
                raise discovery.CampaignError(
                    "promotion prediction has multiple successful attempts"
                )
            successful_result = result
            successful_attempt_root = attempt_root

    if successful_result is None:
        attempt_number = len(attempt_roots)
        attempt_root = attempts / f"attempt_{attempt_number:03d}"
        output_dir = attempt_root / "output"
        command = _prediction_command(
            python=python,
            trainer=trainer,
            predict_input=predict_input,
            checkpoint=checkpoint,
            scaler=scaler,
            output_dir=output_dir,
            target_sealed_capability_receipt=target_sealed_capability_receipt,
            expected_admitted_context={
                **usage_identity, "attempt_number": attempt_number
            },
            outer_fold=item.outer_fold,
            seed=item.seed,
            variant=str(selection["selected_variant"]),
            release_mode=str(selection["selected_release_mode"]),
            promotion_authorization=promotion_authorization,
            device=device,
            amp=amp,
        )
        context = {**usage_identity, "attempt_number": attempt_number}
        attempt_root = _publish_prediction_attempt_directory(
            attempts,
            attempt_number=attempt_number,
            attempt_invocation={
                "schema_version": 1,
                "classification": "adaptive_v3r1_target_free_promotion_prediction_invocation",
                "campaign_id": discovery.CAMPAIGN_ID,
                "campaign_revision": CAMPAIGN_REVISION,
                "infrastructure_revision": INFRASTRUCTURE_REVISION,
                "outer_fold": item.outer_fold,
                "seed": item.seed,
                "variant": selection["selected_variant"],
                "release_mode": selection["selected_release_mode"],
                "attempt_number": attempt_number,
                "governance": dict(governance),
                "model_source_kind": model_source_kind,
                "model_source_receipt": dict(model_source_receipt_binding),
                "scientific_signature_sha256": scientific_signature_sha256,
                "predict_input": discovery.bind_file(predict_input),
                "input_receipt": discovery.bind_file(input_receipt),
                "checkpoint": discovery.bind_file(checkpoint),
                "scaler": discovery.bind_file(scaler),
                "trainer_command": command,
                "usage_ledger_identity": discovery.bind_file(
                    usage_ledger_identity_path
                ),
                "target_access_authorized": False,
            },
            context=context,
            workload_command=command,
        )
        attempt_invocation_path = attempt_root / "invocation.json"
        execution_invocation_path = attempt_root / "execution_invocation.json"
        result_path = attempt_root / "terminal_result.json"
        lifecycle_invocations.append(
            discovery.bind_file(execution_invocation_path)
        )
        admitted = discovery._admitted_command(
            python=python,
            wrapper=wrapper,
            gpu_lock=gpu_lock,
            gpu_ledger=gpu_ledger,
            usage_ledger=usage_ledger,
            result_file=result_path,
            phase="promotion_prediction",
            context=context,
            invocation_sha256=discovery.sha256_file(execution_invocation_path),
            authorization_path=admitted_authorization_path,
            authorization_sha256=admitted_authorization_sha256,
            trainer_command=command,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        command_runner(admitted, float(discovery.GPU_BUDGET_SECONDS))
        if not result_path.is_file():
            state = discovery._verify_usage_state(usage_ledger)
            if int(state.remaining_ns) <= 0:
                raise discovery.BudgetExhausted("ten-GPU-hour budget is exhausted")
            raise discovery.CampaignError(
                "GPU supervisor exited without a promotion-prediction terminal result"
            )
        successful_result, binding = discovery._load_execution_terminal_result(
            invocation_path=execution_invocation_path,
            result_path=result_path,
            usage_ledger=usage_ledger,
            phase="promotion_prediction",
            context=context,
            unit_invocation_path=attempt_invocation_path,
            workload_command=command,
        )
        terminal_results.append(binding)
        successful_attempt_root = attempt_root
        if successful_result.get("reusable_success") is not True:
            if successful_result.get("hard_timeout_reached") is True:
                raise discovery.BudgetExhausted(
                    "promotion prediction reached hard GPU-hour ceiling"
                )
            raise discovery.CampaignError(
                f"promotion prediction failed for {item.outer_fold}/{item.seed}: "
                f"{successful_result.get('return_code')}"
            )
    assert successful_attempt_root is not None
    attempt_root = successful_attempt_root
    output_dir = attempt_root / "output"
    validated = validate_target_free_prediction(
        output_dir,
        expected_index=expected_index,
        outer_fold=item.outer_fold,
        seed=item.seed,
        variant=str(selection["selected_variant"]),
        release_mode=str(selection["selected_release_mode"]),
        predict_input=predict_input,
        checkpoint=checkpoint,
        scaler=scaler,
    )
    usage_fields = discovery.completion_usage_fields(
        usage_ledger,
        final_record_sha256=str(successful_result["terminal_record_sha256"]),
        expected_phase="promotion_prediction",
        expected_identity=usage_identity,
        expected_command_sha256=expected_usage_command_sha256,
        terminal_results=terminal_results,
        lifecycle_invocations=lifecycle_invocations,
        gpu_ledger=gpu_ledger,
        gpu_lock=gpu_lock,
    )
    return discovery.create_once_json(
        completion_path,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_target_free_promotion_prediction_completion",
            "campaign_id": discovery.CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "outer_fold": item.outer_fold,
            "seed": item.seed,
            "variant": selection["selected_variant"],
            "release_mode": selection["selected_release_mode"],
            "output_dir": str(output_dir),
            "attempt_invocation": discovery.bind_file(attempt_root / "invocation.json"),
            "promotion_model_source": model_source_binding,
            "governance": dict(governance),
            "selection_lock_sha256": selection_lock_sha256,
            "promotion_authorization_sha256": promotion_authorization_sha256,
            "input_receipt": input_receipt_binding,
            **usage_fields,
            "validated_output": validated,
            "target_fields_accessed_or_emitted": False,
            "commercial_claim_authorized": False,
        },
    )


def build_exact_cover_seal(
    *,
    project_root: Path,
    run_root: Path,
    receipts: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    governance: Mapping[str, Any],
    usage_ledger: Path,
    training: Mapping[tuple[int, int], discovery.TrainingInput],
) -> dict[str, Any]:
    expected_units = {(fold, seed) for fold in range(6) for seed in discovery.SEEDS}
    declared_keys = [
        (int(receipt["outer_fold"]), int(receipt["seed"])) for receipt in receipts
    ]
    if len(declared_keys) != 18 or len(set(declared_keys)) != 18 or set(declared_keys) != expected_units:
        raise discovery.CampaignError("promotion predictions are not the exact 6x3 unit cover")

    governance_owner = run_root / "PROMOTION_PREDICTION_EXACT_COVER_SEAL.json"
    selection_path = discovery.verify_binding(
        governance.get("selection_lock", {}),
        project_root=project_root,
        owner=governance_owner,
        label="promotion exact-cover selection lock",
    )
    authorization_path = discovery.verify_binding(
        governance.get("promotion_authorization", {}),
        project_root=project_root,
        owner=governance_owner,
        label="promotion exact-cover authorization",
    )

    discovery_seal_binding = selection.get("discovery_completion_seal")
    if not isinstance(discovery_seal_binding, Mapping):
        raise discovery.CampaignError("selection lacks its discovery completion seal")
    selection_owner = project_root / "selection_lock.json"
    discovery_seal_path = discovery.verify_binding(
        discovery_seal_binding,
        project_root=project_root,
        owner=selection_owner,
        label="selection discovery completion seal",
    )
    discovery_seal = discovery.load_json(
        discovery_seal_path, "promotion-bound discovery completion seal"
    )
    if discovery.canonical_content_sha256(discovery_seal) != discovery_seal.get(
        "content_sha256"
    ):
        raise discovery.CampaignError("promotion-bound discovery seal content drifted")
    discovery_units = discovery_seal.get("units")
    if not isinstance(discovery_units, list) or len(discovery_units) != 18:
        raise discovery.CampaignError("promotion-bound discovery seal is incomplete")
    receipt_specs: list[tuple[Mapping[str, Any], str, Mapping[str, Any]]] = []
    benchmark_owner = discovery_seal.get("pre_discovery_efficiency_benchmark")
    if not isinstance(benchmark_owner, Mapping) or not (
        benchmark_owner.get("included_in_gpu_exact_cover") is True
        and benchmark_owner.get("excluded_from_selection") is True
        and benchmark_owner.get("artifacts_quarantined") is True
    ):
        raise discovery.CampaignError(
            "promotion-bound discovery seal lacks its quarantined benchmark owner"
        )
    benchmark_receipt_path = discovery.verify_binding(
        benchmark_owner.get("receipt", {}),
        project_root=project_root,
        owner=discovery_seal_path,
        label="pre-discovery efficiency benchmark receipt",
    )
    benchmark_receipt = discovery.load_json(
        benchmark_receipt_path, "pre-discovery efficiency benchmark receipt"
    )
    if discovery.canonical_content_sha256(benchmark_receipt) != benchmark_receipt.get(
        "content_sha256"
    ):
        raise discovery.CampaignError("efficiency benchmark receipt content drifted")
    expected_benchmark_identity = {
        "campaign_revision": CAMPAIGN_REVISION,
        "infrastructure_revision": INFRASTRUCTURE_REVISION,
        "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    if not (
        benchmark_receipt.get("phase") == "efficiency_benchmark"
        and benchmark_receipt.get("usage_identity") == expected_benchmark_identity
        and all(
            benchmark_receipt.get(name) == value
            for name, value in expected_benchmark_identity.items()
        )
    ):
        raise discovery.CampaignError("efficiency benchmark owner identity drifted")
    receipt_specs.append(
        (benchmark_receipt, "efficiency_benchmark", expected_benchmark_identity)
    )
    benchmark_usage_hashes = benchmark_receipt.get("usage_record_sha256s")
    if not isinstance(benchmark_usage_hashes, list) or not benchmark_usage_hashes:
        raise discovery.CampaignError("efficiency benchmark receipt lacks GPU usage history")
    discovery_keys: set[tuple[int, int, str]] = set()
    discovery_usage_hashes: list[str] = []
    for unit in discovery_units:
        if not isinstance(unit, Mapping):
            raise discovery.CampaignError("promotion-bound discovery unit is invalid")
        key = (int(unit["outer_fold"]), int(unit["seed"]), str(unit["variant"]))
        if key in discovery_keys:
            raise discovery.CampaignError("promotion-bound discovery unit is duplicated")
        discovery_keys.add(key)
        receipt_path = discovery.verify_binding(
            unit.get("receipt", {}),
            project_root=project_root,
            owner=discovery_seal_path,
            label=f"promotion-bound discovery receipt {key}",
        )
        receipt = discovery.load_json(receipt_path, "promotion-bound discovery receipt")
        if discovery.canonical_content_sha256(receipt) != receipt.get("content_sha256"):
            raise discovery.CampaignError("promotion-bound discovery receipt content drifted")
        receipt_specs.append(
            (
                receipt,
                "discovery",
                {
                    "campaign_revision": CAMPAIGN_REVISION,
                    "infrastructure_revision": INFRASTRUCTURE_REVISION,
                    "outer_fold": key[0],
                    "seed": key[1],
                    "variant": key[2],
                },
            )
        )
        hashes = receipt.get("usage_record_sha256s")
        if not isinstance(hashes, list):
            raise discovery.CampaignError("discovery receipt lacks GPU usage history")
        discovery_usage_hashes.extend(str(value) for value in hashes)
    if discovery_keys != set(discovery.EXPECTED_DISCOVERY_UNITS):
        raise discovery.CampaignError("promotion-bound discovery units are not exact")

    if set(training) != expected_units:
        raise discovery.CampaignError("promotion training index is not the exact 6x3 cover")
    model_source_keys: set[tuple[int, int]] = set()
    local_training_keys: set[tuple[int, int]] = set()
    pointer_keys: set[tuple[int, int]] = set()
    model_sources: dict[
        tuple[int, int], locked_inputs.PromotionModelSource
    ] = {}
    model_source_bindings: dict[tuple[int, int], dict[str, Any]] = {}
    local_training_terminals: dict[tuple[int, int], str] = {}
    for key in sorted(expected_units):
        item = training[key]
        source = locked_inputs.resolve_promotion_model_source(
            project_root=project_root,
            run_root=run_root,
            cache_dir=item.cache_dir,
            outer_fold=key[0],
            seed=key[1],
            variant=str(selection["selected_variant"]),
        )
        if key in model_source_keys:
            raise discovery.CampaignError("promotion model source is duplicated")
        model_source_keys.add(key)
        model_sources[key] = source
        model_source_bindings[key] = {
            "kind": source.kind,
            "receipt": discovery.bind_file(source.receipt_path),
            "checkpoint": discovery.bind_file(source.checkpoint),
            "scaler": discovery.bind_file(source.scaler),
            "scientific_signature_sha256": source.scientific_signature_sha256,
        }
        if source.kind == "local_training":
            local_training_keys.add(key)
            training_terminal = source.receipt.get("usage_record_sha256")
            if not discovery._is_sha256(training_terminal):
                raise discovery.CampaignError(
                    "local promotion training lacks its terminal GPU record"
                )
            local_training_terminals[key] = str(training_terminal)
            receipt_specs.append(
                (
                    source.receipt,
                    "promotion_training",
                    {
                        "campaign_revision": CAMPAIGN_REVISION,
                        "infrastructure_revision": INFRASTRUCTURE_REVISION,
                        "outer_fold": key[0],
                        "seed": key[1],
                        "variant": str(selection["selected_variant"]),
                    },
                )
            )
        elif source.kind == "discovery_pointer":
            pointer_keys.add(key)
        else:
            raise discovery.CampaignError("unregistered promotion model source kind")
    if local_training_keys != EXPECTED_NEW_PROMOTION_TRAINING_UNITS:
        raise discovery.CampaignError("promotion does not have exactly twelve new trainings")
    if pointer_keys != EXPECTED_REUSE_UNITS:
        raise discovery.CampaignError("promotion does not have exactly six reuse pointers")
    by_unit: dict[tuple[int, int], Mapping[str, Any]] = {}
    prediction_terminals: dict[tuple[int, int], str] = {}
    per_seed_indices: dict[int, list[np.ndarray]] = {seed: [] for seed in discovery.SEEDS}
    unit_bindings = []
    prediction_proofs_completed = 0
    for receipt in receipts:
        key = (int(receipt["outer_fold"]), int(receipt["seed"]))
        if key in by_unit:
            raise discovery.CampaignError(f"duplicate promotion prediction receipt: {key}")
        by_unit[key] = receipt
        if receipt.get("promotion_model_source") != model_source_bindings[key]:
            raise discovery.CampaignError(
                "promotion prediction receipt model source differs from resolved owner"
            )
        prediction_terminal = receipt.get("usage_record_sha256")
        if not discovery._is_sha256(prediction_terminal):
            raise discovery.CampaignError(
                "promotion prediction lacks its terminal GPU record"
            )
        prediction_terminals[key] = str(prediction_terminal)
        receipt_specs.append(
            (
                receipt,
                "promotion_prediction",
                {
                    "campaign_revision": CAMPAIGN_REVISION,
                    "infrastructure_revision": INFRASTRUCTURE_REVISION,
                    "outer_fold": key[0],
                    "seed": key[1],
                    "variant": str(selection["selected_variant"]),
                    "release_mode": str(selection["selected_release_mode"]),
                },
            )
        )
        completion_path = (
            run_root
            / "predictions"
            / f"outer_{key[0]}_seed_{key[1]}"
            / "completion_receipt.json"
        )
        disk_receipt = discovery.load_json(
            completion_path, "promotion prediction completion receipt"
        )
        if discovery.canonical_content_sha256(disk_receipt) != disk_receipt.get(
            "content_sha256"
        ):
            raise discovery.CampaignError("promotion prediction receipt content drifted")
        if disk_receipt != receipt:
            raise discovery.CampaignError("in-memory promotion prediction receipt differs from disk")
        work = run_root / "inputs" / f"outer_{key[0]}_seed_{key[1]}"
        safe_anchor = work / "safe_proposer_anchor.npz"
        predict_input = work / "sanitized_test_input.npz"
        input_receipt_path = work / "sanitized_test_input.receipt.json"
        sanitized_receipt = locked_inputs.build_locked_input(
            project_root=project_root,
            cache_dir=training[key].cache_dir,
            proposer_anchor=safe_anchor,
            outer_fold=key[0],
            seed=key[1],
            output=predict_input,
            receipt_path=input_receipt_path,
            selection_lock_path=selection_path,
            promotion_authorization_path=authorization_path,
            model_source=model_sources[key],
        )
        with np.load(predict_input, allow_pickle=False) as archive:
            expected_index = np.asarray(archive["cache_index"], dtype=np.int64)
        output_dir = Path(str(receipt.get("output_dir", ""))).expanduser()
        if not output_dir.is_absolute():
            output_dir = (
                run_root
                / "predictions"
                / f"outer_{key[0]}_seed_{key[1]}"
                / output_dir
            ).resolve()
        validated_output = validate_target_free_prediction(
            output_dir,
            expected_index=expected_index,
            outer_fold=key[0],
            seed=key[1],
            variant=str(selection["selected_variant"]),
            release_mode=str(selection["selected_release_mode"]),
            predict_input=predict_input,
            checkpoint=model_sources[key].checkpoint,
            scaler=model_sources[key].scaler,
        )
        if receipt.get("validated_output") != validated_output:
            raise discovery.CampaignError(
                "promotion prediction validated output differs from live proof"
            )
        if not (
            sanitized_receipt.get("target_fields_present") is False
            and sanitized_receipt.get(
                "target_reference_qc_identity_protocol_columns_physically_present"
            )
            is False
            and validated_output.get("target_fields_present") is False
            and validated_output.get("identity_fields_present") is False
            and validated_output.get("protocol_fields_present") is False
        ):
            raise discovery.CampaignError(
                "promotion prediction target/context-free proof is incomplete"
            )
        prediction_binding = validated_output["prediction"]
        path = discovery.verify_binding(
            prediction_binding,
            project_root=project_root,
            owner=completion_path,
            label=f"promotion prediction output {key}",
        )
        with np.load(path, allow_pickle=False) as archive:
            per_seed_indices[key[1]].append(
                np.asarray(archive["cache_index"], dtype=np.int64)
            )
        prediction_proofs_completed += 1
        unit_bindings.append(
            {
                "outer_fold": key[0],
                "seed": key[1],
                "completion_receipt": discovery.bind_file(completion_path),
                "promotion_model_source": model_source_bindings[key],
                "sanitized_input_receipt": discovery.bind_file(input_receipt_path),
                "sanitized_input": dict(sanitized_receipt["output"]),
                "prediction": prediction_binding,
                "rows": validated_output["rows"],
            }
        )
    if set(by_unit) != expected_units:
        raise discovery.CampaignError("promotion predictions are not the exact 6x3 unit cover")
    canonical_index: np.ndarray | None = None
    seed_covers: dict[str, Any] = {}
    for seed in discovery.SEEDS:
        combined = np.concatenate(per_seed_indices[seed])
        if len(np.unique(combined)) != len(combined):
            raise discovery.CampaignError(f"promotion cache index duplicated for seed {seed}")
        ordered = np.sort(combined)
        if canonical_index is None:
            canonical_index = ordered
        elif not np.array_equal(ordered, canonical_index):
            raise discovery.CampaignError("promotion cache-index cover differs across seeds")
        seed_covers[str(seed)] = {
            "rows": int(len(ordered)),
            "unique_rows": int(len(np.unique(ordered))),
            "cache_index_sha256": discovery.semantic_sha256(ordered.tolist()),
        }
    assert canonical_index is not None
    if len(canonical_index) == 0:
        raise discovery.CampaignError("promotion exact cover is empty")
    if prediction_proofs_completed != 18:
        raise discovery.CampaignError(
            "promotion target/context-free prediction proof is not exact"
        )
    if len(receipt_specs) != 49:
        raise discovery.CampaignError("final GPU execution owner count is not exactly 49")
    try:
        with discovery.gpu_budget_ledger.locked_closed_snapshot(
            usage_ledger,
            budget_ns=discovery.gpu_budget_ledger.GPU_BUDGET_NS,
            expected_legacy_genesis_sha256=discovery._expected_legacy_genesis(
                usage_ledger
            ),
        ) as usage_state:
            records, elapsed = discovery.reconcile_usage_ledger(
                usage_ledger, receipt_specs, usage_state=usage_state
            )
            positions = {
                str(record["record_sha256"]): index
                for index, record in enumerate(records)
            }
            if not discovery_usage_hashes or any(
                record_hash not in positions for record_hash in discovery_usage_hashes
            ):
                raise discovery.CampaignError(
                    "discovery GPU usage history is absent from active ledger"
                )
            if any(
                str(record_hash) not in positions for record_hash in benchmark_usage_hashes
            ):
                raise discovery.CampaignError(
                    "benchmark GPU usage history is absent from active ledger"
                )
            if max(positions[str(value)] for value in benchmark_usage_hashes) >= min(
                positions[value] for value in discovery_usage_hashes
            ):
                raise discovery.CampaignError(
                    "efficiency benchmark did not terminate before discovery"
                )
            for key in sorted(local_training_keys):
                training_terminal = local_training_terminals[key]
                prediction_terminal = prediction_terminals[key]
                if (
                    training_terminal not in positions
                    or prediction_terminal not in positions
                    or positions[training_terminal] >= positions[prediction_terminal]
                ):
                    raise discovery.CampaignError(
                        "local promotion training terminal did not precede "
                        f"its prediction terminal: {key}"
                    )
            discovery_terminal = max(
                discovery_usage_hashes, key=positions.__getitem__
            )
            expected_legacy_genesis = discovery._expected_legacy_genesis(
                usage_ledger
            )
            if expected_legacy_genesis is not None:
                genesis_records = [
                    record
                    for record in records
                    if record.get("record_sha256") == expected_legacy_genesis
                ]
                if not (
                    len(genesis_records) == 1
                    and float(genesis_records[0].get("elapsed_seconds", -1.0))
                    == 377.0
                ):
                    raise discovery.CampaignError(
                        "legacy V1 377-second genesis is not counted exactly once"
                    )
            discovery.verify_usage_ledger_prefix_binding(
                usage_ledger,
                discovery_seal.get("gpu_usage_ledger", {}),
                project_root=project_root,
                owner=discovery_seal_path,
                terminal_record_sha256=discovery_terminal,
                usage_state=usage_state,
            )
            ledger_binding = discovery.usage_snapshot_binding(
                usage_ledger, usage_state
            )
            return discovery.create_once_json(
                run_root / "PROMOTION_PREDICTION_EXACT_COVER_SEAL.json",
                {
            "schema_version": 1,
            "classification": "adaptive_v3r1_target_free_promotion_prediction_exact_cover_seal",
            "campaign_id": discovery.CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "governance": dict(governance),
            "selected_variant": selection["selected_variant"],
            "selected_release_mode": selection["selected_release_mode"],
            "outer_folds": list(range(6)),
            "seeds": list(discovery.SEEDS),
            "completed_new_training_units": 12,
            "completed_reused_training_pointer_units": 6,
            "completed_model_source_units": 18,
            "completed_prediction_units": 18,
            "gpu_execution_owner_count": 49,
            "nonaccounting_pointer_receipt_count": 6,
            "gpu_execution_owner_breakdown": {
                "efficiency_benchmark": 1,
                "discovery_training": 18,
                "new_promotion_training": 12,
                "promotion_prediction": 18,
            },
            "legacy_v1_genesis_counted_once": expected_legacy_genesis is not None,
            "legacy_v1_genesis_seconds": (
                377.0 if expected_legacy_genesis is not None else None
            ),
            "reuse_pointer_receipts_own_gpu_records": False,
            "pre_discovery_efficiency_benchmark": {
                "receipt": discovery.bind_file(benchmark_receipt_path),
                "included_in_gpu_exact_cover": True,
                "excluded_from_selection": True,
                "artifacts_quarantined": True,
            },
            "seed_exact_covers": seed_covers,
            "canonical_cache_index_rows": int(len(canonical_index)),
            "canonical_cache_index_sha256": discovery.semantic_sha256(
                canonical_index.tolist()
            ),
            "all_prediction_fields_target_free": prediction_proofs_completed == 18,
            "all_prediction_fields_identity_protocol_free": prediction_proofs_completed == 18,
            "all_18_predictions_sealed_before_target_join": True,
            "target_join_performed": False,
            "target_access_authorized_by_this_seal": False,
            "release_mode_or_threshold_adapted": False,
            "gpu_elapsed_seconds": elapsed,
            "gpu_hours_hard": discovery.GPU_HOURS_HARD,
            "gpu_usage_ledger": ledger_binding,
            "gpu_usage_ledger_identity": discovery.bind_file(
                run_root / "GPU_USAGE_LEDGER_IDENTITY.json"
            ),
            "units": sorted(unit_bindings, key=lambda item: (item["outer_fold"], item["seed"])),
            "adaptive_retrospective_only": True,
            "commercial_claim_authorized": False,
                },
            )
    except (OSError, ValueError, RuntimeError) as error:
        if isinstance(error, discovery.CampaignError):
            raise
        raise discovery.CampaignError(
            f"cannot seal a stable closed GPU usage snapshot: {error}"
        ) from error


def _under(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_fixed_gpu_lock(project_root: Path, requested: Path) -> Path:
    """Fail closed unless the parent and admitted child use one lock identity."""

    gpu_lock = _under(project_root, requested)
    expected = _under(project_root, discovery.DEFAULT_GPU_LOCK)
    if gpu_lock != expected:
        raise discovery.CampaignError(
            "--gpu-lock must equal the canonical V8 GPU admission lock"
        )
    return gpu_lock


def _validate_fixed_runtime_paths(
    project_root: Path,
    *,
    selection_lock: Path,
    promotion_authorization: Path,
    trainer: Path,
    wrapper: Path,
    gpu_lock: Path,
    gpu_ledger: Path,
    usage_ledger: Path,
) -> dict[str, Path]:
    """Keep every parent/child execution ABI path in one authority domain."""

    requested = {
        "selection_lock": _under(project_root, selection_lock),
        "promotion_authorization": _under(
            project_root, promotion_authorization
        ),
        "trainer": _under(project_root, trainer),
        "gpu_wrapper": _under(project_root, wrapper),
        "gpu_lock": _validate_fixed_gpu_lock(project_root, gpu_lock),
        "gpu_execution_ledger": _under(project_root, gpu_ledger),
        "gpu_usage_ledger": _under(project_root, usage_ledger),
    }
    expected = {
        "selection_lock": _under(project_root, locked_inputs.SELECTION_RELATIVE),
        "promotion_authorization": _under(
            project_root, locked_inputs.PROMOTION_AUTH_RELATIVE
        ),
        "trainer": _under(project_root, discovery.TRAINER_RELATIVE),
        "gpu_wrapper": _under(project_root, discovery.GPU_WRAPPER_RELATIVE),
        "gpu_lock": _under(project_root, discovery.DEFAULT_GPU_LOCK),
        "gpu_execution_ledger": _under(project_root, discovery.DEFAULT_GPU_LEDGER),
        "gpu_usage_ledger": _under(project_root, discovery.DEFAULT_USAGE_LEDGER),
    }
    for name in requested:
        if requested[name] != expected[name]:
            raise discovery.CampaignError(
                f"--{name.replace('_', '-')} must equal the canonical V8 path"
            )
    return requested


def _read_frozen_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    lexical = Path(os.path.abspath(path.expanduser()))
    try:
        before = os.stat(lexical, follow_symlinks=False)
        descriptor = os.open(lexical, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise discovery.CampaignError(f"cannot open {label}: {lexical}") from error
    try:
        opened = os.fstat(descriptor)
        if not (
            stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == 0o444
            and (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino)
        ):
            raise discovery.CampaignError(f"{label} is not exact immutable 0444")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        current = os.stat(lexical, follow_symlinks=False)
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(opened, key) != getattr(after, key) for key in stable) or any(
            getattr(after, key) != getattr(current, key) for key in stable
        ):
            raise discovery.CampaignError(f"{label} changed during verification")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    document = _load_json_bytes(raw, path=lexical, label=label)
    if discovery.canonical_content_sha256(document) != document.get("content_sha256"):
        raise discovery.CampaignError(f"{label} canonical content drifted")
    return document, {
        "path": str(lexical), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)
    }


def validate_v8r4_promotion_authority(
    *, project_root: Path, selection_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection, selection_binding = _read_frozen_json(selection_path, "V8R4 selection lock")
    authorization, authorization_binding = _read_frozen_json(
        authorization_path, "V8R4 promotion authorization"
    )
    if not (
        selection.get("classification")
        == "adaptive_v3r1_v8r4_global_discovery_selection_lock"
        and selection.get("campaign_id") == discovery.CAMPAIGN_ID
        and selection.get("campaign_revision") == CAMPAIGN_REVISION
        and selection.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and selection.get("promotion_authorized") is True
        and selection.get("strict_lexicographic_improvement_over_v2") is True
        and selection.get("selection_process_pack_free") is True
        and selection.get("cross_outer_validation_reuse_present") is True
        and selection.get("fully_nested_confirmatory_oof") is False
        and selection.get("prospective_confirmation_required") is True
        and selection.get("commercial_claim_authorized") is False
    ):
        raise discovery.CampaignError("fixed campaign rejects non-V8R4 selection authority")
    scopes = authorization.get("authorized_scopes")
    selected_binding = authorization.get("discovery_selection_lock")
    if not (
        authorization.get("classification")
        == "adaptive_v3r1_v8r4_promotion_authorization"
        and authorization.get("campaign_id") == discovery.CAMPAIGN_ID
        and authorization.get("campaign_revision") == CAMPAIGN_REVISION
        and authorization.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and authorization.get("authorized_now") is True
        and scopes == ["promotion_training_pack", "outer_prediction_pack"]
        and authorization.get("promotion_authorized") is True
        and authorization.get("outer_test_targets_authorized") is False
        and authorization.get("release_mode_or_threshold_change_allowed") is False
        and authorization.get("selected_variant") == selection.get("selected_variant")
        and authorization.get("selected_release_mode") == selection.get("selected_release_mode")
        and authorization.get("cross_outer_validation_reuse_present") is True
        and authorization.get("fully_nested_confirmatory_oof") is False
        and authorization.get("prospective_confirmation_required") is True
        and authorization.get("commercial_claim_authorized") is False
        and isinstance(selected_binding, Mapping)
        and selected_binding.get("sha256") == selection_binding["sha256"]
        and selected_binding.get("bytes") == selection_binding["bytes"]
        and (
            Path(str(selected_binding.get("path", ""))).resolve()
            if Path(str(selected_binding.get("path", ""))).is_absolute()
            else (project_root / Path(str(selected_binding.get("path", "")))).resolve()
        ) == selection_path.resolve()
    ):
        raise discovery.CampaignError("fixed campaign rejects promotion authority ABI drift")
    return selection, authorization, {
        "selection_lock": selection_binding,
        "promotion_authorization": authorization_binding,
    }


def validate_target_sealed_fixed_capability(
    *, project_root: Path, receipt_path: Path, phase: str, outer_fold: int | None
) -> dict[str, Any]:
    result = discovery.validate_target_sealed_capability(
        project_root,
        receipt_path,
        expected_phase=phase,
        expected_outer_fold=outer_fold,
    )
    document = result.get("document", {})
    boundary = document.get("security_boundary", {})
    writable = document.get("writable_roots", {})
    lifecycle = writable.get("lifecycle", {}) if isinstance(writable, Mapping) else {}
    output = writable.get("output", {}) if isinstance(writable, Mapping) else {}
    if phase == "promotion_training":
        canonical_output = (
            project_root
            / DEFAULT_RUN_ROOT
            / "promotion_training_shards"
            / f"outer_{int(outer_fold)}"
        ).resolve()
    elif phase == "promotion_prediction":
        canonical_output = (
            project_root
            / DEFAULT_RUN_ROOT
            / "prediction_shards"
            / f"outer_{int(outer_fold)}"
        ).resolve()
    elif phase == "promotion_aggregation":
        canonical_output = (project_root / FIXED_AGGREGATION_OUTPUT_RELATIVE).resolve()
    else:
        raise discovery.CampaignError("fixed target-sealed phase is invalid")
    if not (
        document.get("classification")
        == "adaptive_v3r1_v8r4a_outer_target_sealed_runtime_capability_receipt"
        and document.get("campaign_revision") == CAMPAIGN_REVISION
        and document.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and Path(str(lifecycle.get("path", ""))).resolve() == receipt_path.parent.resolve()
        and Path(str(output.get("path", ""))).resolve()
        != Path(str(lifecycle.get("path", ""))).resolve()
        and Path(str(output.get("path", ""))).resolve() == canonical_output
        and
        boundary.get("legacy_combined_cache_mounted") is False
        and boundary.get("raw_or_target_root_mounted") is False
        and boundary.get("cross_outer_shard_mounted") is False
        and type(boundary.get("production_execution_authorized")) is bool
        and type(boundary.get("synthetic_validation_only")) is bool
        and boundary.get("synthetic_validation_only")
        is (not boundary.get("production_execution_authorized"))
        and boundary.get("atomic_replace_compatible") is True
        and boundary.get("v8r4a_ledger_migration_required") is False
        and boundary.get("v8r4a_migration_live_replay_validated") is True
        and boundary.get("dedicated_gpu_state_directory_capabilities") is True
        and boundary.get("exactly_three_mutable_state_directory_mounts") is True
        and boundary.get("lifecycle_mounted_read_only") is True
        and boundary.get("source_snapshot_exact_file_mounts") is True
        and boundary.get("complete_project_source_or_config_trees_mounted") is False
    ):
        raise discovery.CampaignError("fixed target-sealed capability boundary drifted")
    return dict(result)


def validate_v8r4_promotion_pack_index(
    *, index_path: Path, phase: str, outer_fold: int,
    authorization_binding: Mapping[str, Any]
) -> dict[str, Any]:
    document, _ = _read_frozen_json(index_path, f"V8R4 {phase} shard index")
    if phase not in {"promotion_training", "promotion_prediction"}:
        raise discovery.CampaignError("V8R4 promotion pack phase is invalid")
    training_phase = phase == "promotion_training"
    expected_class = (
        PROMOTION_TRAINING_INDEX_CLASSIFICATION
        if training_phase
        else PREDICTION_INDEX_CLASSIFICATION
    )
    units = document.get("units")
    expected_artifacts = (
        {"cache_manifest", "proposer_stack", "partition_manifest"}
        if training_phase
        else {
            "prediction_pack_manifest",
            "model_bound_prediction_pack_manifest",
            "outer_predict_input",
            "model_checkpoint",
            "model_scaler",
            "model_source_capability",
        }
    )
    common_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "outer_fold",
        "seeds",
        "unit_count",
        "completed_units",
        "status",
        "outer_test_opened",
        "combined_target_bearing_cache_consumer_access_authorized",
        "cross_outer_shard_mounted",
        "promotion_authorization",
        "units",
        "content_sha256",
    }
    expected_keys = (
        common_keys
        | {
            "physical_nonouter_training_packs",
            "outer_prediction_packs_absent",
            "promotion_scope",
        }
        if training_phase
        else common_keys
        | {
            "infrastructure_revision",
            "selected_variant",
            "physical_target_free_input_and_model_packs",
            "source_paths_or_peer_outputs_authorized_in_child",
            "model_source_shard_seal",
        }
    )
    if not (
        set(document) == expected_keys
        and document.get("classification") == expected_class
        and document.get("campaign_id") == discovery.CAMPAIGN_ID
        and document.get("campaign_revision") == CAMPAIGN_REVISION
        and document.get("outer_fold") == outer_fold
        and document.get("seeds") == list(discovery.SEEDS)
        and document.get("unit_count") == 3
        and document.get("completed_units") == 3
        and document.get("status") == "complete"
        and document.get("combined_target_bearing_cache_consumer_access_authorized") is False
        and document.get("cross_outer_shard_mounted") is False
        and document.get("promotion_authorization") == dict(authorization_binding)
        and isinstance(units, list) and len(units) == 3
        and (
            (
                document.get("physical_nonouter_training_packs") is True
                and document.get("outer_prediction_packs_absent") is True
                and document.get("promotion_scope") == "promotion_training_pack"
            )
            if training_phase
            else (
                document.get("infrastructure_revision") == "V8R4A"
                and isinstance(document.get("selected_variant"), str)
                and document.get("physical_target_free_input_and_model_packs") is True
                and document.get("source_paths_or_peer_outputs_authorized_in_child") is False
                and isinstance(document.get("model_source_shard_seal"), Mapping)
            )
        )
    ):
        raise discovery.CampaignError(
            "promotion pack producer/runtime ABI is not V8R4-ready; no fallback is authorized"
        )
    keys: set[tuple[int, int]] = set()
    for unit in units:
        artifacts = unit.get("artifacts") if isinstance(unit, Mapping) else None
        key = (int(unit.get("outer_fold", -1)), int(unit.get("seed", -1))) if isinstance(unit, Mapping) else (-1, -1)
        if not (
            isinstance(unit, Mapping)
            and unit.get("relative_path")
            == f"units/outer_{outer_fold}_seed_{key[1]}"
            and isinstance(artifacts, Mapping)
            and set(artifacts) == expected_artifacts
            and all(
                isinstance(binding, Mapping)
                and set(binding) == {"path", "sha256", "bytes"}
                for binding in artifacts.values()
            )
        ):
            raise discovery.CampaignError("V8R4 promotion pack artifact schema drifted")
        keys.add(key)
    if keys != {(outer_fold, seed) for seed in discovery.SEEDS}:
        raise discovery.CampaignError("V8R4 promotion pack lacks exact three-seed cover")
    return document


def _resolve_local_pack_binding(
    *,
    index_path: Path,
    binding: Mapping[str, Any],
    expected_path: Path,
    label: str,
) -> Path:
    path = discovery.verify_binding(
        binding,
        project_root=index_path.parent,
        owner=index_path,
        label=label,
    )
    if path != expected_path.resolve():
        raise discovery.CampaignError(f"{label} escaped its mounted shard")
    info = os.stat(path, follow_symlinks=False)
    if not (
        stat.S_ISREG(info.st_mode)
        and stat.S_IMODE(info.st_mode) == 0o444
        and info.st_nlink == 1
        and not path.is_symlink()
    ):
        raise discovery.CampaignError(f"{label} is not immutable 0444/nlink1")
    return path


def _validate_promotion_partition_manifest(
    path: Path,
    *,
    project_root: Path,
    outer_fold: int,
    seed: int,
    cache_manifest: Path,
    proposer_stack: Path,
    authorization_binding: Mapping[str, Any],
) -> dict[str, Any]:
    document = discovery.load_json(path, "V8R4 promotion partition manifest")
    base_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "outer_fold",
        "seed",
        "legacy_row_count",
        "partition",
        "legacy_inputs",
        "outputs",
        "integration_interface",
        "protected_outer_access",
        "preselection_prediction_boundary",
        "serialization",
        "claim_boundary",
        "promotion_scope",
        "promotion_authorization",
        "content_sha256",
    }
    partition = document.get("partition")
    outputs = document.get("outputs")
    if not (
        set(document) == base_keys
        and discovery.canonical_content_sha256(document)
        == document.get("content_sha256")
        and document.get("schema_version") == 1
        and document.get("classification")
        == "adaptive_v3r1_v8r4_sealed_nonouter_partition"
        and document.get("campaign_id") == discovery.CAMPAIGN_ID
        and document.get("campaign_revision") == CAMPAIGN_REVISION
        and document.get("outer_fold") == outer_fold
        and document.get("seed") == seed
        and document.get("promotion_scope") == "promotion_training_pack"
        and document.get("promotion_authorization")
        == dict(authorization_binding)
        and isinstance(partition, Mapping)
        and int(partition.get("discovery_rows", 0)) > 0
        and partition.get("discovery_outer_rows") == 0
        and partition.get("outer_prediction_pack_rows") == 0
        and partition.get("intersection_rows") == 0
        and partition.get("exact_disjoint_complement") is True
        and int(partition.get("union_rows", -1))
        == int(document.get("legacy_row_count", -2))
        and document.get("claim_boundary")
        == {
            "adaptive_retrospective_only": True,
            "commercial_or_confirmatory_claim_allowed": False,
            "outer_targets_opened": False,
        }
        and isinstance(outputs, Mapping)
        and set(outputs)
        == {
            "discovery_cache_manifest",
            "discovery_proposer_stack",
            "discovery_local_to_global_map",
        }
    ):
        raise discovery.CampaignError("V8R4 promotion partition invariant drifted")
    resolved_cache = discovery.verify_binding(
        outputs["discovery_cache_manifest"],
        project_root=project_root,
        owner=path,
        label="promotion partition cache manifest",
    )
    resolved_stack = discovery.verify_binding(
        outputs["discovery_proposer_stack"],
        project_root=project_root,
        owner=path,
        label="promotion partition proposer stack",
    )
    map_path = discovery.verify_binding(
        outputs["discovery_local_to_global_map"],
        project_root=project_root,
        owner=path,
        label="promotion partition cache-index map",
    )
    if not (
        resolved_cache == cache_manifest
        and resolved_stack == proposer_stack
        and map_path == cache_manifest.parent / "local_to_global_cache_index.npy"
    ):
        raise discovery.CampaignError("promotion partition output binding drifted")
    # Historical combined-cache bindings remain opaque: do not resolve them.
    legacy = document.get("legacy_inputs")
    if not (
        isinstance(legacy, Mapping)
        and set(legacy)
        == {"training_index", "cache_manifest", "proposer_stack", "cache_outputs"}
    ):
        raise discovery.CampaignError("promotion partition opaque provenance drifted")
    return document


def load_target_scoped_promotion_training_pack(
    *,
    project_root: Path,
    index_path: Path,
    outer_fold: int,
    authorization_binding: Mapping[str, Any],
) -> tuple[dict[tuple[int, int], discovery.TrainingInput], dict[str, Any]]:
    """Load one mounted promotion shard without resolving legacy provenance."""

    document = validate_v8r4_promotion_pack_index(
        index_path=index_path,
        phase="promotion_training",
        outer_fold=outer_fold,
        authorization_binding=authorization_binding,
    )
    _, index_binding = _read_frozen_json(index_path, "promotion training pack index")
    result: dict[tuple[int, int], discovery.TrainingInput] = {}
    for unit in document["units"]:
        seed = int(unit["seed"])
        key = (outer_fold, seed)
        unit_root = index_path.parent / str(unit["relative_path"])
        artifacts = unit["artifacts"]
        cache_manifest = _resolve_local_pack_binding(
            index_path=index_path,
            binding=artifacts["cache_manifest"],
            expected_path=unit_root / "discovery_cache/manifest.json",
            label=f"promotion cache manifest {key}",
        )
        proposer_stack = _resolve_local_pack_binding(
            index_path=index_path,
            binding=artifacts["proposer_stack"],
            expected_path=unit_root / "discovery_proposer_stack.npz",
            label=f"promotion proposer stack {key}",
        )
        partition_manifest = _resolve_local_pack_binding(
            index_path=index_path,
            binding=artifacts["partition_manifest"],
            expected_path=unit_root / "PARTITION_MANIFEST.json",
            label=f"promotion partition manifest {key}",
        )
        _validate_promotion_partition_manifest(
            partition_manifest,
            project_root=project_root,
            outer_fold=outer_fold,
            seed=seed,
            cache_manifest=cache_manifest,
            proposer_stack=proposer_stack,
            authorization_binding=authorization_binding,
        )
        cache_dir = cache_manifest.parent
        cache_document, cache_binding = discovery.verify_training_cache_inputs(
            project_root, cache_dir, outer_fold=outer_fold
        )
        if not (
            cache_document.get("promotion_scope") == "promotion_training_pack"
            and cache_document.get("promotion_authorization")
            == dict(authorization_binding)
        ):
            raise discovery.CampaignError("promotion cache authorization drifted")
        stack_binding = discovery.verify_training_bound_file(
            project_root,
            proposer_stack,
            expected_sha256=str(artifacts["proposer_stack"]["sha256"]),
            expected_bytes=int(artifacts["proposer_stack"]["bytes"]),
        )
        stack_index = discovery._validate_stack_scope(
            proposer_stack, outer_fold, seed
        )
        try:
            metadata_index = np.genfromtxt(
                cache_dir / "metadata.csv",
                delimiter=",",
                names=True,
                usecols=(0,),
                dtype=np.int64,
                encoding="utf-8",
            )
            if metadata_index.dtype.names:
                metadata_index = np.asarray(
                    metadata_index[metadata_index.dtype.names[0]]
                )
            metadata_index = np.atleast_1d(metadata_index).astype(np.int64)
        except (OSError, ValueError) as error:
            raise discovery.CampaignError(
                f"cannot read promotion training cache index: {key}"
            ) from error
        if not np.array_equal(metadata_index, stack_index):
            raise discovery.CampaignError(
                f"promotion training cache/proposer cover differs: {key}"
            )
        result[key] = discovery.TrainingInput(
            outer_fold=outer_fold,
            seed=seed,
            cache_dir=cache_dir,
            cache_manifest_sha256=discovery.sha256_file(cache_manifest),
            proposer_stack=proposer_stack,
            proposer_stack_sha256=discovery.sha256_file(proposer_stack),
            cache_input_binding=cache_binding,
            proposer_stack_binding=stack_binding,
            partition_manifest_binding=discovery.bind_file(partition_manifest),
        )
    if set(result) != {(outer_fold, seed) for seed in discovery.SEEDS}:
        raise discovery.CampaignError("promotion training pack is not exact three-seed cover")
    return result, index_binding


def load_target_scoped_prediction_pack(
    *,
    index_path: Path,
    outer_fold: int,
    selection: Mapping[str, Any],
    governance: Mapping[str, Any],
) -> tuple[
    dict[tuple[int, int], tuple[discovery.TrainingInput, locked_inputs.MountedPredictionPack]],
    dict[str, Any],
]:
    document = validate_v8r4_promotion_pack_index(
        index_path=index_path,
        phase="promotion_prediction",
        outer_fold=outer_fold,
        authorization_binding=governance["promotion_authorization"],
    )
    _, index_binding = _read_frozen_json(index_path, "model-bound prediction pack index")
    if document.get("selected_variant") != selection.get("selected_variant"):
        raise discovery.CampaignError("prediction pack selected variant drifted")
    result: dict[
        tuple[int, int],
        tuple[discovery.TrainingInput, locked_inputs.MountedPredictionPack],
    ] = {}
    for unit in document["units"]:
        seed = int(unit["seed"])
        key = (outer_fold, seed)
        unit_root = (index_path.parent / str(unit["relative_path"])).resolve()
        mounted = locked_inputs.validate_model_bound_prediction_pack(
            unit_root,
            outer_fold=outer_fold,
            seed=seed,
            selected_variant=str(selection["selected_variant"]),
            expected_promotion_authorization=governance[
                "promotion_authorization"
            ],
            expected_selection_lock=governance["selection_lock"],
        )
        artifacts = unit["artifacts"]
        for name, filename in (
            ("prediction_pack_manifest", "OUTER_PREDICTION_PACK_MANIFEST.json"),
            (
                "model_bound_prediction_pack_manifest",
                locked_inputs.MODEL_BOUND_MANIFEST_FILENAME,
            ),
            ("outer_predict_input", "outer_predict_input.npz"),
            ("model_checkpoint", "model_checkpoint.pt"),
            ("model_scaler", "model_scaler.json"),
            (
                "model_source_capability",
                locked_inputs.MODEL_SOURCE_CAPABILITY_FILENAME,
            ),
        ):
            _resolve_local_pack_binding(
                index_path=index_path,
                binding=artifacts[name],
                expected_path=unit_root / filename,
                label=f"prediction pack {name} {key}",
            )
        item = discovery.TrainingInput(
            outer_fold=outer_fold,
            seed=seed,
            cache_dir=unit_root,
            cache_manifest_sha256=discovery.sha256_file(
                unit_root / locked_inputs.MODEL_BOUND_MANIFEST_FILENAME
            ),
            proposer_stack=mounted.input_path,
            proposer_stack_sha256=discovery.sha256_file(mounted.input_path),
        )
        result[key] = (item, mounted)
    if set(result) != {(outer_fold, seed) for seed in discovery.SEEDS}:
        raise discovery.CampaignError("prediction pack is not exact three-seed cover")
    seal_binding = document.get("model_source_shard_seal")
    if not isinstance(seal_binding, Mapping):
        raise discovery.CampaignError("prediction pack lacks model-source shard seal")
    _resolve_local_pack_binding(
        index_path=index_path,
        binding=seal_binding,
        expected_path=index_path.parent / "MODEL_SOURCE_SHARD_SEAL.json",
        label="model-source shard seal",
    )
    return result, index_binding


def build_promotion_model_shard_completion(
    *,
    project_root: Path,
    run_root: Path,
    outer_fold: int,
    receipts: Sequence[Mapping[str, Any]],
    training: Mapping[tuple[int, int], discovery.TrainingInput],
    selection: Mapping[str, Any],
    governance: Mapping[str, Any],
    training_index_binding: Mapping[str, Any],
    usage_ledger: Path,
    gpu_ledger: Path,
    gpu_lock: Path,
) -> dict[str, Any]:
    expected = {(outer_fold, seed) for seed in discovery.SEEDS}
    keys = {(int(row["outer_fold"]), int(row["seed"])) for row in receipts}
    if (
        outer_fold not in locked_inputs.NEW_PROMOTION_TRAINING_FOLDS
        or len(receipts) != 3
        or keys != expected
        or set(training) != expected
    ):
        raise discovery.CampaignError(
            "promotion model shard is not an exact authorized three-seed cover"
        )
    units: list[dict[str, Any]] = []
    with discovery.gpu_budget_ledger.locked_closed_snapshot(
        usage_ledger,
        budget_ns=discovery.gpu_budget_ledger.GPU_BUDGET_NS,
        expected_legacy_genesis_sha256=discovery._expected_legacy_genesis(
            usage_ledger
        ),
    ) as usage_state:
        for receipt in receipts:
            seed = int(receipt["seed"])
            key = (outer_fold, seed)
            receipt_path = (
                run_root
                / "training"
                / f"outer_{outer_fold}_seed_{seed}"
                / "completion_receipt.json"
            )
            live = discovery.load_json(
                receipt_path, "promotion model unit completion"
            )
            if not (
                live == receipt
                and discovery.canonical_content_sha256(live)
                == live.get("content_sha256")
                and live.get("classification")
                == "adaptive_v3r1_fixed_promotion_training_completion"
                and live.get("variant") == selection.get("selected_variant")
            ):
                raise discovery.CampaignError(
                    "promotion model completion receipt drifted"
                )
            discovery.validate_completion_receipt_usage(
                usage_ledger,
                live,
                expected_phase="promotion_training",
                expected_identity={
                    "campaign_revision": CAMPAIGN_REVISION,
                    "infrastructure_revision": INFRASTRUCTURE_REVISION,
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "variant": selection["selected_variant"],
                },
                expected_gpu_ledger=gpu_ledger,
                expected_gpu_lock=gpu_lock,
                usage_state=usage_state,
            )
            source = locked_inputs.resolve_promotion_model_source(
                project_root=project_root,
                run_root=run_root,
                cache_dir=training[key].cache_dir,
                outer_fold=outer_fold,
                seed=seed,
                variant=str(selection["selected_variant"]),
            )
            if source.kind != "local_training":
                raise discovery.CampaignError(
                    "new promotion model shard resolved a nonlocal source"
                )
            units.append(
                {
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "completion_receipt": discovery.bind_file(receipt_path),
                    "scientific_signature_sha256": source.scientific_signature_sha256,
                    "model_checkpoint": discovery.bind_file(source.checkpoint),
                    "model_scaler": discovery.bind_file(source.scaler),
                }
            )
        usage_prefix = discovery.usage_snapshot_binding(
            usage_ledger, usage_state
        )
        return discovery.create_once_json(
            run_root / "PROMOTION_MODEL_SHARD_COMPLETION_SEAL.json",
            {
                "schema_version": 1,
                "classification": PROMOTION_MODEL_SHARD_SEAL_CLASSIFICATION,
                "campaign_id": discovery.CAMPAIGN_ID,
                "campaign_revision": CAMPAIGN_REVISION,
                "infrastructure_revision": INFRASTRUCTURE_REVISION,
                "outer_fold": outer_fold,
                "seeds": list(discovery.SEEDS),
                "selected_variant": selection["selected_variant"],
                "unit_count": 3,
                "exact_three_seed_cover": True,
                "training_pack_index": dict(training_index_binding),
                "selection_lock": dict(governance["selection_lock"]),
                "promotion_authorization": dict(
                    governance["promotion_authorization"]
                ),
                "gpu_usage_ledger_prefix": usage_prefix,
                "units": sorted(units, key=lambda row: int(row["seed"])),
                "outer_test_opened": False,
                "validation_scores_changed_execution": False,
                "ready_for_host_model_bound_pack_build": True,
                "commercial_claim_authorized": False,
            },
        )


def build_prediction_shard_completion(
    *,
    run_root: Path,
    outer_fold: int,
    receipts: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    governance: Mapping[str, Any],
    prediction_index_binding: Mapping[str, Any],
    model_source_shard_seal_binding: Mapping[str, Any],
    usage_ledger: Path,
    gpu_ledger: Path,
    gpu_lock: Path,
) -> dict[str, Any]:
    expected = {(outer_fold, seed) for seed in discovery.SEEDS}
    keys = {(int(row["outer_fold"]), int(row["seed"])) for row in receipts}
    if len(receipts) != 3 or keys != expected:
        raise discovery.CampaignError(
            "prediction shard is not an exact three-seed cover"
        )
    units: list[dict[str, Any]] = []
    row_proofs: set[tuple[int, str]] = set()
    with discovery.gpu_budget_ledger.locked_closed_snapshot(
        usage_ledger,
        budget_ns=discovery.gpu_budget_ledger.GPU_BUDGET_NS,
        expected_legacy_genesis_sha256=discovery._expected_legacy_genesis(
            usage_ledger
        ),
    ) as usage_state:
        for receipt in receipts:
            seed = int(receipt["seed"])
            receipt_path = (
                run_root
                / "predictions"
                / f"outer_{outer_fold}_seed_{seed}"
                / "completion_receipt.json"
            )
            live = discovery.load_json(
                receipt_path, "promotion prediction unit completion"
            )
            validated = live.get("validated_output")
            if not (
                live == receipt
                and discovery.canonical_content_sha256(live)
                == live.get("content_sha256")
                and live.get("classification")
                == "adaptive_v3r1_target_free_promotion_prediction_completion"
                and live.get("variant") == selection.get("selected_variant")
                and live.get("release_mode")
                == selection.get("selected_release_mode")
                and live.get("target_fields_accessed_or_emitted") is False
                and isinstance(validated, Mapping)
                and type(validated.get("rows")) is int
                and int(validated["rows"]) > 0
                and discovery._is_sha256(validated.get("cache_index_sha256"))
            ):
                raise discovery.CampaignError(
                    "promotion prediction completion receipt drifted"
                )
            discovery.validate_completion_receipt_usage(
                usage_ledger,
                live,
                expected_phase="promotion_prediction",
                expected_identity={
                    "campaign_revision": CAMPAIGN_REVISION,
                    "infrastructure_revision": INFRASTRUCTURE_REVISION,
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "variant": selection["selected_variant"],
                    "release_mode": selection["selected_release_mode"],
                },
                expected_gpu_ledger=gpu_ledger,
                expected_gpu_lock=gpu_lock,
                usage_state=usage_state,
            )
            row_proofs.add(
                (int(validated["rows"]), str(validated["cache_index_sha256"]))
            )
            units.append(
                {
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "completion_receipt": discovery.bind_file(receipt_path),
                    "promotion_model_source": dict(
                        live["promotion_model_source"]
                    ),
                    "rows": int(validated["rows"]),
                    "cache_index_sha256": str(
                        validated["cache_index_sha256"]
                    ),
                    "prediction": dict(validated["prediction"]),
                    "prediction_manifest": dict(validated["manifest"]),
                }
            )
        if len(row_proofs) != 1:
            raise discovery.CampaignError(
                "prediction seeds do not share one exact outer cache-index cover"
            )
        rows, cache_index_sha256 = next(iter(row_proofs))
        usage_prefix = discovery.usage_snapshot_binding(
            usage_ledger, usage_state
        )
        return discovery.create_once_json(
            run_root / "PREDICTION_SHARD_COMPLETION_SEAL.json",
            {
                "schema_version": 1,
                "classification": PREDICTION_SHARD_SEAL_CLASSIFICATION,
                "campaign_id": discovery.CAMPAIGN_ID,
                "campaign_revision": CAMPAIGN_REVISION,
                "infrastructure_revision": INFRASTRUCTURE_REVISION,
                "outer_fold": outer_fold,
                "seeds": list(discovery.SEEDS),
                "selected_variant": selection["selected_variant"],
                "selected_release_mode": selection["selected_release_mode"],
                "unit_count": 3,
                "exact_three_seed_cover": True,
                "row_count_per_seed": rows,
                "cache_index_sha256": cache_index_sha256,
                "prediction_pack_index": dict(prediction_index_binding),
                "model_source_shard_seal": dict(
                    model_source_shard_seal_binding
                ),
                "selection_lock": dict(governance["selection_lock"]),
                "promotion_authorization": dict(
                    governance["promotion_authorization"]
                ),
                "gpu_usage_ledger_prefix": usage_prefix,
                "units": sorted(units, key=lambda row: int(row["seed"])),
                "target_fields_accessed_or_emitted": False,
                "ready_for_pack_free_promotion_aggregation": True,
                "commercial_claim_authorized": False,
            },
        )


def build_pack_free_promotion_exact_cover(
    *,
    run_root: Path,
    model_source_seals: Sequence[tuple[Path, Mapping[str, Any]]],
    prediction_seals: Sequence[tuple[Path, Mapping[str, Any]]],
    selection: Mapping[str, Any],
    governance: Mapping[str, Any],
    usage_ledger: Path,
) -> dict[str, Any]:
    if len(model_source_seals) != 6 or len(prediction_seals) != 6:
        raise discovery.CampaignError(
            "pack-free promotion aggregation requires six model and six prediction seals"
        )
    model_by_fold: dict[int, tuple[Path, Mapping[str, Any]]] = {}
    prediction_by_fold: dict[int, tuple[Path, Mapping[str, Any]]] = {}
    for path, seal in model_source_seals:
        fold = int(seal.get("outer_fold", -1))
        if fold in model_by_fold:
            raise discovery.CampaignError("duplicate model-source shard seal")
        model_by_fold[fold] = (path, seal)
    for path, seal in prediction_seals:
        fold = int(seal.get("outer_fold", -1))
        if fold in prediction_by_fold:
            raise discovery.CampaignError("duplicate prediction shard seal")
        prediction_by_fold[fold] = (path, seal)
    if set(model_by_fold) != set(range(6)) or set(prediction_by_fold) != set(range(6)):
        raise discovery.CampaignError("promotion seals do not cover all six folds")

    normalized_models: list[dict[str, Any]] = []
    normalized_predictions: list[dict[str, Any]] = []
    total_rows = 0
    with discovery.gpu_budget_ledger.locked_closed_snapshot(
        usage_ledger,
        budget_ns=discovery.gpu_budget_ledger.GPU_BUDGET_NS,
        expected_legacy_genesis_sha256=discovery._expected_legacy_genesis(
            usage_ledger
        ),
    ) as usage_state:
        for fold in range(6):
            model_path, model = model_by_fold[fold]
            prediction_path, prediction = prediction_by_fold[fold]
            if not (
                discovery.canonical_content_sha256(model)
                == model.get("content_sha256")
                and model.get("classification")
                == "adaptive_v3r1_v8r4a_model_source_shard_seal"
                and model.get("campaign_id") == discovery.CAMPAIGN_ID
                and model.get("campaign_revision") == CAMPAIGN_REVISION
                and model.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
                and model.get("outer_fold") == fold
                and model.get("seeds") == list(discovery.SEEDS)
                and model.get("selected_variant")
                == selection.get("selected_variant")
                and model.get("unit_count") == 3
                and model.get("exact_three_seed_cover") is True
                and model.get("selection_lock") == governance.get("selection_lock")
                and model.get("promotion_authorization")
                == governance.get("promotion_authorization")
                and model.get("target_or_prediction_values_present") is False
                and model.get("commercial_or_confirmatory_claim_allowed") is False
                and isinstance(model.get("units"), list)
                and len(model["units"]) == 3
            ):
                raise discovery.CampaignError("model-source shard seal drifted")
            model_units = model["units"]
            model_keys = {
                (int(row.get("outer_fold", -1)), int(row.get("seed", -1)))
                for row in model_units
                if isinstance(row, Mapping)
            }
            model_row_proofs = {
                (int(row.get("row_count", -1)), str(row.get("global_cache_index_sha256", "")))
                for row in model_units
                if isinstance(row, Mapping)
            }
            if model_keys != {(fold, seed) for seed in discovery.SEEDS} or len(
                model_row_proofs
            ) != 1:
                raise discovery.CampaignError("model-source seal unit cover drifted")

            if not (
                discovery.canonical_content_sha256(prediction)
                == prediction.get("content_sha256")
                and prediction.get("classification")
                == PREDICTION_SHARD_SEAL_CLASSIFICATION
                and prediction.get("campaign_id") == discovery.CAMPAIGN_ID
                and prediction.get("campaign_revision") == CAMPAIGN_REVISION
                and prediction.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
                and prediction.get("outer_fold") == fold
                and prediction.get("seeds") == list(discovery.SEEDS)
                and prediction.get("selected_variant")
                == selection.get("selected_variant")
                and prediction.get("selected_release_mode")
                == selection.get("selected_release_mode")
                and prediction.get("unit_count") == 3
                and prediction.get("exact_three_seed_cover") is True
                and prediction.get("model_source_shard_seal")
                == discovery.bind_file(model_path)
                and prediction.get("selection_lock")
                == governance.get("selection_lock")
                and prediction.get("promotion_authorization")
                == governance.get("promotion_authorization")
                and prediction.get("target_fields_accessed_or_emitted") is False
                and prediction.get("ready_for_pack_free_promotion_aggregation") is True
                and prediction.get("commercial_claim_authorized") is False
            ):
                raise discovery.CampaignError("prediction shard seal drifted")
            model_rows, model_index_sha = next(iter(model_row_proofs))
            if not (
                prediction.get("row_count_per_seed") == model_rows
                and prediction.get("cache_index_sha256") == model_index_sha
            ):
                raise discovery.CampaignError(
                    "model-source/prediction outer-cover proof differs"
                )
            prefix = prediction.get("gpu_usage_ledger_prefix")
            if not isinstance(prefix, Mapping):
                raise discovery.CampaignError(
                    "prediction shard lacks GPU usage prefix"
                )
            discovery.verify_usage_ledger_prefix_binding(
                usage_ledger,
                prefix,
                project_root=run_root,
                owner=prediction_path,
                terminal_record_sha256=str(
                    prefix.get("terminal_record_sha256", "")
                ),
                usage_state=usage_state,
            )
            total_rows += int(model_rows)
            normalized_models.append(
                {"outer_fold": fold, "seal": discovery.bind_file(model_path)}
            )
            normalized_predictions.append(
                {
                    "outer_fold": fold,
                    "seal": discovery.bind_file(prediction_path),
                    "rows": int(model_rows),
                    "cache_index_sha256": model_index_sha,
                }
            )
        if total_rows <= 0:
            raise discovery.CampaignError("promotion exact cover has no rows")
        ledger_prefix = discovery.usage_snapshot_binding(
            usage_ledger, usage_state
        )
        return discovery.create_once_json(
            run_root / "PROMOTION_PREDICTION_EXACT_COVER_SEAL.json",
            {
                "schema_version": 1,
                "classification": PROMOTION_EXACT_COVER_CLASSIFICATION,
                "campaign_id": discovery.CAMPAIGN_ID,
                "campaign_revision": CAMPAIGN_REVISION,
                "infrastructure_revision": INFRASTRUCTURE_REVISION,
                "outer_folds": list(range(6)),
                "seeds": list(discovery.SEEDS),
                "selected_variant": selection["selected_variant"],
                "selected_release_mode": selection["selected_release_mode"],
                "model_source_shard_count": 6,
                "prediction_shard_count": 6,
                "completed_model_source_units": 18,
                "completed_prediction_units": 18,
                "total_unique_outer_rows": total_rows,
                "model_source_shards": normalized_models,
                "prediction_shards": normalized_predictions,
                "selection_lock": dict(governance["selection_lock"]),
                "promotion_authorization": dict(
                    governance["promotion_authorization"]
                ),
                "gpu_usage_ledger_prefix": ledger_prefix,
                "all_18_predictions_sealed_before_target_join": True,
                "pack_or_peer_output_opened_by_aggregator": False,
                "outer_reference_join_authorized": False,
                "adaptive_retrospective_only": True,
                "fully_nested_confirmatory_oof": False,
                "prospective_confirmation_required": True,
                "commercial_claim_authorized": False,
            },
        )


def require_v8r4a_runtime_migration(
    *, project_root: Path = PROJECT_ROOT, production: bool
) -> dict[str, Any] | None:
    if not production:
        return None
    return discovery.validate_v8r4a_gpu_state(project_root)


# Explicit tombstones: V8/V2 combined-cache and safe-anchor entry points can
# no longer be called by a V8R4 consumer, including by tests or imports.
def _validate_canonical_v2_anchor_source(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
    del args, kwargs
    raise discovery.CampaignError("V8R4 forbids every legacy V2 combined-cache anchor")


def _safe_anchor_from_locked_v2(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
    del args, kwargs
    raise discovery.CampaignError("V8R4 forbids every legacy V2 safe-anchor path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--promotion-model-shard", type=int, choices=range(6))
    mode.add_argument("--prediction-shard", type=int, choices=range(6))
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--sealed-pack-index", type=Path)
    parser.add_argument("--target-sealed-capability-receipt", type=Path)
    parser.add_argument("--synthetic-preflight", action="store_true")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--selection-lock", type=Path, default=locked_inputs.SELECTION_RELATIVE)
    parser.add_argument("--promotion-authorization", type=Path, default=locked_inputs.PROMOTION_AUTH_RELATIVE)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--trainer", type=Path, default=discovery.TRAINER_RELATIVE)
    parser.add_argument("--gpu-wrapper", type=Path, default=discovery.GPU_WRAPPER_RELATIVE)
    parser.add_argument("--gpu-lock", type=Path, default=discovery.DEFAULT_GPU_LOCK)
    parser.add_argument("--gpu-ledger", type=Path, default=discovery.DEFAULT_GPU_LEDGER)
    parser.add_argument("--usage-ledger", type=Path, default=discovery.DEFAULT_USAGE_LEDGER)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # Reject every authority/capability path override before argparse's
    # required shard choice can mask the security failure boundary.
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    early.add_argument("--selection-lock", type=Path, default=locked_inputs.SELECTION_RELATIVE)
    early.add_argument("--promotion-authorization", type=Path, default=locked_inputs.PROMOTION_AUTH_RELATIVE)
    early.add_argument("--trainer", type=Path, default=discovery.TRAINER_RELATIVE)
    early.add_argument("--gpu-wrapper", type=Path, default=discovery.GPU_WRAPPER_RELATIVE)
    early.add_argument("--gpu-lock", type=Path, default=discovery.DEFAULT_GPU_LOCK)
    early.add_argument("--gpu-ledger", type=Path, default=discovery.DEFAULT_GPU_LEDGER)
    early.add_argument("--usage-ledger", type=Path, default=discovery.DEFAULT_USAGE_LEDGER)
    early_args, _ = early.parse_known_args(raw_argv)
    early_root = early_args.project_root.expanduser().resolve()
    try:
        _validate_fixed_runtime_paths(
            early_root,
            selection_lock=early_args.selection_lock,
            promotion_authorization=early_args.promotion_authorization,
            trainer=early_args.trainer,
            wrapper=early_args.gpu_wrapper,
            gpu_lock=early_args.gpu_lock,
            gpu_ledger=early_args.gpu_ledger,
            usage_ledger=early_args.usage_ledger,
        )
    except discovery.CampaignError as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}, sort_keys=True))
        return 2
    args = build_parser().parse_args(raw_argv)
    project_root = args.project_root.expanduser().resolve()
    try:
        _validate_fixed_runtime_paths(
            project_root,
            selection_lock=args.selection_lock,
            promotion_authorization=args.promotion_authorization,
            trainer=args.trainer, wrapper=args.gpu_wrapper,
            gpu_lock=args.gpu_lock, gpu_ledger=args.gpu_ledger,
            usage_ledger=args.usage_ledger,
        )
        v8r4_fixed_execution_plan()
        # This is the authorized production terminal until V8R4A.  It is
        # intentionally before governance, pack, target, output, or GPU access.
        require_v8r4a_runtime_migration(
            project_root=project_root,
            production=not args.synthetic_preflight,
        )
        selection_path = _under(project_root, args.selection_lock)
        authorization_path = _under(project_root, args.promotion_authorization)
        selection, _authorization, governance = validate_v8r4_promotion_authority(
            project_root=project_root,
            selection_path=selection_path,
            authorization_path=authorization_path,
        )
        phase = (
            "promotion_aggregation"
            if args.aggregate
            else (
                "promotion_training"
                if args.promotion_model_shard is not None
                else "promotion_prediction"
            )
        )
        outer_fold = (
            None
            if args.aggregate
            else int(
                args.promotion_model_shard
                if args.promotion_model_shard is not None
                else args.prediction_shard
            )
        )
        if args.target_sealed_capability_receipt is None:
            raise discovery.CampaignError(
                "fixed execution requires one exact target-sealed capability"
            )
        capability = validate_target_sealed_fixed_capability(
            project_root=project_root,
            receipt_path=_under(
                project_root, args.target_sealed_capability_receipt
            ),
            phase=phase,
            outer_fold=outer_fold,
        )
        capability_document = capability.get("document")
        if not isinstance(capability_document, Mapping):
            raise discovery.CampaignError("fixed runtime capability is malformed")
        writable = capability_document.get("writable_roots")
        output_binding = (
            writable.get("output") if isinstance(writable, Mapping) else None
        )
        run_root = _under(project_root, args.run_root)
        if not (
            isinstance(output_binding, Mapping)
            and Path(str(output_binding.get("path", ""))).resolve()
            == run_root.resolve()
        ):
            raise discovery.CampaignError(
                "--run-root must equal the capability's canonical dedicated output"
            )
        capability_governance = capability_document.get("governance_files")
        if not isinstance(capability_governance, Mapping):
            raise discovery.CampaignError("fixed capability lacks governance files")
        for role, canonical in (
            ("selection_lock", governance["selection_lock"]),
            ("promotion_authorization", governance["promotion_authorization"]),
        ):
            mounted = capability_governance.get(role)
            if not (
                isinstance(mounted, Mapping)
                and all(mounted.get(key) == canonical.get(key) for key in ("path", "sha256", "bytes"))
            ):
                raise discovery.CampaignError(
                    f"fixed capability {role} differs from canonical governance"
                )
        active = capability_governance.get("active_authorization")
        if not isinstance(active, Mapping):
            raise discovery.CampaignError(
                "fixed capability lacks active pretrain authorization"
            )
        governance["pretrain_authorization"] = {
            key: active[key] for key in ("path", "sha256", "bytes")
        }
        paths = _validate_fixed_runtime_paths(
            project_root,
            selection_lock=args.selection_lock,
            promotion_authorization=args.promotion_authorization,
            trainer=args.trainer,
            wrapper=args.gpu_wrapper,
            gpu_lock=args.gpu_lock,
            gpu_ledger=args.gpu_ledger,
            usage_ledger=args.usage_ledger,
        )
        python = discovery.executable_path_without_symlink_dereference(
            project_root,
            args.python if args.python is not None else Path(".venv/bin/python"),
        )
        for required in (python, paths["trainer"], paths["gpu_wrapper"]):
            if not required.is_file():
                raise discovery.CampaignError(
                    f"fixed runtime dependency is missing: {required}"
                )
        if not args.synthetic_preflight and (
            args.device != "cuda" or args.smoke_test
        ):
            raise discovery.CampaignError(
                "production fixed execution requires CUDA and the frozen full workload"
            )
        if args.aggregate:
            if args.sealed_pack_index is not None:
                raise discovery.CampaignError("pack-free aggregation rejects every pack capability")
            model_seals: list[tuple[Path, Mapping[str, Any]]] = []
            prediction_seals: list[tuple[Path, Mapping[str, Any]]] = []
            for fold in range(6):
                model_path = discovery._capability_bound_path(
                    project_root,
                    capability,
                    f"model_source_seal_outer{fold}",
                )
                prediction_path = discovery._capability_bound_path(
                    project_root,
                    capability,
                    f"prediction_shard_seal_outer{fold}",
                )
                model, _ = _read_frozen_json(
                    model_path, f"outer-{fold} model-source shard seal"
                )
                prediction, _ = _read_frozen_json(
                    prediction_path, f"outer-{fold} prediction shard seal"
                )
                model_seals.append((model_path, model))
                prediction_seals.append((prediction_path, prediction))
            seal = build_pack_free_promotion_exact_cover(
                run_root=run_root,
                model_source_seals=model_seals,
                prediction_seals=prediction_seals,
                selection=selection,
                governance=governance,
                usage_ledger=paths["gpu_usage_ledger"],
            )
            print(json.dumps(seal, sort_keys=True))
            return 0
        assert outer_fold is not None
        if args.sealed_pack_index is None:
            raise discovery.CampaignError("one shard requires exact pack-index and runtime capabilities")
        index_path = _under(project_root, args.sealed_pack_index)
        sealed_index = capability_document.get("sealed_pack_index")
        live_index_binding = discovery.bind_file(index_path)
        if not (
            isinstance(sealed_index, Mapping)
            and all(
                sealed_index.get(key) == live_index_binding.get(key)
                for key in ("path", "sha256", "bytes")
            )
        ):
            raise discovery.CampaignError(
                "fixed shard index differs from its target-sealed capability"
            )
        capability_path = _under(
            project_root, args.target_sealed_capability_receipt
        )
        if phase == "promotion_training":
            training, index_binding = load_target_scoped_promotion_training_pack(
                project_root=project_root,
                index_path=index_path,
                outer_fold=outer_fold,
                authorization_binding=governance["promotion_authorization"],
            )
            receipts = [
                _run_promotion_training(
                    run_root=run_root,
                    item=training[(outer_fold, seed)],
                    variant=str(selection["selected_variant"]),
                    selection=selection,
                    governance=governance,
                    target_sealed_capability_receipt=capability_path,
                    python=python,
                    trainer=paths["trainer"],
                    wrapper=paths["gpu_wrapper"],
                    promotion_authorization=paths["promotion_authorization"],
                    gpu_lock=paths["gpu_lock"],
                    gpu_ledger=paths["gpu_execution_ledger"],
                    usage_ledger=paths["gpu_usage_ledger"],
                    device=args.device,
                    amp=bool(args.amp),
                    smoke_test=bool(args.smoke_test),
                    command_runner=discovery._run_with_hard_timeout,
                )
                for seed in discovery.SEEDS
            ]
            seal = build_promotion_model_shard_completion(
                project_root=project_root,
                run_root=run_root,
                outer_fold=outer_fold,
                receipts=receipts,
                training=training,
                selection=selection,
                governance=governance,
                training_index_binding=index_binding,
                usage_ledger=paths["gpu_usage_ledger"],
                gpu_ledger=paths["gpu_execution_ledger"],
                gpu_lock=paths["gpu_lock"],
            )
        else:
            prediction, index_binding = load_target_scoped_prediction_pack(
                index_path=index_path,
                outer_fold=outer_fold,
                selection=selection,
                governance=governance,
            )
            receipts = []
            for seed in discovery.SEEDS:
                item, mounted = prediction[(outer_fold, seed)]
                receipts.append(
                    _run_prediction(
                        run_root=run_root,
                        item=item,
                        model_source=mounted.model_source,
                        predict_input=mounted.input_path,
                        input_receipt=mounted.model_source.receipt_path,
                        selection=selection,
                        governance=governance,
                        target_sealed_capability_receipt=capability_path,
                        python=python,
                        trainer=paths["trainer"],
                        wrapper=paths["gpu_wrapper"],
                        promotion_authorization=paths[
                            "promotion_authorization"
                        ],
                        gpu_lock=paths["gpu_lock"],
                        gpu_ledger=paths["gpu_execution_ledger"],
                        usage_ledger=paths["gpu_usage_ledger"],
                        device=args.device,
                        amp=bool(args.amp),
                        command_runner=discovery._run_with_hard_timeout,
                        project_root=project_root,
                    )
                )
            model_seal_path = index_path.parent / "MODEL_SOURCE_SHARD_SEAL.json"
            seal = build_prediction_shard_completion(
                run_root=run_root,
                outer_fold=outer_fold,
                receipts=receipts,
                selection=selection,
                governance=governance,
                prediction_index_binding=index_binding,
                model_source_shard_seal_binding=discovery.bind_file(
                    model_seal_path
                ),
                usage_ledger=paths["gpu_usage_ledger"],
                gpu_ledger=paths["gpu_execution_ledger"],
                gpu_lock=paths["gpu_lock"],
            )
        print(json.dumps(seal, sort_keys=True))
        return 0
    except discovery.CampaignError as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
