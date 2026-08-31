#!/usr/bin/env python3
"""Build causal radar range-frequency inputs and quality-controlled RR labels."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from snn_rr.data import (  # noqa: E402
    build_dataset_manifest,
    load_biopac_mat,
    load_xethru_recording,
)
from snn_rr.acquisition_contract import (  # noqa: E402
    ACQUISITION_COHORT_V1_CONTENT_SHA256,
    ACQUISITION_SCHEMA,
    ACQUISITION_SCHEMA_V3,
    ANNOTATION_ONLY_COLUMNS,
    AcquisitionSessionContract,
    assign_stage_window,
    load_acquisition_reconstruction,
    validate_raw_input_bindings,
)
from snn_rr.preprocess import (  # noqa: E402
    causal_block_mean,
    classical_rr_estimate,
    estimate_reference_window,
    filter_reference_rsp,
    fuse_auxiliary_features,
    identity_for_session,
    protocol_for_session,
    range_frequency_features,
    replace_radar_outliers_past_only,
    SESSION_IDENTITY,
)
from snn_rr.radar_timing import (  # noqa: E402
    CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1,
    CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1,
    canonical_ndarray_sha256,
    causal_uniform_resample_radar_views_v1,
)
from snn_rr.range_tracking import (  # noqa: E402
    RangeTrack,
    fuse_range_track_window_features,
)


FEATURE_CACHE_ACQUISITION_SCHEMA = "snn_rr.feature_cache_acquisition.v2"
FEATURE_CACHE_ACQUISITION_SCHEMA_V3 = "snn_rr.feature_cache_acquisition.v3"
FEATURE_CACHE_SESSION_SCHEMA_V3 = "snn_rr.feature_cache_session.v3"
FEATURE_CACHE_ROOT_SCHEMA_V3 = "snn_rr.feature_cache_root.v3"
V3_INFERENCE_FEATURE_SCHEMA_VERSION = "snn_rr.feature_cache_inference_features.v1"
V3_FEATURE_AVAILABILITY_SCHEMA_VERSION = (
    "snn_rr.feature_cache_inference_availability.v1"
)
V3_TARGET_FIREWALL_SCHEMA_VERSION = "snn_rr.feature_cache_target_firewall.v1"
V3_METADATA_JOIN_SCHEMA_VERSION = "snn_rr.feature_cache_metadata_join.v1"
V3_REFERENCE_SUPPORT_SCHEMA_VERSION = "snn_rr.feature_cache_reference_support.v1"
V3_ANNOTATION_ONLY_COLUMNS = tuple(
    (*ANNOTATION_ONLY_COLUMNS, "reference_mapping_available")
)
V3_MAP_RANGE_FEATURE_NAMES = tuple(
    [f"raw_power_pooled_range_{index:03d}" for index in range(91)]
    + [f"candidate_iq_phase_power_range_{index:03d}" for index in range(91)]
)
V3_MAP_SOURCE_LINEAGE = (
    "radar_only_causal_frequency_power_features_raw_and_candidate_iq_phase_"
    "concatenated_on_range_axis"
)
_RENAME_NOREPLACE = 1
_ACTIVE_V3_ATTEMPT: dict[str, Any] | None = None
REFERENCE_SAMPLE_NEAR_INTEGER_ATOL = 1.0e-9
REFERENCE_SAMPLE_NEAR_INTEGER_ULPS = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/default.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--subjects", nargs="*", help="optional IDs such as S02_RJS")
    parser.add_argument(
        "--acquisition-manifest",
        type=Path,
        help="opt-in reconstruction manifest; requires a new --cache-dir",
    )
    parser.add_argument(
        "--acquisition-mode",
        choices=("strict", "diagnostic"),
        default="strict",
        help="strict requires authorized scientific eligibility; diagnostic invalidates labels",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    return config


def replace_radar_outliers(values: np.ndarray, threshold: float = 0.1) -> tuple[np.ndarray, int]:
    """Replace corrupt samples from past values only.

    This runs before causal block averaging, so consulting the following frame
    would quietly leak future information into an otherwise online feature.
    """

    return replace_radar_outliers_past_only(values, threshold=threshold)


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_regular_file_payload(path: Path, *, label: str) -> tuple[bytes, str]:
    required = ("O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required):
        raise RuntimeError("version-3 cache production requires secure Linux open flags")
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(
        absolute,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a regular file with exactly one hard link")
        fingerprint = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        consumed = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            consumed += len(chunk)
        after = os.fstat(descriptor)
        rebound = os.stat(absolute, follow_symlinks=False)
        after_fingerprint = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        rebound_fingerprint = (
            rebound.st_dev,
            rebound.st_ino,
            rebound.st_mode,
            rebound.st_nlink,
            rebound.st_size,
            rebound.st_mtime_ns,
            rebound.st_ctime_ns,
        )
        if (
            after_fingerprint != fingerprint
            or rebound_fingerprint != fingerprint
            or consumed != before.st_size
        ):
            raise RuntimeError(f"{label} changed during exact-byte consumption")
        return b"".join(chunks), digest.hexdigest()
    finally:
        os.close(descriptor)


def _strict_json_payload(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {value}")

    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse {label}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


def _paths_overlap(first: Path, second: Path) -> bool:
    first_absolute = Path(os.path.abspath(os.fspath(first))).resolve()
    second_absolute = Path(os.path.abspath(os.fspath(second))).resolve()
    return bool(
        first_absolute == second_absolute
        or first_absolute in second_absolute.parents
        or second_absolute in first_absolute.parents
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _private_receipt_document(document: dict[str, Any]) -> dict[str, Any]:
    receipt = dict(document)
    receipt["content_sha256"] = ""
    receipt["content_sha256"] = _canonical_content_sha256(receipt)
    return receipt


def _write_private_json(path: Path, document: dict[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        ).encode("utf-8"),
    )
    os.chmod(path, 0o600, follow_symlinks=False)


def _start_v3_private_attempt(
    final_output_root: Path,
    *,
    acquisition_manifest: Path,
) -> Path:
    """Claim one fresh output name and create a private sibling staging tree."""

    global _ACTIVE_V3_ATTEMPT
    if _ACTIVE_V3_ATTEMPT is not None:
        raise RuntimeError("a version-3 feature-cache attempt is already active")
    final = Path(os.path.abspath(os.fspath(final_output_root)))
    parent = final.parent.resolve()
    final = parent / final.name
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.path.lexists(final):
        raise FileExistsError(
            f"version-3 feature-cache output already exists and cannot be resumed: {final}"
        )
    claim_path = parent / f".{final.name}.attempt_claim.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for required in ("O_NOFOLLOW", "O_CLOEXEC"):
        if not hasattr(os, required):
            raise RuntimeError("private version-3 attempts require secure Linux open flags")
        flags |= int(getattr(os, required))
    initial = _private_receipt_document(
        {
            "schema_version": "snn_rr.feature_cache_v3_attempt.v1",
            "terminal_state": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_output_root": str(final),
            "acquisition_manifest": str(acquisition_manifest.resolve()),
            "command": [str(item) for item in sys.argv],
            "selection_scope": "full_30_to_usable_29",
        }
    )
    descriptor = os.open(claim_path, flags, 0o600)
    # The exclusive claim now exists.  Register its generation before the first
    # fallible write so an I/O failure can be replaced by a terminal receipt
    # rather than leaving an unauditable, permanently blocking partial file.
    _ACTIVE_V3_ATTEMPT = {
        **initial,
        "claim_path": str(claim_path),
        "staging_root": None,
        "final_output_root": str(final),
        "completed_session_ids": [],
        "current_session_id": None,
    }
    try:
        try:
            payload = json.dumps(
                initial,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ).encode("utf-8")
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RuntimeError("short write for version-3 attempt claim")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        staging = Path(tempfile.mkdtemp(dir=parent, prefix=f".{final.name}.staging."))
        os.chmod(staging, 0o700)
        _ACTIVE_V3_ATTEMPT["staging_root"] = str(staging)
        staging_stat = os.lstat(staging)
        _ACTIVE_V3_ATTEMPT["staging_device"] = int(staging_stat.st_dev)
        _ACTIVE_V3_ATTEMPT["staging_inode"] = int(staging_stat.st_ino)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return staging
    except BaseException as error:
        try:
            _record_v3_attempt_failure(error)
        finally:
            _ACTIVE_V3_ATTEMPT = None
        raise


def _secure_and_fsync_v3_tree(root: Path) -> None:
    """Reject links, make the derived-data tree private, and durably flush it."""

    root = Path(root)
    if _ACTIVE_V3_ATTEMPT is not None and str(root) == _ACTIVE_V3_ATTEMPT.get(
        "staging_root"
    ):
        observed = os.lstat(root)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_dev != _ACTIVE_V3_ATTEMPT.get("staging_device")
            or observed.st_ino != _ACTIVE_V3_ATTEMPT.get("staging_inode")
        ):
            raise RuntimeError("version-3 staging root generation changed")
    for current_root, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        observed_root = os.lstat(current)
        if not stat.S_ISDIR(observed_root.st_mode):
            raise RuntimeError(f"version-3 staging entry is not a directory: {current}")
        os.chmod(current, 0o700, follow_symlinks=False)
        for name in directory_names:
            child = current / name
            observed = os.lstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                raise RuntimeError(
                    f"version-3 staging contains a linked/non-directory entry: {child}"
                )
        for name in file_names:
            child = current / name
            observed = os.lstat(child)
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise RuntimeError(
                    f"version-3 staging file must be regular with one link: {child}"
                )
            os.chmod(child, 0o600, follow_symlinks=False)
            descriptor = os.open(child, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                rebound = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(rebound.st_mode)
                    or rebound.st_nlink != 1
                    or (rebound.st_dev, rebound.st_ino)
                    != (observed.st_dev, observed.st_ino)
                ):
                    raise RuntimeError(
                        f"version-3 staging file changed before fsync: {child}"
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for current_root, _, _ in os.walk(root, topdown=False, followlinks=False):
        descriptor = os.open(
            current_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_v3_directory_noreplace(staging: Path, final: Path) -> None:
    staging_input = Path(os.path.abspath(os.fspath(staging)))
    staging = staging_input.parent.resolve() / staging_input.name
    final = Path(os.path.abspath(os.fspath(final)))
    parent = staging.parent
    if final.parent.resolve() != parent or "/" in final.name or final.name in {"", ".", ".."}:
        raise RuntimeError("version-3 staging/final roots must be sibling path leaves")
    before = os.lstat(staging)
    if not stat.S_ISDIR(before.st_mode) or (
        _ACTIVE_V3_ATTEMPT is not None
        and str(staging) == _ACTIVE_V3_ATTEMPT.get("staging_root")
        and (
            before.st_dev != _ACTIVE_V3_ATTEMPT.get("staging_device")
            or before.st_ino != _ACTIVE_V3_ATTEMPT.get("staging_inode")
        )
    ):
        raise RuntimeError("version-3 staging root changed before publication")
    if os.path.lexists(final):
        raise FileExistsError("version-3 final output exists before no-replace publication")
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    parent_fd = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        result = renameat2(
            parent_fd,
            os.fsencode(staging.name),
            parent_fd,
            os.fsencode(final.name),
            _RENAME_NOREPLACE,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(
                    "version-3 final output appeared during no-replace publication"
                )
            raise OSError(error_number, os.strerror(error_number))
        published = os.stat(final.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(published.st_mode)
            or (published.st_dev, published.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError("version-3 published directory inode mismatch")
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _record_v3_attempt_failure(error: BaseException) -> None:
    context = _ACTIVE_V3_ATTEMPT
    if context is None:
        return
    receipt = {
        key: value
        for key, value in context.items()
        if key not in {"claim_path"}
    }
    receipt.update(
        {
            "schema_version": "snn_rr.feature_cache_v3_failure.v1",
            "terminal_state": "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )
    receipt = _private_receipt_document(receipt)
    staging_value = context.get("staging_root")
    claim_value = context.get("claim_path")
    if isinstance(claim_value, str):
        _write_private_json(Path(claim_value), receipt)
    if isinstance(staging_value, str) and os.path.lexists(staging_value):
        staging = Path(staging_value)
        observed = os.lstat(staging)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_dev != context.get("staging_device")
            or observed.st_ino != context.get("staging_inode")
        ):
            raise RuntimeError(
                "version-3 staging root changed before failure receipt"
            )
        _write_private_json(staging / "FAILURE_RECEIPT.json", receipt)
        _secure_and_fsync_v3_tree(staging)


def _canonical_content_sha256(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _inventory_sha256(inventory: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            inventory,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _canonical_value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


_V3_AUX_SCALAR_NAMES = (
    "log_low_power",
    "log_delta_power",
    "log_delta_to_low_ratio",
    "log_median_range_variance",
    "log_q90_range_variance",
    "range_entropy",
    "q90_peak_probability",
    "q98_peak_probability",
)
_V3_AUX_CONSENSUS_NAMES = (
    "radar_peak_std_bpm",
    "radar_peak_range_bpm",
    "q90_correlation_radar_1_radar_2",
    "q90_correlation_radar_1_radar_3",
    "q90_correlation_radar_2_radar_3",
)
_V3_TARGET_DERIVED_METADATA_COLUMNS = frozenset(
    {
        "rr_bpm",
        "rr_spectral_bpm",
        "rr_phase_bpm",
        "rr_events_bpm",
        "reference_valid",
        "reference_quality",
        "reference_sigma_bpm",
        "spectral_concentration",
        "periodicity",
        "interval_cv",
        "estimator_disagreement_bpm",
        "phase_residual_rad",
        "clip_fraction",
        "guard_clip_fraction",
        "plateau_fraction",
        "breath_count",
        "classical_error_bpm",
        "radar_observable",
        "classical_acceptable_within_2bpm",
    }
)
_V3_METADATA_JOIN_COLUMNS = (
    "session_id",
    "identity",
    "window_number",
    "window_start_s",
    "window_end_s",
    "reference_start_sample",
    "reference_end_sample",
    "reference_window_start_biopac_s",
    "reference_window_end_biopac_s",
    "radar_window_start_relative_s",
    "radar_window_end_relative_s",
    "reference_mapping_available",
    "sync_authorized",
    "alignment_scientific_eligible",
    "acquisition_phase",
    "acquisition_phase_name",
    "acquisition_phase_status",
    "acquisition_phase_confidence",
    "phase_overlap_fraction",
    "transition_window",
    "eligible_for_stage_metrics",
    "phase7_assignment",
    "acquisition_batch",
)
_V3_REFERENCE_FLOAT_NAN_COLUMNS = tuple(
    sorted(
        (_V3_TARGET_DERIVED_METADATA_COLUMNS - {
            "reference_valid",
            "radar_observable",
            "classical_acceptable_within_2bpm",
        })
        | {
            "window_start_s",
            "window_end_s",
            "reference_window_start_biopac_s",
            "reference_window_end_biopac_s",
            "sync_confidence",
            "acquisition_phase_confidence",
            "phase_overlap_fraction",
        }
    )
)
_V3_REFERENCE_INTEGER_MINUS_ONE_COLUMNS = (
    "reference_start_sample",
    "reference_end_sample",
)
_V3_REFERENCE_NULL_STRING_COLUMNS = (
    "acquisition_phase",
    "acquisition_phase_name",
    "acquisition_phase_status",
    "phase7_assignment",
)
_V3_REFERENCE_FALSE_BOOLEAN_COLUMNS = (
    "reference_mapping_available",
    "reference_valid",
    "radar_observable",
    "classical_acceptable_within_2bpm",
    "sync_authorized",
    "alignment_scientific_eligible",
    "transition_window",
    "eligible_for_stage_metrics",
)


def _is_v3_acquisition_contract(
    contract: AcquisitionSessionContract | None,
) -> bool:
    manifest = None if contract is None else getattr(contract, "manifest", None)
    return bool(
        isinstance(manifest, dict)
        and manifest.get("schema_version") == ACQUISITION_SCHEMA_V3
    )


def _v3_aux_feature_names(frequency_bins: int) -> tuple[str, ...]:
    if isinstance(frequency_bins, bool) or not isinstance(
        frequency_bins, (int, np.integer)
    ) or int(frequency_bins) < 1:
        raise ValueError("version-3 auxiliary frequency count must be positive")
    count = int(frequency_bins)
    names: list[str] = []
    for radar_id in (1, 2, 3):
        for quantile in ("q90", "q98"):
            names.extend(
                f"radar_{radar_id}_{quantile}_frequency_{index:04d}"
                for index in range(count)
            )
    for radar_id in (1, 2, 3):
        names.extend(
            f"radar_{radar_id}_{name}" for name in _V3_AUX_SCALAR_NAMES
        )
    names.extend(f"fused_median_frequency_{index:04d}" for index in range(count))
    names.extend(f"fused_max_frequency_{index:04d}" for index in range(count))
    names.extend(_V3_AUX_CONSENSUS_NAMES)
    if len(names) != 8 * count + 29 or len(set(names)) != len(names):
        raise RuntimeError("version-3 auxiliary feature-name derivation failed")
    return tuple(names)


def _v3_feature_availability(
    timing_valid: np.ndarray,
    *,
    auxiliary_frequency_bins: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    timing = np.asarray(timing_valid)
    if timing.dtype != np.bool_ or timing.ndim != 3 or timing.shape[1:] != (3, 320):
        raise ValueError("version-3 timing mask must have exact [window,3,320] support")
    map_available = np.all(timing, axis=2)
    frequency_bins = int(auxiliary_frequency_bins)
    aux_names = _v3_aux_feature_names(frequency_bins)
    aux_available = np.zeros((len(timing), len(aux_names)), dtype=np.bool_)
    cursor = 0
    for radar_index in range(3):
        width = 2 * frequency_bins
        aux_available[:, cursor : cursor + width] = map_available[
            :, radar_index, None
        ]
        cursor += width
    for radar_index in range(3):
        width = len(_V3_AUX_SCALAR_NAMES)
        aux_available[:, cursor : cursor + width] = map_available[
            :, radar_index, None
        ]
        cursor += width
    joint = np.all(map_available, axis=1)
    joint_width = 2 * frequency_bins + len(_V3_AUX_CONSENSUS_NAMES)
    aux_available[:, cursor : cursor + joint_width] = joint[:, None]
    cursor += joint_width
    if cursor != len(aux_names):
        raise RuntimeError("version-3 auxiliary availability layout drifted")
    availability_names = tuple(
        [f"map_radar_{radar_id}" for radar_id in (1, 2, 3)]
        + [f"aux:{name}" for name in aux_names]
    )
    return map_available, aux_available, aux_names, availability_names


def _pool_range_frequency_map(raw_maps: np.ndarray) -> np.ndarray:
    """Pool adjacent frequency rows without inventing a feature axis.

    ``range_frequency_features`` already concatenates its 91 raw-power and 91
    candidate-I/Q phase-power coordinates on the final range-feature axis.
    The cache transform only pools adjacent frequency coordinates, therefore
    the truthful output layout is ``[radar, frequency, 182]``.
    """

    values = np.asarray(raw_maps)
    if values.ndim != 3 or values.shape[0] != 3 or values.shape[2] != 182:
        raise ValueError(
            "range-frequency maps must have exact [3,frequency,182] geometry"
        )
    usable_frequencies = values.shape[1] - values.shape[1] % 2
    if usable_frequencies < 2:
        raise ValueError("range-frequency maps require at least two frequency rows")
    return (
        values[:, :usable_frequencies]
        .reshape(3, usable_frequencies // 2, 2, 182)
        .mean(axis=2, dtype=np.float32)
        .astype(np.float16)
    )


def _v3_metadata_scalar(value: Any, *, label: str) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{label} contains a non-finite value")
        return result
    if isinstance(value, str):
        return value
    raise ValueError(f"{label} contains a non-canonical scalar")


def _v3_metadata_join_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    missing = sorted(set(_V3_METADATA_JOIN_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"version-3 metadata join columns are missing: {missing}")
    records: list[dict[str, Any]] = []
    for row_index, row in frame.loc[:, _V3_METADATA_JOIN_COLUMNS].iterrows():
        records.append(
            {
                column: _v3_metadata_scalar(
                    row[column], label=f"version-3 metadata row {row_index}/{column}"
                )
                for column in _V3_METADATA_JOIN_COLUMNS
            }
        )
    return records


def _v3_reference_support_contract(reference_mapping_available: bool) -> dict[str, Any]:
    if type(reference_mapping_available) is not bool:
        raise ValueError("version-3 reference mapping availability must be boolean")
    return {
        "schema_version": V3_REFERENCE_SUPPORT_SCHEMA_VERSION,
        "reference_mapping_available": reference_mapping_available,
        "radar_support_columns": [
            "radar_window_start_relative_s",
            "radar_window_end_relative_s",
        ],
        "unavailable_float_sentinel": "nan",
        "unavailable_float_columns": list(_V3_REFERENCE_FLOAT_NAN_COLUMNS),
        "unavailable_integer_sentinel": -1,
        "unavailable_integer_columns": list(
            _V3_REFERENCE_INTEGER_MINUS_ONE_COLUMNS
        ),
        "unavailable_string_sentinel": None,
        "unavailable_string_columns": list(_V3_REFERENCE_NULL_STRING_COLUMNS),
        "unavailable_boolean_sentinel": False,
        "unavailable_boolean_columns": list(_V3_REFERENCE_FALSE_BOOLEAN_COLUMNS),
        "numeric_zero_never_implies_reference_availability": True,
    }


def _validate_v3_unmapped_reference_rows(frame: pd.DataFrame, *, session_id: str) -> None:
    missing = sorted(
        set(
            _V3_REFERENCE_FLOAT_NAN_COLUMNS
            + _V3_REFERENCE_INTEGER_MINUS_ONE_COLUMNS
            + _V3_REFERENCE_NULL_STRING_COLUMNS
            + _V3_REFERENCE_FALSE_BOOLEAN_COLUMNS
            + (
                "radar_window_start_relative_s",
                "radar_window_end_relative_s",
            )
        )
        - set(frame.columns)
    )
    if missing:
        raise ValueError(
            f"{session_id} unmapped reference sentinel columns are missing: {missing}"
        )
    for column in _V3_REFERENCE_FLOAT_NAN_COLUMNS:
        values = frame[column]
        if not pd.api.types.is_float_dtype(values.dtype) or not values.isna().all():
            raise ValueError(
                f"{session_id} unmapped reference column {column} must be all NaN"
            )
    for column in _V3_REFERENCE_INTEGER_MINUS_ONE_COLUMNS:
        values = frame[column]
        if (
            not pd.api.types.is_integer_dtype(values.dtype)
            or not np.array_equal(values.to_numpy(), np.full(len(frame), -1))
        ):
            raise ValueError(
                f"{session_id} unmapped reference column {column} must be exact -1"
            )
    for column in _V3_REFERENCE_NULL_STRING_COLUMNS:
        values = frame[column]
        if not values.isna().all():
            raise ValueError(
                f"{session_id} unmapped reference column {column} must be null"
            )
    for column in _V3_REFERENCE_FALSE_BOOLEAN_COLUMNS:
        if any(type(value) not in {bool, np.bool_} or bool(value) for value in frame[column]):
            raise ValueError(
                f"{session_id} unmapped reference column {column} must be false"
            )
    try:
        radar_start = pd.to_numeric(
            frame["radar_window_start_relative_s"], errors="raise"
        ).to_numpy(dtype=float)
        radar_end = pd.to_numeric(
            frame["radar_window_end_relative_s"], errors="raise"
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{session_id} unmapped measured radar support is invalid"
        ) from error
    if not (
        np.isfinite(radar_start).all()
        and np.isfinite(radar_end).all()
        and np.allclose(radar_end - radar_start, 32.0, rtol=0.0, atol=1e-12)
        and (len(frame) < 2 or np.all(np.diff(radar_start) > 0))
        and (len(frame) < 2 or np.all(np.diff(radar_end) > 0))
        and len(np.unique(np.column_stack((radar_start, radar_end)), axis=0))
        == len(frame)
    ):
        raise ValueError(f"{session_id} unmapped measured radar support is invalid")


def _validate_v3_feature_config(config: dict[str, Any]) -> None:
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("version-3 feature cache requires a data configuration")
    model_hz = data.get("model_hz")
    window_seconds = data.get("window_seconds")
    if (
        isinstance(model_hz, bool)
        or not isinstance(model_hz, (int, float))
        or float(model_hz) != 10.0
        or isinstance(window_seconds, bool)
        or not isinstance(window_seconds, (int, float))
        or float(window_seconds) != 32.0
        or int(round(float(model_hz) * float(window_seconds))) != 320
    ):
        raise ValueError(
            "version-3 feature cache requires exact 32 s x 10 Hz = 320 timing support"
        )


def _v3_directory_identity(observed: os.stat_result) -> dict[str, int]:
    return {
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mode": int(observed.st_mode),
        "link_count": int(observed.st_nlink),
        "uid": int(observed.st_uid),
        "gid": int(observed.st_gid),
        "bytes": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
    }


def _validate_v3_dataset_catalogue(
    dataset_root: Path,
    *,
    manifest: Any | None = None,
) -> dict[str, Any]:
    """Bind the real top-level 30-session directory generation.

    The general manifest builder intentionally synthesizes missing numbered
    subjects and ignores out-of-range folders.  That compatibility behavior is
    unsuitable for the V3 full-cohort claim, so this gate independently pins
    the root directory and requires the exact canonical session leaves.
    """

    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required_flags):
        raise RuntimeError("version-3 dataset catalogue requires secure Linux flags")
    absolute = Path(os.path.abspath(os.fspath(dataset_root)))
    descriptor = os.open(
        absolute,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError("version-3 dataset root must be a real directory")
        expected = tuple(SESSION_IDENTITY)
        session_like_names = tuple(
            sorted(
                name
                for name in os.listdir(descriptor)
                if re.fullmatch(r"S[0-9]{2}(?:_.*)?", name) is not None
            )
        )
        if session_like_names != expected:
            missing = sorted(set(expected) - set(session_like_names))
            extra = sorted(set(session_like_names) - set(expected))
            raise ValueError(
                "version-3 dataset root must contain the exact canonical 30 "
                f"session folders; missing={missing}, extra={extra}"
            )
        session_entries: list[dict[str, Any]] = []
        for name in expected:
            observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(observed.st_mode):
                raise ValueError(
                    f"version-3 dataset session root is not a real directory: {name}"
                )
            session_entries.append(
                {"session_id": name, "identity": _v3_directory_identity(observed)}
            )
        after = os.fstat(descriptor)
        if _v3_directory_identity(after) != _v3_directory_identity(before):
            raise RuntimeError("version-3 dataset root changed during catalogue binding")
    finally:
        os.close(descriptor)
    if manifest is not None:
        subjects = getattr(manifest, "subjects", None)
        if not isinstance(subjects, tuple) or tuple(
            str(getattr(subject, "subject_id", "")) for subject in subjects
        ) != tuple(SESSION_IDENTITY):
            raise ValueError(
                "version-3 discovered manifest does not preserve the exact canonical "
                "30-session order"
            )
    return {
        "schema_version": "snn_rr.feature_cache_dataset_catalogue.v1",
        "dataset_root": str(absolute),
        "root_identity": _v3_directory_identity(before),
        "session_entries": session_entries,
    }


def _validate_v3_reconstruction_projection(acquisition: Any) -> tuple[str, ...]:
    manifest = acquisition.manifest
    expected_all = tuple(str(item) for item in manifest.get("expected_session_ids", []))
    expected_usable = tuple(
        str(item) for item in manifest.get("expected_usable_session_ids", [])
    )
    selected = tuple(str(item) for item in manifest.get("selected_session_ids", []))
    raw_sessions = manifest.get("sessions")
    if not isinstance(raw_sessions, list) or any(
        not isinstance(item, dict) for item in raw_sessions
    ):
        raise ValueError("version-3 reconstruction sessions must be an exact catalogue")
    catalogue = tuple(str(item.get("session_id")) for item in raw_sessions)
    usable_projection = tuple(
        str(item.get("session_id"))
        for item in raw_sessions
        if item.get("usable") is True
    )
    excluded = tuple(item for item in expected_all if item not in usable_projection)
    loaded_usable = tuple(str(item) for item in acquisition.sessions)
    canonical_all = tuple(SESSION_IDENTITY)
    canonical_usable = tuple(
        session_id for session_id in canonical_all if session_id != "S24_KHJ"
    )
    if (
        manifest.get("schema_version") != ACQUISITION_SCHEMA_V3
        or expected_all != canonical_all
        or expected_usable != canonical_usable
        or selected != expected_all
        or manifest.get("expected_session_ids_sha256")
        != _canonical_value_sha256(list(expected_all))
        or manifest.get("expected_usable_session_ids_sha256")
        != _canonical_value_sha256(list(expected_usable))
        or manifest.get("selected_session_ids_sha256")
        != _canonical_value_sha256(list(selected))
        or catalogue != expected_all
        or usable_projection != expected_usable
        or loaded_usable != expected_usable
        or excluded != ("S24_KHJ",)
        or manifest.get("dataset_session_count") != 30
        or manifest.get("dataset_usable_session_count") != 29
        or manifest.get("dataset_physical_identity_count") != 18
        or manifest.get("selected_session_count") != 30
        or manifest.get("session_count") != 30
        or manifest.get("cohort_authority_content_sha256")
        != ACQUISITION_COHORT_V1_CONTENT_SHA256
        or manifest.get("subjects_filter_applied") is not False
        or manifest.get("selection_scope") != "full_cohort"
        or manifest.get("execution_complete") is not True
        or manifest.get("full_cohort_complete") is not True
        or manifest.get("complete") is not True
        or manifest.get("raw_consumed_bytes_verified") is not True
        or manifest.get("timing_adjudicated") is not True
        or acquisition.full_cohort_complete is not True
        or acquisition.scientific_eligible is not False
        or manifest.get("scientific_eligible") is not False
        or manifest.get("sync_raw_replay_verified") is not False
        or manifest.get("protocol_raw_replay_verified") is not False
    ):
        raise ValueError(
            "version-3 diagnostic feature cache requires the exact full 30-session "
            "reconstruction and its exact 29-session usable projection"
        )
    return expected_usable


def _derive_acquisition_cache_scope(
    *,
    subjects_filter_applied: bool,
    reconstruction_full_cohort_complete: bool,
    expected_usable_session_ids: tuple[str, ...],
    cache_usable_session_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Derive publishable cache scope from selection intent and coverage.

    Exact ID coverage is necessary but not sufficient for a full-cohort claim:
    an explicit ``--subjects`` invocation is targeted even when the caller
    happens to enumerate every expected session.
    """

    if type(subjects_filter_applied) is not bool:
        raise ValueError("subjects_filter_applied must be an explicit boolean")
    full_cohort_complete = bool(
        not subjects_filter_applied
        and reconstruction_full_cohort_complete
        and cache_usable_session_ids == expected_usable_session_ids
    )
    return {
        "subjects_filter_applied": subjects_filter_applied,
        "selection_scope": (
            "full_cohort" if full_cohort_complete else "diagnostic_subset"
        ),
        "full_cohort_complete": full_cohort_complete,
    }


def _half_open_sample_bounds(
    start_s: float,
    end_s: float,
    sample_rate_hz: float,
) -> tuple[int, int]:
    """Map continuous ``[start_s, end_s)`` support to exact sample indices.

    A sample ``i`` is included exactly when ``start_s <= i / fs < end_s``;
    therefore both slice boundaries use ``ceil``. Arithmetic that should land
    on an integer sample coordinate can acquire a few ULPs through affine clock
    mapping, so only coordinates indistinguishable from an integer at the
    declared numerical tolerance are canonicalized before applying ``ceil``.
    """

    start = float(start_s)
    end = float(end_s)
    rate = float(sample_rate_hz)
    if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
        raise ValueError("half-open sample support must be finite with end > start")
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("sample rate must be finite and positive")

    canonical_coordinates: list[float] = []
    for value in (start, end):
        coordinate = value * rate
        if not math.isfinite(coordinate):
            raise ValueError("sample coordinate is non-finite")
        nearest = float(np.rint(coordinate))
        tolerance = max(
            REFERENCE_SAMPLE_NEAR_INTEGER_ATOL,
            REFERENCE_SAMPLE_NEAR_INTEGER_ULPS
            * abs(float(np.spacing(max(abs(coordinate), 1.0)))),
        )
        canonical_coordinates.append(
            nearest if abs(coordinate - nearest) <= tolerance else coordinate
        )

    first, stop = (math.ceil(value) for value in canonical_coordinates)
    for coordinate, index in zip(canonical_coordinates, (first, stop), strict=True):
        if not index - 1 < coordinate <= index:
            raise RuntimeError("half-open sample-boundary proof failed")
    return first, stop


def _pipeline_paths() -> list[Path]:
    return [
        Path(__file__),
        SOURCE_ROOT / "snn_rr" / "data.py",
        SOURCE_ROOT / "snn_rr" / "raw_snapshot.py",
        SOURCE_ROOT / "snn_rr" / "preprocess.py",
        SOURCE_ROOT / "snn_rr" / "acquisition_contract.py",
        SOURCE_ROOT / "snn_rr" / "acquisition_protocol.py",
        SOURCE_ROOT / "snn_rr" / "synchronization.py",
        SOURCE_ROOT / "snn_rr" / "radar_timing.py",
        SOURCE_ROOT / "snn_rr" / "range_tracking.py",
    ]


def _array_inventory(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _metadata_inventory(path: Path, metadata: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": [len(metadata), len(metadata.columns)],
        "dtype": "csv",
    }


def _inventory_files_match(root: Path, inventory: Any) -> bool:
    if not isinstance(inventory, dict) or not inventory:
        return False
    for entry in inventory.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return False
        candidate = (root / entry["path"]).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return False
        if not candidate.is_file():
            return False
        if entry.get("bytes") != candidate.stat().st_size:
            return False
        if entry.get("sha256") != _sha256_file(candidate):
            return False
    return True


def _acquisition_cached_manifest_is_current(
    root: Path,
    manifest: dict[str, Any],
    *,
    require_range_aux: bool,
    is_v3: bool = False,
) -> bool:
    declared_content = manifest.get("content_sha256")
    if not isinstance(declared_content, str) or (
        declared_content != _canonical_content_sha256(manifest)
    ):
        return False
    inventory = manifest.get("file_inventory")
    required = {
        "maps",
        "aux",
        "metadata",
        "frequencies_hz",
        "radar_timing_valid_mask",
    }
    if is_v3:
        required.update(
            {
                "radar_timing_invalid_reason_mask",
                "feature_availability_mask",
            }
        )
    if require_range_aux:
        required.add("range_aux")
    if not isinstance(inventory, dict) or set(inventory) != required:
        return False
    if manifest.get("inventory_sha256") != _inventory_sha256(inventory):
        return False
    return _inventory_files_match(root, inventory)


def _source_fingerprint(subject: Any) -> str:
    """Cheap, deterministic invalidation key for the selected raw sources."""

    paths = [Path(subject.biopac_path)]
    if subject.selected_session is not None:
        for radar_id in (1, 2, 3):
            stream = subject.selected_session.radars[radar_id]
            paths.extend(map(Path, stream.data_paths))
            if stream.meta_path is not None:
                paths.append(Path(stream.meta_path))
    records = []
    for path in sorted(set(paths), key=lambda item: str(item.resolve())):
        stat = path.stat()
        records.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_range_track_contract(
    contract: AcquisitionSessionContract,
    expected_times_s: np.ndarray,
) -> tuple[tuple[RangeTrack, ...], tuple[str, ...]]:
    if contract.range_track_path is None:
        raise ValueError(
            f"{contract.session_id} acquisition contract has no bound range-track artifact"
        )
    with np.load(contract.range_track_path, allow_pickle=False) as archive:
        observed_times = np.asarray(archive["radar_times_s"], dtype=np.float64)
        if observed_times.shape != expected_times_s.shape or not np.allclose(
            observed_times, expected_times_s, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                f"{contract.session_id} range-track timeline does not match feature timing"
            )
        tracks: list[RangeTrack] = []
        for radar_id in (1, 2, 3):
            prefix = f"radar{radar_id}_split_halves"
            tracks.append(
                RangeTrack(
                    layout="split_halves",
                    bin_index=np.asarray(archive[f"{prefix}_bin_index"]),
                    confidence=np.asarray(archive[f"{prefix}_confidence"]),
                    normalized_entropy=np.asarray(
                        archive[f"{prefix}_normalized_entropy"]
                    ),
                    missing=np.asarray(archive[f"{prefix}_missing"]),
                    multimodal=np.asarray(archive[f"{prefix}_multimodal"]),
                    evidence_strength=np.asarray(
                        archive[f"{prefix}_evidence_strength"]
                    ),
                    bin_count=91,
                    sample_rate_hz=10.0,
                )
            )
    _, names = fuse_range_track_window_features(tracks, 0, min(1, len(expected_times_s)))
    return tuple(tracks), names


def _session_acquisition_provenance(
    contract: AcquisitionSessionContract,
    *,
    mode: str,
) -> dict[str, Any]:
    range_document = contract.manifest.get("range_tracking", {})
    protocol = contract.protocol
    return {
        "schema_version": FEATURE_CACHE_ACQUISITION_SCHEMA,
        "acquisition_session_manifest_sha256": contract.content_sha256,
        "sync_receipt_content_sha256": contract.receipt_content_sha256,
        "mapping_sha256": contract.mapping_sha256,
        "manual_approval_content_sha256": (
            None
            if contract.manual_approval is None
            else contract.manual_approval.get("content_sha256")
        ),
        "protocol_annotation_schema_version": protocol.get(
            "annotation_schema_version"
        ),
        "range_artifact_sha256": (
            range_document.get("artifact_sha256")
            if isinstance(range_document, dict)
            else None
        ),
        "reference_alignment_mode": (
            "authorized_marker_affine_v1"
            if mode == "strict"
            else "diagnostic_unapproved_proposal_v1"
        ),
        "scientific_eligible": bool(mode == "strict" and contract.scientific_eligible),
        "measured_timing_eligible": contract.measured_timing_eligible,
        "alignment_eligible": contract.alignment_eligible,
        "stage_metric_eligible": contract.stage_metric_eligible,
        "range_feature_eligible": contract.range_feature_eligible,
        "strict_cache_eligible": contract.strict_cache_eligible,
        "annotation_only_columns": list(ANNOTATION_ONLY_COLUMNS),
    }


def _build_v3_session_manifest(
    *,
    session_id: str,
    contract: AcquisitionSessionContract,
    config_sha256: str,
    pipeline_sha256: str,
    source_fingerprint: str,
    maps: np.ndarray,
    aux: np.ndarray,
    frequencies_hz: np.ndarray,
    timing_valid_mask: np.ndarray,
    timing_reason_mask: np.ndarray,
    feature_availability_mask: np.ndarray,
    aux_feature_names: tuple[str, ...],
    availability_feature_names: tuple[str, ...],
    metadata: pd.DataFrame,
    file_inventory: dict[str, Any],
    biopac_sample_rate_hz: float | None,
) -> dict[str, Any]:
    if (
        contract.manifest.get("schema_version") != ACQUISITION_SCHEMA_V3
        or contract.authorized
        or contract.scientific_eligible
        or contract.stage_metric_eligible
        or contract.manifest.get("raw_consumed_bytes_verified") is not True
        or contract.manifest.get("timing_adjudicated") is not True
        or contract.manifest.get("sync_raw_replay_verified") is not False
        or contract.manifest.get("protocol_raw_replay_verified") is not False
    ):
        raise ValueError(f"{session_id} is not a valid diagnostic acquisition-v3 input")
    receipt_result = contract.receipt.get("result")
    if not isinstance(receipt_result, dict):
        raise ValueError(f"{session_id} version-3 synchronization result is malformed")
    mapping_raw = receipt_result.get("mapping")
    if mapping_raw is not None and not isinstance(mapping_raw, dict):
        raise ValueError(f"{session_id} version-3 diagnostic mapping is malformed")
    reference_mapping_available = isinstance(mapping_raw, dict)
    if reference_mapping_available != (contract.mapping is not None):
        raise ValueError(f"{session_id} upstream mapping availability is inconsistent")
    if reference_mapping_available:
        if biopac_sample_rate_hz is None or float(biopac_sample_rate_hz) != 250.0:
            raise ValueError("version-3 mapped metadata requires exact 250 Hz BIOPAC support")
    elif biopac_sample_rate_hz is not None:
        raise ValueError("version-3 unmapped radar-only metadata cannot claim BIOPAC support")
    mapping = json.loads(json.dumps(mapping_raw, allow_nan=False))
    protocol = json.loads(json.dumps(dict(contract.protocol), allow_nan=False))
    upstream_session_contract = json.loads(
        json.dumps(dict(contract.manifest), allow_nan=False)
    )
    upstream_sync_receipt = json.loads(
        json.dumps(dict(contract.receipt), allow_nan=False)
    )
    upstream_synchronization = upstream_session_contract.get("synchronization")
    if (
        _canonical_content_sha256(upstream_session_contract)
        != contract.content_sha256
        or _canonical_content_sha256(upstream_sync_receipt)
        != contract.receipt_content_sha256
        or not isinstance(upstream_synchronization, dict)
        or upstream_synchronization.get("receipt_content_sha256")
        != contract.receipt_content_sha256
        or upstream_synchronization.get("mapping") != mapping
        or not isinstance(upstream_sync_receipt.get("result"), dict)
        or upstream_sync_receipt["result"].get("mapping") != mapping
        or upstream_session_contract.get("protocol") != protocol
    ):
        raise ValueError(f"{session_id} upstream reference authority binding is inconsistent")
    consumed_metadata = metadata.copy(deep=True)
    if (
        maps.dtype != np.float16
        or aux.dtype != np.float32
        or frequencies_hz.dtype != np.float64
        or timing_valid_mask.dtype != np.bool_
        or timing_reason_mask.dtype != np.uint8
        or feature_availability_mask.dtype != np.bool_
        or timing_valid_mask.shape[1:] != (3, 320)
        or not np.array_equal(timing_reason_mask != 0, ~timing_valid_mask)
    ):
        raise ValueError(f"{session_id} version-3 payload dtypes/support are invalid")
    if (
        maps.ndim != 4
        or maps.shape[1] != 3
        or maps.shape[3] != 182
        or maps.shape[2] != len(frequencies_hz)
        or aux.ndim != 2
        or len(maps) != len(aux)
        or len(maps) != len(consumed_metadata)
        or len(maps) != len(timing_valid_mask)
        or len(maps) != len(feature_availability_mask)
        or not np.isfinite(frequencies_hz).all()
        or not (np.diff(frequencies_hz) > 0).all()
        or not np.isfinite(maps).all()
        or not np.isfinite(aux).all()
        or aux.shape[1] < 37
        or (aux.shape[1] - 29) % 8
    ):
        raise ValueError(f"{session_id} version-3 feature geometry is invalid")
    auxiliary_frequency_bins = (aux.shape[1] - 29) // 8
    if auxiliary_frequency_bins not in {2 * maps.shape[2], 2 * maps.shape[2] + 1}:
        raise ValueError(f"{session_id} version-3 map/aux frequency geometry differs")
    (
        expected_map_available,
        expected_aux_available,
        expected_aux_names,
        expected_availability_names,
    ) = _v3_feature_availability(
        timing_valid_mask,
        auxiliary_frequency_bins=auxiliary_frequency_bins,
    )
    expected_feature_availability = np.concatenate(
        (expected_map_available, expected_aux_available), axis=1
    )
    if (
        tuple(aux_feature_names) != expected_aux_names
        or tuple(availability_feature_names) != expected_availability_names
        or not np.array_equal(
            feature_availability_mask, expected_feature_availability
        )
        or np.any(maps[~expected_map_available] != 0)
        or np.any(np.signbit(maps[~expected_map_available]))
        or np.any(aux[~expected_aux_available] != 0)
        or np.any(np.signbit(aux[~expected_aux_available]))
    ):
        raise ValueError(
            f"{session_id} version-3 explicit feature availability is inconsistent"
        )
    missing_target_metadata = sorted(
        _V3_TARGET_DERIVED_METADATA_COLUMNS - set(consumed_metadata.columns)
    )
    if missing_target_metadata:
        raise ValueError(
            f"{session_id} version-3 target-firewall metadata is incomplete: "
            f"{missing_target_metadata}"
        )
    reference_valid = consumed_metadata["reference_valid"].to_numpy()
    if any(type(value) not in {bool, np.bool_} or bool(value) for value in reference_valid):
        raise ValueError(
            f"{session_id} diagnostic version-3 rows must have reference_valid=false"
        )
    mapping_availability = consumed_metadata.get("reference_mapping_available")
    if mapping_availability is None or any(
        type(value) not in {bool, np.bool_}
        or bool(value) != reference_mapping_available
        for value in mapping_availability
    ):
        raise ValueError(
            f"{session_id} reference mapping availability rows are inconsistent"
        )
    if not reference_mapping_available:
        _validate_v3_unmapped_reference_rows(
            consumed_metadata, session_id=session_id
        )
    feature_schema = {
        "schema_version": V3_INFERENCE_FEATURE_SCHEMA_VERSION,
        "maps": {
            "axes": ["window", "radar_view", "frequency", "range_feature"],
            "radar_names": ["radar_1", "radar_2", "radar_3"],
            "range_feature_names": list(V3_MAP_RANGE_FEATURE_NAMES),
            "shape": list(maps.shape),
            "dtype": "float16",
            "frequency_grid_sha256": canonical_ndarray_sha256(frequencies_hz),
            "source_lineage": V3_MAP_SOURCE_LINEAGE,
            "target_derived_inputs": False,
        },
        "aux": {
            "axes": ["window", "feature"],
            "feature_names": list(aux_feature_names),
            "feature_names_sha256": _canonical_value_sha256(
                list(aux_feature_names)
            ),
            "shape": list(aux.shape),
            "dtype": "float32",
            "source_lineage": "radar_only_causal_auxiliary_spectra_statistics",
            "target_derived_inputs": False,
        },
        "availability": {
            "schema_version": V3_FEATURE_AVAILABILITY_SCHEMA_VERSION,
            "axes": ["window", "feature"],
            "feature_names": list(availability_feature_names),
            "feature_names_sha256": _canonical_value_sha256(
                list(availability_feature_names)
            ),
            "shape": list(feature_availability_mask.shape),
            "dtype": "bool",
            "semantics": (
                "first three cells authorize complete map radar views; remaining "
                "cells explicitly authorize aux columns; numeric zero never implies availability"
            ),
        },
    }
    forbidden = sorted(
        _V3_TARGET_DERIVED_METADATA_COLUMNS | set(V3_ANNOTATION_ONLY_COLUMNS)
    )
    target_firewall = {
        "schema_version": V3_TARGET_FIREWALL_SCHEMA_VERSION,
        "inference_payloads": [
            "maps",
            "aux",
            "feature_availability_mask",
            "frequencies_hz",
        ],
        "target_derived_metadata_columns": sorted(
            _V3_TARGET_DERIVED_METADATA_COLUMNS
        ),
        "annotation_only_columns": sorted(V3_ANNOTATION_ONLY_COLUMNS),
        "forbidden_inference_feature_names": forbidden,
        "radar_observable_role": (
            "target_derived_metadata_only_forbidden_at_inference"
        ),
        "target_values_used_in_inference_features": False,
    }
    if (set(aux_feature_names) | set(availability_feature_names)).intersection(
        forbidden
    ):
        raise RuntimeError("version-3 inference feature names cross the target firewall")
    metadata_join_contract = {
        "schema_version": V3_METADATA_JOIN_SCHEMA_VERSION,
        "reference_mapping_available": reference_mapping_available,
        "mapping": mapping,
        "mapping_sha256": _canonical_value_sha256(mapping),
        "protocol": protocol,
        "protocol_sha256": _canonical_value_sha256(protocol),
        "biopac_sample_rate_hz": (
            250.0 if reference_mapping_available else None
        ),
        "model_hz": 10.0,
        "window_duration_s": 32.0,
        "window_interval_count": 320,
        "window_minimum_overlap_fraction": float(
            contract.window_minimum_overlap_fraction
        ),
        "transition_guard_s": float(contract.transition_guard_s),
        "sync_authorized": False,
        "stage_metric_eligible": False,
        "joined_columns": list(_V3_METADATA_JOIN_COLUMNS),
        "joined_rows_sha256": _canonical_value_sha256(
            _v3_metadata_join_records(consumed_metadata)
        ),
    }
    reference_support_contract = _v3_reference_support_contract(
        reference_mapping_available
    )
    manifest: dict[str, Any] = {
        "schema_version": FEATURE_CACHE_SESSION_SCHEMA_V3,
        "session_id": session_id,
        "content_sha256": "",
        "config_sha256": config_sha256,
        "pipeline_sha256": pipeline_sha256,
        "source_fingerprint": source_fingerprint,
        "window_count": len(metadata),
        "scientific_eligible": False,
        "raw_consumed_bytes_verified": True,
        "timing_adjudicated": True,
        "sync_raw_replay_verified": False,
        "protocol_raw_replay_verified": False,
        "sync_authorized": False,
        "reference_mapping_available": reference_mapping_available,
        "upstream_session_content_sha256": contract.content_sha256,
        "upstream_session_contract": upstream_session_contract,
        "upstream_sync_receipt": upstream_sync_receipt,
        "radar_timing_invalid_reason_schema_version": (
            CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1
        ),
        "radar_timing_invalid_reason_semantics_sha256": (
            CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1
        ),
        "feature_schema": feature_schema,
        "feature_schema_sha256": _canonical_value_sha256(feature_schema),
        "target_firewall": target_firewall,
        "target_firewall_sha256": _canonical_value_sha256(target_firewall),
        "metadata_join_contract": metadata_join_contract,
        "metadata_join_sha256": _canonical_value_sha256(metadata_join_contract),
        "reference_support_contract": reference_support_contract,
        "reference_support_sha256": _canonical_value_sha256(
            reference_support_contract
        ),
        "file_inventory": file_inventory,
        "inventory_sha256": _inventory_sha256(file_inventory),
    }
    manifest["content_sha256"] = _canonical_content_sha256(manifest)
    return manifest


def build_subject(
    subject: Any,
    config: dict[str, Any],
    output_root: Path,
    force: bool,
    *,
    config_sha256: str,
    pipeline_sha256: str,
    acquisition_contract: AcquisitionSessionContract | None = None,
    acquisition_mode: str | None = None,
) -> dict[str, Any]:
    session_id = subject.subject_id
    output_dir = output_root / session_id
    is_v3_acquisition = _is_v3_acquisition_contract(acquisition_contract)
    radar_only_unmapped_v3 = bool(
        is_v3_acquisition
        and acquisition_mode == "diagnostic"
        and acquisition_contract is not None
        and acquisition_contract.mapping is None
    )
    # The acquisition loader exposes only usable session contracts, while a
    # full-cohort build must still traverse and catalogue the frozen unusable
    # session (S24_KHJ).  Skipping it is therefore valid even when the caller
    # is in acquisition mode and has no per-session contract to supply.
    if not subject.usable and acquisition_contract is None:
        return {
            "session_id": session_id,
            "status": "skipped",
            "reason": "missing paired radar/BIOPAC",
        }
    if (acquisition_contract is None) != (acquisition_mode is None):
        raise ValueError("acquisition contract and mode must be supplied together")
    if acquisition_contract is not None:
        if acquisition_contract.session_id != session_id:
            raise ValueError(f"{session_id} acquisition contract ID mismatch")
        if is_v3_acquisition:
            if acquisition_mode != "diagnostic":
                raise ValueError(
                    "version-3 acquisition feature caches are diagnostic-only until "
                    "an independent verifier exists"
                )
            _validate_v3_feature_config(config)
            if (
                acquisition_contract.authorized
                or acquisition_contract.scientific_eligible
                or acquisition_contract.stage_metric_eligible
                or acquisition_contract.manifest.get("sync_raw_replay_verified")
                is not False
                or acquisition_contract.manifest.get("protocol_raw_replay_verified")
                is not False
            ):
                raise ValueError(
                    f"{session_id} version-3 diagnostic contract exceeds its authority"
                )
        if not subject.usable:
            return {
                "session_id": session_id,
                "status": "skipped",
                "reason": "missing paired radar/BIOPAC",
            }
        # Validate content-addressed raw inputs before accepting an existing
        # cache.  The cheap size/mtime fingerprint below is only an
        # invalidation optimization and is never an authority boundary.
        if subject.usable:
            validate_raw_input_bindings(
                acquisition_contract, Path(subject.path).parent
            )
        if acquisition_contract.mapping is None:
            if acquisition_mode == "strict":
                raise ValueError(
                    f"{session_id} has no authorized synchronization mapping"
                )
            if not radar_only_unmapped_v3:
                return {
                    "session_id": session_id,
                    "status": "skipped",
                    "reason": "diagnostic synchronization proposal has no mapping",
                }
        if acquisition_mode == "strict" and not acquisition_contract.authorized:
            raise ValueError(f"{session_id} synchronization is not authorized")
        if (
            acquisition_mode == "strict"
            and not acquisition_contract.scientific_eligible
        ):
            raise ValueError(
                f"{session_id} acquisition reconstruction is not scientifically eligible"
            )
        if (
            acquisition_mode == "strict"
            and not acquisition_contract.measured_timing_eligible
        ):
            raise ValueError(
                f"{session_id} does not have strict measured radar timing"
            )
    source_fingerprint = _source_fingerprint(subject) if subject.usable else "missing"
    complete_files = [
        output_dir / "maps.npy",
        output_dir / "aux.npy",
        output_dir / "metadata.csv",
        output_dir / "frequencies_hz.npy",
        output_dir / "manifest.json",
    ]
    if acquisition_contract is not None:
        complete_files.append(output_dir / "radar_timing_valid_mask.npy")
        if is_v3_acquisition:
            complete_files.extend(
                [
                    output_dir / "radar_timing_invalid_reason_mask.npy",
                    output_dir / "feature_availability_mask.npy",
                ]
            )
    acquisition_provenance = (
        None
        if acquisition_contract is None or is_v3_acquisition
        else _session_acquisition_provenance(
            acquisition_contract, mode=str(acquisition_mode)
        )
    )
    use_range_auxiliary = bool(
        acquisition_contract is not None
        and acquisition_contract.range_feature_eligible
        and acquisition_contract.range_track_path is not None
    )
    if is_v3_acquisition and use_range_auxiliary:
        raise ValueError(
            "version-3 diagnostic feature schema forbids unverified range auxiliary inputs"
        )
    if use_range_auxiliary:
        complete_files.append(output_dir / "range_aux.npy")
    if not force and all(path.is_file() for path in complete_files):
        previous = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        expected = {
            "config_sha256": config_sha256,
            "pipeline_sha256": pipeline_sha256,
            "source_fingerprint": source_fingerprint,
        }
        if acquisition_provenance is not None:
            expected["acquisition_contract"] = acquisition_provenance
        if is_v3_acquisition:
            expected.update(
                {
                    "schema_version": FEATURE_CACHE_SESSION_SCHEMA_V3,
                    "upstream_session_content_sha256": acquisition_contract.content_sha256,
                    "scientific_eligible": False,
                }
            )
        if all(previous.get(key) == value for key, value in expected.items()) and (
            acquisition_contract is None
            or _acquisition_cached_manifest_is_current(
                output_dir,
                previous,
                require_range_aux=use_range_auxiliary,
                is_v3=is_v3_acquisition,
            )
        ):
            return {"session_id": session_id, "status": "ok", "cached": True, **previous}
        print(f"[{session_id}] stale cache provenance; rebuilding", flush=True)

    if not subject.usable:
        return {"session_id": session_id, "status": "skipped", "reason": "missing paired radar/BIOPAC"}
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    qc_cfg = config["reference_qc"]
    radar_hz = float(data_cfg["radar_hz"])
    model_hz = float(data_cfg["model_hz"])
    downsample = int(round(radar_hz / model_hz))
    if not np.isclose(radar_hz / model_hz, downsample):
        raise ValueError("radar_hz/model_hz must be an integer")
    window_samples = int(round(float(data_cfg["window_seconds"]) * model_hz))
    stride_samples = int(round(float(data_cfg["stride_seconds"]) * model_hz))
    if int(data_cfg["range_pool"]) != 2:
        raise ValueError(
            "this two-branch cache layout requires range_pool=2 so the raw and "
            "candidate I/Q phase branches both contain 91 range bins"
        )

    biopac = None
    filtered_reference: np.ndarray | None = None
    bio_start: float | None = None
    bio_end: float | None = None
    if not radar_only_unmapped_v3:
        biopac = load_biopac_mat(
            subject.biopac_path,
            strict=bool(
                acquisition_contract is not None
                and (acquisition_mode == "strict" or is_v3_acquisition)
            ),
        )
        filtered_reference = filter_reference_rsp(
            biopac.rsp,
            fs=biopac.sample_rate_hz,
            band_hz=tuple(map(float, qc_cfg["band_hz"])),
        )
        bio_start = biopac.start_datetime.timestamp()
        bio_end = bio_start + biopac.duration_seconds

    corrected_radar_frames: list[np.ndarray] = []
    outlier_counts: list[int] = []
    radar_starts: list[float] = []
    relative_radar_times: list[np.ndarray] = []
    radar_frame_sequences: list[np.ndarray] = []
    radar_timestamp_sources: list[str] = []
    for radar_id in (1, 2, 3):
        stream = subject.selected_session.radars[radar_id]
        recording = load_xethru_recording(stream.recording_dir, strict=True)
        values = np.asarray(recording.records["bins"], dtype=np.float32)
        values, outlier_count = replace_radar_outliers(values)
        corrected_radar_frames.append(values)
        outlier_counts.append(outlier_count)
        if stream.start_epoch_ms is None:
            raise ValueError(f"{session_id} radar{radar_id} has no absolute start anchor")
        radar_starts.append(stream.start_epoch_ms / 1000.0)
        if acquisition_contract is not None:
            relative_radar_times.append(
                np.asarray(recording.timestamps_ms, dtype=np.float64) / 1000.0
            )
            radar_frame_sequences.append(
                np.asarray(recording.records["frame_sequence"], dtype=np.uint32)
            )
            radar_timestamp_sources.append(
                recording.meta.timestamp_source
                if recording.meta is not None
                else "fallback_40hz"
            )

    # Starts differ by at most a fraction of a 40 Hz frame.  The median anchor
    # keeps a common integer grid and the model receives all views at one index.
    legacy_radar_start = float(np.median(radar_starts)) + (downsample - 1) / (
        2 * radar_hz
    )
    radar_timing_summary: dict[str, Any] | None = None
    model_radar_times_s: np.ndarray | None = None
    if acquisition_contract is None:
        assert bio_start is not None and bio_end is not None
        radar_arrays = [
            causal_block_mean(values, downsample) for values in corrected_radar_frames
        ]
        common_samples = min(map(len, radar_arrays))
        radar_arrays = [array[:common_samples] for array in radar_arrays]
        last_radar_time = legacy_radar_start + (common_samples - 1) / model_hz
        first_end = max(
            window_samples,
            int(np.ceil((bio_start - legacy_radar_start) * model_hz))
            + window_samples,
        )
        last_end = min(
            common_samples,
            int(np.floor((bio_end - legacy_radar_start) * model_hz)),
        )
        end_indices = np.arange(first_end, last_end + 1, stride_samples, dtype=int)
    else:
        resampled = causal_uniform_resample_radar_views_v1(
            corrected_radar_frames,
            relative_radar_times,
            radar_starts,
            radar_frame_sequences,
            output_hz=model_hz,
            max_gap_s=0.050,
            # Reconstruction, canonical features, and SVD must bind the same
            # transform document.  Strictness is enforced explicitly from the
            # structural mask after the shared mask-policy transform.
            gap_policy="mask",
            timestamp_sources=radar_timestamp_sources,
            require_measured_timestamps=(
                acquisition_mode == "strict" or is_v3_acquisition
            ),
        )
        if acquisition_mode == "strict" and not bool(resampled.valid_mask.all()):
            invalid = int(np.size(resampled.valid_mask) - resampled.valid_mask.sum())
            raise ValueError(
                f"{session_id} measured radar timeline contains {invalid} invalid "
                "view-intervals"
            )
        sensor_summary = acquisition_contract.manifest.get("sensor_summary")
        bound_radar_summary = (
            sensor_summary.get("radar")
            if isinstance(sensor_summary, dict)
            else None
        )
        bound_resampling = (
            bound_radar_summary.get("feature_resampling")
            if isinstance(bound_radar_summary, dict)
            else None
        )
        if _canonical_value_sha256(bound_resampling) != _canonical_value_sha256(
            resampled.summary
        ):
            raise ValueError(
                f"{session_id} feature timing differs from the bound reconstruction"
            )
        if bound_radar_summary.get("past_only_outlier_replacements") != outlier_counts:
            raise ValueError(
                f"{session_id} outlier preprocessing differs from the bound reconstruction"
            )
        radar_timing_summary = resampled.summary
        if is_v3_acquisition:
            reason_mask = np.asarray(resampled.invalid_reason_mask)
            if (
                reason_mask.dtype != np.uint8
                or reason_mask.shape != resampled.valid_mask.shape
                or not np.array_equal(reason_mask != 0, ~resampled.valid_mask)
            ):
                raise ValueError(
                    f"{session_id} version-3 timing invalid reasons do not exactly "
                    "explain the structural validity mask"
                )
        radar_arrays = [array for array in resampled.values]
        model_radar_times_s = resampled.times_s
        common_samples = len(model_radar_times_s)
        last_radar_time = resampled.origin_epoch_s + float(model_radar_times_s[-1])
        candidates = np.arange(
            window_samples, common_samples + 1, stride_samples, dtype=int
        )
        if radar_only_unmapped_v3:
            end_indices = candidates
        else:
            assert acquisition_contract.mapping is not None and biopac is not None
            valid_candidates: list[int] = []
            for end in candidates:
                start = end - window_samples
                radar_window_start = float(
                    model_radar_times_s[start] - resampled.interval_s
                )
                radar_window_end = float(model_radar_times_s[end - 1])
                mapped_start = float(
                    acquisition_contract.mapping.radar_to_rsp(radar_window_start)
                )
                mapped_end = float(
                    acquisition_contract.mapping.radar_to_rsp(radar_window_end)
                )
                if 0.0 <= mapped_start < mapped_end <= biopac.duration_seconds:
                    valid_candidates.append(int(end))
            end_indices = np.asarray(valid_candidates, dtype=int)
    if len(end_indices) == 0:
        support_kind = "measured radar support" if radar_only_unmapped_v3 else "overlap"
        raise ValueError(
            f"{session_id} has no {data_cfg['window_seconds']} s {support_kind} windows"
        )

    maps: list[np.ndarray] = []
    auxiliary: list[np.ndarray] = []
    range_auxiliary: list[np.ndarray] = []
    radar_timing_valid_masks: list[np.ndarray] = []
    radar_timing_invalid_reason_masks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    pooled_frequency_grid: np.ndarray | None = None
    range_tracks: tuple[RangeTrack, ...] | None = None
    range_auxiliary_names: tuple[str, ...] = ()
    if use_range_auxiliary:
        assert model_radar_times_s is not None
        range_tracks, range_auxiliary_names = _load_range_track_contract(
            acquisition_contract, model_radar_times_s
        )
    for window_number, end in enumerate(end_indices):
        start = end - window_samples
        if acquisition_contract is not None:
            window_timing_mask = np.asarray(
                resampled.valid_mask[:, start:end], dtype=np.bool_
            )
            if window_timing_mask.shape != (3, window_samples):
                raise RuntimeError(
                    f"{session_id} timing mask does not cover the exact window support"
                )
            radar_timing_valid_masks.append(window_timing_mask)
            if is_v3_acquisition:
                window_reason_mask = np.asarray(
                    resampled.invalid_reason_mask[:, start:end], dtype=np.uint8
                )
                if (
                    window_reason_mask.shape != (3, window_samples)
                    or not np.array_equal(
                        window_reason_mask != 0, ~window_timing_mask
                    )
                ):
                    raise RuntimeError(
                        f"{session_id} timing reason mask does not cover exact window support"
                    )
                radar_timing_invalid_reason_masks.append(window_reason_mask)
        radar_features = [
            range_frequency_features(
                radar[start:end],
                fs=model_hz,
                band_hz=tuple(map(float, data_cfg["respiration_band_hz"])),
                nfft=int(data_cfg["fft_size"]),
                range_pool=int(data_cfg["range_pool"]),
            )
            for radar in radar_arrays
        ]
        raw_maps = np.stack([item.feature_map for item in radar_features])
        # Frequency pooling halves storage while the full-resolution candidate
        # spectra remain in `aux` for sub-bin direct RR estimation.
        usable_frequencies = raw_maps.shape[1] - raw_maps.shape[1] % 2
        feature_map = _pool_range_frequency_map(raw_maps)
        full_grid = radar_features[0].frequencies_hz[:usable_frequencies]
        current_grid = full_grid.reshape(-1, 2).mean(axis=1).astype(
            np.float64 if is_v3_acquisition else np.float32
        )
        if pooled_frequency_grid is None:
            pooled_frequency_grid = current_grid
        elif is_v3_acquisition and not np.array_equal(
            pooled_frequency_grid, current_grid
        ):
            raise RuntimeError(
                f"{session_id} version-3 frequency grid changed between windows"
            )
        maps.append(feature_map)
        auxiliary.append(fuse_auxiliary_features(radar_features))
        if range_tracks is not None:
            current_range_aux, current_range_names = fuse_range_track_window_features(
                range_tracks, start, end
            )
            if current_range_names != range_auxiliary_names:
                raise RuntimeError("range auxiliary feature order changed within a session")
            range_auxiliary.append(current_range_aux)
        classical = classical_rr_estimate(
            radar_features, rr_range_bpm=tuple(map(float, data_cfg["rr_range_bpm"]))
        )

        # Preserve these legacy nominal coordinates exactly: downstream SVD
        # reconstruction for historical caches is bound to them. Acquisition
        # v2 replaces the public window coordinates with measured support.
        window_start_epoch = legacy_radar_start + start / model_hz
        window_end_epoch = legacy_radar_start + (end - 1) / model_hz
        if acquisition_contract is None:
            assert bio_start is not None
            reference_start_biopac_s = window_start_epoch - bio_start
            reference_end_biopac_s = reference_start_biopac_s + float(
                data_cfg["window_seconds"]
            )
            radar_window_start_relative_s = start / model_hz
            radar_window_end_relative_s = end / model_hz
        else:
            assert model_radar_times_s is not None
            radar_window_start_relative_s = float(
                model_radar_times_s[start] - resampled.interval_s
            )
            radar_window_end_relative_s = float(model_radar_times_s[end - 1])
            if radar_only_unmapped_v3:
                reference_start_biopac_s = float("nan")
                reference_end_biopac_s = float("nan")
            else:
                assert acquisition_contract.mapping is not None
                reference_start_biopac_s = float(
                    acquisition_contract.mapping.radar_to_rsp(
                        radar_window_start_relative_s
                    )
                )
                reference_end_biopac_s = float(
                    acquisition_contract.mapping.radar_to_rsp(
                        radar_window_end_relative_s
                    )
                )
        common_row = {
            "session_id": session_id,
            "session_number": subject.subject_number,
            "identity": identity_for_session(session_id),
            "protocol": protocol_for_session(session_id),
            "window_number": window_number,
            "classical_rr_bpm": classical.rr_bpm,
            "classical_confidence": classical.confidence,
            "radar_peak_1_bpm": classical.radar_peaks_bpm[0],
            "radar_peak_2_bpm": classical.radar_peaks_bpm[1],
            "radar_peak_3_bpm": classical.radar_peaks_bpm[2],
            "radar_peak_spread_bpm": classical.consensus_spread_bpm,
        }
        if radar_only_unmapped_v3:
            row = {
                **common_row,
                **{
                    column: float("nan")
                    for column in _V3_REFERENCE_FLOAT_NAN_COLUMNS
                },
                "reference_start_sample": -1,
                "reference_end_sample": -1,
                "reference_window_start_biopac_s": float("nan"),
                "reference_window_end_biopac_s": float("nan"),
                "radar_window_start_relative_s": radar_window_start_relative_s,
                "radar_window_end_relative_s": radar_window_end_relative_s,
                "reference_mapping_available": False,
                "reference_valid": False,
                "radar_observable": False,
                "classical_acceptable_within_2bpm": False,
                "sync_authorized": False,
                "sync_confidence": float("nan"),
                "alignment_scientific_eligible": False,
                "acquisition_phase": None,
                "acquisition_phase_name": None,
                "acquisition_phase_status": None,
                "acquisition_phase_confidence": float("nan"),
                "phase_overlap_fraction": float("nan"),
                "transition_window": False,
                "eligible_for_stage_metrics": False,
                "phase7_assignment": None,
                "acquisition_batch": protocol_for_session(session_id),
            }
        else:
            assert biopac is not None and filtered_reference is not None
            assert bio_start is not None
            canonical_window_start_s = (
                window_start_epoch - bio_start
                if acquisition_contract is None
                else reference_start_biopac_s
            )
            canonical_window_end_s = (
                window_end_epoch - bio_start
                if acquisition_contract is None
                else reference_end_biopac_s
            )
            bio_first, bio_last = _half_open_sample_bounds(
                reference_start_biopac_s,
                reference_end_biopac_s,
                biopac.sample_rate_hz,
            )
            if bio_first < 0 or bio_last > len(biopac.rsp) or bio_last <= bio_first:
                raise RuntimeError(
                    f"{session_id} reference window escaped the verified overlap: "
                    f"[{bio_first}, {bio_last}) of {len(biopac.rsp)}"
                )
            reference = estimate_reference_window(
                biopac.rsp[bio_first:bio_last],
                filtered_reference[bio_first:bio_last],
                fs=biopac.sample_rate_hz,
                rr_range_bpm=tuple(map(float, data_cfg["rr_range_bpm"])),
                min_cycles=int(qc_cfg["min_cycles"]),
                max_clip_fraction=float(qc_cfg["max_clip_fraction"]),
                min_spectral_concentration=float(qc_cfg["min_spectral_concentration"]),
                min_periodicity=float(qc_cfg["min_periodicity"]),
                max_interval_cv=float(qc_cfg["max_interval_cv"]),
                max_estimator_disagreement_bpm=float(
                    qc_cfg["max_estimator_disagreement_bpm"]
                ),
                max_phase_residual_rad=float(qc_cfg["max_phase_residual_rad"]),
            )
            guard_samples = int(round(2.0 * biopac.sample_rate_hz))
            guard_rsp = biopac.rsp[
                max(0, bio_first - guard_samples) : min(
                    len(biopac.rsp), bio_last + guard_samples
                )
            ]
            guard_clip_fraction = float(np.mean(np.abs(guard_rsp) >= 9.8))
            reference_valid = bool(
                reference.valid
                and guard_clip_fraction <= float(qc_cfg["max_clip_fraction"])
            )
            if acquisition_mode == "diagnostic":
                reference_valid = False
            classical_error = abs(classical.rr_bpm - reference.rr_bpm)
            acceptable_prediction = bool(reference_valid and classical_error <= 2.0)
            row = {
                **common_row,
                "window_start_s": canonical_window_start_s,
                "window_end_s": canonical_window_end_s,
                "rr_bpm": reference.rr_bpm,
                "rr_spectral_bpm": reference.rr_spectral_bpm,
                "rr_phase_bpm": reference.rr_phase_bpm,
                "rr_events_bpm": reference.rr_events_bpm,
                "reference_valid": reference_valid,
                "reference_quality": reference.quality,
                "reference_sigma_bpm": float(
                    np.clip(
                        0.35
                        + 0.20 * reference.estimator_disagreement_bpm
                        + 0.50 * (1 - reference.quality),
                        0.35,
                        2.0,
                    )
                ),
                "spectral_concentration": reference.spectral_concentration,
                "periodicity": reference.periodicity,
                "interval_cv": reference.interval_cv,
                "estimator_disagreement_bpm": reference.estimator_disagreement_bpm,
                "phase_residual_rad": reference.phase_residual_rad,
                "clip_fraction": reference.clip_fraction,
                "guard_clip_fraction": guard_clip_fraction,
                "plateau_fraction": reference.plateau_fraction,
                "breath_count": reference.breath_count,
                "classical_error_bpm": classical_error,
                "radar_observable": acceptable_prediction,
                "classical_acceptable_within_2bpm": acceptable_prediction,
            }
        if acquisition_contract is not None and not radar_only_unmapped_v3:
            assignment = assign_stage_window(
                acquisition_contract,
                reference_start_biopac_s,
                reference_end_biopac_s,
            )
            row.update(
                {
                    "reference_start_sample": bio_first,
                    "reference_end_sample": bio_last,
                    "reference_window_start_biopac_s": reference_start_biopac_s,
                    "reference_window_end_biopac_s": reference_end_biopac_s,
                    "radar_window_start_relative_s": radar_window_start_relative_s,
                    "radar_window_end_relative_s": radar_window_end_relative_s,
                    "reference_mapping_available": True,
                    "sync_authorized": acquisition_contract.authorized,
                    "sync_confidence": float(
                        acquisition_contract.receipt["result"]["confidence"]
                    ),
                    "alignment_scientific_eligible": bool(
                        acquisition_mode == "strict"
                        and acquisition_contract.scientific_eligible
                    ),
                    "acquisition_phase": assignment.stage_id,
                    "acquisition_phase_name": assignment.stage_name,
                    "acquisition_phase_status": assignment.stage_status,
                    "acquisition_phase_confidence": assignment.stage_confidence,
                    "phase_overlap_fraction": assignment.overlap_fraction,
                    "transition_window": assignment.transition_window,
                    "eligible_for_stage_metrics": bool(
                        acquisition_mode == "strict"
                        and acquisition_contract.scientific_eligible
                        and acquisition_contract.stage_metric_eligible
                        and assignment.eligible_for_stage_metrics
                    ),
                    "phase7_assignment": assignment.phase7_assignment,
                    # Compatibility value is now named honestly as a batch
                    # assignment; legacy `protocol` remains unchanged.
                    "acquisition_batch": protocol_for_session(session_id),
                }
            )
        rows.append(row)

    map_array = np.stack(maps)
    aux_array = np.stack(auxiliary).astype(np.float32)
    metadata = pd.DataFrame(rows)
    maps_path = output_dir / "maps.npy"
    aux_path = output_dir / "aux.npy"
    metadata_path = output_dir / "metadata.csv"
    frequencies_path = output_dir / "frequencies_hz.npy"
    timing_mask_path = output_dir / "radar_timing_valid_mask.npy"
    timing_reason_path = output_dir / "radar_timing_invalid_reason_mask.npy"
    feature_availability_path = output_dir / "feature_availability_mask.npy"
    range_aux_array: np.ndarray | None = None
    range_aux_path = output_dir / "range_aux.npy"
    if range_tracks is not None:
        range_aux_array = np.stack(range_auxiliary).astype(np.float32)
    radar_timing_valid_mask_array: np.ndarray | None = None
    radar_timing_invalid_reason_mask_array: np.ndarray | None = None
    feature_availability_mask_array: np.ndarray | None = None
    map_view_availability: np.ndarray | None = None
    aux_feature_availability: np.ndarray | None = None
    aux_feature_names: tuple[str, ...] = ()
    availability_feature_names: tuple[str, ...] = ()
    if acquisition_contract is not None:
        radar_timing_valid_mask_array = np.stack(
            radar_timing_valid_masks
        ).astype(np.bool_, copy=False)
    if is_v3_acquisition:
        radar_timing_invalid_reason_mask_array = np.stack(
            radar_timing_invalid_reason_masks
        ).astype(np.uint8, copy=False)
        if not np.array_equal(
            radar_timing_invalid_reason_mask_array != 0,
            ~radar_timing_valid_mask_array,
        ):
            raise RuntimeError(
                f"{session_id} version-3 timing reason union changed before publication"
            )
        if aux_array.shape[1] < 37 or (aux_array.shape[1] - 29) % 8:
            raise RuntimeError(f"{session_id} auxiliary feature geometry is invalid")
        auxiliary_frequency_bins = (aux_array.shape[1] - 29) // 8
        (
            map_view_availability,
            aux_feature_availability,
            aux_feature_names,
            availability_feature_names,
        ) = _v3_feature_availability(
            radar_timing_valid_mask_array,
            auxiliary_frequency_bins=auxiliary_frequency_bins,
        )
        if map_array.ndim != 4 or map_array.shape[1:] != (
            3,
            len(pooled_frequency_grid),
            182,
        ):
            raise RuntimeError(f"{session_id} version-3 map feature geometry is invalid")
        map_array = np.array(map_array, copy=True, order="C")
        aux_array = np.array(aux_array, copy=True, order="C")
        map_array[~map_view_availability] = np.float16(0.0)
        aux_array[~aux_feature_availability] = np.float32(0.0)
        if (
            np.any(map_array[~map_view_availability] != 0)
            or np.any(np.signbit(map_array[~map_view_availability]))
            or np.any(aux_array[~aux_feature_availability] != 0)
            or np.any(np.signbit(aux_array[~aux_feature_availability]))
        ):
            raise RuntimeError(
                f"{session_id} unavailable version-3 features are not exact +0.0"
            )
        feature_availability_mask_array = np.concatenate(
            (map_view_availability, aux_feature_availability), axis=1
        ).astype(np.bool_, copy=False)

    np.save(maps_path, map_array, allow_pickle=False)
    np.save(aux_path, aux_array, allow_pickle=False)
    if range_aux_array is not None:
        np.save(range_aux_path, range_aux_array, allow_pickle=False)
    np.save(frequencies_path, pooled_frequency_grid, allow_pickle=False)
    if radar_timing_valid_mask_array is not None:
        np.save(timing_mask_path, radar_timing_valid_mask_array, allow_pickle=False)
    if radar_timing_invalid_reason_mask_array is not None:
        np.save(
            timing_reason_path,
            radar_timing_invalid_reason_mask_array,
            allow_pickle=False,
        )
    if feature_availability_mask_array is not None:
        np.save(
            feature_availability_path,
            feature_availability_mask_array,
            allow_pickle=False,
        )
    metadata.to_csv(metadata_path, index=False)
    file_inventory = {
        "maps": _array_inventory(maps_path, map_array),
        "aux": _array_inventory(aux_path, aux_array),
        "metadata": _metadata_inventory(metadata_path, metadata),
        "frequencies_hz": _array_inventory(
            frequencies_path, np.asarray(pooled_frequency_grid)
        ),
    }
    if range_aux_array is not None:
        file_inventory["range_aux"] = _array_inventory(
            range_aux_path, range_aux_array
        )
    if radar_timing_valid_mask_array is not None:
        file_inventory["radar_timing_valid_mask"] = _array_inventory(
            timing_mask_path, radar_timing_valid_mask_array
        )
    if radar_timing_invalid_reason_mask_array is not None:
        file_inventory["radar_timing_invalid_reason_mask"] = _array_inventory(
            timing_reason_path, radar_timing_invalid_reason_mask_array
        )
    if feature_availability_mask_array is not None:
        file_inventory["feature_availability_mask"] = _array_inventory(
            feature_availability_path, feature_availability_mask_array
        )
    if is_v3_acquisition:
        assert acquisition_contract is not None
        assert radar_timing_valid_mask_array is not None
        assert radar_timing_invalid_reason_mask_array is not None
        assert feature_availability_mask_array is not None
        consumed_metadata = pd.read_csv(
            metadata_path,
            keep_default_na=False,
            na_values=[""],
        )
        subject_manifest = _build_v3_session_manifest(
            session_id=session_id,
            contract=acquisition_contract,
            config_sha256=config_sha256,
            pipeline_sha256=pipeline_sha256,
            source_fingerprint=source_fingerprint,
            maps=map_array,
            aux=aux_array,
            frequencies_hz=np.asarray(pooled_frequency_grid),
            timing_valid_mask=radar_timing_valid_mask_array,
            timing_reason_mask=radar_timing_invalid_reason_mask_array,
            feature_availability_mask=feature_availability_mask_array,
            aux_feature_names=aux_feature_names,
            availability_feature_names=availability_feature_names,
            metadata=consumed_metadata,
            file_inventory=file_inventory,
            biopac_sample_rate_hz=(
                None if biopac is None else float(biopac.sample_rate_hz)
            ),
        )
        _atomic_write_bytes(
            output_dir / "manifest.json",
            json.dumps(
                subject_manifest,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
        )
        return {
            "session_id": session_id,
            "status": "ok",
            "cached": False,
            **subject_manifest,
        }
    subject_manifest = {
        "schema_version": "snn_rr.feature_cache_session.v2",
        "config_sha256": config_sha256,
        "pipeline_sha256": pipeline_sha256,
        "source_fingerprint": source_fingerprint,
        "window_count": len(metadata),
        "valid_reference_count": int(metadata.reference_valid.sum()),
        "observable_count": int(metadata.radar_observable.sum()),
        "map_shape": list(map_array.shape),
        "aux_shape": list(aux_array.shape),
        "feature_branch_layout": [
            f"raw_power_{182 // int(data_cfg['range_pool'])}_range_bins",
            "candidate_iq_phase_power_91_range_bins",
        ],
        "input_branches": 2,
        "frequency_min_hz": float(pooled_frequency_grid[0]),
        "frequency_max_hz": float(pooled_frequency_grid[-1]),
        "radar_outlier_replacements": outlier_counts,
        "radar_start_epoch": (
            legacy_radar_start
            if acquisition_contract is None
            else resampled.origin_epoch_s
            + float(resampled.summary["first_grid_left_edge_s"])
        ),
        "legacy_radar_start_epoch": (
            None if acquisition_contract is None else legacy_radar_start
        ),
        "radar_last_epoch": last_radar_time,
        "biopac_start_epoch": bio_start,
        "biopac_end_epoch": bio_end,
        "causal_downsample_latency_ms": (
            1000.0 / model_hz
            if acquisition_contract is not None
            else 1000.0 * (downsample - 1) / radar_hz
        ),
        "file_inventory": file_inventory,
        "inventory_sha256": _inventory_sha256(file_inventory),
    }
    if acquisition_provenance is not None:
        subject_manifest.update(
            {
                "acquisition_contract": acquisition_provenance,
                "range_aux_shape": (
                    None if range_aux_array is None else list(range_aux_array.shape)
                ),
                "range_aux_feature_names": list(range_auxiliary_names),
                "radar_timing_summary": radar_timing_summary,
                "radar_timing_valid_mask_shape": list(
                    radar_timing_valid_mask_array.shape
                ),
                "radar_timing_invalid_interval_count": int(
                    np.size(radar_timing_valid_mask_array)
                    - radar_timing_valid_mask_array.sum()
                ),
                "radar_timing_mask_contract": {
                    "mask_required_for_gap_tolerant_consumers": True,
                    "scientific_cache_requires_all_true": True,
                    "diagnostic_cache_trainable": False,
                    "invalid_cells_are_exact_zero_but_not_semantic_measurements": True,
                },
                "measured_window_support": {
                    "timestamp_semantics": "right_edge_exclusive",
                    "window_interval_count": window_samples,
                    "window_duration_s": float(data_cfg["window_seconds"]),
                    "stride_interval_count": stride_samples,
                    "stride_duration_s": float(data_cfg["stride_seconds"]),
                    "reference_sample_indexing": {
                        "sample_timestamp_semantics": "i / sample_rate_hz",
                        "support_membership": "start_s <= i / sample_rate_hz < end_s",
                        "slice_boundary_rule": "ceil_both_boundaries",
                        "near_integer_canonicalization": (
                            "abs(coordinate-rint(coordinate)) <= "
                            "max(1e-9,8*spacing(max(abs(coordinate),1)))"
                        ),
                    },
                },
                "timing_column_semantics": {
                    "window_start_s": "authorized_reference_clock_support_start",
                    "window_end_s": "authorized_reference_clock_support_end",
                    "radar_window_start_relative_s": "measured_grid_support_start",
                    "radar_window_end_relative_s": "measured_grid_support_end",
                },
                "protocol_column_semantics": (
                    "deprecated phase7 batch assignment broadcast for legacy compatibility; "
                    "use acquisition_phase for reconstructed stage"
                ),
                "stage_metric_window_policy": {
                    "session_stage_gate_required": True,
                    "minimum_overlap_fraction": float(
                        acquisition_contract.window_minimum_overlap_fraction
                    ),
                    "transition_guard_s": float(
                        acquisition_contract.transition_guard_s
                    ),
                    "window_duration_s": float(data_cfg["window_seconds"]),
                    "short_stage_policy": (
                        "no metric when the fixed 32 s window cannot satisfy the "
                        "declared overlap and transition guard; no shorter evaluation "
                        "window is synthesized"
                    ),
                },
            }
        )
    subject_manifest["content_sha256"] = _canonical_content_sha256(subject_manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(subject_manifest, indent=2), encoding="utf-8"
    )
    return {"session_id": session_id, "status": "ok", "cached": False, **subject_manifest}


def _main_impl() -> int:
    global _ACTIVE_V3_ATTEMPT
    args = parse_args()
    subjects_filter_applied = args.subjects is not None
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    dataset_root = (args.dataset_root or REPOSITORY_ROOT / config["data"]["root"]).resolve()
    requested_output_root = Path(
        os.path.abspath(
            os.fspath(args.cache_dir or REPOSITORY_ROOT / config["data"]["cache_dir"])
        )
    )
    output_root = requested_output_root.resolve()
    v3_final_output_root: Path | None = None
    acquisition = None
    acquisition_schema: str | None = None
    v3_reconstruction_payload: bytes | None = None
    v3_reconstruction_file_sha256: str | None = None
    v3_dataset_catalogue: dict[str, Any] | None = None
    root_acquisition_contract: dict[str, Any] | None = None
    if args.acquisition_manifest is not None:
        if args.cache_dir is None:
            raise ValueError(
                "--acquisition-manifest requires an explicit new --cache-dir"
            )
        legacy_root = (REPOSITORY_ROOT / config["data"]["cache_dir"]).resolve()
        if output_root == legacy_root:
            raise ValueError("acquisition mode refuses to overwrite the historical cache")
        acquisition = load_acquisition_reconstruction(args.acquisition_manifest)
        acquisition_schema = str(acquisition.manifest.get("schema_version"))
        if acquisition_schema not in {ACQUISITION_SCHEMA, ACQUISITION_SCHEMA_V3}:
            raise ValueError(
                "feature-cache acquisition mode requires reconstruction v2 or v3"
            )
        if acquisition_schema == ACQUISITION_SCHEMA_V3:
            if os.path.lexists(requested_output_root):
                raise FileExistsError(
                    "version-3 feature-cache output already exists and cannot be "
                    f"overwritten or resumed: {requested_output_root}"
                )
            output_root = requested_output_root.parent.resolve() / requested_output_root.name
            if args.acquisition_mode != "diagnostic":
                raise ValueError(
                    "acquisition-v3 feature caches are diagnostic-only until an "
                    "independent verifier exists"
                )
            if subjects_filter_applied:
                raise ValueError(
                    "acquisition-v3 diagnostic cache requires an untargeted exact "
                    "30-to-29 cohort projection"
                )
            if _paths_overlap(output_root, dataset_root):
                raise ValueError(
                    "version-3 feature-cache output must be disjoint from the raw "
                    "dataset tree"
                )
            if _paths_overlap(output_root, args.acquisition_manifest.parent):
                raise ValueError(
                    "version-3 feature-cache output must be disjoint from the "
                    "acquisition reconstruction tree"
                )
            _validate_v3_feature_config(config)
            expected_usable_ids = _validate_v3_reconstruction_projection(acquisition)
            v3_reconstruction_payload, v3_reconstruction_file_sha256 = (
                _stable_regular_file_payload(
                    args.acquisition_manifest,
                    label="version-3 acquisition reconstruction",
                )
            )
            if _strict_json_payload(
                v3_reconstruction_payload,
                label="version-3 acquisition reconstruction snapshot",
            ) != acquisition.manifest:
                raise RuntimeError(
                    "version-3 acquisition reconstruction parsed generation differs "
                    "from its exact-byte snapshot"
                )
            reconstruction_copy = output_root / "acquisition_reconstruction.json"
            if Path(os.path.abspath(os.fspath(args.acquisition_manifest))) == Path(
                os.path.abspath(os.fspath(reconstruction_copy))
            ):
                raise ValueError(
                    "version-3 acquisition input cannot alias its cache-side snapshot"
                )
            root_acquisition_contract = {
                "schema_version": FEATURE_CACHE_ACQUISITION_SCHEMA_V3,
                "upstream_reconstruction_schema_version": ACQUISITION_SCHEMA_V3,
                "reconstruction_manifest": "acquisition_reconstruction.json",
                "reconstruction_manifest_sha256": v3_reconstruction_file_sha256,
                "reconstruction_manifest_bytes": len(v3_reconstruction_payload),
                "reconstruction_content_sha256": acquisition.content_sha256,
                "cohort_authority_sha256": acquisition.manifest[
                    "cohort_authority_sha256"
                ],
                "cohort_authority_content_sha256": acquisition.manifest[
                    "cohort_authority_content_sha256"
                ],
                "mode": "diagnostic",
                "scientific_eligible": False,
                "subjects_filter_applied": False,
                "selection_scope": "full_cohort",
                "full_cohort_complete": True,
                "expected_usable_session_ids": list(expected_usable_ids),
                "expected_usable_session_ids_sha256": _canonical_value_sha256(
                    list(expected_usable_ids)
                ),
                "cache_usable_session_ids": [],
                "cache_usable_session_ids_sha256": _canonical_value_sha256([]),
                "cache_inventory_aggregate_sha256": _canonical_value_sha256({}),
                "annotation_only_columns": list(V3_ANNOTATION_ONLY_COLUMNS),
            }
            v3_dataset_catalogue = _validate_v3_dataset_catalogue(dataset_root)
            v3_final_output_root = output_root
            output_root = _start_v3_private_attempt(
                v3_final_output_root,
                acquisition_manifest=args.acquisition_manifest,
            )
            assert _ACTIVE_V3_ATTEMPT is not None
            _ACTIVE_V3_ATTEMPT.update(
                {
                    "acquisition_reconstruction_file_sha256": (
                        v3_reconstruction_file_sha256
                    ),
                    "acquisition_reconstruction_content_sha256": (
                        acquisition.content_sha256
                    ),
                    "dataset_catalogue_sha256": _canonical_value_sha256(
                        v3_dataset_catalogue
                    ),
                }
            )
        else:
            assert acquisition_schema == ACQUISITION_SCHEMA
        if acquisition_schema == ACQUISITION_SCHEMA and args.acquisition_mode == "strict":
            if subjects_filter_applied:
                raise ValueError(
                    "strict acquisition cache requires an untargeted full-cohort build"
                )
            if not acquisition.full_cohort_complete:
                raise ValueError(
                    "strict acquisition cache requires a complete full-cohort reconstruction"
                )
            if not acquisition.scientific_eligible:
                raise ValueError(
                    "strict acquisition cache requires a scientifically eligible reconstruction"
                )
        if acquisition_schema == ACQUISITION_SCHEMA:
            expected_usable_ids = tuple(
                str(item) for item in acquisition.manifest.get(
                    "expected_usable_session_ids", []
                )
            )
            root_acquisition_contract = {
                "schema_version": FEATURE_CACHE_ACQUISITION_SCHEMA,
                "reconstruction_manifest": str(acquisition.manifest_path),
                "reconstruction_content_sha256": acquisition.content_sha256,
                "cohort_authority_sha256": acquisition.manifest[
                    "cohort_authority_sha256"
                ],
                "cohort_authority_content_sha256": acquisition.manifest[
                    "cohort_authority_content_sha256"
                ],
                "mode": args.acquisition_mode,
                "subjects_filter_applied": subjects_filter_applied,
                "selection_scope": acquisition.selection_scope,
                "reconstruction_full_cohort_complete": acquisition.full_cohort_complete,
                "expected_usable_session_ids": list(expected_usable_ids),
                "expected_usable_session_ids_sha256": _canonical_value_sha256(
                    list(expected_usable_ids)
                ),
                "annotation_only_columns": list(ANNOTATION_ONLY_COLUMNS),
                "scientific_eligible": bool(
                    args.acquisition_mode == "strict"
                    and acquisition.manifest.get("scientific_eligible") is True
                ),
            }
    output_root.mkdir(parents=True, exist_ok=True)
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    pipeline_paths = _pipeline_paths()
    pipeline_digest = _sha256_files(pipeline_paths)
    if _ACTIVE_V3_ATTEMPT is not None:
        _ACTIVE_V3_ATTEMPT.update(
            {
                "config_sha256": config_digest,
                "pipeline_sha256": pipeline_digest,
                "pipeline_files": [str(path.resolve()) for path in pipeline_paths],
            }
        )
    manifest = build_dataset_manifest(dataset_root)
    if acquisition_schema == ACQUISITION_SCHEMA_V3:
        assert v3_dataset_catalogue is not None
        if _validate_v3_dataset_catalogue(
            dataset_root, manifest=manifest
        ) != v3_dataset_catalogue:
            raise RuntimeError(
                "version-3 dataset catalogue changed during manifest discovery"
            )
    selected = manifest.subjects
    if subjects_filter_applied:
        wanted = set(args.subjects)
        selected = tuple(subject for subject in selected if subject.subject_id in wanted)
        unknown = wanted - {subject.subject_id for subject in selected}
        if unknown:
            raise KeyError(f"unknown subjects: {sorted(unknown)}")

    results = []
    for subject in selected:
        if _ACTIVE_V3_ATTEMPT is not None:
            _ACTIVE_V3_ATTEMPT["current_session_id"] = subject.subject_id
        print(f"[{subject.subject_id}] building", flush=True)
        session_contract = (
            None if acquisition is None else acquisition.sessions.get(subject.subject_id)
        )
        if acquisition is not None and subject.usable and session_contract is None:
            raise ValueError(
                f"{subject.subject_id} is absent from the acquisition reconstruction"
            )
        result = build_subject(
                subject,
                config,
                output_root,
                args.force,
                config_sha256=config_digest,
                pipeline_sha256=pipeline_digest,
                acquisition_contract=session_contract,
                acquisition_mode=(
                    None
                    if acquisition is None or session_contract is None
                    else args.acquisition_mode
                ),
            )
        if result.get("status") == "ok":
            session_manifest_path = output_root / subject.subject_id / "manifest.json"
            result["session_manifest_sha256"] = _sha256_file(
                session_manifest_path
            )
            result["session_manifest_content_sha256"] = result.get(
                "content_sha256"
            )
            if _ACTIVE_V3_ATTEMPT is not None:
                completed = _ACTIVE_V3_ATTEMPT["completed_session_ids"]
                assert isinstance(completed, list)
                completed.append(subject.subject_id)
        results.append(result)
    if _ACTIVE_V3_ATTEMPT is not None:
        _ACTIVE_V3_ATTEMPT["current_session_id"] = None

    # A targeted rebuild must not erase the catalogue entries for all other
    # sessions.  Merge by canonical session ID and retain manifest order.
    root_manifest_path = output_root / "manifest.json"
    previous_by_id: dict[str, dict[str, Any]] = {}
    if root_manifest_path.is_file():
        previous_root = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        previous_contract = previous_root.get("acquisition_contract")
        previous_comparable = (
            None
            if previous_contract is None
            else {
                key: value
                for key, value in previous_contract.items()
                if key
                not in {
                    "scientific_eligible",
                    "subjects_filter_applied",
                    "selection_scope",
                    "full_cohort_complete",
                    "cache_usable_session_ids",
                    "cache_usable_session_ids_sha256",
                    "cache_inventory_aggregate_sha256",
                }
            }
        )
        current_comparable = (
            None
            if root_acquisition_contract is None
            else {
                key: value
                for key, value in root_acquisition_contract.items()
                if key
                not in {
                    "scientific_eligible",
                    "subjects_filter_applied",
                    "selection_scope",
                    "full_cohort_complete",
                    "cache_usable_session_ids",
                    "cache_usable_session_ids_sha256",
                    "cache_inventory_aggregate_sha256",
                }
            }
        )
        if previous_comparable != current_comparable:
            raise ValueError(
                "refusing to mix feature-cache entries from different acquisition contracts"
            )
        previous_by_id = {
            item["session_id"]: item for item in previous_root.get("sessions", [])
        }
    updated_by_id = {item["session_id"]: item for item in results}
    merged_results = []
    for subject in manifest.subjects:
        item = updated_by_id.get(subject.subject_id, previous_by_id.get(subject.subject_id))
        if item is not None:
            merged_results.append(item)

    # Publication barrier: a long full-cohort build must not publish a root
    # that combines outputs from different config/source/raw/reconstruction
    # generations.  Re-read every authority and every v2 session payload just
    # before constructing the root manifest.
    if hashlib.sha256(args.config.read_bytes()).hexdigest() != config_digest:
        raise RuntimeError("feature configuration changed during cache build")
    if _sha256_files(pipeline_paths) != pipeline_digest:
        raise RuntimeError("feature pipeline source changed during cache build")
    final_dataset_manifest = build_dataset_manifest(dataset_root)
    if _canonical_value_sha256(final_dataset_manifest.to_dict()) != (
        _canonical_value_sha256(manifest.to_dict())
    ):
        raise RuntimeError("dataset manifest changed during feature cache build")
    if acquisition_schema == ACQUISITION_SCHEMA_V3:
        assert v3_dataset_catalogue is not None
        if _validate_v3_dataset_catalogue(
            dataset_root, manifest=final_dataset_manifest
        ) != v3_dataset_catalogue:
            raise RuntimeError(
                "version-3 dataset catalogue changed before cache publication"
            )
    if acquisition is not None:
        assert args.acquisition_manifest is not None
        final_acquisition = load_acquisition_reconstruction(
            args.acquisition_manifest
        )
        if final_acquisition.content_sha256 != acquisition.content_sha256:
            raise RuntimeError(
                "acquisition reconstruction changed during feature cache build"
            )
        merged_by_id_for_barrier = {
            str(item.get("session_id")): item for item in merged_results
        }
        for session_id, item in merged_by_id_for_barrier.items():
            if item.get("status") != "ok":
                continue
            session_contract = final_acquisition.sessions.get(session_id)
            if session_contract is None:
                raise RuntimeError(
                    f"{session_id} output has no final acquisition contract"
                )
            validate_raw_input_bindings(session_contract, dataset_root)
            session_dir = output_root / session_id
            session_manifest_path = session_dir / "manifest.json"
            if not session_manifest_path.is_file():
                raise RuntimeError(
                    f"{session_id} feature session manifest is missing at publication"
                )
            session_manifest = json.loads(
                session_manifest_path.read_text(encoding="utf-8")
            )
            if (
                session_manifest.get("config_sha256") != config_digest
                or session_manifest.get("pipeline_sha256") != pipeline_digest
            ):
                raise RuntimeError(
                    f"{session_id} feature session source/config generation drifted"
                )
            require_range_aux = bool(
                session_contract.range_feature_eligible
                and session_contract.range_track_path is not None
            )
            if not _acquisition_cached_manifest_is_current(
                session_dir,
                session_manifest,
                require_range_aux=require_range_aux,
                is_v3=acquisition_schema == ACQUISITION_SCHEMA_V3,
            ):
                raise RuntimeError(
                    f"{session_id} feature output inventory changed before publication"
                )
            observed_file_hash = _sha256_file(session_manifest_path)
            observed_content_hash = session_manifest.get("content_sha256")
            if (
                item.get("session_manifest_sha256") != observed_file_hash
                or item.get("session_manifest_content_sha256")
                != observed_content_hash
                or item.get("content_sha256") != observed_content_hash
            ):
                raise RuntimeError(
                    f"{session_id} in-memory/output feature manifest mismatch"
                )
    if acquisition_schema == ACQUISITION_SCHEMA_V3:
        assert acquisition is not None
        assert root_acquisition_contract is not None
        assert args.acquisition_manifest is not None
        assert v3_reconstruction_payload is not None
        assert v3_reconstruction_file_sha256 is not None
        expected_usable_ids = _validate_v3_reconstruction_projection(acquisition)
        merged_by_id = {
            str(item.get("session_id")): item for item in merged_results
        }
        cache_usable_ids = tuple(
            session_id
            for session_id in expected_usable_ids
            if merged_by_id.get(session_id, {}).get("status") == "ok"
        )
        if cache_usable_ids != expected_usable_ids:
            missing = sorted(set(expected_usable_ids) - set(cache_usable_ids))
            raise RuntimeError(
                f"version-3 cache did not produce the exact 29 usable sessions: {missing}"
            )
        root_items: list[dict[str, Any]] = []
        inventory_bindings: dict[str, str] = {}
        for session_id in cache_usable_ids:
            item = merged_by_id[session_id]
            if (
                item.get("schema_version") != FEATURE_CACHE_SESSION_SCHEMA_V3
                or item.get("scientific_eligible") is not False
                or item.get("upstream_session_content_sha256")
                != acquisition.sessions[session_id].content_sha256
            ):
                raise RuntimeError(
                    f"{session_id} version-3 session contract drifted before root publication"
                )
            inventory_sha256 = str(item["inventory_sha256"])
            inventory_bindings[session_id] = inventory_sha256
            root_items.append(
                {
                    "session_id": session_id,
                    "status": "ok",
                    "schema_version": FEATURE_CACHE_SESSION_SCHEMA_V3,
                    "manifest_path": f"{session_id}/manifest.json",
                    "manifest_sha256": item["session_manifest_sha256"],
                    "manifest_content_sha256": item[
                        "session_manifest_content_sha256"
                    ],
                    "inventory_sha256": inventory_sha256,
                    "upstream_session_content_sha256": item[
                        "upstream_session_content_sha256"
                    ],
                    "scientific_eligible": False,
                }
            )
        final_payload, final_payload_sha256 = _stable_regular_file_payload(
            args.acquisition_manifest,
            label="version-3 acquisition reconstruction publication barrier",
        )
        if (
            final_payload != v3_reconstruction_payload
            or final_payload_sha256 != v3_reconstruction_file_sha256
        ):
            raise RuntimeError(
                "version-3 acquisition reconstruction bytes changed during cache build"
            )
        reconstruction_copy = output_root / "acquisition_reconstruction.json"
        _atomic_write_bytes(reconstruction_copy, final_payload)
        published_payload, published_sha256 = _stable_regular_file_payload(
            reconstruction_copy,
            label="published version-3 acquisition reconstruction snapshot",
        )
        if (
            published_payload != final_payload
            or published_sha256 != final_payload_sha256
        ):
            raise RuntimeError(
                "published version-3 reconstruction snapshot differs from consumed bytes"
            )
        root_acquisition_contract.update(
            {
                "reconstruction_manifest_sha256": published_sha256,
                "reconstruction_manifest_bytes": len(published_payload),
                "cache_usable_session_ids": list(cache_usable_ids),
                "cache_usable_session_ids_sha256": _canonical_value_sha256(
                    list(cache_usable_ids)
                ),
                "cache_inventory_aggregate_sha256": _canonical_value_sha256(
                    inventory_bindings
                ),
            }
        )
        root_manifest = {
            "schema_version": FEATURE_CACHE_ROOT_SCHEMA_V3,
            "content_sha256": "",
            "config_sha256": config_digest,
            "pipeline_sha256": pipeline_digest,
            "acquisition_contract": root_acquisition_contract,
            "sessions": root_items,
        }
        root_manifest["content_sha256"] = _canonical_content_sha256(root_manifest)
        _atomic_write_bytes(
            root_manifest_path,
            json.dumps(
                root_manifest,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
        )
        assert v3_final_output_root is not None
        assert _ACTIVE_V3_ATTEMPT is not None
        _secure_and_fsync_v3_tree(output_root)
        _publish_v3_directory_noreplace(output_root, v3_final_output_root)
        claim_path = Path(str(_ACTIVE_V3_ATTEMPT["claim_path"]))
        try:
            claim_path.unlink()
            parent_fd = os.open(
                v3_final_output_root.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except FileNotFoundError:
            pass
        built = [item for item in results if item["status"] == "ok"]
        print(
            f"Built {len(built)} diagnostic acquisition-v3 sessions, "
            f"{sum(int(item['window_count']) for item in built)} windows"
        )
        return 0
    root_manifest = {
        "schema_version": "snn_rr.feature_cache_root.v2",
        "dataset_root": str(dataset_root),
        "config": str(args.config.resolve()),
        "config_sha256": config_digest,
        "pipeline_sha256": pipeline_digest,
        "subjects_filter_applied": subjects_filter_applied,
        "sessions": merged_results,
    }
    if root_acquisition_contract is not None:
        assert acquisition is not None
        ok_contracts = [
            item.get("acquisition_contract")
            for item in merged_results
            if item.get("status") == "ok"
        ]
        merged_by_id = {
            str(item.get("session_id")): item for item in merged_results
        }
        expected_usable_ids = tuple(
            str(item)
            for item in root_acquisition_contract["expected_usable_session_ids"]
        )
        cache_usable_ids = tuple(
            session_id
            for session_id in expected_usable_ids
            if merged_by_id.get(session_id, {}).get("status") == "ok"
        )
        scope_contract = _derive_acquisition_cache_scope(
            subjects_filter_applied=subjects_filter_applied,
            reconstruction_full_cohort_complete=(
                acquisition.full_cohort_complete
            ),
            expected_usable_session_ids=expected_usable_ids,
            cache_usable_session_ids=cache_usable_ids,
        )
        full_cohort_complete = bool(scope_contract["full_cohort_complete"])
        inventory_bindings = {
            session_id: merged_by_id[session_id].get("inventory_sha256")
            for session_id in cache_usable_ids
        }
        root_acquisition_contract.update(
            {
                **scope_contract,
                "cache_usable_session_ids": list(cache_usable_ids),
                "cache_usable_session_ids_sha256": _canonical_value_sha256(
                    list(cache_usable_ids)
                ),
                "cache_inventory_aggregate_sha256": _canonical_value_sha256(
                    inventory_bindings
                ),
            }
        )
        root_acquisition_contract["scientific_eligible"] = bool(
            args.acquisition_mode == "strict"
            and not subjects_filter_applied
            and acquisition.scientific_eligible
            and full_cohort_complete
            and ok_contracts
            and all(
                isinstance(item, dict) and item.get("scientific_eligible") is True
                for item in ok_contracts
            )
        )
        root_manifest["acquisition_contract"] = root_acquisition_contract
    root_manifest["content_sha256"] = _canonical_content_sha256(root_manifest)
    root_manifest_path.write_text(
        json.dumps(root_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ok = [item for item in results if item["status"] == "ok"]
    print(
        f"Built {len(ok)} sessions, {sum(item['window_count'] for item in ok)} windows, "
        f"{sum(item['valid_reference_count'] for item in ok)} valid references"
    )
    return 0


def main() -> int:
    global _ACTIVE_V3_ATTEMPT
    try:
        result = _main_impl()
    except BaseException as error:
        try:
            _record_v3_attempt_failure(error)
        except BaseException as receipt_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    f"version-3 failure receipt could not be finalized: {receipt_error}"
                )
        finally:
            _ACTIVE_V3_ATTEMPT = None
        raise
    _ACTIVE_V3_ATTEMPT = None
    return result


if __name__ == "__main__":
    raise SystemExit(main())
