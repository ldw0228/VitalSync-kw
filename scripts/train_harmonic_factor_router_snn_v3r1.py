#!/usr/bin/env python3
"""Train or run target-free inference for the adaptive DHFER-SNN-v3r1.

The training path is deliberately validation-only: it fits a feature scaler on
the four outer-training identity folds, selects a checkpoint on the contracted
next validation fold, and never constructs the outer-test row set.  Promotion
inference is a separate mode which accepts a strict sanitized NPZ and cannot
load reference, identity, protocol, or quality-control fields.

This remains retrospective historical-cohort engineering.  Nothing written by
this program is evidence of prospective or commercial performance.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import stat
import sys
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:  # Prefer the authorized successor wrapper once it is installed.
    from snn_rr.harmonic_factor_router_models_v3r1 import (  # type: ignore
        DirectedHarmonicFactorExpertSNNV3R1 as DirectedHarmonicFactorExpertSNN,
        FACTOR_CLASSES,
        FEATURE_LAYOUT_SEMANTIC_SHA256,
        FactorRouterState,
    )
    from snn_rr.harmonic_feature_layout_v3r1 import (  # type: ignore
        EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
        OuterTrainFeatureStandardizer,
        build_structural_availability_mask,
        load_and_validate_feature_names,
    )
    MODEL_IMPORT_PATH = "snn_rr.harmonic_factor_router_models_v3r1"
    LAYOUT_IMPORT_PATH = "snn_rr.harmonic_feature_layout_v3r1"
except ModuleNotFoundError as import_error:  # Staged implementation fallback only.
    if import_error.name not in {
        "snn_rr.harmonic_factor_router_models_v3r1",
        "snn_rr.harmonic_feature_layout_v3r1",
    }:
        raise
    from snn_rr.harmonic_factor_router_v3 import (  # type: ignore
        DirectedHarmonicFactorExpertSNN,
        EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
        FACTOR_CLASSES,
        FEATURE_LAYOUT_SEMANTIC_SHA256,
        FactorRouterState,
    )

    MODEL_IMPORT_PATH = "snn_rr.harmonic_factor_router_v3"
    LAYOUT_IMPORT_PATH = "local_contract_compatible_fallback"

    def load_and_validate_feature_names(path: Path) -> tuple[str, ...]:
        document = json.loads(path.read_text(encoding="utf-8"))
        names = tuple(map(str, document.get("node_feature_names", ())))
        digest = semantic_sha256(list(names))
        if len(names) != 571 or digest != EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256:
            raise ValueError("feature_names.json is not the contracted 571-wide layout")
        return names

    def build_structural_availability_mask(
        candidate_rr_bpm: np.ndarray,
        candidate_mask: np.ndarray,
        joint_radar_mask: np.ndarray,
        *,
        rr_min_bpm: float = 6.0,
        rr_max_bpm: float = 45.0,
    ) -> np.ndarray:
        rr = np.asarray(candidate_rr_bpm, dtype=np.float32)
        candidates = np.asarray(candidate_mask, dtype=bool)
        radar = np.asarray(joint_radar_mask, dtype=bool)
        if rr.shape != candidates.shape or rr.ndim != 2:
            raise ValueError("candidate arrays must have shape [rows,K]")
        if radar.shape != (rr.shape[0], 3):
            raise ValueError("joint_radar_mask must have shape [rows,3]")
        node = candidates & radar.any(axis=-1, keepdims=True)
        ratios = np.asarray((0.25, 1 / 3, 0.5, 1, 2, 3, 4), np.float32)
        in_band = node[..., None] & (
            (rr[..., None] * ratios >= rr_min_bpm)
            & (rr[..., None] * ratios <= rr_max_bpm)
        )
        available = np.zeros((*rr.shape, 571), dtype=bool)
        available[..., :46] = node[..., None]
        rf = available[..., 46:424].reshape(*rr.shape, 3, 7, 2, 9)
        rf[..., 0, :] = (
            in_band[..., None, :, None] & radar[:, None, :, None, None]
        )
        svd = available[..., 424:].reshape(*rr.shape, 3, 7, 7)
        svd[...] = in_band[..., None, :, None] & radar[:, None, :, None, None]
        return available

    @dataclass(slots=True)
    class OuterTrainFeatureStandardizer:  # pragma: no cover - successor is normal path
        center: np.ndarray
        scale: np.ndarray
        fit_positions_sha256: str

        @classmethod
        def fit(
            cls,
            features: np.ndarray,
            availability: np.ndarray,
            *,
            fit_positions_sha256: str,
        ) -> "OuterTrainFeatureStandardizer":
            values = np.asarray(features, np.float32)
            mask = np.asarray(availability, bool)
            if values.shape != mask.shape or values.shape[-1] != 571:
                raise ValueError("scaler arrays must have matching [...,571] shape")
            center = np.zeros(571, np.float32)
            scale = np.ones(571, np.float32)
            for column in range(571):
                selected = values[..., column][mask[..., column]]
                selected = selected[np.isfinite(selected)]
                if selected.size:
                    center[column] = np.median(selected)
                    q25, q75 = np.percentile(selected, (25.0, 75.0))
                    scale[column] = max(float((q75 - q25) / 1.349), 1.0e-4)
            return cls(center, scale, fit_positions_sha256)

        def transform(self, features: np.ndarray, availability: np.ndarray) -> np.ndarray:
            values = np.nan_to_num(np.asarray(features, np.float32), copy=True)
            mask = np.asarray(availability, bool)
            transformed = np.clip((values - self.center) / self.scale, -8, 8)
            return np.where(mask, transformed, 0.0).astype(np.float32)

        def to_state(self) -> dict[str, Any]:
            return {
                "center": self.center.tolist(),
                "scale": self.scale.tolist(),
                "fit_positions_sha256": self.fit_positions_sha256,
            }

        @classmethod
        def from_state(cls, state: Mapping[str, Any]) -> "OuterTrainFeatureStandardizer":
            return cls(
                np.asarray(state["center"], np.float32),
                np.asarray(state["scale"], np.float32),
                str(state["fit_positions_sha256"]),
            )

        def state_receipt(self) -> dict[str, Any]:
            return {
                "state_sha256": semantic_sha256(self.to_state()),
                "fit_positions_sha256": self.fit_positions_sha256,
            }

    try:
        from snn_rr.harmonic_feature_layout_v3r1 import (  # type: ignore
            EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
            OuterTrainFeatureStandardizer,
            build_structural_availability_mask,
            load_and_validate_feature_names,
        )
        LAYOUT_IMPORT_PATH = "snn_rr.harmonic_feature_layout_v3r1"
    except ImportError:
        pass


SCHEMA_VERSION = 1
CAMPAIGN_ID = "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
CAMPAIGN_REVISION = "V8R4"
NONOUTER_PACK_CLASSIFICATION = (
    "adaptive_v3r1_v8r4_nonouter_training_validation_pack"
)
NONOUTER_STACK_CLASSIFICATION = (
    "adaptive_v3r1_v8r4_nonouter_causal_proposer_stack"
)
CONTRACT_PATH = PROJECT_ROOT / (
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json"
)
PROMOTION_SELECTION_PATH = PROJECT_ROOT / (
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "DISCOVERY_SELECTION_LOCK.json"
)
PROMOTION_AUTHORIZATION_PATH = PROJECT_ROOT / (
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "PROMOTION_AUTHORIZATION.json"
)
SELECTION_AUTHORITY_PATH = PROJECT_ROOT / "scripts/select_hfr_v3r1_common_variant.py"
CONFIG_PATH = PROJECT_ROOT / "configs/harmonic_factor_router_v3.yaml"
N_FOLDS = 6
RR_MIN_BPM = 6.0
RR_MAX_BPM = 45.0
RELEASE_MODES = ("raw_anchor", "hard_source_argmax", "fixed_confidence_switch")
VARIANTS = ("H0_no_factor", "H1_factor", "H2_full")
AMP_INITIAL_GRADIENT_SCALE = 8192.0
AMP_MINIMUM_GRADIENT_SCALE = 1.0
AMP_MAX_GROUP_RETRIES = 14
PREDICTION_BATCH_SESSIONS = 4
REQUIRED_CACHE_OUTPUTS: dict[str, str] = {
    "feature_names": "feature_names.json",
    "metadata": "metadata.csv",
    "node_features": "node_features.npy",
    "candidate_bpm": "candidate_bpm.npy",
    "candidate_mask": "candidate_mask.npy",
    "joint_radar_mask": "joint_radar_mask.npy",
    "local_to_global_cache_index": "local_to_global_cache_index.npy",
}
NONOUTER_METADATA_COLUMNS = (
    "cache_index",
    "fold",
    "session_id",
    "identity",
    "window_number",
    "rr_bpm",
    "reference_valid",
    "classical_rr_bpm",
)
NONOUTER_PACK_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "format_version",
        "complete",
        "outer_fold",
        "partition",
        "source_combined_cache_open_authorized_by_consumer",
        "outer_test_rows_physically_present",
        "outer_prediction_pack_absent",
        "inputs",
        "outputs",
        "content_sha256",
    }
)
PROMOTION_NONOUTER_PACK_MANIFEST_KEYS = frozenset(
    NONOUTER_PACK_MANIFEST_KEYS | {"promotion_scope", "promotion_authorization"}
)
PROMOTION_TRAINING_PACK_SCOPE = "promotion_training_pack"
DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION = {
    "campaign_revision": CAMPAIGN_REVISION,
    "physical_input_partition": "outer_excluded_training_validation_pack",
    "combined_target_bearing_cache_opened": False,
    "outer_test_rows_physically_present_in_training_pack": False,
    "outer_test_identity_or_classical_context_materialized": False,
    "outer_test_feature_values_materialized_or_forwarded": False,
    "outer_test_reference_fields_opened": False,
    "outer_test_model_or_evaluation_iterator_constructed": False,
    "numpy_row_access_audit_enforced": True,
    "outer_prediction_pack_absent_during_discovery": True,
    "commercial_claim_allowed": False,
}
SCIENTIFIC_SIGNATURE_ORCHESTRATION_FIELDS = frozenset(
    {
        "output_directory",
        "campaign_phase_label",
        "promotion_authorization_path",
        "release_mode",
        "resume_flag",
    }
)
RADAR_SUBSETS = (
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
)
COMMERCIAL_GATES: dict[str, tuple[str, float]] = {
    "overall_mae_bpm": ("maximum", 1.0),
    "identity_macro_mae_bpm": ("maximum", 1.0),
    "rmse_bpm": ("maximum", 1.8),
    "within_2_fraction": ("minimum", 0.90),
    "over_5_fraction": ("maximum", 0.03),
    "high_rr_25_35_mae_bpm": ("maximum", 2.0),
}
LOSS_WEIGHTS = {
    "listwise_kl": 1.0,
    "mixture_nll": 0.25,
    "component_smooth_l1": 0.30,
    "anchor_residual_smooth_l1": 0.50,
    "anchor_nll": 0.15,
    "factor_focal": 0.35,
    "wrong_harmonic_margin": 0.25,
    "factor_candidate_js": 0.10,
    "quality_bce": 0.10,
    "spike_rate": 0.005,
    "cvar20": 0.15,
}
PREDICT_INPUT_KEYS = frozenset(
    {
        "cache_index",
        "node_features",
        "candidate_rr_bpm",
        "candidate_mask",
        "joint_radar_mask",
        "proposer_anchor_bpm",
        "proposer_anchor_std_bpm",
        "proposer_anchor_available",
        "classical_rr_bpm",
        "session_reset",
    }
)
PREDICTION_KEYS = (
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_scientific_signature(
    complete_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the phase-independent V8 scientific configuration.

    V8 permits exactly five top-level orchestration fields to differ between
    discovery and promotion.  Keeping the stripping policy here makes pointer
    reuse independently re-hashable instead of trusting a producer-supplied
    digest.  Nested fields are never stripped.
    """

    actual = set(map(str, complete_configuration))
    missing = sorted(SCIENTIFIC_SIGNATURE_ORCHESTRATION_FIELDS - actual)
    if missing:
        raise ValueError(
            "scientific signature input lacks orchestration fields: "
            + ", ".join(missing)
        )
    scientific = {
        str(key): copy.deepcopy(value)
        for key, value in complete_configuration.items()
        if str(key) not in SCIENTIFIC_SIGNATURE_ORCHESTRATION_FIELDS
    }
    if any(key in scientific for key in SCIENTIFIC_SIGNATURE_ORCHESTRATION_FIELDS):
        raise RuntimeError("scientific signature orchestration stripping failed")
    # Canonical JSON conversion rejects non-finite values now, at issuance.
    semantic_sha256(scientific)
    return _json_ready(scientific)


def scientific_signature_sha256(scientific_signature: Mapping[str, Any]) -> str:
    """Canonical re-hash helper used by discovery/promotion pointer validators."""

    forbidden = sorted(
        SCIENTIFIC_SIGNATURE_ORCHESTRATION_FIELDS & set(map(str, scientific_signature))
    )
    if forbidden:
        raise ValueError(
            "scientific signature contains orchestration fields: "
            + ", ".join(forbidden)
        )
    return semantic_sha256(dict(scientific_signature))


_KILL_SAFE_OUTPUT_FILENAMES = frozenset(
    {
        "run_manifest.json",
        "scaler.json",
        "best.pt",
        "last.pt",
        "history.json",
        "validation_predictions.npz",
        "validation_metrics.json",
        "checkpoint_selection_lock.json",
        "predictions.npz",
        "prediction_manifest.json",
    }
)
_TRAIN_COMPLETED_OUTPUT_FILENAMES = frozenset(
    {
        "run_manifest.json",
        "scaler.json",
        "best.pt",
        "last.pt",
        "history.json",
        "validation_predictions.npz",
        "validation_metrics.json",
        "checkpoint_selection_lock.json",
    }
)
_KILL_SAFE_TEMP_TOKEN = r"[0-9a-f]{32}"
_KILL_SAFE_TEMP_PATTERN = re.compile(
    r"^\.([^/]+)\.v8r4a-tmp-(" + _KILL_SAFE_TEMP_TOKEN + r")$"
)
_FAULT_INJECTION_HOOK: Any | None = None


def _fault_inject(point: str) -> None:
    """Invoke the process-local deterministic crash-test hook, when installed."""

    hook = _FAULT_INJECTION_HOOK
    if hook is not None:
        hook(point)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise RuntimeError("short write to V8R4A immutable temporary")
        offset += written


def _open_output_parent(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path.parent, flags)


def _validate_replaceable_destination(parent_fd: int, name: str) -> None:
    try:
        status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not (
        stat.S_ISREG(status.st_mode)
        and status.st_nlink == 1
        and stat.S_IMODE(status.st_mode) == 0o444
    ):
        raise RuntimeError(
            f"refusing to replace mutable or aliased output artifact: {name}"
        )


def _atomic_publish_immutable(
    path: Path, writer: Any
) -> dict[str, Any]:
    """Publish one create-or-replace artifact whose inode is born 0444."""

    if path.name not in _KILL_SAFE_OUTPUT_FILENAMES:
        raise RuntimeError(f"unregistered kill-safe output filename: {path.name}")
    parent_fd = _open_output_parent(path)
    temporary = f".{path.name}.v8r4a-tmp-{secrets.token_hex(16)}"
    descriptor = -1
    published = False
    published_binding: dict[str, Any] | None = None
    try:
        _validate_replaceable_destination(parent_fd, path.name)
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        previous_umask = os.umask(0)
        try:
            descriptor = os.open(temporary, flags, 0o444, dir_fd=parent_fd)
        finally:
            os.umask(previous_umask)
        born = os.fstat(descriptor)
        if not (
            stat.S_ISREG(born.st_mode)
            and born.st_nlink == 1
            and stat.S_IMODE(born.st_mode) == 0o444
        ):
            raise RuntimeError("V8R4A temporary was not born immutable")
        writer(descriptor)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        committed = os.fstat(descriptor)
        if not (
            stat.S_ISREG(committed.st_mode)
            and committed.st_nlink == 1
            and stat.S_IMODE(committed.st_mode) == 0o444
        ):
            raise RuntimeError("V8R4A temporary metadata drifted while writing")
        _fault_inject("trainer_atomic_after_payload_fsync")
        os.replace(
            temporary,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        published = True
        _fault_inject("trainer_atomic_after_replace_before_directory_fsync")
        os.fsync(parent_fd)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not (
            stat.S_ISREG(named.st_mode)
            and named.st_nlink == 1
            and stat.S_IMODE(named.st_mode) == 0o444
            and (named.st_dev, named.st_ino)
            == (committed.st_dev, committed.st_ino)
        ):
            raise RuntimeError("published V8R4A output binding drifted")
        published_binding = {
            "path": str(path.expanduser().absolute()),
            "sha256": digest.hexdigest(),
            "bytes": int(committed.st_size),
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        # A pathname is not an ownership capability.  On failure an attacker
        # can rename the descriptor-owned inode and install an unrelated file
        # at ``temporary``; unlinking by name would then delete caller data.
        # Preserve every unpublished residue for explicit quarantine.
        os.close(parent_fd)
    if published_binding is None:  # pragma: no cover - successful replace sets it.
        raise RuntimeError("V8R4A publication did not return its committed binding")
    return published_binding


def cleanup_stale_atomic_temporaries(
    output_dir: Path,
    *,
    protected_paths: Iterable[Path] = (),
) -> tuple[str, ...]:
    """Detect orphan temporaries without claiming ownership of their bytes.

    A filename-shaped 0444 file is not an ownership capability.  No
    cross-invocation pathname can be deleted safely in a same-user mutable
    directory, so every match is preserved for explicit quarantine.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(output_dir, flags)
    protected = {
        path.expanduser().absolute() for path in protected_paths
    }
    orphans: list[str] = []
    try:
        for name in sorted(os.listdir(directory_fd)):
            match = _KILL_SAFE_TEMP_PATTERN.fullmatch(name)
            if match is None or match.group(1) not in _KILL_SAFE_OUTPUT_FILENAMES:
                continue
            candidate = (output_dir / name).expanduser().absolute()
            if candidate in protected:
                raise RuntimeError(
                    f"stale temporary aliases a protected input; no files were deleted: {name}"
                )
            status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not (
                stat.S_ISREG(status.st_mode)
                and status.st_nlink == 1
                and stat.S_IMODE(status.st_mode) == 0o444
            ):
                raise RuntimeError(
                    f"unsafe stale V8R4A temporary requires quarantine: {name}"
                )
            orphans.append(name)
    finally:
        os.close(directory_fd)
    if orphans:
        raise RuntimeError(
            "orphan V8R4A temporaries require explicit quarantine; no files "
            f"were deleted: {orphans}"
        )
    return ()


def require_immutable_output_artifact(path: Path) -> None:
    try:
        status = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(f"required output artifact is unavailable: {path.name}") from error
    if not (
        stat.S_ISREG(status.st_mode)
        and status.st_nlink == 1
        and stat.S_IMODE(status.st_mode) == 0o444
    ):
        raise RuntimeError(f"output artifact is mutable or aliased: {path.name}")


def _immutable_output_binding(path: Path) -> dict[str, Any]:
    """Return the exact completed-run binding for one immutable artifact."""

    require_immutable_output_artifact(path)
    binding, _ = _verified_regular_file(path, required_mode=0o444)
    return {
        "sha256": binding["sha256"],
        "bytes": binding["bytes"],
        "mode": "0444",
        "nlink": 1,
    }


def _validate_completed_output_inventory(
    output_dir: Path, lock: Mapping[str, Any]
) -> None:
    """Reject any unknown, missing, mutable, or aliased completed-run entry."""

    observed = set(os.listdir(output_dir))
    if observed != _TRAIN_COMPLETED_OUTPUT_FILENAMES:
        raise RuntimeError(
            "completed output inventory drifted: "
            f"missing={sorted(_TRAIN_COMPLETED_OUTPUT_FILENAMES - observed)}, "
            f"unknown={sorted(observed - _TRAIN_COMPLETED_OUTPUT_FILENAMES)}"
        )
    expected_artifacts = _TRAIN_COMPLETED_OUTPUT_FILENAMES - {
        "checkpoint_selection_lock.json"
    }
    inventory = lock.get("completed_output_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != expected_artifacts:
        raise RuntimeError("completed output binding inventory schema drifted")
    for filename in sorted(expected_artifacts):
        recorded = inventory.get(filename)
        if not isinstance(recorded, Mapping) or set(recorded) != {
            "sha256",
            "bytes",
            "mode",
            "nlink",
        }:
            raise RuntimeError(
                f"completed output binding schema drifted: {filename}"
            )
        observed_binding = _immutable_output_binding(output_dir / filename)
        if dict(recorded) != observed_binding:
            raise RuntimeError(
                f"completed output binding drifted: {filename}"
            )
    require_immutable_output_artifact(
        output_dir / "checkpoint_selection_lock.json"
    )


def atomic_write_json(
    path: Path, value: Any, *, immutable: bool = True
) -> dict[str, Any]:
    del immutable  # V8R4A never publishes mutable artifacts.
    payload = (
        json.dumps(
            _json_ready(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return _atomic_publish_immutable(
        path, lambda descriptor: _write_all(descriptor, payload)
    )


def atomic_torch_save(path: Path, value: Any) -> dict[str, Any]:
    def writer(descriptor: int) -> None:
        duplicate = os.dup(descriptor)
        with os.fdopen(duplicate, "wb", closefd=True) as stream:
            torch.save(value, stream)
            stream.flush()

    return _atomic_publish_immutable(path, writer)


def atomic_save_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    immutable: bool = True,
) -> dict[str, Any]:
    del immutable  # V8R4A never publishes mutable artifacts.

    def writer(descriptor: int) -> None:
        duplicate = os.dup(descriptor)
        with os.fdopen(duplicate, "wb", closefd=True) as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()

    return _atomic_publish_immutable(path, writer)


def _positions_sha256(positions: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(positions, dtype=np.int64))
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


@dataclass(slots=True)
class RowAccessAudit:
    """Fail-closed row-access evidence for physically nonouter arrays.

    Opening an mmap is not represented as a row access.  Every value-bearing
    first-axis selection must pass through :class:`AuditedRowArray`, which
    resolves the selector to local rows, maps those rows to canonical cache
    indexes/folds, and refuses any outer-fold row.  An implicit whole-array
    conversion is always rejected because it would bypass that evidence.
    """

    cache_index: np.ndarray
    fold: np.ndarray
    outer_fold: int
    accessed_cache_indexes: set[int] = field(default_factory=set)
    accesses_by_array: dict[str, int] = field(default_factory=dict)
    selected_rows_by_array: dict[str, int] = field(default_factory=dict)
    implicit_whole_array_conversions: int = 0
    outer_row_access_attempts: int = 0

    def __post_init__(self) -> None:
        self.cache_index = np.asarray(self.cache_index, dtype=np.int64).copy()
        self.fold = np.asarray(self.fold, dtype=np.int16).copy()
        if (
            self.cache_index.ndim != 1
            or self.fold.shape != self.cache_index.shape
            or len(self.cache_index) == 0
            or np.any(self.fold == int(self.outer_fold))
        ):
            raise RuntimeError("row-access audit requires a nonempty physical nonouter pack")

    def _positions(self, selector: Any) -> np.ndarray:
        size = len(self.cache_index)
        if isinstance(selector, slice):
            return np.arange(size, dtype=np.int64)[selector]
        if isinstance(selector, (int, np.integer)) and not isinstance(selector, (bool, np.bool_)):
            value = int(selector)
            if value < 0:
                value += size
            if value < 0 or value >= size:
                raise IndexError("audited row selector is out of bounds")
            return np.asarray([value], dtype=np.int64)
        values = np.asarray(selector)
        if values.ndim == 0:
            raise TypeError("audited row selector must identify first-axis rows")
        if values.dtype.kind == "b":
            if values.ndim != 1 or values.shape != (size,):
                raise IndexError("audited boolean selector must cover the complete local axis")
            return np.flatnonzero(values)
        if values.dtype.kind not in "iu" or values.ndim != 1:
            raise TypeError("audited row selector must be a one-dimensional integer index")
        positions = values.astype(np.int64, copy=True)
        positions[positions < 0] += size
        if np.any((positions < 0) | (positions >= size)):
            raise IndexError("audited row selector is out of bounds")
        return positions

    def record(self, name: str, selector: Any) -> None:
        positions = self._positions(selector)
        selected_folds = self.fold[positions]
        if np.any(selected_folds == int(self.outer_fold)):
            self.outer_row_access_attempts += 1
            raise RuntimeError("outer-test feature row access was attempted")
        self.accesses_by_array[name] = self.accesses_by_array.get(name, 0) + 1
        self.selected_rows_by_array[name] = (
            self.selected_rows_by_array.get(name, 0) + int(len(positions))
        )
        self.accessed_cache_indexes.update(map(int, self.cache_index[positions]))

    def reject_implicit_conversion(self) -> None:
        self.implicit_whole_array_conversions += 1
        raise RuntimeError("implicit whole-array conversion bypasses the V8R4 row audit")

    def snapshot(self) -> dict[str, Any]:
        indexes = np.asarray(sorted(self.accessed_cache_indexes), dtype=np.int64)
        return {
            "campaign_revision": CAMPAIGN_REVISION,
            "outer_fold": int(self.outer_fold),
            "physical_pack_rows": int(len(self.cache_index)),
            "outer_rows_in_physical_pack": 0,
            "outer_row_access_attempts": int(self.outer_row_access_attempts),
            "implicit_whole_array_conversions": int(
                self.implicit_whole_array_conversions
            ),
            "accesses_by_array": {
                key: int(self.accesses_by_array[key])
                for key in sorted(self.accesses_by_array)
            },
            "selected_rows_by_array": {
                key: int(self.selected_rows_by_array[key])
                for key in sorted(self.selected_rows_by_array)
            },
            "unique_accessed_cache_indexes": int(len(indexes)),
            "accessed_cache_indexes_sha256": _positions_sha256(indexes),
        }


class AuditedRowArray:
    """Read-only first-axis proxy that makes every value access auditable."""

    __slots__ = ("_array", "_audit", "_name")

    def __init__(self, array: np.ndarray, audit: RowAccessAudit, name: str) -> None:
        self._array = array
        self._audit = audit
        self._name = str(name)
        if len(array) != len(audit.cache_index):
            raise RuntimeError(f"audited array row count drifted: {name}")

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._array.dtype

    @property
    def ndim(self) -> int:
        return self._array.ndim

    def __len__(self) -> int:
        return len(self._array)

    def __getitem__(self, key: Any) -> np.ndarray | np.generic:
        row_selector = key[0] if isinstance(key, tuple) else key
        if row_selector is Ellipsis or row_selector is None:
            self._audit.reject_implicit_conversion()
        self._audit.record(self._name, row_selector)
        return self._array[key]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        del dtype, copy
        self._audit.reject_implicit_conversion()


@dataclass(slots=True)
class Experiment:
    root: Path
    manifest: dict[str, Any]
    cache_input_binding: dict[str, Any]
    feature_names: tuple[str, ...]
    metadata: pd.DataFrame
    node_features: AuditedRowArray
    candidate_rr: AuditedRowArray
    candidate_mask: AuditedRowArray
    joint_radar_mask: AuditedRowArray
    anchor_rr: AuditedRowArray
    anchor_std: AuditedRowArray
    anchor_available: AuditedRowArray
    proposer_stack: Path
    row_access_audit: RowAccessAudit
    consumed_source_files: dict[str, dict[str, Any]]


@dataclass(slots=True)
class InferenceArrays:
    cache_index: np.ndarray
    node_features: np.ndarray
    candidate_rr: np.ndarray
    candidate_mask: np.ndarray
    joint_radar_mask: np.ndarray
    anchor_rr: np.ndarray
    anchor_std: np.ndarray
    anchor_available: np.ndarray
    classical_rr: np.ndarray
    session_reset: np.ndarray


@dataclass(slots=True)
class PredictionBundle:
    cache_index: np.ndarray
    raw_anchor_bpm: np.ndarray
    raw_anchor_available: np.ndarray
    hard_source_bpm: np.ndarray
    hard_source_available: np.ndarray
    fixed_confidence_switch_bpm: np.ndarray
    fixed_confidence_switch_available: np.ndarray
    selected_source_probability: np.ndarray
    selected_source_code: np.ndarray
    source_scale_bpm: np.ndarray
    quality: np.ndarray
    factor_probabilities: np.ndarray
    spike_rate: np.ndarray


def _load_contract() -> dict[str, Any]:
    _, raw = _capture_verified_file(CONTRACT_PATH)
    document = _strict_json_bytes(raw, CONTRACT_PATH)
    if document.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("adaptive v3r1 contract campaign binding drifted")
    if not document.get("implementation_authorization", {}).get("authorized_now"):
        raise RuntimeError("v3r1 implementation is not authorized")
    return document


def _strict_json_bytes(raw: bytes, path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeError(f"non-finite JSON constant in {path}: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON document: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _verified_regular_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    capture_bytes: bool = False,
    required_mode: int | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    """Hash one exact, unaliased regular-file inode through ``O_NOFOLLOW``.

    The pre/open/post stat sandwich rejects final-component symlinks, hard-link
    aliases, path replacement, truncation, and in-place mutation while the
    digest is being computed.  Callers must use the returned digest rather
    than re-reading a producer-supplied manifest value.
    """

    source = path.expanduser().absolute()
    if expected_sha256 is not None and not (
        isinstance(expected_sha256, str)
        and len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256)
    ):
        raise RuntimeError(f"cache binding SHA-256 is malformed: {source}")
    if expected_bytes is not None and (
        type(expected_bytes) is not int or expected_bytes < 0
    ):
        raise RuntimeError(f"cache binding byte count is malformed: {source}")
    try:
        before_path = os.stat(source, follow_symlinks=False)
        descriptor = os.open(
            source,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise RuntimeError(f"cannot open verified cache input: {source}") from error
    try:
        before_fd = os.fstat(descriptor)
        if not stat.S_ISREG(before_fd.st_mode) or before_fd.st_nlink != 1:
            raise RuntimeError(f"cache input must be a single-link regular file: {source}")
        if required_mode is not None and stat.S_IMODE(before_fd.st_mode) != required_mode:
            raise RuntimeError(
                f"cache input mode must be {required_mode:04o}: {source}"
            )
        if (
            stat.S_ISLNK(before_path.st_mode)
            or before_path.st_nlink != 1
            or (before_path.st_dev, before_path.st_ino)
            != (before_fd.st_dev, before_fd.st_ino)
        ):
            raise RuntimeError(f"cache input path/inode binding is unsafe: {source}")
        digest = hashlib.sha256()
        captured = bytearray() if capture_bytes else None
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if captured is not None:
                captured.extend(block)
        after_fd = os.fstat(descriptor)
        after_path = os.stat(source, follow_symlinks=False)
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
            raise RuntimeError(f"cache input changed while it was verified: {source}")
        actual_sha256 = digest.hexdigest()
        actual_bytes = int(after_fd.st_size)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise RuntimeError(f"cache input SHA-256 drifted: {source}")
        if expected_bytes is not None and actual_bytes != expected_bytes:
            raise RuntimeError(f"cache input byte count drifted: {source}")
        return (
            {
                "path": str(source),
                "sha256": actual_sha256,
                "bytes": actual_bytes,
            },
            None if captured is None else bytes(captured),
        )
    finally:
        os.close(descriptor)


def _capture_verified_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    required_mode: int | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Return the one byte string that was hashed and will be consumed.

    Validation followed by a second pathname open is an ABA boundary.  This
    helper deliberately couples provenance and parsing: the parser/loaders
    below receive only the captured bytes whose digest was checked through the
    stable ``O_NOFOLLOW`` descriptor.
    """

    binding, raw = _verified_regular_file(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        capture_bytes=True,
        required_mode=required_mode,
    )
    if raw is None:  # pragma: no cover - capture_bytes is a local invariant.
        raise RuntimeError(f"verified byte capture failed: {path}")
    return binding, raw


def _load_torch_bytes(
    raw: bytes,
    *,
    source: Path,
    map_location: Any,
) -> Mapping[str, Any]:
    """Deserialize a provenanced checkpoint without general pickle loading."""

    try:
        value = torch.load(
            io.BytesIO(raw),
            map_location=map_location,
            weights_only=True,
        )
    except Exception as error:
        raise RuntimeError(
            f"checkpoint is not a weights-only provenanced payload: {source}"
        ) from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"checkpoint payload must be a mapping: {source}")
    return value


def _load_torch_snapshot(
    path: Path,
    *,
    map_location: Any,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    required_mode: int | None = None,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    binding, raw = _capture_verified_file(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        required_mode=required_mode,
    )
    return (
        _load_torch_bytes(raw, source=path, map_location=map_location),
        binding,
    )


def _assert_file_binding_current(binding: Mapping[str, Any]) -> None:
    """Fail when a pathname no longer names the exact consumed source bytes."""

    if set(binding) != {"path", "sha256", "bytes"}:
        raise RuntimeError("consumed input binding schema drifted")
    current, _ = _verified_regular_file(
        Path(str(binding["path"])),
        expected_sha256=str(binding["sha256"]),
        expected_bytes=int(binding["bytes"]),
    )
    if current != dict(binding):
        raise RuntimeError(f"consumed input path binding drifted: {binding['path']}")


def _assert_file_bindings_current(bindings: Mapping[str, Mapping[str, Any]]) -> None:
    for name in sorted(bindings):
        try:
            _assert_file_binding_current(bindings[name])
        except (OSError, RuntimeError) as error:
            raise RuntimeError(f"consumed source drifted before publication: {name}") from error


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _json_ready(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeError("reuse context is not canonical JSON") from error


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Strict JSON equality (in particular, never alias ``True`` with ``1``)."""

    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def verify_bound_regular_file(
    path: Path, *, expected_sha256: str, expected_bytes: int
) -> dict[str, Any]:
    """Public exact-file verifier for proposer/index campaign inputs."""

    binding, _ = _verified_regular_file(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    return binding


def verify_cache_manifest_outputs(
    cache_root: Path,
    *,
    outer_fold: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify and canonically bind every cache byte consumed by this trainer.

    This public helper is also the campaign/benchmark preflight API.  It must
    run before any reference CSV field or NumPy mmap is opened.
    """

    root = cache_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest_binding, raw = _verified_regular_file(
        manifest_path, capture_bytes=True
    )
    assert raw is not None
    document = _strict_json_bytes(raw, manifest_path)
    manifest_keys = frozenset(document)
    if manifest_keys not in {
        NONOUTER_PACK_MANIFEST_KEYS,
        PROMOTION_NONOUTER_PACK_MANIFEST_KEYS,
    }:
        raise RuntimeError("V8R4 nonouter pack manifest schema drifted")
    if not (
        document.get("schema_version") == 1
        and document.get("classification") == NONOUTER_PACK_CLASSIFICATION
        and document.get("campaign_id") == CAMPAIGN_ID
        and document.get("campaign_revision") == CAMPAIGN_REVISION
        and document.get("complete") is True
        and type(document.get("format_version")) is int
        and document.get("format_version") == 1
        and document.get("outer_fold") == int(outer_fold)
        and document.get("partition") == "outer_excluded_training_validation"
        and document.get("source_combined_cache_open_authorized_by_consumer") is False
        and document.get("outer_test_rows_physically_present") is False
    ):
        raise RuntimeError("V8R4 nonouter pack manifest is incomplete or incompatible")
    if document.get("outer_prediction_pack_absent") is not True:
        raise RuntimeError("outer prediction pack must remain absent during discovery")
    promotion_binding: dict[str, Any] | None = None
    if manifest_keys == PROMOTION_NONOUTER_PACK_MANIFEST_KEYS:
        record = document.get("promotion_authorization")
        if not (
            document.get("promotion_scope") == PROMOTION_TRAINING_PACK_SCOPE
            and isinstance(record, Mapping)
            and set(record) == {"path", "sha256", "bytes"}
            and isinstance(record.get("path"), str)
            and record.get("path")
            and isinstance(record.get("sha256"), str)
            and len(str(record.get("sha256"))) == 64
            and all(
                character in "0123456789abcdef"
                for character in str(record.get("sha256"))
            )
            and type(record.get("bytes")) is int
            and int(record["bytes"]) > 0
        ):
            raise RuntimeError("promotion training pack authority binding schema drifted")
        authority_path = Path(str(record["path"])).expanduser().absolute()
        if authority_path != PROMOTION_AUTHORIZATION_PATH.resolve():
            raise RuntimeError(
                "promotion training pack authority path is not canonical"
            )
        promotion_binding, promotion_raw = _verified_regular_file(
            authority_path,
            expected_sha256=str(record["sha256"]),
            expected_bytes=int(record["bytes"]),
            capture_bytes=True,
            required_mode=0o444,
        )
        if promotion_binding != dict(record):
            raise RuntimeError("promotion training pack authority binding drifted")
        assert promotion_raw is not None
        promotion_document = _strict_json_bytes(promotion_raw, authority_path)
        scopes = promotion_document.get("authorized_scopes")
        if not (
            promotion_document.get("schema_version") == 1
            and promotion_document.get("classification")
            == "adaptive_v3r1_v8r4_promotion_authorization"
            and promotion_document.get("campaign_id") == CAMPAIGN_ID
            and promotion_document.get("campaign_revision") == CAMPAIGN_REVISION
            and promotion_document.get("authorized_now") is True
            and isinstance(scopes, list)
            and len(scopes) == len(set(scopes))
            and all(isinstance(scope, str) and scope for scope in scopes)
            and PROMOTION_TRAINING_PACK_SCOPE in scopes
            and promotion_document.get("training_authorized") is True
            and promotion_document.get("promotion_authorized") is True
            and promotion_document.get("outer_test_targets_authorized") is False
            and promotion_document.get("commercial_claim_authorized") is False
            and promotion_document.get("content_sha256")
            == semantic_sha256(
                {
                    key: value
                    for key, value in promotion_document.items()
                    if key != "content_sha256"
                }
            )
        ):
            raise RuntimeError(
                "promotion training pack authority document is incompatible"
            )
    inputs = document.get("inputs")
    proposer_record = inputs.get("proposer_stack") if isinstance(inputs, Mapping) else None
    if not (
        isinstance(inputs, Mapping)
        and set(inputs) == {"source_combined_cache", "proposer_stack"}
        and isinstance(inputs.get("source_combined_cache"), Mapping)
        and set(inputs["source_combined_cache"]) == {"sha256", "bytes"}
        and isinstance(proposer_record, Mapping)
        and set(proposer_record) == {"sha256", "bytes"}
        and isinstance(proposer_record.get("sha256"), str)
        and len(str(proposer_record.get("sha256"))) == 64
        and type(proposer_record.get("bytes")) is int
        and int(proposer_record["bytes"]) > 0
    ):
        raise RuntimeError("V8R4 nonouter pack input binding schema drifted")
    content_sha256 = document.get("content_sha256")
    if not (
        isinstance(content_sha256, str)
        and len(content_sha256) == 64
        and content_sha256
        == semantic_sha256(
            {key: value for key, value in document.items() if key != "content_sha256"}
        )
    ):
        raise RuntimeError("cache manifest canonical content hash drifted")
    outputs = document.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError("cache manifest lacks its output bindings")
    verified_outputs: dict[str, Any] = {}
    for logical_name, required_filename in REQUIRED_CACHE_OUTPUTS.items():
        record = outputs.get(logical_name)
        if not isinstance(record, Mapping) or set(record) != {
            "filename",
            "sha256",
            "bytes",
        }:
            raise RuntimeError(
                f"cache manifest output binding schema drifted: {logical_name}"
            )
        filename = record.get("filename")
        expected_sha256 = record.get("sha256")
        expected_bytes = record.get("bytes")
        if filename != required_filename:
            raise RuntimeError(
                f"cache manifest output filename drifted: {logical_name}"
            )
        binding, _ = _verified_regular_file(
            root / required_filename,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
        verified_outputs[logical_name] = {
            "filename": required_filename,
            "sha256": binding["sha256"],
            "bytes": binding["bytes"],
        }
    canonical_binding = {
        "manifest": manifest_binding,
        "outputs": verified_outputs,
    }
    if promotion_binding is not None:
        canonical_binding.update(
            {
                "promotion_scope": PROMOTION_TRAINING_PACK_SCOPE,
                "promotion_authorization": promotion_binding,
            }
        )
    # Assert JSON canonicalizability at the trust boundary used by reuse.
    semantic_sha256(canonical_binding)
    return document, canonical_binding


def _required_metadata_columns() -> set[str]:
    return set(NONOUTER_METADATA_COLUMNS)


def _scalar_from_npz(archive: np.lib.npyio.NpzFile, name: str) -> Any:
    if name not in archive.files:
        raise RuntimeError(f"proposer stack lacks {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise RuntimeError(f"proposer stack {name} must be scalar")
    return value.item()


def load_experiment(
    cache_root: Path,
    proposer_stack: Path,
    *,
    outer_fold: int,
    seed: int,
    reference_excluded_folds: Iterable[int] | None = None,
) -> Experiment:
    root = cache_root.expanduser().resolve()
    stack_path = proposer_stack.expanduser().resolve()
    # This is deliberately the first cache operation.  V8R4 consumers accept
    # only a physically outer-excluded pack, and verify every byte in that
    # pack before opening CSV content, references, or feature mmaps.
    manifest, cache_input_binding = verify_cache_manifest_outputs(
        root, outer_fold=int(outer_fold)
    )
    proposer_record = manifest["inputs"]["proposer_stack"]
    proposer_binding, proposer_raw = _capture_verified_file(
        stack_path,
        expected_sha256=str(proposer_record["sha256"]),
        expected_bytes=int(proposer_record["bytes"]),
    )
    cache_input_binding["proposer_stack"] = proposer_binding

    # Every payload is captured through the same descriptor that establishes
    # its digest.  All parsers below consume these private byte strings only;
    # no validated source pathname is reopened.
    consumed_source_files: dict[str, dict[str, Any]] = {
        "manifest": dict(cache_input_binding["manifest"]),
        "proposer_stack": dict(proposer_binding),
    }
    payloads: dict[str, bytes] = {}
    for logical_name, filename in REQUIRED_CACHE_OUTPUTS.items():
        record = cache_input_binding["outputs"][logical_name]
        binding, raw = _capture_verified_file(
            root / filename,
            expected_sha256=str(record["sha256"]),
            expected_bytes=int(record["bytes"]),
        )
        payloads[logical_name] = raw
        consumed_source_files[logical_name] = binding

    feature_path = root / REQUIRED_CACHE_OUTPUTS["feature_names"]
    feature_document = _strict_json_bytes(payloads["feature_names"], feature_path)
    names = feature_document.get("node_feature_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise RuntimeError("feature_names.json node_feature_names schema drifted")
    feature_names = tuple(names)
    if (
        len(feature_names) != 571
        or len(set(feature_names)) != len(feature_names)
        or semantic_sha256(list(feature_names))
        != EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
    ):
        raise RuntimeError("DHFER-SNN-v3r1 feature-name contract drifted")
    metadata_path = root / "metadata.csv"
    metadata_raw = payloads["metadata"]
    header = pd.read_csv(io.BytesIO(metadata_raw), nrows=0)
    if tuple(map(str, header.columns)) != NONOUTER_METADATA_COLUMNS:
        raise RuntimeError("V8R4 nonouter metadata has an unexpected column schema")

    # Establish the physical exclusion boundary before asking the CSV parser
    # for identity, classical context, or reference columns.  This is a real
    # reader boundary, not a post-read row filter.
    topology = pd.read_csv(
        io.BytesIO(metadata_raw), usecols=["cache_index", "fold"]
    )
    cache_index = topology["cache_index"].to_numpy(np.int64, copy=True)
    folds = topology["fold"].to_numpy(np.int16, copy=True)
    if (
        cache_index.ndim != 1
        or len(cache_index) == 0
        or np.any(cache_index < 0)
        or np.any(np.diff(cache_index) <= 0)
    ):
        raise RuntimeError("global cache_index must be unique, increasing, and row-ordered")
    if (
        np.any((folds < 0) | (folds >= N_FOLDS))
        or np.any(folds == int(outer_fold))
        or set(map(int, folds)) != (set(range(N_FOLDS)) - {int(outer_fold)})
    ):
        raise RuntimeError("V8R4 training pack is not the exact physical nonouter fold cover")
    local_to_global = np.load(
        io.BytesIO(payloads["local_to_global_cache_index"]),
        allow_pickle=False,
    )
    if not (
        local_to_global.dtype == np.dtype("int64")
        and local_to_global.shape == cache_index.shape
        and np.array_equal(np.asarray(local_to_global), cache_index)
    ):
        raise RuntimeError("V8R4 local-to-global cache-index proof drifted")

    context_columns = [
        name
        for name in NONOUTER_METADATA_COLUMNS
        if name not in {"rr_bpm", "reference_valid"}
    ]
    metadata = pd.read_csv(io.BytesIO(metadata_raw), usecols=context_columns)
    if not (
        np.array_equal(metadata["cache_index"].to_numpy(np.int64), cache_index)
        and np.array_equal(metadata["fold"].to_numpy(np.int16), folds)
    ):
        raise RuntimeError("V8R4 metadata context read changed physical row topology")
    excluded_folds = (
        {int(outer_fold)}
        if reference_excluded_folds is None
        else {int(value) for value in reference_excluded_folds} | {int(outer_fold)}
    )
    if not excluded_folds <= set(range(N_FOLDS)):
        raise ValueError("reference_excluded_folds must be valid fold indices")
    excluded_reference_rows = np.flatnonzero(np.isin(folds, sorted(excluded_folds)))
    skipped_csv_lines = frozenset(map(int, excluded_reference_rows + 1))
    labels = pd.read_csv(
        io.BytesIO(metadata_raw),
        usecols=["cache_index", "rr_bpm", "reference_valid"],
        skiprows=lambda line: int(line) in skipped_csv_lines,
    )
    expected_non_test = np.flatnonzero(~np.isin(folds, sorted(excluded_folds)))
    if not np.array_equal(
        labels["cache_index"].to_numpy(np.int64), cache_index[expected_non_test]
    ):
        raise RuntimeError("non-test reference loader did not preserve cache_index topology")
    metadata["rr_bpm"] = np.nan
    metadata["reference_valid"] = False
    metadata.loc[expected_non_test, "rr_bpm"] = labels["rr_bpm"].to_numpy(np.float64)
    metadata.loc[expected_non_test, "reference_valid"] = labels[
        "reference_valid"
    ].astype(bool).to_numpy()
    node = np.load(io.BytesIO(payloads["node_features"]), allow_pickle=False)
    candidate_rr = np.load(io.BytesIO(payloads["candidate_bpm"]), allow_pickle=False)
    candidate_mask = np.load(io.BytesIO(payloads["candidate_mask"]), allow_pickle=False)
    radar = np.load(io.BytesIO(payloads["joint_radar_mask"]), allow_pickle=False)
    if node.shape != (*candidate_rr.shape, 571):
        raise RuntimeError("cache node/candidate shape or feature width drifted")
    if candidate_mask.shape != candidate_rr.shape or radar.shape != (len(metadata), 3):
        raise RuntimeError("cache forward array shapes disagree")
    if len(metadata) != candidate_rr.shape[0] or candidate_rr.shape[1] > 12:
        raise RuntimeError("cache row or maximum-candidate contract drifted")

    with np.load(io.BytesIO(proposer_raw), allow_pickle=False) as stack:
        expected_stack_fields = {
            "classification",
            "campaign_revision",
            "partition",
            "cache_index",
            "fold",
            "prediction",
            "rr_std",
            "proposal_available",
            "nested_role",
            "outer_fold",
            "seed",
            "outer_test_opened",
            "outer_rows_present",
        }
        if set(stack.files) != expected_stack_fields:
            raise RuntimeError("V8R4 nonouter proposer stack schema drifted")
        if str(_scalar_from_npz(stack, "classification")) != NONOUTER_STACK_CLASSIFICATION:
            raise RuntimeError("proposer stack classification drifted")
        if str(_scalar_from_npz(stack, "campaign_revision")) != CAMPAIGN_REVISION:
            raise RuntimeError("proposer stack campaign revision drifted")
        if str(_scalar_from_npz(stack, "partition")) != "outer_excluded_training_validation":
            raise RuntimeError("proposer stack partition drifted")
        if int(_scalar_from_npz(stack, "outer_fold")) != int(outer_fold):
            raise RuntimeError("proposer stack outer_fold differs from CLI")
        if int(_scalar_from_npz(stack, "seed")) != int(seed):
            raise RuntimeError("proposer stack seed differs from CLI")
        if bool(_scalar_from_npz(stack, "outer_test_opened")):
            raise RuntimeError("training refuses a proposer stack with outer test opened")
        if bool(_scalar_from_npz(stack, "outer_rows_present")):
            raise RuntimeError("training refuses a proposer stack containing outer rows")
        stack_index = np.asarray(stack["cache_index"], np.int64)
        if not np.array_equal(stack_index, cache_index):
            raise RuntimeError("proposer stack is not the exact nonouter cache_index cover")
        stack_fold = np.asarray(stack["fold"], np.int16)
        if not np.array_equal(stack_fold, folds):
            raise RuntimeError("proposer stack fold topology differs from its nonouter pack")
        nested_role = np.asarray(stack["nested_role"])
        if nested_role.dtype.kind not in "US" or nested_role.shape != cache_index.shape:
            raise RuntimeError("proposer stack nested_role must be pickle-free text")
        if any("outer" in value.lower() or "test" in value.lower() for value in nested_role.astype(str)):
            raise RuntimeError("proposer stack contains an outer/test role")
        anchor_rr = np.asarray(stack["prediction"], np.float32).copy()
        anchor_std = np.asarray(stack["rr_std"], np.float32).copy()
        anchor_available = np.asarray(stack["proposal_available"], bool).copy()
    if not (
        anchor_rr.shape == anchor_std.shape == anchor_available.shape == (len(metadata),)
    ):
        raise RuntimeError("proposer anchor arrays must have one value per cache row")
    if not anchor_available.all():
        raise RuntimeError("V8R4 nonouter proposer exact cover is incomplete")
    valid_anchor = (
        np.isfinite(anchor_rr) & np.isfinite(anchor_std)
        & (anchor_rr >= RR_MIN_BPM) & (anchor_rr <= RR_MAX_BPM)
        & (anchor_std > 0)
    )
    if np.any(anchor_available & ~valid_anchor):
        raise RuntimeError("available proposer anchors must be finite, in range, and positive-scale")
    anchor_rr[~anchor_available] = 0.0
    anchor_std[~anchor_available] = 1.0

    audit = RowAccessAudit(cache_index=cache_index, fold=folds, outer_fold=int(outer_fold))
    return Experiment(
        root,
        manifest,
        cache_input_binding,
        feature_names,
        metadata,
        AuditedRowArray(node, audit, "node_features"),
        AuditedRowArray(candidate_rr, audit, "candidate_bpm"),
        AuditedRowArray(candidate_mask, audit, "candidate_mask"),
        AuditedRowArray(radar, audit, "joint_radar_mask"),
        AuditedRowArray(anchor_rr, audit, "proposer_prediction"),
        AuditedRowArray(anchor_std, audit, "proposer_rr_std"),
        AuditedRowArray(anchor_available, audit, "proposer_available"),
        stack_path,
        audit,
        consumed_source_files,
    )


def split_positions(metadata: pd.DataFrame, outer_fold: int) -> tuple[np.ndarray, np.ndarray, int]:
    if outer_fold not in range(N_FOLDS):
        raise ValueError("outer_fold must be in [0,5]")
    validation_fold = (outer_fold + 1) % N_FOLDS
    folds = metadata["fold"].to_numpy(np.int64)
    if np.any(folds == int(outer_fold)):
        raise RuntimeError("V8R4 split received a physically present outer-test row")
    train = np.flatnonzero((folds != outer_fold) & (folds != validation_fold))
    validation = np.flatnonzero(folds == validation_fold)
    identities = metadata["identity"].astype(str).to_numpy()
    identity_fold_count = metadata.assign(_identity=identities).groupby(
        "_identity", sort=False
    )["fold"].nunique()
    if (
        not len(train) or not len(validation)
        or set(identities[train]) & set(identities[validation])
        or bool((identity_fold_count != 1).any())
    ):
        raise RuntimeError("outer train/validation/test split is empty or identity-overlapping")
    return train, validation, validation_fold


def identity_balanced_weights(metadata: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, np.int64)
    identities = metadata["identity"].astype(str).to_numpy()
    valid = metadata["reference_valid"].astype(bool).to_numpy(copy=True)
    target = metadata["rr_bpm"].to_numpy(np.float32)
    valid &= np.isfinite(target)
    weight = np.zeros(len(metadata), np.float32)
    for identity in sorted(set(identities[positions])):
        owned = positions[(identities[positions] == identity) & valid[positions]]
        if len(owned):
            weight[owned] = 1.0 / float(len(owned))
    positive = weight[positions] > 0
    if positive.any():
        weight[positions[positive]] *= positive.sum() / weight[positions[positive]].sum()
    return weight


def iter_identity_balanced_sessions(
    metadata: pd.DataFrame,
    positions: np.ndarray,
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> Iterator[np.ndarray]:
    selected = metadata.iloc[np.asarray(positions, np.int64)]
    by_identity: dict[str, list[np.ndarray]] = {}
    for (identity, _), group in selected.groupby(["identity", "session_id"], sort=True):
        ordered = group.sort_values("window_number", kind="stable").index.to_numpy(np.int64)
        if len(np.unique(metadata.iloc[ordered]["window_number"])) != len(ordered):
            raise RuntimeError("duplicate window_number within physical session")
        by_identity.setdefault(str(identity), []).append(ordered)
    rng = np.random.default_rng(int(seed) + 1_000_003 * int(epoch))
    identities = sorted(by_identity)
    if shuffle:
        rng.shuffle(identities)
        for sessions in by_identity.values():
            rng.shuffle(sessions)
    queues = {identity: list(by_identity[identity]) for identity in identities}
    while any(queues.values()):
        for identity in identities:
            if queues[identity]:
                yield queues[identity].pop(0)


def _build_availability(
    candidate_rr: np.ndarray, candidate_mask: np.ndarray, radar_mask: np.ndarray
) -> np.ndarray:
    try:
        return np.asarray(
            build_structural_availability_mask(
                candidate_rr,
                candidate_mask,
                radar_mask,
                rr_min_bpm=RR_MIN_BPM,
                rr_max_bpm=RR_MAX_BPM,
            ),
            dtype=bool,
        )
    except TypeError:
        return np.asarray(
            build_structural_availability_mask(candidate_rr, candidate_mask, radar_mask),
            dtype=bool,
        )


def fit_outer_train_standardizer(
    experiment: Experiment, train_positions: np.ndarray
) -> OuterTrainFeatureStandardizer:
    position = np.asarray(train_positions, np.int64)
    candidate = np.asarray(experiment.candidate_rr[position], np.float32)
    candidate_mask = np.asarray(experiment.candidate_mask[position], bool)
    radar = np.asarray(experiment.joint_radar_mask[position], bool)
    availability = _build_availability(candidate, candidate_mask, radar)
    features = np.asarray(experiment.node_features[position], np.float32)
    digest = _positions_sha256(position)
    if LAYOUT_IMPORT_PATH == "local_contract_compatible_fallback":
        return OuterTrainFeatureStandardizer.fit(
            features, availability, fit_positions_sha256=digest
        )
    return OuterTrainFeatureStandardizer.fit(features, availability)


def factor_class_weights(metadata: pd.DataFrame, train_positions: np.ndarray) -> np.ndarray:
    frame = metadata.iloc[np.asarray(train_positions, np.int64)]
    target = frame["rr_bpm"].to_numpy(np.float32)
    classical = frame["classical_rr_bpm"].to_numpy(np.float32)
    valid = frame["reference_valid"].astype(bool).to_numpy()
    error = np.abs(classical[:, None] * np.asarray(FACTOR_CLASSES)[None] - target[:, None])
    label = error.argmin(axis=-1)
    confident = valid & np.isfinite(target) & np.isfinite(classical) & (error.min(axis=-1) <= 2.0)
    count = np.bincount(label[confident], minlength=4).astype(np.float64)
    weights = np.zeros(4, np.float32)
    present = count > 0
    if present.any():
        weights[present] = (count[present].sum() / count[present]).astype(np.float32)
        weights[present] /= weights[present].mean()
    return weights


def detach_state(state: FactorRouterState) -> FactorRouterState:
    return tuple(
        (membrane.detach(), adaptation.detach()) for membrane, adaptation in state
    )  # type: ignore[return-value]


def _scaler_transform(
    scaler: OuterTrainFeatureStandardizer,
    features: np.ndarray,
    availability: np.ndarray,
) -> np.ndarray:
    transformed = np.asarray(scaler.transform(features, availability), np.float32)
    if transformed.shape != features.shape or not np.isfinite(transformed).all():
        raise RuntimeError("outer-train scaler returned invalid transformed features")
    if np.count_nonzero(transformed[~availability]):
        raise RuntimeError("structurally masked cells must be exact zero after scaling")
    return transformed


def _batch_from_experiment(
    experiment: Experiment,
    scaler: OuterTrainFeatureStandardizer,
    positions: np.ndarray,
    device: torch.device,
    *,
    session_offset: int,
    warmup_windows: int,
    radar_subset: tuple[bool, bool, bool] | None,
    include_targets: bool,
) -> dict[str, Tensor]:
    position = np.asarray(positions, np.int64).copy()
    candidate = np.asarray(experiment.candidate_rr[position], np.float32).copy()
    candidate_mask = np.asarray(experiment.candidate_mask[position], bool).copy()
    radar = np.asarray(experiment.joint_radar_mask[position], bool).copy()
    if radar_subset is not None:
        radar &= np.asarray(radar_subset, bool)[None]
    availability = _build_availability(candidate, candidate_mask, radar)
    node = _scaler_transform(
        scaler, np.asarray(experiment.node_features[position], np.float32), availability
    )
    frame = experiment.metadata.iloc[position]
    length = len(position)
    reset = np.zeros(length, bool)
    reset[0] = session_offset == 0
    batch: dict[str, Tensor] = {
        "position": torch.as_tensor(position[None], device=device),
        "node_features": torch.as_tensor(node[None], device=device),
        "candidate_rr": torch.as_tensor(candidate[None], device=device),
        "candidate_mask": torch.as_tensor(candidate_mask[None], device=device),
        "joint_radar_mask": torch.as_tensor(radar[None], device=device),
        "sequence_mask": torch.ones((1, length), dtype=torch.bool, device=device),
        "reset_mask": torch.as_tensor(reset[None], device=device),
        "anchor_rr": torch.as_tensor(experiment.anchor_rr[position][None], device=device),
        "anchor_std": torch.as_tensor(experiment.anchor_std[position][None], device=device),
        "anchor_available": torch.as_tensor(
            experiment.anchor_available[position][None], device=device
        ),
        "classical_rr": torch.as_tensor(
            frame["classical_rr_bpm"].to_numpy(np.float32, copy=True)[None], device=device
        ),
        "warmup_mask": torch.as_tensor(
            (np.arange(session_offset, session_offset + length) < warmup_windows)[None],
            device=device,
        ),
    }
    if include_targets:
        batch["target"] = torch.as_tensor(
            frame["rr_bpm"].to_numpy(np.float32, copy=True)[None], device=device
        )
        batch["reference_valid"] = torch.as_tensor(
            frame["reference_valid"].astype(bool).to_numpy(copy=True)[None], device=device
        )
    return batch


def _pad_session_lane_batches(
    lane_batches: Sequence[Mapping[str, Tensor] | None],
) -> dict[str, Tensor]:
    """Pad one chronological chunk per physical-session lane.

    Lanes stay in a fixed order for the complete accumulation group.  A lane
    that has finished its physical session is represented by an all-false
    sequence mask, which makes the frozen recurrent cell preserve that lane's
    state exactly.  Every tensor is copied only over its real temporal prefix.
    """

    if not lane_batches or not any(batch is not None for batch in lane_batches):
        raise ValueError("at least one session lane must be active")
    active = [batch for batch in lane_batches if batch is not None]
    template = active[0]
    keys = set(template)
    if any(set(batch) != keys for batch in active):
        raise ValueError("session lane batch schemas differ")
    lengths: list[int] = []
    for batch in lane_batches:
        if batch is None:
            lengths.append(0)
            continue
        sequence = batch.get("sequence_mask")
        if not isinstance(sequence, Tensor) or sequence.ndim != 2 or sequence.shape[0] != 1:
            raise ValueError("lane sequence_mask must have shape [1,time]")
        if not bool(sequence.all()):
            raise ValueError("source lane batches must be unpadded")
        lengths.append(int(sequence.shape[1]))
    maximum = max(lengths)
    result: dict[str, Tensor] = {}
    for key in sorted(keys):
        reference = template[key]
        if reference.ndim < 2 or reference.shape[0] != 1:
            raise ValueError(f"lane tensor {key} must begin [1,time]")
        shape = (len(lane_batches), maximum, *reference.shape[2:])
        fill = 1 if key == "anchor_std" else 0
        padded = torch.full(
            shape,
            fill,
            dtype=reference.dtype,
            device=reference.device,
        )
        for lane, (batch, length) in enumerate(zip(lane_batches, lengths, strict=True)):
            if batch is None:
                continue
            value = batch[key]
            if value.shape[1] != length or value.shape[2:] != reference.shape[2:]:
                raise ValueError(f"lane tensor {key} shape differs")
            padded[lane, :length] = value[0]
        result[key] = padded
    result["lane_lengths"] = torch.as_tensor(
        lengths, dtype=torch.int64, device=template["sequence_mask"].device
    )
    if not torch.equal(
        result["sequence_mask"].sum(dim=1), result["lane_lengths"]
    ):
        raise RuntimeError("padded session lane mask/length mismatch")
    if bool((result["reset_mask"] & ~result["sequence_mask"]).any()):
        raise RuntimeError("padded reset escaped a real session lane")
    return result


def forward_model(
    model: DirectedHarmonicFactorExpertSNN,
    batch: Mapping[str, Tensor],
    *,
    state: FactorRouterState | None,
) -> Mapping[str, Any]:
    """The complete label-free forward allowlist, centralized for auditing."""

    return model(
        batch["node_features"],
        batch["candidate_rr"],
        batch["candidate_mask"].bool(),
        batch["sequence_mask"].bool(),
        joint_radar_mask=batch["joint_radar_mask"].bool(),
        proposer_anchor_bpm=batch["anchor_rr"],
        proposer_anchor_std_bpm=batch["anchor_std"],
        proposer_anchor_available=batch["anchor_available"].bool(),
        classical_rr_bpm=batch["classical_rr"],
        state=state,
        reset_mask=batch["reset_mask"].bool(),
    )


def _masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    return F.log_softmax(logits.float().masked_fill(~mask, -1.0e4), dim=-1)


def listwise_responsibility(
    means: Tensor, mask: Tensor, target: Tensor, *, temperature_bpm: float = 0.5
) -> Tensor:
    score = (-(means.float() - target.float().unsqueeze(-1)).abs() / temperature_bpm)
    score = score.masked_fill(~mask, -1.0e4)
    probability = score.softmax(dim=-1) * mask.float()
    return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def gaussian_mixture_nll(
    logits: Tensor, means: Tensor, scales: Tensor, mask: Tensor, target: Tensor
) -> Tensor:
    log_probability = _masked_log_softmax(logits, mask).masked_fill(~mask, -1.0e4)
    scale = scales.float().clamp(0.05, 20.0)
    standardized = (target.float().unsqueeze(-1) - means.float()) / scale
    density = (
        -0.5 * standardized.square() - scale.log() - 0.5 * math.log(2.0 * math.pi)
    ).masked_fill(~mask, -1.0e4)
    return -torch.logsumexp(log_probability + density, dim=-1)


def commercial_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    identity: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    reference = np.asarray(valid, bool) & np.isfinite(target)
    if not reference.any():
        return {name: math.inf for name in COMMERCIAL_GATES}
    prediction_finite = np.isfinite(prediction)
    if np.any(reference & ~prediction_finite):
        # Missing valid-reference predictions are never silently excluded.
        return {
            "overall_mae_bpm": math.inf,
            "identity_macro_mae_bpm": math.inf,
            "rmse_bpm": math.inf,
            "within_2_fraction": 0.0,
            "over_5_fraction": 1.0,
            "high_rr_25_35_mae_bpm": math.inf,
        }
    y = np.asarray(target, np.float64)[reference]
    p = np.asarray(prediction, np.float64)[reference]
    groups = np.asarray(identity).astype(str)[reference]
    error = np.abs(p - y)
    macro = np.mean([error[groups == group].mean() for group in sorted(set(groups))])
    tail = (y >= 25.0) & (y <= 35.0)
    return {
        "overall_mae_bpm": float(error.mean()),
        "identity_macro_mae_bpm": float(macro),
        "rmse_bpm": float(np.sqrt(np.mean((p - y) ** 2))),
        "within_2_fraction": float(np.mean(error <= 2.0)),
        "over_5_fraction": float(np.mean(error > 5.0)),
        "high_rr_25_35_mae_bpm": float(error[tail].mean()) if tail.any() else math.inf,
    }


def commercial_selection_key(
    metrics: Mapping[str, float], *, epoch: int = 0
) -> tuple[float | int, ...]:
    violations: list[float] = []
    for name, (direction, limit) in COMMERCIAL_GATES.items():
        value = float(metrics[name])
        if not math.isfinite(value):
            violation = math.inf
        elif direction == "maximum":
            violation = max(0.0, (value - limit) / limit)
        else:
            violation = max(0.0, (limit - value) / limit)
        violations.append(violation)
    return (
        sum(value > 0 for value in violations),
        max(violations),
        sum(violations),
        float(metrics["identity_macro_mae_bpm"]),
        float(metrics["overall_mae_bpm"]),
        int(epoch),
    )


def factor_supervision(
    target: Tensor, classical: Tensor
) -> tuple[Tensor, Tensor]:
    factors = target.new_tensor(FACTOR_CLASSES)
    classical_valid = (
        torch.isfinite(classical) & (classical >= RR_MIN_BPM)
        & (classical <= RR_MAX_BPM)
    )
    safe_classical = torch.where(classical_valid, classical, torch.zeros_like(classical))
    error = (safe_classical.unsqueeze(-1) * factors - target.unsqueeze(-1)).abs()
    label = error.argmin(dim=-1)
    confident = (
        classical_valid & (error.min(dim=-1).values <= 2.0)
    )
    return label, confident


def _weighted_mean(values: Tensor, weight: Tensor, denominator: Tensor) -> Tensor:
    return (values * weight).sum() / denominator.clamp_min(1.0e-8)


def _per_session_cvar20(
    weighted_errors: Tensor,
    active_mask: Tensor,
    denominator: Tensor,
) -> Tensor:
    if weighted_errors.ndim != 2 or active_mask.shape != weighted_errors.shape:
        raise ValueError("session CVaR inputs must share shape [lanes,time]")
    if active_mask.dtype != torch.bool:
        raise ValueError("session CVaR active_mask must be boolean")
    numerator = weighted_errors.sum() * 0.0
    for lane in range(weighted_errors.shape[0]):
        active = weighted_errors[lane][active_mask[lane]]
        if active.numel():
            count = max(1, int(math.ceil(0.20 * active.numel())))
            numerator = numerator + torch.topk(active, count).values.sum()
    return numerator / denominator.clamp_min(1.0e-8)


def _valid_length_spike_penalty(
    spike_rates: Tensor,
    regularization_fraction: Tensor | float,
) -> Tensor:
    if spike_rates.ndim != 2:
        raise ValueError("spike_rates must have shape [lanes,channels]")
    lane_penalty = (
        F.relu(0.01 - spike_rates).square()
        + F.relu(spike_rates - 0.20).square()
    ).mean(dim=-1)
    fraction = torch.as_tensor(
        regularization_fraction,
        device=lane_penalty.device,
        dtype=lane_penalty.dtype,
    )
    if fraction.ndim == 0:
        return lane_penalty.mean() * fraction
    if fraction.shape == lane_penalty.shape:
        return (lane_penalty * fraction).sum()
    raise ValueError("regularization_fraction must be scalar or one value per lane")


def _session_cvar_inputs(
    output: Mapping[str, Any],
    batch: Mapping[str, Tensor],
    row_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    raw_target = batch["target"].float()
    target_finite = torch.isfinite(raw_target)
    target = torch.where(target_finite, raw_target, torch.zeros_like(raw_target))
    valid = (
        batch["sequence_mask"].bool()
        & batch["reference_valid"].bool()
        & target_finite
        & ~batch["warmup_mask"].bool()
    )
    weight = row_weights[batch["position"].long()] * valid.float()
    weight = weight * (
        1.0 + 2.0 * ((target >= 25.0) & (target <= 35.0)).float()
    )
    source_available = output["source_available"].bool()
    source_rr = output["source_rr_bpm"].float()
    safe_source_rr = torch.where(source_available, source_rr, target.detach())
    selected_error = torch.where(
        source_available,
        (safe_source_rr - target).abs(),
        torch.zeros_like(target),
    )
    active = valid & source_available
    return selected_error * weight * source_available.float(), active


def compute_multitask_loss(
    output: Mapping[str, Any],
    batch: Mapping[str, Tensor],
    row_weights: Tensor,
    factor_weights: Tensor,
    *,
    variant: str,
    normalization_denominator: Tensor | float | None = None,
    regularization_fraction: Tensor | float = 1.0,
    include_cvar: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Contracted loss stack; references are consumed only in this function."""

    if variant not in VARIANTS:
        raise ValueError("unknown contracted variant")
    raw_target = batch["target"].float()
    target_finite = torch.isfinite(raw_target)
    valid = (
        batch["sequence_mask"].bool() & batch["reference_valid"].bool()
        & target_finite & ~batch["warmup_mask"].bool()
    )
    # Invalid reference windows still update causal state, but every
    # target-dependent arithmetic path receives a finite inert placeholder so
    # NaN*zero cannot poison gradients.
    target = torch.where(target_finite, raw_target, torch.zeros_like(raw_target))
    positions = batch["position"].long()
    weight = row_weights[positions] * valid.float()
    tail = ((target >= 25.0) & (target <= 35.0)).float()
    weight = weight * (1.0 + 2.0 * tail)
    denominator = (
        weight.sum() if normalization_denominator is None
        else torch.as_tensor(normalization_denominator, device=target.device, dtype=target.dtype)
    ).clamp_min(1.0e-8)

    candidate_mask = batch["candidate_mask"].bool()
    anchor_mask = output["anchor_available"].bool()
    expert_mask = torch.cat((anchor_mask.unsqueeze(-1), candidate_mask), dim=-1)
    expert_mask &= batch["sequence_mask"].bool().unsqueeze(-1)
    expert_logits = output["expert_logits"].float()
    candidate_mean = output["candidate_mean_bpm"].float()
    corrected_anchor = output["corrected_anchor_rr_bpm"].float()
    expert_means = torch.cat((corrected_anchor.unsqueeze(-1), candidate_mean), dim=-1)
    expert_scales = torch.cat(
        (
            output["corrected_anchor_scale_bpm"].float().unsqueeze(-1),
            output["candidate_scale_bpm"].float(),
        ),
        dim=-1,
    )
    zero = expert_logits.sum() * 0.0

    responsibility = listwise_responsibility(expert_means, expert_mask, target).detach()
    listwise_per = F.kl_div(
        _masked_log_softmax(expert_logits, expert_mask), responsibility,
        reduction="none", log_target=False,
    ).sum(dim=-1)
    mixture_per = gaussian_mixture_nll(
        expert_logits, expert_means, expert_scales, expert_mask, target
    )

    raw_candidate_rr = batch["candidate_rr"].float()
    candidate_rr = torch.where(
        candidate_mask & torch.isfinite(raw_candidate_rr),
        raw_candidate_rr,
        target.unsqueeze(-1),
    )
    candidate_residual = output["candidate_residual_bpm"].float()
    desired_candidate_residual = (target.unsqueeze(-1) - candidate_rr).clamp(-0.75, 0.75)
    reachable = candidate_mask & (
        (target.unsqueeze(-1) - candidate_rr).abs() <= 0.75 + 1.0e-6
    )
    component_each = F.smooth_l1_loss(
        candidate_residual, desired_candidate_residual, beta=0.25, reduction="none"
    )
    component_per = (
        (component_each * reachable.float()).sum(dim=-1)
        / reachable.float().sum(dim=-1).clamp_min(1.0)
    )

    anchor_rr = batch["anchor_rr"].float()
    anchor_available = batch["anchor_available"].bool()
    desired_anchor_residual = (target - anchor_rr).clamp(-12.0, 12.0)
    anchor_residual_per = F.smooth_l1_loss(
        output["anchor_residual_bpm"].float(), desired_anchor_residual,
        beta=1.0, reduction="none",
    )
    anchor_scale = output["corrected_anchor_scale_bpm"].float().clamp(0.05, 20.0)
    anchor_standardized = (target - corrected_anchor) / anchor_scale
    anchor_nll_per = (
        0.5 * anchor_standardized.square() + anchor_scale.log()
        + 0.5 * math.log(2.0 * math.pi)
    )

    source_rr = output["source_rr_bpm"].float()
    source_available = output["source_available"].bool()
    safe_source_rr = torch.where(source_available, source_rr, target.detach())
    quality_target = ((safe_source_rr.detach() - target).abs() <= 2.0).float()
    quality_per = F.binary_cross_entropy_with_logits(
        output["quality_logit"].float(), quality_target, reduction="none"
    )

    factor_label, factor_confident = factor_supervision(target, batch["classical_rr"].float())
    factor_confident &= output["factor_supervision_mask"].bool() & valid
    factor_logits = output["factor_logits"].float()
    factor_probability = factor_logits.softmax(dim=-1)
    factor_pt = factor_probability.gather(-1, factor_label.unsqueeze(-1)).squeeze(-1)
    class_weight = factor_weights.gather(0, factor_label.reshape(-1)).reshape_as(factor_label)
    factor_focal_per = (
        -(1.0 - factor_pt).pow(2.0)
        * factor_pt.clamp_min(1.0e-8).log() * class_weight
    )

    target_factor_logit = factor_logits.gather(
        -1, factor_label.unsqueeze(-1)
    ).squeeze(-1)
    wrong = factor_logits.masked_fill(
        F.one_hot(factor_label, 4).bool(), -1.0e4
    )
    hardest_wrong = torch.topk(wrong, k=2, dim=-1).values
    margin_per = F.relu(1.0 - (target_factor_logit.unsqueeze(-1) - hardest_wrong)).mean(-1)

    candidate_probability = output["candidate_probabilities"].float()
    affinity = output["factor_affinity"].float()
    candidate_factor_raw = (candidate_probability.unsqueeze(-1) * affinity).sum(dim=-2)
    candidate_factor_raw_sum = candidate_factor_raw.sum(dim=-1, keepdim=True)
    candidate_factor_supported = (
        torch.isfinite(candidate_factor_raw_sum.squeeze(-1))
        & (candidate_factor_raw_sum.squeeze(-1) > 0.0)
    )
    # Factor affinity is exp(-distance / bandwidth).  A far harmonic can
    # underflow to exact float32 zero.  F.kl_div handles 0*log(0) in its
    # forward value, but differentiating its target at zero is non-finite.
    # Smooth every supported distribution before the target-side KL and
    # disable candidate-consistency supervision when no candidate has any
    # factor support at all.
    candidate_factor = candidate_factor_raw.clamp_min(1.0e-8)
    candidate_factor = candidate_factor / candidate_factor.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    factor_distribution = output["factor_probabilities"].float()
    js_active = factor_confident & valid & candidate_factor_supported
    uniform = torch.full_like(factor_distribution, 0.25)
    candidate_factor = torch.where(js_active.unsqueeze(-1), candidate_factor, uniform)
    factor_distribution = torch.where(
        js_active.unsqueeze(-1), factor_distribution, uniform
    )
    midpoint = 0.5 * (candidate_factor + factor_distribution)
    js_per = 0.5 * (
        F.kl_div(midpoint.clamp_min(1.0e-8).log(), candidate_factor, reduction="none").sum(-1)
        + F.kl_div(midpoint.clamp_min(1.0e-8).log(), factor_distribution, reduction="none").sum(-1)
    )

    cvar_weighted, cvar_active = _session_cvar_inputs(
        output, batch, row_weights
    )
    # A batch lane is one physical session.  Risk is reduced within each lane
    # before lanes are added under the unchanged group denominator; padding or
    # a longer neighbouring session can therefore never enter another
    # session's tail set.
    cvar = (
        _per_session_cvar20(cvar_weighted, cvar_active, denominator)
        if include_cvar else zero
    )

    supervised_weight = weight * source_available.float()
    candidate_weight = weight * expert_mask.any(dim=-1).float()
    anchor_weight = weight * anchor_available.float()
    factor_weight = weight * factor_confident.float()
    components = {
        "listwise_kl": _weighted_mean(listwise_per, candidate_weight, denominator),
        "mixture_nll": _weighted_mean(mixture_per, candidate_weight, denominator),
        "component_smooth_l1": _weighted_mean(component_per, weight, denominator),
        "anchor_residual_smooth_l1": _weighted_mean(anchor_residual_per, anchor_weight, denominator),
        "anchor_nll": _weighted_mean(anchor_nll_per, anchor_weight, denominator),
        "factor_focal": _weighted_mean(factor_focal_per, factor_weight, denominator),
        "wrong_harmonic_margin": _weighted_mean(margin_per, factor_weight, denominator),
        "factor_candidate_js": _weighted_mean(js_per, factor_weight, denominator),
        "quality_bce": _weighted_mean(quality_per, supervised_weight, denominator),
        "cvar20": cvar,
    }
    spike_rates = output["spike_rates"].float()
    components["spike_rate"] = _valid_length_spike_penalty(
        spike_rates, regularization_fraction
    )

    enabled = dict(LOSS_WEIGHTS)
    if variant == "H0_no_factor":
        # Replacing disabled terms by an autograd-free zero is important:
        # multiplying an undefined KL derivative by numeric zero can still
        # propagate NaN through the candidate router.
        components["factor_focal"] = zero
        components["wrong_harmonic_margin"] = zero
        components["factor_candidate_js"] = zero
    elif variant == "H1_factor":
        components["wrong_harmonic_margin"] = zero
        components["factor_candidate_js"] = zero
    total = sum(components[name] * enabled[name] for name in LOSS_WEIGHTS)
    return total, components


def _authorization_module() -> Any:
    import importlib.util

    path = PROJECT_ROOT / "scripts/validate_hfr_v3r1_authorization.py"
    specification = importlib.util.spec_from_file_location(
        "snn_rr_validate_hfr_v3r1_authorization", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the v3r1 authorization validator")
    module = importlib.util.module_from_spec(specification)
    prior_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prior_dont_write_bytecode
    return module


def validate_pretrain_authorization(
    admitted_binding: Mapping[str, Any] | None = None,
    *,
    target_sealed_capability_receipt: Path | None = None,
    expected_phase: str | None = None,
    expected_context: Mapping[str, Any] | None = None,
    expected_outer_fold: int | None | object = ...,
) -> dict[str, Any]:
    """Fail closed before a caller can open cache metadata or train."""

    module = _authorization_module()
    if target_sealed_capability_receipt is None:
        if admitted_binding is None:
            result = module.validate_pretrain(PROJECT_ROOT)
        else:
            if expected_phase is None or expected_context is None:
                raise RuntimeError(
                    "admitted pretrain validation requires independent phase/context"
                )
            result = module.validate_pretrain_admitted_child(
                PROJECT_ROOT,
                admitted_binding,
                expected_phase=expected_phase,
                expected_context=expected_context,
            )
    elif admitted_binding is None:
        if expected_phase is None:
            raise RuntimeError("target-scoped pretrain validation requires a phase")
        outer_kwargs = (
            {}
            if expected_outer_fold is ...
            else {"expected_outer_fold": expected_outer_fold}
        )
        result = module.validate_pretrain_target_scoped(
            PROJECT_ROOT,
            target_sealed_capability_receipt,
            expected_phase=expected_phase,
            **outer_kwargs,
        )
    else:
        if expected_phase is None or expected_context is None:
            raise RuntimeError(
                "target admitted pretrain validation requires independent phase/context"
            )
        outer_kwargs = (
            {}
            if expected_outer_fold is ...
            else {"expected_outer_fold": expected_outer_fold}
        )
        result = module.validate_pretrain_target_scoped_admitted_child(
            PROJECT_ROOT,
            target_sealed_capability_receipt,
            admitted_binding,
            expected_phase=expected_phase,
            expected_context=expected_context,
            **outer_kwargs,
        )
    if not (
        result.get("valid") is True
        and result.get("training_authorized") is True
        and result.get("commercial_claim_authorized") is False
    ):
        raise RuntimeError("adaptive v3r1 pretrain authorization is invalid")
    return dict(result)


def _strict_json(path: Path) -> dict[str, Any]:
    _, raw = _capture_verified_file(path)
    return _strict_json_bytes(raw, path)


def _selection_authority_module() -> Any:
    """Load the source-snapshot-bound selector used by both parent and child."""

    import importlib.util

    specification = importlib.util.spec_from_file_location(
        "hfr_v3r1_selection_authority_for_trainer", SELECTION_AUTHORITY_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the canonical promotion selection authority")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    validator = getattr(module, "validate_locked_selection_authorization", None)
    if not callable(validator):
        raise RuntimeError("canonical selector lacks promotion replay validation")
    return module


def validate_phase_authorization(
    *,
    phase: str,
    outer_fold: int,
    variant: str,
    release_mode: str,
    pretrain: Mapping[str, Any],
    promotion_authorization: Path | None,
    admitted_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if phase == "discovery":
        if outer_fold not in (3, 4):
            raise RuntimeError("discovery authorization permits only outer folds 3 and 4")
        if promotion_authorization is not None:
            raise RuntimeError("discovery must not accept a promotion authorization")
        return {
            "phase": "discovery",
            "pretrain_authorization_file_sha256": pretrain[
                "pretrain_authorization_file_sha256"
            ],
        }
    if phase != "promotion" or promotion_authorization is None:
        raise RuntimeError("promotion requires --promotion-authorization")
    path = promotion_authorization.expanduser().resolve()
    if path != PROMOTION_AUTHORIZATION_PATH.resolve():
        raise RuntimeError("promotion authorization must use the canonical path")
    try:
        selection_document, document, governance = (
            _selection_authority_module().validate_locked_selection_authorization(
                PROJECT_ROOT,
                selection_lock_path=PROMOTION_SELECTION_PATH,
                promotion_authorization_path=PROMOTION_AUTHORIZATION_PATH,
                admitted_binding=admitted_binding,
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"canonical promotion selection replay failed: {error}"
        ) from error
    pretrain_binding = document.get("pretrain_authorization")
    expected = (
        pretrain_binding.get("sha256")
        if isinstance(pretrain_binding, Mapping)
        else None
    )
    if expected != pretrain.get("pretrain_authorization_file_sha256"):
        raise RuntimeError(
            "promotion authorization does not bind the active pretrain authorization"
        )
    if document.get("selected_variant") != variant or selection_document.get(
        "selected_variant"
    ) != variant:
        raise RuntimeError("promotion authorization selected_variant differs")
    if document.get("selected_release_mode") != release_mode or selection_document.get(
        "selected_release_mode"
    ) != release_mode:
        raise RuntimeError("promotion authorization selected_release_mode differs")
    if float(document.get("fixed_confidence_switch_probability_min", -1.0)) != 0.8:
        raise RuntimeError("promotion authorization fixed confidence threshold drifted")
    return {
        "phase": "promotion",
        "path": str(path),
        "file_sha256": governance["promotion_authorization"]["sha256"],
        "pretrain_authorization_file_sha256": expected,
        "discovery_selection_lock": document["discovery_selection_lock"],
        "governance": governance,
        "selected_variant": variant,
        "selected_release_mode": release_mode,
    }


def _autocast(device: torch.device, amp: bool) -> Any:
    enabled = bool(amp) and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True) if enabled else nullcontext()


def build_gradient_scaler(
    device: torch.device, amp: bool
) -> torch.amp.GradScaler:
    enabled = bool(amp) and device.type == "cuda"
    return torch.amp.GradScaler(
        device.type,
        enabled=enabled,
        init_scale=AMP_INITIAL_GRADIENT_SCALE,
    )


def _next_amp_gradient_scale(current_scale: float) -> float:
    current = float(current_scale)
    if not math.isfinite(current) or current <= AMP_MINIMUM_GRADIENT_SCALE:
        raise RuntimeError("AMP gradient scale cannot be reduced further")
    return max(AMP_MINIMUM_GRADIENT_SCALE, current * 0.5)


def _capture_group_rng_state(
    device: torch.device, radar_rng: np.random.Generator
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            torch.cuda.get_rng_state(device).clone()
            if device.type == "cuda" else None
        ),
        "radar": copy.deepcopy(radar_rng.bit_generator.state),
    }


def _restore_group_rng_state(
    state: Mapping[str, Any],
    device: torch.device,
    radar_rng: np.random.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        cuda_state = state.get("torch_cuda")
        if not isinstance(cuda_state, Tensor):
            raise RuntimeError("CUDA group replay state is missing")
        torch.cuda.set_rng_state(cuda_state, device)
    radar_rng.bit_generator.state = copy.deepcopy(state["radar"])


def _capture_complete_group_replay_state(
    model: DirectedHarmonicFactorExpertSNN,
    optimizer: torch.optim.Optimizer,
    gradient_scaler: torch.amp.GradScaler,
    device: torch.device,
    radar_rng: np.random.Generator,
    *,
    capture_mutable_training_state: bool,
) -> dict[str, Any]:
    state = _capture_group_rng_state(device, radar_rng)
    if capture_mutable_training_state:
        state.update(
            {
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "gradient_scaler": (
                    copy.deepcopy(gradient_scaler.state_dict())
                    if hasattr(gradient_scaler, "state_dict") else None
                ),
            }
        )
    return state


def _restore_complete_group_replay_state(
    state: Mapping[str, Any],
    model: DirectedHarmonicFactorExpertSNN,
    optimizer: torch.optim.Optimizer,
    gradient_scaler: torch.amp.GradScaler,
    device: torch.device,
    radar_rng: np.random.Generator,
    *,
    gradient_scale: float,
) -> None:
    _restore_group_rng_state(state, device, radar_rng)
    if "model" in state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler_state = state.get("gradient_scaler")
        if scaler_state is not None:
            if not hasattr(gradient_scaler, "load_state_dict"):
                raise RuntimeError("AMP scaler cannot restore its replay state")
            gradient_scaler.load_state_dict(scaler_state)
    optimizer.zero_grad(set_to_none=True)
    gradient_scaler.update(new_scale=float(gradient_scale))


def _model_configuration(variant: str) -> dict[str, Any]:
    return {
        "ordered_feature_names_semantic_sha256": EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
        "structural_layout_semantic_sha256": FEATURE_LAYOUT_SEMANTIC_SHA256,
        "variant": variant,
        "hidden_channels": 64,
        "graph_blocks": 2,
        "simulation_steps": 8,
        "dropout": 0.05,
        "maximum_parameters": 400_000,
    }


def build_model(variant: str, device: torch.device) -> DirectedHarmonicFactorExpertSNN:
    model = DirectedHarmonicFactorExpertSNN(**_model_configuration(variant))
    if hasattr(model, "assert_safe_initialization"):
        model.assert_safe_initialization()
    if int(model.parameter_count()) > 400_000:
        raise RuntimeError("model exceeds the contracted 400,000-parameter cap")
    return model.to(device)


def release_predictions(bundle: PredictionBundle, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "raw_anchor":
        use_anchor = bundle.raw_anchor_available
        prediction = np.where(use_anchor, bundle.raw_anchor_bpm, bundle.hard_source_bpm)
        available = use_anchor | bundle.hard_source_available
    elif mode == "hard_source_argmax":
        prediction = bundle.hard_source_bpm.copy()
        available = bundle.hard_source_available.copy()
    elif mode == "fixed_confidence_switch":
        use_source = (
            bundle.hard_source_available
            & ((bundle.selected_source_probability >= 0.80) | ~bundle.raw_anchor_available)
        )
        prediction = np.where(use_source, bundle.hard_source_bpm, bundle.raw_anchor_bpm)
        available = bundle.hard_source_available | bundle.raw_anchor_available
    else:
        raise ValueError(f"unknown release mode: {mode}")
    prediction = np.where(available, prediction, np.nan).astype(np.float32)
    return prediction, available.astype(bool)


def _empty_prediction_arrays(rows: int) -> dict[str, np.ndarray]:
    return {
        "cache_index": np.empty(rows, np.int64),
        "raw_anchor_bpm": np.empty(rows, np.float32),
        "raw_anchor_available": np.empty(rows, bool),
        "hard_source_bpm": np.empty(rows, np.float32),
        "hard_source_available": np.empty(rows, bool),
        "fixed_confidence_switch_bpm": np.empty(rows, np.float32),
        "fixed_confidence_switch_available": np.empty(rows, bool),
        "selected_source_probability": np.empty(rows, np.float32),
        "selected_source_code": np.empty(rows, np.int16),
        "source_scale_bpm": np.empty(rows, np.float32),
        "quality": np.empty(rows, np.float32),
        "factor_probabilities": np.empty((rows, 4), np.float32),
        "spike_rate": np.empty(rows, np.float32),
    }


def _store_chunk_prediction(
    storage: dict[str, np.ndarray],
    destination: np.ndarray,
    batch: Mapping[str, Tensor],
    output: Mapping[str, Any],
    *,
    lane: int = 0,
    valid_length: int | None = None,
) -> None:
    length = len(destination) if valid_length is None else int(valid_length)
    if length != len(destination) or length < 1:
        raise ValueError("prediction destination and valid lane length differ")

    def array(name: str, dtype: np.dtype[Any] | type[Any] | None = None) -> np.ndarray:
        value = output[name]
        assert isinstance(value, Tensor)
        result = value.detach().float().cpu().numpy()[lane, :length]
        return result.astype(dtype, copy=False) if dtype is not None else result

    raw_anchor = batch["anchor_rr"].detach().cpu().numpy()[lane, :length].astype(np.float32)
    raw_available = (
        batch["anchor_available"].detach().cpu().numpy()[lane, :length].astype(bool)
    )
    hard_source = array("source_rr_bpm", np.float32)
    hard_available = array("source_available", bool)
    probability = array("selected_probability", np.float32)
    fixed_use_source = hard_available & ((probability >= 0.80) | ~raw_available)
    fixed_prediction = np.where(fixed_use_source, hard_source, raw_anchor).astype(np.float32)
    fixed_available = hard_available | raw_available
    spike_sequence = output["spike_sequence"]
    assert isinstance(spike_sequence, Tensor)
    per_row_spike = (
        spike_sequence.detach().float().mean(dim=-1).cpu().numpy()[lane, :length]
    )
    source_scale = array("source_scale_bpm", np.float32)
    source_scale = np.where(hard_available, source_scale, np.nan).astype(np.float32)
    values = {
        "cache_index": (
            batch["cache_index"].detach().cpu().numpy()[lane, :length].astype(np.int64)
        ),
        "raw_anchor_bpm": np.where(raw_available, raw_anchor, np.nan).astype(np.float32),
        "raw_anchor_available": raw_available,
        "hard_source_bpm": np.where(hard_available, hard_source, np.nan).astype(np.float32),
        "hard_source_available": hard_available,
        "fixed_confidence_switch_bpm": np.where(
            fixed_available, fixed_prediction, np.nan
        ).astype(np.float32),
        "fixed_confidence_switch_available": fixed_available,
        "selected_source_probability": probability,
        "selected_source_code": array("selected_source_code", np.int16),
        "source_scale_bpm": source_scale,
        "quality": array("quality", np.float32),
        "factor_probabilities": array("factor_probabilities", np.float32),
        "spike_rate": per_row_spike.astype(np.float32),
    }
    for name, value in values.items():
        storage[name][destination] = value


def _bundle_from_storage(storage: Mapping[str, np.ndarray]) -> PredictionBundle:
    return PredictionBundle(**{name: storage[name] for name in PredictionBundle.__dataclass_fields__})


def predict_experiment_positions(
    model: DirectedHarmonicFactorExpertSNN,
    experiment: Experiment,
    positions: np.ndarray,
    scaler: OuterTrainFeatureStandardizer,
    device: torch.device,
    *,
    amp: bool,
    chunk_windows: int,
    batch_sessions: int = PREDICTION_BATCH_SESSIONS,
) -> PredictionBundle:
    requested = np.asarray(positions, np.int64)
    storage = _empty_prediction_arrays(len(requested))
    destination_by_position = {int(position): index for index, position in enumerate(requested)}
    visited: list[int] = []
    model.eval()
    sessions = list(
        iter_identity_balanced_sessions(
            experiment.metadata, requested, seed=0, epoch=0, shuffle=False
        )
    )
    if batch_sessions < 1:
        raise ValueError("batch_sessions must be positive")
    with torch.no_grad():
        for group in _session_groups(sessions, batch_sessions):
            state: FactorRouterState | None = None
            maximum = max(map(len, group))
            for offset in range(0, maximum, chunk_windows):
                lane_batches: list[Mapping[str, Tensor] | None] = []
                lane_chunks: list[np.ndarray | None] = []
                for session in group:
                    chunk = session[offset : offset + chunk_windows]
                    if not len(chunk):
                        lane_batches.append(None)
                        lane_chunks.append(None)
                        continue
                    lane_batch = _batch_from_experiment(
                        experiment, scaler, chunk, device,
                        session_offset=offset, warmup_windows=0,
                        radar_subset=None, include_targets=False,
                    )
                    lane_batch["cache_index"] = torch.as_tensor(
                        experiment.metadata.iloc[chunk]["cache_index"].to_numpy(
                            np.int64, copy=True
                        )[None],
                        device=device,
                    )
                    lane_batches.append(lane_batch)
                    lane_chunks.append(chunk)
                batch = _pad_session_lane_batches(lane_batches)
                with _autocast(device, amp):
                    output = forward_model(model, batch, state=state)
                state = detach_state(output["state"])
                for lane, chunk in enumerate(lane_chunks):
                    if chunk is None:
                        continue
                    destination = np.asarray(
                        [destination_by_position[int(value)] for value in chunk], np.int64
                    )
                    _store_chunk_prediction(
                        storage,
                        destination,
                        batch,
                        output,
                        lane=lane,
                        valid_length=len(chunk),
                    )
                    visited.extend(map(int, chunk))
    if sorted(visited) != sorted(map(int, requested)) or len(visited) != len(set(visited)):
        raise RuntimeError("validation prediction did not exactly cover requested rows")
    return _bundle_from_storage(storage)


def _batch_from_inference(
    arrays: InferenceArrays,
    scaler: OuterTrainFeatureStandardizer,
    positions: np.ndarray,
    device: torch.device,
    *,
    session_offset: int,
) -> dict[str, Tensor]:
    position = np.asarray(positions, np.int64)
    candidate = arrays.candidate_rr[position]
    candidate_mask = arrays.candidate_mask[position]
    radar = arrays.joint_radar_mask[position]
    availability = _build_availability(candidate, candidate_mask, radar)
    node = _scaler_transform(scaler, arrays.node_features[position], availability)
    length = len(position)
    reset = np.zeros(length, bool)
    reset[0] = session_offset == 0
    return {
        "cache_index": torch.as_tensor(arrays.cache_index[position][None], device=device),
        "node_features": torch.as_tensor(node[None], device=device),
        "candidate_rr": torch.as_tensor(candidate[None], device=device),
        "candidate_mask": torch.as_tensor(candidate_mask[None], device=device),
        "joint_radar_mask": torch.as_tensor(radar[None], device=device),
        "sequence_mask": torch.ones((1, length), dtype=torch.bool, device=device),
        "reset_mask": torch.as_tensor(reset[None], device=device),
        "anchor_rr": torch.as_tensor(arrays.anchor_rr[position][None], device=device),
        "anchor_std": torch.as_tensor(arrays.anchor_std[position][None], device=device),
        "anchor_available": torch.as_tensor(
            arrays.anchor_available[position][None], device=device
        ),
        "classical_rr": torch.as_tensor(arrays.classical_rr[position][None], device=device),
    }


def predict_inference_arrays(
    model: DirectedHarmonicFactorExpertSNN,
    arrays: InferenceArrays,
    scaler: OuterTrainFeatureStandardizer,
    device: torch.device,
    *,
    amp: bool,
    chunk_windows: int,
    batch_sessions: int = PREDICTION_BATCH_SESSIONS,
) -> PredictionBundle:
    rows = len(arrays.cache_index)
    storage = _empty_prediction_arrays(rows)
    starts = np.flatnonzero(arrays.session_reset)
    if not len(starts) or starts[0] != 0:
        raise RuntimeError("sanitized inference input must reset at its first row")
    ends = np.r_[starts[1:], rows]
    sessions = [
        np.arange(int(start), int(end), dtype=np.int64)
        for start, end in zip(starts, ends, strict=True)
    ]
    if batch_sessions < 1:
        raise ValueError("batch_sessions must be positive")
    model.eval()
    with torch.no_grad():
        for group in _session_groups(sessions, batch_sessions):
            state: FactorRouterState | None = None
            maximum = max(map(len, group))
            for offset in range(0, maximum, chunk_windows):
                lane_batches: list[Mapping[str, Tensor] | None] = []
                lane_positions: list[np.ndarray | None] = []
                for session in group:
                    position = session[offset : offset + chunk_windows]
                    if not len(position):
                        lane_batches.append(None)
                        lane_positions.append(None)
                        continue
                    lane_batches.append(
                        _batch_from_inference(
                            arrays, scaler, position, device,
                            session_offset=offset,
                        )
                    )
                    lane_positions.append(position)
                batch = _pad_session_lane_batches(lane_batches)
                with _autocast(device, amp):
                    output = forward_model(model, batch, state=state)
                state = detach_state(output["state"])
                for lane, position in enumerate(lane_positions):
                    if position is None:
                        continue
                    _store_chunk_prediction(
                        storage,
                        position,
                        batch,
                        output,
                        lane=lane,
                        valid_length=len(position),
                    )
    return _bundle_from_storage(storage)


def _session_groups(sessions: Sequence[np.ndarray], count: int) -> Iterator[list[np.ndarray]]:
    for start in range(0, len(sessions), count):
        yield list(sessions[start : start + count])


def _group_weight_denominator(
    experiment: Experiment,
    sessions: Sequence[np.ndarray],
    row_weights: np.ndarray,
    *,
    warmup_windows: int,
) -> float:
    total = 0.0
    for session in sessions:
        frame = experiment.metadata.iloc[session]
        target = frame["rr_bpm"].to_numpy(np.float32)
        valid = frame["reference_valid"].astype(bool).to_numpy() & np.isfinite(target)
        valid[: min(warmup_windows, len(valid))] = False
        weight = row_weights[session].astype(np.float64)
        weight *= 1.0 + 2.0 * ((target >= 25.0) & (target <= 35.0))
        total += float(weight[valid].sum())
    return max(total, 1.0e-8)


def run_training_epoch(
    model: DirectedHarmonicFactorExpertSNN,
    experiment: Experiment,
    train_positions: np.ndarray,
    scaler: OuterTrainFeatureStandardizer,
    optimizer: torch.optim.Optimizer,
    gradient_scaler: torch.amp.GradScaler,
    row_weights_tensor: Tensor,
    row_weights_numpy: np.ndarray,
    factor_weights_tensor: Tensor,
    device: torch.device,
    *,
    seed: int,
    epoch: int,
    variant: str,
    amp: bool,
    chunk_windows: int,
    warmup_windows: int,
    accumulation_sessions: int,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    sessions = list(
        iter_identity_balanced_sessions(
            experiment.metadata, train_positions,
            seed=seed, epoch=epoch, shuffle=True,
        )
    )
    radar_rng = np.random.default_rng(int(seed) + 97_003 * (epoch + 1))
    totals = {"total": 0.0, **{name: 0.0 for name in LOSS_WEIGHTS}}
    groups_completed = 0
    amp_overflow_retries = 0
    forward_passes = 0
    processed_windows = 0
    for group in _session_groups(sessions, accumulation_sessions):
        denominator_value = _group_weight_denominator(
            experiment, group, row_weights_numpy, warmup_windows=warmup_windows
        )
        denominator = torch.tensor(denominator_value, device=device, dtype=torch.float32)
        group_windows = max(1, sum(len(session) for session in group))
        replay_state = _capture_complete_group_replay_state(
            model,
            optimizer,
            gradient_scaler,
            device,
            radar_rng,
            capture_mutable_training_state=bool(
                amp and gradient_scaler.is_enabled()
            ),
        )
        group_retry = 0
        retry_gradient_scale = float(gradient_scaler.get_scale())
        while True:
            if group_retry:
                _restore_complete_group_replay_state(
                    replay_state,
                    model,
                    optimizer,
                    gradient_scaler,
                    device,
                    radar_rng,
                    gradient_scale=retry_gradient_scale,
                )
            optimizer.zero_grad(set_to_none=True)
            round_losses: list[Tensor] = []
            cvar_error_rounds: list[Tensor] = []
            cvar_active_rounds: list[Tensor] = []
            attempt_components: dict[str, Tensor] = {}
            # Draw in the old physical-session order before transposing the
            # execution into temporal chunk rounds.
            subsets = [
                RADAR_SUBSETS[int(radar_rng.integers(0, len(RADAR_SUBSETS)))]
                for _ in group
            ]
            state: FactorRouterState | None = None
            maximum = max(map(len, group))
            attempt_forward_passes = 0
            for offset in range(0, maximum, chunk_windows):
                lane_batches: list[Mapping[str, Tensor] | None] = []
                for session, subset in zip(group, subsets, strict=True):
                    chunk = session[offset : offset + chunk_windows]
                    lane_batches.append(
                        None
                        if not len(chunk)
                        else _batch_from_experiment(
                            experiment,
                            scaler,
                            chunk,
                            device,
                            session_offset=offset,
                            warmup_windows=warmup_windows,
                            radar_subset=subset,
                            include_targets=True,
                        )
                    )
                batch = _pad_session_lane_batches(lane_batches)
                valid_lengths = batch["lane_lengths"].to(torch.float32)
                with _autocast(device, amp):
                    output = forward_model(model, batch, state=state)
                    loss, components = compute_multitask_loss(
                        output,
                        batch,
                        row_weights_tensor,
                        factor_weights_tensor,
                        variant=variant,
                        normalization_denominator=denominator,
                        regularization_fraction=valid_lengths / float(group_windows),
                        include_cvar=False,
                    )
                cvar_errors, cvar_active = _session_cvar_inputs(
                    output, batch, row_weights_tensor
                )
                round_losses.append(loss)
                cvar_error_rounds.append(cvar_errors)
                cvar_active_rounds.append(cvar_active)
                for name, value in components.items():
                    detached = value.detach()
                    attempt_components[name] = (
                        detached
                        if name not in attempt_components
                        else attempt_components[name] + detached
                    )
                state = detach_state(output["state"])
                attempt_forward_passes += 1
            # CVaR is selected over each complete physical session, not over a
            # cross-session pool and not independently per TBPTT chunk.
            group_cvar = _per_session_cvar20(
                torch.cat(cvar_error_rounds, dim=1),
                torch.cat(cvar_active_rounds, dim=1),
                denominator,
            )
            group_loss = torch.stack(round_losses).sum()
            group_loss = group_loss + LOSS_WEIGHTS["cvar20"] * group_cvar
            if not bool(torch.isfinite(group_loss)):
                raise RuntimeError("non-finite training loss")
            gradient_scaler.scale(group_loss).backward()
            attempt_components["cvar20"] = group_cvar.detach()
            gradient_scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip
            )
            if torch.isfinite(gradient_norm):
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
                totals["total"] += float(group_loss.detach().cpu())
                for name, value in attempt_components.items():
                    totals[name] += float(value.cpu())
                groups_completed += 1
                forward_passes += attempt_forward_passes
                processed_windows += group_windows
                break
            optimizer.zero_grad(set_to_none=True)
            if not (amp and gradient_scaler.is_enabled()):
                raise RuntimeError("non-finite gradient norm")
            if group_retry >= AMP_MAX_GROUP_RETRIES:
                raise RuntimeError("AMP gradient overflow retry limit exceeded")
            next_scale = _next_amp_gradient_scale(gradient_scaler.get_scale())
            gradient_scaler.update(new_scale=next_scale)
            retry_gradient_scale = next_scale
            group_retry += 1
            amp_overflow_retries += 1
    denominator_groups = max(1, groups_completed)
    return {
        **{name: value / denominator_groups for name, value in totals.items()},
        "optimizer_steps": float(groups_completed),
        "amp_overflow_retries": float(amp_overflow_retries),
        "amp_final_gradient_scale": float(gradient_scaler.get_scale()),
        "forward_passes": float(forward_passes),
        "processed_windows": float(processed_windows),
    }


def _save_scaler(path: Path, scaler: OuterTrainFeatureStandardizer) -> dict[str, Any]:
    state = dict(scaler.to_state())
    receipt = dict(scaler.state_receipt())
    atomic_write_json(path, {**state, "semantic_receipt": receipt})
    return receipt


def _load_scaler(path: Path) -> OuterTrainFeatureStandardizer:
    scaler, _ = _load_scaler_snapshot(path)
    return scaler


def _load_scaler_snapshot(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    required_mode: int | None = None,
) -> tuple[OuterTrainFeatureStandardizer, dict[str, Any]]:
    binding, raw = _capture_verified_file(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        required_mode=required_mode,
    )
    document = _strict_json_bytes(raw, path)
    if "semantic_receipt" not in document:
        raise RuntimeError("scaler JSON semantic_receipt is required")
    return OuterTrainFeatureStandardizer.from_state(document), binding


def _source_bindings() -> dict[str, Any]:
    trainer = Path(__file__).resolve()
    model = PROJECT_ROOT / (
        "src/snn_rr/harmonic_factor_router_models_v3r1.py"
        if MODEL_IMPORT_PATH.endswith("v3r1")
        else "src/snn_rr/harmonic_factor_router_v3.py"
    )
    layout = PROJECT_ROOT / "src/snn_rr/harmonic_feature_layout_v3r1.py"
    validator = PROJECT_ROOT / "scripts/validate_hfr_v3r1_authorization.py"
    bindings: dict[str, Any] = {}
    for name, path in (
        ("trainer", trainer),
        ("model", model),
        ("feature_layout", layout),
        ("authorization_validator", validator),
        ("contract", CONTRACT_PATH),
        ("inherited_config", CONFIG_PATH),
    ):
        binding, _ = _verified_regular_file(path)
        bindings[name] = binding
    return bindings


def _rng_checkpoint_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python_rng_state": random.getstate(),
        # A raw NumPy ndarray requires a general-pickle global during
        # ``torch.load``.  Store the state as tensors/primitives so every
        # checkpoint remains compatible with ``weights_only=True``.
        "numpy_rng_state": {
            "bit_generator": str(numpy_state[0]),
            "keys": torch.as_tensor(
                np.asarray(numpy_state[1], dtype=np.uint32).astype(np.int64)
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_checkpoint_state(checkpoint: Mapping[str, Any]) -> None:
    random.setstate(checkpoint["python_rng_state"])
    numpy_state = checkpoint["numpy_rng_state"]
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise RuntimeError("resume checkpoint NumPy RNG state is not weights-only safe")
    keys = numpy_state["keys"]
    if not isinstance(keys, Tensor):
        raise RuntimeError("resume checkpoint NumPy RNG keys must be a tensor")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            keys.detach().cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])


def _validation_artifacts(
    bundle: PredictionBundle,
    experiment: Experiment,
    validation_positions: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    position = np.asarray(validation_positions, np.int64)
    frame = experiment.metadata.iloc[position]
    target = frame["rr_bpm"].to_numpy(np.float32, copy=True)
    valid = frame["reference_valid"].astype(bool).to_numpy(copy=True)
    identity_values = frame["identity"].astype(str).tolist()
    identity_width = max(1, *(len(value) for value in identity_values))
    identity = np.asarray(identity_values, dtype=f"<U{identity_width}")
    if identity.dtype.kind != "U" or identity.shape != (len(frame),):
        raise RuntimeError("validation identity serialization must be fixed-width Unicode")
    arrays: dict[str, np.ndarray] = {
        "cache_index": bundle.cache_index,
        "reference_rr_bpm": target,
        "reference_valid": valid,
        "identity": identity,
        "raw_anchor_bpm": bundle.raw_anchor_bpm,
        "raw_anchor_available": bundle.raw_anchor_available,
        "hard_source_bpm": bundle.hard_source_bpm,
        "hard_source_available": bundle.hard_source_available,
        "fixed_confidence_switch_bpm": bundle.fixed_confidence_switch_bpm,
        "fixed_confidence_switch_available": bundle.fixed_confidence_switch_available,
        "selected_source_probability": bundle.selected_source_probability,
        "selected_source_code": bundle.selected_source_code,
        "source_scale_bpm": bundle.source_scale_bpm,
        "quality": bundle.quality,
        "factor_probabilities": bundle.factor_probabilities,
        "spike_rate": bundle.spike_rate,
    }
    metrics: dict[str, Any] = {
        "classification": "adaptive_v3r1_v8r4_discovery_validation_only",
        "campaign_revision": CAMPAIGN_REVISION,
        "outer_test_rows_present": False,
        "release_modes": {},
    }
    for mode in RELEASE_MODES:
        prediction, available = release_predictions(bundle, mode)
        mode_metrics = commercial_metrics(target, prediction, identity, valid)
        metrics["release_modes"][mode] = {
            "metrics": mode_metrics,
            "commercial_gate_selection_key_without_epoch": list(
                commercial_selection_key(mode_metrics)[:-1]
            ),
            "valid_prediction_rows": int((valid & available & np.isfinite(prediction)).sum()),
        }
    return arrays, metrics


def _checkpoint_selection_metrics(
    bundle: PredictionBundle,
    experiment: Experiment,
    validation_positions: np.ndarray,
) -> dict[str, float]:
    position = np.asarray(validation_positions, np.int64)
    frame = experiment.metadata.iloc[position]
    target = frame["rr_bpm"].to_numpy(np.float32)
    valid = frame["reference_valid"].astype(bool).to_numpy()
    identity = frame["identity"].astype(str).to_numpy()
    # Checkpoint selection is architecture-internal and fixed to the hard
    # source.  The global release mode is selected later across discovery jobs.
    return commercial_metrics(
        target, bundle.hard_source_bpm, identity,
        valid,
    )


def _validate_completed_lock(
    output_dir: Path,
    *,
    expected_reuse_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = output_dir / "checkpoint_selection_lock.json"
    require_immutable_output_artifact(path)
    lock_binding, lock_raw = _capture_verified_file(path, required_mode=0o444)
    lock = _strict_json_bytes(lock_raw, path)
    _validate_completed_output_inventory(output_dir, lock)
    if not (
        lock.get("campaign_revision") == CAMPAIGN_REVISION
        and lock.get("leakage_boundary") == DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION
        and isinstance(lock.get("row_access_audit"), Mapping)
        and lock["row_access_audit"].get("campaign_revision") == CAMPAIGN_REVISION
        and lock["row_access_audit"].get("outer_rows_in_physical_pack") == 0
        and lock["row_access_audit"].get("outer_row_access_attempts") == 0
        and lock["row_access_audit"].get("implicit_whole_array_conversions") == 0
    ):
        raise RuntimeError("completed checkpoint lacks the V8R4 physical boundary attestation")
    artifact_raw: dict[str, bytes] = {}
    artifact_bindings: dict[str, dict[str, Any]] = {}
    for key, filename in (
        ("best_checkpoint_sha256", "best.pt"),
        ("last_checkpoint_sha256", "last.pt"),
        ("scaler_sha256", "scaler.json"),
        ("history_sha256", "history.json"),
        ("run_manifest_sha256", "run_manifest.json"),
        ("validation_predictions_sha256", "validation_predictions.npz"),
        ("validation_metrics_sha256", "validation_metrics.json"),
    ):
        require_immutable_output_artifact(output_dir / filename)
        binding, raw = _capture_verified_file(
            output_dir / filename,
            expected_sha256=str(lock.get(key)),
            required_mode=0o444,
        )
        inventory = lock["completed_output_inventory"][filename]
        if binding["bytes"] != inventory.get("bytes"):
            raise RuntimeError(
                f"completed checkpoint selection lock byte binding drifted: {filename}"
            )
        artifact_raw[filename] = raw
        artifact_bindings[filename] = binding
    manifest = _strict_json_bytes(
        artifact_raw["run_manifest.json"], output_dir / "run_manifest.json"
    )
    last = _load_torch_bytes(
        artifact_raw["last.pt"],
        source=output_dir / "last.pt",
        map_location="cpu",
    )
    best = _load_torch_bytes(
        artifact_raw["best.pt"],
        source=output_dir / "best.pt",
        map_location="cpu",
    )
    validate_v8r4_resume_checkpoint(last)
    history_document = _strict_json_bytes(
        artifact_raw["history.json"], output_dir / "history.json"
    )
    scaler_document = _strict_json_bytes(
        artifact_raw["scaler.json"], output_dir / "scaler.json"
    )
    validation_metrics_document = _strict_json_bytes(
        artifact_raw["validation_metrics.json"],
        output_dir / "validation_metrics.json",
    )
    if "semantic_receipt" not in scaler_document:
        raise RuntimeError("completed scaler lacks its semantic receipt")
    scaler = OuterTrainFeatureStandardizer.from_state(scaler_document)
    history_epochs = history_document.get("epochs")
    recorded_reuse = lock.get("reuse_context")
    manifest_reuse = manifest.get("reuse_context")
    if not (
        isinstance(recorded_reuse, Mapping)
        and isinstance(manifest_reuse, Mapping)
        and recorded_reuse.get("output_directory")
        == str(output_dir.expanduser().resolve())
        and _exact_json_equal(recorded_reuse, manifest_reuse)
        and lock.get("reuse_context_sha256") == semantic_sha256(recorded_reuse)
        and manifest.get("reuse_context_sha256")
        == semantic_sha256(manifest_reuse)
        and all(
            lock.get(key) == recorded_reuse.get(key)
            for key in (
                "campaign_id",
                "campaign_revision",
                "campaign_phase",
                "outer_fold",
                "validation_fold",
                "seed",
                "variant",
                "release_mode",
                "run_signature_sha256",
                "scientific_signature_sha256",
            )
        )
    ):
        raise RuntimeError("completed run reuse context binding drifted")
    if expected_reuse_context is not None and not _exact_json_equal(
        recorded_reuse, expected_reuse_context
    ):
        raise RuntimeError("completed run does not match the current exact reuse context")
    if not (
        last.get("run_signature_sha256") == lock.get("run_signature_sha256")
        and last.get("scientific_signature_sha256")
        == lock.get("scientific_signature_sha256")
        and last.get("scaler_sha256") == lock.get("scaler_sha256")
        and int(last.get("best_epoch", -1)) == int(lock.get("best_epoch", -2))
        and isinstance(history_epochs, list)
        and int(last.get("epoch", -1)) == len(history_epochs)
        and semantic_sha256(last.get("history"))
        == semantic_sha256(history_epochs)
        and semantic_sha256(last.get("best_selection_key"))
        == semantic_sha256(lock.get("checkpoint_selection_key"))
        and last.get("reuse_context_sha256")
        == lock.get("reuse_context_sha256")
        and best.get("campaign_id") == CAMPAIGN_ID
        and best.get("campaign_revision") == CAMPAIGN_REVISION
        and best.get("checkpoint_compatibility")
        == "v8r4_nonouter_training_validation_pack_only"
        and best.get("run_signature_sha256") == lock.get("run_signature_sha256")
        and best.get("scientific_signature_sha256")
        == lock.get("scientific_signature_sha256")
        and best.get("scaler_sha256") == lock.get("scaler_sha256")
        and best.get("reuse_context_sha256")
        == lock.get("reuse_context_sha256")
        and int(best.get("epoch", -1)) == int(lock.get("best_epoch", -2))
        and semantic_sha256(best.get("validation_selection_key"))
        == semantic_sha256(lock.get("checkpoint_selection_key"))
        and scaler.state_receipt() == scaler_document.get("semantic_receipt")
        and validation_metrics_document.get("validation_predictions_sha256")
        == artifact_bindings["validation_predictions.npz"]["sha256"]
        and validation_metrics_document.get("validation_predictions_bytes")
        == artifact_bindings["validation_predictions.npz"]["bytes"]
        and _exact_json_equal(
            validation_metrics_document.get("consumed_best_checkpoint"),
            artifact_bindings["best.pt"],
        )
    ):
        raise RuntimeError("completed checkpoint/scaler provenance binding drifted")
    signature = manifest.get("scientific_signature")
    if not isinstance(signature, Mapping):
        raise RuntimeError("completed run manifest lacks its scientific signature")
    signature_sha = scientific_signature_sha256(signature)
    if not (
        manifest.get("scientific_signature_sha256") == signature_sha
        and lock.get("scientific_signature_sha256") == signature_sha
        and manifest.get("run_signature_sha256")
        == lock.get("run_signature_sha256")
    ):
        raise RuntimeError("completed scientific signature binding drifted")
    _assert_file_binding_current(lock_binding)
    _assert_file_bindings_current(artifact_bindings)
    return lock


def validate_v8r4_resume_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Reject V8R3 state while preserving the V8R4 checkpoint payload ABI."""

    if not (
        checkpoint.get("campaign_revision") == CAMPAIGN_REVISION
        and checkpoint.get("checkpoint_compatibility")
        == "v8r4_nonouter_training_validation_pack_only"
    ):
        raise RuntimeError("resume checkpoint predates the V8R4 physical boundary")
    required = {
        "run_signature_sha256",
        "scientific_signature_sha256",
        "reuse_context_sha256",
        "scaler_sha256",
        "epoch",
        "stale",
        "best_epoch",
        "best_selection_key",
        "history",
        "model_state",
        "optimizer_state",
        "gradient_scaler_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"V8R4 resume checkpoint fields missing: {missing}")
    numpy_state = checkpoint.get("numpy_rng_state")
    if not (
        isinstance(numpy_state, Mapping)
        and set(numpy_state)
        == {
            "bit_generator",
            "keys",
            "position",
            "has_gauss",
            "cached_gaussian",
        }
        and isinstance(numpy_state.get("keys"), Tensor)
    ):
        raise RuntimeError("V8R4 resume checkpoint NumPy RNG state is unsafe")


def _exact_authorization_mapping_equal(
    supplied: Mapping[str, Any], fresh: Mapping[str, Any]
) -> bool:
    """Compare authorization results as strict canonical JSON values.

    Python mapping equality aliases ``True`` with ``1``.  Authorization
    evidence is JSON, so compare its canonical bytes instead and reject any
    non-JSON caller value rather than normalizing it.
    """

    try:
        supplied_bytes = json.dumps(
            dict(supplied),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        fresh_bytes = json.dumps(
            dict(fresh),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "caller pretrain authorization is not canonical JSON"
        ) from error
    return supplied_bytes == fresh_bytes


def _fresh_entry_authorization(
    args: argparse.Namespace,
    *,
    execution_phase: str,
    supplied_pretrain: Mapping[str, Any] | None,
    admitted_binding: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Revalidate the complete live authorization at an imported API entry.

    ``main`` performs an early admission check, but training, prediction, and
    the efficiency benchmark are also importable APIs.  They must not trust a
    mapping validated by their caller.  The live admitted-child binding,
    phase, context, target capability, and authorization files are replayed
    here before any entry can inspect a cache or checkpoint, create an output
    directory, initialize CUDA, or construct a model.
    """

    expected_context = args.expected_admitted_context_json
    if not isinstance(expected_context, Mapping):
        raise RuntimeError("entry authorization requires independent admitted context")
    active_binding = (
        _admitted_child_binding_for_cli(phase=execution_phase)
        if admitted_binding is None
        else admitted_binding
    )
    if not isinstance(active_binding, Mapping):
        raise RuntimeError("entry authorization requires an admitted-child binding")
    _validate_admitted_cli_scope(
        args,
        active_binding,
        phase=execution_phase,
        expected_context=expected_context,
    )
    fresh = validate_pretrain_authorization(
        active_binding,
        target_sealed_capability_receipt=(
            args.target_sealed_capability_receipt.expanduser().resolve()
        ),
        expected_phase=execution_phase,
        expected_context=expected_context,
        expected_outer_fold=int(args.outer_fold),
    )
    if supplied_pretrain is not None:
        if not isinstance(supplied_pretrain, Mapping):
            raise RuntimeError("caller pretrain authorization must be a mapping")
        if not _exact_authorization_mapping_equal(supplied_pretrain, fresh):
            raise RuntimeError(
                "caller pretrain authorization differs from fresh entry validation"
            )
    return fresh, active_binding


def train(
    args: argparse.Namespace,
    *,
    pretrain: Mapping[str, Any] | None = None,
    admitted_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # This imported entry owns a fresh governance replay.  A caller-provided
    # result is comparison material only and can never authorize execution.
    pretrain, admitted_binding = _fresh_entry_authorization(
        args,
        execution_phase=(
            "discovery"
            if args.campaign_phase == "discovery"
            else "promotion_training"
        ),
        supplied_pretrain=pretrain,
        admitted_binding=admitted_binding,
    )
    phase_binding = validate_phase_authorization(
        phase=args.campaign_phase,
        outer_fold=args.outer_fold,
        variant=args.variant,
        release_mode=args.release_mode,
        pretrain=pretrain,
        promotion_authorization=args.promotion_authorization,
        admitted_binding=admitted_binding,
    )
    output_dir = args.output_dir.expanduser().resolve()
    cache_root = args.cache.expanduser().resolve()
    proposer_path = args.proposer_stack.expanduser().resolve()
    if (
        output_dir == cache_root
        or cache_root in output_dir.parents
        or output_dir in cache_root.parents
        or proposer_path == output_dir
        or proposer_path.parent == output_dir
    ):
        raise RuntimeError("training output must be disjoint from cache/proposer inputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_atomic_temporaries(
        output_dir, protected_paths=(proposer_path,)
    )
    lock_path = output_dir / "checkpoint_selection_lock.json"
    if lock_path.exists() and not args.resume:
        raise RuntimeError("completed output already exists; pass --resume to verify")
    contract = _load_contract()
    seed_everything(int(args.seed), bool(args.deterministic))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = bool(args.amp) and device.type == "cuda"
    experiment = load_experiment(
        cache_root, proposer_path,
        outer_fold=int(args.outer_fold), seed=int(args.seed),
    )
    promotion_pack_binding = experiment.cache_input_binding.get(
        "promotion_authorization"
    )
    if args.campaign_phase == "promotion":
        if not (
            isinstance(promotion_pack_binding, Mapping)
            and promotion_pack_binding.get("sha256")
            == phase_binding.get("file_sha256")
        ):
            raise RuntimeError(
                "promotion training pack does not bind the admitted promotion authority"
            )
    elif promotion_pack_binding is not None:
        raise RuntimeError("discovery must not consume a promotion training pack")
    train_positions, validation_positions, validation_fold = split_positions(
        experiment.metadata, int(args.outer_fold)
    )
    folds = experiment.metadata["fold"].to_numpy(np.int64)
    if np.any(folds[train_positions] == args.outer_fold) or np.any(
        folds[validation_positions] == args.outer_fold
    ):
        raise RuntimeError("outer-test rows crossed the training/validation boundary")

    source_bindings = _source_bindings()
    input_bindings = {
        "cache_manifest": copy.deepcopy(
            experiment.cache_input_binding["manifest"]
        ),
        "verified_cache_inputs": copy.deepcopy(experiment.cache_input_binding),
        "feature_names": {
            "path": str(experiment.root / "feature_names.json"),
            "sha256": experiment.cache_input_binding["outputs"]["feature_names"][
                "sha256"
            ],
            "bytes": experiment.cache_input_binding["outputs"]["feature_names"][
                "bytes"
            ],
            "semantic_sha256": EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
        },
        "proposer_stack": copy.deepcopy(
            experiment.cache_input_binding["proposer_stack"]
        ),
    }
    effective = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_revision": CAMPAIGN_REVISION,
        "classification": (
            "synthetic_implementation_smoke_test"
            if args.smoke_test else "adaptive_retrospective_historical_cohort_engineering"
        ),
        "campaign_phase": args.campaign_phase,
        "outer_fold": int(args.outer_fold),
        "validation_fold": validation_fold,
        "seed": int(args.seed),
        "variant": args.variant,
        "release_mode": args.release_mode,
        "model": _model_configuration(args.variant),
        "optimization": {
            "epochs": int(args.epochs),
            "minimum_epochs": int(args.minimum_epochs),
            "patience": int(args.patience),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "chunk_windows": int(args.chunk_windows),
            "warmup_windows": int(args.warmup_windows),
            "gradient_accumulation_sessions": int(args.gradient_accumulation_sessions),
            "gradient_clip": float(args.gradient_clip),
            "amp": amp_enabled,
            "amp_gradient_scaler": {
                "initial_scale": AMP_INITIAL_GRADIENT_SCALE,
                "minimum_scale": AMP_MINIMUM_GRADIENT_SCALE,
                "maximum_same_group_retries": AMP_MAX_GROUP_RETRIES,
                "failed_group_skip_allowed": False,
                "deterministic_same_group_replay": True,
            },
            "deterministic": bool(args.deterministic),
            "radar_mask_augmentation_subsets": [list(value) for value in RADAR_SUBSETS],
            "loss_weights": LOSS_WEIGHTS,
        },
        "source_bindings": source_bindings,
        "input_bindings": input_bindings,
        "authorization": {"pretrain": pretrain, "phase": phase_binding},
    }
    population = {
        "campaign_revision": CAMPAIGN_REVISION,
        "position_domain": "canonical_global_cache_index",
        "training_folds": sorted(set(map(int, folds[train_positions]))),
        "validation_fold": validation_fold,
        "outer_test_fold_physically_absent": int(args.outer_fold),
        "training_positions_sha256": _positions_sha256(
            experiment.metadata.iloc[train_positions]["cache_index"].to_numpy(np.int64)
        ),
        "validation_positions_sha256": _positions_sha256(
            experiment.metadata.iloc[validation_positions]["cache_index"].to_numpy(np.int64)
        ),
        "training_rows": len(train_positions),
        "validation_rows": len(validation_positions),
    }
    signature_input = {
        # The V8 helper strips exactly these five orchestration-only fields.
        "output_directory": str(output_dir),
        "campaign_phase_label": args.campaign_phase,
        "promotion_authorization_path": (
            None
            if args.promotion_authorization is None
            else str(args.promotion_authorization.expanduser().resolve())
        ),
        "release_mode": args.release_mode,
        "resume_flag": bool(args.resume),
        # Everything below is retained and independently canonicalized.
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "campaign_revision": CAMPAIGN_REVISION,
        "classification": effective["classification"],
        "contract_file_sha256": source_bindings["contract"]["sha256"],
        "outer_fold": int(args.outer_fold),
        "validation_fold": validation_fold,
        "seed": int(args.seed),
        "variant": args.variant,
        "model": effective["model"],
        "optimization": effective["optimization"],
        "source_bindings": source_bindings,
        "input_bindings": input_bindings,
        "pretrain_authorization": pretrain,
        "population": population,
        "batching_execution": {
            "training_batch_unit": "physical_session_group",
            "temporal_schedule": "aligned_tbptt_chunk_rounds",
            "padding_inert": True,
            "per_session_cvar_before_group_reduction": True,
            "valid_length_spike_weighting": True,
            "prediction_batch_sessions": PREDICTION_BATCH_SESSIONS,
        },
        "checkpoint_selection": {
            "scope": "outer_validation_only_hard_source",
            "commercial_gates": {
                name: {"direction": direction, "limit": limit}
                for name, (direction, limit) in COMMERCIAL_GATES.items()
            },
            "lexicographic_key": [
                "failed_accuracy_gates",
                "worst_normalized_gate_violation",
                "summed_normalized_gate_violation",
                "identity_macro_mae_bpm",
                "overall_mae_bpm",
                "epoch",
            ],
        },
    }
    scientific_signature = canonical_scientific_signature(signature_input)
    scientific_signature_sha = scientific_signature_sha256(scientific_signature)
    run_signature = semantic_sha256(effective)
    manifest_path = output_dir / "run_manifest.json"
    scaler_path = output_dir / "scaler.json"
    if scaler_path.exists():
        require_immutable_output_artifact(scaler_path)
        scaler, scaler_binding = _load_scaler_snapshot(
            scaler_path, required_mode=0o444
        )
    else:
        scaler = fit_outer_train_standardizer(experiment, train_positions)
        _save_scaler(scaler_path, scaler)
        scaler, scaler_binding = _load_scaler_snapshot(
            scaler_path, required_mode=0o444
        )
    scaler_sha = str(scaler_binding["sha256"])
    reuse_context = _json_ready(
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "campaign_phase": args.campaign_phase,
            "outer_fold": int(args.outer_fold),
            "validation_fold": int(validation_fold),
            "seed": int(args.seed),
            "variant": str(args.variant),
            "release_mode": str(args.release_mode),
            "output_directory": str(output_dir),
            "semantic_arguments": {
                "mode": str(args.mode),
                "cache": str(args.cache.expanduser().resolve()),
                "proposer_stack": str(args.proposer_stack.expanduser().resolve()),
                "promotion_authorization": (
                    None
                    if args.promotion_authorization is None
                    else str(args.promotion_authorization.expanduser().resolve())
                ),
                "target_sealed_capability_receipt": str(
                    args.target_sealed_capability_receipt.expanduser().resolve()
                ),
                "device": str(args.device),
                "amp": bool(args.amp),
                "deterministic": bool(args.deterministic),
                "epochs": int(args.epochs),
                "minimum_epochs": int(args.minimum_epochs),
                "patience": int(args.patience),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "chunk_windows": int(args.chunk_windows),
                "warmup_windows": int(args.warmup_windows),
                "gradient_accumulation_sessions": int(
                    args.gradient_accumulation_sessions
                ),
                "gradient_clip": float(args.gradient_clip),
                "smoke_test": bool(args.smoke_test),
            },
            "authorization": {"pretrain": pretrain, "phase": phase_binding},
            "source_bindings": source_bindings,
            "input_bindings": input_bindings,
            "population": population,
            "effective_configuration": effective,
            "run_signature_sha256": run_signature,
            "scientific_signature_sha256": scientific_signature_sha,
            "scaler": scaler_binding,
            "scaler_semantic_receipt": scaler.state_receipt(),
        }
    )
    reuse_context_sha = semantic_sha256(reuse_context)
    if manifest_path.exists():
        require_immutable_output_artifact(manifest_path)
        existing = _strict_json(manifest_path)
        if not (
            existing.get("run_signature_sha256") == run_signature
            and existing.get("scientific_signature_sha256")
            == scientific_signature_sha
            and _exact_json_equal(
                existing.get("scientific_signature"), scientific_signature
            )
            and existing.get("reuse_context_sha256") == reuse_context_sha
            and _exact_json_equal(existing.get("reuse_context"), reuse_context)
            and _exact_json_equal(
                existing.get("effective_configuration"), effective
            )
            and _exact_json_equal(existing.get("population"), population)
        ):
            raise RuntimeError(
                "--resume current authorization/configuration/input binding differs "
                "from run manifest"
            )
        if not args.resume:
            raise RuntimeError("output directory is non-empty; pass --resume")
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "campaign_revision": CAMPAIGN_REVISION,
            "created_utc": utc_now(),
            "run_signature_sha256": run_signature,
            "scientific_signature": scientific_signature,
            "scientific_signature_sha256": scientific_signature_sha,
            "reuse_context": reuse_context,
            "reuse_context_sha256": reuse_context_sha,
            "effective_configuration": effective,
            "population": population,
            "leakage_boundary": copy.deepcopy(DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
        }
        atomic_write_json(manifest_path, manifest)

    if lock_path.exists():
        _assert_file_bindings_current(experiment.consumed_source_files)
        _assert_file_bindings_current(source_bindings)
        validated_completed_lock = _validate_completed_lock(
            output_dir, expected_reuse_context=reuse_context
        )
        _assert_file_bindings_current(experiment.consumed_source_files)
        _assert_file_bindings_current(source_bindings)
        return {
            "status": "already_complete",
            "output_dir": str(output_dir),
            "checkpoint_selection_lock": validated_completed_lock,
        }

    model = build_model(args.variant, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    gradient_scaler = build_gradient_scaler(device, amp_enabled)
    row_weights_numpy = identity_balanced_weights(experiment.metadata, train_positions)
    row_weights_tensor = torch.as_tensor(row_weights_numpy, device=device)
    factor_weights_numpy = factor_class_weights(experiment.metadata, train_positions)
    factor_weights_tensor = torch.as_tensor(factor_weights_numpy, device=device)

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "history.json"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    stale = 0
    best_epoch = -1
    best_key: tuple[float | int, ...] = (math.inf,) * 6
    if args.resume and last_path.exists():
        require_immutable_output_artifact(last_path)
        if best_path.exists():
            require_immutable_output_artifact(best_path)
        if history_path.exists():
            require_immutable_output_artifact(history_path)
        checkpoint, _ = _load_torch_snapshot(
            last_path,
            map_location=device,
            required_mode=0o444,
        )
        validate_v8r4_resume_checkpoint(checkpoint)
        if checkpoint.get("run_signature_sha256") != run_signature:
            raise RuntimeError("resume checkpoint run signature drifted")
        if checkpoint.get("scaler_sha256") != scaler_sha:
            raise RuntimeError("resume checkpoint scaler binding drifted")
        if checkpoint.get("reuse_context_sha256") != reuse_context_sha:
            raise RuntimeError("resume checkpoint exact reuse context drifted")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        gradient_scaler.load_state_dict(checkpoint["gradient_scaler_state"])
        start_epoch = int(checkpoint["epoch"])
        stale = int(checkpoint["stale"])
        best_epoch = int(checkpoint["best_epoch"])
        best_key = tuple(checkpoint["best_selection_key"])
        history = list(checkpoint.get("history", ()))
        if not history and history_path.exists():
            history = _strict_json(history_path)["epochs"]
        if len(history) != start_epoch:
            raise RuntimeError("resume history and last checkpoint epoch disagree")
        # Recover an interrupted last-checkpoint -> history atomic handoff.
        if not history_path.exists() or len(_strict_json(history_path)["epochs"]) != start_epoch:
            atomic_write_json(
                history_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "selection_scope": "outer_validation_only",
                    "outer_test_targets_allowed": False,
                    "epochs": history,
                },
            )
        _restore_rng_checkpoint_state(checkpoint)
    elif args.resume and history_path.exists():
        require_immutable_output_artifact(history_path)
        # No optimizer step was durably committed.  The epoch schedule and RNG
        # are seed-derived, so restarting from epoch zero is deterministic.
        history = []

    for epoch in range(start_epoch, int(args.epochs)):
        train_metrics = run_training_epoch(
            model, experiment, train_positions, scaler, optimizer, gradient_scaler,
            row_weights_tensor, row_weights_numpy, factor_weights_tensor, device,
            seed=int(args.seed), epoch=epoch, variant=args.variant,
            amp=amp_enabled, chunk_windows=int(args.chunk_windows),
            warmup_windows=int(args.warmup_windows),
            accumulation_sessions=int(args.gradient_accumulation_sessions),
            gradient_clip=float(args.gradient_clip),
        )
        validation_bundle = predict_experiment_positions(
            model, experiment, validation_positions, scaler, device,
            amp=amp_enabled, chunk_windows=int(args.chunk_windows),
        )
        validation_metrics = _checkpoint_selection_metrics(
            validation_bundle, experiment, validation_positions
        )
        selection_key = commercial_selection_key(validation_metrics, epoch=epoch + 1)
        improved = selection_key < best_key
        if improved:
            best_key = selection_key
            best_epoch = epoch + 1
            stale = 0
            atomic_torch_save(
                best_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "campaign_id": CAMPAIGN_ID,
                    "campaign_revision": CAMPAIGN_REVISION,
                    "checkpoint_compatibility": (
                        "v8r4_nonouter_training_validation_pack_only"
                    ),
                    "classification": effective["classification"],
                    "run_signature_sha256": run_signature,
                    "scientific_signature_sha256": scientific_signature_sha,
                    "reuse_context_sha256": reuse_context_sha,
                    "model_configuration": _model_configuration(args.variant),
                    "model_state": model.state_dict(),
                    "variant": args.variant,
                    "outer_fold": int(args.outer_fold),
                    "validation_fold": validation_fold,
                    "seed": int(args.seed),
                    "epoch": best_epoch,
                    "validation_metrics": validation_metrics,
                    "validation_selection_key": selection_key,
                    "scaler_sha256": scaler_sha,
                    "commercial_claim_allowed": False,
                },
            )
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch + 1,
                "training": train_metrics,
                "validation_hard_source": validation_metrics,
                "checkpoint_selection_key": selection_key,
                "improved": improved,
            }
        )
        atomic_torch_save(
            last_path,
            {
                "schema_version": SCHEMA_VERSION,
                "campaign_revision": CAMPAIGN_REVISION,
                "checkpoint_compatibility": (
                    "v8r4_nonouter_training_validation_pack_only"
                ),
                "run_signature_sha256": run_signature,
                "scientific_signature_sha256": scientific_signature_sha,
                "reuse_context_sha256": reuse_context_sha,
                "scaler_sha256": scaler_sha,
                "epoch": epoch + 1,
                "stale": stale,
                "best_epoch": best_epoch,
                "best_selection_key": best_key,
                "history": history,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "gradient_scaler_state": gradient_scaler.state_dict(),
                **_rng_checkpoint_state(),
            },
        )
        atomic_write_json(
            history_path,
            {
                "schema_version": SCHEMA_VERSION,
                "selection_scope": "outer_validation_only",
                "outer_test_targets_allowed": False,
                "epochs": history,
            },
        )
        if epoch + 1 >= int(args.minimum_epochs) and stale >= int(args.patience):
            break

    if not best_path.exists():
        raise RuntimeError("training produced no validation-selected checkpoint")
    require_immutable_output_artifact(best_path)
    checkpoint, best_checkpoint_binding = _load_torch_snapshot(
        best_path,
        map_location=device,
        required_mode=0o444,
    )
    if not (
        checkpoint.get("run_signature_sha256") == run_signature
        and checkpoint.get("scientific_signature_sha256")
        == scientific_signature_sha
        and checkpoint.get("reuse_context_sha256") == reuse_context_sha
        and checkpoint.get("scaler_sha256") == scaler_sha
    ):
        raise RuntimeError("best checkpoint exact run binding drifted")
    model.load_state_dict(checkpoint["model_state"])
    validation_bundle = predict_experiment_positions(
        model, experiment, validation_positions, scaler, device,
        amp=amp_enabled, chunk_windows=int(args.chunk_windows),
    )
    validation_arrays, validation_metrics_document = _validation_artifacts(
        validation_bundle, experiment, validation_positions
    )
    _assert_file_binding_current(best_checkpoint_binding)
    _assert_file_bindings_current(experiment.consumed_source_files)
    _assert_file_bindings_current(source_bindings)
    validation_npz = output_dir / "validation_predictions.npz"
    validation_metrics_path = output_dir / "validation_metrics.json"
    validation_prediction_binding = atomic_save_npz(
        validation_npz, validation_arrays
    )
    validation_metrics_document.update(
        {
            "schema_version": SCHEMA_VERSION,
            "best_epoch": int(checkpoint["epoch"]),
            "checkpoint_selection_key": checkpoint["validation_selection_key"],
            "validation_predictions_sha256": validation_prediction_binding[
                "sha256"
            ],
            "validation_predictions_bytes": validation_prediction_binding["bytes"],
            "consumed_best_checkpoint": best_checkpoint_binding,
            "commercial_claim_allowed": False,
        }
    )
    atomic_write_json(validation_metrics_path, validation_metrics_document)
    access_audit = experiment.row_access_audit.snapshot()
    if not (
        access_audit["outer_rows_in_physical_pack"] == 0
        and access_audit["outer_row_access_attempts"] == 0
        and access_audit["implicit_whole_array_conversions"] == 0
    ):
        raise RuntimeError("V8R4 row-access audit observed a boundary violation")
    _assert_file_bindings_current(experiment.consumed_source_files)
    _assert_file_bindings_current(source_bindings)
    _assert_file_binding_current(best_checkpoint_binding)
    _assert_file_binding_current(validation_prediction_binding)
    completed_output_inventory = {
        filename: _immutable_output_binding(output_dir / filename)
        for filename in sorted(
            _TRAIN_COMPLETED_OUTPUT_FILENAMES
            - {"checkpoint_selection_lock.json"}
        )
    }
    if not (
        completed_output_inventory["best.pt"]["sha256"]
        == best_checkpoint_binding["sha256"]
        and completed_output_inventory["best.pt"]["bytes"]
        == best_checkpoint_binding["bytes"]
        and completed_output_inventory["validation_predictions.npz"]["sha256"]
        == validation_prediction_binding["sha256"]
        and completed_output_inventory["validation_predictions.npz"]["bytes"]
        == validation_prediction_binding["bytes"]
    ):
        raise RuntimeError("consumed checkpoint/prediction changed before lock publication")
    lock = {
        "schema_version": SCHEMA_VERSION,
        "campaign_revision": CAMPAIGN_REVISION,
        "created_utc": utc_now(),
        "classification": "adaptive_retrospective_validation_checkpoint_selection_lock",
        "campaign_id": CAMPAIGN_ID,
        "campaign_phase": args.campaign_phase,
        "outer_fold": int(args.outer_fold),
        "validation_fold": validation_fold,
        "seed": int(args.seed),
        "variant": args.variant,
        "release_mode": args.release_mode,
        "best_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection_key": checkpoint["validation_selection_key"],
        "run_signature_sha256": run_signature,
        "scientific_signature_sha256": scientific_signature_sha,
        "reuse_context": reuse_context,
        "reuse_context_sha256": reuse_context_sha,
        "best_checkpoint_sha256": best_checkpoint_binding["sha256"],
        "last_checkpoint_sha256": completed_output_inventory["last.pt"]["sha256"],
        "scaler_sha256": scaler_sha,
        "history_sha256": completed_output_inventory["history.json"]["sha256"],
        "run_manifest_sha256": completed_output_inventory["run_manifest.json"][
            "sha256"
        ],
        "validation_predictions_sha256": completed_output_inventory[
            "validation_predictions.npz"
        ]["sha256"],
        "validation_metrics_sha256": completed_output_inventory[
            "validation_metrics.json"
        ]["sha256"],
        "completed_output_inventory": completed_output_inventory,
        "leakage_boundary": copy.deepcopy(DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
        "row_access_audit": access_audit,
        "commercial_claim_allowed": False,
    }
    atomic_write_json(lock_path, lock, immutable=True)
    validated_lock = _validate_completed_lock(
        output_dir, expected_reuse_context=reuse_context
    )
    return {
        "status": "checkpoint_locked",
        "output_dir": str(output_dir),
        "checkpoint_selection_lock": validated_lock,
    }


def _admitted_child_binding_for_cli(*, phase: str) -> Mapping[str, Any]:
    """Consume the wrapper-issued one-shot live-lifecycle capability."""

    import importlib.util

    wrapper_path = PROJECT_ROOT / "scripts/run_gpu_admitted.py"
    specification = importlib.util.spec_from_file_location(
        "snn_rr_run_gpu_admitted_v8_child", wrapper_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the V8 GPU admission wrapper")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    authorization_path = (
        PROJECT_ROOT
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"
    )
    if not authorization_path.is_file():
        raise RuntimeError("V8R4A pretrain authorization is not issued")
    return module.consume_admitted_child_binding(
        phase,
        PROJECT_ROOT
        / "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/gpu_state_v8r4a/usage/"
        "campaign_gpu_usage_chain_v6.jsonl",
        PROJECT_ROOT
        / "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/gpu_state_v8r4a/execution/"
        "gpu_execution_ledger_v7.jsonl",
        authorization_path,
        sha256_file(authorization_path),
        expected_campaign_id=CAMPAIGN_ID,
        expected_gpu_lock_file=(
            PROJECT_ROOT
            / "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/gpu_state_v8r4a/admission/"
            "gpu_admission_v7.lock"
        ),
    )


def _benchmark_invocation_sha256(admitted_binding: Mapping[str, Any]) -> str:
    phase = admitted_binding.get("phase", admitted_binding.get("execution_phase"))
    invocation = admitted_binding.get("invocation_sha256")
    if not (
        admitted_binding.get("classification")
        == "verified_v8_gpu_admitted_child_lifecycle"
        and phase == "efficiency_benchmark"
        and isinstance(invocation, str)
        and len(invocation) == 64
    ):
        raise RuntimeError("efficiency benchmark admitted-child binding is invalid")
    try:
        bytes.fromhex(invocation)
    except ValueError as error:
        raise RuntimeError("efficiency benchmark invocation SHA-256 is malformed") from error
    return invocation


def _validate_admitted_cli_scope(
    args: argparse.Namespace,
    admitted_binding: Mapping[str, Any],
    *,
    phase: str,
    expected_context: Mapping[str, Any],
) -> None:
    """Bind the wrapper lifecycle identity to the exact scientific CLI unit."""

    context = admitted_binding.get("context")
    if not (
        isinstance(expected_context, Mapping)
        and isinstance(context, Mapping)
        and dict(context) == dict(expected_context)
        and admitted_binding.get("phase") == phase
    ):
        raise RuntimeError("GPU admitted-child CLI lifecycle scope is invalid")
    context = dict(expected_context)
    common = {
        "outer_fold": int(args.outer_fold),
        "seed": int(args.seed),
        "variant": str(args.variant),
    }
    if phase == "efficiency_benchmark":
        expected = {
            "campaign_revision": "V8R4",
            "infrastructure_revision": "V8R4A",
            "authorization_generation": "CONTEXT1",
            "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
            **common,
        }
    elif phase in {"discovery", "promotion_training"}:
        execution_number = context.get("execution_number")
        resume = context.get("resume")
        if type(execution_number) is not int or execution_number < 0 or type(resume) is not bool:
            raise RuntimeError("GPU admitted-child training attempt scope is invalid")
        if resume is not bool(args.resume):
            raise RuntimeError("GPU admitted-child training resume scope drifted")
        expected = {
            **common,
            "execution_number": execution_number,
            "resume": resume,
        }
    elif phase == "promotion_prediction":
        attempt_number = context.get("attempt_number")
        if type(attempt_number) is not int or attempt_number < 0:
            raise RuntimeError("GPU admitted-child prediction attempt scope is invalid")
        expected = {
            **common,
            "release_mode": str(args.release_mode),
            "attempt_number": attempt_number,
        }
    else:  # pragma: no cover - phase is derived from the closed parser choices.
        raise RuntimeError("GPU admitted-child CLI phase is unsupported")
    if dict(context) != expected:
        raise RuntimeError("GPU admitted-child CLI unit identity drifted")


def _synchronize_timing_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_efficiency_benchmark(
    args: argparse.Namespace,
    *,
    admitted_binding: Mapping[str, Any],
    pretrain: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the quarantined V8 two-epoch throughput primitive.

    This deliberately has no output-directory writes, checkpoint selection, or
    validation-reference access.  Its caller owns lifecycle reconciliation and
    may persist only the returned strict timing telemetry.
    """

    # The imported benchmark is a separately guarded entry.  A caller's
    # prevalidated dictionary is comparison material only; it can never grant
    # cache/CUDA/model access by itself.
    pretrain, admitted_binding = _fresh_entry_authorization(
        args,
        execution_phase="efficiency_benchmark",
        supplied_pretrain=pretrain,
        admitted_binding=admitted_binding,
    )
    if not (
        pretrain.get("valid") is True
        and pretrain.get("training_authorized") is True
        and pretrain.get("commercial_claim_authorized") is False
    ):
        raise RuntimeError("efficiency benchmark pretrain authorization is invalid")
    invocation_sha = _benchmark_invocation_sha256(admitted_binding)
    if not (
        args.mode == "efficiency_benchmark"
        and int(args.outer_fold) == 3
        and int(args.seed) == 20260828
        and args.variant == "H0_no_factor"
        and int(args.epochs) == 2
        and not bool(args.smoke_test)
    ):
        raise RuntimeError("efficiency benchmark unit or two-epoch schedule drifted")
    _load_contract()
    seed_everything(int(args.seed), bool(args.deterministic))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = bool(args.amp) and device.type == "cuda"
    validation_fold = (int(args.outer_fold) + 1) % N_FOLDS
    experiment = load_experiment(
        args.cache,
        args.proposer_stack,
        outer_fold=int(args.outer_fold),
        seed=int(args.seed),
        reference_excluded_folds={int(args.outer_fold), validation_fold},
    )
    train_positions, validation_positions, actual_validation_fold = split_positions(
        experiment.metadata, int(args.outer_fold)
    )
    if actual_validation_fold != validation_fold:
        raise RuntimeError("efficiency benchmark validation fold drifted")
    if bool(
        experiment.metadata.iloc[validation_positions]["reference_valid"].astype(bool).any()
    ) or bool(
        np.isfinite(
            experiment.metadata.iloc[validation_positions]["rr_bpm"].to_numpy(np.float64)
        ).any()
    ):
        raise RuntimeError("efficiency benchmark opened validation references")

    scaler = fit_outer_train_standardizer(experiment, train_positions)
    model = build_model(args.variant, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    gradient_scaler = build_gradient_scaler(device, amp_enabled)
    row_weights_numpy = identity_balanced_weights(experiment.metadata, train_positions)
    row_weights_tensor = torch.as_tensor(row_weights_numpy, device=device)
    factor_weights_tensor = torch.as_tensor(
        factor_class_weights(experiment.metadata, train_positions), device=device
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epochs: list[dict[str, int]] = []
    optimizer_steps = 0
    training_windows = 0
    validation_windows = 0
    for epoch in range(2):
        _synchronize_timing_device(device)
        start_ns = time.monotonic_ns()
        training = run_training_epoch(
            model,
            experiment,
            train_positions,
            scaler,
            optimizer,
            gradient_scaler,
            row_weights_tensor,
            row_weights_numpy,
            factor_weights_tensor,
            device,
            seed=int(args.seed),
            epoch=epoch,
            variant=args.variant,
            amp=amp_enabled,
            chunk_windows=int(args.chunk_windows),
            warmup_windows=int(args.warmup_windows),
            accumulation_sessions=int(args.gradient_accumulation_sessions),
            gradient_clip=float(args.gradient_clip),
        )
        _synchronize_timing_device(device)
        trained_ns = time.monotonic_ns()
        # The bundle is intentionally discarded: this pass opens no reference
        # and emits no prediction, score, checkpoint, or reusable model bytes.
        predict_experiment_positions(
            model,
            experiment,
            validation_positions,
            scaler,
            device,
            amp=amp_enabled,
            chunk_windows=int(args.chunk_windows),
        )
        _synchronize_timing_device(device)
        finished_ns = time.monotonic_ns()
        epoch_steps = int(training["optimizer_steps"])
        epoch_train_windows = int(training["processed_windows"])
        epoch_validation_windows = int(len(validation_positions))
        epochs.append(
            {
                "epoch": epoch + 1,
                "warmup": epoch == 0,
                "train_ns": trained_ns - start_ns,
                "validation_ns": finished_ns - trained_ns,
                "total_ns": finished_ns - start_ns,
                "optimizer_steps": epoch_steps,
                "training_windows": epoch_train_windows,
                "validation_windows": epoch_validation_windows,
            }
        )
        optimizer_steps += epoch_steps
        training_windows += epoch_train_windows
        validation_windows += epoch_validation_windows
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    telemetry: dict[str, Any] = {
        "invocation_sha256": invocation_sha,
        "epochs_completed": 2,
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "training_windows": training_windows,
        "validation_windows": validation_windows,
        "peak_cuda_memory_bytes": peak_memory,
    }
    if set(telemetry) != {
        "invocation_sha256",
        "epochs_completed",
        "epochs",
        "optimizer_steps",
        "training_windows",
        "validation_windows",
        "peak_cuda_memory_bytes",
    }:
        raise RuntimeError("efficiency benchmark telemetry schema drifted")
    return telemetry


def _resolve_prediction_pack_artifact(
    *,
    index_path: Path,
    unit_relative: str,
    record: Any,
    filename: str,
) -> dict[str, Any]:
    if not (
        isinstance(record, Mapping)
        and set(record) == {"path", "sha256", "bytes"}
        and record.get("path") == f"{unit_relative}/{filename}"
    ):
        raise RuntimeError(f"prediction pack artifact binding drifted: {filename}")
    source = (index_path.parent / str(record["path"])).resolve()
    expected = (index_path.parent / unit_relative / filename).resolve()
    if source != expected or index_path.parent.resolve() not in source.parents:
        raise RuntimeError(f"prediction pack artifact escaped its unit: {filename}")
    binding, _ = _verified_regular_file(
        source,
        expected_sha256=str(record["sha256"]),
        expected_bytes=int(record["bytes"]),
        required_mode=0o444,
    )
    return binding


def _prediction_model_authority(
    args: argparse.Namespace,
    *,
    pretrain: Mapping[str, Any],
    phase_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the exact model/input bytes authorized by the sealed pack.

    Variant/fold/seed are not a checkpoint capability.  Promotion inference is
    authorized only when the target-scoped receipt binds a model-bound shard
    index whose unit binds the exact packed checkpoint, scaler and sanitized
    input bytes.  Absence of that external pointer is a terminal boundary.
    """

    capability = pretrain.get("capability_document")
    index_record = (
        capability.get("sealed_pack_index")
        if isinstance(capability, Mapping)
        else None
    )
    if not (
        args.campaign_phase == "promotion"
        and phase_binding.get("phase") == "promotion"
        and isinstance(capability, Mapping)
        and capability.get("classification")
        == "adaptive_v3r1_v8r4a_outer_target_sealed_runtime_capability_receipt"
        and capability.get("campaign_id") == CAMPAIGN_ID
        and capability.get("campaign_revision") == CAMPAIGN_REVISION
        and capability.get("phase") == "promotion_prediction"
        and capability.get("outer_fold") == int(args.outer_fold)
        and isinstance(index_record, Mapping)
        and {"path", "sha256", "bytes"} <= set(index_record)
    ):
        raise RuntimeError(
            "promotion prediction lacks an exact target-sealed model authority"
        )
    index_path = Path(str(index_record["path"])).expanduser().resolve()
    index_binding, index_raw = _capture_verified_file(
        index_path,
        expected_sha256=str(index_record["sha256"]),
        expected_bytes=int(index_record["bytes"]),
        required_mode=0o444,
    )
    index = _strict_json_bytes(index_raw, index_path)
    units = index.get("units")
    if not (
        index.get("schema_version") == 1
        and index.get("classification")
        == "adaptive_v3r1_v8r4a_model_bound_target_free_prediction_shard_index"
        and index.get("campaign_id") == CAMPAIGN_ID
        and index.get("campaign_revision") == CAMPAIGN_REVISION
        and index.get("outer_fold") == int(args.outer_fold)
        and index.get("selected_variant") == args.variant
        and index.get("status") == "complete"
        and index.get("completed_units") == 3
        and index.get("unit_count") == 3
        and index.get("outer_test_opened") is False
        and index.get("physical_target_free_input_and_model_packs") is True
        and index.get("source_paths_or_peer_outputs_authorized_in_child") is False
        and index.get("content_sha256")
        == semantic_sha256(
            {key: value for key, value in index.items() if key != "content_sha256"}
        )
        and isinstance(units, list)
        and len(units) == 3
    ):
        raise RuntimeError("target-sealed prediction shard index drifted")
    governance = phase_binding.get("governance")
    promotion_binding = (
        governance.get("promotion_authorization")
        if isinstance(governance, Mapping)
        else None
    )
    if not (
        isinstance(promotion_binding, Mapping)
        and isinstance(index.get("promotion_authorization"), Mapping)
        and all(
            index["promotion_authorization"].get(key)
            == promotion_binding.get(key)
            for key in ("sha256", "bytes")
        )
    ):
        raise RuntimeError("prediction pack promotion authority drifted")
    matching = [
        row
        for row in units
        if isinstance(row, Mapping)
        and row.get("outer_fold") == int(args.outer_fold)
        and row.get("seed") == int(args.seed)
    ]
    if len(matching) != 1:
        raise RuntimeError("prediction pack does not contain the exact requested unit")
    unit = matching[0]
    unit_relative = unit.get("relative_path")
    artifacts = unit.get("artifacts")
    expected_relative = f"units/outer_{int(args.outer_fold)}_seed_{int(args.seed)}"
    if not (
        unit_relative == expected_relative
        and isinstance(artifacts, Mapping)
        and set(artifacts)
        == {
            "prediction_pack_manifest",
            "model_bound_prediction_pack_manifest",
            "outer_predict_input",
            "model_checkpoint",
            "model_scaler",
            "model_source_capability",
        }
        and isinstance(unit.get("scientific_signature_sha256"), str)
        and len(str(unit["scientific_signature_sha256"])) == 64
    ):
        raise RuntimeError("prediction pack unit authority schema drifted")
    checkpoint = _resolve_prediction_pack_artifact(
        index_path=index_path,
        unit_relative=expected_relative,
        record=artifacts["model_checkpoint"],
        filename="model_checkpoint.pt",
    )
    scaler = _resolve_prediction_pack_artifact(
        index_path=index_path,
        unit_relative=expected_relative,
        record=artifacts["model_scaler"],
        filename="model_scaler.json",
    )
    predict_input = _resolve_prediction_pack_artifact(
        index_path=index_path,
        unit_relative=expected_relative,
        record=artifacts["outer_predict_input"],
        filename="outer_predict_input.npz",
    )
    source_capability_binding, source_capability_raw = _capture_verified_file(
        Path(
            _resolve_prediction_pack_artifact(
                index_path=index_path,
                unit_relative=expected_relative,
                record=artifacts["model_source_capability"],
                filename="MODEL_SOURCE_CAPABILITY.json",
            )["path"]
        ),
        expected_sha256=str(artifacts["model_source_capability"]["sha256"]),
        expected_bytes=int(artifacts["model_source_capability"]["bytes"]),
        required_mode=0o444,
    )
    source_capability = _strict_json_bytes(
        source_capability_raw, Path(source_capability_binding["path"])
    )
    packed_checkpoint = source_capability.get("packed_checkpoint")
    packed_scaler = source_capability.get("packed_scaler")
    if not (
        source_capability.get("schema_version") == 1
        and source_capability.get("classification")
        == "adaptive_v3r1_v8r4a_promotion_model_source_capability"
        and source_capability.get("campaign_id") == CAMPAIGN_ID
        and source_capability.get("campaign_revision") == CAMPAIGN_REVISION
        and source_capability.get("outer_fold") == int(args.outer_fold)
        and source_capability.get("seed") == int(args.seed)
        and source_capability.get("selected_variant") == args.variant
        and source_capability.get("scientific_signature_sha256")
        == unit.get("scientific_signature_sha256")
        and source_capability.get("source_deep_validated_before_copy") is True
        and source_capability.get("source_paths_or_peer_outputs_authorized_in_child")
        is False
        and source_capability.get("model_bytes_changed") is False
        and source_capability.get("commercial_or_confirmatory_claim_allowed") is False
        and source_capability.get("content_sha256")
        == semantic_sha256(
            {
                key: value
                for key, value in source_capability.items()
                if key != "content_sha256"
            }
        )
        and isinstance(packed_checkpoint, Mapping)
        and packed_checkpoint.get("sha256") == checkpoint["sha256"]
        and packed_checkpoint.get("bytes") == checkpoint["bytes"]
        and isinstance(packed_scaler, Mapping)
        and packed_scaler.get("sha256") == scaler["sha256"]
        and packed_scaler.get("bytes") == scaler["bytes"]
    ):
        raise RuntimeError("packed model-source capability drifted")
    return {
        "schema_version": 1,
        "index": index_binding,
        "model_source_capability": source_capability_binding,
        "checkpoint": checkpoint,
        "scaler": scaler,
        "predict_input": predict_input,
        "scientific_signature_sha256": unit["scientific_signature_sha256"],
        "source_receipt": source_capability.get("source_receipt"),
    }


def load_sanitized_inference_input(
    path: Path,
    *,
    return_binding: bool = False,
) -> InferenceArrays | tuple[InferenceArrays, dict[str, Any]]:
    source = path.expanduser().resolve()
    binding, raw = _capture_verified_file(source)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        actual = frozenset(archive.files)
        if actual != PREDICT_INPUT_KEYS:
            missing = sorted(PREDICT_INPUT_KEYS - actual)
            unknown = sorted(actual - PREDICT_INPUT_KEYS)
            raise RuntimeError(
                f"sanitized inference fields differ; missing={missing}, unknown={unknown}"
            )
        for forbidden in ("target", "reference", "identity", "protocol", "quality", "fold"):
            if any(forbidden in name.lower() for name in archive.files):
                raise RuntimeError(f"forbidden inference field detected: {forbidden}")
        values = {name: np.asarray(archive[name]).copy() for name in archive.files}

    cache_index = np.asarray(values["cache_index"], np.int64)
    node = np.asarray(values["node_features"], np.float32)
    candidate = np.asarray(values["candidate_rr_bpm"], np.float32)
    candidate_mask = np.asarray(values["candidate_mask"])
    radar = np.asarray(values["joint_radar_mask"])
    anchor_rr = np.asarray(values["proposer_anchor_bpm"], np.float32)
    anchor_std = np.asarray(values["proposer_anchor_std_bpm"], np.float32)
    anchor_available = np.asarray(values["proposer_anchor_available"])
    classical = np.asarray(values["classical_rr_bpm"], np.float32)
    reset = np.asarray(values["session_reset"])
    rows = len(cache_index)
    if cache_index.ndim != 1 or len(np.unique(cache_index)) != rows:
        raise RuntimeError("sanitized cache_index must be a unique one-dimensional exact cover")
    if candidate.ndim != 2 or not 1 <= candidate.shape[1] <= 12:
        raise RuntimeError("sanitized candidates must have shape [N,K], K in [1,12]")
    if node.shape != (*candidate.shape, 571):
        raise RuntimeError("sanitized node_features must have shape [N,K,571]")
    if candidate_mask.dtype != np.bool_ or candidate_mask.shape != candidate.shape:
        raise RuntimeError("sanitized candidate_mask must be boolean [N,K]")
    if radar.dtype != np.bool_ or radar.shape != (rows, 3):
        raise RuntimeError("sanitized joint_radar_mask must be boolean [N,3]")
    if anchor_available.dtype != np.bool_ or anchor_available.shape != (rows,):
        raise RuntimeError("sanitized proposer availability must be boolean [N]")
    if reset.dtype != np.bool_ or reset.shape != (rows,):
        raise RuntimeError("sanitized session_reset must be boolean [N]")
    if not rows or not reset[0]:
        raise RuntimeError("sanitized input must be nonempty and reset at row zero")
    for name, value in (
        ("proposer_anchor_bpm", anchor_rr),
        ("proposer_anchor_std_bpm", anchor_std),
        ("classical_rr_bpm", classical),
    ):
        if value.shape != (rows,):
            raise RuntimeError(f"sanitized {name} must have shape [N]")
    valid_anchor = (
        np.isfinite(anchor_rr) & np.isfinite(anchor_std)
        & (anchor_rr >= RR_MIN_BPM) & (anchor_rr <= RR_MAX_BPM)
        & (anchor_std > 0)
    )
    if np.any(anchor_available & ~valid_anchor):
        raise RuntimeError("available sanitized anchor is invalid")
    anchor_rr[~anchor_available] = 0.0
    anchor_std[~anchor_available] = 1.0
    # This target-free call also validates every available candidate and all
    # structural shapes before the model sees the bytes.
    _build_availability(candidate, candidate_mask, radar)
    arrays = InferenceArrays(
        cache_index, node, candidate, candidate_mask, radar,
        anchor_rr, anchor_std, anchor_available, classical, reset,
    )
    return (arrays, binding) if return_binding else arrays


def predict_target_free(
    args: argparse.Namespace,
    *,
    pretrain: Mapping[str, Any] | None = None,
    admitted_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Prediction is an independently guarded imported entry, not a trusted
    # continuation of ``main`` or an orchestration caller.
    pretrain, admitted_binding = _fresh_entry_authorization(
        args,
        execution_phase="promotion_prediction",
        supplied_pretrain=pretrain,
        admitted_binding=admitted_binding,
    )
    phase_binding = validate_phase_authorization(
        phase=args.campaign_phase,
        outer_fold=args.outer_fold,
        variant=args.variant,
        release_mode=args.release_mode,
        pretrain=pretrain,
        promotion_authorization=args.promotion_authorization,
        admitted_binding=admitted_binding,
    )
    output_dir = args.output_dir.expanduser().resolve()
    prediction_path = output_dir / "predictions.npz"
    manifest_path = output_dir / "prediction_manifest.json"
    checkpoint_path = args.checkpoint.expanduser().resolve()
    scaler_path = args.scaler.expanduser().resolve()
    predict_input_path = args.predict_input.expanduser().resolve()
    protected = {checkpoint_path, scaler_path, predict_input_path}
    if (
        prediction_path in protected
        or manifest_path in protected
        or any(path.parent == output_dir for path in protected)
    ):
        raise RuntimeError("prediction outputs must be disjoint from protected inputs")
    model_authority = _prediction_model_authority(
        args, pretrain=pretrain, phase_binding=phase_binding
    )
    if not (
        model_authority.get("checkpoint", {}).get("path") == str(checkpoint_path)
        and model_authority.get("scaler", {}).get("path") == str(scaler_path)
        and model_authority.get("predict_input", {}).get("path")
        == str(predict_input_path)
    ):
        raise RuntimeError(
            "prediction checkpoint/scaler/input paths differ from sealed authority"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_atomic_temporaries(
        output_dir, protected_paths=protected
    )
    _load_contract()
    seed_everything(int(args.seed), bool(args.deterministic))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint, checkpoint_binding = _load_torch_snapshot(
        checkpoint_path, map_location=device
    )
    scaler, scaler_binding = _load_scaler_snapshot(scaler_path)
    if not (
        _exact_json_equal(checkpoint_binding, model_authority["checkpoint"])
        and _exact_json_equal(scaler_binding, model_authority["scaler"])
        and checkpoint.get("campaign_id") == CAMPAIGN_ID
        and checkpoint.get("campaign_revision") == CAMPAIGN_REVISION
        and checkpoint.get("checkpoint_compatibility")
        == "v8r4_nonouter_training_validation_pack_only"
        and isinstance(checkpoint.get("model_state"), Mapping)
        and isinstance(checkpoint.get("run_signature_sha256"), str)
        and isinstance(checkpoint.get("scientific_signature_sha256"), str)
        and isinstance(checkpoint.get("reuse_context_sha256"), str)
        and checkpoint.get("scientific_signature_sha256")
        == model_authority.get("scientific_signature_sha256")
    ):
        raise RuntimeError("checkpoint campaign binding drifted")
    if checkpoint.get("variant") != args.variant:
        raise RuntimeError("checkpoint variant differs from --variant")
    if int(checkpoint.get("outer_fold", -1)) != int(args.outer_fold):
        raise RuntimeError("checkpoint outer_fold differs from --outer-fold")
    if int(checkpoint.get("seed", -1)) != int(args.seed):
        raise RuntimeError("checkpoint seed differs from --seed")
    if checkpoint.get("scaler_sha256") != scaler_binding["sha256"]:
        raise RuntimeError("checkpoint scaler SHA-256 binding drifted")
    if args.campaign_phase == "promotion":
        authorization = _strict_json(args.promotion_authorization.expanduser().resolve())
        if authorization.get("selected_variant") != args.variant:
            raise RuntimeError("promotion authorization selected_variant differs")
        if authorization.get("selected_release_mode") != args.release_mode:
            raise RuntimeError("promotion authorization selected_release_mode differs")

    # Only after governance and checkpoint validation do we open the sanitized
    # input.  Its loader has no code path for a reference-bearing archive.
    loaded = load_sanitized_inference_input(
        predict_input_path, return_binding=True
    )
    if not isinstance(loaded, tuple):  # pragma: no cover - local API invariant.
        raise RuntimeError("sanitized inference snapshot binding was not returned")
    arrays, predict_input_binding = loaded
    if not _exact_json_equal(
        predict_input_binding, model_authority["predict_input"]
    ):
        raise RuntimeError("sanitized input bytes differ from sealed model authority")
    source_bindings = _source_bindings()
    prediction_source_bindings = {
        "checkpoint": checkpoint_binding,
        "scaler": scaler_binding,
        "predict_input": predict_input_binding,
        "sealed_pack_index": model_authority["index"],
        "model_source_capability": model_authority[
            "model_source_capability"
        ],
    }
    prediction_reuse_context = _json_ready(
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "campaign_phase": args.campaign_phase,
            "outer_fold": int(args.outer_fold),
            "seed": int(args.seed),
            "variant": str(args.variant),
            "release_mode": str(args.release_mode),
            "output_directory": str(output_dir),
            "arguments": {
                "mode": str(args.mode),
                "device": str(args.device),
                "amp": bool(args.amp),
                "deterministic": bool(args.deterministic),
                "chunk_windows": int(args.chunk_windows),
                "promotion_authorization": (
                    None
                    if args.promotion_authorization is None
                    else str(args.promotion_authorization.expanduser().resolve())
                ),
                "target_sealed_capability_receipt": str(
                    args.target_sealed_capability_receipt.expanduser().resolve()
                ),
            },
            "authorization": {"pretrain": pretrain, "phase": phase_binding},
            "source_bindings": source_bindings,
            "checkpoint": checkpoint_binding,
            "checkpoint_provenance": {
                "run_signature_sha256": checkpoint["run_signature_sha256"],
                "scientific_signature_sha256": checkpoint[
                    "scientific_signature_sha256"
                ],
                "reuse_context_sha256": checkpoint["reuse_context_sha256"],
                "scaler_sha256": checkpoint["scaler_sha256"],
            },
            "scaler": scaler_binding,
            "scaler_semantic_receipt": scaler.state_receipt(),
            "predict_input": predict_input_binding,
            "predict_input_fields": sorted(PREDICT_INPUT_KEYS),
            "sealed_model_authority": model_authority,
        }
    )
    prediction_reuse_sha = semantic_sha256(prediction_reuse_context)
    prediction_exists = prediction_path.exists()
    manifest_exists = manifest_path.exists()
    if prediction_exists:
        require_immutable_output_artifact(prediction_path)
    if manifest_exists:
        require_immutable_output_artifact(manifest_path)
    if prediction_exists and manifest_exists:
        if not args.resume:
            raise RuntimeError("target-free prediction output already exists")
        manifest_binding, manifest_raw = _capture_verified_file(
            manifest_path, required_mode=0o444
        )
        manifest = _strict_json_bytes(manifest_raw, manifest_path)
        prediction_binding, _ = _verified_regular_file(
            prediction_path,
            expected_sha256=str(manifest.get("predictions_sha256")),
            expected_bytes=manifest.get("predictions_bytes"),
            required_mode=0o444,
        )
        if not (
            manifest.get("prediction_reuse_context_sha256")
            == prediction_reuse_sha
            and _exact_json_equal(
                manifest.get("prediction_reuse_context"),
                prediction_reuse_context,
            )
            and manifest.get("predictions_sha256")
            == prediction_binding["sha256"]
        ):
            raise RuntimeError(
                "existing target-free prediction does not match current exact inputs"
            )
        _assert_file_bindings_current(source_bindings)
        _assert_file_bindings_current(prediction_source_bindings)
        _assert_file_bindings_current(
            {
                "prediction_manifest": manifest_binding,
                "predictions": prediction_binding,
            }
        )
        return {"status": "already_predicted", "prediction_manifest": manifest}
    if manifest_exists and not prediction_exists:
        raise RuntimeError("prediction manifest exists without its bound predictions")
    model = build_model(args.variant, device)
    model.load_state_dict(checkpoint["model_state"])
    bundle = predict_inference_arrays(
        model, arrays, scaler, device,
        amp=bool(args.amp) and device.type == "cuda",
        chunk_windows=int(args.chunk_windows),
    )
    prediction, available = release_predictions(bundle, args.release_mode)
    output_arrays = {
        "cache_index": bundle.cache_index,
        "prediction_bpm": prediction,
        "prediction_available": available,
        "raw_anchor_bpm": bundle.raw_anchor_bpm,
        "raw_anchor_available": bundle.raw_anchor_available,
        "hard_source_bpm": bundle.hard_source_bpm,
        "hard_source_available": bundle.hard_source_available,
        "selected_source_probability": bundle.selected_source_probability,
        "selected_source_code": bundle.selected_source_code,
        "source_scale_bpm": bundle.source_scale_bpm,
        "quality": bundle.quality,
        "factor_probabilities": bundle.factor_probabilities,
        "spike_rate": bundle.spike_rate,
    }
    if tuple(output_arrays) != PREDICTION_KEYS:
        raise RuntimeError("internal target-free prediction schema drifted")
    if any(
        forbidden in name.lower()
        for name in output_arrays
        for forbidden in ("target", "reference", "identity", "protocol", "fold")
    ):
        raise RuntimeError("forbidden field reached the target-free prediction artifact")
    if prediction_exists:
        if not args.resume:
            raise RuntimeError("partial target-free prediction output already exists")
        prediction_binding, existing_raw = _capture_verified_file(
            prediction_path, required_mode=0o444
        )
        with np.load(io.BytesIO(existing_raw), allow_pickle=False) as existing:
            if tuple(existing.files) != PREDICTION_KEYS:
                raise RuntimeError("partial target-free prediction schema drifted")
            for name, expected in output_arrays.items():
                actual = np.asarray(existing[name])
                if actual.shape != expected.shape or not np.array_equal(
                    actual, expected, equal_nan=True
                ):
                    raise RuntimeError(
                        f"partial target-free prediction bytes disagree at {name}"
                    )
    else:
        _assert_file_bindings_current(source_bindings)
        _assert_file_bindings_current(prediction_source_bindings)
        prediction_binding = atomic_save_npz(
            prediction_path, output_arrays, immutable=True
        )
    _assert_file_binding_current(prediction_binding)
    _assert_file_bindings_current(source_bindings)
    _assert_file_bindings_current(prediction_source_bindings)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "classification": "adaptive_retrospective_target_free_model_prediction",
        "campaign_id": CAMPAIGN_ID,
        "campaign_phase": args.campaign_phase,
        "outer_fold": int(args.outer_fold),
        "seed": int(args.seed),
        "variant": args.variant,
        "release_mode": args.release_mode,
        "fixed_confidence_switch_probability_min": 0.80,
        "row_count": len(arrays.cache_index),
        "predict_input": {
            **predict_input_binding,
            "fields": sorted(PREDICT_INPUT_KEYS),
        },
        "checkpoint": checkpoint_binding,
        "scaler": scaler_binding,
        "source_bindings": source_bindings,
        "authorization": {"pretrain": pretrain, "phase": phase_binding},
        "prediction_reuse_context": prediction_reuse_context,
        "prediction_reuse_context_sha256": prediction_reuse_sha,
        "predictions_sha256": prediction_binding["sha256"],
        "predictions_bytes": prediction_binding["bytes"],
        "prediction_fields": list(output_arrays),
        "target_fields_accepted": False,
        "target_fields_emitted": False,
        "identity_or_protocol_fields_emitted": False,
        "continuous_anchor_candidate_blending_used": False,
        "commercial_claim_allowed": False,
    }
    atomic_write_json(manifest_path, manifest, immutable=True)
    _assert_file_binding_current(prediction_binding)
    return {"status": "target_free_prediction_complete", "prediction_manifest": manifest}


def _parse_expected_admitted_context(raw: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate expected admitted context key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite expected admitted context: {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError, RuntimeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("expected admitted context must be an object")
    semantic_sha256(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("train", "predict", "efficiency_benchmark"),
        default="train",
    )
    parser.add_argument("--campaign-phase", choices=("discovery", "promotion"), default="discovery")
    parser.add_argument("--promotion-authorization", type=Path)
    parser.add_argument(
        "--target-sealed-capability-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-admitted-context-json",
        type=_parse_expected_admitted_context,
        required=True,
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--proposer-stack", type=Path)
    parser.add_argument("--predict-input", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scaler", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", "--fold", dest="outer_fold", type=int, required=True, choices=range(6))
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--variant", choices=VARIANTS, default="H2_full")
    parser.add_argument("--release-mode", choices=RELEASE_MODES, default="hard_source_argmax")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--chunk-windows", type=int, default=32)
    parser.add_argument("--warmup-windows", type=int, default=2)
    parser.add_argument("--gradient-accumulation-sessions", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="allow reduced scheduling only for synthetic implementation tests",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.mode in {"train", "efficiency_benchmark"}:
        if args.cache is None or args.proposer_stack is None:
            raise SystemExit(f"{args.mode} mode requires --cache and --proposer-stack")
        if any(value is not None for value in (args.predict_input, args.checkpoint, args.scaler)):
            raise SystemExit(
                f"{args.mode} mode must not accept predict-only input/checkpoint/scaler"
            )
    else:
        if any(value is None for value in (args.predict_input, args.checkpoint, args.scaler)):
            raise SystemExit("predict mode requires --predict-input, --checkpoint, and --scaler")
        if args.cache is not None or args.proposer_stack is not None:
            raise SystemExit("target-free predict mode refuses cache metadata and proposer stacks")
        if args.smoke_test:
            raise SystemExit("predict mode has no --smoke-test scheduling bypass")
    if (
        args.mode != "efficiency_benchmark"
        and args.campaign_phase == "promotion"
        and args.promotion_authorization is None
    ):
        raise SystemExit("promotion phase requires --promotion-authorization")
    if (
        args.mode != "efficiency_benchmark"
        and args.campaign_phase == "discovery"
        and args.promotion_authorization is not None
    ):
        raise SystemExit("discovery phase refuses --promotion-authorization")
    if args.mode == "efficiency_benchmark" and (
        args.campaign_phase != "discovery" or args.promotion_authorization is not None
    ):
        raise SystemExit("efficiency benchmark refuses phase/promotion authorization inputs")
    if not (
        args.epochs >= 1
        and args.minimum_epochs >= 1
        and args.patience >= 1
        and args.epochs <= 120
        and args.learning_rate > 0
        and args.weight_decay >= 0
        and args.chunk_windows >= 1
        and args.warmup_windows >= 0
        and args.gradient_accumulation_sessions >= 1
        and args.gradient_clip > 0
    ):
        raise SystemExit("optimization scheduling values are invalid")
    benchmark_contracted = (
        args.epochs == 2
        and args.outer_fold == 3
        and args.seed == 20260828
        and args.variant == "H0_no_factor"
        and math.isclose(args.learning_rate, 3.0e-4)
        and math.isclose(args.weight_decay, 1.0e-4)
        and args.chunk_windows == 32
        and args.warmup_windows == 2
        and args.gradient_accumulation_sessions == 4
        and math.isclose(args.gradient_clip, 2.0)
        and args.deterministic
        and args.amp
        and not args.smoke_test
    )
    if args.mode == "efficiency_benchmark" and not benchmark_contracted:
        raise SystemExit("efficiency benchmark must use its immutable V8 unit and schedule")
    contracted = (
        args.epochs == 120
        and args.minimum_epochs == 20
        and args.patience == 18
        and math.isclose(args.learning_rate, 3.0e-4)
        and math.isclose(args.weight_decay, 1.0e-4)
        and args.chunk_windows == 32
        and args.warmup_windows == 2
        and args.gradient_accumulation_sessions == 4
        and math.isclose(args.gradient_clip, 2.0)
        and args.deterministic
    )
    if args.mode == "train" and not args.smoke_test and not contracted:
        raise SystemExit("non-smoke training must use the immutable contracted schedule")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "efficiency_benchmark":
        execution_phase = "efficiency_benchmark"
    elif args.mode == "predict":
        execution_phase = "promotion_prediction"
    else:
        execution_phase = (
            "discovery"
            if args.campaign_phase == "discovery"
            else "promotion_training"
        )
    admitted_binding = _admitted_child_binding_for_cli(phase=execution_phase)
    _validate_admitted_cli_scope(
        args,
        admitted_binding,
        phase=execution_phase,
        expected_context=args.expected_admitted_context_json,
    )
    pretrain = validate_pretrain_authorization(
        admitted_binding,
        target_sealed_capability_receipt=(
            args.target_sealed_capability_receipt.expanduser().resolve()
        ),
        expected_phase=execution_phase,
        expected_context=args.expected_admitted_context_json,
        expected_outer_fold=int(args.outer_fold),
    )
    if args.mode == "train":
        result = train(
            args, pretrain=pretrain, admitted_binding=admitted_binding
        )
    elif args.mode == "predict":
        result = predict_target_free(
            args, pretrain=pretrain, admitted_binding=admitted_binding
        )
    else:
        result = run_efficiency_benchmark(
            args,
            admitted_binding=admitted_binding,
            pretrain=pretrain,
        )
    print(json.dumps(_json_ready(result), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
