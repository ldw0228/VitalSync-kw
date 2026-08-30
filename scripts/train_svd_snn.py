#!/usr/bin/env python3
"""Train a leakage-audited source-separated SNN as a safe OOF residual.

The raw-window SVD cache is joined to an already identity-disjoint base OOF by
``cache_index``.  For outer fold ``f`` this program uses fold ``(f + 1) % 6``
only for early stopping and promotion, and the remaining four folds for weight
fitting.  The outer-test loader is not constructed until a validation-only
selection lock has been written.  Consequently test labels or test metadata
cannot affect model weights, early stopping, feature scaling, or promotion.

The source-separated network is deliberately a safe residual: a fold is
promoted only when validation identity-macro MAE improves by at least 0.05 bpm
and the 25--35 bpm macro MAE, >5 bpm error rate, and +/-2 bpm rate are all
non-inferior.  Otherwise its locked-final prediction is exactly the supplied
base OOF prediction.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from snn_rr.metrics import (  # noqa: E402
    grouped_oof_metrics,
    identity_macro_metrics,
    regression_metrics,
)
from snn_rr.models import gaussian_soft_targets  # noqa: E402
from snn_rr.svd_models import SourceSeparatedRRSNN  # noqa: E402


SCHEMA_VERSION = 1
N_FOLDS = 6

# This is an allow-list, not a pattern-based discovery rule.  Every entry is a
# value emitted by the frozen base estimator at inference time.  Reference,
# target, identity, protocol and window-timing columns can therefore never be
# silently added to the model when a cache schema grows.
BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    "prediction_uncalibrated_bpm",
    "prediction_calibrated_bpm",
    "prediction_final_structured_aux_s12_bpm",
    "prediction_final_structured_exact_s12_deterministic_bpm",
    "prediction_final_structured_aux_s12_calibrated_bpm",
    "prediction_final_structured_exact_s12_deterministic_calibrated_bpm",
    "uncertainty_score",
    "uncertainty_uncalibrated",
    "uncertainty_final_structured_aux_s12",
    "uncertainty_final_structured_exact_s12_deterministic",
    "disagreement_bpm",
    "disagreement_coefficient",
    "weight_final_structured_aux_s12",
    "quality",
    "spike_rate",
    "classical_confidence",
    "radar_peak_1_bpm",
    "radar_peak_2_bpm",
    "radar_peak_3_bpm",
    "radar_peak_spread_bpm",
)

LABEL_ONLY_COLUMNS: frozenset[str] = frozenset(
    {
        "rr_bpm",
        "rr_spectral_bpm",
        "rr_phase_bpm",
        "rr_events_bpm",
        "reference_valid",
        "reference_quality",
        "reference_sigma_bpm",
        "classical_error_bpm",
        "classical_acceptable_within_2bpm",
        "breath_count",
        "identity",
        "session_id",
        "session_number",
        "protocol",
        "fold",
        "window_number",
        "window_start_s",
        "window_end_s",
        "cache_index",
    }
)

BINDING_COLUMNS: tuple[str, ...] = (
    "cache_index",
    "session_id",
    "identity",
    "protocol",
    "window_number",
    "window_start_s",
    "window_end_s",
    "rr_bpm",
)


@dataclass(frozen=True, slots=True)
class FoldSplit:
    outer_fold: int
    validation_fold: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_identities: tuple[str, ...]
    validation_identities: tuple[str, ...]
    test_identities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RobustScaler:
    columns: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if tuple(self.columns) != tuple(str(value) for value in self.columns):
            raise RuntimeError("invalid scaler column names")
        raw = frame.loc[:, list(self.columns)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        raw = np.where(np.isfinite(raw), raw, self.center[None, :])
        return np.clip((raw - self.center[None, :]) / self.scale[None, :], -12, 12).astype(
            np.float32
        )


@dataclass(slots=True)
class SVDSessionArrays:
    session_id: str
    spectra: np.ndarray
    attributes: np.ndarray
    frequencies_hz: np.ndarray
    metadata: pd.DataFrame
    manifest: dict[str, Any]
    files_sha256: dict[str, str]
    radar_timing_valid_mask: np.ndarray | None = None
    component_signals: np.ndarray | None = None


@dataclass(slots=True)
class AlignedSVDExperiment:
    cache_root: Path
    oof_csv: Path
    oof_npz: Path
    metadata: pd.DataFrame
    sessions: list[SVDSessionArrays]
    frequencies_hz: np.ndarray
    root_manifest: dict[str, Any]
    provenance: dict[str, Any]

    def arrays_for_position(self, position: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.metadata.iloc[int(position)]
        session = self.sessions[int(row["_session_slot"])]
        local = int(row["_local_row"])
        return session.spectra[local], session.attributes[local]

    def structural_radar_mask_for_position(
        self, position: int
    ) -> np.ndarray | None:
        row = self.metadata.iloc[int(position)]
        session = self.sessions[int(row["_session_slot"])]
        if session.radar_timing_valid_mask is None:
            return None
        local = int(row["_local_row"])
        timing = np.asarray(session.radar_timing_valid_mask[local], dtype=np.bool_)
        if timing.ndim != 2 or timing.shape[0] != 3:
            raise RuntimeError(
                f"invalid structural radar timing mask for {session.session_id}"
            )
        return np.all(timing, axis=1)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    position: np.ndarray
    cache_index: np.ndarray
    target: np.ndarray
    base_prediction: np.ndarray
    candidate_prediction: np.ndarray
    rr_std: np.ndarray
    source_prediction: np.ndarray
    mixture_gate: np.ndarray
    quality: np.ndarray
    spike_rate: np.ndarray
    radar_weights: np.ndarray


def sha256_file(path: Path, *, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_svd_pipeline_sha256() -> str:
    paths = (
        PROJECT_ROOT / "scripts/build_svd_features.py",
        SOURCE_ROOT / "snn_rr/svd_features.py",
        PROJECT_ROOT / "scripts/build_features.py",
        SOURCE_ROOT / "snn_rr/acquisition_contract.py",
        SOURCE_ROOT / "snn_rr/cache.py",
        SOURCE_ROOT / "snn_rr/data.py",
        SOURCE_ROOT / "snn_rr/preprocess.py",
        SOURCE_ROOT / "snn_rr/synchronization.py",
        SOURCE_ROOT / "snn_rr/radar_timing.py",
    )
    return hashlib.sha256(
        "".join(sha256_file(path) for path in paths).encode("utf-8")
    ).hexdigest()


def _feature_cache_inventory_sha256(
    cache_root: Path, root_manifest: Mapping[str, Any]
) -> tuple[str, int]:
    sessions = root_manifest.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("canonical feature-cache session catalogue is missing")
    paths = [cache_root / "manifest.json"]
    for item in sessions:
        if not isinstance(item, Mapping) or item.get("status") != "ok":
            continue
        session_id = item.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("canonical feature-cache session ID is invalid")
        session_dir = cache_root / session_id
        paths.extend(
            session_dir / name
            for name in (
                "manifest.json",
                "maps.npy",
                "aux.npy",
                "metadata.csv",
                "frequencies_hz.npy",
            )
        )
        for optional_name in ("radar_timing_valid_mask.npy", "range_aux.npy"):
            optional_path = session_dir / optional_name
            if optional_path.is_file():
                paths.append(optional_path)
    inventory: list[dict[str, Any]] = []
    for path in sorted(set(paths), key=lambda item: str(item.relative_to(cache_root))):
        if not path.is_file():
            raise RuntimeError(f"canonical feature-cache inventory is missing: {path}")
        inventory.append(
            {
                "path": str(path.relative_to(cache_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return _canonical_sha256(inventory), len(inventory)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _canonical_content_sha256(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    return _canonical_sha256(payload)


_ACQUISITION_V2_SCHEMA = "snn_rr.feature_cache_acquisition.v2"
BASE_OOF_AUTHORITY_SCHEMA = "snn_rr.base_oof_authority.v1"
_V2_SVD_INVENTORY_NAMES: dict[str, str] = {
    "spectra": "spectra.npy",
    "component_signals": "component_signals.npy",
    "attributes": "attributes.npy",
    "frequencies_hz": "frequencies_hz.npy",
    "metadata": "metadata.csv",
    "radar_timing_valid_mask": "radar_timing_valid_mask.npy",
}


def _read_strict_json(path: Path, *, label: str) -> dict[str, Any]:
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
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return document


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _resolve_bound_path(base: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    return path


def _row_fold_binding_sha256(
    cache_index: Sequence[Any], identity: Sequence[Any], fold: Sequence[Any]
) -> str:
    indices = np.asarray(cache_index)
    identities = np.asarray(identity).astype(str)
    folds = np.asarray(fold)
    if not (indices.ndim == identities.ndim == folds.ndim == 1) or not (
        len(indices) == len(identities) == len(folds)
    ):
        raise RuntimeError("base OOF row/fold binding arrays are inconsistent")
    try:
        integer_indices = indices.astype(np.int64)
        integer_folds = folds.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("base OOF row/fold binding is not integral") from error
    if not (
        np.array_equal(indices, integer_indices)
        and np.array_equal(folds, integer_folds)
    ):
        raise RuntimeError("base OOF row/fold binding is not integral")
    return _canonical_sha256(
        [
            {
                "cache_index": int(index),
                "identity": str(person),
                "fold": int(fold_id),
            }
            for index, person, fold_id in zip(
                integer_indices, identities, integer_folds, strict=True
            )
        ]
    )


def _exact_int_vector(value: Any, *, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise RuntimeError(f"{label} must be a one-dimensional integer vector")
    try:
        numeric = raw.astype(np.float64)
        integer = raw.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(f"{label} is not integral") from error
    if not np.isfinite(numeric).all() or not np.array_equal(
        numeric, integer.astype(np.float64)
    ):
        raise RuntimeError(f"{label} is not integral")
    return integer


def _base_oof_payloads(
    csv_path: Path,
    npz_path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], str]:
    """Read either the historical valid OOF or authority-v1 all-window form."""

    frame = pd.read_csv(csv_path).sort_values("cache_index", kind="stable").reset_index(
        drop=True
    )
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if {"index", "target", "prediction", "rr_std", "fold"} <= set(arrays):
        normalized = arrays
        layout = "legacy_valid_oof"
    elif {
        "cache_index",
        "identity",
        "fold",
        "reference_rr_bpm",
        "prediction_bpm",
        "rr_std_bpm",
    } <= set(arrays):
        normalized = {
            **arrays,
            "index": np.asarray(arrays["cache_index"]),
            "target": np.asarray(arrays["reference_rr_bpm"]),
            "prediction": np.asarray(arrays["prediction_bpm"]),
            "rr_std": np.asarray(arrays["rr_std_bpm"]),
        }
        layout = "identity_disjoint_all_windows_v2"
        if "rr_bpm" not in frame and "reference_rr_bpm" in frame:
            frame = frame.rename(columns={"reference_rr_bpm": "rr_bpm"})
    else:
        raise RuntimeError("base OOF NPZ does not match a supported bound layout")
    return frame, normalized, layout


def _validate_base_oof_authority(
    *,
    csv_path: Path,
    npz_path: Path,
    provenance_path: Path,
    svd_root_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Validate a label-free, identity-owned base OOF publication graph."""

    provenance_path = provenance_path.resolve()
    document = _read_strict_json(
        provenance_path, label="base OOF provenance"
    )
    if document.get("schema_version") != BASE_OOF_AUTHORITY_SCHEMA:
        raise RuntimeError("base OOF provenance schema is unsupported")
    declared_content = _require_sha256(
        document.get("content_sha256"), label="base OOF provenance content_sha256"
    )
    if declared_content != _canonical_content_sha256(document):
        raise RuntimeError("base OOF provenance canonical content hash mismatch")

    outputs = document.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError("base OOF provenance outputs binding is missing")
    bound_csv = _resolve_bound_path(
        provenance_path.parent, outputs.get("csv"), label="base OOF CSV"
    )
    bound_npz = _resolve_bound_path(
        provenance_path.parent, outputs.get("npz"), label="base OOF NPZ"
    )
    if bound_csv != csv_path.resolve() or bound_npz != npz_path.resolve():
        raise RuntimeError("base OOF provenance points to different CSV/NPZ files")
    first_hashes: dict[str, str] = {}
    for name, path in (("csv", bound_csv), ("npz", bound_npz)):
        declared_bytes = outputs.get(f"{name}_bytes")
        if type(declared_bytes) is not int or declared_bytes != path.stat().st_size:
            raise RuntimeError(f"base OOF {name.upper()} byte-count mismatch")
        declared_hash = _require_sha256(
            outputs.get(f"{name}_sha256"),
            label=f"base OOF {name.upper()} SHA-256",
        )
        observed_hash = sha256_file(path)
        if declared_hash != observed_hash:
            raise RuntimeError(f"base OOF {name.upper()} SHA-256 mismatch")
        first_hashes[name] = observed_hash

    frame, arrays, layout = _base_oof_payloads(bound_csv, bound_npz)
    for name, path in (("csv", bound_csv), ("npz", bound_npz)):
        if sha256_file(path) != first_hashes[name]:
            raise RuntimeError(f"base OOF {name.upper()} changed while loading")
    if layout != "identity_disjoint_all_windows_v2":
        raise RuntimeError(
            "scientific base OOF authority requires identity-bound all-window outputs"
        )

    if (
        document.get("scientific_eligible") is not True
        or document.get("claim_classification")
        != "retrospective_scientific_noncommercial"
        or document.get("commercial_claim_allowed") is not False
    ):
        raise RuntimeError("base OOF provenance is not scientifically eligible")
    if document.get("identity_disjoint") is not True or document.get(
        "row_exact_cover"
    ) is not True:
        raise RuntimeError("base OOF lacks identity-disjoint exact-cover evidence")
    label_free = document.get("label_free_forward")
    if not isinstance(label_free, Mapping) or (
        label_free.get("verified") is not True
        or label_free.get("model_inputs") != ["map", "radar_mask", "aux"]
        or label_free.get("target_or_qc_inputs") != []
    ):
        raise RuntimeError("base OOF lacks exact label-free forward evidence")

    cache_binding = document.get("canonical_cache_provenance")
    if not isinstance(cache_binding, Mapping):
        raise RuntimeError("base OOF canonical cache provenance is missing")
    if cache_binding.get("content_sha256") != _canonical_sha256(
        {
            key: value
            for key, value in cache_binding.items()
            if key != "content_sha256"
        }
    ):
        raise RuntimeError("base OOF canonical cache provenance hash mismatch")
    if (
        cache_binding.get("classification") != "acquisition_scientific"
        or cache_binding.get("acquisition_schema_version")
        != _ACQUISITION_V2_SCHEMA
        or cache_binding.get("acquisition_mode") != "strict"
        or cache_binding.get("scientific_eligible") is not True
    ):
        raise RuntimeError("base OOF canonical cache is not scientific acquisition-v2")
    canonical_manifest_path = _resolve_bound_path(
        provenance_path.parent,
        cache_binding.get("root_manifest_path"),
        label="base OOF canonical cache manifest",
    )
    canonical_cache_value = svd_root_manifest.get("canonical_cache")
    if not isinstance(canonical_cache_value, str) or not canonical_cache_value:
        raise RuntimeError("SVD canonical cache path is missing")
    canonical_cache_path = Path(canonical_cache_value)
    if not canonical_cache_path.is_absolute():
        canonical_cache_path = PROJECT_ROOT / canonical_cache_path
    if canonical_manifest_path.parent != canonical_cache_path.resolve():
        raise RuntimeError("base OOF/SVD canonical cache path mismatch")
    if sha256_file(canonical_manifest_path) != _require_sha256(
        cache_binding.get("root_manifest_sha256"),
        label="base OOF canonical cache manifest SHA-256",
    ):
        raise RuntimeError("base OOF canonical cache manifest changed")
    canonical_manifest = _read_strict_json(
        canonical_manifest_path, label="base OOF canonical cache manifest"
    )
    if (
        canonical_manifest.get("content_sha256")
        != cache_binding.get("root_manifest_content_sha256")
        or canonical_manifest.get("content_sha256")
        != _canonical_content_sha256(canonical_manifest)
    ):
        raise RuntimeError("base OOF canonical cache content hash mismatch")
    canonical_items = canonical_manifest.get("sessions")
    if not isinstance(canonical_items, list):
        raise RuntimeError("base OOF canonical cache session catalogue is missing")
    canonical_session_ids = [
        str(item.get("session_id"))
        for item in canonical_items
        if isinstance(item, Mapping) and item.get("status") == "ok"
    ]
    if canonical_session_ids != cache_binding.get("selected_sessions"):
        raise RuntimeError("base OOF canonical cache selected sessions mismatch")
    canonical_inventory_sha256, canonical_inventory_count = (
        _feature_cache_inventory_sha256(
            canonical_manifest_path.parent, canonical_manifest
        )
    )
    if (
        canonical_inventory_sha256 != cache_binding.get("inventory_sha256")
        or canonical_inventory_count != cache_binding.get("inventory_file_count")
    ):
        raise RuntimeError("base OOF canonical cache inventory changed")
    canonical_contract = svd_root_manifest.get("canonical_acquisition_contract")
    if not isinstance(canonical_contract, Mapping):
        raise RuntimeError("SVD root canonical acquisition contract is missing")
    if (
        canonical_contract.get("schema_version") != _ACQUISITION_V2_SCHEMA
        or canonical_contract.get("mode") != "strict"
        or canonical_contract.get("scientific_eligible") is not True
    ):
        raise RuntimeError("SVD canonical cache is not strict/scientifically eligible")
    if cache_binding.get("root_manifest_sha256") != svd_root_manifest.get(
        "canonical_manifest_sha256"
    ):
        raise RuntimeError("base OOF/SVD canonical cache manifest hash mismatch")
    reconstruction_hash = svd_root_manifest.get(
        "canonical_acquisition_reconstruction_content_sha256"
    )
    if (
        cache_binding.get("reconstruction_content_sha256") != reconstruction_hash
        or canonical_contract.get("reconstruction_content_sha256")
        != reconstruction_hash
    ):
        raise RuntimeError("base OOF/SVD reconstruction hash mismatch")
    expected_sessions = canonical_contract.get("expected_usable_session_ids")
    if cache_binding.get("selected_sessions") != expected_sessions:
        raise RuntimeError("base OOF/SVD canonical session coverage mismatch")
    reconstruction_value = canonical_contract.get("reconstruction_manifest")
    if not isinstance(reconstruction_value, str) or not reconstruction_value:
        raise RuntimeError("SVD canonical reconstruction path is missing")
    reconstruction_path = Path(reconstruction_value)
    if not reconstruction_path.is_absolute():
        reconstruction_path = canonical_manifest_path.parent / reconstruction_path
    reconstruction_path = reconstruction_path.resolve()
    reconstruction = _read_strict_json(
        reconstruction_path, label="SVD canonical acquisition reconstruction"
    )
    if (
        reconstruction.get("content_sha256") != reconstruction_hash
        or _canonical_content_sha256(reconstruction) != reconstruction_hash
    ):
        raise RuntimeError("SVD canonical acquisition reconstruction changed")

    run_signature = document.get("run_signature")
    if not isinstance(run_signature, str) or not run_signature:
        raise RuntimeError("base OOF run signature is missing")
    run_config_path = _resolve_bound_path(
        provenance_path.parent,
        document.get("source_run_config"),
        label="base OOF source run config",
    )
    if sha256_file(run_config_path) != _require_sha256(
        document.get("source_run_config_sha256"),
        label="base OOF source run config SHA-256",
    ):
        raise RuntimeError("base OOF source run config SHA-256 mismatch")
    run_config = _read_strict_json(run_config_path, label="base OOF source run config")
    if run_config.get("run_signature") != run_signature:
        raise RuntimeError("base OOF source run signature mismatch")
    run_arguments = run_config.get("arguments")
    if not isinstance(run_arguments, Mapping) or (
        run_arguments.get("cache_trust_mode") != "scientific"
        or run_config.get("claim_classification")
        != "retrospective_scientific_noncommercial"
    ):
        raise RuntimeError("base OOF source run is not a scientific acquisition run")
    run_cache_binding = run_config.get("cache_provenance")
    if not isinstance(run_cache_binding, Mapping) or dict(run_cache_binding) != dict(
        cache_binding
    ):
        raise RuntimeError("base OOF source run/canonical cache provenance mismatch")

    source_hashes = document.get("runtime_source_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise RuntimeError("base OOF runtime source binding is missing")
    for relative, digest in source_hashes.items():
        _require_sha256(digest, label=f"base OOF source {relative} SHA-256")
        source_path = Path(str(relative))
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        if not source_path.is_file() or sha256_file(source_path) != digest:
            raise RuntimeError(f"base OOF runtime source changed: {relative}")

    owners = document.get("identity_to_test_fold")
    if not isinstance(owners, Mapping) or not owners:
        raise RuntimeError("base OOF identity_to_test_fold is missing")
    identity_to_fold: dict[str, int] = {}
    for identity, fold_value in owners.items():
        if (
            not isinstance(identity, str)
            or not identity
            or type(fold_value) is not int
            or not 0 <= fold_value < 6
        ):
            raise RuntimeError("base OOF identity_to_test_fold is invalid")
        identity_to_fold[identity] = fold_value
    if document.get("identity_to_test_fold_sha256") != _canonical_sha256(
        identity_to_fold
    ):
        raise RuntimeError("base OOF identity-to-fold hash mismatch")

    csv_index = _exact_int_vector(
        pd.to_numeric(frame["cache_index"], errors="raise").to_numpy(),
        label="base OOF CSV cache_index",
    )
    csv_identity = frame["identity"].astype(str).to_numpy()
    csv_fold = _exact_int_vector(
        pd.to_numeric(frame["fold"], errors="raise").to_numpy(),
        label="base OOF CSV fold",
    )
    npz_index = _exact_int_vector(
        arrays["index"], label="base OOF NPZ cache_index"
    )
    npz_identity = np.asarray(arrays["identity"]).astype(str)
    npz_fold = _exact_int_vector(arrays["fold"], label="base OOF NPZ fold")
    if not (
        np.array_equal(csv_index, npz_index)
        and np.array_equal(csv_identity, npz_identity)
        and np.array_equal(csv_fold, npz_fold)
        and len(np.unique(csv_index)) == len(csv_index)
        and set(csv_identity) == set(identity_to_fold)
    ):
        raise RuntimeError("base OOF CSV/NPZ row identity/fold binding mismatch")
    expected_folds = np.asarray(
        [identity_to_fold[identity] for identity in csv_identity], dtype=np.int64
    )
    if not np.array_equal(csv_fold, expected_folds):
        raise RuntimeError("base OOF row was not predicted by its identity test fold")
    _assert_close_numbers(
        arrays["target"], frame["rr_bpm"], "authority NPZ target", atol=5e-5
    )
    _assert_close_numbers(
        arrays["prediction"],
        frame["prediction_bpm"],
        "authority NPZ prediction",
        atol=5e-5,
    )
    _assert_close_numbers(
        arrays["rr_std"], frame["rr_std_bpm"], "authority NPZ rr_std", atol=5e-5
    )
    row_binding = _row_fold_binding_sha256(csv_index, csv_identity, csv_fold)
    if document.get("row_fold_binding_sha256") != row_binding:
        raise RuntimeError("base OOF row/fold binding hash mismatch")
    if type(document.get("row_count")) is not int or document.get(
        "row_count"
    ) != len(frame):
        raise RuntimeError("base OOF provenance row count mismatch")

    checkpoints = document.get("checkpoints")
    expected_fold_keys = {str(value) for value in range(N_FOLDS)}
    if set(identity_to_fold.values()) != set(range(N_FOLDS)):
        raise RuntimeError("base OOF identity owners do not cover exact folds 0..5")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != expected_fold_keys:
        raise RuntimeError("base OOF checkpoint catalogue does not match its folds")
    for fold_key in sorted(expected_fold_keys, key=int):
        record = checkpoints.get(fold_key)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"base OOF checkpoint {fold_key} binding is invalid")
        checkpoint_path = _resolve_bound_path(
            provenance_path.parent,
            record.get("path"),
            label=f"base OOF checkpoint {fold_key}",
        )
        checkpoint_hash = _require_sha256(
            record.get("sha256"), label=f"base OOF checkpoint {fold_key} SHA-256"
        )
        if sha256_file(checkpoint_path) != checkpoint_hash:
            raise RuntimeError(f"base OOF checkpoint {fold_key} SHA-256 mismatch")
        expected_test_identities = sorted(
            identity
            for identity, owner in identity_to_fold.items()
            if owner == int(fold_key)
        )
        if sorted(map(str, record.get("test_identities", ()))) != (
            expected_test_identities
        ):
            raise RuntimeError(
                f"base OOF checkpoint {fold_key} test identities mismatch"
            )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_split = (
            checkpoint.get("split") if isinstance(checkpoint, Mapping) else None
        )
        normalized_split: dict[str, set[str]] = {}
        split_is_valid = isinstance(checkpoint_split, Mapping)
        if isinstance(checkpoint_split, Mapping):
            for split_name in (
                "train_identities",
                "validation_identities",
                "test_identities",
            ):
                values = checkpoint_split.get(split_name)
                if (
                    not isinstance(values, (list, tuple))
                    or not values
                    or any(not isinstance(value, str) or not value for value in values)
                ):
                    split_is_valid = False
                    normalized_split[split_name] = set()
                else:
                    normalized_split[split_name] = set(values)
            split_sets = list(normalized_split.values())
            split_is_valid = bool(
                split_is_valid
                and not (split_sets[0] & split_sets[1])
                and not (split_sets[0] & split_sets[2])
                and not (split_sets[1] & split_sets[2])
                and set.union(*split_sets) == set(identity_to_fold)
                and normalized_split["test_identities"]
                == set(expected_test_identities)
            )
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("model_type") != "snn"
            or checkpoint.get("run_signature") != run_signature
            or checkpoint.get("fold") != int(fold_key)
            or checkpoint.get("cache_provenance") != cache_binding
            or not split_is_valid
        ):
            raise RuntimeError(
                f"base OOF checkpoint {fold_key} internal authority mismatch"
            )

    inference_signature = document.get("inference_signature")
    if not isinstance(inference_signature, str) or not inference_signature:
        raise RuntimeError("base OOF inference signature is missing")
    commits = document.get("verified_fold_commits")
    if not isinstance(commits, Mapping) or set(commits) != expected_fold_keys:
        raise RuntimeError("base OOF verified fold-commit catalogue is incomplete")
    committed_indices: list[np.ndarray] = []
    for fold_key in sorted(expected_fold_keys, key=int):
        fold_number = int(fold_key)
        marker = commits.get(fold_key)
        checkpoint_record = checkpoints[fold_key]
        if not isinstance(marker, Mapping):
            raise RuntimeError(f"base OOF fold {fold_key} commit is invalid")
        if (
            marker.get("fold") != fold_number
            or marker.get("run_signature") != run_signature
            or marker.get("inference_signature") != inference_signature
            or marker.get("checkpoint_sha256")
            != checkpoint_record.get("sha256")
            or marker.get("frozen_oof_sha256")
            != document.get("frozen_valid_oof_verification", {}).get(
                "source_sha256"
            )
            or marker.get("deployment_allowlist") is None
            or "target" in marker.get("deployment_allowlist", ())
            or "observable" in marker.get("deployment_allowlist", ())
            or sorted(map(str, marker.get("excluded_fields", ())))
            != ["observable", "target"]
        ):
            raise RuntimeError(f"base OOF fold {fold_key} commit binding mismatch")
        artifact = marker.get("artifact")
        if not isinstance(artifact, Mapping):
            raise RuntimeError(f"base OOF fold {fold_key} artifact binding is missing")
        artifact_path = _resolve_bound_path(
            provenance_path.parent,
            artifact.get("path"),
            label=f"base OOF fold {fold_key} artifact",
        )
        if (
            sha256_file(artifact_path)
            != _require_sha256(
                artifact.get("sha256"),
                label=f"base OOF fold {fold_key} artifact SHA-256",
            )
            or type(artifact.get("bytes")) is not int
            or artifact.get("bytes") != artifact_path.stat().st_size
        ):
            raise RuntimeError(f"base OOF fold {fold_key} artifact content mismatch")
        with np.load(artifact_path, allow_pickle=False) as fold_archive:
            required_fold_fields = {
                "index",
                "prediction",
                "rr_std",
                "fold",
                "run_signature",
                "inference_signature",
                "checkpoint_sha256",
            }
            if not required_fold_fields <= set(fold_archive.files) or (
                {"target", "observable"} & set(fold_archive.files)
            ):
                raise RuntimeError(
                    f"base OOF fold {fold_key} artifact field firewall failed"
                )
            artifact_indices = np.asarray(fold_archive["index"], dtype=np.int64)
            artifact_folds = np.asarray(fold_archive["fold"], dtype=np.int64)
            expected_rows = np.flatnonzero(csv_fold == fold_number)
            if (
                not np.array_equal(artifact_indices, csv_index[expected_rows])
                or not np.all(artifact_folds == fold_number)
                or str(np.asarray(fold_archive["run_signature"]).item())
                != run_signature
                or str(np.asarray(fold_archive["inference_signature"]).item())
                != inference_signature
                or str(np.asarray(fold_archive["checkpoint_sha256"]).item())
                != checkpoint_record.get("sha256")
            ):
                raise RuntimeError(
                    f"base OOF fold {fold_key} artifact row/authority mismatch"
                )
            _assert_close_numbers(
                fold_archive["prediction"],
                np.asarray(arrays["prediction"])[expected_rows],
                f"fold {fold_key} artifact prediction",
                atol=5e-5,
            )
            _assert_close_numbers(
                fold_archive["rr_std"],
                np.asarray(arrays["rr_std"])[expected_rows],
                f"fold {fold_key} artifact rr_std",
                atol=5e-5,
            )
        if (
            marker.get("row_count") != len(artifact_indices)
            or marker.get("expected_index_sha256")
            != hashlib.sha256(artifact_indices.astype(np.int64).tobytes()).hexdigest()
        ):
            raise RuntimeError(f"base OOF fold {fold_key} commit row digest mismatch")
        committed_indices.append(artifact_indices)
    if not np.array_equal(np.sort(np.concatenate(committed_indices)), np.sort(csv_index)):
        raise RuntimeError("base OOF fold artifacts do not exactly cover output rows")

    frozen_record = document.get("frozen_valid_oof_verification")
    if not isinstance(frozen_record, Mapping):
        raise RuntimeError("base OOF frozen-valid authority is missing")
    frozen_path = _resolve_bound_path(
        provenance_path.parent,
        frozen_record.get("source"),
        label="base OOF frozen valid OOF",
    )
    frozen_hash = _require_sha256(
        frozen_record.get("source_sha256"),
        label="base OOF frozen valid OOF SHA-256",
    )
    if sha256_file(frozen_path) != frozen_hash:
        raise RuntimeError("base OOF frozen valid OOF SHA-256 mismatch")
    resolved_tolerance = frozen_record.get("resolved_tolerance")
    if not isinstance(resolved_tolerance, Mapping):
        raise RuntimeError("base OOF frozen parity tolerance is missing")
    try:
        prediction_tolerance = float(resolved_tolerance["prediction_bpm"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("base OOF frozen prediction tolerance is invalid") from error
    if not math.isfinite(prediction_tolerance) or not 0.0 <= prediction_tolerance <= 0.4:
        raise RuntimeError("base OOF frozen prediction tolerance exceeds authority cap")
    with np.load(frozen_path, allow_pickle=False) as frozen_archive:
        required_frozen = {"index", "target", "prediction", "fold", "run_signature"}
        if not required_frozen <= set(frozen_archive.files):
            raise RuntimeError("base OOF frozen OOF binding arrays are incomplete")
        if str(np.asarray(frozen_archive["run_signature"]).item()) != run_signature:
            raise RuntimeError("base OOF frozen OOF run signature mismatch")
        frozen_index = np.asarray(frozen_archive["index"], dtype=np.int64)
        frozen_fold = np.asarray(frozen_archive["fold"], dtype=np.int64)
        if len(np.unique(frozen_index)) != len(frozen_index):
            raise RuntimeError("base OOF frozen OOF contains duplicate indices")
        output_position = {int(index): pos for pos, index in enumerate(csv_index)}
        if not set(map(int, frozen_index)) <= set(output_position):
            raise RuntimeError("base OOF frozen OOF indices are absent from all-window output")
        output_rows = np.asarray(
            [output_position[int(index)] for index in frozen_index], dtype=np.int64
        )
        if not np.array_equal(frozen_fold, csv_fold[output_rows]):
            raise RuntimeError("base OOF frozen OOF fold binding mismatch")
        _assert_close_numbers(
            frozen_archive["target"],
            np.asarray(arrays["target"])[output_rows],
            "frozen OOF target",
            atol=5e-5,
        )
        _assert_close_numbers(
            frozen_archive["prediction"],
            np.asarray(arrays["prediction"])[output_rows],
            "frozen OOF prediction",
            atol=prediction_tolerance,
        )

    return frame, arrays, {
        "classification": "verified_scientific_base_oof",
        "scientific_eligible": True,
        "provenance_path": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "provenance_content_sha256": declared_content,
        "csv_sha256": first_hashes["csv"],
        "npz_sha256": first_hashes["npz"],
        "run_signature": run_signature,
        "row_fold_binding_sha256": row_binding,
        "canonical_cache_provenance_content_sha256": cache_binding.get(
            "content_sha256"
        ),
    }


def _binding_is_acquisition_v2(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("schema_version") == _ACQUISITION_V2_SCHEMA
    )


def _session_has_acquisition_v2_indicator(manifest: Mapping[str, Any]) -> bool:
    inventory = manifest.get("file_inventory")
    return bool(
        _binding_is_acquisition_v2(manifest.get("canonical_acquisition_binding"))
        or manifest.get("canonical_acquisition_session_manifest_sha256") is not None
        or manifest.get("radar_timing_valid_mask_shape") is not None
        or manifest.get("radar_timing_summary") is not None
        or (
            isinstance(inventory, Mapping)
            and "radar_timing_valid_mask" in inventory
        )
    )


def _manifest_session_ids(value: Any, *, location: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{location} must be a non-empty session-ID list")
    session_ids: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise RuntimeError(f"{location} contains an invalid session ID")
        session_ids.append(item)
    if len(set(session_ids)) != len(session_ids):
        raise RuntimeError(f"{location} contains duplicate session IDs")
    return session_ids


def _validate_acquisition_v2_svd_scope(
    root_manifest: Mapping[str, Any],
    session_manifests: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Derive, rather than trust, an acquisition-v2 SVD scientific claim."""

    root_contract = root_manifest.get("canonical_acquisition_contract")
    root_sessions_value = root_manifest.get("sessions")
    root_sessions = (
        root_sessions_value if isinstance(root_sessions_value, list) else []
    )
    indicated_v2 = bool(
        _binding_is_acquisition_v2(root_contract)
        or root_manifest.get(
            "canonical_acquisition_reconstruction_content_sha256"
        )
        is not None
        or any(
            isinstance(item, Mapping)
            and _session_has_acquisition_v2_indicator(item)
            for item in root_sessions
        )
        or any(
            _session_has_acquisition_v2_indicator(manifest)
            for manifest in session_manifests.values()
        )
    )
    if not indicated_v2:
        return False
    if not _binding_is_acquisition_v2(root_contract):
        raise RuntimeError(
            "acquisition-v2 SVD input lacks its canonical root acquisition contract"
        )
    if root_manifest.get("content_sha256") != _canonical_content_sha256(
        root_manifest
    ):
        raise RuntimeError("acquisition-v2 SVD root content hash mismatch")

    expected = _manifest_session_ids(
        root_manifest.get("expected_session_ids"),
        location="SVD expected_session_ids",
    )
    selected = _manifest_session_ids(
        root_manifest.get("selected_session_ids"),
        location="SVD selected_session_ids",
    )
    if root_manifest.get("expected_session_ids_sha256") != _canonical_sha256(
        expected
    ):
        raise RuntimeError("SVD expected-session ID hash mismatch")
    if root_manifest.get("selected_session_ids_sha256") != _canonical_sha256(
        selected
    ):
        raise RuntimeError("SVD selected-session ID hash mismatch")
    subjects_filter_applied = root_manifest.get("subjects_filter_applied")
    if type(subjects_filter_applied) is not bool:
        raise RuntimeError("SVD subjects_filter_applied must be an explicit boolean")
    if not isinstance(root_sessions_value, list) or any(
        not isinstance(item, dict) for item in root_sessions_value
    ):
        raise RuntimeError("acquisition-v2 SVD root session catalogue is malformed")
    catalogue_ids = _manifest_session_ids(
        [item.get("session_id") for item in root_sessions_value],
        location="SVD root catalogue session IDs",
    )
    if catalogue_ids != selected:
        raise RuntimeError("SVD selected-session IDs differ from its root catalogue")

    selection_is_full = bool(not subjects_filter_applied and selected == expected)
    derived_scope = "full_cohort" if selection_is_full else "diagnostic_subset"
    if root_manifest.get("selection_scope") != derived_scope:
        raise RuntimeError("SVD selection_scope does not match its selection evidence")

    assert isinstance(root_contract, Mapping)
    canonical_expected = _manifest_session_ids(
        root_contract.get("expected_usable_session_ids"),
        location="canonical acquisition expected usable-session IDs",
    )
    canonical_cache = _manifest_session_ids(
        root_contract.get("cache_usable_session_ids"),
        location="canonical acquisition cache usable-session IDs",
    )
    if root_contract.get(
        "expected_usable_session_ids_sha256"
    ) != _canonical_sha256(canonical_expected):
        raise RuntimeError("canonical acquisition expected-session hash mismatch")
    if root_contract.get("cache_usable_session_ids_sha256") != _canonical_sha256(
        canonical_cache
    ):
        raise RuntimeError("canonical acquisition cache-session hash mismatch")
    if canonical_expected != expected or canonical_cache != expected:
        raise RuntimeError("SVD canonical cache/cohort coverage mismatch")

    canonical_full = bool(
        root_contract.get("selection_scope") == "full_cohort"
        and root_contract.get("full_cohort_complete") is True
        and root_contract.get("reconstruction_full_cohort_complete") is True
    )
    all_results_ok = bool(root_sessions_value) and all(
        item.get("status") == "ok" for item in root_sessions_value
    )
    derived_complete = bool(selection_is_full and all_results_ok and canonical_full)
    if type(root_manifest.get("full_cohort_complete")) is not bool or (
        root_manifest.get("full_cohort_complete") != derived_complete
    ):
        raise RuntimeError(
            "SVD full_cohort_complete does not match its bound cohort evidence"
        )
    if set(session_manifests) != set(selected):
        raise RuntimeError("SVD child manifest set differs from selected sessions")

    bindings_scientific = True
    for item in root_sessions_value:
        session_id = str(item["session_id"])
        child = session_manifests[session_id]
        declared_child_content = child.get("content_sha256")
        if (
            not isinstance(declared_child_content, str)
            or declared_child_content != _canonical_content_sha256(child)
            or item.get("content_sha256") != declared_child_content
        ):
            raise RuntimeError(f"SVD child manifest content mismatch: {session_id}")
        root_binding = item.get("canonical_acquisition_binding")
        child_binding = child.get("canonical_acquisition_binding")
        if root_binding != child_binding:
            raise RuntimeError(
                f"SVD root/child acquisition binding mismatch: {session_id}"
            )
        bindings_scientific = bool(
            bindings_scientific
            and isinstance(child_binding, Mapping)
            and child_binding.get("schema_version") == _ACQUISITION_V2_SCHEMA
            and child_binding.get("scientific_eligible") is True
        )
    derived_scientific = bool(
        derived_complete
        and root_contract.get("mode") == "strict"
        and root_contract.get("scientific_eligible") is True
        and bindings_scientific
    )
    if type(root_manifest.get("scientific_eligible")) is not bool or (
        root_manifest.get("scientific_eligible") != derived_scientific
    ):
        raise RuntimeError(
            "SVD scientific_eligible does not match its acquisition evidence"
        )
    return derived_scientific


def _validate_v2_svd_inventory(
    session_dir: Path,
    root_item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    require_all_timing_valid: bool,
    detach_from_mutable_source: bool,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, str]]:
    """Verify every acquisition-v2 SVD payload against its bound inventory.

    Scientific training must not retain a live mmap into cache files after
    verification.  In that mode every array is copied into owned memory and
    made read-only before a final source-file rehash closes the load/copy
    interval.  Diagnostic acquisition inputs may retain the historical mmap
    behavior because they cannot acquire a scientific claim.
    """

    session_id = str(root_item.get("session_id", ""))
    declared_content = manifest.get("content_sha256")
    if (
        not isinstance(declared_content, str)
        or declared_content != _canonical_content_sha256(manifest)
        or root_item.get("content_sha256") != declared_content
    ):
        raise RuntimeError(f"SVD v2 manifest content mismatch for {session_id}")
    if any(root_item.get(key) != value for key, value in manifest.items()):
        raise RuntimeError(f"SVD v2 root/child manifest mismatch for {session_id}")
    root_binding = root_item.get("canonical_acquisition_binding")
    child_binding = manifest.get("canonical_acquisition_binding")
    if not _binding_is_acquisition_v2(child_binding) or root_binding != child_binding:
        raise RuntimeError(
            f"SVD v2 root/child acquisition binding mismatch for {session_id}"
        )

    inventory = manifest.get("file_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != set(
        _V2_SVD_INVENTORY_NAMES
    ):
        raise RuntimeError(f"SVD v2 file inventory is incomplete for {session_id}")
    if manifest.get("inventory_sha256") != _canonical_sha256(inventory):
        raise RuntimeError(f"SVD v2 inventory hash mismatch for {session_id}")
    if root_item.get("inventory_sha256") != manifest.get("inventory_sha256"):
        raise RuntimeError(f"SVD v2 root/child inventory mismatch for {session_id}")

    actual_hashes: dict[str, str] = {}
    arrays: dict[str, np.ndarray] = {}
    metadata: pd.DataFrame | None = None
    for name, expected_name in _V2_SVD_INVENTORY_NAMES.items():
        binding = inventory.get(name)
        if not isinstance(binding, Mapping) or binding.get("path") != expected_name:
            raise RuntimeError(
                f"SVD v2 inventory redirects {name} for {session_id}"
            )
        path = (session_dir / expected_name).resolve()
        try:
            path.relative_to(session_dir.resolve())
        except ValueError as error:
            raise RuntimeError(
                f"SVD v2 inventory path escapes its session for {session_id}/{name}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(
                f"SVD v2 inventory payload is missing for {session_id}/{name}: {path}"
            )
        declared_bytes = binding.get("bytes")
        if (
            type(declared_bytes) is not int
            or declared_bytes < 0
            or path.stat().st_size != declared_bytes
        ):
            raise RuntimeError(
                f"SVD v2 inventory byte count mismatch for {session_id}/{name}"
            )
        actual_sha = sha256_file(path)
        actual_hashes[name] = actual_sha
        if binding.get("sha256") != actual_sha:
            raise RuntimeError(
                f"SVD v2 inventory SHA-256 mismatch for {session_id}/{name}"
            )
        declared_shape = binding.get("shape")
        declared_dtype = binding.get("dtype")
        if not isinstance(declared_shape, list) or any(
            type(value) is not int or value < 0 for value in declared_shape
        ):
            raise RuntimeError(
                f"SVD v2 inventory shape is invalid for {session_id}/{name}"
            )
        if name == "metadata":
            if declared_dtype != "csv":
                raise RuntimeError(
                    f"SVD v2 metadata dtype is invalid for {session_id}"
                )
            metadata = pd.read_csv(path)
            if list(metadata.shape) != declared_shape:
                raise RuntimeError(
                    f"SVD v2 metadata shape mismatch for {session_id}"
                )
        else:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            arrays[name] = array
            if list(array.shape) != declared_shape or str(array.dtype) != declared_dtype:
                raise RuntimeError(
                    f"SVD v2 array declaration mismatch for {session_id}/{name}"
                )

    timing = arrays["radar_timing_valid_mask"]
    spectra = arrays["spectra"]
    component_signals = arrays["component_signals"]
    attributes = arrays["attributes"]
    frequencies = arrays["frequencies_hz"]
    if metadata is None:
        raise RuntimeError(f"SVD v2 metadata inventory is absent for {session_id}")
    row_count = len(metadata)
    if (
        timing.dtype != np.bool_
        or timing.shape != (row_count, 3, 320)
        or spectra.ndim != 5
        or spectra.shape[0] != row_count
        or spectra.shape[1] != 3
        or component_signals.shape != (*spectra.shape[:4], 320)
        or attributes.shape != (*spectra.shape[:4], 5)
        or frequencies.ndim != 1
        or spectra.shape[-1] != len(frequencies)
    ):
        raise RuntimeError(f"SVD v2 payload shapes are inconsistent for {session_id}")
    if manifest.get("radar_timing_valid_mask_shape") != list(timing.shape):
        raise RuntimeError(f"SVD v2 timing-mask shape claim mismatch for {session_id}")
    invalid_count = int(np.size(timing) - np.count_nonzero(timing))
    if manifest.get("radar_timing_invalid_interval_count") != invalid_count:
        raise RuntimeError(f"SVD v2 timing-mask count claim mismatch for {session_id}")
    expected_mask_contract = {
        "mask_required_for_gap_tolerant_consumers": True,
        "scientific_source_requires_all_true": True,
        "diagnostic_output_trainable": False,
        "invalid_cells_are_exact_zero_but_not_semantic_measurements": True,
    }
    if manifest.get("radar_timing_mask_contract") != expected_mask_contract:
        raise RuntimeError(f"SVD v2 timing-mask policy mismatch for {session_id}")
    if require_all_timing_valid and not bool(np.asarray(timing).all()):
        raise RuntimeError(
            f"scientific SVD input contains invalid radar timing for {session_id}"
        )
    if detach_from_mutable_source:
        detached: dict[str, np.ndarray] = {}
        for name, array in arrays.items():
            owned = np.array(array, copy=True, order="C")
            if not owned.flags.owndata or isinstance(owned, np.memmap):
                raise RuntimeError(
                    f"scientific SVD array was not detached for {session_id}/{name}"
                )
            owned.setflags(write=False)
            detached[name] = owned
        arrays = detached
    for name, expected_name in _V2_SVD_INVENTORY_NAMES.items():
        if sha256_file(session_dir / expected_name) != actual_hashes[name]:
            raise RuntimeError(
                f"SVD v2 inventory payload changed during validation for "
                f"{session_id}/{name}"
            )
    return arrays, metadata, actual_hashes


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def stable_signature(value: Mapping[str, Any], *, length: int = 16) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def validate_input_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Reject label/future/identity columns and anything outside the allow-list."""

    values = tuple(str(value) for value in columns)
    if not values:
        raise ValueError("at least one base feature column is required")
    if len(set(values)) != len(values):
        raise ValueError("base feature columns must be unique")
    forbidden = sorted(set(values) & LABEL_ONLY_COLUMNS)
    unknown = sorted(set(values) - set(BASE_FEATURE_COLUMNS))
    if forbidden or unknown:
        raise ValueError(
            "model inputs must be deployment-time allow-listed features; "
            f"forbidden={forbidden}, unknown={unknown}"
        )
    return values


# A descriptive alias used by tests/audits.
assert_no_future_target_leakage = validate_input_feature_columns


def resolve_base_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns = tuple(name for name in BASE_FEATURE_COLUMNS if name in frame.columns)
    return validate_input_feature_columns(columns)


def fit_robust_scaler(
    frame: pd.DataFrame, train_positions: Sequence[int], columns: Sequence[str]
) -> RobustScaler:
    columns = validate_input_feature_columns(columns)
    positions = np.asarray(train_positions, dtype=np.int64)
    if not len(positions):
        raise ValueError("cannot fit a scaler without training rows")
    raw = frame.iloc[positions].loc[:, list(columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    center = np.nanmedian(raw, axis=0)
    q25 = np.nanquantile(raw, 0.25, axis=0)
    q75 = np.nanquantile(raw, 0.75, axis=0)
    scale = q75 - q25
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return RobustScaler(columns, center.astype(np.float32), scale.astype(np.float32))


def _assert_equal_strings(left: pd.Series, right: pd.Series, name: str) -> None:
    a = left.fillna("").astype(str).to_numpy()
    b = right.fillna("").astype(str).to_numpy()
    if not np.array_equal(a, b):
        mismatch = np.flatnonzero(a != b)[:5]
        raise RuntimeError(f"source/base {name} binding mismatch at rows {mismatch.tolist()}")


def _assert_close_numbers(
    left: pd.Series | np.ndarray,
    right: pd.Series | np.ndarray,
    name: str,
    *,
    atol: float,
) -> None:
    a = np.asarray(pd.to_numeric(left, errors="coerce"), dtype=np.float64)
    b = np.asarray(pd.to_numeric(right, errors="coerce"), dtype=np.float64)
    if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True):
        mismatch = np.flatnonzero(~np.isclose(a, b, rtol=0.0, atol=atol, equal_nan=True))[:5]
        raise RuntimeError(f"source/base {name} binding mismatch at rows {mismatch.tolist()}")


def load_aligned_experiment(
    cache_root: Path,
    oof_csv: Path,
    oof_npz: Path | None = None,
    *,
    base_oof_provenance: Path | None = None,
    verify_file_hashes: bool = True,
) -> AlignedSVDExperiment:
    """Load memmapped SVD sessions and enforce an exact OOF row binding."""

    cache_root = Path(cache_root).resolve()
    oof_csv = Path(oof_csv).resolve()
    oof_npz = (
        Path(oof_npz).resolve()
        if oof_npz is not None
        else oof_csv.with_suffix(".npz").resolve()
    )
    root_manifest_path = cache_root / "manifest.json"
    if not (root_manifest_path.is_file() and oof_csv.is_file() and oof_npz.is_file()):
        raise FileNotFoundError("SVD manifest, base OOF CSV and base OOF NPZ are required")
    root_manifest = _read_strict_json(root_manifest_path, label="SVD root manifest")
    if not bool(root_manifest.get("valid_only", False)):
        raise RuntimeError("this trainer requires a valid-reference-only SVD cache")

    root_contract = root_manifest.get("canonical_acquisition_contract")
    root_is_v2 = _binding_is_acquisition_v2(root_contract)
    root_sessions_value = root_manifest.get("sessions")
    root_session_items = (
        root_sessions_value if isinstance(root_sessions_value, list) else []
    )
    root_indicates_v2 = bool(
        root_is_v2
        or root_manifest.get(
            "canonical_acquisition_reconstruction_content_sha256"
        )
        is not None
        or any(
            isinstance(item, Mapping)
            and _session_has_acquisition_v2_indicator(item)
            for item in root_session_items
        )
    )
    if root_indicates_v2 and not root_is_v2:
        raise RuntimeError(
            "acquisition-v2 SVD input lacks its canonical root acquisition contract"
        )
    if root_is_v2:
        declared_root_content = root_manifest.get("content_sha256")
        if (
            not isinstance(declared_root_content, str)
            or declared_root_content != _canonical_content_sha256(root_manifest)
        ):
            raise RuntimeError("acquisition-v2 SVD root content hash mismatch")
        if root_manifest.get("pipeline_sha256") != _current_svd_pipeline_sha256():
            raise RuntimeError("acquisition-v2 SVD pipeline source hash mismatch")
    root_scientific = root_manifest.get("scientific_eligible") is True

    sessions: list[SVDSessionArrays] = []
    metadata_parts: list[pd.DataFrame] = []
    common_frequencies: np.ndarray | None = None
    file_manifest: dict[str, Any] = {}
    listed = [item for item in root_session_items if item.get("status") == "ok"]
    if not listed:
        raise RuntimeError("SVD root manifest contains no usable sessions")
    for slot, item in enumerate(listed):
        session_id = str(item["session_id"])
        session_dir = (cache_root / session_id).resolve()
        try:
            session_dir.relative_to(cache_root)
        except ValueError as error:
            raise RuntimeError(f"SVD session path escapes its cache: {session_id}") from error
        paths = {
            "manifest": session_dir / "manifest.json",
            "metadata": session_dir / "metadata.csv",
            "spectra": session_dir / "spectra.npy",
            "attributes": session_dir / "attributes.npy",
            "frequencies_hz": session_dir / "frequencies_hz.npy",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete SVD session {session_id}: {missing}")
        manifest = _read_strict_json(
            paths["manifest"], label=f"SVD session manifest {session_id}"
        )
        child_indicates_v2 = _session_has_acquisition_v2_indicator(manifest)
        item_indicates_v2 = _session_has_acquisition_v2_indicator(item)
        if root_is_v2 != child_indicates_v2 or root_is_v2 != item_indicates_v2:
            raise RuntimeError(
                f"SVD acquisition-v2 provenance is partial for {session_id}"
            )
        component_signals: np.ndarray | None = None
        radar_timing_valid_mask: np.ndarray | None = None
        if root_is_v2:
            arrays, session_metadata, inventory_hashes = _validate_v2_svd_inventory(
                session_dir,
                item,
                manifest,
                require_all_timing_valid=root_scientific,
                detach_from_mutable_source=root_scientific,
            )
            spectra = arrays["spectra"]
            component_signals = arrays["component_signals"]
            attributes = arrays["attributes"]
            frequencies = arrays["frequencies_hz"]
            radar_timing_valid_mask = arrays["radar_timing_valid_mask"]
            hashes = {
                "manifest": sha256_file(paths["manifest"]),
                **inventory_hashes,
            }
        else:
            session_metadata = pd.read_csv(paths["metadata"])
            spectra = np.load(paths["spectra"], mmap_mode="r", allow_pickle=False)
            attributes = np.load(paths["attributes"], mmap_mode="r", allow_pickle=False)
            frequencies = np.load(paths["frequencies_hz"], allow_pickle=False)
            hashes = (
                {name: sha256_file(path) for name, path in paths.items()}
                if verify_file_hashes
                else {"manifest": sha256_file(paths["manifest"])}
            )
        count = len(session_metadata)
        if spectra.ndim != 5 or spectra.shape[:1] != (count,) or spectra.shape[1] != 3:
            raise RuntimeError(f"invalid spectra shape for {session_id}: {spectra.shape}")
        if attributes.shape != (*spectra.shape[:4], 5):
            raise RuntimeError(f"invalid attributes shape for {session_id}: {attributes.shape}")
        if spectra.shape[-1] != len(frequencies):
            raise RuntimeError(f"frequency/spectra mismatch for {session_id}")
        # Pipeline-v1 per-session manifests predate the explicit session_id
        # field; the directory/root-manifest binding remains authoritative.
        if "session_id" in manifest and str(manifest["session_id"]) != session_id:
            raise RuntimeError(f"session manifest identity mismatch for {session_id}")
        if int(manifest.get("row_count", -1)) != count:
            raise RuntimeError(f"session manifest row count mismatch for {session_id}")
        if common_frequencies is None:
            common_frequencies = np.asarray(frequencies, dtype=np.float32)
        elif not np.array_equal(common_frequencies, frequencies):
            raise RuntimeError("SVD physical frequency grid differs across sessions")
        if session_metadata["cache_index"].duplicated().any():
            raise RuntimeError(f"duplicate cache_index inside {session_id}")

        file_manifest[session_id] = hashes
        sessions.append(
            SVDSessionArrays(
                session_id=session_id,
                spectra=spectra,
                attributes=attributes,
                frequencies_hz=frequencies,
                metadata=session_metadata,
                manifest=manifest,
                files_sha256=hashes,
                radar_timing_valid_mask=radar_timing_valid_mask,
                component_signals=component_signals,
            )
        )
        part = session_metadata.copy()
        part["_session_slot"] = slot
        part["_local_row"] = np.arange(count, dtype=np.int64)
        metadata_parts.append(part)

    if common_frequencies is None:  # defensive; listed is non-empty
        raise RuntimeError("SVD frequency grid is unavailable")
    source = pd.concat(metadata_parts, ignore_index=True)
    source = source.sort_values("cache_index", kind="stable").reset_index(drop=True)
    if source["cache_index"].duplicated().any():
        raise RuntimeError("cache_index is duplicated across SVD sessions")
    if len(source) != int(root_manifest.get("row_count", -1)):
        raise RuntimeError("SVD root manifest row count does not bind loaded sessions")

    if root_is_v2:
        root_scientific = _validate_acquisition_v2_svd_scope(
            root_manifest,
            {session.session_id: session.manifest for session in sessions},
        )

    authority_path = (
        Path(base_oof_provenance).resolve()
        if base_oof_provenance is not None
        else (oof_csv.parent / "provenance.json").resolve()
    )
    if base_oof_provenance is not None and not authority_path.is_file():
        raise FileNotFoundError(
            f"explicit base OOF provenance is missing: {authority_path}"
        )
    if root_is_v2 and root_scientific:
        if not authority_path.is_file():
            raise RuntimeError(
                "scientific acquisition-v2 SVD training requires an explicit "
                "--base-oof-provenance or the exact CSV-sibling provenance.json"
            )
        base, base_arrays, base_authority = _validate_base_oof_authority(
            csv_path=oof_csv,
            npz_path=oof_npz,
            provenance_path=authority_path,
            svd_root_manifest=root_manifest,
        )
    else:
        base, base_arrays, base_layout = _base_oof_payloads(oof_csv, oof_npz)
        base_authority = {
            "classification": "historical_diagnostic_unverified_base_oof",
            "scientific_eligible": False,
            "layout": base_layout,
            "provenance_path": (
                str(authority_path) if authority_path.is_file() else None
            ),
            "reason": (
                "legacy or diagnostic SVD input; base predictions were not "
                "accepted as scientific OOF authority"
            ),
        }

    # Scientific authority is validated over the complete canonical cache.
    # The SVD cache is valid-reference-only, so select its exact immutable
    # cache indices only after validating the full publication and row owners.
    source_indices = source["cache_index"].to_numpy(dtype=np.int64)
    base_indices = pd.to_numeric(base["cache_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    if root_is_v2 and root_scientific:
        positions_by_index = {
            int(index): position for position, index in enumerate(base_indices)
        }
        if len(positions_by_index) != len(base_indices):
            raise RuntimeError("base OOF cache_index must be unique")
        missing_indices = sorted(set(source_indices) - set(positions_by_index))
        if missing_indices:
            raise RuntimeError(
                "scientific base OOF does not cover SVD cache indices: "
                f"{missing_indices[:5]}"
            )
        selected_positions = np.asarray(
            [positions_by_index[int(index)] for index in source_indices],
            dtype=np.int64,
        )
        base = base.iloc[selected_positions].reset_index(drop=True)
        base_arrays = {
            name: (
                np.asarray(value)[selected_positions]
                if np.asarray(value).ndim >= 1
                and len(np.asarray(value)) == len(base_indices)
                else np.asarray(value)
            )
            for name, value in base_arrays.items()
        }

    if base["cache_index"].duplicated().any():
        raise RuntimeError("base OOF cache_index must be unique")
    if not np.array_equal(
        source["cache_index"].to_numpy(dtype=np.int64),
        base["cache_index"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("SVD cache_index set/order does not exactly match base OOF")
    for column in ("session_id", "identity", "protocol"):
        _assert_equal_strings(source[column], base[column], column)
    for column, tolerance in (
        ("window_number", 0.0),
        ("window_start_s", 1e-6),
        ("window_end_s", 1e-6),
        ("rr_bpm", 5e-5),
    ):
        _assert_close_numbers(source[column], base[column], column, atol=tolerance)

    required = {"index", "target", "prediction", "rr_std", "fold"}
    missing = sorted(required - set(base_arrays))
    if missing:
        raise RuntimeError(f"base OOF NPZ is missing binding arrays: {missing}")
    if not np.array_equal(
        np.asarray(base_arrays["index"]).astype(np.int64),
        base["cache_index"].to_numpy(np.int64),
    ):
        raise RuntimeError("base OOF CSV/NPZ cache_index binding failed")
    _assert_close_numbers(
        base_arrays["target"], base["rr_bpm"], "NPZ target", atol=5e-5
    )
    _assert_close_numbers(
        base_arrays["prediction"],
        base["prediction_bpm"],
        "NPZ prediction",
        atol=5e-5,
    )
    if not np.array_equal(
        np.asarray(base_arrays["fold"]).astype(np.int64),
        base["fold"].to_numpy(np.int64),
    ):
        raise RuntimeError("base OOF CSV/NPZ fold binding failed")

    # Preserve SVD labels/metadata as the canonical side and copy only
    # deployment outputs plus the already-frozen fold from the base OOF.
    joined = source.copy()
    for column in base.columns:
        if column in BINDING_COLUMNS:
            continue
        if column in joined.columns:
            if column in ("reference_quality", "radar_observable", "classical_rr_bpm"):
                _assert_close_numbers(joined[column], base[column], column, atol=5e-5)
            continue
        joined[column] = base[column].to_numpy(copy=True)
    if "fold" not in joined:
        joined["fold"] = base["fold"].to_numpy(dtype=np.int16)
    folds = joined["fold"].to_numpy(dtype=np.int64)
    if set(np.unique(folds)) != set(range(N_FOLDS)):
        raise RuntimeError(f"expected frozen folds 0..5, got {sorted(np.unique(folds))}")
    if not np.isfinite(pd.to_numeric(joined["prediction_bpm"], errors="coerce")).all():
        raise RuntimeError("base predictions must be finite")

    binding_payload = joined.loc[:, ["cache_index", "identity", "fold", "rr_bpm"]].copy()
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "root_manifest_sha256": sha256_file(root_manifest_path),
        "base_oof_csv_sha256": sha256_file(oof_csv),
        "base_oof_npz_sha256": sha256_file(oof_npz),
        "base_oof_authority": base_authority,
        "claim_classification": (
            "retrospective_scientific_noncommercial"
            if base_authority["scientific_eligible"] is True
            else "historical_diagnostic_noncommercial"
        ),
        "session_files_sha256": file_manifest,
        "row_binding_sha256": hashlib.sha256(
            binding_payload.to_csv(index=False, float_format="%.9g").encode("utf-8")
        ).hexdigest(),
        "row_count": int(len(joined)),
        "feature_allowlist": list(resolve_base_feature_columns(joined)),
        "component_signals_status": (
            "acquisition_v2_hash_verified_not_model_input"
            if root_is_v2
            else "retained_for_future_temporal_extension_not_loaded_or_input"
        ),
        "radar_availability_authority": (
            "radar_timing_valid_mask_all_320_intervals"
            if root_is_v2
            else "legacy_numeric_fallback"
        ),
        "acquisition_v2_authority_hashes_mandatory": bool(root_is_v2),
    }
    return AlignedSVDExperiment(
        cache_root=cache_root,
        oof_csv=oof_csv,
        oof_npz=oof_npz,
        metadata=joined,
        sessions=sessions,
        frequencies_hz=common_frequencies,
        root_manifest=root_manifest,
        provenance=provenance,
    )


# Backward-friendly descriptive alias.
align_source_cache_and_oof = load_aligned_experiment


def assert_identity_disjoint_split(metadata: pd.DataFrame, split: FoldSplit) -> None:
    sets = [
        set(split.train_identities),
        set(split.validation_identities),
        set(split.test_identities),
    ]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("train/validation/test identities overlap")
    covered = np.concatenate((split.train, split.validation, split.test))
    if len(covered) != len(metadata) or not np.array_equal(
        np.sort(covered), np.arange(len(metadata), dtype=np.int64)
    ):
        raise RuntimeError("fold split does not cover every row exactly once")
    identity = metadata["identity"].astype(str).to_numpy()
    for positions, expected in (
        (split.train, sets[0]),
        (split.validation, sets[1]),
        (split.test, sets[2]),
    ):
        if set(identity[positions]) != expected:
            raise RuntimeError("fold split identity declaration is inconsistent")


def make_outer_split(
    metadata: pd.DataFrame, outer_fold: int, *, n_folds: int = N_FOLDS
) -> FoldSplit:
    if n_folds < 3 or not 0 <= int(outer_fold) < n_folds:
        raise ValueError("outer fold is outside the configured range")
    fold = pd.to_numeric(metadata["fold"], errors="raise").to_numpy(dtype=np.int64)
    if np.any((fold < 0) | (fold >= n_folds)):
        raise ValueError("metadata contains an invalid frozen fold")
    identity = metadata["identity"].astype(str).to_numpy()
    validation_fold = (int(outer_fold) + 1) % n_folds
    test = np.flatnonzero(fold == int(outer_fold))
    validation = np.flatnonzero(fold == validation_fold)
    train = np.flatnonzero((fold != int(outer_fold)) & (fold != validation_fold))
    if not (len(train) and len(validation) and len(test)):
        raise RuntimeError("empty train, validation, or test partition")
    split = FoldSplit(
        outer_fold=int(outer_fold),
        validation_fold=validation_fold,
        train=train,
        validation=validation,
        test=test,
        train_identities=tuple(sorted(set(identity[train]))),
        validation_identities=tuple(sorted(set(identity[validation]))),
        test_identities=tuple(sorted(set(identity[test]))),
    )
    assert_identity_disjoint_split(metadata, split)
    return split


def parse_fold_selection(value: str, *, n_folds: int = N_FOLDS) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(n_folds))
    try:
        selected = sorted({int(item.strip()) for item in value.split(",")})
    except ValueError as exc:
        raise ValueError("--fold must be 'all' or comma-separated integers") from exc
    if not selected or selected[0] < 0 or selected[-1] >= n_folds:
        raise ValueError(f"folds must lie in [0, {n_folds - 1}]")
    return selected


def identity_rr_tail_sample_weights(
    metadata: pd.DataFrame,
    positions: Sequence[int],
    *,
    rr_bin_width: float = 3.0,
    rr_balance_power: float = 0.65,
    tail_min_bpm: float = 25.0,
    tail_max_bpm: float = 35.0,
    tail_boost: float = 2.0,
) -> np.ndarray:
    """Equalize identity mass while rebalancing RR bins and the hard tail."""

    if rr_bin_width <= 0 or rr_balance_power < 0 or tail_boost <= 0:
        raise ValueError("sampler balancing parameters must be positive")
    positions = np.asarray(positions, dtype=np.int64)
    if not len(positions):
        raise ValueError("sampler needs at least one training row")
    rows = metadata.iloc[positions]
    identity = rows["identity"].astype(str).to_numpy()
    rr = pd.to_numeric(rows["rr_bpm"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(rr).all():
        raise ValueError("training sampler RR values must be finite")
    bucket = np.floor(rr / float(rr_bin_width)).astype(np.int64)
    names, counts = np.unique(bucket, return_counts=True)
    count_map = dict(zip(names.tolist(), counts.tolist(), strict=True))
    typical = float(np.median(counts))
    weights = np.asarray(
        [(typical / count_map[int(value)]) ** rr_balance_power for value in bucket],
        dtype=np.float64,
    )
    weights *= np.where(
        (rr >= float(tail_min_bpm)) & (rr <= float(tail_max_bpm)),
        float(tail_boost),
        1.0,
    )
    # Equal total mass per identity after RR/tail reweighting.
    for name in np.unique(identity):
        selected = identity == name
        weights[selected] /= weights[selected].sum()
    weights /= weights.mean()
    if not (np.isfinite(weights).all() and np.all(weights > 0)):
        raise FloatingPointError("sampler generated invalid weights")
    return weights


# Alternate name matching the generic trainer terminology.
identity_balanced_sample_weights = identity_rr_tail_sample_weights


class SVDSourceDataset(Dataset[dict[str, Tensor]]):
    """Dataset exposing only explicit deployment inputs plus training labels."""

    def __init__(
        self,
        experiment: AlignedSVDExperiment,
        positions: Sequence[int],
        base_features: np.ndarray,
    ) -> None:
        self.experiment = experiment
        self.positions = np.asarray(positions, dtype=np.int64)
        self.base_features = np.asarray(base_features, dtype=np.float32)
        if self.base_features.shape[0] != len(experiment.metadata):
            raise ValueError("base feature matrix must cover the aligned cache")

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        position = int(self.positions[item])
        row = self.experiment.metadata.iloc[position]
        spectra, attributes = self.experiment.arrays_for_position(position)
        # Memmaps opened read-only produce non-writeable views.  A per-sample
        # copy avoids undefined behavior in torch even though this trainer does
        # not intentionally mutate inputs.
        spectra_array = np.array(spectra, copy=True)
        attributes_array = np.array(attributes, copy=True)
        structural_mask = self.experiment.structural_radar_mask_for_position(position)
        if structural_mask is None:
            # Historical caches have no structural timing mask.  Preserve their
            # legacy numeric-availability behavior without allowing it to act as
            # authority for acquisition-v2 inputs.
            radar_mask = np.any(np.abs(spectra_array) > 0, axis=(1, 2, 3))
        else:
            radar_mask = np.asarray(structural_mask, dtype=np.bool_)
            if radar_mask.shape != (spectra_array.shape[0],):
                raise RuntimeError("structural radar mask has an invalid shape")
            spectra_array[~radar_mask] = 0
            attributes_array[~radar_mask] = 0
        base_feature_array = np.array(self.base_features[position], copy=True)
        classical = np.asarray(
            [
                row.get("classical_rr_bpm", np.nan),
                row.get("radar_peak_1_bpm", np.nan),
                row.get("radar_peak_2_bpm", np.nan),
                row.get("radar_peak_3_bpm", np.nan),
            ],
            dtype=np.float32,
        )
        confidence = float(row.get("classical_confidence", 0.0))
        classical_std = np.asarray(
            [1.0 + 3.0 * (1.0 - np.clip(confidence, 0.0, 1.0)), 1.5, 1.5, 1.5],
            dtype=np.float32,
        )
        if structural_mask is not None:
            classical[1:][~radar_mask] = 0
            classical_std[1:][~radar_mask] = 0
            feature_columns = tuple(
                self.experiment.provenance.get("feature_allowlist", ())
            )
            if not bool(radar_mask.all()):
                # The fused classical candidate/confidence were computed from
                # all three cached views before structural timing was known.
                # They are not valid evidence when any contributing view is
                # unavailable.
                classical[0] = 0
                classical_std[0] = 0
                if "classical_confidence" in feature_columns:
                    base_feature_array[
                        feature_columns.index("classical_confidence")
                    ] = 0
            for radar_index, available in enumerate(radar_mask, start=1):
                column = f"radar_peak_{radar_index}_bpm"
                if not available and column in feature_columns:
                    base_feature_array[feature_columns.index(column)] = 0
            if not bool(radar_mask.all()) and "radar_peak_spread_bpm" in feature_columns:
                base_feature_array[feature_columns.index("radar_peak_spread_bpm")] = 0
        reference_sigma = float(row.get("reference_sigma_bpm", 1.0))
        reference_quality = float(row.get("reference_quality", 1.0))
        observable = bool(row.get("radar_observable", True))
        return {
            "spectra": torch.from_numpy(spectra_array),
            "attributes": torch.from_numpy(attributes_array),
            "base_prediction": torch.tensor(float(row["prediction_bpm"]), dtype=torch.float32),
            "base_std": torch.tensor(
                max(0.25, float(row.get("rr_std_bpm", 1.5))), dtype=torch.float32
            ),
            "base_features": torch.from_numpy(base_feature_array),
            "classical_rr": torch.from_numpy(classical),
            "classical_std": torch.from_numpy(classical_std),
            "radar_mask": torch.from_numpy(radar_mask),
            # Everything below is loss/evaluation-only and is never passed to
            # ``forward_model``.
            "rr": torch.tensor(float(row["rr_bpm"]), dtype=torch.float32),
            "reference_valid": torch.tensor(bool(row.get("reference_valid", True))),
            "reference_quality": torch.tensor(reference_quality, dtype=torch.float32),
            "reference_sigma": torch.tensor(reference_sigma, dtype=torch.float32),
            "observable": torch.tensor(observable, dtype=torch.float32),
            "position": torch.tensor(position, dtype=torch.int64),
            "cache_index": torch.tensor(int(row["cache_index"]), dtype=torch.int64),
        }


def _worker_seed(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    experiment: AlignedSVDExperiment,
    positions: Sequence[int],
    base_features: np.ndarray,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
    train: bool,
    samples_per_epoch: int | None = None,
    rr_bin_width: float = 3.0,
    rr_balance_power: float = 0.65,
    tail_min_bpm: float = 25.0,
    tail_max_bpm: float = 35.0,
    tail_boost: float = 2.0,
) -> DataLoader[dict[str, Tensor]]:
    positions = np.asarray(positions, dtype=np.int64)
    dataset = SVDSourceDataset(experiment, positions, base_features)
    generator = torch.Generator().manual_seed(int(seed))
    sampler = None
    if train:
        weights = identity_rr_tail_sample_weights(
            experiment.metadata,
            positions,
            rr_bin_width=rr_bin_width,
            rr_balance_power=rr_balance_power,
            tail_min_bpm=tail_min_bpm,
            tail_max_bpm=tail_max_bpm,
            tail_boost=tail_boost,
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(samples_per_epoch or len(positions)),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=_worker_seed if workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
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


def forward_model(
    model: nn.Module, batch: Mapping[str, Tensor], device: torch.device
) -> Mapping[str, Tensor]:
    """Forward only the deployment allow-list; target fields are unreachable."""

    def move(name: str, *, dtype: torch.dtype | None = None) -> Tensor:
        value = batch[name].to(device, non_blocking=True)
        return value.to(dtype=dtype) if dtype is not None else value

    return model(
        move("spectra", dtype=torch.float32),
        move("attributes", dtype=torch.float32),
        move("base_prediction", dtype=torch.float32),
        move("base_std", dtype=torch.float32),
        move("base_features", dtype=torch.float32),
        move("classical_rr", dtype=torch.float32),
        move("classical_std", dtype=torch.float32),
        move("radar_mask").bool(),
    )


def compute_svd_multitask_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    rr_bins: Tensor,
    *,
    distribution_weight: float = 1.0,
    source_distribution_weight: float = 1.0,
    huber_weight: float = 0.50,
    source_huber_weight: float = 0.50,
    uncertainty_weight: float = 0.10,
    quality_weight: float = 0.05,
    gate_weight: float = 0.10,
    action_regret_weight: float = 0.15,
    spike_sparsity_weight: float = 5e-4,
    oracle_margin_bpm: float = 0.05,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Soft-distribution + robust regression + safe-action training loss."""

    logits = output["logits"].float()
    source_logits = output["source_logits"].float()
    expected = output["expected_rr"].float()
    source_expected = output["source_expected_rr"].float()
    base_expected = output["base_expected_rr"].float()
    log_variance = output["log_variance"].float()
    target = batch["rr"].to(logits.device).float()
    valid = batch["reference_valid"].to(logits.device).bool() & torch.isfinite(target)
    zero = logits.sum() * 0.0
    if valid.any():
        sigma = batch["reference_sigma"].to(logits.device).float()[valid].clamp(0.3, 2.5)
        soft_target = gaussian_soft_targets(
            target[valid], rr_bins.to(logits.device).float(), sigma=sigma
        )
        reference_weight = (
            batch["reference_quality"].to(logits.device).float()[valid].clamp(0.25, 1.0)
        )
        distribution_per = -(soft_target * logits[valid].log_softmax(dim=-1)).sum(dim=-1)
        distribution = (distribution_per * reference_weight).sum() / reference_weight.sum()
        # The deployment posterior starts behind a deliberately closed safety
        # gate.  Training only that posterior suppresses the source branch's
        # gradient by the tiny initial gate probability and can leave the
        # router comparing the base against an effectively random expert.
        # Supervise the source posterior directly as well; the final gate and
        # validation-only promotion rule still decide whether it is safe to
        # use at inference time.
        source_distribution_per = -(
            soft_target * source_logits[valid].log_softmax(dim=-1)
        ).sum(dim=-1)
        source_distribution = (
            source_distribution_per * reference_weight
        ).sum() / reference_weight.sum()
        huber_per = F.smooth_l1_loss(
            expected[valid], target[valid], beta=1.0, reduction="none"
        )
        huber = (huber_per * reference_weight).sum() / reference_weight.sum()
        source_huber_per = F.smooth_l1_loss(
            source_expected[valid], target[valid], beta=1.0, reduction="none"
        )
        source_huber = (
            source_huber_per * reference_weight
        ).sum() / reference_weight.sum()
        error = expected[valid] - target[valid]
        uncertainty_per = 0.5 * (
            error.square() * torch.exp(-log_variance[valid]) + log_variance[valid]
        )
        uncertainty_nll = (uncertainty_per * reference_weight).sum() / reference_weight.sum()

        source_error = (source_expected[valid] - target[valid]).abs()
        base_error = (base_expected[valid] - target[valid]).abs()
        oracle_source = (source_error + float(oracle_margin_bpm) < base_error).float()
        gate_logits = output["mixture_gate_logits"].float()[valid]
        gate_bce_per = F.binary_cross_entropy_with_logits(
            gate_logits, oracle_source, reduction="none"
        )
        gate_bce = (gate_bce_per * reference_weight).sum() / reference_weight.sum()
        # Differentiable excess action cost relative to always taking the safe
        # base action.  This directly penalizes a harmful opened gate.
        action_regret_per = F.softplus(
            ((expected[valid] - target[valid]).abs() - base_error) / 0.25
        ) * 0.25
        action_regret = (action_regret_per * reference_weight).sum() / reference_weight.sum()
        oracle_source_fraction = oracle_source.mean()
    else:
        distribution = source_distribution = huber = source_huber = zero
        uncertainty_nll = gate_bce = action_regret = zero
        oracle_source_fraction = zero

    observable = batch["observable"].to(logits.device).float()
    quality_bce = F.binary_cross_entropy_with_logits(
        output["quality_logits"].float(), observable
    )
    spike_rate = output.get("spike_rate", zero).float().mean()
    total = (
        float(distribution_weight) * distribution
        + float(source_distribution_weight) * source_distribution
        + float(huber_weight) * huber
        + float(source_huber_weight) * source_huber
        + float(uncertainty_weight) * uncertainty_nll
        + float(quality_weight) * quality_bce
        + float(gate_weight) * gate_bce
        + float(action_regret_weight) * action_regret
        + float(spike_sparsity_weight) * spike_rate
    )
    components = {
        "loss": total.detach(),
        "distribution": distribution.detach(),
        "source_distribution": source_distribution.detach(),
        "huber": huber.detach(),
        "source_huber": source_huber.detach(),
        "uncertainty_nll": uncertainty_nll.detach(),
        "quality_bce": quality_bce.detach(),
        "gate_bce": gate_bce.detach(),
        "action_regret": action_regret.detach(),
        "spike_rate": spike_rate.detach(),
        "oracle_source_fraction": oracle_source_fraction.detach(),
        "valid_fraction": valid.float().mean().detach(),
    }
    return total, components


# Short name for callers that use the generic trainer API.
compute_multitask_loss = compute_svd_multitask_loss


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(enabled and device.type == "cuda"),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    amp_scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    amp: bool,
    gradient_clip: float,
    loss_kwargs: Mapping[str, float],
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    examples = 0
    for number, batch in enumerate(loader):
        if max_batches is not None and number >= max_batches:
            break
        count = len(batch["rr"])
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            output = forward_model(model, batch, device)
            loss, components = compute_svd_multitask_loss(
                output, batch, model.rr_bins, **dict(loss_kwargs)
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite SVD training loss")
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        amp_scaler.step(optimizer)
        amp_scaler.update()
        for name, value in components.items():
            totals[name] += float(value) * count
        examples += count
    if not examples:
        raise RuntimeError("training loader yielded no batches")
    return {name: value / examples for name, value in totals.items()}


@torch.inference_mode()
def predict_loader(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None = None,
) -> PredictionResult:
    model.eval()
    values: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for number, batch in enumerate(loader):
        if max_batches is not None and number >= max_batches:
            break
        with _autocast(device, amp):
            output = forward_model(model, batch, device)
        mapping = {
            "position": batch["position"],
            "cache_index": batch["cache_index"],
            "target": batch["rr"],
            "base_prediction": batch["base_prediction"],
            "candidate_prediction": output["expected_rr"],
            "rr_std": output["rr_std"],
            "source_prediction": output["source_expected_rr"],
            "mixture_gate": output["mixture_gate"],
            "quality": output["quality"],
            "spike_rate": output["spike_rate_per_sample"],
            "radar_weights": output["radar_weights"],
        }
        for name, value in mapping.items():
            values[name].append(value.detach().float().cpu().numpy())
    if not values:
        raise RuntimeError("prediction loader yielded no batches")
    result = PredictionResult(
        position=np.concatenate(values["position"]).astype(np.int64),
        cache_index=np.concatenate(values["cache_index"]).astype(np.int64),
        target=np.concatenate(values["target"]).astype(np.float32),
        base_prediction=np.concatenate(values["base_prediction"]).astype(np.float32),
        candidate_prediction=np.concatenate(values["candidate_prediction"]).astype(np.float32),
        rr_std=np.concatenate(values["rr_std"]).astype(np.float32),
        source_prediction=np.concatenate(values["source_prediction"]).astype(np.float32),
        mixture_gate=np.concatenate(values["mixture_gate"]).astype(np.float32),
        quality=np.concatenate(values["quality"]).astype(np.float32),
        spike_rate=np.concatenate(values["spike_rate"]).astype(np.float32),
        radar_weights=np.concatenate(values["radar_weights"]).astype(np.float32),
    )
    order = np.argsort(result.position, kind="stable")
    return PredictionResult(**{name: getattr(result, name)[order] for name in result.__dataclass_fields__})


def _metric_view(
    target: np.ndarray, prediction: np.ndarray, identities: np.ndarray
) -> dict[str, Any]:
    return {
        "overall": regression_metrics(target, prediction),
        "identity_macro": identity_macro_metrics(target, prediction, identities),
    }


def evaluation_snapshot(
    target: np.ndarray,
    prediction: np.ndarray,
    identities: Sequence[str],
    *,
    high_min_bpm: float = 25.0,
    high_max_bpm: float = 35.0,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    identity = np.asarray(identities, dtype=str)
    if not (target.shape == prediction.shape == identity.shape) or target.ndim != 1:
        raise ValueError("metric vectors must have equal one-dimensional shapes")
    result = _metric_view(target, prediction, identity)
    selected = (target >= float(high_min_bpm)) & (target <= float(high_max_bpm))
    result["high_25_35"] = (
        {**_metric_view(target[selected], prediction[selected], identity[selected]), "n": int(selected.sum())}
        if selected.any()
        else None
    )
    return result


def promotion_decision(
    target: np.ndarray,
    candidate_prediction: np.ndarray,
    base_prediction: np.ndarray,
    identities: Sequence[str],
    *,
    minimum_macro_mae_improvement: float = 0.05,
    noninferiority_tolerance: float = 0.0,
    high_min_bpm: float = 25.0,
    high_max_bpm: float = 35.0,
) -> dict[str, Any]:
    """Validation-only safe promotion rule; no test argument exists."""

    if minimum_macro_mae_improvement < 0 or noninferiority_tolerance < 0:
        raise ValueError("promotion margins cannot be negative")
    candidate = evaluation_snapshot(
        target, candidate_prediction, identities, high_min_bpm=high_min_bpm, high_max_bpm=high_max_bpm
    )
    base = evaluation_snapshot(
        target, base_prediction, identities, high_min_bpm=high_min_bpm, high_max_bpm=high_max_bpm
    )
    candidate_macro = float(candidate["identity_macro"]["macro_mae"])
    base_macro = float(base["identity_macro"]["macro_mae"])
    high_evaluable = candidate["high_25_35"] is not None and base["high_25_35"] is not None
    gates = {
        "macro_mae_improves_by_minimum": candidate_macro
        <= base_macro - float(minimum_macro_mae_improvement),
        "high_25_35_macro_mae_noninferior": bool(
            high_evaluable
            and float(candidate["high_25_35"]["identity_macro"]["macro_mae"])
            <= float(base["high_25_35"]["identity_macro"]["macro_mae"])
            + float(noninferiority_tolerance)
        ),
        "catastrophic_over_5_noninferior": float(candidate["overall"]["catastrophic_over_5"])
        <= float(base["overall"]["catastrophic_over_5"])
        + float(noninferiority_tolerance),
        "within_2_noninferior": float(candidate["overall"]["within_2"])
        + float(noninferiority_tolerance)
        >= float(base["overall"]["within_2"]),
    }
    return {
        "promoted": bool(all(gates.values())),
        "gates": gates,
        "minimum_macro_mae_improvement_bpm": float(minimum_macro_mae_improvement),
        "noninferiority_tolerance": float(noninferiority_tolerance),
        "observed_macro_mae_improvement_bpm": base_macro - candidate_macro,
        "high_25_35_evaluable": bool(high_evaluable),
        "candidate": candidate,
        "base": base,
        "selection_inputs": "validation predictions, validation references, validation identities only",
        "test_inputs_used": False,
    }


def _result_arrays(result: PredictionResult, *, promoted: bool | None = None) -> dict[str, np.ndarray]:
    arrays = {name: np.asarray(getattr(result, name)) for name in result.__dataclass_fields__}
    if promoted is not None:
        arrays["prediction_final"] = np.where(
            promoted, result.candidate_prediction, result.base_prediction
        ).astype(np.float32)
        arrays["promoted"] = np.full(len(result.position), bool(promoted), dtype=bool)
    return arrays


def prediction_report(
    result: PredictionResult,
    metadata: pd.DataFrame,
    *,
    promoted: bool | None = None,
) -> dict[str, Any]:
    identities = metadata.iloc[result.position]["identity"].astype(str).to_numpy()
    report: dict[str, Any] = {
        "n": int(len(result.position)),
        "candidate": evaluation_snapshot(result.target, result.candidate_prediction, identities),
        "base": evaluation_snapshot(result.target, result.base_prediction, identities),
    }
    if promoted is not None:
        final = result.candidate_prediction if promoted else result.base_prediction
        report["locked_final"] = evaluation_snapshot(result.target, final, identities)
        report["promoted"] = bool(promoted)
    return report


def capture_rng_state(loader: DataLoader[Any]) -> dict[str, Any]:
    loader_generator = getattr(loader, "generator", None)
    sampler_generator = getattr(getattr(loader, "sampler", None), "generator", None)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader_generator.get_state() if isinstance(loader_generator, torch.Generator) else None,
        "sampler": sampler_generator.get_state() if isinstance(sampler_generator, torch.Generator) else None,
    }


def restore_rng_state(state: Mapping[str, Any], loader: DataLoader[Any]) -> None:
    for required in ("python", "numpy", "torch_cpu", "torch_cuda"):
        if required not in state:
            raise RuntimeError(f"checkpoint RNG state lacks {required}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], device="cpu"))
    cuda_state = list(state["torch_cuda"])
    if cuda_state:
        if not torch.cuda.is_available() or len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA RNG topology differs from this runtime")
        torch.cuda.set_rng_state_all(cuda_state)
    for key, owner in (
        ("loader", getattr(loader, "generator", None)),
        ("sampler", getattr(getattr(loader, "sampler", None), "generator", None)),
    ):
        if state.get(key) is not None:
            if not isinstance(owner, torch.Generator):
                raise RuntimeError(f"checkpoint contains {key} RNG but loader does not")
            owner.set_state(torch.as_tensor(state[key], device="cpu"))


def _checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    amp_scaler: torch.amp.GradScaler,
    loader: DataLoader[Any],
    epoch: int,
    best_epoch: int,
    best_score: float,
    stale_epochs: int,
    fold: int,
    split: FoldSplit,
    run_signature: str,
    model_kwargs: Mapping[str, Any],
    feature_scaler: RobustScaler,
    source_binding_sha256: str,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "amp_scaler_state": amp_scaler.state_dict(),
        "rng_state": capture_rng_state(loader),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "stale_epochs": int(stale_epochs),
        "fold": int(fold),
        "split": {
            "train_identities": list(split.train_identities),
            "validation_identities": list(split.validation_identities),
            "test_identities": list(split.test_identities),
        },
        "run_signature": run_signature,
        "model_kwargs": dict(model_kwargs),
        "base_feature_columns": list(feature_scaler.columns),
        "feature_center": feature_scaler.center,
        "feature_scale": feature_scaler.scale,
        "source_binding_sha256": source_binding_sha256,
    }


def _validate_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    fold: int,
    split: FoldSplit,
    run_signature: str,
    source_binding_sha256: str,
) -> None:
    expected = {
        "fold": int(fold),
        "run_signature": run_signature,
        "source_binding_sha256": source_binding_sha256,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"resume checkpoint {key} does not match this run")
    declared = checkpoint.get("split", {})
    for key, identities in (
        ("train_identities", split.train_identities),
        ("validation_identities", split.validation_identities),
        ("test_identities", split.test_identities),
    ):
        if tuple(declared.get(key, ())) != tuple(identities):
            raise RuntimeError(f"resume checkpoint {key} does not match frozen split")


def _model_kwargs(args: argparse.Namespace, experiment: AlignedSVDExperiment, feature_dim: int) -> dict[str, Any]:
    first = experiment.sessions[0].spectra
    kwargs: dict[str, Any] = {
        "num_variants": int(first.shape[2]),
        "num_components": int(first.shape[3]),
        "num_radars": int(first.shape[1]),
        "base_feature_dim": int(feature_dim),
        "encoder_channels": int(args.encoder_channels),
        "encoder_blocks": int(args.encoder_blocks),
        "hidden_channels": int(args.hidden_channels),
        "num_spiking_blocks": int(args.spiking_blocks),
        "simulation_steps": int(args.simulation_steps),
        "beta": float(args.beta),
        "radar_dropout_p": float(args.radar_dropout),
        "spectral_frequency_min_hz": float(experiment.frequencies_hz[0]),
        "spectral_frequency_max_hz": float(experiment.frequencies_hz[-1]),
        "candidate_sigma": float(args.candidate_sigma),
        "initial_gate_bias": float(args.initial_gate_bias),
        "dropout": float(args.dropout),
    }
    if args.preset == "tiny":
        kwargs.update(
            encoder_channels=12,
            encoder_blocks=1,
            hidden_channels=16,
            num_spiking_blocks=1,
            simulation_steps=min(2, int(args.simulation_steps)),
        )
    elif args.preset == "compact":
        kwargs.update(
            encoder_channels=min(32, int(args.encoder_channels)),
            encoder_blocks=min(2, int(args.encoder_blocks)),
            hidden_channels=min(48, int(args.hidden_channels)),
            num_spiking_blocks=min(2, int(args.spiking_blocks)),
        )
    return kwargs


def train_fold(
    args: argparse.Namespace,
    experiment: AlignedSVDExperiment,
    fold: int,
    device: torch.device,
    run_signature: str,
) -> tuple[PredictionResult, bool, dict[str, Any]]:
    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    split = make_outer_split(experiment.metadata, fold)
    split_record = {
        "outer_test_fold": fold,
        "validation_fold": split.validation_fold,
        "weight_fit_folds": sorted(set(range(N_FOLDS)) - {fold, split.validation_fold}),
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
        "train_identities": split.train_identities,
        "validation_identities": split.validation_identities,
        "test_identities": split.test_identities,
        "identity_overlap_asserted_empty": True,
        "test_metadata_policy": (
            "test fold/identity are used only to define and audit the outer partition; "
            "test labels, protocols, sessions and metrics are unavailable to scaling, "
            "weight fitting, early stopping and promotion"
        ),
    }
    atomic_write_json(fold_dir / "split.json", split_record)

    columns = resolve_base_feature_columns(experiment.metadata)
    feature_scaler = fit_robust_scaler(experiment.metadata, split.train, columns)
    base_features = feature_scaler.transform(experiment.metadata)
    train_loader = make_loader(
        experiment,
        split.train,
        base_features,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        seed=args.seed + 1009 * fold,
        train=True,
        samples_per_epoch=args.samples_per_epoch,
        rr_bin_width=args.rr_balance_bin_width,
        rr_balance_power=args.rr_balance_power,
        tail_min_bpm=args.tail_min_bpm,
        tail_max_bpm=args.tail_max_bpm,
        tail_boost=args.tail_boost,
    )
    validation_loader = make_loader(
        experiment,
        split.validation,
        base_features,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        device=device,
        seed=args.seed + 1009 * fold + 1,
        train=False,
    )

    model_kwargs = _model_kwargs(args, experiment, len(columns))
    model = SourceSeparatedRRSNN(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, args.patience // 3), min_lr=1e-6
    )
    amp_scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    source_binding = str(experiment.provenance["row_binding_sha256"])
    best_score = math.inf
    best_epoch = -1
    stale_epochs = 0
    start_epoch = 0
    last_path = fold_dir / "svd_last.pt"
    best_path = fold_dir / "svd_best.pt"
    resume_path = Path(args.resume_from) if args.resume_from else last_path
    if args.resume and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        _validate_resume_checkpoint(
            checkpoint,
            fold=fold,
            split=split,
            run_signature=run_signature,
            source_binding_sha256=source_binding,
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        amp_scaler.load_state_dict(checkpoint["amp_scaler_state"])
        restore_rng_state(checkpoint["rng_state"], train_loader)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_score = float(checkpoint["best_score"])
        stale_epochs = int(checkpoint["stale_epochs"])
    elif args.resume and args.resume_from:
        raise FileNotFoundError(resume_path)

    loss_kwargs = {
        "distribution_weight": args.distribution_weight,
        "source_distribution_weight": args.source_distribution_weight,
        "huber_weight": args.huber_weight,
        "source_huber_weight": args.source_huber_weight,
        "uncertainty_weight": args.uncertainty_weight,
        "quality_weight": args.quality_weight,
        "gate_weight": args.gate_weight,
        "action_regret_weight": args.action_regret_weight,
        "spike_sparsity_weight": args.spike_sparsity_weight,
        "oracle_margin_bpm": args.oracle_margin_bpm,
    }
    history_path = fold_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        training = train_one_epoch(
            model,
            train_loader,
            optimizer,
            amp_scaler,
            device,
            amp=args.amp,
            gradient_clip=args.gradient_clip,
            loss_kwargs=loss_kwargs,
            max_batches=args.smoke_max_batches,
        )
        validation = predict_loader(
            model,
            validation_loader,
            device,
            amp=args.amp,
            max_batches=args.smoke_max_batches,
        )
        validation_identity = experiment.metadata.iloc[validation.position]["identity"].astype(str)
        validation_metrics = evaluation_snapshot(
            validation.target, validation.candidate_prediction, validation_identity
        )
        score = float(validation_metrics["identity_macro"]["macro_mae"])
        scheduler.step(score)
        improved = score < best_score - args.min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        state = _checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_scaler=amp_scaler,
            loader=train_loader,
            epoch=epoch,
            best_epoch=best_epoch,
            best_score=best_score,
            stale_epochs=stale_epochs,
            fold=fold,
            split=split,
            run_signature=run_signature,
            model_kwargs=model_kwargs,
            feature_scaler=feature_scaler,
            source_binding_sha256=source_binding,
        )
        atomic_torch_save(state, last_path)
        if improved:
            atomic_torch_save(state, best_path)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _json_ready(
                        {
                            "epoch": epoch,
                            "train": training,
                            "validation": validation_metrics,
                            "score": score,
                            "best_score": best_score,
                            "best_epoch": best_epoch,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    ),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
        if stale_epochs >= args.patience:
            break
    if not best_path.is_file():
        raise RuntimeError("no best checkpoint was produced")
    best = torch.load(best_path, map_location=device, weights_only=False)
    _validate_resume_checkpoint(
        best,
        fold=fold,
        split=split,
        run_signature=run_signature,
        source_binding_sha256=source_binding,
    )
    model.load_state_dict(best["model_state"])

    # Recreate validation predictions from the locked best checkpoint.
    validation = predict_loader(
        model,
        validation_loader,
        device,
        amp=args.amp,
        max_batches=args.smoke_max_batches,
    )
    validation_identities = experiment.metadata.iloc[validation.position]["identity"].astype(str).to_numpy()
    selection = promotion_decision(
        validation.target,
        validation.candidate_prediction,
        validation.base_prediction,
        validation_identities,
        minimum_macro_mae_improvement=args.promotion_min_improvement,
        noninferiority_tolerance=args.promotion_noninferiority_tolerance,
        high_min_bpm=args.tail_min_bpm,
        high_max_bpm=args.tail_max_bpm,
    )
    if args.smoke_max_batches is not None:
        selection["promoted"] = False
        selection["smoke_override"] = "promotion disabled because validation was truncated"
    promoted = bool(selection["promoted"])
    atomic_save_npz(
        fold_dir / "validation_predictions.npz",
        **_result_arrays(validation, promoted=promoted),
    )
    atomic_write_json(
        fold_dir / "validation_predictions.json",
        prediction_report(validation, experiment.metadata, promoted=promoted),
    )
    selection_lock = {
        "lock_created_utc": datetime.now(timezone.utc).isoformat(),
        "outer_fold": fold,
        "best_epoch": int(best["best_epoch"]),
        "best_validation_macro_mae": float(best["best_score"]),
        "run_signature": run_signature,
        "checkpoint_sha256": sha256_file(best_path),
        "source_binding_sha256": source_binding,
        "decision": selection,
        "locked_final_action": "candidate" if promoted else "base_only_fallback",
        "test_predictions_generated": False,
        "test_metadata_or_labels_used_for_selection": False,
    }
    atomic_write_json(fold_dir / "selection_lock.json", selection_lock)

    # Test construction and inference happen strictly after the immutable
    # validation lock above.  The model receives no target/test metadata.
    test_loader = make_loader(
        experiment,
        split.test,
        base_features,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        device=device,
        seed=args.seed + 1009 * fold + 2,
        train=False,
    )
    test = predict_loader(
        model,
        test_loader,
        device,
        amp=args.amp,
        max_batches=args.smoke_max_batches,
    )
    atomic_save_npz(fold_dir / "test_predictions.npz", **_result_arrays(test, promoted=promoted))
    test_report = prediction_report(test, experiment.metadata, promoted=promoted)
    test_report.update(
        {
            "outer_fold": fold,
            "selection_lock_sha256": sha256_file(fold_dir / "selection_lock.json"),
            "candidate_saved_even_when_rejected": True,
        }
    )
    atomic_write_json(fold_dir / "test_predictions.json", test_report)
    return test, promoted, {
        "split": split_record,
        "selection": selection,
        "test": test_report,
        "checkpoint_sha256": sha256_file(best_path),
    }


def _build_run_config(args: argparse.Namespace, experiment: AlignedSVDExperiment) -> dict[str, Any]:
    excluded = {"resume", "resume_from"}
    arguments = {
        key: value for key, value in vars(args).items() if key not in excluded
    }
    source_paths = [Path(__file__), SOURCE_ROOT / "snn_rr" / "svd_models.py"]
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": arguments,
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in source_paths},
        "data_provenance": experiment.provenance,
        "split_protocol": {
            "outer_test": "f",
            "validation": "(f+1)%6",
            "weight_fit": "remaining four frozen folds",
            "selection": "validation only; test prediction occurs after lock",
        },
        "component_signals_input": False,
    }
    signature_payload = {
        "arguments": arguments,
        "source_sha256": config["source_sha256"],
        "data_provenance": experiment.provenance,
        "split_protocol": config["split_protocol"],
    }
    config["run_signature"] = stable_signature(signature_payload)
    return config


def _write_oof(
    args: argparse.Namespace,
    experiment: AlignedSVDExperiment,
    fold_results: Mapping[int, tuple[PredictionResult, bool, dict[str, Any]]],
    run_signature: str,
) -> dict[str, Any]:
    count = len(experiment.metadata)
    candidate = np.full(count, np.nan, dtype=np.float32)
    final = np.full(count, np.nan, dtype=np.float32)
    base = pd.to_numeric(experiment.metadata["prediction_bpm"], errors="raise").to_numpy(np.float32)
    rr_std = np.full(count, np.nan, dtype=np.float32)
    source = np.full(count, np.nan, dtype=np.float32)
    gate = np.full(count, np.nan, dtype=np.float32)
    quality = np.full(count, np.nan, dtype=np.float32)
    spike = np.full(count, np.nan, dtype=np.float32)
    radar_weights = np.full((count, 3), np.nan, dtype=np.float32)
    promoted = np.zeros(count, dtype=bool)
    evaluated = np.zeros(count, dtype=bool)
    for _, (result, did_promote, _) in fold_results.items():
        position = result.position
        if evaluated[position].any():
            raise RuntimeError("OOF row predicted by multiple folds")
        candidate[position] = result.candidate_prediction
        final[position] = result.candidate_prediction if did_promote else result.base_prediction
        rr_std[position] = result.rr_std
        source[position] = result.source_prediction
        gate[position] = result.mixture_gate
        quality[position] = result.quality
        spike[position] = result.spike_rate
        radar_weights[position] = result.radar_weights
        promoted[position] = did_promote
        evaluated[position] = True
    positions = np.flatnonzero(evaluated)
    if not len(positions):
        raise RuntimeError("no OOF folds were evaluated")
    metadata = experiment.metadata
    target = pd.to_numeric(metadata["rr_bpm"], errors="raise").to_numpy(np.float32)
    fold = metadata["fold"].to_numpy(np.int16)
    identity = metadata["identity"].astype(str).to_numpy()
    complete = bool(evaluated.all())
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_signature": run_signature,
        "complete_six_fold_oof": complete,
        "evaluated_rows": int(evaluated.sum()),
        "expected_rows": int(count),
        "evaluated_folds": sorted(fold_results),
        "promoted_folds": sorted(fold_id for fold_id, value in fold_results.items() if value[1]),
        "locked_final": grouped_oof_metrics(
            target[positions],
            final[positions],
            identity[positions],
            fold_ids=fold[positions],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed,
        ),
        "candidate": grouped_oof_metrics(
            target[positions],
            candidate[positions],
            identity[positions],
            fold_ids=fold[positions],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed,
        ),
        "base": grouped_oof_metrics(
            target[positions],
            base[positions],
            identity[positions],
            fold_ids=fold[positions],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed,
        ),
        "fold_reports": {str(key): value[2] for key, value in fold_results.items()},
        "commercial_safety_policy": "rejected folds are exact base fallback",
    }
    output_dir = Path(args.output_dir)
    atomic_save_npz(
        output_dir / "svd_oof.npz",
        index=metadata["cache_index"].to_numpy(np.int64),
        target=target,
        fold=fold,
        evaluated=evaluated,
        prediction_base=base,
        prediction_candidate=candidate,
        prediction_final=final,
        rr_std_candidate=rr_std,
        source_prediction=source,
        mixture_gate=gate,
        quality=quality,
        spike_rate=spike,
        radar_weights=radar_weights,
        promoted=promoted,
        run_signature=np.asarray(run_signature),
    )
    table = metadata.loc[:, list(BINDING_COLUMNS) + ["fold"]].copy()
    table["prediction_base_bpm"] = base
    table["prediction_candidate_bpm"] = candidate
    table["prediction_locked_final_bpm"] = final
    table["candidate_rr_std_bpm"] = rr_std
    table["source_prediction_bpm"] = source
    table["mixture_gate"] = gate
    table["quality"] = quality
    table["spike_rate"] = spike
    table["promoted"] = promoted
    table["evaluated"] = evaluated
    temporary = output_dir / "svd_oof.csv.tmp"
    table.to_csv(temporary, index=False)
    temporary.replace(output_dir / "svd_oof.csv")
    atomic_write_json(output_dir / "metrics.json", metrics)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svd-cache", type=Path, default=PROJECT_ROOT / "artifacts/cache/svd_components_v1"
    )
    parser.add_argument(
        "--base-oof-csv",
        type=Path,
        default=PROJECT_ROOT / "artifacts/runs/ensemble_structured_exact/ensemble_oof.csv",
    )
    parser.add_argument(
        "--base-oof-npz",
        type=Path,
        default=PROJECT_ROOT / "artifacts/runs/ensemble_structured_exact/ensemble_oof.npz",
    )
    parser.add_argument(
        "--base-oof-provenance",
        type=Path,
        help=(
            "authority JSON for scientific acquisition-v2 base predictions; "
            "defaults only to provenance.json beside --base-oof-csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/runs/svd_source_snn"
    )
    parser.add_argument("--fold", default="all")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--verify-file-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "verify legacy-cache payload hashes when available; acquisition-v2 "
            "authority hashes are always mandatory"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--preset", choices=("tiny", "compact", "full"), default="compact")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--smoke-max-batches", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--encoder-channels", type=int, default=48)
    parser.add_argument("--encoder-blocks", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--spiking-blocks", type=int, default=2)
    parser.add_argument("--simulation-steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--radar-dropout", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--candidate-sigma", type=float, default=1.0)
    parser.add_argument("--initial-gate-bias", type=float, default=-6.0)
    parser.add_argument("--distribution-weight", type=float, default=1.0)
    parser.add_argument("--source-distribution-weight", type=float, default=1.0)
    parser.add_argument("--huber-weight", type=float, default=0.50)
    parser.add_argument("--source-huber-weight", type=float, default=0.50)
    parser.add_argument("--uncertainty-weight", type=float, default=0.10)
    parser.add_argument("--quality-weight", type=float, default=0.05)
    parser.add_argument("--gate-weight", type=float, default=0.10)
    parser.add_argument("--action-regret-weight", type=float, default=0.15)
    parser.add_argument("--spike-sparsity-weight", type=float, default=5e-4)
    parser.add_argument("--oracle-margin-bpm", type=float, default=0.05)
    parser.add_argument("--rr-balance-bin-width", type=float, default=3.0)
    parser.add_argument("--rr-balance-power", type=float, default=0.65)
    parser.add_argument("--tail-min-bpm", type=float, default=25.0)
    parser.add_argument("--tail-max-bpm", type=float, default=35.0)
    parser.add_argument("--tail-boost", type=float, default=2.0)
    parser.add_argument("--promotion-min-improvement", type=float, default=0.05)
    parser.add_argument("--promotion-noninferiority-tolerance", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("epochs, patience and batch sizes must be positive")
    if args.smoke_max_batches is not None and args.smoke_max_batches < 1:
        parser.error("--smoke-max-batches must be positive")
    if args.resume_from is not None and len(parse_fold_selection(args.fold)) != 1:
        parser.error("--resume-from requires a single --fold")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed, deterministic=args.deterministic)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    experiment = load_aligned_experiment(
        args.svd_cache,
        args.base_oof_csv,
        args.base_oof_npz,
        base_oof_provenance=args.base_oof_provenance,
        verify_file_hashes=args.verify_file_hashes,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_config = _build_run_config(args, experiment)
    atomic_write_json(Path(args.output_dir) / "run_config.json", run_config)
    folds = parse_fold_selection(args.fold)
    fold_results: dict[int, tuple[PredictionResult, bool, dict[str, Any]]] = {}
    for fold in folds:
        seed_everything(args.seed + 1009 * fold, deterministic=args.deterministic)
        fold_results[fold] = train_fold(
            args, experiment, fold, device, str(run_config["run_signature"])
        )
    metrics = _write_oof(
        args, experiment, fold_results, str(run_config["run_signature"])
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "run_signature": run_config["run_signature"],
                "evaluated_folds": folds,
                "promoted_folds": metrics["promoted_folds"],
                "locked_final_mae": metrics["locked_final"]["overall"]["mae"],
                "locked_final_macro_mae": metrics["locked_final"]["identity_macro"]["macro_mae"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
