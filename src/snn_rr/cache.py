"""Feature-cache loading for grouped training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_HISTORY_LAGS = (1, 2, 4, 8)
DEFAULT_HISTORY_ROLLING_WINDOWS = (4, 8)
ACQUISITION_CACHE_SCHEMA_VERSION = "snn_rr.feature_cache_acquisition.v1"

# These fields are deliberately metadata-only.  In particular, the acquisition
# phase and phase-7 assignment may be informed by offline BIOPAC annotations
# and must never be added to an inference feature allowlist.
REQUIRED_ACQUISITION_ANNOTATION_COLUMNS = frozenset(
    {
        "reference_start_sample",
        "reference_end_sample",
        "reference_window_start_biopac_s",
        "reference_window_end_biopac_s",
        "radar_window_start_relative_s",
        "radar_window_end_relative_s",
        "sync_authorized",
        "sync_confidence",
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
    }
)

_ACQUISITION_INDICATOR_KEYS = frozenset(
    {
        "acquisition_contract",
        "acquisition_contract_sha256",
        "acquisition_manifest_sha256",
        "acquisition_reconstruction_manifest_sha256",
        "acquisition_session_manifest_sha256",
        "scientific_eligible",
    }
)


@dataclass(slots=True)
class FeatureCache:
    maps: np.ndarray
    aux: np.ndarray
    metadata: pd.DataFrame
    frequencies_hz: np.ndarray

    def subset(self, indices: np.ndarray) -> "FeatureCache":
        indices = np.asarray(indices)
        return FeatureCache(
            maps=self.maps[indices],
            aux=self.aux[indices],
            metadata=self.metadata.iloc[indices].reset_index(drop=True),
            frequencies_hz=self.frequencies_hz,
        )


def load_feature_cache(
    cache_dir: str | Path,
    *,
    sessions: list[str] | None = None,
    mmap: bool = False,
    require_acquisition_contract: bool = False,
    require_scientific_eligible: bool = False,
) -> FeatureCache:
    """Load and concatenate per-session feature files.

    ``mmap=True`` is only useful when loading one session; NumPy must allocate
    when multiple memory maps are concatenated.

    Legacy caches retain their historical behavior unless either strict flag
    is enabled.  ``require_acquisition_contract`` fail-closes on a partially
    annotated cache: the root reconstruction contract and every usable
    session's content-addressed binding must be mutually consistent, and every
    metadata file must carry the complete offline-annotation schema.
    ``require_scientific_eligible`` implies the acquisition contract check and
    additionally requires the cache root and every usable session to be
    explicitly eligible.  This prevents selecting only the convenient rows or
    sessions from a mixed/unauthorized cache.
    """

    root = Path(cache_dir)
    root_manifest_path = root / "manifest.json"
    if not root_manifest_path.is_file():
        raise FileNotFoundError(f"feature cache manifest missing: {root_manifest_path}")
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    session_items = root_manifest.get("sessions")
    if not isinstance(session_items, list):
        raise ValueError("feature cache root manifest must contain a sessions list")
    if any(not isinstance(item, dict) for item in session_items):
        raise ValueError("feature cache root sessions must contain manifest objects")
    available_items = [item for item in session_items if item.get("status") == "ok"]
    available = [item["session_id"] for item in available_items]
    if len(set(available)) != len(available):
        raise ValueError("feature cache root manifest contains duplicate usable session IDs")
    selected = available if sessions is None else sessions
    if len(set(selected)) != len(selected):
        raise ValueError("requested sessions must not contain duplicates")
    missing = sorted(set(selected) - set(available))
    if missing:
        raise KeyError(f"sessions not present in cache: {missing}")

    strict_acquisition = bool(
        require_acquisition_contract or require_scientific_eligible
    )
    if strict_acquisition:
        _validate_acquisition_cache_contract(
            root,
            root_manifest,
            available_items,
            require_scientific_eligible=require_scientific_eligible,
        )

    map_arrays: list[np.ndarray] = []
    aux_arrays: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    frequency_grid: np.ndarray | None = None
    for session_id in selected:
        session_dir = root / session_id
        mode = "r" if mmap else None
        map_array = np.load(session_dir / "maps.npy", mmap_mode=mode)
        aux_array = np.load(session_dir / "aux.npy", mmap_mode=mode)
        frame = pd.read_csv(session_dir / "metadata.csv")
        frequencies = np.load(session_dir / "frequencies_hz.npy")
        if not (len(map_array) == len(aux_array) == len(frame)):
            raise ValueError(f"cache length mismatch in {session_id}")
        if frequency_grid is None:
            frequency_grid = frequencies
        elif not np.allclose(frequency_grid, frequencies):
            raise ValueError(f"frequency grid mismatch in {session_id}")
        map_arrays.append(map_array)
        aux_arrays.append(aux_array)
        frames.append(frame)

    if not map_arrays:
        raise ValueError("feature cache contains no selected sessions")
    maps = map_arrays[0] if len(map_arrays) == 1 else np.concatenate(map_arrays, axis=0)
    aux = aux_arrays[0] if len(aux_arrays) == 1 else np.concatenate(aux_arrays, axis=0)
    metadata = pd.concat(frames, ignore_index=True)
    return FeatureCache(maps, aux, metadata, np.asarray(frequency_grid))


def _canonical_json(value: Any) -> str:
    """Return a deterministic representation for exact contract comparison."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("acquisition contract must be canonical JSON data") from error


def _canonical_content_sha256(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_sha256(value: Any, *, location: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return digest


def _annotation_columns(value: Any, *, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list")
    columns = tuple(str(column) for column in value)
    if len(set(columns)) != len(columns):
        raise ValueError(f"{location} must not contain duplicates")
    if set(columns) != set(REQUIRED_ACQUISITION_ANNOTATION_COLUMNS):
        missing = sorted(REQUIRED_ACQUISITION_ANNOTATION_COLUMNS - set(columns))
        extra = sorted(set(columns) - REQUIRED_ACQUISITION_ANNOTATION_COLUMNS)
        raise ValueError(
            f"{location} does not match the acquisition metadata schema; "
            f"missing={missing}, extra={extra}"
        )
    return columns


def _has_acquisition_indicator(document: dict[str, Any]) -> bool:
    return bool(_ACQUISITION_INDICATOR_KEYS.intersection(document))


def _explicit_boolean(value: Any, *, location: str) -> bool:
    """Read a JSON/CSV boolean without accepting truthy strings or numbers."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{location} must be an explicit boolean")


def _validate_acquisition_metadata(
    frame: pd.DataFrame,
    *,
    session_id: str,
    session_scientific_eligible: bool,
    synchronization_authorized: bool,
) -> None:
    missing = sorted(REQUIRED_ACQUISITION_ANNOTATION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"acquisition metadata columns missing in {session_id}: {missing}"
        )
    if "session_id" not in frame:
        raise ValueError(f"acquisition metadata in {session_id} is missing session_id")
    observed_sessions = set(frame["session_id"].dropna().astype(str))
    if observed_sessions != {session_id}:
        raise ValueError(
            f"acquisition metadata/session manifest mismatch in {session_id}: "
            f"observed {sorted(observed_sessions)}"
        )

    if frame.empty:
        raise ValueError(f"acquisition metadata contains no rows in {session_id}")

    for column in (
        "sync_authorized",
        "alignment_scientific_eligible",
        "transition_window",
        "eligible_for_stage_metrics",
    ):
        if frame[column].isna().any():
            raise ValueError(f"{session_id} metadata column {column} contains nulls")
        values = frame[column].unique()
        for value in values:
            _explicit_boolean(value, location=f"{session_id} metadata column {column}")

    alignment_values = {
        _explicit_boolean(
            value,
            location=f"{session_id} metadata column alignment_scientific_eligible",
        )
        for value in frame["alignment_scientific_eligible"].unique()
    }
    if alignment_values != {session_scientific_eligible}:
        raise ValueError(f"alignment scientific eligibility mismatch in {session_id}")
    sync_values = {
        _explicit_boolean(
            value, location=f"{session_id} metadata column sync_authorized"
        )
        for value in frame["sync_authorized"].unique()
    }
    if sync_values != {synchronization_authorized}:
        raise ValueError(f"synchronization authorization mismatch in {session_id}")
    if not session_scientific_eligible and any(
        _explicit_boolean(
            value,
            location=f"{session_id} metadata column eligible_for_stage_metrics",
        )
        for value in frame["eligible_for_stage_metrics"].unique()
    ):
        raise ValueError(
            f"ineligible session exposes rows as eligible for stage metrics: {session_id}"
        )

    start = pd.to_numeric(frame["reference_start_sample"], errors="coerce").to_numpy(
        dtype=float
    )
    end = pd.to_numeric(frame["reference_end_sample"], errors="coerce").to_numpy(
        dtype=float
    )
    if not (
        np.isfinite(start).all()
        and np.isfinite(end).all()
        and np.equal(start, np.rint(start)).all()
        and np.equal(end, np.rint(end)).all()
        and (start >= 0).all()
        and (end > start).all()
    ):
        raise ValueError(
            f"acquisition reference sample bounds are invalid in {session_id}"
        )

    overlap = pd.to_numeric(frame["phase_overlap_fraction"], errors="coerce").to_numpy(
        dtype=float
    )
    if not (np.isfinite(overlap).all() and (overlap >= 0).all() and (overlap <= 1).all()):
        raise ValueError(f"phase_overlap_fraction is invalid in {session_id}")

    confidence = pd.to_numeric(
        frame["acquisition_phase_confidence"], errors="coerce"
    ).to_numpy(dtype=float)
    sync_confidence = pd.to_numeric(frame["sync_confidence"], errors="coerce").to_numpy(
        dtype=float
    )
    if not (
        np.isfinite(sync_confidence).all()
        and (sync_confidence >= 0).all()
        and (sync_confidence <= 1).all()
    ):
        raise ValueError(f"acquisition confidence is invalid in {session_id}")

    phase = frame["acquisition_phase"]
    phase_missing = phase.isna() | phase.astype(str).str.strip().eq("")
    for column in ("acquisition_phase_name", "acquisition_phase_status"):
        values = frame[column]
        missing_values = values.isna() | values.astype(str).str.strip().eq("")
        if not np.array_equal(missing_values.to_numpy(), phase_missing.to_numpy()):
            raise ValueError(f"partial acquisition phase annotation in {session_id}")
    confidence_missing = ~np.isfinite(confidence)
    if not np.array_equal(confidence_missing, phase_missing.to_numpy()):
        raise ValueError(f"partial acquisition phase confidence in {session_id}")
    assigned_confidence = confidence[~confidence_missing]
    if not (
        (assigned_confidence >= 0).all() and (assigned_confidence <= 1).all()
    ):
        raise ValueError(f"acquisition phase confidence is invalid in {session_id}")
    if frame.loc[phase_missing, "eligible_for_stage_metrics"].astype(bool).any():
        raise ValueError(f"unassigned acquisition phase is metric-eligible in {session_id}")

    reference_start_s = pd.to_numeric(
        frame["reference_window_start_biopac_s"], errors="coerce"
    ).to_numpy(dtype=float)
    reference_end_s = pd.to_numeric(
        frame["reference_window_end_biopac_s"], errors="coerce"
    ).to_numpy(dtype=float)
    radar_start_s = pd.to_numeric(
        frame["radar_window_start_relative_s"], errors="coerce"
    ).to_numpy(dtype=float)
    radar_end_s = pd.to_numeric(
        frame["radar_window_end_relative_s"], errors="coerce"
    ).to_numpy(dtype=float)
    if not (
        np.isfinite(reference_start_s).all()
        and np.isfinite(reference_end_s).all()
        and np.isfinite(radar_start_s).all()
        and np.isfinite(radar_end_s).all()
        and (reference_end_s > reference_start_s).all()
        and (radar_end_s > radar_start_s).all()
    ):
        raise ValueError(f"acquisition window timing is invalid in {session_id}")

    batch = frame["acquisition_batch"]
    if batch.isna().any() or not batch.astype(str).str.strip().ne("").all():
        raise ValueError(f"acquisition metadata column acquisition_batch is blank in {session_id}")


def _validate_acquisition_cache_contract(
    root: Path,
    root_manifest: dict[str, Any],
    available_items: list[dict[str, Any]],
    *,
    require_scientific_eligible: bool,
) -> None:
    """Fail closed on mixed, inconsistent, or unauthorized acquisition caches."""

    root_contract = root_manifest.get("acquisition_contract")
    if not isinstance(root_contract, dict) or not root_contract:
        raise ValueError("feature cache root acquisition_contract is missing or empty")
    if root_contract.get("schema_version") != ACQUISITION_CACHE_SCHEMA_VERSION:
        raise ValueError("feature cache root acquisition schema_version is unsupported")
    mode = root_contract.get("mode")
    if mode not in {"strict", "diagnostic"}:
        raise ValueError("feature cache acquisition mode must be strict or diagnostic")
    root_annotation_columns = _annotation_columns(
        root_contract.get("annotation_only_columns"),
        location="feature cache root annotation_only_columns",
    )
    root_eligible = _explicit_boolean(
        root_contract.get("scientific_eligible"),
        location="feature cache root acquisition scientific_eligible",
    )
    if root_eligible and mode != "strict":
        raise ValueError("a scientifically eligible acquisition cache must use strict mode")
    if require_scientific_eligible and not root_eligible:
        raise ValueError("feature cache root is not scientifically eligible")

    reconstruction_hash = _validate_sha256(
        root_contract.get("reconstruction_content_sha256"),
        location="feature cache reconstruction_content_sha256",
    )
    reconstruction_value = root_contract.get("reconstruction_manifest")
    if not isinstance(reconstruction_value, str) or not reconstruction_value.strip():
        raise ValueError("feature cache reconstruction_manifest is missing")
    reconstruction_path = Path(reconstruction_value)
    if not reconstruction_path.is_absolute():
        reconstruction_path = root / reconstruction_path
    if not reconstruction_path.is_file():
        raise FileNotFoundError(
            f"bound acquisition reconstruction manifest missing: {reconstruction_path}"
        )
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    if reconstruction.get("content_sha256") != reconstruction_hash:
        raise ValueError("acquisition reconstruction/root contract hash mismatch")
    if _canonical_content_sha256(reconstruction) != reconstruction_hash:
        raise ValueError("acquisition reconstruction canonical content hash mismatch")
    if reconstruction.get("complete") is not True:
        raise ValueError("acquisition reconstruction is not complete")
    _explicit_boolean(
        reconstruction.get("scientific_eligible"),
        location="acquisition reconstruction scientific_eligible",
    )
    reconstruction_sessions_raw = reconstruction.get("sessions")
    if not isinstance(reconstruction_sessions_raw, list):
        raise ValueError("acquisition reconstruction sessions list is missing")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("session_id"), str)
        or not item.get("session_id")
        for item in reconstruction_sessions_raw
    ):
        raise ValueError("acquisition reconstruction contains an invalid session entry")
    reconstruction_sessions = {
        str(item.get("session_id")): item for item in reconstruction_sessions_raw
    }
    if len(reconstruction_sessions) != len(reconstruction_sessions_raw):
        raise ValueError("acquisition reconstruction contains duplicate session IDs")

    if not available_items:
        raise ValueError("feature cache contains no usable acquisition sessions")

    # Inspect the complete usable catalogue, not only a requested subset.  A
    # session selection must never hide a legacy or unauthorized member of the
    # same cache root.
    for item in available_items:
        session_id = str(item.get("session_id", ""))
        if not session_id:
            raise ValueError("usable feature cache session is missing session_id")
        session_manifest_path = root / session_id / "manifest.json"
        if not session_manifest_path.is_file():
            raise FileNotFoundError(
                f"feature cache session manifest missing: {session_manifest_path}"
            )
        session_manifest = json.loads(session_manifest_path.read_text(encoding="utf-8"))
        session_contract = session_manifest.get("acquisition_contract")
        if not isinstance(session_contract, dict) or not session_contract:
            if _has_acquisition_indicator(session_manifest):
                detail = "partial acquisition session manifest"
            else:
                detail = "legacy session mixed into acquisition cache"
            raise ValueError(f"{detail}: {session_id}")
        if session_contract.get("schema_version") != ACQUISITION_CACHE_SCHEMA_VERSION:
            raise ValueError(f"session acquisition schema_version mismatch: {session_id}")

        # The catalogue and on-disk session manifest must be byte-semantically
        # identical.  Unlike the root contract, these are per-session bindings.
        item_contract = item.get("acquisition_contract")
        if not isinstance(item_contract, dict) or not item_contract:
            raise ValueError(f"legacy root catalogue item mixed into acquisition cache: {session_id}")
        if _canonical_json(item_contract) != _canonical_json(session_contract):
            raise ValueError(f"root catalogue/session acquisition contract mismatch: {session_id}")

        session_annotation_columns = _annotation_columns(
            session_contract.get("annotation_only_columns"),
            location=f"session {session_id} annotation_only_columns",
        )
        if session_annotation_columns != root_annotation_columns:
            raise ValueError(f"root/session annotation-only column order mismatch: {session_id}")

        acquisition_session_hash = _validate_sha256(
            session_contract.get("acquisition_session_manifest_sha256"),
            location=f"session {session_id} acquisition_session_manifest_sha256",
        )
        sync_receipt_hash = _validate_sha256(
            session_contract.get("sync_receipt_content_sha256"),
            location=f"session {session_id} sync_receipt_content_sha256",
        )
        mapping_hash = _validate_sha256(
            session_contract.get("mapping_sha256"),
            location=f"session {session_id} mapping_sha256",
        )
        approval_hash = _validate_sha256(
            session_contract.get("manual_approval_content_sha256"),
            location=f"session {session_id} manual_approval_content_sha256",
            nullable=True,
        )
        range_hash = _validate_sha256(
            session_contract.get("range_artifact_sha256"),
            location=f"session {session_id} range_artifact_sha256",
            nullable=True,
        )
        if session_contract.get("protocol_annotation_schema_version") != "acquisition_protocol_v1":
            raise ValueError(f"protocol annotation schema mismatch: {session_id}")
        alignment_mode = session_contract.get("reference_alignment_mode")
        if alignment_mode not in {
            "authorized_marker_affine_v1",
            "diagnostic_unapproved_proposal_v1",
        }:
            raise ValueError(f"reference alignment mode is invalid: {session_id}")
        session_eligible = _explicit_boolean(
            session_contract.get("scientific_eligible"),
            location=f"session {session_id} scientific_eligible",
        )
        if session_eligible and alignment_mode != "authorized_marker_affine_v1":
            raise ValueError(f"eligible session lacks authorized marker alignment: {session_id}")
        if require_scientific_eligible and not session_eligible:
            raise ValueError(f"acquisition session is not scientifically eligible: {session_id}")

        source_item = reconstruction_sessions.get(session_id)
        if source_item is None:
            raise ValueError(f"session missing from acquisition reconstruction: {session_id}")
        if source_item.get("content_sha256") != acquisition_session_hash:
            raise ValueError(f"acquisition session/root reconstruction hash mismatch: {session_id}")
        if _explicit_boolean(
            source_item.get("scientific_eligible"),
            location=f"acquisition reconstruction {session_id} scientific_eligible",
        ) != session_eligible:
            raise ValueError(f"acquisition session scientific eligibility mismatch: {session_id}")
        source_manifest_value = source_item.get("manifest")
        if not isinstance(source_manifest_value, str) or not source_manifest_value:
            raise ValueError(f"acquisition session manifest path is missing: {session_id}")
        source_manifest_path = Path(source_manifest_value)
        if not source_manifest_path.is_absolute():
            source_manifest_path = reconstruction_path.parent / source_manifest_path
        source_manifest_path = source_manifest_path.resolve()
        try:
            source_manifest_path.relative_to(reconstruction_path.parent.resolve())
        except ValueError as error:
            raise ValueError(
                f"acquisition session manifest escapes reconstruction root: {session_id}"
            ) from error
        if not source_manifest_path.is_file():
            raise FileNotFoundError(
                f"bound acquisition session manifest missing: {source_manifest_path}"
            )
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest.get("content_sha256") != acquisition_session_hash or (
            _canonical_content_sha256(source_manifest) != acquisition_session_hash
        ):
            raise ValueError(f"acquisition session canonical content hash mismatch: {session_id}")
        synchronization = source_manifest.get("synchronization")
        if not isinstance(synchronization, dict):
            raise ValueError(f"acquisition synchronization record is missing: {session_id}")
        if synchronization.get("receipt_content_sha256") != sync_receipt_hash:
            raise ValueError(f"synchronization receipt hash mismatch: {session_id}")
        if synchronization.get("manual_approval_content_sha256") != approval_hash:
            raise ValueError(f"manual synchronization approval hash mismatch: {session_id}")
        source_mapping = synchronization.get("mapping")
        if not isinstance(source_mapping, dict) or _canonical_sha256(source_mapping) != mapping_hash:
            raise ValueError(f"synchronization mapping hash mismatch: {session_id}")
        source_range = source_manifest.get("range_tracking")
        if not isinstance(source_range, dict):
            raise ValueError(f"acquisition range-tracking record is missing: {session_id}")
        if source_range.get("artifact_sha256") != range_hash:
            raise ValueError(f"range artifact hash mismatch: {session_id}")
        source_protocol_contract = source_manifest.get("protocol_contract")
        if not isinstance(source_protocol_contract, dict) or source_protocol_contract.get(
            "schema_version"
        ) != session_contract.get("protocol_annotation_schema_version"):
            raise ValueError(f"protocol annotation binding mismatch: {session_id}")

        metadata_path = root / session_id / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"feature cache metadata missing: {metadata_path}")
        _validate_acquisition_metadata(
            pd.read_csv(metadata_path),
            session_id=session_id,
            session_scientific_eligible=session_eligible,
            synchronization_authorized=_explicit_boolean(
                synchronization.get("authorized"),
                location=f"acquisition synchronization {session_id} authorized",
            ),
        )


def fit_aux_scaler(aux: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a train-only robust scaler (median/IQR with finite guards)."""

    train = np.asarray(aux[np.asarray(indices)], dtype=np.float64)
    center = np.nanmedian(train, axis=0)
    q25, q75 = np.nanquantile(train, [0.25, 0.75], axis=0)
    scale = q75 - q25
    standard_deviation = np.nanstd(train, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(standard_deviation > 1e-6, standard_deviation, 1.0))
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
    scale = np.where(np.isfinite(scale), scale, 1.0).astype(np.float32)
    return center, scale


def transform_aux(aux: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    transformed = (np.asarray(aux, dtype=np.float32) - center) / scale
    return np.clip(np.nan_to_num(transformed), -8.0, 8.0).astype(np.float32)


def causal_history_feature_names(
    *,
    lags: tuple[int, ...] = DEFAULT_HISTORY_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_HISTORY_ROLLING_WINDOWS,
) -> list[str]:
    """Return the stable column order used by :func:`causal_history_features`."""

    lags = _positive_unique_ints(lags, "lags")
    rolling_windows = _positive_unique_ints(rolling_windows, "rolling_windows")
    names: list[str] = []
    for lag in lags:
        names.extend(
            [
                f"history_lag_{lag}_classical_rr_bpm",
                f"history_lag_{lag}_classical_confidence",
                f"history_lag_{lag}_radar_peak_spread_bpm",
                f"history_lag_{lag}_available",
            ]
        )
    for window in rolling_windows:
        names.extend(
            [
                f"history_roll_{window}_rr_median_bpm",
                f"history_roll_{window}_rr_mad_bpm",
                f"history_roll_{window}_rr_conf_weighted_mean_bpm",
                f"history_roll_{window}_rr_trend_bpm_per_window",
                f"history_roll_{window}_confidence_mean",
                f"history_roll_{window}_spread_median_bpm",
                f"history_roll_{window}_available_fraction",
                f"history_roll_{window}_sufficient",
            ]
        )
    return names


def causal_history_features(
    metadata: pd.DataFrame,
    *,
    lags: tuple[int, ...] = DEFAULT_HISTORY_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_HISTORY_ROLLING_WINDOWS,
) -> tuple[np.ndarray, list[str]]:
    """Build label-free, strictly causal per-session history features.

    The function only reads the current cache's radar-derived
    ``classical_rr_bpm``, ``classical_confidence`` and
    ``radar_peak_spread_bpm`` columns.  Reference RR, validity and
    observability columns are deliberately never consulted.  An exact lag is
    available only when the same ``session_id`` contains
    ``window_number - lag``; rows from another session are never bridged.

    Rolling summaries likewise use only exact preceding window numbers and
    exclude the current row.  Consequently the result can be computed once on
    the full cache before a train/validation/test split: it contains the same
    information that would have been available online at each window and does
    not depend on split membership.  Missing history values are filled with
    zero and accompanied by explicit availability features.

    The returned rows preserve the input order, even when ``metadata`` is not
    sorted chronologically.
    """

    lags = _positive_unique_ints(lags, "lags")
    rolling_windows = _positive_unique_ints(rolling_windows, "rolling_windows")
    names = causal_history_feature_names(lags=lags, rolling_windows=rolling_windows)
    required = {
        "session_id",
        "window_number",
        "classical_rr_bpm",
        "classical_confidence",
        "radar_peak_spread_bpm",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise KeyError(f"metadata missing causal-history columns: {missing}")
    if metadata.empty:
        return np.empty((0, len(names)), dtype=np.float32), names

    session_values = metadata["session_id"].to_numpy()
    if pd.isna(session_values).any():
        raise ValueError("session_id contains missing values")
    raw_windows = pd.to_numeric(metadata["window_number"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(raw_windows).all() or not np.all(raw_windows == np.rint(raw_windows)):
        raise ValueError("window_number must contain finite integers")
    window_numbers = raw_windows.astype(np.int64)

    rr = pd.to_numeric(metadata["classical_rr_bpm"], errors="coerce").to_numpy(dtype=float)
    confidence = pd.to_numeric(metadata["classical_confidence"], errors="coerce").to_numpy(dtype=float)
    spread = pd.to_numeric(metadata["radar_peak_spread_bpm"], errors="coerce").to_numpy(dtype=float)
    values = np.zeros((len(metadata), len(names)), dtype=np.float32)

    # ``sort=False`` preserves first-seen session order, while all assignment
    # is by original row position so input row order is retained.
    sessions = pd.Series(np.arange(len(metadata))).groupby(session_values, sort=False).groups
    for session_id, row_index in sessions.items():
        rows = np.asarray(list(row_index), dtype=np.int64)
        numbers = window_numbers[rows]
        if len(np.unique(numbers)) != len(numbers):
            raise ValueError(f"duplicate window_number within session {session_id!r}")
        row_for_window = {int(number): int(row) for number, row in zip(numbers, rows, strict=True)}

        for row in rows:
            current_window = int(window_numbers[row])
            column = 0
            for lag in lags:
                previous_row = row_for_window.get(current_window - lag)
                if previous_row is not None:
                    history = np.asarray(
                        [rr[previous_row], confidence[previous_row], spread[previous_row]], dtype=float
                    )
                    finite = np.isfinite(history)
                    values[row, column : column + 3] = np.where(finite, history, 0.0)
                    values[row, column + 3] = float(finite.all())
                column += 4

            for window in rolling_windows:
                previous_rows = [
                    row_for_window[number]
                    for number in range(current_window - window, current_window)
                    if number in row_for_window
                ]
                if previous_rows:
                    previous_rows_array = np.asarray(previous_rows, dtype=np.int64)
                    rr_values = rr[previous_rows_array]
                    confidence_values = confidence[previous_rows_array]
                    spread_values = spread[previous_rows_array]
                    complete = (
                        np.isfinite(rr_values)
                        & np.isfinite(confidence_values)
                        & np.isfinite(spread_values)
                    )
                    rr_valid = rr_values[complete]
                    confidence_valid = confidence_values[complete]
                    spread_valid = spread_values[complete]
                    previous_windows = window_numbers[previous_rows_array][complete]
                else:
                    rr_valid = np.empty(0, dtype=float)
                    confidence_valid = np.empty(0, dtype=float)
                    spread_valid = np.empty(0, dtype=float)
                    previous_windows = np.empty(0, dtype=np.int64)

                available = len(rr_valid)
                if available:
                    rr_median = float(np.median(rr_valid))
                    rr_mad = float(np.median(np.abs(rr_valid - rr_median)))
                    nonnegative_weights = np.clip(confidence_valid, 0.0, None)
                    if float(nonnegative_weights.sum()) > 1e-12:
                        weighted_mean = float(np.average(rr_valid, weights=nonnegative_weights))
                    else:
                        weighted_mean = rr_median
                    if available >= 2 and np.ptp(previous_windows) > 0:
                        centered_window = previous_windows - previous_windows.mean()
                        trend = float(
                            np.dot(centered_window, rr_valid - rr_valid.mean())
                            / np.dot(centered_window, centered_window)
                        )
                    else:
                        trend = 0.0
                    values[row, column : column + 8] = (
                        rr_median,
                        rr_mad,
                        weighted_mean,
                        trend,
                        float(np.mean(confidence_valid)),
                        float(np.median(spread_valid)),
                        available / window,
                        float(available >= 2),
                    )
                # With no complete preceding radar estimate all eight fields
                # retain their initialized zero values.
                column += 8

    return values, names


def append_causal_history_features(
    aux: np.ndarray,
    metadata: pd.DataFrame,
    *,
    lags: tuple[int, ...] = DEFAULT_HISTORY_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_HISTORY_ROLLING_WINDOWS,
) -> tuple[np.ndarray, list[str]]:
    """Append :func:`causal_history_features` to an auxiliary feature matrix."""

    aux_array = np.asarray(aux)
    if aux_array.ndim != 2 or len(aux_array) != len(metadata):
        raise ValueError("aux must be a 2-D array with one row per metadata row")
    history, names = causal_history_features(
        metadata, lags=lags, rolling_windows=rolling_windows
    )
    output_dtype = np.result_type(aux_array.dtype, np.float32)
    augmented = np.concatenate(
        [aux_array.astype(output_dtype, copy=False), history.astype(output_dtype, copy=False)], axis=1
    )
    return augmented, names


def _positive_unique_ints(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if any(value <= 0 for value in normalized):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized
