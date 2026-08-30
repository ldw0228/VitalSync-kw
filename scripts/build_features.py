#!/usr/bin/env python3
"""Build causal radar range-frequency inputs and quality-controlled RR labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
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
)
from snn_rr.radar_timing import (  # noqa: E402
    causal_uniform_resample_radar_views_v1,
)
from snn_rr.range_tracking import (  # noqa: E402
    RangeTrack,
    fuse_range_track_window_features,
)


FEATURE_CACHE_ACQUISITION_SCHEMA = "snn_rr.feature_cache_acquisition.v2"
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
        SOURCE_ROOT / "snn_rr" / "preprocess.py",
        SOURCE_ROOT / "snn_rr" / "acquisition_contract.py",
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
    acquisition_provenance = (
        None
        if acquisition_contract is None
        else _session_acquisition_provenance(
            acquisition_contract, mode=str(acquisition_mode)
        )
    )
    use_range_auxiliary = bool(
        acquisition_contract is not None
        and acquisition_contract.range_feature_eligible
        and acquisition_contract.range_track_path is not None
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
        if all(previous.get(key) == value for key, value in expected.items()) and (
            acquisition_contract is None
            or _acquisition_cached_manifest_is_current(
                output_dir,
                previous,
                require_range_aux=use_range_auxiliary,
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

    biopac = load_biopac_mat(
        subject.biopac_path,
        strict=bool(
            acquisition_contract is not None and acquisition_mode == "strict"
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
            require_measured_timestamps=acquisition_mode == "strict",
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
        radar_arrays = [array for array in resampled.values]
        model_radar_times_s = resampled.times_s
        common_samples = len(model_radar_times_s)
        last_radar_time = resampled.origin_epoch_s + float(model_radar_times_s[-1])
        candidates = np.arange(
            window_samples, common_samples + 1, stride_samples, dtype=int
        )
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
        raise ValueError(f"{session_id} has no {data_cfg['window_seconds']} s overlap windows")

    maps: list[np.ndarray] = []
    auxiliary: list[np.ndarray] = []
    range_auxiliary: list[np.ndarray] = []
    radar_timing_valid_masks: list[np.ndarray] = []
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
        feature_map = raw_maps[:, :usable_frequencies].reshape(
            3, usable_frequencies // 2, 2, raw_maps.shape[-1]
        ).mean(axis=2, dtype=np.float32).astype(np.float16)
        full_grid = radar_features[0].frequencies_hz[:usable_frequencies]
        current_grid = full_grid.reshape(-1, 2).mean(axis=1).astype(np.float32)
        if pooled_frequency_grid is None:
            pooled_frequency_grid = current_grid
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
            max_estimator_disagreement_bpm=float(qc_cfg["max_estimator_disagreement_bpm"]),
            max_phase_residual_rad=float(qc_cfg["max_phase_residual_rad"]),
        )
        guard_samples = int(round(2.0 * biopac.sample_rate_hz))
        guard_rsp = biopac.rsp[
            max(0, bio_first - guard_samples) : min(len(biopac.rsp), bio_last + guard_samples)
        ]
        guard_clip_fraction = float(np.mean(np.abs(guard_rsp) >= 9.8))
        reference_valid = bool(
            reference.valid
            and guard_clip_fraction <= float(qc_cfg["max_clip_fraction"])
        )
        if acquisition_mode == "diagnostic":
            # Diagnostic proposals are retained for visual/quantitative audit,
            # but no row may masquerade as a trainable reference.
            reference_valid = False
        classical_error = abs(classical.rr_bpm - reference.rr_bpm)
        acceptable_prediction = bool(reference_valid and classical_error <= 2.0)
        row = {
            "session_id": session_id,
            "session_number": subject.subject_number,
            "identity": identity_for_session(session_id),
            "protocol": protocol_for_session(session_id),
            "window_number": window_number,
            "window_start_s": canonical_window_start_s,
            "window_end_s": canonical_window_end_s,
            "rr_bpm": reference.rr_bpm,
            "rr_spectral_bpm": reference.rr_spectral_bpm,
            "rr_phase_bpm": reference.rr_phase_bpm,
            "rr_events_bpm": reference.rr_events_bpm,
            "reference_valid": reference_valid,
            "reference_quality": reference.quality,
            "reference_sigma_bpm": float(
                np.clip(0.35 + 0.20 * reference.estimator_disagreement_bpm + 0.50 * (1 - reference.quality), 0.35, 2.0)
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
            "classical_rr_bpm": classical.rr_bpm,
            "classical_confidence": classical.confidence,
            "classical_error_bpm": classical_error,
            # Historical column name retained for cache compatibility.  This
            # is a target-dependent classical acceptability label, not an
            # intrinsic radar observability assertion.
            "radar_observable": acceptable_prediction,
            "classical_acceptable_within_2bpm": acceptable_prediction,
            "radar_peak_1_bpm": classical.radar_peaks_bpm[0],
            "radar_peak_2_bpm": classical.radar_peaks_bpm[1],
            "radar_peak_3_bpm": classical.radar_peaks_bpm[2],
            "radar_peak_spread_bpm": classical.consensus_spread_bpm,
        }
        if acquisition_contract is not None:
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
    np.save(maps_path, map_array, allow_pickle=False)
    np.save(aux_path, aux_array, allow_pickle=False)
    range_aux_array: np.ndarray | None = None
    range_aux_path = output_dir / "range_aux.npy"
    if range_tracks is not None:
        range_aux_array = np.stack(range_auxiliary).astype(np.float32)
        np.save(range_aux_path, range_aux_array, allow_pickle=False)
    np.save(frequencies_path, pooled_frequency_grid, allow_pickle=False)
    radar_timing_valid_mask_array: np.ndarray | None = None
    if acquisition_contract is not None:
        radar_timing_valid_mask_array = np.stack(
            radar_timing_valid_masks
        ).astype(np.bool_, copy=False)
        np.save(
            timing_mask_path,
            radar_timing_valid_mask_array,
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


def main() -> int:
    args = parse_args()
    subjects_filter_applied = args.subjects is not None
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    dataset_root = (args.dataset_root or REPOSITORY_ROOT / config["data"]["root"]).resolve()
    output_root = (args.cache_dir or REPOSITORY_ROOT / config["data"]["cache_dir"]).resolve()
    acquisition = None
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
        if acquisition.manifest.get("schema_version") != (
            "snn_rr.acquisition_reconstruction.v2"
        ):
            raise ValueError(
                "feature-cache schema v2 requires an acquisition reconstruction v2 "
                "manifest, including in diagnostic mode"
            )
        if args.acquisition_mode == "strict":
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
    manifest = build_dataset_manifest(dataset_root)
    selected = manifest.subjects
    if subjects_filter_applied:
        wanted = set(args.subjects)
        selected = tuple(subject for subject in selected if subject.subject_id in wanted)
        unknown = wanted - {subject.subject_id for subject in selected}
        if unknown:
            raise KeyError(f"unknown subjects: {sorted(unknown)}")

    results = []
    for subject in selected:
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
        results.append(result)

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


if __name__ == "__main__":
    raise SystemExit(main())
