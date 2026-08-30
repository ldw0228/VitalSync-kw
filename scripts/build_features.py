#!/usr/bin/env python3
"""Build causal radar range-frequency inputs and quality-controlled RR labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
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
)
from snn_rr.radar_timing import (  # noqa: E402
    block_mean_times,
    fuse_common_radar_timeline,
)
from snn_rr.range_tracking import (  # noqa: E402
    RangeTrack,
    fuse_range_track_window_features,
)


FEATURE_CACHE_ACQUISITION_SCHEMA = "snn_rr.feature_cache_acquisition.v1"


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

    values = np.asarray(values, dtype=np.float32).copy()
    locations = np.argwhere(np.abs(values) > threshold)
    for frame, range_bin in locations:
        history = values[max(0, frame - 4) : frame, range_bin]
        history = history[np.isfinite(history) & (np.abs(history) <= threshold)]
        values[frame, range_bin] = float(np.median(history)) if len(history) else 0.0
    return values, len(locations)


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
    source_fingerprint = _source_fingerprint(subject) if subject.usable else "missing"
    complete_files = [
        output_dir / "maps.npy",
        output_dir / "aux.npy",
        output_dir / "metadata.csv",
        output_dir / "frequencies_hz.npy",
        output_dir / "manifest.json",
    ]
    acquisition_provenance = (
        None
        if acquisition_contract is None
        else _session_acquisition_provenance(
            acquisition_contract, mode=str(acquisition_mode)
        )
    )
    if acquisition_contract is not None:
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
        if all(previous.get(key) == value for key, value in expected.items()):
            return {"session_id": session_id, "status": "ok", "cached": True, **previous}
        print(f"[{session_id}] stale cache provenance; rebuilding", flush=True)

    if not subject.usable:
        return {"session_id": session_id, "status": "skipped", "reason": "missing paired radar/BIOPAC"}
    if (acquisition_contract is None) != (acquisition_mode is None):
        raise ValueError("acquisition contract and mode must be supplied together")
    if acquisition_contract is not None:
        if acquisition_contract.session_id != session_id:
            raise ValueError(f"{session_id} acquisition contract ID mismatch")
        validate_raw_input_bindings(acquisition_contract, Path(subject.path).parent)
        if acquisition_mode == "strict" and not acquisition_contract.authorized:
            raise ValueError(f"{session_id} synchronization is not authorized")
        if acquisition_mode == "strict" and not acquisition_contract.scientific_eligible:
            raise ValueError(f"{session_id} acquisition reconstruction is not scientifically eligible")
        range_document = acquisition_contract.manifest.get("range_tracking", {})
        if not isinstance(range_document, dict) or range_document.get(
            "selected_session_layout"
        ) != "split_halves":
            raise ValueError(f"{session_id} does not have a verified split-halves range layout")
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

    biopac = load_biopac_mat(subject.biopac_path, strict=False)
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
        common_raw_frames = min(map(len, corrected_radar_frames))
        timeline = fuse_common_radar_timeline(
            relative_radar_times, radar_starts, radar_frame_sequences
        )
        radar_timing_summary = timeline.summary
        radar_arrays = [
            causal_block_mean(values[:common_raw_frames], downsample)
            for values in corrected_radar_frames
        ]
        model_radar_times_s = block_mean_times(
            timeline.times_s[:common_raw_frames], downsample
        )
        common_samples = min(
            len(model_radar_times_s), *(len(array) for array in radar_arrays)
        )
        model_radar_times_s = model_radar_times_s[:common_samples]
        radar_arrays = [array[:common_samples] for array in radar_arrays]
        last_radar_time = timeline.origin_epoch_s + float(model_radar_times_s[-1])
        candidates = np.arange(
            window_samples, common_samples + 1, stride_samples, dtype=int
        )
        valid_candidates: list[int] = []
        for end in candidates:
            start = end - window_samples
            radar_window_start = float(model_radar_times_s[start])
            period = float(np.median(np.diff(model_radar_times_s)))
            radar_window_end = float(model_radar_times_s[end - 1] + period)
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
    rows: list[dict[str, Any]] = []
    pooled_frequency_grid: np.ndarray | None = None
    range_tracks: tuple[RangeTrack, ...] | None = None
    range_auxiliary_names: tuple[str, ...] = ()
    if acquisition_contract is not None:
        assert model_radar_times_s is not None
        range_tracks, range_auxiliary_names = _load_range_track_contract(
            acquisition_contract, model_radar_times_s
        )
    for window_number, end in enumerate(end_indices):
        start = end - window_samples
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
        # reconstruction and one forward allowlist are bound to them.
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
            model_period_s = float(np.median(np.diff(model_radar_times_s)))
            radar_window_start_relative_s = float(model_radar_times_s[start])
            radar_window_end_relative_s = float(
                model_radar_times_s[end - 1] + model_period_s
            )
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
        bio_first = int(round(reference_start_biopac_s * biopac.sample_rate_hz))
        bio_last = int(round(reference_end_biopac_s * biopac.sample_rate_hz))
        if bio_first < 0 or bio_last > len(biopac.rsp):
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
            "window_start_s": window_start_epoch - bio_start,
            "window_end_s": window_end_epoch - bio_start,
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
    np.save(output_dir / "maps.npy", map_array, allow_pickle=False)
    np.save(output_dir / "aux.npy", aux_array, allow_pickle=False)
    range_aux_array: np.ndarray | None = None
    if acquisition_contract is not None:
        range_aux_array = np.stack(range_auxiliary).astype(np.float32)
        np.save(output_dir / "range_aux.npy", range_aux_array, allow_pickle=False)
    np.save(output_dir / "frequencies_hz.npy", pooled_frequency_grid, allow_pickle=False)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    subject_manifest = {
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
        "radar_start_epoch": legacy_radar_start,
        "radar_last_epoch": last_radar_time,
        "biopac_start_epoch": bio_start,
        "biopac_end_epoch": bio_end,
        "causal_downsample_latency_ms": 1000.0 * (downsample - 1) / radar_hz,
    }
    if acquisition_provenance is not None:
        subject_manifest.update(
            {
                "acquisition_contract": acquisition_provenance,
                "range_aux_shape": list(range_aux_array.shape),
                "range_aux_feature_names": list(range_auxiliary_names),
                "radar_timing_summary": radar_timing_summary,
                "legacy_timing_columns_preserved": [
                    "window_start_s",
                    "window_end_s",
                ],
                "protocol_column_semantics": (
                    "deprecated phase7 batch assignment broadcast for legacy compatibility; "
                    "use acquisition_phase for reconstructed stage"
                ),
            }
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(subject_manifest, indent=2), encoding="utf-8"
    )
    return {"session_id": session_id, "status": "ok", "cached": False, **subject_manifest}


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
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
        root_acquisition_contract = {
            "schema_version": FEATURE_CACHE_ACQUISITION_SCHEMA,
            "reconstruction_manifest": str(acquisition.manifest_path),
            "reconstruction_content_sha256": acquisition.content_sha256,
            "mode": args.acquisition_mode,
            "annotation_only_columns": list(ANNOTATION_ONLY_COLUMNS),
            "scientific_eligible": bool(
                args.acquisition_mode == "strict"
                and acquisition.manifest.get("scientific_eligible") is True
            ),
        }
    output_root.mkdir(parents=True, exist_ok=True)
    config_digest = hashlib.sha256(args.config.read_bytes()).hexdigest()
    pipeline_digest = _sha256_files(
        [
            Path(__file__),
            SOURCE_ROOT / "snn_rr" / "data.py",
            SOURCE_ROOT / "snn_rr" / "preprocess.py",
            SOURCE_ROOT / "snn_rr" / "acquisition_contract.py",
            SOURCE_ROOT / "snn_rr" / "radar_timing.py",
            SOURCE_ROOT / "snn_rr" / "range_tracking.py",
        ]
    )
    manifest = build_dataset_manifest(dataset_root)
    selected = manifest.subjects
    if args.subjects:
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
        results.append(
            build_subject(
                subject,
                config,
                output_root,
                args.force,
                config_sha256=config_digest,
                pipeline_sha256=pipeline_digest,
                acquisition_contract=session_contract,
                acquisition_mode=(
                    None if acquisition is None else args.acquisition_mode
                ),
            )
        )

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
                if key != "scientific_eligible"
            }
        )
        current_comparable = (
            None
            if root_acquisition_contract is None
            else {
                key: value
                for key, value in root_acquisition_contract.items()
                if key != "scientific_eligible"
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
    root_manifest = {
        "dataset_root": str(dataset_root),
        "config": str(args.config.resolve()),
        "config_sha256": config_digest,
        "pipeline_sha256": pipeline_digest,
        "sessions": merged_results,
    }
    if root_acquisition_contract is not None:
        ok_contracts = [
            item.get("acquisition_contract")
            for item in merged_results
            if item.get("status") == "ok"
        ]
        root_acquisition_contract["scientific_eligible"] = bool(
            ok_contracts
            and all(
                isinstance(item, dict) and item.get("scientific_eligible") is True
                for item in ok_contracts
            )
        )
        root_manifest["acquisition_contract"] = root_acquisition_contract
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
