from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "$2"
OUT = WORKSPACE / "outputs" / "breathing_v3"
BOUNDARIES = OUT / "all27_boundary_decisions.csv"
DATASET = OUT / "waveform_training_dataset.npz"

sys.path.insert(0, str(WORKSPACE / "breathing_v3"))
sys.path.insert(0, str(WORKSPACE / "heart_rate_hold"))
from run_development_baseline import bio_interval, load_biopac, read_csv, sync_offset  # noqa: E402
from analyze_heart_rate import RADAR_FS, RANGE_BINS, fft_bandpass, load_radar_interval, spectral_peak  # noqa: E402


WINDOW_S = 20.0
STRIDE_S = 5.0
MODEL_FS = 10.0
STEPS = int(WINDOW_S * MODEL_FS)
PATHS_PER_RADAR = 8
COMMON_LOW_HZ = 0.07
COMMON_HIGH_HZ = 0.70


def definitions(row: dict):
    s1 = float(row["s01_content_start_s"])
    s2 = float(row["s02_content_start_s"])
    apnea_start = float(row["apnea_start_s"])
    apnea_resume = float(row["apnea_resume_s"])
    return [
        ("S01", "FACE_R1", s1 + 5.0, float(row["turn1_s"]) - 5.0),
        ("S01", "FACE_R2", float(row["turn1_s"]) + 5.0, float(row["turn2_s"]) - 5.0),
        ("S01", "FACE_R3", float(row["turn2_s"]) + 5.0, float(row["m02_s"]) - 5.0),
        ("S02", "NORMAL", s2 + 5.0, s2 + 55.0),
        ("S02", "SLOW", s2 + 65.0, s2 + 115.0),
        ("S02", "HOLD", apnea_start + 2.0, apnea_resume - 2.0),
        ("S02", "POST_HOLD", apnea_resume + 5.0, s2 + 205.0),
        ("S02", "SQUAT", s2 + 230.0, s2 + 280.0),
        ("S02", "POST_EXERCISE", s2 + 305.0, s2 + 355.0),
    ]


def fixed_windows(start: float, end: float):
    t = float(start)
    while t + WINDOW_S <= float(end) + 1e-9:
        yield round(t, 6), round(t + WINDOW_S, 6)
        t += STRIDE_S


def resample(x: np.ndarray, source_fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    source_t = np.arange(len(x), dtype=np.float64) / source_fs
    target_t = np.arange(STEPS, dtype=np.float64) / MODEL_FS
    return np.interp(target_t, source_t, x).astype(np.float32)


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    scale = float(np.std(x))
    if scale < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - np.mean(x)) / scale, -6.0, 6.0).astype(np.float32)


def iq_whiten(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove I/Q offset, scale imbalance and linear I-Q correlation per range bin."""
    pair = np.column_stack((values.real, values.imag)).astype(np.float64)
    pair -= np.mean(pair, axis=0, keepdims=True)
    covariance = pair.T @ pair / max(len(pair) - 1, 1)
    covariance += np.eye(2) * max(float(np.trace(covariance)) * 1e-6, 1e-9)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    whitening = eigenvector @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalue, 1e-9))) @ eigenvector.T
    corrected = pair @ whitening
    return corrected[:, 0], corrected[:, 1]


def detrend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4:
        return x - np.mean(x)
    t = np.arange(len(x), dtype=np.float64)
    coefficient = np.polyfit(t, x, 1)
    return x - np.polyval(coefficient, t)


def radar_paths(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = []
    corrected = {}
    for bin_index in range(3, RANGE_BINS - 3):
        i_value, q_value = iq_whiten(data[:, bin_index])
        phase = detrend(np.unwrap(np.arctan2(q_value, i_value)))
        phase_bad = float(np.mean(np.abs(np.diff(phase)) > 1.2)) if len(phase) > 1 else 1.0
        filtered = fft_bandpass(phase, RADAR_FS, COMMON_LOW_HZ, COMMON_HIGH_HZ)
        frequency, spectral_score, _, _ = spectral_peak(
            filtered, RADAR_FS, COMMON_LOW_HZ, COMMON_HIGH_HZ, 4096
        )
        if not math.isfinite(float(frequency)):
            continue
        quality = math.log1p(max(float(spectral_score), 0.0))
        quality *= math.sqrt(max(float(np.std(filtered)), 1e-9))
        quality *= max(0.05, 1.0 - min(1.0, 5.0 * phase_bad))
        candidates.append((quality, bin_index))
        corrected[bin_index] = (i_value, q_value, filtered)
    candidates.sort(reverse=True)
    chosen = []
    for _quality, bin_index in candidates:
        if all(abs(bin_index - existing) >= 2 for existing in chosen):
            chosen.append(bin_index)
        if len(chosen) == PATHS_PER_RADAR:
            break
    if len(chosen) < PATHS_PER_RADAR:
        for _quality, bin_index in candidates:
            if bin_index not in chosen:
                chosen.append(bin_index)
            if len(chosen) == PATHS_PER_RADAR:
                break
    if len(chosen) != PATHS_PER_RADAR:
        raise ValueError("eight usable paths were not found")

    channels, scores = [], []
    score_by_bin = {bin_index: quality for quality, bin_index in candidates}
    for bin_index in chosen:
        i_value, q_value, phase = corrected[bin_index]
        channels.extend([
            standardize(resample(i_value, RADAR_FS)),
            standardize(resample(q_value, RADAR_FS)),
            standardize(resample(phase, RADAR_FS)),
        ])
        scores.append(score_by_bin[bin_index])
    return np.stack(channels, axis=0), np.asarray(chosen, dtype=np.int16), np.asarray(scores, dtype=np.float32)


def target_waveform(rsp_segment: np.ndarray, fs: float):
    filtered = fft_bandpass(detrend(rsp_segment), fs, COMMON_LOW_HZ, COMMON_HIGH_HZ)
    rr_hz, _score, _f, _p = spectral_peak(filtered, fs, COMMON_LOW_HZ, COMMON_HIGH_HZ, 16384)
    waveform = standardize(resample(filtered, fs))
    rr_bpm = float(rr_hz * 60.0) if math.isfinite(float(rr_hz)) else math.nan
    return waveform, rr_bpm


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    inputs, targets, rr_targets = [], [], []
    waveform_masks, motion_targets, apnea_targets = [], [], []
    selected_bins, path_scores = [], []
    rows, exclusions = [], []
    for boundary in read_csv(BOUNDARIES):
        subject = boundary["subject"]
        offset = sync_offset(subject)
        rsp, _ecg, bio_fs, _labels, _path = load_biopac(subject)
        before = len(rows)
        for experiment, state, stable_start, stable_end in definitions(boundary):
            for start_s, end_s in fixed_windows(stable_start, stable_end):
                try:
                    radar_channels, bins, scores = [], [], []
                    for radar in (1, 2, 3):
                        current, current_bins, current_scores = radar_paths(
                            load_radar_interval(subject, radar, start_s, end_s)
                        )
                        radar_channels.append(current)
                        bins.append(current_bins)
                        scores.append(current_scores)
                    rsp_segment = bio_interval(rsp, bio_fs, offset, start_s, end_s)
                    waveform, rr_bpm = target_waveform(rsp_segment, bio_fs)

                    waveform_evaluable = state not in {"HOLD", "SQUAT"}
                    motion_evaluable = not (subject == "S18_LJH" and state == "SQUAT")
                    index = len(rows)
                    inputs.append(np.concatenate(radar_channels, axis=0))
                    targets.append(waveform)
                    rr_targets.append(rr_bpm if waveform_evaluable else math.nan)
                    waveform_masks.append(int(waveform_evaluable))
                    motion_targets.append(int(state == "SQUAT") if motion_evaluable else -1)
                    apnea_targets.append(int(state == "HOLD"))
                    selected_bins.append(np.stack(bins, axis=0))
                    path_scores.append(np.stack(scores, axis=0))
                    rows.append({
                        "index": index,
                        "subject": subject,
                        "experiment": experiment,
                        "state": state,
                        "start_s": start_s,
                        "end_s": end_s,
                        "waveform_evaluable": int(waveform_evaluable),
                        "rr_target_bpm": "" if not waveform_evaluable else rr_bpm,
                        "motion_target": motion_targets[-1],
                        "apnea_target": apnea_targets[-1],
                    })
                except Exception as exc:
                    exclusions.append({
                        "subject": subject, "experiment": experiment, "state": state,
                        "start_s": start_s, "end_s": end_s,
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
        print(f"[waveform-data] {subject}: {len(rows)-before} windows", flush=True)

    np.savez_compressed(
        DATASET,
        inputs=np.asarray(inputs, dtype=np.float32),
        waveform_targets=np.asarray(targets, dtype=np.float32),
        rr_targets=np.asarray(rr_targets, dtype=np.float32),
        waveform_masks=np.asarray(waveform_masks, dtype=np.int8),
        motion_targets=np.asarray(motion_targets, dtype=np.int8),
        apnea_targets=np.asarray(apnea_targets, dtype=np.int8),
        subjects=np.asarray([row["subject"] for row in rows], dtype=object),
        experiments=np.asarray([row["experiment"] for row in rows], dtype=object),
        states=np.asarray([row["state"] for row in rows], dtype=object),
        starts=np.asarray([row["start_s"] for row in rows], dtype=np.float32),
        selected_bins=np.asarray(selected_bins, dtype=np.int16),
        path_scores=np.asarray(path_scores, dtype=np.float32),
    )
    write_csv(OUT / "waveform_training_manifest.csv", rows)
    write_csv(
        OUT / "waveform_training_exclusions.csv", exclusions,
        ["subject", "experiment", "state", "start_s", "end_s", "reason"],
    )
    summary = {
        "subject_count": len(set(row["subject"] for row in rows)),
        "window_count": len(rows),
        "waveform_rr_window_count": int(sum(waveform_masks)),
        "motion_positive_count": int(sum(target == 1 for target in motion_targets)),
        "motion_unknown_count": int(sum(target < 0 for target in motion_targets)),
        "apnea_positive_count": int(sum(apnea_targets)),
        "state_counts": dict(sorted(Counter(row["state"] for row in rows).items())),
        "input_shape": list(np.asarray(inputs).shape[1:]),
        "target_shape": list(np.asarray(targets).shape[1:]),
        "window_seconds": WINDOW_S,
        "stride_seconds": STRIDE_S,
        "model_fs_hz": MODEL_FS,
        "paths_per_radar": PATHS_PER_RADAR,
        "channels_per_path": ["iq_whitened_i", "iq_whitened_q", "common_band_phase"],
        "common_band_hz": [COMMON_LOW_HZ, COMMON_HIGH_HZ],
        "biopac_rule": "BIOPAC is target only and never appears in model input or path selection.",
        "state_input_rule": "State labels define training windows and targets but are not model inputs.",
        "exclusion_count": len(exclusions),
    }
    (OUT / "waveform_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
