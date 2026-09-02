from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from run_development_baseline import (
    BOUNDARIES,
    MANIFEST,
    OUT,
    RADAR_FS,
    RR_BANDS,
    aggregate_radars,
    analyze_radar,
    bio_interval,
    load_biopac,
    load_radar_interval,
    read_csv,
    rsp_reference,
    sync_offset,
    zscore,
)


WINDOW_S = 30.0
STRIDE_S = 5.0
MODEL_FS = 5.0
EXPECTED_STEPS = int(WINDOW_S * MODEL_FS)
ALL27_BOUNDARIES = OUT / "all27_boundary_decisions.csv"

SCALAR_NAMES = [
    *[f"radar{r}_{name}" for r in (1, 2, 3) for name in (
        "log_quality", "phase_bad", "spectral_entropy", "peak_fraction",
        "edge_margin_hz", "motion_index", "rr_bpm",
    )],
    "radar_rr_mad_bpm", "edge_quality", "motion_quality",
    "max_motion_index", "confidence", "quality_best_radar_scaled",
]


def fixed_windows(start_s: float, end_s: float):
    t = float(start_s)
    while t + WINDOW_S <= float(end_s) + 1e-9:
        yield round(t, 6), round(t + WINDOW_S, 6)
        t += STRIDE_S


def definitions(row: dict):
    s1 = float(row["s01_content_start_s"])
    s2 = float(row["s02_content_start_s"])
    return [
        ("S01", "FACE_R1", s1 + 5.0, float(row["turn1_s"]) - 5.0, True),
        ("S01", "FACE_R2", float(row["turn1_s"]) + 5.0, float(row["turn2_s"]) - 5.0, True),
        ("S01", "FACE_R3", float(row["turn2_s"]) + 5.0, float(row["m02_s"]) - 5.0, True),
        ("S02", "NORMAL", s2 + 5.0, s2 + 55.0, True),
        ("S02", "SLOW", s2 + 65.0, s2 + 115.0, True),
        ("S02", "POST_HOLD", float(row["apnea_resume_s"]) + 5.0, s2 + 205.0, True),
        ("S02", "SQUAT", s2 + 230.0, s2 + 280.0, False),
        ("S02", "POST_EXERCISE", s2 + 305.0, s2 + 355.0, True),
    ]


def downsample_trace(trace: np.ndarray) -> np.ndarray:
    trace = zscore(np.asarray(trace, dtype=float))
    source_t = np.arange(len(trace), dtype=float) / RADAR_FS
    target_t = np.arange(EXPECTED_STEPS, dtype=float) / MODEL_FS
    sampled = np.interp(target_t, source_t, trace)
    return sampled.astype(np.float32)


def scalar_features(radar_results: list[dict], aggregate: dict) -> np.ndarray:
    values: list[float] = []
    for item in radar_results:
        values.extend([
            math.log1p(max(float(item["quality"]), 0.0)),
            float(item["phase_bad"]),
            float(item["spectral_entropy"]),
            float(item["peak_fraction"]),
            float(item["edge_margin_hz"]),
            float(item["motion_index"]),
            float(item["fused_rr_bpm"]),
        ])
    values.extend([
        float(aggregate["radar_rr_mad_bpm"]),
        float(aggregate["edge_quality"]),
        float(aggregate["motion_quality"]),
        float(aggregate["max_motion_index"]),
        float(aggregate["confidence"]),
        (float(aggregate["quality_best_radar"]) - 1.0) / 2.0,
    ])
    return np.asarray(values, dtype=np.float32)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    if not rows and not fields:
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["development", "all27"], default="development")
    args = parser.parse_args()
    manifest = {r["subject"]: r for r in read_csv(MANIFEST)}
    if args.cohort == "all27":
        boundary_rows = [
            r for r in read_csv(ALL27_BOUNDARIES)
            if manifest.get(r["subject"], {}).get("usability") == "usable_both"
        ]
        prefix = "all27"
    else:
        boundary_rows = [
            r for r in read_csv(BOUNDARIES)
            if manifest.get(r["subject"], {}).get("cohort") == "development"
        ]
        prefix = "development"

    waveforms, scalars, rr_targets, motion_targets = [], [], [], []
    rows, exclusions = [], []
    for row in boundary_rows:
        subject = row["subject"]
        offset = sync_offset(subject)
        rsp, _ecg, bio_fs, _labels, _path = load_biopac(subject)
        subject_before = len(rows)
        for experiment, state, stable_start, stable_end, rr_evaluable in definitions(row):
            band_key = "S01" if experiment == "S01" else state
            low, high = RR_BANDS[band_key]
            state_windows = list(fixed_windows(stable_start, stable_end))
            if not state_windows:
                exclusions.append({
                    "subject": subject, "experiment": experiment, "state": state,
                    "start_s": stable_start, "end_s": stable_end,
                    "reason": "stable interval shorter than 30 seconds",
                })
                continue
            for start_s, end_s in state_windows:
                try:
                    radar_results = [
                        analyze_radar(load_radar_interval(subject, radar, start_s, end_s), low, high)
                        for radar in (1, 2, 3)
                    ]
                    aggregate = aggregate_radars(radar_results)
                    traces = []
                    for item in radar_results:
                        wave = downsample_trace(item["waveform"])
                        delta = np.diff(wave, prepend=wave[0]).astype(np.float32)
                        traces.extend([wave, delta])
                    waveform = np.stack(traces, axis=1)
                    scalar = scalar_features(radar_results, aggregate)

                    rr_target = math.nan
                    if rr_evaluable:
                        rsp_seg = bio_interval(rsp, bio_fs, offset, start_s, end_s)
                        rr_target, _ = rsp_reference(rsp_seg, bio_fs, low, high)
                        if not math.isfinite(rr_target):
                            raise ValueError("BIOPAC RR target is not finite")

                    index = len(rows)
                    motion_evaluable = not (subject == "S18_LJH" and state == "SQUAT")
                    motion_target = int(state == "SQUAT") if motion_evaluable else -1
                    waveforms.append(waveform)
                    scalars.append(scalar)
                    rr_targets.append(rr_target)
                    motion_targets.append(motion_target)
                    rows.append({
                        "index": index, "subject": subject, "experiment": experiment,
                        "state": state, "start_s": start_s, "end_s": end_s,
                        "rr_evaluable": int(rr_evaluable),
                        "rr_target_bpm": "" if not rr_evaluable else rr_target,
                        "motion_evaluable": int(motion_evaluable),
                        "motion_target": motion_target,
                        "radar_consensus_rr_bpm": aggregate["consensus_rr_bpm"],
                        "baseline_abs_error": "" if not rr_evaluable else abs(aggregate["consensus_rr_bpm"] - rr_target),
                        "baseline_confidence": aggregate["confidence"],
                        "baseline_reject": aggregate["reject"],
                    })
                except Exception as exc:
                    exclusions.append({
                        "subject": subject, "experiment": experiment, "state": state,
                        "start_s": start_s, "end_s": end_s,
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
        print(f"[windows] {subject}: {len(rows) - subject_before}개", flush=True)

    waveforms_a = np.asarray(waveforms, dtype=np.float32)
    scalars_a = np.asarray(scalars, dtype=np.float32)
    rr_targets_a = np.asarray(rr_targets, dtype=np.float32)
    motion_targets_a = np.asarray(motion_targets, dtype=np.int64)
    np.savez_compressed(
        OUT / f"{prefix}_training_windows.npz",
        waveforms=waveforms_a,
        scalars=scalars_a,
        rr_targets=rr_targets_a,
        motion_targets=motion_targets_a,
        subjects=np.asarray([r["subject"] for r in rows], dtype=object),
        experiments=np.asarray([r["experiment"] for r in rows], dtype=object),
        states=np.asarray([r["state"] for r in rows], dtype=object),
        starts=np.asarray([r["start_s"] for r in rows], dtype=np.float32),
        ends=np.asarray([r["end_s"] for r in rows], dtype=np.float32),
        scalar_names=np.asarray(SCALAR_NAMES, dtype=object),
    )
    write_csv(OUT / f"{prefix}_training_windows.csv", rows)
    write_csv(
        OUT / f"{prefix}_training_window_exclusions.csv",
        exclusions,
        ["subject", "experiment", "state", "start_s", "end_s", "reason"],
    )

    valid_rr = np.isfinite(rr_targets_a)
    baseline_errors = [float(r["baseline_abs_error"]) for r in rows if r["baseline_abs_error"] != ""]
    summary = {
        "purpose": (
            "ANN/SNN all-27 subject-level cross-validation dataset"
            if args.cohort == "all27"
            else "ANN/SNN development-only window dataset"
        ),
        "window_seconds": WINDOW_S,
        "stride_seconds": STRIDE_S,
        "radar_source_hz": RADAR_FS,
        "model_timeseries_hz": MODEL_FS,
        "time_steps": EXPECTED_STEPS,
        "timeseries_channels": [f"radar{r}_{kind}" for r in (1, 2, 3) for kind in ("phase", "delta")],
        "scalar_features": SCALAR_NAMES,
        "subjects": sorted(set(r["subject"] for r in rows)),
        "subject_count": len(set(r["subject"] for r in rows)),
        "window_count": len(rows),
        "rr_window_count": int(valid_rr.sum()),
        "motion_positive_window_count": int(np.sum(motion_targets_a == 1)),
        "motion_unknown_window_count": int(np.sum(motion_targets_a < 0)),
        "state_counts": dict(sorted(Counter(r["state"] for r in rows).items())),
        "subject_counts": dict(sorted(Counter(r["subject"] for r in rows).items())),
        "exclusion_count": len(exclusions),
        "baseline_window_mae_bpm": float(np.mean(baseline_errors)) if baseline_errors else math.nan,
        "split_rule": "All windows from one subject must stay in the same train/validation/test partition.",
        "biopac_rule": "BIOPAC RSP is used only for rr_targets and evaluation, never as model input.",
    }
    (OUT / f"{prefix}_training_windows_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / f"{prefix}_training_preprocess_config.json").write_text(
        json.dumps({
            "window_seconds": WINDOW_S,
            "stride_seconds": STRIDE_S,
            "source_fs_hz": RADAR_FS,
            "model_fs_hz": MODEL_FS,
            "time_steps": EXPECTED_STEPS,
            "rr_bands_hz": RR_BANDS,
            "scalar_names": SCALAR_NAMES,
            "data_cohort": args.cohort,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
