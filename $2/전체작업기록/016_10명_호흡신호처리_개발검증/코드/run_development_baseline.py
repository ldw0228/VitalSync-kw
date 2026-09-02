from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "$2"
OUT = WORKSPACE / "outputs" / "breathing_v3"
MANIFEST = OUT / "subject_protocol_manifest.csv"
BOUNDARIES = WORKSPACE / "outputs" / "snn_v2" / "boundary_decisions.csv"
SYNC_ROOT = WORKSPACE / "sync_results"

sys.path.insert(0, str(WORKSPACE / "heart_rate_hold"))
sys.path.insert(0, str(WORKSPACE / "orientation_multipath"))
from analyze_heart_rate import RADAR_FS, RANGE_BINS, fft_bandpass, load_radar_interval, spectral_peak  # noqa: E402
from analyze_orientation_multipath import bio_interval, load_biopac, motion_index, respiration_phase_trace  # noqa: E402


RR_BANDS = {
    # The non-slow protocols should not select residual drift at 0.08-0.12 Hz.
    # These lower limits are protocol priors, fixed on the development cohort.
    "S01": (0.15, 0.45),
    "NORMAL": (0.15, 0.50),
    "SLOW": (0.07, 0.40),
    "POST_HOLD": (0.15, 0.55),
    "SQUAT": (0.15, 0.55),
    "POST_EXERCISE": (0.15, 0.65),
}


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x))
    return (x - np.mean(x)) / sd if sd > 1e-12 else np.zeros_like(x)


def spectral_features(trace: np.ndarray, low: float, high: float) -> dict:
    filtered = fft_bandpass(trace, RADAR_FS, low, high)
    freq, snr, band_f, power = spectral_peak(filtered, RADAR_FS, low, high, 8192)
    if not finite(freq) or len(power) == 0:
        return {"trace": filtered, "freq": math.nan, "snr": 0.0, "entropy": 1.0, "peak_fraction": 0.0}
    prob = power / (float(np.sum(power)) + 1e-12)
    entropy = float(-np.sum(prob * np.log(prob + 1e-12)) / max(math.log(len(prob)), 1e-12))
    peak_fraction = float(np.max(prob))
    return {"trace": filtered, "freq": float(freq), "snr": float(snr), "entropy": entropy, "peak_fraction": peak_fraction}


def analyze_radar(data: np.ndarray, low: float, high: float) -> dict:
    candidates = []
    for bin_idx in range(3, RANGE_BINS - 3):
        phase, phase_bad = respiration_phase_trace(data[:, bin_idx])
        spec = spectral_features(phase, low, high)
        if not finite(spec["freq"]):
            continue
        phase_quality = float(np.clip(1.0 - 5.0 * phase_bad, 0.02, 1.0))
        concentration = max(0.0, 1.0 - spec["entropy"])
        edge_margin = max(0.0, spec["freq"] - low)
        edge_factor = float(np.clip(edge_margin / 0.035, 0.10, 1.0))
        base_quality = math.log1p(spec["snr"]) * math.sqrt(max(spec["peak_fraction"], 1e-9))
        base_quality *= phase_quality * math.sqrt(max(float(np.std(spec["trace"])), 1e-12))
        base_quality *= edge_factor
        candidates.append({
            "bin": bin_idx, "phase": phase, **spec, "phase_bad": float(phase_bad),
            "phase_quality": phase_quality, "edge_margin_hz": edge_margin,
            "base_quality": base_quality, "quality": base_quality,
        })
    if not candidates:
        raise ValueError("호흡 후보 없음")

    # Radar-only cross-bin consensus. This does not use BIOPAC labels.
    for item in candidates:
        support_values = []
        for other in candidates:
            if other is item or abs(other["bin"] - item["bin"]) < 3:
                continue
            delta = abs(other["freq"] - item["freq"])
            if delta <= 0.02:
                support_values.append(other["base_quality"] * math.exp(-delta / 0.01))
        support = float(sum(sorted(support_values, reverse=True)[:6]))
        item["support"] = support
        item["quality"] = item["base_quality"] * (1.0 + math.log1p(max(support, 0.0)))

    candidates.sort(key=lambda x: x["quality"], reverse=True)
    primary = candidates[0]
    auxiliary = None
    aux_rank = -math.inf
    for other in candidates[1:25]:
        if abs(other["bin"] - primary["bin"]) < 4:
            continue
        if abs(other["freq"] - primary["freq"]) > 0.03:
            continue
        corr = float(np.corrcoef(zscore(primary["trace"]), zscore(other["trace"]))[0, 1])
        if not finite(corr) or abs(corr) < 0.35:
            continue
        rank = abs(corr) * math.sqrt(max(primary["quality"] * other["quality"], 0.0))
        if rank > aux_rank:
            auxiliary = dict(other)
            auxiliary["correlation"] = corr
            aux_rank = rank

    fused = zscore(primary["trace"])
    if auxiliary is not None:
        sign = 1.0 if auxiliary["correlation"] >= 0 else -1.0
        weight = float(np.clip(auxiliary["quality"] / max(primary["quality"] + auxiliary["quality"], 1e-12), 0.20, 0.45))
        fused = (1.0 - weight) * zscore(primary["trace"]) + weight * sign * zscore(auxiliary["trace"])
    fused_spec = spectral_features(fused, low, high)
    return {
        "primary_bin": primary["bin"],
        "auxiliary_bin": auxiliary["bin"] if auxiliary else "",
        "auxiliary_corr": auxiliary["correlation"] if auxiliary else "",
        "primary_rr_bpm": primary["freq"] * 60.0,
        "fused_rr_bpm": fused_spec["freq"] * 60.0,
        "quality": primary["quality"],
        "phase_bad": primary["phase_bad"],
        "spectral_entropy": primary["entropy"],
        "peak_fraction": primary["peak_fraction"],
        "edge_margin_hz": primary["edge_margin_hz"],
        "motion_index": motion_index(data),
        "waveform": fused,
    }


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cut = 0.5 * float(np.sum(weights))
    return float(values[np.searchsorted(np.cumsum(weights), cut, side="left")])


def aggregate_radars(results: list[dict]) -> dict:
    rr = np.asarray([r["fused_rr_bpm"] for r in results], dtype=float)
    q = np.asarray([max(float(r["quality"]), 1e-12) for r in results], dtype=float)
    best = int(np.argmax(q))
    center = float(np.median(rr))
    keep = np.abs(rr - center) <= 3.0
    if not np.any(keep):
        keep[:] = True
    consensus = weighted_median(rr[keep], q[keep])
    mad = float(np.median(np.abs(rr - np.median(rr))))
    agreement = math.exp(-mad / 2.0)
    phase_quality = math.exp(-5.0 * float(results[best]["phase_bad"]))
    edge_quality = float(np.clip(np.median([r["edge_margin_hz"] for r in results]) / 0.035, 0.0, 1.0))
    median_motion = float(np.median([r["motion_index"] for r in results]))
    max_motion = float(np.max([r["motion_index"] for r in results]))
    motion_quality = float(np.clip((0.075 - max_motion) / 0.035, 0.0, 1.0))
    concentration = float(q[best] / np.sum(q))
    concentration = float(np.clip((concentration - 1 / 3) / (2 / 3), 0.0, 1.0))
    confidence = float(np.clip(0.35 * agreement + 0.20 * phase_quality + 0.15 * concentration + 0.20 * edge_quality + 0.10 * motion_quality, 0.0, 1.0))
    return {
        "quality_best_radar": best + 1,
        "quality_best_rr_bpm": rr[best],
        "consensus_rr_bpm": consensus,
        "radar_rr_mad_bpm": mad,
        "edge_quality": edge_quality,
        "motion_quality": motion_quality,
        "max_motion_index": max_motion,
        "confidence": confidence,
        "reject": int(confidence < 0.58 or mad > 4.0 or max_motion > 0.055),
    }


def max_abs_corr(radar: np.ndarray, rsp: np.ndarray, rsp_fs: float, max_lag_s: float = 2.0) -> float:
    if len(radar) < 16 or len(rsp) < 16:
        return math.nan
    target = np.interp(np.linspace(0, len(rsp) - 1, len(radar)), np.arange(len(rsp)), zscore(rsp))
    radar = zscore(radar)
    max_lag = int(round(max_lag_s * RADAR_FS))
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = radar[-lag:], target[:lag]
        elif lag > 0:
            a, b = radar[:-lag], target[lag:]
        else:
            a, b = radar, target
        if len(a) >= 16:
            corr = abs(float(np.corrcoef(a, b)[0, 1]))
            if finite(corr):
                best = max(best, corr)
    return best


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sync_offset(subject: str) -> float:
    return float(json.loads((SYNC_ROOT / subject / "sync_result.json").read_text(encoding="utf-8"))["offset_s"])


def rsp_reference(rsp: np.ndarray, fs: float, low: float, high: float) -> tuple[float, np.ndarray]:
    filtered = fft_bandpass(rsp, fs, low, high)
    freq, _, _, _ = spectral_peak(filtered, fs, low, high, 16384)
    return (float(freq) * 60.0 if finite(freq) else math.nan), filtered


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    manifest = {r["subject"]: r for r in read_csv(MANIFEST)}
    boundary_rows = [r for r in read_csv(BOUNDARIES) if manifest.get(r["subject"], {}).get("cohort") == "development"]
    observations, segments = [], []
    for row in boundary_rows:
        subject = row["subject"]
        offset = sync_offset(subject)
        rsp, _ecg, bio_fs, _labels, _path = load_biopac(subject)
        s1_content = float(row["s01_content_start_s"])
        s1_defs = [
            ("FACE_R1", 1, s1_content + 5.0, float(row["turn1_s"]) - 5.0),
            ("FACE_R2", 2, float(row["turn1_s"]) + 5.0, float(row["turn2_s"]) - 5.0),
            ("FACE_R3", 3, float(row["turn2_s"]) + 5.0, float(row["m02_s"]) - 5.0),
        ]
        s2_content = float(row["s02_content_start_s"])
        s2_defs = [
            ("NORMAL", None, s2_content + 5.0, s2_content + 55.0),
            ("SLOW", None, s2_content + 65.0, s2_content + 115.0),
            ("POST_HOLD", None, float(row["apnea_resume_s"]) + 5.0, s2_content + 205.0),
            ("SQUAT", None, s2_content + 230.0, s2_content + 280.0),
            ("POST_EXERCISE", None, s2_content + 305.0, s2_content + 355.0),
        ]
        for experiment, definitions in (("S01", s1_defs), ("S02", s2_defs)):
            for state, nominal_radar, start_s, end_s in definitions:
                if end_s - start_s < 20:
                    continue
                low, high = RR_BANDS["S01" if experiment == "S01" else state]
                rsp_seg = bio_interval(rsp, bio_fs, offset, start_s, end_s)
                gt_rr, rsp_filtered = rsp_reference(rsp_seg, bio_fs, low, high)
                radar_results = []
                for radar in (1, 2, 3):
                    result = analyze_radar(load_radar_interval(subject, radar, start_s, end_s), low, high)
                    radar_results.append(result)
                    observations.append({
                        "subject": subject, "experiment": experiment, "state": state,
                        "nominal_facing_radar": nominal_radar or "", "radar": radar,
                        "start_s": round(start_s, 3), "end_s": round(end_s, 3),
                        "duration_s": round(end_s - start_s, 3), "rsp_rr_bpm": gt_rr,
                        "primary_bin": result["primary_bin"], "auxiliary_bin": result["auxiliary_bin"],
                        "auxiliary_corr": result["auxiliary_corr"], "radar_rr_bpm": result["fused_rr_bpm"],
                        "primary_rr_bpm": result["primary_rr_bpm"],
                        "path_fused_rr_bpm": result["fused_rr_bpm"],
                        "primary_rr_abs_error": abs(result["primary_rr_bpm"] - gt_rr),
                        "path_fused_rr_abs_error": abs(result["fused_rr_bpm"] - gt_rr),
                        "rr_abs_error": abs(result["fused_rr_bpm"] - gt_rr),
                        "quality": result["quality"], "phase_bad": result["phase_bad"],
                        "spectral_entropy": result["spectral_entropy"], "peak_fraction": result["peak_fraction"],
                        "edge_margin_hz": result["edge_margin_hz"],
                        "motion_index": result["motion_index"],
                    })
                agg = aggregate_radars(radar_results)
                if state != "SLOW" and agg["edge_quality"] < 0.25:
                    agg["reject"] = 1
                estimate = agg["consensus_rr_bpm"]
                best_wave = radar_results[agg["quality_best_radar"] - 1]["waveform"]
                corr = max_abs_corr(best_wave, rsp_filtered, bio_fs)
                oracle_error = min(abs(r["fused_rr_bpm"] - gt_rr) for r in radar_results)
                segments.append({
                    "subject": subject, "experiment": experiment, "state": state,
                    "nominal_facing_radar": nominal_radar or "", "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3), "duration_s": round(end_s - start_s, 3),
                    "rsp_rr_bpm": gt_rr, "radar1_rr_bpm": radar_results[0]["fused_rr_bpm"],
                    "radar2_rr_bpm": radar_results[1]["fused_rr_bpm"], "radar3_rr_bpm": radar_results[2]["fused_rr_bpm"],
                    "radar1_primary_rr_bpm": radar_results[0]["primary_rr_bpm"],
                    "radar2_primary_rr_bpm": radar_results[1]["primary_rr_bpm"],
                    "radar3_primary_rr_bpm": radar_results[2]["primary_rr_bpm"],
                    **agg, "rr_abs_error": abs(estimate - gt_rr), "oracle_abs_error": oracle_error,
                    "waveform_abs_corr": corr,
                    "median_motion_index": float(np.median([r["motion_index"] for r in radar_results])),
                })
        print(f"[development] {subject} 완료", flush=True)

    write_csv(OUT / "development_radar_observations.csv", observations)
    write_csv(OUT / "development_segment_results.csv", segments)

    eval_rows = [r for r in segments if r["state"] != "SQUAT"]
    eval_observations = [r for r in observations if r["state"] != "SQUAT"]
    accepted = [r for r in eval_rows if not r["reject"]]
    def metric(rows, key="rr_abs_error"):
        values = np.asarray([float(r[key]) for r in rows if finite(r[key])], dtype=float)
        return {"n": int(len(values)), "mae": float(np.mean(values)) if len(values) else math.nan,
                "median_ae": float(np.median(values)) if len(values) else math.nan,
                "within_2": float(np.mean(values <= 2.0)) if len(values) else math.nan}
    summary = {
        "cohort": "development_only",
        "subjects": [r["subject"] for r in boundary_rows],
        "n_subjects": len(boundary_rows),
        "n_segments_total": len(segments),
        "n_rr_evaluable": len(eval_rows),
        "consensus_all": metric(eval_rows),
        "consensus_accepted": metric(accepted),
        "coverage": len(accepted) / max(len(eval_rows), 1),
        "oracle_single_radar": metric(eval_rows, "oracle_abs_error"),
        "path_candidate_effect": {
            "primary": metric(eval_observations, "primary_rr_abs_error"),
            "primary_plus_auxiliary": metric(eval_observations, "path_fused_rr_abs_error"),
            "auxiliary_candidate_ratio": sum(r["auxiliary_bin"] != "" for r in eval_observations) / max(len(eval_observations), 1),
            "terminology": "실험실 기하 확인 전 보조 경로 후보이며 물리적 ghost로 확정하지 않음",
        },
        "median_waveform_abs_corr": float(np.median([float(r["waveform_abs_corr"]) for r in eval_rows if finite(r["waveform_abs_corr"])])),
        "movement_detection": {
            "squat_detected": sum(int(r["reject"]) for r in segments if r["state"] == "SQUAT"),
            "squat_total": sum(r["state"] == "SQUAT" for r in segments),
            "non_squat_motion_false_positive": sum(float(r["max_motion_index"]) > 0.055 for r in eval_rows),
            "note": "S18_LJH 스쿼트 예정 구간은 세 레이더 모두 움직임 임계값 미만이므로 프로토콜 미수행/시간경계 확인 대상으로 기록",
        },
        "by_experiment": {name: metric([r for r in eval_rows if r["experiment"] == name]) for name in ("S01", "S02")},
        "by_state": {state: metric([r for r in eval_rows if r["state"] == state]) for state in sorted({r["state"] for r in eval_rows})},
        "leakage_statement": "채널·경로 선택과 confidence 계산에는 BIOPAC을 사용하지 않으며 BIOPAC은 사후 평가에만 사용함.",
        "validation_statement": "independent_validation 참가자는 본 실행에서 처리하지 않음.",
    }
    (OUT / "development_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
