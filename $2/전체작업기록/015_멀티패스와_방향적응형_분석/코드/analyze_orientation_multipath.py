from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "HAI_EXPERIMENT"
WORKSPACE = ROOT / "$2"
BOUNDARY_CSV = WORKSPACE / "outputs" / "snn_v2" / "boundary_decisions.csv"
MANIFEST_CSV = WORKSPACE / "outputs" / "snn_training" / "dataset_manifest.csv"
SYNC_ROOT = WORKSPACE / "sync_results"
OUT = WORKSPACE / "outputs" / "orientation_multipath"

sys.path.insert(0, str(WORKSPACE))
from matv5_reader import loadmat  # noqa: E402
sys.path.insert(0, str(WORKSPACE / "heart_rate_hold"))
from analyze_heart_rate import (  # noqa: E402
    RADAR_FS,
    RANGE_BINS,
    ecg_reference,
    fft_bandpass,
    load_radar_interval,
    moving_average,
    phase_trace,
    robust_segment_spectrum,
    spectral_peak,
)


TRIM_S = 5.0
RR_LOW = 0.08
RR_HIGH = 0.55
HR_LOW = 0.75
HR_HIGH = 2.70


def fnt(size=18, bold=False):
    candidates = ["malgunbd.ttf" if bold else "malgun.ttf", "arialbd.ttf" if bold else "arial.ttf"]
    for name in candidates:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def decode_rows(value):
    a = np.asarray(value)
    return ["".join(chr(int(c)) for c in row if c) for row in a]


def finite(value):
    return value is not None and math.isfinite(float(value))


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_boundaries():
    with BOUNDARY_CSV.open(encoding="utf-8-sig", newline="") as handle:
        return [{k: (v if k in {"subject", "resume_basis", "marker_basis", "label_basis"} else safe_float(v)) for k, v in row.items()} for row in csv.DictReader(handle)]


def sync_offset(subject):
    path = SYNC_ROOT / subject / "sync_result.json"
    with path.open(encoding="utf-8") as handle:
        return float(json.load(handle)["offset_s"])


def load_biopac(subject):
    paths = sorted((DATA_ROOT / subject / "BIOPAC").glob("*.mat"))
    if not paths:
        raise FileNotFoundError(subject)
    mat = loadmat(paths[0])
    data = np.asarray(mat["data"], dtype=np.float64)
    labels = decode_rows(mat.get("labels", []))
    isi_ms = float(np.asarray(mat.get("isi", [[4]])).reshape(-1)[0])
    fs = 1000.0 / isi_ms
    rsp_idx = next((i for i, label in enumerate(labels) if "RSP" in label.upper()), 0)
    ecg_idx = next((i for i, label in enumerate(labels) if "ECG" in label.upper()), min(1, data.shape[1] - 1))
    return data[:, rsp_idx], data[:, ecg_idx], fs, labels, paths[0]


def bio_interval(signal, fs, offset, start_s, end_s):
    lo = max(0, int(math.floor((start_s + offset) * fs)))
    hi = min(len(signal), int(math.ceil((end_s + offset) * fs)))
    if hi <= lo:
        return np.array([], dtype=float)
    return np.asarray(signal[lo:hi], dtype=float)


def rate_reference(rsp, ecg, fs):
    rsp_filtered = fft_bandpass(rsp, fs, RR_LOW, RR_HIGH)
    rr_hz, rr_score, _, _ = spectral_peak(rsp_filtered, fs, RR_LOW, RR_HIGH, 16384)
    ecg_ref = ecg_reference(ecg, fs)
    return {
        "rr_bpm": rr_hz * 60.0 if finite(rr_hz) else math.nan,
        "rr_score": rr_score,
        "hr_bpm": ecg_ref["hr_bpm"],
        "hr_quality": ecg_ref["quality"],
        "rsp_filtered": rsp_filtered,
    }


def zscore(x):
    x = np.asarray(x, dtype=float)
    scale = float(np.std(x))
    return (x - np.mean(x)) / scale if scale > 1e-12 else np.zeros_like(x)


def motion_index(data):
    if len(data) < 2:
        return math.nan
    denom = float(np.median(np.abs(data))) + 1e-9
    frame_change = np.median(np.abs(np.diff(data, axis=0)), axis=1) / denom
    return float(np.median(frame_change))


def respiration_phase_trace(values):
    """Phase trace for respiration; unlike the HR helper, do not remove a 2 s mean."""
    phase = np.unwrap(np.angle(values))
    bad = np.mean(np.abs(np.diff(phase)) > 1.2) if len(phase) > 1 else 1.0
    t = np.arange(len(phase), dtype=float)
    if len(phase) >= 4:
        coef = np.polyfit(t, phase, 1)
        phase = phase - np.polyval(coef, t)
    else:
        phase = phase - np.mean(phase)
    return phase, float(bad)


def analyze_radar_data(data):
    candidates = []
    traces = {}
    for bin_idx in range(3, RANGE_BINS - 3):
        resp_trace, bad = respiration_phase_trace(data[:, bin_idx])
        rr_trace = fft_bandpass(resp_trace, RADAR_FS, RR_LOW, RR_HIGH)
        rr_hz, rr_score, _, _ = spectral_peak(rr_trace, RADAR_FS, RR_LOW, RR_HIGH, 8192)
        if not finite(rr_hz):
            continue
        amp = float(np.median(np.abs(data[:, bin_idx])))
        resp_std = float(np.std(rr_trace))
        phase_quality = max(0.05, 1.0 - min(1.0, bad * 5.0))
        locate = math.log1p(max(rr_score, 0.0)) * math.sqrt(max(resp_std, 1e-12))
        locate *= math.log1p(max(amp, 0.0)) * phase_quality
        heart_trace, _ = phase_trace(data[:, bin_idx])
        hr_hz, hr_score, _, _, persistence = robust_segment_spectrum(heart_trace, RADAR_FS, HR_LOW, HR_HIGH, 8192, 12.0, 6.0)
        item = {
            "bin": bin_idx,
            "trace": resp_trace,
            "heart_trace": heart_trace,
            "rr_trace": rr_trace,
            "rr_hz": rr_hz,
            "rr_score": rr_score,
            "hr_hz": hr_hz,
            "hr_score": hr_score,
            "hr_persistence": persistence,
            "amplitude": amp,
            "phase_bad": bad,
            "phase_quality": phase_quality,
            "locate_score": locate,
        }
        candidates.append(item)
        traces[bin_idx] = item
    # Respiration is expected to recur across multiple separated bins. Reward
    # cross-bin frequency consensus so a single narrow-band artifact does not win.
    for item in candidates:
        support = 0.0
        for other in candidates:
            if other is item or abs(other["bin"] - item["bin"]) < 3:
                continue
            delta = abs(other["rr_hz"] - item["rr_hz"])
            if delta <= 0.025:
                support += math.log1p(max(other["rr_score"], 0.0)) * math.exp(-delta / 0.012)
        item["consensus_score"] = support
        item["locate_score"] *= math.log1p(max(support, 0.0))
    candidates.sort(key=lambda x: x["locate_score"], reverse=True)
    primary = candidates[0]

    ghost = None
    ghost_rank = -1.0
    p_rr = zscore(primary["rr_trace"])
    for item in candidates[1:20]:
        if abs(item["bin"] - primary["bin"]) < 4:
            continue
        if abs(item["rr_hz"] - primary["rr_hz"]) > 0.05:
            continue
        corr = float(np.corrcoef(p_rr, zscore(item["rr_trace"]))[0, 1])
        if not finite(corr):
            continue
        rank = abs(corr) * math.sqrt(max(item["rr_score"], 0.0) * max(primary["rr_score"], 0.0))
        if rank > ghost_rank:
            ghost_rank = rank
            ghost = dict(item)
            ghost["correlation"] = corr

    fused_trace = primary["trace"].copy()
    fused_heart_trace = primary["heart_trace"].copy()
    if ghost is not None:
        sign = 1.0 if ghost["correlation"] >= 0 else -1.0
        fused_trace = zscore(primary["trace"]) + sign * zscore(ghost["trace"])
        fused_heart_trace = zscore(primary["heart_trace"]) + sign * zscore(ghost["heart_trace"])
    fused_rr_hz, fused_rr_score, _, _ = spectral_peak(
        fft_bandpass(fused_trace, RADAR_FS, RR_LOW, RR_HIGH), RADAR_FS, RR_LOW, RR_HIGH, 8192
    )
    fused_hr_hz, fused_hr_score, _, _, fused_hr_persistence = robust_segment_spectrum(
        fused_heart_trace, RADAR_FS, HR_LOW, HR_HIGH, 8192, 12.0, 6.0
    )
    return {
        "primary_bin": primary["bin"],
        "primary_rr_bpm": primary["rr_hz"] * 60.0,
        "primary_rr_score": primary["rr_score"],
        "primary_hr_bpm": primary["hr_hz"] * 60.0 if finite(primary["hr_hz"]) else math.nan,
        "primary_hr_score": primary["hr_score"],
        "primary_hr_persistence": primary["hr_persistence"],
        "amplitude": primary["amplitude"],
        "phase_bad_ratio": primary["phase_bad"],
        "quality": primary["locate_score"],
        "ghost_bin": ghost["bin"] if ghost else None,
        "ghost_correlation": ghost["correlation"] if ghost else math.nan,
        "fused_rr_bpm": fused_rr_hz * 60.0 if finite(fused_rr_hz) else math.nan,
        "fused_rr_score": fused_rr_score,
        "fused_hr_bpm": fused_hr_hz * 60.0 if finite(fused_hr_hz) else math.nan,
        "fused_hr_score": fused_hr_score,
        "fused_hr_persistence": fused_hr_persistence,
        "motion_index": motion_index(data),
        "primary_trace": primary["trace"],
        "fused_trace": fused_trace,
    }


def analyze_radar(subject, radar, start_s, end_s):
    data = load_radar_interval(subject, radar, start_s, end_s)
    return analyze_radar_data(data)


def orientation_segments(row):
    return [
        ("정면-레이더1", 1, row["s01_content_start_s"] + TRIM_S, row["turn1_s"] - TRIM_S),
        ("정면-레이더2", 2, row["turn1_s"] + TRIM_S, row["turn2_s"] - TRIM_S),
        ("정면-레이더3", 3, row["turn2_s"] + TRIM_S, row["m02_s"] - TRIM_S),
    ]


def relative_angle(facing_radar, radar):
    return (radar - facing_radar) * 45


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_errors(rows, estimate_key, gt_key, threshold):
    pairs = [(float(r[estimate_key]), float(r[gt_key])) for r in rows if finite(r.get(estimate_key)) and finite(r.get(gt_key))]
    if not pairs:
        return {"n": 0, "mae": math.nan, "median_ae": math.nan, "within": math.nan}
    ae = np.asarray([abs(a - b) for a, b in pairs])
    return {"n": len(ae), "mae": float(np.mean(ae)), "median_ae": float(np.median(ae)), "within": float(np.mean(ae <= threshold))}


def aggregate_fusion(observations):
    grouped = defaultdict(list)
    for row in observations:
        grouped[(row["subject"], row["orientation"])].append(row)
    out = []
    for (subject, orientation), rows in grouped.items():
        rows.sort(key=lambda r: int(r["radar"]))
        gt_rr = rows[0]["rsp_rr_bpm"]
        gt_hr = rows[0]["ecg_hr_bpm"]
        expected = int(rows[0]["facing_radar"])
        qualities = np.asarray([max(1e-9, float(r["quality"])) for r in rows])
        weights = np.log1p(qualities)
        weights = weights / np.sum(weights)
        rr = np.asarray([float(r["fused_rr_bpm"]) for r in rows])
        hr = np.asarray([float(r["fused_hr_bpm"]) for r in rows])
        best_idx = int(np.argmax(qualities))
        expected_idx = [int(r["radar"]) for r in rows].index(expected)
        methods = {
            "레이더1 고정": (rr[0], hr[0]),
            "레이더2 고정": (rr[1], hr[1]),
            "레이더3 고정": (rr[2], hr[2]),
            "명목상 정면 레이더": (rr[expected_idx], hr[expected_idx]),
            "품질 최고 레이더": (rr[best_idx], hr[best_idx]),
            "3레이더 중앙값": (float(np.nanmedian(rr)), float(np.nanmedian(hr))),
            "품질 가중 평균": (float(np.nansum(rr * weights)), float(np.nansum(hr * weights))),
        }
        for method, (rr_est, hr_est) in methods.items():
            out.append({
                "subject": subject,
                "orientation": orientation,
                "method": method,
                "rr_est_bpm": rr_est,
                "rsp_rr_bpm": gt_rr,
                "rr_abs_error": abs(rr_est - gt_rr) if finite(rr_est) and finite(gt_rr) else math.nan,
                "hr_est_bpm": hr_est,
                "ecg_hr_bpm": gt_hr,
                "hr_abs_error": abs(hr_est - gt_hr) if finite(hr_est) and finite(gt_hr) else math.nan,
                "selected_radar": int(rows[best_idx]["radar"]) if method == "품질 최고 레이더" else expected if method == "명목상 정면 레이더" else "",
            })
    return out


def method_metrics(fusion_rows):
    methods = sorted({r["method"] for r in fusion_rows})
    out = []
    for method in methods:
        rows = [r for r in fusion_rows if r["method"] == method]
        rr = summarize_errors(rows, "rr_est_bpm", "rsp_rr_bpm", 2.0)
        hr = summarize_errors(rows, "hr_est_bpm", "ecg_hr_bpm", 5.0)
        out.append({
            "method": method,
            "n_segments": len(rows),
            "rr_mae_bpm": rr["mae"],
            "rr_median_ae_bpm": rr["median_ae"],
            "rr_within_2_ratio": rr["within"],
            "hr_mae_bpm": hr["mae"],
            "hr_median_ae_bpm": hr["median_ae"],
            "hr_within_5_ratio": hr["within"],
        })
    return out


def angle_metrics(observations):
    out = []
    for angle in (-90, -45, 0, 45, 90):
        rows = [r for r in observations if int(r["relative_angle_deg"]) == angle]
        rr_primary = summarize_errors(rows, "primary_rr_bpm", "rsp_rr_bpm", 2.0)
        rr_fused = summarize_errors(rows, "fused_rr_bpm", "rsp_rr_bpm", 2.0)
        hr_primary = summarize_errors(rows, "primary_hr_bpm", "ecg_hr_bpm", 5.0)
        hr_fused = summarize_errors(rows, "fused_hr_bpm", "ecg_hr_bpm", 5.0)
        out.append({
            "relative_angle_deg": angle,
            "n": len(rows),
            "median_quality": float(np.median([float(r["quality"]) for r in rows])) if rows else math.nan,
            "primary_rr_mae": rr_primary["mae"],
            "path_fused_rr_mae": rr_fused["mae"],
            "primary_hr_mae": hr_primary["mae"],
            "path_fused_hr_mae": hr_fused["mae"],
            "ghost_candidate_ratio": float(np.mean([bool(r["ghost_bin"] != "" and r["ghost_bin"] is not None) for r in rows])) if rows else math.nan,
        })
    return out


def s2_states(row):
    start = row["s02_content_start_s"]
    resume = row["apnea_resume_s"]
    return [
        ("평소 호흡", start + 5, start + 55),
        ("느린 호흡", start + 65, start + 115),
        ("숨 참기", row["apnea_start_s"] + 2, resume - 1),
        ("숨참기 후 호흡", resume + 5, resume + 55),
        ("운동 준비", resume + 60, resume + 75),
        ("스쿼트", resume + 75, resume + 135),
        ("의자 착석", resume + 135, resume + 150),
        ("운동 후 호흡", resume + 155, resume + 205),
    ]


def marker_like_episodes(rsp, fs, threshold=7.5):
    mask = np.asarray(rsp) >= threshold
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    return int(len(starts))


def analyze_s2(boundaries):
    rows_out = []
    marker_checks = []
    audit_path = WORKSPACE / "audit_build" / "marker_audit_data.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_by_subject = {r["subject"]: r for r in audit["records"]}
    for row in boundaries:
        subject = row["subject"]
        rsp, ecg, bio_fs, _, _ = load_biopac(subject)
        offset = sync_offset(subject)
        audit_row = audit_by_subject[subject]
        internal_candidates = [
            t for t in audit_row.get("candidate_radar", [])
            if row["m03_s"] + 3 < float(t) < row["m04_s"] - 3
        ]
        internal_selected = [
            t for t in audit_row.get("selected_markers", [])
            if row["m03_s"] + 3 < float(t) < row["m04_s"] - 3
        ]
        marker_checks.append({
            "subject": subject,
            "s02_marker_interval_s": row["s02_duration_s"],
            "ppt_title_duration_s": 270,
            "ppt_detailed_duration_s": 360,
            "audit_threshold": audit_row.get("audit_threshold"),
            "internal_raw_candidates": len(internal_candidates),
            "internal_raw_candidate_times_s": ";".join(f"{float(t):.3f}" for t in internal_candidates),
            "internal_selected_markers": len(internal_selected),
            "interpretation": "내부 후보가 있어 운동 오인 가능성-원파형 검토 필요" if internal_candidates else "감사장부 기준 스쿼트 구간을 마커로 검출한 증거 없음",
        })
        for state, start_s, end_s in s2_states(row):
            if end_s <= start_s + 2 or end_s > row["m04_s"] + 1:
                continue
            rsp_seg = bio_interval(rsp, bio_fs, offset, start_s, end_s)
            ecg_seg = bio_interval(ecg, bio_fs, offset, start_s, end_s)
            ref = rate_reference(rsp_seg, ecg_seg, bio_fs)
            for radar in (1, 2, 3):
                result = analyze_radar(subject, radar, start_s, end_s)
                rows_out.append({
                    "subject": subject,
                    "state": state,
                    "radar": radar,
                    "start_s": start_s,
                    "end_s": end_s,
                    "duration_s": end_s - start_s,
                    "rsp_rr_bpm": ref["rr_bpm"],
                    "ecg_hr_bpm": ref["hr_bpm"],
                    "radar_rr_bpm": result["fused_rr_bpm"],
                    "radar_hr_bpm": result["fused_hr_bpm"],
                    "rr_abs_error": abs(result["fused_rr_bpm"] - ref["rr_bpm"]) if finite(ref["rr_bpm"]) else math.nan,
                    "hr_abs_error": abs(result["fused_hr_bpm"] - ref["hr_bpm"]) if finite(ref["hr_bpm"]) else math.nan,
                    "quality": result["quality"],
                    "motion_index": result["motion_index"],
                    "primary_bin": result["primary_bin"],
                    "ghost_bin": result["ghost_bin"] if result["ghost_bin"] is not None else "",
                })
    return rows_out, marker_checks


def analyze_roundtrip(subjects):
    with MANIFEST_CSV.open(encoding="utf-8-sig", newline="") as handle:
        manifest = [r for r in csv.DictReader(handle) if r["subject"] in subjects and r["scenario_class"] == "6"]
    out = []
    for item in manifest:
        start_s, end_s = float(item["start_s"]), float(item["end_s"])
        for radar in (1, 2, 3):
            data = load_radar_interval(item["subject"], radar, start_s, end_s)
            amp = np.abs(data)
            weights = amp - np.percentile(amp, 35, axis=1, keepdims=True)
            weights = np.maximum(weights, 0)
            bins = np.arange(amp.shape[1])[None, :]
            centroid = np.sum(weights * bins, axis=1) / (np.sum(weights, axis=1) + 1e-9)
            smooth = moving_average(centroid, int(RADAR_FS * 0.5))
            excursion = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
            out.append({
                "subject": item["subject"],
                "radar": radar,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "range_centroid_excursion_bins": excursion,
                "motion_index": motion_index(data),
                "interpretation": "왕복 움직임 존재를 보여주는 예비 지표이며 실제 거리 환산은 미보정",
            })
    return out


def inventory():
    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("S"))
    mats = list(DATA_ROOT.rglob("*.mat"))
    dats = list(DATA_ROOT.rglob("*.dat"))
    keys = ("env", "background", "empty", "nohuman", "환경", "빈")
    env = [str(p.relative_to(ROOT)) for p in DATA_ROOT.rglob("*") if p.is_file() and any(k in p.name.lower() for k in keys)]
    s01 = []
    for p in sorted((DATA_ROOT / "S01_CMS").rglob("xethru_datafloat_*.dat")):
        frames = p.stat().st_size // 4 // 185
        s01.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "frames": frames, "duration_s": frames / RADAR_FS})
    return {
        "subject_folders": len(subjects),
        "mat_files": len(mats),
        "dat_files": len(dats),
        "explicit_environment_files": env,
        "s24_uwb_missing": not any((DATA_ROOT / "S24_KHJ").glob("[123]")),
        "s01_recordings": s01,
        "environment_conclusion": "명시적으로 식별 가능한 빈 환경/무인 녹화 파일이 현재 다운로드에 없음",
    }


def bar_chart(path, title, labels, series, ylabel, note=""):
    width, height = 1500, 850
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((70, 35), title, font=fnt(34, True), fill="#102A43")
    if note:
        d.text((70, 85), note, font=fnt(18), fill="#52606D")
    left, top, right, bottom = 120, 145, 1440, 700
    values = [float(v) for vals in series.values() for v in vals if finite(v)]
    ymax = max(values) * 1.18 if values else 1.0
    d.line((left, top, left, bottom), fill="#7B8794", width=2)
    d.line((left, bottom, right, bottom), fill="#7B8794", width=2)
    colors = ["#2F80ED", "#27AE60", "#F2994A", "#9B51E0"]
    nseries = len(series)
    group_w = (right - left) / max(1, len(labels))
    bar_w = min(72, group_w * 0.7 / max(1, nseries))
    for tick in range(6):
        val = ymax * tick / 5
        y = bottom - (bottom - top) * tick / 5
        d.line((left, y, right, y), fill="#E4E7EB", width=1)
        d.text((25, y - 12), f"{val:.1f}", font=fnt(16), fill="#52606D")
    for si, (name, vals) in enumerate(series.items()):
        color = colors[si % len(colors)]
        for i, value in enumerate(vals):
            if not finite(value):
                continue
            x0 = left + i * group_w + group_w * 0.15 + si * bar_w
            x1 = x0 + bar_w * 0.9
            y = bottom - (bottom - top) * float(value) / ymax
            d.rectangle((x0, y, x1, bottom), fill=color)
            d.text((x0, y - 22), f"{float(value):.1f}", font=fnt(14), fill="#243B53")
        lx = 120 + si * 280
        d.rectangle((lx, 760, lx + 24, 784), fill=color)
        d.text((lx + 34, 758), name, font=fnt(17), fill="#243B53")
    for i, label in enumerate(labels):
        x = left + (i + 0.5) * group_w
        d.text((x - 55, bottom + 20), str(label), font=fnt(16), fill="#243B53")
    d.text((20, 120), ylabel, font=fnt(17, True), fill="#243B53")
    img.save(path)


def line_chart(path, title, series, note=""):
    width, height = 1500, 800
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((65, 30), title, font=fnt(34, True), fill="#102A43")
    if note:
        d.text((65, 78), note, font=fnt(18), fill="#52606D")
    left, top, right, bottom = 100, 135, 1440, 690
    d.rectangle((left, top, right, bottom), outline="#BCCCDC", width=2)
    colors = ["#D64545", "#2F80ED", "#27AE60"]
    all_values = np.concatenate([np.asarray(v, dtype=float) for v in series.values()])
    lo, hi = np.percentile(all_values, [2, 98])
    if hi <= lo:
        hi = lo + 1
    for idx, (name, values) in enumerate(series.items()):
        values = np.asarray(values, dtype=float)
        step = max(1, len(values) // 1200)
        values = values[::step]
        pts = []
        for i, val in enumerate(values):
            x = left + (right - left) * i / max(1, len(values) - 1)
            y = bottom - (bottom - top) * np.clip((val - lo) / (hi - lo), 0, 1)
            pts.append((x, y))
        d.line(pts, fill=colors[idx % len(colors)], width=2)
        lx = 120 + idx * 330
        d.line((lx, 745, lx + 35, 745), fill=colors[idx % len(colors)], width=5)
        d.text((lx + 45, 732), name, font=fnt(18), fill="#243B53")
    img.save(path)


def build_figures(observations, angle_rows, method_rows, s2_rows):
    bar_chart(
        OUT / "figure_1_angle_rr_error.png",
        "상대 각도에 따른 호흡수 오차",
        [f"{r['relative_angle_deg']}°" for r in angle_rows],
        {
            "단일 주 경로 후보": [r["primary_rr_mae"] for r in angle_rows],
            "경로 후보 융합": [r["path_fused_rr_mae"] for r in angle_rows],
        },
        "MAE (회/분)",
        "10명 × 3자세 × 3레이더, BIOPAC RSP 기준",
    )
    selected = [r for r in method_rows if r["method"] in {"명목상 정면 레이더", "품질 최고 레이더", "3레이더 중앙값", "품질 가중 평균"}]
    bar_chart(
        OUT / "figure_2_fusion_comparison.png",
        "3레이더 선택·융합 방식 비교",
        [r["method"] for r in selected],
        {
            "호흡수 MAE": [r["rr_mae_bpm"] for r in selected],
            "심박수 MAE": [r["hr_mae_bpm"] for r in selected],
        },
        "MAE (회/분 또는 bpm)",
        "심박 결과는 탐색적 기준선이며 독립 검증 모델이 아님",
    )
    states = ["평소 호흡", "느린 호흡", "숨 참기", "숨참기 후 호흡", "운동 준비", "스쿼트", "의자 착석", "운동 후 호흡"]
    med = []
    for state in states:
        vals = [float(r["motion_index"]) for r in s2_rows if r["state"] == state and finite(r["motion_index"])]
        med.append(float(np.median(vals)) if vals else math.nan)
    bar_chart(
        OUT / "figure_3_s2_motion.png",
        "실험 2 단계별 레이더 움직임 지표",
        states,
        {"3레이더 중앙값": med},
        "정규화 변화량",
        "스쿼트·착석은 생체신호 추정보다 움직임 오염 구간으로 취급",
    )

    diagnostic_rows = sorted(
        observations,
        key=lambda r: abs(float(r["fused_rr_bpm"]) - float(r["rsp_rr_bpm"])) if finite(r["rsp_rr_bpm"]) else 1e9,
    )
    for name, row in (("best", diagnostic_rows[0]), ("worst", diagnostic_rows[-1])):
        subject = row["subject"]
        radar = int(row["radar"])
        start_s, end_s = float(row["start_s"]), float(row["end_s"])
        rsp, _, bio_fs, _, _ = load_biopac(subject)
        offset = sync_offset(subject)
        rsp_seg = bio_interval(rsp, bio_fs, offset, start_s, end_s)
        data = load_radar_interval(subject, radar, start_s, end_s)
        result = analyze_radar_data(data)
        target_len = min(len(result["primary_trace"]), 1600)
        radar_primary = zscore(fft_bandpass(result["primary_trace"], RADAR_FS, RR_LOW, RR_HIGH))[:target_len]
        radar_fused = zscore(fft_bandpass(result["fused_trace"], RADAR_FS, RR_LOW, RR_HIGH))[:target_len]
        rsp_down = np.interp(np.linspace(0, len(rsp_seg) - 1, target_len), np.arange(len(rsp_seg)), zscore(rsp_seg))
        line_chart(
            OUT / f"figure_4_{name}_waveform.png",
            f"{subject} {row['orientation']} radar{radar} - {'대표 양호' if name == 'best' else '대표 실패'} 사례",
            {"BIOPAC RSP": rsp_down, "주 경로 후보": radar_primary, "경로 후보 융합": radar_fused},
            f"RSP {float(row['rsp_rr_bpm']):.1f}, 주경로 {float(row['primary_rr_bpm']):.1f}, 융합 {float(row['fused_rr_bpm']):.1f} 회/분",
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    boundaries = load_boundaries()
    observations = []
    for row in boundaries:
        subject = row["subject"]
        rsp, ecg, bio_fs, _, _ = load_biopac(subject)
        offset = sync_offset(subject)
        for orientation, facing_radar, start_s, end_s in orientation_segments(row):
            if end_s <= start_s + 20:
                continue
            rsp_seg = bio_interval(rsp, bio_fs, offset, start_s, end_s)
            ecg_seg = bio_interval(ecg, bio_fs, offset, start_s, end_s)
            ref = rate_reference(rsp_seg, ecg_seg, bio_fs)
            for radar in (1, 2, 3):
                result = analyze_radar(subject, radar, start_s, end_s)
                observations.append({
                    "subject": subject,
                    "orientation": orientation,
                    "facing_radar": facing_radar,
                    "radar": radar,
                    "relative_angle_deg": relative_angle(facing_radar, radar),
                    "start_s": start_s,
                    "end_s": end_s,
                    "duration_s": end_s - start_s,
                    "sync_offset_s": offset,
                    "rsp_rr_bpm": ref["rr_bpm"],
                    "rsp_rr_score": ref["rr_score"],
                    "ecg_hr_bpm": ref["hr_bpm"],
                    "ecg_quality": ref["hr_quality"],
                    "primary_bin": result["primary_bin"],
                    "ghost_bin": result["ghost_bin"] if result["ghost_bin"] is not None else "",
                    "ghost_correlation": result["ghost_correlation"],
                    "primary_rr_bpm": result["primary_rr_bpm"],
                    "fused_rr_bpm": result["fused_rr_bpm"],
                    "primary_hr_bpm": result["primary_hr_bpm"],
                    "fused_hr_bpm": result["fused_hr_bpm"],
                    "primary_rr_abs_error": abs(result["primary_rr_bpm"] - ref["rr_bpm"]),
                    "fused_rr_abs_error": abs(result["fused_rr_bpm"] - ref["rr_bpm"]),
                    "primary_hr_abs_error": abs(result["primary_hr_bpm"] - ref["hr_bpm"]),
                    "fused_hr_abs_error": abs(result["fused_hr_bpm"] - ref["hr_bpm"]),
                    "primary_rr_score": result["primary_rr_score"],
                    "fused_rr_score": result["fused_rr_score"],
                    "quality": result["quality"],
                    "motion_index": result["motion_index"],
                    "phase_bad_ratio": result["phase_bad_ratio"],
                })
    fusion = aggregate_fusion(observations)
    methods = method_metrics(fusion)
    angles = angle_metrics(observations)
    s2_rows, marker_checks = analyze_s2(boundaries)
    roundtrip = analyze_roundtrip({r["subject"] for r in boundaries})
    inv = inventory()

    write_csv(OUT / "s1_orientation_radar_observations.csv", observations)
    write_csv(OUT / "s1_fusion_predictions.csv", fusion)
    write_csv(OUT / "s1_method_metrics.csv", methods)
    write_csv(OUT / "s1_angle_metrics.csv", angles)
    write_csv(OUT / "s2_state_quality.csv", s2_rows)
    write_csv(OUT / "s2_marker_confusion_check.csv", marker_checks)
    write_csv(OUT / "s6_roundtrip_preliminary.csv", roundtrip)
    with (OUT / "inventory.json").open("w", encoding="utf-8") as handle:
        json.dump(inv, handle, ensure_ascii=False, indent=2)

    overall_primary_rr = summarize_errors(observations, "primary_rr_bpm", "rsp_rr_bpm", 2.0)
    overall_fused_rr = summarize_errors(observations, "fused_rr_bpm", "rsp_rr_bpm", 2.0)
    overall_primary_hr = summarize_errors(observations, "primary_hr_bpm", "ecg_hr_bpm", 5.0)
    overall_fused_hr = summarize_errors(observations, "fused_hr_bpm", "ecg_hr_bpm", 5.0)
    summary = {
        "analysis_subjects": [r["subject"] for r in boundaries],
        "n_subjects": len(boundaries),
        "n_s1_radar_observations": len(observations),
        "n_s1_orientation_segments": len(fusion) // 7,
        "s1_primary_rr": overall_primary_rr,
        "s1_path_fused_rr": overall_fused_rr,
        "s1_primary_hr": overall_primary_hr,
        "s1_path_fused_hr": overall_fused_hr,
        "method_metrics": methods,
        "angle_metrics": angles,
        "s2_duration_issue": {
            "ppt_title_s": 270,
            "ppt_detail_sum_s": 360,
            "actual_marker_interval_median_s": float(np.median([r["s02_marker_interval_s"] for r in marker_checks])),
            "conclusion": "PPT 제목의 4분30초는 세부 절차 및 실제 마커 간격과 불일치",
        },
        "s2_internal_marker_like_subjects": [r["subject"] for r in marker_checks if r["internal_raw_candidates"] > 0],
        "environment_file_available": bool(inv["explicit_environment_files"]),
        "limitations": [
            "빈 환경 녹화 파일을 명시적으로 식별할 수 없어 환경 차감은 실행하지 못함",
            "멀티패스 후보는 주 호흡 주파수와 상관이 높은 분리 거리 bin이며 물리적 반사 경로로 확정한 것이 아님",
            "심박 추정은 규칙 기반 탐색 결과이고 독립 학습/검증을 거친 최종 모델이 아님",
            "신뢰 경계가 확보된 10명만 우선 분석했으므로 전체 30명 일반화 결과가 아님",
            "왕복 구간의 range-bin 변화는 실제 거리 보정 전 예비 지표임",
        ],
    }
    with (OUT / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    build_figures(observations, angles, methods, s2_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
