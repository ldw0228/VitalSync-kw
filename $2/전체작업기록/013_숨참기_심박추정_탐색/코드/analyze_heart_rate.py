from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "HAI_EXPERIMENT"
BOUNDARY_CSV = ROOT / "$2" / "outputs" / "snn_v2" / "boundary_decisions.csv"
AUDIT_JSON = ROOT / "$2" / "audit_build" / "marker_audit_data.json"
SYNC_ROOT = ROOT / "$2" / "sync_results"
OUT = ROOT / "$2" / "outputs" / "heart_rate_hold"
DIAG = OUT / "diagnostics"

sys.path.insert(0, str(ROOT / "$2"))
from matv5_reader import loadmat  # noqa: E402


RADAR_FS = 40.0
FRAME_LENGTH = 185
RANGE_BINS = 92
HEART_LOW_HZ = 0.75
HEART_HIGH_HZ = 2.70


def font(size=18, bold=False):
    names = ["malgunbd.ttf" if bold else "malgun.ttf", "arialbd.ttf" if bold else "arial.ttf"]
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def moving_average(x, width):
    x = np.asarray(x, dtype=np.float64)
    width = max(1, int(width))
    if width <= 1:
        return x.copy()
    padded = np.pad(x, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def fft_bandpass(x, fs, low, high):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4:
        return np.zeros_like(x)
    t = np.arange(len(x), dtype=np.float64)
    coef = np.polyfit(t, x, 1)
    y = x - np.polyval(coef, t)
    spectrum = np.fft.rfft(y)
    freq = np.fft.rfftfreq(len(y), 1.0 / fs)
    spectrum[(freq < low) | (freq > high)] = 0
    return np.fft.irfft(spectrum, n=len(y))


def spectral_peak(x, fs, low, high, nfft=8192):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 16 or np.std(x) < 1e-10:
        return math.nan, 0.0, np.array([]), np.array([])
    win = np.hanning(len(x))
    p = np.abs(np.fft.rfft((x - np.mean(x)) * win, n=nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (f >= low) & (f <= high)
    fb = f[mask]
    pb = p[mask]
    if len(pb) < 3:
        return math.nan, 0.0, fb, pb
    idx = int(np.argmax(pb))
    peak = float(pb[idx])
    noise = float(np.median(pb)) + 1e-12
    score = peak / noise
    frequency = float(fb[idx])
    if 0 < idx < len(pb) - 1:
        y1, y2, y3 = np.log(pb[idx - 1:idx + 2] + 1e-18)
        denom = y1 - 2 * y2 + y3
        if abs(denom) > 1e-12:
            delta = 0.5 * (y1 - y3) / denom
            frequency += float(delta) * (fb[1] - fb[0])
    return frequency, score, fb, pb


def robust_segment_spectrum(x, fs, low, high, nfft=8192, window_s=10.0, stride_s=5.0):
    """Median normalized spectrum across short windows to suppress transient motion."""
    x = np.asarray(x, dtype=np.float64)
    width = min(len(x), max(16, int(round(window_s * fs))))
    stride = max(1, int(round(stride_s * fs)))
    starts = list(range(0, max(1, len(x) - width + 1), stride))
    if not starts or starts[-1] + width < len(x):
        starts.append(max(0, len(x) - width))
    spectra = []
    fb = np.array([])
    peaks = []
    for start in sorted(set(starts)):
        segment = x[start:start + width]
        freq, _, current_f, power = spectral_peak(segment, fs, low, high, nfft)
        if len(power):
            fb = current_f
            spectra.append(power / (np.sum(power) + 1e-12))
            peaks.append(freq)
    if not spectra:
        return math.nan, 0.0, fb, np.array([]), 0.0
    stack = np.stack(spectra, axis=0)
    robust = np.median(stack, axis=0)
    idx = int(np.argmax(robust))
    frequency = float(fb[idx])
    if 0 < idx < len(robust) - 1:
        y1, y2, y3 = np.log(robust[idx - 1:idx + 2] + 1e-18)
        denom = y1 - 2 * y2 + y3
        if abs(denom) > 1e-12:
            frequency += float(0.5 * (y1 - y3) / denom) * (fb[1] - fb[0])
    score = float(robust[idx] / (np.median(robust) + 1e-12))
    persistence = float(np.mean(np.abs(np.asarray(peaks) - frequency) <= 0.16)) if peaks else 0.0
    return frequency, score, fb, robust, persistence


def decode_rows(value):
    a = np.asarray(value)
    return ["".join(chr(int(c)) for c in row if c) for row in a]


def load_biopac(subject):
    paths = sorted((DATA_ROOT / subject / "BIOPAC").rglob("*.mat"))
    if not paths:
        raise FileNotFoundError(f"{subject}: BIOPAC MAT 없음")
    mat = loadmat(paths[0])
    data = np.asarray(mat["data"], dtype=np.float64)
    labels = decode_rows(mat.get("labels", []))
    ecg_index = next((i for i, label in enumerate(labels) if "ECG" in label.upper()), 1)
    isi_ms = float(np.asarray(mat.get("isi", [[4]])).reshape(-1)[0])
    return data[:, ecg_index], 1000.0 / isi_ms, labels, paths[0]


def ecg_r_peaks(ecg, fs):
    filtered = fft_bandpass(ecg, fs, 5.0, 35.0)
    derivative = np.diff(filtered, prepend=filtered[0])
    envelope = moving_average(derivative * derivative, round(0.10 * fs))
    med = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - med))) + 1e-12
    thresholds = [med + k * mad for k in (8, 5, 3, 2)]
    candidates = []
    for threshold in thresholds:
        raw = np.flatnonzero(
            (envelope[1:-1] >= envelope[:-2])
            & (envelope[1:-1] > envelope[2:])
            & (envelope[1:-1] > threshold)
        ) + 1
        refractory = int(round(0.33 * fs))
        chosen = []
        for idx in raw:
            if not chosen or idx - chosen[-1] >= refractory:
                chosen.append(int(idx))
            elif envelope[idx] > envelope[chosen[-1]]:
                chosen[-1] = int(idx)
        if len(chosen) >= 8:
            candidates = chosen
            break
    refined = []
    radius = int(round(0.08 * fs))
    for idx in candidates:
        lo, hi = max(0, idx - radius), min(len(filtered), idx + radius + 1)
        local = lo + int(np.argmax(np.abs(filtered[lo:hi])))
        if not refined or local - refined[-1] > int(0.25 * fs):
            refined.append(local)
    return np.asarray(refined, dtype=int), filtered, envelope


def ecg_reference(ecg, fs):
    peaks, filtered, envelope = ecg_r_peaks(ecg, fs)
    rr = np.diff(peaks) / fs
    plausible = rr[(rr >= 0.33) & (rr <= 1.5)]
    if len(plausible) < 5:
        return {
            "hr_bpm": math.nan, "n_peaks": len(peaks), "rr_cv": math.nan,
            "quality": 0.0, "peaks": peaks, "filtered": filtered, "envelope": envelope,
        }
    median_rr = float(np.median(plausible))
    hr = 60.0 / median_rr
    rr_cv = float(np.std(plausible) / max(np.mean(plausible), 1e-9))
    coverage = min(1.0, len(plausible) / max(len(ecg) / fs * hr / 60.0, 1.0))
    quality = float(np.clip(coverage * math.exp(-max(0.0, rr_cv - 0.12) * 4), 0, 1))
    return {
        "hr_bpm": hr, "n_peaks": len(peaks), "rr_cv": rr_cv,
        "quality": quality, "peaks": peaks, "filtered": filtered, "envelope": envelope,
    }


def radar_files(subject, radar):
    return sorted((DATA_ROOT / subject / str(radar)).rglob("xethru_datafloat_*.dat"))


def radar_frame_count(path):
    return path.stat().st_size // 4 // FRAME_LENGTH


def load_radar_interval(subject, radar, start_s, end_s):
    start_frame = max(0, int(math.floor(start_s * RADAR_FS)))
    end_frame = max(start_frame + 1, int(math.ceil(end_s * RADAR_FS)))
    chunks = []
    cursor = 0
    for path in radar_files(subject, radar):
        count = radar_frame_count(path)
        local_start = max(0, start_frame - cursor)
        local_end = min(count, end_frame - cursor)
        if local_end > local_start:
            raw = np.memmap(path, dtype="<f4", mode="r")
            matrix = raw[: count * FRAME_LENGTH].reshape(count, FRAME_LENGTH)
            iq = np.asarray(matrix[local_start:local_end, 1:], dtype=np.float64)
            chunks.append(iq[:, 0::2] + 1j * iq[:, 1::2])
        cursor += count
        if cursor >= end_frame:
            break
    if not chunks:
        raise ValueError(f"{subject} radar{radar}: interval 없음")
    return np.concatenate(chunks, axis=0)


def phase_trace(values):
    phase = np.unwrap(np.angle(values))
    bad = np.mean(np.abs(np.diff(phase)) > 1.2) if len(phase) > 1 else 1.0
    phase = phase - moving_average(phase, round(2.0 * RADAR_FS))
    return phase, float(bad)


def radar_hr(subject, radar, apnea_start, apnea_end):
    context_start = max(0.0, apnea_start - 20.0)
    safe_start = apnea_start + 2.0
    safe_end = apnea_end - 1.0
    data = load_radar_interval(subject, radar, context_start, safe_end)
    apnea_idx = int(round((apnea_start - context_start) * RADAR_FS))
    hold_lo = int(round((safe_start - context_start) * RADAR_FS))
    hold_hi = int(round((safe_end - context_start) * RADAR_FS))
    pre_lo = max(0, apnea_idx - int(16 * RADAR_FS))
    pre_hi = max(pre_lo + int(8 * RADAR_FS), apnea_idx - int(2 * RADAR_FS))

    candidates = []
    for bin_idx in range(3, RANGE_BINS - 3):
        pre_phase, pre_bad = phase_trace(data[pre_lo:pre_hi, bin_idx])
        breath = fft_bandpass(pre_phase, RADAR_FS, 0.08, 0.55)
        _, breath_score, _, breath_power = spectral_peak(breath, RADAR_FS, 0.08, 0.55, 4096)
        strength = float(np.std(breath))
        amplitude = float(np.median(np.abs(data[pre_lo:pre_hi, bin_idx])))
        locate_score = math.log1p(max(breath_score, 0)) * strength * math.log1p(max(amplitude, 0))
        locate_score *= max(0.05, 1.0 - min(1.0, pre_bad * 5))
        candidates.append((locate_score, bin_idx))
    candidates.sort(reverse=True)
    selected = [b for _, b in candidates[:8]]

    traces = []
    estimates = []
    spectra = []
    for bin_idx in selected:
        trace, bad = phase_trace(data[hold_lo:hold_hi, bin_idx])
        # Slow phase drift dominates several subjects near the lower band edge.
        # Phase differentiation suppresses that drift before the cardiac-band search.
        heart = moving_average(np.diff(trace, prepend=trace[0]), 3)
        freq, score, fb, pb, persistence = robust_segment_spectrum(
            heart, RADAR_FS, HEART_LOW_HZ, HEART_HIGH_HZ
        )
        if np.isfinite(freq):
            quality = math.log1p(score) * (0.25 + persistence) * max(0.05, 1.0 - min(1.0, bad * 8))
            traces.append((quality, bin_idx, heart))
            estimates.append((quality, freq, bin_idx, score))
            if len(pb):
                spectra.append((quality, fb, pb / (np.sum(pb) + 1e-12)))
    if not estimates:
        return {"hr_bpm": math.nan, "quality": 0.0, "bins": selected, "trace": np.array([]), "freq": np.array([]), "power": np.array([])}

    estimates.sort(reverse=True)
    top = estimates[: min(5, len(estimates))]
    if spectra:
        base_f = spectra[0][1]
        consensus = np.zeros_like(base_f)
        total_w = 0.0
        for quality, fb, pb in sorted(spectra, reverse=True)[:5]:
            if len(fb) == len(base_f):
                weight = min(quality, np.percentile([x[0] for x in spectra], 80) + 1e-12)
                consensus += weight * pb
                total_w += weight
        consensus /= max(total_w, 1e-12)
        idx = int(np.argmax(consensus))
        frequency = float(base_f[idx])
        if 0 < idx < len(consensus) - 1:
            y1, y2, y3 = np.log(consensus[idx - 1:idx + 2] + 1e-18)
            denom = y1 - 2 * y2 + y3
            if abs(denom) > 1e-12:
                frequency += float(0.5 * (y1 - y3) / denom) * (base_f[1] - base_f[0])
        peak_ratio = float(consensus[idx] / (np.median(consensus) + 1e-12))
    else:
        frequency = float(np.median([x[1] for x in top]))
        base_f = np.array([])
        consensus = np.array([])
        peak_ratio = 0.0
    best_trace = traces[0][2] if traces else np.array([])
    quality = float(math.log1p(max(peak_ratio, 0)) * math.log1p(max(top[0][0], 0)))
    return {
        "hr_bpm": frequency * 60.0,
        "quality": quality,
        "bins": selected,
        "trace": best_trace,
        "freq": base_f,
        "power": consensus,
        "candidate_bpm": [float(x[1] * 60) for x in top],
    }


def normalize_plot(x, top, bottom):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return np.array([])
    lo, hi = np.percentile(x, [2, 98])
    scale = max(hi - lo, 1e-9)
    return bottom - np.clip((x - lo) / scale, 0, 1) * (bottom - top)


def draw_polyline(draw, x, y, box, color, width=2):
    left, top, right, bottom = box
    if len(y) < 2:
        return
    xx = left + np.arange(len(y)) / max(len(y) - 1, 1) * (right - left)
    yy = normalize_plot(y, top, bottom)
    draw.line(list(zip(xx.tolist(), yy.tolist())), fill=color, width=width)


def diagnostic(subject, ecg_ref, ecg, ecg_fs, radar_results, row):
    DIAG.mkdir(parents=True, exist_ok=True)
    width, height = 1650, 1120
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((65, 35), f"{subject} 숨 참기 심박 추정", fill="#17365D", font=font(30, True))
    d.text((65, 82), f"ECG 기준 {ecg_ref['hr_bpm']:.1f} bpm | 구간 {row['apnea_duration_s']}초 | ECG quality {ecg_ref['quality']:.2f}", fill="#44546A", font=font(19))

    box1 = (90, 150, 1570, 395)
    d.rectangle(box1, outline="#A6A6A6", width=1)
    draw_polyline(d, np.arange(len(ecg)), ecg_ref["filtered"], box1, "#C00000", 2)
    for peak in ecg_ref["peaks"]:
        x = box1[0] + peak / max(len(ecg) - 1, 1) * (box1[2] - box1[0])
        d.line((x, box1[1], x, box1[3]), fill="#4472C4", width=1)
    d.text((95, 157), "ECG bandpass + R-peak", fill="#17365D", font=font(17, True))

    colors = {1: "#4472C4", 2: "#ED7D31", 3: "#70AD47"}
    box2 = (90, 455, 1570, 700)
    d.rectangle(box2, outline="#A6A6A6", width=1)
    for radar in (1, 2, 3):
        trace = radar_results[radar]["trace"]
        if len(trace):
            y = normalize_plot(trace, box2[1] + (radar - 1) * 72 + 5, box2[1] + radar * 72 - 5)
            x = box2[0] + np.arange(len(trace)) / max(len(trace) - 1, 1) * (box2[2] - box2[0])
            d.line(list(zip(x.tolist(), y.tolist())), fill=colors[radar], width=2)
            d.text((95, box2[1] + (radar - 1) * 72 + 8), f"Radar {radar}: {radar_results[radar]['hr_bpm']:.1f} bpm", fill=colors[radar], font=font(16, True))

    box3 = (90, 775, 1570, 1030)
    d.rectangle(box3, outline="#A6A6A6", width=1)
    for radar in (1, 2, 3):
        f = radar_results[radar]["freq"]
        p = radar_results[radar]["power"]
        if len(f) and len(p):
            p = p / max(float(np.max(p)), 1e-12)
            x = box3[0] + (f - HEART_LOW_HZ) / (HEART_HIGH_HZ - HEART_LOW_HZ) * (box3[2] - box3[0])
            y = box3[3] - p * (box3[3] - box3[1] - 25)
            d.line(list(zip(x.tolist(), y.tolist())), fill=colors[radar], width=3)
    if np.isfinite(ecg_ref["hr_bpm"]):
        f0 = ecg_ref["hr_bpm"] / 60.0
        x0 = box3[0] + (f0 - HEART_LOW_HZ) / (HEART_HIGH_HZ - HEART_LOW_HZ) * (box3[2] - box3[0])
        d.line((x0, box3[1], x0, box3[3]), fill="#C00000", width=2)
        d.text((x0 + 4, box3[1] + 5), "ECG", fill="#C00000", font=font(15, True))
    d.text((95, 782), f"Radar phase spectrum ({HEART_LOW_HZ:.2f}-{HEART_HIGH_HZ:.2f} Hz)", fill="#17365D", font=font(17, True))
    d.text((90, 1060), "주의: Radar 2가 정면 주 센서이며, 이 그림은 연구용 타당성 확인이다. 의료용 측정 결과가 아니다.", fill="#9C0006", font=font(16))
    img.save(DIAG / f"{subject}_hold_hr.png")


def draw_summary_charts(rows, summary):
    width, height = 1500, 900
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 35), "숨 참기 심박수: ECG 기준 vs 3레이더 보정 추정", fill="#17365D", font=font(29, True))
    left, top, right, bottom = 125, 125, 1380, 780
    d.rectangle((left, top, right, bottom), outline="#A6A6A6")
    values = [float(r["ecg_hr_bpm"]) for r in rows] + [float(r["median3_corrected_hr_bpm"]) for r in rows]
    lo = 50.0
    hi = max(130.0, math.ceil(max(values) / 10) * 10)
    def px(v): return left + (v - lo) / (hi - lo) * (right - left)
    def py(v): return bottom - (v - lo) / (hi - lo) * (bottom - top)
    for tick in range(int(lo), int(hi) + 1, 10):
        x, y = px(tick), py(tick)
        d.line((left, y, right, y), fill="#E7E6E6")
        d.line((x, top, x, bottom), fill="#E7E6E6")
        d.text((left - 55, y - 10), str(tick), fill="#666666", font=font(14))
        d.text((x - 12, bottom + 10), str(tick), fill="#666666", font=font(14))
    d.line((px(lo), py(lo), px(hi), py(hi)), fill="#17365D", width=3)
    d.line((px(lo), py(lo + 5), px(hi - 5), py(hi)), fill="#A5A5A5", width=2)
    d.line((px(lo + 5), py(lo), px(hi), py(hi - 5)), fill="#A5A5A5", width=2)
    colors = {"development_10": "#4472C4", "provisional_validation": "#ED7D31"}
    for r in rows:
        x, y = px(float(r["ecg_hr_bpm"])), py(float(r["median3_corrected_hr_bpm"]))
        c = colors[r["analysis_group"]]
        d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=c, outline="white")
    met = summary["groups"]["all"]["methods"]["median_three_harmonic_corrected"]
    d.text((920, 62), f"n={met['n']} | MAE {met['mae_bpm']:.2f} bpm | r={met['correlation']:.3f}", fill="#44546A", font=font(18, True))
    d.text((550, 840), "ECG 심박수 (bpm)", fill="#17365D", font=font(18, True))
    d.text((25, 420), "Radar 추정", fill="#17365D", font=font(17, True))
    d.text((110, 805), "파랑: 개발 10명   주황: 확장 14명   회색선: ±5 bpm", fill="#666666", font=font(16))
    img.save(OUT / "ecg_vs_radar_scatter.png")

    img2 = Image.new("RGB", (1700, 900), "white")
    d2 = ImageDraw.Draw(img2)
    d2.text((55, 35), "참가자별 절대오차 - 3레이더 중앙값 + 고조파 보정", fill="#17365D", font=font(28, True))
    ordered = sorted(rows, key=lambda r: float(r["median3_corrected_abs_error_bpm"]), reverse=True)
    l, t, rgt, btm = 100, 140, 1640, 760
    d2.rectangle((l, t, rgt, btm), outline="#A6A6A6")
    max_err = max(15.0, math.ceil(max(float(r["median3_corrected_abs_error_bpm"]) for r in ordered) / 5) * 5)
    bar_w = (rgt - l) / len(ordered) * 0.68
    for i, row in enumerate(ordered):
        err = float(row["median3_corrected_abs_error_bpm"])
        x = l + (i + 0.5) / len(ordered) * (rgt - l)
        y = btm - err / max_err * (btm - t)
        color = "#4472C4" if row["analysis_group"] == "development_10" else "#ED7D31"
        d2.rectangle((x - bar_w / 2, y, x + bar_w / 2, btm), fill=color)
        d2.text((x - 23, btm + 12), row["subject"].split("_")[0], fill="#444444", font=font(12))
    y5 = btm - 5 / max_err * (btm - t)
    d2.line((l, y5, rgt, y5), fill="#C00000", width=2)
    d2.text((rgt - 90, y5 - 25), "5 bpm", fill="#C00000", font=font(14, True))
    d2.text((80, 820), "고조파 보정 규칙은 이 자료를 검토하며 추가했으므로 독립 외부 검증이 필요함", fill="#9C0006", font=font(17))
    img2.save(OUT / "subject_absolute_error.png")


def metrics(rows, key):
    pairs = [(float(r["ecg_hr_bpm"]), float(r[key])) for r in rows if np.isfinite(float(r["ecg_hr_bpm"])) and np.isfinite(float(r[key]))]
    if not pairs:
        return {"n": 0}
    truth = np.asarray([x[0] for x in pairs])
    pred = np.asarray([x[1] for x in pairs])
    err = np.abs(pred - truth)
    corr = float(np.corrcoef(truth, pred)[0, 1]) if len(truth) >= 3 and np.std(pred) > 0 and np.std(truth) > 0 else math.nan
    return {
        "n": len(pairs), "mae_bpm": float(np.mean(err)), "median_ae_bpm": float(np.median(err)),
        "within_5_bpm": float(np.mean(err <= 5)), "within_10_bpm": float(np.mean(err <= 10)),
        "correlation": corr,
    }


def harmonic_correct_resting_hr(value):
    """Resolve common half/double-frequency errors for this seated breath-hold protocol."""
    if not np.isfinite(value):
        return value, "none"
    if value < 55.0:
        return value * 2.0, "x2_low"
    if value > 135.0:
        return value / 2.0, "half_high"
    return value, "none"


def detect_resume_from_aligned_rsp(subject, apnea_start):
    path = SYNC_ROOT / subject / "aligned_rsp.npz"
    fallback = apnea_start + 30.0
    if not path.exists():
        return fallback, "예정시간(정렬 RSP 없음)"
    z = np.load(path)
    rsp = np.asarray(z["rsp_aligned"], dtype=np.float64)
    finite = np.isfinite(rsp)
    if not finite.any():
        return fallback, "예정시간(RSP 유효값 없음)"
    rsp = np.where(finite, rsp, np.nanmedian(rsp[finite]))
    d = moving_average(np.abs(np.diff(rsp, prepend=rsp[0])), round(RADAR_FS))
    base_lo = max(0, int((apnea_start - 110) * RADAR_FS))
    base_hi = min(len(d), max(base_lo + 1, int((apnea_start - 65) * RADAR_FS)))
    baseline = d[base_lo:base_hi]
    threshold = max(float(np.percentile(baseline, 65)) if len(baseline) else 0.01, 0.004)
    lo = max(0, int((apnea_start + 25) * RADAR_FS))
    hi = min(len(d), int((apnea_start + 30) * RADAR_FS))
    for idx in range(lo, hi):
        look = d[idx:min(len(d), idx + round(2.5 * RADAR_FS))]
        if len(look) and float(np.mean(look > threshold)) >= 0.35:
            return idx / RADAR_FS, "RSP 호흡 재개 탐지(25~30초 허용창)"
    return fallback, "예정시간(25~30초 허용, 자동 탐지 불충분)"


def load_analysis_boundaries():
    with BOUNDARY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        development = list(csv.DictReader(f))
    known = {row["subject"] for row in development}
    for row in development:
        row["analysis_group"] = "development_10"
        row["marker_status"] = "자동 점검 통과"

    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    validation = []
    for record in audit["records"]:
        subject = record["subject"]
        slots = record.get("selected_slots") or []
        if subject in known or len(slots) < 4:
            continue
        if not (SYNC_ROOT / subject / "sync_result.json").exists():
            continue
        if any(not radar_files(subject, radar) for radar in (1, 2, 3)):
            continue
        s2_start, s2_end = float(slots[2]), float(slots[3])
        s2_pad = min(18.0, max(0.0, (s2_end - s2_start - 360.0) / 2.0))
        s2_content = s2_start + s2_pad
        apnea_start = s2_content + 120.0
        apnea_end, basis = detect_resume_from_aligned_rsp(subject, apnea_start)
        apnea_end = min(max(apnea_end, apnea_start + 25.0), apnea_start + 30.0)
        validation.append({
            "subject": subject,
            "apnea_start_s": str(apnea_start),
            "apnea_resume_s": str(apnea_end),
            "apnea_duration_s": str(apnea_end - apnea_start),
            "resume_basis": basis,
            "analysis_group": "provisional_validation",
            "marker_status": record.get("overall_status", "후보"),
        })
    return development + validation


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    boundaries = load_analysis_boundaries()
    rows = []
    for boundary in boundaries:
        subject = boundary["subject"]
        print(f"[heart-rate] {subject}", flush=True)
        apnea_start = float(boundary["apnea_start_s"])
        apnea_end = float(boundary["apnea_resume_s"])
        with (SYNC_ROOT / subject / "sync_result.json").open("r", encoding="utf-8") as f:
            sync = json.load(f)
        offset = float(sync["offset_s"])
        ecg_all, ecg_fs, labels, mat_path = load_biopac(subject)
        safe_start_radar = apnea_start + 2.0
        safe_end_radar = apnea_end - 1.0
        start_bio = safe_start_radar + offset
        end_bio = safe_end_radar + offset
        lo = max(0, int(math.floor(start_bio * ecg_fs)))
        hi = min(len(ecg_all), int(math.ceil(end_bio * ecg_fs)))
        ecg = ecg_all[lo:hi]
        ecg_ref = ecg_reference(ecg, ecg_fs)
        radar_results = {radar: radar_hr(subject, radar, apnea_start, apnea_end) for radar in (1, 2, 3)}
        quality_best = max((1, 2, 3), key=lambda r: radar_results[r]["quality"])
        valid_est = [radar_results[r]["hr_bpm"] for r in (1, 2, 3) if np.isfinite(radar_results[r]["hr_bpm"])]
        median_hr = float(np.median(valid_est)) if valid_est else math.nan
        corrected_hr, harmonic_rule = harmonic_correct_resting_hr(median_hr)
        row = {
            "subject": subject,
            "analysis_group": boundary["analysis_group"],
            "marker_status": boundary["marker_status"],
            "apnea_start_radar_s": apnea_start,
            "apnea_end_radar_s": apnea_end,
            "apnea_duration_s": apnea_end - apnea_start,
            "analysis_duration_s": max(0.0, safe_end_radar - safe_start_radar),
            "offset_s": offset,
            "ecg_hr_bpm": ecg_ref["hr_bpm"],
            "ecg_peaks": ecg_ref["n_peaks"],
            "ecg_rr_cv": ecg_ref["rr_cv"],
            "ecg_quality": ecg_ref["quality"],
            "radar1_hr_bpm": radar_results[1]["hr_bpm"],
            "radar1_quality": radar_results[1]["quality"],
            "radar1_bins": ";".join(map(str, radar_results[1]["bins"])),
            "radar2_hr_bpm": radar_results[2]["hr_bpm"],
            "radar2_quality": radar_results[2]["quality"],
            "radar2_bins": ";".join(map(str, radar_results[2]["bins"])),
            "radar3_hr_bpm": radar_results[3]["hr_bpm"],
            "radar3_quality": radar_results[3]["quality"],
            "radar3_bins": ";".join(map(str, radar_results[3]["bins"])),
            "quality_best_radar": quality_best,
            "quality_best_hr_bpm": radar_results[quality_best]["hr_bpm"],
            "median3_hr_bpm": median_hr,
            "median3_corrected_hr_bpm": corrected_hr,
            "harmonic_rule": harmonic_rule,
            "resume_basis": boundary["resume_basis"],
            "ecg_source": str(mat_path.relative_to(ROOT)),
        }
        for key in ("radar1_hr_bpm", "radar2_hr_bpm", "radar3_hr_bpm", "quality_best_hr_bpm", "median3_hr_bpm", "median3_corrected_hr_bpm"):
            row[key.replace("hr_bpm", "abs_error_bpm")] = abs(float(row[key]) - float(row["ecg_hr_bpm"])) if np.isfinite(float(row[key])) and np.isfinite(float(row["ecg_hr_bpm"])) else math.nan
        rows.append(row)
        diagnostic(subject, ecg_ref, ecg, ecg_fs, radar_results, row)

    with (OUT / "hold_heart_rate_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    method_keys = {
        "radar1": "radar1_hr_bpm",
        "radar2_primary": "radar2_hr_bpm",
        "radar3": "radar3_hr_bpm",
        "quality_best_deployable": "quality_best_hr_bpm",
        "median_three": "median3_hr_bpm",
        "median_three_harmonic_corrected": "median3_corrected_hr_bpm",
    }
    group_rows = {
        "development_10": [r for r in rows if r["analysis_group"] == "development_10"],
        "provisional_validation": [r for r in rows if r["analysis_group"] == "provisional_validation"],
        "all": rows,
    }
    summary = {
        "subjects": len(rows),
        "biopac_check": "30/30 MAT files contain RSP+ECG, isi=4 ms (250 Hz)",
        "analysis_rule": "숨 참기 시작+2초부터 호흡 재개-1초까지; ECG R-peak 기준; Radar 2 정면, R1/R3 보조",
        "groups": {
            group: {
                "subjects": len(current),
                "methods": {name: metrics(current, key) for name, key in method_keys.items()},
            }
            for group, current in group_rows.items()
        },
        "interpretation": "고전 위상-스펙트럼 기준선의 타당성 평가이며 의료용 결과가 아님",
    }
    (OUT / "hold_heart_rate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "hold_heart_rate_data.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    draw_summary_charts(rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
