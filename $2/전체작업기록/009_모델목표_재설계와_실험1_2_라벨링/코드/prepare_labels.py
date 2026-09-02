import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "HAI_EXPERIMENT"
AUDIT_JSON = ROOT / "$2" / "audit_build" / "marker_audit_data.json"
SYNC_ROOT = ROOT / "$2" / "sync_results"
OUT = ROOT / "$2" / "outputs" / "snn_v2"
FEATURE_DIR = OUT / "subject_features"
FIG_DIR = OUT / "diagnostics"

FPS = 40
FRAME_LENGTH = 185
RANGE_BINS = 92
N_ZONES = 8
DOWNSAMPLE = 4


def font(size=18, bold=False):
    candidates = [
        Path(r"C:\Windows\Fonts\malgunbd.ttf" if bold else r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def approved_records():
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    records = [
        r for r in audit["records"]
        if r["overall_status"] == "자동 점검 통과"
        and r["selected_count"] == 22
        and len(r.get("selected_slots") or []) == 22
    ]
    return sorted(records, key=lambda r: r["number"])


def radar_files(subject, radar_index):
    return sorted((DATA_ROOT / subject / str(radar_index)).rglob("xethru_datafloat_*.dat"))


def read_motion_file(path):
    raw = np.memmap(path, dtype="<f4", mode="r")
    n_frames = raw.size // FRAME_LENGTH
    matrix = raw[: n_frames * FRAME_LENGTH].reshape(n_frames, FRAME_LENGTH)
    zones = np.array_split(np.arange(RANGE_BINS), N_ZONES)
    output = np.empty((max(n_frames - 1, 0), N_ZONES), dtype=np.float32)
    chunk_size = 4096
    prev = None
    pos = 0
    for start in range(0, n_frames, chunk_size):
        stop = min(n_frames, start + chunk_size)
        iq = np.asarray(matrix[start:stop, 1:], dtype=np.float32)
        comp = iq[:, 0::2] + 1j * iq[:, 1::2]
        if prev is not None:
            comp = np.concatenate([prev, comp], axis=0)
        if len(comp) < 2:
            prev = comp[-1:]
            continue
        delta = np.abs(np.diff(comp, axis=0))
        block = np.stack([delta[:, zone].mean(axis=1) for zone in zones], axis=1).astype(np.float32)
        output[pos:pos + len(block)] = block
        pos += len(block)
        prev = comp[-1:]
    return output[:pos]


def subject_features(subject):
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    cache = FEATURE_DIR / f"{subject}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["motion"]
    radars = []
    for radar_idx in (1, 2, 3):
        files = radar_files(subject, radar_idx)
        if not files:
            raise FileNotFoundError(f"{subject}: radar{radar_idx} 파일 없음")
        radars.append(np.concatenate([read_motion_file(p) for p in files], axis=0))
    length = min(map(len, radars))
    merged = np.concatenate([r[:length] for r in radars], axis=1)
    usable = len(merged) // DOWNSAMPLE * DOWNSAMPLE
    motion = merged[:usable].reshape(-1, DOWNSAMPLE, merged.shape[1]).mean(axis=1).astype(np.float32)
    np.savez_compressed(cache, motion=motion)
    return motion


def smooth(x, samples):
    if samples <= 1:
        return x
    kernel = np.ones(samples, dtype=np.float64) / samples
    return np.convolve(x, kernel, mode="same")


def detect_peak_time(motion, expected_s, radius_s=10.0):
    rate = FPS / DOWNSAMPLE
    energy = np.log1p(motion).sum(axis=1)
    energy = smooth(energy, max(1, round(rate * 1.5)))
    lo = max(0, int((expected_s - radius_s) * rate))
    hi = min(len(energy), int((expected_s + radius_s) * rate))
    if hi <= lo:
        return expected_s
    return (lo + int(np.argmax(energy[lo:hi]))) / rate


def detect_breath_resume(subject, apnea_start_s, fallback_s):
    path = SYNC_ROOT / subject / "aligned_rsp.npz"
    if not path.exists():
        return fallback_s, "예정시간(정렬 RSP 없음)"
    z = np.load(path)
    rsp = np.asarray(z["rsp_aligned"], dtype=np.float64)
    rate = FPS
    finite = np.isfinite(rsp)
    if not finite.any():
        return fallback_s, "예정시간(RSP 유효값 없음)"
    rsp = np.where(finite, rsp, np.nanmedian(rsp[finite]))
    d = np.abs(np.diff(rsp, prepend=rsp[0]))
    d = smooth(d, max(1, round(rate * 1.0)))
    base_lo = max(0, int((apnea_start_s - 110) * rate))
    base_hi = max(base_lo + 1, int((apnea_start_s - 65) * rate))
    baseline = d[base_lo:min(base_hi, len(d))]
    threshold = max(float(np.percentile(baseline, 65)) if len(baseline) else 0.01, 0.004)
    lo = max(0, int((apnea_start_s + 25) * rate))
    hi = min(len(d), int((apnea_start_s + 30) * rate))
    for idx in range(lo, hi):
        look = d[idx:min(len(d), idx + round(2.5 * rate))]
        if len(look) and float(np.mean(look > threshold)) >= 0.35:
            return idx / rate, "RSP 호흡 재개 탐지(25~30초 허용창)"
    return fallback_s, "예정시간(25~30초 허용, 자동 탐지 불충분)"


def fixed_windows(start, end, length, stride):
    values = []
    t = start
    while t + length <= end + 1e-6:
        values.append((t, t + length))
        t += stride
    return values


def pool_window(motion, start_s, end_s, steps=40):
    rate = FPS / DOWNSAMPLE
    lo = max(0, int(math.floor(start_s * rate)))
    hi = min(len(motion), int(math.ceil(end_s * rate)))
    segment = motion[lo:hi]
    if len(segment) < 2:
        raise ValueError("window too short")
    edges = np.linspace(0, len(segment), steps + 1).astype(int)
    pooled = np.empty((steps, segment.shape[1]), dtype=np.float32)
    for i in range(steps):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            b = min(len(segment), a + 1)
        pooled[i] = segment[a:b].mean(axis=0)
    return pooled


def add_windows(store, motion, subject, task, label, intervals, source, confidence):
    for start_s, end_s in intervals:
        store.append({
            "subject": subject,
            "task": task,
            "label": label,
            "start_s": float(start_s),
            "end_s": float(end_s),
            "source": source,
            "confidence": confidence,
            "feature": pool_window(motion, start_s, end_s),
        })


def draw_diagnostic(subject, motion, markers, boundaries, apnea_resume, resume_source):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rate = FPS / DOWNSAMPLE
    start = markers[0] - 5
    end = markers[3] + 5
    lo = max(0, int(start * rate))
    hi = min(len(motion), int(end * rate))
    segment = np.log1p(motion[lo:hi])
    width, height = 1700, 900
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 25), f"{subject} - 실험 1·2 경계 및 레이더별 움직임", fill="#17223b", font=font(27, True))
    plot_left, plot_right = 90, width - 60
    colors = ["#2b6cb0", "#d97706", "#16856b"]
    for radar in range(3):
        top = 120 + radar * 205
        bottom = top + 145
        values = segment[:, radar * N_ZONES:(radar + 1) * N_ZONES].sum(axis=1)
        q1, q99 = np.percentile(values, [1, 99])
        scale = max(q99 - q1, 1e-6)
        points = []
        for i, value in enumerate(values):
            x = plot_left + i / max(len(values) - 1, 1) * (plot_right - plot_left)
            y = bottom - np.clip((value - q1) / scale, 0, 1) * (bottom - top)
            points.append((x, y))
        if len(points) > 1:
            d.line(points, fill=colors[radar], width=2)
        d.text((25, top + 45), f"Radar {radar + 1}", fill=colors[radar], font=font(18, True))
        d.line((plot_left, bottom, plot_right, bottom), fill="#cbd5e1", width=1)
    timeline_start = lo / rate
    timeline_end = hi / rate
    for name, value, color in boundaries:
        x = plot_left + (value - timeline_start) / max(timeline_end - timeline_start, 1e-6) * (plot_right - plot_left)
        d.line((x, 95, x, 735), fill=color, width=2)
        d.text((x + 3, 82), name, fill=color, font=font(13, True))
    y = 770
    d.text((90, y), f"숨 참기 실제 종료 추정: {apnea_resume:.2f}s · {resume_source}", fill="#4a5568", font=font(18))
    d.text((90, y + 38), "주의: 화면 로그가 없어 PPT 시간표와 신호 기반으로 복원한 경계이며, 전환 구간은 학습에서 제외함.", fill="#9b2c2c", font=font(17))
    img.save(FIG_DIR / f"{subject}_S01_S02.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    windows = []
    decisions = []
    for record in approved_records():
        subject = record["subject"]
        print(f"[prepare] {subject}", flush=True)
        motion = subject_features(subject)
        markers = [float(x) for x in record["selected_slots"][:4]]
        s1_start, s1_end, s2_start, s2_end = markers

        s1_pad = min(15.0, max(0.0, (s1_end - s1_start - 180.0) / 2.0))
        s1_content = s1_start + s1_pad
        turn1_expected = s1_content + 60.0
        turn2_expected = s1_content + 120.0
        turn1 = detect_peak_time(motion, turn1_expected, 9.0)
        turn2 = detect_peak_time(motion, turn2_expected, 9.0)
        if turn2 <= turn1 + 35:
            turn1, turn2 = turn1_expected, turn2_expected
        add_windows(windows, motion, subject, "S01_angle", "FACE_R1", fixed_windows(s1_content + 5, turn1 - 4, 8, 4), "화면 1분+회전 피크", "A")
        add_windows(windows, motion, subject, "S01_angle", "FACE_R2", fixed_windows(turn1 + 4, turn2 - 4, 8, 4), "회전 피크 사이", "A")
        add_windows(windows, motion, subject, "S01_angle", "FACE_R3", fixed_windows(turn2 + 4, s1_content + 175, 8, 4), "두 번째 회전 후", "A")

        s2_pad = min(18.0, max(0.0, (s2_end - s2_start - 360.0) / 2.0))
        s2_content = s2_start + s2_pad
        apnea_start = s2_content + 120.0
        apnea_resume, resume_source = detect_breath_resume(subject, apnea_start, apnea_start + 30.0)
        apnea_resume = min(max(apnea_resume, apnea_start + 25.0), apnea_start + 30.0)
        add_windows(windows, motion, subject, "S02_state", "NORMAL", fixed_windows(s2_content + 5, s2_content + 55, 10, 5), "화면 시간표", "A")
        add_windows(windows, motion, subject, "S02_state", "SLOW", fixed_windows(s2_content + 65, s2_content + 115, 10, 5), "화면 시간표", "A")
        add_windows(windows, motion, subject, "S02_state", "HOLD", fixed_windows(apnea_start + 2, apnea_resume - 2, 8, 4), "25~30초 허용+RSP 확인", "B")
        add_windows(windows, motion, subject, "S02_state", "POST_HOLD", fixed_windows(apnea_resume + 5, s2_content + 205, 10, 5), "RSP 재개 후~운동 준비 전", "B")
        add_windows(windows, motion, subject, "S02_state", "SQUAT", fixed_windows(s2_content + 230, s2_content + 280, 10, 5), "화면 시간표(준비/착석 제외)", "A")
        add_windows(windows, motion, subject, "S02_state", "POST_EXERCISE", fixed_windows(s2_content + 305, s2_content + 355, 10, 5), "화면 시간표(착석 후 5초 제외)", "A")

        decisions.append({
            "subject": subject,
            "m01_s": s1_start,
            "m02_s": s1_end,
            "s01_duration_s": s1_end - s1_start,
            "s01_content_start_s": s1_content,
            "turn1_s": turn1,
            "turn2_s": turn2,
            "m03_s": s2_start,
            "m04_s": s2_end,
            "s02_duration_s": s2_end - s2_start,
            "s02_content_start_s": s2_content,
            "apnea_start_s": apnea_start,
            "apnea_resume_s": apnea_resume,
            "apnea_duration_s": apnea_resume - apnea_start,
            "resume_basis": resume_source,
            "marker_basis": "기존 전수검사 자동 점검 통과 M01~M04",
            "label_basis": "화면 안내 기억+PPT 시간표+레이더 회전 피크+정렬 BIOPAC",
        })
        boundaries = [
            ("M01", s1_start, "#c53030"), ("회전1", turn1, "#805ad5"), ("회전2", turn2, "#805ad5"),
            ("M02", s1_end, "#c53030"), ("M03", s2_start, "#c53030"),
            ("무호흡", apnea_start, "#2b6cb0"), ("호흡재개", apnea_resume, "#16856b"),
            ("스쿼트", s2_content + 225, "#d97706"), ("M04", s2_end, "#c53030"),
        ]
        draw_diagnostic(subject, motion, markers, boundaries, apnea_resume, resume_source)

    features = np.asarray([w.pop("feature") for w in windows], dtype=np.float32)
    np.savez_compressed(
        OUT / "labeled_windows.npz",
        features=features,
        subjects=np.asarray([w["subject"] for w in windows], dtype=object),
        tasks=np.asarray([w["task"] for w in windows], dtype=object),
        labels=np.asarray([w["label"] for w in windows], dtype=object),
        starts=np.asarray([w["start_s"] for w in windows], dtype=np.float32),
        ends=np.asarray([w["end_s"] for w in windows], dtype=np.float32),
        sources=np.asarray([w["source"] for w in windows], dtype=object),
        confidence=np.asarray([w["confidence"] for w in windows], dtype=object),
    )
    with (OUT / "label_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(windows[0].keys()))
        writer.writeheader(); writer.writerows(windows)
    with (OUT / "boundary_decisions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(decisions[0].keys()))
        writer.writeheader(); writer.writerows(decisions)
    summary = {
        "subjects": sorted(set(w["subject"] for w in windows)),
        "window_count": len(windows),
        "task_counts": {task: sum(w["task"] == task for w in windows) for task in sorted(set(w["task"] for w in windows))},
        "label_counts": {label: sum(w["label"] == label for w in windows) for label in sorted(set(w["label"] for w in windows))},
        "protocol_note": "숨 참기는 25초 이후 자율 종료 가능. 25~30초 창에서 RSP 호흡 재개를 탐지하고, 실패 시 30초를 사용.",
    }
    (OUT / "label_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
