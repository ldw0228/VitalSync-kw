import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from matv5_reader import loadmat


FPS = 40
FRAME_LENGTH = 185
RANGE_BINS = 92


def load_radar(folder):
    parts = []
    files = sorted(Path(folder).rglob('xethru_datafloat_*.dat'))
    for path in files:
        raw = np.fromfile(path, dtype='<f4')
        frames = raw.size // FRAME_LENGTH
        if frames < 40:
            continue
        matrix = raw[:frames * FRAME_LENGTH].reshape(frames, FRAME_LENGTH)
        iq = matrix[:, 1:]
        parts.append(iq[:, 0::2] + 1j * iq[:, 1::2])
    if not parts:
        raise FileNotFoundError(f'No radar datafloat files found under {folder}')
    return np.concatenate(parts, axis=0)


def moving_average(values, width):
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode='same')


def detect_markers(rsp, fs, threshold=8.5):
    above = rsp > threshold
    edges = np.diff(np.r_[False, above, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    peaks = []
    for start, stop in zip(starts, stops):
        segment = rsp[start:stop + 1]
        peaks.append((start + int(np.argmax(segment))) / fs)

    groups = []
    for value in sorted(peaks):
        if not groups or value - groups[-1][-1] > 4:
            groups.append([value])
        else:
            groups[-1].append(value)
    return np.asarray([float(np.mean(group)) for group in groups], dtype=np.float64)


def auto_offset(markers, motion, motion_time):
    normalized = (motion - motion.min()) / (np.ptp(motion) + np.finfo(float).eps)
    front = markers[markers < 300]
    if front.size == 0:
        front = markers
    candidates = np.arange(-12, 12.0001, 0.1)
    scores = np.zeros_like(candidates)
    for index, candidate in enumerate(candidates):
        radar_times = front - candidate
        radar_times = radar_times[(radar_times > 0) & (radar_times < motion_time[-1])]
        if radar_times.size:
            scores[index] = np.mean(np.interp(radar_times, motion_time, normalized, left=0, right=0))
    best = int(np.argmax(scores))
    return float(candidates[best]), candidates, scores


def hot_colormap(values):
    x = np.clip(values, 0, 1)
    r = np.clip(3 * x, 0, 1)
    g = np.clip(3 * x - 1, 0, 1)
    b = np.clip(3 * x - 2, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def draw_diagnostic(path, subject, rsp, fs, magnitudes, radar_duration, marker_bio, marker_radar, offset):
    width = 1900
    left = 90
    right = 30
    plot_width = width - left - right
    rsp_height = 260
    heat_height = 210
    gap = 42
    total_height = 60 + rsp_height + 3 * (heat_height + gap) + 30
    canvas = Image.new('RGB', (width, total_height), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((20, 16), f'{subject} UWB-BIOPAC sync | offset={offset:+.2f}s | markers={len(marker_bio)}', fill='black', font=font)

    duration = min(radar_duration, len(rsp) / fs - offset)
    x_scale = plot_width / max(duration, 1)

    y0 = 55
    draw.rectangle((left, y0, left + plot_width, y0 + rsp_height), outline=(80, 80, 80))
    sample_times = np.arange(len(rsp)) / fs - offset
    valid = (sample_times >= 0) & (sample_times <= duration)
    indices = np.flatnonzero(valid)
    if indices.size:
        stride = max(1, indices.size // (plot_width * 2))
        chosen = indices[::stride]
        vals = rsp[chosen]
        lo, hi = np.percentile(vals, [1, 99])
        scale = (rsp_height - 12) / max(hi - lo, 1e-9)
        points = [(left + int(sample_times[i] * x_scale), y0 + rsp_height - 6 - int((rsp[i] - lo) * scale)) for i in chosen]
        if len(points) > 1:
            draw.line(points, fill=(0, 120, 40), width=1)
    draw.text((12, y0 + 5), 'RSP', fill='black', font=font)
    for value in marker_radar:
        x = left + int(value * x_scale)
        if left <= x <= left + plot_width:
            draw.line((x, y0, x, y0 + rsp_height), fill=(220, 0, 0), width=2)

    for radar_index, magnitude in enumerate(magnitudes, start=1):
        y = y0 + rsp_height + gap + (radar_index - 1) * (heat_height + gap)
        target_columns = min(plot_width, magnitude.shape[0])
        sample_idx = np.linspace(0, magnitude.shape[0] - 1, target_columns).astype(int)
        matrix = magnitude[sample_idx, :].T
        cap = np.percentile(matrix, 99.5)
        normalized = matrix / max(cap, 1e-9)
        heat = Image.fromarray(hot_colormap(normalized), mode='RGB').resize((plot_width, heat_height), Image.Resampling.BILINEAR)
        canvas.paste(heat, (left, y))
        draw.rectangle((left, y, left + plot_width, y + heat_height), outline=(80, 80, 80))
        draw.text((12, y + 5), f'R{radar_index}', fill='black', font=font)
        for value in marker_radar:
            x = left + int(value * x_scale)
            if left <= x <= left + plot_width:
                draw.line((x, y, x, y + heat_height), fill=(0, 255, 255), width=2)
        for seconds in range(0, int(duration) + 1, 60):
            x = left + int(seconds * x_scale)
            draw.line((x, y + heat_height, x, y + heat_height + 5), fill='black')
            draw.text((x - 10, y + heat_height + 8), str(seconds), fill='black', font=font)

    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    parser.add_argument('--dataset-root', default=str(Path(__file__).resolve().parent.parent / 'HAI_EXPERIMENT'))
    parser.add_argument('--output-root', default=str(Path(__file__).resolve().parent / 'sync_results'))
    parser.add_argument('--offset', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=8.5)
    parser.add_argument('--manual-biopac-markers', default=None)
    parser.add_argument('--manual-radar-markers', default=None)
    args = parser.parse_args()

    subject_dir = Path(args.dataset_root) / args.subject
    output_dir = Path(args.output_root) / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)

    radars = [load_radar(subject_dir / str(index)) for index in range(1, 4)]
    radar_frame_counts = [int(radar.shape[0]) for radar in radars]
    length = min(radar.shape[0] for radar in radars)
    radars = [radar[:length] for radar in radars]
    radar_time = np.arange(length, dtype=np.float64) / FPS

    magnitudes = []
    motion = np.zeros(length - 1, dtype=np.float64)
    for radar in radars:
        centered = radar - radar.mean(axis=0, keepdims=True)
        magnitudes.append(np.abs(centered))
        motion += np.sum(np.abs(np.diff(radar, axis=0)), axis=1)
    motion = moving_average(motion / 3, round(0.5 * FPS))
    motion_time = np.arange(length - 1, dtype=np.float64) / FPS

    mat_files = sorted((subject_dir / 'BIOPAC').rglob('*.mat'))
    if not mat_files:
        raise FileNotFoundError(f'No BIOPAC MAT-file found for {args.subject}')
    mat = loadmat(mat_files[0])
    rsp = np.asarray(mat['data'][:, 0], dtype=np.float64)
    isi = float(np.asarray(mat.get('isi', [[4]])).reshape(-1)[0])
    fs = 1000.0 / isi
    bio_time = np.arange(rsp.size, dtype=np.float64) / fs

    if args.manual_biopac_markers:
        marker_bio = np.asarray([float(value) for value in args.manual_biopac_markers.split(',')], dtype=np.float64)
    else:
        marker_bio = detect_markers(rsp, fs, args.threshold)
    auto_offset_value, candidates, scores = auto_offset(marker_bio, motion, motion_time)
    offset = auto_offset_value if args.offset is None else float(args.offset)
    detected_marker_radar = marker_bio - offset
    if args.manual_radar_markers:
        marker_radar = np.asarray([float(value) for value in args.manual_radar_markers.split(',')], dtype=np.float64)
    else:
        marker_radar = detected_marker_radar
    aligned_rsp = np.interp(radar_time, bio_time - offset, rsp, left=np.nan, right=np.nan)

    pairs = [[float(marker_radar[i]), float(marker_radar[i + 1])] for i in range(0, len(marker_radar) - 1, 2)]
    result = {
        'subject': args.subject,
        'offset_s': offset,
        'auto_offset_s': auto_offset_value,
        'fps_radar': FPS,
        'radar_frames': int(length),
        'radar_frame_counts': radar_frame_counts,
        'radar_duration_s': float(radar_time[-1]),
        'biopac_fs': fs,
        'biopac_duration_s': float(bio_time[-1]),
        'rsp_min': float(np.min(rsp)),
        'rsp_max': float(np.max(rsp)),
        'rsp_low_clip_ratio': float(np.mean(rsp <= -9.99)),
        'rsp_high_clip_ratio': float(np.mean(rsp >= 9.99)),
        'marker_count': int(len(marker_bio)),
        'marker_threshold': float(args.threshold),
        'marker_biopac_s': marker_bio.tolist(),
        'detected_marker_radar_s': detected_marker_radar.tolist(),
        'marker_radar_s': marker_radar.tolist(),
        'consecutive_pairs': pairs,
    }
    (output_dir / 'sync_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(output_dir / 'aligned_rsp.npz', t_radar=radar_time, rsp_aligned=aligned_rsp,
                        marker_radar_s=marker_radar, marker_biopac_s=marker_bio,
                        offset_s=np.asarray(offset))
    draw_diagnostic(output_dir / 'sync_diagnostic.png', args.subject, rsp, fs, magnitudes,
                    radar_time[-1], marker_bio, marker_radar, offset)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
