import sys
from pathlib import Path

import numpy as np

from matv5_reader import loadmat


ROOT = Path(__file__).resolve().parent.parent / 'HAI_EXPERIMENT'


def prominent_peaks(rsp, fs, max_count=60):
    step = max(1, int(round(fs / 10)))
    y = rsp[::step]
    smooth = np.convolve(y, np.ones(3) / 3, mode='same')
    maxima = np.flatnonzero((smooth[1:-1] >= smooth[:-2]) & (smooth[1:-1] > smooth[2:])) + 1
    flank = 80  # 8 seconds at 10 Hz
    candidates = []
    for index in maxima:
        if index < flank or index + flank >= len(smooth):
            continue
        left_low = np.percentile(smooth[index - flank:index], 10)
        right_low = np.percentile(smooth[index + 1:index + flank + 1], 10)
        prominence = smooth[index] - max(left_low, right_low)
        candidates.append((float(prominence), float(smooth[index]), index / 10.0))
    chosen = []
    for item in sorted(candidates, reverse=True):
        if all(abs(item[2] - other[2]) > 4 for other in chosen):
            chosen.append(item)
        if len(chosen) >= max_count:
            break
    return sorted(chosen, key=lambda item: item[2])


for prefix in sys.argv[1:]:
    subject = next(path for path in ROOT.iterdir() if path.is_dir() and path.name.startswith(prefix))
    mat_path = next((subject / 'BIOPAC').rglob('*.mat'))
    mat = loadmat(mat_path)
    rsp = np.asarray(mat['data'][:, 0], dtype=float)
    fs = 1000.0 / float(np.asarray(mat.get('isi', [[4]])).reshape(-1)[0])
    peaks = prominent_peaks(rsp, fs)
    print(subject.name)
    print(' '.join(f'{time:.1f}({prom:.1f},{height:.1f})' for prom, height, time in peaks))
