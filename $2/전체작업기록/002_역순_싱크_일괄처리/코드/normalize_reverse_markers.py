import json
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / 'HAI_EXPERIMENT'
RESULTS = HERE / 'sync_results'
SYNC = HERE / 'sync_subject.py'

# Highest practical threshold that retains at least the expected 22 boundaries.
THRESHOLDS = {
    28: 7.0, 27: 8.0, 26: 8.0, 25: 8.5, 23: 8.0, 22: 4.5,
    21: 7.5, 20: 6.5, 19: 7.0, 18: 7.0, 17: 7.0, 16: 8.0,
    15: 8.0, 14: 6.5, 13: 9.0, 12: 6.0, 11: 7.0, 10: 8.0,
    9: 7.5, 8: 7.0, 7: 7.5, 6: 8.5, 5: 9.0, 4: 9.0,
}

S29 = np.asarray([17.784,209.824,232.384,606.668,652.486,666.664,682.42,695.828,
                  764.14,781.232,790.8,837.34,848.692,896.944,914.5,927.224,
                  972.348,1158.54,1186.98,1210.275,1236.928,1251.62])
S30 = np.asarray([6.45,192.15,237.275,620.125,661.775,676.4,730.025,741.25,
                  755.025,791.15,800.675,844.1,853.65,896.7,911.825,968.925,
                  974.375,1158.775,1175.375,1219.625,1243.925,1254.325])


def normalized(values):
    return (values - values[0]) / (values[-1] - values[0])


TEMPLATE = (normalized(S29) + normalized(S30)) / 2


def select_template(candidates, count=22):
    c = np.asarray(candidates, dtype=float)
    if len(c) == count:
        return c, 0.0
    best_cost = float('inf')
    best = None
    n = len(c)
    for first in range(0, n - count + 1):
        for last in range(first + count - 1, n):
            span = c[last] - c[first]
            if span <= 0:
                continue
            pred = c[first] + TEMPLATE * span
            dp = np.full((count, n), np.inf)
            prev = np.full((count, n), -1, dtype=int)
            dp[0, first] = 0.0
            for k in range(1, count - 1):
                j_min = first + k
                j_max = last - (count - 1 - k)
                for j in range(j_min, j_max + 1):
                    prior = dp[k - 1, first + k - 1:j]
                    if prior.size == 0:
                        continue
                    expected_gap = pred[k] - pred[k - 1]
                    gap_scale = max(0.015 * span, 0.5 * expected_gap)
                    prior_indices = np.arange(first + k - 1, j)
                    costs = prior + ((c[j] - c[prior_indices] - expected_gap) / gap_scale) ** 2
                    p = int(np.argmin(costs)) + first + k - 1
                    dp[k, j] = costs[p - (first + k - 1)] + ((c[j] - pred[k]) / span) ** 2
                    prev[k, j] = p
            prior = dp[count - 2, first + count - 2:last]
            if prior.size == 0:
                continue
            p = int(np.argmin(prior)) + first + count - 2
            cost = dp[count - 2, p]
            if cost < best_cost:
                indices = [last, p]
                for k in range(count - 2, 0, -1):
                    indices.append(prev[k, indices[-1]])
                indices.reverse()
                best = c[indices]
                best_cost = float(cost)
    if best is None:
        raise RuntimeError('Could not select ordered marker template')
    return best, best_cost


def run_sync(subject, threshold, manual=None):
    cmd = [sys.executable, str(SYNC), subject, '--threshold', str(threshold)]
    if manual is not None:
        cmd += ['--manual-biopac-markers', ','.join(f'{value:.6f}' for value in manual)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.loads((RESULTS / subject / 'sync_result.json').read_text(encoding='utf-8'))


records = []
for number in range(28, 3, -1):
    if number == 24:
        records.append({'number': 24, 'subject': 'S24_KHJ', 'status': 'source_missing',
                        'note': 'Drive/로컬 모두 UWB datafloat 없음(BIOPAC만 존재)'})
        print('S24_KHJ: source missing')
        continue
    subject = next(path.name for path in DATASET.iterdir() if path.is_dir() and path.name.startswith(f'S{number:02d}_'))
    threshold = THRESHOLDS[number]
    candidate_result = run_sync(subject, threshold)
    candidates = candidate_result['marker_biopac_s']
    if len(candidates) < 22:
        records.append({'number': number, 'subject': subject, 'status': 'insufficient_candidates',
                        'threshold': threshold, 'candidate_count': len(candidates)})
        print(f'{subject}: insufficient {len(candidates)}')
        continue
    selected, fit_cost = select_template(candidates)
    final_result = run_sync(subject, threshold, selected)
    records.append({
        'number': number,
        'subject': subject,
        'status': 'normalized',
        'threshold': threshold,
        'candidate_count': len(candidates),
        'removed_candidates': len(candidates) - 22,
        'template_fit_cost': fit_cost,
        'offset_s': final_result['offset_s'],
        'markers_radar_s': final_result['marker_radar_s'],
        'radar_frame_counts': final_result['radar_frame_counts'],
        'rsp_low_clip_ratio': final_result['rsp_low_clip_ratio'],
        'rsp_high_clip_ratio': final_result['rsp_high_clip_ratio'],
    })
    print(f"{subject}: 22 selected from {len(candidates)}, offset={final_result['offset_s']:+.1f}, fit={fit_cost:.4f}")

(RESULTS / 'normalization_records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'기록: {RESULTS / "normalization_records.json"}')
