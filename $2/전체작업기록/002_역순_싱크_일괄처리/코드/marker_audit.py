import json
from pathlib import Path

import numpy as np

from matv5_reader import loadmat
from sync_subject import detect_markers


ROOT = Path(__file__).resolve().parent.parent / 'HAI_EXPERIMENT'


for subject in sorted(ROOT.glob('S*_*/'), reverse=True):
    mat_files = sorted((subject / 'BIOPAC').rglob('*.mat'))
    if not mat_files:
        continue
    mat = loadmat(mat_files[0])
    rsp = np.asarray(mat['data'][:, 0], dtype=np.float64)
    isi = float(np.asarray(mat.get('isi', [[4]])).reshape(-1)[0])
    fs = 1000.0 / isi
    counts = {f'{threshold:.1f}': len(detect_markers(rsp, fs, threshold))
              for threshold in np.arange(4.0, 10.01, 0.5)}
    percentiles = np.percentile(rsp, [0, 1, 25, 50, 75, 95, 99, 99.5, 99.9, 100]).round(3).tolist()
    print(json.dumps({'subject': subject.name, 'percentiles': percentiles, 'counts': counts}, ensure_ascii=False))
