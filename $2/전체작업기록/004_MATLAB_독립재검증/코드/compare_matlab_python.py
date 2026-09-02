import csv
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DATASET = WORKSPACE.parent / 'HAI_EXPERIMENT'
PYTHON_RAW = HERE / 'python_raw'
MATLAB_RESULTS = HERE / 'matlab_raw' / 'matlab_results.json'
FINAL_RESULTS = WORKSPACE / 'sync_results' / 'final_sync_records.json'

matlab = {item['subject']: item for item in json.loads(MATLAB_RESULTS.read_text(encoding='utf-8-sig'))}
final = {item['subject']: item for item in json.loads(FINAL_RESULTS.read_text(encoding='utf-8'))}

rows = []
for subject_dir in sorted(DATASET.glob('S*_*')):
    subject = subject_dir.name
    mat = matlab[subject]
    py = None
    if mat['status'] == 'OK':
        subprocess.run([
            sys.executable, str(WORKSPACE / 'sync_subject.py'), subject,
            '--threshold', '8.5', '--output-root', str(PYTHON_RAW),
        ], check=True, stdout=subprocess.DEVNULL)
        py = json.loads((PYTHON_RAW / subject / 'sync_result.json').read_text(encoding='utf-8'))

    final_item = final.get(subject)
    max_marker_diff = None
    count_match = None
    offset_diff = None
    if py:
        mat_markers = mat['marker_biopac_s']
        py_markers = py['marker_biopac_s']
        count_match = len(mat_markers) == len(py_markers)
        if count_match and mat_markers:
            max_marker_diff = max(abs(a-b) for a, b in zip(mat_markers, py_markers))
        offset_diff = abs(float(mat['offset_s']) - float(py['offset_s']))

    if mat['status'] != 'OK':
        validation = '원본 누락'
    elif count_match and (max_marker_diff or 0) < 1e-9 and (offset_diff or 0) < 1e-9:
        validation = 'MATLAB-Python 완전 일치'
    else:
        validation = '차이 확인 필요'

    rows.append({
        'subject': subject,
        'matlab_status': mat['status'],
        'matlab_marker_count': mat.get('marker_count'),
        'matlab_offset_s': mat.get('offset_s'),
        'python_marker_count': py['marker_count'] if py else None,
        'python_offset_s': py['offset_s'] if py else None,
        'max_marker_difference_s': max_marker_diff,
        'raw_validation': validation,
        'final_status': final_item['status'] if final_item else ('기존값 유지' if subject.startswith(('S02_', 'S03_')) else '미반영'),
        'final_note': final_item.get('note', '') if final_item else '',
    })

(HERE / 'comparison.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
with (HERE / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(json.dumps({
    'subjects': len(rows),
    'raw_exact_matches': sum(row['raw_validation'] == 'MATLAB-Python 완전 일치' for row in rows),
    'source_missing': sum(row['raw_validation'] == '원본 누락' for row in rows),
    'needs_review': [row['subject'] for row in rows if row['final_status'] == 'review_required'],
}, ensure_ascii=False, indent=2))
