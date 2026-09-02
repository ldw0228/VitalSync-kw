import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'sync_results'
records = json.loads((RESULTS / 'normalization_records.json').read_text(encoding='utf-8'))
by_number = {item['number']: item for item in records}

AUTO_CONFIRMED = {28, 26, 23, 21, 20, 19, 18, 17, 16, 15, 14, 10, 9, 8, 7}
MANUAL = {
    27: [6.2,198.3,221.1,596.9,649.1,663.7,682.1,695.7,754.0,767.2,780.3,825.9,838.1,889.0,910.0,922.5,972.0,1159.6,1183.3,1205.4,1225.3,1236.0],
    25: [19.0,216.6,237.0,611.8,700.7,722.7,739.9,749.3,848.9,863.8,874.6,923.9,934.0,989.2,1008.1,1021.9,1093.6,1283.9,1319.4,1345.4,1373.2,1387.4],
    13: [11.5,205.2,228.3,597.4,643.4,656.8,675.5,684.9,699.2,743.0,757.9,795.8,813.2,864.7,874.3,886.4,920.5,1109.9,1129.3,1153.3,1159.6,1178.9],
    6: [22.0,214.0,264.0,638.0,748.4,763.3,778.2,792.3,919.0,934.5,947.0,994.0,1009.8,1059.2,1075.4,1086.0,1160.7,1346.5,1394.6,1424.2,1458.5,1471.3],
    5: [15.0,210.0,241.0,612.0,620.0,656.0,784.0,820.0,931.0,947.0,963.0,1008.0,1022.0,1068.0,1072.0,1089.0,1201.0,1389.0,1438.0,1466.0,1498.0,1509.0],
}
REVIEW = {
    1: '자동 검출 마커 14개로 초기 프로토콜 경계가 부족함 — 수동 검토 필요',
    24: '원본 누락: Drive/로컬 모두 UWB datafloat 없음(BIOPAC만 존재) — 싱크 불가',
    22: 'RSP 초반 마커와 낙상 4번째 종료 마커가 불명확 — 수동 검토 필요',
    12: 'RSP 마커 누락 및 자동 offset이 탐색 경계(-12초)에 도달 — 수동 검토 필요',
    11: '후반 마커 다수 누락/과검출 — 수동 검토 필요',
    4: '후반부 장시간 마커 공백(약 1,230~1,680초) — 수동 검토 필요',
}


def scenario_cells(markers):
    m = [int(round(value)) for value in markers]
    pairs = [f'{m[index]}~{m[index + 1]}' for index in range(0, 22, 2)]
    return [pairs[0], pairs[1], f'{pairs[2]} / {pairs[3]}', '/'.join(pairs[4:8]), pairs[8], pairs[9], pairs[10]]


final = []
for number in range(30, 0, -1):
    if number == 30:
        markers = [6.45,192.15,237.275,620.125,661.775,676.4,730.025,741.25,755.025,791.15,800.675,844.1,853.65,896.7,911.825,968.925,974.375,1158.775,1175.375,1219.625,1243.925,1254.325]
        final.append({'number': number, 'subject': 'S30_SJE', 'status': 'manual_corrected', 'markers_radar_s': markers,
                      'scenario_cells': scenario_cells(markers), 'note': 'BIOPAC RSP 포화/클리핑으로 자동 마커 5개만 검출; 레이더 3대와 결합해 수동 보정'})
    elif number == 29:
        markers = [17.784,209.824,232.384,606.668,652.486,666.664,682.42,695.828,764.14,781.232,790.8,837.34,848.692,896.944,914.5,927.224,972.348,1158.54,1186.98,1210.275,1236.928,1251.62]
        final.append({'number': number, 'subject': 'S29_LHS', 'status': 'manual_corrected', 'markers_radar_s': markers,
                      'scenario_cells': scenario_cells(markers), 'note': '약 970/975초 이중 마커를 1개로 병합'})
    elif number in AUTO_CONFIRMED:
        source = by_number[number]
        markers = source['markers_radar_s']
        final.append({'number': number, 'subject': source['subject'], 'status': 'confirmed',
                      'markers_radar_s': markers, 'scenario_cells': scenario_cells(markers),
                      'note': f"참가자별 임계값 {source['threshold']:.1f}, 후보 {source['candidate_count']}개 중 22개 검증"})
    elif number in MANUAL:
        markers = MANUAL[number]
        final.append({'number': number, 'subject': by_number[number]['subject'], 'status': 'manual_corrected',
                      'markers_radar_s': markers, 'scenario_cells': scenario_cells(markers),
                      'note': '자동 후보를 RSP 돌출·레이더 움직임·실험 순서로 수동 보정'})
    elif number in REVIEW:
        subject = by_number.get(number, {}).get('subject', 'S01_CMS' if number == 1 else f'S{number:02d}')
        final.append({'number': number, 'subject': subject, 'status': 'review_required',
                      'scenario_cells': None, 'note': REVIEW[number]})

(RESULTS / 'final_sync_records.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
with (RESULTS / 'sync_anomalies.csv').open('w', newline='', encoding='utf-8-sig') as handle:
    writer = csv.DictWriter(handle, fieldnames=['number', 'subject', 'status', 'note'])
    writer.writeheader()
    for item in final:
        if item['status'] != 'confirmed':
            writer.writerow({key: item.get(key, '') for key in writer.fieldnames})

print(RESULTS / 'final_sync_records.json')
print(RESULTS / 'sync_anomalies.csv')
