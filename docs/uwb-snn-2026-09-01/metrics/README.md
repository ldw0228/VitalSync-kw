# 원자료

## 최종 결과
| 파일 | 내용 |
|---|---|
| `train_v2.json` | **최종 14조합 결과.** 3분할 + 조기종료. 문서의 모든 수치는 이 파일 기준 |
| `실험비교_마커vs지형지물.xlsx` | 5시트 — 피험자별 구간표 / 모델결과 / 학습설정 / 요약 / 판단근거 |
| `자른구간_전체.csv` | 피험자 28명의 모든 구간 시작·종료 시각 |
| `자른구간_마커비교.csv` | 지형지물 구간과 마커의 시차 |

## 결함 있는 파일 (기록용)
| 파일 | 문제 |
|---|---|
| `train_all.json` | 평가셋으로 에폭 선택 — 최대 0.29 부풀려짐 |
| `train_long.json` | 위와 동일 |
| `sel_*.json` `ablate.json` `joint.json` | 교정 전 방법론 |

## 동기화
`sync12.json` `sync12b.json` `sync12c.json` `sync12_final.json` — 지형지물 정박 단계별 결과.
최종은 `sync12_final.json` (27 확실 / 1 주의 / 1 실패).
`landmarks.json` `anchors.json` `xl_exp2.json` — 지형지물 위치, 마커 엑셀에서 뽑은 실험2 시작.

## 그 밖
`baseline*.json` 기준선 · `match.json` 일치도 · `angle_phys.json` 물리검증 ·
`recovery.json` 회복호흡 · `cross_radar.json` `extrap.json` 교차일반화 ·
`e2e*.json` `wave_rate.json` end-to-end · `ab_marker.json` `ab2.json` 마커/지형지물 비교 ·
`per_subject.json` 피험자별 · `summary_A.json` `summary_B.json` 분류 과제.
