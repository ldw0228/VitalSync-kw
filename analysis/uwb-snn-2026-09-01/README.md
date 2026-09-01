# 코드 — 2026-09-01 세션

문서: [`docs/uwb-snn-2026-09-01/`](../../docs/uwb-snn-2026-09-01/)

원본 데이터(`HAI_EXPERIMENT`)와 전처리 캐시(`_work/*.npz`)는 저장소에 포함하지 않습니다.
경로는 전부 상대 경로입니다. 스크립트를 돌리려면 같은 작업 폴더에 다음이 있어야 합니다.

| 이름 | 내용 |
|---|---|
| `_work/` | 전처리 캐시 `*_mot.npz` (mot, radars, fps, rsp, fsb) · `*_iq.npz` (re, im — bin 8~63, 10 Hz) |
| `*.json` `*.npz` | 중간 산출물. json은 [`docs/uwb-snn-2026-09-01/metrics/`](../../docs/uwb-snn-2026-09-01/metrics/)에 있습니다 |

## 폴더

| 폴더 | 내용 |
|---|---|
| `preprocess/` | 원시 프레임 파싱 검증, I/Q 구조 확인, 배치 기하 |
| `sync/` | BIOPAC–레이더 시각 정렬. offset 추정 시도들과 지형지물 정박 |
| `dataset/` | 학습용 창 생성 |
| `models/` | 인코딩 정의, SNN/ANN 학습, 절제·end-to-end 실험 |
| `analysis/` | 기준선, 물리 검증, 표·엑셀 생성 |
| `figures/` | 문서에 들어간 그림·보고서 생성 |

## 파일별 역할

### preprocess/
| 파일 | 역할 | 관련 문서 |
|---|---|---|
| `ghost.py` | 호흡 대역 봉우리 쌍 탐지 — I/Q 파싱 오류를 드러낸 스크립트 | [02](../../docs/uwb-snn-2026-09-01/02_데이터와_전처리.md) |
| `ghost2.py` | 92 bin 간격·블록내 동일 위치 검증(87사례 중 97%) | 02 |
| `quad.py` `quad2.py` | 파싱 가설 4종 비교 — 앞92실수/뒤92허수 확정 | 02 |
| `offset_check.py` | 레이더 3대 간 프레임 카운터 정합 확인 | 02 |

### sync/
| 파일 | 역할 | 관련 문서 |
|---|---|---|
| `xcorr.py` `offset.py` `phase1.py` | offset 추정 3전 3패 — 마커 기반 정렬 시도들 | [03](../../docs/uwb-snn-2026-09-01/03_동기화.md) |
| `landmark.py` `seg1.py` | 지형지물(회전 동작 봉우리·무호흡 진폭 붕괴) 탐지 | 03 |
| `sync12.py` `sync12b.py` `sync12c.py` | 지형지물 정박 + 등급 판정(확실/주의/실패). 최종본은 `sync12c.py` | 03 |
| `verify_marker.py` | 지형지물 결과를 `HAI_동기화_종합.xlsx` 마커와 대조 | 03 |
| `cuts.py` | 피험자별 구간 시간표 산출 (`자른구간_전체.csv`) | 03 |
| `shift_test.py` | 인위적 시차 0/2/5/10/20초를 넣어 동기 오차의 영향 정량화 | 03 |
| `ab_marker.py` `ab2.py` | 마커 기반 vs 지형지물 기반 짝지어 비교 | 03 |

### dataset/
| 파일 | 역할 |
|---|---|
| `build_A.py` `build_B.py` `build_B2.py` | 초기 분류 과제(각도·호흡상태) 데이터셋 |
| `build_ds.py` `ds2.py` | 후보 선택 데이터셋 초기판 |
| `build_both.py` | **최종.** 마커/지형지물 두 방식 데이터셋을 같은 코드로 생성 |

### models/
| 파일 | 역할 | 관련 문서 |
|---|---|---|
| `enc2.py` | 스파이크 인코딩 5종 정의 (direct/rate/delta/step-forward/population) | [05](../../docs/uwb-snn-2026-09-01/05_모델실험.md) |
| `train_A.py` `train_B.py` `run_A.py` `run_B.py` | 분류 과제 학습 | [01](../../docs/uwb-snn-2026-09-01/01_실험연대기.md) |
| `cross_radar.py` | 레이더 교차 일반화·45° 외삽 | 01 |
| `train_sel.py` `sel_check.py` | 후보 선택기 초기판 | 05 |
| `train_all.py` | **결함 있음** — 평가셋으로 에폭을 골랐음. 기록용으로만 남김 | [06 §8](../../docs/uwb-snn-2026-09-01/06_실패기록.md) |
| `train_v2.py` | **최종.** 학습/검증/평가 3분할 + 조기종료 | 05 |
| `ablate.py` `joint.py` | 절제 실험 (파형만 / 특징만 / 둘 다) | 06 |
| `e2e.py` `e2e2.py` `wave.py` | end-to-end 시도 — 실패 | 06 |

### analysis/
| 파일 | 역할 |
|---|---|
| `baseline.py` | 학습 없는 고전 기준선 (되먹임 필터 + 스펙트럼 봉우리 포물선 보간) |
| `arctan.py` | 원적합·타원적합 arctangent 복조 비교 |
| `angle_check.py` | 각도별 호흡 변조 진폭 물리 검증 |
| `recov.py` | 회복 호흡 진폭·호흡률 재측정 |
| `match.py` `bio_check.py` `rr_range.py` `diag.py` | 레이더–BIOPAC 일치도, 호흡수 범위, 진단 |
| `mkxlsx.py` | 종합 엑셀 초판 (결함 있는 `train_all.json` 사용) |
| `mkxlsx2.py` | **최종.** `train_v2.json` 기준 + 판단근거 시트 |

### figures/
| 파일 | 산출물 |
|---|---|
| `fig_ghost.py` `fig_iq.py` | 파싱 오류·I/Q 설명 그림 |
| `fig_base.py` `fig_match.py` | 기준선·피험자별 일치도 |
| `fig_model.py` | **구버전** — 평가셋 누수 상태의 수치 |
| `fig_final.py` | **최종.** 14조합 결과 |
| `fig_waves.py` | 잘 맞는 파형 / 안 맞는 파형 |
| `charts.py` `build.js` `build_brief.js` | 보고서 문서(docx) 생성 |
