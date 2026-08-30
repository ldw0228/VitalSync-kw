# 상용 배포 후보 SNN 목표 명세 v2

상태: **진행 중 — 현재 데이터는 retrospective engineering set이며 상용·의료 성능 주장의 근거가 아님**

목표 owner: 모델·데이터·검증 파이프라인 전체. 단일 실험의 최고 점수가 아니라, 독립 사용자에 대한 재현 가능한 성능과 안전한 실패 동작을 납품 대상으로 삼는다.

## 0. 현재 기준점과 폐쇄해야 할 격차

동결된 현재 정확도 기준점은 `ensemble_structured_exact`의 identity-disjoint 6-fold OOF이다. 2,327개 valid-reference window / 18 identities에서 overall MAE 1.291 bpm, identity-macro MAE 1.220 bpm, RMSE 2.410 bpm, ±2 bpm 80.79%, >5 bpm 6.23%, 25–35 bpm MAE 4.216 bpm이다. 따라서 목표는 단순한 평균 개선이 아니라 다음 격차를 동시에 닫는 것이다.

| 항목 | 현재 | 목표 | 필요한 변화 |
|---|---:|---:|---:|
| Overall MAE | 1.291 | ≤1.000 | −0.291 bpm 이상 |
| Identity-macro MAE | 1.220 | ≤1.000 | −0.220 bpm 이상 |
| RMSE | 2.410 | ≤1.800 | −0.610 bpm 이상 |
| ±2 bpm | 80.79% | ≥90.0% | +9.21%p 이상 |
| >5 bpm | 6.23% | ≤3.0% | −3.23%p 이상 |
| 25–35 bpm MAE | 4.216 | ≤2.000 | −2.216 bpm 이상 |

이 표는 실험 선택에 쓰는 test leaderboard가 아니다. 후보 선택은 각 outer-test를 보지 않는 nested validation lock으로만 수행하고, 이 기준점은 최종 동결 OOF 비교에만 사용한다.

## 1. 최종 목적

3대 UWB radar의 32초 causal window로 호흡수(RR)를 추정하는 SNN을 구축한다. 모델은 사람 단위로 보지 못한 identity에 일반화하고, radar 결측·시간 중첩·alias/harmonic failure·불확실도·실제 전처리 지연까지 포함한 배포 조건에서 검증되어야 한다. 내부 데이터 성능을 만족한 뒤에도 사전 등록된 외부 prospective cohort를 통과하기 전에는 “상용 성능 달성”으로 판정하지 않는다.

## 2. 고정 1차 정확도 게이트

모든 지표는 frozen 18-identity, 6-fold identity-disjoint OOF의 2,327개 valid-reference window 전체에서 계산한다. 한 지표라도 실패하면 full-coverage 후보는 탈락한다.

| 지표 | 합격 기준 |
|---|---:|
| Overall MAE | ≤ 1.00 bpm |
| Identity-macro MAE | ≤ 1.00 bpm |
| RMSE | ≤ 1.80 bpm |
| ±2 bpm 비율 | ≥ 90.0% |
| >5 bpm catastrophic 비율 | ≤ 3.0% |
| 25–35 bpm MAE | ≤ 2.00 bpm |

보조 보고 지표는 bias, median/P90/P95/P99 absolute error, CCC, identity별 MAE, protocol별 MAE, 6개 fold별 지표, 95% identity-cluster bootstrap CI이다. 평균만 좋아지고 특정 identity/fold가 붕괴하는 후보는 승격하지 않는다.

분모는 임의로 바꾸지 않는다. valid-reference의 정의, RR 단위와 허용 범위, window 시작 시각, 중복 제거 규칙, identity/fold mapping을 manifest에 저장한다. 제외된 행은 사유별 개수와 identity 분포를 보고하고, inference 실패는 삭제하지 않고 catastrophic error 또는 unavailable로 제품 정책에 맞게 산정한다.

### 2.1 승격의 통계 규칙

- 후보는 point estimate뿐 아니라 identity-cluster bootstrap paired difference를 기준점과 비교한다. 전체 MAE와 identity-macro MAE의 95% CI 상한이 비열등 한계 `+0.05 bpm`을 넘으면 승격하지 않는다.
- high-RR 개선 때문에 normal-RR가 붕괴하지 않도록 25–35 bpm, 12–25 bpm, protocol, identity별 paired delta를 함께 잠근다. 어느 identity도 기준점보다 MAE가 1.0 bpm 이상 악화되는 후보는 원인 규명 전 승격하지 않는다.
- 최종 내부 합격은 동일한 protocol과 split에서 사전 고정한 3개 seed 모두가 1차 게이트를 통과할 때만 인정한다. seed를 보고 선택하거나 앙상블 구성에 test 성능을 쓰지 않는다.
- 비교 횟수와 가설 계보를 ledger에 남긴다. outer-test를 본 뒤 수정한 모든 실험은 `retrospective-adaptive`로 태깅하며, 이후 수치가 좋아도 독립 prospective 검증 전에는 확증 결과로 승격하지 않는다.

## 3. 일반화·강건성 게이트

- radar mask 7조건(123, 12, 13, 23, 1, 2, 3)을 모두 평가한다. 학습 시 radar dropout과 mask isolation을 검증한다.
- 32초 window가 겹치는 효과를 분리하기 위해 greedy non-overlap subset과 8개 stride phase를 모두 보고한다.
- full-radar 대비 pair/single-radar degradation, tail degradation, catastrophic 증가를 사전 정의된 비열등 한계로 판정한다.
- 모든 입력 전처리는 session 시작부터 유지되는 causal state로 재현되어야 한다. reference/identity/protocol/QC 결과는 forward 입력에 들어갈 수 없다.
- NaN/Inf, 모든 radar 결측, 부분 radar 결측은 finite output과 명시적 safe fallback을 가져야 한다.

추가 stress suite는 gain/offset 변화, 임펄스 및 burst noise, timestamp jitter/누락, range-bin shift, radar별 phase inversion, 짧은 flatline, session 시작 cold state를 포함한다. 물리적으로 가능한 perturbation 범위는 train 데이터 통계와 장비 사양으로 동결하고, 결과를 clean 성능과 별도 보고한다. 단일 radar 조건에서도 catastrophic 비율이 full-radar 대비 5%p 이상 증가하거나 NaN/무한대가 한 건이라도 발생하면 배포 게이트를 통과하지 못한다.

### 3.1 데이터와 reference 무결성

- raw radar/BIOPAC 원본, 파서, 시간 정렬, 단위 변환, outlier repair, resampling의 SHA-256과 버전을 기록한다.
- reference RR의 품질 기준은 모델 출력을 보지 않고 정한다. label latency, clipping, sensor dropout, 동기 오차를 session별로 감사하며, 불확실한 reference는 정확도 분모에서만 제외하고 모델 입력에는 절대 노출하지 않는다.
- 동일 인물·세션·중첩 raw interval이 train/validation/test 사이를 넘지 않도록 identity 및 interval leakage 검사를 자동화한다.
- weak label은 train identity의 loss에만 사용하고, valid-only 결과와 완전히 분리한 ablation으로 효과와 실패를 보고한다.

## 4. 불확실도·선택적 예측 게이트

- validation identity만으로 interval/conformal calibration을 고정하고 outer-test label로 threshold를 검색하지 않는다.
- 50/70/80/90/95% nominal interval의 empirical coverage와 interval width를 identity-macro로 보고한다.
- risk–coverage curve는 전체 후보와 별도로 보고한다. 낮은 coverage의 좋은 MAE를 full-coverage 성능으로 표현하지 않는다.
- abstention/low-quality 상태에는 제품 동작(재측정, 이전 안정값 유지, unavailable)을 명시한다.

불확실도는 error와의 Spearman 상관, AUROC/AUPRC(`>2`, `>5 bpm`), calibration error, interval score로 평가한다. coverage는 전체뿐 아니라 identity, RR band, protocol, radar mask별 최솟값을 보고한다. `unavailable`을 허용하는 제품 모드에서는 최소 absolute coverage를 사전 고정하고, abstention 행을 제거한 정확도와 함께 전체 분모 기준 failure/unavailable 비율도 반드시 제시한다.

## 5. 배포 게이트

- raw radar ingest부터 preprocessing, SNN forward, posterior decoding, uncertainty까지 동일한 streaming 구현으로 benchmark한다.
- CPU와 목표 accelerator에서 cold/warm p50/p95/p99 latency, peak RAM/VRAM, parameter 수, MAC/operation 및 spike activity를 기록한다.
- 4초 stride 내 p99 완료를 필수로 하고, feature parity를 golden-window hash와 수치 허용오차로 검증한다.
- checkpoint, scaler, thresholds, fold split, source/cache SHA-256, seed/RNG, dependency version을 하나의 immutable manifest로 묶는다.

배포 후보는 offline batch 코드가 아니라 session-state를 직렬화할 수 있는 streaming API로 재현한다. 재시작, state corruption, 순서가 뒤바뀐 window, 중복 window에 대한 idempotence/error policy를 테스트한다. CPU 기준 4초 stride 내 p99라는 hard limit 외에 목표 latency는 warm p95 250 ms 이하, peak RAM 1 GiB 이하로 두며, 하드웨어별 실측이 없으면 해당 플랫폼 승인을 주장하지 않는다. 모델 크기와 spike sparsity는 정확도와 함께 Pareto ledger에 기록한다.

각 release artifact에는 model card, intended use/금지 용도, 데이터 범위, 알려진 failure mode, 입력 계약, rollback 가능한 이전 버전, golden inputs/outputs가 포함되어야 한다. bit-exact가 불가능한 연산은 플랫폼별 수치 허용오차와 그에 따른 RR 출력 허용오차를 별도로 동결한다.

## 6. 모델 개발 루프

1. baseline과 OOF row/fold binding을 동결한다.
2. 가설을 `representation / model / loss / calibration / sequence state` 중 하나로 분류하고 label-free 입력만 정의한다.
3. train identities에서만 scaler, weak-label policy, sampling, action threshold를 적합한다.
4. validation identities에서 early stopping과 승격 결정을 잠근 뒤 lock hash를 기록한다.
5. validation 승자만 outer-test를 1회 생성하고, full 6-fold OOF로 승격한다.
6. full OOF 승자만 radar masks, non-overlap, calibration, E2E를 수행한다.
7. 실패 실험도 prediction/artifact와 실패 원인을 보존하여 같은 test 적응을 반복하지 않는다.

각 반복은 아래 decision record를 생성해야 한다.

1. 단 하나의 검증 가능한 가설과 예상 개선 지표를 선언한다.
2. 데이터 접근 범위와 leakage threat model을 서명한다.
3. 계산 예산, seed, 최대 epoch, early-stop, kill criterion을 실행 전에 고정한다.
4. unit/smoke → 1–2개 discovery outer fold → nested validation → full 6-fold 순으로 비용을 늘린다.
5. 성공·실패와 무관하게 config, stdout, checkpoint, prediction, metrics, environment hash를 원자적으로 보존한다.
6. 다음 반복은 residual을 identity/RR band/radar mask/시간 episode로 분해한 결과에서 선정한다.

우선순위는 `(예상 gate 개선 × 근거 강도 × 재사용성) / 계산·leakage 위험`으로 정한다. 현재 1순위 병목은 25–35 bpm의 지속적인 divisor/alias error이므로, raw/SVD source evidence, causal episode state, 직접 source/divisor supervision, 안전한 base fallback을 먼저 검증한다. oracle routing 성능은 상한 분석일 뿐 실현 성능으로 보고하지 않는다.

희귀 high-RR/alias specialist는 단일 validation fold에 positive support가 부족할 수 있으므로, outer-test를 제외한 identity들에서 nested inner grouped OOF로 gate/threshold를 정하고 별도 safety fold에서 비열등성을 확인한다. 이 변경은 retrospective iteration으로 표시하며 최종 주장에는 사용하지 않는다.

## 7. 허용되는 학습 신호

- valid reference는 train identities의 supervised loss에만 사용한다.
- invalid reference를 쓰는 경우 train identities에 한정한 명시적 weak-label ablation으로 분리하고, QC는 loss weight에만 쓰며 모델 입력에는 쓰지 않는다.
- all-window self-supervision, causal sequence state, radar masking, spectral/time-domain augmentation은 label-free로 허용한다.
- outer-test target, reference quality, test-derived threshold, identity lookup, 미래 window는 학습·선택·forward에 금지한다.

## 8. 반복 중단 조건

내부 목표는 서로 다른 최소 3개 seed에서 1차 게이트를 모두 통과하고, seed 평균뿐 아니라 최악 seed에서도 각 한계를 넘지 않을 때만 달성으로 본다. 이후 다음 외부 검증을 통과해야 상용 배포 후보로 확정한다.

- 사전 등록된 새로운 identity, device placement, protocol을 포함한 prospective cohort
- 결측 radar와 동작/고호흡 구간을 의도적으로 포함한 분포
- reference 장비와 QC를 개발 데이터와 독립적으로 운영
- 모델·threshold·전처리 hash를 데이터 개봉 전에 동결
- 1차 지표의 identity-cluster bootstrap CI 상한/하한까지 사전 기준 충족

외부 데이터 부재, reference 불확실성, 또는 재현 불가능한 preprocessing이 남아 있으면 goal은 완료가 아니라 **blocked**로 유지한다.

## 9. 외부 prospective 확증 설계

- cohort는 개발 identities와 겹치지 않으며 성별/연령/체형/자세/호흡 protocol/device placement/환경을 제품 intended-use 비율로 포함한다. high-RR와 motion, radar 결측은 우연에 맡기지 않고 최소 표본수를 별도 확보한다.
- 표본수는 window 수가 아닌 identity-cluster 구조를 반영해, primary MAE와 catastrophic 비율의 95% CI가 각각 사전 기준을 판정할 power를 갖도록 시뮬레이션으로 정한다.
- raw 데이터 개봉 전에 preprocessing, checkpoint ensemble, threshold, exclusion, primary/secondary endpoints, missing-data 처리와 통계 코드를 preregister한다.
- reference 운영자와 모델 운영자를 분리하고, adjudication은 prediction blind 상태에서 수행한다. 한 번의 확증 분석 후 실패하면 원인 분석은 가능하지만 같은 cohort에서 재튜닝한 결과는 새 확증으로 인정하지 않는다.

## 10. 상용 운영 준비와 종료 정의

출시 후에는 입력 분포 drift, radar availability, uncertainty, unavailable, latency, crash를 label-free로 상시 감시하고, 지연된 reference가 있는 샘플에서는 accuracy drift를 별도로 추적한다. alarm threshold, owner, 대응 SLA, shadow/canary 기간, 자동 rollback 조건을 release 전에 정한다. 개인정보·보존 기간·접근 통제와 감사 로그는 배포 환경의 요구사항을 따른다.

상태는 `research candidate → internally validated → prospectively validated → deployment candidate` 네 단계로만 올린다. 현재 목표의 완료 정의는 마지막 단계까지 필요한 모든 artifact와 독립 prospective 결과가 존재하고, 위 게이트에 미해결 예외가 없는 상태다. 내부 수치만 통과하면 `internally validated`이지 상용 성능 완료가 아니다.

## 11. 2026-08-28 실행 checkpoint

현재 full-coverage identity-disjoint 선두는 `ensemble_structured_exact`이며 정확도 gate는 0/6이다. SVD source/temporal/episode SNN, RR-balanced loss, nested tree router, physics ridge/HMM과 causal decoder까지 validation-lock 방식으로 반복했으나 선두를 안전하게 이긴 full-OOF 후보는 없었다. 상세 ledger와 근거는 `artifacts/COMMERCIAL_SNN_PROGRESS_V2.md`에 고정한다.

5-expert oracle은 MAE 0.689 bpm, macro MAE 0.674 bpm, RMSE 1.129 bpm, ±2 bpm 90.67%, >5 bpm 0.21%, 25–35 bpm MAE 1.224 bpm으로 표현 상한이 존재함을 보였다. 그러나 unseen identity에서 correction factor를 고르는 train-safe router는 6개 fold 모두 승격 기준을 통과하지 못했다. 그러므로 다음 iteration의 필수 자원은 동일 cohort의 추가 threshold 탐색이 아니라 독립 high-RR identity와 hardware-synchronized reference다.

Inference artifact 자체는 `all_windows_cuda_v3`에서 9,576행 exact cover, 6/6 CUDA-AMP strict frozen parity, raw-source fingerprint verification과 immutable SHA binding을 통과했다. 이 무결성 판정은 정확도 gate 실패와 외부 검증 부재를 대체하지 않는다.

현재 단계는 `research candidate`로 유지한다. 아래 네 조건이 모두 해제되기 전에는 goal을 완료로 바꾸지 않는다.

1. 서로 다른 3개 seed에서 내부 정확도 6개 gate 동시 통과
2. 완전히 nested인 base/router/calibration 또는 prospectively frozen 동등 증거
3. 독립 prospective calibration 및 confirmation cohort 통과
4. production streaming preprocessing parity와 실제 device fault campaign 통과
