# SnnProject 상용 후보 도달 Goal v3

- 기준일: 2026-08-31
- 상태: `ACTIVE / INTERNAL GATES FAILED / PROSPECTIVE CONFIRMATION ABSENT`
- 범위: 3대 XeThru UWB radar 기반 causal respiratory-rate 추정
- 과학 분류: 18명 기존 코호트에 대한 retrospective adaptive research
- 절대 금지: 현재 결과를 상용·의료 성능으로 표현, target-oracle을 배포 성능으로 표현, outer-test target로 구조·threshold·calibration 선택

이 문서는 기존 장기 goal을 실행 가능한 상태기계로 확장한다. 목표는 최고 점수 한 번이 아니라, 보지 못한 사람에서 반복 가능하고 결측·동작·고호흡·동기 오차에 안전한 SNN release candidate를 만드는 것이다. 내부 gate를 모두 통과해도 독립 prospective confirmation 전에는 상용 완료가 아니다.

문서 authority는 이 Goal v3와
`artifacts/SNN_PROJECT_DEVELOPMENT_PROGRESS_2026-08-31.md`가 최우선이다. Goal v2,
이전 execution plan/progress, 보존 release manifest와
`artifacts/commercial_goal_report.json`은 역사적 증거이며 현재 training/evaluation
authorization이 아니다.

## 1. 동결 기준점과 정량 격차

동결 기준점은 `ensemble_structured_exact`, 18 identities, identity-disjoint 6-fold, valid-reference 2,327 windows의 full-coverage OOF다.

| 지표 | 현재 | 내부 합격 | 격차 |
|---|---:|---:|---:|
| Overall MAE | 1.291 bpm | ≤1.000 bpm | −0.291 bpm |
| Identity-macro MAE | 1.220 bpm | ≤1.000 bpm | −0.220 bpm |
| RMSE | 2.410 bpm | ≤1.800 bpm | −0.610 bpm |
| 절대오차 ≤2 bpm | 80.79% | ≥90.00% | +9.21%p |
| 절대오차 >5 bpm | 6.23% | ≤3.00% | −3.23%p |
| 25–35 bpm MAE | 4.216 bpm | ≤2.000 bpm | −2.216 bpm |

고정 seed `20260828`, `20260829`, `20260830`이 **각각** 여섯 조건을 모두 통과해야 한다. seed 평균, 가장 좋은 seed, test를 본 후 만든 ensemble로 실패를 숨기지 않는다.

여섯 gate의 기계 판독 의미는 다음과 같다. 비교는 경계값을 포함하며(`le`는 `≤`,
`ge`는 `≥`), 반올림 전 finite float64 값으로 판정한다.

| gate ID | metric | operator | threshold |
|---|---|:---:|---:|
| `overall_mae_bpm` | 모든 authoritative row의 MAE | `le` | 1.0 |
| `identity_macro_mae_bpm` | identity별 MAE의 동일가중 평균 | `le` | 1.0 |
| `rmse_bpm` | 모든 authoritative row의 RMSE | `le` | 1.8 |
| `within_2_bpm_fraction` | `abs(error) ≤ 2` 비율 | `ge` | 0.90 |
| `error_gt_5_bpm_fraction` | `abs(error) > 5` 비율 | `le` | 0.03 |
| `rr_25_35_mae_bpm` | target `25 ≤ RR ≤ 35` MAE | `le` | 2.0 |

분모는 authority가 확정한 full valid-reference rowset 전체다. Fallback prediction도
일반 prediction과 동일하게 오차에 포함한다. No-estimate/unavailable row를 삭제하거나
선택적으로 제외하지 않는다. 하나라도 finite prediction이 없으면
`prediction_coverage_fraction < 1.0`으로 해당 seed/population의 accuracy 판정을
fail-closed 처리한다. NaN/Inf, 중복, 누락, 예상 밖 row도 동일하게 FAIL이다.

후보-oracle은 표현 상한 진단일 뿐이다. target를 사용해 후보를 고른 수치는 forward 가능 성능, router 성능, release 성능이 아니다.

## 2. 최종 완료 상태

Goal은 아래 네 상태를 순서대로 통과한다.

```text
R0_RESEARCH_ACTIVE
  └─ 내부 3-seed 정확도·강건성·calibration·streaming gate 통과
      ↓
R1_INTERNAL_ENGINEERING_PASS
  └─ 독립 prospective calibration cohort 통과
      ↓
R2_PROSPECTIVE_CALIBRATED
  └─ 별도 prospective confirmation + target device/fault/shadow 검증 통과
      ↓
R3_COMMERCIAL_RELEASE_CANDIDATE
```

`R3`도 의료기기 허가나 임상 유효성 선언과 동일하지 않다. intended use에 필요한 규제·품질체계는 별도다.

## 3. 변경할 수 없는 과학 불변식

1. split 단위: physical identity. 반복 session은 같은 fold.
2. outer-test target: sealed final join 외 접근 금지.
3. scaler, proposer, router, threshold, calibrator, ensemble weight, checkpoint 선택: outer-train/inner-validation만 사용.
4. BIOPAC/reference: label construction, 허용 train supervision, sealed evaluation에만 사용.
5. `radar_observable`: target-dependent이므로 inference 입력 금지.
6. missing radar/ratio/branch/candidate: 구조 mask 보존, scaling 후 exact `+0.0`.
7. temporal state: physical session 시작에서 reset, chronological update, future window 금지.
8. 실패 run/GPU ledger/부분 artifact: 삭제 금지.
9. authorization receipt: validator/runtime 발급만 허용, 수기 생성·수정 금지.
10. 기존 V3/V8R4 ancestry와 locked evidence: immutable, 새 실험은 새 version/root.

## 4. 단계별 gate 상태기계

### G0 — 복구·환경·기준점 폐쇄

목적: 다른 host에서도 동일한 source/data contract로 시작.

필수 증거:

- core/V8R4/raw archive SHA-256 검증
- 30 source folders, 29 usable sessions, 18 physical identities exact cover
- source/config/lock/interpreter/torch/CUDA/driver/device hash manifest
- 최소 focused tests + full suite 2회 연속 통과
- real bubblewrap 4 tests capable host 통과
- 기준 OOF row/fold/prediction hash 재검증

승격: 모든 hash와 row count 일치. 환경 차이는 명시적 deviation receipt로 남기며 bitwise 재현성을 주장하지 않음.

현재: source/test 복구는 완료. 이 host의 Torch는 CUDA build지만 driver가 노출되지 않아 CUDA 학습·latency 증거를 만들 수 없음.

### G1 — 원자료·코호트 authority

목적: 모든 usable session과 physical identity를 원자료에서 독립 재도출.

필수 검사:

- XeThru record geometry, chunk order, frame count, header/bin validity
- BIOPAC channel/sample rate/start time/rail/dropout 검사
- `S24_KHJ` empty radar 제외
- `S17_RJS → PJS` identity authority
- `S07_KDM` counter reset deterministic repair
- `S22_KJH` radar-2 단일 outlier past-only repair
- 30/29/18 exact-cover 및 unlisted-session fail-closed
- raw read-only, source byte hash 전후 동일

승격: parser 출력이 cohort authority와 일치하고 예외가 코드+test+hash로 설명됨.

### G2 — 레이더 시간축·causal resampling

목적: frame timing과 10 Hz causal grid를 측정 가능한 근거로 구성.

필수 검사:

- radar별 measured metadata timestamp 존재·단조성
- reset/plateau/gap/drop/duplicate를 명시적으로 분리
- half-open interval 4-frame mean, nominal latency 75 ms
- resample `valid_mask`와 sample count 보존
- invalid interval exact zero이되, numeric zero를 availability로 해석하지 않음
- marker detector가 invalid→zero 경계를 motion marker로 사용하지 못함
- marker용 resample과 feature용 resample의 byte/semantic parity

승격: 세 radar 전 구간의 모든 invalid interval이 원인 코드로 설명되고 structural
mask에 남으며, 설명되지 않은 invalid interval `0`, unaccounted frame `0`, 경계 위조
regression tests 통과. 실제 dropout/reset/gap을 없었던 것으로 만들기 위해 invalid
수를 0으로 강제하지 않는다.

현재: 29 usable 중 measured-timing eligible 19. 전 cohort 과학 학습 authority는 없음.

### G3 — Radar↔BIOPAC 동기화 authority

목적: 공통 hardware trigger가 없는 한계를 숨기지 않고 clock mapping을 독립 검증.

필수 증거:

- radar motion marker: target-free, resample-valid-only
- BIOPAC marker: RSP 원자료에서 재계산
- ordered one-to-one match, epoch prior, constant/affine mapping, drift/residual gate
- receipt marker/match/mapping을 **bound raw input에서 재계산**하는 verifier
- self-hash는 무결성에만 사용; 과학 authority로 사용 금지
- manual approval은 신뢰 anchor와 검증 가능한 서명이 없으면 승인 효력 없음
- reject/manual-review/ambiguous는 fail-closed
- receipt/config/pipeline/raw hash와 session exact binding

승격: raw-recomputed result가 receipt와 일치하고 automatic gate 통과 또는 신뢰 가능한 별도 approval scheme 통과.

현재: authorized 0/29. 진단 reconstruction만 존재.

### G4 — 7-phase block·reference contract

목적: 누움/앉음/걷기/운동/숨참기 등을 단순 시간 비율이 아니라 원 evidence로 구분.

필수 증거:

- 7개 phase config/order/duration prior 고정
- spreadsheet manual intervals와 RSP marker를 source에서 재파싱
- decoder output을 source+config로 독립 재실행
- stage document self-hash만으로 `auto`/metric eligibility 승격 금지
- transition guard 2 s, minimum overlap, attempt 분리
- phase-7 assignment는 phase-7에만 적용; whole-session label 금지
- protocol/identity/reference-quality는 inference feature 금지
- 32 s window의 stage membership, overlap, transition 여부를 deterministic 재계산
- breath-hold/운동은 steady RR만으로 평가하지 않고 transition/recovery stratum 별도 보고

승격: 29 usable 모두 source-recomputed stage contract 일치. `auto`가 아닌 phase/window는 stage metric에서 제외 사유 보고.

현재: stage-metric eligible 0/29.

### G5 — 신호 표현·denoising·mask

목적: motion/harmonic alias를 줄이면서 raw 물리 의미와 결측 구조 유지.

필수 표현:

- causal clip/mean removal/linear detrend/Hann/FFT/band selection/smoothing/noise normalization
- RF raw-power map, frozen IQ branch unavailable mask
- label-free SVD source separation, component reliability
- candidate × `{1/4,1/3,1/2,1,2,3,4}` evidence
- all three radar views와 single/pair mask topology
- outer-train-only scaler; unavailable exact zero
- candidate/radar/ratio/branch coordinate가 evidence와 pooling 전 결합
- gain/offset/impulse/burst/jitter/gap/range shift/phase inversion/flatline fault suite

승격: cache source/schema/config/content hash 일치, strict loader 통과, masked-cell invariance, no target/reference input.

### G6 — split·nested stack·누수 firewall

목적: 보지 못한 physical identity 성능만 측정.

필수 구조:

- 고정 6-fold identity exact cover
- outer `o`: test, `(o+1)%6`: validation, 나머지 train
- proposer도 prediction identity를 학습하지 않는 nested OOF
- overlapping raw interval가 split을 넘지 않음
- all adaptive selection은 inner/outer-validation에 한정
- outer-row-free sealed prediction pack
- final target join은 모든 prediction seal 이후 1회
- 동일 architecture/release mode를 모든 fold/seed에 적용

승격: identity/interval overlap 0, forbidden field open 0, sealed hash closure.

### G7 — 차기 SNN 구조

현재 채택안: `AxisRiskRouterSNNV8R5`, unmeasured proposal.

```text
571-wide candidate evidence
  ├─ Core encoder
  ├─ RF evidence × radar/ratio/branch/candidate coordinate joint MLP
  └─ SVD evidence × radar/ratio/modality/candidate coordinate joint MLP
       ↓ coordinate interaction first
bidirectional axial attention
  ├─ ratio → radar
  └─ radar → ratio
       ↓
7-relation directed harmonic graph × 2 PLIF blocks × 8 steps
       ↓
masked candidate pool + causal PLIF→ALIF session state
       ↓
anchor + candidate expert heads
  ├─ RR residual / predictive scale
  ├─ expected |error|
  ├─ P(|error|>2)
  └─ P(|error|>5)
       ↓
train: differentiable probability-weighted cost
infer: hard expert selection + finite fail-closed fallback
```

해결하려는 결함:

- 기존 `f(evidence)+coordinate embedding→mean`의 coordinate/evidence 대응 소실
- hard argmax 출력과 tail-risk objective 사이 gradient 단절
- 근접 중복 후보를 단일 index 정답으로 벌하는 문제

필수 unit gate:

- coordinate evidence swap sensitivity
- candidate permutation equivariance
- unavailable-cell contamination invariance
- available NaN/Inf fail-closed
- all-missing finite unavailable output
- soft route gradient 도달
- chronological whole/chunk streaming parity
- target/identity/protocol-free forward signature
- parameter cap ≤400,000, safe anchor initialization
- explicit per-feature와 classical-RR availability
- invalid target safe-before-cast, valid target 6–45 bpm strict gate
- value/route/risk head 분리와 실제 module-name stage freeze
- strict-only checkpoint load, route temperature/behavior/dependency hash 필수
- format-v2 cache schema·content·forward-array same-inode 검증

현재 구현 증거: 228,838 parameters, soft-routing gradient path, format-v2 cache
validator와 위 synthetic focused tests 구현·통과. Governed 571-layout과
`EpisodeSpikingCell`을 정확한 source hash로
재사용하는 별도 version proposal이며 V3/V8R4 ancestry와 완전히 독립적이라고
주장하지 않는다. Gradient 연결은 synthetic correctness 증거일 뿐이며 실데이터
routing 또는 정확도 개선은 아직 측정하지 않았다.

### G8 — 학습·선택 loop

각 iteration은 한 개의 지배 가설만 바꾸며 **inner/discovery validation만** 본다.
Outer-test는 iteration dashboard가 아니다.

1. `H0`: immutable leader/base fallback
2. `J0`: coordinate-interaction encoder
3. `J1`: soft expected-cost + equivalence-set + calibrated tail heads
4. `J2`: causal expert continuity/hysteresis
5. `J3`: identity×RR-band group DRO와 mask-balanced exposure
6. `J4`: label-free radar pretraining + analog teacher distillation
7. `J5`: range tracker를 feature contract까지 target-free 연결

학습·교정 순서:

1. axis encoder + expert value head warmup
2. soft risk head/router warmup
3. joint fine-tune
4. temperature anneal
5. inner identity cross-fitting으로 model·loss·temperature·threshold 후보 선택
6. 세 seed에 공통인 architecture/release mode와 fixed update 수 동결
7. outer fold별 non-test identity만 사용한 refit; outer-test target 접근 없음
8. outer fold마다 calibrator가 필요하면 non-test identity 내부 cross-fit/held identity로만 fit
9. 모든 outer prediction pack·selector·source hash seal
10. campaign 전체 outer-test target을 한 번에 공개하여 final join

Epoch 수가 아니라 optimizer update 수, batch construction, gradient clipping, early-stop patience, max wall time를 실행 전에 고정한다. 25–35 bpm과 각 identity의 총 loss mass를 보고하고, group DRO의 group label은 training weight에만 사용한다.

iteration 승격 기준:

- validation common key의 strict lexicographic improvement
- overall/high-RR/tail 중 목표 축 개선
- normal-RR, worst identity, full-coverage 비열등
- 세 seed 방향 일치
- NaN, mask leakage, streaming mismatch 0

각 사전 잠금 campaign의 `outer_reveal_budget`은 정확히 `1`이다. 공개 전 source,
architecture, seed, split, optimizer updates, selector, calibrator, threshold, fallback,
분모, gate를 hash로 잠근다. 공개 후에는 해당 outer 결과를 이용한 hypothesis,
threshold, ensemble, checkpoint, exclusion rule 선택을 금지한다. 실패한 prediction,
config, checkpoint, failure reason은 그대로 동결한다. 같은 18명 cohort에서 이후 수행하는
원인 분석은 retrospective discovery로만 분류하며 confirmatory independence를 재설정하지
않는다.

배포용 최종 교정은 순서를 바꾸지 않는다. Common model을 먼저 development cohort에
fixed-update refit하고 weight hash를 봉인한 뒤, development와 겹치지 않는 untouched
prospective calibration cohort에 한 번 예측한다. Calibration 성능은 identity cross-fit으로
평가하고, 최종 calibrator/threshold는 calibration cohort 전체로 한 번 fit해 봉인한다.
그 뒤 model을 다시 refit하지 않으며 calibration identity도 model weight 학습에 넣지
않는다. Confirmation cohort는 model·calibrator·threshold 모두에 완전히 untouched다.

### G9 — 내부 full OOF 평가

G1–G4가 끝난 뒤 authoritative valid-reference rowset을 원자료에서 다시 만들고, row
key·identity·physical session·half-open start/end·reference-QC·target binding을 manifest로
봉인한다. 최종 row count와 SHA-256은 지금 추정하거나 2,327로 강제하지 않는다. 둘 다
`must_be_frozen_before_prediction`이며 prediction pack 생성 전에 확정돼야 한다.

기본 co-primary는 다음 둘이다. 고정 seed마다 둘 모두 위 여섯 accuracy gate를 통과해야
하며 한 모집단의 통과로 다른 실패를 평균하거나 대체하지 않는다.

1. `AUTHORITATIVE_FULL`: 새 authority가 확정한 valid-reference row 전체. 겹치는
   32초 window의 상관을 명시하고 finite prediction 100%를 요구한다.
2. `AUTHORITATIVE_GREEDY_NONOVERLAP_32S`: `physical_source_session_id`별로만
   grouping한다. 각 session에서 `(window_start_ns, window_end_ns, stable_row_key)`를
   오름차순 정렬하고 첫 row를 고른 뒤 `next.start_ns >= previous.end_ns`인 가장 이른
   row를 반복 선택한다. Session 경계를 넘어 상태나 마지막 end를 이어가지 않는다.

기존 `FROZEN_FULL_2327`은 새 authoritative rowset과 stable key, physical session,
identity, interval, reference-QC, target가 exact bijective crosswalk일 때만 추가 legacy
co-primary다. Crosswalk row count, ordered keys, content hash가 하나라도 다르면
`historical_comparator_only`로 강등하며 새 `AUTHORITATIVE_FULL`을 대체하지 못한다.

추가 시간축 gate는 각 physical source session에서 authoritative cache producer가
기록한 원래 session-local `window_number`에 대해 `window_number mod 8 = p`, `p=0..7`을
적용한다. `window_number`는 row 필터 후 재번호를 붙이지 않으며 producer source/config,
base stride, session start와 함께 provenance에 묶는다. 각 phase는 session별 32초
비중첩 표본이다. Greedy/phase selector code hash, input rowset hash, ordered selected row
keys, count와 selection-manifest hash는 target/prediction 공개 전에 모두
`must_be_frozen_before_prediction` 상태에서 실제 값으로 원자 동결한다.

Retrospective support floor는 `AUTHORITATIVE_FULL`과 greedy 모집단 각각 usable physical
identity 18명, usable source session 29개 exact cover, 25–35 bpm에서 최소 6 identities와
greedy non-overlap 48 rows다. 각 stride phase는 전체 최소 12 identities, 25–35 bpm 최소
4 identities와 16 rows를 요구한다. 미달은 통과가 아니라 `insufficient_support`다.

35–45 bpm은 여섯 accuracy gate와 별도의 promotion safety gate다. Authoritative full과
greedy 각각 최소 3 identities와 greedy non-overlap 24 rows를 확보하고, 각 고정 seed의
`35–45 MAE ≤3.0 bpm` 및 `abs(error)>5 fraction ≤0.05`를 모두 만족해야 한다. 지원량이
부족하면 모델이 좋아 보이더라도 R1 승격을 보류하고 prospective sampling으로 보충한다.

Primary 공통 증거:

- 고정 seed별 여섯 accuracy gate
- fixed seed exact set `20260828/20260829/20260830`
- full denominator, fallback count, no-estimate/unavailable count와 prediction coverage
- authoritative/co-primary/phase row exact cover와 selector source/hash
- identity-macro와 identity-cluster bootstrap 95% CI

필수 strata:

- identity 18, fold 6, session 29
- RR: 6–10/10–15/15–20/20–25/25–35/35–45
- RR stratum 경계: `[lower, upper)`, 마지막 35–45만 `[35, 45]`. 정확히 10, 15,
  20, 25 bpm인 row를 중복 집계하지 않는다. 별도 six-gate의 25–35와 promotion
  safety의 35–45 interval은 각 gate 정의대로 양 끝을 포함하므로 35 bpm row가 두
  안전 판정에 함께 들어가는 것은 의도한 보수적 중복이다.
- phase 1–7, transition, breath-hold, recovery, phase-7 action
- radar masks `123/12/13/23/1/2/3`
- 8 fixed stride phases와 deterministic greedy non-overlap
- cold start/steady/switch episode

Tail/safety:

- median/P90/P95/P99 absolute error
- bias, CCC, >2/>5 AUROC·AUPRC
- per-identity and worst-group catastrophic rate
- candidate coverage, oracle-only diagnostic, routing regret
- no-estimate/fallback/unavailable를 전체 분모로 보고

Calibration:

- outer-test 이전 inner/held-identity cross-fit threshold fitting
- 50/70/80/90/95% interval coverage와 width
- risk–coverage curve, ECE/interval score
- selective subset를 full-coverage 성능으로 표현 금지

### G10 — 배포·prospective confirmation

아래 수치는 규제 또는 의료 안전 보장이 아니라 사전 선언한 **내부 engineering release
criteria**다. 데이터 수집 전 실제 identity-cluster 분산, band prevalence, dropout을 넣은
simulation-based power 분석으로 cohort를 늘릴 수는 있지만 줄일 수 없다. 각 prospective
cohort는 여섯 gate의 최소 power가 `≥0.90`, one-sided alpha `0.05`가 되도록 collection
전에 표본 수를 동결한다.

#### R1 — internal engineering pass

- G0–G9 완료, 세 고정 seed가 모든 required population/phase에서 여섯 inclusive point
  gate와 35–45 safety gate 통과
- calibration: identity-cross-fit ECE `≤0.05`(10 equal-mass bins), nominal
  `50/70/80/90/95%` interval 각각 `abs(empirical-nominal) ≤0.05`; full denominator 사용
- seven radar masks `123/12/13/23/1/2/3`: deployable source가 하나 이상인 fault case의
  estimate availability `≥0.99`, fallback activation success `≥0.99`, spurious fallback
  fraction `≤0.01`; all-source-missing은 quality exact zero와 safe no-estimate `100%`
- fault type/device/mask마다 최소 100 injections, crash `0`, silent state corruption `0`,
  fault-detection recall `≥0.99`, 정상 입력 false alarm `≤0.01`, recovery `≤8 s`
- target device end-to-end warm p95 `≤250 ms`, p99 `<4,000 ms`, peak host RAM `≤1 GiB`,
  peak VRAM `≤2 GiB`, parameter count `≤400,000`; cold p50/p95/p99도 별도 보고

#### R2 — prospective calibrated

- development와 겹치지 않는 최소 40 identities/80 sessions, 2 sites, 3 physical radar
  sets, 10 collection days의 prospective calibration cohort
- independent reference와 blinded QC operator, measurable clock/hardware trigger;
  unplanned exclusion `≤10%` of enrolled sessions
- intended RR 6–45 bpm의 여섯 band 각각 최소 10 identities/40 greedy-nonoverlap rows;
  25–35는 최소 20 identities/160 rows, 35–45는 최소 12 identities/80 rows
- source/model weight를 먼저 봉인하고 prediction을 한 번 생성. Identity-cross-fit
  calibration 평가 후 final calibrator/threshold를 전체 calibration cohort로 한 번 fit·봉인
- identity를 cluster로 10,000회 resample하는 사전 고정 bootstrap seed의 one-sided 95%
  bound: lower-is-better 다섯 metric은 upper bound가 threshold 이하, `within_2`는 lower
  bound가 `0.90` 이상. 여섯 조건 모두 충족
- reference-valid full denominator prediction coverage `1.0`; fallback 포함, no-estimate
  `0`. Enrollment→acquisition→reference-QC→estimate/fallback/no-estimate funnel과 모든
  predeclared exclusion reason/count 보고
- cross-fit ECE `≤0.05`, 각 nominal interval coverage gap `≤0.05`, interval width와
  interval score 전체 공개

#### R3 — commercial release candidate

- calibration/development와 겹치지 않는 최소 100 identities/200 sessions, 3 sites,
  5 physical radar sets, 3 hardware lots, 20 collection days의 prospective confirmation
  cohort; 수집 전 joint six-gate power `≥0.90`
- 각 6–45 bpm band 최소 25 identities/100 greedy-nonoverlap rows; 25–35 최소
  40 identities/320 rows, 35–45 최소 25 identities/160 rows
- model/calibrator/threshold/fallback/exclusion/analysis hash 공개 전 동결. Confirmation
  target는 단 1회 join하고 model 또는 calibrator refit 금지
- 위와 동일한 one-sided identity-cluster bootstrap 10,000회 95% bound가 여섯 accuracy
  threshold를 모두 만족하고, 35–45 safety gate도 point와 upper bound 모두 통과
- full denominator prediction coverage `1.0`, unplanned session exclusion `≤5%`, ECE
  `≤0.03`, 각 50/70/80/90/95% interval coverage gap `≤0.03`
- seven masks와 gain/offset/impulse/burst/jitter/gap/range-shift/phase-inversion/flatline,
  restart/out-of-order/duplicate fault를 target-device unit마다 각 100회 주입. Crash/state
  corruption `0`, eligible-source availability/fallback `≥0.995`, fault recall `≥0.99`,
  false fallback `≤0.01`, recovery `≤8 s`
- confirmation hardware에서 warm p95 `≤250 ms`, p99 `<4,000 ms`, RAM `≤1 GiB`, VRAM
  `≤2 GiB`를 모두 충족
- shadow 최소 30 consecutive days/20 device-units/10,000 eligible windows,
  pipeline availability `≥0.995`, crash `0`; canary 최소 14 days/5 device-units/5,000
  eligible windows. Rollback drill 최소 3회, trigger detection `≤60 s`, signed rollback
  완료 `≤5 min`, model/state schema corruption `0`

이 최소 수와 조건은 달성 수치가 아니며 현재 증거는 전부 부재다. 조건을 만족해도
`R3`는 내부 commercial release candidate일 뿐 의료기기 허가, 임상 유효성 또는 규제
승인을 뜻하지 않는다.

## 5. 우선순위와 계산 자원 배분

우선순위 점수:

```text
(예상 gate 개선 × 근거 강도 × 여러 fold 재사용성)
÷ (계산 비용 × leakage 위험 × provenance 복잡도)
```

현재 순서:

1. G2–G4 authority 결함 폐쇄: 잘못 정렬·잘못 block화된 데이터로 학습하지 않기 위함.
2. G5 mask/cache strict closure: missing 구조와 available zero를 분리.
3. G7 J0/J1: 확인된 routing 표현·gradient 결함 직접 수정.
4. G8 fixed-update discovery: 너무 적은 optimizer update 문제 제거.
5. J2/J3: high-RR tail과 expert switching 안정화.
6. teacher/self-supervision/range feature: 앞 단계가 깨끗할 때만 추가.
7. full 3-seed OOF와 배포 suite.
8. prospective data 수집·확증.

GPU가 없는 host에서는 source/unit/CPU smoke/provenance까지만 수행한다. expensive training이나 CUDA latency를 CPU 수치로 대체하지 않는다.

## 6. 실행 authorization과 artifact 규칙

모든 material run은 다음을 원자적으로 기록한다.

- campaign/version/hypothesis ID
- source/config/data/cache/split/interpreter/dependency/device SHA-256
- physical identities per train/validation/test
- fixed seeds와 RNG state
- exact command, start/end, exit/signal/partial state
- optimizer updates, processed windows, wall/CPU/GPU usage
- checkpoint/prediction/metric hash
- target access phase와 sealed join receipt
- success/failure/blocked classification

새 V8R5 source와 config는 구현 증거일 뿐 training authorization이 아니다. 현재
V8R4 CONTEXT1 receipt trio도 absent 상태다. 더 나아가 이 source generation에는
독립적으로 관리되는 외부 test issuer/runner trust root와 signature verifier 자체가
없다. Local/self-hashed receipt나 constant 변경으로 발급할 수 없으며, 새 governed
trust-root generation 전에는 protected production training을 시작하지 않는다.

## 7. 즉시 중단이 아니라 fail-closed 전환하는 조건

- source/config/cache hash drift → 새 version/context 발급
- identity/interval leakage → run 무효화·보존, split 재구축
- unavailable nonzero 또는 target field 노출 → 학습 전 hard error
- NaN/Inf/streaming mismatch → 후보 승격 금지
- 한 seed/fold만 유리 → 공통 후보 실패
- authorization absent → source/test까지만 진행, training 대기
- CUDA absent → CPU correctness까지만, CUDA evidence 미완료
- external cohort absent → 내부 pass 가능, 상용 완료 불가

중단은 Goal 종료가 아니라 허용 경계까지 완료한 resume-ready 상태다. 차단 사유, 필요한 외부 입력, 재개 명령, immutable hash를 남긴다.

## 8. 완료 체크리스트

- [ ] G0 exact recovery/environment closure
- [ ] G1 raw/cohort authority exact cover
- [ ] G2 measured causal timing full usable cohort
- [ ] G3 raw-recomputed synchronization authority
- [ ] G4 source-recomputed seven-phase blocks
- [ ] G5 strict RF/SVD/mask cache
- [ ] G6 nested identity-disjoint/sealed stack
- [x] V8R5 axis-risk model source + synthetic correctness tests
- [ ] V8R5 authorized fixed-update discovery
- [ ] common architecture/release lock
- [ ] 6 folds × 3 seeds sealed full OOF
- [ ] every seed passes all six accuracy gates
- [ ] minimum identity/high-RR support and 35–45 safety gate
- [ ] 7 masks/non-overlap/8 phases/fault suite
- [ ] held-identity calibration and selective-risk gate
- [ ] streaming/CPU/CUDA/target-device gate
- [ ] independent prospective calibration
- [ ] independent prospective confirmation
- [ ] precollection power and one-sided cluster-bootstrap bounds
- [ ] shadow/canary/rollback validation

현재 정확한 판정: `R0_RESEARCH_ACTIVE`. 내부 상용 정확도 0/6, sync/stage authority 미완료, V8R5 실데이터 미측정, prospective evidence 없음.

## 9. 2026-08-31 전면 개선 감사와 폐쇄 상태

이번 source generation은 성능 수치를 새로 만들기 전에 학습 입력과 실행 권한의
거짓 양성을 막는 항목을 우선 폐쇄한다. 완료 표시는 코드와 회귀 test가 있다는
뜻이며, scientific training authorization이나 정확도 향상을 뜻하지 않는다.

| 우선순위 | 확인된 결함 | 적용한 개선 | 현재 상태 |
|---|---|---|:---:|
| P0 | acquisition diagnostic cache가 일반 trainer로 진입 가능 | RF/SVD trainer에서 inspection-only 강제, scaler·CUDA·output 전에 거부 | 완료 |
| P0 | import 가능한 V3R1 `train`/`predict`가 caller 검증값을 신뢰 | 각 entry에서 admitted context·phase·capability·authorization을 독립 재검증 | 완료 |
| P0 | custom split prediction에서 timing mask/provenance 소실 | mask-aware history, unavailable exact-zero, cache/source 재해시와 publication binding | 완료 |
| P0 | sync motion marker가 invalid resample zero 경계를 사용할 수 있음 | radar별 validity 전달, adjacent/smoothing support dilation, invalid marker 억제 | 완료 |
| P0 | V8R4 CONTEXT1 validator가 self-issued test 증거를 수용할 위험 | exact node inventory를 고정하고 독립 trust root/verifier 부재를 terminal fail-closed로 선언 | 완료·새 governed generation 필요 |
| P0 | harmonic cache가 numeric zero에서 availability를 재추론할 위험 | canonical RF/SVD timing mask를 manifest/hash/shape에 결합하고 masked cell exact-zero 강제 | 완료 |
| P0 | arbitrary proposer NPZ가 target leakage를 숨길 수 있음 | 현 builder output을 모두 inspection-only로 분류하고 versioned nested label-free proposer authority 없이는 trainable 승격 금지 | 완료·authority 설계 대기 |
| P0 | harmonic trainer가 diagnostic/legacy cache를 학습할 수 있음 | manifest content hash·classification·`trainable`을 RNG/output 생성 전에 독립 검증 | 완료 |
| P0 | unavailable harmonic feature가 robust scaling 후 nonzero로 부활 | structural availability로 feature별 fit, transform 후 unavailable exact `+0.0` 재적용 | 완료 |
| P1 | sync ambiguity 검사가 첫 alternative에서 멈춤 | 모든 분리 후보 중 동등 품질의 위험 alternative 탐색 | 완료 |
| P1 | YAML bool/NaN/Inf/분수 counter가 gate 숫자로 통과 가능 | sync/protocol/timing 입력 exact-type·finite·int64 검증 | 완료 |
| P1 | stage document의 overall 값과 phase 배열이 불일치 가능 | 고정 7-phase order/name exact 검증, worst-status·mean-confidence 재도출 | 완료 |
| P1 | legacy manual approval이 reviewer 문자열만으로 권한 부여 | 승인 승격 금지, 자동 승인에 대한 rejection override만 유지 | 완료 |
| P0 | public stage assignment가 contract-level stage authority 없이 `auto`만으로 metric eligibility를 발급 | `contract.stage_metric_eligible AND stage.status==auto` exact 결합, diagnostic contract 회귀 | 완료 |
| P0 | caller metadata/DataFrame 교체로 physical identity·reference-valid row role을 split에 이식 가능 | authoritative metadata same-inode bytes, ordered row-role hash, exact issued object/type/receipt registry 결합 | 완료 |
| P1 | custom split JSON과 checkpoint/config provenance가 실제 사용 bytes와 달라질 수 있음 | same-inode byte snapshot에서 parse+hash, split/cache/source publication barrier | 완료 |
| P1 | derived/cache mmap input이 검증 뒤 실행 중 교체될 수 있음 | same-inode stable byte를 private read-only snapshot으로 복사하고 원본 namespace drift 재검증 | 완료 |
| P1 | import 전 code를 교체한 뒤 disk를 복원하면 post-hoc source hash와 실제 compiled loader가 다를 수 있음 | manifest에 compiled-byte 비결합을 명시하고 externally owned source snapshot→fresh isolated child launcher 전 production authority 차단 | 부분·runtime gate 필요 |
| P1 | harmonic cache v1에 새 per-feature mask 의미를 덧씌울 위험 | format v2 + schema ID + layout source/digest + exact output inventory | 완료 |
| P0 | SVD receipt caller flag, subclass 또는 복제 object가 scientific authority를 이식할 위험 | loader-only exact-object receipt registry, method identity, acquisition-v2 fresh replay | 완료 |
| P0 | base CSV의 임의 feature와 temporal completed-fold artifact가 target 또는 다른 fold 결과를 이식할 위험 | NPZ-bound feature allowlist, cache-index/target/fold/run/source/split/checkpoint exact binding | 완료 |
| P1 | non-v2 SVD가 새 scientific training처럼 실행될 위험 | 명시적 legacy flag + 최초 생성 versioned root + `historical_noncommercial` 분류 | 완료 |
| P1 | custom output이 input을 alias하거나 concurrent/symlink 경로를 덮어쓸 위험 | full input/output disjoint guard, private fd temp, no-clobber atomic publish, fsync/rollback | 완료 |
| P0 | V3R1 completed run/prediction과 same-unit checkpoint를 다른 authorization·fold·source에 이식 가능 | sealed checkpoint capability, exact run/cache/scaler/predict-input signature와 return 직전 binding 재검증 | 완료 |
| P0 | V3R1 cache/checkpoint 검증 뒤 pathname 재개방과 unsafe pickle load | exact private consumed-byte snapshot, `BytesIO + weights_only=True`, publish/reuse ABA barrier | 완료 |
| P1 | 실패 temp 이름만 보고 caller 파일을 stale artifact로 삭제 가능 | cross-invocation 자동 삭제 금지, explicit quarantine fail-closed | 완료 |
| P0 | V8R5 available expert의 nonfinite state가 deployable NaN으로 전파될 수 있음 | parameter·buffer·intermediate·streaming-state finite firewall, learned-source 제거 후 classical fallback 또는 unavailable exact-zero | 완료·미측정 |
| P1 | V8R5 checkpoint가 mutable runtime behavior와 dependency provenance를 exact 결합하지 못함 | canonical persistent uint8 receipt, runtime/buffer/checkpoint 삼자 검증, transactional strict preflight, `assign=False` | 완료·미측정 |
| 구조 | coordinate/evidence pooling 대응 소실 | evidence와 radar/ratio/branch/candidate 좌표를 pooling 전에 joint MLP로 결합 | 완료·미측정 |
| 구조 | hard routing과 risk loss 사이 gradient 단절 | disjoint value/route/risk heads, detached representation, soft expected deployment cost; hard 선택은 inference 전용 | 완료·미측정 |
| 구조 | stale finite classical RR가 availability로 오인될 위험 | explicit boolean availability, no-source quality exact zero, classical-only quality supervision | 완료·미측정 |

아직 source만으로 폐쇄할 수 없는 항목:

- sync receipt의 bound raw signal 독립 재계산 verifier
- spreadsheet/RSP/config에서 7-phase decoder를 독립 재실행하는 authority
- 새 governed CONTEXT source generation, 외부 issuer/runner trust root와 signature verifier
- externally owned source snapshot을 fresh isolated child에서만 import하는 production
  launcher와 actual compiled-loader provenance
- raw XeThru/BIOPAC signal의 manifest hash와 실제 장시간 memmap/load bytes를 동일한
  stable descriptor에 종단 결합하는 source-consumption runtime
- 10개 timing exception의 독립 판정과 29-session strict cache 재구축
- authorized GPU fixed-update discovery, 3 seeds × 6 folds full OOF
- prospective calibration/confirmation cohort와 target-device fault campaign

따라서 이 개선 세대의 올바른 결과는 더 강한 fail-closed 경계와 측정 준비 완료다.
새 성능 수치나 상용 gate 통과 수는 발생하지 않았다.
