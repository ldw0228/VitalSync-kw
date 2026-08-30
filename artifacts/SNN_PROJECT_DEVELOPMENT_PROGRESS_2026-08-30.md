# SnnProject 개발 진행상황 통합 요약

- 기준일: 2026-08-30, Asia/Seoul
- 대상: 3대 XeThru UWB radar 기반 호흡수 추정 hybrid/SNN 연구 시스템
- 현재 상태: `ACTIVE_RESEARCH_CYCLE`
- 증거 수준: 18명 retrospective cohort의 identity-disjoint 연구 결과
- 제품 상태: 상용·의료 제품 아님, 여섯 상용 정확도 gate 중 통과 `0개`

이 문서는 데이터 취득 구조 파악, parser·동기화·block 구성, 전처리, SNN 모델, 학습, 평가, 실행 무결성, 복원·백업까지 현재 개발 상태를 한 파일로 정리한 기준 문서다. **측정 완료**, **구현됐지만 미승인**, **설계 제안**, **외부 검증 필요**를 구분한다.

> 가장 중요한 결론: 현재 최고 full-coverage OOF는 MAE `1.291 bpm`이지만 여섯 내부 상용 정확도 기준을 모두 실패했다. 최신 acquisition-aware 재구성은 아직 전체 29개 usable session에서 승인되지 않았으므로, 기존 성능은 교정된 전체 데이터의 최종 상용 성능으로 해석할 수 없다.

## 1. 상태 범례

| 상태 | 의미 |
|---|---|
| **완료·측정** | 구현과 지정 평가가 끝나 결과 artifact 존재 |
| **완료·실패** | 사전 기준대로 평가 완료, 기준 미달 결과 보존 |
| **구현·미승인** | 코드와 일부 test/smoke 존재, 전체 과학 실행 허가 또는 full OOF 없음 |
| **부분·미완료** | 일부 범위만 완료됐거나 필수 후속 산출물 부재 |
| **차단·미해결** | 사전 조건 또는 재현성 문제가 해결되기 전 실행 금지 |
| **제안** | 원인 분석에 기반한 차기 구조, 아직 실측 결과 없음 |
| **외부 필요** | 새 cohort·target device 등 현재 저장소만으로 완료 불가 |

## 2. 한눈에 보는 현재 상태

| 영역 | 현재 상태 | 판정 |
|---|---|---|
| Raw parser | 740-byte record를 header와 182-float payload로 분리, 예외 처리·감사 구현 | **완료·측정** |
| 물리적 identity 정리 | 30 folders → 29 usable sessions → 18 identities | **완료·측정** |
| 기존 canonical dataset | 9,576 windows, reference-valid 2,327 | **완료·측정** |
| 최신 measured timing·sync | affine sync, timestamp repair, receipt, fail-closed gate 구현 | **구현·미승인** |
| 실험 stage/block | 7개 phase decoder와 manual anchor 경로 구현 | **구현·미승인** |
| 기존 preprocessing | causal repair, RF map, SVD, reference QC | **완료·측정** |
| acquisition-aware strict full cache | 전체 session 승인 뒤 다시 생성해야 함 | **미완료** |
| Structured TriRadarRRSNN | full identity OOF와 2-model locked ensemble 완료 | **완료·측정** |
| HCES v2 | 3 seeds full locked OOF 완료, 모두 gate 실패 | **완료·실패** |
| DHFER v3r1 | 구조·trainer·일부 discovery validation 구현, full OOF 없음 | **구현·미승인** |
| CCHG-SNN/V8R5 | 좌표 보존·soft-risk router 차기 구조 | **제안** |
| 평가 체계 | identity OOF, non-overlap, masks, streaming·provenance 체계 존재 | **부분 완료** |
| V8R4 실행 closure | CONTEXT1 receipt/snapshot/authorization 부재 | **차단** |
| test snapshot | 694 collected, 688 pass, 4 skip, 2 fail | **미해결** |
| prospective 상용 검증 | 독립 cohort/reference/device 검증 없음 | **외부 필요** |
| 복원 백업 | source·raw ZIP·selected evidence·V8R4 state 복원 세트 생성 | **완료** |

## 3. 실제 데이터 취득 구조와 데이터 현황

### 3.1 취득 구조

```text
피험자 1명
├─ XeThru radar 1 ─┐
├─ XeThru radar 2 ─┼─ 같은 실험 session의 3-view radar stream
├─ XeThru radar 3 ─┘
└─ BIOPAC RSP ─────── offline reference·label·동기화 감사 전용

실험 흐름
→ 앉은 자세·호흡/자세
→ 정상/느린 호흡·숨참기·운동 후 회복
→ 물건 집기 코스
→ 낙상 시나리오
→ 16-cell 코스
→ 연속 왕복
→ 배정 동작(Dodge/Strike/Kick)
```

BIOPAC은 추론 입력이 아니다. label 생성, 허용된 supervised training split, 최종 sealed 평가 join에서만 사용한다. 실험 stage도 annotation·분석용이며 inference feature로 사용하지 않는다.

### 3.2 규모

| 항목 | 값 |
|---|---:|
| 원본 session folders | 30 |
| usable 3-radar+BIOPAC sessions | 29 |
| 물리적 identities | 18 |
| 선택된 3-view radar frame records | 4,715,613 |
| radar duration | 약 10.916 h |
| paired BIOPAC duration | 약 11.001 h |
| 32초 후보 windows | 9,576 |
| reference-valid windows | 2,327, `24.30%` |

### 3.3 RR 분포

| Reference RR | valid windows |
|---|---:|
| 6–10 bpm | 202 |
| 10–15 bpm | 852 |
| 15–20 bpm | 742 |
| 20–25 bpm | 361 |
| 25–35 bpm | 164 |
| 35–45 bpm | 6 |

25–35 bpm은 164개뿐이고 greedy non-overlap에서는 42개다. 고호흡수 성능 저하는 모델의 harmonic 혼동과 작은 tail support가 함께 만든 문제로 본다.

### 3.4 확정된 원본 예외

- `S24_KHJ`: 세 radar stream이 비어 제외
- `S01_CMS`: retry가 아닌 세 radar 공통 길이가 가장 긴 3-chunk recording 선택
- `S07_KDM`: timestamp counter reset을 deterministic 방식으로 복구
- `S17_RJS`: physical identity를 `PJS`로 교정
- `S22_KJH`: radar 2 outlier 한 개를 past-only sample로 복원
- RSP rail clipping: 전체 약 `5.22%`, `S30_SJE` 약 `43%`, `S06_LDW` 약 `16%`
- radar 간 start spread 최대 `16 ms`, frame-count spread 최대 1
- radar/BIOPAC 공통 hardware trigger 없음, residual sync uncertainty 유지

## 4. 데이터 처리 파이프라인 진행상황

```text
Raw read-only files
  ↓
Typed parser + integrity audit                         [완료]
  ↓
Measured radar time + plateau/reset repair             [구현·smoke]
  ↓
Radar↔BIOPAC affine synchronization + receipt          [전체 미승인]
  ↓
7-phase protocol block reconstruction                  [구현·진단, 전체 미승인]
  ↓
Causal radar denoising + RF/SVD/range evidence         [기존 경로 완료]
  ↓
BIOPAC reference QC                                    [기존 경로 완료]
  ↓
Physical-identity split                                [완료]
  ↓
Strict acquisition-aware cache                         [미생성]
  ↓
SNN training·OOF evaluation                            [기존 cache 결과만 완료]
```

### 4.1 Raw parser와 감사

- 실제 radar record: `3×uint32 header + 182×float32 payload = 740 bytes`
- legacy 185-float 해석: header 오염 가능성으로 배제
- 검사 항목: record remainder, header, bin count, sequence gap, NaN/Inf, amplitude, metadata frame count·timestamp
- raw 파일: rename·rewrite 없이 read-only 처리
- acquisition 관련 focused tests: 35개 통과 기록

### 4.2 시간축과 sensor 동기화

- radar time: metadata v13 measured timestamp 우선
- plateau repair: `S03`, `S28`, `S30`; `S30` 최대 보정 약 `203.7 ms`, manual review 대상
- sync 모델: `t_rsp = offset + scale × t_radar`
- 승인 조건: residual·confidence·ambiguity gate와 immutable receipt
- 최신 source-consistent smoke: `S02_RJS`, `S03_PSJ`, `S30_SJE` 3개 모두 usable, sync authorized `0/3`, scientific eligible `0/3`
- 위 smoke의 `complete=true`: 선택한 3-session 실행이 끝났다는 뜻이며 full cohort 완료 또는 과학 적격을 뜻하지 않음
- 이전 full diagnostic: 30 sessions 중 29 usable, sync authorized `0/29`, scientific eligible `0/29`; 현재 source와 hash가 다른 진단 artifact

따라서 **모든 피험자의 정확한 offset과 block이 확정됐다고 볼 수 없다**. 최신 전체 strict reconstruction과 승인된 sync receipt가 먼저 필요하다.

### 4.3 실험 block 구성

| Phase | 내용 | 사용 방식 |
|---:|---|---|
| 1 | 각도별 앉은 자세·호흡/자세 | stage별 평가 |
| 2 | 정상/느린 호흡, 숨참기, 운동 후 회복 | 현재 coarse phase, 향후 substage 분리 평가 |
| 3 | 물건 집기 2개 코스 | motion robustness |
| 4 | 낙상 2개 코스 | motion/fault robustness |
| 5 | 16-cell timed course | 위치·동작 변화 평가 |
| 6 | 연속 왕복 | 연속 streaming 평가 |
| 7 | 배정 동작 Dodge/Strike/Kick | 향후 승인 block의 protocol별 평가 |

Stage decoder는 spreadsheet anchor, BIOPAC-derived marker, phase 순서·예상 길이·gap을 결합한 ordered dynamic programming이다. Radar motion marker는 radar↔BIOPAC sync proposal에 사용되며 stage label의 직접 근거가 아니다. 명시적인 spreadsheet 구간은 주로 `S02`, `S03`에만 있어 나머지 usable sessions는 상당수가 candidate/uncertain 상태다.

Corrected full dataset에서 적용할 평가 원칙이며, 현재 전체 artifact로 구현·확정된 상태는 아니다.

- `core`: 신뢰 가능한 stage 내부 window
- `transition`: stage 경계를 걸치는 window
- `uncertain/review`: 경계 또는 sync가 승인되지 않은 window
- `breath-hold`: 향후 독립 substage label을 만든 뒤 유효 RR을 강제로 만들지 않고 별도 상태/coverage로 관리
- stage·BIOPAC annotation: 학습 시 허용된 label/QC 또는 평가용, inference feature 금지

### 4.4 Radar 전처리와 denoising

기존 nominal-time 경로:

1. 40 Hz radar를 non-overlap 4-frame mean으로 10 Hz 변환
2. 32초, 320-sample window; stride 4초, overlap 87.5%
3. `|x| > 0.1` outlier를 같은 bin의 과거 최대 4개 정상값 median으로 대체
4. `[-0.05,+0.05]` clip, mean 제거, linear detrend, Hann taper
5. 2,048-point rFFT, 0.08–0.80 Hz band 선택
6. frequency smoothing, range pooling, median noise-floor normalization, `log1p`
7. RF map `[N,3,73,182]`과 aux `[N,1205]` 생성

추가 표현:

- label-free randomized SVD: raw·velocity·range-difference view, 12 components, Hann+4096 FFT
- causal active-range tracker: bin·confidence·missing·multimodal·evidence 출력
- range tracker는 사람의 신원·미터 단위 거리·3D 위치를 직접 추정하지 않음
- 현재 `range_aux`는 model feature에 아직 연결되지 않음
- split-half I/Q branch는 장비 명세로 확정되지 않아 최신 harmonic path에서는 zero/masked 처리

중요한 구현 차이:

- 새 measured timestamp는 현재 mapping·window metadata에 쓰이지만, `build_features.py`의 신호 resampling은 아직 4-frame mean과 고정 10 Hz FFT를 사용
- strict metadata 모드가 nominal 40 Hz로 fallback할 수 있음
- strict acquisition/scientific cache 검사는 opt-in이며 기본 `train.py` 경로가 강제하지 않음
- corrected strict full cache와 그 cache의 full OOF는 아직 없음

### 4.5 BIOPAC reference

- RSP 전체에 4차 Butterworth 0.10–0.75 Hz zero-phase filter 적용, label 경로 전용
- FFT peak, peak-to-peak IBI, Hilbert phase slope의 median을 target로 사용
- 최소 cycle, clipping, plateau, spectral concentration, autocorrelation, interval CV, estimator disagreement, phase residual, guard QC 적용
- `radar_observable`: reference와 classical radar error를 함께 사용한 target-dependent proxy이므로 inference feature 금지

## 5. 모델 개발 계보

### 5.1 Structured TriRadarRRSNN — 현재 재현 가능한 leader

```text
3 radar × range–frequency map
→ shared 2-D encoder
→ range attention + radar reliability
→ frequency-preserving PLIF/LIF Conv1D blocks
→ 12 internal simulation steps
→ RR posterior + expected RR + uncertainty + quality
→ validation-locked two-component ensemble
```

학습 요소: ANN teacher distillation, identity-balanced sampling, coupled radar dropout, causal history. 현재 최고 full OOF는 structured auxiliary와 exact auxiliary-alignment SNN을 outer-validation에서만 잠근 ensemble 결과다.

### 5.2 HCES v2 — candidate graph full OOF 완료, gate 실패

- 최대 12개 RR 후보를 graph node로 구성
- near와 ×2/×3/×4 harmonic 관계
- residual PLIF graph blocks, masked attention/mean pooling
- causal PLIF→ALIF episode encoder
- 3 fixed seeds의 locked full OOF 완료
- candidate oracle은 우수했지만 deployable router가 후보를 안정적으로 선택하지 못함

### 5.3 DHFER v3r1/V8R4 — 구현됐지만 full OOF 미완료

- candidate node feature: `571 = core 46 + RF 378 + SVD 147`
- 7개 directed relation: near, ×2/×3/×4 양방향
- hidden 64, graph blocks 2개, PLIF graph + causal PLIF→ALIF router
- anchor/candidate hard expert, H0/H1/H2 variants
- parameter 수: 203,669
- 측정 범위: H0의 outer 3 → validation fold 4 한 unit

확인된 구조 결함:

1. `f(evidence)+coordinate embedding → mean` pooling이 좌표와 evidence 대응을 잃음
2. hard argmax 뒤 CVaR/tail-risk loss가 routing choice에 gradient를 전달하지 못함
3. confidence threshold 0.8과 실제 selected probability 0.38–0.50이 불일치
4. 약 235 optimizer updates로 학습량 부족

### 5.4 CCHG-SNN/V8R5 — 현재 권고 차기 구조, 아직 미측정

```text
RF/SVD/range evidence
→ shared analog spatial encoder + direct RR posterior
→ evidence×radar×ratio×branch coordinate interaction
→ directed harmonic candidate graph
→ 8–12 step PLIF/ALIF temporal SNN
→ differentiable expected-risk/tail-risk router
→ hard candidate / direct anchor / no-estimate
```

선정 이유:

- dense radar 입력: analog front-end 유지
- 18명 규모: compact SNN으로 parameter 수 제한
- high-RR harmonic 혼동: direct path와 ×1–×4 candidate path 병행
- 기존 실패 원인: pooling 전에 coordinate와 evidence를 비선형 결합
- 위험한 hard correction: soft-risk training, hard-safe inference, direct fallback
- causal 연속 추론: PLIF/ALIF state와 session reset

이 구조는 원인에 맞춘 제안이며 성능 수치가 없다. 구현·fixed-seed OOF 전에는 leader로 부르지 않는다.

## 6. 실험 결과와 의사결정

### 6.1 주요 모델 결과

| 계열 | 평가 범위 | Overall MAE | Macro MAE | 25–35 MAE | 판정 |
|---|---|---:|---:|---:|---|
| Flat/default SNN | full OOF | 1.407 | 1.360 | — | baseline |
| Structured auxiliary SNN | full OOF | 1.333 | 1.257 | 3.976 | 개선 |
| Structured exact SNN | full OOF | 1.347 | — | 4.485 | ensemble 구성원 |
| Structured+exact ensemble | full OOF | **1.291** | **1.220** | **4.216** | 현재 leader, gate 실패 |
| Alias-gated harmonic SNN | full OOF | 1.351 | — | 4.049 | 기각 |
| Causal alias decoder | full OOF | 1.830 | — | — | 기각 |
| SVD source SNN v1 | locked full OOF | 1.294 | 1.221 | — | leader 미개선 |
| HCES v2 seed 20260828 | locked full OOF | 1.296 | 1.263 | 3.567 | 실패 |
| HCES v2 seed 20260829 | locked full OOF | 1.372 | 1.329 | 4.073 | 실패 |
| HCES v2 seed 20260830 | locked full OOF | 1.377 | 1.357 | 3.696 | 실패 |
| DHFER H0 hard source | discovery validation 1 unit | 1.769 | 1.740 | 4.752 | full OOF 아님 |

RR-balanced variants, SVD temporal/source v2, nested tree router, physics ridge/HMM도 promotion 기준을 충족하지 못해 보류 또는 기각했다.

### 6.2 현재 최고 full-coverage OOF

대상: 18 physical identities, reference-valid 2,327 windows, identity-disjoint OOF.

| 지표 | 현재 leader | 내부 목표 | 결과 |
|---|---:|---:|---|
| Overall MAE | 1.291 bpm | ≤ 1.000 | 실패 |
| Identity-macro MAE | 1.220 bpm | ≤ 1.000 | 실패 |
| RMSE | 2.410 bpm | ≤ 1.800 | 실패 |
| 절대오차 ≤ 2 bpm | 80.79% | ≥ 90.0% | 실패 |
| 절대오차 > 5 bpm | 6.23% | ≤ 3.0% | 실패 |
| 25–35 bpm MAE | 4.216 bpm | ≤ 2.000 | 실패 |

- 통과: `0/6`
- identity-cluster bootstrap MAE 95% CI: `[0.981, 1.473]`
- greedy non-overlap: n=444, MAE 1.570, macro MAE 1.440, RMSE 2.860, >5 bpm 8.78%
- 결론: 연구 baseline으로 의미 있으나 상용 성능이 아님

### 6.3 Candidate oracle의 의미

- HCES discovery-validation oracle MAE: 약 `0.433–0.481 bpm`
- DHFER H0 unit candidate/anchor oracle MAE: `0.559 bpm`
- 과거 5-expert oracle: MAE `0.689 bpm`, macro `0.674 bpm`

Oracle은 target를 보고 가장 가까운 후보를 선택하므로 배포할 수 없다. 다만 좋은 후보가 자주 존재하며 **후보 생성보다 unseen identity에서의 selection/routing이 주 병목**이라는 진단을 제공한다.

## 7. Block 분할과 평가 계약

### 7.1 학습 단위와 block

아래는 corrected evaluation contract다. 현재 historical cache와 stage eligibility 구현이 전부 이 계약을 충족한다는 뜻은 아니다. 특히 현 코드는 `status != review` 조건으로 일부 `uncertain` stage를 metric-eligible로 둘 수 있어 P0에서 분리해야 한다.

```text
Physical identity
└─ session
   └─ 승인된 acquisition stage
      └─ core / transition / uncertain
         └─ 32초 window, 4초 stride
```

- 같은 사람의 반복 sessions: 항상 같은 outer fold
- session boundary: temporal SNN state reset
- future windows: 추론에서 사용 금지
- transition/uncertain: corrected stage metric에서 제외 또는 별도 보고
- reference-invalid window: temporal state 업데이트 가능, supervised weight 0
- stage label: evaluation/QC용, inference feature 금지

### 7.2 고정 physical-identity 6-fold

| Fold | identities | valid windows |
|---:|---|---:|
| 0 | KDM, KTW, SJE | 494 |
| 1 | GEC, LJH, MDO | 408 |
| 2 | CHW, HDH, JKH | 348 |
| 3 | KJH, PSJ, RJS | 403 |
| 4 | KDH, LHS, YSE | 438 |
| 5 | CMS, LDW, PJS | 236 |

각 run에서 outer fold `o`는 test, `(o+1)%6`은 validation, 나머지 4 folds는 train이다. proposer도 inner-OOF로 별도 학습하고 scaler·router·threshold·ensemble weight·checkpoint는 outer-test target를 보지 않는다.

### 7.3 최종 평가 묶음

1. **Primary**: 2,327 reference-valid window 전체 coverage
2. **Identity generalization**: overall과 identity-macro를 함께 보고
3. **Overlap sensitivity**: greedy non-overlap + `window_number mod 8`의 8 temporal phases
4. **Uncertainty**: physical-identity cluster bootstrap CI
5. **RR strata**: 특히 25–35 bpm tail
6. **Stage strata**: reliable core만 주 평가, transition/uncertain 별도
7. **Radar robustness**: `123, 12, 13, 23, 1, 2, 3` 일곱 masks
8. **Calibration/selective risk**: coverage를 고정해 보고, subset 결과를 full coverage 대신 사용 금지
9. **Streaming**: chronological replay, batch parity, state reset, finite output, fallback
10. **System**: CPU/CUDA latency, peak memory, spike rate, fault injection, provenance hash

세 fixed seeds `20260828`, `20260829`, `20260830`을 각각 평가한다. seed 평균으로 실패를 감추거나 fold/seed별로 다른 구조·release mode를 선택하지 않는다.

### 7.4 기존 radar-mask robustness

| Available radar | MAE | Macro MAE | RMSE |
|---|---:|---:|---:|
| 1+2+3 | 1.291 | 1.220 | 2.410 |
| 1+2 | 1.486 | 1.417 | 2.755 |
| 1+3 | 1.445 | 1.374 | 2.723 |
| 2+3 | 1.422 | 1.338 | 2.709 |
| 1 only | 1.636 | 1.575 | 2.895 |
| 2 only | 1.554 | 1.478 | 2.882 |
| 3 only | 1.586 | 1.532 | 2.995 |

현재 raw data에는 세 radar가 모두 있는 usable sessions가 기준이므로 이 결과는 자연 결측률이 아니라 structural mask stress test다.

## 8. 실행 무결성·provenance 진행상황

구현된 항목:

- source/config/contract/checkpoint/data SHA-256 binding
- outer row를 파일에서 제거한 sealed split packs
- train-only scaler와 structural mask exact-zero 검사
- target-free NPZ schema, `allow_pickle=False`
- immutable source snapshot, create-once authorization flow
- bubblewrap no-network sandbox와 denied-path canary
- GPU admission lock, append-only usage/execution ledgers
- SIGKILL recovery와 atomic result publication

보존된 실제 실패:

- `ROOTBIND1`: sandbox와 GPU admission 뒤 pretrain context 전달 누락으로 fail-closed
- 실패 GPU usage 약 2.847초와 terminal evidence 보존
- `CONTEXT1`: source 보강과 tests는 존재하지만 아래 세 artifact는 아직 없음

```text
IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTEXT1.json
V3R1_SOURCE_SNAPSHOT_V8R4A_CONTEXT1.json
PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json
```

이 세 파일을 손으로 만들거나 guard를 우회해 production training을 시작하면 안 된다. independent audit와 capable host의 real-bubblewrap 확인 뒤 validator/runtime 발급 순서를 따라야 한다.

## 9. Test 상태

백업 시점의 두 번 연속 full-suite 결과:

| 항목 | 값 |
|---|---:|
| Collected | 694 |
| Passed | 688 |
| Skipped | 4 |
| Failed | 2 |

실패 tests:

- `test_causal_prefix_is_invariant_to_future_samples`, max absolute diff `3.576e-7`
- `test_chunk_round_padding_forward_loss_state_and_gradient_equivalence`

두 test는 함께 isolated rerun에서 2/2 통과했다. 현재 판정은 clean pass가 아니라 order/thread/floating-point-state sensitivity 미해결이다. skip 4개는 originating sandbox의 namespace 정책으로 막힌 real-bubblewrap tests이며 capable Linux host에서 재실행해야 한다.

## 10. 현재 핵심 차단점

### P0 — 새 metric 생성 전 해결

1. measured timestamp 기반 실제 causal resampling 구현
2. nominal-time fallback을 strict mode에서 fail-closed
3. sync rejected/ambiguous mapping의 manual approval 가능 범위 제한
4. 모든 29 usable sessions의 sync·protocol review와 signed receipt
5. `uncertain` stage와 metric eligibility 분리
6. strict scientific eligibility에서 sync/protocol/range-layout 조건 분리
7. acquisition pipeline hash에 parser·timing source 전체 포함
8. subset run의 `complete=true`와 full-cohort completion 구분
9. reconstruct → strict build → train end-to-end test 추가
10. corrected strict full cache와 baseline OOF 재생성

### P1 — 모델 병목 해결

1. evidence와 radar/ratio/branch coordinate를 pooling 전에 결합
2. expert expected error·`P(error>2)`·`P(error>5)` risk heads
3. soft expected-cost/tail-risk 학습, hard decision은 inference에서만 사용
4. direct anchor fallback과 no-estimate 경로
5. optimizer update 수 사전 고정, warmup→router→joint training
6. identity/RR-band group DRO와 25–35 bpm tail 강화
7. coordinate-swap sensitivity, candidate permutation equivariance, masked invariance unit tests

### P2 — 실행 closure와 최종 검증

1. 두 flaky numeric tests 원인 규명
2. 최신 source real-bubblewrap 4 tests
3. CONTEXT1 receipt → source snapshot → pretrain authorization 발급
4. V8R4 efficiency gate
5. discovery `2 outer folds × 3 seeds × 3 variants = 18 units`
6. 공통 구조·release mode 하나를 잠근 뒤 `6 folds × 3 seeds = 18 units` promotion OOF
7. 7 masks·non-overlap·stage·streaming·latency·fault campaign

### P3 — 상용 주장에 필요한 외부 검증

1. 기존 18명과 겹치지 않는 prospective calibration cohort
2. calibration과도 독립인 prospective confirmation cohort
3. common hardware trigger 또는 측정 가능한 clock sync
4. 독립 reference/capnography와 reference failure annotation
5. target device 성능·전력·thermal·memory 측정
6. packet loss, saturation, gain drift, displacement, clock skew, desync, stuck radar fault tests
7. shadow/canary, rollback, monitoring, 운영 threshold 검증

## 11. 권장 실행 순서

두 트랙을 섞지 않고 각각 닫는다.

```text
[데이터 신뢰성 트랙]
P0 acquisition 계약 수정
→ 29-session strict reconstruction·manual review
→ corrected strict full cache
→ 기존 leader baseline 재평가

[동결된 V8R4 과학 트랙]
numeric flaky test 원인 규명 + real-bwrap
→ CONTEXT1 발급 closure
→ efficiency
→ 18-unit discovery
→ common 선택 lock
→ 18-unit full OOF promotion

[차기 모델 트랙]
V8R4 failure taxonomy 동결
→ CCHG-SNN axis encoder
→ differentiable risk router
→ fixed-seed discovery·full OOF

[제품 검증 트랙]
prospective calibration
→ prospective confirmation
→ target-device fault/shadow/canary/rollback
→ commercial claim decision
```

새 acquisition correction으로 dataset semantics가 바뀌면 기존 locked evidence를 덮어쓰지 않는다. 새 versioned cache·contract·output root를 만들고, 기존 결과는 historical retrospective evidence로 보존한다.

## 12. 복원·백업 상태

### 12.1 완전 복원 세트

복원 세트는 source/config/tests/docs, selected checkpoints와 OOF evidence, strict HCES preprocessing stacks, complete V8R4 state, 두 raw ZIP을 보존한다.

- binary payload 합계: 3,860,608,510 bytes, 약 3.60 GiB
- Drive transport: 38개 sub-100 MiB multipart
- 제한된 restore folder: `https://drive.google.com/drive/folders/150Ztq0JoXdZkKpEHujzOuAfWDUr79Pey`
- 제외: `.venv`, extracted raw duplicate, 약 15 GB deterministic cache, 중복 intermediate tensors
- 권장 여유 공간: source restore 8 GB, raw+cache 30 GB, retraining 45 GB

### 12.2 경량 검토 snapshot

초기 요청 folder에는 source/docs와 aggregate evidence의 경량 snapshot도 업로드됐다. 이는 raw data, checkpoints, per-window arrays, large sealed packs를 제외하므로 단독 full retraining backup은 아니다.

Private physiological data가 포함된 raw archive와 restore folder는 계속 제한 공유로 유지해야 한다.

## 13. 완료 판정

### 이미 완료된 핵심 산출물

- deterministic raw parser와 30-session audit
- 29 usable sessions·18 physical identities 확정
- 기존 canonical RF/SVD cache와 reference QC
- identity-disjoint 6-fold와 nested proposer 체계
- Structured TriRadarRRSNN leader full OOF
- HCES v2 3-seed locked full OOF와 failure decomposition
- DHFER v3r1 source·trainer·sealed execution infrastructure
- acquisition reconstruction code, 3-session smoke, 30-session diagnostic
- provenance, failure ledger, restore bundle, agent restoration guide

### 아직 완료되지 않은 핵심 산출물

- 승인된 29-session measured-time synchronization
- 신뢰 가능한 전체 실험 stage/block manifest
- corrected acquisition-aware strict full cache와 baseline/full OOF
- 깨끗한 694-test closure와 최신 real-bubblewrap 검증
- V8R4 CONTEXT1 authorization trio
- V8R4 efficiency·18-unit discovery·18-unit promotion
- CCHG-SNN 구현·측정
- 세 fixed seeds의 상용 gate 통과
- 독립 prospective·target-device 상용 검증

## 14. 최종 판단

현재 프로젝트는 parser, causal preprocessing, identity-disjoint evaluation, structured SNN, harmonic candidate graph, spiking temporal router, target firewall, provenance와 복원 체계를 갖춘 **성숙한 retrospective 연구 시스템**이다. 하지만 정확도 gate는 0/6이고 최신 acquisition-aware 전체 데이터도 과학 승인 전이다.

따라서 현재 수치는 상용 성능이 아니다. 다음 성능 향상은 모델 크기 확대보다 먼저 데이터 시간축·block 계약을 닫고, 그 위에서 좌표 보존 encoder와 미분 가능한 risk router를 평가해야 한다. 내부 fixed-seed gate를 모두 통과하더라도 독립 prospective cohort와 target-device 검증 전에는 상용·의료 성능을 주장하지 않는다.

## 15. 주요 근거와 시작점

- 작업 계약: [`AGENTS.md`](../AGENTS.md)
- 복원 안내: [`RESTORE_GUIDE.md`](../RESTORE_GUIDE.md)
- 상세 기술 보고서: [`SNN_PROJECT_TECHNICAL_STATUS_REPORT_2026-08-30.md`](SNN_PROJECT_TECHNICAL_STATUS_REPORT_2026-08-30.md)
- 간결 기술 로드맵: [`SNN_PROJECT_COMPACT_TECH_STACK_2026-08-30.md`](SNN_PROJECT_COMPACT_TECH_STACK_2026-08-30.md)
- 프로젝트 README: [`README.md`](../README.md)
- 기존 결과 보고: [`REPORT.md`](../REPORT.md)
- 상용 목표: [`COMMERCIAL_SNN_GOAL_V2.md`](COMMERCIAL_SNN_GOAL_V2.md)
- 연속 실행 계획: [`COMMERCIAL_SNN_CONTINUOUS_EXECUTION_PLAN_V4.md`](COMMERCIAL_SNN_CONTINUOUS_EXECUTION_PLAN_V4.md)
- 최신 acquisition smoke: [`reconstruction_v1_smoke2/manifest.json`](acquisition/reconstruction_v1_smoke2/manifest.json)
- 이전 full acquisition diagnostic: [`reconstruction_v1_sync_diagnostic/manifest.json`](acquisition/reconstruction_v1_sync_diagnostic/manifest.json)
- backup-time test 증거: [`backup_validation_2026-08-30.json`](../restore/backup_validation_2026-08-30.json)
- restore index: [`SnnProject_RESTORE_INDEX_2026-08-30.md`](../restore/SnnProject_RESTORE_INDEX_2026-08-30.md)
- Google Drive snapshot manifest: [`GOOGLE_DRIVE_UPLOAD_MANIFEST_2026-08-30.md`](GOOGLE_DRIVE_UPLOAD_MANIFEST_2026-08-30.md)
- parser·dataset: [`src/snn_rr/data.py`](../src/snn_rr/data.py)
- timing·sync: [`radar_timing.py`](../src/snn_rr/radar_timing.py), [`synchronization.py`](../src/snn_rr/synchronization.py)
- protocol·range: [`acquisition_protocol.py`](../src/snn_rr/acquisition_protocol.py), [`range_tracking.py`](../src/snn_rr/range_tracking.py)
- preprocessing: [`preprocess.py`](../src/snn_rr/preprocess.py)
- HCES: [`harmonic_set_models.py`](../src/snn_rr/harmonic_set_models.py)
- DHFER: [`harmonic_factor_router_v3.py`](../src/snn_rr/harmonic_factor_router_v3.py)
