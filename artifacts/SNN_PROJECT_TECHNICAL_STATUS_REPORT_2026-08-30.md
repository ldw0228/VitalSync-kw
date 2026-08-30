# 3-Radar 호흡수 SNN 프로젝트 기술 현황·방법론·향후 실행 보고서

- 기준 시각: 2026-08-30, Asia/Seoul
- 프로젝트: 3대 XeThru UWB radar 기반 32초 direct respiratory-rate 추정
- 현재 상태: `ACTIVE_RESEARCH_CYCLE`
- 현재 과학 계열: HCES v2 → DHFER-SNN v3r1/V8R4 → 차기 V8R5 설계 준비
- 주장 경계: retrospective engineering evidence이며 상용·의료 성능 확정이 아님

이 문서는 지금까지 구현·학습·평가한 내용을 하나의 기술 기준서로 합친다. 이미 측정된 결과, oracle 진단, 실행 인프라 교정, 아직 구현하지 않은 차기 구조를 서로 구분한다. 같은 18명 코호트를 반복 관찰해 얻은 모든 적응적 결과는 `retrospective_adaptive`로 취급한다.

## 1. 한 문장 결론

현재 파이프라인은 raw byte 감사부터 물리적 사람 단위 nested OOF, 후보 bank, SNN router, target-free sealed inference, 7개 radar mask, calibration·streaming·latency·provenance까지 상당 부분 완성됐지만, 고정 seed별 상용 정확도 기준은 아직 통과하지 못했다. 후보 bank의 validation oracle MAE가 약 0.43–0.56 bpm으로 충분히 낮은 반면 실제 router는 약 1.3–1.8 bpm에 머물러, 현재의 지배적 과학 병목은 후보 부족보다 좌표 정보가 소실되는 encoder와 hard-routing risk 학습의 불일치다.

## 2. 목표와 완료 판정

내부 engineering success는 세 고정 seed `20260828`, `20260829`, `20260830`이 각각 아래 여섯 gate를 모두 통과할 때만 인정한다. 평균 seed나 최고 seed로 실패 seed를 숨기지 않는다.

| 지표 | 고정 목표 |
|---|---:|
| Overall MAE | ≤ 1.000 bpm |
| Identity-macro MAE | ≤ 1.000 bpm |
| RMSE | ≤ 1.800 bpm |
| 절대오차 ≤ 2 bpm | ≥ 90.0% |
| 절대오차 > 5 bpm | ≤ 3.0% |
| Reference 25–35 bpm MAE | ≤ 2.000 bpm |

정확도 외에도 다음이 필요하다.

- 2,327개 reference-valid row 전체 coverage
- 7개 non-empty radar mask 전체 평가
- 겹치지 않는 32초 window와 8개 고정 temporal phase 평가
- held-identity calibration과 selective-risk 평가
- offline batch와 chronological streaming parity
- CPU/CUDA latency, peak memory, parameter 수, spike rate, finite-output 검사
- source/config/checkpoint/data/provenance hash의 exact closure

내부 gate를 통과하더라도 별도 prospective cohort가 없으면 최종 상태는 `INTERNAL_ENGINEERING_PASS / COMMERCIAL_CONFIRMATION_BLOCKED`다. 실제 `COMMERCIAL_RELEASE_READY`에는 독립 identity, 독립 reference, target device, fault injection, shadow/canary, rollback 검증이 추가로 필요하다.

## 3. 지금까지의 발전 경로

### 3.1 12-step structured TriRadarRRSNN

초기 주력 구조는 radar별 range–frequency map을 shared 2-D encoder로 처리하고, range attention과 radar reliability로 세 view를 결합한 뒤 PLIF/LIF residual Conv1D frequency backbone을 12 simulation step 전개하는 hybrid SNN이었다.

```text
3 radar × [73 frequency × 182 range/branch]
        │ shared spatial CNN
        │ range attention + radar reliability
        ▼
topology-preserving spectrum fusion
        ▼
12-step PLIF/LIF frequency backbone
        ├─ 6–45 bpm posterior
        ├─ expected RR
        ├─ uncertainty
        ├─ quality proxy
        └─ spike statistics
```

학습에는 ANN teacher distillation, identity-balanced sampling, coupled radar dropout, strictly causal history를 사용했다. structured auxiliary component와 exact auxiliary-alignment component를 outer-validation identity에서만 잠근 convex ensemble로 결합했다.

이 계열의 최고 full OOF 결과는 18명, 2,327개 row에서 MAE 1.291, identity-macro MAE 1.220, RMSE 2.410, ±2 bpm 80.79%, >5 bpm 6.23%, 25–35 bpm MAE 4.216이었다. 기존 flat/default SNN MAE 1.407보다 개선됐지만 목표는 0/6 통과였다.

### 3.2 HCES v2

고호흡수 harmonic alias를 직접 다루기 위해 회귀 하나 대신 최대 12개 RR 후보를 graph node로 취급하는 Harmonic Candidate-Set Episode SNN을 만들었다. 후보 bank가 좋은 RR을 포함하는지와 router가 그것을 선택하는지를 분리할 수 있게 된 것이 핵심 변화다.

### 3.3 DHFER-SNN v3r1

HCES v2의 locked 결과에서 좋은 후보는 존재하지만 routing regret이 큰 것으로 확인되어, 방향성 harmonic graph와 ×1/×2/×3/×4 factor router, anchor/candidate hard expert를 갖는 Directed Harmonic Factor-Expert SNN을 구현했다.

### 3.4 V8R4/V8R4A 실행 경계

과학 구조와 별도로 outer row를 파일 수준에서 제거한 sealed pack, bubblewrap target sandbox, GPU admission/usage ledger, SIGKILL 복구, immutable source snapshot을 구축했다. 이는 정확도를 올리는 모델 변경이 아니라 outer-test 누수, stale output 재사용, 실패 GPU charge 은폐, 부분 실행을 과학 결과로 오인하는 문제를 막기 위한 실행 계층이다.

## 4. 데이터의 실체

### 4.1 규모

| 항목 | 값 |
|---|---:|
| 원본 session 폴더 | 30 |
| 3-radar + BIOPAC usable session | 29 |
| 물리적 identity | 18 |
| 선택된 3-view radar frame record | 4,715,613 |
| 동기화 radar 시간 | 약 10.916 h |
| paired BIOPAC 시간 | 약 11.001 h |
| 생성된 32초 후보 window | 9,576 |
| reference-valid window | 2,327, 24.30% |

29 session을 29명으로 취급하지 않는다. KTW, LHS, LDW, KDM, MDO, HDH, JKH, SJE, KDH, LJH, PSJ 등은 여러 session에 반복 등장한다. `S17_RJS` 폴더 suffix는 실제로 PJS로 교정했다. split 단위는 항상 session이 아니라 물리적 사람이다.

프로토콜 분포는 Dodge 10 sessions/851 valid, Strike 10/830, Kick 9/646이다. Reference-valid RR은 10–20 bpm에 집중돼 있다.

| RR band | valid rows |
|---|---:|
| 6–10 | 202 |
| 10–15 | 852 |
| 15–20 | 742 |
| 20–25 | 361 |
| 25–35 | 164 |
| 35–45 | 6 |

25–35 bpm은 valid row의 약 7.05%뿐이고 greedy non-overlap 기준으로는 42개다. 따라서 high-RR tail은 모델 구조 문제와 데이터 support 부족이 동시에 존재한다.

### 4.2 원본 예외와 처리

- `S24_KHJ`: 세 radar가 모두 비어 제외했다.
- `S01_CMS`: 긴 3-chunk recording과 짧은 retry가 함께 있어, 세 radar 공통 duration이 가장 긴 logical session을 선택했다.
- `S07_KDM`: metadata timestamp counter reset을 단조 시간축으로 복구했다.
- `S22_KJH`: radar 2의 비정상 sample 한 개를 미래를 보지 않는 past-only 방식으로 복원했다.
- Paired RSP sample 약 5.22%가 rail에 닿았고, `S30_SJE` 약 43%, `S06_LDW` 약 16%로 특히 심했다.
- 세 radar 시작 시각 spread는 최대 16 ms, frame count spread는 최대 1 frame이었다.
- radar와 BIOPAC은 absolute epoch overlap으로 정렬했지만 공통 hardware trigger가 없어 sub-frame residual 오차는 배제할 수 없다.

## 5. 전처리와 denoising

### 5.1 Raw radar 검사

XeThru record에 대해 record-size remainder, zero header, 182-bin count, frame sequence gap, NaN/Inf, amplitude, metadata frame count와 timestamp를 검사한다. 여러 `.dat` chunk는 하나의 read-only continuous recording처럼 읽는다.

### 5.2 Past-only 이상치 복원

`|x| > 0.1`인 sample은 같은 range bin의 직전 최대 4 frame 중 finite·정상 sample의 median으로 대체한다. 과거 정상값이 없으면 0을 쓴다. 다음 frame은 보지 않는다. 전체 canonical cache에서 실제 교체는 한 sample이었다.

### 5.3 Causal downsample과 window

- radar nominal 40 Hz → non-overlap 4-frame block mean → 10 Hz
- causal aggregation latency 75 ms
- 320 samples = 32초 window
- stride 40 samples = 4초
- 인접 window overlap 28초, 87.5%
- 세 radar 공통 길이와 BIOPAC 완전 overlap 안에서만 window 생성
- 짧은 tail을 뒤로 이동시켜 label timestamp와 radar timestamp를 바꾸지 않음

### 5.4 Radar spectral denoising

각 32초 radar window에 다음을 적용한다.

1. 값 `[-0.05,+0.05]` clip
2. range-bin별 mean 제거
3. range-bin별 선형 trend 제거
4. Hann taper
5. 2,048-point real FFT
6. 0.08–0.80 Hz respiration band 선택
7. frequency 방향 `[0.25,0.50,0.25]` smoothing
8. FFT power 이후 인접 range bin pooling
9. range별 median spectral power를 local noise floor로 사용
10. `log1p(power/noise)`와 temporal-variance activity weight 적용
11. 최종 `[0,1]` 범위로 scaling

Radar signal에 zero-phase IIR을 거는 방식은 아니다. causal window 안의 detrend, Hann FFT, band selection, light smoothing, robust gain normalization이 denoising 역할을 한다.

### 5.5 Raw/phase branch

182 값을 두 branch로 표현한다.

- raw-power: 원래 182 bin을 power 이후 2개씩 pooling해 91 range bins
- candidate I/Q-phase: 앞 91을 real, 뒤 91을 imaginary로 해석하는 가설적 branch

I/Q 해석은 장비 명세로 확인되지 않았다. 이전 structured SNN은 두 branch를 사용했지만 현재 harmonic v3 cache는 보수적으로 raw-power만 사용하고 phase branch는 exact zero와 false mask로 처리한다.

### 5.6 Canonical RF cache

- `maps.npy`: `[N,3,73,182]`, float16
- `aux.npy`: `[N,1205]`, float32
- frequency grid: 0.085449–0.788574 Hz, 73 bins
- aux: radar별 q90/q98 spectrum, scalar diagnostics, fused median/max spectrum, peak spread, pairwise correlation

### 5.7 Randomized-SVD source separation

Gross motion이 호흡 component를 덮는 문제를 줄이기 위해 label-free SVD cache도 만들었다.

- representation: raw, standardized raw, temporal velocity, standardized velocity, range difference, standardized range difference, 가설적 split amplitude/phase 계열
- current harmonic forward에는 검증된 앞 6개 view만 사용
- radar/window/view별 randomized SVD 12 components, `n_iter=2`
- component RMS normalization
- Hann + 4,096 FFT
- 0.08–0.80 Hz band
- reliability: singular energy, band fraction, concentration, entropy를 결합

SVD feature extractor는 BIOPAC/reference를 읽지 않는다.

## 6. BIOPAC reference와 quality control

BIOPAC RSP는 production input이 아니라 offline target 생성에만 사용한다. 전체 RSP에 4차 Butterworth 0.10–0.75 Hz zero-phase `sosfiltfilt`를 적용한다. 미래 sample을 사용하는 zero-phase filter지만 label 작성 경로에만 있으므로 inference leakage는 아니다.

각 window에서 세 추정기를 계산한다.

- Hann FFT sub-bin quadratic peak
- peak-to-peak inter-breath interval median
- Hilbert analytic phase slope

세 값의 median을 RR target으로 사용하고, 아래 조건을 모두 만족할 때만 reference-valid로 인정한다.

- 6–45 bpm 내부
- 최소 3 cycles
- raw rail clip fraction ≤ 2%
- plateau fraction < 25%
- spectral concentration ≥ 0.42
- autocorrelation periodicity ≥ 0.28
- interval CV ≤ 0.27
- estimator disagreement ≤ 2.5 bpm
- Hilbert phase residual ≤ 1.25 rad
- 앞뒤 2초 guard clip fraction ≤ 2%

`reference_sigma_bpm`은 독립적으로 calibration된 표준편차가 아니라 disagreement와 quality를 조합한 0.35–2.0 bpm heuristic이다. 역사적 `radar_observable` column도 독립 quality label이 아니라 `reference_valid && |classical-reference|≤2`인 target-dependent proxy이므로 inference feature로 금지한다.

## 7. Harmonic candidate bank

후보 범위는 6–45 bpm, 최대 12개다. 우선순위는 다음과 같다.

1. frozen proposer expected RR
2. frozen proposer MAP RR
3. posterior NMS direct modes 최대 5개, suppression 1.25 bpm
4. classical RR ×1, ×2, ×3, ×4
5. radar 1/2/3 peak

0.5 bpm 안의 후순위 후보는 첫 anchor BPM을 움직이지 않고 source bit와 confidence만 합친다. 마지막에는 BPM stable sort를 한다. 전체 9,576 rows에서 valid candidate node는 81,382개, row당 평균 8.499개다.

각 candidate `c`에 대해 다음 harmonic 위치를 label-free spectrum에서 조회한다.

```text
c × {1/4, 1/3, 1/2, 1, 2, 3, 4}
```

Native frequency grid의 인접 두 bin을 triangular interpolation한다. 범위 밖이면 edge clamp하지 않고 값 0, mask false를 반환한다.

## 8. 571차원 candidate node feature

| 영역 | 차원 | 구조 |
|---|---:|---|
| Core | 46 | candidate BPM/confidence/source, proposer descriptors, expected/MAP distance, gaps 등 |
| RF | 378 | 3 radar × 7 ratio × 2 branch × 9 statistics |
| SVD | 147 | 3 radar × 7 ratio × 7 statistics |
| 합계 | 571 | fixed ordered schema |

Core에는 source one-hot/confidence, direct-mode rank와 probability, posterior local mass, entropy, proposer std/quality/alias/spike, expected/MAP 거리, radar reliability, candidate gap이 들어간다.

RF 통계는 power mean/max/entropy/peak concentration, top-1/2 value와 원 range index, cross-radar consensus다. SVD 통계는 reliability-weighted mean/max, component entropy, peak distance, reliability mean/max다.

Structural availability는 feature 값이 0인가로 추정하지 않고 다음으로 계산한다.

```text
candidate valid
AND radar available
AND candidate×ratio in 6–45 bpm
AND frozen branch available
```

Unavailable cell은 scaler 전후 exact `+0.0`이어야 한다.

## 9. Split, nested proposer, leakage 방지

### 9.1 고정 6-fold

| Fold | identities | valid rows |
|---:|---|---:|
| 0 | KDM, KTW, SJE | 494 |
| 1 | GEC, LJH, MDO | 408 |
| 2 | CHW, HDH, JKH | 348 |
| 3 | KJH, PSJ, RJS | 403 |
| 4 | KDH, LHS, YSE | 438 |
| 5 | CMS, LDW, PJS | 236 |

Outer fold `o`가 test, `(o+1)%6`이 validation, 나머지 4 folds가 train이다. Fit/validation/test identity 교집합은 hard error다.

### 9.2 완전 nested proposer

Router 앞단 proposer도 prediction identity를 학습에 보면 누수다. Outer fold마다 non-test training folds의 inner-OOF proposer 네 개와 outer-validation proposer 하나를 별도로 만든다. Discovery에서는 outer-test proposer와 outer-test prediction pack을 만들지 않는다.

Proposer train identity, proposer validation identity, proposer prediction identity, outer validation identity, outer test identity를 역할별로 분리한다. 모든 checkpoint, split manifest, cache row semantics와 hash를 검증한다.

### 9.3 Outer-row-free sealed pack

V8R4 pack은 outer row를 mask만 하는 것이 아니라 파일에서 물리적으로 제거한다.

- outer 3 pack: 8,241 rows, 15 identities, valid 1,924
- outer 4 pack: 7,918 rows, 15 identities, valid 1,889

Trainer는 pack hash·byte size·exact schema를 확인하고 outer row가 하나라도 있으면 reference mmap을 열기 전에 중단한다.

### 9.4 Scaler와 sampling

- outer-train available cell만으로 float64 mean/std fit
- validation/test에는 frozen transform
- unavailable cell은 transform 후 exact zero
- identity별 총 supervised mass를 같게 하는 weight
- session chronology를 보존한 identity-balanced lane sampling
- invalid-reference window는 state를 업데이트하지만 supervised weight 0
- session 첫 2 windows는 warmup loss 제외
- 25–35 bpm valid row는 총 3배 loss weight

## 10. SNN neuron과 시간 개념

공통 cell은 LIF, PLIF, ALIF를 지원한다.

- 초기 membrane decay `β=0.92`
- PLIF/ALIF decay는 sigmoid parameterization으로 학습
- threshold는 softplus 양수 parameter, 초기 1
- ALIF adaptation decay 0.97, strength 0.4
- surrogate gradient: fast sigmoid, slope 25
- detached reset 사용

```text
u' = βu + I
s  = surrogate_step(u' - threshold)
u' = u' - stop_gradient(s)·threshold
```

두 시간축을 구분해야 한다.

1. 실제 causal time: 4초 간격 32초 window sequence
2. 한 window 내부의 SNN simulation step: DHFER는 8회

Session 경계에서 PLIF/ALIF state를 reset하고 TBPTT chunk 사이에는 state를 detach한다. 미래 window는 입력하지 않는다.

## 11. HCES v2 구조와 결과

### 11.1 구조

- candidate feature + normalized RR + radar coverage를 hidden 64로 projection
- near/×2/×3/×4의 대칭 harmonic graph
- residual PLIF graph block 2개
- 4-head masked candidate attention + masked mean pool
- causal PLIF→ALIF episode encoder
- candidate selection logit, ±0.75 bpm residual, Gaussian scale
- factor head, quality head
- i3에서는 proposer anchor/std/availability를 별도 expert로 사용
- parameter 수: hidden64 약 195,603, hidden96 약 410,131

HCES YAML의 `simulation_steps:8`과 달리 HCES forward는 DHFER처럼 각 window를 8회 반복하지 않는다. HCES의 실제 transition은 graph block과 temporal layer별 한 번이다.

### 11.2 Locked full OOF

| seed | MAE | Macro MAE | RMSE | ≤2 bpm | >5 bpm | 25–35 MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 20260828 | 1.2963 | 1.2632 | 2.4274 | 80.36% | 7.22% | 3.5670 |
| 20260829 | 1.3722 | 1.3292 | 2.6130 | 80.10% | 7.43% | 4.0731 |
| 20260830 | 1.3772 | 1.3573 | 2.5930 | 79.42% | 6.83% | 3.6955 |

세 seed 모두 목표를 실패했다. 안전한 sparse correction policy가 없어서 locked final은 fallback과 같았다.

### 11.3 Candidate oracle

Outer 3/4 discovery validation 674 rows에서 target를 이용해 가장 가까운 후보를 고른 진단용 oracle은 다음과 같았다.

| seed | Oracle MAE | ≤1 bpm | high-RR oracle MAE |
|---:|---:|---:|---:|
| 20260828 | 0.4808 | 90.50% | 1.2087 |
| 20260829 | 0.4327 | 92.43% | 0.9729 |
| 20260830 | 0.4653 | 91.99% | 1.2052 |

Oracle은 deployable 성능이 아니다. 다만 후보 bank에 목표 수준의 정보가 상당 부분 들어 있고 실제 선택기가 이를 실현하지 못한다는 강한 진단이다.

## 12. DHFER-SNN v3r1 구조

### 12.1 Structured encoder

- core MLP: `46→32→32`
- shared RF cell MLP: `9→16→16`
- shared SVD cell MLP: `7→16→16`
- radar/ratio/branch embeddings
- mask-aware RF mean과 SVD mean
- `core32 + RF16 + SVD16 → 64`

### 12.2 Directed graph

관계는 7개다.

1. near
2. receiver = 2×sender
3. sender = 2×receiver
4. receiver = 3×sender
5. sender = 3×receiver
6. receiver = 4×sender
7. sender = 4×receiver

Near tolerance 0.5 bpm, ratio tolerance 0.75 bpm이며 self-edge는 없고 residual path가 self 정보를 보존한다. 관계별 64→64 projection 후 self+7 messages를 512→64로 줄인다. Graph block은 2개이고 각 block의 PLIF는 8 internal step을 반복한다.

### 12.3 Pooling과 temporal factor router

- candidate attention 4 heads + mean pool
- radar availability projection
- anchor RR/std/available + classical RR/available의 5-value context
- causal PLIF→ALIF temporal router
- 각 real window마다 두 layer를 8 internal step 반복
- explicit state와 session reset

### 12.4 Expert decoder

각 candidate는 logit, ±0.75 bpm residual, 0.25–12 bpm scale을 가진다. Anchor는 logit, ±12 bpm residual, scale을 가진다.

Factor center는 다음과 같다.

```text
classical RR × {1,2,3,4}
```

Candidate-factor affinity는 다음이다.

```text
A[k,f] = exp(-|candidate[k] - classical·factor[f]| / 0.75)
```

Factor router가 활성화되면 predicted factor distribution으로 candidate logit에 최대 +2 boost를 준다. 최종 expert는 corrected anchor 1개와 K개 candidate이며 hard argmax로 하나만 선택한다. Continuous midpoint/average는 금지한다. 아무 expert도 없으면 classical RR fallback을 사용한다.

Safe initialization은 candidate logit/residual 0, anchor logit bias +4, anchor residual 0이다. 세 variant 모두 실제 parameter 수는 203,669개다.

### 12.5 H0/H1/H2

| Variant | 내용 |
|---|---|
| H0 | structured encoder + directed graph + hard expert; factor routing/loss off |
| H1 | H0 + factor router + class-balanced focal factor loss |
| H2 | H1 + wrong-harmonic margin + factor/candidate JS consistency |

세 release mode도 고정했다.

- `raw_anchor`
- `hard_source_argmax`
- `fixed_confidence_switch`: selected probability ≥0.8일 때 hard source, 아니면 raw anchor

Fold/seed별로 다른 variant나 release mode를 고를 수 없고 discovery 전체에서 공통 하나를 선택해야 한다.

## 13. DHFER loss와 최적화

| Loss | Weight | 역할 |
|---|---:|---|
| Listwise KL | 1.00 | 좋은 anchor/candidate에 soft responsibility 부여 |
| Gaussian mixture NLL | 0.25 | expert mixture likelihood |
| Candidate residual SmoothL1 | 0.30 | target이 ±0.75 안에 reachable한 candidate 보정 |
| Anchor residual SmoothL1 | 0.50 | proposer anchor correction |
| Anchor NLL | 0.15 | anchor scale 학습 |
| Factor focal | 0.35 | confident ×1/×2/×3/×4 label |
| Wrong-factor margin | 0.25 | H2에서 정답 logit margin 1 |
| Factor-candidate JS | 0.10 | H2에서 두 distribution 정합 |
| Quality BCE | 0.10 | hard-source error ≤2 proxy |
| Spike penalty | 0.005 | spike rate 0.01–0.20 밖 패널티 |
| Per-session worst-20% CVaR | 0.15 | tail error 강조 |

Listwise target는 대략 `q_e ∝ exp(-|μ_e-y|/0.5)`이다. Factor label은 classical×{1,2,3,4} 중 target과 가장 가까운 class이며 최소 오차가 2 bpm 이내일 때만 사용한다.

Optimizer는 AdamW, LR `3e-4`, weight decay `1e-4`, gradient clip 2, 최대 120 epochs, 최소 20, patience 18, AMP/deterministic 요청이다. Session 4개를 한 accumulation group으로 사용하고 TBPTT chunk는 32 windows다. AMP overflow 시 같은 group의 RNG/model/optimizer/scaler 상태를 되돌려 exact replay한다.

Checkpoint는 outer-validation에서 `(실패 gate 수, 최대 normalized violation, violation 합, macro MAE, MAE, epoch)`의 사전 고정 lexicographic key로 선택한다.

## 14. 실제 DHFER H0 결과

V8R3 H0, outer 3의 validation fold 4에서 438 valid/1,658 total rows를 평가했다.

| 지표 | Hard source | Raw anchor |
|---|---:|---:|
| MAE | 1.7688 | 2.0025 |
| Identity-macro MAE | 1.7396 | 1.9072 |
| RMSE | 3.3174 | 3.5834 |
| ≤2 bpm | 76.26% | 71.92% |
| >5 bpm | 10.05% | 12.56% |
| 25–35 bpm MAE | 4.7523 | 6.5085 |

Anchor+candidate oracle MAE는 0.5586, high-RR oracle은 1.5078이었다. Hard source는 anchor 201, candidate 237 rows를 선택했다.

Selected probability는 median 0.378, p90 0.459, p99 0.488, max 0.503이었다. 고정 threshold 0.8을 넘는 row가 0개였으므로 `fixed_confidence_switch`는 사실상 raw anchor와 같았다.

High-RR 55 rows의 hard-source bias는 약 −4.57 bpm이었다. 학습은 epoch 47에 끝났지만 epoch당 optimizer step이 약 5회라 총 update는 약 235회뿐이었다.

## 15. 현재 확인된 핵심 구조 결함

### 15.1 RF/SVD coordinate permutation invariance

현재 encoder는 각 cell에 대해 대략 다음을 계산한다.

```text
z_i = f(evidence_i) + radar_embedding_i + ratio_embedding_i + branch_embedding_i
pooled = mean_i(z_i)
```

고정 mask에서 두 좌표의 evidence를 바꾸면 `Σf(evidence)`와 `Σembedding`이 모두 같아진다. 즉 모델은 evidence vector의 multiset과 사용 좌표의 총합만 알고, 어느 radar/ratio에 어떤 evidence가 있었는지는 잃는다.

실제 동일 mask에서 `(radar1, ratio1/4)`와 `(radar3, ratio1)` RF cell을 교환했을 때 64차원 embedding의 max absolute 차이는 약 `3.58e-7`로 floating reduction noise 수준이었다.

Candidate 순서 permutation에 equivariance가 있는 것은 바람직하지만 radar/ratio evidence swap에 불변인 것은 유해하다. 현재 가장 명확한 구조적 병목이다.

### 15.2 Hard routing과 tail-risk gradient 불일치

Listwise KL과 mixture NLL은 router logit에 gradient를 준다. 그러나 hard argmax 이후 선택된 `source_rr`로 계산하는 CVaR은 선택된 expert 값에는 gradient를 주지만 argmax를 통과해 router choice에는 gradient를 주지 못한다. Quality BCE도 quality head를 학습할 뿐 expert 선택을 직접 미분 가능하게 만들지 않는다.

따라서 tail-risk loss를 넣었지만 “어떤 expert를 선택해야 tail이 줄어드는가”를 직접 최적화하지 못한다.

### 15.3 Confidence scale 부정합

고정 confidence 0.8은 실제 출력 범위 0.38–0.50과 맞지 않아 작동하지 않았다. Test를 본 뒤 threshold를 낮출 수는 없으므로 다음 세대에서는 held-identity calibration과 risk head를 구조적으로 포함해야 한다.

### 15.4 학습 update 부족과 group shift

현재 H0는 약 235 optimizer updates에 그쳤다. 또한 LHS/KDH 등 identity와 high-RR 구간에 오류가 집중돼 단순 row-average loss만으로는 worst-group가 개선되지 않는다.

## 16. Robustness, calibration, streaming, latency에서 한 일

### 16.1 7 radar masks

고정 조건은 `123,12,13,23,1,2,3`이다. 현재 canonical cache는 모두 3 radar가 존재하므로 이는 자연 결측이 아니라 ideal structural masking stress test다. HCES v2에서는 모든 seed·mask를 선택 없이 보고했고 모든 mask accuracy gate는 실패했다.

### 16.2 Non-overlap과 temporal phase

32초/4초 window overlap으로 row가 독립 표본이 아니므로 session별 greedy non-overlap과 `window_number mod 8`의 8개 fixed phase를 모두 평가했다. 이전 leader의 greedy subset은 n=444, MAE 1.570, macro 1.440이었다.

### 16.3 Calibration과 selective prediction

Normal diagnostic과 normalized conformal을 50/80/90/95% coverage에서 평가하고 selective coverage 50/80/90/100%를 고정했다. 일부 low-coverage subset의 MAE는 낮았지만 이는 full-coverage 상용 성능을 대체하지 않는다. 특히 기존 ensemble의 70% retention은 전체 후보 window와 교차하면 약 17%만 남는다.

### 16.4 Streaming

Session reset, whole-chunk/one-window parity, finite output, 7 masks, no-candidate fallback, corrupt-input fail-closed를 검사하는 streaming campaign을 만들었다. HCES v2 release readiness에서는 일부 unit이 spike telemetry/parity gate를 실패했다. 다음 세대에서는 full 18 units의 offline/streaming parity를 promotion 필수 gate로 유지한다.

### 16.5 Latency

이전 structured 12-step component의 raw-window end-to-end 측정은 다음과 같다.

| Component | CPU p50/p95 | RTX 4070 p50/p95 |
|---|---:|---:|
| Structured auxiliary | 49.67/51.67 ms | 29.09/30.93 ms |
| Structured exact | 49.89/63.09 ms | 28.43/29.97 ms |

두 component 합산은 CPU 약 100–115 ms, CUDA 약 58–61 ms의 보수적 상한이었다. 다만 이는 shared preprocessing을 중복 계산한 합이며 직접 ensemble latency가 아니다. 32초 context 대기, acquisition/network, cold disk는 포함하지 않았다.

## 17. 실행·provenance 계층

### 17.1 목적

이 계층의 목적은 보안 제품 개발이 아니라 과학 결과의 무결성이다.

- outer-test target를 학습 프로세스가 열지 못하게 함
- old context output을 새 성공으로 재사용하지 못하게 함
- 실패한 GPU charge를 삭제하지 못하게 함
- SIGKILL 뒤 partial state를 성공으로 오인하지 않게 함
- exact source/config/data bytes에서만 resume

### 17.2 구현한 요소

- source/config/contract/checkpoint/data manifest SHA-256 binding
- active implementation 35 files의 fixed allowlist
- immutable `0444` source snapshot과 create-once authorization
- target-free NPZ exact schema, `allow_pickle=False`
- bubblewrap: clear environment, no network, minimal Python/CUDA runtime
- GPU state parent directory read-only bind + admission/execution/usage 세 child만 read-write overlay
- mountinfo, FD, device, environment, denied canary 검사
- single GPU admission lock
- append-only usage and execution ledgers
- reservation/start/terminal exact closure
- atomic result publication과 directory fsync
- SIGKILL recovery tests

### 17.3 ROOTBIND1 실패와 CONTEXT1

ROOTBIND1 실제 실행은 sandbox와 parent-RO/3-child-RW topology를 통과하고 GPU admission 및 `torch.cuda.is_available()`까지 갔다. 그러나 benchmark worker가 target-scoped pretrain validation 결과를 trainer primitive에 전달하지 않아 trainer의 fail-closed guard가 다음 오류로 중단했다.

```text
RuntimeError: admitted pretrain validation requires independent phase/context
```

이때 cache/proposer/model/epoch/accuracy는 열리지 않았다. GPU usage 약 2.847초는 ledger에 남겼다. 이전 context를 재사용하지 않고 `authorization_generation=CONTEXT1`을 추가해 successor를 만들었다.

CONTEXT1에서는 다음을 구현했다.

- benchmark worker가 fixed phase/context/outer fold로 target-scoped pretrain을 먼저 검증
- validated object를 trainer primitive에 전달
- old ROOTBIND1 failure terminal을 정확히 한 번 infrastructure prefix로 소유
- live usage 77 records/execution 10 records의 exact postfailure prefix 강제
- 6개 superseded lifecycle/output roots를 모두 deny
- trio 문서에 top-level `authorization_generation=CONTEXT1`
- pretrain scope와 mount/canary schema의 exact validation
- dynamic path가 denied root를 가리키면 open 전 거부

### 17.4 현재 검증 상태

현재 13개 fixed test file에서 694 tests가 수집되며 690 passed, 4 skipped다. Skip 4개는 현재 sandbox에서 unprivileged namespace 제약 때문에 실행되지 않은 real bubblewrap tests다. ROOTBIND1 이전 build에서는 escalated real-bwrap 4/4가 통과했지만 최신 CONTEXT1 source에 대해서는 source freeze 후 다시 실행해야 한다.

현재 CONTEXT1 세 문서는 아직 발급하지 않았고 fresh lifecycle/output roots도 없다.

- `IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTEXT1.json`: absent
- `V3R1_SOURCE_SNAPSHOT_V8R4A_CONTEXT1.json`: absent
- `PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json`: absent
- `target_sealed_lifecycle_v8r4a_context1`: absent
- `efficiency_benchmark_v8r4a_context1`: absent

독립 재감사 중 발견된 dynamic phase path와 non-bind mount exact-cover 문제에 대한 코드 보강은 현재 source에 들어가 있고 fixed tests는 통과한다. 그러나 usage-limit로 독립 감사의 최종 종료 보고가 끊겼으므로, 현 상태를 “완전히 발급 완료”로 과장하지 않는다. 다음 실행의 첫 단계는 이 최종 review와 real-bwrap 재검증이다.

## 18. 앞으로의 실행 순서

### Phase 0 — CONTEXT1 closure

1. 현재 validator/runtime 변경의 independent read-only review를 끝낸다.
2. Dynamic governance roles, full mount exact-cover, denied-root pre-open, ledger rollback 음성 테스트를 재확인한다.
3. 694-test fixed suite와 real bubblewrap 4 tests를 통과시킨다.
4. Active 35 files를 `0444`로 동결한다.
5. `create-test-receipt → create-source-snapshot → create-pretrain-authorization` 순서로 create-once trio를 발급한다.
6. Fresh roots absent, ledgers closed, source hash exactness를 마지막으로 검증한다.

### Phase 1 — V8R4 efficiency gate

고정 unit은 outer 3, seed 20260828, H0의 2-epoch workload다.

- train processed windows/epoch: 6,583
- validation processed windows/epoch: 1,658
- epoch 2 steady time: ≤23 seconds
- accuracy metric은 efficiency gate에서 계산하지 않음

통과하지 못하면 과학 실패가 아니라 runtime/throughput 문제로 분류하고 새 context generation에서 원인을 교정한다.

### Phase 2 — V8R4 discovery 18 units

```text
outer folds {3,4}
× seeds {20260828,20260829,20260830}
× variants {H0,H1,H2}
= 18 jobs
```

각 unit은 outer-validation만 보고 raw anchor/hard source/fixed switch를 평가한다. Fold/seed별 best 선택은 금지하고 전체 18-unit evidence에서 공통 variant와 공통 release mode 하나만 잠근다. V2 locked common selection key보다 strict lexicographic improvement가 없으면 promotion하지 않는다.

### Phase 3 — V8R4 promotion OOF

Discovery를 통과한 공통 구조만 6 outer folds × 3 seeds = 18 units에서 target-free inference한다. 모든 18 prediction을 seal한 뒤 target와 exact join한다. 각 seed를 따로 여섯 accuracy gate, 7 masks, phases, non-overlap, strata, calibration, streaming, latency로 평가한다.

### Phase 4 — 실패 시 V8R5

V8R4가 실패하면 결과를 `retrospective_adaptive_failure`로 동결하고 차기 cause-aligned 구조를 새 contract/source/output root에서 실행한다.

## 19. V8R5 상세 설계: Axis-Preserving Risk-Routed DHFER-SNN

V8R5는 아직 측정 결과가 아닌 구현 예정 설계다.

### 19.1 J0_axis — 좌표-값 결합 복원

현재의 `f(x)+embedding → mean`을 폐기한다. Cell evidence와 radar/ratio/branch/candidate coordinate를 pooling 전에 비선형 결합한다.

#### Concatenation option

```text
h_i = MLP([evidence_i,
           radar_emb_i,
           ratio_emb_i,
           branch_emb_i,
           normalized_candidate_rr,
           availability_i])
```

#### FiLM option

```text
h_i = gamma(coord_i,candidate_rr) ⊙ f(evidence_i)
    + beta(coord_i,candidate_rr)
```

그 뒤 다음 axial aggregation을 사용한다.

1. radar별 ratio attention
2. ratio별 radar attention
3. branch-aware summary
4. learned candidate query cross-attention
5. coordinate interaction 이후에만 pooling

필수 unit tests:

- 동일 mask에서 좌표 간 evidence swap 시 embedding이 유의하게 달라짐
- candidate 순서 permutation equivariance 유지
- unavailable cell 변경은 출력 불변
- all-masked path는 finite/fail-closed

### 19.2 J1_risk — 미분 가능한 risk router

각 expert에 다음 head를 둔다.

- expected absolute error
- `P(|error|>2)`
- `P(|error|>5)`
- predictive scale

중복·근접 후보를 단일 index 정답으로 벌하지 않도록 oracle-best 또는 tolerance 안의 expert 집합에 target mass를 나누는 equivalence-set target를 쓴다.

Training에서는 hard argmax 대신 router probability에 직접 gradient가 흐르는 비용을 추가한다.

```text
L_expected_cost = Σ_e p_e · |mu_e - y|
L_tail          = Σ_e p_e · softplus(|mu_e-y|-tau)
```

또는 differentiable soft-CVaR/entropic-risk를 사용한다. Hard argmax는 inference에서만 사용한다. Factor head의 +2 boost는 절대 결정기가 아니라 auxiliary prior/regularizer로 낮춘다.

### 19.3 J2_temporal_dro — causal tracking과 worst-group

- causal candidate track state
- 이전 selected expert와의 continuity
- risk-dependent switching hysteresis
- abrupt switch penalty
- identity/RR-band group DRO
- 25–35 bpm과 worst identity risk 강화
- 7 radar masks의 균형 노출

Identity/group은 training weight에만 사용하고 inference feature에는 넣지 않는다.

### 19.4 V8R5 학습 방식

Epoch보다 optimizer update 수를 사전 고정한다. 현재 H0의 235 updates는 너무 작다.

1. Axis encoder와 expert value head warmup
2. Soft risk-router warmup
3. Joint fine-tuning
4. Router temperature annealing
5. Common architecture/release rule lock
6. Held-identity calibration
7. Outer test를 제외한 15 non-test identities 전체로 fixed-update refit

Discovery ablation은 J0/J1/J2를 outer 3/4 × 3 seeds에서 비교하되, 한 cycle에 지배 원인과 구조 개입을 명시하고 결과를 숨기지 않는다.

## 20. 데이터 확장과 prospective 계획

모델만 복잡하게 만들어 상용 판정을 선언하지 않는다.

1. 현재 18명은 retrospective development cohort로 영구 분류한다.
2. 기존 identity와 겹치지 않는 prospective calibration cohort를 별도로 수집한다.
3. Calibration과도 겹치지 않는 prospective confirmation cohort를 추가한다.
4. 25–35 bpm, transition, motion, posture, clothing, distance/angle/placement, radar dropout을 의도적으로 oversample한다.
5. Radar/reference를 공통 hardware trigger 또는 측정 가능한 clock으로 동기화한다.
6. 가능하면 capnography 또는 adjudicated independent reference를 사용한다.
7. Reference failure와 radar signal failure를 구분하는 독립 quality annotation을 만든다.
8. Packet loss, range corruption, saturation, gain drift, displacement, clock skew, desynchronization, stuck radar를 device-level fault campaign으로 검증한다.
9. Site/device/day까지 group split에 포함하고 모든 scaler/router/calibrator를 group 밖 target에서 분리한다.
10. 최종 model, threshold, calibration, evaluation code를 target 공개 전에 잠근다.

## 21. 현재 상태를 정확히 해석하는 법

### 이미 완료된 것

- 원본 데이터 감사와 deterministic parsing
- causal radar preprocessing과 reference QC
- physical-identity 6-fold split
- structured 12-step SNN과 ensemble 평가
- 완전 nested proposer stack
- HCES v2 3-seed full locked OOF
- candidate oracle 기반 failure decomposition
- DHFER v3r1 구조·loss·trainer·tests
- outer-row-free V8R4 sealed packs
- GPU admission/ledger/SIGKILL recovery/target sandbox
- CONTEXT1 bridge와 ledger succession 코드
- 현재 fixed suite 690 pass/4 skip

### 아직 완료되지 않은 것

- 최신 CONTEXT1의 최종 independent audit 종료
- 최신 source에 대한 real-bubblewrap 4-test 재실행
- CONTEXT1 test receipt/source snapshot/pretrain authorization 발급
- V8R4 efficiency benchmark 성공
- V8R4 18-unit discovery
- common variant selection과 18-unit promotion OOF
- V8R5 구현·학습·평가
- prospective external cohort
- commercial release validation

## 22. 최종 판단

현재 프로젝트는 단순한 SNN prototype 단계는 넘었다. 데이터 parsing, causal preprocessing, identity-disjoint nested validation, 후보 생성, spiking graph/temporal model, target firewall, 실행 복구와 provenance가 실제 코드와 artifact로 존재한다.

그러나 현재 성능은 상용 gate를 통과하지 않았고, 같은 18명 코호트에서 반복적으로 구조를 바꿨기 때문에 retrospective 성능이 좋아지더라도 독립적인 상용 증거가 되지 않는다. 가장 유망한 과학적 단서는 후보 oracle이 목표 수준에 가깝다는 점이며, 가장 명확한 구조 결함은 좌표-값 대응 소실과 hard-routing risk gradient 부재다.

따라서 다음 작업은 무작정 모델 크기를 키우는 것이 아니라 다음 순서를 지키는 것이다.

```text
CONTEXT1 실행 closure
→ V8R4 fixed discovery/promotion을 정직하게 완료
→ failure taxonomy 확정
→ axis-preserving differentiable risk router V8R5
→ seed별 full OOF + masks + streaming + latency
→ independent prospective calibration/confirmation
→ device-level commercial release gate
```

이 순서에서 accuracy failure는 종료 신호가 아니라 다음 구조를 선택하는 증거다. 다만 독립 prospective data가 없을 때는 어떤 내부 수치도 상용 성능 확정으로 바꾸지 않는다.

## 23. 주요 근거 파일

- 데이터·기존 모델 보고서: [`REPORT.md`](../REPORT.md)
- 상용 목표: [`COMMERCIAL_SNN_GOAL_V2.md`](COMMERCIAL_SNN_GOAL_V2.md)
- 연속 실행 계획: [`COMMERCIAL_SNN_CONTINUOUS_EXECUTION_PLAN_V4.md`](COMMERCIAL_SNN_CONTINUOUS_EXECUTION_PLAN_V4.md)
- 기본 전처리 설정: [`configs/default.yaml`](../configs/default.yaml)
- HCES 설정: [`configs/harmonic_set_v2.yaml`](../configs/harmonic_set_v2.yaml)
- DHFER 설정: [`configs/harmonic_factor_router_v3.yaml`](../configs/harmonic_factor_router_v3.yaml)
- Radar/reference 전처리: [`src/snn_rr/preprocess.py`](../src/snn_rr/preprocess.py)
- Candidate feature 생성: [`src/snn_rr/harmonic_set_data.py`](../src/snn_rr/harmonic_set_data.py)
- HCES 모델: [`src/snn_rr/harmonic_set_models.py`](../src/snn_rr/harmonic_set_models.py)
- DHFER 모델: [`src/snn_rr/harmonic_factor_router_v3.py`](../src/snn_rr/harmonic_factor_router_v3.py)
- v3r1 sanitation wrapper: [`src/snn_rr/harmonic_factor_router_models_v3r1.py`](../src/snn_rr/harmonic_factor_router_models_v3r1.py)
- DHFER trainer/loss: [`scripts/train_harmonic_factor_router_snn_v3r1.py`](../scripts/train_harmonic_factor_router_snn_v3r1.py)
- v3r1 contract: [`ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json`](campaigns/directed_harmonic_factor_expert_snn_v3r1/ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json)
- HCES locked OOF: [`locked_hcs_oof_metrics.json`](runs/harmonic_candidate_set_snn_v2/hcs_locked_oof/locked_hcs_oof_metrics.json)
- v2 failure decomposition: [`v2_i3_failure_decomposition_for_adaptive_v3r1.json`](campaigns/directed_harmonic_factor_expert_snn_v3/diagnostics/v2_i3_failure_decomposition_for_adaptive_v3r1.json)
- H0 validation result: [`validation_metrics.json`](runs/directed_harmonic_factor_expert_snn_v3r1/discovery_v7/units/outer_3_seed_20260828_H0_no_factor/attempt_000/output/validation_metrics.json)
- CONTEXT1 failure diagnostic: [`v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_failure.json`](campaigns/directed_harmonic_factor_expert_snn_v3r1/diagnostics/v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_failure.json)
- CONTEXT1 correction authority: [`IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_BENCHMARK_ADMITTED_CONTEXT.json`](campaigns/directed_harmonic_factor_expert_snn_v3r1/IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_BENCHMARK_ADMITTED_CONTEXT.json)

