# 3-Radar 호흡수 추정을 위한 SNN 학습 방법론과 최종 권고안

- 기준일: 2026-08-30 (Asia/Seoul)
- 대상: 3대 XeThru UWB radar로 32초 causal window의 호흡수(RR)를 추정하는 모델
- 문서 상태: 현재 코드·데이터·실험 결과를 반영한 설계 기준서
- 주장 경계: 현재 결과는 retrospective engineering evidence이며 상용·의료 성능을 입증하지 않음

## 0. 결론부터 말하면

프로젝트의 취득 구조, raw radar 형식, 시간축, BIOPAC reference 생성, 전처리, 현재 SNN 계열, harmonic 후보 계열, 평가·누수 차단 구조까지 파악했다. 이 데이터에는 다음 조합이 가장 적합하다.

> **label-free analog radar front-end + 좌표를 보존하는 harmonic candidate graph + 8–12 step PLIF/ALIF residual SNN + surrogate-gradient 직접학습 + ANN teacher distillation + TET형 시간축 loss + distributional RR/불확실도/quality multi-task 학습**

이 선택의 핵심 이유는 세 가지다.

1. 입력은 event camera처럼 이미 spike로 들어오는 데이터가 아니라 40 Hz로 샘플링된 dense radar 배열이다. 전체 raw map을 Poisson rate code로 바꾸는 완전-spiking 방식은 지연과 계산량을 늘리고 입력 정보를 불필요하게 확률화한다. 따라서 공간·스펙트럼 정리는 analog 연산으로 하고, 압축된 주파수/후보/episode 표현에서 SNN을 쓰는 hybrid 구조가 정확도와 효율의 균형이 가장 좋다.
2. 현재 오차의 지배적 원인은 좋은 RR 후보가 없는 것이 아니라, 고호흡수에서 radar peak가 실제 RR의 1/3 또는 1/4 부근에 생길 때 올바른 harmonic factor를 고르지 못하는 것이다. 그러므로 단일 scalar 회귀를 더 크게 만드는 것보다 direct RR posterior와 harmonic candidate router를 함께 학습해야 한다.
3. 사용 가능한 물리적 identity가 18명뿐이므로 대형 Spiking Transformer를 처음부터 학습하는 전략보다, 작은 coordinate-aware SNN을 identity-disjoint 방식으로 강하게 규제하고 ANN teacher와 label-free pretraining을 이용하는 편이 일반화 가능성이 높다.

다만 **새로 교정한 acquisition contract 기준 학습은 아직 열 수 없다.** 최신 source와 일치하는 smoke reconstruction은 S02/S03/S30 세 세션만 포함하며 sync 승인과 scientific eligibility가 모두 `0/3`이다. 한 차례 생성된 29 usable-session diagnostic도 `0/29`였지만, 최신 timing/protocol/range 코드 이전 산출물이므로 현재 source의 완결 증거로 승계할 수 없다. 즉, 현재 사용할 수 있는 full-cohort strict cache와 corrected OOF 결과는 없다. 따라서 다음 두 결과를 엄격히 구분해야 한다.

- 기존 canonical cache: absolute-epoch와 nominal-grid 정렬로 만든 역사적 cache이며, 현재 구조를 비교하고 retrospective baseline을 재현하는 데 사용 가능
- 새 acquisition-aware cache: 수동 marker 검토·승인 또는 더 강한 동기화 근거가 생긴 뒤에만 과학적 학습·평가에 사용 가능

즉, 추천 모델 구조는 분명하지만, corrected-data 성능을 지금 바로 상용 성능으로 주장할 수 있는 상태는 아니다.

---

## 1. 문제를 정확히 정의하기

### 1.1 모델이 받아야 하는 정보

배포 시 모델이 받을 수 있는 것은 오직 radar에서 인과적으로 얻을 수 있는 정보다.

- 3대 radar의 raw payload 또는 그로부터 만든 causal feature
- 각 radar의 실제 availability mask
- measured radar timestamp로 계산한 시간 간격
- 과거 window에서만 만든 causal state
- target-free range-track confidence, missing, multimodal flag
- radar signal 자체에서 계산한 spectrum, SVD component, peak/candidate 정보

### 1.2 모델이 절대로 받아서는 안 되는 정보

다음은 target 작성·평가·표 층화에는 쓸 수 있지만 inference input에는 들어가면 안 된다.

- BIOPAC RSP waveform 또는 그 파생값
- `rr_bpm`, reference-valid 여부, reference quality, reference uncertainty
- `radar_observable`처럼 reference와 radar 예측의 일치로 만든 target-dependent proxy
- sync residual, marker 승인 상태, stage label처럼 배포 시 알 수 없는 annotation
- physical identity, session ID, protocol/action 이름
- 미래 window, centered smoothing, future-aware imputation

특히 BIOPAC은 **정답을 만드는 센서**이지 radar 추론을 돕는 센서가 아니다. 이 경계를 무너뜨리면 수치가 좋아져도 상용 배포 가능한 radar-only 모델이 아니다.

### 1.3 출력 문제는 scalar 하나가 아니다

최종 모델은 최소 다음 출력을 함께 가져야 한다.

1. `6–45 bpm`의 0.25 bpm grid에 대한 RR 확률분포
2. 그 분포의 expected RR
3. candidate/factor posterior와 선택된 hard expert
4. 예측 불확실도 또는 availability score
5. radar별 quality/reliability
6. spike rate와 state health 같은 운영 진단값

이렇게 해야 평균 RR뿐 아니라 harmonic ambiguity, 출력 거부, radar 결측, 운영 효율을 동시에 다룰 수 있다.

---

## 2. 현재 데이터 취득 구조를 모델 관점에서 다시 정리

### 2.1 센서와 세션

- 원본 session 폴더: 30개
- usable session: 29개 (`S24_KHJ`는 세 radar가 비어 제외)
- 물리적 identity: 18명
- radar: XeThru 3대
- BIOPAC RSP: 250 Hz, reference 전용
- radar 실제 frame rate: 약 40 Hz이지만 세션별 drift와 jitter 존재
- 모델 기본 window: 32초
- 모델 stride: 4초
- window overlap: 87.5%

29개 session을 29명으로 취급하면 안 된다. 같은 사람이 여러 session에 반복 등장하므로 모든 split의 단위는 session이 아니라 physical identity여야 한다.

### 2.2 raw radar record

확인된 record 형식은 다음과 같다.

```text
740 bytes
  = uint32 header 3개, 12 bytes
  + float32 payload 182개, 728 bytes
```

따라서 올바른 parser는 12-byte header를 명시적으로 분리하고 182개의 float payload만 반환해야 한다. 과거 MATLAB 도구의 `FL=185` float read 뒤 첫 float만 버리는 로직은 두 header word를 radar float처럼 포함시키므로 raw parser의 권위로 사용할 수 없다.

182 payload의 물리적 I/Q flattening 순서는 장비 명세로 확정되지 않았다. 현재 증거는 `I[0:91] + Q[0:91]`인 split-halves 가설을 interleaved 가설보다 지지하지만, 이것은 여전히 가설이다. 그러므로 phase branch는 다음 조건을 따라야 한다.

- raw-power branch는 기본 입력으로 유지
- split-I/Q phase branch는 명시적인 ablation으로만 사용
- 장비 명세나 독립 calibration으로 확정되기 전에는 phase branch를 필수 정보로 간주하지 않음
- frozen raw-only cache에서는 phase branch mask를 false, 값은 exact zero로 유지

### 2.3 measured time과 동기화

radar metadata v13에는 frame별 relative millisecond timestamp가 있다. nominal 40 Hz index로 시간을 합성하는 것보다 이 measured timestamp가 우선이다. 관측된 stream은 nominal 대비 약 `+199 ppm` 수준의 차이가 있고 세션 끝에서는 약 `0.23–0.39초` drift가 누적될 수 있다.

S03, S28, S30에는 timestamp plateau가 존재한다. frame sequence가 연속이고 payload가 서로 다른 경우에만 plateau 내부를 bounded interpolation으로 복원한다. 최대 수정량이 큰 S30은 자동으로 manual review를 요구한다. 이 보정은 시간을 임의로 매끄럽게 만드는 것이 아니라, 명시된 제한 안에서 중복 timestamp만 복원하고 provenance를 남기는 절차다.

실험 marker는 왼손으로 BIOPAC chest sensor를 눌러 RSP에 큰 신호를 만들면서 오른팔을 약 90도 움직여 radar에도 motion peak를 만드는 방식이다. MATLAB 도구는 RSP `>8.5 V`, 4초 이내 event merge, 앞 300초 marker 탐색, ±12초 offset 탐색과 수동 slider를 사용했다.

중요한 현재 상태는 다음과 같다.

- 최신 source-consistent smoke(S02/S03/S30)의 자동 sync 승인: `0/3`
- 같은 smoke의 scientific-eligible session: `0/3`
- 과거 full diagnostic의 결과는 `0/29`였지만 최신 `radar_timing`·protocol contract·range artifact를 완전히 반영하지 않아 현재 권위 artifact가 아님
- 따라서 지금 존재하는 full-cohort strict feature cache와 strict OOF는 없음
- 진단용 alignment candidate는 만들 수 있지만, 승인 전 radar를 BIOPAC clock에 강제로 이동시키면 안 됨

현재 sync 자동 승인 config의 핵심 gate는 최소 marker pair 3개, confidence 0.80 이상, RMSE 0.30초 이하, 최대 residual 0.75초 이하, drift 1000 ppm 이하이다. 자동 accept가 아니면 동일 sync receipt의 content hash에 정확히 결합된 manual approval이 필요하다. timestamp plateau 수정이 50 ms를 넘으면 자동 승인을 금지하고 manual review로 보낸다.

### 2.4 실험 protocol

추가된 가이드와 스프레드시트에서 확인된 protocol은 대략 일곱 phase다.

1. 착석 호흡과 세 방향 자세
2. 정상·느린 호흡, breath hold, 운동 후 회복
3. 두 종류 pickup course
4. fall scenario/course
5. 16-cell timed course
6. continuous round trip
7. session별로 배정된 자유 동작(Dodge/Strike/Kick 중 하나)

초기/후기 slide 사이에는 phase 2 총시간과 fall scenario 수의 문서 불일치가 있다. 따라서 stage reconstruction은 세션 전체의 action suffix를 정답처럼 쓰지 않고, 시간 순서·duration/gap prior·수동 anchor를 이용한 ordered dynamic programming으로 수행한다. transition 또는 불확실 구간은 `eligible_for_stage_metrics=false`로 제외해야 한다.

기존 `REPORT.md`의 Dodge/Strike/Kick별 표는 이 일곱 phase의 실제 행동별 성능표가 아니다. legacy `protocol_for_session()`이 phase 7 배정값을 session 전체 window에 broadcast한 batch label에 가깝다. 따라서 그 표로 “Kick가 가장 어렵다” 같은 활동별 결론을 내릴 수 없다. corrected cache에서는 이를 `acquisition_batch`로 격리하고, 실제 stage는 `acquisition_phase`, phase name/status/confidence로 별도 보존해야 한다. phase 7 assignment는 phase 7 core window에서만 의미가 있다.

radar marker envelope는 target-free radar 파생값이지만 RSP marker, spreadsheet interval, stage annotation은 reference-side/offline label construction 정보다. 이들은 학습 input이나 router feature가 될 수 없다.

### 2.5 사람 위치를 따로 찾는가

현재 range tracker는 radar마다 likely person/motion range bin을 인과적으로 추적한다. 출력에는 대표 bin, confidence, missing, multimodal 상태가 포함된다. 다만 range calibration이 보존되지 않았으므로 지금 말할 수 있는 것은 `range-bin index`이지 미터 단위 거리나 3차원 사람 좌표가 아니다.

이 tracker는 다음 원칙을 지킨다.

- BIOPAC과 RR target을 읽지 않음
- 과거와 현재 radar만 사용
- 여러 강한 peak가 있으면 multimodal로 표시
- 신뢰도가 낮으면 억지로 한 사람 위치를 확정하지 않음
- 기존 flat auxiliary와 섞지 않고 별도 `range_aux` provenance를 유지

여기서 “tracker가 구현됐다”와 “현재 leader가 tracker 출력을 사용한다”는 다른 말이다. 현재 build 경로는 summary를 `range_aux.npy`로 저장하지만 `FeatureCache`, `CachedRadarDataset`, `TriRadarRRSNN` 입력까지 아직 연결되지 않았다. 역사적 leader가 사람 위치를 명시적으로 추적해 쓰는 것은 아니며, 기존 모델 안에는 target-free range tracker가 아니라 learned range attention만 있다. 또한 이 tracker는 사람 분류기나 다중-person identifier가 아니라 **가장 강한 active motion/respiration range-coordinate tracker**다.

split-halves layout을 session 전체 target-free 통계로 고르는 절차도 online-causal 선택은 아니다. 배포에서는 hardware contract로 고정하거나 training/calibration에서 미리 freeze해야 하며, test session 전체를 보고 layout을 고르면 transductive preprocessing이 된다.

---

## 3. 현재 전처리·denoising 구조

### 3.1 raw 이상치 처리

현재 canonical 전처리는 `|x| > 0.1`인 sample을 같은 range bin의 직전 최대 4 frame에 있는 정상 finite 값의 median으로 대체한다. 이전 정상값이 없으면 0을 쓴다. 미래 frame을 보지 않는 past-only 복원이다. 전체 cache에서 실제 교체된 값은 한 sample이었다.

### 3.2 시간축 처리

기존 baseline은 nominal 40 Hz를 4-frame non-overlap mean으로 줄여 10 Hz를 만들었다. 이는 구현상 75 ms aggregation latency를 만든다. 새 경로에서는 measured timestamp를 공통 시간축으로 사용하고, plateau 복원과 drift provenance를 contract로 묶어야 한다.

원칙은 다음과 같다.

- feature window 경계는 measured time에서 정의
- resampling은 radar끼리 공통 support 안에서만 수행
- BIOPAC clock으로의 이동은 sync authorization이 있을 때만 허용
- 짧은 tail을 채우기 위해 window를 뒤로 이동하지 않음
- train과 deployment가 동일한 causal resampler를 사용

### 3.3 range–frequency map denoising

각 32초 radar window의 현재 기본 처리는 다음과 같다.

1. 값 clip
2. range-bin별 mean 제거
3. range-bin별 linear detrend
4. Hann taper
5. FFT
6. `0.08–0.80 Hz` band 선택
7. frequency 방향의 가벼운 smoothing
8. range pooling
9. range별 robust noise floor 추정
10. `log1p(power/noise)` 압축
11. temporal activity weight와 robust scaling

이 처리는 respiration spectrum을 안정화하지만 motion과 harmonic을 완전히 제거하지는 않는다. 오히려 harmonic 구조는 router가 판단할 수 있도록 보존해야 한다. 지나친 band smoothing이나 top-1 peak만 남기는 denoising은 중요한 ×2/×3/×4 증거를 지울 수 있다.

별도의 학습형 neural denoiser는 현재 구현돼 있지 않다. BIOPAC reference에는 0.10–0.75 Hz Butterworth `sosfiltfilt`가 쓰이지만, 이는 미래 sample을 사용하는 zero-phase 처리이므로 label 작성 경로에만 존재한다. radar inference 경로로 복사하면 causal deployment 계약을 위반한다.

### 3.4 SVD source separation

gross motion과 호흡 component를 분리하기 위해 raw, standardized raw, temporal velocity, range difference 등의 label-free view에 randomized SVD를 적용한 경로가 있다. radar/window/view별 component를 만들고 respiration band에서 component power, concentration, entropy, reliability를 계산한다.

SVD는 유용한 보조 증거이지만 다음 한계가 있다.

- component 순서는 사람이나 세션 사이에 고정된 의미가 아님
- 가장 큰 singular component가 호흡이라는 보장이 없음
- 과도한 component selection은 target을 사용한 oracle이 되기 쉬움
- 따라서 component evidence를 candidate graph의 masked 통계로 사용하고, BIOPAC로 component를 고르는 것은 금지

### 3.5 역사적 canonical feature cache의 실제 형태

| artifact | 형태 | 의미 |
|---|---|---|
| `maps.npy` | `[N, 3, 73, 182]`, float16 | radar × respiration-frequency × raw range/branch map |
| frequency grid | 약 0.085449–0.788574 Hz, 73 bins | RR posterior로 가기 전 물리 frequency topology |
| `aux.npy` | `[N, 1205]`, float32 | radar별 spectrum, global diagnostics, causal history |
| `range_aux.npy` | 새 acquisition-aware 경로의 별도 artifact | causal range tracker summary; 현재 leader 입력에는 미연결 |

structured auxiliary path는 cache의 spectrum을 하나의 flat MLP로만 소거하지 않고 Conv1D로 주파수 topology를 보존한다. train-only robust median/IQR scaler를 쓰며 structural unavailable cell은 scaling 뒤 exact zero가 돼야 한다.

---

## 4. 지금까지 구현된 SNN 구조

### 4.1 현재 정확도 선두: structured TriRadarRRSNN

현재 검증된 기본 구조는 완전-spiking이 아니라 hybrid SNN이다.

```text
3 radar × range–frequency map
        │
        ├─ shared 2-D spatial encoder
        │    ├─ frequency/range coordinate channel
        │    ├─ residual Conv2D
        │    ├─ frequency 축 보존
        │    └─ range attention
        │
        ├─ radar reliability + availability-mask fusion
        │
        ├─ topology-preserving structured auxiliary fusion
        │
        └─ 12-step PLIF/LIF residual Conv1D backbone
             ├─ RR distribution
             ├─ expected RR
             ├─ uncertainty
             ├─ quality proxy
             └─ spike statistics
```

전체 2-D map을 spike train으로 변환하지 않고, analog spatial encoder가 공간 차원을 압축한 뒤 주파수 sequence를 spiking backbone이 처리한다. 이것이 이 데이터에서 합리적인 이유는 raw 입력이 native event가 아니기 때문이다.

정확히 말하면 현재 leader의 12 simulation step은 12개의 실제 radar time sample이 아니다. 한 window에서 만든 static fused frequency current를 같은 SNN에 12번 주입하고 membrane/spike가 누적되는 내부 계산축이다. input LIF와 두 residual block의 LIF는 fast-sigmoid surrogate(slope 25), learnable beta와 threshold, subtract reset을 사용한다. neuron state는 각 window의 `forward()` 시작에서 0으로 초기화된다. window 사이 정보는 recurrent membrane이 아니라 radar-only causal history feature 32개로 전달된다. readout은 평균 spike-rate feature와 마지막 membrane을 결합한다.

현재 leader 학습의 기준 loss와 optimizer는 다음과 같다.

- Gaussian soft-bin cross entropy
- `0.50 × SmoothL1(expected RR, target)`
- `0.10 × heteroscedastic NLL`
- `0.15 × quality BCE`
- `0.35 × teacher KL`, temperature 2
- `0.0005 × spike-rate penalty`
- AdamW, learning rate `1e-3`, weight decay `1e-4`
- batch 48, 최대 80 epoch, patience 12, gradient clip 5, AMP
- whole-radar coupled dropout 0.20
- train-only robust median/IQR scaler와 identity-balanced sampler

이 기본 leader loss에는 tail CVaR, factor loss, RR rebalancing이 활성화돼 있지 않다. `radar_observable`은 quality supervision에 사용된 target-dependent proxy일 뿐 독립 sensor-health ground truth가 아니며, 기본 학습이 reference-invalid row를 제외하므로 quality head를 전체-window validity detector라고 해석해서도 안 된다.

현재 validation-locked 두 SNN ensemble의 결과는 다음과 같다.

| 지표 | 현재 값 | 내부 목표 | 판정 |
|---|---:|---:|:---:|
| Overall MAE | 1.291 bpm | ≤1.000 | FAIL |
| Identity-macro MAE | 1.220 bpm | ≤1.000 | FAIL |
| RMSE | 2.410 bpm | ≤1.800 | FAIL |
| ±2 bpm | 80.79% | ≥90.0% | FAIL |
| >5 bpm | 6.23% | ≤3.0% | FAIL |
| 25–35 bpm MAE | 4.216 bpm | ≤2.000 | FAIL |

평가 population은 18명, 2,327개 reference-valid identity-disjoint OOF window다. 겹치지 않는 subset은 444개이며 MAE 1.570 bpm이다. 따라서 현재는 강한 연구 baseline이지만 상용 성능이 아니다.

이 모든 leader 수치는 `artifacts/cache/rf32s`의 역사적 absolute-epoch + nominal-grid 정렬 결과다. measured common timeline, marker-affine 승인, 7-stage annotation, plateau repair, `range_aux`를 모두 반영해 재구축·재학습한 결과가 아니다. 새 pipeline은 window 경계와 reference QC population 자체를 바꿀 수 있으므로 기존 checkpoint와 1.291 bpm을 corrected pipeline 성능으로 승계하면 안 된다.

### 4.2 harmonic candidate 계열

고호흡수에서 dominant radar peak가 실제 RR의 subharmonic으로 내려가는 문제를 다루기 위해 candidate bank와 graph router가 구현됐다.

- 최대 12개 candidate
- proposer expected/MAP RR
- posterior NMS mode
- classical RR ×1/×2/×3/×4
- radar별 peak
- 각 candidate 주변의 `1/4, 1/3, 1/2, 1, 2, 3, 4` ratio evidence
- RF와 SVD evidence
- anchor/candidate hard expert
- safe anchor fallback

현재 v3 feature width는 571이다.

| 블록 | 차원 | 구조 |
|---|---:|---|
| Core | 46 | candidate/source/proposer/거리·gap·quality descriptor |
| RF | 378 | 3 radar × 7 ratio × 2 branch × 9 statistic |
| SVD | 147 | 3 radar × 7 ratio × 7 statistic |

현재 v3/v3r1 ancestry에는 두 directed harmonic graph block, PLIF graph cell, PLIF/ALIF temporal factor cell, 8 simulation step, hidden 64, 40만 parameter cap이 구현돼 있다. 반면 pooling 전 coordinate-interaction과 soft-risk router는 기존 v3 구현이 아니라 이 문서의 CCHG/V8R5 권고안이다.

HCES v2는 config에 `simulation_steps: 8`이 표기돼 있어도 현재 forward가 graph/temporal layer마다 한 번 transition하는 구조이며 동일 block을 8회 반복 unroll하는 모델은 아니다. 따라서 HCES를 “8-step SNN”이라고 단순 비교하지 않고 실제 forward transition 수와 연산량을 함께 보고해야 한다.

### 4.2.1 후보 계열의 실제 측정 상태

| 계열 | 측정 결과 | 판정 |
|---|---|---|
| Raw-window SVD source SNN | locked final 약 MAE 1.294, macro 1.221 | leader를 안전하게 이기지 못해 기각 |
| Temporal/episode SVD SNN | discovery validation promotion 실패 | outer 승격 안 함 |
| HCES v2, 고정 3 seed | full OOF MAE 약 1.296 / 1.372 / 1.377 | 세 seed gate 실패 |
| HCES candidate oracle | 약 0.433–0.481 bpm | target-dependent 상한 진단일 뿐 |
| DHFER v3r1 H0 discovery | hard-source MAE 1.769, macro 1.740, 25–35 MAE 4.752; oracle 0.559 | full OOF 아님, winner 아님 |

DHFER H0 discovery는 optimizer update도 약 235회 수준이라 “방향성 graph 자체가 불가능하다”는 결론을 내릴 정도로 충분한 학습 증거는 아니다. 반대로 authorization receipt와 완전한 full OOF가 없으므로 아직 성능 후보로 승격할 수도 없다. v3r1/V8R4는 source/실행 계약 보강과 일부 discovery까지의 상태이고, 이 문서가 권하는 coordinate-interaction/soft-risk 구조는 아직 측정된 winner가 아니다.

### 4.3 발견된 구조적 문제

이전 router에는 중요한 표현 문제가 있었다. cell evidence에 coordinate embedding을 더한 뒤 전체를 평균하면, evidence와 `(radar, ratio, branch)` 좌표의 결합이 약하다. 서로 다른 radar/ratio cell의 evidence를 맞바꿔도 출력 변화가 거의 없는 permutation-like invariance가 발생했다.

따라서 새 구조는 다음 순서를 지켜야 한다.

```text
잘못된 형태:
  mean( evidence_encoder(x_cell) + coordinate_embedding(cell) )

권고 형태:
  mean_masked( nonlinear_joint_encoder(
      x_cell,
      radar_coordinate,
      ratio_coordinate,
      branch_coordinate,
      availability
  ))
```

즉, 좌표는 pooling 전에 evidence와 비선형으로 상호작용해야 한다. 단순히 embedding을 더하는 것으로는 “radar 1의 ×3 evidence”와 “radar 3의 ×1 evidence”를 충분히 구분하지 못한다.

### 4.4 hard routing과 risk loss의 불일치

현재 계열의 또 다른 문제는 hard argmax로 선택한 source의 CVaR/error를 계산하면, 선택 자체가 비미분이라 router logit으로 유용한 gradient가 전달되지 않는다는 점이다. expert 회귀는 좋아질 수 있어도 “어느 expert를 선택할지”는 tail loss에서 직접 배우지 못한다. 실제 audit에서는 release threshold가 0.8인데 선택된 source probability가 대체로 약 0.38–0.50인 불일치도 관찰됐다.

권고 원칙은 **soft-risk로 학습하고 hard rule로 배포**하는 것이다.

```text
training:
  pi_e = softmax(router_logit_e / temperature)
  L_route = sum_e pi_e · cost(expert_e, target)

deployment:
  e* = argmin_e predicted_risk_e
  if confidence/safety gate passes: emit expert e*
  else: safe anchor or no-estimate
```

각 expert는 RR mean/scale뿐 아니라 expected absolute error, `P(|error|>2)`, `P(|error|>5)`를 예측하도록 하고, training cost와 soft tail/CVaR가 router probability에 미분 가능하게 연결돼야 한다. 비슷한 BPM의 중복 candidate는 한 index를 억지로 정답으로 만들지 말고 equivalence set으로 supervision한다.

### 4.5 oracle이 알려주는 것과 알려주지 않는 것

target을 보고 가장 좋은 candidate를 고르는 oracle은 약 0.43–0.56 bpm 수준의 매우 낮은 MAE를 보이는 설정들이 있다. 이것은 candidate bank 안에 좋은 답이 자주 있다는 뜻이다. 하지만 oracle은 target-dependent이므로 배포 모델 성능이 아니다.

현재 과학적 해석은 다음과 같다.

- 표현 상한은 존재함
- 주요 병목은 candidate 생성보다 unseen identity에서의 candidate 선택
- alias 존재 여부는 비교적 잘 구분돼도 ×3과 ×4 중 올바른 correction magnitude는 불안정함
- 25–35 bpm non-overlap 표본이 42개뿐이라 router가 identity-independent rule을 배우기에 support가 부족함

---

## 5. SNN을 학습시키는 주요 방법론 비교

아래 평가는 “일반적인 SNN 우수성”이 아니라 이 radar RR 문제에 대한 적합성이다.

| 방법 | 핵심 아이디어 | 장점 | 이 데이터에서의 한계 | 권고 |
|---|---|---|---|:---:|
| STDP / local learning | spike timing의 국소 상관으로 weight 학습 | 생물학적 plausibility, label-free 가능 | 정밀 RR 회귀와 harmonic hard routing에 직접 최적화하기 어려움 | 낮음 |
| ANN→SNN conversion | 잘 학습된 ANN activation을 spike rate로 근사 | static benchmark 정확도 보존에 유리 | 긴 simulation step, rate coding 지연, stateful regression 대응이 약함 | baseline |
| Direct surrogate-gradient BPTT | spike 미분을 surrogate로 근사해 end-to-end 학습 | 짧은 T, 회귀·multi-task·state 학습 가능 | 메모리와 gradient 안정화 필요 | **핵심** |
| SLAYER류 temporal credit assignment | spike response와 시간축 error를 직접 전파 | event-time 구조에 강함 | 입력이 native event가 아니어서 추가 이점이 불확실 | 보조 ablation |
| PLIF/ALIF | membrane time constant 또는 adaptation을 학습 | 서로 다른 시간척도와 지속 ambiguity 표현 | 작은 데이터에서 과도한 자유도는 규제 필요 | **채택** |
| SEW residual SNN | spike element-wise residual로 깊은 SNN 안정화 | 깊은 block의 gradient와 identity path 개선 | 너무 깊게 만들면 18명 데이터에서 과적합 | **얕게 채택** |
| TET | simulation step별 loss와 시간 일관성을 함께 최적화 | 낮은 T에서도 각 step이 의미 있는 출력을 내도록 유도 | physical time과 simulation time을 혼동하면 안 됨 | **채택** |
| ANN teacher distillation | teacher posterior/feature를 SNN에 전달 | 작은 데이터, soft RR ambiguity, 안정적 초기화 | teacher 오류와 harmonic bias도 전달 가능 | **조건부 핵심** |
| Self-supervised pretraining | label 없이 radar 표현을 먼저 학습 | reference-invalid window도 사용 가능 | split-safe pretraining과 target-free objective 필요 | **강력 권고** |
| Spiking Transformer | attention을 spike 기반으로 수행 | 전역 관계와 cross-radar interaction | 데이터·연산 요구량이 크고 과적합 위험 | 소형 ablation |
| Stateful episode SNN | 과거 window state로 지속 harmonic episode 추적 | streaming에서 일시적 ambiguity 완화 | state reset·gap·invalid window 계약이 복잡 | 배포형에 권고 |
| e-prop / DECOLLE류 local online learning | eligibility trace 또는 layer-local loss로 online update | on-device continual learning과 낮은 BPTT memory | 현재 핵심인 supervised expert-selection gradient와 엄격한 model lock에 불리 | 장기 연구 |

### 5.1 STDP가 주 학습법이 아닌 이유

STDP는 radar 표현의 비지도 pretraining에는 실험할 수 있지만, 최종 목적은 `MAE`, catastrophic error, high-RR tail, calibrated availability를 동시에 낮추는 supervised 문제다. STDP만으로는 어떤 spike pattern이 18 bpm인지 27 bpm인지, 어느 candidate가 ×3 correction인지에 대한 직접적인 credit assignment가 부족하다.

따라서 STDP는 다음 정도로 제한하는 것이 맞다.

- encoder 초기화 ablation
- unlabeled motion/respiration pattern clustering
- neuromorphic hardware 친화성 탐색

최종 estimator 학습은 surrogate-gradient가 담당해야 한다.

### 5.2 ANN→SNN conversion이 주 방법이 아닌 이유

conversion은 activation magnitude를 spike count로 근사하므로 simulation step이 충분히 길 때 강하다. 그러나 이 프로젝트는 저지연 8–12 step을 목표로 하고, membrane state와 causal episode state 자체가 예측에 기여해야 한다. 긴 rate code는 다음 문제가 있다.

- full radar map에서 spike 수와 latency 증가
- 낮은 T에서 quantization error 증가
- uncertainty, hard candidate routing, state reset을 함께 학습하기 어려움
- dense sampled radar를 다시 stochastic spike로 바꾸면서 정보 손실

따라서 conversion model은 “같은 parameter/latency에서 direct-trained SNN이 실제로 이득인가”를 확인하는 비교 기준으로만 둔다.

### 5.3 surrogate-gradient 직접학습이 핵심인 이유

spike 함수 `H(u-θ)`는 거의 모든 지점에서 미분이 0이고 threshold에서 불연속이다. 학습 때만 그 미분을 fast-sigmoid, arctangent 등의 부드러운 함수로 대체하면 BPTT로 membrane dynamics, encoder, router, head를 함께 최적화할 수 있다.

개념적인 PLIF update는 다음과 같다.

```text
u_t = β · u_(t-1) + I_t - s_(t-1) · θ
s_t = H(u_t - θ)
β   = sigmoid(b)     # 학습 가능한 decay
```

ALIF는 최근 spike에 따라 threshold가 올라갔다가 천천히 복원되는 adaptation state를 추가한다. PLIF는 candidate graph의 다양한 스펙트럼 시간척도에, ALIF는 지속적인 episode state와 반복 activation 억제에 적합하다.

Surrogate-gradient 학습은 SNN 학습의 표준적인 직접 접근이며, 이론과 실무상의 장단점은 Neftci 등의 정리에 잘 설명돼 있다([IEEE Signal Processing Magazine 논문](https://arxiv.org/abs/1901.09948)). 학습 가능한 membrane time constant를 쓰는 PLIF는 Fang 등의 연구가 직접 근거다([ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.html)).

### 5.4 TET를 결합하는 이유

일반적인 SNN loss가 마지막 step 또는 모든 step 평균 출력만 맞추면, 초반 simulation step은 무의미하고 후반 step에만 정보가 몰릴 수 있다. TET형 loss는 각 step 출력도 target과 일관되게 만들고, 평균 출력의 안정성도 유지한다([ICLR 2022](https://openreview.net/pdf?id=_XNtisL32jv)).

이 프로젝트에서는 다음처럼 적용하는 것이 적합하다.

```text
L_TET = mean_t L_RR(p_t, y)
      + gamma · mean_t D(p_t || mean_t(p_t))
```

단, simulation step `t=1…T`와 radar의 실제 10 Hz physical time을 혼동하면 안 된다. 8–12 simulation step은 한 window/candidate 표현을 SNN이 전개하는 내부 계산축이고, 4초마다 들어오는 chronological window sequence는 별도의 episode 축이다.

### 5.5 ANN teacher distillation이 필요한 이유

현재 표본은 18 identity로 작고 RR posterior는 harmonic 때문에 다봉일 수 있다. hard scalar target만 쓰면 “정답 주변 분포”와 “teacher가 본 대체 mode”를 잃는다. ANN teacher의 soft posterior와 중간 topology를 증류하면 초기 optimization이 안정된다.

권고 distillation은 단순 logit KL 하나가 아니다.

- RR posterior temperature-KL
- coordinate-aware token feature alignment
- radar reliability alignment
- teacher confidence가 낮거나 harmonic tail에서 틀린 경우 KD weight 감소
- 학습 후반 KD weight decay

ANN에서 SNN으로 지식을 이전하는 방법의 한 예는 [CVPR 2023 ANN→SNN/KD 연구](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Constructing_Deep_Spiking_Neural_Networks_From_Artificial_Neural_Networks_With_CVPR_2023_paper.html)다. 중요한 것은 논문의 구조를 그대로 복제하는 것이 아니라, teacher의 bias를 무비판적으로 강제하지 않는 것이다.

### 5.6 self-supervised pretraining이 매우 유리한 이유

현재 9,576개 후보 window 중 reference-valid는 2,327개, 24.30%다. reference-invalid radar window도 radar 신호 자체는 유효할 수 있다. label-free objective를 사용하면 BIOPAC QC로 버려진 radar data를 표현 학습에 활용할 수 있다.

권고 objective는 다음과 같다.

- masked time–range 또는 frequency–range patch reconstruction
- 같은 시간의 서로 다른 radar 사이 contrastive agreement
- 인접 causal crop 사이 representation consistency
- range-track가 안정적인 구간에서 range-shift equivariance
- SVD view와 raw-power view의 cross-view consistency
- temporal order 판별

시계열 contrastive representation 학습의 대표적인 근거로 TS2Vec가 있다([AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/download/20881/20640)). 다만 이 프로젝트에서는 다음 split 규칙이 더 중요하다.

> outer fold `f`의 pretraining은 outer-train identity의 radar만 사용한다. validation/test identity의 unlabeled radar도 기본적으로 pretraining에 넣지 않는다.

transductive unlabeled test pretraining은 연구 목적에 따라 가능할 수 있지만, unseen-identity 일반화와 상용 claim에는 부적절하다. 외부 대규모 unlabeled radar corpus가 있다면 별도 provenance로 사용할 수 있다.

### 5.7 Spiking Transformer를 바로 주력으로 쓰지 않는 이유

Spikformer는 spike 기반 self-attention의 가능성을 보여준다([ICLR 2023](https://openreview.net/pdf?id=frE4fUwz_h)). 그러나 현재 데이터에서는 대형 attention model보다 sample support가 먼저 부족하다. 사용할 경우 다음처럼 제한한다.

- hidden 64 이하의 작은 cross-radar/candidate attention
- attention 전에 coordinate interaction 완료
- parameter-matched Conv/graph SNN과 비교
- outer-validation에서만 선택

즉, Transformer는 후보 구조이지 현재 최우선 구조가 아니다.

### 5.8 spike encoding은 무엇을 선택해야 하는가

SNN에서 neuron 종류만큼 중요한 것이 analog feature를 spike/current로 넣는 방법이다.

| encoding | 설명 | 이 프로젝트 판단 |
|---|---|---|
| Poisson rate code | feature 크기에 비례한 stochastic spike 생성 | dense spectrum을 다시 noisy하게 만들고 낮은 T variance가 커서 비권고 |
| Deterministic rate/current | 같은 analog current를 여러 simulation step에 주입 | 현재 leader가 사용하는 방식이며 기본값으로 적합 |
| Latency/time-to-first-spike | 큰 값일수록 빨리 spike | 양·음, 다봉 spectrum과 missing mask 표현이 까다로워 보조 실험 |
| Delta/event code | physical time에서 변화가 있을 때만 event | chronological episode 또는 raw slow-time 변화에 유망한 ablation |
| Learned spike encoder | 작은 network가 spike timing을 학습 | 데이터가 작아 encoder 자유도와 과적합을 엄격히 제한해야 함 |

기본 선택은 deterministic direct-current encoding이다. static window token은 8–12 내부 step에 current로 넣고, chronological episode에서는 4초마다 새 token과 명시적 time-gap을 state cell에 넣는다. Poisson randomness는 기본 경로에서 제거해 seed variance와 latency를 줄인다.

### 5.9 local/online learning의 위치

e-prop이나 DECOLLE류 방법은 device에서 개인 적응을 해야 할 때 가치가 있다. 하지만 현재는 18명 offline cohort, fixed release model, 엄격한 target leakage 통제가 우선이다. on-device adaptation은 calibration drift와 안전성 검증 범위를 크게 넓힌다. 따라서 먼저 surrogate BPTT 모델을 잠그고, 이후 별도 continual-learning protocol에서만 비교한다. 관련 원 방법론은 [e-prop](https://doi.org/10.1038/s41467-020-17236-y)과 [DECOLLE](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00424/full)에서 확인할 수 있다.

---

## 6. 최종 권고 모델: Causal Coordinate-aware Harmonic Graph SNN

문서에서 제안하는 최종 연구 구조를 `CCHG-SNN`이라고 부른다. 이름보다 중요한 것은 아래 계약이다.

> **상태 주의:** CCHG-SNN, TET 통합, SEW-style block, split-safe SSL, per-expert soft-risk head의 전체 조합은 권고 설계이며 아직 full OOF로 학습·측정된 모델이 아니다. 현재 측정 leader는 4.1의 structured TriRadarRRSNN ensemble이다.

### 6.1 전체 흐름

```text
Measured-time radar payload, 3 views
        │
        ├─ strict raw parser / causal repair / timestamp contract
        ├─ robust normalization and mask
        ├─ raw range–frequency map
        ├─ label-free SVD evidence
        └─ causal range-track auxiliary
        │
        ▼
Shared analog spatial encoder
        │ frequency topology preserved
        │ range attention conditioned on target-free track confidence
        ▼
Per-radar direct RR tokens + harmonic candidate tokens
        │
        ├─ nonlinear evidence × coordinate interaction
        │    radar ID, ratio, branch, frequency/range coordinate
        │
        ├─ directed harmonic graph
        │    near, ×2, ×3, ×4 relations
        │
        └─ radar availability-aware pooling
        ▼
8–12 step SEW-residual PLIF/ALIF SNN
        │
        ├─ direct RR distribution head
        ├─ candidate/factor hard-router head
        ├─ per-expert soft-risk head
        │    E|error|, P(|e|>2), P(|e|>5), scale
        ├─ heteroscedastic uncertainty / availability head
        ├─ radar quality head
        └─ spike/energy diagnostics
        ▼
Validation-locked decision
        ├─ safe anchor fallback
        ├─ hard candidate expert when sufficiently supported
        └─ otherwise no-estimate / low-confidence flag
```

### 6.2 왜 analog front-end를 남기는가

spike를 쓰는 목적은 모든 연산을 억지로 spike로 바꾸는 것이 아니라, 반복적 sparse state computation이 유리한 부분에서 효율과 temporal inductive bias를 얻는 것이다.

analog로 남길 부분:

- raw parser와 timestamp resampling
- FFT, robust normalization, SVD
- 2-D spatial compression
- coordinate construction과 masking

spiking으로 만들 부분:

- 압축된 frequency/candidate token sequence
- directed graph message update
- factor evidence accumulation
- chronological episode state

이렇게 해야 full-map rate coding의 비용을 피하면서도 SNN의 state와 sparse activation을 사용한다.

### 6.3 direct RR path와 candidate path를 둘 다 유지

candidate router만 쓰면 candidate bank가 빠진 경우 또는 router가 불확실할 때 안전성이 나빠진다. direct RR posterior만 쓰면 harmonic tail이 다시 무너진다. 따라서 두 경로가 필요하다.

```text
p_direct(r | x)       : 연속 RR grid posterior
p_candidate(k | x)    : candidate k posterior
p_factor(f | x)       : ×1/×2/×3/×4 posterior
p_anchor              : 검증된 causal proposer/직접 경로 fallback
```

최종 numeric estimate는 arbitrary midpoint가 아니라 validation에서 잠근 hard rule을 사용한다.

1. 모든 source가 unavailable이면 numeric estimate를 내지 않음
2. candidate/factor support가 충분하고 safety condition을 만족하면 hard candidate 선택
3. 그렇지 않으면 safe anchor/direct posterior로 fallback
4. outer-test 결과를 보고 threshold나 blend weight를 다시 바꾸지 않음

학습 시에는 이 hard rule을 그대로 미분하려고 하지 않는다. candidate/anchor별 soft responsibility와 risk-weighted mixture로 router에 gradient를 주고, temperature를 점차 낮춘다. hard argmax, hysteresis, release threshold는 held-identity validation에서 잠근 뒤 inference에만 적용한다.

### 6.4 coordinate-interaction encoder

각 cell의 입력을 다음처럼 정의한다.

```text
z_cell = MLP([
    normalized_evidence,
    radar_embedding,
    log_ratio_embedding,
    branch_embedding,
    candidate_rr_coordinate,
    range/frequency_coordinate,
    availability_bit
])
```

그 뒤에만 masked pooling 또는 graph message passing을 한다. 이 구조가 해결하려는 것은 단순한 위치 encoding이 아니라 evidence의 의미가 좌표에 따라 달라지는 문제다.

필수 unit test:

- 두 radar cell의 evidence swap 시 출력이 유의하게 변함
- ×1과 ×3 cell swap 시 출력이 유의하게 변함
- unavailable cell에 어떤 finite/NaN 값을 넣어도 sanitization 후 출력이 같음
- masked cell은 scaling 전후 exact `+0.0`
- candidate order를 함께 permute하면 equivariant, evidence만 잘못 permute하면 불변이 아님

### 6.5 directed harmonic graph

candidate node `i→j` 관계는 BPM 비율로 만든다.

- near
- receiver ≈ 2× sender / sender ≈ 2× receiver
- receiver ≈ 3× sender / sender ≈ 3× receiver
- receiver ≈ 4× sender / sender ≈ 4× receiver

방향성이 필요한 이유는 “9 bpm이 27 bpm의 1/3일 가능성”과 “27 bpm이 9 bpm의 3배일 가능성”이 같은 의미가 아니기 때문이다. self 정보는 graph self-edge를 중복해 넣기보다 residual path로 유지한다.

### 6.6 PLIF/ALIF와 residual 방식

권고 기본값:

- graph cell: PLIF
- episode/factor state: PLIF + ALIF 병렬 또는 직렬 비교
- simulation steps: 8, 12를 주 비교
- hidden channels: 64부터 시작
- graph block: 2개
- dropout: 0.05–0.15
- parameter cap: 우선 400k 내외 router, 전체 front-end 포함 모델은 현재 1.3M baseline과 parameter-matched 비교

깊은 SNN에서 residual path가 spike operation 때문에 깨지는 문제를 줄이기 위해 SEW-style residual을 사용한다. SEW-ResNet은 깊은 SNN의 residual 설계를 다룬다([NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/afe434653a898da20044041262b3ac74-Abstract.html)). 이 데이터에서는 깊이를 과도하게 늘리지 않고 2–4 block 범위에서만 비교한다.

### 6.7 stateful streaming

한 32초 window 내부의 simulation state와 window 사이의 episode state를 구분한다.

- simulation state: 매 window의 내부 8–12 step 계산
- episode state: 같은 session에서 과거 4초 stride window들이 전달하는 causal state

episode 학습 규칙:

- session 순서 보존
- truncated BPTT chunk 예: 32 windows
- 첫 2 windows는 warmup으로 loss를 약화하거나 제외
- reference-invalid window도 state는 update할 수 있으나 RR supervised loss는 mask
- session 변경, 큰 time gap, clock discontinuity에서 state reset
- radar mask 변경은 reset이 아니라 state input으로 명시
- offline batch와 chronological streaming 출력 parity 검사

episode router에는 candidate switching penalty와 hysteresis를 적용할 수 있다. 다만 “항상 이전 candidate를 유지”하는 방식은 실제 RR transition을 늦출 수 있으므로, time-gap과 predicted transition risk를 함께 쓰고 validation에서 지연–안정성 trade-off를 측정한다.

---

## 7. RR target과 학습 loss

### 7.1 scalar보다 distributional target이 좋은 이유

RR grid를 `r_j`라 하고 BIOPAC reference를 `y`, reference uncertainty heuristic을 `σ_y`라 하면 Gaussian soft target은 다음과 같다.

```text
q_j ∝ exp(-(r_j - y)^2 / (2σ_y^2))
sum_j q_j = 1
```

모델의 RR posterior `p_j=softmax(logit_j)`에서 expected RR은 다음과 같다.

```text
mu = sum_j p_j · r_j
```

이 방식은 0.25 bpm bin 경계에서 hard class가 튀는 문제를 줄이고 posterior entropy를 uncertainty feature로 사용할 수 있다.

### 7.2 권고 총 loss

권고 loss는 다음 구조다.

```text
L_total =
    λ_dist    · KL(q || p_direct)
  + λ_reg     · SmoothL1(mu_direct, y)
  + λ_nll     · HeteroscedasticNLL(y | mu, sigma)
  + λ_kd      · KD(teacher, student)
  + λ_list    · CandidateListwiseLoss
  + λ_route   · SoftExpectedExpertRisk
  + λ_risk    · ExpertErrorProbabilityLoss
  + λ_factor  · ConfidentFactorFocalLoss
  + λ_margin  · WrongHarmonicMarginLoss
  + λ_cons    · JS(p_factor, aggregate(p_candidate))
  + λ_quality · TargetFreeQualityBCE
  + λ_tail    · CVaR20(error)
  + λ_tet     · TemporalEfficientLoss
  + λ_spike   · SpikeRatePenalty
```

각 항의 역할은 다음과 같다.

| 항 | 역할 | 적용 조건 |
|---|---|---|
| Distribution KL | 전체 RR posterior 학습 | reference-valid window |
| SmoothL1 | expected RR의 robust 회귀 | reference-valid window |
| Heteroscedastic NLL | 큰 noise와 prediction scale 학습 | reference-valid, calibration은 별도 |
| KD | teacher의 soft topology 전달 | train identity, teacher 신뢰도 gate |
| Listwise candidate | 후보 전체 순위 최적화 | candidate 존재 + reference-valid |
| Soft expert risk | `sum(pi_e × expert_cost_e)`로 선택 gradient 생성 | reference-valid, soft router |
| Error probability | expert별 expected AE, >2, >5 risk 학습 | train target으로 만들되 input 금지 |
| Factor focal | 희귀 ×2/×3/×4 학습 | factor label이 충분히 명확한 경우만 |
| Wrong-harmonic margin | catastrophic factor 선택 억제 | confident factor supervision |
| JS consistency | factor와 candidate posterior 정합성 | factor/candidate available |
| Quality BCE | missing/multimodal/신호 건전성 | **신규 제안:** target-free signal-health label |
| CVaR20 | 상위 20% tail error 압력 | soft mixture/per-expert cost로 router까지 미분 |
| TET | 낮은 simulation step 안정성 | 모든 supervised SNN step |
| Spike penalty | firing rate·energy 제어 | 정확도 안정화 뒤 점진 적용 |

### 7.3 시작점으로 권하는 loss weight

아래 값은 최종 확정값이 아니라 첫 grouped-validation search의 중심점이다.

| loss | 중심값 | 탐색 범위 |
|---|---:|---:|
| `λ_dist` | 1.0 | 고정 기준 |
| `λ_reg` | 0.5 | 0.25, 0.5, 1.0 |
| `λ_nll` | 0.10 | 0.05, 0.10, 0.20 |
| `λ_kd` | 0.50→0.10 | 초기 0.25–1.0, 후반 decay |
| `λ_list` | 0.50 | 0.25, 0.50, 1.0 |
| `λ_route` | 0.50 | 0.25, 0.50, 1.0 |
| `λ_risk` | 0.20 | 0.10, 0.20, 0.40 |
| `λ_factor` | 0.20 | 0.10, 0.20, 0.40 |
| `λ_margin` | 0.10 | 0.05, 0.10, 0.20 |
| `λ_cons` | 0.05 | 0.02, 0.05, 0.10 |
| `λ_quality` | 0.10 | 0.05, 0.10 |
| `λ_tail` | 0→0.10 | warmup 후 0.05, 0.10 |
| `λ_tet` | 0.10 | 0.05, 0.10, 0.20 |
| `λ_spike` | 0.0005 | 0.0001–0.001 |

모든 조합을 거대한 grid로 돌리면 18명 cohort에 과적합한다. discovery fold에서 구조적 ablation을 좁힌 뒤, 고정된 소수 조합만 nested validation으로 비교해야 한다.

`λ_route`와 `λ_tail`에서 사용하는 예측은 hard argmax 결과가 아니라 soft router probability로 가중한 expert cost여야 한다. hard selection error에만 CVaR를 걸면 선택 logit까지 gradient가 흐르지 않는 현재 문제를 반복한다. 학습 후반 temperature annealing으로 soft/hard gap을 줄이고, 최종 hard policy는 held-identity validation에서 다시 검사한다.

이 표의 target-free quality BCE는 현재 leader loss를 그대로 기술한 것이 아니라 교체 권고다. 현재 leader의 quality BCE target은 `radar_observable`, 즉 classical radar estimate가 reference의 ±2 bpm 안에 들었는지로 만든 target-dependent proxy다. 새 모델에서는 이를 inference input으로 절대 넣지 않고, signal-health head는 missing/flatline/multimodal/packet corruption 같은 target-free label로 학습한다. 예측 오류 가능성은 별도의 risk head가 train-time reference error로 학습하되 그 target도 inference input으로 들어가지 않는다.

### 7.4 factor label의 안전한 생성

factor class는 reference를 사용해 train-time label로 만들 수 있지만, ambiguous sample을 억지로 hard label로 만들면 안 된다.

예를 들어 classical RR을 `c`, reference를 `y`라 할 때 `y/c`가 1, 2, 3, 4 중 하나에 충분히 가깝고, 후보 간 target distance margin이 충분할 때만 confident factor loss를 적용한다. 그렇지 않은 window는 direct RR loss와 listwise soft target만 사용한다.

factor label은 학습용 target이며 inference feature가 아니다.

### 7.5 uncertainty의 의미

현재 uncertainty는 error ranking에는 쓸 수 있지만 calibrated confidence interval이라고 부르면 안 된다. 권고 uncertainty는 다음을 분리한다.

- aleatoric scale: `sigma(x)`
- RR posterior entropy
- direct/candidate disagreement
- radar ensemble disagreement
- availability probability

threshold와 interval scaling은 outer-test가 아니라 별도의 held-identity calibration set에서만 정한다. uncertainty-aware SNN regression의 한 예는 [ICONS 2025 연구](https://ir.cwi.nl/pub/36110/36110.pdf)에서 볼 수 있지만, 프로젝트의 calibration claim은 자체 held-out 검증으로 다시 입증해야 한다.

---

## 8. 학습 curriculum: 처음부터 release까지 하나의 연속 계약

이 절차는 막연한 “나중에 할 다음 단계” 목록이 아니라, 어떤 artifact가 다음 연산을 열 수 있는지를 명시한 실행 순서다.

### Phase A — acquisition authority 확정

입력:

- raw 740-byte radar records
- radar metadata v13
- BIOPAC RSP
- 실험 가이드, spreadsheet, marker 도구

필수 산출물:

- session별 raw parser receipt
- measured timestamp/plateau repair receipt
- marker sync candidate
- 수동 승인 또는 승인 거절 receipt
- stage reconstruction confidence
- causal range-track artifact
- content hash로 묶인 acquisition manifest

통과 조건:

- train에 포함할 session이 `sync_authorized=true`
- `alignment_scientific_eligible=true`
- required hash와 session count가 일치
- stage metric은 별도로 `eligible_for_stage_metrics=true`

현재 source-consistent 권위 artifact에는 승인된 session이 없고 full cohort reconstruction도 완결되지 않았다. 따라서 새 corrected cache의 scientific training gate는 닫혀 있다.

코드 연결도 아직 완결되지 않았다. `load_feature_cache(require_acquisition_contract=True, require_scientific_eligible=True)`의 fail-closed 검증은 구현됐지만 기본 `scripts/train.py` entrypoint는 아직 이 strict flag를 강제하지 않는다. corrected scientific run을 열기 전에 CLI와 training entrypoint가 strict mode를 반드시 호출하도록 연결하고, legacy/acquisition session 혼합을 거부해야 한다. 또한 reconstruction root의 stated pipeline hash에는 실제 의존성인 `src/snn_rr/radar_timing.py`와 `src/snn_rr/data.py`도 포함돼야 한다.

### Phase B — split-safe label-free pretraining

각 outer fold마다 outer-train identity만 사용한다.

1. raw/spatial encoder pretraining
2. radar-view consistency
3. masked spectrum reconstruction
4. range-track consistency
5. SVD/raw cross-view agreement

산출물에는 다음을 저장한다.

- 사용 identity 목록
- raw/session/acquisition manifest hash
- augmentation config hash
- encoder checkpoint hash
- pretraining objective와 epoch

### Phase C — analog teacher 학습

teacher는 student와 동일한 coordinate topology와 candidate graph를 사용하되 neuron을 analog SiLU/GELU로 둔다. teacher가 student보다 무조건 크거나 복잡할 필요는 없지만, posterior와 token feature가 안정적이어야 한다.

- train identity로 weight/scaler 학습
- validation identity로 early stopping
- test identity target은 봉인
- high-RR macro와 catastrophic rate를 함께 checkpoint key로 사용
- teacher checkpoint에 split/config/cache hash 저장

### Phase D — direct-trained SNN distillation

1. pretrained analog front-end load
2. PLIF/ALIF state 초기화
3. 초기 epoch: KD와 distribution loss 중심
4. 중기: candidate/factor, TET 활성화
5. 후기: KD 감소, CVaR/spike penalty 점진 증가
6. gradient clipping, AMP, deterministic request와 실제 nondeterminism 기록

epoch 수만 같게 비교하면 session chunk 길이와 gradient accumulation에 따라 실제 optimizer update 수가 달라질 수 있다. 따라서 architecture 비교는 `fixed_optimizer_updates`를 1차 budget으로 잠그고 epoch는 보조 기록으로 남긴다. H0처럼 수백 update만 수행된 run을 충분히 수렴한 구조와 직접 비교하지 않는다.

권고 optimizer 시작점:

- AdamW
- learning rate `3e-4` for router 또는 `1e-3` for 기존 direct baseline을 중심으로 비교
- weight decay `1e-4`
- cosine decay 또는 validation-stable scheduler
- gradient clip `2.0`
- max 120 epochs, minimum 20, patience 18
- seed `20260828`, `20260829`, `20260830` 고정

### Phase E — stateful episode fine-tuning

- session chunk 32 windows
- warmup 2 windows
- identity-balanced session sampling
- reference-invalid window는 state update만 하고 supervised RR loss는 0
- gap/reset contract 적용
- radar mask augmentation 적용

### Phase F — validation lock

outer fold `f`에 대해:

- test: fold `f`
- validation: fold `(f+1) mod 6`
- train: 나머지 4 folds

corrected cache에서 valid row 수나 session eligibility가 바뀌더라도 기존 baseline과 공정하게 비교하려면 physical identity→fold mapping authority를 고정해 재사용한다. row 수 변화에 따라 `GroupKFold`를 다시 실행해 identity 배치를 바꾸면 전처리 효과와 split 난이도 효과가 섞인다.

validation에서만 다음을 잠근다.

- checkpoint
- direct vs candidate decision rule
- uncertainty threshold
- ensemble weight
- temperature/scale calibration
- safe fallback condition

그 뒤 test identity는 정확히 한 번 평가한다. test 결과를 본 뒤 threshold를 다시 조정한 결과는 새 discovery 결과이며 confirmatory 결과가 아니다.

### Phase G — deployment qualification

정확도와 별도로 다음을 검사한다.

- offline batch vs chronological streaming parity
- 7개 radar mask
- packet loss, flatline, partial corruption, time jitter
- CPU/CUDA latency와 peak memory
- parameter count와 spike rate
- no-NaN/no-Inf
- state reset과 long-run drift
- checkpoint/config/source/cache SHA closure
- cache manifest, 실제 feature file, frozen identity mapping, model source의 exact SHA binding
- target device에서의 energy는 실제 계측

현재 CUDA/PyTorch에서 계산한 spike rate와 spike penalty는 energy proxy일 뿐이다. 낮은 firing rate가 곧바로 낮은 joule이나 상용 전력 효율을 뜻하지 않는다. event-driven neuromorphic target으로 실제 mapping한 뒤 latency, memory traffic, operation count, wall power/energy를 계측하기 전에는 에너지 우위를 주장하지 않는다.

---

## 9. sampling과 augmentation

### 9.1 sampling

window를 균등 sampling하면 session이 길거나 valid window가 많은 identity가 학습을 지배한다. 권고 순서는 다음과 같다.

1. identity 총 weight를 동일하게 정규화
2. identity 내부 RR band를 완만하게 reweight
3. 25–35 bpm tail은 oversampling하되 동일 window 복제에 의존하지 않음
4. session chunk 학습에서는 rare episode를 포함하되 chronological order 보존

Group DRO는 worst-group robustness에 유용하지만 작은 group의 noise까지 과대적합할 수 있다. [ICLR 2020 Group DRO 연구](https://openreview.net/pdf?id=ryxGuJrFvS)가 보여주듯 regularization과 model selection이 중요하다. 이 프로젝트에서는 hard max group loss보다 identity-balanced sampling + 완만한 CVaR/group reweight를 먼저 쓴다.

### 9.2 허용할 augmentation

- whole-radar dropout
- radar별 gain scaling
- packet loss burst와 flatline mask
- 작은 range-bin shift
- additive noise와 impulsive corruption
- phase sign/inversion ablation, 단 I/Q contract가 확정된 경우만
- measured sync uncertainty 안의 작은 causal time jitter
- physically consistent spectral crop
- range-track confidence corruption

### 9.3 금지하거나 매우 조심할 augmentation

- 서로 다른 사람/session의 window mix
- target을 그대로 둔 큰 frequency warp
- centered time shift로 미래를 보는 처리
- reference RR에 맞춰 radar peak를 이동
- target을 보고 선택한 SVD component injection
- session action label을 모든 window에 복제
- 불확실한 I/Q layout을 임의로 섞어 format invariance라고 주장

frequency scaling을 쓴다면 waveform의 실제 시간축과 target RR을 함께 물리적으로 일관되게 바꿔야 한다. 단순 spectrum 이동은 label inconsistency를 만든다.

---

## 10. 평가 방법론

### 10.1 primary population

- physical-identity-disjoint 6-fold OOF
- reference-valid 전체 row, 현재 2,327개
- full coverage를 primary로 사용
- 쉬운 subset이나 uncertainty-retained subset을 primary로 바꾸지 않음

### 10.2 overlapping window 문제

32초 window를 4초 stride로 만들면 인접 window가 28초를 공유한다. 2,327개 row가 모두 독립 표본은 아니다. 따라서 다음을 함께 보고한다.

- 모든 valid window
- greedy non-overlap 32초 subset
- 8개 fixed stride phase
- identity-cluster bootstrap CI
- identity-macro metric
- session/episode level error

### 10.3 필수 strata

- RR: 6–10, 10–15, 15–20, 20–25, 25–35, 35–45 bpm
- 각 physical identity
- protocol/stage, 단 `eligible_for_stage_metrics=true`만
- radar mask 7개
- low range-track confidence/multimodal
- motion/transition
- reference clipping/quality는 평가 해석용

### 10.4 내부 정확도 gate

세 고정 seed 모두에서 아래 여섯 조건을 동시에 만족해야 내부 engineering pass다.

| 지표 | gate |
|---|---:|
| Overall MAE | ≤1.00 bpm |
| Identity-macro MAE | ≤1.00 bpm |
| RMSE | ≤1.80 bpm |
| ±2 bpm | ≥90% |
| >5 bpm | ≤3% |
| 25–35 bpm MAE | ≤2.00 bpm |

이 gate를 통과해도 외부 prospective cohort가 없으면 상용 성능 달성으로 선언하지 않는다.

### 10.5 uncertainty와 selective output

risk–coverage curve는 유용하지만, test OOF score의 사후 quantile로 threshold를 정하면 배포 coverage가 아니다. threshold는 calibration identity에서 잠가야 한다. 다음을 함께 보고한다.

- coverage
- retained MAE/RMSE
- high-RR retention
- identity별 rejection rate
- radar-mask별 rejection rate
- no-estimate false-safe / false-unsafe 비율

현재 70% retention에서 좋은 MAE가 나와도 reference-valid 자체가 전체 후보의 24.30%이고 high-RR retention이 낮으므로 full commercial claim을 대신하지 않는다.

---

## 11. 추천 ablation matrix

한 번에 대형 architecture search를 하지 않고, 각 ablation이 한 질문만 답하게 구성한다.

| ID | 비교 | 답하려는 질문 | 승격 기준 |
|---|---|---|---|
| A0 | 현재 structured 12-step SNN | 고정 baseline | 재현 hash/metric 일치 |
| A1 | scalar regression vs distributional RR | soft posterior가 tail/전체를 개선하는가 | overall·macro 비열등 + RMSE 개선 |
| A2 | fixed LIF vs PLIF | 학습 time constant가 필요한가 | 3 seed 안정 개선 |
| A3 | PLIF vs PLIF+ALIF episode | adaptation이 지속 alias를 줄이는가 | high-RR/catastrophic 개선, normal 비열등 |
| A4 | last-step loss vs TET | 낮은 T 성능이 좋아지는가 | T=8에서 T=12 baseline 비열등 |
| A5 | no KD vs logit KD vs logit+feature KD | teacher가 일반화를 돕는가 | test를 보지 않은 validation 승격 |
| A6 | additive coordinate vs joint interaction | coordinate swap invariance가 해결되는가 | unit test + validation routing regret 감소 |
| A7 | undirected vs directed graph | harmonic 방향성이 유효한가 | factor confusion/catastrophic 개선 |
| A8 | direct-only vs router-only vs dual-path | fallback이 안전성을 높이는가 | 전체·tail gate 동시 개선 |
| A9 | no SSL vs masked vs multi-view SSL | unlabeled radar가 도움이 되는가 | identity-disjoint 3 seed 개선 |
| A10 | stateless vs stateful episode | 과거 지속성이 도움이 되는가 | streaming parity + episode error 개선 |
| A11 | raw only vs raw+SVD vs raw+SVD+range | 보조 신호의 실제 기여 | mask별 안정성과 macro 개선 |
| A12 | Conv/graph SNN vs small Spikformer | attention이 sample 효율을 이기는가 | parameter/latency matched 비교 |
| A13 | direct-trained vs ANN→SNN conversion | 직접학습의 실질 이득 | 같은 latency/energy 조건 비교 |

각 ablation은 discovery outer fold에서 좁히고, 같은 cohort outer 결과를 반복해서 leaderboard로 사용하지 않는다. 구조 결정 횟수와 관찰한 fold를 ledger에 남긴다.

---

## 12. 실패 모드별 대응

### 12.1 high-RR를 1/3 또는 1/4로 예측

원인 후보:

- dominant subharmonic
- coordinate encoder가 ratio 위치를 소실
- high-RR support 부족
- motion recovery에서 fundamental이 약함

대응:

- directed factor graph
- confident factor supervision
- wrong-harmonic margin
- high-RR identity 추가 수집
- direct path fallback과 harmful-correction gate

### 12.2 radar 하나가 망가지면 큰 성능 저하

원인 후보:

- learned fusion이 full 3-view에 과의존
- dropout이 whole-radar corruption을 충분히 모사하지 못함

대응:

- coupled whole-radar dropout
- 7 mask training/evaluation
- partial corruption augmentation
- reliability head와 mask-aware normalization

### 12.3 motion 구간에서 불확실도는 높지만 RR도 잘못 출력

대응:

- numeric output과 availability를 분리
- calibrated no-estimate 정책
- range-track multimodal/missing을 target-free quality로 사용
- risk–coverage뿐 아니라 false-safe/false-unsafe 평가

### 12.4 validation은 좋아지고 unseen identity는 악화

원인:

- 18명 cohort에 대한 architecture overfit
- identity-specific correction prior
- overlapping window가 만든 유효 표본수 착시

대응:

- identity-balanced sampling
- parameter cap
- three fixed seeds
- paired identity-cluster bootstrap
- prospective identity cohort

### 12.5 corrected cache에서 성능이 갑자기 달라짐

가능한 원인:

- nominal time에서 measured time으로 변경
- sync authorization subset 변화
- plateau repair
- stage core window eligibility
- range auxiliary 분리

이 경우 기존 cache와 새 cache를 조용히 섞으면 안 된다. 새로운 dataset version으로 보고, cache context와 모든 hash를 변경한 뒤 baseline부터 재학습한다.

---

## 13. 지금 당장 학습할 수 있는 것과 없는 것

### 13.1 가능한 것

- 기존 canonical cache에서 현재 leader 재현
- coordinate-interaction encoder의 synthetic/unit test
- outer-train identity만 쓰는 SSL 코드와 smoke run
- 기존 retrospective cache에서 ablation 설계 검증
- latency, spike rate, streaming parity 측정
- acquisition contract fail-closed loader 검증

### 13.2 아직 과학적으로 열면 안 되는 것

- 자동 marker candidate를 승인된 sync처럼 취급한 corrected-cache 학습
- 현재 권위 artifact에서 scientific-eligible 0개인 상태로 새 cache 성능 claim
- I/Q split 가설을 확정된 sensor layout처럼 사용
- range bin을 calibration 없이 meter로 변환
- protocol action을 모든 window의 사람 행동 label로 사용
- 기존 18명 outer 결과에 맞춘 반복 threshold로 confirmatory claim

### 13.3 데이터 측면에서 가장 큰 성능 자원

현재 구조 변경보다 가치가 큰 데이터는 다음이다.

- 새로운 physical identity
- 의도적으로 충분히 긴 25–35 bpm episode
- 각 episode의 직접적인 factor/divisor supervision
- hardware-synchronized radar–reference trigger
- 독립 reference 또는 adjudication
- radar placement, 자세, motion, dropout 조건의 계획된 coverage

현재 oracle과 실제 router의 차이를 보면, 단순히 같은 18명에서 network 폭을 늘리는 것보다 새로운 high-RR identity가 올바른 correction rule을 배우는 데 더 중요하다.

---

## 14. 구현 파일과 설계의 연결

| 역할 | 현재 파일 | 상태 |
|---|---|---|
| 기본 feature/config | `configs/default.yaml` | 기존 baseline 권위 |
| ANN teacher / TriRadar SNN | `src/snn_rr/models.py` | 구현됨 |
| 기본 학습 | `scripts/train.py` | 구현됨 |
| harmonic candidate set | `src/snn_rr/harmonic_set_models.py` | 구현됨 |
| harmonic set 학습 | `scripts/train_harmonic_set_snn.py` | 구현됨 |
| v3 factor-router config | `configs/harmonic_factor_router_v3.yaml` | retrospective 설계 계약 |
| DHFER ancestry | `src/snn_rr/harmonic_factor_router_v3.py` | read-only/격리된 ancestry |
| v3r1 runtime wrapper | `src/snn_rr/harmonic_factor_router_models_v3r1.py` | hash-bound wrapper 구현됨 |
| SVD model | `src/snn_rr/svd_models.py` | 구현됨 |
| temporal SVD | `src/snn_rr/svd_temporal_models.py` | 구현됨 |
| episode SVD | `src/snn_rr/svd_episode_models.py` | 구현됨 |
| radar measured timing | `src/snn_rr/radar_timing.py` | 구현됨 |
| marker sync | `src/snn_rr/synchronization.py` | 구현됨; 최신 3-session smoke 승인 0/3, full strict 미완료 |
| protocol reconstruction | `src/snn_rr/acquisition_protocol.py` | 구현됨 |
| causal range tracking | `src/snn_rr/range_tracking.py` | extractor 구현됨; leader model 입력에는 미연결 |
| acquisition reconstruction | `scripts/reconstruct_acquisition.py` | 구현됨 |
| strict acquisition consumer | `src/snn_rr/acquisition_contract.py` | 구현됨 |
| cache strict loading | `src/snn_rr/cache.py` | opt-in fail-closed 검증 구현됨; 기본 `train.py`에는 미강제 |
| SVD provenance binding | `scripts/build_svd_features.py` | acquisition binding 구현됨 |

아직 구현된 winner로 간주하면 안 되는 항목:

- 이 문서의 완성형 `CCHG-SNN` 통합 모델
- TET loss의 본 학습 경로 통합과 full OOF 결과
- SSL pretraining의 full nested fold 결과
- coordinate-interaction redesign의 상용 gate 통과 결과
- prospective cohort 결과

즉, 문서는 현 구조에 근거한 최종 권고 설계이며 일부 구성요소는 이미 구현됐지만 전체 조합의 성능은 아직 측정되지 않았다.

### 14.1 코드에서 바로 찾아볼 exact symbol

```text
src/snn_rr/models.py
  ::_SharedRadarSpatialEncoder
  ::_StructuredAuxiliaryFusion
  ::_SpikingResidualFrequencyBlock
  ::SharedRadarCNNTeacher
  ::TriRadarRRSNN

scripts/train.py
  ::apply_coupled_radar_dropout
  ::identity_balanced_sample_weights
  ::make_fold_assignments
  ::compute_multitask_loss
  ::train_stage

src/snn_rr/preprocess.py
  ::range_frequency_features
  ::filter_reference_rsp
  ::estimate_reference_window

scripts/build_features.py
  ::replace_radar_outliers

src/snn_rr/harmonic_set_data.py
  ::build_candidate_bank

src/snn_rr/harmonic_set_models.py
  ::HarmonicCandidateSetEpisodeSNN

scripts/train_harmonic_set_snn.py
  ::compute_multitask_loss

src/snn_rr/harmonic_factor_router_v3.py
  ::StructuredHarmonicEvidenceEncoder
  ::_DirectedGraphPLIFBlock
  ::_CausalFactorPLIFALIF
  ::DirectedHarmonicFactorExpertSNN

src/snn_rr/harmonic_factor_router_models_v3r1.py
  ::DirectedHarmonicFactorExpertSNNV3R1

scripts/train_harmonic_factor_router_snn_v3r1.py
  ::compute_multitask_loss

src/snn_rr/svd_models.py
  ::SourceSeparatedRRSNN

scripts/train_svd_snn.py
  ::compute_svd_multitask_loss

src/snn_rr/svd_temporal_models.py
  ::TemporalSourceSeparatedRRSNN

src/snn_rr/svd_episode_models.py
  ::EpisodeAliasRRSNN
```

### 14.2 테스트 상태의 해석

acquisition/sync/protocol/range/cache 집중 검증은 최근 `32 passed`, cache/SVD provenance 집중 검증은 `8 passed`였다. 다만 기존 full-suite snapshot은 `688 passed, 4 skipped, 2 order-sensitive float-equivalence failures`였으므로 repository 전체가 clean pass라고 주장할 수 없다. 이번 문서 작성에서는 full suite를 새로 실행하지 않았다. 두 failure는 과학 결과와 별개로 고정하거나 원인을 격리한 뒤 release evidence를 발급해야 한다.

---

## 15. 최종 선택 우선순위

### 1순위 — 직접 surrogate-gradient hybrid SNN

PLIF/ALIF, SEW residual, distributional RR, TET를 사용한다. 현재 데이터 형식과 지연 목표에 가장 잘 맞는다.

### 2순위 — ANN teacher distillation

작은 identity 수에서 optimization을 안정화하되, high-RR에서 teacher가 틀리면 KD gate/decay로 오류 전이를 제한한다.

### 3순위 — coordinate-aware harmonic candidate routing

현재 가장 큰 정확도 병목을 직접 겨냥한다. evidence와 radar/ratio/branch 좌표가 pooling 전에 상호작용해야 한다.

### 4순위 — split-safe self-supervised pretraining

reference-invalid radar window를 label-free로 활용해 표현력을 높인다. outer-test identity를 pretraining에서 제외한다.

### 5순위 — stateful episode 학습과 uncertainty

streaming에서 지속 harmonic episode를 다루고 위험한 출력을 거부한다. invalid window는 state만 update하고 loss는 mask한다.

### 비교 기준으로만 유지

- ANN→SNN conversion
- STDP-only 학습
- 대형 Spiking Transformer
- target-dependent component/router oracle

---

## 16. 최종 판정

이 프로젝트에 가장 적합한 답은 “더 큰 SNN 하나”가 아니다. 다음을 함께 만족하는 작은 hybrid 시스템이다.

1. raw parser와 measured-time acquisition contract가 먼저 데이터를 잠근다.
2. analog front-end가 dense radar의 공간·스펙트럼 구조를 손실 없이 압축한다.
3. coordinate-aware candidate graph가 harmonic factor의 위치 의미를 보존한다.
4. PLIF/ALIF residual SNN이 8–12 step으로 증거와 causal state를 누적한다.
5. surrogate gradient, TET, distillation, distributional/multitask loss로 직접 학습한다.
6. identity-disjoint nested validation에서만 모든 선택을 잠근다.
7. hard candidate가 불안하면 safe anchor 또는 no-estimate로 돌아간다.
8. corrected-data sync 승인과 prospective cohort 없이는 상용 성능이라고 부르지 않는다.

현재 수치상 가장 시급한 목표는 overall MAE 1.291→≤1.0보다도 25–35 bpm MAE 4.216→≤2.0과 >5 bpm error 6.23%→≤3%를 안전하게 낮추는 것이다. oracle 결과는 이론적인 후보 상한이 있음을 보여주지만, 그 격차를 닫으려면 좌표 보존 router와 새로운 high-RR identity supervision이 함께 필요하다.

---

## 17. 주요 참고문헌

- Neftci, Mostafa, Zenke, “Surrogate Gradient Learning in Spiking Neural Networks,” IEEE Signal Processing Magazine, 2019. [arXiv](https://arxiv.org/abs/1901.09948)
- Shrestha, Orchard, “SLAYER: Spike Layer Error Reassignment in Time,” NeurIPS 2018. [공식 논문 페이지](https://papers.neurips.cc/paper_files/paper/2018/hash/82f2b308c3b01637c607ce05f52a2fed-Abstract.html)
- Fang et al., “Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks,” ICCV 2021. [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.html)
- Fang et al., “Deep Residual Learning in Spiking Neural Networks,” NeurIPS 2021. [공식 논문 페이지](https://proceedings.neurips.cc/paper/2021/hash/afe434653a898da20044041262b3ac74-Abstract.html)
- Deng et al., “Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting,” ICLR 2022. [OpenReview PDF](https://openreview.net/pdf?id=_XNtisL32jv)
- Zhou et al., “Spikformer: When Spiking Neural Network Meets Transformer,” ICLR 2023. [OpenReview PDF](https://openreview.net/pdf?id=frE4fUwz_h)
- Xu et al., “Constructing Deep Spiking Neural Networks from Artificial Neural Networks with Knowledge Distillation,” CVPR 2023. [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Constructing_Deep_Spiking_Neural_Networks_From_Artificial_Neural_Networks_With_CVPR_2023_paper.html)
- Yue et al., “TS2Vec: Towards Universal Representation of Time Series,” AAAI 2022. [AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/20881/20640)
- Sagawa et al., “Distributionally Robust Neural Networks for Group Shifts,” ICLR 2020. [OpenReview PDF](https://openreview.net/pdf?id=ryxGuJrFvS)
- Ooijen et al., uncertainty-aware spiking regression, ICONS 2025. [CWI PDF](https://ir.cwi.nl/pub/36110/36110.pdf)
- Bellec et al., “Long Short-Term Memory and Learning-to-Learn in Networks of Spiking Neurons,” NeurIPS 2018. [공식 논문](https://papers.nips.cc/paper/7359-long-short-term-memory-and-learning-to-learn-in-networks-of-spiking-neurons)
- Bellec et al., “A solution to the learning dilemma for recurrent networks of spiking neurons,” Nature Communications, 2020. [DOI](https://doi.org/10.1038/s41467-020-17236-y)
- Kaiser et al., “Synaptic Plasticity Dynamics for Deep Continuous Local Learning,” Frontiers in Neuroscience, 2020. [원 논문](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00424/full)
- Rueckauer et al., “Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification,” Frontiers in Neuroscience, 2017. [원 논문](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00682/full)

## 18. 프로젝트 내부 근거 문서

- `README.md`: 현재 모델·수치·재현 명령의 요약
- `REPORT.md`: full OOF 결과와 실패 분석
- `artifacts/SNN_PROJECT_TECHNICAL_STATUS_REPORT_2026-08-30.md`: 데이터부터 실행 인프라까지의 통합 현황
- `artifacts/COMMERCIAL_SNN_GOAL_V2.md`: 정확도·안전·prospective gate
- `artifacts/COMMERCIAL_SNN_MASTER_EXECUTION_PLAN_V3.md`: 데이터부터 release까지의 실행 계약
- `artifacts/COMMERCIAL_SNN_PROGRESS_V2.md`: 시도한 후보와 승격·기각 ledger
- `artifacts/acquisition/reconstruction_v1_smoke2/manifest.json`: 최신 source-consistent 3-session smoke 상태
- `artifacts/acquisition/reconstruction_v1_sync_diagnostic/manifest.json`: 과거 29-session 진단 상태; 최신 source의 완결 증거는 아님
