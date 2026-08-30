# 상용 SNN 단일 종단 실행 계약 v3

문서 상태: **동결 실행 계약**  
기준일: **2026-08-28 (Asia/Seoul)**  
프로그램 상태: **`RESEARCH_CANDIDATE / RETROSPECTIVE_STAGE_A_KILLED`**  
상용 성능 주장: **금지**  
현재 정확도 게이트: **0/6**  
현재 실행 권한: **동일 18-identity cohort에서 Stage B 신경망 학습·outer-test 개봉·추가 threshold 탐색 금지**

이 문서는 아이디어 목록이나 “다음 단계” 제안서가 아니다. 데이터 동결부터 모델 설계, 완전 nested 검증, prospective 확증, 장치 검증, shadow/canary와 rollback까지 하나의 프로그램으로 끝내기 위한 실행·중단 계약이다. 어느 단계든 사전 게이트를 실패하면 그 지점에서 원자적으로 종료하며, 실패한 동일 cohort를 다시 보면서 architecture, threshold, seed 또는 subset을 바꾸지 않는다.

---

## 0. 집행 요약

### 0.1 최종 목적

3대 UWB radar의 32초 causal window와 4초 stride로 호흡수(RR)를 추정하는 **단 하나의 Harmonic Candidate-Set Episode SNN(HCS-E-SNN)** 을 개발한다. 목표 상태는 단순한 내부 OOF 개선이 아니라 다음 조건을 모두 충족한 `DEPLOYMENT_CANDIDATE`다.

1. identity-disjoint full-coverage 정확도 6개 게이트를 서로 다른 3개 seed가 각각 통과한다.
2. base, candidate proposer, router, threshold, calibration이 모두 nested 또는 사전 동결된 독립 cohort로 검증된다.
3. radar 결측, non-overlap, temporal phase, 입력 교란, calibration, streaming parity, latency와 fault campaign을 통과한다.
4. 개발에 쓰지 않은 prospective confirmation cohort에서 point estimate와 identity-cluster confidence bound를 동시에 통과한다.
5. target device의 shadow/canary, 관측성, 자동 rollback과 감사 가능한 release bundle이 준비된다.

### 0.2 현재 판정

2026-08-28에 사전 등록한 Stage A separability gate를 outer-3 제외 pool과 outer-4 제외 pool에서 각각 grouped inner OOF로 실행했다. 순위 분리력은 높았지만 실제로 안전한 correction policy를 만들지 못했고, 두 partition 모두 동시 게이트를 실패했다. 따라서 `harmonic_factor_snn_v1`은 **사전 규칙에 따라 killed**다.

- Stage B HCS-E-SNN 학습: **열지 않음**
- discovery outer-3/outer-4 test prediction: **열지 않음**
- full 6-fold OOF: **열지 않음**
- 동일 18명에서 threshold/feature/seed 재탐색: **금지**
- frozen leader: **`ensemble_structured_exact` 유지**

이 종료는 계산 부족이 아니라 gate 판정이다. 프로그램을 다시 `A_SCREENING`으로 전이시키는 유일한 입력은 기존 18명과 겹치지 않고 사전 등록된 prospective calibration/development cohort다. 그 cohort에서도 이 문서에 동결한 **같은 단일 architecture family와 같은 단계 순서**를 적용한다.

### 0.3 oracle의 해석 제한

Oracle은 각 window의 정답을 본 뒤 가장 가까운 후보를 고르는 불가능한 선택기다. 다음 수치는 후보 bank 안에 잠재 정보가 있다는 진단일 뿐 모델 성능, validation 결과, 예상 상용 성능 또는 승격 근거가 아니다.

- frozen base + classical `×1..×4` 5-expert oracle: MAE 0.689 bpm, identity-macro MAE 0.674 bpm, RMSE 1.129 bpm, ±2 bpm 90.67%, >5 bpm 0.21%, 25–35 bpm MAE 1.224 bpm.
- base + alias-posterior top modes + classical harmonics를 포함한 더 넓은 candidate-set oracle 진단: MAE 약 0.460 bpm, identity-macro MAE 약 0.442 bpm, RMSE 약 0.921 bpm, ±2 bpm 약 93.30%, >5 bpm 약 0.17%, 25–35 bpm MAE 약 0.979 bpm.

두 oracle 모두 target-leaking upper-bound diagnostic이며 **통과 게이트 수는 0개로 기록**한다. 실제 router Stage A가 실패했으므로 oracle과 실제 deployable selector 사이의 간극이 현재 병목이다.

---

## 1. 범위, 기준점과 불변 조건

### 1.1 제품 입력·출력 계약

| 항목 | 동결 값 |
|---|---|
| Intended input | 동기화된 최대 3대 UWB radar의 causal raw/cache 신호와 radar availability mask |
| Window | 과거 32초만 사용 |
| Stride | 4초 |
| RR support | 6.00–45.00 bpm, 내부 posterior grid 0.25 bpm |
| 정상 출력 | RR point estimate, calibrated interval, quality/availability 상태, provenance ID |
| 금지 출력 해석 | prospective confirmation 전 의료 진단·상용 성능·임상 효용 주장 |
| Causal rule | 미래 window, 전체 session 종결 정보, future QC를 절대 사용하지 않음 |
| Full-coverage rule | valid-reference window를 사후 삭제하지 않음; inference failure도 분모에 남김 |
| Fallback | base가 있으면 bit-for-bit frozen base, 없으면 `UNAVAILABLE`; 임의 보간 금지 |

### 1.2 동결 retrospective population

- 물리적 identity: 18명
- usable session: 29개
- identity-disjoint outer folds: 6개, 각 fold 3 identities
- reference-valid window: 2,327개
- all-window inference population: 9,576개
- validation rule: `validation_fold = (outer_fold + 1) mod 6`
- split authority: `artifacts/runs/final_alias_gate_s12_deterministic/fold_assignments.json`
- split SHA-256: `320ff08e00fd66a609c676b16ea2314ea23636986a9fb35388d59fae0d3794a1`
- frozen OOF: `artifacts/runs/ensemble_structured_exact/ensemble_oof.csv`
- frozen OOF SHA-256: `31e3d9f9b41675329af73003a69c6e30d881ae496290a1580287daaaae0af602`

Identity-to-fold mapping은 `CAMPAIGN_CONTRACT.json`의 값을 단일 authority로 사용하며 실행 중 재균형하지 않는다. 동일 identity, 동일 session 또는 겹치는 raw interval은 train/validation/test 경계를 넘지 못한다.

### 1.3 frozen accuracy leader

| 지표 | frozen leader | 최종 내부 게이트 | 현재 판정 |
|---|---:|---:|:---:|
| Overall MAE | 1.290886 bpm | ≤1.000 bpm | FAIL |
| Identity-macro MAE | 1.220042 bpm | ≤1.000 bpm | FAIL |
| RMSE | 2.410190 bpm | ≤1.800 bpm | FAIL |
| `abs(error) ≤ 2` | 80.7907% | ≥90.0000% | FAIL |
| `abs(error) > 5` | 6.2312% | ≤3.0000% | FAIL |
| 25–35 bpm MAE | 4.216330 bpm | ≤2.000 bpm | FAIL |

25–35 bpm은 `n=164`, identity-macro MAE 3.667 bpm, RMSE 5.984 bpm, bias −4.055 bpm, ±2 bpm 48.78%, >5 bpm 42.68%다. non-overlap high-RR window는 42개뿐이므로 이 cohort에서 반복 튜닝한 tail score는 확증 증거가 아니다.

### 1.4 불변 gate 산식

정답 `y_i`, 예측 `p_i`, identity 집합 `G`, high-RR index `H={i | 25≤y_i≤35}`에 대해 다음 산식을 변경하지 않는다.

- `overall_MAE = mean_i |p_i-y_i|`
- `identity_macro_MAE = mean_g mean_{i∈g}|p_i-y_i|`
- `RMSE = sqrt(mean_i(p_i-y_i)^2)`
- `within_2 = mean_i[|p_i-y_i|≤2]`
- `over_5 = mean_i[|p_i-y_i|>5]`
- `high_RR_MAE = mean_{i∈H}|p_i-y_i|`

경계는 위 식 그대로 inclusive/exclusive를 유지한다. prediction clipping, invalid row 제거, protocol subset 교체, 가장 좋은 seed 선택, 가장 좋은 phase 선택, fold별 모델 수동 교체는 허용하지 않는다. 모든 valid-reference 행에 finite prediction이 없으면 full-coverage gate는 자동 실패다.

### 1.5 최종 통과 규칙

내부 통과는 seeds `20260828`, `20260829`, `20260830` **각각**이 6개 gate를 모두 만족할 때만 인정한다. seed 평균이나 3-seed ensemble이 통과해도 한 seed가 실패하면 탈락이다. 추가로 다음 안전 규칙을 만족해야 한다.

- frozen leader 대비 paired identity-cluster bootstrap MAE delta의 양측 95% CI 상한 ≤ `+0.05 bpm`.
- identity-macro MAE delta의 양측 95% CI 상한 ≤ `+0.05 bpm`.
- 어느 identity도 frozen leader보다 MAE가 `+1.00 bpm` 이상 악화되지 않음.
- 어느 fold도 `over_5`가 `+1.0%p` 이상 증가하지 않음.
- 12–25 bpm identity-macro MAE 악화 ≤ `+0.10 bpm`.
- 25–35 bpm point gate 통과와 함께 high-RR support가 있는 identity 중 90% 이상에서 paired MAE가 악화되지 않음.

Internal bootstrap은 identity를 cluster 단위로 100,000회 복원추출하고 window는 cluster와 함께 이동한다. seed는 `20260828`; percentile 2.5/97.5%를 사용한다. prospective confirmation의 통계 규칙은 12절에 별도로 더 엄격하게 정의한다.

---

## 2. 데이터·평가 threat model

| 위협 | 실패 방식 | 예방 통제 | 자동 fail 조건 |
|---|---|---|---|
| Identity leakage | 같은 사람이 train/test 양쪽에 존재 | physical identity authority, split hash, exact-cover audit | 한 identity라도 경계 중복 |
| Interval leakage | 중첩 32초 raw interval이 split을 넘음 | session/clock interval join 감사 | 1개 interval overlap |
| Stacked-base leakage | candidate/router train feature가 held identity로 학습된 base에서 생성 | 6절의 inner OOF stack | in-sample proposer feature 1행 |
| Target/QC leakage | `rr`, reference validity/quality, adjudication outcome가 forward input에 섞임 | field allowlist, adversarial schema tests | 금지 field 접근 또는 상관 proxy 확인 |
| Future leakage | 이후 window 또는 session 종결 통계 사용 | monotonic timestamp, streaming replay | future timestamp read 1건 |
| Outer adaptation | outer result를 본 뒤 threshold/architecture 변경 | one-shot marker, lock-before-open | lock 이후 config/hash 변경 |
| Multiplicity | 많은 seed/config 중 좋은 것만 보고 | architecture 1개, seed 3개, ledger | 미등록 run 또는 결과 누락 |
| Denominator drift | 실패 row 또는 어려운 protocol 제외 | exact valid index hash, reason-count ledger | expected row 불일치 |
| Reference bias | clock drift, clipping, motion artifact가 label로 들어감 | hardware sync, blind dual-reference QC | sync/QC 허용치 초과 |
| Preprocessing skew | offline cache와 streaming feature 불일치 | golden raw replay, state parity | 허용오차 초과 1개 golden case |
| Missing radar proxy | mask 패턴이 protocol/identity proxy가 됨 | randomized dropout, mask-stratum audit | identity predictability 또는 mask harm gate 실패 |
| Episode reset error | window boundary마다 상태 초기화 | session state serialization, restart tests | replay mismatch |
| Artifact tamper | checkpoint/config/source가 실행 후 바뀜 | SHA-256 DAG, read-only manifest, signature | hash mismatch |
| Partial commit | interrupted run을 complete로 오인 | temporary directory + `COMMITTED.json` last | marker/source/output 불일치 |
| Numeric fault | NaN/Inf/AMP divergence | finite guards, CPU/CUDA parity | nonfinite 1건 또는 RR delta tolerance 초과 |
| Replay/order fault | 중복·역순 packet이 hidden state를 오염 | idempotency key, monotonic sequence ID | state mutation on rejected packet |
| Distribution shift | 자세, 체형, site, placement, motion 변화 | prospective strata, drift monitoring | predeclared subgroup CI 실패 |
| Security/supply chain | dependency/model 교체, unsigned input | pinned lockfile/SBOM/signature | untrusted bundle 또는 signature failure |
| Privacy | identity/raw physiological data 과다 보존 | pseudonym, least privilege, retention policy | mapping exposure 또는 unauthorized access |

Forward-input allowlist는 raw/cache radar evidence, radar mask, causally computed classical RR/confidence, causal hidden state, session-start elapsed time뿐이다. identity, protocol label, reference validity, reference quality, reference RR, target-derived action label, frozen-base error, future context는 금지한다. `frozen_base_prediction`은 learned tensor가 아니라 최종 deterministic fallback/mix 연산에서만 접근한다.

---

## 3. 단일 모델 설계 동결: HCS-E-SNN

### 3.1 설계 가설

현재 tail failure는 고호흡에서 강한 subharmonic을 direct peak로 해석하고, unseen identity에서 `×2/×3/×4` correction을 안전하게 선택하지 못해 발생한다. 단일 개입은 **range-preserving harmonic candidate graph + causal episode state**다. architecture family를 추가하거나 tree/transformer/CNN router를 경쟁시키지 않는다.

### 3.2 모듈 경계

HCS-E-SNN은 다음 네 부분으로 고정한다.

1. **Nested direct-source proposer**: outer-train identity로만 학습한 기존 structured direct-posterior SNN configuration이 6.00–45.00 bpm, 0.25-bpm grid posterior를 생성한다.
2. **Deterministic candidate builder**: direct posterior mode와 classical harmonic을 최대 12개 candidate node로 합친다.
3. **Harmonic graph episode SNN**: candidate별 multi-resolution/radar evidence를 PLIF graph와 ALIF chronological state로 인코딩해 candidate posterior, residual, scale와 quality를 낸다.
4. **Validation-locked safety policy**: learned score가 사전 threshold를 만족할 때만 source candidate 쪽으로 sparse correction하고, 나머지는 frozen base를 bit-for-bit 반환한다.

Frozen leader는 proposer/router feature, loss target 또는 hidden state에 들어가지 않는다. validation의 승격 comparator와 최종 deterministic fallback/mix에만 사용한다.

#### Nested direct-source proposer 고정값

Proposer는 `final_structured_exact_s12_deterministic`의 configuration family를 nested fit하도록 고정한다. authority config는 `artifacts/runs/final_structured_exact_s12_deterministic/run_config.json`이고 SHA-256은 `7b07d7c631ea8d445ea272669c60986260446f9c2f8ebe393eee9c6223dc2ca9`다. 핵심 값은 map branch `both`, structured exact auxiliary, 2 input branches, hidden 192, 12 simulation steps, RR 6–45 bpm/0.25 bpm bins, batch 48, AdamW LR `1e-3`, weight decay `1e-4`, 최대 80 epochs, patience 12, min delta `0.001`, radar dropout 0.20, distillation weight/temperature `0.35/2.0`, quality weight 0.15, spike-rate weight `5e-4`, causal history on이다. Program seed만 해당 outer/inner run seed로 치환한다.

Distillation teacher도 같은 inner-fit identities에서 학습한 nested teacher여야 한다. 기존 global `final_structured_aux_s12` checkpoint를 inner held, validation 또는 test proposer에 재사용하지 않는다. Teacher, proposer, scaler와 causal-history initialization은 하나의 nested provenance unit이며 하나라도 held identity를 학습에 보았으면 그 outer run 전체를 폐기한다.

### 3.3 deterministic candidate bank

각 32초 window에서 다음 순서로 candidate를 만든다.

1. Direct-source posterior에서 probability 내림차순으로 local maximum을 찾는다. 이미 선택한 mode와 0.75 bpm 이내는 suppress하고 최대 5개를 취한다.
2. causally computed fused classical seed `f0`에 `×1, ×2, ×3, ×4`를 적용한 4개를 추가한다.
3. available radar별 classical direct peak를 confidence 내림차순으로 추가하여 총 12개를 채운다.
4. 6–45 bpm 밖 candidate는 버린다.
5. candidate 간 거리가 0.50 bpm 이하면 source priority `direct posterior > fused classical > per-radar classical`, 그 안에서는 confidence가 큰 것을 남기고 duplicate count를 누적한다.
6. 남는 slot은 zero padding하고 `candidate_mask=false`로 둔다. 정렬은 RR 오름차순으로 고정한다.

후보 node scalar metadata는 `[rr/45, direct_probability, classical_confidence, source_onehot(3), duplicate_count/6, boundary_distance/6]`의 8개 값이다. 후보 수·순서·merge는 label을 보지 않으며 code/hash로 고정한다.

### 3.4 label-free evidence tensor

Candidate `c`, radar `r`, causal duration `d∈{8,16,32초}`마다 SVD component 1–6과 raw spectral map으로 아래 16개 값을 계산한다.

1. `log1p(power(c))`
2. local `±1 bpm` median 대비 peak contrast
3. local bandwidth at half prominence
4. local spectral entropy
5. component 1–6 normalized agreement at `c`
6. energy ratio `c/2 : c`
7. energy ratio `2c : c`
8. energy ratio `c/3 : c`
9. energy ratio `3c : c`
10. energy ratio `c/4 : c`
11. energy ratio `4c : c`
12. available-radar magnitude coherence at `c`
13. available-radar phase coherence at `c`
14. candidate peak rank in that radar/duration
15. radar availability bit
16. valid temporal support fraction

Out-of-band harmonic samples는 value 0과 별도 support bit 0으로 표현하고 edge extrapolation을 하지 않는다. scaler는 outer-train identity에서만 median/IQR로 fit하며, IQR은 `max(IQR, 1e-4)`, scaled value는 `[-12,12]`로 clip한다. scaler fit에 invalid-reference all-window radar evidence를 포함할 수 있지만 target/QC는 사용하지 않는다.

각 candidate node의 입력 차원은 `3 radars × 3 durations × 16 + 8 metadata = 152`다. radar가 없으면 해당 16차원 block은 0이고 availability가 0이다. 모든 radar가 없으면 learned path를 실행하지 않고 fallback한다.

### 3.5 harmonic graph와 spiking episode encoder

고정 architecture는 다음과 같다.

| 구성 | 동결 값 |
|---|---|
| Max nodes | 12 |
| Node projection | `Linear(152,64) → SiLU → LayerNorm(64)` |
| Graph blocks | 2개 residual message-passing block |
| Graph state | PLIF 64 channels, learnable beta, init 0.92 |
| Surrogate | fast-sigmoid slope 25 |
| Edge types | proximity, ×2, ×3, ×4, same-source |
| Proximity edge | `abs(c_i-c_j)≤1.0 bpm` |
| Harmonic edge | `abs(c_j/c_i-k)≤0.05k`, `k∈{2,3,4}` |
| Edge features | normalized RR delta, ratio error, edge-type one-hot |
| Candidate pooling | mask-aware attention, 4 heads × 16 dims |
| Episode input projection | `Linear(128,64) → SiLU → LayerNorm(64)` |
| Episode cells | PLIF(64) → ALIF(64) |
| PLIF beta | learnable, init 0.92, constrained `(0.5,0.995)` |
| ALIF beta/decay/strength | 0.92 / 0.97 / 0.40 initial |
| Dropout | 0.05 |
| TBPTT | 32 chronological windows, 8-window causal warm-up, loss on final 24 |
| State reset | physical session start 또는 timestamp gap >12초만 |
| Parameter cap | 750,000 trainable parameters |
| Spike-rate operating band | layer mean 0.02–0.25 spikes/neuron/step |

Graph message는 target candidate와 연결된 neighbor의 projected state를 attention-weighted sum하고 자기 state와 concat한 뒤 `Linear(128,64)`로 PLIF current를 만든다. padded node는 message/state/loss를 모두 0으로 유지한다. inference에서는 session 전체에 state를 이어가고, 32-window training chunk 경계에서는 state를 carry하되 gradient만 detach한다.

### 3.6 출력 head와 deterministic decoder

동일 64-d episode token으로 다음 head를 만든다.

- `candidate_logit[K]`: listwise selection posterior.
- `residual[K]`: `0.75*tanh(z)` bpm.
- `log_scale[K]`: `0.25 + softplus(z)`, 상한 6 bpm.
- `factor_logit[4]`: fused classical `×1..×4` auxiliary class.
- `quality_logit[1]`: target-independent input adequacy score.

Source estimate는 `argmax(candidate_probability)` node의 `candidate_rr + residual`이며 6–45 bpm support 안에서만 유효하다. posterior expectation으로 서로 다른 harmonic mode를 평균하지 않는다.

최종 output policy는 validation에서 고정한 `(t_prob, t_margin, t_entropy, t_quality, pull)`을 사용한다.

```text
learned_ok = radar_available
          and finite(source_rr, scale, all logits)
          and max_probability >= t_prob
          and top1_probability - top2_probability >= t_margin
          and normalized_entropy <= t_entropy
          and sigmoid(quality) >= t_quality

if base_available and learned_ok:
    prediction = (1 - pull) * frozen_base + pull * source_rr
elif base_available:
    prediction = frozen_base                  # bit-for-bit exact
elif learned_ok:
    prediction = source_rr
else:
    status = UNAVAILABLE
```

Validation에서 sparse coverage가 20%를 넘으면 해당 policy는 무조건 탈락한다. frozen base와 동일한 출력은 serialization 후 float32 bit pattern까지 같아야 한다.

---

## 4. 학습 사양과 선택 규칙

### 4.1 고정 seed와 결정성

- model/data seeds: `20260828`, `20260829`, `20260830`
- inner split seed: split authority가 identity fold로 결정되므로 random split 없음
- bootstrap seed: `20260828`
- PyTorch deterministic algorithms: true
- cuDNN benchmark: false
- AMP: CUDA에서 true; lock 후 CPU/CUDA parity 별도 검증
- worker seed: `seed + 1009*outer_fold + 37*inner_fold + worker_id`

중단 후 resume는 RNG, epoch sampler position, optimizer, scheduler, scaler와 session chunk cursor를 모두 복원하지 못하면 거부한다.

### 4.2 supervised target

학습 label은 해당 fit identity의 reference-valid window에서만 사용한다.

- candidate soft target: `q_i ∝ exp(-abs(c_i-y)/0.50)`; masked candidates 제외.
- factor target: `argmin_{k∈{1,2,3,4}} abs(k*f0-y)`; 최소 오차가 2 bpm 초과면 factor loss mask 0.
- residual target: 선택된 nearest candidate에 대해 `clip(y-c_i,-0.75,+0.75)`.
- scale target: candidate absolute residual에 대한 Gaussian NLL.
- quality target은 reference validity가 아니라 label-free augmentation recoverability로 정의한다. 원 signal과 허용 perturbation view의 candidate top-1이 1 bpm 안에서 일치하면 1, 아니면 0이다.

Frozen-base error, “base보다 candidate가 좋은가”, test target/QC는 학습 target으로 쓰지 않는다. invalid-reference window는 chronological state와 label-free consistency에만 들어가며 supervised loss weight는 0이다.

### 4.3 loss

총 loss는 다음 하나로 고정한다.

```text
L = 1.00 * L_listwise_KL
  + 0.60 * L_source_smoothL1
  + 0.35 * L_factor_CE
  + 0.15 * L_residual_smoothL1
  + 0.08 * L_scale_NLL
  + 0.05 * L_view_consistency_JS
  + 0.03 * L_quality_BCE
  + 0.01 * L_spike_rate
  + 0.001 * L_parameter_L2
```

- SmoothL1 beta: 0.5 bpm.
- Factor CE class weight: outer-fit identity의 inverse-sqrt class frequency, `[0.5,4.0]` clip 후 평균 1로 normalize.
- Identity weight: identity마다 총 supervised mass가 동일하도록 `N/(|G|*n_g)`.
- RR-band sampler: `[6,12), [12,20), [20,25), [25,35], (35,45]`을 identity 안에서 inverse-sqrt 빈도로 sampling; 최대 row weight 4.
- Spike penalty: rate가 `[0.02,0.25]` 밖인 양만 squared penalty.
- weak invalid label weight: 0.0.

### 4.4 optimizer와 training budget

| 항목 | 값 |
|---|---:|
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `2e-4` |
| Betas | `(0.9, 0.999)` |
| Batch | 2 session chunks |
| Eval batch | 4 session chunks |
| Gradient accumulation | 4, effective 8 chunks |
| Gradient clip | global norm 2.0 |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 6, min LR `1e-6` |
| Epochs | 최대 120 |
| Minimum epochs | 20 |
| Early-stop patience | 18 |
| Min delta | 0.002 bpm-equivalent score |
| Checkpoint cadence | 매 epoch atomic last, improvement 시 atomic best |

Architecture/hyperparameter search는 없다. 위 값이 실패하면 같은 cohort에서 값을 바꾸지 않고 campaign을 killed로 닫는다.

### 4.5 고정 augmentation

모든 augmentation은 fit identity의 radar 신호에만 적용하고 label은 바꾸지 않는다.

- independent radar dropout: 각 radar `p=0.15`; 세 radar가 모두 drop되면 하나를 원복.
- gain: radar별 log-uniform `[0.75,1.25]`, `p=0.50`.
- DC offset: train radar robust SD의 `[-0.10,+0.10]`, `p=0.25`.
- Gaussian noise: observed robust SD의 `[0,0.05]`, `p=0.30`.
- impulse: frame의 최대 0.2%, amplitude robust SD의 최대 3배, `p=0.10`.
- timestamp jitter: `[-20,+20] ms`, `p=0.15`.
- range-bin shift: `{-1,0,+1}`, probabilities `[0.15,0.70,0.15]`.
- phase inversion: radar 하나에 `p=0.05`.
- flatline: 한 radar 0.4–1.2초, `p=0.05`.

범위를 바꾸거나 특정 fold에 augmentation을 끄지 않는다.

### 4.6 checkpoint 선택과 threshold lock

Checkpoint score는 frozen base를 보지 않는 validation source score로 고정한다.

```text
J_source = source_identity_macro_MAE
         + 0.25 * source_25_35_identity_macro_MAE
         + 2.00 * source_over_5_fraction
         + 0.05 * candidate_NLL
```

`J_source`가 최소인 epoch를 선택하고, 동률 `1e-6` 안에서는 더 이른 epoch를 택한다. 선택된 checkpoint를 잠근 뒤에만 validation에서 safety policy를 탐색한다.

Threshold grid는 정확히 다음 Cartesian product다.

- `t_prob ∈ {0.50,0.525,...,0.95}`
- `t_margin ∈ {0.00,0.05,...,0.40}`
- `t_entropy ∈ {0.30,0.40,...,0.90}`
- `t_quality ∈ {0.50,0.60,0.70,0.80}`
- `pull ∈ {0.25,0.50,0.75,1.00}`

동일 validation set에서 다음 constraints를 모두 만족한 policy만 eligible이다.

- correction precision ≥0.80
- actionable correction recall ≥0.20
- base-good false-positive fraction ≤0.01
- correction coverage ≤0.20
- candidate identity-macro MAE gain ≥0.05 bpm
- tail identity-macro MAE 악화 ≤0.00 bpm
- within-2 drop ≤0.005
- over-5 increase ≤0.002
- 최대 identity MAE harm ≤0.50 bpm

Eligible policy 중 `(macro gain, tail gain, precision, -coverage)`를 lexicographic 내림차순으로 선택한다. 동률이면 더 높은 threshold와 더 낮은 pull을 선택한다. eligible policy가 없으면 그 fold/seed는 exact base fallback으로 기록하는 것이 아니라 **Stage B failure**로 판정한다.

---

## 5. 완전 nested cross-fit DAG

### 5.1 DAG 원칙

어떤 row의 candidate proposer, scaler, model weight, threshold 또는 calibration도 그 row의 identity label을 본 적이 없어야 한다. 표준 outer fold `o`에 대해 다음 집합을 사용한다.

- `E_o`: outer-test fold `o`; 최종 one-shot 평가 전까지 봉인.
- `V_o`: validation fold `(o+1) mod 6`; checkpoint/threshold lock에만 사용.
- `T_o`: 나머지 4개 fold; weight/scaler fit.

### 5.2 inner stacking

`T_o` 안의 4개 physical-fold를 inner held fold로 사용한다. 각 inner fold `j`마다:

1. proposer `P_{o,j,s}`를 `T_o \ j`의 세 identity-fold로 학습한다.
2. scaler/SVD normalization도 `T_o \ j`에서만 fit한다.
3. held `j`의 모든 chronological windows를 prediction-only로 replay한다.
4. `j`에 대해 direct posterior top modes와 deterministic candidates/evidence를 생성한다.
5. 네 held prediction을 concat해 `T_o` exact-cover OOF candidate stack을 만든다.

Router `R_{o,s}`는 이 inner-OOF stack만 학습한다. proposer가 자기 train row를 다시 예측한 in-sample feature는 한 행도 들어가지 않는다.

### 5.3 validation과 test path

1. final proposer `P_{o,final,s}`를 전체 `T_o`로 학습한다.
2. `P_{o,final,s}`로 `V_o` candidate stack을 생성한다.
3. `R_{o,s}` checkpoint는 4.6절의 base-independent `J_source`로 선택한다.
4. frozen base는 그 뒤 validation safety grid에서만 읽는다.
5. checkpoint, scaler, proposer, threshold, pull, schema와 source hashes를 `LOCK.json`에 기록한다.
6. 모든 pre-open gate와 independent audit가 통과한 뒤에만 `E_o` raw/source를 한 번 replay한다.
7. test prediction을 쓴 뒤 `TEST_OPENED.json`을 원자적으로 남기고 재생성을 금지한다.

### 5.4 전체 흐름

```text
raw + metadata authority
        │
        ├── integrity / interval / identity audit
        │
        └── outer fold o
              ├── E_o: sealed
              ├── V_o: validation lock only
              └── T_o
                    ├── inner fold 0: fit proposer on 3 folds → held predictions
                    ├── inner fold 1: fit proposer on 3 folds → held predictions
                    ├── inner fold 2: fit proposer on 3 folds → held predictions
                    └── inner fold 3: fit proposer on 3 folds → held predictions
                              │
                              └── exact-cover T_o OOF candidate stack
                                       │
                                       └── fit HCS-E-SNN router R_o

              fit final proposer on T_o ──→ V_o stack ──→ checkpoint/threshold LOCK
                                                       │
                                                       └── gate pass only
                                                               │
                                                               └── E_o one-shot
                                                                        │
                                                                        └── 6-fold OOF
```

### 5.5 calibration nesting

Internal uncertainty interval은 각 outer fold에서 `V_o`만 이용해 split conformal absolute residual quantile을 적합하고 `E_o`에 적용한다. 최종 product threshold/interval은 prospective calibration cohort 전체에서 한 번 다시 fit해 confirmation 개봉 전에 hash-lock한다. Outer-test 또는 confirmation target으로 quantile/coverage를 고치지 않는다.

---

## 6. 단계별 실행과 decision gate

### Stage A — label-free separability 및 safe-policy screen

#### 목적

신경망 계산 전에 candidate evidence가 unseen identity에서 **안전한 sparse correction**을 만들 가능성이 있는지 grouped inner OOF로 판정한다. 높은 AUC만으로 통과하지 않는다.

#### 입력

- outer-test를 제외한 physical identities만.
- aggregate·verified harmonic/SVD evidence만.
- frozen base는 policy 결과를 validation comparator로만 사용.
- outer-test target/prediction은 열지 않음.

#### 고정 gate

두 discovery partitions 각각에서 다음을 동시에 만족해야 한다.

| Gate | 기준 |
|---|---:|
| Action AUROC | ≥0.80 |
| Action average precision | ≥0.45 |
| Factor accuracy − direct(`×1`) prevalence | ≥0.05 |
| Correction precision | ≥0.80 |
| Actionable correction recall | ≥0.20 |
| Base-good false-positive fraction | ≤0.01 |
| Estimated identity-macro MAE gain | ≥0.10 bpm |

한 threshold가 마지막 네 policy gate를 **동시에** 만족해야 한다. 서로 다른 threshold의 최선 값을 조합하지 않는다.

#### 판정

- 두 partitions 모두 pass: `A_PASS` 후 architecture/code freeze.
- 하나라도 fail: `A_KILLED`; neural train, outer-test, full OOF 금지.

### Stage B — 2-fold × 3-seed neural discovery validation

#### Entry

`A_PASS`, code/schema/unit tests pass, discovery outer folds `[3,4]`, seeds `[20260828,20260829,20260830]`가 모두 lock되어야 한다. Stage A current run이 fail했으므로 현재 cohort의 Stage B entry는 닫혀 있다.

#### 실행

각 discovery fold/seed에 대해 5절의 inner stack을 완성하고 **outer-validation까지만** 평가한다. discovery outer-test는 Stage B에서 열지 않는다.

#### pass rule

6개 `(fold,seed)` 조합 모두에서 다음을 만족한다.

- validation candidate macro MAE gain ≥0.05 bpm
- validation tail macro MAE 악화 ≤0.00 bpm
- within-2 drop ≤0.005
- over-5 increase ≤0.002
- maximum identity MAE harm ≤0.50 bpm
- sparse correction coverage ≤0.20
- no-base/no-radar/nonfinite exact fallback tests 100% pass
- parameter cap와 spike-rate operating band pass

한 조합이라도 fail하면 `B_KILLED`; full OOF를 열지 않는다.

### Stage C — locked 6-fold, 3-seed one-shot OOF

#### Entry

Stage B 6/6 pass, architecture/hyperparameter/source schema hash lock, independent leakage audit pass.

#### 실행

모든 6 outer folds와 3 seeds에 대해 nested DAG를 독립 수행한다. fold lock이 완료되기 전 test access를 차단하고 fold당 outer-test prediction generation은 한 번뿐이다. validation-promoted configuration 이외의 model은 test에서 실행하지 않는다.

#### pass rule

- 각 seed가 1.5절의 6개 accuracy gate 6/6 통과.
- 각 seed가 paired identity safety gate 통과.
- OOF exact rows 2,327, identities 18, folds 6, duplicate/missing 0.
- retrospective cohort에서는 결과가 좋아도 상태를 최대 `INTERNALLY_VALIDATED_RETROSPECTIVE`까지만 올림.

하나라도 실패하면 `C_KILLED`; test 결과를 보고 retrain하지 않는다.

### Stage D — deployment verification 및 product freeze

#### Entry

Stage C 3 seeds 전부 pass. Prospective program에서는 calibration cohort의 C pass가 필요하다.

#### required suites

1. 7 radar masks: `123, 12, 13, 23, 1, 2, 3`.
2. greedy non-overlap와 fixed stride phases 0–7.
3. clean/stress input suite.
4. outer-validation-only conformal/risk-coverage.
5. offline vs streaming golden parity.
6. CPU와 target accelerator batch-one latency/memory/spike telemetry.
7. restart, duplicate, out-of-order, state corruption, all-radar-missing.
8. signed immutable release bundle/SBOM/model card.

#### hard gates

- 모든 mask에서 finite output 또는 명시적 fallback 100%.
- single-radar catastrophic가 full-radar보다 `+5%p` 이상 증가하지 않음.
- pair-radar identity-macro MAE가 full-radar보다 `+0.35 bpm` 이상 증가하지 않음.
- greedy non-overlap에서 6개 primary point gate를 다시 판정하고, 실패하면 `limited-evidence`가 아니라 D fail.
- 8 phases 중 어느 phase도 full-window 대비 MAE `+0.30 bpm`, catastrophic `+2%p` 이상 악화되지 않음.
- streaming golden feature absolute tolerance `1e-6`, RR output tolerance CPU `1e-5 bpm`, CUDA AMP `0.05 bpm`; fallback은 bit-exact.
- batch-one end-to-end p99 <4,000 ms, warm p95 ≤250 ms, peak process RAM ≤1 GiB.
- nonfinite, crash, silent row drop, unsigned artifact 0건.

Stage D를 통과해도 independent confirmation 전 상태는 `PROSPECTIVELY_CALIBRATED`, 상용 배포 후보가 아니다.

---

## 7. 2026-08-28 Stage A 실행 checkpoint와 사전 kill

### 7.1 실행 binding

| 항목 | 값 |
|---|---|
| Campaign | `harmonic_factor_snn_v1` |
| Classification | retrospective-adaptive engineering |
| Feature source | `physics_candidate_features_v1.npy` |
| Feature SHA-256 | `37a9c2a2bed70a06a5c913e2e3844c3ec2fc97554487afaff5a18f96235c13c8` |
| Valid-row/base authority | `ensemble_structured_exact/ensemble_oof.npz` |
| Valid-row/base SHA-256 | `ab8c8fa03a9cf703319e32f57e8db05f6658c2bae59d2d1a6048ab81be4c772a` |
| Features/candidate | 228 aggregate+verified columns |
| Inner CV | physical identity GroupKFold(5) |
| Screen model | XGBoost pseudo-Huber utility ranker, fixed seed family |
| Discovery exclusions | outer folds 3 and 4 separately |

### 7.2 결과

| 지표 | outer-3 제외 pool | outer-4 제외 pool | Gate |
|---|---:|---:|---:|
| Rows / identities | 1,924 / 15 | 1,889 / 15 | audit |
| Reliable factor rows | 1,508 | 1,496 | audit |
| Actionable rows | 249 | 226 | audit |
| Action AUROC | 0.923733 | 0.921723 | ≥0.80 PASS |
| Average precision | 0.623306 | 0.626349 | ≥0.45 PASS |
| `×1` prevalence | 0.773210 | 0.774064 | baseline |
| Factor accuracy | 0.913130 | 0.913102 | audit |
| Factor accuracy gain | 0.139920 | 0.139037 | ≥0.05 PASS |
| Best policy precision | 0.476780 | 0.473684 | ≥0.80 **FAIL** |
| Best policy recall | 0.526104 | 0.048673 | ≥0.20 PASS / **FAIL** |
| Base-good FP | 0.006369 | 0.002571 | ≤0.01 PASS |
| Best macro MAE gain | 0.004363 bpm | 0.007303 bpm | ≥0.10 **FAIL** |
| Passing policies | 0 | 0 | ≥1 **FAIL** |

Outer-3 제외 pool의 best screen policy는 coverage 16.7879%, pull 0.25였지만 precision과 macro gain을 실패했다. Outer-4 제외 pool은 coverage 2.0116%, pull 0.75였고 precision, recall, macro gain을 실패했다. AUC/AP/factor accuracy가 높다는 사실은 안전한 threshold 존재를 의미하지 않는다.

### 7.3 원자적 결정

`CAMPAIGN_CONTRACT.json`의 kill rule에 따라 상태는 다음과 같이 확정한다.

```text
harmonic_factor_snn_v1 = KILLED_AT_STAGE_A
outer_3_test_opened = false
outer_4_test_opened = false
neural_training_authorized = false
full_oof_authorized = false
leader_changed = false
commercial_claim_allowed = false
```

동일 cohort에서 richer oracle, 다른 classifier, 더 넓은 threshold grid 또는 HCS-E-SNN을 실행해 이 kill을 소급 무효화하지 않는다. 새 prospective calibration/development cohort가 등록되면 이 문서의 architecture와 gates를 새 campaign ID에 그대로 bind하고 Stage A부터 실행한다.

---

## 8. 강건성·calibration·latency 검증 계약

### 8.1 radar availability

각 mask의 OOF prediction은 같은 model/checkpoint로 재생성한다. mask별 모델 재학습이나 threshold 변경은 금지한다. 보고 항목은 overall/macro/tail MAE, RMSE, within-2, over-5, unavailable, uncertainty width, spike rate다. identity별 최악값과 paired full-radar delta를 포함한다.

### 8.2 시간 중복 분리

- Greedy non-overlap: session별 timestamp 오름차순, 먼저 등장한 32초 interval을 채택하고 겹치는 후속 interval을 제거한다.
- Fixed phases: original stride index modulo 8의 phase 0–7을 각각 보고한다.
- phase를 보고 선택하지 않으며 전부 동일 표에 남긴다.

### 8.3 perturbation suite

Clean과 아래 stress를 각 5 deterministic seeds로 실행한다.

| Stress | 수준 |
|---|---|
| Gain | `×0.5, ×0.75, ×1.25, ×1.5` |
| Offset | train robust SD의 `±0.1, ±0.25` |
| Gaussian noise | robust SD의 `0.02, 0.05, 0.10` |
| Burst noise | 0.4/1.2/2.0초, 3× robust SD |
| Packet loss | 1%, 5%, 10% contiguous/noncontiguous |
| Timestamp jitter | ±20/50/100 ms |
| Range shift | ±1/±2 bins |
| Phase inversion | radar 1/2/3 each |
| Flatline | 1/4/8초 |
| Cold restart | arbitrary episode window at offsets 1–7 |

장비 사양이나 observed train distribution을 벗어나는 수준은 adversarial diagnostic으로 별도 표시한다. in-spec stress에서 clean 대비 macro MAE `+0.30 bpm`, catastrophic `+2%p`, unavailable `+2%p`를 넘거나 nonfinite가 발생하면 fail이다.

### 8.4 calibration

- nominal intervals: 50/70/80/90/95%.
- conformal score: `abs(y-prediction) / max(predicted_scale,0.25)`.
- quantile은 finite-sample correction `ceil((n+1)*(1-alpha))/n`으로 calibration identities에서 계산.
- 보고: window coverage, identity-macro coverage, width, interval score, worst identity/RR band/protocol/mask coverage.
- 90% interval hard gate: overall coverage `[0.87,0.93]`, identity-macro coverage ≥0.87, 최악 predeclared subgroup ≥0.80.
- uncertainty ranking: Spearman, AUROC/AUPRC for `>2`와 `>5`; point estimate gate 대체 불가.
- selective mode가 있으면 absolute all-window coverage ≥80%, reference-valid retained coverage ≥80%, unavailable ≤20%를 먼저 지키고 retained accuracy와 전체 분모 failure를 함께 보고한다.

### 8.5 end-to-end benchmark

측정 구간은 raw packet ingest, stateful outlier repair/resampling, feature/SVD, candidate build, SNN forward, decode, calibration, serialization 전체다. 100 warm-up 후 10,000 windows를 session chronology로 측정한다.

- CPU: 제품과 동일한 core pinning/thread 수, batch 1.
- accelerator: target hardware, batch 1, AMP on/off parity.
- cold: process/model/state load 포함 100회.
- warm: steady-state 10,000회.
- 보고: p50/p95/p99/max, acquisition queue delay, peak RSS/VRAM, energy/window, parameter count, spike activity.
- hard deadline: 4초 stride 안 p99.
- 목표: warm p95 ≤250 ms, peak RAM ≤1 GiB, memory leak slope의 95% CI가 0을 포함.

기존 두 component의 보수적 CPU p95 합 약 114.76 ms와 CUDA p95 합 약 60.90 ms는 참고치일 뿐 HCS-E-SNN E2E 통과 증거가 아니다. window-boundary outlier-repair parity가 해결되지 않은 현재 benchmark는 D gate를 통과시키지 못한다.

---

## 9. provenance, failure atomicity와 artifact tree

### 9.1 immutable run identity

모든 run은 아래 canonical JSON을 key-sorted UTF-8로 직렬화한 SHA-256의 앞 24자를 `run_id`로 사용한다.

```text
campaign_id, stage, cohort_hash, split_hash, source_hashes,
architecture_hash, config_hash, seed, outer_fold, inner_fold,
code_commit_or_tree_hash, dependency_lock_hash, device_fingerprint
```

상대 경로만 manifest에 저장하며 실행 중 resolve된 absolute path는 audit 필드로 분리한다. raw, metadata, split, scaler, proposer, router, checkpoint, thresholds, predictions와 metrics 사이의 모든 edge에 SHA-256을 기록한다.

### 9.2 atomic write protocol

1. 최종 directory와 같은 filesystem에 `<run_id>.incomplete.<uuid>`를 만든다.
2. config/source manifest를 먼저 쓰고 hash를 검증한다.
3. epoch마다 `last.tmp`에 checkpoint를 쓰고 file fsync 후 `last.pt`로 atomic rename한다.
4. best checkpoint는 prediction/score와 하나의 commit group으로 기록한다.
5. output NPZ/CSV의 row count, semantic key, finite, fold binding, SHA를 독립 재검증한다.
6. 모든 검사가 통과한 뒤 `COMMITTED.json`을 마지막에 쓴다.
7. directory fsync 후 `<run_id>`로 atomic rename한다.
8. `COMMITTED.json` 없는 directory는 resume source나 metric aggregation에서 무시한다.

Interrupted best/last commit, missing checkpoint, metadata reorder, cache-index permutation, base-fold swap, partial all-window를 complete로 표시하는 경우는 모두 fail-closed다. Retry는 같은 config/run identity로만 허용하고 seed/config 변경은 새 registered run이어야 한다.

### 9.3 artifact tree

```text
artifacts/programs/commercial_snn_v3/
├── MASTER_EXECUTION_PLAN.sha256
├── PROGRAM_STATE.json
├── cohort_registry/
│   ├── retrospective_18.lock.json
│   ├── prospective_calibration.lock.json
│   └── prospective_confirmation.sealed.json
├── stage_a/
│   └── <campaign_id>/
│       ├── CONTRACT.json
│       ├── INPUT_HASHES.json
│       ├── partition_3/report.json
│       ├── partition_4/report.json
│       ├── DECISION.json
│       └── SHA256SUMS.json
├── stage_b/
│   └── <campaign_id>/<seed>/outer_<3|4>/
│       ├── inner_<0..3>/proposer/
│       ├── train_stack/
│       ├── router/
│       ├── validation/
│       ├── LOCK.json
│       └── COMMITTED.json
├── stage_c/
│   └── <campaign_id>/<seed>/outer_<0..5>/
│       ├── inner_<0..3>/
│       ├── final_proposer/
│       ├── router/
│       ├── validation_lock/
│       ├── test_prediction/
│       └── COMMITTED.json
├── stage_d/
│   ├── radar_masks/
│   ├── nonoverlap_phases/
│   ├── stress/
│   ├── calibration/
│   ├── streaming_parity/
│   ├── latency/
│   └── DECISION.json
├── prospective_confirmation/
│   ├── PREREGISTRATION.json
│   ├── SEALED_INPUT_MANIFEST.json
│   ├── ONE_SHOT_PREDICTIONS/
│   ├── STATISTICAL_REPORT.json
│   └── DECISION.json
└── release/
    └── <release_id>/
        ├── model/
        ├── preprocessing/
        ├── calibration/
        ├── golden/
        ├── sbom/
        ├── model_card.md
        ├── rollback.json
        ├── RELEASE_MANIFEST.json
        └── SIGNATURE
```

### 9.4 required provenance

- raw radar/reference file hashes, parser version, clock alignment/QC version.
- metadata semantic row key: identity/session/window start/end/protocol/fold/cache index.
- source code tree hash와 dirty-file digest.
- Python/CUDA/cuDNN/PyTorch/snntorch/driver/OS/SBOM.
- CPU/GPU model, clock/power mode, thread/affinity.
- RNG state와 sampler cursor.
- selection score, 모든 eligible/ineligible threshold row, tie-break 이유.
- failed run도 stdout/stderr, last checkpoint, partial audit와 reason code 보존.

---

## 10. prospective calibration/development cohort 설계

동일 18명 Stage A가 killed이므로 HCS-E-SNN의 재개 조건은 새 data authority다. 이 cohort는 수집 시점에는 prospective지만 architecture/threshold 학습에 사용되므로 **confirmation cohort가 아니다**.

### 10.1 cohort 크기와 분리

- 최소 90개의 새로운 physical identities.
- 최소 2 sites, site당 36명 이상; 어느 site도 전체의 60% 초과 금지.
- 개발 18명과 identity/device-record 중복 0.
- 90명 전체를 identity-disjoint 6 folds(각 15명)로 사전 randomization; sex, age, BMI, site, skin/clothing condition, high-RR tolerance를 층화하되 prediction을 보지 않음.
- 12명 이상의 별도 reference-qualification pilot은 sync/QC pipeline 검증에만 쓰고 90명 성능 분모에서 제외한다.

### 10.2 최소 protocol support

각 identity는 안전·윤리 승인 범위에서 다음 block을 수행한다.

- quiet rest 8분.
- posture: seated/supine/standing 각 4분.
- paced RR 12, 18, 24, 30 bpm 각 3분; 30 bpm 불가 시 사전 medical exclusion reason 유지.
- transition: 12→24, 24→30, 30→12 bpm 각 2분.
- mild motion 6분.
- placement perturbation 3개 위치, 각 4분.
- controlled radar masks: 각 pair 2분, 각 single 2분.

전체 cohort에서 최소 support를 identity 기준으로 보장한다.

- 25–35 bpm valid identities ≥72.
- high-RR 독립 episodes ≥216.
- high-RR greedy non-overlap windows ≥1,440.
- single-radar valid identities ≥72/mask.
- motion valid identities ≥72.
- 각 prespecified age/BMI/sex/site subgroup ≥20 identities. 희소 subgroup은 primary가 아니라 descriptive로 사전 표시한다.

한 identity의 많은 overlapping windows로 identity 수 부족을 대체하지 않는다.

### 10.3 reference와 synchronization

- primary reference: hardware-synchronized capnography 또는 동등하게 검증된 breath-by-breath 장치.
- secondary reference: BIOPAC respiration belt; disagreement audit용이며 모델 입력 금지.
- shared TTL trigger 또는 공통 hardware clock.
- 시작 offset absolute ≤50 ms, drift ≤10 ms/min; 초과 session은 prediction-blind sync adjudication.
- reference algorithm/QC는 model output 개봉 전에 version/hash 동결.
- 두 명의 독립 reviewer가 prediction blind 상태로 QC하고 disagreement는 세 번째 reviewer가 adjudicate.
- exclusion reason은 model output 개봉 전에 확정하고 전체 등록 identity 기준 flow diagram을 보고.

### 10.4 calibration cohort 사용 한계

이 90명에서는 Stage A→B→C→D를 한 번 실행한다. Stage A가 또 실패하면 architecture family를 바꾸지 않고 program을 `BLOCKED_DATA_OR_HYPOTHESIS`로 닫는다. Stage C/D 통과 후에만:

- 최종 proposer/router를 90명 전체로 재학습.
- threshold/pull을 90명의 nested OOF prediction에서 고정.
- conformal quantile을 nested OOF residual로 고정.
- preprocessing/model/threshold/calibration bundle을 hash-lock.
- confirmation 통계 code와 subgroup table을 preregister.

90명 결과는 `prospectively_calibrated`이며 상용 확증으로 표현하지 않는다.

---

## 11. prospective confirmation cohort와 통계적 합격

### 11.1 independence와 표본수

- 최소 180개의 새로운 physical identities.
- 최소 3 sites, site당 50명 이상; 그중 최소 1 site는 calibration에 없던 site.
- calibration/development 90명과 retrospective 18명에 대해 identity/device-record/session 중복 0.
- primary analysis에서 등록된 180명 전원을 intent-to-evaluate로 유지.
- 25–35 bpm valid identities ≥120, high-RR independent episodes ≥360, greedy non-overlap high-RR windows ≥2,400.
- controlled single-radar condition valid identities ≥120/mask.
- sex strata 각 ≥70, age `<40`, `40–64`, `≥65` 각 ≥40, BMI `<25`, `25–29.9`, `≥30` 각 ≥40을 목표로 하며 임상적으로 부적절한 forced balancing은 금지한다.

180명은 고정 최소치다. Confirmation 개봉 전, calibration cohort의 **blinded identity-level variance와 missingness만** 사용한 100,000회 simulation으로 다음 power를 확인한다.

- true overall/macro MAE 0.80 bpm에서 one-sided upper 95% bound ≤1.00일 확률 ≥90%.
- true high-RR MAE 1.60 bpm에서 upper bound ≤2.00일 확률 ≥90%.
- true catastrophic 1.5%에서 upper bound ≤3.0%일 확률 ≥90%.
- true within-2 93%에서 lower bound ≥90%일 확률 ≥90%.

어느 endpoint든 power가 90% 미만이면 prediction을 열기 전에 identity 수를 늘린다. 감소는 허용하지 않는다. Sample-size 변경은 blinded statistician과 release authority가 서명한다.

### 11.2 preregistration와 blinding

Raw confirmation data 접근 전에 다음 hash를 공개/서명된 preregistration에 기록한다.

- inclusion/exclusion/QC와 missing-data rule.
- raw parser, streaming preprocessing, model, 3-seed aggregation rule.
- candidate builder, threshold/pull, conformal quantile.
- primary/secondary endpoints와 exact 산식.
- subgroup과 multiplicity 처리.
- statistical code/container.
- failure/unavailable counting policy.

Reference team은 prediction에 blind, inference team은 reference RR/QC outcome에 blind, statistician은 lock 전 model choice에 관여하지 않는다. 전체 confirmation prediction은 한 번 생성하며 실패 후 같은 cohort에서 재튜닝한 결과는 confirmation으로 재사용하지 않는다.

### 11.3 primary acceptance

Point estimate와 identity-cluster confidence bound를 모두 만족해야 한다.

| Endpoint | Point gate | Confirmation confidence gate |
|---|---:|---:|
| Overall MAE | ≤1.00 bpm | one-sided 95% upper ≤1.00 bpm |
| Identity-macro MAE | ≤1.00 bpm | one-sided 95% upper ≤1.00 bpm |
| RMSE | ≤1.80 bpm | one-sided 95% upper ≤1.80 bpm |
| Within ±2 bpm | ≥90% | one-sided 95% lower ≥90% |
| Error >5 bpm | ≤3% | one-sided 95% upper ≤3% |
| 25–35 bpm MAE | ≤2.00 bpm | one-sided 95% upper ≤2.00 bpm |

Confidence interval은 site-stratified identity-cluster bootstrap 100,000회, seed `20261001`, one-sided 95th/5th percentile로 계산한다. Confirmation의 6 endpoints는 모두 primary co-endpoint이므로 multiplicity 보정으로 하나를 구제하지 않고 **전부 통과**해야 한다.

### 11.4 harm·availability·calibration acceptance

- prespecified site/sex/age/BMI/posture/placement subgroup 중 identity ≥20인 group에서 MAE upper 95% bound ≤1.50 bpm.
- 어느 site도 catastrophic upper bound >5% 금지.
- high-RR identity의 90% 이상에서 MAE ≤3 bpm.
- inference crash/nonfinite/silent drop 0.
- full denominator unavailable ≤2%; all-radar-present unavailable ≤0.5%.
- 90% interval overall coverage `[0.87,0.93]`, identity-macro ≥0.87, subgroup ≥0.80.
- radar-mask와 device-fault gate는 Stage D와 동일 기준.
- target-device p99 <4초와 warm p95 ≤250 ms.

한 항목이라도 실패하면 상태는 `CONFIRMATION_FAILED`; 상용 배포 후보로 승격하지 않는다. 원인 분석과 새 architecture 개발은 가능하지만 새 독립 confirmation cohort가 필요하다.

---

## 12. release, shadow, canary, rollback 운영 계약

### 12.1 release bundle

Confirmation을 통과한 bundle에는 다음이 빠짐없이 들어간다.

- signed preprocessing binary/container와 state schema.
- 3-seed model/checkpoints 또는 사전 동결 aggregation artifact.
- scaler, candidate schema, threshold, pull, conformal calibration.
- input/output schema와 golden raw packets/features/predictions.
- CUDA/CPU numeric tolerance.
- SBOM, dependency licenses, vulnerability scan.
- model card, intended use, contraindication, known failure modes.
- data lineage, validation report, subgroup report.
- telemetry schema, privacy/retention policy.
- 직전 stable release ID와 one-command rollback manifest.

### 12.2 shadow

- 기간: 최소 14일.
- 규모: 최소 10,000 sessions, 100,000 windows, intended-use site 2곳 이상.
- output은 사용자 의사결정에 노출하지 않고 stable release와 병렬 비교.
- hard pass: crash 0, schema loss 0, p99 <4초 99.9% 이상, all-radar-present unavailable ≤0.5%, state replay mismatch 0.
- label-free drift: PSI ≤0.20, candidate entropy/mask/quality distribution이 calibration 99% control limit 안.
- delayed reference가 있는 subset에서는 confirmation primary gate의 point threshold를 유지.

### 12.3 canary 단계

| 단계 | Traffic | 최소 기간/표본 | 승격 조건 |
|---|---:|---:|---|
| C1 | 1% | 48시간, 500 sessions | rollback trigger 0 |
| C2 | 5% | 7일, 2,000 sessions | trigger 0, site별 telemetry pass |
| C3 | 25% | 7일, 5,000 sessions | trigger 0, delayed-label gate pass |
| C4 | 50% | 7일, 10,000 sessions | trigger 0, on-call sign-off |
| Candidate | 100% limited release | 14일 | release authority final sign-off |

Traffic 승격은 각 단계의 최소 기간과 최소 표본을 모두 채운 뒤 이루어진다. 자동 승격은 금지하고 QA와 release authority의 공동 서명이 필요하다.

### 12.4 automatic rollback

다음 중 하나면 5분 안에 직전 stable release로 자동 rollback하고 신규 session 배정을 중단한다.

- inference crash/nonfinite 1건.
- p99 ≥4초가 연속 15분 또는 windows의 0.1% 초과.
- all-radar-present unavailable >1.0% 또는 stable 대비 `+0.5%p`.
- state replay/idempotence mismatch 1건.
- input schema/signature/hash mismatch 1건.
- single-radar catastrophic sentinel이 stable 대비 `+5%p`.
- delayed-reference 7일 rolling MAE >1.2 bpm 또는 >5 bpm rate >4%.
- drift PSI >0.30 두 시간 연속 또는 >0.50 한 번.
- 개인정보/보안 incident severity high 이상.

Rollback은 model뿐 아니라 preprocessing, threshold, calibration과 state schema를 하나의 release unit으로 되돌린다. incompatible hidden state는 폐기하고 safe restart/fallback을 적용한다. rollback 후 같은 binary를 재배포하려면 root-cause, corrective action, regression evidence와 release authority 승인이 필요하다.

### 12.5 운영 관측성과 SLA

- 24/7: latency, error, unavailable, radar mask, state reset, drift, version/hash.
- 일별: site/device/firmware별 distribution과 canary/stable diff.
- 주별: delayed-label accuracy, calibration, subgroup harm.
- incident acknowledgement: severity critical 15분, high 1시간, medium 1영업일.
- model owner와 MLOps on-call을 release 전 지정.
- raw physiological payload는 telemetry에 저장하지 않고 pseudonymous aggregate와 approved replay sample만 보존.

---

## 13. RACI, 자원과 일정 상한

### 13.1 RACI

역할 약어: ML(Model Lead), DE(Data/Signal Engineer), CL(Clinical/Reference Lead), ST(Independent Statistician), QA(Validation/QA), OP(MLOps/Device), SP(Security/Privacy), RA(Release Authority), PO(Product Owner).

| Work package | R | A | C | I |
|---|---|---|---|---|
| Goal/gate 변경 통제 | ML, ST | RA | QA, CL, PO | 전체 |
| Raw/sync/reference pipeline | DE, CL | CL | QA, SP | ML, ST |
| Cohort enrollment/QC | CL | CL | ST, SP, QA | RA |
| Split/nested DAG | ML, DE | ML | ST, QA | RA |
| HCS-E-SNN 구현 | ML | ML | DE, QA | OP |
| Leakage/adversarial audit | QA | QA | ML, DE, ST | RA |
| Statistical analysis | ST | ST | QA, CL | ML, RA |
| Streaming/device benchmark | OP, DE | OP | QA, ML | RA |
| Security/privacy/SBOM | SP, OP | SP | QA | RA, PO |
| Release/shadow/canary | OP | RA | QA, ML, CL, SP | PO |
| Rollback/incident | OP | RA | ML, QA, SP | PO, CL |

같은 사람이 model 개발과 independent confirmation 통계 승인 또는 reference adjudication approval을 동시에 맡지 않는다.

### 13.2 compute/storage budget

현재 killed retrospective campaign에 허용된 추가 Stage B/C GPU budget은 **0 GPU-hour**다. 독립 prospective calibration cohort가 등록되고 Stage A를 통과한 경우 아래 상한을 적용한다.

| 항목 | 상한 |
|---|---:|
| Stage A screen | 64 CPU-core-hours, 32 GB RAM |
| Stage B nested discovery | 48 GPU-hours, 256 CPU-core-hours |
| Stage C 6-fold × 3-seed nested OOF | 180 GPU-hours, 1,200 CPU-core-hours |
| Stage D robustness/calibration | 40 GPU-hours, 800 CPU-core-hours |
| Export/parity/device | 20 GPU-hours + target device 120시간 |
| Total model compute | 288 GPU-hours, 2,288 CPU-core-hours |
| Training GPU | RTX 4070급 이상 1대, VRAM ≥12 GB; device fingerprint 고정 |
| Hot cache/scratch | 최대 1.5 TB, campaign 종료 후 reproducible cache만 유지 |
| Immutable artifacts | 최대 400 GB, 2 copies + off-site/controlled backup |
| Prospective raw/reference | 예상 8 TB, 암호화 저장, 별도 retention policy |

상한을 넘기기 전에 결과를 보고 architecture를 바꾸지 않는다. 초과가 필요한 경우 data/implementation fault 증거, unchanged hypothesis, RA/QA 승인을 새 decision record에 남긴다.

### 13.3 calendar execution envelope

| 구간 | 기간 상한 | 종료 산출물 |
|---|---:|---|
| Protocol/reference qualification | 4주 + 승인/IRB lead time | signed acquisition/QC protocol |
| Calibration cohort collection | 8주 | sealed 90-identity authority |
| Calibration Stage A–D | 4주 | `PROSPECTIVELY_CALIBRATED` 또는 kill |
| Confirmation enrollment/collection | 12주 | sealed 180-identity authority |
| One-shot confirmation analysis | 2주 | independent statistical decision |
| Device qualification/release bundle | 3주 | signed release candidate |
| Shadow/canary | 최소 7주 | deployment-candidate decision |

승인·모집 lead time을 제외한 실행 상한은 40주다. 모집과 장비 validation을 병렬화할 수 있지만 confirmation input을 model team에 조기 공개하지 않는다. 일정 압박으로 표본, gate, phase duration을 줄이지 않는다.

---

## 14. 상태 머신과 종료 정의

### 14.1 허용 상태

| 상태 | 의미 | 허용 전이 |
|---|---|---|
| `RESEARCH_CANDIDATE` | frozen baseline, 상용 주장 불가 | `A_SCREENING` |
| `A_SCREENING` | grouped inner-OOF separability 진행 | `A_PASS`, `A_KILLED` |
| `A_KILLED` | 사전 safe-policy gate 실패 | 새 independent cohort가 있어야 새 `A_SCREENING`; 동일 campaign 재개 금지 |
| `B_DISCOVERY` | 2 folds × 3 seeds validation-only | `B_PASS`, `B_KILLED` |
| `B_KILLED` | discovery validation 실패 | 새 independent development cohort 외 재개 금지 |
| `C_ONE_SHOT_OOF` | locked full OOF | `C_PASS`, `C_KILLED` |
| `INTERNALLY_VALIDATED` | 3 seeds/6 gates/안전 gate 통과 | `D_VERIFICATION` |
| `D_VERIFICATION` | robustness/calibration/device freeze | `D_PASS`, `D_KILLED` |
| `PROSPECTIVELY_CALIBRATED` | 새 development cohort 통과, confirmation 아님 | `CONFIRMATION_SEALED` |
| `CONFIRMATION_SEALED` | model/statistics lock, input unopened | `CONFIRMATION_RUNNING` |
| `PROSPECTIVELY_VALIDATED` | independent confirmation 전 gate 통과 | `SHADOW` |
| `SHADOW` | 사용자 영향 없는 production replay | `CANARY`, `ROLLED_BACK` |
| `CANARY` | 제한 traffic | `DEPLOYMENT_CANDIDATE`, `ROLLED_BACK` |
| `DEPLOYMENT_CANDIDATE` | 모든 기술·확증·운영 gate 통과 | controlled release/monitoring |
| `ROLLED_BACK` | 안전 trigger 발생 | root-cause 후 새 release process |
| `BLOCKED` | 외부 cohort/authority/reference/device가 없어 진행 불가 | missing authority 확보 후 직전 미개봉 state |

### 14.2 현재 machine state

```json
{
  "date": "2026-08-28",
  "program_state": "RESEARCH_CANDIDATE",
  "campaign_state": "A_KILLED",
  "campaign_id": "harmonic_factor_snn_v1",
  "accuracy_gates_passed": 0,
  "accuracy_gates_total": 6,
  "outer_test_opened_by_campaign": false,
  "neural_training_authorized_on_same_cohort": false,
  "commercial_claim_allowed": false,
  "unblock_authority": "sealed independent prospective calibration/development cohort"
}
```

### 14.3 완료 정의

Goal은 아래 문장이 모두 true일 때만 complete다.

1. HCS-E-SNN 단일 architecture가 nested calibration/development에서 3 seeds × 6 gates를 통과했다.
2. Stage D의 radar/non-overlap/stress/calibration/parity/latency/fault gate가 모두 통과했다.
3. 180명 이상 independent confirmation에서 6개 point gate와 6개 one-sided confidence gate를 모두 통과했다.
4. confirmation 이후 model/threshold/preprocessing/calibration이 바뀌지 않았다.
5. shadow/canary의 최소 기간·표본과 rollback safety를 통과했다.
6. release bundle, provenance, model card, SBOM, monitoring, rollback owner에 미해결 예외가 없다.

내부 수치만 통과하면 `INTERNALLY_VALIDATED`, prospective development까지 통과하면 `PROSPECTIVELY_CALIBRATED`, independent confirmation까지 통과하면 `PROSPECTIVELY_VALIDATED`다. **상용 deployment candidate는 운영 gate까지 통과한 마지막 상태만 의미한다.**

### 14.4 이번 실행의 종료 판정

현재 프로그램은 Stage A를 실제 실행했고 두 discovery partition 모두 사전 safe-policy gate를 실패했다. 따라서 이 execution pass의 올바른 terminal decision은 `A_KILLED`이며, 같은 cohort에서 신경망을 학습하거나 test를 열어 숫자를 만드는 것은 본 계약 위반이다. Frozen leader와 0/6 판정은 유지된다.

외부 prospective calibration/development와 confirmation이 아직 존재하지 않으므로 전체 상용 목표는 **`BLOCKED_BY_REQUIRED_EXTERNAL_EVIDENCE`**다. 이 상태를 성능 달성으로 표현하지 않는다.

---

## 15. 근거 authority

- 목표 명세: `artifacts/COMMERCIAL_SNN_GOAL_V2.md`
- 실행 결과 ledger: `artifacts/COMMERCIAL_SNN_PROGRESS_V2.md`
- retrospective campaign 계약: `artifacts/campaigns/harmonic_factor_snn_v1/CAMPAIGN_CONTRACT.json`
- Stage A 기계 판정: `artifacts/campaigns/harmonic_factor_snn_v1/STAGE_A_GATE.json`
- Stage A 종료 요약: `artifacts/campaigns/harmonic_factor_snn_v1/CAMPAIGN_SUMMARY.md`
- release manifest: `artifacts/COMMERCIAL_SNN_RELEASE_MANIFEST_V3.json`
- frozen leader metrics: `artifacts/runs/ensemble_structured_exact/metrics.json`
- frozen leader OOF: `artifacts/runs/ensemble_structured_exact/ensemble_oof.csv`
- feature source manifest: `artifacts/discovery_physics_ridge/feature_manifest.json`
- all-window strict artifact: `artifacts/runs/final_alias_gate_s12_deterministic/all_windows_cuda_v3/provenance.json`

이 authority 중 hash가 바뀌면 기존 decision을 자동 계승하지 않고 provenance audit부터 다시 수행한다.
