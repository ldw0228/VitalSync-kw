# 3-radar direct-RR SNN 설계·평가 보고서

## 결론

현재 정확도 선두는 주파수 구조를 보존하는 두 12-step SNN의 **validation-locked ensemble**이다. 18명의 물리적 identity를 완전히 분리한 6-fold OOF에서 BIOPAC RSP reference QC를 통과한 2,327개 window 기준 다음 성능을 기록했다.

- MAE 1.291 bpm
- identity-macro MAE 1.220 bpm
- RMSE 2.410 bpm
- ±2 bpm 80.791%
- >5 bpm 6.231%
- 25–35 bpm MAE 4.216 bpm

이 후보는 이전 flat/default SNN의 MAE 1.407 bpm보다 개선되었지만, 사전에 선언한 상용 목표 6개를 **0/6** 통과했다. 특히 25–35 bpm에서 bias −4.055 bpm, RMSE 5.984 bpm으로 강한 저추정이 남는다. 따라서 현재 상태는 누수를 억제한 강한 retrospective 연구 후보이지, 상용 또는 의료 성능을 입증한 제품이 아니다.

![OOF scatter and identity error](artifacts/report/oof_scatter_identity.png)

## 1. 선언한 상용 목표와 판정

| 지표 | 목표 | 현재 full-coverage OOF | 판정 |
|---|---:|---:|:---:|
| Overall MAE | ≤1.0 bpm | 1.291 | FAIL |
| Identity-macro MAE | ≤1.0 bpm | 1.220 | FAIL |
| RMSE | ≤1.8 bpm | 2.410 | FAIL |
| ±2 bpm | ≥90% | 80.791% | FAIL |
| >5 bpm | ≤3% | 6.231% | FAIL |
| 25–35 bpm MAE | ≤2.0 bpm | 4.216 | FAIL |

Identity-cluster bootstrap MAE 95% CI는 `[0.981, 1.473]`이다. CI 하한이 1.0 부근이라는 사실은 point gate 통과가 아니며 prospective 성능 보장을 뜻하지 않는다. 기계 판독 판정은 `artifacts/commercial_goal_report.json`, 사람이 읽는 요약은 `artifacts/COMMERCIAL_GOAL_AUDIT.md`에 있다.

## 2. 원본 데이터 감사

### 2.1 규모와 pairing

| 항목 | 결과 |
|---|---:|
| 세션 폴더 | 30 |
| 3-radar + BIOPAC 사용 가능 세션 | 29 |
| 물리적 identity | 18 |
| 선택된 3-view radar frame record | 4,715,613 |
| 동기화된 radar 시간량, 40 Hz 기준 | 10.916 h |
| BIOPAC 전체 시간량 | 11.380 h |
| paired 세션 BIOPAC 시간량 | 11.001 h |
| 생성된 32초 후보 window | 9,576 |
| reference-valid window | 2,327, 24.30% |

여러 세션 ID가 같은 사람을 가리키므로 session-random split은 누수를 만든다. PSJ, KTW, LHS, LDW, KDM, MDO, HDH, JKH, SJE, KDH, LJH의 반복 세션을 동일 identity로 묶었다. 모든 model selection과 최종 OOF는 물리적 identity를 일반화 단위로 삼는다.

원본 감사에서 파이프라인에 반영한 예외는 다음과 같다.

- `S24_KHJ`는 세 radar가 비어 있어 제외했다.
- `S01_CMS`는 긴 녹화 3개 chunk와 별도의 501-frame retry가 섞여 있어 긴 연속 기록만 선택했다.
- `S07_KDM` timestamp reset은 단조 시간축으로 복구했다.
- `S22_KJH` radar 2의 단일 비정상 sample은 미래 정보를 쓰지 않는 past-only 방식으로 보정했다.
- Paired BIOPAC RSP sample의 약 5.22%가 rail에 닿았다. `S30_SJE` 약 43.0%, `S06_LDW` 약 16.1%로 특히 심하다.

### 2.2 정렬과 reference

Radar와 BIOPAC은 파일 및 metadata의 absolute epoch overlap으로 맞췄고 radar는 nominal 40 Hz 시간 격자를 사용했다. Sub-frame hardware trigger나 공통 acquisition clock은 제공되지 않아 미세 residual synchronization error를 배제할 수 없다.

명시적 RR label이 없으므로 BIOPAC RSP에 0.10–0.75 Hz band, spectral/event/phase estimator 일치, 최소 cycle 수, 주기성, spectral concentration, interval CV, clipping guard와 phase residual QC를 적용했다. 25–35 bpm reference는 이 다중 estimator가 일치하는 구간이라 단순 label 오류만으로 현재 실패를 설명하기 어렵다. 그러나 capnography 또는 독립 adjudication reference는 아니며, reference가 없던 75.7% window의 성능은 측정할 수 없다.

## 3. Feature와 SNN 구조

### 3.1 입력

- 32초 causal window, 4초 stride
- Radar 40 Hz를 모델용 10 Hz로 causal downsample
- 2,048-point FFT, map frequency 0.08545–0.78857 Hz, 73 bins
- Radar별 `73 × 182` range–frequency map
- 182 값을 raw-power 91-bin branch와 candidate I/Q-phase-power 91-bin branch로 분리
- Radar spectrum/scalar/fusion auxiliary 1,205개와 strictly causal history 32개

182 값의 두 절반이 실제 I/Q 쌍이라는 장비 명세는 제공되지 않았다. 따라서 이 해석을 hypothesis로 명시하고 raw-only, phase-only와 both-map ablation을 보존했다.

### 3.2 Structured auxiliary TriRadarRRSNN

```text
3 radar range–frequency maps
        │ shared 2-D spatial encoder
        │ range attention + learned radar reliability
        ▼
reliability-weighted view fusion
        │
cached per-radar spectra ── topology-preserving structured fusion
        │
        ▼
PLIF/LIF residual Conv1D frequency backbone × 12 steps
        │
        ├── 6–45 bpm posterior / expected RR
        ├── heteroscedastic uncertainty
        ├── quality proxy
        └── spike-rate statistics
```

전체 map을 rate-code하지 않고 spatial compression과 view fusion을 analog CNN에서 수행한 뒤 frequency sequence를 spiking backbone으로 보낸 hybrid 구조다. 학습 가능한 decay/threshold, surrogate gradient와 residual spiking block을 사용한다.

Structured auxiliary path는 cached spectrum을 하나의 flat MLP 입력으로 소거하지 않고 radar와 주파수 topology를 보존한다. 두 번째 component는 auxiliary spectrum을 pair-pooled map grid에 정확히 맞추는 exact alignment를 사용한다. 두 component는 각각 1,298,548 trainable parameters다.

Exact run의 `--deterministic`은 deterministic mode를 요청하지만 CUDA adaptive-average-pool backward에 deterministic implementation이 없다는 warn-only 경고가 발생했다. 따라서 run 이름과 무관하게 bitwise reproducibility는 보장하지 않는다.

### 3.3 학습 목적과 강건성

- Gaussian soft-label RR distribution cross entropy
- Expected RR Huber loss
- Heteroscedastic negative log likelihood
- Quality BCE와 spike-rate penalty
- ANN teacher soft logits distillation
- Identity-balanced sampling과 train-identity 전용 scaler
- Coupled radar-view dropout
- Strictly causal past-window radar history

`radar_observable` quality target은 독립 센서 관측 가능성 주석이 아니다. Cached classical estimator가 reference의 ±2 bpm 안에 들었는지를 나타내는 proxy다.

## 4. 평가 protocol과 누수 통제

각 outer fold는 test identity 3명, validation identity 3명, train identity 12명으로 구성한다. Validation identity는 다음 identity fold로 고정했다. 다음 요소는 outer-test target을 보지 않고 train 또는 validation identity에서만 정한다.

- Auxiliary scaler와 network/teacher weight
- Early stopping
- Fold별 두 SNN convex weight
- Uncertainty disagreement coefficient
- Validation-only affine calibration 후보

선택된 structured component weight는 fold 0–5에서 각각 `0.60, 0.43, 0.58, 0.46, 0.28, 0.41`, disagreement coefficient는 `1, 8, 16, 16, 2, 1`이다. 이 lock을 만든 후에만 각 outer-test prediction에 적용했다.

현재 학습 코드는 외부 teacher checkpoint의 model type, fold, train/validation/test identity split, RR grid, auxiliary scaler, cache context, checkpoint–run-config signature와 SHA-256를 검증하며 mismatch를 거부한다. Resume checkpoint에는 Python/NumPy/Torch/CUDA/DataLoader/sampler RNG 상태와 teacher provenance를 저장한다. 또한 상용 감사 script는 canonical cache의 유효 index 2,327개, 6 folds, 18 identities, fold assignment와 row metadata가 candidate CSV와 완전히 같은지 검증하므로 쉬운 subset만 고른 결과를 full OOF로 인정하지 않는다.

OOF 2,327개 row는 각 identity에 대해 처음 보는 model prediction이다. 다만 32초 window가 4초 stride로 겹치므로 인접 row는 강하게 상관된다. Identity split은 사람 누수를 막지만 2,327개를 독립 표본으로 만들지 않는다. 또한 architecture iteration 전체가 같은 cohort의 outer 결과를 반복 관찰했으므로 완전히 unbiased한 최종 test라고 할 수 없다.

## 5. Full-coverage OOF 결과

| 방법 | MAE | Macro MAE | RMSE | ±2 | P95 AE | >5 | CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Structured + exact locked ensemble | **1.291** | **1.220** | **2.410** | **80.79%** | **5.604** | **6.23%** | **0.889** |
| Structured auxiliary SNN | 1.333 | 1.257 | 2.446 | 79.63% | 5.745 | 6.83% | 0.888 |
| Structured exact-alignment SNN | 1.347 | 1.268 | 2.590 | 80.45% | 5.931 | 6.96% | 0.872 |
| Alias-gated harmonic SNN | 1.351 | 1.294 | 2.536 | — | — | — | — |
| 이전 flat/default SNN | 1.407 | 1.360 | 2.657 | 80.15% | 6.424 | 7.74% | 0.867 |
| Structured ANN teacher | 1.482 | 1.441 | 2.806 | 79.29% | 6.723 | 8.59% | 0.855 |
| ExtraTrees grouped OOF | 1.557 | 1.517 | 2.654 | 75.59% | 6.400 | 7.82% | 0.855 |
| Classical spectral | 4.342 | 4.370 | 7.253 | 60.21% | 16.441 | 34.12% | 0.265 |

Structured auxiliary 단일 모델은 이전 flat/default SNN보다 MAE를 0.074 bpm 낮췄다. Exact-alignment model은 단독 성능이 더 낮지만 validation-selected diversity를 제공해 ensemble MAE를 1.291 bpm까지 낮췄다. Validation affine variant는 MAE 1.2907로 사실상 같지만 macro MAE 1.2200→1.2203, RMSE 2.41019→2.41036, ±2 80.791%→80.748%로 악화되어 uncalibrated blend를 primary로 유지했다.

## 6. Selective prediction

| Valid-reference retention | n | MAE | Macro MAE | RMSE | ±2 | >5 |
|---:|---:|---:|---:|---:|---:|---:|
| 100% | 2,327 | 1.291 | 1.220 | 2.410 | 80.79% | 6.23% |
| 90% | 2,095 | 1.001 | 0.974 | 1.978 | 86.44% | 3.68% |
| 80% | 1,862 | 0.713 | 0.729 | 1.447 | 92.21% | 1.40% |
| 70% | 1,629 | 0.508 | 0.502 | 1.108 | 97.24% | 0.43% |
| 50% | 1,164 | 0.448 | 0.423 | 1.011 | 98.28% | 0.17% |

![Risk coverage](artifacts/report/risk_coverage.png)

70% selective point 결과는 overall gate를 통과하지만 25–35 bpm row는 45.1%만 남는다. 남은 high-RR subset도 RMSE 3.805 bpm이며 >5 bpm 4.05%다. Coverage는 outer-test OOF uncertainty의 사후 quantile이므로 배포 threshold가 아니다. Reference-valid 24.30%와 70% retention의 관측 교집합은 전체 9,576개 후보의 약 17.0%에 불과하다.

## 7. 실패 분석

### 7.1 RR band

| Reference RR | n | MAE | Bias | RMSE | ±2 | >5 |
|---|---:|---:|---:|---:|---:|---:|
| 6–10 | 202 | 1.055 | +0.908 | 2.398 | 88.61% | 6.44% |
| 10–15 | 852 | 0.792 | +0.458 | 1.571 | 90.26% | 3.05% |
| 15–20 | 742 | 1.060 | +0.171 | 1.680 | 81.67% | 2.02% |
| 20–25 | 361 | 1.747 | −1.063 | 2.512 | 66.48% | 5.82% |
| 25–35 | 164 | **4.216** | **−4.055** | **5.984** | **48.78%** | **42.68%** |
| 35–46 | 6 | 1.176 | −1.176 | 1.301 | 100% | 0% |

35–46 bpm의 `n=6`은 일반화 근거가 아니다. 20–35 bpm 전체도 `n=525`, MAE 2.519, macro MAE 1.982, RMSE 3.940, >5 bpm 17.33%다.

고호흡수 window에서 radar dominant evidence가 reference의 약 1/3 또는 1/4 subharmonic에 놓이는 경우가 많다. Direct-vs-alias 여부는 grouped spectral 분석에서 AUC 0.988 수준으로 구분 가능했지만, ×3과 ×4 중 어떤 correction이 맞는지는 spectra만으로 거의 chance였고 identity별 prior도 불안정했다. 겹치지 않는 25–35 bpm 표본은 42개, 16 identities, 19 sessions뿐이다. 즉 핵심 병목은 단순 classifier 용량보다 **correction magnitude의 identity-dependent ambiguity와 tail support 부족**이다.

### 7.2 Protocol

| Protocol | n | MAE | RMSE | >5 |
|---|---:|---:|---:|---:|
| Dodge | 851 | 1.335 | 2.628 | 7.40% |
| Kick | 646 | **1.591** | **2.715** | **8.36%** |
| Strike | 830 | 1.012 | 1.863 | 3.37% |

Kick가 가장 어렵고 identity별 편차도 크다. 전체 평균만으로 사용자별 worst case를 보장할 수 없다.

![RR and protocol failure analysis](artifacts/report/failure_analysis.png)

## 8. 결측 radar 강건성

Fold별 validation에서 잠근 ensemble rule을 radar-mask 조건에도 바꾸지 않고 적용했다.

| 사용 radar | MAE | Macro MAE | RMSE | >5 |
|---|---:|---:|---:|---:|
| 1+2+3 | 1.291 | 1.220 | 2.410 | 6.23% |
| 1+2 | 1.486 | 1.417 | 2.755 | 7.91% |
| 1+3 | 1.445 | 1.374 | 2.723 | 7.31% |
| 2+3 | 1.422 | 1.338 | 2.709 | 7.31% |
| 1 only | 1.636 | 1.575 | 2.895 | 9.76% |
| 2 only | 1.554 | 1.478 | 2.882 | 8.59% |
| 3 only | 1.586 | 1.532 | 2.995 | 8.42% |

![Radar robustness](artifacts/report/radar_robustness.png)

이는 완전한 view를 0으로 mask한 이상적 stress test다. 실제 packet loss, partial corruption, range shift, placement drift와 sensor desynchronization은 더 어려울 수 있다.

## 9. 시간 중복에 대한 보수적 평가

Greedy하게 겹치지 않는 32초 window만 남기면 `n=444`, MAE 1.570, macro MAE 1.440, RMSE 2.860, ±2 77.25%, >5 bpm 8.78%다. 고정 stride phase 8개의 MAE 범위는 1.098–1.441, macro MAE 범위는 1.070–1.354, RMSE 범위는 1.946–2.652 bpm이다. 어느 phase도 선언한 full gate를 통과하지 않는다.

가장 좋은 phase만 test 결과를 보고 고르는 것은 누수이므로 primary claim에 사용하지 않았다. 이 결과는 4초 중첩 window aggregate가 독립 sample 평가보다 낙관적일 수 있음을 보여준다.

## 10. 불확실도와 calibration

Ensemble uncertainty는 validation에서 잠근 두 model uncertainty와 disagreement의 합성 **ranking score**다. 오차 `>2 bpm` 탐지 ROC-AUC는 0.908, `>5 bpm`은 0.899로 selective ranking에는 유용하다. 그러나 disagreement scale이 임의적이므로 RR interval sigma로 해석할 수 없다.

Structured auxiliary 단일 모델의 분포 interval도 under-coverage다.

| Nominal interval | 실제 포함률 |
|---|---:|
| 90% | 87.15% |
| 95% | 91.83% |
| 99% | 96.09% |

따라서 현재 uncertainty를 임상 confidence interval로 표시하면 안 된다. 독립 calibration cohort에서 conformal 또는 identity-aware calibration과 abstention threshold를 먼저 잠가야 한다.

## 11. End-to-end 추론 비용

실제 raw-file의 32초 3-radar window, strictly causal history, window-local feature extraction, tensor construction, host-to-device transfer와 batch-one forward를 측정했다. Resident raw window 200회와 warm-page-cache file path 100회를 사용했고 checkpoint–run-config–preprocessing-config SHA-256 일치를 검증했다.

| Component | CPU p50 / p95 | RTX 4070 p50 / p95 |
|---|---:|---:|
| Structured auxiliary | 49.67 / 51.67 ms | 29.09 / 30.93 ms |
| Structured exact | 49.89 / 63.09 ms | 28.43 / 29.97 ms |
| 순차 실행 산술합 | 99.56 / 114.76 ms | 57.53 / 60.90 ms |

순차 합은 따로 측정한 quantile의 산술합이며 shared preprocessing을 중복 계산하므로 직접 측정한 ensemble latency가 아니다. 그래도 4,000 ms stride budget보다 충분히 작다. 다만 현 benchmark의 outlier repair는 선택 window 경계에서 상태를 초기화하여 첫 4 frame이 전체 recording을 먼저 repair한 cache builder와 bit-exact하지 않을 수 있다. 따라서 이 측정은 timing evidence이지 production feature equivalence 증명은 아니다. 32초 causal 문맥을 채우는 시간, cold-disk, acquisition/network latency와 장시간 thermal behavior도 포함하지 않는다.

## 12. 설계·평가 loop의 결정 기록

| 반복 | 관찰 | 결정 |
|---|---|---|
| Flat/default 12-step SNN | MAE 1.407, macro 1.360 | 비교 기준으로 보존 |
| Structured auxiliary fusion | MAE 1.333, macro 1.257, 25–35 MAE 3.976 | **채택: 주 component** |
| Exact auxiliary alignment | 단독 MAE 1.347로 소폭 악화 | **채택: validation-locked diversity component** |
| Structured + exact locked blend | MAE 1.291, macro 1.220 | **채택: 현재 정확도 선두** |
| Validation affine calibration | MAE 변화는 −0.00018뿐이고 macro/RMSE/±2 악화 | 기각 |
| Tail loss 0.02 + KD error gate | 탐색 fold validation 악화 | 기각 |
| Tail loss 0.02, KD gate 없음 | Validation tail gain은 미미하고 전체 MAE 1.574→1.693 | 기각 |
| Unconditional harmonic head | Combined validation 악화 | 기각 |
| Legacy alias alignment | Exact alignment보다 악화 | 기각 |
| Train-only ×3/×4 hard router | Tail 4.216→4.029이나 full MAE 1.291→1.295, 유해 trigger 34.8% | 기각 |
| Protocol-aware router | 개선 0.00046 bpm | 실질 개선 없음, 기각 |
| Strict high-tail prior | Full MAE 1.299, macro 1.228, >5 6.53%; 유해 correction 71.4% | 기각 |
| 2-source causal decoder | Validation guard가 전 fold에서 비활성 | no-op, 기각 |

### Alias-gated harmonic head 최종 판정

Alias-head SNN은 완전한 6-fold, 2,327 rows, 18 identities에서 standalone expected-RR MAE 1.351, macro MAE 1.294, RMSE 2.536, 25–35 bpm MAE 4.049를 기록했다. Alias classifier 자체는 ROC-AUC 0.973, AP 0.930으로 강했지만 정확한 RR correction으로 이어지지 않았다.

Validation-locked 3-way blend는 fold 2와 4에서만 alias component를 채택했다. 그러나 outer macro MAE가 1.2209→1.2345, high-RR macro MAE가 3.6686→3.7512로 악화되어 기각했다.

Alias evidence를 쓰는 causal decoder는 fold 2와 3에서만 활성화되었고 high-RR macro MAE를 3.7512→3.4047로 개선했다. 반면 full macro MAE가 1.2987로 높아지고, non-overlap macro MAE가 0.0532 bpm, >5 비율이 0.731 percentage point 증가해 사전 guard를 통과하지 못했다. 따라서 최종 후보에는 포함하지 않는다. 근거는 `artifacts/runs/causal_alias_decoder/with_alias_gate/metrics.json`에 있다.

이 outer OOF로 재튜닝하지는 않았지만, accept/reject 판정 자체가 반복 실험 후의 선택 증거이다. 따라서 현 alias 기각 감사도 pristine prospective confirmatory test로 해석하지 않는다.

Classifier AUC가 높아도 correction divisor와 안전한 trigger를 결정할 정보가 부족하면 end metric은 악화될 수 있다. 이 실험은 high-RR 추가 데이터와 직접적인 alias/divisor supervision이 구조 변경보다 우선임을 보여준다.

## 13. 상용화 차단 요인

현재 자료로 아직 입증하지 못한 항목은 다음과 같다.

- 기존 18명과 독립적인 prospective cohort 일반화
- 25–35 bpm 및 고강도 motion에서 full-coverage 정확도
- 전체 운용 window의 reference-backed 성능과 실제 abstention rate
- 나이, 성별, 체형, 의복, 거리, 각도, 자세, 질환, 다중 인원 subgroup
- Capnography 또는 adjudicated independent reference 대비 정확도
- Hardware-triggered synchronization
- 미리 잠근 uncertainty calibration과 reject threshold
- 실제 packet loss, corruption, displacement와 radar 고장 안전성
- Target device의 end-to-end 전력, 열, 메모리와 장시간 drift

상용 claim 전에 필요한 locked 평가는 다음과 같다.

1. Feature schema, model checkpoint, RR range, quality score와 threshold를 먼저 동결한다.
2. 독립적인 다일·다기관 cohort를 모집하고 20–35 bpm, Kick-like motion, 현재 hard identity 조건을 의도적으로 oversample한다.
3. 공통 hardware trigger와 capnography 또는 전문가 adjudication reference를 사용한다.
4. Calibration set에서 threshold와 interval calibration만 fit한다.
5. Locked test set은 한 번만 열고 full coverage와 사전 선언 selective coverage를 함께 평가한다.
6. Identity/session cluster CI, subgroup worst case, 결측·장애 failure rate를 보고한다.
7. Target hardware에서 acquisition부터 output까지 latency, 전력, 메모리, 열과 장시간 안정성을 측정한다.

## 14. 재현 명령

핵심 실행 순서는 다음과 같다. 전체 옵션과 설치 방법은 `README.md`에 있다.

```bash
.venv/bin/python scripts/train.py \
  --config configs/default.yaml --model both --fold all \
  --preset default --simulation-steps 12 --device cuda --amp \
  --aux-fusion structured \
  --output-dir artifacts/runs/final_structured_aux_s12

.venv/bin/python scripts/train.py \
  --config configs/default.yaml --model snn --fold all \
  --preset default --simulation-steps 12 --device cuda --amp \
  --deterministic --aux-fusion structured --exact-aux-alignment \
  --teacher-checkpoint \
    'artifacts/runs/final_structured_aux_s12/fold_{fold}/teacher_best.pt' \
  --output-dir artifacts/runs/final_structured_exact_s12_deterministic

.venv/bin/python scripts/ensemble.py \
  --run-a artifacts/runs/final_structured_aux_s12 \
  --run-b artifacts/runs/final_structured_exact_s12_deterministic \
  --output-dir artifacts/runs/ensemble_structured_exact \
  --device cuda --workers 4

.venv/bin/python scripts/evaluate_ensemble_robustness.py
.venv/bin/python scripts/evaluate_commercial_goal.py
.venv/bin/python scripts/make_report.py
.venv/bin/pytest
```

## 15. 주요 산출물과 provenance

- `artifacts/runs/final_structured_aux_s12/snn_metrics.json`, `snn_oof.npz`
- `artifacts/runs/final_structured_exact_s12_deterministic/snn_metrics.json`, `snn_oof.npz`
- `artifacts/runs/ensemble_structured_exact/metrics.json`, `ensemble_oof.csv`
- `artifacts/robustness/ensemble_structured_exact/report.json`
- `artifacts/benchmarks/commercial/structured_aux_fold0_e2e.json`
- `artifacts/benchmarks/commercial/structured_exact_fold0_e2e.json`
- `artifacts/runs/causal_alias_decoder/with_alias_gate/metrics.json`
- `artifacts/commercial_goal_report.json`
- `artifacts/COMMERCIAL_GOAL_AUDIT.md`
- `artifacts/final_report.json`, `artifacts/report/`

Run signature는 structured auxiliary `d00cccdaf955a29f`, exact alignment `bbda5fc3aba5cc83`, locked ensemble `9d0b9ceeb088d5f5`다. 각 run은 config, fold assignment, checkpoint와 row-level OOF를 보존한다.

최종 해석은 명확하다. **Structured 2-SNN ensemble은 이 cohort에서 지금까지 가장 정확한 후보지만, 선언한 상용 목표는 달성하지 못했다.** 다음 성능 도약의 핵심은 같은 cohort에서 더 복잡한 router를 반복하는 것이 아니라, 독립적인 high-RR/divisor supervision과 prospective locked 검증을 확보하는 것이다.
