# Tri-radar SNN respiratory-rate estimator

3대 XeThru UWB 레이더의 32초 구간에서 호흡수(RR, breaths/min)를 직접 추정하는 재현 가능한 hybrid SNN 연구 파이프라인이다. 평가는 세션명이 아니라 물리적 사람 단위로 분리한 6-fold out-of-fold(OOF) protocol을 사용한다.

현재 정확도 선두는 주파수 구조를 보존하는 두 12-step SNN의 **validation-locked ensemble**이다. BIOPAC RSP reference QC를 통과한 2,327개 window에서 MAE 1.291 bpm, identity-macro MAE 1.220 bpm을 기록했다. 그러나 사전에 선언한 상용 목표 6개를 모두 통과하지 못했다. 특히 25–35 bpm MAE가 4.216 bpm이다. 따라서 이 저장소는 상용·의료 성능을 입증한 제품이 아니라, 외부 prospective 검증 전 단계의 retrospective 연구 후보이다.

현재 문서 authority는 [Goal v3](artifacts/COMMERCIAL_SNN_GOAL_V3_2026-08-31.md)와
[2026-08-31 개발 진행상황](artifacts/SNN_PROJECT_DEVELOPMENT_PROGRESS_2026-08-31.md)이
최우선이다. [기술 상태 보고서](artifacts/SNN_PROJECT_TECHNICAL_STATUS_REPORT_2026-08-30.md)와
[REPORT.md](REPORT.md)는 동결 결과의 근거다. Goal v2, 이전 master/progress 문서,
기존 release manifest와 `commercial_goal_report.json`은 역사적 세대의 증거이며 현재
학습·평가·release authority가 아니다. 서로 충돌하면 Goal v3와 2026-08-31 진행상황을
따른다.

SNN 학습 방식(STDP, ANN→SNN, surrogate-gradient, PLIF/ALIF, TET, distillation, self-supervised pretraining)의 비교와 이 데이터에 맞춘 최종 권고 구조·loss·평가 계약은 [SNN 학습 방법론과 최종 권고안](artifacts/SNN_TRAINING_METHODOLOGY_RECOMMENDATION_2026-08-30.md)에 정리했다.

프로젝트를 raw 취득·동기화·reference·denoising·dataset contract·base model·harmonic router·SNN 학습·검증·배포로 나눈 작업 지도와 파트별 기술 선택표, 의존성 DAG, 평가 funnel은 [작업 분해·기술 선택·검증 청사진](artifacts/SNN_PROJECT_WORKSTREAM_TECHNOLOGY_BLUEPRINT_2026-08-30.md)에서 볼 수 있다.

마지막으로 잠근 `harmonic_factor_snn_v1`은 target-dependent 10-candidate oracle에서 MAE 0.460 bpm의 표현 상한을 보였지만, outer fold 3/4를 각각 제외한 grouped inner-OOF Stage A에서 안전한 correction policy가 모두 0개였다. 사전 kill rule에 따라 neural training과 새 outer-test를 열지 않았고 현재 leader를 유지했다. 계약·기계 판정·종료 원장은 [campaign summary](artifacts/campaigns/harmonic_factor_snn_v1/CAMPAIGN_SUMMARY.md)에 묶여 있다.

## 선언한 상용 목표와 현재 판정

모든 수치는 동일한 18명, 2,327개 reference-valid identity-disjoint OOF row에 대한 full-coverage 결과다.

| 지표 | 목표 | 현재 ensemble | 판정 |
|---|---:|---:|:---:|
| Overall MAE | ≤1.0 bpm | 1.291 | FAIL |
| Identity-macro MAE | ≤1.0 bpm | 1.220 | FAIL |
| RMSE | ≤1.8 bpm | 2.410 | FAIL |
| ±2 bpm | ≥90% | 80.791% | FAIL |
| >5 bpm | ≤3% | 6.231% | FAIL |
| 25–35 bpm MAE | ≤2.0 bpm | 4.216 | FAIL |

즉, 현재 판정은 **0/6 통과**다. Identity-cluster bootstrap MAE 95% CI는 `[0.981, 1.473]`이다. 이 CI의 하한이 1.0 부근이라는 사실은 목표 달성을 뜻하지 않으며, point gate와 외부 검증을 대체하지 않는다.

## 현재 후보 구조

- 각 radar의 `73 × 182` range–frequency map을 공유 2-D spatial encoder로 처리한다.
- raw-power 91-bin branch와 candidate I/Q-phase-power 91-bin branch를 channel로 분리한다. I/Q 해석은 장비 명세로 확인된 사실이 아니라 hypothesis다.
- range attention과 학습된 radar reliability로 세 view를 융합한다.
- cached per-radar spectrum을 flat vector로 소거하지 않고 주파수 topology를 보존하는 structured auxiliary path로 결합한다.
- PLIF/LIF residual Conv1D frequency backbone을 12 simulation step 전개하고 6–45 bpm posterior, expected RR, uncertainty, quality proxy와 spike statistics를 출력한다.
- ANN teacher distillation, identity-balanced sampling, coupled radar-view dropout, strictly causal radar-history 32개 feature를 사용한다.
- 정확도 선두 ensemble은 legacy-grid structured SNN과 exact auxiliary-alignment structured SNN을 fold별 validation identity에서만 선택한 convex weight로 조합한다. Outer-test target으로 weight를 고르지 않는다.

두 component는 각각 1,298,548 trainable parameters다. Exact-alignment run의 `--deterministic`은 재현성 설정을 요청하지만, CUDA adaptive-average-pool backward가 deterministic implementation을 제공하지 않는다는 warn-only 경고가 발생하므로 bitwise deterministic 결과를 보장하지 않는다.

차기 `AxisRiskRouterSNNV8R5`는 frozen 571-wide harmonic candidate layout을 쓰는
228,838-parameter 미측정 제안이다. Governed feature layout과 spiking cell을 source
hash로 명시적으로 재사용한다. Evidence 값과 per-feature availability를
radar/ratio/branch/candidate 좌표와 pooling 전에 결합하고, directed harmonic PLIF
graph, causal PLIF→ALIF state, 분리된 value/route/risk head, explicit classical-RR
availability, differentiable expected-risk training 및 inference-only hard routing을
사용한다. Format-v2 cache contract validator와 synthetic correctness test까지
구현됐고 soft-routing gradient path도 synthetic test로 확인했지만, routing과 정확도
개선은 실데이터에서 미측정이다. Sync/stage/nested-training authority가 없어 상용
개선으로 해석하면 안 된다.

## 핵심 결과

| 후보 | MAE | Macro MAE | RMSE | ±2 bpm | >5 bpm | 25–35 MAE |
|---|---:|---:|---:|---:|---:|---:|
| Structured + exact, validation-locked ensemble | **1.291** | **1.220** | **2.410** | **80.79%** | **6.23%** | 4.216 |
| Structured auxiliary SNN | 1.333 | 1.257 | 2.446 | 79.63% | 6.83% | **3.976** |
| Structured exact-alignment SNN | 1.347 | 1.268 | 2.590 | 80.45% | 6.96% | 4.485 |
| 이전 flat/default SNN | 1.407 | 1.360 | 2.657 | 80.15% | 7.74% | 4.450 |
| ExtraTrees grouped OOF | 1.557 | 1.517 | 2.654 | 75.59% | 7.82% | — |
| Classical spectral | 4.342 | 4.370 | 7.253 | 60.21% | 34.12% | — |

Structured auxiliary 단일 모델은 이전 flat/default SNN보다 MAE를 0.074 bpm 낮췄다. Exact component는 단독으로 더 좋지 않지만 validation-locked diversity를 제공해 ensemble MAE를 1.291 bpm까지 낮췄다. Validation-only affine calibration은 개선하지 못해 최종 결과에 적용하지 않았다.

## Selective, 결측 radar, 비중첩 결과

Ensemble uncertainty는 오차 순위를 매기는 score이며 calibrated RR 표준편차가 아니다.

| Reference-valid retention | n | MAE | Macro MAE | RMSE | ±2 bpm | >5 bpm |
|---:|---:|---:|---:|---:|---:|---:|
| 100% | 2,327 | 1.291 | 1.220 | 2.410 | 80.79% | 6.23% |
| 90% | 2,095 | 1.001 | 0.974 | 1.978 | 86.44% | 3.68% |
| 80% | 1,862 | 0.713 | 0.729 | 1.447 | 92.21% | 1.40% |
| 70% | 1,629 | 0.508 | 0.502 | 1.108 | 97.24% | 0.43% |
| 50% | 1,164 | 0.448 | 0.423 | 1.011 | 98.28% | 0.17% |

70% selective point 결과는 좋아도 25–35 bpm row의 45.1%만 남기며, threshold를 이 OOF test 결과의 quantile로 사후 정한 분석이다. 상용 coverage 또는 배포 threshold로 주장할 수 없다. 전체 9,576개 후보 중 reference-valid 비율은 24.30%이므로 70% retention과의 관측 교집합은 전체 후보의 약 17.0%뿐이다.

| 사용 radar | Overall MAE | Macro MAE | RMSE |
|---|---:|---:|---:|
| 1+2+3 | 1.291 | 1.220 | 2.410 |
| 1+2 | 1.486 | 1.417 | 2.755 |
| 1+3 | 1.445 | 1.374 | 2.723 |
| 2+3 | 1.422 | 1.338 | 2.709 |
| 1 only | 1.636 | 1.575 | 2.895 |
| 2 only | 1.554 | 1.478 | 2.882 |
| 3 only | 1.586 | 1.532 | 2.995 |

이는 이상적인 full-view masking 실험이다. 실제 packet loss, partial corruption, radar displacement를 대신하지 않는다. 서로 겹치지 않도록 greedy하게 고른 32초 subset은 `n=444`, MAE 1.570, macro MAE 1.440, RMSE 2.860, >5 bpm 8.78%다. 고정 stride phase 8개의 MAE 범위도 1.098–1.441 bpm으로 어느 phase도 상용 gate를 통과하지 않았다.

## 설치와 재현

Python 3.12 환경을 전제로 한다. 환경은 직접 설치 명령을 조합하기보다 복구 계약에
포함된 bootstrap을 우선 사용한다.

```bash
bash restore/bootstrap_env.sh
bash restore/verify_restore.sh
```

아래 명령은 보존 leader를 새로 증명하는 현재 과학 campaign이 아니라 **역사적 legacy
reproduction**이다. 현재 acquisition-v2 synchronization authority는 `0/29`이므로 이
실행에서 같은 수치가 나와도 새 과학 증거나 상용 성능이 아니다. 보존 경로를 절대
덮어쓰지 않도록 매 시도마다 고유한 새 versioned root를 지정한다.

원본 zip이 풀려 `HAI_EXPERIMENT/`가 존재한다는 전제에서 다음처럼 실행한다.

```bash
# 실행마다 이름을 바꾼 새 root. 기존 artifacts/runs/final_* 경로 사용 금지.
REPRO_ROOT=artifacts/reproductions/legacy_leader_20260831_attempt_001
mkdir -p artifacts/reproductions
mkdir "$REPRO_ROOT"  # 이미 있으면 중단; 기존 attempt 재사용 금지

# 1. 읽기 전용 원본 감사와 별도 legacy feature cache
.venv/bin/python scripts/audit_dataset.py HAI_EXPERIMENT \
  --format json --output "$REPRO_ROOT/dataset_audit.json"
.venv/bin/python scripts/build_features.py \
  --config configs/default.yaml --cache-dir "$REPRO_ROOT/cache/rf32s" --force

# 2. Structured auxiliary teacher + 12-step SNN, six folds
.venv/bin/python scripts/train.py \
  --config configs/default.yaml --model both --fold all \
  --preset default --simulation-steps 12 --device cuda --amp \
  --aux-fusion structured --cache-dir "$REPRO_ROOT/cache/rf32s" \
  --output-dir "$REPRO_ROOT/structured_aux"

# 3. 같은 teacher를 사용한 exact-alignment SNN
.venv/bin/python scripts/train.py \
  --config configs/default.yaml --model snn --fold all \
  --preset default --simulation-steps 12 --device cuda --amp \
  --deterministic --aux-fusion structured --exact-aux-alignment \
  --cache-dir "$REPRO_ROOT/cache/rf32s" \
  --teacher-checkpoint "$REPRO_ROOT/structured_aux/fold_{fold}/teacher_best.pt" \
  --output-dir "$REPRO_ROOT/structured_exact"

# 4. Validation-only fold lock을 사용하는 ensemble
.venv/bin/python scripts/ensemble.py \
  --run-a "$REPRO_ROOT/structured_aux" \
  --run-b "$REPRO_ROOT/structured_exact" \
  --cache-dir "$REPRO_ROOT/cache/rf32s" \
  --output-dir "$REPRO_ROOT/ensemble" \
  --device cuda --workers 4

# 5. 각 component의 radar-mask 평가 후 locked rule을 그대로 결합
.venv/bin/python scripts/benchmark_robustness.py \
  --run-dir "$REPRO_ROOT/structured_aux" \
  --cache-dir "$REPRO_ROOT/cache/rf32s" \
  --output-dir "$REPRO_ROOT/robustness_aux" \
  --device cuda
.venv/bin/python scripts/benchmark_robustness.py \
  --run-dir "$REPRO_ROOT/structured_exact" \
  --cache-dir "$REPRO_ROOT/cache/rf32s" \
  --output-dir "$REPRO_ROOT/robustness_exact" \
  --device cuda
.venv/bin/python scripts/evaluate_ensemble_robustness.py \
  --ensemble-dir "$REPRO_ROOT/ensemble" \
  --source-a-robustness "$REPRO_ROOT/robustness_aux" \
  --source-b-robustness "$REPRO_ROOT/robustness_exact" \
  --cache-dir "$REPRO_ROOT/cache/rf32s" \
  --output-dir "$REPRO_ROOT/robustness_ensemble"

# 6. Raw 32 s radar window부터 batch-one end-to-end benchmark
.venv/bin/python scripts/benchmark_e2e.py \
  --checkpoint "$REPRO_ROOT/structured_aux/fold_0/snn_best.pt" \
  --cache-dir "$REPRO_ROOT/cache/rf32s" \
  --input-source raw-file --devices all --warmup 10 --repeats 200 \
  --file-repeats 100 \
  --output "$REPRO_ROOT/benchmarks/structured_aux_fold0_e2e.json"
.venv/bin/python scripts/benchmark_e2e.py \
  --checkpoint "$REPRO_ROOT/structured_exact/fold_0/snn_best.pt" \
  --cache-dir "$REPRO_ROOT/cache/rf32s" \
  --input-source raw-file --devices all --warmup 10 --repeats 200 \
  --file-repeats 100 \
  --output "$REPRO_ROOT/benchmarks/structured_exact_fold0_e2e.json"

# 7. 역사적 gate 형식 감사와 별도 보고서, 테스트
.venv/bin/python scripts/evaluate_commercial_goal.py \
  --candidate-csv "$REPRO_ROOT/ensemble/ensemble_oof.csv" \
  --cache-dir "$REPRO_ROOT/cache/rf32s" \
  --fold-assignments "$REPRO_ROOT/structured_aux/fold_assignments.json" \
  --robustness-report "$REPRO_ROOT/robustness_ensemble/report.json" \
  --e2e-reports "$REPRO_ROOT/benchmarks/structured_aux_fold0_e2e.json" \
    "$REPRO_ROOT/benchmarks/structured_exact_fold0_e2e.json" \
  --output-json "$REPRO_ROOT/commercial_goal_report_historical.json" \
  --output-markdown "$REPRO_ROOT/COMMERCIAL_GOAL_AUDIT_HISTORICAL.md"
.venv/bin/python -m pytest -q
```

학습은 fold별 best/last checkpoint와 validation/test prediction을 저장한다. `--resume`은
동일 시도의 `$REPRO_ROOT/structured_aux` 또는 `$REPRO_ROOT/structured_exact`에만 사용한다.
보존 leader 경로, 다른 시도의 root, hash가 다른 cache/checkpoint를 resume 대상으로
사용하지 않는다. GPU가 없으면 `--device cpu --no-amp`를 쓸 수 있지만 시간이 오래
걸리고 CUDA 재현 증거를 대체하지 않는다.

## End-to-end 지연시간

실제 raw-file에서 읽은 32초 3-radar window, strictly causal history, window-local feature extraction, host-to-device transfer와 batch-one forward까지 측정했다. resident raw window 기준이며 checkpoint–run-config–preprocessing-config SHA-256 일치를 검증했다.

| Component | CPU p50 / p95 | RTX 4070 p50 / p95 |
|---|---:|---:|
| Structured auxiliary | 49.67 / 51.67 ms | 29.09 / 30.93 ms |
| Structured exact | 49.89 / 63.09 ms | 28.43 / 29.97 ms |
| 두 component 순차 합산 상한 추정 | 99.56 / 114.76 ms | 57.53 / 60.90 ms |

마지막 행은 서로 따로 측정한 quantile의 산술합이라 shared preprocessing을 중복 계산하며, 직접 측정한 ensemble quantile이 아니다. 그래도 4초 output stride보다 충분히 작다. 현 benchmark의 outlier repair는 선택 window 경계에서 상태를 초기화하므로 첫 4 frame은 전체 recording을 먼저 repair한 cache builder와 bit-exact하지 않을 수 있다. 32초 문맥을 모으는 필수 대기시간, cold-disk I/O, 장치 acquisition/network latency는 포함하지 않는다.

## 주요 산출물

- `artifacts/runs/final_structured_aux_s12/snn_metrics.json`, `snn_oof.npz`: structured 단일 SNN
- `artifacts/runs/final_structured_exact_s12_deterministic/snn_metrics.json`, `snn_oof.npz`: exact-alignment 단일 SNN
- `artifacts/runs/ensemble_structured_exact/metrics.json`, `ensemble_oof.csv`: 현재 정확도 선두와 fold별 lock
- `artifacts/robustness/ensemble_structured_exact/report.json`: locked radar-mask·RR band·protocol·비중첩 감사
- `artifacts/runs/causal_alias_decoder/with_alias_gate/metrics.json`: alias-head/3-way/causal decoder 최종 기각 감사
- `artifacts/benchmarks/commercial/*.json`: raw-window end-to-end benchmark
- `artifacts/commercial_goal_report.json`: 사전 선언 상용 목표 6개 판정
- `artifacts/COMMERCIAL_GOAL_AUDIT.md`: 사람이 읽는 판정표
- `artifacts/baselines/final/metrics.json`: identity-disjoint 비교군
- `artifacts/final_report.json`, `artifacts/report/`: 요약 JSON과 그림

## Alias-head 최종 판정

Full 6-fold alias-gated harmonic SNN은 MAE 1.351, macro MAE 1.294, RMSE 2.536이었다. Alias classifier는 ROC-AUC 0.973/AP 0.930이었지만 25–35 bpm MAE는 4.049 bpm으로 남았다. Validation-locked 3-way blend는 outer macro MAE를 1.2209→1.2345로 악화시켰고, causal decoder는 high-RR macro를 3.7512→3.4047로 낮췄지만 full macro, non-overlap, >5 bpm 오류율을 악화시켰다. 두 후보 모두 retrospective acceptance guard에서 **REJECT**되었으며 test 수치로 사후 재튜닝하지 않았다. 이 outer OOF reject 판정 자체도 반복된 cohort 선택 증거이므로 prospective confirmatory test는 아니다.

## 해석상의 필수 제한

- 18명, 단일 retrospective cohort이고 외부 검증이 없다.
- 32초 window가 4초 간격으로 겹친다. Identity split은 누수를 막지만 2,327개 row를 독립 표본으로 만들지는 않는다.
- reference는 독립 capnography가 아니라 QC를 통과한 BIOPAC RSP 파생 RR이다.
- `radar_observable` quality target은 독립 주석이 아니라 cached classical estimator가 ±2 bpm 이내였는지를 뜻한다.
- Ensemble weight와 disagreement는 각 outer fold의 validation identity에서 잠갔지만, architecture iteration 전체가 같은 cohort를 반복 관찰했다.
- Selective threshold와 uncertainty interval은 prospective calibration cohort에서 잠기지 않았다.
- 결측 radar 결과는 mask stress test이며 실제 장애 조건이 아니다.

현재 결과는 prospective 평가 설계와 다음 모델 선택의 근거다. 상용 또는 의료 claim에는 독립 reference, 미리 잠근 모델/threshold, 외부 cohort와 장치 수준 안전성 검증이 추가로 필요하다.
