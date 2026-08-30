# 상용 후보 SNN 고도화 실행 결과 v2

기준일: 2026-08-28  
상태: **research candidate — 내부 정확도 0/6, 외부 prospective 검증 없음**

이 문서는 `COMMERCIAL_SNN_GOAL_V2.md`에 정의한 목표를 실제 데이터와 코드에 적용해 반복한 결과, 승격·기각 근거, 현재 동결 후보, 남은 차단 조건을 한곳에 묶는다. 수치가 좋아 보이는 사후 subset이나 oracle은 상용 성능으로 계산하지 않았다.

## 1. 결론

현재 정확도 선두는 `ensemble_structured_exact`의 validation-locked 두 SNN ensemble이다. 18명의 identity를 완전히 분리한 6-fold OOF, reference-valid 2,327개 window 전체에서 MAE 1.291 bpm, identity-macro MAE 1.220 bpm이다. 선언한 상용 정확도 목표는 **0/6 통과**다.

SVD raw-window 표현, source/divisor specialist, causal episode state, RR-balanced loss, nested tree router, physics ridge/HMM router, uncertainty와 temporal decoder를 단계적으로 검증했지만, validation safety gate를 지키며 현재 선두를 이긴 후보는 없었다. 5-expert oracle은 목표를 통과하므로 정보 상한은 존재하지만, 새로운 identity에서 어떤 harmonic correction이 안전한지 label-free로 선택하는 router가 재현되지 않았다.

따라서 현재 저장소에서 정직하게 내릴 수 있는 최고 판정은 다음과 같다.

- 내부 선두: `ensemble_structured_exact` 유지
- 신규 full-OOF 승격: 없음
- 배포용 all-window inference 무결성: CUDA-AMP strict parity와 raw-source binding까지 통과
- 상용 성능 달성: 아님
- 다음 필수 입력: 독립 prospective calibration/confirmation cohort와 hardware-synchronized reference

## 2. 동결 정확도 판정

평가 분모는 18 identities, 29 usable sessions, 2,327 reference-valid windows이다. 전체 후보 window는 9,576개이며 reference-valid 비율은 24.30%다.

| 1차 지표 | 목표 | 현재 | 격차 | 판정 |
|---|---:|---:|---:|:---:|
| Overall MAE | ≤1.000 bpm | 1.291 | +0.291 | FAIL |
| Identity-macro MAE | ≤1.000 bpm | 1.220 | +0.220 | FAIL |
| RMSE | ≤1.800 bpm | 2.410 | +0.610 | FAIL |
| ±2 bpm | ≥90.0% | 80.79% | −9.21%p | FAIL |
| >5 bpm | ≤3.0% | 6.23% | +3.23%p | FAIL |
| 25–35 bpm MAE | ≤2.000 bpm | 4.216 | +2.216 | FAIL |

25–35 bpm은 `n=164`, identity-macro MAE 3.667 bpm, RMSE 5.984 bpm, bias −4.055 bpm, ±2 bpm 48.78%, >5 bpm 42.68%다. 겹치지 않는 25–35 bpm window는 42개뿐이므로 평균 성능과 correction router 양쪽 모두 support가 부족하다.

Identity-cluster bootstrap overall-MAE 95% CI는 `[0.981, 1.473]`이다. CI 하한이 1.0 아래라는 사실은 point gate를 통과했다는 뜻이 아니며, 18명 cohort의 불확실성이 크다는 뜻이다.

## 3. 평가 계약

모든 후보는 다음 순서를 지켰다.

1. 물리적 identity 기준 outer 6-fold를 유지한다.
2. scaler, sampler, weak-label policy와 model weight는 outer-train identity에서만 적합한다.
3. epoch, threshold, correction pull, ensemble weight와 승격 여부는 outer-validation identity에서 잠근다.
4. lock 이후 outer-test prediction을 한 번 생성한다.
5. validation에서 승격된 fold만 locked candidate를 쓰고, 나머지는 frozen base로 되돌린다.
6. full 6-fold OOF가 완성된 후보만 현재 선두와 비교한다.
7. outer 결과를 본 뒤 설계한 모든 후속 실험은 `retrospective-adaptive`로 기록한다.

이 계약에도 불구하고 기존 base OOF learner가 episode specialist 관점에서 완전한 outer-nested base는 아니다. 따라서 learned source weight가 test identity와 분리되어도 base comparator와 validation threshold는 확증 증거가 될 수 없다. 새로운 nested base 또는 prospective frozen evaluation 전에는 episode 결과를 상용 주장에 사용하지 않는다.

## 4. 반복 실험 ledger

| 반복 | 핵심 가설 | 검증 결과 | 최종 결정 |
|---|---|---|---|
| Flat/default SNN | 기본 direct posterior | MAE 1.407, macro 1.360 | 과거 기준점 |
| Structured auxiliary SNN | 주파수 topology 보존 | MAE 1.333, macro 1.257, tail 3.976 | 주 component 채택 |
| Exact auxiliary SNN | map–spectrum 정렬 | 단독 MAE 1.347, tail 4.485 | diversity component 채택 |
| Validation-locked two-SNN ensemble | fold별 보완성 | MAE 1.291, macro 1.220 | **현재 선두** |
| Alias-gated harmonic SNN | alias 검출 후 correction | alias AUC 0.973/AP 0.930이나 MAE 1.351, tail 4.049 | 기각 |
| Causal alias decoder | 지속 episode를 이용한 보정 | MAE 1.830, macro 1.734, tail 4.958 | 기각 |
| Raw-window randomized-SVD source SNN v1 | cached map 밖 raw source 정보 | fold 4만 validation 승격; locked full MAE 1.294, macro 1.221 | 기각 |
| SVD source-supervised v2 | source loss 강화 | discovery folds 0/1 모두 승격 실패 | 조기 중단 |
| SVD temporal SNN | causal context가 divisor를 구분 | discovery folds 0/1 모두 승격 실패 | 조기 중단 |
| Nested high-RR tree router v1 | label-free high-RR correction | candidate MAE 1.345, macro 1.288 | 기각 |
| Nested inner-OOF tree router v2 | outer-test 없는 grouped router | 6개 fold 모두 promotion 실패, exact base fallback | 기각 |
| Physics ridge + causal HMM | classical ×1…×4 factor 선택 | train-safe threshold가 6개 fold 모두 없음 | 기각 |
| RR-balanced exact SNN p=0.50 | rare tail oversampling | validation 악화 | 조기 중단 |
| RR-balanced exact SNN p=0.25 | tail/overall 절충 | full MAE 1.353, macro 1.301; tail 4.117 | 기각 |
| Aux + p=0.25 locked ensemble | balanced component diversity | MAE 1.293, macro 1.235 | 기각 |
| Strict episode gate, class-balanced | base-independent gate와 causal state | validation safe-gate coverage 0, source macro 2.088 | test 전에 중단 |
| Strict episode gate, unbalanced | class weighting 부작용 제거 | validation safe-gate coverage 0, source macro 2.529 | test 전에 중단 |
| Low-capacity global episode | variance 감소 | validation source macro 2.487, action accuracy 0.751 | test 전에 중단 |

실패한 후보를 test leaderboard에 맞춰 섞지 않았다. 현재 선두보다 전체 평균이 나쁘지만 tail만 좋아진 후보도 고정된 전체·identity·catastrophic safety gate 때문에 승격하지 않았다.

## 5. Oracle과 실제 병목

Frozen base와 classical RR의 `×1, ×2, ×3, ×4` 5개 expert 중 target에 가장 가까운 값을 사후 선택하면 다음 상한이 나온다.

| 지표 | 현재 선두 | 5-expert oracle |
|---|---:|---:|
| Overall MAE | 1.291 | 0.689 |
| Identity-macro MAE | 1.220 | 0.674 |
| RMSE | 2.410 | 1.129 |
| ±2 bpm | 80.79% | 90.67% |
| >5 bpm | 6.23% | 0.21% |
| 25–35 bpm MAE | 4.216 | 1.224 |

Oracle은 6개 목표의 방향을 모두 만족하지만 배포 가능한 모델이 아니다. Optimistic in-sample factor router는 action AUC 0.894, reliable-row factor accuracy 93.36%를 보였어도, identity-disjoint outer-train에서 잠근 precision/safety 조건을 만족하는 threshold는 한 fold도 없었다. 즉 현재 병목은 expert 부재가 아니라 **새 identity에서 correction 필요 여부와 ×2/×3/×4 크기를 안전하게 결정하는 것**이다.

이 결과가 뜻하는 우선순위는 명확하다.

- 같은 18명을 반복 튜닝해 router threshold를 찾는 것은 중단한다.
- high-RR, motion, placement 변화가 충분한 새로운 identity를 수집한다.
- hardware-synchronized reference로 subharmonic label과 alignment를 다시 확인한다.
- calibration cohort에서만 router/abstention을 잠그고, confirmation cohort는 한 번만 연다.

## 6. 강건성·중복·선택적 예측

### 6.1 Radar 결측

| 사용 radar | MAE | Macro MAE | RMSE |
|---|---:|---:|---:|
| 1+2+3 | 1.291 | 1.220 | 2.410 |
| 1+2 | 1.486 | 1.417 | 2.755 |
| 1+3 | 1.445 | 1.374 | 2.723 |
| 2+3 | 1.422 | 1.338 | 2.709 |
| 1 only | 1.636 | 1.575 | 2.895 |
| 2 only | 1.554 | 1.478 | 2.882 |
| 3 only | 1.586 | 1.532 | 2.995 |

일곱 mask 모두 exact OOF population/fold binding 감사에는 통과했지만, single-radar 성능은 상용 정확도 목표를 만족하지 않는다. 또한 이는 완전 view masking이며 packet loss, partial corruption, range shift, displacement와 desynchronization을 대체하지 않는다.

### 6.2 중첩 효과

Greedy non-overlap subset은 `n=444`, MAE 1.570, macro MAE 1.440, RMSE 2.860, >5 bpm 8.78%다. 8개 fixed stride phase의 MAE 범위는 1.098–1.441 bpm이다. 어떤 phase도 전체 gate를 통과하지 않으며, test 결과를 보고 최선 phase를 선택하지 않았다.

### 6.3 Selective prediction

OOF uncertainty를 사후 quantile로 자른 70% retention은 MAE 0.508, macro 0.502다. 그러나 25–35 bpm retention은 45.1%이고, 전체 9,576개 후보 대비 reference-valid·retained 교집합은 약 17%뿐이다. Threshold가 독립 calibration cohort에서 잠기지 않았으므로 이 결과는 ranking diagnostic이며 제품 coverage 주장이 아니다.

## 7. 구현·무결성 작업

이번 루프에서 다음 코드를 추가하거나 강화했다.

- raw-window randomized-SVD 10개 representation과 valid/all-window cache builder
- source SNN, temporal SNN, causal episode SNN 학습·평가 파이프라인
- source/divisor/residual/quality/gate multi-task output과 base-independent strict gate
- no-radar → exact base, no-base → source의 구조적 fallback
- validation-only threshold/pull grid와 macro/tail/±2/catastrophic safety guard
- train-only weak invalid label quality gate와 divisor class-balance ablation
- epoch-seeded deterministic shuffle, RNG/checkpoint/split/source binding
- completed-resume one-shot test lock, prediction SHA와 exact test-position binding
- all-window identity-disjoint prediction, per-fold atomic verified marker와 field allowlist
- cache/session/runtime-source/checkpoint/frozen-OOF SHA-256, row exact cover와 frozen parity
- target/QC를 forward interface에서 차단하는 adversarial tests

Episode pipeline의 독립 재감사에서 발견한 no-base sanitized placeholder, metadata/source binding, semantic row binding, completed checkpoint binding과 resume atomicity 문제는 adversarial regression test와 함께 fail-closed로 보강했다. 독립 fold assignment authority를 요구하고, 상용 실행의 기본 all-window source도 최신 `all_windows_cuda_v3`로 바꿨다. 기존 episode 실험 수치는 상한/실패 진단으로만 유지하며, 보강 코드가 과거 결과를 자동으로 상용 증거로 바꾸지는 않는다.

전체 Python source는 `compileall`을 통과했고, 최종 unit/adversarial suite는 **168개 전부 통과**했다. Episode 관련 adversarial test는 metadata input 변경, 동일-fold cache-index 의미 순열, base fold 교환, partial all-window의 허위 complete 표시, selected checkpoint 삭제/변조와 interrupted best/last commit을 모두 거부한다.

## 8. CUDA all-window 배포 아티팩트

최신 fail-closed artifact는 `all_windows_cuda_v3`다.

| 항목 | 값 |
|---|---|
| Format | v2 |
| Inference signature | `bfd310c58324d7f8e8f5ec34` |
| Rows | 9,576 exact cover |
| Valid-reference rows | 2,327 |
| Runtime | RTX 4070, CUDA + AMP |
| Frozen parity | strict, 6/6 verified fold commits |
| Raw fingerprints | requested=true, verified=true |
| Deployment freeze eligibility | true, blockers=[] |
| NPZ SHA-256 | `036e4c8586963de484735619d8a2a7685995e9989af8a06ab084ba52a854ea5c` |
| CSV SHA-256 | `a8e0308529c436cc39870e9353aa6048d8b421e080b2c732c8fe34b4d67f51e3` |

생성 직후 동일 명령의 `--reuse`가 output/fold marker/cache/checkpoint/source/frozen parity를 다시 검증했다. 별도 감사에서도 430개 무결성 검사가 통과했다. `deployment_freeze_eligible`은 strict CUDA parity뿐 아니라 raw-source fingerprint 검증까지 true일 때만 true가 되며, 두 정책/결과가 inference signature에 결합된다.

이 eligibility는 **해당 all-window inference artifact의 무결성 판정**이다. 모델의 상용 정확도 또는 의료 효용 판정이 아니다.

## 9. 지연시간과 calibration 상태

기존 raw-window batch-one benchmark에서 두 component의 개별 CPU p95는 51.67 ms와 63.09 ms, RTX 4070 p95는 30.93 ms와 29.97 ms다. 따로 측정한 p95의 보수적 합은 CPU 114.76 ms, CUDA 60.90 ms로 4초 stride 이내다.

다만 outlier repair가 선택한 32초 window 경계에서 상태를 초기화하므로 첫 4 frame은 full-recording cache preprocessing과 bit-exact하지 않을 수 있다. 따라서 timing budget은 통과하지만 production raw preprocessing parity는 미완료다.

Uncertainty는 `>2`/`>5 bpm` error ranking에는 유용하지만 calibrated interval이 아니다. 독립 calibration identity가 없는 상태에서 threshold나 interval width를 상용 confidence로 표시하지 않는다.

## 10. 현 시점의 차단 조건

| 차단 조건 | 왜 코드/동일 cohort만으로 해소할 수 없는가 | 해제 증거 |
|---|---|---|
| 25–35 bpm support 부족 | valid 164, non-overlap 42이고 correction factor가 identity-dependent | 새로운 identity의 의도적 high-RR episodes |
| Retrospective adaptation | architecture/router가 같은 outer 결과를 반복 관찰 | 미개봉 prospective confirmation cohort |
| 완전 nested base 부재 | episode comparator base가 outer-test identity를 학습에 포함했을 수 있음 | nested base OOF 또는 prospectively frozen base |
| 독립 reference 부재 | BIOPAC RSP 파생 RR와 residual alignment 오차를 완전히 배제 못함 | capnography/독립 adjudication + hardware sync |
| Calibration 부재 | 현재 uncertainty threshold는 독립 identity에서 동결되지 않음 | 별도 prospective calibration cohort |
| Production preprocessing parity | window-boundary repair state가 cache builder와 다름 | 동일 streaming state golden test |
| 실제 장애 stress 부재 | mask test는 packet/placement/hardware fault를 대체하지 않음 | device-level injected/observed fault campaign |

이 중 첫 네 가지가 없는 상태에서 같은 18명의 OOF를 더 튜닝해 1.0 bpm 아래로 만드는 것은 commercial validation이 아니라 leaderboard overfitting 위험을 키운다.

## 11. 다음 실행 protocol

### Stage A — prospective calibration/development cohort

- 개발 cohort와 겹치지 않는 identity를 모집한다.
- rest뿐 아니라 의도적 25–35 bpm, transition, motion, 자세, placement, single/pair radar dropout을 포함한다.
- radar와 capnography/reference를 공통 trigger 또는 측정 가능한 clock으로 동기화한다.
- reference QC/adjudication은 prediction blind로 수행한다.
- 현재 model/checkpoint/preprocessing hash를 데이터 개봉 전에 기록한다.
- Stage A에서만 nested base, divisor router, uncertainty/conformal threshold와 unavailable policy를 적합한다.

표본수는 겹친 window 수가 아니라 identity cluster와 독립 high-RR episode 수로 정한다. 개발 cohort의 ICC와 tail prevalence 범위를 사용한 사전 simulation에서 overall MAE, identity-macro MAE, >5% binomial-cluster CI, 25–35 bpm MAE의 판정 power를 동시에 만족하는 identity 수를 고정한다. 계산 결과가 나오기 전 임의의 작은 `N`을 상용 표본수로 선언하지 않는다.

### Stage B — prospective confirmation cohort

- Stage A와도 겹치지 않는 새로운 identity/device placement/site를 사용한다.
- preprocessing, model ensemble, router, calibration, exclusion, missing-data 처리와 통계 코드를 개봉 전에 preregister한다.
- 전체 결과를 한 번만 계산하고 6개 primary accuracy gate와 cluster CI, radar fault, calibration, latency gate를 동시에 판정한다.
- 실패 후 같은 cohort에서 재튜닝한 결과는 confirmation으로 재사용하지 않는다.

### Stage C — deployment candidate

- streaming preprocessing의 state serialization/restart/idempotence와 golden parity를 잠근다.
- 장시간 p50/p95/p99, peak RAM/VRAM, thermal behavior, acquisition/network latency를 목표 장치에서 잰다.
- shadow → canary → limited release 순서와 drift/unavailable/rollback alarm을 운영한다.
- 이 단계까지 예외가 없어야 `deployment candidate`로 상태를 올린다.

## 12. 최종 승격 규칙

내부 후보는 서로 다른 3개 seed 모두에서 6개 정확도 gate를 통과하고, identity별 gross harm와 radar-mask/non-overlap/calibration gate를 통과해야 한다. 외부 confirmation에서는 point estimate뿐 아니라 사전 등록한 identity-cluster CI 판정까지 만족해야 한다.

현재는 이 규칙에 따라 `research candidate`로 고정한다. 성능을 과장하지 않은 것이 실패가 아니라, 실제 상용 성능에 필요한 다음 데이터를 정확히 식별한 결과다.

## 13. 핵심 근거 경로

- 목표 명세: `artifacts/COMMERCIAL_SNN_GOAL_V2.md`
- 현재 정확도 선두: `artifacts/runs/ensemble_structured_exact/metrics.json`
- 선두 OOF: `artifacts/runs/ensemble_structured_exact/ensemble_oof.csv`
- 강건성: `artifacts/robustness/ensemble_structured_exact/report.json`
- 상용 gate 기계 판정: `artifacts/commercial_goal_report.json`
- SVD source full OOF: `artifacts/runs/svd_source_snn_v1/metrics.json`
- Nested router: `artifacts/discovery_svd_signal/nested_innercv_high_router_v2/metrics.json`
- Physics/oracle audit: `artifacts/discovery_physics_ridge/report.json`
- CUDA strict artifact: `artifacts/runs/final_alias_gate_s12_deterministic/all_windows_cuda_v3/provenance.json`
- Episode trainer: `scripts/train_svd_episode_snn.py`
- All-window predictor: `scripts/predict_all_windows.py`
