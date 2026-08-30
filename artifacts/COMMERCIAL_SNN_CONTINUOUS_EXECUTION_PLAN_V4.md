# 상용 수준 SNN 연속 실행 계획 v4

상태: **집행 중인 단일 종단 계획**  
기준 시각: **2026-08-28, Asia/Seoul**  
프로젝트: **3-radar UWB respiration-rate estimation**  
현재 모델 계열: **Harmonic Candidate-Set Episode SNN (HCS-E-SNN)**  

이 문서는 아이디어 목록이나 후속 작업 메모가 아니다. 현재 실행 중인 retrospective 학습부터 고정 평가, 실패 원인에 따른 다음 구조의 재설계, 독립 prospective 확증, 배포 검증까지 하나의 재개 가능한 상태 기계로 정의한다. 프로세스가 중단되어도 완료된 원자 단위를 재사용하고 같은 sealed command를 재실행한다. 정확도 미달은 프로그램 종료가 아니라 다음 설계 주기의 입력이다.

## 1. 종료 조건과 주장 경계

### 1.1 내부 engineering success

각 고정 seed `20260828`, `20260829`, `20260830`가 full valid-reference, identity-disjoint OOF에서 아래 여섯 조건을 **모두** 만족해야 한다.

| 지표 | 고정 게이트 |
|---|---:|
| overall MAE | `<= 1.000 bpm` |
| identity-macro MAE | `<= 1.000 bpm` |
| RMSE | `<= 1.800 bpm` |
| absolute error `<=2 bpm` | `>= 90.0%` |
| absolute error `>5 bpm` | `<= 3.0%` |
| reference 25–35 bpm MAE | `<= 2.000 bpm` |

추가로 full coverage, 7개 radar availability mask, non-overlap windows, temporal phase, calibration, streaming parity, CPU/CUDA latency, finite-output, provenance gate를 통과해야 한다. 평균 seed나 최상 seed로 실패 seed를 숨기지 않는다.

### 1.2 commercial release success

내부 engineering success만으로 상용 성능을 주장하지 않는다. 상용 release는 다음을 모두 만족할 때만 `true`가 된다.

1. 모델·decoder·threshold·calibration·평가 코드가 prospective target 공개 전에 고정됨.
2. 기존 18 identity와 겹치지 않는 prospective cohort가 있음.
3. 사전 등록된 prospective point gate와 identity-cluster confidence-bound gate가 모두 통과함.
4. target device golden replay, streaming parity, latency, memory, fault injection, shadow/canary, rollback 검증이 완료됨.
5. release bundle의 source/config/checkpoint/data-manifest/SBOM hash와 서명이 일치함.

prospective cohort가 없으면 최종 상태는 `INTERNAL_ENGINEERING_PASS / COMMERCIAL_CONFIRMATION_BLOCKED`이며, 이것은 모델 실패나 작업 포기가 아니라 외부 데이터 의존 blocker다.

## 2. 절대 불변 조건

- 물리 identity는 train/validation/test 경계를 넘지 않는다.
- 중첩 raw interval과 같은 session의 파생 row는 split 경계를 넘지 않는다.
- proposer, teacher, scaler, router, calibrator는 해당 prediction identity를 학습에 보지 않는다.
- target RR, reference validity/quality, target-derived action, future window는 forward input에 들어가지 않는다.
- valid-reference row를 inference failure, mask, protocol, high error 때문에 제거하지 않는다.
- outer-test 결과를 본 뒤 동일 cycle의 checkpoint, seed, threshold 또는 subset을 바꾸지 않는다.
- adaptive 재설계 결과는 모두 `retrospective_adaptive`로 표시하고 append-only adaptation ledger에 남긴다.
- oracle은 후보 bank capacity 진단에만 쓰며 deployable 성능으로 보고하지 않는다.
- target은 모든 target-free prediction, seal, spec, calibration, benchmark가 완료된 뒤 release-lock wrapper를 통해 한 번만 연다.
- GPU job은 admission lock을 통해 한 번에 하나만 실행한다.
- runtime seal에 포함된 파일은 campaign이 끝날 때까지 수정하지 않는다.
- 결과 실패, process kill, shell disconnect는 완료로 간주하지 않는다.

## 3. 재개 가능한 실행 상태 기계

모든 장기 campaign은 `plan.json`, `status.json`, immutable manifest, runtime-input seal, unit-level committed artifact를 가진다. unit은 `(seed, outer_fold, inner_prediction_fold)` 또는 명시된 동등 원자 키다.

1. 실행 전 runtime seal을 재해시한다.
2. GPU lock을 획득하고 ledger에 command, PID, start time, seal hash를 기록한다.
3. 아직 committed되지 않은 원자 unit 하나만 실행한다.
4. output schema, row identity, checkpoint/config/source hash를 검증한다.
5. 임시 artifact를 최종 경로로 원자 이동하고 status를 갱신한다.
6. runtime seal을 다시 재해시한다.
7. lock을 해제하고 다음 unit으로 반복한다.

중단 후에는 status와 실제 committed artifact의 exact-cover를 대조한다. 부분 unit은 폐기하고 같은 unit을 다시 실행한다. 이미 검증된 unit은 재학습하지 않는다. `failed_unit`이 있으면 원인과 stderr hash를 보존한 뒤, source 변경이 필요하지 않은 환경 장애는 같은 sealed unit으로 재시도한다. source 변경이 필요한 결함은 기존 campaign을 닫고 새 contract/hash 아래에서만 재개한다.

## 4. 현재 HCS-E-SNN v2 고정 실행 DAG

아래 순서는 변경하지 않는다. 오른쪽 artifact가 완전하고 검증되기 전에는 다음 노드를 열지 않는다.

### A. 완전 nested proposer 생성

1. 6 outer folds × 3 seeds의 non-test inner OOF proposer campaign 90 unit을 완료한다.
2. 기존 source drift 영향을 받은 outer fold 3/4만 현재 sealed source로 30 unit 재학습한다.
3. retrain supervisor가 각 unit 전후 runtime seal을 확인하고 30/30 completion attestation을 만든다.
4. impact audit가 이전/재학습 unit의 key coverage, source hash, prediction schema를 비교한다.
5. merge가 fold 0/1/2/5의 원본과 fold 3/4의 재학습 결과를 exact-cover로 합친다.
6. merge artifact는 retrain execution attestation과 runtime seal을 필수 provenance로 포함한다.
7. governance attestation은 outer-test unopened, target-free, identity-disjoint, no duplicate/missing unit을 검증한다.

### B. calibration과 fixed pre-test runtime

8. merged non-test prediction만 사용하여 outer/seed별 5-way cross-fitted normalized conformal calibration을 만든다.
9. uncertainty quantile `50/80/90/95%`와 selective thresholds `50/80/90/100%`를 고정한다.
10. fixed-i3 runtime seal 생성 시 proposer index를 반드시 merged index로 명시한다.
11. dry-run에서 18개 `(seed, outer_fold)` command와 retrain fold binding을 검증한다.
12. fixed-i3 pre-test 18 unit을 GPU lock 아래 완료한다.
13. completion sealer가 runtime seal, fixed plan, 18 committed outputs, source/checkpoint hashes를 묶어 `fixed_runtime_completion_attestation.json`을 만든다.

### C. target-free inference와 deployment closure

14. full outer-test manifests를 만들되 prediction 파일에는 target/reference field를 쓰지 않는다.
15. runtime-sealed OOF supervisor가 18 prefix를 한 unit씩 실행하며 unit 전후 runtime tree를 재검증한다.
16. 18/18 exact-cover 후 `postlock_runtime_guard_attestation.json`을 만든다.
17. 7개 non-empty radar mask × 18 prefix = 126 unit을 같은 방식으로 실행한다.
18. full-radar mask output이 canonical OOF output과 bit-exact인지 확인하고 `radar_mask_runtime_guard_attestation.json`을 만든다.
19. canonical 18 outputs의 point prediction, fallback decision, uncertainty raw fields를 target 없이 seal한다.
20. offline batch와 chronological streaming replay의 prediction/state parity를 18 prefix 전부에서 검증한다.
21. proposer와 HCS 전체 경로를 CPU에서 측정하고, CUDA가 있으면 별도 output root에 동일 benchmark를 실행한다.
22. cold/warm p50/p95/p99 latency, peak memory, parameter count, spike-rate, nonfinite count를 기록한다.

### D. 평가 프로토콜 동결과 target release

23. primary evaluation spec은 이미 고정된 여섯 accuracy gate와 row universe를 유지한다.
24. primary spec의 uncertainty diagnostic-only 의미를 바꾸지 않는 별도 uncertainty evaluation spec을 0444로 고정한다.
25. release-readiness freeze spec을 target 전에 고정한다. prospective evidence 없이는 commercial result가 `false`가 되도록 한다.
26. pretarget release lock은 다음을 모두 재검증한다: primary 18, mask 126, full-mask parity, uncertainty 18, calibration, primary/uncertainty/deployment/readiness specs, fixed completion, OOF guard, mask guard, runtime tree.
27. target builder의 직접 호출은 capability token 없이는 target 파일을 열기 전에 실패해야 한다.
28. release-lock wrapper만 target authority를 열고 exact ordered target artifact와 access receipt를 만든다.

### E. one-shot 평가

29. target와 canonical point predictions를 key로 exact join한다.
30. seed별/전체/identity/fold/protocol/RR-stratum/phase/non-overlap primary metrics를 계산한다.
31. 7 radar masks에 대해 동일 row universe와 full-mask parity를 확인하고 degradation/availability gate를 계산한다.
32. 고정 conformal interval coverage/width와 selective risk-coverage를 계산한다.
33. identity-cluster bootstrap을 고정 seed와 반복 수로 실행한다.
34. readiness aggregator가 accuracy, robustness, calibration, parity, latency, provenance를 한 보고서에 모은다.
35. 실패 gate를 포함한 모든 결과를 append-only release report에 기록한다.

## 5. v2 실패 시 자동 재설계 루프

v2가 한 gate라도 실패하면 결과를 숨기지 않고 `retrospective_adaptive_failure`로 봉인한다. 그 뒤 아래 한 cycle을 순서대로 수행한다. cycle 번호가 바뀔 때마다 새 contract, config, source closure, manifest root, output root를 사용한다.

### 5.1 오류 분해

각 valid row를 다음 상호배타적 우선순위로 분류한다.

1. candidate coverage failure: 어떤 label-free 후보도 target에서 2 bpm 이내가 아님.
2. routing failure: 좋은 후보가 있으나 잘못된 harmonic factor/node를 선택함.
3. residual failure: 올바른 후보를 선택했지만 residual이 악화시킴.
4. fallback failure: learned path가 거부되어 base tail error가 남음.
5. over-correction: base가 정확했는데 learned path가 2 bpm 이상 악화시킴.
6. availability failure: radar mask 또는 invalid evidence로 finite output을 만들지 못함.
7. calibration failure: point error 대비 interval/quality ordering이 맞지 않음.

분류는 전체, identity, fold, 6–12/12–25/25–35/35–45 bpm, radar mask, protocol, temporal phase별로 집계한다. candidate oracle은 coverage ceiling을 정량화하는 데만 쓴다.

### 5.2 단일 원인-단일 개입 규칙

한 cycle에는 지배적 실패 원인 하나와 그에 대응하는 구조 개입 하나만 허용한다. 예시는 다음과 같다.

| 지배 원인 | 허용 개입 |
|---|---|
| candidate coverage | label-free spectral/harmonic 후보 생성 개선 |
| routing | factor-first expert SNN, pairwise ranking 또는 calibrated abstention |
| residual | residual head 제거/범위 축소/robust loss |
| fallback | quality head와 sparse correction policy 재설계 |
| over-correction | base-preservation loss와 conservative action gate |
| mask degradation | radar-set equivariant fusion와 mask dropout |
| temporal instability | causal episode state/reset/TBPTT 수정 |
| calibration | held-identity cross-fit nonconformity score 수정 |

여러 architecture family를 동시에 경주시키거나 best seed만 고르지 않는다. 변경 가설, 허용 feature, parameter/compute budget, promotion gate를 target 공개 전에 새 contract에 기록한다.

### 5.3 v3 첫 개입

v2의 예상 병목은 좋은 후보가 존재하지만 harmonic node를 고르지 못하는 routing이다. v2 실패가 실제로 이 분해와 일치할 때 첫 개입은 `Directed Harmonic Factor Expert SNN v3`로 고정한다.

- 먼저 `×1/×2/×3/×4` factor를 고르는 작은 causal SNN을 학습한다.
- 각 factor expert는 label-free multi-radar harmonic evidence만 받는다.
- factor posterior와 expert residual을 분리하여 mode averaging을 금지한다.
- base-preservation/over-correction loss와 target-free quality abstention을 포함한다.
- v2와 같은 완전 nested identity-disjoint proposer/teacher/scaler 규칙을 적용한다.
- 기존 cohort에서 얻는 모든 v3 수치는 adaptive retrospective로 표시한다.

v3 contract와 구현의 hash가 고정된 뒤 discovery 18 unit, 공통 promotion rule 동결, full nested 18 unit 순으로 실행한다. 실패하면 5.1로 돌아가되 새로운 증거가 지지하는 한 개입만 다음 cycle에 허용한다.

### 5.4 cycle 승격 규칙

각 cycle은 다음 조건을 모두 만족해야 full retrospective evaluation으로 승격된다.

- discovery partition 양쪽에서 같은 policy가 선택됨.
- 후보 coverage가 이전 cycle보다 악화되지 않음.
- base보다 overall MAE/macro MAE/tail MAE가 모두 개선됨.
- over-correction rate와 catastrophic error가 고정 safety bound 이내임.
- 세 seed 모두 같은 방향의 개선을 보임.
- source/config/manifest/runtime seal과 결과 ledger가 완전함.

승격 실패는 구조 피드백이며 프로그램 종료가 아니다. 다만 label 없는 정보 자체가 목표를 식별할 수 없다는 ceiling이 반복 확인되면 필요한 새 sensor/reference/identity cohort를 구체적인 외부 blocker로 기록한다.

## 6. 데이터 확장과 prospective 확증

내부 gate를 통과한 최초 cycle의 모든 adaptive 요소를 동결한다. 이후 새 cohort에 대해 다음 순서를 지킨다.

1. 기존 18명과 physical identity 중복을 검사한다.
2. site/device/placement/posture/motion/RR support를 사전 sample-size 표에 맞춘다.
3. target를 보지 않고 raw ingestion, synchronization, causal cache, prediction, mask/fault campaign을 완료한다.
4. prediction hash와 evaluation spec을 봉인한다.
5. target를 한 번 열어 full-coverage point/CI/subgroup metrics를 계산한다.
6. point gate뿐 아니라 identity-cluster CI의 불리한 경계가 사전 한계 안에 있어야 통과한다.
7. 실패하면 해당 모델의 상용 승격은 거부한다. 새 cohort 결과를 본 뒤 바꾼 모델은 새 prospective cohort가 필요하다.

새 prospective 데이터가 아직 없으면 acquisition manifest, required strata, minimum identities, collection/QC protocol, exact resume command를 보존한다.

## 7. 배포 검증과 운영 gate

정확도 확증 후에도 다음을 모두 통과해야 한다.

- offline/streaming golden replay bit parity 또는 사전 허용 수치 오차.
- packet duplicate, reorder, drop, timestamp gap, session restart에서 state 안전성.
- radar 1/2/3개 availability와 모든 non-empty mask에서 finite deterministic output.
- CPU target 및 CUDA target의 warm p99 latency와 peak memory 게이트.
- spike-rate operating band, parameter cap, NaN/Inf zero count.
- signed model bundle, dependency lock, SBOM, schema/version compatibility.
- shadow deployment에서 reference-free drift/availability/latency monitoring.
- canary 중 safety threshold 위반 시 자동 rollback.
- audit log에 input schema, model hash, calibration hash, fallback reason 기록.

## 8. 완성 판정

작업은 아래 세 상태 중 하나로만 귀결된다.

1. `COMMERCIAL_RELEASE_READY`: 내부, prospective, 장치, 운영 gate 전부 통과.
2. `INTERNAL_ENGINEERING_PASS / COMMERCIAL_CONFIRMATION_BLOCKED`: 내부·장치 gate는 통과했으나 독립 prospective cohort 또는 배포 권한이 없음.
3. `ACTIVE_RESEARCH_CYCLE`: 내부 gate 미달이며 5절의 오류 분해와 단일 개입 loop가 진행 중.

process 종료, 계산 시간, 한 번의 architecture 실패, 테스트 실패는 위 세 상태를 임의로 바꾸지 않는다. 현재 active goal은 상태 artifact와 sealed campaign으로 계속 이어지며, 진짜 외부 의존 blocker만 정확한 재개 지점과 함께 보고한다.
