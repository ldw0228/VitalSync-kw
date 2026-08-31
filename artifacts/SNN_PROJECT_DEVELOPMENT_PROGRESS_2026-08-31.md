# SnnProject 개발 진행상황 — Goal v4 raw-authority checkpoint

- 기준 시각: 2026-08-31, Asia/Seoul
- 대상: 3대 XeThru UWB radar 기반 호흡수 추정 hybrid/SNN 연구 시스템
- 증거 수준: 18명 retrospective cohort
- 제품 판정: 상용·의료 제품 아님, 내부 상용 정확도 gate `0/6`

현재 실행 authority는 이 문서와
`artifacts/COMMERCIAL_SNN_GOAL_V4_CONTINUATION_2026-08-31.md`,
`artifacts/COMMERCIAL_SNN_GOAL_V3_2026-08-31.md`가 공동 최우선이다. Goal v4는 Goal
v3의 accuracy/prospective gate를 유지한 채 raw-authority 실행 단계를 세분화한다. Goal v2, 이전
execution plan/progress, 보존 release manifest와 `commercial_goal_report.json`은 역사적
증거이며 현재 training/evaluation/release authorization이 아니다.

## 1. 이번 실행의 목표

동일 물리적 사람을 train/validation/test에 중복시키지 않는 완전 nested
identity-disjoint 학습·평가를 유지하면서, 각 고정 seed가 아래 여섯 기준을
동시에 통과할 때까지 데이터·모델·평가 루프를 개선한다.

| 지표 | 고정 목표 | 현재 full OOF leader | 판정 |
|---|---:|---:|:---:|
| MAE | ≤ 1.0 bpm | 1.291 | FAIL |
| identity-macro MAE | ≤ 1.0 bpm | 1.220 | FAIL |
| RMSE | ≤ 1.8 bpm | 2.410 | FAIL |
| ±2 bpm | ≥ 90% | 80.79% | FAIL |
| 오차 >5 bpm | ≤ 3% | 6.23% | FAIL |
| 25–35 bpm MAE | ≤ 2.0 bpm | 4.216 | FAIL |

내부 기준을 통과해도 독립 prospective cohort, 독립 reference, target device,
fault campaign, calibration, shadow/canary, rollback 검증 전에는 상용 성능으로
표현하지 않는다.

## 2. 이번에 확정한 데이터 계약

### 2.1 Frozen cohort authority

`configs/acquisition_cohort_v1.yaml`을 새 cohort 권한으로 추가했다.

- source session 30개, usable 29개, usable physical identity 18명 고정
- `S24_KHJ`: 세 radar stream이 비어 unusable로 고정
- `S17_RJS`: physical identity `PJS`로 고정
- session 순서, usable 집합, identity 대응, semantic content hash 검증
- 누락·추가·중복·순서 변경·identity 위조가 있으면 full-cohort 판정 전 차단
- 명시적 `--subjects` 사용은 모든 ID를 열거해도 diagnostic subset으로 강등

### 2.2 Raw parser와 입력 graph

- raw 원본 read-only 유지
- XeThru record `740 bytes = 3×uint32 header + 182×float32 payload` 검증
- zero header와 `bin_count=182` 검증
- BIOPAC 1개, radar metadata 3개, 선택된 모든 radar chunk의 완전한 입력 graph 검증
- 경로·크기·SHA-256과 선택 session graph를 reconstruction consumer가 독립 재검증
- config, protocol, spreadsheet, cohort, dataset, raw source의 parse 전후·publish 전후
  binding으로 persistent drift 차단. Active swap-and-revert까지 막으려면 sealed/read-only
  runtime 또는 same-inode owned snapshot이 추가로 필요

### 2.3 Measured radar timing

- metadata v13 timestamp 사용
- 하나의 element-local integer-nanosecond 산술 정책 사용
- interval은 정확한 `[left, right)` 의미
- future anchor를 이용한 보간·trimming 제거
- timestamp plateau/reset 영향 구간은 제거하지 않고 structural mask로 보존
- 모든 retained frame을 교집합 이전, leading edge, output interval, trailing edge,
  교집합 이후 중 정확히 하나로 분류
- frame-accounting residual은 계산값으로 기록하며 현재 재구성에서 0 검증
- `S07_KDM`의 정확한 counter-reset warning만 config allowlist로 허용
- 임의·추가·누락·변형 parser warning은 measured-timing 부적격

### 2.4 Radar↔BIOPAC 동기화

- 모델: `t_rsp = offset + scale × t_radar`
- radar motion marker와 RSP marker의 monotonic matching
- residual, drift, confidence, ambiguity, marker span gate 적용
- receipt 내부 marker index/time, residual, mapping, raw/config hash의 상호 일관성 검증
- bound raw signal에서 marker/match/mapping을 독립 재계산하는 verifier는 미완료
- 승인되지 않은 proposal은 diagnostic으로만 사용
- outer-test target이나 `radar_observable`을 sync feature로 사용하지 않음
- 수동 승인 receipt를 생성·수정·우회하지 않음

## 3. Block과 window 처리

### 3.1 Protocol block

spreadsheet anchor, BIOPAC marker, 순서·예상 길이·gap을 이용하는 7-phase ordered
decoder를 유지한다. Stage는 annotation·평가용이며 추론 feature가 아니다.

- `core`: 승인된 stage 내부 window
- `transition`: 둘 이상의 stage 경계를 걸치는 window
- `review/uncertain`: stage 또는 sync 권한 미확정
- breath-hold: 유효 RR을 강제로 만들지 않고 별도 상태/coverage로 관리

현재 full reconstruction의 protocol status는 `review 26`, `uncertain 3`이며,
stage-metric eligible session은 0개다. 따라서 앉기·걷기·운동·숨참기 등 stage별
성능표는 아직 과학 결과로 산출할 수 없다.

### 3.2 Feature window

- measured timing 기반 10 Hz grid
- 32초, 320 sample window
- 4초 stride
- 3 radar × `73 × 182` range–frequency map
- auxiliary feature 1,205개
- BIOPAC support는 continuous half-open interval에 대해 양쪽 경계를 robust `ceil`로
  변환해 `start ≤ i/fs < end`를 정확히 보장
- missing/invalid cell은 별도 boolean mask가 권한이며 numeric zero로 availability를
  추론하지 않음
- invalid radar 입력과 해당 auxiliary cell은 scaling 이후에도 exact zero

## 4. 실제 재구축 결과

### 4.1 Full acquisition reconstruction

경로:
`artifacts/acquisition/reconstruction_v2_20260831_causal_bound_v4_diagnostic`

| 항목 | 결과 |
|---|---:|
| source / usable / identities | 30 / 29 / 18 |
| full-cohort traversal | 완료 |
| metadata warning policy eligible | 29 / 29 |
| measured timing eligible | 19 / 29 |
| measured timing ineligible | 10 / 29 |
| sync decision: manual review / rejected | 13 / 16 |
| sync authorized | 0 / 29 |
| strict-cache eligible | 0 / 29 |
| stage-metric eligible | 0 / 29 |
| scientific eligible | 0 / 29 |

Measured-timing 부적격 session:
`S02_RJS`, `S03_PSJ`, `S07_KDM`, `S11_SJE`, `S12_KDH`, `S14_MDO`,
`S17_RJS`, `S26_LDW`, `S28_KDH`, `S30_SJE`.

Root content SHA-256:
`3c6852aece4fa8bc667639d30504c743a2879d46e6da2c4ee075e43c1c044437`.

### 4.2 Diagnostic RF feature cache

경로:
`artifacts/cache/rf32s_acquisition_v2_20260831_causal_bound_v4r2_diagnostic`

| 항목 | 결과 |
|---|---:|
| mapping proposal이 존재한 sessions | 18 |
| physical identities | 12 |
| windows | 5,826 |
| maps | `[5826, 3, 73, 182]` |
| aux | `[5826, 1205]` |
| timing mask | `[5826, 3, 320]` |
| invalid timing cells / affected windows | 104 / 40 |
| transition windows | 1,826 |
| reference-valid windows | 0 |
| stage-metric eligible rows | 0 |

분류는 `acquisition_diagnostic`, scientific eligibility는 false다. Sync가 승인되지
않은 상태에서 계산한 RR은 학습 label로 승격하지 않으며 `reference_valid`를 전부
false로 고정했다. Strict loader와 scientific trainer가 이 cache를 실제로 거부하는
것도 확인했다.

Root content SHA-256:
`3456d7b88f03e3fbfdf53175019545aa77b9d7e9e20ee2ae34c35ea48de1a9f9`.

### 4.3 SVD smoke

경로:
`artifacts/cache/svd_acquisition_v4r2_smoke_s02_diagnostic`

- `S02_RJS` 316 windows
- 12 components, NFFT 4096, randomized iteration 2
- feature-value label input 없음
- explicit subject filter로 `diagnostic_subset`, scientific=false
- detrend·standardize·SVD를 완전한 32초 window에 fit하므로 window-end causal
- within-window prefix causal 또는 streaming-prefix 표현으로 주장하지 않음
- output inventory와 timing mask content hash 검증

Root content SHA-256:
`79879108fa1d6412e10073ba1800baae7b20f47b0e0e010a48a3e26d5114f93e`.

전체 SVD rebuild는 현재 0 valid reference라 과학 학습에 사용할 수 없어 실행하지
않았다. 한 session smoke로 producer·inventory·mask·causality contract만 검증했다.

## 5. 학습·평가 firewall

- split 단위: physical identity, 6-fold OOF
- 같은 사람의 반복 session은 같은 fold
- outer-test target으로 scaler, proposer, router, threshold, calibration, ensemble
  weight, checkpoint 선택 금지
- acquisition-v2 scientific SVD는 genuine base-OOF authority JSON 필수
- base CSV/NPZ hash·bytes, cache/reconstruction, six-fold identity ownership,
  checkpoint train/validation/test 분리, label-free forward를 재검증
- target-equal CSV/NPZ 위조와 train/test identity 겹침을 adversarial test로 차단
- scientific SVD와 명시적 historical-legacy 입력은 metadata, array, base OOF,
  checkpoint, fold/frozen/completion artifact를 same-inode stable byte로 읽어 private
  read-only snapshot에서만 소비. 복사 뒤 원본과 source/runtime binding을 다시 검사
- acquisition-v2 scientific feature는 authority가 결합한 base NPZ의
  `prediction_bpm`, `rr_std_bpm`만 허용. CSV의 임의 추가 column은 inference feature로
  승격 불가
- loader가 발급한 exact experiment/receipt object와 method identity만 training entry에서
  재사용 가능. 복제 dataclass, subclass, temporal completed-fold transplant는 거부
- identity split은 authoritative session metadata의 same-inode bytes, exact dtype/order,
  `(row_position, session, physical_identity, reference_valid)` hash와 loader-issued exact
  DataFrame/authority object를 결합. Clone, caller role swap, post-issuance mutation 거부
- non-v2 SVD는 기본 training-authorized가 아니며, 명시적
  `--historical-legacy-reproduction`과 최초 생성하는 고유 output root에서만
  `historical_noncommercial` 재현 가능
- `radar_observable`은 target-dependent이므로 추론 feature 금지
- diagnostic cache를 scientific mode로 넘기면 학습 시작 전에 실패

현재 새 scientific training을 시작하지 않은 이유는 계산 부족이 아니라 data
authority failure다. 이를 우회한 학습 수치는 유효한 개선으로 인정할 수 없다.

## 6. 테스트와 재현성

- 최종 Goal v4 generation: 1,945 collected
- full suite 연속 2회: 각각 1,941 passed, 4 skipped
- managed namespace에서 skip된 실제 bubblewrap 4개: capable host context에서 4/4 passed
- active V8R4 fixed collection: 739 node IDs, semantic SHA-256
  `b9b192c084d3f6d69094657bceb9047e368c12cac4e4420c2f75ef3c7fc39df4`
- changed Python 전체 `py_compile`, config YAML parse, `git diff --check` 통과
- import-time Torch thread mutation 제거
- SVD 1×1 projection의 fixed-order accumulation으로 prefix/thread 수치 불안정 제거
- 이전 중간 generation의 1,545/1,541×2와 backup-time 688/4/2 기록은 역사
  evidence로 보존하며 현재 최종 결과로 대체해 표현하지 않음

남은 warning은 Python의 multi-threaded `fork()` deprecation warning이며 test failure가
아니다.

## 7. 현재 종료 경계

이번 루프는 데이터를 임의로 학습 가능 상태로 꾸미지 않고 정확한 실패 지점까지
진행했다.

1. Full 30-session acquisition graph와 29 usable session 검증 완료
2. Causal measured-time resampling, block annotation, RF cache, SVD smoke 완료
3. Sync authorized `0/29`, measured timing eligible `19/29` 확인
4. Strict scientific cache 생성과 새 SNN training은 fail-closed
5. 기존 leader와 상용 gate 판정은 변화 없음, `0/6`

재개 조건은 실제 sync review/authorization과 10개 timing exception의 독립 해결,
그 증거로 다시 만든 full scientific cache다. 그 전에는 모델 구조를 반복 학습해도
정렬 오류를 성능으로 흡수할 가능성이 있어 목표 달성 증거가 되지 않는다. V8R4도
별도의 CONTEXT1 receipt/source snapshot/pretrain authorization 없이는 시작하지 않는다.
현 CONTEXT1 generation에는 독립적으로 관리되는 외부 issuer/runner trust root와
signature verifier가 없어 trio를 발급할 수 없다. Real-bubblewrap 통과, local/self-hash,
self-signature, constant monkeypatch만으로는 충분하지 않으며 새 governed source
generation이 필요하다.

실패한 partial cache와 이전 diagnostic artifact는 과학 기록으로 보존했다.

## 8. 현재 generation 재현 명령

```bash
.venv/bin/python scripts/reconstruct_acquisition.py \
  --output-dir \
    artifacts/acquisition/reconstruction_v2_20260831_causal_bound_v4_diagnostic \
  --force

.venv/bin/python scripts/build_features.py \
  --config configs/default.yaml \
  --acquisition-manifest \
    artifacts/acquisition/reconstruction_v2_20260831_causal_bound_v4_diagnostic/manifest.json \
  --acquisition-mode diagnostic \
  --cache-dir \
    artifacts/cache/rf32s_acquisition_v2_20260831_causal_bound_v4r2_diagnostic \
  --force

.venv/bin/python scripts/build_svd_features.py \
  --dataset-root HAI_EXPERIMENT \
  --canonical-cache \
    artifacts/cache/rf32s_acquisition_v2_20260831_causal_bound_v4r2_diagnostic \
  --output-dir artifacts/cache/svd_acquisition_v4r2_smoke_s02_diagnostic \
  --subjects S02_RJS --all-windows \
  --components 12 --nfft 4096 --n-iter 2 --workers 1 --force
```

이 명령은 diagnostic artifact를 재현한다. `--acquisition-mode strict`로 바꿔
우회 학습하는 명령이 아니며, 현재 reconstruction에서는 strict mode가 의도대로
실패한다.

## 9. Goal v3 전면 hardening generation

이번 generation은 새 OOF 수치를 만들지 않고, 잘못된 데이터나 권한으로 학습이
시작되는 경로와 V8R5의 알려진 구조 결함을 먼저 폐쇄했다.

### 9.1 학습·추론 entry 권한

- `train.py`, `train_svd_snn.py`: acquisition diagnostic cache는 inspection-only.
  seed, CUDA, scaler, model, output directory 생성 전에 실패한다.
- V3R1 factor router의 import 가능한 `train()`과 `predict_target_free()`도 admitted
  context, phase, capability, authorization, outer-fold binding을 각자 재검증한다.
- 완료 run/prediction 재사용도 current fold/seed/variant/release/source/cache/args,
  selected checkpoint/scaler/predict input과 exact 결합한다. Cache/proposer/checkpoint는
  검증한 동일 private bytes에서만 소비하고 checkpoint load는
  `BytesIO + weights_only=True`로 제한한다.
- best/validation/prediction publication과 completed-return 직전에 input·artifact binding을
  재검증한다. 실패 temp는 filename만으로 ownership을 가정해 삭제하지 않고 explicit
  quarantine가 끝날 때까지 fail-closed한다.
- CONTEXT1 validator는 정확한 test node inventory와
  `0 skip/xfail/xpass/deselect`, real-bubblewrap 4개 PASS도 검사하지만, 현 generation에는
  독립적으로 관리되는 외부 issuer/runner trust root와 signature verifier가 없다.
- 따라서 validator는 terminal fail-closed다. Bubblewrap만 통과하거나 local/self-hashed
  receipt를 만들어도 CONTEXT1 trio를 발급하지 않으며, 새 governed source generation과
  독립 감사가 필요하다. V8R5 training authorization도 없다.

### 9.2 Harmonic cache·scaler·proposer

- RF/SVD timing-valid mask를 source hash, schema, shape와 함께 결합하고 두 view의
  all-interval validity를 radar availability로 축약한다.
- unavailable radar feature는 cache 생성과 scaling 뒤 모두 exact `+0.0`이다.
- robust scaler는 structurally available cell만 feature별로 fit한다.
- arbitrary proposer NPZ가 nested label-free ownership을 입증할 수 없으므로 현 builder
  output은 모두 `trainable: false`다. acquisition cache는 `acquisition_diagnostic`,
  legacy source는 `retrospective_legacy_unverified_proposer`로 분류한다.
- harmonic trainer는 manifest content hash, classification, `trainable`을 RNG/output
  생성 전에 검증한다.
- harmonic cache는 format v2와 schema ID로 승격하고 exact per-feature availability,
  571-layout digest, layout source hash, forward output inventory를 결합한다.
- proposer/RF/SVD input은 load 전 binding과 publication 직전 재해시로 persistent
  drift를 차단하고, 실제 소비 bytes를 private snapshot으로 복사한 뒤 원본 namespace를
  재검증한다. Input mmap의 active swap-and-revert가 학습 payload를 바꾸는 경로는
  폐쇄했지만 Python import 이전에 실행 code를 바꿨다가 복원하는 공격은 별도 sealed
  launcher/runtime 없이는 폐쇄됐다고 주장하지 않는다.

### 9.3 Custom prediction·동기화·protocol

- custom all-window prediction은 timing mask를 history에 전달하고 unavailable 값을
  exact zero로 유지한다. Cache는 exact private snapshot에서 소비하고 checkpoint, run
  config, split, reconstruction, dataset, external authority, source tree와 output 경로는
  alias/disjoint를 검사한다. Private stable directory와 `O_TMPFILE`/fd publication,
  no-clobber atomic link 또는 controlled exchange, directory `fsync`, rollback, symlink와
  output-parent rename/rebind 방어를 적용한다.
- 실행 초기화 시 disk source binding과 publish 전 drift는 검사하지만 이미 import되어
  실제로 compile된 loader byte 전체를 증명하지는 못한다. Manifest도
  `binds_actual_loader_compiled_bytes: false`로 명시하며, externally owned read-only
  source snapshot을 fresh isolated child에서 import하는 launcher 전에는 production
  execution authority가 아니다.
- marker detector는 invalid resampling 구간과 smoothing support 경계를 marker로 쓰지
  못한다.
- sync ambiguity는 첫 후보가 아니라 모든 동등 품질 alternative를 검사한다.
- bool/NaN/Inf/complex/fractional counter 같은 축약 입력을 gate 값으로 받지 않는다.
- sync/protocol/cohort/workbook config는 hash와 parse를 분리하지 않고 same-inode snapshot
  bytes를 직접 파싱한다. Workbook session 번호는 exact positive Python `int`만 허용한다.
- protocol은 config와 동일한 7개 phase의 order/name, 유효한 timing, worst stage status,
  mean stage confidence를 강제한다. Stage-metric eligibility는 protocol auto와 alignment
  authority가 모두 있어야 한다.
- legacy manual approval 문자열은 새 과학 권한을 만들지 못한다.

### 9.4 AxisRiskRouterSNN V8R5

새 구조는 frozen 571-wide layout을 유지한 228,838-parameter 미측정 제안이다.

- evidence 값과 per-feature availability를 radar/ratio/branch/candidate 좌표와
  pooling 전에 joint encoding
- ratio→radar와 radar→ratio bidirectional axial attention
- 7-relation directed harmonic graph, 2 PLIF blocks, 8 simulation steps
- causal PLIF→ALIF temporal state와 session-boundary reset contract
- 분리된 anchor/candidate value·route-preference·risk heads의 RR, scale,
  expected-error, `P(error>2)`, `P(error>5)`
- soft-routing gradient path와 differentiable expected deployment cost 구현, inference만
  hard selection; gradient correctness는 synthetic 증거이며 실데이터 routing 개선 미측정
- tail probability 단조성, quality supervision, route-temperature checkpoint binding
- float32 masked normalization으로 FP16 autocast all-missing NaN 차단
- explicit classical-RR availability, no-source quality exact zero, classical-only
  fallback quality supervision
- canonical structural mask parity, padded exact-zero, finite/range/state validation
- invalid target를 float32 변환·산술 전에 안전 치환, valid target 6–45 bpm 강제
- format-v2 cache schema/content/6개 forward payload same-inode 검증
- checkpoint에 source/config/layout/spiking dependency와 runtime-structure canonical uint8
  receipt 저장. Fresh runtime attributes, live buffer, checkpoint receipt를 삼자 비교
- strict load 전 exact key/shape/dtype/layout/finite private preflight, `assign=False`, 실패 시
  live model 무변이·transactional rollback
- parameter/buffer/raw-head/graph/PLIF/ALIF/route/streaming state nonfinite를 source별 차단.
  learned source가 오염되면 finite classical fallback, 없으면 unavailable exact-zero와
  quality untrusted/exact-zero

74개 전용 CPU synthetic test는 shape, gradient, mask parity, permutation, streaming,
autocast, transactional checkpoint/cache round-trip과 NaN/Inf fail-closed를 검증한다.
417개 named parameter/buffer NaN/±Inf 주입에서 trusted deploy output·streaming state
nonfinite 위반은 0이었다. 실제 data authority와 CUDA가 없으므로
학습·성능 측정은 하지 않았고 정확도 leader와 상용 gate `0/6`은 그대로다.

### 9.5 평가·prospective 계약 hardening

- 여섯 accuracy threshold와 inclusive operator, 세 fixed seed, full denominator,
  fallback 포함, no-estimate fail-closed를 YAML에 기계 판독 가능하게 고정
- G1–G4 authority 재구축 뒤 authoritative row count/hash를 prediction 전에 동결.
  기존 2,327 row는 exact bijective crosswalk일 때만 legacy co-primary이며, 다르면
  historical comparator로만 유지
- Greedy non-overlap과 8 stride phase를 physical source session별로 선택하고,
  authoritative session-local `window_number`, selector source/config/selected-key hash를
  prediction 전에 동결
- 최소 identity/25–35 support와 별도 35–45 bpm safety gate를 정의. 지원량 미달은
  통과가 아니라 `insufficient_support`
- 사전 잠금 campaign당 outer target 공개 1회. 각 hypothesis iteration은
  inner/discovery validation만 사용하며 공개 후 같은 campaign에서 선택·재교정 금지
- Final model refit→weight seal→untouched prospective calibration→calibrator seal→untouched
  confirmation 순서 고정
- R1/R2/R3에 cohort floor, collection 전 identity-cluster power, one-sided bootstrap 95%
  accuracy bound, 6–45 support, exclusion/coverage/calibration, seven masks, fault,
  latency/memory, shadow/canary/rollback 수치를 사전 선언

이 기준은 아직 측정·통과하지 않은 내부 engineering release criteria다. 의료·임상·규제
보장이 아니다.

### 9.6 남은 실행 경계

1. bound raw signal에서 sync receipt를 독립 재계산하는 verifier
2. spreadsheet/RSP/config에서 seven-stage decoder를 재실행하는 authority
3. 새 governed CONTEXT source generation, 외부 issuer/runner trust root와 signature verifier
4. externally owned source snapshot을 fresh isolated child에서만 import하는 production
   launcher와 actual compiled-loader provenance
5. raw XeThru/BIOPAC manifest hash와 실제 장시간 memmap/load bytes를 동일 stable
   descriptor에 종단 결합하는 source-consumption runtime
6. measured timing 부적격 10 sessions의 독립 해결
7. 29-session strict RF/SVD cache와 versioned nested proposer/scaler authority
8. V8R5 전용 immutable authorization, GPU fixed-update discovery, 3 seeds × 6 folds OOF
9. 독립 prospective calibration/confirmation과 target-device fault/shadow 검증

현재 상태는 `R0_RESEARCH_ACTIVE`다. Source hardening과 resume-ready 설계는 완료했지만,
권한 부재를 우회해서 만든 성능 수치는 목표 진척으로 인정하지 않는다.

## 10. Goal v4 raw-authority continuation 결과

이 절은 앞의 acquisition-v2 수치와 9.6의 timing/cache 미완료 항목을 대체한다. 이전
artifact와 실패 결과는 역사 증거로 계속 보존한다.

### 10.1 Acquisition v3 full diagnostic reconstruction

경로:
`artifacts/acquisition/reconstruction_v3_20260831_raw_exact_timing_reason_diagnostic`

| 항목 | 결과 |
|---|---:|
| source / usable / identities | 30 / 29 / 18 |
| exact raw consumed-byte receipt | 29 / 29 usable |
| XeThru metadata↔chunk exact join | 90 / 90 metadata |
| BIOPAC parser contract | 30 / 30 MAT |
| measured timing adjudicated | 29 / 29 usable |
| 원래 timing invalid cells | 31 |
| invalid 이유 | empty 3, temporal gap 10, plateau 22 |
| sync decision | manual review 13, rejected 16 |
| sync / stage / scientific authority | 0 / 29 |
| protocol status | review 26, uncertain 3 |

Invalid reason은 중복 가능해 reason 합계가 31보다 크다. 31개 cell을 보간·삭제하지 않고
reason mask와 exact positive zero로 유지했다. Root content SHA-256은
`295a93ca854c07c5ab23067d6a5e401b503b048e2a710352ff6e37c71abf6f2f`다.

Raw reader는 descriptor-pinned one-shot hash+decode, XeThru record/footer geometry,
BIOPAC ISI unit·channel identity, portable/full receipt cross-link를 검증한다.
`sync_signals.npz`도 파일 SHA/bytes와 7개 배열의 dtype·shape·canonical hash로 결합했다.
실행 source hash는 post-import pathname snapshot일 뿐 실제 compiled loader bytes를
증명하지 않음을 manifest에 명시했다.

### 10.2 Feature-cache v3 full diagnostic materialization

성공 경로:
`artifacts/cache/rf32s_acquisition_v3_20260831_diagnostic_a3`

| 항목 | 결과 |
|---|---:|
| sessions / identities | 29 / 18 |
| windows | 9,575 |
| maps / aux | `[9575,3,73,182]` / `[9575,1205]` |
| timing valid/reason mask | `[9575,3,320]` / `[9575,3,320]` |
| feature availability | `[9575,1208]` |
| mapping 있음 / 없음 | 18 sessions·5,830 rows / 11 sessions·3,745 rows |
| window-expanded invalid cells | 208 |
| unavailable radar views | 128 |
| reference-valid rows | 0 |

Mapping이 없는 11 sessions는 BIOPAC·reference·stage를 사용하지 않고 measured
radar-relative 32초 support만 생성한다. Reference 관련 값은 명시적
NaN/-1/null/false sentinel이고 `reference_mapping_available=false`다. Mapping이 있는
18 sessions도 승인되지 않았으므로 target은 전부 invalid다. Missing view/aux는 별도
availability가 권한이며 값은 exact positive zero다.

Loader 재검증 결과: 29 sessions, owned/read-only arrays, inventory 233 files,
classification `acquisition_diagnostic`, scientific false. Scientific load 요청은
`version-3 cache lacks independently verified upstream scientific authority`로 차단됐다.
Root content SHA-256은
`f0bf626e4177b34930317658f2deee67fbab793c9dfe19b7be11e75b8a4acfeb`다.

실패 기록도 유지한다.

- 첫 시도: 실제 4-D map을 가상 5-D branch로 잘못 가정해 S02에서 중단
- 두 번째 시도: mapping 없는 11 sessions를 skip해 exact 29-session gate에서 중단,
  private failure receipt와 staging 보존
- 세 번째 시도: 두 결함 수정 후 private 0700 staging, files 0600, fsync,
  `RENAME_NOREPLACE`로 완전 게시

### 10.3 SVD v3 full diagnostic materialization

정식 diagnostic 경로:
`artifacts/cache/svd_components_v3_20260831_diagnostic_c12_n4096_a1`

| 항목 | 결과 |
|---|---:|
| sessions / rows | 29 / 9,575 |
| components / NFFT / iteration | 12 / 4096 / 2 |
| variants | 10 |
| mapped / unmapped sessions | 18 / 11 |
| unavailable views / invalid intervals | 128 / 208 |
| feature payload bytes | 4,332,894,188 |
| scientific / training authority | false / false |

SVD는 RawSessionReader로 원자료를 다시 one-shot 소비하고 acquisition receipt,
measured resampling, valid/reason mask, radar-relative 320-sample support를 RF cache와 exact
join했다. `--all-windows`를 강제해 target-dependent row selection을 금지한다. Unavailable
view의 spectra/component/attribute는 모두 exact positive zero다. 29개 session manifest가
현재 file inventory와 일치한다. Root content SHA-256은
`c6f979f8438be4959b2bb3effa2b22e8ac17fd87e2f9210cc8954142e3220eec`다.

저비용 전수 smoke도 `components=1, NFFT=512, iteration=0`으로 29/9,575를 먼저
통과했으며 root content SHA-256은
`cb49d316be005eee154b37f44dc45b18dfa4813113162ec408da70c34c6cab8f`다.

### 10.4 현재 정확한 resume boundary

- 완료: C0 raw bytes, C1 timing adjudication, C4 diagnostic RF/SVD
- 미완료: 독립 raw-derived sync verifier, protocol replay trust root,
  fresh-child actual compiled-source binding
- 실제 결과: sync authorized 0/29, stage authority 0/29, scientific/training 0/29
- 실행 환경: PyTorch 2.13.0+cu130, CUDA device unavailable
- 결과: V8R5 protected training과 새 OOF 미실행, 기존 leader와 상용 gate `0/6` 유지

다음 합법적 실행은 threshold 완화나 self-signed receipt가 아니다. 독립 trust root가
있는 sync/protocol/source verifier로 C2–C3를 통과하고, 승인된 alignment에서 label을
새로 구성한 뒤 strict cache를 별도 generation으로 발급하는 순서다.
