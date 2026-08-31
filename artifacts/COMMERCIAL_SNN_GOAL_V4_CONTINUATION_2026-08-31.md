# SnnProject Goal v4 — raw-authority continuation

- 기준 commit: `b273fc6eca00dd2e8768bb4d993f1f958ffda48c`
- 시작 상태: `R0_RESEARCH_ACTIVE`
- 정확도 상태: 내부 상용 gate `0/6`
- 모델 상태: `AxisRiskRouterSNNV8R5` 구현·미측정·학습 비허가
- 데이터 상태: usable 29 sessions / 18 physical identities
- 현재 권한: raw-byte/timing diagnostic 29/29, sync 0/29, stage 0/29,
  scientific/training 0/29
- 절대 경계: 현재 retrospective 결과를 상용·의료 성능으로 표현하지 않음

이 문서는 Goal v3의 정확도·prospective 기준을 바꾸지 않는다. Goal v3를 실행하기
위해 남아 있는 raw-data authority, synchronization, stage replay, strict-cache 단계를
구현 가능한 작업 단위와 승격 gate로 세분화한다. 과거 artifact를 재해석하거나
덮어쓰지 않고 모든 결과를 새 schema와 versioned output root에 기록한다.

## 0. 2026-08-31 실행 checkpoint

| 단계 | 실제 결과 | 승격 상태 |
|---|---|---|
| C0 raw consumed bytes | usable 29/29 exact one-shot receipt | diagnostic PASS |
| C1 timing adjudication | 29/29, 원래 invalid cell 31개 mask 유지 | diagnostic PASS |
| C2 sync raw replay | manual review 13, rejected 16, authorized 0 | FAIL-CLOSED |
| C3 protocol replay | review 26, uncertain 3, stage authority 0 | FAIL-CLOSED |
| C4 RF cache | 29 sessions, 9,575 windows, reference-valid 0 | diagnostic PASS |
| C4 SVD cache | 29 sessions, 9,575 windows, 12 components | diagnostic PASS |
| C5–C6 training/OOF | sync/source authorization와 CUDA 없음 | NOT STARTED |

C0–C4의 `PASS`는 구조·바이트·mask 재현성만 뜻한다. 독립 sync/protocol verifier,
fresh-child executed-source binding, 승인된 label alignment가 없으므로 scientific 또는
training authority를 발급하지 않는다. 현재 leader와 내부 상용 gate `0/6`은 그대로다.

## 1. 최종 목표

세 고정 seed `20260828/20260829/20260830` 각각이 identity-disjoint full OOF에서 아래
여섯 조건을 모두 만족하는 하나의 공통 architecture/release mode를 만든다.

| metric | inclusive gate |
|---|---:|
| overall MAE | `≤1.0 bpm` |
| identity-macro MAE | `≤1.0 bpm` |
| RMSE | `≤1.8 bpm` |
| absolute error `≤2 bpm` | `≥90%` |
| absolute error `>5 bpm` | `≤3%` |
| target 25–35 bpm MAE | `≤2.0 bpm` |

모든 reference-valid row를 분모에 포함하고 fallback도 일반 prediction처럼 평가한다.
No-estimate, NaN/Inf, 누락, 중복, 예상 밖 row가 하나라도 있으면 해당 seed/population은
fail-closed다. Outer target는 사전 잠금 campaign당 한 번만 공개한다.

## 2. 실행 상태기계

```text
C0_RAW_CONSUMED_BYTES
  ↓ exact same descriptor bytes + record evidence
C1_TIMING_ADJUDICATED
  ↓ every invalid cell explained and structurally masked
C2_SYNC_RAW_REPLAYED
  ↓ raw-derived automatic decision; review/reject stays closed
C3_PROTOCOL_RAW_REPLAYED
  ↓ phase-local replay authority; non-auto phase excluded
C4_STRICT_CACHE_AUTHORIZED
  ↓ 29-session RF/SVD + nested proposer/scaler authority
C5_V8R5_DISCOVERY_AUTHORIZED
  ↓ inner/discovery fixed-update iterations only
C6_COMMON_RELEASE_LOCKED
  ↓ 6 folds × 3 seeds, one sealed outer reveal
R1_INTERNAL_ENGINEERING_PASS
```

`R1` 이후 prospective calibration/confirmation, target-device fault/latency,
shadow/canary/rollback 기준은 Goal v3 G10을 그대로 적용한다.

## 3. C0 — raw consumed-byte authority

### 문제

현재 path hash → pathname reopen/load → path rehash 순서는 A→B→A 교체를 탐지하지
못한다. Receipt가 A를 가리키면서 실제 transform이 B를 소비할 수 있다. 현재 scientific
authority가 0이라 성능 증거 우회는 발생하지 않았지만, 새 verifier/cache 전 반드시
폐쇄해야 한다.

### 구현 계약

- dataset root와 session directory를 directory descriptor로 pin
- exact graph path만 `O_RDONLY|O_CLOEXEC|O_NOFOLLOW`로 open
- regular file, `nlink=1`, before/open/after inode·stat exact equality
- radar chunk를 record-aligned block으로 한 번만 순차 소비
- 같은 block에서 SHA-256, zero/header/bin/frame sequence, payload를 동시에 도출
- metadata는 동일 immutable bytes에서 parse·hash
- BIOPAC은 동일 bytes를 `BytesIO`로 `loadmat`
- filename/timezone, chunk order, footer inventory를 receipt에 결합
- downstream에는 owned arrays만 전달; live raw mmap 금지
- raw file/tree 수정 및 3.6 GB 사본 생성 금지

### 승격 gate

- manifest binding과 consumed binding exact equality
- bins/frame sequence/record QC가 동일 byte generation에서 도출됨
- symlink, hardlink, rename/rebind, short read, truncate, append, chunk reorder 실패
- A→B→A adversarial test 실패
- unchanged fixture의 numerical output/hash parity

## 4. C1 — timing adjudication

원자료 감사 결과, 부적격 10 sessions의 원인은 parser 발명이 아니라 실제 timestamp
tie/stall/gap이다. S07 counter reset은 이미 deterministic unwrap이 구현됐다. 나머지
시각은 추정·보간하지 않는다.

### 구현 계약

- view×output interval별 invalid reason bitmask
- reason: empty / temporal gap / sequence gap / timestamp plateau / nonfinite
- reason union은 정확히 `~valid_mask`
- 여러 이유가 겹치면 모두 보존
- invalid value는 scaling 전후 exact `+0.0`
- 모든 retained frame exact accounting, unaccounted `0`
- eligibility는 `valid_mask.all()`이 아니라 모든 invalid가 완전 설명됐는지로 판정

### 실행 결과

measured timing adjudication 29/29 확인. 기존 31개 invalid cell은 삭제·보간하지 않고
reason mask와 exact `+0.0`으로 유지. 이 결과는 sync/scientific authority를 만들지 않음.

## 5. C2 — synchronization raw replay

Verifier는 receipt marker를 입력으로 사용하지 않는다.

```text
exact raw consumption
  → radar geometry/timestamp/outlier/resample
  → target-free motion envelope + radar markers
  → raw RSP channel + RSP markers
  → epoch prior
  → ordered match + constant/affine mapping
  → complete alternative ambiguity scan
  → confidence/gate/decision
  → diagnostic receipt와 마지막에 exact compare
```

금지 입력: RR target, ECG-derived RR, protocol stage, manual interval, prediction,
candidate-oracle, `radar_observable`.

`sync_authorized = raw_replay_verified AND decision==accepted AND !ambiguous AND
exact source/config/raw closure`다. 기존 frozen 결과는 accepted 0, manual-review 13,
rejected 16이므로 verifier 구현 자체의 예상 authorized 수는 0/29다. Threshold를 낮춰
승인을 만드는 행위는 새 retrospective hypothesis이며 동일 authority generation에
섞지 않는다.

## 6. C3 — seven-stage replay

- workbook, protocol/sync config, cohort, BIOPAC를 exact consumed bytes에서 재파싱
- raw RSP marker부터 7-stage DP를 독립 재실행
- stage intervals, attempts, QC, status, confidence, path score/margin, phase-7 assignment
  canonical exact compare
- `protocol_replay_verified`, `alignment_eligible`, `phase_metric_eligible[7]` 분리
- phase metric은 replay·alignment가 모두 있고 해당 phase가 `auto`일 때만 허용
- 한 phase의 review가 다른 auto phase를 session-wide로 제거하지 않음
- transition guard 2 s, overlap 0.80, half-open interval 유지

현재 diagnostic replay는 29/29 canonical match지만 protocol overall은 auto 0,
review 26, uncertain 3이다. 이 재현성은 외부 trust root 없이 과학 권위가 아니다.

## 7. C4 — strict cache와 nested authority

- 새 acquisition schema/root만 입력으로 허용
- RF/SVD timing mask와 per-feature availability exact bind
- unavailable cell robust scaling fit에서 제외, transform 후 exact zero 재적용
- scientific cache loader는 verified owned arrays만 반환
- proposer는 prediction identity를 학습하지 않는 nested OOF
- scaler/proposer/checkpoint/selector/calibrator는 outer-test target 접근 금지
- 29 usable sessions / 18 physical identities exact cover
- 새 root 외 overwrite/resume/legacy 승격 금지

C0–C3 중 하나라도 실패하면 cache는 diagnostic/inspection-only다.

실제 diagnostic materialization:

- RF: `29 sessions / 9,575 windows / maps [9575,3,73,182] / aux [9575,1205]`
- mapping proposal 있음 18 sessions·5,830 windows, 없음 11 sessions·3,745 windows
- mapping 없음: radar-relative 32초 support만 사용, BIOPAC/reference/stage는 명시적
  NaN/-1/null/false sentinel, `reference_mapping_available=false`
- `reference_valid=false` 9,575/9,575, unavailable radar view 128개 exact zero
- SVD: 12 components, NFFT 4096, iteration 2, 10 variants, 29 sessions/9,575 rows
- RF/SVD 모두 private atomic no-replace publication, scientific/training false

## 8. C5–C6 — 모델 개선과 평가

V8R5 iteration은 한 번에 한 지배 가설만 변경한다.

1. coordinate-interaction + soft risk baseline
2. expert continuity/hysteresis
3. identity×RR-band group DRO + seven-mask balanced exposure
4. label-free radar pretraining + analog teacher distillation
5. target-free range tracker feature contract

각 iteration은 inner/discovery identity만 본다. 세 seed 방향 일치, overall/high-RR/tail
개선, normal-RR/worst-identity 비열등, mask/streaming/finite 위반 0일 때만 다음 단계로
승격한다. CUDA가 없으면 correctness/CPU smoke까지만 수행하며 CPU 결과로 CUDA latency나
full training을 대체하지 않는다.

## 9. 현재 즉시 실행 순서

1. stable raw reader + consumed-byte receipt
2. timing reason mask + 29-session 새 diagnostic reconstruction
3. raw synchronization replay verifier
4. phase-local protocol replay verifier
5. strict RF/SVD cache 재구축 가능성 재판정
6. authorization/CUDA가 실제 존재할 때만 V8R5 discovery
7. full suite 2회, capable-host sandbox tests, immutable provenance, 안전 commit/push

## 10. 완료와 차단 판정

다음은 Goal 종료가 아니라 정확한 resume boundary다.

- accepted sync 0/29 → verifier 결과를 보존하고 threshold를 사후 완화하지 않음
- external source launcher/trust root 부재 → local self-signature로 대체하지 않음
- CUDA 부재 → protected GPU training/latency 미실행
- prospective cohort 부재 → R1 이후 상용 완료 선언 금지

모든 code-local gate를 구현·검증하고 위 외부 조건만 남으면 exact blocker, source/data
hash, 실패 artifact, 재개 명령을 보존한다. 현재 leader 수치와 상용 gate 0/6은 새 sealed
OOF가 실제 생성되기 전까지 변경하지 않는다.
