# SnnProject 개발 진행상황 — acquisition v2 causal-bound checkpoint

- 기준 시각: 2026-08-31, Asia/Seoul
- 대상: 3대 XeThru UWB radar 기반 호흡수 추정 hybrid/SNN 연구 시스템
- 증거 수준: 18명 retrospective cohort
- 제품 판정: 상용·의료 제품 아님, 내부 상용 정확도 gate `0/6`

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
  재해시로 TOCTOU 차단

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
- receipt의 marker index/time, residual, mapping, raw/config hash를 재계산
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
- scientific SVD array는 검증 후 mmap에서 독립된 read-only memory snapshot으로 복사,
  복사 뒤 원본을 다시 해시해 학습 중 파일 변조 차단
- `radar_observable`은 target-dependent이므로 추론 feature 금지
- diagnostic cache를 scientific mode로 넘기면 학습 시작 전에 실패

현재 새 scientific training을 시작하지 않은 이유는 계산 부족이 아니라 data
authority failure다. 이를 우회한 학습 수치는 유효한 개선으로 인정할 수 없다.

## 6. 테스트와 재현성

- focused acquisition/cache/train/SVD suites 통과
- 현재 source에서 full suite 두 번 연속 통과
- 각 실행: 1,441 collected, 1,437 passed, 4 skipped
- managed namespace에서 skip된 실제 bubblewrap 4개를 권한 있는 host context에서
  별도 실행: 4/4 passed
- import-time Torch thread mutation 제거
- SVD 1×1 projection의 fixed-order accumulation으로 prefix/thread 수치 불안정 제거
- `py_compile`과 `git diff --check` 통과

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
