# 3-Radar SNN 핵심 기술 로드맵

## 1. 최종 방향

- Dense radar 입력과 18명 규모, **analog front-end + compact SNN**
- 고호흡수 harmonic 혼동, **coordinate-aware candidate graph**
- 안정적인 전체 RR 예측, **direct path + safe router**
- 최종 권고 구조, **Causal Coordinate-aware Harmonic Graph SNN(CCHG-SNN)**

## 2. 전체 로드맵

```text
Raw parser
→ Measured timing·radar–BIOPAC sync
→ BIOPAC RR reference·실험 stage
→ Causal denoising
→ Range–frequency map·SVD·range evidence
→ Physical-identity dataset split
→ Structured direct RR model
→ Coordinate-aware harmonic graph
→ PLIF/ALIF SNN router
→ Risk calibration·safe output
→ Identity OOF·streaming·fault evaluation
```

## 3. 단계별 기술과 선정 이유

| 단계 | 선정 기술 | 간략한 설명 | 선정 이유 |
|---:|---|---|---|
| 1. Raw parser | `3×uint32 + 182×float32` typed parser | 740-byte record의 header·payload 분리 | Header 혼입 방지, 정확한 radar 신호 확보 |
| 2. Radar timing | Metadata measured timestamp + bounded plateau repair | 실제 frame 시간·sequence 기반 시간축 | Nominal 40 Hz의 drift·jitter 보정 |
| 3. Sensor sync | Marker 기반 affine sync | `t_rsp = offset + scale × t_radar` | 시작 offset과 장시간 clock drift 동시 보정 |
| 4. RR reference | FFT·IBI·Hilbert 합의 QC | 세 RR 추정치와 신호 품질의 교차검사 | Harmonic·clipping·불규칙 호흡 label 오류 감소 |
| 5. 실험 stage | Ordered DP + manual anchor | 7개 phase의 순서·길이·gap 결합 | 실험 지연·재시도·시간 편차 반영 |
| 6. Denoising | Past-only repair + detrend + Hann FFT + noise normalization | 미래 sample 없는 causal spectrum 생성 | Streaming 일치, 호흡 harmonic 보존 |
| 7. 기본 표현 | Raw-power range–frequency map | 주파수×range 2차원 evidence | 호흡 peak와 공간 분포 동시 보존 |
| 8. Source 보조 | Label-free SVD | Raw·velocity·range-difference 성분 분리 | Motion/source 구분, BIOPAC 입력 차단 |
| 9. Range 보조 | Causal active-range tracker | Bin·confidence·missing·multimodal 출력 | 사람 가능성이 높은 영역의 target-free 추적 |
| 10. Dataset split | Physical-identity 6-fold | 같은 사람의 모든 session을 같은 fold에 배치 | 반복 피험자·87.5% window overlap 누수 방지 |
| 11. Base RR | Structured TriRadarRRSNN | Shared encoder·range attention·radar reliability | 현재 데이터의 가장 안정적인 direct RR 기반 |
| 12. Harmonic router | Coordinate-aware directed graph | Radar·ratio·branch 좌표와 `×1–×4` 후보 결합 | 25–35 bpm의 배수·약수 선택 오류 대응 |
| 13. SNN core | 8–12 step PLIF/ALIF | 후보 evidence와 session state의 spike 누적 | 작은 데이터·짧은 simulation step에 적합 |
| 14. 학습 | Surrogate BPTT + KD + TET + split-safe SSL | Direct RR·router·state의 end-to-end 최적화 | 학습 안정성·표본 효율·낮은 step 성능 강화 |
| 15. 안전 출력 | Soft expected-risk + hard-safe decision | Candidate·direct anchor·no-estimate 선택 | 위험한 harmonic correction 억제 |
| 16. 평가 | Identity OOF + non-overlap + 7 radar masks | 새 사람·중첩 window·radar 결측 분리 평가 | 일반화·강건성·배포 안정성 확인 |

## 4. 최종 모델 구조

```text
3-radar RF/SVD/range evidence
→ Shared analog spatial encoder
→ Direct RR posterior
  + Evidence×radar×ratio×branch coordinate tokens
→ Directed harmonic candidate graph
→ 8–12 step PLIF/ALIF SNN
→ Candidate/factor posterior + expert risk + quality
→ Hard candidate / direct anchor / no-estimate
```

## 5. 학습 로드맵

| 순서 | 학습 구성 | 목적 |
|---:|---|---|
| 1 | Outer-train identity 전용 SSL | Reference-invalid radar까지 활용한 표현학습 |
| 2 | Analog teacher | 안정적인 RR posterior·중간 feature 생성 |
| 3 | Direct SNN + KD + TET | 짧은 step의 RR 분포 학습 |
| 4 | Harmonic graph + soft-risk | 올바른 `×1–×4` 후보 선택 |
| 5 | Stateful episode fine-tuning | 연속 window의 호흡 상태 유지 |
| 6 | Held-identity calibration | Threshold·fallback·uncertainty 고정 |
| 7 | Identity OOF·streaming replay | 새 사람·시간순 추론 성능 검증 |

## 6. 핵심 선택 요약

| 데이터 특성 | 구조 선택 |
|---|---|
| Native spike가 아닌 dense radar | Analog encoder 유지 |
| 물리적 identity 18명 | Compact parameter 규모 |
| 3대 동종 radar | Shared encoder + reliability fusion |
| Frequency·range 위치 의미 | 좌표 보존 convolution·graph |
| High-RR harmonic 혼동 | Direct RR + `×1–×4` candidate dual path |
| 위험한 candidate correction | Soft-risk 학습 + direct fallback |
| Reference-invalid window 다수 | Split-safe SSL·state update |
| 반복 session·중첩 window | Physical-identity split·non-overlap 평가 |
