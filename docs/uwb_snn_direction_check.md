# UWB 스파이크 인코딩 방향성 확인

날짜: 2026-05-05

이 문서는 최종 졸업작품 결과가 아니라, 앞으로 어떤 방식으로 진행할지 보기 위한 초기 방향성 실험 정리입니다.
목표는 MobiVital 형태의 UWB I/Q 신호를 스파이크 입력으로 변환할 수 있는지, 그리고 일반 CNN baseline과 비교했을 때 어떤 특징이 나오는지 확인하는 것입니다.

## 데이터

- 데이터 샘플: MobiVital `sample.csv`
- 크기: 1500행 x 254열
- 샘플링 주파수: 50 Hz
- 길이: 30초
- UWB I 데이터: 13-132열
- UWB Q 데이터: 133-252열
- 정답 라벨: respiration waveform, 253열

특징 생성 과정:

```text
I/Q -> magnitude = sqrt(I^2 + Q^2)
respiration 라벨과 상관계수가 가장 높은 range bin 선택
10초 window, 2초 stride
```

이번 샘플에서 가장 잘 맞은 bin:

```text
bin 51
거리 추정: 약 2.87 m
respiration과의 상관계수: 약 0.936
```

## 비교한 방법

1. CNN baseline
   - 입력: 연속형 UWB magnitude 신호
   - 모델: 작은 1D CNN
   - 의미: 일반 딥러닝 방식의 성능 기준선

2. Delta-SNN
   - 입력: delta spike encoding을 적용한 UWB magnitude 신호
   - 스파이크 채널: positive event / negative event
   - 모델: surrogate gradient를 사용한 간단한 LIF 기반 spiking CNN
   - 의미: event-driven, low-activity SNN baseline

## 결과

| 방법 | Bin 수 | Threshold scale | 입력 spike rate | hidden spike rate | RMSE | MAE | Corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 1 | - | - | - | 0.3195 | 0.2725 | 0.9785 |
| CNN | 5 | - | - | - | 0.3467 | 0.3032 | 0.9806 |
| Delta-SNN | 1 | 0.50 | 0.2961 | 0.1093 | 0.7420 | 0.6176 | 0.7377 |
| Delta-SNN | 1 | 0.75 | 0.2152 | 0.0885 | 0.7284 | 0.6063 | 0.7433 |
| Delta-SNN | 1 | 1.00 | 0.1517 | 0.0775 | 0.7512 | 0.6279 | 0.7332 |
| Delta-SNN | 5 | 0.75 | 0.1487 | 0.0959 | 0.8092 | 0.6712 | 0.6705 |

## 전처리/디노이징 추가 비교

추가로 `none`, `moving_average`, `fft_bandpass` 전처리를 비교했습니다.

| 모델 | 전처리 | RMSE | MAE | Corr | 입력 spike rate | hidden spike rate |
|---|---|---:|---:|---:|---:|---:|
| CNN | none | 0.3195 | 0.2725 | 0.9785 | - | - |
| CNN | moving_average | 0.3732 | 0.3232 | 0.9737 | - | - |
| CNN | fft_bandpass | 0.3927 | 0.2923 | 0.9474 | - | - |
| Delta-SNN | none | 0.7284 | 0.6063 | 0.7433 | 0.2152 | 0.0885 |
| Delta-SNN | moving_average | 0.5702 | 0.4245 | 0.8470 | 0.2119 | 0.1136 |
| Delta-SNN | fft_bandpass | 0.5836 | 0.4684 | 0.8334 | 0.3135 | 0.1349 |

## Rate vs Delta vs Delta-Rate Hybrid 추가 비교

최근 SNN 시계열 처리에서는 기본 rate encoding, 변화량 중심 delta event encoding, 그리고 둘을 합친 hybrid encoding을 함께 비교하는 흐름이 많습니다. 그래서 가장 좋았던 SNN 조건인 `moving_average + threshold-scale 0.75`에서 rate-only, delta-only, delta-rate hybrid를 비교했습니다.

| 모델 | 인코딩 | 전처리 | RMSE | MAE | Corr | 입력 spike rate | hidden spike rate |
|---|---|---|---:|---:|---:|---:|---:|
| SNN | rate | moving_average | 0.5105 | 0.4339 | 0.9016 | 0.3425 | 0.1701 |
| SNN | delta | moving_average | 0.5702 | 0.4245 | 0.8470 | 0.2119 | 0.1136 |
| SNN | delta-rate hybrid | moving_average | 0.4986 | 0.4261 | 0.9223 | 0.2555 | 0.1389 |

이번 샘플에서는 hybrid가 가장 높은 correlation을 보였고, rate-only도 delta-only보다 정확도는 좋았습니다.

```text
delta-only corr: 0.8470
rate-only corr:  0.9016
hybrid corr:     0.9223
```

다만 rate-only는 입력 spike rate가 가장 높았습니다. 즉, 정확도만 보면 rate 계열이 유리하지만, event-driven 효율까지 보면 delta와 hybrid를 함께 봐야 합니다.

```text
input spike rate:  delta 0.2119 / hybrid 0.2555 / rate 0.3425
hidden spike rate: delta 0.1136 / hybrid 0.1389 / rate 0.1701
```

따라서 hybrid는 rate-only보다 sparse하면서 delta-only보다 정확도가 높은 중간 후보로 보는 것이 좋습니다.

참고: 현재 delta threshold는 window별 `std(diff(signal))` 기반으로 정해지므로, 이미 간단한 adaptive threshold 방식입니다.

해석:

```text
CNN은 no denoising이 가장 좋았습니다.
SNN은 moving average 전처리 후 hybrid encoding을 적용했을 때 가장 좋아졌습니다.
```

따라서 다음 실험의 시작점은 아래 조합이 좋아 보입니다.

```text
CNN baseline: preprocess none
SNN 기본 baseline: preprocess moving_average + rate
SNN event baseline: preprocess moving_average + delta + threshold-scale 0.75
SNN 제안 후보: preprocess moving_average + delta-rate hybrid + threshold-scale 0.75
```

## 현재 해석

clean continuous input에서는 CNN baseline이 훨씬 강합니다. 이건 예상 가능한 결과이고, CNN을 정확도 기준선으로 두는 것이 좋습니다.

Delta-SNN은 직접적인 waveform reconstruction 정확도는 낮지만, sparse event-style 입력과 낮은 hidden spike activity를 보입니다.
따라서 졸업작품 방향은 clean 데이터에서 CNN을 바로 이기는 것이 아니라, spike-friendly 전처리를 통해 강건성과 효율성을 함께 비교하는 쪽이 더 타당합니다.

초기 관찰:

```text
threshold 0.75가 시작점으로 좋아 보입니다.
threshold 0.5는 spike가 많아지지만 성능 이득은 크지 않았습니다.
threshold 1.0은 spike가 줄어드는 대신 정확도가 조금 떨어졌습니다.
이번 작은 샘플에서는 5-bin 입력이 도움이 되지 않았습니다.
```

## 추천 방향

프로젝트 방향은 아래처럼 잡는 것이 좋습니다.

```text
CNN baseline = 연속 신호 기반 고정확도 기준선
Delta-SNN = 저활성 event-based representation 기준선
제안 방법 = spike encoding을 개선하고 noise/artifact 상황에서 성능 방어율을 검증
```

다음 실험에서 봐야 할 것:

1. 더 많은 데이터, 가능하면 여러 subject/session 사용
2. noise와 motion artifact 상황에서 성능 비교
3. 고정 threshold 대신 adaptive thresholding 적용
4. delta encoding, rate encoding, delta-rate hybrid encoding 비교
5. RMSE/correlation뿐 아니라 spike rate와 성능 저하율을 함께 평가

## 추가 SNN 트렌드 확장 후보

| 후보 | 적용 의미 | 장점 | 단점 | 현재 우선순위 |
|---|---|---|---|---|
| Rate encoding | 신호 크기를 spike 빈도로 표현 | 구현이 쉽고 성능이 안정적 | spike 수가 많아질 수 있음 | baseline |
| Delta/event encoding | 변화량이 threshold를 넘을 때만 spike 발생 | sparse하고 event-driven 명분이 강함 | 절대 amplitude 정보가 약함 | baseline |
| Delta-rate hybrid | 변화 정보와 amplitude 정보를 함께 사용 | 이번 샘플에서 정확도와 효율 균형이 좋음 | 채널 수와 spike activity 증가 | 1순위 제안 |
| Adaptive threshold | window/subject별 threshold 자동 조정 | subject 차이와 신호 크기 차이에 대응 | threshold 설계에 따라 결과가 흔들림 | 이미 일부 적용 |
| Spiking TCN | 시간축 convolution을 SNN으로 확장 | 호흡처럼 긴 temporal pattern에 유리 | 현재 LIF-SNN보다 구현/튜닝 부담 증가 | 다음 단계 |
| Learnable encoder | spike 변환 파라미터를 학습 | 데이터가 늘면 성능 향상 가능 | 데이터가 적으면 과적합 위험 | 실제 데이터 확보 후 |

현재 결론은 `moving_average + delta-rate hybrid`를 제안 후보로 두고, `rate`, `delta`, `CNN`을 baseline으로 같이 가져가는 것입니다.

## 팀 공유용 요약

MobiVital UWB 샘플로 방향성 실험을 해봤습니다. 일반 1D CNN은 연속형 UWB magnitude 입력에서 respiration waveform을 매우 잘 따라가므로 정확도 baseline으로 사용할 수 있습니다. 반면 단순 Delta Spike + LIF-SNN은 clean 데이터 정확도는 낮지만 sparse spike activity를 보여주기 때문에, spike-friendly 전처리를 연구하는 방향성은 있습니다.

따라서 방향은 "SNN이 clean accuracy에서 CNN을 바로 이긴다"가 아니라, "UWB 신호를 spike encoding으로 event-driven 저활성 표현으로 바꾸고, CNN baseline과 강건성/효율을 함께 비교한다"로 잡는 것이 좋아 보입니다.
