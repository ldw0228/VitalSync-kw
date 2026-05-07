# UWB CNN/SNN 전체 비교 결과와 추천 방법론

이 문서는 지금까지 실행한 MobiVital, AAL SyncData-like 실험 결과를 한 번에 보기 위한 통합 정리입니다.

핵심 목적:

```text
1. 대학원생 코드 스타일의 중간 산출물 구조를 따라갈 수 있는지 확인
2. CNN continuous baseline과 SNN spike-based 방법을 같은 target 기준으로 비교
3. 실제 UWB + BIOPAC raw 수집 후 어떤 방법론을 가져갈지 결정
```

## 전체 실험 구조

최종 졸업작품에서 가져갈 프로세스는 아래 구조입니다.

```text
UWB raw + BIOPAC raw
-> 대학원생 코드 스타일 동기화/전처리
-> UWB_Biopac_SyncData-like 중간 산출물
-> CNN branch / SNN branch 분기
-> BIOPAC respiration target 예측
-> 정확도, spike activity, noise/artifact 방어율 비교
```

중간 산출물 필드:

```text
Fs_uwb
Fs_biopac
biopac_resp
com_row
com_col
tv_row
tv_col
metadata
```

CNN branch:

```text
SyncData-like UWB matrix
-> continuous preprocessing
-> CNN
-> respiration waveform prediction
```

SNN branch:

```text
SyncData-like UWB matrix
-> spike-friendly preprocessing
-> spike encoding
-> LIF-SNN / Spiking TCN
-> respiration waveform prediction
```

## MobiVital 실험

MobiVital은 이미 UWB I/Q range-bin time series와 respiration label이 정리된 데이터입니다. 따라서 raw parser나 BIOPAC sync 검증보다는 spike encoding 방향성을 보기 위한 실험으로 사용했습니다.

공통 조건:

```text
Data: MobiVital sample.csv
Sampling rate: 50 Hz
Window: 10 s
Stride: 2 s
Best range bin: 51
Target: respiration label
```

### CNN vs Delta-SNN 초기 비교

| 방법 | Bin 수 | Threshold scale | 입력 spike rate | hidden spike rate | RMSE | MAE | Corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 1 | - | - | - | 0.3195 | 0.2725 | 0.9785 |
| CNN | 5 | - | - | - | 0.3467 | 0.3032 | 0.9806 |
| Delta-SNN | 1 | 0.50 | 0.2961 | 0.1093 | 0.7420 | 0.6176 | 0.7377 |
| Delta-SNN | 1 | 0.75 | 0.2152 | 0.0885 | 0.7284 | 0.6063 | 0.7433 |
| Delta-SNN | 1 | 1.00 | 0.1517 | 0.0775 | 0.7512 | 0.6279 | 0.7332 |
| Delta-SNN | 5 | 0.75 | 0.1487 | 0.0959 | 0.8092 | 0.6712 | 0.6705 |

해석:

```text
CNN은 clean continuous input에서 매우 강함.
Delta-SNN은 정확도는 낮지만 spike activity가 낮아 event-driven baseline으로 의미가 있음.
5-bin 입력은 현재 sample에서는 SNN에 도움이 되지 않음.
```

### 전처리/디노이징 비교

| 모델 | 전처리 | RMSE | MAE | Corr | 입력 spike rate | hidden spike rate |
|---|---|---:|---:|---:|---:|---:|
| CNN | none | 0.3195 | 0.2725 | 0.9785 | - | - |
| CNN | moving_average | 0.3732 | 0.3232 | 0.9737 | - | - |
| CNN | fft_bandpass | 0.3927 | 0.2923 | 0.9474 | - | - |
| Delta-SNN | none | 0.7284 | 0.6063 | 0.7433 | 0.2152 | 0.0885 |
| Delta-SNN | moving_average | 0.5702 | 0.4245 | 0.8470 | 0.2119 | 0.1136 |
| Delta-SNN | fft_bandpass | 0.5836 | 0.4684 | 0.8334 | 0.3135 | 0.1349 |

해석:

```text
CNN은 원 신호의 세부 waveform 정보를 직접 활용하므로 clean sample에서는 none이 가장 좋았음.
SNN은 delta encoding 전 작은 jitter를 줄여야 불필요한 spike가 줄어 moving_average에서 개선됨.
```

### Rate vs Delta vs Hybrid

| 모델 | 인코딩 | 전처리 | RMSE | MAE | Corr | 입력 spike rate | hidden spike rate |
|---|---|---|---:|---:|---:|---:|---:|
| SNN | rate | moving_average | 0.5105 | 0.4339 | 0.9016 | 0.3425 | 0.1701 |
| SNN | delta | moving_average | 0.5702 | 0.4245 | 0.8470 | 0.2119 | 0.1136 |
| SNN | delta-rate hybrid | moving_average | 0.4986 | 0.4261 | 0.9223 | 0.2555 | 0.1389 |

해석:

```text
MobiVital sample에서는 amplitude 정보가 강하게 작동해 rate가 delta보다 좋았음.
hybrid가 가장 좋은 correlation을 보임.
단, hybrid는 채널 수가 늘어 총 spike activity가 증가할 수 있음.
```

### 추가 트렌드 후보

| 영역 | 조건 | RMSE | MAE | Corr | 입력 spike density | 예상 입력 spikes/sec | hidden spike rate | 해석 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Level-crossing encoding | `lif_cnn + level_crossing`, 5 levels | 0.8786 | 0.7503 | 0.6206 | 0.0058 | 약 2.9 | 0.0369 | 매우 sparse하지만 현재 설정은 정보 손실이 큼 |
| Adaptive threshold | `lif_cnn + hybrid`, target spike rate 0.2 | 0.5820 | 0.5037 | 0.8748 | 0.1808 | 약 27.1 | 0.1275 | activity를 줄이는 knob로 유용하지만 성능 trade-off 존재 |
| Spiking TCN | `spiking_tcn + hybrid` | 0.5056 | 0.4322 | 0.9582 | 0.2555 | 약 38.3 | 0.2216 | temporal pattern을 더 잘 잡지만 hidden activity 증가 |

MobiVital 기준 추천:

```text
CNN baseline: none + CNN
SNN baseline: moving_average + rate / delta
SNN proposed: moving_average + delta-rate hybrid
SNN model extension: delta-rate hybrid + Spiking TCN
```

## AAL SyncData-like 실험

AAL은 raw radar bScan을 포함하므로, 대학원생 코드 스타일의 중간 산출물 생성 과정을 맞춰보기 위해 사용했습니다.

AAL 변환 과정:

```text
AAL raw bScan
-> detrend 기반 background subtraction
-> row/column normalization
-> UWB_Biopac_SyncData-like .mat 생성
-> CNN/SNN 비교
```

주의:

```text
AAL에는 실제 BIOPAC이 없음.
biopac_resp 필드는 AAL lidar/reference respiration을 호환용으로 넣은 것임.
현재 AAL sample에서는 radar breath와 lidar/reference의 직접 정렬 상관이 낮아,
최종 비교 target은 radar_resp로 두고 raw pipeline 검증용으로 사용했음.
```

생성 파일:

```text
C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat
```

### AAL target 검증

| Target | Model | Encoding | RMSE | MAE | Corr | 해석 |
|---|---|---|---:|---:|---:|---|
| `biopac_resp` | LIF-CNN | hybrid | 0.8999 | 0.7604 | -0.0876 | AAL reference와 radar breath 정렬/센서 차이로 직접 target에 부적합 |
| `radar_resp` | LIF-CNN | hybrid | 0.3046 | 0.2468 | 0.9618 | raw UWB matrix에서 radar breath 재구성 가능 |
| `radar_resp` | Spiking TCN | hybrid | 0.2770 | 0.2206 | 0.9645 | AAL SyncData-like SNN 중 가장 좋음 |

### AAL CNN/SNN 전체 비교

공통 조건:

```text
Target: radar_resp
Selected range bin: 184
Window: 10 s
Stride: 2 s
Epochs: 30
```

| 방법 | 입력 | 전처리 | Encoding | Model | RMSE | MAE | Corr | input spikes/sec | hidden spike rate |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| CNN | continuous | none | 없음 | CNN | 0.0721 | 0.0324 | 0.9962 | - | - |
| CNN | continuous | moving_average | 없음 | CNN | 0.0783 | 0.0415 | 0.9957 | - | - |
| CNN | continuous | fft_bandpass | 없음 | CNN | 0.1574 | 0.1229 | 0.9829 | - | - |
| SNN | spike | moving_average | rate | LIF-CNN | 0.4233 | 0.3304 | 0.9189 | 11.1 | 0.2346 |
| SNN | spike | moving_average | delta | LIF-CNN | 0.3140 | 0.2396 | 0.9615 | 10.4 | 0.1540 |
| SNN | spike | moving_average | delta-rate hybrid | LIF-CNN | 0.3046 | 0.2468 | 0.9618 | 21.5 | 0.2523 |
| SNN | spike | moving_average | adaptive hybrid | LIF-CNN | 0.3422 | 0.2688 | 0.9433 | 14.9 | 0.2061 |
| SNN | spike | moving_average | level-crossing | LIF-CNN | 0.3484 | 0.2890 | 0.9293 | 2.5 | 0.0578 |
| SNN | spike | moving_average | delta-rate hybrid | Spiking TCN | 0.2770 | 0.2206 | 0.9645 | 21.5 | 0.1825 |

해석:

```text
정확도만 보면 CNN none이 가장 강함.
SNN끼리는 delta-rate hybrid + Spiking TCN이 가장 좋음.
level-crossing은 정확도는 낮지만 input spikes/sec가 2.5로 매우 낮아 효율 비교군으로 가치가 있음.
adaptive hybrid는 spike activity를 줄이는 대신 성능도 낮아져 trade-off knob로 봐야 함.
```

## 전체 결론

### 1. CNN은 정확도 baseline으로 반드시 필요

현재 MobiVital과 AAL 모두 clean continuous input에서는 CNN이 강합니다. 따라서 최종 졸업작품에서도 CNN은 비교 대상이 아니라 기준선입니다.

```text
CNN = 정확도 상한 baseline
SNN = spike 효율, event-driven 표현, noise/artifact 방어율 비교 대상
```

### 2. SNN 제안 방법은 hybrid + Spiking TCN이 가장 설득력 있음

SNN 내부 비교에서는 단순 rate, delta보다 hybrid와 Spiking TCN 조합이 가장 좋은 후보입니다.

```text
rate = amplitude 정보
delta = 변화/event 정보
hybrid = amplitude + 변화 정보 결합
Spiking TCN = 호흡의 temporal pattern을 더 길게 반영
```

### 3. adaptive threshold는 최종 방법이라기보다 조절 변수

adaptive hybrid는 spike 수를 줄일 수 있지만 성능이 떨어질 수 있습니다. 따라서 최종 실험에서는 target spike rate를 여러 값으로 sweep해야 합니다.

```text
target spike rate: 0.1 / 0.2 / 0.3 / 0.4
평가: Corr, RMSE, input spikes/sec, hidden spike rate
```

### 4. level-crossing은 저전력/저활성 비교군

level-crossing은 정확도 최상 후보는 아니지만 spike 수가 매우 적습니다. 최종 논리에서 "ultra-sparse encoding baseline"으로 넣으면 좋습니다.

## 실제 raw 데이터 수집 후 추천 실험표

실제 UWB + BIOPAC raw가 들어오면 아래 비교를 그대로 수행하는 것을 추천합니다.

| 역할 | 입력 | 전처리 | Encoding | Model | Target |
|---|---|---|---|---|---|
| 정확도 baseline | continuous UWB | none | 없음 | CNN | BIOPAC respiration |
| 전처리 baseline | continuous UWB | bandpass 또는 moving_average | 없음 | CNN | BIOPAC respiration |
| SNN 기본 baseline | spike | moving_average | rate | LIF-CNN | BIOPAC respiration |
| SNN event baseline | spike | moving_average | delta | LIF-CNN | BIOPAC respiration |
| SNN 효율 비교군 | spike | moving_average | level-crossing | LIF-CNN | BIOPAC respiration |
| SNN 제안 방법 1 | spike | moving_average | delta-rate hybrid | LIF-CNN | BIOPAC respiration |
| SNN 제안 방법 2 | spike | moving_average | adaptive delta-rate hybrid | Spiking TCN | BIOPAC respiration |

최종 평가 지표:

```text
RMSE
MAE
Correlation
input spike density
input spikes/sec
hidden spike rate
accuracy per spike
noise/artifact 상황에서 성능 저하율
```

## 최종 추천

현재까지 결과만 기준으로 하면 아래 방향이 가장 안정적입니다.

```text
1. 대학원생 코드 스타일로 SyncData-like 중간 산출물 생성
2. CNN none을 정확도 baseline으로 둠
3. SNN은 rate, delta, level-crossing을 baseline으로 둠
4. 제안 방법은 adaptive delta-rate hybrid + Spiking TCN으로 둠
5. 실제 BIOPAC target에서 정확도와 spike efficiency, noise robustness를 함께 비교
```

한 문장 요약:

```text
CNN을 이기는 것이 1차 목표가 아니라, UWB 생체신호를 spike-friendly하게 변환했을 때 정확도 손실 대비 spike 효율과 noise/artifact 방어율이 얼마나 나오는지를 검증하는 방향이 가장 타당합니다.
```
