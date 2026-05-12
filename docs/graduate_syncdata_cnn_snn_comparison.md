# 조교/대학원생 SyncData CNN/SNN 비교 결과

이 문서는 조교/대학원생이 제공한 정리된 `UWB_Biopac_SyncData.mat` 파일을 대상으로, 지금까지 구현한 CNN/SNN 방법론을 그대로 적용한 초기 비교 결과입니다.

## 목적

```text
조교 제공 UWB_Biopac_SyncData.mat
-> CNN continuous baseline
-> SNN rate / delta / hybrid / level-crossing / Spiking TCN
-> BIOPAC respiration target 예측 가능성 확인
```

사용한 파일:

```text
C:\Users\hai\Desktop\UWB_BIOPAC_MODEL_DATA\UWB_BIOPAC_MODEL_DATA\1\UWB_Biopac_SyncData.mat
```

확인된 필드:

| 필드 | shape | 의미 |
|---|---:|---|
| `Fs_uwb` | scalar | UWB sampling rate |
| `Fs_biopac` | scalar | BIOPAC sampling rate |
| `biopac_resp` | `[75161]` | BIOPAC respiration target |
| `tv_row` | `[360, 5111]` | TV-side UWB row-normalized matrix |
| `com_row` | `[360, 5110]` | computer-side UWB row-normalized matrix |
| `tv_col` | `[360, 5111]` | TV-side UWB column-normalized matrix |
| `com_col` | `[360, 5110]` | computer-side UWB column-normalized matrix |

## 사전 확인

각 UWB 필드에서 BIOPAC respiration과 가장 직접 상관이 높은 range bin을 확인했습니다.

| 필드 | best bin | 직접 Corr |
|---|---:|---:|
| `com_row` | 346 | -0.1998 |
| `tv_row` | 223 | 0.2206 |
| `com_col` | 86 | 0.0554 |
| `tv_col` | 218 | 0.0454 |

해석:

```text
subject 1 기준으로는 tv_row가 가장 낫지만 직접 상관이 0.22 수준이라 높지 않습니다.
즉 모델 비교 전에 레이더 선택, 구간 선택, sync 정렬, target filtering을 다시 점검할 필요가 있습니다.
```

## 실험 조건

공통 조건:

```text
Target: biopac_resp
Window: 10 s
Stride: 2 s
Epochs: 30
Train/val/test split: 시간 순서 기반 70/15/15
```

비교한 방법:

```text
CNN none
CNN moving_average
CNN fft_bandpass
SNN rate + LIF-CNN
SNN delta + LIF-CNN
SNN delta-rate hybrid + LIF-CNN
SNN adaptive hybrid + LIF-CNN
SNN level-crossing + LIF-CNN
SNN delta-rate hybrid + Spiking TCN
```

## com_row 기준 결과

| 방법 | RMSE | MAE | Corr | input spikes/sec | hidden spike rate | selected bin |
|---|---:|---:|---:|---:|---:|---:|
| CNN none | 1.1289 | 0.9253 | 0.2471 | - | - | 346 |
| CNN moving_average | 1.1609 | 0.9461 | 0.1821 | - | - | 148 |
| CNN fft_bandpass | 1.1194 | 0.9160 | 0.2785 | - | - | 346 |
| SNN rate LIF | 1.1778 | 0.9652 | 0.0029 | 7.9 | 0.1250 | 148 |
| SNN delta LIF | 1.1856 | 0.9671 | 0.0600 | 6.4 | 0.0817 | 148 |
| SNN hybrid LIF | 1.1816 | 0.9630 | 0.0657 | 14.3 | 0.0998 | 148 |
| SNN adaptive hybrid LIF | 1.1778 | 0.9608 | 0.0734 | 11.3 | 0.1006 | 148 |
| SNN level LIF | 1.1217 | 0.9126 | 0.2701 | 1.0 | 0.0557 | 148 |
| SNN hybrid Spiking TCN | 1.1783 | 0.9540 | 0.1564 | 14.3 | 0.1932 | 148 |

## tv_row 기준 결과

`tv_row`는 subject 1에서 BIOPAC과 직접 상관이 가장 높았던 필드입니다.

| 방법 | RMSE | MAE | Corr | input spikes/sec | hidden spike rate | selected bin |
|---|---:|---:|---:|---:|---:|---:|
| CNN none | 1.1067 | 0.9108 | 0.3090 | - | - | 223 |
| CNN moving_average | 1.2025 | 0.9933 | 0.1425 | - | - | 259 |
| CNN fft_bandpass | 1.0325 | 0.8387 | 0.4617 | - | - | 222 |
| SNN rate LIF | 1.1263 | 0.9262 | 0.2957 | 7.6 | 0.1032 | 259 |
| SNN delta LIF | 1.0901 | 0.8900 | 0.3781 | 6.6 | 0.0902 | 259 |
| SNN hybrid LIF | 1.0978 | 0.9009 | 0.3740 | 14.3 | 0.1165 | 259 |
| SNN adaptive hybrid LIF | 1.1099 | 0.9028 | 0.3130 | 11.0 | 0.1084 | 259 |
| SNN level LIF | 1.0854 | 0.9005 | 0.3642 | 1.3 | 0.0481 | 259 |
| SNN hybrid Spiking TCN | 1.1197 | 0.9142 | 0.2743 | 14.3 | 0.1909 | 259 |

## 현재 해석

subject 1 하나만 보면, 정리된 `UWB_Biopac_SyncData.mat`에서도 바로 높은 성능이 나오지는 않았습니다.

관찰:

```text
1. com_row보다 tv_row가 조금 더 나음.
2. tv_row 기준 최고 Corr는 CNN fft_bandpass의 0.4617.
3. SNN 중에서는 delta LIF가 Corr 0.3781로 가장 높음.
4. level-crossing은 Corr 0.3642로 비슷한 수준이면서 input spikes/sec가 1.3으로 매우 낮음.
5. hybrid와 Spiking TCN이 항상 이기지는 않았음.
```

중요한 결론:

```text
이 결과는 "SNN 방법론이 안 된다"라기보다,
subject 1의 BIOPAC target과 UWB matrix 사이의 정렬/구간/레이더 선택이 먼저 점검되어야 한다는 의미가 큽니다.
```

## 왜 AAL/MobiVital보다 낮게 나왔나

가능한 원인:

```text
1. BIOPAC respiration과 UWB matrix의 시간 정렬이 완전히 맞지 않을 수 있음.
2. subject 1 전체 구간 안에 움직임, 무호흡, 센서 흔들림, bad segment가 섞여 있을 수 있음.
3. com/tv 중 실제 호흡이 더 잘 잡힌 radar가 세션마다 다를 수 있음.
4. best bin을 전체 구간 상관으로 한 번만 고르는 방식이 부족할 수 있음.
5. 대학원생 코드는 rate 추정/시각화 중심이고, 우리는 waveform supervised learning을 바로 걸었기 때문에 목적이 다를 수 있음.
```

## 다음에 해야 할 일

조교 데이터 전체를 제대로 활용하려면 바로 모델을 크게 바꾸기보다 아래 순서가 먼저입니다.

```text
1. 모든 UWB_Biopac_SyncData.mat에 대해 quick scan 수행
2. com_row / tv_row / com_col / tv_col 중 best field 자동 선택
3. BIOPAC-UWB lag correction 후보 확인
4. bad segment 또는 movement segment 제외
5. subject/session별 best corr 분포 확인
6. 그 다음 CNN/SNN 전체 비교를 다시 수행
```

이 과정을 자동으로 확인하기 위해 `scan_syncdata_quality.py`를 추가했습니다.

```bash
python research/aal_raw_starter/scan_syncdata_quality.py --root C:\Users\hai\Desktop\UWB_BIOPAC_MODEL_DATA\UWB_BIOPAC_MODEL_DATA --out-dir C:\Users\hai\Desktop\UWB_BIOPAC_MODEL_DATA\quality_scan
```

현재 전체 49개 파일 quick scan 결과, 가장 높은 직접 Corr도 약 0.275 수준이었습니다. 상위 파일에 lag correction을 적용해도 최고 약 0.35 수준이라, 단순 시간 지연만으로 해결되는 문제는 아닌 것으로 보입니다.

권장 quick scan 출력:

| 항목 | 의미 |
|---|---|
| file path | 세션 파일 |
| best field | com_row/tv_row/com_col/tv_col 중 최고 |
| best bin | 최고 상관 range bin |
| best corr | BIOPAC과 직접 상관 |
| duration | 세션 길이 |
| recommended use | train/test 사용 가능 여부 |

## 현재 기준 추천

subject 1 결과만 기준으로는 아래처럼 말하는 것이 안전합니다.

```text
조교 제공 SyncData 파일은 우리 파이프라인으로 읽고 CNN/SNN 비교까지 돌릴 수 있다.
하지만 subject 1에서는 UWB-BIOPAC 직접 상관이 낮아, 전체 방법론 성능 비교 전에 Sync quality scan이 필요하다.
현재 subject 1에서는 CNN fft_bandpass가 가장 높은 Corr를 보였고, SNN 중에서는 delta LIF가 가장 높았다.
level-crossing은 정확도는 조금 낮지만 spike 효율이 매우 좋아 효율 baseline으로 유지할 가치가 있다.
```
