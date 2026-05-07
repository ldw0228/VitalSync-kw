# MobiVital SNN Starter

이 폴더는 졸업작품 방향을 잡기 위한 UWB/SNN 초기 실험 코드입니다.

핵심 질문은 다음과 같습니다.

```text
UWB I/Q 생체신호를 spike-friendly representation으로 바꿔서
SNN에 넣는 방향이 졸업작품 주제로 타당한가?
```

현재 코드는 최종 모델이 아니라, 팀에서 방향을 합의하기 위한 재현 가능한 작은 실험입니다.

## 배경

협업 연구실에서 받은 기존 MATLAB 코드는 UWB/BIOPAC 데이터를 동기화하고, clutter 제거, filtering, row/col normalization 등을 수행하는 전처리 baseline에 가깝습니다.

하지만 그 코드는 SNN을 위한 spike encoding 코드가 아닙니다. 따라서 우리 쪽 기여는 아래처럼 잡는 것이 좋습니다.

```text
기존/전처리된 UWB 신호
-> SNN에 맞는 spike representation으로 변환
-> CNN baseline과 성능, 강건성, spike activity 비교
```

즉, 목표는 "SNN을 그냥 붙인다"가 아니라 "UWB 생체신호를 SNN에 맞게 전처리하는 방법을 비교한다"입니다.

## 사용한 공개 샘플

방향성 확인을 위해 MobiVital 공개 샘플을 사용했습니다.

- 데이터: MobiVital `sample.csv`
- 크기: 1500행 x 254열
- 길이: 30초
- 샘플링 주파수: 50 Hz
- UWB I 데이터: 13-132열
- UWB Q 데이터: 133-252열
- Respiration label: 253열
- Pulse label: 254열

컬럼 번호는 MobiVital README 기준의 1-based 번호입니다.

## 현재 파이프라인

```text
sample.csv
-> UWB I/Q 분리
-> magnitude = sqrt(I^2 + Q^2)
-> respiration label과 가장 상관이 높은 range bin 선택
-> 10초 window 생성, 2초 stride
-> CNN baseline 또는 delta spike encoding + SNN 학습
```

이번 샘플에서 자동 선택된 best bin은 다음과 같습니다.

```text
best bin: 51
거리 추정: 약 2.87 m
respiration label과의 상관계수: 약 0.936
```

이 결과는 단순 magnitude만으로도 특정 range bin에서 호흡 정보가 꽤 강하게 잡힌다는 것을 보여줍니다.

## 구현된 파일

| 파일 | 역할 |
|---|---|
| `mobivital_dataset.py` | MobiVital CSV loader, I/Q 분리, magnitude/phase 생성, window dataset 생성 |
| `spike_encoding.py` | delta spike encoding, rate encoding, delta-rate hybrid encoding 함수 |
| `quick_inspect.py` | 샘플 CSV 구조 확인, best bin 탐색, delta spike plot 생성 |
| `train_baseline_cnn.py` | continuous UWB magnitude 입력으로 1D CNN baseline 학습 |
| `train_delta_snn.py` | delta spike 입력으로 LIF 기반 SNN 학습 |

## 빠른 실행

샘플 구조와 best bin을 확인합니다.

```powershell
python quick_inspect.py --csv C:\Users\hai\Desktop\uwb_sample\sample.csv
```

CNN baseline을 학습합니다.

```powershell
python train_baseline_cnn.py --csv C:\Users\hai\Desktop\uwb_sample\sample.csv --epochs 30
```

Delta-SNN을 학습합니다.

```powershell
python train_delta_snn.py --csv C:\Users\hai\Desktop\uwb_sample\sample.csv --epochs 30 --threshold-scale 0.75
```

Windows에서 PyTorch 설치가 꼬이면 CPU wheel index를 사용하는 것이 안전합니다.

```powershell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

## 비교한 방법

### 1. CNN Baseline

```text
입력: continuous UWB magnitude
모델: 작은 1D CNN
출력: respiration waveform reconstruction
```

이 모델은 SNN과 비교하기 위한 정확도 기준선입니다. clean continuous input에서는 CNN이 강한 것이 자연스럽습니다.

### 2. Delta-SNN

```text
입력: delta spike encoded UWB magnitude
스파이크 채널: positive event / negative event
모델: surrogate gradient를 사용하는 LIF 기반 spiking CNN
출력: respiration waveform reconstruction
```

Delta spike encoding은 시간 변화량이 threshold를 넘을 때만 spike를 발생시킵니다.

```text
x[t] - x[t-1] > threshold  -> positive spike
x[t] - x[t-1] < -threshold -> negative spike
변화가 작음                  -> no spike
```

UWB 생체신호는 절대값보다 미세한 시간 변화가 중요하므로, delta 기반 event representation이 SNN 방향성과 잘 맞습니다.

### 3. Delta-Rate Hybrid SNN

```text
입력: moving_average 적용 UWB magnitude
인코딩: delta spike + rate/amplitude spike
모델: 같은 LIF 기반 spiking CNN
출력: respiration waveform reconstruction
```

Delta-only는 변화량 정보에 강하지만, 현재 신호가 어느 정도 크기인지에 대한 amplitude 정보가 약할 수 있습니다.
Hybrid 방식은 각 입력 bin에 대해 아래 3개 채널을 함께 사용합니다.

```text
positive delta spike
negative delta spike
rate/amplitude spike
```

즉, 변화 정보와 크기 정보를 같이 주는 방식입니다.

## 실험 결과

아래 결과는 MobiVital `sample.csv` 하나에서 나온 초기 방향성 실험입니다. 최종 성능이 아니라 방법 선택을 위한 참고값으로 봐야 합니다.

| 방법 | Bin 수 | Threshold scale | 입력 spike rate | hidden spike rate | RMSE | MAE | Corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 1 | - | - | - | 0.3195 | 0.2725 | 0.9785 |
| CNN | 5 | - | - | - | 0.3467 | 0.3032 | 0.9806 |
| Delta-SNN | 1 | 0.50 | 0.2961 | 0.1093 | 0.7420 | 0.6176 | 0.7377 |
| Delta-SNN | 1 | 0.75 | 0.2152 | 0.0885 | 0.7284 | 0.6063 | 0.7433 |
| Delta-SNN | 1 | 1.00 | 0.1517 | 0.0775 | 0.7512 | 0.6279 | 0.7332 |
| Delta-SNN | 5 | 0.75 | 0.1487 | 0.0959 | 0.8092 | 0.6712 | 0.6705 |

## 전처리/디노이징 비교

추가로 `none`, `moving_average`, `fft_bandpass` 세 가지 전처리를 비교했습니다.

전처리 방식:

```text
none
-> I/Q magnitude를 그대로 z-score 후 사용

moving_average
-> time axis 방향 5-sample causal moving average

fft_bandpass
-> FFT 기반 0.1~0.5 Hz respiration 대역만 통과
```

결과는 다음과 같습니다.

| 모델 | 전처리 | RMSE | MAE | Corr | 입력 spike rate | hidden spike rate |
|---|---|---:|---:|---:|---:|---:|
| CNN | none | 0.3195 | 0.2725 | 0.9785 | - | - |
| CNN | moving_average | 0.3732 | 0.3232 | 0.9737 | - | - |
| CNN | fft_bandpass | 0.3927 | 0.2923 | 0.9474 | - | - |
| Delta-SNN | none | 0.7284 | 0.6063 | 0.7433 | 0.2152 | 0.0885 |
| Delta-SNN | moving_average | 0.5702 | 0.4245 | 0.8470 | 0.2119 | 0.1136 |
| Delta-SNN | fft_bandpass | 0.5836 | 0.4684 | 0.8334 | 0.3135 | 0.1349 |

## Delta vs Delta-Rate Hybrid 비교

최근 SNN 시계열 처리에서는 변화량 기반 event 정보와 amplitude/rate 정보를 함께 쓰는 hybrid encoding이 자주 후보로 올라옵니다. 이를 확인하기 위해 가장 좋았던 SNN 조건인 `moving_average + threshold-scale 0.75`에서 delta-only와 delta-rate hybrid를 비교했습니다.

| 모델 | 인코딩 | 전처리 | RMSE | MAE | Corr | 입력 spike rate | hidden spike rate |
|---|---|---|---:|---:|---:|---:|---:|
| SNN | delta | moving_average | 0.5702 | 0.4245 | 0.8470 | 0.2119 | 0.1136 |
| SNN | delta-rate hybrid | moving_average | 0.4986 | 0.4261 | 0.9223 | 0.2555 | 0.1389 |

이번 샘플에서는 hybrid encoding이 correlation을 크게 올렸습니다.

```text
delta-only corr: 0.8470
hybrid corr:     0.9223
```

대신 입력 spike rate와 hidden spike rate도 증가했습니다.

```text
input spike rate:  0.2119 -> 0.2555
hidden spike rate: 0.1136 -> 0.1389
```

따라서 hybrid는 정확도는 더 좋지만 activity 비용이 조금 더 드는 방식으로 볼 수 있습니다.

현 시점 추천:

```text
SNN 기본 baseline: moving_average + delta
SNN 제안 후보: moving_average + delta-rate hybrid
최종 비교: 정확도와 spike activity를 함께 평가
```

### 디노이징 비교 해석

CNN은 디노이징을 하지 않은 `none`이 가장 좋았습니다. 이는 CNN이 clean sample에서는 원 신호의 세부 변화를 잘 활용한다는 뜻으로 볼 수 있습니다.

반대로 Delta-SNN은 `moving_average` 전처리에서 성능이 크게 좋아졌습니다.

```text
Delta-SNN none           -> RMSE 0.7284, Corr 0.7433
Delta-SNN moving_average -> RMSE 0.5702, Corr 0.8470
```

즉, SNN에는 연속 신호를 바로 spike로 바꾸는 것보다, 작은 고주파 변동을 약간 줄인 뒤 delta spike encoding을 적용하는 편이 더 좋아 보입니다.

FFT band-pass도 SNN 성능을 올리긴 했지만, 입력 spike rate와 hidden spike rate가 같이 증가했습니다. 따라서 현 시점에서는 `moving_average + delta spike`가 더 좋은 출발점으로 보입니다.

## 결과 해석

### CNN은 clean 데이터에서 확실히 강함

CNN baseline은 `RMSE 0.3195`, `Corr 0.9785`로 매우 좋은 결과가 나왔습니다.

이는 이상한 결과가 아니라, continuous UWB magnitude를 그대로 쓰는 CNN이 clean waveform reconstruction에서 강하다는 뜻입니다. 따라서 CNN은 최종 비교에서 반드시 baseline으로 두는 것이 좋습니다.

### Delta-SNN은 정확도는 낮지만 spike activity가 낮음

Delta-SNN은 `threshold-scale 0.75` 기준으로 `RMSE 0.7284`, `Corr 0.7433`이 나왔습니다.

정확도만 보면 CNN보다 약합니다. 하지만 입력 spike rate가 약 `0.2152`, hidden spike rate가 약 `0.0885`로 낮게 나왔습니다. 즉, sparse event-driven 처리라는 SNN의 장점은 확인할 수 있습니다.

### Threshold 0.75가 시작점으로 좋아 보임

threshold별 결과를 보면:

```text
0.50 -> spike가 많아짐, 성능 이득은 크지 않음
0.75 -> 정확도와 spike sparsity의 균형이 가장 좋아 보임
1.00 -> spike는 줄지만 정확도도 조금 떨어짐
```

따라서 다음 실험의 기본값은 `threshold-scale 0.75`로 두는 것이 좋습니다.

전처리까지 포함하면 다음 기본 조합을 우선 추천합니다.

```text
CNN baseline: preprocess none
SNN baseline: preprocess moving_average + threshold-scale 0.75
SNN improved candidate: preprocess moving_average + delta-rate hybrid + threshold-scale 0.75
```

### 5-bin 입력은 이번 샘플에서는 도움이 되지 않았음

best bin 주변 5개 bin을 넣었을 때 CNN은 correlation이 약간 올랐지만 RMSE는 커졌고, SNN은 오히려 성능이 떨어졌습니다.

이번 샘플 하나만 보면 multi-bin 입력이 무조건 좋은 선택은 아닙니다. 다만 실제 데이터가 늘어나면 subject별 거리 변화나 움직임에 도움이 될 수 있으므로, 완전히 버리기보다는 나중에 다시 검증하는 편이 좋습니다.

## 현재 결론

현재 단계에서 팀이 공유하면 좋은 결론은 아래입니다.

```text
1. CNN은 clean continuous UWB 입력에서 강력한 baseline이다.
2. 단순 Delta-SNN은 clean accuracy로 CNN을 이기지는 못한다.
3. 하지만 Delta-SNN은 낮은 spike activity를 보이므로 event-driven 전처리 방향성은 있다.
4. 졸업작품의 핵심은 "SNN이 CNN보다 무조건 정확하다"가 아니라,
   "UWB 신호를 spike-friendly하게 전처리하고 강건성/효율까지 함께 비교한다"로 잡는 것이 좋다.
```

## 추천하는 졸업작품 프레임

```text
기존 전처리 baseline
vs
CNN baseline
vs
Delta Spike + SNN
vs
개선된 adaptive spike encoding + SNN
```

평가 지표는 정확도만 보면 부족합니다.

```text
MAE / RMSE / correlation
noise 추가 후 성능 저하율
motion artifact 구간 성능
input spike rate
hidden spike rate
추론 시간 또는 연산량 proxy
```

이렇게 잡으면 SNN이 clean accuracy에서 CNN보다 낮더라도, 저활성 처리와 강건성 측면에서 연구 주제가 살아납니다.

## 다음 실험 제안

1. 데이터 수 늘리기
   - MobiVital 전체 데이터 또는 우리가 직접 수집할 UWB 데이터를 사용합니다.

2. noise/artifact 테스트 추가
   - Gaussian noise
   - random missing segment
   - motion spike artifact
   - SNR별 성능 저하율

3. spike encoding 비교
   - rate encoding
   - delta encoding
   - delta-rate hybrid encoding
   - adaptive threshold delta encoding

4. SNN 구조 개선
   - 현재는 아주 단순한 LIF-SNN입니다.
   - 이후 Spiking CNN, Spiking TCN, snnTorch 기반 모델 등을 비교할 수 있습니다.

5. 최종 팀 공유 메시지

```text
지금 실험은 SNN이 CNN보다 바로 정확하다는 증거가 아니라,
UWB 생체신호를 spike event로 바꿔서 저활성 처리할 수 있다는 가능성을 본 것이다.
따라서 앞으로는 정확도뿐 아니라 noise/artifact에 대한 성능 방어율과 spike efficiency를 같이 비교하는 방향으로 가는 것이 좋다.
```
