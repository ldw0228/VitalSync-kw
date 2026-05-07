# AAL SyncData-like CNN/SNN 비교 정리

이 문서는 AAL raw UWB 샘플을 대학원생 코드의 `UWB_Biopac_SyncData.mat`와 비슷한 중간 산출물로 변환한 뒤, CNN/SNN 방법론을 비교한 결과입니다.

주의: AAL에는 실제 BIOPAC이 없으므로 `biopac_resp`는 AAL의 lidar/reference respiration을 호환용으로 넣은 필드입니다. 현재 AAL 샘플에서는 radar breath와 lidar/reference의 직접 정렬 상관이 낮았습니다. 따라서 아래 비교는 `target-field=radar_resp`, 즉 raw UWB에서 추출한 radar breathing signal을 target으로 둔 raw pipeline 검증용 결과입니다.

## 전체 프로세스

```text
AAL raw bScan
-> detrend 기반 background subtraction
-> row/column normalization
-> UWB_Biopac_SyncData-like .mat 생성
-> CNN branch / SNN branch 비교
```

대학원생 코드와 맞춘 중간 산출물:

```text
UWB_Biopac_SyncData.Fs_uwb
UWB_Biopac_SyncData.Fs_biopac
UWB_Biopac_SyncData.biopac_resp
UWB_Biopac_SyncData.com_row
UWB_Biopac_SyncData.com_col
UWB_Biopac_SyncData.tv_row
UWB_Biopac_SyncData.tv_col
```

AAL에서 추가한 호환 필드:

```text
UWB_Biopac_SyncData.radar_resp
UWB_Biopac_SyncData.Fs_radar_resp
```

## 실행 명령

SyncData-like 파일 생성:

```bash
python research/aal_raw_starter/export_aal_syncdata_like.py --root C:\Users\hai\Desktop\uwb_aal_raw
```

CNN baseline:

```bash
python research/aal_raw_starter/train_syncdata_cnn.py --mat C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat --target-field radar_resp --bin-index 184 --preprocess none
python research/aal_raw_starter/train_syncdata_cnn.py --mat C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat --target-field radar_resp --bin-index 184 --preprocess moving_average
python research/aal_raw_starter/train_syncdata_cnn.py --mat C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat --target-field radar_resp --bin-index 184 --preprocess fft_bandpass
```

SNN 비교:

```bash
python research/aal_raw_starter/train_syncdata_snn.py --mat C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat --target-field radar_resp --bin-index 184 --preprocess moving_average --encode rate --model lif_cnn
python research/aal_raw_starter/train_syncdata_snn.py --mat C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat --target-field radar_resp --bin-index 184 --preprocess moving_average --encode delta --threshold-scale 0.75 --model lif_cnn
python research/aal_raw_starter/train_syncdata_snn.py --mat C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat --target-field radar_resp --bin-index 184 --preprocess moving_average --encode delta_rate_hybrid --threshold-scale 0.75 --model lif_cnn
python research/aal_raw_starter/train_syncdata_snn.py --mat C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat --target-field radar_resp --bin-index 184 --preprocess moving_average --encode delta_rate_hybrid --threshold-mode target_rate --target-spike-rate 0.2 --model lif_cnn
python research/aal_raw_starter/train_syncdata_snn.py --mat C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat --target-field radar_resp --bin-index 184 --preprocess moving_average --encode level_crossing --levels 5 --model lif_cnn
python research/aal_raw_starter/train_syncdata_snn.py --mat C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat --target-field radar_resp --bin-index 184 --preprocess moving_average --encode delta_rate_hybrid --threshold-scale 0.75 --model spiking_tcn
```

## 비교 결과

공통 조건:

```text
Dataset: AAL sample condition
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

## 해석

정확도만 보면 continuous CNN이 가장 강합니다. 특히 `CNN + none`이 RMSE 0.0721, Corr 0.9962로 가장 좋았습니다. 이 AAL target은 raw UWB에서 추출한 radar respiration이므로, CNN이 같은 continuous matrix에서 target waveform을 복원하기 쉬운 조건입니다.

SNN끼리 비교하면 `delta-rate hybrid + Spiking TCN`이 가장 좋았습니다. LIF-CNN보다 RMSE가 낮고, hidden spike rate도 hybrid LIF-CNN보다 낮았습니다.

`level-crossing`은 정확도는 떨어지지만 input spikes/sec가 2.5로 매우 낮습니다. 따라서 최종 정확도 후보라기보다는 ultra-low-activity 비교군으로 가치가 있습니다.

`adaptive hybrid`는 target spike rate를 0.2로 둔 실험입니다. 고정 hybrid보다 spike activity는 줄지만 성능도 낮아졌습니다. 즉 adaptive threshold는 하나의 정답이라기보다 정확도와 spike activity를 조절하는 knob로 봐야 합니다.

## 실제 raw 수집 후 추천 방법론

실제 UWB + BIOPAC raw가 들어오면 아래 순서로 가는 것이 좋습니다.

```text
1. 대학원생 코드 스타일로 SyncData-like 중간 산출물 생성
2. BIOPAC respiration을 target으로 확정
3. CNN continuous baseline 먼저 학습
4. SNN rate/delta/hybrid 비교
5. hybrid + Spiking TCN을 제안 방법으로 평가
6. noise/artifact 상황에서 성능 방어율과 spike efficiency 평가
```

최종 추천 비교군:

| 역할 | 방법 |
|---|---|
| 정확도 상한 baseline | continuous CNN + none 또는 bandpass |
| SNN 기본 baseline | rate + LIF-CNN |
| SNN event baseline | delta + LIF-CNN |
| SNN 효율 비교군 | level-crossing + LIF-CNN |
| 제안 방법 | adaptive delta-rate hybrid + Spiking TCN |

실제 데이터에서는 AAL과 달리 target이 `radar_resp`가 아니라 실제 `BIOPAC respiration`이어야 합니다. 그래서 CNN이 여전히 압도적으로 좋을지, SNN이 noise/artifact 상황에서 더 잘 방어할지는 실제 수집 데이터로 다시 검증해야 합니다.
