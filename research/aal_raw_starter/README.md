# AAL Raw UWB Starter

이 폴더는 Mendeley의 `Radar Human Breathing Dataset for AAL and Search and Rescue`를 이용해 raw radar에서 breathing signal을 만드는 과정을 확인하기 위한 정리입니다.

MobiVital은 이미 UWB I/Q time-series와 label이 정리된 형태라 SNN/spike encoding 실험에 좋습니다. 반면 이 AAL 데이터셋은 raw radar frame에서 processed breathing signal을 만드는 흐름을 연습하기에 더 적합합니다.

## 사용한 데이터셋

- Dataset: Radar Human Breathing Dataset for AAL and Search and Rescue
- DOI: `10.17632/cbj37wdsdj.2`
- Source: Mendeley Data
- Local sample path used during inspection:

```text
C:\Users\hai\Desktop\uwb_aal_raw
```

데이터 전체는 3180개 파일, 약 1.6GB 수준이라 처음에는 전체 다운로드 대신 한 조건의 샘플만 받았습니다.

## 받은 샘플 조건

```text
DeltaR=10cm
Angle=0
Band1
Supine
Trial1
```

받은 주요 파일:

```text
AboutData.pdf
DataCode.m
CalibrationDataVisualization.m
DataTableBandwidth1.mat
Calibration_Band1_DeltaR=10cm.mat
DeltaR=10cm_Angle=0_Band1_Supine_Trial1.mat
BS_DeltaR=10cm_Angle=0_Band1_Supine_Trial1.mat
Ref_DeltaR=10cm_Angle=0__Band1_Supine_Trial1.mat
FilteredBreath_Ref_DeltaR=10cm_Angle=0__Band1_Supine_Trial1.mat
FilteredBreathRadar_DeltaR=10cm_Angle=0_Band1_Supine_Trial1.mat
Spectrum_Ref_DeltaR=10cm_Angle=0__Band1_Supine_Trial1.mat
SpectrumRadar_DeltaR=10cm_Angle=0_Band1_Supine_Trial1.mat
```

## 파일 구조 확인 결과

| 파일 | 변수 | shape | 의미 |
|---|---|---:|---|
| `DeltaR=10cm_Angle=0_Band1_Supine_Trial1.mat` | `bScan` | `[1099, 256]` | raw radar frame, slow-time x fast-time/range |
| `Calibration_Band1_DeltaR=10cm.mat` | `bScan` | `[1099, 256]` | no-human calibration radar frame |
| `BS_DeltaR=10cm_Angle=0_Band1_Supine_Trial1.mat` | `BS` | `[256, 1099]` | background-subtracted radar data |
| `FilteredBreathRadar_...mat` | `BreathSignalRadar` | `[1, 1099]` | radar에서 추출한 breathing signal |
| `Ref_...mat` | `Ref` | `[1, 180]` | lidar reference signal |
| `FilteredBreath_Ref_...mat` | `Ref` | `[1, 180]` | filtering된 reference breathing signal |

## DataCode.m 기준 처리 흐름

제공된 `DataCode.m`의 핵심 처리는 아래와 같습니다.

```text
1. raw radar bScan 로드
2. BS = detrend(bScan, 1)
3. BS = BS'
4. PreRangeIndx=100 이후 range 구간에서 max response 기반 target range index 선택
5. 선택된 range bin의 slow-time signal 추출
6. 1.5 Hz low-pass filtering
7. BreathSignalRadar 저장
8. lidar Ref도 low-pass 후 spectrum 비교
```

## Python 재현 결과

`inspect_aal_sample.py`로 위 흐름을 Python에서 재현했습니다.

결과:

```text
raw bScan shape: [1099, 256]
background-subtracted BS shape: [256, 1099]
BS reproduction corr with provided BS: 1.0
estimated range index: 184 (0-based)
estimated range: 약 0.7176 m
radar sampling rate: 약 18.76 Hz
reference sampling rate: 약 3.08 Hz
BreathSignalRadar reproduction corr: 약 0.99975
```

즉, 제공된 MATLAB `DataCode.m`의 핵심 raw-to-processed pipeline은 Python에서도 거의 그대로 재현됩니다.

## 왜 중요한가

이 결과는 우리가 직접 UWB raw와 BIOPAC/respiration reference를 수집했을 때 필요한 앞단 pipeline을 미리 연습할 수 있다는 뜻입니다.

MobiVital에서 한 일:

```text
정리된 UWB I/Q
-> spike encoding
-> CNN/SNN 비교
```

AAL raw starter에서 확인한 일:

```text
raw radar frame
-> background removal
-> range bin selection
-> radar breathing signal extraction
```

따라서 최종 졸업작품 pipeline은 아래처럼 연결할 수 있습니다.

```text
우리 UWB raw + BIOPAC raw
-> raw parser / synchronization
-> background removal
-> target range bin selection
-> breathing signal extraction
-> spike-friendly preprocessing
-> delta/adaptive-delta spike encoding
-> CNN/SNN 비교
```

## 다음 작업

1. AAL 샘플을 `SyncData-like` 구조로 변환

```text
fs_radar
fs_reference
raw_bscan
background_subtracted
selected_range_bin
radar_breath_signal
reference_breath_signal
condition metadata
```

현재는 이 단계를 `export_aal_syncdata_like.py`로 구현했습니다.

```bash
python export_aal_syncdata_like.py --root C:\Users\hai\Desktop\uwb_aal_raw
```

생성 산출물:

```text
C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat
C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like_summary.json
```

MATLAB 구조는 대학원생 코드의 최종 산출물과 비슷하게 맞췄습니다.

| 필드 | 의미 |
|---|---|
| `Fs_uwb` | AAL radar frame rate |
| `Fs_biopac` | AAL reference sampling rate. 실제 BIOPAC은 아니지만 호환을 위해 reference를 매핑 |
| `biopac_resp` | AAL filtered lidar/reference respiration |
| `radar_resp` | raw bScan에서 추출한 radar breathing signal |
| `com_row` | background-subtracted UWB matrix를 row 방향 z-score 정규화 |
| `com_col` | background-subtracted UWB matrix를 column 방향 z-score 정규화 |
| `tv_row`, `tv_col` | AAL은 radar가 하나뿐이라 `com_*`를 호환용으로 복사 |

주의할 점:

```text
AAL에는 실제 BIOPAC이 없습니다.
따라서 biopac_resp는 AAL의 lidar/reference respiration을 대학원생 코드 구조에 맞춰 넣은 것입니다.
우리 실제 실험에서는 이 자리에 BIOPAC respiration raw/filter 결과가 들어갑니다.
```

2. AAL SyncData-like 파일에 기존 SNN 방법론 적용

```text
radar_breath_signal
-> moving_average
-> adaptive delta spike encoding
-> SNN
```

현재는 `train_syncdata_snn.py`로 구현했습니다.

```bash
python train_syncdata_snn.py --mat C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat --target-field radar_resp --bin-index 184 --preprocess moving_average --encode delta_rate_hybrid --model lif_cnn

python train_syncdata_snn.py --mat C:\Users\hai\Desktop\uwb_aal_raw\syncdata_like\AAL_UWB_Biopac_SyncData_like.mat --target-field radar_resp --bin-index 184 --preprocess moving_average --encode delta_rate_hybrid --model spiking_tcn
```

초기 결과:

| Target | Model | Encoding | RMSE | MAE | Corr | input spikes/sec | hidden spike rate |
|---|---|---|---:|---:|---:|---:|---:|
| `biopac_resp` | LIF-CNN | delta-rate hybrid | 0.8999 | 0.7604 | -0.0876 | 16.9 | 0.0866 |
| `radar_resp` | LIF-CNN | delta-rate hybrid | 0.3046 | 0.2468 | 0.9618 | 21.5 | 0.2523 |
| `radar_resp` | Spiking TCN | delta-rate hybrid | 0.2770 | 0.2206 | 0.9645 | 21.5 | 0.1825 |

해석:

```text
AAL sample에서는 lidar/reference와 radar breath의 직접 정렬 상관이 낮아 biopac_resp target 학습은 좋지 않았습니다.
반면 raw UWB matrix에서 추출한 radar_resp target은 hybrid SNN으로 잘 재구성됩니다.
즉 AAL은 BIOPAC supervised 최종 검증용이라기보다,
"raw UWB -> 대학원생식 SyncData-like 산출물 -> spike/SNN 방법론 적용"을 연습하는 용도에 더 적합합니다.
```

3. CNN/SNN 전체 비교

SyncData-like 파일을 기준으로 CNN baseline과 SNN 후보를 같은 target에서 비교했습니다. 자세한 표와 추천 방법론은 아래 문서에 정리했습니다.

```text
docs/aal_syncdata_cnn_snn_comparison.md
```

요약:

| 역할 | 현재 추천 |
|---|---|
| 정확도 baseline | continuous CNN + none |
| SNN baseline | rate, delta |
| SNN 효율 비교군 | level-crossing |
| SNN 제안 방법 | delta-rate hybrid + Spiking TCN |

현재 AAL `radar_resp` target에서는 CNN이 가장 정확하고, SNN 내부에서는 `delta-rate hybrid + Spiking TCN`이 가장 좋았습니다. 실제 BIOPAC raw가 들어오면 target을 `biopac_resp`로 바꿔 같은 비교를 반복하면 됩니다.

4. 실제 데이터 수집 전 저장 포맷 확정

```text
UWB raw frame
BIOPAC/respiration raw
timestamps
condition metadata
processed intermediate
final window dataset
```
