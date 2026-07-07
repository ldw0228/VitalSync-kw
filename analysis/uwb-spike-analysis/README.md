# UWB Spike Analysis Scripts

이 폴더는 UWB/BIOPAC 분석을 재현하기 위한 스크립트 모음입니다.

원본 데이터는 repo에 포함하지 않습니다. 로컬 환경에서 `D:\UWB\UWB_BIOPAC_DATA_0601` 경로에 데이터가 있다고 가정합니다.

## MATLAB scripts

| File | Purpose |
|---|---|
| `matlab/spike_encoding_baseline.m` | UWB delta ON/OFF spike feature 생성 및 label classification baseline |
| `matlab/spike_rr_biopac_baseline.m` | BIOPAC RR 기준으로 UWB/BPF/spike RR 비교 |
| `matlab/uwb_shift_analysis.m` | 30초 window / 15초 stride shift 분석 |
| `matlab/extended_shift_tracking_eval.m` | 30초 window / 1초 stride 확장 tracking 평가 |

## Python scripts

| File | Purpose |
|---|---|
| `python/train_snn_holdout.py` | `snntorch` 기반 SNN holdout 학습 |
| `python/requirements_snn.txt` | SNN 실험용 Python dependency |

## Recommended order

```powershell
matlab -batch "run('analysis/uwb-spike-analysis/matlab/spike_encoding_baseline.m')"
matlab -batch "run('analysis/uwb-spike-analysis/matlab/spike_rr_biopac_baseline.m')"
matlab -batch "run('analysis/uwb-spike-analysis/matlab/extended_shift_tracking_eval.m')"
```

SNN은 먼저 MATLAB에서 holdout dataset을 export한 뒤 실행하는 흐름을 권장합니다. 현재 repo에는 원본 데이터와 export된 `.mat` dataset을 포함하지 않습니다.
