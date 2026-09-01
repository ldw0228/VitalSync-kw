# 멀티패스 호흡 파형 모델

비공개 데이터는 `data/waveform_training_dataset.npz`, 피험자 분할표는 `data/split_manifest.csv`에 둔다. 입력 하나는 `[72,200]`이며 3레이더×8경로×보정 I/Q/공통대역 위상, 20초, 10Hz다. 출력은 호흡 파형, 분당 호흡수, 움직임, 무호흡이다. `smoke`는 코드 연결 확인, `pilot`은 1개 피험자 독립 폴드, `formal`은 5폴드×3시드 정식 비교다. 원시데이터는 제외했고 집계결과는 `results/`에 있다.

```powershell
python train_waveform_cnn_snn.py --mode smoke
python train_waveform_cnn_snn.py --mode pilot
python train_waveform_cnn_snn.py --mode formal
```
