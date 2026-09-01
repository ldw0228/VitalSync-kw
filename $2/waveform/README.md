# Multi-path waveform model

Private data are placed under `data/` as `waveform_training_dataset.npz` and `split_manifest.csv`. Each input is `[72,200]`: three radars, eight paths, corrected I/Q/common-band phase, 20 s at 10 Hz. Outputs are respiration waveform, respiratory rate, motion and apnea. `smoke`, `pilot`, and `formal` modes run 3 epochs; one fold with early stopping; or five folds × three seeds respectively. Raw data and checkpoints are excluded. Aggregate dataset/pilot results are in `results/`.

```powershell
python train_waveform_cnn_snn.py --mode smoke
python train_waveform_cnn_snn.py --mode pilot
python train_waveform_cnn_snn.py --mode formal
```
