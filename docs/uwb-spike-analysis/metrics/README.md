# Aggregate Metrics

이 폴더에는 팀원이 결과 수치를 직접 확인할 수 있도록 aggregate 평가표만 모아두었습니다.

원본 `.mat`, subject별/window별 상세 CSV, 학습 dataset, 모델 weight는 포함하지 않았습니다.

## Files

| File | What it contains |
|---|---|
| `spike_baseline_results.csv` | Spike encoding label classification threshold/lambda sweep |
| `spike_per_label_metrics.csv` | Best spike baseline의 class별 precision/recall/F1 |
| `rr_metrics_summary.csv` | BIOPAC RR 기준 UWB/BPF/spike RR 추정 성능 |
| `shift_time_summary.csv` | 30초 window / 15초 stride time-shift sweep summary |
| `shift_range_summary.csv` | Range-bin shift sweep summary |
| `extended_best_shift_by_method.csv` | 30초 window / 1초 stride tracking에서 method별 best shift |
| `extended_time_shift_tracking_metrics.csv` | 1초 stride time-shift sweep의 MAE/RMSE/trend/correlation 지표 |
| `extended_zero_shift_function_metrics.csv` | FFT/autocorr/peak-interval 등 RR 추정 함수별 지표 |
| `snn_holdout_baseline_metrics.csv` | SNN holdout과 비교한 ridge/majority baseline |
| `snn_training_history.csv` | SNN epoch별 train/test accuracy 및 macro F1 |
| `snn_confusion_matrix.csv` | SNN holdout test confusion matrix |

## Key Numbers

| Question | Best current result |
|---|---:|
| Spike label classification macro F1 | 0.859 |
| RR MAE, original BPF COM FFT baseline | 4.79 bpm |
| RR MAE, best extended method | 3.87 bpm |
| RR within 5 bpm, best extended method | 0.709 |
| RR trend accuracy, best extended method | 0.295 |
| SNN holdout test macro F1 | 0.749 |

## Interpretation

- Spike encoding is useful for label classification.
- For RR estimation, `BPF mean + peak_interval` currently gives the lowest MAE.
- Time shift correction alone barely improves MAE, so synchronization is probably not the main bottleneck.
- RR value estimation is moderately useful, but temporal trend tracking is still weak.
- Next work should focus on ROI stabilization, movement rejection, and repeated subject-holdout evaluation.
