# Research history: 001–021

This is a reconstructed lab record, not a verbatim conversation export. Completed evidence, interpretation, and future work are kept separate.

| No. | Stage | Key outcome |
|---:|---|---|
| 001 | Project scope | Use guide, workbook, MATLAB sync, UWB and BIOPAC; BIOPAC is ground truth only |
| 002 | Data inventory | 27 usable for final S01/S02 modeling; S01/S22 partial, S24 UWB missing |
| 003 | Reverse sync | Align by RSP chest-press candidates and three-radar motion; keep manual cases |
| 004 | MATLAB revalidation | Raw MATLAB/batch marker times matched for all 29 available sets |
| 005 | Marker audit | Protocol 20 markers for participants S01–S03, 22 from S04; raw counts are only candidates |
| 006 | Experiment 1 | Orientation is useful as a robustness factor, weak as final classification target |
| 007 | Experiment 2 | Primary breathing/apnea/motion scenario |
| 008–012 | Experiments 3–7 | Marker audit completed; only roundtrip has preliminary analysis |
| 013 | Initial state SNN | ANN and SNN both weak; abandoned as final direction |
| 014 | Hold heart rate | Interesting but heuristic and exploratory |
| 015 | 10-subject RR baseline | Strong development result, not independent evidence |
| 016 | All-27 boundary lock | Protocol and synchronized evidence fixed before formal training |
| 017 | All-27 ANN/SNN CV | Best compact SNN MAE 1.426; DSP baseline 0.992, so learned model lost |
| 018 | Feedback integration | Preserve multiple paths and reconstruct waveform |
| 019 | Waveform dataset | 27 subjects, 1,564 windows, 72×200 input |
| 020 | Waveform pilot | CNN-GRU currently ahead of CNN-SNN; neither is final |
| 021 | Current decision | Tune, lock, then run five folds × three seeds and compare with DSP |

## Common definitions

- Protocol marker count: designed start/end boundaries.
- Automatic raw count: threshold peaks from BIOPAC RSP; not the true count.
- Selected marker: candidate accepted using order, duration, RSP shape, radar motion and alternatives.
- Sync equation: `biopac_time = radar_time + offset`.
- Deployment input: UWB only. BIOPAC supplies synchronization/labels/targets/evaluation during development.
