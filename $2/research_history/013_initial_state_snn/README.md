# 013 — Initial state SNN

Ten subjects produced 840 windows: 341 S01 angle and 499 S02 state. Leave-one-subject-out balanced accuracy/macro-F1 were SNN `0.484/0.409` vs ANN `0.500/0.444` for angle, and SNN `0.372/0.348` vs ANN `0.407/0.367` for state. Both were weak. The small fast model was a prototype on summarized windows, not a literature-scale final SNN, and the target shifted to physiological waveform/RR.
