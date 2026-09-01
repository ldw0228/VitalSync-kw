# 019 — Multi-path waveform dataset

Twenty-seven subjects yielded 1,564 overlapping 20-s windows at 10 Hz; 1,325 have waveform/RR targets. Input shape is 72×200: `3 radars × 8 paths × [whitened I, whitened Q, common-band phase]`. There are 182 motion-positive and 50 apnea-positive windows. State labels select/audit windows but are not model inputs; BIOPAC is target/evaluation only.
