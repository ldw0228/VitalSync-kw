# 020 — Waveform pilot

Both models use a shared three-block 1D CNN per radar and learned three-radar softmax gate. CNN-GRU has 108,949 parameters; CNN-SNN uses recurrent LIF 128+64 and has 95,509. One fold/seed pilot: CNN-GRU best epoch 20 of 35, RR MAE 3.843, waveform correlation 0.337; CNN-SNN best epoch 10 of 25, RR MAE 4.595, correlation 0.268. This validates execution only; it currently favors GRU and remains worse than DSP.
