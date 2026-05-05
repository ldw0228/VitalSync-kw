# UWB Spike Encoding Direction Check

Date: 2026-05-05

This is an early direction-finding experiment, not a final graduation-project result.
The goal was to see whether a MobiVital-style UWB I/Q signal can be converted into
spike inputs and compared against a normal CNN baseline.

## Data

- Dataset sample: MobiVital `sample.csv`
- Shape: 1500 rows x 254 columns
- Sampling rate: 50 Hz
- Duration: 30 seconds
- UWB I: columns 13-132
- UWB Q: columns 133-252
- Label: respiration waveform, column 253

Feature construction:

```text
I/Q -> magnitude = sqrt(I^2 + Q^2)
best range bin selected by correlation with respiration
10-second windows, 2-second stride
```

Best bin in this sample:

```text
bin 51
distance estimate: about 2.87 m
correlation with respiration: about 0.936
```

## Compared Methods

1. CNN baseline
   - Input: continuous UWB magnitude
   - Model: small 1D CNN
   - Meaning: normal deep-learning upper baseline

2. Delta-SNN
   - Input: delta spike encoded UWB magnitude
   - Spike channels: positive / negative events
   - Model: small LIF-based spiking CNN with surrogate gradient
   - Meaning: event-driven, low-activity SNN baseline

## Results

| Method | Bins | Threshold scale | Input spike rate | Hidden spike rate | RMSE | MAE | Corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 1 | - | - | - | 0.3195 | 0.2725 | 0.9785 |
| CNN | 5 | - | - | - | 0.3467 | 0.3032 | 0.9806 |
| Delta-SNN | 1 | 0.50 | 0.2961 | 0.1093 | 0.7420 | 0.6176 | 0.7377 |
| Delta-SNN | 1 | 0.75 | 0.2152 | 0.0885 | 0.7284 | 0.6063 | 0.7433 |
| Delta-SNN | 1 | 1.00 | 0.1517 | 0.0775 | 0.7512 | 0.6279 | 0.7332 |
| Delta-SNN | 5 | 0.75 | 0.1487 | 0.0959 | 0.8092 | 0.6712 | 0.6705 |

## Current Interpretation

The CNN baseline is much stronger on clean continuous input. That is expected.
It gives us a useful upper reference.

The Delta-SNN is weaker in direct reconstruction accuracy, but it uses sparse
event-style inputs and low hidden spike activity. This makes it a reasonable
candidate for the graduation-project angle if we compare not only clean accuracy
but also robustness and efficiency.

Early observation:

```text
threshold 0.75 looks like a good starting point.
threshold 0.5 creates more spikes but does not improve much.
threshold 1.0 creates fewer spikes but loses a little accuracy.
5-bin input did not help in this tiny sample.
```

## Recommended Direction

For the project direction, frame it like this:

```text
CNN baseline = high-accuracy continuous-signal reference
Delta-SNN = low-activity event-based representation baseline
Proposed work = improve spike encoding and test robustness/noise defense
```

Next experiments should focus on:

1. More data, preferably multiple subjects/sessions.
2. Noise and artifact tests.
3. Adaptive thresholding instead of fixed delta threshold.
4. Compare delta encoding, rate encoding, and delta-rate hybrid encoding.
5. Use spike rate and robustness drop together with RMSE/correlation.

## Suggested Team Message

We tested a small MobiVital UWB sample to check the project direction. A normal
1D CNN on continuous UWB magnitude reconstructs respiration very well, so it can
serve as the accuracy baseline. A simple Delta Spike + LIF-SNN model is less
accurate on clean data, but it runs with sparse spike activity, which supports
the idea of studying spike-friendly preprocessing. The direction should not be
"SNN beats CNN on clean data immediately"; it should be "spike encoding can offer
event-driven, low-activity processing, and we compare its robustness and efficiency
against CNN baselines."

