# MobiVital SNN Starter

This starter follows path B: use the MobiVital README only as the column
definition, then build our own PyTorch-ready loader and spike encoding
pipeline.

## Data Assumption

Each MobiVital CSV row is one timestamp.

- Sampling rate: 50 Hz
- Columns 1-6: IMU
- Columns 13-132: UWB I, 120 range bins
- Columns 133-252: UWB Q, 120 range bins
- Column 253: respiration waveform label
- Column 254: pulse waveform label

Column numbers above are 1-based as described by the dataset README.

## Current Pipeline

```text
sample.csv
-> split UWB I/Q
-> magnitude or phase feature
-> select range bin
-> sliding windows
-> delta spike encoding
-> PyTorch Dataset output
```

## Quick Start

```powershell
python quick_inspect.py --csv C:\Users\hai\Desktop\uwb_sample\sample.csv
```

If Python is not available on the machine yet, install Python first and then:

```powershell
pip install -r requirements.txt
```

On Windows, install PyTorch from the CPU wheel index:

```powershell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

## Why This Direction

The senior-lab MATLAB code gives a useful UWB/BIOPAC preprocessing baseline.
For the graduation project, our contribution should be the spike-friendly
representation and SNN comparison:

```text
existing/preprocessed radar signal
vs
delta or delta-rate spike representation + SNN
```
