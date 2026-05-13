from __future__ import annotations

import numpy as np


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a causal moving average along the time axis."""
    if window <= 1:
        return x.astype(np.float32, copy=True)
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    acc = np.zeros(x.shape[1:], dtype=np.float32)
    for i in range(x.shape[0]):
        acc += x[i]
        if i >= window:
            acc -= x[i - window]
        out[i] = acc / min(window, i + 1)
    return out


def fft_bandpass(
    x: np.ndarray,
    fs: float,
    low_hz: float = 0.1,
    high_hz: float = 0.5,
) -> np.ndarray:
    """Simple FFT band-pass filter along the time axis.

    This is intentionally dependency-free so the starter can run without SciPy.
    It is a direction-check filter, not a production medical signal pipeline.
    """
    x = np.asarray(x, dtype=np.float32)
    centered = x - np.mean(x, axis=0, keepdims=True)
    spec = np.fft.rfft(centered, axis=0)
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / fs)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spec[~mask] = 0
    return np.fft.irfft(spec, n=x.shape[0], axis=0).astype(np.float32)


def apply_preprocess(
    x: np.ndarray,
    mode: str = "none",
    fs: float = 50.0,
) -> np.ndarray:
    if mode == "none":
        return x.astype(np.float32, copy=True)
    if mode == "moving_average":
        return moving_average(x, window=5)
    if mode == "fft_bandpass":
        return fft_bandpass(x, fs=fs, low_hz=0.1, high_hz=0.5)
    raise ValueError(f"Unknown preprocess mode: {mode}")

