from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray, axis: int | None = None, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


def delta_spike_encode(
    x: np.ndarray,
    threshold: float | None = None,
    threshold_scale: float = 0.75,
    bipolar: bool = True,
) -> np.ndarray:
    """Encode temporal changes into spikes.

    Args:
        x: Array shaped [..., time].
        threshold: Absolute delta threshold. If None, use
            threshold_scale * std(diff(x)).
        threshold_scale: Scale used when threshold is inferred.
        bipolar: If True, return two channels [positive, negative].
            If False, return a signed spike train {-1, 0, 1}.

    Returns:
        If bipolar: array shaped [..., 2, time].
        Else: array shaped [..., time].
    """
    x = np.asarray(x, dtype=np.float32)
    dx = np.diff(x, axis=-1, prepend=x[..., :1])
    if threshold is None:
        threshold = float(threshold_scale * np.std(dx))
    if threshold <= 0:
        threshold = 1e-8

    pos = (dx > threshold).astype(np.float32)
    neg = (dx < -threshold).astype(np.float32)
    if bipolar:
        return np.stack([pos, neg], axis=-2)
    return pos - neg


def rate_encode(
    x: np.ndarray,
    steps: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Simple rate coding baseline.

    Normalizes x to [0, 1] and samples Bernoulli spikes over `steps`.
    Output shape is [steps, *x.shape].
    """
    x = np.asarray(x, dtype=np.float32)
    x_min = np.min(x)
    x_max = np.max(x)
    prob = (x - x_min) / (x_max - x_min + 1e-8)
    rng = np.random.default_rng(seed)
    return (rng.random((steps, *x.shape)) < prob).astype(np.float32)

