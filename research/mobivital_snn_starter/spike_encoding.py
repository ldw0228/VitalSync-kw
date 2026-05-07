from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray, axis: int | None = None, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


def infer_delta_threshold(
    dx: np.ndarray,
    threshold_scale: float = 0.75,
    threshold_mode: str = "std",
    threshold_percentile: float = 75.0,
    target_spike_rate: float = 0.2,
) -> float:
    """Infer an adaptive delta threshold from a signal derivative window."""
    abs_dx = np.abs(np.asarray(dx, dtype=np.float32))
    if threshold_mode == "std":
        threshold = threshold_scale * float(np.std(dx))
    elif threshold_mode == "mad":
        median = float(np.median(dx))
        mad = float(np.median(np.abs(dx - median)))
        threshold = threshold_scale * 1.4826 * mad
    elif threshold_mode == "percentile":
        threshold = threshold_scale * float(np.percentile(abs_dx, threshold_percentile))
    elif threshold_mode == "target_rate":
        keep_rate = float(np.clip(target_spike_rate, 1e-4, 0.9999))
        threshold = float(np.percentile(abs_dx, 100.0 * (1.0 - keep_rate)))
    else:
        raise ValueError(f"Unknown threshold mode: {threshold_mode}")
    return max(threshold, 1e-8)


def delta_spike_encode(
    x: np.ndarray,
    threshold: float | None = None,
    threshold_scale: float = 0.75,
    threshold_mode: str = "std",
    threshold_percentile: float = 75.0,
    target_spike_rate: float = 0.2,
    bipolar: bool = True,
) -> np.ndarray:
    """Encode temporal changes into spikes.

    Args:
        x: Array shaped [..., time].
        threshold: Absolute delta threshold. If None, use
            an adaptive value inferred from diff(x).
        threshold_scale: Scale used by std, mad, and percentile modes.
        threshold_mode: One of std, mad, percentile, target_rate.
        bipolar: If True, return two channels [positive, negative].
            If False, return a signed spike train {-1, 0, 1}.

    Returns:
        If bipolar: array shaped [..., 2, time].
        Else: array shaped [..., time].
    """
    x = np.asarray(x, dtype=np.float32)
    dx = np.diff(x, axis=-1, prepend=x[..., :1])
    if threshold is None:
        threshold = infer_delta_threshold(
            dx,
            threshold_scale=threshold_scale,
            threshold_mode=threshold_mode,
            threshold_percentile=threshold_percentile,
            target_spike_rate=target_spike_rate,
        )
    threshold = max(float(threshold), 1e-8)

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


def rate_spike_encode(
    x: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Encode signal amplitude into one binary spike channel.

    This keeps the same [channels, time] layout used by the 1D SNN starter.
    Values are min-max normalized per window and sampled once with a fixed seed.
    """
    x = np.asarray(x, dtype=np.float32)
    x_min = np.min(x, axis=-1, keepdims=True)
    x_max = np.max(x, axis=-1, keepdims=True)
    prob = (x - x_min) / (x_max - x_min + 1e-8)
    rng = np.random.default_rng(seed)
    return (rng.random(x.shape) < prob).astype(np.float32)


def level_crossing_encode(
    x: np.ndarray,
    levels: int = 5,
    level_min: float = -1.5,
    level_max: float = 1.5,
) -> np.ndarray:
    """Encode upward/downward crossings of fixed amplitude levels.

    Shape:
        input [channels, time] -> output [channels * levels * 2, time]
    """
    x = np.asarray(x, dtype=np.float32)
    if levels < 1:
        raise ValueError("levels must be >= 1")
    thresholds = np.linspace(level_min, level_max, levels, dtype=np.float32)
    prev = np.concatenate([x[..., :1], x[..., :-1]], axis=-1)
    encoded = []
    for level in thresholds:
        up = ((prev < level) & (x >= level)).astype(np.float32)
        down = ((prev >= level) & (x < level)).astype(np.float32)
        encoded.extend([up, down])
    return np.stack(encoded, axis=-2).reshape(-1, x.shape[-1]).astype(np.float32)


def delta_rate_hybrid_encode(
    x: np.ndarray,
    threshold: float | None = None,
    threshold_scale: float = 0.75,
    threshold_mode: str = "std",
    threshold_percentile: float = 75.0,
    target_spike_rate: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """Concatenate delta-event channels and rate/amplitude spike channels.

    For each input channel, output channels are:
    - positive delta spikes
    - negative delta spikes
    - rate/amplitude spikes

    Shape:
        input [channels, time] -> output [channels * 3, time]
    """
    delta = delta_spike_encode(
        x,
        threshold=threshold,
        threshold_scale=threshold_scale,
        threshold_mode=threshold_mode,
        threshold_percentile=threshold_percentile,
        target_spike_rate=target_spike_rate,
        bipolar=True,
    )
    delta = delta.reshape(-1, delta.shape[-1])
    rate = rate_spike_encode(x, seed=seed)
    return np.concatenate([delta, rate], axis=0).astype(np.float32)
