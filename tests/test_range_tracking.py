from __future__ import annotations

import numpy as np

from snn_rr.range_tracking import (
    causal_range_track,
    compare_iq_layouts,
    complex_frames,
    fuse_range_track_window_features,
    range_track_window_features,
)


def _split_half_scene(
    *,
    frames: int = 480,
    fs: float = 20.0,
    moving: bool = False,
    disappear_after: int | None = None,
    second_target: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260830)
    bins = 91
    time = np.arange(frames) / fs
    kernel = np.ones(9, dtype=float) / 9.0
    clutter = (
        np.convolve(rng.normal(0.0, 4e-4, bins), kernel, mode="same")
        + 1j * np.convolve(rng.normal(0.0, 4e-4, bins), kernel, mode="same")
    ).astype(np.complex64)
    scene = np.tile(clutter, (frames, 1)).astype(np.complex64)
    scene += (
        rng.normal(0.0, 1e-5, scene.shape)
        + 1j * rng.normal(0.0, 1e-5, scene.shape)
    ).astype(np.complex64)
    if moving:
        path = np.rint(np.linspace(18, 58, frames)).astype(int)
    else:
        path = np.full(frames, 37, dtype=int)
    phase = 0.9 * np.sin(2 * np.pi * 0.27 * time)
    active_stop = frames if disappear_after is None else disappear_after
    for index in range(active_stop):
        center = path[index]
        for delta, weight in ((-1, 0.35), (0, 1.0), (1, 0.35)):
            scene[index, center + delta] += weight * 0.012 * np.exp(1j * phase[index])
        if second_target:
            scene[index, 68] += 0.011 * np.exp(1j * (1.3 * phase[index] + 0.4))
    payload = np.concatenate([scene.real, scene.imag], axis=1).astype(np.float32)
    return payload, path


def test_complex_layouts_and_raw_validation() -> None:
    payload, _ = _split_half_scene(frames=20)
    split = complex_frames(payload, "split_halves")
    interleaved = complex_frames(payload, "interleaved")
    assert split.shape == interleaved.shape == (20, 91)
    np.testing.assert_allclose(split.real, payload[:, :91])
    np.testing.assert_allclose(split.imag, payload[:, 91:])


def test_causal_tracker_follows_moving_split_half_target() -> None:
    payload, expected = _split_half_scene(moving=True)
    track = causal_range_track(payload, fs=20.0, layout="split_halves")
    usable = (~track.missing) & (np.arange(len(track.bin_index)) > 80)
    assert usable.mean() > 0.45
    error = np.abs(track.bin_index[usable].astype(float) - expected[usable])
    assert np.median(error) <= 3.0
    assert np.quantile(error, 0.90) <= 8.0


def test_tracker_detects_stationary_phase_motion_and_disappearance() -> None:
    payload, _ = _split_half_scene(disappear_after=260)
    track = causal_range_track(payload, fs=20.0, layout="split_halves")
    assert np.mean(track.missing[80:240]) < 0.45
    assert np.median(track.bin_index[80:240][~track.missing[80:240]]) in range(33, 42)
    assert np.mean(track.missing[360:]) > np.mean(track.missing[80:240])


def test_two_targets_raise_multimodal_flag() -> None:
    single, _ = _split_half_scene(second_target=False)
    double, _ = _split_half_scene(second_target=True)
    first = causal_range_track(single, fs=20.0, layout="split_halves")
    second = causal_range_track(double, fs=20.0, layout="split_halves")
    assert np.mean(second.multimodal[100:]) > np.mean(first.multimodal[100:])


def test_tracking_prefix_is_bit_exact_when_future_is_appended() -> None:
    payload, _ = _split_half_scene(frames=520, moving=True)
    prefix = causal_range_track(payload[:300], fs=20.0, layout="split_halves")
    full = causal_range_track(payload, fs=20.0, layout="split_halves")
    for field in (
        "bin_index",
        "confidence",
        "normalized_entropy",
        "missing",
        "multimodal",
        "evidence_strength",
    ):
        np.testing.assert_array_equal(getattr(prefix, field), getattr(full, field)[:300])


def test_layout_comparison_is_fail_safe_and_can_support_split_halves() -> None:
    ambiguous = np.zeros((200, 182), dtype=np.float32)
    result = compare_iq_layouts(ambiguous, fs=20.0)
    assert result.selected_layout == "unknown"

    payload, _ = _split_half_scene(frames=800, moving=True)
    supported = compare_iq_layouts(payload, fs=20.0, minimum_margin=0.005)
    assert supported.split_halves_score > supported.interleaved_score
    assert supported.selected_layout == "split_halves"


def test_window_and_three_view_features_are_finite() -> None:
    payload, _ = _split_half_scene(frames=400, moving=True)
    track = causal_range_track(payload, fs=20.0, layout="split_halves")
    values, names = range_track_window_features(track, 80, 300)
    assert values.shape == (len(names),)
    assert np.isfinite(values).all()
    fused, fused_names = fuse_range_track_window_features([track, track, track], 80, 300)
    assert fused.shape == (fused_names.__len__(),)
    assert np.isfinite(fused).all()
    assert fused_names[-1] == "range_all_views_missing"
