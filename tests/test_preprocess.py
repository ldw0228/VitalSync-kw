import numpy as np

from snn_rr.preprocess import (
    causal_block_mean,
    estimate_reference_window,
    filter_reference_rsp,
    fuse_auxiliary_features,
    range_frequency_features,
    replace_radar_outliers_past_only,
)


def test_reference_estimator_accepts_clean_breathing_and_rejects_clipping():
    fs = 250.0
    time = np.arange(int(32 * fs)) / fs
    raw = 2.0 * np.sin(2 * np.pi * 0.25 * time)
    filtered = filter_reference_rsp(raw, fs=fs)
    estimate = estimate_reference_window(raw, filtered, fs=fs)
    assert estimate.valid
    assert abs(estimate.rr_bpm - 15.0) < 0.3

    clipped = raw.copy()
    clipped[: int(4 * fs)] = 10.0
    clipped_estimate = estimate_reference_window(clipped, filtered, fs=fs)
    assert not clipped_estimate.valid
    assert clipped_estimate.clip_fraction > 0.02


def test_radar_features_are_finite_and_peak_near_input_frequency():
    fs = 10.0
    time = np.arange(320) / fs
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1e-4, (len(time), 182)).astype(np.float32)
    x[:, 60:66] += (1e-3 * np.sin(2 * np.pi * 0.30 * time))[:, None]
    item = range_frequency_features(x, fs=fs)
    peak = item.frequencies_hz[np.argmax(item.aggregate_spectra[1])]
    # Raw 182-bin power pooled to 91 plus an optional 91-bin I/Q phase branch.
    assert item.feature_map.shape[1] == 182
    assert abs(peak - 0.30) < 0.05
    assert np.isfinite(item.feature_map).all()
    aux = fuse_auxiliary_features([item, item, item])
    assert np.isfinite(aux).all()
    assert aux.ndim == 1


def test_causal_block_mean_drops_only_incomplete_tail():
    x = np.arange(18, dtype=np.float32).reshape(9, 2)
    result = causal_block_mean(x, factor=4)
    np.testing.assert_allclose(result, [[3, 4], [11, 12]])


def test_radar_outlier_repair_uses_only_past_same_bin_history():
    values = np.asarray(
        [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [9.0, 0.08]],
        dtype=np.float32,
    )
    repaired, count = replace_radar_outliers_past_only(values, threshold=0.1)
    assert count == 1
    assert repaired[3, 0] == np.median(values[:3, 0])
    np.testing.assert_array_equal(repaired[:3], values[:3])
    np.testing.assert_array_equal(repaired[:, 1], values[:, 1])
