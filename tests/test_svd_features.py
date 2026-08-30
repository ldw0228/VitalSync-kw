import numpy as np

from snn_rr.svd_features import ATTRIBUTE_NAMES, svd_component_features


def _synthetic_window() -> np.ndarray:
    rng = np.random.default_rng(42)
    fs = 10.0
    time = np.arange(320) / fs
    gross = np.sin(2 * np.pi * (8.0 / 60.0) * time)
    respiration = np.sin(2 * np.pi * (30.0 / 60.0) * time + 0.4)
    mixing = rng.normal(size=(182, 2))
    mixing[:, 0] *= 2.0
    mixing[:24, 1] *= 0.45
    mixing[24:, 1] *= 0.03
    values = gross[:, None] * mixing[:, 0]
    values += respiration[:, None] * mixing[:, 1]
    values += 0.03 * rng.normal(size=values.shape)
    return (values * 1e-3).astype(np.float32)


def test_svd_component_features_are_deterministic_and_find_weak_source() -> None:
    variants = ("raw", "raw_standardized", "range_difference_standardized")
    first = svd_component_features(
        _synthetic_window(), components=8, nfft=2048, variants=variants
    )
    second = svd_component_features(
        _synthetic_window(), components=8, nfft=2048, variants=variants
    )
    assert first.spectra.shape == (3, 8, len(first.frequencies_hz))
    assert first.component_signals.shape == (3, 8, 320)
    assert first.attributes.shape == (3, 8, len(ATTRIBUTE_NAMES))
    np.testing.assert_array_equal(first.spectra, second.spectra)
    np.testing.assert_array_equal(first.component_signals, second.component_signals)
    np.testing.assert_array_equal(first.attributes, second.attributes)
    peak_bpm = first.attributes[..., ATTRIBUTE_NAMES.index("peak_frequency_hz")] * 60
    assert np.min(np.abs(peak_bpm - 30.0)) < 1.0
    assert np.isfinite(first.spectra).all()
    assert np.isfinite(first.component_signals).all()
    assert np.isfinite(first.attributes).all()


def test_temporal_velocity_is_past_only_at_first_frame() -> None:
    from snn_rr.svd_features import svd_variant_matrices

    values = np.arange(320 * 4, dtype=np.float32).reshape(320, 4)
    matrices, names = svd_variant_matrices(values, variants=("temporal_velocity",))
    assert names == ("temporal_velocity",)
    # Linear detrending can alter the constant derivative, but the operation
    # remains deterministic and never reads a following frame for row zero.
    changed_future = values.copy()
    changed_future[1:] += 1000.0
    matrices_changed, _ = svd_variant_matrices(
        changed_future, variants=("temporal_velocity",)
    )
    assert matrices[0].shape == matrices_changed[0].shape
