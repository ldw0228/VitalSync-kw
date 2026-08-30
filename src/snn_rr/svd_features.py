"""Label-free source-separated slow-time features for UWB RR estimation.

The legacy range-frequency cache exposes power at every range bin, but a
large gross-motion component can dominate a weak respiratory component.  This
module performs a small, deterministic randomized SVD inside each causal
window and preserves the spectra of several separated components.  No BIOPAC
or reference-derived value is read here.

The split-half amplitude/phase variants are hypotheses rather than an asserted
device layout.  Raw real-valued and derivative variants are always retained,
and manifests must keep the variant names so the hypothesis remains auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.utils.extmath import randomized_svd


DEFAULT_SVD_VARIANTS = (
    "raw",
    "raw_standardized",
    "temporal_velocity",
    "temporal_velocity_standardized",
    "range_difference",
    "range_difference_standardized",
    "split_amplitude",
    "split_amplitude_standardized",
    "split_phase",
    "split_phase_standardized",
)


@dataclass(frozen=True, slots=True)
class SVDWindowFeatures:
    """Component spectra and label-free diagnostics for one radar window.

    ``spectra`` has shape ``[variant, component, frequency]``.  Every spectrum
    is non-negative and sums to one unless the component has zero band power.
    ``attributes`` has the final layout described by ``ATTRIBUTE_NAMES``.
    """

    spectra: np.ndarray
    component_signals: np.ndarray
    attributes: np.ndarray
    frequencies_hz: np.ndarray
    variant_names: tuple[str, ...]


ATTRIBUTE_NAMES = (
    "singular_energy_fraction",
    "band_power_fraction",
    "spectral_concentration",
    "spectral_entropy",
    "peak_frequency_hz",
)


def _linear_detrend(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) < 16:
        raise ValueError("values must have shape [time>=16, channels]")
    time = np.linspace(-1.0, 1.0, len(values), dtype=np.float32)
    centered_time = time - time.mean()
    centered = values - values.mean(axis=0, keepdims=True)
    denominator = float(np.dot(centered_time, centered_time))
    slope = (centered_time[:, None] * centered).sum(axis=0) / denominator
    output = centered - centered_time[:, None] * slope[None, :]
    return np.nan_to_num(output, copy=False).astype(np.float32, copy=False)


def _standardize_channels(values: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=0))
    finite = np.isfinite(scale) & (scale > 1e-8)
    safe = np.where(finite, scale, 1.0).astype(np.float32)
    standardized = values / safe[None, :]
    standardized[:, ~finite] = 0.0
    return np.clip(np.nan_to_num(standardized), -20.0, 20.0).astype(np.float32)


def svd_variant_matrices(
    window: np.ndarray,
    *,
    variants: Iterable[str] = DEFAULT_SVD_VARIANTS,
) -> tuple[list[np.ndarray], tuple[str, ...]]:
    """Return deterministic slow-time matrices selected by ``variants``.

    The temporal derivative is past-only: the first velocity frame is zero and
    every later frame is current minus immediately previous.  Split-half
    variants are emitted only for an even channel count and are explicitly
    named as hypotheses in downstream provenance.
    """

    raw = np.asarray(window, dtype=np.float32)
    if raw.ndim != 2 or len(raw) < 16 or raw.shape[1] < 2:
        raise ValueError("window must have shape [time>=16, range>=2]")
    raw = np.clip(np.nan_to_num(raw), -0.05, 0.05)
    detrended_raw = _linear_detrend(raw)
    temporal_velocity = np.diff(
        detrended_raw, axis=0, prepend=detrended_raw[:1]
    ).astype(np.float32)
    range_difference = np.diff(detrended_raw, axis=1).astype(np.float32)

    available: dict[str, np.ndarray] = {
        "raw": detrended_raw,
        "raw_standardized": _standardize_channels(detrended_raw),
        "temporal_velocity": _linear_detrend(temporal_velocity),
        "temporal_velocity_standardized": _standardize_channels(
            _linear_detrend(temporal_velocity)
        ),
        "range_difference": _linear_detrend(range_difference),
        "range_difference_standardized": _standardize_channels(
            _linear_detrend(range_difference)
        ),
    }
    if raw.shape[1] % 2 == 0:
        half = raw.shape[1] // 2
        split_complex = raw[:, :half] + 1j * raw[:, half:]
        amplitude = _linear_detrend(np.abs(split_complex).astype(np.float32))
        phase = _linear_detrend(
            np.unwrap(np.angle(split_complex), axis=0).astype(np.float32)
        )
        available.update(
            {
                "split_amplitude": amplitude,
                "split_amplitude_standardized": _standardize_channels(amplitude),
                "split_phase": phase,
                "split_phase_standardized": _standardize_channels(phase),
            }
        )

    selected = tuple(str(value) for value in variants)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("variants must be a non-empty unique sequence")
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"variants unavailable for this input: {missing}")
    return [available[name] for name in selected], selected


def svd_component_features(
    window: np.ndarray,
    *,
    fs: float = 10.0,
    band_hz: tuple[float, float] = (0.08, 0.80),
    nfft: int = 4096,
    components: int = 12,
    n_iter: int = 2,
    random_state: int = 20260828,
    variants: Iterable[str] = DEFAULT_SVD_VARIANTS,
) -> SVDWindowFeatures:
    """Separate slow-time sources and return their normalized band spectra."""

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive")
    low, high = map(float, band_hz)
    if not 0 < low < high < fs / 2:
        raise ValueError("band_hz must lie strictly inside the Nyquist range")
    if nfft < len(window) or components < 1 or n_iter < 0:
        raise ValueError("nfft/components/n_iter configuration is invalid")

    matrices, variant_names = svd_variant_matrices(window, variants=variants)
    frequencies_all = np.fft.rfftfreq(int(nfft), d=1.0 / float(fs))
    keep = (frequencies_all >= low) & (frequencies_all <= high)
    frequencies = frequencies_all[keep].astype(np.float32)
    if not len(frequencies):
        raise ValueError("configured band contains no FFT bins")
    taper = np.hanning(len(window)).astype(np.float32)[:, None]

    spectra = np.zeros(
        (len(matrices), components, len(frequencies)), dtype=np.float32
    )
    component_signals = np.zeros(
        (len(matrices), components, len(window)), dtype=np.float32
    )
    attributes = np.zeros(
        (len(matrices), components, len(ATTRIBUTE_NAMES)), dtype=np.float32
    )
    for variant_index, matrix in enumerate(matrices):
        rank = min(int(components), min(matrix.shape))
        if rank < 1 or not np.any(matrix):
            continue
        u, singular, _ = randomized_svd(
            matrix,
            n_components=rank,
            n_iter=int(n_iter),
            random_state=int(random_state + 1009 * variant_index),
            flip_sign=True,
        )
        component_signal = u * singular[None, :]
        component_rms = np.sqrt(
            np.mean(np.square(component_signal, dtype=np.float64), axis=0)
        )
        component_signals[variant_index, :rank] = (
            component_signal / np.maximum(component_rms[None, :], 1e-8)
        ).T.astype(np.float32)
        full_power = np.abs(
            np.fft.rfft(component_signal * taper, n=int(nfft), axis=0)
        ) ** 2
        band_power = np.asarray(full_power[keep].T, dtype=np.float64)
        band_sum = band_power.sum(axis=1)
        full_sum = np.asarray(full_power.sum(axis=0), dtype=np.float64)
        normalized = np.divide(
            band_power,
            band_sum[:, None],
            out=np.zeros_like(band_power),
            where=band_sum[:, None] > 0,
        )
        spectra[variant_index, :rank] = normalized.astype(np.float32)

        total_energy = max(float(np.square(matrix, dtype=np.float64).sum()), 1e-20)
        singular_fraction = np.square(singular, dtype=np.float64) / total_energy
        band_fraction = np.divide(
            band_sum,
            np.maximum(full_sum, 1e-20),
        )
        concentration = normalized.max(axis=1)
        entropy = -np.sum(
            normalized * np.log(np.maximum(normalized, 1e-20)), axis=1
        ) / np.log(max(2, normalized.shape[1]))
        peak = frequencies[np.argmax(normalized, axis=1)]
        attributes[variant_index, :rank] = np.stack(
            [singular_fraction, band_fraction, concentration, entropy, peak], axis=1
        ).astype(np.float32)

    if not (
        np.isfinite(spectra).all()
        and np.isfinite(component_signals).all()
        and np.isfinite(attributes).all()
    ):
        raise FloatingPointError("SVD feature extraction produced non-finite values")
    return SVDWindowFeatures(
        spectra=spectra,
        component_signals=component_signals,
        attributes=attributes,
        frequencies_hz=frequencies,
        variant_names=variant_names,
    )
