"""Signal preparation and label quality control for direct RR estimation.

Only past samples inside a radar window are used to build model inputs.  The
BIOPAC filter is deliberately zero-phase because it is used to construct an
offline reference label, never as a production input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt


SESSION_IDENTITY: dict[str, str] = {
    "S01_CMS": "CMS",
    "S02_RJS": "RJS",
    "S03_PSJ": "PSJ",
    "S04_KTW": "KTW",
    "S05_LHS": "LHS",
    "S06_LDW": "LDW",
    "S07_KDM": "KDM",
    "S08_MDO": "MDO",
    "S09_HDH": "HDH",
    "S10_JKH": "JKH",
    "S11_SJE": "SJE",
    "S12_KDH": "KDH",
    "S13_KTW": "KTW",
    "S14_MDO": "MDO",
    "S15_JKH": "JKH",
    "S16_LJH": "LJH",
    # The folder suffix is wrong.  The spreadsheet identifies this person as PJS.
    "S17_RJS": "PJS",
    "S18_LJH": "LJH",
    "S19_CHW": "CHW",
    "S20_YSE": "YSE",
    "S21_PSJ": "PSJ",
    "S22_KJH": "KJH",
    "S23_KDM": "KDM",
    "S24_KHJ": "KHJ",
    "S25_GEC": "GEC",
    "S26_LDW": "LDW",
    "S27_HDH": "HDH",
    "S28_KDH": "KDH",
    "S29_LHS": "LHS",
    "S30_SJE": "SJE",
}


def protocol_for_session(session_id: str) -> str:
    number = int(session_id[1:3])
    if number <= 10:
        return "Dodge"
    if number <= 20:
        return "Strike"
    return "Kick"


def identity_for_session(session_id: str) -> str:
    try:
        return SESSION_IDENTITY[session_id]
    except KeyError as exc:
        raise ValueError(f"Unknown session identity: {session_id}") from exc


def causal_block_mean(x: np.ndarray, factor: int = 4) -> np.ndarray:
    """Downsample by non-overlapping, past-only boxcar averaging."""

    x = np.asarray(x)
    if factor < 1:
        raise ValueError("factor must be positive")
    usable = x.shape[0] - x.shape[0] % factor
    if usable == 0:
        return np.empty((0, *x.shape[1:]), dtype=np.float32)
    shape = (usable // factor, factor, *x.shape[1:])
    return x[:usable].reshape(shape).mean(axis=1, dtype=np.float32)


def filter_reference_rsp(
    rsp: np.ndarray,
    fs: float = 250.0,
    band_hz: tuple[float, float] = (0.10, 0.75),
) -> np.ndarray:
    """Create the offline BIOPAC respiration reference waveform."""

    rsp = np.asarray(rsp, dtype=np.float64)
    if rsp.ndim != 1:
        raise ValueError("RSP must be one-dimensional")
    sos = butter(4, band_hz, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, rsp).astype(np.float32)


def _quadratic_spectral_peak(freq: np.ndarray, power: np.ndarray) -> float:
    idx = int(np.argmax(power))
    if idx == 0 or idx == len(power) - 1:
        return float(freq[idx])
    y = np.log(np.maximum(power[idx - 1 : idx + 2], np.finfo(float).tiny))
    denominator = y[0] - 2.0 * y[1] + y[2]
    offset = 0.0 if abs(denominator) < 1e-12 else 0.5 * (y[0] - y[2]) / denominator
    return float(freq[idx] + np.clip(offset, -1.0, 1.0) * (freq[1] - freq[0]))


@dataclass(frozen=True)
class ReferenceEstimate:
    rr_bpm: float
    rr_spectral_bpm: float
    rr_phase_bpm: float
    rr_events_bpm: float
    valid: bool
    quality: float
    spectral_concentration: float
    periodicity: float
    interval_cv: float
    estimator_disagreement_bpm: float
    phase_residual_rad: float
    clip_fraction: float
    plateau_fraction: float
    breath_count: int


def estimate_reference_window(
    raw_rsp: np.ndarray,
    filtered_rsp: np.ndarray,
    *,
    fs: float,
    rr_range_bpm: tuple[float, float] = (6.0, 45.0),
    min_cycles: int = 3,
    max_clip_fraction: float = 0.02,
    min_spectral_concentration: float = 0.42,
    min_periodicity: float = 0.28,
    max_interval_cv: float = 0.27,
    max_estimator_disagreement_bpm: float = 2.5,
    max_phase_residual_rad: float = 1.25,
) -> ReferenceEstimate:
    """Estimate average RR and reject corrupted or non-periodic references.

    Three independent estimators are compared: a sub-bin FFT peak, a robust
    median inter-breath interval, and the slope of the analytic-signal phase.
    This prevents belt motion or saturation from silently becoming a target.
    """

    raw = np.asarray(raw_rsp, dtype=np.float64)
    y = np.asarray(filtered_rsp, dtype=np.float64)
    if raw.ndim != 1 or y.ndim != 1 or len(raw) != len(y):
        raise ValueError("raw_rsp and filtered_rsp must be equal-length vectors")
    if len(y) < int(8 * fs):
        raise ValueError("reference window is too short")

    y = y - np.mean(y)
    nfft = max(4096, 1 << (len(y) - 1).bit_length())
    power = np.abs(np.fft.rfft(y * np.hanning(len(y)), n=nfft)) ** 2
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    lo_hz, hi_hz = np.asarray(rr_range_bpm) / 60.0
    band = (freq >= lo_hz) & (freq <= hi_hz)
    band_freq = freq[band]
    band_power = power[band]
    spectral_hz = _quadratic_spectral_peak(band_freq, band_power)
    spectral_bpm = 60.0 * spectral_hz
    peak_neighborhood = np.abs(band_freq - spectral_hz) <= 0.03
    concentration = float(
        band_power[peak_neighborhood].sum() / max(float(band_power.sum()), np.finfo(float).tiny)
    )

    autocorrelation = np.correlate(y, y, mode="full")[len(y) - 1 :]
    autocorrelation /= max(float(autocorrelation[0]), np.finfo(float).tiny)
    period_lag = int(round(fs / spectral_hz))
    periodicity = float(autocorrelation[period_lag]) if period_lag < len(y) else -1.0

    minimum_distance = max(1, int(0.70 * fs / hi_hz))
    peaks, _ = find_peaks(y, distance=minimum_distance, prominence=0.35 * np.std(y))
    intervals = np.diff(peaks) / fs
    if len(intervals) >= 2:
        events_bpm = float(60.0 / np.median(intervals))
        interval_cv = float(np.std(intervals) / max(float(np.mean(intervals)), 1e-12))
    else:
        events_bpm = float("nan")
        interval_cv = float("inf")

    phase = np.unwrap(np.angle(hilbert(y)))
    time = np.arange(len(y), dtype=np.float64) / fs
    # Ignore Hilbert edge transients while retaining at least 80% of a window.
    edge = min(int(2 * fs), len(y) // 10)
    phase_slice = slice(edge, len(y) - edge if edge else None)
    slope, intercept = np.polyfit(time[phase_slice], phase[phase_slice], 1)
    phase_fit = slope * time[phase_slice] + intercept
    phase_residual = float(np.std(phase[phase_slice] - phase_fit))
    phase_bpm = float(60.0 * slope / (2.0 * np.pi))

    estimates = np.asarray([spectral_bpm, phase_bpm, events_bpm], dtype=float)
    finite = np.isfinite(estimates)
    disagreement = float(np.ptp(estimates[finite])) if finite.sum() >= 2 else float("inf")
    rr_bpm = float(np.median(estimates[finite])) if finite.any() else float("nan")

    clip_fraction = float(np.mean(np.abs(raw) >= 9.8))
    raw_delta = np.abs(np.diff(raw))
    flat_epsilon = max(1e-7, 1e-5 * float(np.nanpercentile(np.abs(raw), 95)))
    plateau_fraction = float(np.mean(raw_delta <= flat_epsilon)) if len(raw_delta) else 1.0

    in_range = rr_range_bpm[0] < rr_bpm < rr_range_bpm[1]
    valid = bool(
        in_range
        and len(peaks) >= min_cycles
        and clip_fraction <= max_clip_fraction
        and plateau_fraction < 0.25
        and concentration >= min_spectral_concentration
        and periodicity >= min_periodicity
        and interval_cv <= max_interval_cv
        and disagreement <= max_estimator_disagreement_bpm
        and phase_residual <= max_phase_residual_rad
    )

    components = np.asarray(
        [
            np.clip((concentration - 0.15) / 0.60, 0.0, 1.0),
            np.clip((periodicity - 0.05) / 0.70, 0.0, 1.0),
            np.clip(1.0 - interval_cv / 0.60, 0.0, 1.0),
            np.clip(1.0 - disagreement / 6.0, 0.0, 1.0),
            np.clip(1.0 - phase_residual / 2.5, 0.0, 1.0),
            np.clip(1.0 - clip_fraction / 0.10, 0.0, 1.0),
            np.clip(1.0 - plateau_fraction / 0.25, 0.0, 1.0),
        ]
    )
    quality = float(np.prod(np.maximum(components, 1e-4)) ** (1.0 / len(components)))

    return ReferenceEstimate(
        rr_bpm=rr_bpm,
        rr_spectral_bpm=spectral_bpm,
        rr_phase_bpm=phase_bpm,
        rr_events_bpm=events_bpm,
        valid=valid,
        quality=quality,
        spectral_concentration=concentration,
        periodicity=periodicity,
        interval_cv=interval_cv,
        estimator_disagreement_bpm=disagreement,
        phase_residual_rad=phase_residual,
        clip_fraction=clip_fraction,
        plateau_fraction=plateau_fraction,
        breath_count=len(peaks),
    )


@dataclass(frozen=True)
class RadarFrequencyFeatures:
    feature_map: np.ndarray
    aggregate_spectra: np.ndarray
    scalars: np.ndarray
    frequencies_hz: np.ndarray


@dataclass(frozen=True)
class ClassicalEstimate:
    rr_bpm: float
    confidence: float
    radar_peaks_bpm: tuple[float, float, float]
    consensus_spread_bpm: float


def range_frequency_features(
    window: np.ndarray,
    *,
    fs: float = 10.0,
    band_hz: tuple[float, float] = (0.08, 0.85),
    nfft: int = 2048,
    range_pool: int = 2,
) -> RadarFrequencyFeatures:
    """Build a gain-robust range-frequency map for one radar/window.

    The window is detrended using only samples available at the output time.
    Power is pooled after the FFT so opposite phases in adjacent range bins do
    not cancel.  Aggregate spectra retain candidate peaks for the compact SNN.
    """

    x = np.asarray(window, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("window must have shape [time, range]")
    if len(x) < 16:
        raise ValueError("radar window is too short")
    if range_pool < 1:
        raise ValueError("range_pool must be positive")

    x = np.clip(x, -0.05, 0.05)
    t = np.linspace(-1.0, 1.0, len(x), dtype=np.float32)
    centered_t = t - t.mean()
    def detrend(values: np.ndarray) -> np.ndarray:
        centered = values - values.mean(axis=0, keepdims=True)
        slope = (centered_t[:, None] * centered).sum(axis=0) / np.sum(centered_t**2)
        return centered - centered_t[:, None] * slope[None, :]

    detrended = detrend(x)
    temporal_variance = np.mean(detrended**2, axis=0)
    delta_power = float(np.mean(np.diff(detrended, axis=0) ** 2))

    spectrum = np.fft.rfft(detrended * np.hanning(len(x))[:, None], n=nfft, axis=0)
    power_all = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / fs)
    keep = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    frequencies = frequencies[keep].astype(np.float32)
    power = power_all[keep].astype(np.float32)

    # Light frequency smoothing makes sub-bin peak position learnable and
    # reduces isolated FFT noise without erasing close respiratory candidates.
    if len(power) >= 3:
        power[1:-1] = 0.25 * power[:-2] + 0.50 * power[1:-1] + 0.25 * power[2:]

    def pool_and_scale(
        branch_power: np.ndarray,
        branch_variance: np.ndarray,
        factor: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        usable = branch_power.shape[1] - branch_power.shape[1] % factor
        pooled_power = branch_power[:, :usable].reshape(
            len(frequencies), usable // factor, factor
        ).mean(axis=2)
        pooled_variance = branch_variance[:usable].reshape(-1, factor).mean(axis=1)
        noise = np.median(pooled_power, axis=0, keepdims=True)
        relative_power = np.log1p(
            pooled_power / np.maximum(noise, np.finfo(np.float32).tiny)
        )
        median_var = max(float(np.median(pooled_variance)), np.finfo(float).tiny)
        activity_weight = np.sqrt(np.clip(pooled_variance / median_var, 0.10, 10.0))
        scaled = np.clip(relative_power * activity_weight[None, :], 0.0, 12.0) / 12.0
        return pooled_power, pooled_variance, scaled

    pooled, variance_pooled, raw_map = pool_and_scale(power, temporal_variance, range_pool)
    feature_branches = [raw_map]
    aggregate_branches = [pooled]
    raw_weight = np.clip(
        variance_pooled / max(float(np.median(variance_pooled)), 1e-20), 0.05, 20.0
    ) ** 0.25
    aggregate_weights = [raw_weight.astype(np.float32)]

    # XeThru float recordings may be raw 182-bin frames or flattened 91-bin
    # baseband I/Q.  Retaining the raw branch is therefore mandatory; this
    # additional phase branch lets the model exploit the I/Q interpretation
    # when it is supported by the data without baking that assumption in.
    if x.shape[1] == 182:
        complex_frame = x[:, :91].astype(np.float32) + 1j * x[:, 91:].astype(np.float32)
        amplitude = np.abs(complex_frame).astype(np.float32)
        phase = np.unwrap(np.angle(complex_frame), axis=0).astype(np.float32)
        phase_detrended = detrend(phase)
        phase_variance = np.mean(phase_detrended**2, axis=0)
        phase_spectrum = np.fft.rfft(
            phase_detrended * np.hanning(len(x))[:, None], n=nfft, axis=0
        )
        phase_power = (np.abs(phase_spectrum[keep]) ** 2).astype(np.float32)
        if len(phase_power) >= 3:
            phase_power[1:-1] = (
                0.25 * phase_power[:-2]
                + 0.50 * phase_power[1:-1]
                + 0.25 * phase_power[2:]
            )
        amplitude_reliability = np.median(amplitude, axis=0)
        amplitude_reliability /= max(float(np.median(amplitude_reliability)), 1e-20)
        unit_phase = complex_frame / np.maximum(amplitude, 1e-12)
        phase_coherence = np.abs(np.mean(unit_phase, axis=0))
        phase_reliability = (
            np.clip(amplitude_reliability, 0.05, 10.0) ** 0.25
            * np.clip(phase_coherence, 0.0, 1.0) ** 4
        )
        phase_power *= np.clip(amplitude_reliability, 0.1, 10.0)[None, :]
        phase_pooled, _, phase_map = pool_and_scale(phase_power, phase_variance, 1)
        feature_branches.append(phase_map)
        aggregate_branches.append(phase_pooled)
        aggregate_weights.append(phase_reliability.astype(np.float32))

    feature_map = np.concatenate(feature_branches, axis=1)
    aggregate_power = np.concatenate(aggregate_branches, axis=1)
    normalized = aggregate_power / np.maximum(
        aggregate_power.sum(axis=0, keepdims=True), 1e-20
    )
    normalized *= np.concatenate(aggregate_weights)[None, :]
    q90 = np.quantile(normalized, 0.90, axis=1)
    q98 = np.quantile(normalized, 0.98, axis=1)
    q90 /= max(float(q90.sum()), 1e-20)
    q98 /= max(float(q98.sum()), 1e-20)
    aggregates = np.stack([q90, q98]).astype(np.float32)

    range_weights = variance_pooled / max(float(variance_pooled.sum()), 1e-20)
    range_entropy = float(
        -np.sum(range_weights * np.log(np.maximum(range_weights, 1e-20)))
        / np.log(max(2, len(range_weights)))
    )
    low_power = float(np.mean(detrended**2))
    scalars = np.asarray(
        [
            np.log1p(low_power * 1e8),
            np.log1p(delta_power * 1e8),
            np.log1p(delta_power / max(low_power, 1e-20)),
            np.log1p(float(np.median(variance_pooled)) * 1e8),
            np.log1p(float(np.quantile(variance_pooled, 0.90)) * 1e8),
            range_entropy,
            float(np.max(q90)),
            float(np.max(q98)),
        ],
        dtype=np.float32,
    )
    return RadarFrequencyFeatures(
        feature_map=feature_map.astype(np.float16),
        aggregate_spectra=aggregates,
        scalars=scalars,
        frequencies_hz=frequencies,
    )


def fuse_auxiliary_features(features: Iterable[RadarFrequencyFeatures]) -> np.ndarray:
    """Flatten per-radar spectra and append sensor-consensus diagnostics."""

    items = list(features)
    if len(items) != 3:
        raise ValueError("exactly three radar feature sets are required")
    frequency_count = len(items[0].frequencies_hz)
    if any(len(item.frequencies_hz) != frequency_count for item in items):
        raise ValueError("radar frequency grids do not match")

    per_radar_spectra = np.stack([item.aggregate_spectra for item in items])
    q90 = per_radar_spectra[:, 0]
    q98 = per_radar_spectra[:, 1]
    fused_median = np.median(q90, axis=0)
    fused_max = np.max(q98, axis=0)
    fused_median /= max(float(fused_median.sum()), 1e-20)
    fused_max /= max(float(fused_max.sum()), 1e-20)

    peak_indices = np.argmax(q90, axis=1)
    peak_bpm = items[0].frequencies_hz[peak_indices] * 60.0
    correlations = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        correlations.append(float(np.corrcoef(q90[first], q90[second])[0, 1]))
    consensus = np.asarray(
        [
            float(np.std(peak_bpm)),
            float(np.ptp(peak_bpm)),
            *correlations,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            per_radar_spectra.reshape(-1),
            np.concatenate([item.scalars for item in items]),
            fused_median.astype(np.float32),
            fused_max.astype(np.float32),
            consensus,
        ]
    ).astype(np.float32)


def classical_rr_estimate(
    features: Iterable[RadarFrequencyFeatures],
    *,
    rr_range_bpm: tuple[float, float] = (6.0, 45.0),
) -> ClassicalEstimate:
    """Harmonic-aware, three-radar spectral consensus sanity baseline."""

    items = list(features)
    if len(items) != 3:
        raise ValueError("exactly three radar feature sets are required")
    frequency_bpm = items[0].frequencies_hz.astype(np.float64) * 60.0
    keep = (frequency_bpm >= rr_range_bpm[0]) & (frequency_bpm <= rr_range_bpm[1])
    frequency_bpm = frequency_bpm[keep]
    spectra = np.stack(
        [0.35 * item.aggregate_spectra[0, keep] + 0.65 * item.aggregate_spectra[1, keep] for item in items]
    ).astype(np.float64)
    if spectra.shape[1] >= 3:
        spectra[:, 1:-1] = (
            0.25 * spectra[:, :-2] + 0.50 * spectra[:, 1:-1] + 0.25 * spectra[:, 2:]
        )
    spectra /= np.maximum(spectra.max(axis=1, keepdims=True), 1e-20)

    radar_peaks = tuple(float(frequency_bpm[np.argmax(row)]) for row in spectra)
    fused = np.median(spectra, axis=0)
    candidate_indices, _ = find_peaks(fused, distance=max(1, int(1.0 / np.diff(frequency_bpm).mean())))
    for row in spectra:
        local, _ = find_peaks(row, distance=max(1, int(1.0 / np.diff(frequency_bpm).mean())))
        if local.size:
            candidate_indices = np.concatenate([candidate_indices, local[np.argsort(row[local])[-4:]]])
    if candidate_indices.size == 0:
        candidate_indices = np.asarray([int(np.argmax(fused))])
    candidate_indices = np.unique(candidate_indices)

    def support_at(row: np.ndarray, candidate: float, tolerance: float = 1.5) -> float:
        nearby = np.abs(frequency_bpm - candidate) <= tolerance
        return float(np.max(row[nearby])) if nearby.any() else 0.0

    candidate_scores: list[float] = []
    direct_supports: list[np.ndarray] = []
    for index in candidate_indices:
        candidate = float(frequency_bpm[index])
        direct = np.asarray([support_at(row, candidate) for row in spectra])
        harmonic = np.asarray(
            [
                max(
                    support_at(row, candidate * 2.0, 2.0),
                    0.5 * support_at(row, candidate / 2.0, 1.0),
                )
                for row in spectra
            ]
        )
        # At least two sensors must support a confident output.  Harmonics only
        # break near-ties; they cannot overwhelm direct multi-view evidence.
        ordered = np.sort(direct)
        score = 0.55 * ordered[-2] + 0.30 * np.mean(direct) + 0.15 * np.median(harmonic)
        candidate_scores.append(float(score))
        direct_supports.append(direct)

    order = np.argsort(candidate_scores)
    best_position = int(order[-1])
    best_index = int(candidate_indices[best_position])
    rr = _quadratic_spectral_peak(frequency_bpm, fused) if best_index == int(np.argmax(fused)) else float(frequency_bpm[best_index])
    # The generic quadratic helper assumes the selected point is the maximum;
    # refine an alternate consensus candidate locally instead.
    if best_index != int(np.argmax(fused)) and 0 < best_index < len(fused) - 1:
        local_frequency = frequency_bpm[best_index - 1 : best_index + 2]
        local_power = fused[best_index - 1 : best_index + 2]
        rr = _quadratic_spectral_peak(local_frequency, local_power)
    margin = candidate_scores[best_position] - (
        candidate_scores[int(order[-2])] if len(order) > 1 else 0.0
    )
    direct = direct_supports[best_position]
    spread = float(np.std(radar_peaks))
    confidence = float(
        np.clip(
            0.45 * np.sort(direct)[-2]
            + 0.25 * np.mean(direct)
            + 0.20 * np.clip(margin / 0.25, 0.0, 1.0)
            + 0.10 * np.exp(-spread / 4.0),
            0.0,
            1.0,
        )
    )
    return ClassicalEstimate(rr, confidence, radar_peaks, spread)
