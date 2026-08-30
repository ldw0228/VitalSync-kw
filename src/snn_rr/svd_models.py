"""Source-separated spiking RR model for the raw-window SVD cache.

The cache built by :mod:`snn_rr.svd_features` retains several label-free SVD
component spectra per radar.  This module keeps that component topology until
the first learned 1x1 projection, fuses only available radars, and performs the
frequency reasoning with a small snnTorch LIF residual network.

The model is deliberately a *safe residual* over an existing estimator.  Its
final posterior is a learned mixture of the supplied base Gaussian posterior
and the source-separated posterior.  The mixture gate starts with a strongly
negative bias, so an untrained model remains close to the supplied base
prediction and falls back to it exactly when every radar is unavailable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError as exc:  # pragma: no cover - broken runtime only
    raise ImportError(
        "SourceSeparatedRRSNN requires snntorch. Install the project runtime "
        "dependencies before importing snn_rr.svd_models."
    ) from exc

from .models import apply_radar_dropout, make_rr_bins


SVD_RR_MIN = 6.0
SVD_RR_MAX = 45.0
SVD_RR_STEP = 0.25
SVD_NUM_RR_BINS = int(round((SVD_RR_MAX - SVD_RR_MIN) / SVD_RR_STEP)) + 1


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(maximum), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _masked_radar_softmax(scores: Tensor, available: Tensor) -> Tensor:
    """Softmax over available views, returning exact zeros for an empty row."""

    masked = scores.masked_fill(~available, -1.0e4)
    weights = masked.softmax(dim=1) * available.to(dtype=scores.dtype)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)


def _posterior_statistics(probabilities: Tensor, bins: Tensor) -> tuple[Tensor, Tensor]:
    expected = (probabilities * bins).sum(dim=-1)
    variance = (
        probabilities * (bins.unsqueeze(0) - expected.unsqueeze(1)).square()
    ).sum(dim=-1)
    return expected, variance.clamp_min(torch.finfo(probabilities.dtype).eps).sqrt()


class _SharedComponentEncoder(nn.Module):
    """Shared per-radar encoder for raw and quality-weighted SVD spectra."""

    def __init__(
        self,
        *,
        num_variants: int,
        num_components: int,
        channels: int,
        residual_blocks: int,
    ) -> None:
        super().__init__()
        if num_variants < 1 or num_components < 1:
            raise ValueError("num_variants and num_components must be positive")
        if channels < 8 or residual_blocks < 1:
            raise ValueError("encoder channels >= 8 and residual_blocks >= 1 required")
        self.num_variants = int(num_variants)
        self.num_components = int(num_components)
        input_channels = 2 * self.num_variants * self.num_components
        # The first learned operation is intentionally pointwise: component and
        # variant identities are channels, while physical frequency is kept.
        self.component_projection = nn.Sequential(
            nn.Conv1d(input_channels, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    5,
                    padding=2,
                    groups=channels,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(channels), channels),
                nn.SiLU(),
                nn.Conv1d(channels, channels, 1, bias=False),
                nn.GroupNorm(_group_count(channels), channels),
            )
            for _ in range(int(residual_blocks))
        )
        # Mean/max spectral summaries plus mean/max of all five component
        # attributes.  The same head is shared across the three radars.
        reliability_dim = 2 * channels + 10
        self.reliability = nn.Sequential(
            nn.Linear(reliability_dim, max(16, channels // 2)),
            nn.SiLU(),
            nn.Linear(max(16, channels // 2), 1),
        )
        self.out_channels = int(channels)

    @staticmethod
    def component_quality(spectra: Tensor, attributes: Tensor) -> Tensor:
        """Return deterministic label-free component weights.

        Attributes are ordered as energy fraction, band fraction,
        concentration, entropy and peak frequency.  Peak location is not a
        quality criterion; the first four diagnostics define the weight.  A
        per-radar normalization keeps the weighted branch on roughly the same
        scale as the raw branch without introducing reference-derived state.
        """

        energy = attributes[..., 0].clamp(0.0, 1.0)
        band = attributes[..., 1].clamp(0.0, 1.0)
        concentration = attributes[..., 2].clamp(0.0, 1.0)
        entropy = attributes[..., 3].clamp(0.0, 1.0)
        energy_band = energy * band
        # ``clamp_min`` avoids an infinite sqrt derivative for deliberately
        # zeroed missing-radar attributes; the explicit indicator preserves an
        # exact zero weight for such components.
        quality = (
            torch.sqrt(energy_band.clamp_min(1.0e-8))
            * (energy_band > 0).to(dtype=energy_band.dtype)
            * concentration
            * (1.0 - 0.5 * entropy)
        )
        has_power = spectra.sum(dim=-1) > 0
        quality = quality * has_power.to(dtype=quality.dtype)
        positive = quality > 0
        denominator = (
            quality.sum(dim=(-2, -1), keepdim=True)
            / positive.sum(dim=(-2, -1), keepdim=True).clamp_min(1).to(quality.dtype)
        ).clamp_min(1.0e-8)
        return (quality / denominator).clamp(0.0, 4.0) * positive.to(quality.dtype)

    @staticmethod
    def _normalized_attribute_summary(attributes: Tensor) -> Tensor:
        bounded_diagnostics = attributes[..., :4].clamp(0.0, 1.0)
        # SVD cache peak frequencies live below one hertz.  Clamping makes this
        # summary robust to corrupt metadata while retaining physical ordering.
        bounded_peak = attributes[..., 4:].clamp(0.0, 2.0) / 2.0
        values = torch.cat((bounded_diagnostics, bounded_peak), dim=-1)
        return torch.cat(
            (values.mean(dim=(-3, -2)), values.amax(dim=(-3, -2))), dim=-1
        )

    def forward(
        self, spectra: Tensor, attributes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, radars, variants, components, frequencies = spectra.shape
        quality = self.component_quality(spectra, attributes)
        raw = spectra.reshape(batch * radars, variants * components, frequencies)
        weighted = (spectra * quality.unsqueeze(-1)).reshape(
            batch * radars, variants * components, frequencies
        )
        feature = self.component_projection(torch.cat((raw, weighted), dim=1))
        for block in self.blocks:
            feature = F.silu(feature + block(feature))
        pooled = torch.cat((feature.mean(dim=-1), feature.amax(dim=-1)), dim=1)
        attribute_summary = self._normalized_attribute_summary(attributes).reshape(
            batch * radars, 10
        )
        reliability = self.reliability(
            torch.cat((pooled, attribute_summary), dim=1)
        ).squeeze(1)
        return feature, reliability, quality


class _LIFResidualFrequencyBlock(nn.Module):
    """One snnTorch LIF residual block operating on the frequency axis."""

    def __init__(
        self,
        channels: int,
        *,
        beta: float,
        kernel_size: int,
        learn_beta: bool,
        learn_threshold: bool,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("spike_kernel_size must be a positive odd integer")
        padding = kernel_size // 2
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.lif1 = snn.Leaky(
            beta=beta,
            threshold=1.0,
            spike_grad=spike_grad,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            reset_mechanism="subtract",
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.lif2 = snn.Leaky(
            beta=beta,
            threshold=1.0,
            spike_grad=spike_grad,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            reset_mechanism="subtract",
        )
        self.residual_gain = nn.Parameter(torch.tensor(0.0))

    def forward_step(
        self,
        spikes: Tensor,
        state: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        if state is None:
            membrane1 = torch.zeros_like(spikes)
            membrane2 = torch.zeros_like(spikes)
        else:
            membrane1, membrane2 = state
        current1 = self.norm1(self.conv1(spikes))
        spikes1, membrane1 = self.lif1(current1, membrane1)
        current2 = self.norm2(self.conv2(spikes1))
        current2 = current2 + torch.sigmoid(self.residual_gain) * spikes
        spikes2, membrane2 = self.lif2(current2, membrane2)
        return spikes2, (membrane1, membrane2), (spikes1, spikes2)


class SourceSeparatedRRSNN(nn.Module):
    """Quality-weighted, source-separated SNN with safe base-posterior fusion.

    Parameters
    ----------
    num_variants, num_components:
        The ``V`` and ``C`` dimensions of the SVD cache.
    base_feature_dim:
        Number of additional label-free/base-model scalar inputs.  Pass zero
        when no such scalars are used.

    Forward inputs
    --------------
    spectra:
        ``[B, 3, V, C, F]`` normalized component spectra.
    attributes:
        ``[B, 3, V, C, 5]`` SVD diagnostics.
    base_prediction, base_std:
        Existing estimator mean and standard deviation, each ``[B]``.
    base_features:
        Optional ``[B, base_feature_dim]`` scalar evidence.
    classical_rr, classical_std:
        Optional ``[B]`` or ``[B, K]`` classical rates and uncertainties.
        Each rate generates explicit x1, x2, x3 and x4 Gaussian candidate
        priors.  If omitted, the base estimate supplies one classical seed.
    radar_mask:
        Boolean ``[B, 3]`` availability mask; ``True`` means available.
    """

    _CANDIDATE_MULTIPLIERS = (1.0, 2.0, 3.0, 4.0)

    def __init__(
        self,
        *,
        num_variants: int = 10,
        num_components: int = 12,
        num_radars: int = 3,
        base_feature_dim: int = 0,
        rr_min: float = SVD_RR_MIN,
        rr_max: float = SVD_RR_MAX,
        rr_step: float = SVD_RR_STEP,
        encoder_channels: int = 48,
        encoder_blocks: int = 2,
        hidden_channels: int = 64,
        num_spiking_blocks: int = 2,
        simulation_steps: int = 8,
        beta: float = 0.9,
        learn_beta: bool = True,
        learn_threshold: bool = True,
        spike_kernel_size: int = 5,
        radar_dropout_p: float = 0.15,
        spectral_frequency_min_hz: float = 0.08,
        spectral_frequency_max_hz: float = 0.80,
        candidate_sigma: float = 1.0,
        initial_gate_bias: float = -6.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if num_radars < 1:
            raise ValueError("num_radars must be positive")
        if base_feature_dim < 0:
            raise ValueError("base_feature_dim cannot be negative")
        if not (
            math.isfinite(rr_min)
            and math.isfinite(rr_max)
            and math.isfinite(rr_step)
            and rr_max > rr_min
            and rr_step > 0
        ):
            raise ValueError("RR limits and step must be finite and increasing")
        span_steps = (float(rr_max) - float(rr_min)) / float(rr_step)
        if not math.isclose(span_steps, round(span_steps), abs_tol=1.0e-7):
            raise ValueError("rr_step must exactly partition the inclusive RR range")
        if not 1 <= simulation_steps <= 64:
            raise ValueError("simulation_steps must be in [1, 64]")
        if hidden_channels < 8 or num_spiking_blocks < 1:
            raise ValueError("hidden_channels >= 8 and num_spiking_blocks >= 1 required")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be strictly between zero and one")
        if not 0.0 <= radar_dropout_p <= 1.0:
            raise ValueError("radar_dropout_p must be in [0, 1]")
        if not (
            math.isfinite(spectral_frequency_min_hz)
            and math.isfinite(spectral_frequency_max_hz)
            and 0 < spectral_frequency_min_hz < spectral_frequency_max_hz
        ):
            raise ValueError("spectral frequency limits must be finite and increasing")
        if rr_min / 60.0 < spectral_frequency_min_hz or rr_max / 60.0 > spectral_frequency_max_hz:
            raise ValueError("the direct RR grid must lie inside the spectral grid")
        if not math.isfinite(candidate_sigma) or candidate_sigma <= 0:
            raise ValueError("candidate_sigma must be finite and positive")

        self.num_variants = int(num_variants)
        self.num_components = int(num_components)
        self.num_radars = int(num_radars)
        self.base_feature_dim = int(base_feature_dim)
        self.rr_min = float(rr_min)
        self.rr_max = float(rr_max)
        self.rr_step = float(rr_step)
        self.num_rr_bins = int(round(span_steps)) + 1
        self.simulation_steps = int(simulation_steps)
        self.radar_dropout_p = float(radar_dropout_p)
        self.spectral_frequency_min_hz = float(spectral_frequency_min_hz)
        self.spectral_frequency_max_hz = float(spectral_frequency_max_hz)
        self.candidate_sigma = float(candidate_sigma)
        self.register_buffer(
            "rr_bins",
            make_rr_bins(rr_min, rr_max, self.num_rr_bins),
            persistent=True,
        )
        self.register_buffer(
            "candidate_multipliers",
            torch.tensor(self._CANDIDATE_MULTIPLIERS, dtype=torch.float32),
            persistent=False,
        )

        self.radar_encoder = _SharedComponentEncoder(
            num_variants=self.num_variants,
            num_components=self.num_components,
            channels=int(encoder_channels),
            residual_blocks=int(encoder_blocks),
        )
        self.input_projection = nn.Sequential(
            nn.Conv1d(encoder_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
        )
        self.candidate_projection = nn.Sequential(
            nn.Conv1d(len(self._CANDIDATE_MULTIPLIERS), hidden_channels, 1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
        )
        scalar_channels = max(8, hidden_channels // 4)
        if self.base_feature_dim:
            self.base_feature_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(self.base_feature_dim, scalar_channels),
                nn.SiLU(),
                nn.Linear(scalar_channels, scalar_channels),
                nn.SiLU(),
            )
            self.scalar_current = nn.Linear(scalar_channels, hidden_channels, bias=False)
            scalar_output_dim = scalar_channels
        else:
            self.base_feature_encoder = None
            self.scalar_current = None
            scalar_output_dim = 0
        self.scalar_output_dim = scalar_output_dim
        self.input_scale_unconstrained = nn.Parameter(torch.tensor(0.0))
        self.input_lif = snn.Leaky(
            beta=beta,
            threshold=1.0,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            reset_mechanism="subtract",
        )
        self.spiking_blocks = nn.ModuleList(
            _LIFResidualFrequencyBlock(
                hidden_channels,
                beta=beta,
                kernel_size=spike_kernel_size,
                learn_beta=learn_beta,
                learn_threshold=learn_threshold,
            )
            for _ in range(int(num_spiking_blocks))
        )
        readout_channels = 2 * hidden_channels
        self.local_logit_head = nn.Sequential(
            nn.Conv1d(readout_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, 1, 1),
        )
        global_dim = 4 * hidden_channels + scalar_output_dim + 1 + len(
            self._CANDIDATE_MULTIPLIERS
        )
        self.global_logit_head = nn.Sequential(
            nn.Linear(global_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, self.num_rr_bins),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(global_dim, max(16, hidden_channels // 2)),
            nn.SiLU(),
            nn.Linear(max(16, hidden_channels // 2), 1),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(global_dim, max(16, hidden_channels // 2)),
            nn.SiLU(),
            nn.Linear(max(16, hidden_channels // 2), 1),
        )
        # Exact, auditable safe start: before training, every sample gets the
        # same small source correction regardless of random upstream weights.
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, float(initial_gate_bias))

    def _validate_inputs(
        self,
        spectra: Tensor,
        attributes: Tensor,
        base_prediction: Tensor,
        base_std: Tensor,
        base_features: Tensor | None,
        radar_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        expected_spectral_prefix = (
            self.num_radars,
            self.num_variants,
            self.num_components,
        )
        if spectra.ndim != 5 or tuple(spectra.shape[1:4]) != expected_spectral_prefix:
            raise ValueError(
                "spectra must have shape "
                f"[batch, {self.num_radars}, {self.num_variants}, "
                f"{self.num_components}, frequency], got {tuple(spectra.shape)}"
            )
        if spectra.shape[-1] < 3:
            raise ValueError("spectra must contain at least three frequency bins")
        expected_attributes = (*spectra.shape[:4], 5)
        if tuple(attributes.shape) != expected_attributes:
            raise ValueError(
                f"attributes must have shape {expected_attributes}, got "
                f"{tuple(attributes.shape)}"
            )
        batch = spectra.shape[0]
        if base_prediction.shape not in {(batch,), (batch, 1)}:
            raise ValueError(f"base_prediction must have shape {(batch,)}")
        if base_std.shape not in {(batch,), (batch, 1)}:
            raise ValueError(f"base_std must have shape {(batch,)}")
        base_prediction = base_prediction.reshape(batch)
        base_std = base_std.reshape(batch)
        if self.base_feature_dim:
            if base_features is None:
                base_features = spectra.new_zeros((batch, self.base_feature_dim))
            elif base_features.shape != (batch, self.base_feature_dim):
                raise ValueError(
                    f"base_features must have shape {(batch, self.base_feature_dim)}, "
                    f"got {tuple(base_features.shape)}"
                )
        elif base_features is not None and base_features.numel():
            raise ValueError(
                "base_features were supplied but base_feature_dim=0 at construction"
            )
        if radar_mask is None:
            radar_mask = torch.ones(
                (batch, self.num_radars), device=spectra.device, dtype=torch.bool
            )
        elif radar_mask.shape != (batch, self.num_radars):
            raise ValueError(
                f"radar_mask must have shape {(batch, self.num_radars)}, got "
                f"{tuple(radar_mask.shape)}"
            )

        if not torch.is_floating_point(spectra):
            spectra = spectra.float()
        device, dtype = spectra.device, spectra.dtype
        spectra = torch.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        attributes = torch.nan_to_num(
            attributes.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0
        )
        base_prediction = base_prediction.to(device=device, dtype=dtype)
        base_std = base_std.to(device=device, dtype=dtype)
        if base_features is not None:
            base_features = torch.nan_to_num(
                base_features.to(device=device, dtype=dtype),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ).clamp(-20.0, 20.0)
        return (
            spectra,
            attributes,
            base_prediction,
            base_std,
            base_features,
            radar_mask.to(device=device) > 0,
        )

    def _base_posterior(
        self, base_prediction: Tensor, base_std: Tensor
    ) -> tuple[Tensor, Tensor]:
        valid = torch.isfinite(base_prediction) & torch.isfinite(base_std) & (base_std > 0)
        midpoint = 0.5 * (self.rr_min + self.rr_max)
        mean = torch.where(valid, base_prediction, base_prediction.new_full((), midpoint))
        sigma = torch.where(valid, base_std, base_std.new_full((), 4.0)).clamp(
            min=max(self.rr_step, 1.0e-3), max=self.rr_max - self.rr_min
        )
        bins = self.rr_bins.to(device=mean.device, dtype=mean.dtype)
        log_weight = -0.5 * ((bins.unsqueeze(0) - mean.unsqueeze(1)) / sigma.unsqueeze(1)).square()
        log_weight = log_weight - log_weight.amax(dim=1, keepdim=True)
        probability = log_weight.exp()
        probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        return probability, valid

    def _candidate_priors(
        self,
        base_prediction: Tensor,
        base_std: Tensor,
        classical_rr: Tensor | None,
        classical_std: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = base_prediction.shape[0]
        if classical_rr is None:
            classical_rr = base_prediction.unsqueeze(1)
            if classical_std is None:
                classical_std = base_std.unsqueeze(1)
        else:
            classical_rr = classical_rr.to(
                device=base_prediction.device, dtype=base_prediction.dtype
            )
            if classical_rr.ndim == 1:
                classical_rr = classical_rr.unsqueeze(1)
            if classical_rr.ndim != 2 or classical_rr.shape[0] != batch:
                raise ValueError("classical_rr must have shape [batch] or [batch, K]")
        if classical_std is None:
            classical_std = classical_rr.new_full(classical_rr.shape, self.candidate_sigma)
        else:
            classical_std = classical_std.to(
                device=base_prediction.device, dtype=base_prediction.dtype
            )
            if classical_std.ndim == 0:
                classical_std = classical_std.expand_as(classical_rr)
            elif classical_std.ndim == 1:
                if classical_std.shape[0] != batch:
                    raise ValueError("one-dimensional classical_std must have length batch")
                classical_std = classical_std.unsqueeze(1)
            if classical_std.shape != classical_rr.shape:
                raise ValueError("classical_std must be scalar or match classical_rr")

        multipliers = self.candidate_multipliers.to(
            device=base_prediction.device, dtype=base_prediction.dtype
        )
        means = classical_rr.unsqueeze(-1) * multipliers.view(1, 1, -1)
        sigmas = classical_std.unsqueeze(-1).abs() * multipliers.view(1, 1, -1)
        valid = (
            torch.isfinite(means)
            & torch.isfinite(sigmas)
            & (classical_rr.unsqueeze(-1) > 0)
            & (sigmas > 0)
            & (means >= self.rr_min)
            & (means <= self.rr_max)
        )
        safe_means = torch.where(valid, means, means.new_full((), self.rr_min))
        safe_sigmas = torch.where(
            valid, sigmas, sigmas.new_full((), self.candidate_sigma)
        ).clamp(min=self.rr_step, max=self.rr_max - self.rr_min)
        bins = self.rr_bins.to(device=means.device, dtype=means.dtype)
        log_weight = -0.5 * (
            (bins.view(1, 1, 1, -1) - safe_means.unsqueeze(-1))
            / safe_sigmas.unsqueeze(-1)
        ).square()
        log_weight = log_weight - log_weight.amax(dim=-1, keepdim=True)
        probability = log_weight.exp() * valid.unsqueeze(-1).to(dtype=means.dtype)
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        valid_count = valid.sum(dim=1).clamp_min(1).to(dtype=means.dtype)
        aggregated = probability.sum(dim=1) / valid_count.unsqueeze(-1)
        multiplier_valid = valid.any(dim=1)
        aggregated = aggregated * multiplier_valid.unsqueeze(-1).to(dtype=means.dtype)
        return aggregated, multiplier_valid, safe_means

    def _sample_direct_frequency(self, local_logits: Tensor) -> Tensor:
        bins_hz = self.rr_bins.to(
            device=local_logits.device, dtype=local_logits.dtype
        ) / 60.0
        position = (
            (bins_hz - self.spectral_frequency_min_hz)
            / (self.spectral_frequency_max_hz - self.spectral_frequency_min_hz)
            * (local_logits.shape[-1] - 1)
        ).clamp(0.0, float(local_logits.shape[-1] - 1))
        left = position.floor().long()
        right = (left + 1).clamp_max(local_logits.shape[-1] - 1)
        weight = (position - left.to(position.dtype)).view(1, 1, -1)
        return (
            local_logits.index_select(-1, left) * (1.0 - weight)
            + local_logits.index_select(-1, right) * weight
        ).squeeze(1)

    def _candidate_priors_on_spectral_grid(
        self, candidate_priors: Tensor, frequency_bins: int
    ) -> Tensor:
        """Sample RR-grid candidate priors on the cache's physical Hz grid."""

        spectral_rr = torch.linspace(
            60.0 * self.spectral_frequency_min_hz,
            60.0 * self.spectral_frequency_max_hz,
            int(frequency_bins),
            device=candidate_priors.device,
            dtype=candidate_priors.dtype,
        )
        position = (spectral_rr - self.rr_min) / self.rr_step
        valid = (position >= 0.0) & (position <= float(self.num_rr_bins - 1))
        position = position.clamp(0.0, float(self.num_rr_bins - 1))
        left = position.floor().long()
        right = (left + 1).clamp_max(self.num_rr_bins - 1)
        weight = (position - left.to(position.dtype)).view(1, 1, -1)
        sampled = (
            candidate_priors.index_select(-1, left) * (1.0 - weight)
            + candidate_priors.index_select(-1, right) * weight
        )
        return sampled * valid.view(1, 1, -1).to(dtype=sampled.dtype)

    @staticmethod
    def _sample_spike_rate(spikes: Tensor) -> Tensor:
        return spikes.flatten(start_dim=1).mean(dim=1)

    def forward(
        self,
        spectra: Tensor,
        attributes: Tensor,
        base_prediction: Tensor,
        base_std: Tensor,
        base_features: Tensor | None = None,
        classical_rr: Tensor | None = None,
        classical_std: Tensor | None = None,
        radar_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        (
            spectra,
            attributes,
            base_prediction,
            base_std,
            base_features,
            radar_mask,
        ) = self._validate_inputs(
            spectra,
            attributes,
            base_prediction,
            base_std,
            base_features,
            radar_mask,
        )
        spectra, available = apply_radar_dropout(
            spectra,
            radar_mask,
            p=self.radar_dropout_p,
            training=self.training,
            ensure_one=True,
        )
        mask_shape = (*available.shape, 1, 1, 1)
        attributes = attributes * available.view(mask_shape).to(dtype=attributes.dtype)
        batch, radars, _, _, frequencies = spectra.shape

        radar_feature, reliability, component_quality = self.radar_encoder(
            spectra, attributes
        )
        channels = radar_feature.shape[1]
        radar_feature = radar_feature.reshape(batch, radars, channels, frequencies)
        reliability = reliability.reshape(batch, radars)
        radar_weights = _masked_radar_softmax(reliability, available)
        radar_feature = radar_feature * available[:, :, None, None].to(
            dtype=radar_feature.dtype
        )
        fused = (radar_feature * radar_weights[:, :, None, None]).sum(dim=1)

        base_probability, base_valid = self._base_posterior(base_prediction, base_std)
        candidate_priors, candidate_valid, candidate_centers = self._candidate_priors(
            base_prediction, base_std, classical_rr, classical_std
        )
        candidate_current = self._candidate_priors_on_spectral_grid(
            candidate_priors, frequencies
        )
        base_current = self.input_projection(fused) + self.candidate_projection(
            candidate_current
        )
        if self.base_feature_encoder is not None:
            if base_features is None:  # guarded by _validate_inputs
                raise RuntimeError("base feature validation failed")
            scalar_embedding = self.base_feature_encoder(base_features)
            base_current = base_current + self.scalar_current(scalar_embedding).unsqueeze(-1)
        else:
            scalar_embedding = fused.new_zeros((batch, 0))
        base_current = (
            F.softplus(self.input_scale_unconstrained) + 0.5
        ) * base_current

        input_membrane = torch.zeros_like(base_current)
        block_states: list[tuple[Tensor, Tensor] | None] = [
            None for _ in self.spiking_blocks
        ]
        accumulated_spikes = torch.zeros_like(base_current)
        layer_sample_rates = [
            base_current.new_zeros((batch,))
            for _ in range(1 + 2 * len(self.spiking_blocks))
        ]
        final_membrane = input_membrane
        for _ in range(self.simulation_steps):
            spikes, input_membrane = self.input_lif(base_current, input_membrane)
            layer_sample_rates[0] += self._sample_spike_rate(spikes)
            for block_index, block in enumerate(self.spiking_blocks):
                spikes, block_states[block_index], internal = block.forward_step(
                    spikes, block_states[block_index]
                )
                layer_sample_rates[1 + 2 * block_index] += self._sample_spike_rate(
                    internal[0]
                )
                layer_sample_rates[2 + 2 * block_index] += self._sample_spike_rate(
                    internal[1]
                )
            accumulated_spikes += spikes
            if block_states[-1] is None:  # impossible with >=1 block
                raise RuntimeError("spiking block state was not initialized")
            final_membrane = block_states[-1][1]

        spike_rate_feature = accumulated_spikes / float(self.simulation_steps)
        readout = torch.cat((spike_rate_feature, torch.tanh(final_membrane)), dim=1)
        local_logits = self._sample_direct_frequency(self.local_logit_head(readout))
        coverage = available.float().mean(dim=1, keepdim=True).to(dtype=readout.dtype)
        global_feature = torch.cat(
            (
                readout.mean(dim=-1),
                readout.amax(dim=-1),
                scalar_embedding,
                coverage,
                candidate_valid.to(dtype=readout.dtype),
            ),
            dim=1,
        )
        source_logits = local_logits + 0.25 * self.global_logit_head(global_feature)
        source_probability = source_logits.softmax(dim=-1)
        raw_gate_logits = self.gate_head(global_feature).squeeze(1)
        any_available = available.any(dim=1)
        mixture_gate = raw_gate_logits.sigmoid() * any_available.to(dtype=readout.dtype)
        probabilities = (
            (1.0 - mixture_gate.unsqueeze(1)) * base_probability
            + mixture_gate.unsqueeze(1) * source_probability
        )
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(
            1.0e-8
        )
        logits = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        bins = self.rr_bins.to(device=logits.device, dtype=logits.dtype)
        expected_rr, rr_std = _posterior_statistics(probabilities, bins)
        base_expected_rr, base_posterior_std = _posterior_statistics(
            base_probability, bins
        )
        source_expected_rr, source_std = _posterior_statistics(source_probability, bins)
        map_rr = bins[probabilities.argmax(dim=1)]
        topk_probability, topk_index = probabilities.topk(
            min(5, self.num_rr_bins), dim=1
        )
        posterior_entropy = -(
            probabilities
            * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        ).sum(dim=1)
        quality_logits = torch.where(
            any_available,
            self.quality_head(global_feature).squeeze(1),
            logits.new_full((), -20.0),
        )
        quality = quality_logits.sigmoid() * any_available.to(dtype=logits.dtype)

        layer_rates = torch.stack(layer_sample_rates, dim=1) / float(
            self.simulation_steps
        )
        spike_rate_per_sample = layer_rates.mean(dim=1)
        reliability_output = reliability.masked_fill(~available, -20.0)
        return {
            "logits": logits,
            "rr_logits": logits,
            "probabilities": probabilities,
            "rr_probs": probabilities,
            "expected_rr": expected_rr,
            "rr": expected_rr,
            "map_rr": map_rr,
            "rr_std": rr_std,
            "log_variance": (rr_std.square().clamp_min(1.0e-8)).log(),
            "topk_rr": bins[topk_index],
            "topk_probability": topk_probability,
            "posterior_entropy": posterior_entropy,
            "quality_logits": quality_logits,
            "quality": quality,
            "base_probabilities": base_probability,
            "base_expected_rr": base_expected_rr,
            "base_posterior_std": base_posterior_std,
            "base_valid": base_valid,
            "source_logits": source_logits,
            "source_probabilities": source_probability,
            "source_expected_rr": source_expected_rr,
            "source_std": source_std,
            "mixture_gate_logits": raw_gate_logits,
            "mixture_gate": mixture_gate,
            "candidate_priors": candidate_priors,
            "candidate_valid_mask": candidate_valid,
            "candidate_centers_rr": candidate_centers,
            "component_quality": component_quality,
            "radar_weights": radar_weights,
            "radar_reliability_logits": reliability_output,
            "radar_mask": available,
            "embedding": readout,
            "spike_rate": spike_rate_per_sample.mean(),
            "spike_rate_per_sample": spike_rate_per_sample,
            "layer_spike_rates": layer_rates.mean(dim=0),
            "layer_spike_rates_per_sample": layer_rates,
        }


# Short aliases for experiment configuration files.
SVDSourceRRSNN = SourceSeparatedRRSNN
SourceSeparatedTriRadarRRSNN = SourceSeparatedRRSNN
SourceSeparatedSVDRRSNN = SourceSeparatedRRSNN


__all__ = [
    "SVD_NUM_RR_BINS",
    "SVD_RR_MAX",
    "SVD_RR_MIN",
    "SVD_RR_STEP",
    "SVDSourceRRSNN",
    "SourceSeparatedRRSNN",
    "SourceSeparatedSVDRRSNN",
    "SourceSeparatedTriRadarRRSNN",
]
