"""Neural models for direct respiratory-rate estimation from UWB maps.

The models in this module consume a pre-computed range--frequency map with
shape ``[batch, radar, frequency, range]``.  A value of ``True`` in
``radar_mask`` means that the corresponding radar is available.  Both models
are deliberately agnostic to the number of frequency and range bins.

``SharedRadarCNNTeacher`` is the accuracy-oriented ANN teacher.
``TriRadarRRSNN`` keeps the shared spatial encoder and radar fusion in the
analog domain and uses a compact snnTorch temporal/frequency backbone.  This
hybrid design avoids rate-coding the full two-dimensional input while still
making the expensive sequence-processing stage spike driven.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError as exc:  # pragma: no cover - exercised only in a broken env
    raise ImportError(
        "TriRadarRRSNN requires snntorch. Install the project's runtime "
        "dependencies before importing snn_rr.models."
    ) from exc


DEFAULT_RR_MIN = 4.0
DEFAULT_RR_MAX = 60.0
DEFAULT_NUM_RR_BINS = 225  # 0.25 breaths/min over the inclusive 4--60 range


def make_rr_bins(
    rr_min: float = DEFAULT_RR_MIN,
    rr_max: float = DEFAULT_RR_MAX,
    num_bins: int = DEFAULT_NUM_RR_BINS,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return monotonically increasing respiratory-rate bin centers."""

    if not math.isfinite(rr_min) or not math.isfinite(rr_max):
        raise ValueError("rr_min and rr_max must be finite")
    if rr_max <= rr_min:
        raise ValueError("rr_max must be greater than rr_min")
    if num_bins < 2:
        raise ValueError("num_bins must be at least two")
    return torch.linspace(
        float(rr_min),
        float(rr_max),
        int(num_bins),
        device=device,
        dtype=dtype or torch.float32,
    )


def gaussian_soft_targets(
    target_rr: Tensor,
    rr_bins: Tensor | None = None,
    *,
    sigma: float | Tensor = 1.0,
    rr_min: float = DEFAULT_RR_MIN,
    rr_max: float = DEFAULT_RR_MAX,
    num_bins: int = DEFAULT_NUM_RR_BINS,
    eps: float = 1e-8,
) -> Tensor:
    """Create normalized Gaussian soft labels for RR distribution learning.

    ``target_rr`` may have any leading shape.  The returned tensor appends one
    RR-bin dimension.  A tensor ``sigma`` is broadcast against ``target_rr``;
    this is useful when reference-label uncertainty differs per sample.
    Targets outside the bin range naturally become boundary-concentrated soft
    labels rather than producing an invalid distribution.
    """

    if not torch.is_floating_point(target_rr):
        target_rr = target_rr.float()
    if rr_bins is None:
        rr_bins = make_rr_bins(
            rr_min,
            rr_max,
            num_bins,
            device=target_rr.device,
            dtype=target_rr.dtype,
        )
    else:
        if rr_bins.ndim != 1 or rr_bins.numel() < 2:
            raise ValueError("rr_bins must be a one-dimensional tensor")
        rr_bins = rr_bins.to(device=target_rr.device, dtype=target_rr.dtype)

    sigma_tensor = torch.as_tensor(
        sigma, device=target_rr.device, dtype=target_rr.dtype
    )
    if torch.any(~torch.isfinite(sigma_tensor)) or torch.any(sigma_tensor <= 0):
        raise ValueError("sigma must contain finite positive values")

    z = (rr_bins - target_rr.unsqueeze(-1)) / sigma_tensor.unsqueeze(-1)
    # Subtracting the maximum keeps very narrow or out-of-range targets stable.
    log_weight = -0.5 * z.square()
    log_weight = log_weight - log_weight.amax(dim=-1, keepdim=True)
    weight = log_weight.exp()
    return weight / weight.sum(dim=-1, keepdim=True).clamp_min(eps)


def expected_rr_from_logits(
    logits: Tensor,
    rr_bins: Tensor | None = None,
    *,
    dim: int = -1,
    rr_min: float = DEFAULT_RR_MIN,
    rr_max: float = DEFAULT_RR_MAX,
) -> Tensor:
    """Return the differentiable expectation of an RR logit distribution."""

    if logits.ndim == 0:
        raise ValueError("logits must have at least one dimension")
    dim = dim if dim >= 0 else logits.ndim + dim
    if not 0 <= dim < logits.ndim:
        raise ValueError("dim is outside logits")
    n_bins = logits.shape[dim]
    if rr_bins is None:
        rr_bins = make_rr_bins(
            rr_min,
            rr_max,
            n_bins,
            device=logits.device,
            dtype=logits.dtype,
        )
    elif rr_bins.ndim != 1 or rr_bins.numel() != n_bins:
        raise ValueError("rr_bins length must match the selected logits dimension")
    else:
        rr_bins = rr_bins.to(device=logits.device, dtype=logits.dtype)

    shape = [1] * logits.ndim
    shape[dim] = n_bins
    probs = logits.softmax(dim=dim)
    return (probs * rr_bins.view(shape)).sum(dim=dim)


def _coerce_radar_mask(x: Tensor, radar_mask: Tensor | None) -> Tensor:
    batch, num_radars = x.shape[:2]
    if radar_mask is None:
        return torch.ones((batch, num_radars), dtype=torch.bool, device=x.device)
    if radar_mask.shape != (batch, num_radars):
        raise ValueError(
            f"radar_mask must have shape {(batch, num_radars)}, got "
            f"{tuple(radar_mask.shape)}"
        )
    return radar_mask.to(device=x.device) > 0


def apply_radar_dropout(
    x: Tensor,
    radar_mask: Tensor | None = None,
    *,
    p: float = 0.0,
    training: bool = True,
    ensure_one: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Mask complete radar views and return ``(masked_x, available_mask)``.

    Existing missing radars are never re-enabled.  With ``ensure_one=True``, a
    sample that originally had at least one radar retains one random available
    view even if Bernoulli dropout selected all of them.
    """

    if x.ndim < 3:
        raise ValueError("x must have batch and radar dimensions")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    available = _coerce_radar_mask(x, radar_mask)

    if training and p > 0.0:
        random = torch.rand(
            available.shape, device=x.device, generator=generator
        )
        kept = available & (random >= p)
        if ensure_one:
            lost_all = available.any(dim=1) & ~kept.any(dim=1)
            if lost_all.any():
                # The same random draw chooses a uniformly random available
                # radar via argmax while avoiding a CPU/GPU synchronization.
                rescue_score = random.masked_fill(~available, -1.0)
                rescue_index = rescue_score.argmax(dim=1)
                rows = torch.arange(x.shape[0], device=x.device)
                rescue = torch.zeros_like(kept)
                rescue[rows, rescue_index] = lost_all
                kept = kept | rescue
        available = kept

    view_shape = (*available.shape, *([1] * (x.ndim - 2)))
    return x * available.view(view_shape).to(dtype=x.dtype), available


class RadarDropout(nn.Module):
    """Module wrapper around :func:`apply_radar_dropout`."""

    def __init__(self, p: float = 0.0, ensure_one: bool = True) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = float(p)
        self.ensure_one = bool(ensure_one)

    def forward(
        self, x: Tensor, radar_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        return apply_radar_dropout(
            x,
            radar_mask,
            p=self.p,
            training=self.training,
            ensure_one=self.ensure_one,
        )


# Short, discoverable functional alias.
radar_dropout = apply_radar_dropout
gaussian_soft_target = gaussian_soft_targets


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualSpatialBlock(nn.Module):
    """2-D residual block that downsamples range but never frequency."""

    def __init__(self, in_channels: int, out_channels: int, range_stride: int) -> None:
        super().__init__()
        stride = (1, range_stride)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(5, 5),
            stride=stride,
            padding=(2, 2),
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.depthwise = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        if in_channels != out_channels or range_stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=False
                ),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.pointwise(self.depthwise(x)))
        return F.silu(x + residual)


class _SharedRadarSpatialEncoder(nn.Module):
    """Shared 2-D encoder with range attention and radar reliability."""

    def __init__(self, channels: Sequence[int], *, signal_channels: int = 1) -> None:
        super().__init__()
        if len(channels) < 2 or any(c < 4 for c in channels):
            raise ValueError("channels must contain at least two widths >= 4")
        if signal_channels < 1:
            raise ValueError("signal_channels must be positive")
        channels = tuple(int(c) for c in channels)
        # Signal branch(es), normalized frequency position and normalized range
        # position.  The feature cache stores the raw-power and optional I/Q
        # phase-power maps next to each other along its final axis; callers can
        # expose those as separate signal channels instead of creating a false
        # neighbourhood across the branch boundary.
        self.stem = nn.Sequential(
            nn.Conv2d(signal_channels + 2, channels[0], 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.SiLU(),
        )
        blocks: list[nn.Module] = []
        for index, (c_in, c_out) in enumerate(zip(channels[:-1], channels[1:])):
            blocks.append(
                _ResidualSpatialBlock(
                    c_in, c_out, range_stride=2 if index < 3 else 1
                )
            )
        self.blocks = nn.Sequential(*blocks)
        out_channels = channels[-1]
        attention_hidden = max(8, out_channels // 4)
        self.range_attention = nn.Sequential(
            nn.Conv2d(out_channels, attention_hidden, 1),
            nn.SiLU(),
            nn.Conv2d(attention_hidden, 1, 1),
        )
        self.reliability = nn.Sequential(
            nn.Linear(2 * out_channels, max(16, out_channels // 2)),
            nn.SiLU(),
            nn.Linear(max(16, out_channels // 2), 1),
        )
        self.out_channels = out_channels

    @staticmethod
    def _coordinate_channels(x: Tensor) -> tuple[Tensor, Tensor]:
        freq = torch.linspace(-1.0, 1.0, x.shape[-2], device=x.device, dtype=x.dtype)
        distance = torch.linspace(
            -1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype
        )
        freq = freq.view(1, 1, -1, 1).expand(x.shape[0], 1, -1, x.shape[-1])
        distance = distance.view(1, 1, 1, -1).expand(
            x.shape[0], 1, x.shape[-2], -1
        )
        return freq, distance

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # x is [B * radar, signal branch, F, R].  Frequency is never pooled or
        # strided.
        freq_coord, range_coord = self._coordinate_channels(x)
        feature = self.blocks(self.stem(torch.cat((x, freq_coord, range_coord), dim=1)))
        attention = self.range_attention(feature).softmax(dim=-1)
        token = (feature * attention).sum(dim=-1)  # [BR, C, F]
        pooled = torch.cat((token.mean(dim=-1), token.amax(dim=-1)), dim=1)
        reliability = self.reliability(pooled).squeeze(-1)
        return token, reliability, attention.squeeze(1)


def _masked_radar_softmax(scores: Tensor, available: Tensor) -> Tensor:
    masked = scores.masked_fill(~available, -1.0e4)
    weight = masked.softmax(dim=1) * available.to(dtype=scores.dtype)
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(1.0e-8)


class _AuxiliaryFusion(nn.Module):
    def __init__(self, channels: int, aux_dim: int) -> None:
        super().__init__()
        if aux_dim < 0:
            raise ValueError("aux_dim cannot be negative")
        self.aux_dim = int(aux_dim)
        self.channels = int(channels)
        if self.aux_dim:
            aux_channels = min(64, max(16, channels // 2))
            self.aux_encoder = nn.Sequential(
                nn.Linear(self.aux_dim, aux_channels),
                nn.SiLU(),
                nn.Linear(aux_channels, aux_channels),
                nn.SiLU(),
            )
            self.fuse = nn.Sequential(
                nn.Conv1d(channels + aux_channels, channels, 1, bias=False),
                nn.GroupNorm(_group_count(channels), channels),
                nn.SiLU(),
            )
            self.aux_channels = aux_channels
        else:
            self.aux_encoder = None
            self.fuse = nn.Identity()
            self.aux_channels = 0

    def forward(self, feature: Tensor, aux: Tensor | None) -> Tensor:
        if not self.aux_dim:
            if aux is not None and aux.numel() > 0:
                raise ValueError("aux was provided, but the model was built with aux_dim=0")
            return feature
        if aux is None:
            aux = feature.new_zeros((feature.shape[0], self.aux_dim))
        if aux.shape != (feature.shape[0], self.aux_dim):
            raise ValueError(
                f"aux must have shape {(feature.shape[0], self.aux_dim)}, got "
                f"{tuple(aux.shape)}"
            )
        aux = aux.to(device=feature.device, dtype=feature.dtype)
        aux_feature = self.aux_encoder(aux).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        return self.fuse(torch.cat((feature, aux_feature), dim=1))


def _split_cached_auxiliary(
    aux: Tensor, base_aux_dim: int
) -> tuple[Tensor, Tensor]:
    """Return topology-preserving spectra and global cached diagnostics.

    The base schema is six per-radar spectra, 24 radar scalars, two fused
    spectra and five consensus values.  Strictly-causal history, when present,
    follows ``base_aux_dim`` and belongs only to the global output.
    """

    if base_aux_dim < 37 or (base_aux_dim - 29) % 8:
        raise ValueError(
            "base_aux_dim does not match the cached three-radar schema"
        )
    if aux.ndim != 2 or aux.shape[1] < base_aux_dim:
        raise ValueError("aux is shorter than its declared base layout")
    frequency_bins = (base_aux_dim - 29) // 8
    per_radar_end = 6 * frequency_bins
    scalar_end = per_radar_end + 24
    fused_end = scalar_end + 2 * frequency_bins
    consensus_end = fused_end + 5
    if consensus_end != base_aux_dim:
        raise RuntimeError("cached auxiliary layout arithmetic is inconsistent")
    per_radar = aux[:, :per_radar_end].reshape(-1, 6, frequency_bins)
    fused = aux[:, scalar_end:fused_end].reshape(-1, 2, frequency_bins)
    spectra = torch.cat((per_radar, fused), dim=1)
    global_values = torch.cat(
        (
            aux[:, per_radar_end:scalar_end],
            aux[:, fused_end:consensus_end],
            aux[:, base_aux_dim:],
        ),
        dim=1,
    )
    return spectra, global_values


def _align_auxiliary_spectra_to_map(feature: Tensor, target_bins: int) -> Tensor:
    """Align full FFT evidence to the cache's pair-pooled map frequency grid."""

    if target_bins < 1:
        raise ValueError("target_bins must be positive")
    if feature.shape[-1] == target_bins:
        return feature
    if feature.shape[-1] in {2 * target_bins, 2 * target_bins + 1}:
        return F.avg_pool1d(
            feature[..., : 2 * target_bins], kernel_size=2, stride=2
        )
    return F.interpolate(
        feature,
        size=target_bins,
        mode="linear",
        align_corners=False,
    )


class _StructuredAuxiliaryFusion(nn.Module):
    """Fuse cached spectra without destroying their frequency topology.

    ``fuse_auxiliary_features`` stores six per-radar spectra, 24 radar
    scalars, two fused spectra and five consensus values.  The legacy flat
    MLP compresses all of those columns to one vector and broadcasts it over
    frequency.  This module instead applies a small Conv1D encoder to the
    eight aligned spectra and keeps only genuinely global diagnostics in an
    MLP.  Any strictly-causal history columns appended after ``base_aux_dim``
    join the global path.
    """

    def __init__(
        self,
        channels: int,
        aux_dim: int,
        base_aux_dim: int,
        *,
        exact_frequency_alignment: bool = False,
    ) -> None:
        super().__init__()
        if aux_dim < 1 or not 0 < base_aux_dim <= aux_dim:
            raise ValueError(
                "structured auxiliary fusion requires 0 < base_aux_dim <= aux_dim"
            )
        if base_aux_dim < 37 or (base_aux_dim - 29) % 8:
            raise ValueError(
                "base_aux_dim does not match the cached three-radar schema"
            )
        self.channels = int(channels)
        self.aux_dim = int(aux_dim)
        self.base_aux_dim = int(base_aux_dim)
        self.frequency_bins = (self.base_aux_dim - 29) // 8
        self.exact_frequency_alignment = bool(exact_frequency_alignment)
        spectral_channels = min(64, max(16, channels // 2))
        self.spectral_encoder = nn.Sequential(
            nn.Conv1d(8, spectral_channels, 5, padding=2, bias=False),
            nn.GroupNorm(_group_count(spectral_channels), spectral_channels),
            nn.SiLU(),
            nn.Conv1d(
                spectral_channels,
                spectral_channels,
                5,
                padding=2,
                groups=spectral_channels,
                bias=False,
            ),
            nn.Conv1d(spectral_channels, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
        )
        global_dim = 29 + (self.aux_dim - self.base_aux_dim)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, max(16, channels // 2)),
            nn.SiLU(),
            nn.Linear(max(16, channels // 2), channels),
        )
        self.spectral_gate = nn.Conv1d(2 * channels, channels, 1)
        self.output_norm = nn.GroupNorm(_group_count(channels), channels)

    def _split(self, aux: Tensor) -> tuple[Tensor, Tensor]:
        return _split_cached_auxiliary(aux, self.base_aux_dim)

    def forward(self, feature: Tensor, aux: Tensor | None) -> Tensor:
        if aux is None:
            aux = feature.new_zeros((feature.shape[0], self.aux_dim))
        if aux.shape != (feature.shape[0], self.aux_dim):
            raise ValueError(
                f"aux must have shape {(feature.shape[0], self.aux_dim)}, got "
                f"{tuple(aux.shape)}"
            )
        aux = aux.to(device=feature.device, dtype=feature.dtype)
        spectra, global_values = self._split(aux)
        spectral_feature = self.spectral_encoder(spectra)
        if spectral_feature.shape[-1] != feature.shape[-1]:
            if self.exact_frequency_alignment:
                # Feature construction pair-averages the first 2*F full FFT
                # bins and may leave one trailing auxiliary-only bin.  Mirror
                # that exact physical alignment rather than warping endpoints.
                spectral_feature = _align_auxiliary_spectra_to_map(
                    spectral_feature, feature.shape[-1]
                )
            else:
                # Compatibility path for checkpoints trained before the exact
                # cache-grid alignment was introduced.
                spectral_feature = F.interpolate(
                    spectral_feature,
                    size=feature.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
        gate = torch.sigmoid(
            self.spectral_gate(torch.cat((feature, spectral_feature), dim=1))
        )
        global_feature = self.global_encoder(global_values).unsqueeze(-1)
        return F.silu(
            self.output_norm(feature + gate * spectral_feature + 0.25 * global_feature)
        )


class _HarmonicAuxiliaryLogitHead(nn.Module):
    """Score RR candidates from direct and subharmonic spectral evidence.

    Radar motion can expose a strong component at ``f/3`` or ``f/4`` while
    the respiratory fundamental at ``f`` is weak.  A conventional local
    frequency head interprets that component only as the lower rate.  This
    head explicitly evaluates every RR candidate against the train-standardized
    cached spectra
    at ``f``, ``f/2``, ``f/3`` and ``f/4`` so the learned mixer can represent
    the competing harmonic hypotheses.

    Every convolution is bias-free, making a zeroed auxiliary vector produce
    exactly zero residual logits.  That property preserves the model's
    conservative behaviour when aggregate auxiliary evidence is masked after
    a radar goes missing.
    """

    _DIVISORS = (1.0, 2.0, 3.0, 4.0)

    def __init__(
        self,
        *,
        aux_dim: int,
        base_aux_dim: int,
        rr_min: float,
        rr_max: float,
        num_rr_bins: int,
        auxiliary_frequency_min_hz: float,
        auxiliary_frequency_max_hz: float,
        hidden_channels: int = 24,
        alias_gate: bool = False,
    ) -> None:
        super().__init__()
        if aux_dim < 1 or not 0 < base_aux_dim <= aux_dim:
            raise ValueError(
                "harmonic auxiliary head requires 0 < base_aux_dim <= aux_dim"
            )
        if base_aux_dim < 37 or (base_aux_dim - 29) % 8:
            raise ValueError(
                "base_aux_dim does not match the cached three-radar schema"
            )
        if not (
            math.isfinite(auxiliary_frequency_min_hz)
            and math.isfinite(auxiliary_frequency_max_hz)
            and auxiliary_frequency_max_hz > auxiliary_frequency_min_hz
        ):
            raise ValueError(
                "auxiliary frequency limits must be finite and increasing"
            )
        if hidden_channels < 4:
            raise ValueError("harmonic hidden_channels must be at least four")

        self.aux_dim = int(aux_dim)
        self.base_aux_dim = int(base_aux_dim)
        self.frequency_bins = (self.base_aux_dim - 29) // 8
        self.num_rr_bins = int(num_rr_bins)
        self.auxiliary_frequency_min_hz = float(auxiliary_frequency_min_hz)
        self.auxiliary_frequency_max_hz = float(auxiliary_frequency_max_hz)
        self.alias_gate = bool(alias_gate)

        hidden_channels = int(hidden_channels)
        self.spectral_encoder = nn.Sequential(
            nn.Conv1d(8, hidden_channels, 5, padding=2, bias=False),
            nn.SiLU(),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.SiLU(),
        )
        self.hypothesis_mixer = nn.Sequential(
            nn.Conv1d(
                len(self._DIVISORS) * hidden_channels,
                2 * hidden_channels,
                1,
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv1d(2 * hidden_channels, 1, 1, bias=False),
        )
        if self.alias_gate:
            self.alias_pool_bins = 8
            global_dim = 29 + (self.aux_dim - self.base_aux_dim)
            alias_feature_dim = (
                hidden_channels * (2 + self.alias_pool_bins) + global_dim
            )
            alias_hidden = max(16, hidden_channels)
            self.alias_gate_head: nn.Sequential | None = nn.Sequential(
                nn.Linear(alias_feature_dim, alias_hidden),
                nn.SiLU(),
                nn.Linear(alias_hidden, 1),
            )
            nn.init.constant_(self.alias_gate_head[-1].bias, -1.0)
        else:
            self.alias_pool_bins = 0
            self.alias_gate_head = None
        # Begin as a small residual correction so this path cannot erase the
        # useful direct-frequency solution at initialization.
        initial_gain = 0.50
        self.gain_unconstrained = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_gain)), dtype=torch.float32)
        )

        rr_hz = make_rr_bins(rr_min, rr_max, num_rr_bins) / 60.0
        sample_frequency = torch.stack(
            [rr_hz / divisor for divisor in self._DIVISORS], dim=0
        )
        position = (
            (sample_frequency - self.auxiliary_frequency_min_hz)
            / (
                self.auxiliary_frequency_max_hz
                - self.auxiliary_frequency_min_hz
            )
            * (self.frequency_bins - 1)
        )
        valid = (position >= 0.0) & (position <= float(self.frequency_bins - 1))
        if not bool(valid[0].all()):
            raise ValueError(
                "the direct RR candidate grid must lie entirely inside the "
                "auxiliary frequency range"
            )
        clamped = position.clamp(0.0, float(self.frequency_bins - 1))
        left = clamped.floor().long()
        right = (left + 1).clamp_max(self.frequency_bins - 1)
        weight = clamped - left.to(dtype=clamped.dtype)
        self.register_buffer("sample_left", left, persistent=False)
        self.register_buffer("sample_right", right, persistent=False)
        self.register_buffer("sample_weight", weight, persistent=False)
        self.register_buffer("sample_valid", valid, persistent=False)

    def _sample_hypotheses(self, feature: Tensor) -> Tensor:
        """Return a ``[B, 4*C, RR-bin]`` tensor of sampled evidence."""

        samples: list[Tensor] = []
        for hypothesis in range(len(self._DIVISORS)):
            left = self.sample_left[hypothesis]
            right = self.sample_right[hypothesis]
            weight = self.sample_weight[hypothesis].to(dtype=feature.dtype).view(
                1, 1, -1
            )
            sampled = (
                feature.index_select(-1, left) * (1.0 - weight)
                + feature.index_select(-1, right) * weight
            )
            valid = self.sample_valid[hypothesis].to(dtype=feature.dtype).view(
                1, 1, -1
            )
            samples.append(sampled * valid)
        return torch.cat(samples, dim=1)

    def forward(
        self, aux: Tensor | None, reference: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if aux is None:
            zero = reference.new_zeros((reference.shape[0], self.num_rr_bins))
            return zero, reference.new_zeros(()), reference.new_zeros(reference.shape[0])
        if aux.shape != (reference.shape[0], self.aux_dim):
            raise ValueError(
                f"aux must have shape {(reference.shape[0], self.aux_dim)}, got "
                f"{tuple(aux.shape)}"
            )
        aux = aux.to(device=reference.device, dtype=reference.dtype)
        spectra, global_values = _split_cached_auxiliary(aux, self.base_aux_dim)
        encoded = self.spectral_encoder(spectra)
        raw_logits = self.hypothesis_mixer(
            self._sample_hypotheses(encoded)
        ).squeeze(1)
        gain = F.softplus(self.gain_unconstrained).to(dtype=raw_logits.dtype)
        if self.alias_gate_head is not None:
            alias_feature = torch.cat(
                (
                    encoded.mean(dim=-1),
                    encoded.amax(dim=-1),
                    F.adaptive_avg_pool1d(
                        encoded, self.alias_pool_bins
                    ).flatten(start_dim=1),
                    global_values,
                ),
                dim=1,
            )
            alias_logits = self.alias_gate_head(alias_feature).squeeze(1)
            gated_logits = alias_logits.sigmoid().unsqueeze(1) * gain * raw_logits
        else:
            alias_logits = raw_logits.new_zeros(raw_logits.shape[0])
            gated_logits = gain * raw_logits
        return gated_logits, gain, alias_logits


class _FrequencyResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=5,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        value, gate = self.conv1(F.silu(self.norm1(x))).chunk(2, dim=1)
        x = value * torch.sigmoid(gate)
        x = self.dropout(self.conv2(F.silu(self.norm2(x))))
        return x + residual


class _DistributionModelBase(nn.Module):
    """Shared validation and probabilistic-output helpers."""

    def _init_distribution(
        self,
        *,
        rr_min: float,
        rr_max: float,
        num_rr_bins: int,
        num_bins: int | None,
        num_radars: int,
        radar_dropout_p: float,
        input_branches: int,
        input_frequency_min_hz: float | None,
        input_frequency_max_hz: float | None,
    ) -> None:
        if num_bins is not None:
            num_rr_bins = num_bins
        if num_radars < 1:
            raise ValueError("num_radars must be positive")
        self.num_radars = int(num_radars)
        if input_branches < 1:
            raise ValueError("input_branches must be positive")
        self.input_branches = int(input_branches)
        if (input_frequency_min_hz is None) != (input_frequency_max_hz is None):
            raise ValueError(
                "input_frequency_min_hz and input_frequency_max_hz must be set together"
            )
        if input_frequency_min_hz is not None:
            if not (
                math.isfinite(input_frequency_min_hz)
                and math.isfinite(input_frequency_max_hz)
                and input_frequency_max_hz > input_frequency_min_hz
            ):
                raise ValueError("input frequency limits must be finite and increasing")
        self.input_frequency_min_hz = input_frequency_min_hz
        self.input_frequency_max_hz = input_frequency_max_hz
        self.num_rr_bins = int(num_rr_bins)
        self.rr_min = float(rr_min)
        self.rr_max = float(rr_max)
        self.register_buffer(
            "rr_bins",
            make_rr_bins(rr_min, rr_max, self.num_rr_bins),
            persistent=True,
        )
        self.radar_dropout_layer = RadarDropout(radar_dropout_p, ensure_one=True)

    def _branch_input(self, x: Tensor) -> Tensor:
        """Expose cache branches as channels for the shared spatial encoder."""

        batch, radars, frequencies, stored_range = x.shape
        if stored_range % self.input_branches:
            raise ValueError(
                f"range dimension {stored_range} is not divisible by "
                f"input_branches={self.input_branches}"
            )
        range_bins = stored_range // self.input_branches
        return (
            x.reshape(batch, radars, frequencies, self.input_branches, range_bins)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * radars, self.input_branches, frequencies, range_bins)
        )

    @staticmethod
    def _mask_aux_for_available(aux: Tensor | None, available: Tensor) -> Tensor | None:
        """Prevent full-radar aggregate features from bypassing a radar mask.

        The cache's auxiliary vector contains per-radar spectra as well as
        fused three-radar summaries.  Those fused fields cannot be recomputed
        generically inside the model.  The conservative behaviour is therefore
        to use auxiliary evidence only when every expected radar is available;
        masked/dropout examples are learned from the remaining map inputs.
        """

        if aux is None:
            return None
        return aux * available.all(dim=1, keepdim=True).to(dtype=aux.dtype)

    def _resample_frequency_logits(self, local_logits: Tensor) -> Tensor:
        """Map physical input frequencies to RR bins without axis warping."""

        if self.input_frequency_min_hz is None:
            return F.interpolate(
                local_logits,
                size=self.num_rr_bins,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        input_bins = local_logits.shape[-1]
        if input_bins == 1:
            return local_logits.expand(-1, -1, self.num_rr_bins).squeeze(1)
        rr_hz = self.rr_bins.to(
            device=local_logits.device, dtype=local_logits.dtype
        ) / 60.0
        position = (
            (rr_hz - float(self.input_frequency_min_hz))
            / (float(self.input_frequency_max_hz) - float(self.input_frequency_min_hz))
            * (input_bins - 1)
        )
        position = position.clamp(0.0, float(input_bins - 1))
        left = position.floor().long()
        right = (left + 1).clamp_max(input_bins - 1)
        weight = (position - left.to(position.dtype)).view(1, 1, -1)
        return (
            local_logits.index_select(-1, left) * (1.0 - weight)
            + local_logits.index_select(-1, right) * weight
        ).squeeze(1)

    def _validate_input(
        self, x: Tensor, radar_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if x.ndim != 4:
            raise ValueError(
                "input must have shape [batch, radar, frequency, range], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.num_radars:
            raise ValueError(
                f"model expects {self.num_radars} radars, got {x.shape[1]}"
            )
        if x.shape[-2] < 1 or x.shape[-1] < 1:
            raise ValueError("frequency and range dimensions cannot be empty")
        if not torch.is_floating_point(x):
            x = x.float()
        return self.radar_dropout_layer(x, radar_mask)

    def _probabilistic_outputs(
        self,
        logits: Tensor,
        raw_log_variance: Tensor,
        raw_quality_logits: Tensor,
        available: Tensor,
    ) -> dict[str, Tensor]:
        probs = logits.softmax(dim=-1)
        bins = self.rr_bins.to(dtype=logits.dtype)
        expected = (probs * bins).sum(dim=-1)
        map_index = probs.argmax(dim=-1)
        map_rr = bins[map_index]
        topk_probability, topk_index = probs.topk(
            k=min(5, probs.shape[-1]), dim=-1
        )
        topk_rr = bins[topk_index]
        posterior_entropy = -(
            probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
        ).sum(dim=-1)
        log_variance = 6.0 * torch.tanh(raw_log_variance / 6.0)
        any_available = available.any(dim=1)
        quality_logits = torch.where(
            any_available, raw_quality_logits, raw_quality_logits.new_full((), -20.0)
        )
        quality = quality_logits.sigmoid()
        return {
            "logits": logits,
            "rr_logits": logits,
            "probabilities": probs,
            "rr_probs": probs,
            "expected_rr": expected,
            "rr": expected,
            "map_rr": map_rr,
            "topk_rr": topk_rr,
            "topk_probability": topk_probability,
            "posterior_entropy": posterior_entropy,
            "log_variance": log_variance,
            "rr_std": (0.5 * log_variance).exp(),
            "quality_logits": quality_logits,
            "quality": quality,
        }


class SharedRadarCNNTeacher(_DistributionModelBase):
    """Shared-radar 2-D CNN teacher with frequency-preserving prediction.

    The range axis is compressed with learned attention.  The frequency axis is
    retained at every spatial stage and is interpolated only once, when mapping
    the final frequency evidence to the requested RR-bin grid.
    """

    def __init__(
        self,
        *,
        num_radars: int = 3,
        rr_min: float = DEFAULT_RR_MIN,
        rr_max: float = DEFAULT_RR_MAX,
        num_rr_bins: int = DEFAULT_NUM_RR_BINS,
        num_bins: int | None = None,
        spatial_channels: Sequence[int] = (32, 48, 72, 96),
        frequency_dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
        radar_dropout_p: float = 0.15,
        aux_dim: int = 0,
        structured_auxiliary: bool = False,
        aux_base_dim: int | None = None,
        exact_auxiliary_alignment: bool = False,
        harmonic_auxiliary: bool = False,
        alias_gated_harmonic: bool = False,
        auxiliary_frequency_min_hz: float | None = None,
        auxiliary_frequency_max_hz: float | None = None,
        input_branches: int = 1,
        input_frequency_min_hz: float | None = None,
        input_frequency_max_hz: float | None = None,
    ) -> None:
        super().__init__()
        self._init_distribution(
            rr_min=rr_min,
            rr_max=rr_max,
            num_rr_bins=num_rr_bins,
            num_bins=num_bins,
            num_radars=num_radars,
            radar_dropout_p=radar_dropout_p,
            input_branches=input_branches,
            input_frequency_min_hz=input_frequency_min_hz,
            input_frequency_max_hz=input_frequency_max_hz,
        )
        self.spatial_encoder = _SharedRadarSpatialEncoder(
            spatial_channels, signal_channels=self.input_branches
        )
        channels = self.spatial_encoder.out_channels
        if structured_auxiliary:
            if aux_base_dim is None:
                raise ValueError(
                    "aux_base_dim is required when structured_auxiliary=True"
                )
            self.auxiliary_fusion = _StructuredAuxiliaryFusion(
                channels,
                aux_dim,
                aux_base_dim,
                exact_frequency_alignment=exact_auxiliary_alignment,
            )
        else:
            self.auxiliary_fusion = _AuxiliaryFusion(channels, aux_dim)
        self.structured_auxiliary = bool(structured_auxiliary)
        self.exact_auxiliary_alignment = bool(exact_auxiliary_alignment)
        self.aux_base_dim = aux_base_dim
        if alias_gated_harmonic and not harmonic_auxiliary:
            raise ValueError(
                "alias_gated_harmonic requires harmonic_auxiliary=True"
            )
        if harmonic_auxiliary:
            if aux_base_dim is None:
                raise ValueError(
                    "aux_base_dim is required when harmonic_auxiliary=True"
                )
            if (
                auxiliary_frequency_min_hz is None
                or auxiliary_frequency_max_hz is None
            ):
                raise ValueError(
                    "explicit auxiliary frequency limits are required when "
                    "harmonic_auxiliary=True; the auxiliary and map grids differ"
                )
            self.harmonic_logit_head: _HarmonicAuxiliaryLogitHead | None = (
                _HarmonicAuxiliaryLogitHead(
                    aux_dim=aux_dim,
                    base_aux_dim=aux_base_dim,
                    rr_min=rr_min,
                    rr_max=rr_max,
                    num_rr_bins=self.num_rr_bins,
                    auxiliary_frequency_min_hz=auxiliary_frequency_min_hz,
                    auxiliary_frequency_max_hz=auxiliary_frequency_max_hz,
                    hidden_channels=min(24, max(16, channels // 4)),
                    alias_gate=alias_gated_harmonic,
                )
            )
        else:
            self.harmonic_logit_head = None
        self.harmonic_auxiliary = bool(harmonic_auxiliary)
        self.alias_gated_harmonic = bool(alias_gated_harmonic)
        self.frequency_encoder = nn.Sequential(
            *(
                _FrequencyResidualBlock(channels, int(d), dropout)
                for d in frequency_dilations
            )
        )
        self.local_logit_head = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.Conv1d(channels, max(16, channels // 2), 1),
            nn.SiLU(),
            nn.Conv1d(max(16, channels // 2), 1, 1),
        )
        global_dim = 2 * channels + 1
        self.global_logit_head = nn.Sequential(
            nn.Linear(global_dim, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, self.num_rr_bins),
        )
        self.uncertainty_quality_head = nn.Sequential(
            nn.Linear(global_dim, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 2),
        )
        self.rr_logit_bias = nn.Parameter(torch.zeros(self.num_rr_bins))

    def forward(
        self,
        x: Tensor,
        radar_mask: Tensor | None = None,
        aux: Tensor | None = None,
    ) -> dict[str, Tensor]:
        x, available = self._validate_input(x, radar_mask)
        batch, radars, freq_bins, _ = x.shape
        token, reliability, range_attention = self.spatial_encoder(
            self._branch_input(x)
        )
        channels = token.shape[1]
        token = token.reshape(batch, radars, channels, freq_bins)
        reliability = reliability.reshape(batch, radars)
        radar_weight = _masked_radar_softmax(reliability, available)
        token = token * available[:, :, None, None].to(dtype=token.dtype)
        fused = (token * radar_weight[:, :, None, None]).sum(dim=1)
        aux = self._mask_aux_for_available(aux, available)
        fused = self.auxiliary_fusion(fused, aux)
        fused = self.frequency_encoder(fused)

        local_logits = self.local_logit_head(fused)
        local_logits = self._resample_frequency_logits(local_logits)
        if self.harmonic_logit_head is not None:
            harmonic_logits, harmonic_gain, alias_logits = (
                self.harmonic_logit_head(aux, fused)
            )
        else:
            harmonic_logits = local_logits.new_zeros(local_logits.shape)
            harmonic_gain = local_logits.new_zeros(())
            alias_logits = local_logits.new_zeros(local_logits.shape[0])
        coverage = available.float().mean(dim=1, keepdim=True).to(fused.dtype)
        global_feature = torch.cat(
            (fused.mean(dim=-1), fused.amax(dim=-1), coverage), dim=1
        )
        logits = (
            local_logits
            + harmonic_logits
            + 0.25 * self.global_logit_head(global_feature)
            + self.rr_logit_bias
        )
        uq = self.uncertainty_quality_head(global_feature)
        output = self._probabilistic_outputs(
            logits, uq[:, 0], uq[:, 1], available
        )
        output.update(
            {
                "radar_weights": radar_weight,
                "radar_reliability_logits": reliability,
                "radar_mask": available,
                "range_attention": range_attention.reshape(
                    batch, radars, freq_bins, -1
                )
                * available[:, :, None, None].to(range_attention.dtype),
                "embedding": fused,
            }
        )
        if self.harmonic_logit_head is not None:
            output.update(
                {
                    "harmonic_logits": harmonic_logits,
                    "harmonic_gain": harmonic_gain,
                    "alias_logits": alias_logits,
                    "alias_probability": alias_logits.sigmoid(),
                }
            )
        return output


class _SpikingResidualFrequencyBlock(nn.Module):
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
        if kernel_size % 2 == 0:
            raise ValueError("spiking kernel_size must be odd")
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
        self.residual_gain = nn.Parameter(torch.tensor(0.5))

    def forward_step(
        self,
        spikes: Tensor,
        state: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        if state is None:
            mem1 = torch.zeros_like(spikes)
            mem2 = torch.zeros_like(spikes)
        else:
            mem1, mem2 = state
        current1 = self.norm1(self.conv1(spikes))
        spikes1, mem1 = self.lif1(current1, mem1)
        current2 = self.norm2(self.conv2(spikes1))
        current2 = current2 + torch.sigmoid(self.residual_gain) * spikes
        spikes2, mem2 = self.lif2(current2, mem2)
        return spikes2, (mem1, mem2), (spikes1, spikes2)


class TriRadarRRSNN(_DistributionModelBase):
    """Compact hybrid SNN for direct respiratory-rate regression.

    A shared analog CNN performs range compression for each radar.  Reliability
    fusion is followed by a PLIF-style (learnable-beta LIF) residual frequency
    network unrolled for ``simulation_steps``.  The model returns differentiable
    spike-rate statistics so an energy/sparsity term can be added to the loss.
    """

    def __init__(
        self,
        *,
        num_radars: int = 3,
        rr_min: float = DEFAULT_RR_MIN,
        rr_max: float = DEFAULT_RR_MAX,
        num_rr_bins: int = DEFAULT_NUM_RR_BINS,
        num_bins: int | None = None,
        spatial_channels: Sequence[int] = (24, 40, 64),
        hidden_channels: int = 96,
        num_spiking_blocks: int = 2,
        simulation_steps: int = 12,
        num_steps: int | None = None,
        beta: float = 0.9,
        learn_beta: bool = True,
        learn_threshold: bool = True,
        spike_kernel_size: int = 5,
        dropout: float = 0.05,
        radar_dropout_p: float = 0.2,
        aux_dim: int = 0,
        structured_auxiliary: bool = False,
        aux_base_dim: int | None = None,
        exact_auxiliary_alignment: bool = False,
        harmonic_auxiliary: bool = False,
        alias_gated_harmonic: bool = False,
        auxiliary_frequency_min_hz: float | None = None,
        auxiliary_frequency_max_hz: float | None = None,
        input_branches: int = 1,
        input_frequency_min_hz: float | None = None,
        input_frequency_max_hz: float | None = None,
    ) -> None:
        super().__init__()
        if num_steps is not None:
            simulation_steps = num_steps
        if not 1 <= simulation_steps <= 64:
            raise ValueError("simulation_steps must be in [1, 64]")
        if hidden_channels < 8 or num_spiking_blocks < 1:
            raise ValueError("hidden_channels >= 8 and num_spiking_blocks >= 1 required")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be between zero and one")
        self._init_distribution(
            rr_min=rr_min,
            rr_max=rr_max,
            num_rr_bins=num_rr_bins,
            num_bins=num_bins,
            num_radars=num_radars,
            radar_dropout_p=radar_dropout_p,
            input_branches=input_branches,
            input_frequency_min_hz=input_frequency_min_hz,
            input_frequency_max_hz=input_frequency_max_hz,
        )
        self.simulation_steps = int(simulation_steps)
        self.spatial_encoder = _SharedRadarSpatialEncoder(
            spatial_channels, signal_channels=self.input_branches
        )
        spatial_width = self.spatial_encoder.out_channels
        if structured_auxiliary:
            if aux_base_dim is None:
                raise ValueError(
                    "aux_base_dim is required when structured_auxiliary=True"
                )
            self.auxiliary_fusion = _StructuredAuxiliaryFusion(
                spatial_width,
                aux_dim,
                aux_base_dim,
                exact_frequency_alignment=exact_auxiliary_alignment,
            )
        else:
            self.auxiliary_fusion = _AuxiliaryFusion(spatial_width, aux_dim)
        self.structured_auxiliary = bool(structured_auxiliary)
        self.exact_auxiliary_alignment = bool(exact_auxiliary_alignment)
        self.aux_base_dim = aux_base_dim
        if alias_gated_harmonic and not harmonic_auxiliary:
            raise ValueError(
                "alias_gated_harmonic requires harmonic_auxiliary=True"
            )
        if harmonic_auxiliary:
            if aux_base_dim is None:
                raise ValueError(
                    "aux_base_dim is required when harmonic_auxiliary=True"
                )
            if (
                auxiliary_frequency_min_hz is None
                or auxiliary_frequency_max_hz is None
            ):
                raise ValueError(
                    "explicit auxiliary frequency limits are required when "
                    "harmonic_auxiliary=True; the auxiliary and map grids differ"
                )
            self.harmonic_logit_head: _HarmonicAuxiliaryLogitHead | None = (
                _HarmonicAuxiliaryLogitHead(
                    aux_dim=aux_dim,
                    base_aux_dim=aux_base_dim,
                    rr_min=rr_min,
                    rr_max=rr_max,
                    num_rr_bins=self.num_rr_bins,
                    auxiliary_frequency_min_hz=auxiliary_frequency_min_hz,
                    auxiliary_frequency_max_hz=auxiliary_frequency_max_hz,
                    hidden_channels=16,
                    alias_gate=alias_gated_harmonic,
                )
            )
        else:
            self.harmonic_logit_head = None
        self.harmonic_auxiliary = bool(harmonic_auxiliary)
        self.alias_gated_harmonic = bool(alias_gated_harmonic)
        self.input_projection = nn.Sequential(
            nn.Conv1d(spatial_width, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
        )
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
            _SpikingResidualFrequencyBlock(
                hidden_channels,
                beta=beta,
                kernel_size=spike_kernel_size,
                learn_beta=learn_beta,
                learn_threshold=learn_threshold,
            )
            for _ in range(num_spiking_blocks)
        )
        readout_channels = 2 * hidden_channels
        self.local_logit_head = nn.Sequential(
            nn.Conv1d(readout_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, 1, 1),
        )
        global_dim = 2 * readout_channels + 1
        self.global_logit_head = nn.Sequential(
            nn.Linear(global_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, self.num_rr_bins),
        )
        self.uncertainty_quality_head = nn.Sequential(
            nn.Linear(global_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )
        self.rr_logit_bias = nn.Parameter(torch.zeros(self.num_rr_bins))

    @staticmethod
    def _sample_spike_rate(spikes: Tensor) -> Tensor:
        return spikes.flatten(start_dim=1).mean(dim=1)

    def forward(
        self,
        x: Tensor,
        radar_mask: Tensor | None = None,
        aux: Tensor | None = None,
    ) -> dict[str, Tensor]:
        x, available = self._validate_input(x, radar_mask)
        batch, radars, freq_bins, _ = x.shape
        token, reliability, range_attention = self.spatial_encoder(
            self._branch_input(x)
        )
        channels = token.shape[1]
        token = token.reshape(batch, radars, channels, freq_bins)
        reliability = reliability.reshape(batch, radars)
        radar_weight = _masked_radar_softmax(reliability, available)
        token = token * available[:, :, None, None].to(token.dtype)
        fused = (token * radar_weight[:, :, None, None]).sum(dim=1)
        aux = self._mask_aux_for_available(aux, available)
        fused = self.auxiliary_fusion(fused, aux)
        base_current = self.input_projection(fused)
        input_scale = F.softplus(self.input_scale_unconstrained) + 0.5
        base_current = input_scale * base_current

        input_mem = torch.zeros_like(base_current)
        block_states: list[tuple[Tensor, Tensor] | None] = [
            None for _ in self.spiking_blocks
        ]
        accumulated_output = torch.zeros_like(base_current)
        # One statistic for the input LIF and two for each residual block.
        layer_sample_rates = [
            base_current.new_zeros((batch,))
            for _ in range(1 + 2 * len(self.spiking_blocks))
        ]
        final_membrane = input_mem

        for _ in range(self.simulation_steps):
            spikes, input_mem = self.input_lif(base_current, input_mem)
            layer_sample_rates[0] = layer_sample_rates[0] + self._sample_spike_rate(spikes)
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
            accumulated_output = accumulated_output + spikes
            final_membrane = block_states[-1][1]

        spike_rate_feature = accumulated_output / float(self.simulation_steps)
        readout_feature = torch.cat(
            (spike_rate_feature, torch.tanh(final_membrane)), dim=1
        )
        local_logits = self.local_logit_head(readout_feature)
        local_logits = self._resample_frequency_logits(local_logits)
        if self.harmonic_logit_head is not None:
            harmonic_logits, harmonic_gain, alias_logits = (
                self.harmonic_logit_head(aux, readout_feature)
            )
        else:
            harmonic_logits = local_logits.new_zeros(local_logits.shape)
            harmonic_gain = local_logits.new_zeros(())
            alias_logits = local_logits.new_zeros(local_logits.shape[0])
        coverage = available.float().mean(dim=1, keepdim=True).to(readout_feature.dtype)
        global_feature = torch.cat(
            (
                readout_feature.mean(dim=-1),
                readout_feature.amax(dim=-1),
                coverage,
            ),
            dim=1,
        )
        logits = (
            local_logits
            + harmonic_logits
            + 0.25 * self.global_logit_head(global_feature)
            + self.rr_logit_bias
        )
        uq = self.uncertainty_quality_head(global_feature)
        output = self._probabilistic_outputs(
            logits, uq[:, 0], uq[:, 1], available
        )

        layer_sample_rate_tensor = torch.stack(layer_sample_rates, dim=1) / float(
            self.simulation_steps
        )
        spike_rate_per_sample = layer_sample_rate_tensor.mean(dim=1)
        output.update(
            {
                "radar_weights": radar_weight,
                "radar_reliability_logits": reliability,
                "radar_mask": available,
                "range_attention": range_attention.reshape(
                    batch, radars, freq_bins, -1
                )
                * available[:, :, None, None].to(range_attention.dtype),
                "embedding": readout_feature,
                "spike_rate": spike_rate_per_sample.mean(),
                "spike_rate_per_sample": spike_rate_per_sample,
                "layer_spike_rates": layer_sample_rate_tensor.mean(dim=0),
                "layer_spike_rates_per_sample": layer_sample_rate_tensor,
            }
        )
        if self.harmonic_logit_head is not None:
            output.update(
                {
                    "harmonic_logits": harmonic_logits,
                    "harmonic_gain": harmonic_gain,
                    "alias_logits": alias_logits,
                    "alias_probability": alias_logits.sigmoid(),
                }
            )
        return output


# Compatibility aliases: keep one canonical implementation while making the
# intended roles obvious to training scripts and notebooks.
RadarCNNTeacher = SharedRadarCNNTeacher
TriRadarANNTeacher = SharedRadarCNNTeacher
ANNTeacher = SharedRadarCNNTeacher
HybridTriRadarRRSNN = TriRadarRRSNN


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters in ``model``."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "ANNTeacher",
    "DEFAULT_NUM_RR_BINS",
    "DEFAULT_RR_MAX",
    "DEFAULT_RR_MIN",
    "HybridTriRadarRRSNN",
    "RadarCNNTeacher",
    "RadarDropout",
    "SharedRadarCNNTeacher",
    "TriRadarANNTeacher",
    "TriRadarRRSNN",
    "apply_radar_dropout",
    "count_trainable_parameters",
    "expected_rr_from_logits",
    "gaussian_soft_target",
    "gaussian_soft_targets",
    "make_rr_bins",
    "radar_dropout",
]
