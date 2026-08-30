"""Spiking model over ordered, full-window source-separated signals.

Unlike :mod:`snn_rr.svd_models`, which treats frequency bins as repeated SNN
simulation steps, this module advances its neuron state in the physical order
of the 32-second slow-time window.  The nominal cache input has shape
``[B, 3, V, C, 320]``.  A causal anti-alias filter reduces it to 80 steps, and
every LIF/PLIF/ALIF state update then corresponds to one successive 0.4-second
component coordinate.  The cached component signals themselves are computed
with full-window detrending, standardization, and SVD.  Model-level prefix
invariance therefore applies only after feature extraction; it is not a claim
that raw-to-component preprocessing is streaming-prefix causal.  The complete
predictor is causal at the 32-second window end because no sample after that
endpoint is used.

The network is deliberately a safe residual estimator.  It predicts a
posterior over the classical-rate hypotheses x1, x2, x3 and x4, a bounded
residual for each hypothesis, and its uncertainty.  A negatively initialized
gate mixes this source posterior with the supplied base Gaussian, making an
untrained network stay very close to the base estimator.  With no available
radar the fallback is exact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from snntorch import surrogate
except ImportError as exc:  # pragma: no cover - broken runtime only
    raise ImportError(
        "TemporalSourceSeparatedRRSNN requires snntorch. Install the project "
        "runtime dependencies before importing snn_rr.svd_temporal_models."
    ) from exc

from .models import make_rr_bins


TEMPORAL_RR_MIN = 6.0
TEMPORAL_RR_MAX = 45.0
TEMPORAL_RR_STEP = 0.25
TEMPORAL_NUM_RR_BINS = (
    int(round((TEMPORAL_RR_MAX - TEMPORAL_RR_MIN) / TEMPORAL_RR_STEP)) + 1
)


def _inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _masked_softmax(scores: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Masked softmax with exact all-zero output for an empty mask."""

    masked = scores.masked_fill(~mask, -1.0e4)
    weights = masked.softmax(dim=dim) * mask.to(dtype=scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0e-8)


def _posterior_statistics(probabilities: Tensor, bins: Tensor) -> tuple[Tensor, Tensor]:
    expected = (probabilities * bins.unsqueeze(0)).sum(dim=-1)
    variance = (
        probabilities * (bins.unsqueeze(0) - expected.unsqueeze(1)).square()
    ).sum(dim=-1)
    return expected, variance.clamp_min(torch.finfo(probabilities.dtype).eps).sqrt()


class _AttributeComponentCompressor(nn.Module):
    """Compress component/variant/radar axes without mixing physical time.

    Only the five label-free SVD diagnostics determine component and radar
    attention.  A shared 1x1 convolution then projects the weighted component
    channels independently at every time sample.  Consequently this stage
    cannot leak information from a future slow-time sample into an earlier one.
    """

    def __init__(
        self,
        *,
        num_variants: int,
        num_components: int,
        channels: int,
    ) -> None:
        super().__init__()
        if num_variants < 1 or num_components < 1 or channels < 4:
            raise ValueError("variants/components must be positive and channels >= 4")
        self.num_variants = int(num_variants)
        self.num_components = int(num_components)
        flattened = self.num_variants * self.num_components
        attention_hidden = max(8, channels // 2)
        self.attribute_attention = nn.Sequential(
            nn.Linear(5, attention_hidden),
            nn.SiLU(),
            nn.Linear(attention_hidden, 1),
        )
        self.component_projection = nn.Conv1d(
            flattened, int(channels), kernel_size=1, bias=False
        )
        self.channel_norm = nn.LayerNorm(int(channels))
        self.radar_attention = nn.Sequential(
            nn.Linear(10, attention_hidden),
            nn.SiLU(),
            nn.Linear(attention_hidden, 1),
        )
        self.out_channels = int(channels)

    @staticmethod
    def _normalize_attributes(attributes: Tensor) -> Tensor:
        diagnostics = attributes[..., :4].clamp(0.0, 1.0)
        # The supported slow-time band is below 1 Hz; retain ordering while
        # bounding corrupt cache metadata.
        peak_frequency = attributes[..., 4:].clamp(0.0, 2.0) / 2.0
        return torch.cat((diagnostics, peak_frequency), dim=-1)

    @staticmethod
    def _deterministic_quality(attributes: Tensor) -> Tensor:
        energy = attributes[..., 0].clamp(0.0, 1.0)
        band = attributes[..., 1].clamp(0.0, 1.0)
        concentration = attributes[..., 2].clamp(0.0, 1.0)
        entropy = attributes[..., 3].clamp(0.0, 1.0)
        return (
            torch.sqrt((energy * band).clamp_min(1.0e-8))
            * concentration
            * (1.0 - 0.5 * entropy)
        )

    def forward(
        self,
        component_signals: Tensor,
        attributes: Tensor,
        available: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, radars, variants, components, time_steps = component_signals.shape
        normalized = self._normalize_attributes(attributes)
        learned_score = self.attribute_attention(normalized).squeeze(-1)
        quality = self._deterministic_quality(attributes)
        # The deterministic term gives physically meaningful attention before
        # any training while the small learned correction can adapt it.
        component_score = learned_score + quality.clamp_min(1.0e-6).log()
        component_attention = component_score.flatten(2).softmax(dim=-1)
        component_attention = component_attention.reshape(
            batch, radars, variants, components
        )
        component_attention = component_attention * available[:, :, None, None].to(
            dtype=component_attention.dtype
        )

        weighted = component_signals * component_attention.unsqueeze(-1)
        # Preserve the average input scale as V*C changes.
        weighted = weighted * float(variants * components)
        projected = self.component_projection(
            weighted.reshape(batch * radars, variants * components, time_steps)
        )
        projected = projected.transpose(1, 2)
        projected = F.silu(self.channel_norm(projected)).transpose(1, 2)
        projected = projected.reshape(
            batch, radars, self.out_channels, time_steps
        )

        summary = torch.cat(
            (normalized.mean(dim=(-3, -2)), normalized.amax(dim=(-3, -2))), dim=-1
        )
        radar_logits = self.radar_attention(summary).squeeze(-1)
        radar_weights = _masked_softmax(radar_logits, available, dim=1)
        fused = (projected * radar_weights[:, :, None, None]).sum(dim=1)
        return fused, radar_weights, component_attention


class _CausalAntiAliasDownsample(nn.Module):
    """Positive low-pass depthwise Conv1d followed by stride-4 sampling.

    With a nine-tap filter and left padding of five, output step ``j`` observes
    input samples no later than ``4*j + 3``.  Thus 320 samples produce exactly
    80 chronological steps, including the final sample, and a 160-sample
    prefix produces exactly the same first 40 outputs as the full sequence.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        stride: int = 4,
        kernel_size: int = 9,
    ) -> None:
        super().__init__()
        if stride < 1 or kernel_size < stride or kernel_size % 2 == 0:
            raise ValueError("anti-alias kernel must be odd and at least the stride")
        self.channels = int(channels)
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        # A positive triangular FIR is a stable anti-alias initializer.  Its
        # softmax parameterization preserves positivity and unit DC gain.
        midpoint = kernel_size // 2 + 1
        triangle = torch.tensor(
            list(range(1, midpoint + 1)) + list(range(midpoint - 1, 0, -1)),
            dtype=torch.float32,
        )
        triangle = triangle / triangle.sum()
        self.kernel_logits = nn.Parameter(
            triangle.clamp_min(1.0e-6).log().repeat(int(channels), 1)
        )
        self.pointwise = nn.Conv1d(
            int(channels), int(hidden_channels), kernel_size=1, bias=False
        )
        self.norm = nn.LayerNorm(int(hidden_channels))

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[1] != self.channels:
            raise ValueError("anti-alias input must have shape [batch, channels, time]")
        # Align every output with the end of its four-sample interval.
        left_padding = self.kernel_size - self.stride
        padded = F.pad(values, (left_padding, 0))
        kernel = self.kernel_logits.softmax(dim=-1).unsqueeze(1)
        filtered = F.conv1d(
            padded,
            kernel.to(device=values.device, dtype=values.dtype),
            stride=self.stride,
            groups=self.channels,
        )
        # A single 1x1 Conv1d may select a different GEMM reduction kernel when
        # the physical-time length changes (for example 80 steps versus a
        # sealed 40-step prefix).  The resulting few-ULP difference is not a
        # future-data leak, but it breaks the bitwise prefix contract and makes
        # CPU results depend on the process thread state.  Accumulating the
        # fixed channel axis explicitly keeps the arithmetic order identical
        # for every shared time step while retaining the checkpoint-compatible
        # Conv1d parameter layout.
        pointwise_weight = self.pointwise.weight[:, :, 0].to(
            device=filtered.device,
            dtype=filtered.dtype,
        )
        projected_channels = (
            filtered[:, 0:1]
            * pointwise_weight[:, 0].reshape(1, -1, 1)
        )
        for channel_index in range(1, self.channels):
            projected_channels = projected_channels + (
                filtered[:, channel_index : channel_index + 1]
                * pointwise_weight[:, channel_index].reshape(1, -1, 1)
            )
        projected = projected_channels.transpose(1, 2)
        return self.norm(projected).transpose(1, 2)


class _RecurrentSpikingCell(nn.Module):
    """A compact chronological LIF, parametric-LIF, or adaptive-LIF cell."""

    _VALID_TYPES = frozenset(("lif", "plif", "alif"))

    def __init__(
        self,
        channels: int,
        *,
        cell_type: str,
        beta: float = 0.9,
        adaptation_decay: float = 0.95,
        adaptation_strength: float = 0.5,
    ) -> None:
        super().__init__()
        cell_type = str(cell_type).lower()
        if cell_type not in self._VALID_TYPES:
            raise ValueError(f"cell_type must be one of {sorted(self._VALID_TYPES)}")
        if not 0.0 < beta < 1.0 or not 0.0 < adaptation_decay < 1.0:
            raise ValueError("beta and adaptation_decay must be in (0, 1)")
        if adaptation_strength < 0:
            raise ValueError("adaptation_strength cannot be negative")
        self.channels = int(channels)
        self.cell_type = cell_type
        beta_logit = torch.tensor(_inverse_sigmoid(float(beta)), dtype=torch.float32)
        if cell_type == "lif":
            self.register_buffer("beta_logit", beta_logit, persistent=True)
        else:
            self.beta_logit = nn.Parameter(beta_logit)
        self.threshold_unconstrained = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        if cell_type == "alif":
            self.adaptation_decay_logit = nn.Parameter(
                torch.tensor(
                    _inverse_sigmoid(float(adaptation_decay)), dtype=torch.float32
                )
            )
            self.adaptation_strength_unconstrained = nn.Parameter(
                torch.tensor(
                    _inverse_softplus(max(float(adaptation_strength), 1.0e-4)),
                    dtype=torch.float32,
                )
            )
        else:
            self.register_buffer(
                "adaptation_decay_logit",
                torch.tensor(_inverse_sigmoid(float(adaptation_decay))),
                persistent=False,
            )
            self.register_buffer(
                "adaptation_strength_unconstrained",
                torch.tensor(_inverse_softplus(1.0e-4)),
                persistent=False,
            )
        self.spike_function = surrogate.fast_sigmoid(slope=25)

    def initial_state(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        zero = torch.zeros_like(reference)
        return zero, zero.clone()

    def forward_step(
        self,
        current: Tensor,
        state: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        membrane, adaptation = state
        beta = self.beta_logit.sigmoid().to(dtype=current.dtype)
        base_threshold = F.softplus(self.threshold_unconstrained).to(
            dtype=current.dtype
        )
        if self.cell_type == "alif":
            strength = F.softplus(self.adaptation_strength_unconstrained).to(
                dtype=current.dtype
            )
            threshold = base_threshold + strength * adaptation
        else:
            threshold = base_threshold
        membrane = beta * membrane + current
        spikes = self.spike_function(membrane - threshold)
        # Detaching the reset prevents unstable reset-gradient loops while the
        # emitted spike retains its surrogate gradient.
        membrane = membrane - spikes.detach() * threshold
        if self.cell_type == "alif":
            decay = self.adaptation_decay_logit.sigmoid().to(dtype=current.dtype)
            adaptation = decay * adaptation + spikes
        else:
            adaptation = torch.zeros_like(adaptation)
        return spikes, (membrane, adaptation)


class _ChronologicalSpikingEncoder(nn.Module):
    """Advance a stack of recurrent spiking cells one physical step at a time."""

    def __init__(
        self,
        channels: int,
        *,
        cell_types: Sequence[str],
        beta: float,
        dropout: float,
    ) -> None:
        super().__init__()
        normalized_types = tuple(str(value).lower() for value in cell_types)
        if not normalized_types:
            raise ValueError("cell_types cannot be empty")
        self.cell_types = normalized_types
        self.synapses = nn.ModuleList(
            nn.Linear(int(channels), int(channels)) for _ in normalized_types
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(int(channels)) for _ in normalized_types
        )
        self.cells = nn.ModuleList(
            _RecurrentSpikingCell(
                int(channels), cell_type=cell_type, beta=float(beta)
            )
            for cell_type in normalized_types
        )
        self.dropout = nn.Dropout(float(dropout))
        self.readout = nn.Sequential(
            nn.Linear(3 * int(channels), int(channels)),
            nn.SiLU(),
        )
        self.channels = int(channels)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        if values.ndim != 3 or values.shape[1] != self.channels:
            raise ValueError("spiking encoder input must have shape [B, H, time]")
        sequence = values.transpose(1, 2)
        batch = sequence.shape[0]
        states = [
            cell.initial_state(sequence.new_zeros((batch, self.channels)))
            for cell in self.cells
        ]
        state_outputs: list[Tensor] = []
        spike_sums = sequence.new_zeros((batch, len(self.cells)))
        for time_index in range(sequence.shape[1]):
            analog = sequence[:, time_index]
            current = analog
            last_membrane = analog.new_zeros(analog.shape)
            last_spikes = analog.new_zeros(analog.shape)
            for layer_index, (synapse, norm, cell) in enumerate(
                zip(self.synapses, self.norms, self.cells, strict=True)
            ):
                current = norm(synapse(current))
                spikes, states[layer_index] = cell.forward_step(
                    current, states[layer_index]
                )
                spike_sums[:, layer_index] += spikes.mean(dim=1)
                last_spikes = spikes
                last_membrane = states[layer_index][0]
                current = self.dropout(spikes)
            state_outputs.append(
                self.readout(
                    torch.cat((analog, last_spikes, torch.tanh(last_membrane)), dim=1)
                )
            )
        state_sequence = torch.stack(state_outputs, dim=1)
        per_sample_layer_rates = spike_sums / float(sequence.shape[1])
        return state_sequence, per_sample_layer_rates


def _causal_window_mean(sequence: Tensor, window: int) -> Tensor:
    """Return a rolling mean whose element ``t`` only consumes ``<= t``."""

    if window < 1:
        raise ValueError("window must be positive")
    cumulative = torch.cat(
        (sequence.new_zeros((*sequence.shape[:1], 1, sequence.shape[-1])),
         sequence.cumsum(dim=1)),
        dim=1,
    )
    time_steps = sequence.shape[1]
    end = torch.arange(1, time_steps + 1, device=sequence.device)
    start = (end - int(window)).clamp_min(0)
    total = cumulative.index_select(1, end) - cumulative.index_select(1, start)
    count = (end - start).to(dtype=sequence.dtype).view(1, time_steps, 1)
    return total / count


class TemporalSourceSeparatedRRSNN(nn.Module):
    """Causal multi-resolution SNN over cached SVD component signals.

    ``component_signals`` normally has shape ``[B, 3, V, C, 320]``.  Prefixes
    divisible by four are also accepted to make causality auditable.  At the
    nominal 10-Hz sample rate the four readout windows represent the recent
    4, 8, 16 and 32 seconds.
    """

    _MULTIPLIERS = (1.0, 2.0, 3.0, 4.0)

    def __init__(
        self,
        *,
        num_variants: int = 10,
        num_components: int = 12,
        num_radars: int = 3,
        rr_min: float = TEMPORAL_RR_MIN,
        rr_max: float = TEMPORAL_RR_MAX,
        rr_step: float = TEMPORAL_RR_STEP,
        compressor_channels: int = 24,
        hidden_channels: int = 48,
        cell_types: Sequence[str] = ("lif", "plif", "alif"),
        beta: float = 0.9,
        downsample_stride: int = 4,
        downsample_kernel_size: int = 9,
        input_sample_rate_hz: float = 10.0,
        max_residual_bpm: float = 1.5,
        minimum_candidate_std: float = 0.35,
        maximum_candidate_std: float = 8.0,
        initial_candidate_std: float = 1.0,
        initial_gate_bias: float = -8.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if num_variants < 1 or num_components < 1 or num_radars < 1:
            raise ValueError("variant/component/radar counts must be positive")
        if compressor_channels < 4 or hidden_channels < 8:
            raise ValueError("compressor_channels >= 4 and hidden_channels >= 8 required")
        if not (
            math.isfinite(rr_min)
            and math.isfinite(rr_max)
            and math.isfinite(rr_step)
            and rr_max > rr_min
            and rr_step > 0
        ):
            raise ValueError("RR grid values must be finite and increasing")
        span_steps = (float(rr_max) - float(rr_min)) / float(rr_step)
        if not math.isclose(span_steps, round(span_steps), abs_tol=1.0e-7):
            raise ValueError("rr_step must exactly partition the inclusive RR range")
        if not math.isfinite(input_sample_rate_hz) or input_sample_rate_hz <= 0:
            raise ValueError("input_sample_rate_hz must be positive")
        if not math.isfinite(max_residual_bpm) or max_residual_bpm <= 0:
            raise ValueError("max_residual_bpm must be positive")
        if not (
            0 < minimum_candidate_std < initial_candidate_std <= maximum_candidate_std
        ):
            raise ValueError("candidate std limits/initial value are inconsistent")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_variants = int(num_variants)
        self.num_components = int(num_components)
        self.num_radars = int(num_radars)
        self.rr_min = float(rr_min)
        self.rr_max = float(rr_max)
        self.rr_step = float(rr_step)
        self.num_rr_bins = int(round(span_steps)) + 1
        self.downsample_stride = int(downsample_stride)
        self.input_sample_rate_hz = float(input_sample_rate_hz)
        self.max_residual_bpm = float(max_residual_bpm)
        self.minimum_candidate_std = float(minimum_candidate_std)
        self.maximum_candidate_std = float(maximum_candidate_std)
        self.register_buffer(
            "rr_bins",
            make_rr_bins(rr_min, rr_max, self.num_rr_bins),
            persistent=True,
        )
        self.register_buffer(
            "candidate_multipliers",
            torch.tensor(self._MULTIPLIERS, dtype=torch.float32),
            persistent=False,
        )

        self.compressor = _AttributeComponentCompressor(
            num_variants=self.num_variants,
            num_components=self.num_components,
            channels=int(compressor_channels),
        )
        self.downsample = _CausalAntiAliasDownsample(
            int(compressor_channels),
            int(hidden_channels),
            stride=self.downsample_stride,
            kernel_size=int(downsample_kernel_size),
        )
        self.spiking_encoder = _ChronologicalSpikingEncoder(
            int(hidden_channels),
            cell_types=cell_types,
            beta=float(beta),
            dropout=float(dropout),
        )

        output_rate_hz = self.input_sample_rate_hz / self.downsample_stride
        self.context_windows = tuple(
            max(1, int(round(seconds * output_rate_hz)))
            for seconds in (4.0, 8.0, 16.0, 32.0)
        )
        token_dim = 4 * int(hidden_channels)
        # base mean/std, four normalized classical centers, four valid bits,
        # and radar coverage.
        scalar_dim = 11
        head_input_dim = token_dim + scalar_dim
        head_hidden = max(24, int(hidden_channels))

        def make_head(outputs: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(head_input_dim, head_hidden),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(head_hidden, int(outputs)),
            )

        self.divisor_head = make_head(4)
        self.residual_head = make_head(4)
        self.uncertainty_head = make_head(4)
        self.gate_head = make_head(1)
        self.quality_head = make_head(1)

        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        initial_uncertainty = _inverse_softplus(
            float(initial_candidate_std) - self.minimum_candidate_std
        )
        nn.init.constant_(self.uncertainty_head[-1].bias, initial_uncertainty)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, float(initial_gate_bias))

    def _validate_inputs(
        self,
        component_signals: Tensor,
        attributes: Tensor,
        base_prediction: Tensor,
        base_std: Tensor,
        classical_rr: Tensor | None,
        radar_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        expected_prefix = (
            self.num_radars,
            self.num_variants,
            self.num_components,
        )
        if (
            component_signals.ndim != 5
            or tuple(component_signals.shape[1:4]) != expected_prefix
        ):
            raise ValueError(
                "component_signals must have shape "
                f"[batch, {self.num_radars}, {self.num_variants}, "
                f"{self.num_components}, time], got {tuple(component_signals.shape)}"
            )
        time_steps = component_signals.shape[-1]
        if time_steps < 16 or time_steps % self.downsample_stride:
            raise ValueError(
                "component_signals time length must be >=16 and divisible by "
                f"{self.downsample_stride}"
            )
        expected_attributes = (*component_signals.shape[:4], 5)
        if tuple(attributes.shape) != expected_attributes:
            raise ValueError(
                f"attributes must have shape {expected_attributes}, got "
                f"{tuple(attributes.shape)}"
            )
        batch = component_signals.shape[0]
        if base_prediction.shape not in {(batch,), (batch, 1)}:
            raise ValueError(f"base_prediction must have shape {(batch,)}")
        if base_std.shape not in {(batch,), (batch, 1)}:
            raise ValueError(f"base_std must have shape {(batch,)}")
        if radar_mask is None:
            radar_mask = torch.ones(
                (batch, self.num_radars),
                device=component_signals.device,
                dtype=torch.bool,
            )
        elif radar_mask.shape != (batch, self.num_radars):
            raise ValueError(
                f"radar_mask must have shape {(batch, self.num_radars)}, got "
                f"{tuple(radar_mask.shape)}"
            )
        if classical_rr is not None:
            if classical_rr.ndim == 1:
                if classical_rr.shape[0] != batch:
                    raise ValueError("classical_rr must have length batch")
            elif classical_rr.ndim != 2 or classical_rr.shape[0] != batch:
                raise ValueError("classical_rr must have shape [batch] or [batch, K]")

        if not torch.is_floating_point(component_signals):
            component_signals = component_signals.float()
        device, dtype = component_signals.device, component_signals.dtype
        component_signals = torch.nan_to_num(
            component_signals, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)
        attributes = torch.nan_to_num(
            attributes.to(device=device, dtype=dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        base_prediction = base_prediction.reshape(batch).to(device=device, dtype=dtype)
        base_std = base_std.reshape(batch).to(device=device, dtype=dtype)
        if classical_rr is not None:
            classical_rr = classical_rr.to(device=device, dtype=dtype)
            if classical_rr.ndim == 1:
                classical_rr = classical_rr.unsqueeze(1)
        available = radar_mask.to(device=device) > 0
        component_signals = component_signals * available[:, :, None, None, None].to(
            dtype=dtype
        )
        attributes = attributes * available[:, :, None, None, None].to(dtype=dtype)
        return (
            component_signals,
            attributes,
            base_prediction,
            base_std,
            classical_rr,
            available,
        )

    def _base_posterior(
        self, base_prediction: Tensor, base_std: Tensor
    ) -> tuple[Tensor, Tensor]:
        valid = (
            torch.isfinite(base_prediction)
            & torch.isfinite(base_std)
            & (base_std > 0)
        )
        midpoint = 0.5 * (self.rr_min + self.rr_max)
        mean = torch.where(valid, base_prediction, base_prediction.new_full((), midpoint))
        sigma = torch.where(valid, base_std, base_std.new_full((), 4.0)).clamp(
            min=self.rr_step, max=self.rr_max - self.rr_min
        )
        bins = self.rr_bins.to(device=mean.device, dtype=mean.dtype)
        logits = -0.5 * (
            (bins.unsqueeze(0) - mean.unsqueeze(1)) / sigma.unsqueeze(1)
        ).square()
        probability = (logits - logits.amax(dim=1, keepdim=True)).exp()
        probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(
            1.0e-8
        )
        return probability, valid

    def _classical_candidates(
        self,
        base_prediction: Tensor,
        classical_rr: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if classical_rr is None:
            classical_rr = base_prediction.unsqueeze(1)
        multipliers = self.candidate_multipliers.to(
            device=base_prediction.device, dtype=base_prediction.dtype
        )
        means = classical_rr.unsqueeze(-1) * multipliers.view(1, 1, 4)
        seed_valid = torch.isfinite(classical_rr) & (classical_rr > 0)
        valid = (
            seed_valid.unsqueeze(-1)
            & torch.isfinite(means)
            & (means >= self.rr_min)
            & (means <= self.rr_max)
        )
        midpoint = 0.5 * (self.rr_min + self.rr_max)
        safe_means = torch.where(valid, means, means.new_full((), midpoint))
        count = valid.sum(dim=1)
        aggregated = (
            torch.where(valid, safe_means, torch.zeros_like(safe_means)).sum(dim=1)
            / count.clamp_min(1).to(dtype=means.dtype)
        )
        multiplier_valid = count > 0
        aggregated = torch.where(
            multiplier_valid,
            aggregated,
            aggregated.new_full((), midpoint),
        )
        return safe_means, valid, aggregated, multiplier_valid

    def _multiscale_tokens(self, state_sequence: Tensor) -> Tensor:
        return torch.cat(
            tuple(
                _causal_window_mean(state_sequence, window)
                for window in self.context_windows
            ),
            dim=-1,
        )

    def _scalar_features(
        self,
        base_prediction: Tensor,
        base_std: Tensor,
        centers: Tensor,
        candidate_valid: Tensor,
        available: Tensor,
    ) -> Tensor:
        midpoint = 0.5 * (self.rr_min + self.rr_max)
        half_span = 0.5 * (self.rr_max - self.rr_min)
        base_mean = torch.nan_to_num(
            (base_prediction - midpoint) / half_span,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-2.0, 2.0)
        safe_std = torch.nan_to_num(
            base_std, nan=4.0, posinf=4.0, neginf=4.0
        ).abs().clamp_min(self.rr_step)
        base_log_std = (safe_std.log() / math.log(self.rr_max - self.rr_min)).clamp(
            -2.0, 2.0
        )
        normalized_centers = ((centers - midpoint) / half_span).clamp(-1.0, 1.0)
        normalized_centers = normalized_centers * candidate_valid.to(
            dtype=centers.dtype
        )
        coverage = available.to(dtype=centers.dtype).mean(dim=1, keepdim=True)
        return torch.cat(
            (
                base_mean.unsqueeze(1),
                base_log_std.unsqueeze(1),
                normalized_centers,
                candidate_valid.to(dtype=centers.dtype),
                coverage,
            ),
            dim=1,
        )

    def _expert_posteriors(
        self,
        raw_means: Tensor,
        seed_valid: Tensor,
        residual: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        corrected = raw_means + residual.unsqueeze(1)
        bins = self.rr_bins.to(device=corrected.device, dtype=corrected.dtype)
        logits = -0.5 * (
            (bins.view(1, 1, 1, -1) - corrected.unsqueeze(-1))
            / sigma[:, None, :, None]
        ).square()
        logits = logits - logits.amax(dim=-1, keepdim=True)
        probabilities = logits.exp() * seed_valid.unsqueeze(-1).to(logits.dtype)
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        count = seed_valid.sum(dim=1).clamp_min(1).to(dtype=logits.dtype)
        return probabilities.sum(dim=1) / count.unsqueeze(-1)

    def forward(
        self,
        component_signals: Tensor,
        attributes: Tensor,
        base_prediction: Tensor,
        base_std: Tensor,
        classical_rr: Tensor | None = None,
        radar_mask: Tensor | None = None,
        *,
        return_sequences: bool = False,
    ) -> dict[str, Tensor]:
        (
            component_signals,
            attributes,
            base_prediction,
            base_std,
            classical_rr,
            available,
        ) = self._validate_inputs(
            component_signals,
            attributes,
            base_prediction,
            base_std,
            classical_rr,
            radar_mask,
        )
        fused, radar_weights, component_attention = self.compressor(
            component_signals, attributes, available
        )
        downsampled = self.downsample(fused)
        state_sequence, layer_rates_per_sample = self.spiking_encoder(downsampled)
        token_sequence = self._multiscale_tokens(state_sequence)
        final_token = token_sequence[:, -1]

        base_probability, base_valid = self._base_posterior(
            base_prediction, base_std
        )
        raw_means, seed_valid, centers, candidate_valid = self._classical_candidates(
            base_prediction, classical_rr
        )
        scalar = self._scalar_features(
            base_prediction,
            base_std,
            centers,
            candidate_valid,
            available,
        )
        head_feature = torch.cat((final_token, scalar), dim=1)

        divisor_logits_raw = self.divisor_head(head_feature)
        divisor_probabilities = _masked_softmax(
            divisor_logits_raw, candidate_valid, dim=1
        )
        residual_per_divisor = self.max_residual_bpm * torch.tanh(
            self.residual_head(head_feature)
        )
        candidate_std = self.minimum_candidate_std + F.softplus(
            self.uncertainty_head(head_feature)
        )
        candidate_std = candidate_std.clamp(max=self.maximum_candidate_std)
        expert_probabilities = self._expert_posteriors(
            raw_means,
            seed_valid,
            residual_per_divisor,
            candidate_std,
        )
        source_probability = (
            expert_probabilities * divisor_probabilities.unsqueeze(-1)
        ).sum(dim=1)
        any_candidate = candidate_valid.any(dim=1)
        source_probability = torch.where(
            any_candidate.unsqueeze(1), source_probability, base_probability
        )
        source_probability = source_probability / source_probability.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)

        raw_gate_logits = self.gate_head(head_feature).squeeze(1)
        usable_source = available.any(dim=1) & any_candidate
        mixture_gate = raw_gate_logits.sigmoid() * usable_source.to(
            dtype=head_feature.dtype
        )
        probabilities = (
            (1.0 - mixture_gate.unsqueeze(1)) * base_probability
            + mixture_gate.unsqueeze(1) * source_probability
        )
        probabilities = probabilities / probabilities.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)

        bins = self.rr_bins.to(device=probabilities.device, dtype=probabilities.dtype)
        logits = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        expected_rr, rr_std = _posterior_statistics(probabilities, bins)
        base_expected_rr, base_posterior_std = _posterior_statistics(
            base_probability, bins
        )
        source_expected_rr, source_std = _posterior_statistics(
            source_probability, bins
        )
        map_rr = bins[probabilities.argmax(dim=1)]
        posterior_entropy = -(
            probabilities
            * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        ).sum(dim=1)
        quality_logits = torch.where(
            available.any(dim=1),
            self.quality_head(head_feature).squeeze(1),
            head_feature.new_full((), -20.0),
        )
        quality = quality_logits.sigmoid() * available.any(dim=1).to(
            dtype=head_feature.dtype
        )
        residual_rr = (residual_per_divisor * divisor_probabilities).sum(dim=1)
        corrected_centers = centers + residual_per_divisor
        divisor_index = divisor_probabilities.argmax(dim=1)
        divisor_map = self.candidate_multipliers.to(
            device=divisor_index.device, dtype=probabilities.dtype
        )[divisor_index]
        divisor_map = divisor_map * any_candidate.to(dtype=probabilities.dtype)
        layer_rates = layer_rates_per_sample.mean(dim=0)
        spike_rate_per_sample = layer_rates_per_sample.mean(dim=1)

        output = {
            "logits": logits,
            "rr_logits": logits,
            "probabilities": probabilities,
            "rr_probs": probabilities,
            "expected_rr": expected_rr,
            "rr": expected_rr,
            "map_rr": map_rr,
            "rr_std": rr_std,
            "log_variance": rr_std.square().clamp_min(1.0e-8).log(),
            "posterior_entropy": posterior_entropy,
            "base_probabilities": base_probability,
            "base_expected_rr": base_expected_rr,
            "base_posterior_std": base_posterior_std,
            "base_valid": base_valid,
            "source_probabilities": source_probability,
            "source_expected_rr": source_expected_rr,
            "source_std": source_std,
            "mixture_gate_logits": raw_gate_logits,
            "mixture_gate": mixture_gate,
            "divisor_logits": divisor_logits_raw,
            "divisor_probabilities": divisor_probabilities,
            "divisor_map": divisor_map,
            "candidate_valid_mask": candidate_valid,
            "candidate_centers_rr": corrected_centers,
            "candidate_raw_centers_rr": centers,
            "candidate_std_rr": candidate_std,
            "expert_probabilities": expert_probabilities,
            "residual_rr": residual_rr,
            "residual_rr_per_divisor": residual_per_divisor,
            "quality_logits": quality_logits,
            "quality": quality,
            "radar_weights": radar_weights,
            "radar_mask": available,
            "component_attention": component_attention,
            "temporal_embedding": final_token,
            "downsampled_steps": probabilities.new_tensor(
                downsampled.shape[-1], dtype=torch.long
            ),
            "spike_rate": spike_rate_per_sample.mean(),
            "spike_rate_per_sample": spike_rate_per_sample,
            "layer_spike_rates": layer_rates,
            "layer_spike_rates_per_sample": layer_rates_per_sample,
        }
        if return_sequences:
            output.update(
                {
                    "downsampled_sequence": downsampled.transpose(1, 2),
                    "temporal_state_sequence": state_sequence,
                    "multiscale_token_sequence": token_sequence,
                }
            )
        return output


# Experiment-config aliases.
ChronologicalSVDRRSNN = TemporalSourceSeparatedRRSNN
SVDTemporalRRSNN = TemporalSourceSeparatedRRSNN


__all__ = [
    "ChronologicalSVDRRSNN",
    "SVDTemporalRRSNN",
    "TEMPORAL_NUM_RR_BINS",
    "TEMPORAL_RR_MAX",
    "TEMPORAL_RR_MIN",
    "TEMPORAL_RR_STEP",
    "TemporalSourceSeparatedRRSNN",
]
