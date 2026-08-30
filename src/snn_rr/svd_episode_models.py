"""Causal episode-level SNN for respiratory-rate alias resolution.

The model consumes one compact, label-free evidence vector per radar and
candidate at every 4-second window.  Its recurrent state advances exactly once
per chronological window, including windows without a usable reference.  No
reference validity or reference-quality value is accepted by :meth:`forward`.

The output distribution is a conservative mixture of a frozen base Gaussian
and four source hypotheses (classical RR x1..x4).  An untrained model is an
almost exact base fallback, while rows without a base prediction use the source
posterior.  Padding does not update the spiking state, and an episode with no
available radar falls back exactly to the base prediction when one exists.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from snntorch import surrogate
except ImportError as exc:  # pragma: no cover - broken installation only
    raise ImportError(
        "EpisodeAliasRRSNN requires snntorch; install the project runtime dependencies"
    ) from exc

from .models import make_rr_bins


EPISODE_RR_MIN = 6.0
EPISODE_RR_MAX = 45.0
EPISODE_RR_STEP = 0.25


def _inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _masked_softmax(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    masked = values.masked_fill(~mask, -1.0e4)
    weights = masked.softmax(dim=dim) * mask.to(values.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0e-8)


def _posterior_stats(probability: Tensor, bins: Tensor) -> tuple[Tensor, Tensor]:
    mean = (probability * bins).sum(dim=-1)
    variance = (probability * (bins - mean.unsqueeze(-1)).square()).sum(dim=-1)
    return mean, variance.clamp_min(torch.finfo(probability.dtype).eps).sqrt()


class EpisodeSpikingCell(nn.Module):
    """One recurrent LIF, PLIF, or adaptive-LIF state transition."""

    VALID_TYPES = frozenset(("lif", "plif", "alif"))

    def __init__(
        self,
        channels: int,
        *,
        cell_type: str,
        beta: float = 0.92,
        adaptation_decay: float = 0.97,
        adaptation_strength: float = 0.4,
    ) -> None:
        super().__init__()
        cell_type = str(cell_type).lower()
        if cell_type not in self.VALID_TYPES:
            raise ValueError(f"cell_type must be one of {sorted(self.VALID_TYPES)}")
        if channels < 1 or not 0.0 < beta < 1.0:
            raise ValueError("channels must be positive and beta must be in (0, 1)")
        self.channels = int(channels)
        self.cell_type = cell_type
        beta_value = torch.tensor(_inverse_sigmoid(float(beta)), dtype=torch.float32)
        if cell_type == "lif":
            self.register_buffer("beta_logit", beta_value, persistent=True)
        else:
            self.beta_logit = nn.Parameter(beta_value)
        self.threshold_unconstrained = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        if cell_type == "alif":
            self.adaptation_decay_logit = nn.Parameter(
                torch.tensor(_inverse_sigmoid(float(adaptation_decay)))
            )
            self.adaptation_strength_unconstrained = nn.Parameter(
                torch.tensor(_inverse_softplus(float(adaptation_strength)))
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
        update_mask: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        membrane, adaptation = state
        beta = self.beta_logit.sigmoid().to(current.dtype)
        threshold = F.softplus(self.threshold_unconstrained).to(current.dtype)
        if self.cell_type == "alif":
            threshold = threshold + F.softplus(
                self.adaptation_strength_unconstrained
            ).to(current.dtype) * adaptation
        next_membrane = beta * membrane + current
        spikes = self.spike_function(next_membrane - threshold)
        next_membrane = next_membrane - spikes.detach() * threshold
        if self.cell_type == "alif":
            decay = self.adaptation_decay_logit.sigmoid().to(current.dtype)
            next_adaptation = decay * adaptation + spikes
        else:
            next_adaptation = torch.zeros_like(adaptation)
        active = update_mask.to(current.dtype).unsqueeze(-1)
        membrane = active * next_membrane + (1.0 - active) * membrane
        adaptation = active * next_adaptation + (1.0 - active) * adaptation
        spikes = spikes * active
        return spikes, (membrane, adaptation)


class ChronologicalEpisodeEncoder(nn.Module):
    """Stacked spiking recurrence whose time axis is the window sequence."""

    def __init__(
        self,
        channels: int,
        *,
        cell_types: Sequence[str] = ("lif", "plif", "alif"),
        beta: float = 0.92,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        normalized = tuple(str(value).lower() for value in cell_types)
        if not normalized:
            raise ValueError("cell_types cannot be empty")
        self.channels = int(channels)
        self.cells = nn.ModuleList(
            EpisodeSpikingCell(channels, cell_type=value, beta=beta)
            for value in normalized
        )
        self.synapses = nn.ModuleList(
            nn.Linear(channels, channels) for _ in normalized
        )
        self.norms = nn.ModuleList(nn.LayerNorm(channels) for _ in normalized)
        self.dropout = nn.Dropout(float(dropout))
        self.readout = nn.Sequential(
            nn.Linear(3 * channels, channels), nn.SiLU(), nn.LayerNorm(channels)
        )

    def forward(self, values: Tensor, sequence_mask: Tensor) -> tuple[Tensor, Tensor]:
        if values.ndim != 3 or values.shape[-1] != self.channels:
            raise ValueError("values must have shape [batch, windows, channels]")
        if sequence_mask.shape != values.shape[:2]:
            raise ValueError("sequence_mask must have shape [batch, windows]")
        batch, windows, _ = values.shape
        states = [
            cell.initial_state(values.new_zeros((batch, self.channels)))
            for cell in self.cells
        ]
        outputs: list[Tensor] = []
        spike_sum = values.new_zeros((batch, len(self.cells)))
        step_count = sequence_mask.to(values.dtype).sum(dim=1, keepdim=True).clamp_min(1)
        for step in range(windows):
            analog = values[:, step]
            current = analog
            last_spikes = torch.zeros_like(analog)
            last_membrane = torch.zeros_like(analog)
            for layer, (synapse, norm, cell) in enumerate(
                zip(self.synapses, self.norms, self.cells, strict=True)
            ):
                current = norm(synapse(current))
                spikes, states[layer] = cell.forward_step(
                    current, states[layer], sequence_mask[:, step]
                )
                spike_sum[:, layer] += spikes.mean(dim=-1)
                last_spikes = spikes
                last_membrane = states[layer][0]
                current = self.dropout(spikes)
            token = self.readout(
                torch.cat((analog, last_spikes, torch.tanh(last_membrane)), dim=-1)
            )
            outputs.append(token * sequence_mask[:, step, None].to(token.dtype))
        return torch.stack(outputs, dim=1), spike_sum / step_count


class EpisodeAliasRRSNN(nn.Module):
    """Causal SNN router over complete within-session window episodes.

    Parameters supplied to ``forward`` are all deployment-time quantities.
    ``base_available`` is used only to choose the deterministic fallback and is
    deliberately excluded from the learned encoder and heads.  Thus a missing
    frozen prediction cannot become a proxy input for reference validity.
    """

    MULTIPLIERS = (1.0, 2.0, 3.0, 4.0)

    def __init__(
        self,
        *,
        evidence_features: int = 8,
        context_features: int = 3,
        num_radars: int = 3,
        candidate_channels: int = 16,
        hidden_channels: int = 48,
        cell_types: Sequence[str] = ("lif", "plif", "alif"),
        beta: float = 0.92,
        dropout: float = 0.05,
        rr_min: float = EPISODE_RR_MIN,
        rr_max: float = EPISODE_RR_MAX,
        rr_step: float = EPISODE_RR_STEP,
        max_residual_bpm: float = 1.5,
        minimum_candidate_std: float = 0.35,
        maximum_candidate_std: float = 8.0,
        initial_candidate_std: float = 1.2,
        initial_gate_bias: float = -8.0,
        use_base_features: bool = False,
        strict_alias_gate: bool = False,
    ) -> None:
        super().__init__()
        if evidence_features < 1 or context_features < 1 or num_radars < 1:
            raise ValueError("feature and radar dimensions must be positive")
        if candidate_channels < 4 or hidden_channels < 8:
            raise ValueError("candidate_channels >= 4 and hidden_channels >= 8 required")
        steps = (float(rr_max) - float(rr_min)) / float(rr_step)
        if not math.isfinite(steps) or steps <= 0 or not math.isclose(
            steps, round(steps), abs_tol=1.0e-7
        ):
            raise ValueError("rr_step must exactly partition rr_min..rr_max")
        if not 0 < minimum_candidate_std < initial_candidate_std <= maximum_candidate_std:
            raise ValueError("candidate uncertainty limits are inconsistent")
        self.evidence_features = int(evidence_features)
        self.context_features = int(context_features)
        self.num_radars = int(num_radars)
        self.candidate_channels = int(candidate_channels)
        self.hidden_channels = int(hidden_channels)
        self.rr_min = float(rr_min)
        self.rr_max = float(rr_max)
        self.rr_step = float(rr_step)
        self.max_residual_bpm = float(max_residual_bpm)
        self.minimum_candidate_std = float(minimum_candidate_std)
        self.maximum_candidate_std = float(maximum_candidate_std)
        self.use_base_features = bool(use_base_features)
        self.strict_alias_gate = bool(strict_alias_gate)
        if self.use_base_features and self.strict_alias_gate:
            raise ValueError(
                "strict_alias_gate and use_base_features are mutually exclusive"
            )
        self.register_buffer(
            "rr_bins", make_rr_bins(rr_min, rr_max, int(round(steps)) + 1)
        )
        self.register_buffer(
            "candidate_multipliers",
            torch.tensor(self.MULTIPLIERS, dtype=torch.float32),
            persistent=False,
        )

        self.evidence_encoder = nn.Sequential(
            nn.Linear(self.evidence_features, candidate_channels),
            nn.SiLU(),
            nn.LayerNorm(candidate_channels),
            nn.Linear(candidate_channels, candidate_channels),
            nn.SiLU(),
        )
        self.radar_attention = nn.Sequential(
            nn.Linear(self.evidence_features, max(8, candidate_channels // 2)),
            nn.SiLU(),
            nn.Linear(max(8, candidate_channels // 2), 1),
        )
        # Four fused candidate tokens, deployment context, and masked frozen
        # base mean/std/alias.  No base-availability bit is encoded.
        encoder_input = 4 * candidate_channels + context_features + 3
        self.input_projection = nn.Sequential(
            nn.Linear(encoder_input, hidden_channels),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
        )
        self.spiking_encoder = ChronologicalEpisodeEncoder(
            hidden_channels, cell_types=cell_types, beta=beta, dropout=dropout
        )
        head_input = hidden_channels + 4 * candidate_channels + context_features + 3
        head_hidden = max(24, hidden_channels)

        def make_head(outputs: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(head_input, head_hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, outputs),
            )

        self.divisor_head = make_head(4)
        self.residual_head = make_head(4)
        self.uncertainty_head = make_head(4)
        self.gate_head = make_head(1)
        self.quality_head = make_head(1)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.constant_(
            self.uncertainty_head[-1].bias,
            _inverse_softplus(initial_candidate_std - minimum_candidate_std),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, float(initial_gate_bias))

    def _validate(
        self,
        evidence: Tensor,
        context: Tensor,
        classical_rr: Tensor,
        base_prediction: Tensor,
        base_std: Tensor,
        base_alias_probability: Tensor,
        base_available: Tensor,
        radar_mask: Tensor,
        sequence_mask: Tensor,
    ) -> tuple[Tensor, ...]:
        if evidence.ndim != 5 or tuple(evidence.shape[2:4]) != (self.num_radars, 4):
            raise ValueError(
                "evidence must have shape [batch, windows, radars, 4, features]"
            )
        if evidence.shape[-1] != self.evidence_features:
            raise ValueError("evidence feature dimension does not match the model")
        prefix = evidence.shape[:2]
        expected_vectors = (
            context,
            classical_rr,
            base_prediction,
            base_std,
            base_alias_probability,
            base_available,
            sequence_mask,
        )
        if context.shape != (*prefix, self.context_features):
            raise ValueError("context must have shape [batch, windows, context_features]")
        if any(value.shape != prefix for value in expected_vectors[1:]):
            raise ValueError("scalar episode inputs must have shape [batch, windows]")
        if radar_mask.shape != (*prefix, self.num_radars):
            raise ValueError("radar_mask must have shape [batch, windows, radars]")
        dtype, device = evidence.dtype, evidence.device
        if not torch.is_floating_point(evidence):
            evidence = evidence.float()
            dtype = evidence.dtype
        floats = (context, classical_rr, base_prediction, base_std, base_alias_probability)
        context, classical_rr, base_prediction, base_std, base_alias_probability = (
            torch.nan_to_num(value.to(device=device, dtype=dtype), nan=0.0)
            for value in floats
        )
        evidence = torch.nan_to_num(evidence, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
            -20.0, 20.0
        )
        sequence_mask = sequence_mask.to(device=device).bool()
        radar_mask = radar_mask.to(device=device).bool() & sequence_mask.unsqueeze(-1)
        base_available = base_available.to(device=device).bool() & sequence_mask
        return (
            evidence,
            context,
            classical_rr,
            base_prediction,
            base_std,
            base_alias_probability,
            base_available,
            radar_mask,
            sequence_mask,
        )

    def _gaussian(self, means: Tensor, std: Tensor) -> Tensor:
        bins = self.rr_bins.to(device=means.device, dtype=means.dtype)
        logits = -0.5 * (
            (bins.view(1, 1, 1, -1) - means.unsqueeze(-1))
            / std.unsqueeze(-1).clamp_min(self.rr_step)
        ).square()
        probability = (logits - logits.amax(dim=-1, keepdim=True)).exp()
        return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def forward(
        self,
        evidence: Tensor,
        context: Tensor,
        classical_rr: Tensor,
        base_prediction: Tensor,
        base_std: Tensor,
        base_alias_probability: Tensor,
        base_available: Tensor,
        radar_mask: Tensor,
        sequence_mask: Tensor,
    ) -> dict[str, Tensor]:
        (
            evidence,
            context,
            classical_rr,
            base_prediction,
            base_std,
            base_alias_probability,
            base_available,
            radar_mask,
            sequence_mask,
        ) = self._validate(
            evidence,
            context,
            classical_rr,
            base_prediction,
            base_std,
            base_alias_probability,
            base_available,
            radar_mask,
            sequence_mask,
        )
        batch, windows = evidence.shape[:2]
        encoded = self.evidence_encoder(evidence)
        attention_logits = self.radar_attention(evidence).squeeze(-1)
        radar_weights = _masked_softmax(
            attention_logits, radar_mask.unsqueeze(-1).expand(-1, -1, -1, 4), dim=2
        )
        fused = (encoded * radar_weights.unsqueeze(-1)).sum(dim=2)
        fused_flat = fused.reshape(batch, windows, -1)

        midpoint = 0.5 * (self.rr_min + self.rr_max)
        half_span = 0.5 * (self.rr_max - self.rr_min)
        base_mask_float = base_available.to(evidence.dtype)
        safe_base_features = torch.stack(
            (
                ((base_prediction - midpoint) / half_span).clamp(-2, 2),
                (base_std / 8.0).clamp(0, 2),
                base_alias_probability.clamp(0, 1),
            ),
            dim=-1,
        ) * base_mask_float.unsqueeze(-1)
        if not self.use_base_features:
            # Primary protocol: the frozen OOF stack is not a learned input.
            # Its base learner for a training identity may itself have seen the
            # current outer-test identity.  Keeping these channels identically
            # zero prevents that second-order stacking leak; base is retained
            # only in the deterministic output mixture/fallback.
            safe_base_features = torch.zeros_like(safe_base_features)
        direct = torch.cat((fused_flat, context, safe_base_features), dim=-1)
        analog = self.input_projection(direct)
        state, spike_rate = self.spiking_encoder(analog, sequence_mask)
        head_input = torch.cat((state, direct), dim=-1)

        candidate_valid = (
            torch.isfinite(classical_rr).unsqueeze(-1)
            & (classical_rr.unsqueeze(-1) * self.candidate_multipliers >= self.rr_min)
            & (classical_rr.unsqueeze(-1) * self.candidate_multipliers <= self.rr_max)
            & sequence_mask.unsqueeze(-1)
        )
        centers = classical_rr.unsqueeze(-1) * self.candidate_multipliers.to(
            evidence.dtype
        )
        safe_centers = torch.where(
            candidate_valid, centers, torch.full_like(centers, midpoint)
        )
        divisor_logits = self.divisor_head(head_input).masked_fill(
            ~candidate_valid, -1.0e4
        )
        divisor_probability = _masked_softmax(divisor_logits, candidate_valid, dim=-1)
        no_candidate = ~candidate_valid.any(dim=-1)
        if no_candidate.any():
            fallback = F.one_hot(
                torch.zeros_like(classical_rr, dtype=torch.long), 4
            ).to(evidence.dtype)
            divisor_probability = torch.where(
                no_candidate.unsqueeze(-1), fallback, divisor_probability
            )

        residual = self.max_residual_bpm * torch.tanh(self.residual_head(head_input))
        candidate_mean = (safe_centers + residual).clamp(self.rr_min, self.rr_max)
        candidate_std = (
            self.minimum_candidate_std + F.softplus(self.uncertainty_head(head_input))
        ).clamp(max=self.maximum_candidate_std)
        candidate_posterior = self._gaussian(candidate_mean, candidate_std)
        source_posterior = (
            divisor_probability.unsqueeze(-1) * candidate_posterior
        ).sum(dim=-2)
        bins = self.rr_bins.to(device=evidence.device, dtype=evidence.dtype).view(1, 1, -1)
        source_prediction, source_std = _posterior_stats(source_posterior, bins)

        safe_base_mean = torch.where(
            base_available, base_prediction, torch.full_like(base_prediction, midpoint)
        ).clamp(self.rr_min, self.rr_max)
        safe_base_std = torch.where(
            base_available, base_std, torch.full_like(base_std, 4.0)
        ).clamp(self.rr_step, self.rr_max - self.rr_min)
        base_posterior = self._gaussian(
            safe_base_mean.unsqueeze(-1), safe_base_std.unsqueeze(-1)
        ).squeeze(-2)
        gate_logits = self.gate_head(head_input).squeeze(-1)
        learned_gate = gate_logits.sigmoid()
        has_radar = radar_mask.any(dim=-1)
        # No base => source is the only finite deployable fallback, even when
        # every radar is absent. No radar + a real base => exact frozen
        # fallback. Padding always has a zero gate.
        gate = torch.where(base_available, learned_gate, torch.ones_like(learned_gate))
        gate = torch.where(
            ~has_radar & base_available, torch.zeros_like(gate), gate
        )
        gate = gate * sequence_mask.to(gate.dtype)
        posterior = (1.0 - gate.unsqueeze(-1)) * base_posterior + gate.unsqueeze(
            -1
        ) * source_posterior
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        prediction, rr_std = _posterior_stats(posterior, bins)
        prediction = torch.where(
            base_available & ~has_radar, base_prediction, prediction
        )
        rr_std = torch.where(base_available & ~has_radar, base_std, rr_std)
        expert_probability = torch.cat(
            (
                ((1.0 - gate) * base_available.to(gate.dtype)).unsqueeze(-1),
                gate.unsqueeze(-1) * divisor_probability,
            ),
            dim=-1,
        )
        expert_probability = expert_probability / expert_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        quality_logit = self.quality_head(head_input).squeeze(-1)
        return {
            "probabilities": posterior,
            "expected_rr": prediction,
            "map_rr": self.rr_bins[posterior.argmax(dim=-1)],
            "rr_std": rr_std,
            "source_probabilities": source_posterior,
            "source_prediction": source_prediction,
            "source_std": source_std,
            "base_probabilities": base_posterior,
            "divisor_logits": divisor_logits,
            "divisor_probabilities": divisor_probability,
            "candidate_mean": candidate_mean,
            "candidate_std": candidate_std,
            "residual_rr": residual,
            "mixture_gate": gate,
            "gate_logits": gate_logits,
            "learned_gate": learned_gate,
            "expert_probabilities": expert_probability,
            "quality_logit": quality_logit,
            "quality": quality_logit.sigmoid(),
            "radar_weights": radar_weights,
            "spike_rate_by_layer": spike_rate,
            "spike_rate": spike_rate.mean(dim=-1),
            "state_sequence": state,
            "candidate_valid": candidate_valid,
            "sequence_mask": sequence_mask,
        }
