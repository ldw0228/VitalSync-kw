"""Causal harmonic candidate-set episode SNN.

The model in this module is deliberately a *source* model.  It ranks a small
set of label-free respiratory-rate proposals and carries spiking state across
successive windows.  Iterations one and two use only the candidate evidence.
The optional iteration-three architecture may additionally receive a strict
nested, label-free posterior anchor and its uncertainty.  Reference values,
reference quality, and reference validity are intentionally absent from the
forward interface; a deployment policy may use the source output only after
this model has returned.

Inputs are compact candidate-node features produced by the range/SVD evidence
builder.  Harmonic message passing is local to a window, while the explicit
PLIF -> ALIF state is causal across windows.  Padding and unavailable radar
observations are hard-masked before any pooling operation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .svd_episode_models import EpisodeSpikingCell


NeuronState: TypeAlias = tuple[Tensor, Tensor]
HarmonicSetState: TypeAlias = tuple[NeuronState, NeuronState]


def _masked_softmax(values: Tensor, mask: Tensor, *, dim: int) -> Tensor:
    """Softmax with exact zero probability when every entry is masked."""

    masked_values = values.masked_fill(~mask, -1.0e4)
    weights = masked_values.softmax(dim=dim) * mask.to(values.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0e-8)


def build_harmonic_edge_types(
    candidate_rr: Tensor,
    candidate_mask: Tensor,
    *,
    proximity_bpm: float = 1.0,
    harmonic_relative_tolerance: float = 0.05,
    harmonic_factors: Sequence[int] = (2, 3, 4),
) -> Tensor:
    """Build symmetric near/x2/x3/x4 adjacency channels.

    Parameters
    ----------
    candidate_rr:
        Candidate rates with shape ``[..., K]``.
    candidate_mask:
        Boolean validity mask with the same shape.

    Returns
    -------
    Tensor
        Boolean tensor of shape ``[..., K, K, 4]``.  The final channels are
        proximity, x2, x3, and x4 in that order.  Self edges and any edge that
        touches a padded/non-finite/non-positive node are always false.

    Notes
    -----
    Harmonic edges are symmetric for message passing: a pair is connected
    when ``max(rr_i, rr_j) / min(rr_i, rr_j)`` lies within the relative
    tolerance of the requested integer factor.
    """

    if candidate_rr.ndim < 1 or candidate_rr.shape != candidate_mask.shape:
        raise ValueError("candidate_rr and candidate_mask must share shape [..., K]")
    if candidate_rr.shape[-1] < 1:
        raise ValueError("the candidate dimension cannot be empty")
    factors = tuple(int(value) for value in harmonic_factors)
    if factors != (2, 3, 4):
        raise ValueError("harmonic_factors must be exactly (2, 3, 4)")
    if proximity_bpm <= 0 or harmonic_relative_tolerance <= 0:
        raise ValueError("edge tolerances must be positive")

    rr = candidate_rr if torch.is_floating_point(candidate_rr) else candidate_rr.float()
    valid = candidate_mask.to(torch.bool) & torch.isfinite(rr) & (rr > 0)
    left = rr.unsqueeze(-1)
    right = rr.unsqueeze(-2)
    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    identity = torch.eye(
        rr.shape[-1], device=rr.device, dtype=torch.bool
    ).reshape((1,) * (rr.ndim - 1) + (rr.shape[-1], rr.shape[-1]))
    pair_valid = pair_valid & ~identity

    delta = (left - right).abs()
    near = pair_valid & (delta <= float(proximity_bpm))
    smaller = torch.minimum(left, right).clamp_min(torch.finfo(rr.dtype).tiny)
    ratio = torch.maximum(left, right) / smaller
    relations = [near]
    for factor in factors:
        tolerance = float(harmonic_relative_tolerance) * float(factor)
        relations.append(pair_valid & ((ratio - float(factor)).abs() <= tolerance))
    return torch.stack(relations, dim=-1)


class _HarmonicGraphPLIFBlock(nn.Module):
    """One relation-aware residual message-passing PLIF transition."""

    def __init__(self, channels: int, *, dropout: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.relation_projections = nn.ModuleList(
            nn.Linear(self.channels, self.channels, bias=False) for _ in range(4)
        )
        self.current_projection = nn.Linear(5 * self.channels, self.channels)
        self.current_norm = nn.LayerNorm(self.channels)
        self.readout = nn.Linear(2 * self.channels, self.channels)
        self.output_norm = nn.LayerNorm(self.channels)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        nodes: Tensor,
        edge_types: Tensor,
        node_mask: Tensor,
        *,
        cell: EpisodeSpikingCell,
        state: NeuronState,
    ) -> tuple[Tensor, Tensor, NeuronState]:
        if nodes.ndim != 3 or nodes.shape[-1] != self.channels:
            raise ValueError("graph nodes must have shape [batch, K, channels]")
        if node_mask.shape != nodes.shape[:2]:
            raise ValueError("node_mask must have shape [batch, K]")
        expected_edges = (*node_mask.shape, node_mask.shape[-1], 4)
        if edge_types.shape != expected_edges:
            raise ValueError(
                f"edge_types must have shape {expected_edges}, got {tuple(edge_types.shape)}"
            )

        messages: list[Tensor] = []
        for relation_index, projection in enumerate(self.relation_projections):
            adjacency = edge_types[..., relation_index].to(nodes.dtype)
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            aggregate = torch.matmul(adjacency, nodes) / degree
            messages.append(projection(aggregate))
        current = self.current_norm(
            self.current_projection(torch.cat((nodes, *messages), dim=-1))
        )
        current = current * node_mask[..., None].to(current.dtype)
        spikes, next_state = cell.forward_step(current, state, node_mask)
        membrane = next_state[0]
        update = self.readout(torch.cat((spikes, torch.tanh(membrane)), dim=-1))
        output = self.output_norm(nodes + self.dropout(update))
        output = output * node_mask[..., None].to(output.dtype)
        return output, spikes, next_state


class _MaskedMultiheadCandidatePool(nn.Module):
    """Small mask-aware attention pool with one query per feature head."""

    def __init__(self, channels: int, *, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_channels = self.channels // self.heads
        self.queries = nn.Parameter(torch.empty(self.heads, self.head_channels))
        nn.init.normal_(self.queries, mean=0.0, std=self.head_channels**-0.5)

    def forward(self, nodes: Tensor, node_mask: Tensor) -> tuple[Tensor, Tensor]:
        if nodes.ndim != 4 or nodes.shape[-1] != self.channels:
            raise ValueError("nodes must have shape [batch, time, K, channels]")
        if node_mask.shape != nodes.shape[:3]:
            raise ValueError("node_mask must have shape [batch, time, K]")
        split = nodes.reshape(*nodes.shape[:-1], self.heads, self.head_channels)
        scores = (split * self.queries).sum(dim=-1) / math.sqrt(self.head_channels)
        # [B,T,K,heads] -> normalize over candidates.
        weights = _masked_softmax(scores, node_mask[..., None], dim=2)
        pooled = (split * weights[..., None]).sum(dim=2).flatten(start_dim=-2)
        return pooled, weights


class _CausalPLIFALIFEncoder(nn.Module):
    """State-explicit chronological PLIF -> ALIF encoder."""

    CELL_TYPES = ("plif", "alif")

    def __init__(
        self,
        channels: int,
        *,
        beta: float,
        dropout: float,
        adaptation_decay: float,
        adaptation_strength: float,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.synapses = nn.ModuleList(
            nn.Linear(self.channels, self.channels) for _ in self.CELL_TYPES
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(self.channels) for _ in self.CELL_TYPES
        )
        self.cells = nn.ModuleList(
            (
                EpisodeSpikingCell(
                    self.channels, cell_type="plif", beta=float(beta)
                ),
                EpisodeSpikingCell(
                    self.channels,
                    cell_type="alif",
                    beta=float(beta),
                    adaptation_decay=float(adaptation_decay),
                    adaptation_strength=float(adaptation_strength),
                ),
            )
        )
        self.readout = nn.Sequential(
            nn.Linear(3 * self.channels, self.channels),
            nn.SiLU(),
            nn.LayerNorm(self.channels),
        )
        self.dropout = nn.Dropout(float(dropout))

    def initial_state(self, reference: Tensor) -> HarmonicSetState:
        if reference.ndim != 2 or reference.shape[-1] != self.channels:
            raise ValueError("state reference must have shape [batch, channels]")
        return tuple(  # type: ignore[return-value]
            cell.initial_state(torch.zeros_like(reference)) for cell in self.cells
        )

    def _validate_state(
        self, state: HarmonicSetState, reference: Tensor
    ) -> HarmonicSetState:
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise ValueError("state must contain PLIF and ALIF neuron states")
        normalized: list[NeuronState] = []
        for layer_state in state:
            if not isinstance(layer_state, (tuple, list)) or len(layer_state) != 2:
                raise ValueError("each neuron state must be (membrane, adaptation)")
            membrane, adaptation = layer_state
            if membrane.shape != reference.shape or adaptation.shape != reference.shape:
                raise ValueError(
                    "every state tensor must have shape [batch, hidden_channels]"
                )
            normalized.append(
                (
                    membrane.to(device=reference.device, dtype=reference.dtype),
                    adaptation.to(device=reference.device, dtype=reference.dtype),
                )
            )
        return tuple(normalized)  # type: ignore[return-value]

    def forward(
        self,
        values: Tensor,
        sequence_mask: Tensor,
        *,
        observation_mask: Tensor,
        state: HarmonicSetState | None,
        reset_mask: Tensor,
    ) -> tuple[Tensor, Tensor, HarmonicSetState]:
        if values.ndim != 3 or values.shape[-1] != self.channels:
            raise ValueError("encoder values must have shape [batch, time, channels]")
        if sequence_mask.shape != values.shape[:2]:
            raise ValueError("sequence_mask must have shape [batch, time]")
        if observation_mask.shape != values.shape[:2]:
            raise ValueError("observation_mask must have shape [batch, time]")
        if reset_mask.shape != values.shape[:2]:
            raise ValueError("reset_mask must have shape [batch, time]")

        batch, time_steps, _ = values.shape
        reference = values.new_zeros((batch, self.channels))
        states = list(
            self.initial_state(reference)
            if state is None
            else self._validate_state(state, reference)
        )
        outputs: list[Tensor] = []
        spike_outputs: list[Tensor] = []
        for time_index in range(time_steps):
            reset = reset_mask[:, time_index]
            keep = (~reset).to(values.dtype).unsqueeze(-1)
            states = [
                (membrane * keep, adaptation * keep)
                for membrane, adaptation in states
            ]

            analog = values[:, time_index]
            observed = observation_mask[:, time_index].to(values.dtype).unsqueeze(-1)
            current = analog * observed
            layer_spikes: list[Tensor] = []
            last_spikes = torch.zeros_like(analog)
            last_membrane = torch.zeros_like(analog)
            for layer, (synapse, norm, cell) in enumerate(
                zip(self.synapses, self.norms, self.cells, strict=True)
            ):
                current = norm(synapse(current)) * observed
                spikes, states[layer] = cell.forward_step(
                    current, states[layer], sequence_mask[:, time_index]
                )
                layer_spikes.append(spikes.mean(dim=-1))
                last_spikes = spikes
                last_membrane = states[layer][0]
                current = self.dropout(spikes)
            token = self.readout(
                torch.cat((analog, last_spikes, torch.tanh(last_membrane)), dim=-1)
            )
            token = token * sequence_mask[:, time_index, None].to(token.dtype)
            outputs.append(token)
            spike_outputs.append(torch.stack(layer_spikes, dim=-1))

        return (
            torch.stack(outputs, dim=1),
            torch.stack(spike_outputs, dim=1),
            tuple(states),  # type: ignore[return-value]
        )


class HarmonicCandidateSetEpisodeSNN(nn.Module):
    """Rank at most twelve RR candidates with a causal graph/episode SNN.

    The compact forward contract is::

        output = model(
            node_features,      # [B,T,K,F]
            candidate_rr,       # [B,T,K]
            candidate_mask,     # [B,T,K]
            sequence_mask,      # [B,T]
            radar_mask=...,     # optional [B,T,R]
            state=...,          # optional explicit PLIF/ALIF state
            reset_mask=...,     # optional [B,T], reset before this window
        )

    No reference value, QC value, frozen-base error, or future context is
    accepted.  Chunked inference carries ``output["state"]`` into the next
    invocation.  Candidate logits are listwise and masked; decoding uses the
    top candidate rather than an expectation across incompatible harmonics.
    """

    MAX_CANDIDATES = 12

    def __init__(
        self,
        *,
        node_features: int = 152,
        hidden_channels: int = 64,
        num_radars: int = 3,
        graph_blocks: int = 2,
        attention_heads: int = 4,
        beta: float = 0.92,
        adaptation_decay: float = 0.97,
        adaptation_strength: float = 0.40,
        dropout: float = 0.05,
        max_residual_bpm: float = 0.75,
        minimum_scale_bpm: float = 0.25,
        maximum_scale_bpm: float = 6.0,
        initial_scale_bpm: float = 1.0,
        rr_min: float = 6.0,
        rr_max: float = 45.0,
        anchor_enabled: bool = False,
        anchor_max_residual_bpm: float = 12.0,
        anchor_minimum_scale_bpm: float = 0.25,
        anchor_maximum_scale_bpm: float = 12.0,
        anchor_initial_scale_bpm: float = 1.5,
        anchor_distance_weight: float = 1.0,
        anchor_source_mode: str = "learned_blend",
    ) -> None:
        super().__init__()
        if node_features < 1 or hidden_channels < 8 or num_radars < 1:
            raise ValueError("feature/radar dimensions must be positive and hidden >= 8")
        if graph_blocks != 2:
            raise ValueError("the locked architecture requires exactly two graph blocks")
        if hidden_channels % attention_heads:
            raise ValueError("hidden_channels must be divisible by attention_heads")
        if not 0.0 < beta < 1.0 or not 0.0 <= dropout < 1.0:
            raise ValueError("beta/dropout are outside their valid ranges")
        if not 0.0 < minimum_scale_bpm < initial_scale_bpm <= maximum_scale_bpm:
            raise ValueError("candidate scale bounds/initializer are inconsistent")
        if not 0.0 < max_residual_bpm or not rr_min < rr_max:
            raise ValueError("RR limits and max residual must be positive/increasing")
        if anchor_source_mode not in {"corrected_anchor", "learned_blend"}:
            raise ValueError(
                "anchor_source_mode must be 'corrected_anchor' or 'learned_blend'"
            )
        if anchor_enabled and not (
            anchor_max_residual_bpm > 0.0
            and 0.0 < anchor_minimum_scale_bpm
            < anchor_initial_scale_bpm
            <= anchor_maximum_scale_bpm
            and anchor_distance_weight >= 0.0
        ):
            raise ValueError("anchor residual/scale/distance settings are inconsistent")

        self.node_feature_count = int(node_features)
        self.hidden_channels = int(hidden_channels)
        self.num_radars = int(num_radars)
        self.max_residual_bpm = float(max_residual_bpm)
        self.minimum_scale_bpm = float(minimum_scale_bpm)
        self.maximum_scale_bpm = float(maximum_scale_bpm)
        self.rr_min = float(rr_min)
        self.rr_max = float(rr_max)
        self.anchor_enabled = bool(anchor_enabled)
        self.anchor_max_residual_bpm = float(anchor_max_residual_bpm)
        self.anchor_minimum_scale_bpm = float(anchor_minimum_scale_bpm)
        self.anchor_maximum_scale_bpm = float(anchor_maximum_scale_bpm)
        self.anchor_distance_weight = float(anchor_distance_weight)
        self.anchor_source_mode = str(anchor_source_mode)

        # Candidate RR and radar coverage are deployment-time, label-free
        # scalars.  Appending them here makes graph ordering and missingness
        # observable without requiring the evidence builder to duplicate them.
        self.node_projection = nn.Sequential(
            nn.Linear(self.node_feature_count + 2, self.hidden_channels),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_channels),
        )
        self.graph = nn.ModuleList(
            _HarmonicGraphPLIFBlock(
                self.hidden_channels, dropout=float(dropout)
            )
            for _ in range(2)
        )
        # One parametric neuron is intentionally shared across both graph
        # transitions.  Besides matching the locked architecture, carrying its
        # membrane makes the learnable beta identifiable (a one-step PLIF
        # initialized at zero would otherwise make beta inert).
        self.graph_cell = EpisodeSpikingCell(
            self.hidden_channels, cell_type="plif", beta=float(beta)
        )
        self.candidate_pool = _MaskedMultiheadCandidatePool(
            self.hidden_channels, heads=int(attention_heads)
        )
        self.episode_projection = nn.Sequential(
            nn.Linear(2 * self.hidden_channels, self.hidden_channels),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_channels),
        )
        self.radar_context = nn.Linear(self.num_radars, self.hidden_channels, bias=False)
        self.episode_encoder = _CausalPLIFALIFEncoder(
            self.hidden_channels,
            beta=float(beta),
            dropout=float(dropout),
            adaptation_decay=float(adaptation_decay),
            adaptation_strength=float(adaptation_strength),
        )

        candidate_head_features = 2 * self.hidden_channels

        def candidate_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(candidate_head_features, self.hidden_channels),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_channels, 1),
            )

        self.candidate_logit_head = candidate_head()
        self.residual_head = candidate_head()
        self.scale_head = candidate_head()
        self.factor_head = nn.Linear(self.hidden_channels, 4)
        self.quality_head = nn.Linear(self.hidden_channels, 1)

        # The posterior-anchor path is instantiated only for iteration three.
        # Consequently the default/disabled state_dict is exactly compatible
        # with checkpoints written before this optional architecture existed.
        # The context is causal and label-free: current posterior mean/std plus
        # an availability bit, projected without a bias so an unavailable
        # anchor is structurally identical to no anchor context.
        if self.anchor_enabled:
            self.anchor_context_projection = nn.Linear(
                3, self.hidden_channels, bias=False
            )

            def anchor_head() -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(self.hidden_channels, self.hidden_channels),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.hidden_channels, 1),
                )

            self.anchor_residual_head = anchor_head()
            self.anchor_scale_head = anchor_head()
            self.anchor_snap_gate_head = anchor_head()

            # A zero residual and exactly-zero snap gate make the initial i3
            # source bit-equal to the strict raw anchor.  ``clamp`` at zero has
            # a useful subgradient, so the gate can still learn away from the
            # safe baseline.  Candidate-distance guidance is multiplied by
            # this same gate and is therefore initially completely disabled.
            nn.init.zeros_(self.anchor_residual_head[-1].weight)
            nn.init.zeros_(self.anchor_residual_head[-1].bias)
            nn.init.zeros_(self.anchor_snap_gate_head[-1].weight)
            nn.init.zeros_(self.anchor_snap_gate_head[-1].bias)
            nn.init.zeros_(self.anchor_scale_head[-1].weight)
            anchor_scale_raw = math.log(
                math.expm1(
                    float(anchor_initial_scale_bpm)
                    - self.anchor_minimum_scale_bpm
                )
            )
            nn.init.constant_(self.anchor_scale_head[-1].bias, anchor_scale_raw)

        # Start as an exact candidate-center decoder with a moderate, positive
        # scale.  Selection logits remain trainable from the first update.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.scale_head[-1].weight)
        scale_raw = math.log(
            math.expm1(float(initial_scale_bpm) - self.minimum_scale_bpm)
        )
        nn.init.constant_(self.scale_head[-1].bias, scale_raw)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> HarmonicSetState:
        """Return an explicit zero PLIF/ALIF state for streaming inference."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        reference = torch.zeros(
            (int(batch_size), self.hidden_channels),
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )
        return self.episode_encoder.initial_state(reference)

    def _validate_inputs(
        self,
        node_features: Tensor,
        candidate_rr: Tensor,
        candidate_mask: Tensor,
        sequence_mask: Tensor,
        radar_mask: Tensor | None,
        reset_mask: Tensor | None,
        anchor_rr: Tensor | None,
        anchor_std: Tensor | None,
        anchor_available: Tensor | None,
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]:
        if node_features.ndim != 4:
            raise ValueError("node_features must have shape [batch, time, K, F]")
        batch, time_steps, candidates, features = node_features.shape
        if features != self.node_feature_count:
            raise ValueError(
                f"expected {self.node_feature_count} node features, got {features}"
            )
        if not 1 <= candidates <= self.MAX_CANDIDATES:
            raise ValueError("candidate count K must be in [1, 12]")
        if candidate_rr.shape != (batch, time_steps, candidates):
            raise ValueError("candidate_rr must have shape [batch, time, K]")
        if candidate_mask.shape != candidate_rr.shape:
            raise ValueError("candidate_mask must have shape [batch, time, K]")
        if sequence_mask.shape != (batch, time_steps):
            raise ValueError("sequence_mask must have shape [batch, time]")
        if radar_mask is None:
            radar_mask = torch.ones(
                (batch, time_steps, self.num_radars),
                device=node_features.device,
                dtype=torch.bool,
            )
        elif radar_mask.shape != (batch, time_steps, self.num_radars):
            raise ValueError(
                f"radar_mask must have shape [batch, time, {self.num_radars}]"
            )
        if reset_mask is None:
            reset_mask = torch.zeros(
                (batch, time_steps), device=node_features.device, dtype=torch.bool
            )
        elif reset_mask.shape != (batch, time_steps):
            raise ValueError("reset_mask must have shape [batch, time]")

        anchor_values = (anchor_rr, anchor_std, anchor_available)
        if not self.anchor_enabled:
            if any(value is not None for value in anchor_values):
                raise ValueError(
                    "posterior-anchor inputs require anchor_enabled=True"
                )
            anchor_rr = torch.zeros(
                (batch, time_steps), device=node_features.device,
                dtype=node_features.dtype,
            )
            anchor_std = torch.ones_like(anchor_rr)
            anchor_available = torch.zeros(
                (batch, time_steps), device=node_features.device,
                dtype=torch.bool,
            )
        elif all(value is None for value in anchor_values):
            # Omitting the entire context is the explicit no-anchor structural
            # path.  Supplying only part of the three-field binding is invalid.
            anchor_rr = torch.zeros(
                (batch, time_steps), device=node_features.device,
                dtype=node_features.dtype,
            )
            anchor_std = torch.ones_like(anchor_rr)
            anchor_available = torch.zeros(
                (batch, time_steps), device=node_features.device,
                dtype=torch.bool,
            )
        elif any(value is None for value in anchor_values):
            raise ValueError(
                "anchor_rr, anchor_std, and anchor_available must be supplied together"
            )
        else:
            assert anchor_rr is not None
            assert anchor_std is not None
            assert anchor_available is not None
            if anchor_rr.shape != (batch, time_steps):
                raise ValueError("anchor_rr must have shape [batch, time]")
            if anchor_std.shape != (batch, time_steps):
                raise ValueError("anchor_std must have shape [batch, time]")
            if anchor_available.shape != (batch, time_steps):
                raise ValueError("anchor_available must have shape [batch, time]")
            if anchor_available.dtype != torch.bool:
                raise ValueError("anchor_available must have boolean dtype")

        if not torch.is_floating_point(node_features):
            node_features = node_features.float()
        device, dtype = node_features.device, node_features.dtype
        candidate_rr = candidate_rr.to(device=device, dtype=dtype)
        candidate_mask = candidate_mask.to(device=device, dtype=torch.bool)
        sequence_mask = sequence_mask.to(device=device, dtype=torch.bool)
        radar_mask = radar_mask.to(device=device, dtype=torch.bool)
        reset_mask = reset_mask.to(device=device, dtype=torch.bool)
        anchor_rr = anchor_rr.to(device=device, dtype=dtype)
        anchor_std = anchor_std.to(device=device, dtype=dtype)
        anchor_available = anchor_available.to(device=device, dtype=torch.bool)
        if not torch.isfinite(anchor_rr).all():
            raise ValueError("anchor_rr must be finite, including unavailable placeholders")
        if not torch.isfinite(anchor_std).all():
            raise ValueError("anchor_std must be finite, including unavailable placeholders")
        invalid_available_anchor = anchor_available & (
            (anchor_rr < self.rr_min)
            | (anchor_rr > self.rr_max)
            | (anchor_std <= 0.0)
        )
        if invalid_available_anchor.any():
            raise ValueError(
                "available anchors require in-range RR and positive finite std"
            )
        anchor_available = anchor_available & sequence_mask
        anchor_rr = torch.where(
            anchor_available, anchor_rr, torch.zeros_like(anchor_rr)
        )
        anchor_std = torch.where(
            anchor_available, anchor_std, torch.ones_like(anchor_std)
        )
        valid_rr = (
            torch.isfinite(candidate_rr)
            & (candidate_rr >= self.rr_min)
            & (candidate_rr <= self.rr_max)
        )
        radar_available = radar_mask.any(dim=-1)
        candidate_mask = (
            candidate_mask
            & valid_rr
            & sequence_mask[..., None]
            & radar_available[..., None]
        )
        node_features = torch.nan_to_num(
            node_features, nan=0.0, posinf=12.0, neginf=-12.0
        ).clamp(-12.0, 12.0)
        node_features = node_features * candidate_mask[..., None].to(dtype)
        candidate_rr = torch.where(
            candidate_mask, candidate_rr, torch.zeros_like(candidate_rr)
        )
        return (
            node_features,
            candidate_rr,
            candidate_mask,
            sequence_mask,
            radar_mask,
            reset_mask,
            anchor_rr,
            anchor_std,
            anchor_available,
        )

    def forward(
        self,
        node_features: Tensor,
        candidate_rr: Tensor,
        candidate_mask: Tensor,
        sequence_mask: Tensor,
        *,
        radar_mask: Tensor | None = None,
        state: HarmonicSetState | None = None,
        reset_mask: Tensor | None = None,
        anchor_rr: Tensor | None = None,
        anchor_std: Tensor | None = None,
        anchor_available: Tensor | None = None,
    ) -> dict[str, Tensor | HarmonicSetState]:
        """Run a chronological chunk and return predictions plus final state."""

        (
            node_features,
            candidate_rr,
            candidate_mask,
            sequence_mask,
            radar_mask,
            reset_mask,
            anchor_rr,
            anchor_std,
            anchor_available,
        ) = self._validate_inputs(
            node_features,
            candidate_rr,
            candidate_mask,
            sequence_mask,
            radar_mask,
            reset_mask,
            anchor_rr,
            anchor_std,
            anchor_available,
        )
        batch, time_steps, candidates, _ = node_features.shape
        dtype = node_features.dtype
        radar_fraction = radar_mask.to(dtype).mean(dim=-1)
        normalized_rr = candidate_rr / self.rr_max
        node_input = torch.cat(
            (
                node_features,
                normalized_rr[..., None],
                radar_fraction[..., None, None].expand(-1, -1, candidates, -1),
            ),
            dim=-1,
        )
        nodes = self.node_projection(node_input)
        nodes = nodes * candidate_mask[..., None].to(dtype)

        edge_types = build_harmonic_edge_types(candidate_rr, candidate_mask)
        flat_nodes = nodes.reshape(batch * time_steps, candidates, self.hidden_channels)
        flat_edges = edge_types.reshape(batch * time_steps, candidates, candidates, 4)
        flat_mask = candidate_mask.reshape(batch * time_steps, candidates)
        graph_spikes: list[Tensor] = []
        graph_state = self.graph_cell.initial_state(torch.zeros_like(flat_nodes))
        for block in self.graph:
            flat_nodes, spikes, graph_state = block(
                flat_nodes,
                flat_edges,
                flat_mask,
                cell=self.graph_cell,
                state=graph_state,
            )
            graph_spikes.append(spikes)
        nodes = flat_nodes.reshape(
            batch, time_steps, candidates, self.hidden_channels
        )

        attention_pool, attention_weights = self.candidate_pool(nodes, candidate_mask)
        node_count = candidate_mask.to(dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_pool = nodes.sum(dim=2) / node_count
        episode_input = self.episode_projection(
            torch.cat((attention_pool, mean_pool), dim=-1)
        )
        episode_input = episode_input + self.radar_context(radar_mask.to(dtype))
        candidate_observation = (
            candidate_mask.any(dim=-1) & radar_mask.any(dim=-1)
        )
        if self.anchor_enabled:
            anchor_midpoint = 0.5 * (self.rr_min + self.rr_max)
            anchor_half_range = 0.5 * (self.rr_max - self.rr_min)
            anchor_context_values = torch.stack(
                (
                    (anchor_rr - anchor_midpoint) / anchor_half_range,
                    torch.log1p(anchor_std).clamp(max=4.0) / 4.0,
                    anchor_available.to(dtype),
                ),
                dim=-1,
            ) * anchor_available[..., None].to(dtype)
            anchor_context = self.anchor_context_projection(anchor_context_values)
            episode_input = episode_input + anchor_context
            observation_mask = candidate_observation | anchor_available
        else:
            observation_mask = candidate_observation
        episode_input = episode_input * observation_mask[..., None].to(dtype)
        state_sequence, temporal_spikes, final_state = self.episode_encoder(
            episode_input,
            sequence_mask,
            observation_mask=observation_mask,
            state=state,
            reset_mask=reset_mask,
        )

        expanded_state = state_sequence.unsqueeze(2).expand(
            -1, -1, candidates, -1
        )
        candidate_context = torch.cat((nodes, expanded_state), dim=-1)
        candidate_logits = self.candidate_logit_head(candidate_context).squeeze(-1)
        candidate_residual = self.max_residual_bpm * torch.tanh(
            self.residual_head(candidate_context).squeeze(-1)
        )
        candidate_residual = candidate_residual * candidate_mask.to(dtype)
        candidate_scale = self.minimum_scale_bpm + F.softplus(
            self.scale_head(candidate_context).squeeze(-1)
        )
        candidate_scale = candidate_scale.clamp(max=self.maximum_scale_bpm)
        candidate_mean = (candidate_rr + candidate_residual).clamp(
            min=self.rr_min, max=self.rr_max
        )
        candidate_mean = candidate_mean * candidate_mask.to(dtype)

        # Iteration-three posterior-anchor residual path.  Every prediction is
        # a function only of the chronological SNN state and the current
        # label-free nested posterior.  The corrected-anchor distance term is
        #   -lambda * gate * |candidate_mean - corrected_anchor| / uncertainty
        # and is applied only on explicitly available anchor rows.  The gate
        # is exactly zero at initialization, preserving the strict anchor and
        # the unguided candidate logits until learned evidence changes it.
        anchor_output: dict[str, Tensor] = {}
        if self.anchor_enabled:
            anchor_residual = self.anchor_max_residual_bpm * torch.tanh(
                self.anchor_residual_head(state_sequence).squeeze(-1)
            )
            anchor_residual = anchor_residual * anchor_available.to(dtype)
            corrected_anchor = (anchor_rr + anchor_residual).clamp(
                min=self.rr_min, max=self.rr_max
            )
            corrected_anchor = torch.where(
                anchor_available, corrected_anchor, torch.zeros_like(corrected_anchor)
            )
            anchor_scale = self.anchor_minimum_scale_bpm + F.softplus(
                self.anchor_scale_head(state_sequence).squeeze(-1)
            )
            anchor_scale = anchor_scale.clamp(max=self.anchor_maximum_scale_bpm)
            anchor_scale = torch.where(
                anchor_available,
                anchor_scale,
                torch.full_like(anchor_scale, self.anchor_maximum_scale_bpm),
            )
            gate_raw = self.anchor_snap_gate_head(state_sequence).squeeze(-1)
            # Subtract the same tensor operation at zero (instead of a Python
            # constant) so the initialized gate is exactly +0 in fp16/bf16 as
            # well as fp32.
            anchor_snap_gate = (
                F.softplus(gate_raw) - F.softplus(torch.zeros_like(gate_raw))
            ).clamp(min=0.0, max=1.0)
            anchor_snap_gate = anchor_snap_gate * anchor_available.to(dtype)
            distance = (
                candidate_mean - corrected_anchor.unsqueeze(-1)
            ).abs() / anchor_scale.unsqueeze(-1).clamp_min(
                self.anchor_minimum_scale_bpm
            )
            guided_logits = candidate_logits - (
                self.anchor_distance_weight
                * anchor_snap_gate.unsqueeze(-1)
                * distance
            )
            # Preserve the original candidate path byte-for-byte whenever the
            # anchor is unavailable; do not rely on floating +/- zero.
            candidate_logits = torch.where(
                anchor_available.unsqueeze(-1), guided_logits, candidate_logits
            )
            anchor_output = {
                "raw_anchor_rr": anchor_rr,
                "raw_anchor_std_bpm": anchor_std,
                "anchor_available": anchor_available,
                "anchor_residual_bpm": anchor_residual,
                "anchor_residual_limit_bpm": torch.full_like(
                    anchor_residual, self.anchor_max_residual_bpm
                ),
                "corrected_anchor_rr": corrected_anchor,
                "corrected_anchor_scale_bpm": anchor_scale,
                "anchor_snap_gate": anchor_snap_gate,
                "anchor_snap_gate_logit": gate_raw,
            }

        candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1.0e4)
        candidate_probabilities = _masked_softmax(
            candidate_logits, candidate_mask, dim=-1
        )

        factor_logits = self.factor_head(state_sequence)
        quality_logit = self.quality_head(state_sequence).squeeze(-1)
        candidate_available = candidate_mask.any(dim=-1)
        candidate_source_available = (
            sequence_mask & candidate_available & radar_mask.any(dim=-1)
        )
        selected_index = candidate_logits.argmax(dim=-1)
        selected_index = torch.where(
            candidate_source_available,
            selected_index,
            torch.full_like(selected_index, -1),
        )
        gather_index = selected_index.clamp_min(0).unsqueeze(-1)

        def selected(values: Tensor) -> Tensor:
            return values.gather(dim=-1, index=gather_index).squeeze(-1)

        selected_mean = selected(candidate_mean)
        selected_probability = selected(candidate_probabilities)
        selected_scale = selected(candidate_scale)
        candidate_source_rr = torch.where(
            candidate_source_available, selected_mean, torch.zeros_like(selected_mean)
        )
        selected_probability = torch.where(
            candidate_source_available,
            selected_probability,
            torch.zeros_like(selected_probability),
        )
        # A finite maximum scale is safer than zero/NaN when no source exists;
        # source_available remains the authoritative usability flag.
        candidate_source_scale = torch.where(
            candidate_source_available,
            selected_scale,
            torch.full_like(selected_scale, self.maximum_scale_bpm),
        )

        if self.anchor_enabled:
            corrected_anchor = anchor_output["corrected_anchor_rr"]
            anchor_scale = anchor_output["corrected_anchor_scale_bpm"]
            snap_gate = anchor_output["anchor_snap_gate"]
            anchor_usable = sequence_mask & anchor_available
            if self.anchor_source_mode == "learned_blend":
                blend_gate = snap_gate * candidate_source_available.to(dtype)
                blended_rr = corrected_anchor + blend_gate * (
                    candidate_source_rr - corrected_anchor
                )
                blended_scale = anchor_scale + blend_gate * (
                    candidate_source_scale - anchor_scale
                )
                # Fully-off gate is an exact raw/corrected anchor copy.
                blended_rr = torch.where(
                    blend_gate == 0.0, corrected_anchor, blended_rr
                )
                blended_scale = torch.where(
                    blend_gate == 0.0, anchor_scale, blended_scale
                )
            else:
                blended_rr, blended_scale = corrected_anchor, anchor_scale
            source_rr = torch.where(
                anchor_usable, blended_rr, candidate_source_rr
            )
            source_scale = torch.where(
                anchor_usable, blended_scale, candidate_source_scale
            )
            source_available = anchor_usable | candidate_source_available
            source_rr = torch.where(
                source_available, source_rr, torch.zeros_like(source_rr)
            )
            source_scale = torch.where(
                source_available,
                source_scale,
                torch.full_like(source_scale, self.maximum_scale_bpm),
            )
            anchor_output["candidate_source_rr"] = candidate_source_rr
            anchor_output["candidate_source_scale_bpm"] = candidate_source_scale
            anchor_output["candidate_source_available"] = candidate_source_available
        else:
            source_rr = candidate_source_rr
            source_scale = candidate_source_scale
            source_available = candidate_source_available

        graph_spike_sequence = torch.stack(
            [
                spikes.reshape(
                    batch, time_steps, candidates, self.hidden_channels
                ).sum(dim=(-2, -1))
                / (candidate_mask.to(dtype).sum(dim=-1) * self.hidden_channels).clamp_min(1.0)
                for spikes in graph_spikes
            ],
            dim=-1,
        )
        spike_sequence = torch.cat((graph_spike_sequence, temporal_spikes), dim=-1)
        sequence_count = sequence_mask.to(dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        spike_rates = (
            spike_sequence * sequence_mask[..., None].to(dtype)
        ).sum(dim=1) / sequence_count

        result: dict[str, Tensor | HarmonicSetState] = {
            "candidate_logits": candidate_logits,
            "candidate_probabilities": candidate_probabilities,
            "candidate_residual_bpm": candidate_residual,
            "candidate_mean_bpm": candidate_mean,
            "candidate_scale_bpm": candidate_scale,
            "factor_logits": factor_logits,
            "quality_logit": quality_logit,
            "quality": quality_logit.sigmoid(),
            "selected_index": selected_index,
            "selected_probability": selected_probability,
            "source_rr": source_rr,
            "source_scale_bpm": source_scale,
            "source_available": source_available,
            "node_embeddings": nodes,
            "candidate_attention": attention_weights,
            "state_sequence": state_sequence,
            "spike_sequence": spike_sequence,
            "spike_rates": spike_rates,
            "state": final_state,
        }
        result.update(anchor_output)
        return result


# Short import name for training and deployment code.
HarmonicCandidateSetSNN = HarmonicCandidateSetEpisodeSNN


__all__ = [
    "HarmonicCandidateSetEpisodeSNN",
    "HarmonicCandidateSetSNN",
    "HarmonicSetState",
    "build_harmonic_edge_types",
]
