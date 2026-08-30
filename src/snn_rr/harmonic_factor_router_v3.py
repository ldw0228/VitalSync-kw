"""Directed harmonic factor-expert SNN for the retrospective v3 campaign.

This module is deliberately self-contained and has a strict, label-free
forward interface.  It consumes the frozen 571-wide harmonic cache layout,
builds a *directed* candidate graph, and routes between a corrected causal
anchor and the candidate experts with a hard argmax.  In particular, it never
forms an arithmetic blend between incompatible harmonic hypotheses.

The recurrent state is explicit.  Passing ``output["state"]`` to the next
chronological chunk is therefore equivalent to evaluating the whole physical
session at once (in evaluation mode).  ``reset_mask`` resets both factor-router
neuron states immediately before the corresponding window.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Final, TypeAlias

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .svd_episode_models import EpisodeSpikingCell


NeuronState: TypeAlias = tuple[Tensor, Tensor]
FactorRouterState: TypeAlias = tuple[NeuronState, NeuronState]

EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256: Final[str] = (
    "d7553f8b11733903393575d02bc6acd4a8edefd5ce0e538491295ec84d938f05"
)
FACTOR_CLASSES: Final[tuple[float, ...]] = (1.0, 2.0, 3.0, 4.0)
RF_SVD_RATIOS: Final[tuple[float, ...]] = (
    0.25,
    1.0 / 3.0,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
)
DIRECTED_RELATIONS: Final[tuple[str, ...]] = (
    "near",
    "receiver_is_2x_sender",
    "sender_is_2x_receiver",
    "receiver_is_3x_sender",
    "sender_is_3x_receiver",
    "receiver_is_4x_sender",
    "sender_is_4x_receiver",
)

FEATURE_LAYOUT: Final[dict[str, object]] = {
    "total_width": 571,
    "concatenation_order": ["core", "rf", "svd"],
    "core": {"offset": 0, "width": 46, "shape": [46]},
    "rf": {
        "offset": 46,
        "width": 378,
        "shape": [3, 7, 2, 9],
        "axis_order": ["radar", "ratio", "branch", "statistic"],
    },
    "svd": {
        "offset": 424,
        "width": 147,
        "shape": [3, 7, 7],
        "axis_order": ["radar", "ratio", "statistic"],
    },
    "ordered_names_semantic_sha256": EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
}


def _semantic_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


FEATURE_LAYOUT_SEMANTIC_SHA256: Final[str] = _semantic_sha256(FEATURE_LAYOUT)


def validate_feature_layout_binding(
    *,
    total_width: int,
    ordered_feature_names_semantic_sha256: str,
    structural_layout_semantic_sha256: str = FEATURE_LAYOUT_SEMANTIC_SHA256,
) -> None:
    """Fail closed when a cache's ordered schema is not the frozen v3 layout."""

    if int(total_width) != 571:
        raise ValueError(f"DHFER-SNN requires exactly 571 features, got {total_width}")
    if ordered_feature_names_semantic_sha256 != EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256:
        raise ValueError("ordered feature-name semantic digest does not match v3")
    if structural_layout_semantic_sha256 != FEATURE_LAYOUT_SEMANTIC_SHA256:
        raise ValueError("structural feature-layout semantic digest does not match v3")


def _masked_softmax(values: Tensor, mask: Tensor, *, dim: int) -> Tensor:
    masked = values.masked_fill(~mask, -1.0e4)
    probability = masked.softmax(dim=dim) * mask.to(values.dtype)
    return probability / probability.sum(dim=dim, keepdim=True).clamp_min(1.0e-8)


def build_directed_harmonic_relations(
    candidate_rr_bpm: Tensor,
    candidate_mask: Tensor,
    *,
    near_tolerance_bpm: float = 0.5,
    ratio_tolerance_bpm: float = 0.75,
) -> Tensor:
    """Return receiver-by-sender adjacency for the seven fixed relations.

    The result has shape ``[..., receiver, sender, 7]``.  The x2/x3/x4
    channels intentionally retain direction; no symmetric union is made.
    """

    if candidate_rr_bpm.ndim < 1 or candidate_rr_bpm.shape != candidate_mask.shape:
        raise ValueError("candidate_rr_bpm and candidate_mask must share shape [..., K]")
    if candidate_rr_bpm.shape[-1] < 1:
        raise ValueError("candidate dimension cannot be empty")
    if near_tolerance_bpm <= 0.0 or ratio_tolerance_bpm <= 0.0:
        raise ValueError("directed graph tolerances must be positive")
    rr = (
        candidate_rr_bpm
        if torch.is_floating_point(candidate_rr_bpm)
        else candidate_rr_bpm.float()
    )
    valid = candidate_mask.to(torch.bool) & torch.isfinite(rr) & (rr > 0.0)
    receiver = rr.unsqueeze(-1)
    sender = rr.unsqueeze(-2)
    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    identity = torch.eye(rr.shape[-1], device=rr.device, dtype=torch.bool).reshape(
        (1,) * (rr.ndim - 1) + (rr.shape[-1], rr.shape[-1])
    )
    pair_valid = pair_valid & ~identity
    relations = [
        pair_valid & ((receiver - sender).abs() <= float(near_tolerance_bpm))
    ]
    for factor in (2.0, 3.0, 4.0):
        relations.append(
            pair_valid
            & ((receiver - factor * sender).abs() <= float(ratio_tolerance_bpm))
        )
        relations.append(
            pair_valid
            & ((sender - factor * receiver).abs() <= float(ratio_tolerance_bpm))
        )
    return torch.stack(relations, dim=-1)


def factor_candidate_affinity(
    candidate_rr_bpm: Tensor,
    candidate_mask: Tensor,
    classical_rr_bpm: Tensor,
    classical_available: Tensor,
    *,
    bandwidth_bpm: float = 0.75,
) -> Tensor:
    """Compute the fixed label-free candidate-to-factor affinity tensor."""

    if candidate_rr_bpm.shape != candidate_mask.shape:
        raise ValueError("candidate RR and mask shapes differ")
    if classical_rr_bpm.shape != candidate_rr_bpm.shape[:-1]:
        raise ValueError("classical_rr_bpm must have shape candidate_rr_bpm.shape[:-1]")
    if classical_available.shape != classical_rr_bpm.shape:
        raise ValueError("classical_available shape differs from classical_rr_bpm")
    if bandwidth_bpm <= 0.0:
        raise ValueError("factor affinity bandwidth must be positive")
    factors = candidate_rr_bpm.new_tensor(FACTOR_CLASSES)
    centers = classical_rr_bpm.unsqueeze(-1) * factors
    affinity = torch.exp(
        -(
            candidate_rr_bpm.unsqueeze(-1) - centers.unsqueeze(-2)
        ).abs()
        / float(bandwidth_bpm)
    )
    mask = candidate_mask.unsqueeze(-1) & classical_available[..., None, None]
    return torch.where(mask, affinity, torch.zeros_like(affinity))


class _SharedCellEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.LayerNorm(16),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class StructuredHarmonicEvidenceEncoder(nn.Module):
    """Encode core/RF/SVD fields without collapsing their frozen axes."""

    def __init__(self, *, hidden_channels: int = 64) -> None:
        super().__init__()
        if hidden_channels != 64:
            raise ValueError("the v3 structured encoder requires hidden_channels=64")
        self.core_encoder = nn.Sequential(
            nn.Linear(46, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.LayerNorm(32),
        )
        self.rf_cell_encoder = _SharedCellEncoder(9)
        self.svd_cell_encoder = _SharedCellEncoder(7)
        self.radar_embedding = nn.Embedding(3, 16)
        self.ratio_embedding = nn.Embedding(7, 16)
        self.rf_branch_embedding = nn.Embedding(2, 16)
        self.output_projection = nn.Sequential(
            nn.Linear(64, 64), nn.SiLU(), nn.LayerNorm(64)
        )

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor, *, dims: tuple[int, ...]) -> Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        numerator = (values * weights).sum(dim=dims)
        denominator = weights.sum(dim=dims).clamp_min(1.0)
        return numerator / denominator

    def forward(
        self,
        features: Tensor,
        node_mask: Tensor,
        radar_mask: Tensor,
        ratio_in_band: Tensor,
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        if features.ndim != 4 or features.shape[-1] != 571:
            raise ValueError("features must have shape [batch,time,K,571]")
        if node_mask.shape != features.shape[:3]:
            raise ValueError("node_mask must have shape [batch,time,K]")
        if radar_mask.shape != (*features.shape[:2], 3):
            raise ValueError("radar_mask must have shape [batch,time,3]")
        if ratio_in_band.shape != (*features.shape[:3], 7):
            raise ValueError("ratio_in_band must have shape [batch,time,K,7]")

        core = features[..., :46]
        rf = features[..., 46:424].reshape(*features.shape[:3], 3, 7, 2, 9)
        svd = features[..., 424:].reshape(*features.shape[:3], 3, 7, 7)

        cell_base = (
            node_mask[..., None, None]
            & radar_mask[..., None, :, None]
            & ratio_in_band[..., None, :]
        )
        rf_mask = torch.stack(
            (cell_base, torch.zeros_like(cell_base)), dim=-1
        )
        svd_mask = cell_base

        # The cache contract says masked cells are exact zero after fitting the
        # outer-train scaler.  Checking this catches layout/axis drift instead
        # of silently treating a zero-valued available statistic as missing.
        if torch.count_nonzero(core.masked_select(~node_mask[..., None])):
            raise ValueError("masked core features must be exact zero")
        if torch.count_nonzero(rf.masked_select(~rf_mask[..., None])):
            raise ValueError("masked RF cells (including IQ branch) must be exact zero")
        if torch.count_nonzero(svd.masked_select(~svd_mask[..., None])):
            raise ValueError("masked SVD cells must be exact zero")

        core_encoded = self.core_encoder(core) * node_mask[..., None].to(features.dtype)

        radar_index = torch.arange(3, device=features.device).reshape(1, 1, 1, 3, 1, 1)
        ratio_index = torch.arange(7, device=features.device).reshape(1, 1, 1, 1, 7, 1)
        branch_index = torch.arange(2, device=features.device).reshape(1, 1, 1, 1, 1, 2)
        rf_encoded = self.rf_cell_encoder(rf)
        rf_encoded = rf_encoded + self.radar_embedding(radar_index)
        rf_encoded = rf_encoded + self.ratio_embedding(ratio_index)
        rf_encoded = rf_encoded + self.rf_branch_embedding(branch_index)
        rf_pooled = self._masked_mean(rf_encoded, rf_mask, dims=(-4, -3, -2))

        svd_radar_index = torch.arange(3, device=features.device).reshape(1, 1, 1, 3, 1)
        svd_ratio_index = torch.arange(7, device=features.device).reshape(1, 1, 1, 1, 7)
        svd_encoded = self.svd_cell_encoder(svd)
        svd_encoded = svd_encoded + self.radar_embedding(svd_radar_index)
        svd_encoded = svd_encoded + self.ratio_embedding(svd_ratio_index)
        svd_pooled = self._masked_mean(svd_encoded, svd_mask, dims=(-3, -2))

        encoded = self.output_projection(
            torch.cat((core_encoded, rf_pooled, svd_pooled), dim=-1)
        )
        encoded = encoded * node_mask[..., None].to(encoded.dtype)
        diagnostics = {
            "rf_cell_mask": rf_mask,
            "svd_cell_mask": svd_mask,
            "ratio_in_band_mask": ratio_in_band,
        }
        return encoded, diagnostics


class _DirectedGraphPLIFBlock(nn.Module):
    def __init__(self, channels: int, *, dropout: float, beta: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.relation_projections = nn.ModuleList(
            nn.Linear(channels, channels, bias=False) for _ in DIRECTED_RELATIONS
        )
        self.current_projection = nn.Linear(8 * channels, channels)
        self.current_norm = nn.LayerNorm(channels)
        self.cell = EpisodeSpikingCell(channels, cell_type="plif", beta=beta)
        self.readout = nn.Linear(2 * channels, channels)
        self.output_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: Tensor,
        relations: Tensor,
        node_mask: Tensor,
        *,
        simulation_steps: int,
    ) -> tuple[Tensor, Tensor]:
        messages: list[Tensor] = []
        for relation, projection in enumerate(self.relation_projections):
            adjacency = relations[..., relation].to(nodes.dtype)
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            messages.append(projection(torch.matmul(adjacency, nodes) / degree))
        current = self.current_norm(
            self.current_projection(torch.cat((nodes, *messages), dim=-1))
        ) * node_mask[..., None].to(nodes.dtype)
        state = self.cell.initial_state(torch.zeros_like(current))
        spike_sum = torch.zeros_like(current)
        for _ in range(simulation_steps):
            spikes, state = self.cell.forward_step(current, state, node_mask)
            spike_sum = spike_sum + spikes
        mean_spikes = spike_sum / float(simulation_steps)
        update = self.readout(torch.cat((mean_spikes, torch.tanh(state[0])), dim=-1))
        output = self.output_norm(nodes + self.dropout(update))
        return output * node_mask[..., None].to(output.dtype), mean_spikes


class _MaskedCandidatePool(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("candidate pool heads must divide channels")
        self.heads = int(heads)
        self.head_width = int(channels // heads)
        self.queries = nn.Parameter(torch.empty(heads, self.head_width))
        nn.init.normal_(self.queries, std=self.head_width**-0.5)

    def forward(self, nodes: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        split = nodes.reshape(*nodes.shape[:-1], self.heads, self.head_width)
        score = (split * self.queries).sum(dim=-1) / math.sqrt(self.head_width)
        weights = _masked_softmax(score, mask[..., None], dim=2)
        pooled = (split * weights[..., None]).sum(dim=2).flatten(start_dim=-2)
        return pooled, weights


class _CausalFactorPLIFALIF(nn.Module):
    CELL_TYPES = ("plif", "alif")

    def __init__(
        self,
        channels: int,
        *,
        beta: float,
        dropout: float,
        adaptation_decay: float,
        adaptation_strength: float,
        simulation_steps: int,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.simulation_steps = int(simulation_steps)
        self.synapses = nn.ModuleList(nn.Linear(channels, channels) for _ in range(2))
        self.norms = nn.ModuleList(nn.LayerNorm(channels) for _ in range(2))
        self.cells = nn.ModuleList(
            (
                EpisodeSpikingCell(channels, cell_type="plif", beta=beta),
                EpisodeSpikingCell(
                    channels,
                    cell_type="alif",
                    beta=beta,
                    adaptation_decay=adaptation_decay,
                    adaptation_strength=adaptation_strength,
                ),
            )
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(3 * channels, channels), nn.SiLU(), nn.LayerNorm(channels)
        )

    def initial_state(self, reference: Tensor) -> FactorRouterState:
        return tuple(  # type: ignore[return-value]
            cell.initial_state(torch.zeros_like(reference)) for cell in self.cells
        )

    def _validate_state(
        self, state: FactorRouterState, reference: Tensor
    ) -> FactorRouterState:
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise ValueError("state must contain PLIF and ALIF states")
        result: list[NeuronState] = []
        for layer in state:
            if not isinstance(layer, (tuple, list)) or len(layer) != 2:
                raise ValueError("each layer state must be (membrane, adaptation)")
            membrane, adaptation = layer
            if membrane.shape != reference.shape or adaptation.shape != reference.shape:
                raise ValueError("state tensors must have shape [batch, hidden_channels]")
            result.append(
                (
                    membrane.to(device=reference.device, dtype=reference.dtype),
                    adaptation.to(device=reference.device, dtype=reference.dtype),
                )
            )
        return tuple(result)  # type: ignore[return-value]

    def forward(
        self,
        values: Tensor,
        sequence_mask: Tensor,
        reset_mask: Tensor,
        state: FactorRouterState | None,
    ) -> tuple[Tensor, Tensor, FactorRouterState]:
        batch, windows, channels = values.shape
        if channels != self.channels or sequence_mask.shape != (batch, windows):
            raise ValueError("causal factor encoder input shape mismatch")
        reference = values.new_zeros((batch, channels))
        states = list(
            self.initial_state(reference)
            if state is None
            else self._validate_state(state, reference)
        )
        outputs: list[Tensor] = []
        rates: list[Tensor] = []
        for window in range(windows):
            keep = (~reset_mask[:, window]).to(values.dtype).unsqueeze(-1)
            states = [(m * keep, a * keep) for m, a in states]
            analog = values[:, window]
            layer_sums = values.new_zeros((batch, 2))
            last_spikes = torch.zeros_like(analog)
            last_membrane = torch.zeros_like(analog)
            for _ in range(self.simulation_steps):
                current = analog
                for layer, (synapse, norm, cell) in enumerate(
                    zip(self.synapses, self.norms, self.cells, strict=True)
                ):
                    current = norm(synapse(current))
                    spikes, states[layer] = cell.forward_step(
                        current, states[layer], sequence_mask[:, window]
                    )
                    layer_sums[:, layer] += spikes.mean(dim=-1)
                    last_spikes = spikes
                    last_membrane = states[layer][0]
                    current = self.dropout(spikes)
            token = self.readout(
                torch.cat((analog, last_spikes, torch.tanh(last_membrane)), dim=-1)
            ) * sequence_mask[:, window, None].to(values.dtype)
            outputs.append(token)
            rates.append(layer_sums / float(self.simulation_steps))
        return torch.stack(outputs, dim=1), torch.stack(rates, dim=1), tuple(states)  # type: ignore[return-value]


class DirectedHarmonicFactorExpertSNN(nn.Module):
    """Contracted DHFER-SNN with a hard anchor/candidate expert decoder."""

    MAX_CANDIDATES = 12
    VALID_VARIANTS = frozenset(("H0_no_factor", "H1_factor", "H2_full"))

    def __init__(
        self,
        *,
        ordered_feature_names_semantic_sha256: str,
        structural_layout_semantic_sha256: str = FEATURE_LAYOUT_SEMANTIC_SHA256,
        variant: str = "H2_full",
        hidden_channels: int = 64,
        graph_blocks: int = 2,
        simulation_steps: int = 8,
        dropout: float = 0.05,
        beta: float = 0.92,
        adaptation_decay: float = 0.97,
        adaptation_strength: float = 0.40,
        rr_min_bpm: float = 6.0,
        rr_max_bpm: float = 45.0,
        candidate_residual_limit_bpm: float = 0.75,
        candidate_minimum_scale_bpm: float = 0.25,
        candidate_initial_scale_bpm: float = 1.0,
        candidate_maximum_scale_bpm: float = 12.0,
        anchor_residual_limit_bpm: float = 12.0,
        anchor_minimum_scale_bpm: float = 0.25,
        anchor_initial_scale_bpm: float = 1.5,
        anchor_maximum_scale_bpm: float = 12.0,
        factor_logit_boost: float = 2.0,
        factor_affinity_bandwidth_bpm: float = 0.75,
        maximum_parameters: int = 400_000,
    ) -> None:
        super().__init__()
        validate_feature_layout_binding(
            total_width=571,
            ordered_feature_names_semantic_sha256=ordered_feature_names_semantic_sha256,
            structural_layout_semantic_sha256=structural_layout_semantic_sha256,
        )
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VALID_VARIANTS)}")
        if hidden_channels != 64 or graph_blocks != 2 or simulation_steps != 8:
            raise ValueError("v3 locks hidden=64, graph_blocks=2, simulation_steps=8")
        if not 0.0 <= dropout < 1.0 or not 0.0 < beta < 1.0:
            raise ValueError("dropout/beta are outside their valid ranges")
        if not rr_min_bpm < rr_max_bpm:
            raise ValueError("RR bounds must be increasing")
        if candidate_residual_limit_bpm != 0.75:
            raise ValueError("candidate residual limit is locked to 0.75 bpm")
        if anchor_residual_limit_bpm != 12.0:
            raise ValueError("anchor residual limit is locked to 12 bpm")
        if factor_logit_boost != 2.0 or factor_affinity_bandwidth_bpm != 0.75:
            raise ValueError("factor route boost/bandwidth are locked to 2.0/0.75")
        if not (
            0.0 < candidate_minimum_scale_bpm
            < candidate_initial_scale_bpm
            <= candidate_maximum_scale_bpm
        ):
            raise ValueError("candidate scale bounds are inconsistent")
        if not (
            0.0 < anchor_minimum_scale_bpm
            < anchor_initial_scale_bpm
            <= anchor_maximum_scale_bpm
        ):
            raise ValueError("anchor scale bounds are inconsistent")

        self.variant = variant
        self.factor_router_enabled = variant != "H0_no_factor"
        self.hidden_channels = int(hidden_channels)
        self.simulation_steps = int(simulation_steps)
        self.rr_min_bpm = float(rr_min_bpm)
        self.rr_max_bpm = float(rr_max_bpm)
        self.candidate_residual_limit_bpm = float(candidate_residual_limit_bpm)
        self.candidate_minimum_scale_bpm = float(candidate_minimum_scale_bpm)
        self.candidate_maximum_scale_bpm = float(candidate_maximum_scale_bpm)
        self.anchor_residual_limit_bpm = float(anchor_residual_limit_bpm)
        self.anchor_minimum_scale_bpm = float(anchor_minimum_scale_bpm)
        self.anchor_maximum_scale_bpm = float(anchor_maximum_scale_bpm)
        self.factor_logit_boost = float(factor_logit_boost)
        self.factor_affinity_bandwidth_bpm = float(factor_affinity_bandwidth_bpm)
        self.ordered_feature_names_semantic_sha256 = (
            ordered_feature_names_semantic_sha256
        )
        self.structural_layout_semantic_sha256 = structural_layout_semantic_sha256

        self.structured_encoder = StructuredHarmonicEvidenceEncoder(
            hidden_channels=hidden_channels
        )
        self.graph = nn.ModuleList(
            _DirectedGraphPLIFBlock(
                hidden_channels, dropout=dropout, beta=beta
            )
            for _ in range(graph_blocks)
        )
        self.candidate_pool = _MaskedCandidatePool(hidden_channels)
        self.episode_projection = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
        )
        self.radar_context = nn.Linear(3, hidden_channels, bias=False)
        # normalized anchor RR/std/available + classical RR/available
        self.source_context = nn.Linear(5, hidden_channels, bias=False)
        self.factor_temporal = _CausalFactorPLIFALIF(
            hidden_channels,
            beta=beta,
            dropout=dropout,
            adaptation_decay=adaptation_decay,
            adaptation_strength=adaptation_strength,
            simulation_steps=simulation_steps,
        )

        candidate_context_width = 2 * hidden_channels
        self.candidate_logit_head = nn.Linear(candidate_context_width, 1)
        self.candidate_residual_head = nn.Sequential(
            nn.Linear(candidate_context_width, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.candidate_scale_head = nn.Sequential(
            nn.Linear(candidate_context_width, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.factor_head = nn.Linear(hidden_channels, 4)
        self.anchor_logit_head = nn.Linear(hidden_channels, 1)
        self.anchor_residual_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.anchor_scale_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.quality_head = nn.Linear(hidden_channels, 1)

        nn.init.zeros_(self.candidate_logit_head.weight)
        nn.init.zeros_(self.candidate_logit_head.bias)
        nn.init.zeros_(self.anchor_logit_head.weight)
        nn.init.constant_(self.anchor_logit_head.bias, 4.0)
        nn.init.zeros_(self.candidate_residual_head[-1].weight)
        nn.init.zeros_(self.candidate_residual_head[-1].bias)
        nn.init.zeros_(self.anchor_residual_head[-1].weight)
        nn.init.zeros_(self.anchor_residual_head[-1].bias)
        nn.init.zeros_(self.candidate_scale_head[-1].weight)
        nn.init.zeros_(self.anchor_scale_head[-1].weight)
        nn.init.constant_(
            self.candidate_scale_head[-1].bias,
            math.log(
                math.expm1(candidate_initial_scale_bpm - candidate_minimum_scale_bpm)
            ),
        )
        nn.init.constant_(
            self.anchor_scale_head[-1].bias,
            math.log(math.expm1(anchor_initial_scale_bpm - anchor_minimum_scale_bpm)),
        )

        parameter_count = self.parameter_count()
        if parameter_count > int(maximum_parameters):
            raise ValueError(
                f"DHFER-SNN parameter cap exceeded: {parameter_count}>{maximum_parameters}"
            )

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def layout_receipt(self) -> dict[str, str | int]:
        return {
            "total_width": 571,
            "ordered_feature_names_semantic_sha256": self.ordered_feature_names_semantic_sha256,
            "structural_layout_semantic_sha256": self.structural_layout_semantic_sha256,
            "parameter_count": self.parameter_count(),
        }

    def assert_safe_initialization(self) -> None:
        checks = (
            self.candidate_logit_head.weight,
            self.candidate_logit_head.bias,
            self.anchor_logit_head.weight,
            self.candidate_residual_head[-1].weight,
            self.candidate_residual_head[-1].bias,
            self.anchor_residual_head[-1].weight,
            self.anchor_residual_head[-1].bias,
        )
        if any(torch.count_nonzero(value.detach()) for value in checks):
            raise RuntimeError("safe hard-expert initialization has drifted")
        expected = torch.full_like(self.anchor_logit_head.bias.detach(), 4.0)
        if not torch.equal(self.anchor_logit_head.bias.detach(), expected):
            raise RuntimeError("anchor expert bias must initialize to exactly 4.0")

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> FactorRouterState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        reference = torch.zeros(
            (batch_size, self.hidden_channels),
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )
        return self.factor_temporal.initial_state(reference)

    def _validate_inputs(
        self,
        node_features: Tensor,
        candidate_rr_bpm: Tensor,
        candidate_mask: Tensor,
        sequence_mask: Tensor,
        joint_radar_mask: Tensor,
        proposer_anchor_bpm: Tensor,
        proposer_anchor_std_bpm: Tensor,
        proposer_anchor_available: Tensor,
        classical_rr_bpm: Tensor,
        reset_mask: Tensor | None,
    ) -> tuple[Tensor, ...]:
        if node_features.ndim != 4 or node_features.shape[-1] != 571:
            raise ValueError("node_features must have shape [batch,time,K,571]")
        batch, windows, candidates, _ = node_features.shape
        if not 1 <= candidates <= self.MAX_CANDIDATES:
            raise ValueError("candidate count must be in [1,12]")
        if candidate_rr_bpm.shape != (batch, windows, candidates):
            raise ValueError("candidate_rr_bpm must have shape [batch,time,K]")
        if candidate_mask.shape != candidate_rr_bpm.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean [batch,time,K]")
        if sequence_mask.shape != (batch, windows) or sequence_mask.dtype != torch.bool:
            raise ValueError("sequence_mask must be boolean [batch,time]")
        if joint_radar_mask.shape != (batch, windows, 3) or joint_radar_mask.dtype != torch.bool:
            raise ValueError("joint_radar_mask must be boolean [batch,time,3]")
        for name, value in (
            ("proposer_anchor_bpm", proposer_anchor_bpm),
            ("proposer_anchor_std_bpm", proposer_anchor_std_bpm),
            ("classical_rr_bpm", classical_rr_bpm),
        ):
            if value.shape != (batch, windows):
                raise ValueError(f"{name} must have shape [batch,time]")
        if (
            proposer_anchor_available.shape != (batch, windows)
            or proposer_anchor_available.dtype != torch.bool
        ):
            raise ValueError("proposer_anchor_available must be boolean [batch,time]")
        if reset_mask is None:
            reset_mask = torch.zeros_like(sequence_mask)
        elif reset_mask.shape != sequence_mask.shape or reset_mask.dtype != torch.bool:
            raise ValueError("reset_mask must be boolean [batch,time]")
        if (reset_mask & ~sequence_mask).any():
            raise ValueError("reset_mask may only be set on real sequence windows")
        if not torch.is_floating_point(node_features):
            node_features = node_features.float()
        if not torch.isfinite(node_features).all():
            raise ValueError("node_features must be finite after outer-train scaling")
        device, dtype = node_features.device, node_features.dtype
        candidate_rr_bpm = candidate_rr_bpm.to(device=device, dtype=dtype)
        candidate_mask = candidate_mask.to(device=device)
        sequence_mask = sequence_mask.to(device=device)
        joint_radar_mask = joint_radar_mask.to(device=device)
        reset_mask = reset_mask.to(device=device)
        proposer_anchor_bpm = proposer_anchor_bpm.to(device=device, dtype=dtype)
        proposer_anchor_std_bpm = proposer_anchor_std_bpm.to(device=device, dtype=dtype)
        proposer_anchor_available = proposer_anchor_available.to(device=device)
        classical_rr_bpm = classical_rr_bpm.to(device=device, dtype=dtype)

        valid_candidate_rr = (
            torch.isfinite(candidate_rr_bpm)
            & (candidate_rr_bpm >= self.rr_min_bpm)
            & (candidate_rr_bpm <= self.rr_max_bpm)
        )
        if (candidate_mask & ~valid_candidate_rr).any():
            raise ValueError("available candidates require finite in-range RR")
        anchor_available = proposer_anchor_available & sequence_mask
        valid_anchor = (
            torch.isfinite(proposer_anchor_bpm)
            & torch.isfinite(proposer_anchor_std_bpm)
            & (proposer_anchor_bpm >= self.rr_min_bpm)
            & (proposer_anchor_bpm <= self.rr_max_bpm)
            & (proposer_anchor_std_bpm > 0.0)
        )
        if (anchor_available & ~valid_anchor).any():
            raise ValueError("available anchor requires finite in-range RR and positive std")
        classical_available = (
            sequence_mask
            & torch.isfinite(classical_rr_bpm)
            & (classical_rr_bpm >= self.rr_min_bpm)
            & (classical_rr_bpm <= self.rr_max_bpm)
        )
        radar_available = joint_radar_mask.any(dim=-1)
        node_mask = candidate_mask & valid_candidate_rr & sequence_mask[..., None]
        node_mask = node_mask & radar_available[..., None]
        candidate_rr_bpm = torch.where(
            node_mask, candidate_rr_bpm, torch.zeros_like(candidate_rr_bpm)
        )
        proposer_anchor_bpm = torch.where(
            anchor_available, proposer_anchor_bpm, torch.zeros_like(proposer_anchor_bpm)
        )
        proposer_anchor_std_bpm = torch.where(
            anchor_available,
            proposer_anchor_std_bpm,
            torch.ones_like(proposer_anchor_std_bpm),
        )
        classical_rr_bpm = torch.where(
            classical_available, classical_rr_bpm, torch.zeros_like(classical_rr_bpm)
        )
        ratios = node_features.new_tensor(RF_SVD_RATIOS)
        ratio_rr = candidate_rr_bpm.unsqueeze(-1) * ratios
        ratio_in_band = (
            node_mask[..., None]
            & (ratio_rr >= self.rr_min_bpm)
            & (ratio_rr <= self.rr_max_bpm)
        )
        return (
            node_features,
            candidate_rr_bpm,
            node_mask,
            sequence_mask,
            joint_radar_mask,
            proposer_anchor_bpm,
            proposer_anchor_std_bpm,
            anchor_available,
            classical_rr_bpm,
            classical_available,
            reset_mask,
            ratio_in_band,
        )

    def forward(
        self,
        node_features: Tensor,
        candidate_rr_bpm: Tensor,
        candidate_mask: Tensor,
        sequence_mask: Tensor,
        *,
        joint_radar_mask: Tensor,
        proposer_anchor_bpm: Tensor,
        proposer_anchor_std_bpm: Tensor,
        proposer_anchor_available: Tensor,
        classical_rr_bpm: Tensor,
        state: FactorRouterState | None = None,
        reset_mask: Tensor | None = None,
    ) -> dict[str, Tensor | FactorRouterState | Mapping[str, Tensor]]:
        (
            node_features,
            candidate_rr_bpm,
            candidate_mask,
            sequence_mask,
            joint_radar_mask,
            proposer_anchor_bpm,
            proposer_anchor_std_bpm,
            anchor_available,
            classical_rr_bpm,
            classical_available,
            reset_mask,
            ratio_in_band,
        ) = self._validate_inputs(
            node_features,
            candidate_rr_bpm,
            candidate_mask,
            sequence_mask,
            joint_radar_mask,
            proposer_anchor_bpm,
            proposer_anchor_std_bpm,
            proposer_anchor_available,
            classical_rr_bpm,
            reset_mask,
        )
        batch, windows, candidates, _ = node_features.shape
        dtype = node_features.dtype
        nodes, layout_diagnostics = self.structured_encoder(
            node_features,
            candidate_mask,
            joint_radar_mask,
            ratio_in_band,
        )
        relations = build_directed_harmonic_relations(
            candidate_rr_bpm,
            candidate_mask,
            near_tolerance_bpm=0.5,
            ratio_tolerance_bpm=0.75,
        )
        flat_nodes = nodes.reshape(batch * windows, candidates, self.hidden_channels)
        flat_relations = relations.reshape(batch * windows, candidates, candidates, 7)
        flat_mask = candidate_mask.reshape(batch * windows, candidates)
        graph_rates: list[Tensor] = []
        for block in self.graph:
            flat_nodes, spikes = block(
                flat_nodes,
                flat_relations,
                flat_mask,
                simulation_steps=self.simulation_steps,
            )
            denominator = (
                flat_mask.to(dtype).sum(dim=-1) * self.hidden_channels
            ).clamp_min(1.0)
            graph_rates.append(spikes.sum(dim=(-2, -1)) / denominator)
        nodes = flat_nodes.reshape(batch, windows, candidates, self.hidden_channels)

        attention_pool, attention_weights = self.candidate_pool(nodes, candidate_mask)
        count = candidate_mask.to(dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_pool = nodes.sum(dim=2) / count
        episode_input = self.episode_projection(
            torch.cat((attention_pool, mean_pool), dim=-1)
        )
        midpoint = 0.5 * (self.rr_min_bpm + self.rr_max_bpm)
        half_range = 0.5 * (self.rr_max_bpm - self.rr_min_bpm)
        source_context_values = torch.stack(
            (
                (proposer_anchor_bpm - midpoint) / half_range,
                torch.log1p(proposer_anchor_std_bpm).clamp(max=4.0) / 4.0,
                anchor_available.to(dtype),
                (classical_rr_bpm - midpoint) / half_range,
                classical_available.to(dtype),
            ),
            dim=-1,
        )
        source_context_values[..., :2] *= anchor_available[..., None].to(dtype)
        source_context_values[..., 3] *= classical_available.to(dtype)
        episode_input = episode_input + self.radar_context(joint_radar_mask.to(dtype))
        episode_input = episode_input + self.source_context(source_context_values)
        episode_input = episode_input * sequence_mask[..., None].to(dtype)
        temporal, temporal_rates, final_state = self.factor_temporal(
            episode_input, sequence_mask, reset_mask, state
        )

        expanded_temporal = temporal.unsqueeze(2).expand(-1, -1, candidates, -1)
        candidate_context = torch.cat((nodes, expanded_temporal), dim=-1)
        base_candidate_logits = self.candidate_logit_head(candidate_context).squeeze(-1)
        candidate_residual = self.candidate_residual_limit_bpm * torch.tanh(
            self.candidate_residual_head(candidate_context).squeeze(-1)
        )
        candidate_residual = candidate_residual * candidate_mask.to(dtype)
        candidate_mean = (candidate_rr_bpm + candidate_residual).clamp(
            self.rr_min_bpm, self.rr_max_bpm
        )
        candidate_mean = candidate_mean * candidate_mask.to(dtype)
        candidate_scale = self.candidate_minimum_scale_bpm + F.softplus(
            self.candidate_scale_head(candidate_context).squeeze(-1)
        )
        candidate_scale = candidate_scale.clamp(max=self.candidate_maximum_scale_bpm)

        factor_logits_raw = self.factor_head(temporal)
        factor_probability_raw = factor_logits_raw.softmax(dim=-1)
        factor_affinity = factor_candidate_affinity(
            candidate_rr_bpm,
            candidate_mask,
            classical_rr_bpm,
            classical_available,
            bandwidth_bpm=self.factor_affinity_bandwidth_bpm,
        )
        if self.factor_router_enabled:
            factor_probability = torch.where(
                classical_available[..., None],
                factor_probability_raw,
                torch.zeros_like(factor_probability_raw),
            )
            route_boost = self.factor_logit_boost * (
                factor_affinity * factor_probability.unsqueeze(-2)
            ).sum(dim=-1)
        else:
            factor_probability = torch.zeros_like(factor_probability_raw)
            route_boost = torch.zeros_like(base_candidate_logits)
        candidate_logits = base_candidate_logits + route_boost
        candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1.0e4)
        candidate_probability = _masked_softmax(
            candidate_logits, candidate_mask, dim=-1
        )

        anchor_residual = self.anchor_residual_limit_bpm * torch.tanh(
            self.anchor_residual_head(temporal).squeeze(-1)
        )
        anchor_residual = anchor_residual * anchor_available.to(dtype)
        corrected_anchor = (proposer_anchor_bpm + anchor_residual).clamp(
            self.rr_min_bpm, self.rr_max_bpm
        )
        corrected_anchor = torch.where(
            anchor_available, corrected_anchor, torch.zeros_like(corrected_anchor)
        )
        anchor_scale = self.anchor_minimum_scale_bpm + F.softplus(
            self.anchor_scale_head(temporal).squeeze(-1)
        )
        anchor_scale = anchor_scale.clamp(max=self.anchor_maximum_scale_bpm)
        anchor_logit = self.anchor_logit_head(temporal).squeeze(-1)
        anchor_logit = anchor_logit.masked_fill(~anchor_available, -1.0e4)

        expert_logits = torch.cat((anchor_logit.unsqueeze(-1), candidate_logits), dim=-1)
        expert_mask = torch.cat((anchor_available.unsqueeze(-1), candidate_mask), dim=-1)
        expert_logits = expert_logits.masked_fill(~expert_mask, -1.0e4)
        expert_probability = _masked_softmax(expert_logits, expert_mask, dim=-1)
        any_expert = expert_mask.any(dim=-1) & sequence_mask
        # torch.argmax returns the first maximum, which is the stable lower
        # expert index required by the contract.
        selected_expert_index = expert_logits.argmax(dim=-1)
        selected_expert_index = torch.where(
            any_expert,
            selected_expert_index,
            torch.full_like(selected_expert_index, -1),
        )
        expert_means = torch.cat((corrected_anchor.unsqueeze(-1), candidate_mean), dim=-1)
        expert_scales = torch.cat((anchor_scale.unsqueeze(-1), candidate_scale), dim=-1)
        gather_index = selected_expert_index.clamp_min(0).unsqueeze(-1)
        selected_expert_rr = expert_means.gather(-1, gather_index).squeeze(-1)
        selected_expert_scale = expert_scales.gather(-1, gather_index).squeeze(-1)
        selected_expert_probability = expert_probability.gather(
            -1, gather_index
        ).squeeze(-1)

        use_classical_fallback = ~any_expert & classical_available & sequence_mask
        source_available = any_expert | use_classical_fallback
        source_rr = torch.where(
            any_expert,
            selected_expert_rr,
            torch.where(
                use_classical_fallback,
                classical_rr_bpm,
                torch.full_like(classical_rr_bpm, float("nan")),
            ),
        )
        source_scale = torch.where(
            any_expert,
            selected_expert_scale,
            torch.where(
                use_classical_fallback,
                torch.full_like(classical_rr_bpm, self.anchor_maximum_scale_bpm),
                torch.full_like(classical_rr_bpm, float("nan")),
            ),
        )
        selected_expert_probability = torch.where(
            any_expert,
            selected_expert_probability,
            torch.zeros_like(selected_expert_probability),
        )
        source_code = torch.where(
            any_expert,
            selected_expert_index,
            torch.where(
                use_classical_fallback,
                torch.full_like(selected_expert_index, -2),
                torch.full_like(selected_expert_index, -1),
            ),
        )

        graph_spike_sequence = torch.stack(
            [rate.reshape(batch, windows) for rate in graph_rates], dim=-1
        )
        spike_sequence = torch.cat((graph_spike_sequence, temporal_rates), dim=-1)
        denominator = sequence_mask.to(dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        spike_rates = (
            spike_sequence * sequence_mask[..., None].to(dtype)
        ).sum(dim=1) / denominator
        quality_logit = self.quality_head(temporal).squeeze(-1)

        return {
            "candidate_base_logits": base_candidate_logits,
            "candidate_route_boost": route_boost,
            "candidate_logits": candidate_logits,
            "candidate_probabilities": candidate_probability,
            "candidate_residual_bpm": candidate_residual,
            "candidate_mean_bpm": candidate_mean,
            "candidate_scale_bpm": candidate_scale,
            "factor_logits": factor_logits_raw,
            "factor_probabilities": factor_probability,
            "factor_affinity": factor_affinity,
            "factor_supervision_mask": classical_available,
            "raw_anchor_rr_bpm": proposer_anchor_bpm,
            "raw_anchor_std_bpm": proposer_anchor_std_bpm,
            "anchor_available": anchor_available,
            "anchor_logit": anchor_logit,
            "anchor_residual_bpm": anchor_residual,
            "corrected_anchor_rr_bpm": corrected_anchor,
            "corrected_anchor_scale_bpm": anchor_scale,
            "expert_logits": expert_logits,
            "expert_probabilities": expert_probability,
            "selected_expert_index": selected_expert_index,
            "selected_source_code": source_code,
            "selected_probability": selected_expert_probability,
            "source_rr_bpm": source_rr,
            "source_scale_bpm": source_scale,
            "source_available": source_available,
            "quality_logit": quality_logit,
            "quality": quality_logit.sigmoid(),
            "node_embeddings": nodes,
            "directed_relations": relations,
            "candidate_attention": attention_weights,
            "temporal_state_sequence": temporal,
            "spike_sequence": spike_sequence,
            "spike_rates": spike_rates,
            "spike_rate": spike_rates.mean(),
            "layout_diagnostics": layout_diagnostics,
            "state": final_state,
        }


DHFER_SNN = DirectedHarmonicFactorExpertSNN


__all__ = [
    "DHFER_SNN",
    "DIRECTED_RELATIONS",
    "EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256",
    "FACTOR_CLASSES",
    "FEATURE_LAYOUT",
    "FEATURE_LAYOUT_SEMANTIC_SHA256",
    "FactorRouterState",
    "DirectedHarmonicFactorExpertSNN",
    "StructuredHarmonicEvidenceEncoder",
    "build_directed_harmonic_relations",
    "factor_candidate_affinity",
    "validate_feature_layout_binding",
]
