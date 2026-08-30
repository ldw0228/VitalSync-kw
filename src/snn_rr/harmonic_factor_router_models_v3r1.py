"""Hash-bound v3r1 wrapper around the quarantined DHFER-SNN ancestry.

The old v3 implementation is permitted only as read-only implementation
ancestry.  Its bytes and the adaptive v3r1 contract are verified *before* the
ancestry module is imported.  The wrapper then adds target-free structural
sanitisation so stale IQ or out-of-band cache values cannot reach the model.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Final

import torch
from torch import Tensor

from .harmonic_feature_layout_v3r1 import (
    EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
    FEATURE_LAYOUT,
    FEATURE_LAYOUT_SEMANTIC_SHA256,
    RF_SVD_RATIOS,
    TOTAL_FEATURE_WIDTH,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ANCESTRY_SOURCE_RELATIVE: Final[Path] = Path(
    "src/snn_rr/harmonic_factor_router_v3.py"
)
V3R1_CONTRACT_RELATIVE: Final[Path] = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
    "ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json"
)
EXPECTED_ANCESTRY_SOURCE_SHA256: Final[str] = (
    "1669399c3bb3925370e8b94a1c8bc79cc489b5cd55c6e21e7ec5ed191297edbc"
)
EXPECTED_V3R1_CONTRACT_FILE_SHA256: Final[str] = (
    "532d150f0241d9675873368107d09adec7aeaee5e018e09537e8a340eb6fa2bd"
)
EXPECTED_V3R1_CONTRACT_CONTENT_SHA256: Final[str] = (
    "6912e9760d1ab937604ba7868fe4742554804bd7179b5be2d6c8c5b34115aa2d"
)


class V3R1RuntimeBindingError(RuntimeError):
    """Raised before model use if a contract or ancestry binding has drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V3R1RuntimeBindingError(f"duplicate contract JSON key: {key}")
        result[key] = value
    return result


def verify_v3r1_runtime_bindings(
    *, project_root: str | Path = PROJECT_ROOT
) -> dict[str, object]:
    """Verify exact immutable bytes and contract semantics, fail closed."""

    root = Path(project_root).resolve()
    ancestry = root / ANCESTRY_SOURCE_RELATIVE
    contract_path = root / V3R1_CONTRACT_RELATIVE
    if not ancestry.is_file() or not contract_path.is_file():
        raise V3R1RuntimeBindingError("v3r1 runtime binding file is missing")
    ancestry_sha = _sha256(ancestry)
    if ancestry_sha != EXPECTED_ANCESTRY_SOURCE_SHA256:
        raise V3R1RuntimeBindingError("quarantined v3 ancestry source SHA-256 drifted")
    ancestry_mode = stat.S_IMODE(ancestry.stat().st_mode)
    if ancestry_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise V3R1RuntimeBindingError("quarantined v3 ancestry source must be read-only")
    contract_raw = contract_path.read_bytes()
    contract_sha = hashlib.sha256(contract_raw).hexdigest()
    if contract_sha != EXPECTED_V3R1_CONTRACT_FILE_SHA256:
        raise V3R1RuntimeBindingError("v3r1 campaign contract byte SHA-256 drifted")
    try:
        contract = json.loads(
            contract_raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V3R1RuntimeBindingError("v3r1 campaign contract is not strict JSON") from error
    if not isinstance(contract, dict):
        raise V3R1RuntimeBindingError("v3r1 campaign contract must be an object")
    embedded_content_sha = contract.get("content_sha256")
    payload = {key: value for key, value in contract.items() if key != "content_sha256"}
    computed_content_sha = _semantic_sha256(payload)
    if (
        embedded_content_sha != EXPECTED_V3R1_CONTRACT_CONTENT_SHA256
        or computed_content_sha != EXPECTED_V3R1_CONTRACT_CONTENT_SHA256
    ):
        raise V3R1RuntimeBindingError("v3r1 campaign contract semantic SHA-256 drifted")
    try:
        ancestry_binding = contract["immutable_inputs"]["inherited_model_design_source"]
        implementation = contract["implementation_authorization"]
        architecture = contract["architecture"]
    except (KeyError, TypeError) as error:
        raise V3R1RuntimeBindingError("v3r1 contract binding sections are missing") from error
    if not isinstance(ancestry_binding, Mapping) or (
        ancestry_binding.get("path") != ANCESTRY_SOURCE_RELATIVE.as_posix()
        or ancestry_binding.get("file_sha256") != EXPECTED_ANCESTRY_SOURCE_SHA256
        or ancestry_binding.get("old_contract_authority") is not False
    ):
        raise V3R1RuntimeBindingError("v3r1 ancestry contract binding drifted")
    if not isinstance(implementation, Mapping) or (
        implementation.get("authorized_now") is not True
        or implementation.get("training_authorized_now") is not False
        or ANCESTRY_SOURCE_RELATIVE.as_posix()
        not in implementation.get("existing_read_only_ancestry_import_allowed", [])
    ):
        raise V3R1RuntimeBindingError("v3r1 implementation authorization drifted")
    if not isinstance(architecture, Mapping) or (
        architecture.get("parameter_cap") != 400_000
        or architecture.get("hidden_channels") != 64
        or architecture.get("graph_blocks") != 2
        or architecture.get("simulation_steps") != 8
    ):
        raise V3R1RuntimeBindingError("v3r1 architecture contract drifted")
    return {
        "valid": True,
        "classification": "adaptive_retrospective_historical_cohort_engineering_not_confirmatory",
        "ancestry_source_path": ANCESTRY_SOURCE_RELATIVE.as_posix(),
        "ancestry_source_sha256": ancestry_sha,
        "ancestry_source_mode": format(ancestry_mode, "03o"),
        "contract_path": V3R1_CONTRACT_RELATIVE.as_posix(),
        "contract_file_sha256": contract_sha,
        "contract_content_sha256": computed_content_sha,
        "commercial_claim_allowed": False,
    }


# Verify the quarantined file before Python executes/imports any of its code.
RUNTIME_BINDING_RECEIPT: Final[dict[str, object]] = verify_v3r1_runtime_bindings()

from .harmonic_factor_router_v3 import (  # noqa: E402  (verification must precede import)
    DIRECTED_RELATIONS,
    FACTOR_CLASSES,
    FactorRouterState,
    StructuredHarmonicEvidenceEncoder,
    DirectedHarmonicFactorExpertSNN as _AncestryDHFER,
    build_directed_harmonic_relations,
    factor_candidate_affinity,
    validate_feature_layout_binding,
)


def _build_structural_availability_mask_torch(
    candidate_rr_bpm: Tensor,
    candidate_mask: Tensor,
    joint_radar_mask: Tensor,
    *,
    rr_min_bpm: float,
    rr_max_bpm: float,
) -> Tensor:
    if candidate_rr_bpm.ndim < 1 or candidate_rr_bpm.shape != candidate_mask.shape:
        raise ValueError("candidate_rr_bpm and candidate_mask must share shape [...,K]")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must have boolean dtype")
    if joint_radar_mask.dtype != torch.bool or joint_radar_mask.shape != (
        *candidate_rr_bpm.shape[:-1],
        3,
    ):
        raise ValueError("joint_radar_mask has invalid shape or dtype")
    rr = (
        candidate_rr_bpm
        if torch.is_floating_point(candidate_rr_bpm)
        else candidate_rr_bpm.float()
    )
    node_available = candidate_mask & joint_radar_mask.any(dim=-1).unsqueeze(-1)
    ratios = rr.new_tensor(RF_SVD_RATIOS)
    ratio_rr = rr.unsqueeze(-1) * ratios
    ratio_available = (
        node_available.unsqueeze(-1)
        & (ratio_rr >= float(rr_min_bpm))
        & (ratio_rr <= float(rr_max_bpm))
    )
    cell = (
        node_available[..., None, None]
        & joint_radar_mask[..., None, :, None]
        & ratio_available[..., None, :]
    )
    core = node_available[..., None].expand(*node_available.shape, 46)
    rf = torch.stack((cell, torch.zeros_like(cell)), dim=-1)
    rf = rf[..., None].expand(*rf.shape, 9).reshape(*node_available.shape, 378)
    svd = cell[..., None].expand(*cell.shape, 7).reshape(*node_available.shape, 147)
    return torch.cat((core, rf, svd), dim=-1)


class DirectedHarmonicFactorExpertSNNV3R1(_AncestryDHFER):
    """Adaptive v3r1 model with immutable ancestry and input sanitisation."""

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
        self.runtime_binding_receipt = verify_v3r1_runtime_bindings()
        super().__init__(
            ordered_feature_names_semantic_sha256=ordered_feature_names_semantic_sha256,
            structural_layout_semantic_sha256=structural_layout_semantic_sha256,
            variant=variant,
            hidden_channels=hidden_channels,
            graph_blocks=graph_blocks,
            simulation_steps=simulation_steps,
            dropout=dropout,
            beta=beta,
            adaptation_decay=adaptation_decay,
            adaptation_strength=adaptation_strength,
            rr_min_bpm=rr_min_bpm,
            rr_max_bpm=rr_max_bpm,
            candidate_residual_limit_bpm=candidate_residual_limit_bpm,
            candidate_minimum_scale_bpm=candidate_minimum_scale_bpm,
            candidate_initial_scale_bpm=candidate_initial_scale_bpm,
            candidate_maximum_scale_bpm=candidate_maximum_scale_bpm,
            anchor_residual_limit_bpm=anchor_residual_limit_bpm,
            anchor_minimum_scale_bpm=anchor_minimum_scale_bpm,
            anchor_initial_scale_bpm=anchor_initial_scale_bpm,
            anchor_maximum_scale_bpm=anchor_maximum_scale_bpm,
            factor_logit_boost=factor_logit_boost,
            factor_affinity_bandwidth_bpm=factor_affinity_bandwidth_bpm,
            maximum_parameters=maximum_parameters,
        )
        if self.parameter_count() > 400_000:
            raise RuntimeError("v3r1 model exceeded the immutable 400,000 parameter cap")
        self.assert_safe_initialization()

    def layout_receipt(self) -> dict[str, str | int]:
        receipt = dict(super().layout_receipt())
        receipt.update(
            {
                "v3r1_contract_file_sha256": EXPECTED_V3R1_CONTRACT_FILE_SHA256,
                "v3r1_contract_content_sha256": EXPECTED_V3R1_CONTRACT_CONTENT_SHA256,
                "quarantined_ancestry_source_sha256": EXPECTED_ANCESTRY_SOURCE_SHA256,
            }
        )
        return receipt

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
        if node_features.ndim != 4 or node_features.shape[-1] != TOTAL_FEATURE_WIDTH:
            raise ValueError("node_features must have shape [batch,time,K,571]")
        if candidate_mask.shape != node_features.shape[:3]:
            raise ValueError("candidate_mask must have shape [batch,time,K]")
        if candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must have boolean dtype")
        if sequence_mask.shape != node_features.shape[:2] or sequence_mask.dtype != torch.bool:
            raise ValueError("sequence_mask must be boolean [batch,time]")
        if (
            joint_radar_mask.shape != (*node_features.shape[:2], 3)
            or joint_radar_mask.dtype != torch.bool
        ):
            raise ValueError("joint_radar_mask must be boolean [batch,time,3]")
        mask_device = node_features.device
        mask_sequence = sequence_mask.to(device=mask_device)
        effective_candidate_mask = (
            candidate_mask.to(device=mask_device) & mask_sequence[..., None]
        )
        effective_radar_mask = (
            joint_radar_mask.to(device=mask_device) & mask_sequence[..., None]
        )
        structural_mask = _build_structural_availability_mask_torch(
            candidate_rr_bpm.to(device=mask_device),
            effective_candidate_mask,
            effective_radar_mask,
            rr_min_bpm=self.rr_min_bpm,
            rr_max_bpm=self.rr_max_bpm,
        )
        if not torch.is_floating_point(node_features):
            node_features = node_features.float()
        available_values = node_features.masked_select(structural_mask)
        if available_values.numel() and not torch.isfinite(available_values).all():
            raise ValueError("structurally available node features must be finite")
        sanitized_features = torch.where(
            structural_mask, node_features, torch.zeros_like(node_features)
        )
        return super().forward(
            sanitized_features,
            candidate_rr_bpm,
            candidate_mask,
            sequence_mask,
            joint_radar_mask=joint_radar_mask,
            proposer_anchor_bpm=proposer_anchor_bpm,
            proposer_anchor_std_bpm=proposer_anchor_std_bpm,
            proposer_anchor_available=proposer_anchor_available,
            classical_rr_bpm=classical_rr_bpm,
            state=state,
            reset_mask=reset_mask,
        )


AncestryDirectedHarmonicFactorExpertSNN = _AncestryDHFER
DirectedHarmonicFactorExpertSNN = DirectedHarmonicFactorExpertSNNV3R1
DHFER_SNN_V3R1 = DirectedHarmonicFactorExpertSNNV3R1
DHFER_SNN = DirectedHarmonicFactorExpertSNNV3R1


__all__ = [
    "ANCESTRY_SOURCE_RELATIVE",
    "AncestryDirectedHarmonicFactorExpertSNN",
    "DHFER_SNN",
    "DHFER_SNN_V3R1",
    "DIRECTED_RELATIONS",
    "DirectedHarmonicFactorExpertSNN",
    "DirectedHarmonicFactorExpertSNNV3R1",
    "EXPECTED_ANCESTRY_SOURCE_SHA256",
    "EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256",
    "EXPECTED_V3R1_CONTRACT_CONTENT_SHA256",
    "EXPECTED_V3R1_CONTRACT_FILE_SHA256",
    "FACTOR_CLASSES",
    "FEATURE_LAYOUT",
    "FEATURE_LAYOUT_SEMANTIC_SHA256",
    "FactorRouterState",
    "RUNTIME_BINDING_RECEIPT",
    "StructuredHarmonicEvidenceEncoder",
    "V3R1RuntimeBindingError",
    "V3R1_CONTRACT_RELATIVE",
    "build_directed_harmonic_relations",
    "factor_candidate_affinity",
    "validate_feature_layout_binding",
    "verify_v3r1_runtime_bindings",
]
