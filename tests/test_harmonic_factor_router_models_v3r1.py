from __future__ import annotations

import inspect
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from snn_rr.harmonic_feature_layout_v3r1 import (
    EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
)
from snn_rr.harmonic_factor_router_models_v3r1 import (
    ANCESTRY_SOURCE_RELATIVE,
    DIRECTED_RELATIONS,
    EXPECTED_ANCESTRY_SOURCE_SHA256,
    EXPECTED_V3R1_CONTRACT_FILE_SHA256,
    RUNTIME_BINDING_RECEIPT,
    V3R1_CONTRACT_RELATIVE,
    V3R1RuntimeBindingError,
    DirectedHarmonicFactorExpertSNNV3R1,
    build_directed_harmonic_relations,
    verify_v3r1_runtime_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_CACHE = ROOT / (
    "artifacts/cache/"
    "harmonic_set_v2_i2r_nested_o3_s20260828_nms125_base_emap_svd12_m050"
)


def _model(*, variant: str = "H2_full") -> DirectedHarmonicFactorExpertSNNV3R1:
    torch.manual_seed(7)
    model = DirectedHarmonicFactorExpertSNNV3R1(
        ordered_feature_names_semantic_sha256=(
            EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
        ),
        variant=variant,
        dropout=0.0,
    )
    model.eval()
    return model


def _inputs(time: int = 4) -> dict[str, torch.Tensor]:
    candidate_rr = torch.tensor(
        [[[10.0, 20.0, 30.0, 0.0]]], dtype=torch.float32
    ).expand(1, time, 4).clone()
    candidate_mask = torch.tensor([[[True, True, True, False]]]).expand(
        1, time, 4
    ).clone()
    sequence_mask = torch.ones((1, time), dtype=torch.bool)
    radar_mask = torch.tensor([[[True, False, True]]]).expand(1, time, 3).clone()
    generator = torch.Generator().manual_seed(10)
    # Deliberately contaminate every structurally unavailable location.  The
    # v3r1 wrapper must sanitize it before the strict ancestry encoder runs.
    features = torch.randn((1, time, 4, 571), generator=generator)
    features[..., 46:424].reshape(1, time, 4, 3, 7, 2, 9)[..., 1, :] = 9_999.0
    return {
        "node_features": features,
        "candidate_rr_bpm": candidate_rr,
        "candidate_mask": candidate_mask,
        "sequence_mask": sequence_mask,
        "joint_radar_mask": radar_mask,
        "proposer_anchor_bpm": torch.full((1, time), 12.0),
        "proposer_anchor_std_bpm": torch.full((1, time), 1.0),
        "proposer_anchor_available": torch.ones((1, time), dtype=torch.bool),
        "classical_rr_bpm": torch.full((1, time), 10.0),
        "reset_mask": torch.tensor([[True] + [False] * (time - 1)]),
    }


def _forward(
    model: DirectedHarmonicFactorExpertSNNV3R1,
    values: dict[str, torch.Tensor],
    *,
    state=None,
) -> dict:
    return model(
        values["node_features"],
        values["candidate_rr_bpm"],
        values["candidate_mask"],
        values["sequence_mask"],
        joint_radar_mask=values["joint_radar_mask"],
        proposer_anchor_bpm=values["proposer_anchor_bpm"],
        proposer_anchor_std_bpm=values["proposer_anchor_std_bpm"],
        proposer_anchor_available=values["proposer_anchor_available"],
        classical_rr_bpm=values["classical_rr_bpm"],
        state=state,
        reset_mask=values["reset_mask"],
    )


def _slice(values: dict[str, torch.Tensor], begin: int, end: int) -> dict[str, torch.Tensor]:
    return {key: value[:, begin:end] for key, value in values.items()}


def test_runtime_verifies_read_only_ancestry_and_adaptive_contract() -> None:
    receipt = verify_v3r1_runtime_bindings()
    assert receipt == RUNTIME_BINDING_RECEIPT
    assert receipt["ancestry_source_sha256"] == EXPECTED_ANCESTRY_SOURCE_SHA256
    assert receipt["contract_file_sha256"] == EXPECTED_V3R1_CONTRACT_FILE_SHA256
    assert receipt["ancestry_source_mode"] == "444"
    assert receipt["commercial_claim_allowed"] is False


def test_runtime_binding_fails_closed_on_ancestry_or_mode_drift(tmp_path: Path) -> None:
    ancestry = tmp_path / ANCESTRY_SOURCE_RELATIVE
    contract = tmp_path / V3R1_CONTRACT_RELATIVE
    ancestry.parent.mkdir(parents=True)
    contract.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / ANCESTRY_SOURCE_RELATIVE, ancestry)
    shutil.copyfile(ROOT / V3R1_CONTRACT_RELATIVE, contract)
    ancestry.chmod(0o444)
    assert verify_v3r1_runtime_bindings(project_root=tmp_path)["valid"] is True
    ancestry.chmod(0o644)
    with pytest.raises(V3R1RuntimeBindingError, match="read-only"):
        verify_v3r1_runtime_bindings(project_root=tmp_path)
    ancestry.chmod(0o444)
    contract.write_bytes(contract.read_bytes() + b"\n")
    with pytest.raises(V3R1RuntimeBindingError, match="contract byte"):
        verify_v3r1_runtime_bindings(project_root=tmp_path)


def test_model_parameter_cap_and_safe_initialization_are_enforced() -> None:
    model = _model()
    assert model.parameter_count() == 203_669
    assert model.parameter_count() <= 400_000
    model.assert_safe_initialization()
    receipt = model.layout_receipt()
    assert receipt["v3r1_contract_file_sha256"] == EXPECTED_V3R1_CONTRACT_FILE_SHA256
    assert receipt["quarantined_ancestry_source_sha256"] == EXPECTED_ANCESTRY_SOURCE_SHA256
    with pytest.raises(ValueError, match="parameter cap"):
        DirectedHarmonicFactorExpertSNNV3R1(
            ordered_feature_names_semantic_sha256=EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
            maximum_parameters=100,
        )


@pytest.mark.parametrize("variant", ["H0_no_factor", "H1_factor", "H2_full"])
def test_synthetic_contaminated_structural_cells_are_sanitized_before_forward(
    variant: str,
) -> None:
    model = _model(variant=variant)
    values = _inputs(time=2)
    # Masked padding may even be non-finite; available cells remain finite.
    values["node_features"][..., 3, :] = torch.nan
    output = _forward(model, values)
    assert output["source_rr_bpm"].shape == (1, 2)
    assert torch.isfinite(output["source_rr_bpm"]).all()
    assert torch.isfinite(output["candidate_logits"][..., :3]).all()
    rf_mask = output["layout_diagnostics"]["rf_cell_mask"]
    assert not rf_mask[..., 1].any()
    assert output["selected_expert_index"].min() >= 0


def test_available_nonfinite_feature_fails_closed() -> None:
    model = _model()
    values = _inputs(time=1)
    values["node_features"][0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="structurally available"):
        _forward(model, values)


def test_streaming_state_matches_whole_session_evaluation() -> None:
    model = _model()
    values = _inputs(time=5)
    with torch.no_grad():
        whole = _forward(model, values)
        first = _forward(model, _slice(values, 0, 2))
        second_values = _slice(values, 2, 5)
        # Reset occurred at the physical session start, not at chunk start.
        second_values["reset_mask"] = torch.zeros_like(second_values["reset_mask"])
        second = _forward(model, second_values, state=first["state"])
    for name in (
        "source_rr_bpm",
        "source_scale_bpm",
        "candidate_logits",
        "factor_logits",
        "expert_logits",
        "quality",
        "spike_sequence",
        "temporal_state_sequence",
    ):
        streamed = torch.cat((first[name], second[name]), dim=1)
        torch.testing.assert_close(streamed, whole[name], rtol=1.0e-6, atol=1.0e-6)
    for streamed_layer, whole_layer in zip(second["state"], whole["state"], strict=True):
        for streamed_tensor, whole_tensor in zip(streamed_layer, whole_layer, strict=True):
            torch.testing.assert_close(
                streamed_tensor, whole_tensor, rtol=2.0e-5, atol=1.0e-5
            )


def test_directed_relation_channels_are_asymmetric() -> None:
    rr = torch.tensor([[10.0, 20.0, 30.0]])
    mask = torch.ones_like(rr, dtype=torch.bool)
    relation = build_directed_harmonic_relations(rr, mask)
    receiver_is_2x = DIRECTED_RELATIONS.index("receiver_is_2x_sender")
    sender_is_2x = DIRECTED_RELATIONS.index("sender_is_2x_receiver")
    assert relation[0, 1, 0, receiver_is_2x]
    assert not relation[0, 0, 1, receiver_is_2x]
    assert relation[0, 0, 1, sender_is_2x]
    assert not relation[0, 1, 0, sender_is_2x]


def test_forward_interface_and_outputs_have_no_target_or_identity_context() -> None:
    parameters = set(inspect.signature(DirectedHarmonicFactorExpertSNNV3R1.forward).parameters)
    forbidden = {
        "reference_rr_bpm",
        "reference_validity",
        "reference_quality",
        "identity",
        "protocol",
        "future_windows",
    }
    assert not parameters & forbidden
    output = _forward(_model(), _inputs(time=1))
    assert not set(output) & forbidden


def test_real_cache_sample_with_nonzero_out_of_band_raw_cells_forwards() -> None:
    node = np.array(np.load(REAL_CACHE / "node_features.npy", mmap_mode="r")[:2])
    rr = np.array(np.load(REAL_CACHE / "candidate_bpm.npy", mmap_mode="r")[:2])
    candidate_mask = np.array(
        np.load(REAL_CACHE / "candidate_mask.npy", mmap_mode="r")[:2]
    )
    radar_mask = np.array(
        np.load(REAL_CACHE / "joint_radar_mask.npy", mmap_mode="r")[:2]
    )
    # Candidate 11.75 has cached raw r4 support although 47 bpm is outside the
    # model band.  This would make the strict ancestry reject the raw cache.
    assert np.count_nonzero(node[..., 46:424]) > 0
    values = {
        "node_features": torch.from_numpy(node).unsqueeze(0),
        "candidate_rr_bpm": torch.from_numpy(rr).unsqueeze(0),
        "candidate_mask": torch.from_numpy(candidate_mask).unsqueeze(0),
        "sequence_mask": torch.ones((1, 2), dtype=torch.bool),
        "joint_radar_mask": torch.from_numpy(radar_mask).unsqueeze(0),
        "proposer_anchor_bpm": torch.from_numpy(rr[:, 0]).unsqueeze(0),
        "proposer_anchor_std_bpm": torch.ones((1, 2)),
        "proposer_anchor_available": torch.ones((1, 2), dtype=torch.bool),
        "classical_rr_bpm": torch.from_numpy(rr[:, 0]).unsqueeze(0),
        "reset_mask": torch.tensor([[True, False]]),
    }
    with torch.no_grad():
        output = _forward(_model(), values)
    assert torch.isfinite(output["source_rr_bpm"]).all()
    assert torch.isfinite(output["spike_rate"])
