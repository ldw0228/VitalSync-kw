from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from snn_rr.harmonic_set_models import (
    HarmonicCandidateSetEpisodeSNN,
    build_harmonic_edge_types,
)
from snn_rr.svd_episode_models import EpisodeSpikingCell


FEATURES = 9
HIDDEN = 16


def _model() -> HarmonicCandidateSetEpisodeSNN:
    torch.manual_seed(20260828)
    return HarmonicCandidateSetEpisodeSNN(
        node_features=FEATURES,
        hidden_channels=HIDDEN,
        num_radars=3,
        dropout=0.0,
    )


def _inputs(
    *, batch: int = 2, time_steps: int = 7, candidates: int = 6
) -> dict[str, torch.Tensor]:
    torch.manual_seed(1701)
    centers = torch.tensor(
        [8.0, 8.5, 16.0, 24.0, 32.0, 40.0, 12.0, 36.0, 18.0, 27.0, 42.0, 10.0]
    )[:candidates]
    rr = centers.reshape(1, 1, candidates).expand(batch, time_steps, -1).clone()
    rr = rr + 0.02 * torch.arange(time_steps).reshape(1, -1, 1)
    return {
        "node_features": torch.randn(batch, time_steps, candidates, FEATURES),
        "candidate_rr": rr,
        "candidate_mask": torch.ones(
            batch, time_steps, candidates, dtype=torch.bool
        ),
        "sequence_mask": torch.ones(batch, time_steps, dtype=torch.bool),
        "radar_mask": torch.ones(batch, time_steps, 3, dtype=torch.bool),
        "reset_mask": torch.zeros(batch, time_steps, dtype=torch.bool),
    }


def _slice_time(
    inputs: dict[str, torch.Tensor], start: int, stop: int
) -> dict[str, torch.Tensor]:
    return {name: value[:, start:stop].clone() for name, value in inputs.items()}


def _assert_state_close(
    first: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    second: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> None:
    assert len(first) == len(second) == 2
    for first_layer, second_layer in zip(first, second, strict=True):
        for first_tensor, second_tensor in zip(
            first_layer, second_layer, strict=True
        ):
            torch.testing.assert_close(first_tensor, second_tensor, atol=2e-6, rtol=2e-6)


def test_harmonic_edge_builder_marks_near_and_x2_x3_x4_without_padding() -> None:
    rr = torch.tensor([[8.0, 8.5, 16.0, 24.0, 32.0, 40.0]])
    mask = torch.tensor([[True, True, True, True, True, False]])
    edges = build_harmonic_edge_types(rr, mask)
    assert edges.shape == (1, 6, 6, 4)
    assert edges[0, 0, 1, 0] and edges[0, 1, 0, 0]
    assert edges[0, 0, 2, 1] and edges[0, 2, 0, 1]
    assert edges[0, 0, 3, 2] and edges[0, 3, 0, 2]
    assert edges[0, 0, 4, 3] and edges[0, 4, 0, 3]
    assert not torch.diagonal(edges.any(dim=-1), dim1=-2, dim2=-1).any()
    assert not edges[0, 5].any()
    assert not edges[0, :, 5].any()


def test_episode_spiking_cell_supports_batched_candidate_mask() -> None:
    torch.manual_seed(11)
    cell = EpisodeSpikingCell(5, cell_type="alif", beta=0.9)
    current = torch.randn(2, 4, 5, requires_grad=True)
    update_mask = torch.tensor(
        [[True, False, True, False], [False, True, True, False]]
    )
    state = cell.initial_state(torch.zeros_like(current))
    spikes, (membrane, adaptation) = cell.forward_step(
        current, state, update_mask
    )
    assert spikes.shape == membrane.shape == adaptation.shape == (2, 4, 5)
    assert torch.count_nonzero(spikes[~update_mask]) == 0
    assert torch.count_nonzero(membrane[~update_mask]) == 0
    assert torch.count_nonzero(adaptation[~update_mask]) == 0
    (spikes.sum() + membrane.square().sum() + adaptation.sum()).backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert float(current.grad[update_mask].abs().sum()) > 0.0


def test_forward_contract_masks_listwise_outputs_and_decodes_argmax() -> None:
    model = _model().eval()
    inputs = _inputs(batch=2, time_steps=5, candidates=7)
    inputs["candidate_mask"][0, 1, 4:] = False
    with torch.no_grad():
        output = model(**inputs)

    required_keys = {
        "candidate_logits",
        "candidate_probabilities",
        "candidate_residual_bpm",
        "candidate_scale_bpm",
        "factor_logits",
        "quality_logit",
        "quality",
        "selected_index",
        "selected_probability",
        "source_rr",
        "source_scale_bpm",
        "source_available",
        "node_embeddings",
        "candidate_attention",
        "state_sequence",
        "spike_sequence",
        "spike_rates",
        "state",
    }
    assert required_keys <= output.keys()
    assert output["candidate_logits"].shape == (2, 5, 7)
    assert output["factor_logits"].shape == (2, 5, 4)
    assert output["state_sequence"].shape == (2, 5, HIDDEN)
    assert output["spike_sequence"].shape == (2, 5, 4)
    torch.testing.assert_close(
        output["candidate_probabilities"].sum(dim=-1),
        torch.ones(2, 5),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.count_nonzero(output["candidate_probabilities"][0, 1, 4:]) == 0
    assert torch.all(
        output["candidate_residual_bpm"].abs()
        <= model.max_residual_bpm + 1e-7
    )
    assert torch.all(output["candidate_scale_bpm"] > 0)
    assert torch.all(output["source_scale_bpm"] > 0)
    selected = output["candidate_logits"].argmax(dim=-1)
    torch.testing.assert_close(output["selected_index"], selected)
    gathered = (
        inputs["candidate_rr"]
        + output["candidate_residual_bpm"]
    ).gather(-1, selected.unsqueeze(-1)).squeeze(-1).clamp(6.0, 45.0)
    torch.testing.assert_close(output["source_rr"], gathered)
    assert sum(parameter.numel() for parameter in model.parameters()) < 750_000

    signature = inspect.signature(model.forward)
    assert set(signature.parameters) == {
        "node_features",
        "candidate_rr",
        "candidate_mask",
        "sequence_mask",
        "radar_mask",
        "state",
        "reset_mask",
        "anchor_rr",
        "anchor_std",
        "anchor_available",
    }
    for forbidden in ("reference", "quality_control", "base_error", "target"):
        assert forbidden not in signature.parameters


def test_anchor_disabled_state_dict_is_legacy_compatible_and_rejects_context() -> None:
    legacy = _model().eval()
    restored = _model().eval()
    restored.load_state_dict(legacy.state_dict(), strict=True)
    assert set(legacy.state_dict()) == set(restored.state_dict())
    assert not any("anchor" in name for name in legacy.state_dict())
    inputs = _inputs(batch=1, time_steps=3, candidates=4)
    with pytest.raises(ValueError, match="anchor_enabled"):
        legacy(
            **inputs,
            anchor_rr=torch.full((1, 3), 12.0),
            anchor_std=torch.ones(1, 3),
            anchor_available=torch.ones(1, 3, dtype=torch.bool),
        )


def test_i3_initial_source_is_exact_raw_anchor_and_snap_is_fully_off() -> None:
    torch.manual_seed(20260828)
    model = HarmonicCandidateSetEpisodeSNN(
        node_features=FEATURES,
        hidden_channels=HIDDEN,
        num_radars=3,
        dropout=0.0,
        anchor_enabled=True,
    ).eval()
    inputs = _inputs(batch=2, time_steps=6, candidates=6)
    anchor = torch.tensor(
        [[10.25, 11.5, 12.75, 14.0, 15.25, 16.5],
         [18.25, 19.5, 20.75, 22.0, 23.25, 24.5]],
        dtype=torch.float32,
    )
    available = torch.ones_like(anchor, dtype=torch.bool)
    with torch.no_grad():
        output = model(
            **inputs,
            anchor_rr=anchor,
            anchor_std=torch.full_like(anchor, 1.25),
            anchor_available=available,
        )
    assert torch.equal(output["source_rr"], anchor)
    assert torch.equal(output["corrected_anchor_rr"], anchor)
    assert torch.count_nonzero(output["anchor_residual_bpm"]) == 0
    assert torch.count_nonzero(output["anchor_snap_gate"]) == 0


def test_i3_no_anchor_is_bit_exact_original_candidate_path() -> None:
    disabled = _model().eval()
    enabled = HarmonicCandidateSetEpisodeSNN(
        node_features=FEATURES,
        hidden_channels=HIDDEN,
        num_radars=3,
        dropout=0.0,
        anchor_enabled=True,
    ).eval()
    incompatible = enabled.load_state_dict(disabled.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys and all(
        name.startswith("anchor_") for name in incompatible.missing_keys
    )
    inputs = _inputs(batch=2, time_steps=6, candidates=6)
    with torch.no_grad():
        original = disabled(**inputs)
        no_anchor = enabled(**inputs)
    for key in (
        "candidate_logits", "candidate_probabilities",
        "candidate_residual_bpm", "candidate_scale_bpm", "factor_logits",
        "quality_logit", "source_rr", "source_scale_bpm",
        "source_available", "selected_index", "state_sequence",
        "spike_sequence", "spike_rates",
    ):
        assert torch.equal(original[key], no_anchor[key]), key
    _assert_state_close(original["state"], no_anchor["state"])


def test_anchor_context_validation_is_strict() -> None:
    model = HarmonicCandidateSetEpisodeSNN(
        node_features=FEATURES,
        hidden_channels=HIDDEN,
        dropout=0.0,
        anchor_enabled=True,
    )
    inputs = _inputs(batch=1, time_steps=4, candidates=4)
    with pytest.raises(ValueError, match="supplied together"):
        model(**inputs, anchor_rr=torch.full((1, 4), 12.0))
    with pytest.raises(ValueError, match="shape"):
        model(
            **inputs,
            anchor_rr=torch.full((1, 3), 12.0),
            anchor_std=torch.ones(1, 4),
            anchor_available=torch.ones(1, 4, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="boolean"):
        model(
            **inputs,
            anchor_rr=torch.full((1, 4), 12.0),
            anchor_std=torch.ones(1, 4),
            anchor_available=torch.ones(1, 4),
        )
    bad = torch.full((1, 4), 12.0)
    bad[0, 1] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        model(
            **inputs,
            anchor_rr=bad,
            anchor_std=torch.ones(1, 4),
            anchor_available=torch.zeros(1, 4, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="positive"):
        model(
            **inputs,
            anchor_rr=torch.full((1, 4), 12.0),
            anchor_std=torch.zeros(1, 4),
            anchor_available=torch.ones(1, 4, dtype=torch.bool),
        )


def test_padded_candidate_nodes_cannot_change_real_node_or_source_outputs() -> None:
    model = _model().eval()
    compact = _inputs(batch=1, time_steps=6, candidates=5)
    padded = _inputs(batch=1, time_steps=6, candidates=9)
    for name in ("node_features", "candidate_rr", "candidate_mask"):
        padded[name][:, :, :5] = compact[name]
    padded["node_features"][:, :, 5:] = 1.0e20
    padded["node_features"][:, 2:, 7:] = torch.nan
    padded["candidate_rr"][:, :, 5:] = torch.tensor([7.0, 20.0, 45.0, 15.0])
    padded["candidate_rr"][:, 4:, 8] = torch.nan
    padded["candidate_mask"][:, :, 5:] = False
    with torch.no_grad():
        compact_output = model(**compact)
        padded_output = model(**padded)
    for key in (
        "candidate_logits",
        "candidate_probabilities",
        "candidate_residual_bpm",
        "candidate_scale_bpm",
        "node_embeddings",
    ):
        torch.testing.assert_close(
            compact_output[key],
            padded_output[key][:, :, :5],
            atol=2e-6,
            rtol=2e-6,
        )
    for key in (
        "factor_logits",
        "quality_logit",
        "source_rr",
        "source_scale_bpm",
        "selected_probability",
        "state_sequence",
        "spike_sequence",
    ):
        torch.testing.assert_close(
            compact_output[key], padded_output[key], atol=2e-6, rtol=2e-6
        )
    torch.testing.assert_close(
        compact_output["selected_index"], padded_output["selected_index"]
    )
    _assert_state_close(compact_output["state"], padded_output["state"])


def test_future_changes_do_not_change_causal_prefix() -> None:
    model = _model().eval()
    inputs = _inputs(batch=1, time_steps=9, candidates=6)
    changed = {name: value.clone() for name, value in inputs.items()}
    changed["node_features"][:, 5:] = -500.0
    changed["candidate_rr"][:, 5:] = torch.tensor(
        [10.0, 20.0, 30.0, 40.0, 44.0, 12.0]
    )
    changed["candidate_mask"][:, 6:, 1::2] = False
    changed["radar_mask"][:, 7:, :2] = False
    changed["reset_mask"][:, 8] = True
    with torch.no_grad():
        original = model(**inputs)
        future_changed = model(**changed)
    for key in (
        "candidate_logits",
        "candidate_probabilities",
        "source_rr",
        "quality_logit",
        "state_sequence",
        "spike_sequence",
    ):
        torch.testing.assert_close(
            original[key][:, :5], future_changed[key][:, :5], rtol=0, atol=0
        )


def test_explicit_state_makes_chunked_inference_match_full_stream() -> None:
    model = _model().eval()
    inputs = _inputs(batch=2, time_steps=10, candidates=6)
    inputs["candidate_mask"][1, 4, 3:] = False
    inputs["radar_mask"][0, 6, 2] = False
    inputs["reset_mask"][1, 7] = True
    with torch.no_grad():
        full = model(**inputs)
        first = model(**_slice_time(inputs, 0, 4))
        second = model(**_slice_time(inputs, 4, 10), state=first["state"])
    for key in (
        "candidate_logits",
        "candidate_probabilities",
        "candidate_residual_bpm",
        "candidate_scale_bpm",
        "factor_logits",
        "quality_logit",
        "source_rr",
        "source_scale_bpm",
        "state_sequence",
        "spike_sequence",
    ):
        chunked = torch.cat((first[key], second[key]), dim=1)
        torch.testing.assert_close(full[key], chunked, atol=2e-6, rtol=2e-6)
    _assert_state_close(full["state"], second["state"])


def test_i3_anchor_one_step_and_chunked_stream_are_identical() -> None:
    model = HarmonicCandidateSetEpisodeSNN(
        node_features=FEATURES,
        hidden_channels=HIDDEN,
        dropout=0.0,
        anchor_enabled=True,
    ).eval()
    inputs = _inputs(batch=1, time_steps=8, candidates=5)
    inputs.update(
        anchor_rr=torch.linspace(11.0, 18.0, 8).reshape(1, -1),
        anchor_std=torch.linspace(0.8, 1.5, 8).reshape(1, -1),
        anchor_available=torch.tensor(
            [[True, True, False, True, True, True, False, True]]
        ),
    )
    with torch.no_grad():
        full = model(**inputs)
        state = None
        steps: list[dict[str, object]] = []
        for index in range(8):
            step = model(**_slice_time(inputs, index, index + 1), state=state)
            state = step["state"]
            steps.append(step)
    for key in (
        "candidate_logits", "candidate_probabilities", "source_rr",
        "source_scale_bpm", "corrected_anchor_rr", "anchor_residual_bpm",
        "anchor_snap_gate", "state_sequence", "spike_sequence",
    ):
        one_step = torch.cat([step[key] for step in steps], dim=1)
        torch.testing.assert_close(full[key], one_step, atol=2e-6, rtol=2e-6)
    _assert_state_close(full["state"], state)


def test_future_anchor_changes_cannot_change_i3_prefix() -> None:
    model = HarmonicCandidateSetEpisodeSNN(
        node_features=FEATURES,
        hidden_channels=HIDDEN,
        dropout=0.0,
        anchor_enabled=True,
    ).eval()
    inputs = _inputs(batch=1, time_steps=7, candidates=5)
    inputs.update(
        anchor_rr=torch.linspace(12.0, 18.0, 7).reshape(1, -1),
        anchor_std=torch.ones(1, 7),
        anchor_available=torch.ones(1, 7, dtype=torch.bool),
    )
    changed = {name: value.clone() for name, value in inputs.items()}
    changed["anchor_rr"][:, 4:] = 40.0
    changed["anchor_std"][:, 4:] = 9.0
    changed["anchor_available"][:, 5:] = False
    with torch.no_grad():
        first = model(**inputs)
        second = model(**changed)
    for key in (
        "candidate_logits", "source_rr", "corrected_anchor_rr",
        "anchor_residual_bpm", "anchor_snap_gate", "state_sequence",
    ):
        assert torch.equal(first[key][:, :4], second[key][:, :4])


def test_session_or_gap_reset_clears_carried_state_before_window() -> None:
    model = _model().eval()
    inputs = _inputs(batch=1, time_steps=8, candidates=6)
    inputs["sequence_mask"][:, 3] = False  # missing/gap slot does not update state
    inputs["reset_mask"][:, 4] = True       # first post-gap window resets state
    suffix = _slice_time(inputs, 4, 8)
    suffix["reset_mask"].zero_()
    with torch.no_grad():
        reset_stream = model(**inputs)
        fresh_suffix = model(**suffix)
    for key in (
        "candidate_logits",
        "candidate_probabilities",
        "source_rr",
        "quality_logit",
        "state_sequence",
        "spike_sequence",
    ):
        torch.testing.assert_close(
            reset_stream[key][:, 4:], fresh_suffix[key], atol=2e-6, rtol=2e-6
        )
    _assert_state_close(reset_stream["state"], fresh_suffix["state"])


def test_no_candidate_or_radar_path_is_finite_and_explicitly_unavailable() -> None:
    model = _model().eval()
    inputs = _inputs(batch=2, time_steps=5, candidates=4)
    inputs["candidate_mask"].zero_()
    inputs["radar_mask"].zero_()
    inputs["node_features"].fill_(torch.nan)
    inputs["candidate_rr"].fill_(torch.nan)
    with torch.no_grad():
        output = model(**inputs)
    for name, value in output.items():
        if name == "state":
            for layer_state in value:
                for state_tensor in layer_state:
                    assert torch.isfinite(state_tensor).all()
        elif torch.is_tensor(value):
            assert torch.isfinite(value).all(), name
    assert not output["source_available"].any()
    assert torch.all(output["selected_index"] == -1)
    assert torch.count_nonzero(output["candidate_probabilities"]) == 0
    assert torch.count_nonzero(output["source_rr"]) == 0
    torch.testing.assert_close(
        output["source_scale_bpm"],
        torch.full((2, 5), model.maximum_scale_bpm),
        rtol=0,
        atol=0,
    )
    assert torch.all(output["candidate_scale_bpm"] > 0)


def test_listwise_residual_uncertainty_and_spikes_all_receive_gradients() -> None:
    model = _model().train()
    inputs = _inputs(batch=2, time_steps=6, candidates=6)
    inputs["node_features"].requires_grad_()
    output = model(**inputs)
    targets = torch.arange(12).reshape(2, 6) % 6
    listwise = F.nll_loss(
        output["candidate_probabilities"].clamp_min(1e-8).log().flatten(0, 1),
        targets.flatten(),
    )
    residual = (output["candidate_residual_bpm"] - 0.2).square().mean()
    scale = (output["candidate_scale_bpm"] - 0.8).square().mean()
    factor = F.cross_entropy(
        output["factor_logits"].flatten(0, 1), (targets % 4).flatten()
    )
    quality = F.binary_cross_entropy_with_logits(
        output["quality_logit"], torch.ones_like(output["quality_logit"])
    )
    loss = listwise + residual + scale + factor + quality
    loss.backward()

    assert inputs["node_features"].grad is not None
    assert torch.isfinite(inputs["node_features"].grad).all()
    assert float(inputs["node_features"].grad.abs().sum()) > 0
    for parameter in (
        model.candidate_logit_head[-1].weight,
        model.residual_head[-1].weight,
        model.scale_head[-1].weight,
        model.graph_cell.beta_logit,
        model.episode_encoder.cells[1].beta_logit,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0
    assert torch.all((output["spike_sequence"] >= 0) & (output["spike_sequence"] <= 1))
    assert float(output["spike_sequence"].sum().detach()) > 0


def test_i3_anchor_residual_uncertainty_gate_and_context_receive_gradients() -> None:
    model = HarmonicCandidateSetEpisodeSNN(
        node_features=FEATURES,
        hidden_channels=HIDDEN,
        dropout=0.0,
        anchor_enabled=True,
    ).train()
    with torch.no_grad():
        model.anchor_residual_head[-1].weight.fill_(0.01)
        model.anchor_scale_head[-1].weight.fill_(0.01)
        model.anchor_snap_gate_head[-1].weight.fill_(0.01)
    inputs = _inputs(batch=2, time_steps=6, candidates=6)
    anchor = torch.full((2, 6), 14.0)
    output = model(
        **inputs,
        anchor_rr=anchor,
        anchor_std=torch.full_like(anchor, 1.2),
        anchor_available=torch.ones_like(anchor, dtype=torch.bool),
    )
    loss = (
        (output["anchor_residual_bpm"] - 1.5).square().mean()
        + (output["corrected_anchor_scale_bpm"] - 0.8).square().mean()
        + F.binary_cross_entropy_with_logits(
            output["anchor_snap_gate_logit"],
            torch.ones_like(output["anchor_snap_gate_logit"]),
        )
    )
    loss.backward()
    for parameter in (
        model.anchor_context_projection.weight,
        model.anchor_residual_head[-1].weight,
        model.anchor_scale_head[-1].weight,
        model.anchor_snap_gate_head[-1].weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_candidate_limit_is_enforced() -> None:
    model = _model()
    inputs = _inputs(batch=1, time_steps=2, candidates=12)
    model(**inputs)
    oversized = {
        "node_features": torch.zeros(1, 2, 13, FEATURES),
        "candidate_rr": torch.full((1, 2, 13), 12.0),
        "candidate_mask": torch.ones(1, 2, 13, dtype=torch.bool),
        "sequence_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match=r"\[1, 12\]"):
        model(**oversized)
