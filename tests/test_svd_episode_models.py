from __future__ import annotations

import torch

from snn_rr.svd_episode_models import EpisodeAliasRRSNN


def _inputs(*, batch: int = 2, windows: int = 7) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260828)
    return {
        "evidence": torch.rand(batch, windows, 3, 4, 10),
        "context": torch.rand(batch, windows, 3),
        "classical_rr": torch.full((batch, windows), 9.0),
        "base_prediction": torch.full((batch, windows), 18.0),
        "base_std": torch.ones(batch, windows),
        "base_alias_probability": torch.full((batch, windows), 0.2),
        "base_available": torch.ones(batch, windows, dtype=torch.bool),
        "radar_mask": torch.ones(batch, windows, 3, dtype=torch.bool),
        "sequence_mask": torch.ones(batch, windows, dtype=torch.bool),
    }


def _model(*, use_base_features: bool = False) -> EpisodeAliasRRSNN:
    return EpisodeAliasRRSNN(
        evidence_features=10,
        context_features=3,
        candidate_channels=8,
        hidden_channels=12,
        cell_types=("lif", "plif", "alif"),
        dropout=0.0,
        use_base_features=use_base_features,
    )


def test_episode_output_is_normalized_and_has_five_safe_experts() -> None:
    model = _model()
    inputs = _inputs()
    output = model(**inputs)
    assert output["probabilities"].shape == (2, 7, 157)
    assert output["divisor_probabilities"].shape == (2, 7, 4)
    assert output["expert_probabilities"].shape == (2, 7, 5)
    torch.testing.assert_close(output["learned_gate"], output["gate_logits"].sigmoid())
    torch.testing.assert_close(output["quality"], output["quality_logit"].sigmoid())
    torch.testing.assert_close(
        output["probabilities"].sum(dim=-1), torch.ones(2, 7), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        output["expert_probabilities"].sum(dim=-1),
        torch.ones(2, 7),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.all(output["residual_rr"].abs() <= model.max_residual_bpm + 1e-6)
    assert float(output["learned_gate"].max().detach()) < 0.01


def test_no_radar_is_exact_base_fallback_and_no_base_uses_source() -> None:
    model = _model()
    inputs = _inputs(batch=1, windows=5)
    inputs["radar_mask"].zero_()
    output = model(**inputs)
    torch.testing.assert_close(output["expected_rr"], inputs["base_prediction"], rtol=0, atol=0)
    torch.testing.assert_close(output["rr_std"], inputs["base_std"], rtol=0, atol=0)
    assert torch.count_nonzero(output["mixture_gate"]) == 0

    inputs["base_available"].zero_()
    inputs["radar_mask"].fill_(True)
    output = model(**inputs)
    torch.testing.assert_close(output["mixture_gate"], torch.ones(1, 5), rtol=0, atol=0)
    torch.testing.assert_close(
        output["expected_rr"], output["source_prediction"], atol=1e-5, rtol=1e-5
    )

    inputs["radar_mask"].zero_()
    output = model(**inputs)
    assert torch.isfinite(output["probabilities"]).all()
    torch.testing.assert_close(
        output["probabilities"].sum(dim=-1), torch.ones(1, 5), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        output["expected_rr"], output["source_prediction"], atol=1e-5, rtol=1e-5
    )


def test_physical_window_axis_is_causal_and_padding_does_not_update_state() -> None:
    model = _model().eval()
    inputs = _inputs(batch=1, windows=8)
    with torch.no_grad():
        original = model(**inputs)
        changed = {name: value.clone() for name, value in inputs.items()}
        changed["evidence"][:, 5:] = 100.0
        changed["context"][:, 5:] = -100.0
        future_changed = model(**changed)
    torch.testing.assert_close(
        original["source_prediction"][:, :5],
        future_changed["source_prediction"][:, :5],
        rtol=0,
        atol=0,
    )
    padded = {name: value.clone() for name, value in inputs.items()}
    padded["sequence_mask"][:, 5:] = False
    with torch.no_grad():
        padded_output = model(**padded)
    torch.testing.assert_close(
        original["state_sequence"][:, :5], padded_output["state_sequence"][:, :5]
    )
    assert torch.count_nonzero(padded_output["state_sequence"][:, 5:]) == 0


def test_strict_primary_router_is_invariant_to_frozen_base_features() -> None:
    model = _model(use_base_features=False).eval()
    inputs = _inputs(batch=1, windows=6)
    altered = {name: value.clone() for name, value in inputs.items()}
    altered["base_prediction"].fill_(40.0)
    altered["base_std"].fill_(7.0)
    altered["base_alias_probability"].fill_(0.99)
    with torch.no_grad():
        first = model(**inputs)
        second = model(**altered)
    for name in ("source_prediction", "divisor_probabilities", "residual_rr", "learned_gate"):
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_strict_alias_gate_weights_are_invariant_to_all_frozen_base_channels() -> None:
    model = EpisodeAliasRRSNN(
        evidence_features=10,
        context_features=3,
        candidate_channels=8,
        hidden_channels=12,
        cell_types=("lif", "plif"),
        dropout=0.0,
        strict_alias_gate=True,
    ).eval()
    inputs = _inputs(batch=1, windows=6)
    altered = {name: value.clone() for name, value in inputs.items()}
    altered["base_prediction"].fill_(41.0)
    altered["base_std"].fill_(7.0)
    altered["base_alias_probability"].fill_(0.99)
    altered["base_available"].zero_()
    with torch.no_grad():
        first = model(**inputs)
        second = model(**altered)
    for name in (
        "gate_logits",
        "learned_gate",
        "source_prediction",
        "divisor_probabilities",
        "residual_rr",
        "state_sequence",
    ):
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_episode_source_supervision_backpropagates_behind_closed_gate() -> None:
    model = _model()
    inputs = _inputs(batch=2, windows=5)
    output = model(**inputs)
    loss = -output["source_probabilities"][..., 60].clamp_min(1e-8).log().mean()
    loss.backward()
    divisor_grad = model.divisor_head[-1].weight.grad
    residual_grad = model.residual_head[-1].weight.grad
    assert divisor_grad is not None and torch.isfinite(divisor_grad).all()
    assert residual_grad is not None and torch.isfinite(residual_grad).all()
