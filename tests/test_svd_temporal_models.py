from __future__ import annotations

import pytest
import torch

from snn_rr.svd_temporal_models import (
    TEMPORAL_NUM_RR_BINS,
    TEMPORAL_RR_MAX,
    TEMPORAL_RR_MIN,
    TEMPORAL_RR_STEP,
    TemporalSourceSeparatedRRSNN,
)


def _inputs(
    batch: int = 2,
    *,
    variants: int = 2,
    components: int = 3,
    time_steps: int = 320,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260828)
    signals = torch.randn(
        batch,
        3,
        variants,
        components,
        time_steps,
        generator=generator,
    )
    attributes = torch.rand(
        batch, 3, variants, components, 5, generator=generator
    )
    attributes[..., 0] /= attributes[..., 0].sum(dim=-1, keepdim=True)
    attributes[..., 4] *= 0.8
    base_prediction = torch.tensor([18.25, 31.50])[:batch]
    base_std = torch.tensor([0.8, 1.2])[:batch]
    classical_rr = torch.tensor([[9.0, 9.5], [15.5, 16.0]])[:batch]
    return signals, attributes, base_prediction, base_std, classical_rr


def _small_model(**kwargs: object) -> TemporalSourceSeparatedRRSNN:
    return TemporalSourceSeparatedRRSNN(
        num_variants=2,
        num_components=3,
        compressor_channels=8,
        hidden_channels=12,
        cell_types=("lif", "plif", "alif"),
        dropout=0.0,
        **kwargs,
    )


def test_temporal_snn_shapes_finite_chronology_and_backward() -> None:
    signals, attributes, base_rr, base_std, classical_rr = _inputs()
    signals.requires_grad_()
    attributes.requires_grad_()
    model = _small_model()
    radar_mask = torch.tensor([[True, False, True], [True, True, True]])
    output = model(
        signals,
        attributes,
        base_rr,
        base_std,
        classical_rr,
        radar_mask,
        return_sequences=True,
    )

    assert output["probabilities"].shape == (2, TEMPORAL_NUM_RR_BINS)
    assert output["expert_probabilities"].shape == (
        2,
        4,
        TEMPORAL_NUM_RR_BINS,
    )
    assert output["divisor_probabilities"].shape == (2, 4)
    assert output["candidate_centers_rr"].shape == (2, 4)
    assert output["candidate_std_rr"].shape == (2, 4)
    assert output["radar_weights"].shape == (2, 3)
    assert output["component_attention"].shape == (2, 3, 2, 3)
    assert output["downsampled_steps"].item() == 80
    assert output["downsampled_sequence"].shape == (2, 80, 12)
    assert output["temporal_state_sequence"].shape == (2, 80, 12)
    assert output["multiscale_token_sequence"].shape == (2, 80, 48)
    assert output["layer_spike_rates_per_sample"].shape == (2, 3)
    assert output["radar_weights"][0, 1] == 0
    assert tuple(model.spiking_encoder.cell_types) == ("lif", "plif", "alif")
    assert model.context_windows == (10, 20, 40, 80)
    torch.testing.assert_close(output["probabilities"].sum(dim=1), torch.ones(2))
    torch.testing.assert_close(
        output["divisor_probabilities"].sum(dim=1), torch.ones(2)
    )
    torch.testing.assert_close(output["radar_weights"].sum(dim=1), torch.ones(2))
    assert torch.all(output["residual_rr_per_divisor"].abs() <= 1.5)
    assert torch.all(output["candidate_std_rr"] >= 0.35)
    assert torch.all(output["candidate_std_rr"] <= 8.0)
    torch.testing.assert_close(
        model.rr_bins[:3], torch.tensor([TEMPORAL_RR_MIN, 6.25, 6.50])
    )
    assert model.rr_bins[-1] == TEMPORAL_RR_MAX
    torch.testing.assert_close(
        torch.diff(model.rr_bins),
        torch.full((TEMPORAL_NUM_RR_BINS - 1,), TEMPORAL_RR_STEP),
    )
    for key, value in output.items():
        if torch.is_floating_point(value):
            assert torch.isfinite(value).all(), key

    loss = (
        output["expected_rr"].mean()
        + output["source_expected_rr"].mean()
        + output["rr_std"].mean()
        + output["quality_logits"].mean()
        + 0.01 * output["spike_rate"]
    )
    loss.backward()
    assert signals.grad is not None and torch.isfinite(signals.grad).all()
    assert attributes.grad is not None and torch.isfinite(attributes.grad).all()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        gradient is None or torch.isfinite(gradient).all() for gradient in gradients
    )
    assert sum(parameter.numel() for parameter in model.parameters()) < 250_000


def test_causal_prefix_is_invariant_to_future_samples() -> None:
    model = _small_model().eval()
    signals, attributes, base_rr, base_std, classical_rr = _inputs()
    full = model(
        signals,
        attributes,
        base_rr,
        base_std,
        classical_rr,
        return_sequences=True,
    )

    changed = signals.clone()
    changed[..., 160:] = 50.0 * torch.randn_like(changed[..., 160:])
    future_changed = model(
        changed,
        attributes,
        base_rr,
        base_std,
        classical_rr,
        return_sequences=True,
    )
    prefix_only = model(
        signals[..., :160],
        attributes,
        base_rr,
        base_std,
        classical_rr,
        return_sequences=True,
    )

    for key in (
        "downsampled_sequence",
        "temporal_state_sequence",
        "multiscale_token_sequence",
    ):
        torch.testing.assert_close(
            full[key][:, :40], future_changed[key][:, :40], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            full[key][:, :40], prefix_only[key], atol=0.0, rtol=0.0
        )
    assert prefix_only["downsampled_steps"].item() == 40


def test_negative_gate_is_safe_and_initial_residual_is_zero() -> None:
    model = _small_model().eval()
    signals, attributes, base_rr, base_std, classical_rr = _inputs()
    output = model(signals, attributes, base_rr, base_std, classical_rr)

    assert torch.all(output["mixture_gate"] < 0.0004)
    assert torch.max(
        torch.abs(output["expected_rr"] - output["base_expected_rr"])
    ) < 0.02
    assert torch.count_nonzero(output["residual_rr_per_divisor"]) == 0


def test_missing_radars_are_isolated_and_all_missing_is_exact_base() -> None:
    model = _small_model().eval()
    signals, attributes, base_rr, base_std, classical_rr = _inputs()
    radar_mask = torch.tensor([[True, False, True], [False, True, True]])
    first = model(
        signals,
        attributes,
        base_rr,
        base_std,
        classical_rr,
        radar_mask,
        return_sequences=True,
    )
    changed_signals = signals.clone()
    changed_attributes = attributes.clone()
    changed_signals[0, 1] = float("nan")
    changed_attributes[0, 1] = float("inf")
    changed_signals[1, 0] = -1.0e9
    changed_attributes[1, 0] = -1.0e9
    second = model(
        changed_signals,
        changed_attributes,
        base_rr,
        base_std,
        classical_rr,
        radar_mask,
        return_sequences=True,
    )
    for key in (
        "probabilities",
        "expected_rr",
        "rr_std",
        "radar_weights",
        "downsampled_sequence",
        "temporal_state_sequence",
    ):
        torch.testing.assert_close(first[key], second[key])
    assert first["radar_weights"][0, 1] == 0
    assert first["radar_weights"][1, 0] == 0

    missing = model(
        signals,
        attributes,
        base_rr,
        base_std,
        classical_rr,
        torch.zeros(2, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(missing["probabilities"], missing["base_probabilities"])
    torch.testing.assert_close(missing["expected_rr"], missing["base_expected_rr"])
    assert torch.count_nonzero(missing["radar_weights"]) == 0
    assert torch.count_nonzero(missing["mixture_gate"]) == 0
    assert torch.count_nonzero(missing["quality"]) == 0


def test_multiplier_mask_and_input_contracts() -> None:
    model = _small_model().eval()
    signals, attributes, base_rr, base_std, _ = _inputs()
    output = model(
        signals,
        attributes,
        base_rr,
        base_std,
        torch.tensor([[5.0], [20.0]]),
    )
    expected_valid = torch.tensor(
        [[False, True, True, True], [True, True, False, False]]
    )
    torch.testing.assert_close(output["candidate_valid_mask"], expected_valid)
    assert torch.count_nonzero(output["divisor_probabilities"][~expected_valid]) == 0
    torch.testing.assert_close(
        output["divisor_probabilities"].sum(dim=1), torch.ones(2)
    )

    corrupt = model(
        signals,
        attributes,
        base_rr,
        base_std,
        torch.tensor([[float("nan"), 5.0], [float("inf"), -2.0]]),
    )
    for value in corrupt.values():
        if torch.is_floating_point(value):
            assert torch.isfinite(value).all()

    with pytest.raises(ValueError, match="component_signals must have shape"):
        model(signals[:, :, :, :2], attributes, base_rr, base_std)
    with pytest.raises(ValueError, match="attributes must have shape"):
        model(signals, attributes[..., :4], base_rr, base_std)
    with pytest.raises(ValueError, match="time length"):
        model(signals[..., :-1], attributes, base_rr, base_std)
    with pytest.raises(ValueError, match="classical_rr"):
        model(signals, attributes, base_rr, base_std, torch.ones(3))
    with pytest.raises(ValueError, match="radar_mask"):
        model(
            signals,
            attributes,
            base_rr,
            base_std,
            radar_mask=torch.ones(2, 2),
        )
