from __future__ import annotations

import pytest
import torch

from snn_rr.svd_models import (
    SVD_NUM_RR_BINS,
    SVD_RR_MAX,
    SVD_RR_MIN,
    SVD_RR_STEP,
    SourceSeparatedRRSNN,
)


def _inputs(
    batch: int = 2,
    *,
    variants: int = 2,
    components: int = 3,
    frequencies: int = 41,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260828)
    spectra = torch.rand(
        batch, 3, variants, components, frequencies, generator=generator
    )
    spectra = spectra / spectra.sum(dim=-1, keepdim=True)
    attributes = torch.rand(
        batch, 3, variants, components, 5, generator=generator
    )
    # Energy fractions have a meaningful per-variant normalization.
    attributes[..., 0] /= attributes[..., 0].sum(dim=-1, keepdim=True)
    attributes[..., 4] *= 0.8
    base_prediction = torch.tensor([18.25, 31.50])[:batch]
    base_std = torch.tensor([0.8, 1.2])[:batch]
    base_features = torch.randn(batch, 4, generator=generator)
    return spectra, attributes, base_prediction, base_std, base_features


def _small_model(**kwargs: object) -> SourceSeparatedRRSNN:
    return SourceSeparatedRRSNN(
        num_variants=2,
        num_components=3,
        base_feature_dim=4,
        encoder_channels=12,
        encoder_blocks=1,
        hidden_channels=16,
        num_spiking_blocks=1,
        simulation_steps=2,
        radar_dropout_p=0.0,
        **kwargs,
    )


def test_source_separated_snn_shapes_finite_probabilities_and_backward() -> None:
    spectra, attributes, base_rr, base_std, base_features = _inputs()
    spectra.requires_grad_()
    attributes.requires_grad_()
    base_features.requires_grad_()
    model = _small_model()
    classical_rr = torch.tensor([[9.0, float("nan")], [15.5, 100.0]])
    classical_std = torch.tensor([[0.5, 1.0], [0.7, 2.0]])
    output = model(
        spectra,
        attributes,
        base_rr,
        base_std,
        base_features,
        classical_rr,
        classical_std,
        torch.tensor([[True, False, True], [True, True, True]]),
    )

    assert output["logits"].shape == (2, SVD_NUM_RR_BINS)
    assert output["probabilities"].shape == (2, SVD_NUM_RR_BINS)
    assert output["topk_rr"].shape == (2, 5)
    assert output["topk_probability"].shape == (2, 5)
    assert output["candidate_priors"].shape == (2, 4, SVD_NUM_RR_BINS)
    assert output["candidate_valid_mask"].shape == (2, 4)
    assert output["component_quality"].shape == (2, 3, 2, 3)
    assert output["radar_weights"].shape == (2, 3)
    assert output["radar_mask"].shape == (2, 3)
    assert output["layer_spike_rates_per_sample"].shape == (2, 3)
    assert output["radar_weights"][0, 1] == 0
    torch.testing.assert_close(output["probabilities"].sum(dim=1), torch.ones(2))
    torch.testing.assert_close(output["radar_weights"].sum(dim=1), torch.ones(2))
    torch.testing.assert_close(
        model.rr_bins[:3], torch.tensor([SVD_RR_MIN, 6.25, 6.50])
    )
    assert model.rr_bins[-1] == SVD_RR_MAX
    assert torch.allclose(
        torch.diff(model.rr_bins), torch.full((SVD_NUM_RR_BINS - 1,), SVD_RR_STEP)
    )
    for key, value in output.items():
        if torch.is_floating_point(value):
            assert torch.isfinite(value).all(), key

    loss = (
        output["expected_rr"].mean()
        + output["rr_std"].mean()
        + output["source_expected_rr"].mean()
        + output["quality_logits"].mean()
        + 0.01 * output["spike_rate"]
    )
    loss.backward()
    assert spectra.grad is not None and torch.isfinite(spectra.grad).all()
    assert attributes.grad is not None and torch.isfinite(attributes.grad).all()
    assert base_features.grad is not None and torch.isfinite(base_features.grad).all()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        gradient is None or torch.isfinite(gradient).all() for gradient in gradients
    )


def test_initial_negative_gate_keeps_prediction_close_to_base_posterior() -> None:
    model = _small_model().eval()
    spectra, attributes, base_rr, base_std, base_features = _inputs()
    output = model(spectra, attributes, base_rr, base_std, base_features)

    assert torch.all(output["mixture_gate"] < 0.003)
    # The strict bound follows from gate<0.003 and the 39-bpm output support.
    assert torch.max(
        torch.abs(output["expected_rr"] - output["base_expected_rr"])
    ) < 0.12


def test_all_radars_masked_is_exact_finite_base_fallback() -> None:
    model = _small_model().eval()
    spectra, attributes, base_rr, base_std, base_features = _inputs()
    output = model(
        spectra,
        attributes,
        base_rr,
        base_std,
        base_features,
        radar_mask=torch.zeros(2, 3, dtype=torch.bool),
    )

    torch.testing.assert_close(output["probabilities"], output["base_probabilities"])
    torch.testing.assert_close(output["expected_rr"], output["base_expected_rr"])
    assert torch.count_nonzero(output["mixture_gate"]) == 0
    assert torch.count_nonzero(output["radar_weights"]) == 0
    assert torch.count_nonzero(output["quality"]) == 0
    assert torch.isfinite(output["source_probabilities"]).all()
    assert torch.isfinite(output["spike_rate"])


def test_masked_radar_cannot_change_prediction_or_attention() -> None:
    model = _small_model().eval()
    spectra, attributes, base_rr, base_std, base_features = _inputs()
    radar_mask = torch.tensor([[True, False, True], [False, True, True]])
    first = model(
        spectra,
        attributes,
        base_rr,
        base_std,
        base_features,
        radar_mask=radar_mask,
    )
    changed_spectra = spectra.clone()
    changed_attributes = attributes.clone()
    changed_spectra[0, 1] = 1.0e6
    changed_attributes[0, 1] = -1.0e6
    changed_spectra[1, 0] = float("nan")
    changed_attributes[1, 0] = float("inf")
    second = model(
        changed_spectra,
        changed_attributes,
        base_rr,
        base_std,
        base_features,
        radar_mask=radar_mask,
    )

    for key in (
        "probabilities",
        "expected_rr",
        "rr_std",
        "source_probabilities",
        "radar_weights",
        "spike_rate_per_sample",
    ):
        torch.testing.assert_close(first[key], second[key])
    assert first["radar_weights"][0, 1] == 0
    assert first["radar_weights"][1, 0] == 0


def test_classical_multiplier_priors_mask_invalid_and_out_of_range() -> None:
    model = _small_model().eval()
    spectra, attributes, base_rr, base_std, base_features = _inputs()
    output = model(
        spectra,
        attributes,
        base_rr,
        base_std,
        base_features,
        classical_rr=torch.tensor([[5.0], [20.0]]),
        classical_std=torch.tensor(0.5),
    )

    expected_valid = torch.tensor(
        [[False, True, True, True], [True, True, False, False]]
    )
    torch.testing.assert_close(output["candidate_valid_mask"], expected_valid)
    invalid = ~expected_valid
    assert torch.count_nonzero(output["candidate_priors"][invalid]) == 0
    torch.testing.assert_close(
        output["candidate_priors"][expected_valid].sum(dim=1),
        torch.ones(int(expected_valid.sum())),
    )


def test_input_shape_contracts_are_strict() -> None:
    model = _small_model().eval()
    spectra, attributes, base_rr, base_std, base_features = _inputs()
    with pytest.raises(ValueError, match="spectra must have shape"):
        model(spectra[:, :, :, :2], attributes, base_rr, base_std, base_features)
    with pytest.raises(ValueError, match="attributes must have shape"):
        model(spectra, attributes[..., :4], base_rr, base_std, base_features)
    with pytest.raises(ValueError, match="base_features must have shape"):
        model(spectra, attributes, base_rr, base_std, base_features[:, :3])
    with pytest.raises(ValueError, match="radar_mask must have shape"):
        model(
            spectra,
            attributes,
            base_rr,
            base_std,
            base_features,
            radar_mask=torch.ones(2, 2),
        )
