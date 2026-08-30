from __future__ import annotations

import pytest
import torch

from snn_rr.models import (
    SharedRadarCNNTeacher,
    TriRadarRRSNN,
    _HarmonicAuxiliaryLogitHead,
    _align_auxiliary_spectra_to_map,
    _split_cached_auxiliary,
    apply_radar_dropout,
    count_trainable_parameters,
    expected_rr_from_logits,
    gaussian_soft_targets,
    make_rr_bins,
)


def _assert_finite_output(output: dict[str, torch.Tensor]) -> None:
    for key in (
        "logits",
        "probabilities",
        "expected_rr",
        "log_variance",
        "rr_std",
        "quality_logits",
        "quality",
        "map_rr",
        "topk_rr",
        "topk_probability",
        "posterior_entropy",
        "radar_weights",
    ):
        assert torch.isfinite(output[key]).all(), key


def _assert_finite_gradients(model: torch.nn.Module) -> None:
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_gaussian_soft_targets_and_expected_rr() -> None:
    bins = make_rr_bins(6.0, 40.0, 137)
    target = torch.tensor([8.0, 20.25, 38.0])
    soft = gaussian_soft_targets(target, bins, sigma=0.4)

    assert soft.shape == (3, 137)
    torch.testing.assert_close(soft.sum(dim=-1), torch.ones(3))
    reconstructed = (soft * bins).sum(dim=-1)
    torch.testing.assert_close(reconstructed, target, atol=0.02, rtol=0.0)

    # The utility is differentiable and uses the same bin semantics as models.
    logits = soft.clamp_min(1e-8).log().requires_grad_()
    expected = expected_rr_from_logits(logits, bins)
    expected.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_radar_dropout_respects_existing_mask_and_keeps_one() -> None:
    torch.manual_seed(7)
    x = torch.ones(12, 3, 5, 7)
    available = torch.tensor([[True, True, False]]).expand(12, -1)
    dropped_x, kept = apply_radar_dropout(
        x, available, p=1.0, training=True, ensure_one=True
    )

    assert kept.shape == (12, 3)
    assert kept[:, 2].sum() == 0
    assert torch.all(kept.sum(dim=1) == 1)
    torch.testing.assert_close(
        dropped_x.sum(dim=(2, 3)), kept.to(dtype=x.dtype) * 35.0
    )


@pytest.mark.parametrize("range_bins", [91, 182])
def test_teacher_arbitrary_shape_mask_aux_and_backward(range_bins: int) -> None:
    torch.manual_seed(11)
    model = SharedRadarCNNTeacher(
        num_rr_bins=81,
        spatial_channels=(16, 24, 40),
        frequency_dilations=(1, 2),
        radar_dropout_p=0.0,
        aux_dim=4,
    )
    # Batch size one intentionally exercises all GroupNorm-based paths.
    x = torch.randn(1, 3, 23, range_bins, requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    aux = torch.randn(1, 4, requires_grad=True)
    output = model(x, mask, aux)

    assert output["logits"].shape == (1, 81)
    assert output["expected_rr"].shape == (1,)
    assert output["log_variance"].shape == (1,)
    assert output["quality"].shape == (1,)
    assert output["radar_weights"].shape == (1, 3)
    assert output["range_attention"].shape[:3] == (1, 3, 23)
    assert output["radar_weights"][0, 1] == 0
    torch.testing.assert_close(output["radar_weights"].sum(dim=1), torch.ones(1))
    _assert_finite_output(output)
    assert count_trainable_parameters(model) < 3_000_000

    loss = (
        output["expected_rr"].mean()
        + output["log_variance"].mean()
        + output["quality_logits"].mean()
        + 0.01 * output["logits"].square().mean()
    )
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert aux.grad is not None and torch.isfinite(aux.grad).all()
    _assert_finite_gradients(model)


def test_teacher_map_only_api_and_all_missing_are_finite() -> None:
    model = SharedRadarCNNTeacher(
        num_rr_bins=33,
        spatial_channels=(12, 20, 32),
        frequency_dilations=(1,),
        radar_dropout_p=0.0,
    ).eval()
    x = torch.randn(1, 3, 17, 29)
    output = model(x, torch.zeros(1, 3, dtype=torch.bool))

    _assert_finite_output(output)
    assert torch.count_nonzero(output["radar_weights"]) == 0
    assert output["quality"].item() < 1e-6


def test_cache_branches_are_exposed_as_channels() -> None:
    model = SharedRadarCNNTeacher(
        num_rr_bins=33,
        spatial_channels=(12, 20, 32),
        frequency_dilations=(1,),
        radar_dropout_p=0.0,
        input_branches=2,
    ).eval()
    output = model(torch.randn(2, 3, 17, 182))
    assert output["logits"].shape == (2, 33)
    assert output["range_attention"].shape == (2, 3, 17, 23)
    _assert_finite_output(output)

    with pytest.raises(ValueError, match="not divisible"):
        model(torch.randn(2, 3, 17, 181))


def test_physical_frequency_resampling_and_masked_auxiliary_isolation() -> None:
    model = SharedRadarCNNTeacher(
        rr_min=6.0,
        rr_max=30.0,
        num_rr_bins=5,
        spatial_channels=(8, 12),
        frequency_dilations=(1,),
        radar_dropout_p=0.0,
        aux_dim=2,
        input_frequency_min_hz=0.1,
        input_frequency_max_hz=0.5,
    ).eval()
    local = torch.arange(5.0).view(1, 1, 5)
    torch.testing.assert_close(
        model._resample_frequency_logits(local), torch.arange(5.0).view(1, 5)
    )

    x = torch.randn(1, 3, 9, 17)
    mask = torch.tensor([[True, False, True]])
    first = model(x, mask, torch.tensor([[1.0, 2.0]]))["logits"]
    second = model(x, mask, torch.tensor([[100.0, -50.0]]))["logits"]
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("model_type", ["teacher", "snn"])
def test_structured_auxiliary_preserves_frequency_path_and_backward(
    model_type: str,
) -> None:
    # F=5 gives base_dim=3*(2F+8)+2F+5=69; append three causal columns.
    common = dict(
        num_rr_bins=33,
        spatial_channels=(8, 12),
        radar_dropout_p=0.0,
        aux_dim=72,
        structured_auxiliary=True,
        aux_base_dim=69,
    )
    if model_type == "teacher":
        model = SharedRadarCNNTeacher(
            **common,
            frequency_dilations=(1,),
        )
    else:
        model = TriRadarRRSNN(
            **common,
            hidden_channels=16,
            num_spiking_blocks=1,
            simulation_steps=2,
        )
    x = torch.randn(2, 3, 9, 17, requires_grad=True)
    aux = torch.randn(2, 72, requires_grad=True)
    output = model(x, torch.ones(2, 3, dtype=torch.bool), aux)
    _assert_finite_output(output)
    output["expected_rr"].sum().backward()
    assert aux.grad is not None and torch.isfinite(aux.grad).all()
    assert torch.count_nonzero(aux.grad[:, :40]) > 0

    with pytest.raises(ValueError, match="base_aux_dim"):
        SharedRadarCNNTeacher(
            **{**common, "aux_base_dim": 68},
            frequency_dilations=(1,),
        )


@pytest.mark.parametrize("model_type", ["teacher", "snn"])
def test_harmonic_auxiliary_head_is_trainable_and_mask_safe(
    model_type: str,
) -> None:
    # Five full-resolution spectra bins give base_dim=69; three history fields
    # follow the current-window layout.
    common = dict(
        rr_min=6.0,
        rr_max=30.0,
        num_rr_bins=25,
        spatial_channels=(8, 12),
        radar_dropout_p=0.0,
        aux_dim=72,
        structured_auxiliary=True,
        aux_base_dim=69,
        exact_auxiliary_alignment=True,
        harmonic_auxiliary=True,
        alias_gated_harmonic=True,
        auxiliary_frequency_min_hz=0.10,
        auxiliary_frequency_max_hz=0.50,
        input_frequency_min_hz=0.10,
        input_frequency_max_hz=0.50,
    )
    if model_type == "teacher":
        model = SharedRadarCNNTeacher(**common, frequency_dilations=(1,))
    else:
        model = TriRadarRRSNN(
            **common,
            hidden_channels=16,
            num_spiking_blocks=1,
            simulation_steps=2,
        )

    # Two map bins exercise the exact 5 -> 2 pair-pooling alignment path.
    x = torch.randn(2, 3, 2, 17)
    aux = torch.randn(2, 72, requires_grad=True)
    full_mask = torch.ones(2, 3, dtype=torch.bool)
    output = model(x, full_mask, aux)
    assert output["harmonic_logits"].shape == (2, 25)
    assert 0.4 < float(output["harmonic_gain"].detach()) < 0.6
    assert output["alias_logits"].shape == (2,)
    assert torch.all((output["alias_probability"] >= 0) & (output["alias_probability"] <= 1))
    output["harmonic_logits"].square().sum().backward()
    gradients = [
        parameter.grad
        for parameter in model.harmonic_logit_head.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in gradients)

    model.eval()
    partial_mask = torch.tensor([[True, False, True], [True, True, False]])
    first = model(x, partial_mask, torch.randn(2, 72))["harmonic_logits"]
    second = model(x, partial_mask, 100.0 * torch.randn(2, 72))["harmonic_logits"]
    torch.testing.assert_close(first, torch.zeros_like(first))
    torch.testing.assert_close(second, torch.zeros_like(second))


def test_harmonic_sampling_schema_boundaries_and_roundtrip() -> None:
    # F=9 gives base_dim=101.  Use an exact 0.05 Hz grid from 0.10 to 0.50.
    head = _HarmonicAuxiliaryLogitHead(
        aux_dim=104,
        base_aux_dim=101,
        rr_min=6.0,
        rr_max=24.0,
        num_rr_bins=4,
        auxiliary_frequency_min_hz=0.10,
        auxiliary_frequency_max_hz=0.50,
        hidden_channels=4,
    )
    ramp = (10.0 + torch.arange(9.0)).view(1, 1, 9)
    sampled = head._sample_hypotheses(ramp).reshape(1, 4, 4)[0]
    torch.testing.assert_close(sampled[0], torch.tensor([10.0, 12.0, 14.0, 16.0]))
    torch.testing.assert_close(sampled[1], torch.tensor([0.0, 10.0, 11.0, 12.0]))
    torch.testing.assert_close(
        sampled[2], torch.tensor([0.0, 0.0, 10.0, 10.0 + 2.0 / 3.0])
    )
    torch.testing.assert_close(sampled[3], torch.tensor([0.0, 0.0, 0.0, 10.0]))

    aligned = _align_auxiliary_spectra_to_map(
        torch.tensor([[[0.0, 2.0, 4.0, 6.0, 100.0]]]), 2
    )
    torch.testing.assert_close(aligned, torch.tensor([[[1.0, 5.0]]]))

    sentinel = torch.arange(104.0).view(1, -1)
    spectra, global_values = _split_cached_auxiliary(sentinel, 101)
    assert spectra.shape == (1, 8, 9)
    torch.testing.assert_close(spectra[0, :6].reshape(-1), torch.arange(54.0))
    torch.testing.assert_close(spectra[0, 6:].reshape(-1), torch.arange(78.0, 96.0))
    expected_global = torch.cat(
        (torch.arange(54.0, 78.0), torch.arange(96.0, 101.0), torch.arange(101.0, 104.0))
    )
    torch.testing.assert_close(global_values[0], expected_global)

    reference = torch.zeros(1, 3)
    none_logits, _, _ = head(None, reference)
    zero_logits, _, _ = head(torch.zeros(1, 104), reference)
    torch.testing.assert_close(none_logits, torch.zeros(1, 4))
    torch.testing.assert_close(zero_logits, torch.zeros(1, 4))

    clone = _HarmonicAuxiliaryLogitHead(
        aux_dim=104,
        base_aux_dim=101,
        rr_min=6.0,
        rr_max=24.0,
        num_rr_bins=4,
        auxiliary_frequency_min_hz=0.10,
        auxiliary_frequency_max_hz=0.50,
        hidden_channels=4,
    )
    clone.load_state_dict(head.state_dict())
    random_aux = torch.randn(1, 104)
    torch.testing.assert_close(head(random_aux, reference)[0], clone(random_aux, reference)[0])


def test_production_harmonic_grid_valid_counts_and_flat_fusion() -> None:
    head = _HarmonicAuxiliaryLogitHead(
        aux_dim=1237,
        base_aux_dim=1205,
        rr_min=6.0,
        rr_max=45.0,
        num_rr_bins=157,
        auxiliary_frequency_min_hz=0.0830078125,
        auxiliary_frequency_max_hz=0.7958984375,
        hidden_channels=4,
    )
    assert head.frequency_bins == 147
    assert head.sample_valid.sum(dim=1).tolist() == [157, 141, 121, 101]

    # Harmonic evidence is independently usable with the legacy flat fusion;
    # the training CLI permits this combination for controlled ablations.
    model = SharedRadarCNNTeacher(
        rr_min=6.0,
        rr_max=30.0,
        num_rr_bins=25,
        spatial_channels=(8, 12),
        frequency_dilations=(1,),
        radar_dropout_p=0.0,
        aux_dim=69,
        structured_auxiliary=False,
        aux_base_dim=69,
        harmonic_auxiliary=True,
        auxiliary_frequency_min_hz=0.10,
        auxiliary_frequency_max_hz=0.50,
        input_frequency_min_hz=0.10,
        input_frequency_max_hz=0.50,
    ).eval()
    output = model(
        torch.randn(1, 3, 3, 17),
        torch.ones(1, 3, dtype=torch.bool),
        torch.randn(1, 69),
    )
    assert output["harmonic_logits"].shape == (1, 25)


def test_snn_shapes_spike_statistics_and_finite_backward() -> None:
    torch.manual_seed(19)
    model = TriRadarRRSNN(
        num_rr_bins=65,
        spatial_channels=(16, 24, 40),
        hidden_channels=48,
        num_spiking_blocks=2,
        simulation_steps=8,
        radar_dropout_p=0.0,
        aux_dim=3,
    )
    x = torch.randn(2, 3, 21, 97, requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, True, True]])
    aux = torch.randn(2, 3, requires_grad=True)
    output = model(x, mask, aux)

    assert output["logits"].shape == (2, 65)
    assert output["expected_rr"].shape == (2,)
    assert output["spike_rate"].ndim == 0
    assert output["spike_rate_per_sample"].shape == (2,)
    assert output["layer_spike_rates"].shape == (5,)
    assert output["layer_spike_rates_per_sample"].shape == (2, 5)
    assert torch.all((output["layer_spike_rates"] >= 0.0))
    assert torch.all((output["layer_spike_rates"] <= 1.0))
    assert output["radar_weights"][0, 2] == 0
    assert output["radar_weights"][1, 0] == 0
    torch.testing.assert_close(output["radar_weights"].sum(dim=1), torch.ones(2))
    _assert_finite_output(output)
    assert count_trainable_parameters(model) < 1_500_000

    loss = (
        output["expected_rr"].mean()
        + output["log_variance"].mean()
        + output["quality_logits"].mean()
        + 0.1 * output["spike_rate"]
        + 0.01 * output["logits"].square().mean()
    )
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert aux.grad is not None and torch.isfinite(aux.grad).all()
    _assert_finite_gradients(model)


def test_input_validation() -> None:
    model = SharedRadarCNNTeacher(
        num_rr_bins=17,
        spatial_channels=(8, 12),
        frequency_dilations=(1,),
        radar_dropout_p=0.0,
        aux_dim=2,
    )
    with pytest.raises(ValueError, match="input must have shape"):
        model(torch.randn(2, 3, 10))
    with pytest.raises(ValueError, match="radar_mask"):
        model(torch.randn(2, 3, 10, 11), torch.ones(2, 2))
    with pytest.raises(ValueError, match="aux must have shape"):
        model(torch.randn(2, 3, 10, 11), aux=torch.randn(2, 3))
