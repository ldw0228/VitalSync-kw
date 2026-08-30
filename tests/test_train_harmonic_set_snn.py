from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from snn_rr.harmonic_set_models import HarmonicCandidateSetEpisodeSNN


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_harmonic_set_snn.py"
_SPEC = importlib.util.spec_from_file_location("snn_rr_train_harmonic_set", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
trainer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = trainer
_SPEC.loader.exec_module(trainer)


def _synthetic_cache(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "cache"
    root.mkdir()
    rng = np.random.default_rng(12)
    rows_per_fold = 10
    rows = trainer.N_FOLDS * rows_per_fold
    candidates = 3
    features = 6
    metadata: list[dict[str, object]] = []
    candidate_rr = np.empty((rows, candidates), np.float32)
    fallback_rows: list[dict[str, float | int]] = []
    for fold in range(trainer.N_FOLDS):
        for window in range(rows_per_fold):
            position = fold * rows_per_fold + window
            target = 10.0 + fold + 0.05 * window
            metadata.append(
                {
                    "cache_index": position,
                    "fold": fold,
                    "session_id": f"S{fold}",
                    "identity": f"I{fold}",
                    "protocol": "synthetic",
                    "window_number": window,
                    "window_start_s": float(window * 4),
                    "window_end_s": float(window * 4 + 32),
                    "rr_bpm": target,
                    "reference_valid": True,
                    "classical_rr_bpm": target + 0.2,
                }
            )
            candidate_rr[position] = [target - 1.0, target + 0.1, target + 3.0]
            fallback_rows.append({"cache_index": position, "prediction_bpm": target + 0.4, "rr_std_bpm": 1.0})
    node = rng.normal(size=(rows, candidates, features)).astype(np.float32)
    mask = np.ones((rows, candidates), bool)
    radar = np.ones((rows, 3), bool)
    np.save(root / "node_features.npy", node)
    np.save(root / "candidate_bpm.npy", candidate_rr)
    np.save(root / "candidate_mask.npy", mask)
    np.save(root / "joint_radar_mask.npy", radar)
    pd.DataFrame(metadata).to_csv(root / "metadata.csv", index=False)
    (root / "manifest.json").write_text(
        json.dumps({"format_version": 1, "complete": True, "row_count": rows}),
        encoding="utf-8",
    )
    fallback = tmp_path / "fallback.csv"
    pd.DataFrame(fallback_rows).to_csv(fallback, index=False)
    return root, fallback


def _args(cache: Path, fallback: Path, output: Path, *extra: str):
    return trainer.parse_args(
        [
            "--cache", str(cache), "--fallback-oof", str(fallback),
            "--output-dir", str(output), "--fold", "0", "--seed", "7",
            "--device", "cpu", "--preset", "tiny", "--epochs", "1",
            "--minimum-epochs", "1", "--patience", "1", "--deterministic",
            *extra,
        ]
    )


def _prediction(*, base_available: np.ndarray, source_available: np.ndarray) -> trainer.Predictions:
    rows = len(base_available)
    return trainer.Predictions(
        position=np.arange(rows), cache_index=np.arange(rows),
        target=np.linspace(10.0, 11.0, rows).astype(np.float32),
        identity=np.asarray(["A"] * rows),
        base_prediction=np.zeros(rows, np.float32),
        base_std=np.full(rows, 4.0, np.float32),
        base_available=np.asarray(base_available, bool),
        source_prediction=np.linspace(10.0, 11.0, rows).astype(np.float32),
        source_scale=np.ones(rows, np.float32),
        source_available=np.asarray(source_available, bool),
        selected_index=np.ones(rows, np.int64),
        selected_probability=np.ones(rows, np.float32),
        margin=np.ones(rows, np.float32), entropy=np.zeros(rows, np.float32),
        quality=np.ones(rows, np.float32), spike_rate=np.full(rows, 0.05, np.float32),
    )


def test_policy_grid_cardinality_is_frozen() -> None:
    assert sum(1 for _ in trainer.iter_policy_grid()) == 19 * 9 * 7 * 4 * 4


def test_commercial_gate_key_prefers_passing_epoch_over_lower_macro_failure() -> None:
    passing = {
        "mae": 0.98,
        "identity_macro_mae": 0.99,
        "rmse": 1.75,
        "within_2": 0.91,
        "catastrophic_over_5": 0.029,
        "tail_25_35_mae": 1.95,
    }
    lower_macro_but_failing = {
        "mae": 0.90,
        "identity_macro_mae": 0.90,
        "rmse": 1.79,
        "within_2": 0.87,
        "catastrophic_over_5": 0.045,
        "tail_25_35_mae": 2.20,
    }
    passing_key = trainer.commercial_gate_selection_key(passing)
    failing_key = trainer.commercial_gate_selection_key(lower_macro_but_failing)
    assert passing_key[0] == 0
    assert failing_key[0] == 3
    assert passing_key < failing_key


def test_policy_missed_recall_floor_locks_fail_closed_no_action() -> None:
    prediction = _prediction(
        base_available=np.ones(3, dtype=bool),
        source_available=np.ones(3, dtype=bool),
    )
    prediction.target[:] = 10.0
    prediction.base_prediction[:] = 14.0
    prediction.source_prediction[:] = 10.0
    prediction.valid_candidate_count = np.full(3, 3, np.int64)
    prediction.normalized_entropy = np.zeros(3, np.float32)
    policy, applied = trainer.select_fallback_policy(
        prediction,
        maximum_coverage=0.0,
        maximum_fpr=0.0,
        minimum_precision=1.0,
        minimum_correction_recall=1.0,
        gate_aware=True,
    )
    assert policy.selection_status == "fail_closed_no_action"
    assert policy.promotion_eligible is False
    assert policy.safeguards["correction_recall_at_least_floor"] is False
    assert policy.safeguards["promotion_eligible"] is False
    assert policy.validation_precision == pytest.approx(1.0)
    assert policy.validation_fpr == pytest.approx(0.0)
    assert np.count_nonzero(applied.applied_pull) == 0
    assert np.array_equal(applied.final_prediction, prediction.base_prediction)


def test_identity_weights_give_equal_identity_mass() -> None:
    metadata = pd.DataFrame(
        {"identity": ["A", "A", "A", "B"], "reference_valid": [True] * 4}
    )
    weights = trainer.identity_balanced_weights(metadata, np.arange(4))
    assert weights[:3].sum() == pytest.approx(weights[3:].sum())
    assert weights.sum() == pytest.approx(4.0)


def test_i2_i3_objectives_are_materially_bound_and_warmup_is_configurable() -> None:
    i1 = trainer.resolve_iteration_objective(1)
    i2 = trainer.resolve_iteration_objective(2)
    i3 = trainer.resolve_iteration_objective(3)
    assert (i1.campaign_id, i1.warmup_windows, i1.gradient_accumulation_sessions) == (
        "v2_i1_candidate_graph", 8, 1
    )
    assert (i2.campaign_id, i2.warmup_windows, i2.gradient_accumulation_sessions) == (
        "v2_i2_harmonic_evidence", 2, 4
    )
    assert i2.listwise_temperature_bpm < i1.listwise_temperature_bpm
    assert (i3.campaign_id, i3.tail_weight, i3.cvar_weight) == (
        "v2_i3_causal_posterior_anchor", 2.0, 0.15
    )
    assert (
        i3.anchor_residual_weight,
        i3.anchor_nll_weight,
        i3.anchor_gate_weight,
    ) == pytest.approx((0.75, 0.20, 0.08))
    assert trainer.resolve_iteration_objective(2, warmup_windows=0).warmup_windows == 0


def test_i3_model_and_large_capacity_are_checkpoint_bound_without_i2_drift() -> None:
    historical_i2 = trainer._model_configuration("default", 17)
    assert historical_i2 == {
        "node_features": 17,
        "graph_blocks": 2,
        "hidden_channels": 64,
        "attention_heads": 4,
        "dropout": 0.05,
    }
    i3 = trainer._model_configuration(
        "large",
        17,
        adaptive_iteration=3,
        anchor_residual_mode="causal_posterior",
    )
    assert i3["hidden_channels"] == 96
    assert i3["anchor_enabled"] is True
    assert i3["anchor_max_residual_bpm"] == 12.0
    assert i3["anchor_source_mode"] == "learned_blend"
    with pytest.raises(ValueError, match="only for i3"):
        trainer._model_configuration(
            "default", 17, adaptive_iteration=2,
            anchor_residual_mode="causal_posterior",
        )


def test_gaussian_mixture_and_clipped_residual_reach_non_argmax_candidates() -> None:
    logits = torch.tensor([[[3.0, 1.0, -2.0]]], requires_grad=True)
    candidate_rr = torch.tensor([[[9.6, 10.4, 20.0]]])
    residual = torch.zeros_like(candidate_rr, requires_grad=True)
    scale = torch.ones_like(candidate_rr, requires_grad=True)
    mask = torch.ones_like(candidate_rr, dtype=torch.bool)
    target = torch.tensor([[10.0]])
    nll = trainer.gaussian_candidate_mixture_nll(
        logits, candidate_rr, residual, scale, mask, target
    ).mean()
    nll.backward(retain_graph=True)
    # Candidate 1 is not the hard argmax but its mixture mean, scale and weight
    # all receive supervision.
    assert logits.argmax(dim=-1).item() == 0
    assert abs(float(logits.grad[0, 0, 1])) > 0
    assert abs(float(residual.grad[0, 0, 1])) > 0
    assert abs(float(scale.grad[0, 0, 1])) > 0

    residual.grad.zero_()
    per_row, reachable = trainer.all_candidate_residual_loss(
        candidate_rr, residual, mask, target, maximum_residual_bpm=0.75
    )
    assert reachable.item()
    per_row.sum().backward()
    assert torch.count_nonzero(residual.grad[0, 0, :2]) == 2
    assert residual.grad[0, 0, 2].item() == 0.0


def test_quality_target_describes_selected_source_not_bank_correctability() -> None:
    target = torch.tensor([[10.0, 10.0, 10.0]])
    source = torch.tensor([[15.0, 11.0, 12.5]])
    base = torch.tensor([[10.2, 10.1, 16.0]])
    available = torch.ones_like(target, dtype=torch.bool)
    quality = trainer.quality_supervision_target(
        source, target, base_prediction=base, base_available=available
    )
    # Wrong selected source remains negative even if another bank candidate
    # (not passed here) is correct; selected correctness and true fallback
    # improvement are the only positive cases.
    assert quality.tolist() == [[0.0, 1.0, 1.0]]


def _weighted_loss_logit_gradient(
    row_weights: torch.Tensor, *, tail_weight: float
) -> torch.Tensor:
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    candidate_rr = torch.tensor([[[9.5, 12.0], [29.5, 32.0]]])
    candidate_residual = torch.zeros_like(candidate_rr, requires_grad=True)
    candidate_scale = torch.ones_like(candidate_rr, requires_grad=True)
    target = torch.tensor([[10.0, 30.0]])
    output = {
        "candidate_logits": logits,
        "candidate_residual_bpm": candidate_residual,
        "candidate_scale_bpm": candidate_scale,
        "source_rr": candidate_rr[..., 0],
        "source_scale_bpm": candidate_scale[..., 0],
        "source_available": torch.ones(1, 2, dtype=torch.bool),
        "factor_logits": torch.zeros(1, 2, 4, requires_grad=True),
        "quality_logit": torch.zeros(1, 2, requires_grad=True),
        "spike_rates": torch.full((1, 4), 0.05, requires_grad=True),
    }
    batch = {
        "candidate_rr": candidate_rr,
        "candidate_mask": torch.ones_like(candidate_rr, dtype=torch.bool),
        "target": target,
        "sequence_mask": torch.ones(1, 2, dtype=torch.bool),
        "reference_valid": torch.ones(1, 2, dtype=torch.bool),
        "warmup_mask": torch.zeros(1, 2, dtype=torch.bool),
        "position": torch.tensor([[0, 1]]),
        "classical_rr": target,
        "base_prediction": target,
        "base_available": torch.ones(1, 2, dtype=torch.bool),
    }
    objective = trainer.resolve_iteration_objective(2, tail_weight=tail_weight)
    loss, _ = trainer.compute_multitask_loss(
        output, batch, row_weights, objective=objective,
        normalization_denominator=2.0,
    )
    loss.backward()
    return logits.grad.detach().abs()[0, :, 0]


def test_identity_and_tail_weights_survive_fixed_group_normalization() -> None:
    equal = _weighted_loss_logit_gradient(torch.ones(2), tail_weight=0.0)
    identity = _weighted_loss_logit_gradient(torch.tensor([2.0, 0.5]), tail_weight=0.0)
    tail = _weighted_loss_logit_gradient(torch.ones(2), tail_weight=2.0)
    assert equal[1] / equal[0] == pytest.approx(1.0, rel=1e-5)
    assert identity[0] / identity[1] == pytest.approx(4.0, rel=1e-5)
    assert tail[1] / tail[0] == pytest.approx(3.0, rel=1e-5)


def test_invalid_label_has_zero_loss_but_observation_updates_state() -> None:
    torch.manual_seed(2)
    model = HarmonicCandidateSetEpisodeSNN(node_features=3, hidden_channels=8, attention_heads=1, dropout=0.0)
    node = torch.randn(1, 1, 2, 3)
    rr = torch.tensor([[[10.0, 20.0]]])
    candidate_mask = torch.ones(1, 1, 2, dtype=torch.bool)
    sequence = torch.ones(1, 1, dtype=torch.bool)
    output = model(node, rr, candidate_mask, sequence, radar_mask=torch.ones(1, 1, 3, dtype=torch.bool))
    batch = {
        "candidate_rr": rr, "candidate_mask": candidate_mask,
        "target": torch.tensor([[float("nan")]]), "sequence_mask": sequence,
        "reference_valid": torch.zeros_like(sequence), "warmup_mask": torch.zeros_like(sequence),
        "position": torch.tensor([[0]]), "classical_rr": torch.tensor([[10.0]]),
    }
    loss, _ = trainer.compute_multitask_loss(output, batch, torch.ones(1))
    assert loss.item() == 0.0
    state = output["state"]
    assert any(torch.count_nonzero(tensor).item() for layer in state for tensor in layer)


def test_i3_clipped_anchor_residual_loss_has_identity_weighted_gradients() -> None:
    model = HarmonicCandidateSetEpisodeSNN(
        node_features=3,
        hidden_channels=8,
        attention_heads=1,
        dropout=0.0,
        anchor_enabled=True,
    )
    node = torch.randn(1, 2, 2, 3)
    candidate = torch.tensor([[[10.0, 20.0], [11.0, 22.0]]])
    candidate_mask = torch.ones(1, 2, 2, dtype=torch.bool)
    sequence = torch.ones(1, 2, dtype=torch.bool)
    anchor = torch.tensor([[8.0, 30.0]])
    output = model(
        node, candidate, candidate_mask, sequence,
        radar_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        anchor_rr=anchor,
        anchor_std=torch.ones_like(anchor),
        anchor_available=torch.ones_like(anchor, dtype=torch.bool),
    )
    batch = {
        "node_features": node,
        "candidate_rr": candidate,
        "candidate_mask": candidate_mask,
        "sequence_mask": sequence,
        "radar_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "reset_mask": torch.zeros(1, 2, dtype=torch.bool),
        "target": torch.tensor([[25.0, 10.0]]),
        "reference_valid": sequence,
        "warmup_mask": torch.zeros_like(sequence),
        "position": torch.tensor([[0, 1]]),
        "classical_rr": torch.tensor([[10.0, 11.0]]),
        "base_prediction": anchor,
        "base_std": torch.ones_like(anchor),
        "base_available": torch.ones_like(anchor, dtype=torch.bool),
    }
    objective = trainer.resolve_iteration_objective(3, anchor_enabled=True)
    loss, components = trainer.compute_multitask_loss(
        output,
        batch,
        torch.tensor([2.0, 0.5]),
        objective=objective,
        normalization_denominator=2.5,
    )
    loss.backward()
    assert components["anchor_residual"].item() > 0
    assert components["anchor_nll"].item() > 0
    assert model.anchor_residual_head[-1].bias.grad is not None
    assert abs(float(model.anchor_residual_head[-1].bias.grad)) > 0
    # Both residual targets exceed the architecture bound and are clipped.
    assert output["anchor_residual_limit_bpm"].unique().item() == 12.0


def test_synthetic_identity_disjoint_anchor_residual_improves_unseen_identity() -> None:
    torch.manual_seed(44)
    model = HarmonicCandidateSetEpisodeSNN(
        node_features=2,
        hidden_channels=8,
        attention_heads=1,
        dropout=0.0,
        anchor_enabled=True,
        anchor_source_mode="corrected_anchor",
    )
    # A/B are training identities and C is held out.  Identity labels are
    # intentionally never passed to the model; all share the same causal bias.
    train_node = torch.zeros(2, 4, 2, 2)
    train_candidate = torch.tensor([10.0, 20.0]).reshape(1, 1, 2).expand(2, 4, 2)
    mask = torch.ones(2, 4, 2, dtype=torch.bool)
    sequence = torch.ones(2, 4, dtype=torch.bool)
    train_anchor = torch.tensor([[10.0] * 4, [18.0] * 4])
    train_target = train_anchor + 2.0
    optimizer = torch.optim.Adam(model.anchor_residual_head.parameters(), lr=0.04)
    model.train()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            train_node, train_candidate, mask, sequence,
            radar_mask=torch.ones(2, 4, 3, dtype=torch.bool),
            anchor_rr=train_anchor,
            anchor_std=torch.ones_like(train_anchor),
            anchor_available=torch.ones_like(train_anchor, dtype=torch.bool),
        )
        loss = torch.nn.functional.smooth_l1_loss(
            output["source_rr"], train_target, beta=0.75
        )
        loss.backward()
        optimizer.step()
    model.eval()
    held_anchor = torch.full((1, 4), 25.0)
    with torch.no_grad():
        held = model(
            torch.zeros(1, 4, 2, 2),
            train_candidate[:1], mask[:1], sequence[:1],
            radar_mask=torch.ones(1, 4, 3, dtype=torch.bool),
            anchor_rr=held_anchor,
            anchor_std=torch.ones_like(held_anchor),
            anchor_available=torch.ones_like(held_anchor, dtype=torch.bool),
        )
    raw_mae = torch.full((1,), 2.0).item()
    corrected_mae = (held["source_rr"] - (held_anchor + 2.0)).abs().mean().item()
    assert corrected_mae < 0.5 * raw_mae


def test_structural_fallback_is_bit_exact_and_placeholder_is_not_eligible() -> None:
    prediction = _prediction(
        base_available=np.asarray([False, True]),
        source_available=np.asarray([True, False]),
    )
    prediction.base_prediction[1] = 19.25
    policy = trainer.FallbackPolicy(0, 0, 10, 0, 0.5, 0, 1, 0, 0, {}, 19152)
    applied = trainer.apply_fallback_policy(prediction, policy)
    assert applied.final_prediction[0].tobytes() == prediction.source_prediction[0].tobytes()
    assert applied.final_prediction[1].tobytes() == prediction.base_prediction[1].tobytes()
    assert applied.applied_pull.tolist() == [1.0, 0.0]


def test_corrected_policy_precision_fpr_and_recall_use_final_pull_errors() -> None:
    prediction = _prediction(
        base_available=np.ones(6, dtype=bool),
        source_available=np.ones(6, dtype=bool),
    )
    prediction.target[:] = 10.0
    prediction.base_prediction[:] = [14.0, 14.0, 10.0, 10.0, 10.0, 10.0]
    prediction.source_prediction[:] = [10.0, 10.0, 11.0, 9.0, 12.0, 10.0]
    prediction.selected_probability[:] = 1.0
    prediction.valid_candidate_count = np.full(6, 3, np.int64)
    prediction.normalized_entropy = np.zeros(6, np.float32)
    permissive = trainer.FallbackPolicy(
        0, 0, 1, 0, 1.0, 0, 0, 0, 0, {}, 1
    )
    applied = trainer.apply_fallback_policy(prediction, permissive)
    diagnostics = trainer.correction_policy_diagnostics(prediction, applied)
    assert diagnostics["precision"] == pytest.approx(2 / 6)
    assert diagnostics["recall"] == pytest.approx(1.0)
    # Four eligible base-good rows; three are made materially worse.
    assert diagnostics["fpr"] == pytest.approx(3 / 4)

    # Confidence separates the two truly actionable corrections.  The locked
    # selector must satisfy all three corrected denominators simultaneously.
    prediction.selected_probability[:] = [0.95, 0.95, 0.4, 0.4, 0.4, 0.4]
    policy, selected = trainer.select_fallback_policy(
        prediction,
        maximum_coverage=0.5,
        minimum_precision=1.0,
        minimum_correction_recall=1.0,
        maximum_fpr=0.0,
    )
    locked = trainer.correction_policy_diagnostics(prediction, selected)
    assert policy.validation_precision == pytest.approx(1.0)
    assert policy.validation_recall == pytest.approx(1.0)
    assert policy.validation_fpr == pytest.approx(0.0)
    assert locked["actions"] == 2


def test_policy_context_gates_normalized_entropy_scale_disagreement_and_count() -> None:
    prediction = _prediction(
        base_available=np.ones(2, dtype=bool),
        source_available=np.ones(2, dtype=bool),
    )
    prediction.base_prediction[:] = [8.0, 8.0]
    prediction.base_std[:] = [1.0, 3.0]
    prediction.source_prediction[:] = [10.0, 10.0]
    prediction.source_scale[:] = [0.5, 2.0]
    prediction.normalized_entropy = np.asarray([0.2, 0.8], np.float32)
    prediction.valid_candidate_count = np.asarray([3, 2], np.int64)
    policy = trainer.FallbackPolicy(
        0, 0, 0.5, 0, 1.0, 0, 1, 0, 0, {}, 1,
        2.0, 1.0, 1.0, 3.0, 3, 1.0,
    )
    applied = trainer.apply_fallback_policy(prediction, policy)
    assert applied.applied_pull.tolist() == [1.0, 0.0]


def test_multi_fold_selector_concatenates_before_selecting_one_policy() -> None:
    first = _prediction(
        base_available=np.ones(2, dtype=bool),
        source_available=np.ones(2, dtype=bool),
    )
    second = _prediction(
        base_available=np.ones(2, dtype=bool),
        source_available=np.ones(2, dtype=bool),
    )
    for prediction in (first, second):
        prediction.target[:] = 10.0
        prediction.base_prediction[:] = 14.0
        prediction.source_prediction[:] = 10.0
        prediction.selected_probability[:] = 0.95
        prediction.normalized_entropy = np.zeros(2, np.float32)
        prediction.valid_candidate_count = np.full(2, 3, np.int64)
    policy, combined = trainer.select_fallback_policy_multi(
        [first, second], maximum_coverage=1.0, minimum_precision=1.0,
        minimum_correction_recall=1.0, maximum_fpr=0.0,
    )
    assert len(combined.target) == 4
    assert policy.validation_recall == pytest.approx(1.0)


def test_optimizer_never_steps_mid_carried_session(tmp_path: Path) -> None:
    cache, fallback = _synthetic_cache(tmp_path)
    experiment = trainer.load_experiment(cache, fallback)
    positions = np.arange(10, dtype=np.int64)  # one ten-window physical session
    scaler = trainer.fit_robust_scaler(experiment, positions)
    model = HarmonicCandidateSetEpisodeSNN(
        node_features=6, hidden_channels=8, attention_heads=1, dropout=0.0
    )

    class CountingAdamW(torch.optim.AdamW):
        def __init__(self, parameters):
            super().__init__(parameters, lr=1e-3)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    optimizer = CountingAdamW(model.parameters())
    row_weights = torch.as_tensor(
        trainer.identity_balanced_weights(experiment.metadata, positions)
    )
    losses = trainer.run_training_epoch(
        model, experiment, positions, scaler, optimizer, row_weights,
        torch.device("cpu"), amp=False,
        gradient_scaler=torch.amp.GradScaler("cpu", enabled=False),
        seed=1, epoch=0, adaptive_iteration=2, tail_weight=0.0,
        cvar_weight=0.0, gradient_accumulation_sessions=1,
        warmup_windows=0, chunk_size=2,
    )
    assert optimizer.step_calls == 1
    assert losses["optimizer_steps"] == 1.0


def test_source_decoder_uses_argmax_candidate_not_expectation() -> None:
    torch.manual_seed(3)
    model = HarmonicCandidateSetEpisodeSNN(node_features=2, hidden_channels=8, attention_heads=1, dropout=0.0)
    output = model(
        torch.randn(1, 2, 3, 2),
        torch.tensor([[[8.0, 16.0, 32.0], [9.0, 18.0, 36.0]]]),
        torch.ones(1, 2, 3, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        radar_mask=torch.ones(1, 2, 3, dtype=torch.bool),
    )
    index = output["selected_index"]
    centers = torch.tensor([[[8.0, 16.0, 32.0], [9.0, 18.0, 36.0]]]).gather(-1, index.unsqueeze(-1)).squeeze(-1)
    residual = output["candidate_residual_bpm"].gather(-1, index.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(output["source_rr"], (centers + residual).clamp(6.0, 45.0))


def test_tiny_cpu_discovery_locks_without_opening_test_and_tamper_fails(tmp_path: Path) -> None:
    cache, fallback = _synthetic_cache(tmp_path)
    output = tmp_path / "run"
    result = trainer.train(_args(cache, fallback, output, "--discovery-only"))
    assert result["status"] == "locked"
    assert (output / "selection_lock.json").is_file()
    assert (output / "validation_predictions.npz").is_file()
    assert not (output / "test_predictions.npz").exists()
    with (output / "scaler.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(RuntimeError, match="tamper"):
        trainer.train(_args(cache, fallback, output, "--discovery-only", "--resume"))


def test_i2_tiny_smoke_binds_objective_and_locks_before_test(tmp_path: Path) -> None:
    cache, fallback = _synthetic_cache(tmp_path)
    output = tmp_path / "i2_run"
    result = trainer.train(
        _args(
            cache, fallback, output,
            "--adaptive-iteration", "2",
            "--warmup-windows", "0",
            "--gradient-accumulation-sessions", "2",
            "--discovery-only",
        )
    )
    assert result["status"] == "locked"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    objective = manifest["optimization"]["iteration_objective"]
    assert objective["campaign_id"] == "v2_i2_harmonic_evidence"
    assert objective["warmup_windows"] == 0
    assert objective["gradient_accumulation_sessions"] == 2
    effective = manifest["iteration_effective_configuration"]
    assert manifest["iteration_effective_configuration_sha256"] == trainer.sha256_json(effective)
    bindings = manifest["source_and_config_bindings"]
    assert bindings["harmonic_set_model"]["sha256"] == trainer.sha256_file(
        Path(bindings["harmonic_set_model"]["path"])
    )
    lock = json.loads((output / "selection_lock.json").read_text(encoding="utf-8"))
    assert lock["effective_configuration_sha256"] == manifest["iteration_effective_configuration_sha256"]
    assert lock["source_bindings"] == bindings
    assert not (output / "test_predictions.npz").exists()


def test_i3_tiny_smoke_binds_anchor_allowlist_and_gate_selection(tmp_path: Path) -> None:
    cache, fallback = _synthetic_cache(tmp_path)
    output = tmp_path / "i3_run"
    result = trainer.train(
        _args(
            cache,
            fallback,
            output,
            "--adaptive-iteration", "3",
            "--warmup-windows", "0",
            "--gradient-accumulation-sessions", "2",
            "--discovery-only",
        )
    )
    assert result["status"] == "locked"
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["model_config"]["anchor_enabled"] is True
    allowlist = manifest["iteration_effective_configuration"]["forward_allowlist"]
    assert {"anchor_rr", "anchor_std", "anchor_available"} <= set(allowlist)
    capacity = manifest["iteration_effective_configuration"]["model_capacity"]
    assert capacity["parameter_count"] <= capacity["maximum_parameters"] == 750_000
    assert capacity["hard_limit_enforced_before_training"] is True
    checkpoint = torch.load(output / "best_checkpoint.pt", weights_only=False)
    assert checkpoint["selection_objective"] == trainer.COMMERCIAL_SELECTION_OBJECTIVE
    lock = json.loads((output / "selection_lock.json").read_text())
    assert lock["checkpoint_selection_objective"] == trainer.COMMERCIAL_SELECTION_OBJECTIVE
    assert lock["policy_selection_objective"] == trainer.COMMERCIAL_SELECTION_OBJECTIVE
    with np.load(output / "validation_predictions.npz", allow_pickle=False) as prediction:
        assert "raw_anchor_rr_bpm" in prediction.files
        assert "corrected_anchor_rr_bpm" in prediction.files
        assert "candidate_source_rr_bpm" in prediction.files


def test_model_parameter_limit_fails_before_training(tmp_path: Path) -> None:
    cache, fallback = _synthetic_cache(tmp_path)
    output = tmp_path / "parameter_limit"
    with pytest.raises(RuntimeError, match="model parameter limit exceeded"):
        trainer.train(
            _args(
                cache,
                fallback,
                output,
                "--maximum-parameters", "1",
                "--discovery-only",
            )
        )
    assert not (output / "best_checkpoint.pt").exists()
    assert not (output / "history.json").exists()


def test_recover_prelock_validates_and_locks_without_test_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, fallback = _synthetic_cache(tmp_path)
    output = tmp_path / "prelock"
    original_selector = trainer.select_fallback_policy

    def interrupted_selector(*args, **kwargs):
        raise RuntimeError("synthetic selector interruption")

    monkeypatch.setattr(trainer, "select_fallback_policy", interrupted_selector)
    with pytest.raises(RuntimeError, match="synthetic selector interruption"):
        trainer.train(_args(cache, fallback, output, "--discovery-only"))
    assert (output / "best_checkpoint.pt").is_file()
    assert (output / "history.json").is_file()
    assert not (output / "selection_lock.json").exists()

    monkeypatch.setattr(trainer, "select_fallback_policy", original_selector)
    result = trainer.train(
        _args(cache, fallback, output, "--recover-prelock")
    )
    assert result["recovered_prelock"] is True
    assert (output / "selection_lock.json").is_file()
    assert (output / "validation_predictions.npz").is_file()
    assert (output / "recovery_provenance.json").is_file()
    assert not (output / "test_predictions.npz").exists()


def test_historical_lock_recovery_uses_snapshots_without_rewriting_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, fallback = _synthetic_cache(tmp_path)
    output = tmp_path / "historical_lock"
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    trainer_snapshot = snapshot_root / "train_harmonic_set_snn.py"
    model_snapshot = snapshot_root / "harmonic_set_models.py"
    trainer_snapshot.write_bytes(_SCRIPT_PATH.read_bytes())
    model_source = Path(__file__).resolve().parents[1] / "src/snn_rr/harmonic_set_models.py"
    model_snapshot.write_bytes(model_source.read_bytes())

    source_function = trainer._source_bindings
    actual = source_function()
    historical = {name: dict(binding) for name, binding in actual.items()}
    historical["trainer"]["path"] = str(tmp_path / "changed/train_harmonic_set_snn.py")
    historical["harmonic_set_model"]["path"] = str(tmp_path / "changed/harmonic_set_models.py")
    monkeypatch.setattr(trainer, "_source_bindings", lambda: historical)
    with pytest.raises(RuntimeError, match="source/config tamper"):
        trainer.train(_args(cache, fallback, output, "--discovery-only"))
    lock_before = (output / "selection_lock.json").read_bytes()
    policy_before = (output / "fallback_policy.json").read_bytes()
    checkpoint_before = trainer.sha256_file(output / "best_checkpoint.pt")
    assert not (output / "validation_predictions.npz").exists()

    monkeypatch.setattr(trainer, "_source_bindings", source_function)
    result = trainer.train(
        _args(
            cache,
            fallback,
            output,
            "--recover-prelock",
            "--recovery-source-snapshot-root", str(snapshot_root),
        )
    )
    assert result["recovered_prelock"] is True
    assert (output / "selection_lock.json").read_bytes() == lock_before
    assert (output / "fallback_policy.json").read_bytes() == policy_before
    assert trainer.sha256_file(output / "best_checkpoint.pt") == checkpoint_before
    assert (output / "validation_predictions.npz").is_file()
    provenance = json.loads((output / "recovery_provenance.json").read_text())
    assert provenance["payload_sha256"] == trainer.sha256_json(provenance["payload"])
    resolutions = provenance["payload"]["source_resolution"]
    assert resolutions["trainer"]["resolution"] == "snapshot"
    assert resolutions["harmonic_set_model"]["resolution"] == "snapshot"
    assert resolutions["anchor_disabled_forward_compatibility"]["status"] == "bit_exact"
    assert not (output / "test_predictions.npz").exists()


def test_outer_test_is_written_only_after_existing_lock(tmp_path: Path) -> None:
    cache, fallback = _synthetic_cache(tmp_path)
    output = tmp_path / "run"
    trainer.train(_args(cache, fallback, output, "--discovery-only"))
    result = trainer.train(_args(cache, fallback, output, "--resume"))
    assert result["test_status"] == "evaluated_once"
    assert (output / "selection_lock.json").stat().st_mtime_ns <= (output / "test_predictions.npz").stat().st_mtime_ns
    assert (output / "test_predictions.npz").stat().st_mode & 0o222 == 0
