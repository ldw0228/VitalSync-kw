from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from snn_rr.svd_episode_models import EpisodeAliasRRSNN


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_svd_episode_snn.py"
_SPEC = importlib.util.spec_from_file_location("snn_rr_train_svd_episode", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = TRAIN
_SPEC.loader.exec_module(TRAIN)


def test_commercial_default_uses_latest_strict_cuda_all_window_artifact() -> None:
    args = TRAIN.parse_args([])
    assert args.all_window_base.name == "all_windows_cuda_v3"


def _metadata(identity: str, session: str, fold: int, start_index: int) -> pd.DataFrame:
    rows = 4
    target = np.asarray([16.0, 30.0, 18.0, 21.0], dtype=np.float32)
    valid = np.asarray([False, True, True, False])
    classical = np.asarray([8.0, 10.0, 9.0, 10.5], dtype=np.float32)
    return pd.DataFrame(
        {
            "cache_index": np.arange(start_index, start_index + rows, dtype=np.int64),
            "session_id": [session] * rows,
            "session_number": [fold + 1] * rows,
            "identity": [identity] * rows,
            "protocol": ["synthetic"] * rows,
            "window_number": np.arange(rows),
            "window_start_s": np.arange(rows, dtype=float) * 4.0,
            "window_end_s": np.arange(rows, dtype=float) * 4.0 + 32.0,
            "rr_bpm": target,
            "reference_valid": valid,
            "reference_quality": np.asarray([0.1, 0.9, 0.8, 0.1]),
            "reference_sigma_bpm": np.full(rows, 0.7),
            "classical_rr_bpm": classical,
            "classical_confidence": np.full(rows, 0.8),
            "radar_peak_1_bpm": classical + 0.1,
            "radar_peak_2_bpm": classical - 0.1,
            "radar_peak_3_bpm": classical + 0.2,
            "radar_peak_spread_bpm": np.full(rows, 0.3),
        }
    )


def _write_experiment(root: Path) -> tuple[Path, Path, Path, Path]:
    cache = root / "cache"
    cache.mkdir()
    generator = np.random.default_rng(20260828)
    root_sessions = []
    valid_tables = []
    all_tables = []
    offset = 0
    for fold in range(6):
        session_id = f"S{fold + 1:02d}_P{fold}"
        identity = f"P{fold}"
        session_dir = cache / session_id
        session_dir.mkdir()
        metadata = _metadata(identity, session_id, fold, offset)
        spectra = np.abs(generator.normal(size=(4, 3, 2, 2, 17))).astype(np.float16)
        attributes = generator.uniform(size=(4, 3, 2, 2, 5)).astype(np.float32)
        attributes[..., 4] *= 0.8
        frequencies = np.linspace(0.08, 0.8, 17, dtype=np.float32)
        np.save(session_dir / "spectra.npy", spectra)
        np.save(session_dir / "attributes.npy", attributes)
        np.save(session_dir / "frequencies_hz.npy", frequencies)
        metadata.to_csv(session_dir / "metadata.csv", index=False)
        manifest = {
            "session_id": session_id,
            "row_count": 4,
            "valid_only": False,
            "label_inputs": [],
            "spectra_shape": list(spectra.shape),
            "attributes_shape": list(attributes.shape),
        }
        (session_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        root_sessions.append({"session_id": session_id, "status": "ok"})
        valid = metadata[metadata["reference_valid"]].copy()
        valid["fold"] = fold
        valid["prediction_bpm"] = valid["rr_bpm"] + 0.6
        valid["rr_std_bpm"] = 1.0
        valid_tables.append(valid)
        deployment = metadata.loc[
            :,
            [
                "cache_index",
                "session_id",
                "identity",
                "protocol",
                "window_number",
                "window_start_s",
                "window_end_s",
                "rr_bpm",
                "reference_valid",
            ],
        ].copy()
        deployment["fold"] = fold
        deployment["prediction_bpm"] = metadata["rr_bpm"] + 0.7
        deployment["rr_std_bpm"] = 1.1
        deployment["alias_probability"] = 0.2
        all_tables.append(deployment)
        offset += 4
    root_manifest = {
        "valid_only": False,
        "label_inputs": [],
        "row_count": offset,
        "pipeline_sha256": "synthetic",
        "sessions": root_sessions,
    }
    (cache / "manifest.json").write_text(json.dumps(root_manifest), encoding="utf-8")
    base = pd.concat(valid_tables, ignore_index=True)
    base_csv = root / "base.csv"
    base.to_csv(base_csv, index=False)
    alias_csv = root / "alias.csv"
    base.assign(alias_probability=0.2).to_csv(alias_csv, index=False)
    all_csv = root / "all.csv"
    pd.concat(all_tables, ignore_index=True).to_csv(all_csv, index=False)
    (root / "fold_assignments.json").write_text(
        json.dumps(
            {
                "identity_to_fold": {f"P{fold}": fold for fold in range(6)},
                "validation_rule": "(outer_fold + 1) % 6",
            }
        ),
        encoding="utf-8",
    )
    return cache, base_csv, alias_csv, all_csv


def _tiny_run_arguments(
    cache: Path, base: Path, alias: Path, all_csv: Path, output: Path
) -> list[str]:
    return [
        "--svd-cache",
        str(cache),
        "--base-oof-csv",
        str(base),
        "--alias-oof-csv",
        str(alias),
        "--all-window-base",
        str(all_csv),
        "--output-dir",
        str(output),
        "--fold",
        "0",
        "--preset",
        "tiny",
        "--epochs",
        "1",
        "--minimum-epochs",
        "1",
        "--source-warmup-epochs",
        "0",
        "--patience",
        "1",
        "--workers",
        "0",
        "--device",
        "cpu",
        "--no-amp",
        "--no-verify-file-hashes",
        "--bootstrap-samples",
        "20",
    ]


def test_candidate_evidence_is_label_free_and_finite() -> None:
    assert "target" not in inspect.signature(TRAIN.compute_candidate_evidence).parameters
    generator = np.random.default_rng(7)
    spectra = np.abs(generator.normal(size=(3, 3, 2, 2, 17))).astype(np.float32)
    attributes = generator.uniform(size=(3, 3, 2, 2, 5)).astype(np.float32)
    attributes[..., 4] *= 0.8
    frequency = np.linspace(0.08, 0.8, 17, dtype=np.float32)
    classical = np.asarray([8.0, 10.0, 12.0], dtype=np.float32)
    peaks = np.stack((classical, classical + 0.2, classical - 0.2), axis=1)
    evidence, mask = TRAIN.compute_candidate_evidence(
        spectra, attributes, frequency, classical, peaks
    )
    assert evidence.shape == (3, 3, 4, len(TRAIN.EVIDENCE_NAMES))
    assert mask.shape == (3, 3)
    assert np.isfinite(evidence).all()
    assert np.all((evidence >= 0) & (evidence <= 1))


def test_all_window_loader_and_split_keep_every_session_in_one_partition(
    tmp_path: Path,
) -> None:
    cache, base, alias, all_csv = _write_experiment(tmp_path)
    experiment = TRAIN.load_episode_experiment(
        cache, base, alias, all_csv, verify_file_hashes=False
    )
    assert len(experiment.metadata) == 24
    assert experiment.metadata["reference_valid"].sum() == 12
    assert experiment.metadata["_base_prediction"].notna().all()
    valid = experiment.metadata["reference_valid"].astype(bool)
    np.testing.assert_allclose(
        experiment.metadata.loc[valid, "_base_prediction"],
        experiment.metadata.loc[valid, "rr_bpm"] + 0.6,
    )
    np.testing.assert_allclose(
        experiment.metadata.loc[~valid, "_base_prediction"],
        experiment.metadata.loc[~valid, "rr_bpm"] + 0.7,
    )
    assert experiment.provenance["all_window_prediction_status"].startswith("loaded")
    split = TRAIN.make_episode_split(experiment, 0)
    assert len(split.train_identities) == 4
    assert split.validation_identities == ("P1",)
    assert split.test_identities == ("P0",)
    for session in experiment.sessions:
        position = session.metadata["_position"].to_numpy(int)
        memberships = [
            np.isin(position, split.train).all(),
            np.isin(position, split.validation).all(),
            np.isin(position, split.test).all(),
        ]
        assert sum(memberships) == 1


def test_primary_forward_boundary_has_no_reference_or_qc_inputs() -> None:
    parameters = set(inspect.signature(EpisodeAliasRRSNN.forward).parameters)
    forbidden = {
        "rr",
        "target",
        "reference_valid",
        "reference_quality",
        "reference_sigma",
        "radar_observable",
    }
    assert not parameters & forbidden


def test_multitask_loss_uses_invalid_only_under_explicit_weak_weight() -> None:
    torch.manual_seed(4)
    model = EpisodeAliasRRSNN(
        evidence_features=10,
        context_features=3,
        candidate_channels=4,
        hidden_channels=8,
        cell_types=("lif",),
        dropout=0,
    )
    batch = {
        "evidence": torch.rand(1, 4, 3, 4, 10),
        "context": torch.rand(1, 4, 3),
        "classical_rr": torch.tensor([[8.0, 10.0, 9.0, 10.5]]),
        "base_prediction": torch.tensor([[16.5, 30.5, 18.5, 21.5]]),
        "base_std": torch.ones(1, 4),
        "base_alias_probability": torch.zeros(1, 4),
        "base_available": torch.ones(1, 4, dtype=torch.bool),
        "radar_mask": torch.ones(1, 4, 3, dtype=torch.bool),
        "sequence_mask": torch.ones(1, 4, dtype=torch.bool),
        "rr": torch.tensor([[16.0, 30.0, 18.0, 21.0]]),
        "reference_valid": torch.tensor([[False, True, True, False]]),
        "reference_quality": torch.tensor([[0.1, 0.9, 0.8, 0.1]]),
        "reference_sigma": torch.full((1, 4), 0.7),
    }
    output = TRAIN.forward_episode_model(model, batch, torch.device("cpu"))
    calibration = TRAIN.TrainActionCalibration(0.75, 0.75, 0.5, "train", 2)
    strict_loss, strict = TRAIN.compute_episode_multitask_loss(
        output, batch, model.rr_bins, action_calibration=calibration
    )
    weak_loss, weak = TRAIN.compute_episode_multitask_loss(
        output,
        batch,
        model.rr_bins,
        action_calibration=calibration,
        weak_invalid_weight=0.1,
    )
    assert torch.isfinite(strict_loss) and torch.isfinite(weak_loss)
    assert int(strict["strict_valid_rows"]) == 2
    assert int(strict["weak_invalid_rows"]) == 0
    assert int(weak["weak_invalid_rows"]) == 2
    strict_loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_strict_gate_amp_safe_gradient_and_weak_quality_filter() -> None:
    torch.manual_seed(9)
    model = EpisodeAliasRRSNN(
        evidence_features=10,
        context_features=3,
        candidate_channels=4,
        hidden_channels=8,
        cell_types=("lif",),
        dropout=0,
        strict_alias_gate=True,
    )
    batch = {
        "evidence": torch.rand(1, 4, 3, 4, 10),
        "context": torch.rand(1, 4, 3),
        "classical_rr": torch.tensor([[8.0, 10.0, 9.0, 10.5]]),
        "base_prediction": torch.tensor([[40.0, 7.0, 35.0, 9.0]]),
        "base_std": torch.ones(1, 4),
        "base_alias_probability": torch.tensor([[0.9, 0.1, 0.8, 0.2]]),
        "base_available": torch.ones(1, 4, dtype=torch.bool),
        "radar_mask": torch.ones(1, 4, 3, dtype=torch.bool),
        "sequence_mask": torch.ones(1, 4, dtype=torch.bool),
        "rr": torch.tensor([[16.0, 30.0, 9.0, 21.0]]),
        "reference_valid": torch.tensor([[False, True, True, False]]),
        "reference_quality": torch.tensor([[0.8, 0.9, 0.8, 0.2]]),
        "reference_sigma": torch.full((1, 4), 0.7),
    }
    calibration = TRAIN.TrainActionCalibration(0.75, 0.75, 0.5, "train", 2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = TRAIN.forward_episode_model(model, batch, torch.device("cpu"))
        loss, parts = TRAIN.compute_episode_multitask_loss(
            output,
            batch,
            model.rr_bins,
            action_calibration=calibration,
            weak_invalid_weight=0.1,
            weak_invalid_min_quality=0.5,
            strict_alias_gate=True,
            strict_gate_margin_bpm=0.75,
            strict_gate_false_positive_cost=10.0,
        )
        altered = {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in batch.items()
        }
        altered["base_prediction"].fill_(44.0)
        altered["base_std"].fill_(7.0)
        altered["base_alias_probability"].fill_(0.99)
        altered["base_available"].zero_()
        altered_output = TRAIN.forward_episode_model(
            model, altered, torch.device("cpu")
        )
        altered_loss, altered_parts = TRAIN.compute_episode_multitask_loss(
            altered_output,
            altered,
            model.rr_bins,
            action_calibration=calibration,
            weak_invalid_weight=0.1,
            weak_invalid_min_quality=0.5,
            strict_alias_gate=True,
            strict_gate_margin_bpm=0.75,
            strict_gate_false_positive_cost=10.0,
        )
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, altered_loss, rtol=0, atol=0)
    torch.testing.assert_close(
        parts["gate_bce"], altered_parts["gate_bce"], rtol=0, atol=0
    )
    assert int(parts["weak_invalid_quality_eligible_rows"]) == 1
    assert int(parts["weak_invalid_rows"]) == 1
    loss.backward()
    gate_gradient = model.gate_head[-1].weight.grad
    assert gate_gradient is not None
    assert torch.isfinite(gate_gradient).all()
    assert torch.count_nonzero(gate_gradient) > 0


def test_divisor_balance_is_train_only_and_default_is_backward_compatible() -> None:
    frame = pd.DataFrame(
        {
            "reference_valid": np.ones(10, dtype=bool),
            "rr_bpm": [10.0] * 8 + [20.0] * 2,
            "classical_rr_bpm": [10.0] * 10,
            "_base_prediction": [10.0] * 10,
        }
    )
    positions = np.arange(10, dtype=np.int64)
    default = TRAIN.fit_train_action_calibration(frame, positions)
    balanced = TRAIN.fit_train_action_calibration(
        frame, positions, divisor_class_balance_power=0.5
    )
    assert default.divisor_class_weights == (1.0, 1.0, 1.0, 1.0)
    assert balanced.divisor_class_counts[:2] == (8, 2)
    assert balanced.divisor_class_weights[1] > balanced.divisor_class_weights[0]
    assert balanced.fit_positions_sha256 == TRAIN._positions_digest(positions)


def test_locked_strict_gate_policy_applies_mixture_and_structural_fallbacks() -> None:
    rows = 4
    result = TRAIN.EpisodePrediction(
        position=np.arange(rows, dtype=np.int64),
        cache_index=np.arange(100, 104, dtype=np.int64),
        target=np.asarray([10.0, 20.0, 30.0, 15.0], np.float32),
        # Row 2 reproduces EpisodeDataset's sanitized missing-base values.
        # The explicit availability bit, not finite 0/4 placeholders, must
        # select the source fallback.
        base_prediction=np.asarray([12.0, 18.0, 0.0, 15.0], np.float32),
        base_std=np.asarray([1.0, 1.0, 4.0, 1.0], np.float32),
        base_available=np.asarray([True, True, False, True]),
        candidate_prediction=np.zeros(rows, np.float32),
        rr_std=np.ones(rows, np.float32),
        source_prediction=np.asarray([10.0, 20.0, 30.0, 20.0], np.float32),
        source_std=np.ones(rows, np.float32),
        mixture_gate=np.zeros(rows, np.float32),
        learned_gate=np.asarray([0.9, 0.8, 0.7, 0.1], np.float32),
        applied_gate=np.zeros(rows, np.float32),
        divisor_probabilities=np.full((rows, 4), 0.25, np.float32),
        residual_rr=np.zeros(rows, np.float32),
        candidate_std=np.ones((rows, 4), np.float32),
        quality=np.ones(rows, np.float32),
        radar_weights=np.ones((rows, 3), np.float32) / 3,
        spike_rate=np.zeros(rows, np.float32),
    )
    result.radar_weights[1] = 0.0
    policy = TRAIN.StrictGatePolicy(0.5, 0.5, 0.0, 0.0, None, {}, 1)
    applied = TRAIN.apply_strict_gate_policy(result, policy)
    assert applied.candidate_prediction[0] == pytest.approx(11.0)
    assert applied.candidate_prediction[1] == pytest.approx(18.0)  # no radar => base
    assert applied.candidate_prediction[2] == pytest.approx(30.0)  # no base => source
    assert applied.candidate_prediction[3] == pytest.approx(15.0)
    assert applied.learned_gate[0] == pytest.approx(0.9)
    assert applied.applied_gate.tolist() == pytest.approx([0.5, 0.0, 1.0, 0.0])


def test_validation_only_gate_selection_returns_a_sparse_mixture_candidate() -> None:
    rows = 8
    target = np.asarray([10.0, 20.0, 30.0, 15.0, 12.0, 24.0, 28.0, 18.0], np.float32)
    base = target + np.asarray([0.2, 3.0, 5.0, 0.2, 0.2, 2.5, 4.0, 0.3], np.float32)
    source = target + np.asarray([2.0, 0.4, 0.7, 2.0, 1.5, 0.4, 0.5, 1.5], np.float32)
    result = TRAIN.EpisodePrediction(
        position=np.arange(rows, dtype=np.int64),
        cache_index=np.arange(rows, dtype=np.int64),
        target=target,
        base_prediction=base,
        base_std=np.ones(rows, np.float32),
        base_available=np.ones(rows, dtype=bool),
        candidate_prediction=source.copy(),
        rr_std=np.ones(rows, np.float32),
        source_prediction=source,
        source_std=np.ones(rows, np.float32),
        mixture_gate=np.zeros(rows, np.float32),
        learned_gate=np.asarray([0.01, 0.9, 0.95, 0.02, 0.03, 0.85, 0.92, 0.04], np.float32),
        applied_gate=np.zeros(rows, np.float32),
        divisor_probabilities=np.full((rows, 4), 0.25, np.float32),
        residual_rr=np.zeros(rows, np.float32),
        candidate_std=np.ones((rows, 4), np.float32),
        quality=np.ones(rows, np.float32),
        radar_weights=np.ones((rows, 3), np.float32) / 3,
        spike_rate=np.zeros(rows, np.float32),
    )
    metadata = pd.DataFrame(
        {
            "identity": ["A"] * 4 + ["B"] * 4,
            "classical_rr_bpm": target,
        }
    )
    policy, selected = TRAIN.select_strict_gate_policy(
        result, metadata, maximum_coverage=0.5
    )
    assert policy.validation_coverage <= 0.5
    np.testing.assert_allclose(
        selected.candidate_prediction,
        base + selected.applied_gate * (source - base),
        atol=1e-6,
    )
    assert np.any(selected.applied_gate > 0)
    assert np.any(selected.applied_gate == 0)


def test_gate_selection_excludes_sanitized_missing_base_from_coverage() -> None:
    result = TRAIN.EpisodePrediction(
        position=np.arange(2, dtype=np.int64),
        cache_index=np.arange(2, dtype=np.int64),
        target=np.asarray([30.0, 31.0], np.float32),
        base_prediction=np.asarray([30.0, 0.0], np.float32),
        base_std=np.asarray([1.0, 4.0], np.float32),
        base_available=np.asarray([True, False]),
        candidate_prediction=np.asarray([32.0, 31.0], np.float32),
        rr_std=np.ones(2, np.float32),
        source_prediction=np.asarray([32.0, 31.0], np.float32),
        source_std=np.ones(2, np.float32),
        mixture_gate=np.zeros(2, np.float32),
        learned_gate=np.ones(2, np.float32),
        applied_gate=np.zeros(2, np.float32),
        divisor_probabilities=np.full((2, 4), 0.25, np.float32),
        residual_rr=np.zeros(2, np.float32),
        candidate_std=np.ones((2, 4), np.float32),
        quality=np.ones(2, np.float32),
        radar_weights=np.ones((2, 3), np.float32) / 3,
        spike_rate=np.zeros(2, np.float32),
    )
    metadata = pd.DataFrame(
        {"identity": ["A", "B"], "classical_rr_bpm": [30.0, 31.0]}
    )
    policy, selected = TRAIN.select_strict_gate_policy(
        result, metadata, maximum_coverage=0.5
    )
    assert policy.validation_coverage == pytest.approx(0.0)
    # Row 0 is the sole correction-eligible row and must remain on its exact base.
    # Row 1 is an unconditional source fallback and is not policy coverage.
    assert selected.applied_gate.tolist() == pytest.approx([0.0, 1.0])
    assert selected.candidate_prediction.tolist() == pytest.approx([30.0, 31.0])


def test_gate_modes_are_mutually_exclusive_and_defaults_remain_off() -> None:
    defaults = TRAIN.parse_args([])
    assert defaults.strict_alias_gate is False
    assert defaults.allow_stacked_base_training is False
    assert defaults.divisor_class_balance_power == 0
    assert defaults.weak_invalid_min_quality == 0
    with pytest.raises(SystemExit):
        TRAIN.parse_args(["--strict-alias-gate", "--allow-stacked-base-training"])


def test_tiny_cpu_smoke_writes_lock_before_single_test_evaluation(tmp_path: Path) -> None:
    cache, base, alias, all_csv = _write_experiment(tmp_path)
    output = tmp_path / "run"
    code = TRAIN.main(
        [
            "--svd-cache",
            str(cache),
            "--base-oof-csv",
            str(base),
            "--alias-oof-csv",
            str(alias),
            "--all-window-base",
            str(all_csv),
            "--output-dir",
            str(output),
            "--fold",
            "0",
            "--preset",
            "tiny",
            "--epochs",
            "1",
            "--minimum-epochs",
            "1",
            "--source-warmup-epochs",
            "0",
            "--patience",
            "1",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "1",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--no-amp",
            "--no-verify-file-hashes",
            "--bootstrap-samples",
            "20",
        ]
    )
    assert code == 0
    fold = output / "fold_0"
    lock = json.loads((fold / "selection_lock.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (fold / "test_evaluation_manifest.json").read_text(encoding="utf-8")
    )
    run = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert manifest["test_fold_evaluation_invocations"] == 1
    assert manifest["selection_lock_sha256"] == hashlib.sha256(
        (fold / "selection_lock.json").read_bytes()
    ).hexdigest()
    assert lock["strict_nested_base_unavailable"] is True
    assert run["stacking_audit"]["strict_nested_base_unavailable"] is True
    assert run["weak_invalid_label_ablation"]["enabled"] is False


def test_completed_resume_rejects_tampered_selection_lock(tmp_path: Path) -> None:
    cache, base, alias, all_csv = _write_experiment(tmp_path)
    output = tmp_path / "resume_run"
    arguments = [
        "--svd-cache",
        str(cache),
        "--base-oof-csv",
        str(base),
        "--alias-oof-csv",
        str(alias),
        "--all-window-base",
        str(all_csv),
        "--output-dir",
        str(output),
        "--fold",
        "0",
        "--preset",
        "tiny",
        "--epochs",
        "1",
        "--minimum-epochs",
        "1",
        "--source-warmup-epochs",
        "0",
        "--patience",
        "1",
        "--workers",
        "0",
        "--device",
        "cpu",
        "--no-amp",
        "--no-verify-file-hashes",
        "--bootstrap-samples",
        "20",
    ]
    assert TRAIN.main(arguments) == 0
    assert TRAIN.main([*arguments, "--resume"]) == 0
    lock_path = output / "fold_0" / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["best_epoch"] = 999
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(RuntimeError, match="selection-lock hash mismatch"):
        TRAIN.main([*arguments, "--resume"])


def test_strict_gate_smoke_locks_policy_and_saves_distinct_gate_fields(
    tmp_path: Path,
) -> None:
    cache, base, alias, all_csv = _write_experiment(tmp_path)
    output = tmp_path / "strict_run"
    assert (
        TRAIN.main(
            [
                "--svd-cache",
                str(cache),
                "--base-oof-csv",
                str(base),
                "--alias-oof-csv",
                str(alias),
                "--all-window-base",
                str(all_csv),
                "--output-dir",
                str(output),
                "--fold",
                "0",
                "--preset",
                "tiny",
                "--epochs",
                "1",
                "--minimum-epochs",
                "1",
                "--source-warmup-epochs",
                "0",
                "--patience",
                "1",
                "--workers",
                "0",
                "--device",
                "cpu",
                "--no-amp",
                "--no-verify-file-hashes",
                "--strict-alias-gate",
                "--divisor-class-balance-power",
                "0.5",
                "--bootstrap-samples",
                "20",
            ]
        )
        == 0
    )
    fold = output / "fold_0"
    lock = json.loads((fold / "selection_lock.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert lock["strict_alias_gate_policy"] is not None
    assert lock["strict_alias_gate_policy"]["validation_coverage"] <= 0.15
    assert "no base error" in run["stacking_audit"]["strict_alias_gate_target"]
    with np.load(fold / "test_predictions.npz", allow_pickle=False) as values:
        assert {"learned_gate", "applied_gate"}.issubset(values.files)
        expected = values["base_prediction"] + values["applied_gate"] * (
            values["source_prediction"] - values["base_prediction"]
        )
        np.testing.assert_allclose(values["candidate_prediction"], expected, atol=1e-5)


@pytest.mark.parametrize("tampered_source", ["base_fold", "alias_row", "all_window_row"])
def test_semantic_bindings_reject_same_set_wrong_row_meaning(
    tmp_path: Path, tampered_source: str
) -> None:
    root = tmp_path / tampered_source
    root.mkdir()
    cache, base_path, alias_path, all_path = _write_experiment(root)
    if tampered_source == "base_fold":
        frame = pd.read_csv(base_path)
        frame.loc[frame["identity"] == "P0", "fold"] = 1
        frame.loc[frame["identity"] == "P1", "fold"] = 0
        frame.to_csv(base_path, index=False)
        match = "independent frozen assignment"
    else:
        path = alias_path if tampered_source == "alias_row" else all_path
        frame = pd.read_csv(path)
        first_session = frame["session_id"].iloc[0]
        rows = frame.index[frame["session_id"] == first_session][:2].to_numpy()
        frame.loc[rows, "cache_index"] = frame.loc[rows[::-1], "cache_index"].to_numpy()
        frame.to_csv(path, index=False)
        match = "semantic row binding mismatch"
    with pytest.raises(RuntimeError, match=match):
        TRAIN.load_episode_experiment(
            cache, base_path, alias_path, all_path, verify_file_hashes=False
        )


def test_partial_all_window_binding_is_reported_as_partial_not_complete(
    tmp_path: Path,
) -> None:
    cache, base, alias, all_path = _write_experiment(tmp_path)
    pd.read_csv(all_path).iloc[:1].to_csv(all_path, index=False)
    experiment = TRAIN.load_episode_experiment(
        cache, base, alias, all_path, verify_file_hashes=False
    )
    assert experiment.provenance["all_window_supplied_rows_binding_exact"] is True
    assert experiment.provenance["all_window_complete_exact"] is False
    assert experiment.provenance["all_window_index_fold_binding_exact"] is False
    assert "partial" in experiment.provenance["all_window_prediction_status"]


def test_metadata_input_change_invalidates_completed_resume_even_without_tensor_hashes(
    tmp_path: Path,
) -> None:
    cache, base, alias, all_path = _write_experiment(tmp_path)
    output = tmp_path / "metadata_bound_run"
    arguments = _tiny_run_arguments(cache, base, alias, all_path, output)
    assert TRAIN.main(arguments) == 0
    metadata_path = cache / "S01_P0" / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    metadata["classical_confidence"] += 0.125
    metadata.to_csv(metadata_path, index=False)
    with pytest.raises(RuntimeError, match="run_config does not match"):
        TRAIN.main([*arguments, "--resume"])


def test_completed_and_partial_resume_bind_the_committed_best_checkpoint(
    tmp_path: Path,
) -> None:
    cache, base, alias, all_path = _write_experiment(tmp_path)
    output = tmp_path / "checkpoint_bound_run"
    arguments = _tiny_run_arguments(cache, base, alias, all_path, output)
    assert TRAIN.main(arguments) == 0
    fold = output / "fold_0"
    best_path = fold / "episode_best.pt"
    best_bytes = best_path.read_bytes()
    last = torch.load(fold / "episode_last.pt", map_location="cpu", weights_only=False)
    assert last["committed_best_checkpoint_sha256"] == hashlib.sha256(
        best_bytes
    ).hexdigest()

    best_path.unlink()
    with pytest.raises(RuntimeError, match="selected checkpoint is missing"):
        TRAIN.main([*arguments, "--resume"])

    best_path.write_bytes(best_bytes)
    for name in (
        "test_evaluation_manifest.json",
        "test_predictions.npz",
        "selection_lock.json",
        "test_predictions.json",
        "test_evaluation_started.json",
    ):
        (fold / name).unlink(missing_ok=True)
    best_path.write_bytes(b"interrupted-best-replacement")
    with pytest.raises(RuntimeError, match="committed best checkpoint SHA-256 mismatch"):
        TRAIN.main([*arguments, "--resume"])
