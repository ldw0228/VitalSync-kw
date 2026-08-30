from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_svd_snn.py"
_SPEC = importlib.util.spec_from_file_location("snn_rr_train_svd_script", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_TRAIN = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TRAIN
_SPEC.loader.exec_module(_TRAIN)


def _synthetic_cache(tmp_path: Path, *, corrupt_npz_target: bool = False) -> tuple[Path, Path, Path]:
    cache_root = tmp_path / "svd"
    session_dir = cache_root / "SESSION"
    session_dir.mkdir(parents=True)
    count = 12
    cache_index = np.arange(100, 100 + count, dtype=np.int64)
    identity = np.repeat([f"P{index}" for index in range(6)], 2)
    fold = np.repeat(np.arange(6), 2)
    target = np.linspace(10.0, 31.0, count, dtype=np.float32)
    metadata = pd.DataFrame(
        {
            "cache_index": cache_index,
            "session_id": "SESSION",
            "session_number": 1,
            "identity": identity,
            "protocol": np.where(np.arange(count) % 2, "B", "A"),
            "window_number": np.arange(count),
            "window_start_s": np.arange(count, dtype=float) * 4,
            "window_end_s": np.arange(count, dtype=float) * 4 + 32,
            "rr_bpm": target,
            "reference_valid": True,
            "reference_quality": 0.8,
            "reference_sigma_bpm": 0.7,
            "classical_rr_bpm": target - 0.5,
            "classical_confidence": 0.75,
            "radar_observable": True,
            "radar_peak_1_bpm": target - 1,
            "radar_peak_2_bpm": target,
            "radar_peak_3_bpm": target + 1,
            "radar_peak_spread_bpm": 0.5,
        }
    )
    metadata.to_csv(session_dir / "metadata.csv", index=False)
    generator = np.random.default_rng(7)
    spectra = generator.random((count, 3, 2, 3, 21), dtype=np.float32)
    spectra /= spectra.sum(axis=-1, keepdims=True)
    attributes = generator.random((count, 3, 2, 3, 5), dtype=np.float32)
    np.save(session_dir / "spectra.npy", spectra.astype(np.float16))
    np.save(session_dir / "attributes.npy", attributes)
    np.save(session_dir / "frequencies_hz.npy", np.linspace(0.08, 0.80, 21, dtype=np.float32))
    (session_dir / "manifest.json").write_text(
        json.dumps({"row_count": count, "spectra_shape": list(spectra.shape)}),
        encoding="utf-8",
    )
    (cache_root / "manifest.json").write_text(
        json.dumps(
            {
                "valid_only": True,
                "row_count": count,
                "sessions": [{"session_id": "SESSION", "status": "ok"}],
            }
        ),
        encoding="utf-8",
    )
    base_prediction = target + np.tile(np.asarray([1.0, -1.0], dtype=np.float32), 6)
    oof = metadata.loc[
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
        ],
    ].copy()
    oof["fold"] = fold
    oof["prediction_bpm"] = base_prediction
    oof["prediction_uncalibrated_bpm"] = base_prediction + 0.1
    oof["rr_std_bpm"] = 1.0
    oof_csv = tmp_path / "base.csv"
    oof_npz = tmp_path / "base.npz"
    oof.to_csv(oof_csv, index=False)
    np.savez_compressed(
        oof_npz,
        index=cache_index,
        target=target + (1.0 if corrupt_npz_target else 0.0),
        prediction=base_prediction,
        rr_std=np.ones(count, dtype=np.float32),
        fold=fold,
    )
    return cache_root, oof_csv, oof_npz


def test_source_cache_oof_binding_and_frozen_split_are_exact(tmp_path: Path) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    experiment = _TRAIN.load_aligned_experiment(
        cache, csv_path, npz_path, verify_file_hashes=False
    )
    assert len(experiment.metadata) == 12
    assert experiment.sessions[0].spectra.shape == (12, 3, 2, 3, 21)
    assert experiment.provenance["component_signals_status"].endswith("not_loaded_or_input")

    split = _TRAIN.make_outer_split(experiment.metadata, 2)
    assert split.validation_fold == 3
    assert set(experiment.metadata.iloc[split.test]["fold"]) == {2}
    assert set(experiment.metadata.iloc[split.validation]["fold"]) == {3}
    assert set(experiment.metadata.iloc[split.train]["fold"]) == {0, 1, 4, 5}
    assert not (set(split.train_identities) & set(split.test_identities))


def test_oof_npz_target_tampering_is_rejected(tmp_path: Path) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path, corrupt_npz_target=True)
    with pytest.raises(RuntimeError, match="NPZ target binding mismatch"):
        _TRAIN.load_aligned_experiment(
            cache, csv_path, npz_path, verify_file_hashes=False
        )


def test_input_allowlist_blocks_target_identity_and_unlisted_values() -> None:
    assert _TRAIN.validate_input_feature_columns(
        ["prediction_uncalibrated_bpm", "quality"]
    ) == ("prediction_uncalibrated_bpm", "quality")
    for columns in (["rr_bpm"], ["identity"], ["future_signal"], ["quality", "quality"]):
        with pytest.raises(ValueError):
            _TRAIN.validate_input_feature_columns(columns)


def test_scaler_fits_training_rows_only() -> None:
    frame = pd.DataFrame({"quality": [0.0, 1.0, 2.0, 1000.0]})
    scaler = _TRAIN.fit_robust_scaler(frame, [0, 1, 2], ["quality"])
    np.testing.assert_allclose(scaler.center, [1.0])
    # The held-out extreme is transformed, but cannot alter fitted state.
    assert scaler.transform(frame)[-1, 0] == 12.0


def test_identity_rr_tail_sampler_equalizes_identity_mass_and_boosts_tail() -> None:
    frame = pd.DataFrame(
        {
            "identity": ["A"] * 5 + ["B"] * 3,
            "rr_bpm": [10, 11, 12, 13, 30, 10, 11, 30],
        }
    )
    weights = _TRAIN.identity_rr_tail_sample_weights(
        frame,
        np.arange(len(frame)),
        rr_balance_power=0.0,
        tail_boost=3.0,
    )
    np.testing.assert_allclose(weights[:5].sum(), weights[5:].sum())
    assert weights[4] > weights[0]
    assert weights[7] > weights[5]


def _model_batch() -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    torch.manual_seed(17)
    model = _TRAIN.SourceSeparatedRRSNN(
        num_variants=2,
        num_components=3,
        base_feature_dim=2,
        encoder_channels=12,
        encoder_blocks=1,
        hidden_channels=16,
        num_spiking_blocks=1,
        simulation_steps=2,
        radar_dropout_p=0.0,
        spectral_frequency_min_hz=0.08,
        spectral_frequency_max_hz=0.80,
    )
    spectra = torch.rand(3, 3, 2, 3, 21)
    spectra /= spectra.sum(dim=-1, keepdim=True)
    batch = {
        "spectra": spectra,
        "attributes": torch.rand(3, 3, 2, 3, 5),
        "base_prediction": torch.tensor([12.0, 18.0, 27.0]),
        "base_std": torch.tensor([1.0, 1.2, 1.5]),
        "base_features": torch.randn(3, 2),
        "classical_rr": torch.tensor(
            [[12.0, 11.0, 12.0, 13.0], [9.0, 8.5, 9.0, 9.5], [7.0, 6.5, 7.0, 7.5]]
        ),
        "classical_std": torch.ones(3, 4),
        "radar_mask": torch.ones(3, 3, dtype=torch.bool),
        "rr": torch.tensor([12.2, 18.4, 28.0]),
        "reference_valid": torch.ones(3, dtype=torch.bool),
        "reference_quality": torch.tensor([0.9, 0.8, 0.7]),
        "reference_sigma": torch.tensor([0.5, 0.7, 0.8]),
        "observable": torch.tensor([1.0, 1.0, 0.0]),
    }
    return model, batch


def test_svd_multitask_loss_contains_requested_terms_and_backpropagates() -> None:
    model, batch = _model_batch()
    output = _TRAIN.forward_model(model, batch, torch.device("cpu"))
    loss, components = _TRAIN.compute_svd_multitask_loss(
        output, batch, model.rr_bins
    )
    assert set(
        [
            "distribution",
            "source_distribution",
            "huber",
            "source_huber",
            "uncertainty_nll",
            "gate_bce",
            "action_regret",
            "spike_rate",
        ]
    ) <= set(components)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(value is not None for value in gradients)
    assert all(value is None or torch.isfinite(value).all() for value in gradients)


def test_forward_model_cannot_read_test_targets() -> None:
    model, batch = _model_batch()
    model.eval()
    first = _TRAIN.forward_model(model, batch, torch.device("cpu"))["expected_rr"]
    poisoned = dict(batch)
    poisoned["rr"] = torch.full_like(batch["rr"], 10000.0)
    poisoned["identity"] = ["TEST_SECRET"] * 3
    second = _TRAIN.forward_model(model, poisoned, torch.device("cpu"))["expected_rr"]
    torch.testing.assert_close(first, second)


def test_validation_promotion_requires_all_safety_gates() -> None:
    target = np.asarray([10.0, 11.0, 26.0, 28.0, 12.0, 30.0])
    identities = np.asarray(["A", "A", "A", "B", "B", "B"])
    base = target + np.asarray([1.0, -1.0, 1.5, -1.5, 1.0, -1.0])
    candidate = target + np.asarray([0.2, -0.2, 0.4, -0.4, 0.2, -0.3])
    decision = _TRAIN.promotion_decision(target, candidate, base, identities)
    assert decision["promoted"]
    assert decision["test_inputs_used"] is False

    harmful_tail = candidate.copy()
    harmful_tail[target >= 25] = target[target >= 25] + 2.0
    rejected = _TRAIN.promotion_decision(target, harmful_tail, base, identities)
    assert not rejected["promoted"]
    assert not rejected["gates"]["high_25_35_macro_mae_noninferior"]


def test_parse_fold_selection_and_cli_resume_guard() -> None:
    assert _TRAIN.parse_fold_selection("all") == list(range(6))
    assert _TRAIN.parse_fold_selection("5,1,1") == [1, 5]
    with pytest.raises(ValueError):
        _TRAIN.parse_fold_selection("6")
    with pytest.raises(SystemExit):
        _TRAIN.parse_args(["--resume-from", "x.pt", "--fold", "all"])
