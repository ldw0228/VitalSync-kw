from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from snn_rr.svd_temporal_models import TemporalSourceSeparatedRRSNN

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "train_svd_temporal_snn.py"
)
_SPEC = importlib.util.spec_from_file_location("snn_rr_train_svd_temporal", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TRAIN = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TRAIN
_SPEC.loader.exec_module(_TRAIN)

TemporalAlignedExperiment = _TRAIN.TemporalAlignedExperiment
TemporalSessionArrays = _TRAIN.TemporalSessionArrays
TrainOnlyActionCalibration = _TRAIN.TrainOnlyActionCalibration
_positions_digest = _TRAIN._positions_digest
apply_coupled_temporal_radar_dropout = _TRAIN.apply_coupled_temporal_radar_dropout
compute_temporal_multitask_loss = _TRAIN.compute_temporal_multitask_loss
fit_temporal_signal_normalizer = _TRAIN.fit_temporal_signal_normalizer
fit_train_only_action_calibration = _TRAIN.fit_train_only_action_calibration
load_temporal_aligned_experiment = _TRAIN.load_temporal_aligned_experiment
main = _TRAIN.main


def _svd_provenance_fields() -> dict[str, object]:
    return {
        "valid_only": True,
        "label_inputs": [
            "canonical_metadata.reference_valid (row_selection_only)"
        ],
        "feature_value_label_inputs": [],
        "target_dependent_row_selection": True,
        "causality_scope": {
            "window_end_prediction_uses_only_past_32s": True,
            "within_window_prefix_causal_representation": False,
            "streaming_prefix_causality_claim_allowed": False,
        },
    }


def _acquisition_v2_scope_fixture(
    *,
    expected: list[str] | None = None,
    selected: list[str] | None = None,
    subjects_filter_applied: bool = False,
    canonical_scientific: bool = True,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    expected_ids = ["S01_A", "S02_B"] if expected is None else list(expected)
    selected_ids = list(expected_ids if selected is None else selected)
    binding = {
        "schema_version": "snn_rr.feature_cache_acquisition.v2",
        "scientific_eligible": True,
    }
    children: dict[str, dict[str, object]] = {}
    catalogue: list[dict[str, object]] = []
    for session_id in selected_ids:
        child: dict[str, object] = {
            "canonical_acquisition_binding": dict(binding),
            "valid_only": True,
        }
        child["content_sha256"] = _TRAIN._canonical_content_sha256(child)
        children[session_id] = child
        catalogue.append(
            {
                "session_id": session_id,
                "status": "ok",
                "canonical_acquisition_binding": dict(binding),
                "content_sha256": child["content_sha256"],
            }
        )
    selection_is_full = bool(
        not subjects_filter_applied and selected_ids == expected_ids
    )
    canonical_contract = {
        "schema_version": "snn_rr.feature_cache_acquisition.v2",
        "mode": "strict",
        "selection_scope": "full_cohort",
        "reconstruction_full_cohort_complete": True,
        "full_cohort_complete": True,
        "scientific_eligible": canonical_scientific,
        "expected_usable_session_ids": expected_ids,
        "expected_usable_session_ids_sha256": _TRAIN._canonical_sha256(
            expected_ids
        ),
        "cache_usable_session_ids": expected_ids,
        "cache_usable_session_ids_sha256": _TRAIN._canonical_sha256(expected_ids),
    }
    root: dict[str, object] = {
        "canonical_acquisition_contract": canonical_contract,
        "expected_session_ids": expected_ids,
        "expected_session_ids_sha256": _TRAIN._canonical_sha256(expected_ids),
        "selected_session_ids": selected_ids,
        "selected_session_ids_sha256": _TRAIN._canonical_sha256(selected_ids),
        "subjects_filter_applied": subjects_filter_applied,
        "selection_scope": (
            "full_cohort" if selection_is_full else "diagnostic_subset"
        ),
        "full_cohort_complete": selection_is_full,
        "scientific_eligible": selection_is_full and canonical_scientific,
        "sessions": catalogue,
    }
    root["content_sha256"] = _TRAIN._canonical_content_sha256(root)
    return root, children


def _metadata(rows: int) -> pd.DataFrame:
    fold = np.arange(rows) % 6
    within_fold = np.arange(rows) // 6
    rr = np.asarray(
        [12.0, 30.0, 18.0, 24.0, 27.0, 15.0], dtype=np.float32
    )[fold] + within_fold.astype(np.float32)
    return pd.DataFrame(
        {
            "cache_index": np.arange(100, 100 + rows, dtype=np.int64),
            "session_id": ["SYNTH"] * rows,
            "identity": [f"P{value}" for value in fold],
            "protocol": ["synthetic"] * rows,
            "window_number": np.arange(rows),
            "window_start_s": np.arange(rows, dtype=float) * 4.0,
            "window_end_s": np.arange(rows, dtype=float) * 4.0 + 32.0,
            "rr_bpm": rr,
            "reference_valid": np.ones(rows, dtype=bool),
            "reference_quality": np.full(rows, 0.9, dtype=np.float32),
            "reference_sigma_bpm": np.full(rows, 0.6, dtype=np.float32),
            "radar_observable": np.ones(rows, dtype=bool),
            "classical_rr_bpm": 0.5 * rr,
            "_session_slot": np.zeros(rows, dtype=np.int64),
            "_local_row": np.arange(rows, dtype=np.int64),
            "fold": fold,
            "prediction_bpm": rr + 0.8,
            "rr_std_bpm": np.full(rows, 1.0, dtype=np.float32),
            "classical_confidence": np.full(rows, 0.8, dtype=np.float32),
            "radar_peak_1_bpm": 0.5 * rr + 0.1,
            "radar_peak_2_bpm": 0.5 * rr - 0.1,
            "radar_peak_3_bpm": 0.5 * rr + 0.2,
            "uncertainty_score": np.full(rows, 0.5, dtype=np.float32),
        }
    )


def _memory_experiment(*, rows: int = 12, time_steps: int = 32) -> TemporalAlignedExperiment:
    generator = np.random.default_rng(20260828)
    signals = generator.normal(size=(rows, 3, 2, 2, time_steps)).astype(np.float32)
    attributes = generator.uniform(size=(rows, 3, 2, 2, 5)).astype(np.float32)
    attributes[..., 4] *= 0.8
    metadata = _metadata(rows)
    session = TemporalSessionArrays(
        session_id="SYNTH",
        component_signals=signals,
        attributes=attributes,
        metadata=metadata,
        manifest=_svd_provenance_fields(),
        component_signals_sha256="synthetic",
    )
    return TemporalAlignedExperiment(
        cache_root=Path("/tmp/synthetic"),
        oof_csv=Path("/tmp/synthetic.csv"),
        oof_npz=Path("/tmp/synthetic.npz"),
        metadata=metadata,
        sessions=[session],
        root_manifest={"valid_only": True, "row_count": rows},
        provenance={"row_binding_sha256": "synthetic", "row_count": rows},
    )


def _write_disk_experiment(root: Path, *, rows: int = 12) -> tuple[Path, Path, Path]:
    cache_root = root / "cache"
    session_dir = cache_root / "SYNTH"
    session_dir.mkdir(parents=True)
    metadata = _metadata(rows).drop(columns=["_session_slot", "_local_row", "fold"])
    generator = np.random.default_rng(20260828)
    signals = generator.normal(size=(rows, 3, 2, 2, 320)).astype(np.float16)
    spectra = np.abs(generator.normal(size=(rows, 3, 2, 2, 9))).astype(np.float16)
    attributes = generator.uniform(size=(rows, 3, 2, 2, 5)).astype(np.float32)
    attributes[..., 4] *= 0.8
    frequencies = np.linspace(0.08, 0.8, 9, dtype=np.float32)
    np.save(session_dir / "component_signals.npy", signals)
    np.save(session_dir / "spectra.npy", spectra)
    np.save(session_dir / "attributes.npy", attributes)
    np.save(session_dir / "frequencies_hz.npy", frequencies)
    metadata.to_csv(session_dir / "metadata.csv", index=False)
    session_manifest = {
        "session_id": "SYNTH",
        "row_count": rows,
        **_svd_provenance_fields(),
        "component_signals_shape": list(signals.shape),
        "attributes_shape": list(attributes.shape),
        "spectra_shape": list(spectra.shape),
    }
    (session_dir / "manifest.json").write_text(
        json.dumps(session_manifest), encoding="utf-8"
    )
    root_manifest = {
        "valid_only": True,
        "row_count": rows,
        "sessions": [{"session_id": "SYNTH", "status": "ok"}],
    }
    (cache_root / "manifest.json").write_text(
        json.dumps(root_manifest), encoding="utf-8"
    )

    full = _metadata(rows)
    base = full.drop(columns=["_session_slot", "_local_row"]).copy()
    base_csv = root / "base.csv"
    base.to_csv(base_csv, index=False)
    base_npz = root / "base.npz"
    np.savez_compressed(
        base_npz,
        index=base["cache_index"].to_numpy(np.int64),
        target=base["rr_bpm"].to_numpy(np.float32),
        prediction=base["prediction_bpm"].to_numpy(np.float32),
        rr_std=base["rr_std_bpm"].to_numpy(np.float32),
        fold=base["fold"].to_numpy(np.int16),
    )
    return cache_root, base_csv, base_npz


def test_train_only_normalizer_and_action_threshold_ignore_outer_rows() -> None:
    experiment = _memory_experiment(rows=12)
    train = np.arange(8, dtype=np.int64)
    experiment.sessions[0].component_signals[8:] = 10_000.0
    normalizer = fit_temporal_signal_normalizer(
        experiment,
        train,
        max_samples_per_variant=128,
        clip_quantile=0.99,
        seed=7,
    )
    calibration = fit_train_only_action_calibration(experiment.metadata, train)

    assert normalizer.fit_positions_sha256 == _positions_digest(train)
    assert calibration.fit_positions_sha256 == _positions_digest(train)
    assert np.max(np.abs(normalizer.center)) < 1.0
    assert np.max(normalizer.clip) <= 20.0
    transformed = normalizer.transform(experiment.sessions[0].component_signals[8])
    assert np.isfinite(transformed).all()
    assert np.max(np.abs(transformed)) <= np.max(normalizer.clip)
    assert 0.05 <= calibration.gate_margin_bpm <= 0.50
    assert calibration.fit_identity_count == len(
        np.unique(experiment.metadata.iloc[train]["identity"])
    )


def test_acquisition_v2_training_scope_accepts_bound_full_cohort() -> None:
    root, children = _acquisition_v2_scope_fixture()
    _TRAIN._validate_acquisition_v2_training_scope(root, children)


def test_acquisition_v2_subset_is_diagnostic_and_not_trainable() -> None:
    root, children = _acquisition_v2_scope_fixture(
        selected=["S01_A"], subjects_filter_applied=True
    )
    with pytest.raises(RuntimeError, match="requires a strict.*full-cohort"):
        _TRAIN._validate_acquisition_v2_training_scope(root, children)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_session_ids_sha256", "0" * 64, "selected-session ID hash"),
        ("selection_scope", "full_cohort", "selection_scope"),
        ("full_cohort_complete", True, "full_cohort_complete"),
        ("scientific_eligible", True, "scientific_eligible"),
    ],
)
def test_acquisition_v2_subset_claim_tampering_fails_closed(
    field: str, value: object, message: str
) -> None:
    root, children = _acquisition_v2_scope_fixture(
        selected=["S01_A"], subjects_filter_applied=True
    )
    root[field] = value
    root["content_sha256"] = _TRAIN._canonical_content_sha256(root)
    with pytest.raises(RuntimeError, match=message):
        _TRAIN._validate_acquisition_v2_training_scope(root, children)


def test_acquisition_v2_canonical_downgrade_or_binding_strip_fails_closed() -> None:
    root, children = _acquisition_v2_scope_fixture()
    del root["canonical_acquisition_contract"]
    root["content_sha256"] = _TRAIN._canonical_content_sha256(root)
    with pytest.raises(RuntimeError, match="lacks its canonical root"):
        _TRAIN._validate_acquisition_v2_training_scope(root, children)

    root, children = _acquisition_v2_scope_fixture(canonical_scientific=False)
    with pytest.raises(RuntimeError, match="requires a strict.*full-cohort"):
        _TRAIN._validate_acquisition_v2_training_scope(root, children)

    root, children = _acquisition_v2_scope_fixture()
    del root["canonical_acquisition_contract"]
    for child in children.values():
        del child["canonical_acquisition_binding"]
        child["radar_timing_valid_mask_shape"] = [1, 3, 320]
        child["content_sha256"] = _TRAIN._canonical_content_sha256(child)
    for item in root["sessions"]:
        session_id = str(item["session_id"])
        replacement = {
            "session_id": session_id,
            "status": "ok",
            **children[session_id],
        }
        item.clear()
        item.update(replacement)
    root["content_sha256"] = _TRAIN._canonical_content_sha256(root)
    with pytest.raises(RuntimeError, match="lacks its canonical root"):
        _TRAIN._validate_acquisition_v2_training_scope(root, children)


def test_acquisition_scope_validator_preserves_legacy_behavior() -> None:
    _TRAIN._validate_acquisition_v2_training_scope(
        {"sessions": [{"session_id": "LEGACY", "status": "ok"}]},
        {"LEGACY": {"valid_only": True}},
    )


def test_temporal_multitask_loss_backward_and_soft_actions() -> None:
    torch.manual_seed(11)
    model = TemporalSourceSeparatedRRSNN(
        num_variants=2,
        num_components=2,
        compressor_channels=8,
        hidden_channels=12,
        cell_types=("lif", "plif"),
        dropout=0.0,
    )
    batch = {
        "component_signals": torch.randn(3, 3, 2, 2, 32),
        "attributes": torch.rand(3, 3, 2, 2, 5),
        "base_prediction": torch.tensor([18.0, 27.0, 32.0]),
        "base_std": torch.tensor([1.0, 1.2, 0.8]),
        "classical_rr": torch.tensor(
            [[9.0, 8.8], [9.0, 13.5], [8.0, float("nan")]]
        ),
        "radar_mask": torch.tensor(
            [[True, True, True], [True, False, True], [False, True, True]]
        ),
        "rr": torch.tensor([18.5, 27.5, 31.5]),
        "reference_valid": torch.ones(3, dtype=torch.bool),
        "reference_quality": torch.tensor([1.0, 0.8, 0.9]),
        "reference_sigma": torch.tensor([0.5, 0.7, 0.6]),
        "observable": torch.ones(3),
    }
    output = model(
        batch["component_signals"],
        batch["attributes"],
        batch["base_prediction"],
        batch["base_std"],
        batch["classical_rr"],
        batch["radar_mask"],
    )
    calibration = TrainOnlyActionCalibration(0.1, 0.5, 0.75, "train", 3)
    loss, parts = compute_temporal_multitask_loss(
        output,
        batch,
        model.rr_bins,
        action_calibration=calibration,
    )
    assert torch.isfinite(loss)
    for name in (
        "posterior_nll",
        "source_nll",
        "crps",
        "mae",
        "divisor_ce",
        "divisor_regret",
        "residual",
        "uncertainty_nll",
        "gate_bce",
        "action_regret",
        "safe_gate",
    ):
        assert name in parts and torch.isfinite(parts[name])
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(value is not None for value in gradients)
    assert all(value is None or torch.isfinite(value).all() for value in gradients)


def test_coupled_radar_dropout_never_leaks_attributes_or_drops_everything() -> None:
    torch.manual_seed(3)
    signals = torch.ones(8, 3, 2, 2, 32)
    attributes = torch.ones(8, 3, 2, 2, 5)
    existing = torch.tensor([[True, True, False]]).expand(8, -1)
    dropped, dropped_attributes, available = apply_coupled_temporal_radar_dropout(
        signals, attributes, existing, p=1.0, training=True
    )
    assert torch.all(available.sum(dim=1) == 1)
    assert torch.count_nonzero(available[:, 2]) == 0
    torch.testing.assert_close(
        dropped.sum(dim=(2, 3, 4)), available.float() * (2 * 2 * 32)
    )
    torch.testing.assert_close(
        dropped_attributes.sum(dim=(2, 3, 4)), available.float() * (2 * 2 * 5)
    )


def test_temporal_dataset_uses_structural_mask_not_numeric_signal_values() -> None:
    experiment = _memory_experiment(rows=12)
    experiment.sessions[0].component_signals[:, 0] = 0
    timing = np.ones((12, 3, 320), dtype=np.bool_)
    timing[:, 1, 0] = False
    experiment.sessions[0].radar_timing_valid_mask = timing
    normalizer = _TRAIN.TemporalSignalNormalizer(
        center=np.zeros(2, dtype=np.float32),
        scale=np.ones(2, dtype=np.float32),
        clip=np.full(2, 10.0, dtype=np.float32),
        fit_positions_sha256="synthetic",
        sampled_values_per_variant=1,
    )
    sample = _TRAIN.TemporalSVDDataset(experiment, [0], normalizer)[0]

    assert bool(sample["radar_mask"][0]) is True
    assert torch.count_nonzero(sample["component_signals"][0]) == 0
    assert bool(sample["radar_mask"][1]) is False
    assert torch.count_nonzero(sample["component_signals"][1]) == 0
    assert torch.count_nonzero(sample["attributes"][1]) == 0
    assert sample["classical_rr"][0] == 0
    assert sample["classical_rr"][2] == 0


def test_valid_only_disk_alignment_loads_component_signal_binding(tmp_path: Path) -> None:
    cache, base_csv, base_npz = _write_disk_experiment(tmp_path)
    experiment = load_temporal_aligned_experiment(
        cache, base_csv, base_npz, verify_file_hashes=False
    )
    assert len(experiment.metadata) == 12
    assert experiment.sessions[0].component_signals.shape == (12, 3, 2, 2, 320)
    assert experiment.provenance["valid_only_alignment_enforced"] is True
    assert experiment.provenance["component_signals_status"].startswith(
        "window_end"
    )
    np.testing.assert_array_equal(
        experiment.metadata["cache_index"].to_numpy(), np.arange(100, 112)
    )

    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["valid_only"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        load_temporal_aligned_experiment(
            cache, base_csv, base_npz, verify_file_hashes=False
        )
    except RuntimeError as exc:
        assert "valid-reference-only" in str(exc)
    else:  # pragma: no cover - safety assertion
        raise AssertionError("non-valid-only cache was accepted")


def test_cpu_tiny_cli_locks_validation_then_evaluates_test_once(tmp_path: Path) -> None:
    cache, base_csv, base_npz = _write_disk_experiment(tmp_path / "data")
    output = tmp_path / "run"
    arguments = [
        "--svd-cache",
        str(cache),
        "--base-oof-csv",
        str(base_csv),
        "--base-oof-npz",
        str(base_npz),
        "--output-dir",
        str(output),
        "--fold",
        "0",
        "--preset",
        "tiny",
        "--cell-types",
        "lif",
        "--epochs",
        "1",
        "--patience",
        "1",
        "--batch-size",
        "2",
        "--eval-batch-size",
        "2",
        "--workers",
        "0",
        "--device",
        "cpu",
        "--no-amp",
        "--no-verify-file-hashes",
        "--smoke-max-batches",
        "1",
        "--bootstrap-samples",
        "10",
        "--normalizer-max-samples",
        "64",
        "--normalizer-clip-quantile",
        "0.99",
    ]
    assert main(arguments) == 0

    fold_dir = output / "fold_0"
    lock = json.loads((fold_dir / "selection_lock.json").read_text(encoding="utf-8"))
    completion_path = fold_dir / "test_evaluation_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert lock["test_loader_constructed"] is False
    assert lock["test_labels_or_metrics_used_for_selection"] is False
    assert lock["maximum_permitted_test_evaluations"] == 1
    assert completion["test_fold_evaluation_invocations"] == 1
    assert completion[
        "validation_selection_completed_before_test_loader_construction"
    ] is True
    assert completion["test_metrics_used_for_model_or_action_selection"] is False
    assert (fold_dir / "temporal_best.pt").is_file()
    assert (output / "temporal_oof.npz").is_file()
    assert (output / "temporal_oof.csv").is_file()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluated_folds"] == [0]
    assert metrics["complete_six_fold_oof"] is False
    assert "tail_25_35" in metrics["locked_final"]

    first_manifest_sha = hashlib.sha256(completion_path.read_bytes()).hexdigest()
    assert main([*arguments, "--resume"]) == 0
    second_manifest_sha = hashlib.sha256(completion_path.read_bytes()).hexdigest()
    assert first_manifest_sha == second_manifest_sha
    completion_again = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion_again["test_fold_evaluation_invocations"] == 1
