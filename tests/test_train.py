from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from snn_rr.cache import CacheProvenance, FeatureCache
import snn_rr.split_authority as split_authority_module
from snn_rr.split_authority import (
    canonical_content_sha256,
    load_identity_split_authority,
    sha256_file,
)

_TRAIN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_SPEC = importlib.util.spec_from_file_location("snn_rr_train_script", _TRAIN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TRAIN = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TRAIN
_SPEC.loader.exec_module(_TRAIN)

PredictionBundle = _TRAIN.PredictionBundle
apply_coupled_radar_dropout = _TRAIN.apply_coupled_radar_dropout
compute_multitask_loss = _TRAIN.compute_multitask_loss
make_alias_gate_targets = _TRAIN.make_alias_gate_targets
identity_balanced_sample_weights = _TRAIN.identity_balanced_sample_weights
infer_auxiliary_layout = _TRAIN.infer_auxiliary_layout
infer_auxiliary_frequency_range = _TRAIN.infer_auxiliary_frequency_range
make_fold_assignments = _TRAIN.make_fold_assignments
parse_fold_selection = _TRAIN.parse_fold_selection
detailed_prediction_summary = _TRAIN.detailed_prediction_summary
capture_rng_state = _TRAIN.capture_rng_state
restore_rng_state = _TRAIN.restore_rng_state
validate_external_teacher_checkpoint = _TRAIN.validate_external_teacher_checkpoint
teacher_checkpoint_provenance = _TRAIN.teacher_checkpoint_provenance


def _legacy_cache_provenance(tmp_path: Path) -> CacheProvenance:
    return CacheProvenance(
        classification="legacy",
        root_manifest_path=str(tmp_path / "cache/manifest.json"),
        root_manifest_sha256="1" * 64,
        root_manifest_content_sha256="2" * 64,
        acquisition_schema_version=None,
        acquisition_mode=None,
        scientific_eligible=False,
        config_sha256="3" * 64,
        pipeline_sha256="4" * 64,
        reconstruction_content_sha256=None,
        inventory_sha256="5" * 64,
        inventory_file_count=4,
        selected_sessions=("SA", "SB", "SC", "SD"),
    )


def _write_cache_metadata_files(
    cache_dir: Path, metadata: pd.DataFrame
) -> pd.DataFrame:
    session_ids = sorted(metadata["session_id"].astype(str).unique())
    canonical_frames: list[pd.DataFrame] = []
    for session_id in session_ids:
        session_dir = cache_dir / session_id
        session_dir.mkdir(exist_ok=True)
        metadata_path = session_dir / "metadata.csv"
        metadata.loc[metadata["session_id"].astype(str) == session_id].to_csv(
            metadata_path, index=False
        )
        canonical_frames.append(pd.read_csv(metadata_path))
    return pd.concat(canonical_frames, ignore_index=True)


def _write_custom_split_fixture(
    tmp_path: Path,
    *,
    identities: dict[str, list[str]] | None = None,
    metadata: pd.DataFrame | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    if metadata is None:
        metadata = pd.DataFrame(
            {
                "session_id": ["SA", "SA", "SB", "SB", "SC", "SC", "SD", "SD"],
                "identity": ["A", "A", "B", "B", "C", "C", "D", "D"],
                "reference_valid": [True, False, True, True, True, True, True, False],
            }
        )
    session_ids = sorted(metadata["session_id"].astype(str).unique())
    metadata = _write_cache_metadata_files(cache_dir, metadata)
    cache_manifest = {
        "config_sha256": "1" * 64,
        "pipeline_sha256": "2" * 64,
        "sessions": [
            {"session_id": session, "status": "ok"}
            for session in session_ids
        ],
    }
    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest_path.write_text(
        json.dumps(cache_manifest, sort_keys=True), encoding="utf-8"
    )
    folds_path = tmp_path / "fold_assignments.json"
    folds_path.write_text(
        json.dumps({"identity_to_fold": {"A": 0, "B": 1, "C": 2, "D": 3}}),
        encoding="utf-8",
    )
    split_identities = identities or {
        "train": ["A"],
        "validation": ["B"],
        "prediction": ["C"],
        "excluded": ["D"],
        "scaler": ["A"],
    }
    document: dict[str, object] = {
        "schema_version": 1,
        "fold_id": 7,
        "fold_assignments": {
            "path": str(folds_path),
            "sha256": sha256_file(folds_path),
        },
        "cache": {
            "manifest_path": str(cache_manifest_path),
            "manifest_sha256": sha256_file(cache_manifest_path),
        },
        "identities": split_identities,
    }
    document["content_sha256"] = canonical_content_sha256(document)
    split_path = tmp_path / "identity_split.json"
    split_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return split_path, cache_dir, metadata


def test_fold_assignment_is_identity_disjoint_and_complete() -> None:
    identities = np.repeat([f"P{index:02d}" for index in range(18)], [index + 2 for index in range(18)])
    metadata = pd.DataFrame(
        {
            "identity": identities,
            "reference_valid": np.ones(len(identities), dtype=bool),
        }
    )
    assignment, identity_to_fold = make_fold_assignments(metadata, 6, 20260827)

    assert set(identity_to_fold) == set(metadata["identity"])
    assert set(assignment) == set(range(6))
    for identity in metadata["identity"].unique():
        assert len(np.unique(assignment[metadata["identity"] == identity])) == 1
    assert parse_fold_selection("all", 6) == list(range(6))
    assert parse_fold_selection("5,1,1", 6) == [1, 5]


def test_identity_balanced_weights_equalize_total_mass() -> None:
    metadata = pd.DataFrame(
        {
            "identity": ["A", "A", "A", "B", "B", "C"],
            "reference_valid": [True, False, False, True, True, True],
        }
    )
    weights = identity_balanced_sample_weights(metadata, np.arange(6), valid_boost=2.0)
    totals = {
        identity: weights[metadata["identity"].to_numpy() == identity].sum()
        for identity in ("A", "B", "C")
    }
    np.testing.assert_allclose(list(totals.values()), totals["A"])
    assert weights[0] > weights[1]


def test_auxiliary_frequency_range_recovers_unpooled_fft_grid() -> None:
    minimum, maximum = infer_auxiliary_frequency_range(
        np.asarray([0.085, 0.105, 0.125]), 7
    )
    np.testing.assert_allclose([minimum, maximum], [0.08, 0.14], atol=1e-12)


def test_rr_balance_upweights_rare_band_without_breaking_identity_balance() -> None:
    metadata = pd.DataFrame(
        {
            "identity": ["A"] * 5 + ["B"] * 5,
            "reference_valid": [True] * 10,
            "rr_bpm": [10.0, 11.0, 12.0, 13.0, 28.0] * 2,
        }
    )
    weights = identity_balanced_sample_weights(
        metadata,
        np.arange(10),
        rr_balance_power=1.0,
        rr_balance_bin_width=5.0,
    )
    assert weights[4] > weights[0]
    np.testing.assert_allclose(weights[:5].sum(), weights[5:].sum())


def _dummy_loss(invalid_rr: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = torch.tensor(
        [[0.0, 0.5, 1.0, 0.0, -0.5], [1.0, 0.0, -1.0, 0.0, 1.0]],
        requires_grad=True,
    )
    probabilities = logits.softmax(dim=-1)
    bins = torch.linspace(6.0, 10.0, 5)
    expected = (probabilities * bins).sum(dim=-1)
    quality_logits = torch.tensor([0.3, -0.4], requires_grad=True)
    output = {
        "logits": logits,
        "expected_rr": expected,
        "log_variance": torch.zeros(2, requires_grad=True),
        "quality_logits": quality_logits,
        "spike_rate": logits.sigmoid().mean(),
    }
    batch = {
        "rr": torch.tensor([8.0, invalid_rr]),
        "reference_valid": torch.tensor([True, False]),
        "reference_quality": torch.tensor([0.9, 0.1]),
        "reference_sigma": torch.tensor([0.5, 0.5]),
        "observable": torch.tensor([1.0, 0.0]),
    }
    model = SimpleNamespace(rr_bins=bins)
    return compute_multitask_loss(output, batch, model)


def test_multitask_loss_masks_invalid_rr_and_backpropagates() -> None:
    loss_a, components_a = _dummy_loss(7.0)
    loss_b, components_b = _dummy_loss(1000.0)

    torch.testing.assert_close(loss_a, loss_b)
    for key in ("distribution", "huber", "uncertainty_nll"):
        torch.testing.assert_close(components_a[key], components_b[key])
    assert torch.isfinite(loss_a)
    loss_a.backward()


def test_tail_distance_and_teacher_error_gate_are_effective() -> None:
    bins = torch.linspace(6.0, 30.0, 25)
    logits = torch.zeros(2, 25, requires_grad=True)
    output = {
        "logits": logits,
        "expected_rr": torch.tensor([12.0, 18.0], requires_grad=True),
        "log_variance": torch.zeros(2, requires_grad=True),
        "quality_logits": torch.zeros(2, requires_grad=True),
        "spike_rate": logits.sum() * 0.0,
    }
    batch = {
        "rr": torch.tensor([12.0, 28.0]),
        "reference_valid": torch.tensor([True, True]),
        "reference_quality": torch.ones(2),
        "reference_sigma": torch.full((2,), 0.5),
        "observable": torch.ones(2),
    }
    teacher_logits = torch.full((2, 25), -8.0)
    teacher_logits[0, 6] = 8.0  # 12 bpm: accurate
    teacher_logits[1, 6] = 8.0  # 12 bpm: badly wrong for the 28 bpm target
    model = SimpleNamespace(rr_bins=bins)

    _, base = compute_multitask_loss(
        output,
        batch,
        model,
        teacher_logits=teacher_logits,
        tail_loss_weight=0.0,
        distill_error_gate_bpm=0.0,
    )
    loss, guarded = compute_multitask_loss(
        output,
        batch,
        model,
        teacher_logits=teacher_logits,
        tail_loss_weight=0.05,
        tail_min_bpm=22.0,
        tail_max_bpm=35.0,
        tail_underprediction_ratio=1.5,
        distill_error_gate_bpm=2.0,
    )
    assert guarded["tail_distance"] > 0
    assert guarded["distillation"] < base["distillation"]
    loss.backward()
    assert output["expected_rr"].grad is not None
    assert output["expected_rr"].grad[1].abs() > output["expected_rr"].grad[0].abs()


def test_alias_gate_targets_ignore_unroutable_rows_and_bce_backpropagates() -> None:
    target = torch.tensor([8.0, 24.0, 30.0, 40.0])
    classical = torch.tensor([8.0, 8.0, 7.5, 6.0])
    alias, confident = make_alias_gate_targets(
        target,
        classical,
        torch.ones(4, dtype=torch.bool),
        rr_min=6.0,
        rr_max=45.0,
        tolerance_bpm=2.0,
    )
    torch.testing.assert_close(alias, torch.tensor([False, True, True, True]))
    torch.testing.assert_close(confident, torch.tensor([True, True, True, False]))

    bins = torch.linspace(6.0, 30.0, 25)
    logits = torch.zeros(3, 25, requires_grad=True)
    alias_logits = torch.zeros(3, requires_grad=True)
    output = {
        "logits": logits,
        "expected_rr": torch.tensor([8.0, 20.0, 22.0], requires_grad=True),
        "log_variance": torch.zeros(3, requires_grad=True),
        "quality_logits": torch.zeros(3, requires_grad=True),
        "spike_rate": logits.sum() * 0.0,
        "alias_logits": alias_logits,
        "radar_mask": torch.tensor(
            [[True, True, True], [True, True, True], [True, False, True]]
        ),
    }
    batch = {
        "rr": torch.tensor([8.0, 24.0, 30.0]),
        "classical_rr": torch.tensor([8.0, 8.0, 7.5]),
        "reference_valid": torch.ones(3, dtype=torch.bool),
        "reference_quality": torch.ones(3),
        "reference_sigma": torch.full((3,), 0.5),
        "observable": torch.ones(3),
    }
    model = SimpleNamespace(rr_bins=bins)
    loss, components = compute_multitask_loss(
        output,
        batch,
        model,
        alias_loss_weight=0.05,
        alias_positive_weight=3.0,
    )
    assert components["alias_bce"] > 0
    assert components["alias_positive_fraction"] == 0.5
    loss.backward()
    assert alias_logits.grad is not None
    assert alias_logits.grad[0] > 0  # direct target pushes its logit down.
    assert alias_logits.grad[1] < 0  # alias target pushes its logit up.
    assert alias_logits.grad[2] == 0  # partial-radar row is ignored.


def test_prediction_bundle_length() -> None:
    bundle = PredictionBundle(
        index=np.arange(3),
        target=np.ones(3),
        prediction=np.ones(3),
        rr_std=np.ones(3),
        uncertainty=np.ones(3),
        quality=np.ones(3),
        observable=np.ones(3, dtype=bool),
        reference_valid=np.ones(3, dtype=bool),
        spike_rate=np.zeros(3),
        radar_weights=np.full((3, 3), 1 / 3),
    )
    assert len(bundle) == 3


def test_prediction_bundle_preserves_large_custom_fold_identifier(
    tmp_path: Path,
) -> None:
    bundle = PredictionBundle(
        index=np.asarray([0]),
        target=np.asarray([12.0]),
        prediction=np.asarray([12.0]),
        rr_std=np.asarray([0.2]),
        uncertainty=np.asarray([0.1]),
        quality=np.asarray([0.9]),
        observable=np.asarray([True]),
        reference_valid=np.asarray([True]),
        spike_rate=np.asarray([0.01]),
        radar_weights=np.full((1, 3), 1 / 3),
    )
    path = tmp_path / "prediction.npz"
    large_fold = 2**40
    _TRAIN.save_prediction_bundle(
        path, bundle, fold=large_fold, run_signature="signature"
    )
    _, stored_fold, signature = _TRAIN.load_prediction_bundle(path)
    assert stored_fold.dtype == np.int64
    assert stored_fold.tolist() == [large_fold]
    assert signature == "signature"


def test_coupled_radar_dropout_neutralizes_auxiliary_bypass() -> None:
    # F=1 produces base_dim = 3*(2 spectra + 8 scalars) + 2 fused + 5 = 37.
    layout = infer_auxiliary_layout(37)
    radar_map = torch.ones(4, 3, 2, 6)
    radar_mask = torch.ones(4, 3, dtype=torch.bool)
    aux = torch.ones(4, 40)  # three causal-history values follow base aux.
    dropped_map, kept, dropped_aux = apply_coupled_radar_dropout(
        radar_map, radar_mask, aux, p=1.0, layout=layout
    )

    assert torch.all(kept.sum(dim=1) == 1)
    torch.testing.assert_close(
        dropped_map.sum(dim=(2, 3)), kept.to(dtype=torch.float32) * 12
    )
    # All fused current-window features are neutral, while strictly causal
    # appended history remains available to the caller.
    assert torch.count_nonzero(dropped_aux[:, 30:37]) == 0
    assert torch.all(dropped_aux[:, 37:] == 1)


def _timing_mask_metadata(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["S01_A"] * rows,
            "identity": ["A"] * rows,
            "window_number": np.arange(rows),
            "rr_bpm": np.linspace(12.0, 14.0, rows),
            "reference_valid": np.ones(rows, dtype=bool),
            "reference_quality": np.ones(rows),
            "reference_sigma_bpm": np.ones(rows),
            "radar_observable": np.ones(rows, dtype=bool),
            "classical_rr_bpm": np.linspace(11.0, 13.0, rows),
            "classical_confidence": np.full(rows, 0.8),
            "radar_peak_spread_bpm": np.full(rows, 0.4),
        }
    )


def test_cached_dataset_uses_structural_timing_mask_not_numeric_zero() -> None:
    layout = infer_auxiliary_layout(61)  # F=4.
    maps = np.full((2, 3, 2, 4), 7.0, dtype=np.float16)
    maps[0, 0] = 0.0  # A legitimate numeric-zero view remains available.
    aux = np.ones((2, layout.base_dim), dtype=np.float32)
    timing = np.ones((2, 3, 5), dtype=np.bool_)
    timing[1, 1, 2] = False  # Nonzero payload, structurally unavailable view.
    cache = FeatureCache(
        maps=maps,
        aux=aux,
        metadata=_timing_mask_metadata(2),
        frequencies_hz=np.asarray([0.1, 0.2], dtype=np.float32),
        radar_timing_valid_mask=timing,
    )
    dataset = _TRAIN.CachedRadarDataset(
        cache,
        aux,
        np.arange(2),
        auxiliary_layout=layout,
    )

    numeric_zero = dataset[0]
    assert numeric_zero["radar_mask"].tolist() == [True, True, True]
    assert torch.count_nonzero(numeric_zero["map"][0]) == 0
    assert torch.all(numeric_zero["aux"] == 1)

    structurally_invalid = dataset[1]
    assert structurally_invalid["radar_mask"].tolist() == [True, False, True]
    assert torch.count_nonzero(structurally_invalid["map"][1]) == 0
    assert torch.all(structurally_invalid["map"][[0, 2]] == 7)
    # Per-view spectra and scalars plus all fused current-window cells are
    # neutralized after scaling.  Valid per-view cells remain untouched.
    assert torch.all(structurally_invalid["aux"][8:16] == 0)
    assert torch.all(structurally_invalid["aux"][32:40] == 0)
    assert torch.all(structurally_invalid["aux"][48:61] == 0)
    assert torch.all(structurally_invalid["aux"][0:8] == 1)
    assert torch.all(structurally_invalid["aux"][16:32] == 1)
    assert torch.all(structurally_invalid["aux"][40:48] == 1)


def test_causal_history_excludes_structurally_invalid_prior_window() -> None:
    metadata = _timing_mask_metadata(3)
    timing = np.ones((3, 3, 5), dtype=np.bool_)
    timing[0, 2, 4] = False
    cache = FeatureCache(
        maps=np.ones((3, 3, 2, 4), dtype=np.float16),
        aux=np.ones((3, 1), dtype=np.float32),
        metadata=metadata,
        frequencies_hz=np.asarray([0.1, 0.2], dtype=np.float32),
        radar_timing_valid_mask=timing,
    )
    augmented, names = _TRAIN.append_mask_aware_causal_history_features(cache)
    history = augmented[:, 1:]
    lag_rr = names.index("history_lag_1_classical_rr_bpm")
    lag_available = names.index("history_lag_1_available")

    assert history[1, lag_rr] == 0.0
    assert history[1, lag_available] == 0.0
    assert history[2, lag_rr] == pytest.approx(metadata.loc[1, "classical_rr_bpm"])
    assert history[2, lag_available] == 1.0


def _teacher_checkpoint_fixture(tmp_path: Path) -> tuple[
    Path,
    dict[str, object],
    dict[str, list[str]],
    dict[str, object],
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    checkpoint_path = tmp_path / "teacher_run" / "fold_0" / "teacher_best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    split = {
        "train_identities": ["A", "B"],
        "validation_identities": ["C"],
        "test_identities": ["D"],
    }
    expected_model_context: dict[str, object] = {
        "num_radars": 3,
        "rr_min": 6.0,
        "rr_max": 45.0,
        "num_rr_bins": 157,
        "aux_dim": 7,
        "input_branches": 2,
        "input_frequency_min_hz": 0.08,
        "input_frequency_max_hz": 0.79,
    }
    aux_center = np.arange(7, dtype=np.float32)
    aux_scale = np.arange(1, 8, dtype=np.float32)
    checkpoint: dict[str, object] = {
        "format_version": 1,
        "model_type": "teacher",
        "model_kwargs": {**expected_model_context, "spatial_channels": (8, 12)},
        "model_state": {
            "rr_bins": torch.linspace(6.0, 45.0, 157),
        },
        "fold": 0,
        "split": split,
        "aux_center": torch.from_numpy(aux_center),
        "aux_scale": torch.from_numpy(aux_scale),
        "run_signature": "teacher-signature",
    }
    torch.save(checkpoint, checkpoint_path)
    current_run_context: dict[str, object] = {
        "folds": 6,
        "rr_range": [6.0, 45.0],
        "rr_bin_width": 0.25,
        "map_branch": "both",
        "input_branches": 2,
        "use_aux": True,
        "causal_history": True,
        "cache_dir": tmp_path / "cache",
        "cache_shape": {"maps": [12, 3, 73, 182], "aux": [12, 7]},
    }
    run_config = {
        "run_signature": "teacher-signature",
        "arguments": {
            key: value
            for key, value in current_run_context.items()
            if key != "cache_shape"
        },
        "cache_shape": current_run_context["cache_shape"],
    }
    (checkpoint_path.parent.parent / "run_config.json").write_text(
        json.dumps(run_config, default=str), encoding="utf-8"
    )
    return (
        checkpoint_path,
        checkpoint,
        split,
        expected_model_context,
        aux_center,
        aux_scale,
        current_run_context,
    )


def test_external_teacher_validation_accepts_matching_legacy_artifact(
    tmp_path: Path,
) -> None:
    (
        path,
        checkpoint,
        split,
        model_context,
        center,
        scale,
        run_context,
    ) = _teacher_checkpoint_fixture(tmp_path)
    provenance = validate_external_teacher_checkpoint(
        checkpoint,
        path=path,
        fold=0,
        split=split,
        expected_model_context=model_context,
        aux_center=center,
        aux_scale=scale,
        current_run_context=run_context,
    )
    assert provenance == teacher_checkpoint_provenance(path, checkpoint)
    assert provenance["model_type"] == "teacher"
    assert len(provenance["sha256"]) == 64


def test_external_teacher_validation_binds_custom_split_authority(
    tmp_path: Path,
) -> None:
    (
        path,
        checkpoint,
        _,
        model_context,
        center,
        scale,
        run_context,
    ) = _teacher_checkpoint_fixture(tmp_path)
    custom_split = {
        "train_identities": ["A"],
        "validation_identities": ["C"],
        "prediction_identities": ["D"],
        "excluded_identities": ["B"],
        "scaler_identities": ["A"],
    }
    authority = {
        "mode": "custom_identity_split",
        "schema_version": 1,
        "fold_id": 0,
        "split_manifest_content_sha256": "a" * 64,
        "split_manifest_file_sha256": "b" * 64,
        "fold_assignments_sha256": "c" * 64,
        "cache_manifest_sha256": "d" * 64,
        **custom_split,
    }
    checkpoint["split"] = custom_split
    checkpoint["split_authority_provenance"] = authority
    torch.save(checkpoint, path)
    run_context["split_authority_provenance"] = authority
    run_config_path = path.parent.parent / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["split_authority"] = authority
    run_config_path.write_text(json.dumps(run_config, default=str), encoding="utf-8")

    provenance = validate_external_teacher_checkpoint(
        checkpoint,
        path=path,
        fold=0,
        split=custom_split,
        expected_model_context=model_context,
        aux_center=center,
        aux_scale=scale,
        current_run_context=run_context,
    )
    assert provenance["split_manifest_content_sha256"] == "a" * 64
    assert provenance["excluded_identities"] == ["B"]
    assert provenance["scaler_identities"] == ["A"]

    changed = copy.deepcopy(checkpoint)
    changed["split_authority_provenance"]["excluded_identities"] = ["LEAK"]
    with pytest.raises(RuntimeError, match="split authority provenance mismatch"):
        validate_external_teacher_checkpoint(
            changed,
            path=path,
            fold=0,
            split=custom_split,
            expected_model_context=model_context,
            aux_center=center,
            aux_scale=scale,
            current_run_context=run_context,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(model_type="snn"), "model_type"),
        (lambda value: value.update(fold=1), "fold mismatch"),
        (
            lambda value: value["split"].update(test_identities=["LEAK"]),
            "test_identities mismatch",
        ),
        (
            lambda value: value["model_state"].update(
                rr_bins=torch.linspace(7.0, 45.0, 157)
            ),
            "RR grid",
        ),
        (
            lambda value: value["model_kwargs"].update(aux_dim=8),
            "model context aux_dim mismatch",
        ),
    ],
)
def test_external_teacher_validation_rejects_fold_split_grid_and_context_mismatch(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    (
        path,
        checkpoint,
        split,
        model_context,
        center,
        scale,
        run_context,
    ) = _teacher_checkpoint_fixture(tmp_path)
    changed = copy.deepcopy(checkpoint)
    mutation(changed)  # type: ignore[operator]
    with pytest.raises(RuntimeError, match=message):
        validate_external_teacher_checkpoint(
            changed,
            path=path,
            fold=0,
            split=split,
            expected_model_context=model_context,
            aux_center=center,
            aux_scale=scale,
            current_run_context=run_context,
        )


def test_external_teacher_validation_rejects_run_checkpoint_signature_mismatch(
    tmp_path: Path,
) -> None:
    (
        path,
        checkpoint,
        split,
        model_context,
        center,
        scale,
        run_context,
    ) = _teacher_checkpoint_fixture(tmp_path)
    changed = copy.deepcopy(checkpoint)
    changed["run_signature"] = "different"
    with pytest.raises(RuntimeError, match="run_config signatures differ"):
        validate_external_teacher_checkpoint(
            changed,
            path=path,
            fold=0,
            split=split,
            expected_model_context=model_context,
            aux_center=center,
            aux_scale=scale,
            current_run_context=run_context,
        )


def test_run_signature_binds_resume_sensitive_parameters() -> None:
    args = _TRAIN.parse_args([])
    args.input_branches = 2
    base_signature = _TRAIN._run_signature(args)
    changes = {
        "samples_per_epoch": 123,
        "gradient_clip": args.gradient_clip + 1.0,
        "patience": args.patience + 1,
        "min_delta": args.min_delta + 0.01,
        "teacher_checkpoint": "/tmp/different_teacher/fold_{fold}.pt",
        "identity_split_manifest_sha256": "a" * 64,
        "cache_provenance_sha256": "b" * 64,
    }
    for key, value in changes.items():
        changed = copy.deepcopy(args)
        setattr(changed, key, value)
        assert _TRAIN._run_signature(changed) != base_signature, key


def test_rng_capture_restore_includes_process_and_sampler_generators() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    generator = torch.Generator().manual_seed(91)
    dataset = TensorDataset(torch.arange(4))
    sampler = WeightedRandomSampler(
        torch.ones(4, dtype=torch.double),
        num_samples=4,
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(dataset, sampler=sampler, generator=generator)
    state = capture_rng_state(loader)

    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        torch.rand(3, generator=generator),
    )
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    generator.manual_seed(999)
    restore_rng_state(state, loader)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        torch.rand(3, generator=generator),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])
    torch.testing.assert_close(actual[3], expected[3])

    checkpoint = _TRAIN._base_checkpoint(
        model=nn.Linear(1, 1),
        model_type="teacher",
        model_kwargs={},
        epoch=0,
        best_epoch=0,
        best_score=1.0,
        fold=0,
        split={
            "train_identities": ["A"],
            "validation_identities": ["B"],
            "test_identities": ["C"],
        },
        aux_center=np.empty(0, dtype=np.float32),
        aux_scale=np.empty(0, dtype=np.float32),
        run_signature="signature",
        rng_state=state,
        cache_provenance={"content_sha256": "1" * 64},
        distillation_teacher_provenance=None,
    )
    assert checkpoint["format_version"] == 2
    assert set(checkpoint["rng_state"]) >= {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "sampler_generator",
    }


def test_resume_rejects_changed_cache_provenance(tmp_path: Path) -> None:
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    torch.save(
        {
            "run_signature": "same-signature",
            "cache_provenance": {"content_sha256": "1" * 64},
        },
        fold_dir / "teacher_last.pt",
    )
    args = _TRAIN.parse_args(["--resume", "--device", "cpu"])
    loader = DataLoader(TensorDataset(torch.zeros((1, 1))))

    with pytest.raises(RuntimeError, match="resume cache provenance mismatch"):
        _TRAIN.train_stage(
            model=nn.Linear(1, 1),
            model_type="teacher",
            model_kwargs={},
            train_loader=loader,
            validation_loader=loader,
            metadata=pd.DataFrame(),
            device=torch.device("cpu"),
            fold_dir=fold_dir,
            fold=0,
            split={
                "train_identities": ["A"],
                "validation_identities": ["B"],
                "test_identities": ["C"],
            },
            aux_center=np.empty(0, dtype=np.float32),
            aux_scale=np.empty(0, dtype=np.float32),
            run_signature="same-signature",
            args=args,
            quality_positive_weight=1.0,
            auxiliary_layout=None,
            cache_provenance={"content_sha256": "2" * 64},
        )


def test_alias_detailed_report_uses_requested_range_and_tolerance() -> None:
    bundle = PredictionBundle(
        index=np.arange(2),
        target=np.asarray([20.5, 10.0], dtype=np.float32),
        prediction=np.asarray([20.5, 10.0], dtype=np.float32),
        rr_std=np.ones(2, dtype=np.float32),
        uncertainty=np.zeros(2, dtype=np.float32),
        quality=np.ones(2, dtype=np.float32),
        observable=np.ones(2, dtype=bool),
        reference_valid=np.ones(2, dtype=bool),
        spike_rate=np.zeros(2, dtype=np.float32),
        radar_weights=np.full((2, 3), 1 / 3, dtype=np.float32),
        alias_probability=np.asarray([0.9, 0.1], dtype=np.float32),
    )
    metadata = pd.DataFrame(
        {"identity": ["A", "B"], "classical_rr_bpm": [10.0, 10.0]}
    )
    permissive = detailed_prediction_summary(
        bundle,
        metadata,
        bootstrap_samples=20,
        coverages=[1.0],
        rr_range=[6.0, 25.0],
        alias_target_tolerance_bpm=0.6,
    )
    strict = detailed_prediction_summary(
        bundle,
        metadata,
        bootstrap_samples=20,
        coverages=[1.0],
        rr_range=[6.0, 25.0],
        alias_target_tolerance_bpm=0.4,
    )
    narrow = detailed_prediction_summary(
        bundle,
        metadata,
        bootstrap_samples=20,
        coverages=[1.0],
        rr_range=[6.0, 15.0],
        alias_target_tolerance_bpm=0.6,
    )
    assert permissive["alias_gate"]["n"] == 2
    assert permissive["alias_gate"]["rr_range_bpm"] == [6.0, 25.0]
    assert permissive["alias_gate"]["target_tolerance_bpm"] == 0.6
    assert strict["alias_gate"]["n"] == 1
    assert narrow["alias_gate"]["n"] == 1


def test_custom_split_exact_cover_and_explicit_indices(tmp_path: Path) -> None:
    split_path, cache_dir, metadata = _write_custom_split_fixture(tmp_path)
    authority = load_identity_split_authority(
        split_path, metadata=metadata, cache_dir=cache_dir
    )
    explicit = authority.explicit_indices(metadata, include_invalid=False)

    assert authority.fold_id == 7
    assert set(explicit.split) == {
        "train_identities",
        "validation_identities",
        "prediction_identities",
        "excluded_identities",
        "scaler_identities",
    }
    assert explicit.train_index.tolist() == [0]
    assert explicit.validation_index.tolist() == [2, 3]
    assert explicit.prediction_index.tolist() == [4, 5]
    selected = np.concatenate(
        [
            explicit.train_index,
            explicit.validation_index,
            explicit.prediction_index,
        ]
    )
    assert not set(metadata.iloc[selected]["identity"]) & {"D"}
    assert authority.checkpoint_provenance()["scaler_identities"] == ["A"]


def test_custom_split_binds_exact_metadata_bytes_content_and_object(
    tmp_path: Path,
) -> None:
    split_path, cache_dir, metadata = _write_custom_split_fixture(tmp_path)
    authority = load_identity_split_authority(
        split_path, metadata=metadata, cache_dir=cache_dir
    )
    provenance = authority.checkpoint_provenance()

    assert provenance["authority_receipt_version"] == 2
    assert provenance["metadata_row_count"] == len(metadata)
    assert len(provenance["metadata_source_content_sha256"]) == 64
    assert len(provenance["metadata_row_roles_sha256"]) == 64
    assert len(authority.metadata_file_bindings) == 4

    with pytest.raises(RuntimeError, match="exact loader-bound metadata object"):
        authority.explicit_indices(metadata.copy(deep=True), include_invalid=False)
    copied_authority = copy.copy(authority)
    with pytest.raises(RuntimeError, match="not issued by the loader"):
        copied_authority.explicit_indices(metadata, include_invalid=False)

    object.__setattr__(authority, "train_identities", ("C",))
    with pytest.raises(RuntimeError, match="authority changed after loader issuance"):
        authority.explicit_indices(metadata, include_invalid=False)


def test_custom_split_rejects_session_identity_role_transplant_before_issuance(
    tmp_path: Path,
) -> None:
    split_path, cache_dir, metadata = _write_custom_split_fixture(tmp_path)
    forged = metadata.copy(deep=True)
    forged.loc[forged["session_id"] == "SA", "identity"] = "C"
    forged.loc[forged["session_id"] == "SC", "identity"] = "A"

    with pytest.raises(ValueError, match="exact cache metadata byte snapshot"):
        load_identity_split_authority(
            split_path, metadata=forged, cache_dir=cache_dir
        )


def test_custom_split_rejects_in_place_reference_role_mutation(
    tmp_path: Path,
) -> None:
    split_path, cache_dir, metadata = _write_custom_split_fixture(tmp_path)
    authority = load_identity_split_authority(
        split_path, metadata=metadata, cache_dir=cache_dir
    )
    metadata.loc[0, "reference_valid"] = False

    with pytest.raises(RuntimeError, match="metadata changed after authority issuance"):
        authority.explicit_indices(metadata, include_invalid=False)


def test_custom_split_rejects_non_boolean_reference_roles(tmp_path: Path) -> None:
    split_path, cache_dir, metadata = _write_custom_split_fixture(tmp_path)
    forged = metadata.copy(deep=True)
    forged["reference_valid"] = forged["reference_valid"].map(
        {True: "true", False: "false"}
    )

    with pytest.raises(ValueError, match="reference_valid.*exact booleans"):
        load_identity_split_authority(
            split_path, metadata=forged, cache_dir=cache_dir
        )


def test_custom_split_rejects_metadata_path_change_after_private_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_path, cache_dir, metadata = _write_custom_split_fixture(tmp_path)
    original = split_authority_module._read_regular_file_snapshot

    def mutate_after_snapshot(path: Path, label: str):
        payload, digest, byte_count = original(path, label)
        if label == "cache metadata SA":
            path.write_bytes(payload + b"\n")
        return payload, digest, byte_count

    monkeypatch.setattr(
        split_authority_module,
        "_read_regular_file_snapshot",
        mutate_after_snapshot,
    )
    with pytest.raises(ValueError, match="cache metadata SA changed"):
        load_identity_split_authority(
            split_path, metadata=metadata, cache_dir=cache_dir
        )


def test_custom_split_rejects_manifest_and_referenced_hash_mutation(
    tmp_path: Path,
) -> None:
    split_path, cache_dir, metadata = _write_custom_split_fixture(tmp_path)
    document = json.loads(split_path.read_text(encoding="utf-8"))
    document["identities"]["prediction"] = ["D"]
    split_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical content SHA-256 mismatch"):
        load_identity_split_authority(
            split_path, metadata=metadata, cache_dir=cache_dir
        )

    split_path, cache_dir, metadata = _write_custom_split_fixture(
        tmp_path / "referenced"
    )
    document = json.loads(split_path.read_text(encoding="utf-8"))
    folds_path = Path(document["fold_assignments"]["path"])
    folds_path.write_text(
        json.dumps({"identity_to_fold": {"A": 3, "B": 2, "C": 1, "D": 0}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fold_assignments SHA-256 mismatch"):
        load_identity_split_authority(
            split_path, metadata=metadata, cache_dir=cache_dir
        )


def test_custom_split_rejects_incomplete_cover_overlap_and_scaler_mismatch(
    tmp_path: Path,
) -> None:
    cases = [
        (
            {
                "train": ["A"],
                "validation": ["B"],
                "prediction": ["C"],
                "excluded": [],
                "scaler": ["A"],
            },
            "do not exactly cover",
        ),
        (
            {
                "train": ["A"],
                "validation": ["A", "B"],
                "prediction": ["C"],
                "excluded": ["D"],
                "scaler": ["A"],
            },
            "train/validation overlap",
        ),
        (
            {
                "train": ["A"],
                "validation": ["B"],
                "prediction": ["C"],
                "excluded": ["D"],
                "scaler": ["B"],
            },
            "scaler identities must exactly equal train identities",
        ),
    ]
    for index, (identities, message) in enumerate(cases):
        case_root = tmp_path / str(index)
        split_path, cache_dir, metadata = _write_custom_split_fixture(
            case_root, identities=identities
        )
        with pytest.raises(ValueError, match=message):
            load_identity_split_authority(
                split_path, metadata=metadata, cache_dir=cache_dir
            )


def test_custom_split_rejects_excluded_scaler_leak_and_session_identity_crossing(
    tmp_path: Path,
) -> None:
    split_path, cache_dir, metadata = _write_custom_split_fixture(tmp_path / "clean")
    authority = load_identity_split_authority(
        split_path, metadata=metadata, cache_dir=cache_dir
    )
    with pytest.raises(ValueError, match="scaler identity binding mismatch"):
        authority.validate_scaler_indices(metadata, np.asarray([0, 6]))

    crossed = metadata.copy()
    crossed.loc[1, "identity"] = "B"
    with pytest.raises(ValueError, match="crosses identities"):
        load_identity_split_authority(
            split_path, metadata=crossed, cache_dir=cache_dir
        )


def test_custom_run_bypasses_rotating_split_and_oof_and_never_loads_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_path, cache_dir, metadata = _write_custom_split_fixture(tmp_path)
    metadata = metadata.assign(
        rr_bpm=np.linspace(10.0, 17.0, len(metadata)),
        reference_quality=1.0,
        reference_sigma_bpm=0.5,
        radar_observable=True,
        classical_rr_bpm=np.linspace(10.0, 17.0, len(metadata)),
        classical_confidence=1.0,
    )
    metadata = _write_cache_metadata_files(cache_dir, metadata)
    auxiliary = np.ones((len(metadata), 61), dtype=np.float32)
    auxiliary[:, 0] = np.arange(len(metadata))
    cache = FeatureCache(
        maps=np.ones((len(metadata), 3, 2, 4), dtype=np.float16),
        aux=auxiliary,
        metadata=metadata,
        frequencies_hz=np.asarray([0.1, 0.2], dtype=np.float32),
        provenance=_legacy_cache_provenance(tmp_path),
    )
    def fake_cache_loader(path: Path, **kwargs: object) -> FeatureCache:
        assert Path(path) == cache_dir
        assert kwargs == {
            "require_acquisition_contract": False,
            "require_scientific_eligible": False,
        }
        return cache

    monkeypatch.setattr(_TRAIN, "load_feature_cache", fake_cache_loader)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("legacy split/OOF helper must not run in custom mode")

    monkeypatch.setattr(_TRAIN, "make_fold_assignments", forbidden)
    monkeypatch.setattr(_TRAIN, "_fold_split", forbidden)
    monkeypatch.setattr(_TRAIN, "aggregate_oof", forbidden)
    scaler_indices: list[np.ndarray] = []

    def fake_scaler(aux: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaler_indices.append(np.asarray(indices).copy())
        return (
            np.zeros(aux.shape[1], dtype=np.float32),
            np.ones(aux.shape[1], dtype=np.float32),
        )

    monkeypatch.setattr(_TRAIN, "fit_aux_scaler", fake_scaler)
    transformed_rows: list[np.ndarray] = []

    def fake_transform(
        values: np.ndarray, center: np.ndarray, scale: np.ndarray
    ) -> np.ndarray:
        del center, scale
        transformed_rows.append(values[:, 0].astype(np.int64))
        return values.astype(np.float32)

    monkeypatch.setattr(_TRAIN, "transform_aux", fake_transform)
    loader_indices: list[np.ndarray] = []

    def fake_loader(
        cache: FeatureCache,
        aux_scaled: np.ndarray,
        indices: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        del cache, aux_scaled, kwargs
        result = np.asarray(indices).copy()
        loader_indices.append(result)
        return result

    monkeypatch.setattr(_TRAIN, "make_loader", fake_loader)
    monkeypatch.setattr(_TRAIN, "build_model", lambda *args, **kwargs: nn.Linear(1, 1))

    def fake_train_stage(**kwargs: object) -> tuple[nn.Module, dict[str, object]]:
        provenance = kwargs["split_authority_provenance"]
        assert isinstance(provenance, dict)
        assert provenance["excluded_identities"] == ["D"]
        assert provenance["scaler_identities"] == ["A"]
        cache_binding = kwargs["cache_provenance"]
        assert isinstance(cache_binding, dict)
        assert cache_binding["classification"] == "legacy"
        return kwargs["model"], {"best_validation_macro_mae": 1.0}  # type: ignore[return-value]

    monkeypatch.setattr(_TRAIN, "train_stage", fake_train_stage)

    def fake_predict(
        model: nn.Module,
        loader: np.ndarray,
        device: torch.device,
        **kwargs: object,
    ) -> PredictionBundle:
        del model, device, kwargs
        count = len(loader)
        target = metadata.iloc[loader]["rr_bpm"].to_numpy(np.float32)
        return PredictionBundle(
            index=loader,
            target=target,
            prediction=target,
            rr_std=np.zeros(count, dtype=np.float32),
            uncertainty=np.zeros(count, dtype=np.float32),
            quality=np.ones(count, dtype=np.float32),
            observable=np.ones(count, dtype=bool),
            reference_valid=np.ones(count, dtype=bool),
            spike_rate=np.zeros(count, dtype=np.float32),
            radar_weights=np.full((count, 3), 1 / 3, dtype=np.float32),
        )

    monkeypatch.setattr(_TRAIN, "predict", fake_predict)
    monkeypatch.setattr(
        _TRAIN,
        "detailed_prediction_summary",
        lambda *args, **kwargs: {"overall": {"mae": 0.0}},
    )
    saved: list[Path] = []
    monkeypatch.setattr(
        _TRAIN,
        "save_prediction_bundle",
        lambda path, *args, **kwargs: saved.append(Path(path)),
    )

    args = _TRAIN.parse_args(
        [
            "--identity-split-manifest",
            str(split_path),
            "--cache-dir",
            str(cache_dir),
            "--cache-trust-mode",
            "legacy",
            "--output-dir",
            str(tmp_path / "run"),
            "--model",
            "teacher",
            "--no-causal-history",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--bootstrap-samples",
            "2",
        ]
    )
    report = _TRAIN.run(args)

    assert [metadata.iloc[value]["identity"].unique().tolist() for value in scaler_indices] == [["A"]]
    loaded_identities = [set(metadata.iloc[value]["identity"]) for value in loader_indices]
    assert loaded_identities == [{"A"}, {"B"}, {"C"}]
    assert all("D" not in identities for identities in loaded_identities)
    assert transformed_rows[0].tolist() == [0, 2, 3, 4, 5]
    assert saved == [tmp_path / "run" / "fold_7" / "teacher_prediction_predictions.npz"]
    assert report["oof"] == {}
    assert report["claim_classification"] == "retrospective_legacy_noncommercial"
    assert report["commercial_claim_allowed"] is False
    assert report["prediction"]["teacher"]["overall"]["mae"] == 0.0
    run_config = json.loads(
        (tmp_path / "run" / "run_config.json").read_text(encoding="utf-8")
    )
    assert run_config["arguments"]["cache_trust_mode"] == "legacy"
    assert run_config["cache_provenance"]["content_sha256"] == (
        cache.provenance.content_sha256
    )
    assert run_config["claim_classification"] == "retrospective_legacy_noncommercial"
    assert run_config["commercial_claim_allowed"] is False


def test_legacy_split_and_parser_behavior_remain_available() -> None:
    args = _TRAIN.parse_args([])
    assert args.identity_split_manifest is None
    assert args.cache_trust_mode == "scientific"
    legacy_args = _TRAIN.parse_args(["--cache-trust-mode", "legacy"])
    assert legacy_args.cache_trust_mode == "legacy"
    with pytest.raises(SystemExit):
        _TRAIN.parse_args(["--cache-trust-mode", "acquisition-diagnostic"])
    metadata = pd.DataFrame(
        {
            "identity": ["A", "A", "B", "B", "C", "C"],
            "reference_valid": [True, False, True, True, True, True],
        }
    )
    assignment = np.asarray([0, 0, 1, 1, 2, 2])
    train, validation, test, split = _TRAIN._fold_split(
        metadata, assignment, fold=0, n_splits=3, include_invalid=False
    )
    assert train.tolist() == [4, 5]
    assert validation.tolist() == [2, 3]
    assert test.tolist() == [0]
    assert set(split) == {
        "train_identities",
        "validation_identities",
        "test_identities",
    }


@pytest.mark.parametrize(
    ("trust_mode", "expected"),
    [
        (
            "scientific",
            {
                "require_acquisition_contract": True,
                "require_scientific_eligible": True,
            },
        ),
        (
            "legacy",
            {
                "require_acquisition_contract": False,
                "require_scientific_eligible": False,
            },
        ),
    ],
)
def test_run_passes_explicit_cache_trust_policy_to_loader(
    trust_mode: str,
    expected: dict[str, bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LoaderReached(RuntimeError):
        pass

    observed: dict[str, object] = {}

    def stop_at_loader(path: Path, **kwargs: object) -> FeatureCache:
        observed["path"] = path
        observed.update(kwargs)
        raise LoaderReached

    monkeypatch.setattr(_TRAIN, "load_feature_cache", stop_at_loader)
    args = _TRAIN.parse_args(
        ["--cache-trust-mode", trust_mode, "--device", "cpu"]
    )
    with pytest.raises(LoaderReached):
        _TRAIN.run(args)
    assert observed["path"] == args.cache_dir
    assert {key: observed[key] for key in expected} == expected


@pytest.mark.parametrize(
    ("trust_mode", "schema_version", "classification", "scientific", "mask"),
    [
        (
            "scientific",
            "snn_rr.feature_cache_acquisition.v2",
            "acquisition_scientific",
            True,
            np.asarray([[[True], [False], [True]]], dtype=np.bool_),
        ),
    ],
)
def test_acquisition_training_fails_closed_without_complete_structural_timing(
    trust_mode: str,
    schema_version: str,
    classification: str,
    scientific: bool,
    mask: np.ndarray | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = CacheProvenance(
        classification=classification,
        root_manifest_path=str(tmp_path / "manifest.json"),
        root_manifest_sha256="1" * 64,
        root_manifest_content_sha256="2" * 64,
        acquisition_schema_version=schema_version,
        acquisition_mode="strict",
        scientific_eligible=scientific,
        config_sha256="3" * 64,
        pipeline_sha256="4" * 64,
        reconstruction_content_sha256="5" * 64,
        inventory_sha256="6" * 64,
        inventory_file_count=5,
        selected_sessions=("S01_A",),
    )
    cache = FeatureCache(
        maps=np.ones((1, 3, 2, 4), dtype=np.float16),
        aux=np.ones((1, 37), dtype=np.float32),
        metadata=_timing_mask_metadata(1),
        frequencies_hz=np.asarray([0.1, 0.2], dtype=np.float32),
        provenance=provenance,
        radar_timing_valid_mask=mask,
    )
    monkeypatch.setattr(_TRAIN, "load_feature_cache", lambda *args, **kwargs: cache)
    args = _TRAIN.parse_args(
        ["--cache-trust-mode", trust_mode, "--device", "cpu"]
    )

    with pytest.raises((ValueError, RuntimeError), match="timing mask|interval"):
        _TRAIN.run(args)


def test_diagnostic_cache_is_rejected_before_training_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = CacheProvenance(
        classification="acquisition_diagnostic",
        root_manifest_path=str(tmp_path / "cache" / "manifest.json"),
        root_manifest_sha256="1" * 64,
        root_manifest_content_sha256="2" * 64,
        acquisition_schema_version="snn_rr.feature_cache_acquisition.v2",
        acquisition_mode="diagnostic",
        scientific_eligible=False,
        config_sha256="3" * 64,
        pipeline_sha256="4" * 64,
        reconstruction_content_sha256="5" * 64,
        inventory_sha256="6" * 64,
        inventory_file_count=5,
        selected_sessions=("S01_A",),
    )
    cache = FeatureCache(
        maps=np.ones((1, 3, 2, 4), dtype=np.float16),
        aux=np.ones((1, 37), dtype=np.float32),
        metadata=_timing_mask_metadata(1),
        frequencies_hz=np.asarray([0.1, 0.2], dtype=np.float32),
        provenance=provenance,
        radar_timing_valid_mask=np.ones((1, 3, 1), dtype=np.bool_),
    )
    events: list[str] = []
    monkeypatch.setattr(_TRAIN, "load_feature_cache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(
        _TRAIN,
        "seed_everything",
        lambda *args, **kwargs: events.append("seed"),
    )
    monkeypatch.setattr(
        _TRAIN.torch.cuda,
        "is_available",
        lambda: events.append("cuda") or True,
    )
    monkeypatch.setattr(
        _TRAIN,
        "fit_aux_scaler",
        lambda *args, **kwargs: events.append("scaler"),
    )
    output_dir = tmp_path / "must_not_exist"
    args = _TRAIN.parse_args(
        [
            "--cache-trust-mode",
            "scientific",
            "--device",
            "cuda",
            "--output-dir",
            str(output_dir),
        ]
    )

    with pytest.raises(ValueError, match="verified v2 full-cohort"):
        _TRAIN.run(args)

    assert events == []
    assert not output_dir.exists()


def test_removed_diagnostic_trust_mode_fails_before_cache_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _TRAIN.parse_args(["--cache-trust-mode", "scientific"])
    args.cache_trust_mode = "acquisition-diagnostic"
    monkeypatch.setattr(
        _TRAIN,
        "load_feature_cache",
        lambda *args, **kwargs: pytest.fail("cache loader must not be reached"),
    )

    with pytest.raises(ValueError, match="inspection-only"):
        _TRAIN.run(args)
