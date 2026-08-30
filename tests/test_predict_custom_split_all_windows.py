from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from snn_rr.cache import CacheProvenance
import snn_rr.split_authority as split_authority_module
from snn_rr.split_authority import canonical_content_sha256, sha256_file


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "predict_custom_split_all_windows.py"
)
_SPEC = importlib.util.spec_from_file_location("predict_custom_split_all_windows", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_PREDICT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PREDICT
_SPEC.loader.exec_module(_PREDICT)


class _Dataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, labels: tuple[float, float]) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "map": torch.full((3, 2, 4), index + 1, dtype=torch.float32),
            "radar_mask": torch.ones(3, dtype=torch.bool),
            "aux": torch.tensor([index, 1.0], dtype=torch.float32),
            "rr": torch.tensor(self.labels[index]),
            "reference_valid": torch.tensor(index == 0),
            "observable": torch.tensor(False),
            "reference_quality": torch.tensor(999.0),
            "reference_sigma": torch.tensor(-999.0),
            "classical_rr": torch.tensor(777.0),
            "classical_confidence": torch.tensor(-777.0),
            "index": torch.tensor(index, dtype=torch.int64),
        }


class _Model(nn.Module):
    def forward(
        self, x: torch.Tensor, radar_mask: torch.Tensor, aux: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logits = torch.stack((x.mean((1, 2, 3)), aux[:, 0], aux[:, 1]), dim=1)
        probability = logits.softmax(1)
        bins = x.new_tensor([10.0, 11.0, 12.0])
        top_probability, top_index = probability.topk(3, dim=1)
        expected = (probability * bins).sum(1)
        return {
            "expected_rr": expected,
            "map_rr": bins[probability.argmax(1)],
            "rr_std": torch.full_like(expected, 0.5),
            "quality": torch.full_like(expected, 0.8),
            "radar_weights": radar_mask.float() / radar_mask.sum(1, keepdim=True),
            "posterior_entropy": -(probability * probability.log()).sum(1),
            "topk_rr": bins[top_index],
            "topk_probability": top_probability,
            "probabilities": probability,
            "alias_probability": torch.full_like(expected, 0.2),
            "spike_rate_per_sample": torch.full_like(expected, 0.1),
        }


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["S_A", "S_A", "S_B", "S_C", "S_D"],
            "identity": ["A", "A", "B", "C", "D"],
            "protocol": ["rest", "walk", "rest", "rest", "rest"],
            "window_number": [0, 1, 0, 0, 0],
            "reference_valid": [True, False, True, True, True],
            "rr_bpm": [12.0, 999.0, 13.0, 14.0, 15.0],
        }
    )


def _write_authority(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    metadata = _metadata()
    for session_id in ("S_A", "S_B", "S_C", "S_D"):
        session_dir = cache_dir / session_id
        session_dir.mkdir()
        metadata.loc[metadata["session_id"] == session_id].to_csv(
            session_dir / "metadata.csv", index=False
        )
    cache_manifest = {
        "sessions": [
            {"session_id": "S_A", "status": "ok"},
            {"session_id": "S_B", "status": "ok"},
            {"session_id": "S_C", "status": "ok"},
            {"session_id": "S_D", "status": "ok"},
        ]
    }
    cache_path = cache_dir / "manifest.json"
    cache_path.write_text(json.dumps(cache_manifest), encoding="utf-8")
    fold_path = tmp_path / "folds.json"
    fold_path.write_text(
        json.dumps({"identity_to_fold": {"A": 0, "B": 1, "C": 2, "D": 3}}),
        encoding="utf-8",
    )
    document = {
        "schema_version": 1,
        "fold_id": 7,
        "identities": {
            "train": ["B"],
            "validation": ["C"],
            "prediction": ["A"],
            "excluded": ["D"],
            "scaler": ["B"],
        },
        "fold_assignments": {"path": str(fold_path), "sha256": sha256_file(fold_path)},
        "cache": {
            "manifest_path": str(cache_path),
            "manifest_sha256": sha256_file(cache_path),
        },
    }
    document["content_sha256"] = canonical_content_sha256(document)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    authority = _PREDICT.load_identity_split_authority(
        manifest_path, metadata=metadata, cache_dir=cache_dir
    )
    return cache_dir, manifest_path, fold_path, authority, metadata


def _cache_provenance() -> CacheProvenance:
    return CacheProvenance(
        classification="legacy",
        root_manifest_path="/tmp/unit-cache/manifest.json",
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
        selected_sessions=("S_A", "S_B", "S_C", "S_D"),
    )


def _provenance_document(cache: object) -> dict[str, object]:
    document = cache.provenance.to_dict()
    document["content_sha256"] = cache.provenance.content_sha256
    return document


def _cache(metadata: pd.DataFrame | None = None) -> object:
    return _PREDICT.FeatureCache(
        maps=np.zeros((5, 3, 2, 4), dtype=np.float16),
        aux=np.asarray([[1.0, 2.0], [2.0, 3.0], [5.0, 7.0], [8.0, 9.0], [3.0, 4.0]], dtype=np.float32),
        metadata=_metadata() if metadata is None else metadata,
        frequencies_hz=np.asarray([0.1, 0.2], dtype=np.float32),
        provenance=_cache_provenance(),
    )


def _checkpoint(path: Path, authority: object, cache: object) -> dict[str, object]:
    train_index = np.asarray([2], dtype=np.int64)
    center, scale = _PREDICT.fit_aux_scaler(cache.aux, train_index)
    checkpoint: dict[str, object] = {
        "format_version": 2,
        "model_type": "snn",
        "model_kwargs": {},
        "model_state": {"rr_bins": torch.tensor([10.0, 11.0, 12.0])},
        "fold": authority.fold_id,
        "run_signature": "run",
        "split_authority_provenance": authority.checkpoint_provenance(),
        "split": {
            "train_identities": ["B"],
            "validation_identities": ["C"],
            "prediction_identities": ["A"],
            "excluded_identities": ["D"],
            "scaler_identities": ["B"],
        },
        "aux_center": torch.from_numpy(center),
        "aux_scale": torch.from_numpy(scale),
        "cache_provenance": _provenance_document(cache),
    }
    torch.save(checkpoint, path)
    return checkpoint


def _run_config(authority: object) -> dict[str, object]:
    cache = _cache()
    return {
        "run_signature": "run",
        "cache_provenance": _provenance_document(cache),
        "arguments": {
            "identity_split_manifest_sha256": authority.content_sha256,
            "include_invalid": False,
            "use_aux": True,
            "input_branches": 2,
            "rr_range": [5.0, 45.0],
            "rr_bin_width": 0.25,
        },
    }


def _structural_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["S_A", "S_A", "S_A"],
            "identity": ["A", "A", "A"],
            "protocol": ["rest", "rest", "rest"],
            "window_number": [0, 1, 2],
            "reference_valid": [False, False, False],
            "rr_bpm": [np.nan, np.nan, np.nan],
            "reference_quality": [0.0, 0.0, 0.0],
            "reference_sigma_bpm": [1.0, 1.0, 1.0],
            "radar_observable": [False, False, False],
            "classical_rr_bpm": [99.0, 12.0, 13.0],
            "classical_confidence": [0.9, 0.8, 0.7],
            "radar_peak_spread_bpm": [1.0, 2.0, 3.0],
        }
    )


def _structural_cache() -> object:
    timing = np.ones((3, 3, 5), dtype=np.bool_)
    timing[0, 1, 2] = False
    return _PREDICT.FeatureCache(
        maps=np.full((3, 3, 2, 4), 7.0, dtype=np.float16),
        # F=1 gives the valid base layout 3*(2*F+8)+2*F+5 = 37.
        aux=np.full((3, 37), 11.0, dtype=np.float32),
        metadata=_structural_metadata(),
        frequencies_hz=np.asarray([0.1, 0.2], dtype=np.float32),
        provenance=_cache_provenance(),
        radar_timing_valid_mask=timing,
    )


def test_forward_is_sanitized_and_reference_values_cannot_change_predictions() -> None:
    model = _Model()
    _PREDICT.validate_label_free_forward_interface(model)
    first = _PREDICT.predict_label_free(
        model, DataLoader(_Dataset((12.0, 999.0)), batch_size=2), torch.device("cpu"), amp=False
    )
    second = _PREDICT.predict_label_free(
        model, DataLoader(_Dataset((-500.0, 500.0)), batch_size=2), torch.device("cpu"), amp=False
    )
    assert np.array_equal(first.prediction, second.prediction)
    assert np.array_equal(first.posterior_probability, second.posterior_probability)
    assert not np.array_equal(first.target, second.target)


def test_prediction_owner_is_all_row_exact_cover_including_invalid_reference() -> None:
    index = _PREDICT.validate_prediction_ownership(_metadata(), ["A"])
    assert np.array_equal(index, np.asarray([0, 1]))
    assert not _metadata().iloc[index]["reference_valid"].iloc[1]
    with pytest.raises(RuntimeError, match="not every"):
        _PREDICT.validate_prediction_ownership(_metadata(), ["A", "MISSING"])


def test_custom_cache_preserves_mask_provenance_and_excludes_invalid_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _structural_cache()
    _, history_names = _PREDICT.append_mask_aware_causal_history_features(raw)
    run_config = {
        "arguments": {
            "cache_dir": str(tmp_path),
            "map_branch": "both",
            "input_branches": 2,
            "use_aux": True,
            "causal_history": True,
        },
        "cache_shape": {
            "maps": list(raw.maps.shape),
            "aux": [len(raw.metadata), raw.aux.shape[1] + len(history_names)],
        },
        "causal_history_feature_names": history_names,
    }
    monkeypatch.setattr(_PREDICT, "load_feature_cache", lambda *args, **kwargs: raw)

    prepared, base_aux_dim = _PREDICT.prepare_custom_cache(tmp_path, run_config)

    assert base_aux_dim == 37
    assert prepared.provenance == raw.provenance
    np.testing.assert_array_equal(
        prepared.radar_timing_valid_mask, raw.radar_timing_valid_mask
    )
    history_column = {
        name: base_aux_dim + offset for offset, name in enumerate(history_names)
    }
    # Window 0 has one invalid radar interval, so its nonzero classical
    # payload cannot appear as window 1's apparently available history.
    assert prepared.aux[1, history_column["history_lag_1_available"]] == 0.0
    assert prepared.aux[1, history_column["history_lag_1_classical_rr_bpm"]] == 0.0
    # The valid window 1 remains available to the strictly later window 2.
    assert prepared.aux[2, history_column["history_lag_1_available"]] == 1.0
    assert prepared.aux[2, history_column["history_lag_1_classical_rr_bpm"]] == 12.0


def test_custom_loader_zeros_nonzero_payload_for_invalid_timing() -> None:
    cache = _structural_cache()
    loader = _PREDICT.make_loader(
        cache,
        np.full_like(cache.aux, 17.0),
        np.asarray([0], dtype=np.int64),
        batch_size=1,
        workers=0,
        device=torch.device("cpu"),
        seed=1,
        train=False,
        auxiliary_layout=_PREDICT.infer_auxiliary_layout(37),
    )
    batch = next(iter(loader))

    assert batch["radar_mask"].tolist() == [[True, False, True]]
    assert torch.count_nonzero(batch["map"][0, 1]).item() == 0
    assert torch.count_nonzero(batch["map"][0, 0]).item() > 0
    aux = batch["aux"][0]
    # F=1: radar-1 spectra [2:4], radar-1 scalars [14:22], fused [30:37].
    assert torch.count_nonzero(aux[2:4]).item() == 0
    assert torch.count_nonzero(aux[14:22]).item() == 0
    assert torch.count_nonzero(aux[30:37]).item() == 0
    assert torch.count_nonzero(aux[0:2]).item() > 0


def test_manifest_and_referenced_hash_tampering_fail_closed(tmp_path: Path) -> None:
    cache_dir, manifest_path, fold_path, _, _ = _write_authority(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["fold_id"] = 8
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical content SHA-256 mismatch"):
        _PREDICT.load_identity_split_authority(
            manifest_path, metadata=_metadata(), cache_dir=cache_dir
        )

    # Restore a valid manifest, then alter the separately content-bound fold file.
    _, manifest_path, fold_path, _, _ = _write_authority(tmp_path / "second")
    fold_path.write_text(json.dumps({"identity_to_fold": {"A": 99}}), encoding="utf-8")
    with pytest.raises(ValueError, match="fold_assignments SHA-256 mismatch"):
        _PREDICT.load_identity_split_authority(
            manifest_path,
            metadata=_metadata(),
            cache_dir=manifest_path.parent / "cache",
        )


def test_split_loader_rejects_reference_changed_after_verified_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir, manifest_path, fold_path, _, _ = _write_authority(tmp_path)
    original = split_authority_module._read_regular_json_snapshot

    def mutate_after_snapshot(path: Path, label: str):
        document, digest = original(path, label)
        if label == "fold_assignments":
            path.write_text(
                json.dumps(
                    {"identity_to_fold": {"A": 1, "B": 0, "C": 2, "D": 3}}
                ),
                encoding="utf-8",
            )
        return document, digest

    monkeypatch.setattr(
        split_authority_module,
        "_read_regular_json_snapshot",
        mutate_after_snapshot,
    )
    with pytest.raises(ValueError, match="fold assignments changed"):
        _PREDICT.load_identity_split_authority(
            manifest_path, metadata=_metadata(), cache_dir=cache_dir
        )
    assert fold_path.is_file()


def test_checkpoint_split_fold_hash_and_scaler_tampering_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, authority, metadata = _write_authority(tmp_path)
    cache = _cache(metadata)
    checkpoint_path = tmp_path / "snn_best.pt"
    checkpoint = _checkpoint(checkpoint_path, authority, cache)
    run_config = _run_config(authority)
    monkeypatch.setattr(_PREDICT, "validate_model_kwargs", lambda *args, **kwargs: None)
    validated = _PREDICT.validate_custom_checkpoint(
        checkpoint_path,
        authority=authority,
        cache=cache,
        base_aux_dim=2,
        run_config=run_config,
    )
    assert validated["fold"] == 7

    checkpoint["fold"] = 8
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match="checkpoint fold"):
        _PREDICT.validate_custom_checkpoint(
            checkpoint_path, authority=authority, cache=cache, base_aux_dim=2, run_config=run_config
        )
    checkpoint["fold"] = 7
    checkpoint["split_authority_provenance"] = dict(authority.checkpoint_provenance())
    checkpoint["split_authority_provenance"]["cache_manifest_sha256"] = "0" * 64
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match="split-authority provenance"):
        _PREDICT.validate_custom_checkpoint(
            checkpoint_path, authority=authority, cache=cache, base_aux_dim=2, run_config=run_config
        )
    checkpoint["split_authority_provenance"] = authority.checkpoint_provenance()
    checkpoint["aux_center"] = checkpoint["aux_center"] + 1
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match="train-only fitted"):
        _PREDICT.validate_custom_checkpoint(
            checkpoint_path, authority=authority, cache=cache, base_aux_dim=2, run_config=run_config
        )


def test_checkpoint_cache_provenance_is_exact_and_missing_authority_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, authority, metadata = _write_authority(tmp_path)
    cache = _cache(metadata)
    checkpoint_path = tmp_path / "snn_best.pt"
    checkpoint = _checkpoint(checkpoint_path, authority, cache)
    run_config = _run_config(authority)
    monkeypatch.setattr(_PREDICT, "validate_model_kwargs", lambda *args, **kwargs: None)

    changed_run = dict(run_config)
    changed_run["cache_provenance"] = dict(run_config["cache_provenance"])
    changed_run["cache_provenance"]["inventory_sha256"] = "9" * 64
    with pytest.raises(RuntimeError, match="run_config/loaded cache provenance"):
        _PREDICT.validate_custom_checkpoint(
            checkpoint_path,
            authority=authority,
            cache=cache,
            base_aux_dim=2,
            run_config=changed_run,
        )

    checkpoint["cache_provenance"] = dict(checkpoint["cache_provenance"])
    checkpoint["cache_provenance"]["inventory_sha256"] = "8" * 64
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match="checkpoint/loaded cache provenance"):
        _PREDICT.validate_custom_checkpoint(
            checkpoint_path,
            authority=authority,
            cache=cache,
            base_aux_dim=2,
            run_config=run_config,
        )

    missing = _PREDICT.FeatureCache(
        maps=cache.maps,
        aux=cache.aux,
        metadata=cache.metadata,
        frequencies_hz=cache.frequencies_hz,
    )
    with pytest.raises(RuntimeError, match="no verified provenance"):
        _PREDICT._verified_cache_provenance(missing)


def test_execution_input_snapshot_detects_midrun_replacement(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    run_config = tmp_path / "run_config.json"
    checkpoint.write_bytes(b"before")
    run_config.write_text("{}", encoding="utf-8")
    checkpoint_hash = _PREDICT._sha256_file(checkpoint)
    run_config_hash = _PREDICT._sha256_file(run_config)
    source_hashes = _PREDICT._runtime_source_hashes()

    _PREDICT._assert_execution_inputs_current(
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        run_config_path=run_config,
        run_config_sha256=run_config_hash,
        source_hashes=source_hashes,
    )
    checkpoint.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        _PREDICT._assert_execution_inputs_current(
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_hash,
            run_config_path=run_config,
            run_config_sha256=run_config_hash,
            source_hashes=source_hashes,
        )


def test_direct_entry_disk_binding_rejects_persistent_disk_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "entry.py"
    source.write_text("GENERATION = 'A'\n", encoding="utf-8")
    initial_binding = _PREDICT._capture_direct_entry_disk_binding(source)

    source.write_text("GENERATION = 'B'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="initialization-time disk binding"):
        _PREDICT._assert_direct_entry_disk_binding_current(initial_binding)

    assert source.read_text(encoding="utf-8") == "GENERATION = 'B'\n"


def test_runtime_source_hash_inventory_includes_upstream_cache_generation_stack() -> None:
    observed = set(_PREDICT._runtime_source_hashes())
    required = {
        "scripts/predict_custom_split_all_windows.py",
        "scripts/__init__.py",
        "scripts/predict_all_windows.py",
        "scripts/train.py",
        "scripts/build_features.py",
        "src/snn_rr/cache.py",
        "src/snn_rr/__init__.py",
        "src/snn_rr/data.py",
        "src/snn_rr/preprocess.py",
        "src/snn_rr/acquisition_contract.py",
        "src/snn_rr/acquisition_protocol.py",
        "src/snn_rr/synchronization.py",
        "src/snn_rr/radar_timing.py",
        "src/snn_rr/range_tracking.py",
        "src/snn_rr/split_authority.py",
    }
    assert required <= observed


def test_private_feature_cache_snapshot_ignores_same_inode_mutate_restore(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    session = cache / "S_A"
    session.mkdir(parents=True)
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "sessions": [
                    {"session_id": "S_A", "status": "ok", "window_count": 1}
                ]
            }
        ),
        encoding="utf-8",
    )
    (session / "manifest.json").write_text(
        json.dumps({"session_id": "S_A"}), encoding="utf-8"
    )
    np.save(
        session / "maps.npy",
        np.ones((1, 3, 2, 4), dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        session / "aux.npy", np.asarray([[1.0, 2.0]], np.float32), allow_pickle=False
    )
    pd.DataFrame(
        {
            "session_id": ["S_A"],
            "identity": ["A"],
            "protocol": ["rest"],
            "window_number": [0],
            "reference_valid": [False],
            "rr_bpm": [np.nan],
        }
    ).to_csv(session / "metadata.csv", index=False)
    np.save(
        session / "frequencies_hz.npy",
        np.asarray([0.1, 0.2], np.float32),
        allow_pickle=False,
    )

    snapshot = _PREDICT._materialize_feature_cache_snapshot(cache)
    source = session / "maps.npy"
    original = source.read_bytes()
    inode = source.stat().st_ino
    live = np.load(source, mmap_mode="r+", allow_pickle=False)
    live[...] = np.float32(77.0)
    live.flush()
    inventory_hash, inventory_count = _PREDICT._cache_inventory_sha256(
        snapshot.cache_dir, snapshot.session_ids
    )
    provenance = CacheProvenance(
        classification="legacy",
        root_manifest_path=str((cache / "manifest.json").resolve()),
        root_manifest_sha256="1" * 64,
        root_manifest_content_sha256="2" * 64,
        acquisition_schema_version=None,
        acquisition_mode=None,
        scientific_eligible=False,
        config_sha256=None,
        pipeline_sha256=None,
        reconstruction_content_sha256=None,
        inventory_sha256=inventory_hash,
        inventory_file_count=inventory_count,
        selected_sessions=("S_A",),
    )
    consumed = _PREDICT._load_feature_cache_snapshot_payload(
        snapshot, validated_provenance=provenance
    )
    assert np.all(consumed.maps == np.float32(1.0))
    source.write_bytes(original)
    assert source.stat().st_ino == inode
    with pytest.raises(RuntimeError, match="feature-cache input changed"):
        snapshot.assert_source_current()
    snapshot.assert_private_current()
    snapshot.cleanup()


def test_feature_cache_snapshot_rejects_session_path_traversal_without_escape(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    escaped_source = tmp_path / "escaped_session"
    escaped_source.mkdir()
    (cache / "manifest.json").write_text(
        json.dumps(
            {"sessions": [{"session_id": "../escaped_session", "status": "ok"}]}
        ),
        encoding="utf-8",
    )
    # These files made the old traversal path copyable outside the private
    # TemporaryDirectory.  The new validator must reject before touching any.
    for name in ("manifest.json", "metadata.csv"):
        (escaped_source / name).write_text("{}", encoding="utf-8")
    for name in ("maps.npy", "aux.npy", "frequencies_hz.npy"):
        np.save(escaped_source / name, np.asarray([1], dtype=np.float32))

    snapshot_parent = tmp_path / "snapshot-parent"
    with pytest.raises(RuntimeError, match="unsafe feature-cache session_id"):
        _PREDICT._materialize_feature_cache_snapshot(
            cache, parent=snapshot_parent
        )

    assert not (snapshot_parent / "escaped_session").exists()
    assert not list(snapshot_parent.glob(".custom-feature-cache-snapshot.*"))


def test_feature_cache_snapshot_cleans_temp_when_initialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(
        json.dumps({"sessions": []}), encoding="utf-8"
    )
    snapshot_parent = tmp_path / "snapshot-parent"
    monkeypatch.setattr(
        _PREDICT,
        "_copy_regular_file_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("synthetic snapshot copy failure")
        ),
    )
    with pytest.raises(OSError, match="synthetic snapshot copy failure"):
        _PREDICT._materialize_feature_cache_snapshot(
            cache, parent=snapshot_parent
        )
    assert not list(snapshot_parent.glob(".custom-feature-cache-snapshot.*"))


def test_feature_cache_snapshot_cleanup_survives_parent_rename_rebind(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache-rebind"
    session = cache / "S_A"
    session.mkdir(parents=True)
    (cache / "manifest.json").write_text(
        json.dumps({"sessions": [{"session_id": "S_A", "status": "ok"}]}),
        encoding="utf-8",
    )
    (session / "manifest.json").write_text("{}", encoding="utf-8")
    np.save(session / "maps.npy", np.zeros((1, 3, 1, 1), np.float32))
    np.save(session / "aux.npy", np.zeros((1, 1), np.float32))
    pd.DataFrame({"session_id": ["S_A"]}).to_csv(
        session / "metadata.csv", index=False
    )
    np.save(session / "frequencies_hz.npy", np.asarray([0.1], np.float32))
    snapshot_parent = tmp_path / "snapshot-parent-rebind"
    snapshot = _PREDICT._materialize_feature_cache_snapshot(
        cache,
        parent=snapshot_parent,
    )
    snapshot_name = snapshot.cache_dir.name
    moved_parent = tmp_path / "moved-snapshot-parent"
    snapshot_parent.rename(moved_parent)
    snapshot_parent.mkdir()

    snapshot.cleanup()

    assert not (moved_parent / snapshot_name).exists()
    assert not list(snapshot_parent.glob(".custom-feature-cache-snapshot.*"))


def test_stable_snapshot_directory_collision_preserves_preexisting_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "collision"
    existing = tmp_path / f".custom-feature-cache-snapshot.{token}"
    existing.mkdir()
    monkeypatch.setattr(_PREDICT.secrets, "token_hex", lambda _count: token)

    with pytest.raises(RuntimeError, match="cannot allocate"):
        _PREDICT._StablePrivateDirectory.create(
            tmp_path,
            prefix=".custom-feature-cache-snapshot.",
        )

    assert existing.is_dir()


def test_run_cleans_snapshot_after_post_materialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "run" / "fold" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (checkpoint.parent.parent / "run_config.json").write_text(
        "{}", encoding="utf-8"
    )
    cleaned = False
    private_cache = tmp_path / "private-cache"
    private_cache.mkdir()
    (private_cache / "manifest.json").write_text(
        json.dumps({"sessions": []}), encoding="utf-8"
    )

    class _FakeSnapshot:
        cache_dir = private_cache

        def cleanup(self) -> None:
            nonlocal cleaned
            cleaned = True

    monkeypatch.setattr(
        _PREDICT,
        "_materialize_feature_cache_snapshot",
        lambda *args, **kwargs: _FakeSnapshot(),
    )

    class _NoopOutputGuard:
        def assert_disjoint(self, _path: Path) -> None:
            return None

    monkeypatch.setattr(
        _PREDICT,
        "_build_output_isolation_guard",
        lambda **kwargs: _NoopOutputGuard(),
    )

    def fail_loader(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic cache validation failure")

    monkeypatch.setattr(_PREDICT, "load_feature_cache", fail_loader)
    arguments = argparse.Namespace(
        cache_dir=tmp_path / "cache",
        checkpoint=checkpoint,
        identity_split_manifest=tmp_path / "split.json",
        output=tmp_path / "output" / "prediction.npz",
        device="cpu",
        batch_size=1,
        workers=0,
        amp=False,
        verify_raw_sources=False,
        force=False,
    )
    with pytest.raises(RuntimeError, match="synthetic cache validation failure"):
        _PREDICT.run(arguments)
    assert cleaned


def _output_guard_fixture(tmp_path: Path) -> dict[str, Path]:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(
        json.dumps({"sessions": []}), encoding="utf-8"
    )
    checkpoint = tmp_path / "run" / "fold" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-original")
    run_config = checkpoint.parent.parent / "run_config.json"
    run_config.write_text("{}", encoding="utf-8")
    folds = tmp_path / "folds.json"
    folds.write_text("{}", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "fold_assignments": {"path": str(folds)},
                "cache": {"manifest_path": str(cache / "manifest.json")},
            }
        ),
        encoding="utf-8",
    )
    return {
        "cache": cache,
        "checkpoint": checkpoint,
        "run_config": run_config,
        "folds": folds,
        "split": split,
    }


def test_output_guard_rejects_same_path_hardlink_and_cache_tree(
    tmp_path: Path,
) -> None:
    paths = _output_guard_fixture(tmp_path)
    arguments = {
        "cache_dir": paths["cache"],
        "checkpoint_path": paths["checkpoint"],
        "run_config_path": paths["run_config"],
        "split_manifest_path": paths["split"],
    }
    with pytest.raises(RuntimeError, match="protected inference input"):
        _PREDICT._build_output_isolation_guard(
            output_path=paths["checkpoint"], **arguments
        )

    hardlink = tmp_path / "checkpoint-hardlink.npz"
    hardlink.hardlink_to(paths["checkpoint"])
    with pytest.raises(RuntimeError, match="inode aliases protected"):
        _PREDICT._build_output_isolation_guard(
            output_path=hardlink, **arguments
        )

    with pytest.raises(RuntimeError, match="protected inference input tree"):
        _PREDICT._build_output_isolation_guard(
            output_path=paths["cache"] / "prediction.npz", **arguments
        )
    cache_alias = tmp_path / "cache-alias"
    cache_alias.symlink_to(paths["cache"], target_is_directory=True)
    with pytest.raises(RuntimeError, match="protected inference input tree"):
        _PREDICT._build_output_isolation_guard(
            output_path=cache_alias / "prediction.npz", **arguments
        )


def test_force_cannot_overwrite_checkpoint_alias(tmp_path: Path) -> None:
    paths = _output_guard_fixture(tmp_path)
    original = paths["checkpoint"].read_bytes()
    arguments = argparse.Namespace(
        cache_dir=paths["cache"],
        checkpoint=paths["checkpoint"],
        identity_split_manifest=paths["split"],
        output=paths["checkpoint"],
        device="cpu",
        batch_size=1,
        workers=0,
        amp=False,
        verify_raw_sources=False,
        force=True,
    )
    with pytest.raises(RuntimeError, match="protected inference input"):
        _PREDICT.run(arguments)
    assert paths["checkpoint"].read_bytes() == original


@pytest.mark.parametrize("target_kind", ["session_manifest", "external_sync_config"])
def test_force_cannot_overwrite_acquisition_authority_graph(
    tmp_path: Path, target_kind: str
) -> None:
    paths = _output_guard_fixture(tmp_path)
    authority_root = tmp_path / "acquisition-authority"
    session_dir = authority_root / "sessions" / "S_A"
    session_dir.mkdir(parents=True)
    session_manifest = session_dir / "manifest.json"
    session_manifest.write_bytes(b"session-authority-original")
    external_sync = tmp_path / "sync-authority.yaml"
    external_sync.write_bytes(b"sync-authority-original")
    reconstruction = authority_root / "reconstruction.json"
    reconstruction.write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "session_id": "S_A",
                        "manifest": "sessions/S_A/manifest.json",
                    }
                ],
                "sync_config": str(external_sync),
            }
        ),
        encoding="utf-8",
    )
    (paths["cache"] / "manifest.json").write_text(
        json.dumps(
            {
                "sessions": [],
                "acquisition_contract": {
                    "reconstruction_manifest": str(reconstruction)
                },
            }
        ),
        encoding="utf-8",
    )
    target = session_manifest if target_kind == "session_manifest" else external_sync
    original = target.read_bytes()
    arguments = argparse.Namespace(
        cache_dir=paths["cache"],
        checkpoint=paths["checkpoint"],
        identity_split_manifest=paths["split"],
        output=target,
        device="cpu",
        batch_size=1,
        workers=0,
        amp=False,
        verify_raw_sources=False,
        force=True,
    )
    with pytest.raises(RuntimeError, match="protected inference input"):
        _PREDICT.run(arguments)
    assert target.read_bytes() == original


def _acquisition_authority_binding_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    cache = tmp_path / "cache-authority-binding"
    cache.mkdir()
    authority_root = tmp_path / "acquisition-authority-binding"
    session_root = authority_root / "sessions" / "S_A"
    session_root.mkdir(parents=True)
    receipt = authority_root / "receipts" / "S_A.json"
    approval = authority_root / "approvals" / "S_A.json"
    receipt.parent.mkdir()
    approval.parent.mkdir()
    receipt.write_text('{"receipt":true}\n', encoding="utf-8")
    approval.write_text('{"approval":true}\n', encoding="utf-8")
    range_artifact = session_root / "range.npz"
    range_artifact.write_bytes(b"range-artifact")
    session_manifest = session_root / "manifest.json"
    session_manifest.write_text(
        json.dumps(
            {
                "session_id": "S_A",
                "usable": True,
                "synchronization": {
                    "receipt": "receipts/S_A.json",
                    "manual_approval": "approvals/S_A.json",
                },
                "range_tracking": {
                    "status": "built",
                    "artifact": "range.npz",
                },
            }
        ),
        encoding="utf-8",
    )
    external: dict[str, Path] = {}
    for key in ("cohort_authority", "sync_config", "protocol_config", "spreadsheet"):
        path = tmp_path / f"{key}.authority"
        path.write_bytes(f"{key}-authority".encode("utf-8"))
        external[key] = path
    reconstruction = authority_root / "reconstruction.json"
    reconstruction.write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "session_id": "S_A",
                        "manifest": "sessions/S_A/manifest.json",
                    }
                ],
                **{key: str(path) for key, path in external.items()},
            }
        ),
        encoding="utf-8",
    )
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "sessions": [],
                "acquisition_contract": {
                    "reconstruction_manifest": str(reconstruction)
                },
            }
        ),
        encoding="utf-8",
    )
    return cache, {
        "reconstruction": reconstruction,
        "session_manifest": session_manifest,
        "receipt": receipt,
        "manual_approval": approval,
        "range_artifact": range_artifact,
        **external,
    }


@pytest.mark.parametrize(
    "target_kind",
    [
        "reconstruction",
        "session_manifest",
        "receipt",
        "manual_approval",
        "range_artifact",
        "cohort_authority",
        "sync_config",
        "protocol_config",
        "spreadsheet",
    ],
)
def test_acquisition_authority_binding_rejects_exact_byte_inode_replacement(
    tmp_path: Path,
    target_kind: str,
) -> None:
    cache, paths = _acquisition_authority_binding_fixture(tmp_path)
    bindings = _PREDICT._capture_acquisition_authority_bindings(cache)
    assert set(bindings) == {str(path.resolve()) for path in paths.values()}
    target = paths[target_kind]
    original = target.read_bytes()
    replacement = target.with_name(f".{target.name}.replacement")
    replacement.write_bytes(original)
    os.replace(replacement, target)
    assert _PREDICT._sha256_file(target) == hashlib.sha256(original).hexdigest()

    with pytest.raises(RuntimeError, match="authority graph changed"):
        _PREDICT._assert_acquisition_authority_bindings_current(cache, bindings)


def test_acquisition_authority_binding_detects_in_place_mutate_then_restore(
    tmp_path: Path,
) -> None:
    cache, paths = _acquisition_authority_binding_fixture(tmp_path)
    bindings = _PREDICT._capture_acquisition_authority_bindings(cache)
    receipt = paths["receipt"]
    original = receipt.read_bytes()
    with receipt.open("r+b") as stream:
        stream.seek(0)
        stream.write(b"X" * len(original))
        stream.flush()
        os.fsync(stream.fileno())
        stream.seek(0)
        stream.write(original)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    assert receipt.read_bytes() == original

    with pytest.raises(RuntimeError, match="authority graph changed"):
        _PREDICT._assert_acquisition_authority_bindings_current(cache, bindings)


def test_full_authority_replay_binds_unusable_session_even_without_raw_verify_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache-full-authority"
    cache.mkdir()
    authority_root = tmp_path / "full-authority"
    session_root = authority_root / "sessions" / "S_UNUSABLE"
    session_root.mkdir(parents=True)
    unusable_manifest = session_root / "manifest.json"
    unusable_manifest.write_text(
        json.dumps({"session_id": "S_UNUSABLE", "usable": False}),
        encoding="utf-8",
    )
    reconstruction = authority_root / "reconstruction.json"
    reconstruction_hash = "a" * 64
    reconstruction.write_text(
        json.dumps(
            {
                "content_sha256": reconstruction_hash,
                "sessions": [
                    {
                        "session_id": "S_UNUSABLE",
                        "manifest": "sessions/S_UNUSABLE/manifest.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cache_manifest = {
        "sessions": [],
        "acquisition_contract": {
            "reconstruction_manifest": str(reconstruction),
            "reconstruction_content_sha256": reconstruction_hash,
        },
    }
    (cache / "manifest.json").write_text(
        json.dumps(cache_manifest), encoding="utf-8"
    )
    bindings = _PREDICT._capture_acquisition_authority_bindings(
        cache, cache_manifest=cache_manifest
    )
    replayed: list[Path] = []

    def replay(path: Path) -> object:
        replayed.append(Path(path))
        return argparse.Namespace(
            manifest_path=reconstruction,
            manifest={"content_sha256": reconstruction_hash},
        )

    monkeypatch.setattr(_PREDICT, "load_acquisition_reconstruction", replay)
    # There is intentionally no verify_raw_sources argument: canonical replay
    # remains mandatory even for the command's --no-verify-raw-sources mode.
    _PREDICT._revalidate_full_acquisition_authority(
        cache,
        cache_manifest=cache_manifest,
        expected_bindings=bindings,
    )
    assert replayed == [reconstruction.resolve()]

    original = unusable_manifest.read_bytes()
    replacement = unusable_manifest.with_name(".manifest.replacement")
    replacement.write_bytes(original)
    os.replace(replacement, unusable_manifest)
    with pytest.raises(RuntimeError, match="authority graph changed"):
        _PREDICT._revalidate_full_acquisition_authority(
            cache,
            cache_manifest=cache_manifest,
            expected_bindings=bindings,
        )


def test_atomic_npz_no_clobber_preserves_concurrent_sentinel(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prediction.npz"
    sentinel = b"concurrent-sentinel"

    def create_concurrent_output() -> None:
        output.write_bytes(sentinel)

    with pytest.raises(FileExistsError):
        _PREDICT._atomic_npz(
            output,
            {"value": np.asarray([1.0], dtype=np.float32)},
            before_replace=create_concurrent_output,
            replace_existing=False,
        )
    assert output.read_bytes() == sentinel
    assert not list(tmp_path.glob(".prediction.npz.*.tmp"))


def test_atomic_npz_random_exclusive_temp_ignores_old_symlink_preemption(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prediction.npz"
    protected = tmp_path / "protected.bin"
    protected.write_bytes(b"do-not-touch")
    predictable = tmp_path / f".{output.name}.{os.getpid()}.tmp"
    predictable.symlink_to(protected)

    digest = _PREDICT._atomic_npz(
        output,
        {"value": np.asarray([2.0], dtype=np.float32)},
        replace_existing=False,
    )
    assert protected.read_bytes() == b"do-not-touch"
    assert predictable.is_symlink()
    assert digest == _PREDICT._sha256_file(output)


def test_atomic_npz_uses_no_named_serialization_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "prediction.npz"
    monkeypatch.setattr(
        _PREDICT.tempfile,
        "mkstemp",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("named temporary must not be created")
        ),
    )
    digest = _PREDICT._atomic_npz(
        output,
        {"value": np.asarray([3.0], dtype=np.float32)},
        replace_existing=False,
    )
    assert digest == _PREDICT._sha256_file(output)
    assert not list(tmp_path.glob(".prediction.npz.*"))


def test_atomic_npz_no_clobber_preserves_swap_after_fd_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "prediction.npz"
    sentinel = b"concurrent-sentinel-after-link"
    real_link = _PREDICT._link_open_fd_noreplace

    def link_then_swap(
        descriptor: int, parent_descriptor: int, destination_name: str
    ) -> None:
        real_link(descriptor, parent_descriptor, destination_name)
        output.unlink()
        output.write_bytes(sentinel)

    monkeypatch.setattr(_PREDICT, "_link_open_fd_noreplace", link_then_swap)
    with pytest.raises(RuntimeError, match="foreign entry preserved"):
        _PREDICT._atomic_npz(
            output,
            {"value": np.asarray([3.5], dtype=np.float32)},
            replace_existing=False,
        )
    assert output.read_bytes() == sentinel


def test_atomic_npz_force_preserves_old_output_when_stage_swapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "prediction.npz"
    old_output = b"old-output"
    output.write_bytes(old_output)
    protected = tmp_path / "protected.bin"
    protected.write_bytes(b"do-not-touch")
    real_exchange = _PREDICT._exchange_stage_with_output

    def swap_stage_then_exchange(
        stage_descriptor: int,
        stage_name: str,
        parent_descriptor: int,
        output_name: str,
        expected_new: os.stat_result,
    ) -> None:
        os.unlink(stage_name, dir_fd=stage_descriptor)
        os.symlink(
            protected,
            stage_name,
            dir_fd=stage_descriptor,
        )
        real_exchange(
            stage_descriptor,
            stage_name,
            parent_descriptor,
            output_name,
            expected_new,
        )

    monkeypatch.setattr(
        _PREDICT, "_exchange_stage_with_output", swap_stage_then_exchange
    )
    with pytest.raises(RuntimeError, match="stage changed before exchange"):
        _PREDICT._atomic_npz(
            output,
            {"value": np.asarray([3.75], dtype=np.float32)},
            replace_existing=True,
        )
    assert output.read_bytes() == old_output
    assert protected.read_bytes() == b"do-not-touch"


def test_atomic_npz_force_handles_output_removed_after_private_stage_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "prediction.npz"
    output.write_bytes(b"old-output")
    real_link = _PREDICT._link_open_fd_noreplace

    def link_then_remove_old_output(
        descriptor: int,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        real_link(descriptor, destination_descriptor, destination_name)
        if destination_name == "payload.npz":
            output.unlink()

    monkeypatch.setattr(
        _PREDICT,
        "_link_open_fd_noreplace",
        link_then_remove_old_output,
    )
    digest = _PREDICT._atomic_npz(
        output,
        {"value": np.asarray([3.875], dtype=np.float32)},
        replace_existing=True,
    )
    assert digest == _PREDICT._sha256_file(output)
    assert not list(tmp_path.glob(".prediction.npz.*.stage"))


def test_atomic_npz_parent_rebinding_fails_before_publication_and_cleans_stage(
    tmp_path: Path,
) -> None:
    output_parent = tmp_path / "output-parent"
    output_parent.mkdir()
    output = output_parent / "prediction.npz"
    moved_root = tmp_path / "protected-tree"
    moved_root.mkdir()
    moved_parent = moved_root / "captured-output-parent"

    def rebind_parent() -> None:
        output_parent.rename(moved_parent)
        output_parent.mkdir()

    with pytest.raises(RuntimeError, match="output parent changed"):
        _PREDICT._atomic_npz(
            output,
            {"value": np.asarray([3.9375], dtype=np.float32)},
            before_replace=rebind_parent,
            replace_existing=False,
        )
    assert not output.exists()
    assert not (moved_parent / output.name).exists()
    assert not list(moved_parent.glob(".prediction.npz.*.stage"))


def test_atomic_npz_force_replaces_ordinary_output_and_returns_published_hash(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prediction.npz"
    output.write_bytes(b"old-sentinel")

    digest = _PREDICT._atomic_npz(
        output,
        {"value": np.asarray([4.0], dtype=np.float32)},
        replace_existing=True,
    )

    assert digest == _PREDICT._sha256_file(output)
    with np.load(output, allow_pickle=False) as payload:
        np.testing.assert_array_equal(
            payload["value"], np.asarray([4.0], dtype=np.float32)
        )


def test_atomic_npz_does_not_use_path_unlink_for_no_clobber_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "prediction.npz"
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Path.unlink must not be used")
        ),
    )
    digest = _PREDICT._atomic_npz(
        output,
        {"value": np.asarray([5.0], dtype=np.float32)},
        replace_existing=False,
    )
    assert digest == _PREDICT._sha256_file(output)


def test_atomic_npz_revalidates_after_serialization_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    run_config = tmp_path / "run_config.json"
    output = tmp_path / "prediction.npz"
    checkpoint.write_bytes(b"before")
    run_config.write_text("{}", encoding="utf-8")
    checkpoint_hash = _PREDICT._sha256_file(checkpoint)
    run_config_hash = _PREDICT._sha256_file(run_config)
    source_hashes = _PREDICT._runtime_source_hashes()
    real_savez_compressed = _PREDICT.np.savez_compressed

    def mutate_during_serialization(stream: object, **arrays: object) -> None:
        real_savez_compressed(stream, **arrays)
        checkpoint.write_bytes(b"changed-during-serialization")

    monkeypatch.setattr(
        _PREDICT.np, "savez_compressed", mutate_during_serialization
    )

    def publication_barrier() -> None:
        _PREDICT._assert_execution_inputs_current(
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_hash,
            run_config_path=run_config,
            run_config_sha256=run_config_hash,
            source_hashes=source_hashes,
        )

    with pytest.raises(RuntimeError, match="checkpoint changed"):
        _PREDICT._atomic_npz(
            output,
            {"value": np.asarray([1.0], dtype=np.float32)},
            before_replace=publication_barrier,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".prediction.npz.*.tmp"))


def test_output_masks_invalid_reference_and_contains_nested_provenance(
    tmp_path: Path
) -> None:
    _, _, _, authority, _ = _write_authority(tmp_path)
    cache = _cache()
    checkpoint_path = tmp_path / "snn_best.pt"
    checkpoint = _checkpoint(checkpoint_path, authority, cache)
    run_config_path = tmp_path / "run_config.json"
    run_config_path.write_text(json.dumps(_run_config(authority)), encoding="utf-8")
    bundle = _PREDICT.predict_label_free(
        _Model(), DataLoader(_Dataset((12.0, 999.0)), batch_size=2), torch.device("cpu"), amp=False
    )
    arrays = _PREDICT.build_output_arrays(
        bundle,
        cache.metadata,
        np.asarray([0, 1]),
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        authority=authority,
        run_config_path=run_config_path,
        cache_source={
            "canonical_cache_provenance": _provenance_document(cache),
            "cache_content_signature_sha256": "6" * 64,
            "raw_source_fingerprints_verified": True,
            "raw_input_sha256_bindings_verified": False,
        },
        acquisition_authority_bindings={},
        source_hashes=_PREDICT._runtime_source_hashes(),
        direct_entry_disk_binding=dict(_PREDICT._DIRECT_ENTRY_DISK_BINDING),
        checkpoint_sha256=_PREDICT._sha256_file(checkpoint_path),
        run_config_sha256=_PREDICT._sha256_file(run_config_path),
    )
    assert arrays["reference_valid"].tolist() == [True, False]
    assert arrays["reference_rr_bpm"][0] == 12.0
    assert np.isnan(arrays["reference_rr_bpm"][1])
    assert arrays["identity"].tolist() == ["A", "A"]
    provenance = json.loads(str(arrays["provenance_json"]))
    assert provenance["labels_forwarded_to_model"] is False
    assert provenance["strict_nested_role"] == "prediction"
    assert provenance["commercial_performance_claim_eligible"] is False
    assert provenance["canonical_cache_provenance"] == _provenance_document(cache)
    assert provenance["cache_content_signature_sha256"] == "6" * 64
    assert provenance["acquisition_authority_file_bindings"] == {}
    assert provenance["acquisition_authority_file_count"] == 0
    assert provenance["acquisition_authority_bindings_sha256"] == (
        _PREDICT._canonical_hash({"files": {}})
    )
    assert len(provenance["source_hashes"]) >= 6
    execution = provenance["execution_source_generation"]
    assert execution["guard_scope"] == "initialization_time_direct_entry_disk_only"
    assert (
        execution["direct_entry_disk_binding"]
        == _PREDICT._DIRECT_ENTRY_DISK_BINDING
    )
    assert execution["binds_actual_loader_compiled_bytes"] is False
    assert execution["complete_private_import_closure"] is False


def test_split_authority_publication_barrier_detects_replacement(
    tmp_path: Path,
) -> None:
    _, _, fold_path, authority, _ = _write_authority(tmp_path)
    _PREDICT._assert_split_authority_current(authority)
    fold_path.write_text(
        json.dumps({"identity_to_fold": {"A": 1, "B": 0, "C": 2, "D": 3}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="fold assignments changed"):
        _PREDICT._assert_split_authority_current(authority)
