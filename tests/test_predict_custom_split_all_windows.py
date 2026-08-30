from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
        manifest_path, metadata=_metadata(), cache_dir=cache_dir
    )
    return cache_dir, manifest_path, fold_path, authority


def _cache() -> object:
    return _PREDICT.FeatureCache(
        maps=np.zeros((5, 3, 2, 4), dtype=np.float16),
        aux=np.asarray([[1.0, 2.0], [2.0, 3.0], [5.0, 7.0], [8.0, 9.0], [3.0, 4.0]], dtype=np.float32),
        metadata=_metadata(),
        frequencies_hz=np.asarray([0.1, 0.2], dtype=np.float32),
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
    }
    torch.save(checkpoint, path)
    return checkpoint


def _run_config(authority: object) -> dict[str, object]:
    return {
        "run_signature": "run",
        "arguments": {
            "identity_split_manifest_sha256": authority.content_sha256,
            "include_invalid": False,
            "use_aux": True,
            "input_branches": 2,
            "rr_range": [5.0, 45.0],
            "rr_bin_width": 0.25,
        },
    }


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


def test_manifest_and_referenced_hash_tampering_fail_closed(tmp_path: Path) -> None:
    cache_dir, manifest_path, fold_path, _ = _write_authority(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["fold_id"] = 8
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical content SHA-256 mismatch"):
        _PREDICT.load_identity_split_authority(
            manifest_path, metadata=_metadata(), cache_dir=cache_dir
        )

    # Restore a valid manifest, then alter the separately content-bound fold file.
    _, manifest_path, fold_path, _ = _write_authority(tmp_path / "second")
    fold_path.write_text(json.dumps({"identity_to_fold": {"A": 99}}), encoding="utf-8")
    with pytest.raises(ValueError, match="fold_assignments SHA-256 mismatch"):
        _PREDICT.load_identity_split_authority(
            manifest_path,
            metadata=_metadata(),
            cache_dir=manifest_path.parent / "cache",
        )


def test_checkpoint_split_fold_hash_and_scaler_tampering_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, authority = _write_authority(tmp_path)
    cache = _cache()
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


def test_output_masks_invalid_reference_and_contains_nested_provenance(
    tmp_path: Path
) -> None:
    _, _, _, authority = _write_authority(tmp_path)
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
    )
    assert arrays["reference_valid"].tolist() == [True, False]
    assert arrays["reference_rr_bpm"][0] == 12.0
    assert np.isnan(arrays["reference_rr_bpm"][1])
    assert arrays["identity"].tolist() == ["A", "A"]
    provenance = json.loads(str(arrays["provenance_json"]))
    assert provenance["labels_forwarded_to_model"] is False
    assert provenance["strict_nested_role"] == "prediction"
    assert provenance["commercial_performance_claim_eligible"] is False
    assert len(provenance["source_hashes"]) >= 6
