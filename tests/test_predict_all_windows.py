from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from snn_rr.cache import CacheProvenance, FeatureCache

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "predict_all_windows.py"
_SPEC = importlib.util.spec_from_file_location("snn_rr_predict_all_windows", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_PREDICT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PREDICT
_SPEC.loader.exec_module(_PREDICT)

FoldBinding = _PREDICT.FoldBinding
FrozenTolerance = _PREDICT.FrozenTolerance
PredictionBundle = _PREDICT.PredictionBundle
_load_reusable_fold = _PREDICT._load_reusable_fold
_save_fold_result = _PREDICT._save_fold_result
_commit_verified_fold = _PREDICT._commit_verified_fold
_load_frozen = _PREDICT._load_frozen
_cache_session_content_binding = _PREDICT._cache_session_content_binding
_runtime_source_hashes = _PREDICT._runtime_source_hashes
_atomic_npz = _PREDICT._atomic_npz
_fold_marker_path = _PREDICT._fold_marker_path
_verify_complete_reuse = _PREDICT._verify_complete_reuse
_deployment_freeze_assessment = _PREDICT._deployment_freeze_assessment
_sha256_file = _PREDICT._sha256_file
FOLD_DEPLOYMENT_FIELDS = _PREDICT.FOLD_DEPLOYMENT_FIELDS
build_csv = _PREDICT.build_csv
build_final_arrays = _PREDICT.build_final_arrays
parse_args = _PREDICT.parse_args
predict_label_free = _PREDICT.predict_label_free
validate_identity_partition = _PREDICT.validate_identity_partition
validate_label_free_forward_interface = _PREDICT.validate_label_free_forward_interface
verify_frozen_valid_predictions = _PREDICT.verify_frozen_valid_predictions


class _TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rr: tuple[float, ...] = (12.0, 99.0)) -> None:
        self.rr = rr

    def __len__(self) -> int:
        return len(self.rr)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "map": torch.full((3, 2, 4), float(index + 1), dtype=torch.float16),
            "aux": torch.tensor([float(index), 1.0], dtype=torch.float32),
            "radar_mask": torch.ones(3, dtype=torch.bool),
            # Deliberately absurd target/QC values.  The audited forward call
            # has no route by which these can reach the model.
            "rr": torch.tensor(self.rr[index], dtype=torch.float32),
            "reference_valid": torch.tensor(index == 0),
            "observable": torch.tensor(float(index == 0)),
            "reference_quality": torch.tensor(-123.0),
            "reference_sigma": torch.tensor(999.0),
            "classical_rr": torch.tensor(-88.0),
            "classical_confidence": torch.tensor(-77.0),
            "index": torch.tensor(index, dtype=torch.int64),
        }


class _TinyLabelFreeModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        radar_mask: torch.Tensor | None = None,
        aux: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert radar_mask is not None and aux is not None
        batch = x.shape[0]
        logits = torch.stack(
            [x.float().mean(dim=(1, 2, 3)), aux[:, 0], aux[:, 1]], dim=1
        )
        probabilities = logits.softmax(dim=1)
        bins = x.new_tensor([10.0, 11.0, 12.0], dtype=torch.float32)
        expected = (probabilities * bins).sum(dim=1)
        top_probability, top_index = probabilities.topk(3, dim=1)
        return {
            "expected_rr": expected,
            "rr_std": torch.full((batch,), 0.5, device=x.device),
            "quality": torch.full((batch,), 0.8, device=x.device),
            "radar_weights": radar_mask.float() / radar_mask.float().sum(1, keepdim=True),
            "map_rr": bins[probabilities.argmax(dim=1)],
            "posterior_entropy": -(probabilities * probabilities.log()).sum(dim=1),
            "topk_rr": bins[top_index],
            "topk_probability": top_probability,
            "probabilities": probabilities,
            "alias_probability": torch.full((batch,), 0.2, device=x.device),
            "spike_rate_per_sample": torch.full((batch,), 0.1, device=x.device),
        }


class _TargetReadingModel(nn.Module):
    def forward(self, x: torch.Tensor, radar_mask: torch.Tensor, target: torch.Tensor):
        del x, radar_mask, target
        return {}


class _VariadicModel(nn.Module):
    def forward(self, x: torch.Tensor, *args: torch.Tensor, **kwargs: torch.Tensor):
        del x, args, kwargs
        return {}


def _bundle(index: np.ndarray | None = None) -> PredictionBundle:
    index = np.arange(2, dtype=np.int64) if index is None else np.asarray(index)
    count = len(index)
    prediction = np.linspace(10.0, 11.0, count, dtype=np.float32)
    probability = np.tile(
        np.asarray([[0.1, 0.2, 0.7]], dtype=np.float16), (count, 1)
    )
    return PredictionBundle(
        index=index,
        target=prediction.copy(),
        prediction=prediction,
        rr_std=np.full(count, 0.5, dtype=np.float32),
        uncertainty=np.full(count, 0.625, dtype=np.float32),
        quality=np.full(count, 0.8, dtype=np.float32),
        observable=np.ones(count, dtype=np.float32),
        reference_valid=np.asarray([True] + [False] * (count - 1)),
        spike_rate=np.full(count, 0.1, dtype=np.float32),
        radar_weights=np.full((count, 3), 1 / 3, dtype=np.float32),
        map_prediction=prediction.copy(),
        posterior_entropy=np.full(count, 0.9, dtype=np.float32),
        topk_rr=np.tile(np.asarray([[12.0, 11.0, 10.0]], dtype=np.float32), (count, 1)),
        topk_probability=np.tile(
            np.asarray([[0.7, 0.2, 0.1]], dtype=np.float32), (count, 1)
        ),
        posterior_probability=probability,
        alias_probability=np.full(count, 0.2, dtype=np.float32),
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["S01_A", "S02_B"],
            "identity": ["A", "B"],
            "protocol": ["rest", "walk"],
            "window_number": [0, 0],
            "window_start_s": [1.0, 2.0],
            "window_end_s": [33.0, 34.0],
            "reference_valid": [True, False],
            "rr_bpm": [10.0, 999.0],
        }
    )


def test_cli_defaults_are_locked_to_full_cache() -> None:
    args = parse_args(["--device", "cpu", "--no-amp"])
    assert args.expected_rows == 9576
    assert args.cache_dir.name == "rf32s"
    assert args.output_dir.name == "all_windows"
    assert not args.amp
    assert args.verify_raw_sources


def test_prepare_cache_preserves_timing_mask_and_provenance_through_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = pd.DataFrame(
        {
            "session_id": ["S01_A", "S01_A"],
            "window_number": [0, 1],
            "classical_rr_bpm": [12.0, 13.0],
            "classical_confidence": [0.8, 0.9],
            "radar_peak_spread_bpm": [0.4, 0.3],
        }
    )
    timing = np.ones((2, 3, 5), dtype=np.bool_)
    timing[0, 1, 0] = False
    provenance = CacheProvenance(
        classification="acquisition_diagnostic",
        root_manifest_path="manifest.json",
        root_manifest_sha256="1" * 64,
        root_manifest_content_sha256="2" * 64,
        acquisition_schema_version="snn_rr.feature_cache_acquisition.v2",
        acquisition_mode="strict",
        scientific_eligible=False,
        config_sha256="3" * 64,
        pipeline_sha256="4" * 64,
        reconstruction_content_sha256="5" * 64,
        inventory_sha256="6" * 64,
        inventory_file_count=10,
        selected_sessions=("S01_A",),
    )
    cache = FeatureCache(
        maps=np.ones((2, 3, 2, 4), dtype=np.float16),
        aux=np.ones((2, 37), dtype=np.float32),
        metadata=metadata,
        frequencies_hz=np.asarray([0.1, 0.2], dtype=np.float32),
        provenance=provenance,
        radar_timing_valid_mask=timing,
    )
    expected_aux, expected_names = (
        _PREDICT.append_mask_aware_causal_history_features(cache)
    )
    monkeypatch.setattr(_PREDICT, "load_feature_cache", lambda *args, **kwargs: cache)
    run_config = {
        "arguments": {"use_aux": True, "causal_history": True},
        "cache_shape": {
            "maps": list(cache.maps.shape),
            "aux": list(expected_aux.shape),
        },
        "causal_history_feature_names": expected_names,
    }

    prepared, base_aux_dim, names = _PREDICT.prepare_cache(
        Path("unused"), run_config
    )

    assert base_aux_dim == 37
    assert names == expected_names
    assert prepared.provenance == provenance
    np.testing.assert_array_equal(prepared.radar_timing_valid_mask, timing)
    np.testing.assert_array_equal(prepared.aux, expected_aux)


def test_deployment_freeze_fails_closed_without_raw_source_verification() -> None:
    ready = _deployment_freeze_assessment(
        strict_frozen_parity=True,
        raw_source_fingerprints_verified=True,
    )
    assert ready["eligible"]
    assert ready["blockers"] == []

    no_raw = _deployment_freeze_assessment(
        strict_frozen_parity=True,
        raw_source_fingerprints_verified=False,
    )
    assert not no_raw["eligible"]
    assert no_raw["blockers"] == [
        "raw_source_fingerprint_verification_required"
    ]

    no_parity = _deployment_freeze_assessment(
        strict_frozen_parity=False,
        raw_source_fingerprints_verified=True,
    )
    assert not no_parity["eligible"]
    assert no_parity["blockers"] == [
        "strict_cuda_amp_frozen_parity_required"
    ]


def test_identity_partition_is_an_exact_row_cover() -> None:
    metadata = _metadata()
    bindings = [
        FoldBinding(0, Path("a.pt"), "a" * 64, ("A",), np.asarray([0])),
        FoldBinding(1, Path("b.pt"), "b" * 64, ("B",), np.asarray([1])),
    ]
    assert np.array_equal(
        validate_identity_partition(bindings, metadata), np.asarray([0, 1])
    )

    duplicate = [bindings[0], FoldBinding(1, Path("b.pt"), "b" * 64, ("A", "B"), np.asarray([0, 1]))]
    with pytest.raises(RuntimeError, match="belongs to folds"):
        validate_identity_partition(duplicate, metadata)


def test_label_and_qc_cannot_enter_model_forward() -> None:
    validate_label_free_forward_interface(_TinyLabelFreeModel())
    with pytest.raises(RuntimeError, match="forbidden label/QC"):
        validate_label_free_forward_interface(_TargetReadingModel())
    with pytest.raises(RuntimeError, match="variadic"):
        validate_label_free_forward_interface(_VariadicModel())


def test_cpu_tiny_prediction_is_independent_of_target_and_qc() -> None:
    model = _TinyLabelFreeModel()
    first = predict_label_free(
        model,
        DataLoader(_TinyDataset((12.0, 99.0)), batch_size=2),
        torch.device("cpu"),
        amp=False,
    )
    second = predict_label_free(
        model,
        DataLoader(_TinyDataset((-1000.0, 1000.0)), batch_size=2),
        torch.device("cpu"),
        amp=False,
    )
    assert np.array_equal(first.prediction, second.prediction)
    assert np.array_equal(first.posterior_probability, second.posterior_probability)
    assert not np.array_equal(first.target, second.target)


def test_frozen_valid_verification_checks_index_and_numerics() -> None:
    bundle = _bundle()
    frozen = {
        "index": np.asarray([0]),
        "fold": np.asarray([3], dtype=np.int16),
        "prediction": bundle.prediction[:1],
        "map_prediction": bundle.map_prediction[:1],
        "rr_std": bundle.rr_std[:1],
        "uncertainty": bundle.uncertainty[:1],
        "quality": bundle.quality[:1],
        "alias_probability": bundle.alias_probability[:1],
        "posterior_entropy": bundle.posterior_entropy[:1],
        "topk_rr": bundle.topk_rr[:1],
        "topk_probability": bundle.topk_probability[:1],
        "posterior_probability": bundle.posterior_probability[:1],
        "spike_rate": bundle.spike_rate[:1],
        "radar_weights": bundle.radar_weights[:1],
    }
    tolerance = FrozenTolerance(
        *(0.0 for _ in range(len(FrozenTolerance.__dataclass_fields__)))
    )
    audit = verify_frozen_valid_predictions(bundle, 3, frozen, tolerance)
    assert audit["rows"] == 1
    assert audit["fields"]["prediction"]["max_abs_difference"] == 0.0

    frozen["prediction"] = frozen["prediction"] + 1.0
    with pytest.raises(RuntimeError, match="field=prediction"):
        verify_frozen_valid_predictions(bundle, 3, frozen, tolerance)


def test_final_serialization_masks_invalid_reference_and_contains_provenance() -> None:
    bundle = _bundle()
    metadata = _metadata()
    provenance = {"source": "unit-test", "identity_disjoint": True}
    arrays = build_final_arrays(
        bundle,
        np.asarray([0, 1], dtype=np.int16),
        metadata,
        np.asarray([10.0, 11.0, 12.0], dtype=np.float32),
        run_signature="run123",
        inference_signature="infer123",
        provenance=provenance,
    )
    assert arrays["reference_rr_bpm"][0] == 10.0
    assert np.isnan(arrays["reference_rr_bpm"][1])
    assert json.loads(str(arrays["provenance_json"])) == provenance

    bindings = {
        0: FoldBinding(0, Path("a.pt"), "a" * 64, ("A",), np.asarray([0])),
        1: FoldBinding(1, Path("b.pt"), "b" * 64, ("B",), np.asarray([1])),
    }
    cache_source = {
        "cache_manifest_sha256": "c" * 64,
        "source_fingerprint_by_session": {"S01_A": "d" * 64, "S02_B": "e" * 64},
    }
    frame = build_csv(arrays, binding_for_fold=bindings, cache_source=cache_source)
    required = {
        "cache_index",
        "fold",
        "posterior_probability_json",
        "posterior_top1_rr_bpm",
        "radar_1_weight",
        "run_signature",
        "inference_signature",
        "checkpoint_sha256",
        "source_fingerprint",
    }
    assert required <= set(frame.columns)
    assert len(json.loads(frame.loc[0, "posterior_probability_json"])) == 3


def test_atomic_fold_file_can_be_verified_and_resumed(tmp_path: Path) -> None:
    bundle = _bundle()
    binding = FoldBinding(
        fold=2,
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint_sha256="f" * 64,
        test_identities=("A", "B"),
        expected_indices=np.asarray([0, 1]),
    )
    path = tmp_path / "fold_2_all_windows.npz"
    artifact = _save_fold_result(
        path,
        bundle,
        binding=binding,
        run_signature="run",
        inference_signature="inference",
    )
    assert not _fold_marker_path(path).exists()
    _commit_verified_fold(
        path,
        artifact=artifact,
        binding=binding,
        run_signature="run",
        inference_signature="inference",
        frozen_oof_sha256="0" * 64,
        parity={"rows": 1, "fields": {}},
        reference_valid_count=1,
    )
    assert _fold_marker_path(path).is_file()
    resumed = _load_reusable_fold(
        path,
        binding=binding,
        run_signature="run",
        inference_signature="inference",
        frozen_oof_sha256="0" * 64,
    )
    assert np.array_equal(resumed.prediction, bundle.prediction)
    assert np.isnan(resumed.target).all()
    assert np.isnan(resumed.observable).all()
    with np.load(path, allow_pickle=False) as data:
        assert "target" not in data.files
        assert "observable" not in data.files
        assert set(FOLD_DEPLOYMENT_FIELDS) <= set(data.files)
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_rejects_invalid_row_tamper_even_when_valid_row_is_unchanged(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    binding = FoldBinding(
        fold=2,
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint_sha256="f" * 64,
        test_identities=("A", "B"),
        expected_indices=np.asarray([0, 1]),
    )
    path = tmp_path / "fold_2_all_windows.npz"
    artifact = _save_fold_result(
        path,
        bundle,
        binding=binding,
        run_signature="run",
        inference_signature="inference",
    )
    _commit_verified_fold(
        path,
        artifact=artifact,
        binding=binding,
        run_signature="run",
        inference_signature="inference",
        frozen_oof_sha256="0" * 64,
        parity={"rows": 1, "fields": {}},
        reference_valid_count=1,
    )
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays["prediction"] = arrays["prediction"].copy()
    arrays["prediction"][1] += 50.0  # row 1 is reference-invalid
    _atomic_npz(path, arrays)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _load_reusable_fold(
            path,
            binding=binding,
            run_signature="run",
            inference_signature="inference",
            frozen_oof_sha256="0" * 64,
        )


def test_frozen_target_requires_exact_float32_cache_binding(tmp_path: Path) -> None:
    path = tmp_path / "frozen.npz"
    _atomic_npz(
        path,
        {
            "run_signature": np.asarray("run"),
            "index": np.asarray([0], dtype=np.int64),
            "reference_valid": np.asarray([True]),
            "target": np.asarray([10.0], dtype=np.float32),
        },
    )
    loaded = _load_frozen(path, "run", _metadata())
    assert loaded["target"][0] == np.float32(10.0)

    changed = np.nextafter(np.float32(10.0), np.float32(11.0))
    _atomic_npz(
        path,
        {
            "run_signature": np.asarray("run"),
            "index": np.asarray([0], dtype=np.int64),
            "reference_valid": np.asarray([True]),
            "target": np.asarray([changed], dtype=np.float32),
        },
    )
    with pytest.raises(RuntimeError, match="exactly float32-bound"):
        _load_frozen(path, "run", _metadata())
    with pytest.raises(FileNotFoundError, match="frozen OOF is missing"):
        _load_frozen(tmp_path / "missing.npz", "run", _metadata())


def test_cache_content_binding_changes_when_aux_bytes_change(tmp_path: Path) -> None:
    session = tmp_path / "SESSION"
    session.mkdir()
    for name, payload in {
        "maps.npy": b"maps",
        "aux.npy": b"aux-a",
        "metadata.csv": b"cache_index\n0\n",
        "frequencies_hz.npy": b"frequencies",
        "manifest.json": b"{}",
    }.items():
        (session / name).write_bytes(payload)
    first = _cache_session_content_binding(tmp_path, "SESSION")
    (session / "aux.npy").write_bytes(b"aux-b")
    second = _cache_session_content_binding(tmp_path, "SESSION")
    assert first["files"]["maps.npy"] == second["files"]["maps.npy"]
    assert first["files"]["aux.npy"] != second["files"]["aux.npy"]
    assert first["content_signature_sha256"] != second["content_signature_sha256"]


def test_runtime_signature_binds_train_cache_and_model_sources() -> None:
    hashes = _runtime_source_hashes()
    assert {
        "scripts/predict_all_windows.py",
        "scripts/train.py",
        "src/snn_rr/cache.py",
        "src/snn_rr/models.py",
    } <= set(hashes)
    assert all(len(value) == 64 for value in hashes.values())


def test_complete_reuse_rechecks_frozen_parity_not_only_output_hashes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "all_windows"
    output.mkdir()
    bundle = _bundle()
    metadata = _metadata()
    arrays = build_final_arrays(
        bundle,
        np.asarray([0, 1], dtype=np.int16),
        metadata,
        np.asarray([10.0, 11.0, 12.0], dtype=np.float32),
        run_signature="run",
        inference_signature="inference",
        provenance={},
    )
    npz_path = output / "snn_all_windows.npz"
    csv_path = output / "snn_all_windows.csv"
    _atomic_npz(npz_path, arrays)
    csv_path.write_text("cache_index\n0\n1\n", encoding="utf-8")
    frozen = {
        "index": np.asarray([0], dtype=np.int64),
        "fold": np.asarray([0], dtype=np.int16),
        "target": np.asarray([10.0], dtype=np.float32),
        "prediction": bundle.prediction[:1],
        "map_prediction": bundle.map_prediction[:1],
        "rr_std": bundle.rr_std[:1],
        "uncertainty": bundle.uncertainty[:1],
        "quality": bundle.quality[:1],
        "alias_probability": bundle.alias_probability[:1],
        "posterior_entropy": bundle.posterior_entropy[:1],
        "topk_rr": bundle.topk_rr[:1],
        "topk_probability": bundle.topk_probability[:1],
        "posterior_probability": bundle.posterior_probability[:1],
        "spike_rate": bundle.spike_rate[:1],
        "radar_weights": bundle.radar_weights[:1],
    }
    frozen_sha = "9" * 64
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "inference_signature": "inference",
                "frozen_valid_oof_verification": {"source_sha256": frozen_sha},
                "outputs": {
                    "npz_sha256": _sha256_file(npz_path),
                    "csv_sha256": _sha256_file(csv_path),
                },
            }
        ),
        encoding="utf-8",
    )
    zero = FrozenTolerance(
        *(0.0 for _ in range(len(FrozenTolerance.__dataclass_fields__)))
    )
    assert _verify_complete_reuse(
        output,
        inference_signature="inference",
        expected_rows=2,
        run_signature="run",
        frozen=frozen,
        frozen_oof_sha256=frozen_sha,
        tolerance=zero,
        bindings=[],
    )
    changed_frozen = dict(frozen)
    changed_frozen["prediction"] = frozen["prediction"] + 0.25
    with pytest.raises(RuntimeError, match="frozen parity mismatch"):
        _verify_complete_reuse(
            output,
            inference_signature="inference",
            expected_rows=2,
            run_signature="run",
            frozen=changed_frozen,
            frozen_oof_sha256=frozen_sha,
            tolerance=zero,
            bindings=[],
        )


def test_locked_artifacts_have_six_unique_test_identity_sets() -> None:
    root = Path("artifacts/runs/final_alias_gate_s12_deterministic")
    config_path = root / "run_config.json"
    if not config_path.is_file():
        pytest.skip("locked project artifacts are not present")
    run_signature = json.loads(config_path.read_text())["run_signature"]
    owners: dict[str, int] = {}
    for fold in range(6):
        checkpoint = torch.load(
            root / f"fold_{fold}" / "snn_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["run_signature"] == run_signature
        assert checkpoint["fold"] == fold
        assert checkpoint["model_type"] == "snn"
        for identity in checkpoint["split"]["test_identities"]:
            assert identity not in owners
            owners[identity] = fold
    assert len(owners) == 18
