from __future__ import annotations

import importlib.util
import hashlib
import json
from dataclasses import replace
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


def _inventory_entry(path: Path, *, shape: list[int], dtype: str) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": _TRAIN.sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": shape,
        "dtype": dtype,
    }


def _upgrade_synthetic_cache_to_acquisition_v2(
    cache_root: Path,
    *,
    scientific: bool = True,
    zero_radar: int | None = None,
    invalid_radar: int | None = None,
) -> None:
    session_dir = cache_root / "SESSION"
    spectra_path = session_dir / "spectra.npy"
    spectra = np.load(spectra_path, allow_pickle=False)
    attributes_path = session_dir / "attributes.npy"
    attributes = np.load(attributes_path, allow_pickle=False)
    if zero_radar is not None:
        spectra[:, zero_radar] = 0
        attributes[:, zero_radar] = 0
        np.save(spectra_path, spectra)
        np.save(attributes_path, attributes)
    component_signals = np.ones((*spectra.shape[:4], 320), dtype=np.float16)
    component_path = session_dir / "component_signals.npy"
    np.save(component_path, component_signals)
    timing = np.ones((len(spectra), 3, 320), dtype=np.bool_)
    if invalid_radar is not None:
        timing[:, invalid_radar, 0] = False
    timing_path = session_dir / "radar_timing_valid_mask.npy"
    np.save(timing_path, timing)

    metadata_path = session_dir / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    frequencies_path = session_dir / "frequencies_hz.npy"
    frequencies = np.load(frequencies_path, allow_pickle=False)
    inventory = {
        "spectra": _inventory_entry(
            spectra_path, shape=list(spectra.shape), dtype=str(spectra.dtype)
        ),
        "component_signals": _inventory_entry(
            component_path,
            shape=list(component_signals.shape),
            dtype=str(component_signals.dtype),
        ),
        "attributes": _inventory_entry(
            attributes_path,
            shape=list(attributes.shape),
            dtype=str(attributes.dtype),
        ),
        "frequencies_hz": _inventory_entry(
            frequencies_path,
            shape=list(frequencies.shape),
            dtype=str(frequencies.dtype),
        ),
        "metadata": _inventory_entry(
            metadata_path, shape=list(metadata.shape), dtype="csv"
        ),
        "radar_timing_valid_mask": _inventory_entry(
            timing_path, shape=list(timing.shape), dtype=str(timing.dtype)
        ),
    }
    binding = {
        "schema_version": "snn_rr.feature_cache_acquisition.v2",
        "scientific_eligible": scientific,
    }
    session_manifest = {
        "session_id": "SESSION",
        "row_count": len(metadata),
        "valid_only": True,
        "spectra_shape": list(spectra.shape),
        "component_signals_shape": list(component_signals.shape),
        "attributes_shape": list(attributes.shape),
        "canonical_acquisition_binding": binding,
        "canonical_acquisition_session_manifest_sha256": "1" * 64,
        "radar_timing_valid_mask_shape": list(timing.shape),
        "radar_timing_invalid_interval_count": int(
            timing.size - np.count_nonzero(timing)
        ),
        "radar_timing_mask_contract": {
            "mask_required_for_gap_tolerant_consumers": True,
            "scientific_source_requires_all_true": True,
            "diagnostic_output_trainable": False,
            "invalid_cells_are_exact_zero_but_not_semantic_measurements": True,
        },
        "file_inventory": inventory,
        "inventory_sha256": _TRAIN._canonical_sha256(inventory),
    }
    session_manifest["content_sha256"] = _TRAIN._canonical_content_sha256(
        session_manifest
    )
    (session_dir / "manifest.json").write_text(
        json.dumps(session_manifest), encoding="utf-8"
    )
    root_item = {
        "session_id": "SESSION",
        "status": "ok",
        "cached": False,
        **session_manifest,
    }
    root_manifest = {
        "valid_only": True,
        "row_count": len(metadata),
        "pipeline_sha256": _TRAIN._current_svd_pipeline_sha256(),
        "canonical_acquisition_contract": {
            "schema_version": "snn_rr.feature_cache_acquisition.v2",
            "mode": "strict" if scientific else "diagnostic",
            "selection_scope": "full_cohort",
            "reconstruction_full_cohort_complete": True,
            "full_cohort_complete": True,
            "expected_usable_session_ids": ["SESSION"],
            "expected_usable_session_ids_sha256": _TRAIN._canonical_sha256(
                ["SESSION"]
            ),
            "cache_usable_session_ids": ["SESSION"],
            "cache_usable_session_ids_sha256": _TRAIN._canonical_sha256(
                ["SESSION"]
            ),
            "reconstruction_content_sha256": "2" * 64,
            "scientific_eligible": scientific,
        },
        "canonical_acquisition_reconstruction_content_sha256": "2" * 64,
        "expected_session_ids": ["SESSION"],
        "expected_session_ids_sha256": _TRAIN._canonical_sha256(["SESSION"]),
        "selected_session_ids": ["SESSION"],
        "selected_session_ids_sha256": _TRAIN._canonical_sha256(["SESSION"]),
        "subjects_filter_applied": False,
        "selection_scope": "full_cohort",
        "full_cohort_complete": True,
        "scientific_eligible": scientific,
        "sessions": [root_item],
    }
    root_manifest["content_sha256"] = _TRAIN._canonical_content_sha256(root_manifest)
    (cache_root / "manifest.json").write_text(
        json.dumps(root_manifest), encoding="utf-8"
    )


def _publish_scientific_base_oof_authority(
    cache_root: Path, csv_path: Path, npz_path: Path
) -> Path:
    root_path = cache_root / "manifest.json"
    svd_root = json.loads(root_path.read_text(encoding="utf-8"))
    identity = np.repeat([f"P{index}" for index in range(6)], 2)
    fold = np.repeat(np.arange(6, dtype=np.int16), 2)
    cache_index = np.arange(100, 112, dtype=np.int64)
    target = np.linspace(10.0, 31.0, 12, dtype=np.float32)
    prediction = target + np.tile(np.asarray([1.0, -1.0], dtype=np.float32), 6)
    rr_std = np.ones(12, dtype=np.float32)

    frame = pd.read_csv(csv_path)
    frame["prediction_bpm"] = prediction
    frame["rr_std_bpm"] = rr_std
    frame.to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        cache_index=cache_index,
        identity=identity,
        fold=fold,
        reference_rr_bpm=target,
        prediction_bpm=prediction,
        rr_std_bpm=rr_std,
    )

    reconstruction_dir = cache_root.parent / "reconstruction"
    reconstruction_dir.mkdir()
    reconstruction = {
        "schema_version": "snn_rr.acquisition_reconstruction.v2",
        "scientific_eligible": True,
    }
    reconstruction["content_sha256"] = _TRAIN._canonical_content_sha256(
        reconstruction
    )
    reconstruction_path = reconstruction_dir / "manifest.json"
    reconstruction_path.write_text(json.dumps(reconstruction), encoding="utf-8")

    canonical_root = cache_root.parent / "canonical"
    canonical_root.mkdir()
    canonical_session = canonical_root / "SESSION"
    canonical_session.mkdir()
    np.save(canonical_session / "maps.npy", np.zeros((1, 1), dtype=np.float16))
    np.save(canonical_session / "aux.npy", np.zeros((1, 1), dtype=np.float32))
    np.save(
        canonical_session / "frequencies_hz.npy",
        np.asarray([0.1], dtype=np.float32),
    )
    pd.DataFrame({"session_id": ["SESSION"]}).to_csv(
        canonical_session / "metadata.csv", index=False
    )
    (canonical_session / "manifest.json").write_text("{}", encoding="utf-8")
    canonical_contract = dict(svd_root["canonical_acquisition_contract"])
    canonical_contract["reconstruction_manifest"] = str(reconstruction_path)
    canonical_contract["reconstruction_content_sha256"] = reconstruction[
        "content_sha256"
    ]
    canonical = {
        "acquisition_contract": canonical_contract,
        "sessions": [{"session_id": "SESSION", "status": "ok"}],
    }
    canonical["content_sha256"] = _TRAIN._canonical_content_sha256(canonical)
    canonical_manifest_path = canonical_root / "manifest.json"
    canonical_manifest_path.write_text(json.dumps(canonical), encoding="utf-8")
    canonical_manifest_sha256 = _TRAIN.sha256_file(canonical_manifest_path)
    canonical_inventory_sha256, canonical_inventory_count = (
        _TRAIN._feature_cache_inventory_sha256(canonical_root, canonical)
    )

    svd_root["canonical_cache"] = str(canonical_root)
    svd_root["canonical_manifest_sha256"] = canonical_manifest_sha256
    svd_root["canonical_acquisition_contract"] = canonical_contract
    svd_root[
        "canonical_acquisition_reconstruction_content_sha256"
    ] = reconstruction["content_sha256"]
    svd_root["content_sha256"] = _TRAIN._canonical_content_sha256(svd_root)
    root_path.write_text(json.dumps(svd_root), encoding="utf-8")

    cache_provenance = {
        "classification": "acquisition_scientific",
        "root_manifest_path": str(canonical_manifest_path.resolve()),
        "root_manifest_sha256": canonical_manifest_sha256,
        "root_manifest_content_sha256": canonical["content_sha256"],
        "acquisition_schema_version": "snn_rr.feature_cache_acquisition.v2",
        "acquisition_mode": "strict",
        "scientific_eligible": True,
        "config_sha256": "3" * 64,
        "pipeline_sha256": "4" * 64,
        "reconstruction_content_sha256": reconstruction["content_sha256"],
        "inventory_sha256": canonical_inventory_sha256,
        "inventory_file_count": canonical_inventory_count,
        "selected_sessions": ["SESSION"],
    }
    cache_provenance["content_sha256"] = _TRAIN._canonical_sha256(
        cache_provenance
    )
    run_signature = "scientific-run-signature"
    run_config = {
        "run_signature": run_signature,
        "arguments": {"cache_trust_mode": "scientific"},
        "claim_classification": "retrospective_scientific_noncommercial",
        "cache_provenance": cache_provenance,
    }
    run_config_path = cache_root.parent / "run_config.json"
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
    source_path = cache_root.parent / "runtime_source.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")

    identity_to_fold = {f"P{index}": index for index in range(6)}
    checkpoints: dict[str, object] = {}
    commits: dict[str, object] = {}
    frozen_path = cache_root.parent / "snn_oof.npz"
    np.savez_compressed(
        frozen_path,
        index=cache_index,
        target=target,
        prediction=prediction,
        fold=fold,
        run_signature=np.asarray(run_signature),
    )
    frozen_sha256 = _TRAIN.sha256_file(frozen_path)
    inference_signature = "identity-disjoint-inference-signature"
    for fold_number in range(6):
        fold_key = str(fold_number)
        test_identities = [f"P{fold_number}"]
        checkpoint_path = cache_root.parent / f"fold_{fold_number}.pt"
        validation_identity = f"P{(fold_number + 1) % 6}"
        train_identities = sorted(
            set(identity_to_fold) - set(test_identities) - {validation_identity}
        )
        torch.save(
            {
                "model_type": "snn",
                "fold": fold_number,
                "run_signature": run_signature,
                "cache_provenance": cache_provenance,
                "split": {
                    "train_identities": train_identities,
                    "validation_identities": [validation_identity],
                    "test_identities": test_identities,
                },
            },
            checkpoint_path,
        )
        checkpoint_sha256 = _TRAIN.sha256_file(checkpoint_path)
        checkpoints[fold_key] = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "test_identities": test_identities,
        }
        rows = np.flatnonzero(fold == fold_number)
        artifact_path = cache_root.parent / f"fold_{fold_number}_all_windows.npz"
        np.savez_compressed(
            artifact_path,
            index=cache_index[rows],
            prediction=prediction[rows],
            rr_std=rr_std[rows],
            fold=np.full(len(rows), fold_number, dtype=np.int16),
            run_signature=np.asarray(run_signature),
            inference_signature=np.asarray(inference_signature),
            checkpoint_sha256=np.asarray(checkpoint_sha256),
        )
        artifact_sha256 = _TRAIN.sha256_file(artifact_path)
        commits[fold_key] = {
            "fold": fold_number,
            "run_signature": run_signature,
            "inference_signature": inference_signature,
            "checkpoint_sha256": checkpoint_sha256,
            "frozen_oof_sha256": frozen_sha256,
            "row_count": len(rows),
            "expected_index_sha256": hashlib.sha256(
                cache_index[rows].astype(np.int64).tobytes()
            ).hexdigest(),
            "deployment_allowlist": ["index", "prediction", "rr_std"],
            "excluded_fields": ["target", "observable"],
            "artifact": {
                "path": str(artifact_path),
                "sha256": artifact_sha256,
                "bytes": artifact_path.stat().st_size,
            },
        }

    provenance = {
        "schema_version": _TRAIN.BASE_OOF_AUTHORITY_SCHEMA,
        "scientific_eligible": True,
        "claim_classification": "retrospective_scientific_noncommercial",
        "commercial_claim_allowed": False,
        "identity_disjoint": True,
        "row_exact_cover": True,
        "row_count": len(frame),
        "source_run_config": str(run_config_path),
        "source_run_config_sha256": _TRAIN.sha256_file(run_config_path),
        "run_signature": run_signature,
        "inference_signature": inference_signature,
        "runtime_source_sha256": {
            str(source_path.resolve()): _TRAIN.sha256_file(source_path)
        },
        "canonical_cache_provenance": cache_provenance,
        "identity_to_test_fold": identity_to_fold,
        "identity_to_test_fold_sha256": _TRAIN._canonical_sha256(identity_to_fold),
        "row_fold_binding_sha256": _TRAIN._row_fold_binding_sha256(
            cache_index, identity, fold
        ),
        "checkpoints": checkpoints,
        "verified_fold_commits": commits,
        "label_free_forward": {
            "verified": True,
            "model_inputs": ["map", "radar_mask", "aux"],
            "target_or_qc_inputs": [],
        },
        "frozen_valid_oof_verification": {
            "source": str(frozen_path),
            "source_sha256": frozen_sha256,
            "resolved_tolerance": {"prediction_bpm": 0.03},
        },
        "outputs": {
            "csv": str(csv_path.resolve()),
            "npz": str(npz_path.resolve()),
            "csv_sha256": _TRAIN.sha256_file(csv_path),
            "npz_sha256": _TRAIN.sha256_file(npz_path),
            "csv_bytes": csv_path.stat().st_size,
            "npz_bytes": npz_path.stat().st_size,
        },
    }
    provenance["content_sha256"] = _TRAIN._canonical_content_sha256(provenance)
    provenance_path = csv_path.parent / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    return provenance_path


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


@pytest.mark.parametrize("payload", ["component_signals.npy", "spectra.npy"])
def test_acquisition_v2_payload_hash_is_mandatory_even_when_disabled(
    tmp_path: Path, payload: str
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache)
    path = cache / "SESSION" / payload
    values = np.load(path, allow_pickle=False)
    values.flat[0] = values.flat[0] + 1
    np.save(path, values)
    with pytest.raises(RuntimeError, match="inventory SHA-256 mismatch"):
        _TRAIN.load_aligned_experiment(
            cache, csv_path, npz_path, verify_file_hashes=False
        )


def test_acquisition_v2_timing_mask_hash_is_mandatory_even_when_disabled(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache)
    path = cache / "SESSION" / "radar_timing_valid_mask.npy"
    timing = np.load(path, allow_pickle=False)
    timing[0, 0, 0] = False
    np.save(path, timing)
    with pytest.raises(RuntimeError, match="inventory SHA-256 mismatch"):
        _TRAIN.load_aligned_experiment(
            cache, csv_path, npz_path, verify_file_hashes=False
        )


def test_scientific_acquisition_v2_requires_base_oof_authority(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=True)
    with pytest.raises(RuntimeError, match="requires.*base-oof-provenance"):
        _TRAIN.load_aligned_experiment(cache, csv_path, npz_path)


def test_standard_svd_loader_derives_scope_before_accepting_scientific_flag(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=True)
    root_path = cache / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["subjects_filter_applied"] = True
    root["selection_scope"] = "diagnostic_subset"
    root["full_cohort_complete"] = False
    root["content_sha256"] = _TRAIN._canonical_content_sha256(root)
    root_path.write_text(json.dumps(root), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scientific_eligible does not match"):
        _TRAIN.load_aligned_experiment(cache, csv_path, npz_path)


def test_scientific_base_oof_authority_accepts_exact_identity_owned_graph(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=True)
    provenance_path = _publish_scientific_base_oof_authority(
        cache, csv_path, npz_path
    )
    experiment = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        base_oof_provenance=provenance_path,
    )
    assert experiment.provenance["base_oof_authority"]["scientific_eligible"]
    assert (
        experiment.provenance["claim_classification"]
        == "retrospective_scientific_noncommercial"
    )


def test_scientific_v2_arrays_are_owned_and_survive_source_mutation(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=True)
    provenance_path = _publish_scientific_base_oof_authority(
        cache, csv_path, npz_path
    )
    experiment = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        base_oof_provenance=provenance_path,
    )
    session = experiment.sessions[0]
    loaded = {
        "spectra.npy": session.spectra,
        "component_signals.npy": session.component_signals,
        "attributes.npy": session.attributes,
        "frequencies_hz.npy": session.frequencies_hz,
        "radar_timing_valid_mask.npy": session.radar_timing_valid_mask,
    }
    snapshots: dict[str, np.ndarray] = {}
    for name, array in loaded.items():
        assert array is not None
        assert not isinstance(array, np.memmap)
        assert array.flags.owndata
        assert not array.flags.writeable
        snapshots[name] = np.array(array, copy=True)

    session_dir = cache / "SESSION"
    for name in loaded:
        mutable = np.load(session_dir / name, mmap_mode="r+", allow_pickle=False)
        if mutable.dtype == np.bool_:
            mutable.flat[0] = not bool(mutable.flat[0])
        else:
            mutable.flat[0] = mutable.flat[0] + 0.25
        mutable.flush()
        del mutable

    for name, array in loaded.items():
        assert array is not None
        assert np.array_equal(array, snapshots[name])

    output_dir = tmp_path / "source_drift_output_must_not_exist"
    args = _TRAIN.parse_args(["--output-dir", str(output_dir)])
    with pytest.raises(RuntimeError, match="authority source changed"):
        _TRAIN.train_fold(args, experiment, 0, torch.device("cpu"), "test")
    assert not output_dir.exists()


def test_forged_target_equal_base_csv_and_npz_are_rejected(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=True)
    provenance_path = _publish_scientific_base_oof_authority(
        cache, csv_path, npz_path
    )
    frame = pd.read_csv(csv_path)
    frame["prediction_bpm"] = frame["rr_bpm"]
    frame.to_csv(csv_path, index=False)
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["prediction_bpm"] = np.asarray(arrays["reference_rr_bpm"]).copy()
    np.savez_compressed(npz_path, **arrays)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _TRAIN.load_aligned_experiment(
            cache,
            csv_path,
            npz_path,
            base_oof_provenance=provenance_path,
        )


def test_base_oof_checkpoint_test_identity_cannot_also_be_training_identity(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=True)
    provenance_path = _publish_scientific_base_oof_authority(
        cache, csv_path, npz_path
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(provenance["checkpoints"]["0"]["path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["split"]["train_identities"].append("P0")
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = _TRAIN.sha256_file(checkpoint_path)
    provenance["checkpoints"]["0"]["sha256"] = checkpoint_sha256

    marker = provenance["verified_fold_commits"]["0"]
    marker["checkpoint_sha256"] = checkpoint_sha256
    artifact_path = Path(marker["artifact"]["path"])
    with np.load(artifact_path, allow_pickle=False) as archive:
        artifact_arrays = {
            name: np.asarray(archive[name]) for name in archive.files
        }
    artifact_arrays["checkpoint_sha256"] = np.asarray(checkpoint_sha256)
    np.savez_compressed(artifact_path, **artifact_arrays)
    marker["artifact"]["sha256"] = _TRAIN.sha256_file(artifact_path)
    marker["artifact"]["bytes"] = artifact_path.stat().st_size
    provenance["content_sha256"] = _TRAIN._canonical_content_sha256(provenance)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(RuntimeError, match="checkpoint 0 internal authority"):
        _TRAIN.load_aligned_experiment(
            cache,
            csv_path,
            npz_path,
            base_oof_provenance=provenance_path,
        )


def test_acquisition_v2_structural_mask_overrides_numeric_zero_and_zeroes_invalid(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path / "zero")
    _upgrade_synthetic_cache_to_acquisition_v2(
        cache, scientific=False, zero_radar=0
    )
    experiment = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        verify_file_hashes=False,
        allow_diagnostic_acquisition_inspection=True,
    )
    sample = _TRAIN.SVDSourceDataset(
        experiment, [0], np.zeros((len(experiment.metadata), 1), dtype=np.float32)
    )[0]
    assert bool(sample["radar_mask"][0]) is True
    assert torch.count_nonzero(sample["spectra"][0]) == 0
    assert float(sample["classical_rr"][1]) > 0

    cache, csv_path, npz_path = _synthetic_cache(tmp_path / "invalid")
    _upgrade_synthetic_cache_to_acquisition_v2(
        cache, scientific=False, invalid_radar=1
    )
    experiment = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        verify_file_hashes=False,
        allow_diagnostic_acquisition_inspection=True,
    )
    feature_columns = tuple(experiment.provenance["feature_allowlist"])
    sample = _TRAIN.SVDSourceDataset(
        experiment,
        [0],
        np.ones((len(experiment.metadata), len(feature_columns)), dtype=np.float32),
    )[0]
    assert bool(sample["radar_mask"][1]) is False
    assert torch.count_nonzero(sample["spectra"][1]) == 0
    assert torch.count_nonzero(sample["attributes"][1]) == 0
    assert sample["classical_rr"][0] == 0
    assert sample["classical_rr"][2] == 0
    assert sample["classical_std"][0] == 0
    assert sample["classical_std"][2] == 0
    assert sample["base_features"][feature_columns.index("classical_confidence")] == 0
    assert sample["base_features"][feature_columns.index("radar_peak_2_bpm")] == 0
    assert sample["base_features"][feature_columns.index("radar_peak_spread_bpm")] == 0


def test_acquisition_v2_cannot_be_downgraded_by_stripping_bindings(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache)
    session_path = cache / "SESSION" / "manifest.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    del session["canonical_acquisition_binding"]
    session["content_sha256"] = _TRAIN._canonical_content_sha256(session)
    session_path.write_text(json.dumps(session), encoding="utf-8")
    root_path = cache / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    del root["canonical_acquisition_contract"]
    del root["canonical_acquisition_reconstruction_content_sha256"]
    root["sessions"] = [
        {"session_id": "SESSION", "status": "ok", "cached": False, **session}
    ]
    root["content_sha256"] = _TRAIN._canonical_content_sha256(root)
    root_path.write_text(json.dumps(root), encoding="utf-8")
    with pytest.raises(RuntimeError, match="lacks its canonical root"):
        _TRAIN.load_aligned_experiment(
            cache, csv_path, npz_path, verify_file_hashes=False
        )


def test_svd_main_rejects_diagnostic_cache_before_training_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=False)
    events: list[str] = []
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
        "fit_robust_scaler",
        lambda *args, **kwargs: events.append("scaler"),
    )
    monkeypatch.setattr(
        _TRAIN,
        "SourceSeparatedRRSNN",
        lambda *args, **kwargs: events.append("model"),
    )
    output_dir = tmp_path / "must_not_exist"

    with pytest.raises(RuntimeError, match="inspection-only"):
        _TRAIN.main(
            [
                "--svd-cache",
                str(cache),
                "--base-oof-csv",
                str(csv_path),
                "--base-oof-npz",
                str(npz_path),
                "--device",
                "cuda",
                "--output-dir",
                str(output_dir),
            ]
        )

    assert events == []
    assert not output_dir.exists()


def test_svd_train_fold_rejects_explicit_diagnostic_inspection_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=False)
    experiment = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        verify_file_hashes=False,
        allow_diagnostic_acquisition_inspection=True,
    )
    events: list[str] = []
    monkeypatch.setattr(
        _TRAIN,
        "fit_robust_scaler",
        lambda *args, **kwargs: events.append("scaler"),
    )
    monkeypatch.setattr(
        _TRAIN,
        "SourceSeparatedRRSNN",
        lambda *args, **kwargs: events.append("model"),
    )
    output_dir = tmp_path / "must_not_exist"
    args = _TRAIN.parse_args(["--output-dir", str(output_dir)])

    with pytest.raises(RuntimeError, match="inspection-only"):
        _TRAIN.train_fold(args, experiment, 0, torch.device("cpu"), "test")

    assert events == []
    assert not output_dir.exists()


def test_svd_train_fold_rejects_forged_in_memory_scientific_promotion(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=False)
    experiment = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        verify_file_hashes=False,
        allow_diagnostic_acquisition_inspection=True,
    )
    output_dir = tmp_path / "forged_output_must_not_exist"
    args = _TRAIN.parse_args(["--output-dir", str(output_dir)])

    # These were the complete mutable-field bypass before the loader-issued
    # receipt became the authority source.
    root_contract = experiment.root_manifest["canonical_acquisition_contract"]
    root_contract["mode"] = "strict"
    root_contract["scientific_eligible"] = True
    experiment.root_manifest["scientific_eligible"] = True
    experiment.provenance["base_oof_authority"]["scientific_eligible"] = True
    experiment.provenance[
        "claim_classification"
    ] = "retrospective_scientific_noncommercial"

    with pytest.raises(RuntimeError, match="inspection-only"):
        _TRAIN.train_fold(args, experiment, 0, torch.device("cpu"), "test")

    # A receipt-shaped copy with its decision flipped was not issued by the
    # loader and must fail before any output directory is created.
    experiment.training_authority_receipt = replace(
        experiment.training_authority_receipt,
        training_authorized=True,
    )
    with pytest.raises(RuntimeError, match="issued by load_aligned_experiment"):
        _TRAIN.train_fold(args, experiment, 0, torch.device("cpu"), "test")

    assert not output_dir.exists()


def test_svd_train_fold_rejects_target_encoded_allowed_metadata_before_output(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=True)
    provenance_path = _publish_scientific_base_oof_authority(
        cache, csv_path, npz_path
    )
    experiment = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        base_oof_provenance=provenance_path,
    )
    experiment.metadata["prediction_bpm"] = experiment.metadata["rr_bpm"].to_numpy()
    output_dir = tmp_path / "metadata_forgery_must_not_exist"
    args = _TRAIN.parse_args(["--output-dir", str(output_dir)])

    with pytest.raises(RuntimeError, match="in-memory experiment"):
        _TRAIN.train_fold(args, experiment, 0, torch.device("cpu"), "test")

    assert not output_dir.exists()


def test_svd_train_fold_rejects_owned_readonly_target_encoded_array_before_output(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=True)
    provenance_path = _publish_scientific_base_oof_authority(
        cache, csv_path, npz_path
    )
    experiment = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        base_oof_provenance=provenance_path,
    )
    session = experiment.sessions[0]
    forged = np.array(session.spectra, copy=True, order="C")
    forged[:, 0, 0, 0, 0] = experiment.metadata["rr_bpm"].to_numpy(
        dtype=forged.dtype
    )
    forged.setflags(write=False)
    assert forged.flags.owndata and not forged.flags.writeable
    session.spectra = forged
    output_dir = tmp_path / "array_forgery_must_not_exist"
    args = _TRAIN.parse_args(["--output-dir", str(output_dir)])

    with pytest.raises(RuntimeError, match="session payload"):
        _TRAIN.train_fold(args, experiment, 0, torch.device("cpu"), "test")

    assert not output_dir.exists()


def test_diagnostic_v2_cannot_be_reissued_as_legacy_authority(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=False)
    experiment = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        allow_diagnostic_acquisition_inspection=True,
    )
    assert experiment.training_authority_receipt is not None
    assert experiment.training_authority_receipt.acquisition_v2 is True
    assert experiment.training_authority_receipt.training_authorized is False

    # The issuer has no caller-controlled acquisition/scientific booleans and
    # consumes the loader's pending exact-object registration only once.
    with pytest.raises(RuntimeError, match="only be issued once"):
        _TRAIN._issue_svd_training_authority_receipt(
            experiment=experiment,
            authority_path=tmp_path / "missing-provenance.json",
        )


def test_svd_authority_rejects_exact_clone_subclass_and_session_subclass(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=True)
    provenance_path = _publish_scientific_base_oof_authority(
        cache, csv_path, npz_path
    )
    experiment = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        base_oof_provenance=provenance_path,
    )
    fields = {
        name: getattr(experiment, name)
        for name in _TRAIN.AlignedSVDExperiment.__dataclass_fields__
    }
    clone = _TRAIN.AlignedSVDExperiment(**fields)

    class TargetInjectingExperiment(_TRAIN.AlignedSVDExperiment):
        def arrays_for_position(self, position: int) -> tuple[np.ndarray, np.ndarray]:
            spectra, attributes = super().arrays_for_position(position)
            forged = np.array(spectra, copy=True)
            forged.flat[0] = float(self.metadata.iloc[position]["rr_bpm"])
            return forged, attributes

    subclass = TargetInjectingExperiment(**fields)
    args = _TRAIN.parse_args(["--output-dir", str(tmp_path / "must_not_exist")])
    for candidate in (clone, subclass):
        with pytest.raises(RuntimeError, match="issued by load_aligned_experiment"):
            _TRAIN.train_fold(
                args, candidate, 0, torch.device("cpu"), "test"
            )

    original_sessions = experiment.sessions

    class TargetInjectingSession(_TRAIN.SVDSessionArrays):
        pass

    source = original_sessions[0]
    experiment.sessions = [
        TargetInjectingSession(
            **{
                name: getattr(source, name)
                for name in _TRAIN.SVDSessionArrays.__dataclass_fields__
            }
        )
    ]
    with pytest.raises(RuntimeError, match="in-memory experiment"):
        _TRAIN.train_fold(args, experiment, 0, torch.device("cpu"), "test")
    experiment.sessions = original_sessions
    assert not Path(args.output_dir).exists()


def test_scientific_extra_model_output_cannot_enter_feature_graph(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=True)
    frame = pd.read_csv(csv_path)
    frame["prediction_uncalibrated_bpm"] = frame["rr_bpm"]
    frame["quality"] = frame["rr_bpm"]
    frame.to_csv(csv_path, index=False)
    provenance_path = _publish_scientific_base_oof_authority(
        cache, csv_path, npz_path
    )
    experiment = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        base_oof_provenance=provenance_path,
    )
    assert experiment.provenance["feature_allowlist"] == [
        "prediction_bpm",
        "rr_std_bpm",
    ]
    assert "prediction_uncalibrated_bpm" not in experiment.metadata
    assert "quality" not in experiment.metadata
    np.testing.assert_allclose(
        experiment.metadata["prediction_bpm"],
        np.load(npz_path, allow_pickle=False)["prediction_bpm"],
    )


def test_legacy_training_requires_loader_and_caller_explicit_mode(
    tmp_path: Path,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    inspection = _TRAIN.load_aligned_experiment(cache, csv_path, npz_path)
    assert inspection.training_authority_receipt is not None
    assert inspection.training_authority_receipt.acquisition_v2 is False
    assert inspection.training_authority_receipt.training_authorized is False
    with pytest.raises(RuntimeError, match="inspection-only"):
        _TRAIN._assert_training_cache_authority(inspection)

    reproduction = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        historical_legacy_reproduction=True,
    )
    assert reproduction.training_authority_receipt is not None
    assert reproduction.training_authority_receipt.training_authorized is True
    with pytest.raises(RuntimeError, match="explicit historical"):
        _TRAIN._assert_training_cache_authority(reproduction)
    _TRAIN._assert_training_cache_authority(
        reproduction, historical_legacy_reproduction=True
    )


def test_scientific_archives_are_consumed_from_stable_byte_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, csv_path, npz_path = _synthetic_cache(tmp_path)
    _upgrade_synthetic_cache_to_acquisition_v2(cache, scientific=True)
    provenance_path = _publish_scientific_base_oof_authority(
        cache, csv_path, npz_path
    )
    original_numpy_load = np.load
    original_torch_load = _TRAIN.torch.load
    numpy_sources: list[object] = []
    torch_calls: list[tuple[object, object]] = []

    def guarded_numpy_load(source: object, *args: object, **kwargs: object) -> object:
        numpy_sources.append(source)
        if isinstance(source, (str, Path)):
            raise AssertionError("scientific archive reopened by path")
        return original_numpy_load(source, *args, **kwargs)

    def guarded_torch_load(source: object, *args: object, **kwargs: object) -> object:
        torch_calls.append((source, kwargs.get("weights_only")))
        return original_torch_load(source, *args, **kwargs)

    monkeypatch.setattr(_TRAIN.np, "load", guarded_numpy_load)
    monkeypatch.setattr(_TRAIN.torch, "load", guarded_torch_load)
    experiment = _TRAIN.load_aligned_experiment(
        cache,
        csv_path,
        npz_path,
        base_oof_provenance=provenance_path,
    )
    assert experiment.training_authority_receipt is not None
    assert experiment.training_authority_receipt.training_authorized is True
    assert numpy_sources
    assert torch_calls
    assert all(not isinstance(source, (str, Path)) for source in numpy_sources)
    assert all(not isinstance(source, (str, Path)) for source, _ in torch_calls)
    assert all(weights_only is True for _, weights_only in torch_calls)


def test_training_source_closure_includes_trainers_models_and_forward_dependencies() -> None:
    paths = {path.resolve() for path in _TRAIN._training_source_paths()}
    required = {
        _TRAIN.Path(_TRAIN.__file__).resolve(),
        (_TRAIN.SOURCE_ROOT / "snn_rr/svd_models.py").resolve(),
        (_TRAIN.SOURCE_ROOT / "snn_rr/models.py").resolve(),
        (_TRAIN.SOURCE_ROOT / "snn_rr/metrics.py").resolve(),
    }
    assert required <= paths


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
