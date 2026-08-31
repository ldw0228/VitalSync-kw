from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

from snn_rr.acquisition_contract import load_acquisition_cohort_authority
from snn_rr.synchronization import TimeMapping


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_features.py"
_SPEC = importlib.util.spec_from_file_location("build_features_acquisition_v2", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_BUILD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BUILD
_SPEC.loader.exec_module(_BUILD)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("start_coordinate", "end_coordinate", "expected"),
    [
        (10.0, 20.0, (10, 20)),
        (10.25, 20.25, (11, 21)),
        (10.49, 20.49, (11, 21)),
        (10.51, 20.51, (11, 21)),
        (10.0 + 5.0e-10, 20.0 - 5.0e-10, (10, 20)),
        (10.0 + 2.0e-9, 20.0 - 2.0e-9, (11, 20)),
    ],
)
def test_half_open_sample_bounds_use_robust_ceil_for_both_edges(
    start_coordinate: float,
    end_coordinate: float,
    expected: tuple[int, int],
) -> None:
    sample_rate_hz = 250.0
    assert _BUILD._half_open_sample_bounds(
        start_coordinate / sample_rate_hz,
        end_coordinate / sample_rate_hz,
        sample_rate_hz,
    ) == expected


def test_half_open_sample_bounds_reject_invalid_support() -> None:
    with pytest.raises(ValueError, match="end > start"):
        _BUILD._half_open_sample_bounds(1.0, 1.0, 250.0)
    with pytest.raises(ValueError, match="sample rate"):
        _BUILD._half_open_sample_bounds(0.0, 1.0, 0.0)


def test_feature_pipeline_digest_includes_synchronization_mapping_source() -> None:
    paths = _BUILD._pipeline_paths()
    for filename in (
        "synchronization.py",
        "raw_snapshot.py",
        "acquisition_protocol.py",
        "radar_timing.py",
    ):
        assert _BUILD.SOURCE_ROOT / "snn_rr" / filename in paths


@pytest.mark.parametrize(
    "cache_ids",
    [
        ("S01_A",),
        ("S01_A", "S02_B"),
    ],
)
def test_explicit_subject_filter_can_never_claim_untargeted_full_cohort(
    cache_ids: tuple[str, ...],
) -> None:
    expected = ("S01_A", "S02_B")
    derived = _BUILD._derive_acquisition_cache_scope(
        subjects_filter_applied=True,
        reconstruction_full_cohort_complete=True,
        expected_usable_session_ids=expected,
        cache_usable_session_ids=cache_ids,
    )
    assert derived == {
        "subjects_filter_applied": True,
        "selection_scope": "diagnostic_subset",
        "full_cohort_complete": False,
    }


def test_untargeted_exact_coverage_can_claim_full_cohort() -> None:
    expected = ("S01_A", "S02_B")
    derived = _BUILD._derive_acquisition_cache_scope(
        subjects_filter_applied=False,
        reconstruction_full_cohort_complete=True,
        expected_usable_session_ids=expected,
        cache_usable_session_ids=expected,
    )
    assert derived == {
        "subjects_filter_applied": False,
        "selection_scope": "full_cohort",
        "full_cohort_complete": True,
    }


def test_raw_bindings_are_checked_before_any_cache_fast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = SimpleNamespace(
        subject_id="S01_TEST",
        usable=True,
        path=tmp_path / "dataset" / "S01_TEST",
    )
    contract = SimpleNamespace(session_id="S01_TEST")

    def reject_raw(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("raw binding changed")

    monkeypatch.setattr(_BUILD, "validate_raw_input_bindings", reject_raw)
    with pytest.raises(RuntimeError, match="raw binding changed"):
        _BUILD.build_subject(
            subject,
            {},
            tmp_path / "cache",
            False,
            config_sha256="1" * 64,
            pipeline_sha256="2" * 64,
            acquisition_contract=contract,
            acquisition_mode="diagnostic",
        )


def test_full_cohort_acquisition_build_skips_unusable_session_without_contract(
    tmp_path: Path,
) -> None:
    subject = SimpleNamespace(subject_id="S24_KHJ", usable=False)
    result = _BUILD.build_subject(
        subject,
        {},
        tmp_path / "cache",
        False,
        config_sha256="1" * 64,
        pipeline_sha256="2" * 64,
        acquisition_contract=None,
        acquisition_mode="diagnostic",
    )
    assert result == {
        "session_id": "S24_KHJ",
        "status": "skipped",
        "reason": "missing paired radar/BIOPAC",
    }


def test_cached_inventory_hashes_structural_timing_mask(tmp_path: Path) -> None:
    mask_path = tmp_path / "radar_timing_valid_mask.npy"
    mask = np.ones((1, 3, 320), dtype=np.bool_)
    np.save(mask_path, mask, allow_pickle=False)
    inventory = {
        "radar_timing_valid_mask": {
            "path": mask_path.name,
            "bytes": mask_path.stat().st_size,
            "sha256": _sha256(mask_path),
            "shape": list(mask.shape),
            "dtype": "bool",
        }
    }
    assert _BUILD._inventory_files_match(tmp_path, inventory)

    mask[0, 1, 17] = False
    np.save(mask_path, mask, allow_pickle=False)
    assert not _BUILD._inventory_files_match(tmp_path, inventory)


def test_acquisition_fast_path_requires_exact_inventory_and_manifest_hash(
    tmp_path: Path,
) -> None:
    arrays = {
        "maps": np.zeros((1, 3, 2, 4), dtype=np.float16),
        "aux": np.zeros((1, 2), dtype=np.float32),
        "frequencies_hz": np.asarray([0.1, 0.2], dtype=np.float32),
        "radar_timing_valid_mask": np.ones((1, 3, 320), dtype=np.bool_),
    }
    inventory: dict[str, object] = {}
    for key, array in arrays.items():
        path = tmp_path / f"{key}.npy"
        np.save(path, array, allow_pickle=False)
        inventory[key] = _BUILD._array_inventory(path, array)
    metadata = pd.DataFrame({"session_id": ["S01_TEST"]})
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    inventory["metadata"] = _BUILD._metadata_inventory(metadata_path, metadata)
    manifest = {
        "file_inventory": inventory,
        "inventory_sha256": _BUILD._inventory_sha256(inventory),
    }
    manifest["content_sha256"] = _BUILD._canonical_content_sha256(manifest)

    assert _BUILD._acquisition_cached_manifest_is_current(
        tmp_path, manifest, require_range_aux=False
    )

    reduced = dict(manifest)
    reduced_inventory = dict(inventory)
    reduced_inventory.pop("radar_timing_valid_mask")
    reduced["file_inventory"] = reduced_inventory
    reduced["inventory_sha256"] = _BUILD._inventory_sha256(reduced_inventory)
    reduced["content_sha256"] = _BUILD._canonical_content_sha256(reduced)
    assert not _BUILD._acquisition_cached_manifest_is_current(
        tmp_path, reduced, require_range_aux=False
    )


def _diagnostic_v3_reconstruction() -> SimpleNamespace:
    authority_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "acquisition_cohort_v1.yaml"
    )
    authority = load_acquisition_cohort_authority(authority_path)
    all_ids = tuple(authority.expected_session_ids)
    usable_ids = tuple(authority.expected_usable_session_ids)
    manifest = {
        "schema_version": _BUILD.ACQUISITION_SCHEMA_V3,
        "expected_session_ids": list(all_ids),
        "expected_session_ids_sha256": _BUILD._canonical_value_sha256(
            list(all_ids)
        ),
        "expected_usable_session_ids": list(usable_ids),
        "expected_usable_session_ids_sha256": _BUILD._canonical_value_sha256(
            list(usable_ids)
        ),
        "selected_session_ids": list(all_ids),
        "selected_session_ids_sha256": _BUILD._canonical_value_sha256(
            list(all_ids)
        ),
        "dataset_session_count": 30,
        "dataset_usable_session_count": 29,
        "dataset_physical_identity_count": 18,
        "selected_session_count": 30,
        "session_count": 30,
        "cohort_authority_content_sha256": authority.content_sha256,
        "subjects_filter_applied": False,
        "selection_scope": "full_cohort",
        "execution_complete": True,
        "full_cohort_complete": True,
        "complete": True,
        "scientific_eligible": False,
        "raw_consumed_bytes_verified": True,
        "timing_adjudicated": True,
        "sync_raw_replay_verified": False,
        "protocol_raw_replay_verified": False,
        "sessions": [
            {
                "session_id": session_id,
                "usable": session_id != "S24_KHJ",
            }
            for session_id in all_ids
        ],
    }
    return SimpleNamespace(
        manifest=manifest,
        sessions={session_id: object() for session_id in usable_ids},
        full_cohort_complete=True,
        scientific_eligible=False,
    )


def test_v3_projection_requires_real_full30_to_exact29() -> None:
    acquisition = _diagnostic_v3_reconstruction()
    usable = _BUILD._validate_v3_reconstruction_projection(acquisition)
    assert len(usable) == 29
    assert "S24_KHJ" not in usable

    acquisition.manifest["selected_session_ids"] = acquisition.manifest[
        "selected_session_ids"
    ][:-1]
    with pytest.raises(ValueError, match="exact full 30-session"):
        _BUILD._validate_v3_reconstruction_projection(acquisition)


@pytest.mark.parametrize(
    "data",
    [
        {"model_hz": 9.999, "window_seconds": 32.0},
        {"model_hz": 10.0, "window_seconds": 31.9},
        {"model_hz": True, "window_seconds": 32.0},
    ],
)
def test_v3_config_rejects_nonexact_32s_by_10hz_support(
    data: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="exact 32 s x 10 Hz = 320"):
        _BUILD._validate_v3_feature_config({"data": data})

    _BUILD._validate_v3_feature_config(
        {"data": {"model_hz": 10.0, "window_seconds": 32.0}}
    )


def test_v3_output_tree_disjointness_detects_nested_input_aliases(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    assert _BUILD._paths_overlap(raw / "cache", raw)
    assert _BUILD._paths_overlap(raw, raw / "session")
    assert not _BUILD._paths_overlap(tmp_path / "cache", raw)


def _v3_manifest_fixture() -> dict[str, object]:
    frequencies = np.asarray([0.1, 0.2], dtype=np.float64)
    maps = np.ones((1, 3, 2, 182), dtype=np.float16)
    timing_valid = np.ones((1, 3, 320), dtype=np.bool_)
    timing_valid[0, 1, 3] = False
    reasons = np.zeros(timing_valid.shape, dtype=np.uint8)
    reasons[0, 1, 3] = 1
    (
        map_available,
        aux_available,
        aux_names,
        availability_names,
    ) = _BUILD._v3_feature_availability(
        timing_valid,
        auxiliary_frequency_bins=4,
    )
    aux = np.ones((1, len(aux_names)), dtype=np.float32)
    maps[~map_available] = np.float16(0.0)
    aux[~aux_available] = np.float32(0.0)
    availability = np.concatenate((map_available, aux_available), axis=1)

    metadata_values: dict[str, object] = {
        "session_id": "S12_KDH",
        "identity": "KDH",
        "window_number": 0,
        "window_start_s": 10.0,
        "window_end_s": 42.0,
        "reference_start_sample": 2500,
        "reference_end_sample": 10500,
        "reference_window_start_biopac_s": 10.0,
        "reference_window_end_biopac_s": 42.0,
        "radar_window_start_relative_s": 8.0,
        "radar_window_end_relative_s": 40.0,
        "reference_mapping_available": True,
        "sync_authorized": False,
        "sync_confidence": 0.5,
        "alignment_scientific_eligible": False,
        "acquisition_phase": "phase1",
        "acquisition_phase_name": "baseline",
        "acquisition_phase_status": "auto",
        "acquisition_phase_confidence": 0.95,
        "phase_overlap_fraction": 1.0,
        "transition_window": False,
        "eligible_for_stage_metrics": False,
        "phase7_assignment": None,
        "acquisition_batch": "v3",
    }
    for column in _BUILD._V3_TARGET_DERIVED_METADATA_COLUMNS:
        metadata_values[column] = False if column in {
            "reference_valid",
            "radar_observable",
            "classical_acceptable_within_2bpm",
        } else np.nan
    metadata = pd.DataFrame([metadata_values])
    mapping = TimeMapping(mode="constant", offset_s=2.0)
    mapping_document = mapping.to_dict()
    protocol = {"session_id": "S12_KDH", "stages": []}
    receipt = {
        "session_id": "S12_KDH",
        "result": {"mapping": mapping_document},
        "content_sha256": "",
    }
    receipt["content_sha256"] = _BUILD._canonical_content_sha256(receipt)
    upstream_manifest = {
        "schema_version": _BUILD.ACQUISITION_SCHEMA_V3,
        "session_id": "S12_KDH",
        "raw_consumed_bytes_verified": True,
        "timing_adjudicated": True,
        "sync_raw_replay_verified": False,
        "protocol_raw_replay_verified": False,
        "protocol": protocol,
        "synchronization": {
            "mapping": mapping_document,
            "receipt_content_sha256": receipt["content_sha256"],
        },
        "content_sha256": "",
    }
    upstream_manifest["content_sha256"] = _BUILD._canonical_content_sha256(
        upstream_manifest
    )
    contract = SimpleNamespace(
        manifest=upstream_manifest,
        authorized=False,
        scientific_eligible=False,
        stage_metric_eligible=False,
        receipt=receipt,
        receipt_content_sha256=receipt["content_sha256"],
        mapping=mapping,
        protocol=protocol,
        window_minimum_overlap_fraction=0.8,
        transition_guard_s=2.0,
        content_sha256=upstream_manifest["content_sha256"],
    )
    inventory = {
        name: {
            "path": filename,
            "sha256": "8" * 64,
            "bytes": 1,
            "shape": [],
            "dtype": dtype,
        }
        for name, filename, dtype in (
            ("maps", "maps.npy", "float16"),
            ("aux", "aux.npy", "float32"),
            ("metadata", "metadata.csv", "csv"),
            ("frequencies_hz", "frequencies_hz.npy", "float64"),
            ("radar_timing_valid_mask", "radar_timing_valid_mask.npy", "bool"),
            (
                "radar_timing_invalid_reason_mask",
                "radar_timing_invalid_reason_mask.npy",
                "uint8",
            ),
            (
                "feature_availability_mask",
                "feature_availability_mask.npy",
                "bool",
            ),
        )
    }
    return {
        "session_id": "S12_KDH",
        "contract": contract,
        "config_sha256": "5" * 64,
        "pipeline_sha256": "6" * 64,
        "source_fingerprint": "9" * 64,
        "maps": maps,
        "aux": aux,
        "frequencies_hz": frequencies,
        "timing_valid_mask": timing_valid,
        "timing_reason_mask": reasons,
        "feature_availability_mask": availability,
        "aux_feature_names": aux_names,
        "availability_feature_names": availability_names,
        "metadata": metadata,
        "file_inventory": inventory,
        "biopac_sample_rate_hz": 250.0,
    }


def test_v3_manifest_binds_features_availability_and_target_firewall() -> None:
    kwargs = _v3_manifest_fixture()
    manifest = _BUILD._build_v3_session_manifest(**kwargs)
    assert manifest["scientific_eligible"] is False
    assert manifest["feature_schema"]["maps"]["target_derived_inputs"] is False
    maps_schema = manifest["feature_schema"]["maps"]
    assert maps_schema["axes"] == [
        "window",
        "radar_view",
        "frequency",
        "range_feature",
    ]
    assert "branch_names" not in maps_schema
    assert len(maps_schema["range_feature_names"]) == 182
    assert (
        manifest["target_firewall"]["target_values_used_in_inference_features"]
        is False
    )
    assert "radar_observable" in manifest["target_firewall"][
        "forbidden_inference_feature_names"
    ]


def test_real_range_frequency_transform_has_truthful_3_by_73_by_182_layout() -> None:
    time_s = np.arange(320, dtype=np.float32) / np.float32(10.0)
    range_scale = np.linspace(0.5, 1.5, 182, dtype=np.float32)
    base = (
        np.sin(2.0 * np.pi * np.float32(0.25) * time_s)[:, None]
        * range_scale[None, :]
    ).astype(np.float32)
    raw_maps = np.stack(
        [
            _BUILD.range_frequency_features(
                base * np.float32(1.0 + 0.05 * radar_index),
                fs=10.0,
                band_hz=(0.08, 0.80),
                nfft=2048,
                range_pool=2,
            ).feature_map
            for radar_index in range(3)
        ]
    )
    assert raw_maps.shape == (3, 147, 182)
    pooled = _BUILD._pool_range_frequency_map(raw_maps)
    assert pooled.shape == (3, 73, 182)
    assert pooled.dtype == np.float16


def test_s02_real_v3_build_subject_publishes_truthful_map_geometry(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "default.yaml"
    config_bytes = config_path.read_bytes()
    config = _BUILD.yaml.safe_load(config_bytes)
    dataset = _BUILD.build_dataset_manifest(project_root / "HAI_EXPERIMENT")
    subject = next(
        item for item in dataset.subjects if item.subject_id == "S02_RJS"
    )
    acquisition = _BUILD.load_acquisition_reconstruction(
        project_root
        / "artifacts"
        / "acquisition"
        / "reconstruction_v3_20260831_raw_exact_timing_reason_diagnostic"
        / "manifest.json"
    )
    result = _BUILD.build_subject(
        subject,
        config,
        tmp_path / "cache",
        True,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        pipeline_sha256=_BUILD._sha256_files(_BUILD._pipeline_paths()),
        acquisition_contract=acquisition.sessions["S02_RJS"],
        acquisition_mode="diagnostic",
    )
    maps = np.load(
        tmp_path / "cache" / "S02_RJS" / "maps.npy", allow_pickle=False
    )
    assert result["status"] == "ok"
    assert maps.shape == (316, 3, 73, 182)
    maps_schema = result["feature_schema"]["maps"]
    assert maps_schema["shape"] == [316, 3, 73, 182]
    assert maps_schema["axes"] == [
        "window",
        "radar_view",
        "frequency",
        "range_feature",
    ]
    assert "branch_names" not in maps_schema
    assert len(maps_schema["range_feature_names"]) == 182


def test_s01_real_v3_build_subject_uses_radar_only_support_without_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "default.yaml"
    config_bytes = config_path.read_bytes()
    config = _BUILD.yaml.safe_load(config_bytes)
    dataset = _BUILD.build_dataset_manifest(project_root / "HAI_EXPERIMENT")
    subject = next(
        item for item in dataset.subjects if item.subject_id == "S01_CMS"
    )
    acquisition = _BUILD.load_acquisition_reconstruction(
        project_root
        / "artifacts"
        / "acquisition"
        / "reconstruction_v3_20260831_raw_exact_timing_reason_diagnostic"
        / "manifest.json"
    )
    contract = acquisition.sessions["S01_CMS"]
    assert contract.mapping is None

    def forbidden_biopac_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("unmapped radar-only path must not consume BIOPAC")

    monkeypatch.setattr(_BUILD, "load_biopac_mat", forbidden_biopac_load)
    cache_root = tmp_path / "cache"
    result = _BUILD.build_subject(
        subject,
        config,
        cache_root,
        True,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        pipeline_sha256=_BUILD._sha256_files(_BUILD._pipeline_paths()),
        acquisition_contract=contract,
        acquisition_mode="diagnostic",
    )
    metadata = pd.read_csv(
        cache_root / "S01_CMS" / "metadata.csv",
        keep_default_na=False,
        na_values=[""],
    )
    maps = np.load(cache_root / "S01_CMS" / "maps.npy", allow_pickle=False)

    assert result["status"] == "ok"
    assert result["reference_mapping_available"] is False
    assert result["reference_support_contract"] == (
        _BUILD._v3_reference_support_contract(False)
    )
    assert result["metadata_join_contract"]["mapping"] is None
    assert result["metadata_join_contract"]["biopac_sample_rate_hz"] is None
    assert result["upstream_session_contract"]["synchronization"]["mapping"] is None
    assert maps.shape[0] == len(metadata) > 0
    assert maps.shape[1:] == (3, 73, 182)
    assert not metadata["reference_mapping_available"].any()
    assert not metadata["reference_valid"].any()
    assert (metadata.loc[:, ["reference_start_sample", "reference_end_sample"]] == -1).all().all()
    assert metadata.loc[:, _BUILD._V3_REFERENCE_FLOAT_NAN_COLUMNS].isna().all().all()
    assert metadata.loc[:, _BUILD._V3_REFERENCE_NULL_STRING_COLUMNS].isna().all().all()
    radar_start = metadata["radar_window_start_relative_s"].to_numpy(dtype=float)
    radar_end = metadata["radar_window_end_relative_s"].to_numpy(dtype=float)
    assert np.allclose(radar_end - radar_start, 32.0, rtol=0.0, atol=1e-12)
    assert len(radar_start) < 2 or np.all(np.diff(radar_start) > 0)


@pytest.mark.parametrize("tamper", ("availability", "target"))
def test_v3_manifest_rejects_availability_or_target_authority_tamper(
    tamper: str,
) -> None:
    kwargs = _v3_manifest_fixture()
    if tamper == "availability":
        kwargs["feature_availability_mask"][0, 1] = True
        match = "explicit feature availability"
    else:
        kwargs["metadata"].loc[0, "reference_valid"] = True
        match = "reference_valid=false"
    with pytest.raises(ValueError, match=match):
        _BUILD._build_v3_session_manifest(**kwargs)


def test_v3_reconstruction_snapshot_rejects_symlink_and_hardlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b'{"schema_version":"snn_rr.acquisition_reconstruction.v3"}')
    payload, digest = _BUILD._stable_regular_file_payload(
        source, label="test reconstruction"
    )
    assert payload == source.read_bytes()
    assert digest == _sha256(source)

    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(source)
    with pytest.raises(OSError):
        _BUILD._stable_regular_file_payload(symlink, label="symlink")

    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(ValueError, match="exactly one hard link"):
        _BUILD._stable_regular_file_payload(source, label="hardlink")


def test_v3_reconstruction_snapshot_detects_same_inode_read_restore_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "reconstruction.json"
    original_payload = b'{"schema_version":"snn_rr.acquisition_reconstruction.v3"}'
    source.write_bytes(original_payload)
    target = source.resolve()
    original_read = _BUILD.os.read
    attacked = False

    def adversarial_read(descriptor: int, count: int) -> bytes:
        nonlocal attacked
        descriptor_path = Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
        if not attacked and descriptor_path == target:
            attacked = True
            changed = bytes([original_payload[0] ^ 1]) + original_payload[1:]
            with source.open("r+b", buffering=0) as stream:
                stream.write(changed)
                os.fsync(stream.fileno())
            consumed = original_read(descriptor, count)
            with source.open("r+b", buffering=0) as stream:
                stream.write(original_payload)
                stream.truncate(len(original_payload))
                os.fsync(stream.fileno())
            return consumed
        return original_read(descriptor, count)

    monkeypatch.setattr(_BUILD.os, "read", adversarial_read)
    with pytest.raises(RuntimeError, match="changed during exact-byte consumption"):
        _BUILD._stable_regular_file_payload(source, label="reconstruction")
    assert attacked
    assert source.read_bytes() == original_payload


@pytest.mark.parametrize("mutation", ("missing_s24", "extra_s31", "symlink_s24"))
def test_v3_dataset_catalogue_rejects_noncanonical_real_session_tree(
    tmp_path: Path,
    mutation: str,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for session_id in _BUILD.SESSION_IDENTITY:
        (dataset_root / session_id).mkdir()
    manifest = SimpleNamespace(
        subjects=tuple(
            SimpleNamespace(subject_id=session_id)
            for session_id in _BUILD.SESSION_IDENTITY
        )
    )
    evidence = _BUILD._validate_v3_dataset_catalogue(
        dataset_root, manifest=manifest
    )
    assert [item["session_id"] for item in evidence["session_entries"]] == list(
        _BUILD.SESSION_IDENTITY
    )

    s24 = dataset_root / "S24_KHJ"
    if mutation == "missing_s24":
        s24.rmdir()
    elif mutation == "extra_s31":
        (dataset_root / "S31_EXTRA").mkdir()
    else:
        s24.rmdir()
        s24.symlink_to(dataset_root / "S23_KDM", target_is_directory=True)
    with pytest.raises(ValueError, match="exact canonical 30|not a real directory"):
        _BUILD._validate_v3_dataset_catalogue(dataset_root, manifest=manifest)


def test_v3_attempt_claim_write_failure_becomes_terminal_private_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "cache_v3_claim_failure"
    acquisition = tmp_path / "acquisition" / "manifest.json"
    acquisition.parent.mkdir()
    acquisition.write_text("{}", encoding="utf-8")
    original_write = _BUILD.os.write
    failed = False

    def fail_first_write(descriptor: int, payload: object) -> int:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError(errno.EIO, "injected claim write failure")
        return original_write(descriptor, payload)

    monkeypatch.setattr(_BUILD.os, "write", fail_first_write)
    with pytest.raises(OSError, match="injected claim write failure"):
        _BUILD._start_v3_private_attempt(
            final, acquisition_manifest=acquisition
        )
    assert failed
    assert _BUILD._ACTIVE_V3_ATTEMPT is None
    claim = final.parent / f".{final.name}.attempt_claim.json"
    receipt = json.loads(claim.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "snn_rr.feature_cache_v3_failure.v1"
    assert receipt["terminal_state"] == "failed"
    assert receipt["error_type"] == "OSError"
    assert receipt["content_sha256"] == _BUILD._canonical_content_sha256(receipt)
    assert stat.S_IMODE(claim.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        _BUILD._start_v3_private_attempt(
            final, acquisition_manifest=acquisition
        )


def test_main_publishes_diagnostic_v3_root_for_exact_30_to_29_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisition = _diagnostic_v3_reconstruction()
    authority_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "acquisition_cohort_v1.yaml"
    )
    authority = load_acquisition_cohort_authority(authority_path)
    all_ids = list(authority.expected_session_ids)
    usable_ids = list(authority.expected_usable_session_ids)
    session_hashes = {
        session_id: hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        for session_id in all_ids
    }
    acquisition.manifest.update(
        {
            "content_sha256": "",
            "expected_session_ids_sha256": _BUILD._canonical_value_sha256(
                all_ids
            ),
            "expected_usable_session_ids_sha256": (
                _BUILD._canonical_value_sha256(usable_ids)
            ),
            "selected_session_ids_sha256": _BUILD._canonical_value_sha256(
                all_ids
            ),
            "dataset_session_count": 30,
            "dataset_usable_session_count": 29,
            "dataset_physical_identity_count": 18,
            "selected_session_count": 30,
            "session_count": 30,
            "raw_consumed_bytes_verified": True,
            "timing_adjudicated": True,
            "cohort_authority_sha256": authority.file_sha256,
            "cohort_authority_content_sha256": authority.content_sha256,
            "sessions": [
                {
                    "session_id": session_id,
                    "usable": session_id != "S24_KHJ",
                    "content_sha256": session_hashes[session_id],
                    "scientific_eligible": False,
                    "raw_consumed_bytes_verified": session_id != "S24_KHJ",
                    "timing_adjudicated": session_id != "S24_KHJ",
                    "sync_raw_replay_verified": False,
                    "protocol_raw_replay_verified": False,
                    "sync_authorized": False,
                }
                for session_id in all_ids
            ],
        }
    )
    acquisition.manifest["content_sha256"] = _BUILD._canonical_content_sha256(
        acquisition.manifest
    )
    acquisition.content_sha256 = acquisition.manifest["content_sha256"]
    acquisition.sessions = {
        session_id: SimpleNamespace(
            content_sha256=session_hashes[session_id],
            range_feature_eligible=False,
            range_track_path=None,
        )
        for session_id in usable_ids
    }

    reconstruction_path = tmp_path / "acquisition" / "acquisition_v3.json"
    reconstruction_path.parent.mkdir()
    reconstruction_path.write_text(
        json.dumps(acquisition.manifest, sort_keys=True), encoding="utf-8"
    )
    acquisition.manifest_path = reconstruction_path
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "root": "unused",
                    "cache_dir": "unused",
                    "model_hz": 10.0,
                    "window_seconds": 32.0,
                }
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "feature_cache_v3"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for session_id in all_ids:
        (dataset_root / session_id).mkdir()
    subjects = tuple(
        SimpleNamespace(
            subject_id=session_id,
            usable=session_id != "S24_KHJ",
        )
        for session_id in all_ids
    )
    dataset_manifest = SimpleNamespace(
        subjects=subjects,
        to_dict=lambda: {"session_ids": all_ids},
    )
    args = SimpleNamespace(
        config=config_path,
        dataset_root=dataset_root,
        cache_dir=output_root,
        acquisition_manifest=reconstruction_path,
        acquisition_mode="diagnostic",
        subjects=None,
        force=False,
    )

    def fake_build_subject(
        subject: SimpleNamespace,
        _config: dict[str, object],
        root: Path,
        _force: bool,
        *,
        config_sha256: str,
        pipeline_sha256: str,
        acquisition_contract: object | None,
        acquisition_mode: str | None,
    ) -> dict[str, object]:
        if not subject.usable:
            return {
                "session_id": subject.subject_id,
                "status": "skipped",
                "reason": "missing paired radar/BIOPAC",
            }
        assert acquisition_contract is not None
        assert acquisition_mode == "diagnostic"
        session_dir = root / subject.subject_id
        session_dir.mkdir(parents=True, exist_ok=True)
        document = {
            "config_sha256": config_sha256,
            "pipeline_sha256": pipeline_sha256,
            "content_sha256": session_hashes[subject.subject_id],
        }
        (session_dir / "manifest.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        return {
            "session_id": subject.subject_id,
            "status": "ok",
            "cached": False,
            "schema_version": _BUILD.FEATURE_CACHE_SESSION_SCHEMA_V3,
            "scientific_eligible": False,
            "upstream_session_content_sha256": session_hashes[
                subject.subject_id
            ],
            "inventory_sha256": hashlib.sha256(
                f"inventory:{subject.subject_id}".encode("utf-8")
            ).hexdigest(),
            "window_count": 1,
            "content_sha256": session_hashes[subject.subject_id],
        }

    monkeypatch.setattr(_BUILD, "parse_args", lambda: args)
    monkeypatch.setattr(
        _BUILD, "load_acquisition_reconstruction", lambda _path: acquisition
    )
    monkeypatch.setattr(
        _BUILD, "build_dataset_manifest", lambda _path: dataset_manifest
    )
    monkeypatch.setattr(_BUILD, "build_subject", fake_build_subject)
    monkeypatch.setattr(_BUILD, "validate_raw_input_bindings", lambda *_args: None)
    monkeypatch.setattr(
        _BUILD, "_acquisition_cached_manifest_is_current", lambda *_args, **_kwargs: True
    )

    assert _BUILD.main() == 0
    root = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert root["schema_version"] == _BUILD.FEATURE_CACHE_ROOT_SCHEMA_V3
    assert len(root["sessions"]) == 29
    assert [item["session_id"] for item in root["sessions"]] == usable_ids
    contract = root["acquisition_contract"]
    assert contract["schema_version"] == _BUILD.FEATURE_CACHE_ACQUISITION_SCHEMA_V3
    assert contract["mode"] == "diagnostic"
    assert contract["scientific_eligible"] is False
    assert contract["selection_scope"] == "full_cohort"
    assert contract["cache_usable_session_ids"] == usable_ids
    assert (output_root / "acquisition_reconstruction.json").read_bytes() == (
        reconstruction_path.read_bytes()
    )
    assert stat.S_IMODE(output_root.stat().st_mode) == 0o700
    for current_root, directory_names, file_names in os.walk(output_root):
        for name in directory_names:
            assert stat.S_IMODE((Path(current_root) / name).stat().st_mode) == 0o700
        for name in file_names:
            assert stat.S_IMODE((Path(current_root) / name).stat().st_mode) == 0o600
    assert not (output_root.parent / f".{output_root.name}.attempt_claim.json").exists()
    assert not list(output_root.parent.glob(f".{output_root.name}.staging.*"))
    published_root_bytes = (output_root / "manifest.json").read_bytes()
    args.force = True
    with pytest.raises(FileExistsError, match="cannot be overwritten or resumed"):
        _BUILD.main()
    assert (output_root / "manifest.json").read_bytes() == published_root_bytes


def test_v3_failed_attempt_preserves_private_failure_receipt_and_blocks_resume(
    tmp_path: Path,
) -> None:
    final = tmp_path / "cache_v3_failed"
    acquisition = tmp_path / "acquisition" / "manifest.json"
    acquisition.parent.mkdir()
    acquisition.write_text("{}", encoding="utf-8")
    staging = _BUILD._start_v3_private_attempt(
        final, acquisition_manifest=acquisition
    )
    try:
        assert _BUILD._ACTIVE_V3_ATTEMPT is not None
        _BUILD._ACTIVE_V3_ATTEMPT.update(
            {
                "config_sha256": "1" * 64,
                "pipeline_sha256": "2" * 64,
                "completed_session_ids": ["S01_CMS"],
                "current_session_id": "S02_RJS",
            }
        )
        session = staging / "S01_CMS"
        session.mkdir()
        (session / "partial.npy").write_bytes(b"partial-derived-data")
        _BUILD._record_v3_attempt_failure(RuntimeError("synthetic failure"))

        assert not final.exists()
        receipt_path = staging / "FAILURE_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["terminal_state"] == "failed"
        assert receipt["completed_session_ids"] == ["S01_CMS"]
        assert receipt["current_session_id"] == "S02_RJS"
        assert receipt["config_sha256"] == "1" * 64
        assert receipt["pipeline_sha256"] == "2" * 64
        assert receipt["error_type"] == "RuntimeError"
        assert receipt["content_sha256"] == _BUILD._canonical_content_sha256(
            receipt
        )
        assert stat.S_IMODE(staging.stat().st_mode) == 0o700
        assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
        assert stat.S_IMODE((session / "partial.npy").stat().st_mode) == 0o600
        claim = final.parent / f".{final.name}.attempt_claim.json"
        assert stat.S_IMODE(claim.stat().st_mode) == 0o600
        _BUILD._ACTIVE_V3_ATTEMPT = None
        with pytest.raises(FileExistsError):
            _BUILD._start_v3_private_attempt(
                final, acquisition_manifest=acquisition
            )
    finally:
        _BUILD._ACTIVE_V3_ATTEMPT = None


def test_v3_atomic_directory_publish_preserves_concurrent_destination(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".cache.staging.test"
    staging.mkdir(mode=0o700)
    (staging / "manifest.json").write_text("{}", encoding="utf-8")
    final = tmp_path / "cache"
    final.mkdir()
    sentinel = final / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="final output exists"):
        _BUILD._publish_v3_directory_noreplace(staging, final)
    assert staging.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_v3_attempt_rejects_dangling_output_symlink_before_side_effect(
    tmp_path: Path,
) -> None:
    final = tmp_path / "cache"
    final.symlink_to(tmp_path / "missing")
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _BUILD._start_v3_private_attempt(
            final, acquisition_manifest=acquisition
        )
    assert not (tmp_path / ".cache.attempt_claim.json").exists()


@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
def test_v3_staging_privacy_rejects_linked_payloads(
    tmp_path: Path,
    kind: str,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    outside = tmp_path / "outside.npy"
    outside.write_bytes(b"protected")
    linked = staging / "maps.npy"
    if kind == "symlink":
        linked.symlink_to(outside)
    else:
        os.link(outside, linked)
    with pytest.raises(RuntimeError, match="one link|linked/non-directory"):
        _BUILD._secure_and_fsync_v3_tree(staging)
    assert outside.read_bytes() == b"protected"
