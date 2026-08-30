from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_svd_features.py"
_SPEC = importlib.util.spec_from_file_location("build_svd_features_provenance", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SVD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SVD
_SPEC.loader.exec_module(_SVD)


def _v2_root_contract(
    expected: list[str],
    *,
    cache_usable: list[str] | None = None,
    strict: bool = True,
    full: bool = True,
) -> dict[str, object]:
    cache_ids = list(expected if cache_usable is None else cache_usable)
    return {
        "schema_version": "snn_rr.feature_cache_acquisition.v2",
        "mode": "strict" if strict else "diagnostic",
        "selection_scope": "full_cohort" if cache_ids == expected else "diagnostic_subset",
        "reconstruction_full_cohort_complete": full,
        "full_cohort_complete": full and cache_ids == expected,
        "scientific_eligible": strict and full and cache_ids == expected,
        "expected_usable_session_ids": list(expected),
        "expected_usable_session_ids_sha256": _SVD._canonical_sha256(expected),
        "cache_usable_session_ids": cache_ids,
        "cache_usable_session_ids_sha256": _SVD._canonical_sha256(cache_ids),
    }


def _v2_results(session_ids: list[str]) -> list[dict[str, object]]:
    return [
        {
            "session_id": session_id,
            "status": "ok",
            "canonical_acquisition_binding": {
                "schema_version": "snn_rr.feature_cache_acquisition.v2",
                "scientific_eligible": True,
            },
        }
        for session_id in session_ids
    ]


def _task() -> dict[str, object]:
    return {
        "pipeline_sha256": "1" * 64,
        "canonical_root_manifest_sha256": "0" * 64,
        "canonical_source_fingerprint": "source",
        "canonical_session_manifest_sha256": "2" * 64,
        "canonical_acquisition_session_manifest_sha256": "3" * 64,
        "canonical_acquisition_binding": {
            "schema_version": "snn_rr.feature_cache_acquisition.v1",
            "acquisition_session_manifest_sha256": "3" * 64,
            "mapping_sha256": "4" * 64,
        },
        "acquisition_reconstruction_content_sha256": "6" * 64,
        "dataset_root": "/bound/private/dataset",
        "selected_rows_sha256": "5" * 64,
        "valid_only": False,
        "components": 12,
        "nfft": 4096,
        "n_iter": 2,
        "variant_names": ["raw", "centered"],
    }


def test_svd_pipeline_digest_includes_synchronization_mapping_source() -> None:
    assert (
        _SVD.SOURCE_ROOT / "snn_rr" / "synchronization.py"
        in _SVD._pipeline_paths()
    )


def test_svd_publication_barrier_rejects_pipeline_source_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    digest = _SVD._pipeline_sha256([source])
    _SVD._assert_pipeline_unchanged([source], digest)

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pipeline source changed"):
        _SVD._assert_pipeline_unchanged([source], digest)


def test_svd_full_cohort_scope_requires_exact_untargeted_expected_set() -> None:
    session_ids = ["S01_A", "S02_B"]
    contract = _v2_root_contract(session_ids)
    derived = _SVD._derive_output_selection_contract(
        expected_session_ids=session_ids,
        selected_session_ids=session_ids,
        results=_v2_results(session_ids),
        subjects_filter_applied=False,
        canonical_acquisition_contract=contract,
    )
    assert derived == {
        "expected_session_ids": session_ids,
        "expected_session_ids_sha256": _SVD._canonical_sha256(session_ids),
        "selected_session_ids": session_ids,
        "selected_session_ids_sha256": _SVD._canonical_sha256(session_ids),
        "subjects_filter_applied": False,
        "selection_scope": "full_cohort",
        "full_cohort_complete": True,
        "scientific_eligible": True,
    }


@pytest.mark.parametrize(
    ("selected", "filter_applied"),
    [
        (["S01_A"], True),
        (["S01_A", "S02_B"], True),
    ],
)
def test_svd_subject_selection_is_always_diagnostic(
    selected: list[str], filter_applied: bool
) -> None:
    expected = ["S01_A", "S02_B"]
    derived = _SVD._derive_output_selection_contract(
        expected_session_ids=expected,
        selected_session_ids=selected,
        results=_v2_results(selected),
        subjects_filter_applied=filter_applied,
        canonical_acquisition_contract=_v2_root_contract(expected),
    )
    assert derived["selection_scope"] == "diagnostic_subset"
    assert derived["full_cohort_complete"] is False
    assert derived["scientific_eligible"] is False


def test_svd_incomplete_result_cannot_claim_full_or_scientific() -> None:
    session_ids = ["S01_A", "S02_B"]
    results = _v2_results(session_ids)
    results[1] = {"session_id": "S02_B", "status": "skipped"}
    derived = _SVD._derive_output_selection_contract(
        expected_session_ids=session_ids,
        selected_session_ids=session_ids,
        results=results,
        subjects_filter_applied=False,
        canonical_acquisition_contract=_v2_root_contract(session_ids),
    )
    assert derived["selection_scope"] == "full_cohort"
    assert derived["full_cohort_complete"] is False
    assert derived["scientific_eligible"] is False


def test_svd_expected_ids_follow_v2_authority_and_reject_hash_tampering() -> None:
    expected = ["S01_A", "S02_B"]
    available = ["S01_A"]
    contract = _v2_root_contract(expected, cache_usable=available, full=False)
    assert _SVD._canonical_expected_session_ids(available, contract) == expected

    contract["expected_usable_session_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="expected-session hash mismatch"):
        _SVD._canonical_expected_session_ids(available, contract)


def test_svd_session_signature_binds_canonical_manifest_and_acquisition_hashes() -> None:
    baseline = _SVD._session_signature(_task())

    changed_manifest = _task()
    changed_manifest["canonical_session_manifest_sha256"] = "a" * 64
    assert _SVD._session_signature(changed_manifest) != baseline

    changed_mapping = _task()
    changed_mapping["canonical_acquisition_binding"] = dict(
        changed_mapping["canonical_acquisition_binding"]
    )
    changed_mapping["canonical_acquisition_binding"]["mapping_sha256"] = "b" * 64
    assert _SVD._session_signature(changed_mapping) != baseline

    changed_acquisition = _task()
    changed_acquisition["canonical_acquisition_session_manifest_sha256"] = "d" * 64
    assert _SVD._session_signature(changed_acquisition) != baseline

    changed_root = _task()
    changed_root["canonical_root_manifest_sha256"] = "e" * 64
    assert _SVD._session_signature(changed_root) != baseline

    changed_dataset = _task()
    changed_dataset["dataset_root"] = "/different/dataset"
    assert _SVD._session_signature(changed_dataset) != baseline


def test_svd_acquisition_binding_is_exact_detached_json() -> None:
    manifest = {
        "acquisition_contract": {
            "schema_version": "snn_rr.feature_cache_acquisition.v1",
            "mapping_sha256": "c" * 64,
        }
    }
    binding = _SVD._canonical_acquisition_binding(manifest)
    assert binding == manifest["acquisition_contract"]
    assert binding is not manifest["acquisition_contract"]
    assert _SVD._canonical_acquisition_binding({}) is None


def test_bound_canonical_file_rejects_payload_tampering(tmp_path: Path) -> None:
    array_path = tmp_path / "radar_timing_valid_mask.npy"
    original = np.ones((1, 3, 320), dtype=np.bool_)
    np.save(array_path, original, allow_pickle=False)
    manifest = {
        "file_inventory": {
            "radar_timing_valid_mask": {
                "path": array_path.name,
                "bytes": array_path.stat().st_size,
                "sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
                "shape": list(original.shape),
                "dtype": "bool",
            }
        }
    }
    assert _SVD._verify_bound_canonical_file(
        tmp_path,
        manifest,
        "radar_timing_valid_mask",
        array_path.name,
    ) == array_path.resolve()

    tampered = original.copy()
    tampered[0, 0, 0] = False
    np.save(array_path, tampered, allow_pickle=False)
    with pytest.raises(RuntimeError, match="differs from its inventory"):
        _SVD._verify_bound_canonical_file(
            tmp_path,
            manifest,
            "radar_timing_valid_mask",
            array_path.name,
        )


def test_svd_fast_path_requires_exact_v2_inventory_and_manifest_hash(
    tmp_path: Path,
) -> None:
    arrays = {
        "spectra": np.zeros((1, 3, 1, 1, 2), dtype=np.float16),
        "component_signals": np.zeros((1, 3, 1, 1, 320), dtype=np.float16),
        "attributes": np.zeros((1, 3, 1, 1, 5), dtype=np.float32),
        "frequencies_hz": np.asarray([0.1, 0.2], dtype=np.float32),
        "radar_timing_valid_mask": np.ones((1, 3, 320), dtype=np.bool_),
    }
    inventory: dict[str, object] = {}
    for key, array in arrays.items():
        path = tmp_path / f"{key}.npy"
        np.save(path, array, allow_pickle=False)
        inventory[key] = _SVD._array_inventory(path, array)
    metadata = pd.DataFrame({"session_id": ["S01_TEST"]})
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    inventory["metadata"] = _SVD._metadata_inventory(metadata_path, metadata)
    manifest = {
        "session_signature": "1" * 64,
        "file_inventory": inventory,
        "inventory_sha256": _SVD._canonical_sha256(inventory),
    }
    manifest["content_sha256"] = _SVD._canonical_sha256(manifest)
    assert _SVD._cached_svd_manifest_is_current(
        tmp_path, manifest, acquisition_v2=True
    )

    reduced = dict(manifest)
    reduced_inventory = dict(inventory)
    reduced_inventory.pop("radar_timing_valid_mask")
    reduced["file_inventory"] = reduced_inventory
    reduced["inventory_sha256"] = _SVD._canonical_sha256(reduced_inventory)
    content_payload = dict(reduced)
    content_payload.pop("content_sha256", None)
    reduced["content_sha256"] = _SVD._canonical_sha256(content_payload)
    assert not _SVD._cached_svd_manifest_is_current(
        tmp_path, reduced, acquisition_v2=True
    )
