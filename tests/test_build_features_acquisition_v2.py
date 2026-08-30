from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest


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
    assert (
        _BUILD.SOURCE_ROOT / "snn_rr" / "synchronization.py"
        in _BUILD._pipeline_paths()
    )


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
