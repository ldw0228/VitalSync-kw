from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_build_harmonic_set_cache import (
    _BUILD,
    _upgrade_fixture_for_i2,
    _write_fixture,
)


_ROOT_CONTRACT = {
    "schema_version": _BUILD.ACQUISITION_V2_SCHEMA,
    "mode": "diagnostic",
    "scientific_eligible": False,
    "full_cohort_complete": False,
    "subjects_filter_applied": False,
}
_SESSION_BINDING = {
    "schema_version": _BUILD.ACQUISITION_V2_SCHEMA,
    "scientific_eligible": False,
    "strict_cache_eligible": False,
}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _rewrite_bound_mask(
    directory: Path,
    mask: np.ndarray,
    *,
    kind: str,
) -> None:
    path = directory / "radar_timing_valid_mask.npy"
    np.save(path, np.asarray(mask, dtype=np.bool_), allow_pickle=False)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("content_sha256", None)
    binding = _BUILD._file_binding(path)
    inventory = {
        "radar_timing_valid_mask": {
            "path": path.name,
            "sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "shape": list(mask.shape),
            "dtype": "bool",
        }
    }
    manifest.update(
        {
            "file_inventory": inventory,
            "inventory_sha256": _BUILD._canonical_digest(inventory),
            "radar_timing_valid_mask_shape": list(mask.shape),
            "radar_timing_invalid_interval_count": int(
                mask.size - np.count_nonzero(mask)
            ),
            "radar_timing_mask_contract": (
                dict(_BUILD.RF_TIMING_MASK_CONTRACT)
                if kind == "rf"
                else dict(_BUILD.SVD_TIMING_MASK_CONTRACT)
            ),
        }
    )
    if kind == "rf":
        manifest["acquisition_contract"] = dict(_SESSION_BINDING)
    else:
        manifest["canonical_acquisition_binding"] = dict(_SESSION_BINDING)
    manifest["content_sha256"] = _BUILD._canonical_digest(manifest)
    _write_json(manifest_path, manifest)


def _acquisition_v2_fixture(tmp_path: Path) -> object:
    args = _write_fixture(tmp_path)
    masks = {
        "S01_A": np.ones((1, 3, 4), dtype=np.bool_),
        "S02_B": np.ones((1, 3, 4), dtype=np.bool_),
    }
    masks["S01_A"][0, 0, 2] = False
    for session_id, mask in masks.items():
        _rewrite_bound_mask(args.rf_cache / session_id, mask, kind="rf")
        _rewrite_bound_mask(args.svd_cache / session_id, mask, kind="svd")

    rf_root_path = args.rf_cache / "manifest.json"
    rf_root = json.loads(rf_root_path.read_text(encoding="utf-8"))
    rf_root["acquisition_contract"] = dict(_ROOT_CONTRACT)
    _write_json(rf_root_path, rf_root)

    svd_root_path = args.svd_cache / "manifest.json"
    svd_root = json.loads(svd_root_path.read_text(encoding="utf-8"))
    svd_root["canonical_acquisition_contract"] = dict(_ROOT_CONTRACT)
    svd_root["scientific_eligible"] = False
    svd_root["canonical_manifest_sha256"] = _BUILD.sha256_file(rf_root_path)
    _write_json(svd_root_path, svd_root)
    return args


def test_nonzero_payload_cannot_enable_false_acquisition_timing_view(
    tmp_path: Path,
) -> None:
    args = _acquisition_v2_fixture(tmp_path)
    result = _BUILD.build(args)

    joint = np.load(args.output_dir / "joint_radar_mask.npy", allow_pickle=False)
    sources = np.load(
        args.output_dir / "candidate_source_mask.npy", allow_pickle=False
    )
    features = np.load(args.output_dir / "node_features.npy", allow_pickle=False)
    availability = np.load(
        args.output_dir / "node_feature_availability.npy", allow_pickle=False
    )
    names = json.loads(
        (args.output_dir / "feature_names.json").read_text(encoding="utf-8")
    )["node_feature_names"]
    radar1 = [
        index
        for index, name in enumerate(names)
        if name.startswith(("rf_radar1_", "svd_radar1_"))
    ]
    radar2 = [
        index
        for index, name in enumerate(names)
        if name.startswith(("rf_radar2_", "svd_radar2_"))
    ]

    assert not joint[0, 0]
    assert joint[0, 1:].all()
    classical_sources = slice(
        int(_BUILD.CandidateSource.CLASSICAL_X1),
        int(_BUILD.CandidateSource.CLASSICAL_X4) + 1,
    )
    # No independent fused-classical timing proof exists.  One unavailable
    # contributing radar therefore revokes all four classical candidates.
    assert np.count_nonzero(sources[0, :, classical_sources]) == 0
    classical_columns = [
        index
        for index, name in enumerate(names)
        if name.startswith(("source_classical_", "source_confidence_classical_"))
    ]
    assert classical_columns
    assert not availability[0, :, classical_columns].any()
    assert np.count_nonzero(features[0, :, classical_columns]) == 0
    assert radar1 and np.count_nonzero(features[0, :, radar1]) == 0
    assert radar2 and np.count_nonzero(features[0, :, radar2]) > 0
    manifest = result["manifest"]
    assert manifest["classification"] == "acquisition_diagnostic"
    assert manifest["scientific_eligible"] is False
    assert manifest["trainable"] is False
    assert manifest["acquisition_v2"]["numeric_payload_cannot_enable_a_masked_view"]
    for side in ("rf", "svd"):
        declared = manifest["inputs"]["sessions"]["S01_A"][side][
            "radar_timing_valid_mask"
        ]
        actual = args.__dict__[f"{side}_cache"] / "S01_A" / "radar_timing_valid_mask.npy"
        assert declared["sha256"] == _BUILD.sha256_file(actual)


def test_all_false_timing_row_has_zero_classical_source_bits(tmp_path: Path) -> None:
    args = _upgrade_fixture_for_i2(_acquisition_v2_fixture(tmp_path))
    unavailable = np.zeros((1, 3, 4), dtype=np.bool_)
    _rewrite_bound_mask(args.rf_cache / "S01_A", unavailable, kind="rf")
    _rewrite_bound_mask(args.svd_cache / "S01_A", unavailable, kind="svd")

    _BUILD.build(args)
    joint = np.load(args.output_dir / "joint_radar_mask.npy", allow_pickle=False)
    sources = np.load(
        args.output_dir / "candidate_source_mask.npy", allow_pickle=False
    )
    features = np.load(args.output_dir / "node_features.npy", allow_pickle=False)
    availability = np.load(
        args.output_dir / "node_feature_availability.npy", allow_pickle=False
    )
    names = json.loads(
        (args.output_dir / "feature_names.json").read_text(encoding="utf-8")
    )["node_feature_names"]
    assert not joint[0].any()
    classical_sources = slice(
        int(_BUILD.CandidateSource.CLASSICAL_X1),
        int(_BUILD.CandidateSource.CLASSICAL_X4) + 1,
    )
    assert np.count_nonzero(sources[0, :, classical_sources]) == 0
    assert not sources[0, :, int(_BUILD.CandidateSource.BASE)].any()
    assert not sources[0, :, int(_BUILD.CandidateSource.DIRECT_MODE)].any()
    proposer_columns = [
        names.index(name) for name in _BUILD.PROPOSER_NODE_FEATURE_NAMES
    ]
    assert not availability[0, :, proposer_columns].any()
    assert np.count_nonzero(features[0, :, proposer_columns]) == 0


def test_unavailable_view_masks_radar_peak_source_and_proposer_descriptor(
    tmp_path: Path,
) -> None:
    args = _upgrade_fixture_for_i2(_acquisition_v2_fixture(tmp_path))
    result = _BUILD.build(args)
    assert result["status"] == "built"

    source = np.load(
        args.output_dir / "candidate_source_mask.npy", allow_pickle=False
    )
    features = np.load(args.output_dir / "node_features.npy", allow_pickle=False)
    availability = np.load(
        args.output_dir / "node_feature_availability.npy", allow_pickle=False
    )
    names = json.loads(
        (args.output_dir / "feature_names.json").read_text(encoding="utf-8")
    )["node_feature_names"]
    radar1_source = int(_BUILD.CandidateSource.RADAR_PEAK_1)
    radar2_source = int(_BUILD.CandidateSource.RADAR_PEAK_2)
    assert not source[0, :, radar1_source].any()
    assert source[0, :, radar2_source].any()
    assert not source[0, :, int(_BUILD.CandidateSource.BASE)].any()
    assert not source[0, :, int(_BUILD.CandidateSource.DIRECT_MODE)].any()

    radar1_weight = names.index("proposer_radar1_weight")
    radar2_weight = names.index("proposer_radar2_weight")
    radar1_source_confidence = names.index("source_confidence_radar_peak_1")
    radar2_source_confidence = names.index("source_confidence_radar_peak_2")
    candidate_confidence = names.index("candidate_confidence")
    assert np.count_nonzero(features[0, :, radar1_weight]) == 0
    assert np.count_nonzero(features[0, :, radar1_source_confidence]) == 0
    assert not availability[0, :, radar2_source_confidence].any()
    assert np.count_nonzero(features[0, :, radar2_source_confidence]) == 0
    assert not availability[0, :, candidate_confidence].any()
    assert np.count_nonzero(features[0, :, candidate_confidence]) == 0
    persisted_confidence = np.load(
        args.output_dir / "candidate_confidence.npy", allow_pickle=False
    )
    assert np.count_nonzero(persisted_confidence[0]) == 0
    # A per-radar proposer descriptor has no independent timing authority;
    # acquisition-v2 therefore revokes the entire proposer row unless all
    # three contributing radar views are available.
    assert np.count_nonzero(features[0, :, radar2_weight]) == 0
    proposer_source_columns = [
        names.index(name)
        for name in (
            "source_base",
            "source_direct_mode",
            "source_confidence_base",
            "source_confidence_direct_mode",
        )
    ]
    assert not availability[0, :, proposer_source_columns].any()
    assert np.count_nonzero(features[0, :, proposer_source_columns]) == 0


def test_finite_timing_cannot_admit_nonfinite_svd_view_as_candidate_source(
    tmp_path: Path,
) -> None:
    args = _acquisition_v2_fixture(tmp_path)
    # Use an otherwise fully valid timing mask for this row/view, then make
    # the actual selected SVD evidence non-finite.
    valid = np.ones((1, 3, 4), dtype=np.bool_)
    _rewrite_bound_mask(args.rf_cache / "S02_B", valid, kind="rf")
    _rewrite_bound_mask(args.svd_cache / "S02_B", valid, kind="svd")
    spectra_path = args.svd_cache / "S02_B" / "spectra.npy"
    spectra = np.load(spectra_path, allow_pickle=False)
    spectra[:, 1] = np.nan
    np.save(spectra_path, spectra, allow_pickle=False)

    _BUILD.build(args)
    joint = np.load(args.output_dir / "joint_radar_mask.npy", allow_pickle=False)
    sources = np.load(
        args.output_dir / "candidate_source_mask.npy", allow_pickle=False
    )
    assert not joint[1, 1]
    assert not sources[1, :, int(_BUILD.CandidateSource.RADAR_PEAK_2)].any()


def test_timing_mask_replacement_during_initial_binding_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _acquisition_v2_fixture(tmp_path)
    original = _BUILD.collect_input_bindings
    calls = 0

    def mutate_after_first_binding(*call_args: object, **call_kwargs: object):
        nonlocal calls
        result = original(*call_args, **call_kwargs)
        calls += 1
        if calls == 1:
            replacement = np.ones((1, 3, 4), dtype=np.bool_)
            _rewrite_bound_mask(
                args.rf_cache / "S01_A", replacement, kind="rf"
            )
            _rewrite_bound_mask(
                args.svd_cache / "S01_A", replacement, kind="svd"
            )
        return result

    monkeypatch.setattr(_BUILD, "collect_input_bindings", mutate_after_first_binding)
    with pytest.raises(RuntimeError, match="initial inputs loaded"):
        _BUILD.build(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("side", ["rf", "svd"])
def test_missing_acquisition_timing_mask_rejects_before_output_publication(
    tmp_path: Path, side: str
) -> None:
    args = _acquisition_v2_fixture(tmp_path)
    cache = args.rf_cache if side == "rf" else args.svd_cache
    (cache / "S02_B" / "radar_timing_valid_mask.npy").unlink()

    with pytest.raises(RuntimeError, match="radar timing mask file is missing"):
        _BUILD.build(args)
    assert not args.output_dir.exists()


def test_semantically_different_rf_svd_timing_masks_reject_before_output_publication(
    tmp_path: Path,
) -> None:
    args = _acquisition_v2_fixture(tmp_path)
    svd_mask = np.ones((1, 3, 4), dtype=np.bool_)
    svd_mask[0, 2, 1] = False
    _rewrite_bound_mask(args.svd_cache / "S02_B", svd_mask, kind="svd")

    with pytest.raises(RuntimeError, match="radar timing masks differ"):
        _BUILD.build(args)
    assert not args.output_dir.exists()


def test_upstream_strict_self_claim_cannot_make_unverified_proposer_trainable(
    tmp_path: Path,
) -> None:
    args = _acquisition_v2_fixture(tmp_path)
    for session_id in ("S01_A", "S02_B"):
        all_valid = np.ones((1, 3, 4), dtype=np.bool_)
        _rewrite_bound_mask(args.rf_cache / session_id, all_valid, kind="rf")
        _rewrite_bound_mask(args.svd_cache / session_id, all_valid, kind="svd")
    strict = {
        **_ROOT_CONTRACT,
        "mode": "strict",
        "scientific_eligible": True,
        "full_cohort_complete": True,
    }
    rf_path = args.rf_cache / "manifest.json"
    rf = json.loads(rf_path.read_text(encoding="utf-8"))
    rf["acquisition_contract"] = strict
    _write_json(rf_path, rf)
    svd_path = args.svd_cache / "manifest.json"
    svd = json.loads(svd_path.read_text(encoding="utf-8"))
    svd["canonical_acquisition_contract"] = strict
    svd["scientific_eligible"] = True
    svd["canonical_manifest_sha256"] = _BUILD.sha256_file(rf_path)
    _write_json(svd_path, svd)

    result = _BUILD.build(args)
    manifest = result["manifest"]
    assert manifest["classification"] == "acquisition_diagnostic"
    assert manifest["scientific_eligible"] is False
    assert manifest["trainable"] is False
    assert manifest["acquisition_v2"]["scientific_promotion_authority"] == "absent"
