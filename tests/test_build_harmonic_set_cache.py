from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_harmonic_set_cache.py"
_SPEC = importlib.util.spec_from_file_location("build_harmonic_set_cache", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_BUILD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BUILD
_SPEC.loader.exec_module(_BUILD)


def _metadata(session_id: str, identity: str, cache_index: int, target_delta: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": [session_id],
            "session_number": [cache_index + 1],
            "identity": [identity],
            "protocol": ["rest"],
            "window_number": [0],
            "window_start_s": [float(cache_index * 4)],
            "window_end_s": [float(cache_index * 4 + 32)],
            "rr_bpm": [12.0 + target_delta],
            "reference_valid": [target_delta < 100],
            "reference_quality": [0.9 + target_delta],
            "classical_rr_bpm": [10.0],
            "classical_confidence": [0.8],
            "classical_error_bpm": [2.0 + target_delta],
            "radar_observable": [target_delta == 0],
            "radar_peak_1_bpm": [11.0],
            "radar_peak_2_bpm": [13.0],
            "radar_peak_3_bpm": [17.0],
            "radar_peak_spread_bpm": [2.0],
        }
    )


def _write_fixture(root: Path, *, target_delta: float = 0.0) -> argparse.Namespace:
    rf = root / "rf"
    svd = root / "svd"
    rf.mkdir(parents=True)
    svd.mkdir(parents=True)
    sessions = [("S01_A", "A"), ("S02_B", "B")]
    rf_root = {"sessions": [{"session_id": sid, "status": "ok"} for sid, _ in sessions]}
    (rf / "manifest.json").write_text(json.dumps(rf_root), encoding="utf-8")
    frequency = np.asarray([0.10, 0.20, 0.30, 0.40], dtype=np.float32)
    variant_names = list(_BUILD.VERIFIED_SVD_VARIANT_NAMES)
    for cache_index, (session_id, identity) in enumerate(sessions):
        rf_dir = rf / session_id
        svd_dir = svd / session_id
        rf_dir.mkdir()
        svd_dir.mkdir()
        metadata = _metadata(session_id, identity, cache_index, target_delta)
        metadata.to_csv(rf_dir / "metadata.csv", index=False)
        metadata.insert(0, "cache_index", cache_index)
        metadata.to_csv(svd_dir / "metadata.csv", index=False)
        maps = np.ones((1, 3, len(frequency), 182), dtype=np.float16)
        maps[..., 1::3] *= 2
        spectra = np.ones((1, 3, 6, 6, len(frequency)), dtype=np.float16)
        spectra[..., 2] *= 3
        attributes = np.zeros((1, 3, 6, 6, 5), dtype=np.float32)
        attributes[..., 0] = 0.6
        attributes[..., 1] = 0.7
        attributes[..., 2] = 0.8
        attributes[..., 3] = 0.2
        attributes[..., 4] = 0.2
        np.save(rf_dir / "maps.npy", maps, allow_pickle=False)
        np.save(rf_dir / "frequencies_hz.npy", frequency, allow_pickle=False)
        np.save(svd_dir / "spectra.npy", spectra, allow_pickle=False)
        np.save(svd_dir / "attributes.npy", attributes, allow_pickle=False)
        np.save(svd_dir / "frequencies_hz.npy", frequency, allow_pickle=False)
        (rf_dir / "manifest.json").write_text(
            json.dumps({"session_id": session_id}), encoding="utf-8"
        )
        (svd_dir / "manifest.json").write_text(
            json.dumps(
                {"session_id": session_id, "valid_only": False, "label_inputs": []}
            ),
            encoding="utf-8",
        )
    svd_root = {
        "valid_only": False,
        "label_inputs": [],
        "components": 6,
        "variant_names": variant_names,
        "canonical_manifest_sha256": _BUILD.sha256_file(rf / "manifest.json"),
        "sessions": [{"session_id": sid, "status": "ok"} for sid, _ in sessions],
    }
    (svd / "manifest.json").write_text(json.dumps(svd_root), encoding="utf-8")
    folds = root / "folds.json"
    folds.write_text(json.dumps({"identity_to_fold": {"A": 0, "B": 1}}), encoding="utf-8")
    proposer = root / "proposer.npz"
    np.savez_compressed(
        proposer,
        cache_index=np.asarray([0, 1], dtype=np.int64),
        fold=np.asarray([0, 1], dtype=np.int16),
        session_id=np.asarray(["S01_A", "S02_B"]),
        identity=np.asarray(["A", "B"]),
        protocol=np.asarray(["rest", "rest"]),
        window_number=np.asarray([0, 0], dtype=np.int32),
        window_start_s=np.asarray([0.0, 4.0]),
        window_end_s=np.asarray([32.0, 36.0]),
        topk_rr_bpm=np.asarray(
            [[10.0, 12.0, 18.0, 24.0, 40.0], [10.0, 12.0, 18.0, 24.0, 40.0]],
            dtype=np.float32,
        ),
        topk_probability=np.asarray(
            [[0.4, 0.25, 0.15, 0.12, 0.08], [0.4, 0.25, 0.15, 0.12, 0.08]],
            dtype=np.float32,
        ),
        # These fields must be irrelevant to feature construction.
        reference_rr_bpm=np.asarray([12.0 + target_delta] * 2),
        reference_valid=np.asarray([target_delta < 100] * 2),
    )
    return argparse.Namespace(
        rf_cache=rf,
        svd_cache=svd,
        proposer=proposer,
        fold_assignments=folds,
        output_dir=root / "output",
        merge_radius_bpm=0.1,
        batch_size=1,
    )


def _upgrade_fixture_for_i2(args: argparse.Namespace) -> argparse.Namespace:
    for session_id in ("S01_A", "S02_B"):
        directory = args.svd_cache / session_id
        spectra_path = directory / "spectra.npy"
        attributes_path = directory / "attributes.npy"
        spectra = np.load(spectra_path, allow_pickle=False)
        attributes = np.load(attributes_path, allow_pickle=False)
        extra_spectra = np.full((*spectra.shape[:3], 6, spectra.shape[-1]), 7.0, np.float16)
        extra_spectra[..., 2] = 31.0
        extra_attributes = np.repeat(attributes[:, :, :, :1, :], 6, axis=3)
        np.save(
            spectra_path,
            np.concatenate((spectra, extra_spectra), axis=3),
            allow_pickle=False,
        )
        np.save(
            attributes_path,
            np.concatenate((attributes, extra_attributes), axis=3),
            allow_pickle=False,
        )
    root_manifest_path = args.svd_cache / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    root_manifest["components"] = 12
    root_manifest_path.write_text(json.dumps(root_manifest), encoding="utf-8")

    with np.load(args.proposer, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    grid = np.asarray([6.0, 8.0, 10.0, 12.0, 14.0, 18.0, 24.0], np.float32)
    posterior = np.asarray(
        [[0.05, 0.05, 0.30, 0.20, 0.10, 0.20, 0.10], [0, 0, 0, 0, 0, 0, 0]],
        np.float32,
    )
    values.update(
        proposal_available=np.asarray([True, False]),
        posterior_probability=posterior,
        posterior_rr_grid_bpm=grid,
        rr_std=np.asarray([1.5, 0.0], np.float32),
        quality=np.asarray([0.8, 0.0], np.float32),
        alias_probability=np.asarray([0.2, 0.0], np.float32),
        spike_rate=np.asarray([0.15, 0.0], np.float32),
        radar_weights=np.asarray([[0.2, 0.3, 0.5], [0.0, 0.0, 0.0]], np.float32),
    )
    np.savez_compressed(args.proposer, **values)
    args.proposal_selection = "posterior-nms"
    args.posterior_nms_suppression_bpm = 1.25
    args.base_proposals = "expected-map"
    args.svd_components = 12
    args.proposer_features = True
    return args


def test_posterior_nms_has_stable_ties_separation_and_unavailable_rows() -> None:
    grid = np.asarray([10.0, 10.5, 12.0, 13.0, 15.0], np.float32)
    posterior = np.asarray(
        [[0.35, 0.35, 0.15, 0.10, 0.05], [0, 0, 0, 0, 0]], np.float32
    )
    rr, confidence, mask = _BUILD.posterior_nms_modes(
        posterior,
        grid,
        np.asarray([True, False]),
        top_k=3,
        suppression_bpm=1.25,
    )
    # Equal 10/10.5 peaks choose the lower grid bin; 10.5 and 13 are suppressed
    # around already selected 10 and 12 respectively.
    np.testing.assert_array_equal(rr[0, mask[0]], [10.0, 12.0, 15.0])
    np.testing.assert_allclose(confidence[0, mask[0]], [0.35, 0.15, 0.05])
    assert np.all(np.diff(rr[0, mask[0]]) > 1.25)
    assert not mask[1].any()
    assert not rr[1].any() and not confidence[1].any()


def test_i2_full_posterior_base_sources_features_svd12_and_outer_unavailable(
    tmp_path: Path,
) -> None:
    args = _upgrade_fixture_for_i2(_write_fixture(tmp_path))
    result = _BUILD.build(args)
    assert result["status"] == "built"
    mask = np.load(args.output_dir / "candidate_mask.npy", allow_pickle=False)
    source = np.load(args.output_dir / "candidate_source_mask.npy", allow_pickle=False)
    bpm = np.load(args.output_dir / "candidate_bpm.npy", allow_pickle=False)
    features = np.load(args.output_dir / "node_features.npy", allow_pickle=False)
    names = json.loads(
        (args.output_dir / "feature_names.json").read_text(encoding="utf-8")
    )["node_feature_names"]

    expected = float(
        np.dot(
            np.asarray([0.05, 0.05, 0.30, 0.20, 0.10, 0.20, 0.10]),
            np.asarray([6.0, 8.0, 10.0, 12.0, 14.0, 18.0, 24.0]),
        )
    )
    expected_index = int(
        np.flatnonzero(np.isclose(bpm[0], expected, atol=1.0e-5) & mask[0])[0]
    )
    map_index = int(np.flatnonzero(np.isclose(bpm[0], 10.0) & mask[0])[0])
    assert source[0, expected_index, int(_BUILD.CandidateSource.BASE)]
    # BASE MAP is admitted before NMS direct mode, then collects the DIRECT bit.
    assert source[0, map_index, int(_BUILD.CandidateSource.BASE)]
    assert source[0, map_index, int(_BUILD.CandidateSource.DIRECT_MODE)]

    # Row 1 models a sealed outer-test identity.  Classical nodes remain usable,
    # but no BASE/DIRECT bit or proposer-derived feature may enter the model.
    assert not source[1, :, int(_BUILD.CandidateSource.BASE)].any()
    assert not source[1, :, int(_BUILD.CandidateSource.DIRECT_MODE)].any()
    proposer_columns = [names.index(name) for name in _BUILD.PROPOSER_NODE_FEATURE_NAMES]
    assert not np.any(features[1, mask[1]][:, proposer_columns])

    available_column = names.index("proposer_available")
    rank_column = names.index("direct_mode_rank")
    entropy_column = names.index("posterior_entropy_normalized")
    log_std_column = names.index("proposer_log_rr_std_bpm")
    radar3_column = names.index("proposer_radar3_weight")
    direct_confidence_column = names.index("source_confidence_direct_mode")
    assert features[0, map_index, available_column] == 1.0
    assert features[0, map_index, rank_column] == 1.0
    assert 0.0 < features[0, map_index, entropy_column] < 1.0
    assert features[0, map_index, log_std_column] == pytest.approx(np.log(1.5))
    assert features[0, map_index, radar3_column] == pytest.approx(0.5)
    assert features[0, map_index, direct_confidence_column] == pytest.approx(0.30)
    assert result["manifest"]["settings"]["svd_components"] == 12
    assert result["manifest"]["settings"]["posterior_nms_suppression_bpm"] == 1.25
    assert _BUILD.build(args)["status"] == "reused"


def test_i2_posterior_tamper_and_nonfinite_available_value_fail_closed(
    tmp_path: Path,
) -> None:
    args = _upgrade_fixture_for_i2(_write_fixture(tmp_path))
    with np.load(args.proposer, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    values["posterior_probability"] = values["posterior_probability"].copy()
    values["posterior_probability"][0, 0] = np.nan
    np.savez_compressed(args.proposer, **values)
    with pytest.raises(RuntimeError, match="posterior probabilities"):
        _BUILD.build(args)
    assert not args.output_dir.exists()


def test_i2_full_posterior_features_are_independent_of_reference_and_qc(
    tmp_path: Path,
) -> None:
    first = _upgrade_fixture_for_i2(
        _write_fixture(tmp_path / "first", target_delta=0.0)
    )
    second = _upgrade_fixture_for_i2(
        _write_fixture(tmp_path / "second", target_delta=1000.0)
    )
    _BUILD.build(first)
    _BUILD.build(second)
    for filename in (
        "node_features.npy",
        "candidate_bpm.npy",
        "candidate_mask.npy",
        "candidate_confidence.npy",
        "candidate_source_mask.npy",
        "joint_radar_mask.npy",
    ):
        np.testing.assert_array_equal(
            np.load(first.output_dir / filename, allow_pickle=False),
            np.load(second.output_dir / filename, allow_pickle=False),
            err_msg=filename,
        )


def test_build_exact_cover_direct_source_oob_and_verified_reuse(tmp_path: Path) -> None:
    args = _write_fixture(tmp_path)
    result = _BUILD.build(args)
    assert result["status"] == "built"
    output = args.output_dir
    metadata = pd.read_csv(output / "metadata.csv")
    assert metadata["cache_index"].tolist() == [0, 1]
    assert metadata["fold"].tolist() == [0, 1]
    candidate_mask = np.load(output / "candidate_mask.npy", allow_pickle=False)
    source = np.load(output / "candidate_source_mask.npy", allow_pickle=False)
    bpm = np.load(output / "candidate_bpm.npy", allow_pickle=False)
    assert candidate_mask.shape == (2, 12)
    assert not source[..., int(_BUILD.CandidateSource.BASE)].any()
    for row in range(2):
        for proposal in (10.0, 12.0, 18.0, 24.0, 40.0):
            index = int(np.flatnonzero(np.isclose(bpm[row], proposal) & candidate_mask[row])[0])
            assert source[row, index, int(_BUILD.CandidateSource.DIRECT_MODE)]
    rf_count = np.load(output / "rf_support_count.npy", allow_pickle=False)
    svd_count = np.load(output / "svd_support_count.npy", allow_pickle=False)
    candidate_40 = int(np.flatnonzero(np.isclose(bpm[0], 40.0))[0])
    assert rf_count[0, candidate_40, -1] == 0
    assert svd_count[0, candidate_40, -1] == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_lineage_sha256"]
    assert manifest["evidence_policy"]["out_of_band_policy"].startswith("exact_zero")
    assert _BUILD.build(args)["status"] == "reused"


def test_target_and_qc_perturbation_cannot_change_forward_features(tmp_path: Path) -> None:
    first = _write_fixture(tmp_path / "first", target_delta=0.0)
    second = _write_fixture(tmp_path / "second", target_delta=1000.0)
    _BUILD.build(first)
    _BUILD.build(second)
    for filename in (
        "node_features.npy",
        "candidate_bpm.npy",
        "candidate_mask.npy",
        "candidate_confidence.npy",
        "candidate_source_mask.npy",
        "joint_radar_mask.npy",
        "rf_support_count.npy",
        "svd_support_count.npy",
    ):
        assert np.array_equal(
            np.load(first.output_dir / filename, allow_pickle=False),
            np.load(second.output_dir / filename, allow_pickle=False),
        ), filename
    features = np.load(first.output_dir / "node_features.npy", allow_pickle=False)
    names = json.loads(
        (first.output_dir / "feature_names.json").read_text(encoding="utf-8")
    )["node_feature_names"]
    phase = [index for index, name in enumerate(names) if "candidate_iq_phase" in name]
    assert phase and np.count_nonzero(features[..., phase]) == 0


def test_existing_output_fails_closed_after_input_or_output_tamper(tmp_path: Path) -> None:
    args = _write_fixture(tmp_path)
    _BUILD.build(args)
    feature_names = args.output_dir / "feature_names.json"
    original = feature_names.read_text(encoding="utf-8")
    feature_names.write_text(original.replace("node_feature_names", "node_feature_namEs", 1), encoding="utf-8")
    with pytest.raises(RuntimeError, match="output SHA-256 mismatch"):
        _BUILD.build(args)

    other = _write_fixture(tmp_path / "input", target_delta=0.0)
    _BUILD.build(other)
    with np.load(other.proposer, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["topk_probability"] = arrays["topk_probability"].copy()
    arrays["topk_probability"][0, 0] += 0.01
    np.savez_compressed(other.proposer, **arrays)
    with pytest.raises(RuntimeError, match="different settings or inputs"):
        _BUILD.build(other)


def test_semantic_tamper_fails_before_output_commit(tmp_path: Path) -> None:
    args = _write_fixture(tmp_path)
    svd_metadata = args.svd_cache / "S02_B" / "metadata.csv"
    frame = pd.read_csv(svd_metadata)
    frame.loc[0, "window_end_s"] += 1.0
    frame.to_csv(svd_metadata, index=False)
    with pytest.raises(RuntimeError, match="semantic field mismatch"):
        _BUILD.build(args)
    assert not args.output_dir.exists()


def test_phase_only_rf_radar_is_excluded_by_raw_only_policy(tmp_path: Path) -> None:
    args = _write_fixture(tmp_path)
    maps_path = args.rf_cache / "S01_A" / "maps.npy"
    maps = np.load(maps_path, allow_pickle=False)
    maps[0, 0, :, :91] = 0.0
    maps[0, 0, :, 91:] = 9.0
    np.save(maps_path, maps, allow_pickle=False)
    _BUILD.build(args)
    joint = np.load(args.output_dir / "joint_radar_mask.npy", allow_pickle=False)
    assert not joint[0, 0]
    assert joint[0, 1:].all()
