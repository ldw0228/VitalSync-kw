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
    assert result["manifest"]["format_version"] == 2
    assert result["manifest"]["schema"] == _BUILD.SCHEMA_ID
    execution = result["manifest"]["inputs"]["execution_source_generation"]
    assert execution["guard_scope"] == "initialization_time_direct_entry_disk_only"
    assert execution["binds_actual_loader_compiled_bytes"] is False
    assert execution["complete_private_import_closure"] is False
    assert result["manifest"]["axis_risk_router_v8r5_compatible"] is True
    assert (
        result["manifest"]["ordered_feature_names_semantic_sha256"]
        == _BUILD.EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
    )
    feature_schema = json.loads(
        (args.output_dir / "feature_names.json").read_text(encoding="utf-8")
    )
    assert feature_schema["axis_risk_router_v8r5_compatible"] is True
    assert (
        feature_schema["ordered_feature_names_semantic_sha256"]
        == _BUILD.EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
    )
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


def test_input_replacement_during_build_refuses_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _write_fixture(tmp_path)
    original = _BUILD._output_bindings

    def mutate_after_arrays_are_built(stage: Path) -> dict[str, object]:
        result = original(stage)
        with np.load(args.proposer, allow_pickle=False) as archive:
            values = {name: np.asarray(archive[name]) for name in archive.files}
        values["topk_rr_bpm"] = np.asarray(values["topk_rr_bpm"]).copy()
        values["topk_rr_bpm"][0, 0] += 0.125
        np.savez_compressed(args.proposer, **values)
        return result

    monkeypatch.setattr(_BUILD, "_output_bindings", mutate_after_arrays_are_built)
    with pytest.raises(RuntimeError, match="input changed during build"):
        _BUILD.build(args)
    assert not args.output_dir.exists()


def test_same_inode_payload_mutation_and_restore_cannot_enter_consumed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _write_fixture(tmp_path / "baseline")
    attacked = _write_fixture(tmp_path / "attacked")
    _BUILD.build(baseline)

    source_path = attacked.rf_cache / "S01_A" / "maps.npy"
    source_bytes = source_path.read_bytes()
    source_inode = source_path.stat().st_ino
    original_resolve = _BUILD._resolve_session_joint_radar_mask
    attacked_once = False

    def mutate_original_while_private_mmap_is_consumed(
        rf_maps: np.ndarray,
        svd_spectra: np.ndarray,
        svd_attributes: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        nonlocal attacked_once
        if not attacked_once:
            attacked_once = True
            live = np.load(source_path, mmap_mode="r+", allow_pickle=False)
            live[...] = np.float16(123.0)
            live.flush()
            # The array handed to scientific code is a different, private COW
            # inode and therefore still contains the initially bound values.
            assert not np.all(np.asarray(rf_maps) == np.float16(123.0))
            source_path.write_bytes(source_bytes)
            assert source_path.stat().st_ino == source_inode
        return original_resolve(
            rf_maps, svd_spectra, svd_attributes, **kwargs
        )

    monkeypatch.setattr(
        _BUILD,
        "_resolve_session_joint_radar_mask",
        mutate_original_while_private_mmap_is_consumed,
    )
    _BUILD.build(attacked)
    assert attacked_once
    assert source_path.read_bytes() == source_bytes
    for filename in (
        "node_features.npy",
        "node_feature_availability.npy",
        "candidate_bpm.npy",
        "candidate_mask.npy",
        "candidate_source_mask.npy",
        "joint_radar_mask.npy",
    ):
        np.testing.assert_array_equal(
            np.load(baseline.output_dir / filename, allow_pickle=False),
            np.load(attacked.output_dir / filename, allow_pickle=False),
            err_msg=filename,
        )


def test_private_input_snapshot_is_cleaned_after_post_snapshot_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _write_fixture(tmp_path)

    def fail_after_snapshot(_path: Path) -> dict[str, int]:
        raise RuntimeError("synthetic post-snapshot failure")

    monkeypatch.setattr(_BUILD, "_fold_map", fail_after_snapshot)
    with pytest.raises(RuntimeError, match="synthetic post-snapshot failure"):
        _BUILD.build(args)
    assert not list(tmp_path.glob(".harmonic-input-snapshot.*"))
    assert not args.output_dir.exists()


def test_private_input_snapshot_is_cleaned_after_reuse_mismatch(
    tmp_path: Path,
) -> None:
    args = _write_fixture(tmp_path)
    _BUILD.build(args)
    args.batch_size += 1

    with pytest.raises(RuntimeError, match="different settings or inputs"):
        _BUILD.build(args)
    assert not list(tmp_path.glob(".harmonic-input-snapshot.*"))


def test_private_input_snapshot_cleans_temp_when_initialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _write_fixture(tmp_path)
    monkeypatch.setattr(
        _BUILD,
        "_copy_bound_regular_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("synthetic harmonic snapshot copy failure")
        ),
    )
    with pytest.raises(OSError, match="synthetic harmonic snapshot copy failure"):
        _BUILD.build(args)
    assert not list(tmp_path.glob(".harmonic-input-snapshot.*"))
    assert not args.output_dir.exists()


def test_stable_private_directory_collision_never_deletes_preexisting_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "collision"
    existing = tmp_path / f".harmonic-input-snapshot.{token}"
    existing.mkdir()
    monkeypatch.setattr(_BUILD.secrets, "token_hex", lambda _count: token)

    with pytest.raises(RuntimeError, match="cannot allocate"):
        _BUILD._StablePrivateDirectory.create(
            tmp_path,
            prefix=".harmonic-input-snapshot.",
        )

    assert existing.is_dir()


def test_private_input_snapshot_rejects_session_path_traversal_without_escape(
    tmp_path: Path,
) -> None:
    escaped = tmp_path / "escaped_harmonic_snapshot"
    with pytest.raises(RuntimeError, match="unsafe session_id"):
        _BUILD._materialize_bound_input_snapshot(
            rf_cache=tmp_path / "rf",
            svd_cache=tmp_path / "svd",
            proposer_path=tmp_path / "proposer.npz",
            folds_path=tmp_path / "folds.json",
            sessions=("../../escaped_harmonic_snapshot",),
            input_bindings={},
            parent=tmp_path,
        )

    assert not escaped.exists()
    assert not list(tmp_path.glob(".harmonic-input-snapshot.*"))


def test_direct_entry_disk_binding_rejects_persistent_disk_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "builder.py"
    source.write_text("GENERATION = 'A'\n", encoding="utf-8")
    initial_binding = _BUILD._capture_direct_entry_disk_binding(source)

    source.write_text("GENERATION = 'B'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="initialization-time disk binding"):
        _BUILD._assert_direct_entry_disk_binding_current(initial_binding)

    assert source.read_text(encoding="utf-8") == "GENERATION = 'B'\n"


def test_publication_fsyncs_stage_then_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _write_fixture(tmp_path)
    observed: list[Path] = []
    original = _BUILD._fsync_directory

    def record(path: Path) -> None:
        observed.append(path.resolve())
        original(path)

    monkeypatch.setattr(_BUILD, "_fsync_directory", record)
    _BUILD.build(args)
    assert len(observed) == 2
    assert ".building." in observed[0].name
    assert observed[1] == args.output_dir.parent.resolve()


def test_publication_atomically_preserves_concurrent_empty_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _write_fixture(tmp_path)
    original = _BUILD._fsync_directory
    injected = False

    def create_concurrent_output_after_stage_fsync(path: Path) -> None:
        nonlocal injected
        original(path)
        if not injected and ".building." in path.name:
            args.output_dir.mkdir()
            injected = True

    monkeypatch.setattr(
        _BUILD,
        "_fsync_directory",
        create_concurrent_output_after_stage_fsync,
    )
    with pytest.raises(FileExistsError, match="appeared concurrently"):
        _BUILD.build(args)

    assert injected
    assert args.output_dir.is_dir()
    assert not list(args.output_dir.iterdir())
    assert not list(tmp_path.glob(f".{args.output_dir.name}.building.*"))
    assert not list(tmp_path.glob(".harmonic-input-snapshot.*"))


def test_build_parent_rebinding_fails_and_cleans_moved_snapshot_and_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _write_fixture(tmp_path)
    publish_parent = tmp_path / "publish-parent"
    publish_parent.mkdir()
    args.output_dir = publish_parent / "output"
    moved_parent = tmp_path / "moved-publish-parent"
    original = _BUILD._fsync_directory
    rebound = False

    def rebind_after_stage_fsync(path: Path) -> None:
        nonlocal rebound
        original(path)
        if not rebound and ".building." in path.name:
            publish_parent.rename(moved_parent)
            publish_parent.mkdir()
            rebound = True

    monkeypatch.setattr(_BUILD, "_fsync_directory", rebind_after_stage_fsync)
    with pytest.raises(RuntimeError, match="output parent changed"):
        _BUILD.build(args)

    assert rebound
    assert not args.output_dir.exists()
    assert not list(moved_parent.glob(".harmonic-input-snapshot.*"))
    assert not list(moved_parent.glob(f".{args.output_dir.name}.building.*"))
    assert not list(publish_parent.iterdir())


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
    availability = np.load(
        output / "node_feature_availability.npy", allow_pickle=False
    )
    features = np.load(output / "node_features.npy", allow_pickle=False)
    names = json.loads(
        (output / "feature_names.json").read_text(encoding="utf-8")
    )["node_feature_names"]
    candidate_40 = int(np.flatnonzero(np.isclose(bpm[0], 40.0))[0])
    assert rf_count[0, candidate_40, -1] == 0
    assert svd_count[0, candidate_40, -1] == 0
    assert availability.dtype == np.bool_
    assert availability.shape == features.shape
    assert np.count_nonzero(features[~availability]) == 0
    assert not availability[~candidate_mask].any()
    # The source grid ends at 24 bpm.  Exact persisted support therefore
    # rejects 40 bpm at x1 but admits its 20 bpm x1/2 harmonic.
    assert not availability[
        0, candidate_40, names.index("rf_radar1_r1_raw_power_mean")
    ]
    assert availability[
        0, candidate_40, names.index("rf_radar1_r1_2_raw_power_mean")
    ]
    iq_columns = [
        index for index, name in enumerate(names)
        if "_candidate_iq_phase_power_" in name
    ]
    assert iq_columns and not availability[..., iq_columns].any()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_lineage_sha256"]
    assert manifest["evidence_policy"]["out_of_band_policy"].startswith("exact_zero")
    assert manifest["classification"] == "retrospective_legacy_unverified_proposer"
    assert manifest["format_version"] == 2
    assert manifest["schema"] == _BUILD.SCHEMA_ID
    assert manifest["axis_risk_router_v8r5_compatible"] is False
    assert manifest["scientific_eligible"] is False
    assert manifest["trainable"] is False
    assert manifest["node_feature_availability_shape"] == list(features.shape)
    assert manifest["node_feature_availability_dtype"] == "bool"
    assert manifest["outputs"]["node_feature_availability"]["sha256"] == (
        _BUILD.sha256_file(output / "node_feature_availability.npy")
    )
    assert _BUILD.build(args)["status"] == "reused"


def test_target_and_qc_perturbation_cannot_change_forward_features(tmp_path: Path) -> None:
    first = _write_fixture(tmp_path / "first", target_delta=0.0)
    second = _write_fixture(tmp_path / "second", target_delta=1000.0)
    _BUILD.build(first)
    _BUILD.build(second)
    for filename in (
        "node_features.npy",
        "node_feature_availability.npy",
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


def test_reuse_rejects_self_rehashed_manifest_with_missing_output_binding(
    tmp_path: Path,
) -> None:
    args = _write_fixture(tmp_path)
    _BUILD.build(args)
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["outputs"]["node_feature_availability"]
    manifest.pop("content_sha256")
    manifest["content_sha256"] = _BUILD._canonical_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="binding inventory"):
        _BUILD.build(args)


def test_reuse_rejects_pre_availability_v1_schema_even_if_rehashed(
    tmp_path: Path,
) -> None:
    args = _write_fixture(tmp_path)
    _BUILD.build(args)
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 1
    manifest["schema"] = "snn_rr.harmonic_candidate_cache.v1"
    manifest.pop("content_sha256")
    manifest["content_sha256"] = _BUILD._canonical_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="partial/incompatible"):
        _BUILD.build(args)


def test_layout_source_binding_drift_invalidates_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _write_fixture(tmp_path)
    result = _BUILD.build(args)
    layout_binding = result["manifest"]["inputs"]["source"][
        "harmonic_feature_layout_v3r1"
    ]
    assert layout_binding["sha256"] == _BUILD.sha256_file(
        _BUILD.PROJECT_ROOT / "src/snn_rr/harmonic_feature_layout_v3r1.py"
    )

    original = _BUILD.collect_input_bindings

    def drift_layout(*call_args, **call_kwargs):
        bindings = original(*call_args, **call_kwargs)
        bindings["source"]["harmonic_feature_layout_v3r1"] = dict(
            bindings["source"]["harmonic_feature_layout_v3r1"]
        )
        bindings["source"]["harmonic_feature_layout_v3r1"]["sha256"] = "0" * 64
        return bindings

    monkeypatch.setattr(_BUILD, "collect_input_bindings", drift_layout)
    with pytest.raises(RuntimeError, match="different settings or inputs"):
        _BUILD.build(args)


def test_legacy_joint_mask_requires_both_raw_rf_and_svd_evidence(
    tmp_path: Path,
) -> None:
    args = _write_fixture(tmp_path)
    spectra_path = args.svd_cache / "S01_A" / "spectra.npy"
    spectra = np.load(spectra_path, allow_pickle=False)
    spectra[:, 0] = 0.0
    np.save(spectra_path, spectra, allow_pickle=False)

    _BUILD.build(args)
    joint = np.load(args.output_dir / "joint_radar_mask.npy", allow_pickle=False)
    sources = np.load(
        args.output_dir / "candidate_source_mask.npy", allow_pickle=False
    )
    assert not joint[0, 0]
    assert joint[0, 1:].all()
    assert not sources[0, :, int(_BUILD.CandidateSource.RADAR_PEAK_1)].any()


def test_gap_and_cross_radar_consensus_use_exact_structural_support(
    tmp_path: Path,
) -> None:
    args = _write_fixture(tmp_path)
    for session_id in ("S01_A", "S02_B"):
        maps_path = args.rf_cache / session_id / "maps.npy"
        maps = np.load(maps_path, allow_pickle=False)
        maps[:, 1:] = 0.0
        np.save(maps_path, maps, allow_pickle=False)
        spectra_path = args.svd_cache / session_id / "spectra.npy"
        spectra = np.load(spectra_path, allow_pickle=False)
        spectra[:, 1:] = 0.0
        np.save(spectra_path, spectra, allow_pickle=False)

    _BUILD.build(args)
    features = np.load(args.output_dir / "node_features.npy", allow_pickle=False)
    availability = np.load(
        args.output_dir / "node_feature_availability.npy", allow_pickle=False
    )
    candidate = np.load(
        args.output_dir / "candidate_mask.npy", allow_pickle=False
    )
    bpm = np.load(args.output_dir / "candidate_bpm.npy", allow_pickle=False)
    names = json.loads(
        (args.output_dir / "feature_names.json").read_text(encoding="utf-8")
    )["node_feature_names"]

    previous = availability[..., names.index("previous_candidate_gap_bpm")]
    next_gap = availability[..., names.index("next_candidate_gap_bpm")]
    expected_previous = np.zeros_like(candidate)
    expected_previous[:, 1:] = candidate[:, 1:] & candidate[:, :-1]
    expected_next = np.zeros_like(candidate)
    expected_next[:, :-1] = candidate[:, :-1] & candidate[:, 1:]
    np.testing.assert_array_equal(previous, expected_previous)
    np.testing.assert_array_equal(next_gap, expected_next)

    candidate_12 = int(np.flatnonzero(np.isclose(bpm[0], 12.0))[0])
    mean_column = names.index("rf_radar1_r1_raw_power_mean")
    consensus_column = names.index(
        "rf_radar1_r1_raw_power_cross_radar_consensus"
    )
    assert availability[0, candidate_12, mean_column]
    assert not availability[0, candidate_12, consensus_column]
    assert features[0, candidate_12, consensus_column] == 0.0


def test_missing_feature_batch_cannot_publish_a_partially_zero_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _write_fixture(tmp_path)
    original = _BUILD.iter_compact_node_feature_batches

    def omit_the_only_batch(*call_args: object, **call_kwargs: object):
        for _batch in original(*call_args, **call_kwargs):
            continue
        if False:  # pragma: no cover - keeps this function a generator
            yield None

    monkeypatch.setattr(
        _BUILD, "iter_compact_node_feature_batches", omit_the_only_batch
    )
    with pytest.raises(RuntimeError, match="exactly cover session rows"):
        _BUILD.build(args)
    assert not args.output_dir.exists()


def test_root_manifest_replacement_during_parse_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _write_fixture(tmp_path)
    original = _BUILD._validate_root_manifests

    def mutate_after_parse(rf_cache: Path, svd_cache: Path):
        result = original(rf_cache, svd_cache)
        path = svd_cache / "manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["components"] = 0
        path.write_text(json.dumps(document), encoding="utf-8")
        return result

    monkeypatch.setattr(_BUILD, "_validate_root_manifests", mutate_after_parse)
    with pytest.raises(RuntimeError, match="root manifest changed"):
        _BUILD.build(args)
    assert not args.output_dir.exists()


def test_root_manifest_replacement_before_first_full_binding_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _write_fixture(tmp_path)
    original = _BUILD.collect_input_bindings
    first = True

    def mutate_before_binding(*call_args: object, **call_kwargs: object):
        nonlocal first
        if first:
            first = False
            path = args.svd_cache / "manifest.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["components"] = 0
            path.write_text(json.dumps(document), encoding="utf-8")
        return original(*call_args, **call_kwargs)

    monkeypatch.setattr(_BUILD, "collect_input_bindings", mutate_before_binding)
    with pytest.raises(RuntimeError, match="before full input binding"):
        _BUILD.build(args)
    assert not args.output_dir.exists()


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
