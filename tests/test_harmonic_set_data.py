import numpy as np
import pandas as pd
import pytest

from snn_rr.harmonic_set_data import (
    CANDIDATE_SOURCE_NAMES,
    HARMONIC_RATIOS,
    CandidateSource,
    assert_exact_semantic_row_binding,
    build_candidate_bank,
    build_compact_node_features,
    candidate_bank_from_metadata,
    iter_compact_node_feature_batches,
    reshape_rf_branches,
    resolve_joint_radar_mask,
    sample_rf_harmonic_support,
    sample_svd_harmonic_support,
    select_forward_metadata,
    semantic_row_binding_sha256,
    triangular_sample_native_grid,
)


def _one_candidate(rows: int = 1):
    return build_candidate_bank(
        proposal_bpm=np.full((rows, 1), 12.0, dtype=np.float32),
        proposal_confidence=np.full((rows, 1), 0.9, dtype=np.float32),
        classical_rr_bpm=np.full(rows, np.nan),
        classical_confidence=np.zeros(rows),
        radar_peaks_bpm=np.full((rows, 3), np.nan),
    )


def _synthetic_evidence(rows: int = 1, radars: int = 3):
    frequencies = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    maps = np.full((rows, radars, 3, 182), 0.01, dtype=np.float32)
    maps[:, :, 1, 17] = 3.0
    maps[:, :, 1, 91 + 17] = 9.0
    spectra = np.zeros((rows, radars, 10, 8, 3), dtype=np.float32)
    for variant in range(10):
        spectra[:, :, variant, :, 1] = float(variant + 1)
    attributes = np.zeros((rows, radars, 10, 8, 5), dtype=np.float32)
    attributes[..., 0] = 0.5
    attributes[..., 1] = 0.8
    attributes[..., 2] = 0.9
    attributes[..., 3] = 0.2
    attributes[..., 4] = 0.2
    return frequencies, maps, spectra, attributes


def test_candidate_bank_has_stable_priority_merge_and_sorted_order() -> None:
    bank = build_candidate_bank(
        proposal_bpm=np.asarray([[10.0, 20.0]]),
        proposal_confidence=np.asarray([[0.9, 0.8]]),
        classical_rr_bpm=np.asarray([8.0]),
        classical_confidence=np.asarray([0.6]),
        radar_peaks_bpm=np.asarray([[10.4, 30.0, 50.0]]),
    )
    np.testing.assert_array_equal(
        bank.bpm[0, bank.mask[0]], np.asarray([8, 10, 16, 20, 24, 30, 32])
    )
    merged = int(np.flatnonzero(bank.bpm[0] == 10.0)[0])
    assert bank.primary_source[0, merged] == int(CandidateSource.BASE)
    assert bank.source_mask[0, merged, CandidateSource.BASE]
    assert bank.source_mask[0, merged, CandidateSource.RADAR_PEAK_1]
    assert bank.confidence[0, merged] == pytest.approx(0.9)
    assert not np.any(bank.bpm[0] == 50.0)
    assert np.all(bank.primary_source[0, ~bank.mask[0]] == -1)
    assert bank.source_mask.shape[-1] == len(CANDIDATE_SOURCE_NAMES)


def test_exact_only_dedup_is_configurable_and_radius_is_manifest_bound() -> None:
    arguments = {
        "proposal_bpm": np.asarray([[10.0, 10.4, 10.0]]),
        "proposal_confidence": np.asarray([[0.9, 0.8, 0.7]]),
        "classical_rr_bpm": np.asarray([np.nan]),
        "classical_confidence": np.asarray([0.0]),
        "radar_peaks_bpm": np.full((1, 3), np.nan),
    }
    merged = build_candidate_bank(**arguments)
    exact = build_candidate_bank(**arguments, merge_radius_bpm=0.0)
    np.testing.assert_array_equal(
        merged.bpm[0, merged.mask[0]], np.asarray([10.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        exact.bpm[0, exact.mask[0]], np.asarray([10.0, 10.4], dtype=np.float32)
    )
    assert merged.manifest()["merge_radius_bpm"] == 0.5
    assert exact.manifest()["merge_radius_bpm"] == 0.0
    assert merged.manifest()["content_sha256"] != exact.manifest()["content_sha256"]


def test_native_triangular_sampling_recovers_synthetic_line() -> None:
    frequencies = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    line = np.asarray([0.0, 10.0, 20.0], dtype=np.float32)
    sampled, mask = triangular_sample_native_grid(
        line, frequencies, np.asarray([6.0, 9.0, 12.0, 15.0, 18.0])
    )
    np.testing.assert_allclose(sampled, [0.0, 5.0, 10.0, 15.0, 20.0])
    assert mask.tolist() == [True, True, True, True, True]


def test_out_of_band_sampling_is_zero_and_never_edge_clamped() -> None:
    frequencies = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    line = np.asarray([7.0, 8.0, 9.0], dtype=np.float32)
    sampled, mask = triangular_sample_native_grid(
        line, frequencies, np.asarray([3.0, 6.0, 18.0, 24.0])
    )
    np.testing.assert_array_equal(mask, [False, True, True, False])
    np.testing.assert_allclose(sampled, [0.0, 7.0, 9.0, 0.0])


def test_rf_branch_reshape_and_sampling_preserve_raw_91_bin_indices() -> None:
    frequencies, maps, _, _ = _synthetic_evidence()
    branches = reshape_rf_branches(maps)
    assert branches.shape == (1, 3, 3, 2, 91)
    assert branches[0, 0, 1, 0, 17] == 3.0
    assert branches[0, 0, 1, 1, 17] == 9.0

    support = sample_rf_harmonic_support(
        maps, frequencies, _one_candidate(), ratios=(1.0,)
    )
    assert support.values.shape == (1, 12, 3, 1, 2, 91)
    assert support.mask[0, 0].all()
    assert support.values[0, 0, 0, 0, 0, 17] == 3.0
    assert support.values[0, 0, 0, 0, 1, 17] == 9.0


def test_svd_sampler_uses_only_verified_variants_and_preserves_components() -> None:
    frequencies, _, spectra, attributes = _synthetic_evidence()
    # Unverified split-layout variants carry an unmistakable sentinel that
    # must never reach the result.
    spectra[:, :, 6:, :, 1] = 999.0
    support = sample_svd_harmonic_support(
        spectra, attributes, frequencies, _one_candidate(), ratios=(1.0,)
    )
    assert support.values.shape == (1, 12, 3, 1, 6, 6)
    np.testing.assert_allclose(
        support.values[0, 0, 0, 0, :, 0], np.arange(1.0, 7.0)
    )
    assert np.max(support.values) < 999.0
    assert support.reliability.shape == (1, 3, 6, 6)
    assert np.all(support.reliability > 0)
    assert support.component_mask.all()


def test_svd_twelve_components_change_evidence_without_changing_compact_width() -> None:
    frequencies = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    maps = np.ones((1, 3, 3, 182), dtype=np.float32)
    spectra = np.ones((1, 3, 6, 12, 3), dtype=np.float32)
    # Components 7..12 carry distinct candidate-frequency evidence.  A six
    # component implementation cannot observe this perturbation.
    spectra[:, :, :, 6:, 1] = 50.0
    attributes = np.zeros((1, 3, 6, 12, 5), dtype=np.float32)
    attributes[..., 0:3] = 0.8
    attributes[..., 3] = 0.1
    attributes[..., 4] = 0.2
    candidates = _one_candidate()
    rf = sample_rf_harmonic_support(
        maps, frequencies, candidates, ratios=(1.0,)
    )
    svd6 = sample_svd_harmonic_support(
        spectra,
        attributes,
        frequencies,
        candidates,
        ratios=(1.0,),
        components=6,
    )
    svd12 = sample_svd_harmonic_support(
        spectra,
        attributes,
        frequencies,
        candidates,
        ratios=(1.0,),
        components=12,
    )
    assert svd6.values.shape[-1] == 6
    assert svd12.values.shape[-1] == 12
    np.testing.assert_array_equal(svd6.mask, svd12.mask)
    nodes6 = build_compact_node_features(candidates, rf, svd6)
    nodes12 = build_compact_node_features(candidates, rf, svd12)
    assert nodes6.feature_names == nodes12.feature_names
    assert nodes6.features.shape == nodes12.features.shape
    weighted_name = "svd_radar1_r1_reliability_weighted_mean"
    column = nodes6.feature_names.index(weighted_name)
    assert nodes12.features[0, 0, column] > nodes6.features[0, 0, column]


def test_candidate_bank_retains_per_source_confidence_after_merge() -> None:
    bank = build_candidate_bank(
        proposal_bpm=np.asarray([[10.0, 10.2]]),
        proposal_confidence=np.asarray([[0.9, 0.4]]),
        proposal_source=np.asarray(
            [[CandidateSource.BASE, CandidateSource.DIRECT_MODE]], dtype=np.int16
        ),
        classical_rr_bpm=np.asarray([10.1]),
        classical_confidence=np.asarray([0.6]),
        radar_peaks_bpm=np.full((1, 3), np.nan),
    )
    index = int(np.flatnonzero(bank.mask[0])[0])
    assert bank.source_confidence[0, index, CandidateSource.BASE] == pytest.approx(0.9)
    assert bank.source_confidence[0, index, CandidateSource.DIRECT_MODE] == pytest.approx(0.4)
    assert bank.source_confidence[0, index, CandidateSource.CLASSICAL_X1] == pytest.approx(0.6)


def test_target_and_qc_mutation_cannot_change_forward_candidates() -> None:
    metadata = pd.DataFrame(
        {
            "window_number": [0, 1],
            "window_start_s": [10.0, 14.0],
            "classical_rr_bpm": [8.0, 9.0],
            "classical_confidence": [0.8, 0.7],
            "radar_peak_1_bpm": [8.0, 9.1],
            "radar_peak_2_bpm": [8.2, 8.9],
            "radar_peak_3_bpm": [7.9, 9.0],
            "radar_peak_spread_bpm": [0.1, 0.1],
            "rr_bpm": [16.0, 27.0],
            "reference_valid": [True, False],
            "reference_quality": [0.99, 0.01],
            "protocol": ["rest", "exercise"],
        }
    )
    changed = metadata.copy()
    changed["rr_bpm"] = [-1000.0, 1000.0]
    changed["reference_valid"] = ~changed["reference_valid"]
    changed["reference_quality"] = 1.0 - changed["reference_quality"]
    changed["protocol"] = ["mutated", "mutated"]
    first_fields = select_forward_metadata(metadata)
    second_fields = select_forward_metadata(changed)
    for name in first_fields:
        np.testing.assert_array_equal(first_fields[name], second_fields[name])
    first = candidate_bank_from_metadata(metadata)
    second = candidate_bank_from_metadata(changed)
    for name in ("bpm", "mask", "confidence", "source_mask", "primary_source"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    with pytest.raises(ValueError, match="outside the forward allow-list"):
        select_forward_metadata(metadata, fields=("rr_bpm",))


def test_missing_radar_uses_one_joint_fail_closed_mask() -> None:
    frequencies, maps, spectra, attributes = _synthetic_evidence()
    maps[:, 1] = 0.0  # radar 2 absent from RF
    spectra[:, 2] = 0.0  # radar 3 absent from SVD
    joint = resolve_joint_radar_mask(
        maps, spectra, svd_attributes=attributes
    )
    np.testing.assert_array_equal(joint, [[True, False, False]])
    candidates = _one_candidate()
    rf = sample_rf_harmonic_support(
        maps, frequencies, candidates, radar_mask=joint, ratios=(1.0,)
    )
    svd = sample_svd_harmonic_support(
        spectra,
        attributes,
        frequencies,
        candidates,
        radar_mask=joint,
        ratios=(1.0,),
    )
    np.testing.assert_array_equal(rf.radar_mask, svd.radar_mask)
    assert rf.mask[0, 0, 0, 0] and svd.mask[0, 0, 0, 0]
    assert not rf.mask[0, 0, 1:, 0].any()
    assert not svd.mask[0, 0, 1:, 0].any()
    assert not np.any(rf.values[0, 0, 1:])
    assert not np.any(svd.values[0, 0, 1:])


def test_compact_nodes_keep_top_range_location_and_are_fixed_width() -> None:
    frequencies, maps, spectra, attributes = _synthetic_evidence()
    candidates = _one_candidate()
    joint = resolve_joint_radar_mask(
        maps, spectra, svd_attributes=attributes
    )
    rf = sample_rf_harmonic_support(
        maps, frequencies, candidates, radar_mask=joint, ratios=(1.0,)
    )
    svd = sample_svd_harmonic_support(
        spectra,
        attributes,
        frequencies,
        candidates,
        radar_mask=joint,
        ratios=(1.0,),
    )
    nodes = build_compact_node_features(candidates, rf, svd, rf_top_k=2)
    assert nodes.features.shape[:2] == (1, 12)
    assert nodes.features.shape[-1] == len(nodes.feature_names)
    assert np.isfinite(nodes.features).all()
    raw_name = "rf_radar1_r1_raw_power_top1_range_index_unit"
    iq_name = "rf_radar1_r1_candidate_iq_phase_power_top1_range_index_unit"
    raw_value = nodes.features[0, 0, nodes.feature_names.index(raw_name)]
    iq_value = nodes.features[0, 0, nodes.feature_names.index(iq_name)]
    assert raw_value == pytest.approx(17.0 / 90.0)
    assert iq_value == pytest.approx(17.0 / 90.0)
    assert not np.any(nodes.features[0, 1:])


def test_lazy_builder_slices_rows_and_keeps_feature_schema_constant() -> None:
    frequencies, maps, spectra, attributes = _synthetic_evidence(rows=2)
    candidates = _one_candidate(rows=2)
    batches = list(
        iter_compact_node_feature_batches(
            maps,
            frequencies,
            spectra,
            attributes,
            frequencies,
            candidates,
            ratios=(0.5, 1.0),
            batch_size=1,
        )
    )
    assert [(batch.row_slice.start, batch.row_slice.stop) for batch in batches] == [
        (0, 1),
        (1, 2),
    ]
    assert batches[0].nodes.feature_names == batches[1].nodes.feature_names
    assert batches[0].rf_support.values.shape[0] == 1
    assert batches[0].svd_support.values.shape[0] == 1


def _semantic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cache_index": [10, 11],
            "session_id": ["S01_A", "S01_A"],
            "identity": ["A", "A"],
            "protocol": ["rest", "rest"],
            "fold": [2, 2],
            "window_number": [0, 1],
            "window_start_s": [1.25, 5.25],
            "window_end_s": [33.15, 37.15],
        }
    )


def test_semantic_row_binding_is_exact_and_mismatch_fails_closed() -> None:
    expected = _semantic_frame()
    source = expected.copy()
    digest = assert_exact_semantic_row_binding(source, expected, label="stack")
    assert digest == semantic_row_binding_sha256(expected)
    assert len(digest) == 64

    wrong = source.copy()
    wrong.loc[1, "window_number"] = 2
    with pytest.raises(RuntimeError, match="semantic row binding mismatch"):
        assert_exact_semantic_row_binding(wrong, expected, label="stack")
    reordered = source.iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="semantic row binding mismatch"):
        assert_exact_semantic_row_binding(reordered, expected, label="stack")
    missing = source.drop(columns="protocol")
    with pytest.raises(RuntimeError, match="cannot be verified"):
        assert_exact_semantic_row_binding(missing, expected, label="stack")


def test_contract_harmonic_ratios_include_all_required_native_queries() -> None:
    assert HARMONIC_RATIOS == (0.25, 1 / 3, 0.5, 1.0, 2.0, 3.0, 4.0)
