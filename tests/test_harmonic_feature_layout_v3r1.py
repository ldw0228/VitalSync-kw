from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from snn_rr.harmonic_feature_layout_v3r1 import (
    EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
    FEATURE_LAYOUT_SEMANTIC_SHA256,
    FeatureLayoutContractError,
    OuterTrainFeatureStandardizer,
    build_structural_availability_mask,
    load_and_validate_feature_names,
    sanitize_structural_features,
    validate_ordered_feature_names,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_CACHE = ROOT / (
    "artifacts/cache/"
    "harmonic_set_v2_i2r_nested_o3_s20260828_nms125_base_emap_svd12_m050"
)


def test_frozen_feature_schema_and_structural_layout_are_exact() -> None:
    names = load_and_validate_feature_names(REAL_CACHE / "feature_names.json")
    assert len(names) == 571
    assert validate_ordered_feature_names(names) == EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
    assert FEATURE_LAYOUT_SEMANTIC_SHA256 == (
        "8c8a7d8be241c4dddfe8d6d96324e102a7db37081553974dbec1f04a52655d01"
    )


def test_feature_schema_fails_closed_on_byte_or_order_drift(tmp_path: Path) -> None:
    source = REAL_CACHE / "feature_names.json"
    comment_drift = tmp_path / "feature_names.json"
    comment_drift.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(FeatureLayoutContractError, match="byte digest"):
        load_and_validate_feature_names(comment_drift)

    names = json.loads(source.read_text(encoding="utf-8"))["node_feature_names"]
    names[0], names[1] = names[1], names[0]
    with pytest.raises(FeatureLayoutContractError, match="semantic digest"):
        validate_ordered_feature_names(names)


def test_structural_mask_has_core_raw_rf_and_svd_axes_but_never_iq() -> None:
    candidate_rr = np.asarray([[10.0, 30.0, 0.0]], dtype=np.float32)
    candidate_mask = np.asarray([[True, True, False]])
    radar_mask = np.asarray([[True, False, True]])
    mask = build_structural_availability_mask(
        candidate_rr, candidate_mask, radar_mask
    )
    assert mask.shape == (1, 3, 571)
    assert mask.dtype == np.bool_
    assert mask[0, 0, :46].all()
    assert mask[0, 1, :46].all()
    assert not mask[0, 2].any()

    rf = mask[..., 46:424].reshape(1, 3, 3, 7, 2, 9)
    svd = mask[..., 424:].reshape(1, 3, 3, 7, 7)
    assert not rf[..., 1, :].any()  # frozen IQ branch is unavailable
    assert not rf[:, :, 1].any()  # radar 2 is unavailable
    assert not svd[:, :, 1].any()
    # 10 bpm: ratios 1,2,3,4 are in [6,45]; lower ratios are out of band.
    assert rf[0, 0, 0, :, 0, 0].tolist() == [False, False, False, True, True, True, True]
    # 30 bpm: ratios 1/4,1/3,1/2,1 are in band; upper ratios are not.
    assert svd[0, 1, 2, :, 0].tolist() == [True, True, True, True, False, False, False]


def test_structural_mask_rejects_shape_dtype_and_available_rr_errors() -> None:
    rr = np.asarray([[10.0]], dtype=np.float32)
    candidates = np.ones((1, 1), dtype=bool)
    radars = np.ones((1, 3), dtype=bool)
    with pytest.raises(FeatureLayoutContractError, match="boolean"):
        build_structural_availability_mask(rr, candidates.astype(np.int8), radars)
    with pytest.raises(FeatureLayoutContractError, match="joint_radar_mask"):
        build_structural_availability_mask(rr, candidates, np.ones((1, 2), bool))
    with pytest.raises(FeatureLayoutContractError, match="in-range"):
        build_structural_availability_mask(
            np.asarray([[np.nan]], dtype=np.float32), candidates, radars
        )


def test_sanitizer_overwrites_masked_nan_inf_and_nonzero_but_rejects_available_nan() -> None:
    rr = np.asarray([[10.0]], dtype=np.float32)
    candidates = np.ones((1, 1), dtype=bool)
    radars = np.asarray([[True, False, False]])
    availability = build_structural_availability_mask(rr, candidates, radars)
    values = np.full((1, 1, 571), 17.0, dtype=np.float32)
    values[~availability] = np.nan
    clean = sanitize_structural_features(values, availability)
    assert np.isfinite(clean).all()
    assert np.all(clean[~availability] == 0.0)
    assert np.all(clean[availability] == 17.0)
    values[..., 0] = np.nan
    with pytest.raises(FeatureLayoutContractError, match="available feature"):
        sanitize_structural_features(values, availability)


def test_outer_train_standardizer_uses_only_fit_positions_and_round_trips_json(
    tmp_path: Path,
) -> None:
    rr = np.asarray([[10.0], [10.0], [10.0]], dtype=np.float32)
    candidate_mask = np.ones((3, 1), dtype=bool)
    radar_mask = np.ones((3, 3), dtype=bool)
    availability = build_structural_availability_mask(rr, candidate_mask, radar_mask)
    values = np.zeros((3, 1, 571), dtype=np.float32)
    values[0, 0, availability[0, 0]] = 1.0
    values[1, 0, availability[1, 0]] = 3.0
    values[2, 0, availability[2, 0]] = 10_000.0  # held-out sentinel
    values[~availability] = np.inf  # unavailable bytes never enter the fit
    fit_positions = np.asarray([True, True, False])
    scaler = OuterTrainFeatureStandardizer.fit(
        values, availability, fit_positions=fit_positions
    )
    observed = scaler.observed_count > 0
    assert np.all(scaler.mean[observed] == 2.0)
    assert np.all(scaler.scale[observed] == 1.0)
    assert np.all(scaler.observed_count[~observed] == 0)
    transformed = scaler.transform(values, availability)
    assert transformed.dtype == np.float32
    assert np.all(transformed[~availability] == 0.0)
    assert np.all(transformed[0, 0, availability[0, 0]] == -1.0)
    assert np.all(transformed[1, 0, availability[1, 0]] == 1.0)

    state_path = tmp_path / "scaler.json"
    receipt = scaler.save_json(state_path)
    assert receipt["semantic_sha256"] == scaler.state_receipt()["semantic_sha256"]
    restored = OuterTrainFeatureStandardizer.load_json(state_path)
    assert restored.state_receipt() == scaler.state_receipt()
    np.testing.assert_array_equal(
        restored.transform(values, availability), transformed
    )

    document = json.loads(state_path.read_text(encoding="utf-8"))
    document["mean"][0] += 1.0
    state_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(FeatureLayoutContractError, match="semantic receipt"):
        OuterTrainFeatureStandardizer.load_json(state_path)

    state_path.write_text(json.dumps(scaler.to_state()), encoding="utf-8")
    with pytest.raises(FeatureLayoutContractError, match="semantic_receipt"):
        OuterTrainFeatureStandardizer.load_json(state_path)


def test_standardizer_is_deterministic_and_outer_train_scope_fails_closed() -> None:
    generator = np.random.default_rng(123)
    rr = np.full((4, 2), 12.0, dtype=np.float32)
    candidate_mask = np.ones((4, 2), dtype=bool)
    radar_mask = np.ones((4, 3), dtype=bool)
    availability = build_structural_availability_mask(rr, candidate_mask, radar_mask)
    values = generator.normal(size=availability.shape).astype(np.float32)
    values[~availability] = np.nan
    first = OuterTrainFeatureStandardizer.fit(values, availability)
    second = OuterTrainFeatureStandardizer.fit(values.copy(), availability.copy())
    assert first.state_receipt() == second.state_receipt()
    with pytest.raises(FeatureLayoutContractError, match="outer_train_only"):
        OuterTrainFeatureStandardizer.fit(
            values, availability, fit_scope="outer_validation"
        )
    bad = values.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(FeatureLayoutContractError, match="outer-train"):
        OuterTrainFeatureStandardizer.fit(bad, availability)


def test_real_cache_sample_masks_out_of_band_raw_and_iq_then_scales_finitely() -> None:
    node = np.asarray(
        np.load(REAL_CACHE / "node_features.npy", mmap_mode="r")[:10]
    )
    rr = np.asarray(np.load(REAL_CACHE / "candidate_bpm.npy", mmap_mode="r")[:10])
    candidate_mask = np.asarray(
        np.load(REAL_CACHE / "candidate_mask.npy", mmap_mode="r")[:10]
    )
    radar_mask = np.asarray(
        np.load(REAL_CACHE / "joint_radar_mask.npy", mmap_mode="r")[:10]
    )
    availability = build_structural_availability_mask(rr, candidate_mask, radar_mask)
    rf = node[..., 46:424].reshape(10, 12, 3, 7, 2, 9)
    rf_mask = availability[..., 46:424].reshape(10, 12, 3, 7, 2, 9)
    # The frozen cache genuinely contains raw nonzeros beyond the model's
    # [6,45] ratio band; the wrapper must remove them, not merely test zeros.
    assert np.count_nonzero(rf[~rf_mask]) > 0
    clean = sanitize_structural_features(node, availability)
    clean_rf = clean[..., 46:424].reshape(10, 12, 3, 7, 2, 9)
    assert np.count_nonzero(clean_rf[..., 1, :]) == 0
    assert np.count_nonzero(clean[~availability]) == 0
    scaler = OuterTrainFeatureStandardizer.fit(clean, availability)
    transformed = scaler.transform(clean, availability)
    assert np.isfinite(transformed).all()
    assert np.count_nonzero(transformed[~availability]) == 0
