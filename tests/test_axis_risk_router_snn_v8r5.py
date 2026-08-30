from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import snn_rr.axis_risk_router_snn_v8r5 as v8r5_module
from snn_rr.axis_risk_router_snn_v8r5 import (
    AxisRiskRouterSNNV8R5,
    build_directed_harmonic_edge_weights,
    build_structural_availability_mask,
    soft_risk_routing_loss,
    validate_v8r5_cache_contract,
)
from snn_rr.harmonic_feature_layout_v3r1 import (
    EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
    FEATURE_LAYOUT_SEMANTIC_SHA256,
    RF_SVD_RATIOS,
    build_structural_availability_mask as build_numpy_structural_availability_mask,
)
from snn_rr.harmonic_set_data import CANDIDATE_SOURCE_NAMES, RF_BRANCH_NAMES
from scripts.build_harmonic_set_cache import PROPOSER_NODE_FEATURE_NAMES


def _model() -> AxisRiskRouterSNNV8R5:
    torch.manual_seed(71)
    model = AxisRiskRouterSNNV8R5(
        ordered_feature_names_semantic_sha256=(
            EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
        ),
        dropout=0.0,
    )
    model.eval()
    return model


def _inputs(time: int = 4) -> dict[str, torch.Tensor]:
    candidate_rr = torch.tensor([[[10.0, 20.0, 30.0, 0.0]]]).expand(
        1, time, 4
    ).clone()
    candidate_mask = torch.tensor([[[True, True, True, False]]]).expand(
        1, time, 4
    ).clone()
    generator = torch.Generator().manual_seed(113)
    features = torch.randn((1, time, 4, 571), generator=generator)
    sequence_mask = torch.ones((1, time), dtype=torch.bool)
    radar_mask = torch.tensor([[[True, False, True]]]).expand(1, time, 3).clone()
    availability = build_structural_availability_mask(
        candidate_rr, candidate_mask, radar_mask
    )
    return {
        "node_features": features,
        "node_feature_availability": availability,
        "candidate_rr_bpm": candidate_rr,
        "candidate_mask": candidate_mask,
        "sequence_mask": sequence_mask,
        "joint_radar_mask": radar_mask,
        "proposer_anchor_bpm": torch.full((1, time), 18.0),
        "proposer_anchor_std_bpm": torch.full((1, time), 1.2),
        "proposer_anchor_available": torch.ones((1, time), dtype=torch.bool),
        "classical_rr_bpm": torch.full((1, time), 10.0),
        "classical_rr_available": torch.ones((1, time), dtype=torch.bool),
        "reset_mask": torch.tensor([[True] + [False] * (time - 1)]),
    }


def _forward(
    model: AxisRiskRouterSNNV8R5,
    values: dict[str, torch.Tensor],
    *,
    state=None,
) -> dict:
    return model(
        values["node_features"],
        values["candidate_rr_bpm"],
        values["candidate_mask"],
        values["sequence_mask"],
        node_feature_availability=values["node_feature_availability"],
        joint_radar_mask=values["joint_radar_mask"],
        proposer_anchor_bpm=values["proposer_anchor_bpm"],
        proposer_anchor_std_bpm=values["proposer_anchor_std_bpm"],
        proposer_anchor_available=values["proposer_anchor_available"],
        classical_rr_bpm=values["classical_rr_bpm"],
        classical_rr_available=values["classical_rr_available"],
        reset_mask=values["reset_mask"],
        state=state,
    )


def _slice(
    values: dict[str, torch.Tensor], begin: int, end: int
) -> dict[str, torch.Tensor]:
    return {key: value[:, begin:end].clone() for key, value in values.items()}


def _canonical_feature_names() -> list[str]:
    names = ["candidate_bpm", "candidate_bpm_unit", "candidate_confidence"]
    names.extend(f"source_{name}" for name in CANDIDATE_SOURCE_NAMES)
    names.extend(f"source_confidence_{name}" for name in CANDIDATE_SOURCE_NAMES)
    names.extend(PROPOSER_NODE_FEATURE_NAMES)
    names.extend(
        ("candidate_count_fraction", "previous_candidate_gap_bpm", "next_candidate_gap_bpm")
    )
    ratio_names = ("r1_4", "r1_3", "r1_2", "r1", "r2", "r3", "r4")
    assert len(ratio_names) == len(RF_SVD_RATIOS)
    for radar in range(3):
        for ratio_name in ratio_names:
            for branch_name in RF_BRANCH_NAMES:
                prefix = f"rf_radar{radar + 1}_{ratio_name}_{branch_name}"
                names.extend(
                    (
                        prefix + "_mean",
                        prefix + "_max",
                        prefix + "_entropy",
                        prefix + "_peak_concentration",
                        prefix + "_top1_value",
                        prefix + "_top1_range_index_unit",
                        prefix + "_top2_value",
                        prefix + "_top2_range_index_unit",
                        prefix + "_cross_radar_consensus",
                    )
                )
    for radar in range(3):
        for ratio_name in ratio_names:
            prefix = f"svd_radar{radar + 1}_{ratio_name}"
            names.extend(
                (
                    prefix + "_reliability_weighted_mean",
                    prefix + "_max",
                    prefix + "_component_entropy",
                    prefix + "_reliability_weighted_peak_distance_bpm",
                    prefix + "_closest_peak_distance_bpm",
                    prefix + "_reliability_mean",
                    prefix + "_reliability_max",
                )
            )
    assert len(names) == 571
    return names


def test_parameter_cap_safe_initialization_and_unmeasured_receipt() -> None:
    model = _model()
    assert model.parameter_count() == 228_838
    assert model.parameter_count() <= 400_000
    model.assert_safe_initialization()
    receipt = model.layout_receipt()
    assert receipt["training_authorized"] is False
    assert receipt["commercial_claim_allowed"] is False
    assert receipt["receipt_schema_version"] == 2
    assert len(str(receipt["model_source_sha256"])) == 64
    assert len(str(receipt["spiking_cell_source_sha256"])) == 64
    assert len(str(receipt["proposal_config_sha256"])) == 64
    assert len(str(receipt["behavior_contract_sha256"])) == 64
    assert len(str(receipt["runtime_structure_receipt_sha256"])) == 64
    assert receipt["runtime_structure_receipt_schema_version"] == 1
    assert receipt["proposal_config_absence_policy"] == "module_import_fails_closed"
    assert receipt["binds_actual_loader_compiled_bytes"] is False
    assert receipt["model_source_binding_scope"] == (
        "initialization_time_disk_bytes_not_actual_loader_compiled_bytes"
    )
    assert receipt["training_authorization_terminal_blocker"] == (
        "external_launcher_executed_byte_closure_and_verifier_absent"
    )
    assert "unmeasured_proposal" in str(receipt["model_family"])
    assert receipt["continuous_edge_evidence"] == "directed_log_ratio_proximity"
    assert receipt["near_relation_tolerance_bpm"] == 0.5
    assert receipt["ratio_relation_tolerance_bpm"] == 0.75
    assert receipt["edge_log_ratio_bandwidth"] == 0.08
    assert receipt["factor_affinity_bandwidth_bpm"] == 0.75
    assert receipt["temporal_state_dtype"] == "float32"
    assert torch.count_nonzero(model.factor_head.weight) == 0
    assert torch.count_nonzero(model.quality_head.weight) == 0
    with pytest.raises(ValueError, match="parameter cap"):
        AxisRiskRouterSNNV8R5(
            ordered_feature_names_semantic_sha256=(
                EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
            ),
            maximum_parameters=100,
        )


def test_concrete_v8r5_cache_contract_reads_and_hashes_on_disk_payloads(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    features = np.zeros((1, 12, 571), dtype=np.float32)
    availability = np.zeros_like(features, dtype=bool)
    candidate_bpm = np.zeros((1, 12), dtype=np.float32)
    candidate_mask = np.zeros((1, 12), dtype=bool)
    joint_radar_mask = np.zeros((1, 3), dtype=bool)
    np.save(cache / "node_features.npy", features, allow_pickle=False)
    np.save(
        cache / "node_feature_availability.npy", availability, allow_pickle=False
    )
    np.save(cache / "candidate_bpm.npy", candidate_bpm, allow_pickle=False)
    np.save(cache / "candidate_mask.npy", candidate_mask, allow_pickle=False)
    np.save(cache / "joint_radar_mask.npy", joint_radar_mask, allow_pickle=False)
    names_document = {
        "node_feature_names": _canonical_feature_names(),
        "forward_arrays": [
            "node_features",
            "node_feature_availability",
            "candidate_bpm",
            "candidate_mask",
            "joint_radar_mask",
        ],
        "ordered_feature_names_semantic_sha256": (
            EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
        ),
        "feature_layout_semantic_sha256": FEATURE_LAYOUT_SEMANTIC_SHA256,
        "axis_risk_router_v8r5_compatible": True,
    }
    names_path = cache / "feature_names.json"
    names_path.write_text(json.dumps(names_document), encoding="utf-8")

    def binding(filename: str) -> dict[str, str | int]:
        path = cache / filename
        return {
            "filename": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    manifest = {
        "schema": "snn_rr.harmonic_candidate_cache.v2",
        "format_version": 2,
        "complete": True,
        "row_count": 1,
        "node_feature_shape": [1, 12, 571],
        "node_feature_dtype": "float32",
        "node_feature_availability_shape": [1, 12, 571],
        "node_feature_availability_dtype": "bool",
        "ordered_feature_names_semantic_sha256": (
            EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
        ),
        "feature_layout_semantic_sha256": FEATURE_LAYOUT_SEMANTIC_SHA256,
        "axis_risk_router_v8r5_compatible": True,
        "settings": {
            "format_version": 2,
            "maximum_candidates": 12,
            "proposer_features": True,
            "svd_components": 12,
        },
        "outputs": {
            "node_features": binding("node_features.npy"),
            "node_feature_availability": binding(
                "node_feature_availability.npy"
            ),
            "candidate_bpm": binding("candidate_bpm.npy"),
            "candidate_mask": binding("candidate_mask.npy"),
            "joint_radar_mask": binding("joint_radar_mask.npy"),
            "feature_names": binding("feature_names.json"),
        },
    }
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = cache / "manifest.json"
    manifest_text = json.dumps(manifest)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    receipt = validate_v8r5_cache_contract(cache)
    assert receipt["schema_compatible"] is True
    assert receipt["training_authorized"] is False

    def write_rehashed(document: dict) -> None:
        document.pop("content_sha256", None)
        document["content_sha256"] = hashlib.sha256(
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        manifest_path.write_text(json.dumps(document), encoding="utf-8")

    bool_shape = json.loads(manifest_text)
    bool_shape["node_feature_shape"][0] = True
    bool_shape["node_feature_availability_shape"][0] = True
    write_rehashed(bool_shape)
    with pytest.raises(ValueError, match="shape or dtype contract"):
        validate_v8r5_cache_contract(cache)
    float_version = json.loads(manifest_text)
    float_version["format_version"] = 2.0
    write_rehashed(float_version)
    with pytest.raises(ValueError, match="format_version=2"):
        validate_v8r5_cache_contract(cache)
    bool_setting = json.loads(manifest_text)
    bool_setting["settings"]["maximum_candidates"] = True
    write_rehashed(bool_setting)
    with pytest.raises(ValueError, match="canonical V8R5 settings"):
        validate_v8r5_cache_contract(cache)
    manifest_path.write_text(manifest_text, encoding="utf-8")

    manifest_path.write_text(
        manifest_text[:-1] + ', "nonfinite": NaN}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        validate_v8r5_cache_contract(cache)
    manifest_path.write_text(manifest_text, encoding="utf-8")

    # A truthful-looking caller constant cannot hide changed on-disk bytes.
    with (cache / "node_feature_availability.npy").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="inode/size mismatch"):
        validate_v8r5_cache_contract(cache)


def test_torch_structural_mask_matches_frozen_numpy_layout_with_no_radar() -> None:
    candidate_rr = torch.tensor([[[12.0, 24.0]]])
    candidate_mask = torch.tensor([[[True, False]]])
    radar_mask = torch.zeros((1, 1, 3), dtype=torch.bool)

    actual = build_structural_availability_mask(
        candidate_rr, candidate_mask, radar_mask
    )
    expected = build_numpy_structural_availability_mask(
        candidate_rr.numpy(), candidate_mask.numpy(), radar_mask.numpy()
    )

    assert torch.equal(actual, torch.from_numpy(expected))
    assert actual[0, 0, 0, :46].all()
    assert not actual[0, 0, 0, 46:].any()
    with pytest.raises(ValueError, match="finite and in range"):
        build_structural_availability_mask(
            torch.tensor([[[5.0, 46.0]]]),
            torch.ones((1, 1, 2), dtype=torch.bool),
            torch.ones((1, 1, 3), dtype=torch.bool),
        )


def test_masked_nonfinite_rr_cannot_create_nonfinite_edge_weights() -> None:
    rr = torch.tensor([[[12.0, float("nan"), 24.0]]])
    mask = torch.tensor([[[True, False, True]]])
    weights = build_directed_harmonic_edge_weights(rr, mask)
    assert torch.isfinite(weights).all()
    assert not weights[..., 1, :, :].any()
    assert not weights[..., :, 1, :].any()


def test_amp_fully_masked_axes_remain_finite() -> None:
    model = _model()
    values = _inputs(time=2)
    values["joint_radar_mask"] = torch.tensor(
        [[[True, False, False], [False, False, False]]]
    )
    values["node_feature_availability"] = build_structural_availability_mask(
        values["candidate_rr_bpm"],
        values["candidate_mask"],
        values["joint_radar_mask"],
    )

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        output = _forward(model, values)

    for key in (
        "expert_logits",
        "expert_probabilities",
        "node_embeddings",
        "source_rr_bpm",
        "source_scale_bpm",
        "temporal_state_sequence",
    ):
        assert torch.isfinite(output[key]).all(), key


def test_forward_contract_is_target_and_identity_free() -> None:
    parameters = set(inspect.signature(AxisRiskRouterSNNV8R5.forward).parameters)
    forbidden = {
        "target",
        "reference_rr_bpm",
        "reference_validity",
        "reference_quality",
        "identity",
        "protocol",
        "future_windows",
    }
    assert not parameters & forbidden
    output = _forward(_model(), _inputs(time=1))
    assert not set(output) & forbidden


def test_internal_ancestry_is_exact_and_hash_bound_not_claimed_independent() -> None:
    source_path = (
        Path(__file__).parents[1] / "src/snn_rr/axis_risk_router_snn_v8r5.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    internal_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }
    assert internal_imports == {
        "harmonic_feature_layout_v3r1",
        "svd_episode_models",
    }
    receipt = _model().layout_receipt()
    assert receipt["ancestry_policy"] == (
        "successor_with_hash_bound_layout_and_spiking_cell_reuse"
    )
    assert len(receipt["spiking_cell_source_sha256"]) == 64
    assert len(receipt["feature_layout_source_sha256"]) == 64
    config = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "configs/axis_risk_router_snn_v8r5.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["model"]["independent_implementation_claim"] is False
    assert config["model"]["governed_reused_dependencies"] == [
        "harmonic_feature_layout_v3r1",
        "svd_episode_models.EpisodeSpikingCell",
    ]
    dependency_hashes = config["model"]["governed_dependency_source_sha256"]
    assert receipt["feature_layout_source_sha256"] == dependency_hashes[
        "harmonic_feature_layout_v3r1"
    ]
    assert receipt["spiking_cell_source_sha256"] == dependency_hashes[
        "svd_episode_models"
    ]


def test_structural_availability_keeps_iq_absent_and_exact_width() -> None:
    values = _inputs(time=1)
    availability = build_structural_availability_mask(
        values["candidate_rr_bpm"],
        values["candidate_mask"],
        values["joint_radar_mask"],
    )
    assert availability.shape == (1, 1, 4, 571)
    rf = availability[..., 46:424].reshape(1, 1, 4, 3, 7, 2, 9)
    assert rf[..., 0, :].any()
    assert not rf[..., 1, :].any()
    assert not availability[..., 3, :].any()


def test_coordinate_evidence_swap_changes_node_embedding() -> None:
    model = _model()
    values = _inputs(time=1)
    swapped = {key: value.clone() for key, value in values.items()}
    rf = swapped["node_features"][..., 46:424].reshape(1, 1, 4, 3, 7, 2, 9)
    left = rf[0, 0, 1, 0, 3, 0].clone()
    right = rf[0, 0, 1, 2, 3, 0].clone()
    rf[0, 0, 1, 0, 3, 0] = right
    rf[0, 0, 1, 2, 3, 0] = left
    with torch.no_grad():
        original_output = _forward(model, values)
        swapped_output = _forward(model, swapped)
    difference = (
        original_output["node_embeddings"][0, 0, 1]
        - swapped_output["node_embeddings"][0, 0, 1]
    ).abs().max()
    assert difference > 1.0e-5


def test_candidate_permutation_is_equivariant() -> None:
    model = _model()
    values = _inputs(time=2)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = {key: value.clone() for key, value in values.items()}
    for key in (
        "node_features",
        "node_feature_availability",
        "candidate_rr_bpm",
        "candidate_mask",
    ):
        permuted[key] = values[key].index_select(2, permutation)
    with torch.no_grad():
        original = _forward(model, values)
        changed = _forward(model, permuted)
    torch.testing.assert_close(
        changed["node_embeddings"],
        original["node_embeddings"].index_select(2, permutation),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    torch.testing.assert_close(
        changed["expert_mean_bpm"][..., 1:],
        original["expert_mean_bpm"][..., 1:].index_select(2, permutation),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    torch.testing.assert_close(
        changed["expert_logits"][..., 0], original["expert_logits"][..., 0]
    )


def test_unavailable_cells_are_sanitized_and_cannot_change_output() -> None:
    model = _model()
    values = _inputs(time=2)
    availability = build_structural_availability_mask(
        values["candidate_rr_bpm"],
        values["candidate_mask"],
        values["joint_radar_mask"],
    )
    contaminated = {key: value.clone() for key, value in values.items()}
    contaminated["node_features"][~availability] = float("nan")
    with torch.no_grad():
        clean = _forward(model, values)
        dirty = _forward(model, contaminated)
    for name in (
        "source_rr_bpm",
        "expert_logits",
        "expert_mean_bpm",
        "node_embeddings",
        "temporal_state_sequence",
    ):
        torch.testing.assert_close(dirty[name], clean[name], rtol=0.0, atol=0.0)


def test_explicit_exact_availability_can_only_narrow_the_canonical_ceiling() -> None:
    model = _model()
    values = _inputs(time=1)
    narrowed = {key: value.clone() for key, value in values.items()}
    narrowed["node_feature_availability"][0, 0, 0, 46] = False
    narrowed["node_features"][0, 0, 0, 46] = float("nan")
    with torch.no_grad():
        output = _forward(model, narrowed)
    assert torch.isfinite(output["node_embeddings"]).all()
    assert not output["layout_diagnostics"]["structural_availability"][
        0, 0, 0, 46
    ]

    widened = {key: value.clone() for key, value in values.items()}
    impossible = int(
        torch.nonzero(
            ~widened["node_feature_availability"][0, 0, 0], as_tuple=False
        )[0]
    )
    widened["node_feature_availability"][0, 0, 0, impossible] = True
    with pytest.raises(ValueError, match="canonical structural ceiling"):
        _forward(model, widened)


def test_per_feature_mask_distinguishes_missing_from_measured_zero() -> None:
    model = _model()
    measured_zero = _inputs(time=1)
    rf_mask = measured_zero["node_feature_availability"][
        0, 0, 0, 46:424
    ].reshape(3, 7, 2, 9)
    cell = torch.nonzero(rf_mask.all(dim=-1), as_tuple=False)[0]
    radar, ratio, branch = (int(value) for value in cell)
    base = 46 + (((radar * 7 + ratio) * 2 + branch) * 9)
    feature_index = base + 1
    measured_zero["node_features"][0, 0, 0, feature_index] = 0.0
    missing = {key: value.clone() for key, value in measured_zero.items()}
    missing["node_feature_availability"][0, 0, 0, feature_index] = False

    with torch.no_grad():
        observed = _forward(model, measured_zero)
        absent = _forward(model, missing)
    assert absent["layout_diagnostics"]["rf_branch_mask"][
        0, 0, 0, radar, ratio, branch
    ]
    difference = (
        observed["node_embeddings"][0, 0, 0]
        - absent["node_embeddings"][0, 0, 0]
    ).abs().max()
    assert difference > 1.0e-6


    # Attention normalization must address the same radar/ratio dimension that
    # is reduced.  Negative axes change after the scorer removes channels.
    output = _forward(_model(), _inputs(time=1))
    diagnostics = output["layout_diagnostics"]
    for prefix, mask_name in (
        ("rf", "rf_cell_mask"),
        ("svd", "svd_cell_mask"),
    ):
        mask = diagnostics[mask_name]
        attention = diagnostics[f"{prefix}_axial_attention"]
        ratio_first = attention["ratio_then_radar_ratio_weights"]
        radar_second = attention["ratio_then_radar_radar_weights"]
        radar_first = attention["radar_then_ratio_radar_weights"]
        ratio_second = attention["radar_then_ratio_ratio_weights"]
        torch.testing.assert_close(
            ratio_first.sum(dim=-1),
            mask.any(dim=-1).to(ratio_first.dtype),
        )
        torch.testing.assert_close(
            radar_second.sum(dim=-1),
            mask.any(dim=(-2, -1)).to(radar_second.dtype),
        )
        torch.testing.assert_close(
            radar_first.sum(dim=-2),
            mask.any(dim=-2).to(radar_first.dtype),
        )
        torch.testing.assert_close(
            ratio_second.sum(dim=-1),
            mask.any(dim=(-2, -1)).to(ratio_second.dtype),
        )


def test_available_nonfinite_feature_fails_closed() -> None:
    values = _inputs(time=1)
    values["node_features"][0, 0, 0, 0] = float("nan")
    output = _forward(_model(), values)
    assert not output["candidate_mask"][0, 0, 0]
    assert output["source_integrity_failed"].all()
    assert output["source_available"].all()
    assert torch.isfinite(output["source_rr_bpm"]).all()
    assert torch.equal(output["quality"], torch.zeros_like(output["quality"]))


def test_all_sources_missing_returns_finite_unavailable_state() -> None:
    values = _inputs(time=2)
    values["joint_radar_mask"].zero_()
    values["candidate_mask"].zero_()
    values["node_feature_availability"].zero_()
    values["proposer_anchor_available"].zero_()
    values["classical_rr_available"].zero_()
    values["classical_rr_bpm"].fill_(float("nan"))
    values["node_features"].fill_(float("nan"))
    with torch.no_grad():
        output = _forward(_model(), values)
    assert not output["source_available"].any()
    assert (output["selected_source_code"] == -1).all()
    assert torch.equal(output["source_rr_bpm"], torch.zeros_like(output["source_rr_bpm"]))
    assert torch.equal(
        output["source_scale_bpm"], torch.zeros_like(output["source_scale_bpm"])
    )
    assert not output["quality_available"].any()
    assert torch.equal(output["quality"], torch.zeros_like(output["quality"]))
    assert torch.isfinite(output["temporal_state_sequence"]).all()


def test_classical_rr_requires_explicit_provenance_availability() -> None:
    values = _inputs(time=1)
    values["candidate_mask"].zero_()
    values["node_feature_availability"].zero_()
    values["proposer_anchor_available"].zero_()
    values["classical_rr_available"].zero_()
    # A plausible numeric value is not evidence of availability.
    with torch.no_grad():
        output = _forward(_model(), values)
    assert not output["source_available"].any()
    assert not output["classical_rr_available"].any()
    assert torch.equal(
        output["classical_rr_bpm"], torch.zeros_like(output["classical_rr_bpm"])
    )

    invalid = {key: value.clone() for key, value in values.items()}
    invalid["classical_rr_available"].fill_(True)
    invalid["classical_rr_bpm"].fill_(float("nan"))
    invalid_output = _forward(_model(), invalid)
    assert invalid_output["source_integrity_failed"].all()
    assert not invalid_output["classical_rr_available"].any()
    assert not invalid_output["source_available"].any()
    assert torch.equal(
        invalid_output["source_rr_bpm"],
        torch.zeros_like(invalid_output["source_rr_bpm"]),
    )
    assert torch.equal(
        invalid_output["quality"], torch.zeros_like(invalid_output["quality"])
    )


def test_classical_only_fallback_quality_is_available_and_supervised() -> None:
    values = _inputs(time=1)
    values["candidate_mask"].zero_()
    values["node_feature_availability"].zero_()
    values["proposer_anchor_available"].zero_()
    output = _forward(_model(), values)
    assert output["source_available"].all()
    assert (output["selected_source_code"] == -2).all()
    assert output["quality_available"].all()
    loss, components = soft_risk_routing_loss(
        output,
        torch.tensor([[30.0]]),
        torch.ones((1, 1), dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    assert components["supervised_weight"].item() == 0.0
    assert components["quality_supervised_weight"].item() == 1.0
    assert components["quality_bce"] > 0.0


def test_untrained_hard_inference_selects_safe_anchor() -> None:
    output = _forward(_model(), _inputs(time=2))
    assert (output["selected_expert_index"] == 0).all()
    assert torch.equal(output["source_rr_bpm"], output["corrected_anchor_rr_bpm"])


def test_soft_risk_loss_routes_gradient_without_differentiating_argmax() -> None:
    model = _model()
    model.train()
    output = _forward(model, _inputs(time=3))
    output["expert_logits"].retain_grad()
    target = torch.tensor([[10.0, 20.0, 30.0]])
    valid = torch.ones_like(target, dtype=torch.bool)
    loss, components = soft_risk_routing_loss(output, target, valid)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert output["expert_logits"].grad is not None
    assert output["expert_logits"].grad.abs().sum() > 0.0
    assert model.candidate_value_head[-1].weight.grad is not None
    assert model.candidate_value_head[-1].weight.grad.abs().sum() > 0.0
    assert model.candidate_route_head[-1].weight.grad is not None
    assert model.candidate_route_head[-1].weight.grad.abs().sum() > 0.0
    assert model.candidate_risk_head[-1].weight.grad is not None
    assert model.candidate_risk_head[-1].weight.grad.abs().sum() > 0.0
    assert model.quality_head.weight.grad is not None
    assert model.quality_head.weight.grad.abs().sum() > 0.0


def test_routing_gradient_cannot_train_calibrated_risk_outputs_to_understate() -> None:
    model = _model()
    model.train()
    with torch.no_grad():
        model.candidate_route_head[-1].weight.normal_(std=0.1)
        model.anchor_route_head[-1].weight.normal_(std=0.1)
    output = _forward(model, _inputs(time=2))
    route_only = output["expert_logits"].masked_select(
        output["expert_mask"]
    ).sum()
    route_only.backward()
    assert model.candidate_route_head[0].weight.grad is not None
    assert model.candidate_route_head[0].weight.grad.abs().sum() > 0
    assert model.anchor_route_head[0].weight.grad is not None
    assert model.anchor_route_head[0].weight.grad.abs().sum() > 0
    for head in (model.candidate_risk_head, model.anchor_risk_head):
        assert all(parameter.grad is None for parameter in head.parameters())
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.temporal.parameters())


def test_invalid_reference_rows_are_inert_in_risk_loss() -> None:
    model = _model()
    output = _forward(model, _inputs(time=2))
    target = torch.tensor([[float("nan"), 20.0]])
    invalid = torch.zeros_like(target, dtype=torch.bool)
    loss, components = soft_risk_routing_loss(output, target, invalid)
    assert torch.equal(loss, loss.new_zeros(()))
    assert components["supervised_weight"].item() == 0.0


def test_streaming_chunks_match_whole_session() -> None:
    model = _model()
    values = _inputs(time=5)
    with torch.no_grad():
        whole = _forward(model, values)
        first = _forward(model, _slice(values, 0, 2))
        remainder = _slice(values, 2, 5)
        remainder["reset_mask"].zero_()
        second = _forward(model, remainder, state=first["state"])
    for name in (
        "source_rr_bpm",
        "source_scale_bpm",
        "expert_logits",
        "expert_mean_bpm",
        "expert_expected_abs_error_bpm",
        "expert_tail2_probabilities",
        "expert_tail5_probabilities",
        "quality",
        "temporal_state_sequence",
        "spike_sequence",
    ):
        streamed = torch.cat((first[name], second[name]), dim=1)
        torch.testing.assert_close(streamed, whole[name], rtol=2.0e-5, atol=2.0e-5)
    for streamed_layer, whole_layer in zip(second["state"], whole["state"], strict=True):
        for streamed_tensor, whole_tensor in zip(streamed_layer, whole_layer, strict=True):
            torch.testing.assert_close(
                streamed_tensor, whole_tensor, rtol=2.0e-5, atol=2.0e-5
            )


def test_amp_streaming_chunks_preserve_canonical_float32_state() -> None:
    model = _model()
    with torch.no_grad():
        generator = torch.Generator().manual_seed(911)
        for module in (
            model.candidate_value_head,
            model.anchor_value_head,
            model.candidate_route_head,
            model.anchor_route_head,
            model.candidate_risk_head,
            model.anchor_risk_head,
        ):
            head = module[-1]
            head.weight.copy_(
                torch.randn(head.weight.shape, generator=generator) * 0.08
            )
            head.bias.copy_(
                torch.randn(head.bias.shape, generator=generator) * 0.08
            )
    values = _inputs(time=24)
    chunks: list[dict] = []
    state = None
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        whole = _forward(model, values)
        for start in range(0, 24, 4):
            part = _slice(values, start, start + 4)
            if start:
                part["reset_mask"].zero_()
            output = _forward(model, part, state=state)
            chunks.append(output)
            state = output["state"]

    assert state is not None
    for layer in whole["state"]:
        assert all(tensor.dtype == torch.float32 for tensor in layer)
    for layer in state:
        assert all(tensor.dtype == torch.float32 for tensor in layer)
    for name in (
        "temporal_state_sequence",
        "expert_logits",
        "expert_mean_bpm",
        "deployment_hard_rr_bpm",
    ):
        streamed = torch.cat([chunk[name] for chunk in chunks], dim=1)
        torch.testing.assert_close(
            streamed, whole[name], rtol=2.0e-4, atol=2.0e-4
        )


def test_future_feature_mutation_cannot_change_causal_prefix() -> None:
    model = _model()
    original = _inputs(time=8)
    changed = {key: value.clone() for key, value in original.items()}
    changed["node_features"][:, 4:] = torch.randn_like(
        changed["node_features"][:, 4:]
    ) * 100.0
    with torch.no_grad():
        first = _forward(model, original)
        second = _forward(model, changed)
    for name in (
        "temporal_state_sequence",
        "expert_logits",
        "expert_mean_bpm",
        "deployment_hard_rr_bpm",
    ):
        torch.testing.assert_close(
            second[name][:, :4], first[name][:, :4], rtol=0.0, atol=0.0
        )


def test_tail_probabilities_are_structurally_monotone_under_autocast() -> None:
    model = _model()
    with torch.no_grad():
        torch.manual_seed(809)
        model.candidate_risk_head[-1].weight.normal_(mean=0.0, std=0.4)
        model.candidate_risk_head[-1].bias.normal_(mean=0.0, std=2.0)
        model.anchor_risk_head[-1].weight.normal_(mean=0.0, std=0.4)
        model.anchor_risk_head[-1].bias.normal_(mean=0.0, std=2.0)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        output = _forward(model, _inputs(time=3))
    probability2 = output["expert_tail2_probabilities"]
    probability5 = output["expert_tail5_probabilities"]
    mask = output["expert_mask"]
    assert torch.all(probability5 <= probability2)
    assert torch.equal(
        probability2.masked_select(~mask),
        torch.zeros_like(probability2.masked_select(~mask)),
    )
    assert torch.equal(
        probability5.masked_select(~mask),
        torch.zeros_like(probability5.masked_select(~mask)),
    )


def test_cpu_autocast_soft_training_loss_and_gradients_remain_finite() -> None:
    model = _model()
    model.train()
    target = torch.tensor([[10.0, 20.0, 30.0]])
    valid = torch.ones_like(target, dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.float16):
        output = _forward(model, _inputs(time=3))
        loss, components = soft_risk_routing_loss(output, target, valid)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_route_temperature_checkpoint_round_trip_and_setter(
    tmp_path: Path,
) -> None:
    model = _model()
    model.set_route_temperature(0.37)
    assert model.route_temperature == pytest.approx(0.37)
    assert "_route_temperature" in model.state_dict()
    checkpoint = tmp_path / "v8r5.pt"
    torch.save(model.state_dict(), checkpoint)

    restored = _model()
    checkpoint_state = torch.load(checkpoint, weights_only=True)
    with pytest.raises(RuntimeError, match="temperature.*differs"):
        restored.load_state_dict(checkpoint_state)
    restored.set_route_temperature(0.37)
    restored.load_state_dict(checkpoint_state)
    assert restored.route_temperature == pytest.approx(0.37)
    with torch.no_grad():
        expected = _forward(model, _inputs(time=2))["expert_probabilities"]
        actual = _forward(restored, _inputs(time=2))["expert_probabilities"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    for invalid in (0.0, -1.0, 1.0e-50, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="route_temperature"):
            restored.set_route_temperature(invalid)
    corrupt = restored.state_dict()
    corrupt["_route_temperature"] = torch.tensor(float("nan"))
    with pytest.raises(RuntimeError, match="temperature"):
        restored.load_state_dict(corrupt)
    missing_temperature = dict(restored.state_dict())
    missing_temperature.pop("_route_temperature")
    with pytest.raises(RuntimeError, match="Missing key.*_route_temperature"):
        restored.load_state_dict(missing_temperature)
    missing_head = dict(restored.state_dict())
    missing_head.pop("candidate_risk_head.3.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(missing_head)
    with pytest.raises(ValueError, match="strict=True"):
        restored.load_state_dict(restored.state_dict(), strict=False)

    incompatible = AxisRiskRouterSNNV8R5(
        ordered_feature_names_semantic_sha256=(
            EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
        ),
        tail2_risk_weight=2.0,
        dropout=0.0,
    )
    with pytest.raises(RuntimeError, match="behavior contract"):
        incompatible.load_state_dict(torch.load(checkpoint, weights_only=True))


def test_missing_mandatory_proposal_config_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="config is unavailable.*fails closed"):
        v8r5_module._read_required_proposal_config_bytes(
            tmp_path / "missing-v8r5-config.yaml"
        )


def test_checkpoint_rejects_nonfinite_parameter_or_buffer_before_mutation() -> None:
    source = _model()
    parameter_name, _ = next(source.named_parameters())

    for contaminated_name in (parameter_name, "_behavior_contract"):
        target = _model()
        before = {
            name: value.detach().clone()
            for name, value in target.state_dict().items()
        }
        state = {
            name: value.detach().clone()
            for name, value in source.state_dict().items()
        }
        state[contaminated_name].reshape(-1)[0] = float("nan")
        with pytest.raises(RuntimeError, match="non-finite"):
            target.load_state_dict(state)
        after = target.state_dict()
        for name, expected in before.items():
            assert torch.equal(after[name], expected), name


def test_checkpoint_runtime_behavior_is_immutable_and_freshly_cross_checked() -> None:
    source = _model()
    checkpoint = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
    }
    target = _model()
    with pytest.raises(AttributeError, match="immutable"):
        target.tail5_risk_weight = 9.0

    # Bypass normal assignment to prove strict load freshly derives behavior
    # from runtime attributes instead of trusting only the persistent buffer.
    object.__setattr__(target, "tail5_risk_weight", 9.0)
    before = {
        name: value.detach().clone()
        for name, value in torch.nn.Module.state_dict(target).items()
    }
    with pytest.raises(RuntimeError, match="runtime behavior"):
        target.load_state_dict(checkpoint)
    for name, expected in before.items():
        assert torch.equal(torch.nn.Module.state_dict(target)[name], expected), name


@pytest.mark.parametrize(
    "mutation",
    (
        "encoder_rr_bound",
        "head_dropout",
        "head_dropout_inplace",
        "child_training_mode",
        "layer_norm_eps",
        "maximum_candidates",
        "temporal_steps",
        "spiking_cell_type",
    ),
)
def test_strict_load_rejects_mutated_derived_runtime_behavior(
    mutation: str,
) -> None:
    source = _model()
    target = _model()
    before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
    }
    if mutation == "encoder_rr_bound":
        target.encoder.rr_min_bpm = 9.0
    elif mutation == "head_dropout":
        target.candidate_value_head[2].p = 0.9
    elif mutation == "head_dropout_inplace":
        target.candidate_value_head[2].inplace = True
    elif mutation == "child_training_mode":
        target.candidate_value_head[2].train(True)
    elif mutation == "layer_norm_eps":
        target.encoder.core[-1].eps = 0.5
    elif mutation == "maximum_candidates":
        object.__setattr__(target, "MAX_CANDIDATES", 1)
    elif mutation == "temporal_steps":
        target.temporal.simulation_steps = 7
    else:
        target.temporal.cells[0].cell_type = "lif"
    with pytest.raises(RuntimeError, match="runtime"):
        target.layout_receipt()
    with pytest.raises(RuntimeError, match="runtime"):
        target.state_dict()
    with pytest.raises(RuntimeError, match="runtime"):
        target.load_state_dict(source.state_dict())
    for name, expected in before.items():
        assert torch.equal(torch.nn.Module.state_dict(target)[name], expected), name


def test_strict_checkpoint_preflight_rejects_assign_shape_dtype_and_keys_without_mutation() -> None:
    source = _model()
    base = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
    }
    parameter_name, parameter = next(source.named_parameters())

    cases: list[tuple[dict[str, torch.Tensor], str]] = []
    wrong_shape = dict(base)
    wrong_shape[parameter_name] = parameter.detach().reshape(-1)[:-1].clone()
    cases.append((wrong_shape, "shape differs"))

    wrong_dtype = dict(base)
    wrong_dtype[parameter_name] = torch.full(
        parameter.shape, 1.0e300, dtype=torch.float64
    )
    cases.append((wrong_dtype, "dtype differs"))

    unexpected = dict(base)
    unexpected["unexpected.tensor"] = torch.zeros(1)
    cases.append((unexpected, "Unexpected key"))

    for state, message in cases:
        target = _model()
        before = {
            name: value.detach().clone()
            for name, value in target.state_dict().items()
        }
        with pytest.raises(RuntimeError, match=message):
            target.load_state_dict(state)
        for name, expected in before.items():
            assert torch.equal(target.state_dict()[name], expected), (message, name)

    target = _model()
    before_ids = {
        name: id(value) for name, value in target.named_parameters()
    }
    before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
    }
    with pytest.raises(ValueError, match="assign=False"):
        target.load_state_dict(base, assign=True)
    assert before_ids == {
        name: id(value) for name, value in target.named_parameters()
    }
    for name, expected in before.items():
        assert torch.equal(target.state_dict()[name], expected), name


def test_checkpoint_rejects_nonfinite_live_model_before_copy() -> None:
    source = _model()
    checkpoint = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
    }
    target = _model()
    parameter_name, parameter = next(target.named_parameters())
    with torch.no_grad():
        parameter.reshape(-1)[0] = float("nan")
    before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
    }
    with pytest.raises(RuntimeError, match="constructed model.*non-finite"):
        target.load_state_dict(checkpoint)
    after = target.state_dict()
    for name, expected in before.items():
        torch.testing.assert_close(
            after[name], expected, rtol=0.0, atol=0.0, equal_nan=True,
            msg=lambda message, name=name: f"{name}: {message}",
        )
    assert torch.isnan(after[parameter_name].reshape(-1)[0])


def test_checkpoint_rejects_nonfinite_nonpersistent_live_buffer() -> None:
    source = _model()
    target = _model()
    persistent_names = set(target.state_dict())
    buffer_name, buffer = next(
        (name, value)
        for name, value in target.named_buffers()
        if name not in persistent_names and value.is_floating_point()
    )
    with torch.no_grad():
        buffer.reshape(-1)[0] = float("nan")
    before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
    }
    with pytest.raises(
        RuntimeError, match=f"buffer:{buffer_name!s}.*non-finite"
    ):
        target.load_state_dict(source.state_dict())
    for name, expected in before.items():
        assert torch.equal(target.state_dict()[name], expected), name
    assert torch.isnan(buffer.reshape(-1)[0])


def test_checkpoint_load_hooks_are_rejected_before_any_mutation() -> None:
    source = _model()
    target = _model()
    before = {
        name: value.detach().clone()
        for name, value in target._named_live_parameters_and_buffers().items()
    }

    called = False

    def mutate_then_fail(module, _incompatible_keys) -> None:
        nonlocal called
        called = True
        if module is target:
            with torch.no_grad():
                next(module.parameters()).reshape(-1)[0].add_(1.0)
            module.temporal.simulation_steps = 1
            raise RuntimeError("injected live post-load failure")

    handle = target.register_load_state_dict_post_hook(mutate_then_fail)
    try:
        with pytest.raises(RuntimeError, match="forbids state-dict load hooks"):
            target.load_state_dict(source.state_dict())
    finally:
        handle.remove()
    assert called is False
    assert target.temporal.simulation_steps == 8
    after = target._named_live_parameters_and_buffers()
    for name, expected in before.items():
        assert torch.equal(after[name], expected), name
    target._assert_all_live_parameters_and_buffers_finite()


def test_checkpoint_receipt_rejects_dependency_or_source_transplant() -> None:
    source = _model()
    state = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
    }
    receipt_name = "_checkpoint_source_receipt"
    assert receipt_name in state
    receipt_document = json.loads(bytes(state[receipt_name].tolist()).decode("ascii"))
    assert receipt_document["model_source_sha256"] == source.layout_receipt()[
        "model_source_sha256"
    ]
    assert receipt_document["feature_layout_source_sha256"] == source.layout_receipt()[
        "feature_layout_source_sha256"
    ]
    assert receipt_document["spiking_cell_source_sha256"] == source.layout_receipt()[
        "spiking_cell_source_sha256"
    ]
    assert receipt_document["proposal_config_sha256"] == source.layout_receipt()[
        "proposal_config_sha256"
    ]
    assert receipt_document["binds_actual_loader_compiled_bytes"] is False
    assert receipt_document["training_authorization_terminal_blocker"] == (
        "external_launcher_executed_byte_closure_and_verifier_absent"
    )

    transplanted = dict(state)
    transplanted[receipt_name] = state[receipt_name].clone()
    transplanted[receipt_name][0] ^= 1
    with pytest.raises(
        RuntimeError, match="source/layout/config/dependency receipt"
    ):
        _model().load_state_dict(transplanted)

    runtime_transplant = dict(state)
    runtime_name = "_runtime_structure_receipt"
    runtime_transplant[runtime_name] = state[runtime_name].clone()
    runtime_transplant[runtime_name][0] ^= 1
    with pytest.raises(RuntimeError, match="runtime structure receipt"):
        _model().load_state_dict(runtime_transplant)


def test_nonfinite_available_route_falls_back_finite_with_zero_quality() -> None:
    model = _model()
    def nonfinite_output(_module, _inputs, output):
        return torch.full_like(output, float("nan"))

    handles = [
        model.anchor_route_head.register_forward_hook(nonfinite_output),
        model.candidate_route_head.register_forward_hook(nonfinite_output),
    ]
    try:
        output = _forward(model, _inputs(time=2))
    finally:
        for handle in handles:
            handle.remove()
    assert output["source_integrity_failed"].all()
    assert not output["expert_mask"].any()
    assert output["source_available"].all()
    assert (output["selected_source_code"] == -2).all()
    assert torch.equal(
        output["source_rr_bpm"], output["classical_rr_bpm"]
    )
    assert torch.equal(output["quality"], torch.zeros_like(output["quality"]))
    for name in (
        "expert_logits",
        "expert_probabilities",
        "expert_mean_bpm",
        "expert_scale_bpm",
        "source_rr_bpm",
        "source_scale_bpm",
        "selected_probability",
        "quality",
    ):
        assert torch.isfinite(output[name]).all(), name


def test_nonfinite_live_route_parameter_follows_declared_fallback_policy() -> None:
    model = _model()
    with torch.no_grad():
        model.anchor_route_head[-1].bias.fill_(float("nan"))
        model.candidate_route_head[-1].bias.fill_(float("nan"))
    output = _forward(model, _inputs(time=1))
    assert output["source_integrity_failed"].all()
    assert not output["expert_mask"].any()
    assert output["source_available"].all()
    assert (output["selected_source_code"] == -2).all()
    assert torch.isfinite(output["source_rr_bpm"]).all()
    assert torch.equal(output["quality"], torch.zeros_like(output["quality"]))


@pytest.mark.parametrize(
    "corruption",
    ("temporal_beta", "temporal_synapse", "graph_beta"),
)
def test_nonfinite_internal_spiking_state_is_sanitized_and_stream_continues(
    corruption: str,
) -> None:
    model = _model()
    if corruption == "temporal_beta":
        tensor = model.temporal.cells[0].beta_logit
    elif corruption == "temporal_synapse":
        tensor = model.temporal.synapses[0].weight
    else:
        tensor = model.graph[0].cell.beta_logit
    with torch.no_grad():
        tensor.reshape(-1)[0] = float("nan")

    output = _forward(model, _inputs(time=2))
    assert output["source_integrity_failed"].all()
    assert not output["execution_state_integrity"].all()
    assert not output["expert_mask"].any()
    assert output["source_available"].all()
    assert (output["selected_source_code"] == -2).all()
    assert torch.isfinite(output["source_rr_bpm"]).all()
    assert torch.equal(output["quality"], torch.zeros_like(output["quality"]))
    for membrane, adaptation in output["state"]:
        assert torch.isfinite(membrane).all()
        assert torch.isfinite(adaptation).all()

    next_values = _inputs(time=1)
    next_values["reset_mask"].zero_()
    continued = _forward(model, next_values, state=output["state"])
    assert torch.isfinite(continued["source_rr_bpm"]).all()
    for membrane, adaptation in continued["state"]:
        assert torch.isfinite(membrane).all()
        assert torch.isfinite(adaptation).all()


@pytest.mark.parametrize("overflow_head", ("anchor_value", "candidate_risk"))
def test_finite_checkpoint_arithmetic_overflow_removes_affected_expert(
    overflow_head: str,
) -> None:
    source = _model()
    head = (
        source.anchor_value_head
        if overflow_head == "anchor_value"
        else source.candidate_risk_head
    )
    sign = 1.0 if overflow_head == "anchor_value" else -1.0
    with torch.no_grad():
        head[0].weight.zero_()
        head[0].bias.fill_(1.0)
        head[-1].weight.fill_(sign * torch.finfo(torch.float32).max)
        head[-1].bias.zero_()
    state = source.state_dict()
    assert all(
        torch.isfinite(value).all()
        for value in state.values()
        if value.is_floating_point() or value.is_complex()
    )
    target = _model()
    target.load_state_dict(state)
    output = _forward(target, _inputs(time=1))
    assert output["source_integrity_failed"].all()
    if overflow_head == "anchor_value":
        assert not output["expert_mask"][..., 0].any()
    else:
        assert not output["expert_mask"][..., 1:].any()
    assert output["source_available"].all()
    assert torch.isfinite(output["source_rr_bpm"]).all()
    assert torch.isfinite(output["source_scale_bpm"]).all()
    assert torch.equal(output["quality"], torch.zeros_like(output["quality"]))


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("dropout", float("nan")),
        ("beta", float("inf")),
        ("adaptation_decay", 1.0),
        ("adaptation_strength", 0.0),
        ("adaptation_strength", -0.01),
        ("rr_min_bpm", 0.0),
        ("rr_max_bpm", float("inf")),
        ("route_temperature", 0.0),
        ("tail2_risk_weight", float("nan")),
        ("tail5_risk_weight", -0.01),
        ("candidate_residual_limit_bpm", -0.01),
        ("anchor_residual_limit_bpm", float("inf")),
        ("near_relation_tolerance_bpm", 0.0),
        ("ratio_relation_tolerance_bpm", float("nan")),
        ("edge_log_ratio_bandwidth", -1.0),
        ("factor_affinity_bandwidth_bpm", float("inf")),
    ),
)
def test_nonfinite_or_out_of_range_float_hyperparameters_fail_closed(
    name: str, value: float
) -> None:
    with pytest.raises(ValueError):
        AxisRiskRouterSNNV8R5(
            ordered_feature_names_semantic_sha256=(
                EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
            ),
            **{name: value},
        )


def test_nonfinite_state_and_empty_batch_or_time_fail_closed() -> None:
    model = _model()
    values = _inputs(time=2)
    state = model.initial_state(1)
    contaminated = tuple(
        (membrane.clone(), adaptation.clone())
        for membrane, adaptation in state
    )
    contaminated[0][0][0, 0] = float("nan")
    with pytest.raises(ValueError, match="state must be finite"):
        _forward(model, values, state=contaminated)

    half_state = tuple(
        (membrane.half(), adaptation.half())
        for membrane, adaptation in state
    )
    with pytest.raises(ValueError, match="canonical float32"):
        _forward(model, values, state=half_state)
    with pytest.raises(ValueError, match="locked to float32"):
        model.initial_state(1, dtype=torch.float16)

    float64 = {key: value.clone() for key, value in values.items()}
    float64["node_features"] = float64["node_features"].double()
    with pytest.raises(ValueError, match="canonical float32"):
        _forward(model, float64)

    empty_time = _slice(values, 0, 0)
    with pytest.raises(ValueError, match="empty batch/time"):
        _forward(model, empty_time)
    empty_batch = {key: value[:0].clone() for key, value in values.items()}
    with pytest.raises(ValueError, match="empty batch/time"):
        _forward(model, empty_batch)


def test_padded_quality_and_factor_outputs_are_exact_zero_after_training() -> None:
    model = _model()
    with torch.no_grad():
        model.factor_head.weight.fill_(0.1)
        model.factor_head.bias.fill_(1.7)
        model.quality_head.weight.fill_(0.1)
        model.quality_head.bias.fill_(2.3)
    values = _inputs(time=3)
    values["sequence_mask"][0, -1] = False
    with torch.no_grad():
        output = _forward(model, values)
    for name in ("factor_logits", "factor_probabilities"):
        padded = output[name][0, -1]
        assert torch.equal(padded, torch.zeros_like(padded)), name
    for name in ("quality_logit", "quality"):
        padded = output[name][0, -1]
        assert torch.equal(padded, torch.zeros_like(padded)), name
    assert output["factor_logits"][0, 0].abs().sum() > 0.0
    assert output["quality"][0, 0] > 0.0


def test_calibration_losses_normalize_experts_within_each_row() -> None:
    output = _forward(_model(), _inputs(time=2))
    target = torch.tensor([[18.0, 18.0]])
    valid = torch.ones_like(target, dtype=torch.bool)
    expert_mask = torch.tensor(
        [[[True, False, False, False, False], [True, True, True, True, False]]]
    )
    expert_means = torch.tensor(
        [[[19.0, 0.0, 0.0, 0.0, 0.0], [18.0, 19.0, 20.0, 21.0, 0.0]]]
    )
    predicted_abs = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    probability = expert_mask.float()
    probability /= probability.sum(dim=-1, keepdim=True)
    output.update(
        {
            "expert_mask": expert_mask,
            "expert_logits": torch.where(
                expert_mask, torch.zeros_like(probability), torch.full_like(probability, -1.0e4)
            ),
            "expert_probabilities": probability,
            "expert_mean_bpm": expert_means,
            "expert_scale_bpm": expert_mask.float(),
            "expert_expected_abs_error_bpm": predicted_abs,
            "expert_tail2_logits": torch.zeros_like(probability),
            "expert_tail5_logits": torch.zeros_like(probability),
            "training_soft_rr_bpm": (probability * expert_means).sum(dim=-1),
            "quality_logit": torch.zeros_like(target),
        }
    )
    _, components = soft_risk_routing_loss(output, target, valid)
    absolute_error = (expert_means - target.unsqueeze(-1)).abs()
    per_expert = torch.nn.functional.smooth_l1_loss(
        predicted_abs,
        torch.where(expert_mask, absolute_error, torch.zeros_like(absolute_error)),
        beta=0.25,
        reduction="none",
    )
    row_mean = (
        (per_expert * expert_mask).sum(dim=-1)
        / expert_mask.sum(dim=-1).clamp_min(1)
    ).mean()
    global_candidate_mean = (per_expert * expert_mask).sum() / expert_mask.sum()
    torch.testing.assert_close(
        components["expected_abs_error_calibration"], row_mean
    )
    assert not torch.isclose(row_mean, global_candidate_mean)


def test_quality_supervision_matches_deployment_argmax_not_soft_mean() -> None:
    output = dict(_forward(_model(), _inputs(time=1)))
    expert_mask = torch.tensor([[[True, True, False, False, False]]])
    probability = torch.tensor([[[0.5, 0.5, 0.0, 0.0, 0.0]]])
    means = torch.tensor([[[10.0, 30.0, 0.0, 0.0, 0.0]]])
    output.update(
        {
            "expert_mask": expert_mask,
            "expert_logits": torch.tensor(
                [[[2.0, 1.0, -1.0e4, -1.0e4, -1.0e4]]]
            ),
            "expert_probabilities": probability,
            "expert_mean_bpm": means,
            "expert_scale_bpm": expert_mask.float(),
            "expert_expected_abs_error_bpm": torch.zeros_like(means),
            "expert_tail2_logits": torch.zeros_like(means),
            "expert_tail5_logits": torch.zeros_like(means),
            # The soft mean equals the target exactly, but deployment argmax
            # chooses 10 bpm and is therefore catastrophically wrong.
            "training_soft_rr_bpm": torch.tensor([[20.0]]),
            "quality_logit": torch.tensor([[10.0]]),
            "sequence_mask": torch.ones((1, 1), dtype=torch.bool),
        }
    )
    _, components = soft_risk_routing_loss(
        output,
        torch.tensor([[20.0]]),
        torch.ones((1, 1), dtype=torch.bool),
    )
    assert components["quality_bce"] > 9.0


def test_loss_contract_is_strict_and_invalid_rows_remain_target_inert() -> None:
    output = _forward(_model(), _inputs(time=2))
    valid = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="floating point"):
        soft_risk_routing_loss(output, torch.tensor([[10, 20]]), valid)
    with pytest.raises(ValueError, match="valid target rows must be finite"):
        soft_risk_routing_loss(
            output, torch.tensor([[10.0, float("nan")]]), valid
        )
    with pytest.raises(ValueError, match="valid_mask must be boolean"):
        soft_risk_routing_loss(
            output, torch.tensor([[10.0, 20.0]]), valid.float()
        )
    with pytest.raises(ValueError, match="sample_weight"):
        soft_risk_routing_loss(
            output,
            torch.tensor([[10.0, 20.0]]),
            valid,
            sample_weight=torch.ones((1, 2), dtype=torch.int64),
        )
    corrupted = dict(output)
    corrupted["quality_logit"] = output["quality_logit"].clone()
    corrupted["quality_logit"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        soft_risk_routing_loss(
            corrupted, torch.tensor([[10.0, 20.0]]), valid
        )

    # A finite float64 sentinel must be replaced before float32 conversion on
    # invalid rows; otherwise it becomes inf and later creates inf*0 -> NaN.
    huge_invalid = torch.tensor([[1.0e300, 1.0e300]], dtype=torch.float64)
    inert_loss, inert_components = soft_risk_routing_loss(
        output, huge_invalid, torch.zeros_like(valid)
    )
    assert torch.equal(inert_loss, inert_loss.new_zeros(()))
    assert all(torch.isfinite(value) for value in inert_components.values())
    for out_of_range in (5.99, 45.01, 1.0e30):
        with pytest.raises(ValueError, match="locked RR bounds"):
            soft_risk_routing_loss(
                output,
                torch.tensor([[out_of_range, 20.0]], dtype=torch.float32),
                valid,
            )


def test_training_soft_rr_is_explicit_and_hard_outputs_are_loss_inert() -> None:
    model = _model()
    model.train()
    output = _forward(model, _inputs(time=2))
    expected_soft = (
        output["expert_probabilities"] * output["expert_mean_bpm"]
    ).sum(dim=-1)
    torch.testing.assert_close(output["training_soft_rr_bpm"], expected_soft)
    hard_names = {
        "selected_expert_index",
        "deployment_hard_selected_expert_index",
        "selected_source_code",
        "selected_probability",
        "source_rr_bpm",
        "source_scale_bpm",
        "source_available",
        "deployment_hard_rr_bpm",
        "deployment_hard_scale_bpm",
        "deployment_hard_available",
    }
    assert not hard_names & set(output)
    with pytest.raises(RuntimeError, match="forbidden while.*training"):
        model(
            _inputs(time=1)["node_features"],
            _inputs(time=1)["candidate_rr_bpm"],
            _inputs(time=1)["candidate_mask"],
            _inputs(time=1)["sequence_mask"],
            node_feature_availability=_inputs(time=1)["node_feature_availability"],
            joint_radar_mask=_inputs(time=1)["joint_radar_mask"],
            proposer_anchor_bpm=_inputs(time=1)["proposer_anchor_bpm"],
            proposer_anchor_std_bpm=_inputs(time=1)["proposer_anchor_std_bpm"],
            proposer_anchor_available=_inputs(time=1)["proposer_anchor_available"],
            classical_rr_bpm=_inputs(time=1)["classical_rr_bpm"],
            classical_rr_available=_inputs(time=1)["classical_rr_available"],
            reset_mask=_inputs(time=1)["reset_mask"],
            deployment_mode=True,
        )

    target = torch.tensor([[10.0, 20.0]])
    valid = torch.ones_like(target, dtype=torch.bool)
    original_loss, original_components = soft_risk_routing_loss(
        output, target, valid
    )
    corrupted_hard = dict(output)
    corrupted_hard["source_rr_bpm"] = torch.full_like(target, 9999.0)
    corrupted_hard["deployment_hard_rr_bpm"] = torch.full_like(target, -9999.0)
    corrupted_hard["selected_expert_index"] = torch.full_like(
        target, -999, dtype=torch.long
    )
    changed_loss, changed_components = soft_risk_routing_loss(
        corrupted_hard, target, valid
    )
    torch.testing.assert_close(changed_loss, original_loss, rtol=0.0, atol=0.0)
    for name in original_components:
        torch.testing.assert_close(
            changed_components[name], original_components[name], rtol=0.0, atol=0.0
        )


def test_candidate_coordinate_and_relation_settings_are_receipt_bound() -> None:
    model = AxisRiskRouterSNNV8R5(
        ordered_feature_names_semantic_sha256=(
            EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256
        ),
        rr_min_bpm=10.0,
        rr_max_bpm=40.0,
        near_relation_tolerance_bpm=0.6,
        ratio_relation_tolerance_bpm=0.9,
        edge_log_ratio_bandwidth=0.12,
        factor_affinity_bandwidth_bpm=0.8,
        tail2_risk_weight=1.7,
        tail5_risk_weight=4.2,
        candidate_residual_limit_bpm=0.9,
        anchor_residual_limit_bpm=10.0,
        dropout=0.0,
    )
    coordinate = model.encoder._candidate_coordinate_components(
        torch.tensor([10.0, 20.0, 40.0])
    )
    torch.testing.assert_close(coordinate[[0, 2], 0], torch.tensor([-1.0, 1.0]))
    receipt = model.layout_receipt()
    assert receipt["rr_min_bpm"] == 10.0
    assert receipt["rr_max_bpm"] == 40.0
    assert receipt["near_relation_tolerance_bpm"] == 0.6
    assert receipt["ratio_relation_tolerance_bpm"] == 0.9
    assert receipt["edge_log_ratio_bandwidth"] == 0.12
    assert receipt["factor_affinity_bandwidth_bpm"] == 0.8
    assert receipt["tail2_risk_weight"] == 1.7
    assert receipt["tail5_risk_weight"] == 4.2
    assert receipt["candidate_residual_limit_bpm"] == 0.9
    assert receipt["anchor_residual_limit_bpm"] == 10.0
    assert receipt["behavior_contract_sha256"] != _model().layout_receipt()[
        "behavior_contract_sha256"
    ]


def test_yaml_contract_matches_implemented_soft_hard_and_stride_semantics() -> None:
    path = Path(__file__).parents[1] / "configs" / "axis_risk_router_snn_v8r5.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["model"]["continuous_edge_evidence"] == (
        "directed_log_ratio_proximity"
    )
    assert payload["model"]["training_rr_output"] == "training_soft_rr_bpm"
    assert payload["model"]["deployment_rr_output"] == "deployment_hard_rr_bpm"
    assert payload["model"]["temporal_state_dtype"] == "float32"
    assert payload["model"]["checkpoint_state_contract"] == {
        "strict_load_only": True,
        "assign_false_only": True,
        "exact_key_shape_dtype_and_dense_layout_preflight": True,
        "private_tensor_snapshot_before_copy": True,
        "private_model_recursive_load_preflight": True,
        "transactional_live_rollback_on_unexpected_failure": True,
        "all_floating_and_complex_tensors_finite_before_copy": True,
        "persistent_and_nonpersistent_live_parameters_and_buffers_finite_before_copy": True,
        "source_layout_config_dependency_receipt": (
            "persistent_uint8_canonical_json"
        ),
        "non_tensor_runtime_structure_receipt": (
            "persistent_uint8_canonical_json"
        ),
        "behavior_contract_match": (
            "fresh_runtime_attributes_live_buffer_and_checkpoint_exact"
        ),
        "constructor_bound_runtime_behavior_attributes_immutable": True,
        "root_and_child_training_eval_mode_coherence_required": True,
        "state_dict_load_hooks_forbidden": True,
        "state_dict_export_revalidates_runtime_receipts": True,
        "route_temperature_match": "exact_expected_float32_value",
    }
    assert payload["model"]["proposal_config_absence_policy"] == (
        "module_import_fails_closed"
    )
    assert payload["model"]["source_execution_binding"] == {
        "scope": "initialization_time_disk_bytes_not_actual_loader_compiled_bytes",
        "binds_actual_loader_compiled_bytes": False,
        "terminal_training_authorization_blocker": (
            "external_launcher_executed_byte_closure_and_verifier_absent"
        ),
    }
    assert payload["input"]["per_feature_availability_forward_input_required"] is True
    assert payload["input"]["classical_rr_availability_forward_input_required"] is True
    assert payload["input"]["concrete_cache_contract_validator"] == (
        "validate_v8r5_cache_contract"
    )
    assert payload["routing"]["tail_probability_monotonicity_structural"] is True
    assert payload["routing"]["quality_head_supervision"].startswith(
        "deployment_argmax"
    )
    assert payload["routing"]["calibrated_risk_to_router_gradient"] == "stopped"
    assert payload["routing"]["nonfinite_source_policy"] == (
        "remove_nonfinite_expert_then_finite_classical_fallback_or_"
        "unavailable_quality_zero"
    )
    assert payload["routing"][
        "nonfinite_raw_head_preactivations_removed_before_nonlinear_transforms"
    ] is True
    assert payload["routing"]["nonfinite_graph_or_temporal_state_policy"] == (
        "remove_experts_zero_sanitize_stream_state_then_finite_classical_"
        "fallback_or_unavailable"
    )
    assert payload["routing"]["quality_zero_on_source_integrity_failure"] is True
    assert payload["loss"]["calibration_reduction"].startswith(
        "available_expert_mean_per_row"
    )
    stages = payload["planned_training_contract"]["stages"]
    assert all(isinstance(stage, dict) for stage in stages)
    assert sum(stage["optimizer_updates"] for stage in stages) == 1200
    model = _model()
    child_names = {name for name, _ in model.named_children()}
    for stage in stages:
        configured = set(model.configure_training_stage(stage["name"]))
        declared_trainable = set(stage["trainable_modules"])
        if declared_trainable == {"all"}:
            assert all(parameter.requires_grad for parameter in model.parameters())
            assert configured == child_names
            continue
        declared_frozen = set(stage["frozen_modules"])
        assert configured == declared_trainable
        assert declared_trainable | declared_frozen == child_names
        assert not declared_trainable & declared_frozen
        for name, module in model.named_children():
            gradients_enabled = {parameter.requires_grad for parameter in module.parameters()}
            assert gradients_enabled == {name in declared_trainable}, name

    # ``active_losses`` is an executable loss contract, not documentation.
    # Verify that every declared stage selects exactly the configured weighted
    # components and that the all-loss alias expands to every declared loss.
    output = _forward(_model(), _inputs(time=1))
    target = torch.tensor([[20.0]])
    valid = torch.ones_like(target, dtype=torch.bool)
    loss_names = (
        "soft_expected_deployment_cost",
        "equivalence_set_cross_entropy",
        "equivalence_value_smooth_l1",
        "expected_abs_error_calibration",
        "tail2_bce",
        "tail5_bce",
        "scale_nll",
        "quality_bce",
    )
    loss_weights = {name: float(payload["loss"][name]) for name in loss_names}
    for stage in stages:
        total, components = soft_risk_routing_loss(
            output,
            target,
            valid,
            training_stage=stage["name"],
        )
        active = stage["active_losses"]
        if active == ["all_declared_losses"]:
            active = list(loss_names)
        expected = sum(loss_weights[name] * components[name] for name in active)
        torch.testing.assert_close(total, expected)
    with pytest.raises(ValueError, match="unknown V8R5 loss training stage"):
        soft_risk_routing_loss(
            output,
            target,
            valid,
            training_stage="outer_test_adaptation",
        )
    with pytest.raises(ValueError, match="unknown V8R5 training stage"):
        model.configure_training_stage("outer_test_adaptation")
    stride = payload["evaluation_contract"]["stride_phase_evaluation"]
    assert stride["phase_count"] == 8
    assert stride["phase_offsets"] == list(range(8))
