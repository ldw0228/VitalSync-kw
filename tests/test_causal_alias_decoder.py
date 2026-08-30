from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "causal_alias_decoder.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "snn_rr_causal_alias_decoder_script", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
DECODER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = DECODER
_SPEC.loader.exec_module(DECODER)


def _bundle(
    prediction: np.ndarray,
    target: np.ndarray,
    posterior: np.ndarray,
) -> DECODER.PredictionBundle:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    count = len(prediction)
    return DECODER.PredictionBundle(
        index=np.arange(count, dtype=np.int64),
        target=target,
        prediction=prediction,
        rr_std=np.linspace(0.2, 0.5, count, dtype=np.float32),
        uncertainty=np.linspace(0.1, 0.4, count, dtype=np.float32),
        quality=np.linspace(0.9, 0.6, count, dtype=np.float32),
        observable=np.ones(count, dtype=bool),
        reference_valid=np.ones(count, dtype=bool),
        spike_rate=np.zeros(count, dtype=np.float32),
        radar_weights=np.full((count, 3), 1.0 / 3.0, dtype=np.float32),
        map_prediction=prediction + 0.25,
        posterior_entropy=np.linspace(0.3, 0.7, count, dtype=np.float32),
        posterior_probability=np.asarray(posterior, dtype=np.float32),
    )


def test_candidate_matrix_and_alias_labels_respect_range_and_confidence() -> None:
    classical = np.array([10.0, 8.0, 7.0, 20.0, 2.0])
    target = np.array([10.4, 24.2, 44.0, 42.5, 8.0])

    candidates = DECODER.candidate_matrix(classical)
    alias, confident, best = DECODER.alias_targets(
        target,
        classical,
        tolerance_bpm=2.0,
        rr_range=(6.0, 45.0),
    )

    np.testing.assert_allclose(
        candidates,
        classical[:, None] * np.array([[1.0, 2.0, 3.0, 4.0]]),
    )
    assert best.tolist() == [0, 2, 3, 1, 3]
    assert alias.tolist() == [False, True, True, True, True]
    assert confident.tolist() == [True, True, False, False, True]

    with pytest.raises(ValueError, match="one-dimensional"):
        DECODER.candidate_matrix(classical[:, None])


def test_posterior_and_structured_history_features_are_target_free() -> None:
    count = 3
    posterior = np.zeros((count, 40), dtype=np.float32)
    # The 40-bin grid is exactly 6..45 BPM.  Row zero therefore gives all
    # posterior support to the x2 candidate of its 10 BPM classical estimate.
    posterior[0, 20 - 6] = 1.0
    posterior[1, 18 - 6] = 1.0
    posterior[2, 24 - 6] = 1.0
    target = np.array([11.0, 19.0, 24.0], dtype=np.float32)
    first = _bundle(np.array([10.5, 18.0, 25.0]), target, posterior)
    second = _bundle(np.array([11.0, 18.5, 24.0]), target, posterior)

    metadata = pd.DataFrame(
        {
            "classical_rr_bpm": [10.0, 9.0, 8.0],
            "classical_confidence": [0.9, 0.7, 0.4],
            "radar_peak_spread_bpm": [0.2, 0.3, 0.4],
            "radar_peak_1_bpm": [10.0, 9.0, 8.0],
            "radar_peak_2_bpm": [10.5, 9.5, 8.5],
            "radar_peak_3_bpm": [9.5, 8.5, 7.5],
            # Deliberately present reference-like columns must not enter the
            # deployable feature construction path.
            "reference_rr_bpm": [11.0, 19.0, 24.0],
            "target_rr_bpm": [11.0, 19.0, 24.0],
        }
    )
    history_names = tuple(DECODER.HISTORY_FEATURES)
    structured_prefix = np.arange(count * 5, dtype=np.float32).reshape(count, 5)
    causal_history = (
        np.arange(count * len(history_names), dtype=np.float32).reshape(
            count, len(history_names)
        )
        / 10.0
    )
    causal_auxiliary = np.column_stack([structured_prefix, causal_history])

    features, names, context = DECODER.build_alias_features(
        first,
        second,
        metadata,
        causal_auxiliary,
        history_names,
        weight_a=0.25,
        rr_min=6.0,
        rr_max=45.0,
    )

    assert names[-len(history_names) :] == history_names
    np.testing.assert_allclose(features[:, -len(history_names) :], causal_history)
    assert context["posterior_support"].shape == (count, 4)
    assert context["posterior_support"][0, 1] == pytest.approx(1.0)
    assert context["posterior_support"][0, 0] == pytest.approx(0.0)
    assert "target" not in inspect.signature(DECODER.build_alias_features).parameters

    changed_target = np.array([44.0, 7.0, 35.0], dtype=np.float32)
    changed_metadata = metadata.copy()
    changed_metadata["reference_rr_bpm"] = changed_target
    changed_metadata["target_rr_bpm"] = changed_target[::-1]
    changed_features, changed_names, changed_context = DECODER.build_alias_features(
        _bundle(first.prediction, changed_target, posterior),
        _bundle(second.prediction, changed_target, posterior),
        changed_metadata,
        causal_auxiliary,
        history_names,
        weight_a=0.25,
        rr_min=6.0,
        rr_max=45.0,
    )

    assert changed_names == names
    np.testing.assert_array_equal(changed_features, features)
    for key in context:
        np.testing.assert_array_equal(changed_context[key], context[key])


def test_loio_crossfit_never_trains_on_the_held_out_identity() -> None:
    identities = np.repeat(np.array(["a", "b", "c", "d"]), 2)
    features = np.column_stack(
        [np.linspace(-1.5, 1.5, 8), np.tile([0.0, 1.0], 4)]
    )
    labels = np.tile(np.array([False, True]), 4)
    confident = np.ones(8, dtype=bool)
    classical = np.full(8, 10.0)
    target = np.where(labels, 30.0, 10.0)

    probability, multiplier, high_alias_prior, audit = DECODER.crossfit_alias_evidence(
        features,
        labels,
        confident,
        identities,
        target,
        classical,
        ("trend", "phase"),
        regularization_c=1.0,
    )

    assert probability.shape == multiplier.shape == high_alias_prior.shape == (8,)
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert len(audit) == 4
    for row in audit:
        assert row["held_out_identity"] not in row["train_identities"]
        assert set(row["train_identities"]) == {
            "a",
            "b",
            "c",
            "d",
        } - {row["held_out_identity"]}
        assert row["fit_rows"] == 6

    # Supervision belonging to identity a cannot affect its own cross-fitted
    # evidence or multiplier: that identity is excluded from both fits.
    changed_labels = labels.copy()
    changed_labels[identities == "a"] = ~changed_labels[identities == "a"]
    changed_target = target.copy()
    changed_target[identities == "a"] = np.array([45.0, 6.0])
    (
        changed_probability,
        changed_multiplier,
        changed_high_alias_prior,
        _,
    ) = DECODER.crossfit_alias_evidence(
        features,
        changed_labels,
        confident,
        identities,
        changed_target,
        classical,
        ("trend", "phase"),
        regularization_c=1.0,
    )
    held_out_a = identities == "a"
    np.testing.assert_allclose(
        changed_probability[held_out_a], probability[held_out_a], atol=1e-12
    )
    np.testing.assert_allclose(
        changed_multiplier[held_out_a], multiplier[held_out_a], atol=1e-12
    )
    np.testing.assert_allclose(
        changed_high_alias_prior[held_out_a],
        high_alias_prior[held_out_a],
        atol=1e-12,
    )


def test_causal_decode_is_prefix_invariant_and_has_no_future_dependence() -> None:
    base = np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    classical = np.array([8.0, 8.1, 8.2, 8.3, 8.4, 8.5])
    confidence = np.full(6, 0.9)
    probability = np.array([0.9, 0.8, 0.7, 0.1, 0.95, 0.2])
    uncertainty = np.linspace(0.1, 0.9, 6)
    session = np.full(6, "session-a")
    windows = np.arange(6)
    params = DECODER.DecoderParams(
        alias_threshold=0.3,
        alias_pull=0.8,
        state_decay=0.75,
        continuity=0.4,
        uncertainty_gain=0.5,
    )

    full = DECODER.causal_decode(
        base,
        classical,
        confidence,
        probability,
        uncertainty,
        session,
        windows,
        3.0,
        27.5,
        params,
    )
    prefix = DECODER.causal_decode(
        base[:4],
        classical[:4],
        confidence[:4],
        probability[:4],
        uncertainty[:4],
        session[:4],
        windows[:4],
        3.0,
        27.5,
        params,
    )
    for full_value, prefix_value in zip(full, prefix, strict=True):
        np.testing.assert_allclose(full_value[:4], prefix_value, atol=1e-12)

    future_base = base.copy()
    future_classical = classical.copy()
    future_probability = probability.copy()
    future_base[4:] = [44.0, 6.0]
    future_classical[4:] = [6.0, 15.0]
    future_probability[4:] = [0.0, 1.0]
    changed = DECODER.causal_decode(
        future_base,
        future_classical,
        confidence,
        future_probability,
        uncertainty,
        session,
        windows,
        3.0,
        27.5,
        params,
    )
    for original_value, changed_value in zip(full, changed, strict=True):
        np.testing.assert_allclose(original_value[:4], changed_value[:4], atol=1e-12)


def test_causal_decode_resets_state_and_continuity_on_session_or_gap() -> None:
    # Deliberately unsorted rows also exercise the decoder's stable
    # session/window ordering while retaining outputs in source-row order.
    sessions = np.array(["b", "a", "a", "b", "a"])
    windows = np.array([0, 0, 1, 2, 3])
    base = np.array([30.0, 10.0, 11.0, 31.0, 12.0])
    classical = np.array([8.0, 7.0, 7.0, 8.0, 7.0])
    confidence = np.ones(5)
    probability = np.array([0.2, 0.4, 0.8, 0.9, 0.6])
    uncertainty = np.ones(5)
    params = DECODER.DecoderParams(
        alias_threshold=0.1,
        alias_pull=0.8,
        state_decay=0.5,
        continuity=0.75,
        uncertainty_gain=0.0,
    )

    decoded, state, _ = DECODER.causal_decode(
        base,
        classical,
        confidence,
        probability,
        uncertainty,
        sessions,
        windows,
        3.0,
        27.5,
        params,
    )

    # a:0 -> a:1 is contiguous; a:3 is a gap.  b:0 -> b:2 is also a gap.
    np.testing.assert_allclose(state, [0.2, 0.4, 0.6, 0.9, 0.6])
    for row in (0, 3, 4):
        isolated, isolated_state, _ = DECODER.causal_decode(
            base[row : row + 1],
            classical[row : row + 1],
            confidence[row : row + 1],
            probability[row : row + 1],
            uncertainty[row : row + 1],
            sessions[row : row + 1],
            windows[row : row + 1],
            3.0,
            27.5,
            params,
        )
        assert decoded[row] == pytest.approx(isolated[0])
        assert state[row] == pytest.approx(isolated_state[0])


def test_apply_locked_decoder_rule_does_not_accept_or_use_targets() -> None:
    lock = {
        "alias_classifier": {
            "feature_names": ["posterior_ratio", "history_trend"],
            "mean": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "coefficient": [1.5, -0.25],
            "intercept": -0.1,
            "constant_probability": None,
            "regularization_c": 1.0,
            "fit_rows": 12,
            "positive_rows": 5,
        },
        "uncertainty_normalization": {"lower_q10": 0.1, "upper_q90": 0.9},
        "selected_params": {
            "enabled": True,
            "alias_threshold": 0.3,
            "alias_pull": 0.75,
            "state_decay": 0.5,
            "continuity": 0.25,
            "uncertainty_gain": 0.5,
            "confidence_scale": 0.2,
            "rr_min": 6.0,
            "rr_max": 45.0,
        },
        "alias_multiplier": 3.0,
        "high_alias_rr_prior_bpm": 27.5,
        "uncertainty_correction_scale_bpm": 2.0,
    }
    features = np.array([[-1.0, 0.2], [0.2, -0.1], [1.1, 0.3]])
    context = {
        "blend": np.array([12.0, 13.0, 14.0]),
        "classical_rr": np.array([7.0, 8.0, 9.0]),
        "classical_confidence": np.array([0.8, 0.9, 0.7]),
        "raw_uncertainty": np.array([0.2, 0.5, 0.8]),
    }
    sessions = np.array(["s", "s", "s"])
    windows = np.arange(3)

    result = DECODER.apply_decoder_lock(
        lock, features, context, sessions, windows
    )
    assert "target" not in inspect.signature(DECODER.apply_decoder_lock).parameters
    assert all(np.isfinite(value).all() for value in result)

    context_with_labels = {
        **context,
        "target": np.array([45.0, 6.0, 32.0]),
        "reference_rr_bpm": np.array([6.0, 45.0, 7.0]),
    }
    changed = DECODER.apply_decoder_lock(
        lock, features, context_with_labels, sessions, windows
    )
    for original_value, changed_value in zip(result, changed, strict=True):
        np.testing.assert_array_equal(original_value, changed_value)


def test_optional_alias_head_is_evidence_only_and_uses_locked_threshold() -> None:
    lock = {
        "alias_classifier": {
            "feature_names": ["unused_meta"],
            "mean": [0.0],
            "scale": [1.0],
            "coefficient": [0.0],
            "intercept": 0.0,
            "constant_probability": None,
            "regularization_c": 1.0,
            "fit_rows": 3,
            "positive_rows": 1,
        },
        "selected_evidence_source": "alias_head_probability",
        "uncertainty_normalization": {"lower_q10": 0.0, "upper_q90": 1.0},
        "selected_params": {
            "enabled": True,
            "alias_route": "high_alias_prior",
            "alias_threshold": 0.6,
            "alias_pull": 1.0,
            "state_decay": 0.0,
            "continuity": 0.0,
            "uncertainty_gain": 0.0,
            "confidence_scale": 0.1,
            "rr_min": 6.0,
            "rr_max": 45.0,
        },
        "alias_multiplier": 3.0,
        "high_alias_rr_prior_bpm": 28.0,
        "uncertainty_correction_scale_bpm": 1.0,
    }
    features = np.zeros((3, 1))
    context = {
        "blend": np.full(3, 14.0),
        "classical_rr": np.full(3, 8.0),
        "classical_confidence": np.ones(3),
        "raw_uncertainty": np.zeros(3),
        # A tempting RR prediction from the alias model is deliberately not
        # part of the accepted application context/API.
        "alias_model_rr_prediction": np.full(3, 44.0),
    }
    probability = np.array([0.2, 0.8, 1.0])

    prediction, _, used_probability, _, _ = DECODER.apply_decoder_lock(
        lock,
        features,
        context,
        np.array(["a", "b", "c"]),
        np.zeros(3, dtype=int),
        external_alias_probability=probability,
    )

    np.testing.assert_array_equal(used_probability, probability)
    assert prediction[0] == pytest.approx(14.0)
    assert 14.0 < prediction[1] < 28.0
    assert prediction[2] == pytest.approx(28.0)
    assert np.all(prediction < 44.0)


def test_high_alias_prior_uses_only_validation_targets_in_25_to_35_band() -> None:
    target = np.array([18.0, 19.0, 27.0, 29.0, 34.0])
    identities = np.array(["a", "a", "a", "b", "b"])
    selected_alias = np.ones(5, dtype=bool)

    prior = DECODER.fit_high_alias_prior(target, identities, selected_alias)

    assert 27.0 <= prior <= 34.0
    assert prior != pytest.approx(18.0)

    changed_low_alias_targets = target.copy()
    changed_low_alias_targets[:2] = [6.0, 24.0]
    assert DECODER.fit_high_alias_prior(
        changed_low_alias_targets, identities, selected_alias
    ) == pytest.approx(prior)


def test_optional_third_rr_component_is_validation_guarded() -> None:
    target = np.array([10.0, 11.0, 20.0, 21.0, 30.0, 31.0])
    identities = np.repeat(np.array(["a", "b", "c"]), 2)
    prediction_a = target + 1.0
    prediction_b = target + 2.0

    selected_weights, selected_prediction, selected_report = (
        DECODER.select_optional_third_component(
            target,
            prediction_a,
            prediction_b,
            target,
            identities,
            step=0.1,
            minimum_improvement=0.01,
        )
    )
    assert selected_report["third_component_selected"] is True
    assert selected_weights[2] > 0.0
    np.testing.assert_allclose(selected_prediction, target, atol=1e-12)

    rejected_weights, rejected_prediction, rejected_report = (
        DECODER.select_optional_third_component(
            target,
            prediction_a,
            prediction_b,
            target + 4.0,
            identities,
            step=0.1,
            minimum_improvement=0.01,
        )
    )
    assert rejected_report["third_component_selected"] is False
    assert rejected_weights[2] == pytest.approx(0.0)
    _, expected_two_way, _ = DECODER.select_blend(
        target, prediction_a, prediction_b, identities, step=0.1
    )
    np.testing.assert_allclose(rejected_prediction, expected_two_way, atol=1e-12)


def test_fold_split_audit_requires_nonempty_disjoint_identity_groups() -> None:
    valid = {
        "train_identities": ["a", "b"],
        "validation_identities": ["c"],
        "test_identities": ["d", "e"],
    }
    DECODER.audit_fold_split(valid)

    overlap = {
        **valid,
        "test_identities": ["b", "e"],
    }
    with pytest.raises(RuntimeError, match="not disjoint"):
        DECODER.audit_fold_split(overlap)

    empty = {
        **valid,
        "validation_identities": [],
    }
    with pytest.raises(ValueError, match="must be non-empty"):
        DECODER.audit_fold_split(empty)


def test_checkpoint_must_match_declared_run_signature(tmp_path: Path) -> None:
    path = tmp_path / "snn_best.pt"
    torch.save(
        {
            "model_type": "snn",
            "model_kwargs": {},
            "model_state": {},
            "fold": 0,
            "split": {},
            "aux_center": torch.zeros(1),
            "aux_scale": torch.ones(1),
            "run_signature": "checkpoint-signature",
        },
        path,
    )

    with pytest.raises(RuntimeError, match="signature mismatch"):
        DECODER._load_checkpoint(
            path,
            fold=0,
            expected_run_signature="run-config-signature",
        )


@pytest.mark.parametrize(
    ("static_supported", "decoder_supported", "expected"),
    [
        (False, False, ("reject", "validation_locked_two_way_blend")),
        (True, False, ("accept", "validation_locked_blend")),
        (False, True, ("accept", "validation_locked_causal_alias_decoder")),
        (True, True, ("accept", "validation_locked_causal_alias_decoder")),
    ],
)
def test_static_stack_can_be_accepted_independently_of_decoder(
    static_supported: bool,
    decoder_supported: bool,
    expected: tuple[str, str],
) -> None:
    assert DECODER.choose_deployment_variant(
        static_stack_supported=static_supported,
        causal_decoder_supported=decoder_supported,
    ) == expected
