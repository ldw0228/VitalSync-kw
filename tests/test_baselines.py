from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "baselines.py"
_SPEC = importlib.util.spec_from_file_location("snn_rr_baselines_script", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_BASELINES = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASELINES
_SPEC.loader.exec_module(_BASELINES)


build_radar_feature_matrix = _BASELINES.build_radar_feature_matrix
combine_tree_distributions = _BASELINES.combine_tree_distributions
fold_integrity_report = _BASELINES.fold_integrity_report
identity_balanced_weights = _BASELINES.identity_balanced_weights
make_grouped_folds = _BASELINES.make_grouped_folds
tree_prediction_distribution = _BASELINES.tree_prediction_distribution


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "window_number": [0, 1, 2, 0, 1, 2],
            "classical_rr_bpm": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            "classical_confidence": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "radar_peak_spread_bpm": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "rr_bpm": [8.0, 9.0, 10.0, 18.0, 19.0, 20.0],
            "reference_valid": [True, True, False, True, False, True],
            "reference_quality": [0.9, 0.8, 0.1, 0.7, 0.1, 0.8],
            "radar_observable": [True, False, False, True, False, True],
            "classical_error_bpm": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )


def test_radar_feature_contract_is_label_independent() -> None:
    aux = np.arange(24, dtype=np.float32).reshape(6, 4)
    metadata = _metadata()
    first, names = build_radar_feature_matrix(aux, metadata)

    changed = metadata.copy()
    changed["rr_bpm"] = np.linspace(-1e6, 1e6, len(changed))
    changed["reference_valid"] = ~changed["reference_valid"]
    changed["reference_quality"] = 0.0
    changed["radar_observable"] = ~changed["radar_observable"]
    changed["classical_error_bpm"] = 12345.0
    second, second_names = build_radar_feature_matrix(aux, changed)

    np.testing.assert_array_equal(first, second)
    assert names == second_names
    assert names[:4] == [f"cached_radar_aux_{index:04d}" for index in range(4)]
    assert not set(names) & set(_BASELINES.PROHIBITED_MODEL_COLUMNS)


def test_identity_folds_are_complete_and_disjoint() -> None:
    identities = np.repeat(
        [f"P{index:02d}" for index in range(18)],
        np.arange(2, 20),
    )
    fold_ids, assignment = make_grouped_folds(
        identities, n_splits=6, seed=20260827
    )
    repeated, repeated_assignment = make_grouped_folds(
        identities, n_splits=6, seed=20260827
    )
    np.testing.assert_array_equal(fold_ids, repeated)
    assert assignment == repeated_assignment
    assert set(fold_ids) == set(range(6))
    assert set(assignment) == set(np.unique(identities))
    report = fold_integrity_report(identities, fold_ids, n_splits=6)
    assert report["passed"]
    assert report["every_identity_in_one_test_fold"]
    for identity in np.unique(identities):
        assert len(np.unique(fold_ids[identities == identity])) == 1
    for row in report["per_fold"].values():
        assert row["identity_overlap"] == []


def test_identity_balanced_weights_equal_total_subject_mass() -> None:
    identities = np.asarray(["A", "A", "A", "A", "B", "B", "C"])
    weights = identity_balanced_weights(identities)
    masses = [weights[identities == identity].sum() for identity in ("A", "B", "C")]
    np.testing.assert_allclose(masses, masses[0])
    assert np.isclose(weights.mean(), 1.0)
    assert weights[identities == "C"][0] > weights[identities == "A"][0]


def test_tree_distribution_and_combined_moments_match_explicit_trees() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(36, 5)).astype(np.float32)
    target = (2.0 * features[:, 0] - features[:, 1]).astype(np.float32)
    first = ExtraTreesRegressor(n_estimators=5, random_state=1, min_samples_leaf=2)
    second = ExtraTreesRegressor(n_estimators=7, random_state=2, min_samples_leaf=3)
    first.fit(features[:28], target[:28])
    second.fit(features[:28], target[:28])

    mean, std, first_trees = tree_prediction_distribution(first, features[28:])
    _, _, second_trees = tree_prediction_distribution(second, features[28:])
    explicit_first = np.stack(
        [tree.predict(features[28:]) for tree in first.estimators_], axis=0
    )
    np.testing.assert_allclose(mean, explicit_first.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(std, explicit_first.std(axis=0), rtol=1e-6)

    combined_mean, combined_std = combine_tree_distributions(
        [first_trees, second_trees]
    )
    explicit_combined = np.concatenate([first_trees, second_trees], axis=0)
    np.testing.assert_allclose(combined_mean, explicit_combined.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(combined_std, explicit_combined.std(axis=0), rtol=1e-6)

