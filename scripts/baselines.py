#!/usr/bin/env python3
"""Leakage-audited grouped baselines for the cached UWB RR dataset.

The script deliberately treats a physical identity as the generalisation
unit.  Every valid-reference window from one identity is held out in exactly
one of six folds.  Extra Trees sees only the radar-derived cache vector and
strictly causal history of cached radar estimates; reference RR, reference
quality, validity, observability, and error columns are never model inputs.

Outputs include row-aligned OOF NPZ/CSV predictions, an integrity/provenance
record, and identity-aware metrics with clustered bootstrap confidence
intervals and risk/coverage curves.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_rr.cache import append_causal_history_features, load_feature_cache  # noqa: E402
from snn_rr.metrics import grouped_oof_metrics, risk_coverage_curve  # noqa: E402


RADAR_HISTORY_COLUMNS = (
    "session_id",
    "window_number",
    "classical_rr_bpm",
    "classical_confidence",
    "radar_peak_spread_bpm",
)

# These columns may be exported as labels/audit context, but must never enter
# ``build_radar_feature_matrix``.  Keeping the list explicit makes a future
# schema change fail review instead of silently introducing target leakage.
PROHIBITED_MODEL_COLUMNS = (
    "rr_bpm",
    "rr_spectral_bpm",
    "rr_phase_bpm",
    "rr_events_bpm",
    "reference_valid",
    "reference_quality",
    "reference_sigma_bpm",
    "classical_error_bpm",
    "radar_observable",
    "classical_acceptable_within_2bpm",
)

DEFAULT_COVERAGES = (1.0, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.20)


@dataclass(frozen=True, slots=True)
class ExtraTreesSpec:
    """One member of the required fold-level tree ensemble."""

    name: str
    min_samples_leaf: int
    max_features: float
    seed_offset: int


DEFAULT_TREE_SPECS = (
    ExtraTreesSpec("extratrees_leaf2_mf1p0", 2, 1.0, 101),
    ExtraTreesSpec("extratrees_leaf5_mf0p8", 5, 0.8, 503),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_radar_feature_matrix(
    aux: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Return current radar features plus strictly causal radar history.

    The history builder has a narrow allow-list in :mod:`snn_rr.cache` and
    reads only ``RADAR_HISTORY_COLUMNS``.  This wrapper checks that contract,
    generates stable feature names, and normalises non-finite radar cache
    values without fitting a scaler.  Trees do not need feature scaling.
    """

    missing = sorted(set(RADAR_HISTORY_COLUMNS) - set(metadata.columns))
    if missing:
        raise KeyError(f"metadata missing radar-history columns: {missing}")
    aux_array = np.asarray(aux)
    if aux_array.ndim != 2 or len(aux_array) != len(metadata):
        raise ValueError("aux must be a 2-D matrix with one row per metadata row")
    augmented, history_names = append_causal_history_features(aux_array, metadata)
    feature_names = [f"cached_radar_aux_{index:04d}" for index in range(aux_array.shape[1])]
    feature_names.extend(history_names)
    prohibited = set(feature_names) & set(PROHIBITED_MODEL_COLUMNS)
    if prohibited:  # defensive: currently impossible unless names are edited
        raise RuntimeError(f"prohibited label features selected: {sorted(prohibited)}")
    matrix = np.nan_to_num(
        np.asarray(augmented, dtype=np.float32),
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=np.finfo(np.float32).min,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("radar feature matrix contains non-finite values")
    return matrix, feature_names


def identity_balanced_weights(identities: Sequence[str] | np.ndarray) -> np.ndarray:
    """Assign equal total fitting mass to every training identity."""

    groups = np.asarray(identities, dtype=str)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("identities must be a non-empty vector")
    weights = np.empty(len(groups), dtype=np.float64)
    for identity in np.unique(groups):
        selected = groups == identity
        weights[selected] = 1.0 / int(selected.sum())
    weights /= weights.mean()
    return weights


def make_grouped_folds(
    identities: Sequence[str] | np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign each identity to one deterministic OOF test fold."""

    groups = np.asarray(identities, dtype=str)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("identities must be a non-empty vector")
    unique_groups = np.unique(groups)
    if n_splits < 2 or len(unique_groups) < n_splits:
        raise ValueError("n_splits must be between 2 and the identity count")
    try:
        splitter = GroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(seed),
        )
    except TypeError:  # pragma: no cover - compatibility for old sklearn
        splitter = GroupKFold(n_splits=n_splits)

    fold_ids = np.full(len(groups), -1, dtype=np.int16)
    identity_to_fold: dict[str, int] = {}
    dummy = np.zeros(len(groups), dtype=np.uint8)
    for fold, (_, test_index) in enumerate(splitter.split(dummy, groups=groups)):
        fold_ids[test_index] = fold
        for identity in np.unique(groups[test_index]):
            if identity in identity_to_fold:
                raise RuntimeError(f"identity {identity!r} assigned to multiple folds")
            identity_to_fold[str(identity)] = fold

    if np.any(fold_ids < 0) or set(identity_to_fold) != set(unique_groups):
        raise RuntimeError("grouped splitter did not assign every row and identity")
    for identity in unique_groups:
        if len(np.unique(fold_ids[groups == identity])) != 1:
            raise RuntimeError(f"identity {identity!r} spans multiple test folds")
    return fold_ids, identity_to_fold


def fold_integrity_report(
    identities: Sequence[str] | np.ndarray,
    fold_ids: Sequence[int] | np.ndarray,
    *,
    n_splits: int,
) -> dict[str, Any]:
    """Validate and describe identity-disjoint OOF coverage."""

    groups = np.asarray(identities, dtype=str)
    folds = np.asarray(fold_ids, dtype=int)
    if groups.shape != folds.shape or groups.ndim != 1:
        raise ValueError("identities and fold_ids must be equal-length vectors")
    expected = set(range(n_splits))
    observed = set(np.unique(folds).tolist())
    if observed != expected:
        raise ValueError(f"expected folds {sorted(expected)}, observed {sorted(observed)}")
    identity_folds = {
        identity: np.unique(folds[groups == identity]).tolist()
        for identity in np.unique(groups)
    }
    violations = {key: value for key, value in identity_folds.items() if len(value) != 1}
    if violations:
        raise ValueError(f"identities span multiple folds: {violations}")

    per_fold: dict[str, Any] = {}
    for fold in range(n_splits):
        test = folds == fold
        train = ~test
        test_identities = sorted(np.unique(groups[test]).tolist())
        train_identities = sorted(np.unique(groups[train]).tolist())
        overlap = sorted(set(test_identities) & set(train_identities))
        if overlap:
            raise ValueError(f"fold {fold} train/test identity overlap: {overlap}")
        per_fold[str(fold)] = {
            "train_rows": int(train.sum()),
            "test_rows": int(test.sum()),
            "train_identities": train_identities,
            "test_identities": test_identities,
            "identity_overlap": overlap,
        }
    return {
        "passed": True,
        "n_rows": int(len(groups)),
        "n_identities": int(len(np.unique(groups))),
        "n_splits": int(n_splits),
        "every_row_predicted_once": True,
        "every_identity_in_one_test_fold": True,
        "per_fold": per_fold,
    }


def tree_prediction_distribution(
    model: ExtraTreesRegressor,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-row tree mean/std and the underlying tree predictions."""

    if not model.estimators_:
        raise ValueError("ExtraTreesRegressor must be fitted")
    predictions = np.stack(
        [tree.predict(features) for tree in model.estimators_], axis=0
    ).astype(np.float32)
    return (
        predictions.mean(axis=0, dtype=np.float64).astype(np.float32),
        predictions.std(axis=0, dtype=np.float64).astype(np.float32),
        predictions,
    )


def combine_tree_distributions(
    tree_prediction_sets: Iterable[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine member trees exactly, preserving within/between-model spread."""

    arrays = [np.asarray(values, dtype=np.float32) for values in tree_prediction_sets]
    if not arrays or any(array.ndim != 2 for array in arrays):
        raise ValueError("tree prediction sets must be a non-empty list of 2-D arrays")
    row_counts = {array.shape[1] for array in arrays}
    if len(row_counts) != 1:
        raise ValueError("tree prediction sets must have the same row count")
    combined = np.concatenate(arrays, axis=0)
    return (
        combined.mean(axis=0, dtype=np.float64).astype(np.float32),
        combined.std(axis=0, dtype=np.float64).astype(np.float32),
    )


def _summarize_method(
    target: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    identities: np.ndarray,
    fold_ids: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    coverages: Iterable[float],
) -> dict[str, Any]:
    if not (
        np.isfinite(target).all()
        and np.isfinite(prediction).all()
        and np.isfinite(uncertainty).all()
    ):
        raise ValueError("metric inputs contain non-finite values")
    summary = grouped_oof_metrics(
        target,
        prediction,
        identities,
        fold_ids=fold_ids,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    summary["risk_coverage"] = risk_coverage_curve(
        target,
        prediction,
        uncertainty,
        coverages=coverages,
        identities=identities,
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest_hash_before = _sha256(manifest_path)
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cache = load_feature_cache(cache_dir)
    metadata = cache.metadata.copy()
    required = {
        "session_id",
        "window_number",
        "identity",
        "reference_valid",
        "rr_bpm",
        "classical_rr_bpm",
        "classical_confidence",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise KeyError(f"cache metadata missing required columns: {missing}")

    features, feature_names = build_radar_feature_matrix(cache.aux, metadata)
    valid_mask = metadata["reference_valid"].to_numpy(dtype=bool)
    valid_index = np.flatnonzero(valid_mask).astype(np.int64)
    if not len(valid_index):
        raise ValueError("cache contains no valid reference windows")
    target = pd.to_numeric(metadata.loc[valid_mask, "rr_bpm"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    identities = metadata.loc[valid_mask, "identity"].astype(str).to_numpy()
    if not np.isfinite(target).all():
        raise ValueError("valid reference targets contain non-finite values")

    fold_ids, identity_to_fold = make_grouped_folds(
        identities,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    integrity = fold_integrity_report(
        identities,
        fold_ids,
        n_splits=args.n_splits,
    )

    method_names = [spec.name for spec in DEFAULT_TREE_SPECS]
    method_names.append("extratrees_ensemble")
    predictions = {
        name: np.full(len(valid_index), np.nan, dtype=np.float32) for name in method_names
    }
    uncertainties = {
        name: np.full(len(valid_index), np.nan, dtype=np.float32) for name in method_names
    }
    fold_diagnostics: dict[str, Any] = {}

    for fold in range(args.n_splits):
        train_local = np.flatnonzero(fold_ids != fold)
        test_local = np.flatnonzero(fold_ids == fold)
        train_global = valid_index[train_local]
        test_global = valid_index[test_local]
        train_identities = identities[train_local]
        test_identities = identities[test_local]
        overlap = sorted(set(train_identities) & set(test_identities))
        if overlap:
            raise RuntimeError(f"fold {fold} identity leakage: {overlap}")
        sample_weight = identity_balanced_weights(train_identities)

        tree_sets: list[np.ndarray] = []
        member_diagnostics: dict[str, Any] = {}
        for spec in DEFAULT_TREE_SPECS:
            model = ExtraTreesRegressor(
                n_estimators=args.n_estimators,
                criterion="squared_error",
                min_samples_leaf=spec.min_samples_leaf,
                max_features=spec.max_features,
                max_depth=None,
                bootstrap=False,
                random_state=args.seed + spec.seed_offset + fold * 1009,
                n_jobs=args.n_jobs,
            )
            model.fit(
                features[train_global],
                target[train_local],
                sample_weight=sample_weight,
            )
            mean, standard_deviation, tree_predictions = tree_prediction_distribution(
                model, features[test_global]
            )
            predictions[spec.name][test_local] = mean
            uncertainties[spec.name][test_local] = standard_deviation
            tree_sets.append(tree_predictions)
            member_diagnostics[spec.name] = {
                "spec": asdict(spec),
                "n_trees": int(len(model.estimators_)),
                "tree_prediction_std_mean_bpm": float(standard_deviation.mean()),
                "tree_prediction_std_p95_bpm": float(
                    np.quantile(standard_deviation, 0.95)
                ),
            }

        ensemble_mean, ensemble_std = combine_tree_distributions(tree_sets)
        predictions["extratrees_ensemble"][test_local] = ensemble_mean
        uncertainties["extratrees_ensemble"][test_local] = ensemble_std
        fold_diagnostics[str(fold)] = {
            "train_rows": int(len(train_local)),
            "test_rows": int(len(test_local)),
            "train_identities": sorted(np.unique(train_identities).tolist()),
            "test_identities": sorted(np.unique(test_identities).tolist()),
            "identity_overlap": overlap,
            "sample_weight_mean": float(sample_weight.mean()),
            "sample_weight_identity_mass": {
                identity: float(sample_weight[train_identities == identity].sum())
                for identity in np.unique(train_identities)
            },
            "members": member_diagnostics,
            "ensemble_tree_count": int(sum(values.shape[0] for values in tree_sets)),
            "ensemble_tree_prediction_std_mean_bpm": float(ensemble_std.mean()),
        }
        _write_json(output_dir / f"fold_{fold}.json", fold_diagnostics[str(fold)])
        print(
            f"fold {fold}: train={len(train_local)} test={len(test_local)} "
            f"identities={','.join(sorted(np.unique(test_identities)))}",
            flush=True,
        )

    for name in method_names:
        if not (np.isfinite(predictions[name]).all() and np.isfinite(uncertainties[name]).all()):
            raise RuntimeError(f"{name} did not produce exactly one finite OOF result per row")

    classical_prediction = pd.to_numeric(
        metadata.loc[valid_mask, "classical_rr_bpm"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    classical_confidence = pd.to_numeric(
        metadata.loc[valid_mask, "classical_confidence"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    if not (np.isfinite(classical_prediction).all() and np.isfinite(classical_confidence).all()):
        raise ValueError("cached classical predictions/confidence contain non-finite values")
    predictions = {"cached_classical": classical_prediction, **predictions}
    uncertainties = {
        "cached_classical": 1.0 - np.clip(classical_confidence, 0.0, 1.0),
        **uncertainties,
    }

    metrics: dict[str, Any] = {}
    for method_index, name in enumerate(predictions):
        metrics[name] = _summarize_method(
            target,
            predictions[name],
            uncertainties[name],
            identities,
            fold_ids,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed + 10000 + method_index,
            coverages=args.coverages,
        )

    selected_metadata_columns = [
        column
        for column in (
            "session_id",
            "session_number",
            "identity",
            "protocol",
            "window_number",
            "window_start_s",
            "window_end_s",
        )
        if column in metadata.columns
    ]
    oof_frame = metadata.loc[valid_mask, selected_metadata_columns].reset_index(drop=True)
    oof_frame.insert(0, "cache_row_index", valid_index)
    oof_frame["fold"] = fold_ids
    oof_frame["target_rr_bpm"] = target
    for name in predictions:
        oof_frame[f"{name}_prediction_bpm"] = predictions[name]
        oof_frame[f"{name}_uncertainty"] = uncertainties[name]
        oof_frame[f"{name}_absolute_error_bpm"] = np.abs(predictions[name] - target)
    csv_temporary = output_dir / "oof_predictions.csv.tmp"
    oof_frame.to_csv(csv_temporary, index=False, float_format="%.8g")
    csv_temporary.replace(output_dir / "oof_predictions.csv")

    npz_values: dict[str, np.ndarray] = {
        "cache_row_index": valid_index,
        "fold": fold_ids,
        "target_rr_bpm": target,
        "identity": identities.astype("U"),
        "session_id": metadata.loc[valid_mask, "session_id"].astype(str).to_numpy(dtype="U"),
    }
    for name in predictions:
        npz_values[f"{name}_prediction_bpm"] = predictions[name]
        npz_values[f"{name}_uncertainty"] = uncertainties[name]
    npz_temporary = output_dir / "oof_predictions.npz.tmp"
    with npz_temporary.open("wb") as handle:
        np.savez_compressed(handle, **npz_values)
    npz_temporary.replace(output_dir / "oof_predictions.npz")

    manifest_hash_after = _sha256(manifest_path)
    if manifest_hash_after != manifest_hash_before:
        raise RuntimeError("cache manifest changed during baseline execution")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "cache_manifest_sha256": manifest_hash_before,
        "cache_config_sha256": cache_manifest.get("config_sha256"),
        "cache_pipeline_sha256": cache_manifest.get("pipeline_sha256"),
        "script_sha256": _sha256(Path(__file__)),
        "cache_rows": int(len(metadata)),
        "valid_reference_rows": int(len(valid_index)),
        "identities": int(len(np.unique(identities))),
        "sessions": int(metadata["session_id"].nunique()),
        "base_aux_features": int(cache.aux.shape[1]),
        "causal_history_features": int(len(feature_names) - cache.aux.shape[1]),
        "total_model_features": int(len(feature_names)),
        "model_feature_sources": {
            "cached_aux": "radar-only spectra/statistics emitted by build_features.py",
            "causal_history_metadata_allowlist": list(RADAR_HISTORY_COLUMNS),
            "prohibited_columns": list(PROHIBITED_MODEL_COLUMNS),
            "reference_or_label_features_used": False,
            "scaler_fitted": False,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    run_config = {
        "seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "n_estimators_per_member_per_fold": int(args.n_estimators),
        "n_jobs": int(args.n_jobs),
        "bootstrap_samples": int(args.bootstrap_samples),
        "coverages": [float(value) for value in args.coverages],
        "tree_specs": [asdict(spec) for spec in DEFAULT_TREE_SPECS],
        "identity_to_fold": identity_to_fold,
        "feature_names": feature_names,
    }
    report = {
        "provenance": provenance,
        "run_config": run_config,
        "integrity": integrity,
        "fold_diagnostics": fold_diagnostics,
        "methods": metrics,
    }
    _write_json(output_dir / "fold_assignments.json", identity_to_fold)
    _write_json(output_dir / "run_config.json", run_config)
    _write_json(output_dir / "metrics.json", report)
    print(json.dumps({name: values["overall"] for name, values in metrics.items()}, indent=2))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "artifacts/cache/rf32s"))
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "artifacts/baselines/final")
    )
    parser.add_argument("--n-splits", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--coverages",
        type=float,
        nargs="+",
        default=list(DEFAULT_COVERAGES),
    )
    args = parser.parse_args(argv)
    if args.n_splits < 2:
        parser.error("--n-splits must be at least 2")
    if args.n_estimators <= 0:
        parser.error("--n-estimators must be positive")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if any(not 0 < value <= 1 for value in args.coverages):
        parser.error("--coverages values must be in (0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
