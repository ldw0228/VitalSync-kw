#!/usr/bin/env python3
"""Reproduce the locked Stage-A harmonic-candidate separability audit.

This audit is deliberately narrower than a model-training run.  For each of
the two predeclared discovery exclusions (outer folds 3 and 4), it uses only
the remaining identities and obtains candidate-utility scores from grouped
inner out-of-fold fits.  The learned scorer receives verified, aggregate,
label-free physics features for the classical x1/x2/x3/x4 hypotheses.  The
frozen ensemble is never a learned input; it is read only to evaluate whether
a sparse correction would improve the exact fallback.

A failed scientific gate is an expected experiment outcome and therefore
still produces an atomic JSON report and exits zero.  Integrity failures
(source hash mismatch, row-binding failure, or identity leakage) fail closed
with a non-zero exit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "harmonic_factor_snn_v1"
FORMAT_VERSION = 1
FACTORS = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
OUTER_EXCLUSIONS = (3, 4)
INNER_SPLITS = 5

# These are the exact inputs on which the Stage-A decision was predeclared.
# A path override is useful when reproducing the repository elsewhere, but its
# content must still match the locked digest.
PINNED_SOURCE_SHA256 = {
    "physics_source": "ed9cad030e60ea2c83c212070d7035bf7b4eac34ec7a0d36696cbb8043b56382",
    "feature_manifest": "9ca8e5c868b45746e12d2f961cb0c1223d5d7450d2b731cf0aa13cee99fb18ff",
    "feature_map": "37a9c2a2bed70a06a5c913e2e3844c3ec2fc97554487afaff5a18f96235c13c8",
    "frozen_base_npz": "ab8c8fa03a9cf703319e32f57e8db05f6658c2bae59d2d1a6048ab81be4c772a",
    "alias_oof_csv": "a8e8c2a7a5ef992cbe1352f35f3c89792635e00687652cd6f3a1ce6635acc6da",
}

PREDECLARED_STAGE_A_REQUIREMENTS = {
    "action_auc_min": 0.8,
    "action_average_precision_min": 0.45,
    "factor_accuracy_gain_over_x1_prevalence_min": 0.05,
    "correction_precision_min": 0.8,
    "correction_recall_min": 0.2,
    "baseline_good_false_positive_fraction_max": 0.01,
    "estimated_macro_mae_gain_bpm_min": 0.1,
}

MODEL_PARAMETERS = {
    "objective": "reg:pseudohubererror",
    "n_estimators": 500,
    "max_depth": 2,
    "min_child_weight": 12,
    "learning_rate": 0.025,
    "subsample": 0.85,
    "colsample_bytree": 0.65,
    "reg_lambda": 30.0,
    "reg_alpha": 0.2,
    "tree_method": "hist",
}

RETROSPECTIVE_NOTICE = (
    "Grouped inner-OOF retrospective separability audit on an already observed "
    "cohort. Passing Stage A would authorize only the locked discovery training "
    "stage, never a commercial-performance claim."
)


@dataclass(frozen=True, slots=True)
class StageAThresholds:
    action_auc_min: float
    action_average_precision_min: float
    factor_accuracy_gain_over_x1_prevalence_min: float
    correction_precision_min: float
    correction_recall_min: float
    baseline_good_false_positive_fraction_max: float
    estimated_macro_mae_gain_bpm_min: float

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "StageAThresholds":
        try:
            recorded = contract["stage_gates"]["A_separability"]["required"]
        except (KeyError, TypeError) as error:
            raise RuntimeError("campaign contract lacks Stage-A requirements") from error
        if not isinstance(recorded, Mapping):
            raise RuntimeError("campaign Stage-A requirements must be a JSON object")
        missing = sorted(set(PREDECLARED_STAGE_A_REQUIREMENTS) - set(recorded))
        if missing:
            raise RuntimeError(f"campaign Stage-A requirements are incomplete: {missing}")
        values = {key: float(recorded[key]) for key in PREDECLARED_STAGE_A_REQUIREMENTS}
        for key, expected in PREDECLARED_STAGE_A_REQUIREMENTS.items():
            if values[key] != float(expected):
                raise RuntimeError(
                    f"predeclared Stage-A threshold changed for {key}: "
                    f"expected {expected}, found {values[key]}"
                )
        return cls(**values)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON source {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def strict_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return strict_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            strict_json(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verified_file(
    path: Path,
    expected_sha256: str,
    *,
    role: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required {role} source is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{role} SHA-256 mismatch: expected {expected_sha256}, found {actual}"
        )
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "verified": True,
    }


def _contract_binding(
    contract: Mapping[str, Any], key: str, *, project_root: Path
) -> tuple[Path, str]:
    try:
        binding = contract["immutable_population"][key]
        path = project_root / str(binding["path"])
        expected = str(binding["sha256"])
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"campaign contract lacks immutable binding {key}") from error
    if len(expected) != 64:
        raise RuntimeError(f"campaign binding {key} has an invalid SHA-256")
    return path, expected


def verify_sources(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = load_json(args.campaign_contract)
    if contract.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError(
            f"unexpected campaign_id: expected {CAMPAIGN_ID}, "
            f"found {contract.get('campaign_id')!r}"
        )
    feature_manifest = load_json(args.feature_manifest)
    records = [
        _verified_file(
            args.physics_source,
            PINNED_SOURCE_SHA256["physics_source"],
            role="physics_feature_layout_source",
        ),
        _verified_file(
            args.feature_manifest,
            PINNED_SOURCE_SHA256["feature_manifest"],
            role="physics_feature_manifest",
        ),
        _verified_file(
            args.feature_map,
            PINNED_SOURCE_SHA256["feature_map"],
            role="physics_candidate_feature_map",
        ),
        _verified_file(
            args.base_oof,
            PINNED_SOURCE_SHA256["frozen_base_npz"],
            role="frozen_base_oof_npz",
        ),
        _verified_file(
            args.alias_oof_csv,
            PINNED_SOURCE_SHA256["alias_oof_csv"],
            role="frozen_alias_snn_oof_csv_for_oracle_diagnostic",
        ),
    ]

    manifest_feature_sha = str(feature_manifest.get("sha256", ""))
    if manifest_feature_sha != PINNED_SOURCE_SHA256["feature_map"]:
        raise RuntimeError("feature manifest does not bind the pinned feature map")
    try:
        signature_payload = feature_manifest["signature_payload"]
        recorded_signature = str(feature_manifest["feature_signature"])
    except KeyError as error:
        raise RuntimeError("feature manifest lacks its content signature") from error
    recomputed_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if recomputed_signature != recorded_signature:
        raise RuntimeError("feature-manifest signature payload is not self-consistent")

    for key, role in (
        ("base_oof_csv", "campaign_frozen_base_oof_csv"),
        ("fold_assignments", "campaign_fold_assignments"),
        ("svd_all_window_manifest", "campaign_svd_all_window_manifest"),
    ):
        path, expected = _contract_binding(contract, key, project_root=PROJECT_ROOT)
        records.append(_verified_file(path, expected, role=role))

    svd_manifest_path, svd_manifest_sha = _contract_binding(
        contract, "svd_all_window_manifest", project_root=PROJECT_ROOT
    )
    if Path(args.svd_cache / "manifest.json").resolve() != svd_manifest_path.resolve():
        raise RuntimeError("requested SVD cache is not the campaign-bound SVD cache")
    payload_svd_sha = str(signature_payload.get("svd_manifest_sha256", ""))
    if payload_svd_sha != svd_manifest_sha:
        raise RuntimeError("feature map is not bound to the campaign SVD manifest")

    return (
        contract,
        feature_manifest,
        {
            "all_verified": True,
            "campaign_contract": {
                "path": str(args.campaign_contract.resolve()),
                "sha256": sha256_file(args.campaign_contract),
            },
            "feature_signature": recorded_signature,
            "files": records,
        },
    )


def _load_physics_source(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_harmonic_stage_a_physics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import verified physics source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def identity_weights(identities: np.ndarray) -> np.ndarray:
    values = np.asarray(identities, dtype=str)
    if values.ndim != 1 or not len(values):
        raise ValueError("identities must be a non-empty one-dimensional array")
    result = np.zeros(len(values), dtype=np.float64)
    names = np.unique(values)
    for name in names:
        selected = values == name
        result[selected] = len(values) / (len(names) * int(selected.sum()))
    return result


def regression_metrics(
    target: np.ndarray, prediction: np.ndarray, identities: np.ndarray
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    identities = np.asarray(identities, dtype=str)
    if target.ndim != 1 or prediction.shape != target.shape or identities.shape != target.shape:
        raise ValueError("target, prediction and identities must be aligned vectors")
    if not len(target) or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("metrics require non-empty finite target and prediction vectors")
    error = np.abs(prediction - target)
    names = np.unique(identities)
    return {
        "n": int(len(target)),
        "identity_count": int(len(names)),
        "mae": float(error.mean()),
        "macro_mae": float(
            np.mean([error[identities == name].mean() for name in names])
        ),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "within_2": float(np.mean(error <= 2.0)),
        "over_5": float(np.mean(error > 5.0)),
    }


def _oracle_metric_block(
    target: np.ndarray, prediction: np.ndarray, identities: np.ndarray
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    identities = np.asarray(identities, dtype=str)
    result: dict[str, Any] = {
        "full_coverage": regression_metrics(target, prediction, identities)
    }
    tail = (target >= 25.0) & (target <= 35.0)
    result["high_rr_25_35"] = (
        regression_metrics(target[tail], prediction[tail], identities[tail])
        if tail.any()
        else {
            "n": 0,
            "identity_count": 0,
            "mae": None,
            "macro_mae": None,
            "rmse": None,
            "within_2": None,
            "over_5": None,
        }
    )
    return result


def target_dependent_candidate_oracle(
    *,
    alias_oof_csv: Path,
    metadata: Any,
    base: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compute a non-deployable upper bound; never feed it into Stage-A gates."""

    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("pandas is required for the oracle alignment audit") from error

    frame = pd.read_csv(alias_oof_csv)
    required = {
        "cache_index",
        "fold",
        "session_id",
        "identity",
        "rr_bpm",
        "classical_rr_bpm",
        *{f"posterior_top{rank}_rr_bpm" for rank in range(1, 6)},
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"alias OOF CSV lacks oracle fields: {missing}")

    valid_index = np.asarray(base["index"], dtype=np.int64)
    alias_index = frame["cache_index"].to_numpy(dtype=np.int64)
    if len(frame) != len(valid_index):
        raise RuntimeError("alias OOF CSV denominator differs from frozen base")
    if not frame["cache_index"].is_unique:
        raise RuntimeError("alias OOF CSV contains duplicate cache_index values")
    # Exact order is intentional: accepting a set-equivalent reorder would make
    # candidate/target concatenation vulnerable to an unnoticed row join bug.
    if not np.array_equal(alias_index, valid_index):
        raise RuntimeError(
            "alias OOF CSV cache_index is not exactly aligned to frozen base order"
        )

    expected = metadata.iloc[valid_index]
    fold = np.asarray(base["fold"], dtype=np.int16)
    target = np.asarray(base["target"], dtype=np.float64)
    base_prediction = np.asarray(base["prediction"], dtype=np.float64)
    classical = expected["classical_rr_bpm"].to_numpy(dtype=np.float64)
    identities = expected["identity"].astype(str).to_numpy()
    sessions = expected["session_id"].astype(str).to_numpy()
    if not np.array_equal(frame["fold"].to_numpy(dtype=np.int16), fold):
        raise RuntimeError("alias OOF CSV fold is not exactly aligned to frozen base")
    if not np.array_equal(frame["identity"].astype(str).to_numpy(), identities):
        raise RuntimeError("alias OOF CSV identity is not exactly aligned to metadata")
    if not np.array_equal(frame["session_id"].astype(str).to_numpy(), sessions):
        raise RuntimeError("alias OOF CSV session is not exactly aligned to metadata")
    if not np.allclose(
        frame["rr_bpm"].to_numpy(dtype=np.float64),
        target,
        rtol=1e-7,
        atol=1e-6,
    ):
        raise RuntimeError("alias OOF CSV target is not aligned to frozen base")
    if not np.allclose(
        frame["classical_rr_bpm"].to_numpy(dtype=np.float64),
        classical,
        rtol=1e-7,
        atol=1e-6,
    ):
        raise RuntimeError("alias OOF CSV classical RR is not aligned to metadata")

    alias_candidates = frame[
        [f"posterior_top{rank}_rr_bpm" for rank in range(1, 6)]
    ].to_numpy(dtype=np.float64)
    candidates = np.column_stack(
        [
            base_prediction,
            classical[:, None] * FACTORS[None, :],
            alias_candidates,
        ]
    )
    if candidates.shape != (len(target), 10) or not np.isfinite(candidates).all():
        raise RuntimeError("oracle candidate bank is incomplete or non-finite")
    names = [
        "frozen_base",
        "classical_x1",
        "classical_x2",
        "classical_x3",
        "classical_x4",
        "alias_posterior_top1",
        "alias_posterior_top2",
        "alias_posterior_top3",
        "alias_posterior_top4",
        "alias_posterior_top5",
    ]
    selected = np.abs(candidates - target[:, None]).argmin(axis=1)
    oracle_prediction = candidates[np.arange(len(target)), selected]

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        local_selected = selected[mask]
        return {
            "rows": int(mask.sum()),
            "identity_count": int(len(np.unique(identities[mask]))),
            "metrics": _oracle_metric_block(
                target[mask], oracle_prediction[mask], identities[mask]
            ),
            "base_metrics": _oracle_metric_block(
                target[mask], base_prediction[mask], identities[mask]
            ),
            "selected_candidate_counts": {
                name: int(np.sum(local_selected == index))
                for index, name in enumerate(names)
            },
        }

    folds = {
        str(outer_fold): summarize(fold == outer_fold)
        for outer_fold in sorted(np.unique(fold).tolist())
    }
    return {
        "diagnostic_only": True,
        "target_dependent": True,
        "deployable": False,
        "commercial_evidence": False,
        "stage_A_gate_input": False,
        "reason": (
            "The reference target selects the lowest-error candidate on each row; "
            "this is an upper-bound representation diagnostic, not a forward policy."
        ),
        "candidate_names": names,
        "candidate_count": int(len(names)),
        "alignment": {
            "cache_index_exact_order": True,
            "fold_exact": True,
            "identity_exact": True,
            "session_exact": True,
            "target_numeric": True,
            "classical_rr_numeric": True,
            "cache_index_sha256": sha256_array(alias_index),
        },
        "overall": summarize(np.ones(len(target), dtype=bool)),
        "folds": folds,
        "selected_candidate_index_sha256": sha256_array(selected),
        "oracle_prediction_sha256": sha256_array(oracle_prediction),
    }


def validate_identity_splits(
    identities: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    """Fail closed unless splits are identity-disjoint and held rows cover once."""

    identities = np.asarray(identities, dtype=str)
    if identities.ndim != 1 or not len(identities):
        raise ValueError("identities must be a non-empty one-dimensional array")
    if not splits:
        raise RuntimeError("no inner splits were produced")
    held_count = np.zeros(len(identities), dtype=np.int16)
    expected_rows = np.arange(len(identities), dtype=np.int64)
    audit: list[dict[str, Any]] = []
    for number, (fit_raw, held_raw) in enumerate(splits):
        fit = np.asarray(fit_raw, dtype=np.int64)
        held = np.asarray(held_raw, dtype=np.int64)
        if fit.ndim != 1 or held.ndim != 1 or not len(fit) or not len(held):
            raise RuntimeError(f"inner split {number} is empty or not one-dimensional")
        if (
            np.any(fit < 0)
            or np.any(held < 0)
            or np.any(fit >= len(identities))
            or np.any(held >= len(identities))
        ):
            raise RuntimeError(f"inner split {number} contains an out-of-range row")
        if len(np.unique(fit)) != len(fit) or len(np.unique(held)) != len(held):
            raise RuntimeError(f"inner split {number} contains duplicate rows")
        if np.intersect1d(fit, held).size:
            raise RuntimeError(f"inner split {number} has train/held row overlap")
        if not np.array_equal(np.sort(np.concatenate([fit, held])), expected_rows):
            raise RuntimeError(f"inner split {number} is not an exact row partition")
        fit_identities = np.unique(identities[fit])
        held_identities = np.unique(identities[held])
        leaked = np.intersect1d(fit_identities, held_identities)
        if leaked.size:
            raise RuntimeError(
                f"inner split {number} has identity leakage: {leaked.tolist()}"
            )
        held_count[held] += 1
        audit.append(
            {
                "inner_fold": int(number),
                "fit_rows": int(len(fit)),
                "held_rows": int(len(held)),
                "fit_identity_count": int(len(fit_identities)),
                "held_identity_count": int(len(held_identities)),
                "held_identities": held_identities.tolist(),
            }
        )
    if not np.all(held_count == 1):
        bad = np.flatnonzero(held_count != 1)
        raise RuntimeError(
            "grouped inner held-out rows are not an exact one-time cover: "
            f"{len(bad)} bad rows"
        )
    return audit


def build_grouped_splits(
    identities: np.ndarray, *, n_splits: int = INNER_SPLITS
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    identities = np.asarray(identities, dtype=str)
    if len(np.unique(identities)) < n_splits:
        raise RuntimeError("fewer identities than requested grouped inner folds")
    splitter = GroupKFold(n_splits=n_splits)
    splits = [
        (np.asarray(fit, dtype=np.int64), np.asarray(held, dtype=np.int64))
        for fit, held in splitter.split(np.zeros(len(identities)), groups=identities)
    ]
    return splits, validate_identity_splits(identities, splits)


def evaluate_policy(
    *,
    target: np.ndarray,
    base_prediction: np.ndarray,
    correction: np.ndarray,
    selected: np.ndarray,
    identities: np.ndarray,
    actionable: np.ndarray,
    base_good: np.ndarray,
    pull: float,
    threshold: float | None,
    thresholds: StageAThresholds,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    base_prediction = np.asarray(base_prediction, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    selected = np.asarray(selected, dtype=bool)
    identities = np.asarray(identities, dtype=str)
    actionable = np.asarray(actionable, dtype=bool)
    base_good = np.asarray(base_good, dtype=bool)
    shapes = {
        value.shape
        for value in (
            target,
            base_prediction,
            correction,
            selected,
            identities,
            actionable,
            base_good,
        )
    }
    if len(shapes) != 1 or target.ndim != 1:
        raise ValueError("policy inputs must be aligned vectors")
    if not 0.0 <= pull <= 1.0:
        raise ValueError("pull must lie in [0, 1]")
    prediction = base_prediction.copy()
    prediction[selected] = (
        (1.0 - pull) * base_prediction[selected] + pull * correction[selected]
    )
    gain = np.abs(base_prediction - target) - np.abs(prediction - target)
    precision = float(np.mean(gain[selected] > 0.0)) if selected.any() else 1.0
    actionable_count = int(actionable.sum())
    recall = float(np.sum(selected & actionable) / max(1, actionable_count))
    base_good_count = int(base_good.sum())
    base_good_false_positive = float(
        np.sum(selected & base_good & (np.abs(prediction - target) > 2.0))
        / max(1, base_good_count)
    )
    base_metrics = regression_metrics(target, base_prediction, identities)
    candidate_metrics = regression_metrics(target, prediction, identities)
    macro_gain = float(base_metrics["macro_mae"] - candidate_metrics["macro_mae"])
    mae_gain = float(base_metrics["mae"] - candidate_metrics["mae"])
    checks = {
        "correction_precision": precision >= thresholds.correction_precision_min,
        "correction_recall": recall >= thresholds.correction_recall_min,
        "baseline_good_false_positive_fraction": (
            base_good_false_positive
            <= thresholds.baseline_good_false_positive_fraction_max
        ),
        "estimated_macro_mae_gain_bpm": (
            macro_gain >= thresholds.estimated_macro_mae_gain_bpm_min
        ),
    }
    return {
        "threshold": threshold,
        "pull": float(pull),
        "selected": int(selected.sum()),
        "coverage": float(selected.mean()),
        "precision": precision,
        "recall": recall,
        "base_good_false_positive_fraction": base_good_false_positive,
        "macro_mae_gain_bpm": macro_gain,
        "mae_gain_bpm": mae_gain,
        "candidate_metrics": candidate_metrics,
        "gate_checks": checks,
        "passes": bool(all(checks.values())),
    }


def rank_policies(policies: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def key(policy: Mapping[str, Any]) -> tuple[Any, ...]:
        threshold = policy.get("threshold")
        return (
            not bool(policy["passes"]),
            -float(policy["macro_mae_gain_bpm"]),
            -float(policy["precision"]),
            float(policy["coverage"]),
            float("inf") if threshold is None else -float(threshold),
            -float(policy["pull"]),
        )

    return [dict(value) for value in sorted(policies, key=key)]


def evaluate_stage_gate(
    partitions: Mapping[str, Mapping[str, Any]],
    thresholds: StageAThresholds,
) -> dict[str, Any]:
    expected = {str(value) for value in OUTER_EXCLUSIONS}
    if set(partitions) != expected:
        raise RuntimeError(
            f"Stage-A gate requires partitions {sorted(expected)}, "
            f"found {sorted(partitions)}"
        )
    partition_checks: dict[str, Any] = {}
    for name in sorted(partitions, key=int):
        result = partitions[name]
        checks = {
            "action_auc": float(result["action_auc"]) >= thresholds.action_auc_min,
            "action_average_precision": (
                float(result["action_average_precision"])
                >= thresholds.action_average_precision_min
            ),
            "factor_accuracy_gain_over_x1_prevalence": (
                float(result["factor_accuracy_gain_over_x1_prevalence"])
                >= thresholds.factor_accuracy_gain_over_x1_prevalence_min
            ),
            "safe_policy_exists": int(result["passing_policy_count"]) > 0,
        }
        partition_checks[name] = {
            "checks": checks,
            "passed": bool(all(checks.values())),
        }
    passed = bool(all(item["passed"] for item in partition_checks.values()))
    return {
        "partitions": partition_checks,
        "passed": passed,
        "decision": (
            "advance_to_locked_stage_B_neural_validation"
            if passed
            else "kill_before_neural_training_and_preserve_exact_frozen_base"
        ),
    }


def _validate_dataset_binding(
    *,
    metadata: Any,
    base: Mapping[str, np.ndarray],
    feature_map: np.ndarray,
    layout: Any,
    feature_manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    valid_index = np.asarray(base["index"], dtype=np.int64)
    fold = np.asarray(base["fold"], dtype=np.int16)
    prediction = np.asarray(base["prediction"], dtype=np.float64)
    expected_population = contract["immutable_population"]
    if len(valid_index) != int(expected_population["valid_reference_rows"]):
        raise RuntimeError("valid-reference denominator differs from campaign contract")
    if len(metadata) != int(expected_population["all_window_rows"]):
        raise RuntimeError("all-window denominator differs from campaign contract")
    if len(np.unique(valid_index)) != len(valid_index):
        raise RuntimeError("base valid indices are not unique")
    if fold.shape != valid_index.shape or prediction.shape != valid_index.shape:
        raise RuntimeError("base index/fold/prediction arrays are not aligned")
    if not np.isfinite(prediction).all():
        raise RuntimeError("base prediction contains non-finite values")
    identities = metadata.iloc[valid_index]["identity"].astype(str).to_numpy()
    identity_to_fold: dict[str, int] = {}
    for identity in np.unique(identities):
        assigned = np.unique(fold[identities == identity])
        if len(assigned) != 1:
            raise RuntimeError(f"identity {identity} spans outer folds {assigned.tolist()}")
        identity_to_fold[str(identity)] = int(assigned[0])
    locked_mapping = {
        str(key): int(value)
        for key, value in expected_population["identity_to_fold"].items()
    }
    if identity_to_fold != locked_mapping:
        raise RuntimeError("base identity-to-fold assignment differs from campaign contract")
    if len(identity_to_fold) != int(expected_population["identity_count"]):
        raise RuntimeError("identity denominator differs from campaign contract")
    if sorted(np.unique(fold).tolist()) != list(range(int(expected_population["fold_count"]))):
        raise RuntimeError("outer fold cover differs from campaign contract")

    recorded_shape = tuple(int(value) for value in feature_manifest["shape"])
    if tuple(feature_map.shape) != recorded_shape:
        raise RuntimeError("feature-map shape differs from its manifest")
    if str(feature_map.dtype) != str(feature_manifest["dtype"]):
        raise RuntimeError("feature-map dtype differs from its manifest")
    if tuple(layout.names) != tuple(map(str, feature_manifest["feature_names"])):
        raise RuntimeError("verified feature layout differs from its manifest")
    columns = np.flatnonzero(layout.aggregate_mask & layout.verified_mask)
    if len(columns) != 228:
        raise RuntimeError(f"expected 228 aggregate+verified features, found {len(columns)}")

    # Bind the metadata's classical estimator to the immutable feature map.  The
    # scalar is repeated for every factor and stored as float16 in the map.
    try:
        classical_column = layout.names.index("classical_rr_bpm")
        factor_column = layout.names.index("factor")
    except ValueError as error:
        raise RuntimeError("feature map lacks classical/factor binding columns") from error
    classical = metadata.iloc[valid_index]["classical_rr_bpm"].to_numpy(dtype=np.float64)
    target = metadata.iloc[valid_index]["rr_bpm"].to_numpy(dtype=np.float64)
    stored_classical = np.asarray(
        feature_map[valid_index, :, classical_column], dtype=np.float64
    )
    stored_factor = np.asarray(feature_map[valid_index, :, factor_column], dtype=np.float64)
    if not np.allclose(stored_classical, classical[:, None], rtol=1e-3, atol=2e-2):
        raise RuntimeError("metadata classical RR is not bound to the feature map")
    if not np.array_equal(stored_factor, np.broadcast_to(FACTORS, stored_factor.shape)):
        raise RuntimeError("feature-map factor ordering differs from x1/x2/x3/x4")
    if not np.isfinite(classical).all() or not np.isfinite(target).all():
        raise RuntimeError("target/classical evaluation fields contain non-finite values")

    return {
        "all_window_rows": int(len(metadata)),
        "valid_reference_rows": int(len(valid_index)),
        "identity_count": int(len(identity_to_fold)),
        "fold_count": int(len(np.unique(fold))),
        "identity_to_fold": identity_to_fold,
        "aggregate_verified_feature_count": int(len(columns)),
        "valid_cache_index_sha256": sha256_array(valid_index),
        "target_sha256": sha256_array(target),
        "classical_rr_sha256": sha256_array(classical),
        "base_prediction_sha256": sha256_array(prediction),
    }


def _fit_inner_oof_scores(
    *,
    features: np.ndarray,
    errors: np.ndarray,
    identities: np.ndarray,
    outer_exclusion: int,
    n_jobs: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError("xgboost is required for the locked Stage-A audit") from error

    splits, split_audit = build_grouped_splits(identities, n_splits=INNER_SPLITS)
    scores = np.full((len(identities), len(FACTORS)), np.nan, dtype=np.float64)
    for inner_fold, (fit, held) in enumerate(splits):
        fit_features = np.asarray(features[fit], dtype=np.float32)
        fit_features = fit_features.reshape(-1, fit_features.shape[-1])
        fit_target = -np.minimum(errors[fit], 10.0).reshape(-1)
        row_weight = identity_weights(identities[fit])
        weights = np.repeat(row_weight, len(FACTORS))
        best = errors[fit].argmin(axis=1)
        bonus = np.ones((len(fit), len(FACTORS)), dtype=np.float64)
        bonus[np.arange(len(fit)), best] = 2.5
        weights *= bonus.reshape(-1)
        model = xgb.XGBRegressor(
            **MODEL_PARAMETERS,
            n_jobs=int(n_jobs),
            random_state=20260828 + 1009 * int(outer_exclusion) + inner_fold,
        )
        model.fit(fit_features, fit_target, sample_weight=weights, verbose=False)
        held_features = np.asarray(features[held], dtype=np.float32)
        scores[held] = model.predict(
            held_features.reshape(-1, held_features.shape[-1])
        ).reshape(-1, len(FACTORS))
    if not np.isfinite(scores).all():
        raise RuntimeError("grouped inner-OOF candidate scores are incomplete")
    return scores, split_audit


def audit_partition(
    *,
    outer_exclusion: int,
    features: np.ndarray,
    target: np.ndarray,
    classical: np.ndarray,
    base_prediction: np.ndarray,
    identities: np.ndarray,
    thresholds: StageAThresholds,
    n_jobs: int,
) -> dict[str, Any]:
    errors = np.abs(classical[:, None] * FACTORS[None, :] - target[:, None])
    best_factor = errors.argmin(axis=1)
    reliable = errors.min(axis=1) <= 2.0
    scores, split_audit = _fit_inner_oof_scores(
        features=features,
        errors=errors,
        identities=identities,
        outer_exclusion=outer_exclusion,
        n_jobs=n_jobs,
    )
    chosen = scores.argmax(axis=1)
    non_direct_best = 1 + scores[:, 1:].argmax(axis=1)
    margin = scores[np.arange(len(scores)), non_direct_best] - scores[:, 0]
    action_label = reliable & (best_factor > 0)
    if len(np.unique(action_label)) != 2:
        raise RuntimeError(
            f"outer exclusion {outer_exclusion} has a degenerate action label"
        )
    action_auc = float(roc_auc_score(action_label, margin))
    action_ap = float(average_precision_score(action_label, margin))
    if not reliable.any():
        raise RuntimeError(f"outer exclusion {outer_exclusion} has no reliable factors")
    x1_prevalence = float(np.mean(best_factor[reliable] == 0))
    factor_accuracy = float(np.mean(chosen[reliable] == best_factor[reliable]))

    alternative_best = 1 + errors[:, 1:].argmin(axis=1)
    actionable = (
        (errors[:, 1:].min(axis=1) + 0.75 < np.abs(base_prediction - target))
        & (alternative_best == best_factor)
    )
    base_good = np.abs(base_prediction - target) <= 2.0
    threshold_values = [float("inf")]
    threshold_values.extend(
        float(np.quantile(margin, q)) for q in np.linspace(0.80, 0.995, 80)
    )
    unique_thresholds = sorted(set(threshold_values), reverse=True)
    policies: list[dict[str, Any]] = []
    correction = classical * FACTORS[non_direct_best]
    for threshold in unique_thresholds:
        selected = margin >= threshold
        for pull in (0.25, 0.5, 0.75, 1.0):
            policies.append(
                evaluate_policy(
                    target=target,
                    base_prediction=base_prediction,
                    correction=correction,
                    selected=selected,
                    identities=identities,
                    actionable=actionable,
                    base_good=base_good,
                    pull=pull,
                    threshold=None if not np.isfinite(threshold) else float(threshold),
                    thresholds=thresholds,
                )
            )
    ranked = rank_policies(policies)
    return {
        "outer_exclusion": int(outer_exclusion),
        "pool_rows": int(len(target)),
        "pool_identity_count": int(len(np.unique(identities))),
        "reliable_rows": int(reliable.sum()),
        "action_positive_rows": int(action_label.sum()),
        "actionable_rows": int(actionable.sum()),
        "x1_prevalence_among_reliable": x1_prevalence,
        "factor_accuracy": factor_accuracy,
        "factor_accuracy_gain_over_x1_prevalence": factor_accuracy - x1_prevalence,
        "action_auc": action_auc,
        "action_average_precision": action_ap,
        "base_metrics": regression_metrics(target, base_prediction, identities),
        "best_policy": ranked[0],
        "passing_policy_count": int(sum(bool(item["passes"]) for item in policies)),
        "policy_count": int(len(policies)),
        "inner_splits": split_audit,
        "oof_score_sha256": sha256_array(scores),
        "oof_margin_sha256": sha256_array(margin),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive")
    contract, feature_manifest, source_verification = verify_sources(args)
    thresholds = StageAThresholds.from_contract(contract)
    physics = _load_physics_source(args.physics_source)
    metadata, _ = physics.load_metadata(args.svd_cache)
    base = physics.load_base(args.base_oof, metadata)
    feature_map = np.load(args.feature_map, mmap_mode="r", allow_pickle=False)
    layout = physics.make_feature_layout(
        tuple(feature_manifest["variant_names"]), int(feature_manifest["components"])
    )
    binding = _validate_dataset_binding(
        metadata=metadata,
        base=base,
        feature_map=feature_map,
        layout=layout,
        feature_manifest=feature_manifest,
        contract=contract,
    )
    oracle_diagnostic = target_dependent_candidate_oracle(
        alias_oof_csv=args.alias_oof_csv,
        metadata=metadata,
        base=base,
    )

    valid_index = np.asarray(base["index"], dtype=np.int64)
    fold = np.asarray(base["fold"], dtype=np.int16)
    base_lookup = np.full(len(metadata), np.nan, dtype=np.float64)
    base_lookup[valid_index] = np.asarray(base["prediction"], dtype=np.float64)
    target_all = metadata.iloc[valid_index]["rr_bpm"].to_numpy(dtype=np.float64)
    classical_all = metadata.iloc[valid_index]["classical_rr_bpm"].to_numpy(
        dtype=np.float64
    )
    identity_all = metadata.iloc[valid_index]["identity"].astype(str).to_numpy()
    columns = np.flatnonzero(layout.aggregate_mask & layout.verified_mask)
    features_all = np.asarray(feature_map[valid_index][:, :, columns], dtype=np.float32)
    if not np.isfinite(features_all).all():
        raise RuntimeError("selected aggregate+verified feature map is non-finite")

    partitions: dict[str, Any] = {}
    for outer_exclusion in OUTER_EXCLUSIONS:
        pool = np.flatnonzero(fold != outer_exclusion)
        if np.any(fold[pool] == outer_exclusion):
            raise RuntimeError(f"outer fold {outer_exclusion} was not excluded")
        result = audit_partition(
            outer_exclusion=outer_exclusion,
            features=features_all[pool],
            target=target_all[pool],
            classical=classical_all[pool],
            base_prediction=base_lookup[valid_index[pool]],
            identities=identity_all[pool],
            thresholds=thresholds,
            n_jobs=args.n_jobs,
        )
        result["excluded_rows"] = int(np.sum(fold == outer_exclusion))
        result["excluded_identities"] = sorted(
            np.unique(identity_all[fold == outer_exclusion]).tolist()
        )
        partitions[str(outer_exclusion)] = result

    stage_gate = evaluate_stage_gate(partitions, thresholds)
    passed = bool(stage_gate["passed"])
    return {
        "schema_version": FORMAT_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "audit_id": "stage_A_grouped_inner_oof_harmonic_candidate_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "retrospective_adaptive_engineering_evidence",
        "retrospective_notice": RETROSPECTIVE_NOTICE,
        "status": "stage_A_passed" if passed else "stage_A_killed",
        "scientific_gate_failure_exit_code": 0,
        "commercial_claim_allowed": False,
        "forward_input_policy": {
            "learned_inputs": [
                "verified_aggregate_current_window_physics_features_at_x1_x2_x3_x4"
            ],
            "frozen_base_prediction_is_learned_input": False,
            "frozen_base_use": "evaluation_comparator_and_exact_fallback_only",
            "reference_target_use": "inner_fit_supervision_and_held_evaluation_only",
            "identity_use": "grouped_split_construction_and_macro_metric_only",
        },
        "source_verification": source_verification,
        "dataset_binding": binding,
        "configuration": {
            "outer_exclusions": list(OUTER_EXCLUSIONS),
            "inner_splitter": "GroupKFold",
            "inner_splits": INNER_SPLITS,
            "factors": FACTORS.tolist(),
            "features": "aggregate_mask AND verified_mask",
            "model": MODEL_PARAMETERS,
            "random_seed_formula": "20260828 + 1009*outer_exclusion + inner_fold",
            "best_factor_bonus_weight": 2.5,
            "utility_target": "-min(abs(classical_rr*factor-reference_rr),10)",
            "actionable_minimum_gain_bpm": 0.75,
            "policy_threshold_quantiles": {
                "start": 0.80,
                "stop": 0.995,
                "count": 80,
                "include_select_none": True,
            },
            "policy_pulls": [0.25, 0.5, 0.75, 1.0],
            "n_jobs_operational_only": int(args.n_jobs),
        },
        "predeclared_thresholds": asdict(thresholds),
        "partitions": partitions,
        "stage_gate": stage_gate,
        "diagnostics": {
            "target_dependent_candidate_oracle": oracle_diagnostic,
        },
        "termination": {
            "neural_training_authorized": passed,
            "exact_frozen_base_preserved": not passed,
            "commercial_claim_allowed": False,
            "reason": (
                "Stage A passed; only the locked Stage-B discovery validation is authorized."
                if passed
                else "No safe grouped-inner-OOF policy passed every predeclared condition in both partitions."
            ),
        },
        "implementation": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-contract",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/campaigns/harmonic_factor_snn_v1/CAMPAIGN_CONTRACT.json",
    )
    parser.add_argument(
        "--svd-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts/cache/svd_components_all_v1",
    )
    parser.add_argument(
        "--base-oof",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/runs/ensemble_structured_exact/ensemble_oof.npz",
    )
    parser.add_argument(
        "--feature-map",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/discovery_physics_ridge/physics_candidate_features_v1.npy",
    )
    parser.add_argument(
        "--alias-oof-csv",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/runs/final_alias_gate_s12_deterministic/snn_oof.csv",
        help=(
            "Pinned alias-SNN OOF source used only for the target-dependent "
            "candidate-bank oracle diagnostic."
        ),
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts/discovery_physics_ridge/feature_manifest.json",
    )
    parser.add_argument(
        "--physics-source",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/discovery_physics_ridge/run_physics_ridge_hmm.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/campaigns/harmonic_factor_snn_v1/STAGE_A_GATE.json",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=3,
        help="XGBoost CPU threads; this does not alter the locked estimator.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_audit(args)
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "stage_A_passed": report["stage_gate"]["passed"],
                "commercial_claim_allowed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    # A predeclared scientific kill is a completed audit, not a process error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
