#!/usr/bin/env python3
"""Apply a validation-locked two-SNN blend to radar-dropout OOF predictions.

``benchmark_robustness.py`` evaluates one checkpoint family at a time.  This
companion keeps the already-locked per-fold blend and disagreement rules from
``scripts/ensemble.py`` fixed, applies them to matching masked-radar OOF files,
and reports the resulting ensemble robustness.  No target is used to select a
mask-specific weight or uncertainty rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from snn_rr.cache import load_feature_cache  # noqa: E402
from snn_rr.metrics import grouped_oof_metrics, risk_coverage_curve  # noqa: E402
from scripts.benchmark_robustness import (  # noqa: E402
    RADAR_MASKS,
    _condition_summaries,
)
from scripts.train import write_json  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_recorded_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    required = ("index", "target", "prediction", "rr_std", "quality", "fold")
    with np.load(path, allow_pickle=False) as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        return {key: np.asarray(handle[key]).copy() for key in required}


def _ensemble_source_records(
    ensemble: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Normalize the modern source mapping while retaining legacy list support."""

    source_runs = ensemble.get("source_runs")
    if isinstance(source_runs, Mapping):
        records = []
        for label, raw in source_runs.items():
            if not isinstance(raw, Mapping) or "path" not in raw:
                raise TypeError("ensemble source_runs entries must contain paths")
            records.append(
                {
                    "label": str(label),
                    "path": str(raw["path"]),
                    "run_signature": str(raw.get("run_signature", "")),
                }
            )
        return records
    if isinstance(source_runs, list):
        # Older reports stored only paths.  The signature is recovered from the
        # immutable source run_config and then checked against each checkpoint.
        return [
            {"label": Path(str(path)).name, "path": str(path), "run_signature": ""}
            for path in source_runs
        ]
    raise TypeError("ensemble metrics source_runs must be a mapping or list")


def _checkpoint_identity(path: Path) -> tuple[str, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    if "run_signature" not in checkpoint or "fold" not in checkpoint:
        raise KeyError(f"checkpoint lacks run_signature/fold provenance: {path}")
    return str(checkpoint["run_signature"]), int(checkpoint["fold"])


def validate_masked_source_provenance(
    source_record: Mapping[str, str],
    robustness_dir: Path,
) -> dict[str, Any]:
    """Bind a masked OOF directory to the exact ensemble source run."""

    expected_run_dir = _resolve_recorded_path(source_record["path"])
    robustness_dir = robustness_dir.resolve()
    run_config_path = expected_run_dir / "run_config.json"
    robustness_report_path = robustness_dir / "report.json"
    run_config = _load_json(run_config_path)
    robustness_report = _load_json(robustness_report_path)
    actual_signature = str(run_config.get("run_signature", ""))
    declared_signature = str(source_record.get("run_signature", ""))
    if declared_signature and declared_signature != actual_signature:
        raise RuntimeError(
            f"ensemble/source run signature mismatch for {source_record['label']}"
        )
    if not actual_signature:
        raise RuntimeError(f"source run has no run signature: {expected_run_dir}")

    model_record = robustness_report.get("model")
    if not isinstance(model_record, Mapping) or "checkpoint" not in model_record:
        raise RuntimeError(
            f"masked robustness report has no model checkpoint: {robustness_report_path}"
        )
    reported_checkpoint = _resolve_recorded_path(str(model_record["checkpoint"]))
    if reported_checkpoint.parent.parent != expected_run_dir:
        raise RuntimeError(
            f"masked robustness source directory does not match ensemble source "
            f"{source_record['label']}"
        )
    reported_signature, reported_fold = _checkpoint_identity(reported_checkpoint)
    if reported_signature != actual_signature or reported_fold != 0:
        raise RuntimeError(
            f"masked robustness checkpoint provenance mismatch for {source_record['label']}"
        )

    folds = int(run_config.get("arguments", {}).get("folds", 0))
    if folds < 1:
        raise RuntimeError(f"invalid source fold count: {run_config_path}")
    checkpoint_sha256: dict[str, str] = {}
    for fold in range(folds):
        checkpoint_path = expected_run_dir / f"fold_{fold}" / "snn_best.pt"
        signature, checkpoint_fold = _checkpoint_identity(checkpoint_path)
        if signature != actual_signature or checkpoint_fold != fold:
            raise RuntimeError(
                f"source checkpoint signature/fold mismatch: {checkpoint_path}"
            )
        checkpoint_sha256[str(fold)] = _sha256(checkpoint_path)

    reference_coverage = robustness_report.get("reference_coverage", {})
    return {
        "label": str(source_record["label"]),
        "ensemble_source_run_dir": str(expected_run_dir),
        "ensemble_declared_run_signature": declared_signature or actual_signature,
        "source_run_config": str(run_config_path),
        "source_run_config_sha256": _sha256(run_config_path),
        "source_run_signature": actual_signature,
        "source_fold_count": folds,
        "source_checkpoint_sha256": checkpoint_sha256,
        "robustness_dir": str(robustness_dir),
        "robustness_report": str(robustness_report_path),
        "robustness_report_sha256": _sha256(robustness_report_path),
        "reported_checkpoint": str(reported_checkpoint),
        "reported_checkpoint_run_signature": reported_signature,
        "reported_checkpoint_fold": reported_fold,
        "reference_coverage": reference_coverage,
        "passed": True,
        "binding_method": (
            "masked report checkpoint parent, checkpoint run_signature, source "
            "run_config, and all fold checkpoint signatures"
        ),
    }


def _load_locked_ensemble_oof(
    path: Path, prediction_key: str
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        required = ("index", "target", "fold", prediction_key)
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"locked ensemble OOF is missing {missing}: {path}")
        result = {
            "index": np.asarray(handle["index"]).copy(),
            "target": np.asarray(handle["target"]).copy(),
            "fold": np.asarray(handle["fold"]).copy(),
            "prediction": np.asarray(handle[prediction_key]).copy(),
        }
    order = np.argsort(result["index"], kind="stable")
    result = {key: value[order] for key, value in result.items()}
    if len(np.unique(result["index"])) != len(result["index"]):
        raise RuntimeError("locked ensemble OOF contains duplicate cache indices")
    return result


def validate_full_mask_consistency(
    full_result: Mapping[str, np.ndarray],
    locked_oof: Mapping[str, np.ndarray],
    *,
    prediction_atol_bpm: float,
) -> dict[str, Any]:
    """Require full-radar re-inference to reproduce the locked ensemble OOF."""

    full = {key: np.asarray(value) for key, value in full_result.items()}
    locked = {key: np.asarray(value) for key, value in locked_oof.items()}
    order = np.argsort(full["index"], kind="stable")
    full = {key: value[order] for key, value in full.items()}
    if not np.array_equal(full["index"].astype(np.int64), locked["index"].astype(np.int64)):
        raise RuntimeError("full-mask indices differ from locked ensemble OOF")
    if not np.array_equal(full["fold"].astype(np.int64), locked["fold"].astype(np.int64)):
        raise RuntimeError("full-mask folds differ from locked ensemble OOF")
    if not np.allclose(
        full["target"].astype(float),
        locked["target"].astype(float),
        atol=1e-5,
        rtol=0.0,
    ):
        raise RuntimeError("full-mask targets differ from locked ensemble OOF")
    delta = np.abs(
        full["prediction"].astype(float) - locked["prediction"].astype(float)
    )
    maximum = float(np.max(delta)) if len(delta) else 0.0
    if maximum > prediction_atol_bpm:
        raise RuntimeError(
            "full-mask predictions do not reproduce locked ensemble OOF within "
            f"{prediction_atol_bpm:g} bpm (max={maximum:g})"
        )
    return {
        "passed": True,
        "rows": int(len(delta)),
        "prediction_atol_bpm": float(prediction_atol_bpm),
        "prediction_max_abs_difference_bpm": maximum,
        "prediction_mean_abs_difference_bpm": float(np.mean(delta)),
        "prediction_p99_abs_difference_bpm": float(np.quantile(delta, 0.99)),
        "exact_index_match": True,
        "exact_fold_match": True,
        "target_atol_bpm": 1e-5,
        "target_match": True,
        "interpretation": (
            "the declared absolute tolerance permits only bounded numerical "
            "re-inference drift; it is not a fitted metric tolerance"
        ),
    }


def align_masked_sources(
    source_a: Mapping[str, np.ndarray],
    source_b: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return cache-index-aligned source bundles and reject provenance drift."""

    a = {key: np.asarray(value) for key, value in source_a.items()}
    b = {key: np.asarray(value) for key, value in source_b.items()}
    lengths_a = {len(value) for value in a.values()}
    lengths_b = {len(value) for value in b.values()}
    if len(lengths_a) != 1 or len(lengths_b) != 1:
        raise RuntimeError("masked source arrays have inconsistent lengths")
    for label, index in (("A", a["index"]), ("B", b["index"])):
        numeric_index = index.astype(float)
        if not np.isfinite(numeric_index).all() or not np.equal(
            numeric_index, np.rint(numeric_index)
        ).all():
            raise RuntimeError(f"source {label} contains non-integral cache indices")
    order_a = np.argsort(a["index"], kind="stable")
    order_b = np.argsort(b["index"], kind="stable")
    a = {key: value[order_a] for key, value in a.items()}
    b = {key: value[order_b] for key, value in b.items()}
    if len(np.unique(a["index"])) != len(a["index"]):
        raise RuntimeError("source A contains duplicate cache indices")
    if len(np.unique(b["index"])) != len(b["index"]):
        raise RuntimeError("source B contains duplicate cache indices")
    if not np.array_equal(a["index"], b["index"]):
        raise RuntimeError("masked source cache indices differ")
    if not np.array_equal(a["fold"], b["fold"]):
        raise RuntimeError("masked source fold assignments differ")
    if not np.allclose(a["target"], b["target"], atol=1e-5, rtol=0.0):
        raise RuntimeError("masked source targets differ")
    for label, bundle in (("A", a), ("B", b)):
        for key in ("target", "prediction", "rr_std", "quality"):
            if not np.isfinite(bundle[key].astype(float)).all():
                raise RuntimeError(f"source {label} {key} contains non-finite values")
    return a, b


def locked_parameters(
    fold_selection: Mapping[str, Any],
    folds: np.ndarray,
    source_a_label: str,
    source_b_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand validation-selected fold parameters to row vectors."""

    folds = np.asarray(folds, dtype=np.int64)
    weight = np.empty(len(folds), dtype=np.float64)
    disagreement = np.empty(len(folds), dtype=np.float64)
    key_a = f"weight_{source_a_label}"
    key_b = f"weight_{source_b_label}"
    for fold in np.unique(folds):
        selected = fold_selection[str(int(fold))]["selected"]
        if key_a not in selected or key_b not in selected:
            raise KeyError(f"fold {fold} does not contain source weights")
        weight_a = float(selected[key_a])
        weight_b = float(selected[key_b])
        if not np.isclose(weight_a + weight_b, 1.0, atol=1e-9):
            raise RuntimeError(f"fold {fold} source weights do not sum to one")
        rows = folds == fold
        weight[rows] = weight_a
        disagreement[rows] = float(selected["disagreement_coefficient"])
    return weight, disagreement


def apply_locked_mask_blend(
    source_a: Mapping[str, np.ndarray],
    source_b: Mapping[str, np.ndarray],
    *,
    weight_a: np.ndarray,
    disagreement: np.ndarray,
) -> dict[str, np.ndarray]:
    """Blend a masked condition with weights frozen before masked evaluation."""

    a, b = align_masked_sources(source_a, source_b)
    weight = np.asarray(weight_a, dtype=np.float64)
    coefficient = np.asarray(disagreement, dtype=np.float64)
    if weight.shape != a["target"].shape or coefficient.shape != weight.shape:
        raise ValueError("locked parameter shape differs from masked predictions")
    prediction_a = a["prediction"].astype(np.float64)
    prediction_b = b["prediction"].astype(np.float64)
    score_a = a["rr_std"].astype(np.float64) / np.clip(
        a["quality"].astype(np.float64), 0.05, None
    )
    score_b = b["rr_std"].astype(np.float64) / np.clip(
        b["quality"].astype(np.float64), 0.05, None
    )
    prediction = weight * prediction_a + (1.0 - weight) * prediction_b
    # ensemble_uncertainty accepts a scalar weight.  The fold-specific vector
    # form is the same declared equation and is expanded here without fitting.
    uncertainty = (
        weight * score_a
        + (1.0 - weight) * score_b
        + coefficient * np.abs(prediction_a - prediction_b)
    )
    if not np.isfinite(prediction).all() or not np.isfinite(uncertainty).all():
        raise RuntimeError("locked masked ensemble produced non-finite values")
    return {
        "index": a["index"].astype(np.int64),
        "target": a["target"].astype(np.float64),
        "prediction": prediction,
        "uncertainty": uncertainty,
        "fold": a["fold"].astype(np.int16),
        "weight_a": weight,
        "disagreement_coefficient": coefficient,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensemble_dir = args.ensemble_dir.resolve()
    source_a_dir = args.source_a_robustness.resolve()
    source_b_dir = args.source_b_robustness.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ensemble_metrics_path = ensemble_dir / "metrics.json"
    ensemble_config_path = ensemble_dir / "run_config.json"
    ensemble_oof_path = ensemble_dir / "ensemble_oof.npz"
    ensemble = _load_json(ensemble_metrics_path)
    ensemble_config = _load_json(ensemble_config_path)
    source_records = _ensemble_source_records(ensemble)
    if len(source_records) != 2:
        raise RuntimeError("robustness audit requires exactly two ensemble sources")
    configured_labels = ensemble_config.get("source_labels", {})
    label_a = str(configured_labels.get("a", source_records[0]["label"]))
    label_b = str(configured_labels.get("b", source_records[1]["label"]))
    records_by_label = {record["label"]: record for record in source_records}
    if set(records_by_label) != {label_a, label_b}:
        raise RuntimeError("ensemble source labels differ between metrics and run config")
    sources = [records_by_label[label_a], records_by_label[label_b]]
    source_provenance = [
        validate_masked_source_provenance(sources[0], source_a_dir),
        validate_masked_source_provenance(sources[1], source_b_dir),
    ]
    if ensemble.get("complete_oof") is not True:
        raise RuntimeError("ensemble metrics do not declare a complete OOF")
    locked_oof = _load_locked_ensemble_oof(
        ensemble_oof_path, args.ensemble_prediction_key
    )
    cache = load_feature_cache(args.cache_dir.resolve())
    expected_index = np.flatnonzero(
        cache.metadata["reference_valid"].to_numpy(dtype=bool)
    )
    if not np.array_equal(
        locked_oof["index"].astype(np.int64), expected_index.astype(np.int64)
    ):
        raise RuntimeError(
            "locked ensemble OOF does not cover the canonical valid-reference indices"
        )
    expected_rows = int(len(expected_index))
    if int(ensemble.get("n_rows", -1)) != expected_rows:
        raise RuntimeError("ensemble metrics row count differs from canonical cache")
    expected_identities = int(
        cache.metadata.iloc[expected_index]["identity"].astype(str).nunique()
    )
    if int(ensemble.get("n_identities", -1)) != expected_identities:
        raise RuntimeError("ensemble metrics identity count differs from canonical cache")
    for provenance in source_provenance:
        coverage = provenance["reference_coverage"]
        if int(coverage.get("valid_windows", -1)) != expected_rows:
            raise RuntimeError(
                f"masked source {provenance['label']} has incomplete reference coverage"
            )

    conditions: dict[str, dict[str, Any]] = {}
    condition_results: dict[str, dict[str, np.ndarray]] = {}
    full_result: dict[str, np.ndarray] | None = None
    for condition in RADAR_MASKS:
        a, b = align_masked_sources(
            _load_npz(source_a_dir / f"{condition}_oof.npz"),
            _load_npz(source_b_dir / f"{condition}_oof.npz"),
        )
        if not np.array_equal(
            a["index"].astype(np.int64), locked_oof["index"].astype(np.int64)
        ):
            raise RuntimeError(
                f"masked condition {condition} does not cover locked OOF indices"
            )
        if not np.array_equal(
            a["fold"].astype(np.int64), locked_oof["fold"].astype(np.int64)
        ):
            raise RuntimeError(
                f"masked condition {condition} fold assignments differ from locked OOF"
            )
        weight, disagreement = locked_parameters(
            ensemble["fold_selection"], a["fold"], label_a, label_b
        )
        result = apply_locked_mask_blend(
            a, b, weight_a=weight, disagreement=disagreement
        )
        identity = (
            cache.metadata.iloc[result["index"]]["identity"].astype(str).to_numpy()
        )
        grouped = grouped_oof_metrics(
            result["target"],
            result["prediction"],
            identity,
            fold_ids=result["fold"],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=20260828,
        )
        grouped["risk_coverage"] = risk_coverage_curve(
            result["target"],
            result["prediction"],
            result["uncertainty"],
            identities=identity,
        )
        conditions[condition] = grouped
        condition_results[condition] = result
        if condition == "radars_123":
            full_result = result

    assert full_result is not None
    full_mask_consistency = validate_full_mask_consistency(
        full_result,
        locked_oof,
        prediction_atol_bpm=args.full_mask_prediction_atol_bpm,
    )
    ensemble_signature = str(ensemble.get("run_signature", ""))
    if not ensemble_signature:
        raise RuntimeError("ensemble metrics have no run signature")
    if str(ensemble_config.get("run_signature", "")) != ensemble_signature:
        raise RuntimeError("ensemble run-config signature differs from metrics")
    for key, source in (("source_a", sources[0]), ("source_b", sources[1])):
        configured_path = _resolve_recorded_path(str(ensemble_config.get(key, "")))
        declared_path = _resolve_recorded_path(source["path"])
        if configured_path != declared_path:
            raise RuntimeError(
                f"ensemble {key} path differs between metrics and run config"
            )
    with np.load(ensemble_oof_path, allow_pickle=False) as handle:
        if "run_signature" not in handle:
            raise RuntimeError("locked ensemble OOF has no run signature")
        oof_signature = str(np.asarray(handle["run_signature"]).item())
    if oof_signature != ensemble_signature:
        raise RuntimeError("locked ensemble OOF signature differs from metrics")
    for condition, result in condition_results.items():
        np.savez_compressed(
            output_dir / f"{condition}_oof.npz",
            **result,
            ensemble_run_signature=np.asarray(ensemble_signature),
            source_a_run_signature=np.asarray(
                source_provenance[0]["source_run_signature"]
            ),
            source_b_run_signature=np.asarray(
                source_provenance[1]["source_run_signature"]
            ),
        )
    condition_input = {
        "index": full_result["index"],
        "target": full_result["target"],
        "prediction": full_result["prediction"],
        # _condition_summaries uses these for interval/error detection.  The
        # locked combined uncertainty is supplied as rr_std with unit quality.
        "rr_std": full_result["uncertainty"],
        "quality": np.ones_like(full_result["uncertainty"]),
    }
    stratified = _condition_summaries(condition_input, cache.metadata)
    # The ensemble score contains a fold-locked disagreement penalty and is a
    # ranking score, not a posterior standard deviation.  Sigma interval
    # coverage would therefore be dimensionally misleading.
    stratified["uncertainty_interval_coverage"] = {
        "status": "not_applicable",
        "reason": (
            "ensemble uncertainty is a validation-locked ranking score with "
            "an arbitrary disagreement scale, not a calibrated RR sigma"
        ),
    }
    provenance_audit = {
        "passed": True,
        "ensemble_dir": str(ensemble_dir),
        "ensemble_metrics": str(ensemble_metrics_path),
        "ensemble_metrics_sha256": _sha256(ensemble_metrics_path),
        "ensemble_run_config": str(ensemble_config_path),
        "ensemble_run_config_sha256": _sha256(ensemble_config_path),
        "ensemble_oof": str(ensemble_oof_path),
        "ensemble_oof_sha256": _sha256(ensemble_oof_path),
        "ensemble_run_signature": ensemble_signature,
        "source_runs": source_provenance,
        "canonical_cache_dir": str(args.cache_dir.resolve()),
        "canonical_valid_reference_rows": expected_rows,
        "canonical_identities": expected_identities,
        "all_masked_conditions_exact_index_and_fold_match": True,
        "full_mask_consistency": full_mask_consistency,
    }
    report = {
        "schema_version": 2,
        "method": "validation-locked two-SNN blend applied unchanged per radar mask",
        "selection_guarantee": ensemble["selection_guarantee"],
        "source_robustness": [str(source_a_dir), str(source_b_dir)],
        "provenance_audit": provenance_audit,
        "radar_conditions": conditions,
        "stratified": stratified,
        "latency_scope": (
            "prediction blend only; end-to-end component latency is reported "
            "by scripts/benchmark_e2e.py"
        ),
    }
    write_json(output_dir / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensemble-dir",
        type=Path,
        default=Path("artifacts/runs/ensemble_structured_exact"),
    )
    parser.add_argument(
        "--source-a-robustness",
        type=Path,
        default=Path("artifacts/robustness/final_structured_aux_s12"),
    )
    parser.add_argument(
        "--source-b-robustness",
        type=Path,
        default=Path(
            "artifacts/robustness/final_structured_exact_s12_deterministic"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/robustness/ensemble_structured_exact"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/cache/rf32s")
    )
    parser.add_argument(
        "--ensemble-prediction-key",
        default="prediction_uncalibrated",
        help="locked ensemble NPZ prediction field reproduced by the full-radar mask",
    )
    parser.add_argument(
        "--full-mask-prediction-atol-bpm",
        type=float,
        default=0.025,
        help="absolute re-inference tolerance against the locked ensemble OOF",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap samples must be positive")
    if (
        not np.isfinite(args.full_mask_prediction_atol_bpm)
        or args.full_mask_prediction_atol_bpm < 0
    ):
        parser.error("full-mask prediction tolerance must be finite and non-negative")
    for path in (
        args.ensemble_dir / "metrics.json",
        args.ensemble_dir / "run_config.json",
        args.ensemble_dir / "ensemble_oof.npz",
        args.source_a_robustness / "report.json",
        args.source_b_robustness / "report.json",
        args.cache_dir / "manifest.json",
    ):
        if not path.is_file():
            parser.error(f"required provenance input not found: {path}")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "conditions": list(result["radar_conditions"]),
                "method": result["method"],
                "provenance_audit_passed": result["provenance_audit"]["passed"],
            },
            ensure_ascii=False,
        )
    )
