#!/usr/bin/env python3
"""Regenerate the final machine-readable summary and report figures.

This script is intentionally analysis-only: it never trains a model and it
never edits source experiment artifacts.  It checks that the OOF rows used by
the neural and tree pipelines are aligned, reads the committed metric reports,
and writes deterministic figures plus ``artifacts/final_report.json``.

Examples
--------
From the repository root::

    .venv/bin/python scripts/make_report.py

Use non-default artifact locations::

    .venv/bin/python scripts/make_report.py \
        --primary-dir artifacts/runs/final_structured_aux_s12 \
        --output-dir artifacts/report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

# The managed workspace may expose a read-only ~/.config.  Set this before
# importing matplotlib so documented commands stay warning-free.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_rr_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "snn": "#1769AA",
    "ensemble": "#6A1B9A",
    "teacher": "#E07A1F",
    "trees": "#2E8B57",
    "classical": "#7A7A7A",
    "danger": "#C23B22",
    "neutral": "#AAB4BE",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_keys(mapping: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"{label} is missing keys: {missing}")


def _configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9DEE3",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "axes.unicode_minus": False,
            "svg.hashsalt": "snn-rr-final-report",
        }
    )


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=240,
        metadata={"Software": "SnnProject scripts/make_report.py"},
    )
    fig.savefig(
        output_dir / f"{stem}.svg",
        metadata={"Date": None, "Creator": "SnnProject scripts/make_report.py"},
    )
    plt.close(fig)


def _load_aligned_oof(primary_npz: Path, baseline_npz: Path) -> dict[str, np.ndarray]:
    with np.load(primary_npz, allow_pickle=False) as primary, np.load(
        baseline_npz, allow_pickle=False
    ) as baseline:
        _require_keys(primary, ("index", "target", "prediction", "fold"), "SNN OOF")
        _require_keys(
            baseline,
            ("cache_row_index", "target_rr_bpm", "identity", "fold"),
            "baseline OOF",
        )
        if not np.array_equal(primary["index"], baseline["cache_row_index"]):
            raise RuntimeError("SNN and baseline OOF cache-row order differs")
        if not np.allclose(primary["target"], baseline["target_rr_bpm"], atol=1e-5):
            raise RuntimeError("SNN and baseline OOF targets differ")
        if not np.array_equal(primary["fold"], baseline["fold"]):
            raise RuntimeError("SNN and baseline OOF folds differ")
        return {
            "index": primary["index"].copy(),
            "target": primary["target"].copy(),
            "prediction": primary["prediction"].copy(),
            "identity": baseline["identity"].astype(str),
            "fold": primary["fold"].copy(),
        }


def plot_oof_and_identity(
    oof: Mapping[str, np.ndarray],
    primary_metrics: Mapping[str, Any],
    output_dir: Path,
    *,
    method_label: str,
) -> None:
    target = np.asarray(oof["target"], dtype=float)
    prediction = np.asarray(oof["prediction"], dtype=float)
    per_identity = primary_metrics["per_identity"]
    ordered = sorted(per_identity, key=lambda name: per_identity[name]["mae"])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.15), gridspec_kw={"width_ratios": [1.05, 1.45]})
    ax = axes[0]
    limits = (5.5, 45.5)
    density = ax.hexbin(
        target,
        prediction,
        gridsize=44,
        extent=(*limits, *limits),
        mincnt=1,
        bins="log",
        cmap="Blues",
        linewidths=0.15,
    )
    ax.plot(limits, limits, color="#222222", linewidth=1.25, label="Identity line")
    ax.fill_between(
        limits,
        np.asarray(limits) - 2,
        np.asarray(limits) + 2,
        color=COLORS["snn"],
        alpha=0.10,
        label="±2 bpm",
    )
    ax.set(xlim=limits, ylim=limits, xlabel="Reference RR (bpm)", ylabel="Predicted RR (bpm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("A. Identity-disjoint OOF predictions")
    overall = primary_metrics["overall"]
    ax.text(
        0.035,
        0.965,
        f"n={int(overall['n']):,}\nMAE={overall['mae']:.3f} bpm\nCCC={overall['ccc']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CAD2D9", "alpha": 0.94},
    )
    ax.legend(loc="lower right")
    colorbar = fig.colorbar(density, ax=ax, fraction=0.047, pad=0.03)
    colorbar.set_label("Window density (log count)")

    ax = axes[1]
    values = np.asarray([per_identity[name]["mae"] for name in ordered])
    counts = np.asarray([int(per_identity[name]["n"]) for name in ordered])
    bar_colors = [
        COLORS["danger"] if value >= 2.0 else COLORS["snn"] if value <= 1.0 else "#5B8DB8"
        for value in values
    ]
    positions = np.arange(len(ordered))
    ax.bar(positions, values, color=bar_colors, width=0.76)
    ax.axhline(
        primary_metrics["identity_macro"]["macro_mae"],
        color="#202020",
        linestyle="--",
        linewidth=1.1,
        label=f"Macro MAE {primary_metrics['identity_macro']['macro_mae']:.3f}",
    )
    for x, value, count in zip(positions, values, counts):
        ax.text(x, value + 0.055, f"{count}", ha="center", va="bottom", fontsize=7.2, rotation=90)
    ax.set_xticks(positions, ordered, rotation=55, ha="right")
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("B. Per-identity error (label = valid windows)")
    ax.set_ylim(0, max(3.0, float(values.max()) + 0.5))
    ax.legend(loc="upper left")
    fig.suptitle(
        f"{method_label} — six-fold out-of-fold evaluation",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "oof_scatter_identity")


def _curve_xy(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    points = sorted(
        ((100.0 * float(row["coverage"]), float(row[key])) for row in rows),
        key=lambda pair: pair[0],
    )
    return np.asarray([point[0] for point in points]), np.asarray([point[1] for point in points])


def plot_risk_coverage(
    primary_metrics: Mapping[str, Any],
    teacher_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    output_dir: Path,
    ensemble_metrics: Mapping[str, Any] | None,
) -> None:
    methods: list[tuple[str, Sequence[Mapping[str, Any]], str, str]] = [
        ("Structured-aux SNN (12 steps)", primary_metrics["risk_coverage"], COLORS["snn"], "o-"),
        ("ANN teacher", teacher_metrics["risk_coverage"], COLORS["teacher"], "s-"),
        (
            "ExtraTrees ensemble",
            baseline_metrics["methods"]["extratrees_ensemble"]["risk_coverage"],
            COLORS["trees"],
            "^-",
        ),
        (
            "Classical spectral",
            baseline_metrics["methods"]["cached_classical"]["risk_coverage"],
            COLORS["classical"],
            "D--",
        ),
    ]
    if ensemble_metrics is not None:
        methods.insert(
            0,
            (
                "Validation-locked structured 2-SNN ensemble",
                ensemble_metrics["risk_coverage"]["uncalibrated"],
                COLORS["ensemble"],
                "P-",
            ),
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.85))
    for label, rows, color, style in methods:
        x, y = _curve_xy(rows, "mae")
        axes[0].plot(x, y, style, color=color, linewidth=1.9, markersize=5.2, label=label)
        x, y = _curve_xy(rows, "macro_mae")
        axes[1].plot(x, y, style, color=color, linewidth=1.9, markersize=5.2, label=label)
    for ax in axes:
        ax.axvline(70, color="#222222", linestyle=":", linewidth=1.0)
        ax.set_xlim(18, 102)
        ax.set_xticks([20, 30, 50, 70, 80, 90, 100])
        ax.set_xlabel("Retained coverage among valid-reference windows (%)")
        ax.set_ylabel("MAE (bpm)")
    axes[0].set_title("A. Window-weighted selective risk")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("B. Identity-macro selective risk")
    axes[1].set_ylim(bottom=0)
    axes[0].legend(loc="upper left", fontsize=8.3)
    fig.suptitle("Retrospective risk–coverage curves (model-specific uncertainty ranking)", fontsize=14, y=1.01)
    fig.text(
        0.5,
        -0.01,
        "Coverage is conditional on reference-valid windows; quantile thresholds were not prospectively locked.",
        ha="center",
        fontsize=9,
        color="#4D5964",
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "risk_coverage")


def plot_radar_robustness(
    robustness: Mapping[str, Any],
    output_dir: Path,
    *,
    method_label: str,
) -> None:
    conditions = ["radars_123", "radars_12", "radars_13", "radars_23", "radar_1", "radar_2", "radar_3"]
    labels = ["1+2+3", "1+2", "1+3", "2+3", "1 only", "2 only", "3 only"]
    records = robustness["radar_conditions"]
    overall = np.asarray([records[key]["overall"]["mae"] for key in conditions])
    macro = np.asarray([records[key]["identity_macro"]["macro_mae"] for key in conditions])
    x = np.arange(len(labels))
    width = 0.37

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars1 = ax.bar(x - width / 2, overall, width, color=COLORS["snn"], label="Overall MAE")
    bars2 = ax.bar(x + width / 2, macro, width, color="#8FB4D3", label="Identity-macro MAE")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
            )
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE (bpm)")
    ax.set_xlabel("Available radar views at inference")
    ax.set_ylim(0, max(2.0, float(max(overall.max(), macro.max())) + 0.25))
    ax.set_title(f"{method_label} robustness to missing radar views")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save_figure(fig, output_dir, "radar_robustness")


def _bar_failure_panel(
    ax: plt.Axes,
    labels: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    title: str,
) -> None:
    mae = np.asarray([float(record["mae"]) for record in records])
    catastrophic = np.asarray([100.0 * float(record["catastrophic_over_5"]) for record in records])
    counts = np.asarray([int(record["n"]) for record in records])
    colors = [COLORS["danger"] if value >= 2.0 else COLORS["snn"] for value in mae]
    x = np.arange(len(labels))
    bars = ax.bar(x, mae, color=colors, width=0.7)
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylabel("MAE (bpm)")
    ax.set_title(title)
    ax.set_ylim(0, max(2.1, float(mae.max()) + 0.85))
    for bar, value, failure, count in zip(bars, mae, catastrophic, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.06,
            f"n={count}\n>5: {failure:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )


def plot_failure_analysis(
    robustness: Mapping[str, Any],
    output_dir: Path,
    *,
    method_label: str,
) -> None:
    stratified = robustness["stratified"]
    rr_order = ["6_10_bpm", "10_15_bpm", "15_20_bpm", "20_25_bpm", "25_35_bpm", "35_46_bpm"]
    rr_labels = ["6–10", "10–15", "15–20", "20–25", "25–35", "35–46"]
    protocol_order = ["Dodge", "Kick", "Strike"]
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.9), gridspec_kw={"width_ratios": [1.55, 1]})
    _bar_failure_panel(
        axes[0],
        rr_labels,
        [stratified["per_rr_band"][key] for key in rr_order],
        "A. Error by reference RR band",
    )
    axes[0].set_xlabel("Reference RR band (bpm)")
    _bar_failure_panel(
        axes[1],
        protocol_order,
        [stratified["per_protocol"][key] for key in protocol_order],
        "B. Error by activity protocol",
    )
    axes[1].set_xlabel("Protocol")
    fig.suptitle(
        f"{method_label} failure-mode audit (full valid-reference OOF)",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "failure_analysis")


def _metric_subset(metrics: Mapping[str, Any]) -> dict[str, float]:
    keys = (
        "n",
        "mae",
        "rmse",
        "median_ae",
        "p95_ae",
        "bias",
        "ccc",
        "within_1",
        "within_2",
        "within_3",
        "catastrophic_over_5",
    )
    return {key: float(metrics[key]) for key in keys if key in metrics}


def _risk_subset(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    keys = (
        "requested_coverage",
        "coverage",
        "n",
        "n_identities",
        "mae",
        "macro_mae",
        "rmse",
        "p95_ae",
        "within_2",
        "catastrophic_over_5",
        "uncertainty_threshold",
    )
    return [
        {key: float(row[key]) for key in keys if key in row}
        for row in rows
    ]


def build_final_summary(
    *,
    primary_metrics: Mapping[str, Any],
    teacher_metrics: Mapping[str, Any],
    compact_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    robustness: Mapping[str, Any],
    accuracy_robustness: Mapping[str, Any],
    cache_manifest: Mapping[str, Any],
    ensemble_metrics: Mapping[str, Any] | None,
    commercial_goal: Mapping[str, Any] | None,
    alias_audit: Mapping[str, Any] | None,
    sources: Mapping[str, Path],
) -> dict[str, Any]:
    sessions = cache_manifest["sessions"]
    ok_sessions = [entry for entry in sessions if entry.get("status") == "ok"]
    skipped_sessions = [entry for entry in sessions if entry.get("status") != "ok"]
    reference = robustness["reference_coverage"]
    primary_risk = _risk_subset(primary_metrics["risk_coverage"])
    if ensemble_metrics is None:
        gate_risk = primary_risk
        gate_candidate = "primary structured-aux SNN"
    else:
        gate_risk = _risk_subset(ensemble_metrics["risk_coverage"]["uncalibrated"])
        gate_candidate = "validation-locked structured two-SNN ensemble"
    risk_70 = min(gate_risk, key=lambda row: abs(row["requested_coverage"] - 0.7))
    gate_targets = {
        "mae_max_bpm": 1.0,
        "rmse_max_bpm": 1.5,
        "within_2_min": 0.95,
        "p95_ae_max_bpm": 3.0,
        "catastrophic_over_5_max": 0.01,
    }
    gate_checks = {
        "mae": risk_70["mae"] <= gate_targets["mae_max_bpm"],
        "rmse": risk_70["rmse"] <= gate_targets["rmse_max_bpm"],
        "within_2": risk_70["within_2"] >= gate_targets["within_2_min"],
        "p95_ae": risk_70["p95_ae"] <= gate_targets["p95_ae_max_bpm"],
        "catastrophic_over_5": risk_70["catastrophic_over_5"] <= gate_targets["catastrophic_over_5_max"],
    }
    methods = baseline_metrics["methods"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "declared_full_coverage_commercial_goal_not_met",
        "dataset": {
            "paired_sessions": len(ok_sessions),
            "skipped_sessions": skipped_sessions,
            "identities": 18,
            "candidate_windows": int(reference["total_windows"]),
            "valid_reference_windows": int(reference["valid_windows"]),
            "valid_reference_fraction": float(reference["fraction"]),
            "window_seconds": 32.0,
            "stride_seconds": 4.0,
            "radars": 3,
            "cache_config_sha256": cache_manifest["config_sha256"],
            "cache_pipeline_sha256": cache_manifest["pipeline_sha256"],
        },
        "evaluation": {
            "outer_folds": int(primary_metrics["expected_folds"]),
            "split_unit": "physical identity",
            "complete_oof": bool(primary_metrics["complete_oof"]),
            "overlap_caveat": "32 s windows use a 4 s stride; rows within an identity are correlated",
            "selection_caveat": "model iterations used this cohort; prospective locked-cohort validation remains required",
        },
        "primary_single_snn": {
            "name": "TriRadarRRSNN structured auxiliary, 12 simulation steps",
            "parameters": int(primary_metrics["n_parameters"]),
            "full_valid_reference_oof": _metric_subset(primary_metrics["overall"]),
            "identity_macro": primary_metrics["identity_macro"],
            "identity_cluster_bootstrap_mae": primary_metrics["identity_cluster_bootstrap_mae"],
            "selective_risk": primary_risk,
            "quality_classifier": primary_metrics["quality_classifier"],
            "mean_spike_rate": float(primary_metrics["spike_activity"]["mean_rate"]),
        },
        "comparators": {
            "ann_teacher": {
                "parameters": int(teacher_metrics["n_parameters"]),
                "overall": _metric_subset(teacher_metrics["overall"]),
                "macro_mae": float(teacher_metrics["identity_macro"]["macro_mae"]),
            },
            "compact_snn_8_steps": {
                "parameters": int(compact_metrics["n_parameters"]),
                "overall": _metric_subset(compact_metrics["overall"]),
                "macro_mae": float(compact_metrics["identity_macro"]["macro_mae"]),
            },
            "extratrees_ensemble": {
                "overall": _metric_subset(methods["extratrees_ensemble"]["overall"]),
                "macro_mae": float(methods["extratrees_ensemble"]["identity_macro"]["macro_mae"]),
            },
            "cached_classical": {
                "overall": _metric_subset(methods["cached_classical"]["overall"]),
                "macro_mae": float(methods["cached_classical"]["identity_macro"]["macro_mae"]),
            },
        },
        "robustness": {
            "radar_conditions": {
                key: {
                    "mae": float(value["overall"]["mae"]),
                    "macro_mae": float(value["identity_macro"]["macro_mae"]),
                }
                for key, value in accuracy_robustness["radar_conditions"].items()
            },
            "nonoverlapping_32s_windows": accuracy_robustness["stratified"]["nonoverlapping_32s_windows"],
            "per_protocol": accuracy_robustness["stratified"]["per_protocol"],
            "per_rr_band": accuracy_robustness["stratified"]["per_rr_band"],
            "error_detection": accuracy_robustness["stratified"]["error_detection"],
            "uncertainty_interval_coverage": accuracy_robustness["stratified"]["uncertainty_interval_coverage"],
        },
        "deployment_benchmark": {
            "checkpoint_size_mb": float(robustness["model"]["checkpoint_size_mb"]),
            "latency": robustness["latency"],
            "scope": "model forward only; feature extraction and I/O excluded",
        },
        "internal_selective_gate": {
            "scope": "retrospective 70% coverage among valid-reference windows only; not a commercial acceptance test",
            "candidate": gate_candidate,
            "targets": gate_targets,
            "achieved": risk_70,
            "checks": gate_checks,
            "all_checks_pass": bool(all(gate_checks.values())),
            "accepted_fraction_of_all_candidate_windows": float(reference["fraction"] * risk_70["coverage"]),
        },
        "commercial_readiness": {
            "claim": "not established",
            "blocking_evidence": [
                "only 18 identities from one retrospective laboratory cohort",
                "reference RR is derived from quality-controlled BIOPAC respiration, not an independent capnography adjudication",
                "only 24.3% of candidate windows have a valid reference label",
                "outer-fold results were observed during model iteration",
                "no prospective locked cohort, demographic coverage, clinical subgroup analysis, or target-device/worst-case end-to-end benchmark",
                "the ensemble uncertainty is only a ranking score, not a calibrated RR interval, and a deployment abstention threshold has not been prospectively locked",
            ],
        },
        "artifact_sources": {
            name: {"path": _relative(path), "sha256": _sha256(path)}
            for name, path in sources.items()
            if path.is_file()
        },
    }
    if ensemble_metrics is not None:
        ensemble = ensemble_metrics["grouped_metrics"]["uncalibrated"]
        summary["locked_two_snn_ensemble"] = {
            "role": "accuracy leader; see the commercial-goal audit for masked-radar and end-to-end component benchmarks",
            "selection": ensemble_metrics["selection_guarantee"],
            "overall": _metric_subset(ensemble["overall"]),
            "identity_macro": ensemble["identity_macro"],
            "identity_cluster_bootstrap_mae": ensemble["identity_cluster_bootstrap_mae"],
            "selective_risk": _risk_subset(ensemble_metrics["risk_coverage"]["uncalibrated"]),
        }
    if commercial_goal is not None:
        summary["declared_commercial_goal"] = {
            "candidate": commercial_goal["candidate"],
            "goal": commercial_goal["goal"],
            "full_valid_reference_oof": commercial_goal["metrics"]["full"],
            "high_rr_25_35": commercial_goal["metrics"]["high_rr_25_35"],
            "end_to_end_benchmark": commercial_goal.get("end_to_end_benchmark"),
        }
    if alias_audit is not None:
        variants = alias_audit["metrics"]
        summary["rejected_alias_iteration"] = {
            "conclusion": alias_audit["conclusion"],
            "alias_gate_snn": {
                "full": variants["alias_gate_snn_rr_diagnostic"]["full"],
                "high_rr_25_35": variants["alias_gate_snn_rr_diagnostic"]["high_rr_25_35"],
            },
            "validation_locked_three_way": {
                "full": variants["validation_locked_blend"]["full"],
                "high_rr_25_35": variants["validation_locked_blend"]["high_rr_25_35"],
            },
            "validation_locked_causal_decoder": {
                "full": variants["validation_locked_causal_alias_decoder"]["full"],
                "high_rr_25_35": variants["validation_locked_causal_alias_decoder"]["high_rr_25_35"],
            },
        }
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-dir",
        type=Path,
        default=Path("artifacts/runs/final_structured_aux_s12"),
    )
    parser.add_argument("--compact-dir", type=Path, default=Path("artifacts/runs/final_compact_s8"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baselines/final"))
    parser.add_argument(
        "--robustness-report",
        type=Path,
        default=Path("artifacts/robustness/final_structured_aux_s12/report.json"),
        help="single-model robustness report used for reference coverage and forward benchmark provenance",
    )
    parser.add_argument(
        "--accuracy-robustness-report",
        type=Path,
        default=Path("artifacts/robustness/ensemble_structured_exact/report.json"),
        help="robustness report for the headline accuracy candidate; falls back to --robustness-report",
    )
    parser.add_argument("--cache-manifest", type=Path, default=Path("artifacts/cache/rf32s/manifest.json"))
    parser.add_argument(
        "--ensemble-dir",
        type=Path,
        default=Path("artifacts/runs/ensemble_structured_exact"),
        help="optional validation-locked ensemble; ignored when metrics.json is absent",
    )
    parser.add_argument(
        "--commercial-goal-report",
        type=Path,
        default=Path("artifacts/commercial_goal_report.json"),
        help="optional predeclared acceptance audit embedded in the final JSON",
    )
    parser.add_argument(
        "--alias-audit-report",
        type=Path,
        default=Path("artifacts/runs/causal_alias_decoder/with_alias_gate/metrics.json"),
        help="optional final alias-head/stack/causal-decoder audit",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/report"))
    parser.add_argument("--summary-json", type=Path, default=Path("artifacts/final_report.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else PROJECT_ROOT / path
    primary_dir = resolve(args.primary_dir)
    compact_dir = resolve(args.compact_dir)
    baseline_dir = resolve(args.baseline_dir)
    robustness_path = resolve(args.robustness_report)
    accuracy_robustness_path = resolve(args.accuracy_robustness_report)
    cache_manifest_path = resolve(args.cache_manifest)
    ensemble_dir = resolve(args.ensemble_dir)
    commercial_goal_path = resolve(args.commercial_goal_report)
    alias_audit_path = resolve(args.alias_audit_report)
    output_dir = resolve(args.output_dir)
    summary_json = resolve(args.summary_json)

    source_paths = {
        "primary_metrics": primary_dir / "snn_metrics.json",
        "primary_oof": primary_dir / "snn_oof.npz",
        "teacher_metrics": primary_dir / "teacher_metrics.json",
        "compact_metrics": compact_dir / "snn_metrics.json",
        "baseline_metrics": baseline_dir / "metrics.json",
        "baseline_oof": baseline_dir / "oof_predictions.npz",
        "robustness_report": robustness_path,
        "cache_manifest": cache_manifest_path,
    }
    primary_metrics = _read_json(source_paths["primary_metrics"])
    teacher_metrics = _read_json(source_paths["teacher_metrics"])
    compact_metrics = _read_json(source_paths["compact_metrics"])
    baseline_metrics = _read_json(source_paths["baseline_metrics"])
    robustness = _read_json(source_paths["robustness_report"])
    if accuracy_robustness_path.is_file():
        source_paths["accuracy_robustness_report"] = accuracy_robustness_path
        accuracy_robustness = _read_json(accuracy_robustness_path)
    else:
        accuracy_robustness = robustness
    cache_manifest = _read_json(source_paths["cache_manifest"])
    ensemble_metrics_path = ensemble_dir / "metrics.json"
    ensemble_metrics = _read_json(ensemble_metrics_path) if ensemble_metrics_path.is_file() else None
    commercial_goal = _read_json(commercial_goal_path) if commercial_goal_path.is_file() else None
    if commercial_goal is not None:
        source_paths["commercial_goal_report"] = commercial_goal_path
    alias_audit = _read_json(alias_audit_path) if alias_audit_path.is_file() else None
    if alias_audit is not None:
        source_paths["alias_audit_report"] = alias_audit_path
    if ensemble_metrics is not None:
        source_paths["locked_ensemble_metrics"] = ensemble_metrics_path
        ensemble_oof_path = ensemble_dir / "ensemble_oof.npz"
        if ensemble_oof_path.is_file():
            source_paths["locked_ensemble_oof"] = ensemble_oof_path

    if not primary_metrics.get("complete_oof"):
        raise RuntimeError("primary SNN report is not a complete OOF result")
    headline_oof_path = source_paths.get("locked_ensemble_oof", source_paths["primary_oof"])
    oof = _load_aligned_oof(headline_oof_path, source_paths["baseline_oof"])
    if len(oof["target"]) != int(primary_metrics["expected_valid_rows"]):
        raise RuntimeError("OOF row count does not match primary metric provenance")

    _configure_plotting()
    if ensemble_metrics is None or "locked_ensemble_oof" not in source_paths:
        headline_metrics = primary_metrics
        headline_label = "Structured-aux 12-step SNN"
    else:
        headline_metrics = ensemble_metrics["grouped_metrics"]["uncalibrated"]
        headline_label = "Validation-locked structured two-SNN ensemble"
    plot_oof_and_identity(
        oof,
        headline_metrics,
        output_dir,
        method_label=headline_label,
    )
    plot_risk_coverage(primary_metrics, teacher_metrics, baseline_metrics, output_dir, ensemble_metrics)
    plot_radar_robustness(
        accuracy_robustness,
        output_dir,
        method_label=headline_label,
    )
    plot_failure_analysis(
        accuracy_robustness,
        output_dir,
        method_label=headline_label,
    )

    summary = build_final_summary(
        primary_metrics=primary_metrics,
        teacher_metrics=teacher_metrics,
        compact_metrics=compact_metrics,
        baseline_metrics=baseline_metrics,
        robustness=robustness,
        accuracy_robustness=accuracy_robustness,
        cache_manifest=cache_manifest,
        ensemble_metrics=ensemble_metrics,
        commercial_goal=commercial_goal,
        alias_audit=alias_audit,
        sources=source_paths,
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": _relative(summary_json),
                "figures": sorted(path.name for path in output_dir.glob("*.png")),
                "ensemble_included": ensemble_metrics is not None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
