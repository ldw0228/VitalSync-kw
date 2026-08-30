#!/usr/bin/env python3
"""Train a causal all-window episode SNN with a locked identity protocol.

Every session is presented in chronological 4-second-window order.  Invalid
reference windows still update the spiking state from deployment-time radar
evidence, while the default loss is applied only to the 2,327 frozen valid
references.  ``--weak-invalid-weight`` enables an explicit train-only noisy
label ablation; validation and test metrics remain strictly valid-only.

Outer fold ``f`` is test, ``(f+1)%6`` is validation, and the other four frozen
identity folds fit every model parameter, feature normalization statistic, and
action threshold.  A durable validation selection lock is written before the
test episode loader is constructed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import train_svd_snn as shared  # noqa: E402
from snn_rr.models import gaussian_soft_targets  # noqa: E402
from snn_rr.svd_episode_models import EpisodeAliasRRSNN  # noqa: E402


SCHEMA_VERSION = 1
N_FOLDS = shared.N_FOLDS
EVIDENCE_NAMES = (
    "classical_weighted_power",
    "classical_max_power",
    "classical_peak_consensus",
    "classical_quality_consensus",
    "own_seed_weighted_power",
    "own_seed_peak_consensus",
    "cross_seed_max_power",
    "cross_seed_mean_power",
    "classical_seed_agreement",
    "candidate_in_band",
)
CONTEXT_NAMES = ("classical_rr_bpm", "classical_confidence", "radar_peak_spread_bpm")


@dataclass(slots=True)
class EpisodeSession:
    session_id: str
    identity: str
    fold: int
    metadata: pd.DataFrame
    evidence: np.ndarray
    context: np.ndarray
    radar_mask: np.ndarray
    manifest: dict[str, Any]
    source_hashes: dict[str, str]


@dataclass(slots=True)
class EpisodeExperiment:
    cache_root: Path
    metadata: pd.DataFrame
    sessions: list[EpisodeSession]
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EpisodeFeatureScaler:
    evidence_center: np.ndarray
    evidence_scale: np.ndarray
    context_center: np.ndarray
    context_scale: np.ndarray
    fit_positions_sha256: str

    def transform_evidence(self, values: np.ndarray) -> np.ndarray:
        result = (np.nan_to_num(values) - self.evidence_center) / self.evidence_scale
        return np.clip(result, -8.0, 8.0).astype(np.float32, copy=False)

    def transform_context(self, values: np.ndarray) -> np.ndarray:
        result = (np.nan_to_num(values) - self.context_center) / self.context_scale
        return np.clip(result, -8.0, 8.0).astype(np.float32, copy=False)

    def record(self) -> dict[str, Any]:
        return {
            "evidence_names": list(EVIDENCE_NAMES),
            "context_names": list(CONTEXT_NAMES),
            "evidence_center": self.evidence_center.reshape(-1).tolist(),
            "evidence_scale": self.evidence_scale.reshape(-1).tolist(),
            "context_center": self.context_center.reshape(-1).tolist(),
            "context_scale": self.context_scale.reshape(-1).tolist(),
            "fit_positions_sha256": self.fit_positions_sha256,
            "fit_scope": "all chronological windows from outer-training identities only",
        }


@dataclass(frozen=True, slots=True)
class TrainActionCalibration:
    divisor_temperature_bpm: float
    gate_margin_bpm: float
    gate_temperature_bpm: float
    fit_positions_sha256: str
    valid_rows: int
    divisor_class_balance_power: float = 0.0
    divisor_class_counts: tuple[int, int, int, int] = (0, 0, 0, 0)
    divisor_class_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    def record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "fit_scope": "valid references from outer-training identities only",
        }


@dataclass(frozen=True, slots=True)
class StrictGatePolicy:
    threshold: float
    correction_pull: float
    validation_coverage: float
    validation_macro_mae: float
    validation_tail_macro_mae: float | None
    safety_gates: dict[str, bool]
    candidate_count: int

    def record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "fit_scope": "outer-validation identities/valid references only",
            "score": "base-independent learned gate sigmoid",
            "application": (
                "gate>=threshold applies fixed correction_pull from frozen base "
                "posterior mean toward source mean"
            ),
        }


@dataclass(frozen=True, slots=True)
class EpisodePrediction:
    position: np.ndarray
    cache_index: np.ndarray
    target: np.ndarray
    base_prediction: np.ndarray
    base_std: np.ndarray
    # Preserve the pre-sanitization availability bit.  Sanitized 0/4 values
    # are valid numeric placeholders for the neural forward pass, but must
    # never make a truly missing base look available to the locked policy.
    base_available: np.ndarray
    candidate_prediction: np.ndarray
    rr_std: np.ndarray
    source_prediction: np.ndarray
    source_std: np.ndarray
    mixture_gate: np.ndarray
    learned_gate: np.ndarray
    applied_gate: np.ndarray
    divisor_probabilities: np.ndarray
    residual_rr: np.ndarray
    candidate_std: np.ndarray
    quality: np.ndarray
    radar_weights: np.ndarray
    spike_rate: np.ndarray


def _positions_digest(positions: Sequence[int] | np.ndarray) -> str:
    values = np.asarray(positions, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    shared.atomic_write_json(path, _json_ready(value))


def _nearest_frequency_index(frequencies: np.ndarray, centers_bpm: np.ndarray) -> np.ndarray:
    centers_hz = np.asarray(centers_bpm, dtype=np.float32) / 60.0
    right = np.searchsorted(frequencies, centers_hz, side="left")
    right = np.clip(right, 0, len(frequencies) - 1)
    left = np.clip(right - 1, 0, len(frequencies) - 1)
    choose_left = np.abs(frequencies[left] - centers_hz) <= np.abs(
        frequencies[right] - centers_hz
    )
    return np.where(choose_left, left, right).astype(np.int64)


def compute_candidate_evidence(
    spectra: np.ndarray,
    attributes: np.ndarray,
    frequencies_hz: np.ndarray,
    classical_rr: np.ndarray,
    radar_peaks: np.ndarray,
    *,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive compact label-free x1..x4 evidence from cached SVD spectra.

    The function never accepts a reference target or QC value.  Power is
    normalized independently for each component, and component consensus is
    weighted only by the five SVD diagnostics stored in the label-free cache.
    """

    if spectra.ndim != 5 or attributes.shape != (*spectra.shape[:-1], 5):
        raise ValueError("spectra/attributes shapes must be [N, R, V, C, F/5]")
    rows, radars, _, _, _ = spectra.shape
    if classical_rr.shape != (rows,) or radar_peaks.shape != (rows, radars):
        raise ValueError("classical_rr/radar_peaks shape mismatch")
    frequencies = np.asarray(frequencies_hz, dtype=np.float32)
    if frequencies.ndim != 1 or len(frequencies) != spectra.shape[-1]:
        raise ValueError("frequency grid does not match spectra")
    output = np.zeros((rows, radars, 4, len(EVIDENCE_NAMES)), dtype=np.float32)
    radar_mask = np.isfinite(radar_peaks) & (radar_peaks > 0)
    for start in range(0, rows, int(chunk_size)):
        stop = min(rows, start + int(chunk_size))
        power = np.asarray(spectra[start:stop], dtype=np.float32)
        maximum = power.max(axis=-1, keepdims=True)
        power = power / np.maximum(maximum, 1.0e-6)
        attr = np.asarray(attributes[start:stop], dtype=np.float32)
        quality = (
            np.sqrt(np.maximum(attr[..., 0] * attr[..., 1], 1.0e-8))
            * attr[..., 2]
            * (1.0 - 0.5 * np.clip(attr[..., 3], 0.0, 1.0))
        )
        component_peak_bpm = attr[..., 4] * 60.0
        local_classical = classical_rr[start:stop]
        local_peaks = radar_peaks[start:stop]
        for multiplier_index, multiplier in enumerate((1.0, 2.0, 3.0, 4.0)):
            classical_center = local_classical * multiplier
            classical_index = _nearest_frequency_index(frequencies, classical_center)
            classical_power = np.take_along_axis(
                power,
                classical_index[:, None, None, None, None],
                axis=-1,
            )[..., 0]
            quality_sum = quality.sum(axis=(-2, -1)).clip(min=1.0e-6)
            weighted_power = (classical_power * quality).sum(axis=(-2, -1)) / quality_sum
            peak_match = np.abs(component_peak_bpm - classical_center[:, None, None, None]) <= 0.75
            peak_consensus = peak_match.mean(axis=(-2, -1))
            quality_consensus = (peak_match * quality).sum(axis=(-2, -1)) / quality_sum

            own_center = local_peaks * multiplier
            own_index = _nearest_frequency_index(frequencies, own_center)
            own_power = np.take_along_axis(
                power, own_index[:, :, None, None, None], axis=-1
            )[..., 0]
            own_weighted = (own_power * quality).sum(axis=(-2, -1)) / quality_sum
            own_consensus = (
                np.abs(component_peak_bpm - own_center[:, :, None, None]) <= 0.75
            ).mean(axis=(-2, -1))

            cross_values: list[np.ndarray] = []
            for seed_radar in range(radars):
                seed_center = local_peaks[:, seed_radar] * multiplier
                seed_index = _nearest_frequency_index(frequencies, seed_center)
                seed_power = np.take_along_axis(
                    power, seed_index[:, None, None, None, None], axis=-1
                )[..., 0]
                cross_values.append(
                    (seed_power * quality).sum(axis=(-2, -1)) / quality_sum
                )
            cross = np.stack(cross_values, axis=-1)
            finite_seed = np.isfinite(local_peaks)[:, None, :]
            cross_safe = np.where(finite_seed, cross, np.nan)
            with np.errstate(invalid="ignore"):
                cross_max = np.nanmax(cross_safe, axis=-1)
                cross_mean = np.nanmean(cross_safe, axis=-1)
            agreement = np.exp(-np.abs(own_center - classical_center[:, None]) / 2.0)
            in_band = (
                np.isfinite(classical_center)
                & (classical_center >= frequencies[0] * 60.0)
                & (classical_center <= frequencies[-1] * 60.0)
            )
            output[start:stop, :, multiplier_index, :] = np.stack(
                (
                    weighted_power,
                    classical_power.max(axis=(-2, -1)),
                    peak_consensus,
                    quality_consensus,
                    own_weighted,
                    own_consensus,
                    cross_max,
                    cross_mean,
                    agreement,
                    np.broadcast_to(in_band[:, None], own_weighted.shape),
                ),
                axis=-1,
            )
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
    output[..., :8] = np.clip(output[..., :8], 0.0, 1.0)
    output[..., 8:] = np.clip(output[..., 8:], 0.0, 1.0)
    radar_mask &= np.isfinite(output).all(axis=(-2, -1))
    output *= radar_mask[:, :, None, None]
    return output.astype(np.float32, copy=False), radar_mask


def _resolve_all_window_prediction(path: Path | None) -> Path | None:
    if path is None or not path.exists():
        return None
    if path.is_file():
        return path
    preferred = (
        "all_window_oof.csv",
        "all_windows_oof.csv",
        "snn_all_windows.csv",
        "oof_predictions.csv",
    )
    for name in preferred:
        candidate = path / name
        if candidate.is_file():
            return candidate
    candidates = sorted(path.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    # The deployment inference job writes one immutable NPZ per outer fold.
    # Returning the directory lets the loader combine completed folds without
    # ever reading their target/reference arrays.
    return path if list(path.glob("fold_*_all_windows.npz")) else None


def _read_all_window_prediction(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if path.is_file():
        frame = pd.read_csv(path)
        file_hash = shared.sha256_file(path)
        provenance_path = path.parent / "provenance.json"
        provenance_record: dict[str, Any] | None = None
        if provenance_path.is_file():
            provenance_record = json.loads(provenance_path.read_text(encoding="utf-8"))
            recorded_outputs = provenance_record.get("outputs", {})
            if recorded_outputs.get("csv_sha256") != file_hash:
                raise RuntimeError(
                    "all-window CSV SHA-256 does not match its provenance record"
                )
        return frame, {
            "format": "csv",
            "files": {str(path.resolve()): file_hash},
            "run_signatures": sorted(frame["run_signature"].astype(str).unique().tolist())
            if "run_signature" in frame
            else [],
            "inference_signatures": sorted(
                frame["inference_signature"].astype(str).unique().tolist()
            )
            if "inference_signature" in frame
            else [],
            "provenance_path": str(provenance_path.resolve())
            if provenance_record is not None
            else None,
            "provenance_sha256": shared.sha256_file(provenance_path)
            if provenance_record is not None
            else None,
            "deployment_freeze_eligible": provenance_record.get(
                "deployment_freeze_eligible"
            )
            if provenance_record is not None
            else None,
            "semantic_binding_source": "explicit CSV row metadata",
        }
    parts: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    marker_hashes: dict[str, str] = {}
    signatures: set[str] = set()
    for fold_path in sorted(path.glob("fold_*_all_windows.npz")):
        marker_path = fold_path.with_name(fold_path.stem + ".verified.json")
        if not marker_path.is_file():
            raise RuntimeError(
                f"all-window fold has no verified commit marker: {marker_path}"
            )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        artifact = marker.get("artifact", {})
        fold_hash = shared.sha256_file(fold_path)
        if artifact.get("sha256") != fold_hash or int(
            artifact.get("bytes", -1)
        ) != fold_path.stat().st_size:
            raise RuntimeError(f"all-window fold/marker hash mismatch: {fold_path}")
        with np.load(fold_path, allow_pickle=False) as values:
            required = {
                "index",
                "prediction",
                "rr_std",
                "fold",
                "run_signature",
                "inference_signature",
                "checkpoint_sha256",
            }
            if not required.issubset(values.files):
                raise RuntimeError(f"{fold_path} lacks deployment prediction arrays")
            indices = np.asarray(values["index"], dtype=np.int64)
            expected_index_sha = hashlib.sha256(indices.tobytes()).hexdigest()
            if marker.get("expected_index_sha256") != expected_index_sha:
                raise RuntimeError(f"all-window fold index/marker mismatch: {fold_path}")
            declared_fold = np.asarray(values["fold"], dtype=np.int16)
            marker_fold = int(marker.get("fold", -1))
            if not np.all(declared_fold == marker_fold):
                raise RuntimeError(f"all-window fold labels do not match marker: {fold_path}")
            run_signature = str(np.asarray(values["run_signature"]).item())
            inference_signature = str(np.asarray(values["inference_signature"]).item())
            checkpoint_sha = str(np.asarray(values["checkpoint_sha256"]).item())
            for key, actual in (
                ("run_signature", run_signature),
                ("inference_signature", inference_signature),
                ("checkpoint_sha256", checkpoint_sha),
            ):
                if marker.get(key) != actual:
                    raise RuntimeError(
                        f"all-window fold {key}/marker mismatch: {fold_path}"
                    )
            data: dict[str, np.ndarray] = {
                "cache_index": indices,
                "prediction_bpm": np.asarray(values["prediction"], dtype=np.float32),
                "rr_std_bpm": np.asarray(values["rr_std"], dtype=np.float32),
                "fold": declared_fold,
            }
            signatures.add(run_signature)
            signatures.add("inference:" + inference_signature)
            if "alias_probability" in values.files:
                data["alias_probability"] = np.asarray(
                    values["alias_probability"], dtype=np.float32
                )
            parts.append(pd.DataFrame(data))
        hashes[str(fold_path.resolve())] = fold_hash
        marker_hashes[str(marker_path.resolve())] = shared.sha256_file(marker_path)
    if not parts:
        raise RuntimeError("all-window prediction directory contains no fold artifacts")
    return pd.concat(parts, ignore_index=True), {
        "format": "per_outer_fold_npz",
        "files": hashes,
        "verified_commit_markers": marker_hashes,
        "run_signatures": sorted(signatures),
        "semantic_binding_source": "verified fold marker and exact index SHA-256",
    }


def _prediction_columns(frame: pd.DataFrame) -> tuple[str, str, str | None]:
    mean_candidates = (
        "prediction_bpm",
        "prediction_final_bpm",
        "decoded_prediction",
        "prediction_locked_final_bpm",
    )
    std_candidates = ("rr_std_bpm", "final_uncertainty", "uncertainty_score")
    mean = next((name for name in mean_candidates if name in frame), None)
    std = next((name for name in std_candidates if name in frame), None)
    alias = "alias_probability" if "alias_probability" in frame else None
    if mean is None or std is None:
        raise RuntimeError("frozen prediction table lacks a recognized mean/std column")
    return mean, std, alias


def _resolve_fold_assignments(
    base_oof_csv: Path,
    all_window_base: Path | None,
    explicit_path: Path | None,
) -> tuple[Path, dict[str, int]]:
    """Resolve an independently hashed identity-to-fold authority."""

    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    else:
        base_path = Path(base_oof_csv).resolve()
        candidates.append(base_path.parent / "fold_assignments.json")
        if all_window_base is not None:
            all_path = Path(all_window_base).resolve()
            candidates.append(all_path / "fold_assignments.json")
            candidates.append(all_path.parent / "fold_assignments.json")
            candidates.append(all_path.parent.parent / "fold_assignments.json")
    resolved = next((path for path in candidates if path.is_file()), None)
    if resolved is None:
        raise RuntimeError(
            "an independent fold_assignments.json is required to bind base folds"
        )
    document = json.loads(resolved.read_text(encoding="utf-8"))
    raw_mapping = document.get("identity_to_fold", document)
    if not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise RuntimeError("fold assignment document has no identity_to_fold mapping")
    mapping = {str(identity): int(fold) for identity, fold in raw_mapping.items()}
    if any(fold < 0 or fold >= N_FOLDS for fold in mapping.values()):
        raise RuntimeError("fold assignment document contains an invalid fold")
    return resolved.resolve(), mapping


def _assert_semantic_row_binding(
    source: pd.DataFrame,
    bound_cache: pd.DataFrame,
    *,
    label: str,
    target_columns: Sequence[str],
    require_reference_valid: bool = False,
    require_classical_rr: bool = False,
) -> None:
    """Reject correct index sets whose row meaning was permuted or relabelled."""

    required = {
        "session_id",
        "identity",
        "protocol",
        "fold",
        "window_number",
        "window_start_s",
        "window_end_s",
    }
    if require_reference_valid:
        required.add("reference_valid")
    if require_classical_rr:
        required.add("classical_rr_bpm")
    missing = required - set(source.columns)
    if missing:
        raise RuntimeError(f"{label} lacks semantic binding columns: {sorted(missing)}")
    for name in ("session_id", "identity", "protocol"):
        if not np.array_equal(
            source[name].astype(str).to_numpy(),
            bound_cache[name].astype(str).to_numpy(),
        ):
            raise RuntimeError(f"{label} {name} semantic row binding mismatch")
    numeric = (
        ("fold", "fold", 0.0),
        ("window_number", "window_number", 0.0),
        ("window_start_s", "window_start_s", 1.0e-5),
        ("window_end_s", "window_end_s", 1.0e-5),
    )
    if require_classical_rr:
        numeric += (("classical_rr_bpm", "classical_rr_bpm", 1.0e-4),)
    for source_name, cache_name, tolerance in numeric:
        left = pd.to_numeric(source[source_name], errors="raise").to_numpy(float)
        right = pd.to_numeric(bound_cache[cache_name], errors="raise").to_numpy(float)
        if not np.allclose(left, right, rtol=0.0, atol=tolerance, equal_nan=False):
            raise RuntimeError(f"{label} {source_name} semantic row binding mismatch")
    if require_reference_valid:
        left_valid = source["reference_valid"].astype(bool).to_numpy()
        right_valid = bound_cache["reference_valid"].astype(bool).to_numpy()
        if not np.array_equal(left_valid, right_valid):
            raise RuntimeError(f"{label} reference_valid semantic row binding mismatch")
    target_name = next((name for name in target_columns if name in source), None)
    if target_name is None:
        raise RuntimeError(
            f"{label} lacks one of the target binding columns: {list(target_columns)}"
        )
    target_left = pd.to_numeric(source[target_name], errors="coerce").to_numpy(float)
    target_right = pd.to_numeric(bound_cache["rr_bpm"], errors="coerce").to_numpy(float)
    selected = (
        bound_cache["reference_valid"].astype(bool).to_numpy()
        if require_reference_valid
        else np.ones(len(bound_cache), dtype=bool)
    )
    if not np.allclose(
        target_left[selected],
        target_right[selected],
        rtol=0.0,
        atol=1.0e-4,
        equal_nan=False,
    ):
        raise RuntimeError(f"{label} target semantic row binding mismatch")


def load_episode_experiment(
    cache_root: Path,
    base_oof_csv: Path,
    alias_oof_csv: Path | None = None,
    all_window_base: Path | None = None,
    *,
    fold_assignments_json: Path | None = None,
    verify_file_hashes: bool = True,
) -> EpisodeExperiment:
    """Load all windows and bind frozen deployment predictions by cache index."""

    cache_root = Path(cache_root).resolve()
    root_manifest_path = cache_root / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    if bool(root_manifest.get("valid_only", True)):
        raise RuntimeError("episode training requires the all-window SVD cache")
    if root_manifest.get("label_inputs", []):
        raise RuntimeError("SVD cache declares forbidden label-derived inputs")
    base = pd.read_csv(base_oof_csv)
    required_base = {"cache_index", "identity", "fold", "prediction_bpm", "rr_std_bpm"}
    missing = required_base - set(base.columns)
    if missing:
        raise RuntimeError(f"base OOF is missing columns: {sorted(missing)}")
    if base["cache_index"].duplicated().any():
        raise RuntimeError("base OOF cache_index is not unique")
    identity_folds = base.groupby(base["identity"].astype(str))["fold"].nunique()
    if (identity_folds != 1).any():
        raise RuntimeError("a base identity appears in more than one frozen fold")
    fold_by_identity = {
        str(identity): int(group["fold"].iloc[0])
        for identity, group in base.groupby(base["identity"].astype(str), sort=False)
    }
    fold_assignment_path, canonical_fold_by_identity = _resolve_fold_assignments(
        base_oof_csv, all_window_base, fold_assignments_json
    )
    if fold_by_identity != canonical_fold_by_identity:
        raise RuntimeError(
            "base OOF identity folds do not match the independent frozen assignment"
        )
    frozen = base.loc[:, ["cache_index", "prediction_bpm", "rr_std_bpm"]].rename(
        columns={
            "prediction_bpm": "_base_prediction",
            "rr_std_bpm": "_base_std",
        }
    )
    frozen["_alias_probability"] = np.nan
    alias: pd.DataFrame | None = None
    if alias_oof_csv is not None and Path(alias_oof_csv).is_file():
        alias = pd.read_csv(alias_oof_csv)
        if "alias_probability" not in alias or alias["cache_index"].duplicated().any():
            raise RuntimeError("alias OOF lacks a unique alias_probability binding")
        alias_map = alias.set_index("cache_index")["alias_probability"]
        frozen["_alias_probability"] = frozen["cache_index"].map(alias_map)

    resolved_all = _resolve_all_window_prediction(all_window_base)
    all_window_status = "not_available_valid_oof_only"
    all_window_record: dict[str, Any] | None = None
    deployment_binding: pd.DataFrame | None = None
    if resolved_all is not None:
        deployment, all_window_record = _read_all_window_prediction(resolved_all)
        deployment_binding = deployment.copy()
        if "cache_index" not in deployment or deployment["cache_index"].duplicated().any():
            raise RuntimeError("all-window prediction table requires unique cache_index")
        mean_column, std_column, alias_column = _prediction_columns(deployment)
        columns = ["cache_index", mean_column, std_column]
        if alias_column:
            columns.append(alias_column)
        deployment_frozen = deployment.loc[:, columns].rename(
            columns={
                mean_column: "_base_prediction",
                std_column: "_base_std",
                alias_column: "_alias_probability" if alias_column else alias_column,
            }
        )
        if "_alias_probability" not in deployment_frozen:
            deployment_frozen["_alias_probability"] = np.nan
        # The commercial accuracy baseline is the frozen ensemble OOF on every
        # valid-reference row.  The all-window artifact comes from a different
        # single SNN and may only fill rows for which that ensemble has no OOF
        # prediction.  Letting it overwrite valid rows would silently change
        # both the safety fallback and the promotion comparator.
        fallback = frozen.set_index("cache_index")
        deployment_indexed = deployment_frozen.set_index("cache_index")
        frozen = fallback.combine_first(deployment_indexed).reset_index()
        all_window_status = (
            "loaded_complete_deployment_time_predictions"
            if len(deployment_indexed) == int(root_manifest.get("row_count", -1))
            else f"loaded_partial_deployment_time_predictions_{len(deployment_indexed)}_rows"
        )

    sessions: list[EpisodeSession] = []
    metadata_parts: list[pd.DataFrame] = []
    cache_indices: set[int] = set()
    source_hashes: dict[str, dict[str, str]] = {}
    global_offset = 0
    declared_sessions = [
        str(item["session_id"])
        for item in root_manifest.get("sessions", [])
        if item.get("status", "ok") == "ok"
    ]
    if not declared_sessions:
        raise RuntimeError("all-window cache manifest has no successful sessions")
    for session_slot, session_id in enumerate(declared_sessions):
        session_dir = cache_root / session_id
        session_manifest_path = session_dir / "manifest.json"
        manifest = json.loads(session_manifest_path.read_text(encoding="utf-8"))
        if bool(manifest.get("valid_only", True)) or manifest.get("label_inputs", []):
            raise RuntimeError(f"session {session_id} is not label-free all-window data")
        metadata_path = session_dir / "metadata.csv"
        metadata = pd.read_csv(metadata_path)
        spectra_path = session_dir / "spectra.npy"
        attributes_path = session_dir / "attributes.npy"
        frequencies_path = session_dir / "frequencies_hz.npy"
        spectra = np.load(spectra_path, mmap_mode="r", allow_pickle=False)
        attributes = np.load(attributes_path, mmap_mode="r", allow_pickle=False)
        frequencies = np.load(frequencies_path, allow_pickle=False)
        if len(metadata) != spectra.shape[0] or attributes.shape[:-1] != spectra.shape[:-1]:
            raise RuntimeError(f"row/array mismatch in {session_id}")
        if metadata["session_id"].astype(str).nunique() != 1 or str(
            metadata["session_id"].iloc[0]
        ) != session_id:
            raise RuntimeError(f"session binding mismatch in {session_id}")
        identities = metadata["identity"].astype(str).unique()
        if len(identities) != 1 or identities[0] not in fold_by_identity:
            raise RuntimeError(f"session {session_id} is not one frozen-fold identity")
        identity = str(identities[0])
        fold = fold_by_identity[identity]
        starts = pd.to_numeric(metadata["window_start_s"], errors="raise").to_numpy(float)
        numbers = pd.to_numeric(metadata["window_number"], errors="raise").to_numpy(int)
        if np.any(np.diff(starts) <= 0) or np.any(np.diff(numbers) != 1):
            raise RuntimeError(f"session {session_id} is not strictly chronological")
        if len(starts) > 1 and not np.allclose(np.diff(starts), 4.0, atol=1.0e-3):
            raise RuntimeError(f"session {session_id} does not use the required 4-second stride")
        indices = pd.to_numeric(metadata["cache_index"], errors="raise").to_numpy(np.int64)
        if cache_indices.intersection(indices.tolist()):
            raise RuntimeError("cache_index repeats across sessions")
        cache_indices.update(indices.tolist())
        classical = pd.to_numeric(metadata["classical_rr_bpm"], errors="coerce").to_numpy(
            np.float32
        )
        radar_peaks = metadata[
            ["radar_peak_1_bpm", "radar_peak_2_bpm", "radar_peak_3_bpm"]
        ].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        evidence, radar_mask = compute_candidate_evidence(
            spectra, attributes, frequencies, classical, radar_peaks
        )
        context = metadata[list(CONTEXT_NAMES)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(np.float32)
        local = metadata.merge(frozen, on="cache_index", how="left", validate="one_to_one")
        local["fold"] = fold
        local["_session_slot"] = session_slot
        local["_local_row"] = np.arange(len(local), dtype=np.int64)
        local["_position"] = np.arange(global_offset, global_offset + len(local), dtype=np.int64)
        global_offset += len(local)
        hashes = {
            # Metadata contains deployment inputs (classical evidence, context,
            # peaks and chronology), so it is always bound even when the caller
            # opts out of hashing the much larger tensor files.
            "metadata": shared.sha256_file(metadata_path),
            "spectra": shared.sha256_file(spectra_path) if verify_file_hashes else "not_computed",
            "attributes": shared.sha256_file(attributes_path)
            if verify_file_hashes
            else "not_computed",
            "frequencies": shared.sha256_file(frequencies_path)
            if verify_file_hashes
            else "not_computed",
            "manifest": shared.sha256_file(session_manifest_path),
        }
        source_hashes[session_id] = hashes
        sessions.append(
            EpisodeSession(
                session_id=session_id,
                identity=identity,
                fold=fold,
                metadata=local,
                evidence=evidence,
                context=context,
                radar_mask=radar_mask,
                manifest=manifest,
                source_hashes=hashes,
            )
        )
        metadata_parts.append(local)
    combined = pd.concat(metadata_parts, ignore_index=True)
    if len(combined) != int(root_manifest.get("row_count", -1)):
        raise RuntimeError("all-window root manifest row count mismatch")
    valid = combined["reference_valid"].astype(bool).to_numpy()
    base_valid_indices = set(pd.to_numeric(base["cache_index"], errors="raise").astype(int))
    if set(combined.loc[valid, "cache_index"].astype(int)) != base_valid_indices:
        raise RuntimeError("valid-reference rows do not exactly bind the frozen base OOF")
    cache_by_index = combined.set_index("cache_index")
    base_by_index = base.set_index("cache_index").loc[
        cache_by_index.index[valid]
    ]
    bound_valid = cache_by_index.loc[base_by_index.index]
    _assert_semantic_row_binding(
        base_by_index,
        bound_valid,
        label="base OOF",
        target_columns=("rr_bpm", "target_rr_bpm"),
        require_classical_rr=True,
    )
    alias_semantic_exact = False
    if alias is not None:
        alias_indices = set(pd.to_numeric(alias["cache_index"], errors="raise").astype(int))
        if alias_indices != base_valid_indices:
            raise RuntimeError("alias OOF index set does not exactly match frozen base OOF")
        alias_by_index = alias.set_index("cache_index").loc[
            base_by_index.index
        ]
        _assert_semantic_row_binding(
            alias_by_index,
            bound_valid,
            label="alias OOF",
            target_columns=("rr_bpm", "target_rr_bpm"),
            require_classical_rr=True,
        )
        alias_semantic_exact = True
    all_window_supplied_rows_exact = False
    all_window_complete_exact = False
    if deployment_binding is not None:
        deployment_indices = pd.to_numeric(
            deployment_binding["cache_index"], errors="raise"
        ).to_numpy(np.int64)
        if not set(deployment_indices).issubset(set(cache_by_index.index.astype(int))):
            raise RuntimeError("all-window prediction contains unknown cache indices")
        deployment_bound = cache_by_index.loc[deployment_indices]
        if all_window_record is not None and all_window_record.get("format") == "csv":
            deployment_semantic = deployment_binding.set_index("cache_index").loc[
                deployment_indices
            ]
            _assert_semantic_row_binding(
                deployment_semantic,
                deployment_bound,
                label="all-window prediction",
                target_columns=("reference_rr_bpm", "rr_bpm", "target_rr_bpm"),
                require_reference_valid=True,
            )
        else:
            declared_fold = pd.to_numeric(
                deployment_binding["fold"], errors="raise"
            ).to_numpy(np.int64)
            actual_fold = deployment_bound["fold"].to_numpy(np.int64)
            if not np.array_equal(declared_fold, actual_fold):
                raise RuntimeError("all-window prediction fold binding mismatch")
        all_window_supplied_rows_exact = True
        all_window_complete_exact = bool(
            len(deployment_indices) == len(combined)
            and set(deployment_indices) == set(cache_by_index.index.astype(int))
        )
    for session in sessions:
        positions = session.metadata["_position"].to_numpy(np.int64)
        if combined.iloc[positions]["fold"].nunique() != 1:
            raise RuntimeError("a session crosses fold partitions")
    provenance = {
        "cache_root": str(cache_root),
        "root_manifest_sha256": shared.sha256_file(root_manifest_path),
        "root_pipeline_sha256": root_manifest.get("pipeline_sha256"),
        "row_count": int(len(combined)),
        "valid_reference_rows": int(valid.sum()),
        "invalid_reference_rows": int((~valid).sum()),
        "session_count": int(len(sessions)),
        "identity_count": int(combined["identity"].astype(str).nunique()),
        "source_hashes": source_hashes,
        "fold_assignments_json": str(fold_assignment_path),
        "fold_assignments_sha256": shared.sha256_file(fold_assignment_path),
        "fold_assignments_exact": True,
        "base_oof_csv": str(Path(base_oof_csv).resolve()),
        "base_oof_sha256": shared.sha256_file(Path(base_oof_csv)),
        "alias_oof_csv": str(Path(alias_oof_csv).resolve())
        if alias_oof_csv is not None and Path(alias_oof_csv).exists()
        else None,
        "alias_oof_sha256": shared.sha256_file(Path(alias_oof_csv))
        if alias_oof_csv is not None and Path(alias_oof_csv).exists()
        else None,
        "base_binding_exact": True,
        "base_semantic_row_binding_exact": True,
        "alias_binding_exact": alias_semantic_exact,
        "alias_semantic_row_binding_exact": alias_semantic_exact,
        "all_window_supplied_rows_binding_exact": all_window_supplied_rows_exact,
        "all_window_complete_exact": all_window_complete_exact,
        # Retained for schema compatibility, but now means a complete exact
        # cover rather than merely that some supplied rows had fold labels.
        "all_window_index_fold_binding_exact": all_window_complete_exact,
        "all_window_prediction_csv": str(resolved_all.resolve()) if resolved_all else None,
        "all_window_prediction_status": all_window_status,
        "all_window_prediction_artifacts": all_window_record,
        "deployment_feature_contract": (
            "SVD spectra/attributes, classical RR/confidence/spread, radar peaks, "
            "and frozen base outputs only; reference target/validity/QC excluded from forward"
        ),
        "chronology": "every 4-second window updates state in session order",
    }
    return EpisodeExperiment(cache_root, combined, sessions, provenance)


def fit_episode_feature_scaler(
    experiment: EpisodeExperiment, train_positions: Sequence[int] | np.ndarray
) -> EpisodeFeatureScaler:
    positions = np.asarray(train_positions, dtype=np.int64)
    slots = np.unique(experiment.metadata.iloc[positions]["_session_slot"].to_numpy(int))
    evidence = np.concatenate(
        [experiment.sessions[int(slot)].evidence.reshape(-1, len(EVIDENCE_NAMES)) for slot in slots],
        axis=0,
    )
    context = np.concatenate([experiment.sessions[int(slot)].context for slot in slots], axis=0)

    def robust(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        center = np.median(values, axis=0)
        q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
        scale = (q75 - q25) / 1.349
        fallback = np.std(values, axis=0)
        scale = np.where(scale > 1.0e-5, scale, fallback)
        scale = np.where(scale > 1.0e-5, scale, 1.0)
        return center.astype(np.float32), scale.astype(np.float32)

    evidence_center, evidence_scale = robust(evidence)
    context_center, context_scale = robust(context)
    return EpisodeFeatureScaler(
        evidence_center.reshape(1, 1, 1, -1),
        evidence_scale.reshape(1, 1, 1, -1),
        context_center.reshape(1, -1),
        context_scale.reshape(1, -1),
        _positions_digest(positions),
    )


def fit_train_action_calibration(
    metadata: pd.DataFrame,
    train_positions: Sequence[int] | np.ndarray,
    *,
    divisor_class_balance_power: float = 0.0,
    strict_alias_gate: bool = False,
    strict_gate_margin_bpm: float = 0.75,
) -> TrainActionCalibration:
    if divisor_class_balance_power < 0:
        raise ValueError("divisor_class_balance_power cannot be negative")
    positions = np.asarray(train_positions, dtype=np.int64)
    frame = metadata.iloc[positions]
    valid = frame["reference_valid"].astype(bool).to_numpy()
    target = pd.to_numeric(frame["rr_bpm"], errors="coerce").to_numpy(float)
    classical = pd.to_numeric(frame["classical_rr_bpm"], errors="coerce").to_numpy(float)
    usable = valid & np.isfinite(target) & np.isfinite(classical) & (classical > 0)
    if not usable.any():
        raise RuntimeError("outer-training split has no valid action calibration rows")
    error = np.abs(classical[usable, None] * np.arange(1, 5)[None, :] - target[usable, None])
    oracle_action = error.argmin(axis=1)
    class_counts_array = np.bincount(oracle_action, minlength=4).astype(np.int64)
    if float(divisor_class_balance_power) == 0.0:
        class_weights_array = np.ones(4, dtype=np.float64)
    else:
        class_weights_array = np.zeros(4, dtype=np.float64)
        observed = class_counts_array > 0
        class_weights_array[observed] = class_counts_array[observed].astype(
            np.float64
        ) ** (-float(divisor_class_balance_power))
        frequency = class_counts_array.astype(np.float64) / class_counts_array.sum()
        normalizer = float(np.sum(frequency * class_weights_array))
        class_weights_array /= max(normalizer, 1.0e-12)
    sorted_error = np.sort(error, axis=1)
    action_gap = sorted_error[:, 1] - sorted_error[:, 0]
    divisor_temperature = float(np.clip(np.median(action_gap), 0.35, 2.0))
    if strict_alias_gate:
        gate_margin = float(strict_gate_margin_bpm)
        x1_error = error[:, 0]
        higher_error = error[:, 1:].min(axis=1)
        gate_temperature = float(
            np.clip(np.median(np.abs(x1_error - higher_error)), 0.25, 1.5)
        )
    else:
        base = pd.to_numeric(frame["_base_prediction"], errors="coerce").to_numpy(float)[usable]
        base_error = np.abs(base - target[usable])
        finite_base = np.isfinite(base_error)
        if finite_base.any():
            improvement = base_error[finite_base] - sorted_error[finite_base, 0]
            positive = improvement[improvement > 0]
            # Gate actions must be rare and materially useful, not merely better by
            # numerical noise.  This train-only threshold is intentionally at least
            # 0.5 bpm and is paired with an asymmetric false-positive cost.
            gate_margin = (
                float(np.clip(np.quantile(positive, 0.25), 0.5, 1.25))
                if len(positive)
                else 0.75
            )
            gate_temperature = float(
                np.clip(np.median(np.abs(improvement)), 0.25, 1.5)
            )
        else:
            gate_margin, gate_temperature = 0.75, 0.5
    return TrainActionCalibration(
        divisor_temperature,
        gate_margin,
        gate_temperature,
        _positions_digest(positions),
        int(usable.sum()),
        float(divisor_class_balance_power),
        tuple(int(value) for value in class_counts_array),
        tuple(float(value) for value in class_weights_array),
    )


def make_episode_split(
    experiment: EpisodeExperiment, outer_fold: int
) -> shared.FoldSplit:
    split = shared.make_outer_split(experiment.metadata, outer_fold)
    partition_by_position = np.full(len(experiment.metadata), -1, dtype=np.int8)
    partition_by_position[split.train] = 0
    partition_by_position[split.validation] = 1
    partition_by_position[split.test] = 2
    for session in experiment.sessions:
        positions = session.metadata["_position"].to_numpy(np.int64)
        values = np.unique(partition_by_position[positions])
        if len(values) != 1 or values[0] < 0:
            raise RuntimeError(f"session {session.session_id} crosses an outer partition")
        if session.metadata["identity"].astype(str).nunique() != 1:
            raise RuntimeError(f"session {session.session_id} crosses identities")
    return split


class EpisodeDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        experiment: EpisodeExperiment,
        session_slots: Sequence[int],
        scaler: EpisodeFeatureScaler,
    ) -> None:
        self.experiment = experiment
        self.session_slots = tuple(int(value) for value in session_slots)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.session_slots)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        session = self.experiment.sessions[self.session_slots[int(index)]]
        frame = session.metadata
        base = pd.to_numeric(frame["_base_prediction"], errors="coerce").to_numpy(np.float32)
        base_std = pd.to_numeric(frame["_base_std"], errors="coerce").to_numpy(np.float32)
        alias = pd.to_numeric(frame["_alias_probability"], errors="coerce").to_numpy(np.float32)
        base_available = np.isfinite(base) & np.isfinite(base_std) & (base_std > 0)
        return {
            "evidence": torch.from_numpy(self.scaler.transform_evidence(session.evidence)),
            "context": torch.from_numpy(self.scaler.transform_context(session.context)),
            "classical_rr": torch.from_numpy(
                pd.to_numeric(frame["classical_rr_bpm"], errors="coerce")
                .fillna(0.0)
                .to_numpy(np.float32)
            ),
            "base_prediction": torch.from_numpy(np.nan_to_num(base, nan=0.0)),
            "base_std": torch.from_numpy(np.nan_to_num(base_std, nan=4.0)),
            "base_alias_probability": torch.from_numpy(np.nan_to_num(alias, nan=0.0)),
            "base_available": torch.from_numpy(base_available),
            "radar_mask": torch.from_numpy(session.radar_mask),
            "rr": torch.from_numpy(
                pd.to_numeric(frame["rr_bpm"], errors="coerce").fillna(0.0).to_numpy(np.float32)
            ),
            "reference_valid": torch.from_numpy(
                frame["reference_valid"].astype(bool).to_numpy(copy=True)
            ),
            "reference_quality": torch.from_numpy(
                pd.to_numeric(frame["reference_quality"], errors="coerce")
                .fillna(0.0)
                .to_numpy(np.float32)
            ),
            "reference_sigma": torch.from_numpy(
                pd.to_numeric(frame["reference_sigma_bpm"], errors="coerce")
                .fillna(2.0)
                .to_numpy(np.float32)
            ),
            "position": torch.from_numpy(
                frame["_position"].to_numpy(dtype=np.int64, copy=True)
            ),
            "cache_index": torch.from_numpy(
                frame["cache_index"].to_numpy(dtype=np.int64, copy=True)
            ),
            "length": torch.tensor(len(frame), dtype=torch.int64),
            "session_slot": torch.tensor(self.session_slots[int(index)], dtype=torch.int64),
        }


def collate_episode_batch(items: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    if not items:
        raise ValueError("cannot collate an empty episode batch")
    maximum = max(int(item["length"]) for item in items)
    batch = len(items)
    result: dict[str, Tensor] = {}
    time_keys = (
        "evidence",
        "context",
        "classical_rr",
        "base_prediction",
        "base_std",
        "base_alias_probability",
        "base_available",
        "radar_mask",
        "rr",
        "reference_valid",
        "reference_quality",
        "reference_sigma",
        "position",
        "cache_index",
    )
    for key in time_keys:
        sample = items[0][key]
        fill = -1 if key in {"position", "cache_index"} else 0
        output = torch.full((batch, maximum, *sample.shape[1:]), fill, dtype=sample.dtype)
        for row, item in enumerate(items):
            length = int(item["length"])
            output[row, :length] = item[key]
        result[key] = output
    lengths = torch.stack([item["length"] for item in items])
    result["length"] = lengths
    result["session_slot"] = torch.stack([item["session_slot"] for item in items])
    result["sequence_mask"] = torch.arange(maximum).unsqueeze(0) < lengths.unsqueeze(1)
    return result


def _partition_session_slots(
    experiment: EpisodeExperiment, positions: Sequence[int] | np.ndarray
) -> list[int]:
    slots = experiment.metadata.iloc[np.asarray(positions, dtype=np.int64)][
        "_session_slot"
    ].to_numpy(int)
    selected = sorted(set(slots.tolist()))
    expected_positions = np.concatenate(
        [experiment.sessions[slot].metadata["_position"].to_numpy(np.int64) for slot in selected]
    )
    if not np.array_equal(np.sort(expected_positions), np.sort(np.asarray(positions, dtype=np.int64))):
        raise RuntimeError("partition does not contain each selected session in full")
    return selected


def make_episode_loader(
    experiment: EpisodeExperiment,
    positions: Sequence[int] | np.ndarray,
    scaler: EpisodeFeatureScaler,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Tensor]]:
    slots = _partition_session_slots(experiment, positions)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        EpisodeDataset(experiment, slots, scaler),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        collate_fn=collate_episode_batch,
        generator=generator,
        persistent_workers=bool(workers),
    )


def apply_episode_radar_dropout(
    evidence: Tensor,
    radar_mask: Tensor,
    *,
    probability: float,
    training: bool,
) -> tuple[Tensor, Tensor]:
    if not training or probability <= 0:
        return evidence, radar_mask
    available = radar_mask.bool()
    keep = torch.rand_like(available.float()) >= float(probability)
    keep &= available
    # Preserve one observed radar for each real window that originally had one.
    needs_one = available.any(dim=-1) & ~keep.any(dim=-1)
    if needs_one.any():
        first = available.to(torch.int64).argmax(dim=-1)
        rescue = F.one_hot(first, available.shape[-1]).bool() & needs_one.unsqueeze(-1)
        keep |= rescue
    return evidence * keep[..., None, None].to(evidence.dtype), keep


def forward_episode_model(
    model: EpisodeAliasRRSNN,
    batch: Mapping[str, Tensor],
    device: torch.device,
    *,
    radar_dropout: float = 0.0,
    training: bool = False,
) -> dict[str, Tensor]:
    evidence = batch["evidence"].to(device, non_blocking=True)
    radar_mask = batch["radar_mask"].to(device, non_blocking=True)
    evidence, radar_mask = apply_episode_radar_dropout(
        evidence, radar_mask, probability=radar_dropout, training=training
    )
    # Intentionally no reference_valid, reference_quality, target, or QC field
    # is passed through this deployment forward boundary.
    return model(
        evidence,
        batch["context"].to(device, non_blocking=True),
        batch["classical_rr"].to(device, non_blocking=True),
        batch["base_prediction"].to(device, non_blocking=True),
        batch["base_std"].to(device, non_blocking=True),
        batch["base_alias_probability"].to(device, non_blocking=True),
        batch["base_available"].to(device, non_blocking=True),
        radar_mask,
        batch["sequence_mask"].to(device, non_blocking=True),
    )


def compute_episode_multitask_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    rr_bins: Tensor,
    *,
    action_calibration: TrainActionCalibration,
    weak_invalid_weight: float = 0.0,
    weak_invalid_min_quality: float = 0.0,
    allow_stacked_base_training: bool = False,
    strict_alias_gate: bool = False,
    strict_gate_training_enabled: bool = True,
    strict_gate_margin_bpm: float = 0.75,
    strict_gate_false_positive_cost: float = 10.0,
    posterior_weight: float = 1.0,
    mae_weight: float = 0.5,
    divisor_weight: float = 0.35,
    action_regret_weight: float = 0.25,
    residual_weight: float = 0.15,
    uncertainty_weight: float = 0.08,
    gate_weight: float = 0.15,
    safe_gate_weight: float = 0.30,
    quality_weight: float = 0.03,
    temporal_weight: float = 0.02,
    spike_weight: float = 5.0e-4,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Source supervision plus mutually exclusive stacked or base-free gates."""

    if allow_stacked_base_training and strict_alias_gate:
        raise ValueError(
            "allow_stacked_base_training and strict_alias_gate are mutually exclusive"
        )
    if not 0.0 <= weak_invalid_min_quality <= 1.0:
        raise ValueError("weak_invalid_min_quality must be in [0, 1]")
    if strict_gate_margin_bpm < 0 or strict_gate_false_positive_cost < 1:
        raise ValueError("strict gate margin must be nonnegative and FP cost >= 1")

    source_probability = output["source_probabilities"].float()
    source_prediction = output["source_prediction"].float()
    target = batch["rr"].to(source_probability.device).float()
    sequence = batch["sequence_mask"].to(source_probability.device).bool()
    reference_valid = batch["reference_valid"].to(source_probability.device).bool()
    quality = batch["reference_quality"].to(source_probability.device).float()
    finite = torch.isfinite(target)
    strict_mask = sequence & reference_valid & finite
    weak_quality_eligible = quality >= float(weak_invalid_min_quality)
    weak_mask = (
        sequence
        & ~reference_valid
        & finite
        & weak_quality_eligible
        & (float(weak_invalid_weight) > 0)
    )
    loss_mask = strict_mask | weak_mask
    weight = torch.zeros_like(target)
    weight[strict_mask] = quality[strict_mask].clamp(0.25, 1.0)
    weight[weak_mask] = float(weak_invalid_weight)
    denominator = weight.sum().clamp_min(1.0e-8)
    zero = source_probability.sum() * 0.0
    if loss_mask.any():
        sigma = batch["reference_sigma"].to(source_probability.device).float()[loss_mask]
        sigma = torch.where(
            reference_valid[loss_mask], sigma.clamp(0.3, 2.5), torch.full_like(sigma, 2.0)
        )
        soft_target = gaussian_soft_targets(
            target[loss_mask], rr_bins.to(source_probability.device).float(), sigma=sigma
        )
        posterior_per = -(
            soft_target * source_probability[loss_mask].clamp_min(1.0e-8).log()
        ).sum(dim=-1)
        posterior_nll = (posterior_per * weight[loss_mask]).sum() / denominator
        mae = (
            F.smooth_l1_loss(
                source_prediction[loss_mask], target[loss_mask], beta=1.0, reduction="none"
            )
            * weight[loss_mask]
        ).sum() / denominator

        centers = batch["classical_rr"].to(source_probability.device).float().unsqueeze(-1)
        centers = centers * torch.arange(1, 5, device=centers.device, dtype=centers.dtype)
        candidate_valid = output["candidate_valid"].bool()
        action_error = (centers - target.unsqueeze(-1)).abs().masked_fill(
            ~candidate_valid, 1.0e4
        )
        action_soft = (-action_error / action_calibration.divisor_temperature_bpm).softmax(
            dim=-1
        )
        action_soft = action_soft * candidate_valid.to(action_soft.dtype)
        action_soft = action_soft / action_soft.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        oracle_action = action_error.argmin(dim=-1)
        divisor_class_weights = torch.as_tensor(
            action_calibration.divisor_class_weights,
            device=source_probability.device,
            dtype=torch.float32,
        )
        action_row_weight = weight * divisor_class_weights[oracle_action]
        action_denominator = action_row_weight[loss_mask].sum().clamp_min(1.0e-8)
        divisor_ce_per = -(
            action_soft
            * output["divisor_probabilities"].float().clamp_min(1.0e-8).log()
        ).sum(dim=-1)
        divisor_ce = (
            divisor_ce_per[loss_mask] * action_row_weight[loss_mask]
        ).sum() / action_denominator
        expected_regret = (
            output["divisor_probabilities"].float() * action_error.clamp_max(20.0)
        ).sum(dim=-1)
        oracle_error = action_error.amin(dim=-1)
        action_regret = (
            (expected_regret - oracle_error).clamp_min(0)[loss_mask]
            * action_row_weight[loss_mask]
        ).sum() / action_denominator
        candidate_error = output["candidate_mean"].float() - target.unsqueeze(-1)
        residual_per = (
            action_soft
            * F.smooth_l1_loss(
                candidate_error, torch.zeros_like(candidate_error), beta=0.75, reduction="none"
            )
        ).sum(dim=-1)
        residual = (residual_per[loss_mask] * weight[loss_mask]).sum() / denominator
        candidate_std = output["candidate_std"].float().clamp_min(0.2)
        uncertainty_per = (
            action_soft
            * (candidate_error.square() / (2.0 * candidate_std.square()) + candidate_std.log())
        ).sum(dim=-1)
        uncertainty_nll = (
            uncertainty_per[loss_mask] * weight[loss_mask]
        ).sum() / denominator
    else:
        posterior_nll = mae = divisor_ce = action_regret = residual = uncertainty_nll = zero

    radar_present = batch["radar_mask"].to(source_probability.device).any(dim=-1)
    gate_positive_fraction = zero
    if strict_alias_gate and strict_gate_training_enabled:
        candidate_valid = output["candidate_valid"].bool()
        gate_rows = strict_mask & radar_present & candidate_valid[..., 1:].any(dim=-1)
        if gate_rows.any():
            classical = batch["classical_rr"].to(source_probability.device).float()
            x1_error = (classical - target).abs()
            higher_centers = classical.unsqueeze(-1) * torch.arange(
                2, 5, device=classical.device, dtype=classical.dtype
            )
            higher_error = (higher_centers - target.unsqueeze(-1)).abs().masked_fill(
                ~candidate_valid[..., 1:], 1.0e4
            )
            higher_oracle_error = higher_error.amin(dim=-1)
            alias_needed = (
                x1_error - higher_oracle_error >= float(strict_gate_margin_bpm)
            ).float()
            gate_logits = output["gate_logits"].float()
            gate_bce_per = F.binary_cross_entropy_with_logits(
                gate_logits[gate_rows], alias_needed[gate_rows], reduction="none"
            )
            false_positive_weight = torch.where(
                alias_needed[gate_rows] > 0,
                torch.ones_like(gate_bce_per),
                torch.full_like(
                    gate_bce_per, float(strict_gate_false_positive_cost)
                ),
            )
            gate_reference_weight = weight[gate_rows].clamp_min(0.25)
            gate_bce = (
                gate_bce_per * false_positive_weight * gate_reference_weight
            ).sum() / (false_positive_weight * gate_reference_weight).sum().clamp_min(
                1.0e-8
            )
            source_error = (source_prediction - target).abs()
            harm = (source_error - x1_error).clamp_min(0)
            safe_gate = (
                output["learned_gate"].float()[gate_rows]
                * harm[gate_rows]
                * gate_reference_weight
            ).sum() / gate_reference_weight.sum().clamp_min(1.0e-8)
            gate_positive_fraction = alias_needed[gate_rows].mean()
        else:
            gate_bce = safe_gate = zero
    elif strict_alias_gate:
        # Source warmup: the gate head and its shared encoder receive no gate
        # gradient at all until the configured warmup has completed.
        gate_bce = safe_gate = zero
    elif allow_stacked_base_training:
        base_available = batch["base_available"].to(source_probability.device).bool()
        gate_rows = strict_mask & base_available & radar_present
        if not gate_rows.any():
            gate_bce = safe_gate = zero
        else:
            base = batch["base_prediction"].to(source_probability.device).float()
            source_error = (source_prediction - target).abs()
            base_error = (base - target).abs()
            oracle_source = (
                source_error + float(action_calibration.gate_margin_bpm) < base_error
            ).float()
            gate_logits = output["gate_logits"].float()
            gate_bce_per = F.binary_cross_entropy_with_logits(
                gate_logits[gate_rows], oracle_source[gate_rows], reduction="none"
            )
            false_positive_weight = torch.where(
                oracle_source[gate_rows] > 0,
                torch.ones_like(gate_bce_per),
                torch.full_like(gate_bce_per, 5.0),
            )
            gate_bce = (gate_bce_per * false_positive_weight).mean()
            harm = (source_error - base_error).clamp_min(0)
            safe_gate = (
                output["learned_gate"].float()[gate_rows] * harm[gate_rows]
            ).mean()
            gate_positive_fraction = oracle_source[gate_rows].mean()
    else:
        gate_bce = zero
        # In the strict primary protocol the learned gate is held closed.  The
        # source candidate is selected only by the fold-level validation lock.
        safe_gate = output["learned_gate"].float()[sequence].mean() if sequence.any() else zero

    if strict_mask.any():
        quality_bce = F.binary_cross_entropy_with_logits(
            output["quality_logit"].float()[strict_mask],
            quality[strict_mask].clamp(0, 1),
        )
    else:
        quality_bce = zero
    adjacent = sequence[:, 1:] & sequence[:, :-1]
    if adjacent.any():
        temporal_delta = (
            output["divisor_probabilities"][:, 1:].float()
            - output["divisor_probabilities"][:, :-1].float()
        ).abs().mean(dim=-1)
        temporal_consistency = temporal_delta[adjacent].mean()
    else:
        temporal_consistency = zero
    spike_rate = output["spike_rate"].float().mean()
    total = (
        float(posterior_weight) * posterior_nll
        + float(mae_weight) * mae
        + float(divisor_weight) * divisor_ce
        + float(action_regret_weight) * action_regret
        + float(residual_weight) * residual
        + float(uncertainty_weight) * uncertainty_nll
        + float(gate_weight) * gate_bce
        + float(safe_gate_weight) * safe_gate
        + float(quality_weight) * quality_bce
        + float(temporal_weight) * temporal_consistency
        + float(spike_weight) * spike_rate
    )
    parts = {
        "loss": total.detach(),
        "posterior_nll": posterior_nll.detach(),
        "mae": mae.detach(),
        "divisor_ce": divisor_ce.detach(),
        "action_regret": action_regret.detach(),
        "residual": residual.detach(),
        "uncertainty_nll": uncertainty_nll.detach(),
        "gate_bce": gate_bce.detach(),
        "safe_gate": safe_gate.detach(),
        "gate_positive_fraction": gate_positive_fraction.detach(),
        "quality_bce": quality_bce.detach(),
        "temporal_consistency": temporal_consistency.detach(),
        "spike_rate": spike_rate.detach(),
        "strict_valid_rows": strict_mask.sum().detach(),
        "weak_invalid_rows": weak_mask.sum().detach(),
        "weak_invalid_quality_eligible_rows": (
            sequence & ~reference_valid & finite & weak_quality_eligible
        ).sum().detach(),
    }
    return total, parts


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        enabled=bool(enabled and device.type == "cuda"),
        dtype=torch.float16,
    )


def train_one_epoch(
    model: EpisodeAliasRRSNN,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    action_calibration: TrainActionCalibration,
    weak_invalid_weight: float,
    weak_invalid_min_quality: float,
    allow_stacked_base_training: bool,
    strict_alias_gate: bool,
    strict_gate_training_enabled: bool,
    strict_gate_margin_bpm: float,
    strict_gate_false_positive_cost: float,
    radar_dropout: float,
    amp: bool,
    gradient_clip: float,
    max_batches: int | None = None,
    loss_kwargs: Mapping[str, float] | None = None,
) -> dict[str, float]:
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            output = forward_episode_model(
                model,
                batch,
                device,
                radar_dropout=radar_dropout,
                training=True,
            )
            loss, parts = compute_episode_multitask_loss(
                output,
                batch,
                model.rr_bins,
                action_calibration=action_calibration,
                weak_invalid_weight=weak_invalid_weight,
                weak_invalid_min_quality=weak_invalid_min_quality,
                allow_stacked_base_training=allow_stacked_base_training,
                strict_alias_gate=strict_alias_gate,
                strict_gate_training_enabled=strict_gate_training_enabled,
                strict_gate_margin_bpm=strict_gate_margin_bpm,
                strict_gate_false_positive_cost=strict_gate_false_positive_cost,
                **dict(loss_kwargs or {}),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        scaler.step(optimizer)
        scaler.update()
        for name, value in parts.items():
            totals[name] += float(value.detach().cpu())
        batches += 1
    if not batches:
        raise RuntimeError("training loader yielded no batches")
    return {name: value / batches for name, value in totals.items()}


@torch.inference_mode()
def predict_episode_loader(
    model: EpisodeAliasRRSNN,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None = None,
) -> EpisodePrediction:
    model.eval()
    values: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        with _autocast(device, amp):
            output = forward_episode_model(model, batch, device)
        selected = batch["sequence_mask"].bool() & batch["reference_valid"].bool()
        if not selected.any():
            continue
        device_selected = selected.to(device)
        divisor = output["divisor_probabilities"].float()
        residual = (divisor * output["residual_rr"].float()).sum(dim=-1)
        radar_weights = (
            output["radar_weights"].float()
            * divisor.unsqueeze(2)
        ).sum(dim=-1)
        session_spike = output["spike_rate"].float().unsqueeze(1).expand_as(
            output["source_prediction"]
        )
        mappings = {
            "position": batch["position"][selected],
            "cache_index": batch["cache_index"][selected],
            "target": batch["rr"][selected],
            "base_prediction": batch["base_prediction"][selected],
            "base_std": batch["base_std"][selected],
            "base_available": batch["base_available"][selected],
            # Strict candidate is the source-only estimate.  The frozen base
            # remains the validation-locked fallback and never trains primary weights.
            "candidate_prediction": (
                output["expected_rr"]
                if model.use_base_features or model.strict_alias_gate
                else output["source_prediction"]
            )[device_selected],
            "rr_std": (
                output["rr_std"]
                if model.use_base_features or model.strict_alias_gate
                else output["source_std"]
            )[device_selected],
            "source_prediction": output["source_prediction"][device_selected],
            "source_std": output["source_std"][device_selected],
            "mixture_gate": output["mixture_gate"][device_selected],
            "learned_gate": output["learned_gate"][device_selected],
            "applied_gate": output["mixture_gate"][device_selected],
            "divisor_probabilities": divisor[device_selected],
            "residual_rr": residual[device_selected],
            "candidate_std": output["candidate_std"].float()[device_selected],
            "quality": output["quality"].float()[device_selected],
            "radar_weights": radar_weights[device_selected],
            "spike_rate": session_spike[device_selected],
        }
        for name, tensor in mappings.items():
            values[name].append(tensor.detach().cpu().numpy())
    if not values:
        raise RuntimeError("prediction loader produced no valid-reference rows")
    arrays = {name: np.concatenate(parts) for name, parts in values.items()}
    result = EpisodePrediction(
        position=arrays["position"].astype(np.int64),
        cache_index=arrays["cache_index"].astype(np.int64),
        target=arrays["target"].astype(np.float32),
        base_prediction=arrays["base_prediction"].astype(np.float32),
        base_std=arrays["base_std"].astype(np.float32),
        base_available=arrays["base_available"].astype(bool),
        candidate_prediction=arrays["candidate_prediction"].astype(np.float32),
        rr_std=arrays["rr_std"].astype(np.float32),
        source_prediction=arrays["source_prediction"].astype(np.float32),
        source_std=arrays["source_std"].astype(np.float32),
        mixture_gate=arrays["mixture_gate"].astype(np.float32),
        learned_gate=arrays["learned_gate"].astype(np.float32),
        applied_gate=arrays["applied_gate"].astype(np.float32),
        divisor_probabilities=arrays["divisor_probabilities"].astype(np.float32),
        residual_rr=arrays["residual_rr"].astype(np.float32),
        candidate_std=arrays["candidate_std"].astype(np.float32),
        quality=arrays["quality"].astype(np.float32),
        radar_weights=arrays["radar_weights"].astype(np.float32),
        spike_rate=arrays["spike_rate"].astype(np.float32),
    )
    order = np.argsort(result.position, kind="stable")
    return EpisodePrediction(
        **{name: getattr(result, name)[order] for name in result.__dataclass_fields__}
    )


def apply_strict_gate_policy(
    result: EpisodePrediction, policy: StrictGatePolicy
) -> EpisodePrediction:
    """Apply a locked threshold/pull without consulting targets or metadata."""

    learned = np.asarray(result.learned_gate, dtype=np.float64)
    base = np.asarray(result.base_prediction, dtype=np.float64)
    base_std = np.asarray(result.base_std, dtype=np.float64)
    source = np.asarray(result.source_prediction, dtype=np.float64)
    source_std = np.asarray(result.source_std, dtype=np.float64)
    has_base = np.asarray(result.base_available, dtype=bool)
    if has_base.shape != base.shape:
        raise RuntimeError("base availability mask shape does not match predictions")
    if np.any(
        has_base & (~np.isfinite(base) | ~np.isfinite(base_std) | (base_std <= 0))
    ):
        raise RuntimeError("available base rows must contain finite prediction/std")
    has_radar = np.asarray(result.radar_weights).sum(axis=1) > 1.0e-8
    correction = (
        (learned >= float(policy.threshold)) & has_base & has_radar
    ).astype(np.float64) * float(policy.correction_pull)
    # Structural fallbacks are policy-independent.
    applied = np.where(has_base, correction, 1.0)
    applied = np.where(has_base & ~has_radar, 0.0, applied)
    prediction = (1.0 - applied) * np.where(has_base, base, source) + applied * source
    variance = (
        (1.0 - applied)
        * (
            np.where(has_base, base_std, source_std) ** 2
            + (np.where(has_base, base, source) - prediction) ** 2
        )
        + applied * (source_std**2 + (source - prediction) ** 2)
    )
    values = {
        name: np.asarray(getattr(result, name)).copy()
        for name in result.__dataclass_fields__
    }
    values["candidate_prediction"] = prediction.astype(np.float32)
    values["rr_std"] = np.sqrt(np.maximum(variance, 1.0e-8)).astype(np.float32)
    values["mixture_gate"] = applied.astype(np.float32)
    values["applied_gate"] = applied.astype(np.float32)
    return EpisodePrediction(**values)


def select_strict_gate_policy(
    result: EpisodePrediction,
    metadata: pd.DataFrame,
    *,
    maximum_coverage: float = 0.15,
    tail_min_bpm: float = 25.0,
    tail_max_bpm: float = 35.0,
) -> tuple[StrictGatePolicy, EpisodePrediction]:
    """Validation-only safety-grid selection for a cost-biased gate score."""

    if not 0.0 < maximum_coverage <= 1.0:
        raise ValueError("maximum_coverage must be in (0, 1]")
    base = np.asarray(result.base_prediction, dtype=np.float64)
    base_std = np.asarray(result.base_std, dtype=np.float64)
    source = np.asarray(result.source_prediction, dtype=np.float64)
    has_base = np.asarray(result.base_available, dtype=bool)
    if has_base.shape != base.shape:
        raise RuntimeError("base availability mask shape does not match predictions")
    if np.any(
        has_base & (~np.isfinite(base) | ~np.isfinite(base_std) | (base_std <= 0))
    ):
        raise RuntimeError("available base rows must contain finite prediction/std")
    identities = metadata.iloc[result.position]["identity"].astype(str).to_numpy()
    # Missing-base rows contain finite dataset placeholders (0 bpm / 4 bpm std).
    # They are structural source fallbacks, never eligible base corrections.
    base_reference = np.where(has_base, base, source)
    base_snapshot = shared.evaluation_snapshot(
        result.target,
        base_reference,
        identities,
        high_min_bpm=tail_min_bpm,
        high_max_bpm=tail_max_bpm,
    )
    scores = np.asarray(result.learned_gate, dtype=np.float64)
    finite_scores = scores[np.isfinite(scores)]
    fixed_thresholds = np.asarray(
        [
            0.0,
            1.0e-4,
            2.5e-4,
            5.0e-4,
            1.0e-3,
            2.5e-3,
            5.0e-3,
            1.0e-2,
            2.5e-2,
            5.0e-2,
            0.1,
            0.2,
            0.35,
            0.5,
            0.75,
            0.9,
            1.1,
        ],
        dtype=np.float64,
    )
    quantiles = (
        np.quantile(finite_scores, [0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99])
        if len(finite_scores)
        else np.empty(0, dtype=np.float64)
    )
    thresholds = np.unique(np.concatenate((fixed_thresholds, quantiles)))
    pulls = (0.25, 0.5, 0.75, 1.0)
    candidates: list[tuple[tuple[float, float, float], StrictGatePolicy, EpisodePrediction]] = []
    has_radar = np.asarray(result.radar_weights).sum(axis=1) > 1.0e-8
    eligible_count = max(int(np.sum(has_base & has_radar)), 1)
    for threshold in thresholds:
        for pull in pulls:
            provisional = StrictGatePolicy(
                float(threshold), float(pull), 0.0, math.inf, None, {}, 0
            )
            applied = apply_strict_gate_policy(result, provisional)
            coverage = float(
                np.sum((applied.applied_gate > 0) & has_base & has_radar)
                / eligible_count
            )
            snapshot = shared.evaluation_snapshot(
                applied.target,
                applied.candidate_prediction,
                identities,
                high_min_bpm=tail_min_bpm,
                high_max_bpm=tail_max_bpm,
            )
            high = snapshot["high_25_35"]
            base_high = base_snapshot["high_25_35"]
            gates = {
                "coverage_at_most_cap": coverage <= float(maximum_coverage) + 1.0e-12,
                "macro_mae_noninferior": float(
                    snapshot["identity_macro"]["macro_mae"]
                )
                <= float(base_snapshot["identity_macro"]["macro_mae"]),
                "catastrophic_noninferior": float(
                    snapshot["overall"]["catastrophic_over_5"]
                )
                <= float(base_snapshot["overall"]["catastrophic_over_5"]),
                "within_2_noninferior": float(snapshot["overall"]["within_2"])
                >= float(base_snapshot["overall"]["within_2"]),
                "tail_macro_mae_noninferior": bool(
                    high is not None
                    and base_high is not None
                    and float(high["identity_macro"]["macro_mae"])
                    <= float(base_high["identity_macro"]["macro_mae"])
                ),
            }
            policy = StrictGatePolicy(
                threshold=float(threshold),
                correction_pull=float(pull),
                validation_coverage=coverage,
                validation_macro_mae=float(snapshot["identity_macro"]["macro_mae"]),
                validation_tail_macro_mae=(
                    float(high["identity_macro"]["macro_mae"])
                    if high is not None
                    else None
                ),
                safety_gates=gates,
                candidate_count=int(len(thresholds) * len(pulls)),
            )
            if all(gates.values()):
                tail_score = (
                    float(high["identity_macro"]["macro_mae"])
                    if high is not None
                    else math.inf
                )
                candidates.append(
                    (
                        (
                            float(snapshot["identity_macro"]["macro_mae"]),
                            tail_score,
                            coverage,
                        ),
                        policy,
                        applied,
                    )
                )
    if candidates:
        _, policy, applied = min(candidates, key=lambda item: item[0])
        return policy, applied
    fallback = StrictGatePolicy(
        threshold=1.1,
        correction_pull=0.0,
        validation_coverage=0.0,
        validation_macro_mae=float(base_snapshot["identity_macro"]["macro_mae"]),
        validation_tail_macro_mae=(
            float(base_snapshot["high_25_35"]["identity_macro"]["macro_mae"])
            if base_snapshot["high_25_35"] is not None
            else None
        ),
        safety_gates={
            "coverage_at_most_cap": True,
            "macro_mae_noninferior": True,
            "catastrophic_noninferior": True,
            "within_2_noninferior": True,
            "tail_macro_mae_noninferior": base_snapshot["high_25_35"] is not None,
        },
        candidate_count=int(len(thresholds) * len(pulls)),
    )
    return fallback, apply_strict_gate_policy(result, fallback)


def _prediction_arrays(
    result: EpisodePrediction, *, promoted: bool | None = None
) -> dict[str, np.ndarray]:
    arrays = {
        name: np.asarray(getattr(result, name)) for name in result.__dataclass_fields__
    }
    if promoted is not None:
        arrays["prediction_final"] = np.where(
            promoted, result.candidate_prediction, result.base_prediction
        ).astype(np.float32)
        arrays["promoted"] = np.full(len(result.position), bool(promoted), dtype=bool)
    return arrays


def _prediction_report(
    result: EpisodePrediction,
    metadata: pd.DataFrame,
    *,
    promoted: bool | None = None,
    tail_min_bpm: float = 25.0,
    tail_max_bpm: float = 35.0,
    strict_gate_margin_bpm: float = 0.75,
) -> dict[str, Any]:
    identities = metadata.iloc[result.position]["identity"].astype(str).to_numpy()
    final = (
        np.where(promoted, result.candidate_prediction, result.base_prediction)
        if promoted is not None
        else result.candidate_prediction
    )
    frame = metadata.iloc[result.position]
    classical = pd.to_numeric(frame["classical_rr_bpm"], errors="coerce").to_numpy(float)
    action_error = np.abs(classical[:, None] * np.arange(1, 5)[None, :] - result.target[:, None])
    oracle_action = np.argmin(action_error, axis=1)
    selected_action = np.argmax(result.divisor_probabilities, axis=1)
    action_regret = (
        result.divisor_probabilities * action_error
    ).sum(axis=1) - action_error.min(axis=1)
    alias_needed = (
        action_error[:, 0] - action_error[:, 1:].min(axis=1)
        >= float(strict_gate_margin_bpm)
    )
    corrected = result.applied_gate > 0
    true_positive = int(np.sum(corrected & alias_needed))
    selected_count = int(np.sum(corrected))
    positive_count = int(np.sum(alias_needed))
    return {
        "rows": int(len(result.position)),
        "candidate": shared.evaluation_snapshot(
            result.target,
            result.candidate_prediction,
            identities,
            high_min_bpm=tail_min_bpm,
            high_max_bpm=tail_max_bpm,
        ),
        "base": shared.evaluation_snapshot(
            result.target,
            result.base_prediction,
            identities,
            high_min_bpm=tail_min_bpm,
            high_max_bpm=tail_max_bpm,
        ),
        "locked_final": shared.evaluation_snapshot(
            result.target,
            final,
            identities,
            high_min_bpm=tail_min_bpm,
            high_max_bpm=tail_max_bpm,
        )
        if promoted is not None
        else None,
        "mean_gate": float(np.mean(result.mixture_gate)),
        "mean_learned_gate": float(np.mean(result.learned_gate)),
        "mean_applied_gate": float(np.mean(result.applied_gate)),
        "applied_gate_coverage": float(np.mean(result.applied_gate > 0)),
        "strict_alias_gate_diagnostics": {
            "margin_bpm": float(strict_gate_margin_bpm),
            "alias_needed_fraction": float(np.mean(alias_needed)),
            "selected_rows": selected_count,
            "precision": float(true_positive / selected_count)
            if selected_count
            else None,
            "recall": float(true_positive / positive_count)
            if positive_count
            else None,
            "false_positive_fraction_among_selected": float(
                (selected_count - true_positive) / selected_count
            )
            if selected_count
            else None,
            "mean_learned_gate_alias_needed": float(
                np.mean(result.learned_gate[alias_needed])
            )
            if positive_count
            else None,
            "mean_learned_gate_direct": float(
                np.mean(result.learned_gate[~alias_needed])
            )
            if np.any(~alias_needed)
            else None,
        },
        "mean_spike_rate": float(np.mean(result.spike_rate)),
        "source_divisor_diagnostics": {
            "oracle_action_accuracy": float(np.mean(selected_action == oracle_action)),
            "mean_soft_action_regret_bpm": float(np.mean(action_regret)),
            "mean_divisor_probabilities": result.divisor_probabilities.mean(axis=0).tolist(),
            "mean_absolute_residual_bpm": float(np.mean(np.abs(result.residual_rr))),
        },
    }


def _source_binding(experiment: EpisodeExperiment) -> str:
    return shared.stable_signature(experiment.provenance, length=32)


def _scaler_signature(scaler: EpisodeFeatureScaler) -> str:
    return shared.stable_signature(scaler.record(), length=32)


def _action_signature(calibration: TrainActionCalibration) -> str:
    return shared.stable_signature(calibration.record(), length=32)


def _checkpoint(
    *,
    model: EpisodeAliasRRSNN,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler,
    epoch: int,
    best_epoch: int,
    best_score: float,
    stale_epochs: int,
    run_signature: str,
    source_binding: str,
    feature_scaler: EpisodeFeatureScaler,
    action_calibration: TrainActionCalibration,
    split: shared.FoldSplit,
    model_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "stale_epochs": int(stale_epochs),
        "run_signature": run_signature,
        "source_binding": source_binding,
        "feature_scaler_signature": _scaler_signature(feature_scaler),
        "action_calibration_signature": _action_signature(action_calibration),
        "split": {
            "outer_fold": split.outer_fold,
            "validation_fold": split.validation_fold,
            "train_identities": list(split.train_identities),
            "validation_identities": list(split.validation_identities),
            "test_identities": list(split.test_identities),
        },
        "model_kwargs": dict(model_kwargs),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "grad_scaler": grad_scaler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    run_signature: str,
    source_binding: str,
    feature_scaler: EpisodeFeatureScaler,
    action_calibration: TrainActionCalibration,
    split: shared.FoldSplit,
) -> None:
    expected = {
        "run_signature": run_signature,
        "source_binding": source_binding,
        "feature_scaler_signature": _scaler_signature(feature_scaler),
        "action_calibration_signature": _action_signature(action_calibration),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"resume checkpoint {key} does not match this run")
    stored = checkpoint.get("split", {})
    if stored.get("outer_fold") != split.outer_fold or stored.get(
        "validation_fold"
    ) != split.validation_fold:
        raise RuntimeError("resume checkpoint split does not match")
    for key, expected_identities in (
        ("train_identities", split.train_identities),
        ("validation_identities", split.validation_identities),
        ("test_identities", split.test_identities),
    ):
        if tuple(stored.get(key, ())) != tuple(expected_identities):
            raise RuntimeError(f"resume checkpoint {key} does not match")


def _validate_committed_best_checkpoint(
    last_checkpoint: Mapping[str, Any], best_path: Path
) -> None:
    """Fail closed if a last/best two-file commit was interrupted or altered."""

    best_epoch = int(last_checkpoint.get("best_epoch", -1))
    expected_hash = last_checkpoint.get("committed_best_checkpoint_sha256")
    if best_epoch < 0:
        if expected_hash is not None or best_path.exists():
            raise RuntimeError(
                "resume checkpoint has an uncommitted best-checkpoint artifact"
            )
        return
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise RuntimeError("resume checkpoint lacks a committed best-checkpoint hash")
    if not best_path.is_file():
        raise RuntimeError("resume committed best checkpoint is missing")
    if shared.sha256_file(best_path) != expected_hash:
        raise RuntimeError("resume committed best checkpoint SHA-256 mismatch")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if int(best.get("epoch", -1)) != best_epoch or int(
        best.get("best_epoch", -1)
    ) != best_epoch:
        raise RuntimeError("resume committed best checkpoint epoch mismatch")
    if not math.isclose(
        float(best.get("best_score", math.inf)),
        float(last_checkpoint.get("best_score", math.inf)),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("resume committed best checkpoint score mismatch")


def _load_prediction(path: Path) -> EpisodePrediction:
    with np.load(path, allow_pickle=False) as values:
        return EpisodePrediction(
            **{
                name: np.asarray(values[name])
                for name in EpisodePrediction.__dataclass_fields__
            }
        )


def _model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    preset = {
        "tiny": {"candidate_channels": 4, "hidden_channels": 8, "cell_types": ("lif",)},
        "compact": {
            "candidate_channels": args.candidate_channels,
            "hidden_channels": args.hidden_channels,
            "cell_types": tuple(value.strip() for value in args.cell_types.split(",")),
        },
        "full": {
            "candidate_channels": max(args.candidate_channels, 24),
            "hidden_channels": max(args.hidden_channels, 64),
            "cell_types": ("lif", "plif", "alif"),
        },
    }[args.preset]
    return {
        "evidence_features": len(EVIDENCE_NAMES),
        "context_features": len(CONTEXT_NAMES),
        **preset,
        "beta": args.beta,
        "dropout": args.dropout,
        "max_residual_bpm": args.max_residual_bpm,
        "initial_gate_bias": args.initial_gate_bias,
        "use_base_features": bool(args.allow_stacked_base_training),
        "strict_alias_gate": bool(args.strict_alias_gate),
    }


def _loss_kwargs(args: argparse.Namespace) -> dict[str, float]:
    return {
        "posterior_weight": args.posterior_weight,
        "mae_weight": args.mae_weight,
        "divisor_weight": args.divisor_weight,
        "action_regret_weight": args.action_regret_weight,
        "residual_weight": args.residual_weight,
        "uncertainty_weight": args.uncertainty_weight,
        "gate_weight": args.gate_weight,
        "safe_gate_weight": args.safe_gate_weight,
        "quality_weight": args.quality_weight,
        "temporal_weight": args.temporal_weight,
        "spike_weight": args.spike_weight,
    }


def _completed_fold_result(
    fold_dir: Path,
    *,
    run_signature: str,
    source_binding: str,
    split: shared.FoldSplit,
    experiment: EpisodeExperiment,
) -> tuple[EpisodePrediction, bool, dict[str, Any]] | None:
    manifest_path = fold_dir / "test_evaluation_manifest.json"
    prediction_path = fold_dir / "test_predictions.npz"
    selection_path = fold_dir / "selection_lock.json"
    report_path = fold_dir / "test_predictions.json"
    if not (
        manifest_path.is_file()
        and prediction_path.is_file()
        and selection_path.is_file()
        and report_path.is_file()
    ):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        manifest.get("test_fold_evaluation_invocations") != 1
        or not bool(manifest.get("complete_outer_test_evaluation"))
    ):
        raise RuntimeError("saved fold violates the one-test-evaluation contract")
    if selection.get("run_signature") != run_signature:
        raise RuntimeError("completed fold run_signature does not match this run")
    if selection.get("source_binding") != source_binding:
        raise RuntimeError("completed fold source binding does not match current inputs")
    if int(selection.get("outer_fold", -1)) != split.outer_fold:
        raise RuntimeError("completed fold outer-fold binding mismatch")
    best_path = fold_dir / "episode_best.pt"
    expected_checkpoint_sha = selection.get("checkpoint_sha256")
    if not best_path.is_file():
        raise RuntimeError("completed fold selected checkpoint is missing")
    if not isinstance(expected_checkpoint_sha, str) or shared.sha256_file(
        best_path
    ) != expected_checkpoint_sha:
        raise RuntimeError("completed fold selected checkpoint SHA-256 mismatch")
    lock_sha = shared.sha256_file(selection_path)
    if manifest.get("selection_lock_sha256") != lock_sha:
        raise RuntimeError("completed fold selection-lock hash mismatch")
    if manifest.get("test_predictions_sha256") != shared.sha256_file(prediction_path):
        raise RuntimeError("completed fold prediction hash mismatch")
    if report.get("selection_lock_sha256") != lock_sha:
        raise RuntimeError("completed fold report selection-lock hash mismatch")
    if report.get("selection") != selection.get("decision"):
        raise RuntimeError("completed fold report/lock selection mismatch")
    stored_split = report.get("split", {})
    if (
        stored_split.get("outer_fold") != split.outer_fold
        or stored_split.get("validation_fold") != split.validation_fold
        or tuple(stored_split.get("test_identities", ())) != split.test_identities
    ):
        raise RuntimeError("completed fold split record mismatch")
    result = _load_prediction(prediction_path)
    expected_position = split.test[
        experiment.metadata.iloc[split.test]["reference_valid"].astype(bool).to_numpy()
    ]
    expected_position = np.sort(expected_position.astype(np.int64))
    expected_cache_index = experiment.metadata.iloc[expected_position][
        "cache_index"
    ].to_numpy(np.int64)
    if not np.array_equal(result.position, expected_position) or not np.array_equal(
        result.cache_index, expected_cache_index
    ):
        raise RuntimeError("completed fold prediction row binding is not exact")
    if int(manifest.get("test_valid_rows_expected", -1)) != len(expected_position):
        raise RuntimeError("completed fold expected-row manifest mismatch")
    promoted = bool(selection["decision"]["promoted"])
    return result, promoted, {
        "selection": selection["decision"],
        "test": report,
        "test_reused_without_evaluation": True,
    }


def train_fold(
    args: argparse.Namespace,
    experiment: EpisodeExperiment,
    fold: int,
    device: torch.device,
    run_signature: str,
) -> tuple[EpisodePrediction, bool, dict[str, Any]]:
    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    split = make_episode_split(experiment, fold)
    source_binding = _source_binding(experiment)
    completed = _completed_fold_result(
        fold_dir,
        run_signature=run_signature,
        source_binding=source_binding,
        split=split,
        experiment=experiment,
    )
    if completed is not None:
        if not args.resume:
            raise RuntimeError(
                f"fold {fold} already has a completed one-shot test; use --resume to reuse it"
            )
        return completed
    test_started_path = fold_dir / "test_evaluation_started.json"
    if test_started_path.exists():
        raise RuntimeError(
            f"fold {fold} test evaluation was started but not completed; use a fresh output directory"
        )
    scaler = fit_episode_feature_scaler(experiment, split.train)
    action = fit_train_action_calibration(
        experiment.metadata,
        split.train,
        divisor_class_balance_power=args.divisor_class_balance_power,
        strict_alias_gate=args.strict_alias_gate,
        strict_gate_margin_bpm=args.strict_gate_margin_bpm,
    )
    weak_invalid_rows = int(
        (~experiment.metadata.iloc[split.train]["reference_valid"].astype(bool)).sum()
    )
    train_frame = experiment.metadata.iloc[split.train]
    weak_invalid_quality_eligible = int(
        (
            ~train_frame["reference_valid"].astype(bool)
            & (
                pd.to_numeric(train_frame["reference_quality"], errors="coerce")
                >= float(args.weak_invalid_min_quality)
            )
        ).sum()
    )
    split_record = {
        "outer_fold": fold,
        "validation_fold": split.validation_fold,
        "train_identities": list(split.train_identities),
        "validation_identities": list(split.validation_identities),
        "test_identities": list(split.test_identities),
        "train_rows_all_windows": int(len(split.train)),
        "validation_rows_all_windows": int(len(split.validation)),
        "test_rows_all_windows": int(len(split.test)),
        "train_valid_rows": int(
            experiment.metadata.iloc[split.train]["reference_valid"].astype(bool).sum()
        ),
        "weak_invalid_weight": float(args.weak_invalid_weight),
        "weak_invalid_min_quality": float(args.weak_invalid_min_quality),
        "weak_invalid_train_rows_available": weak_invalid_rows,
        "weak_invalid_train_rows_quality_eligible": weak_invalid_quality_eligible,
        "weak_invalid_train_rows_used": weak_invalid_quality_eligible
        if args.weak_invalid_weight > 0
        else 0,
        "invalid_labels_never_used_for_validation_or_test": True,
        "session_and_identity_partition_asserted": True,
    }
    atomic_write_json(fold_dir / "split.json", split_record)
    atomic_write_json(fold_dir / "feature_scaler.json", scaler.record())
    atomic_write_json(fold_dir / "train_only_action_calibration.json", action.record())

    model_kwargs = _model_kwargs(args)
    model = EpisodeAliasRRSNN(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    grad_scaler = torch.amp.GradScaler(
        device.type, enabled=bool(args.amp and device.type == "cuda")
    )
    validation_loader = make_episode_loader(
        experiment,
        split.validation,
        scaler,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed + 1009 * fold + 1,
    )
    best_path = fold_dir / "episode_best.pt"
    last_path = fold_dir / "episode_last.pt"
    start_epoch, best_epoch, stale_epochs, best_score = 0, -1, 0, math.inf
    resume_path = args.resume_from if args.resume_from is not None else last_path
    if args.resume and not Path(resume_path).is_file():
        raise RuntimeError(f"requested resume checkpoint is missing: {resume_path}")
    if not args.resume and (best_path.exists() or last_path.exists()):
        raise RuntimeError(
            f"fold {fold} has incomplete training checkpoints; use --resume or a fresh output"
        )
    if args.resume:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        _validate_checkpoint(
            checkpoint,
            run_signature=run_signature,
            source_binding=source_binding,
            feature_scaler=scaler,
            action_calibration=action,
            split=split,
        )
        _validate_committed_best_checkpoint(checkpoint, best_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_score = float(checkpoint["best_score"])
        stale_epochs = int(checkpoint["stale_epochs"])
        rng = checkpoint.get("rng", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
    history_path = fold_dir / "history.jsonl"
    loss_kwargs = _loss_kwargs(args)
    for epoch in range(start_epoch, args.epochs):
        # Epoch-derived shuffling makes a resumed run consume exactly the same
        # episode order as an uninterrupted run without serializing a hidden
        # DataLoader generator state.
        train_loader = make_episode_loader(
            experiment,
            split.train,
            scaler,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=True,
            seed=args.seed + 1009 * fold + 104729 * epoch,
        )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            grad_scaler,
            device,
            action_calibration=action,
            weak_invalid_weight=args.weak_invalid_weight,
            weak_invalid_min_quality=args.weak_invalid_min_quality,
            allow_stacked_base_training=(
                args.allow_stacked_base_training and epoch >= args.source_warmup_epochs
            ),
            strict_alias_gate=args.strict_alias_gate,
            strict_gate_training_enabled=(epoch >= args.source_warmup_epochs),
            strict_gate_margin_bpm=args.strict_gate_margin_bpm,
            strict_gate_false_positive_cost=args.strict_gate_false_positive_cost,
            radar_dropout=args.radar_dropout,
            amp=args.amp,
            gradient_clip=args.gradient_clip,
            max_batches=args.smoke_max_batches,
            loss_kwargs=loss_kwargs,
        )
        validation_raw = predict_episode_loader(
            model,
            validation_loader,
            device,
            amp=args.amp,
            max_batches=args.smoke_max_batches,
        )
        if args.strict_alias_gate:
            epoch_gate_policy, validation = select_strict_gate_policy(
                validation_raw,
                experiment.metadata,
                maximum_coverage=args.strict_gate_max_coverage,
                tail_min_bpm=args.tail_min_bpm,
                tail_max_bpm=args.tail_max_bpm,
            )
        else:
            epoch_gate_policy, validation = None, validation_raw
        identities = experiment.metadata.iloc[validation.position]["identity"].astype(str)
        snapshot = shared.evaluation_snapshot(
            validation.target,
            validation.candidate_prediction,
            identities,
            high_min_bpm=args.tail_min_bpm,
            high_max_bpm=args.tail_max_bpm,
        )
        source_selection_snapshot = shared.evaluation_snapshot(
            validation_raw.target,
            validation_raw.source_prediction,
            identities,
            high_min_bpm=args.tail_min_bpm,
            high_max_bpm=args.tail_max_bpm,
        )
        # In strict mode even checkpoint/epoch selection is independent of the
        # frozen base. Threshold/pull is calibrated only after source weights
        # are frozen at the best source-only validation checkpoint.
        score = float(
            (
                source_selection_snapshot
                if args.strict_alias_gate
                else snapshot
            )["identity_macro"]["macro_mae"]
        )
        eligible_for_selection = epoch + 1 >= args.minimum_epochs
        improved = eligible_for_selection and score < best_score - args.min_delta
        if improved:
            best_score, best_epoch, stale_epochs = score, epoch, 0
        elif eligible_for_selection:
            stale_epochs += 1
        else:
            stale_epochs = 0
        checkpoint = _checkpoint(
            model=model,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_score=best_score,
            stale_epochs=stale_epochs,
            run_signature=run_signature,
            source_binding=source_binding,
            feature_scaler=scaler,
            action_calibration=action,
            split=split,
            model_kwargs=model_kwargs,
        )
        # Commit best first, then atomically publish last as the transaction
        # record that binds the exact best bytes. A crash between these writes
        # is detected on resume instead of silently mixing epochs.
        if improved:
            shared.atomic_torch_save(checkpoint, best_path)
        if best_epoch >= 0:
            if not best_path.is_file():
                raise RuntimeError("best checkpoint is missing before last-checkpoint commit")
            checkpoint["committed_best_checkpoint_sha256"] = shared.sha256_file(best_path)
        else:
            checkpoint["committed_best_checkpoint_sha256"] = None
        shared.atomic_torch_save(checkpoint, last_path)
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    _json_ready(
                        {
                            "epoch": epoch,
                            "train": train_metrics,
                            "validation": snapshot,
                            "validation_source_checkpoint_selection": source_selection_snapshot,
                            "validation_source_divisor": _prediction_report(
                                validation,
                                experiment.metadata,
                                tail_min_bpm=args.tail_min_bpm,
                                tail_max_bpm=args.tail_max_bpm,
                                strict_gate_margin_bpm=args.strict_gate_margin_bpm,
                            )["source_divisor_diagnostics"],
                            "eligible_for_selection": eligible_for_selection,
                            "strict_gate_policy": epoch_gate_policy.record()
                            if epoch_gate_policy is not None
                            else None,
                            "best_epoch": best_epoch,
                            "best_score": best_score,
                        }
                    ),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
        if stale_epochs >= args.patience and epoch + 1 >= args.minimum_epochs:
            break
    if not last_path.is_file():
        raise RuntimeError("training completed without a committed last checkpoint")
    committed_last = torch.load(last_path, map_location="cpu", weights_only=False)
    _validate_committed_best_checkpoint(committed_last, best_path)
    if not best_path.is_file():
        raise RuntimeError("training completed without a finite validation checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    _validate_checkpoint(
        best,
        run_signature=run_signature,
        source_binding=source_binding,
        feature_scaler=scaler,
        action_calibration=action,
        split=split,
    )
    model.load_state_dict(best["model"])
    validation_raw = predict_episode_loader(
        model,
        validation_loader,
        device,
        amp=args.amp,
        max_batches=args.smoke_max_batches,
    )
    if args.strict_alias_gate:
        strict_gate_policy, validation = select_strict_gate_policy(
            validation_raw,
            experiment.metadata,
            maximum_coverage=args.strict_gate_max_coverage,
            tail_min_bpm=args.tail_min_bpm,
            tail_max_bpm=args.tail_max_bpm,
        )
    else:
        strict_gate_policy, validation = None, validation_raw
    shared.atomic_save_npz(
        fold_dir / "validation_predictions.npz", **_prediction_arrays(validation)
    )
    validation_identities = experiment.metadata.iloc[validation.position]["identity"].astype(str)
    decision = shared.promotion_decision(
        validation.target,
        validation.candidate_prediction,
        validation.base_prediction,
        validation_identities,
        minimum_macro_mae_improvement=args.promotion_min_improvement,
        noninferiority_tolerance=args.promotion_noninferiority_tolerance,
        high_min_bpm=args.tail_min_bpm,
        high_max_bpm=args.tail_max_bpm,
    )
    promoted = bool(decision["promoted"])
    atomic_write_json(
        fold_dir / "validation_predictions.json",
        _prediction_report(
            validation,
            experiment.metadata,
            tail_min_bpm=args.tail_min_bpm,
            tail_max_bpm=args.tail_max_bpm,
        ),
    )
    selection_lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outer_fold": fold,
        "best_epoch": int(best["best_epoch"]),
        "best_validation_macro_mae": float(best["best_score"]),
        "run_signature": run_signature,
        "checkpoint_sha256": shared.sha256_file(best_path),
        "source_binding": source_binding,
        "feature_scaler_signature": _scaler_signature(scaler),
        "action_calibration_signature": _action_signature(action),
        "decision": decision,
        "locked_final_action": (
            "validation_locked_sparse_gate_mixture"
            if promoted and args.strict_alias_gate
            else "source_episode_candidate"
            if promoted
            else "exact_frozen_base"
        ),
        "strict_nested_base_unavailable": not args.allow_stacked_base_training,
        "strict_alias_gate_policy": strict_gate_policy.record()
        if strict_gate_policy is not None
        else None,
        "test_loader_constructed": False,
        "test_labels_or_metrics_used_for_selection": False,
    }
    atomic_write_json(fold_dir / "selection_lock.json", selection_lock)
    lock_sha = shared.sha256_file(fold_dir / "selection_lock.json")
    atomic_write_json(
        test_started_path,
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "outer_fold": fold,
            "selection_lock_sha256": lock_sha,
            "test_loader_constructed_after_this_marker": True,
            "test_fold_evaluation_invocation": 1,
        },
    )
    test_loader = make_episode_loader(
        experiment,
        split.test,
        scaler,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed + 1009 * fold + 2,
    )
    test_raw = predict_episode_loader(
        model,
        test_loader,
        device,
        amp=args.amp,
        max_batches=args.smoke_max_batches,
    )
    test = (
        apply_strict_gate_policy(test_raw, strict_gate_policy)
        if strict_gate_policy is not None
        else test_raw
    )
    shared.atomic_save_npz(
        fold_dir / "test_predictions.npz", **_prediction_arrays(test, promoted=promoted)
    )
    test_report = _prediction_report(
        test,
        experiment.metadata,
        promoted=promoted,
        tail_min_bpm=args.tail_min_bpm,
        tail_max_bpm=args.tail_max_bpm,
        strict_gate_margin_bpm=args.strict_gate_margin_bpm,
    )
    test_report.update(
        {
            "outer_fold": fold,
            "selection_lock_sha256": lock_sha,
            "selection": decision,
            "split": split_record,
            "test_evaluated_once_after_validation_lock": True,
        }
    )
    atomic_write_json(fold_dir / "test_predictions.json", test_report)
    expected_valid = int(
        experiment.metadata.iloc[split.test]["reference_valid"].astype(bool).sum()
    )
    atomic_write_json(
        fold_dir / "test_evaluation_manifest.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "outer_fold": fold,
            "test_fold_evaluation_invocations": 1,
            "test_valid_rows_expected": expected_valid,
            "test_valid_rows_evaluated": int(len(test.position)),
            "complete_outer_test_evaluation": bool(
                args.smoke_max_batches is None and len(test.position) == expected_valid
            ),
            "selection_lock_sha256": lock_sha,
            "test_predictions_sha256": shared.sha256_file(fold_dir / "test_predictions.npz"),
            "test_metrics_used_for_selection": False,
        },
    )
    return test, promoted, {
        "split": split_record,
        "selection": decision,
        "test": test_report,
        "test_reused_without_evaluation": False,
    }


def _build_run_config(
    args: argparse.Namespace, experiment: EpisodeExperiment
) -> dict[str, Any]:
    arguments = {
        key: value for key, value in vars(args).items() if key not in {"resume", "resume_from"}
    }
    sources = (
        Path(__file__),
        SOURCE_ROOT / "snn_rr" / "svd_episode_models.py",
        SOURCE_ROOT / "snn_rr" / "svd_features.py",
        PROJECT_ROOT / "scripts" / "train_svd_snn.py",
    )
    strict = not args.allow_stacked_base_training
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": arguments,
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): shared.sha256_file(path) for path in sources
        },
        "data_provenance": experiment.provenance,
        "split_protocol": {
            "outer_test": "identity fold f",
            "validation": "identity fold (f+1)%6",
            "weight_fit": "remaining four complete identity/session partitions",
            "normalization_and_actions": "outer-training identities only",
            "invalid_windows": "always update state; loss disabled unless weak train-only ablation",
            "selection": "source candidate vs frozen base using validation valid rows only",
            "test": "one pass after durable selection lock",
            "resume_shuffle": "deterministic epoch-derived DataLoader seed",
        },
        "stacking_audit": {
            "strict_nested_base_unavailable": strict,
            "primary_learned_inputs": "label-free SVD evidence plus classical RR/confidence/spread",
            "frozen_oof_base_use": (
                (
                    "deterministic output fallback plus validation-locked threshold/pull; "
                    "never a learned encoder/head input or learned gate target"
                    if args.strict_alias_gate
                    else "deterministic output fallback and fold-level validation policy only"
                )
                if strict
                else "secondary non-commercial ablation: learned base/alias inputs and gate"
            ),
            "risk": (
                "row-level base OOF excludes its own identity, but a training-row base learner "
                "may have seen the episode outer-test identity"
            ),
            "commercial_claim_allowed": False,
            "commercial_claim_requirement": "prospectively frozen nested base stack on unseen identities",
            "strict_alias_gate": bool(args.strict_alias_gate),
            "strict_alias_gate_target": (
                "train-valid-only indicator that best classical x2..x4 improves over "
                f"classical x1 by >= {args.strict_gate_margin_bpm:g} bpm; no base error"
                if args.strict_alias_gate
                else None
            ),
            "strict_alias_gate_risk": (
                "learned weights are base-independent, but threshold/pull and final promotion "
                "remain validation-dependent and require a prospective nested-base audit"
                if args.strict_alias_gate
                else None
            ),
        },
        "model_input_audit": {
            "reference_target": False,
            "reference_valid": False,
            "reference_quality_or_qc": False,
            "chronological_invalid_window_state_updates": True,
            "source_posterior_supervised_independently_of_gate": True,
            "source_warmup_epochs": int(args.source_warmup_epochs),
            "strict_checkpoint_selection": (
                "source-only validation macro MAE; frozen base cannot select model weights"
                if args.strict_alias_gate
                else None
            ),
            "raw_gate_logits_saved": True,
            "learned_gate_and_applied_gate_saved_separately": True,
        },
        "divisor_class_balance": {
            "power": float(args.divisor_class_balance_power),
            "fit_scope": "outer-training valid-reference oracle actions only",
            "applied_losses": ["divisor_ce", "action_regret"],
        },
        "weak_invalid_label_ablation": {
            "enabled": bool(args.weak_invalid_weight > 0),
            "weight": float(args.weak_invalid_weight),
            "minimum_quality": float(args.weak_invalid_min_quality),
            "scope": "outer-training identities only",
            "validation_test": "strict valid references only",
        },
    }
    signature_payload = {
        "arguments": arguments,
        "source_sha256": config["source_sha256"],
        "data_provenance": experiment.provenance,
        "split_protocol": config["split_protocol"],
        "stacking_audit": config["stacking_audit"],
    }
    config["run_signature"] = shared.stable_signature(signature_payload)
    return config


def _metric_bundle(
    target: np.ndarray,
    prediction: np.ndarray,
    identities: np.ndarray,
    folds: np.ndarray,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result = shared.grouped_oof_metrics(
        target,
        prediction,
        identities,
        fold_ids=folds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    result["tail_25_35"] = shared.evaluation_snapshot(
        target,
        prediction,
        identities,
        high_min_bpm=args.tail_min_bpm,
        high_max_bpm=args.tail_max_bpm,
    )["high_25_35"]
    return result


def write_oof(
    args: argparse.Namespace,
    experiment: EpisodeExperiment,
    fold_results: Mapping[int, tuple[EpisodePrediction, bool, dict[str, Any]]],
    run_signature: str,
) -> dict[str, Any]:
    count = len(experiment.metadata)
    metadata = experiment.metadata
    base = pd.to_numeric(metadata["_base_prediction"], errors="coerce").to_numpy(np.float32)
    target = pd.to_numeric(metadata["rr_bpm"], errors="coerce").to_numpy(np.float32)
    candidate = np.full(count, np.nan, np.float32)
    final = np.full(count, np.nan, np.float32)
    rr_std = np.full(count, np.nan, np.float32)
    source = np.full(count, np.nan, np.float32)
    source_std = np.full(count, np.nan, np.float32)
    gate = np.full(count, np.nan, np.float32)
    learned_gate = np.full(count, np.nan, np.float32)
    applied_gate = np.full(count, np.nan, np.float32)
    divisor = np.full((count, 4), np.nan, np.float32)
    residual = np.full(count, np.nan, np.float32)
    candidate_std = np.full((count, 4), np.nan, np.float32)
    quality = np.full(count, np.nan, np.float32)
    radar_weights = np.full((count, 3), np.nan, np.float32)
    spike = np.full(count, np.nan, np.float32)
    promoted = np.zeros(count, bool)
    evaluated = np.zeros(count, bool)
    for fold, (result, did_promote, _) in fold_results.items():
        position = result.position
        if evaluated[position].any():
            raise RuntimeError("valid OOF row was evaluated by more than one episode model")
        candidate[position] = result.candidate_prediction
        final[position] = (
            result.candidate_prediction if did_promote else result.base_prediction
        )
        rr_std[position] = result.rr_std
        source[position] = result.source_prediction
        source_std[position] = result.source_std
        gate[position] = result.mixture_gate
        learned_gate[position] = result.learned_gate
        applied_gate[position] = result.applied_gate
        divisor[position] = result.divisor_probabilities
        residual[position] = result.residual_rr
        candidate_std[position] = result.candidate_std
        quality[position] = result.quality
        radar_weights[position] = result.radar_weights
        spike[position] = result.spike_rate
        promoted[position] = bool(did_promote)
        evaluated[position] = True
    positions = np.flatnonzero(evaluated)
    if not len(positions):
        raise RuntimeError("no valid OOF rows were evaluated")
    identities = metadata["identity"].astype(str).to_numpy()
    folds = metadata["fold"].to_numpy(np.int16)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_signature": run_signature,
        "all_window_state_rows": int(count),
        "valid_reference_rows": int(metadata["reference_valid"].astype(bool).sum()),
        "evaluated_valid_rows": int(evaluated.sum()),
        "evaluated_folds": sorted(fold_results),
        "promoted_folds": sorted(fold for fold, result in fold_results.items() if result[1]),
        "complete_six_fold_valid_oof": bool(
            len(fold_results) == N_FOLDS
            and np.array_equal(evaluated, metadata["reference_valid"].astype(bool).to_numpy())
        ),
        "locked_final": _metric_bundle(
            target[positions], final[positions], identities[positions], folds[positions], args=args
        ),
        "candidate_source": _metric_bundle(
            target[positions], candidate[positions], identities[positions], folds[positions], args=args
        ),
        "candidate_mode": (
            "validation_locked_sparse_gate_mixture"
            if args.strict_alias_gate
            else "source_only" if not args.allow_stacked_base_training else "stacked_mixture"
        ),
        "frozen_base": _metric_bundle(
            target[positions], base[positions], identities[positions], folds[positions], args=args
        ),
        "fold_reports": {str(fold): result[2] for fold, result in fold_results.items()},
        "strict_nested_base_unavailable": not args.allow_stacked_base_training,
        "commercial_claim_allowed": False,
        "commercial_claim_blocker": "prospective frozen nested base/episode validation required",
    }
    output_dir = Path(args.output_dir)
    shared.atomic_save_npz(
        output_dir / "episode_oof.npz",
        index=metadata["cache_index"].to_numpy(np.int64),
        target=target,
        fold=folds,
        reference_valid=metadata["reference_valid"].astype(bool).to_numpy(),
        evaluated=evaluated,
        prediction_base=base,
        prediction_candidate=candidate,
        prediction_final=final,
        rr_std=rr_std,
        source_prediction=source,
        source_std=source_std,
        mixture_gate=gate,
        learned_gate=learned_gate,
        applied_gate=applied_gate,
        divisor_probabilities=divisor,
        residual_rr=residual,
        candidate_std=candidate_std,
        quality=quality,
        radar_weights=radar_weights,
        spike_rate=spike,
        promoted=promoted,
        run_signature=np.asarray(run_signature),
    )
    table_columns = [
        "cache_index",
        "fold",
        "session_id",
        "identity",
        "protocol",
        "window_number",
        "window_start_s",
        "window_end_s",
        "rr_bpm",
        "reference_valid",
        "classical_rr_bpm",
    ]
    table = metadata.loc[:, table_columns].copy()
    table["prediction_base_bpm"] = base
    table["prediction_candidate_bpm"] = candidate
    table["prediction_locked_final_bpm"] = final
    table["candidate_rr_std_bpm"] = rr_std
    table["source_prediction_bpm"] = source
    table["mixture_gate"] = gate
    table["learned_gate"] = learned_gate
    table["applied_gate"] = applied_gate
    for index in range(4):
        table[f"divisor_probability_x{index + 1}"] = divisor[:, index]
        table[f"candidate_std_x{index + 1}_bpm"] = candidate_std[:, index]
    table["residual_rr_bpm"] = residual
    table["quality"] = quality
    table["spike_rate"] = spike
    table["promoted"] = promoted
    table["evaluated"] = evaluated
    temporary = output_dir / "episode_oof.csv.tmp"
    table.to_csv(temporary, index=False)
    temporary.replace(output_dir / "episode_oof.csv")
    atomic_write_json(output_dir / "metrics.json", metrics)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svd-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts/cache/svd_components_all_v1",
    )
    parser.add_argument(
        "--base-oof-csv",
        type=Path,
        default=PROJECT_ROOT / "artifacts/runs/ensemble_structured_exact/ensemble_oof.csv",
    )
    parser.add_argument(
        "--alias-oof-csv",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/runs/causal_alias_decoder/causal_alias_decoder_oof.csv",
    )
    parser.add_argument(
        "--all-window-base",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/runs/final_alias_gate_s12_deterministic/all_windows_cuda_v3",
    )
    parser.add_argument(
        "--fold-assignments-json",
        type=Path,
        help=(
            "independent identity_to_fold authority; when omitted it is resolved "
            "next to the base or all-window source"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/runs/svd_episode_snn",
    )
    parser.add_argument("--fold", default="all")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--verify-file-hashes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--preset", choices=("tiny", "compact", "full"), default="compact")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--minimum-epochs", type=int, default=12)
    parser.add_argument("--source-warmup-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--smoke-max-batches", type=int)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=2.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--candidate-channels", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cell-types", default="lif,plif,alif")
    parser.add_argument("--beta", type=float, default=0.92)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--radar-dropout", type=float, default=0.15)
    parser.add_argument("--max-residual-bpm", type=float, default=1.5)
    parser.add_argument("--initial-gate-bias", type=float, default=-8.0)
    parser.add_argument("--weak-invalid-weight", type=float, default=0.0)
    parser.add_argument("--weak-invalid-min-quality", type=float, default=0.0)
    parser.add_argument("--divisor-class-balance-power", type=float, default=0.0)
    gate_mode = parser.add_mutually_exclusive_group()
    gate_mode.add_argument(
        "--allow-stacked-base-training",
        action="store_true",
        help="secondary leakage-risk ablation; never use for a commercial claim",
    )
    gate_mode.add_argument(
        "--strict-alias-gate",
        action="store_true",
        help="base-independent alias-needed gate with validation-locked sparse correction",
    )
    parser.add_argument("--strict-gate-margin-bpm", type=float, default=0.75)
    parser.add_argument("--strict-gate-false-positive-cost", type=float, default=10.0)
    parser.add_argument("--strict-gate-max-coverage", type=float, default=0.15)
    parser.add_argument("--posterior-weight", type=float, default=1.0)
    parser.add_argument("--mae-weight", type=float, default=0.6)
    parser.add_argument("--divisor-weight", type=float, default=0.45)
    parser.add_argument("--action-regret-weight", type=float, default=0.30)
    parser.add_argument("--residual-weight", type=float, default=0.15)
    parser.add_argument("--uncertainty-weight", type=float, default=0.08)
    parser.add_argument("--gate-weight", type=float, default=0.12)
    parser.add_argument("--safe-gate-weight", type=float, default=0.35)
    parser.add_argument("--quality-weight", type=float, default=0.03)
    parser.add_argument("--temporal-weight", type=float, default=0.02)
    parser.add_argument("--spike-weight", type=float, default=5.0e-4)
    parser.add_argument("--tail-min-bpm", type=float, default=25.0)
    parser.add_argument("--tail-max-bpm", type=float, default=35.0)
    parser.add_argument("--promotion-min-improvement", type=float, default=0.05)
    parser.add_argument("--promotion-noninferiority-tolerance", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    positive = (
        args.epochs,
        args.minimum_epochs,
        args.patience,
        args.batch_size,
        args.eval_batch_size,
        args.bootstrap_samples,
    )
    if any(value < 1 for value in positive):
        parser.error("epochs, patience, batch sizes and bootstrap samples must be positive")
    if args.minimum_epochs > args.epochs:
        parser.error("--minimum-epochs cannot exceed --epochs")
    if args.source_warmup_epochs < 0:
        parser.error("--source-warmup-epochs cannot be negative")
    if not 0 <= args.weak_invalid_weight <= 1:
        parser.error("--weak-invalid-weight must be in [0, 1]")
    if not 0 <= args.weak_invalid_min_quality <= 1:
        parser.error("--weak-invalid-min-quality must be in [0, 1]")
    if args.divisor_class_balance_power < 0:
        parser.error("--divisor-class-balance-power cannot be negative")
    if args.strict_gate_margin_bpm < 0:
        parser.error("--strict-gate-margin-bpm cannot be negative")
    if args.strict_gate_false_positive_cost < 1:
        parser.error("--strict-gate-false-positive-cost must be >= 1")
    if not 0 < args.strict_gate_max_coverage <= 1:
        parser.error("--strict-gate-max-coverage must be in (0, 1]")
    if not 0 <= args.radar_dropout <= 1:
        parser.error("--radar-dropout must be in [0, 1]")
    if args.smoke_max_batches is not None and args.smoke_max_batches < 1:
        parser.error("--smoke-max-batches must be positive")
    if args.resume_from is not None and args.fold == "all":
        parser.error("--resume-from requires a single --fold")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    shared.seed_everything(args.seed, deterministic=args.deterministic)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    experiment = load_episode_experiment(
        args.svd_cache,
        args.base_oof_csv,
        args.alias_oof_csv,
        args.all_window_base,
        fold_assignments_json=args.fold_assignments_json,
        verify_file_hashes=args.verify_file_hashes,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_config = _build_run_config(args, experiment)
    run_config_path = Path(args.output_dir) / "run_config.json"
    if run_config_path.is_file():
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_config.get("run_signature") != run_config.get("run_signature"):
            raise RuntimeError(
                "existing output run_config does not match this invocation; "
                "use a fresh output directory"
            )
        # Preserve the original creation time and exact audited configuration.
        run_config = existing_config
    else:
        atomic_write_json(run_config_path, run_config)
    folds = shared.parse_fold_selection(args.fold)
    results: dict[int, tuple[EpisodePrediction, bool, dict[str, Any]]] = {}
    for fold in folds:
        shared.seed_everything(args.seed + 1009 * fold, deterministic=args.deterministic)
        results[fold] = train_fold(
            args, experiment, fold, device, str(run_config["run_signature"])
        )
    metrics = write_oof(
        args, experiment, results, str(run_config["run_signature"])
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "run_signature": run_config["run_signature"],
                "evaluated_folds": folds,
                "promoted_folds": metrics["promoted_folds"],
                "locked_final_mae": metrics["locked_final"]["overall"]["mae"],
                "locked_final_macro_mae": metrics["locked_final"]["identity_macro"][
                    "macro_mae"
                ],
                "strict_nested_base_unavailable": metrics[
                    "strict_nested_base_unavailable"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
