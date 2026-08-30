#!/usr/bin/env python3
"""Build a bounded-memory, label-free harmonic candidate-set feature cache."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_rr.harmonic_set_data import (  # noqa: E402
    CANDIDATE_SOURCE_NAMES,
    FORBIDDEN_TARGET_QC_FIELDS,
    HARMONIC_RATIOS,
    RF_BRANCH_NAMES,
    SEMANTIC_ROW_FIELDS,
    VERIFIED_SVD_VARIANT_INDICES,
    VERIFIED_SVD_VARIANT_NAMES,
    CandidateSource,
    candidate_bank_from_metadata,
    iter_compact_node_feature_batches,
    semantic_row_binding_sha256,
)


DEFAULT_RF_CACHE = PROJECT_ROOT / "artifacts/cache/rf32s"
DEFAULT_SVD_CACHE = PROJECT_ROOT / "artifacts/cache/svd_components_all_v1"
DEFAULT_PROPOSER = (
    PROJECT_ROOT
    / "artifacts/runs/final_alias_gate_s12_deterministic/all_windows_cuda_v3"
    / "snn_all_windows.npz"
)
DEFAULT_FOLDS = (
    PROJECT_ROOT
    / "artifacts/runs/final_alias_gate_s12_deterministic/fold_assignments.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/cache/harmonic_set_v2"
FORMAT_VERSION = 1
MAX_CANDIDATES = 12
TOPK_PROPOSALS = 5
POSTERIOR_GRID_KEYS = ("posterior_rr_grid_bpm", "posterior_rr_bins_bpm")
BASE_PROPOSAL_CHOICES = ("none", "expected", "map", "expected-map")

ARRAY_FILES = {
    "node_features": "node_features.npy",
    "candidate_bpm": "candidate_bpm.npy",
    "candidate_mask": "candidate_mask.npy",
    "candidate_confidence": "candidate_confidence.npy",
    "candidate_source_mask": "candidate_source_mask.npy",
    "candidate_primary_source": "candidate_primary_source.npy",
    "joint_radar_mask": "joint_radar_mask.npy",
    "rf_support_count": "rf_support_count.npy",
    "svd_support_count": "svd_support_count.npy",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _canonical_digest(value: Mapping[str, Any], *, exclude: str | None = None) -> str:
    payload = dict(value)
    if exclude is not None:
        payload.pop(exclude, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _successful_sessions(manifest: Mapping[str, Any], label: str) -> list[str]:
    raw = manifest.get("sessions")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"{label} manifest has no sessions")
    result: list[str] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"{label} manifest session entry is invalid")
        if entry.get("status", "ok") == "ok":
            session_id = str(entry.get("session_id", ""))
            if not session_id:
                raise RuntimeError(f"{label} manifest has an empty session_id")
            result.append(session_id)
    if not result or len(set(result)) != len(result):
        raise RuntimeError(f"{label} successful session order is invalid")
    return result


def _fold_map(path: Path) -> dict[str, int]:
    document = _load_json(path)
    raw = document.get("identity_to_fold", document)
    if not isinstance(raw, Mapping) or not raw:
        raise RuntimeError("fold assignments contain no identity map")
    result: dict[str, int] = {}
    for identity, fold in raw.items():
        if not isinstance(identity, str) or not identity:
            raise RuntimeError("fold assignment identity is invalid")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
            raise RuntimeError(f"fold assignment for {identity!r} is invalid")
        result[identity] = int(fold)
    return result


def _normalized_semantic_frame(
    frame: pd.DataFrame, cache_index: np.ndarray, fold: np.ndarray
) -> pd.DataFrame:
    result = frame.copy()
    result["cache_index"] = np.asarray(cache_index, dtype=np.int64)
    result["fold"] = np.asarray(fold, dtype=np.int16)
    return result.loc[:, list(SEMANTIC_ROW_FIELDS)]


def _assert_common_rows(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    fields = (
        "session_id",
        "identity",
        "protocol",
        "window_number",
        "window_start_s",
        "window_end_s",
    )
    if len(left) != len(right):
        raise RuntimeError(f"{label} row count mismatch")
    for field in fields:
        if field not in left or field not in right:
            raise RuntimeError(f"{label} lacks semantic field {field}")
        if field in {"session_id", "identity", "protocol"}:
            equal = np.array_equal(
                left[field].astype(str).to_numpy(), right[field].astype(str).to_numpy()
            )
        elif field == "window_number":
            equal = np.array_equal(
                pd.to_numeric(left[field], errors="raise").to_numpy(np.int64),
                pd.to_numeric(right[field], errors="raise").to_numpy(np.int64),
            )
        else:
            equal = np.allclose(
                pd.to_numeric(left[field], errors="raise").to_numpy(float),
                pd.to_numeric(right[field], errors="raise").to_numpy(float),
                rtol=0.0,
                atol=5.0e-7,
            )
        if not equal:
            raise RuntimeError(f"{label} semantic field mismatch: {field}")


def _proposer_frame(data: Mapping[str, np.ndarray]) -> pd.DataFrame:
    required = {
        "cache_index",
        "fold",
        "session_id",
        "identity",
        "protocol",
        "window_number",
        "window_start_s",
        "window_end_s",
    }
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(f"proposer lacks semantic fields: {missing}")
    return pd.DataFrame({field: np.asarray(data[field]) for field in required})


@dataclass(frozen=True, slots=True)
class ProposalBundle:
    """Validated label-free proposer content in fixed priority order."""

    bpm: np.ndarray
    confidence: np.ndarray
    mask: np.ndarray
    source: np.ndarray
    availability: np.ndarray
    direct_bpm: np.ndarray
    direct_confidence: np.ndarray
    direct_mask: np.ndarray
    posterior_probability: np.ndarray | None
    posterior_rr_grid_bpm: np.ndarray | None
    expected_bpm: np.ndarray | None
    map_bpm: np.ndarray | None
    entropy_normalized: np.ndarray | None
    rr_std_bpm: np.ndarray | None
    quality: np.ndarray | None
    alias_probability: np.ndarray | None
    spike_rate: np.ndarray | None
    radar_weights: np.ndarray | None
    posterior_grid_input_key: str | None


def _proposal_availability(data: Mapping[str, np.ndarray], rows: int) -> np.ndarray:
    if "proposal_available" not in data:
        return np.ones(rows, dtype=bool)
    raw = np.asarray(data["proposal_available"])
    if raw.dtype != np.bool_ or raw.shape != (rows,):
        raise RuntimeError("proposal_available must be a boolean [rows] array")
    return raw.astype(bool, copy=True)


def _topk_proposal_arrays(
    data: Mapping[str, np.ndarray], availability: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rr_key = "topk_rr_bpm" if "topk_rr_bpm" in data else "topk_rr"
    probability_key = "topk_probability"
    if rr_key not in data or probability_key not in data:
        raise RuntimeError("proposer lacks top-5 RR/probability arrays")
    rr = np.asarray(data[rr_key], dtype=np.float32)
    probability = np.asarray(data[probability_key], dtype=np.float32)
    if rr.ndim != 2 or probability.shape != rr.shape or rr.shape[1] < TOPK_PROPOSALS:
        raise RuntimeError("proposer top-k arrays are incompatible with top5 construction")
    rr = rr[:, :TOPK_PROPOSALS]
    probability = probability[:, :TOPK_PROPOSALS]
    if len(rr) != len(availability):
        raise RuntimeError("proposer top5/availability row count mismatch")
    if not np.isfinite(rr[availability]).all() or not np.isfinite(
        probability[availability]
    ).all():
        raise RuntimeError("available proposer top5 contains non-finite values")
    if np.any(probability[availability] < 0) or np.any(
        probability[availability] > 1.0 + 1.0e-6
    ):
        raise RuntimeError("available proposer top5 confidence is outside [0,1]")
    if np.any((rr[availability] < 6.0) | (rr[availability] > 45.0)):
        raise RuntimeError("available proposer top5 RR is outside [6,45] bpm")
    mask = np.broadcast_to(availability[:, None], rr.shape).copy()
    rr = np.where(mask, rr, 0.0).astype(np.float32, copy=False)
    probability = np.where(mask, probability, 0.0).astype(np.float32, copy=False)
    return rr, probability, mask


def _posterior_grid(
    data: Mapping[str, np.ndarray], rows: int, availability: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    key = next((name for name in POSTERIOR_GRID_KEYS if name in data), None)
    if key is None or "posterior_probability" not in data:
        raise RuntimeError(
            "posterior selection/features require posterior_probability and "
            "posterior_rr_grid_bpm"
        )
    probability = np.asarray(data["posterior_probability"], dtype=np.float64)
    grid = np.asarray(data[key], dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] != rows:
        raise RuntimeError("posterior_probability must have shape [rows,rr_grid]")
    if grid.ndim != 1 or probability.shape[1] != len(grid) or len(grid) < 2:
        raise RuntimeError("posterior RR grid shape is inconsistent with probabilities")
    if not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise RuntimeError("posterior RR grid must be finite and strictly increasing")
    if grid[0] < 6.0 - 1.0e-6 or grid[-1] > 45.0 + 1.0e-6:
        raise RuntimeError("posterior RR grid is outside the declared [6,45] bpm range")
    active = probability[availability]
    if not np.isfinite(active).all() or np.any(active < 0):
        raise RuntimeError("available posterior probabilities are non-finite or negative")
    sums = active.sum(axis=1)
    if np.any(sums <= 0) or not np.allclose(sums, 1.0, rtol=0.0, atol=2.0e-3):
        raise RuntimeError("available posterior rows are not normalized probabilities")
    inactive = probability[~availability]
    if inactive.size and (
        not np.isfinite(inactive).all() or np.count_nonzero(inactive) != 0
    ):
        raise RuntimeError(
            "unavailable proposer rows must carry an exactly zero finite posterior"
        )
    normalized = np.zeros_like(probability, dtype=np.float64)
    normalized[availability] = active / sums[:, None]
    return normalized, grid, key


def posterior_nms_modes(
    posterior_probability: np.ndarray,
    posterior_rr_grid_bpm: np.ndarray,
    availability: np.ndarray,
    *,
    top_k: int = TOPK_PROPOSALS,
    suppression_bpm: float = 1.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy deterministic posterior NMS with stable lower-grid tie priority."""

    probability = np.asarray(posterior_probability, dtype=np.float64)
    grid = np.asarray(posterior_rr_grid_bpm, dtype=np.float64)
    available = np.asarray(availability, dtype=bool)
    if probability.ndim != 2 or grid.shape != (probability.shape[1],):
        raise ValueError("posterior probability/grid shapes are incompatible")
    if available.shape != (probability.shape[0],):
        raise ValueError("posterior availability shape mismatch")
    if int(top_k) < 1 or not np.isfinite(suppression_bpm) or suppression_bpm < 0:
        raise ValueError("posterior NMS settings are invalid")
    rr = np.zeros((len(probability), int(top_k)), dtype=np.float32)
    confidence = np.zeros_like(rr)
    mask = np.zeros(rr.shape, dtype=bool)
    for row in np.flatnonzero(available):
        # mergesort/stable preserves increasing RR-grid order for exact ties.
        order = np.argsort(-probability[row], kind="stable")
        selected: list[int] = []
        for index in order:
            if probability[row, index] <= 0:
                break
            if any(
                abs(float(grid[index] - grid[other]))
                <= float(suppression_bpm) + 1.0e-12
                for other in selected
            ):
                continue
            selected.append(int(index))
            if len(selected) == int(top_k):
                break
        if selected:
            count = len(selected)
            rr[row, :count] = grid[selected]
            confidence[row, :count] = probability[row, selected]
            mask[row, :count] = True
    return rr, confidence, mask


def _required_scalar(
    data: Mapping[str, np.ndarray],
    keys: Sequence[str],
    availability: np.ndarray,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> np.ndarray:
    key = next((name for name in keys if name in data), None)
    if key is None:
        raise RuntimeError(f"proposer lacks required {label} array")
    values = np.asarray(data[key], dtype=np.float64)
    if values.shape != availability.shape:
        raise RuntimeError(f"proposer {label} must have shape [rows]")
    active = values[availability]
    if not np.isfinite(active).all():
        raise RuntimeError(f"available proposer {label} contains non-finite values")
    if strictly_positive and np.any(active <= 0):
        raise RuntimeError(f"available proposer {label} must be positive")
    if minimum is not None and np.any(active < minimum - 1.0e-7):
        raise RuntimeError(f"available proposer {label} is below {minimum}")
    if maximum is not None and np.any(active > maximum + 1.0e-7):
        raise RuntimeError(f"available proposer {label} is above {maximum}")
    return np.where(availability, values, 0.0).astype(np.float32)


def _proposal_bundle(
    data: Mapping[str, np.ndarray],
    *,
    selection: str,
    suppression_bpm: float,
    base_proposals: str,
    include_features: bool,
) -> ProposalBundle:
    rows = len(np.asarray(data.get("cache_index", ())))
    if rows < 1:
        raise RuntimeError("proposer contains no canonical rows")
    availability = _proposal_availability(data, rows)
    require_posterior = (
        selection == "posterior-nms"
        or base_proposals != "none"
        or include_features
    )
    posterior: np.ndarray | None = None
    grid: np.ndarray | None = None
    grid_key: str | None = None
    expected: np.ndarray | None = None
    map_bpm: np.ndarray | None = None
    entropy: np.ndarray | None = None
    rr_std: np.ndarray | None = None
    quality: np.ndarray | None = None
    alias: np.ndarray | None = None
    spike: np.ndarray | None = None
    radar_weights: np.ndarray | None = None
    if require_posterior:
        posterior, grid, grid_key = _posterior_grid(data, rows, availability)
        expected = (posterior @ grid).astype(np.float32)
        map_index = np.argmax(posterior, axis=1)
        map_bpm = np.where(availability, grid[map_index], 0.0).astype(np.float32)
        entropy = np.zeros(rows, dtype=np.float32)
        if availability.any():
            active = posterior[availability]
            entropy[availability] = (
                -(
                    active * np.log(np.maximum(active, np.finfo(np.float64).tiny))
                ).sum(axis=1)
                / np.log(float(active.shape[1]))
            ).astype(np.float32)
    if selection == "topk":
        direct_rr, direct_conf, direct_mask = _topk_proposal_arrays(
            data, availability
        )
    elif selection == "posterior-nms":
        assert posterior is not None and grid is not None
        direct_rr, direct_conf, direct_mask = posterior_nms_modes(
            posterior,
            grid,
            availability,
            top_k=TOPK_PROPOSALS,
            suppression_bpm=suppression_bpm,
        )
    else:
        raise ValueError("proposal selection must be topk or posterior-nms")

    proposal_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    base_order = (
        ("expected", "map")
        if base_proposals == "expected-map"
        else () if base_proposals == "none" else (base_proposals,)
    )
    for kind in base_order:
        values = expected if kind == "expected" else map_bpm
        assert values is not None and posterior is not None and grid is not None
        if kind == "expected":
            base_confidence = np.asarray(
                [
                    posterior[row, np.abs(grid - values[row]) <= 1.0 + 1.0e-12].sum()
                    if availability[row]
                    else 0.0
                    for row in range(rows)
                ],
                dtype=np.float32,
            )
        else:
            base_confidence = posterior.max(axis=1).astype(np.float32)
        base_mask = availability & np.isfinite(values) & (values >= 6.0) & (values <= 45.0)
        proposal_parts.append(np.where(base_mask, values, 0.0)[:, None])
        confidence_parts.append(np.where(base_mask, base_confidence, 0.0)[:, None])
        mask_parts.append(base_mask[:, None])
        source_parts.append(
            np.full((rows, 1), int(CandidateSource.BASE), dtype=np.int16)
        )
    proposal_parts.append(direct_rr)
    confidence_parts.append(direct_conf)
    mask_parts.append(direct_mask)
    source_parts.append(
        np.full(direct_rr.shape, int(CandidateSource.DIRECT_MODE), dtype=np.int16)
    )

    if include_features:
        rr_std = _required_scalar(
            data,
            ("rr_std", "rr_std_bpm"),
            availability,
            label="rr_std",
            strictly_positive=True,
        )
        quality = _required_scalar(
            data, ("quality",), availability, label="quality", minimum=0.0, maximum=1.0
        )
        alias = _required_scalar(
            data,
            ("alias_probability",),
            availability,
            label="alias_probability",
            minimum=0.0,
            maximum=1.0,
        )
        spike = _required_scalar(
            data,
            ("spike_rate",),
            availability,
            label="spike_rate",
            minimum=0.0,
            maximum=1.0,
        )
        if "radar_weights" not in data:
            raise RuntimeError("proposer lacks required radar_weights array")
        raw_weights = np.asarray(data["radar_weights"], dtype=np.float64)
        if raw_weights.shape != (rows, 3):
            raise RuntimeError("proposer radar_weights must have shape [rows,3]")
        if (
            not np.isfinite(raw_weights[availability]).all()
            or np.any(raw_weights[availability] < -1.0e-7)
            or np.any(raw_weights[availability] > 1.0 + 1.0e-7)
        ):
            raise RuntimeError("available proposer radar_weights are invalid")
        radar_weights = np.where(availability[:, None], raw_weights, 0.0).astype(
            np.float32
        )

    return ProposalBundle(
        bpm=np.concatenate(proposal_parts, axis=1).astype(np.float32),
        confidence=np.concatenate(confidence_parts, axis=1).astype(np.float32),
        mask=np.concatenate(mask_parts, axis=1).astype(bool),
        source=np.concatenate(source_parts, axis=1).astype(np.int16),
        availability=availability,
        direct_bpm=direct_rr,
        direct_confidence=direct_conf,
        direct_mask=direct_mask,
        posterior_probability=posterior,
        posterior_rr_grid_bpm=grid,
        expected_bpm=expected,
        map_bpm=map_bpm,
        entropy_normalized=entropy,
        rr_std_bpm=rr_std,
        quality=quality,
        alias_probability=alias,
        spike_rate=spike,
        radar_weights=radar_weights,
        posterior_grid_input_key=grid_key,
    )


def _load_proposer(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


PROPOSER_NODE_FEATURE_NAMES = (
    "proposer_available",
    "direct_mode_rank",
    "direct_mode_reciprocal_rank",
    "direct_mode_selected_probability",
    "posterior_nearest_bin_probability",
    "posterior_peak_probability",
    "posterior_local_mass_pm0p5_bpm",
    "posterior_local_mass_pm1p0_bpm",
    "posterior_entropy_normalized",
    "proposer_log_rr_std_bpm",
    "proposer_quality",
    "proposer_alias_probability",
    "proposer_spike_rate",
    "candidate_minus_expected_bpm",
    "candidate_abs_expected_distance_bpm",
    "candidate_minus_map_bpm",
    "candidate_abs_map_distance_bpm",
    "proposer_expected_anchor_match",
    "proposer_map_anchor_match",
    "proposer_radar1_weight",
    "proposer_radar2_weight",
    "proposer_radar3_weight",
)


def proposer_candidate_node_features(
    bundle: ProposalBundle,
    candidates: Any,
    row_selector: slice | np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Bind full-posterior deployment descriptors to the final sorted anchors."""

    required = (
        bundle.posterior_probability,
        bundle.posterior_rr_grid_bpm,
        bundle.expected_bpm,
        bundle.map_bpm,
        bundle.entropy_normalized,
        bundle.rr_std_bpm,
        bundle.quality,
        bundle.alias_probability,
        bundle.spike_rate,
        bundle.radar_weights,
    )
    if any(value is None for value in required):
        raise RuntimeError("full proposer node features were not validated")
    posterior = np.asarray(bundle.posterior_probability)[row_selector]
    grid = np.asarray(bundle.posterior_rr_grid_bpm)
    expected = np.asarray(bundle.expected_bpm)[row_selector]
    map_bpm = np.asarray(bundle.map_bpm)[row_selector]
    entropy = np.asarray(bundle.entropy_normalized)[row_selector]
    rr_std = np.asarray(bundle.rr_std_bpm)[row_selector]
    quality = np.asarray(bundle.quality)[row_selector]
    alias = np.asarray(bundle.alias_probability)[row_selector]
    spike = np.asarray(bundle.spike_rate)[row_selector]
    radar_weights = np.asarray(bundle.radar_weights)[row_selector]
    availability = np.asarray(bundle.availability)[row_selector]
    direct_rr = np.asarray(bundle.direct_bpm)[row_selector]
    direct_confidence = np.asarray(bundle.direct_confidence)[row_selector]
    direct_mask = np.asarray(bundle.direct_mask)[row_selector]
    bpm = np.asarray(candidates.bpm, dtype=np.float64)
    candidate_mask = np.asarray(candidates.mask, dtype=bool)
    if bpm.shape[0] != len(availability):
        raise RuntimeError("proposer/candidate row subset mismatch")
    features = np.zeros(
        (*bpm.shape, len(PROPOSER_NODE_FEATURE_NAMES)), dtype=np.float32
    )
    name_to_index = {
        name: index for index, name in enumerate(PROPOSER_NODE_FEATURE_NAMES)
    }
    for row in range(len(bpm)):
        if not availability[row]:
            # This is the crucial outer-test behavior: even classical-only nodes
            # receive no proposer descriptor when their nested proposal is absent.
            continue
        active_candidates = np.flatnonzero(candidate_mask[row])
        for candidate in active_candidates:
            value = float(bpm[row, candidate])
            output = features[row, candidate]
            output[name_to_index["proposer_available"]] = 1.0
            eligible = np.flatnonzero(
                direct_mask[row]
                & (
                    np.abs(np.asarray(direct_rr[row], dtype=np.float64) - value)
                    <= float(candidates.merge_radius_bpm) + 1.0e-12
                )
            )
            if len(eligible):
                # Direct proposals are already in stable NMS/top-k rank order.
                rank_index = int(eligible[0])
                output[name_to_index["direct_mode_rank"]] = float(rank_index + 1)
                output[name_to_index["direct_mode_reciprocal_rank"]] = 1.0 / float(
                    rank_index + 1
                )
                output[name_to_index["direct_mode_selected_probability"]] = float(
                    direct_confidence[row, rank_index]
                )
            nearest = int(np.argmin(np.abs(grid - value)))
            output[name_to_index["posterior_nearest_bin_probability"]] = float(
                posterior[row, nearest]
            )
            local_peak = np.abs(grid - value) <= 0.5 + 1.0e-12
            output[name_to_index["posterior_peak_probability"]] = float(
                posterior[row, local_peak].max(initial=0.0)
            )
            for radius, feature_name in (
                (0.5, "posterior_local_mass_pm0p5_bpm"),
                (1.0, "posterior_local_mass_pm1p0_bpm"),
            ):
                output[name_to_index[feature_name]] = float(
                    posterior[row, np.abs(grid - value) <= radius + 1.0e-12].sum()
                )
            output[name_to_index["posterior_entropy_normalized"]] = float(
                entropy[row]
            )
            output[name_to_index["proposer_log_rr_std_bpm"]] = float(
                np.log(max(float(rr_std[row]), 1.0e-8))
            )
            output[name_to_index["proposer_quality"]] = float(quality[row])
            output[name_to_index["proposer_alias_probability"]] = float(alias[row])
            output[name_to_index["proposer_spike_rate"]] = float(spike[row])
            expected_delta = value - float(expected[row])
            map_delta = value - float(map_bpm[row])
            output[name_to_index["candidate_minus_expected_bpm"]] = expected_delta
            output[name_to_index["candidate_abs_expected_distance_bpm"]] = abs(
                expected_delta
            )
            output[name_to_index["candidate_minus_map_bpm"]] = map_delta
            output[name_to_index["candidate_abs_map_distance_bpm"]] = abs(map_delta)
            output[name_to_index["proposer_expected_anchor_match"]] = float(
                abs(expected_delta)
                <= float(candidates.merge_radius_bpm) + 1.0e-12
            )
            output[name_to_index["proposer_map_anchor_match"]] = float(
                abs(map_delta) <= float(candidates.merge_radius_bpm) + 1.0e-12
            )
            output[
                [
                    name_to_index["proposer_radar1_weight"],
                    name_to_index["proposer_radar2_weight"],
                    name_to_index["proposer_radar3_weight"],
                ]
            ] = radar_weights[row]
    if not np.isfinite(features[candidate_mask]).all():
        raise RuntimeError("proposer candidate feature construction is non-finite")
    features *= candidate_mask[..., None]
    return features, PROPOSER_NODE_FEATURE_NAMES


def _validate_root_manifests(
    rf_cache: Path, svd_cache: Path
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    rf_path = rf_cache / "manifest.json"
    svd_path = svd_cache / "manifest.json"
    rf = _load_json(rf_path)
    svd = _load_json(svd_path)
    rf_sessions = _successful_sessions(rf, "RF")
    svd_sessions = _successful_sessions(svd, "SVD")
    if rf_sessions != svd_sessions:
        raise RuntimeError("RF/SVD successful session order differs")
    if bool(svd.get("valid_only", True)):
        raise RuntimeError("SVD cache is not all-window")
    if svd.get("label_inputs", []):
        raise RuntimeError("SVD cache declares label-derived inputs")
    if int(svd.get("components", -1)) < 6:
        raise RuntimeError("SVD cache has fewer than six components")
    names = tuple(svd.get("variant_names", ()))
    if tuple(names[index] for index in VERIFIED_SVD_VARIANT_INDICES) != tuple(
        VERIFIED_SVD_VARIANT_NAMES
    ):
        raise RuntimeError("SVD verified variant names/order mismatch")
    if str(svd.get("canonical_manifest_sha256", "")) != sha256_file(rf_path):
        raise RuntimeError("SVD cache is not bound to the requested RF manifest")
    return rf, svd, rf_sessions


def collect_input_bindings(
    rf_cache: Path,
    svd_cache: Path,
    proposer_path: Path,
    folds_path: Path,
    sessions: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": {
            "builder": _file_binding(Path(__file__)),
            "harmonic_set_data": _file_binding(
                PROJECT_ROOT / "src/snn_rr/harmonic_set_data.py"
            ),
        },
        "rf_root_manifest": _file_binding(rf_cache / "manifest.json"),
        "svd_root_manifest": _file_binding(svd_cache / "manifest.json"),
        "proposer": _file_binding(proposer_path),
        "fold_assignments": _file_binding(folds_path),
        "sessions": {},
    }
    for session_id in sessions:
        rf_dir = rf_cache / session_id
        svd_dir = svd_cache / session_id
        result["sessions"][session_id] = {
            "rf": {
                name: _file_binding(rf_dir / filename)
                for name, filename in (
                    ("maps", "maps.npy"),
                    ("metadata", "metadata.csv"),
                    ("frequencies", "frequencies_hz.npy"),
                    ("manifest", "manifest.json"),
                )
            },
            "svd": {
                name: _file_binding(svd_dir / filename)
                for name, filename in (
                    ("spectra", "spectra.npy"),
                    ("attributes", "attributes.npy"),
                    ("metadata", "metadata.csv"),
                    ("frequencies", "frequencies_hz.npy"),
                    ("manifest", "manifest.json"),
                )
            },
        }
    return result


def _verify_reuse(
    output_dir: Path, build_signature: str, input_bindings: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("output exists but has no complete manifest")
    manifest = _load_json(manifest_path)
    if manifest.get("format_version") != FORMAT_VERSION or not manifest.get("complete"):
        raise RuntimeError("output exists but is partial/incompatible")
    if manifest.get("content_sha256") != _canonical_digest(
        manifest, exclude="content_sha256"
    ):
        raise RuntimeError("existing output manifest content hash mismatch")
    if manifest.get("build_signature_sha256") != build_signature:
        raise RuntimeError("existing output was built from different settings or inputs")
    if manifest.get("inputs") != input_bindings:
        raise RuntimeError("existing output input binding mismatch")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError("existing output has no output bindings")
    for name, binding in outputs.items():
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"existing output binding is invalid: {name}")
        path = output_dir / str(binding.get("filename", ""))
        if not path.is_file():
            raise RuntimeError(f"existing output file is missing: {name}")
        if path.stat().st_size != int(binding.get("bytes", -1)):
            raise RuntimeError(f"existing output size mismatch: {name}")
        if sha256_file(path) != binding.get("sha256"):
            raise RuntimeError(f"existing output SHA-256 mismatch: {name}")
    return {"status": "reused", "output_dir": str(output_dir), "manifest": manifest}


def _open_output_arrays(stage: Path, rows: int) -> dict[str, np.memmap]:
    specifications = {
        "candidate_bpm": (np.float32, (rows, MAX_CANDIDATES)),
        "candidate_mask": (np.bool_, (rows, MAX_CANDIDATES)),
        "candidate_confidence": (np.float32, (rows, MAX_CANDIDATES)),
        "candidate_source_mask": (
            np.bool_,
            (rows, MAX_CANDIDATES, len(CANDIDATE_SOURCE_NAMES)),
        ),
        "candidate_primary_source": (np.int16, (rows, MAX_CANDIDATES)),
        "joint_radar_mask": (np.bool_, (rows, 3)),
        "rf_support_count": (np.uint8, (rows, MAX_CANDIDATES, len(HARMONIC_RATIOS))),
        "svd_support_count": (np.uint8, (rows, MAX_CANDIDATES, len(HARMONIC_RATIOS))),
    }
    return {
        name: np.lib.format.open_memmap(
            stage / ARRAY_FILES[name], mode="w+", dtype=dtype, shape=shape
        )
        for name, (dtype, shape) in specifications.items()
    }


def _flush(arrays: Mapping[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def _output_bindings(stage: Path) -> dict[str, Any]:
    names = {**ARRAY_FILES, "metadata": "metadata.csv", "feature_names": "feature_names.json"}
    return {
        name: {
            "filename": filename,
            "sha256": sha256_file(stage / filename),
            "bytes": (stage / filename).stat().st_size,
        }
        for name, filename in names.items()
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    rf_cache = args.rf_cache.expanduser().resolve()
    svd_cache = args.svd_cache.expanduser().resolve()
    proposer_path = args.proposer.expanduser().resolve()
    folds_path = args.fold_assignments.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    proposal_selection = str(getattr(args, "proposal_selection", "topk"))
    suppression_bpm = float(
        getattr(args, "posterior_nms_suppression_bpm", 1.25)
    )
    base_proposals = str(getattr(args, "base_proposals", "none"))
    include_proposer_features = bool(
        getattr(args, "proposer_features", False)
    )
    svd_components = int(getattr(args, "svd_components", 6))
    if args.batch_size < 1 or args.merge_radius_bpm < 0:
        raise ValueError("batch size must be positive and merge radius non-negative")
    if proposal_selection not in ("topk", "posterior-nms"):
        raise ValueError("proposal selection must be topk or posterior-nms")
    if base_proposals not in BASE_PROPOSAL_CHOICES:
        raise ValueError(f"base proposals must be one of {BASE_PROPOSAL_CHOICES}")
    if not np.isfinite(suppression_bpm) or suppression_bpm < 0:
        raise ValueError("posterior NMS suppression must be finite and non-negative")
    if svd_components not in (6, 12):
        raise ValueError("svd components must be 6 or 12")
    _, svd_root, sessions = _validate_root_manifests(rf_cache, svd_cache)
    if int(svd_root.get("components", -1)) < svd_components:
        raise RuntimeError(
            f"SVD cache contains fewer than the requested {svd_components} components"
        )
    folds = _fold_map(folds_path)
    proposer = _load_proposer(proposer_path)
    proposer_frame = _proposer_frame(proposer)
    proposal_bundle = _proposal_bundle(
        proposer,
        selection=proposal_selection,
        suppression_bpm=suppression_bpm,
        base_proposals=base_proposals,
        include_features=include_proposer_features,
    )
    if len(proposer_frame) != len(proposal_bundle.bpm):
        raise RuntimeError("proposer semantic/proposal row count mismatch")
    cache_index = pd.to_numeric(
        proposer_frame["cache_index"], errors="raise"
    ).to_numpy(np.int64)
    if not np.array_equal(cache_index, np.arange(len(cache_index), dtype=np.int64)):
        raise RuntimeError("proposer cache_index must be canonical global order 0..N-1")
    if proposer_frame["cache_index"].duplicated().any():
        raise RuntimeError("proposer cache_index is not unique")
    expected_fold = np.asarray(
        [folds.get(str(identity), -1) for identity in proposer_frame["identity"]],
        dtype=np.int16,
    )
    if np.any(expected_fold < 0) or not np.array_equal(
        proposer_frame["fold"].to_numpy(np.int16), expected_fold
    ):
        raise RuntimeError("proposer fold ownership differs from fold assignments")
    if set(map(str, proposer_frame["identity"])) != set(folds):
        raise RuntimeError("fold assignments/proposer identity cover mismatch")

    input_bindings = collect_input_bindings(
        rf_cache, svd_cache, proposer_path, folds_path, sessions
    )
    settings = {
        "format_version": FORMAT_VERSION,
        "merge_radius_bpm": float(args.merge_radius_bpm),
        "batch_size": int(args.batch_size),
        "maximum_candidates": MAX_CANDIDATES,
        "proposer_topk": TOPK_PROPOSALS,
        "proposal_selection": proposal_selection,
        "posterior_nms_suppression_bpm": suppression_bpm,
        "posterior_grid_input_key": proposal_bundle.posterior_grid_input_key,
        "base_proposals": base_proposals,
        "proposal_priority": [
            *( ["base_expected"] if base_proposals in ("expected", "expected-map") else [] ),
            *( ["base_map"] if base_proposals in ("map", "expected-map") else [] ),
            f"{TOPK_PROPOSALS}_{proposal_selection}_direct_modes",
            "classical_x1_x2_x3_x4",
            "radar_peaks_1_2_3",
        ],
        "proposer_features": include_proposer_features,
        "proposer_feature_names": (
            list(PROPOSER_NODE_FEATURE_NAMES) if include_proposer_features else []
        ),
        "harmonic_ratios": list(HARMONIC_RATIOS),
        "rf_branch_policy": "raw_power_only_phase_columns_zeroed",
        "verified_svd_variant_indices": list(VERIFIED_SVD_VARIANT_INDICES),
        "svd_components": svd_components,
    }
    build_signature = _canonical_digest({"settings": settings, "inputs": input_bindings})
    if output_dir.exists():
        return _verify_reuse(output_dir, build_signature, input_bindings)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building.", dir=output_dir.parent))
    arrays: dict[str, np.memmap] = {}
    node_features: np.memmap | None = None
    feature_names: tuple[str, ...] | None = None
    metadata_parts: list[pd.DataFrame] = []
    session_records: list[dict[str, Any]] = []
    offset = 0
    try:
        arrays = _open_output_arrays(stage, len(proposer_frame))
        for session_id in sessions:
            rf_dir = rf_cache / session_id
            svd_dir = svd_cache / session_id
            rf_metadata = pd.read_csv(rf_dir / "metadata.csv")
            svd_metadata = pd.read_csv(svd_dir / "metadata.csv")
            _assert_common_rows(rf_metadata, svd_metadata, f"RF/SVD {session_id}")
            local_rows = len(svd_metadata)
            local_index = pd.to_numeric(
                svd_metadata["cache_index"], errors="raise"
            ).to_numpy(np.int64)
            if not np.array_equal(local_index, np.arange(offset, offset + local_rows)):
                raise RuntimeError(f"SVD cache_index is not contiguous for {session_id}")
            local_fold = np.asarray(
                [folds.get(str(identity), -1) for identity in svd_metadata["identity"]],
                dtype=np.int16,
            )
            if np.any(local_fold < 0):
                raise RuntimeError(f"SVD identity has no fold owner in {session_id}")
            rf_semantic = _normalized_semantic_frame(rf_metadata, local_index, local_fold)
            svd_semantic = _normalized_semantic_frame(svd_metadata, local_index, local_fold)
            if semantic_row_binding_sha256(rf_semantic) != semantic_row_binding_sha256(
                svd_semantic
            ):
                raise RuntimeError(f"RF/SVD semantic row binding mismatch: {session_id}")
            proposer_local = proposer_frame.iloc[offset : offset + local_rows]
            if semantic_row_binding_sha256(svd_semantic) != semantic_row_binding_sha256(
                proposer_local
            ):
                raise RuntimeError(f"SVD/proposer semantic row binding mismatch: {session_id}")

            rf_maps = np.load(rf_dir / "maps.npy", mmap_mode="r", allow_pickle=False)
            rf_frequency = np.load(rf_dir / "frequencies_hz.npy", allow_pickle=False)
            svd_spectra = np.load(
                svd_dir / "spectra.npy", mmap_mode="r", allow_pickle=False
            )
            svd_attributes = np.load(
                svd_dir / "attributes.npy", mmap_mode="r", allow_pickle=False
            )
            svd_frequency = np.load(
                svd_dir / "frequencies_hz.npy", allow_pickle=False
            )
            if not (
                rf_maps.shape[0]
                == svd_spectra.shape[0]
                == svd_attributes.shape[0]
                == local_rows
            ):
                raise RuntimeError(f"evidence row shape mismatch: {session_id}")
            if rf_maps.ndim != 4 or rf_maps.shape[-1] != 182:
                raise RuntimeError(f"RF branch/range layout is incompatible: {session_id}")
            raw_rf = np.asarray(rf_maps[..., :91])
            raw_radar_mask = np.isfinite(raw_rf).all(axis=(2, 3)) & np.any(
                raw_rf != 0, axis=(2, 3)
            )

            selector = slice(offset, offset + local_rows)
            bank = candidate_bank_from_metadata(
                rf_metadata,
                proposal_bpm=proposal_bundle.bpm[selector],
                proposal_confidence=proposal_bundle.confidence[selector],
                proposal_mask=proposal_bundle.mask[selector],
                proposal_source=proposal_bundle.source[selector],
                merge_radius_bpm=float(args.merge_radius_bpm),
                max_candidates=MAX_CANDIDATES,
            )
            base_entered = np.asarray(bank.source_mask)[
                ..., int(CandidateSource.BASE)
            ].any()
            if base_proposals == "none" and base_entered:
                raise RuntimeError("implicit BASE source entered the direct-mode candidate bank")
            if base_proposals != "none":
                expected_base_rows = proposal_bundle.availability[selector]
                observed_base_rows = np.asarray(bank.source_mask)[
                    ..., int(CandidateSource.BASE)
                ].any(axis=1)
                if not np.array_equal(observed_base_rows, expected_base_rows):
                    raise RuntimeError("explicit BASE proposer candidate availability mismatch")
            arrays["candidate_bpm"][selector] = bank.bpm
            arrays["candidate_mask"][selector] = bank.mask
            arrays["candidate_confidence"][selector] = bank.confidence
            arrays["candidate_source_mask"][selector] = bank.source_mask
            arrays["candidate_primary_source"][selector] = bank.primary_source

            proposer_nodes: np.ndarray | None = None
            proposer_names: tuple[str, ...] | None = None
            if include_proposer_features:
                proposer_nodes, proposer_names = proposer_candidate_node_features(
                    proposal_bundle, bank, selector
                )

            for batch in iter_compact_node_feature_batches(
                rf_maps,
                rf_frequency,
                svd_spectra,
                svd_attributes,
                svd_frequency,
                bank,
                explicit_radar_mask=raw_radar_mask,
                ratios=HARMONIC_RATIOS,
                batch_size=int(args.batch_size),
                svd_components=svd_components,
                proposer_node_features=proposer_nodes,
                proposer_feature_names=proposer_names,
                include_source_confidence=include_proposer_features,
            ):
                start = offset + int(batch.row_slice.start or 0)
                stop = offset + int(batch.row_slice.stop or local_rows)
                output_slice = slice(start, stop)
                names = tuple(batch.nodes.feature_names)
                values = np.asarray(batch.nodes.features, dtype=np.float32).copy()
                # The second RF branch is a phase-power hypothesis, not verified
                # deployment evidence.  Preserve a stable schema but make every
                # such column exactly zero so no model can learn from it.
                phase_columns = np.asarray(
                    ["_candidate_iq_phase_power_" in name for name in names], dtype=bool
                )
                values[..., phase_columns] = 0.0
                if not np.isfinite(values).all():
                    raise RuntimeError("node feature construction produced non-finite values")
                if node_features is None:
                    feature_names = names
                    node_features = np.lib.format.open_memmap(
                        stage / ARRAY_FILES["node_features"],
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(proposer_frame), MAX_CANDIDATES, len(names)),
                    )
                elif names != feature_names:
                    raise RuntimeError("node feature schema changed between sessions")
                node_features[output_slice] = values
                joint = np.asarray(batch.rf_support.radar_mask, dtype=bool) & np.asarray(
                    batch.svd_support.radar_mask, dtype=bool
                )
                arrays["joint_radar_mask"][output_slice] = joint
                arrays["rf_support_count"][output_slice] = np.asarray(
                    batch.rf_support.mask, dtype=np.uint8
                ).sum(axis=2)
                arrays["svd_support_count"][output_slice] = np.asarray(
                    batch.svd_support.mask, dtype=np.uint8
                ).sum(axis=2)

            canonical_metadata = rf_metadata.copy()
            canonical_metadata.insert(0, "cache_index", local_index)
            canonical_metadata.insert(1, "fold", local_fold)
            metadata_parts.append(canonical_metadata)
            session_records.append(
                {
                    "session_id": session_id,
                    "row_start": offset,
                    "row_stop_exclusive": offset + local_rows,
                    "rows": local_rows,
                    "identity": str(rf_metadata["identity"].iloc[0]),
                    "fold": int(local_fold[0]),
                    "rf_frequency_grid": {
                        "count": len(rf_frequency),
                        "minimum_hz": float(rf_frequency[0]),
                        "maximum_hz": float(rf_frequency[-1]),
                        "sha256": sha256_file(rf_dir / "frequencies_hz.npy"),
                    },
                    "svd_frequency_grid": {
                        "count": len(svd_frequency),
                        "minimum_hz": float(svd_frequency[0]),
                        "maximum_hz": float(svd_frequency[-1]),
                        "sha256": sha256_file(svd_dir / "frequencies_hz.npy"),
                    },
                }
            )
            offset += local_rows

        if offset != len(proposer_frame) or node_features is None or feature_names is None:
            raise RuntimeError("session construction did not exactly cover proposer rows")
        metadata = pd.concat(metadata_parts, ignore_index=True)
        if not np.array_equal(metadata["cache_index"].to_numpy(np.int64), cache_index):
            raise RuntimeError("constructed metadata cache_index exact cover failed")
        lineage = semantic_row_binding_sha256(metadata.loc[:, list(SEMANTIC_ROW_FIELDS)])
        if lineage != semantic_row_binding_sha256(proposer_frame):
            raise RuntimeError("final metadata/proposer row lineage mismatch")
        metadata.to_csv(stage / "metadata.csv", index=False)
        _write_json(
            stage / "feature_names.json",
            {
                "node_feature_names": list(feature_names),
                "candidate_source_names": list(CANDIDATE_SOURCE_NAMES),
                "forward_arrays": [
                    "node_features",
                    "candidate_bpm",
                    "candidate_mask",
                    "candidate_confidence",
                    "candidate_source_mask",
                    "joint_radar_mask",
                ],
                "forbidden_target_qc_forward_fields": [
                    *sorted(FORBIDDEN_TARGET_QC_FIELDS),
                ],
            },
        )
        node_features.flush()
        _flush(arrays)
        outputs = _output_bindings(stage)
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "complete": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "build_signature_sha256": build_signature,
            "row_count": len(metadata),
            "session_count": len(sessions),
            "identity_count": int(metadata["identity"].nunique()),
            "fold_count": int(metadata["fold"].nunique()),
            "node_feature_shape": list(node_features.shape),
            "node_feature_dtype": "float32",
            "row_lineage_sha256": lineage,
            "settings": settings,
            "candidate_policy": {
                "maximum_candidates": MAX_CANDIDATES,
                "priority": settings["proposal_priority"],
                "merge_radius_bpm": float(args.merge_radius_bpm),
                "merge_anchor_policy": "first_source_anchor_never_moves_then_stable_bpm_sort",
                "proposal_selection": proposal_selection,
                "posterior_nms_suppression_bpm": suppression_bpm,
                "base_source_policy": (
                    "explicit_expected_then_map_before_direct_modes"
                    if base_proposals == "expected-map"
                    else f"explicit_{base_proposals}"
                    if base_proposals != "none"
                    else "none_direct_modes_never_implicit_base"
                ),
                "unavailable_proposer_policy": "no_base_or_direct_candidate_and_all_proposer_features_zero",
            },
            "evidence_policy": {
                "harmonic_ratios": list(HARMONIC_RATIOS),
                "native_frequency_grid_sampling": True,
                "out_of_band_policy": "exact_zero_and_false_mask_never_edge_clamp",
                "rf_branch_policy": "raw_power_only_phase_feature_columns_exact_zero",
                "rf_range_policy": "preserve_91_raw_range_indices_before_compaction",
                "svd_variant_indices": list(VERIFIED_SVD_VARIANT_INDICES),
                "svd_variant_names": list(VERIFIED_SVD_VARIANT_NAMES),
                "svd_component_policy": (
                    f"first_{svd_components}_components_preserved_before_fixed_width_"
                    "reliability_compaction"
                ),
                "svd_components": svd_components,
                "proposer_posterior_feature_policy": (
                    "full_posterior_candidate_local_summaries_plus_exact_row_diagnostics"
                    if include_proposer_features
                    else "disabled_backward_compatible_i1_schema"
                ),
            },
            "model_boundary": {
                "target_qc_excluded_from_candidate_and_feature_construction": True,
                "metadata_is_lineage_and_training_target_storage_not_a_forward_input": True,
                "identity_session_protocol_fold_excluded_from_forward_features": True,
                "proposal_reference_fields_ignored": True,
            },
            "inputs": input_bindings,
            "sessions": session_records,
            "outputs": outputs,
        }
        manifest["content_sha256"] = _canonical_digest(manifest)
        _write_json(stage / "manifest.json", manifest)
        if output_dir.exists():
            raise RuntimeError("output appeared concurrently; refusing to overwrite")
        stage.replace(output_dir)
        return {"status": "built", "output_dir": str(output_dir), "manifest": manifest}
    except BaseException:
        for array in arrays.values():
            try:
                array.flush()
            except Exception:
                pass
        if node_features is not None:
            try:
                node_features.flush()
            except Exception:
                pass
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-cache", type=Path, default=DEFAULT_RF_CACHE)
    parser.add_argument("--svd-cache", type=Path, default=DEFAULT_SVD_CACHE)
    parser.add_argument("--proposer", type=Path, default=DEFAULT_PROPOSER)
    parser.add_argument("--fold-assignments", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--merge-radius-bpm", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--proposal-selection",
        choices=("topk", "posterior-nms"),
        default="topk",
        help="Use stored top-k modes (i1 compatibility) or stable full-posterior NMS.",
    )
    parser.add_argument(
        "--posterior-nms-suppression-bpm",
        type=float,
        default=1.25,
        help="Inclusive RR separation suppressed after each stable posterior mode.",
    )
    parser.add_argument(
        "--base-proposals",
        choices=BASE_PROPOSAL_CHOICES,
        default="none",
        help="Optional full-posterior expected/MAP anchors, explicitly marked BASE.",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        choices=(6, 12),
        default=6,
        help="Retain 6 (i1 compatibility) or all 12 cached SVD components.",
    )
    parser.add_argument(
        "--proposer-features",
        action="store_true",
        help="Append validated full-posterior/node-local proposer diagnostics.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = build(parse_args(argv))
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": result["output_dir"],
                "rows": result["manifest"]["row_count"],
                "build_signature_sha256": result["manifest"]["build_signature_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
