#!/usr/bin/env python3
"""Train and lock the retrospective harmonic candidate-set episode SNN.

The script enforces the leakage boundary which matters for this experiment:
features and their robust scaler are fitted with outer-training identities,
model selection and the fallback policy use only the next validation fold, and
the outer-test episode iterator is not created until ``selection_lock.json``
has been durably written.  The cache metadata contains targets for training
bookkeeping, but only arrays named by the cache's forward manifest are passed
to :class:`HarmonicCandidateSetEpisodeSNN`.
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
import random
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_rr.harmonic_set_models import (  # noqa: E402
    HarmonicCandidateSetEpisodeSNN,
    HarmonicSetState,
)


SCHEMA_VERSION = 1
N_FOLDS = 6
CHUNK_SIZE = 32
WARMUP_WINDOWS = 8
DEFAULT_CACHE = PROJECT_ROOT / "artifacts/cache/harmonic_set_v2"
DEFAULT_FALLBACK = PROJECT_ROOT / "artifacts/runs/ensemble_structured_exact/ensemble_oof.csv"
PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {"hidden_channels": 8, "attention_heads": 1, "dropout": 0.0},
    "compact": {"hidden_channels": 32, "attention_heads": 4, "dropout": 0.05},
    "default": {"hidden_channels": 64, "attention_heads": 4, "dropout": 0.05},
    # Optional capacity probe for the causal anchor residual.  It is never the
    # default and is fully recorded in model_config/checkpoint/lock hashes.
    "large": {"hidden_channels": 96, "attention_heads": 4, "dropout": 0.05},
}

# Frozen Cartesian policy grid: 19 * 9 * 7 * 4 * 4 = 19,152 policies.
PROBABILITY_THRESHOLDS = (
    0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.925, 0.95, 0.965, 0.975, 0.985, 0.995, 1.10,
)
MARGIN_THRESHOLDS = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.10)
ENTROPY_THRESHOLDS = (0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 10.0)
QUALITY_THRESHOLDS = (0.00, 0.25, 0.50, 0.75)
CORRECTION_PULLS = (0.25, 0.50, 0.75, 1.00)

# The confidence dimensions below are deliberately a small set of *joint*
# profiles rather than another Cartesian product.  This keeps the frozen
# selector tractable (5 * 19,152 candidates) while allowing the policy to use
# all of the deployment-time uncertainty/context signals identified by the
# discovery audit.  The first profile is the backwards-compatible permissive
# profile and guarantees that structural source/base fallbacks are unchanged.
# (base_std_max, source_scale_max, disagreement_min, disagreement_max,
#  minimum_valid_candidates)
POLICY_CONTEXT_PROFILES = (
    (math.inf, math.inf, 0.00, math.inf, 1),
    (4.00, 2.00, 0.25, 6.00, 2),
    (2.00, 1.50, 0.25, 4.00, 2),
    (1.50, 1.00, 0.50, 3.00, 3),
    (2.00, 1.50, 0.00, 2.00, 2),
)

# Frozen commercial-target selection gates.  They prioritize satisfying the
# entire validation contract over a small improvement in one average metric.
# This is retrospective model selection, never an external commercial claim.
COMMERCIAL_SOURCE_GATES: dict[str, tuple[str, float]] = {
    "mae": ("maximum", 1.0),
    "identity_macro_mae": ("maximum", 1.0),
    "rmse": ("maximum", 1.8),
    "within_2": ("minimum", 0.90),
    "catastrophic_over_5": ("maximum", 0.03),
    "tail_25_35_mae": ("maximum", 2.0),
}
COMMERCIAL_SELECTION_OBJECTIVE = (
    "commercial_gate_v1:(fail_count,max_normalized_violation,"
    "sum_normalized_violation,identity_macro_mae,mae)"
)


@dataclass(frozen=True, slots=True)
class IterationObjective:
    """Resolved, immutable loss/scheduling contract for one family iteration."""

    iteration: int
    campaign_id: str
    listwise_temperature_bpm: float
    listwise_weight: float
    nearest_ce_weight: float
    selected_residual_weight: float
    all_candidate_residual_weight: float
    mixture_nll_weight: float
    factor_weight: float
    quality_weight: float
    spike_weight: float
    warmup_windows: int
    gradient_accumulation_sessions: int
    tail_weight: float
    cvar_weight: float
    anchor_residual_weight: float
    anchor_nll_weight: float
    anchor_gate_weight: float


def resolve_iteration_objective(
    iteration: int,
    *,
    warmup_windows: int | None = None,
    gradient_accumulation_sessions: int | None = None,
    tail_weight: float | None = None,
    cvar_weight: float | None = None,
    anchor_enabled: bool | None = None,
    anchor_residual_weight: float | None = None,
    anchor_nll_weight: float | None = None,
    anchor_gate_weight: float | None = None,
) -> IterationObjective:
    """Bind an iteration number to a materially different training contract.

    Iteration 1 retains the historical warmup/scheduling defaults.  Iteration
    2 sharpens exact/NMS responsibilities and enables identity-diverse
    accumulation.  Iteration 3 keeps that objective and additionally turns on
    tail/CVaR weighting by default.  Explicit CLI values remain authoritative.
    """

    if int(iteration) not in (1, 2, 3):
        raise ValueError("adaptive iteration must be 1, 2, or 3")
    iteration = int(iteration)
    default_warmup = 8 if iteration == 1 else 2
    default_accumulation = 1 if iteration == 1 else 4
    use_anchor = (iteration == 3) if anchor_enabled is None else bool(anchor_enabled)
    if use_anchor and iteration != 3:
        raise ValueError("the posterior-anchor residual architecture is i3-only")
    default_tail = 2.0 if iteration == 3 else 0.0
    default_cvar = 0.15 if iteration == 3 else 0.0
    default_anchor_residual = 0.75 if use_anchor else 0.0
    default_anchor_nll = 0.20 if use_anchor else 0.0
    default_anchor_gate = 0.08 if use_anchor else 0.0
    warmup = default_warmup if warmup_windows is None else int(warmup_windows)
    accumulation = (
        default_accumulation
        if gradient_accumulation_sessions is None
        else int(gradient_accumulation_sessions)
    )
    tail = default_tail if tail_weight is None else float(tail_weight)
    cvar = default_cvar if cvar_weight is None else float(cvar_weight)
    anchor_residual = (
        default_anchor_residual
        if anchor_residual_weight is None
        else float(anchor_residual_weight)
    )
    anchor_nll = (
        default_anchor_nll if anchor_nll_weight is None else float(anchor_nll_weight)
    )
    anchor_gate = (
        default_anchor_gate
        if anchor_gate_weight is None
        else float(anchor_gate_weight)
    )
    if (
        warmup < 0 or accumulation < 1 or tail < 0 or cvar < 0
        or anchor_residual < 0 or anchor_nll < 0 or anchor_gate < 0
    ):
        raise ValueError("warmup/accumulation/tail/CVaR/anchor settings are invalid")
    if not use_anchor and any(
        value > 0 for value in (anchor_residual, anchor_nll, anchor_gate)
    ):
        raise ValueError("anchor loss weights require the i3 anchor architecture")
    if iteration == 1:
        return IterationObjective(
            iteration, "v2_i1_candidate_graph", 1.25,
            1.0, 0.50, 0.30, 0.15, 0.10, 0.12, 0.08, 5.0e-3,
            warmup, accumulation, tail, cvar, 0.0, 0.0, 0.0,
        )
    campaign_id = (
        "v2_i2_harmonic_evidence"
        if iteration == 2
        else (
            "v2_i3_causal_posterior_anchor"
            if use_anchor
            else "v2_i3_tail_risk"
        )
    )
    return IterationObjective(
        iteration, campaign_id, 0.50,
        1.0, 0.20, 0.20, 0.35, 0.20, 0.10, 0.12, 5.0e-3,
        warmup, accumulation, tail, cvar,
        anchor_residual, anchor_nll, anchor_gate,
    )


def iteration_objective_record(objective: IterationObjective) -> dict[str, Any]:
    """Serialize without changing the frozen i1/i2 provenance schema."""

    record = asdict(objective)
    if objective.iteration in (1, 2):
        for key in (
            "anchor_residual_weight", "anchor_nll_weight", "anchor_gate_weight"
        ):
            if float(record[key]) != 0.0:
                raise ValueError("i1/i2 cannot serialize nonzero anchor loss weights")
            record.pop(key)
    return record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def sha256_json(value: Any) -> str:
    """Hash a canonical, finite JSON rendering of an effective config."""

    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_ready(value), indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        path.chmod(0o444)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_recovery_provenance(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Write a self-hashed immutable recovery attestation sidecar."""

    path = output_dir / "recovery_provenance.json"
    if path.exists():
        raise RuntimeError("recovery provenance already exists")
    normalized = _json_ready(dict(payload))
    document = {
        "schema_version": SCHEMA_VERSION,
        "signature_algorithm": "sha256-canonical-json",
        "payload_sha256": sha256_json(normalized),
        "payload": normalized,
    }
    atomic_write_json(path, document)
    path.chmod(0o444)
    return path


@dataclass(frozen=True, slots=True)
class RobustNodeScaler:
    center: np.ndarray
    scale: np.ndarray
    fit_positions_sha256: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = (np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) - self.center) / self.scale
        return np.clip(transformed, -8.0, 8.0).astype(np.float32, copy=False)

    def record(self) -> dict[str, Any]:
        return {
            "center": self.center.reshape(-1).tolist(),
            "scale": self.scale.reshape(-1).tolist(),
            "fit_positions_sha256": self.fit_positions_sha256,
            "fit_scope": "candidate-masked nodes from outer-training identities only",
            "method": "median and max(IQR/1.349, 1e-4)",
        }


@dataclass(slots=True)
class Experiment:
    root: Path
    metadata: pd.DataFrame
    node_features: np.ndarray
    candidate_rr: np.ndarray
    candidate_mask: np.ndarray
    radar_mask: np.ndarray
    base_prediction: np.ndarray
    base_std: np.ndarray
    base_available: np.ndarray
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FallbackPolicy:
    probability_threshold: float
    margin_threshold: float
    entropy_threshold: float
    quality_threshold: float
    correction_pull: float
    validation_coverage: float
    validation_precision: float
    validation_fpr: float
    validation_macro_mae: float
    safeguards: dict[str, bool]
    grid_cardinality: int
    base_std_max: float = math.inf
    source_scale_max: float = math.inf
    disagreement_min: float = 0.0
    disagreement_max: float = math.inf
    minimum_valid_candidates: int = 1
    validation_recall: float = 1.0
    selection_status: str = "promoted"
    promotion_eligible: bool = True
    selection_objective: str = "legacy_macro_mae"


@dataclass(slots=True)
class Predictions:
    position: np.ndarray
    cache_index: np.ndarray
    target: np.ndarray
    identity: np.ndarray
    base_prediction: np.ndarray
    base_std: np.ndarray
    base_available: np.ndarray
    source_prediction: np.ndarray
    source_scale: np.ndarray
    source_available: np.ndarray
    selected_index: np.ndarray
    selected_probability: np.ndarray
    margin: np.ndarray
    entropy: np.ndarray
    quality: np.ndarray
    spike_rate: np.ndarray
    final_prediction: np.ndarray | None = None
    applied_pull: np.ndarray | None = None
    normalized_entropy: np.ndarray | None = None
    valid_candidate_count: np.ndarray | None = None
    raw_anchor_prediction: np.ndarray | None = None
    raw_anchor_std: np.ndarray | None = None
    corrected_anchor_prediction: np.ndarray | None = None
    anchor_residual: np.ndarray | None = None
    anchor_snap_gate: np.ndarray | None = None
    candidate_source_prediction: np.ndarray | None = None


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if not document.get("complete") or int(document.get("format_version", -1)) != 1:
        raise RuntimeError("harmonic cache manifest is incomplete or incompatible")
    return document


def _choose_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame), None)


def load_experiment(cache_root: Path, fallback_csv: Path) -> Experiment:
    root = cache_root.expanduser().resolve()
    manifest = _load_manifest(root)
    metadata = pd.read_csv(root / "metadata.csv")
    required_metadata = {"cache_index", "fold", "session_id", "identity", "window_number", "rr_bpm", "reference_valid", "classical_rr_bpm"}
    missing = sorted(required_metadata - set(metadata))
    if missing:
        raise RuntimeError(f"cache metadata lacks required fields: {missing}")
    cache_index = metadata["cache_index"].to_numpy(np.int64)
    if not np.array_equal(cache_index, np.arange(len(metadata), dtype=np.int64)):
        raise RuntimeError("cache_index must be a unique contiguous row binding")

    node_features = np.load(root / "node_features.npy", mmap_mode="r", allow_pickle=False)
    candidate_rr = np.load(root / "candidate_bpm.npy", mmap_mode="r", allow_pickle=False)
    candidate_mask = np.load(root / "candidate_mask.npy", mmap_mode="r", allow_pickle=False)
    radar_mask = np.load(root / "joint_radar_mask.npy", mmap_mode="r", allow_pickle=False)
    if not (node_features.shape[:2] == candidate_rr.shape == candidate_mask.shape):
        raise RuntimeError("cache forward-array candidate shapes disagree")
    if node_features.shape[0] != len(metadata) or radar_mask.shape[0] != len(metadata):
        raise RuntimeError("cache forward-array rows disagree with metadata")

    fallback = pd.read_csv(fallback_csv)
    if "cache_index" not in fallback or fallback["cache_index"].duplicated().any():
        raise RuntimeError("fallback OOF must have unique cache_index")
    prediction_column = _choose_column(fallback, ("prediction_bpm", "prediction_locked_final_bpm", "prediction_candidate_bpm"))
    std_column = _choose_column(fallback, ("rr_std_bpm", "candidate_rr_std_bpm", "source_std_bpm"))
    if prediction_column is None:
        raise RuntimeError("fallback OOF has no supported prediction column")
    positions = {int(value): index for index, value in enumerate(cache_index)}
    base_prediction = np.zeros(len(metadata), dtype=np.float32)
    base_std = np.full(len(metadata), 4.0, dtype=np.float32)
    base_available = np.zeros(len(metadata), dtype=bool)
    for row in fallback.itertuples(index=False):
        index = positions.get(int(getattr(row, "cache_index")))
        if index is None:
            raise RuntimeError("fallback OOF contains a cache_index absent from cache")
        prediction = float(getattr(row, prediction_column))
        standard_deviation = float(getattr(row, std_column)) if std_column else 2.0
        if np.isfinite(prediction) and np.isfinite(standard_deviation) and standard_deviation > 0:
            base_prediction[index] = prediction
            base_std[index] = standard_deviation
            base_available[index] = True
    return Experiment(
        root=root, metadata=metadata, node_features=node_features,
        candidate_rr=candidate_rr, candidate_mask=candidate_mask,
        radar_mask=radar_mask, base_prediction=base_prediction,
        base_std=base_std, base_available=base_available, manifest=manifest,
    )


def _positions_sha256(positions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(positions, dtype=np.int64).view(np.uint8)).hexdigest()


def fit_robust_scaler(experiment: Experiment, train_positions: np.ndarray) -> RobustNodeScaler:
    position = np.asarray(train_positions, dtype=np.int64)
    mask = np.asarray(experiment.candidate_mask[position], dtype=bool)
    values = np.asarray(experiment.node_features[position], dtype=np.float32)[mask]
    if values.ndim != 2 or len(values) == 0:
        raise RuntimeError("outer-training identities contain no candidate nodes")
    center = np.nanmedian(values, axis=0).astype(np.float32)
    q25, q75 = np.nanpercentile(values, [25.0, 75.0], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1.0e-4).astype(np.float32)
    center = np.nan_to_num(center, nan=0.0)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    return RobustNodeScaler(center.reshape(1, 1, -1), scale.reshape(1, 1, -1), _positions_sha256(position))


def identity_balanced_weights(metadata: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    """Each identity contributes equal total weight over its valid references."""

    positions = np.asarray(positions, dtype=np.int64)
    weights = np.zeros(len(metadata), dtype=np.float32)
    valid = metadata["reference_valid"].astype(bool).to_numpy()
    identities = metadata["identity"].astype(str).to_numpy()
    selected_identities = sorted(set(identities[positions]))
    for identity in selected_identities:
        owned = positions[(identities[positions] == identity) & valid[positions]]
        if len(owned):
            weights[owned] = 1.0 / float(len(owned))
    positive = weights[positions] > 0
    if positive.any():
        weights[positions[positive]] *= float(positive.sum()) / float(weights[positions[positive]].sum())
    return weights


def split_positions(metadata: pd.DataFrame, outer_fold: int) -> tuple[np.ndarray, np.ndarray, int]:
    if not 0 <= outer_fold < N_FOLDS:
        raise ValueError("fold must be in [0, 5]")
    validation_fold = (outer_fold + 1) % N_FOLDS
    folds = metadata["fold"].to_numpy(np.int64)
    train = np.flatnonzero((folds != outer_fold) & (folds != validation_fold))
    validation = np.flatnonzero(folds == validation_fold)
    if not len(train) or not len(validation):
        raise RuntimeError("empty outer-training or validation split")
    return train, validation, validation_fold


def iter_session_positions(metadata: pd.DataFrame, positions: np.ndarray, *, shuffle: bool, rng: np.random.Generator) -> Iterator[np.ndarray]:
    selected = metadata.iloc[np.asarray(positions, dtype=np.int64)].copy()
    groups = [
        group.sort_values("window_number", kind="stable").index.to_numpy(np.int64)
        for _, group in selected.groupby("session_id", sort=True)
    ]
    if shuffle:
        rng.shuffle(groups)
    yield from groups


def iter_identity_balanced_session_positions(
    metadata: pd.DataFrame,
    positions: np.ndarray,
    *,
    shuffle: bool,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    """Yield chronological sessions in identity-round-robin order.

    Session state is never shared between physical sessions.  Shuffling occurs
    only at the identity and whole-session levels; window order stays causal.
    Round-robin scheduling prevents a gradient-accumulation group from being
    dominated by an identity merely because it owns more sessions.
    """

    selected = metadata.iloc[np.asarray(positions, dtype=np.int64)].copy()
    by_identity: dict[str, list[np.ndarray]] = {}
    for (identity, _session_id), group in selected.groupby(
        ["identity", "session_id"], sort=True
    ):
        by_identity.setdefault(str(identity), []).append(
            group.sort_values("window_number", kind="stable").index.to_numpy(np.int64)
        )
    identities = sorted(by_identity)
    if shuffle:
        rng.shuffle(identities)
        for sessions in by_identity.values():
            rng.shuffle(sessions)
    queues = {identity: list(sessions) for identity, sessions in by_identity.items()}
    while any(queues.values()):
        for identity in identities:
            if queues[identity]:
                yield queues[identity].pop(0)


def detach_state(state: HarmonicSetState) -> HarmonicSetState:
    return tuple((membrane.detach(), adaptation.detach()) for membrane, adaptation in state)  # type: ignore[return-value]


def valid_label_loss_mask(sequence_mask: Tensor, reference_valid: Tensor, target: Tensor, warmup_mask: Tensor | None = None) -> Tensor:
    mask = sequence_mask.bool() & reference_valid.bool() & torch.isfinite(target)
    if warmup_mask is not None:
        mask &= ~warmup_mask.bool()
    return mask


def _batch_for_positions(
    experiment: Experiment,
    scaler: RobustNodeScaler,
    position: np.ndarray,
    device: torch.device,
    *,
    warmup_offset: int,
    warmup_windows: int = WARMUP_WINDOWS,
) -> dict[str, Tensor]:
    position = np.asarray(position, dtype=np.int64).copy()
    node = scaler.transform(np.asarray(experiment.node_features[position], dtype=np.float32))
    candidate = np.asarray(experiment.candidate_rr[position], dtype=np.float32).copy()
    candidate_mask = np.asarray(experiment.candidate_mask[position], dtype=bool).copy()
    radar = np.asarray(experiment.radar_mask[position], dtype=bool).copy()
    metadata = experiment.metadata.iloc[position]
    length = len(position)
    return {
        "position": torch.as_tensor(position[None], device=device),
        "node_features": torch.as_tensor(node[None], device=device),
        "candidate_rr": torch.as_tensor(candidate[None], device=device),
        "candidate_mask": torch.as_tensor(candidate_mask[None], device=device),
        "radar_mask": torch.as_tensor(radar[None], device=device),
        "sequence_mask": torch.ones((1, length), device=device, dtype=torch.bool),
        "reset_mask": torch.as_tensor([[warmup_offset == 0] + [False] * (length - 1)], device=device),
        "target": torch.as_tensor(metadata["rr_bpm"].to_numpy(np.float32, copy=True)[None], device=device),
        "reference_valid": torch.as_tensor(metadata["reference_valid"].astype(bool).to_numpy(copy=True)[None], device=device),
        "classical_rr": torch.as_tensor(metadata["classical_rr_bpm"].to_numpy(np.float32, copy=True)[None], device=device),
        # This cache-index-bound strict nested posterior remains a post-forward
        # fallback in i1/i2.  In i3 it is also the explicit label-free posterior
        # anchor supplied to the causal residual architecture.  It is never a
        # reference/QC value and its raw value remains the policy baseline.
        "base_prediction": torch.as_tensor(experiment.base_prediction[position].copy()[None], device=device),
        "base_std": torch.as_tensor(experiment.base_std[position].copy()[None], device=device),
        "base_available": torch.as_tensor(experiment.base_available[position].copy()[None], device=device),
        "warmup_mask": torch.as_tensor(
            (np.arange(warmup_offset, warmup_offset + length) < int(warmup_windows))[None],
            device=device,
        ),
    }


def forward_source_model(
    model: HarmonicCandidateSetEpisodeSNN,
    batch: Mapping[str, Tensor],
    *,
    state: HarmonicSetState | None,
) -> dict[str, Tensor | HarmonicSetState]:
    """Apply the iteration-bound, label-free model forward allowlist.

    The branch is on immutable model configuration, never on labels or split
    metadata.  i1/i2 retain their historical call exactly.  i3 binds the raw
    nested posterior mean/std/availability as the only additional context.
    """

    keyword: dict[str, Any] = {
        "radar_mask": batch["radar_mask"],
        "state": state,
        "reset_mask": batch["reset_mask"],
    }
    if model.anchor_enabled:
        keyword.update(
            anchor_rr=batch["base_prediction"],
            anchor_std=batch["base_std"],
            anchor_available=batch["base_available"],
        )
    return model(
        batch["node_features"],
        batch["candidate_rr"],
        batch["candidate_mask"],
        batch["sequence_mask"],
        **keyword,
    )


def gaussian_candidate_mixture_nll(
    logits: Tensor,
    candidate_rr: Tensor,
    candidate_residual: Tensor,
    candidate_scale: Tensor,
    candidate_mask: Tensor,
    target: Tensor,
    *,
    rr_min: float = 6.0,
    rr_max: float = 45.0,
) -> Tensor:
    """Per-row Gaussian-mixture NLL over every available candidate.

    This is intentionally different from applying a Gaussian loss only to the
    hard deployment winner.  ``logsumexp(log p_k + log N_k)`` sends useful
    gradients to the logits, residual means, and scales of non-argmax
    candidates whenever they assign density to the reference.
    """

    if not (
        logits.shape == candidate_rr.shape == candidate_residual.shape
        == candidate_scale.shape == candidate_mask.shape
    ):
        raise ValueError("candidate mixture tensors must have the same [B,T,K] shape")
    if target.shape != logits.shape[:-1]:
        raise ValueError("mixture target must have shape [B,T]")
    mask = candidate_mask.bool()
    safe_logits = logits.float().masked_fill(~mask, -1.0e4)
    log_probability = F.log_softmax(safe_logits, dim=-1).masked_fill(~mask, -1.0e4)
    mean = (candidate_rr.float() + candidate_residual.float()).clamp(
        min=float(rr_min), max=float(rr_max)
    )
    scale = candidate_scale.float().clamp(min=0.05, max=20.0)
    standardized = (target.float().unsqueeze(-1) - mean) / scale
    gaussian_log_density = (
        -0.5 * standardized.square()
        - scale.log()
        - 0.5 * math.log(2.0 * math.pi)
    ).masked_fill(~mask, -1.0e4)
    return -torch.logsumexp(log_probability + gaussian_log_density, dim=-1)


def exact_candidate_responsibilities(
    candidate_rr: Tensor,
    candidate_mask: Tensor,
    target: Tensor,
    *,
    temperature_bpm: float,
) -> Tensor:
    """Tight target responsibilities suitable for exact/NMS candidate banks."""

    if temperature_bpm <= 0:
        raise ValueError("responsibility temperature must be positive")
    error = (candidate_rr.float() - target.float().unsqueeze(-1)).abs()
    mask = candidate_mask.bool()
    score = (-error / float(temperature_bpm)).masked_fill(~mask, -1.0e4)
    responsibility = F.softmax(score, dim=-1) * mask.float()
    return responsibility / responsibility.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def all_candidate_residual_loss(
    candidate_rr: Tensor,
    candidate_residual: Tensor,
    candidate_mask: Tensor,
    target: Tensor,
    *,
    maximum_residual_bpm: float = 0.75,
) -> tuple[Tensor, Tensor]:
    """Supervise all candidates whose bounded residual can reach the target."""

    target_residual = (target.unsqueeze(-1) - candidate_rr).clamp(
        min=-float(maximum_residual_bpm), max=float(maximum_residual_bpm)
    )
    reachable = (
        candidate_mask.bool()
        & torch.isfinite(target.unsqueeze(-1))
        & ((target.unsqueeze(-1) - candidate_rr).abs() <= float(maximum_residual_bpm) + 1.0e-6)
    )
    per_candidate = F.smooth_l1_loss(
        candidate_residual, target_residual, beta=0.25, reduction="none"
    )
    count = reachable.float().sum(dim=-1)
    per_row = (per_candidate * reachable.float()).sum(dim=-1) / count.clamp_min(1.0)
    return per_row, count > 0


def quality_supervision_target(
    source_rr: Tensor,
    target: Tensor,
    *,
    base_prediction: Tensor | None,
    base_available: Tensor | None,
    correctness_bpm: float = 2.0,
    improvement_margin_bpm: float = 0.05,
) -> Tensor:
    """Correctness/improvement label for the actually selected source.

    The target never asks whether *some* bank candidate could be correct.  It
    describes the hard source that deployment would emit, and (when present)
    its improvement over the strict fallback.
    """

    source_error = (source_rr.detach() - target).abs()
    source_correct = source_error <= float(correctness_bpm)
    if base_prediction is None or base_available is None:
        return source_correct.float()
    available = base_available.bool()
    base_error = (base_prediction - target).abs()
    improves = available & (
        source_error + float(improvement_margin_bpm) < base_error
    )
    return (source_correct | improves).float()


def compute_multitask_loss(
    output: Mapping[str, Tensor | HarmonicSetState], batch: Mapping[str, Tensor],
    row_weights: Tensor, *, adaptive_iteration: int = 1,
    tail_weight: float | None = 0.0, cvar_weight: float | None = 0.0,
    objective: IterationObjective | None = None,
    normalization_denominator: Tensor | float | None = None,
    regularization_fraction: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    objective = objective or resolve_iteration_objective(
        adaptive_iteration, tail_weight=tail_weight, cvar_weight=cvar_weight
    )
    logits = output["candidate_logits"]
    assert isinstance(logits, Tensor)
    candidate_rr = batch["candidate_rr"]
    candidate_mask = batch["candidate_mask"].bool()
    target = batch["target"]
    label_mask = valid_label_loss_mask(batch["sequence_mask"], batch["reference_valid"], target, batch["warmup_mask"])
    source_available = output["source_available"]
    assert isinstance(source_available, Tensor)
    label_mask &= source_available.bool()
    positions = batch["position"]
    weight = row_weights[positions].float() * label_mask.float()
    if objective.tail_weight > 0:
        tail = ((target >= 25.0) & (target <= 35.0)).float()
        weight = weight * (1.0 + float(objective.tail_weight) * tail)
    if normalization_denominator is None:
        denominator = weight.sum().clamp_min(1.0e-8)
    else:
        denominator = torch.as_tensor(
            normalization_denominator, device=weight.device, dtype=weight.dtype
        ).clamp_min(1.0e-8)
    zero = logits.sum() * 0.0
    if not label_mask.any():
        components = {name: zero for name in (
            "listwise", "nearest_ce", "residual", "candidate_residual",
            "nll", "factor", "quality", "spike", "cvar",
            "anchor_residual", "anchor_nll", "anchor_gate",
        )}
        return zero, components

    candidate_source_available = output.get("candidate_source_available")
    if isinstance(candidate_source_available, Tensor):
        candidate_weight = weight * candidate_source_available.bool().float()
    else:
        # i1/i2 source availability is exactly candidate availability.
        candidate_weight = weight

    errors = (candidate_rr - target.unsqueeze(-1)).abs().masked_fill(~candidate_mask, 1.0e4)
    soft = exact_candidate_responsibilities(
        candidate_rr, candidate_mask, target,
        temperature_bpm=objective.listwise_temperature_bpm,
    ).detach()
    listwise_per = -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    nearest = errors.argmin(dim=-1)
    nearest_ce_per = F.cross_entropy(logits.transpose(1, 2), nearest, reduction="none")
    source_rr = output["source_rr"]
    source_scale = output["source_scale_bpm"]
    assert isinstance(source_rr, Tensor) and isinstance(source_scale, Tensor)
    residual_per = F.smooth_l1_loss(source_rr, target, beta=1.0, reduction="none")
    candidate_residual = output["candidate_residual_bpm"]
    candidate_scale = output["candidate_scale_bpm"]
    assert isinstance(candidate_residual, Tensor) and isinstance(candidate_scale, Tensor)
    candidate_residual_per, residual_reachable = all_candidate_residual_loss(
        candidate_rr, candidate_residual, candidate_mask, target
    )
    candidate_residual_per = candidate_residual_per * residual_reachable.float()
    nll_per = gaussian_candidate_mixture_nll(
        logits, candidate_rr, candidate_residual, candidate_scale,
        candidate_mask, target,
    )
    factors = torch.arange(1, 5, device=target.device, dtype=target.dtype)
    factor_target = (batch["classical_rr"].unsqueeze(-1) * factors - target.unsqueeze(-1)).abs().argmin(dim=-1)
    factor_logits = output["factor_logits"]
    quality_logit = output["quality_logit"]
    assert isinstance(factor_logits, Tensor) and isinstance(quality_logit, Tensor)
    factor_per = F.cross_entropy(factor_logits.transpose(1, 2), factor_target, reduction="none")
    quality_target = quality_supervision_target(
        source_rr,
        target,
        base_prediction=batch.get("base_prediction"),
        base_available=batch.get("base_available"),
    )
    quality_per = F.binary_cross_entropy_with_logits(
        quality_logit, quality_target, reduction="none"
    )

    def weighted(values: Tensor, weights: Tensor = weight) -> Tensor:
        return (values * weights).sum() / denominator

    listwise = weighted(listwise_per, candidate_weight)
    nearest_ce = weighted(nearest_ce_per, candidate_weight)
    residual = weighted(residual_per)
    candidate_residual_loss = weighted(candidate_residual_per, candidate_weight)
    nll = weighted(nll_per, candidate_weight)
    factor = weighted(factor_per, candidate_weight)
    quality = weighted(quality_per)
    spike_rates = output["spike_rates"]
    assert isinstance(spike_rates, Tensor)
    spike = (
        F.relu(0.01 - spike_rates).square()
        + F.relu(spike_rates - 0.20).square()
    ).mean() * float(regularization_fraction)
    cvar = zero
    if objective.cvar_weight > 0 and label_mask.any():
        active = (residual_per * weight)[label_mask]
        count = max(1, int(math.ceil(0.20 * active.numel())))
        # Use the same stable denominator as the other weighted objectives so
        # tail/identity weights cannot disappear through local renormalization.
        cvar = torch.topk(active, count).values.sum() / denominator

    anchor_residual_loss = zero
    anchor_nll = zero
    anchor_gate = zero
    anchor_loss_enabled = any(
        value > 0.0
        for value in (
            objective.anchor_residual_weight,
            objective.anchor_nll_weight,
            objective.anchor_gate_weight,
        )
    )
    if anchor_loss_enabled:
        required_anchor_outputs = (
            "anchor_residual_bpm",
            "anchor_residual_limit_bpm",
            "corrected_anchor_rr",
            "corrected_anchor_scale_bpm",
            "anchor_snap_gate_logit",
            "candidate_source_rr",
            "candidate_source_available",
        )
        missing = [name for name in required_anchor_outputs if name not in output]
        if missing:
            raise RuntimeError(
                f"i3 anchor objective lacks model outputs: {missing}"
            )
        anchor_rr = batch.get("base_prediction")
        anchor_available = batch.get("base_available")
        if anchor_rr is None or anchor_available is None:
            raise RuntimeError("i3 anchor objective lacks bound posterior context")
        predicted_anchor_residual = output["anchor_residual_bpm"]
        anchor_residual_limit = output["anchor_residual_limit_bpm"]
        corrected_anchor_rr = output["corrected_anchor_rr"]
        corrected_anchor_scale = output["corrected_anchor_scale_bpm"]
        snap_gate_logit = output["anchor_snap_gate_logit"]
        candidate_source_rr = output["candidate_source_rr"]
        candidate_source_available = output["candidate_source_available"]
        assert all(
            isinstance(value, Tensor)
            for value in (
                predicted_anchor_residual,
                anchor_residual_limit,
                corrected_anchor_rr,
                corrected_anchor_scale,
                snap_gate_logit,
                candidate_source_rr,
                candidate_source_available,
            )
        )
        anchor_weight = weight * anchor_available.bool().float()
        raw_anchor_residual_target = target - anchor_rr
        clipped_residual_target = torch.maximum(
            torch.minimum(raw_anchor_residual_target, anchor_residual_limit),
            -anchor_residual_limit,
        )
        anchor_residual_per = F.smooth_l1_loss(
            predicted_anchor_residual,
            clipped_residual_target,
            beta=0.75,
            reduction="none",
        )
        safe_scale = corrected_anchor_scale.float().clamp(0.05, 20.0)
        standardized_residual = (
            clipped_residual_target.float()
            - predicted_anchor_residual.float()
        ) / safe_scale
        anchor_nll_per = (
            0.5 * standardized_residual.square()
            + safe_scale.log()
            + 0.5 * math.log(2.0 * math.pi)
        )
        candidate_better = (
            candidate_source_available.bool()
            & (
                (candidate_source_rr.detach() - target).abs() + 0.05
                < (corrected_anchor_rr.detach() - target).abs()
            )
        ).float()
        gate_per = F.binary_cross_entropy_with_logits(
            snap_gate_logit, candidate_better, reduction="none"
        )
        gate_weight = anchor_weight * candidate_source_available.bool().float()
        anchor_residual_loss = weighted(anchor_residual_per, anchor_weight)
        anchor_nll = weighted(anchor_nll_per, anchor_weight)
        anchor_gate = weighted(gate_per, gate_weight)
    total = (
        objective.listwise_weight * listwise
        + objective.nearest_ce_weight * nearest_ce
        + objective.selected_residual_weight * residual
        + objective.all_candidate_residual_weight * candidate_residual_loss
        + objective.mixture_nll_weight * nll
        + objective.factor_weight * factor
        + objective.quality_weight * quality
        + objective.spike_weight * spike
        + objective.cvar_weight * cvar
        + objective.anchor_residual_weight * anchor_residual_loss
        + objective.anchor_nll_weight * anchor_nll
        + objective.anchor_gate_weight * anchor_gate
    )
    return total, {
        "listwise": listwise, "nearest_ce": nearest_ce,
        "residual": residual, "candidate_residual": candidate_residual_loss,
        "nll": nll, "factor": factor, "quality": quality,
        "spike": spike, "cvar": cvar,
        "anchor_residual": anchor_residual_loss,
        "anchor_nll": anchor_nll,
        "anchor_gate": anchor_gate,
    }


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def supervision_weight_denominator(
    metadata: pd.DataFrame,
    sessions: Sequence[np.ndarray],
    row_weights: Tensor,
    *,
    warmup_windows: int,
    tail_weight: float,
) -> Tensor:
    """Return one fixed denominator for an entire accumulation group."""

    mass = row_weights.new_zeros(())
    for session in sessions:
        position = np.asarray(session, dtype=np.int64).copy()
        frame = metadata.iloc[position]
        valid = (
            frame["reference_valid"].astype(bool).to_numpy()
            & np.isfinite(frame["rr_bpm"].to_numpy(float))
        )
        if warmup_windows:
            valid[: min(int(warmup_windows), len(valid))] = False
        index = torch.as_tensor(position, device=row_weights.device)
        selected = row_weights[index].float()
        if tail_weight > 0:
            target = torch.as_tensor(
                frame["rr_bpm"].to_numpy(np.float32, copy=True),
                device=row_weights.device,
            )
            selected = selected * (
                1.0 + float(tail_weight) * ((target >= 25.0) & (target <= 35.0)).float()
            )
        mass = mass + (selected * torch.as_tensor(valid, device=row_weights.device)).sum()
    return mass.clamp_min(1.0e-8)


def run_training_epoch(
    model: HarmonicCandidateSetEpisodeSNN,
    experiment: Experiment,
    positions: np.ndarray,
    scaler: RobustNodeScaler,
    optimizer: torch.optim.Optimizer,
    row_weights: Tensor,
    device: torch.device,
    *,
    amp: bool,
    gradient_scaler: torch.amp.GradScaler,
    seed: int,
    epoch: int,
    adaptive_iteration: int,
    tail_weight: float,
    cvar_weight: float,
    objective: IterationObjective | None = None,
    gradient_accumulation_sessions: int = 1,
    warmup_windows: int = WARMUP_WINDOWS,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, float]:
    """Train one epoch without ever updating parameters mid-session.

    Gradients are normalized once over an identity-diverse group of complete
    sessions.  State remains causal inside each session, is detached at chunk
    boundaries for truncated BPTT, and is discarded only at a physical session
    boundary.  The optimizer steps after the final chunk of the final session
    in the group.
    """

    if gradient_accumulation_sessions < 1 or chunk_size < 1 or warmup_windows < 0:
        raise ValueError("accumulation/chunk/warmup settings are invalid")
    objective = objective or resolve_iteration_objective(
        adaptive_iteration,
        warmup_windows=warmup_windows,
        gradient_accumulation_sessions=gradient_accumulation_sessions,
        tail_weight=tail_weight,
        cvar_weight=cvar_weight,
    )
    model.train()
    rng = np.random.default_rng(seed + 7919 * epoch)
    totals: dict[str, float] = {}
    updates = 0
    chunks_seen = 0
    sessions = list(
        iter_identity_balanced_session_positions(
            experiment.metadata, positions, shuffle=True, rng=rng
        )
    )
    for group_start in range(0, len(sessions), int(gradient_accumulation_sessions)):
        group = sessions[group_start : group_start + int(gradient_accumulation_sessions)]
        denominator = supervision_weight_denominator(
            experiment.metadata,
            group,
            row_weights,
            warmup_windows=int(warmup_windows),
            tail_weight=float(objective.tail_weight),
        )
        group_window_count = max(sum(len(session) for session in group), 1)
        optimizer.zero_grad(set_to_none=True)
        accumulated_gradient = False
        for session in group:
            state: HarmonicSetState | None = None
            for start in range(0, len(session), int(chunk_size)):
                chunk = session[start : start + int(chunk_size)]
                batch = _batch_for_positions(
                    experiment,
                    scaler,
                    chunk,
                    device,
                    warmup_offset=start,
                    warmup_windows=int(warmup_windows),
                )
                with _autocast(device, amp):
                    output = forward_source_model(model, batch, state=state)
                    state = detach_state(output["state"])  # type: ignore[arg-type]
                    loss, components = compute_multitask_loss(
                        output,
                        batch,
                        row_weights,
                        adaptive_iteration=adaptive_iteration,
                        objective=objective,
                        normalization_denominator=denominator,
                        regularization_fraction=len(chunk) / group_window_count,
                    )
                if (
                    loss.requires_grad
                    and torch.isfinite(loss)
                    and float(loss.detach()) != 0.0
                ):
                    gradient_scaler.scale(loss).backward()
                    accumulated_gradient = True
                    chunks_seen += 1
                    for name, value in {"total": loss, **components}.items():
                        totals[name] = totals.get(name, 0.0) + float(value.detach())
        if accumulated_gradient:
            gradient_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            updates += 1
    divisor = max(chunks_seen, 1)
    averaged = {name: value / divisor for name, value in totals.items()}
    averaged["optimizer_steps"] = float(updates)
    return averaged


@torch.inference_mode()
def predict_positions(model: HarmonicCandidateSetEpisodeSNN, experiment: Experiment, positions: np.ndarray, scaler: RobustNodeScaler, device: torch.device, *, amp: bool) -> Predictions:
    model.eval()
    fields: dict[str, list[np.ndarray]] = {name: [] for name in (
        "position", "source_prediction", "source_scale", "source_available",
        "selected_index", "selected_probability", "margin", "entropy",
        "normalized_entropy", "valid_candidate_count", "quality", "spike_rate",
    )}
    if model.anchor_enabled:
        fields.update(
            {
                name: []
                for name in (
                    "raw_anchor_prediction",
                    "raw_anchor_std",
                    "corrected_anchor_prediction",
                    "anchor_residual",
                    "anchor_snap_gate",
                    "candidate_source_prediction",
                )
            }
        )
    rng = np.random.default_rng(0)
    for session in iter_session_positions(experiment.metadata, positions, shuffle=False, rng=rng):
        state: HarmonicSetState | None = None
        for start in range(0, len(session), CHUNK_SIZE):
            chunk = session[start : start + CHUNK_SIZE]
            batch = _batch_for_positions(experiment, scaler, chunk, device, warmup_offset=start)
            with _autocast(device, amp):
                output = forward_source_model(model, batch, state=state)
            state = output["state"]  # type: ignore[assignment]
            probabilities = output["candidate_probabilities"]
            assert isinstance(probabilities, Tensor)
            top = torch.topk(probabilities, k=min(2, probabilities.shape[-1]), dim=-1).values
            margin = top[..., 0] - (top[..., 1] if top.shape[-1] > 1 else 0.0)
            entropy = -(probabilities * probabilities.clamp_min(1.0e-8).log()).sum(dim=-1)
            valid_candidate_count = batch["candidate_mask"].sum(dim=-1)
            entropy_denominator = valid_candidate_count.clamp_min(2).float().log()
            normalized_entropy = torch.where(
                valid_candidate_count > 1,
                entropy / entropy_denominator.clamp_min(1.0e-8),
                torch.zeros_like(entropy),
            )
            spike_sequence = output["spike_sequence"]
            assert isinstance(spike_sequence, Tensor)
            mapping = {
                "position": batch["position"], "source_prediction": output["source_rr"],
                "source_scale": output["source_scale_bpm"], "source_available": output["source_available"],
                "selected_index": output["selected_index"], "selected_probability": output["selected_probability"],
                "margin": margin, "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "valid_candidate_count": valid_candidate_count,
                "quality": output["quality"],
                "spike_rate": spike_sequence.mean(dim=-1),
            }
            if model.anchor_enabled:
                mapping.update(
                    raw_anchor_prediction=output["raw_anchor_rr"],
                    raw_anchor_std=output["raw_anchor_std_bpm"],
                    corrected_anchor_prediction=output["corrected_anchor_rr"],
                    anchor_residual=output["anchor_residual_bpm"],
                    anchor_snap_gate=output["anchor_snap_gate"],
                    candidate_source_prediction=output["candidate_source_rr"],
                )
            for name, tensor in mapping.items():
                assert isinstance(tensor, Tensor)
                fields[name].append(tensor.reshape(-1).detach().cpu().numpy())
    arrays = {name: np.concatenate(parts) for name, parts in fields.items()}
    order = np.argsort(arrays["position"], kind="stable")
    position = arrays["position"][order].astype(np.int64)
    metadata = experiment.metadata.iloc[position]
    valid = metadata["reference_valid"].astype(bool).to_numpy() & np.isfinite(metadata["rr_bpm"].to_numpy(float))
    position = position[valid]
    def take(name: str, dtype: Any = np.float32) -> np.ndarray:
        return arrays[name][order][valid].astype(dtype)
    return Predictions(
        position=position, cache_index=metadata["cache_index"].to_numpy(np.int64)[valid],
        target=metadata["rr_bpm"].to_numpy(np.float32)[valid],
        identity=metadata["identity"].astype(str).to_numpy()[valid],
        base_prediction=experiment.base_prediction[position].copy(),
        base_std=experiment.base_std[position].copy(), base_available=experiment.base_available[position].copy(),
        source_prediction=take("source_prediction"), source_scale=take("source_scale"),
        source_available=take("source_available", bool), selected_index=take("selected_index", np.int64),
        selected_probability=take("selected_probability"), margin=take("margin"),
        entropy=take("entropy"), quality=take("quality"), spike_rate=take("spike_rate"),
        normalized_entropy=take("normalized_entropy"),
        valid_candidate_count=take("valid_candidate_count", np.int64),
        raw_anchor_prediction=(
            take("raw_anchor_prediction") if model.anchor_enabled else None
        ),
        raw_anchor_std=(take("raw_anchor_std") if model.anchor_enabled else None),
        corrected_anchor_prediction=(
            take("corrected_anchor_prediction") if model.anchor_enabled else None
        ),
        anchor_residual=(take("anchor_residual") if model.anchor_enabled else None),
        anchor_snap_gate=(
            take("anchor_snap_gate") if model.anchor_enabled else None
        ),
        candidate_source_prediction=(
            take("candidate_source_prediction") if model.anchor_enabled else None
        ),
    )


def evaluation_metrics(target: np.ndarray, prediction: np.ndarray, identity: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, float); prediction = np.asarray(prediction, float)
    error = np.abs(prediction - target)
    per_identity = {str(name): float(error[identity == name].mean()) for name in sorted(set(map(str, identity)))}
    tail = (target >= 25.0) & (target <= 35.0)
    tail_ids = {str(name): float(error[tail & (identity == name)].mean()) for name in sorted(set(map(str, identity[tail])))} if tail.any() else {}
    return {
        "rows": int(len(target)), "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "within_2": float(np.mean(error <= 2.0)),
        "catastrophic_over_5": float(np.mean(error > 5.0)),
        "identity_macro_mae": float(np.mean(list(per_identity.values()))),
        "identity_mae": per_identity,
        "tail_25_35_mae": float(error[tail].mean()) if tail.any() else None,
        "tail_25_35_identity_macro_mae": float(np.mean(list(tail_ids.values()))) if tail_ids else None,
    }


def commercial_gate_selection_key(
    metrics: Mapping[str, Any],
) -> tuple[int, float, float, float, float]:
    """Deterministic lexicographic key over all six frozen source gates."""

    violations: list[float] = []
    for metric, (direction, threshold) in COMMERCIAL_SOURCE_GATES.items():
        raw = metrics.get(metric)
        value = float(raw) if raw is not None else math.inf
        if not math.isfinite(value):
            violation = math.inf
        elif direction == "maximum":
            violation = max(0.0, value / float(threshold) - 1.0)
        elif direction == "minimum":
            violation = max(0.0, (float(threshold) - value) / float(threshold))
        else:  # pragma: no cover - constant definition invariant
            raise RuntimeError(f"unknown gate direction: {direction}")
        violations.append(float(violation))
    fail_count = sum(value > 0.0 for value in violations)
    return (
        int(fail_count),
        float(max(violations, default=0.0)),
        float(sum(violations)),
        float(metrics["identity_macro_mae"]),
        float(metrics["mae"]),
    )


def iter_policy_grid() -> Iterator[tuple[float, float, float, float, float]]:
    for probability in PROBABILITY_THRESHOLDS:
        for margin in MARGIN_THRESHOLDS:
            for entropy in ENTROPY_THRESHOLDS:
                for quality in QUALITY_THRESHOLDS:
                    for pull in CORRECTION_PULLS:
                        yield probability, margin, entropy, quality, pull


def _prediction_policy_context(
    prediction: Predictions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if prediction.valid_candidate_count is None:
        candidate_count = np.ones(len(prediction.target), dtype=np.int64)
    else:
        candidate_count = np.asarray(prediction.valid_candidate_count, np.int64)
    if prediction.normalized_entropy is None:
        raw_entropy = np.asarray(prediction.entropy, np.float64)
        denominator = np.log(np.maximum(candidate_count, 2))
        normalized_entropy = np.where(
            candidate_count > 1, raw_entropy / np.maximum(denominator, 1.0e-8), 0.0
        )
    else:
        normalized_entropy = np.asarray(prediction.normalized_entropy, np.float64)
    disagreement = np.abs(
        np.asarray(prediction.source_prediction, np.float64)
        - np.asarray(prediction.base_prediction, np.float64)
    )
    return normalized_entropy, candidate_count, disagreement


def apply_fallback_policy(prediction: Predictions, policy: FallbackPolicy) -> Predictions:
    base = np.asarray(prediction.base_prediction, np.float64)
    source = np.asarray(prediction.source_prediction, np.float64)
    has_base = np.asarray(prediction.base_available, bool)
    has_source = np.asarray(prediction.source_available, bool)
    if has_base.shape != base.shape:
        raise RuntimeError("base availability shape mismatch")
    if np.any(has_base & (~np.isfinite(base) | ~np.isfinite(prediction.base_std) | (prediction.base_std <= 0))):
        raise RuntimeError("available fallback rows require finite prediction and positive std")
    normalized_entropy, candidate_count, disagreement = _prediction_policy_context(prediction)
    base_std_max = math.inf if policy.base_std_max is None else float(policy.base_std_max)
    source_scale_max = math.inf if policy.source_scale_max is None else float(policy.source_scale_max)
    disagreement_max = math.inf if policy.disagreement_max is None else float(policy.disagreement_max)
    action = (
        has_base & has_source
        & (prediction.selected_probability >= policy.probability_threshold)
        & (prediction.margin >= policy.margin_threshold)
        & (normalized_entropy <= policy.entropy_threshold)
        & (prediction.quality >= policy.quality_threshold)
        & (prediction.base_std <= base_std_max)
        & (prediction.source_scale <= source_scale_max)
        & (disagreement >= policy.disagreement_min)
        & (disagreement <= disagreement_max)
        & (candidate_count >= policy.minimum_valid_candidates)
    )
    applied = action.astype(np.float64) * float(policy.correction_pull)
    # Exact structural fallback: unavailable base uses source; unavailable source uses base.
    applied = np.where(~has_base & has_source, 1.0, applied)
    applied = np.where(has_base & ~has_source, 0.0, applied)
    final = (1.0 - applied) * np.where(has_base, base, source) + applied * source
    # Preserve exact source/base bytes on structural fallback rows even if the
    # arithmetic implementation or NumPy promotion rules change later.
    final[~has_base & has_source] = source[~has_base & has_source]
    final[has_base & ~has_source] = base[has_base & ~has_source]
    values = {name: getattr(prediction, name) for name in prediction.__dataclass_fields__}
    values["final_prediction"] = final.astype(np.float32)
    values["applied_pull"] = applied.astype(np.float32)
    return Predictions(**values)


def correction_policy_diagnostics(
    prediction: Predictions,
    applied: Predictions,
    *,
    improvement_margin_bpm: float = 0.05,
) -> dict[str, float | int]:
    """Compute correction precision/recall/FPR with explicit denominators."""

    if applied.final_prediction is None or applied.applied_pull is None:
        raise ValueError("an applied policy prediction is required")
    eligible = prediction.base_available & prediction.source_available
    action = (applied.applied_pull > 0) & eligible
    base_error = np.abs(prediction.base_prediction - prediction.target)
    source_error = np.abs(prediction.source_prediction - prediction.target)
    final_error = np.abs(applied.final_prediction - prediction.target)
    actionable = eligible & (
        source_error + float(improvement_margin_bpm) < base_error
    )
    actual_improvement = (
        base_error - final_error > float(improvement_margin_bpm)
    )
    base_good = eligible & (base_error <= 2.0)
    bad_base_good = action & base_good & (
        final_error > base_error + float(improvement_margin_bpm)
    )
    action_count = int(action.sum())
    actionable_count = int(actionable.sum())
    base_good_count = int(base_good.sum())
    return {
        "eligible": int(eligible.sum()),
        "actions": action_count,
        "actionable": actionable_count,
        "base_good": base_good_count,
        "coverage": action_count / max(int(eligible.sum()), 1),
        "precision": float(np.mean(actual_improvement[action])) if action_count else 1.0,
        "recall": (
            float(np.sum(action & actionable) / actionable_count)
            if actionable_count else 1.0
        ),
        "fpr": (
            float(np.sum(bad_base_good) / base_good_count)
            if base_good_count else 0.0
        ),
    }


def select_fallback_policy(
    prediction: Predictions,
    *,
    maximum_coverage: float = 0.15,
    maximum_fpr: float = 0.10,
    minimum_precision: float = 0.60,
    minimum_correction_recall: float = 0.0,
    gate_aware: bool = False,
) -> tuple[FallbackPolicy, Predictions]:
    """Select a correction policy using actual pulled-output outcomes.

    Precision is the fraction of authorized pulls that reduce final absolute
    error.  FPR is the fraction of eligible base-good rows (base error <=2 bpm)
    on which an authorized pull makes the final error worse.  Recall measures
    how many rows with a genuinely improving source are authorized.  These are
    deliberately different denominators.
    """

    base_reference = np.where(prediction.base_available, prediction.base_prediction, prediction.source_prediction)
    base_metrics = evaluation_metrics(prediction.target, base_reference, prediction.identity)
    base_cardinality = len(PROBABILITY_THRESHOLDS) * len(MARGIN_THRESHOLDS) * len(ENTROPY_THRESHOLDS) * len(QUALITY_THRESHOLDS) * len(CORRECTION_PULLS)
    cardinality = base_cardinality * len(POLICY_CONTEXT_PROFILES)
    best: tuple[tuple[float, ...], FallbackPolicy, Predictions] | None = None
    for base_std_max, source_scale_max, disagreement_min, disagreement_max, minimum_candidates in POLICY_CONTEXT_PROFILES:
        for probability, margin, entropy, quality, pull in iter_policy_grid():
            provisional = FallbackPolicy(
                probability, margin, entropy, quality, pull,
                0.0, 1.0, 0.0, math.inf, {}, cardinality,
                base_std_max, source_scale_max, disagreement_min,
                disagreement_max, minimum_candidates, 1.0,
            )
            applied = apply_fallback_policy(prediction, provisional)
            diagnostics = correction_policy_diagnostics(prediction, applied)
            coverage = float(diagnostics["coverage"])
            assert applied.final_prediction is not None
            precision = float(diagnostics["precision"])
            recall = float(diagnostics["recall"])
            fpr = float(diagnostics["fpr"])
            metrics = evaluation_metrics(applied.target, applied.final_prediction, applied.identity)
            safeguards = {
                "coverage_at_most_cap": coverage <= maximum_coverage + 1.0e-12,
                "precision_at_least_floor": precision >= minimum_precision - 1.0e-12,
                "correction_recall_at_least_floor": recall >= minimum_correction_recall - 1.0e-12,
                "base_good_bad_correction_fpr_at_most_cap": fpr <= maximum_fpr + 1.0e-12,
                "macro_mae_noninferior": metrics["identity_macro_mae"] <= base_metrics["identity_macro_mae"] + 1.0e-12,
                "catastrophic_noninferior": metrics["catastrophic_over_5"] <= base_metrics["catastrophic_over_5"] + 1.0e-12,
                "promotion_eligible": True,
            }
            if not all(safeguards.values()):
                continue
            policy = FallbackPolicy(
                probability, margin, entropy, quality, pull,
                coverage, precision, fpr, metrics["identity_macro_mae"],
                safeguards, cardinality, base_std_max, source_scale_max,
                disagreement_min, disagreement_max, minimum_candidates, recall,
                "promoted", True,
                COMMERCIAL_SELECTION_OBJECTIVE if gate_aware else "legacy_macro_mae",
            )
            key = (
                commercial_gate_selection_key(metrics)
                + (-recall, coverage, -precision)
                if gate_aware
                else (
                    metrics["identity_macro_mae"],
                    metrics["catastrophic_over_5"],
                    -metrics["within_2"], -recall, coverage, -precision,
                )
            )
            if best is None or key < best[0]:
                best = key, policy, applied
    if best is None:
        # A missed promotion floor is an experimental result, not a reason to
        # lose an otherwise valid training run.  Lock a structural no-action
        # policy: rows with a raw fallback retain it exactly; rows lacking that
        # fallback may still use the source as the pre-existing structural
        # behavior.  Recall is reported honestly and promotion remains false.
        no_action = FallbackPolicy(
            probability_threshold=1.10,
            margin_threshold=1.10,
            entropy_threshold=0.10,
            quality_threshold=0.75,
            correction_pull=0.0,
            validation_coverage=0.0,
            validation_precision=1.0,
            validation_fpr=0.0,
            validation_macro_mae=float(base_metrics["identity_macro_mae"]),
            safeguards={},
            grid_cardinality=cardinality,
            base_std_max=math.inf,
            source_scale_max=math.inf,
            disagreement_min=math.inf,
            disagreement_max=math.inf,
            minimum_valid_candidates=max(1, HarmonicCandidateSetEpisodeSNN.MAX_CANDIDATES),
            validation_recall=0.0,
            selection_status="fail_closed_no_action",
            promotion_eligible=False,
            selection_objective=(
                COMMERCIAL_SELECTION_OBJECTIVE
                if gate_aware
                else "legacy_macro_mae"
            ),
        )
        applied = apply_fallback_policy(prediction, no_action)
        diagnostics = correction_policy_diagnostics(prediction, applied)
        assert applied.final_prediction is not None
        metrics = evaluation_metrics(
            applied.target, applied.final_prediction, applied.identity
        )
        safeguards = {
            "coverage_at_most_cap": float(diagnostics["coverage"])
            <= maximum_coverage + 1.0e-12,
            "precision_at_least_floor": float(diagnostics["precision"])
            >= minimum_precision - 1.0e-12,
            "correction_recall_at_least_floor": float(diagnostics["recall"])
            >= minimum_correction_recall - 1.0e-12,
            "base_good_bad_correction_fpr_at_most_cap": float(diagnostics["fpr"])
            <= maximum_fpr + 1.0e-12,
            "macro_mae_noninferior": metrics["identity_macro_mae"]
            <= base_metrics["identity_macro_mae"] + 1.0e-12,
            "catastrophic_noninferior": metrics["catastrophic_over_5"]
            <= base_metrics["catastrophic_over_5"] + 1.0e-12,
            "promotion_eligible": False,
        }
        no_action = FallbackPolicy(
            **{
                **asdict(no_action),
                "validation_coverage": float(diagnostics["coverage"]),
                "validation_precision": float(diagnostics["precision"]),
                "validation_fpr": float(diagnostics["fpr"]),
                "validation_macro_mae": float(metrics["identity_macro_mae"]),
                "validation_recall": float(diagnostics["recall"]),
                "safeguards": safeguards,
            }
        )
        return no_action, applied
    return best[1], best[2]


def concatenate_predictions(predictions: Sequence[Predictions]) -> Predictions:
    """Concatenate discovery folds so one policy is selected across all of them."""

    if not predictions:
        raise ValueError("at least one discovery prediction is required")
    values: dict[str, Any] = {}
    for field_name in Predictions.__dataclass_fields__:
        parts = [getattr(prediction, field_name) for prediction in predictions]
        if all(part is None for part in parts):
            values[field_name] = None
        elif any(part is None for part in parts):
            # Optional context introduced after iteration 1 is reconstructed by
            # the selector only when every fold lacks it; mixed schemas fail.
            raise ValueError(f"discovery predictions disagree on optional field {field_name}")
        else:
            values[field_name] = np.concatenate(
                [np.asarray(part) for part in parts], axis=0
            )
    return Predictions(**values)


def select_fallback_policy_multi(
    predictions: Sequence[Predictions], **kwargs: Any
) -> tuple[FallbackPolicy, Predictions]:
    """Reusable multi-fold selector; never average independently fit policies."""

    return select_fallback_policy(concatenate_predictions(predictions), **kwargs)


def _model_configuration(
    preset: str,
    feature_count: int,
    *,
    adaptive_iteration: int = 1,
    anchor_residual_mode: str = "disabled",
    anchor_max_residual_bpm: float = 12.0,
    anchor_minimum_scale_bpm: float = 0.25,
    anchor_maximum_scale_bpm: float = 12.0,
    anchor_initial_scale_bpm: float = 1.5,
    anchor_distance_weight: float = 1.0,
    anchor_source_mode: str = "learned_blend",
) -> dict[str, Any]:
    """Resolve the checkpoint-bound source architecture.

    i1/i2 deliberately return the historical dictionary byte-for-byte.  The
    optional modules and their settings appear only for the explicit i3
    ``causal_posterior`` mode, making architecture drift detectable on resume.
    """

    config = {
        "node_features": int(feature_count),
        "graph_blocks": 2,
        **PRESETS[preset],
    }
    if anchor_residual_mode == "disabled":
        return config
    if anchor_residual_mode != "causal_posterior" or int(adaptive_iteration) != 3:
        raise ValueError("causal posterior-anchor architecture is valid only for i3")
    config.update(
        anchor_enabled=True,
        anchor_max_residual_bpm=float(anchor_max_residual_bpm),
        anchor_minimum_scale_bpm=float(anchor_minimum_scale_bpm),
        anchor_maximum_scale_bpm=float(anchor_maximum_scale_bpm),
        anchor_initial_scale_bpm=float(anchor_initial_scale_bpm),
        anchor_distance_weight=float(anchor_distance_weight),
        anchor_source_mode=str(anchor_source_mode),
    )
    return config


def _source_bindings() -> dict[str, dict[str, str]]:
    """Hash every repository source/config file effective in HCS training."""

    paths = {
        "trainer": Path(__file__).resolve(),
        "harmonic_set_model": SRC_ROOT / "snn_rr/harmonic_set_models.py",
        "spiking_cell_model": SRC_ROOT / "snn_rr/svd_episode_models.py",
        "campaign_config": PROJECT_ROOT / "configs/harmonic_set_v2.yaml",
        "project_configuration": PROJECT_ROOT / "pyproject.toml",
        "adaptive_campaign_contract": (
            PROJECT_ROOT
            / "artifacts/campaigns/harmonic_candidate_set_snn_v2/ADAPTIVE_CAMPAIGN_CONTRACT.json"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"iteration-effective source/config files are absent: {missing}")
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _resolve_recorded_source_bindings(
    recorded: Mapping[str, Any],
    current: Mapping[str, Mapping[str, str]],
    *,
    snapshot_root: Path | None,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Resolve historical hashes to original files or supplied snapshots."""

    if set(recorded) != set(current):
        raise RuntimeError("recovery source/config binding names disagree")
    root = snapshot_root.expanduser().resolve() if snapshot_root is not None else None
    resolved: dict[str, dict[str, str]] = {}
    provenance: dict[str, Any] = {}
    for name in sorted(recorded):
        binding = recorded[name]
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"recovery source binding is malformed: {name}")
        expected_hash = str(binding.get("sha256", ""))
        original_path = Path(str(binding.get("path", "")))
        candidate: Path | None = None
        resolution = "original"
        if original_path.is_file() and sha256_file(original_path) == expected_hash:
            candidate = original_path.resolve()
        elif root is not None:
            snapshot = root / original_path.name
            if snapshot.is_file() and sha256_file(snapshot) == expected_hash:
                candidate = snapshot.resolve()
                resolution = "snapshot"
        if candidate is None:
            raise RuntimeError(
                f"recovery source hash cannot be resolved for {name}; supply matching snapshots"
            )
        resolved[name] = {"path": str(candidate), "sha256": expected_hash}
        provenance[name] = {
            "logical_original_path": str(original_path),
            "resolved_path": str(candidate),
            "sha256": expected_hash,
            "resolution": resolution,
        }
    return resolved, provenance


def _verify_anchor_disabled_snapshot_forward_compatibility(
    *,
    snapshot_model_path: Path,
    expected_snapshot_sha256: str,
    model_config: Mapping[str, Any],
    model_state: Mapping[str, Tensor],
) -> dict[str, Any]:
    """Bit-compare current i1/i2 forward against the hash-bound snapshot."""

    if bool(model_config.get("anchor_enabled", False)):
        raise RuntimeError("historical compatibility verifier is anchor-disabled only")
    if sha256_file(snapshot_model_path) != expected_snapshot_sha256:
        raise RuntimeError("historical model snapshot hash mismatch")
    module_name = f"snn_rr._hcs_recovery_{expected_snapshot_sha256[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, snapshot_model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import historical harmonic model snapshot")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        historical_class = getattr(module, "HarmonicCandidateSetEpisodeSNN")
        historical = historical_class(**dict(model_config)).cpu().eval()
        current_model = HarmonicCandidateSetEpisodeSNN(**dict(model_config)).cpu().eval()
        cpu_state = {
            str(name): tensor.detach().cpu()
            for name, tensor in model_state.items()
        }
        historical.load_state_dict(cpu_state, strict=True)
        current_model.load_state_dict(cpu_state, strict=True)
        features = int(model_config["node_features"])
        batch, time_steps, candidates = 2, 5, 4
        node = torch.linspace(
            -1.0, 1.0, batch * time_steps * candidates * features
        ).reshape(batch, time_steps, candidates, features)
        rr = torch.tensor([8.0, 16.0, 24.0, 32.0]).reshape(1, 1, -1).expand(
            batch, time_steps, -1
        ).clone()
        candidate_mask = torch.ones(batch, time_steps, candidates, dtype=torch.bool)
        candidate_mask[0, 2, 3] = False
        sequence_mask = torch.ones(batch, time_steps, dtype=torch.bool)
        radar_mask = torch.ones(batch, time_steps, 3, dtype=torch.bool)
        radar_mask[1, 3, 2] = False
        reset_mask = torch.zeros(batch, time_steps, dtype=torch.bool)
        reset_mask[1, 4] = True
        arguments = (node, rr, candidate_mask, sequence_mask)
        keyword = {"radar_mask": radar_mask, "reset_mask": reset_mask}
        with torch.inference_mode():
            old_output = historical(*arguments, **keyword)
            new_output = current_model(*arguments, **keyword)
        keys = (
            "candidate_logits", "candidate_probabilities",
            "candidate_residual_bpm", "candidate_scale_bpm", "factor_logits",
            "quality_logit", "source_rr", "source_scale_bpm",
            "source_available", "selected_index", "state_sequence",
            "spike_sequence", "spike_rates",
        )
        for key in keys:
            if not torch.equal(old_output[key], new_output[key]):
                raise RuntimeError(
                    f"current anchor-disabled forward differs from snapshot: {key}"
                )
        old_state, new_state = old_output["state"], new_output["state"]
        for old_layer, new_layer in zip(old_state, new_state, strict=True):
            for old_tensor, new_tensor in zip(old_layer, new_layer, strict=True):
                if not torch.equal(old_tensor, new_tensor):
                    raise RuntimeError(
                        "current anchor-disabled causal state differs from snapshot"
                    )
        vector_hash = hashlib.sha256()
        for key in keys:
            tensor = new_output[key].detach().cpu().contiguous()
            vector_hash.update(tensor.numpy().tobytes())
        return {
            "status": "bit_exact",
            "historical_model_sha256": expected_snapshot_sha256,
            "test_vector_sha256": vector_hash.hexdigest(),
            "checked_outputs": list(keys) + ["state"],
        }
    finally:
        sys.modules.pop(module_name, None)


def _effective_configuration(
    args: argparse.Namespace,
    objective: IterationObjective,
    model_config: Mapping[str, Any],
    *,
    validation_fold: int,
    cache_manifest_sha256: str,
    fallback_oof_sha256: str,
) -> dict[str, Any]:
    """Canonical record of every runtime choice that can alter a locked run."""

    configuration: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "outer_fold": int(args.fold),
        "validation_fold": int(validation_fold),
        "seed": int(args.seed),
        "model": dict(model_config),
        "iteration_objective": iteration_objective_record(objective),
        "optimization": {
            "epochs": int(args.epochs),
            "minimum_epochs": int(args.minimum_epochs),
            "patience": int(args.patience),
            "learning_rate": float(args.learning_rate),
            "optimizer": "AdamW",
            "weight_decay": 1.0e-4,
            "gradient_clip_norm": 2.0,
            "amp": bool(args.amp),
            "deterministic_algorithms_warn_only": bool(args.deterministic),
            "chunk_windows": int(args.chunk_windows),
        },
        "policy": {
            "maximum_coverage": float(args.maximum_coverage),
            "maximum_fpr": float(args.maximum_fpr),
            "minimum_precision": float(args.minimum_precision),
            "minimum_correction_recall": float(args.minimum_correction_recall),
            "probability_thresholds": PROBABILITY_THRESHOLDS,
            "margin_thresholds": MARGIN_THRESHOLDS,
            "normalized_entropy_thresholds": ENTROPY_THRESHOLDS,
            "quality_thresholds": QUALITY_THRESHOLDS,
            "correction_pulls": CORRECTION_PULLS,
            "context_profiles": POLICY_CONTEXT_PROFILES,
        },
        "data_bindings": {
            "cache_manifest_sha256": cache_manifest_sha256,
            "fallback_oof_sha256": fallback_oof_sha256,
        },
        "split_protocol": {
            "fold_count": N_FOLDS,
            "validation_rule": "(outer_fold + 1) % 6",
            "unit": "physical identity/session",
        },
        "forward_allowlist": (
            [
                "node_features", "candidate_rr", "candidate_mask", "radar_mask",
                "sequence_mask", "causal_state", "reset_mask",
            ]
            + (
                ["anchor_rr", "anchor_std", "anchor_available"]
                if bool(model_config.get("anchor_enabled", False))
                else []
            )
        ),
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }
    if bool(model_config.get("anchor_enabled", False)):
        configuration["source_architecture"] = {
            "anchor_residual_mode": str(args.anchor_residual_mode),
            "deployment_source": str(args.anchor_source_mode),
            "unsafe_or_unavailable_anchor_fallback": (
                "bit-exact candidate source, then raw fallback policy"
            ),
        }
        configuration["posterior_anchor_contract"] = {
            "enabled": True,
            "source": "same cache_index-bound strict nested fallback/proposer",
            "causality": "current/past state only; no future lags",
            "forbidden": [
                "target", "reference_valid", "reference_qc",
                "identity", "protocol", "fold",
            ],
            "raw_anchor_role": "uncorrected fallback baseline",
            "source_role": "checkpoint-bound corrected/blended architecture",
        }
        configuration["retrospective_validation_selection"] = {
            "checkpoint_objective": COMMERCIAL_SELECTION_OBJECTIVE,
            "policy_objective": COMMERCIAL_SELECTION_OBJECTIVE,
            "commercial_source_gates": COMMERCIAL_SOURCE_GATES,
            "external_commercial_claim": False,
        }
    return configuration


def _validate_lock(output_dir: Path) -> dict[str, Any]:
    lock_path = output_dir / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for key, filename in (
        ("checkpoint_sha256", "best_checkpoint.pt"),
        ("scaler_sha256", "scaler.json"),
        ("policy_sha256", "fallback_policy.json"),
        ("run_manifest_sha256", "run_manifest.json"),
    ):
        if str(lock.get(key, "")) != sha256_file(output_dir / filename):
            raise RuntimeError(f"selection lock tamper detected: {filename}")
    for name, binding in lock.get("source_bindings", {}).items():
        path = Path(str(binding.get("path", "")))
        if not path.is_file() or str(binding.get("sha256", "")) != sha256_file(path):
            raise RuntimeError(f"selection lock source/config tamper detected: {name}")
    if "history_sha256" in lock and str(lock["history_sha256"]) != sha256_file(
        output_dir / "history.json"
    ):
        raise RuntimeError("selection lock tamper detected: history.json")
    return lock


def _validate_prelock_recovery(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    experiment: Experiment,
    train_positions: np.ndarray,
    validation_fold: int,
    model_config: Mapping[str, Any],
    objective: IterationObjective,
    effective_configuration_sha256: str,
    source_bindings: Mapping[str, Mapping[str, str]],
    cache_manifest_sha256: str,
    fallback_oof_sha256: str,
    device: torch.device,
    snapshot_root: Path | None,
) -> tuple[
    RobustNodeScaler,
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, str]],
    dict[str, Any],
]:
    """Validate a training-complete, selection-incomplete run without mutation."""

    if (output_dir / "selection_lock.json").exists():
        raise RuntimeError("prelock recovery refuses an already locked run; use --resume")
    forbidden_partial = (
        "fallback_policy.json",
        "validation_predictions.npz",
        "validation_metrics.json",
        "test_predictions.npz",
        "test_metrics.json",
    )
    present = [name for name in forbidden_partial if (output_dir / name).exists()]
    if present:
        raise RuntimeError(
            f"prelock recovery refuses ambiguous post-training artifacts: {present}"
        )
    required = (
        "run_manifest.json", "scaler.json", "best_checkpoint.pt", "history.json"
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"prelock recovery lacks required artifacts: {missing}")

    run_manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    recorded_effective = run_manifest.get("iteration_effective_configuration")
    recorded_effective_sha = str(
        run_manifest.get("iteration_effective_configuration_sha256", "")
    )
    if not isinstance(recorded_effective, Mapping) or (
        sha256_json(recorded_effective) != recorded_effective_sha
    ):
        raise RuntimeError("prelock run manifest effective configuration is invalid")
    if recorded_effective_sha != effective_configuration_sha256:
        raise RuntimeError("prelock effective configuration disagrees with this invocation")
    if run_manifest.get("model_config") != dict(model_config):
        raise RuntimeError("prelock model configuration disagrees with this invocation")
    if int(run_manifest.get("outer_fold", -1)) != int(args.fold):
        raise RuntimeError("prelock outer fold disagrees with this invocation")
    if int(run_manifest.get("validation_fold", -1)) != int(validation_fold):
        raise RuntimeError("prelock validation fold disagrees with this invocation")
    optimization = run_manifest.get("optimization", {})
    if int(optimization.get("seed", -1)) != int(args.seed):
        raise RuntimeError("prelock seed disagrees with this invocation")
    if optimization.get("iteration_objective") != iteration_objective_record(objective):
        raise RuntimeError("prelock iteration objective disagrees with this invocation")

    recorded_sources = run_manifest.get("source_and_config_bindings")
    if not isinstance(recorded_sources, Mapping):
        raise RuntimeError("prelock run manifest lacks source/config bindings")
    resolved_sources, source_provenance = _resolve_recorded_source_bindings(
        recorded_sources,
        source_bindings,
        snapshot_root=snapshot_root,
    )
    input_bindings = run_manifest.get("input_bindings", {})
    if str(input_bindings.get("cache_manifest_sha256", "")) != cache_manifest_sha256:
        raise RuntimeError("prelock cache manifest hash mismatch")
    if str(input_bindings.get("fallback_oof_sha256", "")) != fallback_oof_sha256:
        raise RuntimeError("prelock fallback hash mismatch")

    scaler_document = json.loads(
        (output_dir / "scaler.json").read_text(encoding="utf-8")
    )
    scaler = RobustNodeScaler(
        center=np.asarray(scaler_document["center"], dtype=np.float32).reshape(
            1, 1, -1
        ),
        scale=np.asarray(scaler_document["scale"], dtype=np.float32).reshape(
            1, 1, -1
        ),
        fit_positions_sha256=str(scaler_document["fit_positions_sha256"]),
    )
    expected_scaler = fit_robust_scaler(experiment, train_positions)
    if (
        scaler.fit_positions_sha256 != _positions_sha256(train_positions)
        or not np.array_equal(scaler.center, expected_scaler.center)
        or not np.array_equal(scaler.scale, expected_scaler.scale)
    ):
        raise RuntimeError("prelock scaler is tampered or fit on another split")

    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise RuntimeError("prelock history must contain at least one completed epoch")
    checkpoint = torch.load(
        output_dir / "best_checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    checks = {
        "model_config": dict(model_config),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "adaptive_iteration": int(args.adaptive_iteration),
        "iteration_objective": iteration_objective_record(objective),
        "effective_configuration_sha256": effective_configuration_sha256,
    }
    for key, expected in checks.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"prelock checkpoint {key} mismatch")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    history_epochs = {
        int(item.get("epoch", -1))
        for item in history
        if isinstance(item, Mapping)
    }
    if checkpoint_epoch < 1 or checkpoint_epoch not in history_epochs:
        raise RuntimeError("prelock checkpoint epoch is absent from history")
    if not math.isfinite(float(checkpoint.get("validation_source_macro_mae", math.nan))):
        raise RuntimeError("prelock checkpoint validation metric is non-finite")
    if not isinstance(checkpoint.get("model_state"), Mapping):
        raise RuntimeError("prelock checkpoint lacks a model state")
    model_resolution = source_provenance["harmonic_set_model"]
    if bool(model_config.get("anchor_enabled", False)):
        if model_resolution["resolution"] != "original":
            raise RuntimeError("anchor-enabled prelock recovery requires exact current source")
        compatibility = {
            "status": "exact_current_source_hash",
            "historical_model_sha256": str(model_resolution["sha256"]),
        }
    else:
        compatibility = _verify_anchor_disabled_snapshot_forward_compatibility(
            snapshot_model_path=Path(model_resolution["resolved_path"]),
            expected_snapshot_sha256=str(model_resolution["sha256"]),
            model_config=model_config,
            model_state=checkpoint["model_state"],
        )
    source_provenance["anchor_disabled_forward_compatibility"] = compatibility
    return (
        scaler,
        checkpoint,
        run_manifest,
        resolved_sources,
        source_provenance,
    )


def _validate_locked_recovery(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    experiment: Experiment,
    train_positions: np.ndarray,
    validation_fold: int,
    model_config: Mapping[str, Any],
    objective: IterationObjective,
    effective_configuration_sha256: str,
    current_source_bindings: Mapping[str, Mapping[str, str]],
    cache_manifest_sha256: str,
    fallback_oof_sha256: str,
    device: torch.device,
    snapshot_root: Path | None,
) -> tuple[
    RobustNodeScaler,
    dict[str, Any],
    FallbackPolicy,
    dict[str, Any],
    dict[str, Any],
]:
    """Validate a historical lock whose source paths may now point at new code."""

    if (output_dir / "recovery_provenance.json").exists():
        raise RuntimeError("locked recovery provenance already exists")
    if (output_dir / "validation_predictions.npz").exists() or (
        output_dir / "validation_metrics.json"
    ).exists():
        raise RuntimeError("locked run already has validation materialization")
    if (output_dir / "test_predictions.npz").exists():
        raise RuntimeError("locked recovery refuses a run with outer-test artifacts")
    lock_path = output_dir / "selection_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("locked recovery requires selection_lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for key, filename in (
        ("checkpoint_sha256", "best_checkpoint.pt"),
        ("scaler_sha256", "scaler.json"),
        ("policy_sha256", "fallback_policy.json"),
        ("run_manifest_sha256", "run_manifest.json"),
    ):
        path = output_dir / filename
        if not path.is_file() or str(lock.get(key, "")) != sha256_file(path):
            raise RuntimeError(f"historical lock tamper detected: {filename}")
    if "history_sha256" in lock and str(lock["history_sha256"]) != sha256_file(
        output_dir / "history.json"
    ):
        raise RuntimeError("historical lock tamper detected: history.json")
    if (
        int(lock.get("outer_fold", -1)) != int(args.fold)
        or int(lock.get("validation_fold", -1)) != int(validation_fold)
        or int(lock.get("seed", -1)) != int(args.seed)
        or int(lock.get("adaptive_iteration", -1)) != int(args.adaptive_iteration)
    ):
        raise RuntimeError("historical lock split/seed/iteration mismatch")
    if lock.get("iteration_objective") != iteration_objective_record(objective):
        raise RuntimeError("historical lock objective mismatch")
    if str(lock.get("effective_configuration_sha256", "")) != effective_configuration_sha256:
        raise RuntimeError("historical lock effective configuration mismatch")
    if str(lock.get("cache_manifest_sha256", "")) != cache_manifest_sha256:
        raise RuntimeError("historical lock cache hash mismatch")
    if str(lock.get("fallback_oof_sha256", "")) != fallback_oof_sha256:
        raise RuntimeError("historical lock fallback hash mismatch")

    run_manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    recorded_effective = run_manifest.get("iteration_effective_configuration")
    if (
        not isinstance(recorded_effective, Mapping)
        or sha256_json(recorded_effective) != effective_configuration_sha256
        or run_manifest.get("model_config") != dict(model_config)
    ):
        raise RuntimeError("historical run manifest configuration mismatch")
    recorded_sources = lock.get("source_bindings")
    if not isinstance(recorded_sources, Mapping):
        raise RuntimeError("historical lock lacks source bindings")
    _, source_provenance = _resolve_recorded_source_bindings(
        recorded_sources,
        current_source_bindings,
        snapshot_root=snapshot_root,
    )

    scaler_document = json.loads(
        (output_dir / "scaler.json").read_text(encoding="utf-8")
    )
    scaler = RobustNodeScaler(
        center=np.asarray(scaler_document["center"], np.float32).reshape(1, 1, -1),
        scale=np.asarray(scaler_document["scale"], np.float32).reshape(1, 1, -1),
        fit_positions_sha256=str(scaler_document["fit_positions_sha256"]),
    )
    expected_scaler = fit_robust_scaler(experiment, train_positions)
    if (
        scaler.fit_positions_sha256 != expected_scaler.fit_positions_sha256
        or not np.array_equal(scaler.center, expected_scaler.center)
        or not np.array_equal(scaler.scale, expected_scaler.scale)
    ):
        raise RuntimeError("historical locked scaler mismatch")
    checkpoint = torch.load(
        output_dir / "best_checkpoint.pt", map_location=device, weights_only=False
    )
    for key, expected in {
        "model_config": dict(model_config),
        "seed": int(args.seed),
        "fold": int(args.fold),
        "adaptive_iteration": int(args.adaptive_iteration),
        "iteration_objective": iteration_objective_record(objective),
        "effective_configuration_sha256": effective_configuration_sha256,
    }.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"historical checkpoint {key} mismatch")
    model_resolution = source_provenance["harmonic_set_model"]
    compatibility = _verify_anchor_disabled_snapshot_forward_compatibility(
        snapshot_model_path=Path(model_resolution["resolved_path"]),
        expected_snapshot_sha256=str(model_resolution["sha256"]),
        model_config=model_config,
        model_state=checkpoint["model_state"],
    )
    source_provenance["anchor_disabled_forward_compatibility"] = compatibility
    policy_document = json.loads(
        (output_dir / "fallback_policy.json").read_text(encoding="utf-8")
    )
    policy = FallbackPolicy(**policy_document["policy"])
    return scaler, checkpoint, policy, lock, source_provenance


def _save_predictions(output_dir: Path, name: str, prediction: Predictions, metrics: Mapping[str, Any]) -> None:
    if prediction.final_prediction is None or prediction.applied_pull is None:
        raise RuntimeError("policy must be applied before saving predictions")
    arrays: dict[str, np.ndarray] = {
        "position": prediction.position,
        "cache_index": prediction.cache_index,
        "target_rr_bpm": prediction.target,
        "identity": prediction.identity.astype("U"),
        # In i3 these are deliberately named twice: ``fallback`` preserves the
        # stable policy schema, while ``raw_anchor`` makes it impossible to
        # confuse the uncorrected baseline with the model's corrected source.
        "fallback_rr_bpm": prediction.base_prediction,
        "fallback_std_bpm": prediction.base_std,
        "fallback_available": prediction.base_available,
        "source_rr_bpm": prediction.source_prediction,
        "source_scale_bpm": prediction.source_scale,
        "source_available": prediction.source_available,
        "selected_index": prediction.selected_index,
        "selected_probability": prediction.selected_probability,
        "margin": prediction.margin,
        "entropy": prediction.entropy,
        "normalized_entropy": (
            prediction.normalized_entropy
            if prediction.normalized_entropy is not None
            else _prediction_policy_context(prediction)[0]
        ),
        "valid_candidate_count": (
            prediction.valid_candidate_count
            if prediction.valid_candidate_count is not None
            else _prediction_policy_context(prediction)[1]
        ),
        "quality": prediction.quality,
        "spike_rate": prediction.spike_rate,
        "applied_pull": prediction.applied_pull,
        "final_rr_bpm": prediction.final_prediction,
    }
    optional_arrays = {
        "raw_anchor_rr_bpm": prediction.raw_anchor_prediction,
        "raw_anchor_std_bpm": prediction.raw_anchor_std,
        "corrected_anchor_rr_bpm": prediction.corrected_anchor_prediction,
        "anchor_residual_bpm": prediction.anchor_residual,
        "anchor_snap_gate": prediction.anchor_snap_gate,
        "candidate_source_rr_bpm": prediction.candidate_source_prediction,
    }
    arrays.update(
        {
            key: np.asarray(value)
            for key, value in optional_arrays.items()
            if value is not None
        }
    )
    atomic_save_npz(output_dir / f"{name}_predictions.npz", **arrays)
    metrics_path = output_dir / f"{name}_metrics.json"
    atomic_write_json(metrics_path, metrics)
    metrics_path.chmod(0o444)


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(int(args.seed), bool(args.deterministic))
    anchor_enabled = str(args.anchor_residual_mode) == "causal_posterior"
    objective = resolve_iteration_objective(
        int(args.adaptive_iteration),
        warmup_windows=args.warmup_windows,
        gradient_accumulation_sessions=args.gradient_accumulation_sessions,
        tail_weight=args.tail_weight,
        cvar_weight=args.cvar_weight,
        anchor_enabled=anchor_enabled,
        anchor_residual_weight=args.anchor_residual_weight,
        anchor_nll_weight=args.anchor_nll_weight,
        anchor_gate_weight=args.anchor_gate_weight,
    )
    output_dir = args.output_dir.expanduser().resolve()
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not args.resume
        and not args.recover_prelock
    ):
        raise RuntimeError(
            "output directory is non-empty; use a new directory, --resume, or --recover-prelock"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = load_experiment(args.cache, args.fallback_oof)
    train_positions, validation_positions, validation_fold = split_positions(experiment.metadata, int(args.fold))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint_path = output_dir / "best_checkpoint.pt"
    scaler_path = output_dir / "scaler.json"
    policy_path = output_dir / "fallback_policy.json"
    lock_path = output_dir / "selection_lock.json"
    model_config = _model_configuration(
        args.preset,
        experiment.node_features.shape[-1],
        adaptive_iteration=int(args.adaptive_iteration),
        anchor_residual_mode=str(args.anchor_residual_mode),
        anchor_max_residual_bpm=float(args.anchor_max_residual_bpm),
        anchor_minimum_scale_bpm=float(args.anchor_minimum_scale_bpm),
        anchor_maximum_scale_bpm=float(args.anchor_maximum_scale_bpm),
        anchor_initial_scale_bpm=float(args.anchor_initial_scale_bpm),
        anchor_distance_weight=float(args.anchor_distance_weight),
        anchor_source_mode=str(args.anchor_source_mode),
    )
    model = HarmonicCandidateSetEpisodeSNN(**model_config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > int(args.maximum_parameters):
        raise RuntimeError(
            "model parameter limit exceeded: "
            f"{parameter_count} > {int(args.maximum_parameters)}"
        )
    cache_manifest_sha256 = sha256_file(experiment.root / "manifest.json")
    fallback_oof_sha256 = sha256_file(args.fallback_oof)
    source_bindings = _source_bindings()
    effective_configuration = _effective_configuration(
        args,
        objective,
        model_config,
        validation_fold=validation_fold,
        cache_manifest_sha256=cache_manifest_sha256,
        fallback_oof_sha256=fallback_oof_sha256,
    )
    if int(args.adaptive_iteration) == 3:
        effective_configuration["model_capacity"] = {
            "parameter_count": int(parameter_count),
            "maximum_parameters": int(args.maximum_parameters),
            "hard_limit_enforced_before_training": True,
        }
    effective_configuration_sha256 = sha256_json(effective_configuration)
    scaler = fit_robust_scaler(experiment, train_positions)
    model = model.to(device)
    recovered_prelock = False

    if args.resume:
        lock = _validate_lock(output_dir)
        if int(lock.get("adaptive_iteration", args.adaptive_iteration)) != int(args.adaptive_iteration):
            raise RuntimeError("resume adaptive iteration disagrees with selection lock")
        locked_effective_sha = lock.get("effective_configuration_sha256")
        if (
            locked_effective_sha is not None
            and str(locked_effective_sha) != effective_configuration_sha256
        ):
            raise RuntimeError("resume effective configuration disagrees with selection lock")
        scaler_document = json.loads(scaler_path.read_text(encoding="utf-8"))
        scaler = RobustNodeScaler(
            center=np.asarray(scaler_document["center"], dtype=np.float32).reshape(1, 1, -1),
            scale=np.asarray(scaler_document["scale"], dtype=np.float32).reshape(1, 1, -1),
            fit_positions_sha256=str(scaler_document["fit_positions_sha256"]),
        )
        if scaler.fit_positions_sha256 != _positions_sha256(train_positions):
            raise RuntimeError("resume scaler was fitted on a different training split")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("model_config") != model_config:
            raise RuntimeError("resume model configuration disagrees with selection lock")
        model.load_state_dict(checkpoint["model_state"])
        policy = FallbackPolicy(**json.loads(policy_path.read_text(encoding="utf-8"))["policy"])
    elif args.recover_prelock and lock_path.exists():
        (
            scaler,
            checkpoint,
            policy,
            lock,
            recovery_source_provenance,
        ) = _validate_locked_recovery(
            output_dir,
            args=args,
            experiment=experiment,
            train_positions=train_positions,
            validation_fold=validation_fold,
            model_config=model_config,
            objective=objective,
            effective_configuration_sha256=effective_configuration_sha256,
            current_source_bindings=source_bindings,
            cache_manifest_sha256=cache_manifest_sha256,
            fallback_oof_sha256=fallback_oof_sha256,
            device=device,
            snapshot_root=args.recovery_source_snapshot_root,
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        validation = predict_positions(
            model,
            experiment,
            validation_positions,
            scaler,
            device,
            amp=bool(args.amp),
        )
        locked_validation = apply_fallback_policy(validation, policy)
        validation_metrics = {
            "source": evaluation_metrics(
                locked_validation.target,
                locked_validation.source_prediction,
                locked_validation.identity,
            ),
            "fallback": evaluation_metrics(
                locked_validation.target,
                np.where(
                    locked_validation.base_available,
                    locked_validation.base_prediction,
                    locked_validation.source_prediction,
                ),
                locked_validation.identity,
            ),
            "locked_final": evaluation_metrics(
                locked_validation.target,
                locked_validation.final_prediction,
                locked_validation.identity,
            ),
        }
        _save_predictions(
            output_dir, "validation", locked_validation, validation_metrics
        )
        write_recovery_provenance(
            output_dir,
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "historical_lock_validation_materialization",
                "historical_artifacts_rewritten": [],
                "new_artifacts": [
                    "validation_predictions.npz", "validation_metrics.json"
                ],
                "source_resolution": recovery_source_provenance,
                "historical_lock_sha256": sha256_file(lock_path),
                "historical_policy_sha256": sha256_file(policy_path),
                "historical_checkpoint_sha256": sha256_file(checkpoint_path),
                "validation_predictions_sha256": sha256_file(
                    output_dir / "validation_predictions.npz"
                ),
                "validation_metrics_sha256": sha256_file(
                    output_dir / "validation_metrics.json"
                ),
                "outer_test_constructed": False,
            },
        )
        recovered_prelock = True
    elif args.recover_prelock:
        (
            scaler,
            checkpoint,
            _,
            recovery_source_bindings,
            recovery_source_provenance,
        ) = _validate_prelock_recovery(
            output_dir,
            args=args,
            experiment=experiment,
            train_positions=train_positions,
            validation_fold=validation_fold,
            model_config=model_config,
            objective=objective,
            effective_configuration_sha256=effective_configuration_sha256,
            source_bindings=source_bindings,
            cache_manifest_sha256=cache_manifest_sha256,
            fallback_oof_sha256=fallback_oof_sha256,
            device=device,
            snapshot_root=args.recovery_source_snapshot_root,
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        validation = predict_positions(
            model,
            experiment,
            validation_positions,
            scaler,
            device,
            amp=bool(args.amp),
        )
        policy, locked_validation = select_fallback_policy(
            validation,
            maximum_coverage=float(args.maximum_coverage),
            maximum_fpr=float(args.maximum_fpr),
            minimum_precision=float(args.minimum_precision),
            minimum_correction_recall=float(args.minimum_correction_recall),
            gate_aware=int(args.adaptive_iteration) == 3,
        )
        atomic_write_json(
            policy_path,
            {
                "schema_version": SCHEMA_VERSION,
                "policy": asdict(policy),
                "selection_scope": "outer-validation valid references only",
                "recovered_prelock": True,
            },
        )
        lock = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "retrospective_only": True,
            "outer_test_not_opened_before_this_lock": True,
            "recovered_prelock": True,
            "outer_fold": int(args.fold),
            "validation_fold": validation_fold,
            "training_folds": sorted(
                set(experiment.metadata.iloc[train_positions]["fold"].astype(int))
            ),
            "seed": int(args.seed),
            "adaptive_iteration": int(args.adaptive_iteration),
            "iteration_objective": iteration_objective_record(objective),
            "effective_configuration_sha256": effective_configuration_sha256,
            "source_bindings": recovery_source_bindings,
            "best_epoch": int(checkpoint["epoch"]),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "scaler_sha256": sha256_file(scaler_path),
            "policy_sha256": sha256_file(policy_path),
            "run_manifest_sha256": sha256_file(output_dir / "run_manifest.json"),
            "history_sha256": sha256_file(output_dir / "history.json"),
            "cache_manifest_sha256": cache_manifest_sha256,
            "fallback_oof_sha256": fallback_oof_sha256,
            "policy_selection_status": policy.selection_status,
            "promotion_eligible": policy.promotion_eligible,
            "checkpoint_selection_objective": checkpoint.get(
                "selection_objective", "legacy_identity_macro_mae"
            ),
            "policy_selection_objective": policy.selection_objective,
            "test_access_policy": "construct iterator only after atomic lock",
        }
        atomic_write_json(lock_path, lock)
        _validate_lock(output_dir)
        validation_metrics = {
            "source": evaluation_metrics(
                locked_validation.target,
                locked_validation.source_prediction,
                locked_validation.identity,
            ),
            "fallback": evaluation_metrics(
                locked_validation.target,
                np.where(
                    locked_validation.base_available,
                    locked_validation.base_prediction,
                    locked_validation.source_prediction,
                ),
                locked_validation.identity,
            ),
            "locked_final": evaluation_metrics(
                locked_validation.target,
                locked_validation.final_prediction,
                locked_validation.identity,
            ),
        }
        _save_predictions(
            output_dir, "validation", locked_validation, validation_metrics
        )
        recovery_sidecar = write_recovery_provenance(
            output_dir,
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "prelock_selection_recovery",
                "historical_artifacts_rewritten": [],
                "new_artifacts": [
                    "fallback_policy.json", "selection_lock.json",
                    "validation_predictions.npz", "validation_metrics.json",
                ],
                "source_resolution": recovery_source_provenance,
                "run_manifest_sha256": sha256_file(output_dir / "run_manifest.json"),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "history_sha256": sha256_file(output_dir / "history.json"),
                "selection_lock_sha256": sha256_file(lock_path),
            },
        )
        recovered_prelock = True
    else:
        identities = experiment.metadata["identity"].astype(str).to_numpy()
        folds = experiment.metadata["fold"].to_numpy(np.int64)
        references = experiment.metadata["reference_valid"].astype(bool).to_numpy()
        run_manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "retrospective_only": True,
            "commercial_claim_authorized": False,
            "outer_fold": int(args.fold),
            "validation_fold": validation_fold,
            "training_folds": sorted(set(map(int, folds[train_positions]))),
            "training_identities": sorted(set(identities[train_positions])),
            "validation_identities": sorted(set(identities[validation_positions])),
            "test_identities_declared_but_not_iterated": sorted(set(identities[folds == int(args.fold)])),
            "rows": {
                "training_all_windows": int(len(train_positions)),
                "training_valid_references": int(references[train_positions].sum()),
                "validation_all_windows": int(len(validation_positions)),
                "validation_valid_references": int(references[validation_positions].sum()),
            },
            "model_config": model_config,
            "iteration_effective_configuration": effective_configuration,
            "iteration_effective_configuration_sha256": effective_configuration_sha256,
            "optimization": {
                "seed": int(args.seed), "epochs": int(args.epochs),
                "minimum_epochs": int(args.minimum_epochs), "patience": int(args.patience),
                "learning_rate": float(args.learning_rate), "amp": bool(args.amp),
                "deterministic": bool(args.deterministic),
                "adaptive_iteration": int(args.adaptive_iteration),
                "iteration_objective": iteration_objective_record(objective),
                "tail_weight": float(objective.tail_weight),
                "cvar_weight": float(objective.cvar_weight),
                "anchor_residual_weight": float(objective.anchor_residual_weight),
                "anchor_nll_weight": float(objective.anchor_nll_weight),
                "anchor_gate_weight": float(objective.anchor_gate_weight),
                "chunk_size": int(args.chunk_windows),
                "warmup_windows": int(objective.warmup_windows),
                "gradient_accumulation_sessions": int(objective.gradient_accumulation_sessions),
            },
            "input_bindings": {
                "cache_manifest_path": str(experiment.root / "manifest.json"),
                "cache_manifest_sha256": cache_manifest_sha256,
                "fallback_oof_path": str(args.fallback_oof.expanduser().resolve()),
                "fallback_oof_sha256": fallback_oof_sha256,
                "fallback_semantics": (
                    "cache-index-bound raw posterior anchor and final fallback"
                    if anchor_enabled
                    else "post-forward fallback and loss context"
                ),
                "trainer_sha256": source_bindings["trainer"]["sha256"],
                "model_source_path": source_bindings["harmonic_set_model"]["path"],
                "model_source_sha256": source_bindings["harmonic_set_model"]["sha256"],
                "spiking_cell_source_path": source_bindings["spiking_cell_model"]["path"],
                "spiking_cell_source_sha256": source_bindings["spiking_cell_model"]["sha256"],
            },
            "source_and_config_bindings": source_bindings,
            "leakage_boundary": {
                "forward_inputs": effective_configuration["forward_allowlist"],
                "identity_session_protocol_fold_are_grouping_only": True,
                "reference_and_reference_qc_are_loss_or_evaluation_only": True,
                "fallback_prediction_and_std_are_loss_context_and_post_forward_policy_only": not anchor_enabled,
                "i3_anchor_is_same_strict_nested_fallback_not_reference": anchor_enabled,
                "no_future_anchor_lags": True,
                "outer_test_iterator_before_atomic_lock": False,
            },
        }
        atomic_write_json(output_dir / "run_manifest.json", run_manifest)
        atomic_write_json(scaler_path, scaler.record())
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1.0e-4)
        gradient_scaler = torch.amp.GradScaler(device.type, enabled=bool(args.amp) and device.type == "cuda")
        row_weights = torch.as_tensor(identity_balanced_weights(experiment.metadata, train_positions), device=device)
        best_macro = math.inf
        best_selection_key: tuple[Any, ...] = (math.inf,) * 5
        best_epoch = -1
        stale = 0
        history: list[dict[str, Any]] = []
        for epoch in range(int(args.epochs)):
            losses = run_training_epoch(
                model, experiment, train_positions, scaler, optimizer,
                row_weights, device, amp=bool(args.amp),
                gradient_scaler=gradient_scaler, seed=int(args.seed), epoch=epoch,
                adaptive_iteration=int(args.adaptive_iteration),
                tail_weight=float(objective.tail_weight),
                cvar_weight=float(objective.cvar_weight),
                objective=objective,
                gradient_accumulation_sessions=int(objective.gradient_accumulation_sessions),
                warmup_windows=int(objective.warmup_windows),
                chunk_size=int(args.chunk_windows),
            )
            validation = predict_positions(model, experiment, validation_positions, scaler, device, amp=bool(args.amp))
            source_metrics = evaluation_metrics(validation.target, validation.source_prediction, validation.identity)
            macro = float(source_metrics["identity_macro_mae"])
            selection_key = (
                commercial_gate_selection_key(source_metrics)
                if int(args.adaptive_iteration) == 3
                else (macro,)
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train": losses,
                    "validation_source": source_metrics,
                    "retrospective_selection_key": selection_key,
                    "selection_objective": (
                        COMMERCIAL_SELECTION_OBJECTIVE
                        if int(args.adaptive_iteration) == 3
                        else "legacy_identity_macro_mae"
                    ),
                }
            )
            atomic_write_json(output_dir / "history.json", history)
            improved = (
                selection_key < best_selection_key
                if int(args.adaptive_iteration) == 3
                else macro < best_macro - 1.0e-8
            )
            if improved:
                best_macro = macro
                best_selection_key = selection_key
                best_epoch, stale = epoch + 1, 0
                atomic_torch_save(checkpoint_path, {
                    "schema_version": SCHEMA_VERSION,
                    "model_state": model.state_dict(), "model_config": model_config,
                    "epoch": best_epoch, "validation_source_macro_mae": best_macro,
                    "seed": int(args.seed), "fold": int(args.fold),
                    "adaptive_iteration": int(args.adaptive_iteration),
                    "iteration_objective": iteration_objective_record(objective),
                    "effective_configuration_sha256": effective_configuration_sha256,
                    "validation_selection_key": selection_key,
                    "selection_objective": (
                        COMMERCIAL_SELECTION_OBJECTIVE
                        if int(args.adaptive_iteration) == 3
                        else "legacy_identity_macro_mae"
                    ),
                })
            else:
                stale += 1
            if epoch + 1 >= int(args.minimum_epochs) and stale >= int(args.patience):
                break
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        validation = predict_positions(model, experiment, validation_positions, scaler, device, amp=bool(args.amp))
        policy, locked_validation = select_fallback_policy(
            validation,
            maximum_coverage=float(args.maximum_coverage),
            maximum_fpr=float(args.maximum_fpr),
            minimum_precision=float(args.minimum_precision),
            minimum_correction_recall=float(args.minimum_correction_recall),
            gate_aware=int(args.adaptive_iteration) == 3,
        )
        atomic_write_json(policy_path, {"schema_version": SCHEMA_VERSION, "policy": asdict(policy), "selection_scope": "outer-validation valid references only"})
        lock = {
            "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "retrospective_only": True, "outer_test_not_opened_before_this_lock": True,
            "outer_fold": int(args.fold), "validation_fold": validation_fold,
            "training_folds": sorted(set(experiment.metadata.iloc[train_positions]["fold"].astype(int))),
            "seed": int(args.seed), "adaptive_iteration": int(args.adaptive_iteration),
            "iteration_objective": iteration_objective_record(objective),
            "effective_configuration_sha256": effective_configuration_sha256,
            "source_bindings": source_bindings,
            "best_epoch": int(checkpoint["epoch"]),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "scaler_sha256": sha256_file(scaler_path), "policy_sha256": sha256_file(policy_path),
            "run_manifest_sha256": sha256_file(output_dir / "run_manifest.json"),
            "history_sha256": sha256_file(output_dir / "history.json"),
            "cache_manifest_sha256": cache_manifest_sha256,
            "fallback_oof_sha256": fallback_oof_sha256,
            "policy_selection_status": policy.selection_status,
            "promotion_eligible": policy.promotion_eligible,
            "checkpoint_selection_objective": checkpoint.get(
                "selection_objective", "legacy_identity_macro_mae"
            ),
            "policy_selection_objective": policy.selection_objective,
            "test_access_policy": "construct iterator only after atomic lock",
        }
        # This atomic replace is the one-way gate.  No test positions/iterator
        # have been constructed above this line.
        atomic_write_json(lock_path, lock)
        _validate_lock(output_dir)
        validation_metrics = {
            "source": evaluation_metrics(locked_validation.target, locked_validation.source_prediction, locked_validation.identity),
            "fallback": evaluation_metrics(locked_validation.target, np.where(locked_validation.base_available, locked_validation.base_prediction, locked_validation.source_prediction), locked_validation.identity),
            "locked_final": evaluation_metrics(locked_validation.target, locked_validation.final_prediction, locked_validation.identity),
        }
        _save_predictions(output_dir, "validation", locked_validation, validation_metrics)

    result: dict[str, Any] = {
        "status": "locked",
        "output_dir": str(output_dir),
        "selection_lock": lock,
        "discovery_only": bool(args.discovery_only),
        "recovered_prelock": recovered_prelock,
    }
    # Recovery is intentionally a selection-only operation.  Even when the
    # caller omits --discovery-only it cannot construct or iterate outer test.
    if recovered_prelock:
        return result
    if not args.discovery_only:
        if not lock_path.is_file():
            raise RuntimeError("refusing to construct outer-test iterator before selection lock")
        _validate_lock(output_dir)
        # Exactly-once immutable test opening.  Existing results are never replaced.
        test_npz = output_dir / "test_predictions.npz"
        if test_npz.exists():
            if not args.resume:
                raise RuntimeError("outer-test result already exists")
            result["test_status"] = "already_evaluated"
        else:
            test_positions = np.flatnonzero(experiment.metadata["fold"].to_numpy(np.int64) == int(args.fold))
            test_prediction = predict_positions(model, experiment, test_positions, scaler, device, amp=bool(args.amp))
            locked_test = apply_fallback_policy(test_prediction, policy)
            test_metrics = {
                "source": evaluation_metrics(locked_test.target, locked_test.source_prediction, locked_test.identity),
                "fallback": evaluation_metrics(locked_test.target, np.where(locked_test.base_available, locked_test.base_prediction, locked_test.source_prediction), locked_test.identity),
                "locked_final": evaluation_metrics(locked_test.target, locked_test.final_prediction, locked_test.identity),
            }
            _save_predictions(output_dir, "test", locked_test, test_metrics)
            result["test_status"] = "evaluated_once"
            result["test_metrics"] = test_metrics
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fallback-oof", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(N_FOLDS))
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="default")
    parser.add_argument("--maximum-parameters", type=int, default=750_000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--adaptive-iteration", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--anchor-residual-mode",
        choices=("disabled", "causal_posterior"),
        default=None,
        help="source architecture (iteration default: i1/i2=disabled, i3=causal_posterior)",
    )
    parser.add_argument("--anchor-max-residual-bpm", type=float, default=12.0)
    parser.add_argument("--anchor-minimum-scale-bpm", type=float, default=0.25)
    parser.add_argument("--anchor-maximum-scale-bpm", type=float, default=12.0)
    parser.add_argument("--anchor-initial-scale-bpm", type=float, default=1.5)
    parser.add_argument("--anchor-distance-weight", type=float, default=1.0)
    parser.add_argument(
        "--anchor-source-mode",
        choices=("corrected_anchor", "learned_blend"),
        default="learned_blend",
    )
    parser.add_argument(
        "--anchor-residual-weight", type=float, default=None,
        help="i3 robust clipped-residual weight (default 0.75)",
    )
    parser.add_argument(
        "--anchor-nll-weight", type=float, default=None,
        help="i3 posterior residual NLL weight (default 0.20)",
    )
    parser.add_argument(
        "--anchor-gate-weight", type=float, default=None,
        help="i3 candidate-snap gate weight (default 0.08)",
    )
    parser.add_argument(
        "--tail-weight", type=float, default=None,
        help="tail multiplier (iteration default: i1/i2=0, i3=2)",
    )
    parser.add_argument(
        "--cvar-weight", type=float, default=None,
        help="top-20%% weighted regret coefficient (iteration default: i1/i2=0, i3=0.15)",
    )
    parser.add_argument(
        "--warmup-windows", type=int, default=None,
        help="state-only windows per session (iteration default: i1=8, i2/i3=2; 0 is allowed)",
    )
    parser.add_argument(
        "--gradient-accumulation-sessions", type=int, default=None,
        help="complete identity-round-robin sessions per optimizer step (i1=1, i2/i3=4)",
    )
    parser.add_argument("--chunk-windows", type=int, default=CHUNK_SIZE)
    parser.add_argument("--maximum-coverage", type=float, default=0.15)
    parser.add_argument("--maximum-fpr", type=float, default=0.10)
    parser.add_argument("--minimum-precision", type=float, default=0.60)
    parser.add_argument("--minimum-correction-recall", type=float, default=0.0)
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--recover-prelock",
        action="store_true",
        help="validate and lock a completed training run that failed before selection lock; never opens outer test",
    )
    parser.add_argument(
        "--recovery-source-snapshot-root",
        type=Path,
        default=None,
        help="hash-bound source snapshots used only when historical source paths have changed",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.resume and args.recover_prelock:
        raise SystemExit("--resume and --recover-prelock are mutually exclusive")
    if args.anchor_residual_mode is None:
        args.anchor_residual_mode = (
            "causal_posterior" if args.adaptive_iteration == 3 else "disabled"
        )
    if args.anchor_residual_mode == "causal_posterior" and args.adaptive_iteration != 3:
        raise SystemExit("causal_posterior anchor mode is valid only for iteration 3")
    if args.minimum_epochs < 1 or args.epochs < args.minimum_epochs or args.patience < 1:
        raise SystemExit("epochs/minimum-epochs/patience are inconsistent")
    if args.maximum_parameters < 1:
        raise SystemExit("maximum-parameters must be positive")
    if (
        args.learning_rate <= 0
        or (args.tail_weight is not None and args.tail_weight < 0)
        or (args.cvar_weight is not None and args.cvar_weight < 0)
        or any(
            value is not None and value < 0
            for value in (
                args.anchor_residual_weight,
                args.anchor_nll_weight,
                args.anchor_gate_weight,
            )
        )
    ):
        raise SystemExit(
            "learning/tail/CVaR/anchor weights must be non-negative (learning positive)"
        )
    if not (
        0 < args.anchor_max_residual_bpm <= 12.0
        and 0 < args.anchor_minimum_scale_bpm
        < args.anchor_initial_scale_bpm
        <= args.anchor_maximum_scale_bpm
        and args.anchor_distance_weight >= 0
    ):
        raise SystemExit("anchor residual/scale/distance settings are inconsistent")
    if (
        (args.warmup_windows is not None and args.warmup_windows < 0)
        or (args.gradient_accumulation_sessions is not None and args.gradient_accumulation_sessions < 1)
        or args.chunk_windows < 1
    ):
        raise SystemExit("warmup/chunk/gradient accumulation settings are invalid")
    for value, name in (
        (args.maximum_coverage, "maximum coverage"),
        (args.maximum_fpr, "maximum FPR"),
        (args.minimum_precision, "minimum precision"),
        (args.minimum_correction_recall, "minimum correction recall"),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must lie in [0, 1]")
    # Resolve now so invalid iteration-specific combinations fail before any
    # artifact directory is touched.
    resolve_iteration_objective(
        args.adaptive_iteration,
        warmup_windows=args.warmup_windows,
        gradient_accumulation_sessions=args.gradient_accumulation_sessions,
        tail_weight=args.tail_weight,
        cvar_weight=args.cvar_weight,
        anchor_enabled=args.anchor_residual_mode == "causal_posterior",
        anchor_residual_weight=args.anchor_residual_weight,
        anchor_nll_weight=args.anchor_nll_weight,
        anchor_gate_weight=args.anchor_gate_weight,
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = train(parse_args(argv))
    print(json.dumps(_json_ready(result), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
