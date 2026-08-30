#!/usr/bin/env python3
"""Grouped ANN-teacher -> SNN training and out-of-fold evaluation.

The unit of generalisation in this project is a real person, not a heavily
overlapping window.  This entry point therefore assigns every identity to one
of six outer folds, uses a different identity fold for early stopping, and
never fits the auxiliary-feature scaler outside the remaining training
identities.

Examples
--------
Full six-fold teacher and distilled SNN run::

    python scripts/train.py --model both

Historical-cache reproduction (always classified noncommercial)::

    python scripts/train.py --model both --cache-trust-mode legacy

One-fold smoke run on CPU::

    python scripts/train.py --fold 0 --model teacher --epochs 1 \
        --preset tiny --device cpu --batch-size 2 \
        --max-train-batches 1 --max-eval-batches 2 \
        --output-dir /tmp/snn_rr_smoke

The output directory contains best/last checkpoints per fold, fold-level test
predictions, aggregated OOF NPZ/CSV files, and strict JSON metric reports.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml

# Make ``python scripts/train.py`` work without requiring an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_rr.cache import (  # noqa: E402
    ACQUISITION_CACHE_SCHEMA_VERSION_V2,
    FeatureCache,
    append_causal_history_features,
    fit_aux_scaler,
    load_feature_cache,
    transform_aux,
)
from snn_rr.metrics import (  # noqa: E402
    clustered_bootstrap_mae,
    identity_macro_metrics,
    regression_metrics,
    risk_coverage_curve,
)
from snn_rr.models import (  # noqa: E402
    SharedRadarCNNTeacher,
    TriRadarRRSNN,
    apply_radar_dropout,
    count_trainable_parameters,
    gaussian_soft_targets,
)
from snn_rr.split_authority import (  # noqa: E402
    IdentitySplitAuthority,
    load_identity_split_authority,
)


@dataclass(slots=True)
class PredictionBundle:
    """Model predictions with global cache row indices."""

    index: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    rr_std: np.ndarray
    uncertainty: np.ndarray
    quality: np.ndarray
    observable: np.ndarray
    reference_valid: np.ndarray
    spike_rate: np.ndarray
    radar_weights: np.ndarray
    map_prediction: np.ndarray | None = None
    posterior_entropy: np.ndarray | None = None
    topk_rr: np.ndarray | None = None
    topk_probability: np.ndarray | None = None
    posterior_probability: np.ndarray | None = None
    alias_probability: np.ndarray | None = None

    def __post_init__(self) -> None:
        count = len(self.index)
        prediction = np.asarray(self.prediction)
        if self.map_prediction is None:
            self.map_prediction = prediction.copy()
        if self.posterior_entropy is None:
            self.posterior_entropy = np.full(count, np.nan, dtype=np.float32)
        if self.topk_rr is None:
            self.topk_rr = np.full((count, 5), np.nan, dtype=np.float32)
            self.topk_rr[:, 0] = prediction
        if self.topk_probability is None:
            self.topk_probability = np.zeros((count, 5), dtype=np.float32)
            self.topk_probability[:, 0] = 1.0
        if self.posterior_probability is None:
            self.posterior_probability = np.empty((count, 0), dtype=np.float16)
        if self.alias_probability is None:
            self.alias_probability = np.full(count, np.nan, dtype=np.float32)

        if np.asarray(self.map_prediction).shape != (count,):
            raise ValueError("map_prediction must have one value per row")
        if np.asarray(self.posterior_entropy).shape != (count,):
            raise ValueError("posterior_entropy must have one value per row")
        topk_rr = np.asarray(self.topk_rr)
        topk_probability = np.asarray(self.topk_probability)
        if (
            topk_rr.ndim != 2
            or topk_probability.shape != topk_rr.shape
            or topk_rr.shape[0] != count
        ):
            raise ValueError("top-k posterior arrays must have matching [row, rank] shape")
        posterior_probability = np.asarray(self.posterior_probability)
        if posterior_probability.ndim != 2 or posterior_probability.shape[0] != count:
            raise ValueError("posterior_probability must have [row, RR-bin] shape")
        if np.asarray(self.alias_probability).shape != (count,):
            raise ValueError("alias_probability must have one value per row")

    def __len__(self) -> int:
        return len(self.index)


@dataclass(frozen=True, slots=True)
class AuxiliaryLayout:
    """Known layout of ``fuse_auxiliary_features`` before causal history."""

    base_dim: int
    frequency_bins: int


def infer_auxiliary_layout(base_dim: int) -> AuxiliaryLayout:
    """Infer the full-resolution frequency count from the cache schema.

    The base vector is ``3 * (2*F spectra + 8 scalars) + 2*F fused + 5``.
    Causal-history columns, if enabled, are appended after ``base_dim``.
    """

    if base_dim < 37 or (base_dim - 29) % 8:
        raise ValueError(
            f"aux dimension {base_dim} does not match the three-radar cache schema"
        )
    return AuxiliaryLayout(base_dim=base_dim, frequency_bins=(base_dim - 29) // 8)


def infer_auxiliary_frequency_range(
    pooled_frequencies_hz: np.ndarray,
    frequency_bins: int,
) -> tuple[float, float]:
    """Recover the full-resolution auxiliary FFT grid from the pooled map grid.

    Feature construction stores pair-averaged frequency coordinates for the
    range--frequency map while retaining the original full-resolution spectra
    in ``aux``.  The latter contains either twice as many bins or one trailing
    unpaired bin in addition.  Recovering its physical endpoints avoids a
    systematic half-bin error in harmonic candidate sampling.
    """

    pooled = np.asarray(pooled_frequencies_hz, dtype=np.float64)
    if pooled.ndim != 1 or len(pooled) < 2 or not np.isfinite(pooled).all():
        raise ValueError("pooled frequency grid must contain finite values")
    differences = np.diff(pooled)
    if np.any(differences <= 0):
        raise ValueError("pooled frequency grid must be strictly increasing")
    pooled_step = float(np.median(differences))
    if not np.allclose(differences, pooled_step, rtol=1e-4, atol=1e-8):
        raise ValueError("pooled frequency grid must be uniformly spaced")
    if frequency_bins not in {2 * len(pooled), 2 * len(pooled) + 1}:
        raise ValueError(
            "auxiliary frequency count is inconsistent with pair-pooled map grid"
        )
    full_step = pooled_step / 2.0
    full_min = float(pooled[0] - 0.5 * full_step)
    full_max = float(full_min + (int(frequency_bins) - 1) * full_step)
    return full_min, full_max


def apply_coupled_radar_dropout(
    radar_map: Tensor,
    radar_mask: Tensor,
    aux: Tensor,
    *,
    p: float,
    layout: AuxiliaryLayout | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Drop map views and their current-window auxiliary features together.

    The models' internal dropout cannot see the flattened auxiliary vector.
    Applying the augmentation here prevents per-radar spectra or fused current
    features from leaking a view that was removed from the map.  Strictly
    causal history is retained because it genuinely predates the simulated
    transient sensor loss.
    """

    dropped_map, kept = apply_radar_dropout(
        radar_map,
        radar_mask,
        p=p,
        training=True,
        ensure_one=True,
    )
    if aux.shape[1] == 0 or layout is None:
        return dropped_map, kept, aux
    if aux.shape[1] < layout.base_dim:
        raise ValueError("aux tensor is shorter than its declared base layout")
    masked_aux = aux.clone()
    unavailable = ~kept
    frequency_bins = layout.frequency_bins
    spectra_per_radar = 2 * frequency_bins
    scalar_start = 3 * spectra_per_radar
    for radar in range(3):
        rows = unavailable[:, radar]
        if rows.any():
            masked_aux[
                rows,
                radar * spectra_per_radar : (radar + 1) * spectra_per_radar,
            ] = 0.0
            masked_aux[
                rows,
                scalar_start + radar * 8 : scalar_start + (radar + 1) * 8,
            ] = 0.0
    # Fused spectra and consensus statistics encode all current radars, so a
    # partial mask invalidates them.  Zero is the train-median neutral value
    # after robust scaling.
    any_unavailable = unavailable.any(dim=1)
    fused_start = scalar_start + 3 * 8
    if any_unavailable.any():
        masked_aux[any_unavailable, fused_start : layout.base_dim] = 0.0
    return dropped_map, kept, masked_aux


def append_mask_aware_causal_history_features(
    cache: FeatureCache,
) -> tuple[np.ndarray, list[str]]:
    """Append history without treating structurally invalid windows as evidence."""

    metadata = cache.metadata
    timing_mask = cache.radar_timing_valid_mask
    if timing_mask is not None:
        timing = np.asarray(timing_mask, dtype=np.bool_)
        if timing.ndim != 3 or timing.shape[:2] != (len(metadata), 3):
            raise ValueError(
                "radar_timing_valid_mask must have shape [row, 3, samples]"
            )
        # Classical RR/confidence/spread fuse the current radar views.  If any
        # required interval of any view is invalid, none of those fused values
        # may become a later window's apparently available history feature.
        complete_window = timing.all(axis=(1, 2))
        if not bool(complete_window.all()):
            metadata = metadata.copy()
            metadata.loc[
                ~complete_window,
                [
                    "classical_rr_bpm",
                    "classical_confidence",
                    "radar_peak_spread_bpm",
                ],
            ] = np.nan
    return append_causal_history_features(cache.aux, metadata)


def _mask_structural_radar_inputs(
    radar_map: Tensor,
    aux: Tensor,
    available: Tensor,
    *,
    layout: AuxiliaryLayout | None,
) -> tuple[Tensor, Tensor]:
    """Zero structurally invalid radar cells after all numeric scaling."""

    if radar_map.ndim < 2 or radar_map.shape[0] != 3:
        raise ValueError("cached radar map must have a three-view leading axis")
    if available.shape != (3,):
        raise ValueError("structural radar availability must have shape [3]")
    available = available.to(dtype=torch.bool)
    masked_map = radar_map.clone()
    masked_map[~available] = 0
    masked_aux = aux.clone()
    if bool(available.all()) or masked_aux.numel() == 0:
        return masked_map, masked_aux
    if layout is None:
        # A caller that does not declare the flattened layout cannot safely
        # retain any auxiliary cell after a structural view failure.
        masked_aux.zero_()
        return masked_map, masked_aux
    if masked_aux.ndim != 1 or masked_aux.shape[0] < layout.base_dim:
        raise ValueError("aux tensor is shorter than its declared base layout")
    frequency_bins = layout.frequency_bins
    spectra_per_radar = 2 * frequency_bins
    scalar_start = 3 * spectra_per_radar
    for radar in range(3):
        if not bool(available[radar]):
            masked_aux[
                radar * spectra_per_radar : (radar + 1) * spectra_per_radar
            ] = 0
            masked_aux[
                scalar_start + radar * 8 : scalar_start + (radar + 1) * 8
            ] = 0
    fused_start = scalar_start + 3 * 8
    masked_aux[fused_start : layout.base_dim] = 0
    return masked_map, masked_aux


class CachedRadarDataset(Dataset[dict[str, Any]]):
    """Zero-copy view over cache rows (conversion happens one batch at a time)."""

    def __init__(
        self,
        cache: FeatureCache,
        aux_scaled: np.ndarray,
        indices: Sequence[int] | np.ndarray,
        auxiliary_layout: AuxiliaryLayout | None = None,
    ) -> None:
        self.maps = cache.maps
        self.aux = aux_scaled
        self.indices = np.asarray(indices, dtype=np.int64)
        self.radar_timing_valid_mask = cache.radar_timing_valid_mask
        self.auxiliary_layout = auxiliary_layout
        metadata = cache.metadata
        if self.radar_timing_valid_mask is not None:
            timing = np.asarray(self.radar_timing_valid_mask)
            if (
                timing.dtype != np.bool_
                or timing.ndim != 3
                or timing.shape[0] != len(metadata)
                or timing.shape[1] != self.maps.shape[1]
                or timing.shape[2] <= 0
            ):
                raise ValueError(
                    "radar_timing_valid_mask must be bool [row, radar, samples]"
                )
        self.rr = pd.to_numeric(metadata["rr_bpm"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        self.reference_valid = metadata["reference_valid"].to_numpy(dtype=bool)
        self.reference_quality = pd.to_numeric(
            metadata["reference_quality"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float32)
        self.reference_sigma = pd.to_numeric(
            metadata["reference_sigma_bpm"], errors="coerce"
        ).fillna(1.0).to_numpy(dtype=np.float32)
        self.observable = metadata["radar_observable"].to_numpy(dtype=bool)
        self.classical_rr = pd.to_numeric(
            metadata["classical_rr_bpm"], errors="coerce"
        ).to_numpy(dtype=np.float32)
        self.classical_confidence = pd.to_numeric(
            metadata["classical_confidence"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        # Cache maps are finite float16 arrays.  Keep them in float16 through
        # collation/transfer, then cast once on the target device.
        radar_map = torch.from_numpy(np.array(self.maps[index], copy=True))
        aux = torch.from_numpy(
            np.array(self.aux[index], dtype=np.float32, copy=True)
        )
        if self.radar_timing_valid_mask is None:
            available = torch.ones(radar_map.shape[0], dtype=torch.bool)
        else:
            timing = np.asarray(
                self.radar_timing_valid_mask[index], dtype=np.bool_
            )
            available = torch.from_numpy(np.all(timing, axis=1))
            radar_map, aux = _mask_structural_radar_inputs(
                radar_map,
                aux,
                available,
                layout=self.auxiliary_layout,
            )
        return {
            "map": radar_map,
            "aux": aux,
            "radar_mask": available,
            "rr": torch.tensor(self.rr[index], dtype=torch.float32),
            "reference_valid": torch.tensor(
                self.reference_valid[index], dtype=torch.bool
            ),
            "reference_quality": torch.tensor(
                self.reference_quality[index], dtype=torch.float32
            ),
            "reference_sigma": torch.tensor(
                self.reference_sigma[index], dtype=torch.float32
            ),
            "observable": torch.tensor(self.observable[index], dtype=torch.float32),
            "classical_rr": torch.tensor(
                self.classical_rr[index], dtype=torch.float32
            ),
            "classical_confidence": torch.tensor(
                self.classical_confidence[index], dtype=torch.float32
            ),
            "index": torch.tensor(index, dtype=torch.int64),
        }


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and torch without silently enabling slow kernels."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
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


def determinism_audit(requested: bool) -> dict[str, Any]:
    """Describe the reproducibility policy without overstating bitwise equality.

    ``warn_only=True`` is intentional: some CUDA kernels used by the network
    (notably adaptive-pooling backward in current PyTorch/CUDA builds) do not
    provide a deterministic implementation.  Capturing RNG state improves
    interrupted-run continuity, but cannot turn those kernels into bitwise
    deterministic operators.
    """

    warnings: list[str] = []
    if requested:
        warnings.append(
            "Deterministic algorithms are requested with warn_only=True; "
            "unsupported operators emit a warning and continue."
        )
        warnings.append(
            "CUDA adaptive-pooling backward may lack a deterministic "
            "implementation, so bitwise equality is not guaranteed."
        )
    else:
        warnings.append(
            "--deterministic was not requested, so bitwise equality is not guaranteed."
        )
    return {
        "requested": bool(requested),
        "algorithm_policy": "warn_only" if requested else "default",
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "warn_only_enabled": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "bitwise_not_guaranteed": True,
        "warnings": warnings,
    }


def parse_fold_selection(selection: str, n_splits: int) -> list[int]:
    """Parse ``all``, a single fold, or a comma-separated fold list."""

    if selection.strip().lower() == "all":
        return list(range(n_splits))
    try:
        folds = sorted({int(value.strip()) for value in selection.split(",")})
    except ValueError as exc:
        raise ValueError("--fold must be 'all', an integer, or comma-separated integers") from exc
    if not folds or folds[0] < 0 or folds[-1] >= n_splits:
        raise ValueError(f"folds must be in [0, {n_splits - 1}]")
    return folds


def make_fold_assignments(
    metadata: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign identities using valid-label window counts for fold balancing."""

    if n_splits < 3:
        raise ValueError("at least three folds are required (train/validation/test)")
    identities = metadata["identity"].astype(str).to_numpy()
    valid = metadata["reference_valid"].to_numpy(dtype=bool)
    valid_rows = np.flatnonzero(valid)
    valid_identities = np.unique(identities[valid_rows])
    if len(valid_identities) < n_splits:
        raise ValueError(
            f"only {len(valid_identities)} identities have valid labels; "
            f"cannot construct {n_splits} folds"
        )

    # Recent scikit-learn versions support shuffled GroupKFold.  The fallback
    # remains grouped and deterministic for older environments.
    try:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    except TypeError:  # pragma: no cover - retained for portability
        splitter = GroupKFold(n_splits=n_splits)

    identity_to_fold: dict[str, int] = {}
    dummy = np.zeros(len(valid_rows), dtype=np.float32)
    groups = identities[valid_rows]
    for fold, (_, test_local) in enumerate(splitter.split(dummy, groups=groups)):
        for identity in np.unique(groups[test_local]):
            key = str(identity)
            if key in identity_to_fold:
                raise RuntimeError(f"identity {key} was assigned to multiple folds")
            identity_to_fold[key] = fold
    if set(identity_to_fold) != set(valid_identities):
        raise RuntimeError("some valid-label identities were not assigned to a fold")
    assignment = np.asarray([identity_to_fold.get(name, -1) for name in identities])
    return assignment, identity_to_fold


def identity_balanced_sample_weights(
    metadata: pd.DataFrame,
    indices: Sequence[int] | np.ndarray,
    *,
    valid_boost: float = 2.0,
    rr_balance_power: float = 0.0,
    rr_balance_bin_width: float = 5.0,
) -> np.ndarray:
    """Give each identity equal mass and optionally rebalance rare RR bands."""

    if valid_boost <= 0:
        raise ValueError("valid_boost must be positive")
    if rr_balance_power < 0 or rr_balance_bin_width <= 0:
        raise ValueError("RR balance power must be non-negative and bin width positive")
    indices = np.asarray(indices, dtype=np.int64)
    identity = metadata.iloc[indices]["identity"].astype(str).to_numpy()
    valid = metadata.iloc[indices]["reference_valid"].to_numpy(dtype=bool)
    weights = np.where(valid, valid_boost, 1.0).astype(np.float64)
    if rr_balance_power > 0:
        rr = pd.to_numeric(
            metadata.iloc[indices]["rr_bpm"], errors="coerce"
        ).to_numpy(dtype=float)
        usable = valid & np.isfinite(rr)
        buckets = np.zeros(len(rr), dtype=np.int64)
        buckets[usable] = np.floor(
            rr[usable] / rr_balance_bin_width
        ).astype(np.int64)
        names, counts = np.unique(buckets[usable], return_counts=True)
        if len(names):
            typical = float(np.median(counts))
            count_for_bucket = dict(zip(names.tolist(), counts.tolist(), strict=True))
            rarity = np.ones(len(indices), dtype=np.float64)
            rarity[usable] = np.asarray(
                [
                    (typical / count_for_bucket[int(bucket)]) ** rr_balance_power
                    for bucket in buckets[usable]
                ]
            )
            weights *= np.clip(rarity, 0.25, 4.0)
    for name in np.unique(identity):
        selected = identity == name
        weights[selected] /= weights[selected].sum()
    weights /= weights.mean()
    return weights


def _worker_seed(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    cache: FeatureCache,
    aux_scaled: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
    train: bool,
    valid_boost: float = 2.0,
    rr_balance_power: float = 0.0,
    rr_balance_bin_width: float = 5.0,
    samples_per_epoch: int | None = None,
    auxiliary_layout: AuxiliaryLayout | None = None,
) -> DataLoader[dict[str, Any]]:
    dataset = CachedRadarDataset(
        cache,
        aux_scaled,
        indices,
        auxiliary_layout=auxiliary_layout,
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if train:
        weights = identity_balanced_sample_weights(
            cache.metadata,
            indices,
            valid_boost=valid_boost,
            rr_balance_power=rr_balance_power,
            rr_balance_bin_width=rr_balance_bin_width,
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=samples_per_epoch or len(indices),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=_worker_seed if workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


def _model_kwargs(
    model_type: str,
    args: argparse.Namespace,
    aux_dim: int,
    aux_base_dim: int,
    input_frequency_range_hz: tuple[float, float],
    auxiliary_frequency_range_hz: tuple[float, float] | None = None,
) -> dict[str, Any]:
    rr_min, rr_max = args.rr_range
    num_bins = int(round((rr_max - rr_min) / args.rr_bin_width)) + 1
    structured_auxiliary = args.aux_fusion == "structured" and aux_dim > 0
    harmonic_auxiliary = bool(args.harmonic_head and aux_dim > 0)
    alias_gated_harmonic = bool(args.alias_gate and harmonic_auxiliary)
    common: dict[str, Any] = {
        "num_radars": 3,
        "rr_min": rr_min,
        "rr_max": rr_max,
        "num_rr_bins": num_bins,
        # Coupled map/aux dropout is applied in the training loop.  Keeping the
        # model-local layer disabled avoids a second, aux-unaware mask.
        "radar_dropout_p": 0.0,
        "aux_dim": aux_dim,
        "structured_auxiliary": structured_auxiliary,
        "exact_auxiliary_alignment": bool(
            args.exact_aux_alignment and structured_auxiliary
        ),
        "aux_base_dim": (
            aux_base_dim if structured_auxiliary or harmonic_auxiliary else None
        ),
        "harmonic_auxiliary": harmonic_auxiliary,
        "alias_gated_harmonic": alias_gated_harmonic,
        "input_branches": args.input_branches,
        "input_frequency_min_hz": input_frequency_range_hz[0],
        "input_frequency_max_hz": input_frequency_range_hz[1],
    }
    if harmonic_auxiliary:
        if auxiliary_frequency_range_hz is None:
            raise ValueError(
                "auxiliary frequency range is required by the harmonic head"
            )
        common.update(
            auxiliary_frequency_min_hz=auxiliary_frequency_range_hz[0],
            auxiliary_frequency_max_hz=auxiliary_frequency_range_hz[1],
        )
    if model_type == "teacher":
        if args.preset == "tiny":
            common.update(
                spatial_channels=(8, 12),
                frequency_dilations=(1,),
                dropout=0.05,
            )
        elif args.preset == "compact":
            common.update(
                spatial_channels=(16, 24, 40),
                frequency_dilations=(1, 2, 4),
                dropout=0.10,
            )
        else:
            common.update(
                spatial_channels=(32, 48, 72, 96),
                frequency_dilations=(1, 2, 4, 8),
                dropout=0.10,
            )
    elif model_type == "snn":
        if args.preset == "tiny":
            common.update(
                spatial_channels=(8, 12),
                hidden_channels=16,
                num_spiking_blocks=1,
                simulation_steps=min(args.simulation_steps, 2),
                dropout=0.02,
            )
        elif args.preset == "compact":
            common.update(
                spatial_channels=(16, 24, 40),
                hidden_channels=min(args.hidden_dim, 96),
                num_spiking_blocks=2,
                simulation_steps=args.simulation_steps,
                dropout=0.05,
            )
        else:
            common.update(
                spatial_channels=(24, 40, 64),
                hidden_channels=args.hidden_dim,
                num_spiking_blocks=2,
                simulation_steps=args.simulation_steps,
                dropout=0.05,
            )
    else:
        raise ValueError(f"unknown model type: {model_type}")
    return common


def build_model(model_type: str, kwargs: Mapping[str, Any]) -> nn.Module:
    if model_type == "teacher":
        return SharedRadarCNNTeacher(**dict(kwargs))
    if model_type == "snn":
        return TriRadarRRSNN(**dict(kwargs))
    raise ValueError(f"unknown model type: {model_type}")


def _move_batch(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor) and key != "index":
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    # Convolutions require float32 on CPU.  On CUDA autocast handles the
    # subsequent operators after this one inexpensive conversion.
    moved["map"] = moved["map"].float()
    return moved


def make_alias_gate_targets(
    target_rr: Tensor,
    classical_rr: Tensor,
    valid: Tensor,
    *,
    rr_min: float,
    rr_max: float,
    tolerance_bpm: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """Create confident direct-vs-alias labels from radar-only candidates.

    The best candidate among ``classical_rr * {1,2,3,4}`` supplies the label.
    Rows where no candidate is within ``tolerance_bpm`` of the reference are
    ignored rather than receiving a speculative divisor target.
    """

    if target_rr.shape != classical_rr.shape or valid.shape != target_rr.shape:
        raise ValueError("alias target inputs must have matching shapes")
    if rr_max <= rr_min or tolerance_bpm <= 0:
        raise ValueError("alias target range and tolerance must be positive")
    divisors = target_rr.new_tensor([1.0, 2.0, 3.0, 4.0])
    candidate = classical_rr.float().unsqueeze(1) * divisors.unsqueeze(0)
    in_range = (candidate >= float(rr_min)) & (candidate <= float(rr_max))
    error = (candidate - target_rr.float().unsqueeze(1)).abs().masked_fill(
        ~in_range, float("inf")
    )
    best_error, best_index = error.min(dim=1)
    confident = (
        valid.bool()
        & torch.isfinite(target_rr)
        & torch.isfinite(classical_rr)
        & (best_error <= float(tolerance_bpm))
    )
    alias_target = best_index > 0
    return alias_target, confident


def compute_multitask_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    model: nn.Module,
    *,
    positive_quality_weight: float = 1.0,
    quality_loss_weight: float = 0.15,
    spike_rate_weight: float = 5e-4,
    teacher_logits: Tensor | None = None,
    distill_weight: float = 0.35,
    distill_temperature: float = 2.0,
    distill_error_gate_bpm: float = 0.0,
    tail_loss_weight: float = 0.0,
    tail_min_bpm: float = 22.0,
    tail_max_bpm: float = 35.0,
    tail_underprediction_ratio: float = 1.0,
    alias_loss_weight: float = 0.0,
    alias_positive_weight: float = 3.0,
    alias_target_tolerance_bpm: float = 2.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Quality-aware distribution/regression loss with label masking."""

    logits = output["logits"].float()
    expected = output["expected_rr"].float()
    log_variance = output["log_variance"].float()
    rr = batch["rr"].float()
    valid = batch["reference_valid"].bool() & torch.isfinite(rr)
    zero = logits.sum() * 0.0

    if valid.any():
        sigma = batch["reference_sigma"].float()[valid].clamp(0.30, 2.5)
        soft_target = gaussian_soft_targets(
            rr[valid], model.rr_bins.float(), sigma=sigma
        )
        reference_weight = batch["reference_quality"].float()[valid].clamp(0.25, 1.0)
        distribution_per_sample = -(
            soft_target * logits[valid].log_softmax(dim=-1)
        ).sum(dim=-1)
        distribution = (
            distribution_per_sample * reference_weight
        ).sum() / reference_weight.sum().clamp_min(1e-6)

        error = expected[valid] - rr[valid]
        huber_per_sample = F.smooth_l1_loss(
            expected[valid], rr[valid], beta=1.0, reduction="none"
        )
        huber = (huber_per_sample * reference_weight).sum() / reference_weight.sum()
        nll_per_sample = 0.5 * (
            error.square() * torch.exp(-log_variance[valid])
            + log_variance[valid]
        )
        uncertainty_nll = (
            nll_per_sample * reference_weight
        ).sum() / reference_weight.sum()

        tail_selected = (
            (rr[valid] >= float(tail_min_bpm))
            & (rr[valid] < float(tail_max_bpm))
        )
        if tail_selected.any() and tail_loss_weight > 0:
            tail_error = error[tail_selected]
            # A smooth capped-quadratic retains a distance-sensitive gradient
            # for large alias errors without allowing a handful of tail rows
            # to dominate the full-distribution objective.
            scale = tail_error.new_tensor(5.0)
            tail_distance_per_sample = scale.square() * torch.log1p(
                tail_error.square() / scale.square()
            )
            asymmetry = torch.where(
                tail_error < 0,
                tail_error.new_tensor(float(tail_underprediction_ratio)),
                tail_error.new_tensor(1.0),
            )
            tail_reference_weight = reference_weight[tail_selected]
            tail_distance = (
                tail_distance_per_sample * asymmetry * tail_reference_weight
            ).sum() / tail_reference_weight.sum().clamp_min(1e-6)
        else:
            tail_distance = zero
    else:
        distribution = huber = uncertainty_nll = tail_distance = zero

    quality_target = batch["observable"].float()
    positive_weight = logits.new_tensor(float(positive_quality_weight))
    quality = F.binary_cross_entropy_with_logits(
        output["quality_logits"].float(),
        quality_target,
        pos_weight=positive_weight,
    )

    distillation = zero
    if teacher_logits is not None and distill_weight > 0 and valid.any():
        temperature = float(distill_temperature)
        teacher_probability = (
            teacher_logits.float()[valid] / temperature
        ).softmax(dim=-1)
        distillation_per_sample = F.kl_div(
            (logits[valid] / temperature).log_softmax(dim=-1),
            teacher_probability,
            reduction="none",
        ).sum(dim=-1) * temperature**2
        if distill_error_gate_bpm > 0:
            teacher_expected = (
                teacher_logits.float()[valid].softmax(dim=-1)
                * model.rr_bins.float().unsqueeze(0)
            ).sum(dim=-1)
            teacher_error = (teacher_expected - rr[valid]).abs()
            gate_scale = teacher_error.new_tensor(float(distill_error_gate_bpm))
            distill_gate = 1.0 / (1.0 + (teacher_error / gate_scale).pow(4))
            distillation = (distillation_per_sample * distill_gate).mean()
        else:
            distillation = distillation_per_sample.mean()

    alias_bce = alias_positive_fraction = alias_accuracy = zero
    if alias_loss_weight > 0:
        if "alias_logits" not in output or "classical_rr" not in batch:
            raise ValueError(
                "alias loss requires harmonic alias logits and classical RR"
            )
        alias_target, alias_confident = make_alias_gate_targets(
            rr,
            batch["classical_rr"].float(),
            valid,
            rr_min=float(model.rr_bins[0]),
            rr_max=float(model.rr_bins[-1]),
            tolerance_bpm=alias_target_tolerance_bpm,
        )
        if "radar_mask" in output:
            alias_confident = alias_confident & output["radar_mask"].bool().all(dim=1)
        if alias_confident.any():
            selected_alias_logits = output["alias_logits"].float()[alias_confident]
            selected_alias_target = alias_target[alias_confident].float()
            alias_bce = F.binary_cross_entropy_with_logits(
                selected_alias_logits,
                selected_alias_target,
                pos_weight=selected_alias_logits.new_tensor(
                    float(alias_positive_weight)
                ),
            )
            alias_positive_fraction = selected_alias_target.mean()
            alias_accuracy = (
                (selected_alias_logits >= 0) == selected_alias_target.bool()
            ).float().mean()

    spike_rate = output.get("spike_rate", zero).float()
    total = (
        distribution
        + 0.50 * huber
        + 0.10 * uncertainty_nll
        + tail_loss_weight * tail_distance
        + quality_loss_weight * quality
        + distill_weight * distillation
        + alias_loss_weight * alias_bce
        + spike_rate_weight * spike_rate
    )
    if "harmonic_logits" in output:
        harmonic_logits_for_log = output["harmonic_logits"].float()
        harmonic_rms = harmonic_logits_for_log.square().mean().sqrt()
        total_logit_rms = logits.square().mean().sqrt()
        harmonic_to_total_rms = harmonic_rms / total_logit_rms.clamp_min(1e-8)
        high_rr = valid & (rr >= 20.0) & (rr < 35.0)
        high_rr_harmonic_rms = (
            harmonic_logits_for_log[high_rr].square().mean().sqrt()
            if high_rr.any()
            else zero
        )
    else:
        harmonic_rms = harmonic_to_total_rms = high_rr_harmonic_rms = zero
    components = {
        "loss": total.detach(),
        "distribution": distribution.detach(),
        "huber": huber.detach(),
        "uncertainty_nll": uncertainty_nll.detach(),
        "tail_distance": tail_distance.detach(),
        "quality_bce": quality.detach(),
        "distillation": distillation.detach(),
        "alias_bce": alias_bce.detach(),
        "alias_positive_fraction": alias_positive_fraction.detach(),
        "alias_accuracy": alias_accuracy.detach(),
        "spike_rate": spike_rate.detach(),
        "harmonic_gain": output.get("harmonic_gain", zero).float().detach(),
        "harmonic_logit_std": (
            output["harmonic_logits"].float().std().detach()
            if "harmonic_logits" in output
            else zero.detach()
        ),
        "harmonic_logit_rms": harmonic_rms.detach(),
        "harmonic_to_total_logit_rms": harmonic_to_total_rms.detach(),
        "high_rr_harmonic_logit_rms": high_rr_harmonic_rms.detach(),
        "valid_fraction": valid.float().mean().detach(),
    }
    return total, components


def _autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    amp_scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    amp: bool,
    positive_quality_weight: float,
    quality_loss_weight: float,
    spike_rate_weight: float,
    distill_bank: Tensor | None,
    distill_weight: float,
    distill_temperature: float,
    distill_error_gate_bpm: float,
    tail_loss_weight: float,
    tail_min_bpm: float,
    tail_max_bpm: float,
    tail_underprediction_ratio: float,
    alias_loss_weight: float,
    alias_positive_weight: float,
    alias_target_tolerance_bpm: float,
    gradient_clip: float,
    radar_dropout_p: float,
    auxiliary_layout: AuxiliaryLayout | None,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    examples = 0
    for batch_number, batch_cpu in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break
        global_index = batch_cpu["index"]
        batch = _move_batch(batch_cpu, device)
        batch["map"], batch["radar_mask"], batch["aux"] = (
            apply_coupled_radar_dropout(
                batch["map"],
                batch["radar_mask"],
                batch["aux"],
                p=radar_dropout_p,
                layout=auxiliary_layout,
            )
        )
        teacher_logits = None
        if distill_bank is not None:
            teacher_logits = distill_bank[global_index].to(
                device=device, dtype=torch.float32, non_blocking=True
            )

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            output = model(batch["map"], batch["radar_mask"], batch["aux"])
            loss, components = compute_multitask_loss(
                output,
                batch,
                model,
                positive_quality_weight=positive_quality_weight,
                quality_loss_weight=quality_loss_weight,
                spike_rate_weight=spike_rate_weight,
                teacher_logits=teacher_logits,
                distill_weight=distill_weight,
                distill_temperature=distill_temperature,
                distill_error_gate_bpm=distill_error_gate_bpm,
                tail_loss_weight=tail_loss_weight,
                tail_min_bpm=tail_min_bpm,
                tail_max_bpm=tail_max_bpm,
                tail_underprediction_ratio=tail_underprediction_ratio,
                alias_loss_weight=alias_loss_weight,
                alias_positive_weight=alias_positive_weight,
                alias_target_tolerance_bpm=alias_target_tolerance_bpm,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.item()}")
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        amp_scaler.step(optimizer)
        amp_scaler.update()

        batch_size = len(batch["rr"])
        examples += batch_size
        for key, value in components.items():
            totals[key] += float(value.item()) * batch_size
    if examples == 0:
        raise RuntimeError("training loader produced no batches")
    return {key: value / examples for key, value in totals.items()}


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None = None,
) -> PredictionBundle:
    model.eval()
    collected: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for batch_number, batch_cpu in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break
        batch = _move_batch(batch_cpu, device)
        with _autocast_context(device, amp):
            output = model(batch["map"], batch["radar_mask"], batch["aux"])
        rr_std = output["rr_std"].float()
        quality = output["quality"].float()
        # This is a ranking score, not a calibrated physical unit.  It makes
        # uncertainty and the independently trained observability head both
        # accountable in selective-risk evaluation.
        uncertainty = rr_std / quality.clamp_min(0.05)
        spike = output.get(
            "spike_rate_per_sample", torch.zeros_like(output["expected_rr"])
        )
        values: dict[str, Tensor] = {
            "index": batch_cpu["index"],
            "target": batch["rr"],
            "prediction": output["expected_rr"],
            "rr_std": rr_std,
            "uncertainty": uncertainty,
            "quality": quality,
            "observable": batch["observable"],
            "reference_valid": batch["reference_valid"],
            "spike_rate": spike,
            "radar_weights": output["radar_weights"],
            "map_prediction": output["map_rr"],
            "posterior_entropy": output["posterior_entropy"],
            "topk_rr": output["topk_rr"],
            "topk_probability": output["topk_probability"],
            "posterior_probability": output["probabilities"].to(torch.float16),
            "alias_probability": output.get(
                "alias_probability",
                torch.full_like(output["expected_rr"], float("nan")),
            ),
        }
        for key, value in values.items():
            collected[key].append(value.detach().cpu().numpy())
    if not collected:
        raise RuntimeError("evaluation loader produced no batches")
    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    return PredictionBundle(**arrays)


def quick_regression_summary(
    bundle: PredictionBundle, metadata: pd.DataFrame
) -> dict[str, float]:
    valid = bundle.reference_valid.astype(bool)
    y_true = bundle.target[valid]
    y_pred = bundle.prediction[valid]
    identity = metadata.iloc[bundle.index[valid]]["identity"].astype(str).to_numpy()
    if len(y_true) == 0:
        raise RuntimeError("prediction bundle contains no valid reference labels")
    return {
        **regression_metrics(y_true, y_pred),
        **identity_macro_metrics(y_true, y_pred, identity),
    }


def detailed_prediction_summary(
    bundle: PredictionBundle,
    metadata: pd.DataFrame,
    *,
    bootstrap_samples: int,
    coverages: Iterable[float],
    rr_range: Sequence[float],
    alias_target_tolerance_bpm: float,
) -> dict[str, Any]:
    if len(rr_range) != 2 or float(rr_range[1]) <= float(rr_range[0]):
        raise ValueError("rr_range must contain an increasing [minimum, maximum]")
    if alias_target_tolerance_bpm <= 0:
        raise ValueError("alias_target_tolerance_bpm must be positive")
    valid = bundle.reference_valid.astype(bool)
    y_true = bundle.target[valid].astype(float)
    y_pred = bundle.prediction[valid].astype(float)
    uncertainty = bundle.uncertainty[valid].astype(float)
    rr_std = bundle.rr_std[valid].astype(float)
    quality = bundle.quality[valid].astype(float)
    observable = bundle.observable[valid].astype(bool)
    identity = metadata.iloc[bundle.index[valid]]["identity"].astype(str).to_numpy()
    if len(y_true) == 0:
        raise RuntimeError("cannot report metrics without valid references")

    per_identity = {
        name: regression_metrics(y_true[identity == name], y_pred[identity == name])
        for name in np.unique(identity)
    }
    bootstrap = clustered_bootstrap_mae(
        y_true,
        y_pred,
        identity,
        samples=bootstrap_samples,
    )
    quality_metrics: dict[str, float | None] = {
        "brier": float(np.mean((quality - observable.astype(float)) ** 2)),
        "mean_quality": float(np.mean(quality)),
        "observable_fraction": float(np.mean(observable)),
    }
    if np.unique(observable).size == 2:
        quality_metrics["roc_auc"] = float(roc_auc_score(observable, quality))
        quality_metrics["average_precision"] = float(
            average_precision_score(observable, quality)
        )
    else:
        quality_metrics["roc_auc"] = None
        quality_metrics["average_precision"] = None

    summary: dict[str, Any] = {
        "overall": regression_metrics(y_true, y_pred),
        "identity_macro": identity_macro_metrics(y_true, y_pred, identity),
        "map_decoder": {
            "overall": regression_metrics(
                y_true, np.asarray(bundle.map_prediction)[valid].astype(float)
            ),
            "identity_macro": identity_macro_metrics(
                y_true,
                np.asarray(bundle.map_prediction)[valid].astype(float),
                identity,
            ),
        },
        "identity_cluster_bootstrap_mae": {
            "estimate": bootstrap[0],
            "ci95_low": bootstrap[1],
            "ci95_high": bootstrap[2],
            "samples": bootstrap_samples,
        },
        "per_identity": per_identity,
        "risk_coverage": risk_coverage_curve(
            y_true,
            y_pred,
            uncertainty,
            coverages=coverages,
            identities=identity,
        ),
        "risk_coverage_rr_std_only": risk_coverage_curve(
            y_true,
            y_pred,
            rr_std,
            coverages=coverages,
            identities=identity,
        ),
        "risk_coverage_quality_only": risk_coverage_curve(
            y_true,
            y_pred,
            -quality,
            coverages=coverages,
            identities=identity,
        ),
        "quality_classifier": quality_metrics,
        "prediction_uncertainty": {
            "mean_rr_std_bpm": float(np.mean(rr_std)),
            "median_rr_std_bpm": float(np.median(rr_std)),
            "mean_combined_score": float(np.mean(uncertainty)),
        },
    }
    finite_spike = bundle.spike_rate[valid][np.isfinite(bundle.spike_rate[valid])]
    if finite_spike.size:
        summary["spike_activity"] = {
            "mean_rate": float(np.mean(finite_spike)),
            "median_rate": float(np.median(finite_spike)),
            "p95_rate": float(np.quantile(finite_spike, 0.95)),
        }
    alias_probability = np.asarray(bundle.alias_probability)[valid].astype(float)
    classical_rr = pd.to_numeric(
        metadata.iloc[bundle.index[valid]]["classical_rr_bpm"], errors="coerce"
    ).to_numpy(dtype=float)
    candidates = classical_rr[:, None] * np.asarray([1.0, 2.0, 3.0, 4.0])[None, :]
    candidate_in_range = (candidates >= float(rr_range[0])) & (
        candidates <= float(rr_range[1])
    )
    candidate_error = np.abs(candidates - y_true[:, None])
    candidate_error[~candidate_in_range] = np.inf
    best_alias_error = np.min(candidate_error, axis=1)
    alias_target = np.argmin(candidate_error, axis=1) > 0
    alias_selected = (
        np.isfinite(alias_probability)
        & np.isfinite(classical_rr)
        & (best_alias_error <= float(alias_target_tolerance_bpm))
    )
    if alias_selected.any():
        alias_truth = alias_target[alias_selected]
        alias_score = alias_probability[alias_selected]
        alias_metrics: dict[str, Any] = {
            "n": int(alias_selected.sum()),
            "positive_fraction": float(np.mean(alias_truth)),
            "accuracy_at_0_5": float(
                np.mean((alias_score >= 0.5) == alias_truth)
            ),
            "mean_probability": float(np.mean(alias_score)),
            "rr_range_bpm": [float(rr_range[0]), float(rr_range[1])],
            "target_tolerance_bpm": float(alias_target_tolerance_bpm),
        }
        if np.unique(alias_truth).size == 2:
            alias_metrics["roc_auc"] = float(
                roc_auc_score(alias_truth, alias_score)
            )
            alias_metrics["average_precision"] = float(
                average_precision_score(alias_truth, alias_score)
            )
        summary["alias_gate"] = alias_metrics
    return summary


def _quality_positive_weight(metadata: pd.DataFrame, indices: np.ndarray) -> float:
    target = metadata.iloc[indices]["radar_observable"].to_numpy(dtype=bool)
    positive = int(target.sum())
    negative = int(len(target) - positive)
    if positive == 0:
        return 1.0
    return float(np.clip(negative / positive, 0.25, 4.0))


def _strict_json_value(value: Any) -> Any:
    """Convert NumPy/torch values and non-finite floats to strict JSON."""

    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, Tensor):
        return _strict_json_value(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _strict_json_value(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _loader_generator(loader: DataLoader[Any]) -> torch.Generator | None:
    generator = getattr(loader, "generator", None)
    return generator if isinstance(generator, torch.Generator) else None


def _sampler_generator(loader: DataLoader[Any]) -> torch.Generator | None:
    generator = getattr(getattr(loader, "sampler", None), "generator", None)
    return generator if isinstance(generator, torch.Generator) else None


def capture_rng_state(train_loader: DataLoader[Any]) -> dict[str, Any]:
    """Capture process and sampling RNG state at an epoch boundary."""

    loader_generator = _loader_generator(train_loader)
    sampler_generator = _sampler_generator(train_loader)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "data_loader_generator": (
            loader_generator.get_state() if loader_generator is not None else None
        ),
        "sampler_generator": (
            sampler_generator.get_state() if sampler_generator is not None else None
        ),
    }


def restore_rng_state(
    state: Mapping[str, Any], train_loader: DataLoader[Any]
) -> None:
    """Restore a state produced by :func:`capture_rng_state`.

    A CUDA device-count change is rejected instead of partially restoring a
    state and silently claiming interrupted-run equivalence.
    """

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"checkpoint RNG state is incomplete: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], device="cpu"))

    cuda_states = list(state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "checkpoint CUDA RNG device count does not match the current runtime"
            )
        torch.cuda.set_rng_state_all(cuda_states)

    loader_generator = _loader_generator(train_loader)
    loader_state = state.get("data_loader_generator")
    if loader_state is not None:
        if loader_generator is None:
            raise RuntimeError(
                "checkpoint contains a DataLoader generator state but the loader has none"
            )
        loader_generator.set_state(torch.as_tensor(loader_state, device="cpu"))

    sampler_generator = _sampler_generator(train_loader)
    sampler_state = state.get("sampler_generator")
    if sampler_state is not None:
        if sampler_generator is None:
            raise RuntimeError(
                "checkpoint contains a sampler generator state but the sampler has none"
            )
        sampler_generator.set_state(torch.as_tensor(sampler_state, device="cpu"))


def _base_checkpoint(
    *,
    model: nn.Module,
    model_type: str,
    model_kwargs: Mapping[str, Any],
    epoch: int,
    best_epoch: int,
    best_score: float,
    fold: int,
    split: Mapping[str, Sequence[str]],
    aux_center: np.ndarray,
    aux_scale: np.ndarray,
    run_signature: str,
    rng_state: Mapping[str, Any],
    cache_provenance: Mapping[str, Any],
    distillation_teacher_provenance: Mapping[str, Any] | None,
    split_authority_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = {
        "format_version": 2,
        "model_type": model_type,
        "model_kwargs": dict(model_kwargs),
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "fold": fold,
        "split": {key: list(value) for key, value in split.items()},
        "aux_center": torch.from_numpy(aux_center),
        "aux_scale": torch.from_numpy(aux_scale),
        "run_signature": run_signature,
        "rng_state": dict(rng_state),
        "cache_provenance": dict(cache_provenance),
        "distillation_teacher_provenance": (
            dict(distillation_teacher_provenance)
            if distillation_teacher_provenance is not None
            else None
        ),
    }
    if split_authority_provenance is not None:
        checkpoint["split_authority_provenance"] = dict(
            split_authority_provenance
        )
    return checkpoint


def load_checkpoint_model(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_type"], checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device), checkpoint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_checkpoint_provenance(
    path: Path, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    """Return stable provenance used to bind an SNN resume to its teacher."""

    resolved = path.expanduser().resolve()
    split_provenance = checkpoint.get("split_authority_provenance")
    if not isinstance(split_provenance, Mapping):
        split_provenance = None
    provenance = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "checkpoint_run_signature": str(checkpoint.get("run_signature", "")),
        "checkpoint_format_version": int(checkpoint.get("format_version", -1)),
        "fold": int(checkpoint.get("fold", -1)),
        "model_type": str(checkpoint.get("model_type", "")),
    }
    if split_provenance is not None:
        provenance.update(
            split_manifest_content_sha256=str(
                split_provenance.get("split_manifest_content_sha256", "")
            ),
            excluded_identities=list(
                split_provenance.get("excluded_identities", ())
            ),
            scaler_identities=list(split_provenance.get("scaler_identities", ())),
        )
    cache_provenance = checkpoint.get("cache_provenance")
    if isinstance(cache_provenance, Mapping):
        provenance["cache_provenance_sha256"] = str(
            cache_provenance.get("content_sha256", "")
        )
    return provenance


def _resolve_context_path(value: Any) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_external_teacher_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    path: Path,
    fold: int,
    split: Mapping[str, Sequence[str]],
    expected_model_context: Mapping[str, Any],
    aux_center: np.ndarray,
    aux_scale: np.ndarray,
    current_run_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject an external teacher that is not the same fold/data/RR context.

    Teacher capacity and teacher-only feature-head choices may differ from the
    student.  The identity split, tensor interface, RR grid, scaler, cache
    context, and checkpoint/run provenance may not.
    """

    resolved = path.expanduser().resolve()
    errors: list[str] = []
    if checkpoint.get("model_type") != "teacher":
        errors.append("model_type must be 'teacher'")
    if checkpoint.get("format_version") not in {1, 2}:
        errors.append("unsupported checkpoint format_version")
    if int(checkpoint.get("fold", -1)) != int(fold):
        errors.append(
            f"fold mismatch (checkpoint={checkpoint.get('fold')!r}, student={fold})"
        )

    checkpoint_split = checkpoint.get("split")
    if not isinstance(checkpoint_split, Mapping):
        errors.append("checkpoint split is missing or invalid")
    else:
        split_keys = (
            (
                "train_identities",
                "validation_identities",
                "prediction_identities",
                "excluded_identities",
                "scaler_identities",
            )
            if "prediction_identities" in split
            else (
                "train_identities",
                "validation_identities",
                "test_identities",
            )
        )
        for key in split_keys:
            expected = sorted(str(value) for value in split.get(key, ()))
            actual = sorted(str(value) for value in checkpoint_split.get(key, ()))
            if actual != expected:
                errors.append(f"{key} mismatch")

    expected_split_provenance = current_run_context.get(
        "split_authority_provenance"
    )
    checkpoint_split_provenance = checkpoint.get("split_authority_provenance")
    if checkpoint_split_provenance != expected_split_provenance:
        errors.append("split authority provenance mismatch")
    expected_cache_provenance = current_run_context.get("cache_provenance")
    if (
        expected_cache_provenance is not None
        and checkpoint.get("cache_provenance") != expected_cache_provenance
    ):
        errors.append("cache provenance mismatch")

    model_kwargs = checkpoint.get("model_kwargs")
    if not isinstance(model_kwargs, Mapping):
        errors.append("checkpoint model_kwargs are missing or invalid")
        model_kwargs = {}
    float_context_keys = {
        "rr_min",
        "rr_max",
        "input_frequency_min_hz",
        "input_frequency_max_hz",
    }
    for key, expected in expected_model_context.items():
        if key not in model_kwargs:
            errors.append(f"model context key {key!r} is missing")
            continue
        actual = model_kwargs[key]
        if key in float_context_keys:
            if not math.isclose(
                float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-8
            ):
                errors.append(
                    f"model context {key} mismatch (checkpoint={actual}, student={expected})"
                )
        elif actual != expected:
            errors.append(
                f"model context {key} mismatch (checkpoint={actual}, student={expected})"
            )

    try:
        checkpoint_center = np.asarray(
            checkpoint["aux_center"].detach().cpu(), dtype=np.float32
        )
        checkpoint_scale = np.asarray(
            checkpoint["aux_scale"].detach().cpu(), dtype=np.float32
        )
    except (AttributeError, KeyError, TypeError) as exc:
        errors.append(f"auxiliary scaler is missing or invalid ({exc})")
    else:
        if not (
            checkpoint_center.shape == aux_center.shape
            and checkpoint_scale.shape == aux_scale.shape
            and np.allclose(checkpoint_center, aux_center)
            and np.allclose(checkpoint_scale, aux_scale)
        ):
            errors.append("auxiliary scaler does not match the student fold")

    expected_rr_grid = np.linspace(
        float(expected_model_context["rr_min"]),
        float(expected_model_context["rr_max"]),
        int(expected_model_context["num_rr_bins"]),
        dtype=np.float32,
    )
    try:
        checkpoint_rr_grid = np.asarray(
            checkpoint["model_state"]["rr_bins"].detach().cpu(), dtype=np.float32
        )
    except (AttributeError, KeyError, TypeError) as exc:
        errors.append(f"checkpoint RR grid is missing or invalid ({exc})")
    else:
        if checkpoint_rr_grid.shape != expected_rr_grid.shape or not np.allclose(
            checkpoint_rr_grid, expected_rr_grid, rtol=1e-6, atol=1e-6
        ):
            errors.append("checkpoint RR grid does not match the student RR grid")

    run_config_path = resolved.parent.parent / "run_config.json"
    if not run_config_path.is_file():
        errors.append(f"teacher run_config.json is missing: {run_config_path}")
    else:
        try:
            teacher_run = json.loads(run_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"teacher run_config.json is invalid ({exc})")
            teacher_run = {}
        checkpoint_signature = checkpoint.get("run_signature")
        if not isinstance(checkpoint_signature, str) or not checkpoint_signature:
            errors.append("checkpoint run_signature is missing")
        elif teacher_run.get("run_signature") != checkpoint_signature:
            errors.append("checkpoint and teacher run_config signatures differ")

        teacher_arguments = teacher_run.get("arguments", {})
        if not isinstance(teacher_arguments, Mapping):
            errors.append("teacher run arguments are missing or invalid")
            teacher_arguments = {}
        for key in (
            "folds",
            "rr_range",
            "rr_bin_width",
            "map_branch",
            "input_branches",
            "use_aux",
            "causal_history",
        ):
            expected = current_run_context[key]
            actual = teacher_arguments.get(key)
            if key == "rr_range":
                matches = np.allclose(
                    np.asarray(actual, dtype=float),
                    np.asarray(expected, dtype=float),
                    rtol=0.0,
                    atol=1e-8,
                ) if actual is not None else False
            elif key == "rr_bin_width":
                matches = actual is not None and math.isclose(
                    float(actual), float(expected), rel_tol=0.0, abs_tol=1e-8
                )
            else:
                matches = actual == expected
            if not matches:
                errors.append(f"teacher run argument {key!r} is incompatible")

        teacher_cache_dir = teacher_arguments.get("cache_dir")
        if teacher_cache_dir is None or _resolve_context_path(
            teacher_cache_dir
        ) != _resolve_context_path(current_run_context["cache_dir"]):
            errors.append("teacher cache_dir is incompatible with the student run")
        teacher_cache_shape = teacher_run.get("cache_shape")
        if teacher_cache_shape != current_run_context["cache_shape"]:
            errors.append("teacher cache shape is incompatible with the student run")
        if teacher_run.get("split_authority") != expected_split_provenance:
            errors.append("teacher run split authority provenance mismatch")
        if (
            expected_cache_provenance is not None
            and teacher_run.get("cache_provenance") != expected_cache_provenance
        ):
            errors.append("teacher run cache provenance mismatch")

    if errors:
        raise RuntimeError(
            f"incompatible external teacher checkpoint {resolved}: "
            + "; ".join(errors)
        )
    return teacher_checkpoint_provenance(resolved, checkpoint)


def train_stage(
    *,
    model: nn.Module,
    model_type: str,
    model_kwargs: Mapping[str, Any],
    train_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    metadata: pd.DataFrame,
    device: torch.device,
    fold_dir: Path,
    fold: int,
    split: Mapping[str, Sequence[str]],
    aux_center: np.ndarray,
    aux_scale: np.ndarray,
    run_signature: str,
    args: argparse.Namespace,
    quality_positive_weight: float,
    auxiliary_layout: AuxiliaryLayout | None,
    cache_provenance: Mapping[str, Any],
    distill_bank: Tensor | None = None,
    distillation_teacher_provenance: Mapping[str, Any] | None = None,
    split_authority_provenance: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, args.patience // 3),
        min_lr=1e-6,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_path = fold_dir / f"{model_type}_best.pt"
    last_path = fold_dir / f"{model_type}_last.pt"
    log_path = fold_dir / f"{model_type}_history.jsonl"
    start_epoch = 0
    best_score = float("inf")
    best_epoch = -1
    bad_epochs = 0

    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("run_signature") != run_signature:
            raise RuntimeError(f"resume signature mismatch: {last_path}")
        if checkpoint.get("cache_provenance") != dict(cache_provenance):
            raise RuntimeError(f"resume cache provenance mismatch: {last_path}")
        if checkpoint.get("distillation_teacher_provenance") != (
            dict(distillation_teacher_provenance)
            if distillation_teacher_provenance is not None
            else None
        ):
            raise RuntimeError(
                f"resume teacher provenance mismatch: {last_path}"
            )
        if checkpoint.get("split_authority_provenance") != (
            dict(split_authority_provenance)
            if split_authority_provenance is not None
            else None
        ):
            raise RuntimeError(f"resume split authority mismatch: {last_path}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("amp_scaler_state"):
            amp_scaler.load_state_dict(checkpoint["amp_scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
        rng_state = checkpoint.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise RuntimeError(f"resume checkpoint has no complete RNG state: {last_path}")
        restore_rng_state(rng_state, train_loader)

    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            amp_scaler,
            device,
            amp=amp_enabled,
            positive_quality_weight=quality_positive_weight,
            quality_loss_weight=args.quality_loss_weight,
            spike_rate_weight=(
                args.spike_rate_weight if model_type == "snn" else 0.0
            ),
            distill_bank=distill_bank if model_type == "snn" else None,
            distill_weight=args.distill_weight if model_type == "snn" else 0.0,
            distill_temperature=args.distill_temperature,
            distill_error_gate_bpm=(
                args.distill_error_gate_bpm if model_type == "snn" else 0.0
            ),
            tail_loss_weight=args.tail_loss_weight,
            tail_min_bpm=args.tail_min_bpm,
            tail_max_bpm=args.tail_max_bpm,
            tail_underprediction_ratio=args.tail_underprediction_ratio,
            alias_loss_weight=(
                args.alias_loss_weight if model_type == "snn" else 0.0
            ),
            alias_positive_weight=args.alias_positive_weight,
            alias_target_tolerance_bpm=args.alias_target_tolerance_bpm,
            gradient_clip=args.gradient_clip,
            radar_dropout_p=args.radar_dropout,
            auxiliary_layout=auxiliary_layout,
            max_batches=args.max_train_batches,
        )
        validation = predict(
            model,
            validation_loader,
            device,
            amp=amp_enabled,
            max_batches=args.max_eval_batches,
        )
        validation_metrics = quick_regression_summary(validation, metadata)
        score = float(validation_metrics["macro_mae"])
        scheduler.step(score)
        improved = score < best_score - args.min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
            "best_macro_mae": best_score,
            "best_epoch": best_epoch,
            "improved": improved,
        }
        history.append(record)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_strict_json_value(record), allow_nan=False) + "\n")

        rng_state = capture_rng_state(train_loader)
        base = _base_checkpoint(
            model=model,
            model_type=model_type,
            model_kwargs=model_kwargs,
            epoch=epoch,
            best_epoch=best_epoch,
            best_score=best_score,
            fold=fold,
            split=split,
            aux_center=aux_center,
            aux_scale=aux_scale,
            run_signature=run_signature,
            rng_state=rng_state,
            cache_provenance=cache_provenance,
            distillation_teacher_provenance=distillation_teacher_provenance,
            split_authority_provenance=split_authority_provenance,
        )
        last = {
            **base,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "amp_scaler_state": amp_scaler.state_dict(),
            "bad_epochs": bad_epochs,
        }
        atomic_torch_save(last, last_path)
        if improved:
            atomic_torch_save(base, best_path)

        print(
            f"fold={fold} model={model_type} epoch={epoch + 1}/{args.epochs} "
            f"loss={train_metrics['loss']:.4f} "
            f"val_macro_mae={score:.4f} best={best_score:.4f}",
            flush=True,
        )
        if bad_epochs >= args.patience:
            break

    if not best_path.is_file():
        raise RuntimeError(f"best checkpoint was not produced: {best_path}")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    best_validation = predict(
        model,
        validation_loader,
        device,
        amp=amp_enabled,
        max_batches=args.max_eval_batches,
    )
    validation_summary: dict[str, Any] = quick_regression_summary(
        best_validation, metadata
    )
    validation_valid = best_validation.reference_valid.astype(bool)
    validation_target = best_validation.target[validation_valid]
    validation_prediction = best_validation.prediction[validation_valid]
    validation_map_prediction = np.asarray(best_validation.map_prediction)[
        validation_valid
    ]
    validation_identity = metadata.iloc[
        best_validation.index[validation_valid]
    ]["identity"].astype(str).to_numpy()
    validation_summary["map_decoder"] = {
        "overall": regression_metrics(
            validation_target, validation_map_prediction
        ),
        "identity_macro": identity_macro_metrics(
            validation_target,
            validation_map_prediction,
            validation_identity,
        ),
    }
    tail = (
        (validation_target >= float(args.tail_min_bpm))
        & (validation_target < float(args.tail_max_bpm))
    )
    validation_summary["tail_band_bpm"] = [
        float(args.tail_min_bpm),
        float(args.tail_max_bpm),
    ]
    validation_summary["tail"] = (
        regression_metrics(validation_target[tail], validation_prediction[tail])
        if tail.any()
        else None
    )
    validation_summary["map_tail"] = (
        regression_metrics(validation_target[tail], validation_map_prediction[tail])
        if tail.any()
        else None
    )
    save_prediction_bundle(
        fold_dir / f"{model_type}_validation_predictions.npz",
        best_validation,
        fold=fold,
        run_signature=run_signature,
    )
    stage = {
        "best_epoch": int(best_checkpoint["best_epoch"]),
        "best_validation_macro_mae": float(best_checkpoint["best_score"]),
        "epochs_executed": len(history),
        "parameters": count_trainable_parameters(model),
        "best_validation": validation_summary,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "cache_provenance": dict(cache_provenance),
        "distillation_teacher_provenance": (
            dict(distillation_teacher_provenance)
            if distillation_teacher_provenance is not None
            else None
        ),
    }
    if split_authority_provenance is not None:
        stage["split_authority_provenance"] = dict(
            split_authority_provenance
        )
    return model, stage


@torch.inference_mode()
def precompute_teacher_bank(
    teacher: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    total_rows: int,
    num_bins: int,
    amp: bool,
) -> Tensor:
    """Cache full-radar teacher logits once instead of forwarding every SNN epoch."""

    teacher.eval()
    bank = torch.full((total_rows, num_bins), float("nan"), dtype=torch.float16)
    for batch_cpu in loader:
        batch = _move_batch(batch_cpu, device)
        with _autocast_context(device, amp):
            output = teacher(batch["map"], batch["radar_mask"], batch["aux"])
        bank[batch_cpu["index"]] = output["logits"].detach().cpu().to(torch.float16)
    return bank


def save_prediction_bundle(
    path: Path,
    bundle: PredictionBundle,
    *,
    fold: int | np.ndarray,
    run_signature: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if np.isscalar(fold):
        fold_number = int(fold)
        fold_dtype = (
            np.int16
            if np.iinfo(np.int16).min <= fold_number <= np.iinfo(np.int16).max
            else np.int64
        )
        fold_values = np.full(len(bundle), fold_number, dtype=fold_dtype)
    else:
        raw_fold = np.asarray(fold)
        if raw_fold.shape != (len(bundle),):
            raise ValueError("fold array must have one value per prediction row")
        if not np.issubdtype(raw_fold.dtype, np.integer):
            raise ValueError("fold array must contain integers")
        use_int16 = bool(
            len(raw_fold) == 0
            or (
                raw_fold.min() >= np.iinfo(np.int16).min
                and raw_fold.max() <= np.iinfo(np.int16).max
            )
        )
        fold_values = raw_fold.astype(np.int16 if use_int16 else np.int64)
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **asdict(bundle),
            fold=fold_values,
            run_signature=np.asarray(run_signature),
        )
    temporary.replace(path)


def load_prediction_bundle(path: Path) -> tuple[PredictionBundle, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as data:
        bundle = PredictionBundle(
            **{
                field: np.asarray(data[field])
                for field in PredictionBundle.__dataclass_fields__
                if field in data.files
            }
        )
        fold = np.asarray(data["fold"])
        signature = str(np.asarray(data["run_signature"]).item())
    return bundle, fold, signature


def concatenate_bundles(bundles: Sequence[PredictionBundle]) -> PredictionBundle:
    if not bundles:
        raise ValueError("no prediction bundles supplied")
    posterior_widths = {
        np.asarray(bundle.posterior_probability).shape[1] for bundle in bundles
    }
    if len(posterior_widths - {0}) > 1:
        raise ValueError("prediction bundles use incompatible posterior grids")
    posterior_width = max(posterior_widths)

    def concatenate_field(field: str) -> np.ndarray:
        values = [np.asarray(getattr(bundle, field)) for bundle in bundles]
        if field == "posterior_probability" and posterior_width:
            values = [
                value
                if value.shape[1] == posterior_width
                else np.full(
                    (value.shape[0], posterior_width), np.nan, dtype=np.float16
                )
                for value in values
            ]
        return np.concatenate(values, axis=0)

    result = PredictionBundle(
        **{
            field: concatenate_field(field)
            for field in PredictionBundle.__dataclass_fields__
        }
    )
    order = np.argsort(result.index, kind="stable")
    return PredictionBundle(
        **{field: getattr(result, field)[order] for field in PredictionBundle.__dataclass_fields__}
    )


def aggregate_oof(
    output_dir: Path,
    model_type: str,
    metadata: pd.DataFrame,
    *,
    run_signature: str,
    expected_folds: int,
    bootstrap_samples: int,
    coverages: Iterable[float],
    rr_range: Sequence[float],
    alias_target_tolerance_bpm: float,
) -> dict[str, Any] | None:
    bundles: list[PredictionBundle] = []
    folds: list[np.ndarray] = []
    for path in sorted(output_dir.glob(f"fold_*/{model_type}_test_predictions.npz")):
        bundle, fold, signature = load_prediction_bundle(path)
        if signature == run_signature:
            bundles.append(bundle)
            folds.append(fold)
    if not bundles:
        return None
    combined = concatenate_bundles(bundles)
    combined_folds = np.concatenate(folds, axis=0)
    combined_order = np.argsort(
        np.concatenate([bundle.index for bundle in bundles]), kind="stable"
    )
    combined_folds = combined_folds[combined_order]
    if len(np.unique(combined.index)) != len(combined.index):
        raise RuntimeError(f"duplicate {model_type} OOF cache indices detected")

    npz_path = output_dir / f"{model_type}_oof.npz"
    save_prediction_bundle(
        npz_path,
        combined,
        fold=combined_folds,
        run_signature=run_signature,
    )
    rows = metadata.iloc[combined.index][
        [
            "session_id",
            "identity",
            "protocol",
            "window_number",
            "window_start_s",
            "window_end_s",
            "rr_bpm",
            "reference_quality",
            "radar_observable",
            "classical_rr_bpm",
        ]
    ].reset_index(drop=True)
    rows.insert(0, "cache_index", combined.index)
    rows.insert(1, "fold", combined_folds)
    rows["prediction_bpm"] = combined.prediction
    rows["map_prediction_bpm"] = np.asarray(combined.map_prediction)
    rows["posterior_entropy"] = np.asarray(combined.posterior_entropy)
    rows["alias_probability"] = np.asarray(combined.alias_probability)
    for rank in range(np.asarray(combined.topk_rr).shape[1]):
        rows[f"posterior_top{rank + 1}_rr_bpm"] = np.asarray(combined.topk_rr)[:, rank]
        rows[f"posterior_top{rank + 1}_probability"] = np.asarray(
            combined.topk_probability
        )[:, rank]
    rows["rr_std_bpm"] = combined.rr_std
    rows["quality"] = combined.quality
    rows["uncertainty_score"] = combined.uncertainty
    rows["spike_rate"] = combined.spike_rate
    rows.to_csv(output_dir / f"{model_type}_oof.csv", index=False)

    summary = detailed_prediction_summary(
        combined,
        metadata,
        bootstrap_samples=bootstrap_samples,
        coverages=coverages,
        rr_range=rr_range,
        alias_target_tolerance_bpm=alias_target_tolerance_bpm,
    )
    summary["folds_present"] = sorted(set(combined_folds.astype(int).tolist()))
    expected_rows = int(metadata["reference_valid"].to_numpy(dtype=bool).sum())
    summary["complete_oof"] = bool(
        len(summary["folds_present"]) == expected_folds
        and len(combined) == expected_rows
    )
    summary["expected_folds"] = expected_folds
    summary["expected_valid_rows"] = expected_rows
    checkpoint_paths = sorted(output_dir.glob(f"fold_*/{model_type}_best.pt"))
    summary["n_parameters"] = None
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run_signature") == run_signature:
            checkpoint_model = build_model(
                checkpoint["model_type"], checkpoint["model_kwargs"]
            )
            summary["n_parameters"] = count_trainable_parameters(checkpoint_model)
            break
    write_json(output_dir / f"{model_type}_metrics.json", summary)
    return summary


def _resolve_teacher_checkpoint(args: argparse.Namespace, fold: int, fold_dir: Path) -> Path:
    if args.teacher_checkpoint:
        return Path(args.teacher_checkpoint.format(fold=fold)).expanduser().resolve()
    return fold_dir / "teacher_best.pt"


def _fold_split(
    metadata: pd.DataFrame,
    assignment: np.ndarray,
    fold: int,
    n_splits: int,
    *,
    include_invalid: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    valid = metadata["reference_valid"].to_numpy(dtype=bool)
    test_fold = fold
    validation_fold = (fold + 1) % n_splits
    test = np.flatnonzero((assignment == test_fold) & valid)
    validation = np.flatnonzero((assignment == validation_fold) & valid)
    train_mask = (assignment >= 0) & (assignment != test_fold) & (
        assignment != validation_fold
    )
    if not include_invalid:
        train_mask &= valid
    train = np.flatnonzero(train_mask)
    identities = metadata["identity"].astype(str).to_numpy()
    split = {
        "train_identities": sorted(set(identities[train].tolist())),
        "validation_identities": sorted(set(identities[validation].tolist())),
        "test_identities": sorted(set(identities[test].tolist())),
    }
    sets = [set(value) for value in split.values()]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("identity leakage detected in fold split")
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError(f"fold {fold} has an empty split")
    return train, validation, test, split


def _run_signature(args: argparse.Namespace) -> str:
    # Epoch count and stage selection are deliberately excluded so a run can
    # be resumed for longer or completed teacher-first/SNN-second.
    keys = (
        "seed",
        "deterministic",
        "amp",
        "cache_dir",
        "cache_trust_mode",
        "cache_provenance_sha256",
        "identity_split_manifest_sha256",
        "folds",
        "rr_range",
        "rr_bin_width",
        "preset",
        "simulation_steps",
        "hidden_dim",
        "radar_dropout",
        "map_branch",
        "input_branches",
        "use_aux",
        "aux_fusion",
        "exact_aux_alignment",
        "harmonic_head",
        "alias_gate",
        "causal_history",
        "keep_aux_in_branch_ablation",
        "include_invalid",
        "valid_boost",
        "rr_balance_power",
        "rr_balance_bin_width",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "samples_per_epoch",
        "gradient_clip",
        "patience",
        "min_delta",
        "workers",
        "num_threads",
        "distill_weight",
        "distill_temperature",
        "distill_error_gate_bpm",
        "teacher_checkpoint",
        "tail_loss_weight",
        "tail_min_bpm",
        "tail_max_bpm",
        "tail_underprediction_ratio",
        "alias_loss_weight",
        "alias_positive_weight",
        "alias_target_tolerance_bpm",
        "quality_loss_weight",
        "spike_rate_weight",
        "max_train_batches",
        "max_eval_batches",
    )
    payload = {key: getattr(args, key, None) for key in keys}
    # Preserve every historical legacy signature byte-for-byte.  The custom
    # authority binding is present only when that mode is explicitly enabled.
    if payload["identity_split_manifest_sha256"] is None:
        payload.pop("identity_split_manifest_sha256")
    if payload["cache_provenance_sha256"] is None:
        payload.pop("cache_provenance_sha256")
    encoded = json.dumps(_strict_json_value(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive")
    seed_everything(args.seed, deterministic=args.deterministic)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    torch.set_float32_matmul_precision("high")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    amp_enabled = bool(args.amp and device.type == "cuda")

    trust_mode = str(args.cache_trust_mode)
    cache = load_feature_cache(
        args.cache_dir,
        require_acquisition_contract=trust_mode != "legacy",
        require_scientific_eligible=trust_mode == "scientific",
    )
    if cache.provenance is None:
        raise RuntimeError("feature cache loader returned no verified provenance")
    cache_classification = cache.provenance.classification
    if trust_mode == "scientific":
        if cache_classification != "acquisition_scientific":
            raise ValueError(
                "scientific cache trust mode requires a verified v2 full-cohort "
                "acquisition cache"
            )
        if cache.radar_timing_valid_mask is None:
            raise RuntimeError(
                "scientific acquisition training requires the bound structural "
                "radar timing mask"
            )
        if not bool(np.asarray(cache.radar_timing_valid_mask, dtype=np.bool_).all()):
            raise RuntimeError(
                "scientific acquisition training requires every radar interval "
                "to be structurally valid"
            )
        claim_classification = "retrospective_scientific_noncommercial"
    elif trust_mode == "acquisition-diagnostic":
        if not cache_classification.startswith("acquisition_"):
            raise ValueError(
                "acquisition-diagnostic trust mode requires an acquisition-aware cache"
            )
        if (
            cache.provenance.acquisition_schema_version
            != ACQUISITION_CACHE_SCHEMA_VERSION_V2
            or cache.radar_timing_valid_mask is None
        ):
            raise ValueError(
                "acquisition-diagnostic training requires a verified v2 cache and "
                "its structural radar timing mask; older acquisition caches are "
                "inspection-only"
            )
        claim_classification = "acquisition_diagnostic_noncommercial"
    elif trust_mode == "legacy":
        if cache_classification != "legacy":
            raise ValueError(
                "legacy trust mode cannot downgrade an acquisition-aware cache; "
                "use scientific or acquisition-diagnostic"
            )
        claim_classification = "retrospective_legacy_noncommercial"
    else:  # argparse guards this; keep programmatic callers fail closed.
        raise ValueError(f"unknown cache trust mode: {trust_mode}")
    cache_provenance = cache.provenance.to_dict()
    cache_provenance["content_sha256"] = cache.provenance.content_sha256
    args.cache_provenance_sha256 = cache.provenance.content_sha256
    args.claim_classification = claim_classification
    args.radar_timing_mask_policy = (
        "all_required_window_intervals"
        if cache.radar_timing_valid_mask is not None
        else "legacy_all_views_assumed"
    )
    stored_range_bins = int(cache.maps.shape[-1])
    if stored_range_bins % 2:
        raise ValueError(
            "raw/phase branch selection requires an even cached range dimension"
        )
    half_range = stored_range_bins // 2
    if args.map_branch == "both":
        selected_maps = cache.maps
        args.input_branches = 2
    elif args.map_branch == "raw":
        selected_maps = cache.maps[..., :half_range]
        args.input_branches = 1
    elif args.map_branch == "phase":
        selected_maps = cache.maps[..., half_range:]
        args.input_branches = 1
    else:  # argparse guards this; keep the programmatic API defensive.
        raise ValueError(f"unknown map branch: {args.map_branch}")
    if (
        args.map_branch != "both"
        and args.use_aux
        and not args.keep_aux_in_branch_ablation
    ):
        # The current auxiliary cache aggregates raw and phase evidence before
        # storage and cannot be separated after the fact.  Disabling it makes
        # this an honest branch ablation instead of leaking the held-out branch.
        args.use_aux = False
    if args.harmonic_head and not args.use_aux:
        raise ValueError(
            "harmonic head requires auxiliary spectra; enable --use-aux and use "
            "the combined map branch or explicitly retain auxiliary evidence"
        )
    cache = FeatureCache(
        maps=selected_maps,
        aux=cache.aux,
        metadata=cache.metadata,
        frequencies_hz=cache.frequencies_hz,
        provenance=cache.provenance,
        radar_timing_valid_mask=cache.radar_timing_valid_mask,
    )
    base_aux_dim = int(cache.aux.shape[1])
    auxiliary_layout = (
        infer_auxiliary_layout(base_aux_dim) if args.use_aux else None
    )
    auxiliary_frequency_range_hz = (
        infer_auxiliary_frequency_range(
            cache.frequencies_hz, auxiliary_layout.frequency_bins
        )
        if auxiliary_layout is not None
        else None
    )
    history_names: list[str] = []
    if args.use_aux and args.causal_history:
        augmented_aux, history_names = append_mask_aware_causal_history_features(cache)
        cache = FeatureCache(
            maps=cache.maps,
            aux=augmented_aux,
            metadata=cache.metadata,
            frequencies_hz=cache.frequencies_hz,
            provenance=cache.provenance,
            radar_timing_valid_mask=cache.radar_timing_valid_mask,
        )
    if cache.maps.shape[-1] % args.input_branches:
        raise ValueError(
            f"cached range dimension {cache.maps.shape[-1]} is not divisible by "
            f"--input-branches={args.input_branches}"
        )
    split_authority: IdentitySplitAuthority | None = None
    assignment: np.ndarray | None = None
    identity_to_fold: Mapping[str, int]
    if args.identity_split_manifest is not None:
        split_authority = load_identity_split_authority(
            args.identity_split_manifest,
            metadata=cache.metadata,
            cache_dir=args.cache_dir,
        )
        args.identity_split_manifest_sha256 = split_authority.content_sha256
        identity_to_fold = split_authority.identity_to_fold
        selected_folds = [split_authority.fold_id]
    else:
        args.identity_split_manifest_sha256 = None
        assignment, generated_identity_to_fold = make_fold_assignments(
            cache.metadata, args.folds, args.seed
        )
        identity_to_fold = generated_identity_to_fold
        selected_folds = parse_fold_selection(args.fold, args.folds)
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path("artifacts/runs") / f"snn_rr_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = _run_signature(args)

    existing_config = output_dir / "run_config.json"
    if existing_config.is_file():
        existing = json.loads(existing_config.read_text(encoding="utf-8"))
        if existing.get("run_signature") != signature:
            raise RuntimeError(
                f"output directory contains a different run signature: {output_dir}"
            )
        if existing.get("cache_provenance") != cache_provenance:
            raise RuntimeError(
                f"output directory contains different cache provenance: {output_dir}"
            )
        expected_authority = (
            split_authority.checkpoint_provenance()
            if split_authority is not None
            else None
        )
        if existing.get("split_authority") != expected_authority:
            raise RuntimeError(
                f"output directory contains a different split authority: {output_dir}"
            )
    run_arguments = dict(vars(args))
    if split_authority is None:
        run_arguments.pop("identity_split_manifest", None)
        run_arguments.pop("identity_split_manifest_sha256", None)
    run_config = {
        "run_signature": signature,
        "arguments": run_arguments,
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "determinism": determinism_audit(args.deterministic),
        "cache_provenance": cache_provenance,
        "claim_classification": claim_classification,
        "commercial_claim_allowed": False,
        "cache_shape": {
            "maps": list(cache.maps.shape),
            "aux": list(cache.aux.shape),
        },
        "causal_history_feature_names": history_names,
        "map_branch_interpretation": {
            "selection": args.map_branch,
            "stored_range_bins": stored_range_bins,
            "model_range_bins": int(cache.maps.shape[-1] // args.input_branches),
            "input_branches": args.input_branches,
            "iq_interpretation_is_hypothesis": True,
            "aux_disabled_for_pure_ablation": bool(
                args.map_branch != "both" and not args.keep_aux_in_branch_ablation
            ),
        },
        "auxiliary_layout": (
            asdict(auxiliary_layout) if auxiliary_layout is not None else None
        ),
        "auxiliary_frequency_range_hz": (
            list(auxiliary_frequency_range_hz)
            if auxiliary_frequency_range_hz is not None
            else None
        ),
        "radar_dropout_implementation": (
            "coupled map/per-radar-aux training mask; the model conservatively suppresses "
            "all auxiliary evidence whenever fewer than three current radars are available"
        ),
        "quality_target_definition": (
            "radar_observable is the cached classical estimator's <=2 bpm success label; "
            "it is not a ground-truth sensor-observability annotation"
        ),
        "evaluation_caveat": (
            (
                "The explicit prediction partition is identity-disjoint and manifest-bound, "
                "but it is a single custom split rather than OOF; its intended nesting or "
                "prospective role must be interpreted from the campaign protocol."
            )
            if split_authority is not None
            else (
                "OOF is identity-grouped, but choosing hyperparameters from these same outer-fold "
                "results is exploratory; a locked prospective cohort is required for an unbiased final claim"
            )
        ),
        "valid_reference_windows": int(
            cache.metadata["reference_valid"].to_numpy(dtype=bool).sum()
        ),
        "identities": int(cache.metadata["identity"].nunique()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if split_authority is not None:
        run_config["split_authority"] = split_authority.checkpoint_provenance()
    write_json(existing_config, run_config)
    if split_authority is None:
        write_json(
            output_dir / "fold_assignments.json",
            {
                "identity_to_fold": identity_to_fold,
                "validation_rule": "validation_fold = (test_fold + 1) mod n_folds",
            },
        )
    else:
        write_json(
            output_dir / "split_authority.json",
            split_authority.run_provenance(),
        )
    print(
        f"device={device} amp={amp_enabled} folds={selected_folds} "
        f"output={output_dir} signature={signature}",
        flush=True,
    )

    fold_reports: dict[str, Any] = {}
    for fold in selected_folds:
        if split_authority is not None:
            # Custom split IDs are provenance identifiers and may exceed the
            # NumPy legacy seeder's uint32 range.  Derive a stable bounded seed
            # without changing the historical rotating-fold seed sequence.
            seed_material = (
                f"{args.seed}:{fold}:{split_authority.content_sha256}"
            ).encode("utf-8")
            fold_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:4], "big"
            ) % (2**32 - 4)
        else:
            fold_seed = args.seed + 1009 * fold
        seed_everything(fold_seed, deterministic=args.deterministic)
        if split_authority is not None:
            explicit = split_authority.explicit_indices(
                cache.metadata, include_invalid=args.include_invalid
            )
            train_index = explicit.train_index
            validation_index = explicit.validation_index
            # The existing train/evaluate mechanics call this the test index;
            # only custom mode maps the manifest's explicit prediction role to
            # that internal slot.  It is never fed to legacy OOF aggregation.
            test_index = explicit.prediction_index
            split = explicit.split
            split_authority.validate_scaler_indices(cache.metadata, train_index)
        else:
            if assignment is None:  # defensive assertion for type narrowing.
                raise RuntimeError("legacy fold assignments were not constructed")
            train_index, validation_index, test_index, split = _fold_split(
                cache.metadata,
                assignment,
                fold,
                args.folds,
                include_invalid=args.include_invalid,
            )
        aux_center, aux_scale = fit_aux_scaler(cache.aux, train_index)
        if args.use_aux:
            if split_authority is None:
                aux_scaled = transform_aux(cache.aux, aux_center, aux_scale)
            else:
                # Do not even pass excluded rows through the fitted transform.
                # The full-size neutral buffer preserves global cache indices,
                # while only rows reachable by a loader are materialized.
                aux_scaled = np.zeros(cache.aux.shape, dtype=np.float32)
                reachable_index = np.unique(
                    np.concatenate(
                        [train_index, validation_index, test_index]
                    )
                )
                aux_scaled[reachable_index] = transform_aux(
                    cache.aux[reachable_index], aux_center, aux_scale
                )
        else:
            aux_scaled = np.empty((len(cache.metadata), 0), dtype=np.float32)
            aux_center = np.empty(0, dtype=np.float32)
            aux_scale = np.empty(0, dtype=np.float32)
        aux_dim = int(aux_scaled.shape[1])
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_record: dict[str, Any] = {
            **split,
            "train_windows": len(train_index),
            "validation_windows": len(validation_index),
            "include_invalid_training_windows": args.include_invalid,
        }
        if split_authority is not None:
            split_record.update(
                fold_seed=fold_seed,
                prediction_windows=len(test_index),
                excluded_windows=int(
                    cache.metadata["identity"]
                    .astype(str)
                    .isin(split_authority.excluded_identities)
                    .sum()
                ),
                split_manifest_content_sha256=split_authority.content_sha256,
            )
        else:
            split_record["test_windows"] = len(test_index)
        write_json(fold_dir / "split.json", split_record)

        train_loader = make_loader(
            cache,
            aux_scaled,
            train_index,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            seed=fold_seed,
            train=True,
            valid_boost=args.valid_boost,
            rr_balance_power=args.rr_balance_power,
            rr_balance_bin_width=args.rr_balance_bin_width,
            samples_per_epoch=args.samples_per_epoch,
            auxiliary_layout=auxiliary_layout,
        )
        validation_loader = make_loader(
            cache,
            aux_scaled,
            validation_index,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            seed=fold_seed + 1,
            train=False,
            auxiliary_layout=auxiliary_layout,
        )
        test_loader = make_loader(
            cache,
            aux_scaled,
            test_index,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            seed=fold_seed + 2,
            train=False,
            auxiliary_layout=auxiliary_layout,
        )
        quality_positive_weight = _quality_positive_weight(
            cache.metadata, train_index
        )
        fold_report: dict[str, Any] = {"split": split, "models": {}}
        teacher: nn.Module | None = None
        frequency_range_hz = (
            float(cache.frequencies_hz[0]),
            float(cache.frequencies_hz[-1]),
        )

        if args.model in {"teacher", "both"}:
            kwargs = _model_kwargs(
                "teacher",
                args,
                aux_dim,
                base_aux_dim,
                frequency_range_hz,
                auxiliary_frequency_range_hz,
            )
            teacher = build_model("teacher", kwargs)
            teacher, stage = train_stage(
                model=teacher,
                model_type="teacher",
                model_kwargs=kwargs,
                train_loader=train_loader,
                validation_loader=validation_loader,
                metadata=cache.metadata,
                device=device,
                fold_dir=fold_dir,
                fold=fold,
                split=split,
                aux_center=aux_center,
                aux_scale=aux_scale,
                run_signature=signature,
                args=args,
                quality_positive_weight=quality_positive_weight,
                auxiliary_layout=auxiliary_layout,
                cache_provenance=cache_provenance,
                split_authority_provenance=(
                    split_authority.checkpoint_provenance()
                    if split_authority is not None
                    else None
                ),
            )
            test_prediction = predict(
                teacher,
                test_loader,
                device,
                amp=amp_enabled,
                max_batches=args.max_eval_batches,
            )
            test_summary = detailed_prediction_summary(
                test_prediction,
                cache.metadata,
                bootstrap_samples=args.bootstrap_samples,
                coverages=args.coverages,
                rr_range=args.rr_range,
                alias_target_tolerance_bpm=args.alias_target_tolerance_bpm,
            )
            save_prediction_bundle(
                fold_dir
                / (
                    "teacher_prediction_predictions.npz"
                    if split_authority is not None
                    else "teacher_test_predictions.npz"
                ),
                test_prediction,
                fold=fold,
                run_signature=signature,
            )
            stage["prediction" if split_authority is not None else "test"] = (
                test_summary
            )
            fold_report["models"]["teacher"] = stage

        if args.model in {"snn", "both"}:
            distill_bank = None
            distillation_teacher_provenance: dict[str, Any] | None = None
            if args.distill_weight > 0:
                if teacher is None:
                    teacher_path = _resolve_teacher_checkpoint(args, fold, fold_dir)
                    if not teacher_path.is_file():
                        raise FileNotFoundError(
                            f"teacher checkpoint required for distillation: {teacher_path}. "
                            "Run --model both/teacher first or set --distill-weight 0."
                        )
                    teacher_checkpoint = torch.load(
                        teacher_path, map_location="cpu", weights_only=False
                    )
                    rr_min, rr_max = map(float, args.rr_range)
                    expected_model_context = {
                        "num_radars": 3,
                        "rr_min": rr_min,
                        "rr_max": rr_max,
                        "num_rr_bins": int(
                            round((rr_max - rr_min) / args.rr_bin_width)
                        )
                        + 1,
                        "aux_dim": aux_dim,
                        "input_branches": args.input_branches,
                        "input_frequency_min_hz": frequency_range_hz[0],
                        "input_frequency_max_hz": frequency_range_hz[1],
                    }
                    current_run_context = {
                        "folds": args.folds,
                        "rr_range": list(map(float, args.rr_range)),
                        "rr_bin_width": float(args.rr_bin_width),
                        "map_branch": args.map_branch,
                        "input_branches": args.input_branches,
                        "use_aux": args.use_aux,
                        "causal_history": args.causal_history,
                        "cache_dir": args.cache_dir,
                        "cache_shape": {
                            "maps": list(cache.maps.shape),
                            "aux": list(cache.aux.shape),
                        },
                        "cache_provenance": cache_provenance,
                        "split_authority_provenance": (
                            split_authority.checkpoint_provenance()
                            if split_authority is not None
                            else None
                        ),
                    }
                    distillation_teacher_provenance = (
                        validate_external_teacher_checkpoint(
                            teacher_checkpoint,
                            path=teacher_path,
                            fold=fold,
                            split=split,
                            expected_model_context=expected_model_context,
                            aux_center=aux_center,
                            aux_scale=aux_scale,
                            current_run_context=current_run_context,
                        )
                    )
                    teacher = build_model(
                        teacher_checkpoint["model_type"],
                        teacher_checkpoint["model_kwargs"],
                    )
                    teacher.load_state_dict(teacher_checkpoint["model_state"])
                    teacher = teacher.to(device)
                else:
                    teacher_path = fold_dir / "teacher_best.pt"
                    teacher_checkpoint = torch.load(
                        teacher_path, map_location="cpu", weights_only=False
                    )
                    distillation_teacher_provenance = teacher_checkpoint_provenance(
                        teacher_path, teacher_checkpoint
                    )
                valid_train_index = train_index[
                    cache.metadata.iloc[train_index]["reference_valid"].to_numpy(
                        dtype=bool
                    )
                ]
                distill_loader = make_loader(
                    cache,
                    aux_scaled,
                    valid_train_index,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    device=device,
                    seed=fold_seed + 3,
                    train=False,
                    auxiliary_layout=auxiliary_layout,
                )
                distill_bank = precompute_teacher_bank(
                    teacher,
                    distill_loader,
                    device,
                    total_rows=len(cache.metadata),
                    num_bins=teacher.num_rr_bins,
                    amp=amp_enabled,
                )
                teacher = teacher.cpu()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            kwargs = _model_kwargs(
                "snn",
                args,
                aux_dim,
                base_aux_dim,
                frequency_range_hz,
                auxiliary_frequency_range_hz,
            )
            snn_model = build_model("snn", kwargs)
            snn_model, stage = train_stage(
                model=snn_model,
                model_type="snn",
                model_kwargs=kwargs,
                train_loader=train_loader,
                validation_loader=validation_loader,
                metadata=cache.metadata,
                device=device,
                fold_dir=fold_dir,
                fold=fold,
                split=split,
                aux_center=aux_center,
                aux_scale=aux_scale,
                run_signature=signature,
                args=args,
                quality_positive_weight=quality_positive_weight,
                auxiliary_layout=auxiliary_layout,
                cache_provenance=cache_provenance,
                distill_bank=distill_bank,
                distillation_teacher_provenance=distillation_teacher_provenance,
                split_authority_provenance=(
                    split_authority.checkpoint_provenance()
                    if split_authority is not None
                    else None
                ),
            )
            test_prediction = predict(
                snn_model,
                test_loader,
                device,
                amp=amp_enabled,
                max_batches=args.max_eval_batches,
            )
            test_summary = detailed_prediction_summary(
                test_prediction,
                cache.metadata,
                bootstrap_samples=args.bootstrap_samples,
                coverages=args.coverages,
                rr_range=args.rr_range,
                alias_target_tolerance_bpm=args.alias_target_tolerance_bpm,
            )
            save_prediction_bundle(
                fold_dir
                / (
                    "snn_prediction_predictions.npz"
                    if split_authority is not None
                    else "snn_test_predictions.npz"
                ),
                test_prediction,
                fold=fold,
                run_signature=signature,
            )
            stage["prediction" if split_authority is not None else "test"] = (
                test_summary
            )
            fold_report["models"]["snn"] = stage

        fold_reports[str(fold)] = fold_report
        write_json(fold_dir / "report.json", fold_report)

    report: dict[str, Any] = {
        "run_signature": signature,
        "cache_provenance": cache_provenance,
        "claim_classification": claim_classification,
        "commercial_claim_allowed": False,
        "determinism": determinism_audit(args.deterministic),
        "selected_folds": selected_folds,
        "folds": fold_reports,
        "oof": {},
    }
    if split_authority is None:
        for model_type in ("teacher", "snn"):
            summary = aggregate_oof(
                output_dir,
                model_type,
                cache.metadata,
                run_signature=signature,
                expected_folds=args.folds,
                bootstrap_samples=args.bootstrap_samples,
                coverages=args.coverages,
                rr_range=args.rr_range,
                alias_target_tolerance_bpm=args.alias_target_tolerance_bpm,
            )
            if summary is not None:
                report["oof"][model_type] = summary
    else:
        # A single custom prediction partition is not OOF.  Calling the legacy
        # aggregator here would silently compare its row count to the full
        # cache and could mislabel a nested/prospective result as partial OOF.
        report["split_authority"] = split_authority.run_provenance()
        report["prediction"] = {
            model_type: fold_reports[str(split_authority.fold_id)]["models"][
                model_type
            ]["prediction"]
            for model_type in ("teacher", "snn")
            if model_type
            in fold_reports[str(split_authority.fold_id)]["models"]
        }
    write_json(output_dir / "metrics.json", report)
    print(f"completed output={output_dir}", flush=True)
    return report


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    return value


def build_parser(config: Mapping[str, Any]) -> argparse.ArgumentParser:
    data = dict(config.get("data", {}))
    model = dict(config.get("model", {}))
    training = dict(config.get("training", {}))
    evaluation = dict(config.get("evaluation", {}))
    parser = argparse.ArgumentParser(
        description="Identity-grouped ANN teacher and distilled SNN training"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path(data.get("cache_dir", "artifacts/cache/rf32s")))
    parser.add_argument(
        "--cache-trust-mode",
        choices=("scientific", "acquisition-diagnostic", "legacy"),
        default="scientific",
        help=(
            "scientific (default) requires a verified v2 full-cohort acquisition "
            "cache; acquisition-diagnostic explicitly declares a structural-mask-aware "
            "diagnostic-only path and forbids scientific claims; legacy is an explicit noncommercial "
            "compatibility mode for historical caches"
        ),
    )
    parser.add_argument(
        "--identity-split-manifest",
        type=Path,
        help=(
            "content-addressed explicit train/validation/prediction/excluded "
            "identity split; bypasses rotating fold construction"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", choices=("teacher", "snn", "both"), default="both")
    parser.add_argument("--fold", default="all", help="all, one zero-based fold, or comma list")
    parser.add_argument("--folds", type=int, default=int(training.get("folds", 6)))
    parser.add_argument("--epochs", type=int, default=int(training.get("epochs", 80)))
    parser.add_argument("--batch-size", type=int, default=int(training.get("batch_size", 48)))
    parser.add_argument("--learning-rate", type=float, default=float(training.get("learning_rate", 1e-3)))
    parser.add_argument("--weight-decay", type=float, default=float(training.get("weight_decay", 1e-4)))
    parser.add_argument("--patience", type=int, default=int(training.get("patience", 12)))
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=bool(training.get("amp", True)))
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=int(config.get("seed", 20260827)))
    parser.add_argument("--num-threads", type=int, default=None)

    rr_range = data.get("rr_range_bpm", [6.0, 45.0])
    parser.add_argument("--rr-range", type=float, nargs=2, default=tuple(map(float, rr_range)), metavar=("MIN", "MAX"))
    parser.add_argument("--rr-bin-width", type=float, default=float(data.get("rr_bin_width_bpm", 0.25)))
    parser.add_argument("--preset", choices=("tiny", "compact", "default"), default="default")
    parser.add_argument("--simulation-steps", type=int, default=int(model.get("simulation_steps", 12)))
    parser.add_argument("--hidden-dim", type=int, default=int(model.get("hidden_dim", 192)))
    parser.add_argument("--radar-dropout", type=float, default=float(model.get("radar_dropout", 0.20)))
    parser.add_argument("--map-branch", choices=("both", "raw", "phase"), default="both", help="I/Q interpretation ablation; raw/phase modes use one 91-bin half")
    parser.add_argument("--use-aux", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--aux-fusion",
        choices=("flat", "structured"),
        default="flat",
        help="flat legacy MLP or frequency-topology-preserving cached-spectrum fusion",
    )
    parser.add_argument(
        "--exact-aux-alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="align full auxiliary spectra to the pair-pooled map grid exactly",
    )
    parser.add_argument(
        "--harmonic-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="score each RR candidate from auxiliary evidence at f, f/2, f/3 and f/4",
    )
    parser.add_argument(
        "--alias-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="gate the harmonic residual with a learned direct-vs-alias probability",
    )
    parser.add_argument("--keep-aux-in-branch-ablation", action=argparse.BooleanOptionalAction, default=False, help="retain inseparable raw+phase aggregate aux in raw/phase runs (not a pure ablation)")
    parser.add_argument("--causal-history", action=argparse.BooleanOptionalAction, default=True, help="append strictly causal, radar-only history features")
    parser.add_argument("--include-invalid", action=argparse.BooleanOptionalAction, default=False, help="include invalid-reference windows as quality-head negatives")
    parser.add_argument("--valid-boost", type=float, default=2.0, help="valid-label sampling multiplier within each identity")
    parser.add_argument("--rr-balance-power", type=float, default=0.0, help="inverse-frequency RR-band sampling exponent (0 disables)")
    parser.add_argument("--rr-balance-bin-width", type=float, default=5.0, help="RR-band width in bpm for sampling balance")
    parser.add_argument("--samples-per-epoch", type=int)

    parser.add_argument("--quality-loss-weight", type=float, default=0.15)
    parser.add_argument("--spike-rate-weight", type=float, default=float(training.get("spike_rate_weight", 5e-4)))
    parser.add_argument("--distill-weight", type=float, default=0.35)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--distill-error-gate-bpm",
        type=float,
        default=0.0,
        help="softly suppress KD where the teacher misses the training target; 0 disables",
    )
    parser.add_argument(
        "--tail-loss-weight",
        type=float,
        default=0.0,
        help="weight of a separately normalized robust distance loss in the high-RR band",
    )
    parser.add_argument("--tail-min-bpm", type=float, default=22.0)
    parser.add_argument("--tail-max-bpm", type=float, default=35.0)
    parser.add_argument(
        "--tail-underprediction-ratio",
        type=float,
        default=1.0,
        help="extra cost multiplier for tail underprediction",
    )
    parser.add_argument(
        "--alias-loss-weight",
        type=float,
        default=0.0,
        help="weight of confident direct-vs-harmonic-alias gate supervision",
    )
    parser.add_argument(
        "--alias-positive-weight",
        type=float,
        default=3.0,
        help="positive-class BCE weight for the supervised alias gate",
    )
    parser.add_argument(
        "--alias-target-tolerance-bpm",
        type=float,
        default=2.0,
        help="maximum best harmonic-candidate error for a hard alias label",
    )
    parser.add_argument("--teacher-checkpoint", help="path or format string containing {fold}")

    parser.add_argument("--bootstrap-samples", type=int, default=int(evaluation.get("bootstrap_samples", 2000)))
    parser.add_argument("--coverages", type=float, nargs="+", default=list(map(float, evaluation.get("uncertainty_coverages", [1.0, 0.9, 0.8, 0.7, 0.5]))))
    parser.add_argument("--max-train-batches", type=int, help="debug/smoke limit per epoch")
    parser.add_argument("--max-eval-batches", type=int, help="debug/smoke limit per evaluation")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    known, _ = bootstrap.parse_known_args(argv)
    config = _load_config(known.config)
    parser = build_parser(config)
    args = parser.parse_args(argv)
    if args.rr_range[1] <= args.rr_range[0]:
        parser.error("--rr-range MAX must exceed MIN")
    if args.rr_bin_width <= 0:
        parser.error("--rr-bin-width must be positive")
    if not 0 <= args.radar_dropout < 1:
        parser.error("--radar-dropout must be in [0, 1)")
    if args.distill_temperature <= 0 or args.distill_weight < 0:
        parser.error("distillation temperature must be positive and weight non-negative")
    if args.distill_error_gate_bpm < 0:
        parser.error("--distill-error-gate-bpm must be non-negative")
    if args.tail_loss_weight < 0 or args.tail_underprediction_ratio <= 0:
        parser.error("tail loss weight must be non-negative and ratio positive")
    if args.tail_max_bpm <= args.tail_min_bpm:
        parser.error("--tail-max-bpm must exceed --tail-min-bpm")
    if (
        args.alias_loss_weight < 0
        or args.alias_positive_weight <= 0
        or args.alias_target_tolerance_bpm <= 0
    ):
        parser.error("alias loss weight must be non-negative and its scales positive")
    if args.exact_aux_alignment and args.aux_fusion != "structured":
        parser.error("--exact-aux-alignment requires --aux-fusion structured")
    if args.alias_gate and not args.harmonic_head:
        parser.error("--alias-gate requires --harmonic-head")
    if args.alias_loss_weight > 0 and not args.alias_gate:
        parser.error("--alias-loss-weight requires --alias-gate")
    if args.rr_balance_power < 0 or args.rr_balance_bin_width <= 0:
        parser.error("RR balance power must be non-negative and bin width positive")
    if args.harmonic_head and not args.use_aux:
        parser.error("--harmonic-head requires --use-aux")
    if any(not 0 < value <= 1 for value in args.coverages):
        parser.error("all --coverages values must be in (0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
