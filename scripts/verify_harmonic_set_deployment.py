#!/usr/bin/env python3
"""Verify a locked HCS checkpoint as a label-free streaming deployment unit.

This verifier deliberately does not import the training data loader.  It reads
only the cache columns needed to establish chronological session boundaries and
the locked label-free forward arrays.  An i3 checkpoint may additionally bind
the strict cache-indexed fallback estimate and uncertainty as its causal
posterior anchor.  In particular, reference RR, reference
validity, fold, identity, protocol, and any test prediction artifact are never
opened by parity or latency code.

The resulting report is a retrospective engineering verification.  Passing it
does not authorize a commercial claim; prospective cohort validation remains a
separate release gate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_rr.harmonic_set_models import (  # noqa: E402
    HarmonicCandidateSetEpisodeSNN,
    HarmonicSetState,
)


SCHEMA_VERSION = 1
DEPLOYMENT_METADATA_COLUMNS = ("cache_index", "session_id", "window_number")
FORWARD_ARRAY_FILES = {
    "node_features": "node_features.npy",
    "candidate_rr": "candidate_bpm.npy",
    "candidate_mask": "candidate_mask.npy",
    "radar_mask": "joint_radar_mask.npy",
}
BASE_TIME_OUTPUT_KEYS = (
    "candidate_logits",
    "candidate_probabilities",
    "candidate_residual_bpm",
    "candidate_mean_bpm",
    "candidate_scale_bpm",
    "factor_logits",
    "quality_logit",
    "quality",
    "selected_index",
    "selected_probability",
    "source_rr",
    "source_scale_bpm",
    "source_available",
    "node_embeddings",
    "candidate_attention",
    "state_sequence",
    "spike_sequence",
)
ANCHOR_FORWARD_KEYS = ("anchor_rr", "anchor_std", "anchor_available")
ALL_NONEMPTY_RADAR_MASKS = tuple(
    tuple(bool(bits & (1 << index)) for index in range(3))
    for bits in range(1, 8)
)
RADAR_FEATURE_PATTERN = re.compile(r"(?:^|_)radar([123])(?:_|$)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact root must be an object: {path}")
    return value


def _require_hash(path: Path, expected: object, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"locked artifact is missing: {label}: {path}")
    actual = sha256_file(path)
    if actual != str(expected):
        raise RuntimeError(f"locked artifact tamper detected: {label}")
    return actual


def _cache_content_sha256(manifest: Mapping[str, Any]) -> str:
    value = dict(manifest)
    value.pop("content_sha256", None)
    return canonical_json_sha256(value)


def _resolved_path(value: object) -> Path:
    return Path(str(value)).expanduser().resolve()


def validate_locked_artifacts(
    run_dir: Path,
    cache_root: Path,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Validate every lock/config/source/cache binding before model execution."""

    run_dir = run_dir.expanduser().resolve()
    cache_root = cache_root.expanduser().resolve()
    checkpoint_path = (
        run_dir / "best_checkpoint.pt"
        if checkpoint_path is None
        else checkpoint_path.expanduser().resolve()
    )
    lock_path = run_dir / "selection_lock.json"
    manifest_path = run_dir / "run_manifest.json"
    scaler_path = run_dir / "scaler.json"
    policy_path = run_dir / "fallback_policy.json"
    cache_manifest_path = cache_root / "manifest.json"
    lock = _load_json(lock_path)
    run_manifest = _load_json(manifest_path)
    cache_manifest = _load_json(cache_manifest_path)

    if lock.get("retrospective_only") is not True:
        raise RuntimeError("selection lock does not declare retrospective_only")
    if run_manifest.get("retrospective_only") is not True:
        raise RuntimeError("run manifest does not declare retrospective_only")
    if run_manifest.get("commercial_claim_authorized") is not False:
        raise RuntimeError("run manifest must explicitly deny a commercial claim")

    checked_files: dict[str, dict[str, Any]] = {}
    locked_files = {
        "checkpoint": (checkpoint_path, lock.get("checkpoint_sha256")),
        "scaler": (scaler_path, lock.get("scaler_sha256")),
        "fallback_policy": (policy_path, lock.get("policy_sha256")),
        "run_manifest": (manifest_path, lock.get("run_manifest_sha256")),
    }
    for name, (path, expected) in locked_files.items():
        digest = _require_hash(path, expected, name)
        checked_files[name] = {
            "path": str(path), "sha256": digest, "bytes": path.stat().st_size
        }

    cache_manifest_digest = sha256_file(cache_manifest_path)
    if cache_manifest_digest != str(lock.get("cache_manifest_sha256", "")):
        raise RuntimeError("selection lock cache manifest binding mismatch")
    if cache_manifest.get("complete") is not True or int(
        cache_manifest.get("format_version", -1)
    ) != 1:
        raise RuntimeError("cache manifest is incomplete or incompatible")
    if cache_manifest.get("content_sha256") != _cache_content_sha256(cache_manifest):
        raise RuntimeError("cache manifest canonical content hash mismatch")

    run_input = run_manifest.get("input_bindings", {})
    if not isinstance(run_input, Mapping) or str(
        run_input.get("cache_manifest_sha256", "")
    ) != cache_manifest_digest:
        raise RuntimeError("run manifest cache binding mismatch")

    effective = run_manifest.get("iteration_effective_configuration")
    effective_digest = run_manifest.get("iteration_effective_configuration_sha256")
    if not isinstance(effective, Mapping) or not isinstance(effective_digest, str):
        raise RuntimeError("run manifest lacks a canonical effective configuration")
    if canonical_json_sha256(effective) != effective_digest:
        raise RuntimeError("effective configuration canonical hash mismatch")
    if str(lock.get("effective_configuration_sha256", "")) != effective_digest:
        raise RuntimeError("selection lock effective configuration binding mismatch")
    data_bindings = effective.get("data_bindings", {})
    if not isinstance(data_bindings, Mapping) or str(
        data_bindings.get("cache_manifest_sha256", "")
    ) != cache_manifest_digest:
        raise RuntimeError("effective configuration cache binding mismatch")

    forward_allowlist = effective.get("forward_allowlist", ())
    model_config_candidate = run_manifest.get("model_config")
    if not isinstance(model_config_candidate, Mapping):
        raise RuntimeError("run manifest lacks a model configuration")
    anchor_enabled = bool(model_config_candidate.get("anchor_enabled", False))
    required_forward = {
        "node_features", "candidate_rr", "candidate_mask", "radar_mask",
        "sequence_mask", "causal_state", "reset_mask",
    }
    if anchor_enabled:
        required_forward.update(ANCHOR_FORWARD_KEYS)
    if set(map(str, forward_allowlist)) != required_forward:
        raise RuntimeError("effective forward allowlist is not the locked label-free contract")

    source_bindings = lock.get("source_bindings")
    if not isinstance(source_bindings, Mapping) or not source_bindings:
        raise RuntimeError("selection lock has no source/config bindings")
    if run_manifest.get("source_and_config_bindings") != source_bindings:
        raise RuntimeError("run and selection-lock source bindings disagree")
    checked_sources: dict[str, dict[str, Any]] = {}
    for name, raw_binding in source_bindings.items():
        if not isinstance(raw_binding, Mapping):
            raise RuntimeError(f"invalid source/config binding: {name}")
        path = _resolved_path(raw_binding.get("path", ""))
        digest = _require_hash(path, raw_binding.get("sha256"), f"source:{name}")
        checked_sources[str(name)] = {
            "path": str(path), "sha256": digest, "bytes": path.stat().st_size
        }

    outputs = cache_manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise RuntimeError("cache manifest contains no output bindings")
    checked_cache_outputs: dict[str, dict[str, Any]] = {}
    for name, raw_binding in outputs.items():
        if not isinstance(raw_binding, Mapping):
            raise RuntimeError(f"invalid cache output binding: {name}")
        filename = str(raw_binding.get("filename", ""))
        path = (cache_root / filename).resolve()
        try:
            path.relative_to(cache_root)
        except ValueError as exc:
            raise RuntimeError(f"cache output escapes cache root: {name}") from exc
        digest = _require_hash(path, raw_binding.get("sha256"), f"cache:{name}")
        if path.stat().st_size != int(raw_binding.get("bytes", -1)):
            raise RuntimeError(f"cache output size mismatch: {name}")
        checked_cache_outputs[str(name)] = {
            "path": str(path), "sha256": digest, "bytes": path.stat().st_size
        }

    for filename in FORWARD_ARRAY_FILES.values():
        if not (cache_root / filename).is_file():
            raise RuntimeError(f"required forward cache array is absent: {filename}")
    feature_document = _load_json(cache_root / "feature_names.json")
    declared_forward = set(map(str, feature_document.get("forward_arrays", ())))
    if not {"node_features", "candidate_bpm", "candidate_mask", "joint_radar_mask"} <= declared_forward:
        raise RuntimeError("cache feature manifest lacks the required forward arrays")
    forbidden = set(map(str, feature_document.get("forbidden_target_qc_forward_fields", ())))
    if forbidden & declared_forward:
        raise RuntimeError("cache declares target/QC fields in its forward arrays")

    anchor_binding: dict[str, Any] | None = None
    fallback_path_raw = run_input.get("fallback_oof_path")
    fallback_sha_raw = run_input.get("fallback_oof_sha256")
    if fallback_path_raw is not None or fallback_sha_raw is not None or anchor_enabled:
        if fallback_path_raw is None or fallback_sha_raw is None:
            raise RuntimeError("anchor/fallback input binding is incomplete")
        fallback_path = _resolved_path(fallback_path_raw)
        fallback_digest = _require_hash(
            fallback_path, fallback_sha_raw, "label-free anchor/fallback input"
        )
        if "fallback_oof_sha256" in lock and str(
            lock.get("fallback_oof_sha256")
        ) != fallback_digest:
            raise RuntimeError("selection lock fallback/anchor binding mismatch")
        effective_fallback = data_bindings.get("fallback_oof_sha256")
        if effective_fallback is not None and str(effective_fallback) != fallback_digest:
            raise RuntimeError("effective configuration fallback/anchor binding mismatch")
        anchor_binding = {
            "path": str(fallback_path),
            "sha256": fallback_digest,
            "bytes": fallback_path.stat().st_size,
            "label_free_columns_only": [
                value
                for value in _label_free_anchor_columns(fallback_path)
                if value is not None
            ],
            "uncertainty_default_bpm_when_column_absent": 2.0,
        }

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("checkpoint root is not a mapping")
    model_config = run_manifest.get("model_config")
    if not isinstance(model_config, Mapping) or checkpoint.get("model_config") != model_config:
        raise RuntimeError("checkpoint and run model configurations disagree")
    if effective.get("model") != model_config:
        raise RuntimeError("effective and run model configurations disagree")
    if str(checkpoint.get("effective_configuration_sha256", "")) != effective_digest:
        raise RuntimeError("checkpoint effective configuration binding mismatch")
    for field in ("seed", "adaptive_iteration"):
        if field in lock and checkpoint.get(field) != lock.get(field):
            raise RuntimeError(f"checkpoint and lock disagree on {field}")
    if "fold" in checkpoint and checkpoint.get("fold") != lock.get("outer_fold"):
        raise RuntimeError("checkpoint and lock disagree on outer fold")

    return {
        "run_dir": str(run_dir),
        "cache_root": str(cache_root),
        "checkpoint_path": str(checkpoint_path),
        "selection_lock_sha256": sha256_file(lock_path),
        "cache_manifest": {
            "path": str(cache_manifest_path),
            "sha256": cache_manifest_digest,
            "content_sha256": cache_manifest.get("content_sha256"),
        },
        "locked_files": checked_files,
        "source_and_config_files": checked_sources,
        "cache_outputs": checked_cache_outputs,
        "effective_configuration_sha256": effective_digest,
        "model_config": dict(model_config),
        "anchor_enabled": anchor_enabled,
        "anchor_input": anchor_binding,
        "row_count": int(cache_manifest.get("row_count", -1)),
        "retrospective_only": True,
        "commercial_claim_authorized": False,
    }


@dataclass(frozen=True, slots=True)
class Scaler:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def load(cls, path: Path, feature_count: int) -> "Scaler":
        document = _load_json(path)
        center = np.asarray(document.get("center"), dtype=np.float32).reshape(-1)
        scale = np.asarray(document.get("scale"), dtype=np.float32).reshape(-1)
        if center.shape != (feature_count,) or scale.shape != (feature_count,):
            raise RuntimeError("locked scaler feature shape disagrees with cache")
        if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise RuntimeError("locked scaler is non-finite or non-positive")
        return cls(center=center, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip((values - self.center) / self.scale, -8.0, 8.0).astype(
            np.float32, copy=False
        )


@dataclass(frozen=True, slots=True)
class ForwardBatch:
    node_features: Tensor
    candidate_rr: Tensor
    candidate_mask: Tensor
    sequence_mask: Tensor
    radar_mask: Tensor
    reset_mask: Tensor
    anchor_rr: Tensor | None = None
    anchor_std: Tensor | None = None
    anchor_available: Tensor | None = None

    @property
    def windows(self) -> int:
        return int(self.node_features.shape[1])

    def time_slice(self, start: int, stop: int) -> "ForwardBatch":
        return ForwardBatch(**{
            name: (
                None
                if getattr(self, name) is None
                else getattr(self, name)[:, start:stop].clone()
            )
            for name in self.__dataclass_fields__
        })

    def to(self, device: torch.device) -> "ForwardBatch":
        return ForwardBatch(**{
            name: (
                None
                if getattr(self, name) is None
                else getattr(self, name).to(device=device)
            )
            for name in self.__dataclass_fields__
        })

    def model_kwargs(self) -> dict[str, Tensor]:
        result = {
            "node_features": self.node_features,
            "candidate_rr": self.candidate_rr,
            "candidate_mask": self.candidate_mask,
            "sequence_mask": self.sequence_mask,
            "radar_mask": self.radar_mask,
            "reset_mask": self.reset_mask,
        }
        anchors = (self.anchor_rr, self.anchor_std, self.anchor_available)
        if all(value is None for value in anchors):
            return result
        if any(value is None for value in anchors):
            raise RuntimeError("anchor_rr/anchor_std/anchor_available must travel together")
        result.update({
            "anchor_rr": self.anchor_rr,
            "anchor_std": self.anchor_std,
            "anchor_available": self.anchor_available,
        })
        return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DeploymentStream:
    raw_node_features: np.ndarray
    candidate_rr: np.ndarray
    candidate_mask: np.ndarray
    native_radar_mask: np.ndarray
    reset_mask: np.ndarray
    session_ids: tuple[str, ...]
    session_lengths: tuple[int, ...]
    cache_indices: np.ndarray
    feature_names: tuple[str, ...]
    scaler: Scaler
    truncated: bool
    anchor_rr: np.ndarray | None = None
    anchor_std: np.ndarray | None = None
    anchor_available: np.ndarray | None = None
    structural_fallback_rr: np.ndarray | None = None
    structural_fallback_std: np.ndarray | None = None
    structural_fallback_available: np.ndarray | None = None

    @property
    def windows(self) -> int:
        return int(len(self.cache_indices))

    def forward_batch(
        self,
        *,
        radar_override: np.ndarray | None = None,
        raw_override: np.ndarray | None = None,
    ) -> ForwardBatch:
        radar = (
            self.native_radar_mask.copy()
            if radar_override is None
            else np.asarray(radar_override, dtype=bool).copy()
        )
        if radar.shape != (self.windows, 3):
            raise ValueError("radar override must have shape [windows, 3]")
        raw = (
            self.raw_node_features.copy()
            if raw_override is None
            else np.asarray(raw_override, dtype=np.float32).copy()
        )
        if raw.shape != self.raw_node_features.shape:
            raise ValueError("raw node override shape mismatch")
        raw = zero_unavailable_radar_features(raw, radar, self.feature_names)
        node = self.scaler.transform(raw)
        anchor_values = (self.anchor_rr, self.anchor_std, self.anchor_available)
        if any(value is None for value in anchor_values) and not all(
            value is None for value in anchor_values
        ):
            raise RuntimeError("deployment stream anchor arrays are incomplete")
        return ForwardBatch(
            node_features=torch.from_numpy(node[None]),
            candidate_rr=torch.from_numpy(self.candidate_rr.copy()[None]),
            candidate_mask=torch.from_numpy(self.candidate_mask.copy()[None]),
            sequence_mask=torch.ones((1, self.windows), dtype=torch.bool),
            radar_mask=torch.from_numpy(radar[None]),
            reset_mask=torch.from_numpy(self.reset_mask.copy()[None]),
            anchor_rr=(
                None
                if self.anchor_rr is None
                else torch.from_numpy(self.anchor_rr.copy()[None])
            ),
            anchor_std=(
                None
                if self.anchor_std is None
                else torch.from_numpy(self.anchor_std.copy()[None])
            ),
            anchor_available=(
                None
                if self.anchor_available is None
                else torch.from_numpy(self.anchor_available.copy()[None])
            ),
        )


def radar_feature_columns(feature_names: Sequence[str]) -> dict[int, np.ndarray]:
    result: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for index, name in enumerate(feature_names):
        matches = {int(value) - 1 for value in RADAR_FEATURE_PATTERN.findall(str(name))}
        for radar_index in matches:
            result[radar_index].append(index)
    return {
        radar_index: np.asarray(indices, dtype=np.int64)
        for radar_index, indices in result.items()
    }


def zero_unavailable_radar_features(
    raw_node_features: np.ndarray,
    radar_mask: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Apply missing-radar semantics before the locked robust scaler."""

    values = np.asarray(raw_node_features, dtype=np.float32).copy()
    if values.ndim != 3 or radar_mask.shape != (values.shape[0], 3):
        raise ValueError("raw nodes/radar mask must be [T,K,F] and [T,3]")
    if len(feature_names) != values.shape[-1]:
        raise ValueError("feature-name count disagrees with node features")
    for radar_index, columns in radar_feature_columns(feature_names).items():
        if len(columns):
            missing_rows = np.flatnonzero(~radar_mask[:, radar_index])
            if len(missing_rows):
                values[np.ix_(missing_rows, np.arange(values.shape[1]), columns)] = 0.0
    return values


def _metadata_sessions(cache_root: Path) -> pd.DataFrame:
    """Read grouping columns only; this is the sole metadata access."""

    path = cache_root / "metadata.csv"
    try:
        frame = pd.read_csv(path, usecols=list(DEPLOYMENT_METADATA_COLUMNS))
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"deployment metadata grouping columns are invalid: {exc}") from exc
    if tuple(frame.columns) != DEPLOYMENT_METADATA_COLUMNS:
        frame = frame.loc[:, list(DEPLOYMENT_METADATA_COLUMNS)]
    cache_index = frame["cache_index"].to_numpy(np.int64)
    if not np.array_equal(cache_index, np.arange(len(frame), dtype=np.int64)):
        raise RuntimeError("cache_index is not a contiguous exact row binding")
    if frame["session_id"].isna().any() or frame["window_number"].isna().any():
        raise RuntimeError("deployment grouping metadata contains missing values")
    return frame


def _label_free_anchor_columns(path: Path) -> tuple[str, str, str | None]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), None)
    if not header:
        raise RuntimeError("anchor/fallback CSV has no header")
    prediction_column = next(
        (
            name for name in (
                "prediction_bpm", "prediction_locked_final_bpm",
                "prediction_candidate_bpm",
            )
            if name in header
        ),
        None,
    )
    std_column = next(
        (
            name for name in (
                "rr_std_bpm", "candidate_rr_std_bpm", "source_std_bpm"
            )
            if name in header
        ),
        None,
    )
    if "cache_index" not in header or prediction_column is None:
        raise RuntimeError(
            "anchor/fallback CSV needs cache_index and a supported prediction column"
        )
    return "cache_index", prediction_column, std_column


def _load_label_free_anchor_rows(
    path: Path, rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load only cache-indexed fallback estimates used by the locked i3 anchor."""

    index_column, prediction_column, std_column = _label_free_anchor_columns(path)
    accessed = [index_column, prediction_column]
    if std_column is not None:
        accessed.append(std_column)
    frame = pd.read_csv(path, usecols=accessed)
    if frame["cache_index"].duplicated().any():
        raise RuntimeError("anchor/fallback CSV cache_index is not unique")
    anchor_rr = np.zeros(rows, dtype=np.float32)
    anchor_std = np.full(rows, 2.0, dtype=np.float32)
    available = np.zeros(rows, dtype=bool)
    for row in frame.itertuples(index=False):
        index = int(getattr(row, "cache_index"))
        if not 0 <= index < rows:
            raise RuntimeError("anchor/fallback CSV cache_index is outside the cache")
        prediction = float(getattr(row, prediction_column))
        standard_deviation = (
            2.0 if std_column is None else float(getattr(row, std_column))
        )
        if np.isfinite(prediction) and np.isfinite(standard_deviation) and standard_deviation > 0:
            anchor_rr[index] = prediction
            anchor_std[index] = standard_deviation
            available[index] = True
    return anchor_rr, anchor_std, available


def load_deployment_stream(
    cache_root: Path,
    scaler_path: Path,
    *,
    requested_sessions: Sequence[str] = (),
    maximum_sessions: int = 2,
    maximum_windows_per_session: int = 0,
    anchor_input_path: Path | None = None,
    anchor_forward_enabled: bool = True,
) -> DeploymentStream:
    cache_root = cache_root.expanduser().resolve()
    metadata = _metadata_sessions(cache_root)
    node = np.load(cache_root / "node_features.npy", mmap_mode="r", allow_pickle=False)
    candidate = np.load(cache_root / "candidate_bpm.npy", mmap_mode="r", allow_pickle=False)
    candidate_mask = np.load(cache_root / "candidate_mask.npy", mmap_mode="r", allow_pickle=False)
    radar = np.load(cache_root / "joint_radar_mask.npy", mmap_mode="r", allow_pickle=False)
    if node.ndim != 3 or not (node.shape[:2] == candidate.shape == candidate_mask.shape):
        raise RuntimeError("cache forward candidate array shapes disagree")
    if node.shape[0] != len(metadata) or radar.shape != (len(metadata), 3):
        raise RuntimeError("cache forward row/radar array shapes disagree")
    if candidate.shape[1] < 1 or candidate.shape[1] > 12:
        raise RuntimeError("cache candidate count is outside [1, 12]")

    feature_document = _load_json(cache_root / "feature_names.json")
    names = tuple(map(str, feature_document.get("node_feature_names", ())))
    if len(names) != node.shape[-1]:
        raise RuntimeError("cache feature-name count disagrees with node array")
    scaler = Scaler.load(scaler_path, node.shape[-1])

    available_sessions = list(dict.fromkeys(metadata["session_id"].astype(str)))
    if requested_sessions:
        unknown = sorted(set(map(str, requested_sessions)) - set(available_sessions))
        if unknown:
            raise RuntimeError(f"requested deployment sessions are absent: {unknown}")
        selected_sessions = list(map(str, requested_sessions))
    else:
        if maximum_sessions < 1:
            raise ValueError("maximum_sessions must be positive")
        selected_sessions = available_sessions[:maximum_sessions]
    if not selected_sessions:
        raise RuntimeError("no session is available for deployment verification")

    pieces: list[np.ndarray] = []
    reset_pieces: list[np.ndarray] = []
    session_lengths: list[int] = []
    truncated = False
    for session_id in selected_sessions:
        group = metadata.loc[metadata["session_id"].astype(str) == session_id]
        group = group.sort_values(["window_number", "cache_index"], kind="stable")
        window_number = pd.to_numeric(
            group["window_number"], errors="raise"
        ).to_numpy(np.int64)
        if len(np.unique(window_number)) != len(window_number):
            raise RuntimeError(f"session has duplicate window_number values: {session_id}")
        position = group["cache_index"].to_numpy(np.int64)
        if maximum_windows_per_session > 0 and len(position) > maximum_windows_per_session:
            position = position[:maximum_windows_per_session]
            window_number = window_number[:maximum_windows_per_session]
            truncated = True
        if not len(position):
            raise RuntimeError(f"selected session contains no windows: {session_id}")
        pieces.append(position)
        local_reset = np.zeros(len(position), dtype=bool)
        local_reset[0] = True
        if len(position) > 1:
            local_reset[1:] = np.diff(window_number) != 1
        reset_pieces.append(local_reset)
        session_lengths.append(len(position))
    positions = np.concatenate(pieces)
    reset = np.concatenate(reset_pieces)
    anchor_rr: np.ndarray | None = None
    anchor_std: np.ndarray | None = None
    anchor_available: np.ndarray | None = None
    structural_fallback_rr: np.ndarray | None = None
    structural_fallback_std: np.ndarray | None = None
    structural_fallback_available: np.ndarray | None = None
    if anchor_input_path is not None:
        all_rr, all_std, all_available = _load_label_free_anchor_rows(
            anchor_input_path.expanduser().resolve(), len(metadata)
        )
        structural_fallback_rr = all_rr[positions].copy()
        structural_fallback_std = all_std[positions].copy()
        structural_fallback_available = all_available[positions].copy()
        if anchor_forward_enabled:
            anchor_rr = structural_fallback_rr.copy()
            anchor_std = structural_fallback_std.copy()
            anchor_available = structural_fallback_available.copy()
    return DeploymentStream(
        raw_node_features=np.asarray(node[positions], dtype=np.float32).copy(),
        candidate_rr=np.asarray(candidate[positions], dtype=np.float32).copy(),
        candidate_mask=np.asarray(candidate_mask[positions], dtype=bool).copy(),
        native_radar_mask=np.asarray(radar[positions], dtype=bool).copy(),
        reset_mask=reset,
        session_ids=tuple(selected_sessions),
        session_lengths=tuple(session_lengths),
        cache_indices=positions,
        feature_names=names,
        scaler=scaler,
        truncated=truncated,
        anchor_rr=anchor_rr,
        anchor_std=anchor_std,
        anchor_available=anchor_available,
        structural_fallback_rr=structural_fallback_rr,
        structural_fallback_std=structural_fallback_std,
        structural_fallback_available=structural_fallback_available,
    )


def load_model(
    checkpoint_path: Path,
    model_config: Mapping[str, Any],
    device: torch.device,
) -> HarmonicCandidateSetEpisodeSNN:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = HarmonicCandidateSetEpisodeSNN(**dict(model_config))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    parameters = set(inspect.signature(model.forward).parameters)
    base_parameters = {
        "node_features", "candidate_rr", "candidate_mask", "sequence_mask",
        "radar_mask", "state", "reset_mask",
    }
    if not base_parameters <= parameters:
        raise RuntimeError("loaded model does not expose the locked causal forward contract")
    anchor_enabled = bool(model_config.get("anchor_enabled", False))
    anchor_parameters = set(ANCHOR_FORWARD_KEYS)
    if anchor_enabled and not anchor_parameters <= parameters:
        raise RuntimeError("anchor-enabled model does not expose all locked anchor inputs")
    if hasattr(model, "anchor_enabled") and bool(model.anchor_enabled) != anchor_enabled:
        raise RuntimeError("constructed model anchor mode disagrees with model_config")
    return model.to(device=device).eval()


def _detached_state(state: HarmonicSetState) -> HarmonicSetState:
    return tuple(
        (membrane.detach(), adaptation.detach())
        for membrane, adaptation in state
    )  # type: ignore[return-value]


def _schedule_slices(windows: int, schedule: Sequence[int]) -> Iterable[tuple[int, int]]:
    if windows < 1 or not schedule or any(int(value) < 1 for value in schedule):
        raise ValueError("window count and chunk schedule must be positive")
    start = 0
    step = 0
    while start < windows:
        stop = min(windows, start + int(schedule[step % len(schedule)]))
        yield start, stop
        start = stop
        step += 1


def _time_output_keys(output: Mapping[str, Tensor | HarmonicSetState]) -> tuple[str, ...]:
    """Discover chronological outputs, including an enabled locked anchor path."""

    keys: list[str] = []
    for name, value in output.items():
        if name in {"state", "spike_rates"} or not isinstance(value, Tensor):
            continue
        if value.ndim >= 2:
            keys.append(str(name))
    missing = sorted(set(BASE_TIME_OUTPUT_KEYS) - set(keys))
    if missing:
        raise RuntimeError(f"model omitted required chronological outputs: {missing}")
    return tuple(keys)


def run_chunk_schedule(
    model: HarmonicCandidateSetEpisodeSNN,
    batch: ForwardBatch,
    schedule: Sequence[int],
) -> dict[str, Tensor | HarmonicSetState]:
    state: HarmonicSetState | None = None
    chunks: list[Mapping[str, Tensor | HarmonicSetState]] = []
    with torch.inference_mode():
        for start, stop in _schedule_slices(batch.windows, schedule):
            output = model(**batch.time_slice(start, stop).model_kwargs(), state=state)
            state = _detached_state(output["state"])  # type: ignore[arg-type]
            chunks.append(output)
    time_keys = _time_output_keys(chunks[0])
    if any(_time_output_keys(part) != time_keys for part in chunks[1:]):
        raise RuntimeError("model chronological output contract changed between chunks")
    combined: dict[str, Tensor | HarmonicSetState] = {
        key: torch.cat([part[key] for part in chunks], dim=1)  # type: ignore[list-item]
        for key in time_keys
    }
    combined["state"] = state  # type: ignore[assignment]
    return combined


def _state_tensors(state: HarmonicSetState) -> list[Tensor]:
    return [tensor for layer in state for tensor in layer]


def compare_outputs(
    reference: Mapping[str, Tensor | HarmonicSetState],
    candidate: Mapping[str, Tensor | HarmonicSetState],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    strict_pass = True
    discrete_bitwise_pass = True
    all_float_bitwise = True
    reference_keys = tuple(key for key in reference if key != "state")
    candidate_keys = tuple(key for key in candidate if key != "state")
    if reference_keys != candidate_keys:
        raise RuntimeError("parity chronological output key mismatch")
    for key in reference_keys:
        left = reference[key]
        right = candidate[key]
        assert isinstance(left, Tensor) and isinstance(right, Tensor)
        if left.shape != right.shape:
            raise RuntimeError(f"parity output shape mismatch: {key}")
        bitwise = torch.equal(left, right)
        if torch.is_floating_point(left):
            difference = (left - right).abs()
            max_absolute = float(difference.max().item()) if difference.numel() else 0.0
            strict = bool(torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False))
            all_float_bitwise &= bitwise
        else:
            max_absolute = 0.0 if bitwise else math.inf
            strict = bitwise
            discrete_bitwise_pass &= bitwise
        strict_pass &= strict
        details.append({
            "output": key,
            "bitwise_equal": bool(bitwise),
            "strict_numerical_equal": bool(strict),
            "max_absolute_difference": max_absolute,
        })

    reference_state = _state_tensors(reference["state"])  # type: ignore[arg-type]
    candidate_state = _state_tensors(candidate["state"])  # type: ignore[arg-type]
    if len(reference_state) != len(candidate_state):
        raise RuntimeError("parity state structure mismatch")
    for index, (left, right) in enumerate(zip(reference_state, candidate_state, strict=True)):
        bitwise = torch.equal(left, right)
        max_absolute = float((left - right).abs().max().item()) if left.numel() else 0.0
        strict = bool(torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False))
        strict_pass &= strict
        all_float_bitwise &= bitwise
        details.append({
            "output": f"state_tensor_{index}",
            "bitwise_equal": bool(bitwise),
            "strict_numerical_equal": bool(strict),
            "max_absolute_difference": max_absolute,
        })
    return {
        "passed": bool(strict_pass and discrete_bitwise_pass),
        "strict_numerical_pass": bool(strict_pass),
        "discrete_bitwise_pass": bool(discrete_bitwise_pass),
        "all_float_outputs_bitwise_equal": bool(all_float_bitwise),
        "atol": float(atol),
        "rtol": float(rtol),
        "details": details,
    }


def parity_verification(
    model: HarmonicCandidateSetEpisodeSNN,
    batch: ForwardBatch,
    schedules: Sequence[Sequence[int]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    whole = run_chunk_schedule(model, batch, (batch.windows,))
    cases: list[dict[str, Any]] = []
    for schedule in schedules:
        output = run_chunk_schedule(model, batch, schedule)
        comparison = compare_outputs(whole, output, atol=atol, rtol=rtol)
        cases.append({"schedule": list(map(int, schedule)), **comparison})
    one_window = run_chunk_schedule(model, batch, (1,))
    comparison = compare_outputs(whole, one_window, atol=atol, rtol=rtol)
    cases.append({"schedule": [1], "mode": "one_window_streaming", **comparison})
    if not all(case["passed"] for case in cases):
        raise RuntimeError("whole-session/chunk/streaming numerical parity failed")
    return {
        "passed": True,
        "reference_mode": "all selected sessions in one offline call with explicit reset_mask",
        "cases": cases,
    }


def session_reset_verification(
    model: HarmonicCandidateSetEpisodeSNN,
    batch: ForwardBatch,
    session_lengths: Sequence[int],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if sum(map(int, session_lengths)) != batch.windows:
        raise ValueError("session lengths do not exactly cover the deployment stream")
    combined = run_chunk_schedule(model, batch, (batch.windows,))
    pieces: list[Mapping[str, Tensor | HarmonicSetState]] = []
    start = 0
    for length in session_lengths:
        stop = start + int(length)
        fresh = batch.time_slice(start, stop)
        fresh.reset_mask[:, 0] = True
        pieces.append(run_chunk_schedule(model, fresh, (int(length),)))
        start = stop
    independent: dict[str, Tensor | HarmonicSetState] = {
        key: torch.cat([piece[key] for piece in pieces], dim=1)  # type: ignore[list-item]
        for key in pieces[0]
        if key != "state"
    }
    independent["state"] = pieces[-1]["state"]
    comparison = compare_outputs(combined, independent, atol=atol, rtol=rtol)
    if not comparison["passed"]:
        raise RuntimeError("explicit reset/session-boundary isolation failed")
    return {
        "passed": True,
        "session_count": len(session_lengths),
        "session_lengths": list(map(int, session_lengths)),
        "comparison": comparison,
    }


def _all_finite(output: Mapping[str, Tensor | HarmonicSetState]) -> bool:
    for name, value in output.items():
        tensors = _state_tensors(value) if name == "state" else [value]  # type: ignore[arg-type,list-item]
        for tensor in tensors:
            if isinstance(tensor, Tensor) and torch.is_floating_point(tensor):
                if not bool(torch.isfinite(tensor).all()):
                    return False
    return True


def route_structural_fallback(
    source_rr: Tensor,
    source_available: Tensor,
    fallback_rr: Tensor,
    fallback_available: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Route unavailable source rows without silently inventing an RR value."""

    if not (
        source_rr.shape == source_available.shape == fallback_rr.shape
        == fallback_available.shape
    ):
        raise ValueError("source/fallback tensors must have identical shapes")
    source_available = source_available.bool()
    fallback_available = fallback_available.bool()
    available = source_available | fallback_available
    prediction = torch.where(
        source_available,
        source_rr,
        torch.where(fallback_available, fallback_rr, torch.zeros_like(source_rr)),
    )
    route = torch.where(
        source_available,
        torch.ones_like(source_rr, dtype=torch.int8),
        torch.where(
            fallback_available,
            torch.full_like(source_rr, 2, dtype=torch.int8),
            torch.zeros_like(source_rr, dtype=torch.int8),
        ),
    )
    return prediction, available, route


def robustness_verification(
    model: HarmonicCandidateSetEpisodeSNN,
    stream: DeploymentStream,
    *,
    windows: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    eligible = np.flatnonzero(
        stream.native_radar_mask.all(axis=1) & stream.candidate_mask.any(axis=1)
    )
    if not len(eligible):
        raise RuntimeError("cache has no all-radar/candidate window for seven-mask verification")
    selected = eligible[: max(1, min(int(windows), len(eligible)))]
    substream = replace(
        stream,
        raw_node_features=stream.raw_node_features[selected].copy(),
        candidate_rr=stream.candidate_rr[selected].copy(),
        candidate_mask=stream.candidate_mask[selected].copy(),
        native_radar_mask=stream.native_radar_mask[selected].copy(),
        reset_mask=np.asarray([True] + [False] * (len(selected) - 1), dtype=bool),
        session_ids=("robustness_label_free_subset",),
        session_lengths=(len(selected),),
        cache_indices=stream.cache_indices[selected].copy(),
        truncated=True,
        anchor_rr=(
            None if stream.anchor_rr is None else stream.anchor_rr[selected].copy()
        ),
        anchor_std=(
            None if stream.anchor_std is None else stream.anchor_std[selected].copy()
        ),
        anchor_available=(
            None
            if stream.anchor_available is None
            else stream.anchor_available[selected].copy()
        ),
        structural_fallback_rr=(
            None
            if stream.structural_fallback_rr is None
            else stream.structural_fallback_rr[selected].copy()
        ),
        structural_fallback_std=(
            None
            if stream.structural_fallback_std is None
            else stream.structural_fallback_std[selected].copy()
        ),
        structural_fallback_available=(
            None
            if stream.structural_fallback_available is None
            else stream.structural_fallback_available[selected].copy()
        ),
    )
    columns = radar_feature_columns(stream.feature_names)
    mask_results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for mask_tuple in ALL_NONEMPTY_RADAR_MASKS:
            radar = np.tile(np.asarray(mask_tuple, dtype=bool), (substream.windows, 1))
            clean_batch = substream.forward_batch(radar_override=radar)
            clean = model(**clean_batch.model_kwargs())
            if not _all_finite(clean):
                raise RuntimeError(f"non-finite output for radar mask {mask_tuple}")
            if not bool(clean["source_available"].all()):
                raise RuntimeError(f"source unexpectedly unavailable for nonempty mask {mask_tuple}")

            corrupted_raw = substream.raw_node_features.copy()
            for radar_index, present in enumerate(mask_tuple):
                if not present and len(columns[radar_index]):
                    corrupted_raw[..., columns[radar_index]] = np.nan
            corrupt_batch = substream.forward_batch(
                radar_override=radar, raw_override=corrupted_raw
            )
            corrupt = model(**corrupt_batch.model_kwargs())
            comparison = compare_outputs(clean, corrupt, atol=atol, rtol=rtol)
            if not comparison["passed"]:
                raise RuntimeError(f"missing-channel corruption was observable: {mask_tuple}")
            mask_results.append({
                "mask": [int(value) for value in mask_tuple],
                "finite": True,
                "source_available": True,
                "missing_feature_corruption_invariant": True,
                "radar_owned_feature_columns": {
                    str(index + 1): int(len(value)) for index, value in columns.items()
                },
                "comparison": comparison,
            })

        base = substream.forward_batch()
        zero_batch = replace(base, node_features=torch.zeros_like(base.node_features))
        zero_output = model(**zero_batch.model_kwargs())
        if not _all_finite(zero_output):
            raise RuntimeError("all-zero feature scenario produced non-finite output")

        no_candidate_batch = replace(
            base,
            candidate_mask=torch.zeros_like(base.candidate_mask),
            anchor_available=(
                None
                if base.anchor_available is None
                else torch.zeros_like(base.anchor_available)
            ),
        )
        no_candidate = model(**no_candidate_batch.model_kwargs())
        no_source = (
            not bool(no_candidate["source_available"].any())
            and bool((no_candidate["selected_index"] == -1).all())
            and bool((no_candidate["source_rr"] == 0).all())
        )
        if (
            substream.structural_fallback_rr is not None
            and substream.structural_fallback_available is not None
        ):
            fallback_value = torch.from_numpy(
                substream.structural_fallback_rr.copy()[None]
            ).to(device=no_candidate["source_rr"].device)
            fallback_available = torch.from_numpy(
                substream.structural_fallback_available.copy()[None]
            ).to(device=no_candidate["source_rr"].device)
            fallback_source = "locked_cache_index_bound_fallback"
        else:
            fallback_value = torch.full_like(no_candidate["source_rr"], 17.25)
            fallback_available = torch.ones_like(no_candidate["source_available"])
            fallback_source = "synthetic_router_sentinel_no_bound_fallback"
        routed, available, route = route_structural_fallback(
            no_candidate["source_rr"],
            no_candidate["source_available"],
            fallback_value,
            fallback_available,
        )
        structural_fallback = (
            no_source
            and torch.equal(available, fallback_available)
            and torch.equal(
                route,
                torch.where(
                    fallback_available,
                    torch.full_like(route, 2),
                    torch.zeros_like(route),
                ),
            )
            and torch.equal(
                routed,
                torch.where(
                    fallback_available,
                    fallback_value,
                    torch.zeros_like(fallback_value),
                ),
            )
        )
        if not structural_fallback:
            raise RuntimeError("no-candidate structural fallback routing failed")

        anchor_only_checked = base.anchor_available is not None
        anchor_only_available = False
        if anchor_only_checked:
            anchor_only_batch = replace(
                base, candidate_mask=torch.zeros_like(base.candidate_mask)
            )
            anchor_only = model(**anchor_only_batch.model_kwargs())
            assert anchor_only_batch.anchor_available is not None
            anchor_only_available = bool(
                torch.equal(
                    anchor_only["source_available"],
                    anchor_only_batch.anchor_available,
                )
                and _all_finite(anchor_only)
            )
            if not anchor_only_available:
                raise RuntimeError("anchor-only structural source route failed")

        corrupt_nodes = torch.full_like(base.node_features, float("nan"))
        corrupt_nodes[..., 0] = float("inf")
        if corrupt_nodes.shape[-1] > 1:
            corrupt_nodes[..., 1] = float("-inf")
        corrupt_rr = torch.full_like(base.candidate_rr, float("nan"))
        corrupt_rr[..., 0] = float("inf")
        corrupt_batch = replace(
            base,
            node_features=corrupt_nodes,
            candidate_rr=corrupt_rr,
            anchor_available=(
                None
                if base.anchor_available is None
                else torch.zeros_like(base.anchor_available)
            ),
        )
        corrupt_output = model(**corrupt_batch.model_kwargs())
        corrupt_safe = (
            _all_finite(corrupt_output)
            and not bool(corrupt_output["source_available"].any())
            and bool((corrupt_output["selected_index"] == -1).all())
        )
        if not corrupt_safe:
            raise RuntimeError("corrupt input sanitization/availability contract failed")

        corrupt_anchor_rejected = False
        if base.anchor_rr is not None and base.anchor_available is not None:
            bad_anchor_rr = base.anchor_rr.clone()
            bad_anchor_available = base.anchor_available.clone()
            bad_anchor_rr[:, 0] = float("nan")
            bad_anchor_available[:, 0] = True
            try:
                model(**replace(
                    base,
                    anchor_rr=bad_anchor_rr,
                    anchor_available=bad_anchor_available,
                ).model_kwargs())
            except ValueError:
                corrupt_anchor_rejected = True
            if not corrupt_anchor_rejected:
                raise RuntimeError("non-finite available anchor was not rejected")

        no_radar_batch = replace(
            base,
            radar_mask=torch.zeros_like(base.radar_mask),
            anchor_available=(
                None
                if base.anchor_available is None
                else torch.zeros_like(base.anchor_available)
            ),
        )
        no_radar = model(**no_radar_batch.model_kwargs())
        missing_all_safe = (
            _all_finite(no_radar)
            and not bool(no_radar["source_available"].any())
            and bool((no_radar["selected_index"] == -1).all())
        )
        if not missing_all_safe:
            raise RuntimeError("all-radar-missing availability contract failed")

        wrong_shape_rejected = False
        try:
            model(**replace(
                base, node_features=base.node_features[..., :-1]
            ).model_kwargs())
        except ValueError:
            wrong_shape_rejected = True
        if not wrong_shape_rejected:
            raise RuntimeError("corrupt feature shape was not rejected")

    return {
        "passed": True,
        "windows": int(substream.windows),
        "seven_nonempty_radar_masks": mask_results,
        "zero_features_finite": True,
        "no_candidate_source_unavailable": bool(no_source),
        "no_candidate_structural_fallback_route": bool(structural_fallback),
        "structural_fallback_source": fallback_source,
        "structural_fallback_available_fraction": float(
            fallback_available.float().mean().item()
        ),
        "anchor_only_route_checked": bool(anchor_only_checked),
        "anchor_only_route_matches_availability": bool(anchor_only_available),
        "corrupt_nan_inf_inputs_finite_and_unavailable": bool(corrupt_safe),
        "corrupt_available_anchor_rejected": bool(corrupt_anchor_rejected),
        "all_radars_missing_finite_and_unavailable": bool(missing_all_safe),
        "wrong_feature_shape_rejected": bool(wrong_shape_rejected),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.  This is a process-wide high-water mark,
    # so the report labels it as non-isolated rather than pretending it is a
    # tensor-allocation measurement.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError("latency samples are empty or non-finite")
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "minimum_ms": float(values.min()),
        "maximum_ms": float(values.max()),
        "mean_ms": float(values.mean()),
    }


def benchmark_device(
    checkpoint_path: Path,
    model_config: Mapping[str, Any],
    batch: ForwardBatch,
    device: torch.device,
    *,
    warmup_repeats: int,
    repeats: int,
    cold_repeats: int,
    chunk_windows: int,
) -> list[dict[str, Any]]:
    if min(warmup_repeats, repeats, cold_repeats, chunk_windows) < 1:
        raise ValueError("benchmark repeat/window counts must be positive")
    one = batch.time_slice(0, 1).to(device)
    cold_samples: list[float] = []
    for _ in range(cold_repeats):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter_ns()
        cold_model = load_model(checkpoint_path, model_config, device)
        with torch.inference_mode():
            cold_model(**one.model_kwargs())
        _synchronize(device)
        cold_samples.append((time.perf_counter_ns() - started) / 1.0e6)
        del cold_model
    cold_peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else _cpu_peak_rss_bytes()
    )

    model = load_model(checkpoint_path, model_config, device)
    with torch.inference_mode():
        for _ in range(warmup_repeats):
            model(**one.model_kwargs())
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    warm_samples: list[float] = []
    state: HarmonicSetState | None = None
    for repeat in range(repeats):
        index = repeat % batch.windows
        window = batch.time_slice(index, index + 1).to(device)
        if index == 0 or bool(window.reset_mask[0, 0]):
            state = None
        started = time.perf_counter_ns()
        with torch.inference_mode():
            output = model(**window.model_kwargs(), state=state)
        _synchronize(device)
        warm_samples.append((time.perf_counter_ns() - started) / 1.0e6)
        state = _detached_state(output["state"])  # type: ignore[arg-type]
    warm_peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else _cpu_peak_rss_bytes()
    )

    chunk_length = min(int(chunk_windows), batch.windows)
    chunk = batch.time_slice(0, chunk_length).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    chunk_samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        with torch.inference_mode():
            model(**chunk.model_kwargs(), state=None)
        _synchronize(device)
        chunk_samples.append((time.perf_counter_ns() - started) / 1.0e6)
    chunk_peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else _cpu_peak_rss_bytes()
    )

    memory_method = (
        "torch.cuda.max_memory_allocated"
        if device.type == "cuda"
        else "process_ru_maxrss_nonisolated"
    )
    rows = [
        {
            "device": str(device),
            "operation": "checkpoint_load_model_init_plus_first_window_cold",
            "windows_per_request": 1,
            "repeats": int(cold_repeats),
            **_latency_summary(cold_samples),
            "throughput_windows_per_second": 1000.0 / _latency_summary(cold_samples)["p50_ms"],
            "peak_memory_bytes": cold_peak,
            "peak_memory_method": memory_method,
        },
        {
            "device": str(device),
            "operation": "stateful_one_window_warm",
            "windows_per_request": 1,
            "repeats": int(repeats),
            **_latency_summary(warm_samples),
            "throughput_windows_per_second": 1000.0 / _latency_summary(warm_samples)["p50_ms"],
            "peak_memory_bytes": warm_peak,
            "peak_memory_method": memory_method,
        },
        {
            "device": str(device),
            "operation": "stateless_chunk_warm",
            "windows_per_request": chunk_length,
            "repeats": int(repeats),
            **_latency_summary(chunk_samples),
            "throughput_windows_per_second": (
                1000.0 * chunk_length / _latency_summary(chunk_samples)["p50_ms"]
            ),
            "peak_memory_bytes": chunk_peak,
            "peak_memory_method": memory_method,
        },
    ]
    if device.type == "cuda":
        for row in rows:
            row["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return rows


def _parse_schedules(value: str) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []
    for raw_schedule in value.split(";"):
        raw_schedule = raw_schedule.strip()
        if not raw_schedule:
            continue
        schedule = tuple(int(item.strip()) for item in raw_schedule.split(","))
        if not schedule or any(item < 1 for item in schedule):
            raise argparse.ArgumentTypeError("chunk schedules must contain positive integers")
        result.append(schedule)
    if not result:
        raise argparse.ArgumentTypeError("at least one chunk schedule is required")
    return result


def _device_names(value: str) -> list[str]:
    raw = value.strip().lower()
    if raw == "auto":
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names or any(name not in {"cpu", "cuda"} for name in names):
        raise ValueError("devices must be auto or a comma-separated subset of cpu,cuda")
    if "cuda" in names and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    return list(dict.fromkeys(names))


def _report_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report["parity"]["cases"]:
        rows.append({
            "category": "parity",
            "name": case.get("mode", "chunk_schedule"),
            "device": report["parity_device"],
            "status": "pass" if case["passed"] else "fail",
            "p50_ms": "", "p95_ms": "", "p99_ms": "",
            "throughput_windows_per_second": "",
            "detail": json.dumps(case["schedule"], separators=(",", ":")),
        })
    for result in report["robustness"]["seven_nonempty_radar_masks"]:
        rows.append({
            "category": "radar_mask",
            "name": "".join(map(str, result["mask"])),
            "device": report["parity_device"],
            "status": "pass",
            "p50_ms": "", "p95_ms": "", "p99_ms": "",
            "throughput_windows_per_second": "",
            "detail": "finite, available, missing-channel corruption invariant",
        })
    for latency in report["benchmarks"]:
        rows.append({
            "category": "latency",
            "name": latency["operation"],
            "device": latency["device"],
            "status": "measured",
            "p50_ms": latency["p50_ms"],
            "p95_ms": latency["p95_ms"],
            "p99_ms": latency["p99_ms"],
            "throughput_windows_per_second": latency["throughput_windows_per_second"],
            "detail": f"peak_memory_bytes={latency['peak_memory_bytes']}",
        })
    return rows


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Harmonic candidate-set SNN deployment verification",
        "",
        "> **RETROSPECTIVE ENGINEERING VERIFICATION — NOT A COMMERCIAL PERFORMANCE CLAIM.**",
        "",
        f"Overall status: **{str(report['status']).upper()}**",
        "",
        (
            "No reference target, reference-validity field, identity/fold field, "
            "or test prediction artifact was accessed."
        ),
        "",
        "## Scope",
        "",
        f"- Sessions: {', '.join(report['input_scope']['session_ids'])}",
        f"- Windows: {report['input_scope']['windows']}",
        f"- Session reset boundaries: {report['input_scope']['session_lengths']}",
        f"- Parity device: {report['parity_device']}",
        "",
        "## Verification gates",
        "",
        "- Lock/source/config/cache hashes: PASS",
        f"- Whole/chunk/one-window parity: "
        f"{'PASS' if report['parity']['passed'] else 'FAIL'}",
        f"- Explicit reset/session isolation: "
        f"{'PASS' if report['session_reset']['passed'] else 'FAIL'}",
        f"- Seven nonempty radar masks: "
        f"{'PASS' if report['robustness']['passed'] else 'FAIL'}",
        f"- No-candidate structural fallback: "
        f"{'PASS' if report['robustness']['no_candidate_structural_fallback_route'] else 'FAIL'}",
        f"- Corrupt input handling: "
        f"{'PASS' if report['robustness']['corrupt_nan_inf_inputs_finite_and_unavailable'] else 'FAIL'}",
        "",
        "## Latency",
        "",
        "| Device | Operation | p50 ms | p95 ms | p99 ms | Windows/s | Peak memory bytes |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["benchmarks"]:
        lines.append(
            f"| {row['device']} | {row['operation']} | {row['p50_ms']:.4f} | "
            f"{row['p95_ms']:.4f} | {row['p99_ms']:.4f} | "
            f"{row['throughput_windows_per_second']:.2f} | {row['peak_memory_bytes']} |"
        )
    lines.extend([
        "",
        (
            "CPU peak memory is a non-isolated process high-water mark; CUDA peak "
            "memory uses PyTorch's allocator counter. Cold latency includes checkpoint "
            "load, model construction, and first inference but does not flush the "
            "operating-system page cache."
        ),
        (
            "Radar-mask checks operate at the locked cache/model boundary; raw "
            "RF/SVD/proposer regeneration is a separate end-to-end gate."
        ),
        "",
        "## Release limitation",
        "",
        (
            "This report verifies artifact integrity and execution behavior only. "
            "Independent prospective-cohort accuracy, calibration, safety, and "
            "operational monitoring are still required before any commercial claim."
        ),
        "",
    ])
    return "\n".join(lines)


def _write_reports(output_dir: Path, report: Mapping[str, Any]) -> dict[str, str]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError("immutable report output already exists; choose a new output directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        json_path = stage / "deployment_verification.json"
        csv_path = stage / "deployment_verification.csv"
        markdown_path = stage / "deployment_verification.md"
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        rows = _report_rows(report)
        fieldnames = (
            "category", "name", "device", "status", "p50_ms", "p95_ms",
            "p99_ms", "throughput_windows_per_second", "detail",
        )
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        hashes = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (json_path, csv_path, markdown_path)
        }
        hashes_path = stage / "artifact_hashes.json"
        hashes_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "immutable": True,
                    "retrospective_only": True,
                    "commercial_claim_authorized": False,
                    "artifacts": hashes,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        for path in (*[json_path, csv_path, markdown_path], hashes_path):
            path.chmod(0o444)
        os.replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        name: sha256_file(output_dir / name)
        for name in (
            "deployment_verification.json",
            "deployment_verification.csv",
            "deployment_verification.md",
            "artifact_hashes.json",
        )
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    cache_root = args.cache.expanduser().resolve()
    checkpoint = (
        run_dir / "best_checkpoint.pt"
        if args.checkpoint is None
        else args.checkpoint.expanduser().resolve()
    )
    bindings = validate_locked_artifacts(run_dir, cache_root, checkpoint)
    anchor_binding = bindings.get("anchor_input")
    stream = load_deployment_stream(
        cache_root,
        run_dir / "scaler.json",
        requested_sessions=args.session_id,
        maximum_sessions=int(args.maximum_sessions),
        maximum_windows_per_session=int(args.maximum_windows_per_session),
        anchor_input_path=(
            Path(str(anchor_binding["path"]))
            if isinstance(anchor_binding, Mapping)
            else None
        ),
        anchor_forward_enabled=bool(bindings.get("anchor_enabled")),
    )
    devices = _device_names(args.devices)
    parity_device = torch.device(args.parity_device)
    if parity_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA parity requested but CUDA is unavailable")
    model = load_model(checkpoint, bindings["model_config"], parity_device)
    batch = stream.forward_batch().to(parity_device)

    schedules = _parse_schedules(args.chunk_schedules)
    rng = random.Random(int(args.schedule_seed))
    for _ in range(int(args.random_schedules)):
        schedules.append(tuple(rng.randint(1, int(args.random_chunk_max)) for _ in range(7)))
    parity = parity_verification(
        model, batch, schedules, atol=float(args.atol), rtol=float(args.rtol)
    )
    reset = session_reset_verification(
        model,
        batch,
        stream.session_lengths,
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    robustness = robustness_verification(
        model,
        stream,
        windows=int(args.robustness_windows),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    del model

    benchmarks: list[dict[str, Any]] = []
    cpu_batch = stream.forward_batch()
    for device_name in devices:
        benchmarks.extend(benchmark_device(
            checkpoint,
            bindings["model_config"],
            cpu_batch,
            torch.device(device_name),
            warmup_repeats=int(args.warmup_repeats),
            repeats=int(args.benchmark_repeats),
            cold_repeats=int(args.cold_repeats),
            chunk_windows=int(args.benchmark_chunk_windows),
        ))

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "classification": "retrospective_deployment_engineering_verification",
        "retrospective_only": True,
        "commercial_claim_authorized": False,
        "prospective_cohort_required_for_commercial_claim": True,
        "label_access": {
            "target_or_test_labels_accessed": False,
            "metadata_columns_accessed": list(DEPLOYMENT_METADATA_COLUMNS),
            "fallback_or_anchor_columns_accessed": (
                list(anchor_binding["label_free_columns_only"])
                if isinstance(anchor_binding, Mapping)
                else []
            ),
            "test_prediction_artifacts_opened": False,
            "forward_allowlist_only": True,
        },
        "bindings": bindings,
        "input_scope": {
            "session_ids": list(stream.session_ids),
            "session_lengths": list(stream.session_lengths),
            "windows": stream.windows,
            "explicit_reset_count": int(stream.reset_mask.sum()),
            "cache_indices_sha256": hashlib.sha256(
                np.ascontiguousarray(stream.cache_indices, dtype=np.int64).view(np.uint8)
            ).hexdigest(),
            "truncated_by_user_limit": bool(stream.truncated),
        },
        "parity_device": str(parity_device),
        "parity": parity,
        "session_reset": reset,
        "robustness": robustness,
        "benchmarks": benchmarks,
        "limitations": [
            "No target/reference/test label was accessed, so this report contains no accuracy claim.",
            "Passing retrospective parity and robustness checks is not prospective clinical or commercial validation.",
            "CPU peak memory is a process-wide non-isolated high-water mark.",
            (
                "Radar-mask checks zero radar-owned cached feature columns and exercise "
                "the locked model; they do not regenerate RF/SVD/proposer caches from "
                "raw radar recordings."
            ),
            (
                "Cold latency includes checkpoint load/model construction/first "
                "inference in the same process and does not flush the operating-system "
                "page cache."
            ),
        ],
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    hashes = _write_reports(args.output_dir, report)
    return {"status": "passed", "output_dir": str(args.output_dir.resolve()), "hashes": hashes, "report": report}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument("--maximum-sessions", type=int, default=2)
    parser.add_argument(
        "--maximum-windows-per-session", type=int, default=0,
        help="0 verifies complete sessions; positive values are explicit smoke-test truncation",
    )
    parser.add_argument("--parity-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--chunk-schedules", default="7,3,11,2;13,1,5,2,8")
    parser.add_argument("--random-schedules", type=int, default=2)
    parser.add_argument("--random-chunk-max", type=int, default=23)
    parser.add_argument("--schedule-seed", type=int, default=20260828)
    parser.add_argument("--atol", type=float, default=2.0e-6)
    parser.add_argument("--rtol", type=float, default=2.0e-6)
    parser.add_argument("--robustness-windows", type=int, default=8)
    parser.add_argument(
        "--devices", default="auto",
        help="auto, cpu, cuda, or cpu,cuda",
    )
    parser.add_argument("--warmup-repeats", type=int, default=5)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    parser.add_argument("--cold-repeats", type=int, default=3)
    parser.add_argument("--benchmark-chunk-windows", type=int, default=32)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.maximum_sessions < 1 or args.maximum_windows_per_session < 0:
        raise SystemExit("session limits are invalid")
    if args.random_schedules < 0 or args.random_chunk_max < 1:
        raise SystemExit("random schedule settings are invalid")
    if args.atol < 0 or args.rtol < 0 or args.robustness_windows < 1:
        raise SystemExit("parity/robustness settings are invalid")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = verify(parse_args(argv))
    print(json.dumps({key: value for key, value in result.items() if key != "report"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
