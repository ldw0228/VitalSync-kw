#!/usr/bin/env python3
"""Benchmark raw-window-to-RR latency for a trained checkpoint.

The primary path starts with one 32-second, three-radar 40 Hz window already
resident in host memory.  It times the window-local causal outlier
repair/downsampling, range-frequency feature generation, auxiliary/scaler and
tensor construction, host-to-device transfer, and model forward pass.  When
raw XeThru files are available, a second warm-page-cache path also includes the
memmap slice/copy needed to obtain that same window.

Examples
--------
Short CPU/GPU benchmark of a structured checkpoint::

    python scripts/benchmark_e2e.py \
      --checkpoint artifacts/runs/final_structured_aux_s12/fold_0/snn_best.pt \
      --devices all --warmup 5 --repeats 30

Fast smoke check::

    python scripts/benchmark_e2e.py --devices cpu --warmup 1 --repeats 2 \
      --input-source synthetic
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from snn_rr.cache import (  # noqa: E402
    append_causal_history_features,
    load_feature_cache,
    transform_aux,
)
from snn_rr.data import (  # noqa: E402
    DatasetManifest,
    SplitRadarMemmap,
    build_dataset_manifest,
    open_xethru_files,
)
from snn_rr.preprocess import (  # noqa: E402
    causal_block_mean,
    classical_rr_estimate,
    fuse_auxiliary_features,
    range_frequency_features,
)
from scripts.build_features import replace_radar_outliers  # noqa: E402


def _load_train_module():
    path = PROJECT_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("snn_rr_train_for_e2e", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = _load_train_module()


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    radar_map: np.ndarray
    base_aux: np.ndarray
    pooled_frequencies_hz: np.ndarray
    classical_rr_bpm: float
    classical_confidence: float
    classical_spread_bpm: float
    outlier_count: int


@dataclass(frozen=True, slots=True)
class PreparedCheckpoint:
    path: Path
    checkpoint: Mapping[str, Any]
    run_config: Mapping[str, Any]
    model_type: str
    model_kwargs: Mapping[str, Any]
    aux_center: np.ndarray
    aux_scale: np.ndarray
    map_branch: str
    expected_aux_dim: int


@dataclass(frozen=True, slots=True)
class RawFileSource:
    subject_id: str
    streams: tuple[SplitRadarMemmap, SplitRadarMemmap, SplitRadarMemmap]
    start_frame: int
    frame_count: int
    data_paths: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]

    def read(self) -> np.ndarray:
        stop = self.start_frame + self.frame_count
        return np.stack(
            [
                np.asarray(stream[self.start_frame:stop]["bins"], dtype=np.float32)
                for stream in self.streams
            ],
            axis=0,
        )


def load_yaml_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("data"), dict):
        raise ValueError(f"invalid pipeline configuration: {path}")
    return value


def find_run_config(checkpoint_path: Path) -> dict[str, Any]:
    for parent in checkpoint_path.resolve().parents:
        candidate = parent / "run_config.json"
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
    return {}


def prepare_checkpoint(path: Path) -> PreparedCheckpoint:
    checkpoint_path = path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "model_type",
        "model_kwargs",
        "model_state",
        "aux_center",
        "aux_scale",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise KeyError(f"checkpoint is missing fields {missing}: {checkpoint_path}")
    kwargs = dict(checkpoint["model_kwargs"])
    center = np.asarray(checkpoint["aux_center"].cpu(), dtype=np.float32)
    scale = np.asarray(checkpoint["aux_scale"].cpu(), dtype=np.float32)
    expected_aux_dim = int(kwargs.get("aux_dim", len(center)))
    if center.shape != (expected_aux_dim,) or scale.shape != (expected_aux_dim,):
        raise ValueError("checkpoint auxiliary scaler/model dimensions differ")
    run_config = find_run_config(checkpoint_path)
    if run_config:
        checkpoint_signature = checkpoint.get("run_signature")
        config_signature = run_config.get("run_signature")
        if not checkpoint_signature or not config_signature:
            raise RuntimeError(
                "checkpoint/run_config provenance is incomplete: both must contain "
                f"run_signature ({checkpoint_path})"
            )
        if str(checkpoint_signature) != str(config_signature):
            raise RuntimeError(
                "checkpoint/run_config run_signature mismatch: "
                f"{checkpoint_signature!r} != {config_signature!r}"
            )
    arguments = run_config.get("arguments", {})
    map_branch = str(arguments.get("map_branch", "both"))
    if map_branch not in {"both", "raw", "phase"}:
        raise ValueError(f"unsupported checkpoint map branch: {map_branch}")
    return PreparedCheckpoint(
        path=checkpoint_path,
        checkpoint=checkpoint,
        run_config=run_config,
        model_type=str(checkpoint["model_type"]),
        model_kwargs=kwargs,
        aux_center=center,
        aux_scale=scale,
        map_branch=map_branch,
        expected_aux_dim=expected_aux_dim,
    )


def validate_pipeline_config_provenance(
    prepared: PreparedCheckpoint,
    supplied_config_path: Path,
) -> dict[str, Any]:
    """Bind the benchmark preprocessing config to the training run when possible.

    New and current project runs declare ``arguments.config`` in run_config.json.
    Truly legacy checkpoints without a run config remain loadable, but the report
    marks their preprocessing provenance as unverified rather than implying an
    exact training/benchmark match.
    """

    supplied = supplied_config_path.resolve()
    supplied_sha256 = sha256_file(supplied)
    if not prepared.run_config:
        return {
            "verified": False,
            "status": "unverified_legacy_checkpoint_without_run_config",
            "supplied_path": str(supplied),
            "supplied_sha256": supplied_sha256,
            "training_path": None,
            "training_sha256": None,
        }
    arguments = prepared.run_config.get("arguments", {})
    declared = arguments.get("config")
    declared_sha256 = prepared.run_config.get("config_sha256")
    training_path: Path | None = None
    if declared:
        training_path = Path(str(declared)).expanduser()
        if not training_path.is_absolute():
            training_path = PROJECT_ROOT / training_path
        training_path = training_path.resolve()
        if not training_path.is_file():
            raise RuntimeError(
                f"training run declares a missing preprocessing config: {training_path}"
            )
        training_sha256 = sha256_file(training_path)
        if declared_sha256 is not None and str(declared_sha256) != training_sha256:
            raise RuntimeError("run_config config_sha256 does not match its config file")
    elif declared_sha256 is not None:
        training_sha256 = str(declared_sha256)
    else:
        raise RuntimeError(
            "run_config does not bind the checkpoint to a preprocessing config"
        )
    if supplied_sha256 != training_sha256:
        raise RuntimeError(
            "benchmark preprocessing config differs from the checkpoint training config"
        )
    return {
        "verified": True,
        "status": "sha256_match",
        "supplied_path": str(supplied),
        "supplied_sha256": supplied_sha256,
        "training_path": str(training_path) if training_path is not None else None,
        "training_sha256": training_sha256,
    }


def build_model(prepared: PreparedCheckpoint, device: torch.device) -> torch.nn.Module:
    # Using serialized kwargs rather than preset names makes structured aux,
    # harmonic heads/projections, and later checkpoint-compatible options
    # automatic as long as the current model implementation supports them.
    model = TRAIN.build_model(prepared.model_type, prepared.model_kwargs)
    model.load_state_dict(prepared.checkpoint["model_state"])
    return model.to(device).eval()


def pipeline_option_summary(prepared: PreparedCheckpoint) -> dict[str, Any]:
    option_keys = {
        key: value
        for key, value in prepared.model_kwargs.items()
        if "structured" in key.lower() or "harmonic" in key.lower()
    }
    return {
        "model_type": prepared.model_type,
        "map_branch": prepared.map_branch,
        "input_branches": int(prepared.model_kwargs.get("input_branches", 1)),
        "aux_dim": prepared.expected_aux_dim,
        "serialized_special_options": option_keys,
        "model_kwargs_loaded_verbatim": True,
    }


def synthetic_raw_window(
    *,
    frames: int,
    bins: int = 182,
    seed: int = 20260828,
) -> np.ndarray:
    """Deterministic radar-like signal for environments without raw files."""

    rng = np.random.default_rng(seed)
    time_axis = np.arange(frames, dtype=np.float32) / 40.0
    ranges = np.linspace(0.0, 2.0 * np.pi, bins, dtype=np.float32)
    views: list[np.ndarray] = []
    for radar in range(3):
        phase = 0.6 * radar
        respiration = np.sin(2 * np.pi * 0.27 * time_axis[:, None] + ranges[None, :] + phase)
        harmonic = 0.25 * np.sin(
            2 * np.pi * 0.54 * time_axis[:, None] - 0.5 * ranges[None, :] + phase
        )
        noise = rng.normal(0.0, 0.0015, size=(frames, bins))
        views.append((0.004 * respiration + 0.001 * harmonic + noise).astype(np.float32))
    return np.stack(views)


def preprocess_raw_window(raw_window: np.ndarray, data_config: Mapping[str, Any]) -> FeatureBundle:
    """Run the cache builder's causal radar feature path for one window."""

    raw = np.asarray(raw_window, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != 3 or raw.shape[2] != 182:
        raise ValueError("raw_window must have shape [3, frames, 182]")
    radar_hz = float(data_config["radar_hz"])
    model_hz = float(data_config["model_hz"])
    downsample = int(round(radar_hz / model_hz))
    if downsample < 1 or not np.isclose(radar_hz / model_hz, downsample):
        raise ValueError("radar_hz/model_hz must be a positive integer")
    expected_frames = int(round(float(data_config["window_seconds"]) * radar_hz))
    if raw.shape[1] != expected_frames:
        raise ValueError(
            f"raw window has {raw.shape[1]} frames; expected {expected_frames}"
        )
    repaired: list[np.ndarray] = []
    outlier_count = 0
    for radar in raw:
        values, count = replace_radar_outliers(radar)
        repaired.append(causal_block_mean(values, downsample))
        outlier_count += count
    radar_features = [
        range_frequency_features(
            radar,
            fs=model_hz,
            band_hz=tuple(map(float, data_config["respiration_band_hz"])),
            nfft=int(data_config["fft_size"]),
            range_pool=int(data_config["range_pool"]),
        )
        for radar in repaired
    ]
    raw_maps = np.stack([item.feature_map for item in radar_features])
    usable_frequencies = raw_maps.shape[1] - raw_maps.shape[1] % 2
    radar_map = raw_maps[:, :usable_frequencies].reshape(
        3, usable_frequencies // 2, 2, raw_maps.shape[-1]
    ).mean(axis=2, dtype=np.float32).astype(np.float16)
    full_grid = radar_features[0].frequencies_hz[:usable_frequencies]
    pooled_grid = full_grid.reshape(-1, 2).mean(axis=1).astype(np.float32)
    base_aux = fuse_auxiliary_features(radar_features)
    classical = classical_rr_estimate(
        radar_features,
        rr_range_bpm=tuple(map(float, data_config["rr_range_bpm"])),
    )
    return FeatureBundle(
        radar_map=radar_map,
        base_aux=base_aux,
        pooled_frequencies_hz=pooled_grid,
        classical_rr_bpm=classical.rr_bpm,
        classical_confidence=classical.confidence,
        classical_spread_bpm=classical.consensus_spread_bpm,
        outlier_count=outlier_count,
    )


def history_tail_from_cache(
    cache_dir: Path,
    *,
    subject_id: str | None,
    expected_tail_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one already-available strictly causal state vector for timing.

    Past-window history is deployment state and is therefore resident before
    the current raw window arrives.  It is not regenerated from pandas inside
    every latency trial.  The timed input stage still includes concatenation
    and scaling of this state.
    """

    if expected_tail_dim <= 0:
        return np.empty(0, dtype=np.float32), {
            "source": "none",
            "dimension": 0,
        }
    try:
        sessions = [subject_id] if subject_id else None
        cache = load_feature_cache(cache_dir, sessions=sessions)
        augmented, names = append_causal_history_features(cache.aux, cache.metadata)
        if augmented.shape[1] - cache.aux.shape[1] == expected_tail_dim:
            candidates = np.flatnonzero(
                cache.metadata["window_number"].to_numpy(dtype=int) >= 8
            )
            row = int(candidates[0]) if len(candidates) else len(cache.metadata) - 1
            tail = np.asarray(augmented[row, -expected_tail_dim:], dtype=np.float32)
            return tail, {
                "source": "cache_strictly_causal_state",
                "dimension": expected_tail_dim,
                "session_id": str(cache.metadata.iloc[row]["session_id"]),
                "window_number": int(cache.metadata.iloc[row]["window_number"]),
                "feature_names": names,
            }
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return np.zeros(expected_tail_dim, dtype=np.float32), {
        "source": "zero_initialized_state",
        "dimension": expected_tail_dim,
        "warning": (
            "checkpoint tail does not match the current causal-history cache; "
            "zero state was used for latency only"
        ),
    }


def select_map_branch(radar_map: np.ndarray, map_branch: str) -> np.ndarray:
    value = np.asarray(radar_map)
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError("radar_map must have shape [3, frequency, range]")
    if map_branch == "both":
        return value
    if value.shape[-1] % 2:
        raise ValueError("raw/phase selection requires an even range dimension")
    half = value.shape[-1] // 2
    if map_branch == "raw":
        return value[..., :half]
    if map_branch == "phase":
        return value[..., half:]
    raise ValueError(f"unsupported map branch: {map_branch}")


def construct_numpy_model_inputs(
    feature: FeatureBundle,
    prepared: PreparedCheckpoint,
    history_tail: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply branch selection, auxiliary layout, and checkpoint scaler."""

    radar_map = select_map_branch(feature.radar_map, prepared.map_branch)
    expected_branches = int(prepared.model_kwargs.get("input_branches", 1))
    if radar_map.shape[-1] % expected_branches:
        raise ValueError("map range dimension is incompatible with checkpoint branches")
    if prepared.expected_aux_dim == 0:
        aux = np.empty(0, dtype=np.float32)
    else:
        base_aux_dim = int(
            prepared.model_kwargs.get("aux_base_dim") or len(feature.base_aux)
        )
        if base_aux_dim != len(feature.base_aux):
            raise ValueError(
                f"generated base aux has {len(feature.base_aux)} columns, "
                f"checkpoint declares {base_aux_dim}"
            )
        required_tail = prepared.expected_aux_dim - base_aux_dim
        tail = np.asarray(history_tail, dtype=np.float32)
        if required_tail < 0 or tail.shape != (required_tail,):
            raise ValueError(
                f"checkpoint requires auxiliary tail {required_tail}, got {tail.shape}"
            )
        unscaled = np.concatenate([feature.base_aux.astype(np.float32), tail])
        aux = transform_aux(
            unscaled,
            prepared.aux_center,
            prepared.aux_scale,
        )
    radar_mask = np.ones(3, dtype=bool)
    return radar_map, aux, radar_mask


def tensors_to_device(
    radar_map: np.ndarray,
    aux: np.ndarray,
    radar_mask: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    map_tensor = torch.from_numpy(np.ascontiguousarray(radar_map)).unsqueeze(0)
    aux_tensor = torch.from_numpy(np.ascontiguousarray(aux)).unsqueeze(0)
    mask_tensor = torch.from_numpy(np.ascontiguousarray(radar_mask)).unsqueeze(0)
    return (
        map_tensor.to(device=device).float(),
        aux_tensor.to(device=device, dtype=torch.float32),
        mask_tensor.to(device=device),
    )


def resolve_raw_file_source(
    manifest: DatasetManifest,
    *,
    subject_id: str | None,
    frame_count: int,
    start_frame: int | None,
    frame_alignment: int = 1,
) -> RawFileSource:
    if frame_alignment < 1:
        raise ValueError("frame_alignment must be positive")
    subject = (
        manifest.by_subject(subject_id)
        if subject_id
        else manifest.usable_subjects[0]
    )
    if subject.selected_session is None:
        raise ValueError(f"subject has no complete radar session: {subject.subject_id}")
    streams = tuple(
        open_xethru_files(subject.selected_session.radars[radar].data_paths)
        for radar in (1, 2, 3)
    )
    common = min(len(stream) for stream in streams)
    if common < frame_count:
        raise ValueError(
            f"subject {subject.subject_id} has only {common} common frames"
        )
    selected_start = (
        int(start_frame)
        if start_frame is not None
        else max(0, (common - frame_count) // 2)
    )
    if start_frame is None:
        # The cache builder downsamples the recording from frame zero.  Keep
        # the benchmark window on that identical block boundary.
        selected_start -= selected_start % frame_alignment
    elif selected_start % frame_alignment:
        raise ValueError(
            f"start frame must be aligned to {frame_alignment} raw frames"
        )
    if selected_start < 0 or selected_start + frame_count > common:
        raise ValueError("requested raw file window is out of bounds")
    paths = tuple(
        tuple(str(path) for path in subject.selected_session.radars[radar].data_paths)
        for radar in (1, 2, 3)
    )
    return RawFileSource(
        subject_id=subject.subject_id,
        streams=streams,  # type: ignore[arg-type]
        start_frame=selected_start,
        frame_count=frame_count,
        data_paths=paths,  # type: ignore[arg-type]
    )


def summarize_latency(samples_ms: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(samples_ms, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("latency samples must be a finite non-empty vector")
    return {
        "repeats": int(len(values)),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "quantile_method": "linear",
        "samples_ms": values.tolist(),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def run_latency_trials(
    *,
    model: torch.nn.Module,
    prepared: PreparedCheckpoint,
    data_config: Mapping[str, Any],
    history_tail: np.ndarray,
    device: torch.device,
    raw_supplier: Callable[[], np.ndarray],
    include_raw_load: bool,
    repeats: int,
    warmup: int,
    amp: bool,
) -> dict[str, Any]:
    """Measure serialized batch-one latency with explicit CUDA barriers."""

    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    model.eval()

    def iteration(record: bool) -> tuple[dict[str, float], float]:
        _synchronize(device)
        if include_raw_load:
            total_start = time.perf_counter_ns()
            load_start = total_start
            raw = raw_supplier()
            load_end = time.perf_counter_ns()
        else:
            raw = raw_supplier()
            total_start = time.perf_counter_ns()
            load_start = load_end = total_start
        feature = preprocess_raw_window(raw, data_config)
        feature_end = time.perf_counter_ns()
        radar_map, aux, radar_mask = construct_numpy_model_inputs(
            feature, prepared, history_tail
        )
        input_end = time.perf_counter_ns()
        map_tensor, aux_tensor, mask_tensor = tensors_to_device(
            radar_map, aux, radar_mask, device
        )
        _synchronize(device)
        transfer_end = time.perf_counter_ns()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            output = model(map_tensor, radar_mask=mask_tensor, aux=aux_tensor)
        _synchronize(device)
        forward_end = time.perf_counter_ns()
        # Retain a scalar without adding a device-to-host copy to the measured
        # path; synchronization above guarantees the forward is complete.
        expected_rr = float(output["expected_rr"].detach().cpu()[0]) if record else 0.0
        factor = 1e-6
        stages = {
            "raw_load_ms": (load_end - load_start) * factor if include_raw_load else 0.0,
            "feature_generation_ms": (feature_end - load_end) * factor,
            "input_construction_ms": (input_end - feature_end) * factor,
            "tensor_and_transfer_ms": (transfer_end - input_end) * factor,
            "forward_ms": (forward_end - transfer_end) * factor,
            "total_ms": (forward_end - total_start) * factor,
        }
        return stages, expected_rr

    for _ in range(warmup):
        iteration(False)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    samples: dict[str, list[float]] = {
        name: []
        for name in (
            "raw_load_ms",
            "feature_generation_ms",
            "input_construction_ms",
            "tensor_and_transfer_ms",
            "forward_ms",
            "total_ms",
        )
    }
    last_rr = float("nan")
    try:
        for _ in range(repeats):
            stages, last_rr = iteration(True)
            for name, value in stages.items():
                samples[name].append(value)
    finally:
        if gc_was_enabled:
            gc.enable()
    summary = {name: summarize_latency(values) for name, values in samples.items()}
    total_p50 = summary["total_ms"]["p50_ms"]
    total_p95 = summary["total_ms"]["p95_ms"]
    result: dict[str, Any] = {
        "batch_size": 1,
        "warmup": warmup,
        "repeats": repeats,
        "amp": bool(amp and device.type == "cuda"),
        "raw_load_included": include_raw_load,
        "stages": summary,
        "throughput_from_p50_windows_per_second": 1000.0 / total_p50,
        "p95_fraction_of_4s_stride_budget": total_p95 / 4000.0,
        "last_expected_rr_bpm": last_rr,
    }
    if device.type == "cuda":
        result["peak_allocated_mb"] = float(
            torch.cuda.max_memory_allocated(device) / 2**20
        )
    return result


def resolve_devices(selection: str) -> tuple[list[torch.device], list[str]]:
    warnings: list[str] = []
    if selection == "all":
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        else:
            warnings.append("CUDA unavailable; only CPU benchmark was run")
        return devices, warnings
    if selection == "auto":
        return [
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ], warnings
    device = torch.device(selection)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return [device], warnings


def _cpu_model_name() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return platform.processor() or None


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _strict_json_value(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def strip_raw_samples(report: dict[str, Any]) -> None:
    for device_report in report.get("devices", {}).values():
        for path_report in device_report.get("paths", {}).values():
            for stage in path_report.get("stages", {}).values():
                stage.pop("samples_ms", None)


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    prepared = prepare_checkpoint(args.checkpoint)
    config_provenance = validate_pipeline_config_provenance(prepared, args.config)
    config = load_yaml_config(args.config)
    data_config = config["data"]
    frame_count = int(
        round(float(data_config["window_seconds"]) * float(data_config["radar_hz"]))
    )
    frame_alignment = int(
        round(float(data_config["radar_hz"]) / float(data_config["model_hz"]))
    )

    raw_file_source: RawFileSource | None = None
    manifest: DatasetManifest | None = None
    input_warnings: list[str] = []
    if args.input_source in {"auto", "raw-file"}:
        try:
            manifest = build_dataset_manifest(args.data_root)
            raw_file_source = resolve_raw_file_source(
                manifest,
                subject_id=args.subject,
                frame_count=frame_count,
                start_frame=args.window_start_frame,
                frame_alignment=frame_alignment,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            if args.input_source == "raw-file":
                raise
            input_warnings.append(
                f"raw file input unavailable ({exc}); deterministic synthetic input used"
            )
    if raw_file_source is not None:
        memory_raw = raw_file_source.read()
        subject_id = raw_file_source.subject_id
        input_kind = "raw_file_window"
        if raw_file_source.start_frame > 0:
            input_warnings.append(
                "outlier repair is initialized at the selected window boundary; "
                "the first four frames do not carry production pre-window repair state"
            )
    else:
        memory_raw = synthetic_raw_window(
            frames=frame_count,
            seed=args.seed,
        )
        subject_id = None
        input_kind = "deterministic_synthetic"
        input_warnings.append(
            "synthetic/window-local input has no production pre-window outlier-repair state"
        )

    # Validate feature/layout once before timing and obtain the generated base
    # dimension for legacy checkpoints that do not serialize aux_base_dim.
    validation_feature = preprocess_raw_window(memory_raw, data_config)
    if prepared.expected_aux_dim:
        base_dim = int(
            prepared.model_kwargs.get("aux_base_dim")
            or len(validation_feature.base_aux)
        )
        tail_dim = prepared.expected_aux_dim - base_dim
        if tail_dim < 0:
            raise ValueError("checkpoint aux_dim is smaller than generated base aux")
    else:
        tail_dim = 0
    history_tail, history_report = history_tail_from_cache(
        args.cache_dir,
        subject_id=subject_id,
        expected_tail_dim=tail_dim,
    )
    validation_inputs = construct_numpy_model_inputs(
        validation_feature, prepared, history_tail
    )

    devices, device_warnings = resolve_devices(args.devices)
    device_reports: dict[str, Any] = {}
    for device in devices:
        model = build_model(prepared, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        paths: dict[str, Any] = {}
        paths["raw_window_in_memory"] = run_latency_trials(
            model=model,
            prepared=prepared,
            data_config=data_config,
            history_tail=history_tail,
            device=device,
            raw_supplier=lambda raw=memory_raw: raw,
            include_raw_load=False,
            repeats=args.repeats,
            warmup=args.warmup,
            amp=args.amp,
        )
        if args.include_file_load and raw_file_source is not None:
            paths["warm_memmap_read_included"] = run_latency_trials(
                model=model,
                prepared=prepared,
                data_config=data_config,
                history_tail=history_tail,
                device=device,
                raw_supplier=raw_file_source.read,
                include_raw_load=True,
                repeats=args.file_repeats or args.repeats,
                warmup=args.warmup,
                amp=args.amp,
            )
        device_key = str(device)
        device_reports[device_key] = {
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else _cpu_model_name()
            ),
            "torch_threads": torch.get_num_threads(),
            "paths": paths,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    checkpoint_options = pipeline_option_summary(prepared)
    report: dict[str, Any] = {
        "benchmark_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(prepared.path),
            "sha256": sha256_file(prepared.path),
            "format_version": prepared.checkpoint.get("format_version"),
            "fold": prepared.checkpoint.get("fold"),
            "run_signature": prepared.checkpoint.get("run_signature"),
            "options": checkpoint_options,
            "model_kwargs": dict(prepared.model_kwargs),
            "trainable_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in TRAIN.build_model(
                        prepared.model_type, prepared.model_kwargs
                    ).parameters()
                    if parameter.requires_grad
                )
            ),
        },
        "pipeline": {
            "config_path": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config.resolve()),
            "data_config": dict(data_config),
            "checkpoint_config_provenance": config_provenance,
            "cache_dir": str(args.cache_dir.resolve()),
            "data_root": str(args.data_root.resolve()),
        },
        "input": {
            "kind": input_kind,
            "shape": list(memory_raw.shape),
            "dtype": str(memory_raw.dtype),
            "window_seconds": float(data_config["window_seconds"]),
            "radar_hz": float(data_config["radar_hz"]),
            "model_hz": float(data_config["model_hz"]),
            "subject_id": subject_id,
            "start_frame": (
                raw_file_source.start_frame if raw_file_source is not None else None
            ),
            "raw_paths": (
                raw_file_source.data_paths if raw_file_source is not None else None
            ),
            "feature_shapes": {
                "radar_map_before_branch": list(validation_feature.radar_map.shape),
                "radar_map_model": list(validation_inputs[0].shape),
                "base_aux": list(validation_feature.base_aux.shape),
                "model_aux": list(validation_inputs[1].shape),
            },
            "history_state": history_report,
            "outlier_count_in_selected_window": validation_feature.outlier_count,
            "warnings": input_warnings,
        },
        "measurement_contract": {
            "primary_start": "raw 32 s x 3 radar window resident in host memory",
            "primary_end": "synchronized batch-one model forward complete",
            "gpu_includes_host_to_device_transfer": True,
            "preprocessing_device": "CPU NumPy for both CPU and GPU model paths",
            "history_state": (
                "strictly causal past-window state is resident before the current window; "
                "timing includes append and checkpoint scaling"
            ),
            "outlier_repair_state": (
                "window-local initialization; first four frames can differ from the "
                "cache builder when a selected raw window starts after recording frame 0"
            ),
            "production_feature_bit_exact": False,
            "file_load_path": (
                "repeated memmap slice/copy with warm operating-system page cache; "
                "not a cold-disk benchmark"
            ),
            "cuda_barriers": "synchronize before timing and after transfer/forward",
            "batch_size": 1,
            "stride_budget_ms": 4000.0,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cpu": _cpu_model_name(),
            "cpu_count": os.cpu_count(),
            "seed": args.seed,
            "cudnn_benchmark": (
                bool(torch.backends.cudnn.benchmark)
                if hasattr(torch.backends, "cudnn")
                else None
            ),
            "cudnn_deterministic": (
                bool(torch.backends.cudnn.deterministic)
                if hasattr(torch.backends, "cudnn")
                else None
            ),
        },
        "warnings": device_warnings,
        "devices": device_reports,
    }
    if not args.save_samples:
        strip_raw_samples(report)
    write_json(args.output, report)
    print(f"wrote {args.output.resolve()}", flush=True)
    for device, device_report in device_reports.items():
        for name, path_report in device_report["paths"].items():
            total = path_report["stages"]["total_ms"]
            print(
                f"{device} {name}: p50={total['p50_ms']:.3f} ms "
                f"p95={total['p95_ms']:.3f} ms",
                flush=True,
            )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark batch-one latency from a resident raw three-radar window "
            "through window-local preprocessing and checkpoint forward inference."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/runs/final_structured_aux_s12/fold_0/snn_best.pt",
        help=(
            "trained teacher/SNN checkpoint; serialized structured and harmonic "
            "model options are loaded verbatim"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/default.yaml",
        help="preprocessing configuration used to train the checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "HAI_EXPERIMENT",
        help="raw dataset root used for the optional memmap path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/cache/rf32s",
        help="feature cache used only to obtain a representative causal history state",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/benchmarks/e2e_latency.json",
    )
    parser.add_argument(
        "--input-source",
        choices=("auto", "raw-file", "synthetic"),
        default="auto",
        help="auto uses raw data when present and otherwise deterministic synthetic input",
    )
    parser.add_argument("--subject", help="optional raw-data subject ID")
    parser.add_argument(
        "--window-start-frame",
        type=int,
        help="optional aligned raw frame offset; default selects a middle window",
    )
    parser.add_argument(
        "--devices",
        default="all",
        help="all, auto, cpu, cuda, or an explicit CUDA device such as cuda:0",
    )
    parser.add_argument(
        "--include-file-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also measure repeated warm-page-cache memmap read when raw files exist",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--file-repeats",
        type=int,
        help="optional repeat count for raw-file-included path (defaults to --repeats)",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use CUDA float16 autocast; CPU always uses float32",
    )
    parser.add_argument(
        "--save-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain individual timings in addition to p50/p95 summaries",
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.file_repeats is not None and args.file_repeats < 1:
        parser.error("--file-repeats must be positive")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.window_start_frame is not None and args.window_start_frame < 0:
        parser.error("--window-start-frame must be non-negative")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.config.is_file():
        parser.error(f"config does not exist: {args.config}")
    try:
        resolve_devices(args.devices)
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
