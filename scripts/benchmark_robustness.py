#!/usr/bin/env python3
"""Benchmark the final grouped SNN under sensor loss and deployment loads."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_rr.cache import (  # noqa: E402
    FeatureCache,
    append_causal_history_features,
    load_feature_cache,
    transform_aux,
)
from snn_rr.metrics import (  # noqa: E402
    grouped_oof_metrics,
    identity_macro_metrics,
    regression_metrics,
    risk_coverage_curve,
)


def _load_train_module():
    path = PROJECT_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("snn_rr_training_for_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = _load_train_module()


RADAR_MASKS = {
    "radars_123": (True, True, True),
    "radars_12": (True, True, False),
    "radars_13": (True, False, True),
    "radars_23": (False, True, True),
    "radar_1": (True, False, False),
    "radar_2": (False, True, False),
    "radar_3": (False, False, True),
}


@torch.inference_mode()
def predict_with_mask(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_pattern: tuple[bool, bool, bool],
    *,
    amp: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for batch_cpu in loader:
        batch = TRAIN._move_batch(batch_cpu, device)
        mask = torch.as_tensor(mask_pattern, device=device, dtype=torch.bool).expand(
            len(batch["rr"]), -1
        )
        with TRAIN._autocast_context(device, amp):
            output = model(batch["map"], mask, batch["aux"])
        values = {
            "index": batch_cpu["index"],
            "target": batch["rr"],
            "prediction": output["expected_rr"],
            "rr_std": output["rr_std"],
            "quality": output["quality"],
            "spike_rate": output["spike_rate_per_sample"],
            "radar_weights": output["radar_weights"],
        }
        for key, value in values.items():
            collected[key].append(value.detach().float().cpu().numpy())
    return {key: np.concatenate(value) for key, value in collected.items()}


def _summarize(
    result: dict[str, np.ndarray],
    metadata,
    fold_ids: np.ndarray,
    bootstrap_samples: int,
) -> dict[str, Any]:
    index = result["index"].astype(np.int64)
    identities = metadata.iloc[index]["identity"].astype(str).to_numpy()
    uncertainty = result["rr_std"] / np.clip(result["quality"], 0.05, None)
    summary = grouped_oof_metrics(
        result["target"],
        result["prediction"],
        identities,
        fold_ids=fold_ids,
        bootstrap_samples=bootstrap_samples,
    )
    summary["risk_coverage"] = risk_coverage_curve(
        result["target"],
        result["prediction"],
        uncertainty,
        identities=identities,
    )
    summary["mean_quality"] = float(np.mean(result["quality"]))
    summary["mean_spike_rate"] = float(np.mean(result["spike_rate"]))
    return summary


def _nonoverlap_indices(metadata) -> np.ndarray:
    """Greedily retain reference windows whose 32 s intervals do not overlap."""

    kept: list[int] = []
    for _, frame in metadata.groupby("session_id", sort=False):
        last_end = -np.inf
        ordered = frame.sort_values(["window_end_s", "window_number"])
        for index, row in ordered.iterrows():
            if float(row.window_start_s) >= last_end:
                kept.append(int(index))
                last_end = float(row.window_end_s)
    return np.asarray(kept, dtype=np.int64)


def _condition_summaries(full: dict[str, np.ndarray], metadata) -> dict[str, Any]:
    index = full["index"].astype(np.int64)
    frame = metadata.iloc[index].reset_index(drop=True)
    target = full["target"]
    prediction = full["prediction"]
    protocol = {
        str(name): regression_metrics(target[rows], prediction[rows])
        for name, rows in frame.groupby("protocol", sort=True).indices.items()
    }
    rr_bands = {}
    for low, high in ((6, 10), (10, 15), (15, 20), (20, 25), (25, 35), (35, 46)):
        selected = (target >= low) & (target < high)
        if selected.any():
            rr_bands[f"{low}_{high}_bpm"] = regression_metrics(
                target[selected], prediction[selected]
            )

    # Convert greedy original metadata indices to positions in this OOF result.
    valid_frame = metadata.iloc[index].copy()
    valid_frame.index = index
    selected_cache_index = set(_nonoverlap_indices(valid_frame).tolist())
    selected = np.asarray([value in selected_cache_index for value in index])
    identities = frame.loc[selected, "identity"].astype(str).to_numpy()
    nonoverlap = {
        **regression_metrics(target[selected], prediction[selected]),
        **identity_macro_metrics(target[selected], prediction[selected], identities),
    }

    error = np.abs(prediction - target)
    rr_std = np.clip(full["rr_std"], 1e-6, None)
    uncertainty = rr_std / np.clip(full["quality"], 0.05, None)
    interval = {
        "within_1_sigma": float(np.mean(error <= rr_std)),
        "within_90_interval": float(np.mean(error <= 1.645 * rr_std)),
        "within_95_interval": float(np.mean(error <= 1.96 * rr_std)),
        "within_99_interval": float(np.mean(error <= 2.576 * rr_std)),
    }
    detection = {}
    for threshold in (2.0, 5.0):
        event = error > threshold
        detection[f"error_over_{threshold:g}_bpm"] = {
            "fraction": float(np.mean(event)),
            "uncertainty_roc_auc": float(roc_auc_score(event, uncertainty)),
            "rr_std_roc_auc": float(roc_auc_score(event, rr_std)),
            "negative_quality_roc_auc": float(
                roc_auc_score(event, -full["quality"])
            ),
        }
    return {
        "per_protocol": protocol,
        "per_rr_band": rr_bands,
        "nonoverlapping_32s_windows": nonoverlap,
        "uncertainty_interval_coverage": interval,
        "error_detection": detection,
    }


@torch.inference_mode()
def _latency_once(
    model: torch.nn.Module,
    radar_map: Tensor,
    aux: Tensor,
    mask: Tensor,
    device: torch.device,
    *,
    repeats: int,
    warmup: int,
    amp: bool,
) -> dict[str, float]:
    model = model.to(device).eval()
    radar_map = radar_map.to(device)
    aux = aux.to(device)
    mask = mask.to(device)
    for _ in range(warmup):
        with TRAIN._autocast_context(device, amp):
            model(radar_map, mask, aux)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with TRAIN._autocast_context(device, amp):
            model(radar_map, mask, aux)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(1000.0 * (time.perf_counter() - start))
    values = np.asarray(samples)
    result = {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "min_ms": float(values.min()),
        "repeats": float(repeats),
    }
    if device.type == "cuda":
        result["peak_allocated_mb"] = float(
            torch.cuda.max_memory_allocated(device) / 2**20
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    amp = bool(device.type == "cuda")

    base_cache = load_feature_cache(args.cache_dir)
    augmented_aux, _ = append_causal_history_features(
        base_cache.aux, base_cache.metadata
    )
    cache = FeatureCache(
        base_cache.maps,
        augmented_aux,
        base_cache.metadata,
        base_cache.frequencies_hz,
    )
    conditions: dict[str, list[dict[str, np.ndarray]]] = {
        name: [] for name in RADAR_MASKS
    }
    condition_folds: dict[str, list[np.ndarray]] = {
        name: [] for name in RADAR_MASKS
    }
    latency_inputs = None
    latency_model = None
    first_checkpoint = None

    for fold in range(6):
        checkpoint_path = args.run_dir / f"fold_{fold}" / "snn_best.pt"
        model, checkpoint = TRAIN.load_checkpoint_model(checkpoint_path, device)
        test_identities = set(checkpoint["split"]["test_identities"])
        identity = cache.metadata["identity"].astype(str).to_numpy()
        valid = cache.metadata["reference_valid"].to_numpy(dtype=bool)
        test_index = np.flatnonzero(valid & np.isin(identity, list(test_identities)))
        aux = transform_aux(
            cache.aux,
            checkpoint["aux_center"].detach().cpu().numpy().astype(np.float32),
            checkpoint["aux_scale"].detach().cpu().numpy().astype(np.float32),
        )
        loader = TRAIN.make_loader(
            cache,
            aux,
            test_index,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            seed=20260827 + fold,
            train=False,
        )
        for name, pattern in RADAR_MASKS.items():
            result = predict_with_mask(model, loader, device, pattern, amp=amp)
            conditions[name].append(result)
            condition_folds[name].append(
                np.full(len(result["index"]), fold, dtype=np.int16)
            )
        if fold == 0:
            sample = test_index[:1]
            latency_inputs = (
                torch.from_numpy(np.asarray(cache.maps[sample])).float(),
                torch.from_numpy(aux[sample]).float(),
                torch.ones((1, 3), dtype=torch.bool),
            )
            latency_model = model
            first_checkpoint = checkpoint_path
        else:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report: dict[str, Any] = {"radar_conditions": {}}
    concatenated: dict[str, dict[str, np.ndarray]] = {}
    for name, parts in conditions.items():
        result = {
            key: np.concatenate([part[key] for part in parts]) for key in parts[0]
        }
        order = np.argsort(result["index"], kind="stable")
        result = {key: value[order] for key, value in result.items()}
        folds = np.concatenate(condition_folds[name])[order]
        concatenated[name] = result
        report["radar_conditions"][name] = _summarize(
            result, cache.metadata, folds, args.bootstrap_samples
        )
        np.savez_compressed(output_dir / f"{name}_oof.npz", **result, fold=folds)

    full = concatenated["radars_123"]
    report["stratified"] = _condition_summaries(full, cache.metadata)
    report["reference_coverage"] = {
        "valid_windows": int(cache.metadata.reference_valid.sum()),
        "total_windows": int(len(cache.metadata)),
        "fraction": float(cache.metadata.reference_valid.mean()),
    }
    report["model"] = {
        "checkpoint": str(first_checkpoint),
        "checkpoint_size_mb": float(first_checkpoint.stat().st_size / 2**20),
        "parameters": TRAIN.count_trainable_parameters(latency_model),
        "simulation_steps": int(latency_model.simulation_steps),
    }

    radar_map, aux, mask = latency_inputs
    latency: dict[str, Any] = {}
    if torch.cuda.is_available():
        latency["cuda_amp_batch1"] = _latency_once(
            latency_model,
            radar_map,
            aux,
            mask,
            torch.device("cuda"),
            repeats=args.latency_repeats,
            warmup=min(20, args.latency_repeats),
            amp=True,
        )
        latency["cuda_device"] = torch.cuda.get_device_name(0)
    cpu_model, _ = TRAIN.load_checkpoint_model(first_checkpoint, torch.device("cpu"))
    original_threads = torch.get_num_threads()
    for threads in (1, min(args.cpu_threads, os.cpu_count() or 1)):
        torch.set_num_threads(threads)
        latency[f"cpu_{threads}_threads_batch1"] = _latency_once(
            cpu_model,
            radar_map,
            aux,
            mask,
            torch.device("cpu"),
            repeats=max(10, args.latency_repeats // 4),
            warmup=5,
            amp=False,
        )
    torch.set_num_threads(original_threads)
    report["latency"] = latency

    TRAIN.write_json(output_dir / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("artifacts/runs/final_default_s12")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/cache/rf32s")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/robustness/final_default_s12"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"output": "artifacts/robustness/final_default_s12", "conditions": list(result["radar_conditions"])}, ensure_ascii=False))
