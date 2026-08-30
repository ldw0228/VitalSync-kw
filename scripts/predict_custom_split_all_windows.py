#!/usr/bin/env python3
"""Label-free inference for one immutable custom identity-split checkpoint.

Unlike the six-fold deployment predictor, this entry point owns exactly one
``prediction`` identity partition.  It deliberately predicts every cache row
for those identities (including rows without a valid reference) and treats
reference values only as masked output metadata.  The split manifest, its
referenced cache/fold artifacts, the checkpoint split, and the training-fitted
auxiliary scaler all fail closed before model construction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.predict_all_windows import (  # noqa: E402
    _as_numpy_scaler,
    _sha256_file,
    predict_label_free,
    validate_label_free_forward_interface,
    validate_model_kwargs,
)
from scripts.train import (  # noqa: E402
    FeatureCache,
    append_causal_history_features,
    build_model,
    fit_aux_scaler,
    load_feature_cache,
    make_loader,
    transform_aux,
)
from snn_rr.split_authority import (  # noqa: E402
    IdentitySplitAuthority,
    load_identity_split_authority,
)


FORMAT_VERSION = 1
OUTPUT_FIELDS = (
    "cache_index",
    "session_id",
    "identity",
    "protocol",
    "window_number",
    "reference_valid",
    "reference_rr_bpm",
    "prediction",
    "map_prediction",
    "rr_std",
    "uncertainty",
    "quality",
    "alias_probability",
    "posterior_entropy",
    "topk_rr",
    "topk_probability",
    "posterior_probability",
    "spike_rate",
    "radar_weights",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return result


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        PROJECT_ROOT / "scripts/predict_all_windows.py",
        PROJECT_ROOT / "scripts/train.py",
        PROJECT_ROOT / "src/snn_rr/split_authority.py",
        PROJECT_ROOT / "src/snn_rr/cache.py",
        PROJECT_ROOT / "src/snn_rr/models.py",
    )
    return {
        str(path.resolve().relative_to(PROJECT_ROOT)): _sha256_file(path.resolve())
        for path in paths
    }


def _resolve_recorded_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def prepare_custom_cache(
    cache_dir: Path, run_config: Mapping[str, Any]
) -> tuple[FeatureCache, int]:
    """Recreate branch selection and causal auxiliary topology from training."""

    arguments = run_config.get("arguments")
    if not isinstance(arguments, Mapping):
        raise RuntimeError("run_config arguments are missing")
    if _resolve_recorded_path(arguments.get("cache_dir")) != cache_dir.resolve():
        raise RuntimeError("--cache-dir differs from the cache bound in run_config")

    raw = load_feature_cache(cache_dir, mmap=False)
    stored_bins = int(raw.maps.shape[-1])
    if stored_bins % 2:
        raise RuntimeError("cache range dimension is not raw/phase separable")
    branch = str(arguments.get("map_branch", "both"))
    if branch == "both":
        maps = raw.maps
        expected_branches = 2
    elif branch == "raw":
        maps = raw.maps[..., : stored_bins // 2]
        expected_branches = 1
    elif branch == "phase":
        maps = raw.maps[..., stored_bins // 2 :]
        expected_branches = 1
    else:
        raise RuntimeError(f"unsupported checkpoint map_branch: {branch}")
    if int(arguments.get("input_branches", -1)) != expected_branches:
        raise RuntimeError("run_config map_branch/input_branches binding is inconsistent")

    cache = FeatureCache(
        maps=maps,
        aux=raw.aux,
        metadata=raw.metadata,
        frequencies_hz=raw.frequencies_hz,
    )
    base_aux_dim = int(cache.aux.shape[1])
    history_names: list[str] = []
    if bool(arguments.get("use_aux", False)) and bool(
        arguments.get("causal_history", False)
    ):
        augmented, history_names = append_causal_history_features(
            cache.aux, cache.metadata
        )
        cache = FeatureCache(
            maps=cache.maps,
            aux=augmented,
            metadata=cache.metadata,
            frequencies_hz=cache.frequencies_hz,
        )
    recorded_shape = run_config.get("cache_shape")
    observed_shape = {"maps": list(cache.maps.shape), "aux": list(cache.aux.shape)}
    if recorded_shape != observed_shape:
        raise RuntimeError(
            f"cache topology differs from run_config: {observed_shape} != {recorded_shape}"
        )
    if list(run_config.get("causal_history_feature_names", [])) != history_names:
        raise RuntimeError("causal-history feature schema differs from run_config")
    return cache, base_aux_dim


def validate_prediction_ownership(
    metadata: Any, prediction_identities: Sequence[str]
) -> np.ndarray:
    identities = metadata["identity"].astype(str).to_numpy()
    expected = np.flatnonzero(np.isin(identities, tuple(prediction_identities)))
    if len(expected) == 0:
        raise RuntimeError("custom prediction identities own no cache rows")
    observed = set(identities[expected].tolist())
    if observed != set(prediction_identities):
        raise RuntimeError("not every custom prediction identity owns a cache row")
    return expected.astype(np.int64, copy=False)


def validate_custom_checkpoint(
    checkpoint_path: Path,
    *,
    authority: IdentitySplitAuthority,
    cache: FeatureCache,
    base_aux_dim: int,
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate custom split, fold, scaler and model/cache provenance."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 2:
        raise RuntimeError("custom proposer checkpoint must use format_version=2")
    if checkpoint.get("model_type") != "snn":
        raise RuntimeError("custom proposer checkpoint must be an SNN")
    if int(checkpoint.get("fold", -1)) != authority.fold_id:
        raise RuntimeError("checkpoint fold does not match split manifest fold_id")
    if checkpoint.get("run_signature") != run_config.get("run_signature"):
        raise RuntimeError("checkpoint/run_config signature mismatch")
    expected_provenance = authority.checkpoint_provenance()
    if checkpoint.get("split_authority_provenance") != expected_provenance:
        raise RuntimeError("checkpoint split-authority provenance mismatch")

    split = checkpoint.get("split")
    if not isinstance(split, Mapping):
        raise RuntimeError("checkpoint custom split is missing")
    expected_split = {
        "train_identities": list(authority.train_identities),
        "validation_identities": list(authority.validation_identities),
        "prediction_identities": list(authority.prediction_identities),
        "excluded_identities": list(authority.excluded_identities),
        "scaler_identities": list(authority.scaler_identities),
    }
    normalized = {
        key: sorted(map(str, split.get(key, ()))) for key in expected_split
    }
    if normalized != {key: sorted(value) for key, value in expected_split.items()}:
        raise RuntimeError("checkpoint identities differ from the custom split manifest")

    arguments = run_config.get("arguments")
    if not isinstance(arguments, Mapping):
        raise RuntimeError("run_config arguments are missing")
    if str(arguments.get("identity_split_manifest_sha256", "")) != authority.content_sha256:
        raise RuntimeError("run_config split manifest content hash mismatch")
    validate_model_kwargs(
        checkpoint, cache, base_aux_dim=base_aux_dim, run_config=run_config
    )

    identity = cache.metadata["identity"].astype(str).to_numpy()
    reference_valid = cache.metadata["reference_valid"].to_numpy(dtype=bool)
    train_mask = np.isin(identity, authority.train_identities)
    if not bool(arguments.get("include_invalid", False)):
        train_mask &= reference_valid
    train_index = np.flatnonzero(train_mask)
    authority.validate_scaler_indices(cache.metadata, train_index)
    expected_center, expected_scale = fit_aux_scaler(cache.aux, train_index)
    center = _as_numpy_scaler(checkpoint, "aux_center")
    scale = _as_numpy_scaler(checkpoint, "aux_scale")
    use_aux = bool(arguments.get("use_aux", False))
    if use_aux:
        if center.shape != (cache.aux.shape[1],) or scale.shape != center.shape:
            raise RuntimeError("checkpoint auxiliary scaler dimension mismatch")
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise RuntimeError("checkpoint auxiliary scaler is invalid")
        if not np.allclose(center, expected_center, rtol=1e-6, atol=1e-7):
            raise RuntimeError("checkpoint auxiliary center is not train-only fitted")
        if not np.allclose(scale, expected_scale, rtol=1e-6, atol=1e-7):
            raise RuntimeError("checkpoint auxiliary scale is not train-only fitted")
    elif center.size or scale.size:
        raise RuntimeError("aux-disabled checkpoint unexpectedly contains a scaler")
    return checkpoint


def _posterior_grid(checkpoint: Mapping[str, Any]) -> np.ndarray:
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping) or "rr_bins" not in state:
        raise RuntimeError("checkpoint lacks posterior RR grid")
    grid = np.asarray(state["rr_bins"].detach().cpu(), dtype=np.float32)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all():
        raise RuntimeError("checkpoint posterior RR grid is invalid")
    return grid


def build_output_arrays(
    bundle: Any,
    metadata: Any,
    expected_index: np.ndarray,
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    authority: IdentitySplitAuthority,
    run_config_path: Path,
) -> dict[str, Any]:
    index = np.asarray(bundle.index, dtype=np.int64)
    if not np.array_equal(index, expected_index):
        raise RuntimeError("inference did not exactly cover prediction-identity rows")
    if len(np.unique(index)) != len(index):
        raise RuntimeError("inference returned duplicate cache rows")
    rows = metadata.iloc[index]
    observed_identities = set(rows["identity"].astype(str))
    if observed_identities != set(authority.prediction_identities):
        raise RuntimeError("inference result violates prediction identity ownership")

    reference_valid = rows["reference_valid"].to_numpy(dtype=bool)
    if not np.array_equal(np.asarray(bundle.reference_valid, dtype=bool), reference_valid):
        raise RuntimeError("prediction bundle reference mask differs from cache metadata")
    reference_rr = rows["rr_bpm"].to_numpy(dtype=np.float32)
    reference_rr = np.where(reference_valid, reference_rr, np.nan).astype(np.float32)

    source_hashes = _runtime_source_hashes()
    provenance: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "custom_identity_split_label_free_all_windows",
        "strict_nested_role": "prediction",
        "strict_retrospective": True,
        "labels_forwarded_to_model": False,
        "reference_invalid_rows_included": True,
        "commercial_performance_claim_eligible": False,
        "fold_id": authority.fold_id,
        "prediction_identities": list(authority.prediction_identities),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_run_signature": str(checkpoint.get("run_signature", "")),
        "run_config_path": str(run_config_path.resolve()),
        "run_config_sha256": _sha256_file(run_config_path),
        "split_manifest_path": str(authority.manifest_path),
        "split_manifest_file_sha256": authority.manifest_file_sha256,
        "split_manifest_content_sha256": authority.content_sha256,
        "fold_assignments_sha256": authority.fold_assignments_sha256,
        "cache_manifest_sha256": authority.cache_manifest_sha256,
        "source_hashes": source_hashes,
        "row_count": len(index),
        "reference_valid_count": int(reference_valid.sum()),
        "output_allowlist": list(OUTPUT_FIELDS),
        "excluded_model_inputs": ["reference_rr", "reference_valid", "identity", "protocol"],
    }
    provenance["inference_signature_sha256"] = _canonical_hash(provenance)
    arrays: dict[str, Any] = {
        "cache_index": index,
        "session_id": rows["session_id"].astype(str).to_numpy(dtype=np.str_),
        "identity": rows["identity"].astype(str).to_numpy(dtype=np.str_),
        "protocol": rows["protocol"].astype(str).to_numpy(dtype=np.str_),
        "window_number": rows["window_number"].to_numpy(dtype=np.int32),
        "reference_valid": reference_valid,
        "reference_rr_bpm": reference_rr,
        "prediction": np.asarray(bundle.prediction, dtype=np.float32),
        "map_prediction": np.asarray(bundle.map_prediction, dtype=np.float32),
        "rr_std": np.asarray(bundle.rr_std, dtype=np.float32),
        "uncertainty": np.asarray(bundle.uncertainty, dtype=np.float32),
        "quality": np.asarray(bundle.quality, dtype=np.float32),
        "alias_probability": np.asarray(bundle.alias_probability, dtype=np.float32),
        "posterior_entropy": np.asarray(bundle.posterior_entropy, dtype=np.float32),
        "topk_rr": np.asarray(bundle.topk_rr, dtype=np.float32),
        "topk_probability": np.asarray(bundle.topk_probability, dtype=np.float32),
        "posterior_probability": np.asarray(bundle.posterior_probability, dtype=np.float16),
        "spike_rate": np.asarray(bundle.spike_rate, dtype=np.float32),
        "radar_weights": np.asarray(bundle.radar_weights, dtype=np.float32),
        "posterior_rr_grid_bpm": _posterior_grid(checkpoint),
        "fold_id": np.asarray(authority.fold_id, dtype=np.int16),
        "checkpoint_sha256": np.asarray(provenance["checkpoint_sha256"]),
        "split_manifest_file_sha256": np.asarray(authority.manifest_file_sha256),
        "split_manifest_content_sha256": np.asarray(authority.content_sha256),
        "fold_assignments_sha256": np.asarray(authority.fold_assignments_sha256),
        "cache_manifest_sha256": np.asarray(authority.cache_manifest_sha256),
        "run_config_sha256": np.asarray(provenance["run_config_sha256"]),
        "inference_signature_sha256": np.asarray(provenance["inference_signature_sha256"]),
        "strict_retrospective": np.asarray(True),
        "strict_nested_prediction_role": np.asarray(True),
        "provenance_json": np.asarray(
            json.dumps(provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        ),
    }
    for name in OUTPUT_FIELDS:
        if name not in arrays:
            raise RuntimeError(f"internal output allowlist omission: {name}")
    return arrays


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = args.cache_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.identity_split_manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"output exists; pass --force to replace: {output_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    run_config_path = checkpoint_path.parent.parent / "run_config.json"
    run_config = _load_json(run_config_path)
    cache, base_aux_dim = prepare_custom_cache(cache_dir, run_config)
    authority = load_identity_split_authority(
        manifest_path, metadata=cache.metadata, cache_dir=cache_dir
    )
    checkpoint = validate_custom_checkpoint(
        checkpoint_path,
        authority=authority,
        cache=cache,
        base_aux_dim=base_aux_dim,
        run_config=run_config,
    )
    expected_index = validate_prediction_ownership(
        cache.metadata, authority.prediction_identities
    )

    arguments = run_config["arguments"]
    if bool(arguments.get("use_aux", False)):
        center = _as_numpy_scaler(checkpoint, "aux_center")
        scale = _as_numpy_scaler(checkpoint, "aux_scale")
        aux = transform_aux(cache.aux, center, scale)
    else:
        aux = np.empty((len(cache.metadata), 0), dtype=np.float32)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp = bool(args.amp and device.type == "cuda")
    loader = make_loader(
        cache,
        aux,
        expected_index,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        seed=int(arguments.get("seed", 0)) + 99001,
        train=False,
    )
    model = build_model(str(checkpoint["model_type"]), checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device)
    validate_label_free_forward_interface(model)
    bundle = predict_label_free(model, loader, device, amp=amp)
    arrays = build_output_arrays(
        bundle,
        cache.metadata,
        expected_index,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        authority=authority,
        run_config_path=run_config_path,
    )
    _atomic_npz(output_path, arrays)
    return {
        "output": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "rows": len(expected_index),
        "invalid_reference_rows": int((~arrays["reference_valid"]).sum()),
        "fold_id": authority.fold_id,
        "prediction_identities": list(authority.prediction_identities),
        "inference_signature_sha256": str(arrays["inference_signature_sha256"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity-split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
