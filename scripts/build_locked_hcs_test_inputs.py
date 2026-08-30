#!/usr/bin/env python3
"""Build target-free outer-test proposer and HCS inference artifacts.

The ordinary harmonic cache builder deliberately retains training targets in
``metadata.csv`` and therefore is not suitable for the post-lock test path.
This module provides three narrow operations for that path:

``stitch``
    Select only the previously unavailable outer-test rows from an immutable
    discovery stack and fill them from one locked test-proposer prediction.
    Reference arrays present in the proposer archive are never loaded.
``cache``
    Rebuild candidate/node features for those rows while reading RF/SVD CSVs
    through an explicit semantic/forward-only ``usecols`` allow-list.
``predict``
    Run a frozen HCS checkpoint with dummy, invalid targets and emit only the
    target-free raw fields consumed by :mod:`run_locked_hcs_oof`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts import build_harmonic_set_cache as BASE  # noqa: E402
from scripts import train as PROPOSER_TRAIN  # noqa: E402
from scripts.predict_all_windows import (  # noqa: E402
    _as_numpy_scaler,
    predict_label_free,
)
from snn_rr.harmonic_set_data import (  # noqa: E402
    CANDIDATE_SOURCE_NAMES,
    FORWARD_METADATA_ALLOWLIST,
    HARMONIC_RATIOS,
    CandidateSource,
    candidate_bank_from_metadata,
    iter_compact_node_feature_batches,
)


FORMAT_VERSION = 1
SAFE_SEMANTIC_FIELDS = (
    "session_id",
    "identity",
    "protocol",
    "window_number",
    "window_start_s",
    "window_end_s",
)
SAFE_METADATA_FIELDS = tuple(dict.fromkeys((*SAFE_SEMANTIC_FIELDS, *FORWARD_METADATA_ALLOWLIST)))
PROPOSER_ROW_FIELDS = (
    "prediction",
    "map_prediction",
    "rr_std",
    "uncertainty",
    "quality",
    "alias_probability",
    "posterior_entropy",
    "spike_rate",
)
PROPOSER_VECTOR_FIELDS = (
    "topk_rr",
    "topk_probability",
    "posterior_probability",
    "radar_weights",
)
LABEL_FIELDS = frozenset(
    {
        "rr_bpm",
        "reference_valid",
        "reference_rr_bpm",
        "target",
        "target_rr_bpm",
        "label",
        "labels",
        "ground_truth",
    }
)
SAFE_PROPOSER_METADATA_FIELDS = tuple(
    dict.fromkeys(
        (
            *SAFE_METADATA_FIELDS,
            "radar_peak_spread_bpm",
            "classical_rr_bpm",
            "classical_confidence",
        )
    )
)
OUTPUT_ARRAY_FILES = {
    "node_features": "node_features.npy",
    "candidate_bpm": "candidate_bpm.npy",
    "candidate_mask": "candidate_mask.npy",
    "joint_radar_mask": "joint_radar_mask.npy",
    "fallback_rr_bpm": "fallback_rr_bpm.npy",
    "fallback_std_bpm": "fallback_std_bpm.npy",
}
RADAR_MASKS: dict[str, tuple[bool, bool, bool]] = {
    "radars_123": (True, True, True),
    "radars_12": (True, True, False),
    "radars_13": (True, False, True),
    "radars_23": (False, True, True),
    "radar_1": (True, False, False),
    "radar_2": (False, True, False),
    "radar_3": (False, False, True),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _scalar(archive: Mapping[str, Any], name: str) -> Any:
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise RuntimeError(f"{name} must be a scalar")
    return value.item()


def stitch_test_stack(
    *,
    discovery_stack: Path,
    test_prediction: Path,
    test_manifest: Path,
    checkpoint: Path,
    outer_fold: int,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    """Fill only discovery-stack outer-test rows without loading references."""

    manifest = _json(test_manifest, "test split manifest")
    identities = manifest.get("identities")
    if (
        manifest.get("schema_version") != 1
        or not isinstance(identities, Mapping)
        or not isinstance(identities.get("prediction"), list)
    ):
        raise RuntimeError("test split manifest has no prediction identity partition")
    content = dict(manifest)
    recorded_content_hash = str(content.pop("content_sha256", ""))
    encoded = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != recorded_content_hash:
        raise RuntimeError("test split manifest canonical content hash mismatch")
    with np.load(discovery_stack, allow_pickle=False) as discovery:
        if bool(_scalar(discovery, "outer_test_opened")):
            raise RuntimeError("discovery stack already records outer-test access")
        if not bool(_scalar(discovery, "strict_nested")):
            raise RuntimeError("discovery stack is not strict nested")
        if int(_scalar(discovery, "outer_fold")) != int(outer_fold):
            raise RuntimeError("discovery stack outer fold mismatch")
        if int(_scalar(discovery, "seed")) != int(seed):
            raise RuntimeError("discovery stack seed mismatch")
        available = np.asarray(discovery["proposal_available"], dtype=bool)
        test_position = np.flatnonzero(~available)
        semantics = {
            name: np.asarray(discovery[name])[test_position].copy()
            for name in (
                "cache_index",
                "session_id",
                "identity",
                "protocol",
                "window_number",
                "window_start_s",
                "window_end_s",
                "fold",
            )
        }
    if len(test_position) == 0 or set(semantics["identity"].astype(str)) != set(
        map(str, identities["prediction"])
    ):
        raise RuntimeError("discovery unavailable rows differ from test manifest ownership")
    if not np.all(semantics["fold"].astype(int) == int(outer_fold)):
        raise RuntimeError("discovery unavailable rows are not exclusively the outer fold")
    with np.load(test_prediction, allow_pickle=False) as proposer:
        # Merely listing fields is safe; reference arrays are deliberately not read.
        required = {
            "cache_index",
            "session_id",
            "identity",
            "protocol",
            "window_number",
            "posterior_rr_grid_bpm",
            "checkpoint_sha256",
            "split_manifest_file_sha256",
            "split_manifest_content_sha256",
            "strict_nested_prediction_role",
            *PROPOSER_ROW_FIELDS,
            *PROPOSER_VECTOR_FIELDS,
        }
        missing = sorted(required - set(proposer.files))
        if missing:
            raise RuntimeError(f"test proposer prediction fields are missing: {missing}")
        if not bool(_scalar(proposer, "strict_nested_prediction_role")):
            raise RuntimeError("test proposer archive has the wrong semantic role")
        if str(_scalar(proposer, "checkpoint_sha256")) != sha256_file(checkpoint):
            raise RuntimeError("test proposer/checkpoint hash mismatch")
        if str(_scalar(proposer, "split_manifest_file_sha256")) != sha256_file(test_manifest):
            raise RuntimeError("test proposer/manifest file hash mismatch")
        if str(_scalar(proposer, "split_manifest_content_sha256")) != recorded_content_hash:
            raise RuntimeError("test proposer/manifest content hash mismatch")
        index = np.asarray(proposer["cache_index"], dtype=np.int64)
        if not np.array_equal(index, semantics["cache_index"].astype(np.int64)):
            raise RuntimeError("test proposer does not exactly fill discovery unavailable rows")
        for name in ("session_id", "identity", "protocol", "window_number"):
            observed = np.asarray(proposer[name])
            expected = np.asarray(semantics[name])
            if not np.array_equal(observed.astype(expected.dtype), expected):
                raise RuntimeError(f"test proposer/discovery semantic mismatch: {name}")
        proposal_arrays = {
            name: np.asarray(proposer[name]).copy()
            for name in (*PROPOSER_ROW_FIELDS, *PROPOSER_VECTOR_FIELDS)
        }
        grid = np.asarray(proposer["posterior_rr_grid_bpm"], dtype=np.float32).copy()
    arrays: dict[str, Any] = {
        **semantics,
        **proposal_arrays,
        "posterior_rr_grid_bpm": grid,
        "proposal_available": np.ones(len(index), dtype=bool),
        "nested_role": np.full(len(index), "hcs_test_post_lock", dtype=np.str_),
        "outer_fold": np.asarray(outer_fold, dtype=np.int16),
        "seed": np.asarray(seed, dtype=np.int64),
        "strict_nested": np.asarray(True),
        "discovery_stack_preserved": np.asarray(True),
        "target_fields_present": np.asarray(False),
    }
    provenance = {
        "format_version": FORMAT_VERSION,
        "classification": "post_lock_outer_test_only_proposer_stack",
        "outer_fold": int(outer_fold),
        "seed": int(seed),
        "row_scope": "outer_test_only",
        "target_fields_read": False,
        "target_fields_present": False,
        "discovery_stack": _binding(discovery_stack),
        "test_prediction": _binding(test_prediction),
        "test_manifest": _binding(test_manifest),
        "test_proposer_checkpoint": _binding(checkpoint),
        "row_count": len(index),
    }
    arrays["provenance_json"] = np.asarray(
        json.dumps(provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    _atomic_npz(output, arrays)
    return {"output": _binding(output), **provenance}


def _read_safe_metadata(path: Path) -> pd.DataFrame:
    """Read values only from a compile-time allow-list (never target/QC columns)."""

    frame = pd.read_csv(path, usecols=list(SAFE_METADATA_FIELDS))
    if set(frame.columns) != set(SAFE_METADATA_FIELDS):
        raise RuntimeError(f"safe metadata schema is incomplete: {path}")
    if set(frame.columns) & LABEL_FIELDS:
        raise RuntimeError("internal error: label field entered safe metadata frame")
    return frame.loc[:, list(SAFE_METADATA_FIELDS)]


def _template_settings(path: Path) -> dict[str, Any]:
    manifest = _json(path, "template HCS cache manifest")
    settings = manifest.get("settings")
    if manifest.get("format_version") != 1 or not isinstance(settings, Mapping):
        raise RuntimeError("template HCS cache manifest is incompatible")
    required = {
        "merge_radius_bpm",
        "batch_size",
        "proposal_selection",
        "posterior_nms_suppression_bpm",
        "base_proposals",
        "proposer_features",
        "svd_components",
    }
    if required - set(settings):
        raise RuntimeError("template HCS cache manifest lacks frozen settings")
    return dict(settings)


def build_test_cache(
    *,
    rf_cache: Path,
    svd_cache: Path,
    proposer: Path,
    template_cache_manifest: Path,
    outer_fold: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Build an outer-test-only feature cache without materializing targets."""

    settings = _template_settings(template_cache_manifest)
    _, svd_root, sessions = BASE._validate_root_manifests(rf_cache, svd_cache)
    if int(svd_root.get("components", -1)) < int(settings["svd_components"]):
        raise RuntimeError("SVD cache has fewer components than the frozen template")
    with np.load(proposer, allow_pickle=False) as archive:
        if set(archive.files) & LABEL_FIELDS:
            raise RuntimeError("test-only proposer unexpectedly contains a target field")
        proposer_data = {name: np.asarray(archive[name]).copy() for name in archive.files}
    proposer_frame = BASE._proposer_frame(proposer_data)
    bundle = BASE._proposal_bundle(
        proposer_data,
        selection=str(settings["proposal_selection"]),
        suppression_bpm=float(settings["posterior_nms_suppression_bpm"]),
        base_proposals=str(settings["base_proposals"]),
        include_features=bool(settings["proposer_features"]),
    )
    wanted_index = proposer_frame["cache_index"].to_numpy(np.int64)
    if len(wanted_index) == 0 or len(np.unique(wanted_index)) != len(wanted_index):
        raise RuntimeError("test-only proposer indices are empty or duplicated")
    if not np.all(proposer_frame["fold"].to_numpy(int) == int(outer_fold)):
        raise RuntimeError("test-only proposer contains a non-test fold")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building.", dir=output_dir.parent))
    row_count = len(wanted_index)
    candidate_bpm = np.zeros((row_count, BASE.MAX_CANDIDATES), dtype=np.float32)
    candidate_mask = np.zeros((row_count, BASE.MAX_CANDIDATES), dtype=bool)
    joint_radar_mask = np.zeros((row_count, BASE.MAX_CANDIDATES, 3), dtype=bool)
    node_features: np.ndarray | None = None
    feature_names: tuple[str, ...] | None = None
    safe_metadata_parts: list[pd.DataFrame] = []
    global_offset = 0
    output_offset = 0
    try:
        for session_id in sessions:
            rf_dir = rf_cache / session_id
            svd_dir = svd_cache / session_id
            rf_metadata = _read_safe_metadata(rf_dir / "metadata.csv")
            svd_metadata = _read_safe_metadata(svd_dir / "metadata.csv")
            BASE._assert_common_rows(rf_metadata, svd_metadata, f"safe RF/SVD {session_id}")
            local_rows = len(rf_metadata)
            owned_mask = (wanted_index >= global_offset) & (
                wanted_index < global_offset + local_rows
            )
            owned_global = wanted_index[owned_mask]
            if len(owned_global) == 0:
                global_offset += local_rows
                continue
            local = owned_global - global_offset
            expected_slice = np.arange(output_offset, output_offset + len(local))
            if not np.array_equal(np.flatnonzero(owned_mask), expected_slice):
                raise RuntimeError("test proposer rows are not in canonical session order")
            selected_rf = rf_metadata.iloc[local].reset_index(drop=True)
            selected_svd = svd_metadata.iloc[local].reset_index(drop=True)
            proposer_selected = proposer_frame.iloc[expected_slice].reset_index(drop=True)
            for name in SAFE_SEMANTIC_FIELDS:
                left = selected_rf[name]
                right = proposer_selected[name]
                if name in {"session_id", "identity", "protocol"}:
                    equal = np.array_equal(left.astype(str), right.astype(str))
                elif name == "window_number":
                    equal = np.array_equal(left.to_numpy(np.int64), right.to_numpy(np.int64))
                else:
                    equal = np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=0, atol=5e-7)
                if not equal:
                    raise RuntimeError(f"safe metadata/proposer semantic mismatch: {name}")
            rf_maps_all = np.load(rf_dir / "maps.npy", mmap_mode="r", allow_pickle=False)
            rf_frequency = np.load(rf_dir / "frequencies_hz.npy", allow_pickle=False)
            svd_spectra_all = np.load(svd_dir / "spectra.npy", mmap_mode="r", allow_pickle=False)
            svd_attributes_all = np.load(svd_dir / "attributes.npy", mmap_mode="r", allow_pickle=False)
            svd_frequency = np.load(svd_dir / "frequencies_hz.npy", allow_pickle=False)
            rf_maps = np.asarray(rf_maps_all[local])
            svd_spectra = np.asarray(svd_spectra_all[local])
            svd_attributes = np.asarray(svd_attributes_all[local])
            raw_rf = rf_maps[..., :91]
            raw_radar_mask = np.isfinite(raw_rf).all(axis=(2, 3)) & np.any(
                raw_rf != 0, axis=(2, 3)
            )
            selector = slice(output_offset, output_offset + len(local))
            bank = candidate_bank_from_metadata(
                selected_rf,
                proposal_bpm=bundle.bpm[selector],
                proposal_confidence=bundle.confidence[selector],
                proposal_mask=bundle.mask[selector],
                proposal_source=bundle.source[selector],
                merge_radius_bpm=float(settings["merge_radius_bpm"]),
                max_candidates=BASE.MAX_CANDIDATES,
            )
            candidate_bpm[selector] = bank.bpm
            candidate_mask[selector] = bank.mask
            proposer_nodes = None
            proposer_names = None
            if bool(settings["proposer_features"]):
                proposer_nodes, proposer_names = BASE.proposer_candidate_node_features(
                    bundle, bank, selector
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
                batch_size=int(settings["batch_size"]),
                svd_components=int(settings["svd_components"]),
                proposer_node_features=proposer_nodes,
                proposer_feature_names=proposer_names,
                include_source_confidence=bool(settings["proposer_features"]),
            ):
                begin = output_offset + int(batch.row_slice.start or 0)
                end = output_offset + int(batch.row_slice.stop or len(local))
                names = tuple(batch.nodes.feature_names)
                values = np.asarray(batch.nodes.features, dtype=np.float32).copy()
                phase = np.asarray(
                    ["_candidate_iq_phase_power_" in name for name in names], dtype=bool
                )
                values[..., phase] = 0.0
                if node_features is None:
                    node_features = np.zeros(
                        (row_count, BASE.MAX_CANDIDATES, len(names)), dtype=np.float32
                    )
                    feature_names = names
                elif feature_names != names:
                    raise RuntimeError("node feature schema changed between sessions")
                node_features[begin:end] = values
                joint_radar_mask[begin:end] = np.asarray(
                    batch.rf_support.radar_mask, dtype=bool
                ) & np.asarray(batch.svd_support.radar_mask, dtype=bool)
            safe = selected_rf.loc[:, list(SAFE_METADATA_FIELDS)].copy()
            safe.insert(0, "cache_index", owned_global)
            safe.insert(1, "outer_fold", int(outer_fold))
            safe_metadata_parts.append(safe)
            output_offset += len(local)
            global_offset += local_rows
        if output_offset != row_count or node_features is None or feature_names is None:
            raise RuntimeError("safe cache build did not exactly cover test proposer rows")
        safe_metadata = pd.concat(safe_metadata_parts, ignore_index=True)
        if set(safe_metadata.columns) & LABEL_FIELDS:
            raise RuntimeError("label field entered derived safe metadata")
        fallback = np.asarray(proposer_data["prediction"], dtype=np.float32)
        fallback_std = np.asarray(proposer_data["rr_std"], dtype=np.float32)
        arrays = {
            "node_features": node_features,
            "candidate_bpm": candidate_bpm,
            "candidate_mask": candidate_mask,
            "joint_radar_mask": joint_radar_mask,
            "fallback_rr_bpm": fallback,
            "fallback_std_bpm": fallback_std,
        }
        for name, filename in OUTPUT_ARRAY_FILES.items():
            np.save(stage / filename, arrays[name], allow_pickle=False)
        safe_metadata.to_csv(stage / "metadata_safe.csv", index=False)
        _write_json(stage / "feature_names.json", {"node_feature_names": list(feature_names)})
        outputs = {
            name: _binding(stage / filename) for name, filename in OUTPUT_ARRAY_FILES.items()
        }
        outputs["metadata_safe"] = _binding(stage / "metadata_safe.csv")
        outputs["feature_names"] = _binding(stage / "feature_names.json")
        manifest = {
            "schema_version": FORMAT_VERSION,
            "artifact_type": "hcs_test_only_inference_cache",
            "row_scope": "outer_test_only",
            "target_fields_present": False,
            "target_fields_read": False,
            "metadata_read_usecols": list(SAFE_METADATA_FIELDS),
            "outer_fold": int(outer_fold),
            "seed": int(seed),
            "row_count": row_count,
            "node_feature_shape": list(node_features.shape),
            "settings": settings,
            "inputs": {
                "rf_root_manifest": _binding(rf_cache / "manifest.json"),
                "svd_root_manifest": _binding(svd_cache / "manifest.json"),
                "test_only_proposer": _binding(proposer),
                "template_cache_manifest": _binding(template_cache_manifest),
                "builder_source": _binding(Path(__file__)),
            },
            "outputs": outputs,
        }
        _write_json(stage / "manifest.json", manifest)
        stage.replace(output_dir)
        return {"output_dir": str(output_dir), "manifest": manifest}
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def predict_hcs(
    *,
    cache_dir: Path,
    selection_lock: Path,
    checkpoint: Path,
    scaler: Path,
    outer_fold: int,
    seed: int,
    output: Path,
    device_name: str,
    amp: bool,
) -> dict[str, Any]:
    """Run frozen HCS inference with no target artifact in memory or output."""

    lock = _json(selection_lock, "HCS selection lock")
    if (
        int(lock.get("outer_fold", -1)) != int(outer_fold)
        or int(lock.get("seed", -1)) != int(seed)
        or int(lock.get("adaptive_iteration", -1)) != 3
        or lock.get("outer_test_not_opened_before_this_lock") is not True
    ):
        raise RuntimeError("HCS selection lock identity/leakage mismatch")
    if lock.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError("HCS checkpoint differs from selection lock")
    if lock.get("scaler_sha256") != sha256_file(scaler):
        raise RuntimeError("HCS scaler differs from selection lock")
    manifest = _json(cache_dir / "manifest.json", "test-only cache manifest")
    if (
        manifest.get("artifact_type") != "hcs_test_only_inference_cache"
        or manifest.get("target_fields_present") is not False
        or manifest.get("target_fields_read") is not False
        or int(manifest.get("outer_fold", -1)) != int(outer_fold)
        or int(manifest.get("seed", -1)) != int(seed)
    ):
        raise RuntimeError("HCS inference cache is not target-free and unit-bound")
    trainer_path = PROJECT_ROOT / "scripts/train_harmonic_set_snn.py"
    spec = importlib.util.spec_from_file_location("locked_hcs_predict_trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen HCS trainer")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint_payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise RuntimeError("HCS checkpoint has no model configuration")
    node = np.load(cache_dir / OUTPUT_ARRAY_FILES["node_features"], mmap_mode="r")
    candidate = np.load(cache_dir / OUTPUT_ARRAY_FILES["candidate_bpm"], mmap_mode="r")
    candidate_mask = np.load(cache_dir / OUTPUT_ARRAY_FILES["candidate_mask"], mmap_mode="r")
    radar = np.load(cache_dir / OUTPUT_ARRAY_FILES["joint_radar_mask"], mmap_mode="r")
    fallback = np.load(cache_dir / OUTPUT_ARRAY_FILES["fallback_rr_bpm"], mmap_mode="r")
    fallback_std = np.load(cache_dir / OUTPUT_ARRAY_FILES["fallback_std_bpm"], mmap_mode="r")
    metadata = pd.read_csv(cache_dir / "metadata_safe.csv")
    rows = len(metadata)
    if any(len(array) != rows for array in (node, candidate, candidate_mask, radar, fallback, fallback_std)):
        raise RuntimeError("test-only cache row shapes disagree")
    metadata = metadata.copy()
    metadata["fold"] = int(outer_fold)
    metadata["rr_bpm"] = np.nan
    metadata["reference_valid"] = False
    experiment = trainer.Experiment(
        root=cache_dir,
        metadata=metadata,
        node_features=node,
        candidate_rr=candidate,
        candidate_mask=candidate_mask,
        radar_mask=radar,
        base_prediction=np.asarray(fallback, dtype=np.float32),
        base_std=np.asarray(fallback_std, dtype=np.float32),
        base_available=np.ones(rows, dtype=bool),
        manifest=manifest,
    )
    scaler_document = _json(scaler, "HCS scaler")
    robust = trainer.RobustNodeScaler(
        center=np.asarray(scaler_document["center"], dtype=np.float32).reshape(1, 1, -1),
        scale=np.asarray(scaler_document["scale"], dtype=np.float32).reshape(1, 1, -1),
        fit_positions_sha256=str(scaler_document["fit_positions_sha256"]),
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = trainer.HarmonicCandidateSetEpisodeSNN(**dict(model_config)).to(device)
    model.load_state_dict(checkpoint_payload["model_state"], strict=True)
    prediction = trainer.predict_positions(
        model,
        experiment,
        np.arange(rows, dtype=np.int64),
        robust,
        device,
        amp=bool(amp and device.type == "cuda"),
    )
    arrays = {
        "cache_index": metadata["cache_index"].to_numpy(np.int64),
        "fallback_rr_bpm": np.asarray(prediction.base_prediction, dtype=np.float32),
        "fallback_std_bpm": np.asarray(prediction.base_std, dtype=np.float32),
        "fallback_available": np.asarray(prediction.base_available, dtype=bool),
        "source_rr_bpm": np.asarray(prediction.source_prediction, dtype=np.float32),
        "source_scale_bpm": np.asarray(prediction.source_scale, dtype=np.float32),
        "source_available": np.asarray(prediction.source_available, dtype=bool),
        "selected_probability": np.asarray(prediction.selected_probability, dtype=np.float32),
        "margin": np.asarray(prediction.margin, dtype=np.float32),
        "entropy": np.asarray(prediction.entropy, dtype=np.float32),
        "quality": np.asarray(prediction.quality, dtype=np.float32),
        "valid_candidate_count": np.asarray(prediction.valid_candidate_count, dtype=np.int16),
        "normalized_entropy": np.asarray(prediction.normalized_entropy, dtype=np.float32),
        "outer_fold": np.asarray(outer_fold, dtype=np.int16),
        "seed": np.asarray(seed, dtype=np.int64),
        "target_fields_present": np.asarray(False),
    }
    if set(arrays) & LABEL_FIELDS:
        raise RuntimeError("internal error: target field entered raw inference")
    _atomic_npz(output, arrays)
    return {
        "output": _binding(output),
        "rows": rows,
        "target_fields_present": False,
        "selection_lock": _binding(selection_lock),
        "checkpoint": _binding(checkpoint),
        "scaler": _binding(scaler),
        "cache_manifest": _binding(cache_dir / "manifest.json"),
    }


def no_action_adapter(
    *, proposer: Path, outer_fold: int, seed: int, output: Path
) -> dict[str, Any]:
    """Adapt a target-free test proposer directly to the frozen no-action ABI."""

    with np.load(proposer, allow_pickle=False) as archive:
        forbidden = sorted(set(archive.files) & LABEL_FIELDS)
        if forbidden:
            raise RuntimeError(f"fast-path proposer contains target fields: {forbidden}")
        required = {"cache_index", "prediction", "rr_std"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"fast-path proposer fields are missing: {missing}")
        index = np.asarray(archive["cache_index"], dtype=np.int64)
        fallback = np.asarray(archive["prediction"])
        standard_deviation = np.asarray(archive["rr_std"], dtype=np.float32)
        if fallback.dtype != np.float32:
            raise RuntimeError("fast-path proposer prediction must already be float32")
    if (
        index.ndim != 1
        or len(index) == 0
        or len(np.unique(index)) != len(index)
        or fallback.shape != index.shape
        or standard_deviation.shape != index.shape
        or not np.isfinite(fallback).all()
        or not np.isfinite(standard_deviation).all()
        or np.any(standard_deviation <= 0)
    ):
        raise RuntimeError("fast-path proposer arrays are invalid")
    rows = len(index)
    arrays = {
        "cache_index": index,
        "fallback_rr_bpm": fallback.copy(),
        "fallback_std_bpm": standard_deviation,
        "fallback_available": np.ones(rows, dtype=bool),
        # Source is a diagnostic placeholder only.  The frozen common policy
        # has zero coverage and the orchestrator verifies bit-exact fallback.
        "source_rr_bpm": fallback.copy(),
        "source_scale_bpm": standard_deviation.copy(),
        "source_available": np.ones(rows, dtype=bool),
        "selected_probability": np.zeros(rows, dtype=np.float32),
        "margin": np.zeros(rows, dtype=np.float32),
        "entropy": np.zeros(rows, dtype=np.float32),
        "quality": np.zeros(rows, dtype=np.float32),
        "valid_candidate_count": np.ones(rows, dtype=np.int16),
        "normalized_entropy": np.zeros(rows, dtype=np.float32),
        "outer_fold": np.asarray(outer_fold, dtype=np.int16),
        "seed": np.asarray(seed, dtype=np.int64),
        "target_fields_present": np.asarray(False),
        "source_is_no_action_placeholder": np.asarray(True),
    }
    _atomic_npz(output, arrays)
    return {
        "output": _binding(output),
        "rows": rows,
        "outer_fold": int(outer_fold),
        "seed": int(seed),
        "target_fields_present": False,
        "source_is_no_action_placeholder": True,
        "input_proposer": _binding(proposer),
    }


def _load_safe_proposer_cache(
    cache_dir: Path, run_config: Mapping[str, Any]
) -> tuple[Any, int]:
    """Recreate proposer inputs while never parsing a reference/QC CSV value."""

    root_manifest = _json(cache_dir / "manifest.json", "RF cache manifest")
    sessions = [
        str(item["session_id"])
        for item in root_manifest.get("sessions", [])
        if isinstance(item, Mapping) and item.get("status") == "ok"
    ]
    if not sessions:
        raise RuntimeError("RF cache has no successful sessions")
    maps_parts: list[np.ndarray] = []
    aux_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    frequency: np.ndarray | None = None
    for session_id in sessions:
        root = cache_dir / session_id
        maps = np.load(root / "maps.npy", mmap_mode="r", allow_pickle=False)
        aux = np.load(root / "aux.npy", mmap_mode="r", allow_pickle=False)
        metadata = pd.read_csv(
            root / "metadata.csv", usecols=list(SAFE_PROPOSER_METADATA_FIELDS)
        )
        grid = np.load(root / "frequencies_hz.npy", allow_pickle=False)
        if len(maps) != len(aux) or len(maps) != len(metadata):
            raise RuntimeError(f"safe proposer cache row mismatch: {session_id}")
        if frequency is None:
            frequency = grid
        elif not np.array_equal(frequency, grid):
            raise RuntimeError("safe proposer cache frequency grids differ")
        maps_parts.append(np.asarray(maps))
        aux_parts.append(np.asarray(aux))
        metadata_parts.append(metadata)
    assert frequency is not None
    maps = np.concatenate(maps_parts, axis=0)
    aux = np.concatenate(aux_parts, axis=0)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    arguments = run_config.get("arguments")
    if not isinstance(arguments, Mapping):
        raise RuntimeError("test proposer run_config arguments are missing")
    stored_bins = maps.shape[-1]
    if stored_bins % 2:
        raise RuntimeError("safe proposer RF cache is not raw/phase separable")
    branch = str(arguments.get("map_branch", "both"))
    if branch == "raw":
        maps = maps[..., : stored_bins // 2]
    elif branch == "phase":
        maps = maps[..., stored_bins // 2 :]
    elif branch != "both":
        raise RuntimeError("safe proposer checkpoint has an unsupported map branch")
    base_aux_dim = int(aux.shape[1])
    history_names: list[str] = []
    if bool(arguments.get("use_aux", False)) and bool(arguments.get("causal_history", False)):
        aux, history_names = PROPOSER_TRAIN.append_causal_history_features(aux, metadata)
    if list(run_config.get("causal_history_feature_names", [])) != history_names:
        raise RuntimeError("safe proposer causal-history schema mismatch")
    # Dataset plumbing expects bookkeeping fields.  These constants are newly
    # created and are not read from the source CSV.
    metadata = metadata.copy()
    metadata["rr_bpm"] = np.nan
    metadata["reference_valid"] = False
    metadata["reference_quality"] = 0.0
    metadata["reference_sigma_bpm"] = 1.0
    metadata["radar_observable"] = True
    cache = PROPOSER_TRAIN.FeatureCache(
        maps=maps,
        aux=aux,
        metadata=metadata,
        frequencies_hz=np.asarray(frequency),
    )
    observed_shape = {"maps": list(cache.maps.shape), "aux": list(cache.aux.shape)}
    if run_config.get("cache_shape") != observed_shape:
        raise RuntimeError("safe proposer input topology differs from run_config")
    return cache, base_aux_dim


def _masked_proposer_loader(
    loader: Any, mask_pattern: tuple[bool, bool, bool]
):
    """Apply one fixed deployment mask without changing cached availability.

    The effective mask is the logical intersection of physical availability
    and the predeclared condition.  No target/reference field is inspected.
    """

    pattern = torch.as_tensor(mask_pattern, dtype=torch.bool)
    for source in loader:
        batch = dict(source)
        observed = batch.get("radar_mask")
        if not isinstance(observed, torch.Tensor) or observed.ndim != 2 or observed.shape[1] != 3:
            raise RuntimeError("safe proposer loader radar_mask topology is invalid")
        batch["radar_mask"] = observed.bool() & pattern.to(observed.device).view(1, 3)
        yield batch


def predict_test_proposer_safe(
    *,
    cache_dir: Path,
    checkpoint: Path,
    run_config_path: Path | None,
    test_manifest: Path,
    output: Path,
    device_name: str,
    batch_size: int,
    amp: bool,
    radar_mask_name: str = "radars_123",
) -> dict[str, Any]:
    """Infer the test proposer without loading or serializing reference fields."""

    manifest = _json(test_manifest, "test proposer split manifest")
    content = dict(manifest)
    expected_content = str(content.pop("content_sha256", ""))
    observed_content = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if expected_content != observed_content:
        raise RuntimeError("test proposer split manifest content hash mismatch")
    identities = manifest.get("identities")
    if not isinstance(identities, Mapping) or not isinstance(identities.get("prediction"), list):
        raise RuntimeError("test proposer manifest prediction identities are missing")
    cache_binding = manifest.get("cache")
    fold_binding = manifest.get("fold_assignments")
    if not isinstance(cache_binding, Mapping) or not isinstance(fold_binding, Mapping):
        raise RuntimeError("test proposer manifest input bindings are missing")
    if sha256_file(cache_dir / "manifest.json") != str(cache_binding.get("manifest_sha256", "")):
        raise RuntimeError("test proposer manifest/RF cache hash mismatch")
    fold_path = Path(str(fold_binding.get("path", ""))).expanduser()
    if not fold_path.is_absolute():
        fold_path = test_manifest.parent / fold_path
    fold_path = fold_path.resolve()
    if not fold_path.is_file() or sha256_file(fold_path) != str(fold_binding.get("sha256", "")):
        raise RuntimeError("test proposer manifest/fold-assignment hash mismatch")
    run_config_path = (
        checkpoint.parent.parent / "run_config.json"
        if run_config_path is None
        else run_config_path
    )
    run_config = _json(run_config_path, "test proposer run_config")
    cache, base_aux_dim = _load_safe_proposer_cache(cache_dir, run_config)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_provenance = checkpoint_payload.get("split_authority_provenance")
    if not isinstance(checkpoint_provenance, Mapping):
        raise RuntimeError("test proposer checkpoint split provenance is missing")
    source_train = set(map(str, checkpoint_provenance.get("train_identities", ())))
    source_validation = set(
        map(str, checkpoint_provenance.get("validation_identities", ()))
    )
    source_prediction = set(
        map(str, checkpoint_provenance.get("prediction_identities", ()))
    )
    source_excluded = set(
        map(str, checkpoint_provenance.get("excluded_identities", ()))
    )
    test_identities = set(map(str, identities["prediction"]))
    if (
        not test_identities
        or not test_identities <= source_excluded
        or test_identities & (source_train | source_validation | source_prediction)
    ):
        raise RuntimeError(
            "bound validation proposer was not trained/selected disjoint from test identities"
        )
    recorded_run_authority = run_config.get("split_authority")
    if recorded_run_authority != checkpoint_provenance:
        raise RuntimeError("test proposer checkpoint/run_config split provenance mismatch")
    if checkpoint_payload.get("run_signature") != run_config.get("run_signature"):
        raise RuntimeError("test proposer checkpoint/run_config signature mismatch")
    model_kwargs = checkpoint_payload.get("model_kwargs")
    if not isinstance(model_kwargs, Mapping):
        raise RuntimeError("test proposer checkpoint model kwargs are missing")
    arguments = run_config.get("arguments", {})
    if bool(arguments.get("use_aux", False)):
        center = _as_numpy_scaler(checkpoint_payload, "aux_center")
        scale = _as_numpy_scaler(checkpoint_payload, "aux_scale")
        if center.shape != (cache.aux.shape[1],) or scale.shape != center.shape:
            raise RuntimeError("test proposer checkpoint auxiliary scaler shape mismatch")
        aux = PROPOSER_TRAIN.transform_aux(cache.aux, center, scale)
    else:
        aux = np.empty((len(cache.metadata), 0), dtype=np.float32)
    identity_values = cache.metadata["identity"].astype(str).to_numpy()
    index = np.flatnonzero(np.isin(identity_values, tuple(identities["prediction"])))
    if len(index) == 0 or set(identity_values[index]) != set(identities["prediction"]):
        raise RuntimeError("test proposer prediction identity ownership mismatch")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    loader = PROPOSER_TRAIN.make_loader(
        cache,
        aux,
        index,
        batch_size=int(batch_size),
        workers=0,
        device=device,
        seed=int(arguments.get("seed", 0)) + 99001,
        train=False,
    )
    model = PROPOSER_TRAIN.build_model(
        str(checkpoint_payload["model_type"]), dict(model_kwargs)
    ).to(device)
    model.load_state_dict(checkpoint_payload["model_state"], strict=True)
    if radar_mask_name not in RADAR_MASKS:
        raise RuntimeError(f"unknown predeclared radar mask: {radar_mask_name}")
    mask_pattern = RADAR_MASKS[radar_mask_name]
    bundle = predict_label_free(
        model,
        _masked_proposer_loader(loader, mask_pattern),
        device,
        amp=bool(amp and device.type == "cuda"),
    )
    observed_index = np.asarray(bundle.index, dtype=np.int64)
    if not np.array_equal(observed_index, index):
        raise RuntimeError("safe test proposer inference row order mismatch")
    rows = cache.metadata.iloc[index]
    state = checkpoint_payload.get("model_state")
    if not isinstance(state, Mapping) or "rr_bins" not in state:
        raise RuntimeError("test proposer checkpoint lacks RR posterior bins")
    arrays = {
        "cache_index": index,
        "session_id": rows["session_id"].astype(str).to_numpy(dtype=np.str_),
        "identity": rows["identity"].astype(str).to_numpy(dtype=np.str_),
        "protocol": rows["protocol"].astype(str).to_numpy(dtype=np.str_),
        "window_number": rows["window_number"].to_numpy(np.int32),
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
        "posterior_rr_grid_bpm": np.asarray(state["rr_bins"].detach().cpu(), dtype=np.float32),
        "fold_id": np.asarray(int(manifest["fold_id"]), dtype=np.int16),
        "checkpoint_sha256": np.asarray(sha256_file(checkpoint)),
        "split_manifest_file_sha256": np.asarray(sha256_file(test_manifest)),
        "split_manifest_content_sha256": np.asarray(expected_content),
        "strict_nested_prediction_role": np.asarray(True),
        "target_fields_present": np.asarray(False),
        "radar_mask_name": np.asarray(radar_mask_name),
        "radar_mask_pattern": np.asarray(mask_pattern, dtype=bool),
    }
    if set(arrays) & LABEL_FIELDS:
        raise RuntimeError("internal error: safe proposer output contains a target field")
    _atomic_npz(output, arrays)
    return {
        "output": _binding(output),
        "rows": len(index),
        "target_fields_read": False,
        "target_fields_present": False,
        "checkpoint": _binding(checkpoint),
        "test_manifest": _binding(test_manifest),
        "safe_metadata_usecols": list(SAFE_PROPOSER_METADATA_FIELDS),
        "base_aux_dim": base_aux_dim,
        "source_checkpoint_split_provenance": dict(checkpoint_provenance),
        "radar_mask_name": radar_mask_name,
        "radar_mask_pattern": list(mask_pattern),
    }


def bind_validation_proposer(
    *,
    source_checkpoint: Path,
    source_run_config: Path,
    source_manifest: Path,
    test_manifest: Path,
    output_checkpoint: Path,
    output_run_config: Path,
) -> dict[str, Any]:
    """Bind (without reselecting) a validation proposer disjoint from test IDs."""

    source_split = _json(source_manifest, "source validation proposer manifest")
    test_split = _json(test_manifest, "outer-test inference manifest")
    for document, label in ((source_split, "source"), (test_split, "test")):
        payload = dict(document)
        expected = str(payload.pop("content_sha256", ""))
        observed = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        if observed != expected:
            raise RuntimeError(f"{label} split manifest content hash mismatch")
    source_identities = source_split.get("identities")
    test_identities = test_split.get("identities")
    if not isinstance(source_identities, Mapping) or not isinstance(test_identities, Mapping):
        raise RuntimeError("proposer binding manifests lack identity partitions")
    outer_test = set(map(str, test_identities.get("prediction", ())))
    excluded = set(map(str, source_identities.get("excluded", ())))
    fitted = set(map(str, source_identities.get("train", ()))) | set(
        map(str, source_identities.get("validation", ()))
    )
    source_predicted = set(map(str, source_identities.get("prediction", ())))
    if not outer_test or not outer_test <= excluded or outer_test & (fitted | source_predicted):
        raise RuntimeError("source validation proposer is not identity-disjoint from outer test")
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    run_config = _json(source_run_config, "source validation proposer run_config")
    provenance = checkpoint.get("split_authority_provenance")
    if not isinstance(provenance, Mapping) or run_config.get("split_authority") != provenance:
        raise RuntimeError("source proposer checkpoint/run_config provenance mismatch")
    if provenance.get("split_manifest_file_sha256") != sha256_file(source_manifest):
        raise RuntimeError("source proposer checkpoint/manifest hash mismatch")
    if checkpoint.get("run_signature") != run_config.get("run_signature"):
        raise RuntimeError("source proposer checkpoint/run_config signature mismatch")
    if output_checkpoint.exists() or output_run_config.exists():
        raise FileExistsError("bound proposer outputs already exist without a stage receipt")
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_run_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_checkpoint, output_checkpoint)
    shutil.copyfile(source_run_config, output_run_config)
    if sha256_file(output_checkpoint) != sha256_file(source_checkpoint):
        raise RuntimeError("bound proposer checkpoint copy is not bit-exact")
    if sha256_file(output_run_config) != sha256_file(source_run_config):
        raise RuntimeError("bound proposer run_config copy is not bit-exact")
    return {
        "output_checkpoint": _binding(output_checkpoint),
        "output_run_config": _binding(output_run_config),
        "source_checkpoint": _binding(source_checkpoint),
        "source_run_config": _binding(source_run_config),
        "source_manifest": _binding(source_manifest),
        "test_manifest": _binding(test_manifest),
        "capacity_or_checkpoint_reselected": False,
        "outer_test_identity_disjoint": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    stitch = subs.add_parser("stitch")
    stitch.add_argument("--discovery-stack", type=Path, required=True)
    stitch.add_argument("--test-prediction", type=Path, required=True)
    stitch.add_argument("--test-manifest", type=Path, required=True)
    stitch.add_argument("--checkpoint", type=Path, required=True)
    stitch.add_argument("--outer-fold", type=int, required=True, choices=range(6))
    stitch.add_argument("--seed", type=int, required=True)
    stitch.add_argument("--output", type=Path, required=True)
    cache = subs.add_parser("cache")
    cache.add_argument("--rf-cache", type=Path, required=True)
    cache.add_argument("--svd-cache", type=Path, required=True)
    cache.add_argument("--proposer", type=Path, required=True)
    cache.add_argument("--template-cache-manifest", type=Path, required=True)
    cache.add_argument("--outer-fold", type=int, required=True, choices=range(6))
    cache.add_argument("--seed", type=int, required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    predict = subs.add_parser("predict")
    predict.add_argument("--cache-dir", type=Path, required=True)
    predict.add_argument("--selection-lock", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--scaler", type=Path, required=True)
    predict.add_argument("--outer-fold", type=int, required=True, choices=range(6))
    predict.add_argument("--seed", type=int, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--amp", action="store_true")
    fast = subs.add_parser("no-action-adapter")
    fast.add_argument("--proposer", type=Path, required=True)
    fast.add_argument("--outer-fold", type=int, required=True, choices=range(6))
    fast.add_argument("--seed", type=int, required=True)
    fast.add_argument("--output", type=Path, required=True)
    safe = subs.add_parser("proposer-predict")
    safe.add_argument("--cache-dir", type=Path, required=True)
    safe.add_argument("--checkpoint", type=Path, required=True)
    safe.add_argument("--run-config", type=Path)
    safe.add_argument("--test-manifest", type=Path, required=True)
    safe.add_argument("--output", type=Path, required=True)
    safe.add_argument("--device", default="cpu")
    safe.add_argument("--batch-size", type=int, default=128)
    safe.add_argument(
        "--radar-mask",
        choices=tuple(RADAR_MASKS),
        default="radars_123",
        help="fixed deployment mask; never selected from outer-test targets",
    )
    safe.add_argument("--amp", action="store_true")
    bind = subs.add_parser("bind-proposer")
    bind.add_argument("--source-checkpoint", type=Path, required=True)
    bind.add_argument("--source-run-config", type=Path, required=True)
    bind.add_argument("--source-manifest", type=Path, required=True)
    bind.add_argument("--test-manifest", type=Path, required=True)
    bind.add_argument("--output-checkpoint", type=Path, required=True)
    bind.add_argument("--output-run-config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "stitch":
        result = stitch_test_stack(
            discovery_stack=args.discovery_stack.resolve(),
            test_prediction=args.test_prediction.resolve(),
            test_manifest=args.test_manifest.resolve(),
            checkpoint=args.checkpoint.resolve(),
            outer_fold=args.outer_fold,
            seed=args.seed,
            output=args.output.resolve(),
        )
    elif args.mode == "cache":
        result = build_test_cache(
            rf_cache=args.rf_cache.resolve(),
            svd_cache=args.svd_cache.resolve(),
            proposer=args.proposer.resolve(),
            template_cache_manifest=args.template_cache_manifest.resolve(),
            outer_fold=args.outer_fold,
            seed=args.seed,
            output_dir=args.output_dir.resolve(),
        )
    elif args.mode == "predict":
        result = predict_hcs(
            cache_dir=args.cache_dir.resolve(),
            selection_lock=args.selection_lock.resolve(),
            checkpoint=args.checkpoint.resolve(),
            scaler=args.scaler.resolve(),
            outer_fold=args.outer_fold,
            seed=args.seed,
            output=args.output.resolve(),
            device_name=args.device,
            amp=args.amp,
        )
    elif args.mode == "no-action-adapter":
        result = no_action_adapter(
            proposer=args.proposer.resolve(),
            outer_fold=args.outer_fold,
            seed=args.seed,
            output=args.output.resolve(),
        )
    elif args.mode == "proposer-predict":
        result = predict_test_proposer_safe(
            cache_dir=args.cache_dir.resolve(),
            checkpoint=args.checkpoint.resolve(),
            run_config_path=(args.run_config.resolve() if args.run_config else None),
            test_manifest=args.test_manifest.resolve(),
            output=args.output.resolve(),
            device_name=args.device,
            batch_size=args.batch_size,
            amp=args.amp,
            radar_mask_name=args.radar_mask,
        )
    else:
        result = bind_validation_proposer(
            source_checkpoint=args.source_checkpoint.resolve(),
            source_run_config=args.source_run_config.resolve(),
            source_manifest=args.source_manifest.resolve(),
            test_manifest=args.test_manifest.resolve(),
            output_checkpoint=args.output_checkpoint.resolve(),
            output_run_config=args.output_run_config.resolve(),
        )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
