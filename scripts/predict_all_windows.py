#!/usr/bin/env python3
"""Identity-disjoint inference for every cached radar window.

The training entry point deliberately evaluates only windows with a valid
reference label.  Deployment and causal sequence experiments, however, need
predictions for the complete recording.  This script reuses the frozen outer
fold checkpoints and assigns *all* windows from an identity to that identity's
held-out test fold.  Reference/QC fields are carried only as output masks; the
model is called with exactly ``(map, radar_mask, aux)``.

The default command is intentionally bound to the locked alias-gated run::

    python scripts/predict_all_windows.py --device cuda --amp

Intermediate fold files are atomically written.  ``--resume`` verifies and
reuses those files after an interruption; ``--reuse`` returns an already
complete result only when its full inference signature still matches.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import the exact training/cache interfaces used to create the checkpoints.
# Keeping this list explicit also makes interface drift fail loudly in tests.
from scripts.train import (  # noqa: E402
    CachedRadarDataset,
    FeatureCache,
    PredictionBundle,
    append_mask_aware_causal_history_features,
    build_model,
    concatenate_bundles,
    fit_aux_scaler,
    infer_auxiliary_layout,
    load_feature_cache,
    make_loader,
    predict,
    transform_aux,
)
from snn_rr.data import build_dataset_manifest  # noqa: E402


DEFAULT_CACHE = PROJECT_ROOT / "artifacts/cache/rf32s"
DEFAULT_RUN = PROJECT_ROOT / "artifacts/runs/final_alias_gate_s12_deterministic"
DEFAULT_OUTPUT = DEFAULT_RUN / "all_windows"
FORMAT_VERSION = 2
BASE_OOF_AUTHORITY_SCHEMA = "snn_rr.base_oof_authority.v1"

# Per-fold restart artifacts are deployment-output caches, not training/eval
# bundles.  In particular, target and observable/QC values are never persisted.
FOLD_DEPLOYMENT_FIELDS = (
    "index",
    "prediction",
    "rr_std",
    "uncertainty",
    "quality",
    "reference_valid",
    "spike_rate",
    "radar_weights",
    "map_prediction",
    "posterior_entropy",
    "topk_rr",
    "topk_probability",
    "posterior_probability",
    "alias_probability",
)

_FORBIDDEN_FORWARD_NAMES = {
    "target",
    "targets",
    "rr",
    "rr_bpm",
    "reference",
    "reference_valid",
    "reference_quality",
    "reference_sigma",
    "observable",
    "radar_observable",
    "qc",
    "label",
    "labels",
    "classical_error",
}


@dataclass(frozen=True, slots=True)
class FoldBinding:
    fold: int
    checkpoint_path: Path
    checkpoint_sha256: str
    test_identities: tuple[str, ...]
    expected_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class FrozenTolerance:
    prediction_bpm: float
    map_prediction_bpm: float
    rr_std_bpm: float
    uncertainty: float
    quality: float
    alias_probability: float
    topk_probability: float
    posterior_probability: float
    posterior_entropy: float
    spike_rate: float
    radar_weight: float
    topk_rr_bpm: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_pipeline(paths: Sequence[Path]) -> str:
    """Match ``build_features.py``'s path-and-content pipeline digest."""

    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(resolved.read_bytes())
    return digest.hexdigest()


def _runtime_source_hashes() -> dict[str, str]:
    """Hash every local Python source participating in inference."""

    paths = (
        Path(__file__),
        PROJECT_ROOT / "scripts/train.py",
        PROJECT_ROOT / "scripts/build_features.py",
        PROJECT_ROOT / "src/snn_rr/cache.py",
        PROJECT_ROOT / "src/snn_rr/models.py",
        PROJECT_ROOT / "src/snn_rr/metrics.py",
        PROJECT_ROOT / "src/snn_rr/data.py",
        PROJECT_ROOT / "src/snn_rr/preprocess.py",
        PROJECT_ROOT / "src/snn_rr/acquisition_contract.py",
        PROJECT_ROOT / "src/snn_rr/acquisition_protocol.py",
        PROJECT_ROOT / "src/snn_rr/synchronization.py",
        PROJECT_ROOT / "src/snn_rr/radar_timing.py",
        PROJECT_ROOT / "src/snn_rr/range_tracking.py",
    )
    return {
        str(path.resolve().relative_to(PROJECT_ROOT)): _sha256_file(path.resolve())
        for path in paths
    }


def _cache_session_content_binding(cache_dir: Path, session_id: str) -> dict[str, Any]:
    """Bind actual tensor/metadata bytes absent from the legacy manifest."""

    session_dir = cache_dir / session_id
    names = [
        "maps.npy",
        "aux.npy",
        "metadata.csv",
        "frequencies_hz.npy",
        "manifest.json",
    ]
    for optional_name in ("radar_timing_valid_mask.npy", "range_aux.npy"):
        if (session_dir / optional_name).is_file():
            names.append(optional_name)
    files: dict[str, dict[str, Any]] = {}
    for name in names:
        path = session_dir / name
        if not path.is_file():
            raise RuntimeError(f"cache session content file is missing: {path}")
        files[name] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    signature_payload = {
        "session_id": session_id,
        "files": files,
    }
    encoded = json.dumps(
        signature_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        **signature_payload,
        "content_signature_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _cache_inventory_sha256(
    cache_dir: Path, session_ids: Sequence[str]
) -> tuple[str, int]:
    paths = [cache_dir / "manifest.json"]
    for session_id in session_ids:
        session_dir = cache_dir / str(session_id)
        paths.extend(
            session_dir / name
            for name in (
                "manifest.json",
                "maps.npy",
                "aux.npy",
                "metadata.csv",
                "frequencies_hz.npy",
            )
        )
        for optional_name in ("radar_timing_valid_mask.npy", "range_aux.npy"):
            optional_path = session_dir / optional_name
            if optional_path.is_file():
                paths.append(optional_path)
    inventory: list[dict[str, Any]] = []
    for path in sorted(set(paths), key=lambda item: str(item.relative_to(cache_dir))):
        if not path.is_file():
            raise RuntimeError(f"cache inventory file is missing: {path}")
        inventory.append(
            {
                "path": str(path.relative_to(cache_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return _canonical_sha256(inventory), len(inventory)


def _verify_acquisition_raw_sha_bindings(
    *,
    cache_dir: Path,
    cache_root_manifest: Mapping[str, Any],
    session_ids: Sequence[str],
) -> tuple[str, int]:
    """Rehash exact raw payloads bound by the v2 reconstruction graph."""

    contract = cache_root_manifest.get("acquisition_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("acquisition-v2 cache contract is missing")
    reconstruction_value = contract.get("reconstruction_manifest")
    if not isinstance(reconstruction_value, str) or not reconstruction_value:
        raise RuntimeError("acquisition-v2 reconstruction path is missing")
    reconstruction_path = Path(reconstruction_value)
    if not reconstruction_path.is_absolute():
        reconstruction_path = cache_dir / reconstruction_path
    reconstruction_path = reconstruction_path.resolve()
    reconstruction = _load_json(reconstruction_path)
    if reconstruction.get("content_sha256") != contract.get(
        "reconstruction_content_sha256"
    ) or reconstruction.get("content_sha256") != _canonical_content_sha256(
        reconstruction
    ):
        raise RuntimeError("acquisition reconstruction content binding mismatch")
    entries = reconstruction.get("sessions")
    if not isinstance(entries, list):
        raise RuntimeError("acquisition reconstruction session catalogue is missing")
    by_id = {
        str(entry.get("session_id")): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    dataset_root = _resolve_recorded_path(cache_root_manifest.get("dataset_root"))
    observed: list[dict[str, Any]] = []
    for session_id in session_ids:
        entry = by_id.get(str(session_id))
        if not isinstance(entry, Mapping):
            raise RuntimeError(
                f"acquisition reconstruction is missing raw authority for {session_id}"
            )
        manifest_value = entry.get("manifest")
        if not isinstance(manifest_value, str) or not manifest_value:
            raise RuntimeError(f"acquisition session manifest path is missing: {session_id}")
        session_manifest_path = Path(manifest_value)
        if not session_manifest_path.is_absolute():
            session_manifest_path = reconstruction_path.parent / session_manifest_path
        session_manifest_path = session_manifest_path.resolve()
        session_manifest = _load_json(session_manifest_path)
        if (
            _sha256_file(session_manifest_path) != entry.get("manifest_sha256")
            or session_manifest.get("content_sha256") != entry.get("content_sha256")
            or session_manifest.get("content_sha256")
            != _canonical_content_sha256(session_manifest)
        ):
            raise RuntimeError(
                f"acquisition session manifest binding mismatch: {session_id}"
            )
        bindings = session_manifest.get("raw_input_bindings")
        if not isinstance(bindings, Mapping) or not bindings:
            raise RuntimeError(f"raw SHA-256 bindings are missing: {session_id}")
        if session_manifest.get("raw_input_bindings_sha256") != _canonical_sha256(
            bindings
        ):
            raise RuntimeError(f"raw binding catalogue hash mismatch: {session_id}")
        for name, binding in sorted(bindings.items()):
            if not isinstance(binding, Mapping):
                raise RuntimeError(f"raw binding is malformed: {session_id}/{name}")
            relative = binding.get("path")
            if not isinstance(relative, str) or not relative:
                raise RuntimeError(f"raw binding path is missing: {session_id}/{name}")
            raw_path = (dataset_root / relative).resolve()
            try:
                raw_path.relative_to(dataset_root)
            except ValueError as error:
                raise RuntimeError(
                    f"raw binding escapes dataset root: {session_id}/{name}"
                ) from error
            if not raw_path.is_file():
                raise RuntimeError(f"raw bound input is missing: {raw_path}")
            size = raw_path.stat().st_size
            digest = _sha256_file(raw_path)
            if size != binding.get("bytes") or digest != binding.get("sha256"):
                raise RuntimeError(
                    f"raw bound input content changed: {session_id}/{name}"
                )
            observed.append(
                {
                    "session_id": str(session_id),
                    "name": str(name),
                    "path": str(relative),
                    "bytes": size,
                    "sha256": digest,
                }
            )
    return _canonical_sha256(observed), len(observed)


def _canonical_signature(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_content_sha256(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    return _canonical_sha256(payload)


def _row_fold_binding_sha256(
    cache_index: Sequence[Any], identity: Sequence[Any], fold: Sequence[Any]
) -> str:
    indices = np.asarray(cache_index, dtype=np.int64)
    identities = np.asarray(identity).astype(str)
    folds = np.asarray(fold, dtype=np.int64)
    if not (indices.ndim == identities.ndim == folds.ndim == 1) or not (
        len(indices) == len(identities) == len(folds)
    ):
        raise RuntimeError("all-window row/fold binding arrays are inconsistent")
    return _canonical_sha256(
        [
            {
                "cache_index": int(index),
                "identity": str(person),
                "fold": int(fold_id),
            }
            for index, person, fold_id in zip(
                indices, identities, folds, strict=True
            )
        ]
    )


def _strict_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _strict_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            _strict_json(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _resolve_recorded_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _source_fingerprint(subject: Any) -> str:
    """Recompute the raw-file stat fingerprint recorded by feature building."""

    if not subject.usable:
        return "missing"
    paths = [Path(subject.biopac_path)]
    for radar_id in (1, 2, 3):
        stream = subject.selected_session.radars[radar_id]
        paths.extend(map(Path, stream.data_paths))
        if stream.meta_path is not None:
            paths.append(Path(stream.meta_path))
    records: list[dict[str, Any]] = []
    for path in sorted(set(paths), key=lambda item: str(item.resolve())):
        stat = path.stat()
        records.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_cache_sources(
    cache_dir: Path,
    metadata: pd.DataFrame,
    *,
    verify_raw_sources: bool = True,
    verified_cache_provenance: Any = None,
) -> dict[str, Any]:
    """Validate cache manifests, current pipeline/config, and raw bindings."""

    cache_dir = cache_dir.resolve()
    manifest_path = cache_dir / "manifest.json"
    root = _load_json(manifest_path)
    root_contract = root.get("acquisition_contract")
    acquisition_v2 = bool(
        isinstance(root_contract, Mapping)
        and root_contract.get("schema_version")
        == "snn_rr.feature_cache_acquisition.v2"
    )
    sessions = root.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise RuntimeError("cache manifest contains no sessions")
    ok = [item for item in sessions if item.get("status") == "ok"]
    # The canonical catalogue intentionally retains unusable/missing subjects
    # (S24 in the current cohort).  They must contribute no cache rows, while
    # every status=ok session must be represented exactly once below.
    unavailable = [
        str(item.get("session_id"))
        for item in sessions
        if item.get("status") != "ok"
    ]
    session_ids = [str(item.get("session_id")) for item in ok]
    if len(session_ids) != len(set(session_ids)):
        raise RuntimeError("cache manifest contains duplicate session IDs")
    observed_session_ids = metadata["session_id"].astype(str).drop_duplicates().tolist()
    if observed_session_ids != session_ids:
        raise RuntimeError("cache metadata session order does not match root manifest")
    if sum(int(item.get("window_count", -1)) for item in ok) != len(metadata):
        raise RuntimeError("cache manifest window counts do not match loaded rows")

    config_path = _resolve_recorded_path(root.get("config"))
    expected_config_hash = str(root.get("config_sha256", ""))
    if not config_path.is_file() or _sha256_file(config_path) != expected_config_hash:
        raise RuntimeError("cache configuration SHA-256 no longer matches its manifest")
    pipeline_paths = [
        PROJECT_ROOT / "scripts/build_features.py",
        PROJECT_ROOT / "src/snn_rr/data.py",
        PROJECT_ROOT / "src/snn_rr/preprocess.py",
    ]
    if acquisition_v2:
        pipeline_paths.extend(
            [
                PROJECT_ROOT / "src/snn_rr/acquisition_contract.py",
                PROJECT_ROOT / "src/snn_rr/synchronization.py",
                PROJECT_ROOT / "src/snn_rr/radar_timing.py",
                PROJECT_ROOT / "src/snn_rr/range_tracking.py",
            ]
        )
    expected_pipeline_hash = str(root.get("pipeline_sha256", ""))
    if _sha256_pipeline(pipeline_paths) != expected_pipeline_hash:
        raise RuntimeError("cache feature-pipeline SHA-256 no longer matches its manifest")

    canonical_cache_provenance: dict[str, Any] | None = None
    if verified_cache_provenance is not None:
        to_dict = getattr(verified_cache_provenance, "to_dict", None)
        content_hash = getattr(verified_cache_provenance, "content_sha256", None)
        if not callable(to_dict) or not isinstance(content_hash, str):
            raise RuntimeError("cache loader returned malformed provenance authority")
        canonical_cache_provenance = dict(to_dict())
        canonical_cache_provenance["content_sha256"] = content_hash
        if _canonical_content_sha256(canonical_cache_provenance) != content_hash:
            raise RuntimeError("cache loader provenance canonical hash mismatch")
    if acquisition_v2:
        if canonical_cache_provenance is None:
            raise RuntimeError(
                "acquisition-v2 all-window inference requires verified cache provenance"
            )
        if (
            canonical_cache_provenance.get("root_manifest_path")
            != str(manifest_path.resolve())
            or canonical_cache_provenance.get("root_manifest_sha256")
            != _sha256_file(manifest_path)
            or canonical_cache_provenance.get("root_manifest_content_sha256")
            != root.get("content_sha256")
            or canonical_cache_provenance.get("acquisition_schema_version")
            != "snn_rr.feature_cache_acquisition.v2"
            or canonical_cache_provenance.get("config_sha256")
            != expected_config_hash
            or canonical_cache_provenance.get("pipeline_sha256")
            != expected_pipeline_hash
            or canonical_cache_provenance.get("reconstruction_content_sha256")
            != root_contract.get("reconstruction_content_sha256")
            or canonical_cache_provenance.get("selected_sessions") != session_ids
        ):
            raise RuntimeError(
                "verified cache provenance does not bind the loaded acquisition-v2 cache"
            )
        current_inventory_sha256, current_inventory_count = _cache_inventory_sha256(
            cache_dir, session_ids
        )
        if (
            current_inventory_sha256
            != canonical_cache_provenance.get("inventory_sha256")
            or current_inventory_count
            != canonical_cache_provenance.get("inventory_file_count")
        ):
            raise RuntimeError(
                "cache files changed between feature loading and provenance validation"
            )

    source_by_session: dict[str, str] = {}
    content_by_session: dict[str, dict[str, Any]] = {}
    for item in ok:
        session_id = str(item["session_id"])
        local = _load_json(cache_dir / session_id / "manifest.json")
        for key in (
            "config_sha256",
            "pipeline_sha256",
            "source_fingerprint",
            "window_count",
            "map_shape",
            "aux_shape",
        ):
            if local.get(key) != item.get(key):
                raise RuntimeError(
                    f"root/local cache manifest mismatch for {session_id}: {key}"
                )
        if local.get("config_sha256") != expected_config_hash:
            raise RuntimeError(f"session {session_id} has a different config hash")
        if local.get("pipeline_sha256") != expected_pipeline_hash:
            raise RuntimeError(f"session {session_id} has a different pipeline hash")
        if int((metadata["session_id"].astype(str) == session_id).sum()) != int(
            local["window_count"]
        ):
            raise RuntimeError(f"metadata row count mismatch for {session_id}")
        fingerprint = str(local.get("source_fingerprint", ""))
        if len(fingerprint) != 64:
            raise RuntimeError(f"invalid source fingerprint for {session_id}")
        source_by_session[session_id] = fingerprint
        content_by_session[session_id] = _cache_session_content_binding(
            cache_dir, session_id
        )

    raw_verified = False
    raw_sha256_verified = False
    raw_sha256_binding_digest: str | None = None
    raw_sha256_binding_count = 0
    dataset_root = _resolve_recorded_path(root.get("dataset_root"))
    if verify_raw_sources:
        raw_manifest = build_dataset_manifest(
            dataset_root, expected_subject_numbers=None
        )
        raw_by_id = {subject.subject_id: subject for subject in raw_manifest.subjects}
        missing = sorted(set(session_ids) - set(raw_by_id))
        if missing:
            raise RuntimeError(f"raw dataset is missing cached sessions: {missing}")
        for session_id in session_ids:
            actual = _source_fingerprint(raw_by_id[session_id])
            if actual != source_by_session[session_id]:
                raise RuntimeError(
                    f"raw source fingerprint changed for cached session {session_id}"
                )
        raw_verified = True
        if acquisition_v2:
            (
                raw_sha256_binding_digest,
                raw_sha256_binding_count,
            ) = _verify_acquisition_raw_sha_bindings(
                cache_dir=cache_dir,
                cache_root_manifest=root,
                session_ids=session_ids,
            )
            raw_sha256_verified = True

    ordered_content_signatures = [
        content_by_session[session_id]["content_signature_sha256"]
        for session_id in session_ids
    ]
    cache_content_signature = hashlib.sha256(
        json.dumps(
            ordered_content_signatures, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "cache_dir": str(cache_dir),
        "cache_manifest_path": str(manifest_path),
        "cache_manifest_sha256": _sha256_file(manifest_path),
        "config_path": str(config_path),
        "config_sha256": expected_config_hash,
        "pipeline_sha256": expected_pipeline_hash,
        "dataset_root": str(dataset_root),
        "raw_source_fingerprints_verified": raw_verified,
        "raw_input_sha256_bindings_verified": raw_sha256_verified,
        "raw_input_sha256_binding_digest": raw_sha256_binding_digest,
        "raw_input_sha256_binding_count": raw_sha256_binding_count,
        "source_fingerprint_by_session": source_by_session,
        "session_content_binding": content_by_session,
        "cache_content_signature_sha256": cache_content_signature,
        "row_count": len(metadata),
        "session_count": len(session_ids),
        "unavailable_catalogue_sessions": unavailable,
        "canonical_cache_provenance": canonical_cache_provenance,
    }


def _assert_publication_sources_current(
    cache_source: Mapping[str, Any], runtime_source_sha256: Mapping[str, str]
) -> None:
    """Publication barrier for the exact cache/source generation used in RAM."""

    cache_dir = Path(str(cache_source["cache_dir"])).resolve()
    manifest_path = Path(str(cache_source["cache_manifest_path"])).resolve()
    if _sha256_file(manifest_path) != cache_source.get("cache_manifest_sha256"):
        raise RuntimeError("cache root manifest changed during all-window inference")
    canonical = cache_source.get("canonical_cache_provenance")
    if isinstance(canonical, Mapping):
        session_ids = canonical.get("selected_sessions")
        if not isinstance(session_ids, list):
            raise RuntimeError("canonical cache selected-session authority is malformed")
        inventory_sha256, inventory_count = _cache_inventory_sha256(
            cache_dir, list(map(str, session_ids))
        )
        if (
            inventory_sha256 != canonical.get("inventory_sha256")
            or inventory_count != canonical.get("inventory_file_count")
        ):
            raise RuntimeError("cache inventory changed during all-window inference")
        if cache_source.get("raw_input_sha256_bindings_verified") is True:
            raw_digest, raw_count = _verify_acquisition_raw_sha_bindings(
                cache_dir=cache_dir,
                cache_root_manifest=_load_json(manifest_path),
                session_ids=list(map(str, session_ids)),
            )
            if (
                raw_digest != cache_source.get("raw_input_sha256_binding_digest")
                or raw_count != cache_source.get("raw_input_sha256_binding_count")
            ):
                raise RuntimeError("raw input SHA-256 graph changed during inference")
    config_path = Path(str(cache_source["config_path"])).resolve()
    if _sha256_file(config_path) != cache_source.get("config_sha256"):
        raise RuntimeError("cache configuration changed during all-window inference")
    pipeline_paths = [
        PROJECT_ROOT / "scripts/build_features.py",
        PROJECT_ROOT / "src/snn_rr/data.py",
        PROJECT_ROOT / "src/snn_rr/preprocess.py",
    ]
    if isinstance(canonical, Mapping) and canonical.get(
        "acquisition_schema_version"
    ) == "snn_rr.feature_cache_acquisition.v2":
        pipeline_paths.extend(
            [
                PROJECT_ROOT / "src/snn_rr/acquisition_contract.py",
                PROJECT_ROOT / "src/snn_rr/synchronization.py",
                PROJECT_ROOT / "src/snn_rr/radar_timing.py",
                PROJECT_ROOT / "src/snn_rr/range_tracking.py",
            ]
        )
    if _sha256_pipeline(pipeline_paths) != cache_source.get("pipeline_sha256"):
        raise RuntimeError("cache pipeline changed during all-window inference")
    for relative, expected_hash in runtime_source_sha256.items():
        source_path = PROJECT_ROOT / str(relative)
        if not source_path.is_file() or _sha256_file(source_path) != expected_hash:
            raise RuntimeError(f"runtime inference source changed: {relative}")


def prepare_cache(
    cache_dir: Path, run_config: Mapping[str, Any]
) -> tuple[FeatureCache, int, list[str]]:
    """Load the canonical cache and reproduce training-time feature topology."""

    cache = load_feature_cache(cache_dir, mmap=False)
    base_aux_dim = int(cache.aux.shape[1])
    arguments = run_config.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise RuntimeError("run_config arguments are missing")
    if not bool(arguments.get("use_aux", False)):
        raise RuntimeError("the frozen run is not bound to auxiliary features")
    history_names: list[str] = []
    if bool(arguments.get("causal_history", False)):
        augmented, history_names = append_mask_aware_causal_history_features(cache)
        cache = FeatureCache(
            maps=cache.maps,
            aux=augmented,
            metadata=cache.metadata,
            frequencies_hz=cache.frequencies_hz,
            provenance=cache.provenance,
            radar_timing_valid_mask=cache.radar_timing_valid_mask,
        )
    recorded_shape = run_config.get("cache_shape")
    actual_shape = {"maps": list(cache.maps.shape), "aux": list(cache.aux.shape)}
    if recorded_shape != actual_shape:
        raise RuntimeError(
            f"cache shape is not checkpoint-run compatible: {actual_shape} != {recorded_shape}"
        )
    if run_config.get("causal_history_feature_names", []) != history_names:
        raise RuntimeError("causal-history schema differs from the frozen training run")
    return cache, base_aux_dim, history_names


def _as_numpy_scaler(checkpoint: Mapping[str, Any], key: str) -> np.ndarray:
    value = checkpoint.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"checkpoint {key} is missing or is not a tensor")
    result = value.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise RuntimeError(f"checkpoint {key} must be a finite vector")
    return result


def validate_model_kwargs(
    checkpoint: Mapping[str, Any],
    cache: FeatureCache,
    *,
    base_aux_dim: int,
    run_config: Mapping[str, Any],
) -> None:
    kwargs = checkpoint.get("model_kwargs")
    if not isinstance(kwargs, Mapping):
        raise RuntimeError("checkpoint model_kwargs are missing")
    arguments = run_config["arguments"]
    required_equal = {
        "num_radars": int(cache.maps.shape[1]),
        "aux_dim": int(cache.aux.shape[1]),
        "input_branches": int(arguments["input_branches"]),
    }
    for key, expected in required_equal.items():
        if kwargs.get(key) != expected:
            raise RuntimeError(
                f"checkpoint model_kwargs[{key!r}]={kwargs.get(key)!r}, expected {expected!r}"
            )
    if int(cache.maps.shape[-1]) % int(kwargs["input_branches"]):
        raise RuntimeError("map range dimension is incompatible with input branches")
    if bool(kwargs.get("structured_auxiliary")) or bool(
        kwargs.get("harmonic_auxiliary")
    ):
        if kwargs.get("aux_base_dim") != base_aux_dim:
            raise RuntimeError("checkpoint aux_base_dim is incompatible with cache")

    rr_min, rr_max = map(float, arguments["rr_range"])
    rr_width = float(arguments["rr_bin_width"])
    num_bins = int(round((rr_max - rr_min) / rr_width)) + 1
    for key, expected in (
        ("rr_min", rr_min),
        ("rr_max", rr_max),
        ("num_rr_bins", num_bins),
        ("input_frequency_min_hz", float(cache.frequencies_hz[0])),
        ("input_frequency_max_hz", float(cache.frequencies_hz[-1])),
    ):
        actual = kwargs.get(key)
        if isinstance(expected, float):
            matches = actual is not None and math.isclose(
                float(actual), expected, rel_tol=1e-7, abs_tol=1e-8
            )
        else:
            matches = actual == expected
        if not matches:
            raise RuntimeError(f"checkpoint RR/frequency context mismatch: {key}")

    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping) or "rr_bins" not in state:
        raise RuntimeError("checkpoint state has no RR grid")
    expected_grid = np.linspace(rr_min, rr_max, num_bins, dtype=np.float32)
    actual_grid = np.asarray(state["rr_bins"].detach().cpu(), dtype=np.float32)
    if actual_grid.shape != expected_grid.shape or not np.allclose(
        actual_grid, expected_grid, rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError("checkpoint state RR grid is incompatible")


def validate_fold_checkpoint(
    checkpoint_path: Path,
    *,
    fold: int,
    cache: FeatureCache,
    base_aux_dim: int,
    run_config: Mapping[str, Any],
    run_dir: Path,
) -> tuple[FoldBinding, dict[str, Any]]:
    """Validate split, scaler, architecture and state/cache binding for a fold."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_signature = str(run_config.get("run_signature", ""))
    if checkpoint.get("format_version") not in {1, 2}:
        raise RuntimeError(f"unsupported checkpoint format: {checkpoint_path}")
    if checkpoint.get("model_type") != "snn":
        raise RuntimeError(f"checkpoint is not an SNN: {checkpoint_path}")
    if int(checkpoint.get("fold", -1)) != fold:
        raise RuntimeError(f"checkpoint fold mismatch: {checkpoint_path}")
    if checkpoint.get("run_signature") != run_signature:
        raise RuntimeError(f"checkpoint run signature mismatch: {checkpoint_path}")
    run_arguments = run_config.get("arguments")
    if not isinstance(run_arguments, Mapping):
        raise RuntimeError("source run arguments are missing")
    run_cache_provenance = run_config.get("cache_provenance")
    checkpoint_cache_provenance = checkpoint.get("cache_provenance")
    if run_arguments.get("cache_trust_mode") == "scientific":
        if (
            not isinstance(run_cache_provenance, Mapping)
            or run_cache_provenance.get("classification")
            != "acquisition_scientific"
            or run_cache_provenance.get("scientific_eligible") is not True
            or checkpoint_cache_provenance != run_cache_provenance
        ):
            raise RuntimeError(
                f"scientific checkpoint/cache authority mismatch: {checkpoint_path}"
            )
    elif (
        checkpoint_cache_provenance is not None
        and run_cache_provenance is not None
        and checkpoint_cache_provenance != run_cache_provenance
    ):
        raise RuntimeError(f"checkpoint/run cache provenance mismatch: {checkpoint_path}")
    validate_model_kwargs(
        checkpoint, cache, base_aux_dim=base_aux_dim, run_config=run_config
    )

    split = checkpoint.get("split")
    if not isinstance(split, Mapping):
        raise RuntimeError(f"checkpoint split is missing: {checkpoint_path}")
    normalized: dict[str, tuple[str, ...]] = {}
    for key in ("train_identities", "validation_identities", "test_identities"):
        values = split.get(key)
        if not isinstance(values, (list, tuple)) or not values:
            raise RuntimeError(f"checkpoint split {key} is invalid")
        normalized[key] = tuple(sorted(str(value) for value in values))
    sets = [set(normalized[key]) for key in normalized]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError(f"identity leakage in fold {fold} checkpoint split")
    cache_identities = set(cache.metadata["identity"].astype(str))
    if set.union(*sets) != cache_identities:
        raise RuntimeError(f"fold {fold} split does not cover cache identities")

    split_path = run_dir / f"fold_{fold}" / "split.json"
    recorded_split = _load_json(split_path)
    for key, values in normalized.items():
        if tuple(sorted(map(str, recorded_split.get(key, ())))) != values:
            raise RuntimeError(f"checkpoint/split.json mismatch for fold {fold}: {key}")

    identity = cache.metadata["identity"].astype(str).to_numpy()
    reference_valid = cache.metadata["reference_valid"].to_numpy(dtype=bool)
    train_mask = np.isin(identity, normalized["train_identities"])
    if not bool(run_config["arguments"].get("include_invalid", False)):
        train_mask &= reference_valid
    train_indices = np.flatnonzero(train_mask)
    expected_center, expected_scale = fit_aux_scaler(cache.aux, train_indices)
    center = _as_numpy_scaler(checkpoint, "aux_center")
    scale = _as_numpy_scaler(checkpoint, "aux_scale")
    if center.shape != (cache.aux.shape[1],) or scale.shape != center.shape:
        raise RuntimeError(f"checkpoint scaler dimension mismatch in fold {fold}")
    if np.any(scale <= 0):
        raise RuntimeError(f"checkpoint aux_scale is not positive in fold {fold}")
    if not np.allclose(center, expected_center, rtol=1e-6, atol=1e-7):
        raise RuntimeError(f"checkpoint aux_center is not train-split fitted in fold {fold}")
    if not np.allclose(scale, expected_scale, rtol=1e-6, atol=1e-7):
        raise RuntimeError(f"checkpoint aux_scale is not train-split fitted in fold {fold}")

    expected_indices = np.flatnonzero(
        np.isin(identity, normalized["test_identities"])
    )
    binding = FoldBinding(
        fold=fold,
        checkpoint_path=checkpoint_path.resolve(),
        checkpoint_sha256=_sha256_file(checkpoint_path),
        test_identities=normalized["test_identities"],
        expected_indices=expected_indices,
    )
    return binding, checkpoint


def validate_identity_partition(
    bindings: Sequence[FoldBinding], metadata: pd.DataFrame
) -> np.ndarray:
    """Return per-row folds after proving identity and row exact-cover."""

    owner: dict[str, int] = {}
    for binding in bindings:
        for identity in binding.test_identities:
            if identity in owner:
                raise RuntimeError(
                    f"identity {identity} belongs to folds {owner[identity]} and {binding.fold}"
                )
            owner[identity] = binding.fold
    identities = metadata["identity"].astype(str).to_numpy()
    missing_identities = sorted(set(identities) - set(owner))
    extra_identities = sorted(set(owner) - set(identities))
    if missing_identities or extra_identities:
        raise RuntimeError(
            f"identity partition mismatch; missing={missing_identities}, extra={extra_identities}"
        )
    fold_for_row = np.asarray([owner[value] for value in identities], dtype=np.int16)
    covered = np.concatenate([binding.expected_indices for binding in bindings])
    if len(covered) != len(metadata):
        raise RuntimeError("fold test identities do not cover every cache row")
    if len(np.unique(covered)) != len(covered):
        raise RuntimeError("fold test identities cover at least one cache row twice")
    if not np.array_equal(np.sort(covered), np.arange(len(metadata))):
        raise RuntimeError("fold test-identity row coverage has omissions")
    for binding in bindings:
        if not np.all(fold_for_row[binding.expected_indices] == binding.fold):
            raise RuntimeError(f"row ownership mismatch for fold {binding.fold}")
    return fold_for_row


def validate_label_free_forward_interface(model: nn.Module) -> None:
    """Reject any model forward signature capable of accepting labels/QC."""

    signature = inspect.signature(type(model).forward)
    parameters = [
        parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
    ]
    names = {parameter.name.lower() for parameter in parameters}
    forbidden = sorted(names & _FORBIDDEN_FORWARD_NAMES)
    if forbidden:
        raise RuntimeError(f"model forward exposes forbidden label/QC inputs: {forbidden}")
    if any(
        parameter.kind
        in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for parameter in parameters
    ):
        raise RuntimeError("model forward may not expose variadic inputs")
    if [parameter.name for parameter in parameters] != ["x", "radar_mask", "aux"]:
        raise RuntimeError(
            "model forward must accept exactly (x, radar_mask, aux), never target/QC"
        )


def predict_label_free(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    *,
    amp: bool,
) -> PredictionBundle:
    """Call the shared predictor while auditing every root-model invocation."""

    validate_label_free_forward_interface(model)
    calls = 0

    def audit_hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if kwargs or len(args) != 3:
            raise RuntimeError("inference attempted to pass target/QC data into the model")

    handle = model.register_forward_pre_hook(audit_hook, with_kwargs=True)
    try:
        result = predict(model, loader, device, amp=amp)
    finally:
        handle.remove()
    if calls == 0:
        raise RuntimeError("model forward was never called")
    return result


def _tolerance_for(device: torch.device, amp: bool, profile: str) -> FrozenTolerance:
    if profile == "auto":
        profile = "strict" if device.type == "cuda" and amp else "cpu"
    if profile == "strict":
        return FrozenTolerance(
            0.03, 0.25, 0.03, 0.08, 0.008, 0.002, 0.008, 0.006,
            0.08, 0.02, 0.004, 0.50
        )
    if profile == "cpu":
        # Frozen files were produced with CUDA float16 autocast.  CPU float32
        # changes near-tied discrete MAP/top-k bins more than expectations.
        # MAP/top-k RR can jump between distant, nearly tied modes even when
        # the complete posterior differs by <0.02 and its expectation by less
        # than one 0.25-bpm bin.  Their broad CPU allowances therefore do not
        # constitute the binding check: expectation and full-posterior limits
        # above do.  The resolved limits are recorded in provenance.
        return FrozenTolerance(
            0.40, 39.0, 0.25, 20.0, 0.10, 0.005, 0.03, 0.02,
            0.25, 0.06, 0.012, 39.0
        )
    raise ValueError(f"unknown tolerance profile: {profile}")


def _finite_max_abs(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    finite = difference[np.isfinite(difference)]
    return float(finite.max()) if len(finite) else 0.0


def verify_frozen_valid_predictions(
    bundle: PredictionBundle,
    fold: int,
    frozen: Mapping[str, np.ndarray],
    tolerance: FrozenTolerance,
) -> dict[str, Any]:
    """Prove valid rows reproduce the locked identity-disjoint OOF outputs."""

    frozen_index = np.asarray(frozen["index"], dtype=np.int64)
    frozen_fold = np.asarray(frozen["fold"], dtype=np.int16)
    selected = np.flatnonzero(frozen_fold == fold)
    expected_indices = frozen_index[selected]
    current_valid = np.asarray(bundle.reference_valid, dtype=bool)
    current_indices = np.asarray(bundle.index, dtype=np.int64)[current_valid]
    if not np.array_equal(current_indices, expected_indices):
        raise RuntimeError(f"frozen valid index/order mismatch for fold {fold}")

    current_fields: dict[str, np.ndarray] = {
        "prediction": np.asarray(bundle.prediction)[current_valid],
        "map_prediction": np.asarray(bundle.map_prediction)[current_valid],
        "rr_std": np.asarray(bundle.rr_std)[current_valid],
        "uncertainty": np.asarray(bundle.uncertainty)[current_valid],
        "quality": np.asarray(bundle.quality)[current_valid],
        "alias_probability": np.asarray(bundle.alias_probability)[current_valid],
        "posterior_entropy": np.asarray(bundle.posterior_entropy)[current_valid],
        "topk_rr": np.asarray(bundle.topk_rr)[current_valid],
        "topk_probability": np.asarray(bundle.topk_probability)[current_valid],
        "posterior_probability": np.asarray(bundle.posterior_probability)[current_valid],
        "spike_rate": np.asarray(bundle.spike_rate)[current_valid],
        "radar_weights": np.asarray(bundle.radar_weights)[current_valid],
    }
    limits = {
        "prediction": tolerance.prediction_bpm,
        "map_prediction": tolerance.map_prediction_bpm,
        "rr_std": tolerance.rr_std_bpm,
        "uncertainty": tolerance.uncertainty,
        "quality": tolerance.quality,
        "alias_probability": tolerance.alias_probability,
        "posterior_entropy": tolerance.posterior_entropy,
        "topk_rr": tolerance.topk_rr_bpm,
        "topk_probability": tolerance.topk_probability,
        "posterior_probability": tolerance.posterior_probability,
        "spike_rate": tolerance.spike_rate,
        "radar_weights": tolerance.radar_weight,
    }
    result: dict[str, Any] = {"rows": len(expected_indices), "fields": {}}
    for field, current in current_fields.items():
        expected = np.asarray(frozen[field])[selected]
        if current.shape != expected.shape:
            raise RuntimeError(f"frozen field shape mismatch for fold {fold}: {field}")
        maximum = _finite_max_abs(current, expected)
        limit = float(limits[field])
        if not np.allclose(current, expected, rtol=0.0, atol=limit, equal_nan=True):
            raise RuntimeError(
                f"frozen valid prediction mismatch fold={fold} field={field}: "
                f"max_abs={maximum:.6g} > atol={limit:.6g}"
            )
        result["fields"][field] = {"max_abs_difference": maximum, "atol": limit}
    return result


def _save_fold_result(
    path: Path,
    bundle: PredictionBundle,
    *,
    binding: FoldBinding,
    run_signature: str,
    inference_signature: str,
) -> dict[str, Any]:
    arrays: dict[str, Any] = {
        field: np.asarray(getattr(bundle, field)) for field in FOLD_DEPLOYMENT_FIELDS
    }
    arrays.update(
        format_version=np.asarray(FORMAT_VERSION, dtype=np.int16),
        fold=np.full(len(bundle), binding.fold, dtype=np.int16),
        run_signature=np.asarray(run_signature),
        inference_signature=np.asarray(inference_signature),
        checkpoint_sha256=np.asarray(binding.checkpoint_sha256),
    )
    _atomic_npz(path, arrays)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "fields": sorted(arrays),
    }


def _fold_marker_path(path: Path) -> Path:
    return path.with_name(path.stem + ".verified.json")


def _commit_verified_fold(
    path: Path,
    *,
    artifact: Mapping[str, Any],
    binding: FoldBinding,
    run_signature: str,
    inference_signature: str,
    frozen_oof_sha256: str,
    parity: Mapping[str, Any],
    reference_valid_count: int,
) -> dict[str, Any]:
    """Write the commit marker only after frozen parity has succeeded."""

    if artifact.get("sha256") != _sha256_file(path) or int(
        artifact.get("bytes", -1)
    ) != path.stat().st_size:
        raise RuntimeError(f"fold artifact changed before verified commit: {path}")
    marker = {
        "format_version": FORMAT_VERSION,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "fold": binding.fold,
        "run_signature": run_signature,
        "inference_signature": inference_signature,
        "checkpoint_sha256": binding.checkpoint_sha256,
        "frozen_oof_sha256": frozen_oof_sha256,
        "row_count": len(binding.expected_indices),
        "reference_valid_count": int(reference_valid_count),
        "expected_index_sha256": hashlib.sha256(
            np.asarray(binding.expected_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "deployment_allowlist": list(FOLD_DEPLOYMENT_FIELDS),
        "excluded_fields": ["target", "observable"],
        "artifact": dict(artifact),
        "frozen_valid_parity": dict(parity),
        "commit_semantics": (
            "marker was atomically created only after artifact serialization and "
            "frozen-valid parity verification succeeded"
        ),
    }
    _atomic_json(_fold_marker_path(path), marker)
    return marker


def _load_reusable_fold(
    path: Path,
    *,
    binding: FoldBinding,
    run_signature: str,
    inference_signature: str,
    frozen_oof_sha256: str,
) -> PredictionBundle:
    marker_path = _fold_marker_path(path)
    if not marker_path.is_file():
        raise RuntimeError(f"resume fold has no verified commit marker: {marker_path}")
    marker = _load_json(marker_path)
    marker_expectations = {
        "format_version": FORMAT_VERSION,
        "fold": binding.fold,
        "run_signature": run_signature,
        "inference_signature": inference_signature,
        "checkpoint_sha256": binding.checkpoint_sha256,
        "frozen_oof_sha256": frozen_oof_sha256,
        "row_count": len(binding.expected_indices),
    }
    for key, expected in marker_expectations.items():
        if marker.get(key) != expected:
            raise RuntimeError(f"resume verified marker mismatch for {key}: {marker_path}")
    artifact = marker.get("artifact")
    if not isinstance(artifact, Mapping):
        raise RuntimeError(f"resume verified marker has no artifact binding: {marker_path}")
    if artifact.get("sha256") != _sha256_file(path):
        raise RuntimeError(f"resume fold artifact SHA-256 mismatch: {path}")
    if int(artifact.get("bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"resume fold artifact size mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        expected_fields = set(FOLD_DEPLOYMENT_FIELDS) | {
            "format_version",
            "fold",
            "run_signature",
            "inference_signature",
            "checkpoint_sha256",
        }
        if set(data.files) != expected_fields:
            raise RuntimeError(f"resume deployment allowlist mismatch: {path}")
        if str(np.asarray(data["run_signature"]).item()) != run_signature:
            raise RuntimeError(f"resume run signature mismatch: {path}")
        if str(np.asarray(data["inference_signature"]).item()) != inference_signature:
            raise RuntimeError(f"resume inference signature mismatch: {path}")
        if str(np.asarray(data["checkpoint_sha256"]).item()) != binding.checkpoint_sha256:
            raise RuntimeError(f"resume checkpoint hash mismatch: {path}")
        values = {field: np.asarray(data[field]) for field in FOLD_DEPLOYMENT_FIELDS}
        count = len(values["index"])
        bundle = PredictionBundle(
            **values,
            target=np.full(count, np.nan, dtype=np.float32),
            observable=np.full(count, np.nan, dtype=np.float32),
        )
        folds = np.asarray(data["fold"], dtype=np.int16)
    if not np.array_equal(bundle.index, binding.expected_indices):
        raise RuntimeError(f"resume row coverage mismatch: {path}")
    if not np.all(folds == binding.fold):
        raise RuntimeError(f"resume fold labels mismatch: {path}")
    if int(marker.get("reference_valid_count", -1)) != int(
        np.asarray(bundle.reference_valid, dtype=bool).sum()
    ):
        raise RuntimeError(f"resume reference-valid count mismatch: {path}")
    return bundle


def _posterior_grid(checkpoint: Mapping[str, Any]) -> np.ndarray:
    state = checkpoint["model_state"]
    return np.asarray(state["rr_bins"].detach().cpu(), dtype=np.float32)


def build_final_arrays(
    bundle: PredictionBundle,
    fold_for_row: np.ndarray,
    metadata: pd.DataFrame,
    posterior_grid: np.ndarray,
    *,
    run_signature: str,
    inference_signature: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not np.array_equal(bundle.index, np.arange(len(metadata))):
        raise RuntimeError("combined predictions are not an exact cache-order cover")
    valid = metadata["reference_valid"].to_numpy(dtype=bool)
    reference = pd.to_numeric(metadata["rr_bpm"], errors="coerce").to_numpy(float)
    reference = np.where(valid, reference, np.nan).astype(np.float32)
    return {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int16),
        "cache_index": np.asarray(bundle.index, dtype=np.int64),
        "fold": np.asarray(fold_for_row, dtype=np.int16),
        "session_id": metadata["session_id"].astype(str).to_numpy(dtype=str),
        "identity": metadata["identity"].astype(str).to_numpy(dtype=str),
        "protocol": metadata["protocol"].astype(str).to_numpy(dtype=str),
        "window_number": metadata["window_number"].to_numpy(dtype=np.int64),
        "window_start_s": metadata["window_start_s"].to_numpy(dtype=np.float64),
        "window_end_s": metadata["window_end_s"].to_numpy(dtype=np.float64),
        "reference_valid": valid,
        "reference_rr_bpm": reference,
        "prediction_bpm": np.asarray(bundle.prediction, dtype=np.float32),
        "map_prediction_bpm": np.asarray(bundle.map_prediction, dtype=np.float32),
        "rr_std_bpm": np.asarray(bundle.rr_std, dtype=np.float32),
        "uncertainty_score": np.asarray(bundle.uncertainty, dtype=np.float32),
        "quality": np.asarray(bundle.quality, dtype=np.float32),
        "alias_probability": np.asarray(bundle.alias_probability, dtype=np.float32),
        "posterior_entropy": np.asarray(bundle.posterior_entropy, dtype=np.float32),
        "topk_rr_bpm": np.asarray(bundle.topk_rr, dtype=np.float32),
        "topk_probability": np.asarray(bundle.topk_probability, dtype=np.float32),
        "posterior_probability": np.asarray(bundle.posterior_probability, dtype=np.float16),
        "posterior_rr_bins_bpm": np.asarray(posterior_grid, dtype=np.float32),
        "spike_rate": np.asarray(bundle.spike_rate, dtype=np.float32),
        "radar_weights": np.asarray(bundle.radar_weights, dtype=np.float32),
        "run_signature": np.asarray(run_signature),
        "inference_signature": np.asarray(inference_signature),
        "provenance_json": np.asarray(
            json.dumps(_strict_json(provenance), sort_keys=True, separators=(",", ":"))
        ),
    }


def build_csv(
    arrays: Mapping[str, np.ndarray],
    *,
    binding_for_fold: Mapping[int, FoldBinding],
    cache_source: Mapping[str, Any],
) -> pd.DataFrame:
    scalar_columns = (
        "cache_index",
        "fold",
        "session_id",
        "identity",
        "protocol",
        "window_number",
        "window_start_s",
        "window_end_s",
        "reference_valid",
        "reference_rr_bpm",
        "prediction_bpm",
        "map_prediction_bpm",
        "rr_std_bpm",
        "uncertainty_score",
        "quality",
        "alias_probability",
        "posterior_entropy",
        "spike_rate",
    )
    frame = pd.DataFrame({key: np.asarray(arrays[key]) for key in scalar_columns})
    top_rr = np.asarray(arrays["topk_rr_bpm"])
    top_probability = np.asarray(arrays["topk_probability"])
    for rank in range(top_rr.shape[1]):
        frame[f"posterior_top{rank + 1}_rr_bpm"] = top_rr[:, rank]
        frame[f"posterior_top{rank + 1}_probability"] = top_probability[:, rank]
    radar_weights = np.asarray(arrays["radar_weights"])
    for radar in range(radar_weights.shape[1]):
        frame[f"radar_{radar + 1}_weight"] = radar_weights[:, radar]
    posterior = np.asarray(arrays["posterior_probability"], dtype=np.float32)
    frame["posterior_probability_json"] = [
        json.dumps(row.tolist(), separators=(",", ":")) for row in posterior
    ]
    grid = np.asarray(arrays["posterior_rr_bins_bpm"], dtype=float)
    frame["posterior_rr_min_bpm"] = float(grid[0])
    frame["posterior_rr_max_bpm"] = float(grid[-1])
    frame["posterior_rr_bin_width_bpm"] = float(np.median(np.diff(grid)))
    run_signature = str(np.asarray(arrays["run_signature"]).item())
    inference_signature = str(np.asarray(arrays["inference_signature"]).item())
    frame["run_signature"] = run_signature
    frame["inference_signature"] = inference_signature
    frame["checkpoint_sha256"] = [
        binding_for_fold[int(fold)].checkpoint_sha256 for fold in frame["fold"]
    ]
    frame["cache_manifest_sha256"] = str(cache_source["cache_manifest_sha256"])
    fingerprints = cache_source["source_fingerprint_by_session"]
    frame["source_fingerprint"] = [fingerprints[str(value)] for value in frame["session_id"]]
    return frame


def _resolve_device(value: str) -> torch.device:
    requested = value.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_frozen(
    path: Path, run_signature: str, metadata: pd.DataFrame
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen OOF is missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    if str(np.asarray(result["run_signature"]).item()) != run_signature:
        raise RuntimeError("frozen OOF run signature differs from checkpoints")
    if len(np.unique(result["index"])) != len(result["index"]):
        raise RuntimeError("frozen OOF contains duplicate cache indices")
    if not np.asarray(result["reference_valid"], dtype=bool).all():
        raise RuntimeError("frozen OOF unexpectedly contains invalid-reference rows")
    index = np.asarray(result["index"], dtype=np.int64)
    cache_valid = metadata["reference_valid"].to_numpy(dtype=bool)
    if not np.array_equal(np.sort(index), np.flatnonzero(cache_valid)):
        raise RuntimeError("frozen OOF does not exactly cover cache valid-reference rows")
    # The training bundle stores float32 targets.  Casting the cache source to
    # that declared dtype must be bit-for-bit identical, not merely close.
    cache_target = pd.to_numeric(
        metadata.iloc[index]["rr_bpm"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    frozen_target = np.asarray(result["target"], dtype=np.float32)
    if not np.array_equal(frozen_target, cache_target, equal_nan=True):
        raise RuntimeError("frozen target is not exactly float32-bound to cache target")
    return result


def _verify_final_frozen_parity(
    npz_path: Path,
    frozen: Mapping[str, np.ndarray],
    tolerance: FrozenTolerance,
) -> dict[str, Any]:
    """Re-run valid-row parity directly against a completed deployment NPZ."""

    with np.load(npz_path, allow_pickle=False) as data:
        final = {key: np.asarray(data[key]) for key in data.files}
    required = {
        "cache_index",
        "fold",
        "reference_valid",
        "reference_rr_bpm",
        "prediction_bpm",
        "map_prediction_bpm",
        "rr_std_bpm",
        "uncertainty_score",
        "quality",
        "alias_probability",
        "posterior_entropy",
        "topk_rr_bpm",
        "topk_probability",
        "posterior_probability",
        "spike_rate",
        "radar_weights",
    }
    missing = sorted(required - set(final))
    if missing:
        raise RuntimeError(f"completed NPZ lacks parity fields: {missing}")
    mapping = {
        "prediction": ("prediction_bpm", tolerance.prediction_bpm),
        "map_prediction": ("map_prediction_bpm", tolerance.map_prediction_bpm),
        "rr_std": ("rr_std_bpm", tolerance.rr_std_bpm),
        "uncertainty": ("uncertainty_score", tolerance.uncertainty),
        "quality": ("quality", tolerance.quality),
        "alias_probability": ("alias_probability", tolerance.alias_probability),
        "posterior_entropy": ("posterior_entropy", tolerance.posterior_entropy),
        "topk_rr": ("topk_rr_bpm", tolerance.topk_rr_bpm),
        "topk_probability": ("topk_probability", tolerance.topk_probability),
        "posterior_probability": (
            "posterior_probability",
            tolerance.posterior_probability,
        ),
        "spike_rate": ("spike_rate", tolerance.spike_rate),
        "radar_weights": ("radar_weights", tolerance.radar_weight),
    }
    cache_index = np.asarray(final["cache_index"], dtype=np.int64)
    fold = np.asarray(final["fold"], dtype=np.int16)
    valid = np.asarray(final["reference_valid"], dtype=bool)
    audits: dict[str, Any] = {}
    for fold_number in range(6):
        current_rows = np.flatnonzero(valid & (fold == fold_number))
        frozen_rows = np.flatnonzero(
            np.asarray(frozen["fold"], dtype=np.int16) == fold_number
        )
        if not np.array_equal(cache_index[current_rows], frozen["index"][frozen_rows]):
            raise RuntimeError(f"completed NPZ frozen index mismatch in fold {fold_number}")
        if not np.array_equal(
            np.asarray(final["reference_rr_bpm"], dtype=np.float32)[current_rows],
            np.asarray(frozen["target"], dtype=np.float32)[frozen_rows],
            equal_nan=True,
        ):
            raise RuntimeError(
                f"completed NPZ target/cache/frozen binding mismatch in fold {fold_number}"
            )
        fields: dict[str, Any] = {}
        for frozen_name, (final_name, limit) in mapping.items():
            current = np.asarray(final[final_name])[current_rows]
            expected = np.asarray(frozen[frozen_name])[frozen_rows]
            maximum = _finite_max_abs(current, expected)
            if current.shape != expected.shape or not np.allclose(
                current, expected, rtol=0.0, atol=float(limit), equal_nan=True
            ):
                raise RuntimeError(
                    f"completed NPZ frozen parity mismatch fold={fold_number} "
                    f"field={final_name}: max_abs={maximum:.6g}, atol={limit:.6g}"
                )
            fields[final_name] = {
                "max_abs_difference": maximum,
                "atol": float(limit),
            }
        audits[str(fold_number)] = {"rows": len(current_rows), "fields": fields}
    return audits


def _verify_complete_reuse(
    output_dir: Path,
    *,
    inference_signature: str,
    expected_rows: int,
    run_signature: str,
    frozen: Mapping[str, np.ndarray],
    frozen_oof_sha256: str,
    tolerance: FrozenTolerance,
    bindings: Sequence[FoldBinding],
) -> bool:
    provenance_path = output_dir / "provenance.json"
    npz_path = output_dir / "snn_all_windows.npz"
    csv_path = output_dir / "snn_all_windows.csv"
    present = [provenance_path.is_file(), npz_path.is_file(), csv_path.is_file()]
    if not any(present):
        return False
    if not all(present):
        raise RuntimeError("--reuse found a partial completed-output set")
    provenance = _load_json(provenance_path)
    authority_v1 = provenance.get("schema_version") == BASE_OOF_AUTHORITY_SCHEMA
    if authority_v1:
        if provenance.get("content_sha256") != _canonical_content_sha256(
            provenance
        ):
            raise RuntimeError("--reuse provenance canonical content hash mismatch")
    if provenance.get("inference_signature") != inference_signature:
        raise RuntimeError("--reuse inference signature mismatch")
    frozen_record = provenance.get("frozen_valid_oof_verification", {})
    if frozen_record.get("source_sha256") != frozen_oof_sha256:
        raise RuntimeError("--reuse frozen OOF SHA-256 mismatch")
    recorded_outputs = provenance.get("outputs")
    if not isinstance(recorded_outputs, Mapping):
        raise RuntimeError("--reuse provenance has no output binding")
    if recorded_outputs.get("npz_sha256") != _sha256_file(npz_path):
        raise RuntimeError("--reuse completed NPZ SHA-256 mismatch")
    if recorded_outputs.get("csv_sha256") != _sha256_file(csv_path):
        raise RuntimeError("--reuse completed CSV SHA-256 mismatch")
    if authority_v1:
        if recorded_outputs.get("npz_bytes") != npz_path.stat().st_size:
            raise RuntimeError("--reuse completed NPZ byte-count mismatch")
        if recorded_outputs.get("csv_bytes") != csv_path.stat().st_size:
            raise RuntimeError("--reuse completed CSV byte-count mismatch")
    with np.load(npz_path, allow_pickle=False) as data:
        if str(np.asarray(data["inference_signature"]).item()) != inference_signature:
            raise RuntimeError("--reuse completed NPZ signature mismatch")
        indices = np.asarray(data["cache_index"], dtype=np.int64)
    if len(indices) != expected_rows or not np.array_equal(
        indices, np.arange(expected_rows)
    ):
        raise RuntimeError("--reuse completed NPZ row exact-cover mismatch")
    for binding in bindings:
        fold_path = output_dir / f"fold_{binding.fold}_all_windows.npz"
        _load_reusable_fold(
            fold_path,
            binding=binding,
            run_signature=run_signature,
            inference_signature=inference_signature,
            frozen_oof_sha256=frozen_oof_sha256,
        )
    _verify_final_frozen_parity(npz_path, frozen, tolerance)
    return True


def _deployment_freeze_assessment(
    *,
    strict_frozen_parity: bool,
    raw_source_fingerprints_verified: bool,
) -> dict[str, Any]:
    """Return the fail-closed deployment-freeze decision and its blockers."""

    blockers: list[str] = []
    if not strict_frozen_parity:
        blockers.append("strict_cuda_amp_frozen_parity_required")
    if not raw_source_fingerprints_verified:
        blockers.append("raw_source_fingerprint_verification_required")
    return {
        "eligible": not blockers,
        "blockers": blockers,
        "strict_frozen_parity": bool(strict_frozen_parity),
        "raw_source_fingerprints_verified": bool(
            raw_source_fingerprints_verified
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = args.cache_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    run_config_path = run_dir / "run_config.json"
    run_config = _load_json(run_config_path)
    arguments = run_config.get("arguments", {})
    if _resolve_recorded_path(arguments.get("cache_dir")) != cache_dir:
        raise RuntimeError("requested cache is not the cache bound in run_config")
    if arguments.get("model") != "snn" or int(arguments.get("folds", -1)) != 6:
        raise RuntimeError("the source run must be a complete six-fold SNN run")

    cache, base_aux_dim, history_names = prepare_cache(cache_dir, run_config)
    if args.expected_rows is not None and len(cache.metadata) != args.expected_rows:
        raise RuntimeError(
            f"cache has {len(cache.metadata)} rows, expected {args.expected_rows}"
        )
    cache_source = validate_cache_sources(
        cache_dir,
        cache.metadata,
        verify_raw_sources=args.verify_raw_sources,
        verified_cache_provenance=cache.provenance,
    )

    bindings: list[FoldBinding] = []
    checkpoints: dict[int, dict[str, Any]] = {}
    for fold in range(6):
        path = run_dir / f"fold_{fold}" / "snn_best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing fold checkpoint: {path}")
        binding, checkpoint = validate_fold_checkpoint(
            path,
            fold=fold,
            cache=cache,
            base_aux_dim=base_aux_dim,
            run_config=run_config,
            run_dir=run_dir,
        )
        bindings.append(binding)
        checkpoints[fold] = checkpoint
    fold_for_row = validate_identity_partition(bindings, cache.metadata)

    assignment_path = run_dir / "fold_assignments.json"
    assignment = _load_json(assignment_path).get("identity_to_fold")
    expected_assignment = {
        identity: binding.fold
        for binding in bindings
        for identity in binding.test_identities
    }
    if assignment != expected_assignment:
        raise RuntimeError("checkpoint identity owners differ from fold_assignments.json")

    run_signature = str(run_config.get("run_signature", ""))
    if not run_signature:
        raise RuntimeError("source run has no run_signature")
    frozen_path = run_dir / "snn_oof.npz"
    frozen = _load_frozen(frozen_path, run_signature, cache.metadata)
    frozen_oof_sha256 = _sha256_file(frozen_path)
    device = _resolve_device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    tolerance = _tolerance_for(device, amp, args.frozen_tolerance)
    source_hashes = _runtime_source_hashes()
    signature_payload = {
        "format_version": FORMAT_VERSION,
        "source_run_signature": run_signature,
        "cache_manifest_sha256": cache_source["cache_manifest_sha256"],
        "cache_content_signature_sha256": cache_source[
            "cache_content_signature_sha256"
        ],
        "frozen_oof_sha256": frozen_oof_sha256,
        "checkpoint_sha256": {
            str(binding.fold): binding.checkpoint_sha256 for binding in bindings
        },
        "runtime_source_sha256": source_hashes,
        # Bind both the requested policy and its observed result.  In
        # particular, an artifact produced with --no-verify-raw-sources must
        # never be reusable as though raw provenance had been checked.
        "verify_raw_sources_requested": bool(args.verify_raw_sources),
        "raw_source_fingerprints_verified": bool(
            cache_source["raw_source_fingerprints_verified"]
        ),
        "device_type": device.type,
        "amp": amp,
        "batch_size": int(args.batch_size),
        "frozen_tolerance": asdict(tolerance),
    }
    inference_signature = _canonical_signature(signature_payload)

    if args.reuse and _verify_complete_reuse(
        output_dir,
        inference_signature=inference_signature,
        expected_rows=len(cache.metadata),
        run_signature=run_signature,
        frozen=frozen,
        frozen_oof_sha256=frozen_oof_sha256,
        tolerance=tolerance,
        bindings=bindings,
    ):
        return _load_json(output_dir / "provenance.json")
    known_outputs = [
        output_dir / "snn_all_windows.npz",
        output_dir / "snn_all_windows.csv",
        output_dir / "provenance.json",
    ]
    if not args.force and not args.resume and not args.reuse and any(
        path.exists() for path in known_outputs
    ):
        raise RuntimeError(
            f"output already exists in {output_dir}; use --reuse, --resume, or --force"
        )

    frozen_index = np.asarray(frozen["index"], dtype=np.int64)
    expected_valid = np.flatnonzero(
        cache.metadata["reference_valid"].to_numpy(dtype=bool)
    )
    if not np.array_equal(np.sort(frozen_index), expected_valid):
        raise RuntimeError("frozen OOF does not exactly cover cache valid-reference rows")

    fold_bundles: list[PredictionBundle] = []
    frozen_audits: dict[str, Any] = {}
    fold_commit_markers: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for binding in bindings:
        fold_path = output_dir / f"fold_{binding.fold}_all_windows.npz"
        if args.resume and fold_path.is_file() and not args.force:
            bundle = _load_reusable_fold(
                fold_path,
                binding=binding,
                run_signature=run_signature,
                inference_signature=inference_signature,
                frozen_oof_sha256=frozen_oof_sha256,
            )
        else:
            checkpoint = checkpoints[binding.fold]
            center = _as_numpy_scaler(checkpoint, "aux_center")
            scale = _as_numpy_scaler(checkpoint, "aux_scale")
            aux_scaled = transform_aux(cache.aux, center, scale)
            loader = make_loader(
                cache,
                aux_scaled,
                binding.expected_indices,
                batch_size=args.batch_size,
                workers=args.workers,
                device=device,
                seed=int(arguments.get("seed", 0)) + 1009 * binding.fold + 2,
                train=False,
                auxiliary_layout=infer_auxiliary_layout(base_aux_dim),
            )
            if not isinstance(loader.dataset, CachedRadarDataset):
                raise RuntimeError("shared loader no longer uses CachedRadarDataset")
            model = build_model(
                str(checkpoint["model_type"]), checkpoint["model_kwargs"]
            )
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model = model.to(device)
            bundle = predict_label_free(model, loader, device, amp=amp)
            if not np.array_equal(bundle.index, binding.expected_indices):
                raise RuntimeError(f"inference row/order mismatch in fold {binding.fold}")
            parity = verify_frozen_valid_predictions(
                bundle, binding.fold, frozen, tolerance
            )
            artifact = _save_fold_result(
                fold_path,
                bundle,
                binding=binding,
                run_signature=run_signature,
                inference_signature=inference_signature,
            )
            _commit_verified_fold(
                fold_path,
                artifact=artifact,
                binding=binding,
                run_signature=run_signature,
                inference_signature=inference_signature,
                frozen_oof_sha256=frozen_oof_sha256,
                parity=parity,
                reference_valid_count=int(
                    np.asarray(bundle.reference_valid, dtype=bool).sum()
                ),
            )
            del aux_scaled, loader, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        frozen_audits[str(binding.fold)] = verify_frozen_valid_predictions(
            bundle, binding.fold, frozen, tolerance
        )
        fold_commit_markers[str(binding.fold)] = _load_json(
            _fold_marker_path(fold_path)
        )
        fold_bundles.append(bundle)

    combined = concatenate_bundles(fold_bundles)
    if len(combined) != len(cache.metadata):
        raise RuntimeError("combined all-window prediction count is incomplete")
    if len(np.unique(combined.index)) != len(combined):
        raise RuntimeError("combined all-window predictions contain duplicates")
    if not np.array_equal(combined.index, np.arange(len(cache.metadata))):
        raise RuntimeError("combined all-window predictions contain omissions")

    grid = _posterior_grid(checkpoints[0])
    for fold in range(1, 6):
        if not np.array_equal(grid, _posterior_grid(checkpoints[fold])):
            raise RuntimeError("fold checkpoints use different posterior RR grids")
    strict_frozen_parity = bool(
        device.type == "cuda"
        and amp
        and args.frozen_tolerance in {"auto", "strict"}
    )
    freeze_assessment = _deployment_freeze_assessment(
        strict_frozen_parity=strict_frozen_parity,
        raw_source_fingerprints_verified=bool(
            cache_source["raw_source_fingerprints_verified"]
        ),
    )
    canonical_cache_provenance = cache_source.get("canonical_cache_provenance")
    run_cache_provenance = run_config.get("cache_provenance")
    source_claims_scientific = bool(
        arguments.get("cache_trust_mode") == "scientific"
        and run_config.get("claim_classification")
        == "retrospective_scientific_noncommercial"
    )
    scientific_eligible = bool(
        source_claims_scientific
        and isinstance(canonical_cache_provenance, Mapping)
        and canonical_cache_provenance.get("classification")
        == "acquisition_scientific"
        and canonical_cache_provenance.get("scientific_eligible") is True
        and run_cache_provenance == canonical_cache_provenance
        and cache_source["raw_input_sha256_bindings_verified"] is True
    )
    if source_claims_scientific and not scientific_eligible:
        raise RuntimeError(
            "scientific source run cannot publish without exact canonical cache "
            "and raw-source authority"
        )
    identity_to_test_fold_sha256 = _canonical_sha256(expected_assignment)
    row_fold_binding_sha256 = _row_fold_binding_sha256(
        combined.index,
        cache.metadata.iloc[np.asarray(combined.index, dtype=np.int64)]["identity"],
        fold_for_row[np.asarray(combined.index, dtype=np.int64)],
    )
    provenance: dict[str, Any] = {
        "schema_version": BASE_OOF_AUTHORITY_SCHEMA,
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "source_run_config": str(run_config_path),
        "source_run_config_sha256": _sha256_file(run_config_path),
        "run_signature": run_signature,
        "inference_signature": inference_signature,
        "signature_payload": signature_payload,
        "runtime_source_sha256": source_hashes,
        "cache": cache_source,
        "canonical_cache_provenance": canonical_cache_provenance,
        "scientific_eligible": scientific_eligible,
        "claim_classification": (
            "retrospective_scientific_noncommercial"
            if scientific_eligible
            else "historical_diagnostic_noncommercial"
        ),
        "commercial_claim_allowed": False,
        "cache_shape": {
            "maps": list(cache.maps.shape),
            "aux": list(cache.aux.shape),
        },
        "causal_history_feature_names": history_names,
        "identity_disjoint": True,
        "row_exact_cover": True,
        "row_count": len(combined),
        "valid_reference_rows": int(
            cache.metadata["reference_valid"].to_numpy(dtype=bool).sum()
        ),
        "invalid_reference_rows": int(
            (~cache.metadata["reference_valid"].to_numpy(dtype=bool)).sum()
        ),
        "identity_to_test_fold": expected_assignment,
        "identity_to_test_fold_sha256": identity_to_test_fold_sha256,
        "row_fold_binding_sha256": row_fold_binding_sha256,
        "checkpoints": {
            str(binding.fold): {
                "path": str(binding.checkpoint_path),
                "sha256": binding.checkpoint_sha256,
                "test_identities": list(binding.test_identities),
                "all_window_rows": len(binding.expected_indices),
            }
            for binding in bindings
        },
        "verified_fold_commits": fold_commit_markers,
        "deployment_field_policy": {
            "per_fold_allowlist": list(FOLD_DEPLOYMENT_FIELDS),
            "per_fold_explicitly_excluded": ["target", "observable"],
            "reference_valid_role": "evaluation/output mask only; never a model input",
        },
        "label_free_forward": {
            "verified": True,
            "model_inputs": ["map", "radar_mask", "aux"],
            "target_or_qc_inputs": [],
            "reference_valid_usage": "output evaluation/masking only",
        },
        "frozen_valid_oof_verification": {
            "source": str(frozen_path),
            "source_sha256": frozen_oof_sha256,
            "tolerance_profile": args.frozen_tolerance,
            "resolved_tolerance": asdict(tolerance),
            "folds": frozen_audits,
            "strict_frozen_parity": strict_frozen_parity,
            "classification": (
                "strict_cuda_amp_frozen_parity"
                if strict_frozen_parity
                else "cpu_mean_rr_near_equivalence_only"
            ),
            "cpu_caveat": (
                None
                if strict_frozen_parity
                else (
                    "The frozen OOF was produced with CUDA float16 autocast. This CPU "
                    "artifact establishes near-equivalence of expected/mean RR and a "
                    "bounded full-posterior difference only. It is NOT strict parity "
                    "for MAP RR, top-k RR ordering, uncertainty, quality, or rr_std; "
                    "a CUDA-AMP regeneration is required before deployment freeze."
                )
            ),
            "non_strict_cpu_fields": (
                []
                if strict_frozen_parity
                else [
                    "map_prediction_bpm",
                    "topk_rr_bpm",
                    "uncertainty_score",
                    "quality",
                    "rr_std_bpm",
                ]
            ),
        },
        "deployment_freeze_eligible": freeze_assessment["eligible"],
        "deployment_freeze_assessment": freeze_assessment,
        "runtime": {
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "amp": amp,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "torch_version": torch.__version__,
        },
        "outputs": {
            "npz": str(output_dir / "snn_all_windows.npz"),
            "csv": str(output_dir / "snn_all_windows.csv"),
        },
    }
    arrays = build_final_arrays(
        combined,
        fold_for_row,
        cache.metadata,
        grid,
        run_signature=run_signature,
        inference_signature=inference_signature,
        provenance=provenance,
    )
    _atomic_npz(output_dir / "snn_all_windows.npz", arrays)
    completed_output_parity = _verify_final_frozen_parity(
        output_dir / "snn_all_windows.npz", frozen, tolerance
    )
    provenance["completed_output_frozen_parity"] = completed_output_parity
    arrays["provenance_json"] = np.asarray(
        json.dumps(_strict_json(provenance), sort_keys=True, separators=(",", ":"))
    )
    _atomic_npz(output_dir / "snn_all_windows.npz", arrays)
    # Recheck after embedding the final provenance payload.
    _verify_final_frozen_parity(output_dir / "snn_all_windows.npz", frozen, tolerance)
    csv = build_csv(
        arrays,
        binding_for_fold={binding.fold: binding for binding in bindings},
        cache_source=cache_source,
    )
    if len(csv) != len(cache.metadata) or not np.array_equal(
        csv["cache_index"].to_numpy(), np.arange(len(cache.metadata))
    ):
        raise RuntimeError("CSV serialization lost or reordered cache rows")
    _atomic_csv(output_dir / "snn_all_windows.csv", csv)
    _assert_publication_sources_current(cache_source, source_hashes)
    provenance["outputs"].update(
        {
            "npz_sha256": _sha256_file(output_dir / "snn_all_windows.npz"),
            "csv_sha256": _sha256_file(output_dir / "snn_all_windows.csv"),
            "npz_bytes": (output_dir / "snn_all_windows.npz").stat().st_size,
            "csv_bytes": (output_dir / "snn_all_windows.csv").stat().st_size,
        }
    )
    provenance = _strict_json(provenance)
    provenance["content_sha256"] = _canonical_content_sha256(provenance)
    _atomic_json(output_dir / "provenance.json", provenance)
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run identity-disjoint frozen SNN inference for all cached windows"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--verify-raw-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute each raw session's path/size/mtime fingerprint",
    )
    parser.add_argument(
        "--frozen-tolerance",
        choices=("auto", "strict", "cpu"),
        default="auto",
        help="numeric equivalence tolerance against CUDA-AMP frozen valid OOF",
    )
    parser.add_argument("--expected-rows", type=int, default=9576)
    parser.add_argument(
        "--resume", action="store_true", help="verify and reuse atomic per-fold files"
    )
    parser.add_argument(
        "--reuse", action="store_true", help="reuse a fully verified completed output"
    )
    parser.add_argument(
        "--force", action="store_true", help="atomically replace this pipeline's outputs"
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers non-negative")
    if args.expected_rows is not None and args.expected_rows <= 0:
        parser.error("--expected-rows must be positive")
    if args.force and (args.resume or args.reuse):
        parser.error("--force cannot be combined with --resume/--reuse")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    provenance = run(args)
    print(
        json.dumps(
            {
                "status": "complete",
                "rows": provenance["row_count"],
                "valid_reference_rows": provenance["valid_reference_rows"],
                "run_signature": provenance["run_signature"],
                "inference_signature": provenance["inference_signature"],
                "output": provenance["outputs"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
