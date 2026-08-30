#!/usr/bin/env python3
"""Build a versioned raw-window SVD component cache.

The cache is deliberately separate from ``artifacts/cache/rf32s`` so an
experimental representation can never silently replace the canonical input.
Feature extraction reads radar samples and timing metadata only.  Reference
columns are copied for downstream evaluation but never enter the extractor.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.build_features import replace_radar_outliers  # noqa: E402
from snn_rr.data import build_dataset_manifest, load_xethru_recording  # noqa: E402
from snn_rr.preprocess import causal_block_mean  # noqa: E402
from snn_rr.svd_features import (  # noqa: E402
    ATTRIBUTE_NAMES,
    DEFAULT_SVD_VARIANTS,
    svd_component_features,
)


PIPELINE_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "HAI_EXPERIMENT"
    )
    parser.add_argument(
        "--canonical-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts/cache/rf32s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/cache/svd_components_v1",
    )
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--all-windows", action="store_true")
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--n-iter", type=int, default=2)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def _canonical_acquisition_binding(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Return the exact acquisition binding carried by a canonical cache manifest.

    The per-session acquisition contract is already content-addressed and
    contains the reconstruction, synchronization, mapping, approval, and range
    artifact hashes.  Binding the entire mapping avoids silently dropping a
    newly added provenance hash from the SVD session signature.
    """

    value = manifest.get("acquisition_contract")
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError("canonical cache acquisition_contract is malformed")
    # Round-trip through canonical JSON to reject non-JSON values and detach
    # the task payload from the mutable manifest object.
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return json.loads(encoded)


def _acquisition_session_manifest_sha256(
    binding: dict[str, Any] | None,
) -> str | None:
    if binding is None:
        return None
    value = binding.get("acquisition_session_manifest_sha256")
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            "canonical acquisition contract lacks acquisition_session_manifest_sha256"
        )
    return value


def _session_signature(task: dict[str, Any]) -> str:
    keys = (
        "pipeline_sha256",
        "canonical_source_fingerprint",
        "canonical_session_manifest_sha256",
        "canonical_acquisition_session_manifest_sha256",
        "canonical_acquisition_binding",
        "selected_rows_sha256",
        "valid_only",
        "components",
        "nfft",
        "n_iter",
        "variant_names",
    )
    payload = {key: task[key] for key in keys}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_session(task: dict[str, Any]) -> dict[str, Any]:
    session_id = str(task["session_id"])
    output_dir = Path(task["output_dir"])
    canonical_dir = Path(task["canonical_dir"])
    canonical_manifest_path = canonical_dir / "manifest.json"
    observed_manifest_sha256 = _sha256(canonical_manifest_path)
    if observed_manifest_sha256 != task["canonical_session_manifest_sha256"]:
        raise RuntimeError(f"{session_id}: canonical session manifest changed after task creation")
    canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    observed_acquisition_binding = _canonical_acquisition_binding(canonical_manifest)
    if observed_acquisition_binding != task["canonical_acquisition_binding"]:
        raise RuntimeError(f"{session_id}: canonical acquisition binding changed after task creation")
    if _acquisition_session_manifest_sha256(observed_acquisition_binding) != task[
        "canonical_acquisition_session_manifest_sha256"
    ]:
        raise RuntimeError(f"{session_id}: canonical acquisition content hash changed")
    signature = _session_signature(task)
    complete = (
        output_dir / "spectra.npy",
        output_dir / "component_signals.npy",
        output_dir / "attributes.npy",
        output_dir / "frequencies_hz.npy",
        output_dir / "metadata.csv",
        output_dir / "manifest.json",
    )
    if not task["force"] and all(path.is_file() for path in complete):
        previous = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("session_signature") == signature:
            return {"session_id": session_id, "status": "ok", "cached": True, **previous}

    metadata = pd.read_csv(canonical_dir / "metadata.csv")
    if task["valid_only"]:
        selected_local = np.flatnonzero(metadata["reference_valid"].to_numpy(dtype=bool))
    else:
        selected_local = np.arange(len(metadata), dtype=np.int64)
    if not len(selected_local):
        return {"session_id": session_id, "status": "skipped", "reason": "no selected rows"}
    selected_metadata = metadata.iloc[selected_local].copy().reset_index(drop=True)
    selected_metadata.insert(0, "cache_index", int(task["cache_offset"]) + selected_local)

    radar_arrays: list[np.ndarray] = []
    outlier_counts: list[int] = []
    for recording_dir in task["recording_dirs"]:
        recording = load_xethru_recording(recording_dir, strict=True)
        values, outlier_count = replace_radar_outliers(
            np.asarray(recording.records["bins"], dtype=np.float32)
        )
        radar_arrays.append(causal_block_mean(values, 4))
        outlier_counts.append(int(outlier_count))
    common = min(map(len, radar_arrays))
    radar_arrays = [values[:common] for values in radar_arrays]

    variants = tuple(task["variant_names"])
    components = int(task["components"])
    nfft = int(task["nfft"])
    sample = svd_component_features(
        radar_arrays[0][:320],
        components=components,
        nfft=nfft,
        n_iter=int(task["n_iter"]),
        variants=variants,
    )
    spectra = np.empty(
        (
            len(selected_metadata),
            3,
            len(variants),
            components,
            sample.spectra.shape[-1],
        ),
        dtype=np.float16,
    )
    component_signals = np.empty(
        (
            len(selected_metadata),
            3,
            len(variants),
            components,
            320,
        ),
        dtype=np.float16,
    )
    attributes = np.empty(
        (len(selected_metadata), 3, len(variants), components, len(ATTRIBUTE_NAMES)),
        dtype=np.float32,
    )
    radar_start = float(canonical_manifest["radar_start_epoch"])
    biopac_start = float(canonical_manifest["biopac_start_epoch"])

    with threadpool_limits(limits=1):
        for output_row, row in selected_metadata.iterrows():
            window_start_epoch = biopac_start + float(row["window_start_s"])
            start = int(round((window_start_epoch - radar_start) * 10.0))
            reconstructed = radar_start + start / 10.0 - biopac_start
            if abs(reconstructed - float(row["window_start_s"])) > 0.051:
                raise RuntimeError(
                    f"{session_id} row {output_row}: timestamp cannot map to 10 Hz grid"
                )
            if start < 0 or start + 320 > common:
                raise RuntimeError(
                    f"{session_id} row {output_row}: raw window [{start},{start + 320}) "
                    f"escapes [0,{common})"
                )
            for radar_index, values in enumerate(radar_arrays):
                feature = svd_component_features(
                    values[start : start + 320],
                    components=components,
                    nfft=nfft,
                    n_iter=int(task["n_iter"]),
                    variants=variants,
                )
                if not np.array_equal(feature.frequencies_hz, sample.frequencies_hz):
                    raise RuntimeError("SVD frequency grid changed within a session")
                spectra[output_row, radar_index] = feature.spectra.astype(np.float16)
                component_signals[output_row, radar_index] = (
                    feature.component_signals.astype(np.float16)
                )
                attributes[output_row, radar_index] = feature.attributes

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save(output_dir / "spectra.npy", spectra)
    _atomic_save(output_dir / "component_signals.npy", component_signals)
    _atomic_save(output_dir / "attributes.npy", attributes)
    _atomic_save(output_dir / "frequencies_hz.npy", sample.frequencies_hz)
    metadata_temporary = output_dir / "metadata.csv.tmp"
    selected_metadata.to_csv(metadata_temporary, index=False)
    metadata_temporary.replace(output_dir / "metadata.csv")
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "session_signature": signature,
        "canonical_source_fingerprint": task["canonical_source_fingerprint"],
        "canonical_session_manifest_sha256": task["canonical_session_manifest_sha256"],
        "canonical_acquisition_session_manifest_sha256": task[
            "canonical_acquisition_session_manifest_sha256"
        ],
        "canonical_acquisition_binding": task["canonical_acquisition_binding"],
        "selected_rows_sha256": task["selected_rows_sha256"],
        "valid_only": bool(task["valid_only"]),
        "row_count": int(len(selected_metadata)),
        "cache_index_min": int(selected_metadata["cache_index"].min()),
        "cache_index_max": int(selected_metadata["cache_index"].max()),
        "spectra_shape": list(spectra.shape),
        "component_signals_shape": list(component_signals.shape),
        "attributes_shape": list(attributes.shape),
        "spectra_dtype": str(spectra.dtype),
        "component_signals_dtype": str(component_signals.dtype),
        "attributes_dtype": str(attributes.dtype),
        "variant_names": list(variants),
        "attribute_names": list(ATTRIBUTE_NAMES),
        "components": components,
        "nfft": nfft,
        "n_iter": int(task["n_iter"]),
        "frequency_min_hz": float(sample.frequencies_hz[0]),
        "frequency_max_hz": float(sample.frequencies_hz[-1]),
        "radar_outlier_replacements": outlier_counts,
        "feature_inputs": [
            "raw_radar_window",
            "past_only_outlier_state_from_session_start",
            "canonical_radar_and_biopac_timestamps_for_alignment",
        ],
        "label_inputs": [],
        "split_half_layout_status": "unverified_hypothesis",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {"session_id": session_id, "status": "ok", "cached": False, **manifest}


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    canonical_root = args.canonical_cache.resolve()
    output_root = args.output_dir.resolve()
    if output_root == canonical_root:
        raise ValueError("experimental SVD cache must not overwrite the canonical cache")
    output_root.mkdir(parents=True, exist_ok=True)

    canonical_root_manifest = json.loads(
        (canonical_root / "manifest.json").read_text(encoding="utf-8")
    )
    dataset_manifest = build_dataset_manifest(dataset_root)
    by_session = {subject.subject_id: subject for subject in dataset_manifest.usable_subjects}
    available = [
        item for item in canonical_root_manifest["sessions"] if item["status"] == "ok"
    ]
    if args.subjects:
        wanted = set(args.subjects)
        available = [item for item in available if item["session_id"] in wanted]
        missing = wanted - {item["session_id"] for item in available}
        if missing:
            raise KeyError(f"unknown or unusable sessions: {sorted(missing)}")

    pipeline_paths = [
        Path(__file__),
        SOURCE_ROOT / "snn_rr" / "svd_features.py",
        PROJECT_ROOT / "scripts" / "build_features.py",
        SOURCE_ROOT / "snn_rr" / "data.py",
        SOURCE_ROOT / "snn_rr" / "preprocess.py",
    ]
    pipeline_digest = hashlib.sha256(
        "".join(_sha256(path) for path in pipeline_paths).encode()
    ).hexdigest()
    offsets: dict[str, int] = {}
    offset = 0
    for item in [
        value for value in canonical_root_manifest["sessions"] if value["status"] == "ok"
    ]:
        offsets[item["session_id"]] = offset
        offset += int(item["window_count"])

    tasks: list[dict[str, Any]] = []
    for item in available:
        session_id = item["session_id"]
        subject = by_session[session_id]
        if subject.selected_session is None:
            raise RuntimeError(f"{session_id} has no selected radar session")
        canonical_dir = canonical_root / session_id
        canonical_session_manifest_path = canonical_dir / "manifest.json"
        canonical_session_manifest = json.loads(
            canonical_session_manifest_path.read_text(encoding="utf-8")
        )
        canonical_acquisition_binding = _canonical_acquisition_binding(
            canonical_session_manifest
        )
        metadata = pd.read_csv(canonical_dir / "metadata.csv")
        selected = (
            np.flatnonzero(metadata["reference_valid"].to_numpy(dtype=bool))
            if not args.all_windows
            else np.arange(len(metadata), dtype=np.int64)
        )
        selected_digest = hashlib.sha256(selected.tobytes()).hexdigest()
        tasks.append(
            {
                "session_id": session_id,
                "recording_dirs": [
                    str(subject.selected_session.radars[radar].recording_dir)
                    for radar in (1, 2, 3)
                ],
                "canonical_dir": str(canonical_dir),
                "canonical_source_fingerprint": item["source_fingerprint"],
                "canonical_session_manifest_sha256": _sha256(
                    canonical_session_manifest_path
                ),
                "canonical_acquisition_session_manifest_sha256": (
                    _acquisition_session_manifest_sha256(
                        canonical_acquisition_binding
                    )
                ),
                "canonical_acquisition_binding": canonical_acquisition_binding,
                "cache_offset": offsets[session_id],
                "output_dir": str(output_root / session_id),
                "selected_rows_sha256": selected_digest,
                "valid_only": not args.all_windows,
                "components": int(args.components),
                "nfft": int(args.nfft),
                "n_iter": int(args.n_iter),
                "variant_names": list(DEFAULT_SVD_VARIANTS),
                "pipeline_sha256": pipeline_digest,
                "force": bool(args.force),
            }
        )

    results: list[dict[str, Any]] = []
    if args.workers == 1:
        for task in tasks:
            print(f"[{task['session_id']}] building", flush=True)
            results.append(_build_session(task))
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {executor.submit(_build_session, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                results.append(result)
                print(
                    f"[{task['session_id']}] {result['status']} "
                    f"rows={result.get('row_count', 0)} cached={result.get('cached', False)}",
                    flush=True,
                )
    order = {task["session_id"]: index for index, task in enumerate(tasks)}
    results.sort(key=lambda value: order[value["session_id"]])
    root_manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "canonical_cache": str(canonical_root),
        "canonical_manifest_sha256": _sha256(canonical_root / "manifest.json"),
        "canonical_acquisition_contract": canonical_root_manifest.get(
            "acquisition_contract"
        ),
        "canonical_acquisition_reconstruction_content_sha256": (
            canonical_root_manifest.get("acquisition_contract", {}).get(
                "reconstruction_content_sha256"
            )
            if isinstance(canonical_root_manifest.get("acquisition_contract"), dict)
            else None
        ),
        "pipeline_sha256": pipeline_digest,
        "valid_only": not args.all_windows,
        "variant_names": list(DEFAULT_SVD_VARIANTS),
        "attribute_names": list(ATTRIBUTE_NAMES),
        "components": int(args.components),
        "nfft": int(args.nfft),
        "n_iter": int(args.n_iter),
        "row_count": int(sum(item.get("row_count", 0) for item in results)),
        "sessions": results,
    }
    _write_json(output_root / "manifest.json", root_manifest)
    print(
        f"Built {sum(item['status'] == 'ok' for item in results)} sessions and "
        f"{root_manifest['row_count']} rows in {output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
