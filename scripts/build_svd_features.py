#!/usr/bin/env python3
"""Build a versioned raw-window SVD component cache.

The cache is deliberately separate from ``artifacts/cache/rf32s`` so an
experimental representation can never silently replace the canonical input.
Feature values read radar samples and timing metadata only.  Reference columns
are copied for downstream evaluation.  Unless ``--all-windows`` is supplied,
the target-derived ``reference_valid`` column selects which rows are emitted;
that supervised row filter is disclosed in every manifest and never enters a
feature value.
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
from snn_rr.acquisition_contract import (  # noqa: E402
    load_acquisition_reconstruction,
    validate_raw_input_bindings,
)
from snn_rr.cache import load_feature_cache  # noqa: E402
from snn_rr.data import build_dataset_manifest, load_xethru_recording  # noqa: E402
from snn_rr.preprocess import causal_block_mean  # noqa: E402
from snn_rr.radar_timing import (  # noqa: E402
    causal_uniform_resample_radar_views_v1,
)
from snn_rr.svd_features import (  # noqa: E402
    ATTRIBUTE_NAMES,
    DEFAULT_SVD_VARIANTS,
    svd_component_features,
)


PIPELINE_VERSION = 4


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


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _pipeline_paths() -> list[Path]:
    return [
        Path(__file__),
        SOURCE_ROOT / "snn_rr" / "svd_features.py",
        PROJECT_ROOT / "scripts" / "build_features.py",
        SOURCE_ROOT / "snn_rr" / "acquisition_contract.py",
        SOURCE_ROOT / "snn_rr" / "cache.py",
        SOURCE_ROOT / "snn_rr" / "data.py",
        SOURCE_ROOT / "snn_rr" / "preprocess.py",
        SOURCE_ROOT / "snn_rr" / "synchronization.py",
        SOURCE_ROOT / "snn_rr" / "radar_timing.py",
    ]


def _pipeline_sha256(paths: list[Path]) -> str:
    return hashlib.sha256(
        "".join(_sha256(path) for path in paths).encode()
    ).hexdigest()


def _assert_pipeline_unchanged(paths: list[Path], expected_sha256: str) -> None:
    if _pipeline_sha256(paths) != expected_sha256:
        raise RuntimeError("SVD pipeline source changed while the cache was being built")


def _array_inventory(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _metadata_inventory(path: Path, metadata: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "shape": [len(metadata), len(metadata.columns)],
        "dtype": "csv",
    }


def _inventory_files_match(root: Path, inventory: Any) -> bool:
    if not isinstance(inventory, dict) or not inventory:
        return False
    for entry in inventory.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return False
        path = (root / entry["path"]).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            return False
        if not path.is_file():
            return False
        if entry.get("bytes") != path.stat().st_size:
            return False
        if entry.get("sha256") != _sha256(path):
            return False
    return True


def _cached_svd_manifest_is_current(
    root: Path,
    manifest: dict[str, Any],
    *,
    acquisition_v2: bool,
) -> bool:
    declared_content = manifest.get("content_sha256")
    if not isinstance(declared_content, str):
        return False
    content_payload = dict(manifest)
    content_payload.pop("content_sha256", None)
    if declared_content != _canonical_sha256(content_payload):
        return False
    inventory = manifest.get("file_inventory")
    required = {
        "spectra",
        "component_signals",
        "attributes",
        "frequencies_hz",
        "metadata",
    }
    if acquisition_v2:
        required.add("radar_timing_valid_mask")
    if not isinstance(inventory, dict) or set(inventory) != required:
        return False
    if manifest.get("inventory_sha256") != _canonical_sha256(inventory):
        return False
    return _inventory_files_match(root, inventory)


def _verify_bound_canonical_file(
    canonical_dir: Path,
    canonical_manifest: dict[str, Any],
    key: str,
    expected_name: str,
) -> Path:
    inventory = canonical_manifest.get("file_inventory")
    if not isinstance(inventory, dict) or not isinstance(inventory.get(key), dict):
        raise RuntimeError(f"canonical cache lacks bound {key} inventory")
    entry = inventory[key]
    if entry.get("path") != expected_name:
        raise RuntimeError(f"canonical {key} inventory redirects its loader target")
    path = (canonical_dir / expected_name).resolve()
    try:
        path.relative_to(canonical_dir.resolve())
    except ValueError as error:
        raise RuntimeError(f"canonical {key} path escapes the session root") from error
    if not path.is_file():
        raise RuntimeError(f"canonical {key} input is missing")
    if entry.get("bytes") != path.stat().st_size or entry.get("sha256") != _sha256(path):
        raise RuntimeError(f"canonical {key} input differs from its inventory")
    return path


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


def _unique_session_ids(value: Any, *, location: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{location} must be a non-empty session-ID list")
    session_ids: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise ValueError(f"{location} contains an invalid session ID")
        session_ids.append(item)
    if len(set(session_ids)) != len(session_ids):
        raise ValueError(f"{location} contains duplicate session IDs")
    return session_ids


def _canonical_expected_session_ids(
    canonical_available_session_ids: list[str],
    canonical_acquisition_contract: Any,
) -> list[str]:
    """Resolve the authoritative cohort independently of an SVD selection."""

    available = _unique_session_ids(
        canonical_available_session_ids,
        location="canonical cache usable-session catalogue",
    )
    if not (
        isinstance(canonical_acquisition_contract, dict)
        and canonical_acquisition_contract.get("schema_version")
        == "snn_rr.feature_cache_acquisition.v2"
    ):
        return available

    expected = _unique_session_ids(
        canonical_acquisition_contract.get("expected_usable_session_ids"),
        location="canonical acquisition expected usable-session IDs",
    )
    cache_usable = _unique_session_ids(
        canonical_acquisition_contract.get("cache_usable_session_ids"),
        location="canonical acquisition cache usable-session IDs",
    )
    if canonical_acquisition_contract.get(
        "expected_usable_session_ids_sha256"
    ) != _canonical_sha256(expected):
        raise ValueError("canonical acquisition expected-session hash mismatch")
    if canonical_acquisition_contract.get(
        "cache_usable_session_ids_sha256"
    ) != _canonical_sha256(cache_usable):
        raise ValueError("canonical acquisition cache-session hash mismatch")
    if cache_usable != available:
        raise ValueError(
            "canonical acquisition usable-session IDs differ from its root catalogue"
        )
    return expected


def _derive_output_selection_contract(
    *,
    expected_session_ids: list[str],
    selected_session_ids: list[str],
    results: list[dict[str, Any]],
    subjects_filter_applied: bool,
    canonical_acquisition_contract: Any,
) -> dict[str, Any]:
    """Derive SVD cohort scope and scientific eligibility from bound evidence."""

    expected = _unique_session_ids(
        expected_session_ids,
        location="SVD expected session IDs",
    )
    selected = _unique_session_ids(
        selected_session_ids,
        location="SVD selected session IDs",
    )
    if type(subjects_filter_applied) is not bool:
        raise ValueError("subjects_filter_applied must be an explicit boolean")
    if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
        raise ValueError("SVD results must be a list of mappings")
    result_ids = _unique_session_ids(
        [item.get("session_id") for item in results],
        location="SVD result session IDs",
    )
    if result_ids != selected:
        raise RuntimeError("SVD result catalogue differs from the selected session IDs")

    selection_is_full = bool(
        not subjects_filter_applied and selected == expected
    )
    selection_scope = "full_cohort" if selection_is_full else "diagnostic_subset"
    all_results_ok = bool(results) and all(
        item.get("status") == "ok" for item in results
    )

    acquisition_v2 = bool(
        isinstance(canonical_acquisition_contract, dict)
        and canonical_acquisition_contract.get("schema_version")
        == "snn_rr.feature_cache_acquisition.v2"
    )
    canonical_full = True
    canonical_scientific = False
    session_bindings_scientific = False
    if acquisition_v2:
        assert isinstance(canonical_acquisition_contract, dict)
        canonical_expected = _unique_session_ids(
            canonical_acquisition_contract.get("expected_usable_session_ids"),
            location="canonical acquisition expected usable-session IDs",
        )
        canonical_cache = _unique_session_ids(
            canonical_acquisition_contract.get("cache_usable_session_ids"),
            location="canonical acquisition cache usable-session IDs",
        )
        if canonical_expected != expected:
            raise ValueError("SVD/canonical acquisition expected-session IDs mismatch")
        if canonical_acquisition_contract.get(
            "expected_usable_session_ids_sha256"
        ) != _canonical_sha256(canonical_expected):
            raise ValueError("canonical acquisition expected-session hash mismatch")
        if canonical_acquisition_contract.get(
            "cache_usable_session_ids_sha256"
        ) != _canonical_sha256(canonical_cache):
            raise ValueError("canonical acquisition cache-session hash mismatch")
        canonical_full = bool(
            canonical_acquisition_contract.get("selection_scope") == "full_cohort"
            and canonical_acquisition_contract.get("full_cohort_complete") is True
            and canonical_acquisition_contract.get(
                "reconstruction_full_cohort_complete"
            )
            is True
            and canonical_cache == canonical_expected
        )
        canonical_scientific = bool(
            canonical_full
            and canonical_acquisition_contract.get("mode") == "strict"
            and canonical_acquisition_contract.get("scientific_eligible") is True
        )
        session_bindings_scientific = bool(
            results
            and all(
                isinstance(item.get("canonical_acquisition_binding"), dict)
                and item["canonical_acquisition_binding"].get("schema_version")
                == "snn_rr.feature_cache_acquisition.v2"
                and item["canonical_acquisition_binding"].get(
                    "scientific_eligible"
                )
                is True
                for item in results
            )
        )

    full_cohort_complete = bool(
        selection_is_full and all_results_ok and canonical_full
    )
    scientific_eligible = bool(
        acquisition_v2
        and full_cohort_complete
        and canonical_scientific
        and session_bindings_scientific
    )
    return {
        "expected_session_ids": expected,
        "expected_session_ids_sha256": _canonical_sha256(expected),
        "selected_session_ids": selected,
        "selected_session_ids_sha256": _canonical_sha256(selected),
        "subjects_filter_applied": subjects_filter_applied,
        "selection_scope": selection_scope,
        "full_cohort_complete": full_cohort_complete,
        "scientific_eligible": scientific_eligible,
    }


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
        "canonical_root_manifest_sha256",
        "canonical_source_fingerprint",
        "canonical_session_manifest_sha256",
        "canonical_acquisition_session_manifest_sha256",
        "canonical_acquisition_binding",
        "acquisition_reconstruction_content_sha256",
        "dataset_root",
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
    canonical_root_manifest_path = canonical_dir.parent / "manifest.json"
    if _sha256(canonical_root_manifest_path) != task[
        "canonical_root_manifest_sha256"
    ]:
        raise RuntimeError(
            f"{session_id}: canonical root manifest changed after task creation"
        )
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
    acquisition_v2 = bool(
        isinstance(observed_acquisition_binding, dict)
        and observed_acquisition_binding.get("schema_version")
        == "snn_rr.feature_cache_acquisition.v2"
    )
    if acquisition_v2:
        reconstruction_path_value = task.get("acquisition_reconstruction_manifest")
        if not isinstance(reconstruction_path_value, str):
            raise RuntimeError(
                f"{session_id}: acquisition reconstruction path is not bound"
            )
        reconstruction = load_acquisition_reconstruction(
            Path(reconstruction_path_value)
        )
        if reconstruction.content_sha256 != task.get(
            "acquisition_reconstruction_content_sha256"
        ):
            raise RuntimeError(
                f"{session_id}: acquisition reconstruction content hash changed"
            )
        session_contract = reconstruction.sessions.get(session_id)
        if session_contract is None or session_contract.content_sha256 != task[
            "canonical_acquisition_session_manifest_sha256"
        ]:
            raise RuntimeError(
                f"{session_id}: acquisition session binding changed"
            )
        # This occurs before the output-cache fast path.  A same-size,
        # same-mtime raw mutation therefore cannot reuse stale SVD features.
        validate_raw_input_bindings(session_contract, Path(task["dataset_root"]))

    metadata_path = (
        _verify_bound_canonical_file(
            canonical_dir,
            canonical_manifest,
            "metadata",
            "metadata.csv",
        )
        if acquisition_v2
        else canonical_dir / "metadata.csv"
    )
    timing_mask_path = (
        _verify_bound_canonical_file(
            canonical_dir,
            canonical_manifest,
            "radar_timing_valid_mask",
            "radar_timing_valid_mask.npy",
        )
        if acquisition_v2
        else None
    )
    signature = _session_signature(task)
    complete = [
        output_dir / "spectra.npy",
        output_dir / "component_signals.npy",
        output_dir / "attributes.npy",
        output_dir / "frequencies_hz.npy",
        output_dir / "metadata.csv",
        output_dir / "manifest.json",
    ]
    if acquisition_v2:
        complete.append(output_dir / "radar_timing_valid_mask.npy")
    if not task["force"] and all(path.is_file() for path in complete):
        previous = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("session_signature") == signature and (
            _cached_svd_manifest_is_current(
                output_dir,
                previous,
                acquisition_v2=acquisition_v2,
            )
        ):
            return {"session_id": session_id, "status": "ok", "cached": True, **previous}

    metadata = pd.read_csv(metadata_path)
    if task["valid_only"]:
        selected_local = np.flatnonzero(metadata["reference_valid"].to_numpy(dtype=bool))
    else:
        selected_local = np.arange(len(metadata), dtype=np.int64)
    if not len(selected_local):
        return {"session_id": session_id, "status": "skipped", "reason": "no selected rows"}
    selected_metadata = metadata.iloc[selected_local].copy().reset_index(drop=True)
    selected_metadata.insert(0, "cache_index", int(task["cache_offset"]) + selected_local)
    selected_timing_masks: np.ndarray | None = None
    if acquisition_v2:
        assert timing_mask_path is not None
        canonical_timing_masks = np.load(timing_mask_path, allow_pickle=False)
        if (
            canonical_timing_masks.dtype != np.bool_
            or canonical_timing_masks.shape != (len(metadata), 3, 320)
        ):
            raise RuntimeError(
                f"{session_id}: canonical timing mask shape/dtype is invalid"
            )
        selected_timing_masks = np.asarray(
            canonical_timing_masks[selected_local], dtype=np.bool_
        )

    radar_arrays: list[np.ndarray] = []
    outlier_counts: list[int] = []
    relative_times: list[np.ndarray] = []
    frame_sequences: list[np.ndarray] = []
    timestamp_sources: list[str] = []
    for recording_dir in task["recording_dirs"]:
        recording = load_xethru_recording(recording_dir, strict=True)
        values, outlier_count = replace_radar_outliers(
            np.asarray(recording.records["bins"], dtype=np.float32)
        )
        radar_arrays.append(values)
        outlier_counts.append(int(outlier_count))
        relative_times.append(
            np.asarray(recording.timestamps_ms, dtype=np.float64) / 1000.0
        )
        frame_sequences.append(
            np.asarray(recording.records["frame_sequence"], dtype=np.uint32)
        )
        timestamp_sources.append(
            recording.meta.timestamp_source
            if recording.meta is not None
            else "fallback_40hz"
        )
    measured_resampling = None
    if acquisition_v2:
        bound_summary = canonical_manifest.get("radar_timing_summary")
        if not isinstance(bound_summary, dict):
            raise RuntimeError(
                f"{session_id}: canonical cache lacks measured timing summary"
            )
        measured_resampling = causal_uniform_resample_radar_views_v1(
            radar_arrays,
            relative_times,
            task["radar_start_epochs_s"],
            frame_sequences,
            output_hz=10.0,
            max_gap_s=0.050,
            gap_policy=str(bound_summary.get("gap_policy")),
            timestamp_sources=timestamp_sources,
            require_measured_timestamps=bool(
                observed_acquisition_binding.get("measured_timing_eligible")
            ),
        )
        if json.dumps(
            measured_resampling.summary,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ) != json.dumps(
            bound_summary,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ):
            raise RuntimeError(
                f"{session_id}: measured SVD timing differs from canonical cache"
            )
        if canonical_manifest.get("radar_outlier_replacements") != outlier_counts:
            raise RuntimeError(
                f"{session_id}: SVD outlier preprocessing differs from canonical cache"
            )
        radar_arrays = [values for values in measured_resampling.values]
    else:
        radar_arrays = [causal_block_mean(values, 4) for values in radar_arrays]
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
            if acquisition_v2:
                assert measured_resampling is not None
                first_left = float(
                    measured_resampling.times_s[0]
                    - measured_resampling.interval_s
                )
                requested_start = float(row["radar_window_start_relative_s"])
                start = int(
                    round(
                        (requested_start - first_left)
                        / measured_resampling.interval_s
                    )
                )
                reconstructed_start = (
                    first_left + start * measured_resampling.interval_s
                )
                if abs(reconstructed_start - requested_start) > 1e-8:
                    raise RuntimeError(
                        f"{session_id} row {output_row}: measured start is off grid"
                    )
                requested_end = float(row["radar_window_end_relative_s"])
                if start + 320 > common or abs(
                    float(measured_resampling.times_s[start + 319]) - requested_end
                ) > 1e-8:
                    raise RuntimeError(
                        f"{session_id} row {output_row}: measured 32 s support mismatch"
                    )
            else:
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
            if acquisition_v2:
                assert measured_resampling is not None
                assert selected_timing_masks is not None
                reconstructed_mask = np.asarray(
                    measured_resampling.valid_mask[:, start : start + 320],
                    dtype=np.bool_,
                )
                if not np.array_equal(
                    reconstructed_mask, selected_timing_masks[output_row]
                ):
                    raise RuntimeError(
                        f"{session_id} row {output_row}: timing mask differs from "
                        "canonical support"
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
    if selected_timing_masks is not None:
        _atomic_save(
            output_dir / "radar_timing_valid_mask.npy",
            selected_timing_masks,
        )
    metadata_temporary = output_dir / "metadata.csv.tmp"
    selected_metadata.to_csv(metadata_temporary, index=False)
    metadata_temporary.replace(output_dir / "metadata.csv")
    file_inventory = {
        "spectra": _array_inventory(output_dir / "spectra.npy", spectra),
        "component_signals": _array_inventory(
            output_dir / "component_signals.npy", component_signals
        ),
        "attributes": _array_inventory(
            output_dir / "attributes.npy", attributes
        ),
        "frequencies_hz": _array_inventory(
            output_dir / "frequencies_hz.npy", sample.frequencies_hz
        ),
        "metadata": _metadata_inventory(
            output_dir / "metadata.csv", selected_metadata
        ),
    }
    if selected_timing_masks is not None:
        file_inventory["radar_timing_valid_mask"] = _array_inventory(
            output_dir / "radar_timing_valid_mask.npy",
            selected_timing_masks,
        )
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
        "radar_timing_valid_mask_shape": (
            None
            if selected_timing_masks is None
            else list(selected_timing_masks.shape)
        ),
        "radar_timing_invalid_interval_count": (
            None
            if selected_timing_masks is None
            else int(np.size(selected_timing_masks) - selected_timing_masks.sum())
        ),
        "radar_timing_mask_contract": (
            None
            if selected_timing_masks is None
            else {
                "mask_required_for_gap_tolerant_consumers": True,
                "scientific_source_requires_all_true": True,
                "diagnostic_output_trainable": False,
                "invalid_cells_are_exact_zero_but_not_semantic_measurements": True,
            }
        ),
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
            (
                "measured_causal_10hz_radar_grid"
                if acquisition_v2
                else "legacy_nominal_10hz_radar_grid"
            ),
            "canonical_radar_and_biopac_timestamps_for_alignment",
        ],
        "causality_scope": {
            "window_end_prediction_uses_only_past_32s": True,
            "within_window_prefix_causal_representation": False,
            "reason": (
                "detrending, standardization, and randomized SVD are fitted over "
                "the complete 32 s window; an early component sample may depend "
                "on later samples inside that already-observed window"
            ),
            "streaming_prefix_causality_claim_allowed": False,
        },
        "radar_timing_summary": (
            None if measured_resampling is None else measured_resampling.summary
        ),
        "file_inventory": file_inventory,
        "inventory_sha256": _canonical_sha256(file_inventory),
        "label_inputs": (
            ["canonical_metadata.reference_valid (row_selection_only)"]
            if task["valid_only"]
            else []
        ),
        "feature_value_label_inputs": [],
        "target_dependent_row_selection": bool(task["valid_only"]),
        "split_half_layout_status": "unverified_hypothesis",
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)
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
    canonical_root_manifest_sha256 = _sha256(canonical_root / "manifest.json")
    dataset_manifest = build_dataset_manifest(dataset_root)
    by_session = {subject.subject_id: subject for subject in dataset_manifest.usable_subjects}
    canonical_available = [
        item for item in canonical_root_manifest["sessions"] if item["status"] == "ok"
    ]
    canonical_available_session_ids = _unique_session_ids(
        [item.get("session_id") for item in canonical_available],
        location="canonical cache usable-session catalogue",
    )
    root_acquisition_binding = canonical_root_manifest.get("acquisition_contract")
    expected_session_ids = _canonical_expected_session_ids(
        canonical_available_session_ids,
        root_acquisition_binding,
    )
    subjects_filter_applied = args.subjects is not None
    available = list(canonical_available)
    if subjects_filter_applied:
        wanted = set(args.subjects)
        available = [item for item in available if item["session_id"] in wanted]
        missing = wanted - {item["session_id"] for item in available}
        if missing:
            raise KeyError(f"unknown or unusable sessions: {sorted(missing)}")
    if not available:
        raise ValueError("canonical cache contains no selected usable sessions")
    selected_session_ids = _unique_session_ids(
        [item.get("session_id") for item in available],
        location="selected canonical cache sessions",
    )

    acquisition_v2 = bool(
        isinstance(root_acquisition_binding, dict)
        and root_acquisition_binding.get("schema_version")
        == "snn_rr.feature_cache_acquisition.v2"
    )
    # One call validates the entire v2 reconstruction/cache graph and every
    # content-addressed session inventory.  Loading one selected session as a
    # memory map avoids concatenating the full cache while retaining the
    # fail-closed graph check.
    load_feature_cache(
        canonical_root,
        sessions=[str(available[0]["session_id"])],
        mmap=True,
        require_acquisition_contract=acquisition_v2,
    )
    acquisition_reconstruction_manifest: Path | None = None
    acquisition_reconstruction_content_sha256: str | None = None
    verified_reconstruction = None
    if acquisition_v2:
        assert isinstance(root_acquisition_binding, dict)
        reconstruction_value = root_acquisition_binding.get(
            "reconstruction_manifest"
        )
        if not isinstance(reconstruction_value, str) or not reconstruction_value:
            raise ValueError(
                "canonical acquisition cache lacks a reconstruction manifest path"
            )
        acquisition_reconstruction_manifest = Path(reconstruction_value)
        if not acquisition_reconstruction_manifest.is_absolute():
            acquisition_reconstruction_manifest = (
                canonical_root / acquisition_reconstruction_manifest
            )
        acquisition_reconstruction_manifest = (
            acquisition_reconstruction_manifest.resolve()
        )
        verified_reconstruction = load_acquisition_reconstruction(
            acquisition_reconstruction_manifest
        )
        acquisition_reconstruction_content_sha256 = (
            verified_reconstruction.content_sha256
        )
        if acquisition_reconstruction_content_sha256 != root_acquisition_binding.get(
            "reconstruction_content_sha256"
        ):
            raise ValueError(
                "canonical cache/reconstruction content hash mismatch"
            )

    pipeline_paths = _pipeline_paths()
    pipeline_digest = _pipeline_sha256(pipeline_paths)
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
                "radar_start_epochs_s": [
                    float(subject.selected_session.radars[radar].start_epoch_ms)
                    / 1000.0
                    for radar in (1, 2, 3)
                ],
                "canonical_dir": str(canonical_dir),
                "canonical_root_manifest_sha256": canonical_root_manifest_sha256,
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
                "acquisition_reconstruction_manifest": (
                    None
                    if acquisition_reconstruction_manifest is None
                    else str(acquisition_reconstruction_manifest)
                ),
                "acquisition_reconstruction_content_sha256": (
                    acquisition_reconstruction_content_sha256
                ),
                "dataset_root": str(dataset_root),
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
    _assert_pipeline_unchanged(pipeline_paths, pipeline_digest)
    if _sha256(canonical_root / "manifest.json") != canonical_root_manifest_sha256:
        raise RuntimeError(
            "canonical cache root changed while the SVD cache was being built"
        )
    final_reconstruction = None
    if acquisition_v2:
        assert acquisition_reconstruction_manifest is not None
        final_reconstruction = load_acquisition_reconstruction(
            acquisition_reconstruction_manifest
        )
        if final_reconstruction.content_sha256 != (
            acquisition_reconstruction_content_sha256
        ):
            raise RuntimeError(
                "acquisition reconstruction changed while SVD was being built"
            )
    result_by_session = {
        str(item["session_id"]): item for item in results
    }
    for task in tasks:
        session_id = str(task["session_id"])
        canonical_dir = Path(task["canonical_dir"])
        session_manifest_path = canonical_dir / "manifest.json"
        if _sha256(session_manifest_path) != task[
            "canonical_session_manifest_sha256"
        ]:
            raise RuntimeError(
                f"{session_id}: canonical session changed while SVD was being built"
            )
        if acquisition_v2:
            current_session_manifest = json.loads(
                session_manifest_path.read_text(encoding="utf-8")
            )
            _verify_bound_canonical_file(
                canonical_dir,
                current_session_manifest,
                "metadata",
                "metadata.csv",
            )
            _verify_bound_canonical_file(
                canonical_dir,
                current_session_manifest,
                "radar_timing_valid_mask",
                "radar_timing_valid_mask.npy",
            )
            assert final_reconstruction is not None
            session_contract = final_reconstruction.sessions.get(session_id)
            if session_contract is None:
                raise RuntimeError(
                    f"{session_id}: acquisition session disappeared during SVD build"
                )
            validate_raw_input_bindings(session_contract, dataset_root)
        result = result_by_session.get(session_id)
        if result is None:
            raise RuntimeError(f"{session_id}: SVD task result is missing")
        if result.get("status") == "ok":
            output_dir = Path(task["output_dir"])
            output_manifest_path = output_dir / "manifest.json"
            if not output_manifest_path.is_file():
                raise RuntimeError(
                    f"{session_id}: SVD output manifest is missing at publication"
                )
            output_manifest = json.loads(
                output_manifest_path.read_text(encoding="utf-8")
            )
            if output_manifest.get("session_signature") != _session_signature(task):
                raise RuntimeError(
                    f"{session_id}: SVD output signature changed before publication"
                )
            if not _cached_svd_manifest_is_current(
                output_dir,
                output_manifest,
                acquisition_v2=acquisition_v2,
            ):
                raise RuntimeError(
                    f"{session_id}: SVD output inventory changed before publication"
                )
            if result.get("content_sha256") != output_manifest.get(
                "content_sha256"
            ):
                raise RuntimeError(
                    f"{session_id}: in-memory/output SVD manifest mismatch"
                )
    selection_contract = _derive_output_selection_contract(
        expected_session_ids=expected_session_ids,
        selected_session_ids=selected_session_ids,
        results=results,
        subjects_filter_applied=subjects_filter_applied,
        canonical_acquisition_contract=root_acquisition_binding,
    )
    root_manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "canonical_cache": str(canonical_root),
        "canonical_manifest_sha256": canonical_root_manifest_sha256,
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
        "target_dependent_row_selection": bool(not args.all_windows),
        "row_selection_label_input": (
            "canonical_metadata.reference_valid"
            if not args.all_windows
            else None
        ),
        "variant_names": list(DEFAULT_SVD_VARIANTS),
        "attribute_names": list(ATTRIBUTE_NAMES),
        "components": int(args.components),
        "nfft": int(args.nfft),
        "n_iter": int(args.n_iter),
        "row_count": int(sum(item.get("row_count", 0) for item in results)),
        "sessions": results,
        **selection_contract,
    }
    _assert_pipeline_unchanged(pipeline_paths, pipeline_digest)
    root_manifest["content_sha256"] = _canonical_sha256(root_manifest)
    _write_json(output_root / "manifest.json", root_manifest)
    print(
        f"Built {sum(item['status'] == 'ok' for item in results)} sessions and "
        f"{root_manifest['row_count']} rows in {output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
