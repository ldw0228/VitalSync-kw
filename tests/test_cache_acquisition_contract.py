from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from snn_rr.cache import (
    ACQUISITION_CACHE_SCHEMA_VERSION,
    REQUIRED_ACQUISITION_ANNOTATION_COLUMNS,
    load_feature_cache,
)


def _content_sha256(document: dict[str, object]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _value_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")


def _metadata(session_id: str, *, eligible: bool) -> pd.DataFrame:
    values: dict[str, list[object]] = {
        "session_id": [session_id],
        "reference_start_sample": [100],
        "reference_end_sample": [6500],
        "reference_window_start_biopac_s": [0.5],
        "reference_window_end_biopac_s": [32.5],
        "radar_window_start_relative_s": [1.0],
        "radar_window_end_relative_s": [33.0],
        "sync_authorized": [eligible],
        "sync_confidence": [0.95],
        "alignment_scientific_eligible": [eligible],
        "acquisition_phase": ["phase1"],
        "acquisition_phase_name": ["angled_seated_respiration"],
        "acquisition_phase_status": ["auto"],
        "acquisition_phase_confidence": [0.9],
        "phase_overlap_fraction": [1.0],
        "transition_window": [False],
        "eligible_for_stage_metrics": [eligible],
        "phase7_assignment": ["not_phase7"],
        "acquisition_batch": ["v2"],
    }
    assert REQUIRED_ACQUISITION_ANNOTATION_COLUMNS <= set(values)
    return pd.DataFrame(values)


def _write_cache(tmp_path: Path, *, eligible: bool) -> tuple[Path, Path]:
    session_id = "S01_TEST"
    acquisition_root = tmp_path / "acquisition"
    source_session_path = acquisition_root / "sessions" / session_id / "session_manifest.json"
    mapping = {"mode": "affine", "offset_s": 1.25, "scale": 1.0001}
    sync_receipt_hash = "1" * 64
    approval_hash = "3" * 64 if eligible else None
    range_hash = "4" * 64
    source_session: dict[str, object] = {
        "schema_version": "snn_rr.acquisition_reconstruction.v1",
        "session_id": session_id,
        "scientific_eligible": eligible,
        "synchronization": {
            "authorized": eligible,
            "receipt_content_sha256": sync_receipt_hash,
            "manual_approval_content_sha256": approval_hash,
            "mapping": mapping,
        },
        "range_tracking": {"artifact_sha256": range_hash},
        "protocol_contract": {"schema_version": "acquisition_protocol_v1"},
    }
    source_session["content_sha256"] = _content_sha256(source_session)
    _write_json(source_session_path, source_session)

    reconstruction: dict[str, object] = {
        "schema_version": "snn_rr.acquisition_reconstruction.v1",
        "complete": True,
        "scientific_eligible": eligible,
        "sessions": [
            {
                "session_id": session_id,
                "manifest": f"sessions/{session_id}/session_manifest.json",
                "content_sha256": source_session["content_sha256"],
                "scientific_eligible": eligible,
            }
        ],
    }
    reconstruction["content_sha256"] = _content_sha256(reconstruction)
    reconstruction_path = acquisition_root / "manifest.json"
    _write_json(reconstruction_path, reconstruction)

    annotation_columns = sorted(REQUIRED_ACQUISITION_ANNOTATION_COLUMNS)
    session_contract: dict[str, object] = {
        "schema_version": ACQUISITION_CACHE_SCHEMA_VERSION,
        "acquisition_session_manifest_sha256": source_session["content_sha256"],
        "sync_receipt_content_sha256": sync_receipt_hash,
        "mapping_sha256": _value_sha256(mapping),
        "manual_approval_content_sha256": approval_hash,
        "protocol_annotation_schema_version": "acquisition_protocol_v1",
        "range_artifact_sha256": range_hash,
        "reference_alignment_mode": (
            "authorized_marker_affine_v1"
            if eligible
            else "diagnostic_unapproved_proposal_v1"
        ),
        "scientific_eligible": eligible,
        "annotation_only_columns": annotation_columns,
    }
    root_contract: dict[str, object] = {
        "schema_version": ACQUISITION_CACHE_SCHEMA_VERSION,
        "reconstruction_manifest": str(reconstruction_path),
        "reconstruction_content_sha256": reconstruction["content_sha256"],
        "mode": "strict" if eligible else "diagnostic",
        "annotation_only_columns": annotation_columns,
        "scientific_eligible": eligible,
    }

    cache = tmp_path / "cache"
    session_dir = cache / session_id
    session_dir.mkdir(parents=True)
    np.save(session_dir / "maps.npy", np.zeros((1, 3, 2, 4), dtype=np.float16))
    np.save(session_dir / "aux.npy", np.zeros((1, 2), dtype=np.float32))
    np.save(session_dir / "frequencies_hz.npy", np.asarray([0.1, 0.2]))
    _metadata(session_id, eligible=eligible).to_csv(session_dir / "metadata.csv", index=False)
    _write_json(session_dir / "manifest.json", {"acquisition_contract": session_contract})
    _write_json(
        cache / "manifest.json",
        {
            "acquisition_contract": root_contract,
            "sessions": [
                {
                    "session_id": session_id,
                    "status": "ok",
                    "acquisition_contract": session_contract,
                }
            ],
        },
    )
    return cache, reconstruction_path


def test_legacy_cache_default_behavior_is_unchanged(tmp_path: Path) -> None:
    cache = tmp_path / "legacy"
    session = cache / "LEGACY"
    session.mkdir(parents=True)
    np.save(session / "maps.npy", np.zeros((1, 3, 2, 4), dtype=np.float16))
    np.save(session / "aux.npy", np.zeros((1, 2), dtype=np.float32))
    np.save(session / "frequencies_hz.npy", np.asarray([0.1, 0.2]))
    pd.DataFrame({"session_id": ["LEGACY"]}).to_csv(session / "metadata.csv", index=False)
    _write_json(
        cache / "manifest.json",
        {"sessions": [{"session_id": "LEGACY", "status": "ok"}]},
    )

    loaded = load_feature_cache(cache)
    assert loaded.maps.shape == (1, 3, 2, 4)
    with pytest.raises(ValueError, match="root acquisition_contract"):
        load_feature_cache(cache, require_acquisition_contract=True)


def test_valid_acquisition_contract_and_scientific_gate(tmp_path: Path) -> None:
    eligible_cache, _ = _write_cache(tmp_path / "eligible", eligible=True)
    loaded = load_feature_cache(
        eligible_cache,
        require_acquisition_contract=True,
        require_scientific_eligible=True,
    )
    assert loaded.metadata.loc[0, "acquisition_phase"] == "phase1"

    diagnostic_cache, _ = _write_cache(tmp_path / "diagnostic", eligible=False)
    load_feature_cache(diagnostic_cache, require_acquisition_contract=True)
    with pytest.raises(ValueError, match="root is not scientifically eligible"):
        load_feature_cache(diagnostic_cache, require_scientific_eligible=True)


def test_contract_mixing_annotation_and_source_hash_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    cache, reconstruction_path = _write_cache(tmp_path, eligible=True)
    session_manifest_path = cache / "S01_TEST" / "manifest.json"
    session_manifest = json.loads(session_manifest_path.read_text(encoding="utf-8"))
    del session_manifest["acquisition_contract"]
    _write_json(session_manifest_path, session_manifest)
    with pytest.raises(ValueError, match="legacy session mixed"):
        load_feature_cache(cache, require_acquisition_contract=True)

    cache, reconstruction_path = _write_cache(tmp_path / "columns", eligible=True)
    metadata_path = cache / "S01_TEST" / "metadata.csv"
    frame = pd.read_csv(metadata_path).drop(columns=["acquisition_phase_status"])
    frame.to_csv(metadata_path, index=False)
    with pytest.raises(ValueError, match="metadata columns missing"):
        load_feature_cache(cache, require_acquisition_contract=True)

    cache, reconstruction_path = _write_cache(tmp_path / "hash", eligible=True)
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    reconstruction["complete"] = False
    _write_json(reconstruction_path, reconstruction)
    with pytest.raises(ValueError, match="canonical content hash mismatch"):
        load_feature_cache(cache, require_acquisition_contract=True)
