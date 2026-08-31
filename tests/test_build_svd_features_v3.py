from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_svd_features.py"
_SPEC = importlib.util.spec_from_file_location("build_svd_features_v3_tests", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SVD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SVD
_SPEC.loader.exec_module(_SVD)


def _write_json(path: Path, value: dict[str, Any]) -> bytes:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _content_address(value: dict[str, Any]) -> dict[str, Any]:
    document = copy.deepcopy(value)
    document["content_sha256"] = ""
    document["content_sha256"] = _SVD._canonical_content_sha256(document)
    return document


def _canonical_root_manifest(
    *,
    root_schema: str | None = None,
    contract_schema: str | None = None,
    session_schemas: list[str] | None = None,
) -> dict[str, Any]:
    return _content_address(
        {
            "schema_version": root_schema
            or _SVD.ACQUISITION_CACHE_ROOT_SCHEMA_VERSION_V3,
            "acquisition_contract": {
                "schema_version": contract_schema
                or _SVD.ACQUISITION_CACHE_SCHEMA_VERSION_V3,
            },
            "sessions": [
                {"schema_version": schema}
                for schema in (
                    session_schemas
                    if session_schemas is not None
                    else [_SVD.ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3]
                )
            ],
        }
    )


def _v3_task() -> dict[str, Any]:
    return {
        "session_id": "S01_TEST",
        "acquisition_v3": True,
        "pipeline_sha256": "1" * 64,
        "canonical_root_manifest_sha256": "2" * 64,
        "canonical_root_content_sha256": "3" * 64,
        "canonical_source_fingerprint": "source-fingerprint",
        "canonical_session_manifest_sha256": "4" * 64,
        "canonical_session_content_sha256": "5" * 64,
        "canonical_acquisition_session_manifest_sha256": "6" * 64,
        "canonical_acquisition_binding": {
            "schema_version": _SVD.ACQUISITION_CACHE_SCHEMA_VERSION_V3,
            "reconstruction_content_sha256": "7" * 64,
        },
        "acquisition_reconstruction_content_sha256": "8" * 64,
        "canonical_reconstruction_file_sha256": "9" * 64,
        "canonical_reconstruction_content_sha256": "a" * 64,
        "canonical_upstream_session_content_sha256": "b" * 64,
        "canonical_raw_portable_content_sha256": "c" * 64,
        "raw_graph_sha256": "d" * 64,
        "dataset_catalogue_sha256": "0" * 64,
        "dataset_root": "/private/raw",
        "selected_rows_sha256": "e" * 64,
        "valid_only": False,
        "reference_mapping_available": False,
        "declared_window_count": 17,
        "cache_offset": 101,
        "components": 2,
        "nfft": 64,
        "n_iter": 1,
        "variant_names": ["raw"],
    }


def _v3_args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "all_windows": True,
        "subjects": None,
        "force": False,
        "components": 1,
        "nfft": 64,
        "n_iter": 1,
        "workers": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _call_v3_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    args: argparse.Namespace,
    dataset_root: Path | None = None,
    canonical_root: Path | None = None,
    output_root: Path | None = None,
) -> int:
    raw = dataset_root or (tmp_path / "raw")
    canonical = canonical_root or (tmp_path / "canonical")
    output = output_root or (tmp_path / "output")

    def forbidden_deep_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("V3 preflight gate reached canonical data materialization")

    monkeypatch.setattr(_SVD, "load_feature_cache", forbidden_deep_read)
    return _SVD._main_v3(
        args,
        dataset_root=raw,
        canonical_root=canonical,
        output_root=output,
        canonical_root_manifest={},
        canonical_root_payload=b"{}\n",
        canonical_root_manifest_sha256="0" * 64,
    )


def _valid_output_manifest(root: Path) -> dict[str, Any]:
    required = {
        "spectra": "spectra.npy",
        "component_signals": "component_signals.npy",
        "attributes": "attributes.npy",
        "frequencies_hz": "frequencies_hz.npy",
        "metadata": "metadata.csv",
        "radar_timing_valid_mask": "radar_timing_valid_mask.npy",
        "radar_timing_invalid_reason_mask": (
            "radar_timing_invalid_reason_mask.npy"
        ),
        "radar_view_availability_mask": "radar_view_availability_mask.npy",
        "raw_consumption_receipt": "raw_consumption_receipt.json",
    }
    inventory: dict[str, Any] = {}
    root.mkdir(mode=0o700)
    for index, (logical_name, filename) in enumerate(required.items()):
        path = root / filename
        path.write_bytes(f"payload-{index}".encode("ascii"))
        os.chmod(path, 0o600)
        inventory[logical_name] = _SVD._v3_file_inventory(
            path,
            shape=None,
            dtype="synthetic",
        )
    manifest: dict[str, Any] = {
        "schema_version": _SVD.V3_SVD_SESSION_SCHEMA,
        "content_sha256": "",
        "pipeline_sha256": "1" * 64,
        "execution_source_generation": _SVD._v3_execution_source_generation(
            "1" * 64
        ),
        "diagnostic_only": True,
        "scientific_eligible": False,
        "training_authorized": False,
        "file_inventory": inventory,
        "inventory_sha256": _SVD._canonical_sha256(inventory),
    }
    manifest["content_sha256"] = _SVD._canonical_content_sha256(manifest)
    return manifest


def test_v3_schema_detection_accepts_only_exact_coherent_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "canonical"
    manifest = _canonical_root_manifest()
    payload = _write_json(root / "manifest.json", manifest)

    detected, observed, observed_payload, file_sha256 = (
        _SVD._detect_v3_canonical_cache(root)
    )

    assert detected is True
    assert observed == manifest
    assert observed_payload == payload
    assert file_sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("root_schema", "contract_schema", "session_schemas"),
    [
        (
            _SVD.ACQUISITION_CACHE_ROOT_SCHEMA_VERSION_V3,
            "snn_rr.feature_cache_acquisition.v2",
            [_SVD.ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3],
        ),
        (
            "snn_rr.feature_cache_root.v2",
            _SVD.ACQUISITION_CACHE_SCHEMA_VERSION_V3,
            [_SVD.ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3],
        ),
        (
            _SVD.ACQUISITION_CACHE_ROOT_SCHEMA_VERSION_V3,
            _SVD.ACQUISITION_CACHE_SCHEMA_VERSION_V3,
            [
                _SVD.ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3,
                "snn_rr.feature_cache_session.v2",
            ],
        ),
        (
            "snn_rr.feature_cache_root.v3.lookalike",
            "snn_rr.feature_cache_acquisition.v2",
            ["snn_rr.feature_cache_session.v2"],
        ),
        (
            "snn_rr.feature_cache_root.v2",
            "snn_rr.feature_cache_acquisition.v2",
            [_SVD.ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3],
        ),
    ],
)
def test_v3_schema_detection_rejects_mixed_or_legacy_laundering(
    tmp_path: Path,
    root_schema: str,
    contract_schema: str,
    session_schemas: list[str],
) -> None:
    root = tmp_path / "canonical"
    _write_json(
        root / "manifest.json",
        _canonical_root_manifest(
            root_schema=root_schema,
            contract_schema=contract_schema,
            session_schemas=session_schemas,
        ),
    )

    with pytest.raises(
        ValueError,
        match="mixed or lookalike version-3 canonical cache schemas are forbidden",
    ):
        _SVD._detect_v3_canonical_cache(root)


def test_pure_legacy_schema_is_not_misclassified_as_v3(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    manifest = _canonical_root_manifest(
        root_schema="snn_rr.feature_cache_root.v2",
        contract_schema="snn_rr.feature_cache_acquisition.v2",
        session_schemas=["snn_rr.feature_cache_session.v2"],
    )
    _write_json(root / "manifest.json", manifest)

    detected, observed, _payload, _digest = _SVD._detect_v3_canonical_cache(root)

    assert detected is False
    assert observed == manifest


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (_v3_args(all_windows=False), "requires --all-windows"),
        (_v3_args(subjects=["S01_TEST"]), "forbids subject subsets"),
        (_v3_args(force=True), "forbids --force"),
    ],
)
def test_v3_cli_scope_gates_fail_before_data_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: argparse.Namespace,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _call_v3_preflight(tmp_path, monkeypatch, args=args)


@pytest.mark.parametrize("protected_name", ["raw", "canonical"])
def test_v3_output_overlap_gate_fails_before_data_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_name: str,
) -> None:
    raw = tmp_path / "raw"
    canonical = tmp_path / "canonical"
    protected = raw if protected_name == "raw" else canonical
    with pytest.raises(ValueError, match="must be disjoint"):
        _call_v3_preflight(
            tmp_path,
            monkeypatch,
            args=_v3_args(),
            dataset_root=raw,
            canonical_root=canonical,
            output_root=protected / "nested-output",
        )


def test_v3_existing_output_is_a_strict_no_clobber_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "already-present"
    output.mkdir()
    marker = output / "owner-data"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        _call_v3_preflight(
            tmp_path,
            monkeypatch,
            args=_v3_args(),
            output_root=output,
        )

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_v3_current_manifest_requires_all_timing_and_availability_payloads(
    tmp_path: Path,
) -> None:
    manifest = _valid_output_manifest(tmp_path / "session")
    assert _SVD._v3_output_manifest_is_current(tmp_path / "session", manifest)

    for key in (
        "radar_timing_valid_mask",
        "radar_timing_invalid_reason_mask",
        "radar_view_availability_mask",
    ):
        tampered = copy.deepcopy(manifest)
        tampered["file_inventory"].pop(key)
        tampered["inventory_sha256"] = _SVD._canonical_sha256(
            tampered["file_inventory"]
        )
        tampered["content_sha256"] = _SVD._canonical_content_sha256(tampered)
        assert not _SVD._v3_output_manifest_is_current(
            tmp_path / "session", tampered
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("diagnostic_only", False),
        ("scientific_eligible", True),
        ("training_authorized", True),
    ],
)
def test_v3_current_manifest_terminal_authority_is_fail_closed(
    tmp_path: Path,
    field: str,
    value: bool,
) -> None:
    manifest = _valid_output_manifest(tmp_path / "session")
    manifest[field] = value
    manifest["content_sha256"] = _SVD._canonical_content_sha256(manifest)

    assert not _SVD._v3_output_manifest_is_current(tmp_path / "session", manifest)


def test_v3_session_masks_unavailable_view_features_to_exact_positive_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_root = tmp_path / "canonical"
    session_id = "S01_TEST"
    output_dir = tmp_path / "staging" / session_id
    output_dir.parent.mkdir(mode=0o700)

    resampling_summary = {"schema_version": "synthetic.resampling.v1"}
    upstream = _content_address(
        {
            "raw_consumption": {"portable_content_sha256": "b" * 64},
            "sensor_summary": {
                "radar": {
                    "feature_resampling": resampling_summary,
                    "past_only_outlier_replacements": [0, 0, 0],
                }
            },
        }
    )
    canonical_session = _content_address(
        {
            "schema_version": _SVD.ACQUISITION_CACHE_SESSION_SCHEMA_VERSION_V3,
            "session_id": session_id,
            "upstream_session_content_sha256": upstream["content_sha256"],
            "upstream_session_contract": upstream,
            "reference_mapping_available": False,
            "radar_timing_invalid_reason_schema_version": (
                _SVD.CAUSAL_UNIFORM_INVALID_REASON_SCHEMA_V1
            ),
            "radar_timing_invalid_reason_semantics_sha256": (
                _SVD.CAUSAL_UNIFORM_INVALID_REASON_SEMANTICS_SHA256_V1
            ),
        }
    )
    root_manifest = _content_address(
        {
            "schema_version": _SVD.ACQUISITION_CACHE_ROOT_SCHEMA_VERSION_V3,
            "acquisition_contract": {
                "schema_version": _SVD.ACQUISITION_CACHE_SCHEMA_VERSION_V3,
            },
        }
    )
    root_payload = _write_json(canonical_root / "manifest.json", root_manifest)
    session_payload = _write_json(
        canonical_root / session_id / "manifest.json", canonical_session
    )

    timing_valid = np.ones((1, 3, 320), dtype=np.bool_)
    timing_valid[0, 1, 17] = False
    timing_reasons = np.zeros((1, 3, 320), dtype=np.uint8)
    timing_reasons[0, 1, 17] = 1
    metadata = pd.DataFrame(
        {
            "session_id": [session_id],
            "window_number": [0],
            "radar_window_start_relative_s": [0.0],
            "radar_window_end_relative_s": [32.0],
            "reference_mapping_available": [False],
            "reference_valid": [False],
        }
    )
    provenance = SimpleNamespace(
        classification="acquisition_diagnostic",
        scientific_eligible=False,
        root_manifest_sha256=hashlib.sha256(root_payload).hexdigest(),
        root_manifest_content_sha256=root_manifest["content_sha256"],
        reconstruction_content_sha256="c" * 64,
    )
    canonical_cache = SimpleNamespace(
        provenance=provenance,
        radar_timing_valid_mask=timing_valid,
        radar_timing_invalid_reason_mask=timing_reasons,
        metadata=metadata,
    )
    monkeypatch.setattr(_SVD, "load_feature_cache", lambda *_a, **_k: canonical_cache)

    receipt_document = {"content_sha256": "d" * 64}
    receipt = SimpleNamespace(to_dict=lambda: receipt_document)
    recording = SimpleNamespace(
        bins=np.ones((320, 2), dtype=np.float32),
        timestamps_ms=np.arange(320, dtype=np.float64) * 100.0,
        frame_sequence=np.arange(320, dtype=np.uint32),
        meta=SimpleNamespace(start_epoch_ms=1000.0, timestamp_source="measured"),
    )
    loaded = SimpleNamespace(radars={1: recording, 2: recording, 3: recording}, receipt=receipt)
    monkeypatch.setattr(
        _SVD,
        "RawSessionReader",
        lambda *_a, **_k: SimpleNamespace(consume=lambda: loaded),
    )
    monkeypatch.setattr(
        _SVD,
        "validate_consumption_against_contract",
        lambda *_a, **_k: {
            "portable_content_sha256": "b" * 64,
            "diagnostic_only": True,
            "scientific_authority": False,
        },
    )
    monkeypatch.setattr(
        _SVD,
        "replace_radar_outliers",
        lambda values: (np.asarray(values, dtype=np.float32), 0),
    )
    measured = SimpleNamespace(
        values=[
            np.ones((320, 2), dtype=np.float32),
            np.ones((320, 2), dtype=np.float32) * 2,
            np.ones((320, 2), dtype=np.float32) * 3,
        ],
        times_s=np.arange(1, 321, dtype=np.float64) / 10.0,
        interval_s=0.1,
        valid_mask=timing_valid[0],
        invalid_reason_mask=timing_reasons[0],
        summary=resampling_summary,
    )
    monkeypatch.setattr(
        _SVD,
        "causal_uniform_resample_radar_views_v1",
        lambda *_a, **_k: measured,
    )

    feature = SimpleNamespace(
        spectra=np.full((1, 1, 2), 3.0, dtype=np.float32),
        component_signals=np.full((1, 1, 320), 4.0, dtype=np.float32),
        attributes=np.full(
            (1, 1, len(_SVD.ATTRIBUTE_NAMES)), 5.0, dtype=np.float32
        ),
        frequencies_hz=np.asarray([0.1, 0.2], dtype=np.float32),
    )
    monkeypatch.setattr(_SVD, "svd_component_features", lambda *_a, **_k: feature)

    task = _v3_task()
    task.update(
        {
            "canonical_root": str(canonical_root),
            "canonical_root_manifest_sha256": hashlib.sha256(root_payload).hexdigest(),
            "canonical_root_content_sha256": root_manifest["content_sha256"],
            "canonical_session_manifest_sha256": hashlib.sha256(
                session_payload
            ).hexdigest(),
            "canonical_session_content_sha256": canonical_session["content_sha256"],
            "canonical_upstream_session_content_sha256": upstream["content_sha256"],
            "canonical_raw_portable_content_sha256": "b" * 64,
            "canonical_reconstruction_content_sha256": "c" * 64,
            "acquisition_reconstruction_content_sha256": "c" * 64,
            "dataset_root": str(tmp_path / "raw"),
            "raw_graph": object(),
            "declared_window_count": 1,
            "cache_offset": 0,
            "selected_rows_sha256": hashlib.sha256(
                np.arange(1, dtype=np.int64).tobytes()
            ).hexdigest(),
            "components": 1,
            "nfft": 64,
            "n_iter": 1,
            "variant_names": ["raw"],
            "output_dir": str(output_dir),
        }
    )

    result = _SVD._build_v3_session(task)

    availability = np.load(output_dir / "radar_view_availability_mask.npy")
    spectra = np.load(output_dir / "spectra.npy")
    components = np.load(output_dir / "component_signals.npy")
    attributes = np.load(output_dir / "attributes.npy")
    assert availability.tolist() == [[True, False, True]]
    for values in (spectra[:, 1], components[:, 1], attributes[:, 1]):
        assert np.all(values == 0)
        assert not np.any(np.signbit(values))

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result["scientific_eligible"] is False
    assert result["training_authorized"] is False
    assert manifest["scientific_eligible"] is False
    assert manifest["training_authorized"] is False
    assert manifest["timing_support_contract"][
        "unavailable_view_feature_values_exact_positive_zero"
    ] is True
    assert {
        "radar_timing_valid_mask",
        "radar_timing_invalid_reason_mask",
        "radar_view_availability_mask",
    }.issubset(manifest["file_inventory"])


def test_v3_private_attempt_publication_secures_tree_and_never_replaces(
    tmp_path: Path,
) -> None:
    output = tmp_path / "published"
    attempt = _SVD._start_v3_attempt(
        output,
        canonical_root=tmp_path / "canonical",
        dataset_root=tmp_path / "raw",
    )
    claim = Path(attempt["claim_path"])
    staging = Path(attempt["staging_root"])
    nested = staging / "S01_TEST"
    nested.mkdir(mode=0o755)
    payload = nested / "payload.bin"
    payload.write_bytes(b"private")
    os.chmod(payload, 0o644)

    assert stat.S_IMODE(claim.stat().st_mode) == 0o600
    assert stat.S_IMODE(staging.stat().st_mode) == 0o700
    published = _SVD._publish_v3_attempt(attempt)

    assert published == output
    assert not claim.exists()
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "S01_TEST").stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "S01_TEST" / "payload.bin").stat().st_mode) == 0o600

    source = tmp_path / "second-staging"
    source.mkdir()
    with pytest.raises(FileExistsError):
        _SVD._rename_noreplace(source, output)
    assert source.is_dir()
    assert (output / "S01_TEST" / "payload.bin").read_bytes() == b"private"


def test_v3_failure_receipt_is_private_hashed_and_preserves_staging(
    tmp_path: Path,
) -> None:
    output = tmp_path / "failed-output"
    attempt = _SVD._start_v3_attempt(
        output,
        canonical_root=tmp_path / "canonical",
        dataset_root=tmp_path / "raw",
    )
    staging = Path(attempt["staging_root"])
    evidence = staging / "partial.bin"
    evidence.write_bytes(b"failed-work")
    attempt["completed_session_ids"].append("S01_TEST")

    _SVD._record_v3_failure(attempt, RuntimeError("synthetic failure"))

    failure_path = Path(attempt["failure_path"])
    receipt = json.loads(failure_path.read_text(encoding="utf-8"))
    assert not Path(attempt["claim_path"]).exists()
    assert failure_path.is_file()
    assert stat.S_IMODE(failure_path.stat().st_mode) == 0o600
    assert receipt["schema_version"] == "snn_rr.svd_component_cache_failure.v1"
    assert receipt["terminal_state"] == "failed"
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["error"] == "synthetic failure"
    assert receipt["completed_session_ids"] == ["S01_TEST"]
    assert receipt["content_sha256"] == _SVD._canonical_content_sha256(receipt)
    assert evidence.read_bytes() == b"failed-work"
    assert not output.exists()

    with pytest.raises(FileExistsError):
        _SVD._record_v3_failure(attempt, RuntimeError("second failure"))
    assert json.loads(failure_path.read_text(encoding="utf-8")) == receipt


def test_v3_committed_publication_cannot_emit_precommit_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "published"
    attempt = _SVD._start_v3_attempt(
        output,
        canonical_root=tmp_path / "canonical",
        dataset_root=tmp_path / "raw",
    )
    Path(attempt["staging_root"]).joinpath("payload.bin").write_bytes(b"ok")
    _SVD._publish_v3_attempt(attempt)

    with pytest.raises(RuntimeError, match="after V3 SVD publication"):
        _SVD._record_v3_failure(attempt, RuntimeError("late cleanup"))
    assert output.is_dir()
    assert not Path(attempt["failure_path"]).exists()


def test_legacy_pipeline_digest_surface_excludes_v3_raw_snapshot() -> None:
    legacy = _SVD._pipeline_paths()
    v3 = _SVD._v3_pipeline_paths()
    raw_snapshot = _SVD.SOURCE_ROOT / "snn_rr" / "raw_snapshot.py"

    assert raw_snapshot not in legacy
    assert v3 == [*legacy, raw_snapshot]


def test_v3_session_signature_binds_cache_reconstruction_raw_and_support() -> None:
    baseline_task = _v3_task()
    baseline = _SVD._session_signature(baseline_task)
    mutations: dict[str, Any] = {
        "canonical_source_fingerprint": "different-source-fingerprint",
        "canonical_root_manifest_sha256": "f" * 64,
        "canonical_root_content_sha256": "0" * 64,
        "canonical_session_manifest_sha256": "1" * 64,
        "canonical_session_content_sha256": "2" * 64,
        "canonical_acquisition_session_manifest_sha256": "3" * 64,
        "canonical_upstream_session_content_sha256": "3" * 64,
        "canonical_reconstruction_file_sha256": "4" * 64,
        "canonical_reconstruction_content_sha256": "5" * 64,
        "acquisition_reconstruction_content_sha256": "6" * 64,
        "canonical_raw_portable_content_sha256": "6" * 64,
        "raw_graph_sha256": "7" * 64,
        "dataset_catalogue_sha256": "8" * 64,
        "selected_rows_sha256": "9" * 64,
        "reference_mapping_available": True,
        "declared_window_count": 18,
        "cache_offset": 102,
    }

    for field, value in mutations.items():
        changed = copy.deepcopy(baseline_task)
        changed[field] = value
        assert _SVD._session_signature(changed) != baseline, field

    changed_binding = copy.deepcopy(baseline_task)
    changed_binding["canonical_acquisition_binding"][
        "reconstruction_content_sha256"
    ] = "9" * 64
    assert _SVD._session_signature(changed_binding) != baseline
