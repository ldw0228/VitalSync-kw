from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import snn_rr.acquisition_contract as acquisition_contract_module
from snn_rr.acquisition_contract import (
    AcquisitionContractError,
    RAW_CONSUMPTION_BINDING_SCHEMA,
    validate_consumption_against_contract,
    validate_timing_reason_authority,
)
from snn_rr.radar_timing import causal_uniform_resample_radar_views_v1
from snn_rr.data import (
    BiopacParserEvidence,
    XeThruMetaChunkEvidence,
    XeThruMetaEvidence,
)
from snn_rr.raw_snapshot import (
    ConsumedFileBinding,
    FileIdentity,
    RadarChunkEvidence,
    RadarMetadataEvidence,
    RadarRecordEvidence,
    RawConsumptionReceipt,
    RawRadarGraph,
    RawSessionGraph,
)
from snn_rr.synchronization import canonical_content_sha256, canonical_json_bytes


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _biopac_parser_evidence() -> BiopacParserEvidence:
    return BiopacParserEvidence(
        source_data_shape=(16, 2),
        effective_data_shape=(16, 2),
        orientation="samples_by_channels",
        channel_labels=("RSP", "ECG"),
        channel_units=("Volts", "mV"),
        rsp_candidate_indices=(0,),
        ecg_candidate_indices=(1,),
        rsp_index=0,
        ecg_index=1,
        source_isi_value_count=1,
        source_isi_value=4.0,
        source_isi_units=("ms",),
        source_isi_unit="ms",
        normalized_source_isi_unit="ms",
        conversion_factor_to_ms=1.0,
        effective_interval_source="converted_mat_metadata",
        effective_isi_ms=4.0,
        effective_sample_rate_hz=250.0,
        issues=(),
    )


def _radar_metadata_evidence(radar_id: int) -> RadarMetadataEvidence:
    filename = f"radar{radar_id}.dat"
    evidence = XeThruMetaEvidence(
        payload_bytes=219,
        record_table_offset=100,
        footer_offset=199,
        footer_end_offset=219,
        declared_chunk_count=1,
        footer_filenames=(filename,),
        frame_record_count=2,
        close_marker_count=1,
        entry_order_mismatch_count=0,
        chunk_index_set=(0,),
        expected_chunk_filenames=(filename,),
        expected_chunk_byte_counts=(2 * 740,),
        chunks=(
            XeThruMetaChunkEvidence(
                chunk_index=0,
                footer_filename=filename,
                frame_count=2,
                record_size_mismatch_count=0,
                file_offset_mismatch_count=0,
                logical_end_mismatch_count=0,
                metadata_chunk_bytes=2 * 740,
                logical_start=0,
                logical_end=2 * 740,
                close_marker_count=1,
                close_encoded_data_size=0,
                close_file_offset=2 * 740,
                close_logical_end=2 * 740,
                last_frame_timestamp_ms=25,
                close_timestamp_ms=26,
                expected_filename=filename,
                expected_bytes=2 * 740,
            ),
        ),
        issues=(),
    )
    return RadarMetadataEvidence(radar_id=radar_id, evidence=evidence)


def _receipt(*, root: str = "/private/raw", identity_offset: int = 0) -> RawConsumptionReceipt:
    graph = RawSessionGraph(
        session_id="S01_TST",
        selected_logical_session_id="20260827T120000.000Z",
        biopac_path="S01_TST/BIOPAC/reference.mat",
        radars=tuple(
            RawRadarGraph(
                radar_id=radar_id,
                metadata_path=f"S01_TST/{radar_id}/recording/meta.dat",
                data_paths=(f"S01_TST/{radar_id}/recording/radar{radar_id}.dat",),
            )
            for radar_id in (1, 2, 3)
        ),
    )
    specs = [("biopac", graph.biopac_path, 128)]
    for radar in graph.radars:
        specs.append((f"radar{radar.radar_id}_meta", radar.metadata_path, 219))
        specs.append(
            (
                f"radar{radar.radar_id}_data_00",
                radar.data_paths[0],
                2 * 740,
            )
        )
    bindings = tuple(
        ConsumedFileBinding(
            key=key,
            role="fixture",
            relative_path=path,
            byte_count=byte_count,
            sha256=_digest(key),
            identity=FileIdentity(
                device=1 + identity_offset,
                inode=index + 10 + identity_offset,
                mode=0o100600,
                link_count=1,
                byte_count=byte_count,
                mtime_ns=100 + identity_offset,
                ctime_ns=200 + identity_offset,
            ),
        )
        for index, (key, path, byte_count) in enumerate(specs)
    )
    record_evidence = tuple(
        RadarRecordEvidence(
            radar_id,
            (
                RadarChunkEvidence(
                    radar_id=radar_id,
                    chunk_index=0,
                    binding_key=f"radar{radar_id}_data_00",
                    filename=f"radar{radar_id}.dat",
                    byte_count=2 * 740,
                    frame_count=2,
                    zero_header_nonzero=0,
                    bin_count_invalid=0,
                ),
            ),
        )
        for radar_id in (1, 2, 3)
    )
    return RawConsumptionReceipt.build(
        session_id="S01_TST",
        dataset_root=root,
        root_identity=FileIdentity(
            device=7 + identity_offset,
            inode=8 + identity_offset,
            mode=0o40700,
            link_count=2,
            byte_count=0,
            mtime_ns=300 + identity_offset,
            ctime_ns=400 + identity_offset,
        ),
        graph=graph,
        timezone_name="Asia/Seoul",
        fallback_rate_hz=40.0,
        biopac_strict=False,
        require_valid_records=True,
        file_bindings=bindings,
        radar_record_evidence=record_evidence,
        radar_metadata_evidence=tuple(
            _radar_metadata_evidence(radar_id) for radar_id in (1, 2, 3)
        ),
        biopac_parser_evidence=_biopac_parser_evidence(),
    )


def _session(receipt: RawConsumptionReceipt) -> dict[str, object]:
    bindings = receipt.input_bindings
    graph = receipt.raw_input_graph
    record = receipt.xethru_record_contract
    metadata = receipt.xethru_metadata_contract
    return {
        "session_id": receipt.session_id,
        "raw_input_bindings": bindings,
        "raw_input_bindings_sha256": hashlib.sha256(
            canonical_json_bytes(bindings)
        ).hexdigest(),
        "raw_input_graph": graph,
        "raw_input_graph_sha256": hashlib.sha256(
            canonical_json_bytes(graph)
        ).hexdigest(),
        "raw_consumption": {
            "schema": RAW_CONSUMPTION_BINDING_SCHEMA,
            "diagnostic_only": True,
            "scientific_authority": False,
            "portable_projection": receipt.portable_projection,
            "portable_content_sha256": receipt.portable_content_sha256,
        },
        "sensor_summary": {
            "radar": {
                "xethru_record_contract": record,
                "xethru_record_contract_evidence_sha256": hashlib.sha256(
                    canonical_json_bytes(record)
                ).hexdigest(),
                "xethru_metadata_contract": metadata,
                "xethru_metadata_contract_evidence_sha256": hashlib.sha256(
                    canonical_json_bytes(metadata)
                ).hexdigest(),
            },
            "biopac": {
                "parser_evidence": receipt.biopac_parser_evidence.to_dict(),
            },
        },
    }


def test_portable_projection_excludes_local_root_and_descriptor_identity() -> None:
    first = _receipt()
    restored = _receipt(root="/restored/project/raw", identity_offset=100)

    assert first.content_sha256 != restored.content_sha256
    assert first.portable_projection == restored.portable_projection
    assert first.portable_content_sha256 == restored.portable_content_sha256
    encoded = canonical_json_bytes(first.portable_projection)
    assert b"/private/raw" not in encoded
    assert b"descriptor_identity" not in encoded
    assert first.portable_projection["reader_contract"]["live_raw_memmap_returned"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("binding", "consumed bindings|portable/session raw bindings"),
        ("graph", "graph"),
        ("record", "XeThru"),
        ("metadata", "XeThru metadata"),
        ("metadata_portable", "metadata/raw graph cross-link"),
        ("portable_hash", "portable raw-consumption hash"),
        ("parser", "parser/reader policy"),
        ("biopac", "BIOPAC parser evidence mismatch"),
        ("extra", "raw_consumption fields"),
    ],
)
def test_consumption_contract_rejects_resealed_cross_field_tamper(
    mutation: str, match: str
) -> None:
    receipt = _receipt()
    session = deepcopy(_session(receipt))
    if mutation == "binding":
        session["raw_input_bindings"]["biopac"]["sha256"] = "0" * 64
        session["raw_input_bindings_sha256"] = hashlib.sha256(
            canonical_json_bytes(session["raw_input_bindings"])
        ).hexdigest()
    elif mutation == "graph":
        session["raw_input_graph"]["selected_logical_session_id"] = "changed"
        session["raw_input_graph_sha256"] = hashlib.sha256(
            canonical_json_bytes(session["raw_input_graph"])
        ).hexdigest()
    elif mutation == "record":
        session["sensor_summary"]["radar"]["xethru_record_contract"]["views"][0][
            "chunks"
        ][0]["filename"] = "other.dat"
        record = session["sensor_summary"]["radar"]["xethru_record_contract"]
        session["sensor_summary"]["radar"][
            "xethru_record_contract_evidence_sha256"
        ] = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
    elif mutation == "metadata":
        metadata = session["sensor_summary"]["radar"][
            "xethru_metadata_contract"
        ]
        metadata["views"][0]["metadata_evidence"][
            "expected_chunk_filenames"
        ][0] = "other.dat"
        session["sensor_summary"]["radar"][
            "xethru_metadata_contract_evidence_sha256"
        ] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    elif mutation == "metadata_portable":
        projection = session["raw_consumption"]["portable_projection"]
        metadata = projection["xethru_metadata_contract"]
        evidence = metadata["views"][0]["metadata_evidence"]
        evidence["footer_filenames"][0] = "other1.dat"
        evidence["expected_chunk_filenames"][0] = "other1.dat"
        evidence["chunks"][0]["footer_filename"] = "other1.dat"
        evidence["chunks"][0]["expected_filename"] = "other1.dat"
        session["sensor_summary"]["radar"]["xethru_metadata_contract"] = (
            deepcopy(metadata)
        )
        session["sensor_summary"]["radar"][
            "xethru_metadata_contract_evidence_sha256"
        ] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
        session["raw_consumption"]["portable_content_sha256"] = hashlib.sha256(
            canonical_json_bytes(projection)
        ).hexdigest()
    elif mutation == "portable_hash":
        session["raw_consumption"]["portable_content_sha256"] = "0" * 64
    elif mutation == "parser":
        session["raw_consumption"]["portable_projection"]["parser_policy"][
            "timezone_name"
        ] = "UTC"
        session["raw_consumption"]["portable_content_sha256"] = hashlib.sha256(
            canonical_json_bytes(
                session["raw_consumption"]["portable_projection"]
            )
        ).hexdigest()
    elif mutation == "biopac":
        session["sensor_summary"]["biopac"]["parser_evidence"] = {
            **session["sensor_summary"]["biopac"]["parser_evidence"],
            "effective_sample_rate_hz": 125.0,
        }
    else:
        session["raw_consumption"]["unexpected"] = True

    with pytest.raises(AcquisitionContractError, match=match):
        validate_consumption_against_contract(receipt, session)


def test_consumption_contract_positive_and_manual_mapping_receipt_rejected() -> None:
    receipt = _receipt()
    session = _session(receipt)
    assert validate_consumption_against_contract(receipt, session) == session[
        "raw_consumption"
    ]
    with pytest.raises(AcquisitionContractError, match="exact RawConsumptionReceipt"):
        validate_consumption_against_contract(receipt.to_dict(), session)  # type: ignore[arg-type]


def test_consumption_contract_rejects_ineligible_biopac_parser_evidence() -> None:
    receipt = _receipt()
    session = deepcopy(_session(receipt))
    projection = session["raw_consumption"]["portable_projection"]
    evidence = projection["biopac_parser_evidence"]
    evidence["parser_eligible"] = False
    evidence["issues"] = [
        {
            "code": "biopac_test_issue",
            "severity": "error",
            "message": "synthetic ineligible parser evidence",
        }
    ]
    session["raw_consumption"]["portable_content_sha256"] = hashlib.sha256(
        canonical_json_bytes(projection)
    ).hexdigest()
    session["sensor_summary"]["biopac"]["parser_evidence"] = deepcopy(evidence)

    with pytest.raises(AcquisitionContractError, match="BIOPAC parser evidence is ineligible"):
        acquisition_contract_module._validate_serialized_raw_consumption(session)


def test_serialized_v3_validation_rejects_graph_binding_order_transplant() -> None:
    receipt = _receipt()
    session = deepcopy(_session(receipt))
    projection = session["raw_consumption"]["portable_projection"]
    projection["raw_input_graph"]["binding_keys"][1:3] = reversed(
        projection["raw_input_graph"]["binding_keys"][1:3]
    )
    session["raw_consumption"]["portable_content_sha256"] = hashlib.sha256(
        canonical_json_bytes(projection)
    ).hexdigest()
    with pytest.raises(AcquisitionContractError, match="ordering/identity|cover"):
        acquisition_contract_module._validate_serialized_raw_consumption(session)


def test_persisted_diagnostic_receipt_load_validation_rejects_split_brain(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    session = _session(receipt)
    path = tmp_path / "sessions" / receipt.session_id / "raw_receipt.json"
    path.parent.mkdir(parents=True)
    document = receipt.to_dict()
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    session["raw_consumption_receipt"] = {
        "artifact": str(path.relative_to(tmp_path)),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content_sha256": document["content_sha256"],
        "diagnostic_only": True,
        "scientific_authority": False,
    }
    acquisition_contract_module._validate_bound_raw_consumption_receipt(
        session, reconstruction_root=tmp_path
    )

    document["session_id"] = "S99_OTHER"
    document["content_sha256"] = canonical_content_sha256(document)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    session["raw_consumption_receipt"]["artifact_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    session["raw_consumption_receipt"]["content_sha256"] = document[
        "content_sha256"
    ]
    with pytest.raises(AcquisitionContractError, match="duplicate fields mismatch"):
        acquisition_contract_module._validate_bound_raw_consumption_receipt(
            session, reconstruction_root=tmp_path
        )


def test_persisted_receipt_rejects_biopac_evidence_split_brain(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    session = _session(receipt)
    path = tmp_path / "sessions" / receipt.session_id / "raw_receipt.json"
    path.parent.mkdir(parents=True)
    document = receipt.to_dict()
    document["biopac_parser_evidence"] = {
        **document["biopac_parser_evidence"],
        "effective_sample_rate_hz": 125.0,
    }
    document["content_sha256"] = canonical_content_sha256(document)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    session["raw_consumption_receipt"] = {
        "artifact": str(path.relative_to(tmp_path)),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content_sha256": document["content_sha256"],
        "diagnostic_only": True,
        "scientific_authority": False,
    }

    with pytest.raises(AcquisitionContractError, match="duplicate fields mismatch"):
        acquisition_contract_module._validate_bound_raw_consumption_receipt(
            session, reconstruction_root=tmp_path
        )


def test_persisted_receipt_rejects_xethru_metadata_split_brain(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    session = _session(receipt)
    path = tmp_path / "sessions" / receipt.session_id / "raw_receipt.json"
    path.parent.mkdir(parents=True)
    document = receipt.to_dict()
    document["xethru_metadata_contract"]["views"][0]["eligible"] = False
    document["content_sha256"] = canonical_content_sha256(document)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    session["raw_consumption_receipt"] = {
        "artifact": str(path.relative_to(tmp_path)),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content_sha256": document["content_sha256"],
        "diagnostic_only": True,
        "scientific_authority": False,
    }

    with pytest.raises(AcquisitionContractError, match="duplicate fields mismatch"):
        acquisition_contract_module._validate_bound_raw_consumption_receipt(
            session, reconstruction_root=tmp_path
        )


def test_timing_reason_authority_accepts_explained_invalid_zero_cells() -> None:
    times = np.asarray(
        [0.0, 0.025, 0.025, 0.075, 0.100, 0.125, 0.150, 0.175],
        dtype=np.float64,
    )
    sequence = np.asarray([0, 1, 3, 4, 5, 6, 7, 8], dtype=np.uint32)
    base = np.arange(times.size, dtype=np.float32)[:, None]
    base[1, 0] = np.nan
    result = causal_uniform_resample_radar_views_v1(
        [base.copy() for _ in range(3)],
        [times.copy() for _ in range(3)],
        [1_800_000_000.0] * 3,
        [sequence.copy() for _ in range(3)],
        output_hz=10.0,
        max_gap_s=0.030,
        gap_policy="mask",
        timestamp_sources=["meta_v13"] * 3,
        require_measured_timestamps=True,
    )
    authority = validate_timing_reason_authority(result)
    assert authority["invalid_interval_count"] > 0
    assert authority["exact_mask_equivalence"] is True
    assert authority["invalid_outputs_exact_positive_zero"] is True
    assert authority["valid_outputs_finite"] is True
    assert np.array_equal(result.valid_mask, result.invalid_reason_mask == 0)

    tampered = result.valid_mask.copy()
    tampered[0, 0] = True
    with pytest.raises(AcquisitionContractError, match="timing reason authority failed"):
        validate_timing_reason_authority(replace(result, valid_mask=tampered))
