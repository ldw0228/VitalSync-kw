from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import hashlib
import os
from pathlib import Path
import struct

import numpy as np
import pytest
from scipy.io import savemat

import snn_rr.raw_snapshot as raw_snapshot_module
from snn_rr.data import (
    BIOPAC_ISI_CANONICAL_UNIT,
    BIOPAC_PARSER_EVIDENCE_SCHEMA,
    BIOPAC_SAMPLE_RATE_HZ,
    DataFormatError,
    RADAR_BINS,
    RADAR_SAMPLE_RATE_HZ,
    RadarStreamInfo,
    XETHRU_MAGIC,
    XETHRU_META_EVIDENCE_SCHEMA,
    XETHRU_RECORD_BYTES,
    XETHRU_RECORD_DTYPE,
    build_dataset_manifest,
    biopac_qc,
    load_biopac_mat,
    load_xethru_recording,
    open_xethru_files,
    parse_xethru_meta,
    parse_xethru_meta_bytes,
    radar_qc,
    validate_biopac_parser_evidence,
    validate_xethru_meta_evidence,
)
from snn_rr.raw_snapshot import (
    RawRadarGraph,
    RawSessionGraph,
    RawSessionReader,
    RawSnapshotError,
)


_META_RECORD = struct.Struct("<BIQQQI")


def _write_radar_chunk(
    path: Path,
    sequences: list[int],
    *,
    spike: tuple[int, int, float] | None = None,
) -> None:
    records = np.zeros(len(sequences), dtype=XETHRU_RECORD_DTYPE)
    records["frame_sequence"] = sequences
    records["bin_count"] = RADAR_BINS
    records["bins"] = np.arange(RADAR_BINS, dtype=np.float32)[None, :] * 1e-6
    if spike is not None:
        records["bins"][spike[0], spike[1]] = spike[2]
    records.tofile(path)


def _write_meta(
    path: Path,
    *,
    chunk_names: list[str],
    frames_per_chunk: list[int],
    timestamps_ms: list[int],
    start_epoch_ms: int = 1_780_000_000_123,
    device_name: str = "S99_TST_radar1",
    truncate_footer: bool = False,
) -> None:
    assert sum(frames_per_chunk) == len(timestamps_ms)
    payload = bytearray(89)
    struct.pack_into("<IIQ", payload, 0, XETHRU_MAGIC, 13, start_epoch_ms)
    encoded_name = device_name.encode() + b"\x00"
    struct.pack_into("<I", payload, 85, len(encoded_name))
    payload.extend(encoded_name)
    payload.extend(b"\x00" * 4)

    timestamp_index = 0
    logical_end = 0
    for chunk_index, frame_count in enumerate(frames_per_chunk):
        for local_index in range(frame_count):
            logical_end += XETHRU_RECORD_BYTES
            payload.extend(
                _META_RECORD.pack(
                    1,
                    1024,
                    (chunk_index << 32) | XETHRU_RECORD_BYTES,
                    local_index * XETHRU_RECORD_BYTES,
                    logical_end,
                    timestamps_ms[timestamp_index],
                )
            )
            timestamp_index += 1
        last_timestamp = timestamps_ms[timestamp_index - 1]
        payload.extend(
            _META_RECORD.pack(
                2,
                1024,
                chunk_index << 32,
                frame_count * XETHRU_RECORD_BYTES,
                logical_end,
                last_timestamp,
            )
        )
    payload.extend(struct.pack("<BI", 3, len(chunk_names)))
    for name in chunk_names:
        encoded = name.encode() + b"\x00"
        payload.extend(struct.pack("<I", len(encoded)))
        payload.extend(encoded)
    if truncate_footer:
        del payload[-3:]
    path.write_bytes(payload)


def _mutate_meta_contract(path: Path, mutation: str) -> None:
    payload = bytearray(path.read_bytes())
    name_length = struct.unpack_from("<I", payload, 85)[0]
    record_offset = 89 + name_length + 4
    entries = [
        list(_META_RECORD.unpack_from(payload, record_offset + index * _META_RECORD.size))
        for index in range(4)
    ]
    footer_offset = record_offset + 4 * _META_RECORD.size
    if mutation == "frame_size":
        entries[0][2] = (entries[0][2] & ~0xFFFFFFFF) | (XETHRU_RECORD_BYTES - 1)
    elif mutation == "frame_file_offset":
        entries[1][3] += 1
    elif mutation == "frame_logical_end":
        entries[1][4] += 1
    elif mutation == "chunk_index":
        entries[0][2] = (2 << 32) | (entries[0][2] & 0xFFFFFFFF)
    elif mutation == "close_size":
        entries[3][2] |= 1
    elif mutation == "close_file_offset":
        entries[3][3] += 1
    elif mutation == "close_logical_end":
        entries[3][4] += 1
    elif mutation == "missing_close":
        entries[3][0] = 1
        entries[3][2] = XETHRU_RECORD_BYTES
    elif mutation == "entry_order":
        entries[2], entries[3] = entries[3], entries[2]
    elif mutation == "footer_count":
        struct.pack_into("<I", payload, footer_offset + 1, 2)
    elif mutation == "footer_filename":
        filename_start = footer_offset + 5 + 4
        payload[filename_start] = ord("y")
    elif mutation == "footer_extra_nul":
        length_offset = footer_offset + 5
        filename_length = struct.unpack_from("<I", payload, length_offset)[0]
        terminator_offset = length_offset + 4 + filename_length - 1
        payload[terminator_offset:terminator_offset] = b"\x00"
        struct.pack_into("<I", payload, length_offset, filename_length + 1)
    elif mutation == "trailing_bytes":
        payload.extend(b"unbound")
    else:  # pragma: no cover - fixture misuse
        raise AssertionError(mutation)
    if mutation not in {
        "footer_count",
        "footer_filename",
        "footer_extra_nul",
        "trailing_bytes",
    }:
        for index, entry in enumerate(entries):
            _META_RECORD.pack_into(
                payload, record_offset + index * _META_RECORD.size, *entry
            )
    path.write_bytes(payload)


def _write_biopac(path: Path, *, samples: int = 32) -> None:
    time = np.arange(samples, dtype=np.float64) / BIOPAC_SAMPLE_RATE_HZ
    savemat(
        path,
        {
            "data": np.column_stack(
                (np.sin(2 * np.pi * 0.25 * time), np.cos(2 * np.pi * time))
            ),
            "labels": np.asarray(["RSP, X", "ECG, Y"]),
            "units": np.asarray(["Volts", "mV"]),
            "isi": np.asarray([[4]], dtype=np.uint8),
            "isi_units": np.asarray(["ms"]),
        },
    )


def _build_raw_session_graph(
    root: Path,
    *,
    frames_per_chunk: tuple[int, ...] = (3, 2),
) -> RawSessionGraph:
    subject = root / "S01_TST"
    biopac = subject / "BIOPAC" / "S1_TST_01_UWB_2026-08-27T12_34_56.mat"
    biopac.parent.mkdir(parents=True)
    _write_biopac(biopac)
    radar_graphs: list[RawRadarGraph] = []
    for radar_id in (1, 2, 3):
        directory = (
            subject
            / str(radar_id)
            / f"xethru_recording_20260827_120000_S01_TST_radar{radar_id}"
        )
        directory.mkdir(parents=True)
        names: list[str] = []
        paths: list[Path] = []
        timestamps: list[int] = []
        next_sequence = 100 * radar_id
        for chunk_index, frame_count in enumerate(frames_per_chunk):
            name = f"xethru_datafloat_20260827_12{chunk_index:02d}00.dat"
            path = directory / name
            sequences = list(range(next_sequence, next_sequence + frame_count))
            next_sequence += frame_count
            _write_radar_chunk(path, sequences)
            names.append(name)
            paths.append(path)
            timestamps.extend(
                25 * index
                for index in range(
                    len(timestamps), len(timestamps) + frame_count
                )
            )
        meta = directory / "xethru_recording_meta.dat"
        _write_meta(
            meta,
            chunk_names=names,
            frames_per_chunk=list(frames_per_chunk),
            timestamps_ms=timestamps,
            start_epoch_ms=1_780_000_000_000 + radar_id,
            device_name=f"S01_TST_radar{radar_id}",
        )
        radar_graphs.append(
            RawRadarGraph(
                radar_id=radar_id,
                metadata_path=meta.relative_to(root).as_posix(),
                data_paths=tuple(path.relative_to(root).as_posix() for path in paths),
            )
        )
    return RawSessionGraph(
        session_id="S01_TST",
        selected_logical_session_id="20260827T120000.000Z",
        biopac_path=biopac.relative_to(root).as_posix(),
        radars=tuple(radar_graphs),
    )


def test_record_dtype_and_split_memmap_are_continuous(tmp_path: Path) -> None:
    assert XETHRU_RECORD_BYTES == 740
    first = tmp_path / "xethru_datafloat_20260101_000000.dat"
    second = tmp_path / "xethru_datafloat_20260101_000100.dat"
    _write_radar_chunk(first, [1, 2, 3])
    _write_radar_chunk(second, [4, 5])

    recording = open_xethru_files([first, second])
    assert recording.chunk_lengths == (3, 2)
    assert len(recording) == 5
    assert int(recording[-1]["frame_sequence"]) == 5
    np.testing.assert_array_equal(recording[1:5]["frame_sequence"], [2, 3, 4, 5])
    np.testing.assert_array_equal(recording["frame_sequence"], [1, 2, 3, 4, 5])


def test_v13_meta_parses_multi_chunk_footer_and_repairs_clock_reset(
    tmp_path: Path,
) -> None:
    names = [
        "xethru_datafloat_20260101_000000.dat",
        "xethru_datafloat_20260101_000100.dat",
    ]
    meta_path = tmp_path / "xethru_recording_meta.dat"
    _write_meta(
        meta_path,
        chunk_names=names,
        frames_per_chunk=[3, 2],
        timestamps_ms=[0, 25, 50, 5, 30],
    )

    meta = parse_xethru_meta(meta_path, expected_frames=5)
    assert meta.version == 13
    assert meta.start_epoch_ms == 1_780_000_000_123
    assert meta.chunk_filenames == tuple(names)
    assert meta.declared_chunk_count == 2
    assert meta.end_marker_count == 2
    assert meta.timestamp_source == "meta_v13"
    assert meta.timestamp_repairs == 0  # 45 ms is jitter, below reset threshold.
    assert meta.metadata_evidence is not None
    evidence = meta.metadata_evidence.to_dict()
    assert evidence["schema"] == XETHRU_META_EVIDENCE_SCHEMA
    assert evidence["internal_geometry_eligible"] is True
    assert evidence["expected_graph_bound"] is False
    assert evidence["consumption_eligible"] is False
    assert [item["metadata_chunk_bytes"] for item in evidence["chunks"]] == [
        3 * XETHRU_RECORD_BYTES,
        2 * XETHRU_RECORD_BYTES,
    ]
    np.testing.assert_array_equal(meta.relative_timestamps_ms, [0, 25, 50, 5, 30])

    # A true recorder counter reset (>100 ms) is made monotonic.
    reset_path = tmp_path / "reset_meta.dat"
    _write_meta(
        reset_path,
        chunk_names=[names[0]],
        frames_per_chunk=[5],
        timestamps_ms=[0, 25, 250, 5, 30],
    )
    repaired = parse_xethru_meta(reset_path, expected_frames=5)
    assert repaired.timestamp_repairs == 1
    assert np.all(np.diff(repaired.relative_timestamps_ms) >= 0)


def test_meta_uses_40hz_fallback_on_frame_mismatch_and_tolerates_footer_damage(
    tmp_path: Path,
) -> None:
    meta_path = tmp_path / "xethru_recording_meta.dat"
    _write_meta(
        meta_path,
        chunk_names=["xethru_datafloat_20260101_000000.dat"],
        frames_per_chunk=[3],
        timestamps_ms=[7, 31, 56],
        truncate_footer=True,
    )
    meta = parse_xethru_meta(meta_path, expected_frames=4)
    assert meta.timestamp_source == "fallback_40hz"
    np.testing.assert_allclose(meta.relative_timestamps_ms, [0, 25, 50, 75])
    assert any("frame mismatch" in item for item in meta.warnings)
    assert any("footer" in item for item in meta.warnings)


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        ("frame_size", "xethru_frame_record_size_invalid"),
        ("frame_file_offset", "xethru_file_offsets_invalid"),
        ("frame_logical_end", "xethru_logical_ends_invalid"),
        ("chunk_index", "xethru_chunk_indices_noncontiguous"),
        ("close_size", "xethru_close_marker_geometry_invalid"),
        ("close_file_offset", "xethru_close_marker_geometry_invalid"),
        ("close_logical_end", "xethru_close_marker_geometry_invalid"),
        ("missing_close", "xethru_close_marker_count_invalid"),
        ("entry_order", "xethru_entry_order_invalid"),
        ("footer_count", "xethru_footer_truncated"),
        ("footer_filename", "xethru_expected_filename_mismatch"),
        ("footer_extra_nul", "xethru_footer_name_embedded_nul"),
        ("trailing_bytes", "xethru_footer_trailing_bytes"),
    ],
)
def test_xethru_metadata_geometry_and_consumed_graph_join_fail_closed(
    tmp_path: Path,
    mutation: str,
    expected_issue: str,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(3,))
    radar = graph.radars[0]
    meta_path = root / radar.metadata_path
    _mutate_meta_contract(meta_path, mutation)
    chunk_paths = tuple(root / item for item in radar.data_paths)
    chunk_bytes = tuple(item.stat().st_size for item in chunk_paths)
    parse_kwargs = {
        "source_path": meta_path,
        "expected_frames": sum(item // XETHRU_RECORD_BYTES for item in chunk_bytes),
        "expected_chunk_filenames": tuple(item.name for item in chunk_paths),
        "expected_chunk_byte_counts": chunk_bytes,
    }

    parsed = parse_xethru_meta_bytes(
        meta_path.read_bytes(), strict=False, **parse_kwargs
    )
    assert parsed.metadata_evidence is not None
    evidence = parsed.metadata_evidence.to_dict()
    assert evidence["internal_geometry_eligible"] is False or (
        evidence["exact_graph_join"] is False
    )
    assert evidence["consumption_eligible"] is False
    assert expected_issue in {item["code"] for item in evidence["issues"]}
    assert validate_xethru_meta_evidence(evidence) == evidence

    with pytest.raises(DataFormatError) as captured:
        parse_xethru_meta_bytes(meta_path.read_bytes(), strict=True, **parse_kwargs)
    assert captured.value.code == "xethru_metadata_ineligible"
    assert captured.value.diagnostics["consumption_eligible"] is False

    with pytest.raises(RawSnapshotError, match="chunk order/metadata geometry"):
        RawSessionReader(root, graph).consume()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("close_offset", "geometry eligibility"),
        ("expected_filename", "expected-graph cross-link"),
        ("eligibility", "eligibility predicates"),
    ],
)
def test_xethru_metadata_evidence_rejects_resealed_semantic_tamper(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(3,))
    radar = graph.radars[0]
    meta_path = root / radar.metadata_path
    chunk_paths = tuple(root / item for item in radar.data_paths)
    parsed = parse_xethru_meta_bytes(
        meta_path.read_bytes(),
        source_path=meta_path,
        expected_frames=3,
        expected_chunk_filenames=tuple(item.name for item in chunk_paths),
        expected_chunk_byte_counts=tuple(item.stat().st_size for item in chunk_paths),
    )
    assert parsed.metadata_evidence is not None
    evidence = parsed.metadata_evidence.to_dict()
    if mutation == "close_offset":
        evidence["chunks"][0]["close_file_offset"] += 1
    elif mutation == "expected_filename":
        evidence["chunks"][0]["expected_filename"] = "transplanted.dat"
    else:
        evidence["consumption_eligible"] = False

    with pytest.raises(DataFormatError, match=match):
        validate_xethru_meta_evidence(evidence)


def test_xethru_close_timestamp_is_bound_but_not_an_invented_geometry_gate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(3,))
    radar = graph.radars[0]
    meta_path = root / radar.metadata_path
    chunk_path = root / radar.data_paths[0]
    parsed = parse_xethru_meta_bytes(
        meta_path.read_bytes(),
        source_path=meta_path,
        expected_frames=3,
        expected_chunk_filenames=(chunk_path.name,),
        expected_chunk_byte_counts=(chunk_path.stat().st_size,),
    )
    assert parsed.metadata_evidence is not None
    evidence = deepcopy(parsed.metadata_evidence.to_dict())
    evidence["chunks"][0]["close_timestamp_ms"] += 16

    validated = validate_xethru_meta_evidence(evidence)
    assert validated["consumption_eligible"] is True
    assert "diagnostic_only" in validated["close_timestamp_policy"]


def test_xethru_metadata_evidence_rejects_non_utf8_filename_structurally(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(3,))
    radar = graph.radars[0]
    meta_path = root / radar.metadata_path
    chunk_path = root / radar.data_paths[0]
    parsed = parse_xethru_meta_bytes(
        meta_path.read_bytes(),
        source_path=meta_path,
        expected_frames=3,
        expected_chunk_filenames=(chunk_path.name,),
        expected_chunk_byte_counts=(chunk_path.stat().st_size,),
    )
    assert parsed.metadata_evidence is not None
    evidence = deepcopy(parsed.metadata_evidence.to_dict())
    evidence["footer_filenames"][0] = "invalid_\ud800.dat"

    with pytest.raises(DataFormatError, match="valid UTF-8") as captured:
        validate_xethru_meta_evidence(evidence)
    assert captured.value.code == "xethru_metadata_evidence_invalid"


def test_biopac_mat_loads_rsp_ecg_isi_and_filename_time(tmp_path: Path) -> None:
    mat_path = tmp_path / "S1_TST_01_UWB_2026-08-27T12_34_56.mat"
    time = np.arange(500, dtype=np.float64) / BIOPAC_SAMPLE_RATE_HZ
    rsp = np.sin(2 * np.pi * 0.25 * time)
    ecg = np.cos(2 * np.pi * 1.2 * time)
    savemat(
        mat_path,
        {
            "data": np.column_stack((rsp, ecg)),
            "labels": np.asarray(["RSP, X", "ECG, Y"]),
            "units": np.asarray(["Volts", "mV"]),
            "isi": np.asarray([[4]], dtype=np.uint8),
            "isi_units": np.asarray(["ms"]),
        },
    )

    recording = load_biopac_mat(mat_path)
    assert recording.data.shape == (500, 2)
    assert recording.sample_rate_hz == BIOPAC_SAMPLE_RATE_HZ
    assert recording.isi_ms == 4
    assert recording.start_datetime.isoformat() == "2026-08-27T12:34:56+09:00"
    np.testing.assert_allclose(recording.rsp, rsp)
    np.testing.assert_allclose(recording.ecg, ecg)

    evidence = recording.parser_evidence.to_dict()
    assert evidence["schema"] == BIOPAC_PARSER_EVIDENCE_SCHEMA
    assert evidence["parser_eligible"] is True
    assert evidence["source_isi_value"] == 4.0
    assert evidence["source_isi_unit"] == "ms"
    assert evidence["canonical_isi_unit"] == BIOPAC_ISI_CANONICAL_UNIT
    assert evidence["effective_isi_ms"] == 4.0
    assert evidence["effective_sample_rate_hz"] == 250.0
    assert evidence["label_count"] == evidence["channel_count"] == 2
    assert evidence["rsp_candidate_indices"] == [0]
    assert evidence["ecg_candidate_indices"] == [1]


@pytest.mark.parametrize(
    ("source_isi", "source_unit", "expected_factor"),
    [
        (4.0, "ms", 1.0),
        (0.004, "seconds", 1000.0),
        (4000.0, "μs", 0.001),
    ],
)
def test_biopac_isi_units_are_explicitly_converted_to_canonical_ms(
    tmp_path: Path,
    source_isi: float,
    source_unit: str,
    expected_factor: float,
) -> None:
    mat_path = tmp_path / "S1_TST_01_UWB_2026-08-27T12_34_56.mat"
    savemat(
        mat_path,
        {
            "data": np.zeros((16, 2), dtype=np.float64),
            "labels": np.asarray(["RSP", "ECG"]),
            "units": np.asarray(["Volts", "mV"]),
            "isi": np.asarray(source_isi),
            "isi_units": np.asarray(source_unit),
        },
    )

    recording = load_biopac_mat(mat_path)
    evidence = recording.parser_evidence.to_dict()
    assert recording.isi_ms == 4.0
    assert recording.sample_rate_hz == 250.0
    assert recording.parser_eligible is True
    assert evidence["source_isi_unit"] == source_unit
    assert evidence["conversion_factor_to_ms"] == expected_factor
    assert evidence["canonical_isi_unit"] == "ms"
    assert evidence["effective_interval_source"] == "converted_mat_metadata"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_isi_units", "biopac_isi_units_cardinality_invalid"),
        ("multiple_isi", "biopac_isi_cardinality_invalid"),
        ("unsupported_isi_units", "biopac_isi_units_unsupported"),
        ("unexpected_effective_isi", "biopac_effective_isi_unexpected"),
        ("channel_unit_count", "biopac_unit_count_mismatch"),
        ("signal_nonfinite", "biopac_signal_nonfinite"),
        ("overflowing_isi", "biopac_effective_interval_invalid"),
    ],
)
def test_biopac_permissive_parse_retains_structured_ineligibility(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    mat_path = tmp_path / "S1_TST_01_UWB_2026-08-27T12_34_56.mat"
    document: dict[str, object] = {
        "data": np.zeros((16, 2), dtype=np.float64),
        "labels": np.asarray(["RSP", "ECG"]),
        "units": np.asarray(["Volts", "mV"]),
        "isi": np.asarray(4.0),
        "isi_units": np.asarray("ms"),
    }
    if mutation == "missing_isi_units":
        document.pop("isi_units")
    elif mutation == "multiple_isi":
        document["isi"] = np.asarray([4.0, 5.0])
    elif mutation == "unsupported_isi_units":
        document["isi_units"] = np.asarray("fortnights")
    elif mutation == "unexpected_effective_isi":
        document["isi"] = np.asarray(8.0)
    elif mutation == "channel_unit_count":
        document["units"] = np.asarray(["Volts"])
    elif mutation == "signal_nonfinite":
        signal = np.zeros((16, 2), dtype=np.float64)
        signal[3, 0] = np.nan
        document["data"] = signal
    else:
        document["isi"] = np.asarray(1e308)
        document["isi_units"] = np.asarray("seconds")
    savemat(mat_path, document)

    recording = load_biopac_mat(mat_path, strict=False)
    evidence = recording.parser_evidence.to_dict()
    issue_codes = {item["code"] for item in evidence["issues"]}
    assert recording.parser_eligible is False
    assert evidence["parser_eligible"] is False
    assert expected_code in issue_codes
    assert recording.warnings
    if mutation in {
        "missing_isi_units",
        "multiple_isi",
        "unsupported_isi_units",
        "overflowing_isi",
    }:
        assert evidence["effective_interval_source"] == "diagnostic_fallback_4ms"
        assert recording.isi_ms == 4.0
    with pytest.raises(DataFormatError) as captured:
        load_biopac_mat(mat_path, strict=True)
    assert captured.value.code == "biopac_parser_ineligible"
    assert captured.value.diagnostics["parser_eligible"] is False
    assert expected_code in {
        item["code"] for item in captured.value.diagnostics["issues"]
    }
    qc = biopac_qc(mat_path)
    assert qc["status"] == "error"
    assert qc["parser_eligible"] is False
    assert expected_code in {
        item["code"] for item in qc["parser_evidence"]["issues"]
    }


@pytest.mark.parametrize(
    ("data", "labels", "expected_code"),
    [
        (
            np.zeros((8, 2), dtype=np.float64),
            ["RSP"],
            "biopac_label_count_mismatch",
        ),
        (
            np.zeros((8, 3), dtype=np.float64),
            ["RSP primary", "RSP backup", "ECG"],
            "biopac_rsp_label_cardinality_invalid",
        ),
        (
            np.zeros((8, 2), dtype=np.float64),
            ["AUX", "ECG"],
            "biopac_rsp_label_cardinality_invalid",
        ),
        (
            np.zeros((8, 2), dtype=np.float64),
            ["RSP", "AUX"],
            "biopac_ecg_label_cardinality_invalid",
        ),
        (
            np.zeros((8, 3), dtype=np.float64),
            ["RSP", "ECG primary", "ECG backup"],
            "biopac_ecg_label_cardinality_invalid",
        ),
        (
            np.zeros((8, 2), dtype=np.float64),
            ["RSP ECG", "AUX"],
            "biopac_rsp_ecg_channel_collision",
        ),
        (
            np.zeros((2, 2), dtype=np.float64),
            ["RSP", "ECG"],
            "biopac_channel_orientation_ambiguous",
        ),
    ],
)
def test_biopac_channel_identity_contract_fails_closed_even_when_not_strict(
    tmp_path: Path,
    data: np.ndarray,
    labels: list[str],
    expected_code: str,
) -> None:
    mat_path = tmp_path / "S1_TST_01_UWB_2026-08-27T12_34_56.mat"
    savemat(
        mat_path,
        {
            "data": data,
            "labels": np.asarray(labels),
            "units": np.asarray(["unit"] * len(labels)),
            "isi": np.asarray(4.0),
            "isi_units": np.asarray("ms"),
        },
    )

    with pytest.raises(DataFormatError) as captured:
        load_biopac_mat(mat_path, strict=False)
    assert captured.value.code == expected_code
    assert captured.value.diagnostics["parser_eligible"] is False
    assert captured.value.diagnostics["issues"][0]["code"] == expected_code
    qc = biopac_qc(mat_path)
    assert qc["status"] == "error"
    assert qc["parser_eligible"] is False
    assert qc["error_code"] == expected_code
    assert qc["parser_evidence"]["issues"][0]["code"] == expected_code


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_data", "biopac_data_missing"),
        ("nonnumeric_data", "biopac_data_type_invalid"),
    ],
)
def test_biopac_structural_mat_errors_have_machine_readable_diagnostics(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    mat_path = tmp_path / "S1_TST_01_UWB_2026-08-27T12_34_56.mat"
    document: dict[str, object] = {
        "data": np.zeros((8, 2), dtype=np.float64),
        "labels": np.asarray(["RSP", "ECG"]),
        "units": np.asarray(["Volts", "mV"]),
        "isi": np.asarray(4.0),
        "isi_units": np.asarray("ms"),
    }
    if mutation == "missing_data":
        document.pop("data")
    else:
        document["data"] = np.asarray(
            [["bad", "data"] for _ in range(8)], dtype="U4"
        )
    savemat(mat_path, document)

    with pytest.raises(DataFormatError) as captured:
        load_biopac_mat(mat_path, strict=False)
    assert captured.value.code == expected_code
    assert captured.value.diagnostics["parser_eligible"] is False
    assert captured.value.diagnostics["issues"][0]["code"] == expected_code


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("effective_interval", "effective interval|converted interval"),
        ("source_unit", "source unit"),
        ("channel_label", "channel identity"),
        ("eligibility", "eligibility"),
    ],
)
def test_biopac_parser_evidence_validator_rejects_resealed_semantic_tamper(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    mat_path = tmp_path / "S1_TST_01_UWB_2026-08-27T12_34_56.mat"
    _write_biopac(mat_path)
    evidence = load_biopac_mat(mat_path).parser_evidence.to_dict()
    if mutation == "effective_interval":
        evidence["effective_isi_ms"] = 8.0
    elif mutation == "source_unit":
        evidence["source_isi_unit"] = "seconds"
    elif mutation == "channel_label":
        evidence["channel_labels"][0] = "AUX"
    else:
        evidence["parser_eligible"] = False

    with pytest.raises(DataFormatError, match=match):
        validate_biopac_parser_evidence(evidence)


def test_manifest_selects_longest_complete_session_and_keeps_missing_subject(
    tmp_path: Path,
) -> None:
    subject = tmp_path / "S01_TST"
    for radar_id in (1, 2, 3):
        for suffix, frames, start in (
            ("20260827_120000", 6, 1_780_000_000_000),
            ("20260827_130000", 2, 1_780_003_600_000),
        ):
            directory = subject / str(radar_id) / f"xethru_recording_{suffix}_S01_TST_radar{radar_id}"
            directory.mkdir(parents=True)
            data_name = f"xethru_datafloat_{suffix}.dat"
            _write_radar_chunk(directory / data_name, list(range(1, frames + 1)))
            _write_meta(
                directory / "xethru_recording_meta.dat",
                chunk_names=[data_name],
                frames_per_chunk=[frames],
                timestamps_ms=[25 * index for index in range(frames)],
                start_epoch_ms=start + radar_id,
                device_name=f"S01_TST_radar{radar_id}",
            )
    (tmp_path / "S24_KHJ").mkdir()

    manifest = build_dataset_manifest(
        tmp_path, expected_subject_numbers=[1, 24], session_tolerance_seconds=1.0
    )
    selected = manifest.by_subject(1).selected_session
    assert selected is not None
    assert selected.complete
    assert selected.common_frame_count == 6
    missing = manifest.by_subject(24)
    assert missing.selected_session is None
    assert missing.missing_radars == (1, 2, 3)


def test_recording_orders_footer_chunks_and_qc_detects_isolated_spike(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "xethru_recording_20260827_120000_S22_TST_radar2"
    directory.mkdir()
    early = directory / "xethru_datafloat_20260827_120000.dat"
    late = directory / "xethru_datafloat_20260827_120100.dat"
    _write_radar_chunk(early, [1, 2])
    _write_radar_chunk(late, [3, 4], spike=(1, 5, -1.25))
    _write_meta(
        directory / "xethru_recording_meta.dat",
        # Deliberately reverse footer order to verify it is authoritative.
        chunk_names=[late.name, early.name],
        frames_per_chunk=[2, 2],
        timestamps_ms=[0, 25, 50, 75],
    )
    loaded = load_xethru_recording(directory)
    assert loaded.records.paths == (late, early)

    stream = RadarStreamInfo(
        radar_id=2,
        recording_dir=directory,
        meta_path=directory / "xethru_recording_meta.dat",
        data_paths=(early, late),
        start_epoch_ms=1_780_000_000_123,
        frame_count=4,
        duration_seconds=4 / RADAR_SAMPLE_RATE_HZ,
        timestamp_source="meta_v13",
    )
    report = radar_qc(stream)
    assert report["status"] == "error"
    assert report["amplitude_outlier_count"] == 1
    assert report["first_amplitude_outlier"]["frame_sequence"] == 4
    assert report["first_amplitude_outlier"]["bin_index"] == 5


def test_raw_session_reader_returns_owned_arrays_and_exact_diagnostic_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root)

    reader = RawSessionReader(root, graph, records_per_block=1)
    loaded = reader.consume()

    assert loaded.session_id == graph.session_id
    assert set(loaded.radars) == {1, 2, 3}
    radar = loaded.radars[1]
    assert radar.chunk_lengths == (3, 2)
    np.testing.assert_array_equal(radar.frame_sequence, [100, 101, 102, 103, 104])
    assert radar.bins.shape == (5, RADAR_BINS)
    for array in (
        radar.zero,
        radar.frame_sequence,
        radar.bin_count,
        radar.bins,
        radar.timestamps_ms,
        loaded.biopac.data,
    ):
        assert not array.flags.writeable
    assert not isinstance(radar.bins, np.memmap)

    receipt = loaded.receipt
    document = receipt.to_dict()
    assert document["diagnostic_only"] is True
    assert document["scientific_authority"] is False
    assert document["reader_contract"]["live_raw_memmap_returned"] is False
    assert document["reader_contract"]["timezone_name"] == "Asia/Seoul"
    assert document["raw_input_graph"] == graph.to_dict()
    assert document["xethru_record_contract"]["eligible"] is True
    metadata_contract = document["xethru_metadata_contract"]
    assert metadata_contract["eligible"] is True
    assert [item["radar_id"] for item in metadata_contract["views"]] == [1, 2, 3]
    assert all(
        item["metadata_evidence"]["consumption_eligible"] is True
        for item in metadata_contract["views"]
    )
    assert (
        document["portable_projection"]["xethru_metadata_contract"]
        == metadata_contract
    )
    assert document["biopac_parser_evidence"]["parser_eligible"] is True
    assert document["biopac_parser_evidence"]["source_isi_unit"] == "ms"
    assert document["biopac_parser_evidence"]["canonical_isi_unit"] == "ms"
    assert document["biopac_parser_evidence"]["effective_isi_ms"] == 4.0
    assert (
        document["portable_projection"]["biopac_parser_evidence"]
        == document["biopac_parser_evidence"]
        == loaded.biopac.parser_evidence.to_dict()
    )
    assert len(receipt.content_sha256) == 64
    for key, binding in receipt.input_bindings.items():
        source = root / binding["path"]
        assert binding["bytes"] == source.stat().st_size
        assert binding["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest(), key

    with pytest.raises(RawSnapshotError, match="one-shot"):
        reader.consume()


def test_raw_receipt_binds_permissive_biopac_parser_ineligibility(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root)
    biopac_path = root / graph.biopac_path
    savemat(
        biopac_path,
        {
            "data": np.zeros((16, 2), dtype=np.float64),
            "labels": np.asarray(["RSP", "ECG"]),
            "units": np.asarray(["Volts", "mV"]),
            "isi": np.asarray(4.0),
            "isi_units": np.asarray("samples"),
        },
    )

    loaded = RawSessionReader(root, graph, biopac_strict=False).consume()
    document = loaded.receipt.to_dict()
    evidence = document["biopac_parser_evidence"]
    assert loaded.biopac.parser_eligible is False
    assert evidence["parser_eligible"] is False
    assert evidence["effective_interval_source"] == "diagnostic_fallback_4ms"
    assert {item["code"] for item in evidence["issues"]} == {
        "biopac_isi_units_unsupported"
    }
    assert document["portable_projection"]["biopac_parser_evidence"] == evidence
    assert document["scientific_authority"] is False


def test_raw_session_reader_hash_decode_and_record_evidence_use_same_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(4,))
    chunk = root / graph.radars[0].data_paths[0]
    records = np.fromfile(chunk, dtype=XETHRU_RECORD_DTYPE).copy()
    records["zero"][1] = 9
    records["bin_count"][2] = RADAR_BINS - 1
    records["bins"][3, 7] = np.float32(0.03125)
    records.tofile(chunk)
    exact_payload = chunk.read_bytes()

    loaded = RawSessionReader(
        root,
        graph,
        require_valid_records=False,
        records_per_block=1,
    ).consume()
    radar = loaded.radars[1]
    binding = loaded.receipt.input_bindings["radar1_data_00"]
    assert binding["sha256"] == hashlib.sha256(exact_payload).hexdigest()
    np.testing.assert_array_equal(radar.zero, records["zero"])
    np.testing.assert_array_equal(radar.frame_sequence, records["frame_sequence"])
    np.testing.assert_array_equal(radar.bin_count, records["bin_count"])
    np.testing.assert_array_equal(radar.bins, records["bins"])
    evidence = radar.evidence.chunks[0]
    assert evidence.zero_header_nonzero == 1
    assert evidence.bin_count_invalid == 1
    assert not loaded.receipt.xethru_record_contract["eligible"]


def test_raw_session_reader_is_deterministic_across_record_block_sizes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(4, 3))

    first = RawSessionReader(root, graph, records_per_block=1).consume()
    second = RawSessionReader(root, graph, records_per_block=64).consume()

    assert first.receipt.to_dict() == second.receipt.to_dict()
    for radar_id in (1, 2, 3):
        np.testing.assert_array_equal(
            first.radars[radar_id].frame_sequence,
            second.radars[radar_id].frame_sequence,
        )
        np.testing.assert_array_equal(
            first.radars[radar_id].bins,
            second.radars[radar_id].bins,
        )
    np.testing.assert_array_equal(first.biopac.data, second.biopac.data)


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_raw_session_reader_rejects_symlink_and_hardlink_aliases(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(3,))
    victim = root / graph.radars[0].data_paths[0]
    if alias_kind == "symlink":
        target = victim.with_name("unbound_target.dat")
        target.write_bytes(victim.read_bytes())
        victim.unlink()
        victim.symlink_to(target.name)
    else:
        os.link(victim, victim.with_name("unbound_hardlink.dat"))

    with pytest.raises(RawSnapshotError, match="pin raw file|unaliased regular file"):
        RawSessionReader(root, graph).consume()


def test_raw_session_reader_rejects_swap_between_path_stat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(3,))
    victim = root / graph.radars[0].data_paths[0]
    original = victim.with_name("original_generation.dat")
    replacement = victim.with_name("replacement_generation.dat")
    _write_radar_chunk(replacement, [900, 901, 902])
    fired = False

    def swap_after_stat(event: str, key: str) -> None:
        nonlocal fired
        if event == "after_file_path_stat" and key == "radar1_data_00" and not fired:
            fired = True
            os.replace(victim, original)
            os.replace(replacement, victim)

    monkeypatch.setattr(raw_snapshot_module, "_TEST_EVENT_HOOK", swap_after_stat)
    with pytest.raises(RawSnapshotError, match="stable, unaliased regular file"):
        RawSessionReader(root, graph).consume()
    assert fired


def test_raw_session_reader_rejects_root_rebind_after_descriptor_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(2,))
    moved = tmp_path / "raw_original"
    fired = False

    def rebind_root(event: str, key: str) -> None:
        nonlocal fired
        if event == "after_root_open" and key == "dataset_root" and not fired:
            fired = True
            os.replace(root, moved)
            root.mkdir()

    monkeypatch.setattr(raw_snapshot_module, "_TEST_EVENT_HOOK", rebind_root)
    with pytest.raises(RawSnapshotError, match="dataset root changed or rebound"):
        RawSessionReader(root, graph).consume()
    assert fired


def test_raw_session_reader_rejects_symlinked_dataset_root_component(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    root = real_parent / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(2,))
    linked_parent = tmp_path / "linked_parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(RawSnapshotError, match="pin dataset root"):
        RawSessionReader(linked_parent / "raw", graph).consume()


def test_raw_session_reader_rejects_midstream_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(4,))
    victim = root / graph.radars[0].data_paths[0]
    fired = False

    def truncate_after_first_record(event: str, key: str) -> None:
        nonlocal fired
        if event == "after_read_block" and key == "radar1_data_00" and not fired:
            fired = True
            with victim.open("r+b") as handle:
                handle.truncate(XETHRU_RECORD_BYTES)

    monkeypatch.setattr(
        raw_snapshot_module, "_TEST_EVENT_HOOK", truncate_after_first_record
    )
    with pytest.raises(RawSnapshotError, match="truncated"):
        RawSessionReader(root, graph, records_per_block=1).consume()
    assert fired


def test_raw_session_reader_rejects_explicit_chunk_order_transplant(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(2, 2))
    first_radar = graph.radars[0]
    transplanted_radar = replace(
        first_radar, data_paths=tuple(reversed(first_radar.data_paths))
    )
    transplanted_graph = replace(
        graph, radars=(transplanted_radar, graph.radars[1], graph.radars[2])
    )

    with pytest.raises(RawSnapshotError, match="chunk order"):
        RawSessionReader(root, transplanted_graph).consume()


def test_raw_session_graph_rejects_escape_and_cross_session_transplant() -> None:
    with pytest.raises(RawSnapshotError, match="contained relative path"):
        RawRadarGraph(
            radar_id=1,
            metadata_path="../S02_TST/xethru_recording_meta.dat",
            data_paths=("S01_TST/1/xethru_datafloat_20260827_120000.dat",),
        )

    radars = tuple(
        RawRadarGraph(
            radar_id=radar_id,
            metadata_path=f"S02_TST/{radar_id}/xethru_recording_meta.dat",
            data_paths=(
                f"S02_TST/{radar_id}/xethru_datafloat_20260827_120000.dat",
            ),
        )
        for radar_id in (1, 2, 3)
    )
    with pytest.raises(RawSnapshotError, match="exact session directory"):
        RawSessionGraph(
            session_id="S01_TST",
            selected_logical_session_id="logical",
            biopac_path="S01_TST/BIOPAC/reference.mat",
            radars=radars,
        )


def test_raw_session_reader_rejects_swap_and_restore_namespace_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    graph = _build_raw_session_graph(root, frames_per_chunk=(3,))
    victim = root / graph.radars[0].data_paths[0]
    original = victim.with_name("held_original.dat")
    replacement = victim.with_name("hostile_generation.dat")
    _write_radar_chunk(replacement, [700, 701, 702])
    swapped = False
    restored = False

    def swap_and_restore(event: str, key: str) -> None:
        nonlocal swapped, restored
        if event == "after_all_files_opened" and not swapped:
            swapped = True
            os.replace(victim, original)
            os.replace(replacement, victim)
        elif event == "before_final_verify" and swapped and not restored:
            restored = True
            os.replace(victim, replacement)
            os.replace(original, victim)

    monkeypatch.setattr(raw_snapshot_module, "_TEST_EVENT_HOOK", swap_and_restore)
    with pytest.raises(RawSnapshotError, match="directory changed|changed or rebound"):
        RawSessionReader(root, graph).consume()
    assert swapped and restored
