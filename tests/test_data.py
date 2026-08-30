from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import struct

import numpy as np
from scipy.io import savemat

from snn_rr.data import (
    BIOPAC_SAMPLE_RATE_HZ,
    RADAR_BINS,
    RADAR_SAMPLE_RATE_HZ,
    RadarStreamInfo,
    XETHRU_MAGIC,
    XETHRU_RECORD_BYTES,
    XETHRU_RECORD_DTYPE,
    build_dataset_manifest,
    load_biopac_mat,
    load_xethru_recording,
    open_xethru_files,
    parse_xethru_meta,
    radar_qc,
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
