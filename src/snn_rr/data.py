"""Reproducible readers and quality-control utilities for HAI_EXPERIMENT.

The raw radar files are deliberately kept memory mapped.  A recording can span
multiple ``xethru_datafloat_*.dat`` chunks; :class:`SplitRadarMemmap` presents
those chunks as one read-only, continuously indexed structured array without
copying the complete recording into RAM.

XeThru recorder metadata version 13 is not a general-purpose interchange
format.  The parser below implements the layout present in HAI_EXPERIMENT and
keeps conservative fallbacks: corrupt or incomplete per-frame timing is
replaced by a deterministic 40 Hz timeline, while a damaged filename footer
does not discard otherwise valid frame timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any, Iterable, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np


RADAR_BINS = 182
RADAR_SAMPLE_RATE_HZ = 40.0
BIOPAC_SAMPLE_RATE_HZ = 250.0
BIOPAC_ISI_MS = 4.0
BIOPAC_ISI_CANONICAL_UNIT = "ms"
BIOPAC_PARSER_EVIDENCE_SCHEMA = "snn_rr.biopac_parser_evidence.v1"
XETHRU_MAGIC = 0xA0B1C2D3
XETHRU_META_VERSION = 13
XETHRU_META_RECORD_FORMAT = 1024
XETHRU_META_EVIDENCE_SCHEMA = "snn_rr.xethru_v13_metadata_evidence.v1"
XETHRU_META_CLOSE_TIMESTAMP_POLICY = (
    "diagnostic_only_not_a_geometry_gate_observed_post_frame_variation"
)

# On disk: uint32 zero, uint32 frame sequence, uint32 bin count, 182 float32.
# Explicit little endian is important when the audit is run on a big-endian
# machine.  ``align=False`` guarantees the observed 740-byte record size.
XETHRU_RECORD_DTYPE = np.dtype(
    [
        ("zero", "<u4"),
        ("frame_sequence", "<u4"),
        ("bin_count", "<u4"),
        ("bins", "<f4", (RADAR_BINS,)),
    ],
    align=False,
)
XETHRU_RECORD_BYTES = XETHRU_RECORD_DTYPE.itemsize

_META_HEADER_NAME_LENGTH_OFFSET = 85
_META_FIXED_RECORD = struct.Struct("<BIQQQI")
_BIOPAC_DATETIME_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{2})[_:-](?P<minute>\d{2})[_:-](?P<second>\d{2})"
)
_BIOPAC_RSP_LABEL_RE = re.compile(r"(?<![A-Z0-9])RSP(?![A-Z0-9])")
_BIOPAC_ECG_LABEL_RE = re.compile(r"(?<![A-Z0-9])ECG(?![A-Z0-9])")
_RADAR_DIR_DATETIME_RE = re.compile(r"xethru_recording_(\d{8})_(\d{6})")
_DATA_CHUNK_DATETIME_RE = re.compile(r"xethru_datafloat_(\d{8})_(\d{6})\.dat$")


class DataFormatError(ValueError):
    """Raised when a raw file cannot be interpreted safely.

    ``diagnostics`` is intentionally JSON-compatible.  Callers using a
    permissive audit path can therefore retain the exact fail-closed reason
    without parsing exception prose.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "data_format_error",
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostics = dict(diagnostics or {})


def _coerce_timezone(value: str | tzinfo | None) -> tzinfo:
    if value is None:
        return ZoneInfo("Asia/Seoul")
    if isinstance(value, str):
        return ZoneInfo(value)
    return value


def _path_sort_key(path: Path) -> tuple[datetime, str]:
    match = _DATA_CHUNK_DATETIME_RE.search(path.name)
    if match:
        stamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    else:
        stamp = datetime.min
    return stamp, path.name


def _fallback_timestamps(frame_count: int, rate_hz: float) -> np.ndarray:
    if rate_hz <= 0:
        raise ValueError("fallback_rate_hz must be positive")
    return np.arange(frame_count, dtype=np.float64) * (1000.0 / rate_hz)


def _unwrap_relative_timestamps(
    raw_ms: Sequence[int], fallback_period_ms: float
) -> tuple[np.ndarray, int]:
    """Repair recorder-clock resets while retaining measured frame jitter.

    Version-13 metadata occasionally resets its relative millisecond counter
    during a recording (the HAI S07 streams contain one such reset).  A reset
    is distinguishable from millisecond quantisation because it moves back by
    much more than a frame.  Small equal timestamps are retained.
    """

    result = np.asarray(raw_ms, dtype=np.float64).copy()
    if result.size < 2:
        return result, 0
    repair_count = 0
    offset = 0.0
    reset_threshold_ms = max(100.0, 4.0 * fallback_period_ms)
    previous = result[0]
    for index in range(1, result.size):
        candidate = result[index] + offset
        if candidate < previous - reset_threshold_ms:
            offset += previous + fallback_period_ms - candidate
            candidate = result[index] + offset
            repair_count += 1
        result[index] = candidate
        previous = candidate
    return result, repair_count


@dataclass(frozen=True, slots=True)
class XeThruMetaIssue:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": "error", "message": self.message}


@dataclass(frozen=True, slots=True)
class XeThruMetaChunkEvidence:
    chunk_index: int
    footer_filename: str | None
    frame_count: int
    record_size_mismatch_count: int
    file_offset_mismatch_count: int
    logical_end_mismatch_count: int
    metadata_chunk_bytes: int
    logical_start: int
    logical_end: int
    close_marker_count: int
    close_encoded_data_size: int | None
    close_file_offset: int | None
    close_logical_end: int | None
    last_frame_timestamp_ms: int | None
    close_timestamp_ms: int | None
    expected_filename: str | None
    expected_bytes: int | None

    @property
    def geometry_eligible(self) -> bool:
        return (
            self.frame_count > 0
            and self.record_size_mismatch_count == 0
            and self.file_offset_mismatch_count == 0
            and self.logical_end_mismatch_count == 0
            and self.close_marker_count == 1
            and self.close_encoded_data_size == 0
            and self.close_file_offset == self.metadata_chunk_bytes
            and self.close_logical_end == self.logical_end
        )

    @property
    def filename_matches(self) -> bool | None:
        if self.expected_filename is None:
            return None
        return self.footer_filename == self.expected_filename

    @property
    def bytes_match(self) -> bool | None:
        if self.expected_bytes is None:
            return None
        return self.metadata_chunk_bytes == self.expected_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "footer_filename": self.footer_filename,
            "frame_count": self.frame_count,
            "record_bytes": XETHRU_RECORD_BYTES,
            "record_size_mismatch_count": self.record_size_mismatch_count,
            "file_offset_mismatch_count": self.file_offset_mismatch_count,
            "logical_end_mismatch_count": self.logical_end_mismatch_count,
            "metadata_chunk_bytes": self.metadata_chunk_bytes,
            "logical_start": self.logical_start,
            "logical_end": self.logical_end,
            "close_marker_count": self.close_marker_count,
            "close_encoded_data_size": self.close_encoded_data_size,
            "close_file_offset": self.close_file_offset,
            "close_logical_end": self.close_logical_end,
            "last_frame_timestamp_ms": self.last_frame_timestamp_ms,
            "close_timestamp_ms": self.close_timestamp_ms,
            "expected_filename": self.expected_filename,
            "expected_bytes": self.expected_bytes,
            "filename_matches": self.filename_matches,
            "bytes_match": self.bytes_match,
            "geometry_eligible": self.geometry_eligible,
        }


@dataclass(frozen=True, slots=True)
class XeThruMetaEvidence:
    """Structured v13 geometry evidence, never an authorization token.

    The gated rules mirror all 90 recorder metadata files present in the
    restored cohort: 87 selected usable radar streams plus three unselected
    streams.  Frame offsets start at zero and advance by 740 bytes within each
    chunk; logical ends advance cumulatively; every chunk has one kind-2 close;
    and the kind-3 footer is an exact ordered inventory.  Close timestamps are
    retained but deliberately not gated: observed close-minus-last-frame
    deltas are 0, 1, 2, or 16 ms, so equality would be an invented rule.
    """

    payload_bytes: int
    record_table_offset: int | None
    footer_offset: int | None
    footer_end_offset: int | None
    declared_chunk_count: int | None
    footer_filenames: tuple[str, ...]
    frame_record_count: int
    close_marker_count: int
    entry_order_mismatch_count: int
    chunk_index_set: tuple[int, ...]
    expected_chunk_filenames: tuple[str, ...] | None
    expected_chunk_byte_counts: tuple[int, ...] | None
    chunks: tuple[XeThruMetaChunkEvidence, ...]
    issues: tuple[XeThruMetaIssue, ...]

    @property
    def internal_geometry_eligible(self) -> bool:
        expected_indices = tuple(range(len(self.chunks)))
        table_span_exact = (
            self.record_table_offset is not None
            and self.footer_offset
            == self.record_table_offset
            + (self.frame_record_count + self.close_marker_count)
            * _META_FIXED_RECORD.size
        )
        try:
            footer_span_exact = (
                self.footer_offset is not None
                and self.footer_end_offset
                == self.footer_offset
                + 5
                + sum(
                    4 + len(name.encode("utf-8")) + 1
                    for name in self.footer_filenames
                )
            )
        except UnicodeEncodeError:
            footer_span_exact = False
        return (
            not self.issues
            and bool(self.chunks)
            and table_span_exact
            and footer_span_exact
            and self.chunk_index_set == expected_indices
            and self.declared_chunk_count == len(self.chunks)
            and self.footer_filenames
            == tuple(item.footer_filename for item in self.chunks)
            and self.frame_record_count
            == sum(item.frame_count for item in self.chunks)
            and self.close_marker_count == len(self.chunks)
            and self.entry_order_mismatch_count == 0
            and self.footer_end_offset == self.payload_bytes
            and all(item.geometry_eligible for item in self.chunks)
        )

    @property
    def expected_graph_bound(self) -> bool:
        return (
            self.expected_chunk_filenames is not None
            and self.expected_chunk_byte_counts is not None
        )

    @property
    def exact_graph_join(self) -> bool:
        return (
            self.expected_graph_bound
            and len(self.chunks) == len(self.expected_chunk_filenames or ())
            and len(self.chunks) == len(self.expected_chunk_byte_counts or ())
            and all(
                item.filename_matches is True and item.bytes_match is True
                for item in self.chunks
            )
        )

    @property
    def consumption_eligible(self) -> bool:
        return self.internal_geometry_eligible and self.exact_graph_join

    def _document(self) -> dict[str, Any]:
        return {
            "schema": XETHRU_META_EVIDENCE_SCHEMA,
            "diagnostic_only": True,
            "scientific_authority": False,
            "record_format": XETHRU_META_RECORD_FORMAT,
            "record_bytes": XETHRU_RECORD_BYTES,
            "close_timestamp_policy": XETHRU_META_CLOSE_TIMESTAMP_POLICY,
            "payload_bytes": self.payload_bytes,
            "record_table_offset": self.record_table_offset,
            "footer_offset": self.footer_offset,
            "footer_end_offset": self.footer_end_offset,
            "declared_chunk_count": self.declared_chunk_count,
            "footer_filenames": list(self.footer_filenames),
            "frame_record_count": self.frame_record_count,
            "close_marker_count": self.close_marker_count,
            "entry_order_mismatch_count": self.entry_order_mismatch_count,
            "chunk_index_set": list(self.chunk_index_set),
            "expected_chunk_filenames": (
                None
                if self.expected_chunk_filenames is None
                else list(self.expected_chunk_filenames)
            ),
            "expected_chunk_byte_counts": (
                None
                if self.expected_chunk_byte_counts is None
                else list(self.expected_chunk_byte_counts)
            ),
            "chunks": [item.to_dict() for item in self.chunks],
            "internal_geometry_eligible": self.internal_geometry_eligible,
            "expected_graph_bound": self.expected_graph_bound,
            "exact_graph_join": self.exact_graph_join,
            "consumption_eligible": self.consumption_eligible,
            "issues": [item.to_dict() for item in self.issues],
        }

    def to_dict(self) -> dict[str, Any]:
        return validate_xethru_meta_evidence(self._document())


@dataclass(frozen=True, slots=True)
class XeThruMeta:
    """Parsed subset of a XeThru recorder version-13 metadata file."""

    path: Path
    magic: int | None
    version: int | None
    start_epoch_ms: int | None
    device_name: str | None
    relative_timestamps_ms: np.ndarray = field(repr=False, compare=False)
    timestamp_source: str = "fallback_40hz"
    chunk_filenames: tuple[str, ...] = ()
    declared_chunk_count: int | None = None
    frame_record_count: int = 0
    end_marker_count: int = 0
    timestamp_repairs: int = 0
    metadata_evidence: XeThruMetaEvidence | None = field(
        default=None, repr=False, compare=False
    )
    warnings: tuple[str, ...] = ()

    @property
    def start_datetime_utc(self) -> datetime | None:
        if self.start_epoch_ms is None:
            return None
        return datetime.fromtimestamp(self.start_epoch_ms / 1000.0, tz=timezone.utc)

    def start_datetime(self, timezone_name: str | tzinfo = "Asia/Seoul") -> datetime | None:
        value = self.start_datetime_utc
        return value.astimezone(_coerce_timezone(timezone_name)) if value else None

    @property
    def duration_seconds(self) -> float:
        if self.relative_timestamps_ms.size == 0:
            return 0.0
        # Include one nominal frame interval in the half-open signal duration.
        if self.relative_timestamps_ms.size > 1:
            deltas = np.diff(self.relative_timestamps_ms)
            positive = deltas[deltas > 0]
            period = float(np.median(positive)) if positive.size else 25.0
        else:
            period = 25.0
        return float(self.relative_timestamps_ms[-1] + period) / 1000.0

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "magic": self.magic,
            "version": self.version,
            "start_epoch_ms": self.start_epoch_ms,
            "device_name": self.device_name,
            "timestamp_source": self.timestamp_source,
            "frame_record_count": self.frame_record_count,
            "duration_seconds": self.duration_seconds,
            "timestamp_repairs": self.timestamp_repairs,
            "chunk_filenames": list(self.chunk_filenames),
            "declared_chunk_count": self.declared_chunk_count,
            "end_marker_count": self.end_marker_count,
            "metadata_evidence": (
                None
                if self.metadata_evidence is None
                else self.metadata_evidence.to_dict()
            ),
            "warnings": list(self.warnings),
        }


def validate_xethru_meta_evidence(
    value: XeThruMetaEvidence | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact v13 table/footer/consumed-chunk join evidence."""

    document = value._document() if type(value) is XeThruMetaEvidence else value
    if not isinstance(document, Mapping):
        raise DataFormatError(
            "XeThru metadata evidence must be an object",
            code="xethru_metadata_evidence_invalid",
        )
    expected_keys = {
        "schema",
        "diagnostic_only",
        "scientific_authority",
        "record_format",
        "record_bytes",
        "close_timestamp_policy",
        "payload_bytes",
        "record_table_offset",
        "footer_offset",
        "footer_end_offset",
        "declared_chunk_count",
        "footer_filenames",
        "frame_record_count",
        "close_marker_count",
        "entry_order_mismatch_count",
        "chunk_index_set",
        "expected_chunk_filenames",
        "expected_chunk_byte_counts",
        "chunks",
        "internal_geometry_eligible",
        "expected_graph_bound",
        "exact_graph_join",
        "consumption_eligible",
        "issues",
    }
    if set(document) != expected_keys:
        raise DataFormatError(
            "XeThru metadata evidence fields are invalid",
            code="xethru_metadata_evidence_invalid",
        )
    if (
        document.get("schema") != XETHRU_META_EVIDENCE_SCHEMA
        or document.get("diagnostic_only") is not True
        or document.get("scientific_authority") is not False
        or document.get("record_format") != XETHRU_META_RECORD_FORMAT
        or document.get("record_bytes") != XETHRU_RECORD_BYTES
        or document.get("close_timestamp_policy")
        != XETHRU_META_CLOSE_TIMESTAMP_POLICY
    ):
        raise DataFormatError(
            "XeThru metadata evidence contract constants are invalid",
            code="xethru_metadata_evidence_invalid",
        )

    def exact_int(name: str, *, nullable: bool = False) -> int | None:
        item = document.get(name)
        if nullable and item is None:
            return None
        if type(item) is not int or item < 0:
            raise DataFormatError(
                f"XeThru metadata evidence {name} is invalid",
                code="xethru_metadata_evidence_invalid",
            )
        return item

    payload_bytes = exact_int("payload_bytes")
    record_table_offset = exact_int("record_table_offset", nullable=True)
    footer_offset = exact_int("footer_offset", nullable=True)
    footer_end_offset = exact_int("footer_end_offset", nullable=True)
    declared_count = exact_int("declared_chunk_count", nullable=True)
    frame_count = exact_int("frame_record_count")
    close_count = exact_int("close_marker_count")
    order_mismatches = exact_int("entry_order_mismatch_count")
    assert payload_bytes is not None
    if payload_bytes <= 0 or any(
        item is not None and item > payload_bytes
        for item in (record_table_offset, footer_offset, footer_end_offset)
    ):
        raise DataFormatError(
            "XeThru metadata evidence byte offsets are invalid",
            code="xethru_metadata_evidence_invalid",
        )
    if (
        record_table_offset is not None
        and footer_offset is not None
        and record_table_offset > footer_offset
    ) or (
        footer_offset is not None
        and footer_end_offset is not None
        and footer_offset > footer_end_offset
    ):
        raise DataFormatError(
            "XeThru metadata evidence offset ordering is invalid",
            code="xethru_metadata_evidence_invalid",
        )

    def string_list(
        name: str, *, nullable: bool = False, allow_empty: bool = False
    ) -> tuple[str, ...] | None:
        item = document.get(name)
        if nullable and item is None:
            return None
        if not isinstance(item, list) or any(
            type(entry) is not str or (not allow_empty and not entry) for entry in item
        ):
            raise DataFormatError(
                f"XeThru metadata evidence {name} is invalid",
                code="xethru_metadata_evidence_invalid",
            )
        return tuple(item)

    footer_names = string_list("footer_filenames", allow_empty=True)
    expected_names = string_list("expected_chunk_filenames", nullable=True)
    assert footer_names is not None
    try:
        encoded_footer_name_lengths = tuple(
            len(name.encode("utf-8")) for name in footer_names
        )
        for name in expected_names or ():
            name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DataFormatError(
            "XeThru metadata filename inventory is not valid UTF-8",
            code="xethru_metadata_evidence_invalid",
        ) from exc
    for name in expected_names or ():
        candidate = PurePosixPath(name)
        if candidate.name != name or name in {".", ".."} or "\x00" in name:
            raise DataFormatError(
                "XeThru metadata evidence filename inventory is invalid",
                code="xethru_metadata_evidence_invalid",
            )
    footer_inventory_valid = all(
        bool(name)
        and PurePosixPath(name).name == name
        and name not in {".", ".."}
        and "\x00" not in name
        for name in footer_names
    ) and len(set(footer_names)) == len(footer_names)
    if expected_names is not None and len(set(expected_names)) != len(expected_names):
        raise DataFormatError(
            "XeThru metadata expected graph repeats a filename",
            code="xethru_metadata_evidence_invalid",
        )

    expected_bytes_raw = document.get("expected_chunk_byte_counts")
    if expected_bytes_raw is None:
        expected_bytes: tuple[int, ...] | None = None
    elif not isinstance(expected_bytes_raw, list) or any(
        type(item) is not int
        or item <= 0
        or item % XETHRU_RECORD_BYTES != 0
        for item in expected_bytes_raw
    ):
        raise DataFormatError(
            "XeThru metadata expected chunk bytes are invalid",
            code="xethru_metadata_evidence_invalid",
        )
    else:
        expected_bytes = tuple(expected_bytes_raw)
    expected_graph_bound = expected_names is not None and expected_bytes is not None
    if (expected_names is None) != (expected_bytes is None):
        raise DataFormatError(
            "XeThru metadata expected graph is partially bound",
            code="xethru_metadata_evidence_invalid",
        )

    index_set_raw = document.get("chunk_index_set")
    if not isinstance(index_set_raw, list) or any(
        type(item) is not int or item < 0 for item in index_set_raw
    ):
        raise DataFormatError(
            "XeThru metadata chunk index set is invalid",
            code="xethru_metadata_evidence_invalid",
        )
    index_set = tuple(index_set_raw)
    if index_set != tuple(sorted(set(index_set))):
        raise DataFormatError(
            "XeThru metadata chunk index set is not canonical",
            code="xethru_metadata_evidence_invalid",
        )

    chunk_documents = document.get("chunks")
    if not isinstance(chunk_documents, list):
        raise DataFormatError(
            "XeThru metadata chunks are invalid",
            code="xethru_metadata_evidence_invalid",
        )
    chunk_keys = {
        "chunk_index",
        "footer_filename",
        "frame_count",
        "record_bytes",
        "record_size_mismatch_count",
        "file_offset_mismatch_count",
        "logical_end_mismatch_count",
        "metadata_chunk_bytes",
        "logical_start",
        "logical_end",
        "close_marker_count",
        "close_encoded_data_size",
        "close_file_offset",
        "close_logical_end",
        "last_frame_timestamp_ms",
        "close_timestamp_ms",
        "expected_filename",
        "expected_bytes",
        "filename_matches",
        "bytes_match",
        "geometry_eligible",
    }
    cumulative = 0
    computed_frame_count = 0
    computed_close_count = 0
    all_geometry_eligible = True
    all_joined = expected_graph_bound and len(chunk_documents) == len(
        expected_names or ()
    ) == len(expected_bytes or ())
    observed_chunk_names: list[str | None] = []
    for expected_index, chunk in enumerate(chunk_documents):
        if not isinstance(chunk, Mapping) or set(chunk) != chunk_keys:
            raise DataFormatError(
                "XeThru metadata chunk evidence fields are invalid",
                code="xethru_metadata_evidence_invalid",
            )

        def chunk_int(name: str, *, nullable: bool = False) -> int | None:
            item = chunk.get(name)
            if nullable and item is None:
                return None
            if type(item) is not int or item < 0:
                raise DataFormatError(
                    f"XeThru metadata chunk {expected_index} {name} is invalid",
                    code="xethru_metadata_evidence_invalid",
                )
            return item

        chunk_index = chunk_int("chunk_index")
        chunk_frames = chunk_int("frame_count")
        record_size_mismatches = chunk_int("record_size_mismatch_count")
        file_offset_mismatches = chunk_int("file_offset_mismatch_count")
        logical_end_mismatches = chunk_int("logical_end_mismatch_count")
        metadata_bytes = chunk_int("metadata_chunk_bytes")
        logical_start = chunk_int("logical_start")
        logical_end = chunk_int("logical_end")
        chunk_closes = chunk_int("close_marker_count")
        close_size = chunk_int("close_encoded_data_size", nullable=True)
        close_file_offset = chunk_int("close_file_offset", nullable=True)
        close_logical_end = chunk_int("close_logical_end", nullable=True)
        last_timestamp = chunk_int("last_frame_timestamp_ms", nullable=True)
        close_timestamp = chunk_int("close_timestamp_ms", nullable=True)
        assert chunk_index is not None and chunk_frames is not None
        assert record_size_mismatches is not None
        assert file_offset_mismatches is not None
        assert logical_end_mismatches is not None
        assert metadata_bytes is not None and logical_start is not None
        assert logical_end is not None and chunk_closes is not None
        if any(
            item is not None and item > 0xFFFFFFFF
            for item in (last_timestamp, close_timestamp)
        ):
            raise DataFormatError(
                "XeThru metadata chunk timestamp is invalid",
                code="xethru_metadata_evidence_invalid",
            )
        footer_filename = chunk.get("footer_filename")
        expected_filename = chunk.get("expected_filename")
        chunk_expected_bytes = chunk.get("expected_bytes")
        if (
            (footer_filename is not None and type(footer_filename) is not str)
            or (expected_filename is not None and type(expected_filename) is not str)
            or (
                chunk_expected_bytes is not None
                and (
                    type(chunk_expected_bytes) is not int
                    or chunk_expected_bytes <= 0
                )
            )
            or chunk.get("record_bytes") != XETHRU_RECORD_BYTES
        ):
            raise DataFormatError(
                "XeThru metadata chunk filename/size fields are invalid",
                code="xethru_metadata_evidence_invalid",
            )
        if expected_graph_bound and expected_index < len(expected_names or ()):
            if (
                expected_filename != (expected_names or ())[expected_index]
                or chunk_expected_bytes != (expected_bytes or ())[expected_index]
            ):
                raise DataFormatError(
                    "XeThru metadata chunk expected-graph cross-link is invalid",
                    code="xethru_metadata_evidence_invalid",
                )
        elif expected_filename is not None or chunk_expected_bytes is not None:
            raise DataFormatError(
                "XeThru metadata unbound chunk claims an expected graph",
                code="xethru_metadata_evidence_invalid",
            )
        expected_geometry = (
            chunk_frames > 0
            and record_size_mismatches == 0
            and file_offset_mismatches == 0
            and logical_end_mismatches == 0
            and metadata_bytes == chunk_frames * XETHRU_RECORD_BYTES
            and logical_start == cumulative
            and logical_end == cumulative + metadata_bytes
            and chunk_closes == 1
            and close_size == 0
            and close_file_offset == metadata_bytes
            and close_logical_end == logical_end
            and last_timestamp is not None
            and close_timestamp is not None
        )
        if chunk.get("geometry_eligible") is not expected_geometry:
            raise DataFormatError(
                "XeThru metadata chunk geometry eligibility mismatch",
                code="xethru_metadata_evidence_invalid",
            )
        filename_matches = (
            None
            if expected_filename is None
            else footer_filename == expected_filename
        )
        bytes_match = (
            None
            if chunk_expected_bytes is None
            else metadata_bytes == chunk_expected_bytes
        )
        if (
            chunk.get("filename_matches") is not filename_matches
            or chunk.get("bytes_match") is not bytes_match
        ):
            raise DataFormatError(
                "XeThru metadata chunk graph-join predicate mismatch",
                code="xethru_metadata_evidence_invalid",
            )
        all_geometry_eligible &= expected_geometry
        all_joined &= filename_matches is True and bytes_match is True
        computed_frame_count += chunk_frames
        computed_close_count += chunk_closes
        cumulative = logical_end
        observed_chunk_names.append(footer_filename)
        if chunk_index != expected_index:
            all_geometry_eligible = False

    issues = document.get("issues")
    if not isinstance(issues, list):
        raise DataFormatError(
            "XeThru metadata evidence issues are invalid",
            code="xethru_metadata_evidence_invalid",
        )
    issue_codes: set[str] = set()
    for issue in issues:
        if (
            not isinstance(issue, Mapping)
            or set(issue) != {"code", "severity", "message"}
            or type(issue.get("code")) is not str
            or not issue.get("code")
            or issue.get("severity") != "error"
            or type(issue.get("message")) is not str
            or not issue.get("message")
            or issue["code"] in issue_codes
        ):
            raise DataFormatError(
                "XeThru metadata evidence issue entry is invalid",
                code="xethru_metadata_evidence_invalid",
            )
        issue_codes.add(issue["code"])
    internal_eligible = (
        not issues
        and bool(chunk_documents)
        and record_table_offset is not None
        and footer_offset
        == record_table_offset
        + (frame_count + close_count) * _META_FIXED_RECORD.size
        and footer_end_offset
        == footer_offset
        + 5
        + sum(4 + length + 1 for length in encoded_footer_name_lengths)
        and index_set == tuple(range(len(chunk_documents)))
        and declared_count == len(chunk_documents)
        and footer_inventory_valid
        and footer_names == tuple(observed_chunk_names)
        and computed_frame_count == frame_count
        and computed_close_count == close_count == len(chunk_documents)
        and order_mismatches == 0
        and footer_end_offset == payload_bytes
        and all_geometry_eligible
    )
    exact_join = expected_graph_bound and all_joined
    consumption_eligible = internal_eligible and exact_join
    if (
        document.get("internal_geometry_eligible") is not internal_eligible
        or document.get("expected_graph_bound") is not expected_graph_bound
        or document.get("exact_graph_join") is not exact_join
        or document.get("consumption_eligible") is not consumption_eligible
    ):
        raise DataFormatError(
            "XeThru metadata evidence eligibility predicates are invalid",
            code="xethru_metadata_evidence_invalid",
        )
    return {
        key: (
            [dict(item) for item in item_value]
            if key == "issues"
            else item_value
        )
        for key, item_value in document.items()
    }


def parse_xethru_meta_bytes(
    payload: bytes,
    *,
    source_path: str | Path,
    expected_frames: int | None = None,
    expected_chunk_filenames: Sequence[str] | None = None,
    expected_chunk_byte_counts: Sequence[int] | None = None,
    fallback_rate_hz: float = RADAR_SAMPLE_RATE_HZ,
    strict: bool = False,
) -> XeThruMeta:
    """Parse exact XeThru metadata bytes without reopening their source path.

    The fixed 33-byte frame entries contain chunk id/size, file and logical
    byte offsets, and a relative millisecond timestamp.  Type-2 entries close
    chunks.  The final variable-size type-3 footer lists one or more data
    filenames.  Timestamp entries remain usable when that footer is missing or
    truncated.  If entries themselves are unusable or disagree with
    ``expected_frames``, a deterministic 40 Hz timeline is returned.

    Supplying both expected chunk arguments performs the consumed-graph join:
    footer name, per-chunk frame geometry, and byte count must all agree.  A
    successful join remains diagnostic provenance; it does not establish
    synchronization or scientific authority.
    """

    if type(payload) is not bytes:
        raise TypeError("XeThru metadata payload must be exact bytes")
    meta_path = Path(source_path)
    warnings: list[str] = []
    issues: list[XeThruMetaIssue] = []

    def add_warning(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    def add_issue(code: str, message: str) -> None:
        if all(item.code != code for item in issues):
            issues.append(XeThruMetaIssue(code=code, message=message))
        add_warning(message)

    if (expected_chunk_filenames is None) != (expected_chunk_byte_counts is None):
        raise ValueError(
            "expected_chunk_filenames and expected_chunk_byte_counts must be bound together"
        )
    expected_names: tuple[str, ...] | None = None
    expected_bytes: tuple[int, ...] | None = None
    if expected_chunk_filenames is not None:
        if isinstance(expected_chunk_filenames, (str, bytes)) or not isinstance(
            expected_chunk_filenames, Sequence
        ):
            raise TypeError("expected_chunk_filenames must be a sequence")
        if isinstance(expected_chunk_byte_counts, (str, bytes)) or not isinstance(
            expected_chunk_byte_counts, Sequence
        ):
            raise TypeError("expected_chunk_byte_counts must be a sequence")
        expected_names = tuple(expected_chunk_filenames)
        assert expected_chunk_byte_counts is not None
        expected_bytes = tuple(expected_chunk_byte_counts)
        if (
            not expected_names
            or len(expected_names) != len(expected_bytes)
            or any(
                type(name) is not str
                or not name
                or PurePosixPath(name).name != name
                or "\x00" in name
                for name in expected_names
            )
            or len(set(expected_names)) != len(expected_names)
            or any(
                type(byte_count) is not int
                or byte_count <= 0
                or byte_count % XETHRU_RECORD_BYTES != 0
                for byte_count in expected_bytes
            )
        ):
            raise ValueError("expected XeThru chunk graph is invalid")
        try:
            for name in expected_names:
                name.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("expected XeThru chunk graph is invalid") from exc
        graph_frame_count = sum(
            byte_count // XETHRU_RECORD_BYTES for byte_count in expected_bytes
        )
        if expected_frames is not None and graph_frame_count != expected_frames:
            raise ValueError("expected frame total disagrees with expected chunk bytes")

    magic: int | None = None
    version: int | None = None
    start_epoch_ms: int | None = None
    device_name: str | None = None
    raw_timestamps: list[int] = []
    chunk_filenames: list[str] = []
    declared_chunks: int | None = None
    end_markers = 0
    timestamp_repairs = 0

    record_offset: int | None = None
    if len(payload) >= 16:
        magic, version = struct.unpack_from("<II", payload, 0)
        start_epoch_ms = struct.unpack_from("<Q", payload, 8)[0]
        if not 946_684_800_000 <= start_epoch_ms <= 4_102_444_800_000:
            add_issue(
                "xethru_start_epoch_implausible",
                f"implausible start epoch milliseconds: {start_epoch_ms}",
            )
            start_epoch_ms = None
        if magic != XETHRU_MAGIC:
            add_issue(
                "xethru_magic_invalid", f"unexpected metadata magic 0x{magic:08x}"
            )
        if version != XETHRU_META_VERSION:
            add_issue(
                "xethru_version_invalid",
                f"metadata version {version}, expected {XETHRU_META_VERSION}",
            )
    else:
        add_issue(
            "xethru_header_truncated",
            f"metadata header truncated ({len(payload)} bytes)",
        )

    if len(payload) >= _META_HEADER_NAME_LENGTH_OFFSET + 4:
        name_length = struct.unpack_from(
            "<I", payload, _META_HEADER_NAME_LENGTH_OFFSET
        )[0]
        name_start = _META_HEADER_NAME_LENGTH_OFFSET + 4
        name_end = name_start + name_length
        if 0 < name_length <= 4096 and name_end <= len(payload):
            encoded_device_name = payload[name_start:name_end]
            if not encoded_device_name.endswith(b"\x00"):
                add_issue(
                    "xethru_device_name_terminator_missing",
                    "metadata device name is not NUL-terminated",
                )
            else:
                if b"\x00" in encoded_device_name[:-1]:
                    add_issue(
                        "xethru_device_name_embedded_nul",
                        "metadata device name contains more than one NUL",
                    )
                encoded_device_name = encoded_device_name[:-1]
            try:
                device_name = encoded_device_name.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                device_name = encoded_device_name.decode("utf-8", errors="replace")
                add_issue(
                    "xethru_device_name_utf8_invalid",
                    "metadata device name is not valid UTF-8",
                )
            # Recorder v13 writes four zero bytes after the NUL-inclusive name.
            candidate = name_end + 4
            if candidate + _META_FIXED_RECORD.size <= len(payload):
                record_offset = candidate
            if payload[name_end:candidate] != b"\x00" * 4:
                add_issue(
                    "xethru_header_padding_invalid",
                    "metadata device-name padding is not four zero bytes",
                )
        else:
            add_issue(
                "xethru_device_name_length_invalid",
                f"invalid device-name length: {name_length}",
            )

    # A conservative recovery path for a damaged name-length field.  Requiring
    # the complete kind/format marker avoids matching arbitrary float payload.
    marker = b"\x01\x00\x04\x00\x00"
    if record_offset is None or payload[record_offset : record_offset + 5] != marker:
        recovered = payload.find(marker, 80, min(len(payload), 4096))
        if recovered >= 0:
            record_offset = recovered
            add_issue(
                "xethru_record_table_recovered",
                "recovered metadata record table by signature",
            )
        else:
            record_offset = None
            add_issue(
                "xethru_record_table_missing", "metadata frame table not found"
            )

    entries: list[tuple[int, int, int, int, int, int]] = []
    footer_offset: int | None = None
    if record_offset is not None:
        cursor = record_offset
        while cursor + _META_FIXED_RECORD.size <= len(payload):
            if payload[cursor] == 3:
                footer_offset = cursor
                break
            entry = _META_FIXED_RECORD.unpack_from(payload, cursor)
            kind, record_format, _encoded_size, _file_offset, _logical_end, _timestamp = entry
            if kind not in (1, 2) or record_format != XETHRU_META_RECORD_FORMAT:
                footer_offset = cursor
                add_issue(
                    "xethru_record_entry_tag_invalid",
                    f"metadata entry at byte {cursor} has kind/format {kind}/{record_format}",
                )
                break
            entries.append(entry)
            cursor += _META_FIXED_RECORD.size
        else:
            footer_offset = cursor

    frame_entries_by_chunk: dict[
        int, list[tuple[int, int, int, int, int, int]]
    ] = {}
    close_entries_by_chunk: dict[
        int, list[tuple[int, int, int, int, int, int]]
    ] = {}
    observed_entry_order: list[tuple[int, int]] = []
    for entry in entries:
        kind, _record_format, encoded_size, _file_offset, _logical_end, timestamp = entry
        chunk_index = int(encoded_size >> 32)
        observed_entry_order.append((int(kind), chunk_index))
        if kind == 1:
            frame_entries_by_chunk.setdefault(chunk_index, []).append(entry)
            raw_timestamps.append(int(timestamp))
        else:
            close_entries_by_chunk.setdefault(chunk_index, []).append(entry)
            end_markers += 1
    chunk_indices = tuple(
        sorted(set(frame_entries_by_chunk) | set(close_entries_by_chunk))
    )
    if chunk_indices != tuple(range(len(chunk_indices))):
        add_issue(
            "xethru_chunk_indices_noncontiguous",
            f"metadata chunk indices are not contiguous from zero: {list(chunk_indices)}",
        )
    expected_entry_order: list[tuple[int, int]] = []
    for chunk_index in chunk_indices:
        expected_entry_order.extend(
            [(1, chunk_index)] * len(frame_entries_by_chunk.get(chunk_index, ()))
        )
        expected_entry_order.extend(
            [(2, chunk_index)] * len(close_entries_by_chunk.get(chunk_index, ()))
        )
    overlap = min(len(observed_entry_order), len(expected_entry_order))
    entry_order_mismatch_count = sum(
        observed_entry_order[index] != expected_entry_order[index]
        for index in range(overlap)
    ) + abs(len(observed_entry_order) - len(expected_entry_order))
    if entry_order_mismatch_count:
        add_issue(
            "xethru_entry_order_invalid",
            f"metadata table has {entry_order_mismatch_count} chunk-transition/order mismatch(es)",
        )

    # Footer: byte kind=3, uint32 filename count, then count repetitions of
    # uint32 NUL-inclusive length + UTF-8 filename.  Parse each independently so
    # a partially written last filename does not invalidate earlier names.
    if footer_offset is not None:
        cursor = footer_offset
        if cursor + 5 <= len(payload) and payload[cursor] == 3:
            declared_chunks = struct.unpack_from("<I", payload, cursor + 1)[0]
            cursor += 5
            if declared_chunks > 100_000:
                add_issue(
                    "xethru_footer_count_implausible",
                    f"implausible footer chunk count: {declared_chunks}",
                )
                declared_chunks = None
            else:
                for index in range(declared_chunks):
                    if cursor + 4 > len(payload):
                        add_issue(
                            "xethru_footer_truncated",
                            f"footer truncated before chunk name {index}",
                        )
                        break
                    length = struct.unpack_from("<I", payload, cursor)[0]
                    cursor += 4
                    if length <= 0 or length > 4096 or cursor + length > len(payload):
                        add_issue(
                            "xethru_footer_name_length_invalid",
                            f"invalid/truncated footer chunk name {index}",
                        )
                        break
                    encoded_filename = payload[cursor : cursor + length]
                    if not encoded_filename.endswith(b"\x00"):
                        add_issue(
                            "xethru_footer_name_terminator_missing",
                            f"footer chunk name {index} is not NUL-terminated",
                        )
                    else:
                        if b"\x00" in encoded_filename[:-1]:
                            add_issue(
                                "xethru_footer_name_embedded_nul",
                                f"footer chunk name {index} contains more than one NUL",
                            )
                        encoded_filename = encoded_filename[:-1]
                    try:
                        filename = encoded_filename.decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        filename = encoded_filename.decode("utf-8", errors="replace")
                        add_issue(
                            "xethru_footer_name_utf8_invalid",
                            f"footer chunk name {index} is not valid UTF-8",
                        )
                    if (
                        not filename
                        or PurePosixPath(filename).name != filename
                        or filename in {".", ".."}
                    ):
                        add_issue(
                            "xethru_footer_name_invalid",
                            f"footer chunk name {index} is not a canonical filename",
                        )
                    chunk_filenames.append(filename)
                    cursor += length
            if cursor != len(payload):
                add_issue(
                    "xethru_footer_trailing_bytes",
                    f"metadata footer leaves {len(payload) - cursor} trailing byte(s)",
                )
            footer_end_offset = cursor
        else:
            add_issue(
                "xethru_footer_missing", "metadata filename footer missing or invalid"
            )
            footer_end_offset = cursor
    else:
        footer_end_offset = None
        add_issue("xethru_footer_missing", "metadata filename footer missing")

    if declared_chunks is not None and declared_chunks != len(chunk_filenames):
        add_issue(
            "xethru_footer_inventory_incomplete",
            f"footer parsed {len(chunk_filenames)} of {declared_chunks} declared filename(s)",
        )
    if len(set(chunk_filenames)) != len(chunk_filenames):
        add_issue(
            "xethru_footer_filename_duplicate",
            "metadata footer repeats a chunk filename",
        )
    if declared_chunks != len(chunk_indices):
        add_issue(
            "xethru_footer_chunk_count_mismatch",
            f"footer chunk count {declared_chunks} differs from table chunk count {len(chunk_indices)}",
        )

    chunk_evidence: list[XeThruMetaChunkEvidence] = []
    cumulative_logical_bytes = 0
    for chunk_index in chunk_indices:
        frames = frame_entries_by_chunk.get(chunk_index, [])
        closes = close_entries_by_chunk.get(chunk_index, [])
        record_size_mismatches = 0
        file_offset_mismatches = 0
        logical_end_mismatches = 0
        for local_index, frame in enumerate(frames):
            _kind, _format, encoded_size, file_offset, logical_end, _timestamp = frame
            data_size = int(encoded_size & 0xFFFFFFFF)
            if data_size != XETHRU_RECORD_BYTES:
                record_size_mismatches += 1
            if int(file_offset) != local_index * XETHRU_RECORD_BYTES:
                file_offset_mismatches += 1
            if int(logical_end) != (
                cumulative_logical_bytes + (local_index + 1) * XETHRU_RECORD_BYTES
            ):
                logical_end_mismatches += 1
        metadata_chunk_bytes = len(frames) * XETHRU_RECORD_BYTES
        logical_start = cumulative_logical_bytes
        logical_end = logical_start + metadata_chunk_bytes
        close = closes[0] if closes else None
        close_size = None if close is None else int(close[2] & 0xFFFFFFFF)
        close_file_offset = None if close is None else int(close[3])
        close_logical_end = None if close is None else int(close[4])
        last_frame_timestamp = None if not frames else int(frames[-1][5])
        close_timestamp = None if close is None else int(close[5])
        footer_filename = (
            chunk_filenames[chunk_index]
            if 0 <= chunk_index < len(chunk_filenames)
            else None
        )
        expected_filename = (
            expected_names[chunk_index]
            if expected_names is not None and chunk_index < len(expected_names)
            else None
        )
        expected_byte_count = (
            expected_bytes[chunk_index]
            if expected_bytes is not None and chunk_index < len(expected_bytes)
            else None
        )
        item = XeThruMetaChunkEvidence(
            chunk_index=chunk_index,
            footer_filename=footer_filename,
            frame_count=len(frames),
            record_size_mismatch_count=record_size_mismatches,
            file_offset_mismatch_count=file_offset_mismatches,
            logical_end_mismatch_count=logical_end_mismatches,
            metadata_chunk_bytes=metadata_chunk_bytes,
            logical_start=logical_start,
            logical_end=logical_end,
            close_marker_count=len(closes),
            close_encoded_data_size=close_size,
            close_file_offset=close_file_offset,
            close_logical_end=close_logical_end,
            last_frame_timestamp_ms=last_frame_timestamp,
            close_timestamp_ms=close_timestamp,
            expected_filename=expected_filename,
            expected_bytes=expected_byte_count,
        )
        chunk_evidence.append(item)
        if not frames:
            add_issue(
                "xethru_chunk_empty", f"metadata chunk {chunk_index} has no frames"
            )
        if record_size_mismatches:
            add_issue(
                "xethru_frame_record_size_invalid",
                f"metadata chunk {chunk_index} has {record_size_mismatches} frame-size mismatch(es)",
            )
        if file_offset_mismatches:
            add_issue(
                "xethru_file_offsets_invalid",
                f"metadata chunk {chunk_index} has {file_offset_mismatches} file-offset mismatch(es)",
            )
        if logical_end_mismatches:
            add_issue(
                "xethru_logical_ends_invalid",
                f"metadata chunk {chunk_index} has {logical_end_mismatches} logical-end mismatch(es)",
            )
        if len(closes) != 1:
            add_issue(
                "xethru_close_marker_count_invalid",
                f"metadata chunk {chunk_index} has {len(closes)} close marker(s), expected one",
            )
        elif (
            close_size != 0
            or close_file_offset != metadata_chunk_bytes
            or close_logical_end != logical_end
        ):
            add_issue(
                "xethru_close_marker_geometry_invalid",
                f"metadata chunk {chunk_index} close marker disagrees with chunk geometry",
            )
        cumulative_logical_bytes = logical_end

    if expected_names is not None:
        if len(expected_names) != len(chunk_evidence):
            add_issue(
                "xethru_expected_chunk_count_mismatch",
                f"consumed graph has {len(expected_names)} chunks but metadata has {len(chunk_evidence)}",
            )
        for item in chunk_evidence:
            if item.filename_matches is not True:
                add_issue(
                    "xethru_expected_filename_mismatch",
                    f"metadata chunk {item.chunk_index} filename does not match consumed graph",
                )
            if item.bytes_match is not True:
                add_issue(
                    "xethru_expected_bytes_mismatch",
                    f"metadata chunk {item.chunk_index} byte count does not match consumed graph",
                )

    evidence = XeThruMetaEvidence(
        payload_bytes=len(payload),
        record_table_offset=record_offset,
        footer_offset=footer_offset,
        footer_end_offset=footer_end_offset,
        declared_chunk_count=declared_chunks,
        footer_filenames=tuple(chunk_filenames),
        frame_record_count=len(raw_timestamps),
        close_marker_count=end_markers,
        entry_order_mismatch_count=entry_order_mismatch_count,
        chunk_index_set=chunk_indices,
        expected_chunk_filenames=expected_names,
        expected_chunk_byte_counts=expected_bytes,
        chunks=tuple(chunk_evidence),
        issues=tuple(issues),
    )
    evidence.to_dict()

    use_fallback = not raw_timestamps
    if expected_frames is not None and len(raw_timestamps) != expected_frames:
        add_warning(
            f"metadata/data frame mismatch: {len(raw_timestamps)} != {expected_frames}"
        )
        use_fallback = True

    if use_fallback:
        count = max(0, int(expected_frames if expected_frames is not None else len(raw_timestamps)))
        timestamps = _fallback_timestamps(count, fallback_rate_hz)
        timestamp_source = f"fallback_{fallback_rate_hz:g}hz"
        if not raw_timestamps:
            add_warning(f"using {fallback_rate_hz:g} Hz timestamp fallback")
    else:
        timestamps, timestamp_repairs = _unwrap_relative_timestamps(
            raw_timestamps, 1000.0 / fallback_rate_hz
        )
        timestamp_source = "meta_v13"
        if timestamp_repairs:
            add_warning(
                f"unwrapped {timestamp_repairs} relative-timestamp counter reset(s)"
            )

    strict_contract_ineligible = (
        not evidence.internal_geometry_eligible
        or (evidence.expected_graph_bound and not evidence.exact_graph_join)
    )
    if strict and (warnings or strict_contract_ineligible):
        reasons = list(warnings)
        if strict_contract_ineligible and not reasons:
            reasons.append("XeThru metadata contract is ineligible")
        raise DataFormatError(
            f"{meta_path}: " + "; ".join(reasons),
            code=(
                "xethru_metadata_ineligible"
                if issues or strict_contract_ineligible
                else "xethru_metadata_strict_warning"
            ),
            diagnostics=evidence.to_dict(),
        )

    return XeThruMeta(
        path=meta_path,
        magic=magic,
        version=version,
        start_epoch_ms=start_epoch_ms,
        device_name=device_name,
        relative_timestamps_ms=timestamps,
        timestamp_source=timestamp_source,
        chunk_filenames=tuple(chunk_filenames),
        declared_chunk_count=declared_chunks,
        frame_record_count=len(raw_timestamps),
        end_marker_count=end_markers,
        timestamp_repairs=timestamp_repairs,
        metadata_evidence=evidence,
        warnings=tuple(warnings),
    )


def parse_xethru_meta(
    path: str | Path,
    *,
    expected_frames: int | None = None,
    expected_chunk_filenames: Sequence[str] | None = None,
    expected_chunk_byte_counts: Sequence[int] | None = None,
    fallback_rate_hz: float = RADAR_SAMPLE_RATE_HZ,
    strict: bool = False,
) -> XeThruMeta:
    """Parse a legacy pathname by first materializing one exact byte payload."""

    meta_path = Path(path)
    try:
        payload = meta_path.read_bytes()
    except OSError as exc:
        if strict:
            raise DataFormatError(f"cannot read metadata {meta_path}: {exc}") from exc
        warnings = (f"metadata unreadable: {exc}",)
        count = max(0, int(expected_frames or 0))
        return XeThruMeta(
            path=meta_path,
            magic=None,
            version=None,
            start_epoch_ms=None,
            device_name=None,
            relative_timestamps_ms=_fallback_timestamps(count, fallback_rate_hz),
            warnings=warnings,
        )
    return parse_xethru_meta_bytes(
        payload,
        source_path=meta_path,
        expected_frames=expected_frames,
        expected_chunk_filenames=expected_chunk_filenames,
        expected_chunk_byte_counts=expected_chunk_byte_counts,
        fallback_rate_hz=fallback_rate_hz,
        strict=strict,
    )


class SplitRadarMemmap:
    """Read-only continuous view over one or more XeThru data chunks.

    Integer and contiguous-slice indexing has NumPy semantics.  A slice wholly
    inside one chunk is still a memmap view; a cross-chunk slice allocates only
    the requested region.  Use :meth:`iter_chunks` for zero-copy streaming.
    """

    dtype = XETHRU_RECORD_DTYPE
    ndim = 1

    def __init__(self, paths: Iterable[str | Path], *, strict: bool = True) -> None:
        ordered = tuple(Path(item) for item in paths)
        if not ordered:
            raise ValueError("at least one radar data path is required")
        self.paths = ordered
        self.partial_bytes: tuple[int, ...] = tuple(
            path.stat().st_size % XETHRU_RECORD_BYTES for path in ordered
        )
        if strict and any(self.partial_bytes):
            details = ", ".join(
                f"{path.name}: {remainder} trailing bytes"
                for path, remainder in zip(ordered, self.partial_bytes, strict=True)
                if remainder
            )
            raise DataFormatError(f"incomplete XeThru record(s): {details}")
        self._chunks = tuple(
            np.memmap(
                path,
                dtype=XETHRU_RECORD_DTYPE,
                mode="r",
                shape=(path.stat().st_size // XETHRU_RECORD_BYTES,),
            )
            for path in ordered
        )
        self.chunk_lengths = tuple(len(chunk) for chunk in self._chunks)
        self._ends = np.cumsum(self.chunk_lengths, dtype=np.int64)
        self.shape = (int(self._ends[-1]),)
        self.size = self.shape[0]
        self.nbytes = self.size * XETHRU_RECORD_BYTES

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"SplitRadarMemmap(shape={self.shape}, chunks={self.chunk_lengths}, "
            f"dtype={self.dtype!r})"
        )

    def iter_chunks(self) -> Iterator[np.memmap]:
        yield from self._chunks

    def _resolve_integer(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.size
        if index < 0 or index >= self.size:
            raise IndexError("radar frame index out of range")
        chunk_index = int(np.searchsorted(self._ends, index, side="right"))
        chunk_start = int(self._ends[chunk_index - 1]) if chunk_index else 0
        return chunk_index, index - chunk_start

    def _contiguous_slice(self, start: int, stop: int) -> np.ndarray:
        if start >= stop:
            return np.empty((0,), dtype=self.dtype)
        first_chunk, first_local = self._resolve_integer(start)
        last_chunk, last_local = self._resolve_integer(stop - 1)
        if first_chunk == last_chunk:
            return self._chunks[first_chunk][first_local : last_local + 1]
        pieces: list[np.ndarray] = [self._chunks[first_chunk][first_local:]]
        pieces.extend(self._chunks[first_chunk + 1 : last_chunk])
        pieces.append(self._chunks[last_chunk][: last_local + 1])
        return np.concatenate(pieces)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key not in self.dtype.fields:
                raise ValueError(f"no field of name {key}")
            if len(self._chunks) == 1:
                return self._chunks[0][key]
            return np.concatenate([chunk[key] for chunk in self._chunks], axis=0)
        if isinstance(key, (int, np.integer)):
            chunk, local = self._resolve_integer(int(key))
            return self._chunks[chunk][local]
        if isinstance(key, slice):
            start, stop, step = key.indices(self.size)
            if step == 1:
                return self._contiguous_slice(start, stop)
            return np.asarray(self)[key]
        return np.asarray(self)[key]

    def __array__(self, dtype: np.dtype[Any] | None = None, copy: bool | None = None) -> np.ndarray:
        if len(self._chunks) == 1:
            result = np.asarray(self._chunks[0])
            if copy:
                result = result.copy()
        else:
            result = np.concatenate(self._chunks)
        return result.astype(dtype, copy=False) if dtype is not None else result

    def read(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Read a half-open continuous frame range."""

        return self[slice(start, self.size if stop is None else stop)]


def open_xethru_files(
    paths: str | Path | Iterable[str | Path], *, strict: bool = True
) -> SplitRadarMemmap:
    """Open one or several ordered ``xethru_datafloat`` files as memmaps."""

    if isinstance(paths, (str, Path)):
        items = [Path(paths)]
    else:
        items = [Path(item) for item in paths]
    return SplitRadarMemmap(items, strict=strict)


# Descriptive alias retained for callers that naturally use "load" terminology.
load_xethru_records = open_xethru_files


@dataclass(slots=True)
class LoadedRadarRecording:
    recording_dir: Path
    records: SplitRadarMemmap
    meta: XeThruMeta | None
    warnings: tuple[str, ...] = ()

    @property
    def timestamps_ms(self) -> np.ndarray:
        if self.meta is not None:
            return self.meta.relative_timestamps_ms
        return _fallback_timestamps(len(self.records), RADAR_SAMPLE_RATE_HZ)


def load_xethru_recording(
    recording_dir: str | Path,
    *,
    strict: bool = True,
    fallback_rate_hz: float = RADAR_SAMPLE_RATE_HZ,
) -> LoadedRadarRecording:
    """Open all chunks in a recorder directory in footer/chronological order."""

    directory = Path(recording_dir)
    discovered = sorted(directory.glob("xethru_datafloat_*.dat"), key=_path_sort_key)
    if not discovered:
        raise FileNotFoundError(f"no xethru_datafloat files in {directory}")
    frame_count = sum(path.stat().st_size // XETHRU_RECORD_BYTES for path in discovered)
    meta_path = directory / "xethru_recording_meta.dat"
    meta: XeThruMeta | None = None
    warnings: list[str] = []
    if meta_path.is_file():
        meta = parse_xethru_meta(
            meta_path,
            expected_frames=frame_count,
            fallback_rate_hz=fallback_rate_hz,
            strict=False,
        )
        warnings.extend(meta.warnings)
        if meta.chunk_filenames:
            by_name = {path.name: path for path in discovered}
            ordered: list[Path] = []
            for name in meta.chunk_filenames:
                match = by_name.pop(name, None)
                if match is None:
                    warnings.append(f"metadata chunk missing on disk: {name}")
                else:
                    ordered.append(match)
            ordered.extend(sorted(by_name.values(), key=_path_sort_key))
            discovered = ordered
    else:
        warnings.append("xethru_recording_meta.dat missing; using 40 Hz timestamps")
    records = open_xethru_files(discovered, strict=strict)
    if meta is None:
        meta = XeThruMeta(
            path=meta_path,
            magic=None,
            version=None,
            start_epoch_ms=None,
            device_name=None,
            relative_timestamps_ms=_fallback_timestamps(len(records), fallback_rate_hz),
            timestamp_source=f"fallback_{fallback_rate_hz:g}hz",
            warnings=tuple(warnings),
        )
    return LoadedRadarRecording(
        recording_dir=directory,
        records=records,
        meta=meta,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def parse_biopac_start_datetime(
    path: str | Path, *, timezone_name: str | tzinfo | None = "Asia/Seoul"
) -> datetime:
    """Parse the local acquisition start time embedded in a BIOPAC filename."""

    match = _BIOPAC_DATETIME_RE.search(Path(path).name)
    if not match:
        raise DataFormatError(f"BIOPAC start datetime not found in filename: {path}")
    parsed = datetime.strptime(
        f"{match.group('date')} {match.group('hour')}:{match.group('minute')}:{match.group('second')}",
        "%Y-%m-%d %H:%M:%S",
    )
    if timezone_name is None:
        return parsed
    return parsed.replace(tzinfo=_coerce_timezone(timezone_name))


def _matlab_strings(value: Any) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim == 0:
        return (str(array.item()).strip(),)
    # Scipy normally returns a 1-D Unicode vector.  Object/cell and 2-D char
    # arrays are handled explicitly for MAT exporters with different settings.
    if array.dtype.kind in "US" and array.ndim == 2 and array.shape[0] > 1:
        if all(len(str(item)) <= 1 for item in array.flat):
            return tuple("".join(map(str, row)).strip() for row in array)
    result: list[str] = []
    for item in array.reshape(-1):
        if isinstance(item, np.ndarray):
            if item.dtype.kind in "US":
                text = "".join(map(str, item.reshape(-1)))
            else:
                text = str(item.squeeze())
        else:
            text = str(item)
        result.append(text.strip())
    return tuple(result)


_BIOPAC_ISI_UNIT_FACTORS_TO_MS: dict[str, float] = {
    "ms": 1.0,
    "msec": 1.0,
    "msecs": 1.0,
    "millisecond": 1.0,
    "milliseconds": 1.0,
    "s": 1000.0,
    "sec": 1000.0,
    "secs": 1000.0,
    "second": 1000.0,
    "seconds": 1000.0,
    "us": 0.001,
    "usec": 0.001,
    "usecs": 0.001,
    "microsecond": 0.001,
    "microseconds": 0.001,
}


def _normalize_biopac_isi_unit(value: str) -> str:
    return (
        value.strip()
        .casefold()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace(" ", "")
        .replace(".", "")
    )


@dataclass(frozen=True, slots=True)
class BiopacParserIssue:
    """One deterministic reason that a permissively parsed MAT is ineligible."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": "error", "message": self.message}


@dataclass(frozen=True, slots=True)
class BiopacParserEvidence:
    """Portable evidence for channel identity and sample-interval semantics.

    ``parser_eligible`` is only a format/metadata predicate.  It grants no
    synchronization, label, training, evaluation, or release authority.
    """

    source_data_shape: tuple[int, int]
    effective_data_shape: tuple[int, int]
    orientation: str
    channel_labels: tuple[str, ...]
    channel_units: tuple[str, ...]
    rsp_candidate_indices: tuple[int, ...]
    ecg_candidate_indices: tuple[int, ...]
    rsp_index: int
    ecg_index: int
    source_isi_value_count: int
    source_isi_value: float | None
    source_isi_units: tuple[str, ...]
    source_isi_unit: str | None
    normalized_source_isi_unit: str | None
    conversion_factor_to_ms: float | None
    effective_interval_source: str
    effective_isi_ms: float
    effective_sample_rate_hz: float
    issues: tuple[BiopacParserIssue, ...]

    @property
    def parser_eligible(self) -> bool:
        return not self.issues

    def _document(self) -> dict[str, Any]:
        return {
            "schema": BIOPAC_PARSER_EVIDENCE_SCHEMA,
            "diagnostic_only": True,
            "scientific_authority": False,
            "parser_eligible": self.parser_eligible,
            "source_data_shape": list(self.source_data_shape),
            "effective_data_shape": list(self.effective_data_shape),
            "orientation": self.orientation,
            "channel_count": self.effective_data_shape[1],
            "label_count": len(self.channel_labels),
            "channel_labels": list(self.channel_labels),
            "unit_count": len(self.channel_units),
            "channel_units": list(self.channel_units),
            "rsp_candidate_indices": list(self.rsp_candidate_indices),
            "ecg_candidate_indices": list(self.ecg_candidate_indices),
            "rsp_index": self.rsp_index,
            "ecg_index": self.ecg_index,
            "source_isi_value_count": self.source_isi_value_count,
            "source_isi_value": self.source_isi_value,
            "source_isi_units": list(self.source_isi_units),
            "source_isi_unit": self.source_isi_unit,
            "normalized_source_isi_unit": self.normalized_source_isi_unit,
            "conversion_factor_to_ms": self.conversion_factor_to_ms,
            "canonical_isi_unit": BIOPAC_ISI_CANONICAL_UNIT,
            "effective_interval_source": self.effective_interval_source,
            "effective_isi_ms": self.effective_isi_ms,
            "effective_sample_rate_hz": self.effective_sample_rate_hz,
            "expected_isi_ms": BIOPAC_ISI_MS,
            "issues": [item.to_dict() for item in self.issues],
        }

    def to_dict(self) -> dict[str, Any]:
        return validate_biopac_parser_evidence(self._document())


def _exact_finite_real(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def validate_biopac_parser_evidence(
    value: BiopacParserEvidence | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return one canonical JSON-compatible evidence document."""

    document = value._document() if type(value) is BiopacParserEvidence else value
    if not isinstance(document, Mapping):
        raise DataFormatError(
            "BIOPAC parser evidence must be an object",
            code="biopac_parser_evidence_invalid",
        )
    expected_keys = {
        "schema",
        "diagnostic_only",
        "scientific_authority",
        "parser_eligible",
        "source_data_shape",
        "effective_data_shape",
        "orientation",
        "channel_count",
        "label_count",
        "channel_labels",
        "unit_count",
        "channel_units",
        "rsp_candidate_indices",
        "ecg_candidate_indices",
        "rsp_index",
        "ecg_index",
        "source_isi_value_count",
        "source_isi_value",
        "source_isi_units",
        "source_isi_unit",
        "normalized_source_isi_unit",
        "conversion_factor_to_ms",
        "canonical_isi_unit",
        "effective_interval_source",
        "effective_isi_ms",
        "effective_sample_rate_hz",
        "expected_isi_ms",
        "issues",
    }
    if set(document) != expected_keys:
        raise DataFormatError(
            "BIOPAC parser evidence fields are invalid",
            code="biopac_parser_evidence_invalid",
        )
    if (
        document.get("schema") != BIOPAC_PARSER_EVIDENCE_SCHEMA
        or document.get("diagnostic_only") is not True
        or document.get("scientific_authority") is not False
        or type(document.get("parser_eligible")) is not bool
        or document.get("canonical_isi_unit") != BIOPAC_ISI_CANONICAL_UNIT
        or _exact_finite_real(document.get("expected_isi_ms")) != BIOPAC_ISI_MS
    ):
        raise DataFormatError(
            "BIOPAC parser evidence authority/unit fields are invalid",
            code="biopac_parser_evidence_invalid",
        )

    def _shape(name: str) -> tuple[int, int]:
        raw = document.get(name)
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or any(type(item) is not int or item <= 0 for item in raw)
        ):
            raise DataFormatError(
                f"BIOPAC parser evidence {name} is invalid",
                code="biopac_parser_evidence_invalid",
            )
        return int(raw[0]), int(raw[1])

    source_shape = _shape("source_data_shape")
    effective_shape = _shape("effective_data_shape")
    orientation = document.get("orientation")
    if orientation == "samples_by_channels":
        expected_shape = source_shape
    elif orientation == "transposed_channels_by_samples":
        expected_shape = (source_shape[1], source_shape[0])
    else:
        raise DataFormatError(
            "BIOPAC parser evidence orientation is invalid",
            code="biopac_parser_evidence_invalid",
        )
    if effective_shape != expected_shape or effective_shape[1] < 2:
        raise DataFormatError(
            "BIOPAC parser evidence shape/orientation mismatch",
            code="biopac_parser_evidence_invalid",
        )

    def _strings(name: str) -> tuple[str, ...]:
        raw = document.get(name)
        if not isinstance(raw, list) or any(type(item) is not str for item in raw):
            raise DataFormatError(
                f"BIOPAC parser evidence {name} is invalid",
                code="biopac_parser_evidence_invalid",
            )
        return tuple(raw)

    labels = _strings("channel_labels")
    units = _strings("channel_units")
    source_isi_units = _strings("source_isi_units")
    channel_count = document.get("channel_count")
    label_count = document.get("label_count")
    unit_count = document.get("unit_count")
    if (
        type(channel_count) is not int
        or channel_count != effective_shape[1]
        or type(label_count) is not int
        or label_count != len(labels)
        or label_count != channel_count
        or type(unit_count) is not int
        or unit_count != len(units)
    ):
        raise DataFormatError(
            "BIOPAC parser evidence channel metadata counts are invalid",
            code="biopac_parser_evidence_invalid",
        )

    def _indices(name: str) -> tuple[int, ...]:
        raw = document.get(name)
        if not isinstance(raw, list) or any(type(item) is not int for item in raw):
            raise DataFormatError(
                f"BIOPAC parser evidence {name} is invalid",
                code="biopac_parser_evidence_invalid",
            )
        return tuple(raw)

    rsp_candidates = _indices("rsp_candidate_indices")
    ecg_candidates = _indices("ecg_candidate_indices")
    normalized_labels = tuple(item.upper() for item in labels)
    recomputed_rsp_candidates = tuple(
        index
        for index, item in enumerate(normalized_labels)
        if _BIOPAC_RSP_LABEL_RE.search(item)
    )
    recomputed_ecg_candidates = tuple(
        index
        for index, item in enumerate(normalized_labels)
        if _BIOPAC_ECG_LABEL_RE.search(item)
    )
    rsp_index = document.get("rsp_index")
    ecg_index = document.get("ecg_index")
    if (
        len(rsp_candidates) != 1
        or len(ecg_candidates) != 1
        or rsp_candidates != recomputed_rsp_candidates
        or ecg_candidates != recomputed_ecg_candidates
        or type(rsp_index) is not int
        or type(ecg_index) is not int
        or rsp_index != rsp_candidates[0]
        or ecg_index != ecg_candidates[0]
        or rsp_index == ecg_index
        or not 0 <= rsp_index < channel_count
        or not 0 <= ecg_index < channel_count
    ):
        raise DataFormatError(
            "BIOPAC parser evidence channel identity is invalid",
            code="biopac_parser_evidence_invalid",
        )

    source_count = document.get("source_isi_value_count")
    source_value_raw = document.get("source_isi_value")
    source_value = (
        None if source_value_raw is None else _exact_finite_real(source_value_raw)
    )
    if (
        type(source_count) is not int
        or source_count < 0
        or (source_value_raw is not None and source_value is None)
    ):
        raise DataFormatError(
            "BIOPAC parser evidence source interval is invalid",
            code="biopac_parser_evidence_invalid",
        )
    source_unit = document.get("source_isi_unit")
    normalized_unit = document.get("normalized_source_isi_unit")
    factor_raw = document.get("conversion_factor_to_ms")
    factor = None if factor_raw is None else _exact_finite_real(factor_raw)
    has_resolved_source_unit = (
        len(source_isi_units) == 1 and bool(source_isi_units[0])
    )
    if (
        (source_unit is not None and type(source_unit) is not str)
        or (normalized_unit is not None and type(normalized_unit) is not str)
        or (factor_raw is not None and factor is None)
        or ((source_unit is not None) != has_resolved_source_unit)
        or ((normalized_unit is not None) != has_resolved_source_unit)
        or (source_unit is not None and source_unit not in source_isi_units)
        or (
            source_unit is not None
            and normalized_unit != _normalize_biopac_isi_unit(source_unit)
        )
    ):
        raise DataFormatError(
            "BIOPAC parser evidence source unit is invalid",
            code="biopac_parser_evidence_invalid",
        )
    if normalized_unit in _BIOPAC_ISI_UNIT_FACTORS_TO_MS:
        if factor != _BIOPAC_ISI_UNIT_FACTORS_TO_MS[normalized_unit]:
            raise DataFormatError(
                "BIOPAC parser evidence conversion factor is invalid",
                code="biopac_parser_evidence_invalid",
            )
    elif factor is not None:
        raise DataFormatError(
            "BIOPAC parser evidence gives a factor to an unsupported unit",
            code="biopac_parser_evidence_invalid",
        )

    interval_source = document.get("effective_interval_source")
    effective_isi_ms = _exact_finite_real(document.get("effective_isi_ms"))
    sample_rate_hz = _exact_finite_real(document.get("effective_sample_rate_hz"))
    if (
        interval_source not in {"converted_mat_metadata", "diagnostic_fallback_4ms"}
        or effective_isi_ms is None
        or effective_isi_ms <= 0.0
        or sample_rate_hz is None
        or sample_rate_hz <= 0.0
        or not np.isclose(
            sample_rate_hz,
            1000.0 / effective_isi_ms,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise DataFormatError(
            "BIOPAC parser evidence effective interval is invalid",
            code="biopac_parser_evidence_invalid",
        )
    if interval_source == "converted_mat_metadata":
        if (
            source_count != 1
            or source_value is None
            or source_value <= 0.0
            or factor is None
            or not np.isclose(
                effective_isi_ms,
                source_value * factor,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise DataFormatError(
                "BIOPAC parser evidence did not bind its converted interval",
                code="biopac_parser_evidence_invalid",
            )
    elif effective_isi_ms != BIOPAC_ISI_MS:
        raise DataFormatError(
            "BIOPAC parser fallback interval is not the declared 4 ms",
            code="biopac_parser_evidence_invalid",
        )

    issues = document.get("issues")
    if not isinstance(issues, list):
        raise DataFormatError(
            "BIOPAC parser evidence issues are invalid",
            code="biopac_parser_evidence_invalid",
        )
    observed_issue_codes: set[str] = set()
    for issue in issues:
        if (
            not isinstance(issue, Mapping)
            or set(issue) != {"code", "severity", "message"}
            or type(issue.get("code")) is not str
            or not issue.get("code")
            or issue.get("severity") != "error"
            or type(issue.get("message")) is not str
            or not issue.get("message")
            or issue["code"] in observed_issue_codes
        ):
            raise DataFormatError(
                "BIOPAC parser evidence issue entry is invalid",
                code="biopac_parser_evidence_invalid",
            )
        observed_issue_codes.add(issue["code"])
    parser_eligible = document["parser_eligible"]
    if parser_eligible != (not issues):
        raise DataFormatError(
            "BIOPAC parser eligibility disagrees with its issues",
            code="biopac_parser_evidence_invalid",
        )
    if parser_eligible and (
        unit_count != channel_count
        or effective_isi_ms != BIOPAC_ISI_MS
        or interval_source != "converted_mat_metadata"
    ):
        raise DataFormatError(
            "BIOPAC parser evidence marks incomplete metadata eligible",
            code="biopac_parser_evidence_invalid",
        )
    return {
        key: (
            [dict(item) for item in item_value]
            if key == "issues"
            else list(item_value)
            if isinstance(item_value, tuple)
            else item_value
        )
        for key, item_value in document.items()
    }


def _biopac_fatal_error(
    path: Path,
    *,
    code: str,
    message: str,
) -> DataFormatError:
    issue = BiopacParserIssue(code=code, message=message).to_dict()
    return DataFormatError(
        f"{path}: {message}",
        code=code,
        diagnostics={
            "schema": BIOPAC_PARSER_EVIDENCE_SCHEMA,
            "diagnostic_only": True,
            "scientific_authority": False,
            "parser_eligible": False,
            "issues": [issue],
        },
    )


@dataclass(slots=True)
class BiopacRecording:
    path: Path
    data: np.ndarray
    rsp_index: int
    ecg_index: int
    labels: tuple[str, ...]
    units: tuple[str, ...]
    isi_ms: float
    sample_rate_hz: float
    start_datetime: datetime
    parser_evidence: BiopacParserEvidence
    warnings: tuple[str, ...] = ()

    @property
    def parser_eligible(self) -> bool:
        return self.parser_evidence.parser_eligible

    @property
    def rsp(self) -> np.ndarray:
        return self.data[:, self.rsp_index]

    @property
    def ecg(self) -> np.ndarray:
        return self.data[:, self.ecg_index]

    @property
    def frame_count(self) -> int:
        return int(self.data.shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate_hz

    @property
    def time_seconds(self) -> np.ndarray:
        return np.arange(self.frame_count, dtype=np.float64) / self.sample_rate_hz

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "shape": list(self.data.shape),
            "labels": list(self.labels),
            "units": list(self.units),
            "rsp_index": self.rsp_index,
            "ecg_index": self.ecg_index,
            "isi_ms": self.isi_ms,
            "isi_units": BIOPAC_ISI_CANONICAL_UNIT,
            "sample_rate_hz": self.sample_rate_hz,
            "start_datetime": self.start_datetime.isoformat(),
            "duration_seconds": self.duration_seconds,
            "parser_eligible": self.parser_eligible,
            "parser_evidence": self.parser_evidence.to_dict(),
            "warnings": list(self.warnings),
        }


def load_biopac_mat_bytes(
    payload: bytes,
    *,
    source_path: str | Path,
    timezone_name: str | tzinfo | None = "Asia/Seoul",
    strict: bool = True,
) -> BiopacRecording:
    """Parse one exact BIOPAC payload through an in-memory file object."""

    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError("scipy is required to read BIOPAC MATLAB files") from exc

    if type(payload) is not bytes:
        raise TypeError("BIOPAC payload must be exact bytes")
    mat_path = Path(source_path)
    try:
        content = loadmat(
            BytesIO(payload),
            variable_names=["data", "labels", "units", "isi", "isi_units"],
            squeeze_me=True,
            struct_as_record=False,
        )
    except Exception as exc:
        raise _biopac_fatal_error(
            mat_path,
            code="biopac_mat_decode_error",
            message=f"cannot decode BIOPAC MAT payload: {exc}",
        ) from exc
    if "data" not in content:
        raise _biopac_fatal_error(
            mat_path,
            code="biopac_data_missing",
            message="BIOPAC MAT file has no 'data' variable",
        )

    data = np.array(content["data"], copy=True)
    if data.ndim != 2 or any(int(item) <= 0 for item in data.shape):
        raise _biopac_fatal_error(
            mat_path,
            code="biopac_data_shape_invalid",
            message=f"BIOPAC data must be a non-empty 2-D matrix, got shape {data.shape}",
        )
    if data.dtype.kind not in "iuf":
        raise _biopac_fatal_error(
            mat_path,
            code="biopac_data_type_invalid",
            message=f"BIOPAC data must be real numeric, got dtype {data.dtype}",
        )
    source_data_shape = (int(data.shape[0]), int(data.shape[1]))
    labels = _matlab_strings(content.get("labels", ()))
    units = _matlab_strings(content.get("units", ()))
    if not labels:
        raise _biopac_fatal_error(
            mat_path,
            code="biopac_labels_missing",
            message="BIOPAC channel labels are missing",
        )
    labels_match_rows = data.shape[0] == len(labels)
    labels_match_columns = data.shape[1] == len(labels)
    if labels_match_rows and labels_match_columns:
        raise _biopac_fatal_error(
            mat_path,
            code="biopac_channel_orientation_ambiguous",
            message=(
                "BIOPAC channel orientation is ambiguous because labels match "
                f"both data axes ({data.shape}, {len(labels)} labels)"
            ),
        )
    if labels_match_rows:
        data = np.array(data.T, copy=True)
        orientation = "transposed_channels_by_samples"
    elif labels_match_columns:
        orientation = "samples_by_channels"
    else:
        raise _biopac_fatal_error(
            mat_path,
            code="biopac_label_count_mismatch",
            message=(
                f"BIOPAC label count {len(labels)} matches neither data axis "
                f"{data.shape}"
            ),
        )
    if data.shape[1] < 2:
        raise _biopac_fatal_error(
            mat_path,
            code="biopac_channel_count_invalid",
            message=f"BIOPAC data requires RSP and ECG channels, got {data.shape}",
        )

    normalized_labels = [item.upper() for item in labels]
    rsp_candidates = tuple(
        index
        for index, item in enumerate(normalized_labels)
        if _BIOPAC_RSP_LABEL_RE.search(item)
    )
    ecg_candidates = tuple(
        index
        for index, item in enumerate(normalized_labels)
        if _BIOPAC_ECG_LABEL_RE.search(item)
    )
    if len(rsp_candidates) != 1:
        raise _biopac_fatal_error(
            mat_path,
            code="biopac_rsp_label_cardinality_invalid",
            message=(
                "BIOPAC labels must identify exactly one RSP channel; found "
                f"indices {list(rsp_candidates)}"
            ),
        )
    if len(ecg_candidates) != 1:
        raise _biopac_fatal_error(
            mat_path,
            code="biopac_ecg_label_cardinality_invalid",
            message=(
                "BIOPAC labels must identify exactly one ECG channel; found "
                f"indices {list(ecg_candidates)}"
            ),
        )
    rsp_index = rsp_candidates[0]
    ecg_index = ecg_candidates[0]
    if rsp_index == ecg_index:
        raise _biopac_fatal_error(
            mat_path,
            code="biopac_rsp_ecg_channel_collision",
            message="BIOPAC RSP and ECG labels resolve to the same channel index",
        )

    issues: list[BiopacParserIssue] = []

    def add_issue(code: str, message: str) -> None:
        if all(item.code != code for item in issues):
            issues.append(BiopacParserIssue(code=code, message=message))

    nonfinite_count = int(np.count_nonzero(~np.isfinite(data)))
    if nonfinite_count:
        add_issue(
            "biopac_signal_nonfinite",
            f"BIOPAC channel matrix contains {nonfinite_count} non-finite sample(s)",
        )

    if len(units) != data.shape[1]:
        add_issue(
            "biopac_unit_count_mismatch",
            f"BIOPAC unit count {len(units)} does not equal channel count {data.shape[1]}",
        )

    raw_isi = content.get("isi")
    if raw_isi is None:
        isi_array = np.asarray(())
        source_isi_value_count = 0
        source_isi_value: float | None = None
        add_issue("biopac_isi_missing", "BIOPAC isi metadata is missing")
    else:
        isi_array = np.asarray(raw_isi)
        source_isi_value_count = int(isi_array.size)
        source_isi_value = None
        if source_isi_value_count != 1:
            add_issue(
                "biopac_isi_cardinality_invalid",
                "BIOPAC isi metadata must contain exactly one numeric value; "
                f"found {source_isi_value_count}",
            )
        elif isi_array.dtype.kind not in "iuf":
            add_issue(
                "biopac_isi_type_invalid",
                f"BIOPAC isi metadata must be numeric, got dtype {isi_array.dtype}",
            )
        else:
            candidate = float(isi_array.reshape(-1)[0])
            if not np.isfinite(candidate):
                add_issue(
                    "biopac_isi_nonfinite",
                    "BIOPAC isi metadata must be finite",
                )
            elif candidate <= 0.0:
                source_isi_value = candidate
                add_issue(
                    "biopac_isi_nonpositive",
                    f"BIOPAC isi metadata must be positive, got {candidate:g}",
                )
            else:
                source_isi_value = candidate

    source_isi_units = _matlab_strings(content.get("isi_units", ()))
    source_isi_unit: str | None = None
    normalized_source_isi_unit: str | None = None
    conversion_factor_to_ms: float | None = None
    if len(source_isi_units) != 1 or not source_isi_units[0]:
        add_issue(
            "biopac_isi_units_cardinality_invalid",
            "BIOPAC isi_units metadata must contain exactly one non-empty unit; "
            f"found {list(source_isi_units)}",
        )
    else:
        source_isi_unit = source_isi_units[0]
        normalized_source_isi_unit = _normalize_biopac_isi_unit(source_isi_unit)
        conversion_factor_to_ms = _BIOPAC_ISI_UNIT_FACTORS_TO_MS.get(
            normalized_source_isi_unit
        )
        if conversion_factor_to_ms is None:
            add_issue(
                "biopac_isi_units_unsupported",
                f"BIOPAC isi_units {source_isi_unit!r} is unsupported",
            )

    can_convert_interval = (
        source_isi_value is not None
        and source_isi_value > 0.0
        and conversion_factor_to_ms is not None
    )
    converted_isi_ms: float | None = None
    converted_sample_rate_hz: float | None = None
    if can_convert_interval:
        with np.errstate(
            over="ignore", under="ignore", divide="ignore", invalid="ignore"
        ):
            converted_isi_ms = float(
                np.float64(source_isi_value) * np.float64(conversion_factor_to_ms)
            )
            converted_sample_rate_hz = float(
                np.float64(1000.0) / np.float64(converted_isi_ms)
            )
        if (
            not np.isfinite(converted_isi_ms)
            or converted_isi_ms <= 0.0
            or not np.isfinite(converted_sample_rate_hz)
            or converted_sample_rate_hz <= 0.0
        ):
            add_issue(
                "biopac_effective_interval_invalid",
                "BIOPAC isi conversion did not produce a finite positive interval/rate",
            )
            can_convert_interval = False
    if can_convert_interval:
        assert converted_isi_ms is not None
        assert converted_sample_rate_hz is not None
        isi_ms = converted_isi_ms
        sample_rate_hz = converted_sample_rate_hz
        effective_interval_source = "converted_mat_metadata"
    else:
        isi_ms = BIOPAC_ISI_MS
        sample_rate_hz = BIOPAC_SAMPLE_RATE_HZ
        effective_interval_source = "diagnostic_fallback_4ms"
    if (
        effective_interval_source == "converted_mat_metadata"
        and not np.isclose(isi_ms, BIOPAC_ISI_MS, rtol=0.0, atol=1e-6)
    ):
        add_issue(
            "biopac_effective_isi_unexpected",
            f"BIOPAC effective isi is {isi_ms:g} ms ({sample_rate_hz:g} Hz), "
            "expected 4 ms (250 Hz)",
        )

    evidence = BiopacParserEvidence(
        source_data_shape=source_data_shape,
        effective_data_shape=(int(data.shape[0]), int(data.shape[1])),
        orientation=orientation,
        channel_labels=labels,
        channel_units=units,
        rsp_candidate_indices=rsp_candidates,
        ecg_candidate_indices=ecg_candidates,
        rsp_index=rsp_index,
        ecg_index=ecg_index,
        source_isi_value_count=source_isi_value_count,
        source_isi_value=source_isi_value,
        source_isi_units=source_isi_units,
        source_isi_unit=source_isi_unit,
        normalized_source_isi_unit=normalized_source_isi_unit,
        conversion_factor_to_ms=conversion_factor_to_ms,
        effective_interval_source=effective_interval_source,
        effective_isi_ms=isi_ms,
        effective_sample_rate_hz=sample_rate_hz,
        issues=tuple(issues),
    )
    evidence_document = evidence.to_dict()
    warnings = tuple(item.message for item in issues)
    if strict and issues:
        raise DataFormatError(
            f"{mat_path}: " + "; ".join(warnings),
            code="biopac_parser_ineligible",
            diagnostics=evidence_document,
        )

    start = parse_biopac_start_datetime(mat_path, timezone_name=timezone_name)
    return BiopacRecording(
        path=mat_path,
        data=data,
        rsp_index=rsp_index,
        ecg_index=ecg_index,
        labels=labels,
        units=units,
        isi_ms=isi_ms,
        sample_rate_hz=sample_rate_hz,
        start_datetime=start,
        parser_evidence=evidence,
        warnings=warnings,
    )


def load_biopac_mat(
    path: str | Path,
    *,
    timezone_name: str | tzinfo | None = "Asia/Seoul",
    strict: bool = True,
) -> BiopacRecording:
    """Load a legacy BIOPAC pathname from one materialized byte snapshot."""

    mat_path = Path(path)
    try:
        payload = mat_path.read_bytes()
    except OSError as exc:
        raise DataFormatError(f"cannot read BIOPAC MAT file {mat_path}: {exc}") from exc
    return load_biopac_mat_bytes(
        payload,
        source_path=mat_path,
        timezone_name=timezone_name,
        strict=strict,
    )


@dataclass(frozen=True, slots=True)
class RadarStreamInfo:
    radar_id: int
    recording_dir: Path
    meta_path: Path | None
    data_paths: tuple[Path, ...]
    start_epoch_ms: int | None
    frame_count: int
    duration_seconds: float
    timestamp_source: str
    warnings: tuple[str, ...] = ()

    @property
    def start_datetime_utc(self) -> datetime | None:
        if self.start_epoch_ms is None:
            return None
        return datetime.fromtimestamp(self.start_epoch_ms / 1000.0, tz=timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "radar_id": self.radar_id,
            "recording_dir": str(self.recording_dir),
            "meta_path": str(self.meta_path) if self.meta_path else None,
            "data_paths": [str(path) for path in self.data_paths],
            "start_epoch_ms": self.start_epoch_ms,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
            "timestamp_source": self.timestamp_source,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class LogicalSession:
    session_id: str
    start_epoch_ms: int | None
    radars: Mapping[int, RadarStreamInfo]

    @property
    def complete(self) -> bool:
        return all(index in self.radars for index in (1, 2, 3))

    @property
    def common_frame_count(self) -> int:
        return min((item.frame_count for item in self.radars.values()), default=0)

    @property
    def common_duration_seconds(self) -> float:
        return min((item.duration_seconds for item in self.radars.values()), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "start_epoch_ms": self.start_epoch_ms,
            "complete": self.complete,
            "common_frame_count": self.common_frame_count,
            "common_duration_seconds": self.common_duration_seconds,
            "radars": {str(key): value.to_dict() for key, value in self.radars.items()},
        }


@dataclass(frozen=True, slots=True)
class SubjectManifest:
    subject_id: str
    subject_number: int
    subject_code: str | None
    path: Path
    biopac_path: Path | None
    sessions: tuple[LogicalSession, ...]
    selected_session: LogicalSession | None
    missing_radars: tuple[int, ...]
    warnings: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.biopac_path is not None and self.selected_session is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject_number": self.subject_number,
            "subject_code": self.subject_code,
            "path": str(self.path),
            "biopac_path": str(self.biopac_path) if self.biopac_path else None,
            "usable": self.usable,
            "missing_radars": list(self.missing_radars),
            "warnings": list(self.warnings),
            "selected_session_id": (
                self.selected_session.session_id if self.selected_session else None
            ),
            "sessions": [item.to_dict() for item in self.sessions],
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    root: Path
    subjects: tuple[SubjectManifest, ...]

    @property
    def usable_subjects(self) -> tuple[SubjectManifest, ...]:
        return tuple(item for item in self.subjects if item.usable)

    def by_subject(self, subject: int | str) -> SubjectManifest:
        if isinstance(subject, int):
            for item in self.subjects:
                if item.subject_number == subject:
                    return item
        else:
            for item in self.subjects:
                if item.subject_id == subject:
                    return item
        raise KeyError(subject)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "subject_count": len(self.subjects),
            "usable_subject_count": len(self.usable_subjects),
            "subjects": [item.to_dict() for item in self.subjects],
        }


def _datetime_from_radar_directory(path: Path, timezone_value: tzinfo) -> datetime | None:
    match = _RADAR_DIR_DATETIME_RE.search(path.name)
    if not match:
        return None
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone_value
    )


def _discover_stream(recording_dir: Path, radar_id: int, timezone_value: tzinfo) -> RadarStreamInfo:
    data_paths = sorted(recording_dir.glob("xethru_datafloat_*.dat"), key=_path_sort_key)
    total_frames = sum(path.stat().st_size // XETHRU_RECORD_BYTES for path in data_paths)
    meta_path = recording_dir / "xethru_recording_meta.dat"
    warnings: list[str] = []
    if meta_path.is_file():
        meta = parse_xethru_meta(meta_path, expected_frames=total_frames)
        warnings.extend(meta.warnings)
        if meta.chunk_filenames:
            by_name = {path.name: path for path in data_paths}
            ordered: list[Path] = []
            for filename in meta.chunk_filenames:
                path = by_name.pop(filename, None)
                if path is None:
                    warnings.append(f"metadata chunk missing on disk: {filename}")
                else:
                    ordered.append(path)
            ordered.extend(sorted(by_name.values(), key=_path_sort_key))
            data_paths = ordered
        start_ms = meta.start_epoch_ms
        duration = meta.duration_seconds
        source = meta.timestamp_source
    else:
        warnings.append("metadata file missing")
        start = _datetime_from_radar_directory(recording_dir, timezone_value)
        start_ms = int(start.timestamp() * 1000) if start else None
        duration = total_frames / RADAR_SAMPLE_RATE_HZ
        source = "fallback_40hz"
    if not data_paths:
        warnings.append("radar data chunks missing")
    return RadarStreamInfo(
        radar_id=radar_id,
        recording_dir=recording_dir,
        meta_path=meta_path if meta_path.is_file() else None,
        data_paths=tuple(data_paths),
        start_epoch_ms=start_ms,
        frame_count=total_frames,
        duration_seconds=duration,
        timestamp_source=source,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _cluster_streams(
    streams: Sequence[RadarStreamInfo], tolerance_ms: int
) -> tuple[LogicalSession, ...]:
    clusters: list[dict[int, RadarStreamInfo]] = []
    centers: list[int | None] = []
    for stream in sorted(
        streams,
        key=lambda item: (
            item.start_epoch_ms if item.start_epoch_ms is not None else 2**63 - 1,
            item.radar_id,
            str(item.recording_dir),
        ),
    ):
        best_index: int | None = None
        best_distance = tolerance_ms + 1
        if stream.start_epoch_ms is not None:
            for index, center in enumerate(centers):
                if center is None:
                    continue
                distance = abs(stream.start_epoch_ms - center)
                if distance <= tolerance_ms and distance < best_distance:
                    best_index, best_distance = index, distance
        if best_index is None:
            clusters.append({stream.radar_id: stream})
            centers.append(stream.start_epoch_ms)
            continue
        existing = clusters[best_index].get(stream.radar_id)
        if existing is None or stream.frame_count > existing.frame_count:
            clusters[best_index][stream.radar_id] = stream
        present_starts = [
            item.start_epoch_ms
            for item in clusters[best_index].values()
            if item.start_epoch_ms is not None
        ]
        centers[best_index] = int(np.median(present_starts)) if present_starts else None

    sessions: list[LogicalSession] = []
    for index, (radars, center) in enumerate(zip(clusters, centers, strict=True), start=1):
        if center is not None:
            stamp = datetime.fromtimestamp(center / 1000.0, tz=timezone.utc)
            session_id = stamp.strftime("%Y%m%dT%H%M%S.%fZ")[:-4] + "Z"
        else:
            session_id = f"unknown-{index:02d}"
        sessions.append(LogicalSession(session_id, center, dict(sorted(radars.items()))))
    return tuple(sessions)


def build_dataset_manifest(
    root: str | Path,
    *,
    timezone_name: str | tzinfo = "Asia/Seoul",
    expected_subject_numbers: Iterable[int] | None = range(1, 31),
    session_tolerance_seconds: float = 5.0,
) -> DatasetManifest:
    """Discover subjects and select the longest complete three-radar session.

    The selection rule is deterministic: maximize the minimum duration shared
    by radars 1/2/3, then the common frame count, then choose the earliest
    start.  Consequently S01's long split-file recording wins over its later
    short retry.  S24 is retained in the manifest with all radars missing.
    """

    dataset_root = Path(root).resolve()
    timezone_value = _coerce_timezone(timezone_name)
    discovered: dict[int, Path] = {}
    for path in sorted(dataset_root.glob("S[0-9][0-9]_*")):
        match = re.match(r"S(\d{2})(?:_(.*))?$", path.name)
        if match and path.is_dir():
            discovered[int(match.group(1))] = path
    numbers = (
        sorted(set(expected_subject_numbers))
        if expected_subject_numbers is not None
        else sorted(discovered)
    )
    subjects: list[SubjectManifest] = []
    for number in numbers:
        path = discovered.get(number, dataset_root / f"S{number:02d}_MISSING")
        subject_id = path.name
        code_match = re.match(r"S\d{2}_(.+)$", subject_id)
        code = code_match.group(1) if code_match else None
        warnings: list[str] = []
        streams: list[RadarStreamInfo] = []
        for radar_id in (1, 2, 3):
            radar_root = path / str(radar_id)
            if radar_root.is_dir():
                for recording_dir in sorted(
                    item
                    for item in radar_root.iterdir()
                    if item.is_dir() and item.name.startswith("xethru_recording_")
                ):
                    streams.append(_discover_stream(recording_dir, radar_id, timezone_value))
        sessions = _cluster_streams(
            streams, int(round(session_tolerance_seconds * 1000.0))
        )
        complete = [session for session in sessions if session.complete]
        selected = (
            max(
                complete,
                key=lambda item: (
                    item.common_duration_seconds,
                    item.common_frame_count,
                    -(item.start_epoch_ms or 2**63 - 1),
                ),
            )
            if complete
            else None
        )
        if selected is None:
            present = {stream.radar_id for stream in streams if stream.data_paths}
            missing_radars = tuple(index for index in (1, 2, 3) if index not in present)
            warnings.append(
                "no complete three-radar logical session"
                + (f"; missing radar(s) {missing_radars}" if missing_radars else "")
            )
        else:
            missing_radars = tuple(index for index in (1, 2, 3) if index not in selected.radars)

        biopac_files = sorted((path / "BIOPAC").glob("*.mat")) if path.is_dir() else []
        biopac_path = biopac_files[0] if biopac_files else None
        if not biopac_files:
            warnings.append("BIOPAC MAT file missing")
        elif len(biopac_files) > 1:
            warnings.append(
                f"multiple BIOPAC MAT files; selected lexicographically first ({biopac_path.name})"
            )
        subjects.append(
            SubjectManifest(
                subject_id=subject_id,
                subject_number=number,
                subject_code=code,
                path=path,
                biopac_path=biopac_path,
                sessions=sessions,
                selected_session=selected,
                missing_radars=missing_radars,
                warnings=tuple(warnings),
            )
        )
    return DatasetManifest(dataset_root, tuple(subjects))


# Short, conventional alias for training/data-pipeline callers.
build_manifest = build_dataset_manifest


def radar_qc(
    stream: RadarStreamInfo,
    *,
    amplitude_limit: float = 0.1,
    block_frames: int = 4096,
) -> dict[str, Any]:
    """Audit shape, headers, counters, finite values and radar outliers."""

    report: dict[str, Any] = {
        "radar_id": stream.radar_id,
        "paths": [str(path) for path in stream.data_paths],
        "frame_count": 0,
        "record_size_remainder_bytes": 0,
        "zero_header_nonzero": 0,
        "bin_count_invalid": 0,
        "counter_gap_count": 0,
        "nan_count": 0,
        "inf_count": 0,
        "max_abs": None,
        "amplitude_limit": amplitude_limit,
        "amplitude_outlier_count": 0,
        "first_amplitude_outlier": None,
        "metadata_frame_count": None,
        "metadata_timestamp_source": stream.timestamp_source,
        "metadata_timestamp_repairs": 0,
        "warnings": list(stream.warnings),
    }
    previous_sequence: int | None = None
    global_offset = 0
    max_abs = -np.inf
    max_location: tuple[int, int] | None = None
    for path in stream.data_paths:
        size = path.stat().st_size
        remainder = size % XETHRU_RECORD_BYTES
        report["record_size_remainder_bytes"] += remainder
        count = size // XETHRU_RECORD_BYTES
        if count == 0:
            report["warnings"].append(f"empty radar chunk: {path.name}")
            continue
        chunk = np.memmap(path, dtype=XETHRU_RECORD_DTYPE, mode="r", shape=(count,))
        report["frame_count"] += count
        report["zero_header_nonzero"] += int(np.count_nonzero(chunk["zero"]))
        report["bin_count_invalid"] += int(np.count_nonzero(chunk["bin_count"] != RADAR_BINS))
        sequence = chunk["frame_sequence"]
        if previous_sequence is not None:
            cross_delta = (int(sequence[0]) - previous_sequence) & 0xFFFFFFFF
            report["counter_gap_count"] += int(cross_delta != 1)
        if count > 1:
            delta = sequence[1:] - sequence[:-1]
            report["counter_gap_count"] += int(np.count_nonzero(delta != np.uint32(1)))
        previous_sequence = int(sequence[-1])

        values = chunk["bins"]
        for local_start in range(0, count, block_frames):
            block = np.asarray(values[local_start : local_start + block_frames])
            report["nan_count"] += int(np.count_nonzero(np.isnan(block)))
            report["inf_count"] += int(np.count_nonzero(np.isinf(block)))
            finite_abs = np.where(np.isfinite(block), np.abs(block), -np.inf)
            block_max_index = int(np.argmax(finite_abs))
            block_max = float(finite_abs.reshape(-1)[block_max_index])
            if block_max > max_abs:
                max_abs = block_max
                row, column = np.unravel_index(block_max_index, block.shape)
                max_location = (global_offset + local_start + int(row), int(column))
            outlier_mask = finite_abs > amplitude_limit
            outlier_count = int(np.count_nonzero(outlier_mask))
            if outlier_count and report["first_amplitude_outlier"] is None:
                row, column = np.argwhere(outlier_mask)[0]
                report["first_amplitude_outlier"] = {
                    "global_frame_index": global_offset + local_start + int(row),
                    "frame_sequence": int(sequence[local_start + int(row)]),
                    "bin_index": int(column),
                    "value": float(block[row, column]),
                    "file": str(path),
                }
            report["amplitude_outlier_count"] += outlier_count
        global_offset += count

    report["max_abs"] = float(max_abs) if np.isfinite(max_abs) else None
    if max_location is not None:
        report["max_abs_location"] = {
            "global_frame_index": max_location[0],
            "bin_index": max_location[1],
        }
    if stream.meta_path and stream.meta_path.is_file():
        meta = parse_xethru_meta(stream.meta_path, expected_frames=report["frame_count"])
        report["metadata_frame_count"] = meta.frame_record_count
        report["metadata_timestamp_source"] = meta.timestamp_source
        report["metadata_timestamp_repairs"] = meta.timestamp_repairs
        report["warnings"].extend(meta.warnings)
    report["warnings"] = list(dict.fromkeys(report["warnings"]))
    failures = any(
        report[key]
        for key in (
            "record_size_remainder_bytes",
            "zero_header_nonzero",
            "bin_count_invalid",
            "counter_gap_count",
            "nan_count",
            "inf_count",
            "amplitude_outlier_count",
        )
    )
    report["status"] = "error" if failures else ("warning" if report["warnings"] else "ok")
    return report


def biopac_qc(path: str | Path) -> dict[str, Any]:
    """Audit BIOPAC shape, sample interval, finite values, and ADC rails."""

    report: dict[str, Any] = {"path": str(path), "status": "error", "warnings": []}
    try:
        recording = load_biopac_mat(path, strict=False)
    except DataFormatError as exc:
        report.update(
            {
                "error": str(exc),
                "error_code": exc.code,
                "parser_eligible": False,
                "parser_evidence": dict(exc.diagnostics),
            }
        )
        return report
    except Exception as exc:
        report["error"] = str(exc)
        return report
    rsp = np.asarray(recording.rsp)
    ecg = np.asarray(recording.ecg)
    rsp_nonfinite = int(np.count_nonzero(~np.isfinite(rsp)))
    ecg_nonfinite = int(np.count_nonzero(~np.isfinite(ecg)))
    # HAI's exported units/observed acquisition rails are +/-10 V for RSP and
    # +/-5 mV for ECG.  Values within 0.001 of a rail are counted as clipping.
    rsp_clipped = int(np.count_nonzero(np.abs(rsp) >= 9.999))
    ecg_clipped = int(np.count_nonzero(np.abs(ecg) >= 4.999))
    report.update(
        {
            "shape": list(recording.data.shape),
            "labels": list(recording.labels),
            "units": list(recording.units),
            "isi_ms": recording.isi_ms,
            "isi_units": BIOPAC_ISI_CANONICAL_UNIT,
            "sample_rate_hz": recording.sample_rate_hz,
            "parser_eligible": recording.parser_eligible,
            "parser_evidence": recording.parser_evidence.to_dict(),
            "start_datetime": recording.start_datetime.isoformat(),
            "duration_seconds": recording.duration_seconds,
            "rsp_nan_or_inf_count": rsp_nonfinite,
            "ecg_nan_or_inf_count": ecg_nonfinite,
            "rsp_min": float(np.nanmin(rsp)),
            "rsp_max": float(np.nanmax(rsp)),
            "ecg_min": float(np.nanmin(ecg)),
            "ecg_max": float(np.nanmax(ecg)),
            "rsp_clipped_count": rsp_clipped,
            "rsp_clipped_fraction": rsp_clipped / max(1, rsp.size),
            "ecg_clipped_count": ecg_clipped,
            "ecg_clipped_fraction": ecg_clipped / max(1, ecg.size),
            "warnings": list(recording.warnings),
        }
    )
    if rsp_clipped:
        report["warnings"].append(f"RSP reaches +/-10 V rail ({rsp_clipped} samples)")
    if ecg_clipped:
        report["warnings"].append(f"ECG reaches +/-5 mV rail ({ecg_clipped} samples)")
    if not recording.parser_eligible or rsp_nonfinite or ecg_nonfinite:
        report["status"] = "error"
    elif report["warnings"]:
        report["status"] = "warning"
    else:
        report["status"] = "ok"
    return report


def audit_manifest(
    manifest: DatasetManifest, *, radar_amplitude_limit: float = 0.1
) -> dict[str, Any]:
    """Run complete dataset QC and synchronization checks."""

    subject_reports: list[dict[str, Any]] = []
    status_rank = {"ok": 0, "warning": 1, "missing": 2, "error": 3}
    for subject in manifest.subjects:
        report: dict[str, Any] = {
            "subject_id": subject.subject_id,
            "subject_number": subject.subject_number,
            "usable": subject.usable,
            "missing_radars": list(subject.missing_radars),
            "warnings": list(subject.warnings),
            "radars": {},
            "biopac": None,
            "sync": None,
        }
        statuses: list[str] = []
        if subject.selected_session is None:
            statuses.append("missing")
        else:
            for radar_id, stream in subject.selected_session.radars.items():
                radar_report = radar_qc(stream, amplitude_limit=radar_amplitude_limit)
                report["radars"][str(radar_id)] = radar_report
                statuses.append(radar_report["status"])
        if subject.biopac_path is None:
            statuses.append("missing")
        else:
            biopac_report = biopac_qc(subject.biopac_path)
            report["biopac"] = biopac_report
            statuses.append(biopac_report["status"])

        if subject.selected_session is not None:
            starts = [
                stream.start_epoch_ms
                for stream in subject.selected_session.radars.values()
                if stream.start_epoch_ms is not None
            ]
            counts = [
                stream.frame_count for stream in subject.selected_session.radars.values()
            ]
            sync: dict[str, Any] = {
                "radar_start_spread_ms": max(starts) - min(starts) if starts else None,
                "radar_frame_count_spread": max(counts) - min(counts) if counts else None,
                "radar_to_biopac_start_seconds": None,
                "radar_end_minus_biopac_end_seconds": None,
                "status": "ok",
                "warnings": [],
            }
            if starts and report["biopac"] and "start_datetime" in report["biopac"]:
                biopac_start = datetime.fromisoformat(report["biopac"]["start_datetime"])
                biopac_start_ms = int(round(biopac_start.timestamp() * 1000.0))
                radar_start_ms = min(starts)
                offset = (radar_start_ms - biopac_start_ms) / 1000.0
                radar_end_ms = max(
                    (stream.start_epoch_ms or radar_start_ms)
                    + int(round(stream.duration_seconds * 1000.0))
                    for stream in subject.selected_session.radars.values()
                )
                biopac_end_ms = biopac_start_ms + int(
                    round(float(report["biopac"]["duration_seconds"]) * 1000.0)
                )
                sync["radar_to_biopac_start_seconds"] = offset
                sync["radar_end_minus_biopac_end_seconds"] = (
                    radar_end_ms - biopac_end_ms
                ) / 1000.0
                if abs(offset) > 120.0:
                    sync["warnings"].append("radar/BIOPAC starts differ by over 120 s")
                if radar_end_ms > biopac_end_ms + 5000:
                    sync["warnings"].append("radar extends more than 5 s past BIOPAC")
            else:
                sync["warnings"].append("insufficient timestamps for synchronization QC")
            if sync["radar_start_spread_ms"] is not None and sync["radar_start_spread_ms"] > 1000:
                sync["warnings"].append("three radar starts differ by over 1 s")
            if sync["radar_frame_count_spread"] is not None and sync["radar_frame_count_spread"] > 2:
                sync["warnings"].append("three radar frame counts differ by over 2")
            if sync["warnings"]:
                sync["status"] = "warning"
                statuses.append("warning")
            report["sync"] = sync

        report["status"] = max(statuses or ["missing"], key=status_rank.__getitem__)
        # S24 remains explicitly represented instead of silently disappearing;
        # S22's isolated radar-2 spike is found by the generic amplitude audit.
        subject_reports.append(report)

    counts = {key: 0 for key in status_rank}
    for report in subject_reports:
        counts[report["status"]] += 1
    return {
        "dataset_root": str(manifest.root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "radar_bins": RADAR_BINS,
            "radar_record_bytes": XETHRU_RECORD_BYTES,
            "radar_fallback_rate_hz": RADAR_SAMPLE_RATE_HZ,
            "biopac_rate_hz": BIOPAC_SAMPLE_RATE_HZ,
            "radar_amplitude_limit": radar_amplitude_limit,
        },
        "summary": {
            "subject_count": len(subject_reports),
            "usable_subject_count": len(manifest.usable_subjects),
            "status_counts": counts,
        },
        "subjects": subject_reports,
    }


__all__ = [
    "BIOPAC_ISI_CANONICAL_UNIT",
    "BIOPAC_ISI_MS",
    "BIOPAC_PARSER_EVIDENCE_SCHEMA",
    "BIOPAC_SAMPLE_RATE_HZ",
    "BiopacParserEvidence",
    "BiopacParserIssue",
    "BiopacRecording",
    "DataFormatError",
    "DatasetManifest",
    "LoadedRadarRecording",
    "LogicalSession",
    "RADAR_BINS",
    "RADAR_SAMPLE_RATE_HZ",
    "RadarStreamInfo",
    "SplitRadarMemmap",
    "SubjectManifest",
    "XETHRU_META_VERSION",
    "XETHRU_META_CLOSE_TIMESTAMP_POLICY",
    "XETHRU_META_EVIDENCE_SCHEMA",
    "XETHRU_RECORD_BYTES",
    "XETHRU_RECORD_DTYPE",
    "XeThruMeta",
    "XeThruMetaChunkEvidence",
    "XeThruMetaEvidence",
    "XeThruMetaIssue",
    "audit_manifest",
    "biopac_qc",
    "build_dataset_manifest",
    "build_manifest",
    "load_biopac_mat",
    "load_biopac_mat_bytes",
    "load_xethru_recording",
    "load_xethru_records",
    "open_xethru_files",
    "parse_biopac_start_datetime",
    "parse_xethru_meta",
    "parse_xethru_meta_bytes",
    "radar_qc",
    "validate_biopac_parser_evidence",
    "validate_xethru_meta_evidence",
]
