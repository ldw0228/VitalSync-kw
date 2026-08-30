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
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np


RADAR_BINS = 182
RADAR_SAMPLE_RATE_HZ = 40.0
BIOPAC_SAMPLE_RATE_HZ = 250.0
BIOPAC_ISI_MS = 4.0
XETHRU_MAGIC = 0xA0B1C2D3
XETHRU_META_VERSION = 13
XETHRU_META_RECORD_FORMAT = 1024

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
_RADAR_DIR_DATETIME_RE = re.compile(r"xethru_recording_(\d{8})_(\d{6})")
_DATA_CHUNK_DATETIME_RE = re.compile(r"xethru_datafloat_(\d{8})_(\d{6})\.dat$")


class DataFormatError(ValueError):
    """Raised when a raw file cannot be interpreted safely."""


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
            "warnings": list(self.warnings),
        }


def parse_xethru_meta(
    path: str | Path,
    *,
    expected_frames: int | None = None,
    fallback_rate_hz: float = RADAR_SAMPLE_RATE_HZ,
    strict: bool = False,
) -> XeThruMeta:
    """Parse HAI XeThru ``xethru_recording_meta.dat`` version 13.

    The fixed 33-byte frame entries contain chunk id/size, file and logical
    byte offsets, and a relative millisecond timestamp.  Type-2 entries close
    chunks.  The final variable-size type-3 footer lists one or more data
    filenames.  Timestamp entries remain usable when that footer is missing or
    truncated.  If entries themselves are unusable or disagree with
    ``expected_frames``, a deterministic 40 Hz timeline is returned.
    """

    meta_path = Path(path)
    warnings: list[str] = []
    magic: int | None = None
    version: int | None = None
    start_epoch_ms: int | None = None
    device_name: str | None = None
    raw_timestamps: list[int] = []
    chunk_filenames: list[str] = []
    declared_chunks: int | None = None
    end_markers = 0
    timestamp_repairs = 0

    try:
        payload = meta_path.read_bytes()
    except OSError as exc:
        if strict:
            raise DataFormatError(f"cannot read metadata {meta_path}: {exc}") from exc
        warnings.append(f"metadata unreadable: {exc}")
        count = max(0, int(expected_frames or 0))
        return XeThruMeta(
            path=meta_path,
            magic=None,
            version=None,
            start_epoch_ms=None,
            device_name=None,
            relative_timestamps_ms=_fallback_timestamps(count, fallback_rate_hz),
            warnings=tuple(warnings),
        )

    record_offset: int | None = None
    if len(payload) >= 16:
        magic, version = struct.unpack_from("<II", payload, 0)
        start_epoch_ms = struct.unpack_from("<Q", payload, 8)[0]
        if not 946_684_800_000 <= start_epoch_ms <= 4_102_444_800_000:
            warnings.append(f"implausible start epoch milliseconds: {start_epoch_ms}")
            start_epoch_ms = None
        if magic != XETHRU_MAGIC:
            warnings.append(f"unexpected metadata magic 0x{magic:08x}")
        if version != XETHRU_META_VERSION:
            warnings.append(f"metadata version {version}, expected {XETHRU_META_VERSION}")
    else:
        warnings.append(f"metadata header truncated ({len(payload)} bytes)")

    if len(payload) >= _META_HEADER_NAME_LENGTH_OFFSET + 4:
        name_length = struct.unpack_from(
            "<I", payload, _META_HEADER_NAME_LENGTH_OFFSET
        )[0]
        name_start = _META_HEADER_NAME_LENGTH_OFFSET + 4
        name_end = name_start + name_length
        if 0 < name_length <= 4096 and name_end <= len(payload):
            device_name = (
                payload[name_start:name_end]
                .rstrip(b"\x00")
                .decode("utf-8", errors="replace")
            )
            # Recorder v13 writes four zero bytes after the NUL-inclusive name.
            candidate = name_end + 4
            if candidate + _META_FIXED_RECORD.size <= len(payload):
                record_offset = candidate
        else:
            warnings.append(f"invalid device-name length: {name_length}")

    # A conservative recovery path for a damaged name-length field.  Requiring
    # the complete kind/format marker avoids matching arbitrary float payload.
    marker = b"\x01\x00\x04\x00\x00"
    if record_offset is None or payload[record_offset : record_offset + 5] != marker:
        recovered = payload.find(marker, 80, min(len(payload), 4096))
        if recovered >= 0:
            record_offset = recovered
            warnings.append("recovered metadata record table by signature")
        else:
            record_offset = None
            warnings.append("metadata frame table not found")

    footer_offset: int | None = None
    if record_offset is not None:
        cursor = record_offset
        while cursor + _META_FIXED_RECORD.size <= len(payload):
            kind, record_format, encoded_size, _file_offset, _logical_end, timestamp = (
                _META_FIXED_RECORD.unpack_from(payload, cursor)
            )
            if kind not in (1, 2) or record_format != XETHRU_META_RECORD_FORMAT:
                footer_offset = cursor
                break
            chunk_index = encoded_size >> 32
            data_size = encoded_size & 0xFFFFFFFF
            if kind == 1:
                if data_size != XETHRU_RECORD_BYTES:
                    warnings.append(
                        f"frame entry declares {data_size} bytes in chunk {chunk_index}, "
                        f"expected {XETHRU_RECORD_BYTES}"
                    )
                raw_timestamps.append(timestamp)
            else:
                end_markers += 1
            cursor += _META_FIXED_RECORD.size
        else:
            footer_offset = cursor

    # Footer: byte kind=3, uint32 filename count, then count repetitions of
    # uint32 NUL-inclusive length + UTF-8 filename.  Parse each independently so
    # a partially written last filename does not invalidate earlier names.
    if footer_offset is not None:
        cursor = footer_offset
        if cursor < len(payload) and payload[cursor] != 3:
            nearby = payload.find(b"\x03", cursor, min(len(payload), cursor + 32))
            cursor = nearby if nearby >= 0 else cursor
        if cursor + 5 <= len(payload) and payload[cursor] == 3:
            declared_chunks = struct.unpack_from("<I", payload, cursor + 1)[0]
            cursor += 5
            if declared_chunks > 100_000:
                warnings.append(f"implausible footer chunk count: {declared_chunks}")
                declared_chunks = None
            else:
                for index in range(declared_chunks):
                    if cursor + 4 > len(payload):
                        warnings.append(f"footer truncated before chunk name {index}")
                        break
                    length = struct.unpack_from("<I", payload, cursor)[0]
                    cursor += 4
                    if length <= 0 or length > 4096 or cursor + length > len(payload):
                        warnings.append(f"invalid/truncated footer chunk name {index}")
                        break
                    filename = (
                        payload[cursor : cursor + length]
                        .rstrip(b"\x00")
                        .decode("utf-8", errors="replace")
                    )
                    chunk_filenames.append(filename)
                    cursor += length
        elif raw_timestamps:
            warnings.append("metadata filename footer missing")

    use_fallback = not raw_timestamps
    if expected_frames is not None and len(raw_timestamps) != expected_frames:
        warnings.append(
            f"metadata/data frame mismatch: {len(raw_timestamps)} != {expected_frames}"
        )
        use_fallback = True

    if use_fallback:
        count = max(0, int(expected_frames if expected_frames is not None else len(raw_timestamps)))
        timestamps = _fallback_timestamps(count, fallback_rate_hz)
        timestamp_source = f"fallback_{fallback_rate_hz:g}hz"
        if not raw_timestamps:
            warnings.append(f"using {fallback_rate_hz:g} Hz timestamp fallback")
    else:
        timestamps, timestamp_repairs = _unwrap_relative_timestamps(
            raw_timestamps, 1000.0 / fallback_rate_hz
        )
        timestamp_source = "meta_v13"
        if timestamp_repairs:
            warnings.append(
                f"unwrapped {timestamp_repairs} relative-timestamp counter reset(s)"
            )

    if strict and warnings:
        raise DataFormatError(f"{meta_path}: " + "; ".join(warnings))

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
        warnings=tuple(dict.fromkeys(warnings)),
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
    warnings: tuple[str, ...] = ()

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
            "sample_rate_hz": self.sample_rate_hz,
            "start_datetime": self.start_datetime.isoformat(),
            "duration_seconds": self.duration_seconds,
            "warnings": list(self.warnings),
        }


def load_biopac_mat(
    path: str | Path,
    *,
    timezone_name: str | tzinfo | None = "Asia/Seoul",
    strict: bool = True,
) -> BiopacRecording:
    """Load the HAI BIOPAC RSP/ECG MATLAB file (250 Hz, ``isi=4 ms``)."""

    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError("scipy is required to read BIOPAC MATLAB files") from exc

    mat_path = Path(path)
    try:
        content = loadmat(
            mat_path,
            variable_names=["data", "labels", "units", "isi", "isi_units"],
            squeeze_me=True,
            struct_as_record=False,
        )
    except Exception as exc:
        raise DataFormatError(f"cannot read BIOPAC MAT file {mat_path}: {exc}") from exc
    if "data" not in content:
        raise DataFormatError(f"BIOPAC MAT file has no 'data' variable: {mat_path}")

    data = np.asarray(content["data"])
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise DataFormatError(f"BIOPAC data must be 2-D, got shape {data.shape}")
    labels = _matlab_strings(content.get("labels", ()))
    units = _matlab_strings(content.get("units", ()))
    if labels and data.shape[0] == len(labels) and data.shape[1] != len(labels):
        data = data.T
    if data.shape[1] < 2:
        raise DataFormatError(f"BIOPAC data requires RSP and ECG channels, got {data.shape}")

    warnings: list[str] = []
    isi_values = np.asarray(content.get("isi", BIOPAC_ISI_MS), dtype=np.float64).reshape(-1)
    isi_ms = float(isi_values[0]) if isi_values.size else BIOPAC_ISI_MS
    if not np.isfinite(isi_ms) or isi_ms <= 0:
        warnings.append(f"invalid isi={isi_ms}; using {BIOPAC_ISI_MS:g} ms")
        isi_ms = BIOPAC_ISI_MS
    sample_rate_hz = 1000.0 / isi_ms
    if not np.isclose(isi_ms, BIOPAC_ISI_MS, rtol=0.0, atol=1e-6):
        warnings.append(
            f"BIOPAC isi is {isi_ms:g} ms ({sample_rate_hz:g} Hz), expected 4 ms (250 Hz)"
        )

    normalized_labels = [item.upper() for item in labels]
    rsp_candidates = [i for i, item in enumerate(normalized_labels) if "RSP" in item]
    ecg_candidates = [i for i, item in enumerate(normalized_labels) if "ECG" in item]
    rsp_index = rsp_candidates[0] if rsp_candidates else 0
    ecg_index = ecg_candidates[0] if ecg_candidates else 1
    if not rsp_candidates:
        warnings.append("RSP label missing; using channel 0")
    if not ecg_candidates:
        warnings.append("ECG label missing; using channel 1")

    start = parse_biopac_start_datetime(mat_path, timezone_name=timezone_name)
    if strict and warnings:
        raise DataFormatError(f"{mat_path}: " + "; ".join(warnings))
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
        warnings=tuple(warnings),
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
            "sample_rate_hz": recording.sample_rate_hz,
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
    if rsp_nonfinite or ecg_nonfinite:
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
    "BIOPAC_ISI_MS",
    "BIOPAC_SAMPLE_RATE_HZ",
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
    "XETHRU_RECORD_BYTES",
    "XETHRU_RECORD_DTYPE",
    "XeThruMeta",
    "audit_manifest",
    "biopac_qc",
    "build_dataset_manifest",
    "build_manifest",
    "load_biopac_mat",
    "load_xethru_recording",
    "load_xethru_records",
    "open_xethru_files",
    "parse_biopac_start_datetime",
    "parse_xethru_meta",
    "radar_qc",
]
