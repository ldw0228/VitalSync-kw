import struct
import zlib
from pathlib import Path

import numpy as np


MI_TYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    9: np.float64,
    12: np.int64,
    13: np.uint64,
}


def _element(buf, pos):
    first = struct.unpack_from('<I', buf, pos)[0]
    small_type = first & 0xFFFF
    small_size = first >> 16
    if small_size:
        data = buf[pos + 4:pos + 4 + small_size]
        return small_type, data, pos + 8
    data_type, size = struct.unpack_from('<II', buf, pos)
    start = pos + 8
    end = start + size
    return data_type, buf[start:end], (end + 7) & ~7


def _numeric(data_type, payload):
    dtype = MI_TYPES.get(data_type)
    if dtype is None:
        return payload
    return np.frombuffer(payload, dtype=np.dtype(dtype).newbyteorder('<')).copy()


def _matrix(payload):
    pos = 0
    flag_type, flag_data, pos = _element(payload, pos)
    flags = _numeric(flag_type, flag_data)
    is_complex = bool(int(flags[0]) & 0x0800)

    dim_type, dim_data, pos = _element(payload, pos)
    dims = tuple(int(x) for x in _numeric(dim_type, dim_data))

    _, name_data, pos = _element(payload, pos)
    name = name_data.decode('utf-8', errors='replace')

    real_type, real_data, pos = _element(payload, pos)
    real = _numeric(real_type, real_data)
    if dims and isinstance(real, np.ndarray):
        real = real.reshape(dims, order='F')

    if is_complex:
        imag_type, imag_data, pos = _element(payload, pos)
        imag = _numeric(imag_type, imag_data)
        if dims and isinstance(imag, np.ndarray):
            imag = imag.reshape(dims, order='F')
        real = real + 1j * imag
    return name, real


def _parse_elements(buf, out):
    pos = 0
    while pos + 8 <= len(buf):
        data_type, payload, next_pos = _element(buf, pos)
        if data_type == 15:  # miCOMPRESSED
            _parse_elements(zlib.decompress(payload), out)
        elif data_type == 14:  # miMATRIX
            name, value = _matrix(payload)
            out[name] = value
        if next_pos <= pos:
            break
        pos = next_pos


def loadmat(path):
    raw = Path(path).read_bytes()
    if not raw.startswith(b'MATLAB 5.0 MAT-file'):
        raise ValueError('Only MATLAB 5 MAT-files are supported')
    if raw[126:128] != b'IM':
        raise ValueError('Only little-endian MAT-files are supported')
    out = {}
    _parse_elements(raw[128:], out)
    return out
