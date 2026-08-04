"""Wire payload decoding shared by the portable synchronous RPC client."""

from __future__ import annotations

import numpy as np

from .generated import policy_data_v2_pb2 as pb


_PROTO_TO_DTYPE = {
    pb.BOOL: np.dtype("?"),
    pb.UINT8: np.dtype("u1"),
    pb.UINT16: np.dtype("<u2"),
    pb.UINT32: np.dtype("<u4"),
    pb.UINT64: np.dtype("<u8"),
    pb.INT8: np.dtype("i1"),
    pb.INT16: np.dtype("<i2"),
    pb.INT32: np.dtype("<i4"),
    pb.INT64: np.dtype("<i8"),
    pb.FLOAT16: np.dtype("<f2"),
    pb.FLOAT32: np.dtype("<f4"),
    pb.FLOAT64: np.dtype("<f8"),
}


def decode_tensor(payload: pb.TensorPayload) -> np.ndarray:
    try:
        dtype = _PROTO_TO_DTYPE[payload.dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype enum: {payload.dtype}") from exc
    shape = tuple(int(item) for item in payload.shape)
    if not shape or any(item <= 0 for item in shape):
        raise ValueError("tensor shape is invalid")
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(payload.data) != expected:
        raise ValueError(
            f"tensor payload has {len(payload.data)} bytes, expected {expected}"
        )
    result = np.frombuffer(payload.data, dtype=dtype).reshape(shape)
    result.setflags(write=False)
    return result
