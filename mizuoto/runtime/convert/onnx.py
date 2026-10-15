"""An ONNX graph's nodes and weights, read without the ``onnx`` package.

Some upstream models exist only as ONNX files (CosyVoice's speech tokenizers and CAM++). Their weights are
the graph's initializers (and the tensors of ``Constant`` nodes); the nodes name them, since exporters give
many weights anonymous names (``onnx::MatMul_2536``). This module parses the protobuf wire format of
``ModelProto`` just far enough: the graph's nodes (name, op type, inputs, outputs, attributes) and its
tensors. Tensor data is read lazily from a memory map, so a large file costs only what is used.

Field numbers are those of ``onnx/onnx.proto`` (IR version 3 onward).
"""

from __future__ import annotations

import mmap
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# TensorProto.DataType → numpy dtype
DTYPES: dict[int, Any] = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    4: np.uint16,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
    12: np.uint32,
    13: np.uint64,
}


class OnnxError(ValueError):
    pass


def _varint(buf: memoryview, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _fields(buf: memoryview, start: int = 0, end: int | None = None) -> Iterator[tuple[int, int, Any]]:
    """(field number, wire type, value): an int for varints and fixed widths, a (start, end) span for bytes."""
    pos, end = start, len(buf) if end is None else end
    while pos < end:
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value, pos = int.from_bytes(buf[pos : pos + 8], "little"), pos + 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value, pos = (pos, pos + length), pos + length
        elif wire == 5:
            value, pos = int.from_bytes(buf[pos : pos + 4], "little"), pos + 4
        else:
            raise OnnxError(f"Unsupported protobuf wire type {wire}")
        yield number, wire, value


def _signed(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


def _packed_varints(buf: memoryview, span: tuple[int, int]) -> list[int]:
    out, pos = [], span[0]
    while pos < span[1]:
        v, pos = _varint(buf, pos)
        out.append(_signed(v))
    return out


@dataclass
class OnnxTensor:
    name: str
    dims: tuple[int, ...]
    data_type: int
    _buf: memoryview = field(repr=False)
    _raw: tuple[int, int] | None = field(default=None, repr=False)
    _values: list[Any] | None = field(default=None, repr=False)

    def numpy(self) -> np.ndarray:
        """A writable copy of the tensor's values."""
        dtype = DTYPES.get(self.data_type)
        if dtype is None:
            raise OnnxError(f"{self.name}: unsupported ONNX data type {self.data_type}")
        if self._raw is not None:
            a, b = self._raw
            arr = np.frombuffer(self._buf[a:b], dtype=np.dtype(dtype).newbyteorder("<")).astype(dtype, copy=True)
        else:
            arr = np.array(self._values or [], dtype=dtype)
        return arr.reshape(self.dims)


def _tensor(buf: memoryview, span: tuple[int, int]) -> OnnxTensor:
    dims: list[int] = []
    data_type = 0
    name = ""
    raw: tuple[int, int] | None = None
    values: list[Any] = []
    for number, wire, value in _fields(buf, *span):
        if number == 1:
            dims += _packed_varints(buf, value) if wire == 2 else [_signed(value)]
        elif number == 2:
            data_type = value
        elif number == 8:
            name = bytes(buf[value[0] : value[1]]).decode("utf-8")
        elif number == 9:
            raw = value
        elif number == 4:  # float_data
            if wire == 2:
                values += list(struct.unpack(f"<{(value[1] - value[0]) // 4}f", buf[value[0] : value[1]]))
            else:
                values.append(struct.unpack("<f", value.to_bytes(4, "little"))[0])
        elif number in (5, 7):  # int32_data, int64_data
            values += _packed_varints(buf, value) if wire == 2 else [_signed(value)]
        elif number == 10:  # double_data
            if wire == 2:
                values += list(struct.unpack(f"<{(value[1] - value[0]) // 8}d", buf[value[0] : value[1]]))
            else:
                values.append(struct.unpack("<d", value.to_bytes(8, "little"))[0])
        elif number == 14 and value == 1:
            raise OnnxError(f"{name or 'a tensor'} keeps its data in an external file, which is not supported")
    return OnnxTensor(name, tuple(dims), data_type, buf, raw, values if raw is None else None)


@dataclass
class OnnxNode:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any]


def _attribute(buf: memoryview, span: tuple[int, int]) -> tuple[str, Any]:
    name = ""
    value: Any = None
    ints: list[int] = []
    floats: list[float] = []
    for number, wire, v in _fields(buf, *span):
        if number == 1:
            name = bytes(buf[v[0] : v[1]]).decode("utf-8")
        elif number == 2:
            value = struct.unpack("<f", v.to_bytes(4, "little"))[0]
        elif number == 3:
            value = _signed(v)
        elif number == 4:
            value = bytes(buf[v[0] : v[1]])
        elif number == 5:
            value = _tensor(buf, v)
        elif number == 7:
            floats += list(struct.unpack(f"<{(v[1] - v[0]) // 4}f", buf[v[0] : v[1]])) if wire == 2 else [struct.unpack("<f", v.to_bytes(4, "little"))[0]]
        elif number == 8:
            ints += _packed_varints(buf, v) if wire == 2 else [_signed(v)]
    if value is None and ints:
        value = ints
    elif value is None and floats:
        value = floats
    return name, value


def _node(buf: memoryview, span: tuple[int, int]) -> OnnxNode:
    inputs: list[str] = []
    outputs: list[str] = []
    name = op_type = ""
    attributes: dict[str, Any] = {}
    for number, _wire, v in _fields(buf, *span):
        if number == 1:
            inputs.append(bytes(buf[v[0] : v[1]]).decode("utf-8"))
        elif number == 2:
            outputs.append(bytes(buf[v[0] : v[1]]).decode("utf-8"))
        elif number == 3:
            name = bytes(buf[v[0] : v[1]]).decode("utf-8")
        elif number == 4:
            op_type = bytes(buf[v[0] : v[1]]).decode("utf-8")
        elif number == 5:
            key, value = _attribute(buf, v)
            attributes[key] = value
    return OnnxNode(name, op_type, inputs, outputs, attributes)


@dataclass
class OnnxGraph:
    nodes: list[OnnxNode]
    initializers: dict[str, OnnxTensor]
    _map: Any = field(default=None, repr=False)
    _buf: memoryview | None = field(default=None, repr=False)

    def constants(self) -> dict[str, OnnxTensor]:
        """The tensors of ``Constant`` nodes, by output name."""
        return {n.outputs[0]: n.attributes["value"] for n in self.nodes if n.op_type == "Constant" and isinstance(n.attributes.get("value"), OnnxTensor)}

    def tensor(self, name: str) -> np.ndarray:
        if name in self.initializers:
            return self.initializers[name].numpy()
        consts = self.constants()
        if name in consts:
            return consts[name].numpy()
        raise KeyError(name)

    def close(self) -> None:
        """Release the memory map; the tensors cannot be read afterwards."""
        for t in self.initializers.values():
            t._buf = memoryview(b"")
        self.initializers = {}
        self.nodes = []
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        if self._map is not None:
            self._map.close()
            self._map = None


def read_onnx(path: str | Path) -> OnnxGraph:
    """The main graph of an ONNX model file. Call ``close()`` when done with its tensors."""
    with Path(path).open("rb") as f:
        mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    buf = memoryview(mapped)
    graph_span = next((v for number, wire, v in _fields(buf) if number == 7 and wire == 2), None)
    if graph_span is None:
        raise OnnxError(f"{Path(path).name} has no graph")
    nodes: list[OnnxNode] = []
    initializers: dict[str, OnnxTensor] = {}
    for number, _wire, v in _fields(buf, *graph_span):
        if number == 1:
            nodes.append(_node(buf, v))
        elif number == 5:
            t = _tensor(buf, v)
            initializers[t.name] = t
    return OnnxGraph(nodes, initializers, mapped, buf)
