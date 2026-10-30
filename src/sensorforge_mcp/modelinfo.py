"""Read a model file's input and output tensors (name, shape, type, quantization) without TensorFlow or ONNX.

Only what the local data check needs: a minimal reader for the TensorFlow Lite flatbuffer and for the ONNX protobuf.
Nothing here runs or loads the model.
"""
from __future__ import annotations

import os
import re
import struct
import zipfile
from dataclasses import dataclass, field

TFLITE_TYPES = {0: "float32", 1: "float16", 2: "int32", 3: "uint8", 4: "int64", 5: "string", 6: "bool", 7: "int16",
                8: "complex64", 9: "int8", 10: "float64", 11: "complex128", 12: "uint64", 13: "resource", 14: "variant",
                15: "uint32", 16: "uint16", 17: "int4", 18: "bfloat16"}
ONNX_TYPES = {1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32", 7: "int64", 8: "string",
              9: "bool", 10: "float16", 11: "float64", 12: "uint32", 13: "uint64", 16: "bfloat16"}


class ModelReadError(ValueError):
    """The file is not a model this reader understands."""


@dataclass
class Tensor:
    name: str
    shape: list            # ints; None for a dimension the file leaves open
    dtype: str
    scale: float | None = None
    zero_point: int | None = None

    def float_range(self):
        """The float values an int8 / uint8 / int16 input can represent, from its quantization, or None."""
        lo_hi = {"int8": (-128, 127), "uint8": (0, 255), "int16": (-32768, 32767)}.get(self.dtype)
        if not lo_hi or not self.scale or self.zero_point is None:
            return None
        return ((lo_hi[0] - self.zero_point) * self.scale, (lo_hi[1] - self.zero_point) * self.scale)


@dataclass
class ModelInfo:
    format: str            # "tflite" or "onnx"
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    channels_first: bool = False   # how the service reads an .npz's X for this model


# ---------------------------------------------------------------- TensorFlow Lite (flatbuffer)

class _FB:
    """Just enough of the flatbuffer wire format: tables, vectors, strings, scalars."""

    def __init__(self, buf):
        self.b = buf

    def u32(self, o):
        return struct.unpack_from("<I", self.b, o)[0]

    def i32(self, o):
        return struct.unpack_from("<i", self.b, o)[0]

    def field(self, table, idx):
        """Absolute position of field idx in the table at `table`, or None when absent."""
        vt = table - self.i32(table)
        vt_len = struct.unpack_from("<H", self.b, vt)[0]
        slot = 4 + 2 * idx
        if slot >= vt_len:
            return None
        off = struct.unpack_from("<H", self.b, vt + slot)[0]
        return table + off if off else None

    def table(self, table, idx):
        p = self.field(table, idx)
        return None if p is None else p + self.u32(p)

    def vector(self, table, idx):
        """(start, length) of a vector field, or (None, 0)."""
        p = self.field(table, idx)
        if p is None:
            return None, 0
        v = p + self.u32(p)
        return v + 4, self.u32(v)

    def ints(self, table, idx, fmt="<i", size=4):
        start, n = self.vector(table, idx)
        return [struct.unpack_from(fmt, self.b, start + i * size)[0] for i in range(n)] if start is not None else []

    def tables(self, table, idx):
        start, n = self.vector(table, idx)
        return [start + i * 4 + self.u32(start + i * 4) for i in range(n)] if start is not None else []

    def string(self, table, idx):
        p = self.field(table, idx)
        if p is None:
            return ""
        s = p + self.u32(p)
        n = self.u32(s)
        return bytes(self.b[s + 4:s + 4 + n]).decode("utf-8", "replace")

    def scalar(self, table, idx, fmt, default):
        p = self.field(table, idx)
        return default if p is None else struct.unpack_from(fmt, self.b, p)[0]


def is_tflite(head: bytes) -> bool:
    return len(head) >= 8 and head[4:8] == b"TFL3"


def read_tflite_bytes(buf) -> ModelInfo:
    if not is_tflite(bytes(buf[:8])):
        raise ModelReadError("not a TensorFlow Lite file (no TFL3 identifier)")
    try:
        fb = _FB(buf)
        model = fb.u32(0)
        subgraphs = fb.tables(model, 2)
        if not subgraphs:
            raise ModelReadError("the file holds no subgraph")
        sg = subgraphs[0]
        tensors = fb.tables(sg, 0)

        def tensor(i):
            t = tensors[i]
            shape = fb.ints(t, 0)
            sig = fb.ints(t, 7)
            if sig and len(sig) == len(shape):
                shape = [None if s < 0 else d for d, s in zip(shape, sig)]
            q = fb.table(t, 4)
            scale = zp = None
            if q is not None:
                sc = fb.ints(q, 2, "<f", 4)
                z = fb.ints(q, 3, "<q", 8)
                if len(sc) == 1:
                    scale = float(sc[0])
                if len(z) == 1:
                    zp = int(z[0])
            return Tensor(name=fb.string(t, 3), shape=list(shape), dtype=TFLITE_TYPES.get(fb.scalar(t, 1, "<b", 0), "unknown"),
                          scale=scale, zero_point=zp)

        return ModelInfo("tflite", [tensor(i) for i in fb.ints(sg, 1)], [tensor(i) for i in fb.ints(sg, 2)], False)
    except (struct.error, IndexError) as e:
        raise ModelReadError(f"the TensorFlow Lite file could not be read ({e})")


# ---------------------------------------------------------------- ONNX (protobuf)

def _varint(b, i):
    r = shift = 0
    while True:
        if i >= len(b):
            raise ModelReadError("the ONNX file is cut short")
        c = b[i]
        i += 1
        r |= (c & 0x7F) << shift
        if not c & 0x80:
            return r, i
        shift += 7
        if shift > 70:
            raise ModelReadError("the ONNX file is damaged")


def _fields(b, start=0, end=None):
    """(field number, wire type, value) of a protobuf message; length-delimited values as (start, end) slices."""
    i, end = start, len(b) if end is None else end
    while i < end:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = (i, i + n), i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ModelReadError("the ONNX file uses an unknown encoding")
        if i > end:
            raise ModelReadError("the ONNX file is cut short")
        yield f, wt, v


def _sub(b, v, want):
    return [val for f, wt, val in _fields(b, *v) if f == want]


def _text(b, v):
    return bytes(b[v[0]:v[1]]).decode("utf-8", "replace")


def _value_info(b, v):
    name, shape, dtype = "", [], "unknown"
    for f, wt, val in _fields(b, *v):
        if f == 1 and wt == 2:
            name = _text(b, val)
        elif f == 2 and wt == 2:                      # TypeProto
            for tt in _sub(b, val, 1):                # tensor_type
                for f2, wt2, val2 in _fields(b, *tt):
                    if f2 == 1 and wt2 == 0:
                        dtype = ONNX_TYPES.get(val2, "unknown")
                    elif f2 == 2 and wt2 == 2:        # TensorShapeProto
                        for dim in _sub(b, val2, 1):
                            d = None
                            for f3, wt3, val3 in _fields(b, *dim):
                                if f3 == 1 and wt3 == 0:
                                    d = val3 if val3 < (1 << 63) else None
                            shape.append(d)
    return Tensor(name=name, shape=shape, dtype=dtype)


def read_onnx_bytes(b) -> ModelInfo:
    try:
        graph = None
        for f, wt, v in _fields(b):
            if f == 7 and wt == 2:
                graph = v
        if graph is None:
            raise ModelReadError("not an ONNX model (no graph)")
        inits, ins, outs = set(), [], []
        for f, wt, v in _fields(b, *graph):
            if f == 5 and wt == 2:                    # initializer: TensorProto, name = 8
                for f2, wt2, v2 in _fields(b, *v):
                    if f2 == 8 and wt2 == 2:
                        inits.add(_text(b, v2))
            elif f == 11 and wt == 2:
                ins.append(_value_info(b, v))
            elif f == 12 and wt == 2:
                outs.append(_value_info(b, v))
        return ModelInfo("onnx", [t for t in ins if t.name not in inits], outs, True)
    except (IndexError, ValueError) as e:
        if isinstance(e, ModelReadError):
            raise
        raise ModelReadError(f"the ONNX file could not be read ({e})")


# ---------------------------------------------------------------- files

_HEX = re.compile(rb"0x([0-9a-fA-F]{1,2})")


def _c_array(text: bytes) -> bytes:
    """The bytes of the first C array in a firmware export's model_data.cc."""
    start = text.find(b"{")
    end = text.find(b"}", start)
    if start < 0 or end < 0:
        raise ModelReadError("model_data.cc holds no byte array")
    return bytes(int(m.group(1), 16) for m in _HEX.finditer(text, start, end))


def read_model(path: str) -> ModelInfo:
    """The inputs and outputs of a .tflite, an .onnx, or a .zip of a firmware export (its .tflite or model_data.cc)."""
    low = path.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            tfl = [n for n in names if n.lower().endswith(".tflite")]
            if tfl:
                return read_tflite_bytes(z.read(min(tfl, key=lambda n: (n.count("/"), n))))
            cc = [n for n in names if os.path.basename(n) == "model_data.cc"]
            if cc:
                return read_tflite_bytes(_c_array(z.read(cc[0])))
            onx = [n for n in names if n.lower().endswith(".onnx")]
            if onx:
                return read_onnx_bytes(z.read(onx[0]))
        raise ModelReadError("the zip holds no .tflite model (or a firmware export's model_data.cc)")
    with open(path, "rb") as f:
        buf = f.read()
    if low.endswith(".onnx"):
        return read_onnx_bytes(buf)
    return read_tflite_bytes(buf)
