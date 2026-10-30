"""Local files: resolving the paths an agent passes, checking them before an upload, and unpacking a download safely."""
from __future__ import annotations

import os
import stat
import zipfile

from .api import MAX_DATA_BYTES, MAX_MODEL_BYTES, ServiceError

MODEL_EXT = (".tflite", ".onnx", ".zip")
DATA_EXT = (".npz", ".zip")
# files that can run code when they are loaded: the service never opens them, and refuses a zip that holds one
SERVICE_CODE_EXT = (".pt", ".pth", ".pkl", ".pickle", ".joblib", ".h5", ".hdf5", ".keras", ".pb", ".pbtxt", ".so", ".dylib",
                    ".dll")
# model formats to export to .tflite or .onnx before uploading
EXPORT_EXT = SERVICE_CODE_EXT + (".ckpt", ".safetensors", ".mlmodel", ".pte", ".ptl", ".torchscript")
EXPORT_FIRST = ("Keras .h5 / .keras and PyTorch .pt / .pth files can run code when loaded, so the service does not open "
                "them: export the model to .tflite or .onnx first (for PyTorch, torch.onnx.export; for Keras, "
                "tf.lite.TFLiteConverter.from_keras_model; an int8 model is best).")
MAX_UNPACK_BYTES = 2 * 1024 * 1024 * 1024
MAX_UNPACK_FILES = 10_000


def resolve(path: str) -> str:
    """An absolute path: ~ expanded, relative paths from the working directory."""
    if not path or not str(path).strip():
        raise ServiceError("No path given.")
    return os.path.abspath(os.path.expanduser(str(path).strip()))


def _mb(n):
    return f"{n / (1024 * 1024):.1f} MB"


def check_model(path: str) -> str:
    """The model's absolute path once it is a file the service takes; otherwise a ServiceError saying what to do."""
    p = resolve(path)
    low = p.lower().rstrip("/")
    if os.path.isdir(p):
        if os.path.exists(os.path.join(p, "saved_model.pb")):
            raise ServiceError(f"{path} is a TensorFlow SavedModel folder. " + EXPORT_FIRST)
        raise ServiceError(f"{path} is a folder. Pass the .tflite or .onnx file, or zip a firmware export folder first "
                           f"(zip -r export.zip {os.path.basename(p)}) and pass the .zip.")
    if low.endswith(EXPORT_EXT):
        raise ServiceError(f"{os.path.basename(p)}: " + EXPORT_FIRST)
    if not os.path.isfile(p):
        raise ServiceError(f"No model file at {p}.")
    if not low.endswith(MODEL_EXT):
        raise ServiceError(f"{os.path.basename(p)} is not a model the service takes: upload a .tflite or .onnx file, or a "
                           ".zip of a firmware export folder.")
    n = os.path.getsize(p)
    if n == 0:
        raise ServiceError(f"{p} is empty.")
    if n > MAX_MODEL_BYTES:
        raise ServiceError(f"{os.path.basename(p)} is {_mb(n)}; the service takes models up to 50 MB.")
    with open(p, "rb") as f:
        head = f.read(8)
    if low.endswith(".tflite") and head[4:8] != b"TFL3":
        raise ServiceError(f"{os.path.basename(p)} is not a TensorFlow Lite file (no TFL3 identifier). Export it again with "
                           "the TensorFlow Lite converter.")
    if low.endswith(".zip"):
        _check_zip(p, 4 * MAX_MODEL_BYTES, need_model=True)
    return p


def check_data_file(path: str) -> str:
    p = resolve(path)
    if os.path.isdir(p):
        raise ServiceError(f"{path} is a folder. Zip it first (zip -r data.zip {os.path.basename(p)}): one folder per "
                           "class inside, or train/ and test/ folders each holding the class folders.")
    if not os.path.isfile(p):
        raise ServiceError(f"No data file at {p}.")
    if not p.lower().endswith(DATA_EXT):
        raise ServiceError(f"{os.path.basename(p)}: Optimize takes data as an .npz (X, y) or a .zip of image folders, one "
                           "per class.")
    n = os.path.getsize(p)
    if n == 0:
        raise ServiceError(f"{p} is empty.")
    if n > MAX_DATA_BYTES:
        raise ServiceError(f"{os.path.basename(p)} is {_mb(n)}; the service takes data up to 500 MB. Use fewer samples (a "
                           "few hundred per class are enough to measure), or float32 rather than float64.")
    if p.lower().endswith(".zip"):
        _check_zip(p, 4 * MAX_DATA_BYTES)
    return p


def _check_zip(p, max_bytes, need_model=False):
    """What the service refuses in an uploaded zip, found before the upload."""
    try:
        with zipfile.ZipFile(p) as z:
            infos = z.infolist()
    except (zipfile.BadZipFile, OSError) as e:
        raise ServiceError(f"{os.path.basename(p)} is not a zip file that can be read ({e}): zip the folder again.")
    total, names = 0, []
    for i in infos:
        n = i.filename.replace("\\", "/")
        if n.startswith("__MACOSX/") or os.path.basename(n.rstrip("/")).startswith("._"):
            continue
        if n.startswith("/") or ".." in n.split("/") or (len(n) > 1 and n[1] == ":"):
            raise ServiceError(f"{os.path.basename(p)} holds a path outside its folder ({i.filename}); the service refuses "
                               "it. Zip the folder again.")
        if (i.external_attr >> 16) & 0o170000 == 0o120000:
            raise ServiceError(f"{os.path.basename(p)} holds a symbolic link ({i.filename}); the service refuses it.")
        if not i.is_dir() and n.lower().endswith(SERVICE_CODE_EXT):
            raise ServiceError(f"{os.path.basename(p)} holds {os.path.basename(n)}, a file that can run code when loaded; "
                               "the service refuses the zip. Remove it and zip again.")
        total += i.file_size
        names.append(n)
    if total > max_bytes:
        raise ServiceError(f"{os.path.basename(p)} unpacks to {_mb(total)}; the service unpacks up to {_mb(max_bytes)}.")
    if need_model and not any(n.endswith(".tflite") or os.path.basename(n) == "model_data.cc" for n in names):
        raise ServiceError(f"{os.path.basename(p)} holds no .tflite model (or a firmware export's model_data.cc).")


# ------------------------------------------------------------ unpacking a download

def _safe_member(name: str):
    n = name.replace("\\", "/")
    parts = [x for x in n.split("/") if x not in ("", ".")]
    if n.startswith("/") or (len(n) > 1 and n[1] == ":") or ".." in parts or not parts:
        return None
    return parts


def unpack(zip_path: str, dest: str, overwrite: bool = False) -> list:
    """Unpacks a download into dest. Every entry is checked first: a path outside dest, a link or a device file stops the
    whole unpack before anything is written. Files are written plain (never executable); returns their relative paths."""
    dest = os.path.realpath(dest)
    with zipfile.ZipFile(zip_path) as z:
        plan, total = [], 0
        for i in z.infolist():
            parts = _safe_member(i.filename)
            if parts is None:
                raise ServiceError(f"The download holds a path outside the folder ({i.filename!r}); nothing was unpacked. "
                                   "Report this run id to SensorForge.")
            mode = (i.external_attr >> 16) & 0o170000
            if mode and mode not in (stat.S_IFREG, stat.S_IFDIR):
                raise ServiceError(f"The download holds a link or special file ({i.filename!r}); nothing was unpacked.")
            target = os.path.realpath(os.path.join(dest, *parts))
            if os.path.commonpath([dest, target]) != dest:
                raise ServiceError(f"The download holds a path outside the folder ({i.filename!r}); nothing was unpacked.")
            total += i.file_size
            plan.append((i, target, "/".join(parts)))
        if len(plan) > MAX_UNPACK_FILES or total > MAX_UNPACK_BYTES:
            raise ServiceError("The download unpacks to more files or bytes than any run produces; nothing was unpacked.")
        if not overwrite:
            clash = [rel for i, t, rel in plan if not i.is_dir() and os.path.lexists(t)]
            if clash:
                raise ServiceError(f"{clash[0]} already exists in {dest}; pass overwrite=True to replace it, or another folder.")
        written, seen = [], 0
        os.makedirs(dest, exist_ok=True)
        for i, target, rel in plan:
            if i.is_dir() or i.filename.endswith("/"):
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.islink(target):
                os.unlink(target)
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o644)
            with z.open(i) as src, os.fdopen(fd, "wb") as out:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    seen += len(b)
                    if seen > MAX_UNPACK_BYTES:
                        raise ServiceError("The download unpacks to more than it said; stopped.")
                    out.write(b)
            written.append(rel)
    return written


def describe(rel: str) -> str:
    """What a file in an unpacked download is for."""
    b = os.path.basename(rel).lower()
    d = os.path.dirname(rel).replace("\\", "/")
    if d.startswith("over_limit"):
        return "the smaller model that was over the accuracy limit (not the default; see the report)"
    if b == "optimized_tflm.tflite":
        return "the optimized model for TensorFlow Lite Micro: the file to deploy"
    if b.endswith("_lossless.tflite"):
        return "the smaller model for TensorFlow Lite Micro, outputs identical to yours: the file to deploy"
    if b.endswith("litert.tflite") or "twin" in b:
        return "the same model in the form desktop LiteRT / TensorFlow Lite loads, for checking it on a PC"
    if b.endswith(".tflite"):
        return "a model file the report describes"
    if b == "model_data.cc":
        return "the model as a C array, to compile into the firmware"
    if b == "model_data.h":
        return "declares the array (g_model_data, g_model_data_len)"
    if b == "model_ops.h":
        return ("the operators to register with TensorFlow Lite Micro's op resolver, the input and output scale and zero "
                "point, and the tensor arena size (TFLM_ARENA_BYTES)")
    if b == "integration.md":
        return "how to build these into the firmware"
    if b.startswith("sf_") and b.endswith((".cc", ".h")):
        return "operators the model needs beside TensorFlow Lite Micro's own: compile into the firmware"
    if b in ("report.md", "lossless.md", "fit.md"):
        return "the run's report: what changed and the measured figures"
    if b.endswith(".pdf"):
        return "the report as a PDF"
    if b.endswith((".cc", ".c", ".h", ".hpp", ".cpp")):
        return "C source for the firmware build"
    return ""
