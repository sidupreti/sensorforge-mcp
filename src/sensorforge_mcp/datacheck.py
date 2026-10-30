"""The local data check: is this .npz or image zip what Optimize needs for this model? Nothing is uploaded.

The rules follow the service's own data guide ("How to prepare data for Optimize"):
  images      a .zip of folders, one per class (JPEG or PNG), optionally train/ and test/ folders each holding the class
              folders; folder names in alphabetical order map to the model's outputs 0, 1, 2, ...
  everything  an .npz with X (the model's inputs exactly as the front end produces them: float32, one row per sample, in
  else        the model's input shape) and y (each sample's class index); X_test / y_test for a fixed test set. Without y
              the original model's own answers become the labels.
"""
from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field

import numpy as np

from .modelinfo import ModelInfo, ModelReadError, read_model

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".webp")
CODE_EXT = (".pt", ".pth", ".pkl", ".pickle", ".joblib", ".h5", ".hdf5", ".keras", ".pb", ".pbtxt", ".so", ".dylib", ".dll")
SCAN_BYTES = 256 << 20          # how much of X is read for the value checks (NaN, range)
MAX_ZIP_FILES = 200_000


@dataclass
class DataReport:
    problems: list = field(default_factory=list)     # each blocks a useful run: fix before optimize
    warnings: list = field(default_factory=list)     # the run can go ahead, but the result may not be what you want
    notes: list = field(default_factory=list)        # how the service will read the data
    facts: list = field(default_factory=list)        # what was found, one line each

    @property
    def ok(self):
        return not self.problems

    def text(self, data_name, model_name):
        out = [f"Data check: {data_name} against {model_name} (local, nothing uploaded)"]
        out += [f"  {f}" for f in self.facts]
        if self.problems:
            out.append(f"Result: NOT READY, {len(self.problems)} problem{'s' if len(self.problems) != 1 else ''} to fix before optimize.")
        else:
            out.append("Result: ready for optimize." + (" Read the warnings first." if self.warnings else ""))
        for title, items in (("Problems", self.problems), ("Warnings", self.warnings), ("Notes", self.notes)):
            if items:
                out.append(f"{title}:")
                out += [f"- {x}" for x in items]
        return "\n".join(out)


def _shape(s):
    return "(" + ", ".join("?" if d is None else str(d) for d in s) + ")"


def _per_sample(model_shape):
    """The model's input shape without its batch dimension: [1, 49, 10, 1] -> [49, 10, 1]."""
    if len(model_shape) >= 2 and (model_shape[0] in (1, None)):
        return list(model_shape[1:])
    return list(model_shape)


def _prod(s):
    p = 1
    for d in s:
        p *= int(d)
    return p


def _model_summary(m: ModelInfo):
    def one(t):
        r = t.float_range()
        extra = f", takes values {r[0]:.3g} to {r[1]:.3g}" if r else ""
        return (f"{t.name} " if len(m.inputs) > 1 and t.name else "") + f"{_shape(t.shape)} {t.dtype}{extra}"
    ins = "; ".join(one(t) for t in m.inputs)
    outs = "; ".join(f"{_shape(t.shape)} {t.dtype}" for t in m.outputs)
    return f"Model: {m.format}, input {ins}; output {outs}"


def _classes_of(m: ModelInfo):
    """How many class scores the model outputs (its first output's last dimension), or None."""
    if not m.outputs or not m.outputs[0].shape:
        return None
    d = m.outputs[0].shape[-1]
    return d if isinstance(d, int) and d > 0 else None


# ---------------------------------------------------------------- .npz

def _npz_members(path):
    """{array name: (shape, dtype, fortran_order)} read from the .npy headers alone (nothing loaded)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with z.open(info) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            out[info.filename[:-4]] = (tuple(shape), dtype, fortran)
    return out


def _scan_values(path, key, dtype, shape):
    """(finite, min, max, p1, p99, rows_read) over the first SCAN_BYTES of X, read straight from the archive."""
    n_rows = shape[0] if shape else 0
    row_bytes = max(1, _prod(shape[1:]) * dtype.itemsize) if shape else dtype.itemsize
    want_rows = max(1, min(n_rows, SCAN_BYTES // row_bytes)) if n_rows else 0
    finite, lo, hi, sample = True, None, None, []
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        version = np.lib.format.read_magic(f)
        (np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        left = want_rows * row_bytes
        chunk_rows = max(1, (16 << 20) // row_bytes)
        while left > 0:
            buf = f.read(min(left, chunk_rows * row_bytes))
            if not buf:
                break
            left -= len(buf)
            usable = len(buf) - len(buf) % dtype.itemsize
            a = np.frombuffer(buf[:usable], dtype=dtype)
            if a.size == 0:
                continue
            if a.dtype.kind == "f":
                good = np.isfinite(a)
                if not good.all():
                    finite = False
                a = a[good]
                if a.size == 0:
                    continue
            lo = float(a.min()) if lo is None else min(lo, float(a.min()))
            hi = float(a.max()) if hi is None else max(hi, float(a.max()))
            sample.append(a[:: max(1, a.size // 200_000)].astype(np.float64))
    p1 = p99 = None
    if sample:
        p1, p99 = (float(v) for v in np.percentile(np.concatenate(sample), [1, 99]))
    return finite, lo, hi, p1, p99, want_rows


def _match_shape(sample, model_in, channels_first):
    """'' when X's per-sample shape fits the model input, else why not; second value: a note to add."""
    want = _per_sample(model_in)
    if any(d is None for d in want):
        fixed = [d for d in want if d is not None]
        if len(want) == len(sample) and all(w is None or w == s for w, s in zip(want, sample)):
            return "", None
        return (f"its samples are {_shape(sample)} and the model takes {_shape(want)}", None) if fixed else ("", None)
    if list(sample) == want:
        return "", None
    if _prod(sample) == _prod(want):
        if sorted(sample) == sorted(want) and len(sample) == len(want) >= 2:
            if channels_first and list(sample[-1:] + sample[:-1]) == want:
                return (f"its samples are {_shape(sample)}, channels last, and this ONNX model takes {_shape(want)}, channels "
                        f"first: move the channel axis, X = np.moveaxis(X, -1, 1)", None)
            if not channels_first and list(sample[1:] + sample[:1]) == want:
                return (f"its samples are {_shape(sample)}, channels first, and this TensorFlow Lite model takes "
                        f"{_shape(want)}, channels last: move the channel axis, X = np.moveaxis(X, 1, -1)", None)
        return "", (f"X's samples are {_shape(sample)} and the model takes {_shape(want)}: the same {_prod(want)} values, "
                    "so the service reshapes them. Check the order matches what your front end feeds the model.")
    return (f"its samples are {_shape(sample)} ({_prod(sample)} values) and the model takes {_shape(want)} "
            f"({_prod(want)} values) per sample", None)


def _labels(y, name, n_classes, n_rows, rep: DataReport, output):
    """Checks one label array; returns the classes it holds (or None)."""
    y = np.asarray(y)
    if len(y) != n_rows:
        rep.problems.append(f"{name} has {len(y)} labels but its X has {n_rows} samples: one label per sample.")
        return None
    if y.ndim == 2 and y.shape[1] >= 2 and np.isin(y, (0, 1)).all():
        if n_classes and y.shape[1] != n_classes:
            rep.problems.append(f"{name} is multi-label with {y.shape[1]} columns, but the model outputs {n_classes} scores.")
        rep.facts.append(f"{name}: multi-label, {y.shape[1]} classes, {int(y.sum())} labels set")
        return None
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or not (y.dtype.kind in "iub" or (y.dtype.kind == "f" and np.all(np.mod(y, 1) == 0))):
        if output == "values":
            rep.notes.append(f"{name} is not class labels {list(y.shape)}; with output='values' it is not needed: the "
                             "optimized model is held to your model's own outputs.")
        elif y.ndim == 2 and y.dtype.kind == "f":
            rep.problems.append(f"{name} looks like scores or one-hot rows {list(y.shape)}: y must be one class index per "
                                f"sample, e.g. y = {name}.argmax(axis=1). For a model whose outputs are values (a detector, "
                                "an autoencoder, regression), use output='values' and leave y out.")
        else:
            rep.problems.append(f"{name} is not one integer class index per sample (shape {list(y.shape)}, {y.dtype}). "
                                "Save the class indices, e.g. y.astype('int64'); for a model whose outputs are values use "
                                "output='values' and leave y out.")
        return None
    if y.dtype.kind == "f":
        rep.warnings.append(f"{name} holds whole numbers stored as {y.dtype}: save it as int64 (y.astype('int64')).")
    yi = y.astype(np.int64)
    if yi.size and yi.min() < 0:
        rep.problems.append(f"{name} has negative labels (lowest {int(yi.min())}): labels are class indices 0, 1, 2, ...")
        return None
    classes = np.unique(yi)
    if n_classes and yi.size and yi.max() >= n_classes:
        rep.problems.append(f"{name} has class {int(yi.max())}, but the model outputs {n_classes} scores (classes 0 to "
                            f"{n_classes - 1}): use the class index the model's outputs are in.")
    counts = np.bincount(yi) if yi.size else np.array([])
    present = counts[counts > 0]
    rep.facts.append(f"{name}: {len(classes)} classes, {int(present.min()) if present.size else 0} to "
                     f"{int(present.max()) if present.size else 0} samples each")
    return set(int(c) for c in classes)


def check_npz(path, m: ModelInfo, output="classes") -> DataReport:
    rep = DataReport()
    rep.facts.append(_model_summary(m))
    try:
        members = _npz_members(path)
    except (zipfile.BadZipFile, OSError, ValueError) as e:
        rep.problems.append(f"{os.path.basename(path)} cannot be read as an .npz ({e}). Save it with "
                            "np.savez('data.npz', X=X.astype('float32'), y=y.astype('int64')).")
        return rep
    keys = set(members)
    rep.facts.append("Arrays: " + ", ".join(f"{k} {_shape(s)} {d}" for k, (s, d, _) in sorted(members.items())))
    obj = [k for k, (_, d, _) in members.items() if d.hasobject]
    if obj:
        rep.problems.append(f"{', '.join(sorted(obj))} hold Python objects (dtype object), which the service does not open. "
                            "Save plain numeric arrays: X.astype('float32'), y.astype('int64').")
        return rep
    xs = [k for k in ("X", "X_train") if k in keys]
    if not xs:
        rep.problems.append(f"No X array: found {sorted(keys) or 'none'}. Save the model's inputs as X (and the labels as y): "
                            "np.savez('data.npz', X=X, y=y), or X_train / y_train / X_test / y_test.")
        return rep
    if "X" in keys and "X_train" in keys:
        rep.warnings.append("Both X and X_train are present; with X_train / y_train / X_test / y_test all present those are "
                            "used, otherwise X / y.")
    pairs = [("X", "y"), ("X_train", "y_train"), ("X_test", "y_test")]
    n_classes = _classes_of(m)
    streaming = len(m.inputs) > 1
    main_in = m.inputs[0] if m.inputs else None
    have_y = any(k in keys for k in ("y", "y_train", "y_test"))
    seen_classes = {}
    for xk, yk in pairs:
        if xk not in keys:
            if yk in keys:
                rep.problems.append(f"{yk} is present without {xk}.")
            continue
        shape, dtype, _ = members[xk]
        if len(shape) < 2:
            rep.problems.append(f"{xk} is {_shape(shape)}: it must be one row per sample, (N, ...) with the sample's own axes.")
            continue
        n, sample = shape[0], list(shape[1:])
        if dtype.kind not in "fiub":
            rep.problems.append(f"{xk} is {dtype}: save the model's inputs as float32, X.astype('float32').")
            continue
        if dtype != np.float32:
            if dtype.kind == "f":
                rep.warnings.append(f"{xk} is {dtype}; the service reads float32. Saving X.astype('float32') halves the "
                                    "upload and keeps the same values.")
            else:
                rep.warnings.append(f"{xk} is {dtype}. The service reads the model's inputs as float32 values exactly as your "
                                    "front end produces them (before the model's own int8 conversion): check these are "
                                    "those values, then save X.astype('float32').")
        if streaming and main_in is not None:
            lasts = [t.shape[-1] for t in m.inputs if t.shape]
            if sample[-1] not in lasts:
                rep.problems.append(f"{xk}'s samples are {_shape(sample)}; none of the model's inputs "
                                    f"({', '.join(_shape(t.shape) for t in m.inputs)}) takes {sample[-1]} values per step.")
        elif main_in is not None:
            why, note = _match_shape(sample, main_in.shape, m.channels_first)
            if why:
                rep.problems.append(f"{xk} does not fit the model's input: {why}.")
            elif note and note not in rep.notes:
                rep.notes.append(note)
        if n == 0:
            rep.problems.append(f"{xk} has no samples.")
            continue
        finite, lo, hi, p1, p99, rows = _scan_values(path, xk, dtype, shape)
        part = "" if rows >= n else f" (first {rows:,} of {n:,} samples checked)"
        if not finite:
            rep.problems.append(f"{xk} holds NaN or infinite values{part}: remove or replace them.")
        if lo is not None:
            rep.facts.append(f"{xk}: {n:,} samples, values {lo:.4g} to {hi:.4g}{part}")
            r = main_in.float_range() if main_in is not None and not streaming else None
            # most of X (1st to 99th percentile) well outside what the int8 input can hold: a scaling step is missing (pixels
            # 0..255 into a model that takes 0..1); a few outliers past the range are normal and clipped
            width = r[1] - r[0] if r else 0
            if r and width > 0 and (p99 > r[1] + width or p1 < r[0] - width):
                rep.warnings.append(f"Most of {xk} lies between {p1:.4g} and {p99:.4g}, but the model's input covers {r[0]:.4g} "
                                    f"to {r[1]:.4g}. X should be the values your front end feeds the model, after its "
                                    "scaling (for images, e.g. pixels / 255).")
        if yk in keys:
            with np.load(path, allow_pickle=False) as z:
                ya = z[yk]
            if streaming:
                continue
            got = _labels(ya, yk, n_classes, n, rep, output)
            if got is not None:
                seen_classes[yk] = got
        elif have_y and not streaming and output != "values":
            rep.problems.append(f"{xk} has no {yk}: give each X array its labels, or leave every y out.")
    if "split" in keys:
        rep.notes.append("split: 0 = train, 1 = validation, 2 = test (validation is the test set when there is no 2).")
    if streaming:
        rep.notes.append(f"The model has {len(m.inputs)} inputs (a streaming model that keeps its state between calls): X is "
                         "the frames your front end produces, in time order, one per row, one recording after another. "
                         "y is not used: the optimized model is held to your model's outputs at every step.")
    elif not have_y:
        if output == "values":
            rep.notes.append("No y: with output='values' the optimized model is held to your model's own outputs.")
        else:
            rep.notes.append("No y: the original model's own answers become the labels, so accuracy means agreement with "
                             "your model. Add y (class indices) to measure real accuracy.")
    if n_classes and seen_classes and output != "values":
        allc = set().union(*seen_classes.values())
        missing = sorted(set(range(n_classes)) - allc)
        if missing and len(missing) < n_classes:
            rep.warnings.append(f"No samples of class{'es' if len(missing) > 1 else ''} {', '.join(map(str, missing[:12]))}"
                                f"{'...' if len(missing) > 12 else ''} (the model has {n_classes} outputs).")
        tr = seen_classes.get("y_train") or seen_classes.get("y")
        te = seen_classes.get("y_test")
        if tr is not None and te is not None and te - tr:
            rep.warnings.append(f"y_test has classes y_train lacks: {sorted(te - tr)[:12]}.")
    if n_classes == 1 and output != "values" and have_y:
        rep.warnings.append("The model has a single output value: if it is a regression, a detector or an autoencoder, use "
                            "output='values' (y is then not needed).")
    _size_notes(path, members, rep)
    return rep


def _size_notes(path, members, rep):
    ns = [s[0] for k, (s, _, _) in members.items() if k.startswith("X") and s]
    if ns and sum(ns) < 200:
        rep.warnings.append(f"Only {sum(ns)} samples in all: a few hundred per class are enough to measure accuracy; more "
                            "helps the retraining.")


# ---------------------------------------------------------------- image zip

def check_image_zip(path, m: ModelInfo, output="classes") -> DataReport:
    rep = DataReport()
    rep.facts.append(_model_summary(m))
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        rep.problems.append(f"{os.path.basename(path)} is not a zip file that can be read ({e}): zip the folder again.")
        return rep
    with z:
        infos = [i for i in z.infolist() if not i.filename.startswith("__MACOSX/")
                 and not os.path.basename(i.filename.rstrip("/")).startswith("._")]
        total = sum(i.file_size for i in infos)
        if len(infos) > MAX_ZIP_FILES or total > 4 * 500 * 1024 * 1024:
            rep.problems.append(f"The zip unpacks to {total / 2**20:.0f} MB in {len(infos):,} files: the service takes up to "
                                f"2,000 MB and {MAX_ZIP_FILES:,} files.")
        bad = _unsafe_entries(infos)
        if bad:
            rep.problems.append(bad)
        files = [i.filename.replace("\\", "/") for i in infos if not i.is_dir()]
    code = [f for f in files if f.lower().endswith(CODE_EXT)]
    if code:
        rep.problems.append(f"The zip holds {os.path.basename(code[0])}: the service refuses zips with files that can run code "
                            "when loaded (.pt, .h5, .pkl, ...). Remove them.")
    parts = [[p for p in f.split("/") if p] for f in files]
    parts = [p for p in parts if p and not p[-1].startswith(".")]
    # a zip of one folder: that folder is the data (the service does the same)
    tops = {p[0] for p in parts if len(p) > 1} | {p[0] for p in parts if len(p) == 1}
    while len(tops) == 1 and all(len(p) > 1 for p in parts):
        parts = [p[1:] for p in parts]
        tops = {p[0] for p in parts}
    imgs = [p for p in parts if p[-1].lower().endswith(IMAGE_EXT)]
    other = [p for p in parts if not p[-1].lower().endswith(IMAGE_EXT)]
    if not imgs:
        rep.problems.append("The zip holds no images (.jpg, .jpeg or .png). For images: a .zip of folders, one per class. "
                            "For anything else use an .npz with X and y.")
        return rep
    names = {p[0].lower(): p[0] for p in parts if len(p) > 1}
    split = "train" in names and any(s in names for s in ("test", "val", "validation"))
    n_classes = _classes_of(m)

    def classes_in(prefix):
        c = {}
        for p in imgs:
            q = p[len(prefix):]
            if len(q) >= 2 and [x.lower() for x in p[:len(prefix)]] == [x.lower() for x in prefix]:
                c[q[0]] = c.get(q[0], 0) + 1
        return c

    if split:
        tr = classes_in([names["train"]])
        tk = next(names[s] for s in ("test", "val", "validation") if s in names)
        te = classes_in([tk])
        rep.facts.append(f"Images: train/ {sum(tr.values()):,} in {len(tr)} classes, {tk}/ {sum(te.values()):,} in "
                         f"{len(te)} classes")
        if set(tr) != set(te):
            rep.problems.append(f"train/ and {tk}/ have different class folders: {sorted(set(tr) ^ set(te))[:10]}. Both need "
                                "the same class folders.")
        classes = tr
    else:
        classes = classes_in([])
        loose = [p for p in imgs if len(p) == 1]
        if not classes and loose:
            rep.facts.append(f"Images: {len(loose):,}, no class folders")
            rep.notes.append("No class folders: the original model's own answers become the labels, so accuracy means "
                             "agreement with your model. Put the images in one folder per class to measure real accuracy.")
            classes = {}
        else:
            rep.facts.append(f"Images: {len(imgs):,} in {len(classes)} class folders "
                             f"({', '.join(f'{k} {v:,}' for k, v in sorted(classes.items())[:8])}"
                             f"{', ...' if len(classes) > 8 else ''})")
            if loose:
                rep.warnings.append(f"{len(loose)} images sit outside the class folders and are not used.")
        rep.notes.append("Part of the images is held out as the test set. To fix the test set yourself, put train/ and test/ "
                         "folders in the zip, each with the class folders.")
    if classes:
        order = sorted(classes)
        rep.notes.append("Folder names in alphabetical order map to the model's outputs: "
                         + ", ".join(f"{i} = {c}" for i, c in enumerate(order[:12])) + (", ..." if len(order) > 12 else ""))
        if n_classes and output != "values" and len(classes) != n_classes:
            rep.problems.append(f"{len(classes)} class folders, but the model outputs {n_classes} scores: one folder per "
                                "output, in the model's output order.")
        if len(classes) == 1:
            rep.problems.append("Only one class folder: put each class's images in a folder of its own.")
        few = [k for k, v in classes.items() if v < 20]
        if few:
            rep.warnings.append(f"Few images in {', '.join(sorted(few)[:6])} (under 20): a few hundred per class are enough "
                                "to measure accuracy; more helps the retraining.")
    if main_in := (m.inputs[0] if m.inputs else None):
        s = main_in.shape
        chans = s[-1] if not m.channels_first else (s[1] if len(s) > 1 else None)
        if len(s) != 4 or chans not in (1, 3):
            rep.warnings.append(f"The model's input {_shape(s)} does not look like an image (N, H, W, C with 1 or 3 channels): "
                                "an image zip fits image models; other models take an .npz.")
        if m.format == "onnx" or (main_in.dtype.startswith("float")):
            rep.notes.append("If the model expects pixels scaled a particular way, set image_scaling ('0..1', '-1..1' or "
                             "'imagenet'); an int8 .tflite carries its own.")
    if other:
        rep.notes.append(f"{len(other)} files that are not images are ignored.")
    return rep


def _unsafe_entries(infos):
    for i in infos:
        n = i.filename.replace("\\", "/")
        if n.startswith("/") or ".." in n.split("/") or (len(n) > 1 and n[1] == ":"):
            return f"The zip holds a path outside its folder ({i.filename}): the service refuses it. Zip the folder again."
        if (i.external_attr >> 16) & 0o170000 == 0o120000:
            return f"The zip holds a symbolic link ({i.filename}): the service refuses it. Zip the files themselves."
    return None


def check(model_path, data_path, output="classes") -> DataReport:
    try:
        m = read_model(model_path)
    except (ModelReadError, OSError, zipfile.BadZipFile) as e:
        rep = DataReport()
        rep.problems.append(f"The model could not be read: {e}.")
        return rep
    if data_path.lower().endswith(".npz"):
        return check_npz(data_path, m, output)
    if data_path.lower().endswith(".zip"):
        return check_image_zip(data_path, m, output)
    rep = DataReport()
    rep.problems.append("Optimize takes data as an .npz (X, y) or a .zip of image folders, one per class.")
    return rep
