"""The local parts: reading model shapes, the data check, path and upload checks, and safe unpacking."""
import os
import stat
import sys
import zipfile

import numpy as np
import pytest

from sensorforge_mcp import datacheck, files
from sensorforge_mcp.api import Config, Service, ServiceError
from sensorforge_mcp.modelinfo import ModelReadError, read_model

HERE = os.path.dirname(__file__)
INT8 = os.path.join(HERE, "data", "tiny_int8.tflite")
FLOAT = os.path.join(HERE, "data", "tiny_float.tflite")
ONNX = os.path.join(HERE, "data", "tiny.onnx")


# ---------------------------------------------------------------- model files

def test_tflite_int8_shapes_and_quantization():
    m = read_model(INT8)
    assert m.format == "tflite" and not m.channels_first
    (i,), (o,) = m.inputs, m.outputs
    assert i.shape == [None, 49, 10, 1] and i.dtype == "int8" and o.shape == [None, 4] and o.dtype == "int8"
    lo, hi = i.float_range()
    assert lo == pytest.approx(-2.0, abs=0.02) and hi == pytest.approx(2.0, abs=0.02)


def test_tflite_float():
    m = read_model(FLOAT)
    assert m.inputs[0].dtype == "float32" and m.inputs[0].float_range() is None
    assert m.inputs[0].shape[1:] == [49, 10, 1] and m.outputs[0].shape[-1] == 4


def test_onnx_shapes_skip_initializers():
    m = read_model(ONNX)
    assert m.format == "onnx" and m.channels_first
    assert [(t.name, t.shape, t.dtype) for t in m.inputs] == [("imu", [1, 6, 100], "float32")]
    assert [(t.name, t.shape) for t in m.outputs] == [("logits", [1, 5])]


def test_firmware_zip_with_tflite_and_with_c_array(tmp_path):
    z1 = tmp_path / "export.zip"
    with zipfile.ZipFile(z1, "w") as z:
        z.write(INT8, "export/model.tflite")
        z.writestr("export/README.md", "hi")
    assert read_model(str(z1)).inputs[0].shape == [None, 49, 10, 1]
    data = open(INT8, "rb").read()
    cc = "const unsigned char g_model_data[] = {\n" + ", ".join(f"0x{b:02x}" for b in data) + "\n};\n"
    z2 = tmp_path / "fw.zip"
    with zipfile.ZipFile(z2, "w") as z:
        z.writestr("fw/src/model_data.cc", cc)
    assert read_model(str(z2)).outputs[0].shape == [None, 4]


def test_garbage_is_refused(tmp_path):
    p = tmp_path / "x.tflite"
    p.write_bytes(b"\x00" * 64)
    with pytest.raises(ModelReadError):
        read_model(str(p))
    q = tmp_path / "x.onnx"
    q.write_bytes(b"\xff\xff\xff")
    with pytest.raises(ModelReadError):
        read_model(str(q))


# ---------------------------------------------------------------- the data check

def npz(tmp_path, name="d.npz", **arrays):
    p = tmp_path / name
    np.savez(p, **arrays)
    return str(p)


def X(n=300, shape=(49, 10, 1), lo=-2, hi=2, dtype="float32"):
    return np.random.RandomState(0).uniform(lo, hi, (n,) + shape).astype(dtype)


def y(n=300, k=4):
    return np.arange(n) % k


def test_good_npz(tmp_path):
    r = datacheck.check(INT8, npz(tmp_path, X=X(), y=y()))
    assert r.ok and not r.warnings, r.text("d", "m")


def test_train_test_split_keys(tmp_path):
    r = datacheck.check(INT8, npz(tmp_path, X_train=X(), y_train=y(), X_test=X(100), y_test=y(100)))
    assert r.ok, r.problems


@pytest.mark.parametrize("arrays,expect", [
    (dict(features=X()), "No X array"),
    (dict(X=X(shape=(40, 10, 1)), y=y()), "does not fit the model's input"),
    (dict(X=X(), y=y(k=9)), "the model outputs 4 scores"),
    (dict(X=X(), y=y(299)), "one label per sample"),
    (dict(X=X(), y=-y() - 1), "negative labels"),
    (dict(X=X(), y=np.eye(4)[y()] * 0.9), "y.argmax(axis=1)"),
    (dict(X=X(), y=y(), X_test=X(50)), "X_test has no y_test"),
    (dict(X=X().reshape(300, 490)[:, :10], y=y()), "does not fit"),
    (dict(X=np.where(np.arange(300 * 490).reshape(300, 49, 10, 1) == 7, np.nan, 0).astype("float32"), y=y()), "NaN"),
    (dict(X=np.full((300, 49, 10, 1), "a"), y=y()), "save the model's inputs as float32"),
])
def test_npz_problems(tmp_path, arrays, expect):
    r = datacheck.check(INT8, npz(tmp_path, **arrays))
    assert not r.ok and any(expect in p for p in r.problems), r.problems


def test_object_arrays(tmp_path):
    p = tmp_path / "o.npz"
    np.savez(p, X=np.array([np.zeros(3), np.zeros(4)], dtype=object), y=y(2))
    r = datacheck.check(INT8, str(p))
    assert not r.ok and "Python objects" in r.problems[0]


def test_npz_warnings_and_notes(tmp_path):
    r = datacheck.check(INT8, npz(tmp_path, X=X(dtype="float64"), y=y().astype("float32")))
    assert r.ok
    assert any("float64" in w for w in r.warnings) and any("int64" in w for w in r.warnings)
    r = datacheck.check(INT8, npz(tmp_path, X=X(lo=0, hi=255)))              # pixels never scaled
    assert any("Most of X lies between" in w for w in r.warnings), r.warnings
    assert any("No y" in n for n in r.notes)
    r = datacheck.check(INT8, npz(tmp_path, X=X().reshape(300, 490), y=y()))
    assert r.ok and any("the service reshapes them" in n for n in r.notes)
    r = datacheck.check(INT8, npz(tmp_path, X=X(30), y=y(30)))
    assert any("a few hundred per class" in w for w in r.warnings)
    r = datacheck.check(INT8, npz(tmp_path, X=X(), y=np.zeros(300, int)))
    assert any("No samples of classes 1, 2, 3" in w for w in r.warnings)


def test_onnx_layout_hint(tmp_path):
    r = datacheck.check(ONNX, npz(tmp_path, X=X(shape=(100, 6)), y=y(k=5)))
    assert not r.ok and any("channels first" in p and "np.moveaxis(X, -1, 1)" in p for p in r.problems), r.problems
    r = datacheck.check(ONNX, npz(tmp_path, X=X(shape=(6, 100)), y=y(k=5)))
    assert r.ok


def test_one_hot_labels(tmp_path):
    r = datacheck.check(INT8, npz(tmp_path, X=X(), y=np.eye(4)[y()]))
    assert r.ok and any("one-hot" in w and "argmax" in w for w in r.warnings)
    r = datacheck.check(INT8, npz(tmp_path, X=X(), y=np.eye(9)[y(k=9)]))
    assert not r.ok and any("9 columns, but the model outputs 4" in p for p in r.problems)


def test_multilabel_and_values(tmp_path):
    ml = (np.random.RandomState(1).rand(300, 4) > 0.5).astype("float32")
    assert datacheck.check(INT8, npz(tmp_path, X=X(), y=ml)).ok
    r = datacheck.check(INT8, npz(tmp_path, X=X(), y=np.random.rand(300, 4).astype("float32")), output="values")
    assert r.ok and any("not needed" in n for n in r.notes)


def image_zip(tmp_path, entries, name="imgs.zip"):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as z:
        for n in entries:
            z.writestr(n, b"\xff\xd8\xff" if n.endswith((".jpg", ".png")) else b"x")
    return str(p)


def test_image_zips(tmp_path):
    cls = lambda c, n=25, pre="": [f"{pre}{c}/{i}.jpg" for i in range(n)]
    four = sum((cls(c) for c in "abcd"), [])
    r = datacheck.check(INT8, image_zip(tmp_path, four + ["__MACOSX/a/._0.jpg"]))
    assert r.ok and any("0 = a, 1 = b, 2 = c, 3 = d" in n for n in r.notes), r.text("z", "m")
    r = datacheck.check(INT8, image_zip(tmp_path, [f"root/{x}" for x in four]))            # one wrapping folder
    assert r.ok
    split = sum((cls(c, pre="train/") + cls(c, 5, pre="test/") for c in "abcd"), [])
    r = datacheck.check(INT8, image_zip(tmp_path, split))
    assert r.ok and any("train/" in f for f in r.facts)
    r = datacheck.check(INT8, image_zip(tmp_path, sum((cls(c) for c in "abc"), [])))
    assert not r.ok and any("3 class folders, but the model outputs 4" in p for p in r.problems)
    r = datacheck.check(INT8, image_zip(tmp_path, four + ["a/model.pt"]))
    assert not r.ok and any("model.pt" in p for p in r.problems)
    r = datacheck.check(INT8, image_zip(tmp_path, ["notes.txt"]))
    assert not r.ok and "no images" in r.problems[0]
    mism = sum((cls(c, pre="train/") for c in "abcd"), []) + cls("a", pre="test/")
    r = datacheck.check(INT8, image_zip(tmp_path, mism))
    assert not r.ok and any("different class folders" in p for p in r.problems)


def test_streaming_model_note(tmp_path):
    class T:   # a model with state inputs, as the reader returns it
        pass
    from sensorforge_mcp.modelinfo import ModelInfo, Tensor
    m = ModelInfo("tflite", [Tensor("state", [1, 64], "int8"), Tensor("frame", [1, 1, 10], "int8")],
                  [Tensor("o", [1, 12], "int8")])
    r = datacheck.check_npz(npz(tmp_path, X=X(shape=(49, 10)), y=y()), m)
    assert r.ok and any("streaming" in n for n in r.notes)
    r = datacheck.check_npz(npz(tmp_path, X=X(shape=(49, 7))), m)
    assert not r.ok


# ---------------------------------------------------------------- upload checks

def test_check_model_and_data_files(tmp_path):
    with pytest.raises(ServiceError, match="is a folder"):
        files.check_model(str(tmp_path))
    p = tmp_path / "m.onnx"
    p.write_bytes(b"")
    with pytest.raises(ServiceError, match="empty"):
        files.check_model(str(p))
    z = tmp_path / "fw.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("fw/readme.txt", "no model")
    with pytest.raises(ServiceError, match="holds no .tflite"):
        files.check_model(str(z))
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("fw/model.tflite", "x")
        f.writestr("fw/weights.h5", "x")
    with pytest.raises(ServiceError, match="can run code"):
        files.check_model(str(z))
    d = tmp_path / "data.csv"
    d.write_text("1,2")
    with pytest.raises(ServiceError, match=r"\.npz \(X, y\)"):
        files.check_data_file(str(d))
    big = tmp_path / "big.npz"
    with open(big, "wb") as f:
        f.truncate(500 * 1024 * 1024 + 1)
    with pytest.raises(ServiceError, match="up to 500 MB"):
        files.check_data_file(str(big))
    with pytest.raises(ServiceError, match="Zip it first"):
        files.check_data_file(str(tmp_path))


# ---------------------------------------------------------------- unpacking

def make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as z:
        for name, data, mode in entries:
            i = zipfile.ZipInfo(name)
            if mode is not None:
                i.external_attr = mode << 16
            z.writestr(i, data)


@pytest.mark.parametrize("bad", ["../x", "/etc/x", "a/../../x", "..\\x", "C:/x", "c:\\windows\\x"])
def test_unpack_refuses_paths_outside(tmp_path, bad):
    z = tmp_path / "d.zip"
    make_zip(z, [("ok.txt", b"1", None), (bad, b"2", None)])
    dest = tmp_path / "out"
    with pytest.raises(ServiceError, match="nothing was unpacked"):
        files.unpack(str(z), str(dest))
    assert not dest.exists() or not any(dest.iterdir())


def test_unpack_refuses_links_and_devices(tmp_path):
    for mode in (stat.S_IFLNK | 0o777, stat.S_IFCHR | 0o600, stat.S_IFIFO | 0o600):
        z = tmp_path / "d.zip"
        make_zip(z, [("ok.txt", b"1", None), ("c/l", b"/etc/passwd", mode)])
        with pytest.raises(ServiceError, match="link or special file"):
            files.unpack(str(z), str(tmp_path / "out"))


@pytest.mark.skipif(sys.platform == "win32", reason="creating links needs privileges on Windows")
def test_unpack_does_not_follow_a_link_already_in_the_folder(tmp_path):
    dest = tmp_path / "out"
    (dest / "c").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, dest / "c" / "sub")
    z = tmp_path / "d.zip"
    make_zip(z, [("c/sub/x.txt", b"1", None)])
    with pytest.raises(ServiceError, match="outside the folder"):
        files.unpack(str(z), str(dest), overwrite=True)
    assert not (outside / "x.txt").exists()


def test_unpack_plain_files_no_exec_and_overwrite(tmp_path):
    z = tmp_path / "d.zip"
    make_zip(z, [("c/", b"", stat.S_IFDIR | 0o755), ("c/run.sh", b"#!/bin/sh", stat.S_IFREG | 0o755), ("m.tflite", b"m", None)])
    dest = tmp_path / "out"
    assert files.unpack(str(z), str(dest)) == ["c/run.sh", "m.tflite"]
    assert not os.stat(dest / "c" / "run.sh").st_mode & 0o111
    with pytest.raises(ServiceError, match="already exists"):
        files.unpack(str(z), str(dest))
    assert files.unpack(str(z), str(dest), overwrite=True)


def test_unpack_size_limits(tmp_path, monkeypatch):
    z = tmp_path / "d.zip"
    make_zip(z, [("a.bin", b"\0" * 5000, None)])
    monkeypatch.setattr(files, "MAX_UNPACK_BYTES", 4000)
    with pytest.raises(ServiceError, match="more files or bytes"):
        files.unpack(str(z), str(tmp_path / "out"))


def test_describe():
    assert "deploy" in files.describe("optimized_tflm.tflite")
    assert "PC" in files.describe("optimized_litert.tflite") and "PC" in files.describe("kws_lossless_litert.tflite")
    assert "identical" in files.describe("kws_lossless.tflite")
    assert "C array" in files.describe("c/model_data.cc")
    assert "op resolver" in files.describe("c/model_ops.h")
    assert "report" in files.describe("report.md")


# ---------------------------------------------------------------- the API client

def test_error_sentences():
    s = Service(Config(url="https://sensorforge.dev/optimize", key="sfk_x"))
    assert "API keys" in str(s.error(401, "Please sign in."))
    assert "paid plans" in str(s.error(402, "Downloading ... is part of the paid plans")) and "plans on" in str(s.error(402, ""))
    assert str(s.error(403, "Only admins.")).startswith("Not allowed: Only admins.")
    assert "No such run" in str(s.error(404, "no such job", "run"))
    assert "model 50 MB, data 500 MB" in str(s.error(413, "big"))
    assert ".tflite" in str(s.error(415, ""))
    assert "did not accept the input: unknown option." in str(s.error(422, "unknown option"))
    assert "account()" in str(s.error(429, "Today's runs are used."))
    assert "Try again in a few minutes" in str(s.error(503, "short of disk"))
    assert "redirect" in str(s.error(307, ""))
    assert "HTTP 500" in str(s.error(500, ""))


def test_config_from_env():
    c = Config.from_env({"SENSORFORGE_URL": "http://localhost:8000/optimize/", "SENSORFORGE_API_KEY": " sfk_abc ",
                         "SENSORFORGE_DEV_USER": "me"})
    assert c.url == "http://localhost:8000/optimize" and c.key == "sfk_abc" and c.dev and c.is_local
    c = Config.from_env({})
    assert c.url == "https://sensorforge.dev/optimize" and c.key is None
    with pytest.raises(ServiceError, match="SENSORFORGE_API_KEY"):
        c.check()
    with pytest.raises(ServiceError, match="not a web address"):
        Config(url="sensorforge.dev", key="k").check()
