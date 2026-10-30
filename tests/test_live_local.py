"""End to end against a local development copy of the service that signs callers in by X-Dev-User headers (no keys):

    SENSORFORGE_E2E_URL=http://127.0.0.1:8765/optimize SENSORFORGE_E2E_MODEL_DIR=<folder with model.tflite and data.npz> \
        pytest -m live -s tests/test_live_local.py

The model and data are copied into a temporary project so the downloads land there. A quick optimize takes 10-15 minutes.
"""
import os
import shutil
import time
import zipfile

import pytest
from mcp import Client

from sensorforge_mcp.api import Config, Service
from sensorforge_mcp.server import RunBook, create_server

URL = os.environ.get("SENSORFORGE_E2E_URL")
SRC = os.path.expanduser(os.environ.get("SENSORFORGE_E2E_MODEL_DIR", ""))
LOG = os.environ.get("SENSORFORGE_E2E_LOG")

pytestmark = [pytest.mark.live, pytest.mark.anyio, pytest.mark.skipif(not (URL and SRC), reason="SENSORFORGE_E2E_URL and SENSORFORGE_E2E_MODEL_DIR not set")]


def out(title, text):
    block = f"\n===== {title}\n{text}\n"
    print(block, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(block)


def text_of(r):
    return "\n".join(c.text for c in r.content if getattr(c, "type", "") == "text")


@pytest.fixture
def project(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    for n in ("model.tflite", "data.npz"):
        shutil.copy(os.path.join(SRC, n), d / n)
    return d


async def test_end_to_end(project, tmp_path):
    svc = Service(Config(url=URL.rstrip("/"), dev_user=os.environ.get("SENSORFORGE_E2E_USER", "mcp-e2e")))
    server = create_server(svc, RunBook(str(tmp_path / "home")))
    model, data = str(project / "model.tflite"), str(project / "data.npz")
    progress = []

    async def on_progress(p, total, msg):
        progress.append((p, total, msg))

    async with Client(server, read_timeout_seconds=900) as c:
        t0 = time.time()
        r = await c.call_tool("list_chips", {})
        out("list_chips", text_of(r))
        assert not r.is_error and "nrf54l15" in text_of(r)

        r = await c.call_tool("account", {})
        out("account", text_of(r))
        assert not r.is_error

        r = await c.call_tool("check_fit", {"model_path": model, "chips": ["nrf54l15", "riscv-512k"]},
                              progress_callback=on_progress)
        out(f"check_fit ({time.time() - t0:.0f} s)", text_of(r))
        assert not r.is_error and "| nrf54l15 |" in text_of(r) and "| riscv-512k |" in text_of(r)

        t1 = time.time()
        r = await c.call_tool("lossless", {"model_path": model, "chip": "nrf54l15"}, progress_callback=on_progress)
        out(f"lossless ({time.time() - t1:.0f} s)", text_of(r))
        assert not r.is_error and "Verdict: PASS" in text_of(r)
        ll = [d for d in (project / "sensorforge").iterdir()]
        assert len(ll) == 1
        got = sorted(p.relative_to(ll[0]).as_posix() for p in ll[0].rglob("*") if p.is_file())
        out("lossless files", "\n".join(got))
        for need in ("model_lossless.tflite", "c/model_data.cc", "c/model_data.h", "c/model_ops.h", "c/INTEGRATION.md",
                     "lossless.md"):
            assert need in got, need
        assert (ll[0] / "model_lossless.tflite").read_bytes()[4:8] == b"TFL3"

        r = await c.call_tool("check_data", {"model_path": model, "data_path": data})
        out("check_data", text_of(r))
        assert not r.is_error and "Result: ready for optimize" in text_of(r)

        t2 = time.time()
        r = await c.call_tool("optimize", {"model_path": model, "data_path": data, "chip": "nrf54l15",
                                           "minimize": "ram", "accuracy_limit": 1.0, "effort": "quick"},
                              progress_callback=on_progress)
        out("optimize", text_of(r))
        assert not r.is_error and "Submitted optimize run" in text_of(r)
        rid = text_of(r).split("Submitted optimize run ", 1)[1].split(":", 1)[0]

        r = await c.call_tool("run_status", {"run_id": rid})
        out("run_status (just after submitting)", text_of(r))

        calls = 0
        while True:
            calls += 1
            r = await c.call_tool("wait_for_run", {"run_id": rid, "max_wait_s": 300}, progress_callback=on_progress)
            t = text_of(r)
            if "Still running" not in t:
                break
            out(f"wait_for_run #{calls} ({time.time() - t2:.0f} s)", t)
            assert calls < 12, "optimize took over an hour"
        out(f"wait_for_run #{calls}, final ({time.time() - t2:.0f} s)", t)
        assert not r.is_error and "Verdict: PASS" in t, t

        r = await c.call_tool("download", {"run_id": rid})
        out("download", text_of(r))
        assert not r.is_error
        dest = project / "sensorforge" / rid
        got = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
        for need in ("optimized_tflm.tflite", "c/model_data.cc", "c/model_data.h", "c/model_ops.h", "c/INTEGRATION.md",
                     "report.md"):
            assert need in got, need
        assert (dest / "optimized_tflm.tflite").read_bytes()[4:8] == b"TFL3"
        assert not any(os.access(dest / g, os.X_OK) for g in got)

        # the unpacked folder holds everything the service's own zip holds
        zp = tmp_path / "direct.zip"
        await svc.download(rid, str(zp))
        with zipfile.ZipFile(zp) as z:
            names = sorted(n for n in z.namelist() if not n.endswith("/"))
        assert names == got, (names, got)

        r = await c.call_tool("download", {"run_id": rid})
        assert r.is_error and "not empty" in text_of(r)

        r = await c.call_tool("get_report", {"run_id": rid})
        out("get_report (first 1,500 characters)", text_of(r)[:1500])
        assert not r.is_error and len(text_of(r)) > 200

        r = await c.call_tool("list_runs", {"limit": 10})
        out("list_runs", text_of(r))
        assert rid in text_of(r)

        r = await c.call_tool("optimize", {"model_path": model, "data_path": data, "chip": "riscv-512k"})
        rid2 = text_of(r).split("Submitted optimize run ", 1)[1].split(":", 1)[0]
        r = await c.call_tool("stop_run", {"run_id": rid2})
        out("stop_run", text_of(r))
        assert not r.is_error and "Stopped" in text_of(r)
        r = await c.call_tool("run_status", {"run_id": rid2})
        out("run_status (stopped run)", text_of(r))
        assert "stopped" in text_of(r)

        r = await c.call_tool("account", {})
        out("account (after)", text_of(r))
        out("progress notifications", f"{len(progress)} received; last: {progress[-1] if progress else None}")
        assert progress
