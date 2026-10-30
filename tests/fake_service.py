"""A small stand-in for the hosted service, built from its API contract and the live code's shapes.

Accounts are API keys:  sfk_team (Team plan), sfk_trial (Trial: no downloads), sfk_capped (today's cap reached),
sfk_pending (waiting for approval). Anything else is a 401.
A job's behaviour follows its model's file name: slow* runs for several polls, fail* fails, reject* is REJECTED,
evil* downloads a zip with a ../ path, link* a zip with a symbolic link, bare* a bare .tflite.
"""
import io
import json
import socket
import stat
import threading
import time
import uuid
import zipfile

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route

TFLITE = b"\x1c\x00\x00\x00TFL3" + b"\x00" * 64
TARGETS = [
    dict(name="nrf54l15", description="Nordic nRF54L15 (Cortex-M33 @128 MHz). TFLite Micro + CMSIS-NN.", ram_kb=150.0,
         flash_kb=1000.0, npu=False, runtime="tflite-micro"),
    dict(name="riscv-512k", description="Generic 32-bit RISC-V MCU, 512 KB SRAM. Reference kernels.", ram_kb=384.0,
         flash_kb=1536.0, npu=False, runtime="tflite-micro"),
    dict(name="cortex-m-cmsis", description="Generic Arm Cortex-M with TFLite Micro + CMSIS-NN. Set its budgets.",
         ram_kb=None, flash_kb=None, npu=False, runtime="tflite-micro"),
]
PLANS = {"sfk_team": "team", "sfk_trial": "trial", "sfk_capped": "team", "sfk_pending": "pending"}


class Fake:
    def __init__(self):
        self.jobs = {}
        self.calls = []              # (method, path, headers)
        self.uploads = []            # what each POST /api/jobs carried
        self.max_model = 50 * 1024 * 1024
        self.slow_polls = 4
        self.flaky = 0                   # the next n run-status calls answer 503 (a deploy in progress)
        self.trial_runs_left = 7
        self.app = Starlette(routes=[Mount("/optimize", routes=[
            Route("/api/targets", self.targets),
            Route("/api/me", self.me),
            Route("/api/jobs", self.jobs_route, methods=["GET", "POST"]),
            Route("/api/jobs/{jid}", self.job),
            Route("/api/jobs/{jid}/stop", self.stop, methods=["POST"]),
            Route("/api/jobs/{jid}/report", self.report),
            Route("/api/jobs/{jid}/download", self.download),
        ])])

    # ------------------------------------------------------------ helpers

    def who(self, request):
        h = request.headers.get("authorization", "")
        key = h[7:] if h.lower().startswith("bearer ") else None
        return key if key in PLANS else None

    def record(self, request):
        self.calls.append((request.method, request.url.path, dict(request.headers)))

    @staticmethod
    def err(status, msg):
        return JSONResponse({"error": msg}, status_code=status)

    def own(self, request):
        key = self.who(request)
        if not key:
            return None, self.err(401, "Please sign in.")
        J = self.jobs.get(request.path_params["jid"])
        if not J or J["key"] != key:
            return None, self.err(404, "no such job")
        return J, None

    # ------------------------------------------------------------ routes

    async def targets(self, request):
        self.record(request)
        return JSONResponse(TARGETS)

    async def me(self, request):
        self.record(request)
        key = self.who(request)
        if not key:
            return self.err(401, "Please sign in.")
        plan = PLANS[key]
        d = dict(org=dict(id=f"user:{key}", name=f"user:{key}", plan=plan, seats=1),
                 allowance=dict(modes=["quick"] if plan == "trial" else ([] if plan == "pending" else ["quick", "thorough"]),
                                download=plan == "team", concurrency=2),
                 month=dict(runs=len(self.jobs), quick=1, thorough=0, cpu_hours=0.1, thorough_cap=50 if plan == "team" else None),
                 billing=dict(status=None, customer=False), admin=False)
        if plan == "trial":
            d["trial"] = dict(runs_left=self.trial_runs_left, days_left=12)
        caps = dict(runs=20, optimize=10, thorough=2)
        used = dict(runs=20, optimize=10, thorough=0) if key == "sfk_capped" else dict(runs=3, optimize=1, thorough=0)
        d["day"] = dict(**used, caps=caps, next_free_at=time.time() + 3600 if key == "sfk_capped" else None)
        return JSONResponse(d)

    async def jobs_route(self, request):
        self.record(request)
        key = self.who(request)
        if not key:
            return self.err(401, "Please sign in.")
        if request.method == "GET":
            rows = [j for j in self.jobs.values() if j["key"] == key]
            rows.sort(key=lambda j: -j["created"])
            n = int(request.query_params.get("limit", "50"))
            return JSONResponse([dict(id=j["id"], kind=j["kind"], mode=j["mode"], status=j["status"], model_name=j["model"],
                                      data_name=j["data"], created=j["created"], started=j["started"], finished=j["finished"],
                                      verdict=j["verdict"], wall_seconds=1.5, error=j["error"]) for j in rows[:n]])
        if PLANS[key] == "pending":
            return self.err(403, "Your access request is waiting for approval.")
        if key == "sfk_capped":
            return self.err(429, "Today's runs are used (20 in the last 24 hours).")
        form = await request.form(max_files=2, max_fields=16, max_part_size=4096)
        model, data = form.get("model"), form.get("data")
        fields = {k: v for k, v in form.items() if isinstance(v, str)}
        sizes = {}
        for name, f in (("model", model), ("data", data)):
            if f is not None and not isinstance(f, str):
                body = await f.read()
                sizes[name] = len(body)
        self.uploads.append(dict(fields=fields, sizes=sizes, names={k: getattr(v, "filename", None) for k, v in
                                                                   (("model", model), ("data", data)) if v is not None},
                                 client=request.headers.get("x-sf-client")))
        if model is None or "kind" not in fields or "target" not in fields:
            return self.err(422, "a job needs a model file, a kind and a chip target")
        if sizes.get("model", 0) > self.max_model:
            return self.err(413, f"{model.filename}: larger than {self.max_model // (1024 * 1024)} MB")
        if not model.filename.lower().endswith((".tflite", ".onnx", ".zip")):
            return self.err(415, "Upload a .tflite or .onnx model, or a .zip of a firmware export folder.")
        if fields["kind"] == "optimize" and data is None:
            return self.err(422, "optimize needs data")
        if fields["target"] not in {t["name"] for t in TARGETS}:
            return self.err(422, f"unknown chip target {fields['target']!r}")
        jid = uuid.uuid4().hex[:16]
        self.jobs[jid] = dict(id=jid, key=key, kind=fields["kind"], mode=fields.get("effort") if fields["kind"] == "optimize" else None,
                              status="queued", model=model.filename, data=getattr(data, "filename", None),
                              params=dict(fields), created=time.time(), started=None, finished=None, verdict=None,
                              error=None, polls=0)
        return JSONResponse(dict(id=jid, status="queued", position=1))

    def advance(self, J):
        J["polls"] += 1
        m = J["model"]
        if J["status"] in ("done", "failed", "stopped"):
            return
        if J["polls"] == 1:
            J.update(status="running", started=time.time())
            return
        if m.startswith("slow") and J["polls"] < self.slow_polls:
            return
        J["finished"] = time.time()
        if m.startswith("fail"):
            J.update(status="failed", error="the model could not be read: no subgraph")
            return
        J["status"] = "done"
        if J["kind"] == "fit":
            J["verdict"] = "runs as is"
        elif m.startswith("reject"):
            J["verdict"] = "REJECTED: no plan met the accuracy limit"
        else:
            J["verdict"] = "PASS"

    def summary(self, J):
        if J["status"] != "done":
            return {}
        if J["kind"] == "fit":
            return dict(verdict=J["verdict"], rows=[dict(target=J["params"]["target"], verdict="runs as is", arena_bytes=24640,
                                                         file_bytes=47616, lossless=20800, ram_build=None)],
                        ops=["CONV_2D"], advice=["A note from the service."], ok=True)
        if J["kind"] == "lossless":
            return dict(verdict="PASS", before=dict(ram=24640, flash=47616, label="your file"),
                        after=dict(ram=20800, flash=33928), ram_build=None, steps=["a step"],
                        changes=["Inputs and outputs: the same as your file's."], warnings=[], advice=[], ok=True)
        ok = J["verdict"] == "PASS"
        return dict(verdict=J["verdict"], before=dict(ram=24640, flash=47616, accuracy=0.9310, label="your file"),
                    after=dict(ram=14500 if ok else None, flash=14200 if ok else None, accuracy=0.9285 if ok else None),
                    ram_measured=True, ram_build=None, task="classification", accuracy_runtime="TensorFlow Lite Micro",
                    accuracy_samples=4874, warnings=["w1", "w2", "w3", "w4"], advice=[] if ok else ["Raise the accuracy limit to 2 points."],
                    plan=None, changes=["Weights at 4 bits in two layers."] if ok else [],
                    refusal=None if ok else dict(advice="The smallest model within 1 point needs 30 KB."), ok=ok)

    async def job(self, request):
        self.record(request)
        if self.flaky > 0:
            self.flaky -= 1
            return Response("upstream restarting", status_code=503)
        J, e = self.own(request)
        if e:
            return e
        self.advance(J)
        plan = PLANS[J["key"]]
        s = self.summary(J)
        return JSONResponse(dict(
            id=J["id"], kind=J["kind"], mode=J["mode"], status=J["status"], position=1 if J["status"] == "queued" else None,
            model=J["model"], data=J["data"], created=J["created"], started=J["started"], finished=J["finished"],
            verdict=J["verdict"], error=J["error"], log=["step one", "| a | table |", "search: step 3 of 10"],
            usage=dict(cpu_seconds=1.0, wall_seconds=1.5, peak_rss_mb=100.0), params=J["params"], summary=s,
            has_report=J["status"] == "done", has_pdf=False,
            has_download=J["status"] == "done" and J["kind"] != "fit" and J["verdict"] == "PASS",
            may_download=plan == "team"))

    async def stop(self, request):
        self.record(request)
        J, e = self.own(request)
        if e:
            return e
        if J["status"] in ("queued", "running"):
            J["status"] = "stopped"
            return JSONResponse(dict(stopped=True))
        return JSONResponse(dict(stopped=False))

    async def report(self, request):
        self.record(request)
        J, e = self.own(request)
        if e:
            return e
        if J["status"] != "done":
            return self.err(404, "no report for this job")
        return PlainTextResponse(f"# Report for {J['id']}\n\n**{J['verdict']}**\n", media_type="text/markdown")

    async def download(self, request):
        self.record(request)
        J, e = self.own(request)
        if e:
            return e
        if PLANS[J["key"]] != "team":
            return self.err(402, "Downloading the optimized model is part of the paid plans; the trial shows the full report.")
        if J["status"] != "done" or J["kind"] == "fit" or J["verdict"] != "PASS":
            return self.err(404, "this job produced no file to download")
        m = J["model"]
        if m.startswith("bare"):
            return Response(TFLITE, media_type="application/octet-stream",
                            headers={"content-disposition": 'attachment; filename="bare_model_lossless.tflite"'})
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            name = "model_lossless.tflite" if J["kind"] == "lossless" else "optimized_tflm.tflite"
            z.writestr(name, TFLITE)
            if J["kind"] == "optimize":
                z.writestr("optimized_litert.tflite", TFLITE)
            z.writestr("c/model_data.cc", "const unsigned char g_model_data[] = {0x1c};\n")
            z.writestr("c/model_data.h", "#pragma once\n")
            z.writestr("c/model_ops.h", "#pragma once\n")
            z.writestr("c/INTEGRATION.md", "# Integration\n")
            z.writestr("lossless.md" if J["kind"] == "lossless" else "report.md", "# Report\n")
            exe = zipfile.ZipInfo("c/run.sh")
            exe.external_attr = (stat.S_IFREG | 0o755) << 16
            z.writestr(exe, "#!/bin/sh\necho never run\n")
            if m.startswith("evil"):
                z.writestr("../evil.txt", "outside")
            if m.startswith("link"):
                li = zipfile.ZipInfo("c/link")
                li.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(li, "/etc/passwd")
        base = m.rsplit(".", 1)[0]
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"content-disposition": f'attachment; filename="{base}_{J["kind"]}.zip"'})


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Running:
    """The fake on a real localhost port, in a thread."""

    def __init__(self):
        self.fake = Fake()
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}/optimize"
        self.server = uvicorn.Server(uvicorn.Config(self.fake.app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        t0 = time.time()
        while not self.server.started:
            if time.time() - t0 > 10:
                raise RuntimeError("fake service did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(5)


if __name__ == "__main__":
    with Running() as r:
        print(json.dumps(dict(url=r.url)), flush=True)
        while True:
            time.sleep(1)
