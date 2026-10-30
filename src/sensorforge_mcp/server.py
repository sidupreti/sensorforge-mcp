"""The MCP server: tools that fit a model to an edge device with the hosted SensorForge Optimizer.

Runs locally over stdio. Reads the model and data from disk, calls the service with the API key, and unpacks the
results into the project. The optimizer runs on the service; nothing here optimizes anything.
"""
import difflib
import functools
import json
import logging
import os
import re
import shutil
import tempfile
import time
from typing import Annotated, Literal

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import __version__, datacheck, files, fmt
from .api import MAX_DATA_BYTES, CountingFile, Service, ServiceError

INSTRUCTIONS = """SensorForge Optimizer: we shrink trained models to fit edge devices. Upload a .tflite or .onnx model; get back
a TensorFlow Lite Micro model (and its C sources) that needs less SRAM and flash, within the accuracy limit you set.
Every figure is measured by running the file.

Typical flow: list_chips -> check_fit (seconds) -> lossless (under a minute, no data, outputs identical) -> check_data
(local) -> optimize (5-60 minutes, needs data) -> wait_for_run (call again while it says still running) -> download.
Paths may be relative to the working directory or start with ~. Needs SENSORFORGE_API_KEY."""

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)
LOCAL_READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
SUBMITS = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True)
WRITES = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True)

FIT_WAIT_S = 240          # check_fit and lossless wait this long before handing back the run id
POLL_S = 10
MAX_WAIT_S = 600
MAX_CHIPS = 8
DEFAULT_CHIP = "cortex-m-cmsis"     # check_fit without chips: a generic Arm Cortex-M
REPORT_MAX_CHARS = 40_000
NORM = {"auto": None, "0..1": "zero_one", "-1..1": "minus_one_one", "imagenet": "imagenet"}


class RunBook:
    """Which local model each run came from, so download() can unpack next to it. A small JSON file in
    SENSORFORGE_MCP_HOME (default ~/.sensorforge-mcp); runs older than the service keeps them (14 days) drop out."""

    def __init__(self, home=None):
        self.home = home or os.environ.get("SENSORFORGE_MCP_HOME") or os.path.expanduser("~/.sensorforge-mcp")
        self.path = os.path.join(self.home, "runs.json")

    def _load(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def add(self, run_id, model_path, kind):
        d = self._load()
        cutoff = time.time() - 15 * 86400
        d = {k: v for k, v in d.items() if isinstance(v, dict) and v.get("at", 0) > cutoff}
        d[run_id] = dict(model=model_path, kind=kind, at=time.time())
        try:
            os.makedirs(self.home, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(d, f)
            os.replace(tmp, self.path)
        except OSError:
            pass                    # only a convenience: download() then defaults to the working directory

    def model_of(self, run_id):
        v = self._load().get(run_id)
        return v.get("model") if isinstance(v, dict) else None


def _tool_errors(fn):
    """A ServiceError (or a bad path) is the agent's to read: its sentence goes back as the tool's error."""
    @functools.wraps(fn)
    async def wrapped(*a, **kw):
        try:
            return await fn(*a, **kw)
        except ToolError:
            raise
        except ServiceError as e:
            raise ToolError(str(e)) from e
        except OSError as e:
            raise ToolError(f"{e.strerror or type(e).__name__}: {e.filename or e}") from e
        except Exception as e:      # a bug here: its type and message, so it can be reported
            raise ToolError(f"Unexpected error in sensorforge-mcp {__version__} ({type(e).__name__}: {fmt.clip(e, 300)}). "
                            "Please report it.") from e
    return wrapped


async def _progress(ctx, done, total, message):
    if ctx is None:
        return
    try:
        await ctx.report_progress(done, total, message)
    except Exception:
        pass                        # a client without progress support


def create_server(service: Service | None = None, runbook: RunBook | None = None, poll_s: float = POLL_S) -> MCPServer:
    svc = service or Service()
    book = runbook or RunBook()
    cache = {"targets": None, "at": 0.0}
    level = (os.environ.get("SENSORFORGE_LOG") or "WARNING").upper()
    mcp = MCPServer(name="sensorforge", title="SensorForge Optimizer", version=__version__, instructions=INSTRUCTIONS,
                    website_url="https://sensorforge.dev/optimize",
                    log_level=level if level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL") else "WARNING")
    for name in ("httpx", "httpcore"):          # stderr is the client's log: one line per request is noise there
        logging.getLogger(name).setLevel(logging.WARNING)

    async def targets(fresh=False):
        if fresh or cache["targets"] is None or time.time() - cache["at"] > 3600:
            cache["targets"] = await svc.targets()
            cache["at"] = time.time()
        return cache["targets"]

    async def check_chip(chip):
        chip = (chip or "").strip()
        names = [t["name"] for t in await targets()]
        if chip in names:
            return chip
        low = {n.lower(): n for n in names}
        if chip.lower() in low:
            return low[chip.lower()]
        near = difflib.get_close_matches(chip.lower(), list(low), n=3, cutoff=0.5)
        hint = f" Did you mean {', '.join(low[n] for n in near)}?" if near else ""
        raise ServiceError(f"Unknown chip {chip!r}.{hint} list_chips shows the chips: " + ", ".join(names) + ".")

    async def preflight(kind, effort=None, runs=1):
        """Before a large upload: the key works, and the plan has a run left today (the service may turn an upload
        away only after it has started)."""
        try:
            me = await svc.me()
        except ServiceError as e:
            if e.status in (401, 403):
                raise
            return None
        org = me.get("org") or {}
        if org.get("plan") == "pending":
            raise ServiceError("This account is waiting for approval and cannot run models yet; you will get an email when "
                               "it can.", 403)
        trial = me.get("trial") or {}
        if trial and trial.get("runs_left") == 0:
            raise ServiceError("The trial's runs are all used. Choose a plan on https://sensorforge.dev/optimize/app.", 402)
        if trial and isinstance(trial.get("runs_left"), int) and trial["runs_left"] < runs:
            raise ServiceError(f"The trial has {trial['runs_left']} runs left and this needs {runs} (one per chip): check "
                               "fewer chips.", 402)
        if trial and trial.get("days_left") == 0:
            raise ServiceError("The trial has ended. Choose a plan on https://sensorforge.dev/optimize/app.", 402)
        if kind == "optimize" and effort and effort not in ((me.get("allowance") or {}).get("modes") or [effort]):
            raise ServiceError(f"effort='{effort}' is not in this plan; use effort='quick'.", 402)
        full = _day_full(me.get("day"), kind, effort)
        if full:
            raise ServiceError(full, 429)
        return me

    async def job_now(rid, last, misses):
        """The run's state; a passing network or server hiccup (a deploy, a dropped line) keeps the last state known,
        up to three times in a row."""
        try:
            j = await svc.job(rid)
            misses[rid] = 0
            return j
        except ServiceError as e:
            transient = e.status is None or e.status >= 500
            misses[rid] = misses.get(rid, 0) + 1
            if not transient or misses[rid] > 3:
                raise
            return last             # None when nothing is known yet: the caller waits and asks again

    async def wait(ids, max_wait, ctx, label):
        t0 = time.monotonic()
        delay = min(2.0, poll_s)
        last, misses = {}, {}
        while True:
            for i in ids:
                last[i] = await job_now(i, last.get(i), misses)
            el = time.monotonic() - t0
            if any(last[i] is None for i in ids):
                if el >= max_wait:
                    raise ServiceError("The service is not answering right now; try again in a few minutes.")
                await anyio.sleep(min(delay, max(0.5, max_wait - el)))
                delay = min(poll_s, delay * 1.5)
                continue
            jobs = [last[i] for i in ids]
            if all(j.get("status") in fmt.FINAL for j in jobs):
                await _progress(ctx, max_wait, max_wait, f"{label}: finished")
                return jobs, True
            if el >= max_wait:
                return jobs, False
            j = next(j for j in jobs if j.get("status") not in fmt.FINAL)
            await _progress(ctx, round(el, 1), max_wait, f"{label}: {fmt.status_line(j)[8:]}")
            await anyio.sleep(min(delay, max(0.5, max_wait - el)))
            delay = min(poll_s, delay * 1.5)

    async def upload(kind, chip, model, data=None, fields=None, ctx=None):
        """Submits the run, streaming the files; for a large upload, reports how much has been sent."""
        opened = []
        total = os.path.getsize(model) + (os.path.getsize(data) if data else 0)
        state = {"r": None, "err": None, "done": False}

        async def report():
            while not state["done"]:
                sent = sum(f.sent for f in opened)
                await _progress(ctx, sent, total, f"uploading: {sent / 2**20:.1f} of {total / 2**20:.1f} MB sent")
                await anyio.sleep(2.0)

        async with anyio.create_task_group() as tg:
            if total > (8 << 20):
                tg.start_soon(report)
            try:
                state["r"] = await svc.submit(kind, chip, model, data, fields, files_out=opened)
            except Exception as e:          # raised after the group, not inside it (no exception group)
                state["err"] = e
            finally:
                state["done"] = True
                tg.cancel_scope.cancel()
        if state["err"] is not None:
            raise state["err"]
        r = state["r"]
        book.add(r["id"], model, kind)
        return r

    async def download_into(j, folder=None, overwrite=False):
        rid = j["id"]
        if j.get("status") != "done":
            raise ServiceError(f"Run {rid} is {j.get('status')}, so there is nothing to download yet"
                               + ("; wait_for_run first." if j.get("status") not in fmt.FINAL else "."))
        if j.get("kind") == "fit":
            raise ServiceError("Check fit gives figures, not a model; lossless or optimize give a model to download.")
        if not j.get("has_download"):
            raise ServiceError(f"Run {rid} produced no model to download (verdict: {j.get('verdict') or 'none'}); only a "
                               "PASS gives one. run_status shows why.")
        if j.get("may_download") is False:
            raise ServiceError("Downloading the model is part of the paid plans; the full report is free: "
                               f"get_report(\"{rid}\"). See the plans on https://sensorforge.dev/optimize/app.", 402)
        if folder:
            dest = files.resolve(folder)
        else:
            m = book.model_of(rid)
            base = os.path.dirname(m) if m and os.path.isdir(os.path.dirname(m)) else os.getcwd()
            dest = os.path.join(base, "sensorforge", rid)
        if os.path.isfile(dest):
            raise ServiceError(f"{dest} is a file; pass a folder.")
        if os.path.isdir(dest) and os.listdir(dest) and not overwrite:
            raise ServiceError(f"{dest} already exists and is not empty; pass overwrite=True to replace its files, or "
                               "another folder.")
        parent = os.path.dirname(dest.rstrip(os.sep)) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".sensorforge-{rid}-", suffix=".part", dir=parent)
        os.close(fd)
        try:
            got = await svc.download(rid, tmp)
            with open(tmp, "rb") as f:
                magic = f.read(4)
            if magic == b"PK\x03\x04":
                written = files.unpack(tmp, dest, overwrite)
            else:
                name = os.path.basename(got.get("filename") or "") or f"{os.path.splitext(j.get('model') or 'model')[0]}.tflite"
                name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name).lstrip(".")
                if not name or name.lower().endswith((".sh", ".py", ".exe", ".bat", ".command")):
                    name = "model.tflite"
                os.makedirs(dest, exist_ok=True)
                target = os.path.join(dest, name)
                if os.path.lexists(target) and not overwrite:
                    raise ServiceError(f"{target} already exists; pass overwrite=True to replace it.")
                shutil.move(tmp, target)
                os.chmod(target, 0o644)
                written = [name]
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if not any(os.path.basename(w).lower() in ("report.md", "lossless.md") for w in written) and j.get("has_report"):
            try:
                text = await svc.report(rid)
                with open(os.path.join(dest, "report.md"), "w", encoding="utf-8") as f:
                    f.write(text)
                written.append("report.md")
            except ServiceError:
                pass
        return dest, written

    def listing(dest, written):
        rows = []
        for w in sorted(written, key=lambda x: (x.count("/"), x)):
            p = os.path.join(dest, w)
            size = os.path.getsize(p) if os.path.isfile(p) else 0
            what = files.describe(w)
            rows.append(f"- {w} ({fmt.kb(size)}){': ' + what if what else ''}")
        return f"Unpacked into {dest}:\n" + "\n".join(rows)

    # ---------------------------------------------------------------- tools

    @mcp.tool(annotations=READ_ONLY, structured_output=False)
    @_tool_errors
    async def list_chips() -> str:
        """List the chips (edge devices) the optimizer targets, with each one's SRAM and flash budget for the model.
        Use a chip's name as `chip` in check_fit, lossless and optimize. Chips without budgets take yours (ram_kb,
        flash_kb in optimize)."""
        ts = await targets(fresh=True)
        lines = [f"{len(ts)} chips (the model's SRAM / flash budget on each):"]
        for t in ts:
            budget = (f"{t['ram_kb']:g} KB SRAM, {t['flash_kb']:g} KB flash" if t.get("ram_kb") and t.get("flash_kb")
                      else "set your own budget")
            desc = _first_sentence(t.get("description") or "")
            lines.append(f"- {t['name']}{' (NPU)' if t.get('npu') else ''}: {budget}. {desc}")
        lines.append("Use the name as `chip`. No chip listed for your device? Pick the closest core and memory, or a "
                     "generic one, and give ram_kb / flash_kb to optimize.")
        return "\n".join(lines)

    @mcp.tool(annotations=SUBMITS, structured_output=False)
    @_tool_errors
    async def check_fit(
        model_path: Annotated[str, Field(description="The model: a .tflite or .onnx file, or a .zip of a firmware export. "
                                                     "Relative to the working directory, or starting with ~.")],
        chips: Annotated[list[str] | str | None, Field(description="Chip names from list_chips (a list, or comma "
                                                                   "separated). Each chip is one run. Default: cortex-m-cmsis, a generic Arm "
                                                                   "Cortex-M.")] = None,
        ctx: Context = None,
    ) -> str:
        """Check whether a model runs on one or more chips and how much SRAM and flash it needs there, measured. Takes
        seconds; no data needed. Returns a per-chip verdict with SRAM and flash against each chip's budget, and the SRAM
        lossless would reach. Use it first, before lossless or optimize."""
        model = files.check_model(model_path)
        names = [c.strip() for c in (chips.split(",") if isinstance(chips, str) else (chips or [])) if c and c.strip()]
        if not names:
            ts = await targets()
            names = [DEFAULT_CHIP if any(t["name"] == DEFAULT_CHIP for t in ts) else ts[0]["name"]]
        if len(names) > MAX_CHIPS:
            raise ServiceError(f"At most {MAX_CHIPS} chips per call (each is one run); pick the ones you are considering.")
        names = list(dict.fromkeys([await check_chip(c) for c in names]))
        await preflight("fit", runs=len(names))
        ids = []
        for c in names:
            r = await upload("fit", c, model, ctx=ctx)
            ids.append(r["id"])
        jobs, finished = await wait(ids, FIT_WAIT_S, ctx, "check fit")
        ts = await targets()
        out = [f"Check fit: {os.path.basename(model)} ({fmt.kb(os.path.getsize(model))}) on {', '.join(names)}"]
        rows, notes, pending = [], [], []
        for j in jobs:
            if j.get("status") == "done":
                rows += fmt.fit_rows(j, ts)
                for a in (j.get("summary") or {}).get("advice") or []:
                    if fmt.clip(a) not in notes:
                        notes.append(fmt.clip(a))
            elif j.get("status") in fmt.FINAL:
                out.append(f"{(j.get('params') or {}).get('target')}: {fmt.status_line(j)}")
            else:
                pending.append(j)
        if rows:
            out.append(fmt.table(fmt.FIT_HEAD, rows))
            rv = any(r.get("ram_build") == "rv32" for j in jobs for r in (j.get("summary") or {}).get("rows") or [])
            out.append(fmt.SRAM_NOTE_RV if rv else fmt.SRAM_NOTE_64)
        if notes:
            out.append("Notes:\n" + "\n".join(f"- {n}" for n in notes[:5]))
        out.append("Run ids: " + ", ".join(ids) + " (get_report for the full report).")
        if pending:
            out.append(f"Still running: {', '.join(j['id'] for j in pending)}. Call wait_for_run on each.")
        elif rows:
            out.append("Next: lossless(model, chip) gives less SRAM/flash with outputs identical to yours (no data); "
                       "optimize(model, data, chip) goes further within an accuracy limit.")
        return "\n".join(out)

    @mcp.tool(annotations=LOCAL_READ, structured_output=False)
    @_tool_errors
    async def check_data(
        model_path: Annotated[str, Field(description="The model the data is for (.tflite, .onnx, or a firmware .zip).")],
        data_path: Annotated[str, Field(description="The data: an .npz (X, y) or a .zip of image folders, one per class.")],
        output: Annotated[Literal["classes", "values"], Field(description="'classes' when the model outputs class scores; "
                                                              "'values' for detectors, autoencoders and regression.")] = "classes",
    ) -> str:
        """Check, locally, that a data file is what optimize needs for this model, before uploading anything. For an
        .npz: X present, float32, each sample in the model's input shape; y one class index per sample within the
        model's outputs; X_test / y_test if given. For an image zip: class folders (or train/ and test/). Explains each
        problem and how to fix it. Nothing is uploaded and no run is used."""
        model = files.check_model(model_path)
        data = files.check_data_file(data_path)
        rep = await anyio.to_thread.run_sync(datacheck.check, model, data, output)
        text = rep.text(os.path.basename(data), os.path.basename(model))
        if rep.ok:
            text += "\nNext: optimize(model_path, data_path, chip)."
        return text

    @mcp.tool(annotations=SUBMITS, structured_output=False)
    @_tool_errors
    async def lossless(
        model_path: Annotated[str, Field(description="The model: .tflite or .onnx, or a .zip of a firmware export.")],
        chip: Annotated[str, Field(description="A chip name from list_chips.")],
        folder: Annotated[str | None, Field(description="Where to unpack the result. Default: "
                                                        "<the model's folder>/sensorforge/<run id>/.")] = None,
        overwrite: Annotated[bool, Field(description="Replace files already in the folder.")] = False,
        ctx: Context = None,
    ) -> str:
        """Make the model smaller in SRAM and flash with outputs bit-identical to the original: no data, no retraining.
        Usually under a minute. Waits for the result, downloads it, and unpacks the model for TensorFlow Lite Micro and
        its C sources into the project. Returns the before -> after figures and the files. If it is still running after
        about four minutes it returns the run id: then call wait_for_run and download."""
        model = files.check_model(model_path)
        chip = await check_chip(chip)
        await preflight("lossless")
        r = await upload("lossless", chip, model, ctx=ctx)
        jobs, finished = await wait([r["id"]], FIT_WAIT_S, ctx, "lossless")
        j = jobs[0]
        ts = await targets()
        if not finished:
            return fmt.job_text(j, ts) + "\nIt is taking longer than usual: call wait_for_run, then download."
        downloaded, extra = False, ""
        if j.get("status") == "done" and j.get("has_download") and j.get("may_download") is not False:
            try:
                dest, written = await download_into(j, folder, overwrite)
                extra, downloaded = listing(dest, written), True
            except ServiceError as e:
                extra = f"Download: {e}"
        text = fmt.job_text(j, ts, downloaded)
        return text + ("\n" + extra if extra else "")

    @mcp.tool(annotations=SUBMITS, structured_output=False)
    @_tool_errors
    async def optimize(
        model_path: Annotated[str, Field(description="The trained model: .tflite or .onnx, or a .zip of a firmware export.")],
        data_path: Annotated[str, Field(description="The data: an .npz (X, y) or a .zip of image folders, one per class. "
                                                    "Run check_data first.")],
        chip: Annotated[str, Field(description="A chip name from list_chips.")],
        minimize: Annotated[Literal["ram", "flash", "both"], Field(description="What to shrink: SRAM ('ram'), flash, or "
                                                                   "both.")] = "ram",
        accuracy_limit: Annotated[float, Field(ge=0.1, le=20, description="How many points of accuracy the result may lose "
                                               "against your model, on the test part of the data.")] = 1.0,
        effort: Annotated[Literal["quick", "thorough"], Field(description="'quick' (about 15 minutes) or 'thorough' (up "
                                                              "to an hour, smaller results; paid plans).")] = "quick",
        ram_kb: Annotated[float | None, Field(gt=0, le=100000, description="SRAM budget in KB (default: the chip's).")] = None,
        flash_kb: Annotated[float | None, Field(gt=0, le=100000, description="Flash budget in KB (default: the chip's).")] = None,
        output: Annotated[Literal["classes", "values"], Field(description="'classes' for class scores; 'values' for "
                                                              "detectors, autoencoders and regression (y not needed).")] = "classes",
        image_scaling: Annotated[Literal["auto", "0..1", "-1..1", "imagenet"], Field(
            description="Image zips only: how the model expects pixels. 'auto' = as the model says (int8 .tflite files "
                        "carry it).")] = "auto",
        check_first: Annotated[bool, Field(description="Run the local data check first and stop on problems.")] = True,
        ctx: Context = None,
    ) -> str:
        """Find the smallest model for the chip within your accuracy limit, retrained on your data. Runs on the service
        for 5-60 minutes, so this returns right away with the run id and its place in the queue: then call
        wait_for_run(run_id) (again while it says still running), and download(run_id) when it passes. The data is used
        for this run only and deleted when it ends."""
        model = files.check_model(model_path)
        data = files.check_data_file(data_path)
        chip = await check_chip(chip)
        if check_first:
            rep = await anyio.to_thread.run_sync(datacheck.check, model, data, output)
            if not rep.ok:
                raise ToolError("Not submitted: the local data check found problems (no run was used).\n"
                                + rep.text(os.path.basename(data), os.path.basename(model))
                                + "\nFix them and call optimize again (check_first=False sends the data anyway).")
        await preflight("optimize", effort)
        fields = dict(minimize=minimize, max_drop=f"{accuracy_limit:g}", effort=effort, task="dense" if output == "values" else "auto",
                      ram_kb=f"{ram_kb:g}" if ram_kb else None, flash_kb=f"{flash_kb:g}" if flash_kb else None,
                      norm=NORM.get(image_scaling) if data.lower().endswith(".zip") else None)     # images only
        size = os.path.getsize(data)
        r = await upload("optimize", chip, model, data, fields, ctx)
        pos = r.get("position")
        return "\n".join([
            f"Submitted optimize run {r['id']}: {os.path.basename(model)} on {chip}, minimize {minimize}, accuracy limit "
            f"{accuracy_limit:g} point{'s' if accuracy_limit != 1 else ''}, {effort} effort "
            f"({fmt.kb(size) if size < 2**20 else f'{size / 2**20:.1f} MB'} of data uploaded).",
            f"Status: {r.get('status', 'queued')}{', ' + fmt.ordinal(pos) + ' in line' if pos else ''}. "
            f"{'Quick runs take about 15 minutes' if effort == 'quick' else 'Thorough runs take up to an hour'}.",
            f"Next: wait_for_run(\"{r['id']}\"); call it again while it says still running. Then download(\"{r['id']}\")."])

    @mcp.tool(annotations=READ_ONLY, structured_output=False)
    @_tool_errors
    async def run_status(run_id: Annotated[str, Field(description="The run id from check_fit, lossless, optimize or "
                                                                  "list_runs.")]) -> str:
        """The state of a run right now, without waiting: queued (with its place in line), running (with its latest
        progress), or finished with the verdict and the before -> after SRAM, flash and accuracy."""
        j = await svc.job(run_id)
        return fmt.job_text(j, await targets())

    @mcp.tool(annotations=READ_ONLY, structured_output=False)
    @_tool_errors
    async def wait_for_run(
        run_id: Annotated[str, Field(description="The run id.")],
        max_wait_s: Annotated[int, Field(ge=10, le=MAX_WAIT_S, description="How long to wait in this call, in seconds "
                                         "(at most 600).")] = 300,
        ctx: Context = None,
    ) -> str:
        """Wait for a run to finish, checking about every 10 seconds and reporting progress. Returns the final result
        (verdict and before -> after figures), or 'still running, call again' when max_wait_s is up: then call
        wait_for_run again. Optimize runs take 5-60 minutes, so expect several calls."""
        jobs, finished = await wait([run_id], max_wait_s, ctx, "run")
        j = jobs[0]
        text = fmt.job_text(j, await targets())
        if not finished:
            text += f"\nStill running after this call's {max_wait_s} s: call wait_for_run(\"{run_id}\") again."
        return text

    @mcp.tool(annotations=READ_ONLY, structured_output=False)
    @_tool_errors
    async def get_report(run_id: Annotated[str, Field(description="The run id.")]) -> str:
        """The run's full report as markdown: the measured figures, what was changed and why, and what to do next.
        Available on every plan, including the trial."""
        text = await svc.report(run_id)
        if len(text) > REPORT_MAX_CHARS:
            text = text[:REPORT_MAX_CHARS] + "\n\n[... cut here; the whole report is in the download as report.md]"
        return text

    @mcp.tool(annotations=WRITES, structured_output=False)
    @_tool_errors
    async def download(
        run_id: Annotated[str, Field(description="The run id of a lossless or optimize run that passed.")],
        folder: Annotated[str | None, Field(description="Where to unpack. Default: <the model's folder>/sensorforge/<run "
                                                        "id>/ (the working directory's when the model is unknown).")] = None,
        overwrite: Annotated[bool, Field(description="Replace files already in the folder.")] = False,
    ) -> str:
        """Download a finished run's result and unpack it into the project: the model for TensorFlow Lite Micro, its C
        sources (c/: the model as a C array, the operator list, integration notes) and the report. Refuses to write
        into a folder that is not empty unless overwrite=True. Returns each file and what it is for."""
        j = await svc.job(run_id)
        dest, written = await download_into(j, folder, overwrite)
        return listing(dest, written) + "\n" + fmt.next_step(j, downloaded=True)

    @mcp.tool(annotations=READ_ONLY, structured_output=False)
    @_tool_errors
    async def list_runs(limit: Annotated[int, Field(ge=1, le=100, description="How many runs, newest first.")] = 20) -> str:
        """The account's recent runs, newest first: id, when, kind, model, status and verdict. Runs are kept 14 days."""
        rows = await svc.jobs(limit)
        if not rows:
            return "No runs yet on this account."
        out = [fmt.table(["Run", "Submitted", "Kind", "Model", "Status", "Verdict"],
                         [[r.get("id"), fmt.short_when(r.get("created")), fmt.KIND.get(r.get("kind"), r.get("kind"))
                           + (f" ({r['mode']})" if r.get("mode") else ""), fmt.clip(r.get("model_name"), 40),
                           "failed" if fmt.failed_verdict(r) else r.get("status"),
                           fmt.clip((r.get("verdict") if r.get("verdict") not in (None, "no result") else r.get("error"))
                                    or r.get("verdict") or "", 70)] for r in rows])]
        out.append(f"Times are local ({time.strftime('%Z')}). run_status(run_id) for one run's figures.")
        return "\n".join(out)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True,
                                          open_world_hint=True), structured_output=False)
    @_tool_errors
    async def stop_run(run_id: Annotated[str, Field(description="The run id.")]) -> str:
        """Stop a queued or running run, for example one sent with the wrong chip or data. A stopped run gives no
        result; submit it again to rerun."""
        r = await svc.stop(run_id)
        j = await svc.job(run_id)
        if r.get("stopped"):
            return f"Stopped run {run_id}."
        return f"Run {run_id} was not stopped: it is {j.get('status')}" + (" (already finished)." if j.get("status") in fmt.FINAL else ".")

    @mcp.tool(annotations=READ_ONLY, structured_output=False)
    @_tool_errors
    async def account() -> str:
        """The account's plan, this month's usage, and how many runs are left today (with when the next one is free
        when a daily cap is reached)."""
        return _account_text(await svc.me())

    # ---------------------------------------------------------------- prompt

    @mcp.prompt(name="fit_model_to_chip", title="Fit a model to a chip")
    def fit_model_to_chip(model_path: str, chip: str = "", data_path: str = "") -> str:
        """Walk through fitting a trained model to an edge device: check fit, lossless, check the data, optimize, wait,
        download."""
        return _prompt_text(model_path, chip, data_path)

    return mcp


_ABBREV = ("e.g", "i.e", "ext", "approx", "vs", "etc")


def _first_sentence(t, n=160):
    """A chip description's first sentence (the rest is detail the agent does not need to pick a chip)."""
    t = " ".join(t.split())
    for m in re.finditer(r"[.;] (?=\S)", t):
        if m.start() >= n:
            break
        word = t[: m.start()].rsplit(" ", 1)[-1].lower()
        if t[m.start()] == "." and word.rstrip(")").strip("(") in _ABBREV:
            continue
        return t[: m.start() + 1]
    return fmt.clip(t, n)


def _day_full(day, kind, effort):
    """The sentence for a daily cap already reached, from /api/me's day block, else None."""
    if not isinstance(day, dict):
        return None
    caps = day.get("caps") if isinstance(day.get("caps"), dict) else {}
    keys = ["runs"] + (["optimize"] if kind == "optimize" else []) + (["thorough"] if kind == "optimize" and effort == "thorough" else [])
    for k in keys:
        cap, used = caps.get(k), day.get(k)
        if isinstance(cap, (int, float)) and isinstance(used, (int, float)) and used >= cap:
            nxt = day.get("next_free_at")
            when = f" The next one is free at {fmt.when(nxt)}." if isinstance(nxt, (int, float)) else ""
            what = {"runs": "runs", "optimize": "optimize runs", "thorough": "thorough runs"}[k]
            return f"Today's {what} are used ({int(used)} of {int(cap)} in the last 24 hours).{when}"
    return None


def _account_text(me):
    org = me.get("org") or {}
    a = me.get("allowance") or {}
    mo = me.get("month") or {}
    plan = str(org.get("plan") or "unknown")
    who = org.get("name") or org.get("id") or ""
    lines = [f"Account: {who} ({plan} plan{', ' + str(org['seats']) + ' seats' if (org.get('seats') or 0) > 1 else ''})"]
    if plan == "pending":
        lines.append("Waiting for approval: no runs yet. You will get an email when the account can run models.")
    lines.append(f"Downloads: {'included' if a.get('download') else 'not in this plan (reports are)'}; runs at once: "
                 f"{a.get('concurrency', '–')}; optimize effort: {', '.join(a.get('modes') or []) or 'none'}.")
    if mo:
        th = (f", {mo.get('thorough', 0)} of {mo['thorough_cap']} thorough" if mo.get("thorough_cap")
              else f", {mo.get('thorough', 0)} thorough")
        lines.append(f"This month: {mo.get('runs', 0)} runs; optimize: {mo.get('quick', 0)} quick{th}.")
    t = me.get("trial")
    if t:
        lines.append(f"Trial: {t.get('runs_left')} runs and {t.get('days_left')} days left.")
    d = me.get("day")
    if isinstance(d, dict):
        caps = d.get("caps") if isinstance(d.get("caps"), dict) else {}
        parts = []
        for k in ("runs", "optimize", "thorough"):
            if k in d or k in caps:
                cap = caps.get(k)
                left = f" ({max(0, int(cap) - int(d.get(k) or 0))} left)" if isinstance(cap, (int, float)) else ""
                parts.append(f"{k} {d.get(k, 0)}{' of ' + str(cap) if cap is not None else ''}{left}")
        if parts:
            lines.append("Last 24 hours: " + "; ".join(parts) + ".")
        nxt = d.get("next_free_at")
        if isinstance(nxt, (int, float)) and nxt > time.time():
            lines.append(f"Next free run: {fmt.when(nxt)}.")
    return "\n".join(lines)


def _prompt_text(model_path, chip, data_path):
    chip_line = (f"The target chip is {chip}." if chip else
                 "First call list_chips and pick the chip closest to the device (ask me if it is unclear).")
    data_line = (f"The data is {data_path}." if data_path else
                 "If there is no data file yet, build one as check_data explains (an .npz with X and y, or a zip of image "
                 "folders) from the project's training data, or stop after lossless.")
    return f"""Fit the model {model_path} to an edge device with the SensorForge tools.
{chip_line}
{data_line}

1. check_fit(model_path, [chip]): read the verdict and the SRAM / flash against the chip's budget.
2. lossless(model_path, chip): less SRAM and flash with identical outputs; it unpacks the files into the project.
   If that already fits with room to spare, stop and report.
3. check_data(model_path, data_path): fix every problem it lists before going on.
4. optimize(model_path, data_path, chip, minimize="ram" or "flash", accuracy_limit=1.0): it returns a run id.
5. wait_for_run(run_id): call it again each time it says still running (quick runs take about 15 minutes).
6. On PASS, download(run_id); on a refusal, tell me which limit to loosen.
Finish with a short table: SRAM, flash and accuracy before and after, the files written, and how to build them in."""


def main():
    create_server().run()


if __name__ == "__main__":
    main()
