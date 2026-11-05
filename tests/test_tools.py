"""Every tool, through the MCP SDK's in-memory client, against the fake service on a real localhost port."""
import os
import re
import shutil
import zipfile

import numpy as np
import pytest
from mcp import Client

from fake_service import Running
from sensorforge_mcp.api import KEY_HELP, Config, Service
from sensorforge_mcp.server import RunBook, create_server

pytestmark = pytest.mark.anyio
HERE = os.path.dirname(__file__)
TINY = os.path.join(HERE, "data", "tiny_int8.tflite")
TOOLS = {"list_chips", "check_fit", "check_data", "lossless", "optimize", "run_status", "wait_for_run", "get_report",
         "download", "list_runs", "stop_run", "account"}
# terms that must never reach an agent: a local, git-ignored list (tests/.private_terms.txt), one regex per line
_TERMS = os.path.join(HERE, ".private_terms.txt")
_LINES = [l.strip() for l in open(_TERMS)] if os.path.exists(_TERMS) else []
INTERNALS = re.compile("|".join(l for l in _LINES if l and not l.startswith("#")) or r"(?!x)x", re.I)


@pytest.fixture(scope="module")
def service():
    with Running() as r:
        yield r


@pytest.fixture
def project(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    for n in ("model", "slow_model", "fail_model", "reject_model", "evil_model", "link_model", "bare_model"):
        shutil.copy(TINY, d / f"{n}.tflite")
    rs = np.random.RandomState(0)
    np.savez(d / "data.npz", X=rs.uniform(-2, 2, (300, 49, 10, 1)).astype("float32"), y=rs.randint(0, 4, 300))
    np.savez(d / "bad.npz", X=rs.uniform(-2, 2, (300, 49, 10, 1)).astype("float32"), y=rs.randint(0, 9, 300))
    return d


def client(service, tmp_path, key="sfk_team", **kw):
    svc = Service(Config(url=service.url, key=key, **kw))
    return Client(create_server(svc, RunBook(str(tmp_path / "home")), poll_s=0.05))


def text(r):
    return "\n".join(c.text for c in r.content if getattr(c, "type", "") == "text")


async def call(c, name, args=None, **kw):
    r = await c.call_tool(name, args or {}, **kw)
    return r, text(r)


# ---------------------------------------------------------------- listing

async def test_lists_every_tool_with_plain_descriptions(service, tmp_path):
    async with client(service, tmp_path) as c:
        tools = (await c.list_tools()).tools
        assert {t.name for t in tools} == TOOLS
        for t in tools:
            assert t.description and len(t.description) > 60, t.name
            blob = t.description + " ".join(str(p.get("description", "")) for p in t.input_schema["properties"].values())
            assert not INTERNALS.search(blob), (t.name, INTERNALS.search(blob).group(0))
        assert {t.name for t in tools if t.annotations and t.annotations.read_only_hint} >= {
            "list_chips", "check_data", "run_status", "get_report", "list_runs", "account"}
        prompts = (await c.list_prompts()).prompts
        assert [p.name for p in prompts] == ["fit_model_to_chip"]
        p = await c.get_prompt("fit_model_to_chip", {"model_path": "m.tflite", "chip": "nrf54l15"})
        body = p.messages[0].content.text
        assert "check_fit" in body and "wait_for_run" in body and "download" in body and not INTERNALS.search(body)
        assert not INTERNALS.search(c.instructions or "")


async def test_list_chips(service, tmp_path):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "list_chips")
        assert not r.is_error
        assert "nrf54l15: 150 KB SRAM, 1000 KB flash" in t and "cortex-m-cmsis: set your own budget" in t
        assert "Reference kernels" not in t           # first sentence only


async def test_every_call_identifies_the_client_and_sends_the_key(service, tmp_path):
    service.fake.calls.clear()
    async with client(service, tmp_path) as c:
        await call(c, "list_chips")
        await call(c, "account")
    assert service.fake.calls
    for _, _, h in service.fake.calls:
        assert h.get("x-sf-client", "").startswith("sensorforge-mcp/")
        assert h.get("authorization") == "Bearer sfk_team"


# ---------------------------------------------------------------- check_fit

async def test_check_fit_default_and_several_chips(service, project, tmp_path):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "check_fit", {"model_path": str(project / "model.tflite")})
        assert not r.is_error, t
        assert "| cortex-m-cmsis | runs as is | 24.1 KB | 46.5 KB | 20.3 KB |" in t
        r, t = await call(c, "check_fit", {"model_path": str(project / "model.tflite"), "chips": "nrf54l15, riscv-512k"})
        assert "| nrf54l15 | runs as is | 24.1 KB / 150 KB | 46.5 KB / 1000 KB | 20.3 KB |" in t
        assert "| riscv-512k |" in t and "Run ids:" in t and "A note from the service." in t


async def test_check_fit_relative_path_and_tilde(service, project, tmp_path, monkeypatch):
    monkeypatch.chdir(project)
    async with client(service, tmp_path) as c:
        r, t = await call(c, "check_fit", {"model_path": "model.tflite", "chips": ["nrf54l15"]})
        assert not r.is_error, t
    monkeypatch.setenv("HOME", str(project)
    monkeypatch.setenv("USERPROFILE", str(project)     # what expanduser reads on Windows)
    async with client(service, tmp_path) as c:
        r, t = await call(c, "check_fit", {"model_path": "~/model.tflite", "chips": ["nrf54l15"]})
        assert not r.is_error, t


async def test_check_fit_unknown_chip_suggests(service, project, tmp_path):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "check_fit", {"model_path": str(project / "model.tflite"), "chips": ["nrf54l1"]})
        assert r.is_error and "Unknown chip 'nrf54l1'" in t and "Did you mean nrf54l15" in t


@pytest.mark.parametrize("name", ["model.pt", "model.pth", "model.h5", "model.keras", "saved_model.pb"])
async def test_refuses_code_formats_before_uploading(service, project, tmp_path, name):
    (project / name).write_bytes(b"x" * 100)
    n = len(service.fake.uploads)
    async with client(service, tmp_path) as c:
        r, t = await call(c, "check_fit", {"model_path": str(project / name)})
    assert r.is_error and "export the model to .tflite or .onnx first" in t
    assert len(service.fake.uploads) == n


async def test_refuses_savedmodel_folder_and_missing_file(service, project, tmp_path):
    sm = project / "exported"
    sm.mkdir()
    (sm / "saved_model.pb").write_bytes(b"x")
    async with client(service, tmp_path) as c:
        r, t = await call(c, "check_fit", {"model_path": str(sm)})
        assert r.is_error and "SavedModel" in t
        r, t = await call(c, "check_fit", {"model_path": str(project / "nope.tflite")})
        assert r.is_error and "No model file" in t
        (project / "fake.tflite").write_bytes(b"not a flatbuffer at all")
        r, t = await call(c, "check_fit", {"model_path": str(project / "fake.tflite")})
        assert r.is_error and "TFL3" in t


async def test_size_checked_before_upload(service, project, tmp_path):
    big = project / "big.tflite"
    with open(big, "wb") as f:
        f.write(b"\x1c\x00\x00\x00TFL3")
        f.truncate(50 * 1024 * 1024 + 1)          # sparse: no 50 MB written
    n = len(service.fake.uploads)
    async with client(service, tmp_path) as c:
        r, t = await call(c, "check_fit", {"model_path": str(big)})
    assert r.is_error and "up to 50 MB" in t and len(service.fake.uploads) == n


async def test_service_413_and_415_are_explained(service, project, tmp_path):
    service.fake.max_model = 10
    try:
        async with client(service, tmp_path) as c:
            r, t = await call(c, "check_fit", {"model_path": str(project / "model.tflite"), "chips": ["nrf54l15"]})
        assert r.is_error and "Too large" in t and "model 50 MB, data 500 MB" in t
    finally:
        service.fake.max_model = 50 * 1024 * 1024


# ---------------------------------------------------------------- lossless and download

def files_in(d):
    return sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())


async def test_lossless_downloads_and_unpacks(service, project, tmp_path):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "lossless", {"model_path": str(project / "model.tflite"), "chip": "nrf54l15"})
    assert not r.is_error, t
    assert "Verdict: PASS" in t and "| Flash (model file) | 46.5 KB | 33.1 KB | -29% |" in t
    runs = list((project / "sensorforge").iterdir())
    assert len(runs) == 1
    got = files_in(runs[0])
    assert got == ["c/INTEGRATION.md", "c/model_data.cc", "c/model_data.h", "c/model_ops.h", "c/run.sh", "lossless.md",
                   "model_lossless.tflite"]
    assert "model_lossless.tflite" in t and "the file to deploy" in t
    for g in got:                                   # nothing from a download is made executable
        assert not os.stat(runs[0] / g).st_mode & 0o111


async def test_lossless_on_trial_shows_figures_without_download(service, project, tmp_path):
    async with client(service, tmp_path, key="sfk_trial") as c:
        r, t = await call(c, "lossless", {"model_path": str(project / "model.tflite"), "chip": "nrf54l15"})
    assert not r.is_error and "Verdict: PASS" in t and "part of the paid plans" in t
    assert not (project / "sensorforge").exists()


async def test_download_bare_tflite_and_report(service, project, tmp_path):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "lossless", {"model_path": str(project / "bare_model.tflite"), "chip": "nrf54l15"})
    assert not r.is_error, t
    d = next((project / "sensorforge").iterdir())
    assert files_in(d) == ["bare_model_lossless.tflite", "report.md"]


@pytest.mark.parametrize("name,why", [("evil_model", "outside"), ("link_model", "link")])
async def test_unsafe_zip_unpacks_nothing(service, project, tmp_path, name, why):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "lossless", {"model_path": str(project / f"{name}.tflite"), "chip": "nrf54l15"})
    assert "nothing was unpacked" in t and why in t
    assert not (project / "sensorforge" / "evil.txt").exists() and not (project / "evil.txt").exists()
    for d in (project / "sensorforge").glob("*"):
        assert d.is_dir() and not any(d.iterdir())
    assert not list((project / "sensorforge").glob(".*.part"))


async def test_download_tool_folder_and_overwrite(service, project, tmp_path):
    async with client(service, tmp_path, key="sfk_trial") as c:
        r, t = await call(c, "lossless", {"model_path": str(project / "model.tflite"), "chip": "nrf54l15"})
    rid = re.search(r"Run ([0-9a-f]{16})", t).group(1)
    async with client(service, tmp_path, key="sfk_trial") as c:
        r, t = await call(c, "download", {"run_id": rid})
        assert r.is_error and "paid plans" in t
    async with client(service, tmp_path) as c:
        r, t = await call(c, "lossless", {"model_path": str(project / "model.tflite"), "chip": "nrf54l15"})
        rid = re.search(r"Run ([0-9a-f]{16})", t).group(1)
        r, t = await call(c, "download", {"run_id": rid})
        assert r.is_error and "not empty" in t
        r, t = await call(c, "download", {"run_id": rid, "overwrite": True})
        assert not r.is_error and "Unpacked into" in t
        out = tmp_path / "elsewhere"
        r, t = await call(c, "download", {"run_id": rid, "folder": str(out)})
        assert not r.is_error and (out / "model_lossless.tflite").exists()


async def test_download_of_a_fit_run_and_an_unknown_run(service, project, tmp_path):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "check_fit", {"model_path": str(project / "model.tflite"), "chips": ["nrf54l15"]})
        rid = re.search(r"Run ids: ([0-9a-f]{16})", t).group(1)
        r, t = await call(c, "download", {"run_id": rid})
        assert r.is_error and "Check fit gives figures" in t
        r, t = await call(c, "download", {"run_id": "0123456789abcdef"})
        assert r.is_error and "No such run" in t
        r, t = await call(c, "run_status", {"run_id": "../../etc"})
        assert r.is_error and "is not a run id" in t


# ---------------------------------------------------------------- data and optimize

async def test_check_data_good_and_bad(service, project, tmp_path):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "check_data", {"model_path": str(project / "model.tflite"), "data_path": str(project / "data.npz")})
        assert not r.is_error and "Result: ready for optimize" in t
        r, t = await call(c, "check_data", {"model_path": str(project / "model.tflite"), "data_path": str(project / "bad.npz")})
        assert "NOT READY" in t and "the model outputs 4 scores" in t
    assert not any(u["names"].get("data") for u in service.fake.uploads if u["names"].get("data") == "bad.npz")


async def test_optimize_submits_and_returns_at_once(service, project, tmp_path):
    n = len(service.fake.uploads)
    async with client(service, tmp_path) as c:
        r, t = await call(c, "optimize", {"model_path": str(project / "model.tflite"), "data_path": str(project / "data.npz"),
                                          "chip": "nrf54l15", "minimize": "both", "accuracy_limit": 2.5, "ram_kb": 64,
                                          "image_scaling": "-1..1", "output": "values"})
    assert not r.is_error, t
    assert "Submitted optimize run" in t and "1st in line" in t and "wait_for_run(" in t
    u = service.fake.uploads[n]
    assert u["fields"] == {"kind": "optimize", "target": "nrf54l15", "minimize": "both", "max_drop": "2.5", "effort": "quick",
                           "task": "dense", "ram_kb": "64"}            # image_scaling is for image zips only
    assert u["sizes"] == {"model": os.path.getsize(project / "model.tflite"), "data": os.path.getsize(project / "data.npz")}
    assert u["client"].startswith("sensorforge-mcp/")


async def test_optimize_stops_on_bad_data_without_uploading(service, project, tmp_path):
    n = len(service.fake.uploads)
    async with client(service, tmp_path) as c:
        r, t = await call(c, "optimize", {"model_path": str(project / "model.tflite"), "data_path": str(project / "bad.npz"),
                                          "chip": "nrf54l15"})
    assert r.is_error and "Not submitted" in t and "no run was used" in t
    assert len(service.fake.uploads) == n


async def test_optimize_daily_cap_is_caught_before_the_upload(service, project, tmp_path):
    n = len(service.fake.uploads)
    async with client(service, tmp_path, key="sfk_capped") as c:
        r, t = await call(c, "optimize", {"model_path": str(project / "model.tflite"), "data_path": str(project / "data.npz"),
                                          "chip": "nrf54l15"})
    assert r.is_error and "Today's runs are used (20 of 20" in t and "next one is free at" in t
    assert len(service.fake.uploads) == n


async def test_optimize_thorough_not_in_trial(service, project, tmp_path):
    async with client(service, tmp_path, key="sfk_trial") as c:
        r, t = await call(c, "optimize", {"model_path": str(project / "model.tflite"), "data_path": str(project / "data.npz"),
                                          "chip": "nrf54l15", "effort": "thorough"})
    assert r.is_error and "effort='thorough' is not in this plan" in t


async def test_pending_account(service, project, tmp_path):
    async with client(service, tmp_path, key="sfk_pending") as c:
        r, t = await call(c, "check_fit", {"model_path": str(project / "model.tflite")})
        assert r.is_error and "waiting for approval" in t
        r, t = await call(c, "account")
        assert "Waiting for approval" in t


# ---------------------------------------------------------------- waiting and status

async def test_wait_for_run_reports_progress_and_finishes(service, project, tmp_path):
    seen = []

    async def on_progress(p, total, msg):
        seen.append((p, total, msg))

    async with client(service, tmp_path) as c:
        r, t = await call(c, "optimize", {"model_path": str(project / "slow_model.tflite"),
                                          "data_path": str(project / "data.npz"), "chip": "nrf54l15"})
        rid = re.search(r"run ([0-9a-f]{16})", t).group(1)
        r, t = await call(c, "run_status", {"run_id": rid})
        assert "Status: running for" in t and "usually take about 15 minutes" in t
        assert "search" not in t and "step one" not in t          # the service's working notes are not relayed
        r, t = await call(c, "wait_for_run", {"run_id": rid, "max_wait_s": 300}, progress_callback=on_progress)
    assert not r.is_error, t
    assert "Verdict: PASS" in t and "| Accuracy (TensorFlow Lite Micro, 4,874 test samples) | 93.10% | 92.85% | -0.25 pts |" in t
    assert "and 1 more in the report" in t and f'download("{rid}")' in t
    assert seen and all(b >= a for (a, _, _), (b, _, _) in zip(seen, seen[1:]))


async def test_wait_for_run_says_call_again(service, project, tmp_path):
    service.fake.slow_polls = 10_000
    try:
        async with client(service, tmp_path) as c:
            r, t = await call(c, "optimize", {"model_path": str(project / "slow_model.tflite"),
                                              "data_path": str(project / "data.npz"), "chip": "nrf54l15"})
            rid = re.search(r"run ([0-9a-f]{16})", t).group(1)
            r, t = await call(c, "wait_for_run", {"run_id": rid, "max_wait_s": 10})
            assert not r.is_error and "Still running" in t and "call wait_for_run" in t
            r, t = await call(c, "stop_run", {"run_id": rid})
            assert t == f"Stopped run {rid}."
            r, t = await call(c, "stop_run", {"run_id": rid})
            assert "was not stopped" in t and "already finished" in t
            r, t = await call(c, "run_status", {"run_id": rid})
            assert "Status: stopped." in t
    finally:
        service.fake.slow_polls = 4


async def test_failed_and_rejected_runs(service, project, tmp_path):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "optimize", {"model_path": str(project / "fail_model.tflite"),
                                          "data_path": str(project / "data.npz"), "chip": "nrf54l15"})
        rid = re.search(r"run ([0-9a-f]{16})", t).group(1)
        r, t = await call(c, "wait_for_run", {"run_id": rid})
        assert "Status: failed. the model could not be read" in t and "fix the input" in t
        r, t = await call(c, "optimize", {"model_path": str(project / "reject_model.tflite"),
                                          "data_path": str(project / "data.npz"), "chip": "nrf54l15"})
        rid = re.search(r"run ([0-9a-f]{16})", t).group(1)
        r, t = await call(c, "wait_for_run", {"run_id": rid})
        assert "Verdict: REJECTED" in t and "The smallest model within 1 point needs 30 KB." in t
        assert "Raise the accuracy limit to 2 points." in t and "Loosen one" in t
        r, t = await call(c, "download", {"run_id": rid})
        assert r.is_error and "only a PASS gives one" in t


async def test_get_report_and_list_runs(service, project, tmp_path):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "check_fit", {"model_path": str(project / "model.tflite"), "chips": ["nrf54l15"]})
        rid = re.search(r"Run ids: ([0-9a-f]{16})", t).group(1)
        r, t = await call(c, "get_report", {"run_id": rid})
        assert not r.is_error and t.startswith(f"# Report for {rid}")
        r, t = await call(c, "list_runs", {"limit": 5})
        assert not r.is_error and rid in t and "| Run | Submitted |" in t and "Check fit" in t


# ---------------------------------------------------------------- account and errors

async def test_account_texts(service, tmp_path):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "account")
        assert "team plan" in t and "Last 24 hours: runs 3 of 20 (17 left); optimize 1 of 10 (9 left)" in t
    async with client(service, tmp_path, key="sfk_trial") as c:
        r, t = await call(c, "account")
        assert "Trial: 7 runs and 12 days left." in t and "not in this plan" in t
    async with client(service, tmp_path, key="sfk_capped") as c:
        r, t = await call(c, "account")
        assert "runs 20 of 20 (0 left)" in t and "Next free run:" in t


async def test_missing_and_bad_key(service, project, tmp_path):
    service.fake.calls.clear()
    async with client(service, tmp_path, key=None) as c:
        r, t = await call(c, "account")
        assert r.is_error and KEY_HELP in t
    assert not service.fake.calls                    # nothing sent without a key
    async with client(service, tmp_path, key="sfk_revoked") as c:
        r, t = await call(c, "account")
        assert r.is_error and "not accepted" in t and KEY_HELP in t
        r, t = await call(c, "check_fit", {"model_path": str(project / "model.tflite")})
        assert r.is_error and KEY_HELP in t


async def test_unreachable_service(tmp_path):
    svc = Service(Config(url="http://127.0.0.1:9/optimize", key="sfk_team"))
    async with Client(create_server(svc, RunBook(str(tmp_path / "home")))) as c:
        r, t = await call(c, "list_chips")
    assert r.is_error and "Could not reach the SensorForge service" in t


async def test_plain_http_to_a_remote_host_is_refused(tmp_path):
    svc = Service(Config(url="http://sensorforge.example/optimize", key="sfk_team"))
    async with Client(create_server(svc, RunBook(str(tmp_path / "home")))) as c:
        r, t = await call(c, "account")
    assert r.is_error and "https://" in t


async def test_dev_headers_only_for_localhost(service, tmp_path):
    service.fake.calls.clear()
    svc = Service(Config(url=service.url, dev_user="tester", dev_org="team-x"))
    h = svc.headers()
    assert h["X-Dev-User"] == "tester" and h["X-Dev-Org"] == "team-x" and "Authorization" not in h
    remote = Service(Config(url="https://sensorforge.dev/optimize", key="sfk_x", dev_user="tester"))
    assert "X-Dev-User" not in remote.headers() and remote.headers()["Authorization"] == "Bearer sfk_x"


async def test_runbook_records_the_model_folder(service, project, tmp_path):
    book = RunBook(str(tmp_path / "home"))
    async with client(service, tmp_path) as c:
        r, t = await call(c, "optimize", {"model_path": str(project / "model.tflite"), "data_path": str(project / "data.npz"),
                                          "chip": "nrf54l15"})
        rid = re.search(r"run ([0-9a-f]{16})", t).group(1)
        assert book.model_of(rid) == str(project / "model.tflite")
        await call(c, "wait_for_run", {"run_id": rid})
        r, t = await call(c, "download", {"run_id": rid})
    assert not r.is_error, t
    got = files_in(project / "sensorforge" / rid)
    assert "optimized_tflm.tflite" in got and "optimized_litert.tflite" in got and "report.md" in got
    assert "desktop LiteRT" in t and "c/model_ops.h" in t


async def test_wait_rides_out_a_restart_but_not_a_long_outage(service, project, tmp_path):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "optimize", {"model_path": str(project / "slow_model.tflite"),
                                          "data_path": str(project / "data.npz"), "chip": "nrf54l15"})
        rid = re.search(r"run ([0-9a-f]{16})", t).group(1)
        await call(c, "run_status", {"run_id": rid})
        service.fake.flaky = 2
        r, t = await call(c, "wait_for_run", {"run_id": rid})
        assert not r.is_error and "Verdict: PASS" in t, t
        r, t = await call(c, "optimize", {"model_path": str(project / "slow_model.tflite"),
                                          "data_path": str(project / "data.npz"), "chip": "nrf54l15"})
        rid = re.search(r"run ([0-9a-f]{16})", t).group(1)
        await call(c, "run_status", {"run_id": rid})
        service.fake.flaky = 10
        try:
            r, t = await call(c, "wait_for_run", {"run_id": rid})
            assert r.is_error and "busy or restarting" in t
        finally:
            service.fake.flaky = 0


async def test_trial_with_fewer_runs_than_chips(service, project, tmp_path):
    service.fake.trial_runs_left = 2
    try:
        async with client(service, tmp_path, key="sfk_trial") as c:
            r, t = await call(c, "check_fit", {"model_path": str(project / "model.tflite"),
                                               "chips": ["nrf54l15", "riscv-512k", "cortex-m-cmsis"]})
        assert r.is_error and "2 runs left and this needs 3" in t
    finally:
        service.fake.trial_runs_left = 7


async def test_optimize_image_zip_sends_scaling(service, project, tmp_path):
    z = project / "images.zip"
    with zipfile.ZipFile(z, "w") as f:
        for c in "abcd":
            for i in range(25):
                f.writestr(f"images/{c}/{i}.jpg", b"\xff\xd8\xff")
    n = len(service.fake.uploads)
    async with client(service, tmp_path) as c:
        r, t = await call(c, "optimize", {"model_path": str(project / "model.tflite"), "data_path": str(z),
                                          "chip": "nrf54l15", "image_scaling": "imagenet", "minimize": "flash"})
    assert not r.is_error, t
    assert service.fake.uploads[n]["fields"]["norm"] == "imagenet"
    assert service.fake.uploads[n]["names"]["data"] == "images.zip"


async def test_download_of_an_unknown_model_goes_to_the_working_folder(service, project, tmp_path, monkeypatch):
    async with client(service, tmp_path) as c:
        r, t = await call(c, "lossless", {"model_path": str(project / "model.tflite"), "chip": "nrf54l15"})
        rid = re.search(r"Run ([0-9a-f]{16})", t).group(1)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    svc = Service(Config(url=service.url, key="sfk_team"))
    async with Client(create_server(svc, RunBook(str(tmp_path / "other-home")), poll_s=0.05)) as c:
        r, t = await call(c, "download", {"run_id": rid})
    assert not r.is_error and (work / "sensorforge" / rid / "model_lossless.tflite").exists(), t
