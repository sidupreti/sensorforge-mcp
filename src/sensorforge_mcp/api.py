"""The hosted SensorForge Optimizer's HTTP API, as this package calls it.

Every call sends the API key (Authorization: Bearer sfk_...) and X-SF-Client. Errors come back as ServiceError: one
sentence saying what went wrong and how to fix it, ready to show to the agent.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from . import __version__

DEFAULT_URL = "https://sensorforge.dev/optimize"
KEYS_PAGE = "https://sensorforge.dev/optimize/app"
KEY_HELP = f"Set SENSORFORGE_API_KEY: create a key on {KEYS_PAGE} (API keys)."
PLANS_HELP = f"See the plans on {KEYS_PAGE}."
MAX_MODEL_BYTES = 50 * 1024 * 1024
MAX_DATA_BYTES = 500 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
CLIENT = f"sensorforge-mcp/{__version__}"


class ServiceError(Exception):
    def __init__(self, message: str, status: int | None = None, detail: str | None = None):
        super().__init__(message)
        self.status = status
        self.detail = detail        # the service's own words, when it sent any


def _local(host: str) -> bool:
    host = (host or "").lower().strip("[]")
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")


@dataclass
class Config:
    url: str = DEFAULT_URL
    key: str | None = None
    # test-only: a local copy of the service run with its development sign-in takes X-Dev-User / X-Dev-Org headers
    # instead of a key. Honoured for localhost URLs only, never sent anywhere else.
    dev_user: str | None = None
    dev_org: str | None = None

    @classmethod
    def from_env(cls, env=None) -> "Config":
        env = os.environ if env is None else env
        return cls(url=(env.get("SENSORFORGE_URL") or DEFAULT_URL).strip().rstrip("/"),
                   key=(env.get("SENSORFORGE_API_KEY") or "").strip() or None,
                   dev_user=(env.get("SENSORFORGE_DEV_USER") or "").strip() or None,
                   dev_org=(env.get("SENSORFORGE_DEV_ORG") or "").strip() or None)

    @property
    def is_local(self) -> bool:
        return _local(urlparse(self.url).hostname or "")

    @property
    def dev(self) -> bool:
        return bool(self.dev_user) and self.is_local

    def check(self):
        u = urlparse(self.url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ServiceError(f"SENSORFORGE_URL is not a web address ({self.url!r}); leave it unset for {DEFAULT_URL}.")
        if u.scheme == "http" and not self.is_local:
            raise ServiceError("SENSORFORGE_URL must start with https:// (the API key is never sent over plain http).")
        if not self.key and not self.dev:
            raise ServiceError(KEY_HELP, 401)


class CountingFile(io.FileIO):
    """A file whose reads are counted, so an upload can report its progress while it streams."""

    def __init__(self, path):
        super().__init__(path, "rb")
        self.sent = 0

    def read(self, size=-1):
        b = super().read(size)
        self.sent += len(b)
        return b

    def seek(self, pos, whence=0):
        r = super().seek(pos, whence)
        if whence == 0 and pos == 0:
            self.sent = 0
        return r


class Service:
    """One call per method; each opens its own connection (calls are minutes apart)."""

    def __init__(self, config: Config | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config or Config.from_env()
        self._transport = transport

    # ------------------------------------------------------------ plumbing

    def headers(self) -> dict:
        h = {"X-SF-Client": CLIENT, "User-Agent": CLIENT, "Accept": "application/json"}
        if self.config.dev:
            h["X-Dev-User"] = self.config.dev_user
            if self.config.dev_org:
                h["X-Dev-Org"] = self.config.dev_org
        elif self.config.key:
            h["Authorization"] = f"Bearer {self.config.key}"
        return h

    def _client(self, timeout) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.config.url + "/", headers=self.headers(), timeout=timeout,
                                 follow_redirects=False, transport=self._transport)

    def _unreachable(self, e: Exception) -> ServiceError:
        kind = "timed out" if isinstance(e, httpx.TimeoutException) else "connection failed"
        return ServiceError(f"Could not reach the SensorForge service at {self.config.url} ({kind}: {type(e).__name__}). "
                            "Check the network, or SENSORFORGE_URL if you set it; then try again.")

    async def request(self, method, path, *, what="", timeout=60.0, **kw) -> httpx.Response:
        self.config.check()
        try:
            async with self._client(timeout) as c:
                r = await c.request(method, path.lstrip("/"), **kw)
        except httpx.HTTPError as e:
            raise self._unreachable(e)
        if r.status_code >= 300:
            raise self.error(r.status_code, _detail(r), what)
        return r

    async def json(self, method, path, **kw):
        r = await self.request(method, path, **kw)
        try:
            return r.json()
        except ValueError:
            raise ServiceError(f"The service sent an answer this client cannot read (HTTP {r.status_code}, "
                               f"{r.headers.get('content-type', 'no type')}). Is SENSORFORGE_URL the optimizer's address?")

    def error(self, status: int, detail: str, what: str = "") -> ServiceError:
        """The one sentence (plus the fix) for an HTTP error."""
        d = (detail or "").strip()
        dd = d if not d or d.endswith((".", "!", "?", ")")) else d + "."
        if status == 401:
            msg = KEY_HELP if not self.config.key else ("The API key was not accepted (it may be mistyped, expired or "
                                                        "revoked). " + KEY_HELP)
        elif status == 402:
            msg = f"{dd or 'This is not part of your plan.'} {PLANS_HELP}"
        elif status == 403:
            msg = f"Not allowed: {dd or 'this account may not do that.'}"
        elif status == 404:
            msg = {"run": "No such run on this account (runs are kept 14 days); list_runs shows the account's runs.",
                   "report": "This run has no report (it may still be running, or ended before writing one).",
                   "download": "This run produced no file to download (only a PASS gives a model; see run_status)."
                   }.get(what, f"Not found: {dd or 'no such item.'}")
        elif status == 413:
            msg = (f"Too large for the service: {dd or 'the upload is over the limit.'} Limits: model 50 MB, data 500 MB "
                   "(data zips up to 2,000 MB unpacked).")
        elif status == 415:
            msg = dd or "Wrong file type: upload a .tflite or .onnx model (or a .zip of a firmware export) and .npz or .zip data."
        elif status == 422:
            msg = f"The service did not accept the input: {dd or 'bad input.'}"
        elif status == 429:
            msg = f"{dd or 'Too many runs right now.'} account() shows today's runs and when the next one is free."
        elif status in (502, 503, 504):
            msg = f"The service is busy or restarting{': ' + dd if d else '.'} Try again in a few minutes."
        elif 300 <= status < 400:
            msg = (f"The service answered with a redirect (HTTP {status}): SENSORFORGE_URL should be the optimizer's address, "
                   f"{DEFAULT_URL} by default.")
        else:
            msg = f"The service had an error (HTTP {status}{': ' + d if d else ''}). Try again in a minute."
        return ServiceError(msg, status, d or None)

    # ------------------------------------------------------------ the API

    async def targets(self) -> list:
        return await self.json("GET", "api/targets")

    async def me(self) -> dict:
        return await self.json("GET", "api/me")

    async def jobs(self, limit=20, before=None) -> list:
        params = {"limit": str(limit)}
        if before:
            params["before"] = before
        return await self.json("GET", "api/jobs", params=params)

    async def job(self, run_id) -> dict:
        return await self.json("GET", f"api/jobs/{_id(run_id)}", what="run")

    async def report(self, run_id) -> str:
        r = await self.request("GET", f"api/jobs/{_id(run_id)}/report", params={"fmt": "md"}, what="report")
        return r.text

    async def stop(self, run_id) -> dict:
        return await self.json("POST", f"api/jobs/{_id(run_id)}/stop", what="run")

    async def submit(self, kind, target, model_path, data_path=None, fields=None, files_out=None) -> dict:
        """POST /api/jobs, streaming the files from disk. files_out (a list) receives the open CountingFiles, so a caller
        can watch the upload's progress."""
        self.config.check()
        form = {"kind": kind, "target": target}
        for k, v in (fields or {}).items():
            if v is not None and v != "":
                form[k] = str(v)
        opened = []
        try:
            mf = CountingFile(model_path)
            opened.append(mf)
            files = {"model": (os.path.basename(model_path), mf, "application/octet-stream")}
            if data_path:
                df = CountingFile(data_path)
                opened.append(df)
                files["data"] = (os.path.basename(data_path), df, "application/octet-stream")
            if files_out is not None:
                files_out.extend(opened)
            # writes have no overall limit (a 500 MB upload on a slow line takes a while); after the last byte the
            # service stores and unpacks the upload before it answers
            timeout = httpx.Timeout(connect=30.0, read=600.0, write=120.0, pool=30.0)
            try:
                async with self._client(timeout) as c:
                    r = await c.post("api/jobs", data=form, files=files)
            except (httpx.WriteError, httpx.RemoteProtocolError, httpx.ReadError) as e:
                # the service may turn an upload away before reading it all (no key, a cap reached) and close the line
                raise ServiceError(f"The service closed the connection during the upload ({type(e).__name__}). Check "
                                   "account() for the plan and today's runs, then try again.", None)
            except httpx.HTTPError as e:
                raise self._unreachable(e)
        finally:
            for f in opened:
                f.close()
        if r.status_code >= 300:
            raise self.error(r.status_code, _detail(r), "submit")
        return r.json()

    async def download(self, run_id, dest_path) -> dict:
        """Streams the run's download to dest_path. Returns {filename, content_type, bytes}."""
        self.config.check()
        timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
        try:
            async with self._client(timeout) as c:
                async with c.stream("GET", f"api/jobs/{_id(run_id)}/download") as r:
                    if r.status_code >= 300:
                        await r.aread()
                        raise self.error(r.status_code, _detail(r), "download")
                    n = 0
                    with open(dest_path, "wb") as f:
                        async for chunk in r.aiter_bytes(1 << 20):
                            n += len(chunk)
                            if n > MAX_DOWNLOAD_BYTES:
                                raise ServiceError("The download is larger than 2 GB, more than any run produces; stopped.")
                            f.write(chunk)
                    return dict(filename=_filename(r.headers.get("content-disposition", "")),
                                content_type=r.headers.get("content-type", ""), bytes=n)
        except httpx.HTTPError as e:
            raise self._unreachable(e)


def _id(run_id) -> str:
    s = str(run_id or "").strip()
    if not s or not all(c.isalnum() or c in "-_" for c in s) or len(s) > 64:
        raise ServiceError(f"{run_id!r} is not a run id (run ids look like 3f9a0c1d2b4e5f60; list_runs shows them).")
    return s


def _detail(r: httpx.Response) -> str:
    try:
        j = r.json()
    except ValueError:
        return ""
    if isinstance(j, dict):
        d = j.get("error") or j.get("detail") or j.get("message") or ""
        if isinstance(d, list):          # FastAPI's validation errors
            d = "; ".join(str(x.get("msg", x)) if isinstance(x, dict) else str(x) for x in d)
        return str(d)[:600]
    return ""


def _filename(cd: str) -> str:
    import re
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd or "")
    return os.path.basename(m.group(1)) if m else ""
