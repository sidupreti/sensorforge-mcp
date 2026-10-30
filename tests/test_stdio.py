"""The installed console script, launched the way a client launches it (stdio), against the fake service."""
import os
import shutil
import sys

import pytest
from mcp import Client, StdioServerParameters

from fake_service import Running

pytestmark = pytest.mark.anyio
TINY = os.path.join(os.path.dirname(__file__), "data", "tiny_int8.tflite")


def script():
    exe = shutil.which("sensorforge-mcp", path=os.path.dirname(sys.executable))
    return [exe] if exe else [sys.executable, "-m", "sensorforge_mcp"]


async def test_stdio_server(tmp_path):
    shutil.copy(TINY, tmp_path / "model.tflite")
    cmd = script()
    with Running() as svc:
        env = dict(os.environ, SENSORFORGE_URL=svc.url, SENSORFORGE_API_KEY="sfk_team",
                   SENSORFORGE_MCP_HOME=str(tmp_path / "home"))
        params = StdioServerParameters(command=cmd[0], args=cmd[1:], env=env, cwd=str(tmp_path))
        async with Client(params, read_timeout_seconds=60) as c:
            assert c.server_info.name == "sensorforge"
            tools = {t.name for t in (await c.list_tools()).tools}
            assert len(tools) == 12
            r = await c.call_tool("check_fit", {"model_path": "model.tflite", "chips": ["nrf54l15"]})
            t = r.content[0].text
            assert not r.is_error and "| nrf54l15 | runs as is |" in t, t
            r = await c.call_tool("lossless", {"model_path": "model.tflite", "chip": "nrf54l15"})
            assert not r.is_error, r.content[0].text
        assert any((tmp_path / "sensorforge").iterdir())


async def test_stdio_older_protocol_client(tmp_path):
    """A client on the 2025 protocol (initialize handshake, JSON-RPC lines, a progress token): what most editors send."""
    import json

    import anyio

    shutil.copy(TINY, tmp_path / "model.tflite")
    cmd = script()
    with Running() as svc:
        env = dict(os.environ, SENSORFORGE_URL=svc.url, SENSORFORGE_API_KEY="sfk_team",
                   SENSORFORGE_MCP_HOME=str(tmp_path / "home"))
        async with await anyio.open_process(cmd, env=env, cwd=str(tmp_path)) as proc:
            buf = b""

            async def send(msg):
                await proc.stdin.send((json.dumps(msg) + "\n").encode())

            async def until(rid):
                nonlocal buf
                seen = []
                with anyio.fail_after(60):
                    while True:
                        while b"\n" not in buf:
                            buf += await proc.stdout.receive()
                        line, buf = buf.split(b"\n", 1)
                        m = json.loads(line)
                        seen.append(m)
                        if m.get("id") == rid:
                            return m, seen

            await send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "old-editor", "version": "1"}}})
            init, _ = await until(1)
            assert init["result"]["protocolVersion"] == "2025-06-18"
            assert init["result"]["serverInfo"]["name"] == "sensorforge"
            await send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            await send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            lst, _ = await until(2)
            assert len(lst["result"]["tools"]) == 12
            await send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "check_fit", "arguments": {"model_path": "model.tflite", "chips": ["nrf54l15"]},
                "_meta": {"progressToken": "p1"}}})
            res, seen = await until(3)
            assert not res["result"].get("isError"), res
            assert "| nrf54l15 | runs as is |" in res["result"]["content"][0]["text"]
            assert any(m.get("method") == "notifications/progress" and m["params"]["progressToken"] == "p1" for m in seen)
            await proc.stdin.aclose()
