"""What the release needs to agree on: versions, the registry name in the README, the console script, no internals."""
import json
import os
import re
import pytest

import sensorforge_mcp

tomllib = pytest.importorskip("tomllib")      # Python 3.11+

ROOT = os.path.dirname(os.path.dirname(__file__))


def read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return f.read()


def test_versions_agree():
    v = sensorforge_mcp.__version__
    s = json.loads(read("server.json"))
    assert s["version"] == v and all(p["version"] == v for p in s["packages"])
    assert f"## [{v}]" in read("CHANGELOG.md")


def test_registry_entry():
    s = json.loads(read("server.json"))
    assert re.fullmatch(r"[a-zA-Z0-9.-]+/[a-zA-Z0-9._-]+", s["name"]) and len(s["description"]) <= 100
    p = s["packages"][0]
    assert p["registryType"] == "pypi" and p["identifier"] == "sensorforge-mcp" and p["transport"] == {"type": "stdio"}
    assert re.search(r"mcp-name: " + re.escape(s["name"]) + r"\s", read("README.md"))     # PyPI ownership check
    key = next(e for e in p["environmentVariables"] if e["name"] == "SENSORFORGE_API_KEY")
    assert key["isRequired"] and key["isSecret"]


def test_console_script_and_license():
    py = tomllib.loads(read("pyproject.toml"))
    assert py["project"]["scripts"]["sensorforge-mcp"] == "sensorforge_mcp.server:main"
    assert py["project"]["license"] == "Apache-2.0"
    assert "Copyright 2026 EdgeForge LLC" in read("LICENSE") and "Apache License" in read("LICENSE")
    assert not any(d.startswith(("torch", "tensorflow", "onnx")) for d in py["project"]["dependencies"])


def test_docs_name_no_competitors_or_prices():
    text = read("README.md") + read("examples", "AGENTS.md") + read("CHANGELOG.md")
    assert "edge devices" in text
    assert not re.search(r"\$\s?\d|\bUSD\b|per seat|per month", text, re.I)
    terms = os.path.join(ROOT, "tests", ".private_terms.txt")       # local, git-ignored list
    if os.path.exists(terms):
        for t in (l.strip() for l in open(terms)):
            if t and not t.startswith("#"):
                assert not re.search(t, text, re.I), t
