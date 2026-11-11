"""
test_mcp_server.py - Part A: the filesystem MCP server on its own (no agent, no LLM, free and fast)

Most tests talk to the server through an in-memory MCP session (the real protocol, no subprocess).
The JSON-RPC tests start the server as a subprocess and send raw JSON lines, like any client would.

Run:  pytest tests/test_mcp_server.py -v
"""

import asyncio
import threading
import time

import pytest
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session

from scripts.jsonrpc_demo import RawMCPConnection
from src.mcp_servers import filesystem_mcp_server as server
from src.mcp_servers.server_config import load_config

RESUME = "data/resumes/chen_martin.pdf"


def run(test_body):
    """Open an in-memory MCP session to the server, run `test_body(session)` and return its result."""
    async def main():
        async with create_connected_server_and_client_session(server.mcp._mcp_server) as session:
            return await test_body(session)
    return asyncio.run(main())


def call(tool: str, arguments: dict):
    return run(lambda session: session.call_tool(tool, arguments))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_tool_discovery_lists_milestone1_and_new_tools():
    tools = run(lambda s: s.list_tools()).tools
    names = {t.name for t in tools}
    assert {"read_file", "list_files", "get_file_info", "search_in_file", "write_file"} <= names
    assert {"watch_directory", "batch_process", "server_status"} <= names
    for tool in tools:  # every tool describes itself with a description and a JSON schema
        assert tool.description and tool.inputSchema["type"] == "object"


def test_resource_discovery():
    async def body(session):
        resources = {str(r.uri) for r in (await session.list_resources()).resources}
        templates = {t.uriTemplate for t in (await session.list_resource_templates()).resourceTemplates}
        return resources, templates
    resources, templates = run(body)
    assert {"config://server", "fs://folders"} <= resources
    assert {"fs://files/{folder}", "fs://files/{folder}/{filename}", "fs://info/{folder}/{filename}",
            "fs://search/{folder}/{filename}/{keyword}"} <= templates


# ---------------------------------------------------------------------------
# Milestone 1 tools
# ---------------------------------------------------------------------------

def test_read_file_returns_text_and_metadata():
    result = call("read_file", {"path": RESUME, "max_chars": 50})
    data = result.structuredContent
    assert not result.isError and data["truncated"] and len(data["content"]) == 50
    assert data["metadata"]["extension"] == ".pdf" and data["metadata"]["word_count"] > 100


def test_folder_names_work_as_paths():
    assert call("read_file", {"path": "resumes/chen_martin.pdf"}).structuredContent["path"] == RESUME
    assert call("list_files", {"directory": "resumes", "extension": "pdf"}).structuredContent["count"] == 100


def test_search_in_file():
    data = call("search_in_file", {"path": RESUME, "keyword": "SALES", "max_matches": 2}).structuredContent
    assert data["match_count"] >= 2 and len(data["matches"]) == 2


def test_write_file_only_in_reports(tmp_path):
    ok = call("write_file", {"path": "reports/_pytest_write.md", "content": "# hi"})
    assert not ok.isError
    (server.CONFIG.folder_path("reports") / "_pytest_write.md").unlink()
    denied = call("write_file", {"path": "data/resumes/evil.txt", "content": "x"})
    assert denied.isError and "[ACCESS_DENIED]" in denied.content[0].text


# ---------------------------------------------------------------------------
# Error handling and status codes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path, code", [
    ("../.env", "ACCESS_DENIED"),
    ("/etc/passwd", "ACCESS_DENIED"),
    ("data/resumes/../../.env", "ACCESS_DENIED"),
    ("data/resumes/nobody.pdf", "NOT_FOUND"),
    ("data/resumes", "NOT_A_FILE"),
    ("data/resumes/_labels.json", "UNSUPPORTED_TYPE"),
    ("", "INVALID_ARGUMENT"),
])
def test_tool_errors_have_status_codes(path, code):
    result = call("read_file", {"path": path})
    assert result.isError and f"[{code}]" in result.content[0].text


def test_too_large_files_are_refused(monkeypatch):
    monkeypatch.setattr(server.CONFIG, "max_file_size_mb", 0.001)
    assert "[TOO_LARGE]" in call("read_file", {"path": RESUME}).content[0].text


@pytest.mark.parametrize("uri, jsonrpc_code", [
    ("fs://files/resumes/nobody.pdf", -32002),   # resource not found
    ("fs://files/secret_folder", -32002),
    ("fs://does/not/exist", -32002),
    ("fs://files/resumes/..%2F..%2F.env", -32602),  # invalid params
])
def test_resource_errors_use_jsonrpc_codes(uri, jsonrpc_code):
    async def body(session):
        with pytest.raises(McpError) as error:
            await session.read_resource(uri)
        return error.value.error.code
    assert run(body) == jsonrpc_code


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def test_read_resources():
    async def body(session):
        text = (await session.read_resource("fs://files/resumes/chen_martin.pdf")).contents[0].text
        info = (await session.read_resource("fs://info/resumes/chen_martin.pdf")).contents[0].text
        config = (await session.read_resource("config://server")).contents[0].text
        return text, info, config
    text, info, config = run(body)
    assert "SALES" in text and '"size_bytes"' in info and '"allowed_dirs"' in config


# ---------------------------------------------------------------------------
# batch_process
# ---------------------------------------------------------------------------

def test_batch_process_reads_a_whole_folder():
    data = call("batch_process", {"directory": "resumes", "operation": "info"}).structuredContent
    assert data["total"] == 100 and data["succeeded"] == 100 and data["failed"] == 0


def test_batch_process_keeps_going_when_one_file_fails():
    data = call("batch_process", {"paths": [RESUME, "data/resumes/nobody.pdf", "../.env"], "max_chars": 20}).structuredContent
    assert data["succeeded"] == 1 and data["failed"] == 2
    assert {e["code"] for e in data["errors"]} == {"NOT_FOUND", "ACCESS_DENIED"}


def test_batch_process_search_and_limits(monkeypatch):
    data = call("batch_process", {"paths": [RESUME], "operation": "search", "keyword": "sales"}).structuredContent
    assert data["results"][0]["match_count"] > 0
    assert call("batch_process", {"paths": [RESUME], "operation": "search"}).isError  # no keyword
    monkeypatch.setattr(server.CONFIG, "batch_max_files", 5)
    assert "[TOO_MANY_FILES]" in call("batch_process", {"directory": "resumes"}).content[0].text


# ---------------------------------------------------------------------------
# watch_directory
# ---------------------------------------------------------------------------

def test_watch_directory_detects_a_new_resume(tmp_path, monkeypatch):
    monkeypatch.setitem(server.CONFIG.allowed_dirs, "inbox", str(tmp_path))  # watch an empty temp folder
    (tmp_path / "old.txt").write_text("already here")

    first = call("watch_directory", {"directory": "inbox", "timeout_seconds": 0}).structuredContent
    assert first["started_watching"] and first["new_files"] == []

    # drop a new resume into the folder 1 second from now, while the server is watching
    threading.Timer(1.0, lambda: (tmp_path / "new_applicant.txt").write_text("Jane Doe, SQL, 5 years")).start()
    start = time.monotonic()
    second = call("watch_directory", {"directory": "inbox", "timeout_seconds": 10}).structuredContent
    assert [p.split("/")[-1] for p in second["new_files"]] == ["new_applicant.txt"]
    assert time.monotonic() - start < 5  # returned as soon as the file appeared, not after 10 s

    third = call("watch_directory", {"directory": "inbox", "timeout_seconds": 0}).structuredContent
    assert third["new_files"] == []  # already reported, not reported twice


# ---------------------------------------------------------------------------
# Configuration management
# ---------------------------------------------------------------------------

def test_config_layers(tmp_path, monkeypatch):
    custom = tmp_path / "fs.json"
    custom.write_text('{"max_file_size_mb": 2, "supported_extensions": ["TXT"]}')
    monkeypatch.setenv("MCP_FS_BATCH_MAX_WORKERS", "3")
    config = load_config(custom)
    assert config.max_file_size_mb == 2          # from the JSON file
    assert config.supported_extensions == [".txt"]  # cleaned up
    assert config.batch_max_workers == 3          # from the environment variable
    assert config.watch_max_seconds == 120        # default


def test_config_rejects_unknown_writable_folder(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"writable_dirs": ["everywhere"]}')
    with pytest.raises(ValueError):
        load_config(bad)


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 over stdio (a real subprocess, raw JSON lines)
# ---------------------------------------------------------------------------

def test_jsonrpc_wire_format():
    connection = RawMCPConnection(verbose=False)
    try:
        hello = connection.initialize()
        assert hello["jsonrpc"] == "2.0" and hello["id"] == 1
        assert hello["result"]["serverInfo"]["name"] == "filesystem"
        assert "tools" in hello["result"]["capabilities"] and "resources" in hello["result"]["capabilities"]

        listed = connection.request("tools/list")
        assert listed["jsonrpc"] == "2.0" and "batch_process" in [t["name"] for t in listed["result"]["tools"]]

        called = connection.request("tools/call", {"name": "get_file_info", "arguments": {"path": RESUME}})
        assert called["result"]["isError"] is False
        assert called["result"]["structuredContent"]["name"] == "chen_martin.pdf"

        missing = connection.request("resources/read", {"uri": "fs://files/resumes/nobody.pdf"})
        assert missing["error"]["code"] == -32002 and "result" not in missing

        unknown = connection.request("no/such/method")
        assert "error" in unknown and unknown["id"] == 5
    finally:
        connection.close()
