"""
test_mcp_client.py - the MCP client talking to BOTH servers over stdio (real subprocesses, no LLM)

Run:  pytest tests/test_mcp_client.py -v
"""

import pytest

from src.mcp_client import MCPClient, MCPClientError
from src.mcp_client.client import load_server_config, parse_tool_error

RESUME = "data/resumes/chen_martin.pdf"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A client connected to both servers. The candidate database is a temporary file."""
    servers = load_server_config()
    servers["candidate_db"]["env"] = {"CANDIDATE_DB_PATH": str(tmp_path_factory.mktemp("db") / "test.db")}
    mcp = MCPClient(servers=servers)
    yield mcp
    mcp.close()


def test_connects_to_every_server(client):
    assert set(client.sessions) == {"filesystem", "candidate_db"}
    assert client.server_info["filesystem"]["protocol_version"]


def test_discovers_tools_from_both_servers(client):
    assert client.tool_server["read_file"] == "filesystem"
    assert client.tool_server["batch_process"] == "filesystem"
    assert client.tool_server["save_screening"] == "candidate_db"


def test_call_tool_success_and_errors(client):
    assert client.call_tool("read_file", {"path": RESUME})["content"]
    assert client.call_tool("read_file", {"path": "../.env"}) == {
        "success": False, "code": "ACCESS_DENIED",
        "error": "Not allowed to access '../.env'. Allowed folders: ['resumes', 'job_descriptions', 'reports']",
    }
    assert client.call_tool("no_such_tool")["code"] == "UNKNOWN_TOOL"


def test_parse_tool_error():
    assert parse_tool_error("Error executing tool x: [NOT_FOUND] gone")["code"] == "NOT_FOUND"
    assert parse_tool_error("something odd")["code"] == "TOOL_ERROR"


def test_resources_across_servers(client):
    uris = {r["uri"] for r in client.list_resources()}
    assert {"fs://files/{folder}/{filename}", "db://screenings", "db://candidates/{candidate_id}"} <= uris
    assert "SALES" in client.read_resource("fs://files/resumes/chen_martin.pdf")
    with pytest.raises(MCPClientError, match="-32002"):
        client.read_resource("fs://files/resumes/nobody.pdf")
    with pytest.raises(MCPClientError):
        client.read_resource("web://nothing")


def test_mcp_tools_become_langchain_tools(client):
    tools = {t.name: t for t in client.langchain_tools(["read_file", "search_in_file", "not_a_tool"])}
    assert set(tools) == {"read_file", "search_in_file"}
    assert "path" in tools["read_file"].args            # schema came from the MCP server
    result = tools["search_in_file"].invoke({"path": RESUME, "keyword": "sales"})
    assert result["match_count"] > 0


def test_candidate_db_server_remembers_screenings_and_notes(client):
    saved = client.call_tool("save_screening", {
        "job_title": "IT Systems Administrator", "filters": ["years:3"],
        "candidates": [{"candidate_id": "elena_reed", "name": "Elena Reed", "rank": 1, "match_score": 78, "decision": "hire"}],
    })
    assert saved["success"]
    client.call_tool("add_candidate_note", {"candidate_id": "elena_reed", "note": "Available from January"})

    history = client.call_tool("get_screening_history", {"job_title": "systems"})
    assert history["screenings"][0]["candidates"][0]["decision"] == "hire"
    person = client.call_tool("get_candidate_history", {"candidate_id": "elena_reed"})
    assert person["notes"][0]["note"] == "Available from January" and person["screenings"][0]["rank"] == 1
    assert "Elena Reed" in client.read_resource("db://screenings")
    assert client.call_tool("add_candidate_note", {"candidate_id": "x", "note": "  "})["code"] == "INVALID_ARGUMENT"


def test_watch_directory_sends_progress_notifications(client):
    client.call_tool("watch_directory", {"directory": "reports", "timeout_seconds": 0})
    result = client.call_tool("watch_directory", {"directory": "reports", "timeout_seconds": 2})
    assert result["success"] and result["waited_seconds"] >= 2
    assert any("Watching reports" in line for line in client.progress_log)
