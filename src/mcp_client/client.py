"""
client.py - Part B: the agent's MCP client

The agent no longer imports file functions. It asks MCP servers instead:

    from src.mcp_client import get_mcp_client
    mcp = get_mcp_client()                      # starts + connects to every server in mcp_servers.json
    mcp.call_tool("read_file", {"path": "data/resumes/chen_wang.pdf"})
    mcp.read_resource("fs://info/resumes/chen_wang.pdf")
    mcp.langchain_tools(["read_file", "search_in_file"])   # MCP tools -> LangChain tools for Claude

Which servers to start is configured in src/config/mcp_servers.json, in the same format
Claude Desktop uses. A server is either started as a subprocess ("command" + "args",
talking JSON-RPC over stdin/stdout) or reached over HTTP ("url").

How it works (you don't need this to use it):
The MCP Python SDK is async, but our LangGraph agent is normal (sync) code. So the client
runs an asyncio event loop in a background thread and keeps ONE session per server open
there. The sync methods below hand work to that loop and wait for the answer. Keeping the
session open means the server starts once, not once per tool call.
"""

import asyncio
import atexit
import json
import re
import sys
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError

from src.config.constants import MCP_CONFIG_FILE, MCP_CONNECT_TIMEOUT, MCP_TOOL_TIMEOUT, ROOT


class MCPClientError(Exception):
    """Could not connect to a server, or a request failed at the protocol level."""


def load_server_config(config_file: str | Path = MCP_CONFIG_FILE) -> dict[str, dict]:
    """Read {"mcpServers": {"filesystem": {"command": ..., "args": [...]}, ...}}."""
    with open(config_file, encoding="utf-8") as f:
        return json.load(f)["mcpServers"]


def parse_tool_error(text: str) -> dict:
    """'[NOT_FOUND] File not found: x.pdf' -> {"success": False, "code": "NOT_FOUND", "error": "File not found: x.pdf"}"""
    found = re.search(r"\[([A-Z_]+)\]\s*(.*)", text, re.DOTALL)
    if found:
        return {"success": False, "code": found.group(1), "error": found.group(2).strip()}
    return {"success": False, "code": "TOOL_ERROR", "error": text}


class MCPClient:
    """Connects to several MCP servers and lets normal (sync) code use their tools and resources."""

    def __init__(self, config_file: str | Path = MCP_CONFIG_FILE, servers: dict[str, dict] | None = None):
        self.server_config = servers if servers is not None else load_server_config(config_file)
        self.sessions: dict[str, ClientSession] = {}    # server name -> open session
        self.server_info: dict[str, dict] = {}          # server name -> name/version/capabilities
        self.tools: dict[str, Any] = {}                 # tool name -> MCP Tool (name, description, inputSchema)
        self.tool_server: dict[str, str] = {}           # tool name -> server name
        self._resources: list[dict] | None = None       # filled the first time list_resources() runs
        self.progress_log: list[str] = []               # progress notifications sent by servers (newest last)

        # the background event loop that owns all the async MCP sessions
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
        self._thread.start()
        self._stop = asyncio.Event()
        self._session_tasks = []

        try:
            for name in self.server_config:
                self._connect(name)
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------------
    # Connecting
    # ------------------------------------------------------------------

    def _run(self, coroutine, timeout: float | None = MCP_TOOL_TIMEOUT):
        """Run a coroutine on the background loop and wait for its result."""
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout)
        except FutureTimeout:
            future.cancel()
            raise MCPClientError(f"MCP request timed out after {timeout} s") from None

    def _transport(self, server: dict):
        """stdio (start the server as a subprocess) or HTTP (connect to a running server)."""
        if "url" in server:
            return streamablehttp_client(server["url"])
        command = server["command"]
        if command in ("python", "python3"):
            command = sys.executable  # the same Python (and virtual env) the agent runs in
        params = StdioServerParameters(
            command=command,
            args=server.get("args", []),
            env={**get_default_environment(), **server.get("env", {})},
            cwd=server.get("cwd", str(ROOT)),
        )
        return stdio_client(params)

    async def _keep_session_open(self, name: str, server: dict, ready: Future) -> None:
        """Open the connection, say hello (initialize), then wait until close() is called."""
        try:
            async with self._transport(server) as streams:
                read_stream, write_stream = streams[0], streams[1]
                async with ClientSession(read_stream, write_stream) as session:
                    hello = await session.initialize()   # JSON-RPC "initialize" handshake
                    self.sessions[name] = session
                    self.server_info[name] = {
                        "name": hello.serverInfo.name,
                        "version": hello.serverInfo.version,
                        "protocol_version": hello.protocolVersion,
                        "capabilities": hello.capabilities.model_dump(exclude_none=True),
                    }
                    ready.set_result(True)
                    await self._stop.wait()
        except Exception as error:
            if not ready.done():
                ready.set_exception(error)
        finally:
            self.sessions.pop(name, None)

    def _connect(self, name: str) -> None:
        ready = Future()
        task = asyncio.run_coroutine_threadsafe(self._keep_session_open(name, self.server_config[name], ready), self._loop)
        self._session_tasks.append(task)
        try:
            ready.result(timeout=MCP_CONNECT_TIMEOUT)
        except Exception as error:
            raise MCPClientError(f"Could not connect to MCP server '{name}': {error}") from error

        # tool discovery: tools/list
        for tool in self._run(self.sessions[name].list_tools()).tools:
            self.tools[tool.name] = tool
            self.tool_server[tool.name] = name

    def close(self) -> None:
        """Close every session (this also stops the server subprocesses)."""
        if not self._loop.is_running():
            return
        self._loop.call_soon_threadsafe(self._stop.set)
        for task in self._session_tasks:
            try:
                task.result(timeout=10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def call_tool(self, name: str, arguments: dict | None = None, timeout: float = MCP_TOOL_TIMEOUT) -> dict:
        """Call a tool on whichever server has it. Always returns a dict, never raises for tool errors.

        Success: the tool's result, e.g. {"success": True, "content": "..."}
        Failure: {"success": False, "code": "NOT_FOUND", "error": "File not found: ..."}
        """
        server = self.tool_server.get(name)
        if server is None:
            return {"success": False, "code": "UNKNOWN_TOOL", "error": f"No MCP server has a tool called '{name}'"}
        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            # e.g. watch_directory reports "Watching data/resumes..." while it waits
            self.progress_log = self.progress_log[-49:] + [f"{name}: {message or ''} ({progress:.0f}/{total or '?'})"]

        try:
            result = self._run(self.sessions[server].call_tool(name, arguments or {}, progress_callback=on_progress),
                               timeout=timeout)
        except McpError as error:            # a JSON-RPC error, e.g. -32602 invalid params
            return {"success": False, "code": "PROTOCOL_ERROR", "jsonrpc_code": error.error.code,
                    "error": error.error.message}
        except MCPClientError as error:
            return {"success": False, "code": "TIMEOUT", "error": str(error)}

        text = "\n".join(block.text for block in result.content if block.type == "text")
        if result.isError:
            return parse_tool_error(text)
        if result.structuredContent is not None:
            return result.structuredContent
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"success": True, "text": text}

    def langchain_tools(self, names: list[str] | None = None) -> list[StructuredTool]:
        """Turn MCP tools into LangChain tools, so Claude can call them like any other tool.

        The MCP tool's description and JSON schema are passed on unchanged - that is the
        "standard interface": the server describes its tools once, every client can use them.
        """
        selected = names if names is not None else list(self.tools)
        return [self._to_langchain(self.tools[n]) for n in selected if n in self.tools]

    def _to_langchain(self, tool) -> StructuredTool:
        def run(**arguments):
            return self.call_tool(tool.name, arguments)

        return StructuredTool(
            name=tool.name,
            description=tool.description or tool.name,
            args_schema=tool.inputSchema,   # the JSON schema from tools/list
            func=run,
            metadata={"mcp_server": self.tool_server[tool.name]},
        )

    # ------------------------------------------------------------------
    # Resources (resource discovery: resources/list and resources/templates/list)
    # ------------------------------------------------------------------

    def list_resources(self, refresh: bool = False) -> list[dict]:
        """All resources and resource templates of all servers (asked once, then remembered)."""
        if self._resources is not None and not refresh:
            return self._resources
        found = []
        for server, session in self.sessions.items():
            if not self.server_info[server]["capabilities"].get("resources"):
                continue
            for r in self._run(session.list_resources()).resources:
                found.append({"server": server, "uri": str(r.uri), "name": r.name, "description": r.description or ""})
            for t in self._run(session.list_resource_templates()).resourceTemplates:
                found.append({"server": server, "uri": t.uriTemplate, "name": t.name,
                              "description": t.description or "", "template": True})
        self._resources = found
        return found

    def read_resource(self, uri: str) -> str:
        """Read a resource by URI, e.g. 'fs://files/resumes/chen_wang.pdf'. Raises MCPClientError if it fails."""
        server = self._server_for_uri(uri)
        try:
            result = self._run(self.sessions[server].read_resource(uri))
        except McpError as error:
            raise MCPClientError(f"JSON-RPC error {error.error.code}: {error.error.message}") from error
        return "\n".join(getattr(c, "text", "") for c in result.contents)

    def _server_for_uri(self, uri: str) -> str:
        """Pick the server by the URI scheme: fs:// and config:// -> filesystem, db:// -> candidate_db."""
        scheme = uri.split("://")[0]
        for resource in self.list_resources():
            if resource["uri"].split("://")[0] == scheme:
                return resource["server"]
        raise MCPClientError(f"No MCP server offers resources for '{scheme}://'")

    def describe(self) -> dict:
        """Everything the client discovered - handy for the UI and the demo."""
        return {
            name: {
                **self.server_info[name],
                "tools": [t for t, s in self.tool_server.items() if s == name],
                "resources": [r["uri"] for r in self.list_resources() if r["server"] == name],
            }
            for name in self.sessions
        }


# ---------------------------------------------------------------------------
# One shared client for the whole program
# ---------------------------------------------------------------------------

_client: MCPClient | None = None
_lock = threading.Lock()


def get_mcp_client() -> MCPClient:
    """Return the shared MCP client, connecting to the servers the first time."""
    global _client
    with _lock:
        if _client is None:
            _client = MCPClient()
            atexit.register(_client.close)
        return _client
