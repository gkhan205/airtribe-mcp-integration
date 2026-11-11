"""
jsonrpc_demo.py - talk to the filesystem MCP server with RAW JSON-RPC 2.0 messages (no MCP library)

This shows what really travels between the agent and the server. Each message is one
line of JSON on the server's stdin; each answer is one line on its stdout.

Run:  python scripts/jsonrpc_demo.py              (add --pause 3 to slow it down for a recording)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RawMCPConnection:
    """Starts the server as a subprocess and sends/receives JSON-RPC lines."""

    def __init__(self, module: str = "src.mcp_servers.filesystem_mcp_server", verbose: bool = True):
        self.verbose = verbose
        self.next_id = 1
        self.process = subprocess.Popen(
            [sys.executable, "-m", module], cwd=ROOT, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def send_raw(self, line: str) -> None:
        if self.verbose:
            print(f"\n--> {line}")
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def receive(self) -> dict:
        message = json.loads(self.process.stdout.readline())
        if self.verbose:
            text = json.dumps(message)
            print(f"<-- {text[:600]}{' ...' if len(text) > 600 else ''}")
        return message

    def request(self, method: str, params: dict | None = None) -> dict:
        """Send a request (it has an id) and wait for the response with the same id."""
        message = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            message["params"] = params
        self.next_id += 1
        self.send_raw(json.dumps(message))
        while True:
            reply = self.receive()
            if reply.get("id") == message["id"]:   # skip notifications (they have no id)
                return reply

    def notify(self, method: str) -> None:
        """Send a notification (no id, so no response is expected)."""
        self.send_raw(json.dumps({"jsonrpc": "2.0", "method": method}))

    def initialize(self) -> dict:
        reply = self.request("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "raw-jsonrpc-demo", "version": "1.0"},
        })
        self.notify("notifications/initialized")
        return reply

    def close(self) -> None:
        self.process.stdin.close()
        self.process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to pause between steps")
    pause = parser.parse_args().pause
    sys.stdout.reconfigure(line_buffering=True)  # show each line at once, even when piped

    def section(title: str) -> None:
        time.sleep(pause)
        print(f"\n=== {title}")

    server = RawMCPConnection()
    section("1. Handshake: initialize")
    server.initialize()

    section("2. Tool discovery: tools/list")
    tools = server.request("tools/list")
    print("    tools:", [t["name"] for t in tools["result"]["tools"]])

    section("3. Resource discovery: resources/list and resources/templates/list")
    server.request("resources/list")
    server.request("resources/templates/list")

    section("4. Call a tool: tools/call read_file")
    server.request("tools/call", {"name": "read_file",
                                  "arguments": {"path": "data/resumes/chen_martin.pdf", "max_chars": 120}})

    section("5. Read a resource: resources/read")
    server.request("resources/read", {"uri": "fs://info/resumes/chen_martin.pdf"})

    section("6. Errors")
    print("    a) tool error -> normal result with isError: true and a status code")
    server.request("tools/call", {"name": "read_file", "arguments": {"path": "../.env"}})
    print("    b) missing resource -> JSON-RPC error -32002")
    server.request("resources/read", {"uri": "fs://files/resumes/nobody.pdf"})
    print("    c) unknown method -> JSON-RPC error response (the MCP SDK answers -32602 'Invalid request parameters')")
    server.request("files/delete_everything")

    server.close()


if __name__ == "__main__":
    main()
