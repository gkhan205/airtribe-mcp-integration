"""
mcp_client_demo.py - a guided tour of the MCP client and both MCP servers (no LLM needed)

Shows: discovery, Milestone 1 tools, batch_process, watch_directory (a resume arrives
while we wait), error status codes, resources, and the second (database) MCP server.

Run:  python scripts/mcp_client_demo.py            (add --pause 2 to slow it down for a recording)
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mcp_client import MCPClient, MCPClientError  # noqa: E402
from src.mcp_client.client import load_server_config  # noqa: E402

PAUSE = 0.0


def step(title: str) -> None:
    time.sleep(PAUSE)
    print(f"\n\033[1;36m=== {title}\033[0m")


def show(label: str, data) -> None:
    text = data if isinstance(data, str) else json.dumps(data, indent=1)
    lines = text.splitlines()
    print(f"\033[1;33m{label}\033[0m")
    print("\n".join(lines[:14]) + ("\n  ..." if len(lines) > 14 else ""))


def main() -> None:
    global PAUSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to pause between steps")
    PAUSE = parser.parse_args().pause

    step("1. Start + connect to every MCP server in src/config/mcp_servers.json")
    servers = load_server_config()
    servers["candidate_db"]["env"] = {"CANDIDATE_DB_PATH": str(Path(tempfile.mkdtemp()) / "demo.db")}
    start = time.perf_counter()
    mcp = MCPClient(servers=servers)
    print(f"connected in {time.perf_counter() - start:.2f} s")
    for name, info in mcp.describe().items():
        print(f"  {name:13} protocol {info['protocol_version']}")
        print(f"  {'':13} tools:     {', '.join(info['tools'])}")
        print(f"  {'':13} resources: {', '.join(info['resources'])}")

    step("2. Milestone 1 tools, now called over MCP")
    r = mcp.call_tool("read_file", {"path": "data/resumes/chen_martin.pdf", "max_chars": 120})
    show("read_file ->", {"path": r["path"], "content": r["content"], "words": r["metadata"]["word_count"]})
    r = mcp.call_tool("search_in_file", {"path": "resumes/chen_martin.pdf", "keyword": "customer", "max_matches": 2})
    show("search_in_file 'customer' ->", {"match_count": r["match_count"], "first": r["matches"][0]["line"]})
    r = mcp.call_tool("list_files", {"directory": "job_descriptions", "extension": ".txt"})
    show("list_files job_descriptions ->", [f["name"] for f in r["files"]])

    step("3. NEW: batch_process - many files in ONE call, in parallel")
    r = mcp.call_tool("batch_process", {"directory": "resumes", "operation": "info"})
    print(f"info  for {r['total']} resumes: {r['succeeded']} ok, {r['failed']} failed, {r['elapsed_seconds']} s")
    paths = [f"data/resumes/{p.name}" for p in sorted((ROOT / "data/resumes").glob("*.pdf"))[:10]]
    r = mcp.call_tool("batch_process", {"paths": paths, "operation": "read", "max_chars": 2000})
    print(f"read  10 resumes (round 2 of the agent): {r['succeeded']} ok in {r['elapsed_seconds']} s")
    r = mcp.call_tool("batch_process", {"directory": "resumes", "operation": "search", "keyword": "Linux"})
    hits = [x["file"].split("/")[-1] for x in r["results"] if x["match_count"]]
    print(f"search 'Linux' in {r['total']} resumes: found in {len(hits)} -> {', '.join(hits[:5])} ...")
    r = mcp.call_tool("batch_process", {"paths": [paths[0], "data/resumes/nobody.pdf", "../.env"]})
    show("one bad file does not stop the batch ->",
         {"succeeded": r["succeeded"], "failed": r["failed"], "errors": [(e["path"], e["code"]) for e in r["errors"]]})

    step("4. NEW: watch_directory - a new resume arrives while the server is watching")
    mcp.call_tool("watch_directory", {"directory": "resumes", "timeout_seconds": 0})
    print("watching data/resumes (up to 30 s) ... another process drops a resume in 4 s")
    dropper = subprocess.Popen([sys.executable, str(ROOT / "scripts/drop_new_resume.py"), "--delay", "4"],
                               stdout=subprocess.DEVNULL)
    try:
        r = mcp.call_tool("watch_directory", {"directory": "resumes", "timeout_seconds": 30})
        show("watch_directory ->", {k: r[k] for k in ("waited_seconds", "new_files", "total_files")})
        print("progress notifications received:", len(mcp.progress_log), "| last:", mcp.progress_log[-1])
        r = mcp.call_tool("batch_process", {"paths": r["new_files"], "max_chars": 160})
        for item in r["results"]:
            print(f"  {item['path']}: {item['content'][:90]!r}")
    finally:
        dropper.wait()
        subprocess.run([sys.executable, str(ROOT / "scripts/drop_new_resume.py"), "--cleanup"], stdout=subprocess.DEVNULL)
        mcp.call_tool("watch_directory", {"directory": "resumes", "timeout_seconds": 0})

    step("5. Error handling: status codes, the sandbox blocks everything outside the allowed folders")
    for path in ["../.env", "/etc/passwd", "data/resumes/nobody.pdf", "data/resumes/_labels.json"]:
        r = mcp.call_tool("read_file", {"path": path})
        print(f"  read_file {path:28} -> {r['code']:17} {r['error'][:60]}")
    r = mcp.call_tool("write_file", {"path": "data/resumes/hack.txt", "content": "x"})
    print(f"  write_file data/resumes/hack.txt  -> {r['code']}")

    step("6. Resources (read-only data, discovered with resources/list)")
    show("fs://info/resumes/chen_martin.pdf ->", mcp.read_resource("fs://info/resumes/chen_martin.pdf"))
    for uri in ["fs://files/resumes/nobody.pdf", "fs://files/resumes/..%2F..%2F.env"]:
        try:
            mcp.read_resource(uri)
        except MCPClientError as error:
            print(f"  {uri} -> {error}")

    step("7. BONUS: second MCP server (candidate_db, SQLite)")
    mcp.call_tool("save_screening", {"job_title": "IT Systems Administrator", "filters": ["years:3", "Linux"], "candidates": [
        {"candidate_id": "elena_reed", "name": "Elena Reed", "rank": 1, "match_score": 78, "decision": "hire"},
        {"candidate_id": "ines_brown", "name": "Ines Brown", "rank": 2, "match_score": 74, "decision": "borderline"}]})
    mcp.call_tool("add_candidate_note", {"candidate_id": "elena_reed", "note": "Available from January"})
    show("get_candidate_history elena_reed ->", mcp.call_tool("get_candidate_history", {"candidate_id": "elena_reed"}))

    mcp.close()
    print("\n\033[1;32mDone - both servers closed.\033[0m")


if __name__ == "__main__":
    main()
