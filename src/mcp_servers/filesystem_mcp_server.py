"""
filesystem_mcp_server.py - Part A: the file system tools as an MCP server

In Milestone 3 the agent imported our file functions directly:

    from src.file_system import fs_tools
    fs_tools.read_file("data/resumes/chen_wang.pdf")

Now those functions live behind an MCP server. The agent (or Claude Desktop, or any
other MCP client) talks to it with JSON-RPC 2.0 messages, for example:

    --> {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "read_file", "arguments": {"path": "data/resumes/chen_wang.pdf"}}}
    <-- {"jsonrpc": "2.0", "id": 3, "result": {"content": [...], "isError": false}}

What the server offers
  TOOLS (actions the client can call)
    Milestone 1:  read_file, list_files, get_file_info, search_in_file, write_file
    New:          watch_directory  - wait for new resumes to appear in a folder
                  batch_process    - read / inspect / search many files in one call (in parallel)
                  server_status    - health check
  RESOURCES (read-only data the client can discover and read by URI)
    config://server                           the server configuration
    fs://folders                              the folders this server may access
    fs://files/{folder}                       list the files in a folder   (Milestone 1 list_files)
    fs://files/{folder}/{filename}            the text of one file          (Milestone 1 read_file)
    fs://info/{folder}/{filename}             file metadata                 (Milestone 1 get_file_info)
    fs://search/{folder}/{filename}/{keyword} keyword search in one file    (Milestone 1 search_in_file)
  (write_file is only a tool: MCP resources are read-only.)

Safety: every path is checked against `allowed_dirs` in the config, so a client can
never read '../.env' or anything else outside the resume / JD / report folders.

Run it:
    python -m src.mcp_servers.filesystem_mcp_server                         # stdio (default)
    python -m src.mcp_servers.filesystem_mcp_server --transport http --port 8765
    npx @modelcontextprotocol/inspector .venv/bin/python -m src.mcp_servers.filesystem_mcp_server
"""

import argparse
import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal


from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData

from src.file_system import fs_tools  # the Milestone 1 functions - only the SERVER uses them now
from src.mcp_servers.server_config import load_config

CONFIG = load_config()

# JSON-RPC 2.0 error codes for resource requests.
# -32602 / -32603 come from the JSON-RPC spec, -32002 is MCP's "resource not found".
RESOURCE_NOT_FOUND = -32002
JSONRPC_CODES = {
    "NOT_FOUND": RESOURCE_NOT_FOUND,
    "UNKNOWN_RESOURCE": RESOURCE_NOT_FOUND,
    "ACCESS_DENIED": INVALID_PARAMS,
    "INVALID_ARGUMENT": INVALID_PARAMS,
    "UNSUPPORTED_TYPE": INVALID_PARAMS,
    "NOT_A_FILE": INVALID_PARAMS,
    "TOO_LARGE": INVALID_PARAMS,
}


class FilesystemMCP(FastMCP):
    """FastMCP with proper JSON-RPC error codes for resources.

    Plain FastMCP answers every failed resources/read with error code 0. Here we look
    at our status code (e.g. "[NOT_FOUND] ...") and send the matching JSON-RPC code.
    """

    async def read_resource(self, uri):
        try:
            return await super().read_resource(uri)
        except Exception as error:
            message = str(error)
            found = re.search(r"\[([A-Z_]+)\]", message)
            status = found.group(1) if found else ("UNKNOWN_RESOURCE" if "Unknown resource" in message else "")
            if found:
                message = message[found.start():]  # drop FastMCP's "Error creating resource..." prefixes
            raise McpError(ErrorData(code=JSONRPC_CODES.get(status, INTERNAL_ERROR), message=message,
                                     data={"uri": str(uri), "status": status or "INTERNAL_ERROR"})) from error


mcp = FilesystemMCP(
    CONFIG.server_name,
    log_level="WARNING",  # INFO would log every single request
    instructions=(
        "File system access for the resume matching project. Paths are relative to the project root, "
        "e.g. 'data/resumes/chen_wang.pdf'. Folder names like 'resumes' also work for directories."
    ),
)


# ---------------------------------------------------------------------------
# Errors with status codes
# ---------------------------------------------------------------------------
# Tool errors are sent back as a normal result with "isError": true and a message
# that starts with a code, e.g. "[NOT_FOUND] File not found: data/resumes/x.pdf".
# The client turns that into {"success": False, "code": "NOT_FOUND", "error": "..."}.

class FsError(Exception):
    """A file system error with a short status code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def tool_error(error: FsError) -> ToolError:
    return ToolError(str(error))


# ---------------------------------------------------------------------------
# Path safety helpers
# ---------------------------------------------------------------------------

def _is_inside(path: Path, folder: Path) -> bool:
    return path == folder or folder in path.parents


def resolve_path(path: str, for_writing: bool = False) -> Path:
    """Turn a user path into a full path, and make sure it is inside an allowed folder.

    Accepts 'data/resumes/x.pdf' (relative to the project root), 'resumes/x.pdf'
    (folder name from the config) or an absolute path inside an allowed folder.
    """
    if not path or not path.strip():
        raise FsError("INVALID_ARGUMENT", "Path cannot be empty")
    path = path.strip()

    # 'resumes' or 'resumes/x.pdf' -> use the folder from the config
    first_part = path.replace("\\", "/").split("/")[0]
    if first_part in CONFIG.allowed_dirs:
        full = (CONFIG.folder_path(first_part) / path[len(first_part):].lstrip("/\\")).resolve()
    else:
        full = (Path(CONFIG.base_dir) / path).resolve()

    allowed = CONFIG.writable_dirs if for_writing else CONFIG.allowed_dirs
    if not any(_is_inside(full, CONFIG.folder_path(name)) for name in allowed):
        action = "write to" if for_writing else "access"
        raise FsError("ACCESS_DENIED", f"Not allowed to {action} '{path}'. Allowed folders: {list(allowed)}")
    return full


def relative(full: Path) -> str:
    """Full path -> 'data/resumes/x.pdf' (what the client sees)."""
    try:
        return str(full.relative_to(Path(CONFIG.base_dir).resolve()))
    except ValueError:
        return str(full)


def check_readable_file(full: Path, original: str) -> None:
    if not full.exists():
        raise FsError("NOT_FOUND", f"File not found: {original}")
    if not full.is_file():
        raise FsError("NOT_A_FILE", f"{original} is a folder, not a file")
    if full.suffix.lower() not in CONFIG.supported_extensions:
        raise FsError("UNSUPPORTED_TYPE", f"Unsupported file type '{full.suffix}'. "
                                          f"Supported: {CONFIG.supported_extensions}")
    size_mb = full.stat().st_size / 1_000_000
    if size_mb > CONFIG.max_file_size_mb:
        raise FsError("TOO_LARGE", f"{original} is {size_mb:.1f} MB (limit {CONFIG.max_file_size_mb} MB)")


def resolve_folder(directory: str) -> Path:
    full = resolve_path(directory)
    if not full.is_dir():
        raise FsError("NOT_FOUND", f"Folder not found: {directory}")
    return full


# ---------------------------------------------------------------------------
# The plain logic (no MCP here) - each function raises FsError on problems
# ---------------------------------------------------------------------------

def do_read_file(path: str, max_chars: int = 0) -> dict:
    full = resolve_path(path)
    check_readable_file(full, path)
    result = fs_tools.read_file(str(full))  # Milestone 1 code does the actual reading
    if not result["success"]:
        raise FsError("READ_FAILED", result["error"])
    content = result["content"]
    truncated = bool(max_chars) and len(content) > max_chars
    metadata = dict(result["metadata"], path=relative(full))
    return {"success": True, "path": relative(full), "content": content[:max_chars] if truncated else content,
            "truncated": truncated, "metadata": metadata}


def do_get_file_info(path: str) -> dict:
    full = resolve_path(path)
    if not full.exists():
        raise FsError("NOT_FOUND", f"File not found: {path}")
    info = fs_tools.get_file_info(str(full))
    info["path"] = relative(full)
    info["extension"] = full.suffix.lower()
    info["supported"] = full.suffix.lower() in CONFIG.supported_extensions
    return {"success": True, **info}


def do_list_files(directory: str, extension: str = "") -> dict:
    full = resolve_folder(directory)
    files = fs_tools.list_files(str(full), extension or None)
    for f in files:
        f["path"] = relative(Path(f["path"]))
    return {"success": True, "directory": relative(full), "count": len(files), "files": files}


def do_search_in_file(path: str, keyword: str, max_matches: int = 20) -> dict:
    if not keyword or not keyword.strip():
        raise FsError("INVALID_ARGUMENT", "Keyword cannot be empty")
    full = resolve_path(path)
    check_readable_file(full, path)
    result = fs_tools.search_in_file(str(full), keyword)
    if not result["success"]:
        raise FsError("READ_FAILED", result["error"])
    result["file"] = relative(full)
    result["matches"] = result["matches"][:max_matches]
    return result


def do_write_file(path: str, content: str, overwrite: bool = True) -> dict:
    full = resolve_path(path, for_writing=True)
    if full.exists() and not overwrite:
        raise FsError("ALREADY_EXISTS", f"{path} already exists (pass overwrite=true to replace it)")
    if full.suffix.lower() not in (".txt", ".md", ".json"):
        raise FsError("UNSUPPORTED_TYPE", "Only .txt, .md and .json files can be written")
    result = fs_tools.write_file(str(full), content)
    if not result["success"]:
        raise FsError("WRITE_FAILED", result["error"])
    return {"success": True, "path": relative(full), "characters": len(content)}


def run_tool(function, *args, **kwargs) -> dict[str, Any]:
    """Run one of the do_* functions and turn an FsError into an MCP tool error."""
    try:
        return function(*args, **kwargs)
    except FsError as error:
        raise tool_error(error) from error


# ---------------------------------------------------------------------------
# MCP TOOLS - Milestone 1 tools
# The docstring and the type hints become the tool's description and JSON schema.
# ---------------------------------------------------------------------------

@mcp.tool()
def read_file(path: str, max_chars: int = 0) -> dict[str, Any]:
    """Read a resume or job description (.pdf, .docx, .txt, .md) and return its text and metadata.

    path: e.g. 'data/resumes/chen_wang.pdf'. max_chars: cut the text to this length (0 = no limit).
    """
    return run_tool(do_read_file, path, max_chars)


@mcp.tool()
def list_files(directory: str = "resumes", extension: str = "") -> dict[str, Any]:
    """List the files in a folder, e.g. 'data/resumes' or 'resumes'. Optionally only one extension like '.pdf'."""
    return run_tool(do_list_files, directory, extension)


@mcp.tool()
def get_file_info(path: str) -> dict[str, Any]:
    """Return a file's metadata: name, size in bytes, last modified time, extension."""
    return run_tool(do_get_file_info, path)


@mcp.tool()
def search_in_file(path: str, keyword: str, max_matches: int = 20) -> dict[str, Any]:
    """Find the lines in a file that contain a keyword (case-insensitive), with the lines around them."""
    return run_tool(do_search_in_file, path, keyword, max_matches)


@mcp.tool()
def write_file(path: str, content: str, overwrite: bool = True) -> dict[str, Any]:
    """Save text to a .txt / .md / .json file in the reports folder, e.g. 'reports/summary.md'."""
    return run_tool(do_write_file, path, content, overwrite)


# ---------------------------------------------------------------------------
# MCP TOOLS - new capabilities
# ---------------------------------------------------------------------------

# folder path -> {file name: last modified time}. Remembered between calls,
# so each watch_directory call reports what changed since the previous call.
_snapshots: dict[str, dict[str, float]] = {}


def take_snapshot(folder: Path) -> dict[str, float]:
    return {
        p.name: p.stat().st_mtime
        for p in folder.iterdir()
        if p.is_file() and not p.name.startswith((".", "_")) and p.suffix.lower() in CONFIG.supported_extensions
    }


@mcp.tool()
async def watch_directory(directory: str = "resumes", timeout_seconds: float = 10, ctx: Context = None) -> dict[str, Any]:
    """Watch a folder for new resumes. Waits up to timeout_seconds and returns as soon as something changes.

    The first call starts watching. Every later call reports the files that were added, changed
    or removed since the previous call, so files dropped in between calls are never missed.
    """
    try:
        folder = resolve_folder(directory)
    except FsError as error:
        raise tool_error(error) from error

    timeout = max(0.0, min(float(timeout_seconds), CONFIG.watch_max_seconds))
    key = str(folder)
    first_call = key not in _snapshots
    if first_call:
        _snapshots[key] = take_snapshot(folder)
    before = _snapshots[key]

    start = time.monotonic()
    while True:
        now = take_snapshot(folder)
        added = sorted(set(now) - set(before))
        removed = sorted(set(before) - set(now))
        modified = sorted(name for name in now if name in before and now[name] != before[name])
        waited = time.monotonic() - start
        if added or removed or modified or waited >= timeout:
            break
        if ctx:  # progress notifications let the client show "still watching..."
            await ctx.report_progress(waited, timeout, f"Watching {relative(folder)}...")
        await asyncio.sleep(CONFIG.watch_poll_seconds)

    _snapshots[key] = now
    if ctx and (added or removed):
        await ctx.info(f"{len(added)} new / {len(removed)} removed file(s) in {relative(folder)}")
        await ctx.session.send_resource_list_changed()  # tell the client that fs://files/... changed

    return {
        "success": True,
        "directory": relative(folder),
        "started_watching": first_call,
        "waited_seconds": round(waited, 1),
        "new_files": [relative(folder / name) for name in added],
        "modified_files": [relative(folder / name) for name in modified],
        "removed_files": [relative(folder / name) for name in removed],
        "total_files": len(now),
    }


@mcp.tool()
def batch_process(
    paths: list[str] | None = None,
    directory: str = "",
    operation: Literal["read", "info", "search"] = "read",
    keyword: str = "",
    max_chars: int = 0,
) -> dict[str, Any]:
    """Process many files in ONE call, in parallel. Much faster than calling read_file once per file.

    Give either `paths` (a list of files) or `directory` (every supported file in that folder).
    operation: 'read' (text + metadata), 'info' (metadata only) or 'search' (needs `keyword`).
    max_chars: cut each file's text to this length when reading (0 = no limit).
    One bad file does not stop the batch: it is reported in `errors` and the rest still run.
    """
    try:
        if paths:
            targets = list(dict.fromkeys(paths))  # remove duplicates, keep the order
        elif directory:
            files = do_list_files(directory)["files"]
            targets = [f["path"] for f in files if Path(f["path"]).suffix.lower() in CONFIG.supported_extensions]
        else:
            raise FsError("INVALID_ARGUMENT", "Give either 'paths' or 'directory'")
        if len(targets) > CONFIG.batch_max_files:
            raise FsError("TOO_MANY_FILES", f"{len(targets)} files; the limit is {CONFIG.batch_max_files} per batch")
        if operation == "search" and not keyword.strip():
            raise FsError("INVALID_ARGUMENT", "operation 'search' needs a keyword")
    except FsError as error:
        raise tool_error(error) from error

    def process_one(path: str) -> dict:
        try:
            if operation == "read":
                return do_read_file(path, max_chars)
            if operation == "info":
                return do_get_file_info(path)
            return do_search_in_file(path, keyword)
        except FsError as error:
            return {"success": False, "path": path, "code": error.code, "error": error.message}
        except Exception as error:  # an unexpected problem with one file must not break the batch
            return {"success": False, "path": path, "code": "INTERNAL_ERROR", "error": str(error)}

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONFIG.batch_max_workers) as pool:
        outcomes = list(pool.map(process_one, targets))  # map() keeps the input order

    results = [o for o in outcomes if o.get("success")]
    errors = [o for o in outcomes if not o.get("success")]
    return {
        "success": True,
        "operation": operation,
        "total": len(targets),
        "succeeded": len(results),
        "failed": len(errors),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "results": results,
        "errors": errors,
    }


@mcp.tool()
def server_status() -> dict[str, Any]:
    """Health check: server name, accessible folders and how many files each one has."""
    folders = {}
    for name in CONFIG.allowed_dirs:
        folder = CONFIG.folder_path(name)
        folders[name] = {
            "path": CONFIG.allowed_dirs[name],
            "exists": folder.is_dir(),
            "files": len(take_snapshot(folder)) if folder.is_dir() else 0,
            "writable": name in CONFIG.writable_dirs,
        }
    return {"success": True, "status": "ok", "server": CONFIG.server_name, "pid": os.getpid(),
            "folders": folders, "watching": [relative(Path(k)) for k in _snapshots]}


# ---------------------------------------------------------------------------
# MCP RESOURCES - read-only data, found with resources/list and resources/templates/list
# A resource function just raises FsError for a bad request; FilesystemMCP (above)
# turns it into a JSON-RPC error with the right code.
# ---------------------------------------------------------------------------

def resource_json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


def folder_path(folder: str, filename: str = "") -> str:
    """Check the folder name from the URI and build the path, e.g. ('resumes', 'x.pdf') -> 'resumes/x.pdf'."""
    if folder not in CONFIG.allowed_dirs:
        raise FsError("NOT_FOUND", f"Unknown folder '{folder}'. Known folders: {list(CONFIG.allowed_dirs)}")
    if "/" in filename or "\\" in filename or filename.startswith(".."):
        raise FsError("ACCESS_DENIED", f"Invalid file name '{filename}'")
    return f"{folder}/{filename}" if filename else folder


@mcp.resource("config://server", mime_type="application/json")
def server_config() -> str:
    """The server configuration (defaults + JSON file + environment overrides)."""
    return resource_json(CONFIG.to_dict())


@mcp.resource("fs://folders", mime_type="application/json")
def folders() -> str:
    """The folders this server can read (and which ones it can write to)."""
    return resource_json({name: {"path": path, "writable": name in CONFIG.writable_dirs}
                          for name, path in CONFIG.allowed_dirs.items()})


@mcp.resource("fs://files/{folder}", mime_type="application/json")
def folder_listing(folder: str) -> str:
    """List the files in a folder: resumes, job_descriptions or reports."""
    return resource_json(do_list_files(folder_path(folder)))


@mcp.resource("fs://files/{folder}/{filename}", mime_type="text/plain")
def file_content(folder: str, filename: str) -> str:
    """The plain text of one file, e.g. fs://files/resumes/chen_wang.pdf"""
    return do_read_file(folder_path(folder, filename))["content"]


@mcp.resource("fs://info/{folder}/{filename}", mime_type="application/json")
def file_info(folder: str, filename: str) -> str:
    """Metadata of one file, e.g. fs://info/resumes/chen_wang.pdf"""
    return resource_json(do_get_file_info(folder_path(folder, filename)))


@mcp.resource("fs://search/{folder}/{filename}/{keyword}", mime_type="application/json")
def file_search(folder: str, filename: str, keyword: str) -> str:
    """Keyword search in one file, e.g. fs://search/resumes/chen_wang.pdf/excel"""
    return resource_json(do_search_in_file(folder_path(folder, filename), keyword))


# ---------------------------------------------------------------------------
# Start the server
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Filesystem MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                        help="stdio: started by the client (default). http: run as a standalone web service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    # make sure the writable folders exist (e.g. reports/)
    for name in CONFIG.writable_dirs:
        CONFIG.folder_path(name).mkdir(parents=True, exist_ok=True)

    if args.transport == "http":
        mcp.settings.host, mcp.settings.port = args.host, args.port
        mcp.run(transport="streamable-http")   # endpoint: http://host:port/mcp
    else:
        mcp.run(transport="stdio")             # never print() here: stdout carries the JSON-RPC messages


if __name__ == "__main__":
    main()
