"""
server_config.py - configuration management for the filesystem MCP server

Settings are loaded in three layers. A later layer overrides an earlier one:

    1. DEFAULTS                          (in this file)
    2. src/config/filesystem_server.json (or the file named by MCP_FS_CONFIG)
    3. environment variables             (MCP_FS_MAX_FILE_SIZE_MB=5, ...)

So you can change the server's behaviour without touching the code, e.g.
    MCP_FS_WATCH_MAX_SECONDS=30 python -m src.mcp_servers.filesystem_mcp_server
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config.constants import ROOT

CONFIG_FILE = ROOT / "src" / "config" / "filesystem_server.json"

DEFAULTS = {
    "server_name": "filesystem",
    # folder name -> path (relative to the project root). The server can ONLY touch files in these folders.
    "allowed_dirs": {"resumes": "data/resumes", "job_descriptions": "data/job_descriptions", "reports": "reports"},
    "writable_dirs": ["reports"],          # write_file only works in these folders
    "supported_extensions": [".pdf", ".docx", ".txt", ".md"],
    "max_file_size_mb": 10,
    "batch_max_files": 200,
    "batch_max_workers": 8,
    "watch_max_seconds": 120,
    "watch_poll_seconds": 1.0,
}

# environment variable -> (setting name, type)
ENV_OVERRIDES = {
    "MCP_FS_MAX_FILE_SIZE_MB": ("max_file_size_mb", float),
    "MCP_FS_BATCH_MAX_FILES": ("batch_max_files", int),
    "MCP_FS_BATCH_MAX_WORKERS": ("batch_max_workers", int),
    "MCP_FS_WATCH_MAX_SECONDS": ("watch_max_seconds", float),
    "MCP_FS_WATCH_POLL_SECONDS": ("watch_poll_seconds", float),
}


@dataclass
class ServerConfig:
    server_name: str
    allowed_dirs: dict[str, str]
    writable_dirs: list[str]
    supported_extensions: list[str]
    max_file_size_mb: float
    batch_max_files: int
    batch_max_workers: int
    watch_max_seconds: float
    watch_poll_seconds: float
    base_dir: str = field(default=str(ROOT))
    config_file: str = ""

    def folder_path(self, name: str) -> Path:
        """'resumes' -> /full/path/to/data/resumes"""
        return (Path(self.base_dir) / self.allowed_dirs[name]).resolve()

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> ServerConfig:
    """Build the server configuration: defaults -> JSON file -> environment variables."""
    settings = dict(DEFAULTS)

    config_file = Path(path or os.getenv("MCP_FS_CONFIG") or CONFIG_FILE)
    if config_file.is_file():
        with open(config_file, encoding="utf-8") as f:
            settings.update(json.load(f))

    for env_name, (key, convert) in ENV_OVERRIDES.items():
        if os.getenv(env_name):
            settings[key] = convert(os.environ[env_name])

    # Make sure every extension starts with a dot and is lower case
    settings["supported_extensions"] = [
        ext.lower() if ext.startswith(".") else "." + ext.lower() for ext in settings["supported_extensions"]
    ]
    unknown = [name for name in settings["writable_dirs"] if name not in settings["allowed_dirs"]]
    if unknown:
        raise ValueError(f"writable_dirs {unknown} must also be listed in allowed_dirs")

    return ServerConfig(**settings, config_file=str(config_file) if config_file.is_file() else "")
