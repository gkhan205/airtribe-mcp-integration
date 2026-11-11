"""
candidate_db_mcp_server.py - Part B.2 (bonus): a second MCP server, backed by a database

The filesystem server knows about FILES. This server is the agent's long-term MEMORY:
a small SQLite database (Python's built-in sqlite3, nothing to install) that keeps

    - every screening the agent has run (job, filters, ranked candidates, decisions)
    - notes the recruiter adds about a candidate ("available from January", ...)

So the agent can answer "Which jobs did we screen before?" or "Has Elena Reed been
shortlisted before?" - even after a restart, because the data is saved on disk.

  TOOLS       save_screening, get_screening_history, get_candidate_history, add_candidate_note
  RESOURCES   db://screenings                    the 10 most recent screenings
              db://candidates/{candidate_id}     everything stored about one candidate

Run:  python -m src.mcp_servers.candidate_db_mcp_server
      (the database file can be changed with the CANDIDATE_DB_PATH environment variable)
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from src.config.constants import ROOT

DB_PATH = Path(os.getenv("CANDIDATE_DB_PATH") or ROOT / "data" / "candidates.db")

mcp = FastMCP("candidate_db", log_level="WARNING",
              instructions="Stores past screenings and recruiter notes about candidates.")

SCHEMA = """
CREATE TABLE IF NOT EXISTS screenings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_title   TEXT NOT NULL,
    filters     TEXT NOT NULL,          -- JSON list, e.g. ["years:3", "Linux"]
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS screening_results (
    screening_id  INTEGER NOT NULL REFERENCES screenings(id),
    candidate_id  TEXT NOT NULL,
    name          TEXT NOT NULL,
    rank          INTEGER NOT NULL,
    match_score   REAL,
    decision      TEXT
);
CREATE TABLE IF NOT EXISTS notes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id  TEXT NOT NULL,
    note          TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    """Open the database (and create the tables the first time)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row      # rows behave like dicts: row["name"]
    db.executescript(SCHEMA)
    return db


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def screening_details(db: sqlite3.Connection, row: sqlite3.Row) -> dict:
    results = db.execute(
        "SELECT candidate_id, name, rank, match_score, decision FROM screening_results "
        "WHERE screening_id = ? ORDER BY rank", (row["id"],)
    ).fetchall()
    return {"screening_id": row["id"], "job_title": row["job_title"], "filters": json.loads(row["filters"]),
            "created_at": row["created_at"], "candidates": [dict(r) for r in results]}


def candidate_details(candidate_id: str) -> dict:
    with connect() as db:
        appearances = db.execute(
            "SELECT s.id AS screening_id, s.job_title, s.created_at, r.rank, r.match_score, r.decision "
            "FROM screening_results r JOIN screenings s ON s.id = r.screening_id "
            "WHERE r.candidate_id = ? ORDER BY s.id DESC", (candidate_id,)
        ).fetchall()
        notes = db.execute("SELECT note, created_at FROM notes WHERE candidate_id = ? ORDER BY id",
                           (candidate_id,)).fetchall()
    return {"success": True, "candidate_id": candidate_id,
            "screenings": [dict(a) for a in appearances], "notes": [dict(n) for n in notes]}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def save_screening(job_title: str, filters: list[str], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Save the result of one screening run.

    candidates: [{"candidate_id": "chen_wang", "name": "Chen Wang", "rank": 1, "match_score": 78, "decision": "hire"}, ...]
    """
    if not job_title.strip():
        raise ToolError("[INVALID_ARGUMENT] job_title cannot be empty")
    with connect() as db:   # "with" commits at the end (or rolls back if something fails)
        cursor = db.execute("INSERT INTO screenings (job_title, filters, created_at) VALUES (?, ?, ?)",
                            (job_title, json.dumps(filters), now()))
        screening_id = cursor.lastrowid
        for i, c in enumerate(candidates, 1):
            if "candidate_id" not in c:
                raise ToolError(f"[INVALID_ARGUMENT] candidate #{i} has no candidate_id")
            db.execute(
                "INSERT INTO screening_results VALUES (?, ?, ?, ?, ?, ?)",
                (screening_id, c["candidate_id"], c.get("name", c["candidate_id"]), c.get("rank", i),
                 c.get("match_score"), c.get("decision")),
            )
    return {"success": True, "screening_id": screening_id, "saved_candidates": len(candidates)}


@mcp.tool()
def get_screening_history(job_title: str = "", limit: int = 5) -> dict[str, Any]:
    """The most recent screenings, newest first. job_title filters by a part of the title (e.g. 'teacher')."""
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM screenings WHERE job_title LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{job_title}%", max(1, min(limit, 50))),
        ).fetchall()
        screenings = [screening_details(db, row) for row in rows]
    return {"success": True, "count": len(screenings), "screenings": screenings}


@mcp.tool()
def get_candidate_history(candidate_id: str) -> dict[str, Any]:
    """Every past screening a candidate appeared in (rank, score, decision) plus the recruiter's notes."""
    return candidate_details(candidate_id)


@mcp.tool()
def add_candidate_note(candidate_id: str, note: str) -> dict[str, Any]:
    """Save a recruiter note about a candidate, e.g. 'Available to start in January'."""
    if not note.strip():
        raise ToolError("[INVALID_ARGUMENT] note cannot be empty")
    with connect() as db:
        db.execute("INSERT INTO notes (candidate_id, note, created_at) VALUES (?, ?, ?)",
                   (candidate_id, note.strip(), now()))
    return {"success": True, "candidate_id": candidate_id, "note": note.strip()}


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("db://screenings", mime_type="application/json")
def recent_screenings() -> str:
    """The 10 most recent screenings."""
    return json.dumps(get_screening_history(limit=10), indent=2)


@mcp.resource("db://candidates/{candidate_id}", mime_type="application/json")
def candidate_record(candidate_id: str) -> str:
    """Everything stored about one candidate."""
    return json.dumps(candidate_details(candidate_id), indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
