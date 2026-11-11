"""File system tools (Milestone 1): list, inspect and read resume / JD files.

Supported formats: .pdf (pypdf), .docx (python-docx), .txt / .md (plain text).
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from src.config.constants import ROOT, SUPPORTED_EXTS


def list_files(directory: str | Path, exts: tuple[str, ...] = SUPPORTED_EXTS) -> list[Path]:
    """Return supported files in `directory` (non-recursive), sorted, skipping hidden/_ files."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(d)
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith((".", "_"))
    )


def file_info(path: str | Path) -> dict:
    p = Path(path)
    st = p.stat()
    return {
        "path": str(p),
        "name": p.name,
        "ext": p.suffix.lower(),
        "size_bytes": st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def rel(path: str | Path) -> str:
    """Path relative to the project root when possible (e.g. 'resumes/john_doe.pdf')."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_pdf(p: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(p))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(p: Path) -> str:
    import docx

    return "\n".join(par.text for par in docx.Document(str(p)).paragraphs)


def _clean(text: str) -> str:
    text = (text.replace("\u00c2\u00b7", "\u2022").replace("\u00c2", " ").replace("\u00a0", " ")
            .replace("\uf0b7", "\u2022").replace("\u00b7", "\u2022"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def read_file(path: str | Path) -> str:
    """Read a document and return cleaned plain text."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        raw = _read_pdf(p)
    elif ext == ".docx":
        raw = _read_docx(p)
    elif ext in (".txt", ".md"):
        raw = p.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return _clean(raw)


if __name__ == "__main__":
    import sys

    from src.config.constants import RESUMES_DIR

    target = sys.argv[1] if len(sys.argv) > 1 else str(RESUMES_DIR)
    if os.path.isdir(target):
        for f in list_files(target):
            print(file_info(f))
    else:
        print(read_file(target))
