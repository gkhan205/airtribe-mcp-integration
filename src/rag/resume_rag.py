"""Part A - Resume RAG system: load -> section-aware chunking -> embed -> FAISS + metadata.

Usage (from the project root):
    python -m src.rag.cli build                           # build and save index/ (default model)
    python -m src.rag.cli build --chunker fixed --model sentence-transformers/all-MiniLM-L6-v2 --index-dir index_minilm
    python -m src.rag.cli query "registered nurse with EMR experience" -k 5
    python -m src.rag.cli show data/resumes/chen_wang.pdf      # print extracted metadata + chunks
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

import faiss
import numpy as np

from src.config.constants import (
    CHUNK_OVERLAP, CHUNKERS, CHUNKS_FILE, CONFIG_FILE, DEFAULT_CHUNKER, DEFAULT_K, DEFAULT_MODEL, DEGREES, EDU_LINE_MAX_CHARS,
    EMBED_BATCH_SIZE, FAISS_FILE, FIXED_CHUNK_CHARS, FIXED_CHUNK_OVERLAP, HEADLINE_MAX_CHARS, INDEX_DIR, LABELS_FILE,
    MAX_CAREER_YEARS, MAX_CHUNK_CHARS, MAX_HEADER_LEN, META_FILE, MIN_CHUNK_CHARS, MONTHS, NOT_A_FIELD,
    QUERY_INSTRUCTIONS, QUERY_K, RESUMES_DIR, SECTION_ALIASES, SKILL_YEARS_WINDOW,
)
from src.file_system.file_tools import file_info, list_files, read_file, rel
from .skills import find_skills, skill_positions

# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------
_HEADER_LOOKUP = {alias: canon for canon, aliases in SECTION_ALIASES.items() for alias in aliases}


def _header_section(line: str) -> str | None:
    key = re.sub(r"[:\s]+$", "", line.strip()).lower()
    key = re.sub(r"\s+", " ", key)
    return _HEADER_LOOKUP.get(key)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split resume text into (section, body) pairs using known header lines.

    Text before the first header is labelled 'profile' (title / contact block).
    """
    sections: list[tuple[str, list[str]]] = [("profile", [])]
    for line in text.split("\n"):
        canon = _header_section(line) if len(line) < MAX_HEADER_LEN else None
        if canon:
            sections.append((canon, []))
        else:
            sections[-1][1].append(line)
    out = []
    for name, lines in sections:
        body = _tidy("\n".join(lines))
        if body:
            out.append((name, body))
    return out


def _tidy(text: str) -> str:
    """Re-join the PDF's broken lines: 'City\\n\\n,\\nState\\n|' -> 'City, State |'."""
    text = re.sub(r"\n\s*([,|\-–])\s*\n?", r" \1 ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _window_split(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split long text on line boundaries into <= max_chars pieces with ~overlap chars carried over."""
    if len(text) <= max_chars:
        return [text]
    lines, pieces, cur = text.split("\n"), [], ""
    for line in lines:
        while len(line) > max_chars:  # pathological single long line
            pieces.append((cur + "\n" + line[:max_chars]).strip())
            cur, line = "", line[max_chars - overlap:]
        if len(cur) + len(line) + 1 > max_chars and cur:
            pieces.append(cur.strip())
            cur = cur[-overlap:].split("\n", 1)[-1] if overlap else ""
        cur = f"{cur}\n{line}" if cur else line
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


def chunk_resume(text: str, strategy: str = DEFAULT_CHUNKER) -> list[dict]:
    """Return chunks [{section, text}] using 'section' (header-aware) or 'fixed' (sliding window)."""
    if strategy == "fixed":
        flat = _tidy(text)
        return [{"section": "fixed", "text": t} for t in _window_split(flat, FIXED_CHUNK_CHARS, FIXED_CHUNK_OVERLAP)]
    secs = split_sections(text)
    if len(secs) <= 1:  # no headers detected -> fall back to sliding window
        return [{"section": "body", "text": t} for t in _window_split(_tidy(text))]
    chunks = []
    for name, body in secs:
        for piece in _window_split(body):
            if len(piece) >= MIN_CHUNK_CHARS:
                chunks.append({"section": name, "text": piece})
    return chunks


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------
_MONTH_RE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_POINT = r"(?:(?P<@m>" + _MONTH_RE + r")\s+|(?P<@n>\d{1,2})\s*/\s*)?(?P<@y>(?:19|20)\d{2})"
DATE_RANGE_RE = re.compile(
    _POINT.replace("@", "s") + r"\s*(?:-|–|—|to|until)\s*(?:" + _POINT.replace("@", "e")
    + r"|(?P<cur>current|present|now|today))",
    re.IGNORECASE,
)
YEARS_STATED_RE = re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+\w+){0,3}?\s+experience", re.I)
SKILL_YEARS_RE = re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?)\b", re.I)

def _to_months(month_name: str | None, month_num: str | None, year: str) -> int:
    m = 1
    if month_name:
        m = MONTHS.get(month_name[:3].lower(), 1)
    elif month_num and 1 <= int(month_num) <= 12:
        m = int(month_num)
    return int(year) * 12 + (m - 1)


def extract_experience_years(text: str) -> float:
    """Total experience: merged employment date ranges; falls back to stated 'X years experience'.

    'Current/Present' resolves to the latest year mentioned in the resume (these resumes
    are a few years old, so using today's date would inflate experience).
    """
    flat = re.sub(r"\s+", " ", text)
    years_seen = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", flat)]
    ref_year = min(max(years_seen), date.today().year) if years_seen else date.today().year
    intervals = []
    for m in DATE_RANGE_RE.finditer(flat):
        start = _to_months(m["sm"], m["sn"], m["sy"])
        end = (ref_year * 12 + 11) if m["cur"] else _to_months(m["em"], m["en"], m["ey"])
        if not m["cur"] and not (m["em"] or m["en"]):
            end += 11  # "2012 - 2014" -> through end of 2014
        if 0 <= end - start <= MAX_CAREER_YEARS * 12:
            intervals.append((start, end))
    total = 0
    for s, e in _merge(intervals):
        total += e - s + 1
    from_dates = round(total / 12, 1)
    stated = [int(x) for x in YEARS_STATED_RE.findall(flat) if int(x) <= MAX_CAREER_YEARS]
    return max(from_dates, max(stated, default=0))


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def extract_education(sections: list[tuple[str, str]], text: str) -> dict:
    edu_text = "\n".join(b for s, b in sections if s == "education") or text
    lines = [l.strip() for l in edu_text.split("\n") if l.strip()]
    best = None
    for i, line in enumerate(lines):
        for level, label, pat in DEGREES:
            m = re.search(rf"(?<!\w)(?:{pat})(?!\w)", line, re.I)
            if not m:
                continue
            if not best or level > best["level"]:
                best = {"level": level, "degree": label,
                        "field": _degree_field(line[m.end():] + " " + " ".join(lines[i + 1:i + 3])),
                        "line": line[:EDU_LINE_MAX_CHARS]}
            break
    return best or {"level": 0, "degree": None, "field": None, "line": None}


def _degree_field(after: str) -> str | None:
    """Field of study from text after the degree: 'of Science (BS) : Information Technology , 2005'."""
    after = re.sub(r"^\s*(?:'?s)?\s*(?:degree)?\s*(?:of|in)?\s*(?:science|arts|applied science|fine arts)?\s*(?:\([A-Z.]+\))?",
                   "", after, flags=re.I)
    m = re.search(r"^[\s:,\-–]*(?:in\s+)?([A-Z][A-Za-z&/ ]{2,60}?)\s*(?:,|\d|$|\bUniversity|\bCollege|\bSchool|\bInstitute|\bGPA)",
                  after)
    field = m.group(1).strip() if m else None
    return field if field and field.lower() not in NOT_A_FIELD else None


def extract_skill_years(text: str, skills: list[str]) -> dict[str, int]:
    """'5 years of Python', 'Python (3 yrs)' -> {'Python': 5}. Looks ~60 chars around a year mention."""
    flat = re.sub(r"\s+", " ", text)
    out: dict[str, int] = {}
    for m in SKILL_YEARS_RE.finditer(flat):
        n = int(m.group(1))
        if n > MAX_CAREER_YEARS:
            continue
        before, after = SKILL_YEARS_WINDOW
        lo, hi = max(0, m.start() - before), m.end() + after
        for s in skills:
            if any(lo <= p <= hi for p in skill_positions(flat, s)):
                out[s] = max(out.get(s, 0), n)
    return out


def extract_metadata(text: str, sections: list[tuple[str, str]], name_hint: str | None = None) -> dict:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    headline = lines[0][:HEADLINE_MAX_CHARS] if lines else ""
    skills_text = "\n".join(b for s, b in sections if s in ("skills", "summary"))
    primary = find_skills(skills_text)
    skills = primary + [s for s in find_skills(text) if s not in primary]
    return {
        "name": name_hint or headline.title(),
        "headline": headline,
        "skills": skills,
        "skills_primary": primary,
        "skill_years": extract_skill_years(text, skills),
        "experience_years": extract_experience_years(text),
        "education": extract_education(sections, text),
        "sections": sorted({s for s, _ in sections}),
    }


# ---------------------------------------------------------------------------
# RAG index
# ---------------------------------------------------------------------------
class ResumeRAG:
    """Embeds resume chunks into a FAISS inner-product index (cosine on normalized vectors).

    FAISS stores only vectors; chunk rows and per-resume metadata live in JSON sidecar
    files and are joined on the vector's integer position.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, index_dir: str | Path = INDEX_DIR,
                 chunker: str = DEFAULT_CHUNKER):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.chunker = chunker
        self._model = None
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []
        self.resumes: dict[str, dict] = {}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        prefix = QUERY_INSTRUCTIONS.get(self.model_name, "")
        v = self.model.encode([prefix + text], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype="float32")

    # -- build ---------------------------------------------------------------
    def build(self, resume_dir: str | Path) -> dict:
        t0 = time.perf_counter()
        resume_dir = Path(resume_dir)
        labels_path = resume_dir / LABELS_FILE
        labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}

        self.chunks, self.resumes = [], {}
        for path in list_files(resume_dir):
            text = read_file(path)
            rid = path.stem
            sections = split_sections(text)
            meta = extract_metadata(text, sections, labels.get(path.name, {}).get("name"))
            meta.update(resume_id=rid, resume_path=rel(path), file=file_info(path), n_chars=len(text))
            chunks = chunk_resume(text, self.chunker)
            meta["n_chunks"] = len(chunks)
            self.resumes[rid] = meta
            for i, c in enumerate(chunks):
                self.chunks.append({"chunk_id": f"{rid}#{i}", "resume_id": rid, **c})
        t_parse = time.perf_counter() - t0

        # Section name is prepended so the embedding knows the chunk's role
        texts = [f"{c['section'].title()}: {c['text']}" for c in self.chunks]
        vecs = self.embed_passages(texts)
        t_embed = time.perf_counter() - t0 - t_parse

        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        return {
            "resumes": len(self.resumes), "chunks": len(self.chunks), "dim": int(vecs.shape[1]),
            "parse_s": round(t_parse, 2), "embed_s": round(t_embed, 2),
            "total_s": round(time.perf_counter() - t0, 2),
        }

    # -- persistence -----------------------------------------------------------
    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / FAISS_FILE))
        with open(self.index_dir / CHUNKS_FILE, "w") as f:
            for c in self.chunks:
                f.write(json.dumps(c) + "\n")
        (self.index_dir / META_FILE).write_text(json.dumps(self.resumes, indent=2))
        (self.index_dir / CONFIG_FILE).write_text(json.dumps(
            {"model_name": self.model_name, "chunker": self.chunker, "ntotal": self.index.ntotal}, indent=2))

    @classmethod
    def load(cls, index_dir: str | Path = INDEX_DIR) -> "ResumeRAG":
        index_dir = Path(index_dir)
        cfg = json.loads((index_dir / CONFIG_FILE).read_text())
        rag = cls(cfg["model_name"], index_dir, cfg["chunker"])
        rag.index = faiss.read_index(str(index_dir / FAISS_FILE))
        rag.chunks = [json.loads(l) for l in open(index_dir / CHUNKS_FILE)]
        rag.resumes = json.loads((index_dir / META_FILE).read_text())
        return rag

    # -- retrieval ---------------------------------------------------------------
    def search(self, query: str | np.ndarray, k: int = DEFAULT_K) -> list[dict]:
        """Top-k chunks for a query string (or pre-computed query vector)."""
        qv = self.embed_query(query) if isinstance(query, str) else query
        scores, ids = self.index.search(qv, min(k, self.index.ntotal))
        return [{**self.chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]

    def chunk_vectors(self) -> np.ndarray:
        return self.index.reconstruct_n(0, self.index.ntotal)


def cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="src.rag.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("resume_dir", nargs="?", default=str(RESUMES_DIR))
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--chunker", choices=CHUNKERS, default=DEFAULT_CHUNKER)
    b.add_argument("--index-dir", default=str(INDEX_DIR))
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=QUERY_K)
    q.add_argument("--index-dir", default=str(INDEX_DIR))
    s = sub.add_parser("show")
    s.add_argument("path")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        rag = ResumeRAG(args.model, args.index_dir, args.chunker)
        stats = rag.build(args.resume_dir)
        rag.save()
        print(json.dumps(stats, indent=2))
        print(f"Index saved to {rel(args.index_dir)}/")
    elif args.cmd == "query":
        rag = ResumeRAG.load(args.index_dir)
        for hit in rag.search(args.text, args.k):
            name = rag.resumes[hit["resume_id"]]["name"]
            print(f"{hit['score']:.3f}  {name:18s} [{hit['section']}] {hit['text'][:110]!r}")
    elif args.cmd == "show":
        text = read_file(args.path)
        sections = split_sections(text)
        print(json.dumps(extract_metadata(text, sections), indent=2))
        for c in chunk_resume(text):
            print(f"\n--- [{c['section']}] ({len(c['text'])} chars)\n{c['text'][:300]}")


if __name__ == "__main__":
    cli()
