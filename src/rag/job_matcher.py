"""Part B - Job matching engine: hybrid (semantic + keyword) retrieval, 0-100 scoring,
must-have filtering and rule-based match reasoning.

Usage (from the project root):
    python -m src.rag.cli match data/job_descriptions/staff_accountant.txt
    python -m src.rag.cli match data/job_descriptions/*.txt --out outputs/
    python -m src.rag.cli match data/job_descriptions/it_systems_admin.txt --must "Python:5" --must "Linux"
    python -m src.rag.cli match --text "Registered nurse with EMR and patient care experience" --k 5
    python -m src.rag.cli match data/job_descriptions/teacher.txt --mode semantic   # ablation: semantic | keyword | hybrid
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.config.constants import (
    BM25_STOPWORDS, CANDIDATE_CHUNKS, DEFAULT_K, DEFAULT_WEIGHTS, DEGREE_FIT_LOWER, DEGREE_FIT_NONE, DEGREES,
    EXCERPT_MAX_CHARS, EXCERPT_MIN_LINE_CHARS, EXP_FIT_WEIGHT, KEYWORD_MODE_BM25_WEIGHT, INDEX_DIR, JD_MIN_DEGREE_LEVEL, JD_SKILL_YEARS_WINDOW,
    MODERATE_SEMANTIC, MODES, MUST_SKILL_BOOST, MUST_SKILL_WEIGHT, N_EXCERPTS, PREFERRED_SKILL_WEIGHT,
    REASONING_MAX_MISSING, REASONING_MAX_SKILLS, REQUIRED_SKILL_WEIGHT, SEMANTIC_CEIL, SEMANTIC_FLOOR,
    SEMANTIC_MAX_WEIGHT, SEMANTIC_TOP_N, STRONG_SEMANTIC,
)
from src.file_system.file_tools import read_file, rel
from .resume_rag import ResumeRAG
from .skills import find_skills, normalize_skill


@dataclass
class Requirement:
    kind: str                 # "skill" | "skill_years" | "experience" | "degree"
    skill: str | None = None
    years: float | None = None
    level: int | None = None  # degree level (see constants.DEGREES)
    source: str = "jd"        # "jd" (auto-detected) | "cli"
    text: str = ""

    def describe(self) -> str:
        if self.kind == "skill_years":
            return f"{self.years:g}+ years {self.skill}"
        if self.kind == "experience":
            return f"{self.years:g}+ years experience"
        if self.kind == "degree":
            return f"degree level >= {self.level}"
        return self.skill or self.text


@dataclass
class JobProfile:
    title: str
    required_skills: list[str]
    preferred_skills: list[str]
    must_haves: list[Requirement] = field(default_factory=list)
    min_years: float | None = None
    degree_level: int | None = None

    @property
    def must_skills(self) -> list[str]:
        return [r.skill for r in self.must_haves if r.skill]


_YEARS_RE = re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?)", re.I)
_MUST_RE = re.compile(r"\bmust[- ]haves?\b|\bmust\b|\brequired\b|\bmandatory\b", re.I)
_PREF_RE = re.compile(r"nice to have|preferred|a plus|\bplus\b|bonus", re.I)


def parse_job_description(text: str) -> JobProfile:
    """Extract skills, must-have requirements, minimum experience and degree from a JD."""
    lines = [l.strip(" -•\t") for l in text.split("\n") if l.strip()]
    title = lines[0] if lines else "Job"
    # each ';' clause is judged on its own: "Bachelor's in Accounting; CPA a plus"
    clauses = [c.strip() for l in lines[1:] for c in l.split(";") if c.strip()]
    required, preferred, musts = [], [], []
    min_years = degree_level = None

    for line in clauses:
        skills = find_skills(line)
        is_must = bool(_MUST_RE.search(line)) and not re.search(r"portfolio required|certification required", line, re.I)
        is_pref = bool(_PREF_RE.search(line)) and not is_must
        years = _YEARS_RE.search(line)

        if is_pref:
            preferred += [s for s in skills if s not in required]
        else:
            required += [s for s in skills if s not in required]

        if years:
            n = float(years.group(1))
            # skills mentioned right after "N years of ..." are what the years apply to
            tail_skills = find_skills(line[years.end():years.end() + JD_SKILL_YEARS_WINDOW])
            if is_must and tail_skills:
                musts += [Requirement("skill_years", s, n, text=line) for s in tail_skills]
            elif is_must:
                musts.append(Requirement("experience", years=n, text=line))
            elif not is_pref:
                min_years = max(min_years or 0, n)
        if is_must:
            have = {r.skill for r in musts}
            musts += [Requirement("skill", s, text=line) for s in skills if s not in have]

        for level, _label, pat in DEGREES:
            if level >= JD_MIN_DEGREE_LEVEL and re.search(rf"(?<!\w)(?:{pat})(?!\w)", line, re.I) and not is_pref:
                degree_level = max(degree_level or 0, level)
                break

    exp_musts = [r.years for r in musts if r.kind in ("experience", "skill_years")]
    if exp_musts:
        min_years = max(min_years or 0, *exp_musts)
    preferred = [s for s in dict.fromkeys(preferred) if s not in required]
    return JobProfile(title, required, preferred, musts, min_years, degree_level)


def parse_cli_must(spec: str) -> Requirement:
    """'Python:5' -> 5+ years Python, 'SQL' -> must know SQL, 'years:5' -> 5+ years overall."""
    name, _, yrs = spec.partition(":")
    if name.strip().lower() in ("years", "experience", "exp"):
        return Requirement("experience", years=float(yrs), source="cli", text=spec)
    skill = normalize_skill(name)
    if not skill:
        raise SystemExit(f"Unknown skill in --must: {name!r} (see skills.py)")
    if yrs:
        return Requirement("skill_years", skill, float(yrs), source="cli", text=spec)
    return Requirement("skill", skill, source="cli", text=spec)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9+#/.]+", text.lower()) if t not in BM25_STOPWORDS and len(t) > 1]


class JobMatcher:
    def __init__(self, rag: ResumeRAG, weights: dict[str, float] | None = None):
        self.rag = rag
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        rag.embed_query("warm-up")  # load the embedding model now so per-query latency is honest
        self.vectors = rag.chunk_vectors()
        self.bm25 = BM25Okapi([_tokenize(f"{c['section']} {c['text']}") for c in rag.chunks])
        self.chunks_by_resume: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(rag.chunks):
            self.chunks_by_resume[c["resume_id"]].append(i)

    # -- requirement checks -------------------------------------------------------
    @staticmethod
    def check_requirement(req: Requirement, meta: dict) -> tuple[bool, str]:
        if req.kind == "skill":
            ok = req.skill in meta["skills"]
            return ok, "" if ok else f"missing must-have skill {req.skill}"
        if req.kind == "skill_years":
            if req.skill not in meta["skills"]:
                return False, f"missing must-have skill {req.skill}"
            # explicit "N years of <skill>" in the resume if present, else total experience as a proxy
            yrs = meta["skill_years"].get(req.skill, meta["experience_years"])
            ok = yrs >= req.years
            return ok, "" if ok else f"{req.skill}: ~{yrs:g} yrs < {req.years:g} required"
        if req.kind == "experience":
            ok = meta["experience_years"] >= req.years
            return ok, "" if ok else f"{meta['experience_years']:g} yrs experience < {req.years:g} required"
        if req.kind == "degree":
            ok = meta["education"]["level"] >= req.level
            return ok, "" if ok else "degree requirement not met"
        return True, ""

    # -- scoring ---------------------------------------------------------------------
    def _skill_coverage(self, job: JobProfile, skills: set[str]) -> tuple[float, list[str], list[str]]:
        must = set(job.must_skills)
        weights = {s: (MUST_SKILL_WEIGHT if s in must else REQUIRED_SKILL_WEIGHT) for s in job.required_skills}
        weights.update({s: PREFERRED_SKILL_WEIGHT for s in job.preferred_skills if s not in weights})
        weights.update({s: MUST_SKILL_WEIGHT for s in must if s not in weights})
        if not weights:
            return 0.0, [], []
        matched = [s for s in weights if s in skills]
        missing = [s for s in weights if s not in skills and weights[s] >= REQUIRED_SKILL_WEIGHT]
        return sum(weights[s] for s in matched) / sum(weights.values()), matched, missing

    @staticmethod
    def _qualification_fit(job: JobProfile, meta: dict) -> float:
        exp = 1.0 if not job.min_years else min(1.0, meta["experience_years"] / job.min_years)
        deg = 1.0 if not job.degree_level else (1.0 if meta["education"]["level"] >= job.degree_level else
                                                DEGREE_FIT_LOWER if meta["education"]["level"] else DEGREE_FIT_NONE)
        return EXP_FIT_WEIGHT * exp + (1 - EXP_FIT_WEIGHT) * deg

    def _excerpts(self, chunk_ids: list[int], skills: list[str], n: int = N_EXCERPTS) -> list[str]:
        """Best line(s) from each top chunk: the ones mentioning the most job skills."""
        out = []
        for ci in chunk_ids[:n]:
            c = self.rag.chunks[ci]
            lines = [l.strip() for l in c["text"].split("\n") if len(l.strip()) > EXCERPT_MIN_LINE_CHARS] or [c["text"]]
            best = max(lines, key=lambda l: (len(set(find_skills(l)) & set(skills)), len(l) > 60))
            snippet = best if len(best) <= EXCERPT_MAX_CHARS else best[:EXCERPT_MAX_CHARS - 3] + "..."
            out.append(f"[{c['section'].title()}] {snippet}")
        return out

    # -- main entry point --------------------------------------------------------------
    def match(self, jd_text: str, k: int = DEFAULT_K, must: list[str] | None = None, use_jd_must: bool = True,
              mode: str = "hybrid", apply_filters: bool = True) -> dict:
        t0 = time.perf_counter()
        job = parse_job_description(jd_text)
        requirements = (job.must_haves if use_jd_must else []) + [parse_cli_must(m) for m in (must or [])]

        # 1) semantic: FAISS top-N chunks
        qv = self.rag.embed_query(jd_text)
        t_embed = time.perf_counter()
        n = min(CANDIDATE_CHUNKS, self.rag.index.ntotal)
        _, sem_ids = self.rag.index.search(qv, n)
        t_search = time.perf_counter()

        # 2) keyword: BM25 top-N chunks (JD tokens + critical skills boosted)
        q_tokens = _tokenize(jd_text) + _tokenize(" ".join(job.must_skills)) * MUST_SKILL_BOOST
        bm25_scores = self.bm25.get_scores(q_tokens)
        kw_ids = np.argsort(-bm25_scores)[:n]

        # 3) candidate set = union of both retrievers; then exact per-resume aggregation
        candidates = {self.rag.chunks[i]["resume_id"] for i in sem_ids[0] if i >= 0}
        candidates |= {self.rag.chunks[i]["resume_id"] for i in kw_ids}
        chunk_sims = self.vectors @ qv[0]
        bm25_max = max(float(bm25_scores.max()), 1e-9)

        rows, filtered = [], []
        for rid in candidates:
            meta = self.rag.resumes[rid]
            ids = sorted(self.chunks_by_resume[rid], key=lambda i: -chunk_sims[i])
            sims = chunk_sims[ids]
            sem_raw = (SEMANTIC_MAX_WEIGHT * float(sims[0])
                       + (1 - SEMANTIC_MAX_WEIGHT) * float(sims[:SEMANTIC_TOP_N].mean()))
            sem = float(np.clip((sem_raw - SEMANTIC_FLOOR) / (SEMANTIC_CEIL - SEMANTIC_FLOOR), 0, 1))
            kw = float(max(bm25_scores[i] for i in ids)) / bm25_max
            cov, matched, missing = self._skill_coverage(job, set(meta["skills"]))
            qual = self._qualification_fit(job, meta)

            if mode == "semantic":
                score = sem
            elif mode == "keyword":
                score = KEYWORD_MODE_BM25_WEIGHT * kw + (1 - KEYWORD_MODE_BM25_WEIGHT) * cov
            else:
                w = self.weights
                score = (w["semantic"] * sem + w["skills"] * cov + w["keyword"] * kw
                         + w["qualification"] * qual) / sum(w.values())

            failures = [msg for ok, msg in (self.check_requirement(r, meta) for r in requirements) if not ok]
            row = {"rid": rid, "meta": meta, "score": score, "ids": ids, "sem": sem, "kw": kw, "cov": cov,
                   "qual": qual, "matched": matched, "missing": missing, "failures": failures}
            (filtered if failures and apply_filters else rows).append(row)

        rows.sort(key=lambda r: -r["score"])
        top = [self._format(job, r) for r in rows[:k]]
        t_end = time.perf_counter()
        return {
            "job_description": jd_text.strip(),
            "job_title": job.title,
            "parsed_requirements": {
                "required_skills": job.required_skills,
                "preferred_skills": job.preferred_skills,
                "must_haves": [r.describe() for r in requirements],
                "min_years": job.min_years,
                "degree_level": job.degree_level,
            },
            "mode": mode,
            "top_matches": top,
            "filtered_out": [{"candidate_name": r["meta"]["name"], "reasons": r["failures"]}
                             for r in sorted(filtered, key=lambda r: -r["score"])],
            "stats": {
                "resumes_indexed": len(self.rag.resumes),
                "candidates_considered": len(candidates),
                "passed_filters": len(rows),
                "timings_ms": {
                    "embed_query": round((t_embed - t0) * 1000, 1),
                    "faiss_search": round((t_search - t_embed) * 1000, 2),
                    "total": round((t_end - t0) * 1000, 1),
                },
            },
        }

    def _format(self, job: JobProfile, r: dict) -> dict:
        meta = r["meta"]
        top_ids = r["ids"][:SEMANTIC_TOP_N]
        sections = list(dict.fromkeys(self.rag.chunks[i]["section"] for i in top_ids))
        return {
            "candidate_name": meta["name"],
            "resume_path": meta["resume_path"],
            "match_score": int(round(100 * r["score"])),
            "matched_skills": r["matched"],
            "relevant_excerpts": self._excerpts(top_ids, job.required_skills + job.preferred_skills),
            "reasoning": self._reasoning(job, r, sections),
            "missing_skills": r["missing"],
            "matched_sections": sections,
            "experience_years": meta["experience_years"],
            "education": meta["education"]["degree"],
            "score_breakdown": {"semantic": round(r["sem"], 3), "skills": round(r["cov"], 3),
                                "keyword": round(r["kw"], 3), "qualification": round(r["qual"], 3)},
        }

    @staticmethod
    def _reasoning(job: JobProfile, r: dict, sections: list[str]) -> str:
        meta = r["meta"]
        strength = "Strong" if r["sem"] >= STRONG_SEMANTIC else "Moderate" if r["sem"] >= MODERATE_SEMANTIC else "Weak"
        sec_names = [s.title() for s in sections if s not in ("profile", "other")] or ["profile"]
        parts = [f"{strength} semantic match for '{job.title}' in {', '.join(sec_names)} "
                 f"(headline: {meta['headline'].title()})"]
        n_req = len(job.required_skills) or 1
        req_matched = [s for s in r["matched"] if s in job.required_skills]
        if req_matched:
            parts.append(f"covers {len(req_matched)}/{n_req} required skills ({', '.join(req_matched[:REASONING_MAX_SKILLS])})")
        if r["missing"]:
            parts.append(f"missing {', '.join(r['missing'][:REASONING_MAX_MISSING])}")
        if job.min_years:
            cmp = "meets" if meta["experience_years"] >= job.min_years else "below"
            parts.append(f"~{meta['experience_years']:g} yrs experience ({cmp} {job.min_years:g}+ asked)")
        else:
            parts.append(f"~{meta['experience_years']:g} yrs experience")
        if meta["education"]["degree"]:
            edu = meta["education"]["degree"] + (f" in {meta['education']['field']}" if meta["education"]["field"] else "")
            parts.append(edu)
        return "; ".join(parts) + "."


def cli(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="src.rag.cli match", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jd_files", nargs="*", help="job description file(s)")
    ap.add_argument("--text", help="job description given inline instead of a file")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--must", action="append", default=[], help='extra must-have, e.g. "Python:5", "SQL", "years:3"')
    ap.add_argument("--no-jd-must", action="store_true", help="ignore must-haves auto-detected in the JD")
    ap.add_argument("--mode", choices=MODES, default=MODES[0])
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    ap.add_argument("--out", help="directory to write <jd>.json results")
    ap.add_argument("--full", action="store_true", help="print full JSON (default prints a summary when --out is set)")
    args = ap.parse_args(argv)
    if not args.jd_files and not args.text:
        ap.error("give a JD file or --text")

    matcher = JobMatcher(ResumeRAG.load(args.index_dir))
    jobs = [("inline", args.text)] if args.text else [(Path(p).stem, read_file(p)) for p in args.jd_files]
    for name, text in jobs:
        result = matcher.match(text, k=args.k, must=args.must, use_jd_must=not args.no_jd_must, mode=args.mode)
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{name}.json").write_text(json.dumps(result, indent=2))
        if args.out and not args.full:
            st = result["stats"]
            print(f"\n== {result['job_title']}  (must-haves: {result['parsed_requirements']['must_haves']})")
            print(f"   passed filters {st['passed_filters']}/{st['candidates_considered']}, "
                  f"{st['timings_ms']['total']} ms -> {rel(out / (name + '.json'))}")
            for m in result["top_matches"]:
                print(f"   {m['match_score']:3d}  {m['candidate_name']:16s} {', '.join(m['matched_skills'][:5])}")
        else:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
