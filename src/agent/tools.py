"""
tools.py - the tools available to the agent

MILESTONE 4 CHANGE: this file no longer touches the file system.
  Before:  from src.file_system import fs_tools  ->  fs_tools.read_file(path)
  Now:     get_mcp_client().call_tool("read_file", {"path": path})
The file tools Claude can call (read_file, search_in_file, list_files, ...) are not written
here any more - they are discovered from the MCP servers and converted in get_all_tools().

The @tool decorator turns a normal Python function into something Claude can call.
Claude decides WHEN to call a tool by reading:
  - the docstring  -> what the tool does
  - the type hints -> what arguments it takes

Some tools take `state: Annotated[dict, InjectedState]`. LangGraph fills that argument
with the current agent state when the tool runs, and Claude never sees it. That is how
compare_candidates knows the current job and shortlist without Claude re-sending them.

Tools return plain data and never crash: they return {"success": False, "error": "..."} instead.

Try a tool on its own:
    from src.agent.tools import rag_search
    rag_search.invoke({"query": "python linux", "k": 3})
"""

import json
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.agent.llm import get_structured_llm
from src.agent.schemas import CandidateAnalysis, CandidateComparison, InterviewQuestions, JobRequirements
from src.config.constants import LLM_MAX_CONCURRENCY, MCP_TOOLS_FOR_LLM, RESUME_CHARS_FOR_LLM, WATCH_DEFAULT_SECONDS
from src.mcp_client import get_mcp_client
from src.rag.job_matcher import JobMatcher, parse_cli_must, parse_job_description
from src.rag.resume_rag import ResumeRAG

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_matcher = None


def get_matcher() -> JobMatcher:
    """Return a JobMatcher, loading the FAISS index the first time only."""
    global _matcher
    if _matcher is None:
        _matcher = JobMatcher(ResumeRAG.load())
    return _matcher


def candidate_id(match: dict) -> str:
    """'data/resumes/chen_wang.pdf' -> 'chen_wang'."""
    return Path(match["resume_path"]).stem


def resolve_candidate(name_or_id: str) -> str | None:
    """Turn 'Chen Wang', 'chen_wang' or 'chen_wang.pdf' into the candidate id, or None if unknown."""
    resumes = get_matcher().rag.resumes
    key = Path(name_or_id.strip()).stem.lower().replace(" ", "_").replace("-", "_")
    if key in resumes:
        return key
    for rid, meta in resumes.items():
        if meta["name"].lower() == name_or_id.strip().lower():
            return rid
    return None


def resume_path(rid: str) -> str:
    """'chen_wang' -> 'data/resumes/chen_wang.pdf' (the path the filesystem MCP server expects)."""
    return get_matcher().rag.resumes[rid]["resume_path"]


def read_resume_text(rid: str) -> str:
    """Full text of a candidate's resume (cut to RESUME_CHARS_FOR_LLM characters), read via MCP."""
    result = get_mcp_client().call_tool("read_file", {"path": resume_path(rid), "max_chars": RESUME_CHARS_FOR_LLM})
    return result["content"] if result["success"] else ""


def read_resumes_batch(rids: list[str]) -> dict[str, str]:
    """Read many resumes with ONE MCP call (batch_process). Returns {candidate_id: text}."""
    paths = {resume_path(rid): rid for rid in rids}
    result = get_mcp_client().call_tool(
        "batch_process", {"paths": list(paths), "operation": "read", "max_chars": RESUME_CHARS_FOR_LLM}
    )
    texts = {rid: "" for rid in rids}  # a resume that failed to load just gets empty text
    for item in result.get("results", []):
        texts[paths[item["path"]]] = item["content"]
    return texts


def save_report_file(filename: str, content: str) -> dict:
    """Save a report in the reports folder via the MCP write_file tool."""
    name = Path(filename).name or "report.md"
    return get_mcp_client().call_tool("write_file", {"path": f"reports/{name}", "content": content})


def start_watching_resumes() -> None:
    """Take the first snapshot of the resumes folder, so check_new_resumes sees every file added from now on."""
    get_mcp_client().call_tool("watch_directory", {"directory": "resumes", "timeout_seconds": 0})


def _must_key(spec: str) -> str:
    """The part of a must-have that identifies it: 'Python:3' -> 'python', 'years:5' -> 'years'."""
    return spec.partition(":")[0].strip().lower()


def _requirement_to_spec(req) -> str:
    """Turn a Requirement found in the JD back into the 'Skill:years' text format."""
    if req.kind == "experience":
        return f"years:{req.years:g}"
    if req.kind == "skill_years":
        return f"{req.skill}:{req.years:g}"
    return req.skill


def build_must_list(jd: str, extra_must: list[str], ignored_must: list[str]) -> tuple[list[str], list[str]]:
    """Combine the JD's own must-haves with the user's changes.

    Returns (valid must-haves, skipped ones). A must-have is skipped when its skill is
    not in the taxonomy (parse_cli_must would stop the program) - the search still uses
    it semantically, it just is not a hard filter.
    """
    ignored = {_must_key(m) for m in ignored_must}
    jd_musts = [_requirement_to_spec(r) for r in parse_job_description(jd).must_haves if r.kind != "degree"]
    valid, skipped, seen = [], [], set()
    for spec in jd_musts + list(extra_must):
        key = _must_key(spec)
        if key in ignored or spec.lower() in seen:
            continue
        seen.add(spec.lower())
        try:
            parse_cli_must(spec)
            valid.append(spec)
        except (SystemExit, ValueError):
            skipped.append(spec)
    return valid, skipped


def run_match(jd: str, k: int, extra_must: list[str] | None = None, ignored_must: list[str] | None = None) -> dict:
    """Run the Milestone 2 matcher with the combined must-have list."""
    must, skipped = build_must_list(jd, extra_must or [], ignored_must or [])
    result = get_matcher().match(jd, k=k, must=must, use_jd_must=False)
    result["skipped_must"] = skipped
    return result


def extract_job_requirements(jd: str) -> dict:
    """Ask the LLM for must-have vs nice-to-have requirements (structured output)."""
    structured_llm = get_structured_llm(JobRequirements)
    reqs = structured_llm.invoke(
        "Extract the hiring requirements from this job description or recruiter request. "
        "Only list what is actually asked for.\n\n" + jd
    )
    return reqs.model_dump()


def candidate_profile(rid: str, state: dict) -> dict:
    """Everything the agent knows about one candidate: match result, round 2 analysis, round 3 decision."""
    meta = get_matcher().rag.resumes[rid]
    shortlist = state.get("shortlist") or []
    match = next((m for m in shortlist if candidate_id(m) == rid), None)
    profile = {
        "candidate_id": rid,
        "name": meta["name"],
        "headline": meta["headline"],
        "experience_years": meta["experience_years"],
        "education": meta["education"]["degree"],
        "skills": meta["skills"],
    }
    if match:
        profile.update(
            rank=shortlist.index(match) + 1,
            match_score=match["match_score"],
            score_breakdown=match["score_breakdown"],
            matched_skills=match["matched_skills"],
            missing_skills=match["missing_skills"],
            reasoning=match["reasoning"],
        )
    else:
        profile["note"] = "Not in the current shortlist"
    if rid in (state.get("analyses") or {}):
        profile["analysis"] = state["analyses"][rid]
    if rid in (state.get("decisions") or {}):
        profile["decision"] = state["decisions"][rid]
    return profile


def _job_summary(state: dict) -> str:
    reqs = state.get("requirements") or {}
    return json.dumps({"title": reqs.get("title"), "must_have": reqs.get("must_have"),
                       "nice_to_have": reqs.get("nice_to_have"), "min_years": reqs.get("min_years"),
                       "extra_must_added_by_user": state.get("extra_must", [])})


# ---------------------------------------------------------------------------
# RAG search tool (from Milestone 2)
# ---------------------------------------------------------------------------

@tool
def rag_search(query: str, k: int = 10, must: list[str] | None = None) -> dict:
    """Search ALL resumes for a job description or skills query and return the top k ranked candidates.

    `must` adds hard requirements in 'Skill' or 'Skill:years' format, e.g. ['Linux', 'Python:3'],
    or 'years:N' for overall experience.
    """
    result = run_match(query, k=k, extra_must=must or [])
    return {
        "success": True,
        "must_haves_applied": result["parsed_requirements"]["must_haves"],
        "skipped_unknown_skills": result["skipped_must"],
        "candidates": [
            {"candidate_id": candidate_id(m), "name": m["candidate_name"], "match_score": m["match_score"],
             "matched_skills": m["matched_skills"], "missing_skills": m["missing_skills"], "reasoning": m["reasoning"]}
            for m in result["top_matches"]
        ],
        "filtered_out": result["filtered_out"][:10],
    }


# ---------------------------------------------------------------------------
# New tools for Milestone 3
# ---------------------------------------------------------------------------

@tool
def extract_requirements(jd: str) -> dict:
    """Parse a job description into must-have vs nice-to-have requirements."""
    return extract_job_requirements(jd)


@tool
def compare_candidates(candidate_ids: list[str], state: Annotated[dict, InjectedState]) -> dict:
    """Compare 2-5 candidates head-to-head for the current job. Ids or names both work, e.g. ['chen_wang', 'Zara Ali']."""
    rids = [resolve_candidate(c) for c in candidate_ids]
    unknown = [c for c, r in zip(candidate_ids, rids) if r is None]
    if unknown:
        return {"success": False, "error": f"Unknown candidates: {unknown}"}
    profiles = [candidate_profile(r, state) for r in rids]

    structured_llm = get_structured_llm(CandidateComparison)
    comparison = structured_llm.invoke(
        "You are a hiring manager. Compare these candidates for the job below. Be specific and fair.\n\n"
        f"JOB:\n{_job_summary(state)}\n\nCANDIDATES:\n{json.dumps(profiles, indent=1)}"
    )
    table = [
        {"candidate": p["name"], "rank": p.get("rank", "-"), "match_score": p.get("match_score", "-"),
         "experience_years": p["experience_years"], "education": p["education"],
         "matched_skills": ", ".join(p.get("matched_skills", [])) or "-",
         "missing_skills": ", ".join(p.get("missing_skills", [])) or "-",
         "decision": p.get("decision", {}).get("decision", "-")}
        for p in profiles
    ]
    return {"success": True, "table": table, **comparison.model_dump()}


@tool
def generate_interview_questions(candidate_id: str, state: Annotated[dict, InjectedState]) -> dict:
    """Create 5 screening interview questions for one candidate, targeting their gaps for the current job."""
    rid = resolve_candidate(candidate_id)
    if not rid:
        return {"success": False, "error": f"Unknown candidate: {candidate_id}"}
    profile = candidate_profile(rid, state)

    structured_llm = get_structured_llm(InterviewQuestions)
    result = structured_llm.invoke(
        "Write 5 screening interview questions for this candidate. Make them specific to THIS person: "
        "verify their claimed strengths and probe their gaps for the job.\n\n"
        f"JOB:\n{_job_summary(state)}\n\nCANDIDATE:\n{json.dumps(profile, indent=1)}\n\n"
        f"RESUME:\n{read_resume_text(rid)}"
    )
    return {"success": True, "candidate": profile["name"], "questions": result.questions}


@tool
def check_new_resumes(state: Annotated[dict, InjectedState], wait_seconds: int = WATCH_DEFAULT_SECONDS) -> dict:
    """Watch the resumes folder for NEW resumes (waits up to wait_seconds), then read and analyse each new one
    for the current job. Use when the user asks to check, wait or watch for new resumes/applicants."""
    mcp = get_mcp_client()
    # 1. MCP watch_directory: what was added since the last check?
    watch = mcp.call_tool("watch_directory", {"directory": "resumes", "timeout_seconds": wait_seconds})
    if not watch["success"]:
        return watch
    new_files = watch["new_files"] + watch["modified_files"]
    if not new_files:
        return {"success": True, "new_files": [], "message": f"No new resumes arrived in {watch['waited_seconds']} s."}

    # 2. MCP batch_process: read all new resumes in one call
    batch = mcp.call_tool("batch_process", {"paths": new_files, "operation": "read", "max_chars": RESUME_CHARS_FOR_LLM})
    if not batch["success"]:
        return batch

    # 3. LLM: a quick fit analysis of each new resume for the current job (in parallel)
    job = _job_summary(state)
    prompts = [f"Analyse this NEW applicant for the job. Base everything on the resume.\n\nJOB:\n{job}\n\n"
               f"RESUME ({item['path']}):\n{item['content']}" for item in batch["results"]]
    analyses = get_structured_llm(CandidateAnalysis).batch(
        prompts, config={"max_concurrency": LLM_MAX_CONCURRENCY}, return_exceptions=True)
    return {
        "success": True,
        "new_files": new_files,
        "analyses": [
            {"file": item["path"], **(a.model_dump() if not isinstance(a, Exception) else {"error": str(a)})}
            for item, a in zip(batch["results"], analyses)
        ],
        "unreadable_files": batch["errors"],
        "note": "New resumes are not in the search index yet, so they are not part of the ranked shortlist.",
    }


# Tools written in this file (they use MCP inside, plus RAG and the LLM).
AGENT_TOOLS = [rag_search, extract_requirements, compare_candidates, generate_interview_questions, check_new_resumes]


def get_all_tools() -> list:
    """Every tool Claude can call: the MCP tools (discovered from the servers) + the agent's own tools."""
    return get_mcp_client().langchain_tools(MCP_TOOLS_FOR_LLM) + AGENT_TOOLS
