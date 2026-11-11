"""
nodes.py

Every node is a plain function:   def my_node(state: AgentState) -> dict
  - it READS what it needs from `state`
  - it RETURNS only the keys it changed (LangGraph merges them into the state)

You can call a node without the graph, which is handy for testing:
    parse_jd({"job_description": "data/job_descriptions/teacher.txt"})

MILESTONE 4: nodes never open files themselves. Reading a JD or resumes, saving the
report and remembering the screening all go through MCP servers (see get_mcp_client()).
"""

import json
import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import interrupt

from src.agent.llm import get_llm, get_structured_llm
from src.agent.schemas import CandidateAnalysis, FeedbackIntent, FinalDecisions
from src.agent.state import AgentState
from src.agent.tools import (
    candidate_id, extract_job_requirements, get_all_tools, read_resumes_batch, resume_path, run_match,
    save_report_file,
)
from src.config.constants import BORDERLINE_SCORE, LLM_MAX_CONCURRENCY, ROUND1_K
from src.mcp_client import get_mcp_client
from src.rag.skills import normalize_skill

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def last_user_text(state: AgentState) -> str:
    """The text of the most recent human message."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def load_job_description(text: str) -> str:
    """If `text` names a JD file (e.g. 'teacher' or 'data/job_descriptions/teacher.txt'), read it via MCP."""
    name = text.strip()
    if "\n" not in name and len(name) < 200 and " " not in name:
        mcp = get_mcp_client()
        for path in (name, f"job_descriptions/{name}", f"job_descriptions/{name}.txt"):
            result = mcp.call_tool("read_file", {"path": path})
            if result["success"]:
                return result["content"]
    return text


def is_short_query(jd: str) -> bool:
    """A one-line request like 'Find me candidates with React and 3+ years' (not a full JD)."""
    return "\n" not in jd.strip()


def requirements_to_must(requirements: dict) -> list[str]:
    """Turn the LLM's must-haves into matcher filters: known skills + minimum years."""
    must = []
    for item in requirements.get("must_have", []):
        skill = normalize_skill(item)
        if skill and skill not in must:
            must.append(skill)
    if requirements.get("min_years"):
        must.append(f"years:{requirements['min_years']}")
    return must


def explain_ranking_changes(previous: list[dict], current: list[dict], filtered_out: list[dict]) -> list[str]:
    """Compare two shortlists and describe, in plain English, what moved and why (Part B.2)."""
    if not previous:
        return []
    prev = {candidate_id(m): (i + 1, m) for i, m in enumerate(previous)}
    cur = {candidate_id(m): (i + 1, m) for i, m in enumerate(current)}
    filter_reasons = {f["candidate_name"]: f["reasons"] for f in filtered_out}
    changes = []

    for rid, (rank, m) in cur.items():
        name = m["candidate_name"]
        if rid not in prev:
            changes.append(f"🆕 {name} entered at #{rank} (score {m['match_score']}).")
        else:
            old_rank, old = prev[rid]
            if old["match_score"] != m["match_score"]:
                why = f"score {old['match_score']} → {m['match_score']} with the new criteria"
            elif rank < old_rank:
                why = f"score unchanged at {m['match_score']}; candidates above were filtered out"
            else:
                why = f"score unchanged at {m['match_score']}; stronger candidates entered above"
            if rank < old_rank:
                changes.append(f"⬆️ {name} moved up #{old_rank} → #{rank} ({why}).")
            elif rank > old_rank:
                changes.append(f"⬇️ {name} moved down #{old_rank} → #{rank} ({why}).")

    for rid, (old_rank, m) in prev.items():
        if rid not in cur:
            name = m["candidate_name"]
            reasons = filter_reasons.get(name)
            why = "; ".join(reasons) if reasons else "pushed out of the top 10 by stronger candidates"
            changes.append(f"❌ {name} (was #{old_rank}) dropped out: {why}.")

    return changes or ["No change in the ranking."]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "job"


# ---------------------------------------------------------------------------
# Main pipeline (Part A):
# Parse JD -> Extract Requirements -> Search Resumes -> Rank Candidates -> ... -> Generate Report
# ---------------------------------------------------------------------------


def parse_jd(state: AgentState) -> dict:
    """Get the job description text: a JD file name/path, pasted JD text, or a one-line search request."""
    text = state.get("job_description") or last_user_text(state)
    jd = load_job_description(text)
    # a new job starts a fresh screening: forget the old must-haves and shortlists
    return {"job_description": jd, "extra_must": [], "ignored_must": [], "shortlist": [],
            "previous_shortlist": [], "ranking_changes": [], "analyses": {}, "decisions": {}}


def extract_requirements(state: AgentState) -> dict:
    """Use the LLM to split the job into must-have vs nice-to-have requirements."""
    return {"requirements": extract_job_requirements(state["job_description"])}


def search_resumes(state: AgentState) -> dict:
    """Round 1 screening: hybrid RAG search over ALL resumes, keep the top 10 that pass the must-haves."""
    jd = state["job_description"]
    requirements = state.get("requirements") or {}
    extra_must = list(state.get("extra_must", []))
    query = jd
    if is_short_query(jd):
        # the matcher treats line 1 as the title, so give a one-line request a title line,
        # and turn the LLM's must-haves into filters (a full JD has its own "Must have:" lines)
        query = f"{requirements.get('title', 'Candidate search')}\n{jd}"
        extra_must = requirements_to_must(requirements) + extra_must

    result = run_match(query, k=ROUND1_K, extra_must=extra_must, ignored_must=state.get("ignored_must", []))
    return {
        "previous_shortlist": state.get("shortlist", []),
        "shortlist": result["top_matches"],
        "filtered_out": result["filtered_out"],
        "resumes_screened": result["stats"]["resumes_indexed"],
        "must_applied": result["parsed_requirements"]["must_haves"],
        "skipped_must": result["skipped_must"],
    }


def rank_candidates(state: AgentState) -> dict:
    """Order the shortlist by match score and explain how it changed since the last run."""
    shortlist = sorted(state.get("shortlist", []), key=lambda m: -m["match_score"])
    changes = explain_ranking_changes(state.get("previous_shortlist", []), shortlist, state.get("filtered_out", []))
    return {"shortlist": shortlist, "ranking_changes": changes}


# ---------------------------------------------------------------------------
# Multi-round screening (Part C)
# ---------------------------------------------------------------------------


def deep_analysis(state: AgentState) -> dict:
    """Round 2: the LLM reads each top-10 resume and lists strengths, gaps and a 1-10 fit score."""
    shortlist = state.get("shortlist", [])
    if not shortlist:
        return {"analyses": {}}
    job = json.dumps(state.get("requirements", {}))
    # ONE MCP call reads all 10 resumes (batch_process) instead of 10 separate read_file calls
    resumes = read_resumes_batch([candidate_id(m) for m in shortlist])
    prompts = [
        "You are screening candidates. Analyse this candidate for the job. Base everything on the resume.\n\n"
        f"JOB REQUIREMENTS:\n{job}\nExtra must-haves from the recruiter: {state.get('extra_must', [])}\n\n"
        f"MATCHER RESULT (score 0-100):\n{json.dumps({k: m[k] for k in ('candidate_name', 'match_score', 'matched_skills', 'missing_skills', 'reasoning', 'experience_years', 'education')})}\n\n"
        f"RESUME:\n{resumes[candidate_id(m)]}"
        for m in shortlist
    ]
    # .batch() runs the calls in parallel - 10 candidates take about as long as 2
    structured_llm = get_structured_llm(CandidateAnalysis)
    # return_exceptions=True: one failed call gives an error object instead of crashing the whole round
    results = structured_llm.batch(prompts, config={"max_concurrency": LLM_MAX_CONCURRENCY}, return_exceptions=True)
    analyses = {}
    for m, r in zip(shortlist, results):
        if isinstance(r, Exception):
            analyses[candidate_id(m)] = {"summary": f"Analysis failed ({type(r).__name__}); using the matcher result.",
                                         "strengths": m["matched_skills"], "gaps": m["missing_skills"], "fit_score": None}
        else:
            analyses[candidate_id(m)] = r.model_dump()
    return {"analyses": analyses}


def final_recommendation(state: AgentState) -> dict:
    """Round 3: hire / borderline / no-hire for every shortlisted candidate, judged side by side."""
    shortlist = state.get("shortlist", [])
    if not shortlist:
        return {"decisions": {}}
    analyses = state.get("analyses", {})
    candidates = [
        {"candidate_id": candidate_id(m), "name": m["candidate_name"], "match_score": m["match_score"],
         "missing_skills": m["missing_skills"], "analysis": analyses.get(candidate_id(m))}
        for m in shortlist
    ]
    low, high = BORDERLINE_SCORE
    structured_llm = get_structured_llm(FinalDecisions)
    result = structured_llm.invoke(
        "You are the hiring committee. Make a final decision for EVERY candidate below.\n"
        "- 'hire': meets the must-haves well, invite to interview (missing nice-to-haves is fine; "
        "things the resume cannot show, like a certificate number, can be checked in the interview)\n"
        f"- 'borderline': close but with a real gap in a must-have (often match scores {low}-{high}); "
        "you MUST give 2-3 concrete improvement_suggestions\n"
        "- 'no-hire': not a fit for this job\n"
        "Be selective but fair: hire the clearly strong candidates.\n\n"
        f"JOB:\n{json.dumps(state.get('requirements', {}))}\n\nCANDIDATES:\n{json.dumps(candidates, indent=1)}"
    )
    valid_ids = {c["candidate_id"] for c in candidates}
    return {"decisions": {d.candidate_id: d.model_dump() for d in result.decisions if d.candidate_id in valid_ids}}


def generate_report(state: AgentState) -> dict:
    """Write the detailed match report (saved as markdown) and post a short summary in the chat."""
    reqs = state.get("requirements", {})
    title = reqs.get("title", "Job")
    shortlist = state.get("shortlist", [])
    analyses = state.get("analyses", {})
    decisions = state.get("decisions", {})
    counts = {d: sum(1 for x in decisions.values() if x["decision"] == d) for d in ("hire", "borderline", "no-hire")}

    # --- full report (markdown file) ---
    lines = [
        f"# Match report: {title}",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}_",
        "",
        "## Requirements",
        f"- **Must have:** {', '.join(reqs.get('must_have', [])) or '-'}",
        f"- **Nice to have:** {', '.join(reqs.get('nice_to_have', [])) or '-'}",
        f"- **Minimum experience:** {reqs.get('min_years', 0)} years",
        f"- **Filters applied:** {', '.join(state.get('must_applied', [])) or 'none'}",
        "",
        "## Screening rounds",
        f"1. **Initial screen:** {state.get('resumes_screened', 0)} resumes → top {len(shortlist)} "
        f"({len(state.get('filtered_out', []))} removed by must-have filters)",
        f"2. **Deep analysis:** {len(analyses)} candidates analysed by the LLM",
        f"3. **Final decision:** {counts['hire']} hire, {counts['borderline']} borderline, {counts['no-hire']} no-hire",
        "",
    ]
    if state.get("ranking_changes"):
        lines += ["## What changed since the last run", *[f"- {c}" for c in state["ranking_changes"]], ""]
    lines += ["## Ranking", "", "| # | Candidate | Match score | LLM fit | Decision |", "|---|---|---|---|---|"]
    for i, m in enumerate(shortlist, 1):
        rid = candidate_id(m)
        fit = analyses.get(rid, {}).get("fit_score", "-")
        lines.append(f"| {i} | {m['candidate_name']} | {m['match_score']} | {fit}/10 | "
                     f"{decisions.get(rid, {}).get('decision', '-')} |")
    lines.append("")
    for i, m in enumerate(shortlist, 1):
        rid = candidate_id(m)
        a, d = analyses.get(rid, {}), decisions.get(rid, {})
        lines += [
            f"## {i}. {m['candidate_name']} — {d.get('decision', '-').upper()}",
            f"**Match score:** {m['match_score']}/100 · **Experience:** {m['experience_years']} yrs · "
            f"**Education:** {m['education'] or '-'}",
            f"**Score breakdown:** {m['score_breakdown']}",
            "",
            f"**Summary:** {a.get('summary', '-')}",
            "",
            "**Strengths**", *[f"- {s}" for s in a.get("strengths", [])],
            "",
            "**Gaps**", *([f"- {g}" for g in a.get("gaps", [])] or ["- none found"]),
            "",
            f"**Decision:** {d.get('reason', '-')}",
        ]
        if d.get("improvement_suggestions"):
            lines += ["", "**How this candidate could become a hire**", *[f"- {s}" for s in d["improvement_suggestions"]]]
        lines.append("")
    if state.get("filtered_out"):
        lines += ["## Removed by must-have filters",
                  *[f"- {f['candidate_name']}: {'; '.join(f['reasons'])}" for f in state["filtered_out"][:15]]]
    report = "\n".join(lines)
    filename = f"{_slug(title)}_{datetime.now():%Y%m%d_%H%M%S}.md"
    saved = save_report_file(filename, report)                 # MCP: filesystem server, write_file
    report_path = saved.get("path", "")
    remember_screening(title, state.get("must_applied", []), shortlist, decisions)  # MCP: candidate_db server

    # --- short summary for the chat ---
    if not shortlist:
        reasons = "\n".join(f"- {f['candidate_name']}: {'; '.join(f['reasons'])}" for f in state.get("filtered_out", [])[:5])
        summary = (f"**{title}**: none of the {state.get('resumes_screened', 0)} resumes pass the must-haves "
                   f"({', '.join(state.get('must_applied', []))}).\n\nClosest candidates and why they failed:\n{reasons}\n\n"
                   "Try relaxing a requirement, e.g. *\"drop the years requirement\"*.")
    else:
        rows = []
        for i, m in enumerate(shortlist, 1):
            rid = candidate_id(m)
            decision = decisions.get(rid, {}).get("decision", "-")
            icon = {"hire": "✅", "borderline": "🟡", "no-hire": "❌"}.get(decision, "")
            rows.append(f"| {i} | {m['candidate_name']} | {m['match_score']} | "
                        f"{analyses.get(rid, {}).get('fit_score', '-')}/10 | {icon} {decision} |")
        parts = [
            f"**{title}**: screened {state.get('resumes_screened', 0)} resumes → top {len(shortlist)} → "
            f"{counts['hire']} hire, {counts['borderline']} borderline, {counts['no-hire']} no-hire",
            f"Filters: {', '.join(state.get('must_applied', [])) or 'none'}",
        ]
        if state.get("skipped_must"):
            parts.append(f"⚠️ Not in the skill list, used for search only (not as a filter): {', '.join(state['skipped_must'])}")
        if state.get("ranking_changes"):
            parts.append("**What changed:**\n" + "\n".join(f"- {c}" for c in state["ranking_changes"]))
        parts.append("| # | Candidate | Match | Fit | Decision |\n|---|---|---|---|---|\n" + "\n".join(rows))
        parts.append(f"Full report with strengths, gaps and suggestions: `{report_path}`")
        parts.append('Try: *"Compare the top 3"*, *"Why did #1 rank higher than #2?"*, '
                     '*"Make Linux a must-have"*, *"Interview questions for the top candidate"*, *"done"*')
        summary = "\n\n".join(parts)

    return {"report": report, "report_path": report_path, "messages": [AIMessage(content=summary)]}


def remember_screening(title: str, filters: list[str], shortlist: list[dict], decisions: dict) -> None:
    """Store this screening in the candidate database (second MCP server), so it is remembered later."""
    candidates = [
        {"candidate_id": candidate_id(m), "name": m["candidate_name"], "rank": i, "match_score": m["match_score"],
         "decision": decisions.get(candidate_id(m), {}).get("decision")}
        for i, m in enumerate(shortlist, 1)
    ]
    get_mcp_client().call_tool("save_screening", {"job_title": title, "filters": filters, "candidates": candidates})


# ---------------------------------------------------------------------------
# Human feedback loop (Part B)
# ---------------------------------------------------------------------------


def human_feedback(state: AgentState) -> dict:
    """Pause the graph until the user replies. The chat UI resumes it with Command(resume=text)."""
    user_text = interrupt("Your turn: refine the requirements, ask a question, or say 'done'.")
    return {"messages": [HumanMessage(content=user_text)]}


def classify_feedback(state: AgentState) -> dict:
    """Decide what the user wants and update the requirements if they changed them."""
    structured_llm = get_structured_llm(FeedbackIntent)
    intent = structured_llm.invoke(
        "Classify the recruiter's latest message.\n"
        f"Current job: {state.get('requirements', {}).get('title')}\n"
        f"Current filters: {state.get('must_applied', [])}\n\n"
        f"Message: {last_user_text(state)}"
    )
    update = {"feedback": intent.model_dump()}

    if intent.intent == "refine":
        removed = {m.partition(':')[0].lower() for m in intent.remove_must}
        extra = [m for m in state.get("extra_must", []) if m.partition(':')[0].lower() not in removed]
        added = [m for m in intent.add_must if m not in extra]
        update["extra_must"] = extra + added
        # adding a must-have back cancels an earlier "ignore" of the same skill
        added_keys = {m.partition(':')[0].lower() for m in added}
        ignored = [m for m in state.get("ignored_must", []) if m.partition(':')[0].lower() not in added_keys]
        update["ignored_must"] = ignored + [m for m in intent.remove_must if m not in ignored]
        # a one-line search's must-haves come from the LLM requirements: keep them in sync
        if state.get("requirements") and is_short_query(state.get("job_description", "")):
            reqs = dict(state["requirements"])
            reqs["must_have"] = [s for s in reqs.get("must_have", []) if s.partition(':')[0].lower() not in removed]
            if "years" in removed:
                reqs["min_years"] = 0
            update["requirements"] = reqs
    elif intent.intent == "new_job":
        update["job_description"] = intent.new_job_description or last_user_text(state)
    elif intent.intent == "done":
        update["messages"] = [AIMessage(content=f"Good luck with the hiring! Final report: `{state.get('report_path', '-')}`")]
    return update


def route_feedback(state: AgentState) -> str:
    """Conditional edge: pick the next node from the classified intent."""
    intent = state["feedback"]["intent"]
    return {"refine": "search_resumes", "new_job": "parse_jd", "question": "answer_question"}.get(intent, END)


def answer_question(state: AgentState) -> dict:
    """Answer free-form questions, calling tools when needed ("Compare the top 3", "Why did A beat B?")."""
    shortlist = state.get("shortlist", [])
    analyses, decisions = state.get("analyses", {}), state.get("decisions", {})
    context = [
        {"rank": i, "candidate_id": candidate_id(m), "name": m["candidate_name"], "resume_path": resume_path(candidate_id(m)),
         "match_score": m["match_score"],
         "score_breakdown": m["score_breakdown"], "matched_skills": m["matched_skills"],
         "missing_skills": m["missing_skills"], "experience_years": m["experience_years"],
         "education": m["education"], "fit_score": analyses.get(candidate_id(m), {}).get("fit_score"),
         "decision": decisions.get(candidate_id(m), {}).get("decision")}
        for i, m in enumerate(shortlist, 1)
    ]
    system = SystemMessage(content=(
        "You are a recruiting assistant helping a recruiter screen candidates.\n"
        f"JOB REQUIREMENTS: {json.dumps(state.get('requirements', {}))}\n"
        f"FILTERS APPLIED: {state.get('must_applied', [])}\n"
        f"CURRENT SHORTLIST (best first):\n{json.dumps(context, indent=1)}\n\n"
        "How the match score works: 100 x (0.55 semantic similarity + 0.25 skill coverage "
        "+ 0.10 BM25 keyword match + 0.10 qualification fit); each score_breakdown value is 0-1.\n"
        "Rules:\n"
        "- To explain a ranking, compare the score_breakdown components, skills and experience of both candidates.\n"
        "- Use compare_candidates for side-by-side comparisons, and show its table as a markdown table.\n"
        "- Use generate_interview_questions for interview questions.\n"
        "- Use the MCP file tools for resume details: read_file / search_in_file with the candidate's resume_path, "
        "or batch_process to read or search several resumes in one call.\n"
        "- Use check_new_resumes when the user asks to check or wait for new resumes.\n"
        "- Use get_screening_history / get_candidate_history for past screenings, and add_candidate_note to "
        "remember something about a candidate (use the candidate_id).\n"
        "- Earlier chat messages may show older shortlists from before the requirements changed. "
        "The CURRENT SHORTLIST above is always the latest truth; do not comment on the difference.\n"
        "- Only state facts from the context or the tools. Keep answers short and well formatted."
    ))
    response = get_llm().bind_tools(get_all_tools()).invoke([system, *state["messages"]])
    return {"messages": [response]}


def route_after_answer(state: AgentState) -> str:
    """If the LLM asked for tools, run them; otherwise wait for the user again."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "human_feedback"
