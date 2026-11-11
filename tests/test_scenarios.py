"""
test_scenarios.py - "Test scenarios: 5+ conversation flows"

Each test plays a whole conversation through the graph and checks the STATE afterwards
(not the exact wording - LLM text changes, the state should not).

Run:  pytest tests/test_scenarios.py -v
These call Claude (they are skipped without ANTHROPIC_API_KEY) and take a few minutes.
"""

import json
import os
import shutil
import threading
import uuid

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from src.agent.matching_agent import build_graph, start_input
from src.agent.tools import candidate_id, get_matcher
from src.config.constants import DATA_DIR, RESUMES_DIR
from src.mcp_client import get_mcp_client

load_dotenv()
pytestmark = pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY in .env")

IT_JD = "data/job_descriptions/it_systems_admin.txt"


class Conversation:
    """A tiny helper that talks to the graph like the chat UI does."""

    def __init__(self, first_message: str):
        self.agent = build_graph()
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        self.agent.invoke(start_input(first_message), self.config)

    def say(self, text: str) -> dict:
        self.agent.invoke(Command(resume=text), self.config)
        return self.state

    @property
    def state(self) -> dict:
        return self.agent.get_state(self.config).values

    def tool_calls(self, name: str) -> list[dict]:
        return [c for m in self.state["messages"] if isinstance(m, AIMessage) for c in m.tool_calls if c["name"] == name]

    def last_reply(self) -> str:
        return next(m.text for m in reversed(self.state["messages"]) if isinstance(m, AIMessage) and m.text)


def test_1_natural_language_search():
    """'Find me candidates with SQL and 3+ years experience' -> every candidate has SQL and 3+ years."""
    chat = Conversation("Find me candidates with SQL and 3+ years experience")
    resumes = get_matcher().rag.resumes
    assert chat.state["shortlist"]
    for m in chat.state["shortlist"]:
        meta = resumes[candidate_id(m)]
        assert "SQL" in meta["skills"] and meta["experience_years"] >= 3


def test_2_compare_top_three():
    """'Compare the top 3 matches side by side' -> compare_candidates is called with the top 3."""
    chat = Conversation(IT_JD)
    top3 = {candidate_id(m) for m in chat.state["shortlist"][:3]}
    chat.say("Compare the top 3 matches side by side")
    calls = chat.tool_calls("compare_candidates")
    assert calls
    resolved = {c.lower().replace(" ", "_") for c in calls[-1]["args"]["candidate_ids"]}
    assert resolved == top3


def test_3_explain_ranking():
    """'Why did A rank higher than B?' -> the answer talks about both candidates."""
    chat = Conversation(IT_JD)
    first, second = (m["candidate_name"] for m in chat.state["shortlist"][:2])
    chat.say(f"Why did {first} rank higher than {second}?")
    answer = chat.last_reply()
    assert first.split()[0] in answer and second.split()[0] in answer


def test_4_refine_requirements_reranks_and_explains():
    """'Make Linux a must-have' -> new shortlist, everyone has Linux, and the changes are explained."""
    chat = Conversation(IT_JD)
    before = [candidate_id(m) for m in chat.state["shortlist"]]
    state = chat.say("Make Linux a must-have")
    resumes = get_matcher().rag.resumes
    assert "Linux" in state["extra_must"]
    assert [candidate_id(m) for m in state["previous_shortlist"]] == before
    assert all("Linux" in resumes[candidate_id(m)]["skills"] for m in state["shortlist"])
    assert state["ranking_changes"] and state["ranking_changes"] != ["No change in the ranking."]


def test_5_multi_round_screening():
    """100 resumes -> top 10 -> deep analysis of each -> hire/no-hire for each, suggestions for borderline."""
    chat = Conversation(IT_JD)
    state = chat.state
    ids = [candidate_id(m) for m in state["shortlist"]]
    assert state["resumes_screened"] == 100
    assert 0 < len(ids) <= 10
    assert set(state["analyses"]) == set(ids)
    assert set(state["decisions"]) == set(ids)
    for decision in state["decisions"].values():
        assert decision["decision"] in ("hire", "borderline", "no-hire")
        if decision["decision"] == "borderline":
            assert decision["improvement_suggestions"]
    assert os.path.exists(state["report_path"])  # written by the filesystem MCP server


def test_6_interview_questions():
    """'Interview questions for the top candidate' -> generate_interview_questions returns 5 questions."""
    chat = Conversation(IT_JD)
    chat.say("Give me interview questions for the top candidate")
    assert chat.tool_calls("generate_interview_questions")
    results = [m for m in chat.state["messages"] if isinstance(m, ToolMessage) and m.name == "generate_interview_questions"]
    assert results and len(json.loads(results[-1].content)["questions"]) == 5


def test_7_new_job_then_done():
    """Switch to another job mid-conversation, then finish."""
    chat = Conversation(IT_JD)
    state = chat.say("Let's look at a different role: data/job_descriptions/teacher.txt")
    assert "Teacher" in state["job_description"]
    assert state["previous_shortlist"] == []  # a new job starts fresh
    chat.say("done, thanks")
    assert not chat.agent.get_state(chat.config).next  # the graph reached END


# ---------------------------------------------------------------------------
# Milestone 4: scenarios that show the agent using MCP resources
# ---------------------------------------------------------------------------

def test_8_mcp_file_tools_answer_a_question():
    """'Does the top candidate mention Windows?' -> Claude reads the resume with the MCP file tools."""
    chat = Conversation(IT_JD)
    chat.say("Look in the top candidate's resume: does it mention Windows? Quote the line.")
    mcp_calls = [c for c in ("read_file", "search_in_file", "batch_process") if chat.tool_calls(c)]
    assert mcp_calls, "Claude should have used an MCP file tool"
    used = [m for m in chat.state["messages"] if isinstance(m, ToolMessage) and m.name in mcp_calls]
    assert used and '"success": false' not in used[-1].content.lower()


def test_9_watch_for_new_resume():
    """'Check for new resumes' -> watch_directory notices a resume dropped in while waiting,
    batch_process reads it, and the agent analyses it for the current job."""
    chat = Conversation(IT_JD)
    sample = DATA_DIR / "new_resumes" / "resume_alex_johnson.txt"
    target = RESUMES_DIR / sample.name
    timer = threading.Timer(5.0, lambda: shutil.copy(sample, target))  # arrives while the agent is watching
    timer.start()
    try:
        chat.say("Check for new resumes for the next 30 seconds")
    finally:
        timer.cancel()
        target.unlink(missing_ok=True)
    results = [json.loads(m.content) for m in chat.state["messages"]
               if isinstance(m, ToolMessage) and m.name == "check_new_resumes"]
    assert results and "data/resumes/resume_alex_johnson.txt" in results[-1]["new_files"]
    assert results[-1]["analyses"][0]["fit_score"] >= 1


def test_10_multi_mcp_notes_and_history():
    """Two MCP servers in one conversation: notes + history from candidate_db, resumes from filesystem."""
    chat = Conversation(IT_JD)
    top = chat.state["shortlist"][0]
    chat.say(f"Add a note for {top['candidate_name']}: available to start in January")
    assert chat.tool_calls("add_candidate_note")
    notes = get_mcp_client().call_tool("get_candidate_history", {"candidate_id": candidate_id(top)})["notes"]
    assert any("January" in n["note"] for n in notes)
    # every finished screening is saved in the database automatically by generate_report
    history = get_mcp_client().call_tool("get_screening_history", {"job_title": "Systems"})
    assert history["count"] >= 1
