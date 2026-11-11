"""
streamlit_app.py - Streamlit chat interface

Run:  streamlit run src/ui/streamlit_app.py
"""

import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run` only puts src/ui/ on the import path; add the project root so `from src...` works
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.matching_agent import build_graph  # noqa: E402
from src.agent.tools import candidate_id  # noqa: E402
from src.config.constants import DOCS_DIR  # noqa: E402
from src.mcp_client import get_mcp_client  # noqa: E402
from src.ui.common import is_finished, latest_reply, run_turn  # noqa: E402

st.set_page_config(page_title="Resume Matching Agent", page_icon="🧑‍💼", layout="wide")


# Streamlit re-runs this whole file on every click, so everything that must
# survive (the graph, the conversation id, the chat) lives in st.session_state.
@st.cache_resource
def get_agent():
    return build_graph()  # loads the FAISS index + embedding model once


def new_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []      # [{"role", "content", "steps"}]
    st.session_state.started = False


if "thread_id" not in st.session_state:
    new_conversation()

agent = get_agent()
config = {"configurable": {"thread_id": st.session_state.thread_id}}


def send(user_text: str):
    """Run one turn: show the steps live, then store the reply."""
    st.session_state.chat.append({"role": "user", "content": user_text, "steps": []})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        steps = []
        with st.status("Agent is working…", expanded=True) as status:
            try:
                for step in run_turn(agent, config, user_text, first_turn=not st.session_state.started):
                    steps.append(step)
                    st.write(step)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as error:
                status.update(label=f"Error: {error}", state="error")
                return
        st.session_state.started = True
        reply = latest_reply(agent, config)
        st.markdown(reply)
    st.session_state.chat.append({"role": "assistant", "content": reply, "steps": steps})


# ---------------------------------------------------------------------------
# Sidebar: pick a job
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Choose a job")
    listing = get_mcp_client().call_tool("list_files", {"directory": "job_descriptions", "extension": ".txt"})
    jd_files = [f["name"] for f in listing.get("files", [])]
    choice = st.selectbox("Job description", ["(paste my own)"] + jd_files, index=jd_files.index("it_systems_admin.txt") + 1)
    if choice == "(paste my own)":
        jd_text = st.text_area("Job description or request", "Find me candidates with SQL and 3+ years experience", height=180)
    else:
        jd_text = get_mcp_client().call_tool("read_file", {"path": f"job_descriptions/{choice}"}).get("content", "")
        with st.expander("Preview"):
            st.text(jd_text)
    start_clicked = st.button("🚀 Start screening", type="primary", width="stretch",
                              disabled=st.session_state.started)
    if st.button("↺ New conversation", width="stretch"):
        new_conversation()
        st.rerun()
    st.caption("You can also just type a request in the chat box to start.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🧑‍💼 Resume Matching Agent")
st.caption("LangGraph + Claude for reasoning · bge-small + FAISS + BM25 for search")

state = agent.get_state(config).values if st.session_state.started else {}
tab_chat, tab_shortlist, tab_report, tab_graph, tab_mcp = st.tabs(
    ["💬 Chat", "📋 Shortlist", "📝 Report", "🗺️ State machine", "🔌 MCP servers"])

with tab_chat:
    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            if turn["steps"]:
                with st.expander("Agent steps"):
                    for step in turn["steps"]:
                        st.write(step)
            st.markdown(turn["content"])

    if start_clicked and not st.session_state.started:
        send(choice if choice != "(paste my own)" else jd_text)
        st.rerun()

with tab_shortlist:
    shortlist = state.get("shortlist", [])
    if not shortlist:
        st.info("Start a screening to see the shortlist.")
    else:
        analyses, decisions = state.get("analyses", {}), state.get("decisions", {})
        st.dataframe(pd.DataFrame([{
            "#": i,
            "Candidate": m["candidate_name"],
            "Match score": m["match_score"],
            "LLM fit /10": analyses.get(candidate_id(m), {}).get("fit_score"),
            "Decision": decisions.get(candidate_id(m), {}).get("decision"),
            "Experience (yrs)": m["experience_years"],
            "Matched skills": ", ".join(m["matched_skills"]),
            "Missing skills": ", ".join(m["missing_skills"]),
        } for i, m in enumerate(shortlist, 1)]), hide_index=True, width="stretch")
        if state.get("ranking_changes"):
            st.subheader("What changed in the last re-rank")
            for change in state["ranking_changes"]:
                st.write(change)
        st.subheader("Candidate details")
        for i, m in enumerate(shortlist, 1):
            rid = candidate_id(m)
            a, d = analyses.get(rid, {}), decisions.get(rid, {})
            with st.expander(f"#{i} {m['candidate_name']} · {m['match_score']} · {d.get('decision', '-')}"):
                st.write(a.get("summary", ""))
                left, right = st.columns(2)
                left.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in a.get("strengths", [])))
                right.markdown("**Gaps**\n" + "\n".join(f"- {g}" for g in a.get("gaps", [])))
                st.markdown(f"**Decision:** {d.get('reason', '-')}")
                if d.get("improvement_suggestions"):
                    st.markdown("**Improvement suggestions**\n" + "\n".join(f"- {s}" for s in d["improvement_suggestions"]))
                st.caption(f"Score breakdown: {m['score_breakdown']} · Matcher reasoning: {m['reasoning']}")

with tab_report:
    if state.get("report"):
        st.download_button("⬇️ Download report", state["report"], file_name=Path(state["report_path"]).name)
        st.markdown(state["report"])
    else:
        st.info("The report appears after the first screening.")

with tab_graph:
    png = DOCS_DIR / "state_machine.png"
    if png.exists():
        st.image(str(png))
    else:
        st.code(agent.get_graph().draw_mermaid(), language="text")

with tab_mcp:
    client = get_mcp_client()
    st.caption("Everything below was discovered over MCP (initialize, tools/list, resources/list).")
    for name, info in client.describe().items():
        st.subheader(f"{name}  ·  {info['name']} {info['version']}  ·  protocol {info['protocol_version']}")
        left, right = st.columns(2)
        left.markdown("**Tools**\n" + "\n".join(f"- `{t}`" for t in info["tools"]))
        right.markdown("**Resources**\n" + "\n".join(f"- `{r}`" for r in info["resources"]))
    st.subheader("Read a resource")
    uri = st.text_input("Resource URI", "fs://info/resumes/chen_martin.pdf")
    if st.button("Read"):
        try:
            st.code(client.read_resource(uri)[:3000])
        except Exception as error:
            st.error(str(error))
    st.subheader("Screening history (candidate_db server)")
    st.json(client.call_tool("get_screening_history", {"limit": 5}), expanded=False)

# The chat box stays at the bottom of the page
finished = st.session_state.started and is_finished(agent, config)
prompt = st.chat_input("Conversation finished - start a new one" if finished else
                       "Ask a question, refine the requirements, or say 'done'", disabled=finished)
if prompt:
    with tab_chat:
        send(prompt)
    st.rerun()
