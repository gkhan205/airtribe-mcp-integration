"""
common.py

Shared code for both chat interfaces: run one conversation turn through the graph
and describe each step, so the user can watch the agent's reasoning.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.agent.matching_agent import start_input

# What each node is doing, shown while the graph runs
STEP_LABELS = {
    "parse_jd": "📄 Reading the job description",
    "extract_requirements": "🧩 Extracting must-have vs nice-to-have requirements (LLM)",
    "search_resumes": "🔎 Round 1: hybrid RAG search over all resumes",
    "rank_candidates": "📊 Ranking candidates and comparing with the previous ranking",
    "deep_analysis": "🧠 Round 2: deep analysis of the top 10 (LLM, in parallel)",
    "final_recommendation": "⚖️ Round 3: hire / borderline / no-hire decisions (LLM)",
    "generate_report": "📝 Writing the match report",
    "human_feedback": "💬 Waiting for you",
    "classify_feedback": "🤔 Understanding your request (LLM)",
    "answer_question": "💡 Thinking about your question (LLM)",
    "tools": "🛠️ Running tools via MCP",
}


def describe_update(node: str, update: dict) -> str:
    """One line describing what a node just did."""
    label = STEP_LABELS.get(node, node)
    update = update or {}
    if node == "extract_requirements":
        reqs = update.get("requirements", {})
        return f"{label} → must: {', '.join(reqs.get('must_have', [])) or '-'}"
    if node == "search_resumes":
        return (f"{label} → {update.get('resumes_screened', 0)} resumes, {len(update.get('shortlist', []))} shortlisted, "
                f"{len(update.get('filtered_out', []))} filtered out")
    if node == "classify_feedback":
        fb = update.get("feedback", {})
        extra = f" (add {fb['add_must']})" if fb.get("add_must") else ""
        extra += f" (remove {fb['remove_must']})" if fb.get("remove_must") else ""
        return f"{label} → {fb.get('intent')}{extra}"
    if node == "tools":
        from src.mcp_client import get_mcp_client
        servers = get_mcp_client().tool_server
        names = [f"{m.name} ({servers.get(m.name, 'agent')})" for m in update.get("messages", [])]
        return f"{label} → {', '.join(names)}"
    if node == "answer_question":
        calls = [c["name"] for m in update.get("messages", []) for c in getattr(m, "tool_calls", [])]
        return f"{label} → calling {', '.join(calls)}" if calls else f"{label} → answering"
    return label


def run_turn(agent, config: dict, user_text: str, first_turn: bool):
    """Run the graph until it pauses for the user (or ends). Yields one line per step."""
    payload = start_input(user_text) if first_turn else Command(resume=user_text)
    for chunk in agent.stream(payload, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node != "__interrupt__":
                yield describe_update(node, update)


def latest_reply(agent, config: dict) -> str:
    """The agent's last message to the user."""
    for message in reversed(agent.get_state(config).values.get("messages", [])):
        if isinstance(message, AIMessage) and message.text:
            return message.text
        if isinstance(message, HumanMessage):
            break
    return ""


def is_finished(agent, config: dict) -> bool:
    """True when the graph reached END (the user said 'done')."""
    return not agent.get_state(config).next
