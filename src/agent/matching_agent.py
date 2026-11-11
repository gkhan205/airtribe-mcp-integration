"""
matching_agent.py - the LangGraph agent, now connected to MCP servers (Milestone 4, Part B)

What changed from Milestone 3:
  - No direct file system tools any more. The agent is an MCP CLIENT:
        filesystem server    read_file, search_in_file, list_files, batch_process, watch_directory, write_file
        candidate_db server  save_screening, get_screening_history, get_candidate_history, add_candidate_note
  - The tools Claude can call are DISCOVERED from the servers (tools/list) and turned into
    LangChain tools, so a new server tool needs no new agent code - just a config entry.
  - The graph itself (nodes and edges) is the same, so the agent behaves exactly as before.

    START -> parse_jd -> extract_requirements -> search_resumes -> rank_candidates
          -> deep_analysis -> final_recommendation -> generate_report -> human_feedback
                                                                            |
                                                                  classify_feedback
                                   "refine"   -> search_resumes   (re-rank with the new criteria)
                                   "new_job"  -> parse_jd         (start over with another job)
                                   "question" -> answer_question <-> tools, then back to human_feedback
                                   "done"     -> END

Run:  python -m src.agent.matching_agent            (one JD through the graph, prints the summary)
      python -m src.agent.matching_agent --mcp      (shows the MCP servers, tools and resources found)
      python -m src.agent.matching_agent --diagram  (saves the state machine diagram into docs/)
"""

import json
import subprocess
import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent import nodes
from src.agent.state import AgentState
from src.agent.tools import get_all_tools, start_watching_resumes
from src.config.constants import DOCS_DIR
from src.mcp_client import get_mcp_client


def build_graph():
    """Create and compile the matching agent graph."""
    get_mcp_client()           # start + connect to the MCP servers (once per program)
    start_watching_resumes()   # from now on, check_new_resumes reports every resume that is added
    graph = StateGraph(AgentState)

    # main pipeline (Part A) + multi-round screening (Part C)
    graph.add_node("parse_jd", nodes.parse_jd)
    graph.add_node("extract_requirements", nodes.extract_requirements)
    graph.add_node("search_resumes", nodes.search_resumes)          # round 1: top 10 of all resumes
    graph.add_node("rank_candidates", nodes.rank_candidates)
    graph.add_node("deep_analysis", nodes.deep_analysis)            # round 2
    graph.add_node("final_recommendation", nodes.final_recommendation)  # round 3
    graph.add_node("generate_report", nodes.generate_report)

    # human feedback loop (Part B)
    graph.add_node("human_feedback", nodes.human_feedback)
    graph.add_node("classify_feedback", nodes.classify_feedback)
    graph.add_node("answer_question", nodes.answer_question)
    graph.add_node("tools", ToolNode(get_all_tools()))   # MCP tools + the agent's own tools

    graph.add_edge(START, "parse_jd")
    graph.add_edge("parse_jd", "extract_requirements")
    graph.add_edge("extract_requirements", "search_resumes")
    graph.add_edge("search_resumes", "rank_candidates")
    graph.add_edge("rank_candidates", "deep_analysis")
    graph.add_edge("deep_analysis", "final_recommendation")
    graph.add_edge("final_recommendation", "generate_report")
    graph.add_edge("generate_report", "human_feedback")
    graph.add_edge("human_feedback", "classify_feedback")
    graph.add_conditional_edges("classify_feedback", nodes.route_feedback, {
        "search_resumes": "search_resumes",
        "parse_jd": "parse_jd",
        "answer_question": "answer_question",
        END: END,
    })
    graph.add_conditional_edges("answer_question", nodes.route_after_answer, {
        "tools": "tools",
        "human_feedback": "human_feedback",
    })
    graph.add_edge("tools", "answer_question")

    # The checkpointer saves the state after every node. That is what lets the graph
    # pause at human_feedback and continue later with the same memory.
    return graph.compile(checkpointer=InMemorySaver())


def start_input(text: str) -> dict:
    """The first input for a conversation: a JD file name/path, a pasted JD, or a one-line request."""
    return {"job_description": text, "messages": [HumanMessage(content=text)]}


def show_mcp() -> None:
    """Print what the MCP client discovered: servers, tools and resources."""
    client = get_mcp_client()
    for name, info in client.describe().items():
        print(f"\n=== MCP server '{name}' ({info['name']} {info['version']}, protocol {info['protocol_version']})")
        print("Tools:     " + ", ".join(info["tools"]))
        print("Resources: " + ", ".join(info["resources"]))
    print("\nfilesystem server_status:", json.dumps(client.call_tool("server_status"), indent=1))


# How the agent nodes use the MCP servers (added to the diagram file)
MCP_DIAGRAM = """flowchart LR
    subgraph Agent["LangGraph agent (MCP client)"]
        parse_jd; deep_analysis; generate_report; tools["tools (Claude's tool calls)"]
    end
    client["MCPClient<br/>JSON-RPC 2.0"]
    subgraph FS["filesystem_mcp_server.py"]
        read_file; batch_process; write_file; watch_directory; search_in_file
    end
    subgraph DB["candidate_db_mcp_server.py"]
        save_screening; get_screening_history; add_candidate_note
    end
    parse_jd --> client; deep_analysis --> client; generate_report --> client; tools --> client
    client -- stdio --> FS
    client -- stdio --> DB
"""


def save_diagram() -> None:
    """Save the graph as an image + mermaid text - the 'state machine diagram' deliverable."""
    drawable = build_graph().get_graph()
    DOCS_DIR.mkdir(exist_ok=True)
    mermaid = drawable.draw_mermaid()
    (DOCS_DIR / "state_machine.md").write_text(
        f"# Agent state machine\n\n```mermaid\n{mermaid}```\n\n"
        f"## Agent ↔ MCP servers\n\n```mermaid\n{MCP_DIAGRAM}```\n"
    )
    (DOCS_DIR / "state_machine.mmd").write_text(mermaid)
    print(f"Saved {DOCS_DIR / 'state_machine.md'} and state_machine.mmd")
    # render the PNG locally with the mermaid CLI (needs Node.js; downloads the CLI on first use)
    command = ["npx", "-y", "@mermaid-js/mermaid-cli@11.4.2", "-i", str(DOCS_DIR / "state_machine.mmd"),
               "-o", str(DOCS_DIR / "state_machine.png"), "-w", "1400", "-b", "white"]
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"Saved {DOCS_DIR / 'state_machine.png'}")
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Could not render the PNG ({error}). Paste state_machine.mmd into https://mermaid.live instead.")


if __name__ == "__main__":
    if "--diagram" in sys.argv:
        save_diagram()
    elif "--mcp" in sys.argv:
        show_mcp()
    else:
        jd = sys.argv[1] if len(sys.argv) > 1 else "data/job_descriptions/it_systems_admin.txt"
        agent = build_graph()
        config = {"configurable": {"thread_id": "demo"}}
        state = agent.invoke(start_input(jd), config)
        print(state["messages"][-1].content)
