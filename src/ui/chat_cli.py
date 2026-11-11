"""
chat_cli.py - Part B: conversational interface in the terminal

Run:  python -m src.ui.chat_cli

Start with a job description file name ("it_systems_admin"), a path, or a request like
"Find me candidates with SQL and 3+ years experience". Then keep chatting:
  - "Compare the top 3 matches side by side"
  - "Why did <name> rank higher than <name>?"
  - "Make Linux a must-have"      (re-ranks and explains what changed)
  - "Interview questions for the top candidate"
  - "Check for new resumes"          (MCP watch_directory + batch_process)
  - "Add a note: Elena Reed is available from January"   (candidate_db MCP server)
  - "done"
"""

import uuid

from src.agent.matching_agent import build_graph
from src.mcp_client import get_mcp_client
from src.ui.common import is_finished, latest_reply, run_turn


def main():
    agent = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("=== Resume Matching Agent ===")
    listing = get_mcp_client().call_tool("list_files", {"directory": "job_descriptions", "extension": ".txt"})
    print("MCP servers connected:", ", ".join(get_mcp_client().sessions))
    print("Job descriptions you can use:", ", ".join(f["name"].removesuffix(".txt") for f in listing.get("files", [])))
    print("Or type a request, e.g. 'Find me candidates with SQL and 3+ years experience'. Type 'exit' to quit.")

    first_turn = True
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            for step in run_turn(agent, config, user_text, first_turn):
                print(f"  {step}")
        except Exception as error:
            print(f"Something went wrong: {error}")
            continue
        first_turn = False

        print(f"\nAgent: {latest_reply(agent, config)}")
        if is_finished(agent, config):
            break


if __name__ == "__main__":
    main()
