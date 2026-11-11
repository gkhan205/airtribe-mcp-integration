"""
state.py - Part A.1: Agent State Design

The state is ONE dictionary that flows through every node of the graph.
Each node reads from it and returns only the keys it wants to change.
The checkpointer saves it after every node, so it survives between chat turns.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # 1. Conversation history.
    #    `add_messages` APPENDS new messages instead of replacing the list.
    #    Every other key below is simply REPLACED when a node returns it.
    messages: Annotated[list[AnyMessage], add_messages]

    # 2. Job understanding
    job_description: str          # raw JD text (or the recruiter's search request)
    requirements: dict            # JobRequirements from the LLM: title, must_have, nice_to_have, ...
    extra_must: list[str]         # must-haves added mid-conversation, e.g. ["Linux", "Python:3"]
    ignored_must: list[str]       # must-haves the user asked to drop
    must_applied: list[str]       # the filters the last search actually used, e.g. ["3+ years experience", "Linux"]
    skipped_must: list[str]       # must-haves not in the skill taxonomy (searched semantically, not filtered)

    # 3. Shortlist + reasoning
    resumes_screened: int         # how many resumes round 1 looked at
    shortlist: list[dict]         # round 1: top 10 from JobMatcher, each with score + reasoning
    previous_shortlist: list[dict]  # the shortlist before the last refinement
    filtered_out: list[dict]      # candidates removed by must-have filters, with reasons
    ranking_changes: list[str]    # human-readable explanation of what moved and why

    # 4. Multi-round screening (Part C)
    analyses: dict                # round 2: candidate_id -> CandidateAnalysis dict
    decisions: dict               # round 3: candidate_id -> HireDecision dict
    report: str                   # the full markdown report
    report_path: str              # where the report was saved

    # 5. Human feedback loop
    feedback: dict                # FeedbackIntent from the last user message
