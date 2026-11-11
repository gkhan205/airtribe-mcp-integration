"""
llm.py

One place that creates the Claude chat model, so every node uses the same settings.
"""

from functools import lru_cache

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from src.config.constants import LLM_MAX_TOKENS, LLM_MODEL

load_dotenv()


@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    """Return the chat model used by the agent (created once, then reused)."""
    return ChatAnthropic(model=LLM_MODEL, max_tokens=LLM_MAX_TOKENS)


def get_structured_llm(schema: type[BaseModel]) -> Runnable:
    """A model that always answers with an instance of `schema` (a Pydantic class).

    method="json_schema" uses Claude's native structured outputs, so the reply is
    guaranteed to match the schema. .with_retry() tries again once if the call still fails.
    """
    return get_llm().with_structured_output(schema, method="json_schema").with_retry(stop_after_attempt=2)
