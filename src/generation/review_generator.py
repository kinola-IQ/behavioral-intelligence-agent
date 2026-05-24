"""Generate behavioural review text from context."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from ..logging.audit_log import log_event
from ..core.prompt_engine import review_generation_prompt
from ..core import utils
from ..core.utils import _message_text, parse_agent_json_response
from ..core.persona_builder import model_user
from ..core.memory_layer import context_store
from ..retrieval.search import retrieve_text
from .review_tools import submit_review

_AGENT = None


def review_generator_agent():
    """ReAct agent for persona modelling, retrieval, and review submission."""
    global _AGENT
    if _AGENT is not None:
        return _AGENT

    log_event("review_generator_agent_create")
    _AGENT = create_react_agent(
        utils.LLM,
        tools=[model_user, context_store, retrieve_text, submit_review],
        prompt=review_generation_prompt(),
    )
    return _AGENT


def _trace_excerpt(messages: list[Any], limit: int = 8) -> str:
    lines: list[str] = []
    for message in messages[-limit:]:
        if isinstance(message, dict):
            role = message.get("type") or message.get("role") or "message"
            content = message.get("content")
        else:
            role = getattr(message, "type", None) or "message"
            content = getattr(message, "content", None)
        text = _message_text(content)
        if text:
            lines.append(f"{role}: {text[:400]}")
    return "\n".join(lines) if lines else "(no tool trace text)"


def _finalize_review_with_llm(user_prompt: str, messages: list[Any]) -> dict[str, Any]:
    if utils.LLM is None:
        raise ValueError("LLM not initialized")

    log_event("review_generator_finalize_fallback")
    response = utils.LLM.invoke(
        [
            SystemMessage(content=review_generation_prompt()),
            HumanMessage(
                content=(
                    "Produce the final review using the request and agent trace below.\n"
                    "Respond with ONLY JSON keys predicted_rating (int 1-5) and "
                    "predicted_review (string).\n\n"
                    f"Request:\n{user_prompt}\n\n"
                    f"Agent trace:\n{_trace_excerpt(messages)}"
                )
            ),
        ]
    )
    return parse_agent_json_response([response])


def generate_review(prompt: str) -> dict[str, Any]:
    """Run the review agent and return predicted_rating / predicted_review."""
    agent = review_generator_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"recursion_limit": 30},
    )
    messages = result["messages"]
    try:
        return parse_agent_json_response(messages)
    except ValueError:
        return _finalize_review_with_llm(prompt, messages)
