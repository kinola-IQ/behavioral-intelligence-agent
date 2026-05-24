"""Shared helpers for core modules."""
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore
import chromadb
from langchain_groq import ChatGroq

from huggingface_hub import InferenceClient
from ..config.constants import EMBEDS_DIR
from ..config.settings import Settings
from ..logging.audit_log import log_event


settings = Settings()

MEMORY: InMemoryStore | None = None
VECTORDB = None
LLM: ChatGroq | None = None
HF_LLM_PROVIDER: InferenceClient | None = None

load_dotenv()


async def startup_resources() -> None:
    """Start up resources used by the system."""
    global MEMORY, VECTORDB, LLM, HF_LLM_PROVIDER

    log_event("startup_resources_begin")

    # for context sharing
    MEMORY = InMemoryStore(
        index={
            "dims": 1536,
            "embed": f"huggingface:{settings.embedding_model}",
        }
    )

    # client for vector database operations
    VECTORDB = chromadb.PersistentClient(path=EMBEDS_DIR)

    # model to be used in review generation
    LLM = ChatGroq(
        # model="meta-llama/llama-4-scout-17b-16e-instruct",
        model="llama-3.1-8b-instant",
        groq_api_key=os.environ['GROQ_API_KEY']
        )

    # model provider to be used for chat
    HF_LLM_PROVIDER = InferenceClient(
        api_key=os.environ["HUGGINGFACE_API_KEY"],
    )

    log_event(
        "startup_resources_complete",
        memory=MEMORY is not None,
        vectordb=VECTORDB is not None,
        llm=LLM is not None,
        hf_provider=HF_LLM_PROVIDER is not None,
    )


def startup_status():
    """checks state of the resources needed"""
    if all([MEMORY, VECTORDB, LLM, HF_LLM_PROVIDER]):
        return "Success"
    return "Failed"


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    candidates: list[str] = []
    fence = _JSON_FENCE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text.strip())

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_review_payload(parsed: dict[str, Any]) -> dict[str, Any] | None:
    rating = parsed.get("predicted_rating", parsed.get("rating"))
    review = (
        parsed.get("predicted_review")
        or parsed.get("review")
        or parsed.get("review_text")
    )
    if rating is None or review is None:
        return None
    try:
        rating_int = max(1, min(5, int(rating)))
    except (TypeError, ValueError):
        return None
    review_text = str(review).strip()
    if not review_text:
        return None
    return {"predicted_rating": rating_int, "predicted_review": review_text}


def _review_from_tool_calls(message: Any) -> dict[str, Any] | None:
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls") or []
    else:
        tool_calls = getattr(message, "tool_calls", None) or []

    for call in tool_calls:
        if isinstance(call, dict):
            name = call.get("name")
            args = call.get("args", {})
        else:
            name = getattr(call, "name", None)
            args = getattr(call, "args", {})

        if name != "submit_review":
            continue
        if isinstance(args, str):
            args = _try_parse_json_object(args) or {}
        if not isinstance(args, dict):
            continue
        normalized = _normalize_review_payload(args)
        if normalized is not None:
            return normalized
    return None


def parse_agent_json_response(messages: list[Any]) -> dict[str, Any]:
    """Return review fields from a LangGraph message list."""
    for message in reversed(messages):
        from_tool_call = _review_from_tool_calls(message)
        if from_tool_call is not None:
            return from_tool_call

        if isinstance(message, dict):
            role = message.get("type") or message.get("role")
            content = message.get("content")
            tool_name = message.get("name")
        else:
            role = getattr(message, "type", None)
            content = getattr(message, "content", None)
            tool_name = getattr(message, "name", None)

        parsed = _try_parse_json_object(_message_text(content))
        if parsed is not None:
            normalized = _normalize_review_payload(parsed)
            if normalized is not None:
                return normalized

        if role == "tool" and tool_name == "submit_review" and parsed is not None:
            normalized = _normalize_review_payload(parsed)
            if normalized is not None:
                return normalized

    raise ValueError("No review JSON found in agent messages")