"""Evaluation metrics (retrieval, generation quality)."""

import os
import re
from typing import Any, Callable

from langchain_groq import ChatGroq
from openevals.llm import create_llm_as_judge
from openevals.prompts import PLAN_ADHERENCE_PROMPT, RAG_HELPFULNESS_PROMPT

from ..core.prompt_engine import recommendation_plan

# Groq's 8b model does not reliably emit OpenEvals' structured score tool calls.
EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "llama-3.3-70b-versatile")


def _evaluation_judge() -> ChatGroq:
    """Return a Groq model suited for OpenEvals structured judge calls."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for evaluation judges.")

    return ChatGroq(
        model=EVAL_JUDGE_MODEL,
        groq_api_key=api_key,
        temperature=0,
        max_retries=2,
    )


def _llm_evaluator(
    prompt: PLAN_ADHERENCE_PROMPT | RAG_HELPFULNESS_PROMPT,
    feedback_key: str,
):
    return create_llm_as_judge(
        prompt=prompt,
        feedback_key=feedback_key,
        judge=_evaluation_judge(),
        use_reasoning=False,
    )


def _parse_failed_generation(text: str, feedback_key: str) -> dict[str, Any] | None:
    """Recover a judge score when Groq returns tool_use_failed."""
    cleaned = re.sub(r"</?function[^>]*>", "", text).strip()
    if not cleaned:
        return None

    score: bool | float | None = None
    bool_match = re.search(r"score should be:\s*(true|false)\b", cleaned, re.I)
    if bool_match:
        score = bool_match.group(1).lower() == "true"
    else:
        num_match = re.search(
            r"score should be:\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+))?",
            cleaned,
            re.I,
        )
        if num_match:
            value = float(num_match.group(1))
            denominator = num_match.group(2)
            if denominator:
                score = value / float(denominator)
            elif value > 1:
                score = value / 10.0
            else:
                score = value

    if score is None:
        return None

    return {"key": feedback_key, "score": score, "comment": cleaned}


def _groq_tool_use_fallback(exc: Exception, feedback_key: str) -> dict[str, Any] | None:
    message = str(exc)
    if "tool_use_failed" not in message and "failed_generation" not in message:
        return None

    match = re.search(
        r"failed_generation['\"]:\s*['\"](.+?)['\"]\s*[,}]",
        message,
        re.DOTALL,
    )
    if not match:
        return None

    failed_generation = match.group(1).encode().decode("unicode_escape")
    return _parse_failed_generation(failed_generation, feedback_key)


def _run_evaluator(
    evaluator: Callable[..., dict[str, Any]],
    feedback_key: str,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        return evaluator(**kwargs)
    except Exception as exc:
        fallback = _groq_tool_use_fallback(exc, feedback_key)
        if fallback is not None:
            return fallback
        raise


def helpfulness(input: str, output: str) -> dict[str, Any]:
    evaluator = _llm_evaluator(RAG_HELPFULNESS_PROMPT, feedback_key="helpfulness")
    return _run_evaluator(
        evaluator,
        feedback_key="helpfulness",
        inputs=input,
        outputs=output,
    )


def plan_adherence(input: str, output: str) -> dict[str, Any]:
    evaluator = _llm_evaluator(PLAN_ADHERENCE_PROMPT, feedback_key="plan_adherence")
    return _run_evaluator(
        evaluator,
        feedback_key="plan_adherence",
        inputs=input,
        outputs=output,
        plan=recommendation_plan(),
    )
