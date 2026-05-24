"""Tools used by the review-generation ReAct agent."""

from typing import Any

from langchain_core.tools import tool


@tool
def submit_review(predicted_rating: int, predicted_review: str) -> dict[str, Any]:
    """Submit the final predicted star rating and review text.

    Call this once you have enough context from other tools. This ends the task.

    Args:
        predicted_rating: Integer from 1 to 5.
        predicted_review: The generated review text in the user's voice.
    """
    rating = max(1, min(5, int(predicted_rating)))
    review = str(predicted_review).strip()
    if not review:
        raise ValueError("predicted_review must not be empty")
    return {
        "predicted_rating": rating,
        "predicted_review": review,
    }
