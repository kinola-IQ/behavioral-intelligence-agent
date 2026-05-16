"""Build and maintain behavioural personas from signals."""
from typing import Any

from langchain_core.tools import tool

from ..api.schemas import UserPersona


def _persona_payload(persona: UserPersona) -> dict[str, Any]:
    return {
        "rating_bias": persona.rating_bias,
        "sentiment_style": persona.sentiment_style,
        "tone": persona.tone,
        "top_concerns": persona.top_concerns,
        "preferred_features": persona.preferred_features,
        "avoid_features": persona.avoid_features,
        "category_patterns": persona.category_patterns,
    }


@tool
def model_user(
    user_persona: str,
    user_history: str,
    product_details: str,
    sentiment_cue: str

) -> dict[str, Any]:
    """Convert raw prompt into a compact behavioral profile."""
    return {
        "user_persona": user_persona,
        "user_history": user_history,
        "product_details": product_details,
        "sentiment_cue": sentiment_cue
    }