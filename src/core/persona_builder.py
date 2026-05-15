"""Build and maintain behavioural personas from signals."""
from ..api.schemas import UserPersona


def build_persona(persona: UserPersona) -> str:
    """
    convert raw user history into a compact behavioral profile \n
    that the rating and review generator can actually use."""
    return f"""
    rating_bias: {persona.rating_bias}
    sentiment_style: {persona.sentiment_style}
    verbosity: {persona.verbosity}
    tone: {persona.tone}
    top_concerns: {persona.top_concerns}
    preferred_features: {persona.preferred_features}
    avoid_features: {persona.avoid_features}
    category_patterns: {persona.category_patterns}
"""
