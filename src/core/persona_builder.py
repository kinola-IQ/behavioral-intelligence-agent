"""Build and maintain behavioural personas from signals."""

import re
from typing import Any

from langchain_core.tools import tool

from ..api.schemas import UserPersona

_STAR_PATTERN = re.compile(r"(\d)\s*stars?", re.IGNORECASE)
_GIVES_PATTERN = re.compile(r"gives?\s+(\d)", re.IGNORECASE)
_PRICE_PATTERN = re.compile(r"\$\s*(\d+(?:\.\d+)?)")

_TONE_KEYWORDS = (
    "practical",
    "enthusiastic",
    "cautious",
    "critical",
    "balanced",
    "price-sensitive",
    "premium",
    "detailed",
    "analytical",
    "generous",
)


def _persona_payload(persona: UserPersona) -> dict[str, Any]:
    return {
        "rating_bias": persona.rating_bias,
        "sentiment_style": persona.sentiment_style,
        "verbosity": persona.verbosity,
        "tone": persona.tone,
        "top_concerns": persona.top_concerns,
        "preferred_features": persona.preferred_features,
        "avoid_features": persona.avoid_features,
        "category_patterns": persona.category_patterns,
    }


def _infer_rating_bias(user_history: str, sentiment_cue: str) -> int:
    ratings: list[int] = []
    for pattern in (_STAR_PATTERN, _GIVES_PATTERN):
        ratings.extend(int(match) for match in pattern.findall(user_history))

    if ratings:
        average = sum(ratings) / len(ratings)
        return max(1, min(5, round(average)))

    cue = sentiment_cue.strip().lower()
    if cue in {"critical", "negative", "harsh"}:
        return 2
    if cue in {"generous", "positive", "enthusiastic"}:
        return 4
    return 3


def _infer_sentiment_style(sentiment_cue: str) -> str:
    return sentiment_cue.strip().lower() or "balanced"


def _infer_verbosity(user_history: str) -> str:
    word_count = len(user_history.split())
    if word_count > 80:
        return "verbose"
    if word_count < 25:
        return "concise"
    return "moderate"


def _infer_tone(*texts: str) -> list[str]:
    combined = " ".join(texts).lower()
    found = [keyword for keyword in _TONE_KEYWORDS if keyword in combined]
    return found[:5] if found else ["balanced"]


def _split_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for segment in re.split(r"[.;]", text):
        cleaned = segment.strip()
        if 5 < len(cleaned) <= 120:
            phrases.append(cleaned)
    return phrases


def _infer_top_concerns(user_persona: str, user_history: str) -> list[str]:
    concern_markers = ("complain", "concern", "issue", "unless", "worry", "avoid")
    combined = f"{user_persona} {user_history}"
    concerns = [
        phrase
        for phrase in _split_phrases(combined)
        if any(marker in phrase.lower() for marker in concern_markers)
    ]
    return concerns[:5] or _split_phrases(combined)[:3] or ["value for money"]


def _infer_preferred_features(user_persona: str, user_history: str, product_details: str) -> list[str]:
    positive_markers = ("values", "highlights", "enjoys", "likes", "prefers", "strong", "excellent", "good")
    combined = f"{user_persona} {user_history} {product_details}"
    preferred = [
        phrase
        for phrase in _split_phrases(combined)
        if any(marker in phrase.lower() for marker in positive_markers)
    ]
    return preferred[:5] or _split_phrases(product_details)[:3] or ["core product quality"]


def _infer_avoid_features(user_persona: str, user_history: str, product_details: str) -> list[str]:
    negative_markers = ("dislike", "avoid", "complain", "weak", "cheap", "narrow", "overpriced", "average")
    combined = f"{user_persona} {user_history} {product_details}"
    avoided = [
        phrase
        for phrase in _split_phrases(combined)
        if any(marker in phrase.lower() for marker in negative_markers)
    ]
    return avoided[:5] or ["unmet expectations"]


def _infer_category_patterns(product_details: str, user_history: str) -> dict[str, str]:
    combined = f"{product_details} {user_history}".lower()
    patterns: dict[str, str] = {}

    price_match = _PRICE_PATTERN.search(product_details)
    if price_match:
        patterns["price_anchor"] = price_match.group(1)

    for label, keyword in (
        ("feature_focus", "battery"),
        ("feature_focus", "sound"),
        ("feature_focus", "comfort"),
        ("feature_focus", "build quality"),
    ):
        if keyword in combined:
            patterns[label] = keyword
            break
    else:
        patterns["feature_focus"] = "general"

    return patterns


def build_user_persona(
    user_persona: str,
    user_history: str,
    product_details: str,
    sentiment_cue: str,
) -> UserPersona:
    """Build a structured persona from raw behavioural signals."""
    return UserPersona(
        rating_bias=_infer_rating_bias(user_history, sentiment_cue),
        sentiment_style=_infer_sentiment_style(sentiment_cue),
        verbosity=_infer_verbosity(user_history),
        tone=_infer_tone(user_persona, user_history),
        top_concerns=_infer_top_concerns(user_persona, user_history),
        preferred_features=_infer_preferred_features(user_persona, user_history, product_details),
        avoid_features=_infer_avoid_features(user_persona, user_history, product_details),
        category_patterns=_infer_category_patterns(product_details, user_history),
    )


@tool
def model_user(
    user_persona: str,
    user_history: str,
    product_details: str,
    sentiment_cue: str,
) -> dict[str, Any]:
    """Convert raw behavioural signals into a structured user persona.

    Implements the User Modelling stage of the recommendation architecture.
    The agent supplies unstructured inputs from the current query and any
    available history; this tool normalizes them into a compact, model-ready
    profile (rating tendency, tone, concerns, preferences, and category focus).

    Args:
        user_persona: Free-text description of who the user is (shopping style,
            priorities, personality in reviews).
        user_history: Past review or interaction patterns (typical ratings,
            recurring themes, length of reviews).
        product_details: Description of the product or context being evaluated.
        sentiment_cue: Desired sentiment for the output (e.g. ``critical``,
            ``balanced``, ``generous``).

    Returns:
        A dictionary matching the ``UserPersona`` schema with keys:
        ``rating_bias``, ``sentiment_style``, ``verbosity``, ``tone``,
        ``top_concerns``, ``preferred_features``, ``avoid_features``, and
        ``category_patterns``. Pass this object to ``context_store`` and the
        prompt synthesizer for downstream generation.
    """
    persona = build_user_persona(
        user_persona=user_persona,
        user_history=user_history,
        product_details=product_details,
        sentiment_cue=sentiment_cue,
    )
    return _persona_payload(persona)
