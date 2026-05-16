from src.core.persona_builder import build_user_persona


def test_build_user_persona_extracts_structured_fields() -> None:
    persona = build_user_persona(
        user_persona="A practical, price-sensitive shopper who values durability.",
        user_history="Usually gives 3 stars unless the product exceeds expectations.",
        product_details="Wireless earbuds, $79, strong battery life, average sound quality.",
        sentiment_cue="critical",
    )

    assert persona.rating_bias == 3
    assert persona.sentiment_style == "critical"
    assert persona.verbosity in {"concise", "moderate", "verbose"}
    assert persona.tone
    assert persona.top_concerns
    assert persona.preferred_features
    assert persona.category_patterns["price_anchor"] == "79"
