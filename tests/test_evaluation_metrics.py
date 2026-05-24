from src.evaluation.metrics import _groq_tool_use_fallback, _parse_failed_generation


def test_parse_failed_generation_fraction_score() -> None:
    text = (
        "<function=score>Reasoning: Good answer. "
        "Thus, the score should be: 8/10.</function>"
    )
    result = _parse_failed_generation(text, "helpfulness")
    assert result is not None
    assert result["score"] == 0.8
    assert result["key"] == "helpfulness"


def test_groq_tool_use_fallback_from_error_message() -> None:
    exc = Exception(
        "Error code: 400 - {'error': {'code': 'tool_use_failed', "
        "'failed_generation': '<function=score>Thus, the score should be: true.</function>'}}"
    )
    result = _groq_tool_use_fallback(exc, "plan_adherence")
    assert result is not None
    assert result["score"] is True
