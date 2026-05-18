import pytest

from src.retrieval.search import _build_where_clause


def test_build_where_clause_single_field() -> None:
    clause = _build_where_clause({"sentiment_bias": "critical"})
    assert clause == {"sentiment bias": {"$eq": "critical"}}


def test_build_where_clause_multiple_fields() -> None:
    clause = _build_where_clause(
        {
            "verbal_style": "concise",
            "persona_type": "practical_shopper",
        }
    )
    assert "$and" in clause
    assert len(clause["$and"]) == 2


def test_build_where_clause_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _build_where_clause({})
