"""Retrieve review histories from the vector store by behavioural metadata."""

from typing import Any

from langchain_core.tools import tool

from ..core.utils import VECTORDB
from ..embeddings.indexer import _get_collection
from .context_summarizer import summarize_context
# Snake_case aliases the agent may pass after deducing filters from UserPersona.
_METADATA_ALIASES: dict[str, str] = {
    "rating_consistency": "rating consistency",
    "sentiment_bias": "sentiment bias",
    "verbal_style": "verbal style",
    "persona_type": "persona type",
}


def _normalize_metadata_key(key: str) -> str:
    return _METADATA_ALIASES.get(key, key)


def _build_where_clause(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build a Chroma ``where`` filter from agent-deduced metadata."""
    conditions: list[dict[str, Any]] = []

    for raw_key, value in metadata.items():
        if value is None or value == "":
            continue

        key = _normalize_metadata_key(raw_key)
        if isinstance(value, list):
            conditions.append({key: {"$in": value}})
        else:
            conditions.append({key: {"$eq": value}})

    if not conditions:
        raise ValueError("metadata must include at least one non-empty filter")

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


@tool
def retrieve_text(
    metadata: dict[str, Any],
    n_results: int = 5,
) -> dict[str, Any]:
    """Fetch review histories that match persona-aligned behavioural metadata.

    Implements the RAG Pipeline stage of the recommendation architecture.
    After ``model_user`` produces a structured persona, the agent deduces which
    indexed behavioural attributes best describe that user and passes them here.
    This tool performs **metadata filtering only** (no semantic search query):
    it returns stored review histories for users whose profile metadata matches.

    Indexed metadata fields (use these keys or snake_case aliases):
        - ``rating consistency`` / ``rating_consistency``: either Stable or Volatile.
        - ``sentiment bias`` / ``sentiment_bias``: either critical or generous.
        - ``verbal style`` / ``verbal_style``: either detailed or concise.
        - ``persona type`` / ``persona_type``: 'Lagos Consumer Proxy' label.
        - ``slangs``: marker phrases (string or list for ``$in`` matching), 
                    ``slangs`` options include and are limited to: sha, correct, abeg, too much, standard, non-standard.

    Suggested mapping from ``UserPersona`` when deducing filters:
        - ``sentiment_style`` → ``sentiment_bias``
        - ``verbosity`` → ``verbal_style``
        - ``rating_bias`` → map to the closest ``rating_consistency`` label
        - dominant ``tone`` entry → ``persona_type`` when applicable

    Args:
        metadata: Non-empty dict of metadata filters the agent inferred from the
            current persona. Values must match what was indexed at upload time.
        n_results: Maximum number of matching user records to return. Defaults
            to ``5``.

    Returns:
        A dictionary with:
        - ``reviews``: List of review history payloads (one per matched user).
        - ``metadatas``: Parallel list of stored metadata dicts per match.
        - ``ids``: Parallel list of user record ids in the vector store.
        - ``count``: Number of matches returned.
        - ``status``: ``"completed"`` on success, or an error descriptor when
          the store is unavailable, filters are empty, or retrieval fails.
    """
    if VECTORDB is None:
        return {
            "reviews": [],
            "metadatas": [],
            "ids": [],
            "count": 0,
            "status": "failed: vector store not initialized",
        }

    if not metadata:
        return {
            "reviews": [],
            "metadatas": [],
            "ids": [],
            "count": 0,
            "status": "failed: metadata filters are required",
        }

    try:
        where_clause = _build_where_clause(metadata)
        collection = _get_collection()
        results = collection.get(
            where=where_clause,
            limit=n_results,
            include=["documents", "metadatas"],
        )

        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        ids = results.get("ids") or []

        # need to summarize retrieved reviews to prevent bloating the model
        summarized_docs = summarize_context(documents)
        return {
            "reviews": summarized_docs,
            "metadatas": metadatas,
            "ids": ids,
            "count": len(ids),
            "status": "completed",
        }
    except ValueError as exc:
        return {
            "reviews": [],
            "metadatas": [],
            "ids": [],
            "count": 0,
            "status": f"failed: {exc}",
        }
    except Exception as exc:
        return {
            "reviews": [],
            "metadatas": [],
            "ids": [],
            "count": 0,
            "status": f"failed: {exc}",
        }
