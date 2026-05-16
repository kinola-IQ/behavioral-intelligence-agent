"""Build and persist vector indexes."""

from typing import Any

from langchain_core.tools import tool

from ..core.utils import VECTORDB
from ..config.settings import get_settings


def _get_collection():
    """Return the Chroma collection, creating it when missing."""
    if VECTORDB is None:
        raise RuntimeError(
            "Vector store not initialized; call startup_resources() before retrieval."
        )

    settings = get_settings()
    try:
        return VECTORDB.get_collection(
            name=settings.vectordb_name,
            embedding_function=settings.embedding_function,
        )
    except Exception:
        return VECTORDB.create_collection(
            name=settings.vectordb_name,
            embedding_function=settings.embedding_function,
        )


@tool
def retrieve_text(
    query: str,
    n_results: int = 3,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve semantically similar documents to ground the recommendation.

    Implements the RAG Pipeline stage of the recommendation architecture.
    Queries the vector index for past interactions, product notes, or other
    embedded context that matches the agent's search query, reducing blind
    generation and improving factual alignment in the final recommendation.

    Args:
        query: Natural-language search text (e.g. product category, user
            concern, or paraphrased user request).
        n_results: Maximum number of matching chunks to return. Defaults to
            ``3``.
        metadata: Optional Chroma ``where`` filter dict to restrict results
            (for example ``{"source": "reviews"}``). Pass ``None`` to search
            the full collection.

    Returns:
        A dictionary with:
        - ``documents``: List of retrieved text chunks (may be empty).
        - ``metadatas``: Parallel list of metadata dicts per chunk.
        - ``distances``: Parallel list of similarity distances when available.
        - ``status``: ``"completed"`` on success, or an error descriptor when
          the vector store is unavailable or the query fails.
    """
    if VECTORDB is None:
        return {
            "documents": [],
            "metadatas": [],
            "distances": [],
            "status": "failed: vector store not initialized",
        }

    try:
        collection = _get_collection()
        query_kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": n_results,
        }
        if metadata:
            query_kwargs["where"] = metadata

        results = collection.query(**query_kwargs)
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
            "status": "completed",
        }
    except Exception as exc:
        return {
            "documents": [],
            "metadatas": [],
            "distances": [],
            "status": f"failed: {exc}",
        }
