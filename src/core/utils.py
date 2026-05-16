"""Shared helpers for core modules."""
from langgraph.store.memory import InMemoryStore
import chromadb

from ..config.constants import EMBEDS_DIR

MEMORY: InMemoryStore | None = None
VECTORDB: chromadb | None = None


async def startup_resources() -> None:
    """Start up resources used by the system."""
    global MEMORY, VECTORDB
    MEMORY = InMemoryStore(
        index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    )

    VECTORDB = chromadb.PersistentClient(path=EMBEDS_DIR)
