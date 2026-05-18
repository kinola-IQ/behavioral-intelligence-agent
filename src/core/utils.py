"""Shared helpers for core modules."""
import os
from langgraph.store.memory import InMemoryStore
import chromadb
from langchain_groq import ChatGroq
from ..config.constants import EMBEDS_DIR

MEMORY: InMemoryStore | None = None
VECTORDB: chromadb | None = None
LLM: ChatGroq | None = None


async def startup_resources() -> None:
    """Start up resources used by the system."""
    global MEMORY, VECTORDB, LLM

    # for context sharing
    MEMORY = InMemoryStore(
        index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    )

    # client for vector database operations
    VECTORDB = chromadb.PersistentClient(path=EMBEDS_DIR)

    # model to be used in review generation
    LLM = ChatGroq(
        model="llama-4-scout-17b",
        groq_api_key=os.environ['GROQ_API_KEY']
        )


def startup_status():
    """informs of the status of resources"""
    if MEMORY and VECTORDB and LLM is not None:
        return 'Success'
    return 'Failed' 