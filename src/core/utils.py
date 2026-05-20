"""Shared helpers for core modules."""
import os
from transformers import pipeline
from langgraph.store.memory import InMemoryStore
import chromadb
from langchain_groq import ChatGroq

from huggingface_hub import InferenceClient
from ..config.constants import EMBEDS_DIR
from..config.settings import Settings


settings = Settings()

MEMORY: InMemoryStore | None = None
VECTORDB: chromadb | None = None
LLM: ChatGroq | None = None
HF_LLM_PROVIDER: InferenceClient | None = None


async def startup_resources() -> None:
    """Start up resources used by the system."""
    global MEMORY, VECTORDB, LLM, HF_LLM_PROVIDER

    # for context sharing
    MEMORY = InMemoryStore(
        index={
            "dims": 1536,
            "embed": f"huggingface:{settings.embedding_model}",
        }
    )

    # client for vector database operations
    VECTORDB = chromadb.PersistentClient(path=EMBEDS_DIR)

    # model to be used in review generation
    LLM = ChatGroq(
        model="llama-4-scout-17b",
        groq_api_key=os.environ['GROQ_API_KEY']
        )

    # model provider to be used for chat
    HF_LLM_PROVIDER = InferenceClient(
        api_key=os.environ["HUGGINGFACE_API_KEY"],
    )


def startup_status():
    """checks state of the resources needed"""
    if all([MEMORY, VECTORDB, LLM, HF_LLM_PROVIDER]):
        return "Success"
    return "Failed"