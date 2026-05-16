"""Text embedding backends and batch encoding."""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

import numpy as np
import pandas as pd
from typing import Iterable

from ..config.settings import Settings

# settings
settings = Settings()

embedding_model = settings.embedding_model


# documents splitter
def semantic_split(
    model: str = embedding_model,
    breakpoint_threshold_type: str = 'gradient',
    breakpoint_threshold_amount: float = 0.8
):
    """splits documents by semantic meaning"""
    # Initialize the SemanticChunker
    text_splitter = SemanticChunker(
        embeddings=model,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
        )
    return text_splitter


# embedder
def _batch(df: pd.DataFrame, batch_size: int = 100):
    """Yield DataFrame chunks with a counter to conserve memory usage."""
    # Split into chunks
    for counter, chunk in enumerate(
            np.array_split(df, max(1, len(df) // batch_size)), start=1):
        yield counter, chunk


def create_embeddings(
        batch,
        model: str = embedding_model):
    """creates embeddings from either queries or documents"""
    try:
        embed_model = HuggingFaceEmbeddings(model_name=model)
        embeddings = embed_model.embed_documents(batch)
        return embeddings
    except Exception as err:
        raise ValueError("check parameters/arguments") from err

