"""Build and persist vector indexes."""
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import math

from ..core.utils import VECTORDB
from ..config.settings import get_settings


# connects to db to be used
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


# upload the document to the vector database
def chunks(dataframe, batch_size: int = 100) -> Iterable[Tuple]:
    """Yield dataframe in memory-efficient batches with a counter."""
    # Calculate number of splits (ceil ensures we don't miss leftover rows)
    num_splits = math.ceil(len(dataframe) / batch_size)
    for counter, batch in enumerate(np.array_split(dataframe, num_splits), start=1):
        yield batch, counter


# function to coalate records for upload
def _prepare_batch_records(batch) -> List[Dict]:
    """
    Prepares records for each user in the batch for upload to a vector database.
    
    Args:
        batch (pd.DataFrame): A dataframe containing user records.
    
    Returns:
        List[Dict]: A list of dictionaries, each representing a vector record.
    """
    vectors = []

    for _, row in batch.iterrows():
        # User ID
        user_id = str(row['record_id'])

        # Reviews (assuming columns are named 'history[0]', 'history[1]', etc.)
        reviews = [row.get(f'history[{i}]') for i in range(5)]

        # Nigerian slangs used (assuming columns exist)
        slangs = [row.get(f'nigerian_adaptation.suggested_markers[{i}]') for i in range(3)]

        # Metadata
        metadata = {
            'rating consistency': row.get('behavioral_profile.rating_consistency'),
            'sentiment bias': row.get('behavioral_profile.sentiment_bias'),
            'verbal style': row.get('behavioral_profile.verbal_style'),
            'persona type': row.get('nigerian_adaptation.persona_type'),
            'slangs': slangs
        }

        # Vector record
        vectors.append({
            'user id': user_id,
            'reviews': reviews,
            'metadata': metadata
        })

    return vectors

def upload_data(dataframe):
    """Upload dataframe records to vector database in batches."""
    collection = _get_collection()
    for batch, _ in chunks(dataframe):
        vectors = _prepare_batch_records(batch)
        # Upsert each record in the batch
        for data in vectors:
            collection.upsert(
                ids=[data['user id']],
                documents=[data['reviews']],
                metadatas=[data['metadata']]
            )
        
