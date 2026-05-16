"""Build and persist vector indexes."""
from langchain.tools import tool

from ..core.utils import VECTORDB
from ..config.settings import Settings
from ..api.schemas import UserRequest

# need to loaded the needed settings
settings = Settings()
embedding_function = settings.embedding_function
vectordb_name = settings.vectordb_name
client = VECTORDB

# get/create database instance
async def _get_collection():
    """gets the vector database"""
    try:
        collection = client.get_collection(
        name=vectordb_name,
        embedding_function=embedding_function)
        return collection
    except ValueError:
        collection = client.create_collection(
                name=vectordb_name,
                embedding_function=embedding_function)
        return collection


@tool
def retrieve_text(persona: UserRequest, query: str, metadata: dict[str, any]):
    """retrieves similar personas based on query"""
    collection = await _get_collection()
    results = collection.query(
            query_texts = [query],
            n_results = 2
            where = metadata
        )