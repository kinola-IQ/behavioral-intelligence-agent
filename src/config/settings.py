"""Application settings loaded from environment."""
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction


load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # app settings
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Rag Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vectordb_name: str = "review_data"

    @property
    def embedding_function(self) -> HuggingFaceEmbeddingFunction:
        return HuggingFaceEmbeddingFunction(
            model_name=self.embedding_model,
            api_key=os.environ.get("HUGGINGFACE_API_KEY"),
        )


def get_settings() -> Settings:
    return Settings()
