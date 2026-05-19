"""Build or refresh embedding index from processed data."""
import pandas as pd
from src.embeddings.indexer import upload_data
from src.logging.audit_log import log_event
from src.config.constants import PROCESSED_DATA_DIR


def main() -> None:
    """pipeline for indexing and uploading to vectordabase"""
    data = PROCESSED_DATA_DIR / "persona_library_formatted_wide.csv"
    try:
        data = pd.read_csv(data)
        upload_data(data)
        return log_event('vector database populated successfully')
    except Exception as exc:
        raise SystemExit("Implement embedding build pipeline.") from exc


if __name__ == "__main__":
    main()
