"""Build or refresh embedding index from processed data."""
import pandas as pd
from src.embeddings.indexer import upload_data
from src.logging.audit_log import log_event


def main() -> None:
    """pipeline for indexing and uploading to vectordabase"""
    try:
        data = pd.read_csv(
            r'data\processed\persona_library_formatted_wide.csv')
        upload_data(data)
        return log_event('vector database populated successfully')
    except Exception as exc:
        raise SystemExit("Implement embedding build pipeline.") from exc


if __name__ == "__main__":
    main()
