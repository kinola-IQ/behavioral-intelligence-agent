"""Build or refresh embedding index from processed data."""
from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import pandas as pd

# Allow `python scripts/build_embeddings.py` without an editable install.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.constants import EMBEDS_DIR, PROCESSED_DATA_DIR
from src.core import utils
from src.embeddings.indexer import upload_data
from src.logging.audit_log import log_event

# Processed artifact name (historical typo: "libray" not "library")
DEFAULT_DATA_FILE = PROCESSED_DATA_DIR / "persona_libray_cleaned.csv"
FALLBACK_DATA_FILE = PROCESSED_DATA_DIR / "persona_library_cleaned.csv"


def _resolve_data_path() -> Path:
    if DEFAULT_DATA_FILE.exists():
        return DEFAULT_DATA_FILE
    if FALLBACK_DATA_FILE.exists():
        return FALLBACK_DATA_FILE
    raise FileNotFoundError(
        f"No cleaned persona CSV found. Expected one of:\n"
        f"  - {DEFAULT_DATA_FILE}\n"
        f"  - {FALLBACK_DATA_FILE}"
    )


def _ensure_vectordb() -> None:
    """Initialize Chroma only (no LLM keys required for indexing)."""
    if utils.VECTORDB is None:
        utils.VECTORDB = chromadb.PersistentClient(path=EMBEDS_DIR)


def main() -> None:
    """Load cleaned personas and upsert review histories into Chroma."""
    data_path = _resolve_data_path()
    try:
        _ensure_vectordb()
        dataframe = pd.read_csv(data_path)
        upload_data(dataframe)
        print(f"Vector database populated from {data_path.name}")
        log_event("vector database populated successfully", path=str(data_path))
    except Exception as exc:
        raise SystemExit(f"Embedding build failed: {exc}") from exc


if __name__ == "__main__":
    main()
