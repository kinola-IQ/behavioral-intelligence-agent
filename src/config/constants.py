"""Shared constants (paths, defaults, feature flags)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
EMBEDS_DIR = PROJECT_ROOT / "data" / "embeddings"
