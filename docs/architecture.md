# Architecture

High-level components:

- **Data**: ingest under `data/raw`, transform to `data/processed`, optional vectors in `data/embeddings`.
- **Library (`src`)**: config, persona/memory/prompt core, embeddings and retrieval, generation, evaluation, FastAPI surface, audit logging.
- **UI (`app`)**: Streamlit multipage app calling `src` modules.
- **Ops**: `Dockerfile` and `docker-compose.yml` for API (and optional Streamlit profile).

Extend this document with diagrams and request flows as the design stabilizes.
