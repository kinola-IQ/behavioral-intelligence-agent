# Behavioural Intelligence Agent

A persona-aware system that models shoppers from behavioural signals, retrieves similar review histories from a vector store, generates simulated product reviews, and produces grounded recommendations. The project ships as a Python library (`src/`), a FastAPI backend, a multipage Streamlit UI, batch scripts, and notebooks for data preparation and experimentation.

## What it does

1. **User modelling** — Turn free-text persona, review history, product context, and a sentiment cue into a structured `UserPersona` (`model_user` / `build_user_persona`).
2. **Context persistence** — Store the persona in an in-memory LangGraph store for multi-turn recommendation (`context_store`).
3. **Retrieval (RAG)** — Filter Chroma by behavioural metadata (rating consistency, sentiment bias, verbal style, persona type, slang markers) and return matching review histories (`retrieve_text`).
4. **Generation** — A LangGraph ReAct agent predicts rating + review; a separate Hugging Face chat path drafts recommendations from stored persona and session history.
5. **Evaluation** — LLM-as-judge metrics (helpfulness, plan adherence) via OpenEvals.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/` | Core library: config, persona/memory, embeddings, retrieval, generation, evaluation, API, logging |
| `app/` | Streamlit multipage UI (review generator, recommendations, evaluation, persona explorer) |
| `data/raw`, `data/processed`, `data/embeddings` | Datasets and persisted Chroma store |
| `models/` | Prompt assets, evaluation outputs, trained artifacts (as added) |
| `notebooks/` | Exploration, prep/EDA, evaluation, cleaning, encoding experiments |
| `scripts/` | Batch jobs: embeddings index, evaluation, export, demo seed |
| `docs/` | Architecture, methodology, experiments, deployment |
| `tests/` | Pytest suite for persona, retrieval, generation, API |

## Prerequisites

- Python **3.11+**
- API keys (see `.env.example`):
  - `GROQ_API_KEY` — review-generation agent (LangGraph + Groq)
  - `HUGGINGFACE_API_KEY` — embeddings, summarization, recommendation chat

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and set your keys.

### Build the vector index

From the project root (required so `src` resolves, or use editable install above):

```bash
python scripts/build_embeddings.py
```

This reads `data/processed/persona_libray_cleaned.csv` and upserts records into `data/embeddings/` (Chroma persistent client).

### Run the API

```bash
uvicorn src.api.main:server --reload --host 0.0.0.0 --port 8000
```

Health: `GET http://localhost:8000/api/v1/health`

Endpoints:

- `POST /api/v1/generate_review` — review simulation agent
- `POST /api/v1/generate_recommendation` — persona-grounded recommendation chat

### Run the Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

The UI calls the API by default (`API_BASE_URL`, default `http://localhost:8000/api/v1`). Start the API first for live generation.

### Docker

```bash
docker compose up --build          # API on :8000
docker compose --profile ui up --build   # API + Streamlit on :8501
```

See [docs/deployment.md](docs/deployment.md) for production-oriented notes.

## Documentation

- [Architecture](docs/architecture.md) — components and request flows
- [Methodology](docs/methodology.md) — data, personas, retrieval, grounding
- [Experiments](docs/experiments.md) — notebooks, metrics, artifacts
- [Deployment](docs/deployment.md) — Docker, CI, secrets, operations

## Development

```bash
pytest
ruff check src tests app scripts   # if ruff installed via dev extras
```

Install dev dependencies: `pip install -e ".[dev]"`.

## License

See [LICENSE](LICENSE).
