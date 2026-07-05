# Behavioural Intelligence Agent

**A research prototype for persona-aware review simulation, retrieval-augmented generation, and automated evaluation of shopper behaviour.**

---

## Abstract

This repository implements a *Behavioural Intelligence Agent* (BIA): a modular system that infers structured user personas from behavioural signals, retrieves analogous review histories from a vector index, and generates persona-conditioned product reviews and recommendations. The pipeline couples deterministic user modelling with metadata-filtered retrieval (RAG), LangGraph-based review generation, and LLM-as-judge evaluation (helpfulness, plan adherence). The artifact is organised as a Python library (`src/`), a FastAPI service layer, a multipage Streamlit research interface, batch preprocessing scripts, and exploratory notebooks.

---

## Research Problem

Consumer-facing language models often produce generic product discourse that does not reflect individual shopping styles, rating tendencies, or review verbosity. BIA addresses this gap by treating **behavioural persona** as a first-class object: inferred from text signals, persisted across dialogue turns, and used both to filter retrieval and to condition generation. The system is designed for **simulation** (how would *this* shopper review *this* product?) rather than open-ended copywriting.

---

## System Pipeline

The end-to-end workflow comprises five stages:

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **User modelling** | Map free-text persona, review history, product context, and sentiment cue to a structured `UserPersona` (`model_user` / `build_user_persona`). |
| 2 | **Context persistence** | Store inferred persona in a LangGraph in-memory store for multi-turn recommendation (`context_store`). |
| 3 | **Retrieval (RAG)** | Query Chroma with behavioural metadata filters—rating consistency, sentiment bias, verbal style, persona type, slang markers—and return matching review histories (`retrieve_text`). |
| 4 | **Generation** | A LangGraph ReAct agent predicts rating and review text; a separate Hugging Face chat path drafts recommendations grounded in stored persona and session history. |
| 5 | **Evaluation** | Score model outputs with OpenEvals judges for helpfulness and plan adherence. |

### Architecture

![System design plan](https://github.com/kinola-IQ/behavioral-intelligence-agent/blob/6c73103a24115db6badd785a4fa97ae51a75ae22/docs/system%20design%20plan.png)

For component-level detail, request flows, and module responsibilities, see [docs/architecture.md](docs/architecture.md).

---

## Repository Structure

| Path | Role |
|------|------|
| `src/` | Core library: configuration, persona inference, memory, embeddings, retrieval, generation, evaluation, API, audit logging |
| `app/` | Streamlit research interface (review generator, recommendations, evaluation, persona explorer) |
| `data/raw`, `data/processed`, `data/embeddings` | Source datasets and persisted Chroma vector store |
| `models/` | Prompt assets, evaluation outputs, and trained artifacts |
| `notebooks/` | Exploratory analysis, preprocessing, encoding experiments, and offline evaluation |
| `scripts/` | Batch jobs: embedding index construction, evaluation runs, result export, demo seeding |
| `docs/` | Architecture, methodology, experiments, deployment, and solution paper |
| `tests/` | Pytest suite covering persona construction, retrieval, generation, and API contracts |

---

## Requirements

- **Python** 3.11 or later
- **API credentials** (see `.env.example`):
  - `GROQ_API_KEY` — review-generation agent (LangGraph + Groq)
  - `HUGGINGFACE_API_KEY` — embeddings, summarization, and recommendation chat

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and configure the required keys before running generation or evaluation endpoints.

---

## Reproducibility

### 1. Construct the vector index

From the project root:

```bash
python scripts/build_embeddings.py
```

This reads `data/processed/persona_libray_cleaned.csv` and upserts behavioural records into `data/embeddings/` (Chroma persistent client). Index construction is a prerequisite for metadata-filtered retrieval experiments.

### 2. Start the API service

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Health check:** `GET http://localhost:8000/api/v1/health`

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v1/generate_review` | Persona-conditioned review simulation |
| `POST` | `/api/v1/generate_recommendation` | Multi-turn, persona-grounded recommendation dialogue |

### 3. Launch the research interface

```bash
streamlit run app/streamlit_app.py
```

The interface communicates with the API at `API_BASE_URL` (default: `http://localhost:8000/api/v1`). Start the API before initiating live generation or evaluation runs.

### 4. Containerised deployment

```bash
docker compose up --build                        # API on port 8000
docker compose --profile ui up --build           # API + Streamlit on port 8501
```

See [docs/deployment.md](docs/deployment.md) for production-oriented configuration.

---

## Documentation

| Document | Contents |
|----------|----------|
| [Architecture](docs/architecture.md) | Component design, data flow, and API integration |
| [Methodology](docs/methodology.md) | Data schema, persona inference rules, retrieval grounding |
| [Experiments](docs/experiments.md) | Notebooks, metrics, and recorded artifacts |
| [Deployment](docs/deployment.md) | Docker, CI, secrets management, and operations |
| [Solution paper](docs/behavioral_intelligence_agent_solution_paper.pdf) | Extended design and evaluation narrative |

---

## Development and Testing

```bash
pytest
ruff check src tests app scripts    # requires dev extras
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

---

## License

See [LICENSE](LICENSE).
