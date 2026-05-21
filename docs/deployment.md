# Deployment

Guidance for running the Behavioural Intelligence Agent locally, in Docker, and in CI-built container images.

## Environments

| Environment | Typical use | Notes |
|-------------|-------------|-------|
| **Local dev** | Notebooks, Streamlit, hot-reload API | `.env` with dev keys; Chroma on disk under `data/embeddings/` |
| **Docker Compose** | Single-host demo / integration | API always; Streamlit via `ui` profile |
| **CI images** | GHCR + Docker Hub on `main` push | See `.github/workflows/docker-image.yml` |

`APP_ENV` and `LOG_LEVEL` are read from `.env` via Pydantic settings (defaults: `development`, `INFO`).

## Secrets and configuration

Copy `.env.example` to `.env`:

```env
APP_ENV=development
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

GROQ_API_KEY=<required for review agent>
HUGGINGFACE_API_KEY=<required for embeddings, summarization, recommendations>
```

| Variable | Required for | Consumed by |
|----------|--------------|-------------|
| `GROQ_API_KEY` | Review generation, eval judges | `utils.startup_resources` → `ChatGroq` |
| `HUGGINGFACE_API_KEY` | Index embeddings, summarization, recommendation chat | Chroma embedding function, `InferenceClient` |

Never commit `.env`. Mount secrets via orchestrator secret stores in production.

Optional future variables (`DATABASE_URL`) are placeholders for external persistence not yet wired.

## Local deployment

### 1. Install

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare vector index

```bash
python scripts/build_embeddings.py
```

Run from the **repository root**. The script adds the project root to `sys.path` if needed and initializes Chroma without starting full API resources (no Groq key required for indexing).

### 3. Start API

```bash
uvicorn src.api.main:server --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

Expected payload when all resources initialized: `{"status":"Success"}`.

### 4. Start UI (optional)

```bash
streamlit run app/streamlit_app.py
```

Set `API_BASE_URL` if the API is not on `http://localhost:8000/api/v1`.

## Docker deployment

### Build

The `Dockerfile`:

- Base: `python:3.12-slim`
- Installs `requirements.txt` and editable package (`pip install -e .`)
- Copies `src/` and `app/`
- Exposes 8000 (API) and 8501 (Streamlit)

### Compose

`docker-compose.yml` defines:

| Service | Port | Command |
|---------|------|---------|
| `api` | 8000 | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` |
| `streamlit` (profile `ui`) | 8501 | `streamlit run app/streamlit_app.py ...` |

Volumes mount `./data`, `./models`, and `./logs` into `/app` so Chroma and artifacts persist on the host.

**Note:** The FastAPI module exports the app instance as `server` in `src/api/main.py`. For a working stack, prefer:

```yaml
command: uvicorn src.api.main:server --host 0.0.0.0 --port 8000
```

Align compose with that if health checks fail against the default `app` attribute.

### Commands

```bash
# API only
docker compose up --build

# API + Streamlit
docker compose --profile ui up --build
```

### Production-like considerations

- **Persistence**: Mount `data/embeddings` and processed CSVs; rebuild index when data changes.
- **Memory**: Sentence-transformers and HF models increase image size and RAM; consider pre-baking embeddings in the image build stage.
- **Scaling**: Current `InMemoryStore` is single-process; run one API replica per user session namespace or externalize memory before horizontal scale-out.
- **Timeouts**: Streamlit uses 120s HTTP timeouts for generation; place reverse-proxy timeouts accordingly.
- **Observability**: Enable structured logging for `audit` logger; extend with request IDs in FastAPI middleware as needed.

## CI/CD

Workflow **Docker Image CI** (on push to `main`):

1. Checkout
2. Login to GHCR and Docker Hub (secrets: `GITHUB_TOKEN`, `DOCKER_USERNAME`, `DOCKER_TOKEN`)
3. Build image tagged `behavioral-intelligence-agent-backend:latest` and `:<sha>`
4. Push to `ghcr.io/kinola-iq/...` and Docker Hub

Deploy pulled images with the same env vars and volume mounts as compose. Run embedding build in an init job or bake `data/embeddings` into the image if the dataset is static.

## Health and readiness

| Check | Endpoint / signal |
|-------|-------------------|
| Liveness | `GET /api/v1/health` returns 200 |
| Readiness | `startup_status() == "Success"` (MEMORY, VECTORDB, LLM, HF client all set) |

If health returns `Failed`, verify keys, network access to Groq/HF, and that Chroma path is writable.

## Operational runbook

| Task | Command |
|------|---------|
| Refresh vector index | `python scripts/build_embeddings.py` |
| Batch evaluation (stub) | `python scripts/run_evaluation.py` |
| Export results (stub) | `python scripts/export_results.py` |
| Demo seed (stub) | `python scripts/seed_demo_data.py` |

## Security

- Restrict API exposure behind auth gateway for public deployments (no auth in scaffold).
- Rotate Groq and Hugging Face keys independently.
- Do not mount `.env` into public images; inject at runtime.
