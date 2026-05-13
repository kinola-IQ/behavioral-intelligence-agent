# Behavioural Intelligence Agent

Project scaffold for persona-aware retrieval, generation, evaluation, and a Streamlit UI.

## Layout

- `src/` — application library (config, core, data, embeddings, retrieval, generation, evaluation, API, logging)
- `app/` — Streamlit multipage app
- `data/` — raw, processed, embeddings, samples
- `models/` — prompts, evaluation assets, trained artifacts
- `notebooks/` — exploration, prep/EDA, evaluation
- `scripts/` — batch jobs (embeddings, evaluation, export, demo seed)
- `docs/` — architecture, methodology, experiments, deployment
- `tests/` — pytest suite

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and adjust values.

Run the API (after implementation):

```bash
uvicorn src.api.main:app --reload
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Docker

```bash
docker compose up --build
```
