# Experiments

This document tracks exploratory work in the repo: notebooks, batch scripts, metrics, and where to store artifacts.

## Hypothesis space

| ID | Hypothesis | How explored |
|----|------------|--------------|
| H1 | Behavioural metadata is enough to retrieve useful review analogues without semantic search | `retrieve_text` metadata filters + manual Streamlit persona explorer |
| H2 | Rule-based `UserPersona` inference is sufficient to steer review tone and rating | `build_user_persona` + review agent vs labelled history in notebooks |
| H3 | Binary-encoded profile features predict outcomes (e.g. consistency class) | `scripts/data_encoding.py`, `04_random-forest-trainning.ipynb` |
| H4 | LLM judges correlate with human usefulness for recommendations | `evaluation_pipeline` / OpenEvals on prompt-output pairs |

## Datasets and versions

| Version | Path | Notes |
|---------|------|-------|
| Flattened library | `data/processed/persona_library_flattened.json` | Keyed by user id; UI persona explorer |
| Flattened CSV/XLSX | `persona_library_flattened.csv`, `.xlsx` | Tabular exports |
| Formatted wide | `persona_libray_formatted_wide.csv` | Input to cleaning |
| Cleaned index set | `persona_libray_cleaned.csv` | **Production index input** for Chroma |
| Encoded wide | `formatted_wide_encoded.csv` | 0/1 categorical encoding for sklearn |

When regenerating data, record:

- Source notebook commit or run date
- Row count and column schema diff
- Whether Chroma was rebuilt (`scripts/build_embeddings.py`)

## Notebooks

| Notebook | Focus |
|----------|-------|
| `01_exploration.ipynb` | Initial data exploration |
| `02_prep_and_eda.ipynb` | Preparation and exploratory analysis |
| `03_eval.ipynb` | Evaluation prototypes |
| `04_random-forest-trainning.ipynb` | Classical model on encoded behavioural features |
| `05_cleaning.ipynb` | Produces `persona_libray_cleaned.csv` from formatted wide |

Run notebooks from project root so relative paths to `data/processed/` resolve.

## Batch scripts

| Script | Status | Purpose |
|--------|--------|---------|
| `scripts/build_embeddings.py` | Implemented | Upsert cleaned CSV into Chroma (`data/embeddings/`) |
| `scripts/data_encoding.py` | Standalone | Maps behavioural/slang columns to binary features |
| `scripts/run_evaluation.py` | Stub | Intended batch eval → `models/evaluation/` |
| `scripts/export_results.py` | Stub | Export API/eval results |
| `scripts/seed_demo_data.py` | Stub | Sample data under `data/samples/` |

## Model and prompt versions

Configured in `src/config/settings.py` (override via `.env` where supported):

| Setting | Default | Used in |
|---------|---------|---------|
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Chroma indexing & memory store embed config |
| `vectordb_name` | `review_data` | Collection name |
| `summarization_model` | `facebook/bart-large-cnn` | `context_summarizer` |
| `chat_model` | `deepseek-ai/DeepSeek-V4-Pro:novita` | Recommendations |
| Groq chat | `llama-4-scout-17b` | Review ReAct agent (`utils.LLM`) |

Log the exact model strings in experiment notes when comparing runs.

## Metrics

### Offline (library)

| Metric | Module | Input |
|--------|--------|-------|
| Helpfulness | `src/evaluation/metrics.helpfulness` | prompt, output |
| Plan adherence | `src/evaluation/metrics.plan_adherence` | prompt, output + `recommendation_plan()` |

Aggregated via `evaluation_pipeline` in `src/evaluation/evaluator.py`.

### Retrieval / generation tests

Pytest modules under `tests/`:

- `test_persona_builder.py` — persona inference
- `test_retrieval.py` — retrieval behaviour
- `test_generation.py` — generation paths
- `test_api.py` — API surface

`tests/experimental-tests.ipynb` — ad-hoc experiments.

### Suggested future metrics

- Retrieval hit rate @ k for held-out metadata filters
- Rating MAE between predicted and historical average star
- Review BLEU/ROUGE vs held-out review (weak for creative text; use with caution)
- Judge inter-rater agreement vs human labels

## Artifacts

Store experiment outputs under:

```
models/evaluation/
  <run-id>/
    config.json      # settings snapshot
    prompts/         # prompt templates used
    predictions.jsonl
    metrics.json
```

The directory is reserved in project layout; populate as batch eval is implemented.

## Example experiment log entry

```markdown
## 2026-05-22 — Index rebuild smoke test

- **Data**: persona_libray_cleaned.csv (N rows from notebook 05)
- **Index**: build_embeddings.py, collection review_data
- **Embedding**: all-MiniLM-L6-v2
- **Observation**: Health Success after API start; retrieve_text returns matches for sentiment_bias=Critical, verbal_style=detailed
- **Next**: Batch helpfulness on 50 held-out prompts via run_evaluation.py
```

## Running an evaluation experiment (manual)

1. Start API with keys set.
2. Generate outputs via Streamlit or `curl` to `/generate_recommendation`.
3. In Streamlit **Evaluation** page (or Python REPL):

```python
from src.evaluation.evaluator import evaluation_pipeline
evaluation_pipeline(prompt="<user question>", output="<model response>")
```

4. Save returned judge payloads under `models/evaluation/<run-id>/`.

## CI experiment surface

Docker image build on `main` does not run ML experiments; it validates container build and push. Add a separate workflow for `pytest` and optional index build if CI resources allow HF model download.
