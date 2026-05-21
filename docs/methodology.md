# Methodology

This document describes how behavioural data becomes personas, how retrieval grounds generation, and what assumptions the system makes.

## Problem framing

The agent simulates **how a particular kind of shopper would review or reason about a product**, not generic copywriting. Grounding comes from:

1. **Structured user modelling** from the current query (persona text, history, product details, sentiment cue).
2. **Similar-user evidence** from a library of real review histories indexed by behavioural metadata.
3. **Explicit guardrails** in recommendation prompts (no invented facts; require prior persona when empty).

## Data sources

### Persona library

Processed artifacts under `data/processed/` derive from a flattened persona/review library:

| Artifact | Description |
|----------|-------------|
| `persona_library_flattened.json` / `.csv` | Nested JSON flattened to dot/bracket column names (`behavioral_profile.*`, `history[n]`, `nigerian_adaptation.*`) |
| `persona_libray_formatted_wide.csv` | Wide table used in cleaning notebooks |
| `persona_libray_cleaned.csv` | **Index source** for Chroma — cleaned, encoding-ready behavioural fields |
| `formatted_wide_encoded.csv` | Categorical fields mapped to 0/1 for sklearn experiments (see `scripts/data_encoding.py`) |

Note: several filenames use the historical typo **libray** instead of **library**.

### Record schema (indexing)

Each row represents one user (`record_id`) with:

- **Review history** — `history[0]` … `history[4]` (text reviews).
- **Behavioural profile** — `behavioral_profile.rating_consistency`, `sentiment_bias`, `verbal_style`, `avg_star_rating`.
- **Nigerian adaptation** — `nigerian_adaptation.persona_type`, `suggested_markers[0..2]` (slang/style markers).

The indexer (`src/embeddings/indexer.py`) maps these into Chroma:

- **Document**: list of review strings (stored as the user’s document payload).
- **Metadata** (lowercased at upload): `rating consistency`, `sentiment bias`, `verbal style`, `persona type`, `slangs` (list of marker strings).

## Persona construction rules

`build_user_persona` / `model_user` use **deterministic heuristics** (no LLM) over the four input strings:

| Field | Inference logic |
|-------|-----------------|
| `rating_bias` | Average of star patterns in history (`N stars`, `gives N`); else map from sentiment cue (critical → 2, generous → 4, default 3) |
| `sentiment_style` | Normalized `sentiment_cue` |
| `verbosity` | Word count: &gt;80 → detailed, &lt;25 → concise, else moderate |
| `tone` | Keyword scan (practical, enthusiastic, cautious, critical, …) |
| `top_concerns` | Phrases with concern markers (complain, issue, worry, …) |
| `preferred_features` | Phrases with positive markers (values, enjoys, excellent, …) |
| `avoid_features` | Phrases with negative markers (dislike, cheap, overpriced, …) |
| `category_patterns` | Price anchor from `$` in product text; feature focus from keywords (battery, sound, comfort, build quality) |

The API schema `UserPersona` is the contract for tools and Streamlit previews.

### Mapping persona → retrieval filters

The agent (or UI demo via `persona_to_retrieval_filters`) should align inferred persona with indexed metadata:

| Persona field | Chroma metadata key | Typical values |
|---------------|---------------------|----------------|
| `sentiment_style` | `sentiment_bias` | critical, generous |
| `verbosity` | `verbal_style` | detailed, concise |
| `rating_bias` | `rating_consistency` | stable / volatile (heuristic thresholds in UI) |
| `tone[0]` or archetype | `persona_type` | e.g. Lagos Consumer Proxy labels in data |
| slang usage | `slangs` | sha, correct, abeg, too much, standard, non-standard |

`retrieve_text` accepts snake_case aliases (`rating_consistency`, etc.) and builds Chroma `$eq` / `$in` / `$and` filters.

## Retrieval strategy

Retrieval is **metadata filtering only** — there is no query embedding step at retrieval time. The design assumes:

- Persona dimensions in the library are sufficiently discriminative.
- The agent deduces the best matching metadata set after `model_user`.
- Returned documents may be **summarized** (`context_summarizer`) to limit prompt size.

Implications:

- Index quality and consistent labelling in `persona_libray_cleaned.csv` directly affect RAG usefulness.
- Cold-start users with rare metadata combinations may get zero matches (`status: failed` or empty lists).

## Generation methodology

### Review simulation

The review path uses a **ReAct agent** (Groq) with tools for modelling, storage, and retrieval. The system prompt requires **JSON-only** output with `predicted_rating` and `predicted_review`. The agent decides tool order; typical flow: model → store → retrieve → generate.

### Recommendations

Recommendations use a **single-shot HF chat** prompt with:

- Stored persona text (`retrieve_user_persona`)
- Session history stub (`session_store` — in-process list; not durable across restarts)
- User question

If persona context is empty, the template instructs the model to refuse with a message directing the user to complete review generation first.

## Evaluation methodology

Offline evaluation uses **LLM-as-judge** (OpenEvals + Groq):

| Metric | Intent |
|--------|--------|
| **Helpfulness** | RAG-style: is the output useful given the input? |
| **Plan adherence** | Does the output follow a generation plan derived from persona rules? |

Streamlit page 3 and `evaluation_pipeline` expose these for ad-hoc runs. Batch reporting via `scripts/run_evaluation.py` is reserved for future automation into `models/evaluation/`.

## Labelling and cultural adaptation assumptions

- **Nigerian adaptation** fields encode persona archetypes and suggested slang markers used as retrieval features, not as language enforcement in the LLM.
- **Sentiment bias** and **verbal style** in source data are categorical labels assigned during dataset construction (see notebooks `02_prep_and_eda`, `05_cleaning`).
- Encoding script maps categories to binary features for classical ML (e.g. random forest notebook `04_random-forest-trainning.ipynb`); the live agent uses string metadata in Chroma, not the encoded CSV.

## Limitations

- In-memory persona store does not survive API restarts or horizontal scaling.
- Persona builder heuristics may miss nuance compared to human labels.
- Metadata-only retrieval cannot rank by semantic similarity to a product description.
- Review agent JSON parsing in routes assumes well-formed agent output; malformed responses surface as failed agent status.

## Reproducibility checklist

1. Regenerate processed CSVs from notebooks if source JSON changes.
2. Run `python scripts/build_embeddings.py` from repo root after `.env` has `HUGGINGFACE_API_KEY` (embeddings download).
3. Start API (`startup_resources`) before UI or integration tests that call live endpoints.
4. Record prompt versions and model IDs from `Settings` when logging experiments (see [experiments.md](experiments.md)).
