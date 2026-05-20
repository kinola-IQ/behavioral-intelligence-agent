"""Explore indexed personas and test retrieval filters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from shared import (
    load_persona_library_ids,
    load_persona_record,
    persona_to_retrieval_filters,
    render_api_sidebar,
    render_persona_detail,
)
from src.core.persona_builder import build_user_persona
from src.retrieval.search import retrieve_text

st.set_page_config(page_title="Persona explorer", layout="wide")
render_api_sidebar()

st.title("Persona explorer")
st.caption("Browse `persona_library_flattened.json` and probe metadata retrieval.")

ids = load_persona_library_ids()
if not ids:
    st.error(
        "Persona library not found. Expected at "
        "`data/processed/persona_library_flattened.json`."
    )
    st.stop()

selected_id = st.selectbox("Sample user ID", ids, index=0)
record = load_persona_record(selected_id)

if record is None:
    st.stop()

profile = record.get("behavioral_profile", {})
history = record.get("history", [])
nigerian = record.get("nigerian_adaptation", {})

st.subheader("Indexed behavioural profile")
p_cols = st.columns(4)
p_cols[0].write(f"**Sentiment:** {profile.get('sentiment_bias', '—')}")
p_cols[1].write(f"**Avg rating:** {profile.get('avg_star_rating', '—')}")
p_cols[2].write(f"**Verbal style:** {profile.get('verbal_style', '—')}")
p_cols[3].write(f"**Consistency:** {profile.get('rating_consistency', '—')}")

if nigerian:
    st.markdown(
        f"**Persona type:** {nigerian.get('persona_type', '—')} · "
        f"**Markers:** {', '.join(nigerian.get('suggested_markers', []))}"
    )

with st.expander(f"Review history ({len(history)} snippets)", expanded=False):
    for idx, snippet in enumerate(history[:5], start=1):
        st.markdown(f"**{idx}.** {snippet[:500]}{'…' if len(snippet) > 500 else ''}")

st.divider()
st.subheader("Rebuild structured persona")
st.caption("Runs `build_user_persona` on a condensed summary of this record.")

summary_history = " ".join(history[:2])[:1200]
persona_text = (
    f"{profile.get('sentiment_bias', 'balanced')} reviewer, "
    f"{profile.get('verbal_style', 'moderate')} style."
)
product_placeholder = "Consumer product or local service (context from review history)."

if st.button("Build structured persona", type="primary"):
    persona = build_user_persona(
        user_persona=persona_text,
        user_history=summary_history,
        product_details=product_placeholder,
        sentiment_cue=str(profile.get("sentiment_bias", "balanced")).lower(),
    )
    payload = persona.model_dump()
    st.session_state["explorer_persona"] = payload
    render_persona_detail(payload)

st.divider()
st.subheader("Retrieval probe")
st.caption("Calls `retrieve_text` with metadata filters deduced from the persona.")

n_results = st.slider("Max results", min_value=1, max_value=10, value=5)

if st.button("Retrieve similar histories"):
    persona_payload = st.session_state.get("explorer_persona")
    if not persona_payload:
        filters = {
            "sentiment_bias": profile.get("sentiment_bias"),
            "verbal_style": profile.get("verbal_style"),
            "rating_consistency": profile.get("rating_consistency"),
            "persona_type": nigerian.get("persona_type"),
        }
        filters = {k: v for k, v in filters.items() if v}
    else:
        filters = persona_to_retrieval_filters(persona_payload)

    st.json({"metadata_filters": filters})

    with st.spinner("Querying vector store…"):
        result = retrieve_text.invoke({"metadata": filters, "n_results": n_results})

    count = result.get("count", 0)
    status = result.get("status", "unknown")
    st.metric("Matches", count)
    st.caption(f"Status: {status}")

    reviews = result.get("reviews") or []
    if reviews:
        for idx, review in enumerate(reviews, start=1):
            with st.expander(f"Match {idx}"):
                st.write(review if isinstance(review, str) else str(review))
    elif status != "completed":
        st.info(
            "No matches. Ensure embeddings are built (`scripts/build_embeddings.py`) "
            "and metadata values match indexed labels."
        )
