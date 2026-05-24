"""Predict ratings and reviews from behavioural signals."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from shared import (
    DEMO_PRODUCT_DETAILS,
    DEMO_USER_HISTORY,
    DEMO_USER_PERSONA,
    SENTIMENT_CUES,
    build_persona_from_inputs,
    check_api_health,
    compose_review_prompt,
    post_review,
    render_persona_detail,
)

st.set_page_config(page_title="Review generator", layout="wide")

st.title("Review generator")
st.caption("Maps to `POST /api/v1/generate_review` and `src.generation.review_generator`.")

tab_form, tab_prompt = st.tabs(["Structured inputs", "Free-form prompt"])

with tab_form:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        user_persona = st.text_area(
            "User persona",
            value=DEMO_USER_PERSONA,
            height=100,
            help="Shopping style, priorities, personality in reviews.",
        )
        user_history = st.text_area(
            "User history",
            value=DEMO_USER_HISTORY,
            height=120,
            help="Past ratings, recurring themes, typical review length.",
        )
        product_details = st.text_area(
            "Product details",
            value=DEMO_PRODUCT_DETAILS,
            height=100,
        )
        sentiment_cue = st.selectbox("Sentiment cue", SENTIMENT_CUES, index=0)

    with col_right:
        st.subheader("Structured persona preview")
        st.caption("Local inference via `build_user_persona` (no API call).")
        if st.button("Preview persona", use_container_width=True):
            persona = build_persona_from_inputs(
                user_persona, user_history, product_details, sentiment_cue
            )
            st.session_state["last_persona"] = persona
            render_persona_detail(persona)

        if "last_persona" in st.session_state:
            with st.expander("Last previewed persona", expanded=False):
                st.json(st.session_state["last_persona"])

    composed_prompt = compose_review_prompt(
        user_persona, user_history, product_details, sentiment_cue
    )
    with st.expander("Agent prompt preview"):
        st.code(composed_prompt, language="text")

    generate = st.button("Generate review", type="primary", use_container_width=True)

with tab_prompt:
    free_prompt = st.text_area(
        "Prompt",
        height=220,
        placeholder="Paste a full review-generation prompt or natural-language request.",
    )
    generate_free = st.button(
        "Generate from prompt", type="primary", use_container_width=True
    )

if generate or generate_free:
    prompt = composed_prompt if generate else free_prompt.strip()
    if not prompt:
        st.error("Provide a prompt before generating.")
        st.stop()

    ok, status = check_api_health()
    if not ok:
        st.warning(
            f"{status}\n\n"
            "You can still preview personas locally. Start the API to run the agent."
        )
        st.stop()

    with st.spinner("Running review agent…"):
        try:
            result = post_review(prompt)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    agent_status = result.get("agent_status", "unknown")
    rating = result.get("predicted_rating", 0)
    review = result.get("predicted_review", "")

    if agent_status.lower() != "success":
        st.error(f"Agent status: {agent_status}")
    else:
        st.success("Review generated")

    m1, m2 = st.columns(2)
    m1.metric("Predicted rating", f"{rating} / 5")
    m2.metric("Agent status", agent_status)

    if review:
        st.subheader("Predicted review")
        st.write(review)
    else:
        st.info("No review text returned. Check API logs and `GROQ_API_KEY`.")

    with st.expander("Raw response"):
        st.json(result)

    st.session_state["last_review_prompt"] = prompt
