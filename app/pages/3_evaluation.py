"""Offline evaluation of model outputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from shared import render_api_sidebar
from src.evaluation.evaluator import evaluation_pipeline

st.set_page_config(page_title="Evaluation", layout="wide")
render_api_sidebar()

st.title("Evaluation")
st.caption("Runs `evaluation_pipeline` — helpfulness and plan adherence (OpenEvals judges).")

st.warning(
    "Evaluation calls the configured LLM judge. "
    "Runs can take 30–90 seconds per metric."
)

default_prompt = st.session_state.get("last_review_prompt", "")
if default_prompt and "eval_prompt_input" not in st.session_state:
    st.session_state["eval_prompt_input"] = default_prompt

if default_prompt and st.button("Use last review prompt as input"):
    st.session_state["eval_prompt_input"] = default_prompt
    st.rerun()

col_in, col_out = st.columns(2)
with col_in:
    prompt = st.text_area(
        "Input prompt",
        height=200,
        placeholder="User question or recommendation request…",
        key="eval_prompt_input",
    )
with col_out:
    output = st.text_area(
        "Model output",
        height=200,
        placeholder="Generated recommendation or review text to score…",
        key="eval_output_input",
    )

run = st.button("Run evaluation", type="primary", use_container_width=True)

if run:
    if not prompt.strip() or not output.strip():
        st.error("Both input prompt and model output are required.")
        st.stop()

    with st.spinner("Running LLM-as-judge metrics…"):
        try:
            results = evaluation_pipeline(prompt.strip(), output.strip())
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            st.stop()

    st.success("Evaluation complete")

    for name, payload in results.items():
        st.subheader(name.replace("_", " ").title())
        if isinstance(payload, dict):
            score = payload.get("score")
            comment = payload.get("comment") or payload.get("reasoning")
            if score is not None:
                st.metric("Score", score)
            if comment:
                st.write(comment)
            with st.expander("Full payload"):
                st.json(payload)
        else:
            st.write(payload)
