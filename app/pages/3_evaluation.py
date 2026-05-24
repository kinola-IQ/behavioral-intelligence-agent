"""Offline evaluation of model outputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from shared import render_payload

st.set_page_config(page_title="Evaluation", layout="wide")

st.title("Evaluation")
st.caption("Runs `evaluation_pipeline` — helpfulness and plan adherence (OpenEvals judges).")

st.warning(
    "Evaluation calls the configured LLM judge. "
    "Runs can take 30–90 seconds per metric."
)

default_prompt = st.session_state.get("last_review_prompt", "")
if "eval_result" not in st.session_state:
    st.warning("use recommendation chat before this page loads")
    st_autorefresh(interval=2000)
else:
    col_in, col_out = st.columns(2)
    with col_in:
        prompt = st.session_state.user_prompt
        with st.expander("user prompt"):
            st.code(prompt, language="text")
        render_payload('helpfulness')

    with col_out:
        response = st.session_state.assistant_reponse
        with st.expander("model response"):
            st.code(response, language="text")
        render_payload('plan_adherence')
