"""Multi-turn recommendation chat."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from shared import (
    check_api_health,
    post_recommendation,
    render_api_sidebar,
)

st.set_page_config(page_title="Recommendations", layout="wide")
render_api_sidebar()

st.title("Recommendations")
st.caption("Maps to `POST /api/v1/generate_recommendation` and `recommendation_llm`.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if st.sidebar.button("Clear chat"):
    st.session_state.chat_messages = []
    st.rerun()

persona = st.session_state.get("last_persona")
if persona:
    with st.sidebar.expander("Persona from review page", expanded=True):
        st.json(persona)
else:
    st.sidebar.info(
        "Build a persona on **Review generator** (Preview persona) "
        "to see it here during recommendations."
    )

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask for a product recommendation…")

if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    ok, status = check_api_health()
    if not ok:
        assistant_text = (
            f"Cannot reach the API ({status}). "
            "Start the server and ensure `GROQ_API_KEY` / model dependencies are set."
        )
    else:
        with st.spinner("Generating recommendation…"):
            try:
                assistant_text = post_recommendation(user_input)
                if isinstance(assistant_text, list):
                    assistant_text = str(assistant_text)
            except Exception as exc:
                assistant_text = f"Request failed: {exc}"

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": str(assistant_text)}
    )
    with st.chat_message("assistant"):
        st.markdown(str(assistant_text))

with st.expander("Example prompts"):
    st.markdown(
        """
- *Based on my persona, recommend wireless earbuds under $100 with strong battery life.*
- *I avoid plastic builds and average sound — what should I consider instead?*
- *Compare two running shoes for a cautious reviewer who values comfort.*
        """
    )
