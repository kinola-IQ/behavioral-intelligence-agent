"""Streamlit entry: multipage Behavioural Intelligence Agent UI."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit_mermaid as stmd
from shared import check_api_health, render_api_sidebar, start_backend


st.set_page_config(
    page_title="Behavioural Intelligence Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "backend_connection" not in st.session_state:
    st.session_state["backend_connection"] = False

if not st.session_state["backend_connection"]:
    start_backend()

render_api_sidebar()

st.title("Behavioural Intelligence Agent")
st.caption(
    "Persona-aware retrieval, review simulation, recommendations, and offline evaluation."
)

ok, status = check_api_health()

if ok:
    st.session_state["backend_connection"] = True

if st.session_state["backend_connection"]:
    st.markdown("""
    This UI mirrors the library architecture:

    1. **User modelling** — infer structured personas from behavioural signals (`model_user`)
    2. **Context store** — persist persona for multi-turn recommendation (`context_store`)
    3. **RAG retrieval** — fetch similar review histories by metadata (`retrieve_text`)
    4. **Generation** — predict reviews or draft recommendations
    5. **Evaluation** — score outputs with helpfulness and plan-adherence judges
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Review generator")
        st.write(
            "Simulate how a specific shopper would rate and review a product "
            "using the LangGraph review agent."
        )
        st.page_link("pages/1_review_generator.py",
                     label="Open review generator →")

    with col2:
        st.subheader("Recommendations")
        st.write(
            "Chat with the recommendation engine, grounded in stored persona "
            "and prior session context."
        )
        st.page_link("pages/2_recommendations.py",
                     label="Open recommendations →")

    with col3:
        st.subheader("Evaluation")
        st.write(
            "Run LLM-as-judge metrics on any prompt/output pair "
            "(helpfulness, plan adherence)."
        )
        st.page_link("pages/3_evaluation.py",
                     label="Open evaluation →")

    st.page_link("pages/4_persona_explorer.py",
                 label="Persona explorer →")

    with st.expander("Request flow (architecture)"):
        stmd.st_mermaid("""
        flowchart LR
            A[Behavioural signals] --> B[model_user]
            B --> C[context_store]
            B --> D[retrieve_text]
            D --> E[Review / Recommender]
            C --> E
            E --> F[evaluation_pipeline]
        """)
else:
    st.error(status)
    st.stop()