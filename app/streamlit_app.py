"""Streamlit entry: multipage Behavioural Intelligence Agent UI."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit_mermaid as stmd
from shared import check_api_health, start_backend
from theme import (
    init_academic_page,
    render_abstract,
    render_academic_footer,
    render_module_card,
    render_section_heading,
    render_status_badge,
)

# need access to endpoints
start_backend()

st.set_page_config(
    page_title="Behavioural Intelligence Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_academic_page(
    title="Aligning Behavioural Heuristics with\
            Agentic RAG for Culturally Adapted \
            Consumer Simulation ",
    subtitle=(
        "Persona-aware retrieval, review simulation, recommendations, "
        "and auto-evaluation — a research interface for behavioural modelling."
    ),
    section="Research Platform",
    sidebar_brand=True,
)

ok, status = check_api_health()
render_status_badge(ok, "System online" if ok else "Backend starting")

if ok:
    render_abstract([
        "<strong>User modelling</strong> — infer structured personas from behavioural signals (<code>model_user</code>)",
        "<strong>Context store</strong> — persist persona for multi-turn recommendation (<code>context_store</code>)",
        "<strong>RAG retrieval</strong> — fetch similar review histories by metadata (<code>retrieve_text</code>)",
        "<strong>Generation</strong> — predict reviews or draft recommendations",
        "<strong>Evaluation</strong> — score outputs with helpfulness and plan-adherence judges",
    ])

    render_section_heading("Research Modules")

    col1, col2, col3 = st.columns(3)

    with col1:
        render_module_card(
            1,
            "Review Generator",
            "Simulate how a specific shopper would rate and review a product "
            "using the LangGraph review agent.",
        )
        st.page_link(
            "pages/1_review_generator.py",
            label="Proceed to review generator →",
        )

    with col2:
        render_module_card(
            2,
            "Recommendations",
            "Chat with the recommendation engine, grounded in stored persona "
            "and prior session context.",
        )
        st.page_link(
            "pages/2_recommendations.py",
            label="Proceed to recommendations →",
        )

    with col3:
        render_module_card(
            3,
            "Evaluation",
            "Run LLM-as-judge metrics on any prompt/output pair "
            "(helpfulness, plan adherence).",
        )
        st.page_link(
            "pages/3_evaluation.py",
            label="Proceed to evaluation →",
        )

    st.markdown('<hr class="academic-rule">', unsafe_allow_html=True)

    render_module_card(
        4,
        "Persona Explorer",
        "Browse the indexed persona library and probe metadata-filtered retrieval.",
    )
    st.page_link(
        "pages/4_persona_explorer.py",
        label="Proceed to persona explorer →",
    )

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

    render_academic_footer()
