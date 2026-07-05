"""Academic visual theme for the Behavioural Intelligence Agent UI."""

from __future__ import annotations

import streamlit as st

ACADEMIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif !important;
}

h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'Crimson Text', Georgia, serif !important;
    color: #1B365D !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1100px;
}

.academic-masthead {
    background: linear-gradient(135deg, #1B365D 0%, #2C4A6E 100%);
    border-bottom: 4px solid #B8860B;
    padding: 2rem 2.5rem;
    margin: -1rem -1rem 2rem -1rem;
}

.academic-masthead h1 {
    color: #F8F6F0 !important;
    font-size: 2rem;
    margin: 0;
    font-weight: 700;
}

.academic-masthead .subtitle {
    color: #D4CFC4;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-style: italic;
    line-height: 1.5;
}

.academic-masthead .section-label {
    display: inline-block;
    background: #B8860B;
    color: #1B365D;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 0.2rem 0.6rem;
    margin-bottom: 0.75rem;
}

.module-card {
    background: #FFFFFF;
    border: 1px solid #C9B99A;
    border-top: 3px solid #1B365D;
    padding: 1.5rem 1.5rem 1rem;
    min-height: 220px;
    box-shadow: 0 2px 8px rgba(27, 54, 93, 0.08);
}

.module-card .module-num {
    font-family: 'Crimson Text', serif;
    font-size: 2.5rem;
    color: #B8860B;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.module-card h3 {
    font-size: 1.15rem !important;
    margin: 0 0 0.75rem 0 !important;
    color: #1B365D !important;
}

.module-card p {
    color: #4A4540;
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0;
}

.abstract-panel {
    background: #FFFFFF;
    border: 1px solid #C9B99A;
    border-left: 4px solid #B8860B;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
}

.abstract-panel .abstract-label {
    font-family: 'Crimson Text', serif;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #1B365D;
    font-weight: 700;
    margin-bottom: 0.75rem;
}

.abstract-panel ol {
    margin: 0;
    padding-left: 1.25rem;
    color: #2D2A26;
    line-height: 1.75;
}

.abstract-panel li {
    margin-bottom: 0.35rem;
}

.academic-rule {
    border: none;
    border-top: 1px solid #C9B99A;
    margin: 2rem 0;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #C9B99A;
    padding: 0.75rem 1rem;
    border-radius: 2px;
}

[data-testid="stMetricLabel"] {
    font-family: 'Crimson Text', serif !important;
    color: #1B365D !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.75rem !important;
}

[data-testid="stSidebar"] {
    background-color: #EDE8DC !important;
    border-right: 1px solid #C9B99A !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #2D2A26;
}

.sidebar-brand {
    padding: 0.5rem 0 1rem;
    border-bottom: 1px solid #C9B99A;
    margin-bottom: 1rem;
}

.sidebar-brand .brand-title {
    font-family: 'Crimson Text', serif;
    color: #1B365D;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0;
    font-weight: 700;
}

.sidebar-brand .brand-subtitle {
    font-size: 0.75rem;
    color: #6B6560;
    font-style: italic;
    margin: 0.35rem 0 0 0;
    line-height: 1.4;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background-color: #1B365D !important;
    border: 1px solid #1B365D !important;
    font-family: 'Source Serif 4', serif !important;
    letter-spacing: 0.02em;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background-color: #2C4A6E !important;
    border-color: #B8860B !important;
}

[data-testid="stExpander"] summary,
.streamlit-expanderHeader {
    font-family: 'Crimson Text', serif !important;
    color: #1B365D !important;
}

.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid #C9B99A;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Crimson Text', serif !important;
    color: #1B365D !important;
    background-color: transparent !important;
}

.stTabs [aria-selected="true"] {
    border-bottom: 3px solid #B8860B !important;
}

a[data-testid="stPageLink-NavLink"] {
    font-family: 'Source Serif 4', serif !important;
    color: #1B365D !important;
    font-weight: 600;
}

.academic-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #C9B99A;
    font-size: 0.8rem;
    color: #6B6560;
    font-style: italic;
    text-align: center;
    line-height: 1.6;
}

.status-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    border-radius: 2px;
}

.status-badge--online {
    background: #E8F0E8;
    color: #2D5A2D;
    border: 1px solid #8FB08F;
}

.status-badge--offline {
    background: #F5EDE8;
    color: #7A4A2A;
    border: 1px solid #C9A88A;
}

.section-heading {
    font-family: 'Crimson Text', serif;
    color: #1B365D;
    font-size: 1.35rem;
    border-bottom: 1px solid #C9B99A;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}
</style>
"""


def inject_academic_theme() -> None:
    st.markdown(ACADEMIC_CSS, unsafe_allow_html=True)


def render_masthead(title: str, subtitle: str, section: str = "Research Platform") -> None:
    st.markdown(
        f"""
        <div class="academic-masthead">
            <div class="section-label">{section}</div>
            <h1>{title}</h1>
            <div class="subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <p class="brand-title">BIA Research Lab</p>
                <p class="brand-subtitle">Behavioural Intelligence Agent — academic prototype</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_module_card(number: int, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="module-card">
            <div class="module-num">{number:02d}</div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_abstract(items: list[str]) -> None:
    rows = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f"""
        <div class="abstract-panel">
            <div class="abstract-label">System Overview</div>
            <ol>{rows}</ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(online: bool, label: str) -> None:
    css_class = "status-badge--online" if online else "status-badge--offline"
    st.markdown(
        f'<span class="status-badge {css_class}">{label}</span>',
        unsafe_allow_html=True,
    )


def render_section_heading(text: str) -> None:
    st.markdown(f'<p class="section-heading">{text}</p>', unsafe_allow_html=True)


def render_academic_footer(note: str = "") -> None:
    default = (
        "Behavioural Intelligence Agent — persona-aware retrieval, "
        "generation, and evaluation research interface."
    )
    st.markdown(
        f'<div class="academic-footer">{note or default}</div>',
        unsafe_allow_html=True,
    )


def init_academic_page(
    title: str | None = None,
    subtitle: str | None = None,
    section: str = "Research Module",
    *,
    sidebar_brand: bool = False,
) -> None:
    inject_academic_theme()
    if sidebar_brand:
        render_sidebar_brand()
    if title:
        render_masthead(title, subtitle or "", section)
