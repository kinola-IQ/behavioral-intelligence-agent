"""Shared Streamlit helpers for the Behavioural Intelligence Agent UI."""

from __future__ import annotations

import json
import os
import socket
import threading
import uvicorn
import sys
from pathlib import Path
from typing import Any
import time

import httpx
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
for _path in (_PROJECT_ROOT, _APP_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.config.constants import PROCESSED_DATA_DIR
from src.core.persona_builder import build_user_persona
from src.logging.audit_log import configure_audit_logging, log_event

from src.config.settings import Settings

# we need access to the server ports and other relevant variables
settings = Settings()


DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
PERSONA_LIBRARY_PATH = PROCESSED_DATA_DIR / "persona_library_flattened.json"


SENTIMENT_CUES = [
    "critical",
    "Generous",
   
]

DEMO_USER_PERSONA = (
    "A practical, price-sensitive shopper who values durability and "
    "complains when products feel overpriced."
)
DEMO_USER_HISTORY = (
    "Usually gives 3 stars unless the product clearly exceeds expectations. "
    "Often mentions build quality, value for money, and battery life."
)
DEMO_PRODUCT_DETAILS = (
    "Wireless earbuds, $79, strong battery life, average sound quality, "
    "plastic build, no active noise cancellation."
)


def api_base_url() -> str:
    return st.session_state.get("api_base_url", DEFAULT_API_BASE).rstrip("/")


# backend startup
def port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def run_backend():
    configure_audit_logging(settings.log_level)
    log_event(
        "streamlit_backend_start",
        host=settings.api_host,
        port=settings.api_port,
    )
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False
    )


@st.cache_resource
def start_backend():
     # we need the server up and running for requests to go through to the backend
    if not port_in_use(settings.api_host, settings.api_port):
        threading.Thread(
            target=run_backend,
            daemon=True
        ).start()


def compose_review_prompt(
    user_persona: str,
    user_history: str,
    product_details: str,
    sentiment_cue: str,
) -> str:
    """Format a review request the agent expects."""
    return (
        "Given the following input schema:\n"
        f"user_persona: '{user_persona.strip()}'\n"
        f"user_history: '{user_history.strip()}'\n"
        f"product_details: '{product_details.strip()}'\n"
        f"sentiment_cue: '{sentiment_cue.strip()}'\n"
        "Generate the predicted rating and review."
    )


def check_api_health(timeout: float = 3.0) -> tuple[bool, str]:
    """Return (ok, status_message) from the FastAPI health endpoint."""
    try:
        response = httpx.get(f"{api_base_url()}/health", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status", "unknown")
        return status.lower() in {"ok", "success", "healthy"}, str(status)
    except httpx.ConnectError:
        return False, "Backend is booting up, please wait a few minutes"
    except Exception as exc:
        return False, f"Health check failed: {exc}"


def post_review(prompt: str, timeout: float = 120.0) -> dict[str, Any]:
    response = httpx.post(
        f"{api_base_url()}/generate_review",
        json={"prompt": prompt},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def post_recommendation(prompt: str, timeout: float = 120.0) -> str:
    response = httpx.post(
        f"{api_base_url()}/generate_recommendation",
        json={"prompt": prompt},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def persona_to_retrieval_filters(persona: dict[str, Any]) -> dict[str, str]:
    """Map UserPersona fields to Chroma metadata keys for retrieval demos."""
    rating_bias = persona.get("rating_bias", 3)
    if rating_bias <= 2:
        consistency = "Volatile"
    elif rating_bias >= 4:
        consistency = "stable"
    else:
        consistency = "stable"

    tone = persona.get("tone") or []
    persona_type = tone[0] if tone else "balanced"

    return {
        "sentiment_bias": str(persona.get("sentiment_style", "")).title(),
        "verbal_style": str(persona.get("verbosity", "")),
        "rating_consistency": consistency,
        "persona_type": str(persona_type),
    }


@st.cache_data(show_spinner=False)
def load_persona_library_ids(limit: int = 200) -> list[str]:
    if not PERSONA_LIBRARY_PATH.exists():
        return []
    with PERSONA_LIBRARY_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return list(data.keys())[:limit]


@st.cache_data(show_spinner=False)
def load_persona_record(user_id: str) -> dict[str, Any] | None:
    if not PERSONA_LIBRARY_PATH.exists():
        return None
    with PERSONA_LIBRARY_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get(user_id)


def render_persona_metrics(persona: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Rating bias", persona.get("rating_bias", "—"))
    cols[1].metric("Sentiment", persona.get("sentiment_style", "—"))
    cols[2].metric("Verbosity", persona.get("verbosity", "—"))
    cols[3].metric("Tone tags", len(persona.get("tone", [])))


def render_persona_detail(persona: dict[str, Any]) -> None:
    render_persona_metrics(persona)
    for label, key in (
        ("Tone", "tone"),
        ("Top concerns", "top_concerns"),
        ("Preferred features", "preferred_features"),
        ("Avoid", "avoid_features"),
    ):
        values = persona.get(key, [])
        if values:
            st.markdown(f"**{label}:** {', '.join(str(v) for v in values)}")
    patterns = persona.get("category_patterns", {})
    if patterns:
        st.markdown(f"**Category patterns:** `{patterns}`")


def build_persona_from_inputs(
    user_persona: str,
    user_history: str,
    product_details: str,
    sentiment_cue: str,
) -> dict[str, Any]:
    model = build_user_persona(
        user_persona=user_persona,
        user_history=user_history,
        product_details=product_details,
        sentiment_cue=sentiment_cue,
    )
    return model.model_dump()


def render_payload(name: str):
    payload = st.session_state.eval_result.get("payload", {})
    st.subheader(
        name.replace("_", " ").title(),
        help="""
        0 = agent failed to meet requiremnts
        1 = the agent met requirements""")
    results = payload.get(name, {})
    score = int(results.get("score", ""))
    comment = comment = results.get("comment","")

    if score is not None:
        st.metric("Score", score)
    if comment:
        st.write(comment)