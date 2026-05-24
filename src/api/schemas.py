"""Pydantic request/response models."""
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class HealthResponse(BaseModel):
    """informs ot resource status on startup"""
    status: str


class UserRequest(BaseModel):
    """request gotten from the user"""
    prompt: str = Field(...)


class AgentResponse(BaseModel):
    """agent response format"""
    predicted_rating: int = Field(...)
    predicted_review: str = Field(...)
    agent_status: str = Field(...)


class ChatResponse(BaseModel):
    """chat respose format"""
    response_text: str
    eval_result: Any


class UserPersona(BaseModel):
    """summarize the user into model-ready signals."""
    rating_bias: int = Field(...)
    sentiment_style: str = Field(...)
    verbosity: str = Field(...)
    tone: List[str] = Field(...)
    top_concerns: List[str] = Field(...)
    preferred_features: List[str] = Field(...)
    avoid_features: List[str] = Field(...)
    category_patterns: Dict[str, str] = Field(...)


class ReviewRetrievalCriteria(BaseModel):
    """Behavioural metadata filters for retrieving matching review histories.

    Fields align with Chroma metadata stored at index time. The agent should
    populate these after deducing fit from a ``UserPersona`` (see
    ``retrieve_text`` docstring for suggested mappings).
    """

    rating_consistency: str | None = Field(
        default=None,
        description="Indexed as 'rating consistency' (i.e stable, variable).",
    )
    sentiment_bias: str | None = Field(
        default=None,
        description=(
            "Indexed as 'sentiment bias'; often maps from sentiment_style."
        ),
    )
    verbal_style: str | None = Field(
        default=None,
        description="Indexed as 'verbal style'; often maps from verbosity.",
    )
    persona_type: str | None = Field(
        default=None,
        description="Indexed as 'persona type'; cultural or archetype label.",
    )
    slangs: str | list[str] | None = Field(
        default=None,
        description=(
            "Indexed as 'slangs'; marker phrase(s) for the user style."
        ),
    )


@dataclass
class Context:
    user_id: str
