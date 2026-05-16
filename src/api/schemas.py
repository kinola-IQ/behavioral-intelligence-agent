"""Pydantic request/response models."""
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List, Dict


class HealthResponse(BaseModel):
    """informs ot resource status on startup"""
    status: str


class UserRequest(BaseModel):
    """instructs the agent what to simulate"""
    user_persona: str = Field(...)
    user_history: str = Field(...)
    product_details: str = Field(...)
    sentiment_cue: str = Field(...)


class AgentResponse(BaseModel):
    """agent response format to instructions"""
    predicted_rating: int = Field(...)
    predicted_review: str = Field(...)


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


@dataclass
class Context:
    user_id: str