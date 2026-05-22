"""Generate behavioural review text from context."""

from langgraph.prebuilt import create_react_agent

# prompt
from ..core.prompt_engine import review_generation_prompt

# model
from ..core import utils

# tools
from ..core.persona_builder import model_user
from ..core.memory_layer import context_store
from ..retrieval.search import retrieve_text

# agent
def review_generator_agent():
    """orchestrator in charge of deducing user persona and predicting review and rating"""
    return create_react_agent(
    utils.LLM,
    tools = [model_user, context_stosre, retrieve_text],
    prompt= review_generation_prompt()
    )