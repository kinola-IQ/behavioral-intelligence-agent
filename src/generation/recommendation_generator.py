"""Generate recommendations from retrieved evidence."""
from transformers import pipeline
from ..core.prompt_engine import recommender_prompt


def recommendation_llm(prompt: str):
    """provides recommendations to users based on persona"""
    try:
        pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-V4-Pro")
        messages = [
            {'role': 'system', 'content': recommender_prompt},
            {'role': 'user', 'content': prompt}
        ]
        return pipe(messages)
    except Exception:
        return "model could not draft a response"
