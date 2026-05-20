"""Generate recommendations from retrieved evidence."""
import asyncio
from ..core import utils
from ..core.prompt_engine import recommender_prompt, session_store
from ..core.memory_layer import retrieve_user_persona
from ..config.settings import Settings


def recommendation_llm(question: str) -> str:
    """Provides recommendations to users based on persona."""
    settings = Settings()

    if utils.HF_LLM_PROVIDER is None:
        raise RuntimeError(
            "HF_LLM_PROVIDER not initialized. Run startup_resources() first."
        )

    try:
        # define prompt format
        prompt = recommender_prompt().format(
            user_persona=asyncio.wait_for(retrieve_user_persona(), timeout=15),
            session_history=asyncio.wait_for(session_store(), timeout=15),
            question=question,
        )
        completion = utils.HF_LLM_PROVIDER.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )

        return completion.choices[0].message.content.strip()

    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}")