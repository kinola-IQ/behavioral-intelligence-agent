"""Generate recommendations from retrieved evidence."""
from ..core import utils
from ..core.prompt_engine import recommender_prompt, session_store
from ..core.memory_layer import retrieve_user_persona
from ..config.settings import Settings
from ..logging.audit_log import log_event


def recommendation_llm(question: str) -> str:
    """Provides recommendations to users based on persona."""
    settings = Settings()

    if utils.HF_LLM_PROVIDER is None:
        raise RuntimeError(
            "HF_LLM_PROVIDER not initialized. Run startup_resources() first."
        )

    try:
        log_event("recommendation_llm_invoke", question_chars=len(question))
        # define prompt format
        prompt = recommender_prompt().format(
            user_persona=retrieve_user_persona(),
            session_history=session_store(fetch=True),
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