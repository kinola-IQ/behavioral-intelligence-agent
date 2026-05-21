"""summarizes retrieved reviews for downstream generation."""
from ..core import utils
from ..config.settings import Settings

def summarize_context(context: list[str]) -> str:
    """Summarizes context into a more compact formats."""
    settings = Settings()
    try:
        # format into single multi-line string
        reviews_str = "\n".join(context)
        # summarize into more compact formats
        summary = utils.HF_LLM_PROVIDER.summarization(
            reviews_str, model=settings.summarization_model)
        return summary['summary_text']
    except Exception as err:
        raise Exception("could not summarize") from err