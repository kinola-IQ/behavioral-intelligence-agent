"""summarizes retrieved reviews for downstream generation."""
from transformers import pipeline
from ..config.settings import Settings

# we want the model to be used for summarization available
settings = Settings()
summarizer_model = settings.summarization_model


def summarize_reviews(reviews: list[str]) -> str:
    """Summarizes a list of reviews into a single string."""
    try:
        # format into single multi-line string
        reviews_str = "\n".join(reviews)
        summarizer = pipeline(
            "summarization", model=summarizer_model)
        summary = summarizer(
            reviews_str, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as err:
        raise Exception("could not summarize") from err