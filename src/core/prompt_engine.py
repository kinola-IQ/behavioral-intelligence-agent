"""Compose prompts from templates and runtime context."""
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.tools import tool
from .memory_layer import retrieve_user_persona


# makeshift session(aka long-term) memory for multiturn conversation
session = []

def session_store(prompt=None, model_response=None, fetch=False):
    """Store or retrieve conversation history."""

    if fetch:
        return session

    if prompt is None or model_response is None:
        raise ValueError("Both prompt and model_response are required.")

    session.append({
        "user_previously_asked": prompt,
        "assistant_responded_with": model_response
    })

    return session


# main review prompt
def review_generation_prompt():
    return """
    You are a user-modeling engine.

    Workflow:
    1. Use model_user, context_store, and retrieve_text when they help.
    2. Finish by calling submit_review with predicted_rating (1-5) and predicted_review.

    Do not end with plain text. Always call submit_review for the final answer.
    """


# recommendation engine prompt
def recommender_prompt():
    """Prompt template for recommendation generation."""
    prompt_template = """
    You are a recommendation generation engine.

    TASK
    Generate exactly one personalized recommendation for the user using only the provided inputs.

    INSTRUCTIONS
    1. Use the user persona, prior session context, questionto produce the recommendation.
    2. Do not invent facts, preferences, products, or history that are not present in the inputs.
    3. Match the wording and tone of the recommendation to the sentiment style.
    4. Use prior interaction context only as supporting context. If the session is empty, treat this as a fresh session.
    5. If the persona context is empty, missing, or contains no usable information, do not attempt a recommendation. Instead, reply exactly with:
    "Please complete the review generation step first, then try again."
    6. Be concise, specific, and directly relevant to the user.
    7. Output only the final recommendation text. Do not add explanations, headings, labels, or analysis.
    8. respond in times new roman font

    CONTEXT
    User persona:
    {user_persona}

    Previous interaction:
    {session_history}

    User question:
    {question}
    """
    return PromptTemplate.from_template(prompt_template)


# evaluator prompts
# plan adherence prompt
def recommendation_plan(persona: str = ""):
    """Generation plan used to evaluate recommendation outputs."""
    persona = retrieve_user_persona()
    return f"""
    Rules:
    - Be accurate and do not hallucinate missing facts.
    - Match tone to the sentiment cue.
    - Base the persona on the following:
    {persona}
"""