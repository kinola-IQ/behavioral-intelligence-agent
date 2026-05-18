"""Compose prompts from templates and runtime context."""
import asyncio

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.messages import SystemMessage
from .memory_layer import retrieve_user_persona

# review generator
def prompt_setup():
    """sets up structure, examples and guardrails to inform model behaviour"""
    # prompt to set in role play
    system_prompt = SystemMessage(
        "You are a user-modeling engine. Given the user persona, user history, product details, "
        "and sentiment cue, infer how this user would realistically rate and review the product. "
        "Preserve the user's style, tone, and rating tendency. Output only valid JSON with the keys "
        "`predicted_rating` and `predicted_review`."
    )

    # guiding model response with examples to learn from
    examples = [
        {
            "question": (
                "Given the following input schema:\n"
                "user_persona: 'A practical, price-sensitive shopper who values durability and complains when products feel overpriced.'\n"
                "user_history: 'Usually gives 3 stars unless the product clearly exceeds expectations. Often mentions build quality, value for money, and battery life.'\n"
                "product_details: 'Wireless earbuds, $79, strong battery life, average sound quality, plastic build, no active noise cancellation.'\n"
                "sentiment_cue: 'critical'\n"
                "Generate the predicted rating and review."
            ),
            "answer": (
                "{\n"
                "  \"predicted_rating\": 3,\n"
                "  \"predicted_review\": \"The battery life is solid and the earbuds are easy to use, but the build feels cheap and the sound is only average for the price. It gets the job done, but I expected better value overall.\"\n"
                "}"
            ),
        },
        {
            "question": (
                "Given the following input schema:\n"
                "user_persona: 'An enthusiastic buyer who enjoys premium products and tends to leave detailed positive reviews.'\n"
                "user_history: 'Frequently gives 4 to 5 stars, highlights design, performance, and overall polish, and rarely complains unless the product is badly flawed.'\n"
                "product_details: 'Mechanical keyboard, $149, aluminum frame, hot-swappable switches, RGB lighting, excellent typing feel.'\n"
                "sentiment_cue: 'generous'\n"
                "Generate the predicted rating and review."
            ),
            "answer": (
                "{\n"
                "  \"predicted_rating\": 5,\n"
                "  \"predicted_review\": \"This keyboard feels premium right away. The aluminum frame is sturdy, the typing experience is excellent, and the hot-swappable switches make it very flexible. The RGB looks great too. It is not cheap, but the quality makes the price feel justified.\"\n"
                "}"
            ),
        },
        {
            "question": (
                "Given the following input schema:\n"
                "user_persona: 'A cautious reviewer who focuses on comfort, reliability, and practical use.'\n"
                "user_history: 'Usually writes balanced reviews, points out both strengths and weaknesses, and tends to rate based on real-world usefulness rather than hype.'\n"
                "product_details: 'Running shoes, moderate price, lightweight design, good grip, narrow toe box, decent cushioning.'\n"
                "sentiment_cue: 'balanced'\n"
                "Generate the predicted rating and review."
            ),
            "answer": (
                "{\n"
                "  \"predicted_rating\": 4,\n"
                "  \"predicted_review\": \"These shoes are lightweight and the grip is very good, which makes them reliable for regular runs. The cushioning is decent, though the toe box feels a bit narrow. Overall, they are practical and comfortable enough for daily use, with just a small fit issue.\"\n"
                "}"
            ),
        },
    ]

    # setting prompt structure for model
    prompt_structure = PromptTemplate.from_template("question: {question}\n{answer}")

    prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=prompt_structure,
        suffix="""
        question: (
            user_persona: {user_persona}
            user_history: {user_history}
            product_details: {product_details}
            sentiment_cue: {sentiment_cue})""",
        input_variables=[
            "user_persona",
            "user_persona",
            "product_details",
            "sentiment_cue"
            ],
    )

    return system_prompt, prompt_template


# makeshift session(aka long-term) memory for multiturn conversation
def session_store(prompt=None, model_response=None):
    """stores interaction data of user and recommender chat"""
    session = []
    try:
        if prompt and model_response is None:
            return f'`{session}`'
        session.append(str({
            'user preciousely asked': prompt,
            'you responded with': model_response
        }))
        return '\n'.join(session)
    except Exception as exc:
        raise Exception("could not store session") from exc


# main review prompt
def review_generation_prompt():
    """sets the structure of the prompt instructing the model"""

    # we need to extract the artifacts that sets up the reasoning prompt
    system_prompt, prompt_template = prompt_setup()

    return system_prompt + prompt_template


# recommendation engine prompt
def recommender_prompt():
    """prompt template for recommendation generation"""
    prompt_template = f"""
    You are a recommendation generation model.

    Given the input fields below, generate one personalized recommendation that best matches the user.

    Rules:
    - Be accurate and do not hallucinate missing facts.
    - Match tone to the sentiment cue.
    - Base the persona on the following below:
    {asyncio.wait_for(retrieve_user_persona(), timeout=15)}

    Note:
    Your previous interaction is as follows:
    {asyncio.wait_for(session_store(), timeout=15)}.
    if previous interaction is `[]` then it means it's a fresh session.

    """ + "question: {question}"

    prompt = PromptTemplate.from_template(
        template=prompt_template
    )
    return prompt


# evaluator prompts
# plan adherence prompt
def recommendation_plan():
    """generation plan on which the chat outputs would be evaluated on"""
    return f"""
        Rules:
        - Be accurate and do not hallucinate missing facts.
        - Match tone to the sentiment cue.
        - Base the persona on the following below:
        {asyncio.wait_for(retrieve_user_persona(), timeout=15)}
        """
