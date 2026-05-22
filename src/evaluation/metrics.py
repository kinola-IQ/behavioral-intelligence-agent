"""Evaluation metrics (retrieval, generation quality)."""

from openevals.llm import create_llm_as_judge
from openevals.prompts import PLAN_ADHERENCE_PROMPT, RAG_HELPFULNESS_PROMPT

# resource
from ..core import utils

# plans
from ..core.prompt_engine import recommendation_plan


# judge
def _llm_evaluator(prompt: PLAN_ADHERENCE_PROMPT | RAG_HELPFULNESS_PROMPT,
                   feedback_key: str):
    return create_llm_as_judge(
        prompt=prompt,
        feedback_key=feedback_key,
        model=utils.LLM,
    )


# generation quality
# helpfullness
async def helpfulness(input: str, output: str):
    evaluator = _llm_evaluator(
        RAG_HELPFULNESS_PROMPT, feedback_key='helpfullness')
    eval_result = evaluator(
        inputs=input,
        outputs=output
        )
    return eval_result


# plan adherance
async def plan_adherence(input: str, output: str):
    evaluator = _llm_evaluator(
        PLAN_ADHERENCE_PROMPT, feedback_key='plan adherence')
    eval_result = evaluator(
        inputs=input,
        outputs=output,
        plan= await recommendation_plan()
        )
    return eval_result
