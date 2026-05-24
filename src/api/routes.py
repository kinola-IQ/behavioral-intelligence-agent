"""HTTP routes."""
import json
from fastapi import APIRouter, BackgroundTasks
from tenacity import retry, wait_random_exponential, stop_after_attempt
from .schemas import UserRequest, AgentResponse, ChatResponse
from ..core.prompt_engine import session_store
from ..core.utils import startup_status
from ..logging.audit_log import log_event
# generators
from ..generation.review_generator import generate_review
from ..generation.recommendation_generator import recommendation_llm
from ..evaluation.evaluator import evaluation_pipeline

router = APIRouter(prefix="/api/v1", tags=["v1"])


@router.get("/health")
def health() -> dict[str, str]:
    """informs us of the status of vital resources"""
    return {"status": startup_status()}


@router.post('/generate_review', response_model=AgentResponse)
@retry(wait=wait_random_exponential(10,60), stop=stop_after_attempt(2))
def review_generator(request: UserRequest):
    """parses request from user to orchestrator"""
    try:
        response = generate_review(request.prompt)
        return AgentResponse(
            predicted_rating=response['predicted_rating'],
            predicted_review=response['predicted_review'],
            agent_status='success'
        )
    except Exception as exc:
        log_event("review_generator_failed", error=str(exc))
        return AgentResponse(
            predicted_rating=0,
            predicted_review='',
            agent_status='Failed'
        )


@router.post('/generate_recommendation', response_model=ChatResponse)
def recommendation_chat(
        request: UserRequest, background_tasks: BackgroundTasks):
    """parses request to the recommendation chat llm"""
    try:
        response = recommendation_llm(request.prompt)

        # evaluate response from model
        eval_result = f"{evaluation_pipeline(request.prompt, response)}"

        # store interaction into memory
        async def store_interaction():
            session_store(request.prompt, response)

        # background tasks
        background_tasks.add_task(store_interaction)

        return ChatResponse(
            response_text=response,
            eval_result={"payload": json.loads(eval_result)}
        )
    except Exception as exc:
        return ChatResponse(
            response_text=f'{exc}: could not generate a response',
            eval_result={}
        )
