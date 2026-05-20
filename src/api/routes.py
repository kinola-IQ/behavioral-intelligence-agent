"""HTTP routes."""

from fastapi import APIRouter, BackgroundTasks
from .schemas import UserRequest, AgentResponse, ChatResponse
from ..core.prompt_engine import session_store
from ..core.utils import startup_status
# generators
from ..generation.review_generator import review_generator_agent
from ..generation.recommendation_generator import recommendation_llm


router = APIRouter(prefix="/api/v1", tags=["v1"])


@router.get("/health")
def health() -> dict[str, str]:
    """informs us of the status of vital resources"""
    return {"status": startup_status()}


@router.post('/generate_review', response_model=AgentResponse)
def review_generator(request: UserRequest):
    """parses request from user to orchestrator"""
    try:
        agent = review_generator_agent(request.prompt)
        query_agent = agent.invoke(
            {"messages": [{"role": "user", "content": request.prompt}]})
        response = query_agent['messages'][-1]['content']
        return AgentResponse(
            predicted_rating=response['predicted_rating'],
            predicted_review=response['predicted_review'],
            agent_status='success'
        )
    except Exception:
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

        # store interaction into memory
        async def store_interaction():
            session_store(request.prompt, response)
        background_tasks.add_task(store_interaction)

        return ChatResponse(
            response_text=response
        )
    except Exception as exc:
        return ChatResponse(
            response_text=f'{exc}: could not generate a response'
        )
