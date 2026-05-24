"""FastAPI application entrypoint."""
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
)


from .routes import router
from ..core.utils import startup_resources
from ..config.settings import get_settings
from ..logging.audit_log import configure_audit_logging, log_event

# config values
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize resources on startup."""
    configure_audit_logging(settings.log_level)
    log_event("api_startup_begin", env=settings.app_env)
    await startup_resources()
    log_event("api_startup_complete")
    yield


@retry(wait=wait_random_exponential(10, 40), stop=stop_after_attempt(5))
def create_app():
    """Create the FastAPI application and register routes."""
    app = FastAPI(
        title="Behavioural Intelligence Agent API",
        lifespan=lifespan
    )
    app.include_router(router)
    return app


app = create_app()


