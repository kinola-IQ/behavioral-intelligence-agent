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
from ..config.settings import Settings

# config values
settings = Settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize resources on startup."""
    await startup_resources()
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


server = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:server", host=settings.api_host,
        port=settings.api_port, reload=True)
