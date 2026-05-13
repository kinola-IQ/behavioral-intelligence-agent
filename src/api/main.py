"""FastAPI application entrypoint."""

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
)


from src.api.routes import router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize resources on startup."""
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
    uvicorn.run("src.api.main:server", host="0.0.0.0", port=8000, reload=True)
