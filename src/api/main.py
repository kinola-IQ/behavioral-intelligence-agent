"""FastAPI application entrypoint."""

from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(title="Behavioural Intelligence Agent API")
app.include_router(router)
