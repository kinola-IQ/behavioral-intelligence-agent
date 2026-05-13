"""HTTP routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["v1"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
