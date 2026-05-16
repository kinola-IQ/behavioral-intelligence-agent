"""Short- and long-term memory for agent context."""

from langchain_core.tools import tool

from .utils import MEMORY
from ..api.schemas import UserPersona
from .persona_builder import _persona_payload
APPLICATION_CONTEXT = "behavioral_intelligence"


def _namespace(user_id: str) -> tuple[str, str]:
    return (user_id, APPLICATION_CONTEXT)


@tool
def context_store(persona: UserPersona, user_id: str = "my-user") -> str:
    """Commit the user persona to memory."""
    if MEMORY is None:
        return "failed to store memory: store not initialized"

    try:
        profile = _persona_payload(persona)
        MEMORY.put(
            _namespace(user_id),
            "persona",
            {
                "kind": "persona",
                "persona": profile,
                "rules": profile.get("category_patterns", []),
            },
        )
        return "completed"
    except Exception:
        return "failed to store memory"


async def retrieve_user_persona(user_id: str = "my-user") -> str | None:
    """Retrieve the user persona from memory."""
    if MEMORY is None:
        return None

    profile = MEMORY.get(_namespace(user_id), "persona").value
    if profile is None:
        return None

    persona = [f'{item.key}: {item.value}' for item in profile.items()]

    return "\n".join(persona)
