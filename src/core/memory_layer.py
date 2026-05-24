"""Short-term memory for agent context."""

from langchain_core.tools import tool

from ..core import utils
from ..api.schemas import UserPersona
from ..logging.audit_log import log_event
from .persona_builder import _persona_payload

APPLICATION_CONTEXT = "behavioral_intelligence"


def _namespace(user_id: str) -> tuple[str, str]:
    return (user_id, APPLICATION_CONTEXT)


def _format_persona(profile: dict) -> str:
    lines = [f"{key}: {value}" for key, value in profile.items()]
    return "\n".join(lines)


@tool
def context_store(persona: UserPersona, user_id: str = "my-user") -> str:
    """Persist the current user persona for later recommendation steps.

    Implements the Store Context stage of the recommendation architecture.
    Writes the structured persona (and derived category rules) into the
    in-memory store so later turns, retrieval, and the recommendation engine
    can access consistent user state instead of treating each query in isolation.

    Args:
        persona: Structured user profile produced by ``model_user`` (or
            equivalent fields matching the ``UserPersona`` schema).
        user_id: Stable identifier for the user or session namespace.
            Defaults to ``"my-user"``.

    Returns:
        ``"completed"`` when the persona was stored successfully;
        ``"failed to store memory: store not initialized"`` if startup has not
        run; ``"failed to store memory"`` on unexpected persistence errors.
    """
    if utils.MEMORY is None:
        log_event("context_store_failed", reason="memory not initialized")
        return "failed to store memory: store not initialized"

    try:
        profile = _persona_payload(persona)
        utils.MEMORY.put(
            _namespace(user_id),
            "persona",
            {
                "kind": "persona",
                "persona": profile,
                "rules": profile.get("category_patterns", []),
            },
        )
        log_event("context_store_success", user_id=user_id)
        return "completed"
    except Exception:
        log_event("context_store_failed", user_id=user_id)
        return "failed to store memory"


def retrieve_user_persona(user_id: str = "my-user") -> str | None:
    """Load a previously stored persona as formatted text for prompt assembly.

    Args:
        user_id: Namespace key used when the persona was stored via
            ``context_store``. Defaults to ``"my-user"``.

    Returns:
        A newline-separated ``key: value`` rendering of the stored persona,
        or ``None`` if the store is unavailable or no persona exists for the
        given ``user_id``.
    """
    if utils.MEMORY is None:
        return None

    item = utils.MEMORY.get(_namespace(user_id), "persona")
    if item is None or item.value is None:
        return None

    stored = item.value
    profile = stored.get("persona", stored) if isinstance(stored, dict) else stored
    if not isinstance(profile, dict):
        return str(profile)

    return _format_persona(profile)
