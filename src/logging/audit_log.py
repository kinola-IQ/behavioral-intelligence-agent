"""Structured audit logging for agent actions."""

import logging

logger = logging.getLogger("audit")


def log_event(event: str, **fields: object) -> None:
    logger.info("%s | %s", event, fields)
