"""Structured audit logging for agent actions."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("uvicorn")
_configured = False


def configure_audit_logging(level: str = "INFO") -> None:
    """Send audit events to stdout so they appear in the terminal."""
    global _configured
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | audit | %(levelname)s | %(message)s")
    )
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True


def log_event(event: str, **fields: object) -> None:
    if not _configured:
        configure_audit_logging()
    if fields:
        logger.info("%s | %s", event, fields)
    else:
        logger.info("%s", event)
