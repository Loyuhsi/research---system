from __future__ import annotations

import logging
import os


def setup_logging(name: str = "auto_research") -> logging.Logger:
    """Configure and return the root logger for auto-research.

    Respects AUTO_RESEARCH_DEBUG env var: set to '1' for DEBUG level.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = logging.DEBUG if os.environ.get("AUTO_RESEARCH_DEBUG") == "1" else logging.INFO
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)

    return logger
