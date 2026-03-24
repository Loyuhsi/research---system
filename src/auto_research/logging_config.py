"""Structured JSON logging for Auto-Research."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "ts": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "line": record.lineno,
            },
            ensure_ascii=False,
        )


def setup_logging(json_mode: bool | None = None) -> None:
    """Configure root logger for auto_research namespace.

    Args:
        json_mode: Force JSON output. If *None*, reads AUTO_RESEARCH_LOG_JSON env var.
    """
    if json_mode is None:
        json_mode = os.environ.get("AUTO_RESEARCH_LOG_JSON", "0") == "1"

    level = logging.DEBUG if os.environ.get("AUTO_RESEARCH_DEBUG") == "1" else logging.INFO
    root = logging.getLogger("auto_research")

    if root.handlers:
        return

    handler = logging.StreamHandler()
    handler.setLevel(level)

    if json_mode:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

    root.setLevel(level)
    root.addHandler(handler)
