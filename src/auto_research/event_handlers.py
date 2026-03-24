"""Production event handlers for Auto-Research.

Wired into the EventBus during bootstrap to automate side-effects:
- EvoLog: logs task outcomes to evo-log
- MemoryDraft: extracts memory drafts on successful tasks
- Telemetry: appends all events to an append-only JSONL file
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict

from .events import EventBus, EventNames
from .services.evoskill import EvoSkillService
from .services.skill_memory import SkillMemoryService

logger = logging.getLogger(__name__)

EventHandler = Callable[[Dict[str, Any]], None]


def on_task_outcome_evo_log(evoskill: EvoSkillService) -> EventHandler:
    """Return a handler that logs task outcomes to evo-log."""

    def handler(payload: Dict[str, Any]) -> None:
        task_id = str(payload.get("task_id", payload.get("session_id", "unknown")))
        status = str(payload.get("status", "unknown"))
        summary = str(payload.get("summary", ""))
        try:
            evoskill.evo_log(task_id=task_id, status=status, summary=summary)
        except Exception:
            logger.exception("EvoLog handler failed for task %s", task_id)

    handler.__name__ = "on_task_outcome_evo_log"
    return handler


def on_task_outcome_memory_draft(skill_memory: SkillMemoryService) -> EventHandler:
    """Return a handler that creates memory drafts on successful tasks."""

    def handler(payload: Dict[str, Any]) -> None:
        if payload.get("status") != "success":
            return
        session_id = str(payload.get("session_id", "unknown"))
        try:
            skill_memory.memory_extract(
                session_id=session_id,
                task_type=str(payload.get("task_type", "research_session")),
                status="success",
                summary_override=payload.get("summary"),
            )
        except Exception:
            logger.exception("MemoryDraft handler failed for session %s", session_id)

    handler.__name__ = "on_task_outcome_memory_draft"
    return handler


def on_any_event_telemetry(telemetry_path: Path) -> EventHandler:
    """Return a handler that appends events to a JSONL telemetry log."""
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)

    def handler(payload: Dict[str, Any]) -> None:
        entry = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            **payload,
        }
        try:
            with open(telemetry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.exception("Telemetry handler failed")

    handler.__name__ = "on_any_event_telemetry"
    return handler


def wire_event_handlers(
    event_bus: EventBus,
    evoskill: EvoSkillService,
    skill_memory: SkillMemoryService,
    telemetry_path: Path,
) -> None:
    """Register all production event handlers on the given EventBus."""
    telemetry_handler = on_any_event_telemetry(telemetry_path)

    event_bus.subscribe(EventNames.TASK_OUTCOME, on_task_outcome_evo_log(evoskill))
    event_bus.subscribe(EventNames.TASK_OUTCOME, on_task_outcome_memory_draft(skill_memory))

    # Telemetry on all known events
    for event_name in (
        EventNames.TASK_OUTCOME,
        EventNames.MEMORY_DRAFT_CREATED,
        EventNames.SKILL_PROMOTED,
        EventNames.RESEARCH_COMPLETED,
        EventNames.TRACE_COMPLETED,
    ):
        event_bus.subscribe(event_name, telemetry_handler)
