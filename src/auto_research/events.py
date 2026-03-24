"""Event Bus for Auto-Research.

Provides synchronous pub/sub for decoupling side-effects and notifications
from the main orchestration flow. Handlers cannot break the main flow.
"""

from typing import Any, Callable, Dict, List, Optional
import logging

from .log import setup_logging

logger = setup_logging(__name__)

EventHandler = Callable[[Dict[str, Any]], None]


class EventNames:
    """Well-known event names used across the system."""

    TASK_OUTCOME = "task_outcome"
    MEMORY_DRAFT_CREATED = "memory_draft_created"
    SKILL_PROMOTED = "skill_promoted"
    RESEARCH_COMPLETED = "research_completed"
    TRACE_COMPLETED = "trace_completed"


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[EventHandler]] = {}

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Subscribe a handler to a specific event name."""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)

    def publish(self, event_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish an event synchronously to all registered handlers.
        Handlers are executed sequentially. If a handler throws an exception,
        it is caught, logged, and execution continues to the next handler.
        
        Returns a summary of execution (successes vs failures).
        """
        success_count = 0
        failure_count = 0
        errors = []

        handlers = self._subscribers.get(event_name, [])
        for handler in handlers:
            try:
                handler(payload)
                success_count += 1
            except Exception as e:
                handler_name = getattr(handler, '__name__', str(handler))
                logger.error("Event handler '%s' failed for event '%s': %s", handler_name, event_name, e, exc_info=True)
                failure_count += 1
                errors.append(str(e))

        return {
            "event": event_name,
            "handlers_executed": success_count + failure_count,
            "success": success_count,
            "failures": failure_count,
            "errors": errors
        }

    def list_subscribers(self, event_name: Optional[str] = None) -> Dict[str, int]:
        """Return a mapping of event_name to the number of registered handlers."""
        if event_name:
            return {event_name: len(self._subscribers.get(event_name, []))}
        return {name: len(handlers) for name, handlers in self._subscribers.items()}
