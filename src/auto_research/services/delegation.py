"""Generic outcome-recording delegation for orchestrator operations."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


def delegate_with_outcome(
    *,
    fn: Callable[..., Any],
    record_fn: Callable[..., None],
    task_id: str,
    action: str,
    success_summary: str,
    fail_summary: str,
    session_id: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Any:
    """Execute *fn* and record the outcome via *record_fn*.

    On success, records status="success" with *success_summary*.
    On failure, records status="failed" with *fail_summary*, then re-raises.
    """
    try:
        result = fn()
    except Exception as exc:
        record_fn(
            task_id=task_id,
            action=action,
            status="failed",
            summary=f"{fail_summary}: {exc}",
            session_id=session_id,
            metadata=dict(metadata) if metadata else None,
        )
        raise
    # If result contains a session_id, use it for the success record
    result_session = session_id
    result_task_id = task_id
    if isinstance(result, dict) and "session_id" in result:
        result_session = str(result["session_id"])
        result_task_id = f"{action}-{result_session}"
    record_fn(
        task_id=result_task_id,
        action=action,
        status="success",
        summary=success_summary,
        session_id=result_session,
        metadata=dict(metadata) if metadata else None,
    )
    return result
