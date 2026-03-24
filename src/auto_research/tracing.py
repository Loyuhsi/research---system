"""Local request tracing for Auto-Research.

Provides lightweight trace/span model for instrumenting key operations.
Data flows to telemetry JSONL via event bus (optional). Designed for
future OpenTelemetry / Langfuse export but has no external dependencies.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SpanKind(str, Enum):
    """Span types instrumented in v3.10."""

    ROOT_TASK = "root_task"
    LLM_CALL = "llm_call"
    QUALITY_GATE = "quality_gate"
    APPROVAL = "approval"
    TELEGRAM_API = "telegram_api"
    RETRIEVAL = "retrieval"
    DISCOVERY = "discovery"
    REFLECTION = "reflection"
    # Reserved for future instrumentation
    BREAKER_EVENT = "breaker_event"
    GUARD_EVENT = "guard_event"
    INDEX_OP = "index_op"
    FETCH_OP = "fetch_op"


@dataclass
class Span:
    """A single operation span within a trace."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error_message: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return round((self.end_time - self.start_time) * 1000, 2)

    def finish(self, status: str = "ok", error: Optional[str] = None) -> None:
        self.end_time = time.monotonic()
        self.status = status
        if error:
            self.error_message = error

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["duration_ms"] = self.duration_ms
        return d


@dataclass
class TraceContext:
    """A collection of spans sharing a trace_id."""

    trace_id: str
    spans: List[Span] = field(default_factory=list)

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        span = Span(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=time.monotonic(),
            attributes=attributes or {},
        )
        self.spans.append(span)
        return span

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_count": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
        }


# ---------------------------------------------------------------------------
# Context management (thread / async safe via contextvars)
# ---------------------------------------------------------------------------

_current_trace: ContextVar[Optional[TraceContext]] = ContextVar(
    "_current_trace", default=None,
)


def new_trace() -> TraceContext:
    """Create and activate a new trace context."""
    ctx = TraceContext(trace_id=uuid.uuid4().hex[:32])
    _current_trace.set(ctx)
    return ctx


def current_trace() -> Optional[TraceContext]:
    """Return the currently active trace, or None."""
    return _current_trace.get()


def clear_trace() -> None:
    """Deactivate the current trace context."""
    _current_trace.set(None)
