"""Tests for the tracing module."""

import time

from auto_research.tracing import (
    SpanKind,
    Span,
    TraceContext,
    new_trace,
    current_trace,
    clear_trace,
)


class TestSpanKind:
    def test_active_span_kinds(self):
        assert SpanKind.ROOT_TASK.value == "root_task"
        assert SpanKind.LLM_CALL.value == "llm_call"
        assert SpanKind.QUALITY_GATE.value == "quality_gate"
        assert SpanKind.APPROVAL.value == "approval"
        assert SpanKind.TELEGRAM_API.value == "telegram_api"

    def test_reserved_span_kinds(self):
        assert SpanKind.BREAKER_EVENT.value == "breaker_event"
        assert SpanKind.GUARD_EVENT.value == "guard_event"
        assert SpanKind.INDEX_OP.value == "index_op"
        assert SpanKind.FETCH_OP.value == "fetch_op"


class TestSpan:
    def test_span_creation_and_finish(self):
        span = Span(
            trace_id="abc123",
            span_id="span001",
            parent_span_id=None,
            name="test.span",
            kind=SpanKind.ROOT_TASK,
            start_time=time.monotonic(),
        )
        assert span.duration_ms is None
        assert span.status == "ok"
        span.finish()
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_span_finish_with_error(self):
        span = Span(
            trace_id="abc",
            span_id="s1",
            parent_span_id=None,
            name="err",
            kind=SpanKind.LLM_CALL,
            start_time=time.monotonic(),
        )
        span.finish(status="error", error="timeout")
        assert span.status == "error"
        assert span.error_message == "timeout"

    def test_span_to_dict(self):
        span = Span(
            trace_id="t1",
            span_id="s1",
            parent_span_id="p1",
            name="test",
            kind=SpanKind.QUALITY_GATE,
            start_time=100.0,
            attributes={"score": 0.8},
        )
        span.end_time = 100.5
        d = span.to_dict()
        assert d["kind"] == "quality_gate"
        assert d["duration_ms"] == 500.0
        assert d["attributes"]["score"] == 0.8
        assert d["parent_span_id"] == "p1"


class TestTraceContext:
    def test_start_span(self):
        ctx = TraceContext(trace_id="trace1")
        span = ctx.start_span("op1", SpanKind.LLM_CALL)
        assert span.trace_id == "trace1"
        assert len(span.span_id) == 16
        assert len(ctx.spans) == 1

    def test_multiple_spans(self):
        ctx = TraceContext(trace_id="trace2")
        s1 = ctx.start_span("root", SpanKind.ROOT_TASK)
        s2 = ctx.start_span("child", SpanKind.LLM_CALL, parent_span_id=s1.span_id)
        assert s2.parent_span_id == s1.span_id
        assert len(ctx.spans) == 2

    def test_to_dict(self):
        ctx = TraceContext(trace_id="trace3")
        ctx.start_span("op", SpanKind.APPROVAL, attributes={"id": "m1"})
        d = ctx.to_dict()
        assert d["trace_id"] == "trace3"
        assert d["span_count"] == 1
        assert len(d["spans"]) == 1


class TestContextManagement:
    def test_new_trace_and_current(self):
        clear_trace()
        assert current_trace() is None
        trace = new_trace()
        assert current_trace() is trace
        assert len(trace.trace_id) == 32
        clear_trace()
        assert current_trace() is None

    def test_independent_traces(self):
        clear_trace()
        t1 = new_trace()
        t1_id = t1.trace_id
        t2 = new_trace()
        assert t2.trace_id != t1_id
        assert current_trace() is t2
        clear_trace()
