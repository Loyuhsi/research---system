"""Tests for event_handlers.py — production handler factories."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from auto_research.event_handlers import (
    on_any_event_telemetry,
    on_task_outcome_evo_log,
    on_task_outcome_memory_draft,
    wire_event_handlers,
)
from auto_research.events import EventBus, EventNames


class TestEvoLogHandler:
    def test_calls_evoskill_evo_log(self):
        evoskill = MagicMock()
        handler = on_task_outcome_evo_log(evoskill)
        handler({"session_id": "s1", "status": "success", "summary": "done"})
        evoskill.evo_log.assert_called_once()

    def test_exception_does_not_propagate(self):
        evoskill = MagicMock()
        evoskill.evo_log.side_effect = RuntimeError("db error")
        handler = on_task_outcome_evo_log(evoskill)
        # Should not raise
        handler({"session_id": "s1", "status": "success", "summary": "done"})

    def test_has_name(self):
        handler = on_task_outcome_evo_log(MagicMock())
        assert handler.__name__ == "on_task_outcome_evo_log"


class TestMemoryDraftHandler:
    def test_calls_memory_extract_on_success(self):
        sm = MagicMock()
        handler = on_task_outcome_memory_draft(sm)
        handler({"session_id": "s1", "status": "success", "summary": "ok"})
        sm.memory_extract.assert_called_once()

    def test_skips_on_failure(self):
        sm = MagicMock()
        handler = on_task_outcome_memory_draft(sm)
        handler({"session_id": "s1", "status": "failed", "summary": "err"})
        sm.memory_extract.assert_not_called()

    def test_exception_does_not_propagate(self):
        sm = MagicMock()
        sm.memory_extract.side_effect = RuntimeError("oops")
        handler = on_task_outcome_memory_draft(sm)
        handler({"session_id": "s1", "status": "success", "summary": "ok"})


class TestTelemetryHandler:
    def test_writes_jsonl(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        handler = on_any_event_telemetry(path)
        handler({"task_id": "t1", "status": "success"})
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "t1"
        assert "ts" in data

    def test_appends_multiple(self, tmp_path):
        path = tmp_path / "telemetry.jsonl"
        handler = on_any_event_telemetry(path)
        handler({"n": 1})
        handler({"n": 2})
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2


class TestWireEventHandlers:
    def test_wires_all_handlers(self):
        bus = EventBus()
        wire_event_handlers(
            event_bus=bus,
            evoskill=MagicMock(),
            skill_memory=MagicMock(),
            telemetry_path=Path("/tmp/test-telemetry.jsonl"),
        )
        subs = bus.list_subscribers()
        # TASK_OUTCOME should have 3 handlers (evo, memory, telemetry)
        assert subs.get(EventNames.TASK_OUTCOME, 0) >= 3
