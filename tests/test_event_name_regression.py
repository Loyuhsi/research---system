"""Regression test: orchestrator must publish EventNames.TASK_OUTCOME so handlers fire.

Bug (V3.6): orchestrator published f"task.{status}" but handlers subscribed to
EventNames.TASK_OUTCOME ("task_outcome") → all event handlers were silently dead
in production.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from auto_research.events import EventBus, EventNames
from auto_research.event_handlers import (
    on_task_outcome_evo_log,
    on_task_outcome_memory_draft,
    on_any_event_telemetry,
    wire_event_handlers,
)
from conftest import create_test_orchestrator, FakeHttpClient


class TestEventNameRegression:
    """Verify orchestrator outcome events reach wired handlers."""

    def test_safe_record_outcome_publishes_task_outcome(self, tmp_repo_config):
        """_safe_record_outcome must publish EventNames.TASK_OUTCOME, not f'task.{status}'."""
        orch = create_test_orchestrator(tmp_repo_config)
        received = []
        orch.event_bus.subscribe(EventNames.TASK_OUTCOME, lambda p: received.append(p))

        orch._safe_record_outcome(
            task_id="test-001",
            action="research_session",
            status="success",
            summary="unit test",
            session_id="sess-001",
        )

        assert len(received) == 1
        assert received[0]["status"] == "success"
        assert received[0]["task_id"] == "test-001"
        assert received[0]["session_id"] == "sess-001"

    def test_old_event_name_no_longer_used(self, tmp_repo_config):
        """Ensure f'task.success' / f'task.failed' are NOT published."""
        orch = create_test_orchestrator(tmp_repo_config)
        old_received = []
        orch.event_bus.subscribe("task.success", lambda p: old_received.append(p))
        orch.event_bus.subscribe("task.failed", lambda p: old_received.append(p))

        orch._safe_record_outcome("t1", "act", "success", "s", session_id="s1")
        orch._safe_record_outcome("t2", "act", "failed", "f", session_id="s2")

        assert old_received == [], "Old event names must not be used"

    def test_evo_log_handler_fires_on_outcome(self, tmp_repo_config):
        """EvoLog handler receives the event when wired to EventBus."""
        orch = create_test_orchestrator(tmp_repo_config)
        evoskill = orch.registry.resolve("service.evoskill")
        evoskill.evo_log = MagicMock()

        handler = on_task_outcome_evo_log(evoskill)
        orch.event_bus.subscribe(EventNames.TASK_OUTCOME, handler)

        orch._safe_record_outcome("t1", "research", "success", "done", session_id="s1")

        evoskill.evo_log.assert_called_once()
        call_kwargs = evoskill.evo_log.call_args
        assert call_kwargs[1]["status"] == "success" or call_kwargs[0][2] == "success" or \
               any(v == "success" for v in call_kwargs[1].values())

    def test_memory_draft_handler_fires_on_success(self, tmp_repo_config):
        """MemoryDraft handler creates a draft when status=success."""
        orch = create_test_orchestrator(tmp_repo_config)
        skill_memory = orch.registry.resolve("service.skill_memory")
        skill_memory.memory_extract = MagicMock(return_value={"draft_path": "/tmp/draft.json"})

        handler = on_task_outcome_memory_draft(skill_memory)
        orch.event_bus.subscribe(EventNames.TASK_OUTCOME, handler)

        orch._safe_record_outcome("t1", "research", "success", "ok", session_id="s1")

        skill_memory.memory_extract.assert_called_once()

    def test_memory_draft_handler_skips_on_failure(self, tmp_repo_config):
        """MemoryDraft handler must NOT fire when status != success."""
        orch = create_test_orchestrator(tmp_repo_config)
        skill_memory = orch.registry.resolve("service.skill_memory")
        skill_memory.memory_extract = MagicMock()

        handler = on_task_outcome_memory_draft(skill_memory)
        orch.event_bus.subscribe(EventNames.TASK_OUTCOME, handler)

        orch._safe_record_outcome("t1", "research", "failed", "err", session_id="s1")

        skill_memory.memory_extract.assert_not_called()

    def test_telemetry_handler_fires_on_outcome(self, tmp_repo_config, tmp_path):
        """Telemetry handler appends to JSONL on TASK_OUTCOME."""
        orch = create_test_orchestrator(tmp_repo_config)
        telemetry_path = tmp_path / "telemetry.jsonl"

        handler = on_any_event_telemetry(telemetry_path)
        orch.event_bus.subscribe(EventNames.TASK_OUTCOME, handler)

        orch._safe_record_outcome("t1", "research", "success", "done", session_id="s1")

        assert telemetry_path.exists()
        import json
        lines = telemetry_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["task_id"] == "t1"
        assert "ts" in entry

    def test_wire_event_handlers_integration(self, tmp_repo_config, tmp_path):
        """Full integration: wire_event_handlers + _safe_record_outcome → all 3 handlers fire."""
        orch = create_test_orchestrator(tmp_repo_config)
        evoskill = orch.registry.resolve("service.evoskill")
        skill_memory = orch.registry.resolve("service.skill_memory")
        telemetry_path = tmp_path / "telemetry.jsonl"

        evoskill.evo_log = MagicMock()
        skill_memory.memory_extract = MagicMock(return_value={"draft_path": "/tmp/d.json"})

        wire_event_handlers(
            event_bus=orch.event_bus,
            evoskill=evoskill,
            skill_memory=skill_memory,
            telemetry_path=telemetry_path,
        )

        orch._safe_record_outcome("t1", "research", "success", "done", session_id="s1")

        # All 3 handlers should have fired
        evoskill.evo_log.assert_called_once()
        skill_memory.memory_extract.assert_called_once()
        assert telemetry_path.exists()
