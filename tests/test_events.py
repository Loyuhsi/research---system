"""Tests for events.py — EventBus and EventNames."""

from __future__ import annotations

import pytest

from auto_research.events import EventBus, EventNames


class TestEventNames:
    def test_task_outcome_value(self):
        assert EventNames.TASK_OUTCOME == "task_outcome"

    def test_all_names_are_strings(self):
        for attr in ("TASK_OUTCOME", "MEMORY_DRAFT_CREATED", "SKILL_PROMOTED", "RESEARCH_COMPLETED"):
            assert isinstance(getattr(EventNames, attr), str)


class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda p: received.append(p))
        bus.publish("test", {"key": "val"})
        assert received == [{"key": "val"}]

    def test_multiple_handlers(self):
        bus = EventBus()
        results = []
        bus.subscribe("e", lambda p: results.append("a"))
        bus.subscribe("e", lambda p: results.append("b"))
        bus.publish("e", {})
        assert results == ["a", "b"]

    def test_publish_returns_summary(self):
        bus = EventBus()
        bus.subscribe("e", lambda p: None)
        result = bus.publish("e", {})
        assert result["success"] == 1
        assert result["failures"] == 0

    def test_handler_exception_does_not_break_others(self):
        bus = EventBus()
        results = []
        bus.subscribe("e", lambda p: (_ for _ in ()).throw(ValueError("boom")))
        bus.subscribe("e", lambda p: results.append("ok"))
        result = bus.publish("e", {})
        assert result["failures"] == 1
        assert result["success"] == 1
        assert results == ["ok"]

    def test_no_handlers_for_event(self):
        bus = EventBus()
        result = bus.publish("unknown", {})
        assert result["handlers_executed"] == 0

    def test_list_subscribers_empty(self):
        bus = EventBus()
        assert bus.list_subscribers() == {}

    def test_list_subscribers_counts(self):
        bus = EventBus()
        bus.subscribe("a", lambda p: None)
        bus.subscribe("a", lambda p: None)
        bus.subscribe("b", lambda p: None)
        subs = bus.list_subscribers()
        assert subs["a"] == 2
        assert subs["b"] == 1

    def test_list_subscribers_by_name(self):
        bus = EventBus()
        bus.subscribe("a", lambda p: None)
        assert bus.list_subscribers("a") == {"a": 1}
        assert bus.list_subscribers("missing") == {"missing": 0}
