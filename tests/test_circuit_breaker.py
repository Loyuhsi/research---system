"""Tests for circuit_breaker.py — narrow-scope per-target circuit breaker."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from auto_research.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
)


class TestCircuitBreakerStates:
    def test_starts_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state == STATE_CLOSED
        assert cb.failure_count == 0

    def test_stays_closed_under_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == STATE_CLOSED
        assert cb.failure_count == 2

    def test_opens_at_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == STATE_OPEN

    def test_success_resets_to_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == STATE_CLOSED
        assert cb.failure_count == 0

    def test_open_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        assert cb.state == STATE_OPEN
        time.sleep(0.15)
        assert cb.state == STATE_HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == STATE_HALF_OPEN
        cb.record_success()
        assert cb.state == STATE_CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == STATE_HALF_OPEN
        cb.record_failure()
        assert cb.state == STATE_OPEN


class TestCircuitBreakerCall:
    def test_call_succeeds_in_closed(self):
        cb = CircuitBreaker("test")
        result = cb.call(lambda: 42)
        assert result == 42

    def test_call_raises_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(lambda: 42)
        assert "OPEN" in str(exc_info.value)
        assert exc_info.value.target == "test"

    def test_call_records_failure_on_exception(self):
        cb = CircuitBreaker("test", failure_threshold=2)

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert cb.failure_count == 1

    def test_call_records_success(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.call(lambda: "ok")
        assert cb.failure_count == 0
        assert cb.state == STATE_CLOSED

    def test_half_open_probe_succeeds(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.1)
        result = cb.call(lambda: "probe_ok")
        assert result == "probe_ok"
        assert cb.state == STATE_CLOSED

    def test_half_open_probe_fails(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.1)

        def fail():
            raise RuntimeError("still broken")

        with pytest.raises(RuntimeError, match="still broken"):
            cb.call(fail)
        assert cb.state == STATE_OPEN


class TestCircuitBreakerStatus:
    def test_status_closed(self):
        cb = CircuitBreaker("ollama")
        s = cb.status()
        assert s["target"] == "ollama"
        assert s["state"] == STATE_CLOSED

    def test_status_open(self):
        cb = CircuitBreaker("ollama", failure_threshold=1)
        cb.record_failure()
        s = cb.status()
        assert s["state"] == STATE_OPEN
        assert s["failure_count"] == 1


class TestCircuitBreakerReset:
    def test_manual_reset(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == STATE_OPEN
        cb.reset()
        assert cb.state == STATE_CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerIsolation:
    def test_separate_targets_independent(self):
        cb1 = CircuitBreaker("ollama", failure_threshold=1)
        cb2 = CircuitBreaker("telegram", failure_threshold=1)
        cb1.record_failure()
        assert cb1.state == STATE_OPEN
        assert cb2.state == STATE_CLOSED  # not affected
