"""Tests for LLM provider breaker + GPU guard integration (P2)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from auto_research.circuit_breaker import CircuitOpenError, STATE_OPEN, STATE_CLOSED
from auto_research.resource_guard import GuardTimeoutError
from auto_research.services.llm_provider import LlmProviderService
from conftest import make_temp_repo, FakeHttpClient


def _make_llm_service(tmpdir: str) -> LlmProviderService:
    repo = make_temp_repo(tmpdir)
    from auto_research.runtime import load_config
    config = load_config(repo_root=repo, environ={})
    return LlmProviderService(config)


class TestCallWithBreakerAndGuard:
    def test_successful_call_acquires_and_releases_guard(self, tmp_path):
        svc = _make_llm_service(str(tmp_path))
        result = svc.call_with_breaker("ollama", lambda: "ok")
        assert result == "ok"
        # Guard should be released after call
        assert svc.guard.status()["state"] == "unlocked"

    def test_failed_call_releases_guard_and_records_failure(self, tmp_path):
        svc = _make_llm_service(str(tmp_path))

        def fail():
            raise RuntimeError("LLM timeout")

        with pytest.raises(RuntimeError, match="LLM timeout"):
            svc.call_with_breaker("ollama", fail)

        # Guard released, breaker recorded failure
        assert svc.guard.status()["state"] == "unlocked"
        assert svc.get_breaker("ollama").failure_count == 1

    def test_breaker_opens_after_threshold(self, tmp_path):
        svc = _make_llm_service(str(tmp_path))

        def fail():
            raise RuntimeError("err")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                svc.call_with_breaker("ollama", fail)

        assert svc.get_breaker("ollama").state == STATE_OPEN

    def test_open_breaker_skips_guard(self, tmp_path):
        svc = _make_llm_service(str(tmp_path))

        def fail():
            raise RuntimeError("err")

        # Trip breaker
        for _ in range(3):
            with pytest.raises(RuntimeError):
                svc.call_with_breaker("ollama", fail)

        # Now call should fail fast without touching guard
        with pytest.raises(CircuitOpenError):
            svc.call_with_breaker("ollama", lambda: "should not run")

        # Guard should not have been acquired
        assert svc.guard.status()["state"] == "unlocked"

    def test_guard_busy_raises_guard_timeout_error(self, tmp_path):
        svc = _make_llm_service(str(tmp_path))
        # Manually hold the guard
        svc.guard.acquire()
        try:
            with pytest.raises(GuardTimeoutError):
                svc.call_with_breaker("ollama", lambda: "should not run")
            # Breaker should NOT have recorded a failure
            assert svc.get_breaker("ollama").failure_count == 0
        finally:
            svc.guard.release()

    def test_guard_timeout_does_not_pollute_breaker(self, tmp_path):
        svc = _make_llm_service(str(tmp_path))
        # Record 2 failures
        def fail():
            raise RuntimeError("err")
        for _ in range(2):
            with pytest.raises(RuntimeError):
                svc.call_with_breaker("ollama", fail)

        assert svc.get_breaker("ollama").failure_count == 2

        # Hold guard → GuardTimeoutError should NOT push to 3 (threshold)
        svc.guard.acquire()
        try:
            with pytest.raises(GuardTimeoutError):
                svc.call_with_breaker("ollama", lambda: "x")
            assert svc.get_breaker("ollama").failure_count == 2  # unchanged
            assert svc.get_breaker("ollama").state == STATE_CLOSED  # still closed
        finally:
            svc.guard.release()

    def test_separate_providers_have_separate_breakers(self, tmp_path):
        svc = _make_llm_service(str(tmp_path))

        def fail():
            raise RuntimeError("err")

        # Trip ollama breaker
        for _ in range(3):
            with pytest.raises(RuntimeError):
                svc.call_with_breaker("ollama", fail)

        assert svc.get_breaker("ollama").state == STATE_OPEN
        # lmstudio should be unaffected
        result = svc.call_with_breaker("lmstudio", lambda: "ok")
        assert result == "ok"


class TestGuardTimeoutError:
    def test_error_has_lock_path(self, tmp_path):
        err = GuardTimeoutError(Path("/tmp/test.lock"))
        assert err.lock_path == Path("/tmp/test.lock")
        assert "GPU execution guard busy" in str(err)

    def test_error_is_runtime_error(self, tmp_path):
        err = GuardTimeoutError(Path("/tmp/test.lock"))
        assert isinstance(err, RuntimeError)


class TestDoctorIncludesGuardStatus:
    def test_doctor_has_gpu_guard(self, tmp_path):
        from conftest import create_test_orchestrator
        from auto_research.runtime import load_config
        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={})
        orch = create_test_orchestrator(config)
        result = orch.doctor()
        assert "gpu_guard" in result
        assert result["gpu_guard"]["state"] == "unlocked"
