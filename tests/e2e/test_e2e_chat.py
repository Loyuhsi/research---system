"""E2E: Real Ollama chat tests."""

from __future__ import annotations

import time

import pytest

from .conftest import skip_no_ollama


@pytest.mark.e2e
@skip_no_ollama
class TestE2EChat:
    def test_real_chat_returns_content(self, e2e_orchestrator):
        """Real Ollama chat produces non-empty reply."""
        t0 = time.monotonic()
        result = e2e_orchestrator.chat(
            session_key="e2e-chat-1",
            text="What is 2+2? Reply with only the number.",
            mode="research_only",
            frontend="cli",
        )
        elapsed = time.monotonic() - t0
        assert result["reply"]
        assert len(result["reply"]) > 0
        assert result["provider"] == "ollama"
        print(f"E2E chat: {elapsed:.2f}s, reply length: {len(result['reply'])}")

    def test_chat_breaker_stays_closed(self, e2e_orchestrator):
        """After a successful chat, breaker should remain closed."""
        e2e_orchestrator.chat(
            session_key="e2e-breaker-test",
            text="Say hello.",
            mode="research_only",
            frontend="cli",
        )
        breaker = e2e_orchestrator.llm.get_breaker("ollama")
        status = breaker.status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
