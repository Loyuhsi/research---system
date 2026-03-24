"""Tests for SessionPreferencesStore."""
from __future__ import annotations

import pytest

from auto_research.session_preferences import SessionPreferencesStore


class TestSessionPreferencesStore:
    def test_get_provider_empty(self):
        store = SessionPreferencesStore()
        assert store.get_provider() is None
        assert store.get_provider("session-1") is None

    def test_global_override(self):
        store = SessionPreferencesStore()
        store.set_provider("vllm", global_scope=True)
        assert store.get_provider() == "vllm"
        assert store.get_provider("any-session") == "vllm"

    def test_session_override_takes_precedence(self):
        store = SessionPreferencesStore()
        store.set_provider("ollama", global_scope=True)
        store.set_provider("lmstudio", session_key="s1")
        assert store.get_provider("s1") == "lmstudio"
        assert store.get_provider("s2") == "ollama"
        assert store.get_provider() == "ollama"

    def test_session_key_required_without_global(self):
        store = SessionPreferencesStore()
        with pytest.raises(ValueError, match="session_key is required"):
            store.set_provider("ollama")

    def test_clear_session(self):
        store = SessionPreferencesStore()
        store.set_provider("ollama", global_scope=True)
        store.set_provider("vllm", session_key="s1")
        store.clear_session("s1")
        assert store.get_provider("s1") == "ollama"

    def test_clear_nonexistent_session(self):
        store = SessionPreferencesStore()
        store.clear_session("nonexistent")  # should not raise

    def test_status_shape(self):
        store = SessionPreferencesStore()
        store.set_provider("ollama", global_scope=True)
        store.set_provider("vllm", session_key="s1")
        status = store.status("s1")
        assert status["global_provider"] == "ollama"
        assert status["session_provider"] == "vllm"

    def test_status_without_session(self):
        store = SessionPreferencesStore()
        status = store.status()
        assert status["global_provider"] is None
        assert status["session_provider"] is None

    def test_set_none_clears_override(self):
        store = SessionPreferencesStore()
        store.set_provider("ollama", session_key="s1")
        store.set_provider(None, session_key="s1")
        assert store.get_provider("s1") is None
