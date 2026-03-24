"""Tests for conversation_store.py — InMemory and JsonFile stores."""

from __future__ import annotations

import json

import pytest

from auto_research.conversation_store import (
    InMemoryConversationStore,
    JsonFileConversationStore,
)


class TestInMemoryConversationStore:
    def test_empty_messages(self):
        store = InMemoryConversationStore()
        assert store.get_messages("s1") == []

    def test_append_and_get(self):
        store = InMemoryConversationStore()
        store.append_turn("s1", "hello", "hi")
        msgs = store.get_messages("s1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_reset_clears(self):
        store = InMemoryConversationStore()
        store.append_turn("s1", "a", "b")
        store.reset("s1")
        assert store.get_messages("s1") == []

    def test_max_rounds_truncation(self):
        store = InMemoryConversationStore(max_rounds=2)
        for i in range(5):
            store.append_turn("s1", f"u{i}", f"a{i}")
        msgs = store.get_messages("s1")
        # 2 rounds = 4 messages
        assert len(msgs) == 4

    def test_build_prompt_messages(self):
        store = InMemoryConversationStore()
        store.append_turn("s1", "prev_q", "prev_a")
        msgs = store.build_prompt_messages("s1", "system_prompt", "new_q")
        assert msgs[0] == {"role": "system", "content": "system_prompt"}
        assert msgs[-1] == {"role": "user", "content": "new_q"}
        assert len(msgs) == 4  # system + 2 history + new user


class TestJsonFileConversationStore:
    def test_empty_messages(self, tmp_path):
        store = JsonFileConversationStore(tmp_path)
        assert store.get_messages("s1") == []

    def test_append_and_get(self, tmp_path):
        store = JsonFileConversationStore(tmp_path)
        store.append_turn("s1", "hello", "hi")
        msgs = store.get_messages("s1")
        assert len(msgs) == 2

    def test_persistence(self, tmp_path):
        store1 = JsonFileConversationStore(tmp_path)
        store1.append_turn("s1", "q", "a")
        # New store instance reads same data
        store2 = JsonFileConversationStore(tmp_path)
        assert len(store2.get_messages("s1")) == 2

    def test_reset_deletes_file(self, tmp_path):
        store = JsonFileConversationStore(tmp_path)
        store.append_turn("s1", "q", "a")
        store.reset("s1")
        assert store.get_messages("s1") == []

    def test_max_rounds_truncation(self, tmp_path):
        store = JsonFileConversationStore(tmp_path, max_rounds=2)
        for i in range(5):
            store.append_turn("s1", f"u{i}", f"a{i}")
        assert len(store.get_messages("s1")) == 4

    def test_corrupted_file_returns_empty(self, tmp_path):
        store = JsonFileConversationStore(tmp_path)
        # Write corrupted JSON
        path = store._session_path("s1")
        path.write_text("not json", encoding="utf-8")
        assert store.get_messages("s1") == []

    def test_session_key_sanitized(self, tmp_path):
        store = JsonFileConversationStore(tmp_path)
        store.append_turn("a:b/c\\d", "q", "a")
        assert len(store.get_messages("a:b/c\\d")) == 2
