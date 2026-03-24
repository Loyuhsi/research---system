from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

MAX_HISTORY_ROUNDS = 12


class ConversationStoreBase(ABC):
    """Abstract base for conversation stores."""

    @abstractmethod
    def reset(self, session_key: str) -> None: ...

    @abstractmethod
    def get_messages(self, session_key: str) -> List[Dict[str, str]]: ...

    @abstractmethod
    def append_turn(self, session_key: str, user_text: str, assistant_text: str) -> None: ...

    def build_prompt_messages(self, session_key: str, system_prompt: str, user_text: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            *self.get_messages(session_key),
            {"role": "user", "content": user_text},
        ]


class InMemoryConversationStore(ConversationStoreBase):
    """Thread-safe in-memory store. Lost on restart."""

    def __init__(self, max_rounds: int = MAX_HISTORY_ROUNDS):
        self.max_rounds = max_rounds
        self._lock = threading.RLock()
        self._messages: Dict[str, List[Dict[str, str]]] = {}

    def reset(self, session_key: str) -> None:
        with self._lock:
            self._messages.pop(session_key, None)

    def get_messages(self, session_key: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._messages.get(session_key, []))

    def append_turn(self, session_key: str, user_text: str, assistant_text: str) -> None:
        with self._lock:
            if session_key not in self._messages:
                self._messages[session_key] = []
            messages = self._messages[session_key]
            messages.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ])
            max_messages = self.max_rounds * 2
            if len(messages) > max_messages:
                del messages[:-max_messages]


class JsonFileConversationStore(ConversationStoreBase):
    """Persistent store using JSON files in a directory."""

    def __init__(self, store_dir: Path, max_rounds: int = MAX_HISTORY_ROUNDS):
        self.store_dir = store_dir
        self.max_rounds = max_rounds
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_key: str) -> Path:
        safe_name = session_key.replace(":", "_").replace("/", "_").replace("\\", "_")
        return self.store_dir / f"{safe_name}.json"

    def reset(self, session_key: str) -> None:
        path = self._session_path(session_key)
        if path.exists():
            path.unlink()

    def get_messages(self, session_key: str) -> List[Dict[str, str]]:
        path = self._session_path(session_key)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read conversation %s: %s", session_key, exc)
            return []

    def append_turn(self, session_key: str, user_text: str, assistant_text: str) -> None:
        messages = self.get_messages(session_key)
        messages.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])
        max_messages = self.max_rounds * 2
        if len(messages) > max_messages:
            del messages[:-max_messages]
        path = self._session_path(session_key)
        path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
