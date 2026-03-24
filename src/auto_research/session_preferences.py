from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SessionPreferences:
    provider: Optional[str] = None


class SessionPreferencesStore:
    """Thread-safe in-memory provider override store for Telegram/Pi sessions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._global_provider: Optional[str] = None
        self._sessions: Dict[str, SessionPreferences] = {}

    def get_provider(self, session_key: Optional[str] = None) -> Optional[str]:
        with self._lock:
            if session_key:
                pref = self._sessions.get(session_key)
                if pref and pref.provider:
                    return pref.provider
            return self._global_provider

    def set_provider(
        self,
        provider: Optional[str],
        *,
        session_key: Optional[str] = None,
        global_scope: bool = False,
    ) -> None:
        with self._lock:
            if global_scope:
                self._global_provider = provider
                return
            if not session_key:
                raise ValueError("session_key is required for session-scoped provider overrides")
            pref = self._sessions.setdefault(session_key, SessionPreferences())
            pref.provider = provider

    def clear_session(self, session_key: str) -> None:
        with self._lock:
            self._sessions.pop(session_key, None)

    def status(self, session_key: Optional[str] = None) -> dict:
        with self._lock:
            return {
                "global_provider": self._global_provider,
                "session_provider": self.get_provider(session_key) if session_key else None,
            }
