"""Per-chat conversation state for Telegram control plane.

Tracks pending confirmations with TTL-based expiry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .intent_parser import ParsedIntent

CONFIRMATION_TTL_SECONDS = 120.0  # pending confirmations expire after 2 minutes


@dataclass
class ChatState:
    """State for a single Telegram chat."""

    pending_confirmation: Optional[ParsedIntent] = None
    pending_timestamp: float = 0.0


class ConversationState:
    """Manages per-chat conversation state."""

    def __init__(self) -> None:
        self._states: Dict[int, ChatState] = {}

    def get(self, chat_id: int) -> ChatState:
        if chat_id not in self._states:
            self._states[chat_id] = ChatState()
        return self._states[chat_id]

    def set_pending(self, chat_id: int, intent: ParsedIntent) -> None:
        state = self.get(chat_id)
        state.pending_confirmation = intent
        state.pending_timestamp = time.monotonic()

    def confirm_pending(self, chat_id: int) -> Optional[ParsedIntent]:
        """Return pending intent if not expired, then clear. Returns None if expired or empty."""
        state = self.get(chat_id)
        if state.pending_confirmation is None:
            return None
        if self._is_expired(state):
            self.clear_pending(chat_id)
            return None
        intent = state.pending_confirmation
        state.pending_confirmation = None
        state.pending_timestamp = 0.0
        return intent

    def clear_pending(self, chat_id: int) -> None:
        state = self.get(chat_id)
        state.pending_confirmation = None
        state.pending_timestamp = 0.0

    def _is_expired(self, state: ChatState) -> bool:
        if state.pending_timestamp == 0.0:
            return True
        return (time.monotonic() - state.pending_timestamp) > CONFIRMATION_TTL_SECONDS
