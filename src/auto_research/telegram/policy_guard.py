"""Policy guard for the Telegram control plane."""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from .intent_parser import ParsedIntent

logger = logging.getLogger(__name__)


class ActionPolicy(str, Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    DISABLED = "disabled"


INTENT_POLICIES: Dict[str, ActionPolicy] = {
    "status": ActionPolicy.SAFE,
    "list_providers": ActionPolicy.SAFE,
    "diagnose_provider": ActionPolicy.SAFE,
    "search_memory": ActionPolicy.SAFE,
    "list_sessions": ActionPolicy.SAFE,
    "show_report": ActionPolicy.SAFE,
    "chat": ActionPolicy.SAFE,
    "start_research": ActionPolicy.CONFIRM,
    "export_obsidian": ActionPolicy.CONFIRM,
    "select_provider": ActionPolicy.CONFIRM,
    "search_sources": ActionPolicy.CONFIRM,
    "research_with_search": ActionPolicy.CONFIRM,
}

DISALLOWED_CHAT_PATTERNS = (
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bdel\s+/[sqf]\b", re.IGNORECASE),
    re.compile(r"\bformat\s+[a-z]:\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\bexecute\s+shell\b", re.IGNORECASE),
    re.compile(r"\brun\s+shell\b", re.IGNORECASE),
    re.compile(r"\bpowershell\b", re.IGNORECASE),
    re.compile(r"\bcmd(?:\.exe)?\b", re.IGNORECASE),
    re.compile(r"\bremove-item\b", re.IGNORECASE),
)


@dataclass
class ActionProposal:
    """Structured proposal for policy-gated action execution."""

    actor_chat_id: int
    intent: ParsedIntent
    risk_level: str
    policy: ActionPolicy
    audit_id: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.audit_id:
            self.audit_id = str(uuid.uuid4())[:12]
        if not self.timestamp:
            self.timestamp = dt.datetime.now(dt.timezone.utc).isoformat()


class PolicyGuard:
    """Checks actions against policy and writes audit telemetry."""

    def __init__(self, telemetry_path: Path) -> None:
        self._telemetry_path = telemetry_path

    def check(self, intent: ParsedIntent) -> Dict[str, Any]:
        policy = self._resolve_policy(intent)
        return {
            "allowed": policy != ActionPolicy.DISABLED,
            "policy": policy.value,
            "reason": f"intent={intent.intent}, policy={policy.value}",
        }

    def propose(self, chat_id: int, intent: ParsedIntent) -> ActionProposal:
        policy = self._resolve_policy(intent)
        return ActionProposal(
            actor_chat_id=chat_id,
            intent=intent,
            risk_level=self._classify_risk(intent, policy),
            policy=policy,
        )

    def _resolve_policy(self, intent: ParsedIntent) -> ActionPolicy:
        if self._looks_dangerous_chat(intent):
            return ActionPolicy.DISABLED
        return INTENT_POLICIES.get(intent.intent, ActionPolicy.DISABLED)

    def _looks_dangerous_chat(self, intent: ParsedIntent) -> bool:
        texts_to_check = [
            str(v).strip()
            for v in intent.args.values()
            if isinstance(v, str) and str(v).strip()
        ]
        if not texts_to_check:
            return False
        return any(
            pattern.search(text)
            for text in texts_to_check
            for pattern in DISALLOWED_CHAT_PATTERNS
        )

    @staticmethod
    def _classify_risk(intent: ParsedIntent, policy: ActionPolicy) -> str:
        if policy == ActionPolicy.DISABLED:
            return "high"
        if policy == ActionPolicy.CONFIRM:
            return "medium"
        if intent.parse_layer == "llm" and intent.confidence < 0.85:
            return "medium"
        return "low"

    def log_action(
        self,
        intent: ParsedIntent,
        result_summary: str,
        proposal: Optional[ActionProposal] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event_type": "telegram_action",
            "intent": intent.intent,
            "args": intent.args,
            "confidence": intent.confidence,
            "parse_layer": intent.parse_layer,
            "result_summary": result_summary[:200],
        }
        if proposal:
            entry["audit_id"] = proposal.audit_id
            entry["risk_level"] = proposal.risk_level
            entry["policy"] = proposal.policy.value
            entry["actor_chat_id"] = proposal.actor_chat_id
        try:
            self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with self._telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.exception("Failed to log Telegram action")
