"""Hybrid intent parser for the Telegram control plane."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ..runtime import AutoResearchConfig

logger = logging.getLogger(__name__)

KNOWN_INTENTS = frozenset(
    {
        "status",
        "list_providers",
        "select_provider",
        "start_research",
        "list_sessions",
        "show_report",
        "search_memory",
        "export_obsidian",
        "diagnose_provider",
        "search_sources",
        "research_with_search",
        "chat",
    }
)

ALLOWED_ARGS: Dict[str, frozenset[str]] = {
    "status": frozenset(),
    "list_providers": frozenset(),
    "select_provider": frozenset({"provider", "scope"}),
    "start_research": frozenset({"topic"}),
    "list_sessions": frozenset(),
    "show_report": frozenset({"session_id"}),
    "search_memory": frozenset({"query"}),
    "export_obsidian": frozenset({"session_id"}),
    "diagnose_provider": frozenset({"provider"}),
    "search_sources": frozenset({"query"}),
    "research_with_search": frozenset({"topic"}),
    "chat": frozenset({"text"}),
}

ACTION_VERBS = re.compile(
    r"(status|health|doctor|list|show|report|start|research|search|find|switch|select|use|provider|"
    r"狀態|健康|診斷|列出|顯示|報告|開始|研究|搜尋|查詢|切換|改用|使用|提供者|來源|記憶)",
    re.IGNORECASE,
)
PROVIDER_RE = re.compile(r"\b(ollama|lmstudio|vllm|llamacpp)\b", re.IGNORECASE)
SESSION_RE = re.compile(r"\b([a-z0-9][a-z0-9_-]{5,})\b", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedIntent:
    intent: str
    args: Dict[str, str]
    confidence: float
    clarification: str
    parse_layer: str


INTENT_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": sorted(KNOWN_INTENTS)},
        "args": {"type": "object"},
        "confidence": {"type": "number"},
    },
    "required": ["intent", "args", "confidence"],
    "additionalProperties": False,
}


class IntentParser:
    CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        config: AutoResearchConfig,
        llm_service: Any = None,
        http_client: Any = None,
    ) -> None:
        self._config = config
        self._llm = llm_service
        self._http = http_client

    def parse(self, text: str) -> ParsedIntent:
        stripped = text.strip()
        if not stripped:
            return ParsedIntent("chat", {"text": ""}, 1.0, "", "keyword")

        keyword_result = self._keyword_match(stripped)
        if keyword_result and keyword_result.confidence >= 0.9:
            return keyword_result

        if ACTION_VERBS.search(stripped) and self._llm and self._http:
            llm_result = self._llm_classify(stripped)
            if llm_result and llm_result.confidence >= self.CONFIDENCE_THRESHOLD:
                return llm_result

        if keyword_result and keyword_result.confidence > 0:
            return ParsedIntent(
                intent=keyword_result.intent,
                args=keyword_result.args,
                confidence=keyword_result.confidence,
                clarification=f"你是要{_intent_label(keyword_result.intent)}，還是一般對話？",
                parse_layer="fallback",
            )

        if ACTION_VERBS.search(stripped):
            return ParsedIntent(
                intent="chat",
                args={"text": stripped},
                confidence=0.5,
                clarification="請再具體一點，例如「查狀態」「列出 provider」「顯示報告」。",
                parse_layer="fallback",
            )

        return ParsedIntent("chat", {"text": stripped}, 1.0, "", "keyword")

    def _keyword_match(self, text: str) -> Optional[ParsedIntent]:
        lower = text.lower()

        if re.search(r"(status|health|doctor|狀態|健康)", lower):
            return ParsedIntent("status", {}, 0.95, "", "keyword")

        if re.search(
            r"(list.*provider|provider.*list|providers|list.*backend|backends|哪些.*provider|哪些.*backend|列出.*provider|列出.*模型提供者)",
            lower,
        ):
            return ParsedIntent("list_providers", {}, 0.95, "", "keyword")

        if re.search(r"(list.*session|sessions|最近.*session|列出.*session)", lower):
            return ParsedIntent("list_sessions", {}, 0.95, "", "keyword")

        if re.search(r"(show.*report|report|顯示.*報告|查看.*報告|研究報告)", lower):
            session_match = SESSION_RE.search(text)
            args = {"session_id": session_match.group(1)} if session_match else {}
            return ParsedIntent("show_report", args, 0.92, "", "keyword")

        if re.search(r"(select|switch|use|改用|切換|使用).*(ollama|lmstudio|vllm|llamacpp)", lower):
            provider_match = PROVIDER_RE.search(lower)
            args = {"provider": provider_match.group(1).lower()} if provider_match else {}
            if re.search(r"(global|default|全域|預設)", lower):
                args["scope"] = "global"
            return ParsedIntent("select_provider", args, 0.95, "", "keyword")

        if re.search(r"(diagnose|診斷).*(ollama|lmstudio|vllm|llamacpp|provider)", lower):
            provider_match = PROVIDER_RE.search(lower)
            args = {"provider": provider_match.group(1).lower()} if provider_match else {}
            return ParsedIntent("diagnose_provider", args, 0.95, "", "keyword")

        if re.search(r"(search|find|搜尋|查詢).*(memo|memory|記憶)", lower):
            query = self._strip_patterns(
                text,
                ["search", "find", "搜尋", "查詢", "memo", "memory", "記憶"],
            )
            return ParsedIntent("search_memory", {"query": query} if query else {}, 0.95, "", "keyword")

        if re.search(r"(export|匯出).*(obsidian|vault|筆記)", lower):
            session_match = SESSION_RE.search(text)
            args = {"session_id": session_match.group(1)} if session_match else {}
            return ParsedIntent("export_obsidian", args, 0.95, "", "keyword")

        if re.search(r"(research with search|search and research|搜尋並研究|研究並搜尋|research.*search)", lower):
            topic = self._strip_patterns(
                text,
                [
                    "research with search",
                    "search and research",
                    "搜尋並研究",
                    "研究並搜尋",
                    "research",
                    "search",
                    "搜尋",
                    "研究",
                    "about",
                    "關於",
                ],
            )
            return ParsedIntent("research_with_search", {"topic": topic} if topic else {}, 0.95, "", "keyword")

        if re.search(r"(search.*sources?|find.*sources?|搜尋來源|找來源)", lower):
            query = self._strip_patterns(
                text,
                ["sources", "source", "search", "find", "搜尋來源", "找來源", "搜尋", "來源"],
            )
            return ParsedIntent("search_sources", {"query": query} if query else {}, 0.95, "", "keyword")

        if re.search(r"(start|begin|開始).*(research|研究|task)", lower):
            topic = self._strip_patterns(
                text,
                ["start", "begin", "開始", "research", "研究", "task", "about", "關於"],
            )
            return ParsedIntent("start_research", {"topic": topic} if topic else {}, 0.95, "", "keyword")

        return None

    def _llm_classify(self, text: str) -> Optional[ParsedIntent]:
        provider = self._config.provider
        model = self._config.model
        system_prompt = (
            "You classify Telegram control messages for a local Auto-Research system. "
            "Choose the most appropriate intent and return JSON only."
        )
        try:
            result = self._llm.call_structured(
                provider,
                self._http,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Known intents: {', '.join(sorted(KNOWN_INTENTS))}\n"
                            "Use args only when needed.\n"
                            f"Message: {text}"
                        ),
                    },
                ],
                schema=INTENT_SCHEMA,
                timeout=20,
            )
            return self._validate_llm_output(result.get("data", {}), text)
        except Exception as exc:
            logger.warning("LLM intent classification failed: %s", exc)
            return None

    def _validate_llm_output(
        self,
        content: str | Mapping[str, object],
        original_text: str,
    ) -> Optional[ParsedIntent]:
        try:
            if isinstance(content, Mapping):
                data = dict(content)
            else:
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if not json_match:
                    return None
                data = json.loads(json_match.group())

            intent = str(data.get("intent", ""))
            if intent not in KNOWN_INTENTS:
                return None

            raw_args = data.get("args", {})
            if not isinstance(raw_args, dict):
                raw_args = {}
            allowed = ALLOWED_ARGS.get(intent, frozenset())
            args = {
                key: str(value).strip()
                for key, value in raw_args.items()
                if key in allowed and str(value).strip()
            }
            if intent == "chat" and "text" not in args:
                args["text"] = original_text

            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            return ParsedIntent(
                intent=intent,
                args=args,
                confidence=confidence,
                clarification="",
                parse_layer="llm",
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    @staticmethod
    def _strip_patterns(text: str, patterns: list[str]) -> str:
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(" ：:")


def _intent_label(intent: str) -> str:
    labels = {
        "status": "查詢系統狀態",
        "list_providers": "列出 provider 狀態",
        "select_provider": "切換 provider",
        "start_research": "開始研究任務",
        "list_sessions": "列出最近 session",
        "show_report": "查看研究報告",
        "search_memory": "搜尋記憶",
        "export_obsidian": "匯出到 Obsidian",
        "diagnose_provider": "診斷 provider",
        "search_sources": "搜尋網路來源",
        "research_with_search": "搜尋並研究",
        "chat": "一般對話",
    }
    return labels.get(intent, intent)
