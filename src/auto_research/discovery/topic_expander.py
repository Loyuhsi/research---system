"""Topic expander: reads existing notes and suggests related research topics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from ..http_client import JsonHttpClient
from ..runtime import AutoResearchConfig
from ..services.llm_provider import LlmProviderService


class TopicExpander:
    """Reads existing research notes and uses an LLM to suggest related topics."""

    def __init__(
        self,
        config: AutoResearchConfig,
        http_client: JsonHttpClient,
        llm: LlmProviderService,
    ) -> None:
        self.config = config
        self.http_client = http_client
        self.llm = llm

    def discover(self, provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, object]:
        """Scan existing notes and propose new research topics."""
        notes_dir = self.config.repo_root / "output" / "notes"
        existing_topics = self._extract_topics(notes_dir)

        if not existing_topics:
            return {
                "existing_topics": [],
                "suggestions": ["No existing notes found. Start with fetch-public to build your first research note."],
                "source": "rule-based",
            }

        selected_provider = (provider or self.config.provider).lower()
        selected_model = model or self.llm.default_model_for_provider(selected_provider)

        try:
            response = self.llm.call_with_breaker(
                selected_provider,
                self.http_client.request_json,
                "POST",
                self.llm.chat_url_for_provider(selected_provider),
                payload={
                    "model": selected_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a research topic advisor. Based on existing research topics, "
                                "suggest five related but still underexplored directions. "
                                "Reply in Traditional Chinese and format the answer as a numbered list."
                            ),
                        },
                        {"role": "user", "content": self._build_prompt(existing_topics)},
                    ],
                    "temperature": 0.5,
                    "think": False,
                },
                headers=self.llm.model_headers(selected_provider),
                timeout=60,
            )
            content = self.llm.extract_message_content(response)
            if content:
                return {
                    "existing_topics": existing_topics,
                    "suggestions": content,
                    "source": "llm",
                    "provider": selected_provider,
                    "model": selected_model,
                }
        except Exception:
            logger.debug("LLM topic expansion failed", exc_info=True)

        return {
            "existing_topics": existing_topics,
            "suggestions": [f"Based on {len(existing_topics)} existing topics, consider deeper analysis of each."],
            "source": "rule-based",
        }

    def _extract_topics(self, notes_dir: Path) -> List[str]:
        """Extract topic names from YAML frontmatter only."""
        topics: List[str] = []
        if not notes_dir.exists():
            return topics
        for note_path in sorted(notes_dir.glob("*.md")):
            try:
                content = note_path.read_text(encoding="utf-8", errors="replace")
                frontmatter = self._frontmatter_map(content)
                topic = str(frontmatter.get("topic", "")).strip().strip('"').strip("'")
                if topic and topic not in topics:
                    topics.append(topic)
            except OSError:
                logger.debug("Skipping unreadable note %s", note_path, exc_info=True)
                continue
        return topics

    def _frontmatter_map(self, content: str) -> Dict[str, str]:
        if not content.startswith("---"):
            return {}
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}
        payload: Dict[str, str] = {}
        for line in lines[1:]:
            stripped = line.strip()
            if stripped == "---":
                break
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            payload[key.strip().lower()] = value.strip()
        return payload

    def _build_prompt(self, topics: List[str]) -> str:
        lines = [
            "以下是目前已完成的研究主題：",
            "",
        ]
        for index, topic in enumerate(topics, 1):
            lines.append(f"{index}. {topic}")
        lines.extend(
            [
                "",
                "請提出 5 個值得繼續延伸的新研究方向。",
                "每個方向都要具體，並說明為什麼值得研究。",
                "避免重複既有主題。",
            ]
        )
        return "\n".join(lines)
