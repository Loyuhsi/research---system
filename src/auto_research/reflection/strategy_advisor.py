"""Strategy advisor: uses an LLM to propose remediation strategies from gap reports."""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from ..http_client import JsonHttpClient
from ..runtime import AutoResearchConfig
from ..services.llm_provider import LlmProviderService
from .gap_detector import GapReport


class StrategyAdvisor:
    """Takes a GapReport and uses an LLM to propose remediation strategies."""

    def __init__(
        self,
        config: AutoResearchConfig,
        http_client: JsonHttpClient,
        llm: LlmProviderService,
    ) -> None:
        self.config = config
        self.http_client = http_client
        self.llm = llm

    def advise(self, gap_report: GapReport, provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, object]:
        """Generate strategy advice from a gap report."""
        if not gap_report.has_gaps:
            return {
                "has_gaps": False,
                "advice": "No gaps detected. System is running smoothly.",
                "source": "rule-based",
            }

        selected_provider = (provider or self.config.provider).lower()
        selected_model = model or self.llm.default_model_for_provider(selected_provider)
        prompt = self._build_prompt(gap_report)

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
                                "You are the Auto-Research system's self-reflection advisor. "
                                "Analyze failure patterns and propose concrete, actionable remediation strategies. "
                                "Reply in Traditional Chinese. Be concise and practical."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "think": False,
                },
                headers=self.llm.model_headers(selected_provider),
                timeout=90,
            )
            content = self.llm.extract_message_content(response)
            if content:
                return {
                    "has_gaps": True,
                    "advice": content,
                    "source": "llm",
                    "provider": selected_provider,
                    "model": selected_model,
                }
        except Exception:
            logger.debug("LLM strategy advice failed", exc_info=True)

        return {
            "has_gaps": True,
            "advice": "\n".join(f"- {item}" for item in gap_report.recommendations),
            "source": "rule-based",
        }

    def _build_prompt(self, report: GapReport) -> str:
        lines = [
            "以下是 Auto-Research 系統最近的失敗與缺口摘要：",
            "",
            f"- 任務總數: {report.total_tasks}",
            f"- 失敗數量: {report.total_failures}",
            f"- 失敗率: {report.failure_rate:.1%}",
        ]
        if report.recurring_keywords:
            lines.append(f"- 常見關鍵詞: {', '.join(report.recurring_keywords)}")
        if report.failure_summaries:
            lines.append("")
            lines.append("失敗摘要：")
            for index, summary in enumerate(report.failure_summaries[:5], 1):
                lines.append(f"{index}. {summary}")
        lines.extend(
            [
                "",
                "請提出一組可執行的改善建議，並優先說明：",
                "1. 最可能的根因",
                "2. 應先補強的工作流或工具層",
                "3. 可以新增或修改的 skill 類型",
                "4. 低風險、可先落地的下一步",
            ]
        )
        return "\n".join(lines)
