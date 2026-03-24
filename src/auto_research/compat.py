"""Backward-compatible shim methods for Orchestrator.

These methods delegate to the appropriate service. They exist so that external
callers (Pi runtime, Telegram bot, CLI) continue to work without changes.
New code should call the service directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

from .services.report import ReportService


class BackwardCompatMixin:
    """Mixin providing backward-compatible delegations.

    Expects the host class to have: vault_service, llm, fetcher, registry,
    http_client, report_service.
    """

    # -- Report service shims --------------------------------------------------

    def _select_report_session(self) -> Optional[str]:
        return self.report_service.select_session()

    def _list_report_sessions(self):
        return self.report_service.list_sessions()

    def _session_mtime(self, session_id: str) -> float:
        return self.report_service.session_mtime(session_id)

    def _research_program_topic(self) -> str:
        return self.report_service.research_program_topic()

    def _parse_frontmatter(self, content: str) -> Dict[str, str]:
        return ReportService.parse_frontmatter(content)

    def _read_note_metadata(self, note_path: Path) -> Dict[str, str]:
        return ReportService.read_note_metadata(note_path)

    def _load_json_file(self, path: Path) -> Dict[str, object]:
        return ReportService.load_json_file(path)

    def _topic_match_score(self, target: str, candidate: str) -> float:
        return ReportService.topic_match_score(target, candidate)

    def _topic_from_session_id(self, session_id: str) -> str:
        return ReportService.topic_from_session_id(session_id)

    def _latest_evaluation_for_session(self, session_id: str) -> Dict[str, object]:
        return self.report_service.latest_evaluation(session_id)

    def _memory_status(self, session_id: str) -> Dict[str, object]:
        return self.report_service.memory_status(session_id)

    def _recent_session_telemetry(self, session_id: str, limit: int = 5) -> Dict[str, object]:
        return self.report_service.recent_telemetry(session_id, limit)

    def _recent_session_trace(self, session_id: str) -> Dict[str, object]:
        return self.report_service.recent_trace(session_id)

    def _resolve_source_count(self, status_payload: Mapping[str, object], note_meta: Mapping[str, str]) -> int:
        return ReportService.resolve_source_count(status_payload, note_meta)

    def _quality_score(self, quality: object) -> float:
        return ReportService.quality_score(quality)

    # -- Service shims kept only where tests or external callers depend on them --

    def extract_message_content(self, response):
        return self.llm.extract_message_content(response)

    @staticmethod
    def is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root.resolve())
            return True
        except ValueError:
            return False
