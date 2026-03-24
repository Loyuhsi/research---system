"""Telegram action registry."""

from __future__ import annotations

import logging
import re
import time as _time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

SLUG_RE = re.compile(r"[^a-z0-9]+")
MINIMUM_VIABLE_SOURCES = 2
MAX_SOURCE_BYTES = 102400


class ActionRegistry:
    """Maps parsed intents to orchestrator-backed handlers."""

    def __init__(self, orchestrator: Any, config: Any) -> None:
        self._orchestrator = orchestrator
        self._config = config
        self._handlers: Dict[str, Callable[[Dict[str, str], str], str]] = {
            "status": self._handle_status,
            "list_providers": self._handle_list_providers,
            "select_provider": self._handle_select_provider,
            "search_memory": self._handle_search_memory,
            "start_research": self._handle_start_research,
            "export_obsidian": self._handle_export_obsidian,
            "diagnose_provider": self._handle_diagnose_provider,
            "chat": self._handle_chat,
            "list_sessions": self._handle_list_sessions,
            "show_report": self._handle_show_report,
            "search_sources": self._handle_search_sources,
            "research_with_search": self._handle_research_with_search,
        }

    def execute(self, intent: str, args: Dict[str, str], session_key: str) -> str:
        handler = self._handlers.get(intent, self._handle_unknown)
        try:
            return handler(args, session_key)
        except Exception as exc:
            logger.exception("Action handler error for intent=%s", intent)
            return f"執行 `{intent}` 失敗：{exc}"

    def execute_proposal(self, proposal: Any, session_key: str) -> str:
        return self.execute(proposal.intent.intent, proposal.intent.args, session_key)

    def _handle_status(self, args: Dict[str, str], session_key: str) -> str:
        doctor = self._orchestrator.doctor()
        services = doctor.get("services", {})
        override_status = (
            self._orchestrator.provider_override_status(session_key)
            if hasattr(self._orchestrator, "provider_override_status")
            else {}
        )
        lines = ["系統狀態"]
        lines.append(f"預設 provider: {doctor.get('provider')} / {doctor.get('model')}")
        if override_status:
            lines.append(
                "目前有效 provider: "
                f"{override_status.get('effective_provider')} / {override_status.get('effective_model')}"
            )
        for name, info in services.items():
            ok = bool(info.get("ok")) if isinstance(info, dict) else False
            lines.append(f"- {name}: {'OK' if ok else 'FAIL'}")
        selection = doctor.get("provider_selection", {})
        if isinstance(selection, dict) and selection:
            lines.append(f"自動選擇: {selection.get('provider')} ({selection.get('reason')})")
        return "\n".join(lines)

    def _handle_list_providers(self, args: Dict[str, str], session_key: str) -> str:
        matrix = self._orchestrator.llm.provider_capability_matrix(self._orchestrator.http_client)
        lines = ["Provider Matrix"]
        for provider, info in matrix.items():
            if not isinstance(info, dict):
                continue
            health = "OK" if info.get("health_ready") else "FAIL"
            inference = "OK" if info.get("inference_ready") else "FAIL"
            embeddings = "OK" if info.get("embedding_ready") else "FAIL"
            marker = " (primary)" if info.get("is_primary") else ""
            lines.append(
                f"- {provider}{marker}: health={health}, "
                f"inference={inference}, embedding={embeddings}"
            )
        return "\n".join(lines)

    def _handle_select_provider(self, args: Dict[str, str], session_key: str) -> str:
        provider = args.get("provider", "").strip().lower()
        if not provider:
            return "請指定 provider，例如：`切換 provider lmstudio`。"
        scope = args.get("scope", "session").strip().lower() or "session"
        global_scope = scope == "global"
        result = self._orchestrator.set_provider_override(
            provider,
            session_key=session_key,
            global_scope=global_scope,
        )
        scope_label = "全域" if global_scope else "目前 session"
        return (
            f"已更新 {scope_label} 的 provider 設定。\n"
            f"設定值: {result.get('provider') or 'auto'}\n"
            f"目前有效 provider: {result.get('effective_provider')} / {result.get('model')}"
        )

    def _handle_search_memory(self, args: Dict[str, str], session_key: str) -> str:
        query = args.get("query", "").strip()
        if not query:
            return "請提供搜尋字詞，例如：`搜尋記憶 Python 測試`。"
        result = self._orchestrator.memory_search(task=query)
        hits = result.get("memory_hits", [])
        if not hits:
            return f"找不到和「{query}」相關的記憶。"
        lines = [f"記憶搜尋結果：{len(hits)} 筆"]
        for hit in hits[:5]:
            lines.append(
                f"- {hit.get('title', hit.get('id', 'memory'))} "
                f"(score={float(hit.get('score', 0)):.2f})"
            )
        return "\n".join(lines)

    def _handle_start_research(self, args: Dict[str, str], session_key: str) -> str:
        topic = args.get("topic", "").strip() or self._load_research_program().get("priority_topic", "")
        if not topic:
            return "請提供研究主題，例如：`開始研究 Python Testing`。"
        session_id = f"tg-{_slug(topic)}-{int(_time.time())}"
        layout = self._config.resolve_layout(session_id)
        layout.ensure()
        if not any(layout.parsed_dir.glob("*.md")):
            (layout.parsed_dir / "seed.md").write_text(f"# {topic}\n\n{topic}\n", encoding="utf-8")
        try:
            result, quality, memory_result, export_path = self._run_bounded_research(
                topic,
                session_id,
                layout,
                run_kind="telegram_bounded",
            )
        except Exception as exc:
            return f"研究任務失敗：{exc}"
        return self._format_research_completion(topic, session_id, result, quality, memory_result, export_path)

    def _handle_export_obsidian(self, args: Dict[str, str], session_key: str) -> str:
        session_id = args.get("session_id", "").strip()
        if session_id:
            try:
                result = self._orchestrator.promote_note(session_id, approved=True)
                return f"已將研究筆記匯出到 Obsidian：{result.get('note_path', session_id)}"
            except Exception as exc:
                return f"Obsidian 匯出失敗：{exc}"
        try:
            from ..integrations.obsidian import ObsidianExporter

            exporter = ObsidianExporter(self._config)
            path = exporter.export_diagnostics(self._orchestrator.doctor())
            return f"已將診斷報告匯出到 Obsidian：{path.name}"
        except Exception as exc:
            return f"Obsidian 匯出失敗：{exc}"

    def _handle_diagnose_provider(self, args: Dict[str, str], session_key: str) -> str:
        provider = args.get("provider", self._config.provider)
        inference = self._orchestrator.llm.check_inference(provider, self._orchestrator.http_client)
        embeddings = self._orchestrator.llm.check_embeddings(provider, self._orchestrator.http_client)
        lines = [f"Provider 診斷：{provider}"]
        lines.append(f"- Inference: {'OK' if inference.get('ok') else 'FAIL'}")
        if inference.get("ok"):
            lines.append(f"  model={inference.get('model')} latency={inference.get('latency_ms')}ms")
        else:
            lines.append(f"  error={inference.get('error', inference.get('response_snippet', '?'))}")
        lines.append(f"- Embeddings: {'OK' if embeddings.get('ok') else 'FAIL'}")
        if not embeddings.get("ok"):
            lines.append(f"  error={embeddings.get('error', '?')}")
        return "\n".join(lines)

    def _handle_chat(self, args: Dict[str, str], session_key: str) -> str:
        text = args.get("text", "")
        if not text:
            return "對話內容是空的。"
        try:
            result = self._orchestrator.chat(
                session_key=session_key,
                text=text,
                mode="research_only",
                frontend="telegram",
            )
            return f"回覆：{result.get('reply', '')}"
        except Exception:
            logger.exception("Chat error")
            return "目前發生錯誤，無法完成對話回覆，請稍後再試。"

    def _handle_list_sessions(self, args: Dict[str, str], session_key: str) -> str:
        notes_dir = self._config.repo_root / "output" / "notes"
        if not notes_dir.exists():
            return "目前還沒有研究 session。"
        notes = sorted(notes_dir.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True)[:10]
        if not notes:
            return "目前還沒有研究 session。"
        lines = ["最近的 sessions"]
        for note in notes:
            lines.append(f"- {note.stem}")
        return "\n".join(lines)

    def _handle_show_report(self, args: Dict[str, str], session_key: str) -> str:
        report = self._orchestrator.show_report(args.get("session_id") or None)
        if not isinstance(report, dict):
            return "報告功能尚未提供。"
        return self._format_report(report)

    def _handle_search_sources(self, args: Dict[str, str], session_key: str) -> str:
        query = args.get("query", "").strip()
        if not query:
            return "請提供搜尋字詞，例如：`搜尋來源 RAG evaluation`。"
        from ..search import WebSearchAdapter

        results = WebSearchAdapter().search(query, max_results=5)
        if not results:
            return f"找不到和「{query}」相關的來源。"
        lines = [f"來源搜尋結果：{len(results)} 筆"]
        for result in results:
            lines.append(f"- {result.title[:70]}")
            lines.append(f"  {result.url}")
        return "\n".join(lines)

    def _handle_research_with_search(self, args: Dict[str, str], session_key: str) -> str:
        topic = args.get("topic", "").strip() or str(self._load_research_program().get("priority_topic", "")).strip()
        if not topic:
            return "請提供研究主題。"
        try:
            from ..search import SourceFetcher, WebSearchAdapter

            results = WebSearchAdapter().search(topic, max_results=3)
            search_count = len(results)
            if not results:
                return f"未找到結果：找不到和「{topic}」相關的來源，因此無法繼續研究。"

            session_id = f"tg-search-{_slug(topic)}-{int(_time.time())}"
            layout = self._config.resolve_layout(session_id)
            layout.ensure()

            fetcher = SourceFetcher()
            fetched_count = 0
            failed_urls: list[str] = []
            for result in results[:3]:
                fetch_result = fetcher.fetch_and_parse(result.url, max_words=5000)
                if fetch_result.fetch_status == "ok":
                    fetcher.write_to_session(layout, result, fetch_result)
                    fetched_count += 1
                else:
                    failed_urls.append(f"{result.url} ({fetch_result.error or fetch_result.fetch_status})")

            if fetched_count == 0:
                lines = [f"所有擷取均失敗：共找到 {search_count} 筆來源，但全部擷取失敗。"]
                lines.extend(f"- {item}" for item in failed_urls)
                return "\n".join(lines)

            min_needed = min(MINIMUM_VIABLE_SOURCES, search_count)
            if fetched_count < min_needed:
                lines = [
                    f"共找到 {search_count} 筆來源，但只成功擷取 {fetched_count} 筆。",
                    f"目前最低需求是 {min_needed} 筆，先停止研究。",
                ]
                lines.extend(f"- {item}" for item in failed_urls)
                return "\n".join(lines)

            total_bytes = sum(item.stat().st_size for item in layout.parsed_dir.glob("*.md"))
            if total_bytes > MAX_SOURCE_BYTES * 2:
                logger.warning("Source bundle %d bytes exceeds safety limit, pruning", total_bytes)
                files = sorted(layout.parsed_dir.glob("*.md"), key=lambda item: item.stat().st_size)
                while total_bytes > MAX_SOURCE_BYTES and len(files) > 1:
                    dropped = files.pop(0)
                    total_bytes -= dropped.stat().st_size
                    dropped.unlink()
                    fetched_count -= 1

            result, quality, memory_result, export_path = self._run_bounded_research(
                topic,
                session_id,
                layout,
                run_kind="search_augmented",
                search_result_count=search_count,
                fetched_source_count=fetched_count,
            )
            return self._format_research_completion(topic, session_id, result, quality, memory_result, export_path)
        except (TimeoutError, ConnectionError) as exc:
            logger.exception("research_with_search timeout/connection error")
            return f"搜尋研究流程超時或連線失敗：{exc}"
        except Exception as exc:
            logger.exception("research_with_search failed")
            return f"搜尋研究流程失敗：{exc}"

    def _run_bounded_research(
        self,
        topic: str,
        session_id: str,
        layout: Any,
        run_kind: str = "telegram_bounded",
        search_result_count: int = 0,
        fetched_source_count: int = 0,
    ):
        result = self._orchestrator.synthesize(
            topic=topic,
            session_id=session_id,
            run_kind=run_kind,
            search_result_count=search_result_count,
            fetched_source_count=fetched_source_count,
        )
        quality = result.get("quality", {})
        note_path = result.get("note_path")
        if note_path:
            path = Path(note_path) if not isinstance(note_path, Path) else note_path
            if path.exists():
                content = path.read_text(encoding="utf-8", errors="replace")
                if not content.startswith("---"):
                    repaired = (
                        f"---\ntopic: \"{topic}\"\nsession_id: \"{session_id}\"\n"
                        f"provider: \"{result.get('provider', 'unknown')}\"\n---\n{content}"
                    )
                    path.write_text(repaired, encoding="utf-8")

        memory_result = None
        if quality.get("passed"):
            try:
                memory_result = self._orchestrator.memory_extract(
                    session_id=session_id,
                    task_type="research_session",
                    status="success",
                )
            except Exception:
                logger.exception("memory_extract failed for %s", session_id)

        export_path = None
        try:
            from ..integrations.obsidian import ObsidianExporter

            exporter = ObsidianExporter(self._config)
            if layout.note_path.exists():
                export_path = exporter.export_note(
                    session_id,
                    layout.note_path,
                    {"provider": result.get("provider"), "model": result.get("model")},
                )
        except Exception as exc:
            logger.warning("Obsidian export failed for %s: %s", session_id, exc)

        return result, quality, memory_result, export_path

    def _format_research_completion(
        self,
        topic: str,
        session_id: str,
        result: Dict[str, object],
        quality: Dict[str, object],
        memory_result: Optional[Dict[str, object]],
        export_path: Optional[Path],
    ) -> str:
        memory_line = f"\n記憶草稿：{memory_result.get('draft_id', '?')}" if memory_result else ""
        export_line = f"\nObsidian：{export_path.name}" if export_path else ""
        return (
            f"研究完成：{topic}\n"
            f"Session: {session_id}\n"
            f"Provider: {result.get('provider', '?')} / {result.get('model', '?')}\n"
            f"Words: {quality.get('word_count', '?')}\n"
            f"Coverage: {quality.get('coverage_score', '?')}\n"
            f"QG Pass: {quality.get('passed', '?')}"
            f"{memory_line}{export_line}"
        )

    def _format_report(self, report: Dict[str, object]) -> str:
        memory_status = report.get("memory_status", {})
        evaluation_status = report.get("evaluation_status", {})
        telemetry_summary = report.get("telemetry_summary", {})
        trace_summary = report.get("trace_summary", {})
        quality = report.get("quality", {})
        lines = [f"研究報告：{report.get('topic', report.get('session_id', 'unknown'))}"]
        lines.append(f"Session: {report.get('session_id')}")
        lines.append(f"Provider: {report.get('provider')} / {report.get('model')}")
        lines.append(f"來源數量: {report.get('source_count', 0)}")
        lines.append(f"Quality: {report.get('quality_score', 0)}")
        if isinstance(quality, dict):
            lines.append(
                "Coverage / Structure / Provenance: "
                f"{quality.get('coverage_score', 0)} / "
                f"{quality.get('structure_score', 0)} / "
                f"{quality.get('provenance_score', 0)}"
            )
        if isinstance(memory_status, dict):
            lines.append(
                f"Memory: approved={memory_status.get('approved_count', 0)}, "
                f"drafts={memory_status.get('draft_count', 0)}"
            )
        if isinstance(evaluation_status, dict) and evaluation_status.get("exists"):
            lines.append(
                f"Evaluation: {evaluation_status.get('eval_id')} "
                f"passed={evaluation_status.get('passed')}"
            )
        if isinstance(trace_summary, dict) and trace_summary.get("exists"):
            lines.append(f"Trace: {trace_summary.get('trace_id')} ({trace_summary.get('span_count')} spans)")
        if isinstance(telemetry_summary, dict):
            lines.append(f"Telemetry events: {telemetry_summary.get('count', 0)}")
        if report.get("note_path"):
            lines.append(f"Note: {report.get('note_path')}")
        return "\n".join(lines)

    def _handle_unknown(self, args: Dict[str, str], session_key: str) -> str:
        return "這個動作不允許在 Telegram 執行。"

    def _load_research_program(self) -> Dict[str, Any]:
        program_path = self._config.repo_root / "research_program.md"
        if not program_path.exists():
            return {}
        content = program_path.read_text(encoding="utf-8", errors="replace")
        if not content.startswith("---"):
            return {}
        lines = content.splitlines()
        try:
            end_index = lines[1:].index("---") + 1
        except ValueError:
            return {}
        payload: Dict[str, Any] = {}
        for line in lines[1:end_index]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            payload[key.strip()] = value.strip().strip("'\"")
        return payload


def _slug(text: str) -> str:
    return SLUG_RE.sub("-", text.lower()).strip("-")[:48]
