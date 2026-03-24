"""Report service — session listing, selection, and metadata assembly."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from ..evaluation import EvaluationStore
from ..runtime import AutoResearchConfig

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class ReportService:
    """Extracts report-assembly helpers out of the Orchestrator."""

    def __init__(self, config: AutoResearchConfig) -> None:
        self.config = config

    # -- session listing -------------------------------------------------------

    def list_sessions(self) -> List[str]:
        session_ids: set[str] = set()
        research_root = self.config.repo_root / "output" / "research"
        if research_root.exists():
            for item in research_root.iterdir():
                if item.is_dir():
                    session_ids.add(item.name)
        notes_root = self.config.repo_root / "output" / "notes"
        if notes_root.exists():
            for item in notes_root.glob("*.md"):
                session_ids.add(item.stem)
        return sorted(session_ids)

    def select_session(self) -> Optional[str]:
        session_ids = self.list_sessions()
        if not session_ids:
            return None

        priority_topic = self.research_program_topic()
        if priority_topic:
            best_session = None
            best_score = 0.0
            best_mtime = -1.0
            for candidate in session_ids:
                layout = self.config.resolve_layout(candidate)
                metadata = self.read_note_metadata(layout.note_path)
                candidate_topic = (
                    str(metadata.get("topic", "")).strip()
                    or self.topic_from_session_id(candidate)
                )
                score = self.topic_match_score(priority_topic, candidate_topic)
                mtime = self.session_mtime(candidate)
                if score > best_score or (score == best_score and mtime > best_mtime):
                    best_session = candidate
                    best_score = score
                    best_mtime = mtime
            if best_session and best_score > 0:
                return best_session

        return max(session_ids, key=self.session_mtime)

    def session_mtime(self, session_id: str) -> float:
        layout = self.config.resolve_layout(session_id)
        candidates = [layout.note_path, layout.status_path]
        mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
        return max(mtimes) if mtimes else 0.0

    # -- topic helpers ---------------------------------------------------------

    def research_program_topic(self) -> str:
        program_path = self.config.repo_root / "research_program.md"
        if not program_path.exists():
            return ""
        text = program_path.read_text(encoding="utf-8", errors="replace")
        metadata = self.parse_frontmatter(text)
        priority_topic = str(metadata.get("priority_topic", "")).strip()
        if priority_topic:
            return priority_topic
        match = re.search(r"^## Current Priority\s*\n+(.+?)(?:\n## |\Z)", text, re.MULTILINE | re.DOTALL)
        if match:
            return " ".join(match.group(1).split())
        return ""

    @staticmethod
    def topic_match_score(target: str, candidate: str) -> float:
        target_norm = " ".join(target.lower().split())
        candidate_norm = " ".join(candidate.lower().split())
        if not target_norm or not candidate_norm:
            return 0.0
        if target_norm in candidate_norm or candidate_norm in target_norm:
            return 1.0
        target_tokens = set(re.findall(r"[a-z0-9]+", target_norm))
        candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate_norm))
        if not target_tokens or not candidate_tokens:
            return 0.0
        overlap = target_tokens & candidate_tokens
        return len(overlap) / max(len(target_tokens), len(candidate_tokens), 1)

    @staticmethod
    def topic_from_session_id(session_id: str) -> str:
        match = re.match(r"^\d{8}-\d{6}-(.+)$", session_id)
        if match:
            return match.group(1).replace("-", " ")
        if "-" in session_id:
            return session_id.split("-", 1)[1].replace("-", " ")
        return session_id

    # -- file I/O helpers ------------------------------------------------------

    @staticmethod
    def parse_frontmatter(content: str) -> Dict[str, str]:
        match = FRONTMATTER_RE.match(content)
        if not match:
            return {}
        metadata: Dict[str, str] = {}
        for raw_line in match.group(1).splitlines():
            if ":" not in raw_line:
                continue
            key, raw_value = raw_line.split(":", 1)
            metadata[key.strip()] = raw_value.strip().strip("'\"")
        return metadata

    @staticmethod
    def read_note_metadata(note_path: Path) -> Dict[str, str]:
        if not note_path.exists():
            return {}
        return ReportService.parse_frontmatter(note_path.read_text(encoding="utf-8", errors="replace"))

    @staticmethod
    def load_json_file(path: Path) -> Dict[str, object]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    # -- evaluation / memory / telemetry lookups -------------------------------

    def latest_evaluation(self, session_id: str) -> Dict[str, object]:
        store = EvaluationStore(self.config.repo_root / "knowledge" / "evaluations")
        records = []
        for record in store.load_all():
            metadata_session = str(record.metadata.get("session_id", "")) if isinstance(record.metadata, dict) else ""
            if metadata_session == session_id or record.eval_id.endswith(session_id):
                records.append(record)
        if not records:
            return {"exists": False}
        records.sort(key=lambda item: item.timestamp, reverse=True)
        latest = records[0]
        return {
            "exists": True,
            "eval_id": latest.eval_id,
            "eval_type": latest.eval_type,
            "passed": latest.passed,
            "candidate_score": latest.candidate_score,
            "timestamp": latest.timestamp,
            "metadata": latest.metadata,
        }

    def memory_status(self, session_id: str) -> Dict[str, object]:
        approved = 0
        drafts = 0
        for path in self.config.memory_records_dir.glob("*.json"):
            payload = self.load_json_file(path)
            if str(payload.get("session_id", "")) == session_id:
                approved += 1
        for path in self.config.memory_drafts_dir.glob("*.json"):
            payload = self.load_json_file(path)
            if str(payload.get("session_id", "")) == session_id:
                drafts += 1
        return {"approved_count": approved, "draft_count": drafts}

    def recent_telemetry(self, session_id: str, limit: int = 5) -> Dict[str, object]:
        telemetry_path = self.config.repo_root / "output" / "telemetry.jsonl"
        if not telemetry_path.exists():
            return {"count": 0, "events": []}
        events: List[Dict[str, object]] = []
        for raw_line in telemetry_path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            payload_session = str(payload.get("session_id", ""))
            if payload_session != session_id and session_id not in str(payload.get("result_summary", "")):
                continue
            events.append(
                {
                    "ts": payload.get("ts"),
                    "event_type": payload.get("event_type", payload.get("action", "event")),
                    "summary": payload.get("summary") or payload.get("result_summary", ""),
                    "status": payload.get("status"),
                }
            )
        return {"count": len(events), "events": events[-limit:]}

    def recent_trace(self, session_id: str) -> Dict[str, object]:
        telemetry_path = self.config.repo_root / "output" / "telemetry.jsonl"
        if not telemetry_path.exists():
            return {"exists": False}
        traces: List[Dict[str, object]] = []
        for raw_line in telemetry_path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or "trace_id" not in payload:
                continue
            spans = payload.get("spans", [])
            if not isinstance(spans, list):
                continue
            matches = False
            for span in spans:
                if not isinstance(span, dict):
                    continue
                attributes = span.get("attributes", {})
                if isinstance(attributes, dict) and (
                    str(attributes.get("session_key", "")) == session_id
                    or str(attributes.get("session_id", "")) == session_id
                ):
                    matches = True
                    break
            if matches:
                traces.append(payload)
        if not traces:
            return {"exists": False}
        latest = traces[-1]
        return {
            "exists": True,
            "trace_id": latest.get("trace_id"),
            "span_count": latest.get("span_count", 0),
        }

    @staticmethod
    def resolve_source_count(status_payload: Mapping[str, object], note_meta: Mapping[str, str]) -> int:
        if isinstance(status_payload.get("sources"), list):
            return len(status_payload.get("sources", []))
        raw = note_meta.get("sources_count", "")
        try:
            return int(str(raw))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def quality_score(quality: object) -> float:
        if not isinstance(quality, Mapping):
            return 0.0
        try:
            coverage = float(str(quality.get("coverage_score", 0.0)))
            structure = float(str(quality.get("structure_score", 0.0)))
            provenance = float(str(quality.get("provenance_score", 0.0)))
        except (TypeError, ValueError):
            return 0.0
        return round((coverage * 0.45) + (structure * 0.35) + (provenance * 0.20), 3)
