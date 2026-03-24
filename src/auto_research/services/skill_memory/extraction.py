"""Draft extraction logic for the Skill-as-Memory layer."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from ...runtime import AutoResearchConfig
from .models import MemoryCitation, MemoryRecord, ProvenanceRecord
from .util import dedupe, strip_frontmatter, tokenize


def memory_extract(
    config: AutoResearchConfig,
    session_id: str,
    task_type: str = "research_session",
    status: str = "success",
    summary_override: Optional[str] = None,
) -> Dict[str, object]:
    layout = config.resolve_layout(session_id)
    status_payload = _load_session_status(layout)
    note_summary = summary_override or _summarize_note(layout.note_path)
    created_at = dt.datetime.now(dt.timezone.utc)
    record_id = _build_record_id(session_id, task_type)
    record = MemoryRecord(
        id=record_id,
        title=_build_title(session_id, task_type, status_payload),
        summary=note_summary or f"{task_type} session {session_id} completed.",
        tags=_extract_tags(status_payload, note_summary),
        task_type=task_type,
        source_types=_extract_source_types(status_payload),
        tool_deps=_extract_tool_deps(task_type, status_payload, config.provider),
        citations=_build_citations(layout, status_payload),
        confidence=0.8 if status == "success" else 0.35,
        success_count=1 if status == "success" else 0,
        failure_count=0 if status == "success" else 1,
        risk_level=_risk_level_for_task(task_type),
        last_validated_at=None,
        expires_at=(created_at + dt.timedelta(days=config.skill_memory_ttl_days)).isoformat(),
        related_skills=[],
        obsidian_links=[str(status_payload["promoted_to"])] if status_payload.get("promoted_to") else [],
        status="draft",
        session_id=session_id,
        created_at=created_at.isoformat(),
        updated_at=created_at.isoformat(),
        evidence_sources=_extract_evidence_sources(layout, status_payload),
        provenance=_extract_provenance(layout, status_payload, created_at.isoformat()),
    )
    draft_dir = config.memory_drafts_dir / record_id
    draft_dir.mkdir(parents=True, exist_ok=True)
    draft_path = draft_dir / "memory-record.json"
    draft_path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"draft_id": record_id, "draft_path": str(draft_path)}


def _load_session_status(layout) -> Dict[str, object]:
    if layout.status_path.exists():
        data = json.loads(layout.status_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    return {}


def _summarize_note(note_path: Path) -> str:
    if not note_path.exists():
        return ""
    content = note_path.read_text(encoding="utf-8", errors="replace")
    body = strip_frontmatter(content)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return ""
    return " ".join(lines[:4])[:500]


def _build_record_id(session_id: str, task_type: str) -> str:
    digest = hashlib.sha1(f"{session_id}:{task_type}".encode("utf-8")).hexdigest()[:10]
    return f"{session_id}-{task_type}-{digest}"


def _build_title(session_id: str, task_type: str, status_payload: Mapping[str, object]) -> str:
    topic = _extract_topic(status_payload)
    if topic:
        return f"{task_type}: {topic}"
    return f"{task_type}: {session_id}"


def _extract_topic(status_payload: Mapping[str, object]) -> str:
    sources = status_payload.get("sources")
    if isinstance(sources, list):
        for item in sources:
            if isinstance(item, Mapping) and item.get("topic"):
                return str(item["topic"])
    return ""


def _extract_tags(status_payload: Mapping[str, object], summary: Optional[str]) -> List[str]:
    tags: List[str] = []
    topic = _extract_topic(status_payload)
    if topic:
        tags.extend(tokenize(topic))
    if summary:
        tags.extend(tokenize(summary)[:6])
    sources = status_payload.get("sources")
    if isinstance(sources, list):
        for item in sources[:4]:
            if isinstance(item, Mapping):
                url = str(item.get("url", ""))
                if "github.com" in url:
                    tags.append("github")
                elif "http" in url:
                    tags.append("web")
                visibility = item.get("visibility")
                if visibility:
                    tags.append(str(visibility))
    return dedupe(tags)


def _extract_source_types(status_payload: Mapping[str, object]) -> List[str]:
    source_types: List[str] = []
    sources = status_payload.get("sources")
    if isinstance(sources, list):
        for item in sources:
            if isinstance(item, Mapping):
                visibility = item.get("visibility")
                if visibility:
                    source_types.append(str(visibility))
                url = str(item.get("url", ""))
                if "github.com" in url:
                    source_types.append("github")
                elif url:
                    source_types.append("web")
    return dedupe(source_types)


def _extract_tool_deps(task_type: str, status_payload: Mapping[str, object], provider: str) -> List[str]:
    tools: List[str] = []
    if task_type in {"fetch_public", "research_session"}:
        tools.append("scrapling")
    if task_type == "fetch_private":
        tools.append("gh")
    if task_type == "synthesize":
        tools.append(provider)
    if task_type == "tool_run":
        tools.append("cli-anything")
    if task_type == "rd_agent":
        tools.append("rd-agent")
    sources = status_payload.get("sources")
    if isinstance(sources, list):
        for item in sources:
            if isinstance(item, Mapping) and item.get("fetch_method"):
                tools.append(str(item["fetch_method"]))
    return dedupe(tools)


def _build_citations(layout, status_payload: Mapping[str, object]) -> List[MemoryCitation]:
    citations: List[MemoryCitation] = [MemoryCitation("artifact", "research status", path=str(layout.status_path))]
    if layout.note_path.exists():
        citations.append(MemoryCitation("artifact", "research note", path=str(layout.note_path)))
    for md_path in sorted(layout.parsed_dir.glob("*.md"))[:3]:
        citations.append(MemoryCitation("artifact", md_path.name, path=str(md_path)))
    promoted_to = status_payload.get("promoted_to")
    if promoted_to:
        citations.append(MemoryCitation("obsidian", "vault note", path=str(promoted_to)))
    return citations


def _extract_evidence_sources(layout, status_payload: Mapping[str, object]) -> List[str]:
    """Collect evidence source identifiers for provenance tracking."""
    evidence: List[str] = []
    # Source URLs from status
    sources = status_payload.get("sources")
    if isinstance(sources, list):
        for item in sources:
            if isinstance(item, Mapping):
                url = str(item.get("url", ""))
                if url:
                    evidence.append(url)
    # Parsed source file paths
    if layout.parsed_dir.exists():
        for md_path in sorted(layout.parsed_dir.glob("*.md"))[:10]:
            evidence.append(str(md_path))
    # Note path
    if layout.note_path.exists():
        evidence.append(str(layout.note_path))
    return dedupe(evidence)


def _extract_provenance(
    layout: object, status_payload: Mapping[str, object], created_at: str,
) -> List[ProvenanceRecord]:
    """Build structured provenance records from session data."""
    records: List[ProvenanceRecord] = []
    sources = status_payload.get("sources")
    if isinstance(sources, list):
        for item in sources:
            if isinstance(item, Mapping):
                url = str(item.get("url", ""))
                if url:
                    source_type = "github" if "github.com" in url else "web"
                    records.append(ProvenanceRecord(
                        source_id=url,
                        source_uri=url,
                        source_type=source_type,
                        extracted_at=created_at,
                    ))
    parsed_dir = getattr(layout, "parsed_dir", None)
    if parsed_dir is not None and parsed_dir.exists():
        for md_path in sorted(parsed_dir.glob("*.md"))[:10]:
            records.append(ProvenanceRecord(
                source_id=md_path.name,
                source_uri=str(md_path),
                source_type="local",
                extracted_at=created_at,
            ))
    note_path = getattr(layout, "note_path", None)
    if note_path is not None and note_path.exists():
        records.append(ProvenanceRecord(
            source_id="note",
            source_uri=str(note_path),
            source_type="local",
            extracted_at=created_at,
        ))
    return records


def _risk_level_for_task(task_type: str) -> str:
    if task_type == "rd_agent":
        return "high"
    if task_type == "tool_run":
        return "medium"
    return "low"
