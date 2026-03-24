"""Data models for the Skill-as-Memory layer."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional


def _optional_str(value: object) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


@dataclass
class ProvenanceRecord:
    """Structured provenance for a single piece of evidence."""

    source_id: str
    source_uri: str
    source_type: str  # "web" | "github" | "local"
    extracted_at: str  # ISO timestamp or ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProvenanceRecord":
        return cls(
            source_id=str(payload.get("source_id", "")),
            source_uri=str(payload.get("source_uri", "")),
            source_type=str(payload.get("source_type", "local")),
            extracted_at=str(payload.get("extracted_at", "")),
        )

    @classmethod
    def from_legacy_string(cls, source_str: str) -> "ProvenanceRecord":
        """Convert a legacy evidence_sources string to ProvenanceRecord."""
        source_type = "local"
        if "github.com" in source_str:
            source_type = "github"
        elif source_str.startswith(("http://", "https://")):
            source_type = "web"
        return cls(
            source_id=source_str,
            source_uri=source_str,
            source_type=source_type,
            extracted_at="",
        )


@dataclass
class MemoryCitation:
    citation_type: str
    label: str
    path: Optional[str] = None
    uri: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value not in (None, "", [])}


@dataclass
class MemoryRecord:
    id: str
    title: str
    summary: str
    tags: List[str]
    task_type: str
    source_types: List[str]
    tool_deps: List[str]
    citations: List[MemoryCitation]
    confidence: float
    success_count: int
    failure_count: int
    risk_level: str
    last_validated_at: Optional[str]
    expires_at: str
    related_skills: List[str]
    obsidian_links: List[str]
    status: str
    session_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    approved_at: Optional[str] = None
    # Provenance: list of evidence source identifiers (paths, URIs, session IDs).
    evidence_sources: Optional[List[str]] = None
    # Structured provenance (v3.9+). Coexists with evidence_sources for backward compat.
    provenance: Optional[List[ProvenanceRecord]] = None

    def effective_provenance(self) -> List[ProvenanceRecord]:
        """Return provenance records, converting legacy evidence_sources if needed."""
        if self.provenance:
            return self.provenance
        if self.evidence_sources:
            return [ProvenanceRecord.from_legacy_string(s) for s in self.evidence_sources]
        return []

    @property
    def evidence_count(self) -> int:
        """Total number of evidence items from provenance."""
        return len(self.effective_provenance())

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["citations"] = [citation.to_dict() for citation in self.citations]
        if self.provenance:
            payload["provenance"] = [p.to_dict() for p in self.provenance]
        else:
            payload.pop("provenance", None)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryRecord":
        citations = [
            MemoryCitation(
                citation_type=str(item.get("citation_type", "artifact")),
                label=str(item.get("label", "")),
                path=item.get("path"),
                uri=item.get("uri"),
            )
            for item in payload.get("citations", [])
            if isinstance(item, Mapping)
        ]
        provenance_raw = payload.get("provenance")
        provenance = None
        if isinstance(provenance_raw, list):
            provenance = [
                ProvenanceRecord.from_dict(item)
                for item in provenance_raw
                if isinstance(item, Mapping)
            ]
        return cls(
            id=str(payload["id"]),
            title=str(payload["title"]),
            summary=str(payload["summary"]),
            tags=[str(item) for item in payload.get("tags", [])],
            task_type=str(payload.get("task_type", "research_session")),
            source_types=[str(item) for item in payload.get("source_types", [])],
            tool_deps=[str(item) for item in payload.get("tool_deps", [])],
            citations=citations,
            confidence=float(payload.get("confidence", 0.0)),
            success_count=int(payload.get("success_count", 0)),
            failure_count=int(payload.get("failure_count", 0)),
            risk_level=str(payload.get("risk_level", "low")),
            last_validated_at=_optional_str(payload.get("last_validated_at")),
            expires_at=str(payload["expires_at"]),
            related_skills=[str(item) for item in payload.get("related_skills", [])],
            obsidian_links=[str(item) for item in payload.get("obsidian_links", [])],
            status=str(payload.get("status", "draft")),
            session_id=_optional_str(payload.get("session_id")),
            created_at=_optional_str(payload.get("created_at")),
            updated_at=_optional_str(payload.get("updated_at")),
            approved_at=_optional_str(payload.get("approved_at")),
            evidence_sources=[str(s) for s in payload["evidence_sources"]] if payload.get("evidence_sources") else None,
            provenance=provenance,
        )
