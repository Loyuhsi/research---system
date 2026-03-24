"""Validation logic for the Skill-as-Memory layer."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Sequence

from ...exceptions import ExecutionError, PolicyError
from ...runtime import AutoResearchConfig
from .models import MemoryCitation, MemoryRecord
from .util import contains_secrets


def memory_validate(
    config: AutoResearchConfig,
    memory_id: str,
    approve: bool = False,
    rebuild_index_fn=None,
) -> Dict[str, object]:

    record, record_path = _load_record_for_validation(config, memory_id)
    has_evidence = bool(record.evidence_sources)
    citation_results = _validate_citations(record.citations)
    validations = {
        "has_secret_leak": contains_secrets(record.summary) or contains_secrets(record.title),
        "citations": citation_results,
        "not_expired": _not_expired(record.expires_at),
        "has_evidence": has_evidence,
    }
    # Base validity: no secret leak, not expired, citations OK
    valid = (not validations["has_secret_leak"]) and validations["not_expired"] and all(
        item["ok"] for item in citation_results
    )
    # Evidence is blocking for approval only — drafts can pass without evidence
    if approve and not has_evidence:
        valid = False
    validation_result = {
        "memory_id": memory_id,
        "valid": valid,
        "checks": validations,
        "record_path": str(record_path),
    }
    validation_path = record_path.with_name("validation.json")
    validation_path.write_text(json.dumps(validation_result, ensure_ascii=False, indent=2), encoding="utf-8")

    if approve:
        from ...tracing import current_trace, SpanKind
        trace = current_trace()
        approval_span = None
        if trace:
            approval_span = trace.start_span(
                "memory.approve", SpanKind.APPROVAL,
                attributes={"memory_id": memory_id, "valid": valid},
            )
        if not valid:
            reason = "Memory record failed validation and cannot be approved."
            if not has_evidence:
                reason = "Approval requires evidence_sources. Add evidence before approving."
            if approval_span:
                approval_span.finish(status="error", error=reason)
            raise PolicyError(reason)
        approved_path = config.memory_records_dir / f"{memory_id}.json"
        config.memory_records_dir.mkdir(parents=True, exist_ok=True)
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        record.status = "approved"
        record.last_validated_at = now
        record.approved_at = now
        record.updated_at = now
        approved_path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if rebuild_index_fn:
            rebuild_index_fn()
        validation_result["approved_path"] = str(approved_path)
        if approval_span:
            approval_span.finish(status="ok")
    elif not valid and record_path.parent == config.memory_records_dir:
        raise ExecutionError(f"Approved memory record {memory_id} is no longer valid.")

    return validation_result


def _load_record_for_validation(config: AutoResearchConfig, memory_id: str) -> tuple[MemoryRecord, Path]:
    draft_path = config.memory_drafts_dir / memory_id / "memory-record.json"
    approved_path = config.memory_records_dir / f"{memory_id}.json"
    path = draft_path if draft_path.exists() else approved_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    return MemoryRecord.from_dict(payload), path


def _validate_citations(citations: Sequence[MemoryCitation]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for citation in citations:
        if citation.path:
            path = Path(citation.path)
            results.append({"label": citation.label, "ok": path.exists(), "path": citation.path})
        elif citation.uri:
            results.append({"label": citation.label, "ok": citation.uri.startswith("http"), "uri": citation.uri})
        else:
            results.append({"label": citation.label, "ok": False})
    return results


def _not_expired(expires_at: str) -> bool:
    try:
        expiry = dt.datetime.fromisoformat(expires_at)
    except ValueError:
        return False
    now = dt.datetime.now(expiry.tzinfo or dt.timezone.utc)
    return expiry >= now
