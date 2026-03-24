"""E2E: Memory lifecycle (extract -> validate -> approve -> index -> search)."""

from __future__ import annotations

import json
import time

import pytest

from .conftest import skip_no_ollama


@pytest.mark.e2e
@skip_no_ollama
class TestE2EMemory:
    def test_full_memory_lifecycle(self, e2e_orchestrator, e2e_config):
        """Extract -> validate -> approve -> rebuild index -> search."""
        session_id = "e2e-mem-001"
        layout = e2e_config.resolve_layout(session_id)

        # Setup: write parsed sources + status + note
        layout.parsed_dir.mkdir(parents=True, exist_ok=True)
        (layout.parsed_dir / "source.md").write_text(
            "# Memory Testing\n\nThis is a test source for memory extraction.\n",
            encoding="utf-8",
        )
        layout.status_path.parent.mkdir(parents=True, exist_ok=True)
        layout.status_path.write_text(json.dumps({
            "topic": "Memory Testing",
            "sources": [{"url": "https://example.com/memory", "topic": "Memory Testing"}],
        }), encoding="utf-8")
        layout.note_path.parent.mkdir(parents=True, exist_ok=True)
        layout.note_path.write_text(
            "# Memory Testing Note\n\nExtracted from memory test session.\n",
            encoding="utf-8",
        )

        t0 = time.monotonic()

        # 1. Extract
        extract_result = e2e_orchestrator.memory_extract(
            session_id=session_id,
            task_type="research_session",
            status="success",
        )
        draft_id = extract_result["draft_id"]
        print(f"  extract: draft_id={draft_id}")

        # 2. Validate (without approval)
        val_result = e2e_orchestrator.memory_validate(memory_id=draft_id, approve=False)
        assert val_result["valid"] is True, f"Validation failed: {val_result}"

        # 3. Approve
        approve_result = e2e_orchestrator.memory_validate(memory_id=draft_id, approve=True)
        assert "approved_path" in approve_result

        # 4. Rebuild index
        index_result = e2e_orchestrator.memory_index_rebuild()
        assert index_result["memory_records"] >= 1
        print(f"  index: {index_result['memory_records']} records, backend={index_result['vector_backend']}")

        # 5. Search
        search_result = e2e_orchestrator.memory_search(task="Memory Testing")
        hits = search_result.get("memory_hits", [])
        assert len(hits) >= 1, f"Expected at least 1 search hit, got {len(hits)}"
        print(f"  search: {len(hits)} hits, top score={hits[0]['score']}")

        elapsed = time.monotonic() - t0
        print(f"E2E memory lifecycle: {elapsed:.2f}s")
