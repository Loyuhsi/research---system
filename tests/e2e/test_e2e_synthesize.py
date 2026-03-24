"""E2E: Real synthesize + quality gate tests."""

from __future__ import annotations

import json
import time

import pytest

from .conftest import skip_no_ollama


@pytest.mark.e2e
@skip_no_ollama
class TestE2ESynthesize:
    def test_synthesize_produces_note(self, e2e_orchestrator, e2e_config):
        """Real Ollama synthesis creates a note file."""
        session_id = "e2e-synth-001"
        layout = e2e_config.resolve_layout(session_id)
        layout.parsed_dir.mkdir(parents=True, exist_ok=True)
        # Write minimal source file
        source_md = layout.parsed_dir / "source1.md"
        source_md.write_text(
            "# Python Testing\n\nPytest is a testing framework for Python.\n"
            "It supports fixtures, parametrize, and markers.\n"
            "Unit tests verify individual functions. Integration tests check components together.\n"
            "Coverage measures how much code is exercised by tests.\n",
            encoding="utf-8",
        )
        # Write status
        status_path = layout.status_path
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps({
            "topic": "Python Testing",
            "sources": [{"url": "https://docs.pytest.org", "topic": "Python Testing"}],
        }), encoding="utf-8")

        t0 = time.monotonic()
        result = e2e_orchestrator.synthesize(
            topic="Python Testing",
            session_id=session_id,
        )
        elapsed = time.monotonic() - t0
        assert result["note_path"]
        print(f"E2E synthesize: {elapsed:.2f}s")
        # Quality gate should have run
        if "quality" in result:
            q = result["quality"]
            print(f"  coverage={q.get('coverage_score')}, structure={q.get('structure_score')}, words={q.get('word_count')}")
