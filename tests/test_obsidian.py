"""Tests for Obsidian vault integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_temp_repo
from auto_research.runtime import load_config
from auto_research.integrations.obsidian import (
    ObsidianExporter,
    ObsidianUri,
    _slug,
    _strip_frontmatter,
)


@pytest.fixture
def exporter(tmp_path: Path):
    repo = make_temp_repo(str(tmp_path / "repo"))
    config = load_config(repo_root=repo, environ={"VAULT_ROOT": str(tmp_path / "vault")})
    return ObsidianExporter(config)


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    return tmp_path / "vault"


class TestSlugAndFrontmatter:
    def test_slug_basic(self):
        assert _slug("Hello World") == "hello-world"

    def test_slug_special_chars(self):
        assert _slug("session-001 (test)") == "session-001-test"

    def test_slug_truncation(self):
        assert len(_slug("x" * 200)) <= 80

    def test_strip_frontmatter(self):
        content = "---\ntopic: test\n---\n# Body\nContent here."
        result = _strip_frontmatter(content)
        assert "# Body" in result
        assert "Content here." in result

    def test_strip_frontmatter_no_frontmatter(self):
        content = "# No Frontmatter\nJust body."
        assert _strip_frontmatter(content) == content

    def test_render_frontmatter_basic(self, exporter):
        fm = exporter._render_frontmatter({
            "id": "test-001",
            "type": "research_note",
            "tags": ["python", "testing"],
            "score": 0.85,
        })
        assert "---" in fm
        assert 'id: "test-001"' in fm
        assert 'type: "research_note"' in fm
        assert "tags:" in fm
        assert "score: 0.85" in fm

    def test_render_frontmatter_escapes_quotes(self, exporter):
        fm = exporter._render_frontmatter({"title": 'He said "hello"'})
        assert '\\"hello\\"' in fm

    def test_render_frontmatter_skips_none(self, exporter):
        fm = exporter._render_frontmatter({"id": "x", "empty": None})
        assert "empty" not in fm


class TestWikilinks:
    def test_wikilink_with_display_title(self, exporter):
        related = [("memory_record", "session-001-research", "Memory: Python Testing")]
        result = exporter._build_wikilinks("research_note", related)
        assert "## Related" in result
        assert "[[20_Memory/session-001-research|Memory: Python Testing]]" in result

    def test_wikilink_empty_related(self, exporter):
        assert exporter._build_wikilinks("research_note", []) == ""

    def test_wikilink_string_fallback(self, exporter):
        result = exporter._build_wikilinks("research_note", ["some-link"])
        assert "[[some-link]]" in result


class TestExporter:
    def test_export_note(self, exporter, vault_root, tmp_path):
        note = tmp_path / "note.md"
        note.write_text("---\ntopic: test\n---\n# Test Note\nContent.", encoding="utf-8")
        out = exporter.export_note("session-001", note, {"provider": "lmstudio"})
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert 'type: "research_note"' in content
        assert "artifact_version:" in content
        assert "# Test Note" in content

    def test_export_memory(self, exporter, vault_root):
        record = {
            "id": "mem-001",
            "title": "Python Testing Knowledge",
            "summary": "Pytest supports fixtures.",
            "tags": ["python"],
            "task_type": "research_session",
            "status": "approved",
            "evidence_sources": ["https://docs.pytest.org"],
            "citations": [{"label": "note", "path": "/tmp/note.md"}],
            "confidence": 0.8,
            "session_id": "sess-001",
            "created_at": "2026-03-20T00:00:00Z",
        }
        out = exporter.export_memory(record)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert 'type: "memory_record"' in content
        assert "## Summary" in content
        assert "## Citations" in content
        assert "## Related" in content

    def test_export_evaluation(self, exporter, vault_root):
        eval_data = {
            "eval_id": "qg-001",
            "eval_type": "quality_gate",
            "baseline_score": 0.5,
            "candidate_score": 0.72,
            "passed": True,
            "timestamp": "2026-03-20T00:00:00Z",
            "artifact_paths": ["/tmp/note.md"],
        }
        out = exporter.export_evaluation(eval_data)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert 'type: "evaluation"' in content
        assert "## Results" in content
        assert "0.72" in content

    def test_export_diagnostics(self, exporter, vault_root):
        doctor = {
            "provider": "lmstudio",
            "model": "nemotron-nano",
            "services": {"ollama": {"ok": True}, "lmstudio": {"ok": True}, "vllm": {"ok": False}},
            "provider_matrix": {
                "lmstudio": {"health_ready": True, "inference_ready": True, "embedding_ready": True, "is_primary": True},
            },
        }
        out = exporter.export_diagnostics(doctor)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert 'type: "diagnostics"' in content
        assert "## Provider Status" in content

    def test_vault_dirs_created(self, exporter, vault_root):
        exporter._ensure_vault_dirs()
        assert (vault_root / "20_Memory").is_dir()
        assert (vault_root / "30_Evaluations").is_dir()
        assert (vault_root / "40_Diagnostics").is_dir()

    def test_existing_vault_paths_preserved(self, exporter, vault_root):
        (vault_root / "00_Inbox").mkdir(parents=True, exist_ok=True)
        (vault_root / "90_Logs").mkdir(parents=True, exist_ok=True)
        exporter._ensure_vault_dirs()
        assert (vault_root / "00_Inbox").is_dir()
        assert (vault_root / "90_Logs").is_dir()


class TestObsidianUri:
    def test_open_note(self):
        uri = ObsidianUri.open_note("MyVault", "20_Memory/test-note")
        assert uri.startswith("obsidian://open?vault=")
        assert "MyVault" in uri
        assert "20_Memory" in uri

    def test_search(self):
        uri = ObsidianUri.search("MyVault", "python testing")
        assert uri.startswith("obsidian://search?vault=")
        assert "python" in uri
