"""Tests for SkillMemoryService — extraction, validation, indexing, retrieval, and export."""

import json
import tempfile
import unittest
from pathlib import Path

from auto_research.orchestrator import PolicyError
from auto_research.runtime import load_config
from auto_research.services.skill_memory import (
    SkillMemoryService,
    MemoryRecord,
    MemoryCitation,
    SECRET_PATTERNS,
)

from conftest import FakeHttpClient, make_temp_repo, create_test_orchestrator


class SkillMemoryServiceTests(unittest.TestCase):
    """Integration tests that exercise SkillMemoryService through the orchestrator."""

    def _setup(self, tmpdir, http_responses=None):
        repo = make_temp_repo(tmpdir)
        config = load_config(repo_root=repo, environ={"SKILL_MEMORY_EMBEDDING_PROVIDER": "disabled"})
        client = FakeHttpClient(responses=http_responses)
        orch = create_test_orchestrator(config, http_client=client)
        return orch, repo

    def _write_session(self, orch, session_id, topic="test topic"):
        layout = orch.config.resolve_layout(session_id)
        layout.ensure()
        layout.status_path.write_text(
            json.dumps({
                "session_id": session_id,
                "sources": [{"topic": topic, "url": "https://example.com", "visibility": "public", "fetch_method": "scrapling"}],
            }),
            encoding="utf-8",
        )
        layout.note_path.write_text(
            f'---\ntopic: "{topic}"\n---\n# Summary\nThis note covers {topic}.\n',
            encoding="utf-8",
        )
        (layout.parsed_dir / "source.md").write_text(f"# Parsed\n{topic} reference\n", encoding="utf-8")

    # -- Extraction -----------------------------------------------------------

    def test_memory_extract_creates_draft(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            self._write_session(orch, "sess-1", topic="AI safety")
            result = orch.memory_extract("sess-1", task_type="research_session")
            self.assertIn("draft_id", result)
            self.assertIn("draft_path", result)
            draft_path = Path(result["draft_path"])
            self.assertTrue(draft_path.exists())
            payload = json.loads(draft_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "draft")
            self.assertIn("AI safety", payload["title"])
            self.assertEqual(payload["task_type"], "research_session")

    # -- Validation -----------------------------------------------------------

    def test_memory_validate_detects_secrets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            self._write_session(orch, "sess-secret", topic="ghp_1234567890abcdef token leak")
            draft = orch.memory_extract("sess-secret", task_type="research_session")
            result = orch.memory_validate(draft["draft_id"], approve=False)
            self.assertTrue(result["checks"]["has_secret_leak"])
            self.assertFalse(result["valid"])

    def test_memory_validate_approve_writes_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            self._write_session(orch, "sess-approve", topic="clean topic")
            draft = orch.memory_extract("sess-approve", task_type="research_session")
            result = orch.memory_validate(draft["draft_id"], approve=True)
            self.assertTrue(result["valid"])
            self.assertIn("approved_path", result)
            approved = json.loads(Path(result["approved_path"]).read_text(encoding="utf-8"))
            self.assertEqual(approved["status"], "approved")
            self.assertIsNotNone(approved["approved_at"])

    def test_memory_validate_reject_secret_blocks_approval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            self._write_session(orch, "sess-blocked", topic="ghp_1234567890abcdef token")
            draft = orch.memory_extract("sess-blocked", task_type="research_session")
            with self.assertRaises(PolicyError):
                orch.memory_validate(draft["draft_id"], approve=True)

    def test_memory_validate_approve_without_evidence_blocked(self):
        """Approval requires evidence_sources; draft validation without evidence is OK."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            self._write_session(orch, "sess-no-ev", topic="no evidence topic")
            draft = orch.memory_extract("sess-no-ev", task_type="research_session")
            # Manually strip evidence_sources from the draft
            draft_path = Path(draft["draft_path"])
            payload = json.loads(draft_path.read_text(encoding="utf-8"))
            payload["evidence_sources"] = None
            draft_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            # Draft validation should pass (evidence is warning-level for drafts)
            result = orch.memory_validate(draft["draft_id"], approve=False)
            self.assertTrue(result["valid"])

            # Approval should fail — evidence required
            with self.assertRaises(PolicyError) as ctx:
                orch.memory_validate(draft["draft_id"], approve=True)
            self.assertIn("evidence_sources", str(ctx.exception))

    # -- Indexing & Retrieval -------------------------------------------------

    def test_memory_index_rebuild_creates_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            self._write_session(orch, "sess-idx", topic="indexing test")
            draft = orch.memory_extract("sess-idx", task_type="research_session")
            orch.memory_validate(draft["draft_id"], approve=True)
            result = orch.memory_index_rebuild()
            self.assertTrue(Path(result["index_path"]).exists())
            self.assertEqual(result["memory_records"], 1)

    def test_retrieve_context_returns_hits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            self._write_session(orch, "sess-ret", topic="retrieval testing")
            draft = orch.memory_extract("sess-ret", task_type="research_session")
            orch.memory_validate(draft["draft_id"], approve=True)
            orch.memory_index_rebuild()
            result = orch.memory_search(task="retrieval testing")
            self.assertGreaterEqual(len(result["memory_hits"]), 1)
            self.assertIn("score", result["memory_hits"][0])

    # -- Skill Materialization ------------------------------------------------

    def test_skill_materialize_creates_scaffolding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            orch.evo_log("task-mat", "failed", "test failure")
            orch.evo_propose("task-mat", "test-skill", "Fix fetch.")
            result = orch.skill_materialize("test-skill")
            self.assertTrue(Path(result["metadata_path"]).exists())
            self.assertTrue(Path(result["citations_path"]).exists())
            metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
            self.assertIn("title", metadata)
            self.assertIn("tags", metadata)

    # -- Skill Export ---------------------------------------------------------

    def test_skill_export_to_github(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            skill_dir = repo / "skills" / "exported-skill"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text("---\nname: Exported\n---\n# Exported\n", encoding="utf-8")
            result = orch.skill_export(target="github")
            self.assertIn("exported-skill", result["exported_skills"])
            target_path = repo / ".github" / "skills" / "exported-skill" / "SKILL.md"
            self.assertTrue(target_path.exists())


class SkillMemoryUnitTests(unittest.TestCase):
    """Unit tests for internal helper functions."""

    def _make_service(self, tmpdir):
        repo = make_temp_repo(tmpdir)
        config = load_config(repo_root=repo, environ={"SKILL_MEMORY_EMBEDDING_PROVIDER": "disabled"})
        from auto_research.services.llm_provider import LlmProviderService
        return SkillMemoryService(config, FakeHttpClient(), LlmProviderService(config))

    def test_contains_secrets_github_pat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            self.assertTrue(svc._contains_secrets("github_pat_1234567890abcdef"))
            self.assertTrue(svc._contains_secrets("ghp_1234567890abcdef"))
            self.assertTrue(svc._contains_secrets("Bearer eyJhbGciOiJIUzI"))
            self.assertTrue(svc._contains_secrets("AIzaSyD1234567890abcdef01234"))
            self.assertFalse(svc._contains_secrets("just a regular sentence"))

    def test_lexical_score_computation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            score = svc._lexical_score("python async", ["python", "web", "async", "server"])
            self.assertGreater(score, 0)
            zero_score = svc._lexical_score("python", ["java", "rust"])
            self.assertEqual(zero_score, 0.0)

    def test_cosine_similarity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            self.assertAlmostEqual(svc._cosine_similarity([1, 0], [1, 0]), 1.0)
            self.assertAlmostEqual(svc._cosine_similarity([1, 0], [0, 1]), 0.0)
            self.assertAlmostEqual(svc._cosine_similarity([1, 0], [-1, 0]), -1.0)
            self.assertEqual(svc._cosine_similarity([], []), 0.0)
            self.assertEqual(svc._cosine_similarity([1], [1, 2]), 0.0)

    def test_strip_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            result = svc._strip_frontmatter("---\ntitle: test\n---\n# Body")
            self.assertEqual(result, "# Body")
            result_no_fm = svc._strip_frontmatter("No frontmatter here")
            self.assertEqual(result_no_fm, "No frontmatter here")

    def test_tokenize_and_dedupe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            tokens = svc._tokenize("Hello World, hello WORLD again!")
            self.assertIn("hello", tokens)
            self.assertIn("world", tokens)
            deduped = svc._dedupe(["A", "a", "B", "b", "C"])
            self.assertEqual(deduped, ["A", "B", "C"])


class MemoryRecordModelTests(unittest.TestCase):
    """Tests for MemoryRecord and MemoryCitation data models."""

    def test_memory_record_roundtrip(self):
        record = MemoryRecord(
            id="test-1", title="Test", summary="A test record",
            tags=["a"], task_type="research_session", source_types=["web"],
            tool_deps=["scrapling"], citations=[MemoryCitation("artifact", "note", path="/tmp/note.md")],
            confidence=0.9, success_count=1, failure_count=0,
            risk_level="low", last_validated_at=None, expires_at="2099-01-01T00:00:00+00:00",
            related_skills=[], obsidian_links=[], status="draft",
        )
        payload = record.to_dict()
        restored = MemoryRecord.from_dict(payload)
        self.assertEqual(restored.id, record.id)
        self.assertEqual(restored.title, record.title)
        self.assertEqual(len(restored.citations), 1)

    def test_memory_citation_to_dict_omits_none(self):
        citation = MemoryCitation("artifact", "label", path="/tmp/x")
        d = citation.to_dict()
        self.assertNotIn("uri", d)
        self.assertIn("path", d)


if __name__ == "__main__":
    unittest.main()
