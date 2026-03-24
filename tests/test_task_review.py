"""Tests for TaskReviewService — outcome logging, event publishing, and post-task review."""

import json
import tempfile
import unittest
from pathlib import Path

from auto_research.events import EventNames
from auto_research.runtime import load_config

from conftest import FakeHttpClient, create_test_orchestrator, make_temp_repo


class TaskReviewTests(unittest.TestCase):

    def _setup(self, tmpdir, responses=None):
        repo = make_temp_repo(tmpdir)
        config = load_config(repo_root=repo, environ={"SKILL_MEMORY_EMBEDDING_PROVIDER": "disabled"})
        client = FakeHttpClient(responses=responses)
        orch = create_test_orchestrator(config, http_client=client)
        return orch, repo

    def _write_session(self, orch, session_id, topic="review topic"):
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

    def test_event_publishing_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            events_received = []
            orch.event_bus.subscribe(EventNames.TASK_OUTCOME, lambda e: events_received.append(e))
            orch._safe_record_outcome("test_task", "test_action", "success", "summary")
            self.assertEqual(len(events_received), 1)
            self.assertEqual(events_received[0]["task_id"], "test_task")
            self.assertEqual(events_received[0]["status"], "success")

    def test_event_publishing_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            events_received = []
            orch.event_bus.subscribe(EventNames.TASK_OUTCOME, lambda e: events_received.append(e))
            orch._safe_record_outcome("fail_task", "test_action", "failed", "it broke")
            self.assertEqual(len(events_received), 1)
            self.assertEqual(events_received[0]["status"], "failed")

    def test_post_task_review_success_creates_memory_draft(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            self._write_session(orch, "sess-review-ok", topic="success review")
            result = orch.post_task_review(session_id="sess-review-ok", status="success", action="research_session")
            self.assertIn("memory_draft", result)
            self.assertIn("memory_validation", result)
            self.assertIn("draft_id", result["memory_draft"])

    def test_post_task_review_failed_creates_gap_report_and_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Strategy advisor needs a fake LLM response for the advise call
            orch, repo = self._setup(tmpdir, responses=[{}])
            self._write_session(orch, "sess-review-fail", topic="failure review")
            result = orch.post_task_review(session_id="sess-review-fail", status="failed", action="research_session")
            self.assertIn("gap_report", result)
            self.assertIn("strategy", result)
            self.assertIn("skill_candidate", result)
            self.assertIn("skill_materialized", result)


if __name__ == "__main__":
    unittest.main()
