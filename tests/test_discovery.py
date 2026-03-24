"""Tests for the discovery layer: TopicExpander + SourceRanker."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from auto_research.discovery.source_ranker import SourceRanker, SourceScore
from auto_research.discovery.topic_expander import TopicExpander
from auto_research.runtime import load_config
from auto_research.services.llm_provider import LlmProviderService
from conftest import FakeHttpClient, make_temp_repo


def _write_note(notes_dir: Path, name: str, topic: str, body: str = "") -> None:
    """Write a minimal research note with frontmatter."""
    notes_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\ntopic: \"{topic}\"\n---\n\n{body}\n"
    (notes_dir / name).write_text(content, encoding="utf-8")


class TopicExpanderTests(unittest.TestCase):
    def _make_expander(self, responses=None, notes=None):
        tmpdir = tempfile.mkdtemp()
        repo = make_temp_repo(tmpdir)
        config = load_config(repo_root=repo, environ={})
        http = FakeHttpClient(responses=responses or [])
        llm = LlmProviderService(config)
        if notes:
            notes_dir = repo / "output" / "notes"
            for name, topic in notes:
                _write_note(notes_dir, name, topic)
        return TopicExpander(config, http, llm)

    def test_no_notes_returns_rule_based(self):
        expander = self._make_expander()
        result = expander.discover()
        self.assertEqual(result["source"], "rule-based")
        self.assertEqual(result["existing_topics"], [])

    def test_llm_success_returns_suggestions(self):
        llm_resp = {"choices": [{"message": {"content": "1. Topic A\n2. Topic B"}}]}
        expander = self._make_expander(
            responses=[llm_resp],
            notes=[("note1.md", "AI Safety"), ("note2.md", "LLM Optimization")],
        )
        result = expander.discover()
        self.assertEqual(result["source"], "llm")
        self.assertEqual(len(result["existing_topics"]), 2)

    def test_llm_failure_fallback(self):
        expander = self._make_expander(
            responses=[{}],  # empty = no content
            notes=[("note1.md", "Testing")],
        )
        result = expander.discover()
        self.assertEqual(result["source"], "rule-based")
        self.assertIn("1 existing topics", result["suggestions"][0])

    def test_dedup_existing_topics(self):
        expander = self._make_expander(
            responses=[{}],
            notes=[("note1.md", "Same Topic"), ("note2.md", "Same Topic")],
        )
        result = expander.discover()
        self.assertEqual(len(result["existing_topics"]), 1)


class SourceRankerTests(unittest.TestCase):
    def _make_ranker(self, session_id="test-session"):
        tmpdir = tempfile.mkdtemp()
        repo = make_temp_repo(tmpdir)
        config = load_config(repo_root=repo, environ={})
        return SourceRanker(config), config, session_id

    def test_empty_session_no_sources(self):
        ranker, config, sid = self._make_ranker()
        result = ranker.rank_session(sid)
        self.assertEqual(result["sources"], [])

    def test_below_min_words_low_score(self):
        ranker, config, sid = self._make_ranker()
        layout = config.resolve_layout(sid)
        layout.parsed_dir.mkdir(parents=True, exist_ok=True)
        (layout.parsed_dir / "tiny.md").write_text("short", encoding="utf-8")
        result = ranker.rank_session(sid)
        self.assertEqual(result["total_sources"], 1)
        self.assertEqual(result["usable_sources"], 0)
        self.assertAlmostEqual(result["sources"][0]["quality_score"], 0.0)

    def test_well_structured_file_scores_high(self):
        ranker, config, sid = self._make_ranker()
        layout = config.resolve_layout(sid)
        layout.parsed_dir.mkdir(parents=True, exist_ok=True)
        good_content = "# Main Heading\n\n" + ("This is a well written paragraph with content. " * 80) + "\n\n## Sub Heading\n\n" + ("More detailed content here with [links](http://example.com). " * 40)
        (layout.parsed_dir / "good.md").write_text(good_content, encoding="utf-8")
        result = ranker.rank_session(sid)
        score = result["sources"][0]["quality_score"]
        self.assertGreater(score, 0.4)

    def test_empty_file_zero_score(self):
        ranker, config, sid = self._make_ranker()
        layout = config.resolve_layout(sid)
        layout.parsed_dir.mkdir(parents=True, exist_ok=True)
        (layout.parsed_dir / "empty.md").write_text("", encoding="utf-8")
        result = ranker.rank_session(sid)
        self.assertAlmostEqual(result["sources"][0]["quality_score"], 0.0)

    def test_ranking_order_descending(self):
        ranker, config, sid = self._make_ranker()
        layout = config.resolve_layout(sid)
        layout.parsed_dir.mkdir(parents=True, exist_ok=True)
        # Write a good file and a bad file
        good = "# Heading\n\n" + ("word " * 500) + "\n\n## Another\n\n" + ("detail " * 200)
        bad = "tiny content"
        (layout.parsed_dir / "good.md").write_text(good, encoding="utf-8")
        (layout.parsed_dir / "bad.md").write_text(bad, encoding="utf-8")
        result = ranker.rank_session(sid)
        scores = [s["quality_score"] for s in result["sources"]]
        self.assertEqual(scores, sorted(scores, reverse=True))


class SourceScoreTests(unittest.TestCase):
    def test_to_dict(self):
        s = SourceScore(filename="test.md", size_bytes=100, has_content=True,
                        word_count=50, quality_score=0.7777)
        d = s.to_dict()
        self.assertEqual(d["filename"], "test.md")
        self.assertEqual(d["quality_score"], 0.778)


class TopicExpanderFrontmatterTests(unittest.TestCase):
    def test_frontmatter_extraction(self):
        tmpdir = tempfile.mkdtemp()
        repo = make_temp_repo(tmpdir)
        config = load_config(repo_root=repo, environ={})
        http = FakeHttpClient()
        llm = LlmProviderService(config)
        expander = TopicExpander(config, http, llm)

        content = '---\ntopic: "My Research"\ncreated: "2026-01-01"\n---\n\nBody text'
        result = expander._frontmatter_map(content)
        self.assertEqual(result["topic"], '"My Research"')

    def test_no_frontmatter(self):
        tmpdir = tempfile.mkdtemp()
        repo = make_temp_repo(tmpdir)
        config = load_config(repo_root=repo, environ={})
        http = FakeHttpClient()
        llm = LlmProviderService(config)
        expander = TopicExpander(config, http, llm)

        result = expander._frontmatter_map("No frontmatter here")
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
