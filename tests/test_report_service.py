"""Tests for the extracted ReportService."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_temp_repo
from auto_research.runtime import load_config
from auto_research.services.report import ReportService


@pytest.fixture
def report_svc(tmp_path):
    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})
    return ReportService(config)


class TestSessionListing:
    def test_no_sessions(self, report_svc):
        assert report_svc.list_sessions() == []

    def test_lists_research_dir_sessions(self, report_svc):
        research = report_svc.config.repo_root / "output" / "research" / "sess-abc"
        research.mkdir(parents=True)
        assert "sess-abc" in report_svc.list_sessions()

    def test_lists_note_sessions(self, report_svc):
        notes = report_svc.config.repo_root / "output" / "notes"
        notes.mkdir(parents=True, exist_ok=True)
        (notes / "sess-xyz.md").write_text("# Note", encoding="utf-8")
        assert "sess-xyz" in report_svc.list_sessions()

    def test_select_session_returns_none_when_empty(self, report_svc):
        assert report_svc.select_session() is None


class TestTopicHelpers:
    def test_topic_from_session_id_with_timestamp(self):
        assert ReportService.topic_from_session_id("20260101-120000-my-topic") == "my topic"

    def test_topic_from_session_id_with_dash(self):
        assert ReportService.topic_from_session_id("prefix-some-topic") == "some topic"

    def test_topic_from_session_id_no_dash(self):
        assert ReportService.topic_from_session_id("singletopic") == "singletopic"

    def test_topic_match_exact_substring(self):
        assert ReportService.topic_match_score("AI safety", "research on AI safety") == 1.0

    def test_topic_match_partial(self):
        score = ReportService.topic_match_score("AI safety research", "machine learning safety")
        assert 0 < score < 1.0

    def test_topic_match_empty(self):
        assert ReportService.topic_match_score("", "anything") == 0.0


class TestFileHelpers:
    def test_parse_frontmatter(self):
        content = "---\ntopic: 'test'\nmodel: qwen\n---\n# Body"
        result = ReportService.parse_frontmatter(content)
        assert result["topic"] == "test"
        assert result["model"] == "qwen"

    def test_parse_frontmatter_no_match(self):
        assert ReportService.parse_frontmatter("no frontmatter") == {}

    def test_load_json_file_missing(self, tmp_path):
        assert ReportService.load_json_file(tmp_path / "nope.json") == {}

    def test_load_json_file_valid(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"key": "val"}), encoding="utf-8")
        assert ReportService.load_json_file(p) == {"key": "val"}

    def test_load_json_file_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        assert ReportService.load_json_file(p) == {}


class TestQualityScore:
    def test_valid_quality(self):
        q = {"coverage_score": 0.9, "structure_score": 0.8, "provenance_score": 0.7}
        score = ReportService.quality_score(q)
        expected = round(0.9 * 0.45 + 0.8 * 0.35 + 0.7 * 0.20, 3)
        assert score == expected

    def test_empty_quality(self):
        assert ReportService.quality_score({}) == 0.0

    def test_non_mapping(self):
        assert ReportService.quality_score("not a dict") == 0.0


class TestResolveSourceCount:
    def test_from_sources_list(self):
        assert ReportService.resolve_source_count({"sources": [1, 2, 3]}, {}) == 3

    def test_from_note_meta(self):
        assert ReportService.resolve_source_count({}, {"sources_count": "5"}) == 5

    def test_fallback_zero(self):
        assert ReportService.resolve_source_count({}, {}) == 0
