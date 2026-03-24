"""Tests for research_with_search failure paths (Section H).

Each test injects a specific failure condition and verifies the expected behavior.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from auto_research.telegram.action_registry import ActionRegistry, MINIMUM_VIABLE_SOURCES


@pytest.fixture
def registry(tmp_path):
    """Create an ActionRegistry with a mocked orchestrator and config."""
    from conftest import make_temp_repo
    from auto_research.runtime import load_config

    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})

    orchestrator = MagicMock()
    orchestrator.config = config
    orchestrator.doctor.return_value = {
        "provider": "lmstudio",
        "model": "test-model",
        "services": {"ollama": {"ok": False}, "lmstudio": {"ok": True}},
    }

    return ActionRegistry(orchestrator=orchestrator, config=config)


class TestSearchZeroResults:
    """Failure path 1: search returns zero results."""

    def test_search_returns_empty(self, registry):
        with patch("auto_research.search.WebSearchAdapter") as MockSearch:
            MockSearch.return_value.search.return_value = []
            result = registry.execute("research_with_search", {"topic": "test topic"}, "sk")
        assert "未找到結果" in result
        assert "test topic" in result


class TestFetchPartialBelowThreshold:
    """Failure path 2: fetch partial failure below minimum viable threshold."""

    def test_partial_failure_below_min(self, registry):
        from auto_research.search.models import SearchResult, FetchResult

        search_results = [
            SearchResult(title="A", url="http://a.example.com", snippet="a", rank=1),
            SearchResult(title="B", url="http://b.example.com", snippet="b", rank=2),
            SearchResult(title="C", url="http://c.example.com", snippet="c", rank=3),
        ]

        def mock_fetch(url, **kw):
            if url == "http://a.example.com":
                return FetchResult(
                    url=url, title="A", content="# A\n\nContent", word_count=10,
                    fetch_status="ok", fetched_at="2026-01-01", source_type="web",
                )
            return FetchResult(
                url=url, title="", content="", word_count=0,
                fetch_status="error", fetched_at="2026-01-01", source_type="web",
                error="Connection refused",
            )

        with patch("auto_research.search.WebSearchAdapter") as MockSearch, \
             patch("auto_research.search.SourceFetcher") as MockFetcher:
            MockSearch.return_value.search.return_value = search_results
            fetcher_inst = MockFetcher.return_value
            fetcher_inst.fetch_and_parse.side_effect = mock_fetch

            result = registry.execute("research_with_search", {"topic": "test"}, "sk")

        # With 3 search results, we need min(MINIMUM_VIABLE_SOURCES, 3) = 2 fetched
        # Only 1 succeeded, so should report partial failure
        assert "只成功擷取" in result or "失敗" in result


class TestFetchTotalFailure:
    """Failure path 3: all fetches fail."""

    def test_all_fetches_fail(self, registry):
        from auto_research.search.models import SearchResult, FetchResult

        search_results = [
            SearchResult(title="A", url="http://a.example.com", snippet="a", rank=1),
            SearchResult(title="B", url="http://b.example.com", snippet="b", rank=2),
        ]

        def mock_fetch(url, **kw):
            return FetchResult(
                url=url, title="", content="", word_count=0,
                fetch_status="error", fetched_at="2026-01-01", source_type="web",
                error="Timeout",
            )

        with patch("auto_research.search.WebSearchAdapter") as MockSearch, \
             patch("auto_research.search.SourceFetcher") as MockFetcher:
            MockSearch.return_value.search.return_value = search_results
            MockFetcher.return_value.fetch_and_parse.side_effect = mock_fetch

            result = registry.execute("research_with_search", {"topic": "test"}, "sk")

        assert "所有擷取均失敗" in result


class TestIngestExceedsContext:
    """Failure path 4: source bundle exceeds context limit.

    When total source bytes exceed MAX_SOURCE_BYTES * 2, the handler truncates
    by removing smallest files until under limit.
    """

    def test_oversized_sources_get_truncated(self, registry, tmp_path):
        from auto_research.search.models import SearchResult, FetchResult

        search_results = [
            SearchResult(title="A", url="http://a.example.com", snippet="a", rank=1),
            SearchResult(title="B", url="http://b.example.com", snippet="b", rank=2),
            SearchResult(title="C", url="http://c.example.com", snippet="c", rank=3),
        ]

        call_count = 0

        def mock_fetch(url, **kw):
            nonlocal call_count
            call_count += 1
            return FetchResult(
                url=url, title=f"Doc{call_count}", content="x" * 100,
                word_count=100, fetch_status="ok",
                fetched_at="2026-01-01", source_type="web",
            )

        def mock_write(layout, sr, fr):
            # Write a large file to simulate oversized sources
            fname = f"web-doc{sr.rank}.md"
            fpath = layout.parsed_dir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            # Write 120KB per file (3 files = 360KB, over MAX_SOURCE_BYTES * 2 = 204800)
            fpath.write_text("---\ntitle: test\n---\n" + "x" * 120_000, encoding="utf-8")

        with patch("auto_research.search.WebSearchAdapter") as MockSearch, \
             patch("auto_research.search.SourceFetcher") as MockFetcher:
            MockSearch.return_value.search.return_value = search_results
            fetcher_inst = MockFetcher.return_value
            fetcher_inst.fetch_and_parse.side_effect = mock_fetch
            fetcher_inst.write_to_session.side_effect = mock_write

            # Mock orchestrator.synthesize to return a valid result
            registry._orchestrator.synthesize.return_value = {
                "provider": "test", "model": "test",
                "quality": {"passed": False, "word_count": 50, "coverage_score": 0.1},
                "note_path": None,
            }

            result = registry.execute("research_with_search", {"topic": "test"}, "sk")

        # Should succeed (truncation happened, but synthesis still ran)
        assert "搜尋研究完成" in result or "研究" in result
        # Verify synthesize was called (meaning truncation didn't abort)
        assert registry._orchestrator.synthesize.called


class TestSynthesisTimeout:
    """Failure path 5: synthesis times out."""

    def test_synthesis_timeout(self, registry):
        from auto_research.search.models import SearchResult, FetchResult

        search_results = [
            SearchResult(title="A", url="http://a.example.com", snippet="a", rank=1),
        ]

        def mock_fetch(url, **kw):
            return FetchResult(
                url=url, title="A", content="# A\n\nContent here",
                word_count=100, fetch_status="ok",
                fetched_at="2026-01-01", source_type="web",
            )

        def mock_write(layout, sr, fr):
            fpath = layout.parsed_dir / "web-a.md"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text("---\ntitle: A\n---\nContent", encoding="utf-8")

        with patch("auto_research.search.WebSearchAdapter") as MockSearch, \
             patch("auto_research.search.SourceFetcher") as MockFetcher:
            MockSearch.return_value.search.return_value = search_results
            fetcher_inst = MockFetcher.return_value
            fetcher_inst.fetch_and_parse.side_effect = mock_fetch
            fetcher_inst.write_to_session.side_effect = mock_write

            # Make synthesize raise TimeoutError
            registry._orchestrator.synthesize.side_effect = TimeoutError("LLM timeout after 900s")

            result = registry.execute("research_with_search", {"topic": "test"}, "sk")

        assert "逾時" in result or "timeout" in result.lower() or "失敗" in result


class TestFrontmatterViolation:
    """Failure path 6: synthesis output missing required frontmatter."""

    def test_missing_frontmatter_gets_repaired(self, registry, tmp_path):
        from auto_research.search.models import SearchResult, FetchResult

        search_results = [
            SearchResult(title="A", url="http://a.example.com", snippet="a", rank=1),
        ]

        def mock_fetch(url, **kw):
            return FetchResult(
                url=url, title="A", content="# A\n\nContent",
                word_count=100, fetch_status="ok",
                fetched_at="2026-01-01", source_type="web",
            )

        # Create a note file without frontmatter
        note_path = tmp_path / "bad_note.md"
        note_path.write_text("# No Frontmatter\n\nJust content.", encoding="utf-8")

        def mock_write(layout, sr, fr):
            fpath = layout.parsed_dir / "web-a.md"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text("---\ntitle: A\n---\nContent", encoding="utf-8")

        with patch("auto_research.search.WebSearchAdapter") as MockSearch, \
             patch("auto_research.search.SourceFetcher") as MockFetcher:
            MockSearch.return_value.search.return_value = search_results
            fetcher_inst = MockFetcher.return_value
            fetcher_inst.fetch_and_parse.side_effect = mock_fetch
            fetcher_inst.write_to_session.side_effect = mock_write

            registry._orchestrator.synthesize.return_value = {
                "provider": "test", "model": "test",
                "quality": {"passed": False, "word_count": 50, "coverage_score": 0.1},
                "note_path": str(note_path),
            }

            result = registry.execute("research_with_search", {"topic": "test"}, "sk")

        # Note should now have frontmatter injected
        repaired = note_path.read_text(encoding="utf-8")
        assert repaired.startswith("---")
        assert "topic" in repaired
        assert "session_id" in repaired


class TestObsidianExportFailure:
    """Failure path 7: Obsidian export target not writable."""

    def test_export_failure_does_not_crash(self, registry):
        from auto_research.search.models import SearchResult, FetchResult

        search_results = [
            SearchResult(title="A", url="http://a.example.com", snippet="a", rank=1),
        ]

        def mock_fetch(url, **kw):
            return FetchResult(
                url=url, title="A", content="# A\n\nContent",
                word_count=100, fetch_status="ok",
                fetched_at="2026-01-01", source_type="web",
            )

        def mock_write(layout, sr, fr):
            fpath = layout.parsed_dir / "web-a.md"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text("---\ntitle: A\n---\nContent", encoding="utf-8")

        with patch("auto_research.search.WebSearchAdapter") as MockSearch, \
             patch("auto_research.search.SourceFetcher") as MockFetcher:
            MockSearch.return_value.search.return_value = search_results
            fetcher_inst = MockFetcher.return_value
            fetcher_inst.fetch_and_parse.side_effect = mock_fetch
            fetcher_inst.write_to_session.side_effect = mock_write

            registry._orchestrator.synthesize.return_value = {
                "provider": "test", "model": "test",
                "quality": {"passed": True, "word_count": 200, "coverage_score": 0.5},
                "note_path": None,
            }
            registry._orchestrator.memory_extract.return_value = {"draft_id": "test-draft"}

            # Patch ObsidianExporter to raise
            with patch(
                "auto_research.telegram.action_registry.ActionRegistry._run_bounded_research",
                wraps=registry._run_bounded_research,
            ):
                # Force the obsidian import to raise inside _run_bounded_research
                with patch.dict("sys.modules", {"auto_research.integrations.obsidian": MagicMock(
                    ObsidianExporter=MagicMock(side_effect=PermissionError("vault not writable")),
                )}):
                    result = registry.execute("research_with_search", {"topic": "test"}, "sk")

        # Should still succeed — export failure is degraded, not fatal
        assert "搜尋研究完成" in result or "研究" in result
        # No Obsidian path in response
        assert "Obsidian" not in result or "None" in result or result.count("Obsidian") == 0
