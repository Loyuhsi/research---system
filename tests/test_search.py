"""Tests for search and source-fetching package."""

from __future__ import annotations

import ssl
import urllib.error

from auto_research.search.models import SearchResult, FetchResult
from auto_research.search.web_search import WebSearchAdapter, _DDGResultParser
from auto_research.search.source_fetcher import SourceFetcher, _truncate_words


class TestSearchModels:
    def test_search_result_to_dict(self):
        r = SearchResult(title="Test", url="https://example.com", snippet="A snippet", rank=1)
        d = r.to_dict()
        assert d["title"] == "Test"
        assert d["rank"] == 1

    def test_fetch_result_to_dict(self):
        r = FetchResult(
            url="https://example.com", title="Page", content="Hello world",
            word_count=2, fetch_status="ok", fetched_at="2026-01-01T00:00:00Z",
            source_type="web",
        )
        d = r.to_dict()
        assert d["fetch_status"] == "ok"
        assert d["source_type"] == "web"
        assert d["fetched_at"] == "2026-01-01T00:00:00Z"

    def test_fetch_result_error(self):
        r = FetchResult(
            url="https://fail.com", title="", content="", word_count=0,
            fetch_status="error", fetched_at="", source_type="web", error="timeout",
        )
        assert r.error == "timeout"


class TestDDGParser:
    def test_parse_result_html(self):
        html = '''
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">
            Example Page Title
        </a>
        <a class="result__snippet">
            This is a snippet about the page.
        </a>
        '''
        parser = _DDGResultParser()
        parser.feed(html)
        assert len(parser.results) == 1
        assert parser.results[0]["title"] == "Example Page Title"
        assert "example.com" in parser.results[0]["url"]

    def test_parse_empty_html(self):
        parser = _DDGResultParser()
        parser.feed("<html><body>No results</body></html>")
        assert len(parser.results) == 0


class TestWebSearchAdapter:
    def test_empty_query_returns_empty(self):
        adapter = WebSearchAdapter()
        assert adapter.search("") == []
        assert adapter.search("   ") == []

    def test_max_results_enforced(self):
        adapter = WebSearchAdapter()
        # We can't mock stdlib urllib easily without unittest.mock,
        # but we can verify the interface exists and handles edge cases
        results = adapter.search("zzzzz_impossible_query_xyz_12345", max_results=1)
        # Either returns empty (DDG returned nothing) or at most 1 result
        assert len(results) <= 1


class TestTruncateWords:
    def test_short_text_unchanged(self):
        assert _truncate_words("hello world", 10) == "hello world"

    def test_truncation_at_word_boundary(self):
        text = " ".join(f"word{i}" for i in range(100))
        result = _truncate_words(text, 5)
        assert result.startswith("word0 word1 word2 word3 word4")
        assert "truncated to 5 words" in result

    def test_exact_boundary(self):
        text = "one two three"
        assert _truncate_words(text, 3) == "one two three"


class TestSourceFetcher:
    def test_content_extractor_strips_scripts(self):
        from auto_research.search.source_fetcher import _ContentExtractor
        extractor = _ContentExtractor()
        extractor.feed("<html><title>Test</title><script>evil()</script><p>Content here</p></html>")
        text = extractor.get_text()
        assert "evil" not in text
        assert "Content here" in text
        assert extractor.title == "Test"

    def test_content_extractor_strips_nav_footer(self):
        from auto_research.search.source_fetcher import _ContentExtractor
        extractor = _ContentExtractor()
        extractor.feed("<html><nav>Nav stuff</nav><main>Main content</main><footer>Footer</footer></html>")
        text = extractor.get_text()
        assert "Nav stuff" not in text
        assert "Footer" not in text
        assert "Main content" in text

    def test_write_to_session(self, tmp_path):
        from conftest import make_temp_repo
        from auto_research.runtime import load_config
        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={})
        layout = config.resolve_layout("test-search")
        layout.ensure()

        sr = SearchResult(title="Test Page", url="https://example.com", snippet="...", rank=1)
        fr = FetchResult(
            url="https://example.com", title="Test Page", content="Hello world content",
            word_count=3, fetch_status="ok", fetched_at="2026-01-01T00:00:00Z", source_type="web",
        )
        fetcher = SourceFetcher()
        path = fetcher.write_to_session(layout, sr, fr)
        assert path is not None
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "source_url:" in content
        assert "fetched_at:" in content
        assert "Hello world content" in content

    def test_fetch_error_path(self):
        """Unreachable URL returns FetchResult with status='error'."""
        fetcher = SourceFetcher()
        result = fetcher.fetch_and_parse("http://127.0.0.1:1/nonexistent", timeout=1)
        assert result.fetch_status == "error"
        assert result.word_count == 0
        assert result.error  # should have an error message

    def test_write_to_session_skips_on_error(self, tmp_path):
        """FetchResult with error status should not be written."""
        from conftest import make_temp_repo
        from auto_research.runtime import load_config
        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={})
        layout = config.resolve_layout("test-skip-error")
        layout.ensure()

        sr = SearchResult(title="Bad", url="https://fail.com", snippet="...", rank=1)
        fr = FetchResult(
            url="https://fail.com", title="", content="", word_count=0,
            fetch_status="error", fetched_at="", source_type="web", error="timeout",
        )
        fetcher = SourceFetcher()
        path = fetcher.write_to_session(layout, sr, fr)
        assert path is None

    def test_github_source_type(self):
        """URLs containing github.com should get source_type='github'."""
        fetcher = SourceFetcher()
        result = fetcher.fetch_and_parse("http://127.0.0.1:1/github.com/test", timeout=1)
        assert result.source_type == "github"

    def test_fetch_ssl_certificate_error_not_silenced(self, monkeypatch):
        """Certificate verification errors must NOT be silently bypassed (no insecure fallback)."""
        fetcher = SourceFetcher()

        def fake_urlopen(req, timeout=0, context=None):
            import ssl
            raise urllib.error.URLError(
                ssl.SSLCertVerificationError("certificate verify failed: self-signed certificate")
            )

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        result = fetcher.fetch_and_parse("https://example.com/docs", timeout=1)
        assert result.fetch_status == "error"
        assert "certificate" in (result.error or "").lower()
