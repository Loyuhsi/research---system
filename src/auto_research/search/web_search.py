"""Bounded web search providers for Auto-Research.

Default implementation uses DuckDuckGo HTML search with stdlib-only HTTP.
The public interface stays stable while allowing future providers to plug in.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import List, Protocol

from .models import SearchResult

logger = logging.getLogger(__name__)

_DDG_URL = "https://html.duckduckgo.com/html/"


class SearchProvider(Protocol):
    provider_name: str

    def search(
        self,
        query: str,
        max_results: int = 5,
        timeout: int = 15,
    ) -> List[SearchResult]:
        ...


class _DDGResultParser(HTMLParser):
    """Extract result titles, URLs, and snippets from DuckDuckGo HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.results: List[dict[str, str]] = []
        self._capture: str = ""
        self._current_url = ""
        self._title_parts: List[str] = []
        self._snippet_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attr_dict = dict(attrs)
        cls = attr_dict.get("class", "")
        if tag == "a" and "result__a" in cls:
            self._capture = "title"
            href = attr_dict.get("href", "")
            match = re.search(r"uddg=([^&]+)", href)
            self._current_url = urllib.parse.unquote(match.group(1)) if match else href
            self._title_parts = []
            return
        if tag in {"a", "div"} and "result__snippet" in cls:
            self._capture = "snippet"
            self._snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._capture == "title" and tag == "a":
            self._capture = ""
            return
        if self._capture == "snippet" and tag in {"a", "div"}:
            self._capture = ""
            title = " ".join(part for part in self._title_parts if part).strip()
            snippet = " ".join(part for part in self._snippet_parts if part).strip()
            if self._current_url and title:
                self.results.append({"title": title, "url": self._current_url, "snippet": snippet})
            self._current_url = ""
            self._title_parts = []
            self._snippet_parts = []

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self._capture == "title":
            self._title_parts.append(cleaned)
        elif self._capture == "snippet":
            self._snippet_parts.append(cleaned)


class DuckDuckGoHtmlSearchProvider:
    """Bounded web search using DuckDuckGo HTML results."""

    provider_name = "duckduckgo_html"

    def search(
        self,
        query: str,
        max_results: int = 5,
        timeout: int = 15,
    ) -> List[SearchResult]:
        if not query.strip():
            return []
        try:
            encoded = urllib.parse.urlencode({"q": query})
            req = urllib.request.Request(
                f"{_DDG_URL}?{encoded}",
                headers={"User-Agent": "Mozilla/5.0 (Auto-Research)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            parser = _DDGResultParser()
            parser.feed(html)

            results: List[SearchResult] = []
            for index, item in enumerate(parser.results[:max_results], 1):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("snippet", ""),
                        rank=index,
                        provider=self.provider_name,
                        score=max(max_results - index + 1, 1) / max(max_results, 1),
                        metadata={"rank_source": "duckduckgo_html"},
                    )
                )
            return results
        except Exception as exc:
            logger.warning("Web search failed for query '%s': %s", query[:80], exc)
            return []


class WebSearchAdapter:
    """Backwards-compatible search adapter with pluggable provider support."""

    DEFAULT_MAX_RESULTS = 5
    DEFAULT_TIMEOUT = 15

    def __init__(self, provider: SearchProvider | None = None) -> None:
        self._provider = provider or DuckDuckGoHtmlSearchProvider()

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    def search(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> List[SearchResult]:
        return self._provider.search(query, max_results=max_results, timeout=timeout)
