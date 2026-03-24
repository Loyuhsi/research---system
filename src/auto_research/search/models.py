"""Data models for search and source-fetching."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class SearchResult:
    """A single search result from a web search adapter."""

    title: str
    url: str
    snippet: str
    rank: int = 0
    provider: str = "duckduckgo_html"
    score: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FetchResult:
    """Result of fetching and parsing a web page."""

    url: str
    title: str
    content: str          # extracted Markdown text
    word_count: int
    fetch_status: str     # "ok" | "error" | "timeout"
    fetched_at: str       # ISO timestamp
    source_type: str      # "web" | "github" | "local"
    extractor: str = "heuristic_html"
    canonical_url: str = ""
    description: str = ""
    published_at: str = ""
    headings: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
