"""Bounded search and source-ingestion for Auto-Research."""

from .models import SearchResult, FetchResult
from .source_fetcher import ContentExtractor, HeuristicHtmlExtractor, SourceFetcher
from .web_search import DuckDuckGoHtmlSearchProvider, SearchProvider, WebSearchAdapter

__all__ = [
    "SearchResult",
    "FetchResult",
    "SearchProvider",
    "DuckDuckGoHtmlSearchProvider",
    "WebSearchAdapter",
    "ContentExtractor",
    "HeuristicHtmlExtractor",
    "SourceFetcher",
]
