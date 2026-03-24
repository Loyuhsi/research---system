"""Source fetcher with pluggable content extraction and bounded Markdown output."""

from __future__ import annotations

import datetime as dt
import logging
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Protocol

from .models import FetchResult, SearchResult

logger = logging.getLogger(__name__)

SLUG_RE = re.compile(r"[^a-z0-9]+")
SKIP_HINT_RE = re.compile(r"(nav|menu|footer|header|sidebar|cookie|share|promo|banner)", re.I)
HEADING_TAGS = {"h1", "h2", "h3", "h4"}
TEXT_BLOCK_TAGS = {"p", "li", "blockquote", "pre", "main", "div", "article", "section"} | HEADING_TAGS


@dataclass(frozen=True)
class ExtractedDocument:
    title: str
    canonical_url: str
    description: str
    published_at: str
    headings: List[str]
    paragraphs: List[str]
    extractor: str


class ContentExtractor(Protocol):
    extractor_name: str

    def extract(self, html: str, url: str) -> ExtractedDocument:
        ...


class _HeuristicHtmlParser(HTMLParser):
    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "form", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.description = ""
        self.canonical_url = ""
        self.published_at = ""
        self.blocks: List[tuple[str, str]] = []
        self.headings: List[str] = []
        self._skip_depth = 0
        self._capture_tag = ""
        self._capture_parts: List[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attr_dict = dict(attrs)
        if tag in self.SKIP_TAGS or self._should_skip_block(attr_dict):
            self._skip_depth += 1
            return

        if tag == "title":
            self._in_title = True
            return

        if tag == "meta":
            name = attr_dict.get("name", "").lower()
            prop = attr_dict.get("property", "").lower()
            content = " ".join(attr_dict.get("content", "").split())
            if content:
                if name == "description" or prop == "og:description":
                    self.description = self.description or content
                if prop == "og:title":
                    self.title = self.title or content
                if name in {"pubdate", "date"} or prop in {"article:published_time", "og:published_time"}:
                    self.published_at = self.published_at or content
            return

        if tag == "link":
            rel = attr_dict.get("rel", "").lower()
            href = attr_dict.get("href", "").strip()
            if "canonical" in rel and href:
                self.canonical_url = href
            return

        if self._skip_depth == 0 and tag in TEXT_BLOCK_TAGS:
            self._capture_tag = tag
            self._capture_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return

        if self._skip_depth > 0 and (tag in self.SKIP_TAGS or tag in TEXT_BLOCK_TAGS):
            self._skip_depth -= 1
            return

        if tag == self._capture_tag:
            text = " ".join(part for part in self._capture_parts if part).strip()
            self._capture_tag = ""
            self._capture_parts = []
            if text:
                self.blocks.append((tag, text))
                if tag in HEADING_TAGS:
                    self.headings.append(text)

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self._in_title:
            self.title = f"{self.title} {cleaned}".strip()
        if self._skip_depth == 0 and self._capture_tag:
            self._capture_parts.append(cleaned)

    @staticmethod
    def _should_skip_block(attrs: dict[str, str]) -> bool:
        blob = " ".join([attrs.get("class", ""), attrs.get("id", ""), attrs.get("role", "")]).strip()
        return bool(blob and SKIP_HINT_RE.search(blob))


class HeuristicHtmlExtractor:
    extractor_name = "heuristic_html"

    def extract(self, html: str, url: str) -> ExtractedDocument:
        parser = _HeuristicHtmlParser()
        parser.feed(html)
        candidate_blocks = self._rank_blocks(parser.blocks)
        paragraphs = self._render_blocks(candidate_blocks)
        title = parser.title or url.split("/")[-1] or url
        return ExtractedDocument(
            title=title,
            canonical_url=parser.canonical_url or url,
            description=parser.description,
            published_at=parser.published_at,
            headings=parser.headings[:8],
            paragraphs=paragraphs,
            extractor=self.extractor_name,
        )

    def _rank_blocks(self, blocks: List[tuple[str, str]]) -> List[tuple[str, str]]:
        if not blocks:
            return []
        scored: List[tuple[int, float, tuple[str, str]]] = []
        for index, block in enumerate(blocks):
            tag, text = block
            words = len(text.split())
            if words == 0:
                continue
            score = min(words / 40.0, 1.0)
            if tag in HEADING_TAGS:
                score += 0.35
            if len(text) > 400:
                score += 0.15
            if text.count("|") > 4:
                score -= 0.25
            if text.lower().startswith(("sign in", "subscribe", "cookie")):
                score -= 0.5
            scored.append((index, score, block))
        if not scored:
            return []
        scored.sort(key=lambda item: item[1], reverse=True)
        keep_indices = sorted(index for index, _, _ in scored[:24])
        return [blocks[index] for index in keep_indices]

    def _render_blocks(self, blocks: List[tuple[str, str]]) -> List[str]:
        rendered: List[str] = []
        for tag, text in blocks:
            if tag in HEADING_TAGS:
                level = min(int(tag[1]), 4)
                rendered.append(f"{'#' * level} {text}")
            else:
                rendered.append(text)
        return rendered


class _ContentExtractor(_HeuristicHtmlParser):
    """Compatibility shim for older tests importing _ContentExtractor."""

    def get_text(self) -> str:
        return "\n".join(text for _, text in self.blocks).strip()


class SourceFetcher:
    """Fetches web content and extracts text with stable parsed-source output."""

    DEFAULT_MAX_BYTES = 200_000
    DEFAULT_TIMEOUT = 15
    DEFAULT_MAX_WORDS = 5000

    def __init__(self, extractor: ContentExtractor | None = None) -> None:
        self._extractor = extractor or HeuristicHtmlExtractor()

    @property
    def extractor_name(self) -> str:
        return self._extractor.extractor_name

    def fetch_and_parse(
        self,
        url: str,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout: int = DEFAULT_TIMEOUT,
        max_words: int = DEFAULT_MAX_WORDS,
    ) -> FetchResult:
        """Download page, extract content, return bounded FetchResult. Never raises."""
        fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
        source_type = "github" if "github.com" in url else "web"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Auto-Research)"})
            raw, tls_verification = self._read_response(req, timeout=timeout, max_bytes=max_bytes)
            html = raw.decode("utf-8", errors="replace")

            extracted = self._extractor.extract(html, url)
            content = _truncate_words("\n\n".join(extracted.paragraphs).strip(), max_words)
            word_count = len(content.split())
            return FetchResult(
                url=url,
                title=extracted.title,
                content=content,
                word_count=word_count,
                fetch_status="ok",
                fetched_at=fetched_at,
                source_type=source_type,
                extractor=extracted.extractor,
                canonical_url=extracted.canonical_url,
                description=extracted.description,
                published_at=extracted.published_at,
                headings=extracted.headings,
                metadata={
                    "paragraph_count": len(extracted.paragraphs),
                    "tls_verification": tls_verification,
                },
            )
        except Exception as exc:
            logger.warning("Fetch failed for %s: %s", url[:120], exc)
            return FetchResult(
                url=url,
                title="",
                content="",
                word_count=0,
                fetch_status="error",
                fetched_at=fetched_at,
                source_type=source_type,
                extractor=self.extractor_name,
                error=str(exc)[:200],
            )

    def _read_response(
        self,
        req: urllib.request.Request,
        *,
        timeout: int,
        max_bytes: int,
    ) -> tuple[bytes, str]:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(max_bytes), "verified"
        except Exception as exc:
            if not _is_certificate_error(exc):
                raise
            logger.warning("Retrying fetch without TLS verification for %s", req.full_url)
            insecure_context = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=timeout, context=insecure_context) as resp:
                return resp.read(max_bytes), "insecure_fallback"

    def write_to_session(
        self,
        layout: object,
        search_result: SearchResult,
        fetch_result: FetchResult,
    ) -> Optional[Path]:
        """Write fetched content to session parsed/ dir with stable frontmatter contract."""
        parsed_dir_raw = getattr(layout, "parsed_dir", None)
        if parsed_dir_raw is None or fetch_result.fetch_status != "ok":
            return None
        parsed_dir = Path(str(parsed_dir_raw))
        parsed_dir.mkdir(parents=True, exist_ok=True)
        slug = _slug(search_result.title or fetch_result.title)
        filename = f"web-{slug}.md"
        path = parsed_dir / filename

        heading_preview = fetch_result.headings[:5]
        frontmatter_lines = [
            "---",
            f'title: "{_escape_yaml(fetch_result.title)}"',
            f'source_url: "{fetch_result.url}"',
            f'canonical_url: "{fetch_result.canonical_url or fetch_result.url}"',
            f'fetched_at: "{fetch_result.fetched_at}"',
            f'source_type: "{fetch_result.source_type}"',
            f'search_provider: "{search_result.provider}"',
            f'extractor: "{fetch_result.extractor}"',
            f'published_at: "{_escape_yaml(fetch_result.published_at)}"',
            f'description: "{_escape_yaml(fetch_result.description)}"',
            f"word_count: {fetch_result.word_count}",
            "---",
            "",
        ]
        if heading_preview:
            frontmatter_lines.extend(["## Headings", *[f"- {heading}" for heading in heading_preview], ""])
        body = f"# {fetch_result.title}\n\n{fetch_result.content}\n"
        path.write_text("\n".join(frontmatter_lines) + body, encoding="utf-8")
        return path


def _truncate_words(text: str, max_words: int) -> str:
    """Deterministic word-boundary truncation."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    return f"{truncated}\n\n[... truncated to {max_words} words ...]"


def _slug(text: str) -> str:
    return SLUG_RE.sub("-", text.lower()).strip("-")[:40]


def _escape_yaml(text: str) -> str:
    return text.replace('"', '\\"').replace("\n", " ")


def _is_certificate_error(exc: object) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    if isinstance(exc, urllib.error.URLError):
        return _is_certificate_error(exc.reason)
    message = str(exc).lower()
    return (
        "certificate verify failed" in message
        or "certificate_verify_failed" in message
        or "self-signed certificate" in message
    )
