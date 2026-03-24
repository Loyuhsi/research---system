"""Source ranker — analyzes fetched sources for quality and relevance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..runtime import AutoResearchConfig


@dataclass
class SourceScore:
    """Quality score for a single fetched source."""
    filename: str
    size_bytes: int
    has_content: bool
    word_count: int
    quality_score: float  # 0.0 - 1.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "has_content": self.has_content,
            "word_count": self.word_count,
            "quality_score": round(self.quality_score, 3),
        }


class SourceRanker:
    """Ranks fetched sources by quality metrics for synthesis prioritization."""

    MIN_USEFUL_WORDS = 50
    OPTIMAL_WORD_COUNT = 2000

    def __init__(self, config: AutoResearchConfig) -> None:
        self.config = config

    def rank_session(self, session_id: str) -> Dict[str, object]:
        """Rank all sources in a session by quality."""
        layout = self.config.resolve_layout(session_id)

        # Try parsed dir first, fall back to legacy
        source_dir = layout.parsed_dir if layout.parsed_dir.exists() else layout.legacy_sources_dir
        if not source_dir.exists():
            return {"session_id": session_id, "sources": [], "summary": "No sources found."}

        scores: List[SourceScore] = []
        for md_file in sorted(source_dir.glob("*.md")):
            score = self._score_file(md_file)
            scores.append(score)

        # Sort by quality descending
        scores.sort(key=lambda s: s.quality_score, reverse=True)

        usable = [s for s in scores if s.has_content]
        return {
            "session_id": session_id,
            "total_sources": len(scores),
            "usable_sources": len(usable),
            "avg_quality": round(sum(s.quality_score for s in scores) / max(len(scores), 1), 3),
            "sources": [s.to_dict() for s in scores],
            "summary": self._build_summary(scores),
        }

    def _score_file(self, path: Path) -> SourceScore:
        """Score a single markdown source file."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return SourceScore(filename=path.name, size_bytes=0, has_content=False, word_count=0, quality_score=0.0)

        size = len(content.encode("utf-8"))
        words = len(content.split())
        has_content = words >= self.MIN_USEFUL_WORDS

        # Quality heuristics
        score = 0.0
        if has_content:
            # Word count score (normalized, peaks at OPTIMAL_WORD_COUNT)
            word_ratio = min(words / self.OPTIMAL_WORD_COUNT, 1.0)
            score += word_ratio * 0.4

            # Structure score (has headings?)
            heading_count = content.count("\n#")
            score += min(heading_count / 5, 1.0) * 0.2

            # Content density (non-whitespace ratio)
            stripped = content.replace(" ", "").replace("\n", "").replace("\t", "")
            density = len(stripped) / max(len(content), 1)
            score += density * 0.2

            # Link/reference score
            link_count = content.count("](")
            score += min(link_count / 10, 1.0) * 0.1

            # No excessive repetition
            unique_words = len(set(content.lower().split()))
            uniqueness = unique_words / max(words, 1)
            score += uniqueness * 0.1

        return SourceScore(
            filename=path.name,
            size_bytes=size,
            has_content=has_content,
            word_count=words,
            quality_score=min(score, 1.0),
        )

    def _build_summary(self, scores: List[SourceScore]) -> str:
        if not scores:
            return "No sources to rank."
        usable = [s for s in scores if s.has_content]
        low_quality = [s for s in scores if s.quality_score < 0.3 and s.has_content]
        parts = [f"{len(usable)}/{len(scores)} sources are usable."]
        if low_quality:
            parts.append(f"{len(low_quality)} sources have low quality and may need re-fetching.")
        best = scores[0] if scores else None
        if best and best.has_content:
            parts.append(f"Best source: {best.filename} (score: {best.quality_score:.2f})")
        return " ".join(parts)
