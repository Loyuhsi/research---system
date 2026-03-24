"""Post-synthesis quality gate — validates research notes against sources."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..runtime import AutoResearchConfig


@dataclass
class QualityReport:
    """Result of a quality check on a synthesized note."""

    coverage_score: float = 0.0
    hallucination_flags: List[str] = field(default_factory=list)
    structure_score: float = 0.0
    word_count: int = 0
    passed: bool = False
    evidence_count: int = 0
    source_diversity: int = 0
    provenance_score: float = 0.0
    retrieval_metadata: Dict[str, object] = field(default_factory=dict)
    response_mode: str = ""
    retry_count: int = 0
    source_ranking: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "coverage_score": round(self.coverage_score, 3),
            "hallucination_flags": self.hallucination_flags,
            "structure_score": round(self.structure_score, 3),
            "word_count": self.word_count,
            "passed": self.passed,
            "evidence_count": self.evidence_count,
            "source_diversity": self.source_diversity,
            "provenance_score": round(self.provenance_score, 3),
            "retrieval_metadata": self.retrieval_metadata,
            "response_mode": self.response_mode,
            "retry_count": self.retry_count,
            "source_ranking": self.source_ranking,
        }


# Defaults — may be overridden via env. Future: integrate into config schema.
DEFAULT_MIN_WORD_COUNT = 200
DEFAULT_COVERAGE_THRESHOLD = 0.3
DEFAULT_STRUCTURE_THRESHOLD = 0.4
DEFAULT_PASS_THRESHOLD = 0.5
NGRAM_SIZE = 4


class QualityGateService:
    """Checks synthesized notes for quality against source material.

    Thresholds are config-driven via env_values with fallback to class defaults.
    Future: these defaults should be incorporated into the formal config schema.
    """

    def __init__(self, config: AutoResearchConfig) -> None:
        self.config = config
        env = config.env_values
        self._min_word_count = _int_or(env.get("QUALITY_MIN_WORDS"), DEFAULT_MIN_WORD_COUNT)
        self._coverage_threshold = _float_or(env.get("QUALITY_COVERAGE_THRESHOLD"), DEFAULT_COVERAGE_THRESHOLD)
        self._structure_threshold = _float_or(env.get("QUALITY_STRUCTURE_THRESHOLD"), DEFAULT_STRUCTURE_THRESHOLD)
        self._pass_threshold = _float_or(env.get("QUALITY_PASS_THRESHOLD"), DEFAULT_PASS_THRESHOLD)

    @property
    def min_word_count(self) -> int:
        return self._min_word_count

    @property
    def coverage_threshold(self) -> float:
        return self._coverage_threshold

    @property
    def structure_threshold(self) -> float:
        return self._structure_threshold

    @property
    def pass_threshold(self) -> float:
        return self._pass_threshold

    @property
    def thresholds(self) -> Dict[str, object]:
        """Return current effective thresholds for diagnostics / doctor."""
        return {
            "min_word_count": self._min_word_count,
            "coverage_threshold": self._coverage_threshold,
            "structure_threshold": self._structure_threshold,
            "pass_threshold": self._pass_threshold,
        }

    def check(
        self,
        note_path: Path,
        sources: List[Path],
        provenance: Optional[Dict[str, object]] = None,
    ) -> QualityReport:
        """Run quality checks on a synthesized note.

        Args:
            provenance: optional dict with 'evidence_count' and 'source_types' keys
                        to factor into quality scoring.
        """
        from ..tracing import current_trace, SpanKind
        trace = current_trace()
        span = None
        if trace:
            span = trace.start_span("quality_gate.check", SpanKind.QUALITY_GATE)

        try:
            content = note_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            if span:
                span.finish(status="error", error="Note file unreadable")
            return QualityReport(hallucination_flags=["Note file unreadable."])

        note_body = self._strip_frontmatter(content)
        word_count = len(note_body.split())

        # Load source texts
        source_texts = []
        for src in sources:
            try:
                source_texts.append(src.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                logger.debug("Skipping unreadable source %s", src, exc_info=True)
                continue

        coverage = self._check_source_coverage(note_body, source_texts)
        hallucination_flags = self._check_hallucination_risk(note_body, source_texts)
        structure = self._check_structure(content)

        # Provenance signals
        evidence_count = int(str(provenance.get("evidence_count", 0))) if provenance else len(sources)
        source_types_raw = provenance.get("source_types", []) if provenance else []
        source_types_list = list(source_types_raw) if isinstance(source_types_raw, (list, tuple, set)) else []
        source_diversity = len(set(source_types_list)) if source_types_list else len(sources)
        provenance_score = self._compute_provenance_score(evidence_count, source_diversity)
        retrieval_metadata = provenance.get("retrieval_metadata", {}) if provenance else {}
        response_mode = str(provenance.get("response_mode", "")) if provenance else ""
        retry_count = int(str(provenance.get("retry_count", 0))) if provenance else 0
        source_ranking = provenance.get("source_ranking", {}) if provenance else {}

        # Composite pass/fail — provenance contributes 10% weight
        composite = (
            (coverage * 0.35)
            + (structure * 0.25)
            + (min(word_count / 500, 1.0) * 0.3)
            + (provenance_score * 0.10)
        )
        passed = (
            word_count >= self._min_word_count
            and coverage >= self._coverage_threshold
            and structure >= self._structure_threshold
            and composite >= self._pass_threshold
        )

        if span:
            span.attributes.update({
                "coverage_score": round(coverage, 3),
                "structure_score": round(structure, 3),
                "word_count": word_count,
                "passed": passed,
                "evidence_count": evidence_count,
                "source_diversity": source_diversity,
                "provenance_score": round(provenance_score, 3),
                "gen_ai.response.mode": response_mode,
                "gen_ai.retry_count": retry_count,
                "retrieval.hit_count": evidence_count,
            })
            span.finish(status="ok")

        return QualityReport(
            coverage_score=coverage,
            hallucination_flags=hallucination_flags,
            structure_score=structure,
            word_count=word_count,
            passed=passed,
            evidence_count=evidence_count,
            source_diversity=source_diversity,
            provenance_score=provenance_score,
            retrieval_metadata=retrieval_metadata if isinstance(retrieval_metadata, dict) else {},
            response_mode=response_mode,
            retry_count=retry_count,
            source_ranking=source_ranking if isinstance(source_ranking, dict) else {},
        )

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from content."""
        if content.startswith("---"):
            lines = content.splitlines()
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == "---":
                    return "\n".join(lines[i + 1 :])
        return content

    def _check_source_coverage(self, note: str, sources: List[str]) -> float:
        """Check what fraction of source key-phrases appear in the note."""
        if not sources:
            return 0.0

        source_ngrams: set[str] = set()
        for source in sources:
            source_ngrams.update(self._extract_ngrams(source))

        if not source_ngrams:
            return 0.0

        note_text_lower = note.lower()
        hits = sum(1 for ng in source_ngrams if ng in note_text_lower)
        return hits / len(source_ngrams)

    def _check_hallucination_risk(self, note: str, sources: List[str]) -> List[str]:
        """Flag note n-grams not found in any source (potential hallucinations)."""
        flags: List[str] = []
        if not sources:
            flags.append("No sources available to verify claims.")
            return flags

        combined_source = " ".join(sources).lower()
        note_ngrams = self._extract_ngrams(note)

        unsupported = [ng for ng in note_ngrams if ng not in combined_source]
        unsupported_ratio = len(unsupported) / max(len(note_ngrams), 1)

        if unsupported_ratio > 0.7:
            flags.append(f"High unsupported content ratio: {unsupported_ratio:.0%}")
        elif unsupported_ratio > 0.5:
            flags.append(f"Moderate unsupported content ratio: {unsupported_ratio:.0%}")

        return flags

    def _check_structure(self, content: str) -> float:
        """Score structural quality: frontmatter, headings, conclusion."""
        score = 0.0

        # Has frontmatter?
        if content.startswith("---"):
            score += 0.3

        # Has headings?
        heading_count = len(re.findall(r"^#{1,3}\s", content, re.MULTILINE))
        score += min(heading_count / 5, 1.0) * 0.4

        # Has a conclusion-like section?
        lower = content.lower()
        if any(marker in lower for marker in ("## 結論", "## conclusion", "## summary", "## 總結", "## 後續建議", "## 來源與限制")):
            score += 0.3

        return min(score, 1.0)

    def _extract_ngrams(self, text: str, n: int = NGRAM_SIZE) -> List[str]:
        """Extract word-level n-grams from text."""
        words = text.lower().split()
        if len(words) < n:
            return [" ".join(words)] if words else []
        return [" ".join(words[i : i + n]) for i in range(0, len(words) - n + 1, n)]

    @staticmethod
    def _compute_provenance_score(evidence_count: int, source_diversity: int) -> float:
        """Score based on evidence quantity and source diversity."""
        ev_score = min(evidence_count / 5.0, 1.0)
        div_score = min(source_diversity / 3.0, 1.0)
        return (ev_score * 0.6) + (div_score * 0.4)


def _float_or(raw: object, default: float) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(str(raw))
    except (ValueError, TypeError):
        return default


def _int_or(raw: object, default: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        return int(str(raw))
    except (ValueError, TypeError):
        return default
