"""Gap detector — scans evo-logs for recurring failure patterns."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..runtime import AutoResearchConfig


@dataclass
class GapReport:
    """Structured report of detected gaps from evo-log analysis."""

    total_tasks: int = 0
    total_failures: int = 0
    failure_rate: float = 0.0
    recurring_keywords: List[str] = field(default_factory=list)
    failure_summaries: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_tasks": self.total_tasks,
            "total_failures": self.total_failures,
            "failure_rate": round(self.failure_rate, 3),
            "recurring_keywords": self.recurring_keywords,
            "failure_summaries": self.failure_summaries,
            "recommendations": self.recommendations,
        }

    @property
    def has_gaps(self) -> bool:
        return self.total_failures > 0


class GapDetector:
    """Analyzes evo-log entries to detect recurring failure patterns and gaps."""

    # Keywords that commonly indicate specific failure types
    FAILURE_KEYWORDS = [
        "timeout", "missing", "not found", "permission", "blocked",
        "rate limit", "network", "parse", "encoding", "empty",
        "invalid", "expired", "unreachable", "crash", "oom",
    ]

    def __init__(self, config: AutoResearchConfig) -> None:
        self.config = config
        self._log_dir = config.repo_root / "knowledge" / "logs"

    def scan(self) -> GapReport:
        """Scan all evo-logs and produce a gap report."""
        logs = self._load_logs()
        if not logs:
            return GapReport(recommendations=["No evo-logs found. Run tasks first to generate data."])

        total = len(logs)
        failures = [log for log in logs if log.get("gap_detected", False)]
        failure_count = len(failures)
        rate = failure_count / total if total > 0 else 0.0

        # Extract failure summaries
        summaries = [str(log.get("summary", "")) for log in failures if log.get("summary")]

        # Find recurring keywords
        keyword_counts = self._extract_keywords(summaries)
        recurring = [kw for kw, count in keyword_counts.most_common(5) if count > 0]

        # Generate recommendations
        recommendations = self._generate_recommendations(rate, recurring, failure_count)

        return GapReport(
            total_tasks=total,
            total_failures=failure_count,
            failure_rate=rate,
            recurring_keywords=recurring,
            failure_summaries=summaries[:10],  # Limit to latest 10
            recommendations=recommendations,
        )

    def _load_logs(self) -> List[Dict[str, object]]:
        if not self._log_dir.exists():
            return []
        logs = []
        for path in sorted(self._log_dir.glob("*.json")):
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                logs.append(entry)
            except (json.JSONDecodeError, OSError):
                logger.debug("Skipping malformed evo-log %s", path, exc_info=True)
                continue
        return logs

    def _extract_keywords(self, summaries: List[str]) -> Counter:
        counter: Counter = Counter()
        combined = " ".join(summaries).lower()
        for keyword in self.FAILURE_KEYWORDS:
            count = combined.count(keyword)
            if count > 0:
                counter[keyword] = count
        return counter

    def _generate_recommendations(self, rate: float, recurring: List[str], failure_count: int) -> List[str]:
        recs: List[str] = []

        if rate == 0:
            recs.append("All tasks succeeded. No action needed.")
            return recs

        if rate > 0.5:
            recs.append(f"High failure rate ({rate:.0%}). Consider reviewing the overall pipeline configuration.")

        if "timeout" in recurring:
            recs.append("Recurring timeouts detected. Consider increasing timeout values or checking network stability.")
        if "missing" in recurring or "not found" in recurring:
            recs.append("Missing resources detected. Ensure source URLs and paths are valid before fetching.")
        if "permission" in recurring or "blocked" in recurring:
            recs.append("Permission issues detected. Verify token/credential configuration.")
        if "rate limit" in recurring:
            recs.append("Rate limiting detected. Add delay between requests or use backoff strategy.")
        if "parse" in recurring or "encoding" in recurring:
            recs.append("Parsing/encoding errors detected. Consider adding robust error handling for source content.")

        if failure_count > 0 and not recs:
            recs.append(f"{failure_count} failures detected. Review failure summaries for specific issues.")

        return recs
