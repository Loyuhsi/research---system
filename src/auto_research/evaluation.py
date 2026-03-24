"""Minimal evaluation records for experiment tracking.

Append-only local records with baseline vs candidate comparison
and simple regression detection. Not an experiment management system —
just reviewable, comparable, reproducible records.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import datetime as dt
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationRecord:
    """A single evaluation result."""

    eval_id: str
    eval_type: str  # "evo_validate" | "quality_gate"
    baseline_score: float
    candidate_score: float
    passed: bool
    timestamp: str = ""
    artifact_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        if not self.eval_id:
            self.eval_id = f"eval-{uuid.uuid4().hex[:10]}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvaluationRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


class EvaluationStore:
    """Persists evaluation records to knowledge/evaluations/."""

    def __init__(self, eval_dir: Path) -> None:
        self._dir = eval_dir

    def save(self, record: EvaluationRecord) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{record.eval_id}.json"
        path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def load_all(self, eval_type: Optional[str] = None) -> List[EvaluationRecord]:
        if not self._dir.exists():
            return []
        records: List[EvaluationRecord] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                record = EvaluationRecord.from_dict(data)
                if eval_type and record.eval_type != eval_type:
                    continue
                records.append(record)
            except (json.JSONDecodeError, OSError, TypeError):
                logger.debug("Skipping malformed evaluation %s", path, exc_info=True)
                continue
        return records

    def check_regression(
        self, eval_type: str, candidate_score: float, threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Compare candidate against best historical score."""
        historical = self.load_all(eval_type=eval_type)
        passed_records = [r for r in historical if r.passed]
        if not passed_records:
            return {"has_baseline": False, "regression": False}
        best = max(r.candidate_score for r in passed_records)
        regression = candidate_score < (best - threshold)
        return {
            "has_baseline": True,
            "best_historical": best,
            "candidate": candidate_score,
            "regression": regression,
            "delta": round(candidate_score - best, 4),
        }
