"""Post-task review and automatic outcome logging for Skill-as-Memory."""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from typing import TYPE_CHECKING, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

from ..runtime import AutoResearchConfig
from ..reflection.gap_detector import GapDetector
from .evoskill import EvoSkillService
from .skill_memory import SkillMemoryService

if TYPE_CHECKING:
    from ..reflection.strategy_advisor import StrategyAdvisor


SLUG_RE = re.compile(r"[^a-z0-9]+")


class TaskReviewService:
    def __init__(
        self,
        config: AutoResearchConfig,
        skill_memory: SkillMemoryService,
        evoskill: EvoSkillService,
        strategy_advisor: StrategyAdvisor,
    ) -> None:
        self.config = config
        self.skill_memory = skill_memory
        self.evoskill = evoskill
        self.strategy_advisor = strategy_advisor

    def record_outcome(
        self,
        task_id: str,
        action: str,
        status: str,
        summary: str,
        session_id: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, object]:
        log_dir = self.config.repo_root / "knowledge" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{task_id}.json"
        payload = {
            "task_id": task_id,
            "action": action,
            "status": status,
            "summary": summary,
            "session_id": session_id,
            "metadata": dict(metadata or {}),
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "gap_detected": status.lower() != "success",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"log_path": str(path), "gap_detected": payload["gap_detected"]}

    def post_task_review(
        self,
        session_id: str,
        status: str,
        action: str = "research_session",
        task_id: Optional[str] = None,
        summary: Optional[str] = None,
        approve_memory: bool = False,
    ) -> Dict[str, object]:
        normalized_action = action or "research_session"
        task_id = task_id or self._task_id(normalized_action, session_id)
        summary = summary or self._default_summary(session_id, normalized_action, status)

        outcome = self.record_outcome(
            task_id=task_id,
            action=normalized_action,
            status=status,
            summary=summary,
            session_id=session_id,
            metadata={"session_id": session_id, "action": normalized_action},
        )

        draft = self.skill_memory.memory_extract(
            session_id=session_id,
            task_type=normalized_action,
            status=status,
            summary_override=summary if status.lower() != "success" else None,
        )
        validation = self.skill_memory.memory_validate(str(draft["draft_id"]), approve=approve_memory)
        result: Dict[str, object] = {
            "task_id": task_id,
            "outcome": outcome,
            "memory_draft": draft,
            "memory_validation": validation,
        }

        if status.lower() != "success":
            gap_report = GapDetector(self.config).scan()
            strategy = self.strategy_advisor.advise(gap_report)
            candidate_name = self._candidate_name(normalized_action, session_id)
            proposal = self.evoskill.evo_propose(task_id, candidate_name, self._build_candidate_prompt(summary, strategy))
            materialized = self.skill_memory.skill_materialize(candidate_name)
            result["gap_report"] = gap_report.to_dict()
            result["strategy"] = strategy
            result["skill_candidate"] = proposal
            result["skill_materialized"] = materialized
            # Save gap detector results as evaluation record
            self._save_gap_evaluation(gap_report)
        elif self._should_materialize_success_skill(normalized_action, validation):
            candidate_name = self._candidate_name(normalized_action, session_id)
            proposal = self.evoskill.evo_propose(
                task_id,
                candidate_name,
                f"Materialize a reusable {normalized_action} skill for session {session_id}. Use the approved memory record as the evidence base.",
            )
            result["skill_candidate"] = proposal
            result["skill_materialized"] = self.skill_memory.skill_materialize(candidate_name)

        # Auto-trigger rule-based discovery suggestions on successful research
        if status.lower() == "success" and normalized_action in {"research_session", "synthesize"}:
            discovery_hints = self._rule_based_discovery(session_id)
            if discovery_hints:
                result["discovery_suggestions"] = discovery_hints

        return result

    def _default_summary(self, session_id: str, action: str, status: str) -> str:
        return f"{action} for session {session_id} finished with status {status}."

    def _task_id(self, action: str, session_id: str) -> str:
        return f"{self._slug(action)}-{self._slug(session_id)}"

    def _candidate_name(self, action: str, session_id: str) -> str:
        return f"{self._slug(action)}-{self._slug(session_id)}-skill"

    def _build_candidate_prompt(self, summary: str, strategy: Mapping[str, object]) -> str:
        advice = str(strategy.get("advice", "")).strip()
        return (
            "Create a reusable mitigation skill from this failure pattern.\n\n"
            f"Failure summary: {summary}\n\n"
            f"Strategy advice:\n{advice}\n"
        )

    def _should_materialize_success_skill(self, action: str, validation: Mapping[str, object]) -> bool:
        if action not in {"synthesize", "research_session"}:
            return False
        return bool(validation.get("valid"))

    def _save_gap_evaluation(self, gap_report) -> None:
        """Persist gap detector results as an evaluation record for tracking."""
        try:
            from ..evaluation import EvaluationRecord, EvaluationStore
            record = EvaluationRecord(
                eval_id="",
                eval_type="gap_detector",
                baseline_score=0.0,
                candidate_score=1.0 - gap_report.failure_rate,
                passed=not gap_report.has_gaps,
                metadata={
                    "total_tasks": gap_report.total_tasks,
                    "total_failures": gap_report.total_failures,
                    "recurring_keywords": gap_report.recurring_keywords,
                },
            )
            store = EvaluationStore(self.config.repo_root / "knowledge" / "evaluations")
            store.save(record)
        except Exception:
            logger.debug("Gap detector evaluation save failed", exc_info=True)

    def _rule_based_discovery(self, session_id: str) -> Optional[list]:
        """Generate rule-based discovery suggestions from existing notes."""
        try:
            notes_dir = self.config.repo_root / "output" / "notes"
            if not notes_dir.exists():
                return None
            topics = []
            for note_path in sorted(notes_dir.glob("*.md")):
                content = note_path.read_text(encoding="utf-8", errors="replace")
                if content.startswith("---"):
                    for line in content.splitlines()[1:]:
                        if line.strip() == "---":
                            break
                        if line.strip().startswith("topic:"):
                            topic = line.split(":", 1)[1].strip().strip('"').strip("'")
                            if topic and topic not in topics:
                                topics.append(topic)
            if len(topics) >= 2:
                return [f"Consider deeper analysis across {len(topics)} topics: {', '.join(topics[:5])}"]
            return None
        except Exception:
            logger.debug("Rule-based discovery failed for %s", session_id, exc_info=True)
            return None

    def _slug(self, value: str) -> str:
        return SLUG_RE.sub("-", value.lower()).strip("-")
