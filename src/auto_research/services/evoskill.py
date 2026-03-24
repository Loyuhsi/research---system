"""EvoSkill loop — log outcomes, propose/validate/promote candidate skills."""

from __future__ import annotations

import json
import logging
import shutil
import time

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Dict, Optional

from ..exceptions import PolicyError, ExecutionError
from ..runtime import AutoResearchConfig

# Default minimum absolute score. Config-driven via EVO_MIN_SCORE env var.
DEFAULT_EVO_MIN_SCORE = 0.6


class EvoSkillService:
    """Self-evolution loop: log → propose → validate → promote."""

    def __init__(self, config: AutoResearchConfig, min_score: Optional[float] = None) -> None:
        self.config = config
        if min_score is not None:
            self.min_score = min_score
        else:
            # Allow config-driven override via env
            raw = config.env_values.get("EVO_MIN_SCORE", "")
            try:
                self.min_score = float(raw) if raw else DEFAULT_EVO_MIN_SCORE
            except ValueError:
                self.min_score = DEFAULT_EVO_MIN_SCORE

    def evo_log(self, task_id: str, status: str, summary: str) -> Dict[str, object]:
        log_dir = self.config.repo_root / "knowledge" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{task_id}.json"
        payload = {
            "task_id": task_id,
            "status": status,
            "summary": summary,
            "recorded_at": int(time.time()),
            "gap_detected": status.lower() != "success",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"log_path": str(path), "gap_detected": payload["gap_detected"]}

    def evo_propose(self, task_id: str, candidate_name: str, prompt: str) -> Dict[str, object]:
        candidate_dir = self.config.repo_root / "staging" / "skills-candidates" / candidate_name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        meta_path = candidate_dir / "candidate.json"
        skill_path = candidate_dir / "SKILL.md"
        meta_path.write_text(
            json.dumps({"task_id": task_id, "candidate_name": candidate_name, "prompt": prompt}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        skill_path.write_text(
            "\n".join(
                [
                    "---",
                    f"name: {candidate_name}",
                    "description: Candidate skill proposed by EvoSkill validator.",
                    "---",
                    "",
                    f"# {candidate_name}",
                    "",
                    prompt,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return {"candidate_dir": str(candidate_dir)}

    def evo_validate(self, candidate_name: str, baseline_score: float, candidate_score: float) -> Dict[str, object]:

        candidate_dir = self.config.repo_root / "staging" / "skills-candidates" / candidate_name
        if not candidate_dir.exists():
            raise ExecutionError(f"Missing candidate skill: {candidate_name}")

        reject_reason = ""
        if candidate_score < self.min_score:
            reject_reason = f"Score {candidate_score:.2f} below minimum threshold {self.min_score:.2f}"
        elif candidate_score <= baseline_score:
            reject_reason = f"Score {candidate_score:.2f} did not exceed baseline {baseline_score:.2f}"

        passed = not reject_reason
        validation_path = candidate_dir / "validation.json"
        result: Dict[str, object] = {
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "min_score": self.min_score,
            "passed": passed,
        }
        if reject_reason:
            result["reject_reason"] = reject_reason
        validation_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Write evaluation record
        try:
            from ..evaluation import EvaluationRecord, EvaluationStore
            eval_store = EvaluationStore(self.config.repo_root / "knowledge" / "evaluations")
            eval_record = EvaluationRecord(
                eval_id=f"evo-{candidate_name}-{int(time.time())}",
                eval_type="evo_validate",
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                passed=passed,
                artifact_paths=[str(validation_path)],
                metadata={"candidate": candidate_name, "min_score": self.min_score},
            )
            eval_path = eval_store.save(eval_record)
            regression = eval_store.check_regression("evo_validate", candidate_score)
            result["eval_path"] = str(eval_path)
            result["regression_check"] = regression
        except Exception:
            logger.debug("Evaluation record failed for %s", candidate_name, exc_info=True)
        return {"validation_path": str(validation_path), "passed": passed, **({} if passed else {"reject_reason": reject_reason})}

    def evo_promote(self, candidate_name: str, approved: bool = False) -> Dict[str, object]:

        if not approved:
            raise PolicyError("Skill promotion requires explicit approval.")
        candidate_dir = self.config.repo_root / "staging" / "skills-candidates" / candidate_name
        validation_path = candidate_dir / "validation.json"
        if not validation_path.exists():
            raise ExecutionError("Candidate skill has not been validated.")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not validation.get("passed"):
            raise PolicyError("Candidate skill failed validation and cannot be promoted.")
        target_dir = self.config.repo_root / "skills" / candidate_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(candidate_dir, target_dir)
        return {"skill_dir": str(target_dir)}

    def list_logs(self, only_failures: bool = False) -> list[Dict[str, object]]:
        """List all evo-log entries; optionally filter to failures only."""
        log_dir = self.config.repo_root / "knowledge" / "logs"
        if not log_dir.exists():
            return []
        logs = []
        for path in sorted(log_dir.glob("*.json")):
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                if only_failures and not entry.get("gap_detected", False):
                    continue
                logs.append(entry)
            except (json.JSONDecodeError, OSError):
                logger.debug("Skipping malformed evo-log %s", path, exc_info=True)
                continue
        return logs
