"""Tests for EvoSkillService — log, propose, validate, promote lifecycle."""

import json
import tempfile
import unittest
from pathlib import Path

from auto_research.orchestrator import ExecutionError, PolicyError
from auto_research.runtime import load_config

from conftest import make_temp_repo, create_test_orchestrator


class EvoSkillServiceTests(unittest.TestCase):

    def _setup(self, tmpdir):
        repo = make_temp_repo(tmpdir)
        config = load_config(repo_root=repo, environ={})
        orch = create_test_orchestrator(config)
        return orch, repo

    def test_evo_log_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            result = orch.evo_log("task-evo-1", "failed", "fetcher timed out")
            log_path = Path(result["log_path"])
            self.assertTrue(log_path.exists())
            payload = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_id"], "task-evo-1")
            self.assertEqual(payload["status"], "failed")
            self.assertTrue(payload["gap_detected"])

    def test_evo_log_success_no_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            result = orch.evo_log("task-ok", "success", "all good")
            self.assertFalse(result["gap_detected"])

    def test_evo_propose_creates_candidate_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            orch.evo_log("task-p", "failed", "issue")
            result = orch.evo_propose("task-p", "new-skill", "Use safer fetch.")
            candidate_dir = Path(result["candidate_dir"])
            self.assertTrue(candidate_dir.exists())
            self.assertTrue((candidate_dir / "candidate.json").exists())
            self.assertTrue((candidate_dir / "SKILL.md").exists())
            skill_content = (candidate_dir / "SKILL.md").read_text(encoding="utf-8")
            self.assertIn("new-skill", skill_content)

    def test_evo_validate_writes_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            orch.evo_log("task-v", "failed", "issue")
            orch.evo_propose("task-v", "val-skill", "prompt")
            result = orch.evo_validate("val-skill", baseline_score=0.4, candidate_score=0.8)
            self.assertTrue(result["passed"])
            validation_path = Path(result["validation_path"])
            self.assertTrue(validation_path.exists())
            data = json.loads(validation_path.read_text(encoding="utf-8"))
            self.assertEqual(data["baseline_score"], 0.4)
            self.assertEqual(data["candidate_score"], 0.8)

    def test_evo_promote_copies_to_skills(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            orch.evo_log("task-promo", "failed", "issue")
            orch.evo_propose("task-promo", "promo-skill", "Fix fetch.")
            orch.evo_validate("promo-skill", baseline_score=0.3, candidate_score=0.9)
            result = orch.evo_promote("promo-skill", approved=True)
            skill_dir = Path(result["skill_dir"])
            self.assertTrue(skill_dir.exists())
            self.assertTrue((skill_dir / "SKILL.md").exists())

    def test_list_logs_filters_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orch, repo = self._setup(tmpdir)
            orch.evo_log("task-success", "success", "ok")
            orch.evo_log("task-fail", "failed", "broken")
            evoskill = orch.registry.resolve("service.evoskill")
            all_logs = evoskill.list_logs()
            self.assertEqual(len(all_logs), 2)
            failures = evoskill.list_logs(only_failures=True)
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["task_id"], "task-fail")


if __name__ == "__main__":
    unittest.main()
