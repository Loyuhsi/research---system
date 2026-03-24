"""Tests for the reflection layer: GapDetector + StrategyAdvisor."""

from __future__ import annotations

import json
import tempfile
import unittest

from auto_research.reflection.gap_detector import GapDetector, GapReport
from auto_research.reflection.strategy_advisor import StrategyAdvisor
from auto_research.runtime import load_config
from conftest import FakeHttpClient, make_temp_repo
from auto_research.services.llm_provider import LlmProviderService


class GapDetectorTests(unittest.TestCase):
    def _make_detector(self, logs: list[dict] | None = None) -> GapDetector:
        tmpdir = tempfile.mkdtemp()
        repo = make_temp_repo(tmpdir)
        config = load_config(repo_root=repo, environ={})
        detector = GapDetector(config)
        if logs:
            detector._log_dir.mkdir(parents=True, exist_ok=True)
            for i, log in enumerate(logs):
                (detector._log_dir / f"log_{i:03d}.json").write_text(
                    json.dumps(log), encoding="utf-8"
                )
        return detector

    def test_no_logs_returns_empty_report(self):
        detector = self._make_detector()
        report = detector.scan()
        self.assertEqual(report.total_tasks, 0)
        self.assertEqual(report.total_failures, 0)
        self.assertFalse(report.has_gaps)
        self.assertTrue(len(report.recommendations) > 0)

    def test_detects_timeout_and_missing_keywords(self):
        logs = [
            {"gap_detected": True, "summary": "timeout connecting to server"},
            {"gap_detected": True, "summary": "missing resource not found"},
            {"gap_detected": False, "summary": "success"},
        ]
        detector = self._make_detector(logs)
        report = detector.scan()
        self.assertEqual(report.total_tasks, 3)
        self.assertEqual(report.total_failures, 2)
        self.assertIn("timeout", report.recurring_keywords)
        self.assertIn("not found", report.recurring_keywords)

    def test_failure_rate_calculation(self):
        logs = [
            {"gap_detected": True, "summary": "fail"},
            {"gap_detected": False, "summary": "ok"},
            {"gap_detected": False, "summary": "ok"},
            {"gap_detected": False, "summary": "ok"},
        ]
        detector = self._make_detector(logs)
        report = detector.scan()
        self.assertAlmostEqual(report.failure_rate, 0.25, places=2)

    def test_recurring_keywords_no_duplicates(self):
        logs = [
            {"gap_detected": True, "summary": "timeout timeout timeout"},
        ]
        detector = self._make_detector(logs)
        report = detector.scan()
        # "timeout" should appear once in recurring_keywords
        self.assertEqual(report.recurring_keywords.count("timeout"), 1)


class StrategyAdvisorTests(unittest.TestCase):
    def _make_advisor(self, responses: list | None = None) -> StrategyAdvisor:
        tmpdir = tempfile.mkdtemp()
        repo = make_temp_repo(tmpdir)
        config = load_config(repo_root=repo, environ={})
        http = FakeHttpClient(responses=responses or [])
        llm = LlmProviderService(config)
        return StrategyAdvisor(config, http, llm)

    def test_no_gaps_returns_rule_based(self):
        advisor = self._make_advisor()
        report = GapReport()
        result = advisor.advise(report)
        self.assertFalse(result["has_gaps"])
        self.assertEqual(result["source"], "rule-based")

    def test_llm_success_returns_llm_advice(self):
        llm_response = {
            "choices": [{"message": {"content": "Here is my advice"}}]
        }
        advisor = self._make_advisor(responses=[llm_response])
        report = GapReport(total_tasks=10, total_failures=5, failure_rate=0.5,
                           recurring_keywords=["timeout"],
                           failure_summaries=["timeout on fetch"],
                           recommendations=["Check network"])
        result = advisor.advise(report)
        self.assertTrue(result["has_gaps"])
        self.assertEqual(result["source"], "llm")
        self.assertIn("advice", result["advice"])

    def test_llm_failure_falls_back_to_rules(self):
        advisor = self._make_advisor(responses=[{}])  # empty = no content
        report = GapReport(total_tasks=5, total_failures=3, failure_rate=0.6,
                           recurring_keywords=["timeout"],
                           failure_summaries=["timeout"],
                           recommendations=["Check network"])
        result = advisor.advise(report)
        self.assertTrue(result["has_gaps"])
        self.assertEqual(result["source"], "rule-based")
        self.assertIn("Check network", result["advice"])

    def test_empty_gap_report_with_gaps_flag(self):
        advisor = self._make_advisor(responses=[{}])
        report = GapReport(total_tasks=1, total_failures=1, failure_rate=1.0,
                           failure_summaries=["unknown error"],
                           recommendations=["Review logs"])
        result = advisor.advise(report)
        self.assertTrue(result["has_gaps"])


class GapReportTests(unittest.TestCase):
    def test_to_dict_structure(self):
        report = GapReport(total_tasks=10, total_failures=3, failure_rate=0.3,
                           recurring_keywords=["timeout"],
                           failure_summaries=["fail1"],
                           recommendations=["fix it"])
        d = report.to_dict()
        self.assertEqual(d["total_tasks"], 10)
        self.assertEqual(d["failure_rate"], 0.3)

    def test_has_gaps_false_when_no_failures(self):
        report = GapReport()
        self.assertFalse(report.has_gaps)

    def test_has_gaps_true_when_failures(self):
        report = GapReport(total_failures=1)
        self.assertTrue(report.has_gaps)


if __name__ == "__main__":
    unittest.main()
