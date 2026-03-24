"""Tests for GapDetector and GapReport."""
from __future__ import annotations

import json

import pytest

from conftest import make_temp_repo
from auto_research.runtime import load_config
from auto_research.reflection.gap_detector import GapDetector, GapReport


@pytest.fixture
def detector(tmp_path):
    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})
    return GapDetector(config)


class TestGapReport:
    def test_defaults(self):
        report = GapReport()
        assert report.total_tasks == 0
        assert report.has_gaps is False

    def test_has_gaps_true(self):
        report = GapReport(total_failures=3)
        assert report.has_gaps is True

    def test_to_dict_shape(self):
        report = GapReport(total_tasks=10, total_failures=2, failure_rate=0.2)
        d = report.to_dict()
        assert d["total_tasks"] == 10
        assert d["failure_rate"] == 0.2
        assert isinstance(d["recurring_keywords"], list)
        assert isinstance(d["recommendations"], list)


class TestGapDetector:
    def test_empty_log_dir(self, detector):
        report = detector.scan()
        assert report.total_tasks == 0
        assert not report.has_gaps
        assert any("No evo-logs" in r for r in report.recommendations)

    def test_all_success(self, detector):
        log_dir = detector._log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (log_dir / f"task-{i}.json").write_text(
                json.dumps({"task_id": f"t{i}", "status": "success", "gap_detected": False}),
                encoding="utf-8",
            )
        report = detector.scan()
        assert report.total_tasks == 3
        assert report.total_failures == 0
        assert report.failure_rate == 0.0
        assert any("All tasks succeeded" in r for r in report.recommendations)

    def test_with_failures(self, detector):
        log_dir = detector._log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "ok.json").write_text(
            json.dumps({"task_id": "t1", "status": "success", "gap_detected": False}),
            encoding="utf-8",
        )
        (log_dir / "fail1.json").write_text(
            json.dumps({"task_id": "t2", "status": "failed", "gap_detected": True, "summary": "timeout connecting"}),
            encoding="utf-8",
        )
        (log_dir / "fail2.json").write_text(
            json.dumps({"task_id": "t3", "status": "failed", "gap_detected": True, "summary": "timeout again"}),
            encoding="utf-8",
        )
        report = detector.scan()
        assert report.total_tasks == 3
        assert report.total_failures == 2
        assert abs(report.failure_rate - 2 / 3) < 0.01
        assert "timeout" in report.recurring_keywords
        assert any("timeout" in r.lower() for r in report.recommendations)

    def test_malformed_json_skipped(self, detector):
        log_dir = detector._log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "bad.json").write_text("not json", encoding="utf-8")
        (log_dir / "good.json").write_text(
            json.dumps({"task_id": "t1", "gap_detected": False}),
            encoding="utf-8",
        )
        report = detector.scan()
        assert report.total_tasks == 1

    def test_high_failure_rate_recommendation(self, detector):
        log_dir = detector._log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            (log_dir / f"fail-{i}.json").write_text(
                json.dumps({"task_id": f"t{i}", "gap_detected": True, "summary": "unknown error"}),
                encoding="utf-8",
            )
        report = detector.scan()
        assert report.failure_rate == 1.0
        assert any("High failure rate" in r for r in report.recommendations)
