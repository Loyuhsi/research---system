"""Tests for the quality gate service."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from auto_research.runtime import load_config
from auto_research.services.quality_gate import QualityGateService, QualityReport
from conftest import make_temp_repo


class QualityGateTests(unittest.TestCase):
    def _make_service(self):
        tmpdir = tempfile.mkdtemp()
        repo = make_temp_repo(tmpdir)
        config = load_config(repo_root=repo, environ={})
        return QualityGateService(config), repo

    def _write_note(self, repo: Path, content: str) -> Path:
        notes_dir = repo / "output" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        note = notes_dir / "test-note.md"
        note.write_text(content, encoding="utf-8")
        return note

    def _write_source(self, repo: Path, name: str, content: str) -> Path:
        source_dir = repo / "output" / "research" / "test" / "parsed"
        source_dir.mkdir(parents=True, exist_ok=True)
        p = source_dir / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_empty_note_fails(self):
        svc, repo = self._make_service()
        note = self._write_note(repo, "")
        report = svc.check(note, [])
        self.assertFalse(report.passed)
        self.assertEqual(report.word_count, 0)

    def test_short_note_fails_word_count(self):
        svc, repo = self._make_service()
        note = self._write_note(repo, "---\ntopic: test\n---\n\nShort note.")
        report = svc.check(note, [])
        self.assertFalse(report.passed)
        self.assertLess(report.word_count, 200)

    def test_well_structured_note_passes(self):
        svc, repo = self._make_service()
        body = "# Main Topic\n\n" + ("This is detailed content about the research. " * 60)
        body += "\n\n## Sub Section\n\n" + ("More analysis and findings here. " * 40)
        body += "\n\n## 結論\n\nThis is the conclusion of the research.\n"
        content = f"---\ntopic: test\n---\n\n{body}"

        source_text = ("This is detailed content about the research. " * 60) + ("More analysis and findings here. " * 40)
        src = self._write_source(repo, "source1.md", source_text)

        note = self._write_note(repo, content)
        report = svc.check(note, [src])
        self.assertTrue(report.passed)
        self.assertGreater(report.structure_score, 0.4)
        self.assertGreater(report.word_count, 200)

    def test_no_sources_low_coverage(self):
        svc, repo = self._make_service()
        body = "# Topic\n\n" + ("Generated content without sources. " * 80) + "\n\n## 結論\n\nDone.\n"
        note = self._write_note(repo, f"---\ntopic: test\n---\n\n{body}")
        report = svc.check(note, [])
        self.assertEqual(report.coverage_score, 0.0)

    def test_unreadable_note_returns_flag(self):
        svc, repo = self._make_service()
        fake_path = repo / "nonexistent.md"
        report = svc.check(fake_path, [])
        self.assertFalse(report.passed)
        self.assertTrue(len(report.hallucination_flags) > 0)

    def test_structure_score_no_frontmatter(self):
        svc, repo = self._make_service()
        content = "Just plain text without any structure.\n" * 50
        note = self._write_note(repo, content)
        report = svc.check(note, [])
        self.assertLess(report.structure_score, 0.4)

    def test_to_dict(self):
        report = QualityReport(
            coverage_score=0.8567,
            hallucination_flags=["flag1"],
            structure_score=0.9,
            word_count=500,
            passed=True,
        )
        d = report.to_dict()
        self.assertEqual(d["coverage_score"], 0.857)
        self.assertEqual(d["word_count"], 500)
        self.assertTrue(d["passed"])


class QualityGateConfigTests(unittest.TestCase):
    """Tests for config-driven thresholds."""

    def _make_service_with_env(self, env_lines: dict):
        """Create a service with extra lines appended to .env."""
        tmpdir = tempfile.mkdtemp()
        repo = make_temp_repo(tmpdir)
        if env_lines:
            env_path = repo / ".env"
            existing = env_path.read_text(encoding="utf-8")
            extra = "\n".join(f"{k}={v}" for k, v in env_lines.items())
            env_path.write_text(existing + "\n" + extra, encoding="utf-8")
        config = load_config(repo_root=repo, environ={})
        return QualityGateService(config)

    def test_defaults_when_no_env(self):
        svc = self._make_service_with_env({})
        self.assertEqual(svc.min_word_count, 200)
        self.assertAlmostEqual(svc.coverage_threshold, 0.3)
        self.assertAlmostEqual(svc.structure_threshold, 0.4)
        self.assertAlmostEqual(svc.pass_threshold, 0.5)

    def test_env_override_min_words(self):
        svc = self._make_service_with_env({"QUALITY_MIN_WORDS": "100"})
        self.assertEqual(svc.min_word_count, 100)

    def test_env_override_coverage_threshold(self):
        svc = self._make_service_with_env({"QUALITY_COVERAGE_THRESHOLD": "0.5"})
        self.assertAlmostEqual(svc.coverage_threshold, 0.5)

    def test_env_override_invalid_falls_back(self):
        svc = self._make_service_with_env({"QUALITY_MIN_WORDS": "not-a-number"})
        self.assertEqual(svc.min_word_count, 200)

    def test_thresholds_property(self):
        svc = self._make_service_with_env({"QUALITY_PASS_THRESHOLD": "0.7"})
        t = svc.thresholds
        self.assertAlmostEqual(t["pass_threshold"], 0.7)
        self.assertEqual(t["min_word_count"], 200)


if __name__ == "__main__":
    unittest.main()
