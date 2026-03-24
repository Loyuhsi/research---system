"""Unit tests for SynthesizerService internals — TSV writing and quality scoring."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from conftest import make_temp_repo
from auto_research.runtime import load_config
from auto_research.services.synthesizer import SynthesizerService


@pytest.fixture
def synth(tmp_path):
    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})
    llm = MagicMock()
    http = MagicMock()
    vault = MagicMock()
    return SynthesizerService(config=config, http_client=http, llm=llm, vault_service=vault)


class TestAppendResultsTsv:
    """Verify TSV header migration and row appending."""

    def _quality(self, cov=0.8, struct=0.7, prov=0.6, wc=300, passed=True):
        return MagicMock(
            coverage_score=cov,
            structure_score=struct,
            provenance_score=prov,
            word_count=wc,
            passed=passed,
        )

    def test_creates_header_on_missing_file(self, synth):
        tsv_path = synth.config.repo_root / "results.tsv"
        assert not tsv_path.exists()
        result = synth._append_results_tsv("sess1", "ollama", "qwen", self._quality(), 1.5)
        assert result is True
        content = tsv_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert lines[0].startswith("timestamp")
        assert "trace_id" in lines[0]
        assert len(lines) == 2  # header + 1 data row

    def test_appends_row_to_existing(self, synth):
        synth._append_results_tsv("sess1", "ollama", "qwen", self._quality(), 1.0)
        synth._append_results_tsv("sess2", "ollama", "qwen", self._quality(), 2.0)
        content = (synth.config.repo_root / "results.tsv").read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows

    def test_migrates_old_header(self, synth):
        """If old header lacks trace_id, it gets replaced."""
        tsv_path = synth.config.repo_root / "results.tsv"
        old_header = "timestamp\tsession_id\tprovider\tmodel\n"
        tsv_path.write_text(old_header + "2026\tsess0\tollama\tqwen\n", encoding="utf-8")

        synth._append_results_tsv("sess1", "ollama", "qwen", self._quality(), 1.0)
        content = tsv_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # New header with trace_id + old data row + new data row
        assert "trace_id" in lines[0]
        assert len(lines) == 3

    def test_returns_false_on_write_error(self, synth, tmp_path):
        """If the append fails after header exists, returns False."""
        tsv_path = synth.config.repo_root / "results.tsv"
        # Create a valid TSV first, then make it unwritable
        synth._append_results_tsv("sess0", "ollama", "qwen", self._quality(), 0.5)
        assert tsv_path.exists()
        # Simulate write failure by patching open
        from unittest.mock import patch
        with patch("builtins.open", side_effect=OSError("disk full")):
            result = synth._append_results_tsv("sess1", "ollama", "qwen", self._quality(), 1.0)
        assert result is False


class TestProgramHash:
    def test_stable_hash(self, synth):
        prog_path = synth.config.repo_root / "research_program.md"
        prog_path.write_text("# My Program\nContent here.\n", encoding="utf-8")
        h1 = synth._program_hash()
        h2 = synth._program_hash()
        assert h1 == h2
        assert len(h1) == 12

    def test_no_program_returns_empty(self, synth):
        assert synth._program_hash() == ""
