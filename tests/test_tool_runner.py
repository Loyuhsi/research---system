"""Tests for services/tool_runner.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_research.services.tool_runner import ToolRunnerService, ALLOWED_FORMATS
from auto_research.runtime import load_config
from conftest import make_temp_repo


class TestToolRunnerService:
    @pytest.fixture
    def runner(self, tmp_repo_config):
        return ToolRunnerService(tmp_repo_config)

    def test_dry_run_returns_command(self, runner, tmp_repo_config):
        # Create a source file
        source = tmp_repo_config.repo_root / "staging" / "tooling" / "test.txt"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("hello", encoding="utf-8")
        output = tmp_repo_config.repo_root / "staging" / "tooling" / "test.pdf"

        result = runner.run_tool_binding(
            "libreoffice-convert", source, output,
            mode="semi_trusted_tooling", dry_run=True,
        )
        assert "command" in result
        assert isinstance(result["command"], list)

    def test_blocked_binding_raises(self, runner):
        from auto_research.orchestrator import PolicyError
        with pytest.raises(PolicyError, match="Blocked"):
            runner.run_tool_binding(
                "nonexistent", Path("/tmp/a"), Path("/tmp/b"),
                mode="research_only",
            )

    def test_wrong_mode_raises(self, runner, tmp_repo_config):
        from auto_research.orchestrator import PolicyError
        source = tmp_repo_config.repo_root / "staging" / "tooling" / "x.txt"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("x", encoding="utf-8")
        output = tmp_repo_config.repo_root / "staging" / "tooling" / "x.pdf"
        with pytest.raises(PolicyError, match="not permitted"):
            runner.run_tool_binding(
                "libreoffice-convert", source, output,
                mode="research_only",
            )

    def test_unsupported_format_raises(self, runner, tmp_repo_config):
        from auto_research.orchestrator import PolicyError
        source = tmp_repo_config.repo_root / "staging" / "tooling" / "x.txt"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("x", encoding="utf-8")
        output = tmp_repo_config.repo_root / "staging" / "tooling" / "x.exe"
        with pytest.raises(PolicyError, match="Unsupported"):
            runner.run_tool_binding(
                "libreoffice-convert", source, output,
                mode="semi_trusted_tooling", fmt="exe",
            )

    def test_source_not_found_raises(self, runner, tmp_repo_config):
        from auto_research.orchestrator import ExecutionError
        source = tmp_repo_config.repo_root / "staging" / "tooling" / "missing.txt"
        output = tmp_repo_config.repo_root / "staging" / "tooling" / "out.pdf"
        with pytest.raises(ExecutionError, match="does not exist"):
            runner.run_tool_binding(
                "libreoffice-convert", source, output,
                mode="semi_trusted_tooling",
            )

    def test_output_outside_roots_raises(self, runner, tmp_repo_config):
        from auto_research.orchestrator import PolicyError
        source = tmp_repo_config.repo_root / "staging" / "tooling" / "x.txt"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("x", encoding="utf-8")
        output = Path("/tmp/evil/out.pdf")
        with pytest.raises(PolicyError, match="outside"):
            runner.run_tool_binding(
                "libreoffice-convert", source, output,
                mode="semi_trusted_tooling",
            )

    def test_rd_agent_dry_run(self, runner):
        result = runner.run_rd_agent("high_risk_execution", dry_run=True)
        assert "command" in result
        assert "docker" in result["command"][0]

    def test_rd_agent_wrong_mode_raises(self, runner):
        from auto_research.orchestrator import PolicyError
        with pytest.raises(PolicyError, match="high_risk_execution"):
            runner.run_rd_agent("research_only")

    def test_allowed_formats_contains_common(self):
        assert "pdf" in ALLOWED_FORMATS
        assert "html" in ALLOWED_FORMATS
        assert "exe" not in ALLOWED_FORMATS
