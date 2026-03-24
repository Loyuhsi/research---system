"""Tests for config_schema.py — two-layer validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_research.config_schema import (
    ConfigValidationError,
    ConfigValidator,
    RuntimeValidator,
    SchemaValidator,
    ValidationReport,
)
from auto_research.runtime import load_config
from conftest import make_temp_repo


# ---- ValidationReport ----

class TestValidationReport:
    def test_empty_report_has_no_errors(self):
        r = ValidationReport()
        assert not r.has_errors
        assert r.to_dict()["error_count"] == 0

    def test_merge_combines_errors_and_warnings(self):
        r1 = ValidationReport(errors=["e1"], warnings=["w1"])
        r2 = ValidationReport(errors=["e2"], warnings=["w2"])
        r1.merge(r2)
        assert r1.errors == ["e1", "e2"]
        assert r1.warnings == ["w1", "w2"]


# ---- SchemaValidator ----

class TestSchemaValidator:
    def setup_method(self):
        self.v = SchemaValidator()

    def test_valid_runtime_modes(self):
        data = {
            "modes": {
                "research_only": {
                    "frontend_allowlist": ["cli"],
                    "tool_scopes": ["research"],
                }
            }
        }
        r = self.v.validate_runtime_modes(data)
        assert not r.has_errors

    def test_missing_modes_key(self):
        r = self.v.validate_runtime_modes({})
        assert r.has_errors
        assert "must be an object" in r.errors[0]

    def test_missing_research_only(self):
        r = self.v.validate_runtime_modes({"modes": {"other": {}}})
        assert r.has_errors
        assert "research_only" in r.errors[0]

    def test_invalid_frontend_allowlist_type(self):
        data = {"modes": {"research_only": {"frontend_allowlist": "not-a-list", "tool_scopes": []}}}
        r = self.v.validate_runtime_modes(data)
        assert r.has_errors
        assert "frontend_allowlist must be a list" in r.errors[0]

    def test_valid_zones(self):
        data = {
            "zones": {
                "trusted": {"roots": ["src"], "secrets_visible": False}
            }
        }
        r = self.v.validate_zones(data)
        assert not r.has_errors

    def test_zones_missing_key(self):
        r = self.v.validate_zones({})
        assert r.has_errors

    def test_zone_roots_wrong_type(self):
        data = {"zones": {"z": {"roots": "not-a-list"}}}
        r = self.v.validate_zones(data)
        assert r.has_errors
        assert "roots must be a list" in r.errors[0]

    def test_zone_secrets_visible_wrong_type(self):
        data = {"zones": {"z": {"roots": [], "secrets_visible": "yes"}}}
        r = self.v.validate_zones(data)
        assert r.has_errors
        assert "secrets_visible must be a boolean" in r.errors[0]

    def test_valid_tool_allowlist(self):
        data = {"bindings": {"tool1": {"command": ["echo"], "allowed_modes": ["research_only"]}}}
        r = self.v.validate_tool_allowlist(data)
        assert not r.has_errors

    def test_tool_missing_command(self):
        data = {"bindings": {"tool1": {"allowed_modes": []}}}
        r = self.v.validate_tool_allowlist(data)
        assert r.has_errors
        assert "missing 'command'" in r.errors[0]

    def test_empty_bindings_warns(self):
        data = {"bindings": {}}
        r = self.v.validate_tool_allowlist(data)
        assert not r.has_errors
        assert len(r.warnings) == 1


# ---- RuntimeValidator ----

class TestRuntimeValidator:
    def setup_method(self):
        self.v = RuntimeValidator()

    def test_valid_paths(self, tmp_repo_config):
        r = self.v.validate_paths(tmp_repo_config)
        assert not r.has_errors

    def test_missing_repo_root(self, tmp_path):
        class FakeConfig:
            repo_root = tmp_path / "nonexistent"
        r = self.v.validate_paths(FakeConfig())
        assert r.has_errors
        assert "does not exist" in r.errors[0]

    def test_valid_provider_config(self, tmp_repo_config):
        r = self.v.validate_provider_config(tmp_repo_config)
        assert not r.has_errors

    def test_invalid_provider(self, tmp_repo_config):
        # Create a fake config with bad provider
        class FakeConfig:
            provider = "openai"
            model = "gpt-4"
            skill_memory_vector_backend = "sqlite-vec"
            max_source_bytes = 102400
            skill_memory_ttl_days = 28
        r = self.v.validate_provider_config(FakeConfig())
        assert r.has_errors
        assert "provider" in r.errors[0]


# ---- ConfigValidator (combined) ----

class TestConfigValidator:
    def test_validate_all_on_valid_config(self, tmp_repo_config):
        v = ConfigValidator()
        modes = {"modes": {"research_only": {"frontend_allowlist": ["cli"], "tool_scopes": ["research"]}}}
        zones = {"zones": {"trusted": {"roots": [], "secrets_visible": False}}}
        tools = {"bindings": {"t": {"command": ["echo"], "allowed_modes": ["research_only"]}}}
        r = v.validate_all(tmp_repo_config, modes, zones, tools)
        assert not r.has_errors

    def test_validate_all_catches_schema_errors(self, tmp_repo_config):
        v = ConfigValidator()
        modes = {"modes": {}}  # missing research_only
        zones = {"zones": {}}
        tools = {"bindings": {}}
        r = v.validate_all(tmp_repo_config, modes, zones, tools)
        assert r.has_errors


# ---- Integration with load_config ----

class TestLoadConfigStrict:
    def test_strict_mode_passes_with_valid_config(self, tmp_path):
        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={}, strict=True)
        assert config.provider == "ollama"

    def test_strict_mode_fails_on_broken_config(self, tmp_path):
        repo = make_temp_repo(str(tmp_path))
        # Break runtime-modes.json by removing research_only
        modes_path = repo / "config" / "runtime-modes.json"
        modes_path.write_text(json.dumps({"modes": {"other": {"frontend_allowlist": [], "tool_scopes": []}}}), encoding="utf-8")
        # load_config itself checks research_only before reaching validator
        with pytest.raises(Exception):
            load_config(repo_root=repo, environ={}, strict=True)

    def test_non_strict_mode_does_not_raise(self, tmp_path):
        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={}, strict=False)
        assert config is not None
