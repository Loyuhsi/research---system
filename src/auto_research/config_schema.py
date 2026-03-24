"""Config validation for Auto-Research — two layers.

Layer 1 (Schema/Structure): validates JSON config file shapes at load time.
Layer 2 (Runtime/Environment): validates that external resources exist / are reachable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

VALID_PROVIDERS = {"ollama", "lmstudio", "vllm"}
VALID_VECTOR_BACKENDS = {"sqlite-vec", "faiss", "numpy"}


@dataclass
class ValidationReport:
    """Collects errors and warnings from validation."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def merge(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_errors": self.has_errors,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class ConfigValidationError(RuntimeError):
    """Raised when schema/structure validation fails in strict mode."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Config validation failed with {len(errors)} error(s): {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# Layer 1: Schema / Structure validation
# ---------------------------------------------------------------------------

class SchemaValidator:
    """Validates the structure and types of raw JSON config data."""

    def validate_runtime_modes(self, data: Mapping[str, Any]) -> ValidationReport:
        report = ValidationReport()
        modes = data.get("modes")
        if not isinstance(modes, dict):
            report.errors.append("runtime-modes.json: 'modes' must be an object")
            return report
        if "research_only" not in modes:
            report.errors.append("runtime-modes.json: must define 'research_only' mode")
        for name, mode_data in modes.items():
            if not isinstance(mode_data, dict):
                report.errors.append(f"runtime-modes.json: mode '{name}' must be an object")
                continue
            if "frontend_allowlist" not in mode_data:
                report.warnings.append(f"runtime-modes.json: mode '{name}' missing 'frontend_allowlist'")
            elif not isinstance(mode_data["frontend_allowlist"], list):
                report.errors.append(f"runtime-modes.json: mode '{name}'.frontend_allowlist must be a list")
            if "tool_scopes" not in mode_data:
                report.warnings.append(f"runtime-modes.json: mode '{name}' missing 'tool_scopes'")
            elif not isinstance(mode_data["tool_scopes"], list):
                report.errors.append(f"runtime-modes.json: mode '{name}'.tool_scopes must be a list")
        return report

    def validate_zones(self, data: Mapping[str, Any]) -> ValidationReport:
        report = ValidationReport()
        zones = data.get("zones")
        if not isinstance(zones, dict):
            report.errors.append("zones.json: 'zones' must be an object")
            return report
        for name, zone_data in zones.items():
            if not isinstance(zone_data, dict):
                report.errors.append(f"zones.json: zone '{name}' must be an object")
                continue
            if "roots" not in zone_data:
                report.warnings.append(f"zones.json: zone '{name}' missing 'roots'")
            elif not isinstance(zone_data["roots"], list):
                report.errors.append(f"zones.json: zone '{name}'.roots must be a list")
            if "secrets_visible" in zone_data and not isinstance(zone_data["secrets_visible"], bool):
                report.errors.append(f"zones.json: zone '{name}'.secrets_visible must be a boolean")
        return report

    def validate_tool_allowlist(self, data: Mapping[str, Any]) -> ValidationReport:
        report = ValidationReport()
        bindings = data.get("bindings")
        if not isinstance(bindings, dict):
            report.errors.append("tool-allowlist.json: 'bindings' must be an object")
            return report
        for name, binding_data in bindings.items():
            if not isinstance(binding_data, dict):
                report.errors.append(f"tool-allowlist.json: binding '{name}' must be an object")
                continue
            if "command" not in binding_data:
                report.errors.append(f"tool-allowlist.json: binding '{name}' missing 'command'")
            elif not isinstance(binding_data["command"], list):
                report.errors.append(f"tool-allowlist.json: binding '{name}'.command must be a list")
            if "allowed_modes" not in binding_data:
                report.warnings.append(f"tool-allowlist.json: binding '{name}' missing 'allowed_modes'")
            elif not isinstance(binding_data["allowed_modes"], list):
                report.errors.append(f"tool-allowlist.json: binding '{name}'.allowed_modes must be a list")
        if not bindings:
            report.warnings.append("tool-allowlist.json: no bindings defined")
        return report


# ---------------------------------------------------------------------------
# Layer 2: Runtime / Environment validation
# ---------------------------------------------------------------------------

class RuntimeValidator:
    """Validates runtime environment conditions (directories, connectivity)."""

    def validate_paths(self, config: Any) -> ValidationReport:
        """Check that essential directories exist."""
        report = ValidationReport()
        repo = config.repo_root
        if not repo.is_dir():
            report.errors.append(f"repo_root does not exist: {repo}")
            return report

        essential_dirs = [
            "config",
            "output",
            "knowledge",
        ]
        for d in essential_dirs:
            if not (repo / d).is_dir():
                report.warnings.append(f"Expected directory missing: {d}")
        return report

    def validate_provider_config(self, config: Any) -> ValidationReport:
        """Check provider/model settings are sensible."""
        report = ValidationReport()
        if config.provider not in VALID_PROVIDERS:
            report.errors.append(
                f"provider '{config.provider}' not in {VALID_PROVIDERS}"
            )
        if not config.model:
            report.errors.append("model is empty")
        if config.skill_memory_vector_backend not in VALID_VECTOR_BACKENDS:
            report.warnings.append(
                f"vector_backend '{config.skill_memory_vector_backend}' not in {VALID_VECTOR_BACKENDS}"
            )
        if config.max_source_bytes < 1024:
            report.warnings.append(f"max_source_bytes={config.max_source_bytes} is very low")
        if config.skill_memory_ttl_days < 1:
            report.errors.append("skill_memory_ttl_days must be >= 1")
        return report

    def validate_connectivity(self, config: Any, timeout: float = 3.0) -> ValidationReport:
        """Best-effort check if LLM provider is reachable. Never raises."""
        report = ValidationReport()
        import urllib.request
        import urllib.error
        base = config.selected_base_url()
        # Ollama: /api/tags; LMStudio: /v1/models
        probe_url = f"{base}/api/tags" if config.provider == "ollama" else f"{base}/v1/models"
        try:
            req = urllib.request.Request(probe_url, method="GET")
            urllib.request.urlopen(req, timeout=timeout)
        except Exception as exc:
            report.warnings.append(f"LLM provider unreachable at {probe_url}: {exc}")
        return report


# ---------------------------------------------------------------------------
# Layer 3: Settings introspection and validation
# ---------------------------------------------------------------------------

@dataclass
class SettingDefinition:
    """One configurable setting with its metadata."""

    key: str
    python_type: type
    default: Any
    env_var: str
    description: str
    validator: Optional[Callable[[Any], bool]] = None


class SettingsSchema:
    """Describes, validates, and displays all Auto-Research settings.

    This is an introspection/validation layer — AutoResearchConfig remains
    the single source of runtime truth.
    """

    DEFINITIONS: List[SettingDefinition] = [
        SettingDefinition("provider", str, "ollama", "TELEGRAM_PROVIDER",
                          "LLM provider", lambda v: v in VALID_PROVIDERS),
        SettingDefinition("model", str, "qwen3.5:9b", "TELEGRAM_MODEL",
                          "LLM model name", lambda v: bool(v)),
        SettingDefinition("ollama_base", str, "http://127.0.0.1:11434", "OLLAMA_BASE",
                          "Ollama API base URL"),
        SettingDefinition("lmstudio_base", str, "http://127.0.0.1:1234", "LMSTUDIO_BASE",
                          "LM Studio API base URL"),
        SettingDefinition("vllm_base", str, "http://127.0.0.1:8000", "VLLM_BASE",
                          "vLLM API base URL"),
        SettingDefinition("skill_memory_embedding_provider", str, "lmstudio",
                          "SKILL_MEMORY_EMBEDDING_PROVIDER", "Embedding provider"),
        SettingDefinition("skill_memory_ttl_days", int, 28, "SKILL_MEMORY_TTL_DAYS",
                          "Memory record TTL in days", lambda v: int(str(v)) >= 1),
        SettingDefinition("max_source_bytes", int, 102400, "MAX_SOURCE_BYTES",
                          "Max source bytes for synthesis", lambda v: int(str(v)) >= 1024),
        SettingDefinition("max_history_rounds", int, 12, "MAX_HISTORY_ROUNDS",
                          "Max conversation history rounds", lambda v: int(str(v)) >= 1),
        SettingDefinition("proxy_url", str, "", "PROXY_URL", "HTTP/SOCKS proxy URL"),
        SettingDefinition("quality_min_words", int, 200, "QUALITY_MIN_WORDS",
                          "Minimum word count for quality gate"),
        SettingDefinition("quality_coverage_threshold", float, 0.3,
                          "QUALITY_COVERAGE_THRESHOLD", "Coverage threshold"),
        SettingDefinition("quality_pass_threshold", float, 0.5,
                          "QUALITY_PASS_THRESHOLD", "Composite pass threshold"),
        SettingDefinition("evo_min_score", float, 0.6, "EVO_MIN_SCORE",
                          "EvoSkill minimum validation score"),
    ]

    def validate_config(self, config: Any) -> ValidationReport:
        """Validate settings with validators defined in DEFINITIONS."""
        report = ValidationReport()
        env = getattr(config, "env_values", {})
        for defn in self.DEFINITIONS:
            # Resolve value: attribute on config -> env override -> skip
            value = getattr(config, defn.key, None)
            if value is None:
                value = env.get(defn.env_var)
            if value is None or value == "":
                continue
            if defn.validator:
                try:
                    if not defn.validator(value):
                        report.warnings.append(
                            f"Setting '{defn.key}' ({defn.env_var}): value '{value}' failed validation"
                        )
                except (ValueError, TypeError):
                    report.warnings.append(
                        f"Setting '{defn.key}' ({defn.env_var}): value '{value}' could not be validated"
                    )
        return report

    def to_dict(self) -> List[Dict[str, Any]]:
        """Return all settings as dicts for doctor/diagnostics display."""
        return [
            {
                "key": d.key,
                "type": d.python_type.__name__,
                "default": d.default,
                "env_var": d.env_var,
                "description": d.description,
                "has_validator": d.validator is not None,
            }
            for d in self.DEFINITIONS
        ]


# ---------------------------------------------------------------------------
# Combined validator
# ---------------------------------------------------------------------------

class ConfigValidator:
    """Runs both schema and runtime validation."""

    def __init__(self) -> None:
        self.schema = SchemaValidator()
        self.runtime = RuntimeValidator()
        self.settings = SettingsSchema()

    def validate_schema(
        self,
        runtime_modes_raw: Mapping[str, Any],
        zones_raw: Mapping[str, Any],
        tool_allowlist_raw: Mapping[str, Any],
    ) -> ValidationReport:
        """Layer 1: structure-only checks on raw JSON data."""
        report = ValidationReport()
        report.merge(self.schema.validate_runtime_modes(runtime_modes_raw))
        report.merge(self.schema.validate_zones(zones_raw))
        report.merge(self.schema.validate_tool_allowlist(tool_allowlist_raw))
        return report

    def validate_runtime(self, config: Any, check_connectivity: bool = False) -> ValidationReport:
        """Layer 2: environment/runtime checks on loaded config."""
        report = ValidationReport()
        report.merge(self.runtime.validate_paths(config))
        report.merge(self.runtime.validate_provider_config(config))
        report.merge(self.settings.validate_config(config))
        if check_connectivity:
            report.merge(self.runtime.validate_connectivity(config))
        return report

    def validate_all(
        self,
        config: Any,
        runtime_modes_raw: Mapping[str, Any],
        zones_raw: Mapping[str, Any],
        tool_allowlist_raw: Mapping[str, Any],
        check_connectivity: bool = False,
    ) -> ValidationReport:
        """Run both layers."""
        report = self.validate_schema(runtime_modes_raw, zones_raw, tool_allowlist_raw)
        report.merge(self.validate_runtime(config, check_connectivity=check_connectivity))
        return report
