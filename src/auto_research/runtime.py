from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, cast


DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_LMSTUDIO_BASE = "http://127.0.0.1:1234"
DEFAULT_VLLM_BASE = "http://127.0.0.1:8000"
DEFAULT_LLAMACPP_BASE = "http://127.0.0.1:8080"
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_EMBEDDING_PROVIDER = "lmstudio"
DEFAULT_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
DEFAULT_MEMORY_TTL_DAYS = 28
DEFAULT_VECTOR_BACKEND = "sqlite-vec"
DEFAULT_MAX_HISTORY_ROUNDS = 12
DEFAULT_MAX_SOURCE_BYTES = 102400


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolBinding:
    name: str
    description: str
    command: tuple[str, ...]
    allowed_modes: frozenset[str]
    allowed_output_roots: tuple[Path, ...]
    pass_env: frozenset[str]


@dataclass(frozen=True)
class RuntimeMode:
    name: str
    description: str
    frontend_allowlist: frozenset[str]
    tool_scopes: frozenset[str]


@dataclass(frozen=True)
class Zone:
    name: str
    roots: tuple[Path, ...]
    localhost_allowlist: tuple[str, ...]
    secrets_visible: bool


@dataclass(frozen=True)
class ArtifactLayout:
    session_id: str
    research_root: Path
    raw_dir: Path
    parsed_dir: Path
    status_path: Path
    note_path: Path
    legacy_sources_dir: Path

    def ensure(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.note_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AutoResearchConfig:
    repo_root: Path
    env_values: Dict[str, str]
    runtime_modes: Dict[str, RuntimeMode]
    zones: Dict[str, Zone]
    tool_bindings: Dict[str, ToolBinding]
    secret_env_allowlist: frozenset[str]
    tool_env_allowlist: Dict[str, frozenset[str]]
    provider: str
    model: str
    ollama_base: str
    lmstudio_base: str
    lmstudio_api_key: str
    vllm_base: str
    vllm_api_key: str
    llamacpp_base: str
    vault_root: Optional[Path]
    vault_subdir: str
    skill_memory_embedding_provider: str
    skill_memory_embedding_model: str
    skill_memory_ttl_days: int
    skill_memory_vector_backend: str
    max_history_rounds: int
    max_source_bytes: int
    wsl_distro: str = "Ubuntu-24.04"
    proxy_url: str = ""

    def resolve_layout(self, session_id: str) -> ArtifactLayout:
        research_root = self.repo_root / "output" / "research" / session_id
        return ArtifactLayout(
            session_id=session_id,
            research_root=research_root,
            raw_dir=research_root / "raw",
            parsed_dir=research_root / "parsed",
            status_path=research_root / "status.json",
            note_path=self.repo_root / "output" / "notes" / f"{session_id}.md",
            legacy_sources_dir=self.repo_root / "output" / "sources" / session_id,
        )

    def safe_env(self, extra_allowlist: Optional[Iterable[str]] = None) -> Dict[str, str]:
        allowlist = set(self.secret_env_allowlist)
        if extra_allowlist:
            allowlist.update(extra_allowlist)

        safe: Dict[str, str] = {}
        for name in sorted(allowlist):
            value = os.environ.get(name) or self.env_values.get(name)
            if value:
                safe[name] = value
        return safe

    def selected_base_url(self) -> str:
        if self.provider == "ollama":
            return self.ollama_base
        if self.provider == "vllm":
            return self.vllm_base
        if self.provider == "llamacpp":
            return self.llamacpp_base
        return self.lmstudio_base

    def selected_chat_url(self) -> str:
        return f"{self.selected_base_url()}/v1/chat/completions"

    @property
    def memory_records_dir(self) -> Path:
        return self.repo_root / "knowledge" / "memory-records"

    @property
    def memory_index_path(self) -> Path:
        return self.repo_root / "knowledge" / "index" / "skill-memory.db"

    @property
    def memory_drafts_dir(self) -> Path:
        return self.repo_root / "staging" / "memory-drafts"

    @property
    def github_skills_dir(self) -> Path:
        return self.repo_root / ".github" / "skills"


def parse_key_value_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value

    return values


def read_setting(
    name: str,
    env_file_values: Mapping[str, str],
    environ: Mapping[str, str],
    default: str = "",
) -> str:
    return (environ.get(name) or env_file_values.get(name) or default).strip()


def normalize_base_url(value: str, fallback: str) -> str:
    normalized = (value or fallback).strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[: -len("/v1")]
    return normalized


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def _sub_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Safely extract a sub-dict from a parsed JSON dict."""
    value = parent.get(key, {})
    if isinstance(value, dict):
        return value
    return {}


def _resolve_repo_path(repo_root: Path, relative: str) -> Path:
    return (repo_root / relative).resolve()


def load_config(
    repo_root: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
    strict: bool = False,
) -> AutoResearchConfig:
    repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    environ = environ or os.environ

    env_values = parse_key_value_file(repo_root / ".env")
    runtime_values = parse_key_value_file(repo_root / ".runtime" / "service-endpoints.env")

    runtime_modes_raw = load_json(repo_root / "config" / "runtime-modes.json")
    zones_raw = load_json(repo_root / "config" / "zones.json")
    tool_allowlist_raw = load_json(repo_root / "config" / "tool-allowlist.json")

    runtime_modes: Dict[str, RuntimeMode] = {}
    for name, data in _sub_dict(runtime_modes_raw, "modes").items():
        runtime_modes[name] = RuntimeMode(
            name=name,
            description=str(data.get("description", "")).strip(),
            frontend_allowlist=frozenset(data.get("frontend_allowlist", [])),
            tool_scopes=frozenset(data.get("tool_scopes", [])),
        )

    if "research_only" not in runtime_modes:
        raise ConfigError("config/runtime-modes.json must define research_only")

    zones: Dict[str, Zone] = {}
    zones_section = _sub_dict(zones_raw, "zones")
    for name, data in zones_section.items():
        zones[name] = Zone(
            name=name,
            roots=tuple(_resolve_repo_path(repo_root, item) for item in data.get("roots", [])),
            localhost_allowlist=tuple(str(item) for item in data.get("localhost_allowlist", [])),
            secrets_visible=bool(data.get("secrets_visible", False)),
        )

    tool_bindings: Dict[str, ToolBinding] = {}
    for name, data in _sub_dict(tool_allowlist_raw, "bindings").items():
        tool_bindings[name] = ToolBinding(
            name=name,
            description=str(data.get("description", "")).strip(),
            command=tuple(str(part) for part in data.get("command", [])),
            allowed_modes=frozenset(data.get("allowed_modes", [])),
            allowed_output_roots=tuple(_resolve_repo_path(repo_root, item) for item in data.get("allowed_output_roots", [])),
            pass_env=frozenset(data.get("pass_env", [])),
        )

    provider = read_setting("TELEGRAM_PROVIDER", env_values, environ, DEFAULT_PROVIDER).lower()
    if provider not in {"ollama", "lmstudio", "vllm", "llamacpp"}:
        raise ConfigError("TELEGRAM_PROVIDER must be one of: ollama, lmstudio, vllm, llamacpp.")

    ollama_base = normalize_base_url(
        runtime_values.get("OLLAMA_BASE") or read_setting("OLLAMA_BASE", env_values, environ),
        DEFAULT_OLLAMA_BASE,
    )
    lmstudio_base = normalize_base_url(
        runtime_values.get("LMSTUDIO_BASE") or read_setting("LMSTUDIO_BASE", env_values, environ),
        DEFAULT_LMSTUDIO_BASE,
    )
    vllm_base = normalize_base_url(
        runtime_values.get("VLLM_BASE") or read_setting("VLLM_BASE", env_values, environ),
        DEFAULT_VLLM_BASE,
    )
    llamacpp_base = normalize_base_url(
        runtime_values.get("LLAMACPP_BASE") or read_setting("LLAMACPP_BASE", env_values, environ),
        DEFAULT_LLAMACPP_BASE,
    )

    vault_root_value = read_setting("VAULT_ROOT", env_values, environ)
    if vault_root_value:
        vault_root = Path(vault_root_value).expanduser()
    else:
        vault_root = (Path.home() / "Documents" / "AutoResearchVault").resolve()

    vault_subdir = str(_sub_dict(zones_raw, "obsidian").get("required_subdir", "10_Research/AutoResearch"))
    tool_env_allowlist = {
        key: frozenset(value)
        for key, value in _sub_dict(zones_raw, "tool_env_allowlist").items()
    }

    embedding_provider = read_setting(
        "SKILL_MEMORY_EMBEDDING_PROVIDER",
        env_values,
        environ,
        DEFAULT_EMBEDDING_PROVIDER,
    ).lower()
    embedding_model = read_setting(
        "SKILL_MEMORY_EMBEDDING_MODEL",
        env_values,
        environ,
        DEFAULT_EMBEDDING_MODEL,
    )
    ttl_raw = read_setting(
        "SKILL_MEMORY_TTL_DAYS",
        env_values,
        environ,
        str(DEFAULT_MEMORY_TTL_DAYS),
    )
    vector_backend = read_setting(
        "SKILL_MEMORY_VECTOR_BACKEND",
        env_values,
        environ,
        DEFAULT_VECTOR_BACKEND,
    ).lower()
    try:
        skill_memory_ttl_days = max(int(ttl_raw), 1)
    except ValueError as exc:
        raise ConfigError("SKILL_MEMORY_TTL_DAYS must be an integer.") from exc

    max_history_raw = read_setting("MAX_HISTORY_ROUNDS", env_values, environ, str(DEFAULT_MAX_HISTORY_ROUNDS))
    max_source_raw = read_setting("MAX_SOURCE_BYTES", env_values, environ, str(DEFAULT_MAX_SOURCE_BYTES))
    try:
        max_history_rounds = max(int(max_history_raw), 1)
    except ValueError:
        max_history_rounds = DEFAULT_MAX_HISTORY_ROUNDS
    try:
        max_source_bytes = max(int(max_source_raw), 1024)
    except ValueError:
        max_source_bytes = DEFAULT_MAX_SOURCE_BYTES

    proxy_url = read_setting("PROXY_URL", env_values, environ)
    wsl_distro = read_setting("WSL_DISTRO", env_values, environ, "Ubuntu-24.04")

    config = AutoResearchConfig(
        repo_root=repo_root,
        env_values=env_values,
        runtime_modes=runtime_modes,
        zones=zones,
        tool_bindings=tool_bindings,
        secret_env_allowlist=frozenset(str(s) for s in zones_raw.get("secret_env_allowlist", [])),
        tool_env_allowlist=tool_env_allowlist,
        provider=provider,
        model=read_setting("TELEGRAM_MODEL", env_values, environ, DEFAULT_MODEL),
        ollama_base=ollama_base,
        lmstudio_base=lmstudio_base,
        lmstudio_api_key=read_setting("LMSTUDIO_API_KEY", env_values, environ),
        vllm_base=vllm_base,
        vllm_api_key=read_setting("VLLM_API_KEY", env_values, environ),
        llamacpp_base=llamacpp_base,
        vault_root=vault_root,
        vault_subdir=vault_subdir,
        skill_memory_embedding_provider=embedding_provider,
        skill_memory_embedding_model=embedding_model,
        skill_memory_ttl_days=skill_memory_ttl_days,
        skill_memory_vector_backend=vector_backend,
        max_history_rounds=max_history_rounds,
        max_source_bytes=max_source_bytes,
        wsl_distro=wsl_distro,
        proxy_url=proxy_url,
    )

    if strict:
        from .config_schema import ConfigValidator, ConfigValidationError
        validator = ConfigValidator()
        report = validator.validate_all(
            config, runtime_modes_raw, zones_raw, tool_allowlist_raw,
        )
        if report.has_errors:
            raise ConfigValidationError(report.errors)

    return config
