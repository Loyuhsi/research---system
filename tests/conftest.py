from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional

import pytest

# Ensure consistent UTF-8 encoding for Chinese strings on Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from auto_research.http_client import JsonHttpClient
from auto_research.runtime import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def copy_repo_config(repo_root: Path) -> None:
    (repo_root / "config").mkdir(parents=True, exist_ok=True)
    for name in ("runtime-modes.json", "zones.json", "tool-allowlist.json"):
        shutil.copy2(REPO_ROOT / "config" / name, repo_root / "config" / name)


def make_temp_repo(tmpdir: str) -> Path:
    repo_root = Path(tmpdir)
    copy_repo_config(repo_root)
    for rel in (
        "output/sources",
        "output/notes",
        "output/research",
        "knowledge/index",
        "knowledge/logs",
        "knowledge/memory-records",
        "staging/skills-candidates",
        "staging/memory-drafts",
        "staging/tooling",
        "sandbox/rd-agent/in",
        "sandbox/rd-agent/out",
        "skills",
        ".github/skills",
    ):
        (repo_root / rel).mkdir(parents=True, exist_ok=True)
    (repo_root / ".env").write_text(
        "\n".join(
            [
                "TELEGRAM_PROVIDER=ollama",
                "TELEGRAM_MODEL=qwen3.5:9b",
                "GITHUB_TOKEN=from-env-file",
                f"VAULT_ROOT={repo_root / 'vault'}",
            ]
        ),
        encoding="utf-8",
    )
    return repo_root


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary repo directory with config and standard directories."""
    repo_root = make_temp_repo(str(tmp_path))
    return repo_root


@pytest.fixture
def tmp_repo_config(tmp_repo):
    """Load config from a temporary repo."""
    return load_config(repo_root=tmp_repo, environ={})


class FakeHttpClient(JsonHttpClient):
    """A fake HTTP client that returns pre-configured responses."""

    def __init__(self, responses: Optional[list] = None):
        super().__init__()
        self._responses = list(responses or [])
        self._call_log: list[dict] = []

    def request_json(
        self,
        method: str,
        url: str,
        payload: Optional[Mapping[str, object]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: int = 30,
    ) -> Mapping[str, object]:
        self._call_log.append({"method": method, "url": url, "payload": payload, "headers": headers, "timeout": timeout})
        if self._responses:
            return self._responses.pop(0)
        return {}

    @property
    def call_log(self) -> list[dict]:
        return self._call_log

@pytest.fixture
def fake_http_client():
    """Provide a FakeHttpClient factory."""
    def _factory(responses=None):
        return FakeHttpClient(responses=responses)
    return _factory

from auto_research.registry import ServiceRegistry
from auto_research.events import EventBus
from auto_research.conversation_store import InMemoryConversationStore
from auto_research.services.fetcher import FetcherService
from auto_research.services.llm_provider import LlmProviderService
from auto_research.services.vault import VaultService
from auto_research.services.synthesizer import SynthesizerService
from auto_research.services.evoskill import EvoSkillService
from auto_research.services.tool_runner import ToolRunnerService
from auto_research.services.skill_memory import SkillMemoryService
from auto_research.services.task_review import TaskReviewService
from auto_research.reflection.strategy_advisor import StrategyAdvisor
from auto_research.orchestrator import Orchestrator

def create_test_orchestrator(config, http_client=None, conversation_store=None):
    registry = ServiceRegistry()
    event_bus = EventBus()
    http_client = http_client or FakeHttpClient()
    conversations = conversation_store or InMemoryConversationStore()
    
    registry.register("core.config", config)
    registry.register("core.events", event_bus)
    registry.register("core.http", http_client)
    
    llm = LlmProviderService(config)
    vault = VaultService(config)
    fetcher = FetcherService(config, vault)
    
    registry.register("service.llm", llm)
    registry.register("service.vault", vault)
    registry.register("service.fetcher", fetcher)
    
    synthesizer = SynthesizerService(config, http_client, llm, vault)
    evoskill = EvoSkillService(config)
    tool_runner = ToolRunnerService(config)
    strategy_advisor = StrategyAdvisor(config, http_client, llm)
    skill_memory = SkillMemoryService(config, http_client, llm)
    task_review = TaskReviewService(config, skill_memory, evoskill, strategy_advisor)
    
    registry.register("service.synthesizer", synthesizer)
    registry.register("service.evoskill", evoskill)
    registry.register("service.tool_runner", tool_runner)
    registry.register("service.strategy_advisor", strategy_advisor)
    registry.register("service.skill_memory", skill_memory)
    registry.register("service.task_review", task_review)
    
    return Orchestrator(
        config=config,
        event_bus=event_bus,
        registry=registry,
        llm_service=llm,
        fetcher_service=fetcher,
        vault_service=vault,
        http_client=http_client,
        conversation_store=conversations
    )
