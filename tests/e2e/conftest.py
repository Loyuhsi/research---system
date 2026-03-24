"""E2E test fixtures — requires a running Ollama or LM Studio instance.

These tests are opt-in: excluded from default pytest runs via the 'e2e' marker.
Run explicitly with: pytest -m e2e
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Provider availability detection
# ---------------------------------------------------------------------------

def _ollama_available(base: str = "http://127.0.0.1:11434") -> bool:
    try:
        req = urllib.request.Request(f"{base}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return isinstance(data.get("models"), list) and len(data["models"]) > 0
    except Exception:
        return False


def _get_ollama_model(base: str = "http://127.0.0.1:11434") -> str:
    req = urllib.request.Request(f"{base}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
        return str(data["models"][0]["name"])


def _lmstudio_available(base: str = "http://127.0.0.1:1234") -> bool:
    try:
        req = urllib.request.Request(f"{base}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return isinstance(data.get("data"), list) and len(data["data"]) > 0
    except Exception:
        return False


def _get_lmstudio_model(base: str = "http://127.0.0.1:1234") -> str:
    req = urllib.request.Request(f"{base}/v1/models", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
        return str(data["data"][0]["id"])


_OLLAMA_UP = _ollama_available()
_LMSTUDIO_UP = _lmstudio_available()
_SKIP_OLLAMA = "Ollama not available or no models loaded"
_SKIP_LMSTUDIO = "LM Studio not available or no models loaded"

skip_no_ollama = pytest.mark.skipif(not _OLLAMA_UP, reason=_SKIP_OLLAMA)
skip_no_lmstudio = pytest.mark.skipif(not _LMSTUDIO_UP, reason=_SKIP_LMSTUDIO)
skip_no_provider = pytest.mark.skipif(
    not _OLLAMA_UP and not _LMSTUDIO_UP,
    reason="No LLM provider available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def e2e_repo(tmp_path: Path) -> Path:
    """Create a minimal temp repo for E2E tests."""
    from conftest import make_temp_repo
    return make_temp_repo(str(tmp_path))


@pytest.fixture
def e2e_config(e2e_repo: Path):
    """Load config with real provider (prefer Ollama, fallback to LM Studio)."""
    from auto_research.runtime import load_config
    if _OLLAMA_UP:
        model = _get_ollama_model()
        provider = "ollama"
    elif _LMSTUDIO_UP:
        model = _get_lmstudio_model()
        provider = "lmstudio"
    else:
        model = "test"
        provider = "ollama"
    return load_config(
        repo_root=e2e_repo,
        environ={
            "TELEGRAM_PROVIDER": provider,
            "TELEGRAM_MODEL": model,
            "SKILL_MEMORY_EMBEDDING_PROVIDER": "disabled",
        },
    )


@pytest.fixture
def e2e_orchestrator(e2e_config):
    """Build orchestrator with real HTTP client (no mocks)."""
    from auto_research.http_client import JsonHttpClient
    from auto_research.registry import ServiceRegistry
    from auto_research.events import EventBus
    from auto_research.services.llm_provider import LlmProviderService
    from auto_research.services.fetcher import FetcherService
    from auto_research.services.vault import VaultService
    from auto_research.conversation_store import InMemoryConversationStore
    from auto_research.orchestrator import Orchestrator

    registry = ServiceRegistry()
    event_bus = EventBus()
    http_client = JsonHttpClient()
    llm = LlmProviderService(e2e_config)
    vault = VaultService(e2e_config)
    fetcher = FetcherService(e2e_config, vault)

    registry.register("core.config", e2e_config)
    registry.register("core.events", event_bus)
    registry.register("core.http", http_client)

    return Orchestrator(
        config=e2e_config,
        event_bus=event_bus,
        registry=registry,
        llm_service=llm,
        fetcher_service=fetcher,
        vault_service=vault,
        http_client=http_client,
        conversation_store=InMemoryConversationStore(),
    )


# ---------------------------------------------------------------------------
# Session-scoped E2E report
# ---------------------------------------------------------------------------

_e2e_results: List[Dict[str, Any]] = []


@pytest.fixture(autouse=True)
def _record_e2e_result(request: pytest.FixtureRequest):
    """Record each E2E test result for the report."""
    t0 = time.monotonic()
    yield
    elapsed = round((time.monotonic() - t0) * 1000, 1)
    outcome = "passed"
    if hasattr(request.node, "rep_call"):
        if request.node.rep_call.failed:  # type: ignore[attr-defined]
            outcome = "failed"
        elif request.node.rep_call.skipped:  # type: ignore[attr-defined]
            outcome = "skipped_by_env"
    elif not _OLLAMA_UP and not _LMSTUDIO_UP:
        outcome = "skipped_by_env"
    _e2e_results.append({
        "name": request.node.nodeid,
        "status": outcome,
        "latency_ms": elapsed,
    })


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write E2E report after all tests complete."""
    if not _e2e_results:
        return
    report_dir = Path("output")
    report_dir.mkdir(parents=True, exist_ok=True)
    verified = [r for r in _e2e_results if r["status"] == "passed"]
    blocked = [r for r in _e2e_results if r["status"] == "skipped_by_env"]
    failed = [r for r in _e2e_results if r["status"] == "failed"]
    report = {
        "summary": {
            "fully_verified": len(verified),
            "blocked_by_environment": len(blocked),
            "failed": len(failed),
            "total": len(_e2e_results),
        },
        "provider_availability": {
            "ollama": _OLLAMA_UP,
            "lmstudio": _LMSTUDIO_UP,
        },
        "tests": _e2e_results,
    }
    (report_dir / "e2e-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )

