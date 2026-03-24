# CLAUDE.md — auto-research

## Project overview

Local auto-research system with LLM synthesis. Fetches public/private sources, synthesizes research notes via local LLM providers (Ollama, LM Studio, vLLM, llama.cpp), manages a skill-as-memory layer, and provides a Telegram control plane.

## Architecture

- `src/auto_research/` — main package
  - `orchestrator.py` — thin facade coordinating service modules
  - `runtime.py` — config loading, dataclasses, layout resolution
  - `exceptions.py` — shared `PolicyError` / `ExecutionError`
  - `services/` — extracted services: fetcher, synthesizer, vault, report, llm_provider, tool_runner, evoskill, quality_gate, skill_memory/
  - `telegram/` — Telegram bot with intent parsing, policy guard, action registry
  - `discovery/` — topic expansion, source ranking
  - `reflection/` — gap detection, strategy advisor
  - `search/` — web search adapter, source fetcher
  - `integrations/` — Obsidian export, PI runtime

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,telegram,memory]"

# Run tests (requires PYTHONIOENCODING=utf-8 on Windows for Chinese strings)
PYTHONIOENCODING=utf-8 python -m pytest tests/ -x --no-cov

# Run with coverage
PYTHONIOENCODING=utf-8 python -m pytest tests/

# Type checking
mypy src/auto_research/
```

## Key conventions

- All user-facing strings are in Traditional Chinese
- Tests use `conftest.make_temp_repo()` to create isolated temp repos with config
- Service modules import exceptions from `auto_research.exceptions`, not orchestrator
- WSL distro is configurable via `WSL_DISTRO` env var (default: Ubuntu-24.04)
- Session IDs must match `^[a-zA-Z0-9_-]{1,128}$` (path traversal protection)
- Policy guard checks ALL string args for dangerous patterns, not just chat text
- Silent exception handlers should always include `logger.debug(..., exc_info=True)`
