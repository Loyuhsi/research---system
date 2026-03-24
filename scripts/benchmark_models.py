"""Benchmark: nemotron-3-nano vs qwen3.5-35b-a3b on identical topic+sources.

Fair comparison controls:
- Same topic, same source bundle (identical seed in both session parsed/ dirs)
- Same prompt structure (SynthesizerService uses identical system/user prompts)
- Same timeout (900s), same temperature (0.2)
- run_kind="benchmark" in results.tsv for identification
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from auto_research.runtime import load_config
from auto_research.http_client import JsonHttpClient
from auto_research.services.llm_provider import LlmProviderService
from auto_research.services.vault import VaultService
from auto_research.services.synthesizer import SynthesizerService
from workspace_utils import prepare_isolated_workspace


TOPIC = "Bounded web search integration for a local multi-provider Auto-Research platform"

SOURCE_CONTENT = """\
---
title: "Auto-Research Search Integration Design"
source_type: "seed"
---

# Auto-Research Search Integration Design

A local multi-provider Auto-Research platform integrates bounded web search
to augment research quality. The system uses DuckDuckGo HTML search (stdlib-only,
no API keys) to find relevant sources, fetches and parses them with a SourceFetcher
that strips scripts/styles/nav/footer elements, and writes parsed content to
session parsed/ directories with stable frontmatter contracts.

Key design constraints:
- Maximum 3 fetched pages per run (bounded resource usage)
- Maximum 5000 words per page (deterministic word-boundary truncation)
- 15-second fetch timeout per page
- All execution local — no cloud LLM APIs
- Parsed sources follow a stable contract: title, source_url, fetched_at, source_type, word_count

The search pipeline feeds into the existing synthesis flow:
WebSearchAdapter → List[SearchResult] → SourceFetcher → FetchResult → write_to_session()
→ parsed/web-{slug}.md → SynthesizerService.synthesize() → quality gate → results.tsv

Provider architecture supports Ollama, LM Studio, and vLLM with inference-aware
auto-select, readiness caching (316x speedup), and circuit breaker per provider.
The Telegram control plane provides a conversational interface with 3-layer hybrid
intent parsing (keyword → LLM → clarification) and policy-gated action execution
(SAFE/CONFIRM/DISABLED).

Results are tracked in results.tsv with 19 columns including run_kind,
search_result_count, fetched_source_count, and program_hash for traceability.
Quality gates measure coverage (n-gram overlap), structure (frontmatter+headings),
and provenance (evidence+diversity).
"""

MODELS = [
    "nvidia/nemotron-3-nano",
    "qwen/qwen3.5-35b-a3b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple models in an isolated workspace")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--model", action="append", dest="models", help="Model id to benchmark; repeat to run multiple")
    return parser.parse_args()


def run_benchmark(repo_root: Path, isolated: bool = False, models: list[str] | None = None):
    workspace = prepare_isolated_workspace(repo_root, "benchmark-models") if isolated else repo_root
    config = load_config(repo_root=workspace)
    http_client = JsonHttpClient()
    llm = LlmProviderService(config)
    vault_service = VaultService(config)
    synth = SynthesizerService(config, http_client, llm, vault_service)

    selected_models = models or MODELS
    results = []
    for model in selected_models:
        session_id = f"v317-benchmark-{model.split('/')[-1]}-{int(time.time())}"
        layout = config.resolve_layout(session_id)
        layout.ensure()

        # Write identical source content
        seed_path = layout.parsed_dir / "seed-benchmark.md"
        seed_path.write_text(SOURCE_CONTENT, encoding="utf-8")

        print(f"\n{'='*60}")
        print(f"Benchmarking: {model}")
        print(f"Session: {session_id}")
        print(f"{'='*60}")

        t0 = time.monotonic()
        try:
            result = synth.synthesize(
                topic=TOPIC,
                session_id=session_id,
                provider="lmstudio",
                model=model,
                run_kind="benchmark",
            )
            elapsed = round(time.monotonic() - t0, 1)
            quality = result.get("quality", {})
            print(f"  Model: {model}")
            print(f"  Words: {quality.get('word_count', '?')}")
            print(f"  Coverage: {quality.get('coverage_score', '?')}")
            print(f"  Structure: {quality.get('structure_score', '?')}")
            print(f"  Provenance: {quality.get('provenance_score', '?')}")
            print(f"  QG Pass: {quality.get('passed', '?')}")
            print(f"  Latency: {elapsed}s")
            print(f"  TSV written: {result.get('results_tsv_written', '?')}")
            results.append({"model": model, "quality": quality, "latency": elapsed})
        except Exception as exc:
            elapsed = round(time.monotonic() - t0, 1)
            print(f"  FAILED after {elapsed}s: {exc}")
            results.append({"model": model, "error": str(exc), "latency": elapsed})

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Topic: {TOPIC}")
    for r in results:
        q = r.get("quality", {})
        if "error" in r:
            print(f"  {r['model']}: FAILED ({r['error'][:80]})")
        else:
            print(f"  {r['model']}: words={q.get('word_count',0)} cov={q.get('coverage_score',0):.3f} "
                  f"struct={q.get('structure_score',0):.3f} QG={'PASS' if q.get('passed') else 'FAIL'} "
                  f"latency={r['latency']}s")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args.repo_root.resolve(), isolated=args.isolated, models=args.models)
