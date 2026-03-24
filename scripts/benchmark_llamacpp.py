"""Benchmark: llama.cpp vs LM Studio on identical topic+sources.

Controls for quantization as an independent variable.
Records gpu_layers_used, peak_vram_mb, quantization per run.
VRAM gate check prevents runs on CPU-offloaded inference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

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
from auto_research.resource_guard import check_vram_available, VramMonitor
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
- All execution local -- no cloud LLM APIs
- Parsed sources follow a stable contract: title, source_url, fetched_at, source_type, word_count

The search pipeline feeds into the existing synthesis flow:
1. WebSearchAdapter.search(topic, max_results=3) returns SearchResult objects
2. SourceFetcher.fetch_and_parse(url) downloads and converts HTML to Markdown
3. Sources are written to session parsed/ dir with frontmatter
4. SynthesizerService.synthesize() processes all sources through LLM
5. QualityGateService verifies coverage, structure, and provenance
6. SkillMemoryService extracts memory records from successful runs
7. ObsidianExporter writes notes to vault with wikilinks
"""


@dataclass
class BenchmarkSpec:
    name: str
    provider: str
    model: str
    run_kind: str
    gpu_layers: str
    quantization: str


@dataclass
class BenchmarkResult:
    spec: BenchmarkSpec
    word_count: int = 0
    coverage: float = 0.0
    structure: float = 0.0
    provenance: float = 0.0
    evaluation_score: float = 0.0
    latency_s: float = 0.0
    qg_pass: bool = False
    peak_vram_mb: int = 0
    error: str = ""


def run_benchmark(spec: BenchmarkSpec, config, http_client) -> BenchmarkResult:
    """Run a single benchmark with VRAM gate and monitoring."""
    result = BenchmarkResult(spec=spec)

    # VRAM gate check
    vram = check_vram_available(min_free_mb=2048)
    if not vram.get("ok"):
        if "error" not in vram:
            result.error = f"VRAM gate failed: {vram.get('used_mb', '?')}MB used, need 2048MB free"
        else:
            result.error = f"VRAM check unavailable: {vram.get('error')}"
        print(f"  [SKIP] {spec.name}: {result.error}")
        return result

    session_id = f"bench-{spec.run_kind}-{int(time.time())}"
    layout = config.resolve_layout(session_id)
    layout.ensure()
    (layout.parsed_dir / "seed.md").write_text(SOURCE_CONTENT, encoding="utf-8")

    llm = LlmProviderService(config)
    vault = VaultService(config)
    synth = SynthesizerService(config, http_client, llm, vault)

    # Start VRAM monitoring
    monitor = VramMonitor(poll_interval=1.0)
    monitor.start()

    try:
        t0 = time.monotonic()
        synth_result = synth.synthesize(
            topic=TOPIC,
            session_id=session_id,
            provider=spec.provider,
            model=spec.model,
            run_kind=spec.run_kind,
        )
        result.latency_s = round(time.monotonic() - t0, 1)

        quality = synth_result.get("quality", {})
        result.word_count = quality.get("word_count", 0)
        result.coverage = quality.get("coverage_score", 0.0)
        result.structure = quality.get("structure_score", 0.0)
        result.provenance = quality.get("provenance_score", 0.0)
        result.qg_pass = quality.get("passed", False)
        result.evaluation_score = round(
            result.coverage * 0.35 + result.structure * 0.25
            + min(result.word_count / 500, 1.0) * 0.3 + result.provenance * 0.10, 3
        )
    except Exception as exc:
        result.error = str(exc)
    finally:
        result.peak_vram_mb = monitor.stop()

    return result


def compare_results(results: List[BenchmarkResult]) -> Dict:
    """Compare benchmark results and produce decision."""
    valid = [r for r in results if not r.error]
    if len(valid) < 2:
        return {"decision": "INSUFFICIENT_DATA", "reason": "Need at least 2 valid runs to compare"}

    # Find LM Studio and llama.cpp results
    lmstudio = [r for r in valid if "lmstudio" in r.spec.run_kind]
    llamacpp = [r for r in valid if "llamacpp" in r.spec.run_kind]

    if not lmstudio or not llamacpp:
        return {"decision": "INSUFFICIENT_DATA", "reason": "Need both providers to compare"}

    lm = lmstudio[0]
    lc = llamacpp[0]

    dimensions = {
        "coverage": (lm.coverage, lc.coverage),
        "structure": (lm.structure, lc.structure),
        "word_count": (lm.word_count, lc.word_count),
        "latency_s": (lm.latency_s, lc.latency_s),
        "provenance": (lm.provenance, lc.provenance),
    }

    # Check if difference exceeds 10% in any dimension
    significant_diff = False
    deltas = {}
    for dim, (lm_val, lc_val) in dimensions.items():
        baseline = max(abs(lm_val), abs(lc_val), 0.001)
        delta_pct = abs(lm_val - lc_val) / baseline * 100
        deltas[dim] = {"lmstudio": lm_val, "llamacpp": lc_val, "delta_pct": round(delta_pct, 1)}
        if delta_pct > 10:
            significant_diff = True

    if not significant_diff:
        return {
            "decision": "REJECT",
            "reason": "Difference <10% across all dimensions — not justified to maintain two providers",
            "deltas": deltas,
        }

    # Determine winner
    lc_wins = sum(1 for d in ["coverage", "structure", "word_count", "provenance"]
                  if deltas[d]["llamacpp"] > deltas[d]["lmstudio"])
    if deltas["latency_s"]["llamacpp"] < deltas["latency_s"]["lmstudio"]:
        lc_wins += 1

    if lc_wins >= 3:
        return {"decision": "ADOPT", "reason": "llama.cpp wins majority of dimensions", "deltas": deltas}
    return {"decision": "KEEP_LMSTUDIO", "reason": "LM Studio wins or ties majority of dimensions", "deltas": deltas}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp vs LM Studio")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--isolated", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = prepare_isolated_workspace(args.repo_root.resolve(), "benchmark-llamacpp") if args.isolated else args.repo_root.resolve()
    config = load_config(repo_root=workspace)
    http_client = JsonHttpClient()

    specs = [
        BenchmarkSpec(
            name="LM Studio Q4_K_M",
            provider="lmstudio",
            model=config.model,
            run_kind="benchmark_lmstudio",
            gpu_layers="auto",
            quantization="Q4_K_M",
        ),
        BenchmarkSpec(
            name="llama.cpp Q4_K_M",
            provider="llamacpp",
            model=config.model,
            run_kind="benchmark_llamacpp",
            gpu_layers="99",
            quantization="Q4_K_M",
        ),
    ]

    print("=" * 60)
    print(f"Benchmark: llama.cpp vs LM Studio (v3.18)")
    print(f"Topic: {TOPIC[:60]}...")
    print("=" * 60)

    results: List[BenchmarkResult] = []
    for spec in specs:
        print(f"\n--- {spec.name} ({spec.provider}) ---")
        result = run_benchmark(spec, config, http_client)
        results.append(result)
        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  Words: {result.word_count}")
            print(f"  Coverage: {result.coverage:.3f}")
            print(f"  Latency: {result.latency_s}s")
            print(f"  Peak VRAM: {result.peak_vram_mb}MB")
            print(f"  QG Pass: {result.qg_pass}")

    # Compare
    comparison = compare_results(results)
    print(f"\n{'=' * 60}")
    print(f"Decision: {comparison['decision']}")
    print(f"Reason: {comparison['reason']}")

    # Write report
    report = {
        "version": "v3.18",
        "topic": TOPIC,
        "results": [asdict(r) for r in results],
        "comparison": comparison,
    }
    report_path = PROJECT_ROOT / "output" / "benchmark_llamacpp_v318.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
