#!/usr/bin/env python3
"""Multi-topic research validation — runs 3 topics through the full pipeline.

Usage:
    python scripts/multi_topic_research.py

Requires:
    - Ollama running with qwen3.5:9b loaded
    - WSL2 available (for fetch workers)
    - Pre-fetched source files OR network access

This script creates local source files to bypass network requirements,
then runs synthesize → quality_gate → reflect → memory_extract.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_research.runtime import load_config
from auto_research.http_client import JsonHttpClient
from auto_research.services.llm_provider import LlmProviderService
from auto_research.services.synthesizer import SynthesizerService
from auto_research.services.vault import VaultService
from auto_research.services.quality_gate import QualityGateService
from auto_research.services.skill_memory import SkillMemoryService
from auto_research.reflection.gap_detector import GapDetector
from auto_research.reflection.strategy_advisor import StrategyAdvisor


TOPICS = [
    {
        "id": "multi-llm-opt",
        "topic": "Local LLM inference optimization techniques",
        "source_content": (
            "# LLM Inference Optimization\n\n"
            "## Quantization\n\n"
            "Quantization reduces model size by converting weights from FP32 to lower precision formats "
            "like INT8, INT4, or GGUF Q4_K_M. This dramatically reduces memory usage and increases throughput. "
            "Tools like llama.cpp, GGML, and bitsandbytes make quantization accessible.\n\n"
            "## KV Cache Optimization\n\n"
            "Key-value cache management is critical for long-context inference. Techniques include "
            "paged attention (vLLM), sliding window attention, and flash attention.\n\n"
            "## Speculative Decoding\n\n"
            "Using a smaller draft model to propose tokens that the larger model verifies in batch "
            "can achieve 2-3x speedup without quality loss.\n\n"
            "## Batch Processing\n\n"
            "Continuous batching allows serving multiple requests simultaneously, improving GPU utilization.\n\n"
            "## 結論\n\nCombining quantization with optimized serving frameworks yields the best results.\n"
        ),
    },
    {
        "id": "multi-supply-chain",
        "topic": "Supply chain security in Python packaging",
        "source_content": (
            "# Python Supply Chain Security\n\n"
            "## Typosquatting\n\n"
            "Attackers upload packages with names similar to popular packages (e.g., 'reqeusts' vs 'requests'). "
            "PyPI has limited protections against this. Users should verify package names carefully.\n\n"
            "## Dependency Confusion\n\n"
            "When internal package names overlap with public PyPI packages, pip may install the wrong one. "
            "Mitigations include using --index-url, PEP 708 alternate package sources.\n\n"
            "## Code Signing\n\n"
            "Sigstore and PEP 480 aim to bring cryptographic verification to Python packages.\n\n"
            "## Lock Files\n\n"
            "Tools like pip-tools, Poetry, and PDM generate lock files with pinned hashes for reproducibility.\n\n"
            "## 結論\n\nA layered approach combining lock files, hash verification, and namespace management is essential.\n"
        ),
    },
    {
        "id": "multi-event-driven",
        "topic": "Event-driven architecture patterns for research automation",
        "source_content": (
            "# Event-Driven Architecture\n\n"
            "## Pub/Sub Pattern\n\n"
            "Publishers emit events without knowing subscribers. This decouples components and enables "
            "extensibility. The EventBus pattern is common in monolithic applications.\n\n"
            "## Event Sourcing\n\n"
            "All changes are stored as immutable events. State is derived by replaying events. "
            "Benefits include audit trails and temporal queries.\n\n"
            "## CQRS\n\n"
            "Command Query Responsibility Segregation separates read and write models, "
            "enabling optimized data stores for each use case.\n\n"
            "## Saga Pattern\n\n"
            "Long-running workflows coordinate via events, with compensating transactions for rollback.\n\n"
            "## 結論\n\nEvent-driven patterns excel in research automation where tasks are loosely coupled.\n"
        ),
    },
]


def run_topic(config, http_client, llm, topic_info):
    """Run full pipeline for a single topic."""
    sid = topic_info["id"]
    topic = topic_info["topic"]
    print(f"\n{'='*60}")
    print(f"Topic: {topic}")
    print(f"Session: {sid}")

    # Create source files
    layout = config.resolve_layout(sid)
    layout.ensure()
    source_path = layout.parsed_dir / "source_01.md"
    source_path.write_text(topic_info["source_content"], encoding="utf-8")
    print(f"  [1/4] Source created: {source_path.name}")

    # Synthesize
    vault = VaultService(config)
    synthesizer = SynthesizerService(config, http_client, llm, vault)
    try:
        result = synthesizer.synthesize(topic, sid)
        print(f"  [2/4] Note synthesized: {result['note_path']}")
        quality = result.get("quality", {})
        print(f"         Quality: coverage={quality.get('coverage_score', '?')}, "
              f"structure={quality.get('structure_score', '?')}, "
              f"passed={quality.get('passed', '?')}")
    except Exception as e:
        print(f"  [2/4] Synthesis FAILED: {e}")
        return {"topic": topic, "session_id": sid, "success": False, "error": str(e)}

    # Memory extract
    skill_memory = SkillMemoryService(config, http_client, llm)
    try:
        mem_result = skill_memory.memory_extract(session_id=sid, task_type="research_session", status="success")
        print(f"  [3/4] Memory draft: {mem_result.get('draft_path', '?')}")
    except Exception as e:
        print(f"  [3/4] Memory extract failed: {e}")

    # Reflect
    try:
        detector = GapDetector(config)
        gap_report = detector.scan()
        advisor = StrategyAdvisor(config, http_client, llm)
        advice = advisor.advise(gap_report)
        print(f"  [4/4] Reflection: {advice.get('source', '?')} — gaps={gap_report.has_gaps}")
    except Exception as e:
        print(f"  [4/4] Reflection failed: {e}")

    return {"topic": topic, "session_id": sid, "success": True, "quality": quality}


def main():
    config = load_config()
    http_client = JsonHttpClient()
    llm = LlmProviderService(config)

    print("Multi-topic research validation")
    print(f"Provider: {config.provider}, Model: {config.model}")

    results = []
    for topic_info in TOPICS:
        result = run_topic(config, http_client, llm, topic_info)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    successes = sum(1 for r in results if r.get("success"))
    print(f"  {successes}/{len(results)} topics completed successfully")
    for r in results:
        status = "OK" if r.get("success") else "FAIL"
        print(f"  [{status}] {r['topic']}")

    report_path = config.repo_root / "output" / "multi-topic-report.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report_path}")

    return 0 if successes == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
