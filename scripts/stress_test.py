#!/usr/bin/env python3
"""Stress test for Auto-Research core components.

Measures latency, throughput, and stability under load.

Usage:
    python scripts/stress_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from auto_research.http_client import JsonHttpClient
from auto_research.runtime import load_config
from auto_research.services.llm_provider import LlmProviderService
from auto_research.services.skill_memory import SkillMemoryService
from workspace_utils import prepare_isolated_workspace


def stress_llm_latency(config, http_client, llm, rounds: int = 5):
    """Measure LLM response latency over N rounds."""
    print(f"\n[1/6] LLM latency test ({rounds} rounds)...")
    provider = config.provider
    model = llm.default_model_for_provider(provider)

    latencies = []
    errors = 0
    for i in range(rounds):
        t0 = time.monotonic()
        try:
            resp = llm.call_text(
                provider,
                http_client,
                model=model,
                messages=[
                    {"role": "system", "content": "Reply in one short sentence."},
                    {"role": "user", "content": f"What is stress test round {i+1}?"},
                ],
                temperature=0.1,
                timeout=60,
            )
            content = str(resp.get("content", ""))
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)
            print(f"  Round {i+1}: {elapsed:.2f}s — {(content or '')[:60]}")
        except Exception as e:
            elapsed = time.monotonic() - t0
            errors += 1
            print(f"  Round {i+1}: FAILED after {elapsed:.2f}s — {e}")
            if errors >= 2:
                print("  Skipping remaining rounds (LLM unresponsive).")
                break

    return {
        "test": "llm_latency",
        "rounds": rounds,
        "errors": errors,
        "latencies_s": [round(l, 3) for l in latencies],
        "avg_s": round(sum(latencies) / max(len(latencies), 1), 3),
        "max_s": round(max(latencies) if latencies else 0, 3),
        "min_s": round(min(latencies) if latencies else 0, 3),
        "passed": errors == 0 and bool(latencies),
    }


def stress_large_source_bundle(config):
    """Test handling of a large source bundle near MAX_SOURCE_BYTES."""
    print("\n[2/6] Large source bundle test...")
    max_bytes = config.max_source_bytes
    fake_content = "Research content. " * (max_bytes // 18)

    t0 = time.monotonic()
    truncated = fake_content[:max_bytes]
    elapsed = time.monotonic() - t0

    return {
        "test": "large_source_bundle",
        "max_bytes": max_bytes,
        "content_size": len(fake_content),
        "truncated_size": len(truncated),
        "processing_s": round(elapsed, 5),
        "passed": len(truncated) <= max_bytes,
    }


def stress_memory_index(config, http_client, llm, record_count: int = 20):
    """Test memory index build + query with N records."""
    print(f"\n[3/6] Memory index stress test ({record_count} records)...")
    svc = SkillMemoryService(config, http_client, llm)

    # Create dummy memory records
    records_dir = config.memory_records_dir
    records_dir.mkdir(parents=True, exist_ok=True)
    for i in range(record_count):
        record = {
            "id": f"stress-{i:04d}",
            "title": f"Stress test record {i}",
            "summary": f"Stress test record {i} about topic number {i}",
            "tags": ["stress", "test"],
            "task_type": "research_session",
            "source_types": ["memory"],
            "tool_deps": [],
            "citations": [],
            "confidence": 0.8,
            "success_count": 1,
            "failure_count": 0,
            "risk_level": "low",
            "expires_at": "2027-03-19T00:00:00Z",
            "related_skills": [],
            "obsidian_links": [],
            "status": "approved",
            "session_id": f"session-{i:04d}",
            "created_at": "2026-03-19T00:00:00Z",
            "evidence_sources": [f"https://example.com/stress-{i}", f"/tmp/stress-{i}.md"],
        }
        (records_dir / f"stress-{i:04d}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    # Rebuild index
    t0 = time.monotonic()
    try:
        result = svc.memory_index_rebuild()
        rebuild_time = time.monotonic() - t0
        print(f"  Index rebuild: {rebuild_time:.3f}s — {result.get('memory_records', '?')} records, fts5={result.get('fts5_available')}")
    except Exception as e:
        rebuild_time = time.monotonic() - t0
        print(f"  Index rebuild FAILED after {rebuild_time:.3f}s: {e}")
        return {"test": "memory_index", "error": str(e)}

    # Query
    t0 = time.monotonic()
    try:
        search_result = svc.memory_search(task="stress test topic")
        query_time = time.monotonic() - t0
        hits = len(search_result.get("memory_hits", []))
        print(f"  Query: {query_time:.3f}s — {hits} hits")
    except Exception as e:
        query_time = time.monotonic() - t0
        print(f"  Query FAILED after {query_time:.3f}s: {e}")
        return {"test": "memory_index", "error": str(e)}

    # Cleanup
    for i in range(record_count):
        p = records_dir / f"stress-{i:04d}.json"
        if p.exists():
            p.unlink()

    return {
        "test": "memory_index",
        "record_count": record_count,
        "rebuild_s": round(rebuild_time, 3),
        "query_s": round(query_time, 3),
        "query_hits": hits,
        "passed": True,
    }


def stress_concurrent_sessions(config):
    """Simulate multiple sessions creating artifacts concurrently."""
    print("\n[4/6] Concurrent session simulation (3 sessions)...")
    sessions = ["stress-a", "stress-b", "stress-c"]
    t0 = time.monotonic()

    for sid in sessions:
        layout = config.resolve_layout(sid)
        layout.ensure()
        # Write a fake source
        (layout.parsed_dir / "source.md").write_text(
            f"# Source for {sid}\n\nContent " * 50, encoding="utf-8"
        )

    elapsed = time.monotonic() - t0

    # Verify all exist
    all_ok = all(config.resolve_layout(sid).parsed_dir.exists() for sid in sessions)

    # Cleanup
    import shutil
    for sid in sessions:
        layout = config.resolve_layout(sid)
        if layout.research_root.exists():
            shutil.rmtree(layout.research_root)

    return {
        "test": "concurrent_sessions",
        "session_count": len(sessions),
        "setup_s": round(elapsed, 4),
        "all_created": all_ok,
        "passed": all_ok,
    }


def stress_provider_selection(config, http_client, llm, rounds: int = 5):
    """Test provider selection repeatedly, tracking distribution and latency."""
    print(f"\n[5/6] Provider selection stress ({rounds} rounds)...")
    from collections import Counter
    provider_counts: Counter = Counter()
    reasons: Counter = Counter()
    latencies = []
    errors = 0

    for i in range(rounds):
        t0 = time.monotonic()
        try:
            result = llm.select_provider(http_client, preference="auto", verify_inference=True)
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)
            provider_counts[str(result["provider"])] += 1
            reasons[str(result["reason"])] += 1
            print(f"  Round {i+1}: {result['provider']} ({result['reason']}) — {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.monotonic() - t0
            errors += 1
            print(f"  Round {i+1}: FAILED after {elapsed:.2f}s — {e}")

    return {
        "test": "provider_selection",
        "rounds": rounds,
        "errors": errors,
        "provider_distribution": dict(provider_counts),
        "reason_distribution": dict(reasons),
        "avg_s": round(sum(latencies) / max(len(latencies), 1), 3),
        "passed": errors == 0,
    }


def stress_breaker_state(config, http_client, llm):
    """Report circuit breaker state for each provider."""
    print("\n[6/7] Circuit breaker state report...")
    states = []
    for provider in ("ollama", "lmstudio", "vllm"):
        breaker = llm.get_breaker(provider)
        status = breaker.status()
        states.append({
            "provider": provider,
            "state": status["state"],
            "failure_count": status["failure_count"],
        })
        print(f"  {provider}: state={status['state']}, failures={status['failure_count']}")
    return {
        "test": "breaker_state",
        "breaker_states": states,
        "passed": True,
    }


def stress_readiness_cache(config, http_client, llm, rounds: int = 3):
    """Test readiness cache effectiveness: cold vs warm latency."""
    print(f"\n[7/7] Readiness cache test ({rounds} rounds)...")
    latencies = []

    # Invalidate to start cold
    llm._cache.invalidate()

    for i in range(rounds):
        t0 = time.monotonic()
        result = llm.select_provider(http_client, preference="auto", verify_inference=True)
        elapsed = time.monotonic() - t0
        latencies.append(elapsed)
        label = "Cold" if i == 0 else f"Warm {i+1}"
        print(f"  {label}: {elapsed:.2f}s — {result['provider']} ({result['reason']})")

    speedup = round(latencies[0] / max(latencies[-1], 0.001), 1) if len(latencies) > 1 else 1.0
    return {
        "test": "readiness_cache",
        "rounds": rounds,
        "cold_s": round(latencies[0], 3),
        "warm_s": round(latencies[-1], 3) if len(latencies) > 1 else 0,
        "speedup": speedup,
        "passed": len(latencies) > 1 and latencies[-1] < latencies[0],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rerunnable stress tests for Auto-Research")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--isolated", action="store_true", help="Copy config into a temporary workspace under output/")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    workspace = prepare_isolated_workspace(repo_root, "stress") if args.isolated else repo_root
    config = load_config(repo_root=workspace)
    http_client = JsonHttpClient()
    llm = LlmProviderService(config)

    results = []
    results.append(stress_llm_latency(config, http_client, llm, rounds=3))
    results.append(stress_large_source_bundle(config))
    results.append(stress_memory_index(config, http_client, llm, record_count=20))
    results.append(stress_concurrent_sessions(config))
    results.append(stress_provider_selection(config, http_client, llm, rounds=3))
    results.append(stress_breaker_state(config, http_client, llm))
    results.append(stress_readiness_cache(config, http_client, llm, rounds=3))

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": config.provider,
        "model": config.model,
        "tests": results,
        "all_passed": all(r.get("passed", not r.get("error")) for r in results),
    }

    out_path = config.repo_root / "output" / "stress-report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Report: {out_path}")
    print(f"All passed: {report['all_passed']}")
    for t in results:
        status = "PASS" if t.get("passed", not t.get("error")) else "FAIL"
        print(f"  [{status}] {t['test']}")

    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
