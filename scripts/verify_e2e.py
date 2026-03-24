"""V3.10 E2E verification script — runs against real providers."""

import json
import sys
import time
import traceback
import shutil
import tempfile

sys.path.insert(0, "src")
from pathlib import Path
from auto_research.runtime import load_config
from auto_research.http_client import JsonHttpClient
from auto_research.services.llm_provider import LlmProviderService
from auto_research.services.vault import VaultService
from auto_research.services.fetcher import FetcherService
from auto_research.services.synthesizer import SynthesizerService
from auto_research.services.evoskill import EvoSkillService
from auto_research.services.skill_memory import SkillMemoryService
from auto_research.services.task_review import TaskReviewService
from auto_research.reflection.strategy_advisor import StrategyAdvisor
from auto_research.registry import ServiceRegistry
from auto_research.events import EventBus
from auto_research.event_handlers import wire_event_handlers
from auto_research.conversation_store import InMemoryConversationStore
from auto_research.orchestrator import Orchestrator

# Setup temp repo
tmp = tempfile.mkdtemp()
repo = Path(tmp)
for d in [
    "config", "output/notes", "output/sources", "output/research",
    "knowledge/index", "knowledge/logs", "knowledge/memory-records",
    "knowledge/evaluations", "staging/skills-candidates", "staging/memory-drafts",
    "staging/tooling", "sandbox/rd-agent/in", "sandbox/rd-agent/out",
    "skills", ".github/skills",
]:
    (repo / d).mkdir(parents=True, exist_ok=True)
for n in ("runtime-modes.json", "zones.json", "tool-allowlist.json"):
    shutil.copy2(f"config/{n}", str(repo / "config" / n))
(repo / ".env").write_text(
    "TELEGRAM_PROVIDER=lmstudio\nTELEGRAM_MODEL=nvidia/nemotron-3-nano\n"
    "SKILL_MEMORY_EMBEDDING_PROVIDER=lmstudio\n",
    encoding="utf-8",
)

config = load_config(repo_root=repo, environ={})
http = JsonHttpClient()
llm = LlmProviderService(config)
vault = VaultService(config)
fetcher = FetcherService(config, vault)
synth = SynthesizerService(config, http, llm, vault)
evoskill = EvoSkillService(config)
skill_mem = SkillMemoryService(config, http, llm)
strategy = StrategyAdvisor(config, http, llm)
task_review = TaskReviewService(config, skill_mem, evoskill, strategy)
registry = ServiceRegistry()
event_bus = EventBus()
registry.register("core.config", config)
registry.register("core.events", event_bus)
registry.register("core.http", http)
registry.register("service.llm", llm)
registry.register("service.vault", vault)
registry.register("service.fetcher", fetcher)
registry.register("service.synthesizer", synth)
registry.register("service.evoskill", evoskill)
registry.register("service.skill_memory", skill_mem)
registry.register("service.strategy_advisor", strategy)
registry.register("service.task_review", task_review)
from auto_research.services.tool_runner import ToolRunnerService
registry.register("service.tool_runner", ToolRunnerService(config))
wire_event_handlers(event_bus, evoskill, skill_mem, repo / "output" / "telemetry.jsonl")

orch = Orchestrator(
    config=config, event_bus=event_bus, registry=registry,
    llm_service=llm, fetcher_service=fetcher, vault_service=vault,
    http_client=http, conversation_store=InMemoryConversationStore(),
)

results = {}

# --- B1. Real Chat ---
print("=" * 60)
print("B1. REAL CHAT (LM Studio)")
print("=" * 60)
try:
    t0 = time.monotonic()
    r = orch.chat(
        session_key="verify-chat",
        text="What is 2+2? Reply briefly.",
        mode="research_only",
        frontend="cli",
    )
    elapsed = time.monotonic() - t0
    print(f"  Reply: {r['reply'][:120]}")
    print(f"  Provider: {r['provider']}, Model: {r['model']}")
    print(f"  Latency: {elapsed:.1f}s")
    print(f"  Context: {r.get('context_hits', {})}")
    results["chat"] = "LIVE_PASS"
except Exception as e:
    print(f"  FAILED: {e}")
    results["chat"] = f"FAILED: {e}"

# --- B2. Synthesize + Quality Gate ---
print("\n" + "=" * 60)
print("B2. SYNTHESIZE + QUALITY GATE")
print("=" * 60)
try:
    sid = "verify-synth-001"
    layout = config.resolve_layout(sid)
    layout.parsed_dir.mkdir(parents=True, exist_ok=True)
    (layout.parsed_dir / "source.md").write_text(
        "# Python Testing\n\nPytest is a framework for testing Python code.\n"
        "It supports fixtures, markers, and parametrize.\n"
        "Unit tests verify functions. Integration tests check components.\n"
        "Coverage measures code exercised by tests.\n"
        "Test-driven development writes tests first.\n"
        "Mocking replaces real objects with test doubles.\n"
        "Property-based testing with Hypothesis generates random inputs.\n",
        encoding="utf-8",
    )
    layout.status_path.parent.mkdir(parents=True, exist_ok=True)
    layout.status_path.write_text(
        json.dumps({
            "topic": "Python Testing",
            "sources": [{"url": "https://docs.pytest.org", "topic": "Python Testing"}],
        }),
        encoding="utf-8",
    )
    t0 = time.monotonic()
    r = orch.synthesize(
        topic="Python Testing",
        session_id=sid,
        provider="lmstudio",
        model="nvidia/nemotron-3-nano",
    )
    elapsed = time.monotonic() - t0
    print(f"  Note: {r.get('note_path', '?')}")
    print(f"  Latency: {elapsed:.1f}s")
    if "quality" in r:
        q = r["quality"]
        print(f"  Coverage: {q.get('coverage_score')}, Structure: {q.get('structure_score')}, Words: {q.get('word_count')}")
        print(f"  Provenance: ev={q.get('evidence_count')}, div={q.get('source_diversity')}, score={q.get('provenance_score')}")
        print(f"  Passed: {q.get('passed')}")
    if "evaluation" in r:
        print(f"  Eval: {json.dumps(r['evaluation'], default=str)[:200]}")
    results["synthesize"] = "LIVE_PASS"
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    results["synthesize"] = f"FAILED: {e}"

# --- B3. Memory Lifecycle ---
print("\n" + "=" * 60)
print("B3. MEMORY LIFECYCLE")
print("=" * 60)
try:
    sid2 = "verify-mem-001"
    layout2 = config.resolve_layout(sid2)
    layout2.parsed_dir.mkdir(parents=True, exist_ok=True)
    (layout2.parsed_dir / "src.md").write_text(
        "# Memory test\nTest data for memory extraction.\n", encoding="utf-8",
    )
    layout2.status_path.parent.mkdir(parents=True, exist_ok=True)
    layout2.status_path.write_text(
        json.dumps({"topic": "MemTest", "sources": [{"url": "https://example.com/mem", "topic": "MemTest"}]}),
        encoding="utf-8",
    )
    layout2.note_path.parent.mkdir(parents=True, exist_ok=True)
    layout2.note_path.write_text("# Memory Test Note\nExtracted from test.\n", encoding="utf-8")

    r1 = orch.memory_extract(session_id=sid2, task_type="research_session", status="success")
    draft_id = r1["draft_id"]
    print(f"  Extract: draft_id={draft_id}")

    r2 = orch.memory_validate(memory_id=draft_id, approve=False)
    print(f"  Validate: valid={r2['valid']}, checks={list(r2.get('checks',{}).keys())}")

    r3 = orch.memory_validate(memory_id=draft_id, approve=True)
    print(f"  Approve: ok={'approved_path' in r3}")

    r4 = orch.memory_index_rebuild()
    print(f"  Index: records={r4['memory_records']}, fts5={r4.get('fts5_available')}")

    r5 = orch.memory_search(task="MemTest")
    hits = r5.get("memory_hits", [])
    print(f"  Search: {len(hits)} hits, backend={r5.get('vector_backend')}")
    if hits:
        print(f"    Top: id={hits[0]['id']}, score={hits[0]['score']}, ev={hits[0].get('evidence_count', 0)}")
    results["memory"] = "LIVE_PASS"
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    results["memory"] = f"FAILED: {e}"

# --- B4. Retrieval Quality ---
print("\n" + "=" * 60)
print("B4. RETRIEVAL QUALITY CHECK")
print("=" * 60)
try:
    r_exact = orch.memory_search(task="MemTest memory extraction")
    r_vague = orch.memory_search(task="completely unrelated quantum physics")
    exact_scores = [h["score"] for h in r_exact.get("memory_hits", [])]
    vague_scores = [h["score"] for h in r_vague.get("memory_hits", [])]
    print(f"  Exact query scores:  {exact_scores}")
    print(f"  Vague query scores:  {vague_scores}")
    print(f"  Exact backend:       {r_exact.get('vector_backend')}")
    print(f"  Vague backend:       {r_vague.get('vector_backend')}")
    if exact_scores and (not vague_scores or exact_scores[0] > vague_scores[0]):
        print("  BM25 discrimination: VERIFIED (exact > vague)")
        results["retrieval"] = "LIVE_PASS"
    else:
        print("  BM25 discrimination: WEAK")
        results["retrieval"] = "PARTIAL"
except Exception as e:
    print(f"  FAILED: {e}")
    results["retrieval"] = f"FAILED: {e}"

# --- B5. Evaluation Records ---
print("\n" + "=" * 60)
print("B5. EVALUATION RECORDS")
print("=" * 60)
try:
    eval_dir = repo / "knowledge" / "evaluations"
    evals = list(eval_dir.glob("*.json")) if eval_dir.exists() else []
    print(f"  Records written: {len(evals)}")
    for ep in evals[:3]:
        data = json.loads(ep.read_text(encoding="utf-8"))
        print(f"    {data['eval_id']}: type={data['eval_type']}, passed={data['passed']}, score={data.get('candidate_score')}")
    results["evaluation"] = "LIVE_PASS" if evals else "NO_RECORDS"
except Exception as e:
    print(f"  FAILED: {e}")
    results["evaluation"] = f"FAILED: {e}"

# --- B6. Telemetry / Tracing ---
print("\n" + "=" * 60)
print("B6. TELEMETRY / TRACING")
print("=" * 60)
try:
    tel_path = repo / "output" / "telemetry.jsonl"
    if tel_path.exists():
        lines = tel_path.read_text(encoding="utf-8").strip().splitlines()
        print(f"  Telemetry entries: {len(lines)}")
        traces = [l for l in lines if '"trace_id"' in l]
        print(f"  Trace entries: {len(traces)}")
        results["telemetry"] = "LIVE_PASS" if traces else "NO_TRACES"
    else:
        print("  No telemetry file")
        results["telemetry"] = "NO_FILE"
except Exception as e:
    print(f"  FAILED: {e}")
    results["telemetry"] = f"FAILED: {e}"

# --- B7. Telegram Readiness ---
print("\n" + "=" * 60)
print("B7. TELEGRAM READINESS")
print("=" * 60)
try:
    env_values = config.env_values
    token = env_values.get("TELEGRAM_BOT_TOKEN", "")
    if token:
        print(f"  Token: SET (length={len(token)})")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from telegram_bot import TelegramApi
        tg = TelegramApi(token, http)
        t0 = time.monotonic()
        me = tg.get_me_direct()
        elapsed = round((time.monotonic() - t0) * 1000, 1)
        print(f"  getMe: OK (@{me.get('username', '?')}, {elapsed}ms)")
        results["telegram"] = "LIVE_PASS"
    else:
        print("  Token: NOT SET — skip API test")
        results["telegram"] = "NO_TOKEN"
except Exception as e:
    print(f"  getMe: FAIL — {e}")
    results["telegram"] = f"FAIL_API: {e}"

# --- B8. Retrieval Fallback Chain ---
print("\n" + "=" * 60)
print("B8. RETRIEVAL FALLBACK CHAIN")
print("=" * 60)
try:
    from auto_research.services.skill_memory.retrieval import retrieve_context as _retrieve_ctx

    # Scenario 1: Normal path (semantic+bm25 or bm25)
    r_normal = orch.memory_search(task="MemTest")
    normal_be = r_normal.get("vector_backend", "?")
    normal_fts = r_normal.get("fts5_used", False)
    print(f"  Normal: backend={normal_be}, fts5_used={normal_fts}, hits={len(r_normal.get('memory_hits', []))}")

    # Scenario 2: BM25-only (disabled embeddings)
    config_no_embed = load_config(repo_root=repo, environ={"SKILL_MEMORY_EMBEDDING_PROVIDER": "disabled"})
    llm_no = LlmProviderService(config_no_embed)
    r_bm25 = _retrieve_ctx(config_no_embed, http, llm_no, task="MemTest")
    bm25_be = r_bm25.get("vector_backend", "?")
    print(f"  BM25-only: backend={bm25_be}, hits={len(r_bm25.get('memory_hits', []))}")

    # Scenario 3: Metadata-only (nonsense query -> fallback)
    r_meta = _retrieve_ctx(config_no_embed, http, llm_no, task="zzzzz_nomatch_xyz_9999")
    meta_be = r_meta.get("vector_backend", "?")
    meta_fb = r_meta.get("fallback_triggered", False)
    print(f"  Metadata-only: backend={meta_be}, fallback={meta_fb}")

    chain_ok = bm25_be == "bm25" and meta_fb
    print(f"  Fallback chain verified: {chain_ok}")
    results["fallback_chain"] = "LIVE_PASS" if chain_ok else "PARTIAL"
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    results["fallback_chain"] = f"FAILED: {e}"

# --- B9. Provider Auto-Select (inference-aware) ---
print("\n" + "=" * 60)
print("B9. PROVIDER AUTO-SELECT (inference-aware)")
print("=" * 60)
try:
    sel = llm.select_provider(http, preference="auto", verify_inference=True)
    print(f"  Auto (inference-verified): provider={sel['provider']}, reason={sel['reason']}")
    if "selection_detail" in sel:
        print(f"  Detail: {json.dumps(sel['selection_detail'], default=str)[:200]}")
    # Check vLLM in matrix
    matrix = llm.provider_capability_matrix(http)
    print(f"  Providers in matrix: {list(matrix.keys())}")
    for prov in matrix:
        h = matrix[prov]["health"]
        print(f"    {prov}: health={h.get('ok')}, primary={matrix[prov].get('is_primary')}")
    sel_fast = llm.select_provider(http, preference="auto", verify_inference=False)
    print(f"  Auto (health-only):        provider={sel_fast['provider']}, reason={sel_fast['reason']}")
    results["auto_select"] = "LIVE_PASS"
except Exception as e:
    print(f"  FAILED: {e}")
    results["auto_select"] = f"FAILED: {e}"

# --- Final Summary ---
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
for k, v in results.items():
    status_icon = "PASS" if "PASS" in str(v) else ("PARTIAL" if "PARTIAL" in str(v) else "FAIL")
    print(f"  [{status_icon}] {k}: {v}")

shutil.rmtree(tmp, ignore_errors=True)
