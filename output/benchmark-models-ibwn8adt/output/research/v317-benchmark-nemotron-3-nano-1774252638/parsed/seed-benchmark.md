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
