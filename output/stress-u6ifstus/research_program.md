---
program_version: "3.17"
priority_topic: "Bounded web search integration for a local multi-provider Auto-Research platform"
timeout_seconds: 300
min_word_count: 200
coverage_threshold: 0.3
preferred_provider: "lmstudio"
preferred_model: "nvidia/nemotron-3-nano"
search_enabled: true
max_search_results: 5
---

# Auto-Research Program

## Current Priority

Designing a safe conversational Telegram control plane for a local
multi-provider Auto-Research platform, inspired by OpenClaw-style chat
operations but constrained by policy-gated action execution.

## Research Questions

1. How to parse natural language commands safely into structured actions?
2. What policy model prevents dangerous actions via mobile interface?
3. How to compare provider/model quality across LM Studio, Ollama, vLLM?
4. What makes karpathy/autoresearch operationally efficient, and which ideas transfer?

## Quality Thresholds

- Minimum word count: 200
- Coverage score: >= 0.3
- Structure score: >= 0.4
- Pass threshold: >= 0.5

## Constraints

- All execution local (no cloud LLM APIs)
- No arbitrary shell execution from Telegram
- Policy guard on all Telegram-triggered actions
- Traditional Chinese output for research notes
- Source-prepared bounded research runs (no autonomous web scraping in v3.14)

## Sources

- Local provider endpoints (LM Studio, Ollama)
- Existing research notes in output/notes/
- Memory records in knowledge/memory-records/
- karpathy/autoresearch design analysis
