# System Validation Report

Date: 2026-03-23
Workspace: `C:\Users\User\Desktop\新增資料夾 (7)`

## Scope

This run validated the current local self-evolving development platform end to end:

- full automated test suite
- live Telegram operator flow with a real user session
- real provider chat / synthesis / memory / search-backed research
- stress test and benchmark smoke
- web verification against current official documentation

## Fixes Applied During Validation

Two real product gaps surfaced during the run and were fixed immediately:

1. `src/auto_research/services/llm_provider.py`
   - `call_text()` now retries once on transient provider-side `HTTP 5xx` failures before fallback/recovery.
   - This resolved a real LM Studio chat failure seen in `verify_e2e.py`.

2. `src/auto_research/search/source_fetcher.py`
   - `SourceFetcher` now retries once with a restricted insecure TLS context only when the initial fetch fails with certificate verification errors.
   - This resolved official-doc fetch failures caused by the local certificate chain while still preserving normal verified TLS behavior for non-error cases.

Supporting regression coverage was added in:

- `tests/test_runtime_upgrades.py`
- `tests/test_search.py`

## Test Results

### Full Pytest

- Command: `python -m pytest -q --no-cov`
- Result: `420 passed, 10 deselected`
- Runtime: `15.64s`

### Telegram Live Automation

- Artifact: `output/telegram_live_automation.json`
- Result: `9/9 LIVE_PASS`
- Validated flows:
  - safe status
  - confirmation execute
  - natural-language intent parsing
  - chat fallback
  - dangerous command rejection
  - confirmation expiry
  - source search
  - search-backed research
  - provider diagnose

### Pi Runtime Smoke

- Surface: local HTTP runtime on `127.0.0.1`
- Result: pass
- Verified endpoints:
  - `GET /health`
  - `POST /v1/doctor`
- Observed provider selection during smoke:
  - effective provider: `lmstudio`
  - effective model: `nvidia/nemotron-3-nano`

### Stress Test

- Artifact: `output/stress-1n46xyi2/output/stress-report.json`
- Result: all checks passed
- Highlights:
  - LLM latency avg: `19.826s`
  - provider selection avg: `2.521s`
  - readiness cache cold/warm: `7.625s / 0.0s`
  - memory index rebuild: `0.477s`

### Benchmark Smoke

- Command: `python scripts/benchmark_models.py --isolated --model nvidia/nemotron-3-nano`
- Result: completed successfully
- Observed metrics:
  - latency: `119.8s`
  - words: `62`
  - coverage: `0.038`
  - structure: `0.84`
  - provenance: `0.253`
  - quality gate: `FAIL`

## Real Research Workflow Run

### Executed Topic

`LM Studio structured output Ollama structured outputs vLLM structured outputs docs`

### Session

- Session ID: `tg-search-lm-studio-structured-output-ollama-structured-ou-1774264952`
- Summary artifact: `output/system_research_run_structured_outputs.json`
- Note: `output/notes/tg-search-lm-studio-structured-output-ollama-structured-ou-1774264952.md`
- Status: `output/research/tg-search-lm-studio-structured-output-ollama-structured-ou-1774264952/status.json`

### Runtime Outcome

- provider: `lmstudio`
- model: `nvidia/nemotron-3-nano`
- fetched official sources: `3`
- packed sections: `13`
- search provider: `duckduckgo_html`
- extractor: `heuristic_html`

### Quality Outcome

- quality score: `0.393`
- coverage score: `0.0`
- structure score: `0.84`
- provenance score: `0.493`
- quality gate: `FAIL`
- hallucination flag: `High unsupported content ratio: 100%`

### Interpretation

The workflow is operational: search, fetch, parse, synthesize, evaluate, and artifact writing all completed successfully against live sources.

The content quality is still not acceptable. The generated note was short and overly compressed, so the current quality gate considered the output unsupported relative to the packed source bundle. This is a synthesis quality issue, not a pipeline availability issue.

## Web Validation Against Official Sources

Validation date: 2026-03-23

Sources used:

- LM Studio docs: https://lmstudio.ai/docs/developer/openai-compat/structured-output
- Ollama docs: https://docs.ollama.com/capabilities/structured-outputs
- vLLM docs: https://docs.vllm.ai/en/latest/features/structured_outputs/

### Claim Check

1. LM Studio supports schema-enforced structured output through the OpenAI-compatible `/v1/chat/completions` API.
   - Verdict: confirmed
   - Evidence: LM Studio documents JSON-schema enforcement via `/v1/chat/completions`, `response_format`, and OpenAI client compatibility.

2. Ollama supports structured output through `format`, including raw JSON mode and schema mode.
   - Verdict: confirmed
   - Evidence: Ollama documents `format: "json"` and passing a schema object to `format`.

3. Ollama supports Pydantic and Zod-driven schemas.
   - Verdict: confirmed
   - Evidence: official examples show `Country.model_json_schema()` and `zodToJsonSchema()`.

4. vLLM supports structured outputs in its OpenAI-compatible server and can enforce JSON, regex, choice, grammar, and structural tags.
   - Verdict: confirmed
   - Evidence: vLLM documents structured outputs in online serving with `extra_body` and `response_format`, plus grammar and JSON schema support.

5. vLLM can combine reasoning with structured outputs, but this requires explicit configuration in some reasoning setups.
   - Verdict: confirmed
   - Evidence: vLLM documents reasoning-output support and notes that some reasoning-enabled Qwen3 Coder setups require `--structured-outputs-config.enable_in_reasoning=True`.

6. Lower temperature can improve deterministic structured outputs.
   - Verdict: partially confirmed
   - Evidence: Ollama explicitly recommends lower temperature in its tips. The other official pages do not frame it as a universal rule, so this should be presented as a reliability heuristic rather than a guaranteed requirement across all backends.

### Validation Conclusion

The research note's central technical claims are broadly aligned with the official docs. The failure was not factual collapse of the pipeline; it was report quality. The current synthesis output is too terse and insufficiently source-grounded for the existing quality gate.

## Current System Assessment

### Healthy

- full test suite
- Telegram control plane
- provider selection and fallback
- real live search-backed workflow
- source fetching from official docs after TLS fallback fix
- evaluation and artifact generation
- stress test stability

### Still Weak

- synthesis faithfulness under quality gate
- output readability in some Traditional Chinese strings due lingering mojibake
- benchmark quality remains below acceptable threshold

## Recommended Next Engineering Step

Prioritize synthesis grounding rather than more infrastructure work:

1. strengthen the synthesis prompt so each bullet is tied to specific source evidence
2. require per-section citations or source tags before final note rendering
3. improve multilingual note rendering and eliminate remaining mojibake text paths
4. consider a second-pass verifier that rejects unsupported summary bullets before note finalization
