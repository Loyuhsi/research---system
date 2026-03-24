from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import FakeHttpClient, create_test_orchestrator, make_temp_repo

from auto_research.evaluation import EvaluationRecord, EvaluationStore
from auto_research.http_client import HttpClientError, JsonHttpClient
from auto_research.integrations.pi.runtime import PiRuntimeServer
from auto_research.runtime import load_config
from auto_research.services.llm_provider import LlmProviderService
from auto_research.services.synthesizer import SynthesizerService
from auto_research.services.vault import VaultService


def _json_request(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_call_structured_accepts_reasoning_content_for_lmstudio(tmp_path):
    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})
    llm = LlmProviderService(config)
    fake = FakeHttpClient(
        responses=[
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": '{"intent":"status","args":{},"confidence":0.91}',
                        }
                    }
                ]
            }
        ]
    )
    schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "args": {"type": "object"},
            "confidence": {"type": "number"},
        },
        "required": ["intent", "args", "confidence"],
        "additionalProperties": False,
    }
    result = llm.call_structured(
        "lmstudio",
        fake,
        model="test-model",
        messages=[{"role": "user", "content": "status"}],
        schema=schema,
    )
    assert result["data"]["intent"] == "status"
    assert result["response_mode"] == "reasoning_content"


def test_call_text_recovers_final_answer_without_leaking_reasoning(tmp_path):
    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})
    llm = LlmProviderService(config)
    fake = FakeHttpClient(
        responses=[
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "hidden chain of thought",
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": '{"final_answer":"可見答案"}',
                        }
                    }
                ]
            },
        ]
    )
    result = llm.call_text(
        "lmstudio",
        fake,
        model="test-model",
        messages=[{"role": "user", "content": "請回答"}],
        allow_fallback=False,
    )
    assert result["content"] == "可見答案"
    assert "hidden chain of thought" not in result["content"]
    assert str(result["response_mode"]).startswith("structured:")


def test_call_text_retries_once_on_transient_http_500(tmp_path):
    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})
    llm = LlmProviderService(config)

    class FlakyClient(JsonHttpClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def request_json(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise HttpClientError("HTTP 500: Internal Server Error")
            return {"choices": [{"message": {"content": "OK"}}]}

    client = FlakyClient()
    result = llm.call_text(
        "lmstudio",
        client,
        model="test-model",
        messages=[{"role": "user", "content": "status"}],
        allow_fallback=False,
    )
    assert result["content"] == "OK"
    assert result["retry_count"] == 1
    assert client.calls == 2


def test_orchestrator_provider_override_and_show_report(tmp_path):
    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})
    orch = create_test_orchestrator(config)

    session_id = "20260323-120000-bounded-web-search"
    topic = "Bounded web search integration for a local multi-provider Auto-Research platform"
    layout = config.resolve_layout(session_id)
    layout.ensure()
    layout.note_path.write_text(
        "\n".join(
            [
                "---",
                f'topic: "{topic}"',
                'provider: "lmstudio"',
                'model: "test-model"',
                "sources_count: 3",
                "---",
                "",
                "## 摘要",
                "內容",
            ]
        ),
        encoding="utf-8",
    )
    layout.status_path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "sources": [{"url": "https://example.com"}] * 3,
                "quality": {
                    "coverage_score": 0.6,
                    "structure_score": 0.8,
                    "provenance_score": 0.7,
                    "passed": True,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = EvaluationStore(repo / "knowledge" / "evaluations")
    store.save(
        EvaluationRecord(
            eval_id="qg-bounded-web-search",
            eval_type="quality_gate",
            baseline_score=0.0,
            candidate_score=0.6,
            passed=True,
            metadata={"session_id": session_id},
        )
    )

    override = orch.set_provider_override("lmstudio", session_key="telegram:1")
    assert override["effective_provider"] == "lmstudio"
    report = orch.show_report()
    assert report["session_id"] == session_id
    assert report["topic"] == topic
    assert report["quality_score"] > 0.0


def test_non_ascii_repo_root_loads_config(tmp_path):
    repo = make_temp_repo(str(tmp_path / "測試工作區"))
    config = load_config(repo_root=repo, environ={})
    assert "測試工作區" in str(config.repo_root)


def test_pi_runtime_health_and_chat(tmp_path):
    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})
    http = FakeHttpClient(responses=[{"choices": [{"message": {"content": "Pi 回覆"}}]}])
    orch = create_test_orchestrator(config, http_client=http)
    orch._retrieve_chat_context = lambda text: {"memory_hits": [], "skill_hits": [], "vector_backend": "metadata+lexical"}
    server = PiRuntimeServer(orch, port=0)
    server.start_in_thread()
    try:
        status, health = _json_request(f"http://127.0.0.1:{server.port}/health")
        assert status == 200
        assert health["ok"] is True

        status, payload = _json_request(
            f"http://127.0.0.1:{server.port}/v1/chat",
            {"message": "你好", "session_key": "pi:test"},
        )
        assert status == 200
        assert payload["result"]["reply"] == "Pi 回覆"
    finally:
        server.shutdown()


def test_pi_runtime_rejects_bad_schema(tmp_path):
    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})
    orch = create_test_orchestrator(config)
    server = PiRuntimeServer(orch, port=0)
    server.start_in_thread()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/v1/chat",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=10)
        assert exc_info.value.code == 400
    finally:
        server.shutdown()


def test_results_tsv_header_includes_new_observability_columns(tmp_path):
    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})
    synth = SynthesizerService(config, JsonHttpClient(), LlmProviderService(config), VaultService(config))

    class FakeQR:
        coverage_score = 0.1
        structure_score = 0.5
        provenance_score = 0.3
        word_count = 100
        passed = False

    synth._append_results_tsv(
        session_id="test-observability",
        provider="lmstudio",
        model="test-model",
        quality_report=FakeQR(),
        latency_s=1.0,
        response_mode="structured:reasoning_content",
        retry_count=1,
        search_provider="duckduckgo_html",
        extractor="heuristic_html",
        trace_id="trace-xyz",
    )
    header = (repo / "results.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert "response_mode" in header
    assert "trace_id" in header


def test_synthesizer_extracts_provider_context_window(tmp_path):
    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})
    synth = SynthesizerService(config, JsonHttpClient(), LlmProviderService(config), VaultService(config))
    error_message = (
        'HTTP 400: {"error":"The number of tokens to keep from the initial prompt '
        'is greater than the context length (n_keep: 4261>= n_ctx: 4096)."}'
    )
    assert synth._extract_context_window(error_message) == 4096
    assert synth._source_bundle_byte_budget(4096) < synth._source_bundle_byte_budget(8192)
