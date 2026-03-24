"""Unit tests for multi-provider readiness features (v3.10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import FakeHttpClient, make_temp_repo
from auto_research.http_client import JsonHttpClient
from auto_research.runtime import load_config
from auto_research.services.llm_provider import LlmProviderService


def _make_llm(tmp_path: Path) -> tuple:
    repo = make_temp_repo(str(tmp_path))
    config = load_config(repo_root=repo, environ={})
    llm = LlmProviderService(config)
    return llm, config


class TestListModels:
    def test_ollama_models(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[
            {"models": [{"name": "qwen:7b"}, {"name": "llama3:8b"}]},
        ])
        result = llm.list_models("ollama", fake)
        assert result["ok"]
        assert result["count"] == 2
        assert "qwen:7b" in result["models"]

    def test_lmstudio_models(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[
            {"data": [{"id": "nemotron-nano"}, {"id": "phi-3"}]},
        ])
        result = llm.list_models("lmstudio", fake)
        assert result["ok"]
        assert result["count"] == 2
        assert "nemotron-nano" in result["models"]

    def test_models_error(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)

        class FailingClient(JsonHttpClient):
            def request_json(self, *a, **kw):
                raise ConnectionError("provider down")

        result = llm.list_models("ollama", FailingClient())
        assert not result["ok"]


class TestCheckInference:
    def test_inference_success(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[
            {"choices": [{"message": {"content": "OK"}}]},
        ])
        result = llm.check_inference("ollama", fake)
        assert result["ok"]
        assert result["response_snippet"] == "OK"

    def test_inference_empty_content(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[{"choices": [{"message": {"content": ""}}]}])
        result = llm.check_inference("ollama", fake)
        assert not result["ok"]


class TestCheckEmbeddings:
    def test_embeddings_ollama(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[{"embedding": [0.1, 0.2]}])
        result = llm.check_embeddings("ollama", fake)
        assert result["ok"]

    def test_embeddings_lmstudio(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[{"data": [{"embedding": [0.1]}]}])
        result = llm.check_embeddings("lmstudio", fake)
        assert result["ok"]


class TestProviderMatrix:
    def test_matrix_structure(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[])
        matrix = llm.provider_capability_matrix(fake)
        assert "ollama" in matrix
        assert "lmstudio" in matrix
        for provider, info in matrix.items():
            assert "health" in info
            assert "models" in info
            assert "inference" in info
            assert "embeddings" in info
            assert "breaker" in info
            assert "is_primary" in info


class TestSelectProvider:
    def test_explicit_available(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[
            {},  # health
            {"choices": [{"message": {"content": "OK"}}]},  # inference
        ])
        result = llm.select_provider(fake, preference="ollama")
        assert result["provider"] == "ollama"
        assert result["reason"] == "explicit"

    def test_explicit_skip_inference(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[{}])  # health only
        result = llm.select_provider(fake, preference="ollama", verify_inference=False)
        assert result["provider"] == "ollama"
        assert result["reason"] == "explicit"

    def test_explicit_health_ok_inference_fail_fallback(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[
            {},  # ollama health OK
            {"choices": [{"message": {"content": ""}}]},  # ollama inference FAIL (primary, no discover needed)
            {},  # lmstudio health OK
            {"data": [{"id": "test-model"}]},  # lmstudio list_models (discover)
            {"choices": [{"message": {"content": "OK"}}]},  # lmstudio inference OK
        ])
        result = llm.select_provider(fake, preference="ollama")
        assert result["provider"] == "lmstudio"
        assert result["reason"] == "fallback_inference_fail"

    def test_auto_primary_inference_fail_fallback(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[
            {},  # primary (ollama) health OK
            {"choices": [{"message": {"content": ""}}]},  # primary inference FAIL
            {},  # lmstudio health OK
            {"data": [{"id": "test-model"}]},  # lmstudio list_models (discover)
            {"choices": [{"message": {"content": "OK"}}]},  # lmstudio inference OK
        ])
        result = llm.select_provider(fake, preference="auto")
        assert result["reason"] == "auto_fallback_inference"

    def test_auto_none_available(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)

        class FailingClient(JsonHttpClient):
            def request_json(self, *a, **kw):
                raise ConnectionError("provider down")

        result = llm.select_provider(FailingClient(), preference="auto")
        assert result["reason"] == "auto_none_available"


class TestExtractOllamaNative:
    def test_native_message_format(self):
        resp = {"message": {"content": "OK"}, "done": True}
        assert LlmProviderService._extract_ollama_native(resp) == "OK"

    def test_native_response_field(self):
        resp = {"response": "OK", "done": True}
        assert LlmProviderService._extract_ollama_native(resp) == "OK"

    def test_think_only_stripped(self):
        resp = {"message": {"content": "<think>reasoning</think>"}}
        assert LlmProviderService._extract_ollama_native(resp) == ""

    def test_think_plus_content(self):
        resp = {"message": {"content": "<think>thinking</think>OK"}}
        assert LlmProviderService._extract_ollama_native(resp) == "OK"

    def test_openai_fallback_to_native(self):
        resp = {"message": {"content": "native answer"}}
        assert LlmProviderService.extract_message_content(resp) == "native answer"


class TestCheckEmbeddingsFallback:
    def test_ollama_new_endpoint(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[
            {"embeddings": [[0.1, 0.2]]},  # /api/embed succeeds
        ])
        result = llm.check_embeddings("ollama", fake)
        assert result["ok"]

    def test_ollama_fallback_to_legacy(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)

        class FallbackClient(JsonHttpClient):
            def __init__(self):
                super().__init__()
                self._call_count = 0
            def request_json(self, *a, **kw):
                self._call_count += 1
                if self._call_count == 1:
                    raise ConnectionError("404")
                return {"embedding": [0.1, 0.2]}

        result = llm.check_embeddings("ollama", FallbackClient())
        assert result["ok"]


class TestVllmModels:
    def test_vllm_models_success(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[
            {"data": [{"id": "meta-llama/Llama-2-7b"}, {"id": "mistralai/Mistral-7B"}]},
        ])
        result = llm.list_models("vllm", fake)
        assert result["ok"]
        assert result["count"] == 2
        assert "meta-llama/Llama-2-7b" in result["models"]

    def test_vllm_models_error(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)

        class FailingClient(JsonHttpClient):
            def request_json(self, *a, **kw):
                raise ConnectionError("vllm down")

        result = llm.list_models("vllm", FailingClient())
        assert not result["ok"]


class TestVllmInference:
    def test_vllm_inference_success(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[
            {"data": [{"id": "test-model"}]},  # list_models for discover
            {"choices": [{"message": {"content": "OK"}}]},  # inference
        ])
        result = llm.check_inference("vllm", fake)
        assert result["ok"]
        assert result["model"] == "test-model"
        assert result["model_source"] == "discovered"

    def test_vllm_inference_no_models(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[{"data": []}])  # empty models
        result = llm.check_inference("vllm", fake)
        assert not result["ok"]
        assert "no model" in str(result.get("error", ""))


class TestVllmEmbeddings:
    def test_vllm_embeddings_success(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[{"data": [{"embedding": [0.1, 0.2]}]}])
        result = llm.check_embeddings("vllm", fake)
        assert result["ok"]

    def test_vllm_embeddings_fail(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)

        class FailingClient(JsonHttpClient):
            def request_json(self, *a, **kw):
                raise ConnectionError("no embeddings")

        result = llm.check_embeddings("vllm", FailingClient())
        assert not result["ok"]


class TestDiscoverModel:
    def test_discover_from_list(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[
            {"data": [{"id": "discovered-model-v1"}]},
        ])
        model = llm._discover_model("vllm", fake)
        assert model == "discovered-model-v1"

    def test_discover_no_models(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[{"data": []}])
        model = llm._discover_model("vllm", fake)
        assert model is None


class TestReadinessCache:
    def test_cache_hit(self):
        from auto_research.services.llm_provider import _ReadinessCache
        cache = _ReadinessCache(ttl=10.0)
        cache.set("ollama:health", {"ok": True})
        assert cache.get("ollama:health") == {"ok": True}

    def test_cache_miss_expired(self):
        from auto_research.services.llm_provider import _ReadinessCache
        import time
        cache = _ReadinessCache(ttl=0.01)
        cache.set("ollama:health", {"ok": True})
        time.sleep(0.02)
        assert cache.get("ollama:health") is None

    def test_cache_invalidate_prefix(self):
        from auto_research.services.llm_provider import _ReadinessCache
        cache = _ReadinessCache(ttl=30.0)
        cache.set("ollama:health", {"ok": True})
        cache.set("ollama:inference", {"ok": False})
        cache.set("lmstudio:health", {"ok": True})
        cache.invalidate(prefix="ollama:")
        assert cache.get("ollama:health") is None
        assert cache.get("ollama:inference") is None
        assert cache.get("lmstudio:health") == {"ok": True}

    def test_cache_invalidate_all(self):
        from auto_research.services.llm_provider import _ReadinessCache
        cache = _ReadinessCache(ttl=30.0)
        cache.set("ollama:health", {"ok": True})
        cache.set("lmstudio:health", {"ok": True})
        cache.invalidate()
        assert cache.get("ollama:health") is None
        assert cache.get("lmstudio:health") is None


class TestThreeProviderFallback:
    def test_primary_second_fail_third_ok(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[
            {},  # ollama health OK
            {"choices": [{"message": {"content": ""}}]},  # ollama inference FAIL
            {},  # lmstudio health OK
            {"data": [{"id": "ls-model"}]},  # lmstudio discover
            {"choices": [{"message": {"content": ""}}]},  # lmstudio inference FAIL
            {},  # vllm health OK
            {"data": [{"id": "vllm-model"}]},  # vllm discover
            {"choices": [{"message": {"content": "OK"}}]},  # vllm inference OK
        ])
        result = llm.select_provider(fake, preference="auto")
        assert result["provider"] == "vllm"
        assert result["reason"] == "auto_fallback_inference"
        assert result["model"] == "vllm-model"

    def test_vllm_in_matrix(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fake = FakeHttpClient(responses=[])
        matrix = llm.provider_capability_matrix(fake)
        assert "vllm" in matrix
        assert "health" in matrix["vllm"]
        assert "is_primary" in matrix["vllm"]

    def test_fallback_providers_excludes_current(self, tmp_path: Path):
        llm, _ = _make_llm(tmp_path)
        fb = llm._fallback_providers("ollama")
        assert "ollama" not in fb
        assert "lmstudio" in fb
        assert "vllm" in fb


class TestClassifyError:
    def test_timeout(self):
        assert LlmProviderService._classify_error("Connection timed out") == "timeout"

    def test_connection_refused(self):
        assert LlmProviderService._classify_error("Connection refused at 127.0.0.1") == "connection_refused"

    def test_gpu(self):
        assert LlmProviderService._classify_error("CUDA out of memory") == "gpu_contention"

    def test_unknown(self):
        assert LlmProviderService._classify_error("something weird") == "unknown"
