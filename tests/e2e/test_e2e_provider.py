"""E2E: Provider capability matrix and multi-provider smoke tests."""

from __future__ import annotations

import pytest

from .conftest import skip_no_provider, skip_no_ollama, skip_no_lmstudio


@pytest.mark.e2e
@skip_no_provider
class TestE2EProviderMatrix:
    def test_provider_matrix_reports_at_least_one(self, e2e_orchestrator):
        """Provider matrix should report at least one available provider."""
        doctor = e2e_orchestrator.doctor()
        matrix = doctor["provider_matrix"]
        assert isinstance(matrix, dict)
        available = [p for p, info in matrix.items() if info["health"]["ok"]]
        assert len(available) >= 1, f"No providers available: {matrix}"
        print(f"Available providers: {available}")

    def test_provider_selection_returns_valid(self, e2e_orchestrator):
        """Provider selection should pick an available provider."""
        doctor = e2e_orchestrator.doctor()
        selection = doctor["provider_selection"]
        assert "provider" in selection
        assert "reason" in selection
        print(f"Selected: {selection}")

    def test_inference_smoke(self, e2e_orchestrator, e2e_config):
        """At least one provider should pass inference check."""
        from auto_research.http_client import JsonHttpClient
        http_client = JsonHttpClient()
        matrix = e2e_orchestrator.llm.provider_capability_matrix(http_client)
        inference_ok = [p for p, info in matrix.items() if info["inference"].get("ok")]
        print(f"Inference OK providers: {inference_ok}")
        for p in inference_ok:
            info = matrix[p]["inference"]
            print(f"  {p}: model={info.get('model')}, latency={info.get('latency_ms')}ms")
        # At least one should work if any provider is up
        available = [p for p, info in matrix.items() if info["health"]["ok"]]
        if available:
            assert len(inference_ok) >= 1 or any(
                "timeout" in str(matrix[p]["inference"].get("error_category", ""))
                for p in available
            ), f"No inference OK and no timeout: {matrix}"


@pytest.mark.e2e
@skip_no_ollama
class TestE2EOllamaSpecific:
    def test_ollama_model_listing(self, e2e_orchestrator):
        from auto_research.http_client import JsonHttpClient
        result = e2e_orchestrator.llm.list_models("ollama", JsonHttpClient())
        assert result["ok"]
        assert result["count"] > 0
        print(f"Ollama models: {result['models'][:5]}")


@pytest.mark.e2e
@skip_no_lmstudio
class TestE2ELmStudioSpecific:
    def test_lmstudio_model_listing(self, e2e_orchestrator):
        from auto_research.http_client import JsonHttpClient
        result = e2e_orchestrator.llm.list_models("lmstudio", JsonHttpClient())
        assert result["ok"]
        assert result["count"] > 0
        print(f"LM Studio models: {result['models'][:5]}")

    def test_lmstudio_inference(self, e2e_orchestrator):
        from auto_research.http_client import JsonHttpClient
        result = e2e_orchestrator.llm.check_inference("lmstudio", JsonHttpClient())
        print(f"LM Studio inference: {result}")
        # May fail if model not loaded, but should at least not crash
        assert "ok" in result
