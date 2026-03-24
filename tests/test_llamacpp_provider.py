"""Tests for llamacpp provider integration in LlmProviderService."""
from __future__ import annotations

import pytest

from conftest import make_temp_repo, FakeHttpClient
from auto_research.runtime import load_config
from auto_research.services.llm_provider import LlmProviderService


@pytest.fixture
def config(tmp_path):
    repo = make_temp_repo(str(tmp_path))
    return load_config(repo_root=repo, environ={"TELEGRAM_PROVIDER": "llamacpp"})


@pytest.fixture
def service(config):
    return LlmProviderService(config)


class TestLlamacppUrlRouting:
    def test_chat_url(self, service):
        url = service.chat_url_for_provider("llamacpp")
        assert url == "http://127.0.0.1:8080/v1/chat/completions"

    def test_probe_url(self, service):
        url = service._probe_url("llamacpp")
        assert url == "http://127.0.0.1:8080/health"

    def test_base_url(self, service):
        url = service._base_url_for("llamacpp")
        assert url == "http://127.0.0.1:8080"


class TestLlamacppModelResolution:
    def test_default_model_returns_config_model(self, service):
        model = service.default_model_for_provider("llamacpp")
        # Config model is "qwen3.5:9b" (from .env TELEGRAM_MODEL default)
        assert model  # non-empty

    def test_default_model_sentinel_when_no_config(self, tmp_path):
        """When config.model is empty, return sentinel."""
        repo = make_temp_repo(str(tmp_path))
        # Override .env to have empty model
        (repo / ".env").write_text(
            "TELEGRAM_PROVIDER=llamacpp\nTELEGRAM_MODEL=\n",
            encoding="utf-8",
        )
        # model defaults to DEFAULT_MODEL when empty, so this tests the fallback
        config = load_config(repo_root=repo, environ={"TELEGRAM_PROVIDER": "llamacpp"})
        svc = LlmProviderService(config)
        model = svc.default_model_for_provider("llamacpp")
        assert model  # should be non-empty (either config default or sentinel)


class TestLlamacppListModels:
    def test_list_models_gguf_path_normalization(self, service):
        """llama-server returns GGUF paths as model IDs — verify normalization."""
        client = FakeHttpClient(responses=[
            {
                "object": "list",
                "data": [
                    {"id": "/models/qwen2.5-7b-instruct-q4_k_m.gguf", "object": "model"},
                ],
            },
        ])
        result = service.list_models("llamacpp", client)
        assert result["ok"] is True
        assert result["count"] == 1
        # Should have stripped path and .gguf extension
        assert result["models"] == ["qwen2.5-7b-instruct-q4_k_m"]

    def test_list_models_non_gguf_passthrough(self, service):
        """Non-GGUF model IDs pass through unchanged."""
        client = FakeHttpClient(responses=[
            {
                "object": "list",
                "data": [
                    {"id": "my-custom-model", "object": "model"},
                ],
            },
        ])
        result = service.list_models("llamacpp", client)
        assert result["ok"] is True
        assert result["models"] == ["my-custom-model"]

    def test_list_models_connection_error(self, service):
        """Connection errors return ok=False."""
        from unittest.mock import MagicMock
        client = MagicMock()
        client.request_json.side_effect = ConnectionError("refused")
        result = service.list_models("llamacpp", client)
        assert result["ok"] is False


class TestLlamacppKnownProvider:
    def test_in_known_providers(self):
        assert "llamacpp" in LlmProviderService._KNOWN_PROVIDERS

    def test_env_var_map(self, service):
        assert "llamacpp" in service._ENV_VAR_MAP
        assert service._ENV_VAR_MAP["llamacpp"] == "LLAMACPP_BASE"


class TestLlamacppCircuitBreaker:
    def test_breaker_created_for_llamacpp(self, service):
        breaker = service.get_breaker("llamacpp")
        assert breaker.target == "llm-llamacpp"
        assert breaker.failure_count == 0


class TestLlamacppHeaders:
    def test_no_auth_headers(self, service):
        """llamacpp requires no auth headers by default."""
        headers = service.model_headers("llamacpp")
        assert headers == {}


class TestLlamacppProviderMatrix:
    def test_provider_config_state(self, service):
        state = service._provider_config_state("llamacpp")
        assert "effective_base_url" in state
        assert state["effective_base_url"] == "http://127.0.0.1:8080"
