"""Tests for the Telegram conversational control plane."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from auto_research.telegram.action_registry import ActionRegistry
from auto_research.telegram.conversation_state import CONFIRMATION_TTL_SECONDS, ConversationState
from auto_research.telegram.intent_parser import KNOWN_INTENTS, IntentParser, ParsedIntent
from auto_research.telegram.policy_guard import ActionPolicy, INTENT_POLICIES, PolicyGuard


class TestIntentParser:
    @pytest.fixture
    def parser(self, tmp_path):
        from auto_research.runtime import load_config
        from conftest import make_temp_repo

        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={})
        return IntentParser(config=config)

    def test_keyword_status(self, parser):
        result = parser.parse("查狀態")
        assert result.intent == "status"
        assert result.confidence >= 0.9
        assert result.parse_layer == "keyword"

    def test_keyword_status_english(self, parser):
        result = parser.parse("show me the health status")
        assert result.intent == "status"

    def test_keyword_search_memory(self, parser):
        result = parser.parse("搜尋記憶 python")
        assert result.intent == "search_memory"

    def test_keyword_list_providers(self, parser):
        result = parser.parse("list providers")
        assert result.intent == "list_providers"

    def test_keyword_export_obsidian(self, parser):
        result = parser.parse("export to obsidian vault")
        assert result.intent == "export_obsidian"

    def test_keyword_start_research(self, parser):
        result = parser.parse("開始研究 Python Testing")
        assert result.intent == "start_research"
        assert "topic" in result.args

    def test_keyword_diagnose_provider(self, parser):
        result = parser.parse("diagnose ollama provider")
        assert result.intent == "diagnose_provider"
        assert result.args.get("provider") == "ollama"

    def test_chat_fallback_no_action_verbs(self, parser):
        result = parser.parse("今天天氣如何")
        assert result.intent == "chat"
        assert result.confidence == 1.0

    def test_ambiguous_with_action_verbs_gets_clarification(self, parser):
        result = parser.parse("幫我看看目前 provider 狀況")
        assert result.intent in {"list_providers", "chat"}
        if result.intent == "chat":
            assert result.clarification

    def test_known_intents_complete(self):
        assert "status" in KNOWN_INTENTS
        assert "chat" in KNOWN_INTENTS
        assert "start_research" in KNOWN_INTENTS
        assert "export_obsidian" in KNOWN_INTENTS


class TestPolicyGuard:
    @pytest.fixture
    def guard(self, tmp_path):
        return PolicyGuard(telemetry_path=tmp_path / "telemetry.jsonl")

    def test_safe_action_allowed(self, guard):
        intent = ParsedIntent("status", {}, 0.95, "", "keyword")
        check = guard.check(intent)
        assert check["allowed"] is True
        assert check["policy"] == "safe"

    def test_confirm_action_needs_approval(self, guard):
        intent = ParsedIntent("start_research", {"topic": "test"}, 0.9, "", "keyword")
        check = guard.check(intent)
        assert check["allowed"] is True
        assert check["policy"] == "confirm"

    def test_unknown_action_disabled(self, guard):
        intent = ParsedIntent("unknown_action", {}, 0.9, "", "keyword")
        check = guard.check(intent)
        assert check["allowed"] is False
        assert check["policy"] == "disabled"

    def test_dangerous_chat_disabled(self, guard):
        intent = ParsedIntent("chat", {"text": "execute shell rm -rf /"}, 1.0, "", "keyword")
        check = guard.check(intent)
        assert check["allowed"] is False
        assert check["policy"] == "disabled"

    def test_audit_log_written(self, guard, tmp_path):
        intent = ParsedIntent("status", {}, 0.95, "", "keyword")
        guard.log_action(intent, "OK result")
        log_path = tmp_path / "telemetry.jsonl"
        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert "telegram_action" in content
        assert "status" in content

    def test_dangerous_args_in_non_chat_intent_disabled(self, guard):
        """Regression: dangerous patterns in any arg (not just chat text) must be blocked."""
        intent = ParsedIntent("start_research", {"topic": "rm -rf /important"}, 0.9, "", "keyword")
        check = guard.check(intent)
        assert check["allowed"] is False
        assert check["policy"] == "disabled"

    def test_dangerous_powershell_in_args_disabled(self, guard):
        intent = ParsedIntent("search_sources", {"query": "use powershell to get data"}, 0.9, "", "keyword")
        check = guard.check(intent)
        assert check["allowed"] is False
        assert check["policy"] == "disabled"

    def test_all_known_intents_have_policy(self):
        for intent in KNOWN_INTENTS:
            assert intent in INTENT_POLICIES, f"Missing policy for {intent}"


class TestConversationState:
    def test_pending_flow(self):
        state = ConversationState()
        intent = ParsedIntent("start_research", {"topic": "test"}, 0.9, "", "keyword")
        state.set_pending(123, intent)
        confirmed = state.confirm_pending(123)
        assert confirmed is not None
        assert confirmed.intent == "start_research"
        assert state.confirm_pending(123) is None

    def test_clear_pending(self):
        state = ConversationState()
        intent = ParsedIntent("export_obsidian", {}, 0.9, "", "keyword")
        state.set_pending(123, intent)
        state.clear_pending(123)
        assert state.confirm_pending(123) is None

    def test_expired_confirmation_returns_none(self):
        state = ConversationState()
        intent = ParsedIntent("start_research", {}, 0.9, "", "keyword")
        state.set_pending(123, intent)
        chat_state = state.get(123)
        chat_state.pending_timestamp = time.monotonic() - CONFIRMATION_TTL_SECONDS - 10
        assert state.confirm_pending(123) is None

    def test_no_pending_returns_none(self):
        state = ConversationState()
        assert state.confirm_pending(999) is None


class TestActionRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        from auto_research.runtime import load_config
        from auto_research.services.llm_provider import LlmProviderService
        from conftest import FakeHttpClient, make_temp_repo

        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={})
        mock_orch = MagicMock()
        mock_orch.config = config
        mock_orch.llm = LlmProviderService(config)
        mock_orch.http_client = FakeHttpClient()
        mock_orch.doctor.return_value = {
            "provider": "lmstudio",
            "model": "test",
            "services": {"ollama": {"ok": True}, "lmstudio": {"ok": True}, "vllm": {"ok": False}},
            "provider_selection": {"provider": "lmstudio", "reason": "auto_primary"},
        }
        mock_orch.show_report.return_value = {
            "session_id": "test-session",
            "topic": "Test Topic",
            "provider": "lmstudio",
            "model": "test-model",
            "source_count": 3,
            "quality_score": 0.72,
            "quality": {"coverage_score": 0.6, "structure_score": 0.8, "provenance_score": 0.7},
            "memory_status": {"approved_count": 1, "draft_count": 0},
            "evaluation_status": {"exists": True, "eval_id": "qg-test", "passed": True},
            "telemetry_summary": {"count": 2},
            "trace_summary": {"exists": True, "trace_id": "trace-123", "span_count": 4},
            "note_path": "C:/tmp/test.md",
        }
        mock_orch.set_provider_override.return_value = {
            "provider": "lmstudio",
            "effective_provider": "lmstudio",
            "model": "test-model",
        }
        mock_orch.memory_search.return_value = {"memory_hits": [{"id": "m1", "title": "Test", "score": 0.8}]}
        return ActionRegistry(orchestrator=mock_orch, config=config)

    def test_status_calls_doctor(self, registry):
        result = registry.execute("status", {}, "test-session")
        assert "系統狀態" in result
        assert "lmstudio" in result

    def test_search_memory(self, registry):
        result = registry.execute("search_memory", {"query": "python"}, "test-session")
        assert "記憶搜尋結果" in result

    def test_unknown_returns_disabled(self, registry):
        result = registry.execute("unknown_action", {}, "test-session")
        assert "Telegram" in result

    def test_list_sessions_empty(self, registry):
        result = registry.execute("list_sessions", {}, "test-session")
        assert "session" in result.lower()

    def test_chat_calls_orchestrator(self, registry):
        registry._orchestrator.chat.return_value = {"reply": "Hello there"}
        result = registry.execute("chat", {"text": "hi"}, "test-session")
        assert "Hello there" in result
        assert result.startswith("回覆：")

    def test_chat_empty_text(self, registry):
        result = registry.execute("chat", {}, "test-session")
        assert result == "對話內容是空的。"

    def test_list_providers_returns_matrix(self, registry):
        registry._orchestrator.llm.provider_capability_matrix = MagicMock(
            return_value={
                "lmstudio": {
                    "health_ready": True,
                    "inference_ready": True,
                    "embedding_ready": False,
                    "is_primary": True,
                },
                "ollama": {
                    "health_ready": True,
                    "inference_ready": False,
                    "embedding_ready": False,
                    "is_primary": False,
                },
            }
        )
        result = registry.execute("list_providers", {}, "test-session")
        assert "Provider Matrix" in result
        assert "lmstudio" in result

    def test_diagnose_provider_success(self, registry):
        registry._orchestrator.llm.check_inference = MagicMock(
            return_value={"ok": True, "model": "test-model", "model_source": "config", "latency_ms": 120}
        )
        registry._orchestrator.llm.check_embeddings = MagicMock(return_value={"ok": True})
        result = registry.execute("diagnose_provider", {"provider": "lmstudio"}, "test-session")
        assert "Provider 診斷" in result
        assert "Inference: OK" in result

    def test_export_obsidian_handler(self, registry):
        result = registry.execute("export_obsidian", {}, "test-session")
        assert "Obsidian" in result

    def test_show_report_formats_result(self, registry):
        result = registry.execute("show_report", {}, "test-session")
        assert "研究報告" in result
        assert "Test Topic" in result
        assert "trace-123" in result

    def test_select_provider_session_scope(self, registry):
        result = registry.execute("select_provider", {"provider": "lmstudio"}, "test-session")
        assert "provider" in result.lower()
        registry._orchestrator.set_provider_override.assert_called_once_with(
            "lmstudio",
            session_key="test-session",
            global_scope=False,
        )

    def test_execute_proposal(self, registry):
        from auto_research.telegram.policy_guard import ActionProposal

        intent = ParsedIntent("status", {}, 0.95, "", "keyword")
        proposal = ActionProposal(actor_chat_id=111, intent=intent, risk_level="low", policy=ActionPolicy.SAFE)
        result = registry.execute_proposal(proposal, "test-session")
        assert "系統狀態" in result


class TestIntentParserLLMLayer:
    @pytest.fixture
    def parser_with_llm(self, tmp_path):
        from auto_research.runtime import load_config
        from conftest import make_temp_repo

        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={})
        return IntentParser(config=config)

    def test_llm_validate_valid_json(self, parser_with_llm):
        content = '{"intent": "diagnose_provider", "args": {"provider": "ollama"}, "confidence": 0.9}'
        result = parser_with_llm._validate_llm_output(content, "check ollama")
        assert result is not None
        assert result.intent == "diagnose_provider"
        assert result.args == {"provider": "ollama"}
        assert result.confidence == 0.9
        assert result.parse_layer == "llm"

    def test_llm_validate_non_json(self, parser_with_llm):
        result = parser_with_llm._validate_llm_output("I don't understand", "some text")
        assert result is None

    def test_llm_validate_unknown_intent(self, parser_with_llm):
        content = '{"intent": "delete_everything", "args": {}, "confidence": 0.95}'
        result = parser_with_llm._validate_llm_output(content, "delete everything")
        assert result is None

    def test_llm_validate_missing_fields(self, parser_with_llm):
        content = '{"args": {"query": "test"}, "confidence": 0.8}'
        result = parser_with_llm._validate_llm_output(content, "search test")
        assert result is None

    def test_llm_validate_strips_disallowed_args(self, parser_with_llm):
        content = '{"intent": "status", "args": {"evil_param": "drop"}, "confidence": 0.9}'
        result = parser_with_llm._validate_llm_output(content, "status check")
        assert result is not None
        assert "evil_param" not in result.args

    def test_llm_validate_confidence_clamped(self, parser_with_llm):
        content = '{"intent": "status", "args": {}, "confidence": 5.0}'
        result = parser_with_llm._validate_llm_output(content, "status")
        assert result is not None
        assert result.confidence == 1.0


class TestSafetyCritical:
    def test_expired_confirmation_does_not_execute(self):
        state = ConversationState()
        intent = ParsedIntent("start_research", {"topic": "danger"}, 0.9, "", "keyword")
        state.set_pending(999, intent)
        chat_state = state.get(999)
        chat_state.pending_timestamp = time.monotonic() - CONFIRMATION_TTL_SECONDS - 10
        result = state.confirm_pending(999)
        assert result is None

    def test_mixed_signal_routing(self, tmp_path):
        from auto_research.runtime import load_config
        from conftest import make_temp_repo

        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={})
        parser = IntentParser(config=config)
        result = parser.parse("幫我查一下目前記憶裡有沒有 provider 相關內容")
        assert result.intent in ("search_memory", "chat")
        if result.intent == "chat":
            assert result.clarification or result.confidence == 1.0

    def test_policy_reject_disabled_intent(self, tmp_path):
        guard = PolicyGuard(telemetry_path=tmp_path / "tel.jsonl")
        intent = ParsedIntent("unknown_dangerous", {}, 0.95, "", "keyword")
        proposal = guard.propose(123, intent)
        assert proposal.policy == ActionPolicy.DISABLED
        assert proposal.risk_level == "high"

    def test_action_proposal_has_audit_id(self):
        from auto_research.telegram.policy_guard import ActionProposal

        intent = ParsedIntent("status", {}, 0.9, "", "keyword")
        proposal = ActionProposal(actor_chat_id=111, intent=intent, risk_level="low", policy=ActionPolicy.SAFE)
        assert len(proposal.audit_id) > 0
        assert len(proposal.timestamp) > 0


class TestSearchKeywordParsing:
    @pytest.fixture
    def parser(self, tmp_path):
        from auto_research.runtime import load_config
        from conftest import make_temp_repo

        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={})
        return IntentParser(config=config)

    def test_search_sources_keyword_zh(self, parser):
        result = parser.parse("搜尋來源 Telegram control plane")
        assert result.intent == "search_sources"
        assert result.args.get("query")

    def test_research_with_search_keyword_zh(self, parser):
        result = parser.parse("搜尋並研究 local multi-provider design")
        assert result.intent == "research_with_search"
        assert result.args.get("topic")

    def test_search_sources_keyword_en(self, parser):
        result = parser.parse("search sources for Python testing")
        assert result.intent == "search_sources"

    def test_research_with_search_keyword_en(self, parser):
        result = parser.parse("research with search about local LLM platform")
        assert result.intent == "research_with_search"


class TestActionRegistrySearch:
    @pytest.fixture
    def registry(self, tmp_path):
        from auto_research.runtime import load_config
        from auto_research.services.llm_provider import LlmProviderService
        from conftest import FakeHttpClient, make_temp_repo

        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={})
        mock_orch = MagicMock()
        mock_orch.config = config
        mock_orch.llm = LlmProviderService(config)
        mock_orch.http_client = FakeHttpClient()
        return ActionRegistry(orchestrator=mock_orch, config=config)

    def test_search_sources_empty_query(self, registry):
        result = registry.execute("search_sources", {}, "test-session")
        assert "請提供搜尋字詞" in result

    def test_research_with_search_no_topic_no_program(self, registry):
        result = registry.execute("research_with_search", {}, "test-session")
        assert "請提供研究主題" in result


class TestResultsTsv:
    def test_results_tsv_creates_header(self, tmp_path):
        from auto_research.http_client import JsonHttpClient
        from auto_research.runtime import load_config
        from auto_research.services.llm_provider import LlmProviderService
        from auto_research.services.synthesizer import SynthesizerService
        from auto_research.services.vault import VaultService
        from conftest import make_temp_repo

        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={})
        synth = SynthesizerService(config, JsonHttpClient(), LlmProviderService(config), VaultService(config))

        class FakeQR:
            coverage_score = 0.1
            structure_score = 0.5
            provenance_score = 0.3
            word_count = 100
            passed = False

        result = synth._append_results_tsv(
            session_id="test-tsv",
            provider="lmstudio",
            model="test-model",
            quality_report=FakeQR(),
            latency_s=1.0,
        )
        assert result is True
        tsv_path = repo / "results.tsv"
        assert tsv_path.exists()
        content = tsv_path.read_text(encoding="utf-8")
        assert "timestamp\t" in content
        assert "test-tsv" in content
        assert "cli_manual" in content

    def test_results_tsv_benchmark_run_kind(self, tmp_path):
        from auto_research.http_client import JsonHttpClient
        from auto_research.runtime import load_config
        from auto_research.services.llm_provider import LlmProviderService
        from auto_research.services.synthesizer import SynthesizerService
        from auto_research.services.vault import VaultService
        from conftest import make_temp_repo

        repo = make_temp_repo(str(tmp_path))
        config = load_config(repo_root=repo, environ={})
        synth = SynthesizerService(config, JsonHttpClient(), LlmProviderService(config), VaultService(config))

        class FakeQR:
            coverage_score = 0.2
            structure_score = 0.6
            provenance_score = 0.4
            word_count = 200
            passed = True

        synth._append_results_tsv(
            session_id="bench-1",
            provider="lmstudio",
            model="model-a",
            quality_report=FakeQR(),
            latency_s=5.0,
            run_kind="benchmark",
        )
        synth._append_results_tsv(
            session_id="bench-2",
            provider="lmstudio",
            model="model-b",
            quality_report=FakeQR(),
            latency_s=50.0,
            run_kind="benchmark",
        )
        content = (repo / "results.tsv").read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 3
        assert "model-a" in lines[1]
        assert "model-b" in lines[2]
        assert "benchmark" in lines[1]
        assert "benchmark" in lines[2]
