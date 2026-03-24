"""Tests for the Telegram bot (scripts/telegram_bot.py)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure scripts/ is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from telegram_bot import (
    AutoResearchTelegramBot,
    BotConfig,
    TelegramApi,
    load_bot_config,
    parse_allowed_chat_ids,
    parse_command_parts,
    split_message_chunks,
)

from auto_research.circuit_breaker import CircuitOpenError, STATE_OPEN, STATE_CLOSED
from auto_research.runtime import ConfigError, load_config
from conftest import FakeHttpClient, make_temp_repo


def _make_bot_config(repo_root: Path, allowed_ids: frozenset[int] | None = None) -> BotConfig:
    return BotConfig(
        repo_root=repo_root,
        token="test-token",
        allowed_chat_ids=allowed_ids if allowed_ids is not None else frozenset({111}),
        provider="ollama",
        model="test-model",
    )


# ---------------------------------------------------------------------------
# TelegramApi tests
# ---------------------------------------------------------------------------

class TelegramApiTests(unittest.TestCase):
    def test_get_me_success(self):
        http = FakeHttpClient(responses=[{"ok": True, "result": {"username": "testbot"}}])
        api = TelegramApi("tok", http)
        result = api.get_me()
        self.assertEqual(result["username"], "testbot")

    def test_get_me_failure_raises(self):
        http = FakeHttpClient(responses=[{"ok": False, "description": "Unauthorized"}])
        api = TelegramApi("tok", http)
        with self.assertRaises(RuntimeError):
            api.get_me()

    def test_get_updates_returns_list(self):
        updates = [{"update_id": 1, "message": {}}]
        http = FakeHttpClient(responses=[{"ok": True, "result": updates}])
        api = TelegramApi("tok", http)
        result = api.get_updates(offset=None, timeout=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["update_id"], 1)

    def test_send_message_chunks_long_text(self):
        http = FakeHttpClient(responses=[
            {"ok": True, "result": {}},
            {"ok": True, "result": {}},
        ])
        api = TelegramApi("tok", http)
        long_text = "A" * 5000
        api.send_message(123, long_text)
        self.assertEqual(len(http.call_log), 2)

    def test_send_message_network_error_raises(self):
        http = FakeHttpClient(responses=[{"ok": False, "description": "Bad Request"}])
        api = TelegramApi("tok", http)
        with self.assertRaises(RuntimeError):
            api.send_message(123, "hello")

    def test_offset_tracking(self):
        http = FakeHttpClient(responses=[{"ok": True, "result": []}])
        api = TelegramApi("tok", http)
        api.get_updates(offset=42, timeout=5)
        payload = http.call_log[0]["payload"]
        self.assertEqual(payload["offset"], 42)


# ---------------------------------------------------------------------------
# Auth gate tests
# ---------------------------------------------------------------------------

class AuthGateTests(unittest.TestCase):
    def _make_bot(self, allowed_ids: frozenset[int]) -> AutoResearchTelegramBot:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = make_temp_repo(tmpdir)
            config = _make_bot_config(repo, allowed_ids)
            http = FakeHttpClient()
            orch = MagicMock()
            bot = AutoResearchTelegramBot(config, http_client=http, orchestrator=orch)
            return bot

    def test_authorized_chat_passes(self):
        bot = self._make_bot(frozenset({111}))
        self.assertTrue(bot.is_authorized(111))

    def test_unauthorized_chat_rejected(self):
        bot = self._make_bot(frozenset({111}))
        self.assertFalse(bot.is_authorized(999))

    def test_empty_allowlist_rejects_all(self):
        bot = self._make_bot(frozenset())
        self.assertFalse(bot.is_authorized(111))


# ---------------------------------------------------------------------------
# Command dispatch tests
# ---------------------------------------------------------------------------

class CommandDispatchTests(unittest.TestCase):
    def _make_bot(self) -> tuple[AutoResearchTelegramBot, MagicMock, FakeHttpClient]:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = make_temp_repo(tmpdir)
        config = _make_bot_config(repo, frozenset({111}))
        http = FakeHttpClient(responses=[{"ok": True, "result": {}}] * 20)
        orch = MagicMock()
        bot = AutoResearchTelegramBot(config, http_client=http, orchestrator=orch)
        return bot, orch, http

    def test_start_authorized(self):
        bot, orch, http = self._make_bot()
        bot.handle_command(111, "/start", [])
        self.assertTrue(len(http.call_log) > 0)
        payload = http.call_log[0]["payload"]
        self.assertIn("Auto-Research", payload["text"])

    def test_start_unauthorized_shows_chat_id(self):
        bot, orch, http = self._make_bot()
        bot.handle_command(999, "/start", [])
        payload = http.call_log[0]["payload"]
        self.assertIn("999", payload["text"])

    def test_help_returns_commands(self):
        bot, orch, http = self._make_bot()
        bot.handle_command(111, "/help", [])
        payload = http.call_log[0]["payload"]
        self.assertIn("/start", payload["text"])
        self.assertIn("/help", payload["text"])

    def test_mode_research_only(self):
        bot, orch, http = self._make_bot()
        orch.set_mode.return_value = "research_only"
        bot.handle_command(111, "/mode", ["research_only"])
        orch.set_mode.assert_called_once_with("telegram", "research_only")

    def test_reset_clears_history(self):
        bot, orch, http = self._make_bot()
        bot.handle_command(111, "/reset", [])
        orch.reset_conversation.assert_called_once_with("telegram:111")


# ---------------------------------------------------------------------------
# Message handler tests
# ---------------------------------------------------------------------------

class MessageHandlerTests(unittest.TestCase):
    def _make_bot(self) -> tuple[AutoResearchTelegramBot, MagicMock, FakeHttpClient]:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = make_temp_repo(tmpdir)
        config = _make_bot_config(repo, frozenset({111}))
        http = FakeHttpClient(responses=[{"ok": True, "result": {}}] * 20)
        orch = MagicMock()
        bot = AutoResearchTelegramBot(config, http_client=http, orchestrator=orch)
        return bot, orch, http

    def test_text_triggers_chat(self):
        bot, orch, http = self._make_bot()
        orch.chat.return_value = {"reply": "Hello back"}
        update = {"message": {"chat": {"id": 111}, "text": "Hello"}}
        bot.handle_update(update)
        orch.chat.assert_called_once()

    def test_error_does_not_expose_exception(self):
        bot, orch, http = self._make_bot()
        orch.chat.side_effect = RuntimeError("SECRET_INTERNAL_ERROR")
        update = {"message": {"chat": {"id": 111}, "text": "test"}}
        bot.handle_update(update)
        payload = http.call_log[0]["payload"]
        # Must not leak internal error details to user
        self.assertNotIn("SECRET_INTERNAL_ERROR", payload["text"])
        self.assertIn("錯誤", payload["text"])

    def test_empty_message_ignored(self):
        bot, orch, http = self._make_bot()
        update = {"message": {"chat": {"id": 111}, "text": "   "}}
        bot.handle_update(update)
        orch.chat.assert_not_called()

    def test_non_text_message_gets_notice(self):
        bot, orch, http = self._make_bot()
        update = {"message": {"chat": {"id": 111}, "photo": []}}
        bot.handle_update(update)
        payload = http.call_log[0]["payload"]
        self.assertIn("text", payload)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class UtilityTests(unittest.TestCase):
    def test_split_message_chunks_short(self):
        chunks = split_message_chunks("hello")
        self.assertEqual(chunks, ["hello"])

    def test_split_message_chunks_empty(self):
        chunks = split_message_chunks("")
        self.assertEqual(chunks, [""])

    def test_split_message_chunks_long(self):
        text = "word " * 2000  # ~10000 chars
        chunks = split_message_chunks(text, limit=4000)
        self.assertTrue(len(chunks) >= 2)
        rejoined = "".join(chunks)
        self.assertEqual(rejoined.strip(), text.strip())

    def test_parse_command_parts_with_bot_mention(self):
        cmd, args = parse_command_parts("/start@mybot extra")
        self.assertEqual(cmd, "/start")
        self.assertEqual(args, ["extra"])

    def test_parse_allowed_chat_ids_valid(self):
        result = parse_allowed_chat_ids("111, 222, 333")
        self.assertEqual(result, frozenset({111, 222, 333}))

    def test_parse_allowed_chat_ids_empty(self):
        result = parse_allowed_chat_ids("")
        self.assertEqual(result, frozenset())

    def test_parse_allowed_chat_ids_invalid_raises(self):
        with self.assertRaises(ConfigError):
            parse_allowed_chat_ids("111, abc")

    def test_load_bot_config_missing_token_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = make_temp_repo(tmpdir)
            # overwrite .env without token
            (repo / ".env").write_text("TELEGRAM_PROVIDER=ollama\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_bot_config(repo_root=repo, environ={})


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------

class TelegramBreakerTests(unittest.TestCase):
    def test_telegram_api_has_breaker(self):
        http = FakeHttpClient()
        api = TelegramApi("tok", http)
        assert api.breaker is not None
        assert api.breaker.status()["target"] == "telegram-api"
        assert api.breaker.status()["state"] == STATE_CLOSED

    def test_breaker_opens_after_consecutive_failures(self):
        http = FakeHttpClient(responses=[
            {"ok": False, "description": "err"},
            {"ok": False, "description": "err"},
            {"ok": False, "description": "err"},
        ])
        api = TelegramApi("tok", http)
        for _ in range(3):
            try:
                api.get_me()
            except RuntimeError:
                pass
        assert api.breaker.status()["state"] == STATE_OPEN

    def test_breaker_open_rejects_immediately(self):
        http = FakeHttpClient(responses=[
            {"ok": False, "description": "err"},
            {"ok": False, "description": "err"},
            {"ok": False, "description": "err"},
        ])
        api = TelegramApi("tok", http)
        for _ in range(3):
            try:
                api.get_me()
            except RuntimeError:
                pass
        with self.assertRaises(CircuitOpenError) as ctx:
            api.get_me()
        assert ctx.exception.target == "telegram-api"

    def test_get_me_direct_bypasses_breaker(self):
        """Diagnostic call should work even when breaker is open."""
        http = FakeHttpClient(responses=[
            {"ok": False, "description": "err"},
            {"ok": False, "description": "err"},
            {"ok": False, "description": "err"},
            # This response is for the direct call
            {"ok": True, "result": {"username": "testbot"}},
        ])
        api = TelegramApi("tok", http)
        # Trip the breaker
        for _ in range(3):
            try:
                api.get_me()
            except RuntimeError:
                pass
        assert api.breaker.status()["state"] == STATE_OPEN
        # Direct call bypasses breaker
        result = api.get_me_direct()
        assert result["username"] == "testbot"

    def test_success_resets_breaker(self):
        http = FakeHttpClient(responses=[
            {"ok": False, "description": "err"},
            {"ok": False, "description": "err"},
            {"ok": True, "result": {"username": "bot"}},
        ])
        api = TelegramApi("tok", http)
        try:
            api.get_me()
        except RuntimeError:
            pass
        try:
            api.get_me()
        except RuntimeError:
            pass
        # Not yet at threshold (3), so this success resets
        api.get_me()
        assert api.breaker.status()["state"] == STATE_CLOSED
        assert api.breaker.failure_count == 0


# ---------------------------------------------------------------------------
# Validate diagnostics tests
# ---------------------------------------------------------------------------

class ValidateDiagnosticsTests(unittest.TestCase):
    def _make_bot(self, proxy_url: str = "", token: str = "test-token") -> tuple[AutoResearchTelegramBot, MagicMock, FakeHttpClient]:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = make_temp_repo(tmpdir)
        config = BotConfig(
            repo_root=repo,
            token=token,
            allowed_chat_ids=frozenset({111}),
            provider="ollama",
            model="test-model",
            proxy_url=proxy_url,
        )
        http = FakeHttpClient(responses=[{"ok": True, "result": {"username": "testbot"}}] * 10)
        orch = MagicMock()
        orch.doctor.return_value = {
            "services": {
                "ollama": {"ok": True, "detail": "running"},
                "lmstudio": {"ok": False, "detail": "not running"},
            },
        }
        bot = AutoResearchTelegramBot(config, http_client=http, orchestrator=orch)
        return bot, orch, http

    def test_validate_success_returns_zero(self, capsys=None):
        bot, orch, http = self._make_bot()
        result = bot.validate()
        assert result == 0

    def test_validate_telegram_fail_returns_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = make_temp_repo(tmpdir)
        config = BotConfig(
            repo_root=repo,
            token="test-token",
            allowed_chat_ids=frozenset({111}),
            provider="ollama",
            model="test-model",
        )
        # getMe fails
        http = FakeHttpClient(responses=[{"ok": False, "description": "Network error"}])
        orch = MagicMock()
        orch.doctor.return_value = {
            "services": {
                "ollama": {"ok": True, "detail": "running"},
                "lmstudio": {"ok": False, "detail": "not running"},
            },
        }
        bot = AutoResearchTelegramBot(config, http_client=http, orchestrator=orch)
        result = bot.validate()
        assert result == 1

    def test_validate_shows_proxy_hint_when_not_set(self, capsys=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = make_temp_repo(tmpdir)
        config = BotConfig(
            repo_root=repo,
            token="test-token",
            allowed_chat_ids=frozenset(),
            provider="ollama",
            model="test-model",
            proxy_url="",
        )
        # Make getMe fail to trigger hint
        def fail_request(*args, **kwargs):
            raise RuntimeError("Connection refused")
        http = FakeHttpClient()
        http.request_json = fail_request
        orch = MagicMock()
        orch.doctor.return_value = {
            "services": {
                "ollama": {"ok": True, "detail": "running"},
                "lmstudio": {"ok": False, "detail": "not running"},
            },
        }
        bot = AutoResearchTelegramBot(config, http_client=http, orchestrator=orch)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bot.validate()
        output = buf.getvalue()
        assert "PROXY_URL is not set" in output

    def test_validate_shows_proxy_configured_hint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = make_temp_repo(tmpdir)
        config = BotConfig(
            repo_root=repo,
            token="test-token",
            allowed_chat_ids=frozenset(),
            provider="ollama",
            model="test-model",
            proxy_url="socks5h://127.0.0.1:1080",
        )
        def fail_request(*args, **kwargs):
            raise RuntimeError("Proxy connection refused")
        http = FakeHttpClient()
        http.request_json = fail_request
        orch = MagicMock()
        orch.doctor.return_value = {
            "services": {
                "ollama": {"ok": True, "detail": "running"},
                "lmstudio": {"ok": False, "detail": "not running"},
            },
        }
        bot = AutoResearchTelegramBot(config, http_client=http, orchestrator=orch)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bot.validate()
        output = buf.getvalue()
        assert "Proxy is configured" in output


if __name__ == "__main__":
    unittest.main()
