#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)

from auto_research.circuit_breaker import CircuitBreaker, CircuitOpenError
from auto_research.http_client import JsonHttpClient
from auto_research.orchestrator import Orchestrator
from auto_research.runtime import ConfigError, parse_key_value_file, read_setting


MAX_TELEGRAM_MESSAGE_LENGTH = 4000


@dataclass(frozen=True)
class BotConfig:
    repo_root: Path
    token: str
    allowed_chat_ids: frozenset[int]
    provider: str
    model: str
    default_mode: str = "research_only"
    proxy_url: str = ""


class TelegramApi:
    def __init__(self, token: str, http_client: JsonHttpClient):
        self._http_client = http_client
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._breaker = CircuitBreaker(target="telegram-api", failure_threshold=3, recovery_timeout=60.0)

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    def get_me(self) -> Mapping[str, object]:
        return self._call("getMe", {})

    def get_me_direct(self) -> Mapping[str, object]:
        """Call getMe bypassing the circuit breaker — for diagnostics only."""
        return self._raw_call("getMe", {})

    def get_updates(self, offset: Optional[int], timeout: int = 30) -> List[Mapping[str, object]]:
        payload: Dict[str, object] = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        result = self._call("getUpdates", payload, timeout=timeout + 10)
        if not isinstance(result, list):
            raise RuntimeError("Telegram getUpdates returned an unexpected payload.")
        return result

    def send_message(self, chat_id: int, text: str) -> None:
        for chunk in split_message_chunks(text):
            self._call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
            )

    def _call(self, method: str, payload: Mapping[str, object], timeout: int = 30) -> Mapping[str, object] | List[Mapping[str, object]]:
        """Execute a Telegram API call through the circuit breaker."""
        try:
            from auto_research.tracing import current_trace, SpanKind
            trace = current_trace()
            span = None
            if trace:
                span = trace.start_span(f"telegram.{method}", SpanKind.TELEGRAM_API, attributes={"method": method})
            result = self._breaker.call(self._raw_call, method, payload, timeout=timeout)
            if span:
                span.finish(status="ok")
            return result
        except ImportError:
            return self._breaker.call(self._raw_call, method, payload, timeout=timeout)
        except Exception as exc:
            if span:  # type: ignore[possibly-undefined]
                span.finish(status="error", error=str(exc))  # type: ignore[possibly-undefined]
            raise

    def _raw_call(self, method: str, payload: Mapping[str, object], timeout: int = 30) -> Mapping[str, object] | List[Mapping[str, object]]:
        """Execute a Telegram API call directly (no breaker)."""
        response = self._http_client.request_json("POST", f"{self._base_url}/{method}", payload=payload, timeout=timeout)
        if not response.get("ok"):
            description = response.get("description", "Telegram API error")
            raise RuntimeError(str(description))
        return response.get("result", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-Research Telegram bot")
    parser.add_argument("--validate-only", action="store_true", help="Validate config and connectivity, then exit")
    return parser.parse_args()


def parse_allowed_chat_ids(raw_value: str) -> frozenset[int]:
    if not raw_value.strip():
        return frozenset()

    chat_ids = set()
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            chat_ids.add(int(item))
        except ValueError as exc:
            raise ConfigError(f"Invalid TELEGRAM_ALLOWED_CHAT_IDS entry: {item}") from exc
    return frozenset(chat_ids)


def load_bot_config(repo_root: Optional[Path] = None, environ: Optional[Mapping[str, str]] = None) -> BotConfig:
    repo_root = repo_root or REPO_ROOT
    environ = environ or os.environ
    env_values = parse_key_value_file(repo_root / ".env")

    token = read_setting("TELEGRAM_BOT_TOKEN", env_values, environ)
    if not token:
        raise ConfigError("Missing TELEGRAM_BOT_TOKEN. Fill it in .env before starting the bot.")

    provider = read_setting("TELEGRAM_PROVIDER", env_values, environ, "ollama").lower()
    if provider not in {"ollama", "lmstudio", "vllm", "llamacpp"}:
        raise ConfigError("TELEGRAM_PROVIDER must be one of: ollama, lmstudio, vllm, llamacpp.")

    model = read_setting("TELEGRAM_MODEL", env_values, environ, "qwen3.5:9b")
    if not model:
        raise ConfigError("TELEGRAM_MODEL cannot be empty.")

    proxy_url = read_setting("PROXY_URL", env_values, environ)

    return BotConfig(
        repo_root=repo_root,
        token=token,
        allowed_chat_ids=parse_allowed_chat_ids(read_setting("TELEGRAM_ALLOWED_CHAT_IDS", env_values, environ)),
        provider=provider,
        model=model,
        proxy_url=proxy_url,
    )


def split_message_chunks(text: str, limit: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> List[str]:
    if not text:
        return [""]

    chunks: List[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        split_at = max(window.rfind("\n"), window.rfind(" "))
        if split_at < limit // 2:
            split_at = limit
        else:
            split_at += 1
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]

    chunks.append(remaining)
    return [chunk for chunk in chunks if chunk]


def _is_affirmative(text: str) -> bool:
    """Check if text is an affirmative confirmation response."""
    lower = text.strip().lower()
    return lower in {"yes", "y", "ok", "confirm", "確認", "好", "是", "對"}


def parse_command_parts(text: str) -> tuple[str, List[str]]:
    tokens = text.split()
    if not tokens:
        return "", []
    command = tokens[0].split("@", 1)[0].lower()
    return command, tokens[1:]


class AutoResearchTelegramBot:
    def __init__(
        self,
        config: BotConfig,
        http_client: Optional[JsonHttpClient] = None,
        orchestrator: Optional[Orchestrator] = None,
    ):
        self.config = config
        self.http_client = http_client or JsonHttpClient()
        self.telegram_api = TelegramApi(config.token, self.http_client)
        if orchestrator:
            self.orchestrator = orchestrator
        else:
            from auto_research.cli import bootstrap
            self.orchestrator = bootstrap()
        self.chat_modes: Dict[int, str] = {}
        # Control plane components
        try:
            from auto_research.telegram import IntentParser, ActionRegistry, PolicyGuard, ConversationState
            self.intent_parser = IntentParser(
                config=self.orchestrator.config,
                llm_service=self.orchestrator.llm,
                http_client=self.http_client,
            )
            self.policy_guard = PolicyGuard(
                telemetry_path=self.orchestrator.config.repo_root / "output" / "telemetry.jsonl",
            )
            self.action_registry = ActionRegistry(
                orchestrator=self.orchestrator,
                config=self.orchestrator.config,
            )
            self.conversation_state = ConversationState()
            self._control_plane_ready = True
        except Exception:
            self._control_plane_ready = False

    def _log_session_event(
        self,
        chat_id: int,
        input_text: str,
        parsed_intent: str = "",
        parse_layer: str = "",
        confidence: float = 0.0,
        policy: str = "",
        action_proposal_id: str = "",
        response_text: str = "",
        latency_ms: float = 0.0,
    ) -> None:
        """Append a structured JSONL line for live-proof observability."""
        log_path = self.config.repo_root / "output" / "telegram_session_log.jsonl"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chat_id": chat_id,
                "input_text": input_text,
                "parsed_intent": parsed_intent,
                "parse_layer": parse_layer,
                "confidence": confidence,
                "policy": policy,
                "action_proposal_id": action_proposal_id,
                "response_text": response_text[:500],
                "latency_ms": round(latency_ms, 1),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Failed to write session log entry", exc_info=True)

    def diagnostics(self) -> Dict[str, object]:
        """Pure config/state report — no API calls, no side effects."""
        return {
            "config": {
                "token_set": bool(self.config.token),
                "proxy": self.config.proxy_url or None,
                "provider": self.config.provider,
                "model": self.config.model,
                "default_mode": self.config.default_mode,
                "allowed_chat_ids": sorted(self.config.allowed_chat_ids),
            },
            "breaker": self.telegram_api.breaker.status(),
            "code_path_available": True,
        }

    def validate_live(self) -> Dict[str, object]:
        """Real API/proxy verification with timing."""
        result: Dict[str, object] = {
            "telegram": {"ok": False},
            "proxy": None,
            "llm": {"ok": False},
            "network_path_available": False,
            "live_e2e_status": "fail",
        }
        # Proxy test
        if self.config.proxy_url:
            try:
                t0 = time.monotonic()
                self.http_client.request_json("GET", "https://api.telegram.org", timeout=10)
                result["proxy"] = {"ok": True, "latency_ms": round((time.monotonic() - t0) * 1000, 1)}
            except Exception as exc:
                result["proxy"] = {"ok": False, "error": str(exc)}
        # Telegram getMe (bypass breaker)
        try:
            t0 = time.monotonic()
            me = self.telegram_api.get_me_direct()
            elapsed = round((time.monotonic() - t0) * 1000, 1)
            result["telegram"] = {"ok": True, "latency_ms": elapsed, "username": me.get("username")}
            result["network_path_available"] = True
        except Exception as exc:
            result["telegram"] = {"ok": False, "error": str(exc)}
        # LLM provider
        try:
            doctor = self.orchestrator.doctor()
            provider_info = doctor["services"][self.config.provider]
            result["llm"] = provider_info
        except Exception as exc:
            result["llm"] = {"ok": False, "error": str(exc)}
        # Overall
        tg_ok = bool(result["telegram"].get("ok")) if isinstance(result["telegram"], dict) else False  # type: ignore[union-attr]
        llm_ok = bool(result["llm"].get("ok")) if isinstance(result["llm"], dict) else False  # type: ignore[union-attr]
        if tg_ok and llm_ok:
            result["live_e2e_status"] = "ready"
        elif tg_ok:
            result["live_e2e_status"] = "partial_llm_down"
        elif llm_ok:
            result["live_e2e_status"] = "partial_telegram_down"
        return result

    def validate(self) -> int:
        print("=" * 50)
        print("Telegram Bot Diagnostics")
        print("=" * 50)

        # 1. Config checks
        print(f"\n[Config]")
        print(f"  Token: {'SET' if self.config.token else 'MISSING'}")
        print(f"  Proxy: {self.config.proxy_url or 'NOT SET'}")
        print(f"  Provider: {self.config.provider}")
        print(f"  Model: {self.config.model}")
        print(f"  Default mode: {self.config.default_mode}")
        if self.config.allowed_chat_ids:
            allowlist = ", ".join(str(cid) for cid in sorted(self.config.allowed_chat_ids))
            print(f"  Allowed chat IDs: {allowlist}")
        else:
            print("  Allowed chat IDs: <empty> (bot stays locked until /start reveals your chat_id)")

        # 2. Telegram connectivity (direct call, bypassing breaker)
        print(f"\n[Telegram Connectivity]")
        telegram_ok = False
        try:
            t0 = time.monotonic()
            me = self.telegram_api.get_me_direct()
            elapsed_ms = (time.monotonic() - t0) * 1000
            username = me.get("username", "<unknown>")
            print(f"  getMe: OK (@{username}, {elapsed_ms:.0f}ms)")
            telegram_ok = True
        except Exception as exc:
            print(f"  getMe: FAIL — {exc}")
            if not self.config.proxy_url:
                print("  Hint: PROXY_URL is not set. If api.telegram.org is blocked,")
                print("        set PROXY_URL in .env (e.g. socks5h://127.0.0.1:1080)")
            else:
                print(f"  Hint: Proxy is configured ({self.config.proxy_url}) but request still failed.")
                print("        Check proxy availability and credentials.")

        # 3. LLM provider health
        print(f"\n[LLM Provider]")
        try:
            doctor = self.orchestrator.doctor()
            for name in ("ollama", "lmstudio"):
                svc = doctor["services"][name]
                status = "OK" if svc["ok"] else "FAIL"
                print(f"  {name}: {status} ({svc['detail']})")
        except Exception as exc:
            print(f"  doctor: FAIL — {exc}")
            doctor = None

        # 4. Summary
        print(f"\n[Summary]")
        provider_ok = False
        if doctor:
            provider_ok = bool(doctor["services"][self.config.provider]["ok"])
        print(f"  Telegram API: {'OK' if telegram_ok else 'FAIL'}")
        print(f"  LLM provider ({self.config.provider}): {'OK' if provider_ok else 'FAIL'}")

        if telegram_ok and provider_ok:
            print("\n  Status: READY")
            return 0
        elif telegram_ok:
            print(f"\n  Status: PARTIAL — Telegram OK, but LLM provider '{self.config.provider}' is not reachable")
            return 1
        elif provider_ok:
            print("\n  Status: PARTIAL — LLM OK, but Telegram API is not reachable")
            return 1
        else:
            print("\n  Status: FAIL — Both Telegram and LLM provider are unreachable")
            return 1

    def run(self) -> int:
        me = self.telegram_api.get_me()
        username = me.get("username", "<unknown>")
        print(f"[info] Telegram bot is running as @{username}", flush=True)

        next_offset: Optional[int] = None
        while True:
            try:
                # Check breaker state before polling
                breaker_status = self.telegram_api.breaker.status()
                if breaker_status["state"] == "open":
                    wait = self.telegram_api.breaker.recovery_timeout
                    logger.warning(
                        "Telegram circuit breaker OPEN (%d failures). Waiting %.0fs before retry.",
                        breaker_status["failure_count"], wait,
                    )
                    time.sleep(wait)
                    continue

                updates = self.telegram_api.get_updates(offset=next_offset, timeout=30)
                for update in updates:
                    update_id = update.get("update_id")
                    if isinstance(update_id, int):
                        next_offset = update_id + 1
                    self.handle_update(update)
            except KeyboardInterrupt:
                print("\n[info] Telegram bot stopped", flush=True)
                return 0
            except CircuitOpenError as exc:
                logger.warning("Telegram circuit open: %s. Waiting %.0fs.", exc, exc.retry_after)
                time.sleep(max(exc.retry_after, 5))
            except Exception:
                logger.exception("Polling failed, retrying in 3s")
                time.sleep(3)

    def handle_update(self, update: Mapping[str, object]) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return
        chat = message.get("chat")
        if not isinstance(chat, dict):
            return
        chat_id = chat.get("id")
        if not isinstance(chat_id, int):
            return
        text = message.get("text")
        if not isinstance(text, str):
            self.telegram_api.send_message(chat_id, "目前只支援純文字訊息。")
            return

        stripped = text.strip()
        if not stripped:
            return
        if stripped.startswith("/"):
            command, args = parse_command_parts(stripped)
            self.handle_command(chat_id, command, args)
            return

        if not self.is_authorized(chat_id):
            self.telegram_api.send_message(chat_id, self.authorization_message(chat_id))
            return

        session_key = self.session_key(chat_id)

        # Conversational control plane (v3.14+)
        if self._control_plane_ready:
            self._handle_control_plane(chat_id, stripped, session_key)
        else:
            # Legacy fallback: direct chat passthrough
            self._handle_legacy_chat(chat_id, stripped, session_key)

    def _handle_control_plane(self, chat_id: int, text: str, session_key: str) -> None:
        """Route text through intent parser → policy → action registry."""
        from auto_research.telegram.policy_guard import ActionPolicy

        t0 = time.monotonic()

        # 1. Check pending confirmation
        state = self.conversation_state.get(chat_id)
        if state.pending_confirmation:
            if _is_affirmative(text):
                intent = self.conversation_state.confirm_pending(chat_id)
                if intent:
                    result = self.action_registry.execute(intent.intent, intent.args, session_key)
                    self.policy_guard.log_action(intent, result[:100])
                    response = f"已確認並執行：{intent.intent}\n\n{result}"
                    self.telegram_api.send_message(chat_id, response)
                    self._log_session_event(
                        chat_id, text, parsed_intent=intent.intent,
                        parse_layer=intent.parse_layer, confidence=intent.confidence,
                        policy="CONFIRM_EXECUTED", response_text=response,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                else:
                    response = "確認已逾時，請重新操作。"
                    self.telegram_api.send_message(chat_id, response)
                    self._log_session_event(
                        chat_id, text, policy="CONFIRM_EXPIRED",
                        response_text=response,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                return
            else:
                self.conversation_state.clear_pending(chat_id)

        # 2. Parse intent (hybrid 3-layer)
        parsed = self.intent_parser.parse(text)

        # 3. If clarification needed
        if parsed.clarification:
            response = f"請確認：{parsed.clarification}"
            self.telegram_api.send_message(chat_id, response)
            self._log_session_event(
                chat_id, text, parsed_intent=parsed.intent,
                parse_layer=parsed.parse_layer, confidence=parsed.confidence,
                policy="CLARIFICATION", response_text=response,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
            return

        # 4. Create proposal (ActionProposal is part of execution path)
        proposal = self.policy_guard.propose(chat_id, parsed)

        # 5. Policy decision based on proposal
        if proposal.policy == ActionPolicy.DISABLED:
            response = "此操作不可透過 Telegram 執行。"
            self.telegram_api.send_message(chat_id, response)
            self._log_session_event(
                chat_id, text, parsed_intent=parsed.intent,
                parse_layer=parsed.parse_layer, confidence=parsed.confidence,
                policy="DISABLED", action_proposal_id=proposal.audit_id,
                response_text=response,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
            return
        if proposal.policy == ActionPolicy.CONFIRM:
            self.conversation_state.set_pending(chat_id, parsed)
            response = (
                f"確認執行：{parsed.intent}\n"
                f"參數：{parsed.args}\n"
                "回覆 yes / ok / 確認 來執行，其他訊息會取消。"
            )
            self.telegram_api.send_message(chat_id, response)
            self._log_session_event(
                chat_id, text, parsed_intent=parsed.intent,
                parse_layer=parsed.parse_layer, confidence=parsed.confidence,
                policy="CONFIRM_PENDING", action_proposal_id=proposal.audit_id,
                response_text=response,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
            return

        # 6. Execute via proposal (SAFE actions)
        result = self.action_registry.execute_proposal(proposal, session_key)
        self.policy_guard.log_action(parsed, result[:100], proposal=proposal)
        self.telegram_api.send_message(chat_id, result)
        self._log_session_event(
            chat_id, text, parsed_intent=parsed.intent,
            parse_layer=parsed.parse_layer, confidence=parsed.confidence,
            policy="SAFE", action_proposal_id=proposal.audit_id,
            response_text=result,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    def _handle_legacy_chat(self, chat_id: int, text: str, session_key: str) -> None:
        """Legacy chat passthrough (when control plane is not available)."""
        mode = self.chat_modes.get(chat_id, self.config.default_mode)
        try:
            result = self.orchestrator.chat(
                session_key=session_key,
                text=text,
                mode=mode,
                frontend="telegram",
            )
        except Exception:
            logger.exception("Error handling chat message from %s", chat_id)
            self.telegram_api.send_message(chat_id, "處理您的訊息時發生錯誤，請稍後重試。")
            return
        self.telegram_api.send_message(chat_id, f"回覆：{result['reply']}")

    def handle_command(self, chat_id: int, command: str, args: List[str]) -> None:
        if command == "/start":
            if self.is_authorized(chat_id):
                self.telegram_api.send_message(chat_id, self.welcome_message(chat_id))
            else:
                self.telegram_api.send_message(chat_id, self.authorization_message(chat_id))
            return

        if not self.is_authorized(chat_id):
            self.telegram_api.send_message(chat_id, self.authorization_message(chat_id))
            return

        if command == "/help":
            self.telegram_api.send_message(chat_id, self.help_message(chat_id))
        elif command == "/status":
            self.telegram_api.send_message(chat_id, self.status_message(chat_id))
        elif command == "/reset":
            self.orchestrator.reset_conversation(self.session_key(chat_id))
            self.telegram_api.send_message(chat_id, "已清除這個 chat 的對話歷史。")
        elif command == "/mode":
            self.telegram_api.send_message(chat_id, self.handle_mode_command(chat_id, args))
        else:
            self.telegram_api.send_message(chat_id, self.help_message(chat_id))

    def handle_mode_command(self, chat_id: int, args: List[str]) -> str:
        if not args:
            return f"目前模式：{self.chat_modes.get(chat_id, self.config.default_mode)}"

        requested_mode = args[0].strip().lower()
        if requested_mode != "research_only":
            return "Telegram 目前只允許 `research_only`。較高風險的模式請改用 terminal 或 Pi.dev 主入口。"

        self.chat_modes[chat_id] = self.orchestrator.set_mode("telegram", requested_mode)
        return f"已切換模式為：{requested_mode}"

    def is_authorized(self, chat_id: int) -> bool:
        return bool(self.config.allowed_chat_ids) and chat_id in self.config.allowed_chat_ids

    def session_key(self, chat_id: int) -> str:
        return f"telegram:{chat_id}"

    def authorization_message(self, chat_id: int) -> str:
        return (
            "這個 bot 目前是 owner-only。\n"
            f"你的 chat_id 是 `{chat_id}`。\n"
            "請把這個值填進 `.env` 的 `TELEGRAM_ALLOWED_CHAT_IDS=`，儲存後重新啟動 bot。"
        )

    def welcome_message(self, chat_id: int) -> str:
        return (
            "Auto-Research Telegram adapter 已連線。\n"
            f"目前 chat_id：`{chat_id}`\n"
            f"目前模式：`{self.chat_modes.get(chat_id, self.config.default_mode)}`\n"
            "可直接輸入自然語言，例如「查狀態」「列出 provider」「顯示報告」「搜尋並研究 AI agent」。"
        )

    def help_message(self, chat_id: int) -> str:
        return (
            "可用指令：\n"
            "/start 顯示啟用資訊與 chat_id\n"
            "/help 顯示這份說明\n"
            "/status 檢查 provider / model / service health\n"
            "/reset 清空這個 chat 的對話歷史\n"
            "/mode 檢視或切換模式；Telegram 僅支援 research_only\n"
            "自然語言操作範例：\n"
            "- 查狀態\n"
            "- 列出 provider\n"
            "- 切換 provider lmstudio\n"
            "- 顯示報告\n"
            "- 搜尋來源 RAG evaluation\n"
            f"目前 provider/model：`{self.config.provider}` / `{self.config.model}`\n"
            f"目前模式：`{self.chat_modes.get(chat_id, self.config.default_mode)}`"
        )

    def status_message(self, chat_id: int) -> str:
        doctor = self.orchestrator.doctor()
        override_text = ""
        if hasattr(self.orchestrator, "provider_override_status"):
            override = self.orchestrator.provider_override_status(self.session_key(chat_id))
            override_text = (
                f"\n目前生效 provider/model：`{override.get('effective_provider')}` / "
                f"`{override.get('effective_model')}`"
            )
        return (
            f"授權狀態：`{'authorized' if self.is_authorized(chat_id) else 'locked'}`\n"
            f"模式：`{self.chat_modes.get(chat_id, self.config.default_mode)}`\n"
            f"provider/model：`{self.config.provider}` / `{self.config.model}`\n"
            f"{override_text}"
            f"Ollama：`{'OK' if doctor['services']['ollama']['ok'] else 'FAIL'}`\n"
            f"LM Studio：`{'OK' if doctor['services']['lmstudio']['ok'] else 'FAIL'}`\n"
            f"Docker daemon：`{'OK' if doctor['docker']['daemon'] else 'FAIL'}`"
        )


def main() -> int:
    args = parse_args()
    config = load_bot_config()
    http_client = JsonHttpClient(proxy_url=config.proxy_url) if config.proxy_url else None
    bot = AutoResearchTelegramBot(config, http_client=http_client)
    if args.validate_only:
        return bot.validate()
    return bot.run()


if __name__ == "__main__":
    raise SystemExit(main())
