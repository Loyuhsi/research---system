#!/usr/bin/env python3
"""Fully automated Telegram live validation using a real user session.

Required env vars:
- TELEGRAM_USER_API_ID
- TELEGRAM_USER_API_HASH
- TELEGRAM_USER_SESSION
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from auto_research.http_client import JsonHttpClient
from telegram_bot import load_bot_config
from live_telegram_test import TEST_CASES, classify, read_new_log_lines

LOG_PATH = REPO_ROOT / "output" / "telegram_session_log.jsonl"
RESULT_PATH = REPO_ROOT / "output" / "telegram_live_automation.json"


def _load_telethon():
    try:
        from telethon import TelegramClient
        from telethon.sessions import StringSession
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Missing Telethon. Install requirements-telegram-automation.txt before running this script."
        ) from exc
    return TelegramClient, StringSession


def _bot_username() -> str:
    config = load_bot_config(repo_root=REPO_ROOT, environ=os.environ)
    http = JsonHttpClient()
    result = http.request_json("POST", f"https://api.telegram.org/bot{config.token}/getMe", payload={}, timeout=30)
    if not result.get("ok"):
        raise RuntimeError(f"Telegram getMe failed: {result.get('description', 'unknown error')}")
    username = str(result.get("result", {}).get("username", "")).strip()
    if not username:
        raise RuntimeError("Bot username not found in getMe response")
    return username


def start_bot() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts" / "telegram_bot.py")],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(5)
    if proc.poll() is not None:
        output = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(f"Bot exited immediately:\n{output}")
    return proc


def stop_bot(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            if sys.platform == "win32":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10)
        except (ValueError, ProcessLookupError, OSError):
            proc.kill()
            proc.wait()
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _parse_timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _wait_for_log_quiet(timeout_s: float = 30.0, quiet_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_size = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
    quiet_since = time.monotonic()
    while time.monotonic() < deadline:
        current_size = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
        if current_size != last_size:
            last_size = current_size
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= quiet_s:
            return
        time.sleep(1.0)


def wait_for_log_entry(
    offset: int,
    *,
    expected_input: str,
    sent_after: float,
    timeout_s: float = 120,
) -> tuple[Dict[str, Any] | None, int]:
    deadline = time.monotonic() + timeout_s
    current_offset = offset
    while time.monotonic() < deadline:
        entries, current_offset = read_new_log_lines(LOG_PATH, current_offset)
        for entry in entries:
            if str(entry.get("input_text", "")).strip() != expected_input:
                continue
            if _parse_timestamp(str(entry.get("timestamp", ""))) < sent_after:
                continue
            return entry, current_offset
        time.sleep(1.5)
    return None, current_offset


def run() -> int:
    api_id = os.environ.get("TELEGRAM_USER_API_ID", "").strip()
    api_hash = os.environ.get("TELEGRAM_USER_API_HASH", "").strip()
    session = os.environ.get("TELEGRAM_USER_SESSION", "").strip()
    if not api_id or not api_hash or not session:
        print("Missing TELEGRAM_USER_API_ID / TELEGRAM_USER_API_HASH / TELEGRAM_USER_SESSION", file=sys.stderr)
        return 2

    results: List[Dict[str, Any]] = []
    runtime_error = ""
    stop_error = ""
    TelegramClient, StringSession = _load_telethon()
    bot_proc: subprocess.Popen | None = None
    try:
        bot_proc = start_bot()
        bot_username = _bot_username()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _wait_for_log_quiet()
        if LOG_PATH.exists():
            LOG_PATH.unlink()

        offset = 0
        with TelegramClient(StringSession(session), int(api_id), api_hash) as client:
            entity = client.loop.run_until_complete(client.get_entity(bot_username))
            for case in TEST_CASES:
                print(f"[case {case['id']}/{len(TEST_CASES)}] send: {case['name']}")
                sent_at = time.time()
                client.loop.run_until_complete(client.send_message(entity, case["message"]))
                entry, offset = wait_for_log_entry(
                    offset,
                    expected_input=case["message"],
                    sent_after=sent_at,
                    timeout_s=180,
                )

                confirm_entry = None
                if case.get("needs_confirm"):
                    delay = float(case.get("confirm_delay_s", 0))
                    if delay > 0:
                        print(f"[case {case['id']}] waiting {int(delay)}s before confirmation")
                        time.sleep(delay)
                    confirm_message = case.get("confirm_message", "yes")
                    confirm_sent_at = time.time()
                    confirm_timeout_s = float(case.get("confirm_timeout_s", 300))
                    client.loop.run_until_complete(client.send_message(entity, confirm_message))
                    confirm_entry, offset = wait_for_log_entry(
                        offset,
                        expected_input=confirm_message,
                        sent_after=confirm_sent_at,
                        timeout_s=confirm_timeout_s,
                    )

                final_entry = confirm_entry if case.get("needs_confirm") else entry
                status = classify(final_entry, case)
                print(f"[case {case['id']}] status={status} policy={(final_entry or {}).get('policy', '')}")
                results.append(
                    {
                        "test_id": case["id"],
                        "name": case["name"],
                        "message": case["message"],
                        "parsed_intent": (final_entry or {}).get("parsed_intent", ""),
                        "policy": (final_entry or {}).get("policy", ""),
                        "response_text": (final_entry or {}).get("response_text", ""),
                        "status": status,
                    }
                )
    except Exception as exc:
        runtime_error = str(exc)
        print(f"[error] {runtime_error}", file=sys.stderr)
    finally:
        if bot_proc is not None:
            try:
                stop_bot(bot_proc)
            except Exception as exc:
                stop_error = str(exc)
                print(f"[stop-error] {stop_error}", file=sys.stderr)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_count": len(results),
        "live_pass": sum(1 for item in results if item["status"] == "LIVE_PASS"),
        "partial": sum(1 for item in results if item["status"] == "PARTIAL"),
        "fail": sum(1 for item in results if item["status"] not in {"LIVE_PASS", "PARTIAL"}),
        "runtime_error": runtime_error,
        "bot_stop_error": stop_error,
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {RESULT_PATH}")
    if runtime_error or stop_error:
        return 1
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
