#!/usr/bin/env python3
"""Live Telegram validation helpers.

Runs the bot as a subprocess, observes `output/telegram_session_log.jsonl`,
and records results for a fixed set of 9 real-world Telegram messages.
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
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "output" / "telegram_session_log.jsonl"
RESULT_PATH = REPO_ROOT / "output" / "telegram_live_test_v318.json"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "SAFE_status",
        "description": "SAFE intent via keyword parser",
        "message": "status",
        "expected_intent": "status",
        "expected_policy": "SAFE",
        "needs_confirm": False,
    },
    {
        "id": 2,
        "name": "CONFIRM_start_research",
        "description": "Research start requires explicit approval",
        "message": "start research Python Testing",
        "confirm_message": "yes",
        "expected_intent": "start_research",
        "expected_policy": "CONFIRM_EXECUTED",
        "needs_confirm": True,
    },
    {
        "id": 3,
        "name": "Layer2_LLM_parse",
        "description": "Ambiguous provider question intended for LLM parsing",
        "message": "show me which inference backends are online right now",
        "expected_intent": "list_providers",
        "expected_policy": "",
        "needs_confirm": False,
    },
    {
        "id": 4,
        "name": "chat_fallback",
        "description": "General conversational message that should fall back to chat",
        "message": "Hello, what can you do for me?",
        "expected_intent": "chat",
        "expected_policy": "SAFE",
        "needs_confirm": False,
    },
    {
        "id": 5,
        "name": "policy_rejection",
        "description": "Message that triggers DISABLED policy rejection",
        "message": "execute shell rm -rf /",
        "expected_intent": "",
        "expected_policy": "DISABLED",
        "needs_confirm": False,
    },
    {
        "id": 6,
        "name": "confirmation_expiry",
        "description": "Research confirmation that expires after the TTL",
        "message": "start research Timeout Testing",
        "confirm_message": "yes",
        "confirm_delay_s": 130,
        "expected_intent": "start_research",
        "expected_policy": "CONFIRM_PENDING",
        "needs_confirm": True,
        "expect_expiry": True,
    },
    {
        "id": 7,
        "name": "search_sources",
        "description": "Source search flow with confirmation",
        "message": "search sources machine learning",
        "confirm_message": "yes",
        "expected_intent": "search_sources",
        "expected_policy": "CONFIRM_EXECUTED",
        "needs_confirm": True,
    },
    {
        "id": 8,
        "name": "research_with_search",
        "description": "Search-augmented research flow",
        "message": "research with search local LLM inference",
        "confirm_message": "yes",
        "confirm_timeout_s": 420,
        "expected_intent": "research_with_search",
        "expected_policy": "CONFIRM_EXECUTED",
        "needs_confirm": True,
    },
    {
        "id": 9,
        "name": "provider_diagnose",
        "description": "SAFE provider diagnostic command",
        "message": "diagnose lmstudio",
        "expected_intent": "diagnose_provider",
        "expected_policy": "SAFE",
        "needs_confirm": False,
    },
]


def read_new_log_lines(path: Path, offset: int) -> tuple[List[Dict[str, Any]], int]:
    """Read JSONL lines added since *offset* bytes."""
    if not path.exists():
        return [], offset
    size = path.stat().st_size
    if size <= offset:
        return [], offset
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries, size


def wait_for_log_entry(
    path: Path,
    offset: int,
    timeout_s: float = 300,
    poll_interval: float = 2.0,
) -> tuple[Optional[Dict[str, Any]], int]:
    """Block until a new log entry appears or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        entries, new_offset = read_new_log_lines(path, offset)
        if entries:
            return entries[-1], new_offset
        time.sleep(poll_interval)
    return None, offset


def classify(entry: Optional[Dict[str, Any]], tc: Dict[str, Any]) -> str:
    if entry is None:
        return "RUNTIME_FAIL"
    if tc.get("expect_expiry"):
        if entry.get("policy") == "CONFIRM_EXPIRED":
            return "LIVE_PASS"
        return "PARTIAL"
    expected_policy = tc.get("expected_policy", "")
    expected_intent = tc.get("expected_intent", "")
    if expected_policy and entry.get("policy") != expected_policy:
        return "PARTIAL"
    if expected_intent and entry.get("parsed_intent") != expected_intent:
        return "PARTIAL"
    if expected_policy or expected_intent:
        return "LIVE_PASS"
    return "PARTIAL"


def start_bot() -> subprocess.Popen:
    """Start telegram_bot.py as a subprocess."""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    bot_script = REPO_ROOT / "scripts" / "telegram_bot.py"
    proc = subprocess.Popen(
        [sys.executable, str(bot_script)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(5)
    if proc.poll() is not None:
        output = proc.stdout.read() if proc.stdout else ""
        print(f"[error] Bot exited immediately. Output:\n{output}")
        raise RuntimeError("Bot process died on startup")
    print(f"[info] Bot subprocess started (pid={proc.pid})")
    return proc


def stop_bot(proc: subprocess.Popen) -> None:
    """Gracefully stop the bot subprocess."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT if sys.platform != "win32" else signal.CTRL_BREAK_EVENT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    print("[info] Bot subprocess stopped")


def run_tests_interactive() -> List[Dict[str, Any]]:
    """Prompt the operator and collect results from the session log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    results: List[Dict[str, Any]] = []
    offset = 0

    print("\n" + "=" * 60)
    print("Telegram Live Test (manual send + auto observe)")
    print("=" * 60)
    print(f"Bot must be running and polling. Session log: {LOG_PATH}\n")

    for tc in TEST_CASES:
        print(f"\n--- Test {tc['id']}/{len(TEST_CASES)}: {tc['name']} ---")
        print(f"  Description: {tc['description']}")
        print("  Send this message to the bot:")
        print(f"     {tc['message']}")
        input("  Press ENTER after sending the message...")

        entry, offset = wait_for_log_entry(LOG_PATH, offset, timeout_s=120)
        if entry:
            print(
                f"  [ok] Received log entry: intent={entry.get('parsed_intent')}, "
                f"policy={entry.get('policy')}, latency={entry.get('latency_ms')}ms"
            )
        else:
            print("  [timeout] No log entry received within 120s")

        confirm_entry = None
        if tc.get("needs_confirm"):
            delay = tc.get("confirm_delay_s", 0)
            if delay > 0:
                print(f"  Wait {delay}s before sending confirmation: {tc['confirm_message']}")
                time.sleep(delay)
            else:
                print(f"  Send confirmation: {tc.get('confirm_message', 'yes')}")
            input("  Press ENTER after sending confirmation...")
            confirm_entry, offset = wait_for_log_entry(LOG_PATH, offset, timeout_s=300)
            if confirm_entry:
                print(
                    f"  [ok] Confirmation result: policy={confirm_entry.get('policy')}, "
                    f"latency={confirm_entry.get('latency_ms')}ms"
                )
            else:
                print("  [timeout] No confirmation log entry received")

        final_entry = confirm_entry if tc.get("needs_confirm") else entry
        status = classify(final_entry, tc)
        results.append(
            {
                "test_id": tc["id"],
                "name": tc["name"],
                "description": tc["description"],
                "message_sent": tc["message"],
                "parse_layer": (final_entry or {}).get("parse_layer", ""),
                "parsed_intent": (final_entry or {}).get("parsed_intent", ""),
                "policy": (final_entry or {}).get("policy", ""),
                "action_proposal_id": (final_entry or {}).get("action_proposal_id", ""),
                "response_text": (final_entry or {}).get("response_text", "")[:200],
                "latency_ms": (final_entry or {}).get("latency_ms", 0),
                "verification_status": status,
            }
        )
        print(f"  Verification: {status}")

    return results


def main() -> int:
    print("[info] Starting Telegram live test session...")
    bot_proc = start_bot()
    try:
        results = run_tests_interactive()
    finally:
        stop_bot(bot_proc)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "version": "v3.18",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_count": len(results),
        "live_pass": sum(1 for row in results if row["verification_status"] == "LIVE_PASS"),
        "partial": sum(1 for row in results if row["verification_status"] == "PARTIAL"),
        "fail": sum(1 for row in results if row["verification_status"] == "RUNTIME_FAIL"),
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] Results written to {RESULT_PATH}")
    print(
        f"  LIVE_PASS: {output['live_pass']}, "
        f"PARTIAL: {output['partial']}, FAIL: {output['fail']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
