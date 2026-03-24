#!/usr/bin/env python3
"""llama.cpp preflight: environment check and readiness probe.

Checks: CUDA, llama-server binary, GPU info, server connectivity,
model loaded, inference probe, VRAM conflict with other providers.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_BASE = os.environ.get("LLAMACPP_BASE", "http://127.0.0.1:8080")


def check_platform() -> Dict[str, Any]:
    """Check OS compatibility."""
    ok = sys.platform in ("win32", "linux", "darwin")
    return {"name": "platform", "ok": ok, "detail": sys.platform}


def check_cuda() -> Dict[str, Any]:
    """Check CUDA availability via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            timeout=10, text=True,
        )
        lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
        return {"name": "cuda", "ok": True, "gpus": lines}
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"name": "cuda", "ok": False, "error": str(exc)}


def check_binary() -> Dict[str, Any]:
    """Check if llama-server or llama-server.exe is available."""
    binary = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if binary:
        return {"name": "binary", "ok": True, "path": binary}
    # Check common locations
    common_paths = [
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-server.exe",
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
        Path("C:/llama.cpp/build/bin/llama-server.exe"),
        Path("C:/tools/llama-server.exe"),
    ]
    for p in common_paths:
        if p.exists():
            return {"name": "binary", "ok": True, "path": str(p)}
    return {
        "name": "binary", "ok": False,
        "error": "llama-server not found on PATH or common locations",
        "remediation": "Download from https://github.com/ggml-org/llama.cpp/releases or build from source",
    }


def check_server_health(base_url: str) -> Dict[str, Any]:
    """Check if llama-server is running and healthy."""
    try:
        from auto_research.http_client import JsonHttpClient
        client = JsonHttpClient()
        # llama-server /health returns {"status": "ok"} when model is loaded
        resp = client.request_json("GET", f"{base_url}/health", timeout=5)
        status = resp.get("status", "unknown")
        return {"name": "server_health", "ok": status == "ok", "status": status}
    except Exception as exc:
        return {"name": "server_health", "ok": False, "error": str(exc)}


def check_model_loaded(base_url: str) -> Dict[str, Any]:
    """Check what model is loaded via /v1/models."""
    try:
        from auto_research.http_client import JsonHttpClient
        client = JsonHttpClient()
        resp = client.request_json("GET", f"{base_url}/v1/models", timeout=5)
        data = resp.get("data", [])
        if data and isinstance(data, list):
            model_id = data[0].get("id", "unknown")
            return {"name": "model_loaded", "ok": True, "model": model_id}
        return {"name": "model_loaded", "ok": False, "error": "no model in /v1/models response"}
    except Exception as exc:
        return {"name": "model_loaded", "ok": False, "error": str(exc)}


def check_inference_probe(base_url: str) -> Dict[str, Any]:
    """Send a minimal chat completion to verify inference works."""
    try:
        from auto_research.http_client import JsonHttpClient
        client = JsonHttpClient()
        resp = client.request_json(
            "POST", f"{base_url}/v1/chat/completions",
            payload={
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "max_tokens": 16,
                "temperature": 0.0,
            },
            timeout=30,
        )
        choices = resp.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            return {"name": "inference", "ok": bool(content), "response_preview": content[:100]}
        return {"name": "inference", "ok": False, "error": "empty choices"}
    except Exception as exc:
        return {"name": "inference", "ok": False, "error": str(exc)}


def check_vram_conflict() -> Dict[str, Any]:
    """Warn if other inference providers have models loaded in VRAM."""
    conflicts: List[str] = []
    try:
        from auto_research.http_client import JsonHttpClient
        client = JsonHttpClient()
        # Check LM Studio
        try:
            resp = client.request_json("GET", "http://127.0.0.1:1234/v1/models", timeout=3)
            if resp.get("data"):
                conflicts.append("LM Studio has model(s) loaded at :1234")
        except Exception:
            pass
        # Check Ollama
        try:
            resp = client.request_json("GET", "http://127.0.0.1:11434/api/tags", timeout=3)
            if resp.get("models"):
                conflicts.append("Ollama has model(s) at :11434")
        except Exception:
            pass
    except ImportError:
        pass

    if conflicts:
        return {
            "name": "vram_conflict", "ok": False,
            "warning": "Other providers detected with loaded models — concurrent GPU usage may cause OOM",
            "conflicts": conflicts,
        }
    return {"name": "vram_conflict", "ok": True, "detail": "No conflicting providers detected"}


def run_preflight(base_url: str = DEFAULT_BASE) -> Dict[str, Any]:
    """Run all preflight checks and return structured result."""
    checks = [
        check_platform(),
        check_cuda(),
        check_binary(),
        check_server_health(base_url),
        check_model_loaded(base_url),
        check_inference_probe(base_url),
        check_vram_conflict(),
    ]

    blockers = [c for c in checks if not c["ok"] and c["name"] not in ("vram_conflict",)]
    warnings = [c for c in checks if not c["ok"] and c["name"] == "vram_conflict"]

    return {
        "ok": len(blockers) == 0,
        "base_url": base_url,
        "checks": checks,
        "blockers": [c["name"] for c in blockers],
        "warnings": [c.get("warning", c.get("error", "")) for c in warnings],
        "remediation": [c.get("remediation", "") for c in blockers if c.get("remediation")],
    }


def main() -> int:
    base = os.environ.get("LLAMACPP_BASE", DEFAULT_BASE)
    print(f"llama.cpp Preflight Check (base={base})")
    print("=" * 50)

    result = run_preflight(base)

    for check in result["checks"]:
        status = "✅" if check["ok"] else "❌"
        name = check["name"]
        detail = check.get("detail", check.get("error", check.get("status", "")))
        print(f"  {status} {name}: {detail}")

    if result["warnings"]:
        print(f"\n⚠️  Warnings:")
        for w in result["warnings"]:
            print(f"    {w}")

    if result["ok"]:
        print(f"\n✅ All checks passed — llama-server is ready at {base}")
    else:
        print(f"\n❌ Blockers: {', '.join(result['blockers'])}")
        for r in result["remediation"]:
            if r:
                print(f"   → {r}")

    # Write JSON report
    report_path = Path(__file__).resolve().parents[1] / "output" / "llamacpp_preflight.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {report_path}")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
