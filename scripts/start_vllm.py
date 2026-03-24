#!/usr/bin/env python3
"""vLLM server startup helper with preflight checks.

Usage:
    python scripts/start_vllm.py [--port 8100] [--model Qwen/Qwen2.5-0.5B]

Prerequisites:
    - vLLM installed: pip install vllm
    - CUDA-capable GPU with free VRAM
    - Compatible torch version (vLLM requires specific torch builds)

Status (v3.13):
    BLOCKED_BY_ENV — vLLM 0.17.1 requires torch._opaque_base which is absent
    in torch 2.11.0.dev (nightly). A stable torch release or different vLLM
    version may resolve this. The Auto-Research provider integration code is
    fully ready; only the runtime startup is blocked.
"""

from __future__ import annotations

import subprocess
import sys


def preflight() -> dict:
    """Run environment preflight checks. Returns {ok, checks, blockers}."""
    checks = {}
    blockers = []

    # Python version
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks["python"] = ver
    if sys.version_info >= (3, 13):
        blockers.append(f"Python {ver}: vLLM may not fully support 3.13 yet")

    # Platform
    import platform
    checks["platform"] = platform.system()
    if platform.system() == "Windows":
        blockers.append("Windows: vLLM is Linux-first, Windows support experimental")

    # Torch
    try:
        import torch
        checks["torch"] = torch.__version__
        checks["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            checks["gpu"] = torch.cuda.get_device_name(0)
        else:
            blockers.append("CUDA not available")
    except ImportError:
        blockers.append("torch not installed")

    # vLLM import
    try:
        import vllm  # noqa: F401
        checks["vllm_import"] = "ok"
    except ImportError as e:
        checks["vllm_import"] = f"fail: {e}"
        blockers.append(f"vLLM import failed: {e}")
    except Exception as e:
        checks["vllm_import"] = f"error: {e}"
        blockers.append(f"vLLM import error: {e}")

    # Free VRAM
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        free_mb = int(result.stdout.strip().split("\n")[0])
        checks["vram_free_mb"] = free_mb
        if free_mb < 1500:
            blockers.append(f"Only {free_mb}MB VRAM free (need ~1500MB for smallest model)")
    except Exception:
        checks["vram_free_mb"] = "unknown"

    return {
        "ok": len(blockers) == 0,
        "checks": checks,
        "blockers": blockers,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Start vLLM OpenAI-compatible server")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("vLLM Server Startup — Preflight")
    print("=" * 50)

    result = preflight()
    for k, v in result["checks"].items():
        print(f"  {k}: {v}")

    if result["blockers"]:
        print(f"\nBlockers ({len(result['blockers'])}):")
        for b in result["blockers"]:
            print(f"  - {b}")
        print(f"\nStatus: BLOCKED_BY_ENV")
        return 1

    if args.preflight_only:
        print("\nPreflight: PASS")
        return 0

    print(f"\nStarting vLLM server on port {args.port} with model {args.model}...")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--max-model-len", "512",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.4",
    ]
    print(f"  Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
