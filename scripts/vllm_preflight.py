#!/usr/bin/env python3
"""vLLM environment preflight checker.

Checks Python/torch/CUDA/platform compatibility for vLLM.
Produces structured JSON output with exact blockers and remediation options.

Usage:
    python scripts/vllm_preflight.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from typing import Any, Dict, List


def preflight() -> Dict[str, Any]:
    """Run all preflight checks. Returns {ok, checks, blockers, remediation}."""
    checks: Dict[str, Any] = {}
    blockers: List[str] = []

    # Python version
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks["python_version"] = ver
    if sys.version_info >= (3, 13):
        blockers.append(f"Python {ver}: vLLM typically targets 3.9-3.12")

    # Platform
    checks["platform"] = platform.system()
    checks["machine"] = platform.machine()
    if platform.system() == "Windows":
        blockers.append("Windows: vLLM is Linux-first, Windows support is experimental")

    # Torch
    try:
        import torch
        checks["torch_version"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            checks["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            blockers.append("CUDA not available")
        # Check for _opaque_base (vLLM 0.17+ requirement)
        try:
            from torch._opaque_base import OpaqueBase  # noqa: F401
            checks["torch_opaque_base"] = "available"
        except ImportError:
            checks["torch_opaque_base"] = "missing"
            blockers.append(
                f"torch._opaque_base missing in torch {torch.__version__}. "
                "vLLM 0.17+ requires this module. Use stable torch >= 2.5 or vLLM <= 0.6.x"
            )
    except ImportError:
        checks["torch_version"] = "not installed"
        blockers.append("torch not installed")

    # vLLM import
    try:
        import vllm  # noqa: F401
        checks["vllm_import"] = "ok"
        checks["vllm_version"] = getattr(vllm, "__version__", "unknown")
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
        if result.returncode == 0:
            free_mb = int(result.stdout.strip().split("\n")[0])
            checks["vram_free_mb"] = free_mb
            if free_mb < 1500:
                blockers.append(f"Only {free_mb}MB VRAM free (need ~1500MB for smallest model)")
        else:
            checks["vram_free_mb"] = "nvidia-smi failed"
    except Exception:
        checks["vram_free_mb"] = "unknown"

    # Remediation
    remediation = []
    if any("opaque_base" in b for b in blockers):
        remediation.append("Install stable torch: pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
        remediation.append("Or use older vLLM: pip install vllm==0.6.6.post1")
    if any("Windows" in b for b in blockers):
        remediation.append("Consider WSL2 or Docker for vLLM serving")
    if any("CUDA not available" in b for b in blockers):
        remediation.append("Ensure NVIDIA drivers and CUDA toolkit are installed")

    return {
        "ok": len(blockers) == 0,
        "checks": checks,
        "blockers": blockers,
        "remediation": remediation,
    }


def main() -> int:
    result = preflight()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
