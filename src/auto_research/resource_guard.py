"""Cooperative execution guard for project-local GPU-heavy operations.

This is NOT a system-wide GPU scheduler. It only prevents concurrent GPU-heavy
operations within this Auto-Research project. External processes on the same
machine are not controlled.

The lock is file-based (PID + timestamp) so it works across processes within
the project. Stale locks (dead PID or expired timeout) are auto-cleared.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOCK_DIR = "output"
DEFAULT_TIMEOUT_S = 300


class GuardTimeoutError(RuntimeError):
    """Raised when the GPU execution guard cannot be acquired.

    This is NOT a provider failure and must NOT pollute circuit breaker state.
    """

    def __init__(self, lock_path: Path, message: str = ""):
        self.lock_path = lock_path
        msg = message or f"GPU execution guard busy (lock: {lock_path}). Another GPU-heavy operation may be in progress."
        super().__init__(msg)


def _pid_alive(pid: int) -> bool:
    """Check if a PID is still running. Cross-platform best-effort."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


class GpuExecutionGuard:
    """File-based cooperative mutex for GPU-heavy operations.

    Args:
        lock_path: Explicit lock file path. If None, derived from lock_dir + scope.
        lock_dir: Directory to place lock file in (default: repo_root/output).
        scope: Lock scope name, used in filename (default: "default").
        timeout: Seconds before a lock is considered stale.
    """

    def __init__(
        self,
        lock_path: Optional[Path] = None,
        lock_dir: Optional[Path] = None,
        scope: str = "default",
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        if lock_path:
            self._lock_path = lock_path
        elif lock_dir:
            self._lock_path = lock_dir / f"gpu-{scope}.lock"
        else:
            self._lock_path = Path(DEFAULT_LOCK_DIR) / f"gpu-{scope}.lock"
        self._timeout = timeout
        self._held = False

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    def _read_lock(self) -> Optional[Dict[str, Any]]:
        if not self._lock_path.exists():
            return None
        try:
            data = json.loads(self._lock_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            return None
        except (json.JSONDecodeError, OSError):
            return None

    def _is_stale(self, lock_data: Dict[str, Any]) -> bool:
        pid = lock_data.get("pid", -1)
        ts = lock_data.get("acquired_at", 0)
        if not _pid_alive(pid):
            return True
        if (time.time() - ts) > self._timeout:
            return True
        return False

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True if acquired, False if busy."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read_lock()
        if existing is not None:
            if self._is_stale(existing):
                logger.info("Clearing stale GPU lock (pid=%s)", existing.get("pid"))
                self._lock_path.unlink(missing_ok=True)
            else:
                return False

        lock_data = {
            "pid": os.getpid(),
            "acquired_at": time.time(),
        }
        self._lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
        self._held = True
        return True

    def release(self) -> None:
        """Release the lock if held by this process."""
        if self._lock_path.exists():
            existing = self._read_lock()
            if existing and existing.get("pid") == os.getpid():
                self._lock_path.unlink(missing_ok=True)
        self._held = False

    def status(self) -> Dict[str, Any]:
        """Return current lock status for diagnostics."""
        existing = self._read_lock()
        if existing is None:
            return {"state": "unlocked"}
        if self._is_stale(existing):
            return {"state": "stale", **existing}
        return {"state": "locked", **existing}

    def __enter__(self) -> "GpuExecutionGuard":
        if not self.acquire():
            raise GuardTimeoutError(self._lock_path)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


# ---------------------------------------------------------------------------
# VRAM monitoring utilities
# ---------------------------------------------------------------------------

def _query_nvidia_smi_vram() -> Optional[Dict[str, int]]:
    """Query nvidia-smi for GPU VRAM.  Returns {total_mb, used_mb, free_mb} or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,nounits,noheader"],
            timeout=10, text=True,
        )
        line = out.strip().splitlines()[0]
        total, used, free = (int(v.strip()) for v in line.split(","))
        return {"total_mb": total, "used_mb": used, "free_mb": free}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def check_vram_available(min_free_mb: int = 2048) -> Dict[str, Any]:
    """Instant VRAM snapshot.  Returns {ok, total_mb, used_mb, free_mb}."""
    result = _query_nvidia_smi_vram()
    if result is None:
        return {"ok": False, "error": "nvidia-smi not available"}
    result["ok"] = result["free_mb"] >= min_free_mb
    return result


class VramMonitor:
    """Background thread that polls nvidia-smi, recording peak VRAM usage.

    Usage::

        monitor = VramMonitor()
        monitor.start()
        # ... run inference ...
        peak_mb = monitor.stop()
    """

    def __init__(self, poll_interval: float = 1.0) -> None:
        self._poll_interval = poll_interval
        self._peak: int = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def peak_mb(self) -> int:
        return self._peak

    def start(self) -> None:
        self._running = True
        self._peak = 0
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        """Stop monitoring and return peak VRAM usage in MB."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        return self._peak

    def _poll(self) -> None:
        while self._running:
            result = _query_nvidia_smi_vram()
            if result:
                self._peak = max(self._peak, result["used_mb"])
            time.sleep(self._poll_interval)
