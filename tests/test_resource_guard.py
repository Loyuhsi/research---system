"""Tests for resource_guard.py — cooperative GPU execution guard."""

from __future__ import annotations

import json
import os

import pytest

from auto_research.resource_guard import GpuExecutionGuard


class TestGpuExecutionGuard:
    def test_acquire_and_release(self, tmp_path):
        guard = GpuExecutionGuard(lock_dir=tmp_path, scope="test")
        assert guard.acquire()
        assert guard.lock_path.exists()
        guard.release()
        assert not guard.lock_path.exists()

    def test_double_acquire_fails(self, tmp_path):
        guard = GpuExecutionGuard(lock_dir=tmp_path, scope="test")
        assert guard.acquire()
        # Second acquire should fail (same process, but lock exists and PID alive)
        guard2 = GpuExecutionGuard(lock_dir=tmp_path, scope="test")
        assert not guard2.acquire()
        guard.release()

    def test_stale_lock_cleared(self, tmp_path):
        lock_path = tmp_path / "gpu-test.lock"
        # Write a lock with a dead PID
        lock_data = {"pid": 999999999, "acquired_at": 0}
        lock_path.write_text(json.dumps(lock_data), encoding="utf-8")

        guard = GpuExecutionGuard(lock_path=lock_path)
        assert guard.acquire()
        guard.release()

    def test_context_manager_success(self, tmp_path):
        guard = GpuExecutionGuard(lock_dir=tmp_path, scope="ctx")
        with guard:
            assert guard.lock_path.exists()
        assert not guard.lock_path.exists()

    def test_context_manager_busy_raises(self, tmp_path):
        guard1 = GpuExecutionGuard(lock_dir=tmp_path, scope="busy")
        guard1.acquire()
        try:
            guard2 = GpuExecutionGuard(lock_dir=tmp_path, scope="busy")
            with pytest.raises(RuntimeError, match="busy"):
                with guard2:
                    pass
        finally:
            guard1.release()

    def test_status_unlocked(self, tmp_path):
        guard = GpuExecutionGuard(lock_dir=tmp_path, scope="st")
        assert guard.status()["state"] == "unlocked"

    def test_status_locked(self, tmp_path):
        guard = GpuExecutionGuard(lock_dir=tmp_path, scope="st")
        guard.acquire()
        status = guard.status()
        assert status["state"] == "locked"
        assert status["pid"] == os.getpid()
        guard.release()

    def test_status_stale(self, tmp_path):
        lock_path = tmp_path / "gpu-st.lock"
        lock_data = {"pid": 999999999, "acquired_at": 0}
        lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
        guard = GpuExecutionGuard(lock_path=lock_path)
        assert guard.status()["state"] == "stale"

    def test_custom_scope_in_filename(self, tmp_path):
        guard = GpuExecutionGuard(lock_dir=tmp_path, scope="ollama-qwen")
        assert "gpu-ollama-qwen.lock" in str(guard.lock_path)
