"""Tests for VRAM monitoring utilities in resource_guard.py."""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest

from auto_research.resource_guard import (
    check_vram_available,
    VramMonitor,
    _query_nvidia_smi_vram,
)


class TestQueryNvidiaSmi:
    def test_parse_valid_output(self):
        with patch("auto_research.resource_guard.subprocess.check_output") as mock:
            mock.return_value = "16384, 4096, 12288\n"
            result = _query_nvidia_smi_vram()
        assert result == {"total_mb": 16384, "used_mb": 4096, "free_mb": 12288}

    def test_nvidia_smi_not_found(self):
        with patch("auto_research.resource_guard.subprocess.check_output") as mock:
            mock.side_effect = FileNotFoundError("nvidia-smi not found")
            result = _query_nvidia_smi_vram()
        assert result is None

    def test_malformed_output(self):
        with patch("auto_research.resource_guard.subprocess.check_output") as mock:
            mock.return_value = "garbage output"
            result = _query_nvidia_smi_vram()
        assert result is None


class TestCheckVramAvailable:
    def test_enough_vram(self):
        with patch("auto_research.resource_guard._query_nvidia_smi_vram") as mock:
            mock.return_value = {"total_mb": 16384, "used_mb": 1000, "free_mb": 15384}
            result = check_vram_available(min_free_mb=2048)
        assert result["ok"] is True
        assert result["free_mb"] == 15384

    def test_insufficient_vram(self):
        with patch("auto_research.resource_guard._query_nvidia_smi_vram") as mock:
            mock.return_value = {"total_mb": 16384, "used_mb": 15000, "free_mb": 1384}
            result = check_vram_available(min_free_mb=2048)
        assert result["ok"] is False

    def test_nvidia_unavailable(self):
        with patch("auto_research.resource_guard._query_nvidia_smi_vram") as mock:
            mock.return_value = None
            result = check_vram_available()
        assert result["ok"] is False
        assert "error" in result


class TestVramMonitor:
    def test_peak_tracking(self):
        """VramMonitor should track the peak VRAM usage across polls."""
        call_count = 0
        values = [
            {"total_mb": 16384, "used_mb": 4000, "free_mb": 12384},
            {"total_mb": 16384, "used_mb": 8000, "free_mb": 8384},
            {"total_mb": 16384, "used_mb": 6000, "free_mb": 10384},
        ]

        def mock_query():
            nonlocal call_count
            if call_count < len(values):
                result = values[call_count]
                call_count += 1
                return result
            return values[-1]

        monitor = VramMonitor(poll_interval=0.05)
        with patch("auto_research.resource_guard._query_nvidia_smi_vram", side_effect=mock_query):
            monitor.start()
            time.sleep(0.3)  # Let it poll a few times
            peak = monitor.stop()

        assert peak == 8000  # Should capture the peak value

    def test_start_stop(self):
        """VramMonitor starts and stops cleanly."""
        monitor = VramMonitor(poll_interval=0.1)
        with patch("auto_research.resource_guard._query_nvidia_smi_vram", return_value=None):
            monitor.start()
            time.sleep(0.1)
            peak = monitor.stop()
        assert peak == 0  # No valid readings

    def test_peak_mb_property(self):
        monitor = VramMonitor()
        assert monitor.peak_mb == 0
