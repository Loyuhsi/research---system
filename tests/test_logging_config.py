"""Tests for logging_config.py — JSON structured logging."""

from __future__ import annotations

import json
import logging

import pytest

from auto_research.logging_config import JsonFormatter, setup_logging


class TestJsonFormatter:
    def test_format_produces_valid_json(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=42, msg="hello %s", args=("world",), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "hello world"
        assert data["level"] == "INFO"
        assert data["line"] == 42
        assert "ts" in data

    def test_format_handles_exception(self):
        formatter = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="test.py",
                lineno=10, msg="failed", args=(), exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "ERROR"


class TestSetupLogging:
    def test_setup_does_not_raise(self):
        # Just verify it runs without error
        setup_logging(json_mode=False)

    def test_setup_json_mode(self):
        setup_logging(json_mode=True)
        # Verify root logger has handler
        root = logging.getLogger()
        assert len(root.handlers) > 0
