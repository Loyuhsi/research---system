"""Tests for registry.py — ServiceRegistry."""

from __future__ import annotations

import pytest

from auto_research.registry import RegistryError, ServiceRegistry


class TestServiceRegistry:
    def test_register_and_resolve(self):
        r = ServiceRegistry()
        r.register("svc", "instance")
        assert r.resolve("svc") == "instance"

    def test_duplicate_register_raises(self):
        r = ServiceRegistry()
        r.register("svc", "a")
        with pytest.raises(RegistryError, match="already registered"):
            r.register("svc", "b")

    def test_resolve_missing_raises(self):
        r = ServiceRegistry()
        with pytest.raises(RegistryError, match="not registered"):
            r.resolve("missing")

    def test_has_true(self):
        r = ServiceRegistry()
        r.register("svc", "a")
        assert r.has("svc")

    def test_has_false(self):
        r = ServiceRegistry()
        assert not r.has("missing")

    def test_list_services(self):
        r = ServiceRegistry()
        r.register("b", 2)
        r.register("a", 1)
        assert r.list_services() == ["a", "b"]

    def test_validate_passes(self):
        r = ServiceRegistry()
        r.register("a", 1)
        r.register("b", 2)
        r.validate(["a", "b"])  # should not raise

    def test_validate_missing_raises(self):
        r = ServiceRegistry()
        r.register("a", 1)
        with pytest.raises(RegistryError, match="Missing"):
            r.validate(["a", "b"])
