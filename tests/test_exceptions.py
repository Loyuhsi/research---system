"""Tests for the shared exceptions module."""
from auto_research.exceptions import ExecutionError, PolicyError


def test_execution_error_is_runtime_error():
    assert issubclass(ExecutionError, RuntimeError)


def test_policy_error_is_runtime_error():
    assert issubclass(PolicyError, RuntimeError)


def test_backward_compat_import_from_orchestrator():
    """Ensure exceptions re-exported from orchestrator for backward compat."""
    from auto_research.orchestrator import ExecutionError as OE, PolicyError as PE
    assert OE is ExecutionError
    assert PE is PolicyError
