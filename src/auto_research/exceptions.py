"""Shared exception types for the auto-research platform."""


class PolicyError(RuntimeError):
    """Raised when an action is blocked by policy."""
    pass


class ExecutionError(RuntimeError):
    """Raised when a runtime operation fails."""
    pass
