"""Narrow-scope circuit breaker for fail-fast behavior.

Each CircuitBreaker instance is scoped to a single target (e.g., "ollama",
"telegram"). Breaker state is NOT shared across targets — one failing endpoint
does not block others.

States:
    CLOSED  → normal operation, calls pass through
    OPEN    → calls fail immediately (fail-fast)
    HALF_OPEN → one probe call allowed; success → CLOSED, failure → OPEN
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, target: str, failures: int, retry_after: float):
        self.target = target
        self.failures = failures
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker OPEN for '{target}' "
            f"({failures} consecutive failures, retry after {retry_after:.1f}s)"
        )


class CircuitBreaker:
    """Per-target circuit breaker.

    Args:
        target: Identifier for the protected endpoint (e.g., "ollama", "telegram").
        failure_threshold: Consecutive failures before opening.
        recovery_timeout: Seconds in OPEN state before transitioning to HALF_OPEN.
        half_open_max: Max concurrent probe calls in HALF_OPEN state.
    """

    def __init__(
        self,
        target: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        half_open_max: int = 1,
    ):
        self.target = target
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self._state = STATE_CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        self._maybe_transition()
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def _maybe_transition(self) -> None:
        if self._state == STATE_OPEN:
            if (time.monotonic() - self._last_failure_time) >= self.recovery_timeout:
                self._state = STATE_HALF_OPEN
                self._half_open_calls = 0
                logger.info("Circuit breaker '%s': OPEN → HALF_OPEN", self.target)

    def record_success(self) -> None:
        if self._state == STATE_HALF_OPEN:
            logger.info("Circuit breaker '%s': HALF_OPEN → CLOSED (probe succeeded)", self.target)
        self._state = STATE_CLOSED
        self._failure_count = 0
        self._half_open_calls = 0

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == STATE_HALF_OPEN:
            self._state = STATE_OPEN
            logger.warning("Circuit breaker '%s': HALF_OPEN → OPEN (probe failed)", self.target)
        elif self._failure_count >= self.failure_threshold:
            self._state = STATE_OPEN
            logger.warning(
                "Circuit breaker '%s': CLOSED → OPEN (%d failures)",
                self.target, self._failure_count,
            )

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute func through the breaker. Raises CircuitOpenError if open."""
        self._maybe_transition()

        if self._state == STATE_OPEN:
            retry_after = self.recovery_timeout - (time.monotonic() - self._last_failure_time)
            raise CircuitOpenError(self.target, self._failure_count, max(retry_after, 0))

        if self._state == STATE_HALF_OPEN:
            if self._half_open_calls >= self.half_open_max:
                raise CircuitOpenError(self.target, self._failure_count, self.recovery_timeout)
            self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def status(self) -> dict:
        """Return current breaker state for diagnostics."""
        self._maybe_transition()
        return {
            "target": self.target,
            "state": self._state,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED."""
        self._state = STATE_CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
