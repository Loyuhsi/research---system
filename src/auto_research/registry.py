"""Service Registry for Auto-Research context.

Provides explicit lightweight DI container without magic dependency graphs.
Used for bootstrapping the application and runtime service resolution.
"""

from typing import Any, Dict, List


class RegistryError(RuntimeError):
    pass


class ServiceRegistry:
    def __init__(self):
        self._services: Dict[str, Any] = {}

    def register(self, name: str, instance: Any) -> None:
        """Register a service explicitly. Overwriting is not allowed to prevent silent bugs."""
        if name in self._services:
            raise RegistryError(f"Service '{name}' is already registered.")
        self._services[name] = instance

    def resolve(self, name: str) -> Any:
        """Get a service by name. Raises RegistryError if not found."""
        if name not in self._services:
            raise RegistryError(f"Service '{name}' is not registered.")
        return self._services[name]

    def has(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._services

    def list_services(self) -> List[str]:
        """List all currently registered service keys."""
        return sorted(list(self._services.keys()))

    def validate(self, required_services: List[str]) -> None:
        """Ensure all required services are registered, useful for bootstrap validation."""
        missing = [s for s in required_services if not self.has(s)]
        if missing:
            raise RegistryError(f"Missing required services during validation: {', '.join(missing)}")
