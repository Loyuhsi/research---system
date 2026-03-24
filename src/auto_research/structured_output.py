from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional


class StructuredOutputError(ValueError):
    """Raised when a structured payload cannot be parsed or validated."""


@dataclass(frozen=True)
class StructuredPayload:
    """Validated structured payload with provenance metadata."""

    data: Any
    source: str
    raw_text: str


_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def extract_json_candidate(text: str) -> Optional[str]:
    """Return the first JSON object/array candidate from free-form model output."""
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    match = _JSON_RE.search(stripped)
    if not match:
        return None
    return match.group(0).strip()


def parse_structured_payload(
    sources: Mapping[str, str],
    schema: Mapping[str, Any],
) -> StructuredPayload:
    """Validate the first candidate source that matches the provided schema."""
    errors: list[str] = []
    for source_name, raw_text in sources.items():
        candidate = extract_json_candidate(raw_text)
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"{source_name}: invalid json ({exc})")
            continue
        try:
            validate_json_schema(data, schema)
        except StructuredOutputError as exc:
            errors.append(f"{source_name}: {exc}")
            continue
        return StructuredPayload(data=data, source=source_name, raw_text=candidate)
    if errors:
        raise StructuredOutputError("; ".join(errors))
    raise StructuredOutputError("no structured payload candidates found")


def validate_json_schema(data: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Small JSON-schema subset validator for local structured outputs."""
    schema_type = str(schema.get("type", "object"))
    if schema_type == "object":
        if not isinstance(data, dict):
            raise StructuredOutputError(f"{path}: expected object")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            properties = {}
        required = schema.get("required", [])
        if not isinstance(required, list):
            required = []
        for key in required:
            if key not in data:
                raise StructuredOutputError(f"{path}.{key}: required property missing")
        additional_allowed = bool(schema.get("additionalProperties", True))
        for key, value in data.items():
            if key not in properties:
                if not additional_allowed:
                    raise StructuredOutputError(f"{path}.{key}: additional property not allowed")
                continue
            prop_schema = properties.get(key, {})
            if isinstance(prop_schema, Mapping):
                validate_json_schema(value, prop_schema, f"{path}.{key}")
        return

    if schema_type == "array":
        if not isinstance(data, list):
            raise StructuredOutputError(f"{path}: expected array")
        item_schema = schema.get("items", {})
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(data):
                validate_json_schema(item, item_schema, f"{path}[{index}]")
        return

    if schema_type == "string":
        if not isinstance(data, str):
            raise StructuredOutputError(f"{path}: expected string")
        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and data not in enum_values:
            raise StructuredOutputError(f"{path}: value {data!r} not in enum")
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(data) < min_length:
            raise StructuredOutputError(f"{path}: expected minLength >= {min_length}")
        return

    if schema_type == "number":
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            raise StructuredOutputError(f"{path}: expected number")
        return

    if schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            raise StructuredOutputError(f"{path}: expected integer")
        return

    if schema_type == "boolean":
        if not isinstance(data, bool):
            raise StructuredOutputError(f"{path}: expected boolean")
        return

    if schema_type == "null":
        if data is not None:
            raise StructuredOutputError(f"{path}: expected null")
        return

    raise StructuredOutputError(f"{path}: unsupported schema type {schema_type!r}")
