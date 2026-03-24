"""Tests for structured_output — JSON extraction, parsing, and schema validation."""
from __future__ import annotations

import pytest

from auto_research.structured_output import (
    StructuredOutputError,
    StructuredPayload,
    extract_json_candidate,
    parse_structured_payload,
    validate_json_schema,
)


class TestExtractJsonCandidate:
    def test_empty(self):
        assert extract_json_candidate("") is None
        assert extract_json_candidate("   ") is None

    def test_pure_json_object(self):
        assert extract_json_candidate('{"key": "val"}') == '{"key": "val"}'

    def test_pure_json_array(self):
        assert extract_json_candidate('[1, 2, 3]') == '[1, 2, 3]'

    def test_embedded_json(self):
        text = 'Here is the result: {"name": "test"} done.'
        result = extract_json_candidate(text)
        assert result is not None
        assert '"name"' in result

    def test_no_json(self):
        assert extract_json_candidate("just plain text") is None


class TestValidateJsonSchema:
    def test_object_valid(self):
        validate_json_schema({"a": 1}, {"type": "object", "properties": {"a": {"type": "integer"}}})

    def test_object_not_dict(self):
        with pytest.raises(StructuredOutputError, match="expected object"):
            validate_json_schema("string", {"type": "object"})

    def test_required_missing(self):
        with pytest.raises(StructuredOutputError, match="required property missing"):
            validate_json_schema({}, {"type": "object", "required": ["name"]})

    def test_additional_properties_forbidden(self):
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        with pytest.raises(StructuredOutputError, match="additional property"):
            validate_json_schema({"extra": 1}, schema)

    def test_array_valid(self):
        validate_json_schema([1, 2], {"type": "array", "items": {"type": "integer"}})

    def test_array_not_list(self):
        with pytest.raises(StructuredOutputError, match="expected array"):
            validate_json_schema("nope", {"type": "array"})

    def test_string_valid(self):
        validate_json_schema("hello", {"type": "string"})

    def test_string_enum(self):
        with pytest.raises(StructuredOutputError, match="not in enum"):
            validate_json_schema("bad", {"type": "string", "enum": ["good", "ok"]})

    def test_string_min_length(self):
        with pytest.raises(StructuredOutputError, match="minLength"):
            validate_json_schema("ab", {"type": "string", "minLength": 5})

    def test_number_valid(self):
        validate_json_schema(3.14, {"type": "number"})

    def test_number_rejects_bool(self):
        with pytest.raises(StructuredOutputError, match="expected number"):
            validate_json_schema(True, {"type": "number"})

    def test_integer_valid(self):
        validate_json_schema(42, {"type": "integer"})

    def test_integer_rejects_float(self):
        with pytest.raises(StructuredOutputError, match="expected integer"):
            validate_json_schema(3.5, {"type": "integer"})

    def test_boolean_valid(self):
        validate_json_schema(True, {"type": "boolean"})

    def test_null_valid(self):
        validate_json_schema(None, {"type": "null"})

    def test_null_rejects_non_none(self):
        with pytest.raises(StructuredOutputError, match="expected null"):
            validate_json_schema(0, {"type": "null"})

    def test_unsupported_type(self):
        with pytest.raises(StructuredOutputError, match="unsupported"):
            validate_json_schema("x", {"type": "custom"})

    def test_nested_validation(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
            },
        }
        validate_json_schema({"items": ["a", "b"]}, schema)
        with pytest.raises(StructuredOutputError):
            validate_json_schema({"items": [1, 2]}, schema)


class TestParseStructuredPayload:
    def test_first_valid_source(self):
        result = parse_structured_payload(
            {"src1": '{"name": "test"}'},
            {"type": "object", "properties": {"name": {"type": "string"}}},
        )
        assert isinstance(result, StructuredPayload)
        assert result.data == {"name": "test"}
        assert result.source == "src1"

    def test_skips_invalid_json(self):
        result = parse_structured_payload(
            {"bad": "not json{", "good": '{"ok": true}'},
            {"type": "object"},
        )
        assert result.source == "good"

    def test_no_candidates(self):
        with pytest.raises(StructuredOutputError, match="no structured payload"):
            parse_structured_payload({"a": "plain text"}, {"type": "object"})

    def test_all_fail_validation(self):
        with pytest.raises(StructuredOutputError):
            parse_structured_payload(
                {"src": '{"wrong": 1}'},
                {"type": "object", "required": ["name"]},
            )
