"""Shared utilities for the Skill-as-Memory layer."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9_\-]+")

SECRET_PATTERNS = [
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{10,}\b", re.IGNORECASE),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{10,}\b", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{12,}", re.IGNORECASE),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
]


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_SPLIT_RE.split(text) if len(token) >= 2]


def dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def strip_frontmatter(content: str) -> str:
    if not content.startswith("---"):
        return content
    parts = content.split("---", 2)
    if len(parts) >= 3:
        return parts[2].strip()
    return content


def contains_secrets(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def extract_skill_frontmatter_field(content: str, field: str) -> str:
    if not content.startswith("---"):
        return ""
    parts = content.split("---", 2)
    if len(parts) < 3:
        return ""
    for line in parts[1].splitlines():
        if line.lower().startswith(f"{field.lower()}:"):
            return line.split(":", 1)[1].strip()
    return ""


def first_heading_or_line(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped and not stripped.startswith("---"):
            return stripped
    return ""


def optional_str(value: object) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)
