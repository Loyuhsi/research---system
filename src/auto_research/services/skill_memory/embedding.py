"""Embedding and similarity computation for the Skill-as-Memory layer."""

from __future__ import annotations

import math
from typing import List, Mapping, Optional, Sequence

from ...http_client import JsonHttpClient
from ...runtime import AutoResearchConfig
from ..llm_provider import LlmProviderService
from .models import MemoryRecord


def embed_texts(
    texts: Sequence[str],
    config: AutoResearchConfig,
    http_client: JsonHttpClient,
    llm: LlmProviderService,
) -> List[Optional[List[float]]]:
    if not texts:
        return []
    provider = config.skill_memory_embedding_provider
    if provider != "lmstudio":
        return [None for _ in texts]
    try:
        response = llm.call_with_breaker(
            provider,
            http_client.request_json,
            "POST",
            f"{config.lmstudio_base}/v1/embeddings",
            payload={
                "model": config.skill_memory_embedding_model,
                "input": list(texts),
            },
            headers=llm.model_headers("lmstudio"),
            timeout=120,
        )
        data = response.get("data")
        if not isinstance(data, list):
            return [None for _ in texts]
        vectors: List[Optional[List[float]]] = []
        for item in data:
            if not isinstance(item, Mapping):
                vectors.append(None)
                continue
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                vectors.append(None)
                continue
            vectors.append([float(value) for value in embedding])
        while len(vectors) < len(texts):
            vectors.append(None)
        return vectors
    except Exception:  # CircuitOpenError, GuardTimeoutError, network errors → graceful degradation
        return [None for _ in texts]


def record_embedding_text(record: MemoryRecord) -> str:
    return " ".join(
        [
            record.title,
            record.summary,
            " ".join(record.tags),
            record.task_type,
            " ".join(record.source_types),
            " ".join(record.tool_deps),
        ]
    )


def skill_embedding_text(skill: Mapping[str, object]) -> str:
    tags = skill.get("tags", [])
    keywords = skill.get("trigger_keywords", [])
    deps = skill.get("tool_deps", [])
    return " ".join(
        [
            str(skill.get("title", "")),
            str(skill.get("summary", "")),
            " ".join(str(item) for item in (tags if isinstance(tags, list) else [])),
            " ".join(str(item) for item in (keywords if isinstance(keywords, list) else [])),
            " ".join(str(item) for item in (deps if isinstance(deps, list) else [])),
        ]
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_mag = math.sqrt(sum(a * a for a in left))
    right_mag = math.sqrt(sum(b * b for b in right))
    if left_mag == 0 or right_mag == 0:
        return 0.0
    return dot / (left_mag * right_mag)
