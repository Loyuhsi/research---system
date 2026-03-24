"""Retrieval and ranking logic for the Skill-as-Memory layer."""

from __future__ import annotations

import json
import logging
import math
import sqlite3

logger = logging.getLogger(__name__)
from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence

from ...http_client import JsonHttpClient
from ...runtime import AutoResearchConfig
from ..llm_provider import LlmProviderService
from .embedding import cosine_similarity, embed_texts
from .indexing import fts5_available_in_index, memory_index_rebuild
from .util import tokenize


def retrieve_context(
    config: AutoResearchConfig,
    http_client: JsonHttpClient,
    llm: LlmProviderService,
    task: str,
    task_type: Optional[str] = None,
    source_types: Optional[Sequence[str]] = None,
    top_memory: int = 3,
    top_skills: int = 2,
) -> Dict[str, object]:
    from ...tracing import current_trace, SpanKind
    trace = current_trace()
    span = None
    if trace:
        span = trace.start_span(
            "retrieval.retrieve_context",
            SpanKind.RETRIEVAL,
            attributes={
                "task": task[:80],
                "gen_ai.operation.name": "retrieve_context",
                "otel.span_kind": "internal",
            },
        )

    source_types = list(source_types or [])
    query_embedding = embed_texts([task], config, http_client, llm)
    query_vector = query_embedding[0] if query_embedding else None

    memory_rows, skill_rows = _load_index_rows(config, http_client, llm)
    idf_table = _build_idf_table(memory_rows, skill_rows)

    # FTS5 bonus signal (additive, optional)
    fts5_enabled = fts5_available_in_index(config)
    fts5_mem: Dict[str, float] = {}
    fts5_skl: Dict[str, float] = {}
    if fts5_enabled:
        fts5_mem = _normalize_fts5_scores(
            _fts5_search(config, "memory_fts", "id", task),
        )
        fts5_skl = _normalize_fts5_scores(
            _fts5_search(config, "skills_fts", "skill_id", task),
        )

    # Fallback chain: semantic+bm25 -> bm25-only -> metadata-only
    has_semantic = query_vector is not None
    backend = "semantic+bm25" if has_semantic else "bm25"

    memory_hits = _rank_memory_rows(
        memory_rows, task, task_type, source_types, query_vector, idf_table, top_memory,
        fts5_scores=fts5_mem,
    )
    skill_hits = _rank_skill_rows(
        skill_rows, task, task_type, query_vector, idf_table, top_skills,
        fts5_scores=fts5_skl,
    )

    # Fallback: if BM25 produced nothing, try metadata-only
    fallback_triggered = False
    if not memory_hits and not skill_hits:
        backend = "metadata-only"
        fallback_triggered = True
        memory_hits = _rank_memory_rows(
            memory_rows, task, task_type, source_types, query_vector, {}, top_memory,
        )
        skill_hits = _rank_skill_rows(
            skill_rows, task, task_type, query_vector, {}, top_skills,
        )

    fts5_used = fts5_enabled and bool(fts5_mem or fts5_skl)

    if span:
        span.attributes.update({
            "backend": backend,
            "memory_hits": len(memory_hits),
            "skill_hits": len(skill_hits),
            "fallback_triggered": fallback_triggered,
            "has_semantic": has_semantic,
            "fts5_used": fts5_used,
            "retrieval.hit_count": len(memory_hits) + len(skill_hits),
            "retrieval.backend": backend,
        })
        span.finish(status="ok")

    return {
        "query": task,
        "task_type": task_type,
        "source_types": source_types,
        "vector_backend": backend,
        "memory_hits": memory_hits,
        "skill_hits": skill_hits,
        "fallback_triggered": fallback_triggered,
        "fts5_used": fts5_used,
        "index_stats": {"memory_count": len(memory_rows), "skill_count": len(skill_rows)},
    }


def memory_search(
    config: AutoResearchConfig,
    http_client: JsonHttpClient,
    llm: LlmProviderService,
    task: str,
    task_type: Optional[str] = None,
    source_types: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    return retrieve_context(config, http_client, llm, task, task_type=task_type, source_types=source_types)


# ---------------------------------------------------------------------------
# FTS5 query helpers
# ---------------------------------------------------------------------------

def _build_fts5_query(query: str) -> str:
    """Convert natural language query to FTS5 MATCH expression.

    Tokenizes, filters to alnum, quotes each, joins with OR.
    """
    tokens = tokenize(query)
    safe = [t for t in tokens if t.replace("-", "").replace("_", "").isalnum()]
    if not safe:
        return ""
    return " OR ".join(f'"{t}"' for t in safe[:10])


def _normalize_fts5_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize FTS5 rank to [0,1]. FTS5 ranks are negative (lower=better)."""
    if not scores:
        return {}
    min_rank = min(scores.values())
    max_rank = max(scores.values())
    spread = max_rank - min_rank
    if spread == 0:
        return {k: 0.5 for k in scores}
    return {k: (max_rank - v) / spread for k, v in scores.items()}


def _fts5_search(
    config: AutoResearchConfig,
    table: str,
    id_col: str,
    query: str,
    limit: int = 20,
) -> Dict[str, float]:
    """Run FTS5 MATCH on a virtual table. Returns {entity_id: rank}."""
    if not config.memory_index_path.exists():
        return {}
    fts_query = _build_fts5_query(query)
    if not fts_query:
        return {}
    conn = sqlite3.connect(config.memory_index_path)
    try:
        rows = conn.execute(
            f"SELECT {id_col}, rank FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, limit),
        ).fetchall()
        return {str(r[0]): float(r[1]) for r in rows}
    except Exception:
        return {}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# IDF / BM25
# ---------------------------------------------------------------------------

def _build_idf_table(
    memory_rows: Sequence[Mapping[str, object]],
    skill_rows: Sequence[Mapping[str, object]],
) -> Dict[str, float]:
    """Compute IDF weights from the full index for BM25 scoring."""
    all_rows = [*memory_rows, *skill_rows]
    n = len(all_rows)
    if n == 0:
        return {}
    doc_freq: Dict[str, int] = {}
    for row in all_rows:
        tokens: set[str] = set()
        for field_name in ("title", "summary"):
            tokens.update(tokenize(str(row.get(field_name, ""))))
        for json_field in ("tags_json", "trigger_keywords_json"):
            raw = row.get(json_field)
            if raw:
                try:
                    items = json.loads(str(raw))
                    for item in items:
                        tokens.update(tokenize(str(item)))
                except (json.JSONDecodeError, TypeError):
                    logger.debug("Skipping malformed IDF field in record", exc_info=True)
        for token in tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    return {
        token: math.log((n - df + 0.5) / (df + 0.5) + 1)
        for token, df in doc_freq.items()
    }


def _bm25_score(
    query: str,
    corpus_texts: Sequence[str],
    idf_table: Dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
    avg_dl: float = 50.0,
) -> float:
    """BM25-like scoring with IDF weighting."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    corpus_tokens: List[str] = []
    for text in corpus_texts:
        corpus_tokens.extend(tokenize(str(text)))
    if not corpus_tokens:
        return 0.0
    dl = len(corpus_tokens)
    tf_map = Counter(corpus_tokens)
    score = 0.0
    for qt in set(query_tokens):
        tf = tf_map.get(qt, 0)
        if tf == 0:
            continue
        idf = idf_table.get(qt, math.log(10))
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (dl / avg_dl))
        score += idf * (numerator / denominator)
    return score


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def _load_index_rows(
    config: AutoResearchConfig,
    http_client: JsonHttpClient,
    llm: LlmProviderService,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not config.memory_index_path.exists():
        memory_index_rebuild(config, http_client, llm)
    conn = sqlite3.connect(config.memory_index_path)
    conn.row_factory = sqlite3.Row
    try:
        memory_rows = [dict(row) for row in conn.execute("SELECT * FROM memory_records").fetchall()]
        skill_rows = [dict(row) for row in conn.execute("SELECT * FROM skills").fetchall()]
        vectors = {
            "memory": _read_vector_map(conn, "memory", config),
            "skill": _read_vector_map(conn, "skill", config),
        }
    finally:
        conn.close()

    for row in memory_rows:
        row["vector"] = vectors["memory"].get(str(row["id"]))
    for row in skill_rows:
        row["vector"] = vectors["skill"].get(str(row["skill_id"]))
    return memory_rows, skill_rows


def _read_vector_map(conn: sqlite3.Connection, entity_type: str, config: AutoResearchConfig) -> Dict[str, List[float]]:
    rows = conn.execute(
        "SELECT entity_id, vector_json FROM embeddings WHERE entity_type = ? AND model = ?",
        (entity_type, config.skill_memory_embedding_model),
    ).fetchall()
    return {str(entity_id): json.loads(vector_json) for entity_id, vector_json in rows}


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _rank_memory_rows(
    rows: Sequence[Mapping[str, object]],
    query: str,
    task_type: Optional[str],
    source_types: Sequence[str],
    query_vector: Optional[List[float]],
    idf_table: Dict[str, float],
    limit: int,
    fts5_scores: Optional[Dict[str, float]] = None,
) -> List[Dict[str, object]]:
    hits: List[Dict[str, object]] = []
    for row in rows:
        tags = json.loads(str(row["tags_json"]))
        row_source_types = json.loads(str(row["source_types_json"]))
        corpus = [*tags, str(row["title"]), str(row["summary"])]
        score = _bm25_score(query, corpus, idf_table) if idf_table else _lexical_score(query, corpus)
        score += _metadata_bonus(task_type, str(row["task_type"]), source_types, row_source_types, str(row["risk_level"]))
        # Evidence count ranking signal
        evidence_count = int(str(row.get("evidence_count", 0)))
        if evidence_count > 0:
            score += min(evidence_count / 5.0, 0.3)
        # FTS5 bonus (additive, max +0.3)
        if fts5_scores:
            score += fts5_scores.get(str(row["id"]), 0.0) * 0.3
        # Semantic signal
        vector = row.get("vector")
        if query_vector and isinstance(vector, list):
            score += cosine_similarity(query_vector, vector)
        if score <= 0:
            continue
        hits.append({
            "id": row["id"],
            "title": row["title"],
            "summary": row["summary"],
            "tags": tags,
            "task_type": row["task_type"],
            "source_types": row_source_types,
            "risk_level": row["risk_level"],
            "evidence_count": evidence_count,
            "score": round(score, 4),
        })
    hits.sort(key=lambda item: float(str(item["score"])), reverse=True)
    return hits[:limit]


def _rank_skill_rows(
    rows: Sequence[Mapping[str, object]],
    query: str,
    task_type: Optional[str],
    query_vector: Optional[List[float]],
    idf_table: Dict[str, float],
    limit: int,
    fts5_scores: Optional[Dict[str, float]] = None,
) -> List[Dict[str, object]]:
    hits: List[Dict[str, object]] = []
    for row in rows:
        tags = json.loads(str(row["tags_json"]))
        keywords = json.loads(str(row["trigger_keywords_json"]))
        corpus = [*tags, *keywords, str(row["title"]), str(row["summary"])]
        score = _bm25_score(query, corpus, idf_table) if idf_table else _lexical_score(query, corpus)
        if task_type and task_type in {kw.lower() for kw in keywords}:
            score += 0.4
        # FTS5 bonus
        if fts5_scores:
            score += fts5_scores.get(str(row["skill_id"]), 0.0) * 0.3
        vector = row.get("vector")
        if query_vector and isinstance(vector, list):
            score += cosine_similarity(query_vector, vector)
        if score <= 0:
            continue
        hits.append({
            "skill_id": row["skill_id"],
            "title": row["title"],
            "summary": row["summary"],
            "tags": tags,
            "trigger_keywords": keywords,
            "risk_level": row["risk_level"],
            "score": round(score, 4),
            "path": row["path"],
        })
    hits.sort(key=lambda item: float(str(item["score"])), reverse=True)
    return hits[:limit]


def _metadata_bonus(
    task_type: Optional[str],
    row_task_type: str,
    source_types: Sequence[str],
    row_source_types: Sequence[str],
    risk_level: str,
) -> float:
    """Bonus score from metadata matching (task type, source types, risk)."""
    score = 0.0
    if task_type and task_type == row_task_type:
        score += 0.5
    if source_types and any(item in row_source_types for item in source_types):
        score += 0.4
    if risk_level == "high":
        score -= 0.05
    return score


def _lexical_score(query: str, corpus: Sequence[str]) -> float:
    """Simple token overlap score — fallback when IDF table is empty."""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    corpus_tokens: set[str] = set()
    for item in corpus:
        corpus_tokens.update(tokenize(str(item)))
    if not corpus_tokens:
        return 0.0
    overlap = query_tokens & corpus_tokens
    return len(overlap) / max(len(query_tokens), 1)
