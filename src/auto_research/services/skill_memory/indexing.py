"""Index rebuild and SQLite management for the Skill-as-Memory layer."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from ...http_client import JsonHttpClient
from ...runtime import AutoResearchConfig
from ..llm_provider import LlmProviderService
from .embedding import embed_texts, record_embedding_text, skill_embedding_text
from .models import MemoryRecord
from .util import dedupe, extract_skill_frontmatter_field, first_heading_or_line, optional_str, tokenize


import logging

_logger = logging.getLogger(__name__)


def memory_index_rebuild(
    config: AutoResearchConfig,
    http_client: JsonHttpClient,
    llm: LlmProviderService,
) -> Dict[str, object]:
    config.memory_index_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.memory_index_path)
    try:
        backend = _prepare_database(conn)
        fts5 = _fts5_available(conn)
        if fts5:
            _prepare_fts5(conn)
        memories = _load_approved_memories(config)
        skills = _load_approved_skills(config)
        _replace_index(conn, memories, skills, config, http_client, llm, fts5_enabled=fts5)
        return {
            "index_path": str(config.memory_index_path),
            "vector_backend": backend,
            "fts5_available": fts5,
            "memory_records": len(memories),
            "skills": len(skills),
        }
    finally:
        conn.close()


def _prepare_database(conn: sqlite3.Connection) -> str:
    backend = "metadata+lexical"
    try:
        import sqlite_vec  # type: ignore

        if hasattr(sqlite_vec, "load"):
            sqlite_vec.load(conn)
            backend = "sqlite-vec"
        elif hasattr(sqlite_vec, "loadable_path"):
            conn.enable_load_extension(True)
            conn.load_extension(sqlite_vec.loadable_path())
            conn.enable_load_extension(False)
            backend = "sqlite-vec"
    except Exception:
        backend = "metadata+lexical"

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_records (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            task_type TEXT NOT NULL,
            source_types_json TEXT NOT NULL,
            tool_deps_json TEXT NOT NULL,
            citations_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            success_count INTEGER NOT NULL,
            failure_count INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            last_validated_at TEXT,
            expires_at TEXT NOT NULL,
            related_skills_json TEXT NOT NULL,
            obsidian_links_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS skills (
            skill_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            trigger_keywords_json TEXT NOT NULL,
            tool_deps_json TEXT NOT NULL,
            citations_json TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            success_rate REAL NOT NULL,
            last_validated_at TEXT,
            path TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS citations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            label TEXT NOT NULL,
            citation_type TEXT NOT NULL,
            path TEXT,
            uri TEXT,
            exists_flag INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            model TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(entity_type, entity_id, model)
        );
        CREATE TABLE IF NOT EXISTS validation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            valid INTEGER NOT NULL,
            detail_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    # Idempotent schema migration: add evidence columns (v3.9+)
    for stmt in (
        "ALTER TABLE memory_records ADD COLUMN evidence_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE memory_records ADD COLUMN provenance_json TEXT NOT NULL DEFAULT '[]'",
    ):
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # Column already exists
    return backend


def fts5_available_in_index(config: AutoResearchConfig) -> bool:
    """Check if FTS5 tables exist in the current index DB."""
    if not config.memory_index_path.exists():
        return False
    conn = sqlite3.connect(config.memory_index_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        return "memory_fts" in tables and "skills_fts" in tables
    except Exception:
        return False
    finally:
        conn.close()


def _fts5_available(conn: sqlite3.Connection) -> bool:
    """Check if FTS5 extension is available in this SQLite build."""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _fts5_probe")
        return True
    except sqlite3.OperationalError:
        _logger.debug("FTS5 not available in this SQLite build")
        return False


def _prepare_fts5(conn: sqlite3.Connection) -> None:
    """Create FTS5 virtual tables (only called when FTS5 is available)."""
    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            id, title, summary, tags_text
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
            skill_id, title, summary, tags_text
        );
    """)


def _replace_index(
    conn: sqlite3.Connection,
    memories: Sequence[MemoryRecord],
    skills: Sequence[Dict[str, object]],
    config: AutoResearchConfig,
    http_client: JsonHttpClient,
    llm: LlmProviderService,
    fts5_enabled: bool = False,
) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn.execute("DELETE FROM citations")
    conn.execute("DELETE FROM embeddings")
    conn.execute("DELETE FROM memory_records")
    conn.execute("DELETE FROM skills")
    if fts5_enabled:
        conn.execute("DELETE FROM memory_fts")
        conn.execute("DELETE FROM skills_fts")

    memory_texts = [record_embedding_text(record) for record in memories]
    memory_vectors = embed_texts(memory_texts, config, http_client, llm)
    for index, record in enumerate(memories):
        prov = record.effective_provenance()
        prov_json = json.dumps([p.to_dict() for p in prov], ensure_ascii=False)
        tags_json = json.dumps(record.tags, ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO memory_records (
                id, title, summary, tags_json, task_type, source_types_json, tool_deps_json,
                citations_json, confidence, success_count, failure_count, risk_level,
                last_validated_at, expires_at, related_skills_json, obsidian_links_json, updated_at,
                evidence_count, provenance_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.title,
                record.summary,
                tags_json,
                record.task_type,
                json.dumps(record.source_types, ensure_ascii=False),
                json.dumps(record.tool_deps, ensure_ascii=False),
                json.dumps([citation.to_dict() for citation in record.citations], ensure_ascii=False),
                record.confidence,
                record.success_count,
                record.failure_count,
                record.risk_level,
                record.last_validated_at,
                record.expires_at,
                json.dumps(record.related_skills, ensure_ascii=False),
                json.dumps(record.obsidian_links, ensure_ascii=False),
                record.updated_at or now,
                len(prov),
                prov_json,
            ),
        )
        if fts5_enabled:
            conn.execute(
                "INSERT INTO memory_fts (id, title, summary, tags_text) VALUES (?, ?, ?, ?)",
                (record.id, record.title, record.summary, " ".join(record.tags)),
            )
        for citation in record.citations:
            conn.execute(
                """
                INSERT INTO citations (entity_type, entity_id, label, citation_type, path, uri, exists_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("memory", record.id, citation.label, citation.citation_type, citation.path, citation.uri, int(bool(citation.path and Path(citation.path).exists()))),
            )
        if index < len(memory_vectors) and memory_vectors[index]:
            conn.execute(
                "INSERT INTO embeddings (entity_type, entity_id, model, vector_json, updated_at) VALUES (?, ?, ?, ?, ?)",
                ("memory", record.id, config.skill_memory_embedding_model, json.dumps(memory_vectors[index]), now),
            )

    skill_texts = [skill_embedding_text(skill) for skill in skills]
    skill_vectors = embed_texts(skill_texts, config, http_client, llm)
    for index, skill in enumerate(skills):
        conn.execute(
            """
            INSERT INTO skills (
                skill_id, title, summary, tags_json, trigger_keywords_json, tool_deps_json,
                citations_json, risk_level, success_rate, last_validated_at, path, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill["skill_id"],
                skill["title"],
                skill["summary"],
                json.dumps(skill["tags"], ensure_ascii=False),
                json.dumps(skill["trigger_keywords"], ensure_ascii=False),
                json.dumps(skill["tool_deps"], ensure_ascii=False),
                json.dumps(skill["citations"], ensure_ascii=False),
                skill["risk_level"],
                float(str(skill["success_rate"])),
                skill.get("last_validated_at"),
                skill["path"],
                now,
            ),
        )
        if fts5_enabled:
            skill_tags = skill["tags"] if isinstance(skill["tags"], list) else []
            conn.execute(
                "INSERT INTO skills_fts (skill_id, title, summary, tags_text) VALUES (?, ?, ?, ?)",
                (skill["skill_id"], skill["title"], skill["summary"], " ".join(str(t) for t in skill_tags)),
            )
        for citation in (skill["citations"] if isinstance(skill["citations"], list) else []):
            conn.execute(
                """
                INSERT INTO citations (entity_type, entity_id, label, citation_type, path, uri, exists_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "skill",
                    skill["skill_id"],
                    citation.get("label", skill["skill_id"]),
                    citation.get("citation_type", "artifact"),
                    citation.get("path"),
                    citation.get("uri"),
                    int(bool(citation.get("path") and Path(str(citation["path"])).exists())),
                ),
            )
        if index < len(skill_vectors) and skill_vectors[index]:
            conn.execute(
                "INSERT INTO embeddings (entity_type, entity_id, model, vector_json, updated_at) VALUES (?, ?, ?, ?, ?)",
                ("skill", skill["skill_id"], config.skill_memory_embedding_model, json.dumps(skill_vectors[index]), now),
            )
    conn.commit()


def _load_approved_memories(config: AutoResearchConfig) -> List[MemoryRecord]:
    records: List[MemoryRecord] = []
    config.memory_records_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(config.memory_records_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(MemoryRecord.from_dict(payload))
    return records


def _load_approved_skills(config: AutoResearchConfig) -> List[Dict[str, object]]:
    skills: List[Dict[str, object]] = []
    skills_root = config.repo_root / "skills"
    if not skills_root.exists():
        return skills
    for skill_dir in sorted(path for path in skills_root.iterdir() if path.is_dir()):
        skills.append(read_skill(skill_dir))
    return skills


def read_skill(skill_dir: Path) -> Dict[str, object]:
    skill_id = skill_dir.name
    skill_md = skill_dir / "SKILL.md"
    content = skill_md.read_text(encoding="utf-8", errors="replace") if skill_md.exists() else ""
    metadata_path = skill_dir / "metadata.json"
    citations_path = skill_dir / "citations.json"
    metadata: Dict[str, object] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    citations: List[Dict[str, object]] = []
    if citations_path.exists():
        loaded = json.loads(citations_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            citations = [dict(item) for item in loaded if isinstance(item, Mapping)]
    title = metadata.get("title") or extract_skill_frontmatter_field(content, "name") or skill_id
    summary = metadata.get("summary") or extract_skill_frontmatter_field(content, "description") or first_heading_or_line(content)
    tags = metadata.get("tags") or dedupe([skill_id, *tokenize(str(summary))[:4]])
    trigger_keywords = metadata.get("trigger_keywords") or dedupe([skill_id, *tokenize(str(title))[:4]])
    tool_deps = metadata.get("tool_deps") or []
    risk_level = metadata.get("risk_level") or "low"
    success_rate = float(str(metadata.get("success_rate") or 0.8))
    last_validated_at = metadata.get("last_validated_at")
    return {
        "skill_id": skill_id,
        "title": str(title),
        "summary": str(summary),
        "tags": [str(item) for item in (tags if isinstance(tags, list) else [])],
        "trigger_keywords": [str(item) for item in (trigger_keywords if isinstance(trigger_keywords, list) else [])],
        "tool_deps": [str(item) for item in (tool_deps if isinstance(tool_deps, list) else [])],
        "citations": citations or [{"citation_type": "artifact", "label": "SKILL.md", "path": str(skill_md)}],
        "risk_level": str(risk_level),
        "success_rate": success_rate,
        "last_validated_at": optional_str(last_validated_at),
        "path": str(skill_dir),
    }
