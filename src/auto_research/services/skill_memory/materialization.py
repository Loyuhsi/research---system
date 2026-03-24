"""Skill materialization and export for the Skill-as-Memory layer."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from ...exceptions import PolicyError, ExecutionError
from ...runtime import AutoResearchConfig
from .util import dedupe, extract_skill_frontmatter_field, first_heading_or_line, tokenize


def skill_materialize(config: AutoResearchConfig, candidate_name: str) -> Dict[str, object]:

    candidate_dir = config.repo_root / "staging" / "skills-candidates" / candidate_name
    if not candidate_dir.exists():
        raise ExecutionError(f"Missing candidate skill: {candidate_name}")

    metadata_path = candidate_dir / "metadata.json"
    citations_path = candidate_dir / "citations.json"
    for path in (candidate_dir / "recipes", candidate_dir / "scripts", candidate_dir / "tests", candidate_dir / "examples"):
        path.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        metadata = _build_skill_metadata(candidate_dir)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if not citations_path.exists():
        citations = _build_skill_citations(candidate_dir)
        citations_path.write_text(json.dumps(citations, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "candidate_dir": str(candidate_dir),
        "metadata_path": str(metadata_path),
        "citations_path": str(citations_path),
    }


def skill_export(config: AutoResearchConfig, target: str = "github", skill_id: Optional[str] = None) -> Dict[str, object]:

    if target != "github":
        raise PolicyError(f"Unsupported export target: {target}")
    config.github_skills_dir.mkdir(parents=True, exist_ok=True)

    exported: List[str] = []
    skills = [config.repo_root / "skills" / skill_id] if skill_id else sorted(
        path for path in (config.repo_root / "skills").iterdir() if path.is_dir()
    )
    for skill_dir in skills:
        if not skill_dir.exists():
            continue
        target_dir = config.github_skills_dir / skill_dir.name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(skill_dir, target_dir)
        exported.append(skill_dir.name)

    return {"target": target, "exported_skills": exported, "path": str(config.github_skills_dir)}


def _build_skill_metadata(candidate_dir: Path) -> Dict[str, object]:
    skill_md = candidate_dir / "SKILL.md"
    content = skill_md.read_text(encoding="utf-8", errors="replace") if skill_md.exists() else ""
    title = extract_skill_frontmatter_field(content, "name") or candidate_dir.name
    summary = extract_skill_frontmatter_field(content, "description") or first_heading_or_line(content)
    tags = dedupe([candidate_dir.name, *tokenize(summary)[:4]])
    return {
        "title": title,
        "summary": summary,
        "tags": tags,
        "trigger_keywords": tags[:4],
        "risk_level": "low",
        "tool_deps": [],
        "success_rate": 0.5,
        "last_validated_at": None,
    }


def _build_skill_citations(candidate_dir: Path) -> List[Dict[str, object]]:
    citations: List[Dict[str, object]] = []
    candidate_json = candidate_dir / "candidate.json"
    if candidate_json.exists():
        citations.append({"citation_type": "artifact", "label": "candidate.json", "path": str(candidate_json)})
    skill_md = candidate_dir / "SKILL.md"
    if skill_md.exists():
        citations.append({"citation_type": "artifact", "label": "SKILL.md", "path": str(skill_md)})
    return citations
