from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def prepare_isolated_workspace(base_repo: Path, prefix: str) -> Path:
    """Create a minimal isolated repo workspace for rerunnable scripts."""
    workspace = Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=str(base_repo / "output")))
    for rel in (
        "config",
        "output/notes",
        "output/sources",
        "output/research",
        "knowledge/index",
        "knowledge/logs",
        "knowledge/memory-records",
        "knowledge/evaluations",
        "staging/skills-candidates",
        "staging/memory-drafts",
        "staging/tooling",
        "sandbox/rd-agent/in",
        "sandbox/rd-agent/out",
        "skills",
        ".github/skills",
    ):
        (workspace / rel).mkdir(parents=True, exist_ok=True)
    for name in ("runtime-modes.json", "zones.json", "tool-allowlist.json"):
        shutil.copy2(base_repo / "config" / name, workspace / "config" / name)
    for name in (".env", "research_program.md"):
        source = base_repo / name
        if source.exists():
            shutil.copy2(source, workspace / name)
    runtime_dir = base_repo / ".runtime"
    if runtime_dir.exists():
        for source in runtime_dir.rglob("*"):
            if source.is_dir():
                continue
            destination = workspace / source.relative_to(base_repo)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return workspace
