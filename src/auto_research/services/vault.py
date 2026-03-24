"""Vault & artifact management — promotion, status tracking, layout description."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from ..exceptions import PolicyError, ExecutionError
from ..runtime import ArtifactLayout, AutoResearchConfig


class VaultService:
    """Manages artifact lifecycle: status updates, layout description, vault promotion."""

    def __init__(self, config: AutoResearchConfig) -> None:
        self.config = config

    def promote_note(self, session_id: str, approved: bool = False) -> Dict[str, object]:

        if not approved:
            raise PolicyError("Promotion requires explicit approval.")
        layout = self.config.resolve_layout(session_id)
        if not layout.note_path.exists():
            raise ExecutionError(f"Missing note file: {layout.note_path}")
        vault_dest = self.resolve_vault_dest()
        if vault_dest is None:
            raise ExecutionError("VAULT_ROOT is not configured.")
        vault_dest.mkdir(parents=True, exist_ok=True)
        target = vault_dest / layout.note_path.name
        shutil.copy2(layout.note_path, target)
        self.update_status(layout, promoted_to=target)
        return {
            "session_id": session_id,
            "vault_path": str(target),
        }

    def resolve_vault_dest(self) -> Optional[Path]:
        if self.config.vault_root is None:
            return None
        return (self.config.vault_root / self.config.vault_subdir).resolve()

    def describe_layout(self, layout: ArtifactLayout) -> Dict[str, str]:
        return {
            "research_root": str(layout.research_root),
            "raw_dir": str(layout.raw_dir),
            "parsed_dir": str(layout.parsed_dir),
            "status_path": str(layout.status_path),
            "note_path": str(layout.note_path),
            "legacy_sources_dir": str(layout.legacy_sources_dir),
        }

    def update_status(
        self,
        layout: ArtifactLayout,
        imported_from: Optional[Path] = None,
        sources: Optional[Iterable[Mapping[str, object]]] = None,
        note_path: Optional[Path] = None,
        promoted_to: Optional[Path] = None,
        quality: Optional[Mapping[str, object]] = None,
    ) -> None:
        layout.status_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {}
        if layout.status_path.exists():
            payload = json.loads(layout.status_path.read_text(encoding="utf-8"))
        payload.setdefault("session_id", layout.session_id)
        payload["research_root"] = str(layout.research_root)
        payload["raw_dir"] = str(layout.raw_dir)
        payload["parsed_dir"] = str(layout.parsed_dir)
        if imported_from is not None:
            payload["legacy_sources_dir"] = str(imported_from)
        if sources is not None:
            payload["sources"] = list(sources)
        if note_path is not None:
            payload["note_path"] = str(note_path)
        if promoted_to is not None:
            payload["promoted_to"] = str(promoted_to)
        if quality is not None:
            payload["quality"] = dict(quality)
        layout.status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
