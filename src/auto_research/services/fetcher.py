"""Fetch pipeline — public/private source fetching and legacy session import."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

_SAFE_SESSION_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_SAFE_SCRIPT_RE = re.compile(r"^scripts/[a-z0-9_-]+\.sh$")

from ..exceptions import ExecutionError
from ..runtime import ArtifactLayout, AutoResearchConfig
from .vault import VaultService

logger = logging.getLogger(__name__)


class FetcherService:
    """Handles data fetching from public and private sources."""

    def __init__(self, config: AutoResearchConfig, vault_service: VaultService) -> None:
        self.config = config
        self.vault_service = vault_service

    def fetch_public(self, topic: str, urls: Sequence[str]) -> Dict[str, object]:
        session_id = self._run_wsl_worker("scripts/fetch-public.sh", ["--topic", topic, *urls])
        layout = self.import_legacy_session(session_id)
        return {
            "session_id": session_id,
            "layout": self.vault_service.describe_layout(layout),
        }

    def fetch_private(self, topic: str, urls: Sequence[str], token_env: Optional[str] = None) -> Dict[str, object]:
        args = ["--topic", topic]
        if token_env:
            args.extend(["--token-env", token_env])
        args.extend(urls)
        session_id = self._run_wsl_worker("scripts/fetch-private-gh.sh", args)
        layout = self.import_legacy_session(session_id)
        return {
            "session_id": session_id,
            "layout": self.vault_service.describe_layout(layout),
        }

    def import_legacy_session(self, session_id: str) -> ArtifactLayout:

        layout = self.config.resolve_layout(session_id)
        layout.ensure()
        if not layout.legacy_sources_dir.exists():
            raise ExecutionError(f"Legacy source directory not found: {layout.legacy_sources_dir}")

        statuses: List[Mapping[str, object]] = []
        for item in sorted(layout.legacy_sources_dir.iterdir()):
            if item.suffix == ".md":
                shutil.copy2(item, layout.parsed_dir / item.name)
            elif item.name.endswith(".raw.json"):
                shutil.copy2(item, layout.raw_dir / item.name)
            elif item.name.endswith(".status.json"):
                statuses.append(json.loads(item.read_text(encoding="utf-8")))

        self.vault_service.update_status(layout, imported_from=layout.legacy_sources_dir, sources=statuses)
        return layout

    def build_worker_env(self, extra_allowlist=None) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(self.config.safe_env(extra_allowlist))
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _run_wsl_worker(
        self,
        script_path: str,
        args: Sequence[str],
        env: Optional[Mapping[str, str]] = None,
        return_session: bool = True,
    ) -> str:
        if not _SAFE_SCRIPT_RE.match(script_path):
            raise ExecutionError(
                f"Script path rejected by allowlist: {script_path!r}"
            )
        wsl_repo_root = self._to_wsl_path(self.config.repo_root)
        quoted_args = " ".join(shlex.quote(arg) for arg in args)
        command = f"./{script_path} {quoted_args}".strip()
        logger.info("Running WSL worker: %s", script_path)
        result = subprocess.run(
            ["wsl", "-d", getattr(self.config, "wsl_distro", "Ubuntu-24.04"),
             "--cd", wsl_repo_root, "bash", "-lc", command],
            cwd=self.config.repo_root,
            env=dict(env or self.build_worker_env()),
            capture_output=True,
            text=True,
            timeout=1200,
        )
        if result.returncode != 0:
            raise ExecutionError(result.stderr.strip() or result.stdout.strip() or f"Worker failed: {script_path}")
        output = result.stdout.strip()
        if not return_session:
            return output
        lines = output.splitlines()
        session_id = lines[-1].strip() if lines else ""
        if not session_id:
            raise ExecutionError(f"Worker did not return a session id: {script_path}")
        if not _SAFE_SESSION_RE.match(session_id):
            raise ExecutionError(
                f"Worker returned an unsafe session id (possible path traversal): {session_id!r}"
            )
        return session_id

    @staticmethod
    def _to_wsl_path(path: Path) -> str:
        resolved = str(path.resolve())
        drive = resolved[0].lower()
        suffix = resolved[3:].replace("\\", "/")
        return f"/mnt/{drive}/{suffix}"
