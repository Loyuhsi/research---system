"""Tool binding execution & RD-Agent sandbox runner."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

from ..exceptions import PolicyError, ExecutionError
from ..runtime import AutoResearchConfig, ToolBinding

logger = logging.getLogger(__name__)

ALLOWED_FORMATS = frozenset({
    "pdf", "html", "png", "svg", "docx", "xlsx", "pptx", "odt", "txt",
})


class ToolRunnerService:
    """Runs allowlisted tool bindings and the RD-Agent sandbox container."""

    def __init__(self, config: AutoResearchConfig) -> None:
        self.config = config

    def run_tool_binding(
        self,
        binding_name: str,
        source: Path,
        output: Path,
        mode: str,
        frontend: str = "cli",
        fmt: str = "pdf",
        dry_run: bool = False,
    ) -> Dict[str, object]:


        binding = self.config.tool_bindings.get(binding_name)
        if binding is None:
            raise PolicyError(f"Blocked tool binding: {binding_name}")
        if mode not in binding.allowed_modes:
            raise PolicyError(f"Binding {binding_name} is not permitted in mode {mode}")

        if fmt and fmt not in ALLOWED_FORMATS:
            raise PolicyError(f"Unsupported output format: {fmt!r}")

        source = source.resolve()
        output = output.resolve()
        if not source.exists():
            raise ExecutionError(f"Source file does not exist: {source}")
        self._assert_allowed_output(binding, output)
        output.parent.mkdir(parents=True, exist_ok=True)

        command = tuple(
            part.format(
                source=str(source),
                output=str(output),
                output_dir=str(output.parent),
                format=fmt,
            )
            for part in binding.command
        )
        env = os.environ.copy()
        env.update(self.config.safe_env(self.config.tool_env_allowlist.get(binding_name, binding.pass_env)))
        if dry_run:
            return {"command": list(command), "output": str(output)}

        logger.info("Executing tool binding %s: %s", binding_name, " ".join(command))
        result = subprocess.run(
            list(command),
            cwd=self.config.repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise ExecutionError(result.stderr.strip() or f"Binding failed: {binding_name}")
        return {
            "binding": binding_name,
            "output": str(output),
            "stdout": result.stdout.strip(),
        }

    def build_rd_agent_command(self, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> list[str]:
        in_dir = (input_dir or (self.config.repo_root / "sandbox" / "rd-agent" / "in")).resolve()
        out_dir = (output_dir or (self.config.repo_root / "sandbox" / "rd-agent" / "out")).resolve()
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--mount",
            f"type=bind,source={in_dir},target=/workspace/in,readonly",
            "--mount",
            f"type=bind,source={out_dir},target=/workspace/out",
            "auto-research-rd-agent:local",
        ]

    def run_rd_agent(self, mode: str, frontend: str = "cli", dry_run: bool = False) -> Dict[str, object]:


        if mode != "high_risk_execution":
            raise PolicyError("RD-Agent is only available in high_risk_execution mode.")
        command = self.build_rd_agent_command()
        if dry_run:
            return {"command": command}
        logger.info("Launching RD-Agent container")
        result = subprocess.run(command, cwd=self.config.repo_root, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise ExecutionError(result.stderr.strip() or "RD-Agent container failed.")
        return {"stdout": result.stdout.strip()}

    def _assert_allowed_output(self, binding: ToolBinding, output: Path) -> None:

        for root in binding.allowed_output_roots:
            try:
                output.relative_to(root.resolve())
                return
            except ValueError:
                continue
        raise PolicyError(f"Output path is outside allowlisted roots: {output}")
