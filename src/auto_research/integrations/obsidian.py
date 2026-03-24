"""Obsidian vault exporter for Auto-Research artifacts.

Exports research notes, memory records, evaluations, and diagnostics
as Markdown files with enriched frontmatter and wikilinks.

This is a pure adapter/exporter — it does NOT modify the orchestrator,
memory lifecycle, evaluation pipeline, or any core service.
"""

from __future__ import annotations

import datetime as dt
import re
import urllib.parse
from pathlib import Path

ARTIFACT_VERSION = "v3.18"
from typing import Any, Dict, List, Mapping, Optional

from ..runtime import AutoResearchConfig

SLUG_RE = re.compile(r"[^a-z0-9]+")

# Vault subdirectories — aligned with existing numbered style
VAULT_DIRS = {
    "research_note": "10_Research/AutoResearch",
    "memory_record": "20_Memory",
    "evaluation": "30_Evaluations",
    "diagnostics": "40_Diagnostics",
}


class ObsidianExporter:
    """Exports Auto-Research artifacts to an Obsidian vault as Markdown."""

    def __init__(self, config: AutoResearchConfig, export_mode: str = "vault") -> None:
        self.vault_root = config.vault_root
        self.vault_subdir = config.vault_subdir
        self.export_mode = export_mode  # "vault" (wikilinks) or "share" (markdown links)

    # -- Public export methods -------------------------------------------------

    def export_note(
        self,
        session_id: str,
        note_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export a research note with enriched frontmatter + wikilinks."""
        self._ensure_vault_dirs()
        content = note_path.read_text(encoding="utf-8", errors="replace")
        body = _strip_frontmatter(content)
        meta = {
            "id": session_id,
            "type": "research_note",
            "artifact_version": ARTIFACT_VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            **(metadata or {}),
        }
        slug = _slug(session_id)
        related = self._build_wikilinks("research_note", meta.get("related_ids", []))
        out = self._vault_path("research_note") / f"{slug}.md"
        out.write_text(
            self._render_frontmatter(meta) + body + "\n" + related,
            encoding="utf-8",
        )
        return out

    def export_memory(self, record_dict: Dict[str, Any]) -> Path:
        """Export a memory record as an Obsidian note."""
        self._ensure_vault_dirs()
        record_id = str(record_dict.get("id", "unknown"))
        title = str(record_dict.get("title", record_id))
        meta = {
            "id": record_id,
            "type": "memory_record",
            "artifact_version": ARTIFACT_VERSION,
            "created_at": record_dict.get("created_at", ""),
            "status": record_dict.get("status", "draft"),
            "tags": record_dict.get("tags", []),
            "provenance_count": record_dict.get("evidence_count", len(record_dict.get("evidence_sources", []))),
            "provenance_refs": record_dict.get("evidence_sources", []),
            "task_type": record_dict.get("task_type", ""),
            "confidence": record_dict.get("confidence", 0.0),
        }
        body_parts = [
            f"# {title}\n",
            f"## Summary\n\n{record_dict.get('summary', '')}\n",
        ]
        citations = record_dict.get("citations", [])
        if citations:
            body_parts.append("## Citations\n")
            for c in citations:
                if isinstance(c, dict):
                    label = c.get("label", "")
                    path = c.get("path", c.get("uri", ""))
                    body_parts.append(f"- {label}: `{path}`\n")
        evidence = record_dict.get("evidence_sources", [])
        if evidence:
            body_parts.append("\n## Evidence Sources\n")
            for e in evidence:
                body_parts.append(f"- {e}\n")
        related_ids = []
        if record_dict.get("session_id"):
            related_ids.append(("research_note", str(record_dict["session_id"]), title))
        related = self._build_wikilinks("memory_record", related_ids)
        slug = _slug(record_id)
        out = self._vault_path("memory_record") / f"{slug}.md"
        out.write_text(
            self._render_frontmatter(meta) + "\n".join(body_parts) + "\n" + related,
            encoding="utf-8",
        )
        return out

    def export_evaluation(self, eval_dict: Dict[str, Any]) -> Path:
        """Export an evaluation record as an Obsidian note."""
        self._ensure_vault_dirs()
        eval_id = str(eval_dict.get("eval_id", "unknown"))
        meta = {
            "id": eval_id,
            "type": "evaluation",
            "artifact_version": ARTIFACT_VERSION,
            "created_at": eval_dict.get("timestamp", ""),
            "status": "passed" if eval_dict.get("passed") else "failed",
            "evaluation_score": eval_dict.get("candidate_score", 0.0),
            "eval_type": eval_dict.get("eval_type", ""),
        }
        body = (
            f"# Evaluation: {eval_id}\n\n"
            f"## Results\n\n"
            f"- **Type**: {eval_dict.get('eval_type', '?')}\n"
            f"- **Baseline**: {eval_dict.get('baseline_score', '?')}\n"
            f"- **Candidate**: {eval_dict.get('candidate_score', '?')}\n"
            f"- **Passed**: {eval_dict.get('passed', '?')}\n\n"
        )
        artifacts = eval_dict.get("artifact_paths", [])
        if artifacts:
            body += "## Artifacts\n\n"
            for a in artifacts:
                body += f"- `{a}`\n"
        slug = _slug(eval_id)
        out = self._vault_path("evaluation") / f"{slug}.md"
        out.write_text(
            self._render_frontmatter(meta) + body,
            encoding="utf-8",
        )
        return out

    def export_diagnostics(self, doctor_report: Dict[str, Any]) -> Path:
        """Export system diagnostics as a daily note."""
        self._ensure_vault_dirs()
        today = dt.date.today().isoformat()
        meta = {
            "id": f"diag-{today}",
            "type": "diagnostics",
            "artifact_version": ARTIFACT_VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "provider": doctor_report.get("provider", "?"),
            "model": doctor_report.get("model", "?"),
        }
        body = f"# Diagnostics: {today}\n\n"
        # Provider status
        body += "## Provider Status\n\n"
        services = doctor_report.get("services", {})
        for name, info in services.items():
            ok = info.get("ok", False) if isinstance(info, dict) else False
            status = "OK" if ok else "FAIL"
            body += f"- **{name}**: {status}\n"
        # Provider matrix summary
        matrix = doctor_report.get("provider_matrix", {})
        if matrix:
            body += "\n## Provider Matrix\n\n"
            for prov, info in matrix.items():
                if isinstance(info, dict):
                    body += (
                        f"### {prov}\n"
                        f"- Health: {info.get('health_ready', '?')}\n"
                        f"- Inference: {info.get('inference_ready', '?')}\n"
                        f"- Embedding: {info.get('embedding_ready', '?')}\n"
                        f"- Primary: {info.get('is_primary', False)}\n\n"
                    )
        slug = f"diag-{today}"
        out = self._vault_path("diagnostics") / f"{slug}.md"
        out.write_text(
            self._render_frontmatter(meta) + body,
            encoding="utf-8",
        )
        return out

    # -- Internal helpers ------------------------------------------------------

    def _vault_path(self, artifact_type: str) -> Path:
        """Resolve vault subdirectory for a given artifact type."""
        if self.vault_root is None:
            raise RuntimeError("VAULT_ROOT is not configured.")
        subdir = VAULT_DIRS.get(artifact_type, "00_Inbox")
        path = self.vault_root / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def export_session_summary(
        self,
        session_id: str,
        topic: str,
        provider: str,
        model: str,
        quality_scores: Dict[str, Any],
        source_count: int = 0,
        word_count: int = 0,
        latency_s: float = 0.0,
        run_kind: str = "research",
    ) -> Path:
        """Export a session summary note after research completion."""
        self._ensure_vault_dirs()
        metadata = {
            "type": "session_summary",
            "artifact_version": ARTIFACT_VERSION,
            "session_id": session_id,
            "topic": topic,
            "provider": provider,
            "model": model,
            "run_kind": run_kind,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "coverage": quality_scores.get("coverage_score", 0.0),
            "structure": quality_scores.get("structure_score", 0.0),
            "provenance": quality_scores.get("provenance_score", 0.0),
            "qg_pass": quality_scores.get("passed", False),
            "word_count": word_count,
            "source_count": source_count,
        }
        frontmatter = self._render_frontmatter(metadata)
        body = (
            f"# Session Summary: {topic}\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Provider | {provider} |\n"
            f"| Model | {model} |\n"
            f"| Coverage | {quality_scores.get('coverage_score', 0.0):.3f} |\n"
            f"| Structure | {quality_scores.get('structure_score', 0.0):.3f} |\n"
            f"| QG Pass | {'Yes' if quality_scores.get('passed') else 'No'} |\n"
            f"| Word Count | {word_count} |\n"
            f"| Sources | {source_count} |\n"
            f"| Latency | {latency_s:.1f}s |\n"
        )
        wikilinks = self._build_wikilinks(
            "research_note", [("research_note", session_id, session_id)]
        )
        content = frontmatter + body + wikilinks

        slug = _slug(session_id)
        filename = f"summary-{slug}.md"

        if self.vault_root:
            vault_path = self.vault_root / VAULT_DIRS["research_note"] / filename
            vault_path.write_text(content, encoding="utf-8")
            return vault_path
        return Path(filename)

    def export_benchmark_summary(
        self,
        benchmark_results: List[Dict[str, Any]],
        decision: str = "",
        reason: str = "",
    ) -> Path:
        """Export a benchmark comparison note."""
        self._ensure_vault_dirs()
        metadata = {
            "type": "benchmark_summary",
            "artifact_version": ARTIFACT_VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "decision": decision,
        }
        frontmatter = self._render_frontmatter(metadata)

        # Build comparison table
        headers = ["Dimension"]
        for r in benchmark_results:
            headers.append(r.get("name", r.get("provider", "?")))
        header_row = " | ".join(headers)
        sep_row = " | ".join(["---"] * len(headers))

        dimensions = ["coverage", "structure", "word_count", "latency_s", "provenance",
                       "peak_vram_mb", "qg_pass"]
        rows = []
        for dim in dimensions:
            cells = [dim]
            for r in benchmark_results:
                val = r.get(dim, "")
                cells.append(str(val))
            rows.append(" | ".join(cells))

        table = f"| {header_row} |\n| {sep_row} |\n"
        for row in rows:
            table += f"| {row} |\n"

        body = (
            f"# Benchmark: llama.cpp vs LM Studio ({ARTIFACT_VERSION})\n\n"
            f"{table}\n"
            f"## Decision\n\n{decision}: {reason}\n\n"
            f"## Methodology\n\n"
            f"Same topic, sources, prompt, timeout=900s, temp=0.2. "
            f"Quantization controlled as independent variable.\n"
        )

        content = frontmatter + body
        filename = f"benchmark-summary-{ARTIFACT_VERSION}.md"

        if self.vault_root:
            vault_path = self.vault_root / VAULT_DIRS["research_note"] / filename
            vault_path.write_text(content, encoding="utf-8")
            return vault_path
        return Path(filename)

    def _ensure_vault_dirs(self) -> None:
        """Create vault subdirectory structure if needed."""
        if self.vault_root is None:
            return
        for subdir in VAULT_DIRS.values():
            (self.vault_root / subdir).mkdir(parents=True, exist_ok=True)

    def _render_frontmatter(self, metadata: Dict[str, Any]) -> str:
        """Render YAML frontmatter block with standard schema."""
        lines = ["---"]
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, list):
                yaml_list = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in value)
                lines.append(f"{key}: [{yaml_list}]")
            elif isinstance(value, str):
                escaped = value.replace('"', '\\"')
                lines.append(f'{key}: "{escaped}"')
            elif isinstance(value, bool):
                lines.append(f"{key}: {'true' if value else 'false'}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("---\n")
        return "\n".join(lines)

    def _build_wikilinks(
        self,
        source_type: str,
        related: List[Any],
    ) -> str:
        """Build a ## Related section with [[wikilinks]]."""
        if not related:
            return ""
        links: List[str] = []
        for item in related:
            if isinstance(item, tuple) and len(item) == 3:
                target_type, target_id, display_title = item
                subdir = VAULT_DIRS.get(target_type, "00_Inbox")
                slug = _slug(str(target_id))
                if self.export_mode == "share":
                    links.append(f"- [{display_title}]({subdir}/{slug}.md)")
                else:
                    links.append(f"- [[{subdir}/{slug}|{display_title}]]")
            elif isinstance(item, str):
                if self.export_mode == "share":
                    links.append(f"- [{item}]({item}.md)")
                else:
                    links.append(f"- [[{item}]]")
        if not links:
            return ""
        return "\n## Related\n\n" + "\n".join(links) + "\n"


class ObsidianUri:
    """Generate obsidian:// URIs for quick navigation."""

    @staticmethod
    def open_note(vault_name: str, file_path: str) -> str:
        """Generate obsidian://open URI."""
        return (
            f"obsidian://open?vault={urllib.parse.quote(vault_name)}"
            f"&file={urllib.parse.quote(file_path)}"
        )

    @staticmethod
    def search(vault_name: str, query: str) -> str:
        """Generate obsidian://search URI."""
        return (
            f"obsidian://search?vault={urllib.parse.quote(vault_name)}"
            f"&query={urllib.parse.quote(query)}"
        )


# -- Module-level helpers -----------------------------------------------------

def _slug(text: str) -> str:
    """Generate safe filesystem slug from text."""
    return SLUG_RE.sub("-", text.lower()).strip("-")[:80]


def _strip_frontmatter(content: str) -> str:
    """Remove existing YAML frontmatter from content."""
    if content.startswith("---"):
        lines = content.splitlines()
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                return "\n".join(lines[i + 1:])
    return content
