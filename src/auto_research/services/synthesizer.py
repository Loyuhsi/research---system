"""Note synthesis engine — generates research notes from fetched sources via LLM."""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
import time as _time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

_logger = logging.getLogger(__name__)

from ..exceptions import ExecutionError
from ..http_client import JsonHttpClient
from ..runtime import AutoResearchConfig
from .llm_provider import LlmProviderService
from .quality_gate import QualityGateService
from .vault import VaultService

MAX_SOURCE_BYTES = 102400


class SynthesizerService:
    """Generates research Markdown notes from imported source files."""

    def __init__(
        self,
        config: AutoResearchConfig,
        http_client: JsonHttpClient,
        llm: LlmProviderService,
        vault_service: VaultService,
    ) -> None:
        self.config = config
        self.http_client = http_client
        self.llm = llm
        self.vault_service = vault_service

    def synthesize(
        self,
        topic: str,
        session_id: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        run_kind: str = "cli_manual",
        search_result_count: int = 0,
        fetched_source_count: int = 0,
    ) -> Dict[str, object]:

        layout = self.config.resolve_layout(session_id)
        source_dir = (
            layout.parsed_dir
            if layout.parsed_dir.exists() and any(layout.parsed_dir.glob("*.md"))
            else layout.legacy_sources_dir
        )
        if not source_dir.exists():
            raise ExecutionError(f"Missing source directory for session {session_id}")
        markdown_files = sorted(source_dir.glob("*.md"))
        if not markdown_files:
            raise ExecutionError(f"No markdown sources found in {source_dir}")

        # Pre-synthesis source ranking — filter low-quality, log warnings
        markdown_files, source_ranking = self._rank_and_filter_sources(markdown_files)
        if not markdown_files:
            raise ExecutionError("All sources filtered out by quality ranking.")

        selected_provider = (provider or self.config.provider).lower()
        selected_model = model or self.llm.default_model_for_provider(selected_provider)
        configured_num_ctx = int(self.config.env_values.get("NUM_CTX", "8192"))
        current_num_ctx = configured_num_ctx
        last_exc: Optional[Exception] = None
        response = None
        source_bundle = ""
        packing_metadata: Dict[str, object] = {}
        t0 = _time.monotonic()
        for _ in range(3):
            (
                source_bundle,
                packing_metadata,
                max_tokens,
                estimated_prompt_tokens,
            ) = self._prepare_bundle_for_context(markdown_files, current_num_ctx)
            messages = [
                {
                    "role": "system",
                    "content": "You produce Markdown research notes from local evidence only. Never expose hidden reasoning.",
                },
                {
                    "role": "user",
                    "content": self.llm.build_synthesis_prompt(
                        topic=topic,
                        session_id=session_id,
                        provider=selected_provider,
                        model=selected_model,
                        sources_count=len(markdown_files),
                        source_bundle=source_bundle,
                    ),
                },
            ]
            try:
                response = self.llm.call_text(
                    selected_provider,
                    self.http_client,
                    model=selected_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                    timeout=900,
                )
                packing_metadata["context_window"] = current_num_ctx
                packing_metadata["estimated_prompt_tokens"] = estimated_prompt_tokens
                break
            except Exception as exc:
                last_exc = exc
                inferred_ctx = self._extract_context_window(str(exc))
                if inferred_ctx and inferred_ctx < current_num_ctx:
                    _logger.warning(
                        "Provider reported smaller context window (%d < %d); retrying with smaller bundle",
                        inferred_ctx,
                        current_num_ctx,
                    )
                    current_num_ctx = inferred_ctx
                    continue
                raise
        if response is None:
            raise last_exc or ExecutionError("Model response did not contain message content.")
        latency_s = round(_time.monotonic() - t0, 1)
        content = str(response.get("content", "")).strip()
        if not content:
            raise ExecutionError("Model response did not contain message content.")

        layout.note_path.parent.mkdir(parents=True, exist_ok=True)
        frontmatter = "\n".join(
            [
                "---",
                f'topic: "{topic}"',
                f'created: "{dt.date.today().isoformat()}"',
                f'session_id: "{session_id}"',
                f'provider: "{selected_provider}"',
                f'model: "{selected_model}"',
                f"sources_count: {len(markdown_files)}",
                "---",
                "",
            ]
        )
        layout.note_path.write_text(f"{frontmatter}\n{content}\n", encoding="utf-8")
        self.vault_service.update_status(layout, note_path=layout.note_path)

        # Post-synthesis quality gate with provenance
        provenance = {
            "evidence_count": len(markdown_files),
            "source_types": [f.suffix for f in markdown_files],
            "retrieval_metadata": packing_metadata,
            "response_mode": response.get("response_mode", "content"),
            "retry_count": int(response.get("retry_count", 0)),
            "source_ranking": source_ranking or {},
        }
        quality_gate = QualityGateService(self.config)
        quality_report = quality_gate.check(layout.note_path, markdown_files, provenance=provenance)
        self.vault_service.update_status(
            layout, note_path=layout.note_path, quality=quality_report.to_dict()
        )

        # Save evaluation record for tracking
        eval_result = self._record_evaluation(session_id, quality_report)

        # Regression check against historical quality scores
        regression = self._check_regression(quality_report)

        # Append to results.tsv (observable failure — never blocks synthesis)
        tsv_written = self._append_results_tsv(
            session_id=session_id, provider=selected_provider, model=selected_model,
            quality_report=quality_report, latency_s=latency_s,
            run_kind=run_kind, search_result_count=search_result_count,
            fetched_source_count=fetched_source_count,
            response_mode=str(response.get("response_mode", "content")),
            retry_count=int(response.get("retry_count", 0)),
            search_provider=str(packing_metadata.get("search_provider", "")),
            extractor=str(packing_metadata.get("extractor", "")),
            trace_id=self._current_trace_id(),
        )

        result: Dict[str, object] = {
            "session_id": session_id,
            "note_path": str(layout.note_path),
            "provider": selected_provider,
            "model": selected_model,
            "quality": quality_report.to_dict(),
            "latency_s": latency_s,
            "results_tsv_written": tsv_written,
            "response_mode": response.get("response_mode", "content"),
            "retry_count": int(response.get("retry_count", 0)),
            "packing": packing_metadata,
        }
        if source_ranking:
            result["source_ranking"] = source_ranking
        if eval_result:
            result["evaluation"] = eval_result
        if regression:
            result["regression_check"] = regression
        return result

    def _rank_and_filter_sources(self, markdown_files: List[Path]) -> tuple:
        """Rank sources and filter out very low quality ones."""
        try:
            from ..discovery.source_ranker import SourceRanker
            ranker = SourceRanker(self.config)
            scores = []
            filtered: List[Path] = []
            for f in markdown_files:
                score = ranker._score_file(f)
                scores.append(score.to_dict())
                # Keep files that have content (even low quality)
                if score.has_content:
                    filtered.append(f)
                else:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Source %s filtered: no usable content (words=%d)", f.name, score.word_count,
                    )
            ranking_summary = {
                "total": len(markdown_files),
                "usable": len(filtered),
                "filtered_out": len(markdown_files) - len(filtered),
                "scores": scores[:10],
            }
            return (filtered if filtered else markdown_files, ranking_summary)
        except Exception:
            return (markdown_files, None)

    def _record_evaluation(self, session_id: str, quality_report) -> Optional[Dict[str, object]]:
        """Save quality gate result as an evaluation record."""
        try:
            from ..evaluation import EvaluationRecord, EvaluationStore
            record = EvaluationRecord(
                eval_id=f"qg-{session_id}",
                eval_type="quality_gate",
                baseline_score=0.0,
                candidate_score=quality_report.coverage_score,
                passed=quality_report.passed,
                metadata={
                    "session_id": session_id,
                    "word_count": quality_report.word_count,
                    "structure_score": quality_report.structure_score,
                    "provenance_score": quality_report.provenance_score,
                    "evidence_count": quality_report.evidence_count,
                    "source_diversity": quality_report.source_diversity,
                    "response_mode": getattr(quality_report, "response_mode", ""),
                    "retry_count": getattr(quality_report, "retry_count", 0),
                    "retrieval_metadata": getattr(quality_report, "retrieval_metadata", {}),
                },
            )
            store = EvaluationStore(self.config.repo_root / "knowledge" / "evaluations")
            path = store.save(record)
            return {"eval_id": record.eval_id, "path": str(path)}
        except Exception:
            _logger.debug("Evaluation record save failed", exc_info=True)
            return None

    def _check_regression(self, quality_report) -> Optional[Dict[str, object]]:
        """Check if current quality regressed compared to historical best."""
        try:
            from ..evaluation import EvaluationStore
            store = EvaluationStore(self.config.repo_root / "knowledge" / "evaluations")
            return store.check_regression("quality_gate", quality_report.coverage_score)
        except Exception:
            _logger.debug("Regression check failed", exc_info=True)
            return None

    def _append_results_tsv(
        self,
        session_id: str,
        provider: str,
        model: str,
        quality_report: object,
        latency_s: float,
        trigger_source: str = "synthesize",
        run_kind: str = "cli_manual",
        search_result_count: int = 0,
        fetched_source_count: int = 0,
        exported_obsidian: bool = False,
        gpu_layers_used: str = "",
        peak_vram_mb: str = "",
        quantization: str = "",
        source_prep_mode: str = "",
        response_mode: str = "",
        retry_count: int = 0,
        search_provider: str = "",
        extractor: str = "",
        trace_id: str = "",
    ) -> bool:
        """Append one row to results.tsv. Returns False on failure (never raises)."""
        path = self.config.repo_root / "results.tsv"
        header = (
            "timestamp\tsession_id\tprovider\tmodel\t"
            "coverage\tstructure\tprovenance\tevaluation_score\t"
            "word_count\tlatency_s\ttimeout_s\ttrigger_source\trun_kind\t"
            "search_result_count\tfetched_source_count\t"
            "exported_obsidian\tqg_pass\tprogram_hash\tdescription\t"
            "gpu_layers_used\tpeak_vram_mb\tquantization\tsource_prep_mode\t"
            "response_mode\tretry_count\tsearch_provider\textractor\ttrace_id\n"
        )
        try:
            existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        except OSError:
            existing = ""
        if not existing:
            path.write_text(header, encoding="utf-8")
        else:
            lines = existing.splitlines()
            if lines and "trace_id" not in lines[0]:
                remaining = lines[1:]
                path.write_text(header + ("\n".join(remaining) + "\n" if remaining else ""), encoding="utf-8")

        iso_ts = dt.datetime.now(dt.timezone.utc).isoformat()
        cov = getattr(quality_report, "coverage_score", 0.0)
        struct = getattr(quality_report, "structure_score", 0.0)
        prov_score = getattr(quality_report, "provenance_score", 0.0)
        wc = getattr(quality_report, "word_count", 0)
        passed = getattr(quality_report, "passed", False)
        composite = round(cov * 0.35 + struct * 0.25 + min(wc / 500, 1.0) * 0.3 + prov_score * 0.10, 3)
        desc = f"cov={cov:.3f} str={struct:.3f} w={wc}"
        p_hash = self._program_hash()

        row = (
            f"{iso_ts}\t{session_id}\t{provider}\t{model}\t"
            f"{cov:.3f}\t{struct:.3f}\t{prov_score:.3f}\t{composite}\t"
            f"{wc}\t{latency_s:.1f}\t900.0\t{trigger_source}\t{run_kind}\t"
            f"{search_result_count}\t{fetched_source_count}\t"
            f"{exported_obsidian}\t{passed}\t{p_hash}\t{desc}\t"
            f"{gpu_layers_used}\t{peak_vram_mb}\t{quantization}\t{source_prep_mode}\t"
            f"{response_mode}\t{retry_count}\t{search_provider}\t{extractor}\t{trace_id}\n"
        )
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(row)
            return True
        except OSError as exc:
            _logger.warning("results.tsv write failed: %s", exc)
            return False

    def _program_hash(self) -> str:
        """Compute short hash of research_program.md for traceability."""
        program_path = self.config.repo_root / "research_program.md"
        if not program_path.exists():
            return ""
        try:
            return hashlib.sha256(program_path.read_bytes()).hexdigest()[:12]
        except OSError:
            return ""

    def _build_source_bundle(
        self,
        markdown_files: List[Path],
        max_bytes: Optional[int] = None,
    ) -> Tuple[str, Dict[str, object]]:
        budget_bytes = max_bytes or self.config.max_source_bytes
        selected_sections: List[tuple[int, int, str, str]] = []
        scored_sections: List[tuple[float, int, int, str, str]] = []
        providers: List[str] = []
        extractors: List[str] = []
        for file_index, file in enumerate(markdown_files):
            raw_content = file.read_text(encoding="utf-8", errors="replace")
            metadata, body = self._split_frontmatter(raw_content)
            providers.append(str(metadata.get("search_provider", "")))
            extractors.append(str(metadata.get("extractor", "")))
            for section_index, section in enumerate(self._split_sections(body)):
                score = self._score_section(section, metadata, section_index)
                scored_sections.append((score, file_index, section_index, file.name, section))

        scored_sections.sort(key=lambda item: item[0], reverse=True)
        used_bytes = 0
        for _, file_index, section_index, file_name, section in scored_sections:
            section_payload = f"\n\n===== SOURCE: {file_name} =====\n{section.strip()}\n"
            section_bytes = len(section_payload.encode("utf-8"))
            if selected_sections and used_bytes + section_bytes > budget_bytes:
                continue
            if section_bytes > budget_bytes:
                truncated_bytes = section_payload.encode("utf-8")[:budget_bytes]
                section_payload = truncated_bytes.decode("utf-8", errors="ignore")
                section_bytes = len(section_payload.encode("utf-8"))
            selected_sections.append((file_index, section_index, file_name, section_payload))
            used_bytes += section_bytes
            if used_bytes >= budget_bytes:
                break

        if not selected_sections:
            fallback = markdown_files[0].read_text(encoding="utf-8", errors="replace")
            bundle = fallback[:budget_bytes]
            return bundle, {
                "budget_bytes": budget_bytes,
                "used_bytes": len(bundle.encode("utf-8")),
                "packed_sections": 1,
                "selected_sources": [markdown_files[0].name],
                "search_provider": "",
                "extractor": "",
            }

        selected_sections.sort(key=lambda item: (item[0], item[1]))
        bundle = "".join(section_payload for _, _, _, section_payload in selected_sections)
        selected_source_names = []
        for _, _, file_name, _ in selected_sections:
            if file_name not in selected_source_names:
                selected_source_names.append(file_name)
        return bundle, {
            "budget_bytes": budget_bytes,
            "used_bytes": len(bundle.encode("utf-8")),
            "packed_sections": len(selected_sections),
            "candidate_sections": len(scored_sections),
            "selected_sources": selected_source_names,
            "dropped_sections": max(len(scored_sections) - len(selected_sections), 0),
            "search_provider": next((item for item in providers if item), ""),
            "extractor": next((item for item in extractors if item), ""),
        }

    def _split_frontmatter(self, content: str) -> Tuple[Dict[str, str], str]:
        if not content.startswith("---"):
            return {}, content
        lines = content.splitlines()
        try:
            end_index = lines[1:].index("---") + 1
        except ValueError:
            return {}, content
        metadata: Dict[str, str] = {}
        for line in lines[1:end_index]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("'\"")
        body = "\n".join(lines[end_index + 1 :])
        return metadata, body

    def _split_sections(self, body: str) -> List[str]:
        cleaned = body.strip()
        if not cleaned:
            return []
        parts = re.split(r"(?=^#{1,3}\s)", cleaned, flags=re.MULTILINE)
        sections = [" ".join(part.split()) if not part.lstrip().startswith("#") else part.strip() for part in parts if part.strip()]
        return sections or [cleaned]

    def _score_section(self, section: str, metadata: Mapping[str, str], section_index: int) -> float:
        word_count = len(section.split())
        score = min(word_count / 180.0, 1.0)
        lowered = section.lower()
        if section.lstrip().startswith("#"):
            score += 0.35
        if any(marker in lowered for marker in ("關鍵", "摘要", "結論", "建議")):
            score += 0.25
        if section_index == 0:
            score += 0.2
        metadata_words = metadata.get("word_count")
        try:
            score += min(float(metadata_words or "0") / 5000.0, 0.3)
        except (TypeError, ValueError):
            pass
        return score

    def _current_trace_id(self) -> str:
        try:
            from ..tracing import current_trace

            trace = current_trace()
            return trace.trace_id if trace else ""
        except Exception:
            _logger.debug("Trace ID retrieval failed", exc_info=True)
            return ""

    def _source_bundle_byte_budget(self, num_ctx: int) -> int:
        reserve_tokens = 1536
        return min(self.config.max_source_bytes, max((num_ctx - reserve_tokens) * 4, 4096))

    def _prepare_bundle_for_context(
        self,
        markdown_files: List[Path],
        num_ctx: int,
    ) -> Tuple[str, Dict[str, object], int, int]:
        source_bundle_budget = self._source_bundle_byte_budget(num_ctx)
        source_bundle, packing_metadata = self._build_source_bundle(
            markdown_files,
            max_bytes=source_bundle_budget,
        )
        estimated_prompt_tokens = len(source_bundle.encode("utf-8")) // 4 + 200
        while estimated_prompt_tokens + 768 > num_ctx and source_bundle_budget > 4096:
            source_bundle_budget = max(4096, int(source_bundle_budget * 0.8))
            source_bundle, packing_metadata = self._build_source_bundle(
                markdown_files,
                max_bytes=source_bundle_budget,
            )
            estimated_prompt_tokens = len(source_bundle.encode("utf-8")) // 4 + 200
        max_tokens = min(4096, max(num_ctx - estimated_prompt_tokens - 256, 512))
        packing_metadata["budget_bytes"] = source_bundle_budget
        return source_bundle, packing_metadata, max_tokens, estimated_prompt_tokens

    def _extract_context_window(self, error_message: str) -> Optional[int]:
        match = re.search(r"n_ctx:\s*(\d+)", error_message, re.IGNORECASE)
        if not match:
            match = re.search(r"context length.*?(\d+)", error_message, re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None
