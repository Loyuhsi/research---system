"""SkillMemoryService — thin facade that delegates to sub-modules."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from ...http_client import JsonHttpClient
from ...runtime import AutoResearchConfig
from ..llm_provider import LlmProviderService
from . import extraction, indexing, materialization, retrieval, validation
from .embedding import cosine_similarity
from .util import contains_secrets, dedupe, strip_frontmatter, tokenize


class SkillMemoryService:
    def __init__(
        self,
        config: AutoResearchConfig,
        http_client: JsonHttpClient,
        llm: LlmProviderService,
    ) -> None:
        self.config = config
        self.http_client = http_client
        self.llm = llm

    # -- Draft extraction -----------------------------------------------------

    def memory_extract(
        self,
        session_id: str,
        task_type: str = "research_session",
        status: str = "success",
        summary_override: Optional[str] = None,
    ) -> Dict[str, object]:
        return extraction.memory_extract(self.config, session_id, task_type, status, summary_override)

    # -- Validation -----------------------------------------------------------

    def memory_validate(self, memory_id: str, approve: bool = False) -> Dict[str, object]:
        return validation.memory_validate(
            self.config, memory_id, approve,
            rebuild_index_fn=self.memory_index_rebuild,
        )

    # -- Retrieval ------------------------------------------------------------

    def retrieve_context(
        self,
        task: str,
        task_type: Optional[str] = None,
        source_types: Optional[Sequence[str]] = None,
        top_memory: int = 3,
        top_skills: int = 2,
    ) -> Dict[str, object]:
        return retrieval.retrieve_context(
            self.config, self.http_client, self.llm,
            task, task_type, source_types, top_memory, top_skills,
        )

    def memory_search(
        self,
        task: str,
        task_type: Optional[str] = None,
        source_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        return retrieval.memory_search(self.config, self.http_client, self.llm, task, task_type, source_types)

    # -- Index rebuild --------------------------------------------------------

    def memory_index_rebuild(self) -> Dict[str, object]:
        return indexing.memory_index_rebuild(self.config, self.http_client, self.llm)

    # -- Skill materialization/export -----------------------------------------

    def skill_materialize(self, candidate_name: str) -> Dict[str, object]:
        return materialization.skill_materialize(self.config, candidate_name)

    def skill_export(self, target: str = "github", skill_id: Optional[str] = None) -> Dict[str, object]:
        return materialization.skill_export(self.config, target, skill_id)

    # -- Backward-compatible internal helpers (used by tests) -----------------

    def _contains_secrets(self, text: str) -> bool:
        return contains_secrets(text)

    def _lexical_score(self, query: str, corpus: Sequence[str]) -> float:
        return retrieval._lexical_score(query, corpus)

    def _cosine_similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        return cosine_similarity(left, right)

    def _strip_frontmatter(self, content: str) -> str:
        return strip_frontmatter(content)

    def _tokenize(self, text: str):
        return tokenize(text)

    def _dedupe(self, items):
        return dedupe(items)
