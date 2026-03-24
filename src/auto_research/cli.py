from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .discovery.source_ranker import SourceRanker
from .discovery.topic_expander import TopicExpander
from .http_client import JsonHttpClient
from .orchestrator import Orchestrator
from .reflection.gap_detector import GapDetector
from .reflection.strategy_advisor import StrategyAdvisor
from .services.llm_provider import LlmProviderService
from .services.fetcher import FetcherService
from .services.vault import VaultService
from .services.synthesizer import SynthesizerService
from .services.evoskill import EvoSkillService
from .services.tool_runner import ToolRunnerService
from .services.skill_memory import SkillMemoryService
from .services.task_review import TaskReviewService
from .runtime import load_config
from .conversation_store import JsonFileConversationStore
from .registry import ServiceRegistry
from .events import EventBus
from .event_handlers import wire_event_handlers

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-Research V3.6 orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Run quick health checks")
    doctor.add_argument("--json", action="store_true", help="Print JSON output")

    chat = subparsers.add_parser("chat", help="Send a chat request through the orchestrator")
    chat.add_argument("--session-key", required=True)
    chat.add_argument("--text", required=True)
    chat.add_argument("--mode", default="research_only")
    chat.add_argument("--frontend", default="cli")

    reset = subparsers.add_parser("reset-chat", help="Reset a session conversation")
    reset.add_argument("--session-key", required=True)

    fetch_public = subparsers.add_parser("fetch-public", help="Run the public fetch worker and import artifacts")
    fetch_public.add_argument("--topic", required=True)
    fetch_public.add_argument("urls", nargs="+")

    fetch_private = subparsers.add_parser("fetch-private", help="Run the private fetch worker and import artifacts")
    fetch_private.add_argument("--topic", required=True)
    fetch_private.add_argument("--token-env")
    fetch_private.add_argument("urls", nargs="+")

    synthesize = subparsers.add_parser("synthesize", help="Generate a note from imported research artifacts")
    synthesize.add_argument("--topic", required=True)
    synthesize.add_argument("--session-id", required=True)
    synthesize.add_argument("--provider")
    synthesize.add_argument("--model")

    promote = subparsers.add_parser("promote-note", help="Copy a note into the Obsidian vault")
    promote.add_argument("--session-id", required=True)
    promote.add_argument("--approve", action="store_true")

    tool = subparsers.add_parser("tool-run", help="Run an allowlisted tooling binding")
    tool.add_argument("--binding", required=True)
    tool.add_argument("--source", required=True)
    tool.add_argument("--output", required=True)
    tool.add_argument("--format", default="pdf")
    tool.add_argument("--mode", default="semi_trusted_tooling")
    tool.add_argument("--frontend", default="cli")
    tool.add_argument("--dry-run", action="store_true")

    rd_agent = subparsers.add_parser("rd-agent-run", help="Run the RD-Agent sandbox container")
    rd_agent.add_argument("--mode", default="high_risk_execution")
    rd_agent.add_argument("--frontend", default="cli")
    rd_agent.add_argument("--dry-run", action="store_true")

    evo_log = subparsers.add_parser("evo-log", help="Record a task outcome")
    evo_log.add_argument("--task-id", required=True)
    evo_log.add_argument("--status", required=True)
    evo_log.add_argument("--summary", required=True)

    evo_propose = subparsers.add_parser("evo-propose", help="Create a candidate skill")
    evo_propose.add_argument("--task-id", required=True)
    evo_propose.add_argument("--candidate-name", required=True)
    evo_propose.add_argument("--prompt", required=True)

    evo_validate = subparsers.add_parser("evo-validate", help="Validate a candidate skill")
    evo_validate.add_argument("--candidate-name", required=True)
    evo_validate.add_argument("--baseline-score", type=float, required=True)
    evo_validate.add_argument("--candidate-score", type=float, required=True)

    evo_promote = subparsers.add_parser("evo-promote", help="Promote a validated candidate skill")
    evo_promote.add_argument("--candidate-name", required=True)
    evo_promote.add_argument("--approve", action="store_true")

    reflect = subparsers.add_parser("reflect", help="Run self-reflection: detect gaps and propose strategies")
    reflect.add_argument("--provider", help="LLM provider for strategy advice")
    reflect.add_argument("--model", help="LLM model for strategy advice")

    discover = subparsers.add_parser("discover", help="Propose new research topics from existing notes")
    discover.add_argument("--provider", help="LLM provider for topic expansion")
    discover.add_argument("--model", help="LLM model for topic expansion")

    rank = subparsers.add_parser("rank-sources", help="Rank sources in a session by quality")
    rank.add_argument("--session-id", required=True)

    memory_extract = subparsers.add_parser("memory-extract", help="Extract a memory draft from a session")
    memory_extract.add_argument("--session-id", required=True)
    memory_extract.add_argument("--task-type", default="research_session")
    memory_extract.add_argument("--status", default="success")
    memory_extract.add_argument("--summary")

    memory_search = subparsers.add_parser("memory-search", help="Search approved memory records and skills")
    memory_search.add_argument("--task", required=True)
    memory_search.add_argument("--task-type")
    memory_search.add_argument("--source-type", action="append", default=[])

    memory_validate = subparsers.add_parser("memory-validate", help="Validate or approve a memory record")
    memory_validate.add_argument("--id", required=True)
    memory_validate.add_argument("--approve", action="store_true")

    memory_index = subparsers.add_parser("memory-index", help="Manage the skill memory index")
    memory_index_subparsers = memory_index.add_subparsers(dest="memory_index_command", required=True)
    memory_index_subparsers.add_parser("rebuild", help="Rebuild the local SQLite skill memory index")

    post_review = subparsers.add_parser("post-task-review", help="Run post-task review and create drafts")
    post_review.add_argument("--session-id", required=True)
    post_review.add_argument("--status", choices=["success", "failed"], required=True)
    post_review.add_argument("--action", default="research_session")
    post_review.add_argument("--task-id")
    post_review.add_argument("--summary")
    post_review.add_argument("--approve-memory", action="store_true")

    skill_materialize = subparsers.add_parser("skill-materialize", help="Add metadata/citations scaffolding to a skill candidate")
    skill_materialize.add_argument("--candidate", required=True)

    skill_export = subparsers.add_parser("skill-export", help="Export approved repo skills to a mirror target")
    skill_export.add_argument("--target", default="github")
    skill_export.add_argument("--skill-id")

    obsidian_export = subparsers.add_parser("obsidian-export", help="Export artifacts to Obsidian vault")
    obsidian_export.add_argument("--session-id", help="Export a specific session note")
    obsidian_export.add_argument("--all", action="store_true", help="Export all approved memories and evaluations")
    obsidian_export.add_argument("--include-diagnostics", action="store_true", help="Include current doctor report")

    return parser



def bootstrap() -> Orchestrator:
    config = load_config()
    registry = ServiceRegistry()
    event_bus = EventBus()
    http_client = JsonHttpClient()
    conversations = JsonFileConversationStore(config.repo_root / "output" / "conversations")
    
    registry.register("core.config", config)
    registry.register("core.events", event_bus)
    registry.register("core.http", http_client)
    
    llm = LlmProviderService(config)
    vault = VaultService(config)
    fetcher = FetcherService(config, vault)
    
    registry.register("service.llm", llm)
    registry.register("service.vault", vault)
    registry.register("service.fetcher", fetcher)
    
    synthesizer = SynthesizerService(config, http_client, llm, vault)
    evoskill = EvoSkillService(config)
    tool_runner = ToolRunnerService(config)
    strategy_advisor = StrategyAdvisor(config, http_client, llm)
    skill_memory = SkillMemoryService(config, http_client, llm)
    task_review = TaskReviewService(config, skill_memory, evoskill, strategy_advisor)
    
    registry.register("service.synthesizer", synthesizer)
    registry.register("service.evoskill", evoskill)
    registry.register("service.tool_runner", tool_runner)
    registry.register("service.strategy_advisor", strategy_advisor)
    registry.register("service.skill_memory", skill_memory)
    registry.register("service.task_review", task_review)
    
    required = [
        "core.config", "core.events", "core.http",
        "service.llm", "service.vault", "service.fetcher",
        "service.synthesizer", "service.evoskill", "service.tool_runner",
        "service.skill_memory", "service.task_review",
    ]
    registry.validate(required)

    wire_event_handlers(
        event_bus=event_bus,
        evoskill=evoskill,
        skill_memory=skill_memory,
        telemetry_path=config.repo_root / "output" / "telemetry.jsonl",
    )

    # Startup validation visibility (warning mode — log but never raise)
    _report_startup_validation(config)

    return Orchestrator(
        config=config,
        event_bus=event_bus,
        registry=registry,
        llm_service=llm,
        fetcher_service=fetcher,
        vault_service=vault,
        http_client=http_client,
        conversation_store=conversations
    )

def _report_startup_validation(config) -> None:
    """Log config validation results at startup. Warning mode only — never raises."""
    try:
        from .config_schema import ConfigValidator
        validator = ConfigValidator()
        report = validator.validate_runtime(config, check_connectivity=False)
        if report.errors:
            for error in report.errors:
                logger.error("[startup-validation] runtime error: %s", error)
        if report.warnings:
            for warning in report.warnings:
                logger.warning("[startup-validation] warning: %s", warning)
        if not report.errors and not report.warnings:
            logger.info("[startup-validation] Config validation passed (no errors, no warnings)")
    except Exception as exc:
        logger.warning("[startup-validation] Could not run config validation: %s", exc)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    orchestrator = bootstrap()

    if args.command == "doctor":
        result = orchestrator.doctor()
        result["registry_services"] = orchestrator.registry.list_services()
        result["event_subscribers"] = orchestrator.event_bus.list_subscribers()
    elif args.command == "chat":
        result = orchestrator.chat(
            session_key=args.session_key,
            text=args.text,
            mode=args.mode,
            frontend=args.frontend,
        )
    elif args.command == "reset-chat":
        orchestrator.reset_conversation(args.session_key)
        result = {"session_key": args.session_key, "reset": True}
    elif args.command == "fetch-public":
        result = orchestrator.fetch_public(args.topic, args.urls)
    elif args.command == "fetch-private":
        result = orchestrator.fetch_private(args.topic, args.urls, token_env=args.token_env)
    elif args.command == "synthesize":
        result = orchestrator.synthesize(args.topic, args.session_id, provider=args.provider, model=args.model)
    elif args.command == "promote-note":
        result = orchestrator.promote_note(args.session_id, approved=args.approve)
    elif args.command == "tool-run":
        result = orchestrator.run_tool_binding(
            binding_name=args.binding,
            source=Path(args.source),
            output=Path(args.output),
            mode=args.mode,
            frontend=args.frontend,
            fmt=args.format,
            dry_run=args.dry_run,
        )
    elif args.command == "rd-agent-run":
        result = orchestrator.run_rd_agent(args.mode, frontend=args.frontend, dry_run=args.dry_run)
    elif args.command == "evo-log":
        result = orchestrator.evo_log(args.task_id, args.status, args.summary)
    elif args.command == "evo-propose":
        result = orchestrator.evo_propose(args.task_id, args.candidate_name, args.prompt)
    elif args.command == "evo-validate":
        result = orchestrator.evo_validate(args.candidate_name, args.baseline_score, args.candidate_score)
    elif args.command == "evo-promote":
        result = orchestrator.evo_promote(args.candidate_name, approved=args.approve)
    elif args.command == "reflect":
        gap_detector = GapDetector(orchestrator.config)
        gap_report = gap_detector.scan()
        advisor = orchestrator.registry.resolve("service.strategy_advisor")
        advice = advisor.advise(gap_report, provider=args.provider, model=args.model)
        result = {"gap_report": gap_report.to_dict(), "strategy": advice}
    elif args.command == "discover":
        http_client = orchestrator.registry.resolve("core.http")
        llm = orchestrator.registry.resolve("service.llm")
        expander = TopicExpander(orchestrator.config, http_client, llm)
        result = expander.discover(provider=args.provider, model=args.model)
    elif args.command == "rank-sources":
        ranker = SourceRanker(orchestrator.config)
        result = ranker.rank_session(args.session_id)
    elif args.command == "memory-extract":
        result = orchestrator.memory_extract(
            session_id=args.session_id,
            task_type=args.task_type,
            status=args.status,
            summary_override=args.summary,
        )
    elif args.command == "memory-search":
        result = orchestrator.memory_search(
            task=args.task,
            task_type=args.task_type,
            source_types=args.source_type,
        )
    elif args.command == "memory-validate":
        result = orchestrator.memory_validate(args.id, approve=args.approve)
    elif args.command == "memory-index":
        if args.memory_index_command == "rebuild":
            result = orchestrator.memory_index_rebuild()
        else:  # pragma: no cover
            parser.error(f"Unsupported memory-index command: {args.memory_index_command}")
            return 2
    elif args.command == "post-task-review":
        result = orchestrator.post_task_review(
            session_id=args.session_id,
            status=args.status,
            action=args.action,
            task_id=args.task_id,
            summary=args.summary,
            approve_memory=args.approve_memory,
        )
    elif args.command == "skill-materialize":
        result = orchestrator.skill_materialize(args.candidate)
    elif args.command == "skill-export":
        result = orchestrator.skill_export(target=args.target, skill_id=args.skill_id)
    elif args.command == "obsidian-export":
        from .integrations.obsidian import ObsidianExporter
        exporter = ObsidianExporter(orchestrator.config)
        exported = []
        if args.session_id:
            layout = orchestrator.config.resolve_layout(args.session_id)
            if layout.note_path.exists():
                path = exporter.export_note(args.session_id, layout.note_path)
                exported.append({"type": "note", "session_id": args.session_id, "path": str(path)})
        if getattr(args, "all", False):
            records_dir = orchestrator.config.memory_records_dir
            if records_dir.exists():
                for rp in records_dir.glob("*.json"):
                    try:
                        rd = json.loads(rp.read_text(encoding="utf-8"))
                        if rd.get("status") == "approved":
                            path = exporter.export_memory(rd)
                            exported.append({"type": "memory", "id": rd.get("id"), "path": str(path)})
                    except Exception:
                        logger.debug("Skipping memory record %s", rp, exc_info=True)
                        continue
            eval_dir = orchestrator.config.repo_root / "knowledge" / "evaluations"
            if eval_dir.exists():
                for ep in eval_dir.glob("*.json"):
                    try:
                        ed = json.loads(ep.read_text(encoding="utf-8"))
                        path = exporter.export_evaluation(ed)
                        exported.append({"type": "evaluation", "id": ed.get("eval_id"), "path": str(path)})
                    except Exception:
                        logger.debug("Skipping evaluation %s", ep, exc_info=True)
                        continue
        if args.include_diagnostics:
            doctor = orchestrator.doctor()
            path = exporter.export_diagnostics(doctor)
            exported.append({"type": "diagnostics", "path": str(path)})
        result = {"exported": exported, "count": len(exported)}
    else:  # pragma: no cover
        parser.error(f"Unsupported command: {args.command}")
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
