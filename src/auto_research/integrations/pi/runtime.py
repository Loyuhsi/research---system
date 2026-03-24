"""Local Pi HTTP runtime for Auto-Research."""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import parse_qs, urlparse

from ...cli import bootstrap
from ...evaluation import EvaluationStore
from ...structured_output import validate_json_schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PiSkill:
    name: str
    description: str
    input_schema: Dict[str, object]
    output_schema: Dict[str, object]
    risk_level: str
    supports_session_override: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "risk_level": self.risk_level,
            "supports_session_override": self.supports_session_override,
        }


class PiRouter:
    def __init__(self, orchestrator: Any, host: str = "127.0.0.1", port: int = 8787, bearer_token: str = "") -> None:
        self.orchestrator = orchestrator
        self.host = host
        self.port = port
        self.bearer_token = bearer_token
        self.skills = self._build_skills()

    def dispatch(
        self,
        method: str,
        raw_path: str,
        body: Optional[Mapping[str, object]] = None,
    ) -> tuple[int, Dict[str, object]]:
        body = dict(body or {})
        parsed = urlparse(raw_path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if method == "GET" and path == "/health":
            return HTTPStatus.OK, self._health()
        if method == "POST" and path == "/v1/doctor":
            self._apply_provider_override(body)
            lightweight = bool(body.get("lightweight", True))
            doctor_payload = self._doctor_summary(lightweight=lightweight)
            return HTTPStatus.OK, {"ok": True, "doctor": doctor_payload, "lightweight": lightweight}
        if method == "POST" and path == "/v1/chat":
            self._validate(body, CHAT_REQUEST_SCHEMA)
            self._apply_provider_override(body)
            result = self.orchestrator.chat(
                session_key=str(body.get("session_key", "pi:default")),
                text=str(body.get("message", "")),
                mode=str(body.get("mode", "research_only")),
                frontend="pi",
            )
            return HTTPStatus.OK, {"ok": True, "result": result}
        if method == "POST" and path == "/v1/skills/search-memory":
            self._validate(body, SEARCH_MEMORY_SCHEMA)
            self._apply_provider_override(body)
            result = self.orchestrator.memory_search(
                task=str(body.get("query", "")),
                task_type=self._optional_str(body.get("task_type")),
                source_types=body.get("source_types"),
            )
            return HTTPStatus.OK, {
                "ok": True,
                "context": self.orchestrator._retrieve_chat_context(str(body.get("query", ""))),
                "result": result,
            }
        if method == "POST" and path == "/v1/skills/synthesize":
            self._validate(body, SYNTHESIZE_SCHEMA)
            self._apply_provider_override(body)
            session_id = self._ensure_session_for_synthesis(body)
            result = self.orchestrator.synthesize(
                topic=str(body.get("topic", "")),
                session_id=session_id,
                provider=self._optional_str(body.get("provider")),
                model=self._optional_str(body.get("model")),
                run_kind=str(body.get("run_kind", "pi_skill")),
            )
            return HTTPStatus.OK, {"ok": True, "session_id": session_id, "result": result}
        if method == "POST" and path == "/v1/skills/approve-memory":
            self._validate(body, APPROVE_MEMORY_SCHEMA)
            result = self.orchestrator.memory_validate(
                memory_id=str(body.get("memory_id", "")),
                approve=bool(body.get("approve", True)),
            )
            return HTTPStatus.OK, {"ok": True, "result": result}
        if method == "POST" and path == "/v1/skills/export-obsidian":
            self._validate(body, EXPORT_OBSIDIAN_SCHEMA)
            result = self._export_obsidian(body)
            return HTTPStatus.OK, {"ok": True, "result": result}
        if method == "GET" and path == "/v1/evaluations":
            eval_type = self._first_query_value(query, "type")
            store = EvaluationStore(self.orchestrator.config.repo_root / "knowledge" / "evaluations")
            records = [record.to_dict() for record in store.load_all(eval_type=eval_type)]
            return HTTPStatus.OK, {"ok": True, "count": len(records), "records": records}
        if method == "GET" and path == "/v1/skills":
            return HTTPStatus.OK, {"ok": True, "skills": [skill.to_dict() for skill in self.skills.values()]}
        return HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown endpoint: {path}"}

    def _health(self) -> Dict[str, object]:
        status = self.orchestrator.provider_override_status()
        return {
            "ok": True,
            "server": {"host": self.host, "bind": self.host, "port": self.port},
            "provider": status.get("effective_provider"),
            "model": status.get("effective_model"),
        }

    def _doctor_summary(self, lightweight: bool = True) -> Dict[str, object]:
        if not lightweight:
            return self.orchestrator.doctor()
        config = self.orchestrator.config
        llm = self.orchestrator.llm
        http = self.orchestrator.http_client
        return {
            "provider": config.provider,
            "model": config.model,
            "provider_overrides": self.orchestrator.provider_override_status(),
            "services": {
                "ollama": llm.check_service(f"{config.ollama_base}/api/tags", http),
                "lmstudio": llm.check_service(f"{config.lmstudio_base}/v1/models", http, llm.model_headers("lmstudio")),
                "vllm": llm.check_service(f"{config.vllm_base}/v1/models", http, llm.model_headers("vllm")),
            },
            "provider_selection": llm.select_provider(http, verify_inference=False),
        }

    def _apply_provider_override(self, body: Mapping[str, object]) -> None:
        provider = self._optional_str(body.get("provider"))
        scope = self._optional_str(body.get("scope")) or "session"
        session_key = self._optional_str(body.get("session_key"))
        if provider or "scope" in body:
            self.orchestrator.set_provider_override(
                provider or "auto",
                session_key=session_key,
                global_scope=scope == "global",
            )

    def _ensure_session_for_synthesis(self, body: Mapping[str, object]) -> str:
        session_id = self._optional_str(body.get("session_id"))
        if session_id:
            return session_id
        topic = str(body.get("topic", "")).strip() or "pi-session"
        session_id = f"pi-{_slug(topic)}-{int(time.time())}"
        layout = self.orchestrator.config.resolve_layout(session_id)
        layout.ensure()
        if not any(layout.parsed_dir.glob("*.md")):
            (layout.parsed_dir / "seed.md").write_text(f"# {topic}\n\n{topic}\n", encoding="utf-8")
        return session_id

    def _export_obsidian(self, body: Mapping[str, object]) -> Dict[str, object]:
        session_id = self._optional_str(body.get("session_id"))
        include_diagnostics = bool(body.get("include_diagnostics", False))
        if session_id:
            return self.orchestrator.promote_note(session_id, approved=True)
        if include_diagnostics:
            from ..obsidian import ObsidianExporter

            exporter = ObsidianExporter(self.orchestrator.config)
            path = exporter.export_diagnostics(self.orchestrator.doctor())
            return {"diagnostics_path": str(path)}
        raise ValueError("session_id or include_diagnostics=true is required")

    def _build_skills(self) -> Dict[str, PiSkill]:
        return {
            "search-memory": PiSkill(
                name="search-memory",
                description="搜尋批准的 memory records 與 skills。",
                input_schema=SEARCH_MEMORY_SCHEMA,
                output_schema={"type": "object"},
                risk_level="low",
            ),
            "synthesize": PiSkill(
                name="synthesize",
                description="產生研究筆記並寫入 output/notes。",
                input_schema=SYNTHESIZE_SCHEMA,
                output_schema={"type": "object"},
                risk_level="medium",
            ),
            "approve-memory": PiSkill(
                name="approve-memory",
                description="批准或拒絕 memory record。",
                input_schema=APPROVE_MEMORY_SCHEMA,
                output_schema={"type": "object"},
                risk_level="medium",
            ),
            "export-obsidian": PiSkill(
                name="export-obsidian",
                description="匯出筆記或診斷報告到 Obsidian。",
                input_schema=EXPORT_OBSIDIAN_SCHEMA,
                output_schema={"type": "object"},
                risk_level="medium",
            ),
        }

    @staticmethod
    def _optional_str(value: object) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _first_query_value(query: Mapping[str, list[str]], key: str) -> Optional[str]:
        values = query.get(key, [])
        return values[0] if values else None

    @staticmethod
    def _validate(payload: Mapping[str, object], schema: Mapping[str, object]) -> None:
        validate_json_schema(payload, schema)


class PiRequestHandler(BaseHTTPRequestHandler):
    router: PiRouter

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        try:
            if not self._check_auth():
                self._write_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "Unauthorized"})
                return
            body = self._read_json_body() if method == "POST" else {}
            status, payload = self.router.dispatch(method, self.path, body)
        except ValueError as exc:
            status, payload = HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)}
        except Exception:
            logger.exception("Pi runtime request failed")
            status, payload = HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "Internal server error"}
        self._write_json(status, payload)

    _MAX_BODY_BYTES = 1_048_576  # 1 MB

    def _read_json_body(self) -> Dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        if content_length > self._MAX_BODY_BYTES:
            raise ValueError(f"Request body too large ({content_length} bytes, max {self._MAX_BODY_BYTES})")
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _write_json(self, status: int, payload: Mapping[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _check_auth(self) -> bool:
        """Validate Bearer token if configured. /health is always open."""
        if not self.router.bearer_token:
            return True
        if self.path == "/health":
            return True
        auth_header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.router.bearer_token}"
        return auth_header == expected

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("Pi runtime: " + format, *args)


class PiRuntimeServer:
    def __init__(self, orchestrator: Any, host: str = "127.0.0.1", port: int = 8787, bearer_token: str = "") -> None:
        self.host = host
        self.port = port
        self.bearer_token = bearer_token
        self.router = PiRouter(orchestrator, host=host, port=port, bearer_token=bearer_token)
        handler_cls = type("BoundPiRequestHandler", (PiRequestHandler,), {})
        handler_cls.router = self.router
        self._server = ThreadingHTTPServer((host, port), handler_cls)
        self.port = self._server.server_port
        self.router.port = self.port
        self._thread: Optional[threading.Thread] = None

    def serve_forever(self) -> None:
        logger.info("Pi runtime listening on http://%s:%s", self.host, self.port)
        self._server.serve_forever()

    def start_in_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)


def build_pi_runtime(orchestrator: Any | None = None, host: str = "127.0.0.1", port: int = 8787) -> PiRuntimeServer:
    token = os.environ.get("PI_RUNTIME_TOKEN", "")
    if not token:
        token = secrets.token_urlsafe(32)
        logger.info("Generated Pi runtime bearer token (set PI_RUNTIME_TOKEN to override)")
    server = PiRuntimeServer(orchestrator or bootstrap(), host=host, port=port, bearer_token=token)
    # Write token to .runtime/ for authorized clients
    runtime_dir = Path(".runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "pi-token").write_text(token, encoding="utf-8")
    logger.info("Pi runtime token written to .runtime/pi-token")
    return server


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


CHAT_REQUEST_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "minLength": 1},
        "session_key": {"type": "string"},
        "mode": {"type": "string"},
        "provider": {"type": "string"},
        "scope": {"type": "string"},
    },
    "required": ["message"],
    "additionalProperties": False,
}

SEARCH_MEMORY_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "task_type": {"type": "string"},
        "provider": {"type": "string"},
        "scope": {"type": "string"},
        "session_key": {"type": "string"},
    },
    "required": ["query"],
    "additionalProperties": False,
}

SYNTHESIZE_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "topic": {"type": "string", "minLength": 1},
        "session_id": {"type": "string"},
        "provider": {"type": "string"},
        "model": {"type": "string"},
        "run_kind": {"type": "string"},
        "scope": {"type": "string"},
        "session_key": {"type": "string"},
    },
    "required": ["topic"],
    "additionalProperties": False,
}

APPROVE_MEMORY_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "memory_id": {"type": "string", "minLength": 1},
        "approve": {"type": "boolean"},
    },
    "required": ["memory_id"],
    "additionalProperties": False,
}

EXPORT_OBSIDIAN_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string"},
        "include_diagnostics": {"type": "boolean"},
    },
    "required": [],
    "additionalProperties": False,
}
