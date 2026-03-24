import json
import tempfile
import unittest
from pathlib import Path

from auto_research.conversation_store import InMemoryConversationStore
from auto_research.orchestrator import ExecutionError, PolicyError
from auto_research.runtime import load_config

from conftest import FakeHttpClient, make_temp_repo, create_test_orchestrator


class OrchestratorTests(unittest.TestCase):
    def _make(self, tmpdir, http_client=None, environ=None):
        repo_root = make_temp_repo(tmpdir)
        effective_environ = {"SKILL_MEMORY_EMBEDDING_PROVIDER": "disabled"}
        if environ:
            effective_environ.update(environ)
        config = load_config(repo_root=repo_root, environ=effective_environ)
        return create_test_orchestrator(config, http_client=http_client), repo_root

    def _write_session(self, orchestrator, session_id: str, topic: str = "agent memory"):
        layout = orchestrator.config.resolve_layout(session_id)
        layout.ensure()
        layout.status_path.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "sources": [
                        {
                            "topic": topic,
                            "url": "https://example.com/source",
                            "visibility": "public",
                            "fetch_method": "scrapling",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        layout.note_path.write_text(
            f'---\ntopic: "{topic}"\n---\n# Summary\nThis note covers {topic}.\n',
            encoding="utf-8",
        )
        (layout.parsed_dir / "source.md").write_text(f"# Parsed\n{topic} reference\n", encoding="utf-8")

    def test_secret_allowlist_filters_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, repo_root = self._make(tmpdir)
            config = load_config(repo_root=repo_root, environ={})
            safe_env = config.safe_env()
            self.assertEqual(safe_env["GITHUB_TOKEN"], "from-env-file")
            self.assertNotIn("TELEGRAM_MODEL", safe_env)

    def test_import_legacy_session_routes_raw_and_parsed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator, repo_root = self._make(tmpdir)
            legacy_dir = repo_root / "output" / "sources" / "session-1"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dir / "source.md").write_text("# parsed", encoding="utf-8")
            (legacy_dir / "source.raw.json").write_text('{"raw": true}', encoding="utf-8")
            (legacy_dir / "source.status.json").write_text('{"url": "https://example.com"}', encoding="utf-8")

            layout = orchestrator.import_legacy_session("session-1")

            self.assertTrue((layout.parsed_dir / "source.md").exists())
            self.assertTrue((layout.raw_dir / "source.raw.json").exists())
            status = json.loads(layout.status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["session_id"], "session-1")
            self.assertEqual(status["sources"][0]["url"], "https://example.com")

    def test_promotion_gate_blocks_without_approval_and_copies_when_approved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator, _ = self._make(tmpdir)
            layout = orchestrator.config.resolve_layout("session-2")
            layout.note_path.parent.mkdir(parents=True, exist_ok=True)
            layout.note_path.write_text("# note", encoding="utf-8")

            with self.assertRaises(PolicyError):
                orchestrator.promote_note("session-2", approved=False)

            result = orchestrator.promote_note("session-2", approved=True)
            self.assertTrue(Path(result["vault_path"]).exists())

    def test_tool_policy_blocks_output_outside_allowlist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator, repo_root = self._make(tmpdir)
            source = repo_root / "document.docx"
            source.write_text("content", encoding="utf-8")

            with self.assertRaises(PolicyError):
                orchestrator.run_tool_binding(
                    binding_name="libreoffice-convert",
                    source=source,
                    output=repo_root / "forbidden" / "document.pdf",
                    mode="semi_trusted_tooling",
                    dry_run=True,
                )

            result = orchestrator.run_tool_binding(
                binding_name="libreoffice-convert",
                source=source,
                output=repo_root / "staging" / "tooling" / "document.pdf",
                mode="semi_trusted_tooling",
                dry_run=True,
            )
            self.assertIn("soffice", result["command"][0])
            self.assertTrue((repo_root / "knowledge" / "logs" / "tool-run-document.json").exists())

    def test_telegram_cannot_use_high_risk_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator, _ = self._make(tmpdir)
            with self.assertRaises(PolicyError):
                orchestrator.set_mode("telegram", "high_risk_execution")

    def test_rd_agent_command_only_mounts_sandbox_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator, repo_root = self._make(tmpdir)
            command = orchestrator.build_rd_agent_command()
            command_text = " ".join(command)
            self.assertIn("/workspace/in", command_text)
            self.assertIn("/workspace/out", command_text)
            self.assertNotIn(str(repo_root / ".env"), command_text)

    def test_evoskill_requires_validation_before_promotion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator, _ = self._make(tmpdir)
            orchestrator.evo_log("task-1", "failed", "missing skill")
            orchestrator.evo_propose("task-1", "candidate-skill", "Use a safer fetch recipe.")
            validation = orchestrator.evo_validate("candidate-skill", baseline_score=0.3, candidate_score=0.7)
            result = orchestrator.evo_promote("candidate-skill", approved=True)
            self.assertTrue(validation["passed"])
            self.assertTrue((Path(result["skill_dir"]) / "SKILL.md").exists())

    def test_chat_injects_approved_memory_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = FakeHttpClient(responses=[{"choices": [{"message": {"content": "已收到"}}]}])
            orchestrator, repo_root = self._make(tmpdir, http_client=fake_client)
            self._write_session(orchestrator, "session-chat", topic="agent memory")
            draft = orchestrator.memory_extract("session-chat", task_type="research_session")
            orchestrator.memory_validate(draft["draft_id"], approve=True)

            skill_dir = repo_root / "skills" / "memory-guidance"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: Memory Guidance",
                        "description: Reuse approved memory records before researching from scratch.",
                        "---",
                    ]
                ),
                encoding="utf-8",
            )
            orchestrator.memory_index_rebuild()

            result = orchestrator.chat(session_key="chat-1", text="agent memory guidance")

            system_prompt = fake_client.call_log[0]["payload"]["messages"][0]["content"]
            self.assertEqual(result["context_hits"]["memory"], 1)
            self.assertGreaterEqual(result["context_hits"]["skills"], 1)
            self.assertIn("Approved memory records", system_prompt)
            self.assertIn("Approved skills", system_prompt)

    def test_extract_message_content_strips_think_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator, _ = self._make(tmpdir)
            content = orchestrator.extract_message_content(
                {"choices": [{"message": {"content": "<think>hidden</think>\nvisible answer"}}]}
            )
            self.assertEqual(content, "visible answer")

class ExtractMessageContentTests(unittest.TestCase):
    def _make_orchestrator(self, tmpdir):
        repo_root = make_temp_repo(tmpdir)
        return create_test_orchestrator(load_config(repo_root=repo_root, environ={"SKILL_MEMORY_EMBEDDING_PROVIDER": "disabled"}))

    def test_empty_choices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._make_orchestrator(tmpdir)
            self.assertEqual(orchestrator.extract_message_content({"choices": []}), "")

    def test_missing_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._make_orchestrator(tmpdir)
            self.assertEqual(orchestrator.extract_message_content({"choices": [{"index": 0}]}), "")

    def test_nested_think_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._make_orchestrator(tmpdir)
            content = orchestrator.extract_message_content(
                {"choices": [{"message": {"content": "<think>a</think>text<think>b</think> end"}}]}
            )
            self.assertNotIn("<think>", content)
            self.assertIn("text", content)

    def test_orphaned_closing_think_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._make_orchestrator(tmpdir)
            content = orchestrator.extract_message_content(
                {"choices": [{"message": {"content": "reasoning stuff</think>\nactual answer"}}]}
            )
            self.assertEqual(content, "actual answer")

    def test_no_choices_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._make_orchestrator(tmpdir)
            self.assertEqual(orchestrator.extract_message_content({}), "")


class ChatPayloadTests(unittest.TestCase):
    def test_chat_builds_correct_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_temp_repo(tmpdir)
            fake_client = FakeHttpClient(responses=[{"choices": [{"message": {"content": "測試回覆"}}]}])
            config = load_config(repo_root=repo_root, environ={"SKILL_MEMORY_EMBEDDING_PROVIDER": "disabled"})
            orchestrator = create_test_orchestrator(config, http_client=fake_client)

            result = orchestrator.chat(session_key="test-1", text="hello", mode="research_only", frontend="cli")

            self.assertEqual(result["reply"], "測試回覆")
            call = fake_client.call_log[0]
            self.assertEqual(call["method"], "POST")
            payload = call["payload"]
            self.assertEqual(payload["model"], "qwen3.5:9b")
            self.assertEqual(payload["temperature"], 0.2)
            self.assertFalse(payload["think"])

    def test_chat_appends_to_conversation_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_temp_repo(tmpdir)
            fake_client = FakeHttpClient(
                responses=[
                    {"choices": [{"message": {"content": "reply1"}}]},
                    {"choices": [{"message": {"content": "reply2"}}]},
                ]
            )
            orchestrator = create_test_orchestrator(
                load_config(repo_root=repo_root, environ={"SKILL_MEMORY_EMBEDDING_PROVIDER": "disabled"}),
                http_client=fake_client,
            )

            orchestrator.chat(session_key="s1", text="msg1")
            orchestrator.chat(session_key="s1", text="msg2")

            messages = orchestrator.conversations.get_messages("s1")
            self.assertEqual(len(messages), 4)
            self.assertEqual(messages[0]["content"], "msg1")
            self.assertEqual(messages[1]["content"], "reply1")


class ConversationStoreTests(unittest.TestCase):
    def test_truncation_at_max_rounds(self):
        store = InMemoryConversationStore(max_rounds=2)
        for index in range(5):
            store.append_turn("key", f"user-{index}", f"assistant-{index}")
        messages = store.get_messages("key")
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["content"], "user-3")

    def test_reset_clears_session(self):
        store = InMemoryConversationStore()
        store.append_turn("key", "hello", "hi")
        store.reset("key")
        self.assertEqual(store.get_messages("key"), [])

    def test_build_prompt_messages_includes_system(self):
        store = InMemoryConversationStore()
        store.append_turn("key", "prev-user", "prev-assistant")
        messages = store.build_prompt_messages("key", "sys prompt", "new msg")
        self.assertEqual(messages[0], {"role": "system", "content": "sys prompt"})
        self.assertEqual(messages[-1], {"role": "user", "content": "new msg"})
        self.assertEqual(len(messages), 4)


class EvoSkillTests(unittest.TestCase):
    def test_evo_validate_failing_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_temp_repo(tmpdir)
            orchestrator = create_test_orchestrator(load_config(repo_root=repo_root, environ={}))
            orchestrator.evo_log("task-1", "failed", "issue")
            orchestrator.evo_propose("task-1", "bad-skill", "prompt")

            result = orchestrator.evo_validate("bad-skill", baseline_score=0.8, candidate_score=0.3)
            self.assertFalse(result["passed"])

            with self.assertRaises(PolicyError):
                orchestrator.evo_promote("bad-skill", approved=True)

    def test_evo_promote_without_validation_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_temp_repo(tmpdir)
            orchestrator = create_test_orchestrator(load_config(repo_root=repo_root, environ={}))
            orchestrator.evo_log("task-1", "failed", "issue")
            orchestrator.evo_propose("task-1", "unvalidated", "prompt")

            with self.assertRaises(ExecutionError):
                orchestrator.evo_promote("unvalidated", approved=True)


class DoctorTests(unittest.TestCase):
    def test_doctor_reports_services_and_memory_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_temp_repo(tmpdir)
            fake_client = FakeHttpClient(
                responses=[
                    {},
                    {},
                ]
            )
            orchestrator = create_test_orchestrator(
                load_config(repo_root=repo_root, environ={"SKILL_MEMORY_EMBEDDING_PROVIDER": "disabled"}),
                http_client=fake_client,
            )
            result = orchestrator.doctor()

            self.assertIn("services", result)
            self.assertTrue(result["services"]["ollama"]["ok"])
            self.assertIn("modes", result)
            self.assertIn("research_only", result["modes"])
            self.assertIn("memory", result)

    def test_doctor_includes_control_plane_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = make_temp_repo(tmpdir)
            fake_client = FakeHttpClient(responses=[{}, {}])
            orchestrator = create_test_orchestrator(
                load_config(repo_root=repo_root, environ={"SKILL_MEMORY_EMBEDDING_PROVIDER": "disabled"}),
                http_client=fake_client,
            )
            result = orchestrator.doctor()
            self.assertIn("event_subscribers", result)
            self.assertIn("config_validation", result)
            self.assertIn("circuit_breakers", result)
            self.assertFalse(result["config_validation"]["has_errors"])

if __name__ == "__main__":
    unittest.main()
