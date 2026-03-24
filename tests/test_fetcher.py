"""Tests for FetcherService — session ID validation and WSL path conversion."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from auto_research.exceptions import ExecutionError
from auto_research.services.fetcher import FetcherService, _SAFE_SESSION_RE, _SAFE_SCRIPT_RE


class TestSafeSessionRegex:
    """Verify the session ID allowlist blocks path traversal."""

    @pytest.mark.parametrize("good_id", [
        "abc123",
        "session-2026-03",
        "my_session_v2",
        "A" * 128,
    ])
    def test_safe_ids_accepted(self, good_id):
        assert _SAFE_SESSION_RE.match(good_id)

    @pytest.mark.parametrize("bad_id", [
        "../etc/passwd",
        "../../secret",
        "session/../../root",
        "a" * 129,
        "",
        "session id with spaces",
        "session;rm -rf /",
        "session\x00null",
    ])
    def test_unsafe_ids_rejected(self, bad_id):
        assert not _SAFE_SESSION_RE.match(bad_id)


class TestRunWslWorkerSessionValidation:
    """_run_wsl_worker must reject unsafe session IDs returned by scripts."""

    def _make_service(self, tmp_path):
        config = MagicMock()
        config.repo_root = tmp_path
        config.wsl_distro = "Ubuntu-24.04"
        vault = MagicMock()
        return FetcherService(config, vault)

    @patch("auto_research.services.fetcher.subprocess.run")
    def test_valid_session_id_returned(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="some logs\nmy-session-123\n", stderr=""
        )
        svc = self._make_service(tmp_path)
        result = svc._run_wsl_worker("scripts/test.sh", [])
        assert result == "my-session-123"

    @patch("auto_research.services.fetcher.subprocess.run")
    def test_path_traversal_session_id_rejected(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="../../etc/passwd\n", stderr=""
        )
        svc = self._make_service(tmp_path)
        with pytest.raises(ExecutionError, match="unsafe session id"):
            svc._run_wsl_worker("scripts/test.sh", [])

    @patch("auto_research.services.fetcher.subprocess.run")
    def test_empty_session_id_rejected(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="\n", stderr=""
        )
        svc = self._make_service(tmp_path)
        with pytest.raises(ExecutionError, match="did not return a session id"):
            svc._run_wsl_worker("scripts/test.sh", [])


class TestScriptPathAllowlist:
    """Verify script path allowlist blocks injection."""

    @pytest.mark.parametrize("good_path", [
        "scripts/fetch-public.sh",
        "scripts/fetch-private-gh.sh",
        "scripts/test.sh",
        "scripts/my-worker_v2.sh",
    ])
    def test_safe_scripts_accepted(self, good_path):
        assert _SAFE_SCRIPT_RE.match(good_path)

    @pytest.mark.parametrize("bad_path", [
        "../scripts/fetch.sh",
        "scripts/../etc/passwd",
        "/usr/bin/evil.sh",
        "scripts/evil.sh; rm -rf /",
        "scripts/UPPER.sh",
        "other/fetch.sh",
        "scripts/fetch.py",
    ])
    def test_unsafe_scripts_rejected(self, bad_path):
        assert not _SAFE_SCRIPT_RE.match(bad_path)

    def test_run_wsl_worker_rejects_bad_script_path(self, tmp_path):
        config = MagicMock()
        config.repo_root = tmp_path
        config.wsl_distro = "Ubuntu-24.04"
        svc = FetcherService(config, MagicMock())
        with pytest.raises(ExecutionError, match="Script path rejected"):
            svc._run_wsl_worker("../malicious.sh", [])


class TestToWslPath:
    def test_converts_windows_path(self):
        p = Path("C:/Users/someone/project")
        result = FetcherService._to_wsl_path(p)
        assert result.startswith("/mnt/")
        assert "\\" not in result
