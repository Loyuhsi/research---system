from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Mapping, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import requests  # type: ignore
except ModuleNotFoundError:
    requests = None

logger = logging.getLogger(__name__)


class HttpClientError(RuntimeError):
    pass


class JsonHttpClient:
    """Shared HTTP client with optional retry, backoff, and rate limiting."""

    def __init__(
        self,
        user_agent: str = "AutoResearch/1.0",
        max_retries: int = 0,
        retry_delay: float = 1.0,
        min_request_interval: float = 0.0,
        proxy_url: str = "",
    ) -> None:
        self.user_agent = user_agent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.min_request_interval = min_request_interval
        self.proxy_url = proxy_url
        self._last_request_time: float = 0.0
        self._requests_session = None
        if requests is not None and proxy_url:
            self._requests_session = requests.Session()
            self._requests_session.proxies = {"http": proxy_url, "https": proxy_url}
        self._urllib_opener = None
        if proxy_url and requests is None:
            proxy_handler = urllib_request.ProxyHandler({"http": proxy_url, "https": proxy_url})
            self._urllib_opener = urllib_request.build_opener(proxy_handler)

    def request_json(
        self,
        method: str,
        url: str,
        payload: Optional[Mapping[str, object]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: int = 30,
    ) -> Any:
        if self.min_request_interval > 0:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

        last_exc: Optional[Exception] = None
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    return self._do_request(method, url, payload, headers, timeout)
                except Exception as exc:
                    last_exc = exc
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.debug("Retry %d/%d after %.1fs: %s", attempt + 1, self.max_retries, delay, exc)
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        finally:
            self._last_request_time = time.monotonic()

    def _do_request(
        self,
        method: str,
        url: str,
        payload: Optional[Mapping[str, object]],
        headers: Optional[Mapping[str, str]],
        timeout: int,
    ) -> Any:
        merged_headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if payload is not None:
            merged_headers["Content-Type"] = "application/json"
        if headers:
            merged_headers.update(headers)

        # Log request but redact sensitive headers
        safe_headers = {
            k: ("***" if k.lower() == "authorization" else v)
            for k, v in merged_headers.items()
        }
        logger.debug("HTTP %s %s headers=%s", method.upper(), url, safe_headers)

        if requests is not None:
            requester = self._requests_session or requests
            response = requester.request(
                method=method.upper(),
                url=url,
                json=dict(payload) if payload is not None else None,
                headers=merged_headers,
                timeout=timeout,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:  # type: ignore[union-attr]
                body = response.text[:1000] if getattr(response, "text", "") else ""
                detail = f"HTTP {response.status_code}"
                if body:
                    detail = f"{detail}: {body}"
                raise HttpClientError(detail) from exc
            if not response.content:
                return {}
            return response.json()

        body: Optional[bytes] = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        req = urllib_request.Request(url=url, data=body, headers=merged_headers, method=method.upper())
        try:
            opener = self._urllib_opener or urllib_request
            with opener.open(req, timeout=timeout) as resp:
                content = resp.read()
        except urllib_error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise HttpClientError(f"HTTP {exc.code}: {response_body}") from exc
        except urllib_error.URLError as exc:
            raise HttpClientError(f"Network error: {exc.reason}") from exc

        if not content:
            return {}
        return json.loads(content.decode("utf-8"))
