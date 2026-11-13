"""Polite HTTP client: identifies itself, rate-limits per host, caches raw responses on disk."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .config import get_settings
from .logging_setup import log

logger = logging.getLogger(__name__)


class SourceUnavailable(RuntimeError):
    """Raised when an upstream provider cannot be reached or refuses the request."""


class PoliteClient:
    def __init__(self, cache_dir: Path | None = None, min_interval_s: float | None = None):
        s = get_settings()
        self.cache_dir = cache_dir or (s.data_dir / "raw")
        self.min_interval = s.http_min_interval_s if min_interval_s is None else min_interval_s
        self.timeout = s.http_timeout_s
        self.session = requests.Session()
        self.session.headers["User-Agent"] = s.http_user_agent
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def _wait(self, host: str) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last.get(host, 0.0)
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last[host] = time.monotonic()

    def _cache_path(self, namespace: str, key: str, suffix: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:24]
        p = self.cache_dir / namespace / f"{h}{suffix}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def get(self, url: str, params: dict | None = None, *, namespace: str, suffix: str = ".txt",
            use_cache: bool = True, retries: int = 3) -> str:
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        path = self._cache_path(namespace, key, suffix)
        if use_cache and path.exists() and path.stat().st_size > 0:
            return path.read_text(encoding="utf-8")
        host = urlparse(url).netloc
        last_exc: Exception | None = None
        for attempt in range(retries):
            self._wait(host)
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise SourceUnavailable(f"{host} returned HTTP {r.status_code}")
                if r.status_code >= 400:
                    raise SourceUnavailable(f"{host} returned HTTP {r.status_code} for {url}")
                text = r.content.decode("utf-8-sig")
                path.write_text(text, encoding="utf-8")
                log(logger, "http_fetch", url=r.url, status=r.status_code, bytes=len(r.content))
                return text
            except (requests.RequestException, SourceUnavailable) as e:
                last_exc = e
                log(logger, "http_retry", level=logging.WARNING, url=url, attempt=attempt + 1, error=str(e))
                time.sleep(2 ** attempt)
        raise SourceUnavailable(str(last_exc))

    def get_json(self, url: str, params: dict | None = None, *, namespace: str, use_cache: bool = True) -> dict:
        return json.loads(self.get(url, params, namespace=namespace, suffix=".json", use_cache=use_cache))
