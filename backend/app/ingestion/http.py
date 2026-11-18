"""A polite HTTP client for crawling public regulator websites.

- obeys robots.txt (RFC 9309), cached per host
- enforces a minimum delay between requests to the same host
- retries transient failures with exponential backoff + jitter, honouring Retry-After
- detects bot-challenge pages and stops instead of trying to get around them
"""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import httpx

from app.ingestion.robots import RobotsPolicy, policy_from_response

log = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
CHALLENGE_MARKERS = (
    b"testing whether you are a human visitor",
    b"please enable javascript to view the page content",
)


class CrawlError(Exception):
    """Base class for crawl failures."""


class DisallowedByRobots(CrawlError):
    pass


class BlockedByBotProtection(CrawlError):
    """The site served an anti-bot challenge. We do not attempt to bypass it."""


class FetchFailed(CrawlError):
    pass


@dataclass(frozen=True)
class FetchResult:
    url: str
    status: int
    content_type: str
    body: bytes


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def is_bot_challenge(body: bytes, content_type: str, expected_type: str | None) -> bool:
    if expected_type and expected_type not in content_type and "text/html" in content_type:
        return True  # asked for a PDF, got an HTML page
    lowered = body[:20_000].lower()
    return any(marker in lowered for marker in CHALLENGE_MARKERS)


class PoliteClient:
    def __init__(
        self,
        user_agent: str,
        min_interval_s: float = 3.0,
        max_retries: int = 3,
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            timeout=timeout_s,
            follow_redirects=True,
            transport=transport,
        )
        self._robots: dict[str, RobotsPolicy] = {}
        self._last_request: dict[str, float] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._sleep = sleep or asyncio.sleep

    async def __aenter__(self) -> "PoliteClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def _throttle(self, host: str) -> None:
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            policy = self._robots.get(host)
            delay = (policy.crawl_delay(self.user_agent) if policy else None) or 0.0
            interval = max(self.min_interval_s, delay)
            wait = self._last_request.get(host, 0.0) + interval - time.monotonic()
            if wait > 0:
                await self._sleep(wait)
            self._last_request[host] = time.monotonic()

    async def _robots_for(self, url: str) -> RobotsPolicy:
        parts = urlsplit(url)
        host = parts.netloc
        if host not in self._robots:
            robots_url = f"{parts.scheme}://{host}/robots.txt"
            await self._throttle(host)
            try:
                r = await self._client.get(robots_url)
                policy = policy_from_response(r.status_code, r.text)
            except httpx.HTTPError:
                policy = policy_from_response(None, "")
            log.info("robots policy for %s: %s", host, policy.state)
            self._robots[host] = policy
        return self._robots[host]

    async def get(self, url: str, expected_type: str | None = None) -> FetchResult:
        policy = await self._robots_for(url)
        if not policy.can_fetch(self.user_agent, url):
            raise DisallowedByRobots(url)

        host = urlsplit(url).netloc
        for attempt in range(self.max_retries + 1):
            await self._throttle(host)
            try:
                r = await self._client.get(url)
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise FetchFailed(f"{url}: {e}") from e
                await self._sleep(self._backoff(attempt))
                continue

            if r.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                wait = _retry_after_seconds(r.headers.get("retry-after")) or self._backoff(attempt)
                log.warning("HTTP %s for %s, retrying in %.1fs", r.status_code, url, wait)
                await self._sleep(wait)
                continue
            if r.status_code != 200:
                raise FetchFailed(f"{url}: HTTP {r.status_code}")

            content_type = r.headers.get("content-type", "")
            if is_bot_challenge(r.content, content_type, expected_type):
                raise BlockedByBotProtection(url)
            return FetchResult(url, r.status_code, content_type, r.content)

        raise FetchFailed(url)  # pragma: no cover - loop always returns or raises

    @staticmethod
    def _backoff(attempt: int) -> float:
        return float(min(60.0, 2**attempt * 2.0) + random.uniform(0, 1))  # noqa: S311
