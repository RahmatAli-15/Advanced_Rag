from collections.abc import Callable

import httpx
import pytest

from app.ingestion.http import (
    BlockedByBotProtection,
    DisallowedByRobots,
    FetchFailed,
    PoliteClient,
)

Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def _client(handler: Handler, rec: Recorder, **kw: float) -> PoliteClient:
    return PoliteClient(
        "TestBot/1.0",
        transport=httpx.MockTransport(handler),
        sleep=rec.sleep,
        **kw,  # type: ignore[arg-type]
    )


def _robots(allow: bool = True) -> httpx.Response:
    return httpx.Response(
        200, text="User-agent: *\n" + ("Allow: /\n" if allow else "Disallow: /\n")
    )


async def test_retries_transient_errors_honouring_retry_after() -> None:
    calls = {"page": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/robots.txt":
            return _robots()
        calls["page"] += 1
        if calls["page"] == 1:
            return httpx.Response(503, headers={"retry-after": "5"})
        return httpx.Response(200, text="ok", headers={"content-type": "text/html"})

    rec = Recorder()
    async with _client(handler, rec, min_interval_s=0) as c:
        r = await c.get("https://example.org/page")
    assert r.body == b"ok"
    assert calls["page"] == 2
    assert 5.0 in rec.sleeps


async def test_gives_up_after_max_retries() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _robots() if req.url.path == "/robots.txt" else httpx.Response(500)

    async with _client(handler, Recorder(), min_interval_s=0, max_retries=2) as c:
        with pytest.raises(FetchFailed):
            await c.get("https://example.org/page")


async def test_does_not_retry_client_errors() -> None:
    calls = {"page": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/robots.txt":
            return _robots()
        calls["page"] += 1
        return httpx.Response(404)

    async with _client(handler, Recorder(), min_interval_s=0) as c:
        with pytest.raises(FetchFailed):
            await c.get("https://example.org/missing")
    assert calls["page"] == 1


async def test_respects_robots_disallow() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/robots.txt":
            return _robots(allow=False)
        raise AssertionError("must not fetch a disallowed page")

    async with _client(handler, Recorder()) as c:
        with pytest.raises(DisallowedByRobots):
            await c.get("https://example.org/page")


@pytest.mark.parametrize(
    ("body", "content_type", "expected_type"),
    [
        (
            b"<html>This question is for testing whether you are a human visitor</html>",
            "text/html",
            None,
        ),
        (b"<!DOCTYPE html><html>...</html>", "text/html", "application/pdf"),
    ],
)
async def test_detects_bot_challenge(
    body: bytes, content_type: str, expected_type: str | None
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/robots.txt":
            return _robots()
        return httpx.Response(200, content=body, headers={"content-type": content_type})

    async with _client(handler, Recorder(), min_interval_s=0) as c:
        with pytest.raises(BlockedByBotProtection):
            await c.get("https://example.org/doc", expected_type=expected_type)


async def test_throttles_requests_to_the_same_host() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _robots() if req.url.path == "/robots.txt" else httpx.Response(200, text="ok")

    rec = Recorder()
    async with _client(handler, rec, min_interval_s=3.0) as c:
        await c.get("https://example.org/a")
        await c.get("https://example.org/b")
    # robots.txt, /a and /b: the second and third requests must each wait
    assert len([s for s in rec.sleeps if s > 2.5]) >= 2
