"""Health endpoint tests. Dependencies are faked so these run without Docker."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app


class _FakeConn:
    def __init__(self, fail: bool) -> None:
        self.fail = fail

    async def __aenter__(self) -> "_FakeConn":
        if self.fail:
            raise ConnectionError("db down")
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def execute(self, _stmt: Any) -> None:
        return None


class _FakeEngine:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def connect(self) -> _FakeConn:
        return _FakeConn(self.fail)


class _FakeRedis:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    async def ping(self) -> bool:
        if not self.ok:
            raise ConnectionError("redis down")
        return True


def _client(db_fail: bool = False, redis_ok: bool = True) -> TestClient:
    app = create_app()
    # Skip the real lifespan; inject fakes.
    app.router.lifespan_context = _noop_lifespan
    app.state.engine = _FakeEngine(db_fail)
    app.state.redis = _FakeRedis(redis_ok)
    return TestClient(app)


@asynccontextmanager
async def _noop_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield


def test_live() -> None:
    assert _client().get("/health/live").json() == {"status": "ok"}


def test_ready_ok() -> None:
    r = _client().get("/health/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "checks": {"postgres": "ok", "redis": "ok"}}


def test_ready_reports_failing_dependency() -> None:
    r = _client(db_fail=True).get("/health/ready")
    assert r.status_code == 503
    assert r.json()["checks"] == {"postgres": "fail", "redis": "ok"}
