"""Liveness and readiness probes.

/health/live  -> the process is up (never touches dependencies)
/health/ready -> dependencies are reachable; returns 503 if any check fails,
                 so a load balancer or Kubernetes stops sending traffic.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import text

router = APIRouter(prefix="/health", tags=["health"])

CHECK_TIMEOUT_S = 2.0


class ReadyResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, Literal["ok", "fail"]]


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


async def _check_postgres(request: Request) -> bool:
    async with request.app.state.engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return True


async def _check_redis(request: Request) -> bool:
    return bool(await request.app.state.redis.ping())


Check = Callable[[Request], Awaitable[bool]]


async def _run(check: Check, request: Request) -> Literal["ok", "fail"]:
    try:
        ok = await asyncio.wait_for(check(request), timeout=CHECK_TIMEOUT_S)
    except Exception:
        return "fail"
    return "ok" if ok else "fail"


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request, response: Response) -> ReadyResponse:
    pg, rd = await asyncio.gather(_run(_check_postgres, request), _run(_check_redis, request))
    checks = {"postgres": pg, "redis": rd}
    healthy = all(v == "ok" for v in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(status="ok" if healthy else "degraded", checks=checks)
