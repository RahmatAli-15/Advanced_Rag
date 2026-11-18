"""Per-client rate limit for the expensive endpoints (/ask, /ask/stream, /search).

Fixed one-minute windows counted in Redis (INCR + EXPIRE): shared by every API process,
two commands per request, and easy to reason about. Its known weakness, a client can
send up to 2x the limit across a window boundary, is acceptable for protecting a GPU
and an LLM quota; it is not a billing meter.

The client is identified by IP address. X-Forwarded-For is deliberately not trusted
here: behind a proxy, configure the proxy to set the real client address instead
(uvicorn --proxy-headers), or anyone could pick a new identity per request.

Fail open: if Redis is unavailable the request is allowed. The limiter protects
capacity; it must not turn a Redis blip into an outage.
"""

import logging
import time

from fastapi import HTTPException, Request

from app.core import metrics

log = logging.getLogger(__name__)

WINDOW_S = 60


async def rate_limit(request: Request) -> None:
    limit: int = getattr(request.app.state, "rate_limit_per_minute", 0)
    redis = getattr(request.app.state, "redis", None)
    if limit <= 0 or redis is None:
        return
    client = request.client.host if request.client else "unknown"
    now = time.time()
    window = int(now // WINDOW_S)
    key = f"rl:{client}:{window}"
    try:
        async with redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, WINDOW_S + 5)
            count, _ = await pipe.execute()
    except Exception as e:
        log.warning("rate limiter unavailable, allowing request: %s", e)
        return
    if count > limit:
        metrics.RATE_LIMITED.labels(request.url.path).inc()
        retry_after = max(1, int((window + 1) * WINDOW_S - now))
        raise HTTPException(
            429,
            f"Rate limit: {limit} requests per minute. Try again in {retry_after} s.",
            headers={"Retry-After": str(retry_after)},
        )
