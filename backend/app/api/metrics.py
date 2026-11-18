"""Prometheus scrape endpoint.

Most metrics are pushed into the registry as things happen (app.core.metrics). The
queue's state lives in Redis, shared with the worker process, so it is read here, at
scrape time: the API reports the queue without the worker exposing its own endpoint.
"""

import logging

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core import metrics
from app.queue.core import Queue

log = logging.getLogger(__name__)

router = APIRouter(tags=["ops"])


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(request: Request) -> Response:
    try:
        for state, n in (await Queue(request.app.state.redis).depth()).items():
            metrics.QUEUE_DEPTH.labels(state).set(n)
    except Exception as e:  # a Redis hiccup must not hide every other metric
        log.warning("could not read queue depth: %s", e)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
