"""Background worker: daily RBI refresh as a chain of queued jobs.

    python -m app.worker                      run the worker (Ctrl+C to stop)
    python -m app.worker enqueue refresh_rbi  queue a job now
    python -m app.worker dead                 list dead-lettered jobs

Every day at 06:30 IST the worker queues "refresh_rbi" with the job id
"refresh-rbi:<date>", so it runs once a day however many workers are running, and
still runs once if the worker starts later in the day. The chain:

    refresh_rbi ──▶ index_pending ──▶ build_changes
    crawl + write     chunk + embed      paragraph diffs for
    change events     anything new       the change events

Each stage is idempotent (skips work already done), so retries and the queue's
at-least-once delivery are safe. A crawl blocked by bot protection is NOT retried
(ADR 0003: we back off, we do not push through); it goes to the dead-letter queue.
"""

import asyncio
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, time, timedelta, timezone

from redis.asyncio import Redis

from app.changes.impact import analyse_pending
from app.changes.service import process_change_events
from app.chunking.pipeline import chunk_versions
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import create_engine, create_session_factory
from app.embeddings.factory import build_embedder, build_reranker
from app.embeddings.pipeline import embed_pending_chunks
from app.ingestion.http import PoliteClient
from app.ingestion.pipeline import crawl_master_directions
from app.ingestion.raw_store import RawStore
from app.llm.factory import build_gateway
from app.queue.core import Handler, PermanentError, Queue, Worker
from app.retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))  # India has no DST: a fixed offset is exact
DAILY_AT = time(6, 30)
DEFAULT_MATCH = "Non-Banking"


async def schedule_daily(queue: Queue, now: datetime | None = None) -> bool:
    """Queue today's refresh once it is past DAILY_AT (IST). Safe to call often."""
    now = (now or datetime.now(IST)).astimezone(IST)
    if now.time() < DAILY_AT:
        return False
    return await queue.enqueue(
        "refresh_rbi", {"match": DEFAULT_MATCH}, job_id=f"refresh-rbi:{now.date().isoformat()}"
    )


def build_handlers(settings: Settings) -> dict[str, Handler]:
    engine = create_engine(settings.database_url)
    sf = create_session_factory(engine)
    store = RawStore(settings.data_dir / "raw")
    embedder = build_embedder(settings)  # loads the model on first use only

    async def refresh_rbi(payload: dict[str, object], q: Queue) -> None:
        match = str(payload.get("match") or DEFAULT_MATCH)
        async with PoliteClient(
            settings.crawler_user_agent, settings.crawler_min_interval_s
        ) as client:
            stats = await crawl_master_directions(client, sf, store, match=match, with_history=True)
        log.info("crawl stats: %s", json.dumps(asdict(stats)))
        if stats.aborted:
            raise PermanentError("crawl aborted (bot protection or site error); not retrying")
        await q.enqueue("index_pending")

    async def index_pending(_: dict[str, object], q: Queue) -> None:
        chunked = await chunk_versions(sf, store)
        embedded = await embed_pending_chunks(sf, embedder)
        log.info("indexed: %s / %s", asdict(chunked), asdict(embedded))
        await q.enqueue("build_changes")

    async def build_changes(_: dict[str, object], q: Queue) -> None:
        n = await process_change_events(sf, store)
        log.info("change events processed: %d", n)
        await q.enqueue("analyse_impact")

    async def analyse_impact(_: dict[str, object], __: Queue) -> None:
        # The change-impact agent (ADR 0014). Patient retries: a batch job can wait out
        # rate limits, a user request should not.
        if not settings.groq_api_keys:
            log.warning("no LLM key configured: skipping change-impact analysis")
            return
        gateway, provider = build_gateway(settings, max_retries=6, max_backoff_s=120)
        retriever = HybridRetriever(sf, embedder, build_reranker(settings))
        try:
            n = await analyse_pending(sf, store, gateway, retriever)
        finally:
            await provider.aclose()
        log.info("changes analysed: %d", n)

    return {
        "refresh_rbi": refresh_rbi,
        "index_pending": index_pending,
        "build_changes": build_changes,
        "analyse_impact": analyse_impact,
    }


async def main(argv: list[str]) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    redis = Redis.from_url(settings.redis_url)
    queue = Queue(redis)
    try:
        if argv[:1] == ["enqueue"] and len(argv) == 2:
            queued = await queue.enqueue(argv[1])
            print("queued" if queued else "already queued")
            return 0
        if argv[:1] == ["dead"]:
            for job in await queue.dead_letters():
                print(json.dumps(job, ensure_ascii=False))
            return 0
        if argv:
            print(__doc__)
            return 2
        worker = Worker(queue, build_handlers(settings))
        stop = asyncio.Event()

        async def tick() -> None:
            await schedule_daily(queue)

        log.info("worker started; daily refresh at %s IST", DAILY_AT.strftime("%H:%M"))
        await worker.run_forever(stop, tick)
        return 0
    finally:
        await redis.aclose()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main(sys.argv[1:])))
    except KeyboardInterrupt:
        # An interrupted job is not acked; it will be reclaimed on the next start.
        print("worker stopped")
