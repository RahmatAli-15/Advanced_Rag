"""FastAPI application factory."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from app.answer.cache import AnswerCache, CorpusFingerprint
from app.answer.service import AnswerService
from app.api import ask, changes, corpus, health, metrics, search
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import create_engine, create_session_factory
from app.embeddings.factory import build_embedder, build_reranker
from app.llm.budget import DailyTokenBudget
from app.llm.factory import build_gateway
from app.retrieval.gate import ConcurrencyGate, Overloaded
from app.retrieval.hybrid import HybridRetriever


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Shared clients are created once and closed cleanly on shutdown.
        app.state.engine = create_engine(settings.database_url)
        app.state.session_factory = create_session_factory(app.state.engine)
        app.state.redis = Redis.from_url(settings.redis_url)
        embedder, reranker = build_embedder(settings), build_reranker(settings)
        gate = ConcurrencyGate(settings.retrieval_max_concurrent, settings.retrieval_max_waiting)
        app.state.retriever = HybridRetriever(
            app.state.session_factory, embedder, reranker, gate=gate
        )
        llm_provider = None
        if settings.groq_api_keys:
            budget = DailyTokenBudget(app.state.redis, settings.llm_daily_token_budget)
            gateway, llm_provider = build_gateway(settings, budget=budget)
            cache = AnswerCache(app.state.redis, CorpusFingerprint(app.state.session_factory))
            app.state.answer_service = AnswerService(
                app.state.retriever, gateway, app.state.session_factory, cache=cache
            )

        # Loading the models takes ~20 s. Do it in the background so the process starts
        # (and /health/live answers) immediately, while /health/ready stays 503 until the
        # models are in memory, so no user request ever pays the cold start.
        def warm_up() -> None:
            embedder.embed_query("warm up")
            reranker.score("warm up", ["warm up"])

        app.state.models_warmup = asyncio.create_task(asyncio.to_thread(warm_up))
        try:
            yield
        finally:
            app.state.models_warmup.cancel()
            if llm_provider is not None:
                await llm_provider.aclose()
            await app.state.redis.aclose()
            await app.state.engine.dispose()

    app = FastAPI(title="AdvancedRag", version="0.1.0", lifespan=lifespan)
    app.state.rate_limit_per_minute = settings.rate_limit_per_minute

    @app.exception_handler(Overloaded)
    async def overloaded(_: Request, e: Overloaded) -> JSONResponse:
        return JSONResponse(
            {"detail": "The service is busy right now. Please try again in a few seconds."},
            status_code=503,
            headers={"Retry-After": str(e.retry_after_s)},
        )

    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    app.include_router(corpus.router)
    app.include_router(changes.router)
    app.include_router(metrics.router)
    return app


app = create_app()
