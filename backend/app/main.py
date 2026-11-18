"""FastAPI application factory."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from app.answer.service import AnswerService
from app.api import ask, health, search
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import create_engine, create_session_factory
from app.embeddings.factory import build_embedder, build_reranker
from app.llm.factory import build_gateway
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
        app.state.retriever = HybridRetriever(app.state.session_factory, embedder, reranker)
        llm_provider = None
        if settings.groq_api_key is not None:
            gateway, llm_provider = build_gateway(settings)
            app.state.answer_service = AnswerService(
                app.state.retriever, gateway, app.state.session_factory
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
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    return app


app = create_app()
