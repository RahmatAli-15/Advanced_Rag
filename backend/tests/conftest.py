"""Shared fixtures.

Integration tests run against a dedicated database (never the dev one), created and
migrated to head once per test session. Each test runs in a transaction that is
rolled back. If Postgres is not reachable, integration tests are skipped.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _test_database_url() -> str:
    if url := os.environ.get("TEST_DATABASE_URL"):
        return url
    dev = make_url(get_settings().database_url)
    return dev.set(database=f"{dev.database}_test").render_as_string(hide_password=False)


async def _ensure_database(url: str) -> None:
    u = make_url(url)
    conn = await asyncpg.connect(
        host=u.host, port=u.port, user=u.username, password=u.password, database="postgres"
    )
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", u.database)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{u.database}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = _test_database_url()
    try:
        asyncio.run(_ensure_database(url))
    except (OSError, asyncpg.PostgresError) as e:
        pytest.skip(f"Postgres not reachable: {e}")
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.attributes["database_url"] = url
    command.upgrade(cfg, "head")
    return url


@pytest.fixture
async def session_factory(
    test_database_url: str,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(test_database_url)
    conn = await engine.connect()
    trans = await conn.begin()
    try:
        yield async_sessionmaker(bind=conn, join_transaction_mode="create_savepoint")
    finally:
        await trans.rollback()
        await conn.close()
        await engine.dispose()


@pytest.fixture
async def session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as s:
        yield s
