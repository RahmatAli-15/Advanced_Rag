"""Command-line entry point for ingestion.

python -m app.ingestion.cli rbi-master-directions --match "Non-Banking" --history
python -m app.ingestion.cli chunk [--rebuild]
python -m app.ingestion.cli embed
python -m app.ingestion.cli search "video KYC for NBFCs" --as-of 2026-03-01
"""

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import date

from app.chunking.pipeline import chunk_versions
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import create_engine, create_session_factory
from app.embeddings.factory import build_embedder, build_reranker
from app.embeddings.pipeline import embed_pending_chunks
from app.ingestion.http import PoliteClient
from app.ingestion.pipeline import crawl_master_directions
from app.ingestion.raw_store import RawStore
from app.retrieval.hybrid import HybridRetriever, SearchFilters


async def _rbi_master_directions(args: argparse.Namespace) -> int:
    s = get_settings()
    engine = create_engine(s.database_url)
    try:
        async with PoliteClient(s.crawler_user_agent, s.crawler_min_interval_s) as client:
            stats = await crawl_master_directions(
                client,
                create_session_factory(engine),
                RawStore(s.data_dir / "raw"),
                match=args.match,
                limit=args.limit,
                with_history=args.history,
            )
    finally:
        await engine.dispose()
    print(json.dumps(asdict(stats), indent=2))
    return 1 if stats.aborted else 0


async def _chunk(args: argparse.Namespace) -> int:
    s = get_settings()
    engine = create_engine(s.database_url)
    try:
        stats = await chunk_versions(
            create_session_factory(engine), RawStore(s.data_dir / "raw"), rebuild=args.rebuild
        )
    finally:
        await engine.dispose()
    print(json.dumps(asdict(stats), indent=2))
    return 1 if stats.failed else 0


async def _embed(_: argparse.Namespace) -> int:
    s = get_settings()
    engine = create_engine(s.database_url)
    try:
        stats = await embed_pending_chunks(create_session_factory(engine), build_embedder(s))
    finally:
        await engine.dispose()
    print(json.dumps(asdict(stats), indent=2))
    return 0


async def _search(args: argparse.Namespace) -> int:
    s = get_settings()
    engine = create_engine(s.database_url)
    retriever = HybridRetriever(
        create_session_factory(engine),
        build_embedder(s),
        None if args.no_rerank else build_reranker(s),
    )
    try:
        filters = SearchFilters(as_of=date.fromisoformat(args.as_of) if args.as_of else None)
        hits, trace = await retriever.search(args.query, k=args.k, filters=filters)
    finally:
        await engine.dispose()
    for i, h in enumerate(hits, 1):
        refs = ", ".join(h.para_refs[:3]) or "-"
        print(
            f"\n#{i}  {h.document_title[:90]}\n    para {refs}   "
            f"vec={h.vector_rank} kw={h.keyword_rank} rerank={h.rerank_score}"
        )
        print("    " + h.text[:240].replace("\n", " "))
    print(f"\nas_of={trace.as_of} timings_ms={trace.timings_ms}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="advancedrag-ingest")
    sub = parser.add_subparsers(dest="command", required=True)

    md = sub.add_parser("rbi-master-directions", help="crawl RBI Master Directions")
    md.add_argument("--match", help="case-insensitive filter on title or category")
    md.add_argument("--limit", type=int, help="max documents to fetch")
    md.add_argument("--history", action="store_true", help="also record past version dates")

    ch = sub.add_parser("chunk", help="chunk captured versions not yet chunked")
    ch.add_argument("--rebuild", action="store_true", help="re-chunk every captured version")

    sub.add_parser("embed", help="embed chunks that have no vector for the configured model")

    se = sub.add_parser("search", help="hybrid search from the terminal")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=5)
    se.add_argument("--as-of", help="YYYY-MM-DD; default today")
    se.add_argument("--no-rerank", action="store_true")

    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    commands = {
        "rbi-master-directions": _rbi_master_directions,
        "chunk": _chunk,
        "embed": _embed,
        "search": _search,
    }
    return asyncio.run(commands[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
