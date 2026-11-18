"""Command-line entry point for crawlers.

python -m app.ingestion.cli rbi-master-directions --match "Non-Banking" --limit 5 --history
"""

import argparse
import asyncio
import json
from dataclasses import asdict

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import create_engine, create_session_factory
from app.ingestion.http import PoliteClient
from app.ingestion.pipeline import crawl_master_directions
from app.ingestion.raw_store import RawStore


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


def main() -> int:
    parser = argparse.ArgumentParser(prog="advancedrag-ingest")
    sub = parser.add_subparsers(dest="command", required=True)

    md = sub.add_parser("rbi-master-directions", help="crawl RBI Master Directions")
    md.add_argument("--match", help="case-insensitive filter on title or category")
    md.add_argument("--limit", type=int, help="max documents to fetch")
    md.add_argument("--history", action="store_true", help="also record past version dates")

    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    if args.command == "rbi-master-directions":
        return asyncio.run(_rbi_master_directions(args))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
