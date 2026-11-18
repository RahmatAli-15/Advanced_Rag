"""Command-line entry point for ingestion.

python -m app.ingestion.cli rbi-master-directions --match "Non-Banking" --history
python -m app.ingestion.cli chunk [--rebuild]
python -m app.ingestion.cli embed
python -m app.ingestion.cli search "video KYC for NBFCs" --as-of 2026-03-01
"""

import argparse
import asyncio
import json
import random
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

from app.chunking.pipeline import chunk_versions
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import create_engine, create_session_factory
from app.embeddings.factory import build_embedder, build_reranker
from app.embeddings.pipeline import embed_pending_chunks
from app.eval import dataset
from app.eval.generate import generate
from app.eval.run import evaluate, write_report
from app.ingestion.http import PoliteClient
from app.ingestion.pipeline import crawl_master_directions
from app.ingestion.raw_store import RawStore
from app.llm.factory import build_gateway
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


async def _eval_generate(args: argparse.Namespace) -> int:
    s = get_settings()
    engine = create_engine(s.database_url)
    gateway, provider = build_gateway(s)
    out = Path(args.out) if args.out else s.eval_dir / "datasets" / "retrieval_v1.jsonl"
    try:
        items, stats = await generate(
            create_session_factory(engine),
            gateway,
            model_label=s.llm_models[0],
            n=args.n,
            per_document=args.per_document,
            seed=args.seed,
        )
    finally:
        await provider.aclose()
        await engine.dispose()
    dataset.save(out, items)
    print(
        json.dumps(
            {**asdict(stats), "out": str(out), "llm": asdict(gateway.stats)}, indent=2, default=str
        )
    )
    return 0


async def _eval_retrieval(args: argparse.Namespace) -> int:
    s = get_settings()
    path = Path(args.dataset) if args.dataset else s.eval_dir / "datasets" / "retrieval_v2.jsonl"
    items = dataset.load(path)
    if args.language:
        items = [i for i in items if i.language == args.language]
    engine = create_engine(s.database_url)
    factory = create_session_factory(engine)
    embedder = build_embedder(s)
    plain = HybridRetriever(factory, embedder, None)
    reranking = HybridRetriever(factory, embedder, build_reranker(s))
    try:
        await reranking.search("warm up")  # load models outside the timed runs
        results = await evaluate(items, plain, reranking, factory)
    finally:
        await engine.dispose()
    meta: dict[str, object] = {
        "dataset": f"{path.name} ({len(items)} questions)",
        "embedding_model": embedder.name,
        "reranker_model": s.reranker_model,
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }
    md, _ = write_report(results, meta, s.eval_dir / "reports")
    print(md.read_text(encoding="utf-8"))
    return 0


async def _eval_answers(args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from app.answer.service import STREAM_PROMPT_VERSION, AnswerService
    from app.db.models import Chunk, Document, DocumentVersion
    from app.eval import faithfulness as fa

    s = get_settings()
    items = dataset.load(s.eval_dir / "datasets" / "retrieval_v2.jsonl")
    if args.language:  # a focused run, e.g. to check one language fix cheaply
        items = [i for i in items if i.language == args.language]
    gold = fa.sample(items, args.n, args.seed)
    unanswerable = fa.load_unanswerable(s.eval_dir / "datasets" / "unanswerable_v1.jsonl")
    if args.no_unanswerable or args.language:
        unanswerable = []
    engine = create_engine(s.database_url)
    factory = create_session_factory(engine)
    retriever = HybridRetriever(factory, build_embedder(s), build_reranker(s))
    # Batch run: wait out rate limits instead of falling back, so every answer comes
    # from the model being evaluated (a fallback model would skew the scores).
    answer_gw, answer_provider = build_gateway(
        s, s.llm_models[:1], max_retries=10, max_backoff_s=900
    )
    judge_gw, judge_provider = build_gateway(s, s.judge_models, max_retries=10, max_backoff_s=900)
    service = AnswerService(retriever, answer_gw, factory)
    rng = random.Random(args.seed)  # noqa: S311 - reproducible planting, not security
    results: list[fa.ItemResult] = []
    try:
        async with factory() as session:
            rows = await session.execute(
                select(Chunk.id, Document.source_key)
                .join(DocumentVersion, Chunk.version_id == DocumentVersion.id)
                .join(Document, DocumentVersion.document_id == Document.id)
            )
            matcher = fa.GoldMatcher({str(cid): key for cid, key in rows})
        await retriever.search("warm up")
        total = len(gold) + len(unanswerable)
        for i, item in enumerate(gold, start=1):
            r = await fa.run_answerable(service, judge_gw, item, matcher, rng)
            results.append(r)
            verdicts = [c.verdict for c in r.claims]
            print(f"[{i}/{total}] {item.id} {r.status} {verdicts}", flush=True)
        for j, u in enumerate(unanswerable, start=len(gold) + 1):
            r = await fa.run_unanswerable(service, u)
            results.append(r)
            print(f"[{j}/{total}] {u.id} {r.status}", flush=True)
    finally:
        await answer_provider.aclose()
        await judge_provider.aclose()
        await engine.dispose()

    summary = fa.summarise(results)
    meta: dict[str, object] = {
        "answerable": f"{len(gold)} questions sampled from retrieval_v2.jsonl "
        f"(seed {args.seed}{', ' + args.language + ' only' if args.language else ''})",
        "unanswerable": f"{len(unanswerable)} questions from unanswerable_v1.jsonl",
        "answer_model": f"{s.llm_models[0]} (no fallback during evaluation)",
        "answer_prompt": f"{STREAM_PROMPT_VERSION} (the one the UI uses)",
        "judge_models": ", ".join(s.judge_models),
        "judge_prompt": fa.JUDGE_PROMPT_VERSION,
        "gateway_retries": answer_gw.stats.retries + judge_gw.stats.retries,
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }
    md, _ = fa.write_report(summary, results, meta, s.eval_dir / "reports")
    print(md.read_text(encoding="utf-8"))
    return 0


async def _bench_retrieval(args: argparse.Namespace) -> int:
    from app.eval.bench import BenchConfig, bench, render

    s = get_settings()
    items = dataset.load(s.eval_dir / "datasets" / "retrieval_v2.jsonl")
    configs = [
        BenchConfig("current", candidates=40, rerank_top=30),
        BenchConfig("rerank 20", candidates=40, rerank_top=20),
        BenchConfig("rerank 15", candidates=40, rerank_top=15),
        BenchConfig("rerank 10", candidates=40, rerank_top=10),
        BenchConfig("no rerank", candidates=40, rerank_top=0),
    ]
    engine = create_engine(s.database_url)
    factory = create_session_factory(engine)
    embedder, reranker = build_embedder(s), build_reranker(s)
    try:
        results = []
        for cfg in configs:
            rr = reranker if cfg.rerank_top else None
            [res] = await bench(items, factory, {"embedder": embedder, "reranker": rr}, [cfg])
            results.append(res)
            print(f"{cfg.name}: total p50 {res.p50.get('total')} ms, hit@5 {res.hit5}", flush=True)
    finally:
        await engine.dispose()
    table = render(results)
    out = s.eval_dir / "reports" / f"bench-{datetime.now(UTC).strftime('%Y%m%d-%H%M')}.md"
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    header = [
        "# Retrieval latency benchmark",
        "",
        f"- questions: {len(items)}",
        f"- device: {s.ml_device}",
        f"- generated: {generated}",
        "",
    ]
    out.write_text("\n".join(header) + table, encoding="utf-8")
    print(table)
    return 0


async def _changes_from_archive(args: argparse.Namespace) -> int:
    from app.changes.archive import backfill_from_archive
    from app.db.models import ChangeEvent

    s = get_settings()
    engine = create_engine(s.database_url)
    try:
        event_id = await backfill_from_archive(
            create_session_factory(engine),
            RawStore(s.data_dir / "raw"),
            args.source_key,
            args.timestamp,
            user_agent=s.crawler_user_agent,
        )
        async with create_session_factory(engine)() as session:
            event = await session.get_one(ChangeEvent, event_id)
            print(f"change event {event_id}: {event.status}, {event.diff.get('counts')}")
    finally:
        await engine.dispose()
    return 0


async def _analyse_impact(args: argparse.Namespace) -> int:
    from app.changes.impact import analyse_change, analyse_pending

    s = get_settings()
    engine = create_engine(s.database_url)
    factory = create_session_factory(engine)
    store = RawStore(s.data_dir / "raw")
    retriever = (
        None if args.no_related else HybridRetriever(factory, build_embedder(s), build_reranker(s))
    )
    gateway, provider = build_gateway(s, max_retries=6, max_backoff_s=120)
    try:
        if args.event:
            impact = await analyse_change(uuid.UUID(args.event), factory, store, gateway, retriever)
            print(json.dumps(impact, ensure_ascii=False, indent=2))
        else:
            print(f"analysed {await analyse_pending(factory, store, gateway, retriever)} changes")
    finally:
        await provider.aclose()
        await engine.dispose()
    return 0


async def _load_test(args: argparse.Namespace) -> int:
    from app.eval.load import render, run

    s = get_settings()
    questions = [i.question for i in dataset.load(s.eval_dir / "datasets" / "retrieval_v2.jsonl")]
    results = []
    for endpoint in args.endpoints.split(","):
        if endpoint == "ask-cached":  # warm the cache so every measured request is a hit
            await run(args.url, "ask-cached", questions[: args.cached_pool], 2, args.cached_pool)
        pool = questions[: args.cached_pool] if endpoint == "ask-cached" else questions
        for c in (int(x) for x in args.concurrency.split(",")):
            res = await run(args.url, endpoint, pool, c, args.requests)
            results.append(res)
            print(
                f"{endpoint} x{c}: {res.throughput:.2f} req/s, p95 {res.pct(95):.0f} ms, "
                f"statuses {res.statuses}, errors {res.errors}",
                flush=True,
            )
    table = render(results)
    stamp = datetime.now(UTC)
    out = s.eval_dir / "reports" / f"load-{stamp.strftime('%Y%m%d-%H%M')}.md"
    header = [
        "# Load test",
        "",
        f"- target: {args.url}",
        f"- requests per run: {args.requests}; questions from retrieval_v2.jsonl",
        f"- generated: {stamp.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]
    out.write_text("\n".join(header) + table, encoding="utf-8")
    print(table)
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

    eg = sub.add_parser("eval-generate", help="generate a synthetic retrieval test set")
    eg.add_argument("-n", type=int, default=60, help="clauses to sample")
    eg.add_argument("--per-document", type=int, default=2)
    eg.add_argument("--seed", type=int, default=42)
    eg.add_argument("--out")

    er = sub.add_parser("eval-retrieval", help="score retrieval configs on a test set")
    er.add_argument("--dataset")
    er.add_argument("--language", choices=["en", "hinglish", "hi"])

    ea = sub.add_parser("eval-answers", help="judge answer faithfulness and abstention")
    ea.add_argument("-n", type=int, default=30, help="answerable questions to sample")
    ea.add_argument("--seed", type=int, default=7)
    ea.add_argument("--no-unanswerable", action="store_true")
    ea.add_argument("--language", choices=["en", "hinglish", "hi"], help="only this language")

    sub.add_parser("bench-retrieval", help="per-stage latency vs quality for retrieval settings")

    ca = sub.add_parser(
        "changes-from-archive", help="diff an Internet Archive snapshot against current text"
    )
    ca.add_argument("source_key", help="RBI document id, e.g. 12957")
    ca.add_argument("timestamp", help="Wayback snapshot timestamp, YYYYMMDDhhmmss")

    ai = sub.add_parser("analyse-impact", help="run the change-impact agent (LangGraph)")
    ai.add_argument("--event", help="one change event id (default: every unanalysed one)")
    ai.add_argument("--no-related", action="store_true", help="skip the related-clause search")

    lt = sub.add_parser("load-test", help="throughput and latency of a running API under load")
    lt.add_argument("--url", default="http://localhost:8000")
    lt.add_argument("--endpoints", default="search,ask-cached", help="search,ask-cached")
    lt.add_argument("--concurrency", default="1,2,4,8", help="simulated users per run")
    lt.add_argument("--requests", type=int, default=48, help="requests per run")
    lt.add_argument("--cached-pool", type=int, default=12, help="questions reused for ask-cached")

    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    commands = {
        "rbi-master-directions": _rbi_master_directions,
        "chunk": _chunk,
        "embed": _embed,
        "search": _search,
        "eval-generate": _eval_generate,
        "eval-retrieval": _eval_retrieval,
        "eval-answers": _eval_answers,
        "bench-retrieval": _bench_retrieval,
        "load-test": _load_test,
        "changes-from-archive": _changes_from_archive,
        "analyse-impact": _analyse_impact,
    }
    return asyncio.run(commands[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
