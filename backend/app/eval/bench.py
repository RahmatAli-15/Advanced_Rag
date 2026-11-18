"""Retrieval latency and quality per configuration: where the time goes, and what each
speed-up costs in accuracy.

    python -m app.ingestion.cli bench-retrieval

For each configuration it runs every test question and reports per-stage p50/p95
(embed, vector search, keyword search, load, rerank, total) next to hit@1, hit@5 and
MRR@10, so a faster setting is only adopted if quality holds.
"""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.eval.dataset import GoldItem
from app.eval.metrics import RetrievedRef, first_match_rank
from app.eval.run import _source_keys
from app.retrieval.hybrid import HybridRetriever, SearchFilters

STAGES = ("embed_query", "vector_search", "retrieve", "load", "rerank", "total")


@dataclass
class BenchConfig:
    name: str
    candidates: int = 40
    rerank_top: int = 30
    ef_search: int = 100


@dataclass
class BenchResult:
    config: BenchConfig
    n: int
    hit1: float
    hit5: float
    mrr10: float
    p50: dict[str, float] = field(default_factory=dict)
    p95: dict[str, float] = field(default_factory=dict)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1)


async def bench(
    items: Sequence[GoldItem],
    factory: async_sessionmaker[AsyncSession],
    retriever_args: dict[str, object],
    configs: Sequence[BenchConfig],
    warmup: int = 5,
) -> list[BenchResult]:
    keys = await _source_keys(factory)
    results = []
    for cfg in configs:
        r = HybridRetriever(
            factory,
            candidates=cfg.candidates,
            rerank_top=cfg.rerank_top,
            ef_search=cfg.ef_search,
            **retriever_args,  # type: ignore[arg-type]
        )
        for item in items[:warmup]:  # CUDA kernels, connection pool
            await r.search(item.question, k=10, filters=SearchFilters(as_of=item.as_of))
        stage_times: dict[str, list[float]] = {s: [] for s in STAGES}
        ranks: list[int | None] = []
        for item in items:
            hits, trace = await r.search(
                item.question, k=10, filters=SearchFilters(as_of=item.as_of)
            )
            for s in STAGES:
                if s in trace.timings_ms:
                    stage_times[s].append(trace.timings_ms[s])
            refs = [RetrievedRef(keys[str(h.document_id)], h.para_refs, h.text) for h in hits]
            ranks.append(first_match_rank(refs, item))
        n = len(ranks)
        results.append(
            BenchResult(
                config=cfg,
                n=n,
                hit1=round(sum(r is not None and r <= 1 for r in ranks) / n, 3),
                hit5=round(sum(r is not None and r <= 5 for r in ranks) / n, 3),
                mrr10=round(sum(1 / r for r in ranks if r is not None) / n, 3),
                p50={s: round(statistics.median(v), 1) for s, v in stage_times.items() if v},
                p95={s: _pct(v, 0.95) for s, v in stage_times.items() if v},
            )
        )
    return results


def render(results: Sequence[BenchResult]) -> str:
    lines = [
        "| Config | cand | rerank | ef | hit@1 | hit@5 | MRR@10 | embed | vector | "
        "keyword+vector | load | rerank | **total p50** | total p95 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c, p = r.config, r.p50
        lines.append(
            f"| {c.name} | {c.candidates} | {c.rerank_top} | {c.ef_search} | {r.hit1} | "
            f"{r.hit5} | {r.mrr10} | {p.get('embed_query', '-')} | "
            f"{p.get('vector_search', '-')} | {p.get('retrieve', '-')} | {p.get('load', '-')} | "
            f"{p.get('rerank', '-')} | **{p.get('total', '-')}** | {r.p95.get('total', '-')} |"
        )
    return "\n".join(lines) + "\n"
