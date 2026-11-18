"""Retrieval metrics.

A hit matches a gold item when it comes from the same document AND either shares a
paragraph ref with the gold clause or contains the gold text snippet. (Re-chunking can
move a clause's boundary; matching on refs + text keeps labels valid.)

- hit@k  (a.k.a. recall@k with one relevant clause): gold found in the top k
- MRR@10: mean of 1/rank of the first match (0 if not in the top 10)
- doc@k: the right *document* in the top k — separates "wrong clause, right
  regulation" from a complete miss
"""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.eval.dataset import GoldItem


@dataclass(frozen=True)
class RetrievedRef:
    source_key: str
    para_refs: list[str]
    text: str


def matches(hit: RetrievedRef, gold: GoldItem) -> bool:
    if hit.source_key != gold.doc_source_key:
        return False
    return bool(set(hit.para_refs) & set(gold.para_refs)) or gold.text_snippet[:80] in hit.text


def first_match_rank(hits: Sequence[RetrievedRef], gold: GoldItem) -> int | None:
    return next((i for i, h in enumerate(hits, start=1) if matches(h, gold)), None)


def first_doc_rank(hits: Sequence[RetrievedRef], gold: GoldItem) -> int | None:
    return next(
        (i for i, h in enumerate(hits, start=1) if h.source_key == gold.doc_source_key), None
    )


@dataclass
class Scores:
    n: int = 0
    hit1: float = 0.0
    hit5: float = 0.0
    hit10: float = 0.0
    mrr10: float = 0.0
    doc5: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    misses: list[str] = field(default_factory=list)


def score(
    ranks: list[int | None], doc_ranks: list[int | None], latencies: list[float], ids: list[str]
) -> Scores:
    n = len(ranks)
    if n == 0:
        return Scores()

    def rate(pred: list[bool]) -> float:
        return round(sum(pred) / n, 3)

    lat = sorted(latencies)
    return Scores(
        n=n,
        hit1=rate([r is not None and r <= 1 for r in ranks]),
        hit5=rate([r is not None and r <= 5 for r in ranks]),
        hit10=rate([r is not None and r <= 10 for r in ranks]),
        mrr10=round(sum(1 / r for r in ranks if r is not None and r <= 10) / n, 3),
        doc5=rate([r is not None and r <= 5 for r in doc_ranks]),
        latency_p50_ms=round(statistics.median(lat), 1),
        latency_p95_ms=round(lat[min(n - 1, int(0.95 * n))], 1),
        misses=[i for i, r in zip(ids, ranks, strict=True) if r is None or r > 10],
    )
