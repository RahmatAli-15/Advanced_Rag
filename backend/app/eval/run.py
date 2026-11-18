"""Run the retrieval evaluation across configurations and write a report.

Configurations isolate each component: keyword only, vector only, hybrid (RRF), and
hybrid + cross-encoder rerank. Results are broken down by question language.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Document
from app.eval.dataset import GoldItem
from app.eval.metrics import RetrievedRef, Scores, first_doc_rank, first_match_rank, score
from app.retrieval.hybrid import HybridRetriever, SearchFilters, SearchMode


@dataclass(frozen=True)
class Config:
    name: str
    mode: SearchMode
    rerank: bool
    keyword_weight: float = 1.0


CONFIGS = [
    Config("keyword", "keyword", rerank=False),
    Config("vector", "vector", rerank=False),
    Config("hybrid", "hybrid", rerank=False),
    Config("hybrid kw=0.3", "hybrid", rerank=False, keyword_weight=0.3),
    Config("vector+rerank", "vector", rerank=True),
    Config("hybrid+rerank", "hybrid", rerank=True),
    Config("hybrid kw=0.3+rerank", "hybrid", rerank=True, keyword_weight=0.3),
]


async def _source_keys(factory: async_sessionmaker[AsyncSession]) -> dict[str, str]:
    async with factory() as s:
        return {str(i): k for i, k in (await s.execute(select(Document.id, Document.source_key)))}


async def evaluate(
    items: Sequence[GoldItem],
    plain: HybridRetriever,
    reranking: HybridRetriever,
    factory: async_sessionmaker[AsyncSession],
    configs: Sequence[Config] = CONFIGS,
    k: int = 10,
) -> dict[str, dict[str, Scores]]:
    keys = await _source_keys(factory)
    results: dict[str, dict[str, Scores]] = {}
    for cfg in configs:
        retriever = reranking if cfg.rerank else plain
        per_lang: dict[str, tuple[list[int | None], list[int | None], list[float], list[str]]] = {}
        for item in items:
            hits, trace = await retriever.search(
                item.question,
                k=k,
                filters=SearchFilters(as_of=item.as_of),
                mode=cfg.mode,
                keyword_weight=cfg.keyword_weight,
            )
            refs = [RetrievedRef(keys[str(h.document_id)], h.para_refs, h.text) for h in hits]
            for bucket in (item.language, "all"):
                ranks, docs, lats, ids = per_lang.setdefault(bucket, ([], [], [], []))
                ranks.append(first_match_rank(refs, item))
                docs.append(first_doc_rank(refs, item))
                lats.append(trace.timings_ms.get("total", 0.0))
                ids.append(item.id)
        results[cfg.name] = {lang: score(*v) for lang, v in per_lang.items()}
    return results


def render_markdown(results: dict[str, dict[str, Scores]], meta: dict[str, object]) -> str:
    lines = [
        "# Retrieval evaluation",
        "",
        *(f"- **{k}:** {v}" for k, v in meta.items()),
        "",
        "## All questions",
        "",
        "| Config | n | hit@1 | hit@5 | hit@10 | MRR@10 | doc@5 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, by_lang in results.items():
        s = by_lang["all"]
        lines.append(
            f"| {name} | {s.n} | {s.hit1:.3f} | {s.hit5:.3f} | {s.hit10:.3f} | "
            f"{s.mrr10:.3f} | {s.doc5:.3f} | {s.latency_p50_ms} | {s.latency_p95_ms} |"
        )
    lines += ["", "## hit@5 by question language", ""]
    langs = sorted({lang for r in results.values() for lang in r if lang != "all"})
    lines.append("| Config | " + " | ".join(langs) + " |")
    lines.append("|---|" + "---|" * len(langs))
    for name, by_lang in results.items():
        lines.append(
            f"| {name} | " + " | ".join(f"{by_lang[lang].hit5:.3f}" for lang in langs) + " |"
        )
    return "\n".join(lines) + "\n"


def write_report(
    results: dict[str, dict[str, Scores]], meta: dict[str, object], out_dir: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    md = out_dir / f"retrieval-{stamp}.md"
    js = out_dir / f"retrieval-{stamp}.json"
    md.write_text(render_markdown(results, meta), encoding="utf-8")
    js.write_text(
        json.dumps(
            {
                "meta": meta,
                "results": {
                    c: {lang: asdict(s) for lang, s in r.items()} for c, r in results.items()
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return md, js
