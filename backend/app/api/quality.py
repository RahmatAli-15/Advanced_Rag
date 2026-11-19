"""GET /quality: how good and how fast the system is, for the UI's Quality page.

Two kinds of numbers, kept apart on purpose:
- measured accuracy, read from the evaluation reports in eval/reports (a live question
  has no known right answer, so accuracy can only be measured on the test sets);
- live health, computed from the answer audit log (answer_logs).
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.db.models import AnswerLog

router = APIRouter(tags=["quality"])

DEFAULT_CONFIG = "hybrid kw=0.3+rerank"  # the retrieval the API uses (ADR 0006)
COMPARISON = [
    ("keyword", "Keyword only"),
    ("vector", "Meaning only (vectors)"),
    ("hybrid kw=0.3", "Meaning + keyword"),
    (DEFAULT_CONFIG, "Meaning + keyword + reranker (used)"),
]
LANGUAGES = [("en", "English"), ("hi", "Hindi"), ("hinglish", "Hinglish")]

# Before/after pairs, each from a committed report (tests check the files exist).
IMPROVEMENTS: list[dict[str, Any]] = [
    {
        "what": "Repeat question",
        "before_ms": 2460,
        "after_ms": 12,
        "how": "Exact-match answer cache in Redis",
        "sources": ["answers-20261118-1044.md", "load-20261118-1136.md"],
    },
    {
        "what": "16 users at once (search, p95)",
        "before_ms": 38085,
        "after_ms": 3820,
        "how": "Load gate: 2 at a time, others told to retry",
        "sources": ["load-20261118-1517.md", "load-20261118-1520.md"],
    },
    {
        "what": "Rerank 30 clauses on GPU (p50)",
        "before_ms": 570,
        "after_ms": 80,
        "how": "Own reranker, distilled 568M → 118M",
        "sources": ["distill-bench-20261119.md"],
    },
    {
        "what": "Rerank 30 clauses on CPU (p50)",
        "before_ms": 29795,
        "after_ms": 3085,
        "how": "Same distilled reranker, for a CPU server",
        "sources": ["distill-bench-20261119.md"],
    },
]


def _newest(reports: Path, pattern: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for path in sorted(reports.glob(pattern), reverse=True):  # names sort by date
        try:
            out.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return out


def _model_name(model: str) -> str:
    return Path(model.replace("\\", "/")).name


def pick_retrieval_report(reports: Path, reranker_model: str) -> tuple[str, dict[str, Any]] | None:
    """Newest retrieval report measured with the reranker the API is configured with."""
    for name, data in _newest(reports, "retrieval-*.json"):
        used = data.get("meta", {}).get("reranker_model", "")
        if _model_name(used) == _model_name(reranker_model) and DEFAULT_CONFIG in data.get(
            "results", {}
        ):
            return name, data
    return None


def pick_answer_report(reports: Path) -> tuple[str, dict[str, Any]] | None:
    """Newest full answer evaluation (30+ answerable and 20+ off-topic questions);
    smaller runs are targeted re-checks, not headline numbers."""
    for name, data in _newest(reports, "answers-*.json"):
        s = data.get("summary", {})
        answerable = s.get("answerable", {}).get("n", 0)
        off_topic = s.get("unanswerable", {}).get("n", 0)
        if answerable >= 30 and off_topic >= 20:
            return name, data
    return None


def accuracy_section(
    retrieval: tuple[str, dict[str, Any]] | None, answers: tuple[str, dict[str, Any]] | None
) -> dict[str, Any] | None:
    if retrieval is None:
        return None
    r_name, r = retrieval
    results = r["results"]
    used = results[DEFAULT_CONFIG]
    by_lang = answers[1]["summary"].get("by_language", {}) if answers else {}
    out: dict[str, Any] = {
        "retrieval_report": r_name,
        "dataset": r["meta"].get("dataset"),
        "reranker": _model_name(r["meta"].get("reranker_model", "")),
        "retrieval": {k: used["all"][k] for k in ("n", "hit1", "hit5", "mrr10")},
        "comparison": [
            {"label": label, "hit5": results[key]["all"]["hit5"]}
            for key, label in COMPARISON
            if key in results
        ],
        "languages": [
            {
                "language": label,
                "n": used[code]["n"],
                "hit1": used[code]["hit1"],
                "hit5": used[code]["hit5"],
                "mrr10": used[code]["mrr10"],
                "claims_supported": by_lang.get(code, {}).get("claim_support"),
            }
            for code, label in LANGUAGES
            if code in used
        ],
        "answers": None,
    }
    if answers:
        a_name, a = answers
        s = a["summary"]
        out["answers"] = {
            "report": a_name,
            "claims_supported": s["claims"]["supported"],
            "numbers_grounded": s["claims"]["numbers_grounded"],
            "citations_accurate": s["claims"]["citation_accurate"],
            "offtopic_refused": round(s["unanswerable"]["abstained"] * s["unanswerable"]["n"]),
            "offtopic_n": s["unanswerable"]["n"],
        }
    return out


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))]


def live_stats(rows: list[tuple[str, bool, dict[str, Any]]], days: int) -> dict[str, Any]:
    """Summarise audit-log rows (status, cached, timings_ms) from the last `days` days."""

    def ms(t: dict[str, Any], key: str) -> float | None:
        v = t.get(key)
        return float(v) if isinstance(v, int | float) else None

    statuses: dict[str, int] = {}
    for status, _, _ in rows:
        statuses[status] = statuses.get(status, 0) + 1
    fresh = [t for status, cached, t in rows if not cached]
    hits = [t for _, cached, t in rows if cached]
    fresh_total = [v for t in fresh if (v := ms(t, "total")) is not None]
    hit_total = [v for t in hits if (v := ms(t, "total")) is not None]
    answered = [t for status, cached, t in rows if status == "answered" and not cached]
    stage_values = {
        key: [v for t in answered if (v := ms(t, key)) is not None] for key in ("retrieval", "llm")
    }
    return {
        "days": days,
        "answers": len(rows),
        "from_cache": len(hits),
        "by_status": dict(sorted(statuses.items(), key=lambda kv: -kv[1])),
        "median_ms": _percentile(fresh_total, 0.5),
        "p95_ms": _percentile(fresh_total, 0.95),
        "median_cached_ms": _percentile(hit_total, 0.5),
        "stages_ms": {key: _percentile(v, 0.5) for key, v in stage_values.items()},
    }


@router.get("/quality")
async def quality(request: Request, days: int = Query(7, ge=1, le=90)) -> dict[str, Any]:
    settings = get_settings()
    reports = settings.eval_dir / "reports"
    accuracy = accuracy_section(
        pick_retrieval_report(reports, settings.reranker_model), pick_answer_report(reports)
    )
    sf: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    since = datetime.now(UTC) - timedelta(days=days)
    async with sf() as s:
        rows = (
            await s.execute(
                select(AnswerLog.status, AnswerLog.cached, AnswerLog.timings_ms)
                .where(AnswerLog.created_at >= since)
                .order_by(AnswerLog.created_at.desc())
                .limit(10_000)
            )
        ).all()
    return {
        "accuracy": accuracy,
        "improvements": IMPROVEMENTS,
        "live": live_stats([(r[0], r[1], r[2] or {}) for r in rows], days),
    }
