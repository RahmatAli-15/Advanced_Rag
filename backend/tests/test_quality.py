"""GET /quality: headline numbers come from the committed reports, live numbers from
the audit log."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.quality import (
    IMPROVEMENTS,
    accuracy_section,
    live_stats,
    pick_answer_report,
    pick_retrieval_report,
)
from app.core.config import get_settings
from app.main import create_app

REPORTS = get_settings().eval_dir / "reports"


def test_headline_numbers_match_the_committed_reports() -> None:
    acc = accuracy_section(
        pick_retrieval_report(REPORTS, "BAAI/bge-reranker-v2-m3"), pick_answer_report(REPORTS)
    )
    assert acc is not None and acc["answers"] is not None
    assert acc["retrieval"]["hit5"] == 0.965  # README: right paragraph in top 5
    assert acc["answers"]["claims_supported"] == 0.959
    assert (acc["answers"]["offtopic_refused"], acc["answers"]["offtopic_n"]) == (20, 20)
    assert [row["language"] for row in acc["languages"]] == ["English", "Hindi", "Hinglish"]
    assert acc["comparison"][-1]["hit5"] == 0.965  # the configuration actually used


def test_report_follows_the_configured_reranker() -> None:
    teacher = pick_retrieval_report(REPORTS, "BAAI/bge-reranker-v2-m3")
    student = pick_retrieval_report(REPORTS, "../data/models/reranker-student")
    assert teacher and student and teacher[0] != student[0]
    assert pick_retrieval_report(REPORTS, "some/other-model") is None


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_small_answer_reruns_are_not_headline_numbers(tmp_path: Path) -> None:
    def summary(n: int, off: int) -> dict[str, Any]:
        return {"summary": {"answerable": {"n": n}, "unanswerable": {"n": off}}}

    _write(tmp_path / "answers-20260101-0900.json", summary(30, 20))
    _write(tmp_path / "answers-20260101-1000.json", summary(10, 0))  # newer but partial
    picked = pick_answer_report(tmp_path)
    assert picked is not None and picked[0] == "answers-20260101-0900.json"


def test_improvement_sources_exist() -> None:
    for item in IMPROVEMENTS:
        assert item["after_ms"] < item["before_ms"]
        for name in item["sources"]:
            assert (REPORTS / name).exists(), name


def test_live_stats_separate_cache_hits_from_fresh_answers() -> None:
    rows = [
        ("answered", False, {"retrieval": 900, "llm": 1200, "total": 2100}),
        ("answered", False, {"retrieval": 1100, "llm": 1400, "total": 2600}),
        ("not_found", False, {"retrieval": 800, "total": 800}),
        ("answered", True, {"cache": 4, "total": 5}),
    ]
    live = live_stats(rows, days=7)
    assert (live["answers"], live["from_cache"]) == (4, 1)
    assert live["by_status"] == {"answered": 3, "not_found": 1}
    assert live["median_ms"] == 2100 and live["median_cached_ms"] == 5
    assert live["stages_ms"] == {"retrieval": 900, "llm": 1200}
    empty = live_stats([], days=7)
    assert empty["answers"] == 0 and empty["median_ms"] is None


class _Result:
    def all(self) -> list[tuple[str, bool, dict[str, Any]]]:
        return [("answered", True, {"total": 5})]


class _Session:
    async def execute(self, _: Any) -> _Result:
        return _Result()


@asynccontextmanager
async def _session() -> AsyncIterator[_Session]:
    yield _Session()


@asynccontextmanager
async def _noop(_: FastAPI) -> AsyncIterator[None]:
    yield


def test_quality_endpoint() -> None:
    app = create_app()
    app.router.lifespan_context = _noop
    app.state.session_factory = _session
    body = TestClient(app).get("/quality").json()
    assert body["live"]["answers"] == 1 and body["live"]["from_cache"] == 1
    assert len(body["improvements"]) == len(IMPROVEMENTS)
