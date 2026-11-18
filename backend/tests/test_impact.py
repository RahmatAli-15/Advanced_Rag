"""Change-impact agent (ADR 0014): the quote check, and the whole LangGraph run against
the test database with a scripted LLM (including the verify -> re-analyse loop)."""

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.changes import impact as impact_mod
from app.changes.impact import MAX_ATTEMPTS, Change, _changes, analyse_change, check_quotes
from app.db.models import ChangeEvent, Document
from app.ingestion.raw_store import RawStore
from app.llm.gateway import LLMGateway, ModelRoute
from app.llm.types import LLMRequest, LLMResponse, StreamEvent, Usage

BEFORE = "72. An NBFC shall ensure land for all projects: (1) PPP projects - 50 per cent."
AFTER = BEFORE + " [Explanation: for generation projects the right of way is per (3).]"
CHANGE = Change(para="72", before=BEFORE, after=AFTER)


def _a(before_quote: str = "", after_quote: str = "") -> dict[str, Any]:
    return {"before_quote": before_quote, "after_quote": after_quote}


# --- the quote check ------------------------------------------------------------------


def test_quote_of_the_new_text_passes() -> None:
    assert check_quotes(CHANGE, _a(after_quote="the right of way is per (3)")) is None


def test_quote_not_in_the_text_is_rejected() -> None:
    problem = check_quotes(CHANGE, _a(after_quote="right of way is 60 per cent"))
    assert problem and "not in the AFTER text" in problem


def test_quote_of_unchanged_text_is_rejected() -> None:
    # True in both versions, so it proves nothing about the change.
    problem = check_quotes(CHANGE, _a(after_quote="An NBFC shall ensure land"))
    assert problem and "both versions" in problem


def test_new_paragraph_cannot_quote_old_text() -> None:
    new = Change(para="73", before="", after="73. A new rule for all NBFCs.")
    assert check_quotes(new, _a(before_quote="old words here")) is not None
    assert check_quotes(new, _a(after_quote="A new rule for")) is None


def test_quote_style_and_spacing_do_not_matter() -> None:
    c = Change(para="3", before="(v) ‘NBFC not availing public funds’", after="(v) Type I")
    assert check_quotes(c, _a(before_quote="'NBFC  not availing public funds'")) is None


def test_empty_quotes_are_rejected() -> None:
    assert check_quotes(CHANGE, _a()) is not None


def test_changes_lists_changed_added_and_removed_in_order() -> None:
    old = {"1": "a b", "2": "same", "3": "gone"}
    new = {"1": "a  c", "2": "same ", "4": "new"}
    assert [(c["para"], bool(c["before"]), bool(c["after"])) for c in _changes(old, new)] == [
        ("1", True, True),
        ("4", False, True),
        ("3", True, False),
    ]


# --- the whole graph --------------------------------------------------------------------


def _analysis(after_quote: str) -> str:
    return json.dumps(
        {
            "what_changed": "An explanation on right of way for generation projects was added.",
            "change_type": "clarification",
            "severity": "low",
            "affected": ["NBFCs financing power projects"],
            "action": "Update the project appraisal note.",
            "before_quote": "",
            "after_quote": after_quote,
            "search_query": "right of way generation transmission",
        }
    )


class ScriptedLLM:
    """Para 72: a made-up quote first, the right one after feedback.
    Para 80: always a made-up quote (must end up unverified). Summary: plain text."""

    name = "fake"

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.calls = {"72": 0, "80": 0}

    async def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        user = request.messages[-1].content
        self.requests.append(user)
        if not request.json_mode:
            return LLMResponse("Para 72 adds a clarification.", model, "fake", Usage(5, 5))
        para = "72" if "Paragraph: 72" in user else "80"
        self.calls[para] += 1
        good = para == "72" and self.calls[para] > 1
        quote = "the right of way is per (3)" if good else "a sentence that was never there"
        return LLMResponse(_analysis(quote), model, "fake", Usage(10, 10))

    async def stream(self, request: LLMRequest, model: str) -> Any:
        yield StreamEvent(done=True)


@pytest.mark.integration
async def test_agent_retries_failed_quotes_and_saves_the_impact(
    session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pages are stored as JSON {para: text} so the test needs no HTML fixtures.
    monkeypatch.setattr(impact_mod, "raw_paragraphs", lambda raw: json.loads(raw))
    store = RawStore(tmp_path)
    old = store.put(json.dumps({"72": BEFORE, "80": "80. Old text."}).encode(), ".json")
    new = store.put(json.dumps({"72": AFTER, "80": "80. New text."}).encode(), ".json")
    async with session_factory() as s, s.begin():
        doc = Document(
            regulator="RBI",
            doc_type="master_direction",
            source_key=f"t-{uuid.uuid4().hex[:8]}",
            title="Credit Facilities Directions",
            source_url="https://rbi.example/cf",
        )
        s.add(doc)
        await s.flush()
        event = ChangeEvent(
            document_id=doc.id,
            kind="new_version",
            old_raw_path=old.relative_path,
            new_raw_path=new.relative_path,
            status="ready",
        )
        s.add(event)
        await s.flush()
        event_id = event.id

    llm = ScriptedLLM()
    gateway = LLMGateway([ModelRoute(llm, "m", requests_per_minute=6000)])
    result = await analyse_change(event_id, session_factory, store, gateway)

    by_para = {p["para"]: p for p in result["paragraphs"]}
    assert by_para["72"]["verified"] is True
    assert by_para["72"]["after_quote"] == "the right of way is per (3)"
    assert by_para["80"]["verified"] is False  # never produced a real quote
    assert llm.calls == {"72": 2, "80": MAX_ATTEMPTS}  # 72 fixed on the first retry
    assert any("was rejected" in r and "Paragraph: 72" in r for r in llm.requests)
    assert result["attempts"] == MAX_ATTEMPTS
    assert result["summary"] == "Para 72 adds a clarification."
    async with session_factory() as s:
        saved = await s.get_one(ChangeEvent, event_id)
        assert saved.impact == result
