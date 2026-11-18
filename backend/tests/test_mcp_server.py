"""MCP server tools against a fake API (httpx.MockTransport): no network, no models."""

import json
from typing import Any

import httpx
from mcp.server.mcpserver.exceptions import ToolError

from app.mcp_server import build_server, format_answer

CLAUSE = {
    "chunk_id": "00000000-0000-0000-0000-000000000001",
    "document_title": "Master Direction - Gold Loans",
    "regulator": "RBI",
    "source_url": "https://rbi.org.in/x",
    "version_valid_from": "2025-04-01",
    "version_valid_to": None,
    "heading_path": ["Chapter III", "Loan to Value"],
    "para_refs": ["12(a)"],
    "text": "The LTV ratio shall not exceed 75 per cent.",
    "amendments": [],
    "scores": {},
}
ANSWER = {
    "status": "answered",
    "answer": "The LTV may not exceed 75% [S1].",
    "citations": [
        {
            "id": "S1",
            "document_title": "Master Direction - Gold Loans",
            "para_refs": ["12(a)"],
            "heading_path": [],
            "source_url": "https://rbi.org.in/x",
            "version_valid_from": "2025-04-01",
            "amendments": [],
            "quote": "",
            "text": "",
        }
    ],
    "as_of": "2026-11-18",
    "model": "m",
    "timings_ms": {},
    "log_id": None,
    "cached": False,
}


class FakeApi:
    def __init__(self, status: int = 200, fail: bool = False) -> None:
        self.requests: list[tuple[str, str, Any]] = []
        self.status, self.fail = status, fail

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.fail:
            raise httpx.ConnectError("refused", request=request)
        body = json.loads(request.content) if request.content else dict(request.url.params)
        self.requests.append((request.method, request.url.path, body))
        if self.status != 200:
            return httpx.Response(self.status, json={})
        if request.url.path == "/search":
            return httpx.Response(200, json={"as_of": "2026-11-18", "hits": [CLAUSE], "trace": {}})
        if request.url.path == "/ask":
            return httpx.Response(200, json=ANSWER)
        return httpx.Response(
            200,
            json=[
                {
                    "id": "00000000-0000-0000-0000-000000000002",
                    "document_title": "Master Direction - KYC",
                    "source_url": "https://rbi.org.in/kyc",
                    "kind": "new_version",
                    "status": "ready",
                    "detected_at": "2026-11-17T01:00:00Z",
                    "counts": {"added": 1, "removed": 0, "changed": 2},
                    "changed": [{"para": "38", "before": "old", "after": "new text"}],
                    "added": [],
                    "removed": [],
                }
            ],
        )


async def _call(api: FakeApi, tool: str, args: dict[str, Any]) -> tuple[str, bool]:
    """(text, is_error). A ToolError is what the client receives as an error result."""
    server = build_server("http://api", transport=httpx.MockTransport(api))
    try:
        result: Any = await server.call_tool(tool, args)
    except ToolError as e:
        return str(e), True
    return result.content[0].text, bool(result.is_error)


async def test_tools_are_listed_read_only_with_schemas() -> None:
    tools = {t.name: t for t in await build_server().list_tools()}
    assert set(tools) == {"ask_compliance", "search_regulations", "recent_changes"}
    for t in tools.values():
        assert t.description and t.annotations and t.annotations.read_only_hint
    assert tools["ask_compliance"].input_schema["required"] == ["question"]


async def test_ask_returns_answer_with_numbered_sources() -> None:
    api = FakeApi()
    text, err = await _call(api, "ask_compliance", {"question": "Gold loan LTV?"})
    assert not err
    assert "75% [S1]" in text
    assert "[S1] Master Direction - Gold Loans, para 12(a)" in text
    assert "https://rbi.org.in/x" in text
    assert api.requests == [("POST", "/ask", {"question": "Gold loan LTV?"})]


async def test_as_of_date_is_passed_through() -> None:
    api = FakeApi()
    await _call(api, "search_regulations", {"query": "LTV", "as_of": "2024-01-31", "k": 50})
    assert api.requests[0][2] == {"query": "LTV", "k": 10, "as_of": "2024-01-31"}


async def test_search_returns_clauses_verbatim() -> None:
    text, _ = await _call(FakeApi(), "search_regulations", {"query": "LTV"})
    assert "[1] Master Direction - Gold Loans, para 12(a)" in text
    assert "Section: Chapter III > Loan to Value" in text
    assert "shall not exceed 75 per cent" in text


def test_unanswerable_tells_the_assistant_not_to_guess() -> None:
    # insufficient_sources comes with the retrieved clauses; they must not be listed.
    declined = {**ANSWER, "status": "insufficient_sources", "answer": "Not covered."}
    text = format_answer(declined)
    assert "Do not answer it from general knowledge" in text
    assert "Sources:" not in text and "Closest clauses:" not in text


def test_many_sub_paragraphs_are_shown_as_a_range() -> None:
    clause = {**CLAUSE, "para_refs": ["27(2)(v)", "27(2)(vi)", "27(2)(vi)(a)", "27(2)(viii)"]}
    text = format_answer({**ANSWER, "citations": [{**ANSWER["citations"][0], **clause}]})
    assert "paras 27(2)(v) to 27(2)(viii)" in text


async def test_recent_changes_summarises_the_diff() -> None:
    text, _ = await _call(FakeApi(), "recent_changes", {"limit": 3})
    assert "2026-11-17: Master Direction - KYC (new version; 1 added, 2 changed)" in text
    assert "para 38 now reads: new text" in text


def test_recent_changes_prefers_the_agent_analysis() -> None:
    from app.mcp_server import format_changes

    row = {
        "detected_at": "2026-11-18T10:00:00Z",
        "document_title": "Credit Facilities Directions",
        "kind": "archive_backfill",
        "source_url": "https://rbi.org.in/x",
        "counts": {"changed": 1},
        "changed": [{"para": "72", "after": "raw text"}],
        "impact": {
            "summary": "Para 72 clarifies right of way.",
            "paragraphs": [
                {
                    "para": "72",
                    "verified": False,
                    "severity": "low",
                    "what_changed": "An explanation was added.",
                    "action": "Update guidance.",
                }
            ],
        },
    }
    text = format_changes([row])
    assert "Impact (AI analysis): Para 72 clarifies right of way." in text
    assert "para 72 [low] [unverified]: An explanation was added. Action: Update guidance." in text
    assert "raw text" not in text  # the analysis replaces the raw excerpt
    assert "Internet Archive" in text


async def test_api_down_is_a_clear_tool_error() -> None:
    text, err = await _call(FakeApi(fail=True), "ask_compliance", {"question": "x?"})
    assert err and "not reachable" in text


async def test_api_starting_up_is_a_clear_tool_error() -> None:
    text, err = await _call(FakeApi(status=503), "search_regulations", {"query": "x"})
    assert err and "starting up" in text
