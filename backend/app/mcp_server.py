"""MCP server: lets AI assistants (Claude Desktop, Claude Code, Cursor, VS Code) search
and cite RBI regulations through tool calls (ADR 0012).

    assistant ─MCP (stdio)─▶ this server ─HTTP─▶ AdvancedRag API (/search, /ask, /changes)

It is a thin client of the running API, not a second copy of the pipeline: no models are
loaded here, every answer still goes through the same retrieval, abstention, citation
check, cache and audit log, and the server works the same against a remote deployment.

Tools return plain text formatted for a language model to read and quote: every clause
comes with its direction, paragraph number, version date and source URL.

    python -m app.mcp_server                          # stdio (what desktop clients run)
    python -m app.mcp_server --transport streamable-http --port 8765
"""

import argparse
import logging
import os
import sys
from datetime import date
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

DEFAULT_API_URL = "http://localhost:8000"
CLAUSE_CHARS = 1500  # enough for a full clause; long ones are cut with a marker

INSTRUCTIONS = """Tools over the Reserve Bank of India's NBFC Master Directions (Indian \
financial regulation), with every rule versioned by date.

- ask_compliance: a cited answer to a compliance question. Prefer it for direct questions.
- search_regulations: the most relevant clauses, verbatim. Use it to read the rules
  yourself or to check an answer.
- recent_changes: what changed in the directions recently, paragraph by paragraph.

Pass as_of (YYYY-MM-DD) to ask what the rules were on a past date. When you use these
results, cite the direction and paragraph and include the source URL. If a tool says the
answer is not in the directions, say so; do not fill the gap from memory. This is not
legal advice."""

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)

log = logging.getLogger(__name__)


class ApiUnavailable(ToolError):
    """Shown to the assistant as the tool's error message. (Any other exception counts
    as a crash, and the SDK hides its text from the client.)"""


def _clip(text: str, limit: int = CLAUSE_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " […]"


def _refs(para_refs: list[str]) -> str:
    if not para_refs:
        return "unnumbered paragraph"
    if len(para_refs) > 3:  # a clause with many sub-paragraphs: show the range
        return f"paras {para_refs[0]} to {para_refs[-1]}"
    return f"para {', '.join(para_refs)}"


def format_search(data: dict[str, Any]) -> str:
    hits = data["hits"]
    if not hits:
        return f"No clauses found in the RBI directions in force on {data['as_of']}."
    out = [f"{len(hits)} clauses from RBI directions in force on {data['as_of']}:"]
    for i, h in enumerate(hits, start=1):
        heading = " > ".join(h["heading_path"])
        out.append(
            f"\n[{i}] {h['document_title']}, {_refs(h['para_refs'])}"
            f" (version in force from {h['version_valid_from']})"
            + (f"\nSection: {heading}" if heading else "")
            + f"\nSource: {h['source_url']}\n\n{_clip(h['text'])}"
        )
    return "\n".join(out)


def format_answer(data: dict[str, Any]) -> str:
    status = data["status"]
    lines = [data["answer"].strip()]
    if status in ("not_found", "no_coverage", "insufficient_sources"):
        lines.append(
            "\n(The RBI directions in this system do not answer this. Do not answer it "
            "from general knowledge as if it were RBI regulation.)"
        )
    elif status == "llm_unavailable":
        lines.append("\n(The answer model is unavailable right now; the closest clauses follow.)")
    elif status == "ungrounded":
        lines.append("\n(The draft answer had no valid citation, so treat it as unverified.)")
    # Sources for an answer, or the closest clauses when only the LLM is down. Not for
    # a declined question: a list of "nearby" clauses invites answering it anyway.
    if data["citations"] and status in ("answered", "ungrounded", "llm_unavailable"):
        lines.append("\nSources:" if status == "answered" else "\nClosest clauses:")
        for c in data["citations"]:
            lines.append(
                f"[{c['id']}] {c['document_title']}, {_refs(c['para_refs'])}"
                f" (in force from {c['version_valid_from']}): {c['source_url']}"
            )
    lines.append(f"\nRules as of {data['as_of']}." + (" (served from cache)" * data["cached"]))
    return "\n".join(lines)


def format_changes(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No changes to the RBI directions have been detected yet."
    kinds = {
        "new_document": "new direction",
        "new_version": "new version",
        "silent_edit": "edited without a new version date",
    }
    out = []
    for r in rows:
        counts = r["counts"]
        summary = ", ".join(f"{n} {k}" for k, n in counts.items() if n) or "no paragraph diff"
        out.append(
            f"- {r['detected_at'][:10]}: {r['document_title']}"
            f" ({kinds.get(r['kind'], r['kind'])}; {summary})\n  {r['source_url']}"
        )
        for ch in r["changed"][:3]:
            out.append(
                f"  · para {ch.get('para', '?')} now reads: {_clip(str(ch.get('after', '')), 300)}"
            )
    return "Recent changes, newest first:\n" + "\n".join(out)


def build_server(
    api_url: str = DEFAULT_API_URL, transport: httpx.AsyncBaseTransport | None = None
) -> MCPServer:
    """`transport` lets tests swap the network for an in-memory fake API."""

    async def call(method: str, path: str, **kw: Any) -> Any:
        try:
            async with httpx.AsyncClient(
                base_url=api_url, transport=transport, timeout=120.0
            ) as client:
                r = await client.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise ApiUnavailable(
                f"The AdvancedRag API at {api_url} is not reachable ({type(e).__name__}). "
                "Start it with: uvicorn app.main:app"
            ) from e
        if r.status_code == 503:
            raise ApiUnavailable("The AdvancedRag API is starting up or degraded; retry shortly.")
        if r.is_error:
            raise ApiUnavailable(f"The AdvancedRag API returned HTTP {r.status_code} for {path}.")
        return r.json()

    def as_of_json(as_of: date | None) -> dict[str, str]:
        return {"as_of": as_of.isoformat()} if as_of else {}

    server = MCPServer("advancedrag", instructions=INSTRUCTIONS, version="0.1.0")

    @server.tool(title="Ask an RBI compliance question", annotations=READ_ONLY)
    async def ask_compliance(question: str, as_of: date | None = None) -> str:
        """Answer a question about RBI NBFC regulation, citing the exact clauses.

        Works in English, Hindi and Hinglish. Says so when the directions do not cover
        the question. as_of: answer with the rules in force on this date (YYYY-MM-DD,
        default today)."""
        data = await call("POST", "/ask", json={"question": question, **as_of_json(as_of)})
        return format_answer(data)

    @server.tool(title="Search RBI regulations", annotations=READ_ONLY)
    async def search_regulations(query: str, as_of: date | None = None, k: int = 5) -> str:
        """Find the RBI NBFC direction clauses most relevant to a query, verbatim, with
        direction, paragraph number, version date and source URL.

        as_of: search the rules in force on this date (YYYY-MM-DD, default today).
        k: number of clauses, 1-10."""
        body = {"query": query, "k": max(1, min(k, 10)), **as_of_json(as_of)}
        return format_search(await call("POST", "/search", json=body))

    @server.tool(title="Recent RBI regulation changes", annotations=READ_ONLY)
    async def recent_changes(limit: int = 5) -> str:
        """List the most recent changes detected in the RBI directions (new directions,
        new versions, edits), with the paragraphs that changed. limit: 1-20."""
        rows = await call("GET", "/changes", params={"limit": max(1, min(limit, 20))})
        return format_changes(rows)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    # stdout carries the MCP protocol over stdio: logs must go to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server = build_server(os.environ.get("ADVANCEDRAG_API_URL", DEFAULT_API_URL))
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", port=args.port)


if __name__ == "__main__":
    main()
