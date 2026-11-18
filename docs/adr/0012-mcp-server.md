# ADR 0012: MCP server for AI assistants

**Status:** Accepted · **Date:** 2026-11-18

## Context

Compliance teams increasingly work inside AI assistants (Claude, Cursor, VS Code
Copilot) rather than in a separate chat UI. The Model Context Protocol (MCP) is the
standard way such assistants call external tools. Without it, an assistant answers RBI
questions from its training data: unversioned, uncited and possibly out of date, which
is exactly the failure this project exists to prevent.

## Decision

**A thin MCP server** (`app/mcp_server.py`, official `mcp` Python SDK 2.x) with three
read-only tools:

| Tool | Calls | Returns |
|---|---|---|
| `ask_compliance(question, as_of?)` | `POST /ask` | Cited answer + numbered sources with paragraph, version date, URL |
| `search_regulations(query, as_of?, k?)` | `POST /search` | The clauses verbatim, with direction, paragraph, section, URL |
| `recent_changes(limit?)` | `GET /changes` | Detected changes, paragraph by paragraph |

**It is a client of the HTTP API, not a second copy of the pipeline.** The server loads
no models and touches no database. So:

- every answer still goes through the same retrieval, abstention gates, citation check,
  cache, metrics and audit log: an assistant gets nothing a UI user would not;
- it starts instantly (the GPU models stay loaded once, in the API);
- pointing `ADVANCEDRAG_API_URL` at a deployed API needs no other change.

**Tools return text written for a model to read and quote**, not raw JSON: each clause
carries its direction, paragraph and source URL, so the assistant can cite properly.
When the system declines a question, the tool says so explicitly and tells the assistant
not to answer from general knowledge, and it does *not* list "nearby" clauses, which
would invite exactly that.

**Errors are tool errors with a clear message** (API not reachable, starting up), raised
as `ToolError` so the assistant can tell the user what to do; other exceptions are
treated as crashes and their text stays on the server.

Transport: stdio by default (what desktop clients launch); `--transport
streamable-http` for remote clients.

## Checked

End to end over stdio with the SDK's client against the running API: a Hinglish
question got a Hinglish answer with two cited sources; a SEBI question was declined with
the "do not answer from general knowledge" note; search returned clauses verbatim.
Unit tests use a fake API (`httpx.MockTransport`).

## Alternatives considered

- **Server in the same process as the API** (mounting MCP on FastAPI): one process, but
  stdio clients need their own process anyway, and it would couple the two lifecycles.
- **Calling the services directly** (no HTTP): would load a second copy of the models
  (~20 s, GPU memory) and skip the API's audit and metrics path.
- **Exposing `/ask/stream`**: MCP tool results are returned whole; streaming tokens to
  the assistant adds complexity for no visible gain.

## Consequences

- The API must be running for the tools to work; the error message says how to start it.
- No authentication yet: fine for a local stdio server, required before the HTTP
  transport is exposed beyond localhost.
