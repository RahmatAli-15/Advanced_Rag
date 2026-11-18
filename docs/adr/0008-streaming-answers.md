# ADR 0008: Streaming answers to the chat UI

**Status:** Accepted · **Date:** 2026-11-18

## Context

A cited answer takes about 2 s (retrieval ~1 s, generation ~1 s). Without streaming the
user stares at a spinner for the whole time. With streaming, the clauses appear after
retrieval and the text starts about 0.8 s later. But streaming conflicts with two things
ADR 0007 relies on:

1. **The "not answerable" signal.** `/ask` uses JSON mode (`{"answerable": …}`). A
   half-written JSON string cannot be shown to a user.
2. **Citation verification** needs the complete answer. Invented citations can only be
   removed, and the status decided, at the end.

## Decision

`POST /ask/stream` returns Server-Sent Events: `stage` → `sources` → `stage` →
`delta`… → `final` (or `error`).

- **SSE over fetch, not WebSocket.** Traffic is one-way, SSE passes through proxies and
  the Vite dev proxy unchanged, and it needs no connection state on the server. The
  question goes in a POST body, so the UI reads the stream with `fetch()` and a small
  incremental parser (`EventSource` only supports GET). Every `data:` payload is JSON,
  so newlines in the answer cannot break the framing.
- **Plain-text generation with a marker line.** The streaming prompt
  (`answer-stream-v1`) asks for markdown and, when the sources do not answer the
  question, a first line `NOT_IN_SOURCES`. The server holds back the first few
  characters until it knows whether the reply starts with the marker, so the marker is
  never shown even when it arrives split across tokens.
- **Deltas are a preview; `final` is the record.** The `final` event carries the
  verified answer (normalised citations, invented ones removed), the status and the
  audit-log id. The UI replaces the preview with it. The UI applies the same
  citation normalisation while streaming, so the text does not jump.
- **Retry only before the first token.** The gateway's retries, backoff, circuit
  breaker and model fallback all apply until the first text event. After that, a
  failure raises `StreamInterrupted` and the user gets `llm_unavailable` with the
  relevant clauses. Splicing a second model's output onto the first would produce text
  no single model wrote. The provider yields only non-empty content deltas, so a
  reasoning model's hidden thinking phase still counts as "before the first token".
- **Cancellation is real.** When the browser aborts (Stop button, closed tab),
  Starlette cancels the generator. `aclosing()` and the gateway's `finally` close the
  provider stream and its HTTP connection, so no more tokens are generated or billed.

## Consequences

- Measured on the live system: sources visible at ~0.7–1.5 s, first token ~0.7–0.8 s
  after that, total ~1.5–2.1 s. Most of the latency is retrieval (cross-encoder on a
  laptop GPU).
- Two prompt variants now exist (`answer-v1` JSON, `answer-stream-v1` markdown). They
  share the rules text in code, and the audit log records which one produced each answer.
- An answer the user stopped mid-stream is not written to the audit log; only completed
  answers are records.
- Observed: a Hinglish question was answered in Devanagari Hindi. Language matching
  needs an evaluation case.
