# ADR 0010: Background jobs and change detection

**Status:** Accepted · **Date:** 2026-11-18

## Context

Regulations change. A compliance copilot must notice a new or amended direction without
someone running commands, and tell users *what* changed, paragraph by paragraph.
Crawling, chunking and embedding are slow and can fail independently (network, site
layout, GPU), so they should not run inside API requests.

## Decision

**A small job queue on Redis Streams** (`app/queue/core.py`), run by `python -m app.worker`:

| Concern | Mechanism |
|---|---|
| Work sharing | Consumer group: each job goes to one worker |
| Delivery | At-least-once: a job is acked only after it succeeds or is re-queued |
| Crash recovery | Jobs held by a dead worker are reclaimed after a visibility timeout (`XAUTOCLAIM`) |
| Retries | Exponential backoff via a delayed sorted set (30 s, 60 s, ...) |
| Dead-letter queue | After 3 attempts, or a `PermanentError`, the job is parked with its error |
| Idempotent enqueue | `job_id` claimed with `SET NX` (e.g. `refresh-rbi:2026-11-18`) |
| Schedule | Daily at 06:30 IST; runs once a day however many workers run |

Hand-written rather than a library: arq and saq pin redis-py below the version this
project uses, and the queue is small enough to read in one sitting.

**Job chain:** `refresh_rbi` (crawl) → `index_pending` (chunk + embed) →
`build_changes` (diffs). Each stage skips work already done, so at-least-once delivery
and retries are safe. A crawl stopped by bot protection raises `PermanentError` and is
**not** retried (ADR 0003: back off, never push through).

**Transactional outbox for change events.** The crawler writes a `change_events` row in
the same transaction as the version it describes (`new_document`, `new_version`,
`silent_edit`). Either both are saved or neither is, so no change is ever lost or
announced without its data. The worker later diffs old vs new text: both are parsed and
chunked with the search chunker, then compared paragraph by paragraph
("para 2(i) changed"). Results: `GET /changes` and the UI's "Recent RBI changes".

## Consequences

- Found and fixed while building this: on a silent edit (same date, new text) the old
  chunks stayed in the search index, because the chunker only processes versions
  without chunks. The edit now deletes them so indexing rebuilds them.
- Diffs need the previous *text*. Past versions RBI lists without text (ADR 0003) give
  "new version, previous text unavailable"; diffs become complete from the first
  version we capture ourselves onward.
- The worker runs on the host for now: the API image does not include the embedding
  models (the optional `ml` extra). Packaging it is part of the deployment work.
- No alerts by email/Slack yet; `GET /changes` is the integration point.
