# ADR 0003: Crawling policy for regulator websites

**Status:** Accepted · **Date:** 2026-11-18

## Context

We ingest public regulatory publications (starting with RBI Master Directions).
Observed on rbi.org.in in November 2026:

- `www.rbi.org.in/robots.txt` answers HTTP 418.
- Master Direction pages (`BS_ViewMasDirections.aspx`) serve the full text as HTML.
- `rbidocs.rbi.org.in` (PDFs, annexes) answers automated clients with a
  "are you human" bot-protection challenge.
- Version lists (`NotificationPreVersion.aspx`) give every past version's date, but
  past versions' text pages come back empty.

## Decision

1. **Obey robots.txt per RFC 9309.** 2xx is parsed and obeyed; 4xx means
   "unavailable", which allows crawling; 5xx, network errors, or an HTML page served
   as robots.txt mean "unreachable", which disallows everything on that host.
2. **Never bypass bot protection.** A challenge page raises `BlockedByBotProtection`
   and aborts the whole run. No browser automation, no spoofed user agents, no
   challenge solving. Annex PDFs are recorded as links, not downloaded.
3. **Be slow.** At most one request every 3 seconds per host (or the robots.txt
   `Crawl-delay`, whichever is longer), with backoff on 429/5xx that honours `Retry-After`.
4. **Identify honestly** with a stable `AdvancedRagBot/x.y` user agent.
5. **Fetch each thing once.** Raw pages are stored content-addressed; unchanged text
   is not re-stored; re-runs are idempotent.
6. **Prefer HTML over PDF.** HTML keeps headings, numbering and tables, with no OCR.

## Consequences

- Past versions are recorded with dates but `content_status = unavailable`. Their
  text accumulates from now on, as the daily crawl captures each new version.
- Annex content (often tables) is missing until an allowed source is found.
- A full NBFC crawl (~45 directions) takes a few minutes. That is fine for a daily job.
- Before commercial use, the terms of each source should be reviewed by a lawyer.
