# ADR 0002: All LLM calls go through an internal gateway

**Status:** Accepted · **Date:** 2026-11-18

## Context

We start on Groq's hosted API for speed and cost, and want to move to local models
(Ollama) later. Hosted APIs rate-limit, time out and have outages.

## Decision

No code calls a provider SDK directly. Every call goes through an `LLMGateway` with
pluggable provider adapters (Groq first, Ollama next). The gateway owns:

- retries with exponential backoff and jitter, honouring `retry-after`
- a client-side token-bucket rate limiter, so we stay inside provider limits
- a fallback chain (primary model → smaller model → local model)
- a circuit breaker per provider
- token and cost accounting per request

## Consequences

- **+** Switching provider or model is configuration, not a code change.
- **+** Rate limits degrade answers gracefully instead of failing them.
- **+** One place for tracing, cost tracking and audit logging.
- **−** We write the adapters ourselves instead of using a framework's abstraction.
