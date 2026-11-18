"""Prometheus metrics: what a dashboard needs to answer "is it fast, is it right, what
does it cost?" (ADR 0011).

Everything is recorded at a few choke points, not sprinkled through the code:
  - answers  -> AnswerService._log   (every answer shown, cache hits included)
  - LLM      -> LLMGateway            (every call attempt, tokens, cost, budget)
  - queue    -> /metrics handler      (read from Redis at scrape time)

Label values are always from small fixed sets (status, stage, model, route): never a
question or a user id, which would explode the number of time series.
"""

from prometheus_client import Counter, Gauge, Histogram

# Latency buckets in seconds: cache hits are ~5 ms, full answers 1-10 s.
_BUCKETS = (0.005, 0.025, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 5, 10, 20, 60)

ANSWERS = Counter(
    "rag_answers_total",
    "Answers returned, by outcome",
    ["mode", "status", "cached"],  # mode: sync | stream
)
STAGE_SECONDS = Histogram(
    "rag_stage_seconds",
    "Time per pipeline stage (retrieval, first_token, llm, total)",
    ["stage", "cached"],
    buckets=_BUCKETS,
)

LLM_CALLS = Counter(
    "llm_calls_total",
    "LLM call attempts by route and outcome",
    ["route", "outcome"],  # ok | rate_limited | transient_error | fatal_error
)
LLM_FALLBACKS = Counter("llm_fallbacks_total", "Requests that moved to a later route")
LLM_TOKENS = Counter("llm_tokens_total", "Tokens used", ["model", "kind"])  # prompt | completion
LLM_COST = Counter(
    "llm_cost_usd_total", "Cost at the provider's list price, in US dollars", ["model"]
)
LLM_BUDGET_USED = Gauge("llm_budget_tokens_used", "Tokens used today on the primary model")
LLM_BUDGET_LIMIT = Gauge("llm_budget_tokens_limit", "Daily token budget (0 = no budget)")
LLM_BUDGET_DOWNGRADES = Counter(
    "llm_budget_downgrades_total", "Requests sent to a cheaper model because of the budget"
)

QUEUE_DEPTH = Gauge("queue_jobs", "Jobs in the queue", ["state"])  # ready | delayed | dead

# Overload protection (app/retrieval/gate.py, app/api/ratelimit.py)
GATE = Gauge("retrieval_gate", "Searches using the models or waiting for them", ["state"])
SHED = Counter("retrieval_shed_total", "Searches refused because the models were saturated")
RATE_LIMITED = Counter("rate_limited_total", "Requests refused by the per-client limit", ["route"])


def cost_usd(prices: dict[str, list[float]], model: str, prompt: int, completion: int) -> float:
    """List-price cost. `prices` maps model -> [USD per 1M input, USD per 1M output].
    Unknown models cost 0 (their tokens are still counted)."""
    p_in, p_out = prices.get(model, [0.0, 0.0])
    return (prompt * p_in + completion * p_out) / 1_000_000
