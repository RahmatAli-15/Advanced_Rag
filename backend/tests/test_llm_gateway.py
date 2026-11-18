"""LLM gateway behaviour with a scripted fake provider and a fake clock (no network)."""

import json

import httpx
import pytest
from pydantic import BaseModel

from app.llm.gateway import CircuitBreaker, LLMGateway, ModelRoute, TokenBucket, complete_json
from app.llm.groq import GroqProvider
from app.llm.types import (
    AllModelsFailed,
    FatalError,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    RateLimited,
    TransientError,
    Usage,
)

REQ = LLMRequest([Message("user", "hi")])


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


class ScriptedProvider:
    """Each call pops the next outcome for that model: an exception or a reply text."""

    name = "fake"

    def __init__(self, script: dict[str, list[LLMError | str]]) -> None:
        self.script = script
        self.calls: list[str] = []

    async def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        self.calls.append(model)
        outcome = self.script[model].pop(0)
        if isinstance(outcome, LLMError):
            raise outcome
        return LLMResponse(outcome, model, "fake", Usage(10, 5))


def _gateway(
    provider: ScriptedProvider, clock: FakeClock, models: list[str], **kw: float
) -> LLMGateway:
    return LLMGateway(
        [ModelRoute(provider, m, requests_per_minute=600) for m in models],
        clock=clock,
        sleep=clock.sleep,
        **kw,  # type: ignore[arg-type]
    )


async def test_retries_rate_limit_honouring_retry_after() -> None:
    clock = FakeClock()
    p = ScriptedProvider({"a": [RateLimited("slow down", retry_after_s=7), "ok"]})
    r = await _gateway(p, clock, ["a"]).complete(REQ)
    assert r.text == "ok" and r.attempts == 2
    assert 7 in clock.sleeps


async def test_falls_back_to_next_model_after_retries_exhausted() -> None:
    clock = FakeClock()
    p = ScriptedProvider({"a": [TransientError("500")] * 2, "b": ["from b"]})
    gw = _gateway(p, clock, ["a", "b"], max_retries=1, breaker_threshold=10)
    r = await gw.complete(REQ)
    assert r.model == "b" and p.calls == ["a", "a", "b"]
    assert gw.stats.fallbacks == 1 and gw.stats.retries == 1


async def test_fatal_error_is_not_retried() -> None:
    clock = FakeClock()
    p = ScriptedProvider({"a": [FatalError("400 bad request")], "b": ["ok"]})
    r = await _gateway(p, clock, ["a", "b"]).complete(REQ)
    assert p.calls == ["a", "b"] and r.text == "ok"


async def test_all_models_failing_raises() -> None:
    clock = FakeClock()
    p = ScriptedProvider({"a": [FatalError("401")], "b": [FatalError("401")]})
    with pytest.raises(AllModelsFailed):
        await _gateway(p, clock, ["a", "b"]).complete(REQ)


async def test_open_circuit_skips_model_until_cooldown() -> None:
    clock = FakeClock()
    p = ScriptedProvider({"a": [TransientError("x")] * 3 + ["a ok"], "b": ["b1", "b2", "b3"]})
    gw = _gateway(p, clock, ["a", "b"], max_retries=5, breaker_threshold=3, breaker_cooldown_s=60)

    assert (await gw.complete(REQ)).model == "b"  # a fails 3x -> circuit opens -> b
    assert gw.breaker_state("a") == "open"
    calls_before = len(p.calls)
    assert (await gw.complete(REQ)).model == "b"  # a skipped entirely
    assert p.calls[calls_before:] == ["b"]

    clock.now += 61  # cool-down over: one trial call is allowed
    assert (await gw.complete(REQ)).model == "a"
    assert gw.breaker_state("a") == "closed"


async def test_usage_is_accounted_per_model() -> None:
    clock = FakeClock()
    p = ScriptedProvider({"a": ["x", "y"]})
    gw = _gateway(p, clock, ["a"])
    await gw.complete(REQ)
    await gw.complete(REQ)
    assert gw.stats.usage_by_model["a"] == Usage(20, 10)


async def test_token_bucket_spaces_requests() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate=1.0, capacity=2, clock=clock, sleep=clock.sleep)
    for _ in range(4):
        await bucket.acquire()
    assert clock.now == pytest.approx(2.0)  # 2 from the burst, then 1 per second


def test_breaker_half_open_failure_reopens() -> None:
    clock = FakeClock()
    b = CircuitBreaker(failure_threshold=2, cooldown_s=10, clock=clock)
    b.record_failure()
    b.record_failure()
    assert b.state == "open"
    clock.now = 11
    assert b.state == "half_open"
    b.record_failure()
    assert b.state == "open"


class Answer(BaseModel):
    question: str


async def test_complete_json_repairs_invalid_output_once() -> None:
    clock = FakeClock()
    p = ScriptedProvider({"a": ["not json", json.dumps({"question": "fixed?"})]})
    out = await complete_json(_gateway(p, clock, ["a"]), REQ, Answer.model_validate_json)
    assert out.question == "fixed?"


async def test_groq_provider_maps_http_errors() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert req.headers["authorization"] == "Bearer k"
        if body["model"] == "limited":
            return httpx.Response(429, headers={"retry-after": "3"})
        if body["model"] == "down":
            return httpx.Response(503)
        if body["model"] == "bad":
            return httpx.Response(400, json={"error": "bad"})
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    p = GroqProvider("k", transport=httpx.MockTransport(handler))
    req = LLMRequest([Message("user", "q")], json_mode=True)
    ok = await p.complete(req, "good")
    assert ok.text == "{}" and ok.usage == Usage(3, 1)
    with pytest.raises(RateLimited) as rl:
        await p.complete(req, "limited")
    assert rl.value.retry_after_s == 3
    with pytest.raises(TransientError):
        await p.complete(req, "down")
    with pytest.raises(FatalError):
        await p.complete(req, "bad")
    await p.aclose()
