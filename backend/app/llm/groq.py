"""Groq provider (OpenAI-compatible chat completions API), called with plain httpx so the
gateway owns retries, limits and fallbacks instead of an SDK doing it implicitly."""

import json
import time
from collections.abc import AsyncGenerator

import httpx

from app.llm.types import (
    FatalError,
    LLMRequest,
    LLMResponse,
    RateLimited,
    StreamEvent,
    TransientError,
    Usage,
)


def _raise_for_status(status: int, headers: httpx.Headers, body: str, model: str) -> None:
    if status == 429:
        retry_after = headers.get("retry-after")
        raise RateLimited(f"{model} rate limited", float(retry_after) if retry_after else None)
    if status >= 500:
        raise TransientError(f"{model}: HTTP {status}")
    if status != 200:
        # Never include the request (it carries the key header); the body is safe.
        raise FatalError(f"{model}: HTTP {status}: {body[:300]}")


class GroqProvider:
    name = "groq"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _payload(request: LLMRequest, model: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_completion_tokens": request.max_tokens,
        }
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def stream(self, request: LLMRequest, model: str) -> AsyncGenerator[StreamEvent]:
        """Server-sent events from the chat completions API. Only non-empty text deltas
        are yielded, so a reasoning model's hidden thinking phase still counts as
        "before the first token" (and stays retryable in the gateway)."""
        payload = {**self._payload(request, model), "stream": True}
        payload["stream_options"] = {"include_usage": True}
        answered_by, usage = model, Usage()
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode(errors="replace")
                    _raise_for_status(r.status_code, r.headers, body, model)
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    answered_by = chunk.get("model", answered_by)
                    u = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
                    if u:
                        usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
                    for choice in chunk.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield StreamEvent(text=text)
        except httpx.TimeoutException as e:
            raise TransientError(f"timeout streaming {model}") from e
        except httpx.TransportError as e:
            raise TransientError(f"network error streaming {model}: {e}") from e
        yield StreamEvent(done=True, model=answered_by, usage=usage)

    async def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        payload = self._payload(request, model)
        started = time.perf_counter()
        try:
            r = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as e:
            raise TransientError(f"timeout calling {model}") from e
        except httpx.TransportError as e:
            raise TransientError(f"network error calling {model}: {e}") from e
        latency_ms = (time.perf_counter() - started) * 1000

        _raise_for_status(r.status_code, r.headers, r.text, model)
        body = r.json()
        usage = body.get("usage") or {}
        return LLMResponse(
            text=body["choices"][0]["message"].get("content") or "",
            model=body.get("model", model),
            provider=self.name,
            usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
            latency_ms=round(latency_ms, 1),
        )
