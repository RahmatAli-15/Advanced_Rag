"""Groq provider (OpenAI-compatible chat completions API), called with plain httpx so the
gateway owns retries, limits and fallbacks instead of an SDK doing it implicitly."""

import time

import httpx

from app.llm.types import (
    FatalError,
    LLMRequest,
    LLMResponse,
    RateLimited,
    TransientError,
    Usage,
)


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

    async def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_completion_tokens": request.max_tokens,
        }
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            r = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as e:
            raise TransientError(f"timeout calling {model}") from e
        except httpx.TransportError as e:
            raise TransientError(f"network error calling {model}: {e}") from e
        latency_ms = (time.perf_counter() - started) * 1000

        if r.status_code == 429:
            retry_after = r.headers.get("retry-after")
            raise RateLimited(f"{model} rate limited", float(retry_after) if retry_after else None)
        if r.status_code >= 500:
            raise TransientError(f"{model}: HTTP {r.status_code}")
        if r.status_code != 200:
            # Never include the request (it carries the key header); the body is safe.
            raise FatalError(f"{model}: HTTP {r.status_code}: {r.text[:300]}")

        body = r.json()
        usage = body.get("usage") or {}
        return LLMResponse(
            text=body["choices"][0]["message"].get("content") or "",
            model=body.get("model", model),
            provider=self.name,
            usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
            latency_ms=round(latency_ms, 1),
        )
