from collections.abc import Sequence

from app.core.config import Settings
from app.llm.gateway import LLMGateway, ModelRoute
from app.llm.groq import GroqProvider


class ProviderPool:
    """The HTTP clients behind a gateway (one per API key), closed together."""

    def __init__(self, providers: list[GroqProvider]) -> None:
        self.providers = providers

    async def aclose(self) -> None:
        for p in self.providers:
            await p.aclose()


def build_gateway(
    settings: Settings,
    models: Sequence[str] | None = None,
    *,
    max_retries: int = 3,
    max_backoff_s: float = 30.0,
) -> tuple[LLMGateway, ProviderPool]:
    """Returns the gateway and its providers (which the caller must aclose()).

    `models` overrides the fallback chain (e.g. settings.judge_models). With several
    API keys, each model is tried on every key before the next model: the best model
    on another key beats a smaller model. Batch jobs can pass higher max_retries /
    max_backoff_s: they can wait out rate limits, where a user request should not."""
    keys = settings.groq_api_keys
    if not keys:
        raise RuntimeError("GROQ_API_KEY is not set")
    providers = [GroqProvider(k.get_secret_value(), settings.groq_base_url) for k in keys]
    routes = [
        ModelRoute(
            provider,
            model,
            settings.llm_requests_per_minute,
            label=f"{model}@key{i}" if len(providers) > 1 else "",
        )
        for model in (models or settings.llm_models)
        for i, provider in enumerate(providers, start=1)
    ]
    gateway = LLMGateway(routes, max_retries=max_retries, max_backoff_s=max_backoff_s)
    return gateway, ProviderPool(providers)
