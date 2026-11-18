from collections.abc import Sequence

from app.core.config import Settings
from app.llm.gateway import LLMGateway, ModelRoute
from app.llm.groq import GroqProvider


def build_gateway(
    settings: Settings,
    models: Sequence[str] | None = None,
    *,
    max_retries: int = 3,
    max_backoff_s: float = 30.0,
) -> tuple[LLMGateway, GroqProvider]:
    """Returns the gateway and the provider (which the caller must aclose()).
    `models` overrides the fallback chain (e.g. settings.judge_models). Batch jobs can
    pass higher max_retries / max_backoff_s: they can afford to wait out rate
    limits, where a user-facing request should fall back instead."""
    if settings.groq_api_key is None:
        raise RuntimeError("GROQ_API_KEY is not set")
    provider = GroqProvider(settings.groq_api_key.get_secret_value(), settings.groq_base_url)
    routes = [
        ModelRoute(provider, model, settings.llm_requests_per_minute)
        for model in (models or settings.llm_models)
    ]
    return LLMGateway(routes, max_retries=max_retries, max_backoff_s=max_backoff_s), provider
