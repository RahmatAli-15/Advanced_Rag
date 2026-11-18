from app.core.config import Settings
from app.llm.gateway import LLMGateway, ModelRoute
from app.llm.groq import GroqProvider


def build_gateway(settings: Settings) -> tuple[LLMGateway, GroqProvider]:
    """Returns the gateway and the provider (which the caller must aclose())."""
    if settings.groq_api_key is None:
        raise RuntimeError("GROQ_API_KEY is not set")
    provider = GroqProvider(settings.groq_api_key.get_secret_value(), settings.groq_base_url)
    routes = [
        ModelRoute(provider, model, settings.llm_requests_per_minute)
        for model in settings.llm_models
    ]
    return LLMGateway(routes), provider
