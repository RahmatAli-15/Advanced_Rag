"""Typed application settings, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env and data/ live at the repo root, one level above backend/
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    app_env: Literal["local", "test", "prod"] = "local"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://advancedrag:advancedrag@localhost:5432/advancedrag"
    redis_url: str = "redis://localhost:6379/0"

    # SecretStr keeps the key out of logs, reprs and tracebacks.
    groq_api_key: SecretStr | None = None
    # Optional extra keys. When the best model is rate-limited on one key, the gateway
    # tries it on the next key before falling back to a smaller model.
    groq_api_key_2: SecretStr | None = None
    groq_api_key_3: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Fallback chain, best first. Override with LLM_MODELS='["a","b"]'.
    llm_models: list[str] = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    llm_requests_per_minute: float = 30.0
    # Answer-evaluation judge: a different model family from the answer model, so the
    # judge is not grading its own phrasing (self-preference bias).
    judge_models: list[str] = ["qwen/qwen3.8-27b"]
    # Daily token budget for the primary model; past it, requests start at the next,
    # cheaper model (ADR 0011). 0 turns the budget off.
    llm_daily_token_budget: int = 1_000_000
    # List prices in USD per 1M tokens [input, output], for the cost metric. Check the
    # provider's pricing page; models not listed are counted at 0.
    llm_prices_usd_per_m: dict[str, list[float]] = {
        "openai/gpt-oss-120b": [0.15, 0.60],
        "openai/gpt-oss-20b": [0.075, 0.30],
    }

    # Overload protection (ADR 0013). Requests per client IP per minute on /ask,
    # /ask/stream and /search (0 = off), and how many searches may use the GPU models
    # at once / wait for them before new ones are refused with 503. Measured on the
    # 4 GB laptop GPU (ADR 0013): without a cap, 16 parallel searches collapsed throughput
    # to 0.5/s (p95 38 s); 2 at a time kept ~1.3/s, and 1 at a time cost ~20%.
    rate_limit_per_minute: int = 20
    retrieval_max_concurrent: int = 2
    retrieval_max_waiting: int = 2

    data_dir: Path = _REPO_ROOT / "data"
    eval_dir: Path = _REPO_ROOT / "eval"  # datasets and reports (committed)
    # Identify the crawler honestly. Add a contact URL here once the repo is public.
    crawler_user_agent: str = "AdvancedRagBot/0.1"
    crawler_min_interval_s: float = 3.0

    # Local models (optional `ml` extra). "auto" = CUDA if available, else CPU.
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    ml_device: str = "auto"
    # "real" loads the models above; "fake" uses tiny test doubles (no downloads).
    models_backend: Literal["real", "fake"] = "real"

    @property
    def groq_api_keys(self) -> list[SecretStr]:
        keys = (self.groq_api_key, self.groq_api_key_2, self.groq_api_key_3)
        return [k for k in keys if k is not None and k.get_secret_value()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
