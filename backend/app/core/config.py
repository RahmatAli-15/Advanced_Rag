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
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Fallback chain, best first. Override with LLM_MODELS='["a","b"]'.
    llm_models: list[str] = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    llm_requests_per_minute: float = 30.0

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
