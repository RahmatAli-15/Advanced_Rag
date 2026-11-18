"""Typed application settings, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lives at the repo root, one level above backend/
_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    app_env: Literal["local", "test", "prod"] = "local"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://advancedrag:advancedrag@localhost:5432/advancedrag"
    redis_url: str = "redis://localhost:6379/0"

    # SecretStr keeps the key out of logs, reprs and tracebacks.
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
