from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # The product name is not decided yet; everything user-facing reads it from here.
    assistant_name: str = "Assistant"
    models_config: Path = ROOT / "config" / "models.yaml"
    # Unset -> conversations are kept in memory only (fine for local dev and tests).
    database_url: str | None = None
    ollama_base_url: str | None = None
    max_agent_steps: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
