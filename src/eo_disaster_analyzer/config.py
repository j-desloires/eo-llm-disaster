from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Manages application-wide settings."""

    OPENAI_API_KEY: SecretStr
    # sentinelhub_client_id: str
    # sentinelhub_client_secret: str

    # Find the project root (which is 3 levels up from this file)
    # and construct the path to the .env file. This makes loading the .env
    # file independent of the current working directory.
    _env_path = Path(__file__).parent.parent.parent / ".env"

    model_config = SettingsConfigDict(
        env_file=_env_path,
        env_file_encoding="utf-8",
        case_sensitive=False,  # Allows OPENAI_API_KEY or openai_api_key
    )


@lru_cache
def get_settings() -> Settings:
    """Returns a cached instance of the application settings."""
    return Settings()
