from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки generation микросервиса из .env файла"""

    llm_provider: str = "deepseek"
    router_llm_provider: str = "qwen"

    retriever_api_url: str = "http://localhost:8010"
    llm_client_api_url: str = "http://localhost:8014"

    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    session_ttl: int = 86400
    max_history_messages: int = 10

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
