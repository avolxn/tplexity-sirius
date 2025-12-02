from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки retriever микросервиса из .env файла"""

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = "your qdrant api key here"
    qdrant_collection_name: str = "documents"

    prefetch_ratio: float = 1.0
    top_k: int = 20
    top_n: int = 10

    enable_query_reformulation: bool = True
    query_reformulation_llm_provider: str = "qwen"

    enable_reranker: bool = False

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
