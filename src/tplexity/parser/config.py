from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки parser микросервиса из .env файла"""

    api_id: int | None = Field(default=None)

    @field_validator("api_id", mode="before")
    @classmethod
    def parse_api_id(cls, v):
        if v == "" or v is None:
            return None

        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return None

        if isinstance(v, int):
            return v
        return None

    api_hash: str | None = None
    phone: str | None = None
    session_name: str = "my_session"
    session_string: str | None = None

    @field_validator("session_string", mode="before")
    @classmethod
    def parse_session_string(cls, v):
        if v == "" or v is None:
            return None
        return v

    channels: str = "omyinvestments,alfa_investments,tb_invest_official,SberInvestments,centralbank_russia,selfinvestor"

    webhook_url: str | None = None

    data_dir: str = "data"

    llm_provider: str = "qwen"
    llm_client_api_url: str = "http://localhost:8014"

    qdrant_host: str | None = None
    qdrant_port: int | None = None
    qdrant_api_key: str | None = None
    qdrant_collection_name: str | None = None

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def get_channels_list(self) -> list[str]:
        """Преобразует строку каналов в список"""
        if not self.channels:
            return []
        return [ch.strip() for ch in self.channels.split(",") if ch.strip()]


settings = Settings()
