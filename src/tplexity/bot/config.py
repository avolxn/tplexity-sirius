from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки Telegram Bot микросервиса из .env файла"""

    bot_token: str = Field(default="")

    @field_validator("bot_token", mode="before")
    @classmethod
    def parse_bot_token(cls, v):
        if v == "" or v is None:
            return ""
        return v

    generation_api_url: str = "http://localhost:8012"
    
    # Список доступных моделей (можно переопределить через переменную окружения)
    available_models: str = Field(
        default="qwen,chatgpt,deepseek",
        description="Список доступных моделей через запятую"
    )

    @property
    def available_models_list(self) -> list[str]:
        """Получает список доступных моделей"""
        return [m.strip() for m in self.available_models.split(",") if m.strip()]

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
