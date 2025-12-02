from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки LLM клиента из .env файла"""

    temperature: float = 0.7
    max_tokens: int = 1000

    qwen_model: str = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"
    qwen_api_key: str = "sk-no-key-required"
    qwen_base_url: str = "http://localhost:8100/v1"

    yandexgpt_model: str = "yandexgpt-lite"
    yandexgpt_api_key: str = ""
    yandexgpt_folder_id: str = ""
    yandexgpt_base_url: str = "https://llm.api.cloud.yandex.net/v1"

    chatgpt_model: str = "gpt-4o-mini"
    chatgpt_api_key: str = ""

    deepseek_model: str = "deepseek-chat"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"

    available_models: list[str] = ["qwen", "chatgpt", "deepseek"]

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
