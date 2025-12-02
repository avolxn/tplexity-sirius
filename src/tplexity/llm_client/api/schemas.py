from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Запрос на генерацию ответа через LLM"""

    provider: str = Field(..., description="Провайдер LLM (qwen, yandexgpt, chatgpt, deepseek)")
    messages: list[dict[str, str]] = Field(
        ...,
        description="Список сообщений в формате OpenAI",
        example=[
            {"role": "system", "content": "Ты - помощник"},
            {"role": "user", "content": "Привет!"},
        ],
    )
    temperature: float | None = Field(None, description="Температура генерации")
    max_tokens: int | None = Field(None, description="Максимальное количество токенов")


class GenerateResponse(BaseModel):
    """Ответ с сгенерированным текстом"""

    answer: str = Field(..., description="Сгенерированный ответ")
    provider: str = Field(..., description="Использованный провайдер LLM")
    model: str = Field(..., description="Использованная модель")

