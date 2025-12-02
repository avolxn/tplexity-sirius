from pydantic import BaseModel, Field, field_validator

try:
    from tplexity.llm_client.config import settings as llm_settings
except ImportError:
    llm_settings = None


class GenerateRequest(BaseModel):
    """Схема для запроса генерации ответа"""

    query: str = Field(..., description="Вопрос пользователя")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Количество релевантных документов для использования в контексте (если не указано, используется значение из config)",
    )
    use_rerank: bool | None = Field(
        default=None,
        description="Использовать ли reranking при поиске документов (если не указано, используется значение из config)",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Температура для генерации (если не указано, используется значение из config)",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=4000,
        description="Максимальное количество токенов в ответе (если не указано, используется значение из config)",
    )
    llm_provider: str | None = Field(
        default=None,
        description="Провайдер LLM для использования (если не указано, используется значение из config)",
    )

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str | None) -> str | None:
        """Валидирует, что указанный провайдер входит в список доступных моделей"""
        if v is None:
            return None

        if llm_settings and hasattr(llm_settings, "available_models"):
            available_models = llm_settings.available_models
        else:
            available_models = ["qwen", "chatgpt", "deepseek"]

        v_lower = v.lower().strip()
        if v_lower not in available_models:
            available_str = ", ".join(f"'{m}'" for m in available_models)
            raise ValueError(
                f"Провайдер '{v}' не найден в списке доступных моделей. " f"Доступные модели: {available_str}"
            )

        return v_lower

    session_id: str | None = Field(
        default=None,
        description="Идентификатор сессии для сохранения истории диалога (если не указано, история не сохраняется)",
    )


class SourceInfo(BaseModel):
    """Схема для информации об источнике"""

    doc_id: str = Field(..., description="ID документа")
    metadata: dict | None = Field(default=None, description="Метаданные документа")


class GenerateResponse(BaseModel):
    """Схема для ответа генерации"""

    answer: str = Field(..., description="Сгенерированный ответ")
    sources: list[SourceInfo] = Field(default_factory=list, description="Список источников (doc_ids и метаданные)")
    query: str = Field(..., description="Исходный запрос пользователя")
    search_time: float | None = Field(
        default=None, description="Время поиска информации в секундах (если поиск выполнялся)"
    )
    generation_time: float = Field(..., description="Время генерации ответа в секундах")
    total_time: float = Field(..., description="Общее время обработки запроса в секундах")


class ClearSessionRequest(BaseModel):
    """Схема для запроса очистки истории сессии"""

    session_id: str = Field(..., description="Идентификатор сессии для очистки")


class ClearSessionResponse(BaseModel):
    """Схема для ответа очистки истории сессии"""

    success: bool = Field(..., description="Успешность операции")
    message: str = Field(..., description="Сообщение о результате")


class GenerateShortAnswerRequest(BaseModel):
    """Схема для запроса генерации краткого ответа"""

    detailed_answer: str = Field(..., description="Детальный ответ для сокращения")
    llm_provider: str | None = Field(
        default=None,
        description="Провайдер LLM для использования (если не указано, используется значение из config)",
    )

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str | None) -> str | None:
        """Валидирует, что указанный провайдер входит в список доступных моделей"""
        if v is None:
            return None

        if llm_settings and hasattr(llm_settings, "available_models"):
            available_models = llm_settings.available_models
        else:
            available_models = ["qwen", "chatgpt", "deepseek"]

        v_lower = v.lower().strip()
        if v_lower not in available_models:
            available_str = ", ".join(f"'{m}'" for m in available_models)
            raise ValueError(
                f"Провайдер '{v}' не найден в списке доступных моделей. " f"Доступные модели: {available_str}"
            )

        return v_lower


class GenerateShortAnswerResponse(BaseModel):
    """Схема для ответа генерации краткого ответа"""

    short_answer: str = Field(..., description="Краткий ответ")
