from pydantic import BaseModel, Field


class StartMonitoringResponse(BaseModel):
    """Ответ на запуск мониторинга"""

    status: str = Field(..., description="Статус операции")
    channels: str = Field(..., description="Список каналов (строка через запятую)")


class StopMonitoringResponse(BaseModel):
    """Ответ на остановку мониторинга"""

    status: str = Field(..., description="Статус операции")


class DownloadMessagesResponse(BaseModel):
    """Ответ на скачивание сообщений"""

    status: str = Field(..., description="Статус операции")
    limit_per_channel: int = Field(..., description="Лимит сообщений на канал")
    results: dict = Field(..., description="Результаты по каналам")


class StatusResponse(BaseModel):
    """Ответ на запрос статуса"""

    status: str = Field(..., description="Статус сервиса")
    config: dict | None = Field(default=None, description="Конфигурация сервиса")
    timestamp: str = Field(..., description="Временная метка")


class HealthResponse(BaseModel):
    """Ответ health check"""

    status: str = Field(..., description="Статус здоровья сервиса")


class RootResponse(BaseModel):
    """Ответ корневого эндпоинта"""

    service: str = Field(..., description="Название сервиса")
    version: str = Field(..., description="Версия сервиса")
    endpoints: dict = Field(..., description="Описание эндпоинтов")
