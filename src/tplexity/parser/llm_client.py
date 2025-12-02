import logging
from typing import Any

import httpx

from tplexity.parser.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """HTTP клиент для взаимодействия с LLM Client API"""

    def __init__(self, base_url: str | None = None):
        """
        Инициализация клиента

        Args:
            base_url: Базовый URL LLM Client API (если None, берется из settings)
        """
        self.base_url = (base_url or settings.llm_client_api_url).rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)
        logger.info(f"🔄 [parser][llm_client] Инициализирован клиент для {self.base_url}")

    async def generate(
        self,
        provider: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Генерация ответа через LLM

        Args:
            provider: Провайдер LLM
            messages: Список сообщений в формате OpenAI
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов

        Returns:
            str: Сгенерированный ответ

        Raises:
            httpx.HTTPStatusError: При HTTP ошибке
            Exception: При других ошибках
        """
        url = f"{self.base_url}/v1/llm/generate"
        payload: dict[str, Any] = {
            "provider": provider,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["answer"]
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ [parser][llm_client] HTTP ошибка от LLM Client API: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"❌ [parser][llm_client] Ошибка при запросе к LLM Client API: {e}")
            raise

    async def close(self) -> None:
        """Закрывает соединение с LLM Client API"""
        await self.client.aclose()
        logger.info("🔌 [parser][llm_client] Соединение с LLM Client API закрыто")
