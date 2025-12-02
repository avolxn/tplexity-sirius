import logging
from typing import Any

import httpx

from tplexity.bot.config import settings

logger = logging.getLogger(__name__)


class GenerationClient:
    """HTTP клиент для взаимодействия с Generation API"""

    def __init__(self, base_url: str | None = None):
        """
        Инициализация клиента

        Args:
            base_url: Базовый URL Generation API (если None, берется из settings)
        """
        self.base_url = (base_url or settings.generation_api_url).rstrip("/")
        self.client = httpx.AsyncClient(timeout=60.0)
        logger.info(f"🔄 [bot][service_client] Инициализирован клиент для {self.base_url}")

    async def send_message(
        self,
        query: str,
        llm_provider: str | None = None,
        session_id: str | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, list[str], list[dict | None], float | None, float, float]:
        """
        Отправляет сообщение в Generation API и получает ответ

        Args:
            query: Вопрос пользователя
            llm_provider: Провайдер LLM
            session_id: Идентификатор сессии
            top_k: Количество релевантных документов
            use_rerank: Использовать ли reranking
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов

        Returns:
            tuple: (answer, doc_ids, sources, search_time, generation_time, total_time)
        """
        url = f"{self.base_url}/v1/generation/generate"
        payload: dict[str, Any] = {"query": query}
        if llm_provider:
            payload["llm_provider"] = llm_provider
        if session_id:
            payload["session_id"] = session_id
        if top_k is not None:
            payload["top_k"] = top_k
        if use_rerank is not None:
            payload["use_rerank"] = use_rerank
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            answer = data["answer"]
            sources = [source.get("metadata") for source in data.get("sources", [])]
            doc_ids = [source.get("doc_id", "") for source in data.get("sources", [])]
            search_time = data.get("search_time")
            generation_time = data.get("generation_time", 0.0)
            total_time = data.get("total_time", 0.0)

            return answer, doc_ids, sources, search_time, generation_time, total_time
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ [bot][service_client] HTTP ошибка от Generation API: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"❌ [bot][service_client] Ошибка при запросе к Generation API: {e}")
            raise

    async def generate_short_answer(
        self,
        detailed_answer: str,
        llm_provider: str | None = None,
    ) -> str:
        """
        Генерирует краткий ответ на основе детального ответа

        Args:
            detailed_answer: Детальный ответ для сокращения
            llm_provider: Провайдер LLM

        Returns:
            str: Краткий ответ
        """
        url = f"{self.base_url}/v1/generation/generate-short-answer"
        payload: dict[str, Any] = {"detailed_answer": detailed_answer}
        if llm_provider:
            payload["llm_provider"] = llm_provider

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["short_answer"]
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ [bot][service_client] HTTP ошибка от Generation API: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"❌ [bot][service_client] Ошибка при генерации краткого ответа: {e}")
            raise

    async def clear_session(self, session_id: str) -> None:
        """
        Очищает историю диалога для указанной сессии

        Args:
            session_id: Идентификатор сессии
        """
        url = f"{self.base_url}/v1/generation/clear-session"
        payload = {"session_id": session_id}

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            logger.info(f"✅ [bot][service_client] История сессии {session_id} очищена")
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ [bot][service_client] HTTP ошибка при очистке сессии: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"❌ [bot][service_client] Ошибка при очистке сессии: {e}")
            raise

    async def close(self) -> None:
        """Закрывает соединение с Generation API"""
        await self.client.aclose()
        logger.info("🔌 [bot][service_client] Соединение с Generation API закрыто")
