"""
Модуль для взаимодействия с inference endpoint.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import httpx

logger = logging.getLogger(__name__)


async def get_documents_from_retriever_async(
    retriever_url: str,
    doc_ids: List[str],
    timeout: int = 30
) -> List[Dict[str, Any]]:
    """
    Асинхронно получает тексты документов из retriever API по doc_id.
    Это те же тексты, которые использовал generation для формирования ответа.
    
    Args:
        retriever_url: URL retriever API (например, http://localhost:8020)
        doc_ids: Список doc_id для получения
        timeout: Таймаут запроса в секундах
        
    Returns:
        Список словарей [{"doc_id": ..., "text": ..., "metadata": {...}}]
    """
    if not doc_ids:
        return []
    
    url = f"{retriever_url}/retriever/documents/get"
    payload = {"doc_ids": doc_ids}
    
    try:
        # Увеличиваем лимиты для параллельных запросов
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            documents = result.get("documents", [])
            
            # Преобразуем в формат [{"doc_id": ..., "text": ..., "metadata": {...}}]
            return [
                {
                    "doc_id": doc.get("doc_id", ""),
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {})
                }
                for doc in documents
            ]
    except httpx.RequestError as e:
        logger.error(f"Ошибка при получении документов из retriever API: {e}")
        return []


def get_documents_from_retriever(
    retriever_url: str,
    doc_ids: List[str],
    timeout: int = 30
) -> List[Dict[str, Any]]:
    """
    Синхронная обертка для get_documents_from_retriever_async (для обратной совместимости).
    """
    return asyncio.run(get_documents_from_retriever_async(retriever_url, doc_ids, timeout))


class InferenceClient:
    """
    Клиент для вызова inference endpoint.
    """
    
    def __init__(
        self, 
        endpoint_url: Optional[str] = None, 
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Инициализация клиента.
        
        Args:
            endpoint_url: URL endpoint для inference (если None - используется mock)
            api_key: API ключ для аутентификации
            timeout: Таймаут запроса в секундах
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout
        self.use_mock = not endpoint_url or endpoint_url.strip() == ""
        self._client: Optional[httpx.AsyncClient] = None
        
        if self.use_mock:
            logger.info("Используется mock inference client")
        else:
            logger.info(f"Inference endpoint: {endpoint_url}")
    
    async def __aenter__(self):
        """Асинхронный контекстный менеджер для создания клиента."""
        if not self.use_mock and not self._client:
            # Увеличиваем лимиты для параллельных запросов
            limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
            self._client = httpx.AsyncClient(timeout=self.timeout, limits=limits)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Закрытие клиента при выходе из контекста."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def generate_async(
        self, 
        query: str, 
        contexts: List[str] = None,
        top_k: int = 5
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Асинхронно генерирует ответ на основе запроса.
        
        Args:
            query: Текст запроса
            contexts: Список контекстов (игнорируется, generation сам делает поиск)
            top_k: Количество документов для использования в контексте (по умолчанию 5 для eval)
        
        Returns:
            Кортеж (answer, sources_info, latency_ms)
            - answer: сгенерированный ответ
            - sources_info: список словарей с информацией об источниках [{"doc_id": ..., "metadata": {...}}]
            - latency_ms: задержка в миллисекундах (время от начала запроса до получения ответа от inference endpoint)
        """
        start_time = time.time()
        
        if self.use_mock:
            answer, sources_info = self._mock_generate(query, contexts or [])
        else:
            answer, sources_info = await self._real_generate_async(query, contexts or [], top_k=top_k)
        
        # Измеряем полное время генерации ответа (включая поиск документов и генерацию через LLM)
        latency_ms = (time.time() - start_time) * 1000
        
        logger.debug(
            f"Сгенерирован ответ за {latency_ms:.2f}ms "
            f"(query_length={len(query)}, n_sources={len(sources_info)}, top_k={top_k})"
        )
        
        return answer, sources_info, latency_ms
    
    def generate(
        self, 
        query: str, 
        contexts: List[str] = None,
        top_k: int = 5
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Генерирует ответ на основе запроса.
        
        Args:
            query: Текст запроса
            contexts: Список контекстов (игнорируется, generation сам делает поиск)
            top_k: Количество документов для использования в контексте (по умолчанию 5 для eval)
        
        Returns:
            Кортеж (answer, sources_info, latency_ms)
            - answer: сгенерированный ответ
            - sources_info: список словарей с информацией об источниках [{"doc_id": ..., "metadata": {...}}]
            - latency_ms: задержка в миллисекундах (время от начала запроса до получения ответа от inference endpoint)
        """
        start_time = time.time()
        
        if self.use_mock:
            answer, sources_info = self._mock_generate(query, contexts or [])
        else:
            answer, sources_info = self._real_generate(query, contexts or [], top_k=top_k)
        
        # Измеряем полное время генерации ответа (включая поиск документов и генерацию через LLM)
        latency_ms = (time.time() - start_time) * 1000
        
        logger.debug(
            f"Сгенерирован ответ за {latency_ms:.2f}ms "
            f"(query_length={len(query)}, n_sources={len(sources_info)}, top_k={top_k})"
        )
        
        return answer, sources_info, latency_ms
    
    def _mock_generate(self, query: str, contexts: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Mock генерация ответа.
        
        Args:
            query: Текст запроса
            contexts: Список контекстов (игнорируется)
            
        Returns:
            Кортеж (answer, sources_info)
        """
        answer = f"MOCK_ANSWER: {query[:100]}..."
        
        # Mock sources_info: пустой список, т.к. mock не использует реальные источники
        sources_info = []
        
        return answer, sources_info
    
    async def _real_generate_async(self, query: str, contexts: List[str], top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Асинхронная реальная генерация через API endpoint с повторными попытками.
        
        Args:
            query: Текст запроса
            contexts: Список контекстов (игнорируется, generation сам делает поиск)
            top_k: Количество документов для использования в контексте
            
        Returns:
            Кортеж (answer, sources_info)
            - answer: сгенерированный ответ
            - sources_info: список словарей [{"doc_id": ..., "metadata": {...}}]
            
        Raises:
            httpx.RequestError: при ошибке запроса после всех попыток
        """
        payload = {
            "query": query,
            "top_k": top_k,  # Ограничиваем количество документов для eval
            "session_id": None  # Явно отключаем память для каждого запроса
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        max_retries = 5
        last_exception = None
        
        # Создаем клиент, если его еще нет
        if not self._client:
            # Увеличиваем лимиты для параллельных запросов
            limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
            self._client = httpx.AsyncClient(timeout=self.timeout, limits=limits)
        
        for attempt in range(1, max_retries + 1):
            try:
                response = await self._client.post(
                    self.endpoint_url,
                    json=payload,
                    headers=headers
                )
                # Проверяем статус код
                response.raise_for_status()
                
                result = response.json()
                
                answer = result.get("answer", "")
                
                # Generation API возвращает sources (список SourceInfo с doc_id и metadata)
                sources = result.get("sources", [])
                sources_info = []
                
                if sources:
                    for src in sources:
                        if isinstance(src, dict):
                            doc_id = src.get("doc_id", "")
                            metadata = src.get("metadata", {})
                        else:
                            # Если это объект SourceInfo
                            doc_id = getattr(src, "doc_id", "")
                            metadata = getattr(src, "metadata", {})
                        
                        sources_info.append({
                            "doc_id": str(doc_id),
                            "metadata": metadata if isinstance(metadata, dict) else {}
                        })
                
                if not answer:
                    logger.warning(f"Endpoint вернул пустой answer (попытка {attempt}/{max_retries})")
                    if attempt < max_retries:
                        await asyncio.sleep(1)  # Небольшая задержка перед повтором
                        continue
                    else:
                        raise ValueError("Endpoint вернул пустой answer после всех попыток")
                
                return answer, sources_info
                
            except (httpx.RequestError, httpx.HTTPStatusError, Exception) as e:
                last_exception = e
                logger.warning(
                    f"Ошибка при вызове inference endpoint (попытка {attempt}/{max_retries}): {e}"
                )
                
                if attempt < max_retries:
                    # Задержка между попытками: 5s, 10s, 20s, 30s
                    delays = [5, 10, 20, 30]
                    delay = delays[min(attempt - 1, len(delays) - 1)]
                    logger.info(f"Повторная попытка через {delay} секунд...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Все {max_retries} попыток неудачны, выбрасывается исключение")
                    raise last_exception
        
        # Этот код не должен выполняться, но на всякий случай
        if last_exception:
            raise last_exception
        raise httpx.RequestError("Неизвестная ошибка")
    
    def _real_generate(self, query: str, contexts: List[str], top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Синхронная обертка для _real_generate_async (для обратной совместимости).
        """
        return asyncio.run(self._real_generate_async(query, contexts, top_k))

