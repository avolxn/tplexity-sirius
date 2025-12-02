import asyncio
import hashlib
import logging
from dataclasses import dataclass, field

from tplexity.parser.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Запрос к LLM с результатом"""

    post_text: str
    llm_provider: str
    future: asyncio.Future = field(default_factory=asyncio.Future)
    cache_key: str = field(default="")

    def __post_init__(self):
        """Вычисляем cache_key после инициализации"""
        if not self.cache_key:
            text_hash = hashlib.md5(self.post_text.encode()).hexdigest() # noqa: S324
            self.cache_key = f"{self.llm_provider}:{text_hash}"


class LLMBatcher:
    """
    Батчер для группировки LLM-запросов.

    Собирает запросы в батчи и обрабатывает их асинхронно.
    Поддерживает кэширование результатов.
    """

    def __init__(
        self,
        batch_size: int = 5,
        batch_timeout: float = 0.5,
        max_cache_size: int = 1000,
        llm_provider: str = "qwen",
        llm_client: LLMClient | None = None,
    ):
        """
        Инициализация батчера

        Args:
            batch_size: Максимальный размер батча
            batch_timeout: Максимальное время ожидания заполнения батча (секунды)
            max_cache_size: Максимальный размер кэша
            llm_provider: Провайдер LLM по умолчанию
            llm_client: HTTP клиент для LLM (если None, создается новый)
        """
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_cache_size = max_cache_size
        self.default_llm_provider = llm_provider
        self.llm_client = llm_client or get_llm_client()

        self.queue: asyncio.Queue[LLMRequest] = asyncio.Queue()

        self.cache: dict[str, tuple[int, str]] = {}

        self.is_running = False

        self.batch_task: asyncio.Task | None = None

    async def start(self):
        """Запускает фоновую задачу обработки батчей"""
        if self.is_running:
            logger.warning("⚠️ [llm_batcher] Батчер уже запущен")
            return

        self.is_running = True
        self.batch_task = asyncio.create_task(self._batch_processor())
        logger.info(
            f"✅ [llm_batcher] Батчер запущен (batch_size={self.batch_size}, batch_timeout={self.batch_timeout}s)"
        )

    async def stop(self):
        """Останавливает батчер"""
        self.is_running = False
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 [llm_batcher] Батчер остановлен")

    async def determine_relevance_days(self, post_text: str, llm_provider: str | None = None) -> tuple[int, str]:
        """
        Определяет количество дней актуальности поста через LLM с батчингом

        Args:
            post_text: Текст поста для анализа
            llm_provider: Провайдер LLM (если None, используется default)

        Returns:
            tuple[int, str]: (Количество дней актуальности, сырой ответ LLM)
        """
        provider = llm_provider or self.default_llm_provider

        cache_key = f"{provider}:{hashlib.md5(post_text.encode()).hexdigest()}" # noqa: S324
        if cache_key in self.cache:
            relevance_days, raw_response = self.cache[cache_key]
            logger.debug(f"💾 [llm_batcher] Результат из кэша для поста (длина: {len(post_text)} символов)")
            return relevance_days, raw_response

        request = LLMRequest(post_text=post_text, llm_provider=provider, cache_key=cache_key)

        await self.queue.put(request)

        try:
            relevance_days, raw_response = await asyncio.wait_for(request.future, timeout=30.0)

            self._add_to_cache(cache_key, relevance_days, raw_response)
            return relevance_days, raw_response
        except TimeoutError:
            logger.error(f"❌ [llm_batcher] Таймаут ожидания результата для поста (длина: {len(post_text)} символов)")

            return 30, "TIMEOUT"

    async def _batch_processor(self):
        """Фоновая задача для обработки батчей"""
        logger.info("🔄 [llm_batcher] Запущена обработка батчей")

        while self.is_running:
            try:
                batch = await self._collect_batch()

                if not batch:
                    continue

                await self._process_batch(batch)

            except asyncio.CancelledError:
                logger.info("🛑 [llm_batcher] Обработка батчей остановлена")
                break
            except Exception as e:
                logger.error(f"❌ [llm_batcher] Ошибка в обработке батчей: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _collect_batch(self) -> list[LLMRequest]:
        """
        Собирает батч запросов из очереди

        Returns:
            Список запросов для обработки
        """
        batch: list[LLMRequest] = []
        first_request = None

        try:
            first_request = await asyncio.wait_for(self.queue.get(), timeout=self.batch_timeout)
            batch.append(first_request)
        except TimeoutError:
            return []

        batch_timeout = self.batch_timeout
        while len(batch) < self.batch_size:
            try:
                request = await asyncio.wait_for(self.queue.get(), timeout=batch_timeout)
                batch.append(request)

                batch_timeout = 0.1
            except TimeoutError:
                break

        logger.debug(f"📦 [llm_batcher] Собран батч из {len(batch)} запросов")
        return batch

    async def _process_batch(self, batch: list[LLMRequest]):
        """
        Обрабатывает батч запросов

        Args:
            batch: Список запросов для обработки
        """
        if not batch:
            return

        requests_by_provider: dict[str, list[LLMRequest]] = {}
        for request in batch:
            provider = request.llm_provider
            if provider not in requests_by_provider:
                requests_by_provider[provider] = []
            requests_by_provider[provider].append(request)

        tasks = []
        for provider, provider_requests in requests_by_provider.items():
            task = self._process_provider_batch(provider, provider_requests)
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_provider_batch(self, provider: str, requests: list[LLMRequest]):
        """
        Обрабатывает батч запросов для одного провайдера

        Args:
            provider: Провайдер LLM
            requests: Список запросов
        """
        try:
            tasks = []
            for request in requests:
                task = self._process_single_request(request, provider)
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"❌ [llm_batcher] Ошибка при обработке батча для провайдера {provider}: {e}", exc_info=True)

            for request in requests:
                if not request.future.done():
                    request.future.set_exception(e)

    async def _process_single_request(self, request: LLMRequest, provider: str):
        """
        Обрабатывает один запрос к LLM

        Args:
            request: Запрос
            provider: Провайдер LLM
        """
        try:
            from tplexity.parser.relevance_analyzer import RELEVANCE_PROMPT

            messages = [
                {
                    "role": "user",
                    "content": RELEVANCE_PROMPT.format(post_text=request.post_text),
                }
            ]

            raw_response = await self.llm_client.generate(
                provider=provider,
                messages=messages,
                temperature=0.0,
                max_tokens=50,
            )

            response = raw_response.strip()
            digits = ""
            for char in response:
                if char.isdigit():
                    digits += char
                elif digits:
                    break

            if not digits:
                logger.warning(
                    f"⚠️ [llm_batcher] Не удалось извлечь число из ответа LLM: {response}, "
                    f"используем значение по умолчанию 30"
                )
                relevance_days = 30
            else:
                relevance_days = int(digits)

                relevance_days = max(1, min(10000, relevance_days))

            if not request.future.done():
                request.future.set_result((relevance_days, raw_response))

            logger.debug(
                f"✅ [llm_batcher] Определена актуальность: {relevance_days} дней "
                f"для поста (длина: {len(request.post_text)} символов)"
            )

        except Exception as e:
            logger.error(
                f"❌ [llm_batcher] Ошибка при обработке запроса: {e}",
                exc_info=True,
            )

            if not request.future.done():
                request.future.set_result((30, f"ERROR: {str(e)}"))

    def _add_to_cache(self, cache_key: str, relevance_days: int, raw_response: str):
        """
        Добавляет результат в кэш с ограничением размера

        Args:
            cache_key: Ключ кэша
            relevance_days: Количество дней актуальности
            raw_response: Сырой ответ LLM
        """

        if len(self.cache) >= self.max_cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = (relevance_days, raw_response)


_batcher_instance: LLMBatcher | None = None


def get_batcher(llm_provider: str = "qwen", llm_client: LLMClient | None = None) -> LLMBatcher:
    """
    Получить глобальный экземпляр батчера (singleton)

    Args:
        llm_provider: Провайдер LLM по умолчанию
        llm_client: HTTP клиент для LLM (если None, создается новый)

    Returns:
        LLMBatcher: Экземпляр батчера

    Примечание:
        Батчер нужно запустить вручную через await batcher.start()
        после получения экземпляра
    """
    global _batcher_instance

    if _batcher_instance is None:
        _batcher_instance = LLMBatcher(llm_provider=llm_provider, llm_client=llm_client)

    return _batcher_instance
