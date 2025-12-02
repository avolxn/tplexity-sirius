import logging

from openai import AsyncOpenAI

from tplexity.llm_client.config import settings

logger = logging.getLogger(__name__)


_llm_instances: dict[str, "LLMClient"] = {}


class LLMClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        **kwargs,
    ):
        """
        Инициализация LLM клиента

        Args:
            model: Название модели
            api_key: API ключ
            base_url: Базовый URL для API (если None, используется стандартный OpenAI API)
            **kwargs: Дополнительные параметры для AsyncOpenAI (например, default_headers={"x-folder-id": "..."})
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            **kwargs,
        )

        logger.info(f"✅ [llm_client] LLM клиент инициализирован: model={model}, base_url={base_url}")

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Генерация ответа через LLM

        Args:
            messages (list[dict[str, str]]): Список сообщений в формате OpenAI
                Пример: [
                    {"role": "system", "content": "Ты - помощник"},
                    {"role": "user", "content": "Привет!"}
                ]
            temperature (float | None): Температура генерации (если None, используется из settings.llm.temperature)
            max_tokens (int | None): Максимальное количество токенов (если None, используется из settings.llm.max_tokens)

        Returns:
            str: Сгенерированный ответ

        Raises:
            Exception: При ошибке вызова LLM API
        """
        temperature = temperature or settings.temperature
        max_tokens = max_tokens or settings.max_tokens

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = response.choices[0].message.content
            logger.debug(f"✅ [llm_client] Ответ получен от LLM (model={self.model})")
            return answer
        except Exception as e:
            logger.error(f"❌ [llm_client] Ошибка при вызове LLM: {e}")
            raise


def get_llm(provider: str) -> LLMClient:
    """
    Получить LLM клиент для указанного провайдера (singleton)

    Args:
        provider (str): Провайдер LLM

    Returns:
        LLMClient: Экземпляр LLM клиента для указанного провайдера
    """
    global _llm_instances

    if provider in _llm_instances:
        return _llm_instances[provider]

    if provider == "qwen":
        client = LLMClient(
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
        )
    elif provider == "yandexgpt":
        model_name = f"gpt://{settings.yandexgpt_folder_id}/{settings.yandexgpt_model}"
        client = LLMClient(
            model=model_name,
            api_key=settings.yandexgpt_api_key,
            base_url=settings.yandexgpt_base_url,
            default_headers={"x-folder-id": settings.yandexgpt_folder_id},
        )
    elif provider == "chatgpt":
        client = LLMClient(
            model=settings.chatgpt_model,
            api_key=settings.chatgpt_api_key,
            base_url=None,
        )
    elif provider == "deepseek":
        client = LLMClient(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    else:
        raise ValueError(f"Неизвестный провайдер LLM: {provider}")

    _llm_instances[provider] = client
    return client
