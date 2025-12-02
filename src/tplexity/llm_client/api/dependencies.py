import logging

from tplexity.llm_client.client import LLMClient, get_llm

logger = logging.getLogger(__name__)


def get_llm_client(provider: str) -> LLMClient:
    """
    Получить LLM клиент для указанного провайдера

    Args:
        provider: Провайдер LLM

    Returns:
        LLMClient: Экземпляр LLM клиента
    """
    return get_llm(provider)
