import logging

from aiogram import Bot, Dispatcher

from tplexity.bot.config import settings
from tplexity.bot.service_client import GenerationClient

logger = logging.getLogger(__name__)


_bot_instance: Bot | None = None
_dp_instance: Dispatcher | None = None
_generation_client_instance: GenerationClient | None = None


def get_bot() -> Bot:
    """
    Получить экземпляр Bot (singleton)

    Returns:
        Bot: Экземпляр Bot
    """
    global _bot_instance

    if _bot_instance is None:
        bot_token = settings.bot_token

        if not bot_token:
            raise ValueError("BOT_TOKEN не установлен в .env файле")

        _bot_instance = Bot(token=bot_token)
        logger.info("✅ [bot][dependencies] Telegram Bot создан")

    return _bot_instance


def get_dispatcher() -> Dispatcher:
    """
    Получить экземпляр Dispatcher (singleton)

    Returns:
        Dispatcher: Экземпляр Dispatcher
    """
    global _dp_instance

    if _dp_instance is None:
        _dp_instance = Dispatcher()
        logger.info("✅ [bot][dependencies] Dispatcher создан")

    return _dp_instance


def get_generation_client() -> GenerationClient:
    """
    Получить экземпляр GenerationClient (singleton)

    Returns:
        GenerationClient: Экземпляр GenerationClient
    """
    global _generation_client_instance

    if _generation_client_instance is None:
        _generation_client_instance = GenerationClient()
        logger.info("✅ [bot][dependencies] GenerationClient создан")

    return _generation_client_instance
