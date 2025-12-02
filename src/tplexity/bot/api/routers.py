import logging

from aiogram import Bot
from fastapi import APIRouter, Depends

from tplexity.bot.api.dependencies import get_bot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/bot", tags=["bot"])


@router.get("/status")
async def status(
    bot: Bot = Depends(get_bot),
) -> dict:
    """
    Статус бота

    Args:
        bot: Экземпляр Bot

    Returns:
        dict: Статус бота
    """
    try:
        bot_info = await bot.get_me()
        return {
            "status": "running",
            "bot_username": bot_info.username,
            "bot_id": bot_info.id,
            "bot_name": bot_info.first_name,
        }
    except Exception as e:
        logger.error(f"[bot][routers] Ошибка при получении статуса бота: {e}")
        return {
            "status": "error",
            "error": str(e),
        }
