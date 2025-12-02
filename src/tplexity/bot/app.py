import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tplexity.bot.api import router as bot_router

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    import asyncio

    from tplexity.bot.api.dependencies import get_bot, get_dispatcher, get_generation_client
    from tplexity.bot.bot import register_handlers, start_polling

    logger.info("🚀 [bot][app] Запуск микросервиса")

    bot = get_bot()
    dp = get_dispatcher()
    generation_client = get_generation_client()

    register_handlers(dp, generation_client)

    polling_task = asyncio.create_task(start_polling(bot, dp))

    yield

    logger.info("🛑 [bot][app] Остановка микросервиса")
    polling_task.cancel()
    try:
        await polling_task
    except asyncio.CancelledError:
        pass

    await generation_client.close()
    await bot.session.close()
    logger.info("[bot][app] Соединение с Generation API закрыто")


app = FastAPI(
    title="Telegram Bot Service API",
    description="Микросервис для Telegram бота с интеграцией Generation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(bot_router)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """
    Health check эндпоинт

    Returns:
        dict: Статус сервиса
    """
    return {"status": "healthy"}


@app.get("/", tags=["info"])
async def root():
    """Корневой эндпоинт с информацией о сервисе"""
    return {
        "service": "Telegram Bot Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "webhook": "/bot/webhook",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "tplexity.bot.app:app",
        host="0.0.0.0",
        port=8013,
        reload=True,
    )
