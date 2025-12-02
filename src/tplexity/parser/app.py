import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tplexity.parser.api import router
from tplexity.parser.api.dependencies import (
    get_config,
    set_monitoring_status,
    set_service,
)
from tplexity.parser.monitor_service import TelegramMonitorService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def start_monitoring_automatically():
    """Автоматически запускает мониторинг при старте приложения"""
    try:
        config = get_config()
        channels_list = config.get_channels_list() if config else []

        if not config or not channels_list:
            logger.warning("⚠️ [parser][app] Конфигурация не загружена или список каналов пуст, мониторинг не запущен")
            return

        if not config.api_id or not config.api_hash:
            logger.warning("⚠️ [parser][app] Не указаны api_id или api_hash, мониторинг не запущен")
            return

        logger.info("🔄 [parser][app] Автоматический запуск мониторинга...")

        service = TelegramMonitorService(
            api_id=config.api_id,
            api_hash=config.api_hash,
            channels=channels_list,
            session_name=config.session_name,
            data_dir=config.data_dir,
            webhook_url=config.webhook_url,
            retry_interval=config.retry_interval,
            session_string=config.session_string,
            llm_provider=config.llm_provider,
            qdrant_host=config.qdrant_host,
            qdrant_port=config.qdrant_port,
            qdrant_api_key=config.qdrant_api_key,
            qdrant_collection_name=config.qdrant_collection_name,
            qdrant_timeout=config.qdrant_timeout,
        )

        await service.initialize()
        set_service(service)

        asyncio.create_task(service.start_monitoring())
        set_monitoring_status(True)
        logger.info(f"✅ [parser][app] Мониторинг автоматически запущен для {len(channels_list)} каналов")
    except Exception as e:
        logger.error(f"❌ [parser][app] Ошибка при автоматическом запуске мониторинга: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения

    Запускается при старте и остановке приложения
    """
    logger.info("🚀 [parser][app] Запуск Telegram Parser микросервиса")

    await start_monitoring_automatically()

    yield

    logger.info("🛑 [parser][app] Остановка Telegram Parser микросервиса")


app = FastAPI(
    title="Telegram Parser API",
    description="Микросервис для мониторинга Telegram каналов и чанкирования постов",
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

app.include_router(router)


@app.get("/health")
async def health_check():
    """Health check эндпоинт"""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Информация о сервисе"""
    return {
        "service": "Telegram Parser API",
        "version": "1.0.0",
        "endpoints": {
            "download": "POST /download - Скачать последние n сообщений из каналов",
            "start": "POST /start - Запустить мониторинг",
            "stop": "POST /stop - Остановить мониторинг",
            "status": "GET /status - Статус сервиса",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Swagger UI",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "tplexity.parser.app:app",
        host="0.0.0.0",
        port=8011,
        reload=True,
        log_level="info",
    )
