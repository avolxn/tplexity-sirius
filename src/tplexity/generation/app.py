import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tplexity.generation.api import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения

    Запускается при старте и остановке приложения
    """
    from tplexity.generation.api.dependencies import get_generation

    logger.info("🚀 [generation][app] Запуск Generation микросервиса")
    yield
    logger.info("🛑 [generation][app] Остановка Generation микросервиса")

    try:
        generation_service = get_generation()
        await generation_service.close()
        logger.info("✅ [generation][app] Соединения закрыты")
    except Exception as e:
        logger.error(f"❌ [generation][app] Ошибка при закрытии соединений: {e}")


app = FastAPI(
    title="Generation API",
    description="Микросервис для генерации ответов с использованием RAG",
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
        "service": "Generation API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "POST /generation/generate - Генерация ответа с RAG",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Swagger UI",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "tplexity.generation.app:app",
        host="0.0.0.0",
        port=8012,
        reload=True,
        log_level="info",
    )
