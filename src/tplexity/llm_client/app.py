import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tplexity.llm_client.api import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения

    Запускается при старте и остановке приложения
    """
    logger.info("🚀 [llm_client][app] Запуск LLM Client микросервиса")
    yield
    logger.info("🛑 [llm_client][app] Остановка LLM Client микросервиса")


app = FastAPI(
    title="LLM Client API",
    description="Микросервис для работы с LLM провайдерами",
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
        "service": "LLM Client API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "POST /v1/llm/generate - Генерация ответа через LLM",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Swagger UI",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "tplexity.llm_client.app:app",
        host="0.0.0.0",
        port=8014,
        reload=True,
        log_level="info",
    )
