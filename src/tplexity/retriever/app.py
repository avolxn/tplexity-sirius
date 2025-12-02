import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tplexity.retriever.api import router
from tplexity.retriever.api.dependencies import get_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения

    Запускается при старте и остановке приложения
    """
    logger.info("🚀 [retriever][app] Запуск Retriever микросервиса")
    logger.info("🔄 [retriever][app] Инициализация RetrieverService и загрузка моделей...")
    get_retriever()
    logger.info("✅ [retriever][app] RetrieverService инициализирован, все модели загружены")
    yield
    logger.info("🛑 [retriever][app] Остановка Retriever микросервиса")


app = FastAPI(
    title="Retriever API",
    description="Микросервис для гибридного поиска по документам с возможностью добавления, получения и удаления документов",
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
        "service": "Retriever API",
        "version": "1.0.0",
        "endpoints": {
            "add_documents": "POST /retriever/documents - Добавить документы",
            "get_documents": "POST /retriever/documents/get - Получить документы по ID",
            "get_all_documents": "GET /retriever/documents/all - Получить все документы",
            "search": "POST /retriever/search - Поиск документов",
            "delete_documents": "DELETE /retriever/documents - Удалить документы",
            "delete_all_documents": "DELETE /retriever/documents/all - Удалить все документы",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Swagger UI",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "tplexity.retriever.app:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
        log_level="info",
    )
