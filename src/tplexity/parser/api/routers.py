import logging
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from tplexity.parser.api.dependencies import (
    get_config,
    get_monitoring_status,
    get_service,
    reset_service,
    set_monitoring_status,
    set_service,
)
from tplexity.parser.api.schemas import (
    DownloadMessagesResponse,
    HealthResponse,
    RootResponse,
    StartMonitoringResponse,
    StatusResponse,
    StopMonitoringResponse,
)
from tplexity.parser.config import Settings
from tplexity.parser.monitor_service import TelegramMonitorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["telegram-monitor"])


@router.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    """
    Информация о микросервисе

    Returns:
        RootResponse: Описание сервиса и эндпоинтов
    """
    return RootResponse(
        service="Telegram Monitor & Chunker",
        version="1.0.0",
        endpoints={
            "download": "POST /download - Скачать последние n сообщений из каналов",
            "start": "POST /start - Запустить мониторинг",
            "stop": "POST /stop - Остановить мониторинг",
            "status": "GET /status - Статус сервиса",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Swagger UI",
        },
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check эндпоинт

    Returns:
        HealthResponse: Статус здоровья сервиса
    """
    return HealthResponse(status="healthy")


@router.get("/status", response_model=StatusResponse)
async def get_status(
    config: Settings = Depends(get_config),
) -> StatusResponse:
    """
    Получить статус сервиса

    Args:
        config: Конфигурация сервиса

    Returns:
        StatusResponse: Текущий статус и конфигурация
    """
    is_monitoring = get_monitoring_status()
    return StatusResponse(
        status="running" if is_monitoring else "stopped",
        config=config.model_dump() if config else None,
        timestamp=datetime.now(UTC).isoformat(),
    )


@router.post("/start", response_model=StartMonitoringResponse)
async def start_monitoring(
    background_tasks: BackgroundTasks,
    config: Settings = Depends(get_config),
) -> StartMonitoringResponse:
    """
    Запустить мониторинг Telegram каналов

    Процесс:
    1. Проверка что мониторинг не запущен
    2. Валидация конфигурации
    3. Инициализация сервиса
    4. Запуск мониторинга в фоне

    Args:
        background_tasks: FastAPI background tasks
        config: Конфигурация сервиса

    Returns:
        StartMonitoringResponse: Статус запуска

    Raises:
        HTTPException: 400 если конфигурация неверная, 409 если уже запущен, 500 при ошибке
    """
    is_monitoring = get_monitoring_status()

    if is_monitoring:
        logger.warning("⚠️ [parser][routers] Мониторинг уже запущен")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Мониторинг уже запущен",
        )

    channels_list = config.get_channels_list() if config else []

    if not config or not channels_list:
        logger.error("❌ [parser][routers] Конфигурация не загружена или список каналов пуст")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Конфигурация не загружена или список каналов пуст",
        )

    if not config.api_id or not config.api_hash:
        logger.error("❌ [parser][routers] Не указаны api_id или api_hash в конфигурации")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не указаны api_id или api_hash в конфигурации",
        )

    try:
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
        background_tasks.add_task(service.start_monitoring)
        set_monitoring_status(True)

        logger.info(f"✅ [parser][routers] Мониторинг запущен для каналов: {channels_list}")
        return StartMonitoringResponse(
            status="started",
            channels=config.channels,
        )
    except ValueError as e:
        logger.error(f"❌ [parser][routers] Ошибка валидации: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"❌ [parser][routers] Ошибка при запуске мониторинга: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при запуске мониторинга: {str(e)}",
        ) from e


@router.post("/stop", response_model=StopMonitoringResponse)
async def stop_monitoring() -> StopMonitoringResponse:
    """
    Остановить мониторинг

    Returns:
        StopMonitoringResponse: Статус остановки

    Raises:
        HTTPException: 409 если мониторинг не запущен
    """
    is_monitoring = get_monitoring_status()
    service = get_service()

    if not is_monitoring or not service:
        logger.warning("⚠️ [parser][routers] Мониторинг не запущен")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Мониторинг не запущен",
        )

    try:
        await service.stop_monitoring()
        reset_service()

        logger.info("✅ [parser][routers] Мониторинг остановлен")
        return StopMonitoringResponse(status="stopped")
    except Exception as e:
        logger.error(f"❌ [parser][routers] Ошибка при остановке мониторинга: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при остановке мониторинга: {str(e)}",
        ) from e


@router.post("/download", response_model=DownloadMessagesResponse)
async def download_messages(
    config: Settings = Depends(get_config),
) -> DownloadMessagesResponse:
    """
    Скачать все доступные сообщения из каждого канала

    Процесс:
    - Скачивает все доступные сообщения из каждого канала (без ограничений)
    - Автоматически удаляет пустые сообщения (где поле text пустое)
    - Сохраняет результаты в data/telegram/[канал]/messages_monitor.json

    Примечание: Для загрузки исторических данных используйте скрипт load_historical_posts.py

    Args:
        config: Конфигурация сервиса

    Returns:
        DownloadMessagesResponse: Статистика по скачанным и сохраненным сообщениям

    Raises:
        HTTPException: 400 если конфигурация неверная, 500 при ошибке
    """
    channels_list = config.get_channels_list() if config else []

    if not config or not channels_list:
        logger.error("❌ [parser][routers] Конфигурация не загружена или список каналов пуст")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Конфигурация не загружена или список каналов пуст",
        )

    if not config.api_id or not config.api_hash:
        logger.error("❌ [parser][routers] Не указаны api_id или api_hash в конфигурации")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не указаны api_id или api_hash в конфигурации",
        )

    try:
        service = get_service()
        if not service:
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

        results = await service.download_initial_messages()

        logger.info(
            f"✅ [parser][routers] Скачивание завершено: "
            f"скачано {results['total_downloaded']}, сохранено {results['total_saved']}"
        )
        return DownloadMessagesResponse(
            status="success",
            limit_per_channel=None,
            results=results,
        )

    except ValueError as e:
        logger.error(f"❌ [parser][routers] Ошибка валидации: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"❌ [parser][routers] Ошибка при скачивании: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при скачивании: {str(e)}",
        ) from e
