import logging

from tplexity.parser.config import settings
from tplexity.parser.monitor_service import TelegramMonitorService

logger = logging.getLogger(__name__)


_service_instance: TelegramMonitorService | None = None
_is_monitoring: bool = False


def get_config():
    """
    Получить конфигурацию сервиса (singleton)

    Returns:
        Settings: Конфигурация из settings
    """
    return settings


def get_service() -> TelegramMonitorService | None:
    """
    Получить экземпляр TelegramMonitorService (singleton)

    Returns:
        TelegramMonitorService | None: Экземпляр сервиса или None если не инициализирован
    """
    global _service_instance
    return _service_instance


def set_service(service: TelegramMonitorService):
    """
    Установить экземпляр TelegramMonitorService

    Args:
        service: Экземпляр сервиса
    """
    global _service_instance
    _service_instance = service


def get_monitoring_status() -> bool:
    """
    Получить статус мониторинга

    Returns:
        bool: True если мониторинг запущен
    """
    global _is_monitoring
    return _is_monitoring


def set_monitoring_status(status: bool):
    """
    Установить статус мониторинга

    Args:
        status: Статус мониторинга
    """
    global _is_monitoring
    _is_monitoring = status


def reset_service():
    """Сбросить сервис и статус мониторинга"""
    global _service_instance, _is_monitoring
    _service_instance = None
    _is_monitoring = False
