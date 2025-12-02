import asyncio
import logging
from datetime import datetime

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointIdsList

logger = logging.getLogger(__name__)


class PostDeletionService:
    """Сервис для удаления устаревших постов из Qdrant"""

    def __init__(
        self,
        qdrant_host: str,
        qdrant_port: int,
        qdrant_api_key: str | None,
        qdrant_collection_name: str,
        qdrant_timeout: int = 60,
    ):
        """
        Инициализация сервиса удаления постов

        Args:
            qdrant_host: Хост Qdrant
            qdrant_port: Порт Qdrant
            qdrant_api_key: API ключ для Qdrant
            qdrant_collection_name: Имя коллекции в Qdrant
            qdrant_timeout: Таймаут для подключения
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_api_key = qdrant_api_key
        self.qdrant_collection_name = qdrant_collection_name
        self.qdrant_timeout = qdrant_timeout

        logger.info("🔄 [post_deletion_service] Инициализация сервиса удаления постов")
        try:
            self.client = AsyncQdrantClient(
                url=f"https://{self.qdrant_host}:{self.qdrant_port}",
                api_key=self.qdrant_api_key,
                timeout=self.qdrant_timeout,
            )
            logger.info(
                f"✅ [post_deletion_service] Клиент Qdrant инициализирован: {self.qdrant_host}:{self.qdrant_port}"
            )
        except Exception as e:
            logger.error(f"❌ [post_deletion_service] Ошибка инициализации клиента Qdrant: {e}")
            raise

    async def delete_expired_posts(self) -> int:
        """
        Удаляет посты с delete_date <= сегодня из Qdrant

        Returns:
            int: Количество удаленных постов

        Примечание:
            Удаляются только посты, у которых есть метаданные delete_date.
            Посты без delete_date игнорируются.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"🗑️ [post_deletion_service] Запуск удаления устаревших постов (дата удаления <= {today})")

        try:
            expired_post_ids = []
            offset = None
            limit = 100

            while True:
                points, next_offset = await self.client.scroll(
                    collection_name=self.qdrant_collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                )

                if not points:
                    break

                for point in points:
                    payload = point.payload or {}
                    delete_date_str = payload.get("delete_date")

                    if not delete_date_str:
                        continue

                    try:
                        if delete_date_str <= today:
                            expired_post_ids.append(str(point.id))
                    except (TypeError, ValueError) as e:
                        logger.debug(f"⚠️ [post_deletion_service] Не удалось сравнить дату {delete_date_str}: {e}")

                if next_offset is None:
                    break
                offset = next_offset

            if not expired_post_ids:
                logger.info("✅ [post_deletion_service] Устаревших постов не найдено")
                return 0

            logger.info(f"📊 [post_deletion_service] Найдено {len(expired_post_ids)} постов для удаления")

            deleted_count = await self._delete_with_retry(expired_post_ids)

            logger.info(
                f"✅ [post_deletion_service] Удаление завершено: удалено {deleted_count} из {len(expired_post_ids)} постов"
            )
            return deleted_count

        except Exception as e:
            logger.error(
                f"❌ [post_deletion_service] Ошибка при удалении устаревших постов: {e}",
                exc_info=True,
            )
            raise

    async def _delete_with_retry(self, post_ids: list[str]) -> int:
        """
        Удаляет посты с механизмом повторных попыток

        Args:
            post_ids: Список ID постов для удаления

        Returns:
            int: Количество успешно удаленных постов
        """
        max_retries = 100
        retry_interval = 20

        remaining_post_ids = post_ids.copy()
        total_deleted = 0

        for attempt in range(1, max_retries + 1):
            if not remaining_post_ids:
                logger.info(f"✅ [post_deletion_service] Все посты успешно удалены. Всего попыток: {attempt - 1}")
                return total_deleted

            try:
                batch_size = 100
                deleted_in_attempt = 0
                failed_batches = []

                for i in range(0, len(remaining_post_ids), batch_size):
                    batch = remaining_post_ids[i : i + batch_size]
                    try:
                        await self.client.delete(
                            collection_name=self.qdrant_collection_name,
                            points_selector=PointIdsList(points=batch),
                        )
                        deleted_in_attempt += len(batch)
                        total_deleted += len(batch)
                        logger.debug(
                            f"✅ [post_deletion_service] Удален батч из {len(batch)} постов (попытка {attempt})"
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ [post_deletion_service] Ошибка при удалении батча на попытке {attempt}: {e}")

                        failed_batches.extend(batch)

                remaining_post_ids = failed_batches

                if not remaining_post_ids:
                    logger.info(f"✅ [post_deletion_service] Все посты успешно удалены с попытки {attempt}")
                    return total_deleted
                else:
                    logger.warning(
                        f"⚠️ [post_deletion_service] Удалено {deleted_in_attempt} из {len(post_ids)} постов "
                        f"на попытке {attempt}. Осталось {len(remaining_post_ids)} постов"
                    )

                    if attempt < max_retries:
                        logger.info(
                            f"🔄 [post_deletion_service] Повторная попытка через {retry_interval} секунд... "
                            f"(попытка {attempt}/{max_retries})"
                        )
                        await asyncio.sleep(retry_interval)
                    else:
                        logger.error(
                            f"❌ [post_deletion_service] Достигнуто максимальное количество попыток ({max_retries}). "
                            f"Не удалось удалить {len(remaining_post_ids)} постов"
                        )
                        return total_deleted

            except Exception as e:
                logger.error(
                    f"❌ [post_deletion_service] Критическая ошибка при удалении на попытке {attempt}/{max_retries}: {e}"
                )

                if attempt < max_retries:
                    logger.info(f"🔄 [post_deletion_service] Повторная попытка через {retry_interval} секунд...")
                    await asyncio.sleep(retry_interval)
                else:
                    logger.error(
                        f"❌ [post_deletion_service] Достигнуто максимальное количество попыток ({max_retries})"
                    )
                    return total_deleted

        return total_deleted
