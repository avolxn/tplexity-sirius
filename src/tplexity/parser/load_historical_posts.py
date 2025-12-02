import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from tplexity.parser.config import settings
from tplexity.parser.telegram_downloader import TelegramDownloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def clear_database(retriever_url: str) -> bool:
    """
    Очищает БД, удаляя все документы.

    Args:
        retriever_url: URL retriever API

    Returns:
        True если успешно, False в противном случае
    """
    try:
        delete_url = f"{retriever_url.rstrip('/')}/v1/retriever/documents/all"
        logger.info(f"🗑️ [parser][load_historical_posts] Очистка БД: {delete_url}")

        async with httpx.AsyncClient() as client:
            response = await client.delete(delete_url, timeout=60.0)
            response.raise_for_status()

        logger.info("✅ [parser][load_historical_posts] БД успешно очищена")
        return True
    except Exception as e:
        logger.error(f"❌ [parser][load_historical_posts] Ошибка при очистке БД: {e}")
        return False


async def send_posts_to_retriever(
    posts: list[dict],
    channel: str,
    retriever_url: str,
    batch_size: int = 50,
    channel_titles: dict[str, str] | None = None,
) -> tuple[int, int]:
    """
    Отправляет посты в retriever (без чанкирования, полностью).

    Args:
        posts: Список постов для отправки
        channel: Название канала
        retriever_url: URL retriever API
        batch_size: Размер батча для отправки

    Returns:
        Кортеж (успешно отправлено, ошибок)
    """
    if not posts:
        return 0, 0

    documents_url = f"{retriever_url.rstrip('/')}/v1/retriever/documents"
    success_count = 0
    error_count = 0

    for i in range(0, len(posts), batch_size):
        batch = posts[i : i + batch_size]
        documents = []

        for post in batch:
            text = (post.get("text") or "").strip()
            if not text:
                continue

            date_str = post.get("date")
            if date_str:
                try:
                    if date_str.endswith("Z"):
                        date_str = date_str.replace("Z", "+00:00")

                    if "T" in date_str:
                        post_date = datetime.fromisoformat(date_str)
                    else:
                        post_date = datetime.fromisoformat(f"{date_str}T00:00:00")

                    formatted_date = post_date.strftime("%Y-%m-%d %H:%M:%S")
                    text = f"{text}\n\n{formatted_date}"
                except (ValueError, AttributeError) as e:
                    logger.debug(
                        f"⚠️ [parser][load_historical_posts] Не удалось распарсить дату: {date_str}, ошибка: {e}"
                    )

            metadata = {k: v for k, v in post.items() if k != "text"}
            metadata["channel_name"] = channel

            if channel_titles:
                channel_title = channel_titles.get(channel, channel)
                metadata["channel_title"] = channel_title
            else:
                metadata["channel_title"] = channel

            documents.append({"text": text, "metadata": metadata})

        if not documents:
            continue

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(documents_url, json={"documents": documents}, timeout=60.0)
                response.raise_for_status()
                success_count += len(documents)
                logger.info(
                    f"📤 [parser][load_historical_posts] Отправлено {len(documents)} постов из {channel} "
                    f"(батч {i // batch_size + 1}/{(len(posts) + batch_size - 1) // batch_size})"
                )
        except Exception as e:
            error_count += len(documents)
            logger.error(f"❌ [parser][load_historical_posts] Ошибка при отправке батча из {channel}: {e}")

    return success_count, error_count


async def load_historical_posts():
    """Основная функция для загрузки исторических постов."""
    logger.info("🚀 [parser][load_historical_posts] Запуск загрузки исторических постов")

    if not settings.api_id or not settings.api_hash:
        logger.error("❌ [parser][load_historical_posts] Не указаны API_ID или API_HASH в конфигурации")
        return

    channels_list = settings.get_channels_list()
    if not channels_list:
        logger.error("❌ [parser][load_historical_posts] Список каналов пуст")
        return

    if not settings.webhook_url:
        logger.error("❌ [parser][load_historical_posts] Не указан WEBHOOK_URL в конфигурации")
        return

    retriever_url = settings.webhook_url.rsplit("/retriever", 1)[0]
    logger.info(f"📡 [parser][load_historical_posts] Retriever URL: {retriever_url}")
    logger.info(f"📋 [parser][load_historical_posts] Каналы для обработки: {', '.join(channels_list)}")

    four_months_ago = datetime.now(UTC) - timedelta(days=120)
    logger.info(
        f"📅 [parser][load_historical_posts] Загружаем посты с {four_months_ago.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    if not await clear_database(retriever_url):
        logger.error("❌ [parser][load_historical_posts] Не удалось очистить БД, прерываем выполнение")
        return

    project_root = Path(__file__).parent.parent.parent.parent
    session_path = project_root / settings.session_name

    logger.info("=" * 60)
    logger.info("📋 [parser][load_historical_posts] Конфигурация подключения:")
    logger.info(f"   API_ID: {settings.api_id}")
    logger.info(f"   API_HASH: {'*' * 10 if settings.api_hash else 'None (не указан!)'}")
    logger.info(f"   SESSION_NAME: {settings.session_name}")
    logger.info(
        f"   TELEGRAM_SESSION_STRING: {'указан' if settings.session_string else 'не указан (будет использован файл)'}"
    )

    if settings.session_string:
        logger.info(
            f"🔑 [parser][load_historical_posts] Используется строка сессии (длина: {len(settings.session_string)} символов)"
        )
        logger.debug(
            f"🔑 [parser][load_historical_posts] Первые 20 символов session_string: {settings.session_string[:20]}..."
        )
    else:
        logger.info(f"📁 [parser][load_historical_posts] Используется файл сессии: {session_path}")
        if session_path.exists():
            logger.info(
                f"📁 [parser][load_historical_posts] Файл сессии существует, размер: {session_path.stat().st_size} байт"
            )
        else:
            logger.warning(f"⚠️ [parser][load_historical_posts] Файл сессии не найден: {session_path}")
            logger.warning(
                "💡 [parser][load_historical_posts] Для использования строки сессии добавьте TELEGRAM_SESSION_STRING в .env"
            )
            logger.warning(
                "💡 [parser][load_historical_posts] Или запустите: poetry run python src/tplexity/parser/authorize_telegram.py"
            )
    logger.info("=" * 60)

    logger.info("🔍 [parser][load_historical_posts] ПЕРЕД созданием TelegramDownloader:")
    logger.info(f"   settings.session_string type: {type(settings.session_string)}")
    logger.info(f"   settings.session_string value: {settings.session_string}")
    logger.info(f"   settings.session_string is None: {settings.session_string is None}")
    logger.info(f"   settings.session_string == '': {settings.session_string == ''}")
    if settings.session_string:
        logger.info(f"   settings.session_string.strip() == '': {settings.session_string.strip() == ''}")
        logger.info(f"   settings.session_string длина: {len(settings.session_string)}")

    logger.info("🔧 [parser][load_historical_posts] Создание TelegramDownloader...")
    downloader = TelegramDownloader(
        api_id=settings.api_id,
        api_hash=settings.api_hash,
        session_name=str(session_path),
        session_string=settings.session_string,
        download_path=str(project_root / settings.data_dir / "telegram"),
    )

    try:
        logger.info("🔌 [parser][load_historical_posts] Подключение к Telegram...")
        try:
            await downloader.client.connect()
            logger.info("✅ [parser][load_historical_posts] Соединение с Telegram установлено")
        except Exception as e:
            logger.error(f"❌ [parser][load_historical_posts] Ошибка при подключении к Telegram: {e}", exc_info=True)
            return

        logger.info("🔍 [parser][load_historical_posts] Проверка авторизации...")
        is_authorized = await downloader.client.is_user_authorized()
        logger.info(f"🔍 [parser][load_historical_posts] Статус авторизации: {is_authorized}")

        if not is_authorized:
            error_msg = (
                "Telegram клиент не авторизован. Требуется авторизация.\n"
                f"Используется: {'строка сессии' if settings.session_string else f'файл сессии ({session_path})'}\n"
                "Запустите скрипт: poetry run python src/tplexity/parser/authorize_telegram.py"
            )
            logger.error(f"❌ [parser][load_historical_posts] {error_msg}")
            return

        logger.info("✅ [parser][load_historical_posts] Подключено к Telegram и авторизовано")

        total_posts_downloaded = 0
        total_posts_sent = 0
        total_errors = 0

        channel_titles: dict[str, str] = {}
        for channel in channels_list:
            try:
                entity = await downloader.client.get_entity(channel)
                channel_title = getattr(entity, "title", None) or channel
                channel_titles[channel] = channel_title
                logger.info(f"📺 [parser][load_historical_posts] Канал {channel}: название '{channel_title}'")
            except Exception as e:
                logger.warning(f"⚠️ [parser][load_historical_posts] Не удалось получить название канала {channel}: {e}")
                channel_titles[channel] = channel

        for channel_idx, channel in enumerate(channels_list, 1):
            logger.info(
                f"\n{'=' * 60}\n"
                f"📥 [parser][load_historical_posts] Обработка канала {channel_idx}/{len(channels_list)}: {channel}\n"
                f"{'=' * 60}"
            )

            try:
                logger.info(f"📥 [parser][load_historical_posts] Скачивание постов из {channel}...")
                all_messages = []

                async for message in downloader.client.iter_messages(
                    channel,
                    limit=None,
                    offset_date=None,
                    reverse=False,
                ):
                    if not hasattr(message, "date") or not message.date:
                        continue

                    if message.date < four_months_ago:
                        break

                    message_dict = await downloader._message_to_dict(message, channel)
                    all_messages.append(message_dict)

                    if len(all_messages) % 100 == 0:
                        logger.info(
                            f"  📥 [parser][load_historical_posts] Скачано {len(all_messages)} сообщений из {channel}..."
                        )

                messages_with_text = [
                    msg
                    for msg in all_messages
                    if msg.get("text") and isinstance(msg.get("text"), str) and msg.get("text").strip()
                ]

                total_posts_downloaded += len(messages_with_text)
                logger.info(
                    f"📊 [parser][load_historical_posts] Канал {channel}: "
                    f"скачано {len(all_messages)} постов, "
                    f"{len(messages_with_text)} с текстом"
                )

                if messages_with_text:
                    success, errors = await send_posts_to_retriever(
                        messages_with_text, channel, retriever_url, channel_titles=channel_titles
                    )
                    total_posts_sent += success
                    total_errors += errors

                    logger.info(
                        f"✅ [parser][load_historical_posts] Канал {channel}: "
                        f"отправлено {success} постов, ошибок: {errors}"
                    )
                else:
                    logger.warning(f"⚠️ [parser][load_historical_posts] Канал {channel}: нет постов с текстом")

            except Exception as e:
                logger.error(
                    f"❌ [parser][load_historical_posts] Ошибка при обработке канала {channel}: {e}", exc_info=True
                )
                total_errors += 1

        logger.info(
            f"\n{'=' * 60}\n"
            f"✅ [parser][load_historical_posts] Загрузка завершена!\n"
            f"{'=' * 60}\n"
            f"📊 Статистика:\n"
            f"  - Всего скачано постов: {total_posts_downloaded}\n"
            f"  - Успешно отправлено в БД: {total_posts_sent}\n"
            f"  - Ошибок: {total_errors}\n"
            f"{'=' * 60}"
        )

    except Exception as e:
        logger.error(f"❌ [parser][load_historical_posts] Критическая ошибка: {e}", exc_info=True)
    finally:
        try:
            await downloader.disconnect()
            logger.info("✅ [parser][load_historical_posts] Отключено от Telegram")
        except Exception as e:
            logger.error(f"❌ [parser][load_historical_posts] Ошибка при отключении: {e}")


def main():
    """Точка входа для запуска скрипта."""
    asyncio.run(load_historical_posts())


if __name__ == "__main__":
    main()
