import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from tplexity.parser.config import settings
from tplexity.parser.llm_batcher import get_batcher
from tplexity.parser.relevance_analyzer import calculate_delete_date
from tplexity.parser.telegram_downloader import TelegramDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def clear_database(retriever_url: str) -> bool:
    """
    Очищает векторную БД, удаляя все документы.

    Args:
        retriever_url: URL retriever API

    Returns:
        True если успешно, False в противном случае
    """
    try:
        delete_url = f"{retriever_url.rstrip('/')}/v1/retriever/documents/all"
        logger.info(f"🗑️ Очистка векторной БД: {delete_url}")

        async with httpx.AsyncClient() as client:
            response = await client.delete(delete_url)
            response.raise_for_status()

        logger.info("✅ Векторная БД успешно очищена")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка при очистке БД: {e}")
        return False


async def send_posts_to_retriever(
    posts: list[dict],
    channel: str,
    retriever_url: str,
    llm_batcher,
    llm_provider: str,
    batch_size: int = 50,
    channel_titles: dict[str, str] | None = None,
) -> tuple[int, int]:
    """
    Отправляет посты в retriever с определением актуальности через LLM.

    Args:
        posts: Список постов для отправки
        channel: Название канала
        retriever_url: URL retriever API
        llm_batcher: Батчер для LLM запросов
        llm_provider: Провайдер LLM
        batch_size: Размер батча для отправки
        channel_titles: Словарь с названиями каналов

    Returns:
        Кортеж (успешно отправлено, ошибок)
    """
    if not posts:
        return 0, 0

    documents_url = f"{retriever_url.rstrip('/')}/v1/retriever/documents"
    success_count = 0
    error_count = 0

    async with httpx.AsyncClient() as http_client:
        for i in range(0, len(posts), batch_size):
            batch = posts[i : i + batch_size]
            prepared_posts: list[dict] = []
            llm_tasks = []

            for post in batch:
                text = (post.get("text") or "").strip()
                if not text:
                    continue

                date_str = post.get("date")
                post_date = None
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
                        logger.debug(f"⚠️ Не удалось распарсить дату: {date_str}, ошибка: {e}")

                metadata = {k: v for k, v in post.items() if k != "text"}
                metadata["channel_name"] = channel

                if channel_titles:
                    channel_title = channel_titles.get(channel, channel)
                    metadata["channel_title"] = channel_title
                else:
                    metadata["channel_title"] = channel

                prepared_posts.append(
                    {
                        "text": text,
                        "metadata": metadata,
                        "post_date": post_date,
                        "original_post_id": post.get("id"),
                    }
                )
                llm_tasks.append(llm_batcher.determine_relevance_days(text, llm_provider))

            if not prepared_posts:
                continue

            llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)

            documents = []
            for prepared, result in zip(prepared_posts, llm_results, strict=False):
                if isinstance(result, Exception):
                    logger.warning(
                        f"⚠️ Ошибка при определении актуальности поста {prepared.get('original_post_id')}: {result}"
                    )
                    documents.append({"text": prepared["text"], "metadata": prepared["metadata"]})
                    continue

                relevance_days, _ = result
                delete_date = calculate_delete_date(relevance_days, prepared["post_date"])
                prepared["metadata"]["delete_date"] = delete_date
                documents.append({"text": prepared["text"], "metadata": prepared["metadata"]})

            if not documents:
                continue

            try:
                response = await http_client.post(documents_url, json={"documents": documents})
                response.raise_for_status()
                success_count += len(documents)
                logger.info(
                    f"📤 Отправлено {len(documents)} постов из {channel} "
                    f"(батч {i // batch_size + 1}/{(len(posts) + batch_size - 1) // batch_size})"
                )
            except Exception as e:
                error_count += len(documents)
                logger.error(f"❌ Ошибка при отправке батча из {channel}: {e}")

    return success_count, error_count


async def markup_last_month(days: int = 14):
    """
    Размечает данные за последние N дней.

    Args:
        days: Количество дней назад для загрузки (по умолчанию 14 - 2 недели)
    """
    logger.info("🚀 Запуск разметки данных за последние 2 недели")
    logger.info(f"📅 Период: последние {days} дней")

    if not settings.api_id or not settings.api_hash:
        logger.error("❌ Не указаны API_ID или API_HASH в конфигурации")
        return

    channels_list = settings.get_channels_list()
    if not channels_list:
        logger.error("❌ Список каналов пуст")
        return

    if not settings.webhook_url:
        logger.error("❌ Не указан WEBHOOK_URL в конфигурации")
        return

    retriever_url = settings.webhook_url.rsplit("/retriever", 1)[0]
    logger.info(f"📡 Retriever URL: {retriever_url}")
    logger.info(f"📋 Каналы для обработки: {', '.join(channels_list)}")

    logger.info("=" * 60)
    logger.info("🗑️ Очистка векторной БД перед разметкой...")
    if not await clear_database(retriever_url):
        logger.error("❌ Не удалось очистить БД, прерываем выполнение")
        return
    logger.info("=" * 60)

    days_ago = datetime.now(UTC) - timedelta(days=days)
    logger.info(f"📅 Загружаем посты с {days_ago.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    project_root = Path(__file__).parent.parent.parent.parent
    session_path = project_root / settings.session_name

    logger.info("🔧 Создание TelegramDownloader...")
    downloader = TelegramDownloader(
        api_id=settings.api_id,
        api_hash=settings.api_hash,
        session_name=str(session_path),
        session_string=settings.session_string,
        download_path=str(project_root / settings.data_dir / "telegram"),
    )

    llm_batcher = get_batcher(settings.llm_provider)
    await llm_batcher.start()
    logger.info("✅ LLM батчер запущен")

    try:
        logger.info("🔌 Подключение к Telegram...")
        try:
            await downloader.client.connect()
            logger.info("✅ Соединение с Telegram установлено")
        except Exception as e:
            logger.error(f"❌ Ошибка при подключении к Telegram: {e}", exc_info=True)
            return

        logger.info("🔍 Проверка авторизации...")
        is_authorized = await downloader.client.is_user_authorized()
        logger.info(f"🔍 Статус авторизации: {is_authorized}")

        if not is_authorized:
            error_msg = (
                "Telegram клиент не авторизован. Требуется авторизация.\n"
                f"Используется: {'строка сессии' if settings.session_string else f'файл сессии ({session_path})'}\n"
                "Запустите скрипт: poetry run python src/tplexity/parser/authorize_telegram.py"
            )
            logger.error(f"❌ {error_msg}")
            return

        logger.info("✅ Подключено к Telegram и авторизовано")

        total_posts_downloaded = 0
        total_posts_sent = 0
        total_errors = 0

        channel_titles: dict[str, str] = {}
        for channel in channels_list:
            try:
                entity = await downloader.client.get_entity(channel)
                channel_title = getattr(entity, "title", None) or channel
                channel_titles[channel] = channel_title
                logger.info(f"📺 Канал {channel}: название '{channel_title}'")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось получить название канала {channel}: {e}")
                channel_titles[channel] = channel

        for channel_idx, channel in enumerate(channels_list, 1):
            logger.info(f"\n{'=' * 60}\n📥 Обработка канала {channel_idx}/{len(channels_list)}: {channel}\n{'=' * 60}")

            try:
                logger.info(f"📥 Скачивание постов из {channel}...")
                all_messages = []

                async for message in downloader.client.iter_messages(
                    channel,
                    limit=None,
                    offset_date=None,
                    reverse=False,
                ):
                    if not hasattr(message, "date") or not message.date:
                        continue

                    if message.date < days_ago:
                        break

                    message_dict = await downloader._message_to_dict(message, channel)
                    all_messages.append(message_dict)

                    if len(all_messages) % 50 == 0:
                        logger.info(f"  📥 Скачано {len(all_messages)} сообщений из {channel}...")

                messages_with_text = [
                    msg
                    for msg in all_messages
                    if msg.get("text") and isinstance(msg.get("text"), str) and msg.get("text").strip()
                ]

                total_posts_downloaded += len(messages_with_text)
                logger.info(
                    f"📊 Канал {channel}: скачано {len(all_messages)} постов, {len(messages_with_text)} с текстом"
                )

                if messages_with_text:
                    success, errors = await send_posts_to_retriever(
                        messages_with_text,
                        channel,
                        retriever_url,
                        llm_batcher,
                        settings.llm_provider,
                        channel_titles=channel_titles,
                    )
                    total_posts_sent += success
                    total_errors += errors

                    logger.info(f"✅ Канал {channel}: отправлено {success} постов, ошибок: {errors}")
                else:
                    logger.warning(f"⚠️ Канал {channel}: нет постов с текстом")

            except Exception as e:
                logger.error(f"❌ Ошибка при обработке канала {channel}: {e}", exc_info=True)
                total_errors += 1

        logger.info(
            f"\n{'=' * 60}\n"
            f"✅ Разметка данных завершена!\n"
            f"{'=' * 60}\n"
            f"📊 Статистика:\n"
            f"  - Всего скачано постов: {total_posts_downloaded}\n"
            f"  - Успешно отправлено в БД: {total_posts_sent}\n"
            f"  - Ошибок: {total_errors}\n"
            f"{'=' * 60}"
        )

    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
    finally:
        await llm_batcher.stop()

        try:
            await downloader.disconnect()
            logger.info("✅ Отключено от Telegram")
        except Exception as e:
            logger.error(f"❌ Ошибка при отключении: {e}")


def main():
    """Точка входа для запуска скрипта."""
    import sys

    days = 14
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            logger.warning(f"⚠️ Неверный аргумент {sys.argv[1]}, используем 14 дней (2 недели)")

    asyncio.run(markup_last_month(days=days))


if __name__ == "__main__":
    main()
