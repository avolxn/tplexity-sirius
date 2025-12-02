import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from telethon import events
from telethon.tl.types import Message

from tplexity.parser.llm_batcher import get_batcher
from tplexity.parser.post_deletion_service import PostDeletionService
from tplexity.parser.relevance_analyzer import calculate_delete_date
from tplexity.parser.telegram_downloader import TelegramDownloader

logger = logging.getLogger(__name__)


@dataclass
class FailedPost:
    """Структура для хранения неудачно отправленного поста"""

    post_data: dict[str, Any]
    channel: str
    retry_count: int = 0


class TelegramMonitorService:
    """Сервис для мониторинга Telegram каналов через WebSocket (события Telethon)."""

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        channels: list[str],
        session_name: str = "my_session",
        data_dir: str = "data",
        webhook_url: str | None = None,
        retry_interval: int = 60,
        session_string: str | None = None,
        llm_provider: str = "qwen",
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
        qdrant_api_key: str | None = None,
        qdrant_collection_name: str | None = None,
        qdrant_timeout: int = 60,
    ):
        self.api_id = api_id
        self.api_hash = api_hash
        self.channels = channels
        self.webhook_url = webhook_url
        self.retry_interval = retry_interval
        self.llm_provider = llm_provider
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_api_key = qdrant_api_key
        self.qdrant_collection_name = qdrant_collection_name
        self.qdrant_timeout = qdrant_timeout

        self.project_root = Path(__file__).parent.parent.parent.parent

        self.data_dir = self.project_root / data_dir
        self.telegram_dir = self.data_dir / "telegram"

        self.downloader: TelegramDownloader | None = None
        self.is_running = False
        self.session_name = session_name
        self.session_string = session_string

        self.failed_posts: deque[FailedPost] = deque()
        self.retry_task: asyncio.Task | None = None
        self.deletion_task: asyncio.Task | None = None

        self.channel_entities: dict[str, Any] = {}

        self.channel_titles: dict[str, str] = {}

        self.deletion_service: PostDeletionService | None = None

        self.http_client: httpx.AsyncClient | None = None

        self.llm_batcher = get_batcher(llm_provider)

    async def initialize(self):
        """Инициализация: загрузка существующих данных."""
        logger.info("🔧 [parser][monitor_service] Инициализация сервиса...")
        logger.info(f"📁 [parser][monitor_service] Корень проекта: {self.project_root}")
        logger.info(f"📁 [parser][monitor_service] Директория данных: {self.data_dir}")

        self.telegram_dir.mkdir(parents=True, exist_ok=True)

        session_path = self.project_root / self.session_name

        logger.info("=" * 60)
        logger.info("📋 [parser][monitor_service] Конфигурация подключения:")
        logger.info(f"   API_ID: {self.api_id}")
        logger.info(f"   API_HASH: {'*' * 10 if self.api_hash else 'None (не указан!)'}")
        logger.info(f"   SESSION_NAME: {self.session_name}")
        logger.info(
            f"   TELEGRAM_SESSION_STRING: {'указан' if self.session_string else 'не указан (будет использован файл)'}"
        )

        if self.session_string:
            logger.info(
                f"🔑 [parser][monitor_service] Используется строка сессии (длина: {len(self.session_string)} символов)"
            )
            logger.debug(
                f"🔑 [parser][monitor_service] Первые 20 символов session_string: {self.session_string[:20]}..."
            )
        else:
            logger.info(f"📁 [parser][monitor_service] Используется файл сессии: {session_path}")
            if session_path.exists():
                logger.info(
                    f"📁 [parser][monitor_service] Файл сессии существует, размер: {session_path.stat().st_size} байт"
                )
            else:
                logger.warning(f"⚠️ [parser][monitor_service] Файл сессии не найден: {session_path}")
                logger.warning(
                    "💡 [parser][monitor_service] Для использования строки сессии добавьте TELEGRAM_SESSION_STRING в .env"
                )
                logger.warning(
                    "💡 [parser][monitor_service] Или запустите: poetry run python src/tplexity/parser/authorize_telegram.py"
                )
        logger.info("=" * 60)

        logger.info("🔍 [parser][monitor_service] ПЕРЕД созданием TelegramDownloader:")
        logger.info(f"   self.session_string type: {type(self.session_string)}")
        logger.info(f"   self.session_string value: {self.session_string}")
        logger.info(f"   self.session_string is None: {self.session_string is None}")
        logger.info(f"   self.session_string == '': {self.session_string == ''}")
        if self.session_string:
            logger.info(f"   self.session_string.strip() == '': {self.session_string.strip() == ''}")
            logger.info(f"   self.session_string длина: {len(self.session_string)}")

        logger.info("🔧 [parser][monitor_service] Создание TelegramDownloader...")
        self.downloader = TelegramDownloader(
            api_id=self.api_id,
            api_hash=self.api_hash,
            session_name=str(session_path),
            session_string=self.session_string,
            download_path=str(self.telegram_dir),
        )

        logger.info("🔌 [parser][monitor_service] Подключение к Telegram...")
        try:
            await self.downloader.client.connect()
            logger.info("✅ [parser][monitor_service] Соединение с Telegram установлено")
        except Exception as e:
            logger.error(f"❌ [parser][monitor_service] Ошибка при подключении к Telegram: {e}", exc_info=True)
            raise

        logger.info("🔍 [parser][monitor_service] Проверка авторизации...")
        is_authorized = await self.downloader.client.is_user_authorized()
        logger.info(f"🔍 [parser][monitor_service] Статус авторизации: {is_authorized}")

        if not is_authorized:
            error_msg = (
                "Telegram клиент не авторизован. Требуется авторизация.\n"
                f"Используется: {'строка сессии' if self.session_string else f'файл сессии ({session_path})'}\n"
                "Запустите скрипт: poetry run python src/tplexity/parser/authorize_telegram.py"
            )
            logger.error(f"❌ [parser][monitor_service] {error_msg}")
            raise ValueError(error_msg)
        else:
            logger.info("✅ [parser][monitor_service] Telegram клиент успешно подключен и авторизован")

        for channel in self.channels:
            try:
                entity = await self.downloader.client.get_entity(channel)
                self.channel_entities[channel] = entity

                channel_title = getattr(entity, "title", None) or channel
                self.channel_titles[channel] = channel_title
                logger.info(
                    f"✅ [parser][monitor_service] Получен entity для канала: {channel} (название: {channel_title})"
                )
            except Exception as e:
                logger.error(f"❌ [parser][monitor_service] Ошибка при получении entity для канала {channel}: {e}")

                self.channel_titles[channel] = channel

        logger.info("=" * 60)
        logger.info("📺 [parser][monitor_service] Каналы для мониторинга:")
        for i, channel in enumerate(self.channels, 1):
            channel_title = self.channel_titles.get(channel, channel)
            logger.info(f"   {i}. {channel} ({channel_title})")
        logger.info(f"📊 [parser][monitor_service] Всего каналов: {len(self.channels)}")
        logger.info("=" * 60)

        self.http_client = httpx.AsyncClient()
        logger.info("✅ [parser][monitor_service] HTTP клиент инициализирован")

        await self.llm_batcher.start()

        logger.info("✅ [parser][monitor_service] Инициализация завершена")

    async def start_monitoring(self):
        """Запускает мониторинг каналов через WebSocket (события Telethon)."""
        if self.is_running:
            logger.warning("⚠️ [parser][monitor_service] Мониторинг уже запущен")
            return

        self.is_running = True
        logger.info("🚀 [parser][monitor_service] Запуск мониторинга каналов через WebSocket...")

        for channel in self.channels:
            if channel not in self.channel_entities:
                logger.warning(f"⚠️ [parser][monitor_service] Канал {channel} не найден, пропускаем")
                continue

            entity = self.channel_entities[channel]

            def make_handler(channel_name: str):
                async def handler(event: events.NewMessage.Event):
                    """Обработчик новых сообщений из канала"""
                    await self._handle_new_message(event, channel_name)

                return handler

            self.downloader.client.add_event_handler(make_handler(channel), events.NewMessage(chats=entity))

            logger.info(f"✅ [parser][monitor_service] Зарегистрирован обработчик для канала: {channel}")

        self.retry_task = asyncio.create_task(self._retry_failed_posts_loop())

        if self.qdrant_host and self.qdrant_port and self.qdrant_collection_name:
            try:
                self.deletion_service = PostDeletionService(
                    qdrant_host=self.qdrant_host,
                    qdrant_port=self.qdrant_port,
                    qdrant_api_key=self.qdrant_api_key,
                    qdrant_collection_name=self.qdrant_collection_name,
                    qdrant_timeout=self.qdrant_timeout,
                )

                asyncio.create_task(self._run_deletion_task(initial_run=True))

                self.deletion_task = asyncio.create_task(self._deletion_loop())
                logger.info("✅ [monitor_service] Сервис удаления постов инициализирован и запущен")
            except Exception as e:
                logger.error(
                    f"❌ [monitor_service] Ошибка при инициализации сервиса удаления постов: {e}",
                    exc_info=True,
                )
        else:
            logger.warning("⚠️ [monitor_service] Параметры Qdrant не настроены, сервис удаления постов не запущен")

        logger.info("✅ [parser][monitor_service] Мониторинг запущен, ожидание новых сообщений...")

    async def _handle_new_message(self, event: events.NewMessage.Event, channel: str):
        """Обрабатывает новое сообщение из канала."""
        try:
            message = event.message
            if not isinstance(message, Message):
                return

            if not message.text or not message.text.strip():
                return

            message_dict = await self.downloader._message_to_dict(message, channel)

            logger.info(
                f"📨 [parser][monitor_service] Новое сообщение из канала {channel}: ID={message.id}, "
                f"длина текста={len(message.text)}"
            )

            await self._save_message(channel, message_dict)

            if self.webhook_url:
                success = await self._send_post_to_retriever(message_dict, channel)
                if not success:
                    failed_post = FailedPost(post_data=message_dict, channel=channel)
                    self.failed_posts.append(failed_post)
                    logger.warning(
                        f"⚠️ [parser][monitor_service] Не удалось отправить пост {message.id} из {channel}, "
                        f"добавлен в очередь повторных попыток"
                    )

        except Exception as e:
            logger.error(
                f"❌ [parser][monitor_service] Ошибка при обработке нового сообщения из {channel}: {e}", exc_info=True
            )

    async def _save_message(self, channel: str, message_dict: dict[str, Any]):
        """Сохраняет новое сообщение в файл (асинхронный I/O)."""
        channel_dir = self.telegram_dir / channel
        channel_dir.mkdir(parents=True, exist_ok=True)

        messages_file = channel_dir / "messages_monitor.json"

        existing_messages = []
        if messages_file.exists():
            async with aiofiles.open(messages_file, encoding="utf-8") as f:
                content = await f.read()
                existing_messages = json.loads(content)

        message_id = message_dict.get("id")
        if not any(msg.get("id") == message_id for msg in existing_messages):
            existing_messages.append(message_dict)

            async with aiofiles.open(messages_file, mode="w", encoding="utf-8") as f:
                await f.write(json.dumps(existing_messages, ensure_ascii=False, indent=2))

            logger.debug(f"💾 [parser][monitor_service] Сообщение сохранено в {messages_file}")

    async def _send_post_to_retriever(self, post_dict: dict[str, Any], channel: str) -> bool:
        """
        Отправляет пост полностью (без чанкирования) в Retriever API.

        Returns:
            True если отправка успешна, False в противном случае
        """
        try:
            text = (post_dict.get("text") or "").strip()
            if not text:
                logger.warning("⚠️ [parser][monitor_service] Пост без текста пропущен")
                return True

            date_str = post_dict.get("date")
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
                    logger.debug(f"⚠️ [parser][monitor_service] Не удалось распарсить дату: {date_str}, ошибка: {e}")

            metadata = {k: v for k, v in post_dict.items() if k != "text"}
            metadata["channel_name"] = channel

            channel_title = self.channel_titles.get(channel, channel)
            metadata["channel_title"] = channel_title

            try:
                relevance_days, llm_response = await self.llm_batcher.determine_relevance_days(text, self.llm_provider)

                post_publication_date = None
                date_str = post_dict.get("date")
                if date_str:
                    try:
                        if date_str.endswith("Z"):
                            date_str = date_str.replace("Z", "+00:00")
                        if "T" in date_str:
                            post_publication_date = datetime.fromisoformat(date_str)
                        else:
                            post_publication_date = datetime.fromisoformat(f"{date_str}T00:00:00")
                    except (ValueError, AttributeError) as e:
                        logger.debug(
                            f"⚠️ [monitor_service] Не удалось распарсить дату публикации: {date_str}, ошибка: {e}"
                        )

                delete_date = calculate_delete_date(relevance_days, post_publication_date)
                metadata["delete_date"] = delete_date
                logger.info(
                    f"📅 [monitor_service] Для поста {post_dict.get('id')} из {channel} определена дата удаления: {delete_date} "
                    f"(актуальность: {relevance_days} дней от даты публикации, ответ LLM: {llm_response})"
                )
            except Exception as e:
                logger.error(
                    f"❌ [monitor_service] Ошибка при определении актуальности поста {post_dict.get('id')}: {e}",
                    exc_info=True,
                )

            document = {"text": text, "metadata": metadata}

            if not self.http_client:
                logger.error("❌ [parser][monitor_service] HTTP клиент не инициализирован")
                return False

            try:
                response = await self.http_client.post(self.webhook_url, json={"documents": [document]}, timeout=30.0)
                response.raise_for_status()
                logger.info(
                    f"📤 [parser][monitor_service] Пост {post_dict.get('id')} из {channel} "
                    f"успешно отправлен в Retriever"
                )
                return True
            except httpx.HTTPError as e:
                logger.error(f"❌ [parser][monitor_service] HTTP ошибка при отправке поста {post_dict.get('id')}: {e}")
                return False
        except Exception as e:
            logger.error(
                f"❌ [parser][monitor_service] Ошибка при отправке поста {post_dict.get('id')} "
                f"из {channel} в Retriever API: {e}"
            )
            return False

    async def _retry_failed_posts_loop(self):
        """Фоновая задача для повторных попыток отправки неудачных постов."""
        logger.info("🔄 [parser][monitor_service] Запущена задача для повторных попыток отправки постов")

        while self.is_running:
            try:
                await asyncio.sleep(self.retry_interval)

                if not self.failed_posts:
                    continue

                logger.info(f"🔄 [parser][monitor_service] Попытка повторной отправки {len(self.failed_posts)} постов")

                posts_to_retry = list(self.failed_posts)
                self.failed_posts.clear()

                for failed_post in posts_to_retry:
                    if not self.is_running:
                        break

                    success = await self._send_post_to_retriever(failed_post.post_data, failed_post.channel)

                    if not success:
                        failed_post.retry_count += 1
                        self.failed_posts.append(failed_post)
                        logger.warning(
                            f"⚠️ [parser][monitor_service] Повторная попытка {failed_post.retry_count} "
                            f"для поста {failed_post.post_data.get('id')} из {failed_post.channel} "
                            f"не удалась, будет повторена через {self.retry_interval} секунд"
                        )
                    else:
                        logger.info(
                            f"✅ [parser][monitor_service] Пост {failed_post.post_data.get('id')} "
                            f"из {failed_post.channel} успешно отправлен после повторной попытки"
                        )

            except asyncio.CancelledError:
                logger.info("🛑 [parser][monitor_service] Задача повторных попыток остановлена")
                break
            except Exception as e:
                logger.error(f"❌ [parser][monitor_service] Ошибка в задаче повторных попыток: {e}", exc_info=True)

    async def download_initial_messages(self) -> dict[str, Any]:
        """
        Скачивает все доступные сообщения из каждого канала.
        Удаляет пустые сообщения и сохраняет результаты.

        Returns:
            Статистика по скачанным сообщениям
        """
        logger.info("📥 [parser][monitor_service] Скачивание всех доступных сообщений из каждого канала...")

        results: dict[str, Any] = {"total_downloaded": 0, "total_saved": 0, "channels": {}}

        for channel in self.channels:
            try:
                logger.info(f"📥 [parser][monitor_service] Скачивание из канала: {channel}")

                messages = await self.downloader.download_messages(channel_username=channel, limit=None)

                downloaded_count = len(messages)

                messages_with_text = [
                    msg
                    for msg in messages
                    if msg.get("text") and isinstance(msg.get("text"), str) and msg.get("text").strip()
                ]
                saved_count = len(messages_with_text)

                logger.info(
                    f"📊 [parser][monitor_service] Канал {channel}: скачано {downloaded_count}, с текстом {saved_count}"
                )

                if messages_with_text:
                    channel_dir = self.telegram_dir / channel
                    channel_dir.mkdir(parents=True, exist_ok=True)

                    messages_file = channel_dir / "messages_monitor.json"
                    async with aiofiles.open(messages_file, mode="w", encoding="utf-8") as f:
                        await f.write(json.dumps(messages_with_text, ensure_ascii=False, indent=2))

                    logger.info(f"💾 [parser][monitor_service] Сохранено в {messages_file}")

                results["channels"][channel] = {
                    "downloaded": downloaded_count,
                    "saved": saved_count,
                    "filtered_out": downloaded_count - saved_count,
                }
                results["total_downloaded"] += downloaded_count
                results["total_saved"] += saved_count

            except Exception as e:
                logger.error(f"❌ [parser][monitor_service] Ошибка при скачивании из {channel}: {e}", exc_info=True)
                results["channels"][channel] = {"error": str(e)}

        logger.info(
            f"✅ [parser][monitor_service] Скачивание завершено. "
            f"Всего скачано: {results['total_downloaded']}, сохранено: {results['total_saved']}"
        )
        return results

    async def stop_monitoring(self):
        """Останавливает мониторинг."""
        logger.info("🛑 [parser][monitor_service] Остановка мониторинга...")
        self.is_running = False

        if self.retry_task:
            self.retry_task.cancel()
            try:
                await self.retry_task
            except asyncio.CancelledError:
                pass

        if self.downloader and self.downloader.client:
            self.downloader.client.remove_event_handlers()
            logger.info("✅ [parser][monitor_service] Обработчики событий удалены")

        if self.downloader and self.downloader.client:
            try:
                await self.downloader.client.disconnect()
                logger.info("✅ [parser][monitor_service] Telegram клиент отключен")
            except Exception as e:
                logger.error(f"❌ [parser][monitor_service] Ошибка при отключении клиента: {e}")

        if self.deletion_task:
            self.deletion_task.cancel()
            try:
                await self.deletion_task
            except asyncio.CancelledError:
                pass

        await self.llm_batcher.stop()

        if self.http_client:
            await self.http_client.aclose()
            logger.info("✅ [parser][monitor_service] HTTP клиент закрыт")

        logger.info(
            f"✅ [monitor_service] Мониторинг остановлен. В очереди повторных попыток: {len(self.failed_posts)} постов"
        )

    async def _run_deletion_task(self, initial_run: bool = False) -> None:
        """
        Запускает задачу удаления устаревших постов

        Args:
            initial_run: Если True, это первый запуск при старте сервиса
        """
        if not self.deletion_service:
            return

        try:
            if initial_run:
                logger.info("🗑️ [monitor_service] Запуск удаления устаревших постов при старте...")
            else:
                logger.info("🗑️ [monitor_service] Запуск периодического удаления устаревших постов...")

            deleted_count = await self.deletion_service.delete_expired_posts()
            logger.info(f"✅ [monitor_service] Удаление завершено: удалено {deleted_count} постов")
        except Exception as e:
            logger.error(
                f"❌ [monitor_service] Ошибка при удалении устаревших постов: {e}",
                exc_info=True,
            )

    async def _deletion_loop(self) -> None:
        """Периодическая задача для удаления устаревших постов (каждые 24 часа)"""
        logger.info("🔄 [monitor_service] Запущена периодическая задача удаления постов (каждые 24 часа)")

        while self.is_running:
            try:
                await asyncio.sleep(86400)

                if not self.is_running:
                    break

                await self._run_deletion_task(initial_run=False)

            except asyncio.CancelledError:
                logger.info("🛑 [monitor_service] Периодическая задача удаления остановлена")
                break
            except Exception as e:
                logger.error(
                    f"❌ [monitor_service] Ошибка в периодической задаче удаления: {e}",
                    exc_info=True,
                )
