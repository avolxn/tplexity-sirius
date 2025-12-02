import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.types import Message


class TelegramDownloader:
    """Класс для скачивания данных из Telegram каналов."""

    @staticmethod
    def parse_channel_link(link: str) -> str:
        """
        Извлекает username канала из ссылки или возвращает как есть.

        Args:
            link: Ссылка на канал (t.me/channel, @channel или просто channel)

        Returns:
            Username канала без @
        """
        link = link.strip()

        if "t.me/" in link or "telegram.me/" in link:
            parts = link.rstrip("/").split("/")
            return parts[-1].lstrip("@")

        if link.startswith("@"):
            return link[1:]

        return link

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_name: str = "telegram_session",
        session_string: str | None = None,
        download_path: str = "data/telegram",
    ):
        """
        Инициализация клиента Telegram.

        Args:
            api_id: ID приложения из https://my.telegram.org
            api_hash: Hash приложения из https://my.telegram.org
            session_name: Имя файла сессии (используется если session_string не указан)
            session_string: Строка сессии (если указана, используется вместо файла)
            download_path: Путь для сохранения данных
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.session_string = session_string
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)

        print(f"🔍 [telegram_downloader.__init__] session_string получен: {session_string is not None}")
        if session_string:
            print(f"🔍 [telegram_downloader.__init__] session_string длина: {len(session_string)}")
            print(f"🔍 [telegram_downloader.__init__] session_string первые 30 символов: {session_string[:30]}...")
            print(f"🔍 [telegram_downloader.__init__] session_string пустая строка: {session_string == ''}")
        else:
            print(f"🔍 [telegram_downloader.__init__] session_string is None, будет использован файл: {session_name}")

        if session_string and session_string.strip():
            print("✅ [telegram_downloader.__init__] Используется StringSession (строка сессии)")
            session = StringSession(session_string)
        else:
            print(f"📁 [telegram_downloader.__init__] Используется файл сессии: {session_name}")
            session = session_name

        self.client = TelegramClient(session, api_id, api_hash)

    async def connect(self, max_retries: int = 3):
        """
        Подключение к Telegram с повторными попытками.

        Args:
            max_retries: Максимальное количество попыток подключения
        """

        session_info = (
            f"session_string (длина: {len(self.session_string)})"
            if self.session_string
            else f"файл: {self.session_name}"
        )
        print(f"🔌 [telegram_downloader] Подключение к Telegram (сессия: {session_info})")

        for attempt in range(max_retries):
            try:
                print(f"🔄 [telegram_downloader] Попытка подключения {attempt + 1}/{max_retries}...")
                await self.client.connect()
                print("✅ [telegram_downloader] Соединение установлено")

                is_authorized = await self.client.is_user_authorized()
                print(f"🔍 [telegram_downloader] Статус авторизации: {is_authorized}")

                if not is_authorized:
                    print("❌ [telegram_downloader] Ошибка: Сессия не авторизована")
                    print(f"📋 [telegram_downloader] Используется: {session_info}")
                    print("💡 [telegram_downloader] Используйте authorize_telegram.py для создания новой сессии")
                    return False

                print("✅ [telegram_downloader] Подключено к Telegram и авторизовано")
                return True
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"❌ [telegram_downloader] Попытка {attempt + 1} не удалась")
                print(f"   Тип ошибки: {error_type}")
                print(f"   Сообщение: {error_msg}")

                if attempt < max_retries - 1:
                    print("⏳ [telegram_downloader] Повторная попытка через 2 секунды...")
                    await asyncio.sleep(2)
                else:
                    print(f"❌ [telegram_downloader] Не удалось подключиться после {max_retries} попыток")
                    import traceback

                    print("📋 [telegram_downloader] Полный traceback:")
                    traceback.print_exc()
                    return False

    async def disconnect(self):
        """Отключение от Telegram."""
        try:
            if self.client.is_connected():
                await self.client.disconnect()
                print("Отключено от Telegram")
        except Exception as e:
            print(f"Ошибка при отключении: {e}")

    async def get_channel_info(self, channel_username: str) -> dict[str, Any]:
        """
        Получить информацию о канале.

        Args:
            channel_username: Username канала (без @) или ссылка

        Returns:
            Словарь с информацией о канале
        """
        channel = await self.client.get_entity(channel_username)

        info = {
            "id": channel.id,
            "title": getattr(channel, "title", None),
            "username": getattr(channel, "username", None),
            "participants_count": getattr(channel, "participants_count", None),
            "description": getattr(channel, "about", None),
        }

        return info

    async def download_messages(
        self,
        channel_username: str,
        limit: int | None = None,
        offset_date: datetime | None = None,
        min_id: int = 0,
        max_id: int = 0,
        reverse: bool = False,
        save_media: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Скачать сообщения из канала.

        Args:
            channel_username: Username канала (без @) или ссылка
            limit: Максимальное количество сообщений (None = все)
            offset_date: Дата, с которой начать скачивание
            min_id: Минимальный ID сообщения
            max_id: Максимальный ID сообщения
            reverse: Скачивать в обратном порядке (от старых к новым)
            save_media: Сохранять медиа файлы

        Returns:
            Список словарей с данными сообщений
        """
        print(f"Скачивание сообщений из канала: {channel_username}")

        channel = await self.client.get_entity(channel_username)
        messages_data = []

        channel_folder = self.download_path / self._sanitize_filename(channel_username)
        channel_folder.mkdir(exist_ok=True)

        if save_media:
            media_folder = channel_folder / "media"
            media_folder.mkdir(exist_ok=True)

        count = 0
        async for message in self.client.iter_messages(
            channel,
            limit=limit,
            offset_date=offset_date,
            min_id=min_id,
            max_id=max_id,
            reverse=reverse,
        ):
            if not isinstance(message, Message):
                continue

            count += 1
            if limit and count % 10 == 0:
                print(f"  Скачано {count}/{limit} сообщений...")
            elif not limit and count % 100 == 0:
                print(f"  Скачано {count} сообщений...")

            message_dict = await self._message_to_dict(message, channel_username)

            if save_media and message.media:
                try:
                    media_path = await message.download_media(file=str(media_folder / f"{message.id}"))
                    message_dict["media_path"] = media_path
                except Exception as e:
                    print(f"Ошибка при скачивании медиа {message.id}: {e}")
                    message_dict["media_path"] = None

            messages_data.append(message_dict)

        print(f"Скачано {len(messages_data)} сообщений")

        return messages_data

    async def _message_to_dict(self, message: Message, channel_username: str = None) -> dict[str, Any]:
        """
        Преобразовать сообщение в словарь.

        Args:
            message: Объект сообщения Telethon
            channel_username: Username канала для формирования ссылки

        Returns:
            Словарь с данными сообщения
        """
        message_link = None
        if channel_username:
            clean_username = channel_username.lstrip("@")
            message_link = f"https://t.me/{clean_username}/{message.id}"

        return {
            "id": message.id,
            "link": message_link,
            "date": message.date.isoformat() if message.date else None,
            "text": message.text,
            "views": message.views,
            "forwards": message.forwards,
            "edit_date": message.edit_date.isoformat() if message.edit_date else None,
            "has_media": message.media is not None,
            "media_type": type(message.media).__name__ if message.media else None,
        }

    @staticmethod
    def filter_empty_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Фильтрует сообщения, удаляя те, у которых пустой текст.

        Args:
            messages: Список сообщений

        Returns:
            Отфильтрованный список сообщений
        """
        filtered = [msg for msg in messages if msg.get("text") and msg["text"].strip()]
        removed_count = len(messages) - len(filtered)
        if removed_count > 0:
            print(f"  Удалено {removed_count} сообщений с пустым текстом")
        return filtered

    def save_to_json(
        self,
        data: list[dict[str, Any]],
        channel_username: str,
        filename: str | None = None,
        filter_empty: bool = False,
    ):
        """
        Сохранить данные в JSON файл.

        Args:
            data: Список словарей с данными
            channel_username: Username канала
            filename: Имя файла (если None, будет сгенерировано автоматически)
            filter_empty: Удалять ли сообщения с пустым текстом
        """
        if filter_empty:
            data = self.filter_empty_messages(data)

        channel_folder = self.download_path / self._sanitize_filename(channel_username)
        channel_folder.mkdir(exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"messages_{timestamp}.json"

        filepath = channel_folder / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Данные сохранены в {filepath}")
        return filepath

    def append_to_json(
        self,
        new_data: list[dict[str, Any]],
        filepath: Path,
    ):
        """
        Добавить новые данные к существующему JSON файлу.

        Args:
            new_data: Список новых сообщений
            filepath: Путь к существующему JSON файлу
        """
        if filepath.exists():
            with open(filepath, encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(new_data)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        print(f"Добавлено {len(new_data)} новых сообщений в {filepath}")

    def save_to_csv(
        self,
        data: list[dict[str, Any]],
        channel_username: str,
        filename: str | None = None,
        filter_empty: bool = False,
    ):
        """
        Сохранить данные в CSV файл.

        Args:
            data: Список словарей с данными
            channel_username: Username канала
            filename: Имя файла (если None, будет сгенерировано автоматически)
            filter_empty: Удалять ли сообщения с пустым текстом
        """
        if filter_empty:
            data = self.filter_empty_messages(data)

        channel_folder = self.download_path / self._sanitize_filename(channel_username)
        channel_folder.mkdir(exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"messages_{timestamp}.csv"

        filepath = channel_folder / filename

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding="utf-8")

        print(f"Данные сохранены в {filepath}")

    def save_to_parquet(
        self,
        data: list[dict[str, Any]],
        channel_username: str,
        filename: str | None = None,
        filter_empty: bool = False,
    ):
        """
        Сохранить данные в Parquet файл.

        Args:
            data: Список словарей с данными
            channel_username: Username канала
            filename: Имя файла (если None, будет сгенерировано автоматически)
            filter_empty: Удалять ли сообщения с пустым текстом
        """
        if filter_empty:
            data = self.filter_empty_messages(data)

        channel_folder = self.download_path / self._sanitize_filename(channel_username)
        channel_folder.mkdir(exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"messages_{timestamp}.parquet"

        filepath = channel_folder / filename

        df = pd.DataFrame(data)
        df.to_parquet(filepath, index=False)

        print(f"Данные сохранены в {filepath}")

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Очистить имя файла от недопустимых символов.

        Args:
            filename: Исходное имя файла

        Returns:
            Очищенное имя файла
        """
        filename = filename.lstrip("@")
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename

    async def download_multiple_channels(
        self,
        channel_usernames: list[str],
        limit: int | None = None,
        save_format: str = "json",
        save_media: bool = False,
    ):
        """
        Скачать сообщения из нескольких каналов.

        Args:
            channel_usernames: Список username каналов
            limit: Максимальное количество сообщений из каждого канала
            save_format: Формат сохранения ('json', 'csv', 'parquet')
            save_media: Сохранять медиа файлы
        """
        for channel in channel_usernames:
            try:
                print(f"\n{'=' * 60}")
                print(f"Обработка канала: {channel}")
                print(f"{'=' * 60}")

                info = await self.get_channel_info(channel)
                print(f"Канал: {info.get('title', channel)}")
                print(f"Подписчиков: {info.get('participants_count', 'N/A')}")

                messages = await self.download_messages(
                    channel,
                    limit=limit,
                    save_media=save_media,
                )

                if save_format == "json":
                    self.save_to_json(messages, channel)
                elif save_format == "csv":
                    self.save_to_csv(messages, channel)
                elif save_format == "parquet":
                    self.save_to_parquet(messages, channel)
                else:
                    raise ValueError(f"Неподдерживаемый формат: {save_format}")

            except Exception as e:
                print(f"Ошибка при обработке канала {channel}: {e}")
                continue


async def main():
    """Пример использования."""
    from dotenv import load_dotenv

    load_dotenv()

    api_id = int(os.getenv("TELEGRAM_API_ID", "0"))
    api_hash = os.getenv("TELEGRAM_API_HASH", "")

    if not api_id or not api_hash:
        print("Ошибка: Укажите TELEGRAM_API_ID и TELEGRAM_API_HASH в .env файле")
        print("Получить можно здесь: https://my.telegram.org")
        return

    downloader = TelegramDownloader(
        api_id=api_id, api_hash=api_hash, session_name="my_session", download_path="data/telegram"
    )

    try:
        connected = await downloader.connect()
        if not connected:
            return

        messages = await downloader.download_messages(
            channel_username="durov",
            limit=100,
            save_media=False,
        )

        downloader.save_to_json(messages, "durov")
        downloader.save_to_csv(messages, "durov")

    finally:
        await downloader.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
