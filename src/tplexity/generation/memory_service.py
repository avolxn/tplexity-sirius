import json
import logging

import redis.asyncio as aioredis

from tplexity.generation.config import settings

logger = logging.getLogger(__name__)


class MemoryService:
    """Сервис для управления историей диалогов в Redis"""

    def __init__(self):
        """Инициализация Redis клиента"""
        self.redis_client: aioredis.Redis | None = None

    async def _ensure_client(self) -> None:
        """Инициализирует Redis клиент, если он еще не создан"""
        if self.redis_client is None:
            self.redis_client = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True,
            )
            logger.info(
                f"✅ [memory_service] Redis клиент инициализирован: {settings.redis_host}:{settings.redis_port}"
            )

    def _get_session_key(self, session_id: str) -> str:
        """Формирует ключ для сессии в Redis"""
        return f"session:{session_id}"

    async def get_history(self, session_id: str) -> list[dict[str, str]]:
        """
        Получает историю диалога для сессии

        Args:
            session_id: Идентификатор сессии

        Returns:
            list[dict[str, str]]: Список сообщений в формате OpenAI
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        await self._ensure_client()
        if not self.redis_client:
            return []

        try:
            session_key = self._get_session_key(session_id)
            history_json = await self.redis_client.get(session_key)

            if history_json:
                history = json.loads(history_json)
                logger.debug(f"📖 [memory_service] Получена история для сессии {session_id}: {len(history)} сообщений")
                return history
            else:
                logger.debug(f"📖 [memory_service] История для сессии {session_id} не найдена")
                return []

        except json.JSONDecodeError as e:
            logger.error(f"❌ [memory_service] Ошибка декодирования JSON для сессии {session_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ [memory_service] Ошибка при получении истории для сессии {session_id}: {e}")
            return []

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Добавляет сообщение в историю диалога

        Args:
            session_id: Идентификатор сессии
            role: Роль сообщения ("user" или "assistant")
            content: Содержимое сообщения
        """
        await self._ensure_client()
        if not self.redis_client:
            return

        try:
            session_key = self._get_session_key(session_id)
            history = await self.get_history(session_id)

            history.append({"role": role, "content": content})

            if len(history) > settings.max_history_messages + 1:
                if history and history[0].get("role") == "system":
                    system_prompt = history[0]

                    history = [system_prompt] + history[-(settings.max_history_messages) :]
                else:
                    history = history[-(settings.max_history_messages) :]

            history_json = json.dumps(history, ensure_ascii=False)
            await self.redis_client.setex(
                session_key,
                settings.session_ttl,
                history_json,
            )

            logger.debug(
                f"💾 [memory_service] Сообщение добавлено в историю сессии {session_id}: {role} ({len(content)} символов)"
            )

        except Exception as e:
            logger.error(f"❌ [memory_service] Ошибка при добавлении сообщения для сессии {session_id}: {e}")

    async def add_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        """
        Добавляет несколько сообщений в историю диалога

        Args:
            session_id: Идентификатор сессии
            messages: Список сообщений в формате OpenAI
        """
        await self._ensure_client()
        if not self.redis_client:
            return

        try:
            session_key = self._get_session_key(session_id)
            history = await self.get_history(session_id)

            history.extend(messages)

            if len(history) > settings.max_history_messages + 1:
                if history and history[0].get("role") == "system":
                    system_prompt = history[0]
                    history = [system_prompt] + history[-(settings.max_history_messages) :]
                else:
                    history = history[-(settings.max_history_messages) :]

            history_json = json.dumps(history, ensure_ascii=False)
            await self.redis_client.setex(
                session_key,
                settings.session_ttl,
                history_json,
            )

            logger.debug(f"💾 [memory_service] Добавлено {len(messages)} сообщений в историю сессии {session_id}")

        except Exception as e:
            logger.error(f"❌ [memory_service] Ошибка при добавлении сообщений для сессии {session_id}: {e}")

    async def clear_history(self, session_id: str) -> None:
        """
        Очищает историю диалога для сессии

        Args:
            session_id: Идентификатор сессии
        """
        await self._ensure_client()
        if not self.redis_client:
            return

        try:
            session_key = self._get_session_key(session_id)
            await self.redis_client.delete(session_key)
            logger.info(f"🗑️ [memory_service] История сессии {session_id} очищена")

        except Exception as e:
            logger.error(f"❌ [memory_service] Ошибка при очистке истории для сессии {session_id}: {e}")

    async def update_ttl(self, session_id: str) -> None:
        """
        Обновляет TTL для сессии (продлевает время жизни)

        Args:
            session_id: Идентификатор сессии
        """
        await self._ensure_client()
        if not self.redis_client:
            return

        try:
            session_key = self._get_session_key(session_id)
            exists = await self.redis_client.exists(session_key)
            if exists:
                await self.redis_client.expire(session_key, settings.session_ttl)
                logger.debug(f"⏰ [memory_service] TTL обновлен для сессии {session_id}")

        except Exception as e:
            logger.error(f"❌ [memory_service] Ошибка при обновлении TTL для сессии {session_id}: {e}")

    async def close(self) -> None:
        """Закрывает соединение с Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            logger.info("🔌 [memory_service] Соединение с Redis закрыто")
