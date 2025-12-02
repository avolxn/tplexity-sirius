import asyncio
import logging
import re
from datetime import datetime
from typing import Any

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

try:
    from .config import settings
    from .service_client import GenerationClient
except ImportError:
    from config import settings
    from service_client import GenerationClient

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_keyboard() -> ReplyKeyboardMarkup:
    """Создает клавиатуру с кнопкой 'Очистить историю'."""
    keyboard = [
        [KeyboardButton(text="🗑️ Очистить историю")],
    ]
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)


def get_clear_history_confirmation_keyboard() -> InlineKeyboardMarkup:
    """Создает inline клавиатуру для подтверждения очистки истории."""
    keyboard = [
        [
            InlineKeyboardButton(text="✅ Да, очистить", callback_data="clear_history_yes"),
            InlineKeyboardButton(text="❌ Отмена", callback_data="clear_history_no"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def escape_html(text: str) -> str:
    """
    Экранирует HTML символы в тексте для безопасного использования в Telegram HTML.

    Args:
        text: Текст для экранирования

    Returns:
        str: Экранированный текст
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def markdown_to_html(text: str) -> str:
    """
    Преобразует Markdown форматирование в HTML для Telegram.

    Преобразования:
    - **текст** → <b>текст</b> (жирный)
    - *текст* → <i>текст</i> (курсив, если не внутри **)
    - `текст` → <code>текст</code> (код)

    Args:
        text: Текст с Markdown форматированием

    Returns:
        str: Текст с HTML форматированием
    """
    if not text:
        return text

    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)

    return text


def extract_channel_name_from_link(link: str) -> str:
    """
    Извлекает название канала из Telegram ссылки.

    Args:
        link: Telegram ссылка (например, https://t.me/selfinvestor/23422)

    Returns:
        str: Название канала (например, selfinvestor)
    """
    import re

    match = re.search(r"https?://t\.me/([^/]+)", link)
    if match:
        channel_name = match.group(1)
        return channel_name.lstrip("@")

    parts = link.rstrip("/").split("/")
    if len(parts) >= 4:
        channel_name = parts[-2]
        return channel_name.lstrip("@")

    return "канал"


def extract_source_link(source: dict, idx: int) -> tuple[str | None, str | None]:
    """
    Извлекает ссылку и название канала из источника.

    Args:
        source: Словарь с источником (содержит metadata)
        idx: Порядковый номер источника (для логирования)

    Returns:
        tuple[str | None, str | None]: (ссылка, название_канала) или (None, None) если не удалось извлечь
    """
    metadata = source.get("metadata") or {}

    link = metadata.get("link")

    if not link:
        channel_id = metadata.get("channel_id")
        message_id = metadata.get("message_id")

        if channel_id and message_id:
            link = f"https://t.me/c/{channel_id}/{message_id}"
            logger.debug(
                f"📋 [bot][bot] extract_source_link: источник {idx} сформирован из channel_id и message_id: {link}"
            )
        else:
            channel_name = metadata.get("channel_name")
            original_id = metadata.get("original_id")
            original_link = metadata.get("original_link")

            if original_link:
                link = original_link
                logger.debug(f"📋 [bot][bot] extract_source_link: источник {idx} использует original_link: {link}")
            elif channel_name and original_id:
                clean_channel = channel_name.lstrip("@")
                link = f"https://t.me/{clean_channel}/{original_id}"
                logger.debug(f"📋 [bot][bot] extract_source_link: источник {idx} сформирован из channel_name: {link}")

    if not link:
        logger.warning(f"⚠️ [bot][bot] Недостаточно данных для источника {idx}: metadata={metadata}")
        return None, None

    channel_name = extract_channel_name_from_link(link)
    return link, channel_name


def extract_citation_numbers(text: str) -> set[int]:
    """
    Извлекает все номера цитат из текста.

    Args:
        text: Текст с цитатами в формате [1], [2], [5][6] и т.д.

    Returns:
        set[int]: Множество номеров цитат, найденных в тексте
    """
    pattern = r"\[(\d+)\]"
    matches = re.findall(pattern, text)
    return {int(match) for match in matches}


def build_citation_map(sources: list[dict], cited_numbers: set[int] | None = None) -> dict[int, str]:
    """
    Создает маппинг номеров цитат к ссылкам источников.

    Args:
        sources: Список источников с метаданными
        cited_numbers: Множество номеров источников, на которые есть ссылки в тексте.
                       Если None, создает маппинг для всех источников.

    Returns:
        dict[int, str]: Словарь {номер_источника: ссылка}
    """
    citation_map = {}

    if cited_numbers:
        for idx in cited_numbers:
            source_idx = idx - 1
            if 0 <= source_idx < len(sources):
                source = sources[source_idx]
                link, _ = extract_source_link(source, idx)
                if link:
                    citation_map[idx] = link
    else:
        for idx, source in enumerate(sources, 1):
            link, _ = extract_source_link(source, idx)
            if link:
                citation_map[idx] = link

    return citation_map


def make_citations_clickable(text: str, citation_map: dict[int, str]) -> str:
    """
    Заменяет цитаты [1], [2], [1][3] в тексте на кликабельные HTML ссылки.
    Каждая цитата становится отдельной ссылкой на соответствующий источник.

    Args:
        text: Текст с цитатами
        citation_map: Словарь {номер_источника: ссылка}

    Returns:
        str: Текст с кликабельными HTML ссылками вместо цитат
    """
    if not citation_map:
        return text

    pattern = r"\[(\d+)\]"

    def replace_citation(match):
        citation_text = match.group(0)
        number = int(match.group(1))

        link = citation_map.get(number)

        if link:
            citation_text_escaped = citation_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            link_escaped = link.replace("&", "&amp;")
            return f'<a href="{link_escaped}">{citation_text_escaped}</a>'
        else:
            return citation_text

    return re.sub(pattern, replace_citation, text)


def format_sources(sources: list[dict], cited_numbers: set[int] | None = None) -> str:
    """
    Форматирует источники для красивого отображения в Telegram.

    Args:
        sources: Список источников с метаданными
        cited_numbers: Множество номеров источников, на которые есть ссылки в тексте.
                       Если None, выводит все источники.

    Returns:
        str: Отформатированная строка с источниками в формате HTML
    """
    if not sources:
        logger.warning("⚠️ [bot][bot] format_sources: sources пуст")
        return ""

    if cited_numbers:
        sorted_numbers = sorted(cited_numbers)
        logger.info(
            f"📋 [bot][bot] format_sources: обрабатываем {len(sorted_numbers)} источников из {len(sources)} доступных"
        )
    else:
        sorted_numbers = list(range(1, len(sources) + 1))
        logger.info(f"📋 [bot][bot] format_sources: обрабатываем все {len(sources)} источников")

    source_items = []
    for idx in sorted_numbers:
        source_idx = idx - 1
        if source_idx < 0 or source_idx >= len(sources):
            logger.warning(
                f"⚠️ [bot][bot] format_sources: источник с номером {idx} не найден (всего источников: {len(sources)})"
            )
            continue

        source = sources[source_idx]
        link, channel_name = extract_source_link(source, idx)
        if not link:
            continue

        metadata = source.get("metadata") or {}

        channel_title = metadata.get("channel_title") or channel_name
        channel_title_escaped = escape_html(channel_title)

        date_str = None
        date_value = metadata.get("date")
        if date_value:
            try:
                if isinstance(date_value, str):
                    if date_value.endswith("Z"):
                        date_value = date_value.replace("Z", "+00:00")

                    if "T" in date_value:
                        post_date = datetime.fromisoformat(date_value)
                    else:
                        post_date = datetime.fromisoformat(f"{date_value}T00:00:00")

                    date_str = post_date.strftime("%d.%m.%Y")
                elif isinstance(date_value, datetime):
                    date_str = date_value.strftime("%d.%m.%Y")
            except (ValueError, AttributeError) as e:
                logger.debug(
                    f"⚠️ [bot][bot] format_sources: не удалось распарсить дату для источника {idx}: {date_value}, ошибка: {e}"
                )

        link_escaped = link.replace("&", "&amp;")

        if date_str:
            source_items.append(f'[{idx}]: <a href="{link_escaped}">{channel_title_escaped}</a> ({date_str})')
        else:
            source_items.append(f'[{idx}]: <a href="{link_escaped}">{channel_title_escaped}</a>')

    if not source_items:
        logger.warning("⚠️ [bot][bot] format_sources: не удалось сформировать ни одной ссылки")
        return ""

    sources_text = "\n".join(source_items)
    logger.info(f"📋 [bot][bot] format_sources: сформирован текст с {len(source_items)} источниками")
    return sources_text


async def start_handler(message: Message) -> None:
    """Обработчик команды /start."""
    welcome_message = """
🟨 <b>Добро пожаловать в T-Plexity!</b>

<b>Интеллектуальная система для работы с инвестиционными новостями</b>

Я отслеживаю в реальном времени публикации из проверенных инвестиционных Telegram-каналов и даю точные, контекстные ответы на ваши вопросы о рынках и новостях.

<b>⚡ Что я умею:</b>
• Отвечать на вопросы о финансовых рынках и новостях
• Работать на самых актуальных данных (минимальная задержка)
• Показывать источники — каждый ответ с ссылками на конкретные сообщения из каналов
• Давать точные ответы с рыночным контекстом

<b>📝 Как пользоваться:</b>
Просто напишите ваш вопрос о рынках или новостях, и я найду актуальную информацию!

Используйте кнопки меню для управления настройками.
    """
    await message.answer(welcome_message, reply_markup=get_keyboard(), parse_mode="HTML")


async def help_handler(message: Message) -> None:
    """Обработчик команды /help."""
    help_text = """
<b>ℹ️ Справка по использованию T-Plexity</b>

<b>📊 О системе:</b>
T-Plexity — интеллектуальная система, которая в реальном времени отслеживает и агрегирует свежие публикации из проверенных инвестиционных Telegram-каналов. Система работает на самых актуальных данных с минимальной задержкой.

<b>📚 Источники информации:</b>
• Только инвестиционные Telegram-каналы, отобранные по качеству и надежности
• Каждый ответ сопровождается ссылками на первоисточники (конкретные сообщения из каналов)

<b>💡 Как использовать:</b>
Просто напишите вопрос о рынках или новостях — я найду актуальную информацию и дам точный ответ с рыночным контекстом.

<b>⚙️ Доступные команды:</b>
/start — Перезапустить бота
/help — Показать эту справку

<b>🔘 Кнопки меню:</b>
🗑️ Очистить историю — удалить контекст диалога

<b>✨ Особенности:</b>
• Источники отображаются под каждым ответом с прямыми ссылками
• История диалога сохраняется для контекста
• Актуальность данных — минимальная задержка между публикацией и возможностью ответить
    """
    await message.answer(help_text, reply_markup=get_keyboard(), parse_mode="HTML")


# Хранилище для данных сообщений (в production лучше использовать Redis или БД)
_message_data: dict[int, dict[str, Any]] = {}


async def echo_handler(message: Message, bot: Bot, generation_client: GenerationClient) -> None:
    """Обрабатывает текстовые сообщения от пользователя."""
    user_message = message.text
    if not user_message:
        return

    logger.info(f"Получено сообщение от {message.from_user.username}: {user_message}")

    if user_message == "🗑️ Очистить историю" or user_message == "Удалить историю из памяти":
        await message.answer(
            "⚠️ <b>Вы уверены, что хотите очистить историю диалога?</b>\n\n"
            "Все контекстные данные будут удалены, и диалог начнется заново.",
            reply_markup=get_clear_history_confirmation_keyboard(),
            parse_mode="HTML",
        )
        return

    if not generation_client:
        await message.answer(
            "❌ <b>Ошибка:</b> Сервис генерации недоступен.\n\n"
            "Пожалуйста, попробуйте позже или обратитесь к администратору.",
            reply_markup=get_keyboard(),
            parse_mode="HTML",
        )
        logger.error("Generation client not found")
        return

    selected_model = "deepseek"
    logger.info(f"📌 [bot][bot] Использование модели: {selected_model}")

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    try:
        user_id = message.from_user.id
        session_id = f"tg:{user_id}"

        answer, _, sources, search_time, generation_time, total_time = await generation_client.send_message(
            user_message, llm_provider=selected_model, session_id=session_id
        )

        logger.info(f"📋 [bot][bot] Получено источников: {len(sources)}")
        if sources:
            logger.debug(f"📋 [bot][bot] Первый источник: {sources[0] if sources else 'нет'}")

        answer_html = markdown_to_html(answer)

        cited_numbers = extract_citation_numbers(answer_html)
        logger.info(f"📋 [bot][bot] Найдено цитат в тексте: {cited_numbers}")

        citation_map = build_citation_map(sources, cited_numbers)

        answer_with_citations = make_citations_clickable(answer_html, citation_map)

        sources_text = format_sources(sources, cited_numbers)

        logger.info(
            f"📋 [bot][bot] Отформатированный текст источников: {sources_text[:100] if sources_text else 'пусто'}..."
        )

        if sources_text:
            response_text = f"{answer_with_citations}\n\n{sources_text}"
        else:
            response_text = answer_with_citations

        used_rag = len(sources) > 0

        reply_markup = None
        if used_rag:
            keyboard = [
                [InlineKeyboardButton(text="📝 Краткий ответ", callback_data=f"short_answer:{message.message_id}")]
            ]
            reply_markup = InlineKeyboardMarkup(inline_keyboard=keyboard)

            message_key = f"detailed_answer_{message.message_id}"
            _message_data[message.message_id] = {
                "detailed_answer": answer_with_citations,
                "sources_text": sources_text,
                "sources": sources,
                "citation_map": citation_map,
            }

        sent_message = await message.answer(
            response_text, disable_web_page_preview=True, parse_mode="HTML", reply_markup=reply_markup
        )

        if used_rag:
            _message_data[message.message_id]["sent_message_id"] = sent_message.message_id

    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}", exc_info=True)
        await message.answer(
            f"❌ <b>Произошла ошибка</b>\n\n"
            f"Не удалось обработать ваше сообщение.\n\n"
            f"<i>Детали: {escape_html(str(e))}</i>\n\n"
            f"Пожалуйста, попробуйте еще раз или обратитесь к администратору.",
            reply_markup=get_keyboard(),
            parse_mode="HTML",
        )


async def short_answer_callback(callback_query: CallbackQuery, bot: Bot, generation_client: GenerationClient) -> None:
    """Обработчик нажатия на кнопки 'Краткий ответ' и 'Подробный ответ'."""
    await callback_query.answer()

    if callback_query.data and callback_query.data.startswith("short_answer:"):
        original_message_id = int(callback_query.data.split(":")[1])
        saved_data = _message_data.get(original_message_id)

        if not saved_data:
            await callback_query.message.edit_text(
                "❌ <b>Ошибка</b>\n\nНе удалось найти детальный ответ. Попробуйте задать вопрос снова.",
                parse_mode="HTML",
            )
            logger.error(f"Не найдены сохраненные данные для message_id={original_message_id}")
            return

        if not generation_client:
            await callback_query.message.edit_text(
                "❌ <b>Ошибка</b>\n\nСервис генерации недоступен.",
                parse_mode="HTML",
            )
            logger.error("Generation client not found")
            return

        await bot.send_chat_action(chat_id=callback_query.message.chat.id, action=ChatAction.TYPING)

        try:
            selected_model = "deepseek"

            detailed_answer = saved_data["detailed_answer"]
            short_answer = await generation_client.generate_short_answer(
                detailed_answer=detailed_answer, llm_provider=selected_model
            )

            short_answer_html = markdown_to_html(short_answer)

            citation_map = saved_data.get("citation_map", {})
            short_answer_with_citations = make_citations_clickable(short_answer_html, citation_map)

            sources_text = saved_data.get("sources_text", "")
            if sources_text:
                response_text = f"{short_answer_with_citations}\n\n{sources_text}"
            else:
                response_text = short_answer_with_citations

            keyboard = [
                [
                    InlineKeyboardButton(
                        text="📄 Подробный ответ", callback_data=f"detailed_answer:{original_message_id}"
                    )
                ]
            ]
            reply_markup = InlineKeyboardMarkup(inline_keyboard=keyboard)

            sent_message_id = saved_data.get("sent_message_id")
            if sent_message_id:
                await bot.edit_message_text(
                    chat_id=callback_query.message.chat.id,
                    message_id=sent_message_id,
                    text=response_text,
                    disable_web_page_preview=True,
                    parse_mode="HTML",
                    reply_markup=reply_markup,
                )
            else:
                await callback_query.message.edit_text(
                    response_text,
                    disable_web_page_preview=True,
                    parse_mode="HTML",
                    reply_markup=reply_markup,
                )

        except Exception as e:
            logger.error(f"Ошибка при генерации краткого ответа: {e}", exc_info=True)
            await callback_query.message.edit_text(
                f"❌ <b>Произошла ошибка</b>\n\nНе удалось сгенерировать краткий ответ.\n\n<i>Детали: {escape_html(str(e))}</i>",
                parse_mode="HTML",
            )

    elif callback_query.data and callback_query.data.startswith("detailed_answer:"):
        original_message_id = int(callback_query.data.split(":")[1])
        saved_data = _message_data.get(original_message_id)

        if not saved_data:
            await callback_query.message.edit_text(
                "❌ <b>Ошибка</b>\n\nНе удалось найти детальный ответ. Попробуйте задать вопрос снова.",
                parse_mode="HTML",
            )
            logger.error(f"Не найдены сохраненные данные для message_id={original_message_id}")
            return

        detailed_answer = saved_data["detailed_answer"]
        sources_text = saved_data.get("sources_text", "")
        if sources_text:
            response_text = f"{detailed_answer}\n\n{sources_text}"
        else:
            response_text = detailed_answer

        keyboard = [
            [InlineKeyboardButton(text="📝 Краткий ответ", callback_data=f"short_answer:{original_message_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(inline_keyboard=keyboard)

        sent_message_id = saved_data.get("sent_message_id")
        if sent_message_id:
            await bot.edit_message_text(
                chat_id=callback_query.message.chat.id,
                message_id=sent_message_id,
                text=response_text,
                disable_web_page_preview=True,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )
        else:
            await callback_query.message.edit_text(
                response_text,
                disable_web_page_preview=True,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )


async def clear_history_callback(callback_query: CallbackQuery, bot: Bot, generation_client: GenerationClient) -> None:
    """Обработчик подтверждения очистки истории через inline кнопки."""
    await callback_query.answer()

    if callback_query.data == "clear_history_yes":
        if not generation_client:
            await callback_query.message.edit_text(
                "❌ <b>Ошибка</b>\n\nСервис генерации недоступен. Пожалуйста, попробуйте позже.",
                reply_markup=None,
                parse_mode="HTML",
            )
            logger.error("Generation client not found")
            return

        user_id = callback_query.from_user.id
        session_id = f"tg:{user_id}"

        try:
            await generation_client.clear_session(session_id)
            await callback_query.message.edit_text(
                "✅ <b>История очищена!</b>\n\nВсе данные диалога удалены. Вы можете начать новый диалог.",
                reply_markup=None,
                parse_mode="HTML",
            )
            logger.info(f"Пользователь {callback_query.from_user.username} очистил историю диалога")
        except Exception as e:
            logger.error(f"Ошибка при очистке истории: {e}", exc_info=True)
            await callback_query.message.edit_text(
                f"❌ <b>Ошибка при очистке истории</b>\n\n<i>{str(e)}</i>",
                reply_markup=None,
                parse_mode="HTML",
            )

    elif callback_query.data == "clear_history_no":
        await callback_query.message.edit_text(
            "✅ <b>Очистка отменена</b>\n\nИстория диалога сохранена.",
            reply_markup=None,
            parse_mode="HTML",
        )
        logger.info(f"Пользователь {callback_query.from_user.username} отменил очистку истории")


def create_router(generation_client: GenerationClient) -> Router:
    """
    Создает router с обработчиками для Telegram бота.

    Args:
        generation_client: Клиент для взаимодействия с Generation API

    Returns:
        Router: Router с зарегистрированными обработчиками
    """
    router = Router()

    # Команды
    router.message.register(start_handler, Command("start"))
    router.message.register(help_handler, Command("help"))

    # Callback queries
    router.callback_query.register(clear_history_callback, F.data.startswith("clear_history_"))
    router.callback_query.register(short_answer_callback, F.data.regexp(r"^(short_answer|detailed_answer):"))

    # Текстовые сообщения
    async def echo_wrapper(message: Message, bot: Bot) -> None:
        await echo_handler(message, bot, generation_client)

    router.message.register(echo_wrapper, F.text)

    logger.info("✅ Обработчики Telegram бота зарегистрированы")
    return router


async def main() -> None:
    """Запуск бота."""
    bot_token = settings.bot_token

    if not bot_token:
        logger.error("❌ BOT_TOKEN не установлен в .env файле!")
        logger.error("Пожалуйста, установите токен бота в файле .env")
        return

    generation_client = GenerationClient()
    bot = Bot(token=bot_token)
    dp = Dispatcher()

    router = create_router(generation_client)
    dp.include_router(router)

    logger.info("🤖 Бот запущен...")
    try:
        commands = [
            BotCommand(command="start", description="🟨 Запустить бота"),
            BotCommand(command="help", description="ℹ️ Справка"),
        ]
        await bot.set_my_commands(commands)

        await dp.start_polling(bot, allowed_updates=["message", "callback_query"], drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("Остановка бота...")
    finally:
        await generation_client.close()
        logger.info("Соединение с Generation API закрыто")
        await bot.session.close()


def register_handlers(dp: Dispatcher, generation_client: GenerationClient) -> None:
    """
    Регистрирует обработчики для Telegram бота.
    Используется при запуске через FastAPI.

    Args:
        dp: Экземпляр Dispatcher
        generation_client: Клиент для взаимодействия с Generation API
    """
    router = create_router(generation_client)
    dp.include_router(router)


async def start_polling(bot: Bot, dp: Dispatcher) -> None:
    """
    Запускает polling для Telegram бота.
    Используется при запуске через FastAPI.

    Args:
        bot: Экземпляр Bot
        dp: Экземпляр Dispatcher
    """
    try:
        commands = [
            BotCommand(command="start", description="🟨 Запустить бота"),
            BotCommand(command="help", description="ℹ️ Справка"),
        ]
        await bot.set_my_commands(commands)

        await dp.start_polling(bot, allowed_updates=["message", "callback_query"], drop_pending_updates=True)
        logger.info("✅ Polling запущен")

        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Остановка бота (polling отменен)...")
    except Exception as e:
        logger.error(f"Ошибка в polling: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
