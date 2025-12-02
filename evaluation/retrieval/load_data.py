"""Скрипт для загрузки данных в Qdrant из JSON файлов"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Добавляем src в путь
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from tplexity.retriever.retriever_service import RetrieverService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def load_messages_to_qdrant(
    messages_path: str = "src/eval/eval_data/messages_diverse_1000posts_all_channels.json",
    clear_existing: bool = True,
):
    """
    Загрузить сообщения из JSON в Qdrant

    Args:
        messages_path: Путь к JSON файлу с сообщениями
        clear_existing: Очистить существующие документы перед загрузкой
    """
    logger.info("=" * 80)
    logger.info("ЗАГРУЗКА ДАННЫХ В QDRANT")
    logger.info("=" * 80)

    # Загрузка сообщений из файла
    logger.info(f"🔄 Загрузка сообщений из {messages_path}...")
    with open(messages_path, encoding="utf-8") as f:
        messages = json.load(f)
    logger.info(f"✅ Загружено {len(messages)} сообщений из файла")

    # Подготовка данных
    documents = []
    metadatas = []

    for msg in messages:
        text = msg.get("text", "").strip()
        if not text:
            continue

        documents.append(text)

        # Метаданные (составной ID храним в метаданных как doc_id)
        metadata = {
            "doc_id": f"{msg['channel_id']}_{msg['id']}",  # Составной ID для удобства
            "message_id": msg["id"],
            "channel_id": msg["channel_id"],
            "date": msg.get("date", ""),
            "link": msg.get("link", ""),
        }
        metadatas.append(metadata)

    logger.info(f"📊 Подготовлено {len(documents)} документов для индексации")

    # Инициализация RetrieverService
    logger.info("🔄 Инициализация RetrieverService...")
    retriever = RetrieverService()

    # Очистка существующих данных
    if clear_existing:
        logger.warning("⚠️ Удаление существующих документов из Qdrant...")
        await retriever.delete_all_documents()

    # Загрузка документов в Qdrant (UUID будут сгенерированы автоматически)
    logger.info("🔄 Загрузка документов в Qdrant...")
    await retriever.add_documents(documents=documents, metadatas=metadatas)

    logger.info("=" * 80)
    logger.info(f"✅ УСПЕШНО ЗАГРУЖЕНО {len(documents)} ДОКУМЕНТОВ")
    logger.info("=" * 80)

    return len(documents)


async def main():
    """Основная функция"""
    try:
        await load_messages_to_qdrant()
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке данных: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
