"""
Модуль для извлечения контекста из сообщений.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def extract_contexts(
    messages_dict: Dict[tuple, Dict[str, Any]], 
    channel_id: int, 
    message_id: int, 
    window: int = 2
) -> List[str]:
    """
    Извлекает контекст: сообщение с message_id + до window предыдущих и следующих сообщений
    того же канала по дате.
    
    Args:
        messages_dict: Словарь сообщений, индексированный по (channel_id, id)
        channel_id: ID канала
        message_id: ID целевого сообщения
        window: Количество соседних сообщений с каждой стороны
        
    Returns:
        Список текстов сообщений в порядке времени (старые → новые)
    """
    # Получаем целевое сообщение
    key = (channel_id, message_id)
    if key not in messages_dict:
        logger.warning(f"Сообщение не найдено: channel_id={channel_id}, id_message={message_id}")
        return []
    
    target_message = messages_dict[key]
    target_date_str = target_message.get("date")
    
    if not target_date_str:
        logger.warning(f"У сообщения {key} отсутствует поле date")
        return [target_message.get("text", "")]
    
    try:
        target_date = datetime.fromisoformat(target_date_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as e:
        logger.warning(f"Не удалось распарсить дату {target_date_str}: {e}")
        return [target_message.get("text", "")]
    
    # Собираем все сообщения того же канала
    channel_messages = []
    for (ch_id, msg_id), msg in messages_dict.items():
        if ch_id == channel_id:
            date_str = msg.get("date")
            if date_str:
                try:
                    msg_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    channel_messages.append((msg_date, msg_id, msg))
                except (ValueError, AttributeError):
                    continue
    
    # Сортируем по дате
    channel_messages.sort(key=lambda x: x[0])
    
    # Находим индекс целевого сообщения
    target_idx = None
    for idx, (date, msg_id, msg) in enumerate(channel_messages):
        if msg_id == message_id:
            target_idx = idx
            break
    
    if target_idx is None:
        logger.warning(f"Не удалось найти сообщение {message_id} в отсортированном списке")
        return [target_message.get("text", "")]
    
    # Извлекаем окно сообщений
    start_idx = max(0, target_idx - window)
    end_idx = min(len(channel_messages), target_idx + window + 1)
    
    contexts = []
    for idx in range(start_idx, end_idx):
        _, _, msg = channel_messages[idx]
        text = msg.get("text", "").strip()
        if text:
            contexts.append(text)
    
    logger.debug(
        f"Извлечено {len(contexts)} контекстов для channel_id={channel_id}, "
        f"message_id={message_id} (window={window})"
    )
    
    return contexts


def extract_contexts_by_sources(
    messages_dict: Dict[tuple, Dict[str, Any]], 
    sources_info: List[Dict[str, Any]]
) -> List[str]:
    """
    Извлекает контексты по sources_info, которые вернул generation.
    Сначала пытается использовать metadata (channel_id, message_id), если нет - пытается парсить doc_id.
    
    Args:
        messages_dict: Словарь сообщений, индексированный по (channel_id, id)
        sources_info: Список словарей [{"doc_id": ..., "metadata": {...}}]
        
    Returns:
        Список текстов сообщений в порядке sources_info
    """
    contexts = []
    
    for source in sources_info:
        doc_id = source.get("doc_id", "")
        metadata = source.get("metadata", {})
        
        channel_id = None
        message_id = None
        
        # Пытаемся получить channel_id и message_id из metadata
        if metadata:
            channel_id = metadata.get("channel_id") or metadata.get("id_channel")
            message_id = metadata.get("message_id") or metadata.get("id_message") or metadata.get("id")
        
        # Если не нашли в metadata, пытаемся парсить doc_id
        if channel_id is None or message_id is None:
            try:
                if '_' in doc_id:
                    # Формат "{channel_id}_{message_id}"
                    parts = doc_id.split('_', 1)
                    channel_id = int(parts[0])
                    message_id = int(parts[1])
                else:
                    logger.warning(f"Не удалось определить channel_id и message_id для doc_id={doc_id}, metadata={metadata}")
                    continue
            except (ValueError, IndexError) as e:
                logger.warning(f"Не удалось распарсить doc_id={doc_id}: {e}")
                continue
        
        # Преобразуем в int если нужно
        try:
            channel_id = int(channel_id)
            message_id = int(message_id)
        except (ValueError, TypeError) as e:
            logger.warning(f"Не удалось преобразовать channel_id={channel_id}, message_id={message_id} в int: {e}")
            continue
        
        # Извлекаем текст сообщения из БД
        key = (channel_id, message_id)
        if key in messages_dict:
            msg = messages_dict[key]
            text = msg.get("text", "").strip()
            if text:
                contexts.append(text)
            else:
                logger.warning(f"Сообщение {key} не содержит текста")
        else:
            logger.warning(f"Сообщение не найдено в БД: doc_id={doc_id}, channel_id={channel_id}, message_id={message_id}, metadata={metadata}")
    
    logger.debug(
        f"Извлечено {len(contexts)} контекстов из {len(sources_info)} источников"
    )
    
    return contexts

