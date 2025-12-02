"""
Модуль для загрузки и валидации входных данных.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def load_messages(path: str) -> Dict[tuple, Dict[str, Any]]:
    """
    Загружает сообщения из JSON файла и индексирует их по (channel_id, id).
    
    Args:
        path: Путь к JSON файлу с сообщениями
        
    Returns:
        Словарь с ключами (channel_id, id) и значениями - объектами сообщений
        
    Raises:
        FileNotFoundError: если файл не найден
        json.JSONDecodeError: если файл содержит невалидный JSON
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл с сообщениями не найден: {path}")
    
    logger.info(f"Загрузка сообщений из {path}")
    
    with open(path_obj, "r", encoding="utf-8") as f:
        messages = json.load(f)
    
    if not isinstance(messages, list):
        raise ValueError(f"Ожидался список сообщений, получен {type(messages)}")
    
    # Индексируем по (channel_id, id)
    messages_dict = {}
    for msg in messages:
        channel_id = msg.get("channel_id")
        msg_id = msg.get("id")
        
        if channel_id is None or msg_id is None:
            logger.warning(f"Пропущено сообщение без channel_id или id: {msg}")
            continue
        
        key = (channel_id, msg_id)
        if key in messages_dict:
            logger.warning(f"Дубликат сообщения: channel_id={channel_id}, id={msg_id}")
        
        messages_dict[key] = msg
    
    logger.info(f"Загружено {len(messages_dict)} сообщений")
    return messages_dict


def load_queries(path: str) -> List[Dict[str, Any]]:
    """
    Загружает запросы из JSON файла.
    
    Args:
        path: Путь к JSON файлу с запросами
        
    Returns:
        Список словарей с запросами
        
    Raises:
        FileNotFoundError: если файл не найден
        json.JSONDecodeError: если файл содержит невалидный JSON
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл с запросами не найден: {path}")
    
    logger.info(f"Загрузка запросов из {path}")
    
    with open(path_obj, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    if not isinstance(queries, list):
        raise ValueError(f"Ожидался список запросов, получен {type(queries)}")
    
    logger.info(f"Загружено {len(queries)} запросов")
    return queries


def validate_queries(
    queries: List[Dict[str, Any]], 
    messages_dict: Dict[tuple, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Валидирует запросы: проверяет, существуют ли указанные message ids.
    
    Args:
        queries: Список запросов
        messages_dict: Словарь сообщений, индексированный по (channel_id, id)
        
    Returns:
        Список валидных запросов (с добавленным флагом is_valid)
    """
    validated_queries = []
    missing_count = 0
    
    for query in queries:
        channel_id = query.get("id_channel")
        message_id = query.get("id_message")
        
        if channel_id is None or message_id is None:
            logger.warning(f"Запрос без id_channel или id_message: {query.get('query', 'N/A')}")
            query["is_valid"] = False
            missing_count += 1
        else:
            key = (channel_id, message_id)
            if key in messages_dict:
                query["is_valid"] = True
            else:
                logger.warning(
                    f"Сообщение не найдено: channel_id={channel_id}, "
                    f"id_message={message_id} для запроса: {query.get('query', 'N/A')[:50]}..."
                )
                query["is_valid"] = False
                missing_count += 1
        
        validated_queries.append(query)
    
    valid_count = len(validated_queries) - missing_count
    logger.info(f"Валидация завершена: {valid_count} валидных, {missing_count} невалидных запросов")
    
    return validated_queries

