"""
Главный модуль для запуска evaluation pipeline.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import yaml
from tqdm import tqdm

# Добавляем путь к src для импортов
sys.path.insert(0, str(Path(__file__).parent))

from dataset_loader import load_messages, load_queries, validate_queries
from context_extractor import extract_contexts_by_sources
from inference_client import InferenceClient, get_documents_from_retriever, get_documents_from_retriever_async
from ragas_runner import run_evaluation, run_evaluation_async
from service_manager import ServiceManager


def setup_logging(output_dir: Path):
    """
    Настраивает логирование.
    
    Args:
        output_dir: Директория для сохранения логов
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "run.log"
    
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Устанавливаем уровень WARNING для httpx, чтобы не показывать успешные запросы (200 OK)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Логирование настроено. Логи сохраняются в {log_file}")
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Загружает конфигурацию из YAML файла.
    
    Args:
        config_path: Путь к конфигурационному файлу
        
    Returns:
        Словарь с конфигурацией
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger = logging.getLogger(__name__)
        logger.warning(f"Конфигурационный файл не найден: {config_path}, используются значения по умолчанию")
        return {}
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Обрабатываем переменные окружения
    if isinstance(config, dict):
        config_str = json.dumps(config)
        config_str = os.path.expandvars(config_str)
        config = json.loads(config_str)
    
    return config or {}


def create_sample_dataset(
    messages_dict: Dict,
    queries: List[Dict],
    inference_client: InferenceClient,
    retriever_url: str = "http://localhost:8020",
    inference_timeout: int = 30,
    max_examples: int = 5
) -> List[Dict[str, Any]]:
    """
    Создает sample dataset из запросов.
    
    Args:
        messages_dict: Словарь сообщений
        queries: Список запросов
        inference_client: Клиент для inference
        retriever_url: URL retriever API
        inference_timeout: Таймаут для запросов
        max_examples: Максимальное количество примеров
        
    Returns:
        Список примеров для dataset
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Создание sample dataset (max {max_examples} примеров)")
    
    dataset = []
    valid_queries = [q for q in queries if q.get("is_valid", True)]
    
    for query in valid_queries[:max_examples]:
        question = query.get("query", "")
        
        # Генерируем ответ (generation сам делает поиск через retriever)
        try:
            answer, sources_info, latency_ms = inference_client.generate(question)
            
            # Получаем тексты документов из retriever API (те же, что использовал generation)
            if sources_info:
                doc_ids = [src.get("doc_id", "") for src in sources_info]
                
                # Получаем тексты из retriever API
                documents = get_documents_from_retriever(retriever_url, doc_ids, timeout=inference_timeout)
                
                if documents:
                    # Используем тексты из retriever API (в том же порядке, что и doc_ids)
                    doc_id_to_text = {doc["doc_id"]: doc["text"] for doc in documents}
                    contexts = [doc_id_to_text.get(doc_id, "") for doc_id in doc_ids if doc_id_to_text.get(doc_id)]
                    
                    if not contexts:
                        # Fallback: пытаемся извлечь из messages_dict
                        contexts = extract_contexts_by_sources(messages_dict, sources_info)
                        if not contexts:
                            logger.warning(f"Не удалось извлечь контексты для query: {question[:50]}...")
                            continue
                else:
                    # Fallback: пытаемся извлечь из messages_dict
                    contexts = extract_contexts_by_sources(messages_dict, sources_info)
                    if not contexts:
                        logger.warning(f"Не удалось извлечь контексты для query: {question[:50]}...")
                        continue
            else:
                logger.warning(f"Generation не вернул источников для query: {question[:50]}...")
                continue
            
            # Извлекаем doc_ids для сохранения
            cited_sources = [src.get("doc_id", "") for src in sources_info]
            
            dataset.append({
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "cited_sources": cited_sources,
                "latency_ms": latency_ms
            })
        except Exception as e:
            logger.error(f"Ошибка при создании sample dataset для query: {question[:50]}...: {e}")
            continue
    
    logger.info(f"Создано {len(dataset)} примеров для sample dataset")
    return dataset


def save_dataset_jsonl(dataset: List[Dict[str, Any]], output_path: Path):
    """
    Сохраняет dataset в формате JSONL.
    
    Args:
        dataset: Список примеров
        output_path: Путь для сохранения
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Сохранение dataset в {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            # Сохраняем только question, contexts, answer
            json_line = json.dumps({
                "question": example["question"],
                "contexts": example["contexts"],
                "answer": example["answer"]
            }, ensure_ascii=False)
            f.write(json_line + "\n")
    
    logger.info(f"Dataset сохранен: {len(dataset)} примеров")


def compute_summary(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Вычисляет агрегированные метрики.
    
    Args:
        results_df: DataFrame с результатами
        
    Returns:
        Словарь с агрегированными метриками
    """
    summary = {}
    
    # Метрики для агрегации
    metrics = [
        "relevance", "faithfulness", 
        "completeness", "off_topic_rate", "latency_ms"
    ]
    
    for metric in metrics:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            if len(values) > 0:
                summary[metric] = {
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "count": int(len(values))
                }
    
    # Общая статистика
    summary["total_examples"] = int(len(results_df))
    summary["judge_errors"] = int(results_df.get("judge_errors", pd.Series([False])).sum())
    
    return summary


def main():
    """
    Главная функция CLI.
    """
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--posts",
        type=str,
        required=True,
        help="Путь к JSON файлу с сообщениями"
    )
    
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Путь к JSON файлу с запросами"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="eval/outputs/",
        help="Директория для сохранения результатов (default: eval/outputs/)"
    )
    
    parser.add_argument(
        "--inference-endpoint",
        type=str,
        default="",
        help="URL inference endpoint (если пусто - используется mock)"
    )
    
    parser.add_argument(
        "--judge-model",
        type=str,
        default="",
        help="Модель для judge (например, yandexgpt, qwen, openai:gpt-4o-mini или mock). По умолчанию: yandexgpt"
    )
    
    parser.add_argument(
        "--window",
        type=int,
        default=2,
        help="Размер окна для контекста (default: 2)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="eval/config/model_config.yaml",
        help="Путь к конфигурационному файлу (default: eval/config/model_config.yaml)"
    )
    
    parser.add_argument(
        "--use-ragas",
        action="store_true",
        help="Использовать Ragas для оценки (если доступен)"
    )
    
    parser.add_argument(
        "--auto-start-services",
        action="store_true",
        default=True,
        help="Автоматически запускать необходимые сервисы (по умолчанию: True)"
    )
    
    parser.add_argument(
        "--no-auto-start-services",
        dest="auto_start_services",
        action="store_false",
        help="Не запускать сервисы автоматически"
    )
    
    parser.add_argument(
        "--use-docker",
        action="store_true",
        default=True,
        help="Использовать docker-compose для запуска сервисов (по умолчанию: True)"
    )
    
    parser.add_argument(
        "--no-docker",
        dest="use_docker",
        action="store_false",
        help="Запускать сервисы напрямую (без docker-compose)"
    )
    
    parser.add_argument(
        "--keep-services",
        action="store_true",
        help="Не останавливать сервисы после завершения"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить количество обрабатываемых запросов (для тестирования)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Размер батча для асинхронной обработки запросов (default: 10)"
    )
    
    parser.add_argument(
        "--show-answers",
        action="store_true",
        help="Выводить ответы модели в консоль при генерации"
    )
    
    args = parser.parse_args()
    
    # Настраиваем пути
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Настраиваем логирование
    logger = setup_logging(output_dir)
    
    logger.info("=" * 80)
    logger.info("Запуск RAG Evaluation Pipeline")
    logger.info("=" * 80)
    
    # Инициализируем менеджер сервисов
    service_manager = None
    if args.auto_start_services:
        service_manager = ServiceManager(
            use_docker=args.use_docker,
            auto_stop=not args.keep_services
        )
        logger.info(f"Инициализирован менеджер сервисов (docker={args.use_docker}, auto_stop={not args.keep_services})")
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    
    # Настраиваем inference client
    inference_endpoint = args.inference_endpoint or config.get("inference_endpoint", "")
    inference_api_key = config.get("inference_api_key", "")
    inference_timeout = config.get("inference_timeout", 30)
    
    # Определяем URL retriever API (из inference_endpoint или по умолчанию)
    if inference_endpoint:
        # Если inference_endpoint указан, retriever должен быть на порту 8020
        # Извлекаем базовый URL (например, http://localhost:8022 -> http://localhost:8020)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(inference_endpoint)
            retriever_url = f"{parsed.scheme}://{parsed.hostname}:8020"
        except Exception:
            retriever_url = "http://localhost:8020"
    else:
        retriever_url = "http://localhost:8020"
    
    # Если указан inference endpoint и включен auto-start, проверяем и запускаем сервисы
    if inference_endpoint and service_manager:
        logger.info("Проверка и запуск необходимых сервисов для inference...")
        if not service_manager.ensure_services_for_inference(inference_endpoint, wait=True):
            logger.warning("⚠️ Не удалось запустить необходимые сервисы, будет использован mock inference")
            inference_endpoint = ""  # Переключаемся на mock
    
    inference_client = InferenceClient(
        endpoint_url=inference_endpoint,
        api_key=inference_api_key,
        timeout=inference_timeout
    )
    
    # Настраиваем judge
    judge_config = config.get("judge", {})
    # По умолчанию используем yandexgpt
    if "provider" not in judge_config:
        judge_config["provider"] = "yandexgpt"
    
    if args.judge_model:
        # Парсим формат "provider:model" или просто "provider" или "model"
        if ":" in args.judge_model:
            provider, model = args.judge_model.split(":", 1)
            judge_config["provider"] = provider
            judge_config["model"] = model
        else:
            # Если указан только один параметр, это может быть провайдер или модель
            # Проверяем, является ли это известным провайдером
            if args.judge_model.lower() in ["qwen", "yandexgpt", "openai", "mock"]:
                judge_config["provider"] = args.judge_model.lower()
            else:
                # Иначе считаем это моделью (для обратной совместимости)
                judge_config["model"] = args.judge_model
    
    # Загружаем данные
    logger.info("Загрузка данных...")
    messages_dict = load_messages(args.posts)
    queries = load_queries(args.queries)
    
    # Валидируем запросы
    validated_queries = validate_queries(queries, messages_dict)
    valid_queries = [q for q in validated_queries if q.get("is_valid", True)]
    
    logger.info(f"Валидных запросов: {len(valid_queries)} из {len(queries)}")
    
    if not valid_queries:
        logger.error("Нет валидных запросов для обработки")
        return 1
    
    # Ограничиваем количество запросов, если указан --limit
    if args.limit and args.limit > 0:
        valid_queries = valid_queries[:args.limit]
        logger.info(f"Ограничение: обрабатывается только {len(valid_queries)} запросов (--limit={args.limit})")
    
    # Обрабатываем запросы асинхронно по батчам
    logger.info(f"Обработка запросов асинхронно (batch_size={args.batch_size})...")
    
    async def process_single_query(query: Dict[str, Any], client: InferenceClient) -> Dict[str, Any] | None:
        """Асинхронно обрабатывает один запрос."""
        question = query.get("query", "")
        
        try:
            # Генерируем ответ асинхронно (generation сам делает поиск через retriever)
            answer, sources_info, latency_ms = await client.generate_async(question)
            
            # Выводим ответ, если включен флаг
            if args.show_answers:
                logger.info("=" * 80)
                logger.info(f"Вопрос: {question}")
                logger.info(f"Ответ ({latency_ms:.2f}ms): {answer}")
                logger.info("=" * 80)
            
            # Получаем тексты документов из retriever API (те же, что использовал generation)
            if sources_info:
                doc_ids = [src.get("doc_id", "") for src in sources_info]
                
                # Получаем тексты документов из retriever API асинхронно
                if inference_endpoint:
                    # Используем retriever API для получения текстов (те же, что использовал generation)
                    documents = await get_documents_from_retriever_async(
                        retriever_url, doc_ids, inference_timeout
                    )
                    
                    if documents:
                        # Используем тексты из retriever API (в том же порядке, что и doc_ids)
                        doc_id_to_text = {doc["doc_id"]: doc["text"] for doc in documents}
                        used_contexts = [doc_id_to_text.get(doc_id, "") for doc_id in doc_ids if doc_id_to_text.get(doc_id)]
                        
                        if not used_contexts:
                            logger.warning(f"Не удалось получить тексты документов из retriever API для query: {question[:50]}...")
                            return None
                    else:
                        logger.warning(f"Retriever API не вернул документы для query: {question[:50]}...")
                        # Fallback: пытаемся извлечь из messages_dict
                        used_contexts = extract_contexts_by_sources(messages_dict, sources_info)
                        if not used_contexts:
                            logger.warning(f"Не удалось извлечь контексты из БД для query: {question[:50]}...")
                            return None
                else:
                    # Mock mode: используем fallback на messages_dict
                    used_contexts = extract_contexts_by_sources(messages_dict, sources_info)
                    if not used_contexts:
                        logger.warning(f"Не удалось извлечь контексты из {len(sources_info)} источников для query: {question[:50]}...")
                        return None
            else:
                logger.warning(f"Generation не вернул источников для query: {question[:50]}...")
                # Пропускаем этот пример, т.к. без контекстов оценка не имеет смысла
                return None
            
            # Извлекаем doc_ids для сохранения
            cited_sources = [src.get("doc_id", "") for src in sources_info]
            
            return {
                "question": question,
                "contexts": used_contexts,  # Используем тексты из retriever API (те же, что использовал generation)
                "answer": answer,
                "cited_sources": cited_sources,
                "latency_ms": latency_ms
            }
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа для query: {question[:50]}...: {e}")
            return None
    
    async def process_batch(queries_batch: List[Dict[str, Any]], client: InferenceClient) -> List[Dict[str, Any]]:
        """Обрабатывает батч запросов параллельно."""
        batch_start_time = time.time()
        logger.debug(f"Начало обработки батча из {len(queries_batch)} запросов")
        
        # Создаем задачи для всех запросов в батче - они будут выполняться параллельно
        tasks = [process_single_query(query, client) for query in queries_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_time = (time.time() - batch_start_time) * 1000
        logger.info(f"Батч из {len(queries_batch)} запросов обработан за {batch_time:.2f}ms (среднее: {batch_time/len(queries_batch):.2f}ms на запрос)")
        
        # Фильтруем None и исключения
        examples = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Ошибка при обработке батча: {result}")
            elif result is not None:
                examples.append(result)
        
        return examples
    
    # Обрабатываем запросы батчами с общим HTTP клиентом
    batch_size = args.batch_size
    
    # Создаем один event loop для всех батчей с общим HTTP клиентом
    async def process_all_batches():
        all_examples = []
        # Создаем общий HTTP клиент для всех запросов в батчах
        async with inference_client as client:
            with tqdm(total=len(valid_queries), desc="Обработка запросов") as pbar:
                for i in range(0, len(valid_queries), batch_size):
                    batch = valid_queries[i:i + batch_size]
                    batch_examples = await process_batch(batch, client)
                    all_examples.extend(batch_examples)
                    pbar.update(len(batch))
        return all_examples
    
    # Запускаем асинхронную обработку всех батчей
    examples = asyncio.run(process_all_batches())
    
    logger.info(f"Обработано {len(examples)} примеров")
    
    if not examples:
        logger.error("Нет примеров для оценки")
        return 1
    
    # Сохраняем sample dataset
    data_dir = Path(args.posts).parent
    sample_dataset_path = data_dir / "sample_dataset.jsonl"
    save_dataset_jsonl(examples[:5], sample_dataset_path)
    logger.info(f"Sample dataset сохранен в {sample_dataset_path}")
    
    # Запускаем оценку асинхронно
    logger.info(f"Запуск оценки метрик асинхронно (batch_size={args.batch_size})...")
    results_df = asyncio.run(run_evaluation_async(examples, judge_config, use_ragas=args.use_ragas, batch_size=args.batch_size))
    
    # Сохраняем результаты
    results_path = output_dir / "results.parquet"
    results_df.to_parquet(results_path, index=False)
    logger.info(f"Результаты сохранены в {results_path}")
    
    # Вычисляем summary
    summary = compute_summary(results_df)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary сохранен в {summary_path}")
    
    # Выводим краткую статистику
    logger.info("=" * 80)
    logger.info("Краткая статистика:")
    logger.info("=" * 80)
    for metric, stats in summary.items():
        if isinstance(stats, dict) and "mean" in stats:
            logger.info(f"{metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    logger.info(f"Всего примеров: {summary.get('total_examples', 0)}")
    logger.info(f"Ошибок judge: {summary.get('judge_errors', 0)}")
    logger.info("=" * 80)
    
    logger.info("Оценка завершена успешно!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())