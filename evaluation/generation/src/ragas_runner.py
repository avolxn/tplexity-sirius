"""
Модуль для запуска оценки через Ragas и кастомные метрики.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Добавляем путь к проекту для импорта tplexity
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from .custom_metrics import (
        JudgeClient,
        score_relevance,
        score_faithfulness,
        score_completeness,
        score_all_metrics,
        score_all_metrics_async
    )
except ImportError:
    from custom_metrics import (
        JudgeClient,
        score_relevance,
        score_faithfulness,
        score_completeness,
        score_all_metrics,
        score_all_metrics_async
    )

logger = logging.getLogger(__name__)


def _print_intermediate_metrics(results: List[Dict[str, Any]], current_idx: int, total: int):
    """
    Выводит промежуточные метрики для уже обработанных примеров.
    
    Args:
        results: Список словарей с результатами оценки
        current_idx: Текущий индекс (0-based)
        total: Общее количество примеров
    """
    if not results:
        return
    
    # Создаем DataFrame из текущих результатов
    df = pd.DataFrame(results)
    
    # Метрики для вывода
    metrics = ["relevance", "faithfulness", "completeness", "off_topic_rate", "latency_ms"]
    
    # Вычисляем средние значения
    metric_values = {}
    for metric in metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                metric_values[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std())
                }
    
    # Выводим промежуточные метрики
    logger.info("=" * 80)
    logger.info(f"📊 Промежуточные метрики (обработано {current_idx + 1}/{total}):")
    logger.info("-" * 80)
    for metric in metrics:
        if metric in metric_values:
            mean = metric_values[metric]["mean"]
            std = metric_values[metric]["std"]
            logger.info(f"  {metric:20s}: {mean:.4f} ± {std:.4f}")
    logger.info("=" * 80)


# Попытка импортировать Ragas
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall
    )
    RAGAS_AVAILABLE = True
    logger.info("Ragas доступен")
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("Ragas не установлен, используется fallback на кастомные метрики")

# Попытка импортировать LangChain для обертки LLM
try:
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.debug("LangChain не доступен для обертки LLM, Ragas будет использовать дефолтные настройки")


def _create_ragas_llm(judge_config: Dict[str, Any]):
    """
    Создает LLM для Ragas на основе конфигурации judge.
    
    Args:
        judge_config: Конфигурация judge модели
        
    Returns:
        LLM обертка для Ragas или None (если не удалось создать)
    """
    provider = judge_config.get("provider", "qwen").lower()
    
    if provider == "qwen":
        try:
            from tplexity.llm_client.config import settings as llm_settings
            
            # Создаем LangChain LLM для Qwen через OpenAI-совместимый API
            if LANGCHAIN_AVAILABLE:
                llm = ChatOpenAI(
                    model=llm_settings.qwen_model,
                    api_key=llm_settings.qwen_api_key,
                    base_url=llm_settings.qwen_base_url,
                    temperature=judge_config.get("temperature", 0.0),
                    timeout=judge_config.get("timeout", 30)
                )
                ragas_llm = LangchainLLMWrapper(llm)
                logger.info(f"Создан Ragas LLM обертка для Qwen: {llm_settings.qwen_model}")
                return ragas_llm
            else:
                logger.warning("LangChain не доступен, Ragas будет использовать дефолтные настройки")
                return None
                
        except Exception as e:
            logger.warning(f"Не удалось создать Ragas LLM для Qwen: {e}")
            return None
    
    elif provider == "yandexgpt":
        try:
            from tplexity.llm_client.config import settings as llm_settings
            
            # Создаем LangChain LLM для YandexGPT через OpenAI-совместимый API
            if LANGCHAIN_AVAILABLE:
                # YandexGPT требует folder_id в заголовках
                model_name = f"gpt://{llm_settings.yandexgpt_folder_id}/{llm_settings.yandexgpt_model}"
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=llm_settings.yandexgpt_api_key,
                    base_url=llm_settings.yandexgpt_base_url,
                    temperature=judge_config.get("temperature", 0.0),
                    timeout=judge_config.get("timeout", 30),
                    default_headers={"x-folder-id": llm_settings.yandexgpt_folder_id}
                )
                ragas_llm = LangchainLLMWrapper(llm)
                logger.info(f"Создан Ragas LLM обертка для YandexGPT: {llm_settings.yandexgpt_model}")
                return ragas_llm
            else:
                logger.warning("LangChain не доступен, Ragas будет использовать дефолтные настройки")
                return None
                
        except Exception as e:
            logger.warning(f"Не удалось создать Ragas LLM для YandexGPT: {e}")
            return None
    
    elif provider == "openai":
        try:
            api_key = judge_config.get("api_key") or os.getenv("OPENAI_API_KEY")
            model = judge_config.get("model", "gpt-4o-mini")
            
            if LANGCHAIN_AVAILABLE and api_key:
                llm = ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    temperature=judge_config.get("temperature", 0.0),
                    timeout=judge_config.get("timeout", 30)
                )
                ragas_llm = LangchainLLMWrapper(llm)
                logger.info(f"Создан Ragas LLM обертка для OpenAI: {model}")
                return ragas_llm
            else:
                logger.warning("OpenAI API key не найден или LangChain не доступен")
                return None
                
        except Exception as e:
            logger.warning(f"Не удалось создать Ragas LLM для OpenAI: {e}")
            return None
    
    else:
        logger.warning(f"Провайдер {provider} не поддерживается для Ragas LLM")
        return None


async def run_evaluation_async(
    examples: List[Dict[str, Any]],
    judge_config: Dict[str, Any],
    use_ragas: bool = True,
    batch_size: int = 10
) -> pd.DataFrame:
    """
    Асинхронная версия run_evaluation с батч-обработкой.
    
    Args:
        examples: Список примеров с полями question, contexts, answer, cited_sources, latency_ms
        judge_config: Конфигурация judge модели
        use_ragas: Использовать ли Ragas (если доступен)
        batch_size: Размер батча для асинхронной обработки
        
    Returns:
        DataFrame с метриками для каждого примера
    """
    if use_ragas and RAGAS_AVAILABLE:
        return await _run_ragas_evaluation_async(examples, judge_config, batch_size)
    else:
        return await _run_custom_evaluation_async(examples, judge_config, batch_size)


async def _run_ragas_evaluation_async(
    examples: List[Dict[str, Any]],
    judge_config: Dict[str, Any],
    batch_size: int
) -> pd.DataFrame:
    """
    Асинхронная версия _run_ragas_evaluation с батч-обработкой.
    """
    logger.info(f"Запуск оценки через Ragas для {len(examples)} примеров (batch_size={batch_size})")
    
    # Создаем LLM для Ragas (синхронная часть)
    ragas_llm = _create_ragas_llm(judge_config)
    if ragas_llm:
        logger.info("Используется кастомная LLM для Ragas")
    else:
        logger.info("Ragas будет использовать дефолтную LLM (проверьте переменные окружения)")
    
    # Подготавливаем данные для Ragas
    ragas_data = []
    for ex in examples:
        ragas_data.append({
            "question": ex["question"],
            "contexts": ex["contexts"],
            "answer": ex["answer"],
            "ground_truth": ""
        })
    
    # Запускаем Ragas метрики (синхронная часть, выполняется один раз)
    try:
        ragas_df = pd.DataFrame(ragas_data)
        
        # Формируем метрики (синхронная часть)
        if ragas_llm:
            metrics = []
            metric_classes = [
                (answer_relevancy, "answer_relevancy"),
                (faithfulness, "faithfulness"),
                (context_precision, "context_precision"),
                (context_recall, "context_recall")
            ]
            
            for metric_class, metric_name in metric_classes:
                try:
                    if callable(metric_class) and not isinstance(metric_class, type):
                        metric = metric_class(llm=ragas_llm)
                    elif isinstance(metric_class, type):
                        metric = metric_class(llm=ragas_llm)
                    else:
                        metric = metric_class
                        if hasattr(metric, 'llm'):
                            metric.llm = ragas_llm
                        elif hasattr(metric, '_llm'):
                            metric._llm = ragas_llm
                    metrics.append(metric)
                except (TypeError, AttributeError) as e:
                    logger.debug(f"Метрика {metric_name} не поддерживает llm параметр, используется стандартная: {e}")
                    metrics.append(metric_class)
            
            if any(hasattr(m, 'llm') or hasattr(m, '_llm') for m in metrics if hasattr(m, '__dict__')):
                logger.info("Метрики Ragas настроены с кастомной LLM")
        else:
            metrics = [
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall
            ]
        
        # Запускаем Ragas оценку (синхронная часть)
        evaluate_kwargs = {
            "dataset": ragas_df,
            "metrics": metrics
        }
        
        if ragas_llm:
            import inspect
            try:
                sig = inspect.signature(evaluate)
                if 'llm' in sig.parameters:
                    evaluate_kwargs['llm'] = ragas_llm
                    logger.info("LLM передана в evaluate()")
            except Exception as e:
                logger.debug(f"Не удалось передать LLM в evaluate: {e}")
        
        # Выполняем Ragas оценку в отдельном потоке
        ragas_results = await asyncio.to_thread(evaluate, **evaluate_kwargs)
        ragas_metrics_df = ragas_results.to_pandas()
        
    except Exception as e:
        logger.error(f"Ошибка при запуске Ragas: {e}")
        logger.warning("Переключение на кастомные метрики")
        return await _run_custom_evaluation_async(examples, judge_config, batch_size)
    
    # Обрабатываем кастомные метрики асинхронно батчами
    judge_client = JudgeClient(
        provider=judge_config.get("provider", "qwen"),
        model=judge_config.get("model", ""),
        api_key=judge_config.get("api_key"),
        temperature=judge_config.get("temperature", 0.0),
        max_retries=judge_config.get("max_retries", 2),
        timeout=judge_config.get("timeout", 30)
    )
    
    async def process_single_example(idx: int, ex: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает один пример асинхронно."""
        result_row = {
            "query_id": idx,
            "question": ex["question"],
            "n_contexts": len(ex["contexts"]),
            "latency_ms": ex.get("latency_ms", 0.0)
        }
        
        # Метрики из Ragas
        if idx < len(ragas_metrics_df):
            result_row["relevance"] = ragas_metrics_df.iloc[idx].get("answer_relevancy", 0.0)
            result_row["faithfulness"] = ragas_metrics_df.iloc[idx].get("faithfulness", 0.0)
            result_row["context_precision"] = ragas_metrics_df.iloc[idx].get("context_precision", 0.0)
            result_row["context_recall"] = ragas_metrics_df.iloc[idx].get("context_recall", 0.0)
        else:
            result_row["relevance"] = 0.0
            result_row["faithfulness"] = 0.0
            result_row["context_precision"] = 0.0
            result_row["context_recall"] = 0.0
        
        # Оцениваем кастомные метрики асинхронно
        cited_sources = ex.get("cited_sources", [])
        
        if result_row.get("faithfulness", 0.0) > 0.0:
            _, _, completeness, off_topic_rate, has_error = await score_all_metrics_async(
                judge_client, ex["question"], ex["answer"], ex["contexts"], cited_sources
            )
        else:
            relevance_custom, faithfulness_custom, completeness, off_topic_rate, has_error = await score_all_metrics_async(
                judge_client, ex["question"], ex["answer"], ex["contexts"], cited_sources
            )
            result_row["relevance"] = relevance_custom
            result_row["faithfulness"] = faithfulness_custom
        
        result_row["completeness"] = completeness
        result_row["off_topic_rate"] = off_topic_rate
        result_row["judge_errors"] = has_error
        
        return result_row
    
    async def process_batch(batch_examples: List[tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Обрабатывает батч примеров параллельно."""
        tasks = [process_single_example(idx, ex) for idx, ex in batch_examples]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Ошибка при обработке примера: {result}")
            else:
                batch_results.append(result)
        
        return batch_results
    
    # Обрабатываем примеры батчами
    results = []
    from tqdm import tqdm
    
    with tqdm(total=len(examples), desc="Оценка метрик") as pbar:
        for i in range(0, len(examples), batch_size):
            batch = [(idx, ex) for idx, ex in enumerate(examples[i:i + batch_size], start=i)]
            batch_results = await process_batch(batch)
            results.extend(batch_results)
            pbar.update(len(batch))
    
    return pd.DataFrame(results)


async def _run_custom_evaluation_async(
    examples: List[Dict[str, Any]],
    judge_config: Dict[str, Any],
    batch_size: int
) -> pd.DataFrame:
    """
    Асинхронная версия _run_custom_evaluation с батч-обработкой.
    """
    logger.info(f"Запуск оценки через кастомные метрики для {len(examples)} примеров (batch_size={batch_size})")
    
    judge_client = JudgeClient(
        provider=judge_config.get("provider", "qwen"),
        model=judge_config.get("model", ""),
        api_key=judge_config.get("api_key"),
        temperature=judge_config.get("temperature", 0.0),
        max_retries=judge_config.get("max_retries", 2),
        timeout=judge_config.get("timeout", 30)
    )
    
    async def process_single_example(idx: int, ex: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает один пример асинхронно."""
        result_row = {
            "query_id": idx,
            "question": ex["question"],
            "n_contexts": len(ex["contexts"]),
            "latency_ms": ex.get("latency_ms", 0.0)
        }
        
        # Оцениваем все метрики асинхронно
        cited_sources = ex.get("cited_sources", [])
        relevance_score, faithfulness_score, completeness_score, off_topic_rate, has_error = await score_all_metrics_async(
            judge_client, ex["question"], ex["answer"], ex["contexts"], cited_sources
        )
        
        result_row["relevance"] = relevance_score
        result_row["faithfulness"] = faithfulness_score
        result_row["completeness"] = completeness_score
        result_row["off_topic_rate"] = off_topic_rate
        result_row["context_precision"] = faithfulness_score  # Используем faithfulness как proxy
        result_row["context_recall"] = min(1.0, len(ex["contexts"]) / 5.0)  # Упрощенная метрика
        result_row["judge_errors"] = has_error
        
        return result_row
    
    async def process_batch(batch_examples: List[tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Обрабатывает батч примеров параллельно."""
        tasks = [process_single_example(idx, ex) for idx, ex in batch_examples]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Ошибка при обработке примера: {result}")
            else:
                batch_results.append(result)
        
        return batch_results
    
    # Обрабатываем примеры батчами
    results = []
    from tqdm import tqdm
    
    with tqdm(total=len(examples), desc="Оценка метрик") as pbar:
        for i in range(0, len(examples), batch_size):
            batch = [(idx, ex) for idx, ex in enumerate(examples[i:i + batch_size], start=i)]
            batch_results = await process_batch(batch)
            results.extend(batch_results)
            pbar.update(len(batch))
    
    return pd.DataFrame(results)


def run_evaluation(
    examples: List[Dict[str, Any]],
    judge_config: Dict[str, Any],
    use_ragas: bool = True
) -> pd.DataFrame:
    """
    Запускает оценку на примерах.
    
    Args:
        examples: Список примеров с полями question, contexts, answer, cited_sources, latency_ms
        judge_config: Конфигурация judge модели
        use_ragas: Использовать ли Ragas (если доступен)
        
    Returns:
        DataFrame с метриками для каждого примера
    """
    if use_ragas and RAGAS_AVAILABLE:
        return _run_ragas_evaluation(examples, judge_config)
    else:
        return _run_custom_evaluation(examples, judge_config)


def _run_ragas_evaluation(
    examples: List[Dict[str, Any]],
    judge_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Запускает оценку через Ragas с добавлением кастомных метрик.
    """
    logger.info(f"Запуск оценки через Ragas для {len(examples)} примеров")
    
    # Создаем LLM для Ragas
    ragas_llm = _create_ragas_llm(judge_config)
    if ragas_llm:
        logger.info("Используется кастомная LLM для Ragas")
    else:
        logger.info("Ragas будет использовать дефолтную LLM (проверьте переменные окружения)")
    
    # Подготавливаем данные для Ragas
    ragas_data = []
    for ex in examples:
        ragas_data.append({
            "question": ex["question"],
            "contexts": ex["contexts"],
            "answer": ex["answer"],
            "ground_truth": ""  # Ragas требует это поле, но мы его не используем
        })
    
    # Запускаем Ragas метрики
    try:
        ragas_df = pd.DataFrame(ragas_data)
        
        # Формируем метрики с LLM, если доступна
        # В Ragas метрики обычно можно инициализировать с параметром llm
        if ragas_llm:
            # Пытаемся создать метрики с кастомной LLM
            # В Ragas метрики могут быть классами, которые принимают llm при инициализации
            metrics = []
            metric_classes = [
                (answer_relevancy, "answer_relevancy"),
                (faithfulness, "faithfulness"),
                (context_precision, "context_precision"),
                (context_recall, "context_recall")
            ]
            
            for metric_class, metric_name in metric_classes:
                try:
                    # Пытаемся создать метрику с LLM
                    if callable(metric_class) and not isinstance(metric_class, type):
                        # Если это функция/фабрика
                        metric = metric_class(llm=ragas_llm)
                    elif isinstance(metric_class, type):
                        # Если это класс
                        metric = metric_class(llm=ragas_llm)
                    else:
                        # Если это уже экземпляр, пытаемся установить LLM
                        metric = metric_class
                        if hasattr(metric, 'llm'):
                            metric.llm = ragas_llm
                        elif hasattr(metric, '_llm'):
                            metric._llm = ragas_llm
                    metrics.append(metric)
                except (TypeError, AttributeError) as e:
                    # Если не получилось создать с LLM, используем стандартную метрику
                    logger.debug(f"Метрика {metric_name} не поддерживает llm параметр, используется стандартная: {e}")
                    metrics.append(metric_class)
            
            if any(hasattr(m, 'llm') or hasattr(m, '_llm') for m in metrics if hasattr(m, '__dict__')):
                logger.info("Метрики Ragas настроены с кастомной LLM")
        else:
            # Используем стандартные метрики без кастомной LLM
            metrics = [
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall
            ]
        
        # Запускаем оценку
        evaluate_kwargs = {
            "dataset": ragas_df,
            "metrics": metrics
        }
        
        # Если Ragas поддерживает передачу LLM напрямую в evaluate
        if ragas_llm:
            import inspect
            try:
                sig = inspect.signature(evaluate)
                if 'llm' in sig.parameters:
                    evaluate_kwargs['llm'] = ragas_llm
                    logger.info("LLM передана в evaluate()")
            except Exception as e:
                logger.debug(f"Не удалось передать LLM в evaluate: {e}")
        
        ragas_results = evaluate(**evaluate_kwargs)
        
        # Конвертируем результаты Ragas в DataFrame
        ragas_metrics_df = ragas_results.to_pandas()
        
    except Exception as e:
        logger.error(f"Ошибка при запуске Ragas: {e}")
        logger.warning("Переключение на кастомные метрики")
        return _run_custom_evaluation(examples, judge_config)
    
    # Добавляем кастомные метрики через judge
    judge_client = JudgeClient(
        provider=judge_config.get("provider", "qwen"),
        model=judge_config.get("model", ""),
        api_key=judge_config.get("api_key"),
        temperature=judge_config.get("temperature", 0.0),
        max_retries=judge_config.get("max_retries", 2),
        timeout=judge_config.get("timeout", 30)
    )
    
    results = []
    for idx, ex in enumerate(examples):
        result_row = {
            "query_id": idx,
            "question": ex["question"],
            "n_contexts": len(ex["contexts"]),
            "latency_ms": ex.get("latency_ms", 0.0)
        }
        
        # Метрики из Ragas
        if idx < len(ragas_metrics_df):
            result_row["relevance"] = ragas_metrics_df.iloc[idx].get("answer_relevancy", 0.0)
            result_row["faithfulness"] = ragas_metrics_df.iloc[idx].get("faithfulness", 0.0)
            result_row["context_precision"] = ragas_metrics_df.iloc[idx].get("context_precision", 0.0)
            result_row["context_recall"] = ragas_metrics_df.iloc[idx].get("context_recall", 0.0)
        else:
            result_row["relevance"] = 0.0
            result_row["faithfulness"] = 0.0
            result_row["context_precision"] = 0.0
            result_row["context_recall"] = 0.0
        
        # Оцениваем кастомные метрики (completeness, off_topic_rate) за один запрос
        cited_sources = ex.get("cited_sources", [])
        
        # Если Ragas уже вычислил faithfulness, используем его
        # Иначе делаем полный запрос для всех метрик
        if result_row.get("faithfulness", 0.0) > 0.0:
            # Ragas уже вычислил faithfulness, делаем запрос только для completeness и off_topic_rate
            # Но для упрощения все равно делаем один запрос для всех кастомных метрик
            # (можно оптимизировать дальше, но это усложнит код)
            _, _, completeness, off_topic_rate, has_error = score_all_metrics(
                judge_client, ex["question"], ex["answer"], ex["contexts"], cited_sources
            )
        else:
            # Ragas не вычислил, используем все метрики из кастомной оценки
            relevance_custom, faithfulness_custom, completeness, off_topic_rate, has_error = score_all_metrics(
                judge_client, ex["question"], ex["answer"], ex["contexts"], cited_sources
            )
            result_row["relevance"] = relevance_custom
            result_row["faithfulness"] = faithfulness_custom
        
        result_row["completeness"] = completeness
        result_row["off_topic_rate"] = off_topic_rate
        result_row["judge_errors"] = has_error
        
        results.append(result_row)
        
        # Выводим промежуточные метрики каждые 10 итераций
        if (idx + 1) % 10 == 0:
            _print_intermediate_metrics(results, idx, len(examples))
    
    return pd.DataFrame(results)


def _run_custom_evaluation(
    examples: List[Dict[str, Any]],
    judge_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Запускает оценку через кастомные метрики (fallback).
    """
    logger.info(f"Запуск оценки через кастомные метрики для {len(examples)} примеров")
    
    judge_client = JudgeClient(
        provider=judge_config.get("provider", "qwen"),
        model=judge_config.get("model", ""),
        api_key=judge_config.get("api_key"),
        temperature=judge_config.get("temperature", 0.0),
        max_retries=judge_config.get("max_retries", 2),
        timeout=judge_config.get("timeout", 30)
    )
    
    results = []
    
    for idx, ex in enumerate(examples):
        result_row = {
            "query_id": idx,
            "question": ex["question"],
            "n_contexts": len(ex["contexts"]),
            "latency_ms": ex.get("latency_ms", 0.0)
        }
        
        # Оцениваем все метрики за один запрос к judge LLM
        cited_sources = ex.get("cited_sources", [])
        relevance_score, faithfulness_score, completeness_score, off_topic_rate, has_error = score_all_metrics(
            judge_client, ex["question"], ex["answer"], ex["contexts"], cited_sources
        )
        
        result_row["relevance"] = relevance_score
        result_row["faithfulness"] = faithfulness_score
        result_row["completeness"] = completeness_score
        result_row["off_topic_rate"] = off_topic_rate
        
        # Context precision и recall (упрощенные версии)
        result_row["context_precision"] = faithfulness_score  # Используем faithfulness как proxy
        result_row["context_recall"] = min(1.0, len(ex["contexts"]) / 5.0)  # Упрощенная метрика
        
        result_row["judge_errors"] = has_error
        
        results.append(result_row)
        
        # Выводим промежуточные метрики каждые 10 итераций
        if (idx + 1) % 10 == 0:
            _print_intermediate_metrics(results, idx, len(examples))
    
    return pd.DataFrame(results)

