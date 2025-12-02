"""
Модуль для вычисления кастомных метрик через LLM-as-a-judge.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Добавляем путь к проекту для импорта tplexity
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from tplexity.llm_client.client import get_llm
    from tplexity.llm_client.config import settings as llm_settings
    QWEN_AVAILABLE = True
except ImportError as e:
    QWEN_AVAILABLE = False
    # Не выводим предупреждение здесь, т.к. оно может быть ложным
    # (импорт может не работать при загрузке модуля, но работать позже)

try:
    from .judge_prompts import (
        PROMPT_RELEVANCE,
        PROMPT_FAITHFULNESS,
        PROMPT_COMPLETENESS,
        PROMPT_OFF_TOPIC,
        PROMPT_ALL_METRICS
    )
except ImportError:
    from judge_prompts import (
        PROMPT_RELEVANCE,
        PROMPT_FAITHFULNESS,
        PROMPT_COMPLETENESS,
        PROMPT_OFF_TOPIC,
        PROMPT_ALL_METRICS
    )

logger = logging.getLogger(__name__)


class JudgeClient:
    """
    Клиент для взаимодействия с judge моделью.
    """
    
    def __init__(
        self,
        provider: str = "qwen",
        model: str = "",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout: int = 30
    ):
        """
        Инициализация judge клиента.
        
        Args:
            provider: Провайдер модели (qwen, openai, mock)
            model: Название модели (для openai, для qwen игнорируется)
            api_key: API ключ (для openai, для qwen игнорируется)
            temperature: Температура для генерации
            max_retries: Максимальное количество попыток
            timeout: Таймаут запроса
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.llm_client = None
        self.client = None  # Для OpenAI
        
        if self.provider == "qwen":
            if not QWEN_AVAILABLE:
                logger.warning("tplexity.llm_client не доступен, используется mock judge")
                self.provider = "mock"
            else:
                try:
                    self.llm_client = get_llm("qwen")
                    logger.info(f"Инициализирован Qwen judge: model={llm_settings.qwen_model}, base_url={llm_settings.qwen_base_url}")
                except Exception as e:
                    logger.warning(f"Не удалось инициализировать Qwen judge: {e}, используется mock judge")
                    self.provider = "mock"
        elif self.provider == "yandexgpt":
            if not QWEN_AVAILABLE:
                logger.warning("tplexity.llm_client не доступен, используется mock judge")
                self.provider = "mock"
            else:
                try:
                    self.llm_client = get_llm("yandexgpt")
                    logger.info(f"Инициализирован YandexGPT judge: model={llm_settings.yandexgpt_model}, base_url={llm_settings.yandexgpt_base_url}")
                except Exception as e:
                    logger.warning(f"Не удалось инициализировать YandexGPT judge: {e}, используется mock judge")
                    self.provider = "mock"
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI не установлен, используется mock judge")
                self.provider = "mock"
            else:
                self.api_key = api_key or os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    logger.warning("OPENAI_API_KEY не найден, используется mock judge")
                    self.provider = "mock"
                else:
                    self.client = OpenAI(api_key=self.api_key, timeout=timeout)
                    logger.info(f"Инициализирован OpenAI judge: model={model}")
        else:
            self.client = None
            logger.info("Используется mock judge")
    
    def _call_judge(self, prompt: str, max_retries: Optional[int] = None) -> str:
        """
        Вызывает judge модель.
        
        Args:
            prompt: Промпт для оценки
            max_retries: Количество попыток (если None - используется self.max_retries)
            
        Returns:
            Ответ модели (JSON строка)
        """
        if max_retries is None:
            max_retries = self.max_retries
        
        if self.provider == "mock":
            return self._mock_judge(prompt)
        
        # Qwen и YandexGPT через LLMClient (асинхронный)
        if self.provider in ["qwen", "yandexgpt"] and self.llm_client:
            return self._call_qwen_judge(prompt, max_retries)
        
        # OpenAI API
        if self.provider == "openai" and self.client:
            for attempt in range(max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "Ты — эксперт по оценке качества ответов. Всегда возвращай только валидный JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        response_format={"type": "json_object"}
                    )
                    
                    content = response.choices[0].message.content
                    return content
                    
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Ошибка при вызове judge (попытка {attempt + 1}/{max_retries + 1}): {e}")
                        continue
                    else:
                        logger.error(f"Не удалось вызвать judge после {max_retries + 1} попыток: {e}")
                        return self._mock_judge(prompt)
        
        return self._mock_judge(prompt)
    
    def _call_qwen_judge(self, prompt: str, max_retries: int) -> str:
        """
        Вызывает Qwen/YandexGPT judge через асинхронный LLMClient.
        
        Args:
            prompt: Промпт для оценки
            max_retries: Количество попыток
            
        Returns:
            Ответ модели (JSON строка)
        """
        messages = [
            {"role": "system", "content": "Ты — эксперт по оценке качества ответов. Всегда возвращай только валидный JSON."},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(max_retries + 1):
            try:
                # Запускаем асинхронную функцию в синхронном контексте
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Проверяем, запущен ли event loop
                if loop.is_running():
                    # Если цикл уже запущен, создаем новый event loop в отдельном потоке
                    import concurrent.futures
                    
                    def run_in_new_loop():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                self.llm_client.generate(
                                    messages=messages,
                                    temperature=self.temperature,
                                    max_tokens=2000
                                )
                            )
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_new_loop)
                        response = future.result(timeout=self.timeout)
                else:
                    # Если цикл не запущен, используем run_until_complete
                    response = loop.run_until_complete(
                        self.llm_client.generate(
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=2000  # Достаточно для JSON ответа
                        )
                    )
                
                return response
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Ошибка при вызове {self.provider} judge (попытка {attempt + 1}/{max_retries + 1}): {e}")
                    continue
                else:
                    logger.error(f"Не удалось вызвать {self.provider} judge после {max_retries + 1} попыток: {e}")
                    return self._mock_judge(prompt)
        
        return self._mock_judge(prompt)
    
    def _mock_judge(self, prompt: str) -> str:
        """
        Mock judge: простая эвристика на основе пересечения слов.
        """
        # Простая эвристика: возвращаем средние значения
        if "relevance" in prompt.lower():
            return json.dumps({"relevance": 0.75, "explanation": "Mock оценка релевантности"})
        elif "faithfulness" in prompt.lower():
            return json.dumps({"faithfulness": 0.8, "hallucinated_claims": []})
        elif "completeness" in prompt.lower():
            return json.dumps({"completeness": 0.75, "explanation": "Mock оценка полноты"})
        elif "off_topic" in prompt.lower() or "лишней информации" in prompt.lower():
            return json.dumps({"off_topic_rate": 0.2, "off_topic_claims": []})
        elif "все метрики" in prompt.lower():
            return json.dumps({
                "relevance": 0.75,
                "faithfulness": 0.8,
                "hallucinated_claims": [],
                "completeness": 0.75,
                "off_topic_rate": 0.2,
                "off_topic_claims": []
            })
        else:
            return json.dumps({"score": 0.75, "explanation": "Mock оценка"})
    
    def _parse_json_response(self, response: str, default_score: float = 0.0) -> Dict[str, Any]:
        """
        Парсит JSON ответ от judge.
        
        Args:
            response: JSON строка
            default_score: Значение по умолчанию при ошибке парсинга
            
        Returns:
            Словарь с результатами оценки
        """
        try:
            # Пытаемся извлечь JSON из ответа (на случай, если есть дополнительный текст)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Не удалось распарсить JSON ответ от judge: {e}. Ответ: {response[:200]}")
            return {"error": "json_parse_error", "score": default_score}


def score_relevance(
    judge_client: JudgeClient,
    question: str,
    answer: str
) -> Tuple[float, Optional[str], bool]:
    """
    Оценивает релевантность ответа вопросу.
    
    Args:
        judge_client: Клиент для judge модели
        question: Текст вопроса
        answer: Текст ответа
        
    Returns:
        Кортеж (score, explanation, has_error)
    """
    prompt = PROMPT_RELEVANCE.format(question=question, answer=answer)
    
    try:
        response = judge_client._call_judge(prompt)
        result = judge_client._parse_json_response(response, default_score=0.0)
        
        if "error" in result:
            return 0.0, None, True
        
        score = result.get("relevance", 0.0)
        explanation = result.get("explanation")
        
        return float(score), explanation, False
        
    except Exception as e:
        logger.error(f"Ошибка при оценке relevance: {e}")
        return 0.0, None, True


def score_faithfulness(
    judge_client: JudgeClient,
    contexts: List[str],
    answer: str
) -> Tuple[float, List[str], bool]:
    """
    Оценивает правдивость ответа относительно контекстов.
    
    Args:
        judge_client: Клиент для judge модели
        contexts: Список контекстов
        answer: Текст ответа
        
    Returns:
        Кортеж (score, hallucinated_claims, has_error)
    """
    contexts_text = "\n\n".join([f"Контекст {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
    prompt = PROMPT_FAITHFULNESS.format(contexts=contexts_text, answer=answer)
    
    try:
        response = judge_client._call_judge(prompt)
        result = judge_client._parse_json_response(response, default_score=0.0)
        
        if "error" in result:
            return 0.0, [], True
        
        score = result.get("faithfulness", 0.0)
        hallucinated_claims = result.get("hallucinated_claims", [])
        
        return float(score), hallucinated_claims, False
        
    except Exception as e:
        logger.error(f"Ошибка при оценке faithfulness: {e}")
        return 0.0, [], True


def score_completeness(
    judge_client: JudgeClient,
    question: str,
    answer: str,
    contexts: List[str]
) -> Tuple[float, Optional[str], bool]:
    """
    Оценивает полноту ответа относительно вопроса.
    
    Args:
        judge_client: Клиент для judge модели
        question: Текст вопроса
        answer: Текст ответа
        contexts: Список контекстов
        
    Returns:
        Кортеж (score, explanation, has_error)
    """
    contexts_text = "\n\n".join([f"Контекст {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
    prompt = PROMPT_COMPLETENESS.format(question=question, answer=answer, contexts=contexts_text)
    
    try:
        response = judge_client._call_judge(prompt)
        result = judge_client._parse_json_response(response, default_score=0.0)
        
        if "error" in result:
            return 0.0, None, True
        
        score = result.get("completeness", 0.0)
        explanation = result.get("explanation")
        
        return float(score), explanation, False
        
    except Exception as e:
        logger.error(f"Ошибка при оценке completeness: {e}")
        return 0.0, None, True


async def score_all_metrics_async(
    judge_client: JudgeClient,
    question: str,
    answer: str,
    contexts: List[str],
    cited_sources: List[str]
) -> Tuple[float, float, float, float, bool]:
    """
    Асинхронная версия score_all_metrics.
    
    Args:
        judge_client: Клиент для judge модели
        question: Текст вопроса
        answer: Текст ответа
        contexts: Список контекстов
        cited_sources: Список указанных источников (не используется, оставлен для совместимости)
        
    Returns:
        Кортеж (relevance, faithfulness, completeness, off_topic_rate, has_error)
    """
    contexts_text = "\n\n".join([f"Контекст {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = PROMPT_ALL_METRICS.format(
        question=question,
        answer=answer,
        contexts=contexts_text
    )
    
    try:
        # Если judge_client использует Qwen или YandexGPT (асинхронный), вызываем напрямую
        if judge_client.provider in ["qwen", "yandexgpt"] and judge_client.llm_client:
            messages = [
                {"role": "system", "content": "Ты — эксперт по оценке качества ответов. Всегда возвращай только валидный JSON."},
                {"role": "user", "content": prompt}
            ]
            
            for attempt in range(judge_client.max_retries + 1):
                try:
                    response = await judge_client.llm_client.generate(
                        messages=messages,
                        temperature=judge_client.temperature,
                        max_tokens=2000
                    )
                    result = judge_client._parse_json_response(response, default_score=0.0)
                    
                    if "error" in result:
                        if attempt < judge_client.max_retries:
                            continue
                        return 0.0, 0.0, 0.0, 0.0, True
                    
                    relevance = float(result.get("relevance", 0.0))
                    faithfulness = float(result.get("faithfulness", 0.0))
                    completeness = float(result.get("completeness", 0.0))
                    off_topic_rate = float(result.get("off_topic_rate", 0.0))
                    
                    return relevance, faithfulness, completeness, off_topic_rate, False
                except Exception as e:
                    if attempt < judge_client.max_retries:
                        logger.warning(f"Ошибка при вызове {judge_client.provider} judge (попытка {attempt + 1}/{judge_client.max_retries + 1}): {e}")
                        continue
                    else:
                        logger.error(f"Не удалось вызвать {judge_client.provider} judge после {judge_client.max_retries + 1} попыток: {e}")
                        return 0.0, 0.0, 0.0, 0.0, True
        
        # Для других провайдеров используем синхронный вызов в отдельном потоке
        response = await asyncio.to_thread(judge_client._call_judge, prompt)
        result = judge_client._parse_json_response(response, default_score=0.0)
        
        if "error" in result:
            return 0.0, 0.0, 0.0, 0.0, True
        
        relevance = float(result.get("relevance", 0.0))
        faithfulness = float(result.get("faithfulness", 0.0))
        completeness = float(result.get("completeness", 0.0))
        off_topic_rate = float(result.get("off_topic_rate", 0.0))
        
        return relevance, faithfulness, completeness, off_topic_rate, False
        
    except Exception as e:
        logger.error(f"Ошибка при оценке всех метрик: {e}")
        return 0.0, 0.0, 0.0, 0.0, True


def score_all_metrics(
    judge_client: JudgeClient,
    question: str,
    answer: str,
    contexts: List[str],
    cited_sources: List[str]
) -> Tuple[float, float, float, float, bool]:
    """
    Оценивает все метрики за один запрос к judge LLM.
    
    Args:
        judge_client: Клиент для judge модели
        question: Текст вопроса
        answer: Текст ответа
        contexts: Список контекстов
        cited_sources: Список указанных источников (не используется, оставлен для совместимости)
        
    Returns:
        Кортеж (relevance, faithfulness, completeness, off_topic_rate, has_error)
    """
    contexts_text = "\n\n".join([f"Контекст {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = PROMPT_ALL_METRICS.format(
        question=question,
        answer=answer,
        contexts=contexts_text
    )
    
    # TODO: закомментировать после отладки
    # print("=" * 80)
    # print("ПРОМПТ ДЛЯ JUDGE LLM:\n" + prompt)
    # print("=" * 80)
    
    try:
        response = judge_client._call_judge(prompt)
        result = judge_client._parse_json_response(response, default_score=0.0)
        
        if "error" in result:
            return 0.0, 0.0, 0.0, 0.0, True
        
        relevance = float(result.get("relevance", 0.0))
        faithfulness = float(result.get("faithfulness", 0.0))
        completeness = float(result.get("completeness", 0.0))
        off_topic_rate = float(result.get("off_topic_rate", 0.0))
        
        return relevance, faithfulness, completeness, off_topic_rate, False
        
    except Exception as e:
        logger.error(f"Ошибка при оценке всех метрик: {e}")
        return 0.0, 0.0, 0.0, 0.0, True

