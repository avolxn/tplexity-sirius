import logging
from datetime import datetime, timedelta

from tplexity.parser.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Глобальный экземпляр клиента (singleton)
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Получить глобальный экземпляр LLM клиента"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


RELEVANCE_EXAMPLES = {
    "новости": 7,
    "макроданные": 30,
    "отчет о доходах": 90,
    "анализ": 60,
    "прогноз": 180,
    "мнение": 14,
}

RELEVANCE_PROMPT = """Ты - эксперт по анализу финансовых новостей и постов. Определи, как долго будет актуальна информация в посте.

КРИТЕРИИ АНАЛИЗА:

1. ТИП КОНТЕНТА:
   - Новость о событии (релиз данных, объявление): 3-7 дней
   - Макроэкономические данные (ВВП, инфляция, ставки): 30-60 дней
   - Финансовые отчеты (квартальные/годовые): 90-180 дней
   - Аналитические обзоры и прогнозы: 60-180 дней
   - Мнения экспертов: 7-14 дней
   - Образовательный контент: 180-365 дней
   - Рыночные прогнозы: 30-90 дней

2. ВРЕМЕННАЯ ПРИРОДА:
   - Срочные новости (уже произошли): 3-7 дней
   - Периодические данные (ежемесячные/квартальные): период до следующего релиза + 7 дней
   - Прогнозы: срок прогноза + 30 дней
   - Исторические данные: 180-365 дней

3. ПРАВИЛА:
   - Конкретные цифры с датой → до следующего релиза аналогичных данных
   - Новость о событии → 3-7 дней
   - Анализ/прогноз → минимум 30 дней, зависит от горизонта
   - Образовательный контент → 180-365 дней
   - Мнение без данных → 7-14 дней

ПРИМЕРЫ:

Пост: "Сегодня ЦБ РФ объявил о повышении ключевой ставки до 16%"
Актуальность: 30
(Макроэкономические данные, актуальны до следующего решения ЦБ)

Пост: "Сбербанк опубликовал отчет за Q3 2024: выручка выросла на 15%"
Актуальность: 90
(Квартальный отчет, актуален до следующего квартального отчета)

Пост: "Эксперт считает, что рынок акций может вырасти в следующем месяце"
Актуальность: 14
(Мнение эксперта, краткосрочный прогноз)

Пост: "Что такое дивиденды и как их получать: подробное руководство"
Актуальность: 365
(Образовательный контент, долгосрочная актуальность)

Пост: "Вчера Apple объявила о запуске нового iPhone"
Актуальность: 7
(Новость о событии, краткосрочная актуальность)

---

ТВОЯ ЗАДАЧА:
Проанализируй пост ниже и верни ТОЛЬКО число (от 1 до 10000) - количество дней актуальности.
Без пояснений, без текста, только число.

Пост:
{post_text}

Количество дней актуальности:"""


async def determine_relevance_days(post_text: str, llm_provider: str = "qwen") -> tuple[int, str]:
    """
    Определяет количество дней актуальности поста через LLM

    Args:
        post_text: Текст поста для анализа
        llm_provider: Провайдер LLM (по умолчанию "qwen")

    Returns:
        tuple[int, str]: (Количество дней актуальности (от 1 до 10000), сырой ответ LLM)
    """
    try:
        llm_client = get_llm_client()

        messages = [
            {
                "role": "user",
                "content": RELEVANCE_PROMPT.format(post_text=post_text),
            }
        ]

        raw_response = await llm_client.generate(
            provider=llm_provider,
            messages=messages,
            temperature=0.0,
            max_tokens=50,
        )

        response = raw_response.strip()

        digits = ""
        for char in response:
            if char.isdigit():
                digits += char
            elif digits:
                break

        if not digits:
            logger.warning(
                f"⚠️ [relevance_analyzer] Не удалось извлечь число из ответа LLM: {response}, используем значение по умолчанию 30"
            )
            return 30, raw_response

        relevance_days = int(digits)

        relevance_days = max(1, min(10000, relevance_days))

        logger.info(
            f"✅ [relevance_analyzer] Определена актуальность: {relevance_days} дней для поста (длина: {len(post_text)} символов)"
        )
        return relevance_days, raw_response

    except Exception as e:
        logger.error(f"❌ [relevance_analyzer] Ошибка при определении актуальности: {e}", exc_info=True)

        return 30, f"ERROR: {str(e)}"


def calculate_delete_date(relevance_days: int, post_date: datetime | None = None) -> str:
    """
    Вычисляет дату удаления поста от даты публикации (дата публикации + relevance_days + 3 дня)

    Args:
        relevance_days: Количество дней актуальности
        post_date: Дата публикации поста. Если None, используется текущая дата (для обратной совместимости)

    Returns:
        str: Дата удаления в формате ISO (YYYY-MM-DD)
    """
    if post_date is None:
        post_date = datetime.now()

    delete_date = post_date + timedelta(days=relevance_days + 3)
    return delete_date.strftime("%Y-%m-%d")
