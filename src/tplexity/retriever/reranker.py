import logging

from transformers import AutoModel

from tplexity.retriever.utils import get_device

logger = logging.getLogger(__name__)


class Reranker:
    """Класс для reranking результатов поиска с использованием jina-reranker-v3

    Модель поддерживает:
    - Listwise reranking до 64 документов одновременно
    - Мультиязычный reranking
    - Контекстное окно до 131K токенов
    """

    def __init__(self, model_name: str = "jinaai/jina-reranker-v3"):
        """
        Инициализация reranker

        Args:
            model_name (str): Имя модели для reranking. По умолчанию используется jinaai/jina-reranker-v3
        """
        self.model_name = model_name
        self.device = get_device()
        logger.info(f"🔄 [retriever][reranker] Загрузка модели reranker: {model_name} на устройстве: {self.device}")

        try:
            self.model = (
                AutoModel.from_pretrained(
                    model_name,
                    dtype="auto",
                    trust_remote_code=True,
                )
                .eval()
                .to(self.device)
            )
        except Exception as e:
            logger.error(f"❌ [retriever][reranker] Ошибка при загрузке модели reranker: {e}")
            raise

    def rerank(self, query: str, documents: list[str], top_n: int = 10) -> list[tuple[int, float]]:
        """
        Переранжировать документы относительно запроса

        Args:
            query (str): Поисковый запрос
            documents (list[str]): Список документов для reranking
            top_n (int): Количество возвращаемых результатов

        Returns:
            list[tuple[int, float]]: Список кортежей (индекс документа, relevance_score), отсортированный по убыванию score
        """
        if not documents:
            return []

        if not query:
            logger.warning("⚠️ [retriever][reranker] Пустой запрос для reranking")
            return []

        if self.model is None:
            logger.error("❌ [retriever][reranker] Модель не инициализирована")
            return [(idx, 0.0) for idx in range(min(len(documents), top_n))]

        try:
            results = self.model.rerank(query, documents, top_n=top_n)

            reranked = [(result["index"], float(result["relevance_score"])) for result in results]
            return reranked

        except Exception as e:
            logger.error(f"❌ [retriever][reranker] Ошибка при reranking: {e}")

            return [(idx, 0.0) for idx in range(min(len(documents), top_n))]


_reranker_instance: Reranker | None = None


def get_reranker() -> Reranker:
    """
    Получить экземпляр модели для reranking (singleton)

    Returns:
        Reranker: Экземпляр Reranker модели jinaai/jina-reranker-v3
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance
