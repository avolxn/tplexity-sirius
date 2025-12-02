import logging
import re

from fastembed import SparseTextEmbedding
from pymorphy3 import MorphAnalyzer
from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)


class BM25:
    """Класс для работы с BM25 поиском с поддержкой лемматизации"""

    def __init__(self, model_name: str = "Qdrant/bm25"):
        """Инициализация BM25 модели с лемматизацией

        Args:
            model_name (str): Имя модели для sparse embeddings. По умолчанию "Qdrant/bm25"
        """
        self.model_name = model_name

        logger.info(f"🔄 [retriever][sparse_embedding] Инициализация BM25 модели: {model_name}")
        try:
            self.sparse_model = SparseTextEmbedding(model_name=model_name)
            logger.info(f"✅ [retriever][sparse_embedding] Sparse модель (BM25) инициализирована: {model_name}")
        except Exception as e:
            logger.error(f"❌ [retriever][sparse_embedding] Ошибка инициализации sparse модели: {e}")
            raise

        try:
            self.morph = MorphAnalyzer()
            logger.info("✅ [retriever][sparse_embedding] Лемматизатор (pymorphy3) инициализирован для BM25")
        except Exception as e:
            logger.warning(f"⚠️ [retriever][sparse_embedding] Не удалось инициализировать лемматизатор: {e}")
            self.morph = None

    def lemmatize_text(self, text: str) -> str:
        """
        Лемматизация текста для улучшения качества BM25 поиска

        Args:
            text (str): Исходный текст

        Returns:
            str: Текст с лемматизированными словами
        """
        if self.morph is None:
            return text

        words = re.findall(r"[а-яёА-ЯЁa-zA-Z]+", text.lower())
        lemmatized_words = []

        for word in words:
            if not word:
                continue
            try:
                parsed = self.morph.parse(word)[0]
                lemma = parsed.normal_form
                lemmatized_words.append(lemma)
            except Exception:
                lemmatized_words.append(word)

        return " ".join(lemmatized_words)

    def encode_documents(self, documents: list[str]) -> list[SparseVector]:
        """
        Создать sparse embeddings для документов с лемматизацией

        Args:
            documents (list[str]): Список документов для индексации

        Returns:
            list[SparseEmbedding]: Список sparse embeddings
        """
        lemmatized_documents = [self.lemmatize_text(doc) for doc in documents]
        sparse_embeddings = list(self.sparse_model.passage_embed(lemmatized_documents))
        return sparse_embeddings

    def encode_query(self, query: str) -> SparseVector:
        """
        Создать sparse embedding для запроса с лемматизацией

        Args:
            query (str): Поисковый запрос

        Returns:
            SparseVector: SparseVector для запроса
        """
        lemmatized_query = self.lemmatize_text(query)
        sparse_query_dict = list(self.sparse_model.query_embed(lemmatized_query))[0].as_object()
        return SparseVector(**sparse_query_dict)


_bm25_instance: BM25 | None = None


def get_bm25_model() -> BM25:
    """
    Получить экземпляр модели для BM25 (singleton)

    Returns:
        BM25: Экземпляр BM25 модели
    """
    global _bm25_instance
    if _bm25_instance is None:
        _bm25_instance = BM25()
    return _bm25_instance
