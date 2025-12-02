import logging
from typing import Literal

from sentence_transformers import SentenceTransformer

from tplexity.retriever.utils import get_device

logger = logging.getLogger(__name__)


PromptNameType = Literal[
    "search_query",
    "search_document",
    "paraphrase",
    "categorize",
    "categorize_sentiment",
    "categorize_topic",
    "categorize_entailment",
]


class Embedding:
    """
    Класс для работы с embeddings модели ai-forever/FRIDA

    Модель поддерживает:
    - prompt_name для различных задач:
        - "search_query": для запросов в асимметричном поиске
        - "search_document": для документов в асимметричном поиске
        - "paraphrase": для симметричного поиска (STS, парафразы)
        - "categorize": для асимметричного сопоставления заголовка и тела документа
        - "categorize_sentiment": для задач, связанных с сентиментом
        - "categorize_topic": для группировки текстов по темам
        - "categorize_entailment": для задач текстового следования (NLI)
    - CLS pooling (используется по умолчанию)
    - Максимальная длина последовательности: 512 токенов
    """

    def __init__(self, model_name: str = "ai-forever/FRIDA"):
        """
        Инициализация класса Embedding

        Args:
            model_name (str): Имя модели для загрузки
        """
        self.model_name = model_name
        device = get_device()
        logger.info(f"🔄 [retriever][dense_embedding] Инициализация модели: {model_name} на устройстве: {device}")
        try:
            self.model = SentenceTransformer(model_name, device=str(device))
            logger.info(f"✅ [retriever][dense_embedding] Модель {model_name} успешно инициализирована на {device}")
        except Exception as e:
            logger.error(f"❌ [retriever][dense_embedding] Ошибка инициализации модели: {e}")
            raise

    def encode(
        self,
        texts: list[str] | str,
        prompt_name: PromptNameType = "search_query",
    ) -> list[list[float]] | list[float]:
        """
        Кодировать тексты в embeddings

        Args:
            texts (list[str] | str): Текст или список текстов для кодирования
            prompt_name (PromptNameType): Имя промпта для задачи:
                - "search_query": для запросов в асимметричном поиске
                - "search_document": для документов в асимметричном поиске
                - "paraphrase": для симметричного поиска (STS, парафразы)
                - "categorize": для асимметричного сопоставления заголовка и тела документа
                - "categorize_sentiment": для задач, связанных с сентиментом
                - "categorize_topic": для группировки текстов по темам
                - "categorize_entailment": для задач текстового следования (NLI)

        Returns:
            list[list[float]] | list[float]: Список embeddings (или один embedding, если передан один текст)
        """

        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False

        logger.debug(f"🔄 [retriever][dense_embedding] Кодирование {len(texts)} текстов, prompt_name: {prompt_name}")
        embeddings = self.model.encode(texts, prompt_name=prompt_name, normalize_embeddings=True)

        if single_text:
            return embeddings[0].tolist() if hasattr(embeddings[0], "tolist") else embeddings[0]

        return [emb.tolist() if hasattr(emb, "tolist") else emb for emb in embeddings]

    def encode_query(self, query: str) -> list[float]:
        """
        Кодировать запрос в embedding

        Args:
            query (str): Текст запроса

        Returns:
            list[float]: Embedding запроса как список float
        """
        logger.debug(f"🔄 [retriever][dense_embedding] Кодирование запроса: {query[:50]}...")
        return self.encode(query, prompt_name="search_query")

    def encode_document(self, documents: list[str]) -> list[list[float]]:
        """
        Кодировать документы в embeddings

        Args:
            documents (list[str]): Список документов для кодирования

        Returns:
            list[list[float]]: Список embeddings документов
        """
        logger.debug(f"🔄 [retriever][dense_embedding] Кодирование {len(documents)} документов")
        return self.encode(documents, prompt_name="search_document")

    def get_sentence_embedding_dimension(self) -> int | None:
        """
        Получить размерность embeddings

        Returns:
            int | None: Размерность embeddings или None, если не удалось определить
        """
        embedding_dim = self.model.get_sentence_embedding_dimension()

        if embedding_dim is None:
            logger.warning(
                "⚠️ [retriever][dense_embedding] Не удалось определить размерность через get_sentence_embedding_dimension(), определяем эмпирически"
            )
            test_embedding = self.encode("test")
            embedding_dim = len(test_embedding)
            logger.info(f"✅ [retriever][dense_embedding] Размерность определена эмпирически: {embedding_dim}")

        return embedding_dim

    def get_model(self) -> SentenceTransformer:
        """
        Получить экземпляр модели SentenceTransformer

        Returns:
            SentenceTransformer: Экземпляр модели SentenceTransformer
        """
        return self.model


_embedding_instance: Embedding | None = None


def get_embedding_model() -> Embedding:
    """
    Получить экземпляр модели для embeddings (singleton)

    Returns:
        Embedding: Экземпляр Embedding модели ai-forever/FRIDA
    """
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = Embedding()
    return _embedding_instance
