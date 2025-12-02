import asyncio
import logging
import time
import traceback

from tplexity.retriever.config import settings
from tplexity.retriever.reranker import get_reranker
from tplexity.retriever.vector_search import VectorSearch

logger = logging.getLogger(__name__)


class RetrieverService:
    """Класс для гибридного поиска с использованием Qdrant

    1. Prefetch
    - Sparse Embeddings: BM25 с лемматизацией
    - Dense Embeddings: ai-forever/FRIDA
    2. RRF для объединения векторов
    3. Reranking: Jina Reranker v3
    """

    def __init__(
        self,
        collection_name: str | None = None,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
    ):
        """
        Инициализация гибридного поисковика

        Args:
            collection_name (str | None): Имя коллекции в Qdrant
            host (str | None): Хост Qdrant
            port (int | None): Порт Qdrant
            api_key (str | None): API ключ для Qdrant
        """
        logger.info("🔄 [retriever][retriever_service] Инициализация гибридного поисковика")

        self._init_config_params(
            collection_name=collection_name,
            host=host,
            port=port,
            api_key=api_key,
        )

        self.vector_search = VectorSearch(
            collection_name=self.collection_name,
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            prefetch_ratio=self.prefetch_ratio,
        )

        self.enable_reranker = settings.enable_reranker
        if self.enable_reranker:
            try:
                self.reranker = get_reranker()
                logger.info("✅ [retriever][retriever_service] Reranker инициализирован")
            except Exception as e:
                logger.warning(
                    f"⚠️ [retriever][retriever_service] Не удалось инициализировать reranker: {e}. "
                    f"Reranker будет отключен."
                )
                self.enable_reranker = False
                self.reranker = None
        else:
            self.reranker = None
            logger.info("ℹ️ [retriever][retriever_service] Reranker отключен в настройках")

        logger.info(
            f"✅ [retriever][retriever_service] Гибридный поисковик инициализирован: "
            f"top_k={self.top_k}, top_n={self.top_n}, prefetch_ratio={self.prefetch_ratio}"
        )

    def _init_config_params(
        self,
        collection_name: str | None = None,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Инициализация всех параметров из config в одном месте.
        Все параметры читаются здесь и сохраняются в атрибуты класса.

        Args:
            collection_name: Имя коллекции (если None, берется из config)
            host: Хост Qdrant (если None, берется из config)
            port: Порт Qdrant (если None, берется из config)
            api_key: API ключ (если None, берется из config)
        """

        self.collection_name = collection_name or settings.qdrant_collection_name
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.api_key = api_key or settings.qdrant_api_key

        self.top_k = settings.top_k
        self.top_n = settings.top_n
        self.prefetch_ratio = settings.prefetch_ratio

    async def add_documents(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """
        Добавить новые документы в векторную базу данных

        Args:
            documents (list[str]): Список новых документов
            metadatas (list[dict] | None): Список словарей с метаданными для каждого документа

        Raises:
            ValueError: Если документы пусты или невалидны
        """
        if not documents:
            raise ValueError("Список документов не может быть пустым")

        if any(not doc or not doc.strip() for doc in documents):
            raise ValueError("Документы не могут быть пустыми или содержать только пробелы")

        try:
            await self.vector_search.add_documents(documents, ids=None, metadatas=metadatas)
            logger.info(f"✅ [retriever][retriever_service] Добавлено {len(documents)} документов в Qdrant")
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                f"❌ [retriever][retriever_service] Ошибка при добавлении документов в Qdrant: {e}\n{error_traceback}",
                exc_info=True,
            )
            raise

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
        use_rerank: bool | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> list[tuple[str, float, str, dict | None]]:
        """
        Гибридный поиск: BM25 + Embeddings → RRF (в Qdrant) → Rerank

        Args:
            query (str): Поисковый запрос (уже переформулированный, если требуется)
            top_k (int | None): Количество документов до реранка. Если None, используется значение из config
            top_n (int | None): Количество документов после реранка (возвращаемые). Если None, используется значение из config
            use_rerank (bool | None): Использовать ли reranking. Если None, используется значение из config
            messages (list[dict[str, str]] | None): Не используется, оставлен для обратной совместимости

        Returns:
            list[tuple[str, float, str, dict | None]]: Список кортежей (doc_id, score, document_text, metadata)

        Raises:
            ValueError: Если запрос пуст или параметры невалидны
        """
        if not query or not query.strip():
            raise ValueError("Поисковый запрос не может быть пустым")

        top_k = top_k or self.top_k
        top_n = top_n or self.top_n
        use_rerank = use_rerank if use_rerank is not None else self.enable_reranker

        if top_k < 1:
            raise ValueError(f"top_k должен быть >= 1, получено: {top_k}")
        if top_n < 1:
            raise ValueError(f"top_n должен быть >= 1, получено: {top_n}")

        logger.info(f"🔍 [retriever][retriever_service] Поиск: '{query[:50]}...' (top_k={top_k}, top_n={top_n})")
        search_start_time = time.time()

        hybrid_start_time = time.time()
        hybrid_results = await self.vector_search.search(query, top_k=top_k, search_type="hybrid")
        hybrid_time = time.time() - hybrid_start_time
        logger.info(
            f"✅ [retriever][retriever_service] Гибридный поиск завершен: найдено {len(hybrid_results)} результатов за {hybrid_time:.2f}с"
        )

        if not hybrid_results:
            logger.warning("⚠️ [retriever][retriever_service] Гибридный поиск не вернул результатов")
            return []

        metadata_map = {}
        doc_id_to_score = {}
        doc_id_to_text = {}
        for doc_id, score, text, metadata in hybrid_results:
            metadata_map[doc_id] = metadata
            doc_id_to_score[doc_id] = score
            doc_id_to_text[doc_id] = text

        rerank_time = None
        if use_rerank and self.enable_reranker and self.reranker and hybrid_results:
            rerank_start_time = time.time()

            rerank_limit = min(top_k, len(hybrid_results))
            rerank_doc_ids = [doc_id for doc_id, _, _, _ in hybrid_results[:rerank_limit]]
            rerank_documents = [doc_id_to_text.get(doc_id, "") for doc_id in rerank_doc_ids]

            rerank_results = await asyncio.to_thread(self.reranker.rerank, query, rerank_documents, top_n=top_n)
            rerank_time = time.time() - rerank_start_time
            logger.info(
                f"✅ [retriever][retriever_service] Reranking завершен: {len(rerank_results)}/{top_n} результатов за {rerank_time:.2f}с (из {rerank_limit} документов)"
            )

            final_results = []
            for rerank_idx, _rerank_score in rerank_results:
                doc_id = rerank_doc_ids[rerank_idx]
                final_results.append(
                    (
                        doc_id,
                        doc_id_to_score.get(doc_id, 0.0),
                        doc_id_to_text.get(doc_id, ""),
                        metadata_map.get(doc_id),
                    )
                )
        else:
            final_results = [
                (doc_id, score, text, metadata_map.get(doc_id)) for doc_id, score, text, _ in hybrid_results[:top_n]
            ]

        total_search_time = time.time() - search_start_time
        rerank_str = f"{rerank_time:.2f}с" if rerank_time is not None else "N/A"
        logger.info(
            f"✅ [retriever][retriever_service] Поиск завершен: {len(final_results)} результатов за {total_search_time:.2f}с "
            f"(hybrid: {hybrid_time:.2f}с, rerank: {rerank_str})"
        )
        return final_results

    async def get_documents(self, doc_ids: list[str]) -> list[tuple[str, str, dict | None]]:
        """
        Получить документы по их ID

        Args:
            doc_ids (list[str]): Список ID документов

        Returns:
            list[tuple[str, str, dict | None]]: Список кортежей (doc_id, text, metadata)

        Raises:
            ValueError: Если список ID пуст
        """
        if not doc_ids:
            raise ValueError("Список ID документов не может быть пустым")

        try:
            results = await self.vector_search.get_documents(doc_ids)
            logger.info(f"✅ [retriever][retriever_service] Получено {len(results)} документов")
            return results
        except Exception as e:
            logger.error(f"❌ [retriever][retriever_service] Ошибка при получении документов: {e}")
            raise

    async def get_all_documents(self) -> list[tuple[str, str, dict | None]]:
        """
        Получить все документы из векторной базы данных

        Returns:
            list[tuple[str, str, dict | None]]: Список кортежей (doc_id, text, metadata)
        """
        try:
            results = await self.vector_search.get_all_documents()
            logger.info(f"✅ [retriever][retriever_service] Получено {len(results)} документов")
            return results
        except Exception as e:
            logger.error(f"❌ [retriever][retriever_service] Ошибка при получении всех документов: {e}")
            raise

    async def delete_documents(self, doc_ids: list[str]) -> None:
        """
        Удалить документы из векторной базы данных

        Args:
            doc_ids (list[str]): Список ID документов для удаления

        Raises:
            ValueError: Если список ID пуст
        """
        if not doc_ids:
            raise ValueError("Список ID документов для удаления не может быть пустым")

        try:
            await self.vector_search.delete_documents(doc_ids)
            logger.info(f"✅ [retriever][retriever_service] Удалено {len(doc_ids)} документов из Qdrant")
        except Exception as e:
            logger.error(f"❌ [retriever][retriever_service] Ошибка при удалении документов: {e}")
            raise

    async def delete_all_documents(self) -> None:
        """Удалить все документы из векторной базы данных"""
        try:
            await self.vector_search.delete_all_documents()
            logger.warning("⚠️ [retriever][retriever_service] Все документы удалены из Qdrant")
        except Exception as e:
            logger.error(f"❌ [retriever][retriever_service] Ошибка при удалении всех документов: {e}")
            raise
