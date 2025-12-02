import asyncio
import logging
import traceback
from typing import Literal
from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    Modifier,
    PointIdsList,
    PointStruct,
    Prefetch,
    SparseVectorParams,
    VectorParams,
)

from tplexity.retriever.config import settings
from tplexity.retriever.dense_embedding import get_embedding_model
from tplexity.retriever.sparse_embedding import get_bm25_model

logger = logging.getLogger(__name__)


class VectorSearch:
    """Класс для векторного поиска через Qdrant с поддержкой dense и sparse векторов"""

    def __init__(
        self,
        collection_name: str,
        host: str,
        port: int,
        api_key: str | None,
        prefetch_ratio: float,
    ):
        """Инициализация векторного поисковика

        Args:
            collection_name (str): Имя коллекции в Qdrant
            host (str): Хост Qdrant
            port (int): Порт Qdrant
            api_key (str | None): API ключ для Qdrant
            prefetch_ratio (float): Во сколько раз больше результатов для prefetch
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.api_key = api_key
        self.prefetch_ratio = prefetch_ratio

        logger.info("🔄 [retriever][vector_search] Инициализация клиента Qdrant")
        try:
            self.client = AsyncQdrantClient(
                url=f"https://{self.host}:{self.port}",
                api_key=self.api_key,
            )

            logger.info(f"✅ [retriever][vector_search] Клиент Qdrant инициализирован: {self.host}:{self.port}")
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                f"❌ [retriever][vector_search] Ошибка инициализации клиента Qdrant: {e}\n{error_traceback}",
                exc_info=True,
            )
            raise

        self.embedding_model = get_embedding_model()
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"✅ [retriever][vector_search] Dense модель инициализирована, размерность: {self.embedding_dim}")

        self.bm25 = get_bm25_model()
        logger.info("✅ [retriever][vector_search] BM25 модель инициализирована")

    async def _ensure_collection(self) -> None:
        """Создать коллекцию с поддержкой dense и sparse векторов, если не существует"""

        try:
            collections = await self.client.get_collections()
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                f"❌ [retriever][vector_search] Ошибка при получении списка коллекций: {type(e).__name__}: {e}\n{error_traceback}",
                exc_info=True,
            )
            raise

        collection_names = [col.name for col in collections.collections]

        if self.collection_name not in collection_names:
            vectors_config = {
                "dense": VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                )
            }

            sparse_vectors_config = {
                "bm25": SparseVectorParams(
                    modifier=Modifier.IDF,
                ),
            }

            try:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config,
                )
                logger.info(
                    f"✅ [retriever][vector_search] Коллекция {self.collection_name} создана с dense и sparse векторами"
                )
            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(
                    f"❌ [retriever][vector_search] Ошибка при создании коллекции: {type(e).__name__}: {e}\n{error_traceback}",
                    exc_info=True,
                )
                raise
        else:
            logger.info(f"✅ [retriever][vector_search] Коллекция {self.collection_name} уже существует")

    async def add_documents(
        self,
        documents: list[str],
        ids: list[str] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        """
        Добавить документы в векторную базу данных с dense и sparse векторами

        Args:
            documents (list[str]): Список документов для добавления
            ids (list[str] | None): Список ID для документов. Если None, генерируются UUID
            metadatas (list[dict] | None): Список словарей с метаданными для каждого документа

        Raises:
            ValueError: Если входные данные невалидны
        """
        if not documents:
            raise ValueError("Список документов не может быть пустым")

        if metadatas is None:
            metadatas = [{}] * len(documents)

        if len(metadatas) != len(documents):
            raise ValueError(
                f"Количество метаданных ({len(metadatas)}) не совпадает с количеством документов ({len(documents)})"
            )

        if ids is None:
            ids = [str(uuid4()) for _ in documents]

        if len(ids) != len(set(ids)):
            raise ValueError("ID документов должны быть уникальными")

        logger.debug(
            f"🔄 [retriever][vector_search] Начало параллельной генерации embeddings для {len(documents)} документов"
        )

        dense_embeddings, sparse_embeddings = await asyncio.gather(
            asyncio.to_thread(self.embedding_model.encode_document, documents),
            asyncio.to_thread(self.bm25.encode_documents, documents),
        )

        logger.debug(
            f"✅ [retriever][vector_search] Embeddings сгенерированы: dense={len(dense_embeddings)}, sparse={len(sparse_embeddings)}"
        )

        points = []
        for document_id, document, dense_emb, sparse_emb, metadata in zip(
            ids, documents, dense_embeddings, sparse_embeddings, metadatas, strict=False
        ):
            vectors = {
                "dense": dense_emb,
                "bm25": sparse_emb.as_object(),
            }
            payload = {"text": document, **metadata}

            points.append(PointStruct(id=document_id, vector=vectors, payload=payload))

        await self._ensure_collection()

        try:
            await self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(
                f"✅ [retriever][vector_search] Добавлено {len(documents)} документов в коллекцию {self.collection_name}"
            )
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                f"❌ [retriever][vector_search] Ошибка при добавлении документов в Qdrant: {type(e).__name__}: {e}\n{error_traceback}",
                exc_info=True,
            )
            raise

    async def search(
        self,
        query: str,
        top_k: int = 10,
        search_type: Literal["dense", "sparse", "hybrid"] = "hybrid",
    ) -> list[tuple[str, float, str, dict | None]]:
        """
        Поиск документов по запросу с использованием различных типов поиска

        Args:
            query (str): Поисковый запрос
            top_k (int): Количество возвращаемых результатов
            search_type (Literal["dense", "sparse", "hybrid"]): Тип поиска (dense, sparse, hybrid). По умолчанию "hybrid"

        Returns:
            list[tuple[str, float, str, dict | None]]: Список кортежей (ID документа, score, текст, метаданные)

        Raises:
            ValueError: Если запрос пуст или параметры невалидны
        """
        if not query or not query.strip():
            logger.warning("⚠️ [retriever][vector_search] Передан пустой запрос")
            return []

        if top_k < 1:
            logger.error(f"❌ [retriever][vector_search] top_k должен быть >= 1, получено: {top_k}")
            return []

        if search_type == "hybrid":
            return await self._hybrid_search(query, top_k, self.prefetch_ratio)
        elif search_type == "dense":
            return await self._dense_search(query, top_k)
        elif search_type == "sparse":
            return await self._sparse_search(query, top_k)

    async def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float, str, dict | None]]:
        """
        Поиск только по dense векторам

        Args:
            query (str): Поисковый запрос
            top_k (int): Количество возвращаемых результатов

        Returns:
            list[tuple[str, float, str, dict | None]]: Список кортежей (ID документа, score, текст, метаданные)
        """
        logger.debug(f"🔍 [retriever][vector_search] Выполнение dense поиска для запроса: {query[:50]}...")
        query_embedding = await asyncio.to_thread(self.embedding_model.encode_query, query)

        try:
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_embedding),
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                f"❌ [retriever][vector_search] Ошибка при dense поиске: {type(e).__name__}: {e}\n{error_traceback}",
                exc_info=True,
            )
            raise

        results = []
        for result in search_results:
            text = result.payload.get("text", "")
            metadata = {k: v for k, v in result.payload.items() if k != "text"}
            results.append((str(result.id), float(result.score), text, metadata))

        return results

    async def _sparse_search(self, query: str, top_k: int) -> list[tuple[str, float, str, dict | None]]:
        """
        Поиск только по sparse векторам

        Args:
            query (str): Поисковый запрос
            top_k (int): Количество возвращаемых результатов

        Returns:
            list[tuple[str, float, str, dict | None]]: Список кортежей (ID документа, score, текст, метаданные)
        """
        logger.debug(f"🔍 [retriever][vector_search] Выполнение sparse поиска для запроса: {query[:50]}...")
        query_embedding = await asyncio.to_thread(self.bm25.encode_query, query)

        try:
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=("bm25", query_embedding),
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                f"❌ [retriever][vector_search] Ошибка при sparse поиске: {type(e).__name__}: {e}\n{error_traceback}",
                exc_info=True,
            )
            raise

        results = []
        for result in search_results:
            text = result.payload.get("text", "")
            metadata = {k: v for k, v in result.payload.items() if k != "text"}
            results.append((str(result.id), float(result.score), text, metadata))

        return results

    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        prefetch_ratio: float,
    ) -> list[tuple[str, float, str, dict | None]]:
        """
        Гибридный поиск с использованием prefetch и RRF

        Args:
            query (str): Поисковый запрос
            top_k (int): Количество возвращаемых результатов
            prefetch_ratio (float): Во сколько раз больше результатов для prefetch

        Returns:
            list[tuple[str, float, str, dict | None]]: Список кортежей (ID документа, score, текст, метаданные)
        """
        logger.debug(f"🔍 [retriever][vector_search] Выполнение гибридного поиска для запроса: {query[:50]}...")

        dense_query, sparse_query = await asyncio.gather(
            asyncio.to_thread(self.embedding_model.encode_query, query),
            asyncio.to_thread(self.bm25.encode_query, query),
        )

        prefetch = [
            Prefetch(
                query=dense_query,
                using="dense",
                limit=int(top_k * prefetch_ratio),
            ),
            Prefetch(
                query=sparse_query,
                using="bm25",
                limit=int(top_k * prefetch_ratio),
            ),
        ]

        try:
            search_results = await self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=FusionQuery(
                    fusion=Fusion.RRF,
                ),
                with_payload=True,
                limit=top_k,
            )
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                f"❌ [retriever][vector_search] Ошибка при гибридном поиске: {type(e).__name__}: {e}\n{error_traceback}",
                exc_info=True,
            )
            raise

        results = []
        for result in search_results.points:
            text = result.payload.get("text", "")
            metadata = {k: v for k, v in result.payload.items() if k != "text"}
            results.append((str(result.id), float(result.score), text, metadata))

        return results

    async def get_documents(self, doc_ids: list[str]) -> list[tuple[str, str, dict | None]]:
        """
        Получить документы по их ID

        Args:
            doc_ids (list[str]): Список ID документов

        Returns:
            list[tuple[str, str, dict | None]]: Список кортежей (doc_id, text, metadata)
        """
        if not doc_ids:
            logger.warning("⚠️ [retriever][vector_search] Передан пустой список ID для получения документов")
            return []

        try:
            results = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=doc_ids,
                with_payload=True,
            )

            documents = []
            for point in results:
                text = point.payload.get("text", "")
                metadata = {k: v for k, v in point.payload.items() if k != "text"}
                documents.append((str(point.id), text, metadata if metadata else None))

            logger.info(
                f"✅ [retriever][vector_search] Получено {len(documents)} документов из {len(doc_ids)} запрошенных"
            )
            return documents
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                f"❌ [retriever][vector_search] Ошибка при получении документов: {type(e).__name__}: {e}\n{error_traceback}",
                exc_info=True,
            )
            raise

    async def get_all_documents(self) -> list[tuple[str, str, dict | None]]:
        """
        Получить все документы из коллекции

        Returns:
            list[tuple[str, str, dict | None]]: Список кортежей (doc_id, text, metadata)
        """

        try:
            points, _ = await self.client.scroll(
                collection_name=self.collection_name,
                limit=None,
                with_payload=True,
            )

            documents = []
            for point in points:
                text = point.payload.get("text", "")
                metadata = {k: v for k, v in point.payload.items() if k != "text"}
                documents.append((str(point.id), text, metadata if metadata else None))

            logger.info(f"✅ [retriever][vector_search] Получено {len(documents)} документов из коллекции")
            return documents
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                f"❌ [retriever][vector_search] Ошибка при получении всех документов: {type(e).__name__}: {e}\n{error_traceback}",
                exc_info=True,
            )
            raise

    async def delete_documents(self, ids: list[str]) -> None:
        """
        Удалить документы из векторной базы данных по их ID

        Args:
            ids (list[str]): Список ID документов для удаления
        """
        if not ids:
            logger.warning("⚠️ [retriever][vector_search] Передан пустой список ID для удаления")
            return

        logger.info(f"🔄 [retriever][vector_search] Удаление {len(ids)} документов из коллекции {self.collection_name}")

        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids),
            )
            logger.info(
                f"✅ [retriever][vector_search] Успешно удалено {len(ids)} документов из коллекции {self.collection_name}"
            )
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                f"❌ [retriever][vector_search] Ошибка при удалении документов из Qdrant: {type(e).__name__}: {e}\n{error_traceback}",
                exc_info=True,
            )
            raise

    async def delete_all_documents(self) -> None:
        """Удалить все документы из коллекции"""
        logger.warning("⚠️ [retriever][vector_search] Удаление всех документов из коллекции")

        try:
            await self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"✅ [retriever][vector_search] Коллекция {self.collection_name} удалена")
            await self._ensure_collection()
            logger.info(f"✅ [retriever][vector_search] Коллекция {self.collection_name} пересоздана")
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                f"❌ [retriever][vector_search] Ошибка при удалении всех документов: {type(e).__name__}: {e}\n{error_traceback}",
                exc_info=True,
            )
            raise
