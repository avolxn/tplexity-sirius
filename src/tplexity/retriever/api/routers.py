import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException, status

from tplexity.retriever.api.dependencies import get_retriever
from tplexity.retriever.api.schemas import (
    DeleteDocumentsRequest,
    DocumentResponse,
    DocumentsRequest,
    DocumentsResponse,
    GetDocumentsRequest,
    MessageResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from tplexity.retriever.retriever_service import RetrieverService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/retriever", tags=["retriever"])


@router.post("/documents", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def add_documents(
    request: DocumentsRequest,
    retriever: RetrieverService = Depends(get_retriever),
) -> MessageResponse:
    """
    Добавить документы в векторную базу данных

    Args:
        request: Запрос с документами для добавления
        retriever: Экземпляр RetrieverService

    Returns:
        MessageResponse: Сообщение об успешном добавлении
    """
    try:
        documents = [doc.text for doc in request.documents]
        metadatas = (
            [doc.metadata for doc in request.documents] if any(doc.metadata for doc in request.documents) else None
        )

        await retriever.add_documents(documents, metadatas=metadatas)

        return MessageResponse(
            message=f"Успешно добавлено {len(documents)} документов",
            success=True,
        )
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"❌ [retriever][routers] Ошибка при добавлении документов: {e}\n{error_traceback}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при добавлении документов: {str(e)}",
        ) from e


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retriever: RetrieverService = Depends(get_retriever),
) -> SearchResponse:
    """
    Поиск документов по запросу

    Args:
        request: Запрос с параметрами поиска
        retriever: Экземпляр RetrieverService

    Returns:
        SearchResponse: Результаты поиска
    """
    try:
        results = await retriever.search(
            query=request.query,
            top_k=request.top_k,
            top_n=request.top_n,
            use_rerank=request.use_rerank,
            messages=request.messages,
        )

        search_results = [
            SearchResult(
                doc_id=doc_id,
                score=score,
                text=text,
                metadata=metadata,
            )
            for doc_id, score, text, metadata in results
        ]

        return SearchResponse(
            results=search_results,
            total=len(search_results),
        )
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"❌ [retriever][routers] Ошибка при поиске: {e}\n{error_traceback}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при поиске: {str(e)}",
        ) from e


@router.post("/documents/get", response_model=DocumentsResponse)
async def get_documents(
    request: GetDocumentsRequest,
    retriever: RetrieverService = Depends(get_retriever),
) -> DocumentsResponse:
    """
    Получить документы по их ID

    Args:
        request: Запрос с ID документов для получения
        retriever: Экземпляр RetrieverService

    Returns:
        DocumentsResponse: Список документов с текстами и метаданными
    """
    try:
        results = await retriever.get_documents(request.doc_ids)

        documents = [
            DocumentResponse(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
            )
            for doc_id, text, metadata in results
        ]

        return DocumentsResponse(
            documents=documents,
            total=len(documents),
        )
    except Exception as e:
        logger.error(f"❌ [retriever][routers] Ошибка при получении документов: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении документов: {str(e)}",
        ) from e


@router.get("/documents/all", response_model=DocumentsResponse)
async def get_all_documents(
    retriever: RetrieverService = Depends(get_retriever),
) -> DocumentsResponse:
    """
    Получить все документы из векторной базы данных

    Args:
        retriever: Экземпляр RetrieverService

    Returns:
        DocumentsResponse: Список всех документов с текстами и метаданными
    """
    try:
        results = await retriever.get_all_documents()

        documents = [
            DocumentResponse(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
            )
            for doc_id, text, metadata in results
        ]

        return DocumentsResponse(
            documents=documents,
            total=len(documents),
        )
    except Exception as e:
        logger.error(f"❌ [retriever][routers] Ошибка при получении всех документов: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении всех документов: {str(e)}",
        ) from e


@router.delete("/documents", response_model=MessageResponse)
async def delete_documents(
    request: DeleteDocumentsRequest,
    retriever: RetrieverService = Depends(get_retriever),
) -> MessageResponse:
    """
    Удалить документы по их ID

    Args:
        request: Запрос с ID документов для удаления
        retriever: Экземпляр RetrieverService

    Returns:
        MessageResponse: Сообщение об успешном удалении
    """
    try:
        await retriever.delete_documents(request.doc_ids)

        return MessageResponse(
            message=f"Успешно удалено {len(request.doc_ids)} документов",
            success=True,
        )
    except Exception as e:
        logger.error(f"❌ [retriever][routers] Ошибка при удалении документов: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при удалении документов: {str(e)}",
        ) from e


@router.delete("/documents/all", response_model=MessageResponse)
async def delete_all_documents(retriever: RetrieverService = Depends(get_retriever)) -> MessageResponse:
    """
    Удалить все документы из векторной базы данных

    Args:
        retriever: Экземпляр RetrieverService

    Returns:
        MessageResponse: Сообщение об успешном удалении
    """
    try:
        await retriever.delete_all_documents()

        return MessageResponse(
            message="Все документы успешно удалены",
            success=True,
        )
    except Exception as e:
        logger.error(f"❌ [retriever][routers] Ошибка при удалении всех документов: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при удалении всех документов: {str(e)}",
        ) from e
