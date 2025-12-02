import logging

from fastapi import APIRouter, Depends, HTTPException, status

from tplexity.generation.api.dependencies import get_generation
from tplexity.generation.api.schemas import (
    ClearSessionRequest,
    ClearSessionResponse,
    GenerateRequest,
    GenerateResponse,
    GenerateShortAnswerRequest,
    GenerateShortAnswerResponse,
    SourceInfo,
)
from tplexity.generation.generation_service import GenerationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/generation", tags=["generation"])


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    generation: GenerationService = Depends(get_generation),
) -> GenerateResponse:
    """
    Генерация ответа с использованием RAG (Retrieval-Augmented Generation)

    Процесс:
    1. Поиск релевантных документов через retriever
    2. Формирование промпта с контекстом
    3. Генерация ответа через LLM

    Args:
        request: Запрос с вопросом пользователя и параметрами
        generation: Экземпляр GenerationService

    Returns:
        GenerateResponse: Сгенерированный ответ с источниками
    """
    try:
        answer, doc_ids, metadatas, search_time, generation_time, total_time = await generation.generate(
            query=request.query,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            llm_provider=request.llm_provider,
            session_id=request.session_id,
        )

        sources = []
        for doc_id, metadata in zip(doc_ids, metadatas, strict=False):
            sources.append(SourceInfo(doc_id=doc_id, metadata=metadata))

        return GenerateResponse(
            answer=answer,
            sources=sources,
            query=request.query,
            search_time=search_time,
            generation_time=generation_time,
            total_time=total_time,
        )
    except ValueError as e:
        logger.error(f"❌ [generation][routers] Ошибка валидации: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"❌ [generation][routers] Ошибка при генерации ответа: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации ответа: {str(e)}",
        ) from e


@router.post("/clear-session", response_model=ClearSessionResponse)
async def clear_session(
    request: ClearSessionRequest,
    generation: GenerationService = Depends(get_generation),
) -> ClearSessionResponse:
    """
    Очистка истории диалога для указанной сессии

    Args:
        request: Запрос с идентификатором сессии
        generation: Экземпляр GenerationService

    Returns:
        ClearSessionResponse: Результат операции очистки
    """
    try:
        await generation.clear_session(request.session_id)
        return ClearSessionResponse(
            success=True,
            message=f"История сессии {request.session_id} успешно очищена",
        )
    except Exception as e:
        logger.error(f"❌ [generation][routers] Ошибка при очистке истории сессии {request.session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при очистке истории сессии: {str(e)}",
        ) from e


@router.post("/generate-short-answer", response_model=GenerateShortAnswerResponse)
async def generate_short_answer(
    request: GenerateShortAnswerRequest,
    generation: GenerationService = Depends(get_generation),
) -> GenerateShortAnswerResponse:
    """
    Генерация краткого ответа на основе детального ответа

    Args:
        request: Запрос с детальным ответом для сокращения
        generation: Экземпляр GenerationService

    Returns:
        GenerateShortAnswerResponse: Краткий ответ
    """
    try:
        short_answer = await generation.generate_short_answer(
            detailed_answer=request.detailed_answer,
            llm_provider=request.llm_provider,
        )

        return GenerateShortAnswerResponse(short_answer=short_answer)
    except ValueError as e:
        logger.error(f"❌ [generation][routers] Ошибка валидации: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"❌ [generation][routers] Ошибка при генерации краткого ответа: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации краткого ответа: {str(e)}",
        ) from e
