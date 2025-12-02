import logging

from fastapi import APIRouter, HTTPException

from tplexity.llm_client.api.dependencies import get_llm_client
from tplexity.llm_client.api.schemas import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/llm", tags=["llm"])


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Генерация ответа через LLM

    Args:
        request: Запрос на генерацию

    Returns:
        GenerateResponse: Ответ с сгенерированным текстом

    Raises:
        HTTPException: При ошибке вызова LLM
    """
    try:
        llm_client = get_llm_client(request.provider)

        answer = await llm_client.generate(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return GenerateResponse(
            answer=answer,
            provider=request.provider,
            model=llm_client.model,
        )
    except ValueError as e:
        logger.error(f"❌ [llm_client][routers] Ошибка валидации: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ [llm_client][routers] Ошибка при генерации: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации: {str(e)}")
