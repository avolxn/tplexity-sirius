from tplexity.retriever.retriever_service import RetrieverService

_retriever_instance: RetrieverService | None = None


def get_retriever() -> RetrieverService:
    """
    Получить экземпляр RetrieverService (singleton)

    Returns:
        RetrieverService: Экземпляр гибридного поисковика
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RetrieverService()
    return _retriever_instance
