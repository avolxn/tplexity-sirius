from tplexity.generation.generation_service import GenerationService

_generation_instance: GenerationService | None = None


def get_generation() -> GenerationService:
    """
    Получить экземпляр GenerationService (singleton)

    Returns:
        GenerationService: Экземпляр сервиса генерации
    """
    global _generation_instance
    if _generation_instance is None:
        _generation_instance = GenerationService()
    return _generation_instance
