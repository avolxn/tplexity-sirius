FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYTHONPATH=/app:/app/src

RUN pip install poetry

WORKDIR /app

COPY src/tplexity/parser/pyproject.toml ./
RUN poetry install --no-root

COPY src/tplexity/__init__.py ./src/tplexity/__init__.py
COPY src/tplexity/parser/ ./src/tplexity/parser/
RUN poetry install --no-root

EXPOSE 8011

CMD ["poetry", "run", "uvicorn", "tplexity.parser.app:app", "--host", "0.0.0.0", "--port", "8011"]
