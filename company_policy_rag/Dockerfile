FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    RERANKER_DEVICE=cpu \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-docker.txt

COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY .streamlit/ ./.streamlit/
COPY data/eval/ ./data/eval/
COPY docker/entrypoint.sh ./docker/entrypoint.sh

RUN chmod +x ./docker/entrypoint.sh \
    && mkdir -p data/policies data/legal data/raw storage logs

EXPOSE 8501

CMD ["./docker/entrypoint.sh"]