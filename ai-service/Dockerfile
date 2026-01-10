FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for underthesea
RUN apt-get update ; apt-get install -y --no-install-recommends build-essential ; rm -rf /var/lib/apt/lists/*

COPY ai-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only what we need
COPY shared /app/shared
COPY ai-service /app/ai_service

ENV PYTHONPATH=/app

# Survival mode: keep workers low (override via env)
CMD ["sh", "-lc", "uvicorn ai_service.app:app --host 0.0.0.0 --port 8001 --workers ${AI_WORKERS:-1}"]
