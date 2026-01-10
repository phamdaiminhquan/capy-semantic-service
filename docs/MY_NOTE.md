powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\dev_api.ps1 -PyVersion 3.11

# Local/dev defaults

# Postgres
POSTGRES_USER=capy
POSTGRES_PASSWORD=capy123
POSTGRES_DB=capy_teacher
POSTGRES_PORT=5432

# App
APP_PORT=8000
APP_INTERNAL_PORT=8000
APP_HOST=0.0.0.0
UVICORN_RELOAD=--reload

# Runtime paths
HF_HOME=/cache/hf
MODEL_DIR=/app/artifacts/phobert_textcls
DATABASE_URL=postgresql://capy:capy123@postgres:5432/capy_teacher
