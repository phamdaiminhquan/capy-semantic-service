from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse


def _is_running_in_docker() -> bool:
    if os.getenv("RUNNING_IN_DOCKER") in {"1", "true", "TRUE", "yes", "YES"}:
        return True
    # Common Docker signal on Linux containers
    return os.path.exists("/.dockerenv")


def _load_env_files() -> None:
    """Load environment variables from .env files when running locally.

    docker-compose already injects env vars for containers, but when running
    `uvicorn api:app --reload` locally we need to load `.env` ourselves.
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    repo_root = Path(__file__).resolve().parent.parent
    for candidate in (repo_root / ".env", repo_root / ".env.production", Path.cwd() / ".env"):
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)


def _normalize_database_url(database_url: str) -> str:
    """Make DATABASE_URL usable for both Docker and local runs.

    The repo's `.env` is designed for docker-compose (`host=postgres`).
    When running on the host OS, that hostname won't resolve, so we rewrite
    `postgres` -> `localhost` unless we're in Docker.
    """
    if _is_running_in_docker():
        return database_url

    parsed = urlparse(database_url)
    if parsed.hostname != "postgres":
        return database_url

    username = parsed.username or ""
    password = parsed.password or ""
    auth = username
    if password:
        auth = f"{username}:{quote(password, safe='')}"

    host = "localhost"
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{auth}@{host}{port}" if auth else f"{host}{port}"

    return urlunparse(parsed._replace(netloc=netloc))


def _build_database_url_from_postgres_env() -> str | None:
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_DB")
    port = os.getenv("POSTGRES_PORT", "5432")
    host = os.getenv("POSTGRES_HOST")
    if not host:
        host = "postgres" if _is_running_in_docker() else "localhost"

    if not (user and password and db):
        return None

    return f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{quote(db, safe='')}"


_load_env_files()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    DATABASE_URL = _normalize_database_url(DATABASE_URL)
else:
    DATABASE_URL = _build_database_url_from_postgres_env()

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. "
        "Create a .env file (or set env vars) with DATABASE_URL or POSTGRES_USER/POSTGRES_PASSWORD/POSTGRES_DB."
    )

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)

