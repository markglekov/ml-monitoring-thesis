from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.common.config import settings


ROOT = Path(__file__).resolve().parents[1]
SQL_FILES = (
    ROOT / "sql" / "init.sql",
    ROOT / "sql" / "monitoring_tables.sql",
    ROOT / "sql" / "quality_tables.sql",
)


def _candidate_database_urls() -> list[str]:
    explicit_url = os.getenv("TEST_DATABASE_URL")
    if explicit_url:
        return [explicit_url]

    urls = [settings.database_url]

    fallback_port = int(os.getenv("TEST_POSTGRES_PORT", "55432"))
    fallback_url = (
        f"postgresql+psycopg2://{settings.postgres_user}:{settings.postgres_password}"
        f"@127.0.0.1:{fallback_port}/{settings.postgres_db}"
    )
    if fallback_url not in urls:
        urls.append(fallback_url)

    localhost_fallback = (
        f"postgresql+psycopg2://{settings.postgres_user}:{settings.postgres_password}"
        f"@localhost:{fallback_port}/{settings.postgres_db}"
    )
    if localhost_fallback not in urls:
        urls.append(localhost_fallback)

    return urls


def _try_connect(url: str) -> Engine | None:
    engine = create_engine(url, future=True, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return engine
    except SQLAlchemyError:
        engine.dispose()
        return None


def _apply_sql_file(url: str, schema: str, sql_path: Path) -> None:
    raw_engine = create_engine(url, future=True, pool_pre_ping=True)
    try:
        raw_connection = raw_engine.raw_connection()
        try:
            cursor = raw_connection.cursor()
            cursor.execute(f'SET search_path TO "{schema}", public')
            cursor.execute(sql_path.read_text(encoding="utf-8"))
            raw_connection.commit()
        finally:
            raw_connection.close()
    finally:
        raw_engine.dispose()


@pytest.fixture(scope="session")
def postgres_base_url() -> str:
    for url in _candidate_database_urls():
        engine = _try_connect(url)
        if engine is not None:
            engine.dispose()
            return url

    pytest.skip("PostgreSQL integration database is not available.")


@pytest.fixture()
def postgres_engine(postgres_base_url: str) -> Engine:
    schema = f"test_{uuid.uuid4().hex[:12]}"
    admin_engine = create_engine(postgres_base_url, future=True, pool_pre_ping=True)

    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    for sql_path in SQL_FILES:
        _apply_sql_file(postgres_base_url, schema, sql_path)

    engine = create_engine(
        postgres_base_url,
        future=True,
        pool_pre_ping=True,
        connect_args={"options": f"-csearch_path={schema},public"},
    )

    try:
        yield engine
    finally:
        engine.dispose()
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin_engine.dispose()
