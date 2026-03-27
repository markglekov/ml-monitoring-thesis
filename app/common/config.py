"""Centralized project configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _get_env(name: str, default: str) -> str:
    """Read an environment variable with a string fallback."""

    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    """Application settings shared by API, training, and background jobs."""

    postgres_db: str
    postgres_user: str
    postgres_password: str
    postgres_host: str
    postgres_port: int

    model_path: Path
    baseline_path: Path
    model_version: str

    api_host: str
    api_port: int

    @property
    def database_url(self) -> str:
        """Build a SQLAlchemy-compatible PostgreSQL connection URL."""

        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings(
    postgres_db=_get_env("POSTGRES_DB", "ml_monitoring"),
    postgres_user=_get_env("POSTGRES_USER", "mlops"),
    postgres_password=_get_env("POSTGRES_PASSWORD", "mlops"),
    postgres_host=_get_env("POSTGRES_HOST", "localhost"),
    postgres_port=int(_get_env("POSTGRES_PORT", "5432")),
    model_path=ROOT / _get_env("MODEL_PATH", "artifacts/models/bank_marketing_model.joblib"),
    baseline_path=ROOT / _get_env("BASELINE_PATH", "artifacts/baselines/baseline_profile.json"),
    model_version=_get_env("MODEL_VERSION", "bank_marketing_v1"),
    api_host=_get_env("API_HOST", "0.0.0.0"),
    api_port=int(_get_env("API_PORT", "8000")),
)
