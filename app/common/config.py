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


def _get_int_env(name: str, default: int) -> int:
    """Read an integer environment variable with a numeric fallback."""

    return int(_get_env(name, str(default)))


def _get_float_env(name: str, default: float) -> float:
    """Read a floating-point environment variable with a numeric fallback."""

    return float(_get_env(name, str(default)))


def _get_optional_env(name: str) -> str | None:
    """Read an environment variable and normalize blank values to None."""

    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _get_bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable using common truthy values."""

    value = _get_optional_env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _get_list_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a comma-separated list from the environment."""

    value = _get_optional_env(name)
    if value is None:
        return default
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or default


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

    smtp_host: str
    smtp_port: int
    smtp_username: str | None
    smtp_password: str | None
    smtp_from: str
    alert_email_to: tuple[str, ...]
    smtp_use_starttls: bool
    smtp_use_ssl: bool
    smtp_timeout_sec: float

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
    postgres_port=_get_int_env("POSTGRES_PORT", 5432),
    model_path=ROOT / _get_env("MODEL_PATH", "artifacts/models/bank_marketing_model.joblib"),
    baseline_path=ROOT / _get_env("BASELINE_PATH", "artifacts/baselines/baseline_profile.json"),
    model_version=_get_env("MODEL_VERSION", "bank_marketing_v1"),
    api_host=_get_env("API_HOST", "0.0.0.0"),
    api_port=_get_int_env("API_PORT", 8000),
    smtp_host=_get_env("SMTP_HOST", ""),
    smtp_port=_get_int_env("SMTP_PORT", 587),
    smtp_username=_get_optional_env("SMTP_USERNAME"),
    smtp_password=_get_optional_env("SMTP_PASSWORD"),
    smtp_from=_get_env("SMTP_FROM", ""),
    alert_email_to=_get_list_env("ALERT_EMAIL_TO", default=()),
    smtp_use_starttls=_get_bool_env("SMTP_USE_STARTTLS", default=True),
    smtp_use_ssl=_get_bool_env("SMTP_USE_SSL", default=False),
    smtp_timeout_sec=_get_float_env("SMTP_TIMEOUT_SEC", 15.0),
)
