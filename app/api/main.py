"""Inference API for serving model predictions and persisting prediction logs."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.common.config import settings
from app.common.logging import get_logger, setup_logging


logger = get_logger(__name__)


class PredictRequest(BaseModel):
    """Incoming payload for a single online inference request."""

    features: dict[str, Any] = Field(..., description="Feature dictionary expected by the model.")
    segment_key: str | None = Field(default=None, description="Optional segment identifier for monitoring.")
    request_id: str | None = Field(default=None, description="Optional external request identifier.")


class PredictResponse(BaseModel):
    """Response payload returned after a successful prediction."""

    request_id: str
    model_version: str
    score: float
    threshold: float
    predicted_label: int
    segment_key: str | None


def to_native(value: Any) -> Any:
    """Convert pandas and numpy objects into JSON-serializable Python values."""

    if isinstance(value, dict):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_native(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def normalize_feature_value(value: Any) -> Any:
    """Convert JSON-null-like values back into pandas-compatible missing values."""

    if value is None:
        return np.nan
    return value


def build_inference_frame(features: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """Build a single-row dataframe with normalized missing values for model inference."""

    row = {column: normalize_feature_value(features[column]) for column in feature_columns}
    return pd.DataFrame([row])


def get_model(app: FastAPI) -> Any:
    """Return the loaded model or raise a 503 if the service is not ready."""

    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return model


def get_engine(app: FastAPI) -> Engine:
    """Return the database engine or raise a 503 if the service is not ready."""

    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Database engine is not initialized.")
    return engine


def get_feature_columns(app: FastAPI) -> list[str]:
    """Return the feature schema extracted from the saved baseline profile."""

    feature_columns = getattr(app.state, "feature_columns", None)
    if feature_columns is None:
        raise HTTPException(status_code=503, detail="Feature schema is not loaded.")
    return feature_columns


def get_threshold(app: FastAPI) -> float:
    """Return the active decision threshold loaded at startup."""

    return float(getattr(app.state, "threshold", 0.5))


def validate_features(features: dict[str, Any], feature_columns: list[str]) -> None:
    """Validate that request features exactly match the trained model schema."""

    missing = [column for column in feature_columns if column not in features]
    extra = [column for column in features if column not in feature_columns]

    if missing or extra:
        detail: dict[str, Any] = {"message": "Feature set does not match the trained baseline schema."}
        if missing:
            detail["missing_features"] = missing
        if extra:
            detail["extra_features"] = extra
        raise HTTPException(status_code=422, detail=detail)


def insert_inference_log(
    engine: Engine,
    request_id: str,
    features: dict[str, Any],
    score: float,
    pred_label: int,
    threshold: float,
    segment_key: str | None,
    latency_ms: float,
) -> None:
    """Persist one prediction event into PostgreSQL."""

    query = text(
        """
        INSERT INTO inference_log (
            request_id,
            model_version,
            features_json,
            score,
            pred_label,
            threshold,
            segment_key,
            latency_ms
        )
        VALUES (
            CAST(:request_id AS UUID),
            :model_version,
            CAST(:features_json AS JSONB),
            :score,
            :pred_label,
            :threshold,
            :segment_key,
            :latency_ms
        )
        """
    )

    payload = {
        "request_id": request_id,
        "model_version": settings.model_version,
        "features_json": json.dumps(to_native(features), ensure_ascii=False),
        "score": float(score),
        "pred_label": int(pred_label),
        "threshold": float(threshold),
        "segment_key": segment_key,
        "latency_ms": float(latency_ms),
    }

    with engine.begin() as connection:
        connection.execute(query, payload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and database resources once for the API process."""

    setup_logging()

    model_path = settings.model_path
    baseline_path = settings.baseline_path

    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")

    if not baseline_path.exists():
        raise RuntimeError(f"Baseline file not found: {baseline_path}")

    logger.info("Loading model from %s", model_path)
    app.state.model = joblib.load(model_path)

    logger.info("Loading baseline profile from %s", baseline_path)
    with baseline_path.open("r", encoding="utf-8") as file_obj:
        baseline_profile = json.load(file_obj)

    app.state.baseline_profile = baseline_profile
    app.state.feature_columns = baseline_profile["feature_columns"]
    app.state.threshold = float(baseline_profile.get("threshold", 0.5))

    logger.info("Connecting to PostgreSQL at %s:%s", settings.postgres_host, settings.postgres_port)
    app.state.engine = create_engine(settings.database_url, future=True, pool_pre_ping=True)

    with app.state.engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    logger.info(
        "API startup completed. model_version=%s, feature_count=%s, threshold=%.4f",
        settings.model_version,
        len(app.state.feature_columns),
        app.state.threshold,
    )

    try:
        yield
    finally:
        engine = getattr(app.state, "engine", None)
        if engine is not None:
            logger.info("Disposing database engine.")
            engine.dispose()


app = FastAPI(
    title="ML Monitoring Thesis API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Return service readiness information for probes and operators."""

    engine = get_engine(app)
    model = get_model(app)
    feature_columns = get_feature_columns(app)

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.exception("Database health check failed.")
        raise HTTPException(status_code=503, detail=f"Database check failed: {exc}") from exc

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": settings.model_version,
        "feature_count": len(feature_columns),
        "threshold": get_threshold(app),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Run online inference and persist the prediction event into PostgreSQL."""

    model = get_model(app)
    engine = get_engine(app)
    feature_columns = get_feature_columns(app)
    threshold = get_threshold(app)

    validate_features(payload.features, feature_columns)

    request_id = payload.request_id or str(uuid.uuid4())
    started_at = time.perf_counter()
    row = {column: payload.features[column] for column in feature_columns}
    X = build_inference_frame(payload.features, feature_columns)

    try:
        score = float(model.predict_proba(X)[:, 1][0])
    except Exception as exc:
        logger.exception("Prediction failed for request_id=%s", request_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    pred_label = int(score >= threshold)
    latency_ms = (time.perf_counter() - started_at) * 1000.0

    try:
        insert_inference_log(
            engine=engine,
            request_id=request_id,
            features=row,
            score=score,
            pred_label=pred_label,
            threshold=threshold,
            segment_key=payload.segment_key,
            latency_ms=latency_ms,
        )
    except IntegrityError as exc:
        logger.warning("Duplicate request_id rejected: %s", request_id)
        raise HTTPException(status_code=409, detail=f"request_id already exists: {request_id}") from exc
    except SQLAlchemyError as exc:
        logger.exception("Failed to persist inference log for request_id=%s", request_id)
        raise HTTPException(status_code=500, detail=f"Failed to write inference log: {exc}") from exc

    logger.info(
        "Prediction served. request_id=%s model_version=%s score=%.4f label=%s latency_ms=%.2f segment_key=%s",
        request_id,
        settings.model_version,
        score,
        pred_label,
        latency_ms,
        payload.segment_key,
    )

    return PredictResponse(
        request_id=request_id,
        model_version=settings.model_version,
        score=score,
        threshold=threshold,
        predicted_label=pred_label,
        segment_key=payload.segment_key,
    )
