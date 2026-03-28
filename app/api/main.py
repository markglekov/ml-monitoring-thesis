"""Inference API for serving model predictions and persisting prediction logs."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import bindparam, create_engine, text
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


class GroundTruthLabelRequest(BaseModel):
    """Incoming payload for one delayed ground-truth label."""

    request_id: str
    y_true: Literal[0, 1]
    label_ts: datetime | None = Field(default=None, description="Optional label timestamp in UTC.")


class GroundTruthLabelsBatchRequest(BaseModel):
    """Batch payload for delayed labels ingestion."""

    labels: list[GroundTruthLabelRequest] = Field(..., min_length=1)


class GroundTruthLabelResponse(BaseModel):
    """Response for a single ingested delayed label."""

    request_id: str
    y_true: int
    label_ts: datetime
    status: str


class GroundTruthLabelsBatchResponse(BaseModel):
    """Response for a batch delayed-label ingestion request."""

    received_count: int
    upserted_count: int
    status: str


class DriftRunResponse(BaseModel):
    """One persisted drift-monitoring run returned by the API."""

    id: int
    ts_started: datetime
    ts_finished: datetime | None
    model_version: str
    window_size: int
    segment_key: str | None
    status: str
    drifted_features_count: int
    total_features_count: int
    overall_drift: bool
    summary: dict[str, Any] | None


class QualityRunResponse(BaseModel):
    """One persisted quality-monitoring run returned by the API."""

    id: int
    ts_started: datetime
    ts_finished: datetime | None
    model_version: str
    window_size: int
    segment_key: str | None
    status: str
    labeled_rows: int
    degraded_metrics_count: int
    summary: dict[str, Any] | None


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


def normalize_label_ts(label_ts: datetime | None) -> datetime:
    """Normalize delayed-label timestamps to timezone-aware UTC datetimes."""

    if label_ts is None:
        return datetime.now(timezone.utc)
    if label_ts.tzinfo is None:
        return label_ts.replace(tzinfo=timezone.utc)
    return label_ts.astimezone(timezone.utc)


def parse_json_object(value: Any) -> dict[str, Any] | None:
    """Normalize a JSON/JSONB field into a Python dict when possible."""

    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": to_native(value)}


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


def validate_limit(limit: int) -> None:
    """Validate common pagination limit for list endpoints."""

    if limit <= 0 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")


def find_missing_request_ids(engine: Engine, request_ids: list[str]) -> list[str]:
    """Return request identifiers that are absent from inference_log."""

    if not request_ids:
        return []

    query = text(
        """
        SELECT request_id::text AS request_id
        FROM inference_log
        WHERE request_id::text IN :request_ids
        """
    ).bindparams(bindparam("request_ids", expanding=True))

    with engine.connect() as connection:
        rows = connection.execute(query, {"request_ids": request_ids}).mappings().all()

    existing_ids = {str(row["request_id"]) for row in rows}
    return [request_id for request_id in request_ids if request_id not in existing_ids]


def find_duplicate_values(values: list[str]) -> list[str]:
    """Return duplicated string values while preserving a stable sorted output."""

    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)

    return sorted(duplicates)


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


def upsert_ground_truth_labels(engine: Engine, labels: list[GroundTruthLabelRequest]) -> int:
    """Insert or update delayed labels in ground_truth."""

    query = text(
        """
        INSERT INTO ground_truth (request_id, y_true, label_ts)
        VALUES (
            CAST(:request_id AS UUID),
            :y_true,
            :label_ts
        )
        ON CONFLICT (request_id) DO UPDATE
        SET
            y_true = EXCLUDED.y_true,
            label_ts = EXCLUDED.label_ts
        """
    )

    payloads = [
        {
            "request_id": label.request_id,
            "y_true": int(label.y_true),
            "label_ts": normalize_label_ts(label.label_ts),
        }
        for label in labels
    ]

    with engine.begin() as connection:
        result = connection.execute(query, payloads)

    return max(int(result.rowcount), 0) if result.rowcount is not None else 0


def list_drift_runs(engine: Engine, limit: int, segment_key: str | None = None) -> list[DriftRunResponse]:
    """List recent drift-monitoring runs with optional segment filtering."""

    if segment_key:
        query = text(
            """
            SELECT
                id,
                ts_started,
                ts_finished,
                model_version,
                window_size,
                segment_key,
                status,
                drifted_features_count,
                total_features_count,
                overall_drift,
                summary_json
            FROM monitoring_runs
            WHERE segment_key = :segment_key
            ORDER BY ts_started DESC, id DESC
            LIMIT :limit
            """
        )
        params = {"segment_key": segment_key, "limit": limit}
    else:
        query = text(
            """
            SELECT
                id,
                ts_started,
                ts_finished,
                model_version,
                window_size,
                segment_key,
                status,
                drifted_features_count,
                total_features_count,
                overall_drift,
                summary_json
            FROM monitoring_runs
            ORDER BY ts_started DESC, id DESC
            LIMIT :limit
            """
        )
        params = {"limit": limit}

    with engine.connect() as connection:
        rows = connection.execute(query, params).mappings().all()

    return [
        DriftRunResponse(
            id=int(row["id"]),
            ts_started=row["ts_started"],
            ts_finished=row["ts_finished"],
            model_version=str(row["model_version"]),
            window_size=int(row["window_size"]),
            segment_key=row["segment_key"],
            status=str(row["status"]),
            drifted_features_count=int(row["drifted_features_count"] or 0),
            total_features_count=int(row["total_features_count"] or 0),
            overall_drift=bool(row["overall_drift"]),
            summary=parse_json_object(row["summary_json"]),
        )
        for row in rows
    ]


def list_quality_runs(engine: Engine, limit: int, segment_key: str | None = None) -> list[QualityRunResponse]:
    """List recent quality-monitoring runs with optional segment filtering."""

    if segment_key:
        query = text(
            """
            SELECT
                id,
                ts_started,
                ts_finished,
                model_version,
                window_size,
                segment_key,
                status,
                labeled_rows,
                degraded_metrics_count,
                summary_json
            FROM quality_runs
            WHERE segment_key = :segment_key
            ORDER BY ts_started DESC, id DESC
            LIMIT :limit
            """
        )
        params = {"segment_key": segment_key, "limit": limit}
    else:
        query = text(
            """
            SELECT
                id,
                ts_started,
                ts_finished,
                model_version,
                window_size,
                segment_key,
                status,
                labeled_rows,
                degraded_metrics_count,
                summary_json
            FROM quality_runs
            ORDER BY ts_started DESC, id DESC
            LIMIT :limit
            """
        )
        params = {"limit": limit}

    with engine.connect() as connection:
        rows = connection.execute(query, params).mappings().all()

    return [
        QualityRunResponse(
            id=int(row["id"]),
            ts_started=row["ts_started"],
            ts_finished=row["ts_finished"],
            model_version=str(row["model_version"]),
            window_size=int(row["window_size"]),
            segment_key=row["segment_key"],
            status=str(row["status"]),
            labeled_rows=int(row["labeled_rows"] or 0),
            degraded_metrics_count=int(row["degraded_metrics_count"] or 0),
            summary=parse_json_object(row["summary_json"]),
        )
        for row in rows
    ]


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


@app.post("/labels", response_model=GroundTruthLabelResponse)
def ingest_label(payload: GroundTruthLabelRequest) -> GroundTruthLabelResponse:
    """Ingest one delayed ground-truth label for a previously served request."""

    engine = get_engine(app)
    missing_request_ids = find_missing_request_ids(engine, [payload.request_id])
    if missing_request_ids:
        raise HTTPException(status_code=404, detail=f"request_id not found: {payload.request_id}")

    try:
        upsert_ground_truth_labels(engine, [payload])
    except SQLAlchemyError as exc:
        logger.exception("Failed to persist delayed label for request_id=%s", payload.request_id)
        raise HTTPException(status_code=500, detail=f"Failed to write delayed label: {exc}") from exc

    normalized_label_ts = normalize_label_ts(payload.label_ts)
    logger.info("Delayed label ingested. request_id=%s y_true=%s", payload.request_id, payload.y_true)
    return GroundTruthLabelResponse(
        request_id=payload.request_id,
        y_true=int(payload.y_true),
        label_ts=normalized_label_ts,
        status="upserted",
    )


@app.post("/labels/batch", response_model=GroundTruthLabelsBatchResponse)
def ingest_labels_batch(payload: GroundTruthLabelsBatchRequest) -> GroundTruthLabelsBatchResponse:
    """Ingest a batch of delayed ground-truth labels."""

    engine = get_engine(app)
    request_ids = [label.request_id for label in payload.labels]
    duplicate_request_ids = find_duplicate_values(request_ids)
    if duplicate_request_ids:
        raise HTTPException(
            status_code=422,
            detail={"message": "Duplicate request_id values in batch payload.", "request_ids": duplicate_request_ids},
        )

    missing_request_ids = find_missing_request_ids(engine, request_ids)
    if missing_request_ids:
        raise HTTPException(
            status_code=422,
            detail={"message": "Some request_id values were not found in inference_log.", "request_ids": missing_request_ids},
        )

    try:
        upserted_count = upsert_ground_truth_labels(engine, payload.labels)
    except SQLAlchemyError as exc:
        logger.exception("Failed to persist delayed labels batch. batch_size=%s", len(payload.labels))
        raise HTTPException(status_code=500, detail=f"Failed to write delayed labels batch: {exc}") from exc

    logger.info("Delayed labels batch ingested. batch_size=%s upserted=%s", len(payload.labels), upserted_count)
    return GroundTruthLabelsBatchResponse(
        received_count=len(payload.labels),
        upserted_count=upserted_count,
        status="upserted",
    )


@app.get("/monitoring/drift/runs", response_model=list[DriftRunResponse])
def get_drift_runs(limit: int = 20, segment_key: str | None = None) -> list[DriftRunResponse]:
    """Return recent drift-monitoring runs for operators and dashboards."""

    validate_limit(limit)
    engine = get_engine(app)

    try:
        return list_drift_runs(engine=engine, limit=limit, segment_key=segment_key)
    except SQLAlchemyError as exc:
        logger.exception("Failed to fetch drift runs.")
        raise HTTPException(status_code=500, detail=f"Failed to fetch drift runs: {exc}") from exc


@app.get("/monitoring/quality/runs", response_model=list[QualityRunResponse])
def get_quality_runs(limit: int = 20, segment_key: str | None = None) -> list[QualityRunResponse]:
    """Return recent quality-monitoring runs for operators and dashboards."""

    validate_limit(limit)
    engine = get_engine(app)

    try:
        return list_quality_runs(engine=engine, limit=limit, segment_key=segment_key)
    except SQLAlchemyError as exc:
        logger.exception("Failed to fetch quality runs.")
        raise HTTPException(status_code=500, detail=f"Failed to fetch quality runs: {exc}") from exc
