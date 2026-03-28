"""Shared Prometheus metrics for the API and monitoring state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from sqlalchemy import text
from sqlalchemy.engine import Engine

HTTP_REQUESTS_TOTAL = Counter(
    "ml_monitoring_http_requests_total",
    "Total number of HTTP requests handled by the API.",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "ml_monitoring_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTIONS_TOTAL = Counter(
    "ml_monitoring_predictions_total",
    "Total number of successful predictions.",
    ["model_version", "predicted_label"],
)

PREDICTION_SCORE = Histogram(
    "ml_monitoring_prediction_score",
    "Distribution of model scores returned by the inference API.",
    ["model_version"],
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

LABELS_UPSERTED_TOTAL = Counter(
    "ml_monitoring_labels_upserted_total",
    "Total number of delayed labels upserted through the API.",
    ["model_version"],
)

DRIFT_LAST_RUN_STATUS = Gauge(
    "ml_monitoring_drift_last_run_status",
    "One-hot status of the latest drift monitoring run.",
    ["model_version", "status"],
)

DRIFT_LAST_RUN_AGE_SECONDS = Gauge(
    "ml_monitoring_drift_last_run_age_seconds",
    "Age in seconds of the latest drift monitoring run.",
    ["model_version"],
)

DRIFT_LAST_RUN_DRIFTED_FEATURES = Gauge(
    "ml_monitoring_drift_last_run_drifted_features",
    "Number of drifted features detected in the latest drift monitoring run.",
    ["model_version"],
)

DRIFT_LAST_RUN_OVERALL = Gauge(
    "ml_monitoring_drift_last_run_overall",
    "Whether the latest drift monitoring run detected overall drift.",
    ["model_version"],
)

QUALITY_LAST_RUN_STATUS = Gauge(
    "ml_monitoring_quality_last_run_status",
    "One-hot status of the latest quality monitoring run.",
    ["model_version", "status"],
)

QUALITY_LAST_RUN_AGE_SECONDS = Gauge(
    "ml_monitoring_quality_last_run_age_seconds",
    "Age in seconds of the latest quality monitoring run.",
    ["model_version"],
)

QUALITY_LAST_RUN_DEGRADED_METRICS = Gauge(
    "ml_monitoring_quality_last_run_degraded_metrics",
    (
        "Number of degraded metrics detected in the latest "
        "quality monitoring run."
    ),
    ["model_version"],
)

QUALITY_LAST_RUN_LABELED_ROWS = Gauge(
    "ml_monitoring_quality_last_run_labeled_rows",
    "Number of labeled rows evaluated in the latest quality monitoring run.",
    ["model_version"],
)

QUALITY_LAST_METRIC_VALUE = Gauge(
    "ml_monitoring_quality_last_metric_value",
    "Metric values from the latest completed quality monitoring run.",
    ["model_version", "metric_name"],
)

QUALITY_LAST_METRIC_BASELINE = Gauge(
    "ml_monitoring_quality_last_metric_baseline",
    "Baseline metric values for the latest completed quality monitoring run.",
    ["model_version", "metric_name"],
)

QUALITY_LAST_METRIC_DELTA = Gauge(
    "ml_monitoring_quality_last_metric_delta",
    "Metric deltas versus baseline for the latest completed quality run.",
    ["model_version", "metric_name"],
)

KNOWN_RUN_STATUSES: tuple[
    Literal["running"],
    Literal["completed"],
    Literal["failed"],
    Literal["skipped_insufficient_data"],
] = ("running", "completed", "failed", "skipped_insufficient_data")


def record_http_request(
    method: str, path: str, status_code: int, duration_seconds: float
) -> None:
    """Record one completed HTTP request."""

    status = str(status_code)
    HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(
        duration_seconds
    )


def record_prediction(
    model_version: str, predicted_label: int, score: float
) -> None:
    """Record one successful prediction result."""

    label = str(int(predicted_label))
    PREDICTIONS_TOTAL.labels(
        model_version=model_version, predicted_label=label
    ).inc()
    PREDICTION_SCORE.labels(model_version=model_version).observe(float(score))


def record_labels_upserted(model_version: str, count: int) -> None:
    """Record delayed labels accepted by the ingestion API."""

    if count <= 0:
        return
    LABELS_UPSERTED_TOTAL.labels(model_version=model_version).inc(count)


def _as_utc_datetime(value: Any) -> datetime | None:
    """Normalize a database timestamp into a timezone-aware UTC datetime."""

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return None


def _seconds_since(value: datetime | None) -> float:
    """Return how many seconds passed since the provided timestamp."""

    if value is None:
        return 0.0
    now = datetime.now(UTC)
    delta = now - value
    return max(delta.total_seconds(), 0.0)


def _refresh_status_gauge(
    gauge: Gauge, model_version: str, status: str | None
) -> None:
    """Write the latest run status as a one-hot gauge vector."""

    gauge.clear()
    active_status = status or "unknown"
    status_values: list[str] = list(KNOWN_RUN_STATUSES)
    if active_status not in status_values:
        status_values.append(active_status)

    for candidate in status_values:
        gauge.labels(model_version=model_version, status=candidate).set(
            1.0 if candidate == active_status else 0.0
        )


def _clear_monitoring_gauges(model_version: str) -> None:
    """Reset derived monitoring gauges when no rows are available yet."""

    DRIFT_LAST_RUN_STATUS.clear()
    DRIFT_LAST_RUN_AGE_SECONDS.clear()
    DRIFT_LAST_RUN_DRIFTED_FEATURES.clear()
    DRIFT_LAST_RUN_OVERALL.clear()
    QUALITY_LAST_RUN_STATUS.clear()
    QUALITY_LAST_RUN_AGE_SECONDS.clear()
    QUALITY_LAST_RUN_DEGRADED_METRICS.clear()
    QUALITY_LAST_RUN_LABELED_ROWS.clear()
    QUALITY_LAST_METRIC_VALUE.clear()
    QUALITY_LAST_METRIC_BASELINE.clear()
    QUALITY_LAST_METRIC_DELTA.clear()

    _refresh_status_gauge(
        DRIFT_LAST_RUN_STATUS, model_version=model_version, status="unknown"
    )
    _refresh_status_gauge(
        QUALITY_LAST_RUN_STATUS, model_version=model_version, status="unknown"
    )
    DRIFT_LAST_RUN_AGE_SECONDS.labels(model_version=model_version).set(0.0)
    DRIFT_LAST_RUN_DRIFTED_FEATURES.labels(model_version=model_version).set(
        0.0
    )
    DRIFT_LAST_RUN_OVERALL.labels(model_version=model_version).set(0.0)
    QUALITY_LAST_RUN_AGE_SECONDS.labels(model_version=model_version).set(0.0)
    QUALITY_LAST_RUN_DEGRADED_METRICS.labels(model_version=model_version).set(
        0.0
    )
    QUALITY_LAST_RUN_LABELED_ROWS.labels(model_version=model_version).set(0.0)


def refresh_monitoring_gauges(engine: Engine, model_version: str) -> None:
    """Refresh gauges derived from the latest monitoring runs."""

    drift_query = text(
        """
        SELECT
            COALESCE(ts_finished, ts_started) AS last_ts,
            status,
            drifted_features_count,
            overall_drift
        FROM monitoring_runs
        ORDER BY ts_started DESC, id DESC
        LIMIT 1
        """
    )

    quality_query = text(
        """
        SELECT
            COALESCE(ts_finished, ts_started) AS last_ts,
            status,
            degraded_metrics_count,
            labeled_rows
        FROM quality_runs
        ORDER BY ts_started DESC, id DESC
        LIMIT 1
        """
    )

    quality_metrics_query = text(
        """
        WITH latest_completed_quality_run AS (
            SELECT id
            FROM quality_runs
            WHERE
                model_version = :model_version
                AND status = 'completed'
            ORDER BY ts_started DESC, id DESC
            LIMIT 1
        )
        SELECT
            metric_name,
            metric_value,
            baseline_value,
            delta_value
        FROM quality_metrics
        WHERE run_id = (SELECT id FROM latest_completed_quality_run)
        """
    )

    with engine.connect() as connection:
        drift_row = connection.execute(drift_query).mappings().first()
        quality_row = connection.execute(quality_query).mappings().first()
        quality_metric_rows = (
            connection.execute(
                quality_metrics_query,
                {"model_version": model_version},
            )
            .mappings()
            .all()
        )

    DRIFT_LAST_RUN_AGE_SECONDS.clear()
    DRIFT_LAST_RUN_DRIFTED_FEATURES.clear()
    DRIFT_LAST_RUN_OVERALL.clear()
    QUALITY_LAST_RUN_AGE_SECONDS.clear()
    QUALITY_LAST_RUN_DEGRADED_METRICS.clear()
    QUALITY_LAST_RUN_LABELED_ROWS.clear()
    QUALITY_LAST_METRIC_VALUE.clear()
    QUALITY_LAST_METRIC_BASELINE.clear()
    QUALITY_LAST_METRIC_DELTA.clear()

    if drift_row is None and quality_row is None:
        _clear_monitoring_gauges(model_version=model_version)
        return

    if drift_row is not None:
        drift_ts = _as_utc_datetime(drift_row["last_ts"])
        _refresh_status_gauge(
            DRIFT_LAST_RUN_STATUS,
            model_version=model_version,
            status=drift_row["status"],
        )
        DRIFT_LAST_RUN_AGE_SECONDS.labels(model_version=model_version).set(
            _seconds_since(drift_ts)
        )
        DRIFT_LAST_RUN_DRIFTED_FEATURES.labels(
            model_version=model_version
        ).set(float(drift_row["drifted_features_count"] or 0))
        DRIFT_LAST_RUN_OVERALL.labels(model_version=model_version).set(
            1.0 if drift_row["overall_drift"] else 0.0
        )
    else:
        _refresh_status_gauge(
            DRIFT_LAST_RUN_STATUS,
            model_version=model_version,
            status="unknown",
        )
        DRIFT_LAST_RUN_AGE_SECONDS.labels(model_version=model_version).set(0.0)
        DRIFT_LAST_RUN_DRIFTED_FEATURES.labels(
            model_version=model_version
        ).set(0.0)
        DRIFT_LAST_RUN_OVERALL.labels(model_version=model_version).set(0.0)

    if quality_row is not None:
        quality_ts = _as_utc_datetime(quality_row["last_ts"])
        _refresh_status_gauge(
            QUALITY_LAST_RUN_STATUS,
            model_version=model_version,
            status=quality_row["status"],
        )
        QUALITY_LAST_RUN_AGE_SECONDS.labels(model_version=model_version).set(
            _seconds_since(quality_ts)
        )
        QUALITY_LAST_RUN_DEGRADED_METRICS.labels(
            model_version=model_version
        ).set(float(quality_row["degraded_metrics_count"] or 0))
        QUALITY_LAST_RUN_LABELED_ROWS.labels(model_version=model_version).set(
            float(quality_row["labeled_rows"] or 0)
        )
    else:
        _refresh_status_gauge(
            QUALITY_LAST_RUN_STATUS,
            model_version=model_version,
            status="unknown",
        )
        QUALITY_LAST_RUN_AGE_SECONDS.labels(model_version=model_version).set(
            0.0
        )
        QUALITY_LAST_RUN_DEGRADED_METRICS.labels(
            model_version=model_version
        ).set(0.0)
        QUALITY_LAST_RUN_LABELED_ROWS.labels(model_version=model_version).set(
            0.0
        )

    for quality_metric_row in quality_metric_rows:
        metric_name = str(quality_metric_row["metric_name"])
        metric_value = quality_metric_row["metric_value"]
        baseline_value = quality_metric_row["baseline_value"]
        delta_value = quality_metric_row["delta_value"]

        if metric_value is not None:
            QUALITY_LAST_METRIC_VALUE.labels(
                model_version=model_version, metric_name=metric_name
            ).set(float(metric_value))

        if baseline_value is not None:
            QUALITY_LAST_METRIC_BASELINE.labels(
                model_version=model_version, metric_name=metric_name
            ).set(float(baseline_value))

        if delta_value is not None:
            QUALITY_LAST_METRIC_DELTA.labels(
                model_version=model_version, metric_name=metric_name
            ).set(float(delta_value))


def render_metrics() -> bytes:
    """Render the Prometheus exposition payload."""

    return generate_latest()


__all__ = [
    "CONTENT_TYPE_LATEST",
    "record_http_request",
    "record_prediction",
    "record_labels_upserted",
    "refresh_monitoring_gauges",
    "render_metrics",
]
