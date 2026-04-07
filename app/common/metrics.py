"""Shared Prometheus metrics for the API and monitoring state."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache
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

from app.common.config import settings

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

ACTIVE_INCIDENTS = Gauge(
    "ml_monitoring_active_incidents",
    "Number of active monitoring incidents grouped by source and severity.",
    ["model_version", "source_type", "severity"],
)

OVERVIEW_SEVERITY = Gauge(
    "ml_monitoring_overview_severity",
    "One-hot overview severity derived from active monitoring incidents.",
    ["model_version", "severity"],
)

DRIFT_SEGMENT_LAST_RUN_STATUS = Gauge(
    "ml_monitoring_drift_segment_last_run_status",
    "One-hot status of the latest drift monitoring run for each segment.",
    ["model_version", "segment_key", "status"],
)

DRIFT_SEGMENT_LAST_RUN_AGE_SECONDS = Gauge(
    "ml_monitoring_drift_segment_last_run_age_seconds",
    "Age in seconds of the latest drift monitoring run for each segment.",
    ["model_version", "segment_key"],
)

DRIFT_SEGMENT_LAST_RUN_DRIFTED_FEATURES = Gauge(
    "ml_monitoring_drift_segment_last_run_drifted_features",
    "Number of drifted features in the latest drift run for each segment.",
    ["model_version", "segment_key"],
)

DRIFT_SEGMENT_LAST_RUN_OVERALL = Gauge(
    "ml_monitoring_drift_segment_last_run_overall",
    "Whether the latest drift run for a segment detected overall drift.",
    ["model_version", "segment_key"],
)

DRIFT_SEGMENT_LAST_RUN_SEVERITY = Gauge(
    "ml_monitoring_drift_segment_last_run_severity",
    "One-hot severity of the latest drift monitoring run for each segment.",
    ["model_version", "segment_key", "severity"],
)

QUALITY_SEGMENT_LAST_RUN_STATUS = Gauge(
    "ml_monitoring_quality_segment_last_run_status",
    "One-hot status of the latest quality monitoring run for each segment.",
    ["model_version", "segment_key", "status"],
)

QUALITY_SEGMENT_LAST_RUN_AGE_SECONDS = Gauge(
    "ml_monitoring_quality_segment_last_run_age_seconds",
    "Age in seconds of the latest quality monitoring run for each segment.",
    ["model_version", "segment_key"],
)

QUALITY_SEGMENT_LAST_RUN_DEGRADED_METRICS = Gauge(
    "ml_monitoring_quality_segment_last_run_degraded_metrics",
    "Number of degraded metrics in the latest quality run for each segment.",
    ["model_version", "segment_key"],
)

QUALITY_SEGMENT_LAST_RUN_LABELED_ROWS = Gauge(
    "ml_monitoring_quality_segment_last_run_labeled_rows",
    "Number of labeled rows in the latest quality run for each segment.",
    ["model_version", "segment_key"],
)

QUALITY_SEGMENT_LAST_RUN_SEVERITY = Gauge(
    "ml_monitoring_quality_segment_last_run_severity",
    "One-hot severity of the latest quality monitoring run for each segment.",
    ["model_version", "segment_key", "severity"],
)

QUALITY_SEGMENT_LAST_METRIC_VALUE = Gauge(
    "ml_monitoring_quality_segment_last_metric_value",
    "Metric values from the latest completed quality run for each segment.",
    ["model_version", "segment_key", "metric_name"],
)

QUALITY_SEGMENT_LAST_METRIC_BASELINE = Gauge(
    "ml_monitoring_quality_segment_last_metric_baseline",
    (
        "Baseline metric values from the latest completed quality run "
        "per segment."
    ),
    ["model_version", "segment_key", "metric_name"],
)

QUALITY_SEGMENT_LAST_METRIC_DELTA = Gauge(
    "ml_monitoring_quality_segment_last_metric_delta",
    (
        "Metric deltas versus baseline for the latest completed quality run "
        "per segment."
    ),
    ["model_version", "segment_key", "metric_name"],
)

ACTIVE_INCIDENTS_BY_SEGMENT = Gauge(
    "ml_monitoring_active_incidents_by_segment",
    (
        "Number of active monitoring incidents grouped by segment, "
        "source, and severity."
    ),
    ["model_version", "segment_key", "source_type", "severity"],
)

DATA_QUALITY_FEATURE_MISSING_RATE = Gauge(
    "ml_monitoring_data_quality_feature_missing_rate",
    "Recent missing-value rate per feature in the inference stream.",
    ["model_version", "feature_name"],
)

DATA_QUALITY_NUMERIC_OUT_OF_RANGE_RATE = Gauge(
    "ml_monitoring_data_quality_numeric_out_of_range_rate",
    "Recent out-of-range rate for numeric features in the inference stream.",
    ["model_version", "feature_name"],
)

DATA_QUALITY_UNKNOWN_CATEGORY_RATE = Gauge(
    "ml_monitoring_data_quality_unknown_category_rate",
    "Recent unexpected-category rate for categorical features.",
    ["model_version", "feature_name"],
)

GLOBAL_SEGMENT_KEY = "__global__"
KNOWN_SEVERITIES: tuple[str, ...] = ("none", "warning", "critical")

KNOWN_RUN_STATUSES: tuple[
    Literal["running"],
    Literal["completed"],
    Literal["completed_proxy"],
    Literal["failed"],
    Literal["skipped_insufficient_data"],
] = (
    "running",
    "completed",
    "completed_proxy",
    "failed",
    "skipped_insufficient_data",
)


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


@lru_cache(maxsize=1)
def _load_baseline_feature_profiles() -> dict[str, dict[str, Any]]:
    """Load feature expectations from the saved baseline profile."""

    baseline_path = settings.baseline_path
    if not baseline_path.exists():
        return {}

    with baseline_path.open("r", encoding="utf-8") as file_obj:
        baseline_profile = json.load(file_obj)

    features = baseline_profile.get("features", {})
    if not isinstance(features, dict):
        return {}

    return {
        str(feature_name): profile
        for feature_name, profile in features.items()
        if isinstance(profile, dict)
    }


def _is_missing_feature_value(value: Any) -> bool:
    """Treat empty strings and null-like values as missing feature inputs."""

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "null", "none"}
    return False


def _to_float(value: Any) -> float | None:
    """Convert arbitrary numeric-like values into floats."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _refresh_data_quality_feature_gauges(
    *,
    model_version: str,
    inference_feature_rows: list[dict[str, Any]],
) -> None:
    """Refresh recent missing/out-of-range/category quality gauges."""

    DATA_QUALITY_FEATURE_MISSING_RATE.clear()
    DATA_QUALITY_NUMERIC_OUT_OF_RANGE_RATE.clear()
    DATA_QUALITY_UNKNOWN_CATEGORY_RATE.clear()

    if not inference_feature_rows:
        return

    feature_profiles = _load_baseline_feature_profiles()
    if not feature_profiles:
        return

    total_rows = float(len(inference_feature_rows))
    missing_counts = {feature_name: 0 for feature_name in feature_profiles}
    out_of_range_counts: dict[str, int] = {}
    unknown_category_counts: dict[str, int] = {}
    expected_categories: dict[str, set[str]] = {}

    for feature_name, profile in feature_profiles.items():
        if profile.get("type") == "numeric":
            out_of_range_counts[feature_name] = 0
        elif profile.get("type") == "categorical":
            unknown_category_counts[feature_name] = 0
            expected_categories[feature_name] = {
                str(category) for category in profile.get("top_values", {})
            }

    for features in inference_feature_rows:
        for feature_name, profile in feature_profiles.items():
            value = features.get(feature_name)
            if _is_missing_feature_value(value):
                missing_counts[feature_name] += 1
                continue

            if profile.get("type") == "numeric":
                numeric_value = _to_float(value)
                if numeric_value is None:
                    out_of_range_counts[feature_name] += 1
                    continue

                min_value = _to_float(profile.get("min"))
                max_value = _to_float(profile.get("max"))
                if (min_value is not None and numeric_value < min_value) or (
                    max_value is not None and numeric_value > max_value
                ):
                    out_of_range_counts[feature_name] += 1
            elif profile.get("type") == "categorical":
                allowed_values = expected_categories.get(feature_name, set())
                if allowed_values and str(value) not in allowed_values:
                    unknown_category_counts[feature_name] += 1

    for feature_name, count in missing_counts.items():
        DATA_QUALITY_FEATURE_MISSING_RATE.labels(
            model_version=model_version,
            feature_name=feature_name,
        ).set(float(count) / total_rows)

    for feature_name, count in out_of_range_counts.items():
        DATA_QUALITY_NUMERIC_OUT_OF_RANGE_RATE.labels(
            model_version=model_version,
            feature_name=feature_name,
        ).set(float(count) / total_rows)

    for feature_name, count in unknown_category_counts.items():
        DATA_QUALITY_UNKNOWN_CATEGORY_RATE.labels(
            model_version=model_version,
            feature_name=feature_name,
        ).set(float(count) / total_rows)


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


def _refresh_segment_status_gauge(
    gauge: Gauge,
    *,
    model_version: str,
    segment_key: str,
    status: str | None,
) -> None:
    """Write a segment-scoped run status as a one-hot gauge vector."""

    active_status = status or "unknown"
    status_values: list[str] = list(KNOWN_RUN_STATUSES)
    if active_status not in status_values:
        status_values.append(active_status)

    for candidate in status_values:
        gauge.labels(
            model_version=model_version,
            segment_key=segment_key,
            status=candidate,
        ).set(1.0 if candidate == active_status else 0.0)


def _refresh_segment_severity_gauge(
    gauge: Gauge,
    *,
    model_version: str,
    segment_key: str,
    severity: str | None,
) -> None:
    """Write a segment-scoped severity as a one-hot gauge vector."""

    active_severity = severity or "none"
    severity_values: list[str] = list(KNOWN_SEVERITIES)
    if active_severity not in severity_values:
        severity_values.append(active_severity)

    for candidate in severity_values:
        gauge.labels(
            model_version=model_version,
            segment_key=segment_key,
            severity=candidate,
        ).set(1.0 if candidate == active_severity else 0.0)


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
    ACTIVE_INCIDENTS.clear()
    OVERVIEW_SEVERITY.clear()
    DRIFT_SEGMENT_LAST_RUN_STATUS.clear()
    DRIFT_SEGMENT_LAST_RUN_AGE_SECONDS.clear()
    DRIFT_SEGMENT_LAST_RUN_DRIFTED_FEATURES.clear()
    DRIFT_SEGMENT_LAST_RUN_OVERALL.clear()
    DRIFT_SEGMENT_LAST_RUN_SEVERITY.clear()
    QUALITY_SEGMENT_LAST_RUN_STATUS.clear()
    QUALITY_SEGMENT_LAST_RUN_AGE_SECONDS.clear()
    QUALITY_SEGMENT_LAST_RUN_DEGRADED_METRICS.clear()
    QUALITY_SEGMENT_LAST_RUN_LABELED_ROWS.clear()
    QUALITY_SEGMENT_LAST_RUN_SEVERITY.clear()
    QUALITY_SEGMENT_LAST_METRIC_VALUE.clear()
    QUALITY_SEGMENT_LAST_METRIC_BASELINE.clear()
    QUALITY_SEGMENT_LAST_METRIC_DELTA.clear()
    ACTIVE_INCIDENTS_BY_SEGMENT.clear()
    DATA_QUALITY_FEATURE_MISSING_RATE.clear()
    DATA_QUALITY_NUMERIC_OUT_OF_RANGE_RATE.clear()
    DATA_QUALITY_UNKNOWN_CATEGORY_RATE.clear()

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
    OVERVIEW_SEVERITY.labels(model_version=model_version, severity="none").set(
        1.0
    )
    OVERVIEW_SEVERITY.labels(
        model_version=model_version, severity="warning"
    ).set(0.0)
    OVERVIEW_SEVERITY.labels(
        model_version=model_version, severity="critical"
    ).set(0.0)
    _refresh_segment_status_gauge(
        DRIFT_SEGMENT_LAST_RUN_STATUS,
        model_version=model_version,
        segment_key=GLOBAL_SEGMENT_KEY,
        status="unknown",
    )
    _refresh_segment_status_gauge(
        QUALITY_SEGMENT_LAST_RUN_STATUS,
        model_version=model_version,
        segment_key=GLOBAL_SEGMENT_KEY,
        status="unknown",
    )
    DRIFT_SEGMENT_LAST_RUN_AGE_SECONDS.labels(
        model_version=model_version,
        segment_key=GLOBAL_SEGMENT_KEY,
    ).set(0.0)
    DRIFT_SEGMENT_LAST_RUN_DRIFTED_FEATURES.labels(
        model_version=model_version,
        segment_key=GLOBAL_SEGMENT_KEY,
    ).set(0.0)
    DRIFT_SEGMENT_LAST_RUN_OVERALL.labels(
        model_version=model_version,
        segment_key=GLOBAL_SEGMENT_KEY,
    ).set(0.0)
    _refresh_segment_severity_gauge(
        DRIFT_SEGMENT_LAST_RUN_SEVERITY,
        model_version=model_version,
        segment_key=GLOBAL_SEGMENT_KEY,
        severity="none",
    )
    QUALITY_SEGMENT_LAST_RUN_AGE_SECONDS.labels(
        model_version=model_version,
        segment_key=GLOBAL_SEGMENT_KEY,
    ).set(0.0)
    QUALITY_SEGMENT_LAST_RUN_DEGRADED_METRICS.labels(
        model_version=model_version,
        segment_key=GLOBAL_SEGMENT_KEY,
    ).set(0.0)
    QUALITY_SEGMENT_LAST_RUN_LABELED_ROWS.labels(
        model_version=model_version,
        segment_key=GLOBAL_SEGMENT_KEY,
    ).set(0.0)
    _refresh_segment_severity_gauge(
        QUALITY_SEGMENT_LAST_RUN_SEVERITY,
        model_version=model_version,
        segment_key=GLOBAL_SEGMENT_KEY,
        severity="none",
    )


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
        WHERE model_version = :model_version
        ORDER BY ts_started DESC, id DESC
        LIMIT 1
        """
    )

    drift_segment_query = text(
        """
        WITH ranked_drift_runs AS (
            SELECT
                COALESCE(segment_key, :global_segment_key) AS segment_key,
                COALESCE(ts_finished, ts_started) AS last_ts,
                status,
                drifted_features_count,
                overall_drift,
                COALESCE(
                    summary_json ->> 'severity',
                    CASE
                        WHEN overall_drift THEN 'warning'
                        ELSE 'none'
                    END
                ) AS severity,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(segment_key, :global_segment_key)
                    ORDER BY ts_started DESC, id DESC
                ) AS rn
            FROM monitoring_runs
            WHERE model_version = :model_version
        )
        SELECT
            segment_key,
            last_ts,
            status,
            drifted_features_count,
            overall_drift,
            severity
        FROM ranked_drift_runs
        WHERE rn = 1
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
        WHERE model_version = :model_version
        ORDER BY ts_started DESC, id DESC
        LIMIT 1
        """
    )

    quality_segment_query = text(
        """
        WITH ranked_quality_runs AS (
            SELECT
                COALESCE(segment_key, :global_segment_key) AS segment_key,
                COALESCE(ts_finished, ts_started) AS last_ts,
                status,
                degraded_metrics_count,
                labeled_rows,
                COALESCE(
                    summary_json ->> 'severity',
                    CASE
                        WHEN degraded_metrics_count > 0 THEN 'warning'
                        ELSE 'none'
                    END
                ) AS severity,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(segment_key, :global_segment_key)
                    ORDER BY ts_started DESC, id DESC
                ) AS rn
            FROM quality_runs
            WHERE model_version = :model_version
        )
        SELECT
            segment_key,
            last_ts,
            status,
            degraded_metrics_count,
            labeled_rows,
            severity
        FROM ranked_quality_runs
        WHERE rn = 1
        """
    )

    quality_metrics_query = text(
        """
        WITH latest_completed_quality_run AS (
            SELECT id
            FROM quality_runs
            WHERE
                model_version = :model_version
                AND status IN ('completed', 'completed_proxy')
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

    quality_segment_metrics_query = text(
        """
        WITH ranked_quality_runs AS (
            SELECT
                id,
                COALESCE(segment_key, :global_segment_key) AS segment_key,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(segment_key, :global_segment_key)
                    ORDER BY ts_started DESC, id DESC
                ) AS rn
            FROM quality_runs
            WHERE
                model_version = :model_version
                AND status IN ('completed', 'completed_proxy')
        )
        SELECT
            ranked_quality_runs.segment_key,
            quality_metrics.metric_name,
            quality_metrics.metric_value,
            quality_metrics.baseline_value,
            quality_metrics.delta_value
        FROM ranked_quality_runs
        JOIN quality_metrics
            ON quality_metrics.run_id = ranked_quality_runs.id
        WHERE ranked_quality_runs.rn = 1
        """
    )

    incident_counts_query = text(
        """
        SELECT
            source_type,
            severity,
            COUNT(*) AS total
        FROM monitoring_incidents
        WHERE status = 'open' AND model_version = :model_version
        GROUP BY source_type, severity
        """
    )

    incident_counts_segment_query = text(
        """
        SELECT
            COALESCE(segment_key, :global_segment_key) AS segment_key,
            source_type,
            severity,
            COUNT(*) AS total
        FROM monitoring_incidents
        WHERE status = 'open' AND model_version = :model_version
        GROUP BY
            COALESCE(segment_key, :global_segment_key),
            source_type,
            severity
        """
    )

    recent_inference_features_query = text(
        """
        SELECT features_json
        FROM inference_log
        WHERE
            model_version = :model_version
            AND ts >= NOW() - INTERVAL '24 hours'
        ORDER BY ts DESC, id DESC
        LIMIT 2000
        """
    )

    with engine.connect() as connection:
        query_params = {
            "model_version": model_version,
            "global_segment_key": GLOBAL_SEGMENT_KEY,
        }
        drift_row = (
            connection.execute(drift_query, {"model_version": model_version})
            .mappings()
            .first()
        )
        quality_row = (
            connection.execute(quality_query, {"model_version": model_version})
            .mappings()
            .first()
        )
        drift_segment_rows = (
            connection.execute(drift_segment_query, query_params)
            .mappings()
            .all()
        )
        quality_segment_rows = (
            connection.execute(quality_segment_query, query_params)
            .mappings()
            .all()
        )
        quality_metric_rows = (
            connection.execute(
                quality_metrics_query,
                {"model_version": model_version},
            )
            .mappings()
            .all()
        )
        quality_segment_metric_rows = (
            connection.execute(quality_segment_metrics_query, query_params)
            .mappings()
            .all()
        )
        incident_rows = (
            connection.execute(
                incident_counts_query, {"model_version": model_version}
            )
            .mappings()
            .all()
        )
        incident_segment_rows = (
            connection.execute(incident_counts_segment_query, query_params)
            .mappings()
            .all()
        )
        recent_inference_rows = (
            connection.execute(
                recent_inference_features_query,
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
    ACTIVE_INCIDENTS.clear()
    OVERVIEW_SEVERITY.clear()
    DRIFT_SEGMENT_LAST_RUN_STATUS.clear()
    DRIFT_SEGMENT_LAST_RUN_AGE_SECONDS.clear()
    DRIFT_SEGMENT_LAST_RUN_DRIFTED_FEATURES.clear()
    DRIFT_SEGMENT_LAST_RUN_OVERALL.clear()
    DRIFT_SEGMENT_LAST_RUN_SEVERITY.clear()
    QUALITY_SEGMENT_LAST_RUN_STATUS.clear()
    QUALITY_SEGMENT_LAST_RUN_AGE_SECONDS.clear()
    QUALITY_SEGMENT_LAST_RUN_DEGRADED_METRICS.clear()
    QUALITY_SEGMENT_LAST_RUN_LABELED_ROWS.clear()
    QUALITY_SEGMENT_LAST_RUN_SEVERITY.clear()
    QUALITY_SEGMENT_LAST_METRIC_VALUE.clear()
    QUALITY_SEGMENT_LAST_METRIC_BASELINE.clear()
    QUALITY_SEGMENT_LAST_METRIC_DELTA.clear()
    ACTIVE_INCIDENTS_BY_SEGMENT.clear()
    DATA_QUALITY_FEATURE_MISSING_RATE.clear()
    DATA_QUALITY_NUMERIC_OUT_OF_RANGE_RATE.clear()
    DATA_QUALITY_UNKNOWN_CATEGORY_RATE.clear()

    inference_feature_rows = [
        row["features_json"]
        for row in recent_inference_rows
        if isinstance(row.get("features_json"), dict)
    ]

    no_monitoring_runs = drift_row is None and quality_row is None

    if no_monitoring_runs:
        _clear_monitoring_gauges(model_version=model_version)
        _refresh_data_quality_feature_gauges(
            model_version=model_version,
            inference_feature_rows=inference_feature_rows,
        )
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

    for drift_segment_row in drift_segment_rows:
        segment_key = str(
            drift_segment_row["segment_key"] or GLOBAL_SEGMENT_KEY
        )
        drift_ts = _as_utc_datetime(drift_segment_row["last_ts"])
        _refresh_segment_status_gauge(
            DRIFT_SEGMENT_LAST_RUN_STATUS,
            model_version=model_version,
            segment_key=segment_key,
            status=drift_segment_row["status"],
        )
        DRIFT_SEGMENT_LAST_RUN_AGE_SECONDS.labels(
            model_version=model_version,
            segment_key=segment_key,
        ).set(_seconds_since(drift_ts))
        DRIFT_SEGMENT_LAST_RUN_DRIFTED_FEATURES.labels(
            model_version=model_version,
            segment_key=segment_key,
        ).set(float(drift_segment_row["drifted_features_count"] or 0))
        DRIFT_SEGMENT_LAST_RUN_OVERALL.labels(
            model_version=model_version,
            segment_key=segment_key,
        ).set(1.0 if drift_segment_row["overall_drift"] else 0.0)
        _refresh_segment_severity_gauge(
            DRIFT_SEGMENT_LAST_RUN_SEVERITY,
            model_version=model_version,
            segment_key=segment_key,
            severity=str(drift_segment_row["severity"] or "none"),
        )

    for quality_segment_row in quality_segment_rows:
        segment_key = str(
            quality_segment_row["segment_key"] or GLOBAL_SEGMENT_KEY
        )
        quality_ts = _as_utc_datetime(quality_segment_row["last_ts"])
        _refresh_segment_status_gauge(
            QUALITY_SEGMENT_LAST_RUN_STATUS,
            model_version=model_version,
            segment_key=segment_key,
            status=quality_segment_row["status"],
        )
        QUALITY_SEGMENT_LAST_RUN_AGE_SECONDS.labels(
            model_version=model_version,
            segment_key=segment_key,
        ).set(_seconds_since(quality_ts))
        QUALITY_SEGMENT_LAST_RUN_DEGRADED_METRICS.labels(
            model_version=model_version,
            segment_key=segment_key,
        ).set(float(quality_segment_row["degraded_metrics_count"] or 0))
        QUALITY_SEGMENT_LAST_RUN_LABELED_ROWS.labels(
            model_version=model_version,
            segment_key=segment_key,
        ).set(float(quality_segment_row["labeled_rows"] or 0))
        _refresh_segment_severity_gauge(
            QUALITY_SEGMENT_LAST_RUN_SEVERITY,
            model_version=model_version,
            segment_key=segment_key,
            severity=str(quality_segment_row["severity"] or "none"),
        )

    for quality_segment_metric_row in quality_segment_metric_rows:
        segment_key = str(
            quality_segment_metric_row["segment_key"] or GLOBAL_SEGMENT_KEY
        )
        metric_name = str(quality_segment_metric_row["metric_name"])
        metric_value = quality_segment_metric_row["metric_value"]
        baseline_value = quality_segment_metric_row["baseline_value"]
        delta_value = quality_segment_metric_row["delta_value"]

        if metric_value is not None:
            QUALITY_SEGMENT_LAST_METRIC_VALUE.labels(
                model_version=model_version,
                segment_key=segment_key,
                metric_name=metric_name,
            ).set(float(metric_value))

        if baseline_value is not None:
            QUALITY_SEGMENT_LAST_METRIC_BASELINE.labels(
                model_version=model_version,
                segment_key=segment_key,
                metric_name=metric_name,
            ).set(float(baseline_value))

        if delta_value is not None:
            QUALITY_SEGMENT_LAST_METRIC_DELTA.labels(
                model_version=model_version,
                segment_key=segment_key,
                metric_name=metric_name,
            ).set(float(delta_value))

    overview_severity = "none"
    for incident_row in incident_rows:
        source_type = str(incident_row["source_type"])
        severity = str(incident_row["severity"])
        total = float(incident_row["total"] or 0)
        ACTIVE_INCIDENTS.labels(
            model_version=model_version,
            source_type=source_type,
            severity=severity,
        ).set(total)
        if severity == "critical":
            overview_severity = "critical"
        elif severity == "warning" and overview_severity != "critical":
            overview_severity = "warning"

    for incident_segment_row in incident_segment_rows:
        ACTIVE_INCIDENTS_BY_SEGMENT.labels(
            model_version=model_version,
            segment_key=str(
                incident_segment_row["segment_key"] or GLOBAL_SEGMENT_KEY
            ),
            source_type=str(incident_segment_row["source_type"]),
            severity=str(incident_segment_row["severity"]),
        ).set(float(incident_segment_row["total"] or 0))

    for severity in KNOWN_SEVERITIES:
        OVERVIEW_SEVERITY.labels(
            model_version=model_version,
            severity=severity,
        ).set(1.0 if severity == overview_severity else 0.0)

    _refresh_data_quality_feature_gauges(
        model_version=model_version,
        inference_feature_rows=inference_feature_rows,
    )


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
