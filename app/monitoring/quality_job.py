from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.common.config import settings
from app.common.logging import get_logger, setup_logging
from app.monitoring.drift_job import (
    calculate_numeric_psi,
    load_current_window,
    load_model,
    load_reference_data,
)
from app.monitoring.incidents import (
    build_incident_key,
    highest_severity,
    sync_monitoring_incident,
)
from app.monitoring.reaction_engine import maybe_execute_critical_reaction
from app.monitoring.unlabeled_quality import (
    estimate_unlabeled_quality_with_bbse,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = settings.baseline_path

DEGRADATION_RULES: dict[str, dict[str, Any]] = {
    "roc_auc": {
        "mode": "min_delta",
        "threshold": 0.02,
        "detector_name": "labeled",
    },
    "pr_auc": {
        "mode": "min_delta",
        "threshold": 0.04,
        "detector_name": "labeled",
    },
    "precision": {
        "mode": "min_delta",
        "threshold": 0.04,
        "detector_name": "labeled",
    },
    "recall": {
        "mode": "min_delta",
        "threshold": 0.03,
        "detector_name": "labeled",
    },
    "f1": {"mode": "min_delta", "threshold": 0.04, "detector_name": "labeled"},
    "brier_score": {
        "mode": "max_delta",
        "threshold": 0.02,
        "detector_name": "labeled",
    },
    "ece": {
        "mode": "max_delta",
        "threshold": 0.03,
        "detector_name": "calibration",
    },
    "positive_rate_pred": {
        "mode": "abs_delta",
        "threshold": 0.08,
        "detector_name": "proxy",
    },
    "positive_rate_true": {
        "mode": "abs_delta",
        "threshold": 0.05,
        "detector_name": "labeled",
    },
    "score_mean": {
        "mode": "abs_delta",
        "threshold": 0.08,
        "detector_name": "proxy",
    },
    "score_std": {
        "mode": "abs_delta",
        "threshold": 0.06,
        "detector_name": "proxy",
    },
    "score_entropy": {
        "mode": "max_delta",
        "threshold": 0.05,
        "detector_name": "proxy",
    },
    "near_threshold_rate": {
        "mode": "abs_delta",
        "threshold": 0.10,
        "detector_name": "proxy",
    },
    "score_psi": {
        "mode": "max_absolute",
        "threshold": 0.20,
        "detector_name": "proxy",
    },
}

logger = get_logger(__name__)


def to_native(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-serializable Python objects."""

    if isinstance(value, dict):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def safe_json(obj: Any) -> str:
    """Serialize Python objects into JSON for PostgreSQL JSONB columns."""

    return json.dumps(to_native(obj), ensure_ascii=False, default=str)


def load_baseline_profile() -> dict[str, Any]:
    """Load the saved baseline profile produced during training."""

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline profile not found: {BASELINE_PATH}")

    with BASELINE_PATH.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def get_baseline_metrics(
    baseline_profile: dict[str, Any], baseline_source: str
) -> dict[str, float]:
    """Return reference metrics from the chosen baseline split."""

    metrics_key = f"{baseline_source}_metrics"
    metrics = baseline_profile.get(metrics_key)
    if not metrics:
        raise ValueError(
            "Baseline profile does not contain metrics for source: "
            f"{baseline_source}"
        )
    return metrics


def get_baseline_proxy_metrics(
    baseline_profile: dict[str, Any], baseline_source: str
) -> dict[str, float]:
    """Return reference proxy metrics from the chosen baseline split."""

    metrics_key = f"{baseline_source}_proxy_metrics"
    metrics = baseline_profile.get(metrics_key)
    if not metrics:
        return {}
    return metrics


def compute_expected_calibration_error(
    y_true: pd.Series,
    y_score: pd.Series,
    bins: int = 10,
) -> float:
    """Compute a simple expected calibration error over equal-width bins."""

    y_true_array = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    y_score_array = np.clip(
        pd.to_numeric(y_score, errors="coerce").fillna(0.0).astype(float),
        0.0,
        1.0,
    )
    if len(y_score_array) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        left = bin_edges[index]
        right = bin_edges[index + 1]
        if index == bins - 1:
            mask = (y_score_array >= left) & (y_score_array <= right)
        else:
            mask = (y_score_array >= left) & (y_score_array < right)
        if not np.any(mask):
            continue

        bin_scores = y_score_array[mask]
        bin_true = y_true_array[mask]
        ece += abs(float(bin_scores.mean()) - float(bin_true.mean())) * (
            len(bin_scores) / len(y_score_array)
        )

    return float(ece)


def compute_proxy_metrics(
    score_series: pd.Series,
    threshold: float,
    threshold_band: float = 0.10,
) -> dict[str, float]:
    """Compute quality proxies for unlabeled windows."""

    scores = np.clip(
        pd.to_numeric(score_series, errors="coerce").fillna(0.0).astype(float),
        0.0,
        1.0,
    )
    entropy = -(
        scores * np.log(np.clip(scores, 1e-6, 1.0))
        + (1.0 - scores) * np.log(np.clip(1.0 - scores, 1e-6, 1.0))
    )
    near_threshold_rate = np.mean(np.abs(scores - threshold) <= threshold_band)

    return {
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_entropy": float(entropy.mean()),
        "near_threshold_rate": float(near_threshold_rate),
        "positive_rate_pred": float(np.mean(scores >= threshold)),
    }


def build_quality_recommended_action(
    metric_name: str,
    detector_name: str,
    severity: str,
) -> str:
    """Return a concise response action for one quality signal."""

    if severity == "none":
        return "No action required."
    if detector_name == "proxy":
        return (
            "Collect delayed labels for the affected segment, inspect score "
            "drift, and confirm the next monitoring windows."
        )
    if detector_name == "calibration":
        return (
            "Inspect probability calibration, validate the current threshold, "
            "and prepare recalibration before retraining."
        )
    return (
        "Investigate recent labeled outcomes, validate the segment, and "
        "prepare rollback or retraining if degradation persists."
    )


def classify_quality_severity(
    *,
    degraded: bool,
    effect_size: float | None,
    threshold: float | None,
) -> str:
    """Convert one metric deviation into a severity label."""

    if not degraded:
        return "none"
    if effect_size is None or threshold is None:
        return "warning"
    if effect_size >= threshold * 2.0:
        return "critical"
    return "warning"


def load_labeled_window(
    engine: Engine, window_size: int, segment_key: str | None = None
) -> pd.DataFrame:
    """Load the latest labeled prediction rows from PostgreSQL."""

    if segment_key:
        query = text(
            """
            SELECT
                il.request_id,
                il.ts AS inference_ts,
                gt.label_ts,
                il.segment_key,
                il.score,
                il.pred_label,
                il.threshold,
                gt.y_true
            FROM ground_truth gt
            JOIN inference_log il
                ON il.request_id = gt.request_id
            WHERE il.segment_key = :segment_key
            ORDER BY gt.label_ts DESC, il.ts DESC
            LIMIT :window_size
            """
        )
        params = {"segment_key": segment_key, "window_size": window_size}
    else:
        query = text(
            """
            SELECT
                il.request_id,
                il.ts AS inference_ts,
                gt.label_ts,
                il.segment_key,
                il.score,
                il.pred_label,
                il.threshold,
                gt.y_true
            FROM ground_truth gt
            JOIN inference_log il
                ON il.request_id = gt.request_id
            ORDER BY gt.label_ts DESC, il.ts DESC
            LIMIT :window_size
            """
        )
        params = {"window_size": window_size}

    with engine.connect() as connection:
        rows = connection.execute(query, params).mappings().all()

    return pd.DataFrame(rows)


def compute_quality_metrics(
    window_df: pd.DataFrame,
) -> dict[str, float | None]:
    """Compute online quality metrics over a labeled prediction window."""

    y_true = (
        pd.to_numeric(window_df["y_true"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    y_pred = (
        pd.to_numeric(window_df["pred_label"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    y_score = (
        pd.to_numeric(window_df["score"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

    metrics: dict[str, float | None] = {
        "precision": float(
            precision_score(y_true, y_pred, zero_division=cast(Any, 0))
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=cast(Any, 0))
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=cast(Any, 0))),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "ece": float(compute_expected_calibration_error(y_true, y_score)),
        "positive_rate_pred": float(y_pred.mean()),
        "positive_rate_true": float(y_true.mean()),
    }

    if y_true.nunique() >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def compute_proxy_quality_metrics(
    current_window_df: pd.DataFrame,
    reference_scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
    """Compute proxy metrics for blind periods without labels."""

    current_scores = pd.to_numeric(
        current_window_df["__score"], errors="coerce"
    ).fillna(0.0)
    metrics: dict[str, float | None] = {}
    for key, value in compute_proxy_metrics(current_scores, threshold).items():
        metrics[key] = float(value)

    reference_score_series = pd.Series(
        np.asarray(reference_scores, dtype=float)
    )
    score_psi = calculate_numeric_psi(reference_score_series, current_scores)
    metrics["score_psi"] = float(score_psi) if score_psi is not None else None

    metric_details: dict[str, dict[str, Any]] = {}
    if len(reference_score_series) >= 10 and len(current_scores) >= 10:
        ks_result = cast(
            Any,
            ks_2samp(
                reference_score_series.to_numpy(dtype=float),
                current_scores.to_numpy(dtype=float),
            ),
        )
        metric_details["score_psi"] = {
            "score_ks_statistic": float(ks_result.statistic),
            "score_ks_pvalue": float(ks_result.pvalue),
        }
    else:
        metric_details["score_psi"] = {"status": "insufficient_score_window"}

    return metrics, metric_details


def detect_metric_degradation(
    metric_name: str,
    metric_value: float | None,
    baseline_value: float | None,
) -> tuple[bool, float | None, float | None, dict[str, Any]]:
    """Compare a live metric with its baseline counterpart."""

    rule = DEGRADATION_RULES.get(metric_name)
    if metric_value is None or rule is None:
        return False, None, None, {"status": "comparison_not_available"}

    mode = str(rule["mode"])
    threshold = float(rule["threshold"])
    detector_name = str(rule["detector_name"])

    if baseline_value is None and mode not in {"max_absolute", "min_absolute"}:
        return (
            False,
            None,
            None,
            {
                "status": "comparison_not_available",
                "mode": mode,
                "threshold": threshold,
                "detector_name": detector_name,
            },
        )

    delta_value = (
        None
        if baseline_value is None
        else float(metric_value) - float(baseline_value)
    )

    if mode == "min_delta":
        assert delta_value is not None
        degraded = delta_value <= -threshold
        effect_size = abs(delta_value)
    elif mode == "max_delta":
        assert delta_value is not None
        degraded = delta_value >= threshold
        effect_size = abs(delta_value)
    elif mode == "abs_delta":
        assert delta_value is not None
        degraded = abs(delta_value) >= threshold
        effect_size = abs(delta_value)
    elif mode == "max_absolute":
        degraded = float(metric_value) >= threshold
        effect_size = float(metric_value)
    elif mode == "min_absolute":
        degraded = float(metric_value) <= threshold
        effect_size = abs(float(metric_value))
    else:
        raise ValueError(f"Unsupported degradation mode: {mode}")

    return (
        degraded,
        delta_value,
        effect_size,
        {
            "mode": mode,
            "threshold": threshold,
            "detector_name": detector_name,
        },
    )


def insert_quality_run(
    engine: Engine, window_size: int, segment_key: str | None
) -> int:
    """Insert a quality monitoring run placeholder row and return its id."""

    query = text(
        """
        INSERT INTO quality_runs (
            model_version,
            window_size,
            segment_key,
            status
        )
        VALUES (
            :model_version,
            :window_size,
            :segment_key,
            'running'
        )
        RETURNING id
        """
    )

    with engine.begin() as connection:
        run_id = connection.execute(
            query,
            {
                "model_version": settings.model_version,
                "window_size": window_size,
                "segment_key": segment_key,
            },
        ).scalar_one()

    return int(run_id)


def finalize_quality_run(
    engine: Engine,
    run_id: int,
    status: str,
    labeled_rows: int,
    degraded_metrics_count: int,
    summary: dict[str, Any],
) -> None:
    """Finalize a quality monitoring run with counters and summary payload."""

    query = text(
        """
        UPDATE quality_runs
        SET
            ts_finished = NOW(),
            status = :status,
            labeled_rows = :labeled_rows,
            degraded_metrics_count = :degraded_metrics_count,
            summary_json = CAST(:summary_json AS JSONB)
        WHERE id = :run_id
        """
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "run_id": run_id,
                "status": status,
                "labeled_rows": labeled_rows,
                "degraded_metrics_count": degraded_metrics_count,
                "summary_json": safe_json(summary),
            },
        )


def insert_quality_metrics(
    engine: Engine,
    run_id: int,
    segment_key: str | None,
    metrics_rows: list[dict[str, Any]],
) -> None:
    """Persist per-metric quality results for one run."""

    if not metrics_rows:
        return

    query = text(
        """
        INSERT INTO quality_metrics (
            run_id,
            segment_key,
            metric_name,
            metric_value,
            baseline_value,
            delta_value,
            detector_name,
            effect_size,
            pvalue_adj,
            severity,
            recommended_action,
            degradation_detected,
            details_json
        )
        VALUES (
            :run_id,
            :segment_key,
            :metric_name,
            :metric_value,
            :baseline_value,
            :delta_value,
            :detector_name,
            :effect_size,
            :pvalue_adj,
            :severity,
            :recommended_action,
            :degradation_detected,
            CAST(:details_json AS JSONB)
        )
        """
    )

    payloads = [
        {
            "run_id": run_id,
            "segment_key": segment_key,
            "metric_name": item["metric_name"],
            "metric_value": item["metric_value"],
            "baseline_value": item["baseline_value"],
            "delta_value": item["delta_value"],
            "detector_name": item["detector_name"],
            "effect_size": item["effect_size"],
            "pvalue_adj": item["pvalue_adj"],
            "severity": item["severity"],
            "recommended_action": item["recommended_action"],
            "degradation_detected": item["degradation_detected"],
            "details_json": safe_json(item["details"]),
        }
        for item in metrics_rows
    ]

    with engine.begin() as connection:
        connection.execute(query, payloads)


def insert_quality_estimates(
    engine: Engine,
    run_id: int,
    segment_key: str | None,
    estimate_rows: list[dict[str, Any]],
) -> None:
    """Persist assumption-based unlabeled quality estimates for one run."""

    if not estimate_rows:
        return

    query = text(
        """
        INSERT INTO quality_estimates (
            run_id,
            segment_key,
            estimated_positive_rate,
            estimated_metric_name,
            estimated_metric_value,
            assumption_type,
            quality_estimate_uncertainty,
            confidence_interval_json,
            details_json
        )
        VALUES (
            :run_id,
            :segment_key,
            :estimated_positive_rate,
            :estimated_metric_name,
            :estimated_metric_value,
            :assumption_type,
            :quality_estimate_uncertainty,
            CAST(:confidence_interval_json AS JSONB),
            CAST(:details_json AS JSONB)
        )
        """
    )

    payloads = [
        {
            "run_id": run_id,
            "segment_key": segment_key,
            "estimated_positive_rate": item["estimated_positive_rate"],
            "estimated_metric_name": item["estimated_metric_name"],
            "estimated_metric_value": item["estimated_metric_value"],
            "assumption_type": item["assumption_type"],
            "quality_estimate_uncertainty": item[
                "quality_estimate_uncertainty"
            ],
            "confidence_interval_json": safe_json(item["confidence_interval"]),
            "details_json": safe_json(item["details"]),
        }
        for item in estimate_rows
    ]

    with engine.begin() as connection:
        connection.execute(query, payloads)


def build_metric_rows(
    current_metrics: dict[str, float | None],
    baseline_metrics: dict[str, float],
    sample_rows: int,
    baseline_source: str,
    evaluation_mode: str,
    metric_details: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build metric comparison rows for insertion into quality_metrics."""

    rows: list[dict[str, Any]] = []
    metric_details = metric_details or {}

    for metric_name, metric_value in current_metrics.items():
        baseline_value = baseline_metrics.get(metric_name)
        degraded, delta_value, effect_size, comparison_details = (
            detect_metric_degradation(
                metric_name, metric_value, baseline_value
            )
        )
        threshold = cast(float | None, comparison_details.get("threshold"))
        detector_name = str(comparison_details.get("detector_name", "unknown"))
        severity = classify_quality_severity(
            degraded=degraded, effect_size=effect_size, threshold=threshold
        )

        rows.append(
            {
                "metric_name": metric_name,
                "metric_value": metric_value,
                "baseline_value": baseline_value,
                "delta_value": delta_value,
                "detector_name": detector_name,
                "effect_size": effect_size,
                "pvalue_adj": None,
                "severity": severity,
                "recommended_action": build_quality_recommended_action(
                    metric_name, detector_name, severity
                ),
                "degradation_detected": degraded,
                "details": {
                    "baseline_source": baseline_source,
                    "sample_rows": sample_rows,
                    "evaluation_mode": evaluation_mode,
                    **comparison_details,
                    **metric_details.get(metric_name, {}),
                },
            }
        )

    return rows


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the quality-monitoring job."""

    parser = argparse.ArgumentParser(description="Quality monitoring job")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--segment-key", type=str, default=None)
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument(
        "--baseline-source",
        type=str,
        choices=["test", "validation"],
        default="test",
    )
    return parser


def run_quality_job(
    window_size: int,
    min_rows: int,
    baseline_source: str = "test",
    segment_key: str | None = None,
) -> dict[str, Any]:
    """Execute one batch quality-monitoring run."""

    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")
    if min_rows <= 0:
        raise ValueError("min_rows must be greater than 0")

    logger.info(
        (
            "Starting quality job. model_version=%s window_size=%s "
            "segment_key=%s baseline_source=%s"
        ),
        settings.model_version,
        window_size,
        segment_key,
        baseline_source,
    )

    baseline_profile = load_baseline_profile()
    baseline_metrics = get_baseline_metrics(
        baseline_profile, baseline_source=baseline_source
    )
    baseline_proxy_metrics = get_baseline_proxy_metrics(
        baseline_profile, baseline_source=baseline_source
    )
    feature_columns = cast(
        list[str] | None, baseline_profile.get("feature_columns")
    )
    threshold = float(baseline_profile.get("threshold", 0.5))
    engine = create_engine(
        settings.database_url, future=True, pool_pre_ping=True
    )
    run_id = insert_quality_run(
        engine, window_size=window_size, segment_key=segment_key
    )

    try:
        labeled_window_df = load_labeled_window(
            engine, window_size=window_size, segment_key=segment_key
        )
        labeled_rows = int(len(labeled_window_df))
        unlabeled_quality_estimates: list[dict[str, Any]] = []
        unlabeled_quality_estimation_error: str | None = None

        if not labeled_window_df.empty and len(labeled_window_df) >= min_rows:
            evaluation_mode = "labeled"
            run_status = "completed"
            sample_rows = labeled_rows
            current_metrics = compute_quality_metrics(labeled_window_df)
            metrics_rows = build_metric_rows(
                current_metrics=current_metrics,
                baseline_metrics=baseline_metrics,
                sample_rows=sample_rows,
                baseline_source=baseline_source,
                evaluation_mode=evaluation_mode,
            )
        else:
            current_window_df = load_current_window(
                engine, window_size=window_size, segment_key=segment_key
            )
            if current_window_df.empty or len(current_window_df) < min_rows:
                summary = {
                    "status": "skipped_insufficient_data",
                    "labeled_rows": labeled_rows,
                    "required_min_rows": int(min_rows),
                    "segment_key": segment_key,
                    "baseline_source": baseline_source,
                }
                finalize_quality_run(
                    engine=engine,
                    run_id=run_id,
                    status="skipped_insufficient_data",
                    labeled_rows=labeled_rows,
                    degraded_metrics_count=0,
                    summary=summary,
                )
                logger.info(
                    "Quality job skipped due to insufficient rows: %s",
                    summary,
                )
                return {
                    "run_id": run_id,
                    "status": "skipped_insufficient_data",
                    "summary": summary,
                }

            evaluation_mode = "proxy"
            run_status = "completed_proxy"
            sample_rows = int(len(current_window_df))
            reference_df = load_reference_data()
            model = load_model()
            reference_scores = model.predict_proba(reference_df)[:, 1]
            current_metrics, metric_details = compute_proxy_quality_metrics(
                current_window_df=current_window_df,
                reference_scores=reference_scores,
                threshold=threshold,
            )
            try:
                unlabeled_quality_estimates = (
                    estimate_unlabeled_quality_with_bbse(
                        current_window_df=current_window_df,
                        model=model,
                        threshold=threshold,
                        baseline_source=baseline_source,
                        feature_columns=feature_columns,
                        baseline_metrics=baseline_metrics,
                    )
                )
            except Exception as exc:
                unlabeled_quality_estimation_error = str(exc)
                logger.warning(
                    (
                        "Failed to compute unlabeled quality estimate. "
                        "run_id=%s segment_key=%s baseline_source=%s error=%s"
                    ),
                    run_id,
                    segment_key,
                    baseline_source,
                    exc,
                    exc_info=True,
                )
            metrics_rows = build_metric_rows(
                current_metrics=current_metrics,
                baseline_metrics=baseline_proxy_metrics,
                sample_rows=sample_rows,
                baseline_source=baseline_source,
                evaluation_mode=evaluation_mode,
                metric_details=metric_details,
            )

        insert_quality_metrics(
            engine,
            run_id=run_id,
            segment_key=segment_key,
            metrics_rows=metrics_rows,
        )
        insert_quality_estimates(
            engine,
            run_id=run_id,
            segment_key=segment_key,
            estimate_rows=unlabeled_quality_estimates,
        )

        degraded_metrics = [
            item for item in metrics_rows if item["degradation_detected"]
        ]
        degraded_metric_names = [
            item["metric_name"] for item in degraded_metrics
        ]
        overall_severity = highest_severity(
            [str(item["severity"]) for item in metrics_rows]
        )
        recommended_action = next(
            (
                str(item["recommended_action"])
                for item in metrics_rows
                if item["severity"] in {"critical", "warning"}
            ),
            "No action required.",
        )

        summary = {
            "segment_key": segment_key,
            "baseline_source": baseline_source,
            "evaluation_mode": evaluation_mode,
            "severity": overall_severity,
            "recommended_action": recommended_action,
            "labeled_rows": labeled_rows,
            "sample_rows": sample_rows,
            "positive_labels_rate": (
                float(
                    pd.to_numeric(labeled_window_df["y_true"], errors="coerce")
                    .fillna(0)
                    .mean()
                )
                if evaluation_mode == "labeled"
                else None
            ),
            "positive_predictions_rate": (
                float(
                    pd.to_numeric(
                        labeled_window_df["pred_label"], errors="coerce"
                    )
                    .fillna(0)
                    .mean()
                )
                if evaluation_mode == "labeled"
                else current_metrics.get("positive_rate_pred")
            ),
            "metrics": current_metrics,
            "baseline_metrics": (
                baseline_metrics
                if evaluation_mode == "labeled"
                else baseline_proxy_metrics
            ),
            "degraded_metrics": degraded_metric_names,
            "unlabeled_quality_estimates": unlabeled_quality_estimates,
            "unlabeled_quality_estimation_error": (
                unlabeled_quality_estimation_error
                if evaluation_mode == "proxy"
                else None
            ),
        }

        finalize_quality_run(
            engine=engine,
            run_id=run_id,
            status=run_status,
            labeled_rows=labeled_rows,
            degraded_metrics_count=len(degraded_metrics),
            summary=summary,
        )

        logger.info(
            (
                "Quality job completed. run_id=%s labeled_rows=%s "
                "sample_rows=%s degraded_metrics=%s severity=%s mode=%s"
            ),
            run_id,
            labeled_rows,
            sample_rows,
            len(degraded_metrics),
            overall_severity,
            evaluation_mode,
        )
        incident_key = build_incident_key("quality", segment_key)
        sync_monitoring_incident(
            engine,
            incident_key=incident_key,
            source_type="quality",
            model_version=settings.model_version,
            segment_key=segment_key,
            severity=overall_severity,
            title=(
                "Quality degradation detected"
                if evaluation_mode == "labeled"
                else "Quality risk detected from proxy signals"
            ),
            recommended_action=recommended_action,
            summary={
                "run_id": run_id,
                "status": run_status,
                **summary,
            },
            latest_run_id=run_id,
        )
        maybe_execute_critical_reaction(engine, incident_key=incident_key)
        return {
            "run_id": run_id,
            "status": run_status,
            "summary": summary,
            "labeled_rows": labeled_rows,
            "sample_rows": sample_rows,
            "degraded_metrics_count": len(degraded_metrics),
        }

    except Exception as exc:
        logger.exception("Quality job failed. run_id=%s", run_id)
        finalize_quality_run(
            engine=engine,
            run_id=run_id,
            status="failed",
            labeled_rows=0,
            degraded_metrics_count=0,
            summary={"error": str(exc)},
        )
        incident_key = build_incident_key("quality", segment_key)
        sync_monitoring_incident(
            engine,
            incident_key=incident_key,
            source_type="quality",
            model_version=settings.model_version,
            segment_key=segment_key,
            severity="critical",
            title="Quality monitoring job failed",
            recommended_action=(
                "Check scheduler and job logs, then restore quality "
                "monitoring before making model decisions."
            ),
            summary={"error": str(exc), "run_id": run_id},
            latest_run_id=run_id,
        )
        maybe_execute_critical_reaction(engine, incident_key=incident_key)
        raise
    finally:
        engine.dispose()


def main() -> None:
    """Execute one batch quality monitoring run over the latest labels."""

    setup_logging()

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.window_size <= 0:
        parser.error("--window-size must be greater than 0")
    if args.min_rows <= 0:
        parser.error("--min-rows must be greater than 0")

    result = run_quality_job(
        window_size=args.window_size,
        min_rows=args.min_rows,
        baseline_source=args.baseline_source,
        segment_key=args.segment_key,
    )

    if result["status"] == "completed":
        print("Quality job completed.")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
