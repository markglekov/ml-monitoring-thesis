from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
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

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = settings.baseline_path

DEGRADATION_RULES: dict[str, dict[str, Any]] = {
    "roc_auc": {"mode": "min", "threshold": 0.02},
    "pr_auc": {"mode": "min", "threshold": 0.04},
    "precision": {"mode": "min", "threshold": 0.04},
    "recall": {"mode": "min", "threshold": 0.03},
    "f1": {"mode": "min", "threshold": 0.04},
    "brier_score": {"mode": "max", "threshold": 0.02},
    "positive_rate_pred": {"mode": "abs", "threshold": 0.08},
    "positive_rate_true": {"mode": "abs", "threshold": 0.05},
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


def detect_metric_degradation(
    metric_name: str,
    metric_value: float | None,
    baseline_value: float | None,
) -> tuple[bool, float | None, dict[str, Any]]:
    """Compare a live metric with its baseline counterpart."""

    rule = DEGRADATION_RULES.get(metric_name)
    if metric_value is None or baseline_value is None or rule is None:
        return False, None, {"status": "comparison_not_available"}

    delta_value = float(metric_value) - float(baseline_value)
    mode = str(rule["mode"])
    threshold = float(rule["threshold"])

    if mode == "min":
        degraded = delta_value <= -threshold
    elif mode == "max":
        degraded = delta_value >= threshold
    elif mode == "abs":
        degraded = abs(delta_value) >= threshold
    else:
        raise ValueError(f"Unsupported degradation mode: {mode}")

    return degraded, delta_value, {"mode": mode, "threshold": threshold}


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
            "degradation_detected": item["degradation_detected"],
            "details_json": safe_json(item["details"]),
        }
        for item in metrics_rows
    ]

    with engine.begin() as connection:
        connection.execute(query, payloads)


def build_metric_rows(
    current_metrics: dict[str, float | None],
    baseline_metrics: dict[str, float],
    labeled_rows: int,
    baseline_source: str,
) -> list[dict[str, Any]]:
    """Build metric comparison rows for insertion into quality_metrics."""

    rows: list[dict[str, Any]] = []

    for metric_name, metric_value in current_metrics.items():
        baseline_value = baseline_metrics.get(metric_name)
        degraded, delta_value, comparison_details = detect_metric_degradation(
            metric_name, metric_value, baseline_value
        )

        rows.append(
            {
                "metric_name": metric_name,
                "metric_value": metric_value,
                "baseline_value": baseline_value,
                "delta_value": delta_value,
                "degradation_detected": degraded,
                "details": {
                    "baseline_source": baseline_source,
                    "labeled_rows": labeled_rows,
                    **comparison_details,
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

        if labeled_window_df.empty or len(labeled_window_df) < min_rows:
            summary = {
                "status": "skipped_insufficient_data",
                "labeled_rows": int(len(labeled_window_df)),
                "required_min_rows": int(min_rows),
                "segment_key": segment_key,
                "baseline_source": baseline_source,
            }
            finalize_quality_run(
                engine=engine,
                run_id=run_id,
                status="skipped_insufficient_data",
                labeled_rows=int(len(labeled_window_df)),
                degraded_metrics_count=0,
                summary=summary,
            )
            logger.info(
                "Quality job skipped due to insufficient labeled rows: %s",
                summary,
            )
            return {
                "run_id": run_id,
                "status": "skipped_insufficient_data",
                "summary": summary,
            }

        current_metrics = compute_quality_metrics(labeled_window_df)
        metrics_rows = build_metric_rows(
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            labeled_rows=int(len(labeled_window_df)),
            baseline_source=baseline_source,
        )
        insert_quality_metrics(
            engine,
            run_id=run_id,
            segment_key=segment_key,
            metrics_rows=metrics_rows,
        )

        degraded_metrics = [
            item for item in metrics_rows if item["degradation_detected"]
        ]
        degraded_metric_names = [
            item["metric_name"] for item in degraded_metrics
        ]

        summary = {
            "segment_key": segment_key,
            "baseline_source": baseline_source,
            "labeled_rows": int(len(labeled_window_df)),
            "positive_labels_rate": float(
                pd.to_numeric(labeled_window_df["y_true"], errors="coerce")
                .fillna(0)
                .mean()
            ),
            "positive_predictions_rate": float(
                pd.to_numeric(labeled_window_df["pred_label"], errors="coerce")
                .fillna(0)
                .mean()
            ),
            "metrics": current_metrics,
            "baseline_metrics": baseline_metrics,
            "degraded_metrics": degraded_metric_names,
        }

        finalize_quality_run(
            engine=engine,
            run_id=run_id,
            status="completed",
            labeled_rows=int(len(labeled_window_df)),
            degraded_metrics_count=len(degraded_metrics),
            summary=summary,
        )

        logger.info(
            (
                "Quality job completed. run_id=%s labeled_rows=%s "
                "degraded_metrics=%s"
            ),
            run_id,
            len(labeled_window_df),
            len(degraded_metrics),
        )
        return {
            "run_id": run_id,
            "status": "completed",
            "summary": summary,
            "labeled_rows": int(len(labeled_window_df)),
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
