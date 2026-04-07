from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from app.api import main
from app.monitoring import drift_job, quality_job


class DummyScoreModel:
    def predict_proba(self, X):
        ages = pd.to_numeric(X["age"], errors="coerce").fillna(0).astype(float)
        scores = np.clip((ages - 20.0) / 60.0, 0.05, 0.95).to_numpy()
        return np.column_stack([1.0 - scores, scores])


def seed_inference_rows(
    engine, *, count: int, segment_key: str, positive_label_every: int = 2
) -> None:
    for index in range(count):
        age = 20 + index
        score = 0.9 if index % positive_label_every == 0 else 0.1
        main.insert_inference_log(
            engine=engine,
            request_id=f"00000000-0000-0000-0000-{index:012d}",
            features={
                "age": age,
                "job": "admin" if index % 2 == 0 else "student",
            },
            score=score,
            pred_label=int(score >= 0.5),
            threshold=0.5,
            segment_key=segment_key,
            latency_ms=5.0 + index,
        )


@pytest.mark.integration
def test_run_quality_job_persists_quality_run_and_metrics(
    monkeypatch, postgres_engine
) -> None:
    segment_key = "quality-it"
    seed_inference_rows(
        postgres_engine,
        count=20,
        segment_key=segment_key,
        positive_label_every=2,
    )

    labels = [
        main.GroundTruthLabelRequest(
            request_id=f"00000000-0000-0000-0000-{index:012d}",
            y_true=0 if index < 15 else 1,
            label_ts=datetime.now(UTC) - timedelta(minutes=index),
        )
        for index in range(20)
    ]
    main.upsert_ground_truth_labels(postgres_engine, labels)

    monkeypatch.setattr(
        quality_job, "create_engine", lambda *args, **kwargs: postgres_engine
    )
    monkeypatch.setattr(
        quality_job,
        "load_baseline_profile",
        lambda: {
            "test_metrics": {
                "roc_auc": 0.90,
                "pr_auc": 0.90,
                "precision": 0.90,
                "recall": 0.90,
                "f1": 0.90,
                "brier_score": 0.10,
                "positive_rate_pred": 0.50,
                "positive_rate_true": 0.50,
            }
        },
    )

    result = quality_job.run_quality_job(
        window_size=20,
        min_rows=10,
        baseline_source="test",
        segment_key=segment_key,
    )

    assert result["status"] == "completed"
    assert result["degraded_metrics_count"] > 0

    with postgres_engine.connect() as connection:
        run_row = (
            connection.execute(
                text(
                    """
                    SELECT status, labeled_rows, degraded_metrics_count
                    FROM quality_runs
                    """
                )
            )
            .mappings()
            .one()
        )
        metric_count = connection.execute(
            text("SELECT COUNT(*) FROM quality_metrics")
        ).scalar_one()

    assert run_row["status"] == "completed"
    assert int(run_row["labeled_rows"]) == 20
    assert int(run_row["degraded_metrics_count"]) > 0
    assert int(metric_count) >= 8


@pytest.mark.integration
def test_run_drift_job_persists_monitoring_run_and_feature_metrics(
    monkeypatch, postgres_engine
) -> None:
    segment_key = "drift-it"
    for index in range(20):
        age = 60 + index
        score = 0.95
        main.insert_inference_log(
            engine=postgres_engine,
            request_id=f"10000000-0000-0000-0000-{index:012d}",
            features={"age": age, "job": "student"},
            score=score,
            pred_label=1,
            threshold=0.5,
            segment_key=segment_key,
            latency_ms=3.0 + index,
        )

    monkeypatch.setattr(
        drift_job, "create_engine", lambda *args, **kwargs: postgres_engine
    )
    monkeypatch.setattr(
        drift_job,
        "load_baseline_profile",
        lambda: {
            "feature_columns": ["age", "job"],
            "numeric_features": ["age"],
            "categorical_features": ["job"],
            "threshold": 0.5,
        },
    )
    monkeypatch.setattr(
        drift_job,
        "load_reference_data",
        lambda: pd.DataFrame(
            {
                "age": list(range(20, 40)),
                "job": ["admin"] * 20,
            }
        ),
    )
    monkeypatch.setattr(drift_job, "load_model", lambda: DummyScoreModel())

    result = drift_job.run_drift_job(
        window_size=20, min_rows=10, segment_key=segment_key
    )

    assert result["status"] == "completed"
    assert result["overall_drift"] is True
    assert result["drifted_features_count"] > 0

    with postgres_engine.connect() as connection:
        run_row = (
            connection.execute(
                text(
                    """
                    SELECT status, drifted_features_count, overall_drift
                    FROM monitoring_runs
                    """
                )
            )
            .mappings()
            .one()
        )
        metric_rows = (
            connection.execute(
                text(
                    """
                    SELECT feature_name, detector_name, drift_detected
                    FROM drift_metrics
                    ORDER BY feature_name, detector_name
                    """
                )
            )
            .mappings()
            .all()
        )

    assert run_row["status"] == "completed"
    assert bool(run_row["overall_drift"]) is True
    assert int(run_row["drifted_features_count"]) > 0
    detector_names = {str(row["detector_name"]) for row in metric_rows}
    assert detector_names >= {
        "univariate",
        "domain_classifier",
        "mmd",
        "wasserstein",
        "cusum",
        "ewma",
        "adwin",
        "ddm",
        "eddm",
    }

    drifted_rows = {
        (str(row["feature_name"]), str(row["detector_name"]))
        for row in metric_rows
        if bool(row["drift_detected"])
    }
    assert ("age", "univariate") in drifted_rows
    assert ("job", "univariate") in drifted_rows
    assert ("__score", "wasserstein") in drifted_rows
    assert ("__score", "cusum") in drifted_rows
    assert ("__score", "ewma") in drifted_rows
    assert ("__score", "adwin") in drifted_rows
    assert ("__score", "ddm") in drifted_rows
    assert ("__score", "eddm") in drifted_rows
    assert ("__multivariate__", "domain_classifier") in drifted_rows
    assert ("__multivariate__", "mmd") in drifted_rows

    advanced_summary_rows = result["summary"]["advanced_drift_detectors"]
    advanced_summary_detector_names = {
        str(item["detector_name"]) for item in advanced_summary_rows
    }
    assert advanced_summary_detector_names == {
        "mmd",
        "wasserstein",
        "cusum",
        "ewma",
        "adwin",
        "ddm",
        "eddm",
    }
    assert all(
        bool(item["drift_detected"]) is True for item in advanced_summary_rows
    )
