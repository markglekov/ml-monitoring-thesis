from __future__ import annotations

import pandas as pd

from app.monitoring import quality_job


def test_compute_quality_metrics_handles_single_class_window() -> None:
    window_df = pd.DataFrame(
        {
            "y_true": [0, 0, 0, 0],
            "pred_label": [0, 0, 1, 0],
            "score": [0.05, 0.10, 0.80, 0.20],
        }
    )

    metrics = quality_job.compute_quality_metrics(window_df)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["positive_rate_pred"] == 0.25
    assert metrics["positive_rate_true"] == 0.0


def test_detect_metric_degradation_supports_min_max_and_abs_rules() -> None:
    degraded_min, delta_min, details_min = quality_job.detect_metric_degradation("roc_auc", 0.60, 0.70)
    degraded_max, delta_max, details_max = quality_job.detect_metric_degradation("brier_score", 0.31, 0.25)
    degraded_abs, delta_abs, details_abs = quality_job.detect_metric_degradation("positive_rate_pred", 0.55, 0.40)

    assert degraded_min is True
    assert round(delta_min, 2) == -0.10
    assert details_min == {"mode": "min", "threshold": 0.02}

    assert degraded_max is True
    assert round(delta_max, 2) == 0.06
    assert details_max == {"mode": "max", "threshold": 0.02}

    assert degraded_abs is True
    assert round(delta_abs, 2) == 0.15
    assert details_abs == {"mode": "abs", "threshold": 0.08}


def test_build_metric_rows_includes_baseline_source_and_degradation_flags() -> None:
    rows = quality_job.build_metric_rows(
        current_metrics={"f1": 0.40, "roc_auc": None},
        baseline_metrics={"f1": 0.50, "roc_auc": 0.70},
        labeled_rows=120,
        baseline_source="test",
    )

    rows_by_name = {row["metric_name"]: row for row in rows}

    assert rows_by_name["f1"]["degradation_detected"] is True
    assert round(rows_by_name["f1"]["delta_value"], 2) == -0.10
    assert rows_by_name["f1"]["details"]["baseline_source"] == "test"
    assert rows_by_name["f1"]["details"]["labeled_rows"] == 120

    assert rows_by_name["roc_auc"]["degradation_detected"] is False
    assert rows_by_name["roc_auc"]["delta_value"] is None
    assert rows_by_name["roc_auc"]["details"]["status"] == "comparison_not_available"
