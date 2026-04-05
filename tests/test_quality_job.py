from __future__ import annotations

import numpy as np
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
    assert metrics["ece"] is not None
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["positive_rate_pred"] == 0.25
    assert metrics["positive_rate_true"] == 0.0


def test_detect_metric_degradation_supports_min_max_and_abs_rules() -> None:
    degraded_min, delta_min, effect_min, details_min = (
        quality_job.detect_metric_degradation("roc_auc", 0.60, 0.70)
    )
    degraded_max, delta_max, effect_max, details_max = (
        quality_job.detect_metric_degradation("brier_score", 0.31, 0.25)
    )
    degraded_abs, delta_abs, effect_abs, details_abs = (
        quality_job.detect_metric_degradation("positive_rate_pred", 0.55, 0.40)
    )
    degraded_absolute, delta_absolute, effect_absolute, details_absolute = (
        quality_job.detect_metric_degradation("score_psi", 0.24, None)
    )

    assert degraded_min is True
    assert delta_min is not None
    assert effect_min is not None
    assert round(delta_min, 2) == -0.10
    assert details_min["mode"] == "min_delta"
    assert details_min["threshold"] == 0.02

    assert degraded_max is True
    assert delta_max is not None
    assert effect_max is not None
    assert round(delta_max, 2) == 0.06
    assert details_max["mode"] == "max_delta"
    assert details_max["threshold"] == 0.02

    assert degraded_abs is True
    assert delta_abs is not None
    assert effect_abs is not None
    assert round(delta_abs, 2) == 0.15
    assert details_abs["mode"] == "abs_delta"
    assert details_abs["threshold"] == 0.08

    assert degraded_absolute is True
    assert delta_absolute is None
    assert effect_absolute == 0.24
    assert details_absolute["mode"] == "max_absolute"
    assert details_absolute["threshold"] == 0.20


def test_build_metric_rows_include_baseline_source_and_flags() -> None:
    rows = quality_job.build_metric_rows(
        current_metrics={"f1": 0.40, "roc_auc": None},
        baseline_metrics={"f1": 0.50, "roc_auc": 0.70},
        sample_rows=120,
        baseline_source="test",
        evaluation_mode="labeled",
    )

    rows_by_name = {row["metric_name"]: row for row in rows}

    assert rows_by_name["f1"]["degradation_detected"] is True
    assert rows_by_name["f1"]["delta_value"] is not None
    assert round(rows_by_name["f1"]["delta_value"], 2) == -0.10
    assert rows_by_name["f1"]["detector_name"] == "labeled"
    assert rows_by_name["f1"]["severity"] == "critical"
    assert rows_by_name["f1"]["details"]["baseline_source"] == "test"
    assert rows_by_name["f1"]["details"]["sample_rows"] == 120
    assert rows_by_name["f1"]["details"]["evaluation_mode"] == "labeled"

    assert rows_by_name["roc_auc"]["degradation_detected"] is False
    assert rows_by_name["roc_auc"]["delta_value"] is None
    assert (
        rows_by_name["roc_auc"]["details"]["status"]
        == "comparison_not_available"
    )


def test_compute_proxy_quality_metrics_adds_score_psi() -> None:
    current_window_df = pd.DataFrame(
        {"__score": [0.51, 0.49, 0.52, 0.48] * 20}
    )
    reference_scores = np.array([0.05, 0.10, 0.90, 0.95] * 20, dtype=float)

    metrics, details = quality_job.compute_proxy_quality_metrics(
        current_window_df=current_window_df,
        reference_scores=reference_scores,
        threshold=0.5,
    )

    assert metrics["score_psi"] is not None
    assert metrics["near_threshold_rate"] is not None
    assert details["score_psi"]["score_ks_pvalue"] is not None
