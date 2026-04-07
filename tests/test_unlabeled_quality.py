from __future__ import annotations

import numpy as np
import pandas as pd

from app.monitoring import unlabeled_quality


class DummyThresholdModel:
    def predict_proba(self, X):
        ages = pd.to_numeric(X["age"], errors="coerce").fillna(0).astype(float)
        scores = np.where(ages >= 50.0, 0.9, 0.1)
        return np.column_stack([1.0 - scores, scores])


def test_estimate_unlabeled_quality_with_bbse_returns_label_shift_rows(
    monkeypatch,
) -> None:
    reference_df = pd.DataFrame(
        {
            "age": [22, 27, 35, 55, 61, 68],
            "job": ["admin", "student", "admin", "admin", "student", "admin"],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )
    current_window_df = pd.DataFrame({"__pred_label": [1, 1, 1, 0, 0]})

    monkeypatch.setattr(
        unlabeled_quality,
        "load_reference_labeled_data",
        lambda baseline_source: reference_df,
    )

    rows = unlabeled_quality.estimate_unlabeled_quality_with_bbse(
        current_window_df=current_window_df,
        model=DummyThresholdModel(),
        threshold=0.5,
        baseline_source="test",
        feature_columns=["age", "job"],
        baseline_metrics={"f1": 0.95, "positive_rate_true": 0.50},
        bootstrap_samples=50,
        random_state=7,
    )

    assert {row["estimated_metric_name"] for row in rows} == set(
        unlabeled_quality.ESTIMATED_METRIC_NAMES
    )

    rows_by_name = {row["estimated_metric_name"]: row for row in rows}
    positive_rate_row = rows_by_name["positive_rate_true"]

    assert positive_rate_row["assumption_type"] == "label_shift"
    assert round(positive_rate_row["estimated_positive_rate"], 2) == 0.60
    assert round(positive_rate_row["estimated_metric_value"], 2) == 0.60
    assert positive_rate_row["quality_estimate_uncertainty"] >= 0.0
    assert positive_rate_row["confidence_interval"]["lower"] <= 0.60
    assert positive_rate_row["confidence_interval"]["upper"] >= 0.60
    assert positive_rate_row["details"]["baseline_source"] == "test"
    assert positive_rate_row["details"]["reference_rows"] == 6
    assert positive_rate_row["details"]["current_rows"] == 5
    assert positive_rate_row["details"]["baseline_metric_value"] == 0.50
    assert rows_by_name["f1"]["details"]["baseline_metric_value"] == 0.95
