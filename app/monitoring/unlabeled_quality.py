from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
BASELINE_LABEL_SPLITS = {
    "test": PROCESSED_DATA_DIR / "test.csv",
    "validation": PROCESSED_DATA_DIR / "val.csv",
}
ESTIMATED_METRIC_NAMES: tuple[str, ...] = (
    "positive_rate_true",
    "precision",
    "recall",
    "f1",
    "accuracy",
)


def ensure_feature_columns(
    df: pd.DataFrame, feature_columns: list[str] | None
) -> pd.DataFrame:
    """Return the reference feature frame with the expected model columns."""

    if feature_columns is None:
        return df.copy()

    out = df.copy()
    for column in feature_columns:
        if column not in out.columns:
            out[column] = np.nan
    return out[feature_columns]


def load_reference_labeled_data(baseline_source: str) -> pd.DataFrame:
    """Load the labeled split used to calibrate unlabeled estimates."""

    split_path = BASELINE_LABEL_SPLITS.get(baseline_source)
    if split_path is None:
        raise ValueError(f"Unsupported baseline source: {baseline_source}")
    if not split_path.exists():
        raise FileNotFoundError(
            f"Baseline labeled split not found: {split_path}"
        )

    reference_df = pd.read_csv(split_path)
    if "target" not in reference_df.columns:
        raise ValueError(
            f"Expected target column in baseline labeled split: {split_path}"
        )
    return reference_df


def compute_confusion_rates(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Estimate P(y_hat | y) from a labeled reference split."""

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
        normalize="true",
    )
    if confusion.shape != (2, 2):
        raise ValueError("Expected a 2x2 confusion matrix for BBSE.")
    if np.any(np.isnan(confusion)):
        raise ValueError("Reference confusion matrix contains NaN values.")
    if np.min(confusion.sum(axis=1)) <= 0.0:
        raise ValueError(
            "Reference confusion matrix must include both target classes."
        )
    return cast(np.ndarray, np.asarray(confusion, dtype=float))


def estimate_label_shift_priors(
    confusion_rates: np.ndarray,
    current_pred_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate current class priors under the BBSE label-shift assumption."""

    current_distribution = np.array(
        [
            float(np.mean(current_pred_labels == 0)),
            float(np.mean(current_pred_labels == 1)),
        ],
        dtype=float,
    )
    raw_priors = np.linalg.pinv(confusion_rates.T) @ current_distribution
    clipped_priors = np.clip(raw_priors, 0.0, None)
    prior_sum = float(clipped_priors.sum())
    if prior_sum <= 0.0:
        clipped_priors = np.array([0.5, 0.5], dtype=float)
    else:
        clipped_priors = clipped_priors / prior_sum
    return clipped_priors, current_distribution


def estimate_metrics_from_priors(
    confusion_rates: np.ndarray,
    class_priors: np.ndarray,
) -> dict[str, float]:
    """Estimate metrics from class priors and reference confusion rates."""

    negative_rate = float(class_priors[0])
    positive_rate = float(class_priors[1])
    true_negative_rate = float(confusion_rates[0, 0])
    false_positive_rate = float(confusion_rates[0, 1])
    false_negative_rate = float(confusion_rates[1, 0])
    true_positive_rate = float(confusion_rates[1, 1])

    true_positive = positive_rate * true_positive_rate
    false_negative = positive_rate * false_negative_rate
    false_positive = negative_rate * false_positive_rate
    true_negative = negative_rate * true_negative_rate

    precision_denominator = true_positive + false_positive
    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0.0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0.0
        else 0.0
    )
    f1_denominator = precision + recall
    f1 = (
        (2.0 * precision * recall) / f1_denominator
        if f1_denominator > 0.0
        else 0.0
    )

    return {
        "positive_rate_true": positive_rate,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(true_positive + true_negative),
    }


def bootstrap_bbse_metrics(
    confusion_rates: np.ndarray,
    current_pred_labels: np.ndarray,
    *,
    bootstrap_samples: int = 200,
    random_state: int = 42,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Bootstrap BBSE estimates from the current unlabeled window."""

    if len(current_pred_labels) == 0:
        return ({name: (0.0, 0.0) for name in ESTIMATED_METRIC_NAMES}, {})

    rng = np.random.default_rng(random_state)
    sampled_metrics = {name: [] for name in ESTIMATED_METRIC_NAMES}
    for _ in range(bootstrap_samples):
        sample_indices = rng.choice(
            len(current_pred_labels),
            size=len(current_pred_labels),
            replace=True,
        )
        sampled_pred_labels = current_pred_labels[sample_indices]
        sampled_priors, _ = estimate_label_shift_priors(
            confusion_rates, sampled_pred_labels
        )
        estimates = estimate_metrics_from_priors(
            confusion_rates, sampled_priors
        )
        for metric_name, metric_value in estimates.items():
            sampled_metrics[metric_name].append(float(metric_value))

    confidence_intervals = {
        metric_name: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for metric_name, values in sampled_metrics.items()
    }
    uncertainties = {
        metric_name: float((interval[1] - interval[0]) / 2.0)
        for metric_name, interval in confidence_intervals.items()
    }
    return confidence_intervals, uncertainties


def estimate_unlabeled_quality_with_bbse(
    *,
    current_window_df: pd.DataFrame,
    model: Any,
    threshold: float,
    baseline_source: str,
    feature_columns: list[str] | None = None,
    baseline_metrics: Mapping[str, float] | None = None,
    bootstrap_samples: int = 200,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Estimate unlabeled quality under the explicit label-shift assumption."""

    if current_window_df.empty:
        return []
    if "__pred_label" not in current_window_df.columns:
        raise ValueError("Current unlabeled window must include __pred_label.")

    reference_df = load_reference_labeled_data(baseline_source)
    reference_feature_columns = [
        column for column in reference_df.columns if column != "target"
    ]
    reference_features = ensure_feature_columns(
        reference_df[reference_feature_columns],
        feature_columns or reference_feature_columns,
    )
    reference_true = (
        pd.to_numeric(reference_df["target"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    reference_scores = np.asarray(
        model.predict_proba(reference_features)[:, 1],
        dtype=float,
    )
    reference_pred = (reference_scores >= threshold).astype(int)

    current_pred_labels = np.asarray(
        pd.to_numeric(current_window_df["__pred_label"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy(),
        dtype=int,
    )

    confusion_rates = compute_confusion_rates(reference_true, reference_pred)
    class_priors, current_pred_distribution = estimate_label_shift_priors(
        confusion_rates,
        current_pred_labels,
    )
    estimated_metrics = estimate_metrics_from_priors(
        confusion_rates,
        class_priors,
    )
    confidence_intervals, uncertainties = bootstrap_bbse_metrics(
        confusion_rates,
        current_pred_labels,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )

    estimated_positive_rate = float(class_priors[1])
    condition_number = float(np.linalg.cond(confusion_rates.T))
    rows: list[dict[str, Any]] = []
    for metric_name, metric_value in estimated_metrics.items():
        interval = confidence_intervals.get(
            metric_name, (metric_value, metric_value)
        )
        rows.append(
            {
                "assumption_type": "label_shift",
                "estimated_positive_rate": estimated_positive_rate,
                "estimated_metric_name": metric_name,
                "estimated_metric_value": float(metric_value),
                "quality_estimate_uncertainty": float(
                    uncertainties.get(metric_name, 0.0)
                ),
                "confidence_interval": {
                    "lower": float(interval[0]),
                    "upper": float(interval[1]),
                },
                "details": {
                    "baseline_source": baseline_source,
                    "reference_rows": int(len(reference_df)),
                    "current_rows": int(len(current_window_df)),
                    "threshold": float(threshold),
                    "current_pred_negative_rate": float(
                        current_pred_distribution[0]
                    ),
                    "current_pred_positive_rate": float(
                        current_pred_distribution[1]
                    ),
                    "reference_confusion_matrix": confusion_rates.tolist(),
                    "matrix_condition_number": condition_number,
                    "baseline_metric_value": (
                        float(baseline_metrics[metric_name])
                        if baseline_metrics is not None
                        and metric_name in baseline_metrics
                        else None
                    ),
                },
            }
        )

    return rows
