from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from river.drift import ADWIN
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.common.config import settings
from app.common.logging import get_logger, setup_logging
from app.monitoring.incidents import (
    build_incident_key,
    highest_severity,
    sync_monitoring_incident,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DATA_PATH = ROOT / "data" / "processed" / "train.csv"
BASELINE_PATH = settings.baseline_path
MODEL_PATH = settings.model_path

logger = get_logger(__name__)

UNIVARIATE_PVALUE_THRESHOLD = 0.05
PSI_WARNING_THRESHOLD = 0.20
PSI_CRITICAL_THRESHOLD = 0.35
DOMAIN_AUC_WARNING_THRESHOLD = 0.65
DOMAIN_AUC_CRITICAL_THRESHOLD = 0.75
DOMAIN_PERMUTATIONS = 31
MMD_WARNING_THRESHOLD = 0.05
MMD_CRITICAL_THRESHOLD = 0.10
MMD_PERMUTATIONS = 31
WASSERSTEIN_WARNING_THRESHOLD = 0.10
WASSERSTEIN_CRITICAL_THRESHOLD = 0.20
CUSUM_WARNING_THRESHOLD = 5.0
CUSUM_CRITICAL_THRESHOLD = 8.0
EWMA_WARNING_THRESHOLD = 3.0
EWMA_CRITICAL_THRESHOLD = 5.0
ADWIN_DELTA = 0.20
ADWIN_WARNING_THRESHOLD = 0.10
ADWIN_CRITICAL_THRESHOLD = 0.20
DDM_WARNING_THRESHOLD = 0.15
DDM_CRITICAL_THRESHOLD = 0.40
EDDM_WARNING_THRESHOLD = 0.20
EDDM_CRITICAL_THRESHOLD = 0.40
ADVANCED_DRIFT_DETECTORS: tuple[str, ...] = (
    "mmd",
    "wasserstein",
    "cusum",
    "ewma",
    "adwin",
    "ddm",
    "eddm",
)


def load_reference_data() -> pd.DataFrame:
    """Load the reference dataset used for drift checks."""

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {REFERENCE_DATA_PATH}"
        )

    df = pd.read_csv(REFERENCE_DATA_PATH)
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    return df


def load_baseline_profile() -> dict[str, Any]:
    """Load the saved baseline profile produced during model training."""

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline profile not found: {BASELINE_PATH}")

    with BASELINE_PATH.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_model() -> Any:
    """Load the persisted model used to compare score distributions."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


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
    """Serialize monitoring payloads into JSON for PostgreSQL JSONB columns."""

    return json.dumps(to_native(obj), ensure_ascii=False, default=str)


def normalize_categorical(series: pd.Series) -> pd.Series:
    """Represent missing categorical values explicitly before comparisons."""

    return series.where(series.notna(), "__nan__").astype(str)


def load_current_window(
    engine: Engine, window_size: int, segment_key: str | None = None
) -> pd.DataFrame:
    """Load the latest inference window from PostgreSQL."""

    if segment_key:
        query = text(
            """
            SELECT ts, features_json, score, pred_label, segment_key
            FROM inference_log
            WHERE segment_key = :segment_key
            ORDER BY ts DESC
            LIMIT :window_size
            """
        )
        params = {"segment_key": segment_key, "window_size": window_size}
    else:
        query = text(
            """
            SELECT ts, features_json, score, pred_label, segment_key
            FROM inference_log
            ORDER BY ts DESC
            LIMIT :window_size
            """
        )
        params = {"window_size": window_size}

    with engine.connect() as connection:
        rows = connection.execute(query, params).mappings().all()

    rows = list(reversed(rows))

    records: list[dict[str, Any]] = []
    for row in rows:
        features = row["features_json"]
        if isinstance(features, str):
            features = json.loads(features)

        record = dict(features)
        record["__score"] = float(row["score"])
        record["__pred_label"] = int(row["pred_label"])
        record["__segment_key"] = row["segment_key"]
        record["__ts"] = str(row["ts"])
        records.append(record)

    return pd.DataFrame(records)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Guarantee a dataframe contains the exact requested columns in order."""

    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns]


def to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values and keep pandas semantics."""

    return cast(pd.Series, pd.to_numeric(series, errors="coerce"))


def benjamini_hochberg_adjust(
    pvalues: list[float | None],
) -> list[float | None]:
    """Adjust p-values with the Benjamini-Hochberg FDR procedure."""

    valid = [
        (index, float(value))
        for index, value in enumerate(pvalues)
        if value is not None
    ]
    adjusted: list[float | None] = [None] * len(pvalues)
    if not valid:
        return adjusted

    sorted_valid = sorted(valid, key=lambda item: item[1])
    total = len(sorted_valid)
    running_min = 1.0

    for rank, (original_index, pvalue) in reversed(
        list(enumerate(sorted_valid, start=1))
    ):
        candidate = min(running_min, (pvalue * total) / rank)
        running_min = candidate
        adjusted[original_index] = float(min(candidate, 1.0))

    return adjusted


def build_drift_recommended_action(
    severity: str, *, feature_name: str, detector_name: str
) -> str:
    """Return a concise operator action for one drift signal."""

    if severity == "critical":
        return (
            "Review upstream data changes, inspect the affected segment, "
            "collect fresh labels, and prepare rollback or retraining."
        )
    if severity == "warning":
        return (
            "Inspect recent windows and confirm whether the shift persists "
            "before changing the model or threshold."
        )
    if (
        feature_name == "__multivariate__"
        and detector_name == "domain_classifier"
    ):
        return "No action required."
    return "No action required."


def classify_univariate_drift_severity(
    *,
    pvalue_adj: float | None,
    effect_size: float | None,
) -> str:
    """Map adjusted p-values and effect size to a severity label."""

    if effect_size is not None and effect_size >= PSI_CRITICAL_THRESHOLD:
        return "critical"
    if pvalue_adj is not None and pvalue_adj <= 0.01:
        return "critical"
    if effect_size is not None and effect_size >= PSI_WARNING_THRESHOLD:
        return "warning"
    if pvalue_adj is not None and pvalue_adj < UNIVARIATE_PVALUE_THRESHOLD:
        return "warning"
    return "none"


def classify_domain_drift_severity(
    auc_value: float, pvalue: float | None
) -> str:
    """Map domain-classifier AUC and permutation p-value to severity."""

    if pvalue is None:
        return "none"
    if auc_value >= DOMAIN_AUC_CRITICAL_THRESHOLD and pvalue <= 0.05:
        return "critical"
    if auc_value >= DOMAIN_AUC_WARNING_THRESHOLD and pvalue <= 0.05:
        return "warning"
    return "none"


def classify_threshold_severity(
    effect_size: float | None,
    *,
    warning_threshold: float,
    critical_threshold: float,
) -> str:
    """Map detector effect sizes to warning and critical severities."""

    if effect_size is None:
        return "none"
    if effect_size >= critical_threshold:
        return "critical"
    if effect_size >= warning_threshold:
        return "warning"
    return "none"


def build_detector_result(
    *,
    feature_name: str,
    feature_type: str,
    detector_name: str,
    severity: str,
    details: dict[str, Any],
    effect_size: float | None = None,
    pvalue_adj: float | None = None,
    ks_pvalue: float | None = None,
    chi2_pvalue: float | None = None,
    psi_value: float | None = None,
) -> dict[str, Any]:
    """Build one drift result row with consistent metadata."""

    return {
        "feature_name": feature_name,
        "feature_type": feature_type,
        "ks_pvalue": float(ks_pvalue) if ks_pvalue is not None else None,
        "chi2_pvalue": (
            float(chi2_pvalue) if chi2_pvalue is not None else None
        ),
        "psi_value": float(psi_value) if psi_value is not None else None,
        "detector_name": detector_name,
        "effect_size": float(effect_size) if effect_size is not None else None,
        "pvalue_adj": float(pvalue_adj) if pvalue_adj is not None else None,
        "severity": severity,
        "recommended_action": build_drift_recommended_action(
            severity,
            feature_name=feature_name,
            detector_name=detector_name,
        ),
        "drift_detected": severity in {"warning", "critical"},
        "details": details,
    }


def build_insufficient_detector_result(
    *,
    feature_name: str,
    feature_type: str,
    detector_name: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a detector result for skipped or data-poor windows."""

    return build_detector_result(
        feature_name=feature_name,
        feature_type=feature_type,
        detector_name=detector_name,
        severity="none",
        details=details or {"status": "insufficient_data"},
    )


def prepare_domain_classifier_frame(
    reference_df: pd.DataFrame, current_df: pd.DataFrame
) -> pd.DataFrame:
    """Prepare a joint encoded dataframe for domain-classifier drift."""

    combined_df = pd.concat(
        [reference_df, current_df], ignore_index=True
    ).copy()

    for column in combined_df.columns:
        series = combined_df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")
            fill_value = (
                float(numeric_series.median())
                if not numeric_series.dropna().empty
                else 0.0
            )
            combined_df[column] = numeric_series.fillna(fill_value)
            continue

        combined_df[column] = (
            combined_df[column].where(combined_df[column].notna(), "__nan__")
        ).astype(str)

    encoded_df = pd.get_dummies(combined_df, dummy_na=False)
    return encoded_df


def prepare_joint_feature_arrays(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    random_state: int = 42,
    max_rows: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode joint feature matrices and return standardized arrays."""

    encoded_df = prepare_domain_classifier_frame(reference_df, current_df)
    matrix = encoded_df.to_numpy(dtype=float)
    means = matrix.mean(axis=0, keepdims=True)
    stds = matrix.std(axis=0, keepdims=True)
    stds = np.where(stds < 1e-9, 1.0, stds)
    matrix = (matrix - means) / stds

    reference_matrix = matrix[: len(reference_df)]
    current_matrix = matrix[len(reference_df) :]
    rng = np.random.default_rng(random_state)

    if len(reference_matrix) > max_rows:
        reference_indices = rng.choice(
            len(reference_matrix), size=max_rows, replace=False
        )
        reference_matrix = reference_matrix[reference_indices]

    if len(current_matrix) > max_rows:
        current_indices = rng.choice(
            len(current_matrix), size=max_rows, replace=False
        )
        current_matrix = current_matrix[current_indices]

    return reference_matrix, current_matrix


def standardize_against_reference(
    reference_values: np.ndarray, current_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Standardize arrays using the reference window statistics."""

    reference_mean = float(np.mean(reference_values))
    reference_std = (
        float(np.std(reference_values, ddof=1))
        if len(reference_values) > 1
        else 0.0
    )
    scale = reference_std if reference_std > 1e-9 else 1.0
    reference_standardized = (reference_values - reference_mean) / scale
    current_standardized = (current_values - reference_mean) / scale
    return (
        reference_standardized,
        current_standardized,
        reference_mean,
        reference_std,
    )


def pairwise_squared_distances(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Compute pairwise squared Euclidean distances."""

    left_sq = np.sum(np.square(left), axis=1)[:, np.newaxis]
    right_sq = np.sum(np.square(right), axis=1)[np.newaxis, :]
    distances = left_sq + right_sq - 2.0 * np.matmul(left, right.T)
    return np.maximum(distances, 0.0)


def estimate_rbf_bandwidth(
    reference_matrix: np.ndarray, current_matrix: np.ndarray
) -> float:
    """Estimate an RBF bandwidth from the pooled pairwise distances."""

    combined = np.vstack([reference_matrix, current_matrix])
    distances = pairwise_squared_distances(combined, combined)
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > 0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def compute_rbf_mmd_statistic(
    reference_matrix: np.ndarray,
    current_matrix: np.ndarray,
    *,
    bandwidth: float,
) -> float:
    """Compute an unbiased RBF-kernel MMD estimate."""

    if len(reference_matrix) < 2 or len(current_matrix) < 2:
        return 0.0

    denominator = max(2.0 * bandwidth, 1e-12)
    kernel_xx = np.exp(
        -pairwise_squared_distances(reference_matrix, reference_matrix)
        / denominator
    )
    kernel_yy = np.exp(
        -pairwise_squared_distances(current_matrix, current_matrix)
        / denominator
    )
    kernel_xy = np.exp(
        -pairwise_squared_distances(reference_matrix, current_matrix)
        / denominator
    )

    np.fill_diagonal(kernel_xx, 0.0)
    np.fill_diagonal(kernel_yy, 0.0)

    reference_term = kernel_xx.sum() / (
        len(reference_matrix) * (len(reference_matrix) - 1)
    )
    current_term = kernel_yy.sum() / (
        len(current_matrix) * (len(current_matrix) - 1)
    )
    cross_term = kernel_xy.mean()
    return float(max(reference_term + current_term - 2.0 * cross_term, 0.0))


def analyze_mmd_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    random_state: int = 42,
    permutations: int = MMD_PERMUTATIONS,
    max_rows: int = 128,
) -> dict[str, Any]:
    """Detect multivariate drift with a kernel MMD test."""

    if len(reference_df) < 10 or len(current_df) < 10:
        return build_insufficient_detector_result(
            feature_name="__multivariate__",
            feature_type="multivariate",
            detector_name="mmd",
        )

    reference_matrix, current_matrix = prepare_joint_feature_arrays(
        reference_df,
        current_df,
        random_state=random_state,
        max_rows=max_rows,
    )
    bandwidth = estimate_rbf_bandwidth(reference_matrix, current_matrix)
    mmd_value = compute_rbf_mmd_statistic(
        reference_matrix,
        current_matrix,
        bandwidth=bandwidth,
    )

    combined = np.vstack([reference_matrix, current_matrix])
    split_index = len(reference_matrix)
    rng = np.random.default_rng(random_state)
    permutation_stats: list[float] = []
    for _ in range(permutations):
        permutation = rng.permutation(len(combined))
        permuted_reference = combined[permutation[:split_index]]
        permuted_current = combined[permutation[split_index:]]
        permutation_stats.append(
            compute_rbf_mmd_statistic(
                permuted_reference,
                permuted_current,
                bandwidth=bandwidth,
            )
        )

    pvalue = float(
        (1 + sum(value >= mmd_value for value in permutation_stats))
        / (len(permutation_stats) + 1)
    )
    severity = (
        classify_threshold_severity(
            mmd_value,
            warning_threshold=MMD_WARNING_THRESHOLD,
            critical_threshold=MMD_CRITICAL_THRESHOLD,
        )
        if pvalue <= 0.05
        else "none"
    )
    return build_detector_result(
        feature_name="__multivariate__",
        feature_type="multivariate",
        detector_name="mmd",
        effect_size=mmd_value,
        pvalue_adj=pvalue,
        severity=severity,
        details={
            "reference_rows": int(len(reference_df)),
            "current_rows": int(len(current_df)),
            "sampled_reference_rows": int(len(reference_matrix)),
            "sampled_current_rows": int(len(current_matrix)),
            "bandwidth": float(bandwidth),
            "permutations": int(permutations),
        },
    )


def analyze_score_wasserstein(
    reference_scores: np.ndarray, current_scores: np.ndarray
) -> dict[str, Any]:
    """Detect score drift with the Wasserstein distance."""

    if len(reference_scores) < 10 or len(current_scores) < 10:
        return build_insufficient_detector_result(
            feature_name="__score",
            feature_type="score",
            detector_name="wasserstein",
        )

    distance = float(wasserstein_distance(reference_scores, current_scores))
    severity = classify_threshold_severity(
        distance,
        warning_threshold=WASSERSTEIN_WARNING_THRESHOLD,
        critical_threshold=WASSERSTEIN_CRITICAL_THRESHOLD,
    )
    return build_detector_result(
        feature_name="__score",
        feature_type="score",
        detector_name="wasserstein",
        effect_size=distance,
        severity=severity,
        details={
            "reference_mean": float(np.mean(reference_scores)),
            "current_mean": float(np.mean(current_scores)),
            "reference_median": float(np.median(reference_scores)),
            "current_median": float(np.median(current_scores)),
            "reference_n": int(len(reference_scores)),
            "current_n": int(len(current_scores)),
        },
    )


def compute_cusum_statistic(
    standardized_values: np.ndarray, *, slack: float = 0.5
) -> tuple[float, float, float]:
    """Return the strongest positive or negative one-sided CUSUM signal."""

    positive_sum = 0.0
    negative_sum = 0.0
    max_signal = 0.0
    for value in standardized_values:
        positive_sum = max(0.0, positive_sum + float(value) - slack)
        negative_sum = max(0.0, negative_sum - float(value) - slack)
        max_signal = max(max_signal, positive_sum, negative_sum)
    return max_signal, positive_sum, negative_sum


def analyze_score_cusum(
    reference_scores: np.ndarray, current_scores: np.ndarray
) -> dict[str, Any]:
    """Detect persistent score shifts with CUSUM."""

    if len(reference_scores) < 10 or len(current_scores) < 10:
        return build_insufficient_detector_result(
            feature_name="__score",
            feature_type="score",
            detector_name="cusum",
        )

    _, current_standardized, reference_mean, reference_std = (
        standardize_against_reference(reference_scores, current_scores)
    )
    max_signal, positive_sum, negative_sum = compute_cusum_statistic(
        current_standardized
    )
    severity = classify_threshold_severity(
        max_signal,
        warning_threshold=CUSUM_WARNING_THRESHOLD,
        critical_threshold=CUSUM_CRITICAL_THRESHOLD,
    )
    return build_detector_result(
        feature_name="__score",
        feature_type="score",
        detector_name="cusum",
        effect_size=max_signal,
        severity=severity,
        details={
            "reference_mean": reference_mean,
            "reference_std": reference_std,
            "positive_cusum": float(positive_sum),
            "negative_cusum": float(negative_sum),
            "window_points": int(len(current_scores)),
        },
    )


def compute_ewma_control_ratio(
    standardized_values: np.ndarray, *, smoothing: float = 0.2
) -> tuple[float, float]:
    """Return the strongest EWMA control ratio over the current window."""

    ewma_value = 0.0
    max_ratio = 0.0
    for index, value in enumerate(standardized_values, start=1):
        ewma_value = smoothing * float(value) + (1.0 - smoothing) * ewma_value
        control_scale = math.sqrt(
            (smoothing / (2.0 - smoothing))
            * (1.0 - (1.0 - smoothing) ** (2 * index))
        )
        ratio = abs(ewma_value) / max(control_scale, 1e-12)
        max_ratio = max(max_ratio, ratio)
    return max_ratio, ewma_value


def analyze_score_ewma(
    reference_scores: np.ndarray, current_scores: np.ndarray
) -> dict[str, Any]:
    """Detect smoothed score shifts with EWMA."""

    if len(reference_scores) < 10 or len(current_scores) < 10:
        return build_insufficient_detector_result(
            feature_name="__score",
            feature_type="score",
            detector_name="ewma",
        )

    _, current_standardized, reference_mean, reference_std = (
        standardize_against_reference(reference_scores, current_scores)
    )
    max_ratio, ewma_value = compute_ewma_control_ratio(current_standardized)
    severity = classify_threshold_severity(
        max_ratio,
        warning_threshold=EWMA_WARNING_THRESHOLD,
        critical_threshold=EWMA_CRITICAL_THRESHOLD,
    )
    return build_detector_result(
        feature_name="__score",
        feature_type="score",
        detector_name="ewma",
        effect_size=max_ratio,
        severity=severity,
        details={
            "reference_mean": reference_mean,
            "reference_std": reference_std,
            "ewma_value": float(ewma_value),
            "smoothing": 0.2,
            "window_points": int(len(current_scores)),
        },
    )


def build_score_proxy_event_stream(
    reference_scores: np.ndarray, current_scores: np.ndarray
) -> tuple[np.ndarray, float, float, float, float]:
    """Project score windows into a proxy surprise stream."""

    reference_mean = float(np.mean(reference_scores))
    reference_std = (
        float(np.std(reference_scores, ddof=1))
        if len(reference_scores) > 1
        else 0.0
    )
    scale = reference_std if reference_std > 1e-9 else 1.0
    lower_bound = reference_mean - 2.5 * scale
    upper_bound = reference_mean + 2.5 * scale
    combined_scores = np.concatenate([reference_scores, current_scores])
    proxy_events = (
        (combined_scores < lower_bound) | (combined_scores > upper_bound)
    ).astype(int)
    reference_event_rate = float(proxy_events[: len(reference_scores)].mean())
    current_event_rate = float(proxy_events[len(reference_scores) :].mean())
    return (
        proxy_events,
        reference_event_rate,
        current_event_rate,
        float(lower_bound),
        float(upper_bound),
    )


def run_ddm(
    proxy_events: np.ndarray, *, min_instances: int = 10
) -> tuple[int | None, int | None]:
    """Run a lightweight DDM-style detector over a binary proxy stream."""

    running_error_rate = 0.0
    min_error_plus_std = math.inf
    warning_index: int | None = None
    drift_index: int | None = None

    for index, event in enumerate(proxy_events, start=1):
        running_error_rate += (float(event) - running_error_rate) / float(
            index
        )
        running_std = math.sqrt(
            max(
                running_error_rate * (1.0 - running_error_rate) / index,
                0.0,
            )
        )
        error_plus_std = running_error_rate + running_std
        if index < min_instances:
            if error_plus_std < min_error_plus_std:
                min_error_plus_std = error_plus_std
            continue

        if error_plus_std < min_error_plus_std:
            min_error_plus_std = error_plus_std
            continue

        if error_plus_std >= min_error_plus_std * 3.0:
            drift_index = index
            break
        if (
            error_plus_std >= min_error_plus_std * 2.0
            and warning_index is None
        ):
            warning_index = index

    return warning_index, drift_index


def analyze_score_adwin(
    reference_scores: np.ndarray, current_scores: np.ndarray
) -> dict[str, Any]:
    """Detect online score mean changes with ADWIN."""

    if len(reference_scores) < 10 or len(current_scores) < 10:
        return build_insufficient_detector_result(
            feature_name="__score",
            feature_type="score",
            detector_name="adwin",
        )

    adwin = ADWIN(
        delta=ADWIN_DELTA,
        clock=1,
        min_window_length=5,
        grace_period=10,
    )
    detection_index: int | None = None
    for index, value in enumerate(
        np.concatenate([reference_scores, current_scores]), start=1
    ):
        adwin.update(float(value))
        if adwin.drift_detected and detection_index is None:
            detection_index = index

    mean_shift = float(
        abs(np.mean(current_scores) - np.mean(reference_scores))
    )
    severity = (
        classify_threshold_severity(
            mean_shift,
            warning_threshold=ADWIN_WARNING_THRESHOLD,
            critical_threshold=ADWIN_CRITICAL_THRESHOLD,
        )
        if detection_index is not None
        else "none"
    )
    return build_detector_result(
        feature_name="__score",
        feature_type="score",
        detector_name="adwin",
        effect_size=mean_shift,
        severity=severity,
        details={
            "reference_mean": float(np.mean(reference_scores)),
            "current_mean": float(np.mean(current_scores)),
            "detection_index": detection_index,
            "n_detections": int(adwin.n_detections),
            "window_width": float(adwin.width),
        },
    )


def analyze_score_ddm(
    reference_scores: np.ndarray, current_scores: np.ndarray
) -> dict[str, Any]:
    """Detect proxy score surprises with a DDM-style control chart."""

    if len(reference_scores) < 10 or len(current_scores) < 10:
        return build_insufficient_detector_result(
            feature_name="__score",
            feature_type="score",
            detector_name="ddm",
        )

    (
        proxy_events,
        reference_event_rate,
        current_event_rate,
        lower_bound,
        upper_bound,
    ) = build_score_proxy_event_stream(reference_scores, current_scores)
    warning_index, drift_index = run_ddm(proxy_events)
    rate_delta = float(max(current_event_rate - reference_event_rate, 0.0))
    severity = (
        classify_threshold_severity(
            rate_delta,
            warning_threshold=DDM_WARNING_THRESHOLD,
            critical_threshold=DDM_CRITICAL_THRESHOLD,
        )
        if drift_index is not None
        else "none"
    )
    return build_detector_result(
        feature_name="__score",
        feature_type="score",
        detector_name="ddm",
        effect_size=rate_delta,
        severity=severity,
        details={
            "reference_event_rate": reference_event_rate,
            "current_event_rate": current_event_rate,
            "warning_index": warning_index,
            "drift_index": drift_index,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        },
    )


def compute_event_spacing_score(proxy_events: np.ndarray) -> float:
    """Summarize spacing between proxy error events."""

    event_indices = np.flatnonzero(proxy_events == 1)
    if len(event_indices) == 0:
        return float(len(proxy_events) + 1)
    if len(event_indices) == 1:
        return float(len(proxy_events))

    distances = np.diff(event_indices).astype(float)
    return float(np.mean(distances) + 2.0 * np.std(distances, ddof=0))


def run_eddm(
    reference_events: np.ndarray, current_events: np.ndarray
) -> tuple[int | None, int | None, float, float | None, float | None]:
    """Run a batch-friendly EDDM-style detector over event spacing."""

    reference_score = compute_event_spacing_score(reference_events)
    current_score = compute_event_spacing_score(current_events)
    ratio = float(current_score / max(reference_score, 1e-12))
    current_event_positions = np.flatnonzero(current_events == 1)
    if len(current_event_positions) == 0:
        return None, None, ratio, reference_score, current_score

    warning_index = (
        int(current_event_positions[0] + 1) if ratio <= 0.8 else None
    )
    drift_index = int(current_event_positions[0] + 1) if ratio <= 0.6 else None
    return (
        warning_index,
        drift_index,
        ratio,
        reference_score,
        current_score,
    )


def analyze_score_eddm(
    reference_scores: np.ndarray, current_scores: np.ndarray
) -> dict[str, Any]:
    """Detect degradation in event spacing with an EDDM-style detector."""

    if len(reference_scores) < 10 or len(current_scores) < 10:
        return build_insufficient_detector_result(
            feature_name="__score",
            feature_type="score",
            detector_name="eddm",
        )

    (
        proxy_events,
        reference_event_rate,
        current_event_rate,
        lower_bound,
        upper_bound,
    ) = build_score_proxy_event_stream(reference_scores, current_scores)
    reference_events = proxy_events[: len(reference_scores)]
    current_events = proxy_events[len(reference_scores) :]
    (
        warning_index,
        drift_index,
        ratio,
        reference_spacing_score,
        current_spacing_score,
    ) = run_eddm(reference_events, current_events)
    effect_size = float(max(1.0 - ratio, 0.0))
    severity = (
        classify_threshold_severity(
            effect_size,
            warning_threshold=EDDM_WARNING_THRESHOLD,
            critical_threshold=EDDM_CRITICAL_THRESHOLD,
        )
        if warning_index is not None
        else "none"
    )
    return build_detector_result(
        feature_name="__score",
        feature_type="score",
        detector_name="eddm",
        effect_size=effect_size,
        severity=severity,
        details={
            "reference_event_rate": reference_event_rate,
            "current_event_rate": current_event_rate,
            "warning_index": warning_index,
            "drift_index": drift_index,
            "ratio": float(ratio),
            "reference_spacing_score": reference_spacing_score,
            "current_spacing_score": current_spacing_score,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        },
    )


def fit_domain_classifier_auc(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    random_state: int = 42,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a simple domain classifier and return test AUC and split indices."""

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=500,
        solver="liblinear",
        random_state=random_state,
    )
    model.fit(X.iloc[train_idx], y[train_idx])
    y_score = model.predict_proba(X.iloc[test_idx])[:, 1]
    auc_value = float(roc_auc_score(y[test_idx], y_score))
    return auc_value, np.asarray(train_idx), np.asarray(test_idx)


def analyze_multivariate_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    random_state: int = 42,
    permutations: int = DOMAIN_PERMUTATIONS,
) -> dict[str, Any]:
    """Detect multivariate drift with a domain classifier and permutations."""

    if len(reference_df) < 20 or len(current_df) < 20:
        return {
            "feature_name": "__multivariate__",
            "feature_type": "multivariate",
            "ks_pvalue": None,
            "chi2_pvalue": None,
            "psi_value": None,
            "detector_name": "domain_classifier",
            "effect_size": None,
            "pvalue_adj": None,
            "severity": "none",
            "recommended_action": "No action required.",
            "drift_detected": False,
            "details": {"status": "insufficient_data"},
        }

    encoded_df = prepare_domain_classifier_frame(reference_df, current_df)
    labels = np.concatenate(
        [
            np.zeros(len(reference_df), dtype=int),
            np.ones(len(current_df), dtype=int),
        ]
    )

    try:
        auc_value, train_idx, test_idx = fit_domain_classifier_auc(
            encoded_df, labels, random_state=random_state
        )
    except ValueError as exc:
        return {
            "feature_name": "__multivariate__",
            "feature_type": "multivariate",
            "ks_pvalue": None,
            "chi2_pvalue": None,
            "psi_value": None,
            "detector_name": "domain_classifier",
            "effect_size": None,
            "pvalue_adj": None,
            "severity": "none",
            "recommended_action": "No action required.",
            "drift_detected": False,
            "details": {"status": "classifier_failed", "error": str(exc)},
        }

    rng = np.random.default_rng(random_state)
    permutation_aucs: list[float] = []
    for _ in range(permutations):
        permuted_labels = rng.permutation(labels)
        if len(np.unique(permuted_labels[train_idx])) < 2:
            permutation_aucs.append(0.5)
            continue
        if len(np.unique(permuted_labels[test_idx])) < 2:
            permutation_aucs.append(0.5)
            continue

        model = LogisticRegression(
            max_iter=500,
            solver="liblinear",
            random_state=random_state,
        )
        model.fit(encoded_df.iloc[train_idx], permuted_labels[train_idx])
        permuted_score = model.predict_proba(encoded_df.iloc[test_idx])[:, 1]
        permutation_aucs.append(
            float(roc_auc_score(permuted_labels[test_idx], permuted_score))
        )

    pvalue = float(
        (1 + sum(auc >= auc_value for auc in permutation_aucs))
        / (len(permutation_aucs) + 1)
    )
    severity = classify_domain_drift_severity(auc_value, pvalue)
    recommended_action = build_drift_recommended_action(
        severity,
        feature_name="__multivariate__",
        detector_name="domain_classifier",
    )

    return {
        "feature_name": "__multivariate__",
        "feature_type": "multivariate",
        "ks_pvalue": None,
        "chi2_pvalue": None,
        "psi_value": None,
        "detector_name": "domain_classifier",
        "effect_size": float(auc_value),
        "pvalue_adj": float(pvalue),
        "severity": severity,
        "recommended_action": recommended_action,
        "drift_detected": severity in {"warning", "critical"},
        "details": {
            "auc_value": float(auc_value),
            "permutation_pvalue": float(pvalue),
            "permutations": int(permutations),
            "reference_rows": int(len(reference_df)),
            "current_rows": int(len(current_df)),
            "encoded_features": int(encoded_df.shape[1]),
        },
    }


def apply_adjusted_pvalues(results: list[dict[str, Any]]) -> None:
    """Adjust univariate p-values and normalize drift metadata in-place."""

    raw_pvalues = [
        (
            float(item["ks_pvalue"])
            if item.get("ks_pvalue") is not None
            else float(item["chi2_pvalue"])
            if item.get("chi2_pvalue") is not None
            else None
        )
        for item in results
    ]
    adjusted_pvalues = benjamini_hochberg_adjust(raw_pvalues)

    for item, adjusted_pvalue in zip(results, adjusted_pvalues, strict=False):
        if item.get("detector_name") == "domain_classifier":
            continue

        item["detector_name"] = "univariate"
        item["pvalue_adj"] = adjusted_pvalue
        effect_size = item.get("psi_value")
        if effect_size is None:
            details = item.get("details", {})
            if item["feature_type"] == "numeric":
                effect_size = details.get("ks_statistic")
            else:
                effect_size = details.get("chi2_statistic")
        item["effect_size"] = (
            float(effect_size) if effect_size is not None else None
        )
        severity = classify_univariate_drift_severity(
            pvalue_adj=adjusted_pvalue,
            effect_size=(
                float(item["effect_size"])
                if item["effect_size"] is not None
                else None
            ),
        )
        item["severity"] = severity
        item["recommended_action"] = build_drift_recommended_action(
            severity,
            feature_name=str(item["feature_name"]),
            detector_name=str(item["detector_name"]),
        )
        item["drift_detected"] = severity in {"warning", "critical"}


def calculate_numeric_psi(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float | None:
    """Calculate PSI for numeric features using reference quantile buckets."""

    ref = to_numeric_series(reference).dropna()
    cur = to_numeric_series(current).dropna()

    if len(ref) < 10 or len(cur) < 10:
        return None

    quantiles = np.unique(
        np.quantile(ref.to_numpy(dtype=float), np.linspace(0.0, 1.0, bins + 1))
    )
    if len(quantiles) < 2:
        return 0.0

    bin_edges = quantiles.tolist()
    ref_binned = pd.cut(
        ref, bins=bin_edges, include_lowest=True, duplicates="drop"
    )
    cur_binned = pd.cut(
        cur, bins=bin_edges, include_lowest=True, duplicates="drop"
    )

    ref_pct = ref_binned.value_counts(normalize=True).sort_index()
    cur_pct = cur_binned.value_counts(normalize=True).sort_index()

    all_bins = ref_pct.index.union(cur_pct.index)
    ref_pct = ref_pct.reindex(all_bins, fill_value=0.0)
    cur_pct = cur_pct.reindex(all_bins, fill_value=0.0)

    eps = 1e-6
    ref_pct_values = np.clip(
        np.asarray(ref_pct.to_numpy(dtype=float), dtype=float), eps, None
    )
    cur_pct_values = np.clip(
        np.asarray(cur_pct.to_numpy(dtype=float), dtype=float), eps, None
    )

    psi = np.sum(
        (cur_pct_values - ref_pct_values)
        * np.log(cur_pct_values / ref_pct_values)
    )
    return float(psi)


def calculate_categorical_psi(
    reference: pd.Series, current: pd.Series
) -> float | None:
    """Calculate PSI for categorical features."""

    ref = normalize_categorical(reference)
    cur = normalize_categorical(current)

    if len(ref) < 10 or len(cur) < 10:
        return None

    ref_pct = ref.value_counts(normalize=True, dropna=False)
    cur_pct = cur.value_counts(normalize=True, dropna=False)

    all_values = ref_pct.index.union(cur_pct.index)
    ref_pct = ref_pct.reindex(all_values, fill_value=0.0)
    cur_pct = cur_pct.reindex(all_values, fill_value=0.0)

    eps = 1e-6
    ref_pct_values = np.clip(
        np.asarray(ref_pct.to_numpy(dtype=float), dtype=float), eps, None
    )
    cur_pct_values = np.clip(
        np.asarray(cur_pct.to_numpy(dtype=float), dtype=float), eps, None
    )

    psi = np.sum(
        (cur_pct_values - ref_pct_values)
        * np.log(cur_pct_values / ref_pct_values)
    )
    return float(psi)


def analyze_numeric_feature(
    feature_name: str, reference: pd.Series, current: pd.Series
) -> dict[str, Any]:
    """Analyze one numeric feature with KS test and PSI."""

    ref = to_numeric_series(reference).dropna()
    cur = to_numeric_series(current).dropna()

    if len(ref) < 10 or len(cur) < 10:
        return {
            "feature_name": feature_name,
            "feature_type": "numeric",
            "ks_pvalue": None,
            "chi2_pvalue": None,
            "psi_value": None,
            "detector_name": "univariate",
            "effect_size": None,
            "pvalue_adj": None,
            "severity": "none",
            "recommended_action": "No action required.",
            "drift_detected": False,
            "details": {"status": "insufficient_data"},
        }

    ks_result = cast(
        Any, ks_2samp(ref.to_numpy(dtype=float), cur.to_numpy(dtype=float))
    )
    psi_value = calculate_numeric_psi(ref, cur)
    ks_stat = float(ks_result.statistic)
    ks_pvalue = float(ks_result.pvalue)
    effect_size = psi_value if psi_value is not None else ks_stat
    severity = classify_univariate_drift_severity(
        pvalue_adj=ks_pvalue, effect_size=effect_size
    )
    drift_detected = bool(
        (ks_pvalue < UNIVARIATE_PVALUE_THRESHOLD)
        or ((psi_value is not None) and (psi_value >= PSI_WARNING_THRESHOLD))
    )

    return {
        "feature_name": feature_name,
        "feature_type": "numeric",
        "ks_pvalue": float(ks_pvalue),
        "chi2_pvalue": None,
        "psi_value": psi_value,
        "detector_name": "univariate",
        "effect_size": effect_size,
        "pvalue_adj": None,
        "severity": severity,
        "recommended_action": build_drift_recommended_action(
            severity,
            feature_name=feature_name,
            detector_name="univariate",
        ),
        "drift_detected": drift_detected,
        "details": {
            "ks_statistic": ks_stat,
            "reference_mean": float(ref.mean()),
            "current_mean": float(cur.mean()),
            "reference_std": float(ref.std()) if len(ref) > 1 else 0.0,
            "current_std": float(cur.std()) if len(cur) > 1 else 0.0,
            "reference_n": int(len(ref)),
            "current_n": int(len(cur)),
        },
    }


def analyze_categorical_feature(
    feature_name: str, reference: pd.Series, current: pd.Series
) -> dict[str, Any]:
    """Analyze one categorical feature with chi-square test and PSI."""

    ref = normalize_categorical(reference)
    cur = normalize_categorical(current)

    if len(ref) < 10 or len(cur) < 10:
        return {
            "feature_name": feature_name,
            "feature_type": "categorical",
            "ks_pvalue": None,
            "chi2_pvalue": None,
            "psi_value": None,
            "detector_name": "univariate",
            "effect_size": None,
            "pvalue_adj": None,
            "severity": "none",
            "recommended_action": "No action required.",
            "drift_detected": False,
            "details": {"status": "insufficient_data"},
        }

    ref_counts = ref.value_counts(dropna=False)
    cur_counts = cur.value_counts(dropna=False)

    all_values = ref_counts.index.union(cur_counts.index)
    ref_counts = ref_counts.reindex(all_values, fill_value=0)
    cur_counts = cur_counts.reindex(all_values, fill_value=0)

    contingency = np.vstack(
        [
            np.asarray(ref_counts.to_numpy(dtype=float), dtype=float),
            np.asarray(cur_counts.to_numpy(dtype=float), dtype=float),
        ]
    )

    if contingency.shape[1] <= 1:
        chi2_stat = 0.0
        chi2_pvalue = 1.0
    else:
        chi2_result = cast(Any, chi2_contingency(contingency))
        chi2_stat = float(chi2_result.statistic)
        chi2_pvalue = float(chi2_result.pvalue)

    psi_value = calculate_categorical_psi(ref, cur)
    effect_size = psi_value if psi_value is not None else chi2_stat
    severity = classify_univariate_drift_severity(
        pvalue_adj=chi2_pvalue, effect_size=effect_size
    )
    drift_detected = bool(
        (chi2_pvalue < UNIVARIATE_PVALUE_THRESHOLD)
        or ((psi_value is not None) and (psi_value >= PSI_WARNING_THRESHOLD))
    )

    top_ref = (
        (ref_counts / ref_counts.sum())
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )
    top_cur = (
        (cur_counts / cur_counts.sum())
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )

    return {
        "feature_name": feature_name,
        "feature_type": "categorical",
        "ks_pvalue": None,
        "chi2_pvalue": float(chi2_pvalue),
        "psi_value": psi_value,
        "detector_name": "univariate",
        "effect_size": effect_size,
        "pvalue_adj": None,
        "severity": severity,
        "recommended_action": build_drift_recommended_action(
            severity,
            feature_name=feature_name,
            detector_name="univariate",
        ),
        "drift_detected": drift_detected,
        "details": {
            "chi2_statistic": float(chi2_stat),
            "reference_top_values": {
                str(key): float(value) for key, value in top_ref.items()
            },
            "current_top_values": {
                str(key): float(value) for key, value in top_cur.items()
            },
            "reference_n": int(len(ref)),
            "current_n": int(len(cur)),
        },
    }


def insert_monitoring_run(
    engine: Engine, window_size: int, segment_key: str | None
) -> int:
    """Insert a monitoring run placeholder row and return its identifier."""

    query = text(
        """
        INSERT INTO monitoring_runs (
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


def finalize_monitoring_run(
    engine: Engine,
    run_id: int,
    status: str,
    drifted_features_count: int,
    total_features_count: int,
    overall_drift: bool,
    summary: dict[str, Any],
) -> None:
    """Finalize a monitoring run row with summary fields."""

    query = text(
        """
        UPDATE monitoring_runs
        SET
            ts_finished = NOW(),
            status = :status,
            drifted_features_count = :drifted_features_count,
            total_features_count = :total_features_count,
            overall_drift = :overall_drift,
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
                "drifted_features_count": drifted_features_count,
                "total_features_count": total_features_count,
                "overall_drift": overall_drift,
                "summary_json": safe_json(summary),
            },
        )


def insert_drift_metrics(
    engine: Engine, run_id: int, results: list[dict[str, Any]]
) -> None:
    """Persist per-feature drift metrics for one monitoring run."""

    if not results:
        return

    query = text(
        """
        INSERT INTO drift_metrics (
            run_id,
            feature_name,
            feature_type,
            ks_pvalue,
            chi2_pvalue,
            psi_value,
            detector_name,
            effect_size,
            pvalue_adj,
            severity,
            recommended_action,
            drift_detected,
            details_json
        )
        VALUES (
            :run_id,
            :feature_name,
            :feature_type,
            :ks_pvalue,
            :chi2_pvalue,
            :psi_value,
            :detector_name,
            :effect_size,
            :pvalue_adj,
            :severity,
            :recommended_action,
            :drift_detected,
            CAST(:details_json AS JSONB)
        )
        """
    )

    payloads = [
        {
            "run_id": run_id,
            "feature_name": item["feature_name"],
            "feature_type": item["feature_type"],
            "ks_pvalue": item["ks_pvalue"],
            "chi2_pvalue": item["chi2_pvalue"],
            "psi_value": item["psi_value"],
            "detector_name": item["detector_name"],
            "effect_size": item["effect_size"],
            "pvalue_adj": item["pvalue_adj"],
            "severity": item["severity"],
            "recommended_action": item["recommended_action"],
            "drift_detected": item["drift_detected"],
            "details_json": safe_json(item["details"]),
        }
        for item in results
    ]

    with engine.begin() as connection:
        connection.execute(query, payloads)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the drift-monitoring job."""

    parser = argparse.ArgumentParser(description="Drift monitoring job")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--segment-key", type=str, default=None)
    parser.add_argument("--min-rows", type=int, default=50)
    return parser


def run_drift_job(
    window_size: int, min_rows: int, segment_key: str | None = None
) -> dict[str, Any]:
    """Execute one batch drift-monitoring run."""

    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")
    if min_rows <= 0:
        raise ValueError("min_rows must be greater than 0")

    logger.info(
        (
            "Starting drift job. model_version=%s window_size=%s "
            "segment_key=%s min_rows=%s"
        ),
        settings.model_version,
        window_size,
        segment_key,
        min_rows,
    )

    baseline_profile = load_baseline_profile()
    feature_columns = baseline_profile["feature_columns"]
    numeric_features = set(baseline_profile.get("numeric_features", []))
    categorical_features = set(
        baseline_profile.get("categorical_features", [])
    )
    engine = create_engine(
        settings.database_url, future=True, pool_pre_ping=True
    )
    run_id = insert_monitoring_run(
        engine, window_size=window_size, segment_key=segment_key
    )

    try:
        reference_df = ensure_columns(load_reference_data(), feature_columns)
        current_df = load_current_window(
            engine, window_size=window_size, segment_key=segment_key
        )

        if current_df.empty or len(current_df) < min_rows:
            summary = {
                "status": "skipped_insufficient_data",
                "current_rows": int(len(current_df)),
                "required_min_rows": int(min_rows),
            }
            finalize_monitoring_run(
                engine=engine,
                run_id=run_id,
                status="skipped_insufficient_data",
                drifted_features_count=0,
                total_features_count=len(feature_columns),
                overall_drift=False,
                summary=summary,
            )
            logger.info(
                "Drift job skipped due to insufficient data: %s", summary
            )
            return {
                "run_id": run_id,
                "status": "skipped_insufficient_data",
                "summary": summary,
            }

        current_features_df = ensure_columns(current_df, feature_columns)

        model = load_model()
        reference_scores = np.asarray(
            model.predict_proba(reference_df)[:, 1], dtype=float
        )
        current_scores = np.asarray(
            pd.to_numeric(current_df["__score"], errors="coerce")
            .fillna(0.0)
            .to_numpy(),
            dtype=float,
        )

        results: list[dict[str, Any]] = []
        for feature in feature_columns:
            ref_col = reference_df[feature]
            cur_col = current_features_df[feature]

            if feature in numeric_features:
                result = analyze_numeric_feature(feature, ref_col, cur_col)
            elif feature in categorical_features:
                result = analyze_categorical_feature(feature, ref_col, cur_col)
            elif pd.api.types.is_numeric_dtype(ref_col):
                result = analyze_numeric_feature(feature, ref_col, cur_col)
            else:
                result = analyze_categorical_feature(feature, ref_col, cur_col)

            results.append(result)

        results.append(
            analyze_numeric_feature(
                feature_name="__score",
                reference=pd.Series(reference_scores),
                current=pd.Series(current_scores),
            )
        )
        apply_adjusted_pvalues(results)
        results.extend(
            [
                analyze_multivariate_drift(reference_df, current_features_df),
                analyze_mmd_drift(reference_df, current_features_df),
                analyze_score_wasserstein(reference_scores, current_scores),
                analyze_score_cusum(reference_scores, current_scores),
                analyze_score_ewma(reference_scores, current_scores),
                analyze_score_adwin(reference_scores, current_scores),
                analyze_score_ddm(reference_scores, current_scores),
                analyze_score_eddm(reference_scores, current_scores),
            ]
        )

        drifted = [item for item in results if item["drift_detected"]]
        insert_drift_metrics(engine, run_id, results)

        feature_results = [
            item
            for item in results
            if item["feature_name"] not in {"__score", "__multivariate__"}
        ]
        score_result = next(
            (
                item
                for item in results
                if item["feature_name"] == "__score"
                and item["detector_name"] == "univariate"
            ),
            None,
        )
        multivariate_result = next(
            (
                item
                for item in results
                if item["feature_name"] == "__multivariate__"
                and item["detector_name"] == "domain_classifier"
            ),
            None,
        )
        advanced_results = [
            item
            for item in results
            if item["detector_name"] in ADVANCED_DRIFT_DETECTORS
        ]
        top_drift = sorted(
            [
                item
                for item in feature_results
                if item["effect_size"] is not None
            ],
            key=lambda item: (
                float(item["effect_size"])
                if item["effect_size"] is not None
                else -1.0
            ),
            reverse=True,
        )[:5]

        threshold = float(baseline_profile.get("threshold", 0.5))
        positive_rate_current = float(
            pd.to_numeric(current_df["__pred_label"], errors="coerce")
            .fillna(0)
            .mean()
        )
        positive_rate_reference = float((reference_scores >= threshold).mean())
        drifted_features_count = len(
            {str(item["feature_name"]) for item in drifted}
        )
        total_features_count = len(
            {str(item["feature_name"]) for item in results}
        )
        overall_severity = highest_severity(
            [str(item["severity"]) for item in results]
        )
        overall_drift = overall_severity in {"warning", "critical"}
        recommended_action = next(
            (
                str(item["recommended_action"])
                for item in results
                if item["severity"] in {"critical", "warning"}
            ),
            "No action required.",
        )

        summary = {
            "reference_rows": int(len(reference_df)),
            "current_rows": int(len(current_df)),
            "severity": overall_severity,
            "recommended_action": recommended_action,
            "adjustment_method": "benjamini-hochberg",
            "drifted_features": [
                feature_name
                for feature_name in dict.fromkeys(
                    item["feature_name"]
                    for item in drifted
                    if item["feature_name"]
                    not in {"__score", "__multivariate__"}
                )
            ],
            "advanced_drift_detectors": [
                {
                    "feature_name": item["feature_name"],
                    "detector_name": item["detector_name"],
                    "effect_size": item["effect_size"],
                    "pvalue_adj": item["pvalue_adj"],
                    "severity": item["severity"],
                    "drift_detected": item["drift_detected"],
                    "details": item["details"],
                }
                for item in advanced_results
            ],
            "top_drift_by_psi": [
                {
                    "feature_name": item["feature_name"],
                    "psi_value": item["psi_value"],
                    "effect_size": item["effect_size"],
                    "pvalue_adj": item["pvalue_adj"],
                    "severity": item["severity"],
                    "drift_detected": item["drift_detected"],
                }
                for item in top_drift
            ],
            "score_drift": (
                {
                    "effect_size": score_result["effect_size"],
                    "pvalue_adj": score_result["pvalue_adj"],
                    "severity": score_result["severity"],
                    "drift_detected": score_result["drift_detected"],
                }
                if score_result is not None
                else None
            ),
            "multivariate_drift": (
                {
                    "effect_size": multivariate_result["effect_size"],
                    "pvalue_adj": multivariate_result["pvalue_adj"],
                    "severity": multivariate_result["severity"],
                    "drift_detected": multivariate_result["drift_detected"],
                    "details": multivariate_result["details"],
                }
                if multivariate_result is not None
                else None
            ),
            "reference_positive_rate_pred": positive_rate_reference,
            "current_positive_rate_pred": positive_rate_current,
            "segment_key": segment_key,
        }

        finalize_monitoring_run(
            engine=engine,
            run_id=run_id,
            status="completed",
            drifted_features_count=drifted_features_count,
            total_features_count=total_features_count,
            overall_drift=overall_drift,
            summary=summary,
        )

        logger.info(
            (
                "Drift job completed. run_id=%s drifted_features=%s "
                "total_features=%s overall_drift=%s severity=%s"
            ),
            run_id,
            drifted_features_count,
            total_features_count,
            overall_drift,
            overall_severity,
        )
        sync_monitoring_incident(
            engine,
            incident_key=build_incident_key("drift", segment_key),
            source_type="drift",
            model_version=settings.model_version,
            segment_key=segment_key,
            severity=overall_severity,
            title="Drift monitoring signal",
            recommended_action=recommended_action,
            summary={
                "run_id": run_id,
                "overall_drift": overall_drift,
                **summary,
            },
            latest_run_id=run_id,
        )
        return {
            "run_id": run_id,
            "status": "completed",
            "summary": summary,
            "overall_drift": overall_drift,
            "drifted_features_count": drifted_features_count,
            "total_features_count": total_features_count,
        }

    except Exception as exc:
        logger.exception("Drift job failed. run_id=%s", run_id)
        finalize_monitoring_run(
            engine=engine,
            run_id=run_id,
            status="failed",
            drifted_features_count=0,
            total_features_count=0,
            overall_drift=False,
            summary={"error": str(exc)},
        )
        sync_monitoring_incident(
            engine,
            incident_key=build_incident_key("drift", segment_key),
            source_type="drift",
            model_version=settings.model_version,
            segment_key=segment_key,
            severity="critical",
            title="Drift monitoring job failed",
            recommended_action=(
                "Check scheduler and job logs, then restore drift monitoring "
                "before relying on the model."
            ),
            summary={"error": str(exc), "run_id": run_id},
            latest_run_id=run_id,
        )
        raise
    finally:
        engine.dispose()


def main() -> None:
    """Execute one batch drift-monitoring run over the latest window."""

    setup_logging()

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.window_size <= 0:
        parser.error("--window-size must be greater than 0")
    if args.min_rows <= 0:
        parser.error("--min-rows must be greater than 0")

    result = run_drift_job(
        window_size=args.window_size,
        min_rows=args.min_rows,
        segment_key=args.segment_key,
    )

    if result["status"] == "completed":
        print("Drift job completed.")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
