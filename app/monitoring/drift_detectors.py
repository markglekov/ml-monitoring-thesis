from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from river.drift import ADWIN
from scipy.stats import wasserstein_distance

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
    recommended_action: str,
    effect_size: float | None = None,
    statistic: float | None = None,
    pvalue: float | None = None,
    pvalue_adj: float | None = None,
    ks_pvalue: float | None = None,
    chi2_pvalue: float | None = None,
    psi_value: float | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    segment_key: str | None = None,
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
        "statistic": float(statistic) if statistic is not None else None,
        "pvalue": float(pvalue) if pvalue is not None else None,
        "effect_size": float(effect_size) if effect_size is not None else None,
        "pvalue_adj": float(pvalue_adj) if pvalue_adj is not None else None,
        "window_start": window_start,
        "window_end": window_end,
        "segment_key": segment_key,
        "severity": severity,
        "recommended_action": recommended_action,
        "drift_detected": severity in {"warning", "critical"},
        "details": details,
    }


def build_insufficient_detector_result(
    *,
    feature_name: str,
    feature_type: str,
    detector_name: str,
    recommended_action: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a detector result for skipped or data-poor windows."""

    return build_detector_result(
        feature_name=feature_name,
        feature_type=feature_type,
        detector_name=detector_name,
        severity="none",
        details=details or {"status": "insufficient_data"},
        recommended_action=recommended_action,
    )


def prepare_domain_classifier_frame(
    reference_df: pd.DataFrame, current_df: pd.DataFrame
) -> pd.DataFrame:
    """Prepare a joint encoded dataframe for drift detectors."""

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

    return pd.get_dummies(combined_df, dummy_na=False)


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
    recommended_action: str = "No action required.",
) -> dict[str, Any]:
    """Detect multivariate drift with a kernel MMD test."""

    if len(reference_df) < 10 or len(current_df) < 10:
        return build_insufficient_detector_result(
            feature_name="__multivariate__",
            feature_type="multivariate",
            detector_name="mmd",
            recommended_action=recommended_action,
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
        statistic=mmd_value,
        pvalue=pvalue,
        pvalue_adj=pvalue,
        severity=severity,
        recommended_action=recommended_action,
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
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    recommended_action: str = "No action required.",
) -> dict[str, Any]:
    """Detect score drift with the Wasserstein distance."""

    if len(reference_scores) < 10 or len(current_scores) < 10:
        return build_insufficient_detector_result(
            feature_name="__score",
            feature_type="score",
            detector_name="wasserstein",
            recommended_action=recommended_action,
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
        statistic=distance,
        severity=severity,
        recommended_action=recommended_action,
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
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    recommended_action: str = "No action required.",
) -> dict[str, Any]:
    """Detect persistent score shifts with CUSUM."""

    if len(reference_scores) < 10 or len(current_scores) < 10:
        return build_insufficient_detector_result(
            feature_name="__score",
            feature_type="score",
            detector_name="cusum",
            recommended_action=recommended_action,
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
        statistic=max_signal,
        severity=severity,
        recommended_action=recommended_action,
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
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    recommended_action: str = "No action required.",
) -> dict[str, Any]:
    """Detect smoothed score shifts with EWMA."""

    if len(reference_scores) < 10 or len(current_scores) < 10:
        return build_insufficient_detector_result(
            feature_name="__score",
            feature_type="score",
            detector_name="ewma",
            recommended_action=recommended_action,
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
        statistic=max_ratio,
        severity=severity,
        recommended_action=recommended_action,
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
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    recommended_action: str = "No action required.",
) -> dict[str, Any]:
    """Detect online score mean changes with ADWIN."""

    if len(reference_scores) < 10 or len(current_scores) < 10:
        return build_insufficient_detector_result(
            feature_name="__score",
            feature_type="score",
            detector_name="adwin",
            recommended_action=recommended_action,
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
        statistic=mean_shift,
        severity=severity,
        recommended_action=recommended_action,
        details={
            "reference_mean": float(np.mean(reference_scores)),
            "current_mean": float(np.mean(current_scores)),
            "detection_index": detection_index,
            "n_detections": int(adwin.n_detections),
            "window_width": float(adwin.width),
        },
    )


def analyze_score_ddm(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    recommended_action: str = "No action required.",
) -> dict[str, Any]:
    """Detect proxy score surprises with a DDM-style control chart."""

    if len(reference_scores) < 10 or len(current_scores) < 10:
        return build_insufficient_detector_result(
            feature_name="__score",
            feature_type="score",
            detector_name="ddm",
            recommended_action=recommended_action,
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
        statistic=rate_delta,
        severity=severity,
        recommended_action=recommended_action,
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
) -> tuple[int | None, int | None, float, float, float]:
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
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    recommended_action: str = "No action required.",
) -> dict[str, Any]:
    """Detect degradation in event spacing with an EDDM-style detector."""

    if len(reference_scores) < 10 or len(current_scores) < 10:
        return build_insufficient_detector_result(
            feature_name="__score",
            feature_type="score",
            detector_name="eddm",
            recommended_action=recommended_action,
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
        statistic=effect_size,
        severity=severity,
        recommended_action=recommended_action,
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
