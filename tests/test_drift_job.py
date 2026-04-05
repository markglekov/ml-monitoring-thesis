from __future__ import annotations

import numpy as np
import pandas as pd

from app.monitoring import drift_job


def test_calculate_numeric_psi_is_near_zero_for_identical_distributions() -> (
    None
):
    reference = pd.Series(np.linspace(0, 100, 200))
    current = pd.Series(np.linspace(0, 100, 200))

    psi_value = drift_job.calculate_numeric_psi(reference, current)

    assert psi_value is not None
    assert abs(psi_value) < 1e-6


def test_analyze_numeric_feature_detects_drift_for_large_shift() -> None:
    reference = pd.Series(np.linspace(0, 10, 200))
    current = pd.Series(np.linspace(30, 40, 200))

    result = drift_job.analyze_numeric_feature("balance", reference, current)

    assert result["feature_name"] == "balance"
    assert result["feature_type"] == "numeric"
    assert result["drift_detected"] is True
    assert result["psi_value"] is not None
    assert (
        result["details"]["current_mean"] > result["details"]["reference_mean"]
    )


def test_analyze_categorical_feature_detects_distribution_shift() -> None:
    reference = pd.Series(["a"] * 180 + ["b"] * 20)
    current = pd.Series(["a"] * 40 + ["b"] * 160)

    result = drift_job.analyze_categorical_feature("job", reference, current)

    assert result["feature_name"] == "job"
    assert result["feature_type"] == "categorical"
    assert result["drift_detected"] is True
    assert result["chi2_pvalue"] is not None
    assert result["psi_value"] is not None


def test_analyze_categorical_feature_handles_insufficient_data() -> None:
    reference = pd.Series(["admin", "student", "admin"])
    current = pd.Series(["admin", "admin", "student"])

    result = drift_job.analyze_categorical_feature("job", reference, current)

    assert result["drift_detected"] is False
    assert result["details"]["status"] == "insufficient_data"


def test_apply_adjusted_pvalues_populates_severity_and_adjusted_values() -> (
    None
):
    results = [
        drift_job.analyze_numeric_feature(
            "balance",
            pd.Series(np.linspace(0, 10, 200)),
            pd.Series(np.linspace(30, 40, 200)),
        ),
        drift_job.analyze_categorical_feature(
            "job",
            pd.Series(["a"] * 180 + ["b"] * 20),
            pd.Series(["a"] * 40 + ["b"] * 160),
        ),
    ]

    drift_job.apply_adjusted_pvalues(results)

    assert all(item["pvalue_adj"] is not None for item in results)
    assert any(item["severity"] in {"warning", "critical"} for item in results)


def test_analyze_multivariate_drift_detects_joint_shift() -> None:
    reference_df = pd.DataFrame(
        {
            "age": np.linspace(20, 35, 120),
            "balance": np.linspace(0, 100, 120),
            "job": ["admin"] * 60 + ["student"] * 60,
        }
    )
    current_df = pd.DataFrame(
        {
            "age": np.linspace(55, 70, 120),
            "balance": np.linspace(250, 500, 120),
            "job": ["management"] * 80 + ["retired"] * 40,
        }
    )

    result = drift_job.analyze_multivariate_drift(reference_df, current_df)

    assert result["feature_name"] == "__multivariate__"
    assert result["detector_name"] == "domain_classifier"
    assert result["effect_size"] is not None
    assert result["pvalue_adj"] is not None
    assert result["severity"] in {"warning", "critical"}
