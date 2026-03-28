from __future__ import annotations

import numpy as np
import pandas as pd

from app.monitoring import drift_job


def test_calculate_numeric_psi_is_near_zero_for_identical_distributions() -> None:
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
    assert result["details"]["current_mean"] > result["details"]["reference_mean"]


def test_analyze_categorical_feature_detects_drift_for_distribution_shift() -> None:
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
