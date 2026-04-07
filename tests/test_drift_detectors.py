from __future__ import annotations

import numpy as np
import pandas as pd

from app.monitoring import drift_detectors


def _reference_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": np.linspace(20, 35, 120),
            "balance": np.linspace(0, 100, 120),
            "job": ["admin"] * 60 + ["student"] * 60,
        }
    )


def _current_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": np.linspace(55, 70, 120),
            "balance": np.linspace(250, 500, 120),
            "job": ["management"] * 80 + ["retired"] * 40,
        }
    )


def test_analyze_mmd_drift_detects_joint_shift() -> None:
    result = drift_detectors.analyze_mmd_drift(
        _reference_feature_frame(),
        _current_feature_frame(),
    )

    assert result["feature_name"] == "__multivariate__"
    assert result["detector_name"] == "mmd"
    assert result["statistic"] is not None
    assert result["pvalue"] is not None
    assert result["effect_size"] is not None
    assert result["pvalue_adj"] is not None
    assert result["drift_detected"] is True
    assert "window_start" in result
    assert "window_end" in result
    assert "segment_key" in result


def test_streaming_score_detectors_fire_on_strong_shift() -> None:
    reference_scores = np.linspace(0.05, 0.30, 120)
    current_scores = np.linspace(0.75, 0.95, 120)

    detector_results = {
        result["detector_name"]: result
        for result in [
            drift_detectors.analyze_score_wasserstein(
                reference_scores, current_scores
            ),
            drift_detectors.analyze_score_cusum(
                reference_scores, current_scores
            ),
            drift_detectors.analyze_score_ewma(
                reference_scores, current_scores
            ),
            drift_detectors.analyze_score_adwin(
                reference_scores, current_scores
            ),
            drift_detectors.analyze_score_ddm(
                reference_scores, current_scores
            ),
            drift_detectors.analyze_score_eddm(
                reference_scores, current_scores
            ),
        ]
    }

    assert set(detector_results) == {
        "wasserstein",
        "cusum",
        "ewma",
        "adwin",
        "ddm",
        "eddm",
    }
    assert all(
        detector_results[name]["drift_detected"] is True
        for name in detector_results
    )
    assert all(
        detector_results[name]["statistic"] is not None
        for name in detector_results
    )
    assert detector_results["adwin"]["details"]["detection_index"] is not None
    assert detector_results["ddm"]["details"]["drift_index"] is not None
    assert detector_results["eddm"]["details"]["drift_index"] is not None
