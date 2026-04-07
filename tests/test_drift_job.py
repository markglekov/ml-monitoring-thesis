from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.monitoring import drift_job


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


class DummyScoreModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ages = pd.to_numeric(X["age"], errors="coerce").fillna(0.0)
        scores = np.asarray(
            np.clip((ages - 20.0) / 60.0, 0.05, 0.95),
            dtype=float,
        )
        return np.column_stack([1.0 - scores, scores])


class DummyEngine:
    def dispose(self) -> None:
        return None


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
    reference_df = _reference_feature_frame()
    current_df = _current_feature_frame()

    result = drift_job.analyze_multivariate_drift(reference_df, current_df)

    assert result["feature_name"] == "__multivariate__"
    assert result["detector_name"] == "domain_classifier"
    assert result["effect_size"] is not None
    assert result["pvalue_adj"] is not None
    assert result["severity"] in {"warning", "critical"}


def test_analyze_mmd_drift_detects_joint_shift() -> None:
    result = drift_job.analyze_mmd_drift(
        _reference_feature_frame(),
        _current_feature_frame(),
    )

    assert result["feature_name"] == "__multivariate__"
    assert result["detector_name"] == "mmd"
    assert result["effect_size"] is not None
    assert result["pvalue_adj"] is not None
    assert result["drift_detected"] is True


def test_advanced_score_detectors_fire_on_strong_shift() -> None:
    reference_scores = np.linspace(0.05, 0.30, 120)
    current_scores = np.linspace(0.75, 0.95, 120)

    detector_results = {
        result["detector_name"]: result
        for result in [
            drift_job.analyze_score_wasserstein(
                reference_scores, current_scores
            ),
            drift_job.analyze_score_cusum(reference_scores, current_scores),
            drift_job.analyze_score_ewma(reference_scores, current_scores),
            drift_job.analyze_score_adwin(reference_scores, current_scores),
            drift_job.analyze_score_ddm(reference_scores, current_scores),
            drift_job.analyze_score_eddm(reference_scores, current_scores),
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
    assert detector_results["adwin"]["details"]["detection_index"] is not None
    assert detector_results["ddm"]["details"]["drift_index"] is not None
    assert detector_results["eddm"]["details"]["drift_index"] is not None


def test_run_drift_job_reports_advanced_detector_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_window = _current_feature_frame()[["age", "job"]].copy()
    current_window["__score"] = np.linspace(0.75, 0.95, len(current_window))
    current_window["__pred_label"] = 1

    captured_results: dict[str, list[dict[str, object]]] = {}

    monkeypatch.setattr(
        drift_job, "create_engine", lambda *args, **kwargs: DummyEngine()
    )
    monkeypatch.setattr(
        drift_job, "insert_monitoring_run", lambda *args, **kwargs: 101
    )
    monkeypatch.setattr(drift_job, "finalize_monitoring_run", lambda **_: None)
    monkeypatch.setattr(
        drift_job,
        "insert_drift_metrics",
        lambda engine, run_id, results: captured_results.setdefault(
            "rows", list(results)
        ),
    )
    monkeypatch.setattr(
        drift_job, "sync_monitoring_incident", lambda *a, **k: None
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
        lambda: _reference_feature_frame()[["age", "job"]].copy(),
    )
    monkeypatch.setattr(
        drift_job, "load_current_window", lambda *a, **k: current_window.copy()
    )
    monkeypatch.setattr(drift_job, "load_model", lambda: DummyScoreModel())

    result = drift_job.run_drift_job(
        window_size=20,
        min_rows=10,
        segment_key="unit-drift",
    )

    detector_names = {
        str(item["detector_name"]) for item in captured_results["rows"]
    }
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

    advanced_summary = result["summary"]["advanced_drift_detectors"]
    assert {str(item["detector_name"]) for item in advanced_summary} == {
        "mmd",
        "wasserstein",
        "cusum",
        "ewma",
        "adwin",
        "ddm",
        "eddm",
    }
    assert all(
        bool(item["drift_detected"]) is True for item in advanced_summary
    )
