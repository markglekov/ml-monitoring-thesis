from __future__ import annotations

from app.api import main
from app.common import metrics


def _full_feature_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "age": 42,
        "job": "admin.",
        "marital": "single",
        "education": "secondary",
        "default": "no",
        "balance": 1200.0,
        "housing": "yes",
        "loan": "no",
        "contact": "telephone",
        "day_of_week": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }
    payload.update(overrides)
    return payload


def test_refresh_data_quality_feature_gauges_exports_recent_rates() -> None:
    metrics._refresh_data_quality_feature_gauges(
        model_version=main.settings.model_version,
        inference_feature_rows=[
            _full_feature_payload(),
            _full_feature_payload(
                age=120,
                job=None,
                contact="satellite",
            ),
        ],
    )
    exposition = metrics.render_metrics().decode("utf-8")

    assert (
        "ml_monitoring_data_quality_feature_missing_rate"
        '{feature_name="job",model_version="bank_marketing_v1"} 0.5'
        in exposition
    )
    assert (
        "ml_monitoring_data_quality_numeric_out_of_range_rate"
        '{feature_name="age",model_version="bank_marketing_v1"} 0.5'
        in exposition
    )
    assert (
        "ml_monitoring_data_quality_unknown_category_rate"
        '{feature_name="contact",model_version="bank_marketing_v1"} 0.5'
        in exposition
    )
