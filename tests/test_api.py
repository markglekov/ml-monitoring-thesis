from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pytest
from fastapi import HTTPException

from app.api import main


class DummyModel:
    def predict_proba(self, X):
        score = 0.9 if float(X.iloc[0]["age"]) >= 30 else 0.1
        return np.array([[1.0 - score, score]])


def configure_app_state() -> None:
    main.app.state.model = DummyModel()
    main.app.state.engine = object()
    main.app.state.feature_columns = ["age", "job"]
    main.app.state.threshold = 0.5


def test_build_inference_frame_normalizes_nulls() -> None:
    frame = main.build_inference_frame(
        {"age": 33, "job": None}, ["age", "job"]
    )

    assert frame.shape == (1, 2)
    assert frame.iloc[0]["age"] == 33
    assert np.isnan(frame.iloc[0]["job"])


def test_predict_returns_prediction_response(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_insert_inference_log(**kwargs):
        captured.update(kwargs)

    def fake_record_prediction(
        model_version: str, predicted_label: int, score: float
    ) -> None:
        captured["metrics"] = {
            "model_version": model_version,
            "predicted_label": predicted_label,
            "score": score,
        }

    configure_app_state()
    monkeypatch.setattr(
        main, "insert_inference_log", fake_insert_inference_log
    )
    monkeypatch.setattr(main, "record_prediction", fake_record_prediction)

    response = main.predict(
        main.PredictRequest(
            request_id="f8d3ce55-c24d-48cc-a8ef-4d71ea2d6fe8",
            segment_key="smoke",
            features={"age": 42, "job": "admin"},
        )
    )

    assert response.request_id == "f8d3ce55-c24d-48cc-a8ef-4d71ea2d6fe8"
    assert response.predicted_label == 1
    assert response.segment_key == "smoke"
    assert response.threshold == 0.5
    assert captured["pred_label"] == 1
    assert captured["segment_key"] == "smoke"
    assert captured["metrics"]["predicted_label"] == 1


def test_predict_rejects_feature_schema_mismatch() -> None:
    configure_app_state()

    with pytest.raises(HTTPException) as exc_info:
        main.predict(
            main.PredictRequest(features={"age": 42, "unexpected": "value"})
        )

    detail = cast(dict[str, Any], exc_info.value.detail)
    assert exc_info.value.status_code == 422
    assert detail["missing_features"] == ["job"]
    assert detail["extra_features"] == ["unexpected"]


def test_ingest_labels_batch_rejects_duplicate_request_ids() -> None:
    configure_app_state()

    with pytest.raises(HTTPException) as exc_info:
        main.ingest_labels_batch(
            main.GroundTruthLabelsBatchRequest(
                labels=[
                    main.GroundTruthLabelRequest(
                        request_id="dup-id", y_true=1
                    ),
                    main.GroundTruthLabelRequest(
                        request_id="dup-id", y_true=0
                    ),
                ]
            )
        )

    detail = cast(dict[str, Any], exc_info.value.detail)
    assert exc_info.value.status_code == 422
    assert detail["message"] == "Duplicate request_id values in batch payload."
    assert detail["request_ids"] == ["dup-id"]


def test_ingest_labels_batch_upserts_labels(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_find_missing_request_ids(
        engine: object, request_ids: list[str]
    ) -> list[str]:
        captured["request_ids"] = request_ids
        return []

    def fake_upsert_ground_truth_labels(
        engine: object, labels: list[main.GroundTruthLabelRequest]
    ) -> int:
        captured["labels"] = labels
        return len(labels)

    def fake_record_labels_upserted(model_version: str, count: int) -> None:
        captured["metrics"] = (model_version, count)

    configure_app_state()
    monkeypatch.setattr(
        main, "find_missing_request_ids", fake_find_missing_request_ids
    )
    monkeypatch.setattr(
        main, "upsert_ground_truth_labels", fake_upsert_ground_truth_labels
    )
    monkeypatch.setattr(
        main, "record_labels_upserted", fake_record_labels_upserted
    )

    response = main.ingest_labels_batch(
        main.GroundTruthLabelsBatchRequest(
            labels=[
                main.GroundTruthLabelRequest(
                    request_id="11111111-1111-1111-1111-111111111111", y_true=1
                ),
                main.GroundTruthLabelRequest(
                    request_id="22222222-2222-2222-2222-222222222222", y_true=0
                ),
            ]
        )
    )

    assert response.received_count == 2
    assert response.upserted_count == 2
    assert response.status == "upserted"
    assert captured["request_ids"] == [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]
    assert captured["metrics"] == (main.settings.model_version, 2)


def test_monitoring_overview_marks_attention_on_quality_degradation(
    monkeypatch,
) -> None:
    main.app.state.engine = object()

    monkeypatch.setattr(
        main,
        "health",
        lambda: {
            "status": "ok",
            "model_loaded": True,
            "model_version": main.settings.model_version,
            "feature_count": 2,
            "threshold": 0.5,
        },
    )
    monkeypatch.setattr(
        main,
        "get_overview_data_snapshot",
        lambda engine: main.OverviewDataSnapshotResponse(
            total_predictions=10,
            total_labels=5,
            labeled_coverage=0.5,
            predictions_last_24h=10,
            labels_last_24h=5,
            latest_inference_ts=datetime.now(UTC),
            latest_label_ts=datetime.now(UTC),
            positive_prediction_rate=0.4,
            positive_label_rate=0.3,
        ),
    )
    monkeypatch.setattr(
        main,
        "get_latest_drift_run",
        lambda engine: main.DriftRunResponse(
            id=1,
            ts_started=datetime.now(UTC),
            ts_finished=datetime.now(UTC),
            model_version=main.settings.model_version,
            window_size=100,
            segment_key=None,
            status="completed",
            drifted_features_count=0,
            total_features_count=5,
            overall_drift=False,
            summary={"ok": True},
        ),
    )
    monkeypatch.setattr(
        main,
        "get_latest_quality_run",
        lambda engine: main.QualityRunResponse(
            id=2,
            ts_started=datetime.now(UTC),
            ts_finished=datetime.now(UTC),
            model_version=main.settings.model_version,
            window_size=100,
            segment_key=None,
            status="completed",
            labeled_rows=80,
            degraded_metrics_count=2,
            summary={
                "degraded_metrics": ["f1", "roc_auc"],
                "recommended_action": "Inspect labeled degradation.",
            },
        ),
    )
    monkeypatch.setattr(main, "get_top_segments", lambda engine: [])
    monkeypatch.setattr(
        main,
        "get_active_incident_responses",
        lambda engine: [
            main.MonitoringIncidentResponse(
                id=3,
                incident_key="quality:__global__",
                source_type="quality",
                model_version=main.settings.model_version,
                segment_key=None,
                status="open",
                severity="critical",
                title="Quality degradation detected",
                recommended_action="Inspect labeled degradation.",
                summary={"severity": "critical"},
                latest_run_id=2,
                acknowledged_by=None,
                mitigation_taken=None,
                ts_opened=datetime.now(UTC),
                ts_updated=datetime.now(UTC),
                ts_resolved=None,
            )
        ],
    )

    response = main.get_monitoring_overview_payload(main.app)

    assert response.service_status == "attention"
    assert response.severity == "critical"
    assert response.active_incidents_count == 1
    assert response.recommended_action == "Inspect labeled degradation."
