from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from app.api import main
from app.common import metrics


@pytest.mark.integration
def test_insert_inference_log_persists_row(postgres_engine) -> None:
    main.insert_inference_log(
        engine=postgres_engine,
        request_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        features={"age": 42, "job": "admin"},
        score=0.91,
        pred_label=1,
        threshold=0.5,
        segment_key="segment-a",
        latency_ms=12.5,
    )

    with postgres_engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT request_id::text AS request_id, model_version, score, pred_label, segment_key
                FROM inference_log
                """
            )
        ).mappings().one()

    assert row["request_id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert row["model_version"] == main.settings.model_version
    assert float(row["score"]) == 0.91
    assert int(row["pred_label"]) == 1
    assert row["segment_key"] == "segment-a"


@pytest.mark.integration
def test_upsert_ground_truth_labels_updates_existing_row(postgres_engine) -> None:
    main.insert_inference_log(
        engine=postgres_engine,
        request_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        features={"age": 27, "job": "student"},
        score=0.14,
        pred_label=0,
        threshold=0.5,
        segment_key="segment-b",
        latency_ms=8.0,
    )

    inserted = main.upsert_ground_truth_labels(
        postgres_engine,
        [
            main.GroundTruthLabelRequest(
                request_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                y_true=1,
                label_ts=datetime(2026, 3, 28, 10, 0, tzinfo=timezone.utc),
            )
        ],
    )
    updated = main.upsert_ground_truth_labels(
        postgres_engine,
        [
            main.GroundTruthLabelRequest(
                request_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                y_true=0,
                label_ts=datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc),
            )
        ],
    )

    with postgres_engine.connect() as connection:
        row = connection.execute(
            text("SELECT y_true, label_ts FROM ground_truth WHERE request_id::text = :request_id"),
            {"request_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"},
        ).mappings().one()

    assert inserted == 1
    assert updated == 1
    assert int(row["y_true"]) == 0
    assert row["label_ts"] == datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc)


@pytest.mark.integration
def test_get_monitoring_overview_payload_reads_real_database(monkeypatch, postgres_engine) -> None:
    main.insert_inference_log(
        engine=postgres_engine,
        request_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
        features={"age": 31, "job": "admin"},
        score=0.88,
        pred_label=1,
        threshold=0.5,
        segment_key="segment-c",
        latency_ms=10.0,
    )
    main.insert_inference_log(
        engine=postgres_engine,
        request_id="dddddddd-dddd-dddd-dddd-dddddddddddd",
        features={"age": 22, "job": "student"},
        score=0.12,
        pred_label=0,
        threshold=0.5,
        segment_key="segment-c",
        latency_ms=7.0,
    )
    main.upsert_ground_truth_labels(
        postgres_engine,
        [
            main.GroundTruthLabelRequest(
                request_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
                y_true=1,
                label_ts=datetime(2026, 3, 28, 13, 0, tzinfo=timezone.utc),
            )
        ],
    )

    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO monitoring_runs (
                    model_version, window_size, segment_key, status,
                    drifted_features_count, total_features_count, overall_drift, summary_json
                )
                VALUES (
                    :model_version, 100, 'segment-c', 'completed', 0, 10, false, CAST(:summary_json AS JSONB)
                )
                """
            ),
            {"model_version": main.settings.model_version, "summary_json": '{"ok": true}'},
        )
        connection.execute(
            text(
                """
                INSERT INTO quality_runs (
                    model_version, window_size, segment_key, status,
                    labeled_rows, degraded_metrics_count, summary_json
                )
                VALUES (
                    :model_version, 100, 'segment-c', 'completed', 1, 0, CAST(:summary_json AS JSONB)
                )
                """
            ),
            {"model_version": main.settings.model_version, "summary_json": '{"ok": true}'},
        )

    main.app.state.engine = postgres_engine
    main.app.state.model = object()
    main.app.state.feature_columns = ["age", "job"]
    main.app.state.threshold = 0.5

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

    overview = main.get_monitoring_overview_payload(main.app)

    assert overview.service_status == "ok"
    assert overview.data_snapshot.total_predictions == 2
    assert overview.data_snapshot.total_labels == 1
    assert overview.data_snapshot.labeled_coverage == 0.5
    assert overview.top_segments[0].segment_key == "segment-c"
    assert overview.top_segments[0].inference_rows == 2
    assert overview.top_segments[0].labeled_rows == 1


@pytest.mark.integration
def test_refresh_monitoring_gauges_reads_latest_run_rows(postgres_engine) -> None:
    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO monitoring_runs (
                    model_version, window_size, segment_key, status,
                    drifted_features_count, total_features_count, overall_drift
                )
                VALUES (:model_version, 300, NULL, 'completed', 3, 10, true)
                """
            ),
            {"model_version": main.settings.model_version},
        )
        connection.execute(
            text(
                """
                INSERT INTO quality_runs (
                    model_version, window_size, segment_key, status,
                    labeled_rows, degraded_metrics_count
                )
                VALUES (:model_version, 300, NULL, 'completed', 42, 2)
                """
            ),
            {"model_version": main.settings.model_version},
        )

    metrics.refresh_monitoring_gauges(postgres_engine, model_version=main.settings.model_version)
    exposition = metrics.render_metrics().decode("utf-8")

    assert 'ml_monitoring_drift_last_run_drifted_features{model_version="bank_marketing_v1"} 3.0' in exposition
    assert 'ml_monitoring_drift_last_run_overall{model_version="bank_marketing_v1"} 1.0' in exposition
    assert 'ml_monitoring_quality_last_run_degraded_metrics{model_version="bank_marketing_v1"} 2.0' in exposition
    assert 'ml_monitoring_quality_last_run_labeled_rows{model_version="bank_marketing_v1"} 42.0' in exposition
