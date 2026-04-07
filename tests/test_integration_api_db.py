from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import text

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


def _mock_data_quality_feature_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        metrics,
        "_load_baseline_feature_profiles",
        lambda: {
            "age": {"type": "numeric", "min": 20.0, "max": 61.0},
            "job": {
                "type": "categorical",
                "top_values": {"admin.": 0.2, "student": 0.1},
            },
            "contact": {
                "type": "categorical",
                "top_values": {"telephone": 0.2, "cellular": 0.7},
            },
        },
    )


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
        row = (
            connection.execute(
                text(
                    """
                SELECT
                    request_id::text AS request_id,
                    model_version,
                    score,
                    pred_label,
                    segment_key
                FROM inference_log
                """
                )
            )
            .mappings()
            .one()
        )

    assert row["request_id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert row["model_version"] == main.settings.model_version
    assert float(row["score"]) == 0.91
    assert int(row["pred_label"]) == 1
    assert row["segment_key"] == "segment-a"


@pytest.mark.integration
def test_upsert_ground_truth_labels_updates_existing_row(
    postgres_engine,
) -> None:
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
                label_ts=datetime(2026, 3, 28, 10, 0, tzinfo=UTC),
            )
        ],
    )
    updated = main.upsert_ground_truth_labels(
        postgres_engine,
        [
            main.GroundTruthLabelRequest(
                request_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                y_true=0,
                label_ts=datetime(2026, 3, 28, 12, 0, tzinfo=UTC),
            )
        ],
    )

    with postgres_engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT y_true, label_ts
                    FROM ground_truth
                    WHERE request_id::text = :request_id
                    """
                ),
                {"request_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"},
            )
            .mappings()
            .one()
        )

    assert inserted == 1
    assert updated == 1
    assert int(row["y_true"]) == 0
    assert row["label_ts"] == datetime(2026, 3, 28, 12, 0, tzinfo=UTC)


@pytest.mark.integration
def test_get_monitoring_overview_payload_reads_real_database(
    monkeypatch, postgres_engine
) -> None:
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
                label_ts=datetime(2026, 3, 28, 13, 0, tzinfo=UTC),
            )
        ],
    )

    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO monitoring_runs (
                    model_version, window_size, segment_key, status,
                    drifted_features_count, total_features_count,
                    overall_drift, summary_json
                )
                VALUES (
                    :model_version, 100, 'segment-c', 'completed',
                    0, 10, false, CAST(:summary_json AS JSONB)
                )
                """
            ),
            {
                "model_version": main.settings.model_version,
                "summary_json": '{"ok": true}',
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO quality_runs (
                    model_version, window_size, segment_key, status,
                    labeled_rows, degraded_metrics_count, summary_json
                )
                VALUES (
                    :model_version, 100, 'segment-c', 'completed',
                    1, 0, CAST(:summary_json AS JSONB)
                )
                """
            ),
            {
                "model_version": main.settings.model_version,
                "summary_json": '{"ok": true}',
            },
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
def test_refresh_monitoring_gauges_reads_latest_run_rows(
    postgres_engine,
) -> None:
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
        quality_run_id = connection.execute(
            text(
                """
                SELECT id
                FROM quality_runs
                ORDER BY id DESC
                LIMIT 1
                """
            )
        ).scalar_one()
        connection.execute(
            text(
                """
                INSERT INTO quality_metrics (
                    run_id,
                    segment_key,
                    metric_name,
                    metric_value,
                    baseline_value,
                    delta_value,
                    degradation_detected,
                    details_json
                )
                VALUES
                    (
                        :run_id,
                        NULL,
                        'roc_auc',
                        0.671,
                        0.701,
                        -0.03,
                        true,
                        CAST(:details_json AS JSONB)
                    ),
                    (
                        :run_id,
                        NULL,
                        'f1',
                        0.53,
                        0.58,
                        -0.05,
                        true,
                        CAST(:details_json AS JSONB)
                    )
                """
            ),
            {"run_id": quality_run_id, "details_json": '{"ok": true}'},
        )

    metrics.refresh_monitoring_gauges(
        postgres_engine, model_version=main.settings.model_version
    )
    exposition = metrics.render_metrics().decode("utf-8")

    assert (
        "ml_monitoring_drift_last_run_drifted_features"
        '{model_version="bank_marketing_v1"} 3.0' in exposition
    )
    assert (
        "ml_monitoring_drift_last_run_overall"
        '{model_version="bank_marketing_v1"} 1.0' in exposition
    )
    assert (
        "ml_monitoring_quality_last_run_degraded_metrics"
        '{model_version="bank_marketing_v1"} 2.0' in exposition
    )
    assert (
        "ml_monitoring_quality_last_run_labeled_rows"
        '{model_version="bank_marketing_v1"} 42.0' in exposition
    )
    assert (
        "ml_monitoring_quality_last_metric_value"
        '{metric_name="roc_auc",model_version="bank_marketing_v1"} 0.671'
        in exposition
    )
    assert (
        "ml_monitoring_quality_last_metric_value"
        '{metric_name="f1",model_version="bank_marketing_v1"} 0.53'
        in exposition
    )


@pytest.mark.integration
def test_refresh_monitoring_gauges_exports_segment_metrics(
    postgres_engine,
) -> None:
    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO monitoring_runs (
                    model_version, window_size, segment_key, status,
                    drifted_features_count, total_features_count,
                    overall_drift, summary_json
                )
                VALUES (
                    :model_version,
                    300,
                    'segment-a',
                    'completed',
                    2,
                    10,
                    true,
                    CAST(:summary_json AS JSONB)
                )
                """
            ),
            {
                "model_version": main.settings.model_version,
                "summary_json": '{"severity": "warning"}',
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO quality_runs (
                    model_version, window_size, segment_key, status,
                    labeled_rows, degraded_metrics_count, summary_json
                )
                VALUES (
                    :model_version,
                    300,
                    'segment-a',
                    'completed_proxy',
                    0,
                    1,
                    CAST(:summary_json AS JSONB)
                )
                """
            ),
            {
                "model_version": main.settings.model_version,
                "summary_json": '{"severity": "warning"}',
            },
        )
        quality_run_id = connection.execute(
            text(
                """
                SELECT id
                FROM quality_runs
                WHERE segment_key = 'segment-a'
                ORDER BY id DESC
                LIMIT 1
                """
            )
        ).scalar_one()
        connection.execute(
            text(
                """
                INSERT INTO quality_metrics (
                    run_id,
                    segment_key,
                    metric_name,
                    metric_value,
                    baseline_value,
                    delta_value,
                    detector_name,
                    severity,
                    degradation_detected,
                    details_json
                )
                VALUES (
                    :run_id,
                    'segment-a',
                    'score_psi',
                    0.31,
                    0.02,
                    0.29,
                    'proxy',
                    'warning',
                    true,
                    CAST(:details_json AS JSONB)
                )
                """
            ),
            {"run_id": quality_run_id, "details_json": '{"ok": true}'},
        )
        connection.execute(
            text(
                """
                INSERT INTO monitoring_incidents (
                    incident_key,
                    source_type,
                    model_version,
                    segment_key,
                    status,
                    severity,
                    title,
                    recommended_action,
                    summary_json,
                    latest_run_id
                )
                VALUES (
                    'quality:segment-a',
                    'quality',
                    :model_version,
                    'segment-a',
                    'open',
                    'warning',
                    'Proxy quality warning',
                    'Collect labels and review the segment.',
                    CAST(:summary_json AS JSONB),
                    :latest_run_id
                )
                """
            ),
            {
                "model_version": main.settings.model_version,
                "summary_json": '{"metric_name": "score_psi"}',
                "latest_run_id": quality_run_id,
            },
        )

    metrics.refresh_monitoring_gauges(
        postgres_engine, model_version=main.settings.model_version
    )
    exposition = metrics.render_metrics().decode("utf-8")

    assert (
        "ml_monitoring_drift_segment_last_run_drifted_features"
        '{model_version="bank_marketing_v1",segment_key="segment-a"} 2.0'
        in exposition
    )
    assert (
        "ml_monitoring_quality_segment_last_run_status"
        '{model_version="bank_marketing_v1",segment_key="segment-a",'
        'status="completed_proxy"} 1.0' in exposition
    )
    assert (
        "ml_monitoring_quality_segment_last_metric_value"
        '{metric_name="score_psi",model_version="bank_marketing_v1",'
        'segment_key="segment-a"} 0.31' in exposition
    )
    assert (
        "ml_monitoring_active_incidents_by_segment"
        '{model_version="bank_marketing_v1",segment_key="segment-a",'
        'severity="warning",source_type="quality"} 1.0' in exposition
    )


@pytest.mark.integration
def test_refresh_monitoring_gauges_exports_recent_data_quality_metrics(
    postgres_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_data_quality_feature_profiles(monkeypatch)

    main.insert_inference_log(
        engine=postgres_engine,
        request_id="dddddddd-dddd-dddd-dddd-dddddddddddd",
        features=_full_feature_payload(),
        score=0.88,
        pred_label=1,
        threshold=0.5,
        segment_key="segment-d",
        latency_ms=9.0,
    )
    main.insert_inference_log(
        engine=postgres_engine,
        request_id="eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
        features=_full_feature_payload(
            age=120,
            job=None,
            contact="satellite",
        ),
        score=0.21,
        pred_label=0,
        threshold=0.5,
        segment_key="segment-d",
        latency_ms=11.0,
    )

    metrics.refresh_monitoring_gauges(
        postgres_engine, model_version=main.settings.model_version
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
