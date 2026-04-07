from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.reporting import build_scenario_summary


def test_discover_scenario_segments_reads_unique_manifest_keys(
    tmp_path: Path,
) -> None:
    baseline_manifest_dir = tmp_path / "baseline" / "manifests"
    baseline_manifest_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "segment_key": [
                "final_demo_baseline",
                "final_demo_baseline",
            ]
        }
    ).to_csv(baseline_manifest_dir / "quality_manifest.csv", index=False)

    segment_manifest_dir = tmp_path / "segment" / "manifests"
    segment_manifest_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "segment_key": [
                "final_demo_segment_stable",
                "final_demo_segment_hot",
            ]
        }
    ).to_csv(segment_manifest_dir / "stable_manifest.csv", index=False)
    pd.DataFrame(
        {
            "segment_key": [
                "final_demo_segment_hot",
                "final_demo_segment_hot",
            ]
        }
    ).to_csv(segment_manifest_dir / "hot_manifest.csv", index=False)

    (tmp_path / "_meta").mkdir()
    (tmp_path / "screenshots").mkdir()

    result = build_scenario_summary.discover_scenario_segments(tmp_path)

    assert result["baseline"] == ["final_demo_baseline"]
    assert set(result["segment"]) == {
        "final_demo_segment_stable",
        "final_demo_segment_hot",
    }


def test_select_scenario_status_prefers_more_important_states() -> None:
    assert (
        build_scenario_summary.select_scenario_status(
            ["completed", "completed_proxy"]
        )
        == "completed_proxy"
    )
    assert (
        build_scenario_summary.select_scenario_status(["completed", "failed"])
        == "failed"
    )
    assert build_scenario_summary.select_scenario_status([]) == "missing"


def test_summarize_scenario_aggregates_latest_signals(
    monkeypatch,
) -> None:
    drift_rows = {
        "segment_a": {
            "id": 1,
            "status": "completed",
            "overall_drift": False,
            "drifted_features_count": 0,
        },
        "segment_b": {
            "id": 2,
            "status": "completed",
            "overall_drift": True,
            "drifted_features_count": 7,
        },
    }
    quality_rows = {
        "segment_a": {
            "id": 11,
            "status": "completed",
            "degraded_metrics_count": 1,
        },
        "segment_b": {
            "id": 12,
            "status": "completed_proxy",
            "degraded_metrics_count": 4,
        },
    }
    incident_counts = {"segment_a": 1, "segment_b": 2}
    estimate_gaps = {"segment_a": None, "segment_b": 0.1685}

    monkeypatch.setattr(
        build_scenario_summary,
        "latest_drift_run",
        lambda engine, *, segment_key: drift_rows.get(segment_key),
    )
    monkeypatch.setattr(
        build_scenario_summary,
        "latest_quality_run",
        lambda engine, *, segment_key: quality_rows.get(segment_key),
    )
    monkeypatch.setattr(
        build_scenario_summary,
        "count_active_incidents",
        lambda engine, *, segment_key: incident_counts[segment_key],
    )
    monkeypatch.setattr(
        build_scenario_summary,
        "latest_estimate_gap",
        lambda engine, *, segment_key: estimate_gaps[segment_key],
    )

    row = build_scenario_summary.summarize_scenario(
        engine=None,  # type: ignore[arg-type]
        scenario="proxy",
        segment_keys=["segment_a", "segment_b"],
    )

    assert row.scenario == "proxy"
    assert row.overall_drift is True
    assert row.drifted_features_count == 7
    assert row.degraded_metrics_count == 4
    assert row.status == "completed_proxy"
    assert row.active_incidents == 3
    assert row.estimated_vs_true_metric_gap == 0.1685
