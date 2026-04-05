from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from app.monitoring import scheduler


def test_validate_args_rejects_disabling_all_jobs() -> None:
    parser = scheduler.build_argument_parser()
    args = argparse.Namespace(
        segment_key=None,
        poll_interval_sec=5.0,
        drift_interval_sec=300.0,
        quality_interval_sec=300.0,
        drift_window_size=300,
        quality_window_size=300,
        drift_min_rows=50,
        quality_min_rows=50,
        quality_baseline_source="test",
        skip_drift=True,
        skip_quality=True,
        max_job_runs=0,
        run_on_start=True,
    )

    with pytest.raises(SystemExit):
        scheduler.validate_args(args, parser)


def test_build_jobs_respects_enabled_flags_and_run_on_start(
    monkeypatch,
) -> None:
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: 100.0)
    args = argparse.Namespace(
        segment_key="segment-a",
        poll_interval_sec=5.0,
        drift_interval_sec=300.0,
        quality_interval_sec=600.0,
        drift_window_size=200,
        quality_window_size=150,
        drift_min_rows=40,
        quality_min_rows=30,
        quality_baseline_source="validation",
        skip_drift=False,
        skip_quality=False,
        max_job_runs=0,
        run_on_start=True,
    )

    jobs = scheduler.build_jobs(args)

    assert [job.name for job in jobs] == [
        "drift[segment-a]",
        "quality[segment-a]",
    ]
    assert [job.next_run_at for job in jobs] == [100.0, 100.0]


def test_build_jobs_runners_delegate_to_monitoring_jobs(monkeypatch) -> None:
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: 50.0)
    monkeypatch.setattr(
        scheduler,
        "run_drift_job",
        lambda window_size, min_rows, segment_key: {
            "job": "drift",
            "window_size": window_size,
            "min_rows": min_rows,
            "segment_key": segment_key,
        },
    )
    monkeypatch.setattr(
        scheduler,
        "run_quality_job",
        lambda window_size, min_rows, baseline_source, segment_key: {
            "job": "quality",
            "window_size": window_size,
            "min_rows": min_rows,
            "baseline_source": baseline_source,
            "segment_key": segment_key,
        },
    )
    args = argparse.Namespace(
        segment_key="segment-b",
        poll_interval_sec=5.0,
        drift_interval_sec=10.0,
        quality_interval_sec=20.0,
        drift_window_size=123,
        quality_window_size=456,
        drift_min_rows=11,
        quality_min_rows=22,
        quality_baseline_source="validation",
        skip_drift=False,
        skip_quality=False,
        max_job_runs=0,
        run_on_start=False,
    )

    drift_job, quality_job = scheduler.build_jobs(args)

    assert drift_job.runner() == {
        "job": "drift",
        "window_size": 123,
        "min_rows": 11,
        "segment_key": "segment-b",
    }
    assert quality_job.runner() == {
        "job": "quality",
        "window_size": 456,
        "min_rows": 22,
        "baseline_source": "validation",
        "segment_key": "segment-b",
    }
    assert drift_job.next_run_at == 60.0
    assert quality_job.next_run_at == 70.0


def test_build_jobs_uses_configured_segments_when_cli_segment_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(
        scheduler,
        "settings",
        SimpleNamespace(
            monitoring_segments=("segment-a", "segment-b"),
            scheduler_include_global_segment=True,
        ),
    )
    args = argparse.Namespace(
        segment_key=None,
        poll_interval_sec=5.0,
        drift_interval_sec=30.0,
        quality_interval_sec=60.0,
        drift_window_size=100,
        quality_window_size=100,
        drift_min_rows=20,
        quality_min_rows=20,
        quality_baseline_source="test",
        skip_drift=False,
        skip_quality=True,
        max_job_runs=0,
        run_on_start=True,
    )

    jobs = scheduler.build_jobs(args)

    assert [job.name for job in jobs] == [
        "drift",
        "drift[segment-a]",
        "drift[segment-b]",
    ]
