from __future__ import annotations

from app.common.config import settings
from app.monitoring import drift_job, quality_job, scheduler
from app.monitoring.monitoring_config import monitoring_config


def test_monitoring_config_file_exists_and_loads_thresholds() -> None:
    assert settings.monitoring_config_path.exists()

    assert (
        monitoring_config.get_float(
            ("drift", "univariate", "psi_warning"), 0.0
        )
        == drift_job.PSI_WARNING_THRESHOLD
    )
    assert (
        monitoring_config.get_float(
            ("quality", "degradation_rules", "score_psi", "threshold"),
            0.0,
        )
        == quality_job.DEGRADATION_RULES["score_psi"]["threshold"]
    )


def test_monitoring_job_cli_defaults_come_from_config() -> None:
    drift_args = drift_job.build_argument_parser().parse_args([])
    quality_args = quality_job.build_argument_parser().parse_args([])
    scheduler_args = scheduler.build_argument_parser().parse_args([])

    assert drift_args.window_size == monitoring_config.get_int(
        ("drift", "window_size"), 0
    )
    assert quality_args.window_size == monitoring_config.get_int(
        ("quality", "window_size"), 0
    )
    assert scheduler_args.drift_interval_sec == monitoring_config.get_float(
        ("scheduler", "drift_interval_sec"), 0.0
    )
