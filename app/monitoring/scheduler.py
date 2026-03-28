from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.logging import get_logger, setup_logging
from app.monitoring.drift_job import run_drift_job
from app.monitoring.quality_job import run_quality_job


logger = get_logger(__name__)


@dataclass
class ScheduledJob:
    """One recurring background job managed by the monitoring scheduler."""

    name: str
    interval_sec: float
    runner: Callable[[], dict[str, Any]]
    next_run_at: float


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the monitoring scheduler."""

    parser = argparse.ArgumentParser(description="Recurring scheduler for drift and quality monitoring jobs")
    parser.add_argument("--segment-key", type=str, default=None)
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    parser.add_argument("--drift-interval-sec", type=float, default=300.0)
    parser.add_argument("--quality-interval-sec", type=float, default=300.0)
    parser.add_argument("--drift-window-size", type=int, default=300)
    parser.add_argument("--quality-window-size", type=int, default=300)
    parser.add_argument("--drift-min-rows", type=int, default=50)
    parser.add_argument("--quality-min-rows", type=int, default=50)
    parser.add_argument("--quality-baseline-source", type=str, choices=["test", "validation"], default="test")
    parser.add_argument("--skip-drift", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--max-job-runs", type=int, default=0)
    parser.add_argument(
        "--no-run-on-start",
        action="store_false",
        dest="run_on_start",
        help="Wait one full interval before the first execution.",
    )
    parser.set_defaults(run_on_start=True)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate scheduler configuration before the loop starts."""

    if args.skip_drift and args.skip_quality:
        parser.error("at least one job must be enabled")
    if args.poll_interval_sec <= 0:
        parser.error("--poll-interval-sec must be greater than 0")
    if args.drift_interval_sec <= 0:
        parser.error("--drift-interval-sec must be greater than 0")
    if args.quality_interval_sec <= 0:
        parser.error("--quality-interval-sec must be greater than 0")
    if args.drift_window_size <= 0:
        parser.error("--drift-window-size must be greater than 0")
    if args.quality_window_size <= 0:
        parser.error("--quality-window-size must be greater than 0")
    if args.drift_min_rows <= 0:
        parser.error("--drift-min-rows must be greater than 0")
    if args.quality_min_rows <= 0:
        parser.error("--quality-min-rows must be greater than 0")
    if args.max_job_runs < 0:
        parser.error("--max-job-runs must be 0 or greater")


def build_jobs(args: argparse.Namespace) -> list[ScheduledJob]:
    """Create the enabled recurring jobs with their initial due times."""

    start_at = time.monotonic()
    initial_offset = 0.0 if args.run_on_start else None
    jobs: list[ScheduledJob] = []

    if not args.skip_drift:
        jobs.append(
            ScheduledJob(
                name="drift",
                interval_sec=float(args.drift_interval_sec),
                next_run_at=start_at if initial_offset == 0.0 else start_at + float(args.drift_interval_sec),
                runner=lambda: run_drift_job(
                    window_size=int(args.drift_window_size),
                    min_rows=int(args.drift_min_rows),
                    segment_key=args.segment_key,
                ),
            )
        )

    if not args.skip_quality:
        jobs.append(
            ScheduledJob(
                name="quality",
                interval_sec=float(args.quality_interval_sec),
                next_run_at=start_at if initial_offset == 0.0 else start_at + float(args.quality_interval_sec),
                runner=lambda: run_quality_job(
                    window_size=int(args.quality_window_size),
                    min_rows=int(args.quality_min_rows),
                    baseline_source=args.quality_baseline_source,
                    segment_key=args.segment_key,
                ),
            )
        )

    return jobs


def execute_job(job: ScheduledJob) -> dict[str, Any]:
    """Run one scheduled job and log its result payload."""

    started_at = time.perf_counter()
    result = job.runner()
    duration_sec = time.perf_counter() - started_at

    logger.info(
        "Scheduled %s job finished. run_id=%s status=%s duration_sec=%.3f",
        job.name,
        result.get("run_id"),
        result.get("status"),
        duration_sec,
    )
    return result


def run_scheduler(args: argparse.Namespace) -> None:
    """Run the monitoring scheduler loop until interrupted or max-job-runs is reached."""

    jobs = build_jobs(args)
    total_runs = 0

    logger.info(
        "Starting monitoring scheduler. jobs=%s segment_key=%s run_on_start=%s max_job_runs=%s",
        [job.name for job in jobs],
        args.segment_key,
        args.run_on_start,
        args.max_job_runs,
    )

    while True:
        now = time.monotonic()
        executed_any = False

        for job in jobs:
            if now < job.next_run_at:
                continue

            executed_any = True
            try:
                execute_job(job)
            except Exception:
                logger.exception("Scheduled %s job failed.", job.name)
            finally:
                job.next_run_at = time.monotonic() + job.interval_sec
                total_runs += 1

            if args.max_job_runs and total_runs >= args.max_job_runs:
                logger.info("Stopping monitoring scheduler after max_job_runs=%s", args.max_job_runs)
                return

        if executed_any:
            continue

        next_run_at = min(job.next_run_at for job in jobs)
        sleep_sec = max(0.1, min(float(args.poll_interval_sec), next_run_at - time.monotonic()))
        time.sleep(sleep_sec)


def main() -> None:
    """CLI entrypoint for the recurring monitoring scheduler."""

    setup_logging()

    parser = build_argument_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    try:
        run_scheduler(args)
    except KeyboardInterrupt:
        logger.info("Monitoring scheduler interrupted. Shutting down.")


if __name__ == "__main__":
    main()
