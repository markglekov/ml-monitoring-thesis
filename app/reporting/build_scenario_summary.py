from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.common.config import settings
from app.common.logging import get_logger, setup_logging

ROOT = Path(__file__).resolve().parents[2]
FINAL_REPORT_DIR = ROOT / "artifacts" / "reports" / "final"
SCENARIO_ORDER = ("baseline", "mild", "severe", "proxy", "segment")

logger = get_logger(__name__)


@dataclass(frozen=True)
class ScenarioSummaryRow:
    scenario: str
    overall_drift: bool | None
    drifted_features_count: int | None
    degraded_metrics_count: int | None
    status: str
    active_incidents: int
    estimated_vs_true_metric_gap: float | None


def write_json(path: Path, payload: Any) -> None:
    """Write one JSON artifact with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write one CSV artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_text(path: Path, content: str) -> None:
    """Write one text artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def scenario_sort_key(path: Path) -> tuple[int, str]:
    """Sort known scenarios first and keep other names stable."""

    try:
        return (SCENARIO_ORDER.index(path.name), path.name)
    except ValueError:
        return (len(SCENARIO_ORDER), path.name)


def discover_scenario_segments(
    final_report_dir: Path,
) -> dict[str, list[str]]:
    """Discover segment keys per scenario from saved manifests."""

    scenario_segments: dict[str, list[str]] = {}

    for scenario_dir in sorted(
        final_report_dir.iterdir(), key=scenario_sort_key
    ):
        if not scenario_dir.is_dir() or scenario_dir.name.startswith("_"):
            continue
        if scenario_dir.name == "screenshots":
            continue

        manifest_dir = scenario_dir / "manifests"
        if not manifest_dir.exists():
            continue

        segments: list[str] = []
        for manifest_path in sorted(manifest_dir.glob("*.csv")):
            try:
                frame = pd.read_csv(manifest_path, usecols=["segment_key"])
            except ValueError:
                continue
            for segment_key in (
                frame["segment_key"].dropna().astype(str).tolist()
            ):
                if segment_key not in segments:
                    segments.append(segment_key)

        if segments:
            scenario_segments[scenario_dir.name] = segments

    return scenario_segments


def select_scenario_status(statuses: list[str]) -> str:
    """Pick one compact scenario status from latest drift/quality runs."""

    if not statuses:
        return "missing"

    precedence = {
        "failed": 4,
        "completed_proxy": 3,
        "completed": 2,
        "skipped": 1,
        "missing": 0,
    }
    return max(statuses, key=lambda status: precedence.get(status, -1))


def latest_drift_run(
    engine: Engine, *, segment_key: str
) -> dict[str, Any] | None:
    """Load the latest drift run for one segment."""

    query = text(
        """
        SELECT
            id,
            status,
            overall_drift,
            drifted_features_count
        FROM monitoring_runs
        WHERE model_version = :model_version
          AND segment_key = :segment_key
        ORDER BY ts_started DESC, id DESC
        LIMIT 1
        """
    )
    with engine.begin() as connection:
        row = (
            connection.execute(
                query,
                {
                    "model_version": settings.model_version,
                    "segment_key": segment_key,
                },
            )
            .mappings()
            .first()
        )
    return dict(row) if row is not None else None


def latest_quality_run(
    engine: Engine, *, segment_key: str
) -> dict[str, Any] | None:
    """Load the latest quality run for one segment."""

    query = text(
        """
        SELECT
            id,
            status,
            degraded_metrics_count
        FROM quality_runs
        WHERE model_version = :model_version
          AND segment_key = :segment_key
        ORDER BY ts_started DESC, id DESC
        LIMIT 1
        """
    )
    with engine.begin() as connection:
        row = (
            connection.execute(
                query,
                {
                    "model_version": settings.model_version,
                    "segment_key": segment_key,
                },
            )
            .mappings()
            .first()
        )
    return dict(row) if row is not None else None


def count_active_incidents(engine: Engine, *, segment_key: str) -> int:
    """Count currently open incidents for one segment."""

    query = text(
        """
        SELECT COUNT(*) AS incident_count
        FROM monitoring_incidents
        WHERE model_version = :model_version
          AND segment_key = :segment_key
          AND status = 'open'
        """
    )
    with engine.begin() as connection:
        row = (
            connection.execute(
                query,
                {
                    "model_version": settings.model_version,
                    "segment_key": segment_key,
                },
            )
            .mappings()
            .one()
        )
    return int(row["incident_count"])


def latest_estimate_gap(engine: Engine, *, segment_key: str) -> float | None:
    """Compute the max absolute estimate-vs-true gap for one segment."""

    query = text(
        """
        WITH latest_proxy_run AS (
            SELECT id
            FROM quality_runs
            WHERE model_version = :model_version
              AND segment_key = :segment_key
              AND status = 'completed_proxy'
            ORDER BY ts_started DESC, id DESC
            LIMIT 1
        ),
        latest_labeled_run AS (
            SELECT id
            FROM quality_runs
            WHERE model_version = :model_version
              AND segment_key = :segment_key
              AND status = 'completed'
            ORDER BY ts_started DESC, id DESC
            LIMIT 1
        )
        SELECT
            ABS(
                quality_estimates.estimated_metric_value
                - quality_metrics.metric_value
            ) AS abs_gap
        FROM latest_proxy_run
        JOIN quality_estimates
            ON quality_estimates.run_id = latest_proxy_run.id
        JOIN latest_labeled_run
            ON TRUE
        JOIN quality_metrics
            ON quality_metrics.run_id = latest_labeled_run.id
           AND quality_metrics.metric_name
               = quality_estimates.estimated_metric_name
        ORDER BY abs_gap DESC, quality_estimates.estimated_metric_name ASC
        LIMIT 1
        """
    )
    with engine.begin() as connection:
        row = (
            connection.execute(
                query,
                {
                    "model_version": settings.model_version,
                    "segment_key": segment_key,
                },
            )
            .mappings()
            .first()
        )
    if row is None or row["abs_gap"] is None:
        return None
    return float(row["abs_gap"])


def aggregate_optional_int(values: list[int]) -> int | None:
    """Return the worst-case integer signal across one scenario."""

    if not values:
        return None
    return max(values)


def aggregate_optional_float(values: list[float]) -> float | None:
    """Return the worst-case numeric gap across one scenario."""

    if not values:
        return None
    return max(values)


def aggregate_optional_bool(values: list[bool]) -> bool | None:
    """Return whether any segment drifted in the scenario."""

    if not values:
        return None
    return any(values)


def summarize_scenario(
    engine: Engine, *, scenario: str, segment_keys: list[str]
) -> ScenarioSummaryRow:
    """Build one compact evidence row for a saved scenario."""

    drift_runs = [
        run
        for run in (
            latest_drift_run(engine, segment_key=segment_key)
            for segment_key in segment_keys
        )
        if run is not None
    ]
    quality_runs = [
        run
        for run in (
            latest_quality_run(engine, segment_key=segment_key)
            for segment_key in segment_keys
        )
        if run is not None
    ]
    active_incidents = sum(
        count_active_incidents(engine, segment_key=segment_key)
        for segment_key in segment_keys
    )
    estimate_gaps = [
        gap
        for gap in (
            latest_estimate_gap(engine, segment_key=segment_key)
            for segment_key in segment_keys
        )
        if gap is not None
    ]

    return ScenarioSummaryRow(
        scenario=scenario,
        overall_drift=aggregate_optional_bool(
            [bool(run["overall_drift"]) for run in drift_runs]
        ),
        drifted_features_count=aggregate_optional_int(
            [int(run["drifted_features_count"]) for run in drift_runs]
        ),
        degraded_metrics_count=aggregate_optional_int(
            [int(run["degraded_metrics_count"]) for run in quality_runs]
        ),
        status=select_scenario_status(
            [str(run["status"]) for run in drift_runs + quality_runs]
        ),
        active_incidents=active_incidents,
        estimated_vs_true_metric_gap=aggregate_optional_float(estimate_gaps),
    )


def markdown_table(rows: list[ScenarioSummaryRow]) -> str:
    """Build a concise markdown summary table."""

    lines = [
        (
            "| scenario | overall_drift | drifted_features_count | "
            "degraded_metrics_count | status | active_incidents | "
            "estimated_vs_true_metric_gap |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.scenario,
                    ""
                    if row.overall_drift is None
                    else str(row.overall_drift).lower(),
                    ""
                    if row.drifted_features_count is None
                    else str(row.drifted_features_count),
                    ""
                    if row.degraded_metrics_count is None
                    else str(row.degraded_metrics_count),
                    row.status,
                    str(row.active_incidents),
                    ""
                    if row.estimated_vs_true_metric_gap is None
                    else f"{row.estimated_vs_true_metric_gap:.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_scenario_summary(
    *, engine: Engine, final_report_dir: Path
) -> list[ScenarioSummaryRow]:
    """Load saved scenarios and summarize them into one evidence table."""

    scenario_segments = discover_scenario_segments(final_report_dir)
    rows: list[ScenarioSummaryRow] = []
    for scenario in SCENARIO_ORDER:
        segment_keys = scenario_segments.get(scenario)
        if not segment_keys:
            continue
        rows.append(
            summarize_scenario(
                engine,
                scenario=scenario,
                segment_keys=segment_keys,
            )
        )
    return rows


def save_scenario_summary(
    *, rows: list[ScenarioSummaryRow], output_dir: Path
) -> None:
    """Persist the scenario summary as JSON, CSV, and Markdown."""

    payload = [asdict(row) for row in rows]
    frame = pd.DataFrame(payload)
    generated_at = datetime.now(UTC).isoformat()

    write_json(output_dir / "scenario_summary.json", payload)
    write_csv(output_dir / "scenario_summary.csv", frame)
    write_text(
        output_dir / "scenario_summary.md",
        "\n".join(
            [
                "# Scenario Summary",
                "",
                f"Generated at: `{generated_at}`",
                "",
                (
                    "`estimated_vs_true_metric_gap` is the maximum absolute "
                    "difference between the latest `completed_proxy` estimate "
                    "and the later latest labeled metric for the same "
                    "segment and metric name."
                ),
                "",
                markdown_table(rows),
                "",
            ]
        ),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for scenario summary generation."""

    parser = argparse.ArgumentParser(
        description="Build one aggregated scenario summary table"
    )
    parser.add_argument(
        "--final-report-dir",
        type=Path,
        default=FINAL_REPORT_DIR,
    )
    return parser


def main() -> None:
    """Build and save one aggregated scenario summary report."""

    setup_logging()
    parser = build_argument_parser()
    args = parser.parse_args()

    final_report_dir = Path(args.final_report_dir)
    if not final_report_dir.exists():
        parser.error(f"final report directory not found: {final_report_dir}")

    engine = create_engine(settings.database_url)
    try:
        rows = build_scenario_summary(
            engine=engine,
            final_report_dir=final_report_dir,
        )
    finally:
        engine.dispose()

    if not rows:
        raise ValueError(
            "No saved scenarios were discovered under artifacts/reports/final."
        )

    save_scenario_summary(rows=rows, output_dir=final_report_dir)
    logger.info(
        "Scenario summary saved. scenarios=%s output_dir=%s",
        len(rows),
        final_report_dir,
    )
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
