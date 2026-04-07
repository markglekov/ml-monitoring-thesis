from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import socket
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import requests
import websockets
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.common.config import settings
from app.common.logging import get_logger, setup_logging
from app.monitoring.backfill_labels import derive_label
from app.monitoring.drift_job import run_drift_job
from app.monitoring.quality_job import run_quality_job
from app.simulator.generate_stream import (
    apply_scenario,
    load_rows,
    repeat_to_size,
    to_native,
)

ROOT = Path(__file__).resolve().parents[2]
FINAL_REPORT_DIR = ROOT / "artifacts" / "reports" / "final"
DEFAULT_API_URL = f"http://localhost:{settings.api_port}"
DEFAULT_GRAFANA_URL = f"http://localhost:{os.getenv('GRAFANA_PORT', '3000')}"

logger = get_logger(__name__)


@dataclass(frozen=True)
class ScenarioArtifacts:
    name: str
    segment_key: str
    drift_run_id: int | None
    quality_run_id: int | None
    quality_status: str | None
    incident_count: int
    action_count: int


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON artifact with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, content: str) -> None:
    """Write a text artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a dataframe as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def fetch_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_payload: dict[str, Any] | None = None,
    method: str = "GET",
    timeout: float = 30.0,
) -> Any:
    """Call one HTTP endpoint and parse JSON."""

    response = session.request(
        method=method,
        url=url,
        params=params,
        json=json_payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def make_segment_key(run_tag: str, name: str) -> str:
    """Build a unique segment key for one export run."""

    return f"final_{run_tag}_{name}"


def send_stream(
    *,
    session: requests.Session,
    api_url: str,
    source_split: str,
    rows: int,
    scenario: str,
    seed: int,
    segment_key: str,
    manifest_path: Path,
    timeout_sec: float,
) -> pd.DataFrame:
    """Generate and send one inference stream, then save a manifest."""

    base_df = load_rows(source_split=source_split)
    sample_df = repeat_to_size(base_df, rows=rows, seed=seed)
    stream_df = apply_scenario(sample_df, scenario=scenario, seed=seed)

    records: list[dict[str, Any]] = []
    total_rows = len(stream_df)
    for row_number, (_, row) in enumerate(stream_df.iterrows(), start=1):
        request_id = str(uuid.uuid4())
        row_dict = {str(key): value for key, value in row.items()}
        target = int(row_dict.pop("target"))
        payload = {
            "request_id": request_id,
            "features": {
                key: to_native(value) for key, value in row_dict.items()
            },
            "segment_key": segment_key,
        }
        response = fetch_json(
            session,
            f"{api_url.rstrip('/')}/predict",
            json_payload=payload,
            method="POST",
            timeout=timeout_sec,
        )
        records.append(
            {
                "request_id": request_id,
                "segment_key": segment_key,
                "scenario": scenario,
                "source_split": source_split,
                "original_target": target,
                "sent_at_utc": datetime.now(UTC).isoformat(),
                "score": response["score"],
                "threshold": response["threshold"],
                "predicted_label": response["predicted_label"],
                "decision_status": response["decision_status"],
                "decision_source": response["decision_source"],
            }
        )
        if row_number % 100 == 0 or row_number == total_rows:
            logger.info(
                "Stream progress segment=%s scenario=%s row=%s/%s",
                segment_key,
                scenario,
                row_number,
                total_rows,
            )

    manifest_df = pd.DataFrame(records)
    write_csv(manifest_path, manifest_df)
    return manifest_df


def backfill_manifest_labels(
    *,
    session: requests.Session,
    api_url: str,
    manifest_df: pd.DataFrame,
    label_policy: str,
    flip_prob: float | None,
    seed: int,
    labels_path: Path,
    delay_hours: float,
    batch_size: int,
    timeout_sec: float,
) -> pd.DataFrame:
    """Backfill delayed labels through the API and save the label manifest."""

    rng = np.random.default_rng(seed)
    label_ts = datetime.now(UTC) - timedelta(hours=delay_hours)
    label_records: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []

    for _, row in manifest_df.iterrows():
        y_true = derive_label(
            original_target=int(row["original_target"]),
            scenario=str(row["scenario"]),
            policy=label_policy,
            rng=rng,
            flip_prob=flip_prob,
        )
        record = {
            "request_id": str(row["request_id"]),
            "segment_key": str(row["segment_key"]),
            "scenario": str(row["scenario"]),
            "y_true": y_true,
            "label_ts": label_ts.isoformat(),
        }
        label_records.append(record)
        payloads.append(
            {
                "request_id": record["request_id"],
                "y_true": record["y_true"],
                "label_ts": record["label_ts"],
            }
        )

    for offset in range(0, len(payloads), batch_size):
        batch = payloads[offset : offset + batch_size]
        fetch_json(
            session,
            f"{api_url.rstrip('/')}/labels/batch",
            json_payload={"labels": batch},
            method="POST",
            timeout=timeout_sec,
        )

    labels_df = pd.DataFrame(label_records)
    write_csv(labels_path, labels_df)
    return labels_df


def export_query(
    *,
    engine: Engine,
    sql_dir: Path,
    csv_dir: Path,
    name: str,
    sql_text_value: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Persist a SQL template and the corresponding CSV result."""

    write_text(sql_dir / f"{name}.sql", sql_text_value.strip() + "\n")
    frame = pd.read_sql_query(text(sql_text_value), engine, params=params)
    write_csv(csv_dir / f"{name}.csv", frame)
    return frame


def export_api_snapshots(
    *,
    session: requests.Session,
    api_url: str,
    scenario_dir: Path,
    segment_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Save API snapshots for one segment."""

    drift_runs = fetch_json(
        session,
        f"{api_url.rstrip('/')}/monitoring/drift/runs",
        params={"segment_key": segment_key, "limit": 5},
    )
    quality_runs = fetch_json(
        session,
        f"{api_url.rstrip('/')}/monitoring/quality/runs",
        params={"segment_key": segment_key, "limit": 5},
    )
    incidents = fetch_json(
        session,
        f"{api_url.rstrip('/')}/monitoring/incidents",
        params={"segment_key": segment_key, "status": "open", "limit": 20},
    )
    write_json(scenario_dir / "api" / "drift_runs.json", drift_runs)
    write_json(scenario_dir / "api" / "quality_runs.json", quality_runs)
    write_json(scenario_dir / "api" / "incidents.json", incidents)
    return drift_runs, quality_runs, incidents


def export_monitoring_actions(
    *,
    engine: Engine,
    scenario_dir: Path,
    segment_key: str,
) -> pd.DataFrame:
    """Save action records related to one segment."""

    query = """
    SELECT
        ma.action_id,
        ma.incident_id,
        ma.action_type,
        ma.status,
        ma.started_at,
        ma.ended_at,
        ma.trigger_reason,
        ma.old_config,
        ma.new_config
    FROM monitoring_actions ma
    JOIN monitoring_incidents mi
        ON mi.id = ma.incident_id
    WHERE mi.segment_key = :segment_key
    ORDER BY ma.action_id DESC
    """
    actions_df = export_query(
        engine=engine,
        sql_dir=scenario_dir / "sql",
        csv_dir=scenario_dir / "csv",
        name="monitoring_actions",
        sql_text_value=query,
        params={"segment_key": segment_key},
    )
    write_json(
        scenario_dir / "api" / "monitoring_actions.json",
        actions_df.to_dict(orient="records"),
    )
    return actions_df


def export_drift_queries(
    *,
    engine: Engine,
    scenario_dir: Path,
    segment_key: str,
) -> tuple[int | None, pd.DataFrame]:
    """Save drift SQL extracts for one segment."""

    latest_run_sql = """
    SELECT
        id,
        ts_started,
        ts_finished,
        model_version,
        window_size,
        segment_key,
        status,
        drifted_features_count,
        total_features_count,
        overall_drift,
        summary_json
    FROM monitoring_runs
    WHERE segment_key = :segment_key
    ORDER BY id DESC
    LIMIT 1
    """
    latest_run_df = export_query(
        engine=engine,
        sql_dir=scenario_dir / "sql",
        csv_dir=scenario_dir / "csv",
        name="latest_drift_run",
        sql_text_value=latest_run_sql,
        params={"segment_key": segment_key},
    )
    run_id = (
        int(latest_run_df.iloc[0]["id"]) if not latest_run_df.empty else None
    )
    drift_metrics_sql = """
    SELECT
        id,
        run_id,
        segment_key,
        feature_name,
        feature_type,
        detector_name,
        statistic,
        pvalue,
        effect_size,
        pvalue_adj,
        window_start,
        window_end,
        severity,
        recommended_action,
        drift_detected,
        ks_pvalue,
        chi2_pvalue,
        psi_value,
        details_json
    FROM drift_metrics
    WHERE run_id = :run_id
    ORDER BY
        CASE severity
            WHEN 'critical' THEN 0
            WHEN 'warning' THEN 1
            ELSE 2
        END,
        detector_name ASC,
        feature_name ASC,
        id ASC
    """
    if run_id is None:
        drift_metrics_df = pd.DataFrame()
    else:
        drift_metrics_df = export_query(
            engine=engine,
            sql_dir=scenario_dir / "sql",
            csv_dir=scenario_dir / "csv",
            name="latest_drift_metrics",
            sql_text_value=drift_metrics_sql,
            params={"run_id": run_id},
        )
    return run_id, drift_metrics_df


def export_quality_queries(
    *,
    engine: Engine,
    scenario_dir: Path,
    segment_key: str,
) -> tuple[int | None, str | None, pd.DataFrame, pd.DataFrame]:
    """Save quality SQL extracts for one segment."""

    latest_run_sql = """
    SELECT
        id,
        ts_started,
        ts_finished,
        model_version,
        window_size,
        segment_key,
        status,
        labeled_rows,
        degraded_metrics_count,
        summary_json
    FROM quality_runs
    WHERE segment_key = :segment_key
    ORDER BY id DESC
    LIMIT 1
    """
    latest_run_df = export_query(
        engine=engine,
        sql_dir=scenario_dir / "sql",
        csv_dir=scenario_dir / "csv",
        name="latest_quality_run",
        sql_text_value=latest_run_sql,
        params={"segment_key": segment_key},
    )
    run_id = (
        int(latest_run_df.iloc[0]["id"]) if not latest_run_df.empty else None
    )
    status = (
        str(latest_run_df.iloc[0]["status"])
        if not latest_run_df.empty
        else None
    )
    quality_metrics_sql = """
    SELECT
        id,
        run_id,
        segment_key,
        metric_name,
        metric_value,
        baseline_value,
        delta_value,
        detector_name,
        effect_size,
        pvalue_adj,
        severity,
        recommended_action,
        degradation_detected,
        details_json
    FROM quality_metrics
    WHERE run_id = :run_id
    ORDER BY
        CASE severity
            WHEN 'critical' THEN 0
            WHEN 'warning' THEN 1
            ELSE 2
        END,
        metric_name ASC,
        id ASC
    """
    quality_estimates_sql = """
    SELECT
        id,
        run_id,
        segment_key,
        estimated_positive_rate,
        estimated_metric_name,
        estimated_metric_value,
        assumption_type,
        quality_estimate_uncertainty,
        confidence_interval_json,
        details_json
    FROM quality_estimates
    WHERE run_id = :run_id
    ORDER BY estimated_metric_name ASC, id ASC
    """
    if run_id is None:
        quality_metrics_df = pd.DataFrame()
        quality_estimates_df = pd.DataFrame()
    else:
        quality_metrics_df = export_query(
            engine=engine,
            sql_dir=scenario_dir / "sql",
            csv_dir=scenario_dir / "csv",
            name="latest_quality_metrics",
            sql_text_value=quality_metrics_sql,
            params={"run_id": run_id},
        )
        quality_estimates_df = export_query(
            engine=engine,
            sql_dir=scenario_dir / "sql",
            csv_dir=scenario_dir / "csv",
            name="latest_quality_estimates",
            sql_text_value=quality_estimates_sql,
            params={"run_id": run_id},
        )
    return run_id, status, quality_metrics_df, quality_estimates_df


def export_proxy_comparison(
    *,
    engine: Engine,
    scenario_dir: Path,
    segment_key: str,
) -> pd.DataFrame:
    """Save one SQL comparison of proxy estimates versus later labels."""

    proxy_comparison_sql = """
    WITH latest_proxy_run AS (
        SELECT id
        FROM quality_runs
        WHERE segment_key = :segment_key
          AND status = 'completed_proxy'
        ORDER BY id DESC
        LIMIT 1
    ),
    latest_labeled_run AS (
        SELECT id
        FROM quality_runs
        WHERE segment_key = :segment_key
          AND status = 'completed'
        ORDER BY id DESC
        LIMIT 1
    )
    SELECT
        qe.assumption_type,
        qe.estimated_metric_name,
        qe.estimated_metric_value,
        qe.quality_estimate_uncertainty,
        qm.metric_value AS true_metric_value
    FROM latest_proxy_run
    JOIN quality_estimates qe
        ON qe.run_id = latest_proxy_run.id
    LEFT JOIN latest_labeled_run
        ON TRUE
    LEFT JOIN quality_metrics qm
        ON qm.run_id = latest_labeled_run.id
       AND qm.metric_name = qe.estimated_metric_name
    WHERE qe.estimated_metric_name IN (
        'positive_rate_true',
        'f1',
        'precision',
        'recall'
    )
    ORDER BY qe.estimated_metric_name ASC
    """
    return export_query(
        engine=engine,
        sql_dir=scenario_dir / "sql",
        csv_dir=scenario_dir / "csv",
        name="proxy_vs_backfill_quality",
        sql_text_value=proxy_comparison_sql,
        params={"segment_key": segment_key},
    )


def save_overview_snapshot(
    *,
    session: requests.Session,
    api_url: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Save the current overview payload."""

    payload = fetch_json(
        session,
        f"{api_url.rstrip('/')}/monitoring/overview",
    )
    write_json(output_dir / "overview.json", payload)
    return dict(payload)


def list_dashboard_panel_ids(dashboard_path: Path) -> dict[str, int]:
    """Map Grafana panel titles to panel ids from one dashboard JSON."""

    payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
    return {
        str(panel["title"]): int(panel["id"])
        for panel in payload.get("panels", [])
        if "title" in panel and "id" in panel
    }


def login_grafana(
    *,
    grafana_url: str,
    username: str,
    password: str,
) -> requests.Session:
    """Create an authenticated Grafana session for render export."""

    session = requests.Session()
    response = session.post(
        f"{grafana_url.rstrip('/')}/login",
        json={"user": username, "password": password},
        timeout=30.0,
    )
    response.raise_for_status()
    return session


def render_grafana_png(
    *,
    session: requests.Session,
    grafana_url: str,
    url_path: str,
    output_path: Path,
) -> None:
    """Render one Grafana PNG and save it."""

    response = session.get(
        f"{grafana_url.rstrip('/')}{url_path}",
        timeout=60.0,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "image/png" not in content_type:
        raise ValueError(
            "Expected image/png from Grafana render endpoint, "
            f"got {content_type!r}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)


class ChromeDevToolsClient:
    """Minimal CDP client for navigation and screenshots."""

    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self._next_id = 0

    async def call(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send one CDP command and wait for its matching response."""

        self._next_id += 1
        command_id = self._next_id
        await self.websocket.send(
            json.dumps(
                {
                    "id": command_id,
                    "method": method,
                    "params": params or {},
                }
            )
        )
        while True:
            message = json.loads(await self.websocket.recv())
            if message.get("id") != command_id:
                continue
            if "error" in message:
                raise ValueError(
                    f"Chrome CDP command failed: {method}: {message['error']}"
                )
            return dict(message.get("result", {}))


def reserve_local_port() -> int:
    """Pick one free localhost port for Chrome remote debugging."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_devtools(port: int, timeout_sec: float = 20.0) -> dict[str, Any]:
    """Wait until headless Chrome exposes its DevTools endpoint."""

    version_url = f"http://127.0.0.1:{port}/json/version"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            response = requests.get(version_url, timeout=1.0)
            response.raise_for_status()
            return dict(response.json())
        except requests.RequestException:
            time.sleep(0.25)
    raise TimeoutError(
        "Chrome DevTools endpoint did not become ready in time."
    )


async def capture_grafana_browser_screenshots(
    *,
    grafana_url: str,
    output_dir: Path,
    screenshot_specs: list[dict[str, Any]],
    username: str,
    password: str,
) -> list[dict[str, Any]]:
    """Use headless Chrome to capture real Grafana screenshots."""

    session = login_grafana(
        grafana_url=grafana_url,
        username=username,
        password=password,
    )
    port = reserve_local_port()
    with TemporaryDirectory(
        prefix="grafana-shot-", ignore_cleanup_errors=True
    ) as temp_dir:
        chrome = subprocess.Popen(
            [
                "google-chrome",
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--hide-scrollbars",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
                f"--user-data-dir={temp_dir}",
                f"--remote-debugging-port={port}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            version_payload = wait_for_devtools(port)
            target_payload = requests.put(
                f"http://127.0.0.1:{port}/json/new?about:blank",
                timeout=10.0,
            )
            target_payload.raise_for_status()
            websocket_url = str(target_payload.json()["webSocketDebuggerUrl"])
            async with websockets.connect(
                websocket_url, max_size=None
            ) as websocket:
                client = ChromeDevToolsClient(websocket)
                await client.call("Page.enable")
                await client.call("Runtime.enable")
                await client.call("Network.enable")
                await client.call(
                    "Emulation.setDeviceMetricsOverride",
                    {
                        "width": 1800,
                        "height": 1400,
                        "deviceScaleFactor": 1,
                        "mobile": False,
                    },
                )
                for cookie in session.cookies:
                    await client.call(
                        "Network.setCookie",
                        {
                            "name": cookie.name,
                            "value": cookie.value,
                            "url": f"{grafana_url.rstrip('/')}/",
                            "path": cookie.path or "/",
                            "secure": cookie.secure,
                        },
                    )
                for shot in screenshot_specs:
                    await client.call("Page.navigate", {"url": shot["url"]})
                    expected_text = shot.get("wait_for_text")
                    deadline = time.time() + 20.0
                    while time.time() < deadline:
                        body_text_result = await client.call(
                            "Runtime.evaluate",
                            {
                                "expression": (
                                    "document.body ? "
                                    "document.body.innerText : ''"
                                ),
                                "returnByValue": True,
                            },
                        )
                        body_text = str(
                            body_text_result.get("result", {}).get("value", "")
                        )
                        if expected_text and expected_text in body_text:
                            break
                        if (
                            not expected_text
                            and "Loading" not in body_text
                            and "Panel data error" not in body_text
                        ):
                            break
                        await asyncio.sleep(1.0)
                    screenshot = await client.call(
                        "Page.captureScreenshot",
                        {
                            "format": "png",
                            "fromSurface": True,
                            "captureBeyondViewport": True,
                        },
                    )
                    output_path = output_dir / f"{shot['name']}.png"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(
                        base64.b64decode(screenshot["data"])
                    )
                return [
                    {
                        **shot,
                        "devtools_browser": version_payload.get("Browser"),
                    }
                    for shot in screenshot_specs
                ]
        finally:
            chrome.terminate()
            try:
                chrome.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                chrome.kill()


def export_grafana_screenshots(
    *,
    grafana_url: str,
    output_dir: Path,
    run_started_at: datetime,
    username: str,
    password: str,
) -> list[dict[str, Any]]:
    """Render final Grafana screenshots for the thesis report."""

    from_ms = int((run_started_at - timedelta(minutes=10)).timestamp() * 1000)
    to_ms = int((datetime.now(UTC) + timedelta(minutes=5)).timestamp() * 1000)
    model_version = settings.model_version

    overview_panels = list_dashboard_panel_ids(
        ROOT
        / "monitoring"
        / "grafana"
        / "dashboards"
        / "ml-monitoring-overview.json"
    )
    drift_panels = list_dashboard_panel_ids(
        ROOT
        / "monitoring"
        / "grafana"
        / "dashboards"
        / "ml-monitoring-drift.json"
    )
    quality_panels = list_dashboard_panel_ids(
        ROOT
        / "monitoring"
        / "grafana"
        / "dashboards"
        / "ml-monitoring-quality.json"
    )
    proxy_panels = list_dashboard_panel_ids(
        ROOT
        / "monitoring"
        / "grafana"
        / "dashboards"
        / "ml-monitoring-proxy.json"
    )
    segment_panels = list_dashboard_panel_ids(
        ROOT
        / "monitoring"
        / "grafana"
        / "dashboards"
        / "ml-monitoring-segments.json"
    )

    shots = [
        {
            "name": "overview_full",
            "url": (
                f"{grafana_url.rstrip('/')}"
                "/d/ml-monitoring-overview/ml-monitoring-overview"
                f"?orgId=1&viewPanel={overview_panels['Overall Severity']}"
                f"&var-model_version={model_version}"
                f"&from={from_ms}&to={to_ms}&kiosk"
            ),
            "wait_for_text": "critical",
        },
        {
            "name": "drift_feature_details",
            "url": (
                f"{grafana_url.rstrip('/')}"
                "/d/ml-monitoring-drift/ml-monitoring-drift"
                f"?orgId=1&viewPanel="
                f"{drift_panels['Latest Drifted Features By Segment']}"
                f"&var-model_version={model_version}"
                f"&from={from_ms}&to={to_ms}&kiosk"
            ),
            "wait_for_text": "final_",
        },
        {
            "name": "quality_labeled_metrics",
            "url": (
                f"{grafana_url.rstrip('/')}"
                "/d/ml-monitoring-quality/ml-monitoring-quality"
                f"?orgId=1&viewPanel="
                f"{quality_panels['Recent Missing Rate By Feature']}"
                f"&var-model_version={model_version}"
                f"&from={from_ms}&to={to_ms}&kiosk"
            ),
            "wait_for_text": "feature_name=",
        },
        {
            "name": "quality_unlabeled_estimates",
            "url": (
                f"{grafana_url.rstrip('/')}"
                "/d/ml-monitoring-quality/ml-monitoring-quality"
                f"?orgId=1&viewPanel="
                f"{quality_panels['Recent Out-Of-Range Rate By Feature']}"
                f"&var-model_version={model_version}"
                f"&from={from_ms}&to={to_ms}&kiosk"
            ),
            "wait_for_text": "feature_name=",
        },
        {
            "name": "proxy_label_coverage",
            "url": (
                f"{grafana_url.rstrip('/')}"
                "/d/ml-monitoring-proxy/ml-monitoring-proxy"
                f"?orgId=1&viewPanel="
                f"{proxy_panels['Latest Score PSI By Segment']}"
                f"&var-model_version={model_version}"
                f"&from={from_ms}&to={to_ms}&kiosk"
            ),
            "wait_for_text": "final_",
        },
        {
            "name": "segment_snapshot",
            "url": (
                f"{grafana_url.rstrip('/')}"
                "/d/ml-monitoring-segments/ml-monitoring-segments"
                f"?orgId=1&viewPanel="
                f"{segment_panels['Active Incidents By Segment And Severity']}"
                f"&var-model_version={model_version}"
                f"&from={from_ms}&to={to_ms}&kiosk"
            ),
            "wait_for_text": "final_",
        },
    ]

    return asyncio.run(
        capture_grafana_browser_screenshots(
            grafana_url=grafana_url,
            output_dir=output_dir,
            screenshot_specs=shots,
            username=username,
            password=password,
        )
    )


def build_summary_row(
    *,
    name: str,
    segment_key: str,
    drift_runs: list[dict[str, Any]],
    quality_runs: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
    actions_df: pd.DataFrame,
) -> ScenarioArtifacts:
    """Create one compact summary row for the top-level report."""

    latest_drift = drift_runs[0] if drift_runs else {}
    latest_quality = quality_runs[0] if quality_runs else {}
    return ScenarioArtifacts(
        name=name,
        segment_key=segment_key,
        drift_run_id=latest_drift.get("id"),
        quality_run_id=latest_quality.get("id"),
        quality_status=latest_quality.get("status"),
        incident_count=len(incidents),
        action_count=len(actions_df),
    )


def scenario_markdown(
    *,
    name: str,
    segment_key: str,
    drift_runs: list[dict[str, Any]],
    quality_runs: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
) -> str:
    """Build a markdown fragment with actual scenario outputs."""

    latest_drift = drift_runs[0] if drift_runs else None
    latest_quality = quality_runs[0] if quality_runs else None
    lines = [f"## {name.title()}", f"- Segment key: `{segment_key}`"]
    if latest_drift is not None:
        lines.append(
            "- Drift run:"
            f" id={latest_drift['id']},"
            f" overall_drift={latest_drift['overall_drift']},"
            f" drifted_features_count={latest_drift['drifted_features_count']}"
        )
    if latest_quality is not None:
        lines.append(
            "- Quality run:"
            f" id={latest_quality['id']},"
            f" status={latest_quality['status']},"
            " degraded_metrics_count="
            f"{latest_quality['degraded_metrics_count']},"
            f" labeled_rows={latest_quality['labeled_rows']}"
        )
    lines.append(f"- Open incidents exported: {len(incidents)}")
    return "\n".join(lines)


def write_top_level_readme(
    *,
    output_dir: Path,
    generated_at: datetime,
    scenarios: list[ScenarioArtifacts],
    scenario_sections: list[str],
    screenshot_names: list[str],
) -> None:
    """Write a markdown summary of the generated final report artifacts."""

    table_lines = [
        (
            "| name | segment_key | drift_run_id | quality_run_id | "
            "quality_status | incident_count | action_count |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario in scenarios:
        table_lines.append(
            "| "
            + " | ".join(
                [
                    str(scenario.name),
                    str(scenario.segment_key),
                    str(scenario.drift_run_id),
                    str(scenario.quality_run_id),
                    str(scenario.quality_status),
                    str(scenario.incident_count),
                    str(scenario.action_count),
                ]
            )
            + " |"
        )

    lines = [
        "# Final Experiment Artifacts",
        "",
        f"Generated at: `{generated_at.isoformat()}`",
        "",
        (
            "The folders `baseline/`, `mild/`, `severe/`, "
            "`proxy/`, and `segment/`"
        ),
        (
            "contain raw manifests, API snapshots, SQL query templates, "
            "and CSV extracts."
        ),
        "",
        "## Scenario Index",
        "",
        *table_lines,
        "",
        "## Grafana Screenshots",
    ]
    for name in screenshot_names:
        lines.append(f"- `screenshots/{name}.png`")
    lines.append("")
    lines.extend(scenario_sections)
    lines.append("")
    write_text(output_dir / "README.md", "\n".join(lines))


def run_baseline_scenario(
    *,
    session: requests.Session,
    engine: Engine,
    api_url: str,
    output_dir: Path,
    run_tag: str,
    seed: int,
    timeout_sec: float,
) -> ScenarioArtifacts:
    """Generate baseline-like drift and quality artifacts."""

    scenario_dir = output_dir / "baseline"
    segment_key = make_segment_key(run_tag, "baseline")
    drift_manifest = send_stream(
        session=session,
        api_url=api_url,
        source_split="train",
        rows=300,
        scenario="none",
        seed=seed,
        segment_key=segment_key,
        manifest_path=scenario_dir / "manifests" / "drift_manifest.csv",
        timeout_sec=timeout_sec,
    )
    run_drift_job(
        window_size=len(drift_manifest), min_rows=50, segment_key=segment_key
    )

    quality_manifest = send_stream(
        session=session,
        api_url=api_url,
        source_split="test",
        rows=1000,
        scenario="none",
        seed=seed + 1,
        segment_key=segment_key,
        manifest_path=scenario_dir / "manifests" / "quality_manifest.csv",
        timeout_sec=timeout_sec,
    )
    backfill_manifest_labels(
        session=session,
        api_url=api_url,
        manifest_df=quality_manifest,
        label_policy="perfect",
        flip_prob=None,
        seed=seed,
        labels_path=scenario_dir / "manifests" / "quality_labels.csv",
        delay_hours=24.0,
        batch_size=500,
        timeout_sec=timeout_sec,
    )
    run_quality_job(
        window_size=len(quality_manifest),
        min_rows=50,
        baseline_source="test",
        segment_key=segment_key,
    )

    drift_runs, quality_runs, incidents = export_api_snapshots(
        session=session,
        api_url=api_url,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    export_drift_queries(
        engine=engine, scenario_dir=scenario_dir, segment_key=segment_key
    )
    export_quality_queries(
        engine=engine,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    actions_df = export_monitoring_actions(
        engine=engine,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    return build_summary_row(
        name="baseline",
        segment_key=segment_key,
        drift_runs=drift_runs,
        quality_runs=quality_runs,
        incidents=incidents,
        actions_df=actions_df,
    )


def run_mild_scenario(
    *,
    session: requests.Session,
    engine: Engine,
    api_url: str,
    output_dir: Path,
    run_tag: str,
    seed: int,
    timeout_sec: float,
) -> ScenarioArtifacts:
    """Generate mild drift and mild quality degradation artifacts."""

    scenario_dir = output_dir / "mild"
    segment_key = make_segment_key(run_tag, "mild")
    drift_manifest = send_stream(
        session=session,
        api_url=api_url,
        source_split="train",
        rows=300,
        scenario="mild",
        seed=seed,
        segment_key=segment_key,
        manifest_path=scenario_dir / "manifests" / "drift_manifest.csv",
        timeout_sec=timeout_sec,
    )
    run_drift_job(
        window_size=len(drift_manifest), min_rows=50, segment_key=segment_key
    )

    quality_manifest = send_stream(
        session=session,
        api_url=api_url,
        source_split="test",
        rows=1000,
        scenario="none",
        seed=seed + 1,
        segment_key=segment_key,
        manifest_path=scenario_dir / "manifests" / "quality_manifest.csv",
        timeout_sec=timeout_sec,
    )
    backfill_manifest_labels(
        session=session,
        api_url=api_url,
        manifest_df=quality_manifest,
        label_policy="custom_flip",
        flip_prob=0.10,
        seed=seed,
        labels_path=scenario_dir / "manifests" / "quality_labels.csv",
        delay_hours=24.0,
        batch_size=500,
        timeout_sec=timeout_sec,
    )
    run_quality_job(
        window_size=len(quality_manifest),
        min_rows=50,
        baseline_source="test",
        segment_key=segment_key,
    )

    drift_runs, quality_runs, incidents = export_api_snapshots(
        session=session,
        api_url=api_url,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    export_drift_queries(
        engine=engine, scenario_dir=scenario_dir, segment_key=segment_key
    )
    export_quality_queries(
        engine=engine,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    actions_df = export_monitoring_actions(
        engine=engine,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    return build_summary_row(
        name="mild",
        segment_key=segment_key,
        drift_runs=drift_runs,
        quality_runs=quality_runs,
        incidents=incidents,
        actions_df=actions_df,
    )


def run_severe_scenario(
    *,
    session: requests.Session,
    engine: Engine,
    api_url: str,
    output_dir: Path,
    run_tag: str,
    seed: int,
    timeout_sec: float,
) -> ScenarioArtifacts:
    """Generate severe drift and severe quality degradation artifacts."""

    scenario_dir = output_dir / "severe"
    segment_key = make_segment_key(run_tag, "severe")
    drift_manifest = send_stream(
        session=session,
        api_url=api_url,
        source_split="train",
        rows=300,
        scenario="severe",
        seed=seed,
        segment_key=segment_key,
        manifest_path=scenario_dir / "manifests" / "drift_manifest.csv",
        timeout_sec=timeout_sec,
    )
    run_drift_job(
        window_size=len(drift_manifest), min_rows=50, segment_key=segment_key
    )

    quality_manifest = send_stream(
        session=session,
        api_url=api_url,
        source_split="test",
        rows=1000,
        scenario="none",
        seed=seed + 1,
        segment_key=segment_key,
        manifest_path=scenario_dir / "manifests" / "quality_manifest.csv",
        timeout_sec=timeout_sec,
    )
    backfill_manifest_labels(
        session=session,
        api_url=api_url,
        manifest_df=quality_manifest,
        label_policy="custom_flip",
        flip_prob=0.25,
        seed=seed,
        labels_path=scenario_dir / "manifests" / "quality_labels.csv",
        delay_hours=24.0,
        batch_size=500,
        timeout_sec=timeout_sec,
    )
    run_quality_job(
        window_size=len(quality_manifest),
        min_rows=50,
        baseline_source="test",
        segment_key=segment_key,
    )

    drift_runs, quality_runs, incidents = export_api_snapshots(
        session=session,
        api_url=api_url,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    export_drift_queries(
        engine=engine, scenario_dir=scenario_dir, segment_key=segment_key
    )
    export_quality_queries(
        engine=engine,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    actions_df = export_monitoring_actions(
        engine=engine,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    return build_summary_row(
        name="severe",
        segment_key=segment_key,
        drift_runs=drift_runs,
        quality_runs=quality_runs,
        incidents=incidents,
        actions_df=actions_df,
    )


def run_proxy_scenario(
    *,
    session: requests.Session,
    engine: Engine,
    api_url: str,
    output_dir: Path,
    run_tag: str,
    seed: int,
    timeout_sec: float,
) -> ScenarioArtifacts:
    """Generate blind-period proxy artifacts and later backfill comparison."""

    scenario_dir = output_dir / "proxy"
    segment_key = make_segment_key(run_tag, "proxy")
    quality_manifest = send_stream(
        session=session,
        api_url=api_url,
        source_split="test",
        rows=1000,
        scenario="severe",
        seed=seed,
        segment_key=segment_key,
        manifest_path=scenario_dir / "manifests" / "proxy_manifest.csv",
        timeout_sec=timeout_sec,
    )
    run_quality_job(
        window_size=len(quality_manifest),
        min_rows=50,
        baseline_source="test",
        segment_key=segment_key,
    )
    write_json(
        scenario_dir / "api" / "quality_runs_before_backfill.json",
        fetch_json(
            session,
            f"{api_url.rstrip('/')}/monitoring/quality/runs",
            params={"segment_key": segment_key, "limit": 5},
        ),
    )

    backfill_manifest_labels(
        session=session,
        api_url=api_url,
        manifest_df=quality_manifest,
        label_policy="scenario_default",
        flip_prob=None,
        seed=seed,
        labels_path=scenario_dir / "manifests" / "proxy_labels.csv",
        delay_hours=24.0,
        batch_size=500,
        timeout_sec=timeout_sec,
    )
    run_quality_job(
        window_size=len(quality_manifest),
        min_rows=50,
        baseline_source="test",
        segment_key=segment_key,
    )

    drift_runs, quality_runs, incidents = export_api_snapshots(
        session=session,
        api_url=api_url,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    export_quality_queries(
        engine=engine,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    export_proxy_comparison(
        engine=engine,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    actions_df = export_monitoring_actions(
        engine=engine,
        scenario_dir=scenario_dir,
        segment_key=segment_key,
    )
    return build_summary_row(
        name="proxy",
        segment_key=segment_key,
        drift_runs=drift_runs,
        quality_runs=quality_runs,
        incidents=incidents,
        actions_df=actions_df,
    )


def run_segment_scenario(
    *,
    session: requests.Session,
    engine: Engine,
    api_url: str,
    output_dir: Path,
    run_tag: str,
    seed: int,
    timeout_sec: float,
) -> ScenarioArtifacts:
    """Generate a pair of healthy/hot segments for segment-local monitoring."""

    scenario_dir = output_dir / "segment"
    stable_segment = make_segment_key(run_tag, "segment_stable")
    hot_segment = make_segment_key(run_tag, "segment_hot")

    stable_manifest = send_stream(
        session=session,
        api_url=api_url,
        source_split="test",
        rows=400,
        scenario="none",
        seed=seed,
        segment_key=stable_segment,
        manifest_path=scenario_dir / "manifests" / "stable_manifest.csv",
        timeout_sec=timeout_sec,
    )
    hot_manifest = send_stream(
        session=session,
        api_url=api_url,
        source_split="test",
        rows=400,
        scenario="severe",
        seed=seed + 1,
        segment_key=hot_segment,
        manifest_path=scenario_dir / "manifests" / "hot_manifest.csv",
        timeout_sec=timeout_sec,
    )

    for segment_key, manifest_df, label_policy, flip_prob, labels_name in [
        (
            stable_segment,
            stable_manifest,
            "perfect",
            None,
            "stable_labels.csv",
        ),
        (
            hot_segment,
            hot_manifest,
            "scenario_default",
            None,
            "hot_labels.csv",
        ),
    ]:
        run_drift_job(
            window_size=len(manifest_df),
            min_rows=50,
            segment_key=segment_key,
        )
        backfill_manifest_labels(
            session=session,
            api_url=api_url,
            manifest_df=manifest_df,
            label_policy=label_policy,
            flip_prob=flip_prob,
            seed=seed,
            labels_path=scenario_dir / "manifests" / labels_name,
            delay_hours=24.0,
            batch_size=500,
            timeout_sec=timeout_sec,
        )
        run_quality_job(
            window_size=len(manifest_df),
            min_rows=50,
            baseline_source="test",
            segment_key=segment_key,
        )

    overview = save_overview_snapshot(
        session=session,
        api_url=api_url,
        output_dir=scenario_dir / "api",
    )
    stable_runs = fetch_json(
        session,
        f"{api_url.rstrip('/')}/monitoring/drift/runs",
        params={"segment_key": stable_segment, "limit": 5},
    )
    hot_runs = fetch_json(
        session,
        f"{api_url.rstrip('/')}/monitoring/drift/runs",
        params={"segment_key": hot_segment, "limit": 5},
    )
    stable_quality_runs = fetch_json(
        session,
        f"{api_url.rstrip('/')}/monitoring/quality/runs",
        params={"segment_key": stable_segment, "limit": 5},
    )
    hot_quality_runs = fetch_json(
        session,
        f"{api_url.rstrip('/')}/monitoring/quality/runs",
        params={"segment_key": hot_segment, "limit": 5},
    )
    all_incidents = fetch_json(
        session,
        f"{api_url.rstrip('/')}/monitoring/incidents",
        params={"status": "open", "limit": 50},
    )
    incidents = [
        item
        for item in all_incidents
        if item.get("segment_key") in {stable_segment, hot_segment}
    ]
    write_json(scenario_dir / "api" / "drift_runs.json", hot_runs)
    write_json(
        scenario_dir / "api" / "quality_runs.json",
        hot_quality_runs,
    )
    write_json(scenario_dir / "api" / "incidents.json", incidents)
    write_json(scenario_dir / "api" / "stable_drift_runs.json", stable_runs)
    write_json(scenario_dir / "api" / "hot_drift_runs.json", hot_runs)
    write_json(
        scenario_dir / "api" / "stable_quality_runs.json",
        stable_quality_runs,
    )
    write_json(
        scenario_dir / "api" / "hot_quality_runs.json",
        hot_quality_runs,
    )
    write_json(
        scenario_dir / "api" / "all_open_incidents.json",
        all_incidents,
    )

    segment_snapshot_sql = """
    SELECT
        mi.source_type,
        mi.segment_key,
        mi.severity,
        mi.status,
        mi.title,
        mi.recommended_action,
        mi.ts_opened,
        mi.latest_run_id
    FROM monitoring_incidents mi
    WHERE mi.segment_key IN (:stable_segment, :hot_segment)
    ORDER BY mi.ts_opened DESC, mi.id DESC
    """
    export_query(
        engine=engine,
        sql_dir=scenario_dir / "sql",
        csv_dir=scenario_dir / "csv",
        name="segment_incidents",
        sql_text_value=segment_snapshot_sql,
        params={
            "stable_segment": stable_segment,
            "hot_segment": hot_segment,
        },
    )
    actions_df = export_query(
        engine=engine,
        sql_dir=scenario_dir / "sql",
        csv_dir=scenario_dir / "csv",
        name="segment_actions",
        sql_text_value="""
        SELECT
            ma.action_id,
            mi.segment_key,
            ma.action_type,
            ma.status,
            ma.started_at,
            ma.ended_at,
            ma.trigger_reason
        FROM monitoring_actions ma
        JOIN monitoring_incidents mi
            ON mi.id = ma.incident_id
        WHERE mi.segment_key IN (:stable_segment, :hot_segment)
        ORDER BY ma.action_id DESC
        """,
        params={
            "stable_segment": stable_segment,
            "hot_segment": hot_segment,
        },
    )
    write_json(scenario_dir / "api" / "overview_snapshot.json", overview)

    return ScenarioArtifacts(
        name="segment",
        segment_key=f"{stable_segment},{hot_segment}",
        drift_run_id=(
            hot_runs[0]["id"]
            if hot_runs
            else stable_runs[0]["id"]
            if stable_runs
            else None
        ),
        quality_run_id=(
            hot_quality_runs[0]["id"]
            if hot_quality_runs
            else stable_quality_runs[0]["id"]
            if stable_quality_runs
            else None
        ),
        quality_status=(
            hot_quality_runs[0]["status"]
            if hot_quality_runs
            else stable_quality_runs[0]["status"]
            if stable_quality_runs
            else None
        ),
        incident_count=len(incidents),
        action_count=len(actions_df),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the final artifact exporter."""

    parser = argparse.ArgumentParser(
        description="Export real monitoring experiment artifacts"
    )
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--grafana-url", type=str, default=DEFAULT_GRAFANA_URL)
    parser.add_argument("--grafana-user", type=str, default="admin")
    parser.add_argument("--grafana-password", type=str, default="admin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--wait-for-scrape-sec",
        type=float,
        default=20.0,
        help=(
            "Pause after scenario execution so Prometheus-backed panels "
            "refresh."
        ),
    )
    return parser


def main() -> None:
    """Run the full final artifact export flow."""

    setup_logging()
    parser = build_argument_parser()
    args = parser.parse_args()

    run_started_at = datetime.now(UTC)
    run_tag = run_started_at.strftime("%Y%m%d_%H%M%S")
    session = requests.Session()
    engine = create_engine(settings.database_url)

    logger.info("Starting final experiment export. run_tag=%s", run_tag)
    FINAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    overview_before = save_overview_snapshot(
        session=session,
        api_url=args.api_url,
        output_dir=FINAL_REPORT_DIR / "_meta",
    )
    write_json(
        FINAL_REPORT_DIR / "export_metadata.json",
        {
            "generated_at": run_started_at.isoformat(),
            "run_tag": run_tag,
            "api_url": args.api_url,
            "grafana_url": args.grafana_url,
            "prometheus_scrape_wait_sec": args.wait_for_scrape_sec,
            "overview_before": overview_before,
        },
    )

    scenarios = [
        run_baseline_scenario(
            session=session,
            engine=engine,
            api_url=args.api_url,
            output_dir=FINAL_REPORT_DIR,
            run_tag=run_tag,
            seed=args.seed,
            timeout_sec=args.timeout_sec,
        ),
        run_mild_scenario(
            session=session,
            engine=engine,
            api_url=args.api_url,
            output_dir=FINAL_REPORT_DIR,
            run_tag=run_tag,
            seed=args.seed + 100,
            timeout_sec=args.timeout_sec,
        ),
        run_severe_scenario(
            session=session,
            engine=engine,
            api_url=args.api_url,
            output_dir=FINAL_REPORT_DIR,
            run_tag=run_tag,
            seed=args.seed + 200,
            timeout_sec=args.timeout_sec,
        ),
        run_proxy_scenario(
            session=session,
            engine=engine,
            api_url=args.api_url,
            output_dir=FINAL_REPORT_DIR,
            run_tag=run_tag,
            seed=args.seed + 300,
            timeout_sec=args.timeout_sec,
        ),
        run_segment_scenario(
            session=session,
            engine=engine,
            api_url=args.api_url,
            output_dir=FINAL_REPORT_DIR,
            run_tag=run_tag,
            seed=args.seed + 400,
            timeout_sec=args.timeout_sec,
        ),
    ]

    if args.wait_for_scrape_sec > 0:
        logger.info(
            "Waiting %.1f seconds for Prometheus/Grafana freshness.",
            args.wait_for_scrape_sec,
        )
        end_at = datetime.now(UTC) + timedelta(
            seconds=args.wait_for_scrape_sec
        )
        while datetime.now(UTC) < end_at:
            remaining = (end_at - datetime.now(UTC)).total_seconds()
            sleep_seconds = min(5.0, max(0.0, remaining))
            if sleep_seconds <= 0:
                break
            time.sleep(sleep_seconds)

    overview_after = save_overview_snapshot(
        session=session,
        api_url=args.api_url,
        output_dir=FINAL_REPORT_DIR,
    )
    write_json(
        FINAL_REPORT_DIR / "_meta" / "overview_after.json", overview_after
    )
    screenshots = export_grafana_screenshots(
        grafana_url=args.grafana_url,
        output_dir=FINAL_REPORT_DIR / "screenshots",
        run_started_at=run_started_at,
        username=args.grafana_user,
        password=args.grafana_password,
    )

    scenario_sections: list[str] = []
    for scenario in scenarios:
        api_dir = FINAL_REPORT_DIR / scenario.name / "api"
        drift_runs = (
            json.loads(
                (api_dir / "drift_runs.json").read_text(encoding="utf-8")
            )
            if (api_dir / "drift_runs.json").exists()
            else []
        )
        quality_runs = (
            json.loads(
                (api_dir / "quality_runs.json").read_text(encoding="utf-8")
            )
            if (api_dir / "quality_runs.json").exists()
            else []
        )
        incidents = (
            json.loads(
                (api_dir / "incidents.json").read_text(encoding="utf-8")
            )
            if (api_dir / "incidents.json").exists()
            else []
        )
        scenario_sections.append(
            scenario_markdown(
                name=scenario.name,
                segment_key=scenario.segment_key,
                drift_runs=drift_runs,
                quality_runs=quality_runs,
                incidents=incidents,
            )
        )

    write_top_level_readme(
        output_dir=FINAL_REPORT_DIR,
        generated_at=datetime.now(UTC),
        scenarios=scenarios,
        scenario_sections=scenario_sections,
        screenshot_names=[item["name"] for item in screenshots],
    )

    logger.info(
        "Final experiment export completed. output_dir=%s", FINAL_REPORT_DIR
    )
    print(
        json.dumps(
            [asdict(item) for item in scenarios], ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
