from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARDS_DIR = REPO_ROOT / "monitoring" / "grafana" / "dashboards"


def _load_dashboard(filename: str) -> dict[str, Any]:
    return json.loads((DASHBOARDS_DIR / filename).read_text())


def _find_panel(dashboard: dict[str, Any], title: str) -> dict[str, Any]:
    for panel in dashboard["panels"]:
        if panel.get("title") == title:
            return panel
    raise AssertionError(f"Panel not found: {title}")


def _threshold_values(panel: dict[str, Any]) -> list[float | None]:
    steps = panel["fieldConfig"]["defaults"]["thresholds"]["steps"]
    return [step["value"] for step in steps]


def test_proxy_segment_score_psi_thresholds_match_overview() -> None:
    overview_dashboard = _load_dashboard("ml-monitoring-overview.json")
    proxy_dashboard = _load_dashboard("ml-monitoring-proxy.json")

    overview_panel = _find_panel(overview_dashboard, "Latest Score PSI")
    proxy_segment_panel = _find_panel(
        proxy_dashboard, "Latest Score PSI By Segment"
    )

    assert _threshold_values(overview_panel) == [None, 0.2, 0.35]
    assert _threshold_values(proxy_segment_panel) == [None, 0.2, 0.35]


def test_proxy_dashboard_version_reflects_threshold_fix() -> None:
    proxy_dashboard = _load_dashboard("ml-monitoring-proxy.json")

    assert proxy_dashboard["version"] >= 4


def test_proxy_dashboard_includes_label_coverage_panel() -> None:
    proxy_dashboard = _load_dashboard("ml-monitoring-proxy.json")

    panel = _find_panel(proxy_dashboard, "Label Coverage Over Time")
    target = panel["targets"][0]

    assert panel["datasource"]["type"] == "postgres"
    assert target["format"] == "time_series"
    assert "LEFT JOIN ground_truth" in target["rawSql"]
    assert "COUNT(gt.request_id)" in target["rawSql"]


def test_quality_dashboard_includes_data_quality_block() -> None:
    quality_dashboard = _load_dashboard("ml-monitoring-quality.json")

    missing_panel = _find_panel(
        quality_dashboard, "Recent Missing Rate By Feature"
    )
    range_panel = _find_panel(
        quality_dashboard, "Recent Out-Of-Range Rate By Feature"
    )
    category_panel = _find_panel(
        quality_dashboard, "Recent Unknown Category Rate"
    )

    assert quality_dashboard["version"] >= 4
    assert (
        "ml_monitoring_data_quality_feature_missing_rate"
        in missing_panel["targets"][0]["expr"]
    )
    assert (
        "ml_monitoring_data_quality_numeric_out_of_range_rate"
        in range_panel["targets"][0]["expr"]
    )
    assert (
        "ml_monitoring_data_quality_unknown_category_rate"
        in category_panel["targets"][0]["expr"]
    )


def test_quality_dashboard_includes_unlabeled_quality_estimates_panel() -> (
    None
):
    quality_dashboard = _load_dashboard("ml-monitoring-quality.json")

    panel = _find_panel(
        quality_dashboard, "Latest Unlabeled Quality Estimates"
    )
    target = panel["targets"][0]

    assert panel["datasource"]["type"] == "postgres"
    assert target["format"] == "table"
    assert "quality_estimates" in target["rawSql"]
    assert "assumption_type" in target["rawSql"]
    assert "confidence_interval_json" in target["rawSql"]
