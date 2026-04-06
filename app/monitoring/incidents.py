from __future__ import annotations

import json
import math
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

SEVERITY_RANK: dict[str, int] = {
    "none": 0,
    "info": 1,
    "warning": 2,
    "critical": 3,
}


def severity_rank(value: str | None) -> int:
    """Return a stable integer rank for one severity string."""

    if value is None:
        return 0
    return SEVERITY_RANK.get(value, 0)


def highest_severity(values: list[str]) -> str:
    """Return the highest severity from a list of values."""

    best = "none"
    for value in values:
        if severity_rank(value) > severity_rank(best):
            best = value
    return best


def build_incident_key(source_type: str, segment_key: str | None) -> str:
    """Build a stable incident key for one source and segment."""

    segment_value = segment_key or "__global__"
    return f"{source_type}:{segment_value}"


def _safe_json(payload: dict[str, Any]) -> str:
    """Serialize incident payloads to JSON for PostgreSQL JSONB columns."""

    return json.dumps(
        _to_json_compatible(payload),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def _to_json_compatible(value: Any) -> Any:
    """Normalize nested values so PostgreSQL JSONB accepts the payload."""

    if isinstance(value, dict):
        return {
            str(key): _to_json_compatible(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _to_json_compatible(value.item())
        except (TypeError, ValueError):
            pass
    return value


def sync_monitoring_incident(
    engine: Engine,
    *,
    incident_key: str,
    source_type: str,
    model_version: str,
    segment_key: str | None,
    severity: str,
    title: str,
    recommended_action: str,
    summary: dict[str, Any],
    latest_run_id: int,
) -> None:
    """Open, update, or resolve an incident for the provided source."""

    select_query = text(
        """
        SELECT id
        FROM monitoring_incidents
        WHERE incident_key = :incident_key AND status = 'open'
        ORDER BY ts_opened DESC, id DESC
        LIMIT 1
        """
    )
    update_query = text(
        """
        UPDATE monitoring_incidents
        SET
            severity = :severity,
            title = :title,
            recommended_action = :recommended_action,
            summary_json = CAST(:summary_json AS JSONB),
            latest_run_id = :latest_run_id,
            ts_updated = NOW()
        WHERE id = :incident_id
        """
    )
    resolve_query = text(
        """
        UPDATE monitoring_incidents
        SET
            status = 'resolved',
            summary_json = CAST(:summary_json AS JSONB),
            latest_run_id = :latest_run_id,
            ts_updated = NOW(),
            ts_resolved = NOW()
        WHERE id = :incident_id
        """
    )
    insert_query = text(
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
            :incident_key,
            :source_type,
            :model_version,
            :segment_key,
            'open',
            :severity,
            :title,
            :recommended_action,
            CAST(:summary_json AS JSONB),
            :latest_run_id
        )
        """
    )

    with engine.begin() as connection:
        incident_row = (
            connection.execute(select_query, {"incident_key": incident_key})
            .mappings()
            .first()
        )

        if severity_rank(severity) <= severity_rank("info"):
            if incident_row is not None:
                connection.execute(
                    resolve_query,
                    {
                        "incident_id": int(incident_row["id"]),
                        "summary_json": _safe_json(summary),
                        "latest_run_id": latest_run_id,
                    },
                )
            return

        payload = {
            "incident_key": incident_key,
            "source_type": source_type,
            "model_version": model_version,
            "segment_key": segment_key,
            "severity": severity,
            "title": title,
            "recommended_action": recommended_action,
            "summary_json": _safe_json(summary),
            "latest_run_id": latest_run_id,
        }

        if incident_row is not None:
            connection.execute(
                update_query,
                {
                    **payload,
                    "incident_id": int(incident_row["id"]),
                },
            )
            return

        connection.execute(insert_query, payload)


def list_monitoring_incidents(
    engine: Engine,
    *,
    limit: int,
    status: str | None = None,
    segment_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent incidents with optional status and segment filters."""

    clauses = []
    params: dict[str, Any] = {"limit": limit}

    if status:
        clauses.append("status = :status")
        params["status"] = status
    if segment_key:
        clauses.append("segment_key = :segment_key")
        params["segment_key"] = segment_key

    where_clause = ""
    if clauses:
        where_clause = "WHERE " + " AND ".join(clauses)

    query = text(
        f"""
        SELECT
            id,
            incident_key,
            source_type,
            model_version,
            segment_key,
            status,
            severity,
            title,
            recommended_action,
            summary_json,
            latest_run_id,
            acknowledged_by,
            mitigation_taken,
            ts_opened,
            ts_updated,
            ts_resolved
        FROM monitoring_incidents
        {where_clause}
        ORDER BY ts_updated DESC, id DESC
        LIMIT :limit
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query, params).mappings().all()

    return [dict(row) for row in rows]


def get_active_incidents(engine: Engine) -> list[dict[str, Any]]:
    """Return open incidents ordered by severity and freshness."""

    query = text(
        """
        SELECT
            id,
            incident_key,
            source_type,
            model_version,
            segment_key,
            status,
            severity,
            title,
            recommended_action,
            summary_json,
            latest_run_id,
            acknowledged_by,
            mitigation_taken,
            ts_opened,
            ts_updated,
            ts_resolved
        FROM monitoring_incidents
        WHERE status = 'open'
        ORDER BY
            CASE severity
                WHEN 'critical' THEN 3
                WHEN 'warning' THEN 2
                WHEN 'info' THEN 1
                ELSE 0
            END DESC,
            ts_updated DESC,
            id DESC
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]
