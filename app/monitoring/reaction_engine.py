from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.common.config import settings
from app.common.logging import get_logger

ACTION_TIGHTEN_THRESHOLD = "tighten_threshold"
ACTION_MANUAL_REVIEW = "manual_review"
ACTION_TYPES = {ACTION_TIGHTEN_THRESHOLD, ACTION_MANUAL_REVIEW}
ACTION_STATUS_EXECUTED = "executed"
ACTION_STATUS_DRY_RUN = "dry_run"
ACTION_STATUS_ROLLED_BACK = "rolled_back"
ACTION_STATUS_SKIPPED = "skipped"
AUTOMATION_TITLES = {
    "Drift monitoring signal",
    "Quality degradation detected",
    "Quality risk detected from proxy signals",
}

logger = get_logger(__name__)


def _to_json_compatible(value: Any) -> Any:
    """Normalize nested values so PostgreSQL JSONB accepts the payload."""

    if isinstance(value, dict):
        return {
            str(key): _to_json_compatible(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _to_json_compatible(value.item())
        except (TypeError, ValueError):
            return value
    return value


def _safe_json(payload: dict[str, Any]) -> str:
    """Serialize a nested config payload for JSONB storage."""

    return json.dumps(
        _to_json_compatible(payload),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def parse_json_object(value: Any) -> dict[str, Any] | None:
    """Parse a JSON/JSONB value into a dictionary when possible."""

    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": value}


def load_baseline_threshold() -> float:
    """Load the training-time threshold used as the policy baseline."""

    with settings.baseline_path.open("r", encoding="utf-8") as file_obj:
        baseline_profile = json.load(file_obj)
    return float(baseline_profile.get("threshold", 0.5))


def get_incident_by_id(engine: Engine, incident_id: int) -> dict[str, Any]:
    """Load one monitoring incident row and parse JSON payloads."""

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
            mitigation_taken,
            summary_json
        FROM monitoring_incidents
        WHERE id = :incident_id
        """
    )
    with engine.connect() as connection:
        row = (
            connection.execute(query, {"incident_id": incident_id})
            .mappings()
            .first()
        )
    if row is None:
        raise ValueError(f"Monitoring incident not found: {incident_id}")

    incident = dict(row)
    incident["summary_json"] = parse_json_object(incident.get("summary_json"))
    return incident


def get_open_incident_by_key(
    engine: Engine, incident_key: str
) -> dict[str, Any] | None:
    """Return the latest open incident for one key."""

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
            mitigation_taken,
            summary_json
        FROM monitoring_incidents
        WHERE incident_key = :incident_key AND status = 'open'
        ORDER BY ts_updated DESC, id DESC
        LIMIT 1
        """
    )
    with engine.connect() as connection:
        row = (
            connection.execute(query, {"incident_key": incident_key})
            .mappings()
            .first()
        )
    if row is None:
        return None
    incident = dict(row)
    incident["summary_json"] = parse_json_object(incident.get("summary_json"))
    return incident


def get_monitoring_action(engine: Engine, action_id: int) -> dict[str, Any]:
    """Load one monitoring action row with parsed configs."""

    query = text(
        """
        SELECT
            action_id,
            incident_id,
            action_type,
            status,
            started_at,
            ended_at,
            trigger_reason,
            old_config,
            new_config
        FROM monitoring_actions
        WHERE action_id = :action_id
        """
    )
    with engine.connect() as connection:
        row = (
            connection.execute(query, {"action_id": action_id})
            .mappings()
            .first()
        )
    if row is None:
        raise ValueError(f"Monitoring action not found: {action_id}")
    return _deserialize_action_row(dict(row))


def _deserialize_action_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize action row payloads after a SQL query."""

    row["old_config"] = parse_json_object(row.get("old_config"))
    row["new_config"] = parse_json_object(row.get("new_config"))
    return row


def _get_active_action_for_incident(
    engine: Engine, incident_id: int, action_type: str
) -> dict[str, Any] | None:
    """Return the currently active action of one type for an incident."""

    query = text(
        """
        SELECT
            action_id,
            incident_id,
            action_type,
            status,
            started_at,
            ended_at,
            trigger_reason,
            old_config,
            new_config
        FROM monitoring_actions
        WHERE incident_id = :incident_id
          AND action_type = :action_type
          AND status = :status
          AND ended_at IS NULL
        ORDER BY started_at DESC, action_id DESC
        LIMIT 1
        """
    )
    with engine.connect() as connection:
        row = (
            connection.execute(
                query,
                {
                    "incident_id": incident_id,
                    "action_type": action_type,
                    "status": ACTION_STATUS_EXECUTED,
                },
            )
            .mappings()
            .first()
        )
    if row is None:
        return None
    return _deserialize_action_row(dict(row))


def resolve_runtime_policy(
    engine: Engine,
    *,
    model_version: str,
    baseline_threshold: float,
    segment_key: str | None,
    exclude_action_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Resolve the active threshold and review policy for one segment."""

    query = text(
        """
        SELECT
            monitoring_actions.action_id,
            monitoring_actions.action_type,
            monitoring_actions.new_config,
            monitoring_incidents.segment_key
        FROM monitoring_actions
        JOIN monitoring_incidents
            ON monitoring_incidents.id = monitoring_actions.incident_id
        WHERE monitoring_actions.status = :status
          AND monitoring_actions.ended_at IS NULL
          AND monitoring_incidents.model_version = :model_version
        ORDER BY monitoring_actions.started_at ASC,
                 monitoring_actions.action_id ASC
        """
    )

    with engine.connect() as connection:
        rows = (
            connection.execute(
                query,
                {
                    "status": ACTION_STATUS_EXECUTED,
                    "model_version": model_version,
                },
            )
            .mappings()
            .all()
        )

    exclude_action_ids = exclude_action_ids or set()
    threshold = float(baseline_threshold)
    manual_review_probability = 0.0
    action_ids: list[int] = []

    for row in rows:
        action_id = int(row["action_id"])
        if action_id in exclude_action_ids:
            continue

        action_segment = row["segment_key"]
        if action_segment is not None and action_segment != segment_key:
            continue

        config = parse_json_object(row["new_config"]) or {}
        action_type = str(row["action_type"])
        if action_type == ACTION_TIGHTEN_THRESHOLD:
            threshold = max(
                threshold, float(config.get("threshold", threshold))
            )
            action_ids.append(action_id)
        elif action_type == ACTION_MANUAL_REVIEW:
            manual_review_probability = max(
                manual_review_probability,
                float(
                    config.get(
                        "manual_review_probability",
                        manual_review_probability,
                    )
                ),
            )
            action_ids.append(action_id)

    return {
        "threshold": float(threshold),
        "manual_review_probability": min(
            max(float(manual_review_probability), 0.0), 1.0
        ),
        "action_ids": action_ids,
    }


def should_route_to_manual_review(request_id: str, probability: float) -> bool:
    """Deterministically sample a request into manual review."""

    bounded_probability = min(max(float(probability), 0.0), 1.0)
    if bounded_probability <= 0.0:
        return False
    if bounded_probability >= 1.0:
        return True

    digest = hashlib.sha256(request_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
    sample = bucket / float(2**64 - 1)
    return sample < bounded_probability


def _choose_automatic_action_type(
    engine: Engine, incident: dict[str, Any]
) -> dict[str, Any] | None:
    """Choose one safe automatic mitigation for an incident."""

    source_type = str(incident["source_type"])
    incident_id = int(incident["id"])

    if source_type == "quality":
        threshold_action = _get_active_action_for_incident(
            engine, incident_id, ACTION_TIGHTEN_THRESHOLD
        )
        if threshold_action is None:
            return {"action_type": ACTION_TIGHTEN_THRESHOLD}

        manual_review_action = _get_active_action_for_incident(
            engine, incident_id, ACTION_MANUAL_REVIEW
        )
        if manual_review_action is None:
            return {"action_type": ACTION_MANUAL_REVIEW}

        return manual_review_action

    manual_review_action = _get_active_action_for_incident(
        engine, incident_id, ACTION_MANUAL_REVIEW
    )
    if manual_review_action is not None:
        return manual_review_action
    return {"action_type": ACTION_MANUAL_REVIEW}


def _build_action_configs(
    *,
    incident: dict[str, Any],
    current_policy: dict[str, Any],
    baseline_threshold: float,
    action_type: str,
    threshold_step: float | None = None,
    manual_review_probability: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build old/new config snapshots for a concrete action."""

    old_config = {
        "segment_key": incident["segment_key"],
        "threshold": float(current_policy["threshold"]),
        "manual_review_probability": float(
            current_policy["manual_review_probability"]
        ),
        "baseline_threshold": float(baseline_threshold),
    }
    new_config = dict(old_config)

    if action_type == ACTION_TIGHTEN_THRESHOLD:
        step = float(
            threshold_step
            if threshold_step is not None
            else settings.reaction_threshold_step
        )
        new_threshold = min(
            float(current_policy["threshold"]) + step,
            float(settings.reaction_threshold_cap),
        )
        if new_threshold <= float(current_policy["threshold"]):
            raise ValueError(
                "Threshold is already at or above the configured cap."
            )
        new_config["threshold"] = round(float(new_threshold), 6)
        return old_config, new_config

    if action_type == ACTION_MANUAL_REVIEW:
        target_probability = (
            float(manual_review_probability)
            if manual_review_probability is not None
            else float(settings.reaction_manual_review_probability)
        )
        bounded_probability = min(max(target_probability, 0.0), 1.0)
        new_probability = max(
            float(current_policy["manual_review_probability"]),
            bounded_probability,
        )
        if new_probability <= float(
            current_policy["manual_review_probability"]
        ):
            raise ValueError(
                "Manual review probability is already at the requested level."
            )
        new_config["manual_review_probability"] = round(
            float(new_probability), 6
        )
        return old_config, new_config

    raise ValueError(f"Unsupported monitoring action type: {action_type}")


def _build_trigger_reason(
    incident: dict[str, Any],
    action_type: str,
    trigger_reason: str | None,
) -> str:
    """Build a short action audit reason."""

    if trigger_reason:
        return trigger_reason
    return (
        f"{incident['severity']} {incident['source_type']} incident "
        f"#{incident['id']} triggered {action_type}"
    )


def _insert_action_record(
    engine: Engine,
    *,
    incident_id: int,
    action_type: str,
    status: str,
    trigger_reason: str,
    old_config: dict[str, Any],
    new_config: dict[str, Any],
    ended_immediately: bool,
) -> dict[str, Any]:
    """Insert one action audit record and return the parsed row."""

    query = text(
        """
        INSERT INTO monitoring_actions (
            incident_id,
            action_type,
            status,
            started_at,
            ended_at,
            trigger_reason,
            old_config,
            new_config
        )
        VALUES (
            :incident_id,
            :action_type,
            :status,
            NOW(),
            CASE WHEN :ended_immediately THEN NOW() ELSE NULL END,
            :trigger_reason,
            CAST(:old_config AS JSONB),
            CAST(:new_config AS JSONB)
        )
        RETURNING
            action_id,
            incident_id,
            action_type,
            status,
            started_at,
            ended_at,
            trigger_reason,
            old_config,
            new_config
        """
    )

    with engine.begin() as connection:
        row = (
            connection.execute(
                query,
                {
                    "incident_id": incident_id,
                    "action_type": action_type,
                    "status": status,
                    "ended_immediately": ended_immediately,
                    "trigger_reason": trigger_reason,
                    "old_config": _safe_json(old_config),
                    "new_config": _safe_json(new_config),
                },
            )
            .mappings()
            .one()
        )

    return _deserialize_action_row(dict(row))


def _update_incident_mitigation(
    engine: Engine, incident_id: int, mitigation_taken: str
) -> None:
    """Persist the latest mitigation note on the incident itself."""

    query = text(
        """
        UPDATE monitoring_incidents
        SET
            mitigation_taken = :mitigation_taken,
            ts_updated = NOW()
        WHERE id = :incident_id
        """
    )
    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "incident_id": incident_id,
                "mitigation_taken": mitigation_taken,
            },
        )


def execute_monitoring_action(
    engine: Engine,
    *,
    incident_id: int,
    action_type: str | None = None,
    dry_run: bool,
    threshold_step: float | None = None,
    manual_review_probability: float | None = None,
    trigger_reason: str | None = None,
    require_critical: bool = False,
) -> dict[str, Any]:
    """Execute or simulate one monitoring mitigation action."""

    incident = get_incident_by_id(engine, incident_id)
    if str(incident["status"]) != "open":
        raise ValueError("Monitoring actions can only target open incidents.")
    if require_critical and str(incident["severity"]) != "critical":
        raise ValueError(
            "Automatic reaction requires an open critical incident."
        )

    if action_type is None:
        action_choice = _choose_automatic_action_type(engine, incident)
        if action_choice is None:
            raise ValueError(
                "No automatic action is available for the incident."
            )
        if "action_id" in action_choice:
            return action_choice
        action_type = str(action_choice["action_type"])

    if action_type not in ACTION_TYPES:
        raise ValueError(f"Unsupported monitoring action type: {action_type}")

    existing_action = _get_active_action_for_incident(
        engine, incident_id, action_type
    )
    if existing_action is not None:
        return existing_action

    baseline_threshold = load_baseline_threshold()
    current_policy = resolve_runtime_policy(
        engine,
        model_version=str(incident["model_version"]),
        baseline_threshold=baseline_threshold,
        segment_key=incident["segment_key"],
    )
    old_config, new_config = _build_action_configs(
        incident=incident,
        current_policy=current_policy,
        baseline_threshold=baseline_threshold,
        action_type=action_type,
        threshold_step=threshold_step,
        manual_review_probability=manual_review_probability,
    )
    status = ACTION_STATUS_DRY_RUN if dry_run else ACTION_STATUS_EXECUTED
    action = _insert_action_record(
        engine,
        incident_id=incident_id,
        action_type=action_type,
        status=status,
        trigger_reason=_build_trigger_reason(
            incident, action_type, trigger_reason
        ),
        old_config=old_config,
        new_config=new_config,
        ended_immediately=dry_run,
    )

    if not dry_run:
        _update_incident_mitigation(
            engine,
            incident_id,
            (
                f"Executed {action_type} action #{action['action_id']} for "
                f"segment {incident['segment_key'] or '__global__'}."
            ),
        )

    logger.info(
        (
            "Monitoring action %s created. action_id=%s incident_id=%s "
            "dry_run=%s segment_key=%s"
        ),
        action_type,
        action["action_id"],
        incident_id,
        dry_run,
        incident["segment_key"],
    )
    return action


def rollback_monitoring_action(
    engine: Engine,
    *,
    action_id: int,
    dry_run: bool,
    trigger_reason: str | None = None,
) -> dict[str, Any]:
    """Rollback one previously executed monitoring action."""

    original_action = get_monitoring_action(engine, action_id)
    if str(original_action["status"]) != ACTION_STATUS_EXECUTED:
        raise ValueError("Only active executed actions can be rolled back.")
    if original_action["ended_at"] is not None:
        raise ValueError("Action is already ended and cannot be rolled back.")

    incident = get_incident_by_id(engine, int(original_action["incident_id"]))
    baseline_threshold = load_baseline_threshold()
    current_policy = resolve_runtime_policy(
        engine,
        model_version=str(incident["model_version"]),
        baseline_threshold=baseline_threshold,
        segment_key=incident["segment_key"],
    )
    restored_policy = resolve_runtime_policy(
        engine,
        model_version=str(incident["model_version"]),
        baseline_threshold=baseline_threshold,
        segment_key=incident["segment_key"],
        exclude_action_ids={int(original_action["action_id"])},
    )

    rollback_action_type = f"rollback_{original_action['action_type']}"
    rollback_reason = trigger_reason or (
        f"Rollback requested for action #{original_action['action_id']}"
    )

    if dry_run:
        return _insert_action_record(
            engine,
            incident_id=int(original_action["incident_id"]),
            action_type=rollback_action_type,
            status=ACTION_STATUS_DRY_RUN,
            trigger_reason=rollback_reason,
            old_config={
                "segment_key": incident["segment_key"],
                "threshold": current_policy["threshold"],
                "manual_review_probability": current_policy[
                    "manual_review_probability"
                ],
            },
            new_config={
                "segment_key": incident["segment_key"],
                "threshold": restored_policy["threshold"],
                "manual_review_probability": restored_policy[
                    "manual_review_probability"
                ],
                "rollback_of_action_id": int(original_action["action_id"]),
            },
            ended_immediately=True,
        )

    update_original_query = text(
        """
        UPDATE monitoring_actions
        SET
            status = :status,
            ended_at = NOW()
        WHERE action_id = :action_id
        """
    )
    insert_query = text(
        """
        INSERT INTO monitoring_actions (
            incident_id,
            action_type,
            status,
            started_at,
            ended_at,
            trigger_reason,
            old_config,
            new_config
        )
        VALUES (
            :incident_id,
            :action_type,
            :status,
            NOW(),
            NOW(),
            :trigger_reason,
            CAST(:old_config AS JSONB),
            CAST(:new_config AS JSONB)
        )
        RETURNING
            action_id,
            incident_id,
            action_type,
            status,
            started_at,
            ended_at,
            trigger_reason,
            old_config,
            new_config
        """
    )

    with engine.begin() as connection:
        connection.execute(
            update_original_query,
            {
                "status": ACTION_STATUS_ROLLED_BACK,
                "action_id": int(original_action["action_id"]),
            },
        )
        rollback_row = (
            connection.execute(
                insert_query,
                {
                    "incident_id": int(original_action["incident_id"]),
                    "action_type": rollback_action_type,
                    "status": ACTION_STATUS_EXECUTED,
                    "trigger_reason": rollback_reason,
                    "old_config": _safe_json(
                        {
                            "segment_key": incident["segment_key"],
                            "threshold": current_policy["threshold"],
                            "manual_review_probability": current_policy[
                                "manual_review_probability"
                            ],
                        }
                    ),
                    "new_config": _safe_json(
                        {
                            "segment_key": incident["segment_key"],
                            "threshold": restored_policy["threshold"],
                            "manual_review_probability": restored_policy[
                                "manual_review_probability"
                            ],
                            "rollback_of_action_id": int(
                                original_action["action_id"]
                            ),
                        }
                    ),
                },
            )
            .mappings()
            .one()
        )

    rollback_action = _deserialize_action_row(dict(rollback_row))
    _update_incident_mitigation(
        engine,
        int(original_action["incident_id"]),
        (
            f"Rolled back action #{original_action['action_id']} via "
            f"rollback action #{rollback_action['action_id']}."
        ),
    )
    return rollback_action


def maybe_execute_critical_reaction(
    engine: Engine, *, incident_key: str
) -> dict[str, Any] | None:
    """Run the configured automatic reaction for one open critical incident."""

    if not settings.reaction_engine_enabled:
        return None

    incident = get_open_incident_by_key(engine, incident_key)
    if incident is None:
        return None
    if str(incident["severity"]) != "critical":
        return None
    if str(incident["title"]) not in AUTOMATION_TITLES:
        return None

    dry_run = settings.reaction_engine_mode.lower() != "real"
    try:
        return execute_monitoring_action(
            engine,
            incident_id=int(incident["id"]),
            action_type=None,
            dry_run=dry_run,
            trigger_reason=(
                "Automatic reaction for critical monitoring incident"
            ),
            require_critical=True,
        )
    except ValueError as exc:
        logger.info(
            "Skipping automatic reaction for incident_id=%s: %s",
            incident["id"],
            exc,
        )
        return None
