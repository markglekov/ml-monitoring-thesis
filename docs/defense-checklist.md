# Defense Checklist

Main defense flow: four commands, three dashboards, two SQL queries, one
incident, one rollback.

## Preflight

- Stack is running.
- `artifacts/reports/final/` is writable.
- Grafana credentials: `admin/admin`.

If the stack is stale, refresh it before the defense:

```bash
docker compose up -d --build --force-recreate
```

## Four Commands

1. Generate the final experiment pack:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.reporting.export_final_artifacts \
  --grafana-user admin \
  --grafana-password admin
```

2. Build the aggregated evidence table:

```bash
make final-summary
```

3. Execute a real mitigation on the latest open critical quality incident:

```bash
curl -X POST http://localhost:8000/monitoring/actions/execute \
  -H 'Content-Type: application/json' \
  -d '{
    "incident_id": <incident_id>,
    "action_type": "tighten_threshold",
    "mode": "real",
    "trigger_reason": "defense demo"
  }'
```

4. Roll the action back after the demo:

```bash
curl -X POST http://localhost:8000/monitoring/actions/rollback \
  -H 'Content-Type: application/json' \
  -d '{
    "action_id": <action_id>,
    "mode": "real",
    "trigger_reason": "defense rollback"
  }'
```

Use `incident_id` from SQL query 2 and `action_id` returned by command 3.

## Three Dashboards

1. Executive overview:
   `http://localhost:3000/d/ml-monitoring-overview`
   Show severity, freshness, and active incidents.

2. Drift analysis:
   `http://localhost:3000/d/ml-monitoring-drift`
   Show baseline vs mild vs severe and the multivariate detector row.

3. Quality and calibration:
   `http://localhost:3000/d/ml-monitoring-quality`
   Show labeled degradation and the data-quality block.

For the blind-period story, use `artifacts/reports/final/proxy/` and
`artifacts/reports/final/screenshots/proxy_label_coverage.png`.

## Two SQL Queries

1. Show the proxy estimate against the later true labeled metric:

```sql
WITH latest_proxy_run AS (
    SELECT id, segment_key
    FROM quality_runs
    WHERE model_version = 'bank_marketing_v1'
      AND segment_key LIKE 'final\_%\_proxy' ESCAPE '\'
      AND status = 'completed_proxy'
    ORDER BY ts_started DESC, id DESC
    LIMIT 1
),
latest_labeled_run AS (
    SELECT id
    FROM quality_runs
    WHERE model_version = 'bank_marketing_v1'
      AND segment_key = (SELECT segment_key FROM latest_proxy_run)
      AND status = 'completed'
    ORDER BY ts_started DESC, id DESC
    LIMIT 1
)
SELECT
    qe.assumption_type,
    qe.estimated_metric_name,
    qe.estimated_metric_value,
    qm.metric_value AS true_metric_value,
    ABS(qe.estimated_metric_value - qm.metric_value) AS abs_gap
FROM latest_proxy_run
JOIN quality_estimates qe
    ON qe.run_id = latest_proxy_run.id
JOIN latest_labeled_run
    ON TRUE
JOIN quality_metrics qm
    ON qm.run_id = latest_labeled_run.id
   AND qm.metric_name = qe.estimated_metric_name
ORDER BY abs_gap DESC, qe.estimated_metric_name ASC;
```

2. Pick the incident for the live action demo and verify the action audit:

```sql
SELECT
    mi.id AS incident_id,
    mi.segment_key,
    mi.source_type,
    mi.severity,
    mi.status,
    ma.action_id,
    ma.action_type,
    ma.status AS action_status,
    ma.started_at,
    ma.ended_at
FROM monitoring_incidents mi
LEFT JOIN monitoring_actions ma
    ON ma.incident_id = mi.id
WHERE mi.model_version = 'bank_marketing_v1'
  AND mi.segment_key LIKE 'final\_%' ESCAPE '\'
ORDER BY mi.ts_updated DESC, ma.action_id DESC NULLS LAST
LIMIT 15;
```

## Incident To Demonstrate

Use the latest open `critical` quality incident for the final proxy or severe
scenario:

- meaningful degradation already exists,
- the system already has an incident,
- `tighten_threshold` is easy to explain,
- rollback visibly closes the loop.

## Rollback To Demonstrate

Rollback the exact action created in command 3. Show:

- returned `action_id`,
- new rollback audit row in `monitoring_actions`,
- `ended_at` filled for the original action,
- restored runtime policy.

## Narrative

- Baseline stays stable; mild and severe scenarios degrade in the expected
  order.
- During the blind period, proxy signals and unlabeled estimates react before
  delayed labels arrive.
- Incidents are actionable: the system records a real mitigation and a real
  rollback.
- The evidence table is in `artifacts/reports/final/scenario_summary.md`.
