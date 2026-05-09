# Чеклист защиты

Основной сценарий защиты: четыре команды, три дашборда, два SQL-запроса,
один инцидент и один rollback.

## Перед началом

- Стек запущен.
- `artifacts/reports/final/` доступен для записи.
- Учетные данные Grafana: `admin/admin`.

Если стек устарел, пересоберите его перед защитой:

```bash
docker compose up -d --build --force-recreate
```

## Четыре команды

1. Сгенерировать финальный пакет экспериментов:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.reporting.export_final_artifacts \
  --grafana-user admin \
  --grafana-password admin
```

2. Собрать агрегированную таблицу доказательств:

```bash
make final-summary
```

3. Выполнить реальное смягчающее действие для последнего открытого
   критического инцидента качества:

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

4. Откатить действие после демонстрации:

```bash
curl -X POST http://localhost:8000/monitoring/actions/rollback \
  -H 'Content-Type: application/json' \
  -d '{
    "action_id": <action_id>,
    "mode": "real",
    "trigger_reason": "defense rollback"
  }'
```

`incident_id` возьмите из SQL-запроса 2, а `action_id` из ответа команды 3.

## Три дашборда

1. Общий обзор:
   `http://localhost:3000/d/ml-monitoring-overview`
   Покажите severity, свежесть данных и активные инциденты.

2. Анализ дрейфа:
   `http://localhost:3000/d/ml-monitoring-drift`
   Покажите baseline, слабый/сильный дрейф и строку многомерного детектора.

3. Качество и калибровка:
   `http://localhost:3000/d/ml-monitoring-quality`
   Покажите деградацию по меткам и блок data quality.

Для истории слепого периода используйте `artifacts/reports/final/proxy/` и
`artifacts/reports/final/screenshots/proxy_label_coverage.png`.

## Два SQL-запроса

1. Сравнить прокси-оценку с поздней истинной метрикой по меткам:

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

2. Выбрать инцидент для live-действия и проверить аудит actions:

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

## Какой инцидент показать

Используйте последний открытый `critical` инцидент качества для финального
proxy- или severe-сценария:

- деградация уже есть и содержательна;
- система уже создала инцидент;
- `tighten_threshold` легко объяснить;
- rollback наглядно замыкает контур реагирования.

## Какой rollback показать

Откатите ровно то действие, которое создали командой 3. Покажите:

- возвращенный `action_id`;
- новую audit-строку rollback в `monitoring_actions`;
- заполненный `ended_at` у исходного действия;
- восстановленную runtime policy.

## Рассказ

- Baseline остается стабильным; слабый и сильный сценарии деградируют в
  ожидаемом порядке.
- В слепом периоде прокси-сигналы и оценки без меток реагируют раньше, чем
  приходят отложенные метки.
- Инциденты actionable: система записывает реальное смягчающее действие и
  реальный rollback.
- Таблица доказательств лежит в `artifacts/reports/final/scenario_summary.md`.
