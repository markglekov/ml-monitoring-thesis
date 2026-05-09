# Экспериментальные сценарии

Документ фиксирует воспроизводимый набор экспериментов мониторинга для demo.
Самый короткий путь для защиты:

```bash
make demo
```

Эта команда обучает модель, поднимает Docker Compose stack, ждет API,
прогоняет сценарии, запускает monitoring jobs, экспортирует финальные
артефакты и собирает summary. Ручные команды ниже предполагают, что
Docker-стек уже запущен.

Реальные результаты для ВКР нужно сохранять в `artifacts/reports/final/`.
Каталог генерируется командой:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.reporting.export_final_artifacts
```

Экспорт записывает:

- manifests и API snapshots по сценариям в
  `artifacts/reports/final/{baseline,mild,severe,proxy,segment}/`;
- SQL-шаблоны и CSV-выгрузки из PostgreSQL для тех же сценариев;
- PNG-скриншоты Grafana в `artifacts/reports/final/screenshots/`;
- компактный markdown-индекс в `artifacts/reports/final/README.md`.

После сохранения запусков соберите короткую таблицу доказательств для защиты:

```bash
make final-summary
```

Команда записывает:

- `artifacts/reports/final/scenario_summary.csv`
- `artifacts/reports/final/scenario_summary.json`
- `artifacts/reports/final/scenario_summary.md`

Один зафиксированный пример результатов, пригодный для просмотра без
локального запуска demo, сохранен в [`docs/results_example.md`](results_example.md).

Настройки one-command demo:

- `DEMO_TRAIN=always|missing|skip`: переобучать модель всегда, только если
  нет артефактов, или пропустить обучение;
- `DEMO_DOCKER_UP=0`: использовать уже запущенный stack;
- `DEMO_RESET=1`: перед запуском удалить Docker volumes через
  `docker compose down -v`;
- `DEMO_SEED=42`: seed финальных сценариев;
- `API_URL` и `GRAFANA_URL`: адреса API и Grafana, если используются
  нестандартные порты.

## Предусловия

1. Артефакты модели существуют в `artifacts/models/` и
   `artifacts/baselines/`.
2. Стек запущен:

```bash
make docker-up
```

3. Команды выполняются из корня репозитория.

## Сценарии дрейфа

Симулятор записывает manifests в `artifacts/reports/manifests/`. Используйте
стабильный `--seed`, чтобы сценарии были воспроизводимыми.

### Baseline-сценарий

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.simulator.generate_stream \
  --source-split train \
  --rows 300 \
  --scenario none \
  --segment-key drift_none_demo \
  --seed 42

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.drift_job \
  --window-size 300 \
  --segment-key drift_none_demo \
  --min-rows 50
```

Ожидаемый результат:

- `overall_drift` обычно остается `false`;
- ложных срабатываний должно быть мало.

### Слабый дрейф

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.simulator.generate_stream \
  --source-split train \
  --rows 300 \
  --scenario mild \
  --segment-key drift_mild_demo \
  --seed 42

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.drift_job \
  --window-size 300 \
  --segment-key drift_mild_demo \
  --min-rows 50
```

Ожидаемый результат:

- появляется несколько признаков с дрейфом;
- `overall_drift` обычно становится `true`.

### Сильный дрейф

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.simulator.generate_stream \
  --source-split train \
  --rows 300 \
  --scenario severe \
  --segment-key drift_severe_demo \
  --seed 42

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.drift_job \
  --window-size 300 \
  --segment-key drift_severe_demo \
  --min-rows 50
```

Ожидаемый результат:

- дрейф заметен по большему числу признаков, чем в слабом сценарии;
- дрейф распределения score обычно тоже виден.

### Многомерный дрейф

Сгенерируйте сильный сегмент и проверьте детектор `__multivariate__` в
`drift_metrics` или `GET /monitoring/drift/runs`.

Ожидаемый результат:

- domain-classifier detector возвращает `warning` или `critical`;
- summary запуска содержит детали `multivariate_drift`.

## Сценарии качества

Мониторингу качества нужны отложенные метки. Переиспользуйте manifests,
сгенерированные симулятором, и меняйте только label policy.

### Baseline-качество

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.simulator.generate_stream \
  --rows 1000 \
  --scenario none \
  --segment-key quality_base_demo \
  --seed 42

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.backfill_labels \
  --label-policy perfect

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.quality_job \
  --window-size 1000 \
  --segment-key quality_base_demo \
  --baseline-source test
```

Ожидаемый результат:

- `degraded_metrics_count` остается близким к нулю.

### Слабое ухудшение качества

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.simulator.generate_stream \
  --rows 1000 \
  --scenario none \
  --segment-key quality_mild_demo \
  --seed 42

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.backfill_labels \
  --label-policy custom_flip \
  --flip-prob 0.10

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.quality_job \
  --window-size 1000 \
  --segment-key quality_mild_demo \
  --baseline-source test
```

Ожидаемый результат:

- появляются отдельные деградировавшие метрики;
- первыми могут сдвинуться `roc_auc`, `recall`, `f1` или `brier_score`.

### Сильное ухудшение качества

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.simulator.generate_stream \
  --rows 1000 \
  --scenario none \
  --segment-key quality_severe_demo \
  --seed 42

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.backfill_labels \
  --label-policy custom_flip \
  --flip-prob 0.25

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.quality_job \
  --window-size 1000 \
  --segment-key quality_severe_demo \
  --baseline-source test
```

Ожидаемый результат:

- `degraded_metrics_count` заметно выше, чем в слабом сценарии;
- сильная деградация видна в Grafana и overview API.

### Прокси-мониторинг в слепом периоде

Запустите только поток инференса и quality job без загрузки меток.

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.simulator.generate_stream \
  --rows 1000 \
  --scenario severe \
  --segment-key quality_proxy_demo \
  --seed 42

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.quality_job \
  --window-size 1000 \
  --segment-key quality_proxy_demo \
  --baseline-source test
```

Ожидаемый результат:

- quality job завершается со `status=completed_proxy`;
- proxy-метрики `score_psi`, `near_threshold_rate` или `score_entropy`
  могут деградировать до появления меток;
- `summary.unlabeled_quality_estimates` содержит оценки на основе допущения
  `assumption_type=label_shift`, например `positive_rate_true`, `precision`,
  `recall` или `f1`;
- открытый инцидент может появиться в `GET /monitoring/incidents`.

Чтобы сравнить оценку до backfill с истинной метрикой после прихода меток,
перезапустите тот же сегмент в два этапа:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.quality_job \
  --window-size 1000 \
  --segment-key quality_proxy_demo \
  --baseline-source test

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.backfill_labels \
  --segment-key quality_proxy_demo \
  --delay-minutes 60

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.quality_job \
  --window-size 1000 \
  --segment-key quality_proxy_demo \
  --baseline-source test
```

Затем сравните proxy-time оценки с более поздней метрикой по меткам:

```sql
WITH latest_proxy_run AS (
    SELECT id
    FROM quality_runs
    WHERE segment_key = 'quality_proxy_demo'
      AND status = 'completed_proxy'
    ORDER BY ts_started DESC, id DESC
    LIMIT 1
),
latest_labeled_run AS (
    SELECT id
    FROM quality_runs
    WHERE segment_key = 'quality_proxy_demo'
      AND status = 'completed'
    ORDER BY ts_started DESC, id DESC
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
WHERE qe.estimated_metric_name IN ('positive_rate_true', 'f1')
ORDER BY qe.estimated_metric_name ASC;
```

### Мониторинг по сегментам

Для scheduler задайте `MONITORING_SEGMENTS=drift_none_demo,drift_severe_demo`
или запускайте jobs вручную с `--segment-key`.

Ожидаемый результат:

- инциденты дрейфа или качества открываются только для затронутого сегмента;
- overview показывает активность по сегментам и активные инциденты.

### Автоматизированный reaction engine

Проект поддерживает минимальный, но реальный action engine вместо простого
сохранения текстового `recommended_action` в инцидентах.

Конфигурация:

- `REACTION_ENGINE_MODE=dry_run` создает audit-записи в `monitoring_actions`,
  но не меняет live-поведение инференса;
- `REACTION_ENGINE_MODE=real` активирует runtime policy, которую использует
  `/predict`.

Реализованы две безопасные реакции:

- `tighten_threshold`: временно поднять operating threshold;
- `manual_review`: направлять часть или все запросы на ручную проверку.

Чтобы явно проверить API:

```bash
curl -X POST http://localhost:8000/monitoring/actions/execute \
  -H 'Content-Type: application/json' \
  -d '{
    "incident_id": 1,
    "action_type": "tighten_threshold",
    "mode": "real"
  }'

curl -X POST http://localhost:8000/monitoring/actions/rollback \
  -H 'Content-Type: application/json' \
  -d '{
    "action_id": 1,
    "mode": "real"
  }'
```

Ожидаемый результат:

- в `monitoring_actions` создается строка;
- в режиме `real` критические инциденты качества ужесточают threshold;
- в режиме `real` критические инциденты дрейфа направляют matching requests
  в `manual_review`;
- rollback закрывает исходное действие и создает audit-запись отката.

## Где сохраняются результаты

- Сырые события предсказаний: `inference_log`
- Отложенные метки: `ground_truth`
- Запуски дрейфа: `monitoring_runs`, `drift_metrics`
- Запуски качества: `quality_runs`, `quality_metrics`, `quality_estimates`
- Инциденты мониторинга: `monitoring_incidents`
- Автоматизированные реакции: `monitoring_actions`
- Manifests симулятора: `artifacts/reports/manifests/`

## Как смотреть результаты

Overview и monitoring API:

- `GET /monitoring/overview`
- `GET /monitoring/drift/runs?limit=10`
- `GET /monitoring/quality/runs?limit=10`
- `GET /monitoring/incidents?limit=10`
- `POST /monitoring/actions/execute`
- `POST /monitoring/actions/rollback`

Prometheus / Grafana:

- `http://localhost:9090`
- `http://localhost:3000`
- `http://localhost:3000/d/ml-monitoring-overview`
- `http://localhost:3000/d/ml-monitoring-drift`
- `http://localhost:3000/d/ml-monitoring-quality`
- `http://localhost:3000/d/ml-monitoring-proxy`
- `http://localhost:3000/d/ml-monitoring-segments`

PostgreSQL:

```bash
make db-shell
```

Полезные запросы:

```sql
SELECT id, status, segment_key, overall_drift, drifted_features_count
FROM monitoring_runs
ORDER BY id DESC
LIMIT 10;

SELECT id, status, segment_key, degraded_metrics_count, labeled_rows
FROM quality_runs
ORDER BY id DESC
LIMIT 10;
```

## Demo-flow для защиты

Для самого короткого одностраничного сценария используйте
`docs/defense-checklist.md`. Раздел ниже дает расширенный narrative.

Дашборды Grafana намеренно используют auto-refresh `15s`, чтобы изменения
быстро появлялись во время защиты. Для production batch- или window-based
мониторинга лучше выбирать более медленную частоту обновления, согласованную
с расписанием drift и quality jobs.

Показывайте дашборды в таком порядке:

1. Начните с общего обзора: severity, активные инциденты, свежесть jobs и
   состояние системы видны сразу.
2. Откройте дашборд дрейфа и сравните baseline, слабый и сильный дрейф.
   Покажите таблицу сегментов и строку детектора `__multivariate__`.
3. Откройте дашборд качества: объясните деградацию по меткам, влияние порога,
   изменения калибровки через `roc_auc`, `f1`, `brier_score` и `ece`, затем
   покажите блок data quality для missing values, out-of-range numerics и
   unexpected categories.
4. Откройте proxy-дашборд для слепого периода, когда меток еще нет. Сначала
   покажите `Label Coverage Over Time`, затем объясните, как `score_psi`,
   `near_threshold_rate` или `score_entropy` реагируют до прихода меток.
5. Завершите дашбордом сегментов и инцидентов: он показывает локализацию
   проблемы по сегменту и сохранение recommended actions во времени.
