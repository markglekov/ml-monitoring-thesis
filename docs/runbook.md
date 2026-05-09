# Runbook операторского реагирования

Документ описывает, что делать при warning/critical алертах мониторинга.
Пороги и action policy вынесены в `monitoring/monitoring_config.yaml`;
Alertmanager доставляет алерты, а Grafana и monitoring API дают контекст.

## Быстрая диагностика

1. Откройте обзор: `http://localhost:8000/overview`.
2. Проверьте активные инциденты:

```bash
curl http://localhost:8000/monitoring/incidents?limit=20
```

3. Откройте релевантный дашборд:
   drift: `http://localhost:3000/d/ml-monitoring-drift`,
   quality: `http://localhost:3000/d/ml-monitoring-quality`,
   proxy: `http://localhost:3000/d/ml-monitoring-proxy`,
   segments: `http://localhost:3000/d/ml-monitoring-segments`.
4. Проверьте свежесть jobs и объем окна: если данных мало или запуск устарел,
   сначала восстановите scheduler/API/DB, а не меняйте модель.

## Drift Warning

Сигнал: открыт warning-инцидент `source_type=drift`.

Действия:

- Посмотрите последние `monitoring_runs` и `drift_metrics`; проверьте
  `segment_key`, `drifted_features_count`, `psi_value`, `pvalue_adj`.
- Сравните affected segment с глобальным потоком: локальный дрейф не всегда
  требует изменения модели.
- Проверьте upstream-изменения: schema, preprocessing, источник трафика,
  маркетинговую кампанию, новую популяцию пользователей.
- Не меняйте threshold сразу; дождитесь следующего окна или соберите свежие
  метки, если сдвиг сохраняется.

Критерий закрытия: следующий drift run возвращает `severity=none`, либо
известная бизнес-причина сдвига подтверждена и задокументирована.

## Drift Critical

Сигнал: открыт critical-инцидент `source_type=drift`.

Действия:

- Немедленно проверьте дашборд дрейфа и timeline инцидентов по сегменту.
- Остановите автоматическое принятие рискованных решений для затронутого
  сегмента: используйте `manual_review`, если reaction engine включен.
- Сравните score drift и feature drift: если меняются и признаки, и score,
  готовьте rollback данных/preprocessing или переобучение.
- Эскалируйте владельцу модели и владельцу upstream data pipeline.

Критерий закрытия: причина устранена, новый run показывает `severity=none`
или warning, а manual-review/rollback действия зафиксированы в
`monitoring_actions`.

## Quality Warning

Сигнал: warning-инцидент `source_type=quality` при наличии меток.

Действия:

- Проверьте `quality_runs` и `quality_metrics`: какие метрики деградировали,
  насколько велик `delta_value`, какой `baseline_source` использован.
- Сравните precision/recall/F1 с business-cost: возможно, нужен не retrain,
  а пересмотр operating threshold.
- Проверьте свежесть и качество delayed labels: лаг, дубликаты, сдвиг
  positive rate, ошибки backfill.
- Запланируйте повторный run после следующего окна меток.

Критерий закрытия: деградировавшие метрики возвращаются в допустимый диапазон
или принято явное решение о threshold/recalibration/retraining.

## Quality Critical

Сигнал: critical-инцидент `source_type=quality` при наличии меток.

Действия:

- Проверьте affected segment и список деградировавших метрик.
- Если ошибка влияет на пользователя, включите `tighten_threshold` или
  `manual_review` через monitoring actions.
- Проверьте calibration (`ece`, `brier_score`) отдельно от ranking quality
  (`roc_auc`, `pr_auc`): разные причины требуют разных исправлений.
- Подготовьте rollback, recalibration или retraining. Все действия должны
  попасть в `monitoring_actions`.

Критерий закрытия: mitigation выполнен или откатан, новые quality runs больше
не показывают critical, а решение отражено в incident summary.

## Proxy Warning / Critical

Сигнал: quality-инцидент в слепом периоде, `status=completed_proxy`, меток
еще нет или покрытие метками низкое.

Действия:

- Проверьте `Label Coverage Over Time` и убедитесь, что проблема именно в
  слепом периоде, а не в падении label ingestion.
- Посмотрите `score_psi`, `near_threshold_rate`, `score_entropy`,
  `positive_rate_pred` и `quality_estimates`.
- Ускорьте сбор меток для affected segment; при critical переведите часть
  запросов на ручную проверку до прихода ground truth.

Критерий закрытия: метки пришли, labeled quality run подтверждает отсутствие
critical-деградации или запускается полноценный quality mitigation.

## Проверочные SQL-запросы

```sql
SELECT id, segment_key, status, overall_drift, drifted_features_count,
       summary_json ->> 'severity' AS severity,
       summary_json ->> 'recommended_action' AS recommended_action
FROM monitoring_runs
ORDER BY ts_started DESC
LIMIT 10;

SELECT id, segment_key, status, degraded_metrics_count, labeled_rows,
       summary_json ->> 'severity' AS severity,
       summary_json ->> 'recommended_action' AS recommended_action
FROM quality_runs
ORDER BY ts_started DESC
LIMIT 10;

SELECT id, source_type, segment_key, severity, status, title,
       recommended_action, ts_updated
FROM monitoring_incidents
ORDER BY ts_updated DESC
LIMIT 20;

SELECT action_id, incident_id, action_type, status, dry_run,
       old_config_json, new_config_json, started_at, ended_at
FROM monitoring_actions
ORDER BY action_id DESC
LIMIT 20;
```
