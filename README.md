# ML Monitoring Thesis

Сквозной прототип мониторинга ML-сервиса бинарной классификации. Проект
показывает обучение модели, онлайн-инференс, отложенные метки, многоуровневый
мониторинг дрейфа и качества, алертинг, трассировку инцидентов и компактный
UI поверх данных мониторинга.

## Исследовательский вклад

- Сквозной прототип для мониторинга с отложенными метками, а не только API
  модели: обучение, serving, логирование, мониторинг, инциденты, дашборды и
  действия реагирования находятся в одной воспроизводимой системе.
- Явный протокол слепого периода: прокси-метрики, покрытие метками и оценки
  качества без меток реализованы рядом с классической оценкой по меткам.
- Единая модель доказательств: PostgreSQL хранит сырые события, запуски,
  метрики, инциденты и действия; Prometheus и Grafana показывают временные
  ряды и операторские сводки.
- Воспроизводимый пакет экспериментов для защиты: финальные сценарии,
  скриншоты и агрегированные таблицы можно пересобрать из репозитория.
- Actionable alerts: операторский runbook описывает реакции на warning и
  critical drift/quality/proxy, а пороги вынесены в YAML-конфиг.

## Что входит

- `FastAPI` API инференса с записью запросов в PostgreSQL
- офлайн-обучение модели на наборе Bank Marketing
- загрузка отложенных меток и мониторинг качества
- мониторинг дрейфа с одномерными и многомерными детекторами
- мониторинг качества в слепом периоде по прокси-сигналам
- интеграция Prometheus, Grafana и Alertmanager
- SMTP-релей для доставки алертов по email
- инциденты мониторинга с рекомендуемыми действиями
- обзорный API и легкий UI для текущего состояния мониторинга
- unit- и integration-тесты

## Реализованные сценарии

- базовые запуски дрейфа и качества
- слабый и сильный дрейф распределения
- слабое и сильное ухудшение качества по меткам
- прокси-мониторинг в слепом периоде до прихода меток
- оценка качества без меток при допущении label shift
- мониторинг и инциденты по сегментам
- автоматизированное реагирование с выполнением действия и rollback
- экспорт финальных доказательств в `artifacts/reports/final/`

## Технологии

- Python 3.12
- `uv` для управления зависимостями
- `FastAPI`, `Pydantic`, `SQLAlchemy`
- `scikit-learn`, `pandas`, `numpy`, `scipy`
- PostgreSQL
- Prometheus, Grafana, Alertmanager
- Ruff, Pyright, Pytest

## Структура проекта

```text
app/
  api/            FastAPI-приложение, monitoring endpoints, overview UI
  common/         общая конфигурация, логирование, метрики
  monitoring/     задачи дрейфа/качества, scheduler, загрузка меток
  notifications/  email-релей для webhook-ов Alertmanager
  simulator/      генератор потока запросов
  train/          обучение модели и генерация артефактов
monitoring/       конфигурация Prometheus, Grafana, Alertmanager и порогов
sql/              SQL-схема для bootstrap базы
tests/            unit- и integration-тесты
docs/             runbook-и и отчеты по воспроизводимым экспериментам
```

## Связь с главами ВКР

- Глава 1, постановка задачи и методы:
  детекторы дрейфа, отложенные метки, слепой период, прокси-мониторинг,
  оценка качества без меток и логика реагирования.
- Глава 2, проектирование системы:
  API, модель хранения, scheduler, задачи мониторинга, инциденты, дашборды
  и архитектура в [`docs/architecture.md`](docs/architecture.md).
- Глава 3, эксперименты и оценка:
  воспроизводимые сценарии в [`docs/experiments.md`](docs/experiments.md),
  короткий сценарий защиты в [`docs/defense-checklist.md`](docs/defense-checklist.md)
  и финальные артефакты в `artifacts/reports/final/`.
- Пример зафиксированных результатов:
  [`docs/results_example.md`](docs/results_example.md) содержит таблицу
  сценариев, 3 скриншота Grafana и SQL-пример incident/action/rollback.
- Карточка модели и данных:
  [`docs/model-data-card.md`](docs/model-data-card.md) фиксирует target,
  признаки, дисбаланс классов, удаление `duration` и ограничения применимости.
- Операторский контур:
  [`docs/runbook.md`](docs/runbook.md) описывает действия при warning/critical
  алертах, а `monitoring/monitoring_config.yaml` хранит thresholds, window
  sizes, severity rules и action policy.

## Быстрый старт

1. Скопируйте `.env.example` в `.env` и заполните нужные значения.
2. Установите зависимости:

```bash
make sync
```

3. Обучите модель и сгенерируйте baseline-артефакты, если они еще не созданы:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.train.train
```

4. Запустите стек:

```bash
make docker-up
```

Чтобы запустить только API локально, без Docker:

```bash
make run-api
```

Если порт `5432` уже занят, переопределите порт PostgreSQL на хосте:

```bash
POSTGRES_PORT=55432 docker compose up -d --build
```

## Полезные endpoints

- документация API: `http://localhost:8000/docs`
- проверка здоровья: `GET /health`
- метрики: `GET /metrics`
- обзор JSON: `GET /monitoring/overview`
- обзорный UI: `GET /overview`
- последние запуски дрейфа: `GET /monitoring/drift/runs`
- последние запуски качества: `GET /monitoring/quality/runs`
- инциденты мониторинга: `GET /monitoring/incidents`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Alertmanager: `http://localhost:9093`

Дашборды Grafana:

- общий обзор: `http://localhost:3000/d/ml-monitoring-overview`
- анализ дрейфа: `http://localhost:3000/d/ml-monitoring-drift`
- качество и калибровка: `http://localhost:3000/d/ml-monitoring-quality`
- прокси-сигналы слепого периода: `http://localhost:3000/d/ml-monitoring-proxy`
- сегменты и инциденты: `http://localhost:3000/d/ml-monitoring-segments`

## Разработка

Установить или обновить локальное окружение:

```bash
make sync
```

Запустить локальную проверку качества:

```bash
make check
```

Установить git hooks:

```bash
make hooks-install
```

Запустить отдельные инструменты:

```bash
make format
make lint
make typecheck
make test
make test-integration
```

Запустить тот же quality gate, что и CI:

```bash
make ci
```

Подключиться к PostgreSQL:

```bash
make db-shell
```

Полезные SQL-запросы внутри `psql`:

```sql
SELECT COUNT(*) FROM inference_log;
SELECT COUNT(*) FROM ground_truth;
SELECT id, status, overall_drift, drifted_features_count
FROM monitoring_runs
ORDER BY id DESC
LIMIT 5;
SELECT id, status, degraded_metrics_count, labeled_rows
FROM quality_runs
ORDER BY id DESC
LIMIT 5;
```

## Быстрый demo-сценарий

Полный сценарий защиты одной командой:

```bash
make demo
```

Команда обучает модель и baseline-артефакты, поднимает Docker Compose stack,
ждет API, прогоняет финальные сценарии мониторинга, экспортирует
`artifacts/reports/final/` и собирает `scenario_summary`. Для повторного
запуска без переобучения используйте:

```bash
DEMO_TRAIN=skip make demo
```

Для чистого стенда можно явно удалить runtime volumes:

```bash
DEMO_RESET=1 make demo
```

`DEMO_RESET=1` удаляет данные PostgreSQL, Prometheus, Alertmanager и Grafana.

1. Запустите весь стек:

```bash
make docker-up
```

2. Сгенерируйте поток инференса, похожий на baseline:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.simulator.generate_stream \
  --source-split train \
  --rows 300 \
  --scenario none \
  --segment-key demo_none \
  --seed 42
```

3. Запустите мониторинг дрейфа на этом сегменте:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.drift_job \
  --window-size 300 \
  --segment-key demo_none
```

4. Загрузите отложенные метки и запустите мониторинг качества:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.backfill_labels \
  --label-policy perfect

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.quality_job \
  --window-size 300 \
  --segment-key demo_none
```

5. Проверьте результат:

- обзорный UI: `http://localhost:8000/overview`
- обзор JSON: `http://localhost:8000/monitoring/overview`
- Grafana: `http://localhost:3000`
- общий дашборд: `http://localhost:3000/d/ml-monitoring-overview`
- дашборд дрейфа: `http://localhost:3000/d/ml-monitoring-drift`
- дашборд качества: `http://localhost:3000/d/ml-monitoring-quality`
- дашборд прокси-сигналов: `http://localhost:3000/d/ml-monitoring-proxy`
- дашборд сегментов: `http://localhost:3000/d/ml-monitoring-segments`
- Prometheus: `http://localhost:9090`
- Alertmanager: `http://localhost:9093`

## Рабочий процесс мониторинга

Сгенерировать поток запросов:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.simulator.generate_stream --rows 300 --scenario none
```

Загрузить отложенные метки через API:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.backfill_labels --label-policy perfect
```

Запустить задачи мониторинга вручную:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.drift_job --window-size 300
UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.quality_job --window-size 300
```

Сервис scheduler может запускать обе задачи непрерывно через Docker Compose.
Переменная `MONITORING_SEGMENTS=segment_a,segment_b` включает запуск по
нескольким сегментам; глобальный запуск управляется через
`SCHEDULER_INCLUDE_GLOBAL_SEGMENT`.

Grafana заранее настроена с источниками Prometheus и PostgreSQL. Обзорные
и time-series панели читают Prometheus, а timelines инцидентов, таблицы
сегментов и детали последних запусков читают PostgreSQL. Такое разделение
оставляет алертинг простым и делает demo-дашборды полезными для защиты.

Воспроизводимые сценарии и ожидаемые результаты описаны в
[`docs/experiments.md`](docs/experiments.md).

Зафиксированный пример результатов с таблицей сценариев, скриншотами Grafana
и SQL-выгрузками находится в [`docs/results_example.md`](docs/results_example.md).

Архитектурная схема и поток данных описаны в
[`docs/architecture.md`](docs/architecture.md).

Краткий сценарий защиты с командами, дашбордами, SQL-запросами, выбором
инцидента и rollback описан в
[`docs/defense-checklist.md`](docs/defense-checklist.md).

Операторский порядок действий при алертах описан в
[`docs/runbook.md`](docs/runbook.md). Конфиг порогов и политик мониторинга
лежит в `monitoring/monitoring_config.yaml`; CLI-аргументы и env-переменные
Docker Compose могут переопределять его дефолты.

## Ограничения

- Прототип сфокусирован на одном кейсе бинарной классификации и одном
  наборе данных; это не benchmark по разным доменам.
- Мониторинг реализован как оконные batch-задачи, а не как полноценная
  распределенная streaming-платформа.
- Evidently и MLflow рассмотрены как альтернативы, но не включены в runtime:
  проект намеренно показывает собственный прозрачный контур
  PostgreSQL/Prometheus/Grafana с SQL-доказательствами и actionable incidents.
- Автоматические реакции намеренно консервативны: реализованы ужесточение
  порога и ручная проверка, а полное автоматическое переобучение вне рамок.
- Часть финальных артефактов защиты создается локально в
  `artifacts/reports/` и не хранится в Git, поэтому перед финальной
  презентацией или snapshot-ом их нужно пересобрать.

## Email-алертинг

В проекте есть SMTP-релей: он принимает webhook-и Alertmanager и отправляет
email. Для Gmail используйте пароль приложения и задайте переменные в `.env`:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM=your_email@gmail.com
ALERT_EMAIL_TO=your_email@gmail.com
SMTP_USE_STARTTLS=true
SMTP_USE_SSL=false
```

Для локальной проверки без реального SMTP-провайдера запустите профиль Mailpit:

```bash
make docker-up-mailpit
```

Затем откройте `http://localhost:8025`.

## Примечания

- Docker-стек ожидает, что модель и baseline-артефакты уже лежат в `artifacts/`.
- Integration-тестам нужен доступный PostgreSQL.
- Сгенерированные модели, обработанные данные и отчеты мониторинга намеренно
  не хранятся в Git.
