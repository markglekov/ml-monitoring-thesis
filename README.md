# ML Monitoring Thesis

End-to-end ML monitoring prototype for a binary classification service. The project covers model training, online inference, delayed labels, drift and quality monitoring, alerting, and a small overview UI on top of the monitoring data.

## What Is Included

- `FastAPI` inference API with request logging to PostgreSQL
- Offline training pipeline for the Bank Marketing dataset
- Delayed labels ingestion and quality monitoring
- Drift monitoring over recent inference windows
- Prometheus, Grafana, and Alertmanager integration
- SMTP email relay for alert delivery
- Overview API and lightweight UI for the latest monitoring state
- Unit and integration tests

## Tech Stack

- Python 3.12
- `uv` for dependency management
- `FastAPI`, `Pydantic`, `SQLAlchemy`
- `scikit-learn`, `pandas`, `numpy`, `scipy`
- PostgreSQL
- Prometheus, Grafana, Alertmanager
- Ruff, Pyright, Pytest

## Project Layout

```text
app/
  api/            FastAPI app, monitoring endpoints, overview UI
  common/         Shared config, logging, metrics
  monitoring/     Drift job, quality job, scheduler, labels backfill client
  notifications/  Email relay for Alertmanager webhooks
  simulator/      Request stream generator
  train/          Training pipeline and artifact generation
monitoring/       Prometheus, Grafana, Alertmanager configuration
sql/              Database schema bootstrap SQL
tests/            Unit and integration tests
docs/             Reproducible experiment runbooks and reports
```

## Quick Start

1. Copy `.env.example` to `.env` and fill in the values you need.
2. Install dependencies:

```bash
make sync
```

3. Train the model and generate baseline artifacts if you have not done it yet:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.train.train
```

4. Start the stack:

```bash
make docker-up
```

To run only the API locally outside Docker:

```bash
make run-api
```

If port `5432` is already busy on your machine, override the host port:

```bash
POSTGRES_PORT=55432 docker compose up -d --build
```

## Useful Endpoints

- API docs: `http://localhost:8000/docs`
- Health: `GET /health`
- Metrics: `GET /metrics`
- Overview JSON: `GET /monitoring/overview`
- Overview UI: `GET /overview`
- Latest drift runs: `GET /monitoring/drift/runs`
- Latest quality runs: `GET /monitoring/quality/runs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Alertmanager: `http://localhost:9093`

## Development Workflow

Install or update the local environment:

```bash
make sync
```

Run the full local quality gate:

```bash
make check
```

Install git hooks:

```bash
make hooks-install
```

Run only selected tools:

```bash
make format
make lint
make typecheck
make test
make test-integration
```

Run the same quality gate locally that CI executes:

```bash
make ci
```

Inspect PostgreSQL directly:

```bash
make db-shell
```

Useful SQL inside `psql`:

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

## Quick Demo Scenario

1. Start the full stack:

```bash
make docker-up
```

2. Generate a baseline-like inference stream:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.simulator.generate_stream \
  --source-split train \
  --rows 300 \
  --scenario none \
  --segment-key demo_none \
  --seed 42
```

3. Run drift monitoring on the same segment:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.drift_job \
  --window-size 300 \
  --segment-key demo_none
```

4. Backfill delayed labels and run quality monitoring:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.backfill_labels \
  --label-policy perfect

UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.quality_job \
  --window-size 300 \
  --segment-key demo_none
```

5. Inspect the result:

- Overview UI: `http://localhost:8000/overview`
- Overview JSON: `http://localhost:8000/monitoring/overview`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- Alertmanager: `http://localhost:9093`

## Monitoring Workflow

Generate a stream of inference requests:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.simulator.generate_stream --rows 300 --scenario none
```

Backfill delayed labels through the API:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.backfill_labels --label-policy perfect
```

Run monitoring jobs manually if needed:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.drift_job --window-size 300
UV_CACHE_DIR=.uv-cache uv run python -m app.monitoring.quality_job --window-size 300
```

The scheduler service can run both jobs continuously through Docker Compose.

For reproducible experiment scenarios and expected outcomes, see
[`docs/experiments.md`](docs/experiments.md).

## Email Alerting

The project ships with an SMTP relay service that receives Alertmanager webhooks and sends emails. For Gmail, use an app password and set these variables in `.env`:

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

For local testing without a real SMTP provider, start the optional Mailpit profile:

```bash
make docker-up-mailpit
```

Then open `http://localhost:8025`.

## Notes

- The Docker stack expects model and baseline artifacts to exist under `artifacts/`.
- Integration tests require an accessible PostgreSQL instance.
- The repository keeps generated models, processed data, and monitoring reports out of Git on purpose.
