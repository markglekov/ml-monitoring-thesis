# ML Monitoring Thesis

End-to-end ML monitoring prototype for a binary classification service. The project covers model training, online inference, delayed labels, multilevel drift and quality monitoring, alerting, incident traceability, and a small overview UI on top of the monitoring data.

## Research Contribution

- End-to-end thesis prototype for delayed-label monitoring, not only a model
  API: the repository includes training, serving, logging, monitoring,
  incidents, dashboards, and mitigation actions in one traceable system.
- Explicit blind-period monitoring protocol: proxy metrics, label coverage, and
  unlabeled quality estimates are implemented alongside classic labeled
  evaluation.
- Combined monitoring evidence model: PostgreSQL stores raw events, runs,
  metrics, incidents, and actions; Prometheus and Grafana expose both
  time-series and operator-facing summaries.
- Reproducible experiment pack for the defense: final scenarios, screenshots,
  and aggregated evidence tables can be regenerated from the repository.

## What Is Included

- `FastAPI` inference API with request logging to PostgreSQL
- Offline training pipeline for the Bank Marketing dataset
- Delayed labels ingestion and quality monitoring
- Drift monitoring with univariate and multivariate detectors
- Blind-period quality monitoring with proxy signals
- Prometheus, Grafana, and Alertmanager integration
- SMTP email relay for alert delivery
- Monitoring incidents with recommended actions
- Overview API and lightweight UI for the latest monitoring state
- Unit and integration tests

## Implemented Scenarios

- Baseline-like drift and quality runs
- Mild and severe distribution drift
- Mild and severe labeled quality degradation
- Blind-period proxy monitoring before delayed labels arrive
- Assumption-based unlabeled quality estimation under label shift
- Segment-localized monitoring and incidents
- Automated reaction flow with action execution and rollback
- Final evidence export in `artifacts/reports/final/`

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

## How This Maps To Thesis Chapters

- Chapter 1, problem statement and methods:
  drift detectors, delayed labels, blind period, proxy monitoring,
  unlabeled quality estimation, and reaction logic.
- Chapter 2, system design:
  API, storage model, scheduler, monitoring jobs, incidents, dashboards,
  and the architecture in [`docs/architecture.md`](docs/architecture.md).
- Chapter 3, experiments and evaluation:
  reproducible scenarios in [`docs/experiments.md`](docs/experiments.md),
  one-page defense flow in [`docs/defense-checklist.md`](docs/defense-checklist.md),
  and saved final artifacts in `artifacts/reports/final/`.

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
- Monitoring incidents: `GET /monitoring/incidents`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Alertmanager: `http://localhost:9093`

Grafana dashboards:

- Executive overview: `http://localhost:3000/d/ml-monitoring-overview`
- Drift analysis: `http://localhost:3000/d/ml-monitoring-drift`
- Quality and calibration: `http://localhost:3000/d/ml-monitoring-quality`
- Blind-period proxy signals: `http://localhost:3000/d/ml-monitoring-proxy`
- Segments and incidents: `http://localhost:3000/d/ml-monitoring-segments`

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
- Executive dashboard: `http://localhost:3000/d/ml-monitoring-overview`
- Drift dashboard: `http://localhost:3000/d/ml-monitoring-drift`
- Quality dashboard: `http://localhost:3000/d/ml-monitoring-quality`
- Proxy dashboard: `http://localhost:3000/d/ml-monitoring-proxy`
- Segments dashboard: `http://localhost:3000/d/ml-monitoring-segments`
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
Set `MONITORING_SEGMENTS=segment_a,segment_b` to let the scheduler fan out
the same jobs across multiple monitored segments, optionally together with the
global run controlled by `SCHEDULER_INCLUDE_GLOBAL_SEGMENT`.

Grafana is provisioned with both Prometheus and PostgreSQL datasources. The
overview and time-series panels read from Prometheus, while incident timelines,
segment tables, and latest run details read from PostgreSQL. This split keeps
alerting simple and also makes the demo dashboards more useful for defense.

For reproducible experiment scenarios and expected outcomes, see
[`docs/experiments.md`](docs/experiments.md).

For the end-to-end system diagram and architectural dataflow, see
[`docs/architecture.md`](docs/architecture.md).

For the shortest defense script with exact commands, dashboards, SQL queries,
incident choice, and rollback steps, see
[`docs/defense-checklist.md`](docs/defense-checklist.md).

## Limitations

- The prototype is centered on one binary-classification use case and one
  dataset; it is not a benchmark across many domains.
- Monitoring is implemented as window-based batch jobs, not as a fully
  distributed streaming platform.
- Automatic reactions are intentionally conservative:
  threshold tightening and manual review are implemented, while full automated
  retraining is out of scope.
- Some final defense artifacts are generated locally under `artifacts/reports/`
  and are ignored by git by design, so they should be regenerated before the
  final presentation or submission snapshot.

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
