#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
API_PORT="${API_PORT:-8000}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
API_URL="${API_URL:-http://localhost:${API_PORT}}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:${GRAFANA_PORT}}"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"
DEMO_TRAIN="${DEMO_TRAIN:-always}"
DEMO_DOCKER_UP="${DEMO_DOCKER_UP:-1}"
DEMO_RESET="${DEMO_RESET:-0}"
DEMO_SEED="${DEMO_SEED:-42}"
DEMO_WAIT_TIMEOUT_SEC="${DEMO_WAIT_TIMEOUT_SEC:-180}"
DEMO_EXPORT_TIMEOUT_SEC="${DEMO_EXPORT_TIMEOUT_SEC:-30}"
DEMO_WAIT_FOR_SCRAPE_SEC="${DEMO_WAIT_FOR_SCRAPE_SEC:-20}"

MODEL_PATH="${MODEL_PATH:-artifacts/models/bank_marketing_model.joblib}"
BASELINE_PATH="${BASELINE_PATH:-artifacts/baselines/baseline_profile.json}"
FINAL_DIR="artifacts/reports/final"

run_uv_python() {
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run python "$@"
}

step() {
  printf "\n==> %s\n" "$1"
}

wait_for_api() {
  local deadline
  deadline=$((SECONDS + DEMO_WAIT_TIMEOUT_SEC))

  while ((SECONDS < deadline)); do
    if curl -fsS "${API_URL}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done

  printf "API healthcheck did not become ready: %s/health\n" "$API_URL" >&2
  return 1
}

if [[ "$DEMO_RESET" == "1" ]]; then
  step "Reset Docker Compose runtime state"
  docker compose down -v
fi

case "$DEMO_TRAIN" in
  always)
    step "Train model and rebuild baseline artifacts"
    run_uv_python -m app.train.train
    ;;
  missing)
    if [[ ! -f "$MODEL_PATH" || ! -f "$BASELINE_PATH" ]]; then
      step "Train model because model or baseline artifact is missing"
      run_uv_python -m app.train.train
    else
      step "Reuse existing model and baseline artifacts"
    fi
    ;;
  skip)
    step "Skip training by DEMO_TRAIN=skip"
    ;;
  *)
    printf "Unsupported DEMO_TRAIN=%s. Use always, missing, or skip.\n" \
      "$DEMO_TRAIN" >&2
    exit 2
    ;;
esac

if [[ "$DEMO_DOCKER_UP" == "1" ]]; then
  step "Build and start Docker Compose stack"
  docker compose up -d --build
else
  step "Reuse already running stack by DEMO_DOCKER_UP=0"
fi

step "Wait for API healthcheck"
wait_for_api

step "Run final monitoring scenarios and export artifacts"
run_uv_python -m app.reporting.export_final_artifacts \
  --api-url "$API_URL" \
  --grafana-url "$GRAFANA_URL" \
  --grafana-user "$GRAFANA_USER" \
  --grafana-password "$GRAFANA_PASSWORD" \
  --seed "$DEMO_SEED" \
  --timeout-sec "$DEMO_EXPORT_TIMEOUT_SEC" \
  --wait-for-scrape-sec "$DEMO_WAIT_FOR_SCRAPE_SEC"

step "Build final scenario summary"
run_uv_python -m app.reporting.build_scenario_summary

step "Save latest monitoring overview snapshot"
mkdir -p "$FINAL_DIR"
curl -fsS "${API_URL}/monitoring/overview" \
  -o "${FINAL_DIR}/overview_latest.json"

step "One-command demo completed"
printf "Final artifacts: %s\n" "$FINAL_DIR"
printf "Scenario summary: %s/scenario_summary.md\n" "$FINAL_DIR"
printf "Overview UI: %s/overview\n" "$API_URL"
printf "Grafana overview: %s/d/ml-monitoring-overview\n" "$GRAFANA_URL"
