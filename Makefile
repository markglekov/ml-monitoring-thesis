.DEFAULT_GOAL := help

UV := UV_CACHE_DIR=.uv-cache uv
PRE_COMMIT_HOME := .pre-commit-cache

.PHONY: help lock sync sync-prod format format-check lint lint-fix typecheck test test-unit test-integration test-all check hooks-install hooks-run ci run-api db-shell docker-up docker-up-mailpit docker-down final-summary

help:
	@printf "%s\n" \
		"Available targets:" \
		"  lock              Generate uv.lock" \
		"  sync              Install all dependencies, including dev" \
		"  sync-prod         Install only runtime dependencies" \
		"  format            Format code with Ruff" \
		"  format-check      Check formatting with Ruff" \
		"  lint              Run Ruff lint checks" \
		"  lint-fix          Run Ruff with auto-fixes" \
		"  typecheck         Run Pyright" \
		"  test              Run unit tests" \
		"  test-unit         Alias for test" \
		"  test-integration  Run integration tests" \
		"  test-all          Run the full test suite" \
		"  check             Run format-check, lint, typecheck, and full tests" \
		"  hooks-install     Install pre-commit and pre-push hooks" \
		"  hooks-run         Run all pre-commit hooks against the whole repo" \
		"  ci                Local CI-equivalent target" \
		"  run-api           Run the FastAPI app locally with uvicorn" \
		"  db-shell          Open psql inside the PostgreSQL container" \
		"  docker-up         Start the full Docker stack" \
		"  docker-up-mailpit Start the stack with Mailpit for local email testing" \
		"  docker-down       Stop the Docker stack" \
		"  final-summary     Build one aggregated scenario evidence table"

lock:
	mkdir -p .uv-cache
	$(UV) lock

sync:
	mkdir -p .uv-cache
	$(UV) sync --all-groups

sync-prod:
	mkdir -p .uv-cache
	$(UV) sync --no-dev

format:
	$(UV) run ruff format .

format-check:
	$(UV) run ruff format --check .

lint:
	$(UV) run ruff check .

lint-fix:
	$(UV) run ruff check --fix .

typecheck:
	$(UV) run pyright

test:
	$(UV) run pytest -q -m "not integration"

test-unit: test

test-integration:
	$(UV) run pytest -q -m integration

test-all:
	$(UV) run pytest -q

check: format-check lint typecheck test-all

hooks-install:
	PRE_COMMIT_HOME=$(PRE_COMMIT_HOME) $(UV) run pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push

hooks-run:
	PRE_COMMIT_HOME=$(PRE_COMMIT_HOME) $(UV) run pre-commit run --all-files

ci: check

run-api:
	$(UV) run uvicorn app.api.main:app --host 0.0.0.0 --port $${API_PORT:-8000} --reload

db-shell:
	docker compose exec postgres psql -U "$${POSTGRES_USER:-mlops}" -d "$${POSTGRES_DB:-ml_monitoring}"

docker-up:
	docker compose up -d --build

docker-up-mailpit:
	docker compose --profile mailpit up -d --build

docker-down:
	docker compose down

final-summary:
	$(UV) run python -m app.reporting.build_scenario_summary
