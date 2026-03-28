-- Initialize storage for delayed-label quality monitoring.

CREATE UNIQUE INDEX IF NOT EXISTS idx_ground_truth_request_id_unique
    ON ground_truth(request_id);

CREATE TABLE IF NOT EXISTS quality_runs (
    id BIGSERIAL PRIMARY KEY,
    ts_started TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ts_finished TIMESTAMPTZ,
    model_version TEXT NOT NULL,
    window_size INTEGER NOT NULL,
    segment_key TEXT,
    status TEXT NOT NULL,
    labeled_rows INTEGER DEFAULT 0,
    degraded_metrics_count INTEGER DEFAULT 0,
    summary_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_quality_runs_ts_started
    ON quality_runs(ts_started);

CREATE INDEX IF NOT EXISTS idx_quality_runs_segment_key
    ON quality_runs(segment_key);

CREATE TABLE IF NOT EXISTS quality_metrics (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES quality_runs(id) ON DELETE CASCADE,
    segment_key TEXT,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    baseline_value DOUBLE PRECISION,
    delta_value DOUBLE PRECISION,
    degradation_detected BOOLEAN NOT NULL DEFAULT FALSE,
    details_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_quality_metrics_run_id
    ON quality_metrics(run_id);

CREATE INDEX IF NOT EXISTS idx_quality_metrics_metric_name
    ON quality_metrics(metric_name);
