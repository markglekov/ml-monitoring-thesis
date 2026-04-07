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
    detector_name TEXT NOT NULL DEFAULT 'labeled',
    effect_size DOUBLE PRECISION,
    pvalue_adj DOUBLE PRECISION,
    severity TEXT NOT NULL DEFAULT 'none',
    recommended_action TEXT,
    degradation_detected BOOLEAN NOT NULL DEFAULT FALSE,
    details_json JSONB
);

ALTER TABLE quality_metrics
    ADD COLUMN IF NOT EXISTS detector_name TEXT NOT NULL DEFAULT 'labeled';

ALTER TABLE quality_metrics
    ADD COLUMN IF NOT EXISTS effect_size DOUBLE PRECISION;

ALTER TABLE quality_metrics
    ADD COLUMN IF NOT EXISTS pvalue_adj DOUBLE PRECISION;

ALTER TABLE quality_metrics
    ADD COLUMN IF NOT EXISTS severity TEXT NOT NULL DEFAULT 'none';

ALTER TABLE quality_metrics
    ADD COLUMN IF NOT EXISTS recommended_action TEXT;

CREATE INDEX IF NOT EXISTS idx_quality_metrics_run_id
    ON quality_metrics(run_id);

CREATE INDEX IF NOT EXISTS idx_quality_metrics_metric_name
    ON quality_metrics(metric_name);

CREATE INDEX IF NOT EXISTS idx_quality_metrics_severity
    ON quality_metrics(severity);

CREATE TABLE IF NOT EXISTS quality_estimates (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES quality_runs(id) ON DELETE CASCADE,
    segment_key TEXT,
    estimated_positive_rate DOUBLE PRECISION,
    estimated_metric_name TEXT NOT NULL,
    estimated_metric_value DOUBLE PRECISION,
    assumption_type TEXT NOT NULL,
    quality_estimate_uncertainty DOUBLE PRECISION,
    confidence_interval_json JSONB,
    details_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_quality_estimates_run_id
    ON quality_estimates(run_id);

CREATE INDEX IF NOT EXISTS idx_quality_estimates_metric_name
    ON quality_estimates(estimated_metric_name);

CREATE INDEX IF NOT EXISTS idx_quality_estimates_assumption_type
    ON quality_estimates(assumption_type);
