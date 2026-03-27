CREATE TABLE IF NOT EXISTS monitoring_runs (
    id BIGSERIAL PRIMARY KEY,
    ts_started TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ts_finished TIMESTAMPTZ,
    model_version TEXT NOT NULL,
    window_size INTEGER NOT NULL,
    segment_key TEXT,
    status TEXT NOT NULL,
    drifted_features_count INTEGER DEFAULT 0,
    total_features_count INTEGER DEFAULT 0,
    overall_drift BOOLEAN DEFAULT FALSE,
    summary_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_monitoring_runs_ts_started
    ON monitoring_runs(ts_started);

CREATE INDEX IF NOT EXISTS idx_monitoring_runs_status
    ON monitoring_runs(status);

CREATE TABLE IF NOT EXISTS drift_metrics (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES monitoring_runs(id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    ks_pvalue DOUBLE PRECISION,
    chi2_pvalue DOUBLE PRECISION,
    psi_value DOUBLE PRECISION,
    drift_detected BOOLEAN NOT NULL,
    details_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_run_id
    ON drift_metrics(run_id);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_feature_name
    ON drift_metrics(feature_name);
