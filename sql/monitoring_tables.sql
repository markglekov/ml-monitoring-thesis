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

CREATE INDEX IF NOT EXISTS idx_monitoring_runs_segment_key
    ON monitoring_runs(segment_key);

CREATE TABLE IF NOT EXISTS drift_metrics (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES monitoring_runs(id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    ks_pvalue DOUBLE PRECISION,
    chi2_pvalue DOUBLE PRECISION,
    psi_value DOUBLE PRECISION,
    detector_name TEXT NOT NULL DEFAULT 'univariate',
    effect_size DOUBLE PRECISION,
    pvalue_adj DOUBLE PRECISION,
    severity TEXT NOT NULL DEFAULT 'none',
    recommended_action TEXT,
    drift_detected BOOLEAN NOT NULL,
    details_json JSONB
);

ALTER TABLE drift_metrics
    ADD COLUMN IF NOT EXISTS detector_name TEXT NOT NULL DEFAULT 'univariate';

ALTER TABLE drift_metrics
    ADD COLUMN IF NOT EXISTS effect_size DOUBLE PRECISION;

ALTER TABLE drift_metrics
    ADD COLUMN IF NOT EXISTS pvalue_adj DOUBLE PRECISION;

ALTER TABLE drift_metrics
    ADD COLUMN IF NOT EXISTS severity TEXT NOT NULL DEFAULT 'none';

ALTER TABLE drift_metrics
    ADD COLUMN IF NOT EXISTS recommended_action TEXT;

CREATE INDEX IF NOT EXISTS idx_drift_metrics_run_id
    ON drift_metrics(run_id);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_feature_name
    ON drift_metrics(feature_name);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_severity
    ON drift_metrics(severity);

CREATE TABLE IF NOT EXISTS monitoring_incidents (
    id BIGSERIAL PRIMARY KEY,
    incident_key TEXT NOT NULL,
    source_type TEXT NOT NULL,
    model_version TEXT NOT NULL,
    segment_key TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    summary_json JSONB,
    latest_run_id BIGINT,
    acknowledged_by TEXT,
    mitigation_taken TEXT,
    ts_opened TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ts_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ts_resolved TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_monitoring_incidents_incident_key
    ON monitoring_incidents(incident_key);

CREATE INDEX IF NOT EXISTS idx_monitoring_incidents_status
    ON monitoring_incidents(status);

CREATE INDEX IF NOT EXISTS idx_monitoring_incidents_severity
    ON monitoring_incidents(severity);

CREATE INDEX IF NOT EXISTS idx_monitoring_incidents_source_type
    ON monitoring_incidents(source_type);

CREATE INDEX IF NOT EXISTS idx_monitoring_incidents_segment_key
    ON monitoring_incidents(segment_key);
