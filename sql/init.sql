-- Initialize storage for inference events and delayed ground-truth labels.

CREATE TABLE IF NOT EXISTS inference_log (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL UNIQUE,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version TEXT NOT NULL,
    features_json JSONB NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    pred_label INTEGER NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    segment_key TEXT,
    latency_ms DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_inference_log_ts
    ON inference_log(ts);

CREATE INDEX IF NOT EXISTS idx_inference_log_model_version
    ON inference_log(model_version);

CREATE INDEX IF NOT EXISTS idx_inference_log_segment_key
    ON inference_log(segment_key);

CREATE TABLE IF NOT EXISTS ground_truth (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    y_true INTEGER NOT NULL CHECK (y_true IN (0, 1)),
    label_ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ground_truth_request_id
    ON ground_truth(request_id);

CREATE INDEX IF NOT EXISTS idx_ground_truth_label_ts
    ON ground_truth(label_ts);
