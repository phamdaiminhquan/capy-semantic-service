-- Tables to support Data Flywheel (gold dataset) and full prediction logging.
-- Apply via your existing NestJS migration system.

CREATE TABLE IF NOT EXISTS prediction_logs (
  id BIGSERIAL PRIMARY KEY,
  raw_text TEXT NOT NULL,
  normalized_text TEXT NOT NULL,
  cache_key CHAR(64) NOT NULL,
  token_count INTEGER NOT NULL,
  amount DOUBLE PRECISION NULL,
  label TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  routing_layer TEXT NOT NULL,
  model_version TEXT NOT NULL,
  source TEXT NOT NULL,
  rationale TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_cache_key ON prediction_logs(cache_key);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_created_at ON prediction_logs(created_at);

CREATE TABLE IF NOT EXISTS gold_dataset (
  id BIGSERIAL PRIMARY KEY,
  raw_text TEXT NOT NULL,
  normalized_text TEXT NOT NULL,
  cache_key CHAR(64) NOT NULL,
  token_count INTEGER NOT NULL,
  amount DOUBLE PRECISION NULL,
  label TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  routing_layer TEXT NOT NULL,
  model_version TEXT NOT NULL,
  source TEXT NOT NULL,
  rationale TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_gold_dataset_cache_key ON gold_dataset(cache_key);
CREATE INDEX IF NOT EXISTS idx_gold_dataset_created_at ON gold_dataset(created_at);

-- Optional: basic routing_layer constraint
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'chk_prediction_logs_routing_layer'
  ) THEN
    ALTER TABLE prediction_logs
      ADD CONSTRAINT chk_prediction_logs_routing_layer
      CHECK (routing_layer IN ('keyword','cache','local_onnx','gemini'));
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'chk_gold_dataset_routing_layer'
  ) THEN
    ALTER TABLE gold_dataset
      ADD CONSTRAINT chk_gold_dataset_routing_layer
      CHECK (routing_layer IN ('keyword','cache','local_onnx','gemini'));
  END IF;
END $$;
